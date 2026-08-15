"""Multi-order attractor fit: VAR(1) dynamics + fixed point + uncertainty.

The user's kinematic model of AR collapse (2026-08-14): each committed chunk's
stat vector moves toward the attractor along a CURVED, DECELERATING path —
step size shrinks chunk over chunk and the direction of travel rotates
(sometimes >90 deg). A per-channel scalar AR(1) cannot represent rotation;
the right model is the coupled linear system

    y_{j+1} = A y_j + b        (y = per-channel chunk means, R^16)

whose complex eigenvalues ARE the rotation, whose eigenvalue magnitudes ARE
the per-step deceleration, and whose fixed point p = (I - A)^{-1} b is where
every extrapolated trajectory crosses — the attractor. Fit once over ALL
directions (the dynamics are the same network; direction identity lives in
the STATE, so pooling is legitimate for the full-matrix model — unlike the
scalar case, where it is Simpson-biased).

Regularisation ("perturb the curvature minimally, maximise the crossing"):
ridge-shrink A toward the isotropic decay alpha_bar*I, lambda chosen by
leave-one-trajectory-out extrapolation error — which doubles as the error
CALIBRATION: bootstrap over trajectories gives the attractor's mean AND
covariance.

Also marks the PROJECTED NOISE POINT: pure noise is zero-mean per channel, a
FIXED point of stat space; its plane coordinates differ per figure only
because each figure's PCA basis has its own origin (this is why the timeline
videos show the noise X at ~(200,30) while other plots suggest ~(100,0)).

Env: AV_RUN (pilot_gt0), AV_K (5 PCA dims), AV_BOOT (200), AV_OUT (png).
"""
import importlib.util
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
FV = f"{ARR}/analysis/eval_final/flow_viz"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


T = _load("t_at", f"{ARR}/utils/test_attractor_tracker.py")
DIRS, DCOL, N_CH = T.DIRS, T.DCOL, T.N_CH
RUN = os.environ.get("AV_RUN", "pilot_gt0")
K = int(os.environ.get("AV_K", "5"))
BOOT = int(os.environ.get("AV_BOOT", "200"))
OUT = os.environ.get("AV_OUT", f"{FV}/attractor_var_{RUN}.png")
EXTRAP = 40          # chunks to extrapolate each trajectory for the plot


def mu_traj(chunks):
    """[n_chunks, 3, C, H, W] -> [n_chunks, C] per-channel means."""
    return chunks.mean(axis=(1, 3, 4))


def fit_var_anchored(trajs, lam, kap, k_basis, iters=12):
    """VAR(1) with the fixed point ANCHORED at the noise point (user prior:
    'the initial guess has to be where the gaussian noise distribution
    lives'). Model: y' - p = A (y - p), ridge A -> a_bar*I (strength lam),
    ridge p -> p0 = 0-mu (strength kap). Alternating least squares. The data
    pulls p away from the noise point only along modes it identifies; the
    slow, unidentified modes stay anchored — which is what makes the
    fixed point WELL-POSED on 6-chunk rollouts (unanchored bootstrap std was
    ~2.9/channel, unusable)."""
    Yall = np.concatenate(trajs, axis=0)
    mean = Yall.mean(0)
    U, S, Vt = np.linalg.svd(Yall - mean, full_matrices=False)
    comps = Vt[:k_basis]
    cs = [(t - mean) @ comps.T for t in trajs]
    X = np.concatenate([c[:-1] for c in cs], axis=0)
    Y = np.concatenate([c[1:] for c in cs], axis=0)
    k = k_basis
    p0 = (np.zeros(N_CH) - mean) @ comps.T                 # noise point, subspace
    p = p0.copy()
    A = None
    for _ in range(iters):
        Xc, Yc = X - p, Y - p
        a_bar = float((Xc * Yc).sum() / max((Xc * Xc).sum(), 1e-9))
        G = Xc.T @ Xc + lam * np.eye(k)
        Rhs = Xc.T @ Yc + lam * a_bar * np.eye(k)
        A = np.linalg.solve(G, Rhs).T
        M = np.eye(k) - A
        r = (Y - X @ A.T)                                   # each row ~ M p
        n = r.shape[0]
        p = np.linalg.solve(n * M.T @ M + kap * np.eye(k),
                            M.T @ r.sum(0) + kap * p0)
    eig = np.linalg.eigvals(A)
    stable = np.all(np.abs(eig) < 1.0)
    p_full = mean + p @ comps
    resid = (Y - p) - (X - p) @ A.T
    return dict(A=A, b=(np.eye(k) - A) @ p, mean=mean, comps=comps, eig=eig,
                stable=stable, p_sub=p, p=p_full, a_bar=a_bar,
                p0_full=mean + p0 @ comps,
                resid_rms=float(np.sqrt((resid ** 2).mean())))


def fit_var(trajs, lam, k_basis):
    """Ridge VAR(1) in the k-dim PCA subspace of the pooled stat cloud.

    trajs: list of [n_chunks, C] arrays. Returns dict with A, b (subspace),
    basis (mean, comps), fixed point in R^C, eigenvalues.
    """
    Yall = np.concatenate(trajs, axis=0)
    mean = Yall.mean(0)
    U, S, Vt = np.linalg.svd(Yall - mean, full_matrices=False)
    comps = Vt[:k_basis]                                   # [k, C]
    cs = [(t - mean) @ comps.T for t in trajs]             # [n, k] each
    X = np.concatenate([c[:-1] for c in cs], axis=0)       # [m, k]
    Y = np.concatenate([c[1:] for c in cs], axis=0)        # [m, k]
    xm, ym = X.mean(0), Y.mean(0)
    Xc, Yc = X - xm, Y - ym
    # isotropic decay target: the "minimal curvature" prior
    a_bar = float((Xc * Yc).sum() / max((Xc * Xc).sum(), 1e-9))
    k = k_basis
    # ridge toward a_bar*I: solve (Xc^T Xc + lam I) A^T = Xc^T Yc + lam a_bar I
    G = Xc.T @ Xc + lam * np.eye(k)
    Rhs = Xc.T @ Yc + lam * a_bar * np.eye(k)
    A = np.linalg.solve(G, Rhs).T                          # [k, k]
    b = ym - A @ xm
    eig = np.linalg.eigvals(A)
    stable = np.all(np.abs(eig) < 1.0)
    p_sub = np.linalg.solve(np.eye(k) - A, b) if stable else None
    p_full = (mean + p_sub @ comps) if p_sub is not None else None
    resid = Y - (X @ A.T + b)
    return dict(A=A, b=b, mean=mean, comps=comps, eig=eig, stable=stable,
                p_sub=p_sub, p=p_full, a_bar=a_bar,
                resid_rms=float(np.sqrt((resid ** 2).mean())))


def extrapolate(fit, y0, steps):
    """Iterate the fitted map from stat vector y0 [C] for `steps` chunks."""
    c = (y0 - fit["mean"]) @ fit["comps"].T
    out = []
    for _ in range(steps):
        c = fit["A"] @ c + fit["b"]
        out.append(fit["mean"] + c @ fit["comps"])
    return np.stack(out)


def loo_error(trajs, lam, kap, k_basis):
    """Leave-one-trajectory-out: fit on the rest, extrapolate the held-out
    trajectory from its chunk-1 state, error at its LAST observed chunk."""
    errs = []
    for i in range(len(trajs)):
        rest = [t for j, t in enumerate(trajs) if j != i]
        f = fit_var_anchored(rest, lam, kap, k_basis)
        if not f["stable"]:
            errs.append(np.inf); continue
        pred = extrapolate(f, trajs[i][1], len(trajs[i]) - 2)[-1]
        errs.append(float(np.linalg.norm(pred - trajs[i][-1])))
    return float(np.mean(errs))


def main():
    student = T.load_student(RUN)
    trajs, labels = [], []
    for d in DIRS:
        for chunks in student.get(d, []):
            trajs.append(mu_traj(chunks))
            labels.append(d)
    print(f"run={RUN}: {len(trajs)} trajectories, k={K} PCA dims")

    # (lambda, kappa) sweep by LOO extrapolation error (the calibration)
    best, blam, bkap = np.inf, 1.0, 1.0
    print("LOO extrap err (mu-space) over (lambda, kappa):")
    for l in (1.0, 3.0, 10.0):
        row = []
        for kp in (0.3, 1.0, 3.0, 10.0, 30.0):
            e = loo_error(trajs, l, kp, K)
            row.append(f"k{kp}:{e:.3f}")
            if e < best:
                best, blam, bkap = e, l, kp
        print(f"  lam={l}: " + "  ".join(row))
    lam, kap = blam, bkap
    print(f"-> lambda={lam}, kappa={kap}, LOO err={best:.3f}")

    fit = fit_var_anchored(trajs, lam, kap, K)
    eig = fit["eig"]
    rot = np.abs(np.angle(eig)) * 180 / np.pi
    print(f"eigenvalues |.|: {np.round(np.abs(eig), 3)}")
    print(f"rotation per chunk-step (deg): {np.round(rot, 1)} "
          f"(complex pairs = the curving the scalar model cannot see)")
    print(f"a_bar (isotropic prior) = {fit['a_bar']:.3f}, "
          f"resid RMS = {fit['resid_rms']:.4f}, stable={fit['stable']}")
    if fit["p"] is None:
        raise SystemExit("unstable fit — no attractor at this lambda")
    print(f"ATTRACTOR mu: Sum mu^2 = {float((fit['p'] ** 2).sum()):.3f}  "
          f"(noise point = 0, data ~ 2.9)")
    print(f"attractor-to-noise distance |p - 0| = "
          f"{float(np.linalg.norm(fit['p'])):.3f}   "
          f"(small = 'attractor is where the noise lives')")

    # bootstrap over trajectories -> attractor mean + covariance
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(BOOT):
        idx = rng.integers(0, len(trajs), len(trajs))
        f = fit_var_anchored([trajs[i] for i in idx], lam, kap, K)
        if f["stable"]:
            boots.append(f["p"])
    boots = np.stack(boots)
    p_mean, p_std = boots.mean(0), boots.std(0)
    print(f"bootstrap ({len(boots)}/{BOOT} stable): attractor mu mean "
          f"Sum^2={float((p_mean ** 2).sum()):.3f}, mean per-ch std "
          f"{float(p_std.mean()):.3f}")

    # ---- plot on the flow-tree plane ------------------------------------
    m0, pcs = T.fit_basis()
    fig, ax = plt.subplots(figsize=(9.5, 9.5))
    for d, tr in zip(labels, trajs):
        pts = np.stack([T.mu_point(y, m0, pcs) for y in tr])
        ax.plot(pts[:, 0], pts[:, 1], "-o", color=DCOL[d], ms=3, lw=1.2,
                alpha=0.7)
        ex = extrapolate(fit, tr[-1], EXTRAP)
        exp = np.stack([T.mu_point(y, m0, pcs) for y in ex])
        ax.plot(np.r_[pts[-1, 0], exp[:, 0]], np.r_[pts[-1, 1], exp[:, 1]],
                "--", color=DCOL[d], lw=1.0, alpha=0.55)
    bpts = np.stack([T.mu_point(p, m0, pcs) for p in boots])
    bm = bpts.mean(0); bc = np.cov(bpts.T)
    ev, evec = np.linalg.eigh(bc)
    ang = float(np.degrees(np.arctan2(evec[1, -1], evec[0, -1])))
    for nsig, alp in ((1, 0.35), (2, 0.15)):
        ax.add_patch(Ellipse(bm, 2 * nsig * np.sqrt(max(ev[-1], 1e-9)),
                             2 * nsig * np.sqrt(max(ev[0], 1e-9)),
                             angle=ang, color="crimson", alpha=alp, zorder=7))
    ax.plot(*bm, marker="*", ms=26, color="crimson", markeredgecolor="k",
            zorder=8, label=f"VAR attractor ±1σ/2σ (boot n={len(boots)})")
    npt = T.mu_point(np.zeros(N_CH), m0, pcs)
    ax.plot(*npt, marker="X", ms=18, color="k", zorder=8,
            label="projected NOISE point (mu=0)")
    # zoom to the data/attractor region, not the (possibly wide) 2-sigma tail
    allp = np.concatenate([np.stack([T.mu_point(y, m0, pcs) for y in tr])
                           for tr in trajs] + [bpts, npt[None]])
    lo, hi = allp.min(0) - 30, allp.max(0) + 30
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_title(f"{RUN}: VAR(1) multi-order attractor fit — trajectories "
                 f"(solid), extrapolations (dashed),\nattractor + uncertainty "
                 f"(crimson), noise point (black X) — flow-tree PCA plane")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT, dpi=140)
    print(f"[av] wrote {OUT}")


if __name__ == "__main__":
    main()
