"""Offline verification of the online AttractorTracker on recorded rollouts.

Feeds the REAL recordings (the same steps.npz used by the flow trees:
flow_<run>/r08_<dir>_s<seed>) through the actual AttractorTracker — the same
observe/sync/_refreeze code the trainer will run — and renders the flow tree
with the predicted attractor locations overlaid.

What this verifies:
  1. The estimator converges on real (not synthetic) drift data, and the
     gates OPEN on data we know is collapsing.
  2. The user's convergence hypothesis: individual trajectories snake, but
     across directions they head to a common region — so the 8 per-direction
     attractor estimates should CLUSTER, and a pooled fit should land inside
     that cluster.
  3. The measured alpha / R^2 / z values on real data, so gate thresholds are
     set from evidence instead of caution.

Env: AT_RUN (default pilot_gt0), AT_PASSES (default 40), AT_OUT (png path).
CPU-only; imports the tracker directly to dodge the CUDA import chain.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
FV = f"{ARR}/analysis/eval_final/flow_viz"
DIRS = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]
DCOL = {"F": "#d62728", "FR": "#ff7f0e", "R": "#bcbd22", "BR": "#2ca02c",
        "B": "#17becf", "BL": "#1f77b4", "L": "#9467bd", "FL": "#e377c2"}
NFB, C, H, W = 3, 16, 60, 104
RUN = os.environ.get("AT_RUN", "pilot_gt0")
PASSES = int(os.environ.get("AT_PASSES", "40"))
OUT = os.environ.get("AT_OUT", f"{FV}/attractor_tracker_test_{RUN}.png")

spec = importlib.util.spec_from_file_location(
    "attractor_tracker", f"{ARR}/action-forcing/af_model/attractor_tracker.py")
_m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_m)
AttractorTracker = _m.AttractorTracker
N_CH = _m.N_CH


def load_student(run):
    """dir -> list over seeds of [n_chunks, 3, C, H, W] committed chunks."""
    out = {}
    for d in DIRS:
        per = []
        for s in (0, 1):
            p = f"{FV}/flow_{run}/r08_{d}_s{s}/steps.npz"
            if not os.path.exists(p):
                continue
            z = np.load(p)
            ch = {}
            for i, (c, r, t) in enumerate(z["sdt"]):
                if t > 0 and int(r) == 3:
                    ch[int(c)] = z[f"x{i}"].astype(np.float32).reshape(NFB, C, H, W)
            if ch:
                per.append(np.stack([ch[c] for c in sorted(ch)]))
        if per:
            out[d] = per
    return out


def load_teacher():
    """dir -> [n_seeds, n_blocks, 3, C, H, W] committed teacher chunks."""
    z = np.load(f"{FV}/trajs_14e8s20_w8.npz")
    out = {}
    for d in DIRS:
        seeds = []
        for s in range(4):
            blocks = []
            for b in range(6):
                k = f"{d}_{s}" if b == 0 else f"b{b}_{d}_{s}"
                if k in z.files:
                    blocks.append(z[k][-1].astype(np.float32).reshape(NFB, C, H, W))
            if len(blocks) == 6:
                seeds.append(np.stack(blocks))
        if seeds:
            out[d] = np.stack(seeds)
    return out


def fit_basis():
    """Same PCA plane the flow trees use (flow_tree_ellipse.fit_basis)."""
    z = np.load(f"{FV}/trajs_14e8s20_w8.npz")
    a = []
    for d in DIRS:
        for s in range(4):
            for b in range(6):
                k = f"{d}_{s}" if b == 0 else f"b{b}_{d}_{s}"
                if k in z.files:
                    v = z[k][-1].astype(np.float32)
                    a.extend(v.reshape(NFB, -1))
    a = np.stack(a); m0 = a.mean(0); R = a - m0
    w, v = np.linalg.eigh(R @ R.T)
    return m0, (R.T @ v[:, -2:]) / np.sqrt(np.maximum(w[-2:], 1e-9))


def proj_latent(lat, m0, pcs):
    """[..., C, H, W] frames -> mean 2-D point on the flow-tree plane."""
    fl = lat.reshape(-1, C * H * W * 1) if lat.ndim == 3 else lat.reshape(len(lat), -1)
    return ((fl - m0) @ pcs).mean(0)


def mu_point(mu, m0, pcs):
    """Stat-space attractor (per-channel means) -> the plane, as the constant
    per-channel latent with those means (the DC component the drift lives in)."""
    lat = np.repeat(np.asarray(mu, dtype=np.float32)[:, None], H * W, axis=1
                    ).reshape(C, H, W)
    return proj_latent(lat[None], m0, pcs)


def main():
    student = load_student(RUN)
    teacher = load_teacher()
    if not student:
        raise SystemExit(f"no recordings under {FV}/flow_{RUN}")
    # fast-cadence tracker so PASSES passes reach steady state; thresholds are
    # the PRODUCTION ones — that is the point of the test.
    tr = AttractorTracker(warmup=0, freeze_k=5, ema_beta=0.99, min_eff_n=8.0)
    step = 0
    for _ in range(PASSES):
        for di, d in enumerate(DIRS):
            if d not in student or d not in teacher:
                continue
            for si, chunks in enumerate(student[d]):
                chain = torch.from_numpy(
                    chunks.reshape(1, -1, C, H, W))            # [1, n*3, C, H, W]
                tgt = torch.from_numpy(
                    teacher[d][si % len(teacher[d])][None])    # [1, 6, 3, C, H, W]
                n = min(chain.shape[1] // NFB, tgt.shape[1])
                tr.observe(chain[:, :n * NFB], tgt[:, :n], di)
        tr.sync(step); step += 1

    # ---- numeric report --------------------------------------------------
    t_mean, t_var = tr._teacher_moments()
    s = t_var.sqrt()
    n = tr.ema_sn.clamp(min=1e-6)
    var_x = (tr.ema_sxx / n - (tr.ema_sx / n).pow(2)).clamp(min=1e-8)
    cov = tr.ema_sxy / n - (tr.ema_sx / n) * (tr.ema_sy / n)
    var_y = (tr.ema_syy / n - (tr.ema_sy / n).pow(2)).clamp(min=1e-8)
    alpha = cov / var_x
    r2 = (cov.pow(2) / (var_x * var_y)).clamp(0, 1)
    print(f"run={RUN}  passes={PASSES}  (production gate thresholds)")
    print(f"{'dir':>4} {'gate':>5} {'alpha(mu)':>10} {'R2(mu)':>7} "
          f"{'z(mu)':>6}  attractor mu (mean over gated ch)")
    A = np.zeros((len(DIRS), N_CH)); G = np.zeros((len(DIRS), N_CH))
    for di, d in enumerate(DIRS):
        g = tr.frozen_gate[di, :N_CH]
        a = tr.frozen_a[di, :N_CH]
        z = ((tr.frozen_a[di] - t_mean[di]).abs() / s[di])[:N_CH]
        A[di], G[di] = a.numpy(), g.numpy()
        am = float((a * g).sum() / g.sum()) if g.sum() > 0 else float("nan")
        print(f"{d:>4} {float(g.sum()):5.1f} {float(alpha[di, :N_CH].mean()):10.3f} "
              f"{float(r2[di, :N_CH].mean()):7.3f} {float(z.mean()):6.2f}  {am:+.3f}")
    # THE GLOBAL ATTRACTOR (stratify-then-combine; the actuation reference)
    gg = tr.frozen_gate_glob[:N_CH]
    ag = tr.frozen_a_glob[:N_CH]
    support = (tr.frozen_gate[:, :N_CH] > 0).float().sum(0)
    print(f"\nGLOBAL attractor: gate {float(gg.sum()):.1f}/16, "
          f"support {float(support.mean()):.1f} bins (need >= {tr.min_bins}), "
          f"mu mean over gated ch = "
          f"{float((ag*gg).sum()/gg.sum()) if gg.sum()>0 else float('nan'):+.3f}")
    logd = tr.log_summary()
    print(f"direction-invariance monitor: bin_scatter="
          f"{logd.get('attr_bin_scatter', float('nan')):.3f} "
          f"(vs pooled teacher spread — small = invariant)")
    # NEGATIVE CONTROL: naive pooled AR(1) across directions (Simpson-biased —
    # direction identity persists chunk-over-chunk, alpha inflates, the fixed
    # point explodes). Kept to demonstrate WHY estimation must be stratified.
    ps = tr.ema_sx.sum(0), tr.ema_sy.sum(0), tr.ema_sxx.sum(0), \
        tr.ema_sxy.sum(0), tr.ema_syy.sum(0), tr.ema_sn.sum(0)
    pn = ps[5].clamp(min=1e-6)
    pvx = (ps[2] / pn - (ps[0] / pn).pow(2)).clamp(min=1e-8)
    pcov = ps[3] / pn - (ps[0] / pn) * (ps[1] / pn)
    pal = pcov / pvx
    pa = ((ps[1] - pal * ps[0]) / pn) / (1 - pal).clamp(min=1e-3)
    print(f"[negative control] naive pooled fit: alpha(mu)="
          f"{float(pal[:N_CH].mean()):.3f} (inflated), attractor mu mean="
          f"{float(pa[:N_CH].mean()):+.3f} (nonsense)")

    # ---- plot ------------------------------------------------------------
    m0, pcs = fit_basis()
    fig, ax = plt.subplots(figsize=(9, 9))
    for di, d in enumerate(DIRS):
        if d not in student:
            continue
        for chunks in student[d]:
            pts = np.stack([proj_latent(c, m0, pcs) for c in chunks])
            ax.plot(pts[:, 0], pts[:, 1], "-o", color=DCOL[d], ms=3,
                    lw=1.2, alpha=0.75, label=d if chunks is student[d][0] else None)
        if G[di].sum() > 0:
            p = mu_point(A[di], m0, pcs)
            ax.plot(*p, marker="*", ms=14, color=DCOL[d], alpha=0.65,
                    markeredgecolor="k", zorder=5)
    # THE global direction-invariant attractor (the actuation reference)
    if float(gg.sum()) > 0:
        pg = mu_point(ag.numpy(), m0, pcs)
        ax.plot(*pg, marker="*", ms=30, color="gold", markeredgecolor="k",
                mew=2.0, zorder=8, label="GLOBAL attractor (actuation ref)")
    pp = mu_point(pa[:N_CH].numpy(), m0, pcs)
    ax.plot(*pp, marker="X", ms=14, color="0.5", zorder=7,
            label="naive pooled fit (Simpson-biased, negative control)")
    for d in DIRS:                     # teacher endpoints for reference
        if d in teacher:
            te = np.stack([proj_latent(teacher[d][s0, -1], m0, pcs)
                           for s0 in range(len(teacher[d]))])
            ax.plot(te[:, 0], te[:, 1], "s", color=DCOL[d], ms=5, alpha=0.35)
    ax.set_title(f"{RUN}: rollout trajectories (chunks 0-5), per-direction "
                 f"attractor estimates (stars),\npooled attractor (X), teacher "
                 f"endpoints (faint squares) — flow-tree PCA plane")
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(OUT, dpi=140)
    print(f"\n[at] wrote {OUT}")


if __name__ == "__main__":
    main()
