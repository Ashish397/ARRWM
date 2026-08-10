"""Flow tree with ACTION-SPREAD ellipses.

Projects every committed chunk onto the teacher's 2-D PCA plane and, for each
commanded direction, draws:
  * the mean trajectory (seed chunk -> chunk 5)
  * a 1-sigma covariance ELLIPSE over that direction's final-chunk points
    (all frames x all seeds), i.e. how tightly that action lands

Action controllability is then readable directly: well-separated ellipse
CENTRES with small ellipses = the model distinguishes the actions; overlapping
or co-located ellipses = it does not, regardless of what the latent statistics
say.

Env: FE_RUNS colon list, FE_OUT png path, FE_TITLE.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
FV = f"{ARR}/analysis/eval_final/flow_viz"
DIRS = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]
DCOL = {"F": "#d62728", "FR": "#ff7f0e", "R": "#bcbd22", "BR": "#2ca02c",
        "B": "#17becf", "BL": "#1f77b4", "L": "#9467bd", "FL": "#e377c2"}
NFB, C = 3, 16
RUNS = [r for r in os.environ.get("FE_RUNS", "").split(":") if r]
OUT = os.environ.get("FE_OUT", f"{FV}/flow_tree_ellipse.png")


def fit_basis():
    z = np.load(f"{FV}/trajs_14e8s20_w8.npz")
    # PER-FRAME basis: the stored teacher vectors are whole 3-frame chunks,
    # but the ellipses need frame-level points (3 frames x 2 seeds) or the
    # covariance is rank-deficient.
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


def load(run, m0, pcs, nseeds=2):
    out = {}
    for d in DIRS:
        per = []
        for s in range(nseeds):
            p = f"{FV}/flow_{run}/r08_{d}_s{s}/steps.npz"
            if not os.path.exists(p):
                continue
            z = np.load(p); sdt = z["sdt"]; ch = {}
            for i, (c, r, t) in enumerate(sdt):
                if t > 0 and int(r) == 3:
                    x = z[f"x{i}"].astype(np.float32).reshape(NFB, C, 60, 104)
                    ch[int(c)] = np.stack(
                        [(x[f].ravel() - m0) @ pcs for f in range(NFB)])
            if ch:
                per.append(np.stack([ch[c] for c in sorted(ch)]))  # [nc,3,2]
        if per:
            out[d] = np.stack(per)                                 # [ns,nc,3,2]
    return out


def gt_line(m0, pcs):
    """Real-video latents for the same window, projected per frame.

    One trajectory only (the ground truth has a single 'direction'), but it
    is the anchor the whole plot lacks: it shows where the REAL future goes
    relative to every commanded action.
    """
    import sys, json
    sys.path.insert(0, ARR)
    from utils.zarr_dataset import ZarrRideDataset
    w = json.load(open(f"{ARR}/analysis/eval_final/phaseA_windows.json"))[8]
    lat = np.asarray(ZarrRideDataset.load_latent_chunk(
        w["zarr_path"], int(w["offset"]), int(w["offset"]) + NFB * 7),
        dtype=np.float32)
    return np.stack([(lat[i].ravel() - m0) @ pcs for i in range(lat.shape[0])])


def ellipse(ax, pts, color):
    if len(pts) < 3:
        return
    mu = pts.mean(0); cov = np.cov(pts.T)
    w_, v_ = np.linalg.eigh(cov)
    order = np.argsort(w_)[::-1]; w_, v_ = w_[order], v_[:, order]
    ang = np.degrees(np.arctan2(v_[1, 0], v_[0, 0]))
    e = Ellipse(mu, 2 * np.sqrt(max(w_[0], 1e-12)), 2 * np.sqrt(max(w_[1], 1e-12)),
                angle=ang, facecolor=color, alpha=0.18, edgecolor=color, lw=1.6)
    ax.add_patch(e)


def main():
    m0, pcs = fit_basis()
    try:
        gt = gt_line(m0, pcs)
    except Exception as e:
        print(f"[fe] GT unavailable ({type(e).__name__}: {e})"); gt = None
    n = len(RUNS)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 6.4), squeeze=False)
    for ax, run in zip(axes[0], RUNS):
        if gt is not None:
            ax.plot(gt[:, 0], gt[:, 1], ":s", color="0.35", lw=2.6, ms=5,
                    alpha=0.9, zorder=12, label="GT (real video)")
            ellipse(ax, gt[-NFB:], "0.35")
        tr = load(run, m0, pcs)
        seps = []
        cents = {}
        for d in DIRS:
            if d not in tr:
                continue
            a = tr[d]                                   # [ns,nc,3,2]
            mean_traj = a.mean(axis=(0, 2))             # [nc,2]
            ax.plot(mean_traj[:, 0], mean_traj[:, 1], "-o", color=DCOL[d],
                    ms=3.5, lw=1.8, alpha=0.9, label=d)
            final = a[:, -1].reshape(-1, 2)             # spread at last chunk
            ellipse(ax, final, DCOL[d])
            cents[d] = final.mean(0)
        # separation-to-spread ratio: how distinguishable are the actions?
        if len(cents) > 1:
            cs = np.stack(list(cents.values()))
            dmat = np.linalg.norm(cs[:, None] - cs[None], axis=-1)
            sep = dmat[np.triu_indices(len(cs), 1)].mean()
            spreads = [np.linalg.norm(tr[d][:, -1].reshape(-1, 2).std(0))
                       for d in cents]
            ax.set_title(f"{run}\nmean action separation {sep:.0f} | "
                         f"mean spread {np.mean(spreads):.0f} | "
                         f"ratio {sep/max(np.mean(spreads),1e-6):.2f}", fontsize=9)
        ax.axhline(0, color="k", lw=0.3); ax.axvline(0, color="k", lw=0.3)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2, loc="lower left")
    fig.suptitle(os.environ.get("FE_TITLE", "Action spread (1-sigma ellipse at final chunk)"),
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT, dpi=130)
    print(f"[fe] saved {OUT}")


if __name__ == "__main__":
    main()
