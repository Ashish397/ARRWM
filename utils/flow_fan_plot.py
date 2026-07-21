"""Re-plot flow_viz trajectories as an action-residual fan (no GPU rollouts).

Reads trajectories_w{W}.npz written by utils/flow_viz.py. The naive shared-PCA
plot superimposes: the shared noise->data flow (~704 units) and inter-seed
offsets (~774 units) dominate the top PCs while the action fan is only
~120-205 units and orthogonal to them. Fix: per (seed, step) subtract the
across-action mean latent, PCA the residuals — every action starts at the
origin and fans out; plus a log-scale separation-vs-step curve.

Env: FV_WINDOW (def 8), FV_NSEEDS (def 4), FV_OUT (def analysis/eval_final/flow_viz).
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
OUT = os.environ.get("FV_OUT", f"{ARR}/analysis/eval_final/flow_viz")
W = int(os.environ.get("FV_WINDOW", "8"))
NS = int(os.environ.get("FV_NSEEDS", "4"))
DIRS = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]
COLORS = {"F": "#1f77b4", "FR": "#17becf", "R": "#2ca02c", "BR": "#bcbd22",
          "B": "#d62728", "BL": "#e377c2", "L": "#9467bd", "FL": "#8c564b"}

z = np.load(f"{OUT}/trajectories_w{W}.npz")
T = z[f"F_0"].shape[0]

res = {}                                     # (dir, seed) -> [T, D] float32 residual
sep = np.zeros(T)
for sd in range(NS):
    tr = np.stack([z[f"{d}_{sd}"] for d in DIRS]).astype(np.float32)   # [8, T, D]
    for si in range(T):
        dd = np.linalg.norm(tr[:, None, si] - tr[None, :, si], axis=-1)
        sep[si] += dd[np.triu_indices(8, 1)].mean() / NS
    tr -= tr.mean(0, keepdims=True)
    for i, d in enumerate(DIRS):
        res[(d, sd)] = tr[i]
    del tr

allr = torch.from_numpy(np.concatenate([res[(d, sd)] for d in DIRS for sd in range(NS)], 0))
U, S, V = torch.pca_lowrank(allr, q=2, niter=6)
P = V[:, :2].numpy()
evr = (S[:2] ** 2 / allr.pow(2).sum()).numpy()
del allr

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 8), gridspec_kw={"width_ratios": [1.6, 1]})
for (d, sd), r in res.items():
    p2 = r @ P
    ax1.plot(p2[:, 0], p2[:, 1], color=COLORS[d], alpha=0.8, lw=1.7,
             label=d if sd == 0 else None)
    ax1.scatter(p2[-1, 0], p2[-1, 1], color=COLORS[d], s=80, marker="*",
                edgecolor="black", lw=0.5, zorder=5)
ax1.scatter([0], [0], color="black", s=40, zorder=6)
ax1.axhline(0, color="gray", lw=0.4); ax1.axvline(0, color="gray", lw=0.4)
ax1.set_title(f"Action-residual denoising trajectories — window r{W:02d}, {NS} seeds\n"
              f"x_t minus across-action mean at each step; all start at origin (shared noise)\n"
              f"PC1 {evr[0]*100:.0f}% / PC2 {evr[1]*100:.0f}% of residual variance")
ax1.legend(ncol=4, fontsize=10)
ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2")

ax2.semilogy(range(T), np.maximum(sep, 1e-3), lw=2, color="#333")
ax2.set_xlabel("denoising step (0 = initial noise)")
ax2.set_ylabel("mean pairwise inter-action distance (log)")
ax2.set_title("When the action bends the flow")
ax2.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(f"{OUT}/flow_fan_w{W}.png", dpi=130)
print(f"saved {OUT}/flow_fan_w{W}.png")
print("sep at steps 1,5,10,20,30,40,48:", [round(float(sep[i]), 1) for i in (1, 5, 10, 20, 30, 40, T - 1)])
