"""Wedge (compass rose) plots of no-op camera drift per model. For each model, the
drift vector (dx,dy of the coherent background motion) of each of the 32 scenes is
binned into 8 compass sectors; wedge radius = summed drift magnitude in that sector.
A still model (good no-op) has tiny wedges all round; a drifter has a long wedge."""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "stationary_wedges.png")
d = pd.read_csv(os.path.join(HERE, "out", "stationary_cotracker.csv"))
disp = {"real": "REAL", "pca8": "Default", "pca4": "pca4", "pca2": "pca2", "16node": "Batch64",
        "4node": "Batch16", "noatok": "No Action Tokens", "noadaln": "No AdaLN", "minwm": "minWM",
        "worldplay": "WorldPlay", "matrixgame": "Matrix-Game", "worldcam": "WorldCam",
        "astra": "Astra", "yume": "Yume"}
order = ["pca4", "4node", "noatok", "pca2", "16node", "pca8", "noadaln",
         "minwm", "matrixgame", "worldplay", "worldcam", "yume", "astra"]
OURS = {"pca8", "pca4", "pca2", "16node", "4node", "noatok", "noadaln"}
NB = 8
edges = np.linspace(-np.pi, np.pi, NB + 1)
theta = (edges[:-1] + edges[1:]) / 2

def sector_r(m):
    g = d[d.model == m]
    ang = np.arctan2(-g.drift_dy.values, g.drift_dx.values)   # image y is down -> flip
    mag = np.hypot(g.drift_dx.values, g.drift_dy.values)
    r = np.zeros(NB)
    for a, mg in zip(ang, mag):
        r[min(np.searchsorted(edges, a) - 1, NB - 1)] += mg
    return r

rs = {m: sector_r(m) for m in order}
CLIP = {"astra", "yume", "noadaln"}                             # outliers -> clip
RMAX = max(rs[m].max() for m in order if m not in CLIP) * 1.02  # rim = tallest non-outlier

fig, axs = plt.subplots(2, 7, figsize=(22, 7.6), subplot_kw={"projection": "polar"})
axf = axs.ravel()
for ax, m in zip(axf, order):
    r = np.clip(rs[m], 0, RMAX)            # anything > pca8 -> clipped at rim
    ax.bar(theta, r, width=2 * np.pi / NB, color="#2E8B57", alpha=1.0, edgecolor="k", lw=0.5)
    clip = "  (clipped)" if rs[m].max() > RMAX else ""
    ax.set_title(f"{disp[m]} — {d[d.model==m].camera_motion.median():.1f}px{clip}", fontsize=10, pad=14)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270])); ax.set_xticklabels(["R", "U", "L", "D"], fontsize=8)
    ax.set_yticklabels([]); ax.set_rmax(RMAX); ax.set_rticks([RMAX / 2, RMAX]); ax.grid(alpha=0.3)
for ax in axf[len(order):]:
    ax.axis("off")
plt.tight_layout(); plt.savefig(OUT, dpi=110, bbox_inches="tight")
print(f"wrote {OUT}  (RMAX={RMAX:.0f})")
