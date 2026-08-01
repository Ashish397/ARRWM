"""Fleet-wide HF-degradation ridgeline, all 13 models. ONE consistent mask:
feature_valid AND active (NO baseline_dirty / sharpness-quantile exclusion).
Same included rows as hf_distribution.py. B from fleet_hf.csv."""
import os
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "hf_fleet_distribution.png")
OURS = {"pca8", "pca4", "pca2", "16node", "4node", "noatok", "noadaln"}

mask = pd.read_csv(os.path.join(HERE, "canonical_static_mask.csv"))
mask["model"] = mask.model.apply(lambda x: "ours_" + x if x in OURS else x)
for c in ("feature_valid", "active"):
    mask[c] = mask[c].astype(bool)
keep = mask[mask.feature_valid & mask.active][["model", "scene"]]
hf = pd.read_csv(os.path.join(HERE, "out", "fleet_hf.csv"))
d = keep.merge(hf[["model", "scene", "B"]], on=["model", "scene"])

DISPLAY = {"ours_pca8": "Default", "ours_16node": "Batch64",
           "minwm": "minWM", "matrixgame": "Matrix-Game", "worldplay": "WorldPlay",
           "worldcam": "WorldCam", "astra": "Astra", "yume": "Yume"}
d = d[d.model.isin(DISPLAY)]
OURS_M = {m for m in DISPLAY if m.startswith("ours_")}

LO, HI = -500, 600
order = d.groupby("model").B.median().sort_values().index.tolist()
fig, ax = plt.subplots(figsize=(7.5, 7.2))
xs = np.linspace(LO, HI, 320)
for i, v in enumerate(order):
    b = d[d.model == v].B.values
    h, e = np.histogram(b, bins=40, range=(LO, HI), density=True)
    c = (e[:-1] + e[1:]) / 2
    y = np.interp(xs, c, h); y = y / (y.max() + 1e-9) * 1.7
    col = "#55A868" if v in OURS_M else "#4C72B0"
    ax.fill_between(xs, i, i + y, color=col, alpha=0.55, lw=0.8, edgecolor="k", zorder=i)
    ax.plot([np.median(b)], [i], "|", color="k", ms=11, mew=1.5, zorder=100)
ax.axvline(0, color="gray", ls="--", lw=0.8)
ax.axvline(150, color="#b2182b", ls=":", lw=1.1)
ax.text(150, len(order) - 0.3, r"$\mathcal{B}>150$", color="#b2182b", fontsize=8, ha="left", va="center")
ax.set_yticks(range(len(order))); ax.set_yticklabels([DISPLAY[v] for v in order])
ax.set_xlim(LO, HI)
ax.set_xlabel(r"HF degradation  $\mathcal{B}$  (sibling-relative sharpness loss; higher = more degraded)")
ax.legend(handles=[Patch(fc="#55A868", alpha=.55, ec="k", label="Ours / ablations"),
                   Patch(fc="#4C72B0", alpha=.55, ec="k", label="External baselines")],
          loc="upper left", fontsize=8, frameon=False)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(OUT, dpi=120, bbox_inches="tight")
print("wrote", OUT, "| rows:", len(d), "| mask = feature_valid & active, no dirty")
