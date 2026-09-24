"""Ablation-only HF-degradation ridgeline. ONE consistent mask: feature_valid AND
active (NO baseline_dirty / sharpness-quantile exclusion). B from fleet_hf.csv."""
import os
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "hf_distribution.png")
OURS = {"pca8", "pca4", "pca2", "16node", "4node", "noatok", "noadaln"}

mask = pd.read_csv(os.path.join(HERE, "canonical_static_mask.csv"))
mask["model"] = mask.model.apply(lambda x: "ours_" + x if x in OURS else x)
for c in ("feature_valid", "active"):
    mask[c] = mask[c].astype(bool)
keep = mask[mask.feature_valid & mask.active][["model", "scene"]]
hf = pd.read_csv(os.path.join(HERE, "out", "fleet_hf.csv"))
d = keep.merge(hf[["model", "scene", "B"]], on=["model", "scene"])

DISPLAY = {"ours_pca8": "Default", "ours_pca4": "pca4", "ours_pca2": "pca2",
           "ours_16node": "Batch64", "ours_4node": "Batch16",
           "ours_noatok": "No Action Tokens", "ours_noadaln": "No AdaLN"}
d = d[d.model.isin(DISPLAY)]

LO, HI = -400, 500
order = d.groupby("model").B.median().sort_values().index.tolist()   # stable (low) at bottom
fig, ax = plt.subplots(figsize=(7.5, 5))
xs = np.linspace(LO, HI, 300)
for i, v in enumerate(order):
    b = d[d.model == v].B.values
    h, e = np.histogram(b, bins=30, range=(LO, HI), density=True)
    c = (e[:-1] + e[1:]) / 2
    y = np.interp(xs, c, h); y = y / (y.max() + 1e-9) * 1.7
    ax.fill_between(xs, i, i + y, color="#55A868", alpha=0.55, lw=0.8, edgecolor="k", zorder=i)
    ax.plot([np.median(b)], [i], "|", color="k", ms=11, mew=1.5, zorder=100)
ax.axvline(0, color="gray", ls="--", lw=0.8)
ax.axvline(150, color="#b2182b", ls=":", lw=1.1)
ax.text(150, len(order) - 0.35, r"$\mathcal{B}>150$", color="#b2182b", fontsize=8, ha="left")
ax.set_yticks(range(len(order))); ax.set_yticklabels([DISPLAY[v] for v in order])
ax.set_xlim(LO, HI)
ax.set_xlabel(r"HF degradation  $\mathcal{B}$  (sibling-relative sharpness loss; higher = more degraded)")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(OUT, dpi=120, bbox_inches="tight")
print("wrote", OUT, "| rows:", len(d), "| mask = feature_valid & active, no dirty")
