"""Per-model score-distribution ridgelines for each quality axis (raw scores,
no threshold). density (y) vs score (x), one ridge per model."""
import os
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "score_distributions.png")

vlm = pd.read_csv(os.path.join(HERE, "results_external_vlm.csv"))
nov = pd.read_csv(os.path.join(HERE, "out/fleet_novelty.csv"))[["scene", "model", "novel_sudden"]]
sc = pd.read_csv(os.path.join(HERE, "fleet_scene_consensus.csv"))[["scene", "model", "consensus_inl"]]
d = vlm.merge(nov, on=["scene", "model"]).merge(sc, on=["scene", "model"])
# scene score: fraction of end-frame inliers retained -> map to a 0..1 "relocation" score
d["reloc_score"] = 1 - np.clip(d.consensus_inl / 200.0, 0, 1)   # 1 = fully relocated

disp = {"worldplay": "WorldPlay", "matrixgame": "Matrix-Game", "worldcam": "WorldCam",
        "astra": "Astra", "yume": "Yume", "minwm": "minWM", "ours_pca8": "pca8",
        "ours_pca4": "pca4", "ours_pca2": "pca2", "ours_16node": "16node",
        "ours_4node": "4node", "ours_noatok": "noatok", "ours_noadaln": "noadaln"}
d["disp"] = d.model.map(disp)
OURSD = {"pca8", "pca4", "pca2", "16node", "4node", "noatok", "noadaln"}

AXES = [("p_style", "Style shift  P(Yes)"), ("p_uncanny", "Geometric corruption  P(Yes)"),
        ("novel_sudden", "Confabulation  (max sudden-novelty)"), ("reloc_score", "Scene relocation  (1 - inliers/200)")]

fig, axs = plt.subplots(1, 4, figsize=(20, 6.5), sharey=True)
for ax, (col, title) in zip(axs, AXES):
    order = d.groupby("disp")[col].mean().sort_values().index.tolist()   # calm at bottom
    xs = np.linspace(0, 1, 200)
    for i, m in enumerate(order):
        v = d[d.disp == m][col].values
        h, edges = np.histogram(v, bins=24, range=(0, 1), density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        y = np.interp(xs, centers, h)
        y = y / (y.max() + 1e-9) * 1.6
        c = "#55A868" if m in OURSD else "#C44E52"
        ax.fill_between(xs, i, i + y, color=c, alpha=0.55, lw=0.8, edgecolor="k", zorder=i)
        ax.plot([np.median(v)], [i], "|", color="k", ms=10, mew=1.4, zorder=100)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=9)
    ax.set_xlabel(title, fontsize=10); ax.set_xlim(0, 1); ax.set_title(title.split("  ")[0], fontsize=12)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
axs[0].set_ylabel("model (green = ours, red = baseline)   |   tick = median", fontsize=9)
plt.suptitle("Per-model score distributions (all 256 rollouts, no threshold)", fontsize=13)
plt.tight_layout(); plt.savefig(OUT, dpi=115, bbox_inches="tight")
print("wrote", OUT)
