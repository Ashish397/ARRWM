"""PCA of the fleet failure-metric space + figure (scree, loadings, model map)."""
import os
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "pca_fleet.png")

vlm = pd.read_csv(os.path.join(HERE, "results_external_vlm.csv"))
nov = pd.read_csv(os.path.join(HERE, "out/fleet_novelty.csv"))[["scene", "model", "novel_sudden"]]
dino = pd.read_csv(os.path.join(HERE, "out/fleet_dino.csv"))[["scene", "model", "dino_drift"]]
sc = pd.read_csv(os.path.join(HERE, "fleet_scene_consensus.csv"))[["scene", "model", "consensus_inl"]]
st = pd.read_csv(os.path.join(HERE, "fleet_static_inl.csv"))[["scene", "model", "static"]]
man = pd.read_csv(os.path.join(HERE, "out/fleet_mangle.csv"))
man["scene"] = man.apply(lambda r: f"r{int(r.scene):02d}_{r.direction}", axis=1)
OURS = ["pca8", "pca4", "pca2", "16node", "4node", "noatok", "noadaln"]
man["model"] = man.model.apply(lambda m: "ours_" + m if m in OURS else m)
man = man[["scene", "model", "pal4vst_max", "qwen_melt_pyes"]]

d = (vlm.merge(nov, on=["scene", "model"]).merge(dino, on=["scene", "model"])
     .merge(sc, on=["scene", "model"]).merge(st, on=["scene", "model"]).merge(man, on=["scene", "model"]))
d = d[d.static == 0].copy()
d["reloc"] = -d.consensus_inl
M = ["p_style", "p_uncanny", "p_novel", "novel_sudden", "pal4vst_max", "qwen_melt_pyes", "dino_drift", "reloc"]
d = d.dropna(subset=M)
X = d[M].astype(float)
Z = (X - X.mean()) / X.std()
U, S, Vt = np.linalg.svd(Z.values, full_matrices=False)
var = S**2 / (S**2).sum()
scores = U * S            # rollout coords in PC space

fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
# scree
ax[0].bar(range(1, len(var) + 1), var * 100, color="#4C72B0")
ax[0].plot(range(1, len(var) + 1), np.cumsum(var) * 100, "-o", color="#C44E52")
ax[0].set_xlabel("principal component"); ax[0].set_ylabel("% variance")
ax[0].set_title("Scree: 1 dominant axis (48%), ~3-4 effective")
for i, v in enumerate(var):
    ax[0].text(i + 1, v * 100 + 1, f"{v*100:.0f}", ha="center", fontsize=8)
# loadings heatmap (first 4 PCs)
k = 4
im = ax[1].imshow(Vt[:k].T, cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="auto")
ax[1].set_xticks(range(k)); ax[1].set_xticklabels([f"PC{i+1}\n{var[i]*100:.0f}%" for i in range(k)])
ax[1].set_yticks(range(len(M))); ax[1].set_yticklabels(M)
for a in range(len(M)):
    for b in range(k):
        ax[1].text(b, a, f"{Vt[b,a]:+.2f}", ha="center", va="center", fontsize=8,
                   color="white" if abs(Vt[b, a]) > 0.4 else "black")
ax[1].set_title("Loadings: PC1=severity, PC2=novelty, PC3=PAL")
fig.colorbar(im, ax=ax[1], fraction=0.046)
# model map in PC1-PC2 (centroids)
disp = {"worldplay": "WorldPlay", "matrixgame": "Matrix", "worldcam": "WorldCam", "astra": "Astra",
        "yume": "Yume", "minwm": "minWM", "ours_pca8": "pca8", "ours_pca4": "pca4", "ours_pca2": "pca2",
        "ours_16node": "16node", "ours_4node": "4node", "ours_noatok": "noatok", "ours_noadaln": "noadaln"}
d["pc1"], d["pc2"] = scores[:, 0], scores[:, 1]
for m, g in d.groupby("model"):
    ours = m.startswith("ours_")
    ax[2].scatter(g.pc1.mean(), g.pc2.mean(), s=90, color="#55A868" if ours else "#C44E52",
                  edgecolor="k", zorder=3)
    ax[2].annotate(disp[m], (g.pc1.mean(), g.pc2.mean()), fontsize=8,
                   xytext=(4, 3), textcoords="offset points")
ax[2].axhline(0, color="gray", lw=0.5); ax[2].axvline(0, color="gray", lw=0.5)
ax[2].set_xlabel("PC1  (general failure severity ->)"); ax[2].set_ylabel("PC2  (novelty ->)")
ax[2].set_title("Model centroids (green=ours, red=baselines)")
plt.tight_layout(); plt.savefig(OUT, dpi=120)
print("wrote", OUT)
