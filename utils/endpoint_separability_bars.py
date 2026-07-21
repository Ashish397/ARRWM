"""Action-world-state consistency bars from grove_metrics.csv.

Endpoint = final VAE latent minus per-scene 8-action mean (8 dirs x 32 scenes,
full latent space; see utils/grove_tangle_metrics.py). Panels: leave-one-scene
-out 5-NN action ID accuracy (chance 0.125) + cosine silhouette by action.
Colors: dark blue = ours (pca8, 16node), teal = ablations, gray = minwm,
warm = external baselines. Writes analysis/eval_final/grove_endpoint_separability.png.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
df = pd.read_csv(f"{ARR}/analysis/eval_final/grove_metrics.csv").set_index("run")
ORDER = ["pca8_8node", "16node", "pca4", "pca2", "4node", "noatok", "noadaln",
         "minwm", "matrixgame", "worldcam", "yume", "worldplay", "astra"]
LAB = {"pca8_8node": "ours pca8", "16node": "ours 16node"}
COLS = ["#08306b", "#2166ac", "#66c2a5", "#8fd0b8", "#a8dbc6", "#c0e5d4", "#d8efe2",
        "#7f7f7f", "#b2182b", "#d6604d", "#e08214", "#8c510a", "#c51b7d"]
df = df.loc[[m for m in ORDER if m in df.index]]

fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
for ax, k, title, chance in [
        (axes[0], "knn_acc", "action ID from latent endpoint\n(leave-one-scene-out 5-NN, cosine)", 0.125),
        (axes[1], "silhouette", "cosine silhouette of endpoints\nby commanded action", 0.0)]:
    v = df[k].values
    ax.bar(range(len(df)), v, color=COLS[:len(df)], alpha=0.9)
    ax.axhline(chance, color="gray", ls=":", lw=1.5)
    ax.text(len(df) - 0.3, chance + 0.01, "chance" if k == "knn_acc" else "no structure",
            color="gray", fontsize=9, ha="right")
    for i, x in enumerate(v):
        ax.text(i, x + 0.01, f"{x:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(df)), [LAB.get(m, m) for m in df.index],
                  rotation=35, ha="right", fontsize=9)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
fig.suptitle("Action–world-state consistency: commanded direction recovered from the final-chunk latent residual\n"
             "(endpoint = final VAE latent minus per-scene 8-action mean; 8 directions x 32 scenes; identical pipeline all models)",
             fontsize=11)
fig.tight_layout()
fig.savefig(f"{ARR}/analysis/eval_final/grove_endpoint_separability.png", dpi=130)
print("[sep] saved grove_endpoint_separability.png")
