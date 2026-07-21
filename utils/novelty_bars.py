"""One-figure center vs edge DINO novelty, all 13 phase-A models.

center_nov: existing suite CSVs (suite_all.csv + suiteA_comp*.csv) — DINO
novelty of the mid-frame crop vs the video's own seed bank (max over time).
edge_nov: edge_nov*.csv from utils/edge_novelty.py — identical recipe on four
periphery strips.

Grouped bars (mean ± sem over the 256 videos per model) + the center/edge
ratio annotated above each pair: new content entering at the periphery is
normal for a moving camera; content materializing dead-center is not.

Writes analysis/eval_final/novelty_center_edge.png. Env: NV_OUT.
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
EF = f"{ARR}/analysis/eval_final"
OUT = os.environ.get("NV_OUT", f"{EF}/novelty_center_edge.png")
ORDER = ["pca8_8node", "16node", "pca4", "pca2", "4node", "noatok", "noadaln",
         "minwm", "matrixgame", "worldcam", "yume", "worldplay", "astra"]
LAB = {"pca8_8node": "ours pca8", "16node": "ours 16node"}

# Preferred source: center_jump_g*.csv (utils/center_jump.py) — first-6s-capped
# center/edge novelty AND the center-region spawn jump, so long-rollout models
# don't get extra time to accumulate novelty. Fallback: whole-video suite +
# edge_nov CSVs with the whole-frame spawn_jump.
cjf = sorted(glob.glob(f"{EF}/center_jump_g*.csv"))
if cjf:
    df = pd.concat([pd.read_csv(f) for f in cjf], ignore_index=True)
    df["spawn_jump"] = df["center_jump"]
    SPAWN_LABEL = "CENTER spawn jump (max single-step mid-view novelty increase)"
    CAP_NOTE = "first 6 s of each rollout"
else:
    parts = [pd.read_csv(f"{EF}/suite_all.csv")]
    for f in sorted(glob.glob(f"{EF}/suiteA_comp*.csv")):
        parts.append(pd.read_csv(f))
    cen = pd.concat(parts, ignore_index=True)[["run", "window", "dir", "center_nov", "spawn_jump"]]
    eparts = [pd.read_csv(f) for f in sorted(glob.glob(f"{EF}/edge_nov*.csv"))]
    edg = pd.concat(eparts, ignore_index=True)[["run", "window", "dir", "edge_nov"]]
    df = cen.merge(edg, on=["run", "window", "dir"], how="inner")
    SPAWN_LABEL = "SPAWN jump (max single-step whole-frame novelty increase)"
    CAP_NOTE = "whole rollout (UNCAPPED — lengths differ per model)"
models = [m for m in ORDER if m in set(df.run)]
print(df.groupby("run")[["center_nov", "edge_nov", "spawn_jump"]].agg(["count", "mean"]).round(3))

x = np.arange(len(models)); w = 0.27
fig, ax = plt.subplots(figsize=(1.45 * len(models) + 3, 6.5))
for k, (col, color, label) in enumerate([
        ("center_nov", "#b2182b", "CENTER novelty (mid-frame crop, max over rollout)"),
        ("edge_nov", "#2166ac", "EDGE novelty (periphery strips, max over rollout)"),
        ("spawn_jump", "#2ca02c", SPAWN_LABEL)]):
    m_ = [df[df.run == m][col].mean() for m in models]
    s_ = [df[df.run == m][col].sem() for m in models]
    ax.bar(x + (k - 1) * w, m_, w, yerr=s_, capsize=2, color=color, alpha=0.88, label=label)
for i, m in enumerate(models):
    c = df[df.run == m]["center_nov"].mean(); e = df[df.run == m]["edge_nov"].mean()
    ax.text(i, e + 0.035, f"c/e={c / e:.2f}", ha="center", fontsize=8, color="#444")
    sj = df[df.run == m]["spawn_jump"]
    ax.text(i + w, sj.mean() + 0.045, f"{(sj > 0.3).mean():.0%}", ha="center",
            fontsize=8, color="#1a6b1a", fontweight="bold")
ax.text(0.99, 0.02, "green % = spawn rate: fraction of videos with a center jump > 0.3",
        transform=ax.transAxes, ha="right", fontsize=9, color="#1a6b1a")
ax.set_xticks(x, [LAB.get(m, m) for m in models], rotation=30, ha="right")
ax.set_ylabel("DINO novelty vs own seed frames")
ax.set_title(f"Center / edge / spawn novelty, phase-A ({CAP_NOTE}; mean ± sem over 256 videos; same DINOv2 seed-bank recipe)\n"
             "edge novelty is expected for a moving camera; high CENTER = frame departs the seed world mid-view; "
             "high SPAWN = discontinuous appearance (gradual drift scores low)")
ax.legend(); ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print(f"[novelty] saved {OUT}")
