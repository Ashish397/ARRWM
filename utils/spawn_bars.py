"""Spawn-detector bar chart, all 13 phase-A models.

Reads spawn_det_g*.csv from utils/spawn_detect.py (interior-blob spawn
events, boundary entries excluded, first 6 s). Two panels:
  1. spawn RATE: fraction of the 256 videos with >= 1 interior spawn
     (robust to one object being counted as several expansion events)
  2. mean spawns per video, with mean max blob area annotated.

Writes analysis/eval_final/spawn_bars.png. Env: SB_OUT.
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
EF = f"{ARR}/analysis/eval_final"
OUT = os.environ.get("SB_OUT", f"{EF}/spawn_bars.png")
ORDER = ["pca8_8node", "16node", "pca4", "pca2", "4node", "noatok", "noadaln",
         "minwm", "matrixgame", "worldcam", "yume", "worldplay", "astra"]
LAB = {"pca8_8node": "ours pca8", "16node": "ours 16node"}
# "ours" = pca8 + 16node only (dark blue); ablations teal; baselines warm
COLS = ["#08306b", "#2166ac", "#66c2a5", "#8fd0b8", "#a8dbc6", "#c0e5d4", "#d8efe2",
        "#7f7f7f", "#b2182b", "#d6604d", "#e08214", "#8c510a", "#c51b7d"]

df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{EF}/spawn_det_g*.csv"))],
               ignore_index=True)
models = [m for m in ORDER if m in set(df.run)]

# OBJECT spawn = fresh interior blob with area >= 12 patches (~1.2% of frame).
# Small fresh blobs (4-8 patches) are dominated by legitimate content flowing
# in from the vanishing point under forward motion, not object spawns — the
# smoke-verified true spawns (taxi 19 / car 14 / truck 15) all clear 12.
rate12 = [(df[df.run == m].max_spawn_area >= 12).mean() for m in models]
area = [df[(df.run == m) & (df.max_spawn_area >= 12)].max_spawn_area.mean() for m in models]
AREAS = [8, 12, 16]
sweep = {a: [(df[df.run == m].max_spawn_area >= a).mean() for m in models] for a in AREAS}
print(pd.DataFrame({"model": models, "obj_spawn_rate(a>=12)": np.round(rate12, 3),
                    "mean_area": np.round(area, 1)}).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
ax = axes[0]
ax.bar(range(len(models)), rate12, color=COLS[:len(models)], alpha=0.9)
for i, v in enumerate(rate12):
    ax.text(i, v + 0.004, f"{v:.0%}", ha="center", fontsize=9)
ax.set_title("OBJECT spawn rate — fraction of 256 videos with a fresh interior blob ≥ 12 patches")
ax.set_ylabel("fraction of videos")
ax = axes[1]
w = 0.26
for k, a in enumerate(AREAS):
    ax.bar(np.arange(len(models)) + (k - 1) * w, sweep[a], w,
           label=f"area ≥ {a}", alpha=0.9)
ax.set_title("threshold sweep — smaller blobs include vanishing-point inflow, "
             "large blobs = object spawns (minwm, astra)")
ax.legend(fontsize=9)
for ax in axes:
    ax.set_xticks(range(len(models)), [LAB.get(m, m) for m in models],
                  rotation=30, ha="right", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
fig.suptitle("Interior object spawning, phase-A first 6 s — DINOv2 patch-novelty blobs appearing fresh in the frame\n"
             "interior (boundary entries and moving/growing existing objects excluded); "
             "identical detector + per-video adaptive threshold for all models", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(OUT, dpi=130)
print(f"[spawn] saved {OUT}")
