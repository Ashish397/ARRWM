"""Response gain on the phase-A eval set, ALL models (ours + baselines).

The original response_gain.png fits realized-vs-commanded slopes over the
training control tests (varying magnitudes) — baselines can't have that. On
phase A every command has |component| in {0.5, 0.3536}, so gain reduces to
mean(realized component / commanded component) per axis. Axes reported:

  g0 gain, forward family  (F, FR, FL)    teacher-read throttle response / command
  g0 gain, backward family (B, BL, BR)
  g1 gain, steer (all 6 turning dirs, sign-corrected)

Bar chart per model (grouped) + printed table.
Writes analysis/response_gain_eval.png (+ _table.csv).
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
EF = f"{ARR}/analysis/eval_final"
OUT = os.environ.get("RG_OUT", f"{ARR}/analysis")
M = 0.5; Dv = M / np.sqrt(2)
CMD = {"F": (M, 0.0), "FR": (Dv, Dv), "R": (0.0, M), "BR": (-Dv, Dv),
       "B": (-M, 0.0), "BL": (-Dv, -Dv), "L": (0.0, -M), "FL": (Dv, -Dv)}
ORDER = ["pca8_8node", "16node", "pca4", "pca2", "4node", "noatok", "noadaln",
         "minwm", "matrixgame", "worldcam", "yume", "worldplay", "astra"]
LABEL = {"pca8_8node": "ours pca8", "16node": "ours 16node"}
GROUPS = [("g0 fwd (F,FR,FL)", ["F", "FR", "FL"], "g0", 0),
          ("g0 bwd (B,BL,BR)", ["B", "BL", "BR"], "g0", 0),
          ("g1 steer (6 dirs)", ["FR", "R", "BR", "BL", "L", "FL"], "g1", 1)]

frames = [pd.read_csv(f"{EF}/headtohead_motion.csv")]
for f in sorted(glob.glob(f"{EF}/headtohead_*.csv")):
    if os.path.basename(f) != "headtohead_motion.csv":
        frames.append(pd.read_csv(f))
df = pd.concat(frames, ignore_index=True)
models = [m for m in ORDER if m in set(df.model)]

rows = []
for m in models:
    sub = df[df.model == m]
    ent = {"model": m}
    for gname, dirs, col, ci in GROUPS:
        vals = []
        for D in dirs:
            c = CMD[D][ci]
            s = sub[sub["dir"] == D][col].astype(float)
            vals.extend((s / c).tolist())            # sign-corrected gain per video
        ent[gname] = float(np.nanmean(vals)) if vals else np.nan
    rows.append(ent)
tab = pd.DataFrame(rows).set_index("model")
tab.to_csv(f"{OUT}/response_gain_eval_table.csv")
print(tab.round(3))

x = np.arange(len(models)); w = 0.26
fig, ax = plt.subplots(figsize=(1.35 * len(models) + 3, 6.5))
for k, (gname, *_rest) in enumerate(GROUPS):
    ax.bar(x + (k - 1) * w, tab[gname], w, label=gname,
           color=["C0", "crimson", "C2"][k], alpha=0.85)
ax.axhline(1.0, color="gray", ls=":", lw=1.5)
ax.axhline(0.0, color="gray", lw=0.7)
ax.text(len(models) - 0.4, 1.03, "gain = 1 (faithful)", color="gray", fontsize=9)
ax.set_xticks(x, [LABEL.get(m, m) for m in models], rotation=30, ha="right")
ax.set_ylabel("response gain (teacher-read / commanded)")
ax.set_title("Phase-A response gain by axis — all models\n"
             "(negative = moves opposite to command; comparable across models: "
             "same videos, same frozen CoTracker→PCA teacher)")
ax.legend()
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(f"{OUT}/response_gain_eval.png", dpi=130)
print(f"[rgain] saved {OUT}/response_gain_eval.png")
