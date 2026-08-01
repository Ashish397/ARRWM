"""Phase-A response CURVES, all models: realized (teacher-read) vs commanded,
per axis, using the 5 signed command levels the 8-direction protocol spans:

  g0 (throttle): F=+0.5, FR/FL=+0.354, R/L=0, BR/BL=-0.354, B=-0.5
  g1 (steer)   : R=+0.5, FR/BR=+0.354, F/B=0, FL/BL=-0.354, L=-0.5

One line per model (mean +- sem over videos at each level), diagonal = gain 1.
Ours in blues, external baselines in warm colors. The all-models analog of the
training-time response_gain curves (which need varying magnitudes and exist
only for our runs). Writes analysis/response_curves_eval.png.
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figures.figure_labels import label

ARR = os.environ.get("AF_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EF = f"{ARR}/analysis/eval_final"
OUT = os.environ.get("RC_OUT", f"{ARR}/analysis")
M = 0.5; Dv = round(M / np.sqrt(2), 4)
CMD = {"F": (M, 0.0), "FR": (Dv, Dv), "R": (0.0, M), "BR": (-Dv, Dv),
       "B": (-M, 0.0), "BL": (-Dv, -Dv), "L": (0.0, -M), "FL": (Dv, -Dv)}
# main models only (continuous action space -> real dose axis -> LINES) +
# external baselines (fixed-strength interfaces -> unconnected MARKERS)
ORDER = ["pca8_8node", "16node",
         "minwm", "matrixgame", "worldcam", "yume", "worldplay", "astra"]
OURS = {"pca8_8node", "16node"}
STYLE = {  # ours = blues/solid lines, external = warm markers
    "pca8_8node": ("#08306b", "-", 2.6), "16node": ("#2166ac", "-", 2.6),
    "minwm": ("#7f7f7f", "o", 1.8), "matrixgame": ("#b2182b", "s", 1.8),
    "worldcam": ("#d6604d", "D", 1.8), "yume": ("#e08214", "^", 1.8),
    "worldplay": ("#8c510a", "v", 1.8), "astra": ("#c51b7d", "P", 1.8)}

frames = [pd.read_csv(f"{EF}/headtohead_motion.csv")]
for f in sorted(glob.glob(f"{EF}/headtohead_*.csv")):
    if os.path.basename(f) != "headtohead_motion.csv":
        frames.append(pd.read_csv(f))
df = pd.concat(frames, ignore_index=True)
df["c0"] = [CMD[d][0] for d in df["dir"]]
df["c1"] = [CMD[d][1] for d in df["dir"]]
models = [m for m in ORDER if m in set(df.model)]

for fname, ccol, gcol in [("response_curves_eval_throttle.png", "c0", "g0"),
                          ("response_curves_eval_steer.png", "c1", "g1")]:
    fig, ax = plt.subplots(figsize=(9, 7.5))
    lo, hi = -0.62, 0.62
    ax.plot([lo, hi], [lo, hi], color="gray", ls=":", lw=1.5, label="gain = 1 (faithful)")
    ax.axhline(0, color="gray", lw=0.6); ax.axvline(0, color="gray", lw=0.6)
    for m in models:
        sub = df[df.model == m]
        g = sub.groupby(ccol)[gcol].agg(["mean", "sem"]).sort_index()
        col, style, lw = STYLE[m]
        if m in OURS:   # continuous action space: a real dose axis -> line
            ax.plot(g.index, g["mean"], color=col, ls=style, lw=lw,
                    marker="o", ms=4, label=label(m, style="curves"))
        else:           # fixed-strength interface: distinct markers, thin line
            ax.plot(g.index, g["mean"], color=col, ls="-", lw=1.4,
                    marker=style, ms=7, alpha=0.9, label=label(m, style="curves"))
        ax.fill_between(g.index, g["mean"] - g["sem"], g["mean"] + g["sem"],
                        color=col, alpha=0.3, lw=0)
    ax.set_xlabel("commanded value"); ax.set_ylabel("teacher-read response (CoTracker→PCA)")
    ax.grid(alpha=0.3)
    ax.set_xlim(lo, hi)
    ax.legend(ncol=2, fontsize=13, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}", dpi=130)
    plt.close(fig)
    print(f"[rcurves] saved {OUT}/{fname}")
