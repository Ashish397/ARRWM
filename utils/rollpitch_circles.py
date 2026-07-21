"""Roll (PC6) and Pitch (PC7) RESPONSE circles by direction wedge, split GT/FLIP.

For the 3 8-node PCA-dim runs (pca2, pca4, pca8): how much roll / pitch the model
produces for commands in each of 16 direction wedges. Uses the GENERATED teacher-
read g6 (roll) / g7 (pitch) -- valid (not the buggy real-ref r). Signed mean:
diverging colour = which way it rolls/pitches; intensity = how big. Global colour
scale across the 3 runs (and GT+FLIP) so images are comparable. Last 1000 steps.

Per run: 2x2 = [roll GT | roll FLIP ; pitch GT | pitch FLIP]  -> rollpitch_circle_{run}.png
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

CSV = "analysis/chunk_metrics.csv" if os.path.exists("analysis/chunk_metrics.csv") else "analysis/ndof_following.csv"
RUNS = {"pca2": "8node top-2 pca", "pca4": "8node top-4 pca", "pca8_8node": "8node top-8 pca"}
NSEC = 16
WIDTH = 360.0 / NSEC
LABELS = ["F", "F/FR", "FR", "FR/R", "R", "R/BR", "BR", "BR/B",
          "B", "B/BL", "BL", "BL/L", "L", "L/FL", "FL", "FL/F"]
LAST = 1000
MIN_N = 5
CMAP = plt.get_cmap("PuOr_r")   # purple = positive, orange = negative (distinct from the other circles)


def num(s):
    return pd.to_numeric(s, errors="coerce")


def wedge_means(d, col):
    c0, c1, v = num(d.cmd0).values, num(d.cmd1).values, num(d[col]).values
    bc = np.degrees(np.arctan2(c1, c0)); sec = (np.round(bc / WIDTH).astype(int)) % NSEC
    out = {}
    for s in range(NSEC):
        m = (sec == s) & np.isfinite(v); n = int(m.sum())
        out[s] = (float(v[m].mean()) if n >= MIN_N else np.nan, n)
    return out


def draw(ax, means, title, norm):
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_xticks(np.radians(np.arange(0, 360, WIDTH))); ax.set_xticklabels(LABELS, fontsize=7)
    ax.set_yticks([]); ax.set_ylim(0, 1)
    for s in range(NSEC):
        v, n = means[s]; theta = np.radians(s * WIDTH)
        color, ec = ("white", "lightgray") if (np.isnan(v) or n < MIN_N) else (CMAP(norm(v)), "k")
        ax.bar(theta, 1.0, width=np.radians(WIDTH), bottom=0, color=color, edgecolor=ec, linewidth=0.6, align="center")
        if n >= MIN_N:
            ax.text(theta, 0.66, f"{v:+.2f}", ha="center", va="center", fontsize=6)
    ax.set_title(title, fontsize=10, pad=14)


def main():
    df = pd.read_csv(CSV)
    # keep only the first 4 offset windows (levels 0-3) so every step/run is comparable
    if "offset" in df.columns:
        df = df[num(df["offset"]) <= 3 * 27]
    parts = [g[g.step >= g.step.max() - LAST] for _, g in df[df.run.isin(RUNS)].groupby("run")]
    df = pd.concat(parts) if parts else df
    stats = {}
    for run in RUNS:
        for br in ("gt", "flip"):
            d = df[(df.run == run) & (df.branch == br)]
            for col in ("g6", "g7"):
                stats[(run, br, col)] = wedge_means(d, col)
    # global scales for roll (g6) and pitch (g7) across all runs+branches
    def gmax(col):
        vals = [abs(v) for (r, b, c), m in stats.items() if c == col for v, n in m.values() if not np.isnan(v)]
        return max(0.05, max(vals) if vals else 0.05)
    rollM, pitchM = gmax("g6"), gmax("g7")
    rN, pN = TwoSlopeNorm(0, -rollM, rollM), TwoSlopeNorm(0, -pitchM, pitchM)
    print(f"global scales: roll ±{rollM:.2f} | pitch ±{pitchM:.2f}  (purple=+, orange=-)")
    for run, lab in RUNS.items():
        fig, ax = plt.subplots(2, 2, figsize=(13, 13), subplot_kw={"projection": "polar"})
        draw(ax[0][0], stats[(run, "gt", "g6")], f"ROLL (PC6) — GT  [±{rollM:.2f}]", rN)
        draw(ax[0][1], stats[(run, "flip", "g6")], "ROLL (PC6) — FLIP", rN)
        draw(ax[1][0], stats[(run, "gt", "g7")], f"PITCH (PC7) — GT  [±{pitchM:.2f}]", pN)
        draw(ax[1][1], stats[(run, "flip", "g7")], "PITCH (PC7) — FLIP", pN)
        fig.suptitle(f"{lab} — roll/pitch response by direction, last {LAST} steps (purple=+, orange=−, white=empty) — shared scale",
                     fontsize=13)
        fig.tight_layout(); fig.savefig(f"analysis/rollpitch_circle_{run}.png", dpi=115); plt.close(fig)
        print(f"saved analysis/rollpitch_circle_{run}.png")


if __name__ == "__main__":
    main()
