"""Circular (compass-wedge) error graphs, per run, over the LAST 500 steps.

16 direction wedges (22.5 deg each). Command vector (throttle=PC0 North, steer=PC1
East) -> bearing -> wedge. GT + FLIP pooled. For each wedge, comparing realized
teacher-read (g0,g1) to commanded (cmd0,cmd1):
  Direction error = signed angular error (deg) of realized bearing vs commanded.
  Magnitude error = |realized| - |commanded| (over/under response).

Layout 2x2:  LEFT column = DIRECTION (yellow<->blue),  RIGHT column = MAGNITUDE (green<->pink)
  top row    = ERROR circles      (diverging; white = correct)
  bottom row = UNCERTAINTY circles (sequential; pale = certain, saturated = uncertain)
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize, LinearSegmentedColormap

CSV = "analysis/chunk_metrics.csv"          # unified source (falls back to ndof_following.csv)
FALLBACK = "analysis/ndof_following.csv"
RUNS = {"pca8_8node": "8node (tokens ON)", "noatok": "noatok (tokens OFF)",
        "16node": "16node (b64)", "pca4": "pca4 (top4)", "pca2": "pca2 (top2)", "4node": "4node (b16)"}
NSEC = 16
WIDTH = 360.0 / NSEC                          # 22.5 deg
LABELS = ["F", "F/FR", "FR", "FR/R", "R", "R/BR", "BR", "BR/B",
          "B", "B/BL", "BL", "BL/L", "L", "L/FL", "FL", "FL/F"]
LAST = 500
MIN_N = 5

# colour families
DIR_ERR = LinearSegmentedColormap.from_list("ylbu", ["#e6a000", "white", "#1f4fd8"])   # yellow<-0->blue
DIR_UNC = LinearSegmentedColormap.from_list("ylbu_seq", ["#ffffcc", "#1a2f8f"])        # pale yellow->deep blue
MAG_ERR = LinearSegmentedColormap.from_list("grpk", ["#1a9850", "white", "#d6006e"])   # green<-0->pink
MAG_UNC = LinearSegmentedColormap.from_list("grpk_seq", ["#d7f0d0", "#980043"])        # pale green->deep pink


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def wedge_stats(d):
    c0, c1 = d.cmd0.values.astype(float), d.cmd1.values.astype(float)
    g0, g1 = d.g0.values.astype(float), d.g1.values.astype(float)
    bc = np.degrees(np.arctan2(c1, c0)); bg = np.degrees(np.arctan2(g1, g0))
    ang_err = wrap180(bg - bc)
    mag_err = np.hypot(g0, g1) - np.hypot(c0, c1)
    sec = (np.round(bc / WIDTH).astype(int)) % NSEC
    st = {}
    for s in range(NSEC):
        m = sec == s; n = int(m.sum())
        if n >= MIN_N:
            st[s] = dict(n=n, ang_mean=float(np.mean(ang_err[m])), ang_std=float(np.std(ang_err[m])),
                         mag_mean=float(np.mean(mag_err[m])), mag_std=float(np.std(mag_err[m])))
        else:
            st[s] = dict(n=n, ang_mean=np.nan, ang_std=np.nan, mag_mean=np.nan, mag_std=np.nan)
    return st


def draw(ax, st, key, title, norm, cmap, fmt):
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_xticks(np.radians(np.arange(0, 360, WIDTH)))
    ax.set_xticklabels(LABELS, fontsize=7)
    ax.set_yticks([]); ax.set_ylim(0, 1)
    for s in range(NSEC):
        v = st[s][key]; theta = np.radians(s * WIDTH)
        color, ec = ("white", "lightgray") if (np.isnan(v) or st[s]["n"] < MIN_N) else (cmap(norm(v)), "k")
        ax.bar(theta, 1.0, width=np.radians(WIDTH), bottom=0, color=color, edgecolor=ec, linewidth=0.6, align="center")
        if st[s]["n"] >= MIN_N:
            ax.text(theta, 0.66, f"{v:{fmt}}", ha="center", va="center", fontsize=6)
    ax.set_title(title, fontsize=10, pad=14)


def load():
    import os
    p = CSV if os.path.exists(CSV) else FALLBACK
    df = pd.read_csv(p)
    # keep only the first 4 offset windows (levels 0-3) so every step/run is comparable
    if "offset" in df.columns:
        df = df[pd.to_numeric(df["offset"], errors="coerce") <= 3 * 27]
    return df


def main():
    df = load()
    df["chunk"] = pd.to_numeric(df["chunk"], errors="coerce")
    # natural = all chunks; flip-settled = drop the first 2 FLIP chunks (decel-then-reverse transient)
    MODES = [
        ("", "all chunks", df),
        ("_flipsettled", "flip chunks 2+ (transient dropped)", df[~((df.branch == "flip") & (df.chunk < 2))]),
    ]
    for suffix, note, dfm in MODES:
        stats = {}
        for run in RUNS:
            d = dfm[dfm.run == run]
            if not len(d):
                continue
            mx = int(d.step.max())
            d = d[d.step >= mx - LAST].dropna(subset=["cmd0", "cmd1", "g0", "g1"])
            if len(d):
                stats[run] = (wedge_stats(d), mx)
        if not stats:
            continue
        # GLOBAL colour scale over ALL runs' valid wedges (this mode)
        def gmax(key, floor):
            vals = [abs(st[s][key]) for st, _ in stats.values() for s in range(NSEC) if not np.isnan(st[s][key])]
            return max(floor, max(vals) if vals else floor)
        aM, mM = gmax("ang_mean", 20.0), gmax("mag_mean", 0.1)
        aU, mU = gmax("ang_std", 1.0), gmax("mag_std", 1.0)
        a_err, m_err = TwoSlopeNorm(0, -aM, aM), TwoSlopeNorm(0, -mM, mM)
        a_unc, m_unc = Normalize(0, aU), Normalize(0, mU)
        print(f"[{note}] GLOBAL scales: dir-err ±{aM:.0f}deg | mag-err ±{mM:.2f} | dir-unc 0..{aU:.0f} | mag-unc 0..{mU:.2f}")
        for run, (st, mx) in stats.items():
            fig, ax = plt.subplots(2, 2, figsize=(14, 14), subplot_kw={"projection": "polar"})
            draw(ax[0][0], st, "ang_mean", f"DIRECTION error (deg) — blue=+, yellow=−, white=correct  [±{aM:.0f}°]",
                 a_err, DIR_ERR, "+.0f")
            draw(ax[1][0], st, "ang_std", f"DIRECTION uncertainty — pale=certain, blue=uncertain  [0..{aU:.0f}°]",
                 a_unc, DIR_UNC, ".0f")
            draw(ax[0][1], st, "mag_mean", f"MAGNITUDE error — pink=over, green=under, white=correct  [±{mM:.2f}]",
                 m_err, MAG_ERR, "+.2f")
            draw(ax[1][1], st, "mag_std", f"MAGNITUDE uncertainty — pale=certain, pink=uncertain  [0..{mU:.2f}]",
                 m_unc, MAG_UNC, ".2f")
            fig.suptitle(f"{RUNS[run]} — 16 wedges, last {LAST} steps ({note}) — SHARED scale across runs", fontsize=13)
            fig.tight_layout(); fig.savefig(f"analysis/circular_{run}{suffix}.png", dpi=115); plt.close(fig)
            print(f"saved analysis/circular_{run}{suffix}.png")


if __name__ == "__main__":
    main()
