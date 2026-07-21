"""Phase A (action-injection) analysis. Reads the chunk_metrics CSV produced from
the injected eval (branch = one of 8 compass directions, all seeds NON-moving,
commanded magnitude fixed ~0.5). Produces, per model and across models:

  1. 8-wedge circle (direction error + magnitude error) per run   -> phaseA_circle_{run}.png
  2. response: realized magnitude & following-cosine per direction -> phaseA_response.png
  3. roll/pitch per direction per run                              -> phaseA_rollpitch_{run}.png
  4. per-direction IQA + COLLAPSE (no-response + degeneracy) table -> phaseA_quality.png + .csv

collapse := chunk commanded |cmd|>0.25 but realized |g|<0.1 (model ignored the command)
         OR niqe/brisque blow-up (rendering degenerated).
"""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

CSV = os.environ.get("PA_CSV", "analysis/eval_final/phaseA_metrics.csv")
OUT = "analysis/eval_final"
RUNS = ["pca8_8node", "pca4", "pca2", "16node", "4node", "noatok", "noadaln"]
DIRS = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]          # 8 wedges, N=throttle+, E=steer+
ANG = {d: i * 45.0 for i, d in enumerate(DIRS)}
NSEC = 8; WIDTH = 45.0
DIR_ERR = LinearSegmentedColormap.from_list("ylbu", ["#e6a000", "white", "#1f4fd8"])
MAG_ERR = LinearSegmentedColormap.from_list("grpk", ["#1a9850", "white", "#d6006e"])


def num(s): return pd.to_numeric(s, errors="coerce")


def wedge_stats(d):
    """per injected-direction wedge: mean signed angular err (deg) & magnitude err."""
    c0, c1 = num(d.cmd0).values, num(d.cmd1).values
    g0, g1 = num(d.g0).values, num(d.g1).values
    bc = np.degrees(np.arctan2(c1, c0)); bg = np.degrees(np.arctan2(g1, g0))
    ang = (bg - bc + 180) % 360 - 180
    mag = np.hypot(g0, g1) - np.hypot(c0, c1)
    sec = (np.round(bc / WIDTH).astype(int)) % NSEC
    st = {}
    for s in range(NSEC):
        m = sec == s; n = int(np.sum(m))
        st[s] = dict(n=n, ang=float(np.nanmean(ang[m])) if n else np.nan,
                     mag=float(np.nanmean(mag[m])) if n else np.nan)
    return st


def draw_circle(ax, st, key, title, norm, cmap, fmt):
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_xticks(np.radians(np.arange(0, 360, WIDTH))); ax.set_xticklabels(DIRS, fontsize=9)
    ax.set_yticks([]); ax.set_ylim(0, 1)
    for s in range(NSEC):
        v = st[s][key]; th = np.radians(s * WIDTH)
        color, ec = ("white", "lightgray") if np.isnan(v) else (cmap(norm(v)), "k")
        ax.bar(th, 1.0, width=np.radians(WIDTH), color=color, edgecolor=ec, lw=.6, align="center")
        if not np.isnan(v):
            ax.text(th, 0.62, f"{v:{fmt}}", ha="center", va="center", fontsize=7)
    ax.set_title(title, fontsize=10, pad=12)


def main():
    df = pd.read_csv(CSV)
    df["mag_g"] = np.hypot(num(df.g0), num(df.g1))
    df["mag_c"] = np.hypot(num(df.cmd0), num(df.cmd1))
    df["cos"] = (num(df.cmd0) * num(df.g0) + num(df.cmd1) * num(df.g1)) / (df.mag_c * df.mag_g).replace(0, np.nan)
    runs = [r for r in RUNS if r in set(df.run.unique())]

    # 1) circles (global scale across runs)
    stats = {r: wedge_stats(df[df.run == r]) for r in runs}
    aM = max(20.0, max(abs(stats[r][s]["ang"]) for r in runs for s in range(NSEC) if not np.isnan(stats[r][s]["ang"])))
    mM = max(0.1, max(abs(stats[r][s]["mag"]) for r in runs for s in range(NSEC) if not np.isnan(stats[r][s]["mag"])))
    for r in runs:
        fig, ax = plt.subplots(1, 2, figsize=(15, 7.5), subplot_kw={"projection": "polar"})
        draw_circle(ax[0], stats[r], "ang", f"DIRECTION error (deg)  [±{aM:.0f}°]",
                    TwoSlopeNorm(0, -aM, aM), DIR_ERR, "+.0f")
        draw_circle(ax[1], stats[r], "mag", f"MAGNITUDE error (realized-commanded)  [±{mM:.2f}]",
                    TwoSlopeNorm(0, -mM, mM), MAG_ERR, "+.2f")
        fig.suptitle(f"{r} — Phase A injected-direction response (8 wedges) — shared scale", fontsize=13)
        fig.tight_layout(); fig.savefig(f"{OUT}/phaseA_circle_{r}.png", dpi=115); plt.close(fig)

    # 2) response: realized magnitude + cosine per direction, runs overlaid
    fig, ax = plt.subplots(1, 2, figsize=(20, 6.5))
    x = np.arange(len(DIRS)); wds = 0.8 / max(1, len(runs))
    for i, r in enumerate(runs):
        d = df[df.run == r]
        magd = [d[d.branch == b].mag_g.mean() for b in DIRS]
        cosd = [d[d.branch == b].cos.mean() for b in DIRS]
        ax[0].bar(x + i * wds, magd, wds, label=r)
        ax[1].bar(x + i * wds, cosd, wds, label=r)
    ax[0].axhline(0.5, color="k", ls="--", lw=1, label="commanded 0.5")
    ax[0].set_title("realized motion magnitude per direction"); ax[0].set_ylabel("|realized|")
    ax[1].axhline(1.0, color="k", ls="--", lw=1); ax[1].set_title("following cosine per direction")
    for a in ax:
        a.set_xticks(x + 0.4); a.set_xticklabels(DIRS); a.legend(fontsize=8); a.grid(alpha=.3)
    fig.suptitle("Phase A — response by injected direction (all seeds non-moving, |cmd|≈0.5)", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{OUT}/phaseA_response.png", dpi=120); plt.close(fig)

    # 3) roll/pitch per direction per run (signed mean g6/g7)
    for r in runs:
        d = df[df.run == r]
        fig, ax = plt.subplots(1, 2, figsize=(18, 5.5))
        for j, (col, nm) in enumerate([("g6", "ROLL (PC6)"), ("g7", "PITCH (PC7)")]):
            vals = [num(d[d.branch == b][col]).mean() for b in DIRS]
            ax[j].bar(DIRS, vals, color="#7a3fae"); ax[j].axhline(0, color="k", lw=.6)
            ax[j].set_title(f"{nm} by injected direction"); ax[j].grid(alpha=.3)
        fig.suptitle(f"{r} — Phase A uncommanded roll/pitch by direction", fontsize=13)
        fig.tight_layout(); fig.savefig(f"{OUT}/phaseA_rollpitch_{r}.png", dpi=115); plt.close(fig)

    # 4) per-direction QUALITY = does the video DEGENERATE?  (following lives in the response graph)
    #    collapse = rendering degenerated: musiq crash OR niqe blow-up (own-distribution 5%/95% tails)
    mus, niq = num(df.musiq), num(df.niqe)
    MUS_LO, NIQ_HI = float(mus.quantile(0.05)), float(niq.quantile(0.95))
    df["collapse"] = ((mus < MUS_LO) | (niq > NIQ_HI)).astype(float)
    rows = []
    for r in runs:
        for b in DIRS:
            d = df[(df.run == r) & (df.branch == b)]
            if not len(d):
                continue
            rows.append(dict(run=r, dir=b, n=len(d),
                             musiq=round(num(d.musiq).mean(), 2), niqe=round(num(d.niqe).mean(), 3),
                             brisque=round(num(d.brisque).mean(), 2),
                             collapse=round(d.collapse.mean(), 3)))
    q = pd.DataFrame(rows); q.to_csv(f"{OUT}/phaseA_quality.csv", index=False)

    def grouped(ax, metric, title):
        x = np.arange(len(DIRS)); wds = 0.8 / max(1, len(runs))
        for i, r in enumerate(runs):
            vals = [q[(q.run == r) & (q.dir == b)][metric].mean() if len(q[(q.run == r) & (q.dir == b)]) else np.nan
                    for b in DIRS]
            ax.bar(x + i * wds, vals, wds, label=r)
        ax.set_xticks(x + 0.4); ax.set_xticklabels(DIRS); ax.set_title(title); ax.grid(alpha=.3)

    # QUALITY / degeneration bar charts (same style as response) -- MUSIQ / NIQE / BRISQUE / collapse
    fig, ax = plt.subplots(1, 4, figsize=(26, 6))
    grouped(ax[0], "musiq", "MUSIQ (higher = better)")
    grouped(ax[1], "niqe", "NIQE (lower = better)")
    grouped(ax[2], "brisque", "BRISQUE (lower = better)")
    grouped(ax[3], "collapse", f"COLLAPSE rate — degenerate frames (musiq<{MUS_LO:.0f} OR niqe>{NIQ_HI:.1f})")
    ax[0].legend(fontsize=8)
    fig.suptitle("Phase A — video-quality / degeneration by injected direction "
                 "(following is in phaseA_response; MAE/LPIPS N/A here — counterfactual, no real ref)", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{OUT}/phaseA_quality.png", dpi=120); plt.close(fig)

    # roll/pitch cross-run comparison (|mean| magnitude per direction, pca2..pca8..)
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))
    for j, col in enumerate(["g6", "g7"]):
        x = np.arange(len(DIRS)); wds = 0.8 / max(1, len(runs))
        for i, r in enumerate(runs):
            vals = [num(df[(df.run == r) & (df.branch == b)][col]).mean() for b in DIRS]
            ax[j].bar(x + i * wds, vals, wds, label=r)
        ax[j].axhline(0, color="k", lw=.6); ax[j].set_xticks(x + 0.4); ax[j].set_xticklabels(DIRS)
        ax[j].set_title(["ROLL (PC6) signed mean", "PITCH (PC7) signed mean"][j]); ax[j].grid(alpha=.3)
    ax[0].legend(fontsize=8)
    fig.suptitle("Phase A — uncommanded roll/pitch by direction, all models (pca2 vs pca4 vs pca8 vs ...)", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{OUT}/phaseA_rollpitch_compare.png", dpi=120); plt.close(fig)

    print("Phase A analysis written to", OUT)
    print(f"collapse thresholds: musiq<{MUS_LO:.1f} or niqe>{NIQ_HI:.2f}")
    print(q.to_string(index=False))


if __name__ == "__main__":
    main()
