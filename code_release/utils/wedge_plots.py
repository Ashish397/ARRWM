"""Wedge (compass polar-bar) plots of per-direction motion metrics.

One panel per model, one wedge per commanded direction at its compass angle;
radius = |median metric| on a PER-PANEL linear scale (each panel's largest
wedge touches its own rim; the panel's absolute max is printed in its title).
Blue = positive, red = negative. 4 panels per row.

Two figure sets over the same metrics (recession, tx, g0, g1, rot, ty_pct,
tx_pct):
  wedge_all_{met}.png  main models + external baselines
                       (pca8_8node, 16node, minwm, matrixgame, worldcam,
                        yume, worldplay, astra)
  wedge_{met}.png      ablation set, ours only (no minwm)
                       (pca8_8node, pca4, pca2, 16node, 4node, noatok, noadaln)

Reads headtohead_motion.csv + headtohead_*.csv + ty_motion.csv.
Env: WG_OUT (default analysis/eval_final/wedges).
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARR = os.environ.get("AF_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EF = f"{ARR}/analysis/eval_final"
OUT = os.environ.get("WG_OUT", f"{EF}/wedges")
os.makedirs(OUT, exist_ok=True)

ANG = {"F": 90, "FR": 45, "R": 0, "BR": -45, "B": -90, "BL": -135, "L": 180, "FL": 135}
METRICS = ["recession", "tx", "g0", "g1", "rot", "g2", "g3", "g4", "g5", "g6", "g7"]
ALL_SET = [("pca8_8node", "Ours (Default)"), ("16node", "Ours (batch 64)"),
           ("minwm", "minWM"), ("matrixgame", "Matrix-Game"), ("worldcam", "WorldCam"),
           ("yume", "Yume"), ("worldplay", "WorldPlay"), ("astra", "Astra")]
# (run, display label); row 1 = encoder family + noatok, row 2 = node family
# + noadaln; pca8_8node appears in both rows as the shared reference model
ABL_SET = [("pca8_8node", "Default"), ("pca4", "PCA4"), ("pca2", "PCA2"),
           ("noatok", "No Action Tokens"),
           ("16node", "batch 64"), ("pca8_8node", "Default"), ("4node", "batch 16"),
           ("noadaln", "No AdaLN")]

frames = [pd.read_csv(f"{EF}/headtohead_motion.csv")]
for f in sorted(glob.glob(f"{EF}/headtohead_*.csv")):
    if os.path.basename(f) != "headtohead_motion.csv":
        frames.append(pd.read_csv(f))
df = pd.concat(frames, ignore_index=True)

# ty (vertical translation = pitch axis) + resolution-normalized tx, if extracted
if os.path.exists(f"{EF}/ty_motion.csv"):
    tydf = pd.read_csv(f"{EF}/ty_motion.csv")
    df = pd.concat([df, tydf[["model", "dir", "tx_pct", "ty_pct"]]], ignore_index=True)
    METRICS = METRICS + ["ty_pct", "tx_pct"]


def draw(met, models, fname, shared=False):
    """shared=True: one rim for the whole set (our ablations share units, so
    panels are directly comparable). shared=False: per-panel rim (baselines'
    command units differ; only shapes compare)."""
    n = len(models)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.9 * nrow),
                             subplot_kw={"projection": "polar"})
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    med = df.groupby(["model", "dir"])[met].median()
    if shared:
        pool = np.abs(pd.to_numeric(df[df.model.isin([m for m, _ in models])][met], errors="coerce").dropna().values)
        vshared = float(np.percentile(pool, 95)) if len(pool) else 1.0
    for ax, (m, label) in zip(axes, models):
        sub = df[df.model == m]
        allv = np.abs(pd.to_numeric(sub[met], errors="coerce").dropna().values)
        vmax = vshared if shared else (float(np.percentile(allv, 95)) if len(allv) else 1.0)
        for d in ANG:
            th = np.radians(ANG[d])
            vs = pd.to_numeric(sub[sub["dir"] == d][met], errors="coerce").dropna().values
            # one translucent wedge PER VIDEO: overlaps build radial density
            for v in vs:
                ax.bar(th, min(abs(v) / vmax, 1.0), width=np.radians(36), bottom=0,
                       color="#2166ac" if v >= 0 else "#b2182b", alpha=0.05,
                       linewidth=0, zorder=3)
            mv = med.get((m, d), np.nan)
            if np.isfinite(mv):    # median as an outlined wedge on top
                ax.bar(th, min(abs(mv) / vmax, 1.0), width=np.radians(36), bottom=0,
                       fill=False, edgecolor="#1a355e" if mv >= 0 else "#7a1024",
                       linewidth=1.4, zorder=4)
            if len(vs) >= 4:       # IQR: dashed arcs at the q25/q75 radii
                span = np.radians(np.linspace(ANG[d] - 14, ANG[d] + 14, 24))
                for q in (np.percentile(vs, 25), np.percentile(vs, 75)):
                    ax.plot(span, [min(abs(q) / vmax, 1.0)] * len(span), ls="--",
                            lw=0.9, color="#1a355e" if q >= 0 else "#7a1024",
                            zorder=5, solid_capstyle="butt")
            ax.text(th, 1.30, d, ha="center", va="center", fontsize=9)
        # labeled reference rings at 25/50/75/100% of the rim
        rings = [0.25, 0.5, 0.75, 1.0]
        ax.set_ylim(0, 1.0)
        ax.set_xticks([])
        ax.set_yticks(rings)
        ax.set_yticklabels([f"{r * vmax:.2g}" for r in rings], fontsize=6, color="#444")
        ax.set_rlabel_position(112.5)                  # labels between F and FL
        ax.grid(axis="y", alpha=0.5, lw=0.6, color="#999", zorder=1)
        ax.set_title(label if shared else f"{label}  (rim {vmax:.2g})", fontsize=11, pad=16)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}", dpi=200)
    plt.close(fig)
    print(f"[wedge] {fname} ({n} models)")


present = set(df.model)
for met in METRICS:
    draw(met, [(m, l) for m, l in ALL_SET if m in present], f"wedge_all_{met}.png", shared=False)
    draw(met, [(m, l) for m, l in ABL_SET if m in present], f"wedge_{met}.png", shared=True)
