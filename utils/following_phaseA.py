"""Phase-A action-following graphs, ONE FORMAT FOR ALL MODELS (ours + baselines).

The original following_{run}_by_REALdir.png graphs come from teacher-forced
control tests on real rides (GT/FLIP branches over training steps) — external
baselines cannot have those. This is the directly-comparable equivalent on the
SAME phase-A eval videos and the SAME frozen CoTracker→PCA teacher readout
for every model (no trained critic anywhere in this pipeline):

Per commanded direction (8 compass panels):
  - cos agreement between commanded [thr, steer] and teacher-read [g0, g1]
    (distribution over the 32 windows: mean bar + per-video dots)
  - realized magnitude along the command (c.t/|c|, sign-preserving) with the
    commanded magnitude 0.5 as the target line

Writes analysis/following_{model}_phaseA.png per model +
analysis/following_ALL_phaseA_summary.png (models x dirs heatmap of cos).

Source: headtohead_motion.csv + headtohead_{model}.csv. minwm rows are already
TRUE-direction. Env: FP_OUT (default analysis/).
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
EF = f"{ARR}/analysis/eval_final"
OUT = os.environ.get("FP_OUT", f"{ARR}/analysis")
DIRS = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]
M = 0.5; Dv = M / np.sqrt(2)
CMD = {"F": (M, 0.0), "FR": (Dv, Dv), "R": (0.0, M), "BR": (-Dv, Dv),
       "B": (-M, 0.0), "BL": (-Dv, -Dv), "L": (0.0, -M), "FL": (Dv, -Dv)}   # (thr, steer)
ORDER = ["pca8_8node", "16node", "pca4", "pca2", "4node", "noatok", "noadaln",
         "minwm", "matrixgame", "worldcam", "yume", "worldplay", "astra"]
LABEL = {"pca8_8node": "ours pca8 (8node)", "16node": "ours 16node"}


def load_all():
    frames = [pd.read_csv(f"{EF}/headtohead_motion.csv")]
    for f in sorted(glob.glob(f"{EF}/headtohead_*.csv")):
        if os.path.basename(f) != "headtohead_motion.csv":
            frames.append(pd.read_csv(f))
    df = pd.concat(frames, ignore_index=True)
    c = np.array([CMD[d] for d in df["dir"]])
    t = df[["g0", "g1"]].to_numpy(float)
    cn = np.linalg.norm(c, axis=1); tn = np.linalg.norm(t, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        df["cos"] = (c * t).sum(1) / np.where(cn * tn > 1e-6, cn * tn, np.nan)
        df["realized"] = (c * t).sum(1) / np.where(cn > 1e-6, cn, np.nan)
    return df


def per_model(df, m):
    sub = df[df.model == m]
    if not len(sub):
        return
    fig, ax = plt.subplots(2, 4, figsize=(22, 10))
    for i, D in enumerate(DIRS):
        a = ax[i // 4][i % 4]
        s = sub[sub["dir"] == D].dropna(subset=["cos"])
        a.axhline(0, color="gray", lw=0.6)
        a.axhline(0.5, color="gray", ls=":", lw=1.5, label="commanded |·| (target)")
        x = np.random.default_rng(0).uniform(-0.13, 0.13, len(s))
        a.scatter(0 + x, s["cos"], s=14, color="C0", alpha=0.5)
        a.bar([0], [s["cos"].mean()], width=0.42, color="C0", alpha=0.35)
        a.scatter(1 + x, s["realized"], s=14, color="crimson", alpha=0.5)
        a.bar([1], [s["realized"].mean()], width=0.42, color="crimson", alpha=0.35)
        a.set_xticks([0, 1], ["cos agreement", "realized |·|"])
        a.set_ylim(-1.15, 1.15)
        # headtohead rows are whole windows (6s of generation each), not the
        # 0.75s chunks used in the by-REALdir curves
        a.set_title(f"commanded {D} ({len(s) * 6.0 / 60:.0f} min)  "
                    f"cos={s['cos'].mean():.2f}  real={s['realized'].mean():.2f}")
        a.grid(alpha=0.3)
        if i == 0:
            a.legend(fontsize=9)
    fig.suptitle(f"{LABEL.get(m, m)} — phase-A following by COMMANDED direction "
                 f"(teacher-read [g0,g1], CoTracker→PCA, vs command; dots = windows)", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/following_{m}_phaseA.png", dpi=110)
    plt.close(fig)
    print(f"[followA] following_{m}_phaseA.png", flush=True)


def summary(df, models):
    Cc = np.full((len(models), 8), np.nan)
    Rr = np.full((len(models), 8), np.nan)
    for i, m in enumerate(models):
        for j, D in enumerate(DIRS):
            s = df[(df.model == m) & (df["dir"] == D)]
            Cc[i, j] = s["cos"].mean(); Rr[i, j] = s["realized"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(19, 0.55 * len(models) + 3))
    for ax, Mx, t, vmin, vmax in [(axes[0], Cc, "cos agreement (cmd vs teacher-read)", -1, 1),
                                  (axes[1], Rr, "realized magnitude (target 0.5)", -0.6, 0.6)]:
        im = ax.imshow(Mx, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(8), DIRS)
        ax.set_yticks(range(len(models)), [LABEL.get(m, m) for m in models])
        for i in range(len(models)):
            for j in range(8):
                if np.isfinite(Mx[i, j]):
                    ax.text(j, i, f"{Mx[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title(t)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Phase-A following, all models x commanded direction", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/following_ALL_phaseA_summary.png", dpi=120)
    plt.close(fig)
    print(f"[followA] following_ALL_phaseA_summary.png", flush=True)


FAM_COLORS = ["#08306b", "#2166ac", "#4393c3", "#92c5de"]


def family(df, fam_models, fam_labels, name):
    """Superimpose several models in one 8-panel figure: per commanded dir,
    grouped bars per model — solid = cos agreement, hatched = realized |·|."""
    fig, ax = plt.subplots(2, 4, figsize=(22, 10))
    n = len(fam_models)
    for i, D in enumerate(DIRS):
        a = ax[i // 4][i % 4]
        a.axhline(0, color="gray", lw=0.6)
        a.axhline(0.5, color="gray", ls=":", lw=1.5)
        for k, m in enumerate(fam_models):
            s = df[(df.model == m) & (df["dir"] == D)].dropna(subset=["cos"])
            if not len(s):
                continue
            a.bar(k - 0.19, s["cos"].mean(), 0.36, yerr=s["cos"].sem(), capsize=2,
                  color=FAM_COLORS[k % len(FAM_COLORS)], alpha=0.95,
                  label=fam_labels[k] if i == 0 else None)
            a.bar(k + 0.19, s["realized"].mean(), 0.36, yerr=s["realized"].sem(), capsize=2,
                  color=FAM_COLORS[k % len(FAM_COLORS)], alpha=0.55, hatch="//")
        a.set_xticks(range(n), fam_labels, fontsize=8)
        a.set_ylim(-1.05, 1.15)
        a.set_title(f"commanded {D}")
        a.grid(alpha=0.3, axis="y")
        if i == 0:
            a.legend(fontsize=9)
    fig.suptitle(f"Phase-A following, {name} family — solid = cos agreement, "
                 f"hatched = realized magnitude (target 0.5, dotted line)", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/following_FAMILY_{name}_phaseA.png", dpi=110)
    plt.close(fig)
    print(f"[followA] following_FAMILY_{name}_phaseA.png", flush=True)


if __name__ == "__main__":
    df = load_all()
    models = [m for m in ORDER if m in set(df.model)]
    for m in models:
        per_model(df, m)
    summary(df, models)
    family(df, ["4node", "pca8_8node", "16node"],
           ["4node", "8node", "16node"], "nodes")
    family(df, ["pca8_8node", "pca4", "pca2", "noatok"],
           ["pca8", "pca4", "pca2", "noatok"], "encoders")
