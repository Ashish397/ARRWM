"""Family-superimposed versions of following_{run}_by_REALdir_strength.png.

Same data and panel layout as utils/following_by_realdir.py (teacher-forced
control tests over training steps, realized magnitude along the command per
REAL ride direction), but several models drawn in the SAME figure, one color
per model: GT branch solid, FLIP branch dashed, gray dotted = commanded target.

Writes analysis/following_FAMILY_{nodes,encoders}_by_REALdir_strength.png.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figures.following_by_realdir import build, DIRS, DNAME, CHUNK_SEC

FAMILIES = {
    "nodes": [("logs/v14e_4node/control_test", "batch size 16", "#4393c3"),
              ("logs/v14e_pca8_raw/control_test", "batch size 32", "#08306b"),
              ("logs/v14e_16node/control_test", "batch size 64", "#e08214")],
    "encoders": [("logs/v14e_pca8_raw/control_test", "pca8", "#08306b"),
                 ("logs/v14e_pca4/control_test", "pca4", "#2ca02c"),
                 ("logs/v14e_pca2/control_test", "pca2", "#9467bd")],
    # injection-pathway ablation: pca8/8node = adaln+tokens (full), noatok =
    # adaln-only, noadaln = tokens-only
    "injection": [("logs/v14e_pca8_raw/control_test", "pca8 (batch size 32, adaln+tokens)", "#08306b"),
                  ("logs/v14e_noatok/control_test", "no action tokens (adaln-only)", "#2ca02c"),
                  ("logs/v14e_noadaln/control_test", "no AdaLN (tokens-only)", "#b2182b")],
}


def plot_family(name, members):
    dfs = {}
    for cdir, label, col in members:
        df = build(cdir)
        if len(df):
            dfs[label] = (df, col)
        else:
            print(f"[fam] {label}: no metrics in {cdir}")
    fig, ax = plt.subplots(2, 4, figsize=(22, 10))
    for i, D in enumerate(DIRS):
        a = ax[i // 4][i % 4]
        # commanded target from the pooled family data
        pool = pd.concat([df[df["dir"] == D] for df, _ in dfs.values()])
        cref = pool.groupby("step")["commanded"].mean().sort_index()
        if len(cref):
            a.plot(cref.index, cref.rolling(9, center=True, min_periods=3).mean().values,
                   ":", color="gray", lw=1.5, label="commanded |·| (target)" if i == 0 else None)
        for label, (df, col) in dfs.items():
            sub = df[(df["dir"] == D) & (df.branch == "flip")]
            g = sub.groupby("step")["realized"].mean().sort_index()
            if len(g):
                a.plot(g.index, g.rolling(9, center=True, min_periods=3).mean().values,
                       "-", color=col, lw=2.5, label=(label if i == 0 else None))
        a.axhline(0, color="gray", lw=0.6)
        a.set_ylim(-0.5, 1.15)
        nD = sum(len(df[(df["dir"] == D) & (df.branch == "flip")]) for df, _ in dfs.values())
        a.set_title(f"{DNAME[D]} ({nD * CHUNK_SEC / 60:.0f} min)", fontsize=17)
        a.set_xlabel("step", fontsize=16)
        if i % 4 == 0:
            a.set_ylabel("strength of followed action", fontsize=16)
        a.tick_params(labelsize=14)
        a.grid(alpha=0.3)
        if i == 0:
            a.legend(fontsize=13, ncol=2)
    fig.tight_layout()
    out = f"analysis/following_FAMILY_{name}_by_REALdir_strength.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[fam] saved {out}")


def plot_family_direction(name, members):
    """Cosine-agreement (non-strength) view, FLIP branch, one line per model."""
    dfs = {}
    for cdir, label, col in members:
        df = build(cdir)
        if len(df):
            dfs[label] = (df, col)
    fig, ax = plt.subplots(2, 4, figsize=(22, 10))
    for i, D in enumerate(DIRS):
        a = ax[i // 4][i % 4]
        for label, (df, col) in dfs.items():
            sub = df[(df["dir"] == D) & (df.branch == "flip")]
            g = sub.groupby("step")["cos"].mean().sort_index()
            if len(g):
                a.plot(g.index, g.rolling(9, center=True, min_periods=3).mean().values,
                       "-", color=col, lw=2.5, label=(label if i == 0 else None))
        a.axhline(0, color="gray", lw=0.6)
        a.set_ylim(-1.05, 1.05)
        nD = sum(len(df[(df["dir"] == D) & (df.branch == "flip")]) for df, _ in dfs.values())
        a.set_title(f"{DNAME[D]} ({nD * CHUNK_SEC / 60:.0f} min)", fontsize=17)
        a.set_xlabel("step", fontsize=16)
        if i % 4 == 0:
            a.set_ylabel("direction agreement (cosine)", fontsize=16)
        a.tick_params(labelsize=14)
        a.grid(alpha=0.3)
        if i == 0:
            a.legend(fontsize=13)
    fig.tight_layout()
    out = f"analysis/following_FAMILY_{name}_by_REALdir.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[fam] saved {out}")


def plot_family_direction_halved(name, members):
    """Forward + Left only, stacked vertically (top/bottom). Same cosine view."""
    dfs = {}
    for cdir, label, col in members:
        df = build(cdir)
        if len(df):
            dfs[label] = (df, col)
    fig, ax = plt.subplots(2, 1, figsize=(5.5, 10))
    for i, D in enumerate(["F", "L"]):
        a = ax[i]
        for label, (df, col) in dfs.items():
            sub = df[(df["dir"] == D) & (df.branch == "flip")]
            g = sub.groupby("step")["cos"].mean().sort_index()
            if len(g):
                a.plot(g.index, g.rolling(9, center=True, min_periods=3).mean().values,
                       "-", color=col, lw=2.5, label=(label if i == 0 else None))
        a.axhline(0, color="gray", lw=0.6)
        a.set_ylim(-1.05, 1.05)
        nD = sum(len(df[(df["dir"] == D) & (df.branch == "flip")]) for df, _ in dfs.values())
        a.set_title(f"{DNAME[D]} ({nD * CHUNK_SEC / 60:.0f} min)", fontsize=17)
        a.set_xlabel("step", fontsize=16)
        a.set_ylabel("direction agreement (cosine)", fontsize=16)
        a.tick_params(labelsize=14)
        a.grid(alpha=0.3)
        if i == 0:
            a.legend(fontsize=13)
    fig.tight_layout()
    out = f"analysis/following_FAMILY_{name}_by_REALdir_halved.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[fam] saved {out}")


if __name__ == "__main__":
    import os
    if os.environ.get("FAM_HALVED_ONLY"):
        for name in ("nodes", "encoders"):
            plot_family_direction_halved(name, FAMILIES[name])
    else:
        for name, members in FAMILIES.items():
            plot_family(name, members)
            plot_family_direction(name, members)
        for name in ("nodes", "encoders"):
            plot_family_direction_halved(name, FAMILIES[name])
