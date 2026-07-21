"""Bar charts for the 4 adopted VLM metric axes, ALL models (ours + baselines).

Sources:
  ours        analysis/eval_final/suite_all.csv   (pal, qwen, spawn_jump,
                                                    center_nov, vlisa columns)
  comparators suiteA_comp*.csv (pal_mean, qwen_top4, spawn_jump, center_nov)
              + vlisa_comp*.csv (upper_mean = band melt fraction, topcut 0.25)

Writes to analysis/eval_final/wedges/:
  bars_vlm_suite_all.png            5 panels (vlisa, pal, qwen, spawn_jump,
                                    center_nov), mean +- sem per model
  bars_{vlisa,pal,qwen}_by_dir_all.png   per-direction grouped bars
Skips comparators whose CSVs have not landed yet (prints what's missing).
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
EF = f"{ARR}/analysis/eval_final"
OUT = f"{EF}/wedges"
DIRS = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]
ORDER = ["pca8_8node", "16node", "pca4", "pca2", "4node", "noatok", "noadaln",
         "minwm", "matrixgame", "worldcam", "yume", "worldplay", "astra"]
LABEL = {"pca8_8node": "ours pca8", "16node": "ours 16node"}
AXES = ["vlisa", "pal", "qwen", "spawn_jump", "center_nov"]
TITLES = {"vlisa": "VideoLISA melt band frac", "pal": "PAL4VST warp frac",
          "qwen": "Qwen P(melt)", "spawn_jump": "DINO novelty jump",
          "center_nov": "DINO center novelty"}


def load():
    ours = pd.read_csv(f"{EF}/suite_all.csv")
    parts = [ours[["run", "window", "dir", "pal", "qwen", "spawn_jump", "center_nov", "vlisa"]]]
    comp = []
    for f in sorted(glob.glob(f"{EF}/suiteA_comp*.csv")):
        comp.append(pd.read_csv(f))
    if comp:
        sc = pd.concat(comp, ignore_index=True).rename(
            columns={"pal_mean": "pal", "qwen_top4": "qwen"})
        vl = []
        for f in sorted(glob.glob(f"{EF}/vlisa_comp*.csv")):
            vl.append(pd.read_csv(f))
        if vl:
            v = pd.concat(vl, ignore_index=True)
            wd = v.window.str.extract(r"r(\d+)_(\w+)")
            v = v.assign(window="r" + wd[0], dir=wd[1]).rename(
                columns={"upper_mean": "vlisa"})[["run", "window", "dir", "vlisa"]]
            sc = sc.merge(v, on=["run", "window", "dir"], how="left")
        else:
            sc["vlisa"] = np.nan
        parts.append(sc[["run", "window", "dir", "pal", "qwen", "spawn_jump", "center_nov", "vlisa"]])
    df = pd.concat(parts, ignore_index=True)
    have = df.groupby("run").size()
    print("[bars] rows per model:", dict(have), flush=True)
    return df


def overall(df, models):
    fig, axes = plt.subplots(1, 5, figsize=(26, 5.5))
    for ax, met in zip(axes, AXES):
        mu = [df[df.run == m][met].mean() for m in models]
        se = [df[df.run == m][met].sem() for m in models]
        cols = ["#2166ac" if m in ("pca8_8node", "16node") else
                ("#92c5de" if m in ORDER[:7] else "#b2182b") for m in models]
        ax.bar(range(len(models)), mu, yerr=se, color=cols, alpha=0.9, capsize=2)
        ax.set_xticks(range(len(models)), [LABEL.get(m, m) for m in models],
                      rotation=40, ha="right", fontsize=8)
        ax.set_title(TITLES[met], fontsize=11)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("VLM metric suite, all models (lower = cleaner; dark blue = our best two, "
                 "light blue = our ablations, red = external)", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/bars_vlm_suite_all.png", dpi=120)
    plt.close(fig)
    print("[bars] bars_vlm_suite_all.png", flush=True)


def by_dir(df, models, met):
    sub = df.dropna(subset=[met])
    ms = [m for m in models if len(sub[sub.run == m])]
    fig, ax = plt.subplots(figsize=(20, 6.5))
    w = 0.8 / max(len(ms), 1)
    cmap = plt.get_cmap("tab20")
    for k, m in enumerate(ms):
        mu = [sub[(sub.run == m) & (sub["dir"] == d)][met].mean() for d in DIRS]
        ax.bar(np.arange(8) + (k - len(ms) / 2 + 0.5) * w, mu, w,
               label=LABEL.get(m, m), color=cmap(k % 20), alpha=0.9)
    ax.set_xticks(range(8), DIRS)
    ax.set_title(f"{TITLES[met]} by commanded direction, all models")
    ax.legend(ncol=7, fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{OUT}/bars_{met}_by_dir_all.png", dpi=120)
    plt.close(fig)
    print(f"[bars] bars_{met}_by_dir_all.png", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    df = load()
    models = [m for m in ORDER if m in set(df.run)]
    overall(df, models)
    for met in ("vlisa", "pal", "qwen"):
        by_dir(df, models, met)
