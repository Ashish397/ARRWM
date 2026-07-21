"""Smoothed IQA-vs-step comparison graphs (no-reference video quality).

Three comparisons, one figure each (musiq / niqe / brisque as subplots, runs overlaid,
rolling-smoothed over step, gt+flip pooled):
  1. PCA dims :  8pca (pca8_8node) vs pca4 vs pca2      -> iqa_cmp_pca.png
  2. Tokens   :  8node8pca (pca8_8node) vs noatok       -> iqa_cmp_tokens.png
  3. Nodes/bs :  4node vs 8node (pca8_8node) vs 16node  -> iqa_cmp_nodes.png

musiq: higher = better.  niqe / brisque: lower = better.
"""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

CSV = "analysis/chunk_metrics.csv"   # unified per-chunk source (musiq/niqe/brisque per chunk)
METRICS = [("musiq", "MUSIQ  (higher = better)", True),
           ("niqe", "NIQE  (lower = better)", False),
           ("brisque", "BRISQUE  (lower = better)", False)]
WIN = 7  # rolling window (in stride steps)

GROUPS = [
    ("iqa_cmp_pca.png", "IQA vs step — PCA dimensionality (8-node)",
     [("pca8_8node", "8 PCA", "C0"), ("pca4", "4 PCA", "C1"), ("pca2", "2 PCA", "C3")]),
    ("iqa_cmp_tokens.png", "IQA vs step — action tokens ON vs OFF (8-node, 8 PCA)",
     [("pca8_8node", "tokens ON (8node8pca)", "C0"), ("noatok", "tokens OFF (noatok)", "C5")]),
    ("iqa_cmp_nodes.png", "IQA vs step — nodes / batch size",
     [("4node", "4node (b16)", "C2"), ("pca8_8node", "8node (b32)", "C0"), ("16node", "16node (b64)", "C4")]),
]


def curve(d, metric):
    g = d.groupby("step")[metric].mean().sort_index()
    return g.index.values, g.rolling(WIN, center=True, min_periods=2).mean().values


def main():
    df = pd.read_csv(CSV)
    for out, title, runs in GROUPS:
        fig, ax = plt.subplots(1, 3, figsize=(21, 6.2))
        for mi, (metric, mlabel, _) in enumerate(METRICS):
            a = ax[mi]
            for run, lab, col in runs:
                d = df[df.run == run].dropna(subset=[metric])
                if not len(d):
                    continue
                x, y = curve(d, metric)
                a.plot(x, y, color=col, lw=2.2, label=lab)
            a.set_title(mlabel); a.set_xlabel("step"); a.grid(alpha=.3)
            a.legend(fontsize=9)
        fig.suptitle(f"{title}  (rolling-{WIN} smoothed, gt+flip pooled)", fontsize=13)
        fig.tight_layout(); fig.savefig(f"analysis/{out}", dpi=120); plt.close(fig)
        print(f"saved analysis/{out}")


if __name__ == "__main__":
    main()
