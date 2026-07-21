"""Phase B (hold-still) analysis. Seeds are NON-moving unseen windows binned by
scene-change (small/medium/big/massive actor activity); the model is commanded
zero action and must hold the scene still. We measure how rendering quality and
fidelity to the real static continuation degrade as scene activity rises.

Reads the chunk_metrics CSV (branch='static') + phaseB_windows.json (the bin per
ride+offset). Produces, per model, IQA/MAE/LPIPS vs scene-change bin.
  -> phaseB_quality.png  (MUSIQ / NIQE / MAE / LPIPS  vs bin, runs overlaid)
  -> phaseB_quality.csv
"""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

CSV = os.environ.get("PB_CSV", "analysis/eval_final/phaseB_metrics.csv")
WIN = "analysis/eval_final/phaseB_windows.json"
OUT = "analysis/eval_final"
RUNS = ["pca8_8node", "pca4", "pca2", "16node", "4node", "noatok", "noadaln"]
BINS = ["small", "medium", "big", "massive"]


def num(s): return pd.to_numeric(s, errors="coerce")


def main():
    df = pd.read_csv(CSV)
    wins = json.load(open(WIN))
    binmap = {(w["ride"], int(w["offset"])): w["bin"] for w in wins}
    df["bin"] = [binmap.get((r, int(o)), None) for r, o in zip(df.ride, df.offset)]
    df = df[df["bin"].notna()]
    runs = [r for r in RUNS if r in set(df.run.unique())]

    METRICS = [("musiq", "MUSIQ (higher=better)", False), ("niqe", "NIQE (lower=better)", True),
               ("mae", "MAE vs real (lower=better)", True), ("lpips", "LPIPS vs real (lower=better)", True)]
    rows = []
    for r in runs:
        for b in BINS:
            d = df[(df.run == r) & (df["bin"] == b)]
            if not len(d):
                continue
            rows.append(dict(run=r, bin=b, n=len(d),
                             musiq=round(num(d.musiq).mean(), 2), niqe=round(num(d.niqe).mean(), 3),
                             brisque=round(num(d.brisque).mean(), 2),
                             mae=round(num(d.mae).mean(), 4), lpips=round(num(d.lpips).mean(), 4)))
    q = pd.DataFrame(rows); q.to_csv(f"{OUT}/phaseB_quality.csv", index=False)

    fig, ax = plt.subplots(1, 4, figsize=(24, 6))
    x = np.arange(len(BINS))
    for j, (metric, ttl, _) in enumerate(METRICS):
        for r in runs:
            d = q[q.run == r]
            y = [d[d.bin == b][metric].mean() if len(d[d.bin == b]) else np.nan for b in BINS]
            ax[j].plot(x, y, marker="o", lw=2, label=r)
        ax[j].set_xticks(x); ax[j].set_xticklabels(BINS, rotation=20)
        ax[j].set_title(ttl); ax[j].set_xlabel("scene-change bin"); ax[j].grid(alpha=.3)
        if j == 0:
            ax[j].legend(fontsize=8)
    fig.suptitle("Phase B — hold-still rendering quality vs scene activity (zero action)", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{OUT}/phaseB_quality.png", dpi=120); plt.close(fig)
    print("Phase B analysis written to", OUT)
    print(q.to_string(index=False))


if __name__ == "__main__":
    main()
