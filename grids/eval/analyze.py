"""Correlate metric battery outputs against human ground truth (gt.json).

- Quality metrics: global + within-grid Spearman vs GT quality score.
- Style metrics: ROC-AUC vs GT style_shift flag, raw and grid-median-relative.
"""
import json, os, glob
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    gt = json.load(open(os.path.join(HERE, "gt.json")))["grids"]
    gt_rows = []
    for grid, variants in gt.items():
        for v, d in variants.items():
            gt_rows.append({"grid": grid, "variant": v, **{k: d[k] for k in ("quality", "style_shift", "action_follow")}})
    gtdf = pd.DataFrame(gt_rows)

    dfs = []
    for f in glob.glob(os.path.join(HERE, "results_*.csv")):
        dfs.append(pd.read_csv(f))
    m = pd.concat(dfs, ignore_index=True)
    return gtdf, m


def auc(labels, scores):
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main():
    gtdf, m = load()
    wide = m.pivot_table(index=["grid", "variant"], columns="metric", values="value").reset_index()
    df = gtdf.merge(wide, on=["grid", "variant"])
    metrics = [c for c in wide.columns if c not in ("grid", "variant")]

    print(f"\n{len(df)} scored videos, {len(metrics)} metrics\n")
    print("=== QUALITY: Spearman vs GT quality (global | mean within-grid | median-rel global) ===")
    qrows = []
    for met in metrics:
        x = df[met].values
        if np.isnan(x).any():
            continue
        rho_g = stats.spearmanr(x, df.quality).statistic
        per_grid = []
        for g, sub in df.groupby("grid"):
            if sub[met].nunique() > 1 and sub.quality.nunique() > 1:
                per_grid.append(stats.spearmanr(sub[met], sub.quality).statistic)
        rho_w = np.mean(per_grid) if per_grid else np.nan
        rel = df.groupby("grid")[met].transform(lambda s: s - s.median())
        rho_rel = stats.spearmanr(rel, df.quality).statistic
        qrows.append({"metric": met, "spearman_global": rho_g, "spearman_within": rho_w, "spearman_medrel": rho_rel})
    q = pd.DataFrame(qrows).sort_values("spearman_within", key=abs, ascending=False)
    print(q.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    print("\n=== STYLE SHIFT: AUC vs GT style_shift flag (raw | grid-median-relative) ===")
    srows = []
    labels = df.style_shift.values
    for met in metrics:
        x = df[met].values
        if np.isnan(x).any():
            continue
        rel = df.groupby("grid")[met].transform(lambda s: s - s.median()).values
        srows.append({"metric": met,
                      "auc_raw": auc(labels, x), "auc_raw_neg": auc(labels, -x),
                      "auc_rel": auc(labels, rel), "auc_rel_neg": auc(labels, -rel)})
    s = pd.DataFrame(srows)
    s["best_auc"] = s[["auc_raw", "auc_raw_neg", "auc_rel", "auc_rel_neg"]].max(1)
    s = s.sort_values("best_auc", ascending=False)
    print(s.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    df.to_csv(os.path.join(HERE, "merged_scores.csv"), index=False)
    q.to_csv(os.path.join(HERE, "corr_quality.csv"), index=False)
    s.to_csv(os.path.join(HERE, "corr_style.csv"), index=False)
    print("\nwrote merged_scores.csv, corr_quality.csv, corr_style.csv")


if __name__ == "__main__":
    main()
