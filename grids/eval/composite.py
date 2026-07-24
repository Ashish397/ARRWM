"""Learned composite metrics with leave-one-grid-out (LOGO) cross-validation.

- Quality: ridge regression over z-scored metric features -> predicted quality.
  Reports LOGO-CV Spearman (global + mean within-grid) so the number is honest.
- Style shift: logistic regression -> LOGO-CV AUC.
- Also greedy forward selection of up to k features to find a small, robust recipe.
"""
import json, os, glob, itertools
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    gt = json.load(open(os.path.join(HERE, "gt.json")))["grids"]
    gt_rows = [{"grid": g, "variant": v, **d} for g, vs in gt.items() for v, d in vs.items()]
    gtdf = pd.DataFrame(gt_rows).drop(columns=["note"])
    m = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(HERE, "results_*.csv"))], ignore_index=True)
    wide = m.pivot_table(index=["grid", "variant"], columns="metric", values="value").reset_index()
    df = gtdf.merge(wide, on=["grid", "variant"])
    metrics = [c for c in wide.columns if c not in ("grid", "variant")]
    metrics = [c for c in metrics if not df[c].isna().any() and df[c].std() > 0]
    return df, metrics


def zscore_fit(X):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    return mu, sd


def ridge_fit(X, y, lam=1.0):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    A[-1, -1] -= lam  # don't regularize bias
    return np.linalg.solve(A, Xb.T @ y)


def ridge_pred(X, w):
    return np.hstack([X, np.ones((len(X), 1))]) @ w


def logo_cv_quality(df, feats, lam=1.0):
    preds = np.zeros(len(df))
    for g in df.grid.unique():
        tr, te = df.grid != g, df.grid == g
        Xtr = df.loc[tr, feats].values
        mu, sd = zscore_fit(Xtr)
        w = ridge_fit((Xtr - mu) / sd, df.loc[tr, "quality"].values, lam)
        preds[te.values] = ridge_pred((df.loc[te, feats].values - mu) / sd, w)
    rho_g = stats.spearmanr(preds, df.quality).statistic
    within = []
    for g, sub in df.assign(pred=preds).groupby("grid"):
        if sub.pred.nunique() > 1 and sub.quality.nunique() > 1:
            within.append(stats.spearmanr(sub.pred, sub.quality).statistic)
    return rho_g, float(np.mean(within)), preds


def auc(labels, scores):
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def logo_cv_style(df, feats, lam=1.0):
    """Ridge on the binary label works fine as a scorer; AUC only needs ranking."""
    preds = np.zeros(len(df))
    for g in df.grid.unique():
        tr, te = df.grid != g, df.grid == g
        Xtr = df.loc[tr, feats].values
        mu, sd = zscore_fit(Xtr)
        w = ridge_fit((Xtr - mu) / sd, df.loc[tr, "style_shift"].values.astype(float), lam)
        preds[te.values] = ridge_pred((df.loc[te, feats].values - mu) / sd, w)
    return auc(df.style_shift.values, preds), preds


def greedy(df, metrics, eval_fn, k=5):
    chosen, best_val = [], -np.inf
    history = []
    for _ in range(k):
        cand_best, cand_val = None, best_val
        for m in metrics:
            if m in chosen:
                continue
            v = eval_fn(df, chosen + [m])
            if v > cand_val:
                cand_best, cand_val = m, v
        if cand_best is None:
            break
        chosen.append(cand_best)
        best_val = cand_val
        history.append((list(chosen), best_val))
    return history


def main():
    df, metrics = load()
    print(f"{len(df)} videos, {len(metrics)} usable metrics")

    # add grid-median-relative versions of every metric (content-controlled)
    rel_feats = []
    for m in metrics:
        rm = f"rel__{m}"
        df[rm] = df.groupby("grid")[m].transform(lambda s: s - s.median())
        rel_feats.append(rm)
    allfeats = metrics + rel_feats

    print("\n--- QUALITY composite (LOGO-CV ridge, all features) ---")
    rho_g, rho_w, _ = logo_cv_quality(df, allfeats)
    print(f"all {len(allfeats)} feats: global={rho_g:+.3f} within={rho_w:+.3f}")

    print("\n--- QUALITY greedy forward selection (objective: mean of global+within) ---")
    def qobj(d, feats):
        g, w, _ = logo_cv_quality(d, feats)
        return (g + w) / 2
    for feats, val in greedy(df, allfeats, qobj, k=5):
        g, w, _ = logo_cv_quality(df, feats)
        print(f"k={len(feats)}: {feats} -> global={g:+.3f} within={w:+.3f}")

    print("\n--- STYLE composite (LOGO-CV, all features) ---")
    a, _ = logo_cv_style(df, allfeats)
    print(f"all feats AUC={a:.3f}")

    print("\n--- STYLE greedy forward selection ---")
    def sobj(d, feats):
        return logo_cv_style(d, feats)[0]
    for feats, val in greedy(df, allfeats, sobj, k=4):
        print(f"k={len(feats)}: {feats} -> AUC={val:.3f}")


if __name__ == "__main__":
    main()
