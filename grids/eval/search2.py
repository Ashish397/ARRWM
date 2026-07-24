"""Expanded combination search for the human-eye quality surrogate.

- Objective: mean within-grid Spearman under LOGO-CV (the "same ordering as the
  human eye on each grid" criterion), with global Spearman reported alongside.
- Greedy forward selection to k=8 over raw + grid-relative features.
- Exhaustive pairs and triples over the top-30 single features.
- For the best combo: per-grid rho, all misordered pairs (GT gap >= 1) and
  pairwise accuracy.
"""
import itertools, json, sys
import numpy as np
import pandas as pd
from scipy import stats
from composite import load, logo_cv_quality, greedy

df, metrics = load()
for m in metrics:
    df[f"rel__{m}"] = df.groupby("grid")[m].transform(lambda s: s - s.median())
allfeats = metrics + [f"rel__{m}" for m in metrics]
print(f"{len(df)} videos, {len(allfeats)} features")


def qobj(d, feats):
    g, w, _ = logo_cv_quality(d, feats)
    return w  # pure within-grid objective


def pairwise_acc(d, preds, min_gap=1.0):
    correct = total = 0
    bad = []
    for g, sub in d.assign(pred=preds).groupby("grid"):
        rows = sub.to_dict("records")
        for a, b in itertools.combinations(rows, 2):
            if abs(a["quality"] - b["quality"]) < min_gap:
                continue
            total += 1
            hi, lo = (a, b) if a["quality"] > b["quality"] else (b, a)
            if hi["pred"] > lo["pred"]:
                correct += 1
            else:
                bad.append((g, hi["variant"], lo["variant"], hi["quality"], lo["quality"]))
    return correct / total, bad


# single-feature ranking by within-grid rho
singles = []
for f in allfeats:
    per = []
    for g, sub in df.groupby("grid"):
        if sub[f].nunique() > 1 and sub.quality.nunique() > 1:
            per.append(stats.spearmanr(sub[f], sub.quality).statistic)
    singles.append((f, np.mean(per)))
singles.sort(key=lambda t: abs(t[1]), reverse=True)
print("\ntop-15 singles (within-grid rho):")
for f, r in singles[:15]:
    print(f"  {f:35s} {r:+.3f}")
top30 = [f for f, _ in singles[:30]]

print("\nexhaustive pairs over top-30:")
best_pairs = []
for combo in itertools.combinations(top30, 2):
    g, w, _ = logo_cv_quality(df, list(combo))
    best_pairs.append((w, g, combo))
best_pairs.sort(reverse=True)
for w, g, c in best_pairs[:5]:
    print(f"  within={w:+.3f} global={g:+.3f}  {c}")

print("\nexhaustive triples over top-20:")
top20 = [f for f, _ in singles[:20]]
best_triples = []
for combo in itertools.combinations(top20, 3):
    g, w, _ = logo_cv_quality(df, list(combo))
    best_triples.append((w, g, combo))
best_triples.sort(reverse=True)
for w, g, c in best_triples[:5]:
    print(f"  within={w:+.3f} global={g:+.3f}  {c}")

print("\ngreedy to k=8 (objective: within-grid):")
hist = greedy(df, allfeats, qobj, k=8)
for feats, val in hist:
    g, w, _ = logo_cv_quality(df, feats)
    print(f"  k={len(feats)}: within={w:+.3f} global={g:+.3f}  + {feats[-1]}")

best_feats = max(
    [list(hist[-1][0])] + [list(c) for _, _, c in best_triples[:1]] + [list(c) for _, _, c in best_pairs[:1]],
    key=lambda f: logo_cv_quality(df, f)[1],
)
g, w, preds = logo_cv_quality(df, best_feats)
print(f"\nBEST: {best_feats}\n global={g:+.3f} within={w:+.3f}")
acc, bad = pairwise_acc(df, preds)
print(f" pairwise accuracy (GT gap>=1): {acc:.1%}")
print(" remaining misordered pairs:")
for b in bad:
    print(f"  {b[0]}: {b[1]} (GT {b[3]}) should beat {b[2]} (GT {b[4]})")
for gname, sub in df.assign(pred=preds).groupby("grid"):
    print(f" {gname}: rho={stats.spearmanr(sub.pred, sub.quality).statistic:+.2f}")
json.dump({"features": best_feats}, open("best_combo.json", "w"), indent=1)
