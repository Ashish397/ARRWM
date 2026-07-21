"""Mechanical ground-truth checker for mangle-metric candidates.

Input: csv with columns (window, run, <score>) — higher = more mangled.
Checks (user ground truths):
  B1 : r08_B  16node is max
  B2 : r08_B  16node >= 2x runner-up ("worst by far")
  B3 : r08_B  pca8 in bottom half ("hardly bad")
  BL1: r08_BL min(pca4,16node,4node,noatok) > max(pca8,pca2,noadaln)  [hard]
  BL2: r08_BL pca8 is the min of the seven ("best"; tolerate tie/noise-floor swap w/ pca2)
  R1 : r01_R  top-2 = {pca4, 4node} (judge consensus, soft)
  RF : REAL   max real <= min over ALL very-mangled BL scores (clean floor)

Usage: python metric_gt_check.py <csv> <score_col> [--label name]
Also supports combining: --combine csv2 col2 (min-max normalized sum).
"""
import sys
import pandas as pd
import numpy as np

OUR7 = ["pca8_8node", "pca4", "pca2", "16node", "4node", "noatok", "noadaln"]
MANGLED_BL = ["pca4", "16node", "4node", "noatok"]
CLEAN_BL = ["pca8_8node", "pca2", "noadaln"]


def load(csv, col):
    df = pd.read_csv(csv)
    wmap = {"r08B": "r08_B", "r08BL": "r08_BL", "r01R": "r01_R"}
    df["window"] = df["window"].map(lambda w: wmap.get(w, w))
    return df.set_index(["window", "run"])[col].astype(float)


def norm(s):
    v = s[[i for i in s.index if i[1] in OUR7 or i[0] == "REAL"]]
    lo, hi = v.min(), v.max()
    return (s - lo) / max(hi - lo, 1e-9)


def check(s, label):
    g = lambda w, r: float(s.get((w, r), np.nan))
    res = {}
    b = {r: g("r08_B", r) for r in OUR7}
    top = max(b, key=b.get)
    rest = sorted(b.values())[:-1]
    res["B1_16node_max"] = top == "16node"
    res["B2_by_far_2x"] = b["16node"] >= 2 * max(rest) if rest else False
    # "hardly bad" = small vs the by-far-worst (rank among floor-noise values is meaningless)
    res["B3_pca8_hardly_bad"] = b["pca8_8node"] <= 0.25 * b["16node"]
    bl = {r: g("r08_BL", r) for r in OUR7}
    res["BL1_separation"] = min(bl[r] for r in MANGLED_BL) > max(bl[r] for r in CLEAN_BL)
    order = sorted(bl, key=bl.get)
    res["BL2_pca8_cleanest"] = order[0] in ("pca8_8node", "pca2") and order[1] in ("pca8_8node", "pca2", "noadaln")
    r1 = {r: g("r01_R", r) for r in OUR7}
    top2 = set(sorted(r1, key=r1.get)[-2:])
    res["R1_pca4_4node_top2"] = top2 == {"pca4", "4node"}
    reals = [v for (w, r), v in s.items() if w == "REAL"]
    res["RF_real_floor"] = (max(reals) <= min(bl[r] for r in MANGLED_BL)) if reals else None
    n_pass = sum(1 for v in res.values() if v)
    print(f"\n[{label}] {n_pass}/{len(res)} PASS")
    for k, v in res.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("  B :", " ".join(f"{r}:{b[r]:.3f}" for r in sorted(b, key=b.get, reverse=True)))
    print("  BL:", " ".join(f"{r}:{bl[r]:.3f}" for r in sorted(bl, key=bl.get, reverse=True)))
    print("  R :", " ".join(f"{r}:{r1[r]:.3f}" for r in sorted(r1, key=r1.get, reverse=True)))
    if reals:
        print(f"  REAL max: {max(reals):.3f}")
    return n_pass


if __name__ == "__main__":
    csv, col = sys.argv[1], sys.argv[2]
    s = load(csv, col)
    if "--combine" in sys.argv:
        i = sys.argv.index("--combine")
        s2 = load(sys.argv[i + 1], sys.argv[i + 2])
        if "--product" in sys.argv:
            # geometric agreement: sqrt(n1 * n2) with small eps so a single-judge
            # miss dampens rather than zeroes (eps = 5% of scale)
            eps = 0.05
            s = np.sqrt((norm(s) + eps) * (norm(s2).reindex(s.index).fillna(0.0) + eps)) - eps
            check(s, f"PRODUCT {csv}:{col} x {sys.argv[i+1]}:{sys.argv[i+2]}")
        else:
            s = norm(s).add(norm(s2), fill_value=0.0)
            check(s, f"COMBINED {csv}:{col} + {sys.argv[i+1]}:{sys.argv[i+2]}")
    else:
        check(s, f"{csv}:{col}")
