"""Matched melt-VLM AUC table on the 84-labelled dev set.

Reads out/melt_vlm_<model>.csv for each candidate and prints:
  dev-AUC : rank-AUC separating SEVERE mangle (note flags mangle & tier<=5) from
            GOOD videos (tier>=6.5), higher p_yes = more mangled.
  P(Yes) diagnostics : mean/std and saturation fractions -- DESCRIPTIVE only, NOT a
            selection gate. Selection is purely on dev-AUC.

Honest framing (per reviewer): every number here is a DEVELOPMENT-set result on the
same 84 labelled videos -- the labels were used both to observe each model's AUC and
(elsewhere) the 0.89 ensemble, so it is not a nested held-out estimate.
"""
import os, re, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["qwen25vl7b", "internvl3_8b", "cosmos_reason1_7b"]
MANGLE_RE = re.compile(r"mangl|mash|warp|melt|jumbl|geometr|black mass|black mess|blob", re.I)


def rank_auc(pos, neg):
    """P(random pos > random neg), ties=0.5. pos should score HIGHER (more mangled)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank().values          # average ranks handle ties
    rp = ranks[:len(pos)].sum()
    u = rp - len(pos) * (len(pos) + 1) / 2.0
    return u / (len(pos) * len(neg))


def main():
    lab = pd.read_csv(os.path.join(HERE, "human_tiers.csv"))
    lab["mangle"] = lab.note.astype(str).str.contains(MANGLE_RE).astype(int)
    print(f"labels: {len(lab)} videos | severe(mangle&tier<=5)="
          f"{((lab.mangle == 1) & (lab.tier <= 5)).sum()} | good(tier>=6.5)={(lab.tier >= 6.5).sum()}\n")
    print(f"{'model':18s} {'dev-AUC':>8s}  {'mean':>5s} {'std':>5s} {'>0.9':>5s} {'<0.1':>5s}   status")
    for m in MODELS:
        f = os.path.join(HERE, "out", f"melt_vlm_{m}.csv")
        if not os.path.exists(f):
            print(f"{m:18s} {'--':>8s}   (not run yet)"); continue
        d = pd.read_csv(f).merge(lab[["scene", "model", "mangle"]], on=["scene", "model"])
        sev = d[(d.mangle == 1) & (d.tier <= 5)].p_yes
        good = d[d.tier >= 6.5].p_yes
        auc = rank_auc(sev, good)
        p = d.p_yes.dropna()
        print(f"{m:18s} {auc:8.3f}  {p.mean():5.2f} {p.std():5.2f} "
              f"{(p > 0.9).mean():5.2f} {(p < 0.1).mean():5.2f}   {len(p)} scored")


if __name__ == "__main__":
    main()
