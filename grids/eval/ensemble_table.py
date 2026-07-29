"""Assemble the 3-component geometry ensemble on the 84 labelled videos and report
component + ensemble dev-AUC, with the melt vote as Qwen (the 0.89 reference) vs
Cosmos (the reviewer's candidate).

Recipe (reproduced from the cluster ensemble):
  components P = [melt_pyes, pal4vst_max, depth_rough_base]
  rel_c   = c - sibling(scene) median          (group-relative, cancels scene difficulty)
  z_rel_c = zscore of rel_c across the fleet
  ens     = mean(z_rel_c over the 3 components)
Higher = more mangled. AUC separates SEVERE (mangle-note & tier<=5) from GOOD (tier>=6.5).
Every number is a DEVELOPMENT-set figure on the same 84 labels (not held-out).
"""
import os, re
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
MANGLE_RE = re.compile(r"mangl|mash|warp|melt|jumbl|geometr|black mass|black mess|blob", re.I)


def rank_auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    allv = np.concatenate([pos, neg]); ranks = pd.Series(allv).rank().values
    u = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0
    return u / (len(pos) * len(neg))


def add_rel(df, cols):
    for c in cols:
        df["rel_" + c] = df[c] - df.groupby("scene")[c].transform("median")
    return df


def zscore(df, cols):
    for c in cols:
        v = df[c]
        df["z_" + c] = (v - v.mean()) / (v.std() + 1e-9)
    return df


def main():
    lab = pd.read_csv(os.path.join(HERE, "human_tiers.csv"))
    lab["mangle"] = lab.note.astype(str).str.contains(MANGLE_RE).astype(int)

    pal = pd.read_csv(os.path.join(OUT, "pal4vst_local.csv"))[["scene", "model", "pal_max"]]
    dep = pd.read_csv(os.path.join(OUT, "geometry_depth_local.csv"))[["scene", "model", "depth_rough_base"]]
    melts = {m: pd.read_csv(os.path.join(OUT, f"melt_vlm_{m}.csv"))[["scene", "model", "p_yes"]]
             .rename(columns={"p_yes": "melt_pyes"})
             for m in ["qwen25vl7b", "cosmos_reason1_7b"]}

    print(f"{'melt vote':14s} | {'melt-AUC':>8s} {'pal-AUC':>8s} {'depth-AUC':>9s} | {'ENSEMBLE':>9s}")
    print("-" * 62)
    for mname, melt in melts.items():
        df = lab.merge(melt, on=["scene", "model"]).merge(pal, on=["scene", "model"]).merge(dep, on=["scene", "model"])
        P = ["melt_pyes", "pal_max", "depth_rough_base"]
        df = add_rel(df, P)
        df = zscore(df, ["rel_" + c for c in P])
        df["ens"] = df[["z_rel_" + c for c in P]].mean(1)

        sev = df[(df.mangle == 1) & (df.tier <= 5)]
        good = df[df.tier >= 6.5]
        aucs = {c: rank_auc(sev["rel_" + c], good["rel_" + c]) for c in P}
        ens_auc = rank_auc(sev.ens, good.ens)
        print(f"{mname:14s} | {aucs['melt_pyes']:8.3f} {aucs['pal_max']:8.3f} "
              f"{aucs['depth_rough_base']:9.3f} | {ens_auc:9.3f}")

    print("\n(component AUCs are group-relative rel_; ensemble = mean of z-scored rel_ components)")
    print(f"severe n={((lab.mangle==1)&(lab.tier<=5)).sum()}  good n={(lab.tier>=6.5).sum()}  (dev set, not held-out)")


if __name__ == "__main__":
    main()
