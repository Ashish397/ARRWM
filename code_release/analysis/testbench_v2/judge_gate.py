"""Gate (c): judge bake-off on the human-tier-labeled scenes.

A judge model may be used in the Bradley-Terry stage only if it matches or
beats the composite's LOSO accuracy on these same labeled scenes; otherwise
BT runs on composite prescreening alone.

  build : emit out/gate_pairs.csv — every cross-tier pair within each
          scene_key (both orders are handled inside the judge). Run it
          through vlm_pairwise.py --mode pair once per candidate judge:
            TB2_JUDGE=Qwen/Qwen3-VL-30B-A3B-Instruct \
              python vlm_pairwise.py --mode pair --pairs out/gate_pairs.csv \
              --out out/gate_vlm_qwen3vl30b.csv
  eval  : score one or more result CSVs against the tiers — cross-tier
          pairwise accuracy (ties = 1/2), macro over scenes — and print
          them next to the composite's number.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fleet


def build(labels_csv, out_csv):
    labels = pd.read_csv(labels_csv)
    refs = {(r.model, r.scene, r.direction): r for r in fleet.labeled_refs()}
    rows = []
    for key, g in labels.groupby("scene_key"):
        recs = g.to_records(index=False)
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                a, b = recs[i], recs[j]
                if a.tier == b.tier:
                    continue
                ra = refs.get((a.model, a.scene, a.direction))
                rb = refs.get((b.model, b.scene, b.direction))
                if ra is None or rb is None:
                    continue
                rows.append(dict(scene_key=key, scene=int(a.scene),
                                 direction=a.direction,
                                 model_a=a.model, model_b=b.model,
                                 vid_a=ra.vid, vid_b=rb.vid,
                                 path_a=ra.path, path_b=rb.path,
                                 tier_a=float(a.tier), tier_b=float(b.tier)))
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"{len(df)} cross-tier pairs over {df.scene_key.nunique()} "
          f"scene_keys -> {out_csv}")


def eval_results(pairs_csv, result_csvs):
    pairs = pd.read_csv(pairs_csv)
    print(f"{len(pairs)} labeled pairs")
    for res_path in result_csvs:
        res = pd.read_csv(res_path)
        m = pairs.merge(res[["vid_a", "vid_b", "p_a", "judge"]],
                        on=["vid_a", "vid_b"], how="inner")
        if not len(m):
            print(f"{res_path}: no overlapping pairs")
            continue
        # correct if the judge prefers the higher-tier video; p_a=0.5 = tie
        good = np.where(m.p_a == 0.5, 0.5,
                        (np.sign(m.p_a - 0.5) ==
                         np.sign(m.tier_a - m.tier_b)).astype(float))
        m = m.assign(good=good)
        macro = m.groupby("scene")["good"].mean().mean()
        judge = m["judge"].iloc[0]
        print(f"{judge} [{os.path.basename(res_path)}]: "
              f"macro cross-tier pairwise acc = {macro:.3f} "
              f"({len(m)}/{len(pairs)} pairs judged)")
        for scene, g in m.groupby("scene_key"):
            print(f"    {scene}: {g['good'].mean():.3f} (n={len(g)})")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "eval"))
    ap.add_argument("--labels", default=os.path.join(here, "labels", "human_tiers.csv"))
    ap.add_argument("--pairs", default=os.path.join(here, "out", "gate_pairs.csv"))
    ap.add_argument("--results", nargs="*",
                    default=sorted(glob.glob(os.path.join(here, "out", "gate_vlm_*.csv"))))
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.labels, args.pairs)
    else:
        eval_results(args.pairs, args.results)


if __name__ == "__main__":
    main()
