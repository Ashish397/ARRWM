"""V2 upgrade 3: per-scene cross-model pairwise judging aggregated with
Bradley-Terry, with the quality composite as a prescreen so VLM calls are
spent only on CLOSE pairs.

  schedule : emit judge_pairs.csv — for every scene(+direction), all model
             pairs whose |composite gap| <= margin go to the VLM; wide-gap
             pairs are auto-decided by the composite (recorded with
             source='composite', p_a in {0.05, 0.95}).
  fit      : merge VLM results back, fit BT per scene and fleet-wide.

BT with fractional wins: P(A beats B) observations enter as w_ab = p_a,
w_ba = 1 - p_a; strengths via the standard MM iteration.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_MARGIN_PCTL = 40.0   # pairs in the closest 40% of gaps go to the VLM


def schedule(features_scored, out_pairs, out_auto, margin_pctl):
    df = pd.read_csv(features_scored)
    df = df[(df.scene >= 0) & (df.model != "real")].dropna(subset=["composite"])
    pairs = []
    for (scene, direction), g in df.groupby(["scene", "direction"]):
        g = g.drop_duplicates("model")
        recs = g[["model", "vid", "path", "composite"]].to_records(index=False)
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                a, b = recs[i], recs[j]
                pairs.append(dict(scene=scene, direction=direction,
                                  model_a=a.model, model_b=b.model,
                                  vid_a=a.vid, vid_b=b.vid,
                                  path_a=a.path, path_b=b.path,
                                  gap=abs(a.composite - b.composite),
                                  # composite is tier-calibrated: higher = better
                                  comp_p_a=0.95 if a.composite > b.composite else 0.05))
    p = pd.DataFrame(pairs)
    margin = np.percentile(p["gap"].values, margin_pctl)
    close = p[p.gap <= margin]
    wide = p[p.gap > margin].copy()
    wide["p_a"], wide["source"] = wide["comp_p_a"], "composite"
    close.to_csv(out_pairs, index=False)
    wide.to_csv(out_auto, index=False)
    print(f"margin={margin:.4f} (p{margin_pctl:.0f} of gaps): "
          f"{len(close)} pairs -> VLM, {len(wide)} auto-decided by composite")


def bt_fit(wins, models, iters=200):
    """wins: dict[(a,b)] -> fractional win count of a over b. MM iteration."""
    idx = {m: i for i, m in enumerate(models)}
    n = len(models)
    W = np.zeros((n, n))
    for (a, b), w in wins.items():
        W[idx[a], idx[b]] += w
    s = np.ones(n)
    for _ in range(iters):
        s_new = np.empty(n)
        for i in range(n):
            num = W[i].sum()
            den = sum((W[i, j] + W[j, i]) / (s[i] + s[j])
                      for j in range(n) if j != i)
            s_new[i] = num / den if den > 0 else s[i]
        s_new /= s_new.sum() / n
        if np.max(np.abs(np.log(s_new + 1e-12) - np.log(s + 1e-12))) < 1e-8:
            s = s_new
            break
        s = s_new
    return {m: float(np.log(s[idx[m]])) for m in models}


def fit(pairs_csv, vlm_csv, auto_csv, out_prefix):
    judged = pd.read_csv(vlm_csv)
    sched = pd.read_csv(pairs_csv)
    close = sched.merge(judged[["vid_a", "vid_b", "p_a"]],
                        on=["vid_a", "vid_b"], how="inner")
    close["source"] = "vlm"
    frames = [close]
    if auto_csv and os.path.exists(auto_csv):
        frames.append(pd.read_csv(auto_csv))
    allp = pd.concat(frames, ignore_index=True)
    n_missing = len(sched) - len(close)
    if n_missing:
        print(f"warning: {n_missing} scheduled pairs missing VLM results")

    models = sorted(set(allp.model_a) | set(allp.model_b))
    rows = []
    for scene, g in allp.groupby("scene"):
        wins = {}
        for _, r in g.iterrows():
            wins[(r.model_a, r.model_b)] = wins.get((r.model_a, r.model_b), 0) + r.p_a
            wins[(r.model_b, r.model_a)] = wins.get((r.model_b, r.model_a), 0) + 1 - r.p_a
        present = sorted(set(g.model_a) | set(g.model_b))
        bt = bt_fit(wins, present)
        for m, v in bt.items():
            rows.append(dict(scene=scene, model=m, bt_score=v))
    per_scene = pd.DataFrame(rows)
    per_scene.to_csv(out_prefix + "_per_scene.csv", index=False)

    wins = {}
    for _, r in allp.iterrows():
        wins[(r.model_a, r.model_b)] = wins.get((r.model_a, r.model_b), 0) + r.p_a
        wins[(r.model_b, r.model_a)] = wins.get((r.model_b, r.model_a), 0) + 1 - r.p_a
    fleet_bt = bt_fit(wins, models)
    fleet_df = (pd.DataFrame([dict(model=m, bt_score=v) for m, v in fleet_bt.items()])
                .sort_values("bt_score", ascending=False))
    fleet_df.to_csv(out_prefix + "_fleet.csv", index=False)
    print(fleet_df.to_string(index=False))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "out")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("schedule", "fit"))
    ap.add_argument("--features", default=os.path.join(out, "features_scored.csv"))
    ap.add_argument("--pairs", default=os.path.join(out, "judge_pairs.csv"))
    ap.add_argument("--auto", default=os.path.join(out, "judge_auto.csv"))
    ap.add_argument("--vlm", default=os.path.join(out, "vlm_pair.csv"))
    ap.add_argument("--margin-pctl", type=float, default=DEFAULT_MARGIN_PCTL)
    ap.add_argument("--out-prefix", default=os.path.join(out, "bt"))
    args = ap.parse_args()
    if args.cmd == "schedule":
        schedule(args.features, args.pairs, args.auto, args.margin_pctl)
    else:
        fit(args.pairs, args.vlm, args.auto, args.out_prefix)


if __name__ == "__main__":
    main()
