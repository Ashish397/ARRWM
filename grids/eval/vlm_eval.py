"""Evaluate vlm_judge_results.jsonl against human tiers in gt.json.

Primary: cross-tier pairwise accuracy within grid (swap-aggregated: both orders must
agree, else tie -> 0.5 credit). Secondary: per-grid Kendall tau_b of within-grid
win-rate scores vs human quality.
"""
import itertools, json, os, sys
from collections import defaultdict
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path=os.path.join(HERE, "vlm_judge_results.jsonl")):
    recs = [json.loads(l) for l in open(path)]
    gt = json.load(open(os.path.join(HERE, "gt.json")))["grids"]
    return recs, gt


def swap_aggregate(recs):
    """-> {(model,prompt,pack,grid,frozenset{a,b}): verdict in {a,b,'T'}}"""
    by_pair = defaultdict(dict)
    for r in recs:
        if not r["verdict"] or "overall" not in r["verdict"]:
            continue
        k = (r["model"], r["prompt"], r["pack"], r["grid"], frozenset((r["a"], r["b"])))
        v = r["verdict"]["overall"]
        winner = r["a"] if v == "A" else (r["b"] if v == "B" else "T")
        by_pair[k][(r["a"], r["b"])] = winner
    out = {}
    for k, orders in by_pair.items():
        ws = list(orders.values())
        if len(ws) == 1:
            out[k] = ws[0]
        else:
            out[k] = ws[0] if ws[0] == ws[1] else "T"
    return out


def main():
    recs, gt = load()
    agg = swap_aggregate(recs)
    configs = sorted(set((k[0], k[1], k[2]) for k in agg))
    for cfg in configs:
        model, prompt, pack = cfg
        per_grid_acc, per_grid_tau, per_grid_n = {}, {}, {}
        for grid, variants in gt.items():
            q = {v: d["quality"] for v, d in variants.items()}
            wins = defaultdict(float); counts = defaultdict(int)
            score = total = 0
            for a, b in itertools.combinations(sorted(q), 2):
                k = (model, prompt, pack, grid, frozenset((a, b)))
                if k not in agg:
                    continue
                w = agg[k]
                counts[a] += 1; counts[b] += 1
                if w == a: wins[a] += 1
                elif w == b: wins[b] += 1
                else: wins[a] += 0.5; wins[b] += 0.5
                if q[a] == q[b]:
                    continue  # same tier: not scored
                total += 1
                hi = a if q[a] > q[b] else b
                if w == hi: score += 1
                elif w == "T": score += 0.5
            if total == 0:
                continue
            per_grid_acc[grid] = score / total
            per_grid_n[grid] = total
            wr = {v: wins[v] / max(counts[v], 1) for v in q}
            per_grid_tau[grid] = stats.kendalltau(
                [wr[v] for v in q], [q[v] for v in q], variant="b").statistic
        if not per_grid_acc:
            continue
        n = sum(per_grid_n.values())
        print(f"\n=== {model} | {prompt} | {pack}  ({len(per_grid_acc)} grids, {n} cross-tier pairs) ===")
        print(f"cross-tier pairwise accuracy: {np.mean(list(per_grid_acc.values())):.1%} (macro) | tau_b: {np.nanmean(list(per_grid_tau.values())):+.3f}")
        for g in sorted(per_grid_acc):
            print(f"  {g}: acc={per_grid_acc[g]:.1%} (n={per_grid_n[g]}) tau_b={per_grid_tau[g]:+.2f}")


if __name__ == "__main__":
    main()
