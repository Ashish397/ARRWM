"""Test a FLEET-INVARIANT relocation metric against the 80-rollout human labels.

Motivation: the deployed relocation score is `max RANSAC-verified ORB inliers to
any OTHER fleet member`, so it changes when models are added or removed (adding
a member can only raise the max, so published rates can only fall). Seed-vs-end
matching would be invariant but fails under sustained yaw -- a Left rollout has a
median of ~6 seed-vs-end inliers while being perfectly good, because a long
baseline destroys overlap.

Proposed fix -- CHAINED place identity, using only the rollout's own frames plus
its real seed:

    anchor at the last real context frame, then step through the generated span
    at chunk boundaries:  ctx-1 -> ctx -> ... -> ctx + 6s
    each link = RANSAC-verified ORB inliers between CONSECUTIVE frames
    score     = min over links (the weakest link)
    relocated = score < threshold

A legitimate turn changes the view gradually, so every adjacent link keeps high
overlap; a teleport breaks exactly one link and the min collapses. No other model
is referenced, so the metric is fleet-invariant.

Benchmark: results_scene_cand.csv (80 rollouts, 27 human-labelled positives).
The deployed consensus rule scores 88.8% agreement on this set (TP23 FP5 FN4 TN48).

Usage:  python reloc_invariant_test.py            # full 80
        python reloc_invariant_test.py --limit 20 # quick smoke
"""
import os, sys
import numpy as np, pandas as pd, cv2
import fleet_common as fc

HERE = os.path.dirname(os.path.abspath(__file__))
GT = os.path.join(HERE, "results_scene_cand.csv")
CONS = os.path.join(HERE, "results_scene_consensus_val.csv")
OUT = os.path.join(HERE, "out", "reloc_invariant_test.csv")
NLINK = 5                                  # 5 links spanning the 6s horizon

_orb = cv2.ORB_create(3000)
_bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def desc(rgb):
    g = cv2.cvtColor(cv2.resize(rgb, (640, 352)), cv2.COLOR_RGB2GRAY)
    return _orb.detectAndCompute(g, None)


def inl(a, b):
    (k0, d0), (k1, d1) = a, b
    if d0 is None or d1 is None or len(k0) < 8 or len(k1) < 8:
        return 0
    ms = _bf.match(d0, d1)
    if len(ms) < 8:
        return 0
    src = np.float32([k0[m.queryIdx].pt for m in ms])
    dst = np.float32([k1[m.trainIdx].pt for m in ms])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


def chain_indices(scene, model):
    n, fps = fc.meta(scene, model)
    ctx = fc.ctx_of(model)
    end = min(n - 1, ctx + int(round(6.0 * fps)))
    pts = np.linspace(ctx, end, NLINK).round().astype(int)
    anchor = max(0, ctx - 1)               # last REAL context frame
    return [anchor] + [int(p) for p in pts]


def main():
    lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    gt = pd.read_csv(GT)
    if lim:
        gt = gt.head(lim)
    cons = pd.read_csv(CONS)
    rows = []
    for i, r in enumerate(gt.itertuples(index=False), 1):
        try:
            idx = chain_indices(r.scene, r.model)
            fr = fc.frames_at(r.scene, r.model, idx)
            if fr is None or len(fr) < len(idx):
                print(f"skip {r.model} {r.scene}: short read", flush=True); continue
            ds = [desc(f) for f in fr]
            links = [inl(ds[j], ds[j + 1]) for j in range(len(ds) - 1)]
            rows.append(dict(scene=r.scene, model=r.model, gt=int(r.gt),
                             chain_min=int(min(links)), seed_link=int(links[0]),
                             gen_min=int(min(links[1:])) if len(links) > 1 else 0))
        except Exception as e:
            print(f"skip {r.model} {r.scene}: {str(e)[:60]}", flush=True); continue
        if i % 20 == 0:
            print(f"[{i}/{len(gt)}]", flush=True)
    d = pd.DataFrame(rows).merge(cons, on=["scene", "model"], how="left")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    d.to_csv(OUT, index=False)

    print(f"\nn={len(d)}  human positives={int(d.gt.sum())}")
    base = (d.consensus_inl < 50).astype(int)
    print(f"BASELINE  consensus<50 (fleet-COUPLED): {100*(base==d.gt).mean():5.1f}%"
          f"  TP{int(((base==1)&(d.gt==1)).sum())} FP{int(((base==1)&(d.gt==0)).sum())}"
          f" FN{int(((base==0)&(d.gt==1)).sum())} TN{int(((base==0)&(d.gt==0)).sum())}")
    print("\nCHAINED (fleet-INVARIANT) threshold sweep:")
    best = (0, None)
    for col in ("chain_min", "gen_min"):
        for t in (10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150):
            pred = (d[col] < t).astype(int)
            acc = (pred == d.gt).mean()
            if acc > best[0]:
                best = (acc, (col, t))
            print(f"  {col:9s} < {t:3d}: {100*acc:5.1f}%"
                  f"  TP{int(((pred==1)&(d.gt==1)).sum())} FP{int(((pred==1)&(d.gt==0)).sum())}"
                  f" FN{int(((pred==0)&(d.gt==1)).sum())} TN{int(((pred==0)&(d.gt==0)).sum())}")
    print(f"\nBEST invariant: {best[1][0]} < {best[1][1]}  ->  {100*best[0]:.1f}%"
          f"   (baseline consensus 88.8%)")
    try:
        from sklearn.metrics import roc_auc_score
        for col in ("chain_min", "gen_min"):
            print(f"AUC({col}) = {roc_auc_score(d.gt, -d[col]):.3f}"
                  f"   AUC(consensus) = {roc_auc_score(d.gt, -d.consensus_inl):.3f}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
