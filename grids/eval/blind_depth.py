"""Depth-field geometry on blind100 (rerun to validate vs reference depth_rough_base).
Reuses geometry_metrics.DepthField; per-video base/end windows defined off each model's
ctx (BASE = 12 frames from ctx; END = last 15 frames) so mixed-context externals work."""
import os, sys
sys.path.insert(0, os.path.expanduser("~"))
import numpy as np, imageio, pandas as pd
import blind100_common as bc
from geometry_metrics import DepthField

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "blind_depth.csv")


def read_all(path):
    r = imageio.get_reader(path)
    fr = [np.asarray(f) for f in r]
    r.close(); return fr


def rough_of(depth, frames, idx, k=3):
    idx = np.asarray(idx)
    pick = idx[np.linspace(0, len(idx) - 1, min(k, len(idx))).astype(int)]
    return float(np.mean([depth.roughness(depth.disparity(frames[i])) for i in pick]))


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    depth = DepthField()
    rows = []
    for r in bc.refs():
        fr = read_all(r["path"]); n = len(fr); ctx = r["ctx"]
        base = list(range(ctx, min(ctx + 12, n)))                       # early generation
        end = list(range(max(ctx, n - 15), n))                          # late generation
        gen = list(np.linspace(ctx, n - 1, 8).round().astype(int))
        rb, re_ = rough_of(depth, fr, base), rough_of(depth, fr, end)
        disps = [depth.disparity(fr[i]) for i in gen]
        tinstab = float(np.mean([np.abs(a - b).mean() for a, b in zip(disps[:-1], disps[1:])]))
        rows.append(dict(blind_id=r["blind_id"], vid=r["vid"], model=r["model"], scene=r["scene"],
                         depth_rough_base=rb, depth_rough_end=re_, depth_tinstab=tinstab))
        print(f"[depth] {r['blind_id']} {r['vid']:20s} base={rb:.4f} end={re_:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"[depth] wrote {OUT}")


if __name__ == "__main__":
    main()
