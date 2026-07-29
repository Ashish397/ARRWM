"""Exact depth_rough_base (+ end/drift/tinstab) on the 84 labelled videos, LOCAL.

Reuses the cluster's fleet.windows + geometry_metrics.DepthField UNCHANGED so the
depth component matches the ensemble's definition exactly. Only the ref-discovery is
swapped: local tiles live at tiles_new/tiles/{scene}__{model}.mp4, not the cluster
layout, so we build the frame set directly per labelled row.

Corrected boundary (TB2_CORRECT_BOUNDARY=1): BASE = 0.75-1.50s = frames 12-23, all
generated -> boundary-clean, no 9-11 leakage.

Env: DEPTH_OUT (default out/geometry_depth_local.csv).
"""
import os, sys
sys.path.insert(0, os.path.expanduser("~"))     # fleet.py + geometry_metrics.py live in ~
os.environ.setdefault("TB2_CORRECT_BOUNDARY", "1")
import numpy as np, pandas as pd
import fleet
from geometry_metrics import DepthField, N_FRAMES

HERE = os.path.dirname(os.path.abspath(__file__))
TILE_DIRS = [os.path.join(HERE, "tiles_new"), os.path.join(HERE, "tiles")]
LAB = os.path.join(HERE, "human_tiers.csv")
OUT = os.environ.get("DEPTH_OUT", os.path.join(HERE, "out", "geometry_depth_local.csv"))


def vid_path(scene, m):
    for d in TILE_DIRS:
        p = os.path.join(d, f"{scene}__{m}.mp4")
        if os.path.exists(p):
            return p
    return None


def rough_of(depth, frames, idx, k=3):
    """Mean |Laplacian| of median-normalized disparity over k frames of a window
    -- identical to geometry_metrics.video_features.rough()."""
    pick = idx[np.linspace(0, len(idx) - 1, min(k, len(idx))).astype(int)]
    return float(np.mean([depth.roughness(depth.disparity(frames[i])) for i in pick]))


def main():
    dry = "--dry" in sys.argv
    lab = pd.read_csv(LAB)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    depth = DepthField()
    rows = []
    for n, (_, r) in enumerate(lab.iterrows()):
        p = vid_path(r.scene, r.model)
        if p is None:
            print(f"[depth] MISSING {r.scene}__{r.model}", flush=True); continue
        frames, times, fps = fleet.load_video(p)
        ctx, base, end = fleet.windows(times)
        ts = fleet.gen_fraction_times(times, N_FRAMES)
        gen = [fleet.frame_at(frames, times, t) for t in ts]
        if dry:
            print(f"fps={fps:.1f} nframes={len(frames)} base_idx={list(base)} "
                  f"end_idx={list(end)} gen_ts={np.round(ts,2).tolist()}")
            return
        rb = rough_of(depth, frames, base)
        re_ = rough_of(depth, frames, end)
        disps = [depth.disparity(f) for f in gen]
        tinstab = float(np.mean([np.abs(a - b).mean() for a, b in zip(disps[:-1], disps[1:])]))
        rows.append(dict(scene=r.scene, model=r.model, tier=r.tier, note=r.note,
                         depth_rough_base=rb, depth_rough_end=re_,
                         depth_rough_drift=re_ - rb, depth_tinstab=tinstab, native_fps=fps))
        print(f"[depth] {r.scene:8s} {r.model:8s} tier={r.tier:<4} "
              f"base={rb:.4f} end={re_:.4f} drift={re_-rb:+.4f} tinstab={tinstab:.4f}", flush=True)
        if (n + 1) % 12 == 0:
            pd.DataFrame(rows).to_csv(OUT, index=False)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"[depth] wrote {OUT} ({len(rows)} videos)", flush=True)


if __name__ == "__main__":
    main()
