"""Global-vs-residual optical-flow split for the no-op (null-action) rollouts.

`motion_ratio` (gen frame-diff / ref frame-diff) is a single number pointing
two opposite ways: a value >1 can mean the camera DRIFTS under a no-op
command, or that the world is hallucinating extra activity; a value <1 can
mean a dead world, or a correctly-still camera in a still scene. This script
separates the two by splitting RAFT flow into a global parametric component
(camera / ego) and the unexplained residual (independently moving objects):

  ego_px_s    median magnitude of the RANSAC homography's displacement field
              over a grid, in px/s at canonical 832x448
  resid_px_s  mean magnitude of flow MINUS that global model, px/s -- the
              part of the motion no single camera model explains
  resid_frac  fraction of grid points whose residual exceeds 2 px (the
              moving-object area share)
  inlier_frac RANSAC support for the global model; low values mean the field
              was too incoherent for ANY camera model to describe, which is
              itself a corruption signal

Derived, per window, against that window's real continuation:
  ego_excess   gen ego_px_s - ref ego_px_s   >0 = drift under a no-op
  world_ratio  gen resid_px_s / ref resid_px_s   <1 = dead, ~1 = alive

Caveat recorded, not hidden: a homography cannot represent parallax under
real ego translation, so some genuine ego motion leaks into the residual.
Both sides of every comparison are measured the same way and the references
are near-stationary (ego ~0.023), so the leak is small and common-mode for a
model that holds still. It is NOT small for a model that drifts hard: pca8
drifts at ~5.8 px/s above the reference and its world_ratio rises to ~5 as a
consequence, not because it invented five times the world activity.

  => world_ratio is diagnostic only where ego_excess is near 0. Read
     ego_excess FIRST; world_ratio answers "is the world alive" only once
     the camera has been shown to be still. A model high on both is a
     drifter, not a drifter-and-hallucinator.

Pairs are taken at a fixed wall-clock spacing (0.25 s) and normalised to
px/s, so the 16-30 fps spread across the fleet is not a confound. Videos
shorter than the reference window are scored only over the span they
actually generated, and `cover` records that fraction -- minwm generates
~4.06 s against a 6 s window, and averaging a repeated final frame into its
motion would manufacture a freeze that the model never produced.

Usage:
  python noop_flow_split.py                      # all models, A+B starts
  python noop_flow_split.py --only matrixgame     # rescore + merge one model
  python noop_flow_split.py --starts A

Role: measurement stage -- global vs residual optical-flow split.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ARR = os.environ.get("AF_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ARR)
sys.path.insert(0, HERE)
import fleet  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
REF_DIR = os.path.join(ARR, "analysis", "eval_final", "noop_refs")
CTX_T = 12 / 16.0        # generation starts at frame 12 of the 16fps refs
GEN_SEC = 6.0            # reference generation window
DT = 0.25                # wall-clock spacing of a flow pair
N_PAIRS = 8              # pairs sampled across the generation window
GRID = 16                # correspondence grid stride (px)
RESID_PX = 2.0           # residual threshold for the moving-area share
ABLATIONS = {"pca8_8node": "pca8", "pca4": "pca4", "pca2": "pca2",
             "16node": "16node", "4node": "4node", "noatok": "noatok",
             "noadaln": "noadaln"}


class FlowSplit:
    def __init__(self):
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        self.model = raft_small(weights=Raft_Small_Weights.DEFAULT).to(DEV).eval()

    @staticmethod
    def _prep(f):
        x = torch.from_numpy(f).to(DEV).permute(2, 0, 1)[None].float() / 127.5 - 1.0
        return x[:, :, :(x.shape[2] // 8) * 8, :(x.shape[3] // 8) * 8]

    @torch.no_grad()
    def pair(self, im_a, im_b, dt):
        """One (ego, resid, resid_frac, inlier_frac) sample, in px/s."""
        import cv2
        flow = self.model(self._prep(im_a), self._prep(im_b))[-1][0].cpu().numpy()
        H, W = flow.shape[1:]
        ys, xs = np.mgrid[GRID // 2:H:GRID, GRID // 2:W:GRID]
        ys, xs = ys.ravel(), xs.ravel()
        p1 = np.stack([xs, ys], 1).astype(np.float64)
        p2 = p1 + np.stack([flow[0, ys, xs], flow[1, ys, xs]], 1)

        Hm, inl = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
        if Hm is None:
            # no coherent global model at all: attribute everything to the
            # residual and flag it via inlier_frac=0
            mag = np.hypot(*(p2 - p1).T)
            return (0.0, float(mag.mean()) / dt,
                    float((mag > RESID_PX).mean()), 0.0)
        proj = cv2.perspectiveTransform(p1[None].astype(np.float32),
                                        Hm)[0].astype(np.float64)
        ego = np.hypot(*(proj - p1).T)          # what the camera model moves
        resid = np.hypot(*(p2 - proj).T)        # what it fails to explain
        return (float(np.median(ego)) / dt, float(resid.mean()) / dt,
                float((resid > RESID_PX).mean()),
                float(inl.mean()) if inl is not None else 0.0)

    def video(self, path):
        """Median split over the generation window, plus coverage."""
        frames, times, fps = fleet.load_video(path)
        t_end = float(times[-1])
        # only sample pairs that fit inside what this video actually generated
        hi = min(CTX_T + GEN_SEC, t_end - DT)
        if hi <= CTX_T:
            raise ValueError(f"no generation window (ends {t_end:.2f}s)")
        ts = np.linspace(CTX_T, hi, N_PAIRS)
        vals = [self.pair(fleet.frame_at(frames, times, t),
                          fleet.frame_at(frames, times, t + DT), DT) for t in ts]
        v = np.asarray(vals)
        return dict(ego_px_s=float(np.median(v[:, 0])),
                    resid_px_s=float(np.median(v[:, 1])),
                    resid_frac=float(np.median(v[:, 2])),
                    inlier_frac=float(np.median(v[:, 3])),
                    native_fps=fps,
                    cover=float(min(1.0, (hi + DT - CTX_T) / GEN_SEC)))


def targets(starts, want):
    """(model, start_set, scene, path) for every no-op video to score."""
    out = []
    if "B" in starts:
        for run, model in ABLATIONS.items():
            for wi in range(64):
                p = os.path.join(ARR, "logs/eval_final/B", run, "control_test",
                                 f"step05000_r{wi:02d}_static_raw.mp4")
                if os.path.exists(p):
                    out.append((model, "B", wi, p))
    if "A" in starts:
        for run, model in ABLATIONS.items():
            for wi in range(32):
                p = os.path.join(ARR, "logs/eval_final/B_Astarts", run,
                                 "control_test",
                                 f"step05000_r{wi:02d}_static_raw.mp4")
                if os.path.exists(p):
                    out.append((model, "A", wi, p))
        ext = (glob.glob(os.path.join(ARR, "logs/eval_final/A_*_nullact"))
               + glob.glob(os.path.join(ARR, "logs/eval_final/A_matrixgame_noop")))
        for d in sorted(ext):
            model = os.path.basename(d)[2:]
            for p in sorted(glob.glob(os.path.join(d, "*.mp4"))):
                m = re.search(r"_r(\d+)_", os.path.basename(p))
                if m:
                    out.append((model, "A", int(m.group(1)), p))
    return [t for t in out if want(t[0])]


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="regex on model name; merges into the existing csv")
    ap.add_argument("--starts", default="AB", help="which start sets: A, B, AB")
    args = ap.parse_args()
    sel = re.compile(args.only) if args.only else None

    def want(m):
        return sel is None or sel.search(m) is not None

    fs = FlowSplit()
    out_p = os.path.join(HERE, "out", "noop_flow.csv")
    tg = targets(args.starts, want)
    print(f"{len(tg)} no-op videos + refs", flush=True)

    # references first: one per (start_set, scene), the denominator
    ref_cache = {}
    for ss, scene in sorted({(t[1], t[2]) for t in tg}):
        rp = os.path.join(REF_DIR, f"ref{ss}_r{scene:02d}.mp4")
        if os.path.exists(rp):
            try:
                ref_cache[(ss, scene)] = fs.video(rp)
            except Exception as e:  # noqa: BLE001
                print(f"[skip ref] {ss} r{scene}: {str(e)[:80]}", flush=True)
    print(f"{len(ref_cache)} refs done", flush=True)

    rows = []
    for i, (model, ss, scene, path) in enumerate(tg):
        r = ref_cache.get((ss, scene))
        if r is None:
            continue
        try:
            g = fs.video(path)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {model} {ss} r{scene}: {str(e)[:80]}", flush=True)
            continue
        rows.append(dict(
            model=model, start_set=ss, scene=scene,
            vid=f"{model}_{ss}_r{scene:02d}",
            **{f"gen_{k}": v for k, v in g.items()},
            **{f"ref_{k}": v for k, v in r.items() if k != "cover"},
            ego_excess=g["ego_px_s"] - r["ego_px_s"],
            world_ratio=g["resid_px_s"] / (r["resid_px_s"] + 1e-6),
        ))
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(tg)}", flush=True)
            pd.DataFrame(rows).to_csv(out_p + ".partial", index=False)

    df = pd.DataFrame(rows)
    if sel is not None and os.path.exists(out_p):
        old = pd.read_csv(out_p)
        keep = old[~old.model.isin(set(df.model.unique()))]
        print(f"merged: kept {len(keep)} rows for {sorted(keep.model.unique())}",
              flush=True)
        df = pd.concat([keep, df], ignore_index=True)
    df.to_csv(out_p, index=False)

    agg = (df.groupby(["model", "start_set"])
           .agg(n=("vid", "count"), cover=("gen_cover", "min"),
                gen_ego=("gen_ego_px_s", "median"),
                ref_ego=("ref_ego_px_s", "median"),
                ego_excess=("ego_excess", "median"),
                world_ratio=("world_ratio", "median"),
                gen_inlier=("gen_inlier_frac", "median")).round(3)
           .sort_values(["start_set", "ego_excess"]))
    agg.to_csv(os.path.join(HERE, "out", "noop_flow_summary.csv"))
    print(agg.to_string())
    print(f"\nwrote {len(df)} -> {out_p}")


if __name__ == "__main__":
    main()
