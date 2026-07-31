"""Stage 2.6 — geometry-grounded low-frequency-collapse qualifiers.

Motivation: mangle preserves pixel statistics and embeddings but scrambles
GEOMETRY. Two cheap qualifiers, both with locally-cached models:

  - depth-field metrics (DepthAnything-V2-Small, cached): a mangled scene's
    monocular depth turns rough and temporally unstable.
      depth_rough_{base,end}: mean |Laplacian| of the median-normalized
        disparity (spatial curvature energy of the depth field)
      depth_rough_drift: end - base
      depth_tinstab: mean |disparity(t+1) - disparity(t)| over adjacent
        generated frames (after per-frame median normalization)
  - line-segment stats (OpenCV LSD, CPU): driving scenes are dominated by
    long straight edges; mangle curls and fragments them ("warped in little
    circles", "curved space warping" in the human notes).
      line_long_frac_{base,end}: fraction of total detected segment length
        in segments >= 40px
      line_long_frac_drift: base - end (positive = long lines collapsed)
      line_total_len_drift: log ratio of total segment length base/end

All features are later group-relativized (sibling median) like everything
else; positive drift = worse at end of generation.
"""
import os
import sys

import numpy as np
import pandas as pd
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fleet

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_FRAMES = 8
LONG_PX = 40


class DepthField:
    def __init__(self):
        from transformers import pipeline
        self.pipe = pipeline("depth-estimation",
                             model="depth-anything/Depth-Anything-V2-Small-hf",
                             device=0 if DEV == "cuda" else -1)

    def disparity(self, img):
        from PIL import Image
        d = np.asarray(self.pipe(Image.fromarray(img))["predicted_depth"],
                       dtype=np.float64)
        d = cv2.resize(d, (208, 112))
        return d / (np.median(d) + 1e-6)

    def roughness(self, disp):
        return float(np.abs(cv2.Laplacian(disp, cv2.CV_64F)).mean())


def line_stats(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lsd = cv2.createLineSegmentDetector()
    out = lsd.detect(gray)[0]
    if out is None or len(out) == 0:
        return 0.0, 0.0
    seg = out.reshape(-1, 4)
    lens = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    total = float(lens.sum())
    long_frac = float(lens[lens >= LONG_PX].sum() / (total + 1e-6))
    return total, long_frac


def video_features(ref, depth):
    frames, times, native_fps = fleet.load_video(ref.path)
    ctx, base, end = fleet.windows(times)
    ts = fleet.gen_fraction_times(times, N_FRAMES)
    gen = [fleet.frame_at(frames, times, t) for t in ts]

    row = dict(model=ref.model, scene=ref.scene, direction=ref.direction,
               vid=ref.vid, native_fps=native_fps)

    # depth roughness base/end + temporal instability over generation
    def rough(idx, k=3):
        pick = idx[np.linspace(0, len(idx) - 1, min(k, len(idx))).astype(int)]
        return float(np.mean([depth.roughness(depth.disparity(frames[i]))
                              for i in pick]))
    row["depth_rough_base"] = rough(base)
    row["depth_rough_end"] = rough(end)
    row["depth_rough_drift"] = row["depth_rough_end"] - row["depth_rough_base"]
    disps = [depth.disparity(f) for f in gen]
    row["depth_tinstab"] = float(np.mean(
        [np.abs(a - b).mean() for a, b in zip(disps[:-1], disps[1:])]))

    # line-segment stats base/end
    def lines(idx, k=3):
        pick = idx[np.linspace(0, len(idx) - 1, min(k, len(idx))).astype(int)]
        st = [line_stats(frames[i]) for i in pick]
        return (float(np.mean([s[0] for s in st])),
                float(np.mean([s[1] for s in st])))
    tot_b, lf_b = lines(base)
    tot_e, lf_e = lines(end)
    row["line_long_frac_base"] = lf_b
    row["line_long_frac_end"] = lf_e
    row["line_long_frac_drift"] = lf_b - lf_e
    row["line_total_len_drift"] = float(np.log(tot_b + 1e-6) - np.log(tot_e + 1e-6))
    return row


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.environ.get("TB2_GEOM_OUT", os.path.join(here, "out",
                                                      "geometry_features.csv"))
    shard = int(os.environ.get("TB2_SHARD", "0"))
    nshard = int(os.environ.get("TB2_NSHARD", "1"))
    if nshard > 1:
        out = out.replace(".csv", f".shard{shard}.csv")

    refs = fleet.refs_from_env()
    refs = [r for i, r in enumerate(refs) if i % nshard == shard]

    depth = DepthField()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    for i, ref in enumerate(refs):
        try:
            rows.append(video_features(ref, depth))
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {ref.vid}: {e}", flush=True)
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(refs)}", flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
