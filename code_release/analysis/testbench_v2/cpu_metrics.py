"""Stage-1 CPU metrics: pixel tracking curves + haze/murk drift features.

Runs on everything (no GPU). Per video, emits:
  - per-frame curves (gray mean/std/median, Laplacian variance, dark-channel
    mean, dark-pixel fraction) -> optional NPZ
  - drift features between the t=1s BASE window and the END window
  - lapvar_base, used fleet-wide for dirty-source detection (p12 rule)

Validated against human haze labels: sibling-relative Laplacian-variance
loss reaches AUC 0.86;
adding z-scored dark-channel rise and contrast (pixel std) loss -> 0.89.
The z-scoring and sibling-relative step happen later in groupnorm/style_shift;
this module only extracts raw per-video features.
"""
import os
import sys

import numpy as np
import cv2
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fleet

DARK_PATCH = 15          # dark-channel min-filter kernel (He et al.)
DARK_PIX_THRESH = 25     # gray value below which a pixel counts as "dark"


def frame_stats(img):
    """img uint8 RGB [H,W,3] -> dict of scalar per-frame stats."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    dark = cv2.erode(img.min(axis=2),
                     cv2.getStructuringElement(cv2.MORPH_RECT,
                                               (DARK_PATCH, DARK_PATCH)))
    return dict(
        mean=float(gray.mean()),
        std=float(gray.std()),
        median=float(np.median(gray)),
        lapvar=float(lap.var()),
        dark=float(dark.mean()),
        dark_frac=float((gray < DARK_PIX_THRESH).mean()),
    )


def video_features(ref, save_curves_dir=None):
    frames, times, native_fps = fleet.load_video(ref.path)
    stats = [frame_stats(f) for f in frames]
    curves = {k: np.array([s[k] for s in stats]) for k in stats[0]}

    ctx, base, end = fleet.windows(times)
    w = lambda c, idx: float(np.median(c[idx]))
    eps = 1e-6
    feats = dict(
        model=ref.model, scene=ref.scene, direction=ref.direction, vid=ref.vid,
        path=ref.path, dur=float(times[-1]), native_fps=native_fps,
        # baselines
        lapvar_base=w(curves["lapvar"], base),
        lapvar_ctx=w(curves["lapvar"], ctx),
        # haze components (positive = worse at end)
        haze_lapvar_loss=float(np.log(w(curves["lapvar"], base) + eps)
                               - np.log(w(curves["lapvar"], end) + eps)),
        haze_dark_rise=w(curves["dark"], end) - w(curves["dark"], base),
        haze_contrast_loss=w(curves["std"], base) - w(curves["std"], end),
        # pixel-tracking drift (murk / bright-haze signatures)
        mean_drift=w(curves["mean"], end) - w(curves["mean"], base),
        median_drift=w(curves["median"], end) - w(curves["median"], base),
        std_drift=w(curves["std"], end) - w(curves["std"], base),
        dark_frac_drift=w(curves["dark_frac"], end) - w(curves["dark_frac"], base),
    )
    # signatures: falling std = contrast collapse; median up + std down =
    # bright haze; mean+median down = darkening murk
    feats["sig_contrast_collapse"] = int(feats["std_drift"] < 0)
    feats["sig_bright_haze"] = int(feats["median_drift"] > 0 > feats["std_drift"])
    feats["sig_murk"] = int(feats["mean_drift"] < 0 and feats["median_drift"] < 0)

    if save_curves_dir:
        os.makedirs(save_curves_dir, exist_ok=True)
        np.savez_compressed(os.path.join(save_curves_dir, ref.vid + ".npz"),
                            times=times, **curves)
    return feats


def main():
    out = os.environ.get("TB2_CPU_OUT",
                         os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "out", "cpu_features.csv"))
    curves_dir = os.environ.get("TB2_CURVES_DIR", "")
    shard = int(os.environ.get("TB2_SHARD", "0"))
    nshard = int(os.environ.get("TB2_NSHARD", "1"))

    refs = fleet.refs_from_env()
    refs = [r for i, r in enumerate(refs) if i % nshard == shard]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    for i, ref in enumerate(refs):
        try:
            rows.append(video_features(ref, curves_dir or None))
        except Exception as e:  # noqa: BLE001 - keep the sweep alive
            print(f"[skip] {ref.vid}: {e}", flush=True)
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(refs)}", flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
