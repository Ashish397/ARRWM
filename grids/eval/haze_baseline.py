"""Baseline-aware haze/blur quantification.

Measures absolute haze and blur at t=1s (frame 16 @16fps — skipping frame-0
artefacts) and at the end of the video. If haze/blur is ALREADY present at the
1s baseline, it belongs to the source video (water on lens, smudge) and the
video has NOT degraded — only the (end - baseline) delta counts as degradation.

Stats per tile:
  base_haze   dark-channel mean at t=1s (high = hazy/veiled already)
  base_blur   Laplacian variance at t=1s (low = blurry already)
  base_contrast  gray std at t=1s
  d_haze / d_blur / d_contrast   end-window minus baseline
Flags:
  baseline_dirty  = base stats abnormal vs fleet percentiles (lens water/smudge)
  degraded_haze   = d_haze large AND baseline was clean

Usage: haze_baseline.py [gridname ...]   (default: all tiles in tiles/ + tiles_new/)
Writes results_hazebase.csv.
"""
import glob, os, sys
import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_hazebase.csv")
FPS = 16
GRID_SRC = "/home/ashish/ARRWM/grids/grids_A/A"
POS = {"pca8": (0, 0), "pca4": (832, 0), "pca2": (1664, 0), "16node": (2496, 0),
       "4node": (0, 480), "noatok": (832, 480), "noadaln": (1664, 480)}


def read_video(path):
    cap = cv2.VideoCapture(path)
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(fr)


def stats(win):
    """win: few frames uint8 RGB."""
    dch = np.mean([cv2.erode(f.min(-1), np.ones((15, 15), np.uint8)).mean() for f in win])
    lap = np.mean([cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() for f in win])
    con = np.mean([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).std() for f in win])
    return float(dch), float(lap), float(con)


def tile_iter(grids):
    if grids:
        for g in grids:
            frames = read_video(os.path.join(GRID_SRC, f"{g}_grid.mp4"))
            for v, (x, y) in POS.items():
                yield g, v, frames[:, y + 32:y + 480, x:x + 832]
    else:
        seen = set()
        for fp in sorted(glob.glob(os.path.join(HERE, "tiles", "*.mp4")) + glob.glob(os.path.join(HERE, "tiles_new", "*.mp4"))):
            name = os.path.basename(fp)[:-4]
            if name in seen:
                continue
            seen.add(name)
            g, v = name.split("__")
            yield g, v, read_video(fp)


def main():
    grids = sys.argv[1:]
    rows = []
    for g, v, frames in tile_iter(grids):
        t1 = FPS  # frame at 1s
        base = frames[t1:t1 + 4]      # 4 frames from t=1s
        end = frames[-8:]
        b_h, b_b, b_c = stats(base)
        e_h, e_b, e_c = stats(end)
        rows.append({"grid": g, "variant": v,
                     "base_haze": round(b_h, 1), "base_blur": round(b_b, 1), "base_contrast": round(b_c, 1),
                     "end_haze": round(e_h, 1), "end_blur": round(e_b, 1),
                     "d_haze": round(e_h - b_h, 1), "d_blur": round(e_b - b_b, 1), "d_contrast": round(e_c - b_c, 1)})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    if os.path.exists(OUT):
        old = pd.read_csv(OUT)
        old = old[~old.set_index(["grid", "variant"]).index.isin(df.set_index(["grid", "variant"]).index)]
        df = pd.concat([old, df], ignore_index=True)
    # fleet percentiles -> flags
    p_haze_hi = df.base_haze.quantile(0.85)
    p_blur_lo = df.base_blur.quantile(0.15)
    df["baseline_dirty"] = (df.base_haze > p_haze_hi) | (df.base_blur < p_blur_lo)
    df["degraded_haze"] = (~df.baseline_dirty) & (df.d_haze > 12)
    df["degraded_blur"] = (~df.baseline_dirty) & (df.d_blur < -0.45 * df.base_blur)
    df.to_csv(OUT, index=False)
    print(f"\nfleet thresholds: base_haze>{p_haze_hi:.1f} or base_blur<{p_blur_lo:.1f} -> baseline_dirty")
    print(f"wrote {OUT} ({len(df)} tiles)")


if __name__ == "__main__":
    main()
