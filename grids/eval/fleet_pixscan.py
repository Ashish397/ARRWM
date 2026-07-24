"""Fleet-wide pixel scan: every grid video in grids_A/A, all 7 tiles.

Per tile: pixel curves stats (mean/std/median, t0->4s and t1s->end deltas),
baseline haze/blur at t=1s, and a rule-based classification into
  HAZE  (bright veil: dark-channel/median rise, contrast falls)
  MURK  (darkening blur: mean+median fall)
  BLUR  (sharpness collapse vs siblings, tone stable)
  DARK-COLLAPSE (extreme darkening)
  clean
plus baseline_dirty (source video already blurry at 1s: lens water/smudge).

Writes fleet_pixscan.csv. Rules calibrated on the 12 labeled grids.
"""
import glob, os, sys
import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GRID_SRC = "/home/ashish/ARRWM/grids/grids_A/A"
OUT = os.path.join(HERE, "fleet_pixscan.csv")
FPS = 16
POS = {"pca8": (0, 0), "pca4": (832, 0), "pca2": (1664, 0), "16node": (2496, 0),
       "4node": (0, 480), "noatok": (832, 480), "noadaln": (1664, 480)}


def scan_grid(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)  # keep BGR; only intensity stats needed
    cap.release()
    if len(frames) < 80:
        return None
    frames = np.stack(frames)
    rows = []
    for v, (x, y) in POS.items():
        tile = frames[:, y + 32:y + 480, x:x + 832]
        gray = tile.mean(-1).astype(np.float32)
        mean_c = gray.reshape(len(gray), -1).mean(1)
        std_c = gray.reshape(len(gray), -1).std(1)
        med_c = np.median(gray.reshape(len(gray), -1), axis=1)
        t1, t4 = FPS, min(4 * FPS, len(gray) - 1)

        def dch(idx):
            return float(np.mean([cv2.erode(tile[i].min(-1), np.ones((15, 15), np.uint8)).mean() for i in idx]))

        def lap(idx):
            return float(np.mean([cv2.Laplacian(cv2.cvtColor(tile[i], cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() for i in idx]))

        base_haze = dch(range(t1, t1 + 4))
        base_blur = lap(range(t1, t1 + 4))
        end_haze = dch(range(len(tile) - 8, len(tile), 2))
        end_blur = lap(range(len(tile) - 8, len(tile), 2))
        rows.append({
            "grid": os.path.basename(path).replace("_grid.mp4", ""), "variant": v,
            "mean_d4s": round(float(mean_c[t4] - mean_c[0]), 1),
            "std_d4s": round(float(std_c[t4] - std_c[0]), 1),
            "median_d4s": round(float(med_c[t4] - med_c[0]), 1),
            "mean_dend": round(float(mean_c[-1] - mean_c[t1]), 1),
            "median_dend": round(float(med_c[-1] - med_c[t1]), 1),
            "std_dend": round(float(std_c[-1] - std_c[t1]), 1),
            "base_haze": round(base_haze, 1), "base_blur": round(base_blur, 1),
            "d_haze": round(end_haze - base_haze, 1), "d_blur": round(end_blur - base_blur, 1),
        })
    return rows


def classify(df):
    df["x_haze"] = df.groupby("grid").d_haze.transform(lambda s: s - s.median())
    df["x_blur"] = df.groupby("grid").d_blur.transform(lambda s: s - s.median())
    df["x_median"] = df.groupby("grid").median_dend.transform(lambda s: s - s.median())
    df["x_mean"] = df.groupby("grid").mean_dend.transform(lambda s: s - s.median())
    blur_lo = df.base_blur.quantile(0.12)
    df["baseline_dirty"] = df.base_blur < blur_lo

    def cls(r):
        if r.baseline_dirty:
            return "baseline_dirty"
        if r.mean_dend < -60:
            return "DARK-COLLAPSE"
        if (r.x_haze > 12 and r.x_median > 5) or (r.x_median > 10 and r.std_dend < -15):
            return "HAZE"
        if r.x_mean < -8 and r.x_median < -8:
            return "MURK"
        if r.x_blur < -0.5 * r.base_blur and abs(r.x_median) < 8:
            return "BLUR"
        return "clean"

    df["label"] = df.apply(cls, axis=1)
    return df


def main():
    files = sorted(glob.glob(os.path.join(GRID_SRC, "*_grid.mp4")))
    print(f"{len(files)} grid videos", flush=True)
    all_rows = []
    for i, fp in enumerate(files):
        r = scan_grid(fp)
        if r:
            all_rows.extend(r)
        if i % 10 == 0:
            print(f"{i}/{len(files)} {os.path.basename(fp)}", flush=True)
    df = pd.DataFrame(all_rows)
    df = classify(df)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(df)} tiles)")
    print(df.label.value_counts().to_string())


if __name__ == "__main__":
    main()
