"""Pixel-statistics tracker: per-frame mean, std, median of all pixels over time.

For each tile video: curves of mean/std/median pixel intensity (0-255, gray and
per-channel mean), plus the delta from t=0 to t=4s (frame 64 @16fps) and to the
end. Writes results_pixstats.csv (metric rows) and pixstat_curves.csv (full curves).
"""
import glob, os
import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_METRICS = os.path.join(HERE, "results_pixstats.csv")
OUT_CURVES = os.path.join(HERE, "pixstat_curves.csv")
FPS = 16
T4 = 4 * FPS  # frame index at 4 seconds


def read_frames(path):
    cap = cv2.VideoCapture(path)
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(fr)


def main():
    files = sorted(set(glob.glob(os.path.join(HERE, "tiles", "*.mp4")) +
                       glob.glob(os.path.join(HERE, "tiles_new", "*.mp4"))))
    seen = set()
    mrows, crows = [], []
    for fp in files:
        name = os.path.basename(fp)[: -len(".mp4")]
        if name in seen:
            continue
        seen.add(name)
        grid, variant = name.split("__")
        frames = read_frames(fp).astype(np.float32)
        gray = frames.mean(-1)  # T,H,W
        mean_c = gray.reshape(len(gray), -1).mean(1)
        std_c = gray.reshape(len(gray), -1).std(1)
        med_c = np.median(gray.reshape(len(gray), -1), axis=1)
        for t in range(len(gray)):
            crows.append({"grid": grid, "variant": variant, "t_sec": round(t / FPS, 3),
                          "mean": round(float(mean_c[t]), 2), "std": round(float(std_c[t]), 2),
                          "median": round(float(med_c[t]), 2)})
        t4 = min(T4, len(gray) - 1)
        m = {
            "px_mean_t0": float(mean_c[0]), "px_std_t0": float(std_c[0]), "px_median_t0": float(med_c[0]),
            "px_mean_d4s": float(mean_c[t4] - mean_c[0]),      # delta 0 -> 4s
            "px_std_d4s": float(std_c[t4] - std_c[0]),
            "px_median_d4s": float(med_c[t4] - med_c[0]),
            "px_mean_dend": float(mean_c[-1] - mean_c[0]),
            "px_std_dend": float(std_c[-1] - std_c[0]),
            "px_median_dend": float(med_c[-1] - med_c[0]),
            # drift magnitude: max absolute excursion of each stat within first 4s
            "px_mean_maxdev4s": float(np.abs(mean_c[:t4 + 1] - mean_c[0]).max()),
            "px_std_maxdev4s": float(np.abs(std_c[:t4 + 1] - std_c[0]).max()),
        }
        for k, v in m.items():
            mrows.append({"grid": grid, "variant": variant, "metric": k, "value": round(v, 3)})
        print(f"{name}: mean {mean_c[0]:.1f}->{mean_c[t4]:.1f} (d4s {m['px_mean_d4s']:+.1f})  "
              f"std {std_c[0]:.1f}->{std_c[t4]:.1f} (d4s {m['px_std_d4s']:+.1f})  "
              f"median d4s {m['px_median_d4s']:+.1f}", flush=True)

    md = pd.DataFrame(mrows)
    if os.path.exists(OUT_METRICS):
        old = pd.read_csv(OUT_METRICS)
        old = old[~old.metric.isin(md.metric.unique())]
        md = pd.concat([old, md], ignore_index=True)
    md.to_csv(OUT_METRICS, index=False)
    pd.DataFrame(crows).to_csv(OUT_CURVES, index=False)
    print(f"wrote {OUT_METRICS} and {OUT_CURVES}")


if __name__ == "__main__":
    main()
