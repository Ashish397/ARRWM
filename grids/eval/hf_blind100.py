"""High-frequency degradation metric (B) for the blind100 set.

Faithful port of grids/eval/fleet_pixscan.py's HF metric, with the TWO
adaptations required because blind100 is heterogeneous (unlike our ablation
grid, which was uniform 832x448 / 16fps / 12-frame context):

  1. RESIZE every frame to 832x448 before the Laplacian. Laplacian variance
     scales with resolution and content; the medians and the B>150 threshold
     were calibrated at 832x448. Without a common resize the sibling-relative
     subtraction compares incomparable magnitudes.
  2. BASE window is 1s into GENERATION (ctx + round(fps)), not absolute frame
     16. For long-context models (worldcam ctx=65) absolute frame 16 lands
     inside the real context, not the generated part.

Metric definition (unchanged from fleet_pixscan.py):
  lap(f)      = variance of the 3x3 Laplacian of the grayscale frame
  base_blur   = mean lap over 4 frames from 1s-into-generation
  end_blur    = mean lap over the last 8 frames, stride 2
  d_blur      = end_blur - base_blur
  x_blur      = d_blur - median(d_blur over same-scene siblings)   # sibling-relative
  B           = -x_blur            # positive = MORE sharpness lost than siblings
  baseline_dirty = base_blur < 12th-percentile(base_blur) over the fleet
  flag        = B > 150            # descriptive threshold (see caveats)

Sibling group = all rollouts of the SAME scene (across models). Dirty-source
rollouts are excluded from reported summaries.

Edit CTX / the CSV column names / the video-path resolver to match blind100.
"""
import os, glob
import cv2, numpy as np, pandas as pd

BLIND_DIR = "/home/ashish/blind100"
LABELS = "/home/ashish/blind100_labels_and_scores.csv"   # maps blind_id/vid -> model, scene
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "blind_hf.csv")
CANON = (832, 448)   # common resolution for the Laplacian (do not change)

# first generated-frame index per model (real-context length in the saved clip)
CTX = {"pca8":12,"pca4":12,"pca2":12,"16node":12,"4node":12,"noatok":12,"noadaln":12,
       "astra":4,"matrixgame":1,"minwm":13,"worldcam":65,"worldplay":1,"yume":1}


def lap_var(frame_bgr):
    g = cv2.cvtColor(cv2.resize(frame_bgr, CANON), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def hf_one(path, ctx):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 16
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    n = len(frames)
    b0 = ctx + int(round(fps))               # 1s into generation
    if b0 + 4 >= n - 8:                        # too short for distinct base/end windows
        return None
    base = np.mean([lap_var(frames[i]) for i in range(b0, b0 + 4)])
    end = np.mean([lap_var(frames[i]) for i in range(n - 8, n, 2)])
    return {"base_blur": base, "end_blur": end, "d_blur": end - base, "fps": fps, "n": n}


def main():
    lab = pd.read_csv(LABELS)   # expected columns include: vid (or blind_id), model, scene
    rows = []
    for _, r in lab.iterrows():
        bid = r["blind_id"]; vid = r["vid"]        # blind_id = filename (V001.mp4); vid = descriptive
        model = r["model"]
        path = os.path.join(BLIND_DIR, f"{bid}.mp4")
        if not os.path.exists(path) or model not in CTX:
            continue
        m = hf_one(path, CTX[model])
        if m is None:
            continue
        rows.append({"blind_id": bid, "vid": vid, "model": model, "scene": r["scene"], **m})
    df = pd.DataFrame(rows)
    # sibling-relative (per scene, across models) and dirty flag (fleet 12th pct)
    df["x_blur"] = df.groupby("scene").d_blur.transform(lambda s: s - s.median())
    df["B"] = -df.x_blur
    df["baseline_dirty"] = df.base_blur < df.base_blur.quantile(0.12)
    df["hf_flag"] = ((~df.baseline_dirty) & (df.B > 150)).astype(int)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(df)} rollouts)")
    clean = df[~df.baseline_dirty]
    print(clean.groupby("model").agg(median_B=("B", "median"),
          pct_B_gt150=("B", lambda s: round(100 * (s > 150).mean()))).to_string())


if __name__ == "__main__":
    main()
