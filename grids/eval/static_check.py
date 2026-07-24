"""Staticness metric: similarity between the first generated frame and the frame
at the 6s-generation horizon. A model that does not move (or generates nothing)
keeps this similarity high and would otherwise sail through the plausibility checks.

Per video:
  static_ncc    normalized cross-correlation of gray first-vs-horizon frame
  static_absdiff mean |first - horizon| (0-255, low = static)
CPU-only. Writes results_external_static.csv (scene, model, metric, value).
"""
import glob, json, os, sys
import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = "/home/ashish/ARRWM/grids/baselines"
MODELS = ["astra", "matrixgame", "minwm", "worldcam", "worldplay", "yume"]
CTX_FRAMES = {"astra": 4, "matrixgame": 1, "minwm": 13, "worldcam": 65, "worldplay": 1, "yume": 1}
OURS = ["pca8", "16node"]
OUT = os.path.join(HERE, "results_external_static.csv")


def first_last(path, ctx_frames):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 16
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    i_first = ctx_frames                                      # first generated frame
    i_last = min(n - 2, ctx_frames + int(round(6.0 * fps)))  # 6s-generation horizon
    frames = {}
    for i in (i_first, i_last):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if not ok:
            cap.release()
            return None
        frames[i] = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (416, 224)).astype(np.float32)
    cap.release()
    a, b = frames[i_first], frames[i_last]
    an, bn = a - a.mean(), b - b.mean()
    ncc = float((an * bn).sum() / (np.sqrt((an ** 2).sum() * (bn ** 2).sum()) + 1e-9))
    return ncc, float(np.abs(a - b).mean())


def main():
    scenes = sys.argv[1:] or sorted(set(
        os.path.basename(p).split("_", 1)[1].replace(".mp4", "")
        for p in glob.glob(os.path.join(BASE_DIR, "A_astra", "*.mp4"))))
    rows = []
    for scene in scenes:
        for m in MODELS:
            fp = os.path.join(BASE_DIR, f"A_{m}", f"{m}_{scene}.mp4")
            if os.path.exists(fp):
                r = first_last(fp, CTX_FRAMES[m])
                if r:
                    rows += [{"scene": scene, "model": m, "metric": "static_ncc", "value": round(r[0], 4)},
                             {"scene": scene, "model": m, "metric": "static_absdiff", "value": round(r[1], 2)}]
        for v in OURS:
            for tdir in ("tiles", "tiles_new"):
                fp = os.path.join(HERE, tdir, f"{scene}__{v}.mp4")
                if os.path.exists(fp):
                    r = first_last(fp, 9)
                    if r:
                        rows += [{"scene": scene, "model": f"ours_{v}", "metric": "static_ncc", "value": round(r[0], 4)},
                                 {"scene": scene, "model": f"ours_{v}", "metric": "static_absdiff", "value": round(r[1], 2)}]
                    break
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(rows)} rows, {len(scenes)} scenes)")


if __name__ == "__main__":
    main()
