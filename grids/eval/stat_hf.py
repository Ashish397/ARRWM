"""High-frequency degradation instrument (sibling-relative Laplacian-variance drift B) on the
stationary set. base = 4 frames at 1s into generation; end = 8 frames (stride 2) near the 6s
horizon; d_blur = end - base; B = -(d_blur - same-scene sibling median). CPU. Writes out/stat_hf.csv."""
import os
import cv2, numpy as np, pandas as pd

DIR = "/home/ashish/stationary_evaluation"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "stat_hf.csv")
MODELS = ["16node", "4node", "pca8", "pca4", "pca2", "noatok", "noadaln",
          "astra", "matrixgame", "minwm", "worldcam", "worldplay", "yume"]
CTX = {"16node": 12, "4node": 12, "pca8": 12, "pca4": 12, "pca2": 12, "noatok": 12, "noadaln": 12,
       "astra": 4, "matrixgame": 1, "minwm": 13, "worldcam": 65, "worldplay": 1, "yume": 1}
SIZE = (832, 448)


def lap_var(rgb):
    g = cv2.cvtColor(cv2.resize(rgb, SIZE), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def frames_at(path, idxs):
    cap = cv2.VideoCapture(path); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); want = {i for i in idxs if 0 <= i < n}
    got = {}; i = 0
    while len(got) < len(want):
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            got[i] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release(); return got, n


def meta(path):
    cap = cv2.VideoCapture(path); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or 16
    cap.release(); return n, fps


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    for scene in range(32):
        for m in MODELS:
            fp = os.path.join(DIR, f"{m}_r{scene:02d}.mp4")
            if not os.path.exists(fp):
                continue
            n, fps = meta(fp); ctx = CTX[m]
            b0 = ctx + int(round(fps))
            end = min(n - 1, ctx + int(round(6.0 * fps)))
            base_idx = [i for i in range(b0, b0 + 4) if i < n]
            end_idx = list(range(max(ctx, end - 14), end + 1, 2))
            if not base_idx or not end_idx:
                continue
            got, _ = frames_at(fp, base_idx + end_idx)
            base = np.mean([lap_var(got[i]) for i in base_idx if i in got])
            endv = np.mean([lap_var(got[i]) for i in end_idx if i in got])
            rows.append(dict(model=m, scene=scene, base_blur=round(base, 1), end_blur=round(endv, 1),
                             d_blur=round(endv - base, 1)))
            print(f"{m}_r{scene:02d} d_blur={endv-base:.0f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)
    df = pd.DataFrame(rows)
    df["x_blur"] = df.groupby("scene").d_blur.transform(lambda s: s - s.median())
    df["B"] = -df["x_blur"]
    df["hf_flag"] = (df.B > 150).astype(int)
    df.to_csv(OUT, index=False)
    print("wrote", OUT, len(df))


if __name__ == "__main__":
    main()
