"""Scene-relocation instrument (ORB+RANSAC verified inliers) on the stationary set.
For each rollout: best RANSAC-verified ORB inliers against the real reference (frame 8 of
real_rXX) OR the six-second-horizon end frame of any sibling model of the same scene. Relocated
when best < 50; static when first-gen vs end inliers > 600. CPU. Writes out/stat_scene.csv."""
import os
import cv2, numpy as np, pandas as pd

DIR = "/home/ashish/stationary_evaluation"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "stat_scene.csv")
MODELS = ["16node", "4node", "pca8", "pca4", "pca2", "noatok", "noadaln",
          "astra", "matrixgame", "minwm", "worldcam", "worldplay", "yume"]
CTX = {"16node": 12, "4node": 12, "pca8": 12, "pca4": 12, "pca2": 12, "noatok": 12, "noadaln": 12,
       "astra": 4, "matrixgame": 1, "minwm": 13, "worldcam": 65, "worldplay": 1, "yume": 1}
SIZE = (640, 352)
orb = cv2.ORB_create(3000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def frames_at(path, idxs):
    cap = cv2.VideoCapture(path); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); out = {}
    want = set(i for i in idxs if 0 <= i < n)
    i = 0
    while want:
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            out[i] = cv2.cvtColor(cv2.resize(f, SIZE), cv2.COLOR_BGR2GRAY); want.discard(i)
        i += 1
    cap.release(); return out, n


def meta(path):
    cap = cv2.VideoCapture(path); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or 16
    cap.release(); return n, fps


def inliers(a, b):
    ka, da = orb.detectAndCompute(a, None); kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return 0
    m = bf.match(da, db)
    if len(m) < 8:
        return 0
    src = np.float32([ka[x.queryIdx].pt for x in m]).reshape(-1, 1, 2)
    dst = np.float32([kb[x.trainIdx].pt for x in m]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    for scene in range(32):
        rp = os.path.join(DIR, f"real_r{scene:02d}.mp4")
        if not os.path.exists(rp):
            continue
        rf, _ = frames_at(rp, [8]); real_ref = rf.get(8)
        # collect each model's end frame (6s horizon) and first gen frame
        end_fr = {}; first_fr = {}
        for m in MODELS:
            fp = os.path.join(DIR, f"{m}_r{scene:02d}.mp4")
            if not os.path.exists(fp):
                continue
            n, fps = meta(fp); ctx = CTX[m]; end = min(n - 1, ctx + int(round(6.0 * fps)))
            fr, _ = frames_at(fp, [ctx, end])
            first_fr[m] = fr.get(ctx); end_fr[m] = fr.get(end)
        refs = ([real_ref] if real_ref is not None else [])
        for m in MODELS:
            if end_fr.get(m) is None:
                continue
            cand = refs + [end_fr[s] for s in MODELS if s != m and end_fr.get(s) is not None]
            best = max((inliers(end_fr[m], c) for c in cand), default=0)
            stat = inliers(first_fr[m], end_fr[m]) if first_fr.get(m) is not None else 0
            rows.append(dict(model=m, scene=scene, inliers=best, relocated=int(best < 50),
                             static=int(stat > 600)))
            print(f"{m}_r{scene:02d} inliers={best} static_inl={stat}", flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("wrote", OUT, len(rows))


if __name__ == "__main__":
    main()
