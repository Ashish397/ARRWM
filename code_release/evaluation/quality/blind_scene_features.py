"""Scene-relocation feature-detector bakeoff on blind100 (CPU): ORB vs SIFT vs AKAZE,
all with RANSAC-homography inlier counting. Same ref/end frames as blind_scene_reloc.
LOW inliers = place identity lost = relocated. Each detector tuned to give its best:
  ORB   : 3000 features, BF-Hamming crossCheck
  SIFT  : all features, BF-L2 kNN + Lowe ratio 0.75
  AKAZE : default, BF-Hamming crossCheck
Writes out/blind_scene_features.csv with <det>_inliers and <det>_raw per detector.
"""
import os
import cv2, numpy as np, pandas as pd
import blind100_common as bc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "blind_scene_features.csv")

orb = cv2.ORB_create(3000)
sift = cv2.SIFT_create()
akaze = cv2.AKAZE_create()
bf_h = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
bf_l2 = cv2.BFMatcher(cv2.NORM_L2)          # kNN + ratio for SIFT


def frame_at(path, i):
    c = cv2.VideoCapture(path); c.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, f = c.read(); c.release()
    return f if ok else None


def meta(path):
    c = cv2.VideoCapture(path); fps = c.get(cv2.CAP_PROP_FPS) or 16
    n = int(c.get(cv2.CAP_PROP_FRAME_COUNT)); c.release()
    return fps, n


def ransac_inliers(src, dst):
    if len(src) < 8:
        return len(src), 0
    H, mask = cv2.findHomography(np.float32(src), np.float32(dst), cv2.RANSAC, 5.0)
    return len(src), (int(mask.sum()) if mask is not None else 0)


def score(det, a, b):
    ga, gb = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    if det == "orb":
        k0, d0 = orb.detectAndCompute(ga, None); k1, d1 = orb.detectAndCompute(gb, None)
        if d0 is None or d1 is None or len(k0) < 8 or len(k1) < 8:
            return 0, 0
        ms = bf_h.match(d0, d1)
        pts = [(k0[m.queryIdx].pt, k1[m.trainIdx].pt) for m in ms]
    elif det == "akaze":
        k0, d0 = akaze.detectAndCompute(ga, None); k1, d1 = akaze.detectAndCompute(gb, None)
        if d0 is None or d1 is None or len(k0) < 8 or len(k1) < 8:
            return 0, 0
        ms = bf_h.match(d0, d1)
        pts = [(k0[m.queryIdx].pt, k1[m.trainIdx].pt) for m in ms]
    else:  # sift + Lowe ratio
        k0, d0 = sift.detectAndCompute(ga, None); k1, d1 = sift.detectAndCompute(gb, None)
        if d0 is None or d1 is None or len(k0) < 8 or len(k1) < 8:
            return 0, 0
        good = []
        for pair in bf_l2.knnMatch(d0, d1, k=2):
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                good.append(pair[0])
        pts = [(k0[m.queryIdx].pt, k1[m.trainIdx].pt) for m in good]
    if len(pts) < 8:
        return len(pts), 0
    return ransac_inliers([p[0] for p in pts], [p[1] for p in pts])


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    for r in bc.refs():
        fps, n = meta(r["path"])
        ref_i = max(0, min(8, r["ctx"] - 1)); end_i = min(n - 2, r["ctx"] + int(round(6.0 * fps)))
        a, b = frame_at(r["path"], ref_i), frame_at(r["path"], end_i)
        if a is None or b is None:
            continue
        a, b = cv2.resize(a, (640, 352)), cv2.resize(b, (640, 352))
        row = dict(blind_id=r["blind_id"], vid=r["vid"], model=r["model"], scene=r["scene"])
        for det in ("orb", "sift", "akaze"):
            raw, inl = score(det, a, b)
            row[f"{det}_raw"] = raw; row[f"{det}_inliers"] = inl
        rows.append(row)
        print(f"[feat] {r['blind_id']} {r['vid']:20s} "
              f"orb={row['orb_inliers']} sift={row['sift_inliers']} akaze={row['akaze_inliers']}", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"[feat] wrote {OUT} ({len(rows)})")


if __name__ == "__main__":
    main()
