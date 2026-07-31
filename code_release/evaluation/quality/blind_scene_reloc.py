"""Scene-relocation scan on blind100 (CPU, ORB + RANSAC homography inliers).

Per video: inliers between a real-context reference frame and the 6s-horizon end
frame. LOW inliers = the generated place no longer matches the real scene = relocated.
Reference = frame min(8, ctx-1) (a real context frame); end = min(n-2, ctx+6s).

Writes out/blind_scene_reloc.csv: blind_id,vid,model,scene,inliers,end_idx,ref_idx.
"""
import os
import cv2, numpy as np, pandas as pd
import blind100_common as bc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "blind_scene_reloc.csv")
orb = cv2.ORB_create(3000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def frame_at(path, i):
    c = cv2.VideoCapture(path); c.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, f = c.read(); c.release()
    return f if ok else None


def meta(path):
    c = cv2.VideoCapture(path); fps = c.get(cv2.CAP_PROP_FPS) or 16
    n = int(c.get(cv2.CAP_PROP_FRAME_COUNT)); c.release()
    return fps, n


def inliers(a, b):
    ga, gb = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    k0, d0 = orb.detectAndCompute(ga, None); k1, d1 = orb.detectAndCompute(gb, None)
    if d0 is None or d1 is None or len(k0) < 8 or len(k1) < 8:
        return 0
    ms = bf.match(d0, d1)
    if len(ms) < 8:
        return 0
    src = np.float32([k0[m.queryIdx].pt for m in ms])
    dst = np.float32([k1[m.trainIdx].pt for m in ms])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    for r in bc.refs():
        fps, n = meta(r["path"])
        ref_idx = max(0, min(8, r["ctx"] - 1))
        end_idx = min(n - 2, r["ctx"] + int(round(6.0 * fps)))
        a, b = frame_at(r["path"], ref_idx), frame_at(r["path"], end_idx)
        if a is None or b is None:
            print(f"[scene] {r['blind_id']} unreadable frames"); continue
        inl = inliers(cv2.resize(a, (640, 352)), cv2.resize(b, (640, 352)))
        rows.append(dict(blind_id=r["blind_id"], vid=r["vid"], model=r["model"],
                         scene=r["scene"], inliers=inl, ref_idx=ref_idx, end_idx=end_idx))
        print(f"[scene] {r['blind_id']} {r['vid']:20s} inliers={inl}", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"[scene] wrote {OUT} ({len(rows)} videos)")


if __name__ == "__main__":
    main()
