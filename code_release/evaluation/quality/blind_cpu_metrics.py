"""CPU competitor panels on blind100 (no GPU): haze/high-freq, static/stillness,
and the relocation-competitor variants (raw ORB, RANSAC inliers).

Per video, comparing a real-context anchor frame vs late generated frames:
  HAZE / high-freq (start ctx window vs end window):
    hf_lap_drift    : Laplacian-variance sharpness LOSS start->end (haze/blur)
    hf_darkchan_drift: dark-channel veiling INCREASE (haze prior)
    hf_contrast_drift: RMS-contrast LOSS
  STATIC / stillness (adjacent generated frames):
    static_absdiff  : mean |frame(t+1)-frame(t)| over generation (low=static)
    static_ncc      : mean NCC of adjacent generated frames (high=static)
  RELOCATION competitors (ref ctx frame vs end frame):
    orb_inliers_ransac : ORB matches surviving RANSAC homography (low=relocated)
    orb_raw_matches    : raw ORB good matches, no RANSAC (the weaker baseline)
Writes out/blind_cpu_metrics.csv.
"""
import os
import cv2, numpy as np, pandas as pd
import blind100_common as bc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "blind_cpu_metrics.csv")
orb = cv2.ORB_create(3000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def read_all(path):
    c = cv2.VideoCapture(path); fps = c.get(cv2.CAP_PROP_FPS) or 16; fr = []
    while True:
        ok, f = c.read()
        if not ok:
            break
        fr.append(cv2.resize(f, (640, 352)))     # BGR
    c.release(); return fr, fps


def sharp(g):        return cv2.Laplacian(g, cv2.CV_64F).var()
def darkchan(bgr):   return cv2.erode(bgr.min(2), np.ones((15, 15), np.uint8)).mean()
def contrast(g):     return float(g.std())


def orb_counts(a, b):
    ga, gb = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    k0, d0 = orb.detectAndCompute(ga, None); k1, d1 = orb.detectAndCompute(gb, None)
    if d0 is None or d1 is None or len(k0) < 8 or len(k1) < 8:
        return 0, 0
    ms = bf.match(d0, d1)
    if len(ms) < 8:
        return len(ms), 0
    src = np.float32([k0[m.queryIdx].pt for m in ms]); dst = np.float32([k1[m.trainIdx].pt for m in ms])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return len(ms), (int(mask.sum()) if mask is not None else 0)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    for r in bc.refs():
        fr, fps = read_all(r["path"]); n = len(fr); ctx = r["ctx"]
        ref_i = max(0, min(8, ctx - 1))
        end_i = min(n - 2, ctx + int(round(6.0 * fps)))
        cs = [max(0, min(ctx - 1, ctx // 2))]                 # context anchor frame(s)
        es = list(range(max(ctx, n - 16), n))                 # end window
        gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in fr]
        # high-freq / haze drift (anchor vs end)
        lap_c = np.mean([sharp(gray[i]) for i in cs]); lap_e = np.mean([sharp(gray[i]) for i in es])
        dc_c = np.mean([darkchan(fr[i]) for i in cs]); dc_e = np.mean([darkchan(fr[i]) for i in es])
        ct_c = np.mean([contrast(gray[i]) for i in cs]); ct_e = np.mean([contrast(gray[i]) for i in es])
        # static / stillness over generation
        gidx = list(np.linspace(ctx, n - 1, 10).round().astype(int))
        ad = [np.abs(fr[a].astype(np.float32) - fr[b].astype(np.float32)).mean()
              for a, b in zip(gidx[:-1], gidx[1:])]
        nccs = []
        for a, b in zip(gidx[:-1], gidx[1:]):
            x, y = gray[a].astype(np.float32).ravel(), gray[b].astype(np.float32).ravel()
            x, y = x - x.mean(), y - y.mean()
            nccs.append(float((x @ y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9)))
        raw, ransac = orb_counts(fr[ref_i], fr[end_i])
        rows.append(dict(blind_id=r["blind_id"], vid=r["vid"], model=r["model"], scene=r["scene"],
                         hf_lap_drift=float(lap_c - lap_e), hf_darkchan_drift=float(dc_e - dc_c),
                         hf_contrast_drift=float(ct_c - ct_e),
                         static_absdiff=float(np.mean(ad)), static_ncc=float(np.mean(nccs)),
                         orb_inliers_ransac=ransac, orb_raw_matches=raw))
        print(f"[cpu] {r['blind_id']} {r['vid']:20s} lapd={lap_c-lap_e:8.1f} "
              f"absdiff={np.mean(ad):6.2f} ransac={ransac} raw={raw}", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"[cpu] wrote {OUT} ({len(rows)})")


if __name__ == "__main__":
    main()
