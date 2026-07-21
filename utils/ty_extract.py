"""Vertical-translation (ty) sweep over ALL phase-A videos, all models.

Same incremental SIFT similarity fit as headtohead_extract.robust(), recording
the cumulative VERTICAL translation M[1,2] — the pitch analog of tx: the scene
sliding down the image = camera pitching up, and vice versa. Summed over the
whole clip like tx/rot (path quantity, signed).

Both tx and ty are stored as % of frame WIDTH/HEIGHT respectively (models
render at different resolutions; raw pixels are not comparable on a shared
wedge scale).

CPU-only (cv2), multiprocessing. Resume-safe append to TY_OUT
(default analysis/eval_final/ty_motion.csv: model,window,dir,tx_pct,ty_pct).
Env: TY_MODELS colon list, TY_WORKERS (def 28).
"""
import os, sys
import numpy as np
import multiprocessing as mp

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
MODELS = os.environ.get(
    "TY_MODELS",
    "pca8_8node:pca4:pca2:16node:4node:noatok:noadaln:minwm:matrixgame:worldcam:yume:worldplay:astra"
).split(":")
OUT = os.environ.get("TY_OUT", f"{ARR}/analysis/eval_final/ty_motion.csv")
WORKERS = int(os.environ.get("TY_WORKERS", "28"))
DIRS = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]
_MWSW = {"L": "R", "R": "L", "FL": "FR", "FR": "FL", "BL": "BR", "BR": "BL"}
_COMPARATORS = ("minwm", "matrixgame", "worldcam", "yume", "worldplay", "astra")


def vid_path(m, wi, d):
    if m == "minwm":   # disk labels yaw-sign-flipped; swap to TRUE direction
        return f"{ARR}/logs/eval_final/A_minwm/minwm_r{wi:02d}_{_MWSW.get(d, d)}.mp4"
    if m in _COMPARATORS:
        return f"{ARR}/logs/eval_final/A_{m}/{m}_r{wi:02d}_{d}.mp4"
    return f"{ARR}/logs/eval_final/A/{m}/control_test/step05000_r{wi:02d}_{d}_raw.mp4"


def one(job):
    m, wi, d = job
    import cv2, imageio
    p = vid_path(m, wi, d)
    if not os.path.exists(p):
        return (m, wi, d, None, None)
    try:
        r = imageio.get_reader(p)
        frames = [np.asarray(f) for f in r]
        r.close()
        H, W = frames[0].shape[:2]
        det = cv2.SIFT_create(); bf = cv2.BFMatcher()
        txs, tys, prev = [], [], None
        for i in range(0, len(frames), 3):
            g = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            if prev is not None:
                k0, d0 = det.detectAndCompute(prev, None)
                kt, dt = det.detectAndCompute(g, None)
                if d0 is not None and dt is not None:
                    pairs = bf.knnMatch(d0, dt, k=2)
                    good = [q[0] for q in pairs if len(q) == 2 and q[0].distance < 0.75 * q[1].distance]
                    if len(good) >= 8:
                        src = np.float32([k0[x.queryIdx].pt for x in good]).reshape(-1, 1, 2)
                        dst = np.float32([kt[x.trainIdx].pt for x in good]).reshape(-1, 1, 2)
                        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                           ransacReprojThreshold=3)
                        if M is not None:
                            s = float(np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2))
                            if 0.5 < s < 2.0:
                                txs.append(float(M[0, 2])); tys.append(float(M[1, 2]))
            prev = g
        tx = 100 * float(np.sum(txs)) / W if txs else None
        ty = 100 * float(np.sum(tys)) / H if tys else None
        return (m, wi, d, tx, ty)
    except Exception as e:
        print(f"[ty] fail {m} r{wi:02d} {d}: {str(e)[:80]}", flush=True)
        return (m, wi, d, None, None)


def main():
    done = set()
    if os.path.exists(OUT):
        import csv
        for r in csv.DictReader(open(OUT)):
            done.add((r["model"], int(r["window"]), r["dir"]))
        print(f"[ty] resuming: {len(done)} done", flush=True)
    jobs = [(m, wi, d) for m in MODELS for wi in range(32) for d in DIRS
            if (m, wi, d) not in done]
    print(f"[ty] {len(jobs)} videos to process, {WORKERS} workers", flush=True)
    f = open(OUT, "a" if done else "w")
    if not done:
        f.write("model,window,dir,tx_pct,ty_pct\n")
        f.flush()
    n = 0
    with mp.Pool(WORKERS) as pool:
        for m, wi, d, tx, ty in pool.imap_unordered(one, jobs, chunksize=4):
            if tx is not None:
                f.write(f"{m},{wi},{d},{tx:.2f},{ty:.2f}\n")
                f.flush()
            n += 1
            if n % 64 == 0:
                print(f"[ty] {n}/{len(jobs)}", flush=True)
    f.close()
    print("[ty] DONE", flush=True)


if __name__ == "__main__":
    main()
