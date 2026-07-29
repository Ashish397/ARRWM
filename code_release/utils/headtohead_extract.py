"""Head-to-head extraction over phase-A videos (our 7 models + minWM).

Per video: robust motion (incremental-SIFT recession%, cum in-plane rot, cum tx)
+ frozen teacher read (CoTracker->PCA g0..g7 mean over gen chunks) — same
pixel-space judge for every model; the trained action critic is NOT used here.
Env: HH_MODELS colon list. Model 'minwm' reads logs/eval_final/A_minwm/minwm_rNN_DIR.mp4;
others logs/eval_final/A/<m>/control_test/step05000_rNN_DIR_raw.mp4.
Appends to analysis/eval_final/headtohead_motion.csv (one file per model shard via HH_OUT).
"""
import os, sys
import numpy as np, pandas as pd, imageio, cv2, torch
sys.path.insert(0, os.environ.get("AF_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("ARRWM_ACTION_ENCODER", "pca_raw")
from utils.ndof_following import load_pca, teacher_read_video, read_mp4

ARR = os.environ.get("AF_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = os.environ.get("HH_MODELS", "minwm").replace(":", ",").split(",")
OUT = os.environ.get("HH_OUT", f"{ARR}/analysis/eval_final/headtohead_motion.csv")
DIRS = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]
det = cv2.SIFT_create(); bf = cv2.BFMatcher()


def vid_path(m, wi, d):
    if m == "minwm":
        return f"{ARR}/logs/eval_final/A_minwm/minwm_r{wi:02d}_{d}.mp4"
    if m == "matrixgame":
        return f"{ARR}/logs/eval_final/A_matrixgame/matrixgame_r{wi:02d}_{d}.mp4"
    if m == "worldcam":
        return f"{ARR}/logs/eval_final/A_worldcam/worldcam_r{wi:02d}_{d}.mp4"
    if m == "yume":
        return f"{ARR}/logs/eval_final/A_yume/yume_r{wi:02d}_{d}.mp4"
    if m == "worldplay":
        return f"{ARR}/logs/eval_final/A_worldplay/worldplay_r{wi:02d}_{d}.mp4"
    if m == "astra":
        return f"{ARR}/logs/eval_final/A_astra/astra_r{wi:02d}_{d}.mp4"
    return f"{ARR}/logs/eval_final/A/{m}/control_test/step05000_r{wi:02d}_{d}_raw.mp4"


def robust(frames, stride=3):
    logs, rots, txs = [], [], []
    prev = None
    for i in range(0, len(frames), stride):
        g = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        if prev is not None:
            k0, d0 = det.detectAndCompute(prev, None); kt, dt = det.detectAndCompute(g, None)
            if d0 is not None and dt is not None:
                pairs = bf.knnMatch(d0, dt, k=2)
                good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
                if len(good) >= 8:
                    src = np.float32([k0[x.queryIdx].pt for x in good]).reshape(-1, 1, 2)
                    dst = np.float32([kt[x.trainIdx].pt for x in good]).reshape(-1, 1, 2)
                    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3)
                    if M is not None:
                        s = float(np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2))
                        if 0.5 < s < 2.0:
                            logs.append(np.log(s))
                            rots.append(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
                            txs.append(float(M[0, 2]))
        prev = g
    return (100 * (1 - float(np.exp(np.sum(logs)))) if logs else np.nan,
            float(np.sum(rots)) if rots else np.nan,
            float(np.sum(txs)) if txs else np.nan)


def main():
    mean, comp_T, scales = load_pca()
    cot = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to("cuda").eval()
    for p in cot.parameters():
        p.requires_grad_(False)
    rows = []
    for m in MODELS:
        for wi in range(32):
            for d in DIRS:
                p = vid_path(m, wi, d)
                if not os.path.exists(p):
                    continue
                try:
                    vid = read_mp4(p)
                    frames = vid[0].permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
                    rec, rot, tx = robust(list(frames))
                    g8 = teacher_read_video(vid, cot, mean, comp_T, scales)[1:].cpu().numpy()
                    gm = np.nanmean(g8, axis=0)
                    rows.append(dict(model=m, window=wi, dir=d, recession=round(rec, 2),
                                     rot=round(rot, 2), tx=round(tx, 1),
                                     **{f"g{i}": round(float(gm[i]), 4) for i in range(8)}))
                except Exception as e:
                    print(f"skip {m} r{wi:02d} {d}: {str(e)[:80]}", flush=True)
            print(f"[hh] {m} w{wi} done", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"saved {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
