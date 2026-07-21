"""Camera attitude (pitch/roll) drift metric — image-based, ground-plane derived.

Per frame: DepthAnything-V2-Small depth -> unproject lower 45% (nominal
intrinsics) -> robust plane fit -> ground normal -> pitch/roll vs seed
reference. Per video:
  pitch_end/roll_end = |mean(last 4 sampled frames) - seed ref|  (UNRECOVERED drift = failure)
  pitch_max/roll_max = max |excursion|                            (allowed if it reconciles)
User spec: transient attitude moves that reconcile with ground texture are OK;
monotonic/unrecovered pitch or roll drift is the failure mode the critic misses.

Env: AD_RUNS colon list ('REAL' = real refs), AD_WINDOWS, AD_STRIDE (def 6), AD_OUT.
"""
import os, glob
import numpy as np
import pandas as pd
import imageio
import torch

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
DEV = "cuda"
RUNS = os.environ.get("AD_RUNS", "pca8_8node:pca4:pca2:16node:4node:noatok:noadaln").split(":")
WINDOWS = os.environ.get("AD_WINDOWS", "r08_B:r08_BL:r01_R:r02_R").split(":")
STRIDE = int(os.environ.get("AD_STRIDE", "6"))
OUT = os.environ.get("AD_OUT", f"{ARR}/analysis/eval_final/attitude_drift.csv")

from transformers import pipeline as hf_pipeline
depth_pipe = hf_pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf",
                         device=0, torch_dtype=torch.float16)


def attitude(img):
    """img (H,W,3) -> (pitch_proxy, roll_deg) from ground-region disparity plane.

    v-disparity principle: for a planar ground, disparity d(x,y) = a*x + b*y + c.
    roll = rotation of the disparity gradient away from vertical = atan2(a, b);
    pitch proxy = vertical disparity slope b (monotone in camera pitch).
    Relative-to-seed deltas only; absolute calibration not needed for drift.
    """
    from PIL import Image
    H, W = img.shape[:2]
    d = np.asarray(depth_pipe(Image.fromarray(img))["predicted_depth"]).astype(np.float64)
    if d.shape != (H, W):
        import cv2
        d = cv2.resize(d, (W, H))
    ys, xs = np.mgrid[0:H, 0:W]
    m = (ys > 0.55 * H) & (xs > 0.1 * W) & (xs < 0.9 * W)
    x = (xs[m] - W / 2) / W
    y = (ys[m] - H / 2) / H
    dd = d[m]
    dd = (dd - dd.mean()) / (dd.std() + 1e-9)
    A = np.stack([x, y, np.ones_like(x)], 1)
    idx = np.random.RandomState(0).choice(len(A), min(6000, len(A)), replace=False)
    A, dd = A[idx], dd[idx]
    w = np.ones(len(A))
    coef = None
    for _ in range(3):
        coef, *_ = np.linalg.lstsq(A * w[:, None], dd * w, rcond=None)
        r = np.abs(A @ coef - dd)
        sc = np.median(r) + 1e-9
        w = 1.0 / (1.0 + (r / (3 * sc)) ** 2)
    a, b, _ = coef
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan"), float("nan")
    roll = np.degrees(np.arctan2(a, b + 1e-9))
    pitch = float(b)          # unitless slope proxy; deltas vs seed are the signal
    return pitch, float(roll)


def video_attitude(path, start_ref_n=3):
    r = imageio.get_reader(path)
    fr = [np.asarray(f) for f in r]
    r.close()
    seed_idx = [0, 4, 8][:start_ref_n]
    gen_idx = list(range(13, len(fr), STRIDE))
    ref = np.nanmean(np.array([attitude(fr[i]) for i in seed_idx]), 0)
    traj = np.array([attitude(fr[i]) for i in gen_idx]) - ref
    endv = np.nanmean(traj[-4:], 0) if len(traj) >= 4 else np.nanmean(traj, 0)
    return dict(pitch_end=round(abs(float(endv[0])), 2), roll_end=round(abs(float(endv[1])), 2),
                pitch_max=round(float(np.nanmax(np.abs(traj[:, 0]))), 2),
                roll_max=round(float(np.nanmax(np.abs(traj[:, 1]))), 2))


_MWSW = {"L": "R", "R": "L", "FL": "FR", "FR": "FL", "BL": "BR", "BR": "BL"}
_COMPARATORS = ("minwm", "matrixgame", "worldcam", "yume", "worldplay", "astra")
DIRS8 = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]


def vid_path(run, w):
    # w like "r08_BL"; minwm disk labels are yaw-sign-flipped -> swap to TRUE dir
    rank, d = w.split("_", 1)
    if run == "minwm":
        return f"{ARR}/logs/eval_final/A_minwm/minwm_{rank}_{_MWSW.get(d, d)}.mp4"
    if run in _COMPARATORS:
        return f"{ARR}/logs/eval_final/A_{run}/{run}_{rank}_{d}.mp4"
    return f"{ARR}/logs/eval_final/A/{run}/control_test/step05000_{w}_raw.mp4"


def main():
    global WINDOWS
    if WINDOWS == ["ALL"]:
        WINDOWS = [f"r{wi:02d}_{d}" for wi in range(32) for d in DIRS8]
    rows, done = [], set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(zip(prev.window, prev.run))
        rows = prev.to_dict("records")
        print(f"[att] resuming: {len(done)} already scored", flush=True)
    fout = open(OUT, "a" if done else "w")
    if not done:
        fout.write("window,run,pitch_end,roll_end,pitch_max,roll_max\n")
        fout.flush()
    for run in RUNS:
        if run == "REAL":
            n = 0
            for p in sorted(glob.glob(f"{ARR}/analysis/eval_final/real_refs/*.mp4")):
                r = imageio.get_reader(p); f0 = np.asarray(r.get_data(5)); r.close()
                if float(f0.mean()) < 60:
                    continue
                rows.append(dict(window="REAL", run=os.path.basename(p)[:22], **video_attitude(p)))
                print(f"[att] REAL {os.path.basename(p)[:22]}: {rows[-1]}", flush=True)
                n += 1
                if n >= 6:
                    break
            continue
        for w in WINDOWS:
            if (w, run) in done:
                continue
            p = vid_path(run, w)
            if not os.path.exists(p):
                continue
            try:
                v = video_attitude(p)
            except Exception as e:
                print(f"[att] skip {w} {run}: {str(e)[:80]}", flush=True)
                continue
            rows.append(dict(window=w, run=run, **v))
            fout.write(f"{w},{run},{v['pitch_end']},{v['roll_end']},{v['pitch_max']},{v['roll_max']}\n")
            fout.flush()
            print(f"[att] {w} {run}: {rows[-1]}", flush=True)
    fout.close()
    pd.DataFrame(rows).to_csv(OUT, index=False)
    df = pd.DataFrame(rows)
    print("\n=== unrecovered drift ranking (pitch_end + roll_end desc) ===")
    df["drift"] = df.pitch_end + df.roll_end
    for w in df.window.unique():
        s = df[df.window == w].sort_values("drift", ascending=False)
        print(w, ":", " ".join(f"{r.run}:{r.drift:.1f}" for r in s.itertuples()))


if __name__ == "__main__":
    main()
