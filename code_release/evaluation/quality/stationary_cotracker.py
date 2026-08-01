"""CoTracker motion + signs-of-life on the stationary (no-op) set.

Per rollout: track a grid of points over the generated segment, fit the dominant
coherent motion (RANSAC affine start->end) = the camera/background transform, and
count tracks whose motion does NOT fit it (residual > thresh) = independent movers
= signs of life. camera_motion = median inlier displacement (should be ~0 for a good
no-op). Compares every model against the real reference.

Writes out/stationary_cotracker.csv.
"""
import os, glob
import numpy as np, torch, cv2, imageio, pandas as pd
from scipy.spatial import cKDTree

DIR = os.environ.get("STATIONARY_DIR",
                     os.path.join(os.environ.get("AF_ROOT", "."), "stationary_evaluation"))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "stationary_cotracker.csv")
CTX = {"astra": 4, "matrixgame": 1, "minwm": 13, "worldcam": 65, "worldplay": 1, "yume": 1}
T, GRID, SIZE, RESID = 24, 30, (512, 288), 4.0


def ctx_of(m):
    return CTX.get(m, 12)          # ours + real = 12


def read_gen(path, ctx):
    r = imageio.get_reader(path); n = r.count_frames()
    idx = np.linspace(min(ctx, n - 2), n - 1, T).round().astype(int)
    fr = [cv2.resize(np.asarray(r.get_data(int(i))), SIZE) for i in idx]
    r.close()
    return np.stack(fr)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cot = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").cuda().eval()
    for p in cot.parameters():
        p.requires_grad_(False)
    files = sorted(glob.glob(os.path.join(DIR, "*.mp4")))
    rows = []
    for k, f in enumerate(files):
        base = os.path.basename(f)[:-4]
        model, scene = base.rsplit("_r", 1); scene = "r" + scene
        try:
            frames = read_gen(f, ctx_of(model))
            vid = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().cuda()
            with torch.no_grad():
                tracks, vis = cot(vid, grid_size=GRID)
            tr = tracks[0].cpu().numpy(); v = vis[0].cpu().numpy() > 0.5
            good = v.all(0)
            p0, p1 = tr[0, good], tr[-1, good]
            if len(p0) < 12:
                continue
            M, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            if M is None:
                continue
            inl = inl.ravel().astype(bool)
            dvec = p1 - p0
            disp = np.linalg.norm(dvec, axis=1)
            # camera drift = median inlier displacement magnitude + direction (for wedges)
            cam = float(np.median(disp[inl])) if inl.any() else float("nan")
            drift = np.median(dvec[inl], axis=0) if inl.any() else np.array([np.nan, np.nan])
            # full similarity decomposition: pan (drift dx,dy = LR,UD), scale -> forward/back,
            # rotation -> roll; scale & roll expressed as px-equivalent at the median track radius.
            s = float(np.hypot(M[0, 0], M[1, 0]))
            theta = float(np.arctan2(M[1, 0], M[0, 0]))
            cxy = np.array([SIZE[0] / 2.0, SIZE[1] / 2.0])
            rref = float(np.median(np.linalg.norm(p0[inl] - cxy, axis=1))) if inl.any() else float("nan")
            fb = (s - 1.0) * rref       # + = dolly forward (zoom in), - = backward
            roll = theta * rref         # + = counter-clockwise, tangential px at rref
            # signs of life = LOCAL motion contrast: a mover deviates from its local
            # background (robust to global camera drift + smooth parallax)
            _, nn = cKDTree(p0).query(p0, k=9)
            local_med = np.median(dvec[nn[:, 1:]], axis=1)
            contrast = np.linalg.norm(dvec - local_med, axis=1)
            life = int((contrast > RESID).sum())
            rows.append(dict(model=model, scene=scene, n_tracks=len(p0),
                             camera_motion=round(cam, 3),
                             drift_dx=round(float(drift[0]), 3), drift_dy=round(float(drift[1]), 3),
                             forward=round(fb, 3), roll=round(roll, 3),
                             signs_of_life=life, frac_life=round(life / len(p0), 4)))
        except Exception as e:
            print(f"[cot] {base} FAIL {str(e)[:70]}", flush=True); continue
        if (k + 1) % 40 == 0:
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(f"[cot] {k+1}/{len(files)}", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"[cot] wrote {OUT} ({len(rows)})")


if __name__ == "__main__":
    main()
