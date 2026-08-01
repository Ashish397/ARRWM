"""Signs-of-life from CoTracker residuals on the stationary set, per the locality
principle: camera motion is GLOBALLY correlated across arrows; signs of life are
LOCALIZED coherent residual clusters (a mover moves together, differently from the
background). We remove the global motion three ways and keep only localized residual:
  ransac : RANSAC affine start->end  (dominant rigid transform)
  pca    : per-rollout PCA of displacement trajectories, remove top-K modes
Residual field -> 8x8 spatial grid -> a cell is "alive" if it holds enough tracks
whose residual vectors are large AND coherent (mean/|std| high); signs_of_life = number
of alive cells (localized), which rejects globally-spread parallax.

Writes out/stationary_signs.csv. Run after the GPU is free (single job).
"""
import os
import os, glob
import numpy as np, torch, cv2, imageio, pandas as pd

DIR = os.environ.get("AF_STATIONARY_DIR",
               os.path.expanduser("~/stationary_evaluation"))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "stationary_signs.csv")
CTX = {"astra": 4, "matrixgame": 1, "minwm": 13, "worldcam": 65, "worldplay": 1, "yume": 1}
T, GRID, SIZE = 24, 40, (512, 288)
GC, RES_THR, MIN_TRK = 8, 2.0, 3       # grid cells per axis, residual px thresh, min tracks/cell


def ctx_of(m):
    return CTX.get(m, 12)


def read_gen(path, ctx):
    r = imageio.get_reader(path); n = r.count_frames()
    idx = np.linspace(min(ctx, n - 2), n - 1, T).round().astype(int)
    fr = [cv2.resize(np.asarray(r.get_data(int(i))), SIZE) for i in idx]
    r.close()
    return np.stack(fr)


def localized_life(p0, resid):
    """# grid cells with enough large + coherent residual tracks (a localized mover)."""
    mag = np.linalg.norm(resid, axis=1)
    big = mag > RES_THR
    if big.sum() < MIN_TRK:
        return 0
    gx = np.clip((p0[:, 0] / SIZE[0] * GC).astype(int), 0, GC - 1)
    gy = np.clip((p0[:, 1] / SIZE[1] * GC).astype(int), 0, GC - 1)
    alive = 0
    for cx in range(GC):
        for cy in range(GC):
            sel = big & (gx == cx) & (gy == cy)
            if sel.sum() < MIN_TRK:
                continue
            v = resid[sel]
            mean_v = v.mean(0); coher = np.linalg.norm(mean_v) / (v.std(0).mean() + 1e-6)
            if coher > 1.0:                 # residuals move together = a mover, not noise
                alive += 1
    return alive


def main():
    cot = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").cuda().eval()
    for p in cot.parameters():
        p.requires_grad_(False)
    files = sorted(glob.glob(f"{DIR}/*.mp4"))
    rows = []
    for k, f in enumerate(files):
        base = os.path.basename(f)[:-4]; model, scene = base.rsplit("_r", 1); scene = "r" + scene
        try:
            vid = torch.from_numpy(read_gen(f, ctx_of(model))).permute(0, 3, 1, 2)[None].float().cuda()
            with torch.no_grad():
                tr, vis = cot(vid, grid_size=GRID)
            P = tr[0].cpu().numpy(); v = vis[0].cpu().numpy() > 0.5; good = v.all(0)
            p0, p1 = P[0, good], P[-1, good]
            if len(p0) < 16:
                continue
            # (1) RANSAC affine residual
            M, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            cam = float(np.median(np.linalg.norm((p1 - p0)[inl.ravel().astype(bool)], axis=1))) if M is not None else np.nan
            res_r = p1 - ((M[:, :2] @ p0.T + M[:, 2:3]).T) if M is not None else (p1 - p0)
            life_ransac = localized_life(p0, res_r)
            # (3) per-rollout PCA residual (remove top-3 global modes)
            disp = (P[:, good, :] - P[0:1, good, :]).transpose(1, 0, 2).reshape(len(p0), 2 * T)
            Xc = disp - disp.mean(0)
            U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
            res_p = (Xc - (U[:, :3] * s[:3] @ Vt[:3])).reshape(len(p0), T, 2)[:, -1, :]  # end-frame residual
            life_pca = localized_life(p0, res_p)
            # global correlation of residual dirs (camera leftover check)
            rn = res_r / (np.linalg.norm(res_r, axis=1, keepdims=True) + 1e-6)
            glob_corr = float(np.linalg.norm(rn.mean(0)))
            rows.append(dict(model=model, scene=scene, n=len(p0), camera_motion=round(cam, 2),
                             life_ransac=life_ransac, life_pca=life_pca, glob_corr=round(glob_corr, 3)))
        except Exception as e:
            print(f"[sign] {base} FAIL {str(e)[:60]}", flush=True); continue
        if (k + 1) % 40 == 0:
            pd.DataFrame(rows).to_csv(OUT, index=False); print(f"[sign] {k+1}/{len(files)}", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"[sign] wrote {OUT} ({len(rows)})")


if __name__ == "__main__":
    main()
