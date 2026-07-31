"""CoTracker + RANSAC egomotion on the full action-forcing fleet (256x13 via fleet_common), for
per-rollout controllability (did the camera move the commanded way). Same similarity decomposition
as stationary_cotracker.py: RANSAC affine start->end over the 6s generated span ->
  camera_motion (median inlier displacement), drift_dx/dy = LR/UD pan, forward = FB (scale),
  roll = rotation (px-equiv). Writes out/fleet_cotracker.csv (resumable)."""
import os
import numpy as np, torch, cv2, pandas as pd
import fleet_common as fc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "fleet_cotracker.csv")
T, GRID, SIZE = 24, 30, (512, 288)


def gen_frames(scene, model):
    n, fps = fc.meta(scene, model); ctx = fc.ctx_of(model)
    end = min(n - 1, ctx + int(round(6.0 * fps)))
    idx = list(np.linspace(min(ctx, n - 2), end, T).round().astype(int))
    fr = fc.frames_at(scene, model, idx)
    return np.stack([cv2.resize(f, SIZE) for f in fr])


def main():
    cot = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").cuda().eval()
    for p in cot.parameters():
        p.requires_grad_(False)
    idx = list(fc.fleet_index())
    done = set(); rows = []
    if os.path.exists(OUT) and os.path.getsize(OUT) > 0:
        prev = pd.read_csv(OUT); done = set(zip(prev.scene, prev.model)); rows = prev.to_dict("records")
    for k, (scene, model) in enumerate(idx):
        if (scene, model) in done:
            continue
        try:
            frames = gen_frames(scene, model)
            vid = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().cuda()
            with torch.no_grad():
                tracks, vis = cot(vid, grid_size=GRID)
            tr = tracks[0].cpu().numpy(); v = vis[0].cpu().numpy() > 0.5
            good = v.all(0); p0, p1 = tr[0, good], tr[-1, good]
            if len(p0) < 12:
                continue
            M, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            if M is None:
                continue
            inl = inl.ravel().astype(bool); dvec = p1 - p0; disp = np.linalg.norm(dvec, axis=1)
            cam = float(np.median(disp[inl])) if inl.any() else float("nan")
            drift = np.median(dvec[inl], axis=0) if inl.any() else np.array([np.nan, np.nan])
            s = float(np.hypot(M[0, 0], M[1, 0])); theta = float(np.arctan2(M[1, 0], M[0, 0]))
            cxy = np.array([SIZE[0] / 2.0, SIZE[1] / 2.0])
            rref = float(np.median(np.linalg.norm(p0[inl] - cxy, axis=1))) if inl.any() else float("nan")
            rows.append(dict(scene=scene, model=model, camera_motion=round(cam, 3),
                             drift_dx=round(float(drift[0]), 3), drift_dy=round(float(drift[1]), 3),
                             forward=round((s - 1.0) * rref, 3), roll=round(theta * rref, 3)))
        except Exception as e:
            print(f"[fcot] {scene} {model} FAIL {str(e)[:60]}", flush=True); continue
        if (k + 1) % 100 == 0:
            pd.DataFrame(rows).to_csv(OUT, index=False); print(f"[fcot] {k+1}/{len(idx)}", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"[fcot] wrote {OUT} ({len(rows)})")


if __name__ == "__main__":
    main()
