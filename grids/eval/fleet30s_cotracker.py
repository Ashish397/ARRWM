"""Windowed CoTracker egomotion for the 30 s fleets: every clip gets a row, nothing is skipped.
The generated span (first generated frame -> 30 s) is sampled at 4 fps and cut into consecutive 1.5 s windows
(7 sampled frames each, windows share their boundary frame). In each window a fresh 30x30 CoTracker3 grid is
tracked; points visible in every frame of the window are kept and a RANSAC similarity (cv2.estimateAffinePartial2D,
reprojection 3 px) is fitted start -> end of the window, as in fleet_cotracker.py. Frames resized to 512x288.
Per window (long table <EVAL_OUT_DIR>/fleet30s_cotracker_windows.csv):
  survival = fraction of the 900 grid points visible through the whole window (trails disappearing -> low),
  valid = >= 12 surviving points and a similarity was found, n_inl, dx/dy = median inlier displacement (pan, px),
  disp = median inlier displacement magnitude (px), forward = (scale-1)*r_ref, roll = angle*r_ref (px-equivalent,
  r_ref = median inlier distance from the image centre).
Per clip and horizon H in 6/15/30 s (windows ending at or before H; <EVAL_OUT_DIR>/fleet30s_cotracker_h<H>.csv):
  n_windows, valid_frac, survival_mean, survival_min, survival_last (last window before H),
  pan_dx/pan_dy/forward/roll = sums over valid windows (net motion), path_px = sum of disp over valid windows,
  motion_per_s = path_px / valid seconds, motion_early = mean disp of the first two valid windows,
  motion_late = mean disp of the last two valid windows before H, late_early_ratio = motion_late / motion_early,
  status = tracked (all windows valid) | partial | lost (no valid window).
A clip whose decode or tracking raises gets status=error with the message (never dropped).
Env: FLEET30S_* filters (fleet30s_common), EVAL_OUT_DIR, EVAL_DECODE_BITEXACT. Resumable on the window table.
"""
import os, sys, numpy as np, torch, cv2, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fleet30s_common as fc
HERE = os.path.dirname(os.path.abspath(__file__))
OD = os.path.join(HERE, os.environ.get("EVAL_OUT_DIR", "out30s")); os.makedirs(OD, exist_ok=True)
WIN_CSV = f"{OD}/fleet30s_cotracker_windows.csv"
GRID, SIZE, SPS, WIN_S, SPAN_S, MIN_PTS = 30, (512, 288), 4, 1.5, 30.0, 12
STEPS = int(round(WIN_S * SPS)); NWIN = int(round(SPAN_S / WIN_S)); HORIZONS = (6, 15, 30)

def sample_idx(scene, model):
    n, fps = fc.meta(scene, model); ctx = fc.ctx_of(model)
    t = np.arange(NWIN * STEPS + 1) / SPS
    return [int(min(n - 1, round(ctx + x * fps))) for x in t]

@torch.no_grad()
def windows(cot, frames):
    vid = torch.from_numpy(np.stack([cv2.resize(f, SIZE) for f in frames])).permute(0, 3, 1, 2).float()
    clips = torch.stack([vid[w * STEPS: w * STEPS + STEPS + 1] for w in range(NWIN)]).cuda()   # [NWIN, 7, 3, H, W]
    out = []
    for b in range(NWIN):   # the offline model does not take a batch of videos (view error); one window per call
        tr, vis = cot(clips[b:b + 1], grid_size=GRID); out.append((tr[0].cpu().numpy(), vis[0].cpu().numpy() > 0.5))
    rows, cxy = [], np.array([SIZE[0] / 2.0, SIZE[1] / 2.0])
    for w, (tr, v) in enumerate(out):
        good = v.all(0); r = dict(w=w, t0_s=w * WIN_S, t1_s=(w + 1) * WIN_S, survival=round(float(good.mean()), 4), valid=0,
                                  n_inl=0, dx=np.nan, dy=np.nan, disp=np.nan, forward=np.nan, roll=np.nan)
        p0, p1 = tr[0, good], tr[-1, good]
        if len(p0) >= MIN_PTS:
            M, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            if M is not None and inl is not None and inl.any():
                inl = inl.ravel().astype(bool); dv = p1[inl] - p0[inl]; rref = float(np.median(np.linalg.norm(p0[inl] - cxy, axis=1)))
                s = float(np.hypot(M[0, 0], M[1, 0])); th = float(np.arctan2(M[1, 0], M[0, 0]))
                r.update(valid=1, n_inl=int(inl.sum()), dx=round(float(np.median(dv[:, 0])), 3), dy=round(float(np.median(dv[:, 1])), 3),
                         disp=round(float(np.median(np.linalg.norm(dv, axis=1))), 3), forward=round((s - 1) * rref, 3), roll=round(th * rref, 3))
        rows.append(r)
    return rows

def summarise(wdf):
    for H in HORIZONS:
        out = []
        for (sc, m), g in wdf.groupby(["scene", "model"], sort=False):
            if (g.status == "error").any():
                out.append(dict(scene=sc, model=m, horizon_s=H, status="error", error=g.error.iloc[0])); continue
            g = g[g.t1_s <= H + 1e-6].sort_values("w"); v = g[g.valid == 1]
            early = v.disp.head(2).mean() if len(v) else np.nan; late = v.disp.tail(2).mean() if len(v) else np.nan
            out.append(dict(scene=sc, model=m, horizon_s=H, n_windows=len(g), valid_frac=round(len(v) / len(g), 3),
                            survival_mean=round(g.survival.mean(), 4), survival_min=round(g.survival.min(), 4), survival_last=round(g.survival.iloc[-1], 4),
                            pan_dx=round(v.dx.sum(), 2), pan_dy=round(v.dy.sum(), 2), forward=round(v.forward.sum(), 2), roll=round(v.roll.sum(), 2),
                            path_px=round(v.disp.sum(), 2), motion_per_s=round(v.disp.sum() / (len(v) * WIN_S), 3) if len(v) else np.nan,
                            motion_early=round(early, 3), motion_late=round(late, 3), late_early_ratio=round(late / early, 3) if early and early > 0 else np.nan,
                            status="tracked" if len(v) == len(g) else ("lost" if len(v) == 0 else "partial")))
        pd.DataFrame(out).to_csv(f"{OD}/fleet30s_cotracker_h{H}.csv", index=False)
        print(f"[cot30w] wrote {OD}/fleet30s_cotracker_h{H}.csv ({len(out)})", flush=True)

def main():
    cot = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").cuda().eval()
    idx = fc.fleet_index(); rows = []; done = set()
    if os.path.exists(WIN_CSV) and os.path.getsize(WIN_CSV) > 0:
        prev = pd.read_csv(WIN_CSV); rows = prev.to_dict("records"); done = set(zip(prev.scene, prev.model))
    for k, (sc, m) in enumerate(idx):
        if (sc, m) in done: continue
        try:
            for r in windows(cot, fc.frames_at(sc, m, sample_idx(sc, m))): rows.append(dict(scene=sc, model=m, status="ok", error="", **r))
        except Exception as e:
            print(f"[cot30w] {sc} {m} FAIL {str(e)[:80]}", flush=True); rows.append(dict(scene=sc, model=m, status="error", error=str(e)[:200], w=-1))
        if (k + 1) % 50 == 0: pd.DataFrame(rows).to_csv(WIN_CSV, index=False); print(f"[cot30w] {k+1}/{len(idx)}", flush=True)
    wdf = pd.DataFrame(rows); wdf.to_csv(WIN_CSV, index=False); print(f"[cot30w] wrote {WIN_CSV} ({wdf[['scene','model']].drop_duplicates().shape[0]} clips)", flush=True)
    summarise(wdf)

if __name__ == "__main__":
    main()
