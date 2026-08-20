"""Stationary (no-op) camera-drift + residual-animation for the No Critic variant,
run alongside the seven published variants so the published Table 15 rows act as
the validation anchor.

Reproduces grids/eval/stationary_cotracker.py exactly (T=24 sampled frames over
the generated span, 30x30 CoTracker grid at 512x288, RANSAC partial-affine
start->end, camera drift = median inlier displacement, movers = tracks whose
displacement deviates from their local 8-neighbour median by > 4 px), reading
the locally-stored per-variant stationary rollouts instead of the flat
stationary_evaluation directory.

  Movement  (Table 15) = median over the 32 rollouts of camera drift
  Animation (Table 15) = mean over the 32 rollouts of localised movers

Also writes the per-rollout drift vector so the Figure 28 stationary wedge can
be regenerated. Writes out/nocritic_stationary.csv.
"""
import os, glob
import numpy as np, torch, cv2, imageio.v2 as imageio, pandas as pd
from scipy.spatial import cKDTree

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "nocritic_stationary.csv")
T, GRID, SIZE, RESID = 24, 30, (512, 288), 4.0
CTX = 12
VARIANTS = {"pca8": "pca8_8node", "16node": "16node", "4node": "4node",
            "pca4": "pca4", "pca2": "pca2", "noatok": "noatok",
            "noadaln": "noadaln", "nocritic": "nocritic"}
# published Table 15 values (Movement, Animation) for the seven known variants
PUB = {"pca8": (2.3, 1.4), "pca4": (2.1, 0.0), "pca2": (2.2, 0.0),
       "16node": (2.0, 0.6), "4node": (2.0, 0.0), "noatok": (2.1, 0.1),
       "noadaln": (32.2, 6.2)}


def read_gen(path, ctx=CTX):
    r = imageio.get_reader(path)
    n = r.count_frames()
    idx = np.linspace(min(ctx, n - 2), n - 1, T).round().astype(int)
    fr = [cv2.resize(np.asarray(r.get_data(int(i))), SIZE) for i in idx]
    r.close()
    return np.stack(fr)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cot = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").cuda().eval()
    for p in cot.parameters():
        p.requires_grad_(False)

    rows = []
    for var, d in VARIANTS.items():
        files = sorted(glob.glob(f"{ARR}/logs/eval_final/B_Astarts/{d}/control_test/*.mp4"))
        for f in files:
            base = os.path.basename(f)[:-4]
            try:
                frames = read_gen(f)
                vid = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().cuda()
                with torch.no_grad():
                    tracks, vis = cot(vid, grid_size=GRID)
                tr = tracks[0].cpu().numpy(); v = vis[0].cpu().numpy() > 0.5
                good = v.all(0)
                p0, p1 = tr[0, good], tr[-1, good]
                if len(p0) < 12:
                    continue
                M, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC,
                                                     ransacReprojThreshold=3.0)
                if M is None:
                    continue
                inl = inl.ravel().astype(bool)
                dvec = p1 - p0
                disp = np.linalg.norm(dvec, axis=1)
                cam = float(np.median(disp[inl])) if inl.any() else float("nan")
                drift = np.median(dvec[inl], axis=0) if inl.any() else np.array([np.nan, np.nan])
                s = float(np.hypot(M[0, 0], M[1, 0]))
                theta = float(np.arctan2(M[1, 0], M[0, 0]))
                cxy = np.array([SIZE[0] / 2.0, SIZE[1] / 2.0])
                rref = float(np.median(np.linalg.norm(p0[inl] - cxy, axis=1))) if inl.any() else float("nan")
                fb = (s - 1.0) * rref
                roll = theta * rref
                _, nn = cKDTree(p0).query(p0, k=9)
                local_med = np.median(dvec[nn[:, 1:]], axis=1)
                contrast = np.linalg.norm(dvec - local_med, axis=1)
                life = int((contrast > RESID).sum())
                rows.append(dict(model=var, scene=base, n_tracks=len(p0),
                                 camera_motion=round(cam, 3),
                                 drift_dx=round(float(drift[0]), 3),
                                 drift_dy=round(float(drift[1]), 3),
                                 forward=round(fb, 3), roll=round(roll, 3),
                                 signs_of_life=life))
            except Exception as e:
                print(f"[stat] {var} {base} FAIL {str(e)[:70]}", flush=True)
                continue
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"[stat] {var}: {len([r for r in rows if r['model']==var])} rollouts", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    print("\n=== Table 15 (Movement = median camera drift px; Animation = mean movers) ===")
    print(f"{'variant':10s} {'n':>3s} {'Movement':>9s} {'Animation':>10s}   {'published':>18s}")
    for var in VARIANTS:
        s = d[d.model == var]
        if not len(s):
            continue
        mv, an = s.camera_motion.median(), s.signs_of_life.mean()
        pub = PUB.get(var)
        tag = f"({pub[0]}, {pub[1]})" if pub else "(NEW)"
        print(f"{var:10s} {len(s):3d} {mv:9.1f} {an:10.1f}   {tag:>18s}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
