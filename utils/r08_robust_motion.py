"""r08: TURN-ROBUST motion decomposition (Part 1 + Part 2 of the final-metric work).

Fix for the frame0->frameT similarity fit breaking under turns (pure R gave -125%
"recession"): accumulate INCREMENTAL frame-to-frame similarity transforms (stride 3,
high overlap every step) in log-scale space, plus Farneback optical-flow divergence
as an independent fwd/back read that is rotation/pan-invariant by construction.

Per r08 video (6 models x 8 branches):
  cum_recession = (1 - exp(sum log s_step)) * 100   [%]  >0 back, <0 forward
  flow_div      = accumulated mean divergence of central flow field  <0 back (contraction)
  cum_rot       = summed image rotation [deg]  (roll proxy)
  cum_tx, cum_ty= summed translation [px]      (yaw / pitch proxies)
Then:
  - validation on pure branches (F/B expect sign, R/L expect |recession| small now)
  - BL combined-motion table (back% + left from tx & critic g1)
  - Part 2: Spearman corr across models of cum_rot vs g0..g7 and cum_ty vs g0..g7
    per branch family, to identify which PCA dims are roll & pitch.
Saves analysis/eval_final/r08_robust_motion.csv
"""
import os
import numpy as np, pandas as pd, imageio, cv2
from multiprocessing import Pool

RUNS = ["pca8_8node", "pca4", "pca2", "16node", "4node", "noatok"]
BRANCHES = ["F", "B", "R", "L", "FR", "FL", "BR", "BL"]
STRIDE = 3
NFRAMES = 106
det = cv2.SIFT_create()
bf = cv2.BFMatcher()


def analyze(args):
    run, br = args
    p = f"logs/eval_final/A/{run}/control_test/step05000_r08_{br}_raw.mp4"
    if not os.path.exists(p):
        return None
    r = imageio.get_reader(p)
    idxs = list(range(0, NFRAMES, STRIDE))
    logs, rots, txs, tys, divs, inls = [], [], [], [], [], []
    prev = None
    for i in idxs:
        try:
            f = cv2.cvtColor(np.asarray(r.get_data(i)), cv2.COLOR_RGB2GRAY)
        except Exception:
            break
        if prev is not None:
            k0, d0 = det.detectAndCompute(prev, None)
            kt, dt = det.detectAndCompute(f, None)
            if d0 is not None and dt is not None:
                good = [a for a, b in bf.knnMatch(d0, dt, k=2) if a.distance < 0.75 * b.distance]
                if len(good) >= 8:
                    src = np.float32([k0[x.queryIdx].pt for x in good]).reshape(-1, 1, 2)
                    dst = np.float32([kt[x.trainIdx].pt for x in good]).reshape(-1, 1, 2)
                    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3)
                    if M is not None:
                        s = float(np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2))
                        if 0.5 < s < 2.0:
                            logs.append(np.log(s))
                            rots.append(float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))))
                            txs.append(float(M[0, 2])); tys.append(float(M[1, 2]))
                            inls.append(int(inl.sum()))
            # flow divergence on central crop (rotation/pan invariant fwd-back read)
            H, W = f.shape
            a = prev[H // 4:3 * H // 4, W // 4:3 * W // 4]
            b = f[H // 4:3 * H // 4, W // 4:3 * W // 4]
            fl = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 21, 3, 5, 1.2, 0)
            dudx = np.gradient(fl[..., 0], axis=1); dvdy = np.gradient(fl[..., 1], axis=0)
            divs.append(float((dudx + dvdy).mean()))
        prev = f
    r.close()
    if not logs:
        return None
    return dict(run=run, branch=br,
                recession=round(100 * (1 - float(np.exp(np.sum(logs)))), 1),
                flow_div=round(float(np.sum(divs)), 3),
                rot=round(float(np.sum(rots)), 2),
                tx=round(float(np.sum(txs)), 1), ty=round(float(np.sum(tys)), 1),
                inl_med=int(np.median(inls)), steps=len(logs))


def main():
    tasks = [(run, br) for run in RUNS for br in BRANCHES]
    with Pool(int(os.environ.get("RM_WORKERS", "16"))) as pool:
        res = [x for x in pool.imap_unordered(analyze, tasks) if x]
    df = pd.DataFrame(res)

    # join critic g's
    d = pd.read_csv("analysis/eval_final/phaseA_metrics.csv")
    for c in [f"g{i}" for i in range(8)]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    G = d[d["rank"] == 8].groupby(["run", "branch"])[[f"g{i}" for i in range(8)]].mean().reset_index()
    df = df.merge(G, on=["run", "branch"], how="left")
    df.to_csv("analysis/eval_final/r08_robust_motion.csv", index=False)

    print("=== VALIDATION: pure branches (mean over models) — R/L recession should now be ~small ===")
    for br in ["F", "B", "R", "L"]:
        s = df[df.branch == br]
        print(f"  {br}: recession={s.recession.mean():+7.1f}%  flow_div={s.flow_div.mean():+7.3f}  "
              f"rot={s.rot.mean():+7.2f}  tx={s.tx.mean():+7.1f}  ty={s.ty.mean():+6.1f}")
    print("\n=== per-model, ALL branches ===")
    for br in BRANCHES:
        s = df[df.branch == br].set_index("run").reindex(RUNS)
        print(f"\n [{br}]")
        print(s[["recession", "flow_div", "rot", "tx", "ty", "g0", "g1", "g6", "g7"]].to_string())

    # Part 2: which PCA dim tracks measured roll (cum_rot) and pitch (cum_ty)?
    from scipy.stats import spearmanr
    print("\n=== Part 2: Spearman(measured, g_i) across models, per branch (n=6 each) ===")
    for meas, col in [("rot(roll)", "rot"), ("ty(pitch)", "ty")]:
        print(f"\n {meas}:")
        for br in ["B", "BL", "BR", "F"]:
            s = df[df.branch == br]
            if len(s) < 5:
                continue
            cors = []
            for i in range(8):
                rho = spearmanr(s[col], s[f"g{i}"])[0]
                cors.append((abs(rho), rho, i))
            cors.sort(reverse=True)
            top = "  ".join(f"g{i}:{rho:+.2f}" for _, rho, i in cors[:3])
            print(f"   {br}: {top}")


if __name__ == "__main__":
    main()
