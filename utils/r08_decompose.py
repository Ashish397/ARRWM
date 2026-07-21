"""r08 only: decompose the frame0->frame_t similarity transform into
  scale     -> forward(>1)/back(<1) translation
  rotation  -> image-plane roll (deg)
  tx, ty    -> horizontal (yaw/left-right) & vertical (pitch) image shift
per model per branch, and join with the CoTracker->PCA action-critic g0..g7.
Lets us calibrate signs on the pure branches (F/B/L/R) and evaluate entangled
ones (BL/BR/...). Prints tables; saves analysis/eval_final/r08_decompose.csv.
"""
import os, numpy as np, pandas as pd, imageio, cv2
det = cv2.SIFT_create(); bf = cv2.BFMatcher()
RUNS = ["pca8_8node", "pca4", "pca2", "16node", "4node", "noatok"]
BRANCHES = ["F", "B", "R", "L", "FR", "FL", "BR", "BL"]
TS = [90, 100]


def gray(path, i):
    r = imageio.get_reader(path); f = cv2.cvtColor(np.asarray(r.get_data(i)), cv2.COLOR_RGB2GRAY); r.close(); return f


def decomp(path):
    g0 = gray(path, 0)
    k0, d0 = det.detectAndCompute(g0, None)
    accs = []
    for t in TS:
        try:
            gt = gray(path, t)
        except Exception:
            continue
        kt, dt = det.detectAndCompute(gt, None)
        if d0 is None or dt is None:
            continue
        good = [a for a, b in bf.knnMatch(d0, dt, k=2) if a.distance < 0.75 * b.distance]
        if len(good) < 8:
            continue
        src = np.float32([k0[x.queryIdx].pt for x in good]).reshape(-1, 1, 2)
        dst = np.float32([kt[x.trainIdx].pt for x in good]).reshape(-1, 1, 2)
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3)
        if M is None:
            continue
        s = float(np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2))
        rot = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
        accs.append((s, rot, float(M[0, 2]), float(M[1, 2]), int(inl.sum())))
    if not accs:
        return None
    a = np.array(accs)
    return dict(scale=a[:, 0].mean(), rot=a[:, 1].mean(), tx=a[:, 2].mean(), ty=a[:, 3].mean(), inl=a[:, 4].mean())


def main():
    d = pd.read_csv("analysis/eval_final/phaseA_metrics.csv")
    for c in [f"g{i}" for i in range(8)]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    G = d[d["rank"] == 8].groupby(["run", "branch"])[[f"g{i}" for i in range(8)]].mean().reset_index()
    rows = []
    for run in RUNS:
        for br in BRANCHES:
            p = f"logs/eval_final/A/{run}/control_test/step05000_r08_{br}_raw.mp4"
            if not os.path.exists(p):
                continue
            dc = decomp(p)
            if dc is None:
                continue
            g = G[(G.run == run) & (G.branch == br)]
            row = dict(run=run, branch=br, recession=round(100 * (1 - dc["scale"]), 1),
                       rot=round(dc["rot"], 2), tx=round(dc["tx"], 1), ty=round(dc["ty"], 1), inl=int(dc["inl"]))
            for i in range(8):
                row[f"g{i}"] = round(float(g[f"g{i}"].iloc[0]), 3) if len(g) else np.nan
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("analysis/eval_final/r08_decompose.csv", index=False)

    print("=== SIGN CALIBRATION on pure branches (mean over 6 models) ===")
    for br in ["F", "B", "R", "L"]:
        s = df[df.branch == br]
        print(f"  {br}: recession={s.recession.mean():+6.1f}%  tx={s.tx.mean():+6.1f}  rot={s.rot.mean():+5.2f}  "
              f"ty={s.ty.mean():+6.1f} | g0(thr)={s.g0.mean():+.3f} g1(str)={s.g1.mean():+.3f}")
    print("\n=== BL (back-left): per model — is 4node best at back+left? ===")
    print("  (recession>0 = back; tx sign & g1<0 = left)")
    s = df[df.branch == "BL"].copy()
    print(s[["run", "recession", "tx", "rot", "g0", "g1"]].to_string(index=False))
    print("\n=== BR (back-right): per model, all 8 PCA dims + measured rot/ty ===")
    s = df[df.branch == "BR"].copy()
    print(s[["run", "recession", "rot", "ty", "tx"] + [f"g{i}" for i in range(8)]].to_string(index=False))


if __name__ == "__main__":
    main()
