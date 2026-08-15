"""Final directional-quality evaluation for the No Critic variant.

The released fleet instruments (fleet_hf.py, fleet_static_inl.py) source the
"ours_*" variants by de-tiling grids/grids_A/A/<scene>_grid.mp4. No Critic was
trained after those grids were built, so it has no tile. We therefore read its
own rollouts and reproduce the instrument conventions exactly:

  * frame geometry : rows 32:480 of the 832x480 rollout == the 448x832 grid tile
  * context        : 12 frames (ours)
  * HF             : base = 4 frames at 1s into generation, end = 8 frames
                     (stride 2) up to the 6s horizon; d_blur = end - base;
                     B = -(d_blur - per-scene sibling median)
  * static         : ORB(3000)+crossCheck+RANSAC inliers between the first
                     generated frame and the +6s frame, resized to 640x352;
                     static if inliers > 600
  * feature-valid  : scenes r11 and r21 excluded (wet lens), 240 of 256
  * active         : feature_valid AND NOT static

Sibling medians are taken from the released fleet_hf.csv so that No Critic is
scored against exactly the same reference as every other variant.

Writes out/nocritic_hf.csv and out/nocritic_static.csv.
"""
import os, glob
import cv2, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
REF_HF = f"{ARR}/code_release/evaluation/quality/reference/fleet_hf.csv"
VID = f"{ARR}/logs/eval_final/A/nocritic/control_test"
OUT = os.path.join(HERE, "out")
CTX, FPS = 12, 16.0
BAD_SCENES = {"r11", "r21"}

_orb = cv2.ORB_create(3000)
_bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def tile(bgr):
    """832x480 rollout -> the 448x832 region the grid de-tiler yields."""
    return bgr[32:480, 0:832]


def lap_var(bgr_tile):
    g = cv2.cvtColor(cv2.resize(bgr_tile, (832, 448)), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def inl(a, b):
    ga, gb = [cv2.cvtColor(cv2.resize(x, (640, 352)), cv2.COLOR_BGR2GRAY) for x in (a, b)]
    k0, d0 = _orb.detectAndCompute(ga, None)
    k1, d1 = _orb.detectAndCompute(gb, None)
    if d0 is None or d1 is None:
        return 0
    ms = _bf.match(d0, d1)
    if len(ms) < 8:
        return 0
    src = np.float32([k0[m.queryIdx].pt for m in ms])
    dst = np.float32([k1[m.trainIdx].pt for m in ms])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


def main():
    os.makedirs(OUT, exist_ok=True)
    ref = pd.read_csv(REF_HF)
    sib = (ref.d_blur - ref.x_blur).groupby(ref.scene).median()   # released reference median

    rows = []
    for p in sorted(glob.glob(f"{VID}/step05000_r*_*_raw.mp4")):
        b = os.path.basename(p)
        wi = b.split("_r")[1].split("_")[0]
        dd = b.split("_")[2]
        scene = f"r{wi}_{dd}"
        cap = cv2.VideoCapture(p)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        b0 = CTX + int(round(FPS))
        end6 = min(n - 1, CTX + int(round(6.0 * FPS)))
        base_idx = [i for i in range(b0, b0 + 4) if i < n]
        end_idx = list(range(max(CTX, end6 - 14), end6 + 1, 2))
        stat_idx = [CTX, min(n - 2, CTX + int(round(6.0 * FPS)))]
        want = set(base_idx + end_idx + stat_idx)
        got, i = {}, 0
        while len(got) < len(want):
            ok, f = cap.read()
            if not ok:
                break
            if i in want:
                got[i] = f
            i += 1
        cap.release()
        if not all(i in got for i in base_idx + end_idx + stat_idx):
            print(f"[nc] {scene} incomplete decode, skipped", flush=True)
            continue
        base = np.mean([lap_var(tile(got[i])) for i in base_idx])
        end = np.mean([lap_var(tile(got[i])) for i in end_idx])
        v = inl(tile(got[stat_idx[0]]), tile(got[stat_idx[1]]))
        rows.append(dict(scene=scene, base_blur=round(base, 1), end_blur=round(end, 1),
                         d_blur=round(end - base, 1), static_inl=v, static=int(v > 600)))
    d = pd.DataFrame(rows)
    d["sib_med"] = d.scene.map(sib)
    d["B"] = -(d.d_blur - d.sib_med)
    d["feature_valid"] = ~d.scene.str.split("_").str[0].isin(BAD_SCENES)
    d["active"] = (d.feature_valid & (d.static == 0)).astype(int)
    d.to_csv(f"{OUT}/nocritic_final_eval.csv", index=False)

    fv, act = int(d.feature_valid.sum()), int(d.active.sum())
    a = d[d.active == 1]
    n150 = int((a.B > 150).sum())
    print(f"\n=== NO CRITIC ===")
    print(f"total rollouts     : {len(d)}")
    print(f"feature-valid      : {fv}/256  (scenes r11,r21 removed)")
    print(f"static             : {int((d.feature_valid & (d.static==1)).sum())} of the feature-valid")
    print(f"ACTIVE             : {act}/240  ({act}/256 of the full fleet)")
    print(f"HF  B>150 (active) : {n150}/{act} = {100*n150/act:.1f}%")
    print(f"median B (active)  : {a.B.median():+.1f}")
    print(f"HF  B>150 (all 256): {int((d.B>150).sum())}/256 = {100*(d.B>150).mean():.1f}%")
    print(f"saved -> {OUT}/nocritic_final_eval.csv")


if __name__ == "__main__":
    main()
