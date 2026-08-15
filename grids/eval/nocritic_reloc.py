"""Scene-relocation (sibling-consensus place identity) for the No Critic variant.

Reproduces grids/eval/scene_consensus.py using the locally-stored per-model
rollouts instead of the grid tiles (No Critic post-dates the grids):

  members per scene = real reference frames (context frames 2,5,8,11 of the
  shared held-out seed) + the +6s end frame of every fleet model
  score(model)      = max RANSAC-verified ORB inliers to any OTHER member
  relocated         = score < 50

Runs a validation pass on Default (ours_pca8) against the released
fleet_scene_consensus.csv before scoring No Critic.

Writes out/nocritic_reloc.csv
"""
import os, sys, glob
import cv2, numpy as np, pandas as pd, imageio.v2 as imageio

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
REF = f"{ARR}/code_release/evaluation/quality/reference/fleet_scene_consensus.csv"

OURS = {"pca8": "pca8_8node", "pca4": "pca4", "pca2": "pca2", "16node": "16node",
        "4node": "4node", "noatok": "noatok", "noadaln": "noadaln"}
EXT_CTX = {"astra": 4, "matrixgame": 1, "minwm": 13, "worldcam": 65, "worldplay": 1, "yume": 1}
OURS_CTX = 12

_orb = cv2.ORB_create(3000)
_bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def frame_at(path, i):
    # PyAV/imageio rather than cv2 seeking: cv2.VideoCapture seeks fail
    # intermittently on this filesystem ("Cannot initialize the conversion
    # context"), silently dropping rollouts from the scored set.
    try:
        r = imageio.get_reader(path)
        a = np.asarray(r.get_data(int(i)))
        r.close()
        return cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def meta(path):
    try:
        r = imageio.get_reader(path)
        n = r.count_frames()
        fps = r.get_meta_data().get("fps", 16) or 16
        r.close()
        return fps, n
    except Exception:
        return 16, 0


def desc(img):
    g = cv2.cvtColor(cv2.resize(img, (640, 352)), cv2.COLOR_BGR2GRAY)
    return _orb.detectAndCompute(g, None)


def inl(kd0, kd1):
    (k0, d0), (k1, d1) = kd0, kd1
    if d0 is None or d1 is None or len(k0) < 8 or len(k1) < 8:
        return 0
    ms = _bf.match(d0, d1)
    if len(ms) < 8:
        return 0
    src = np.float32([k0[m.queryIdx].pt for m in ms])
    dst = np.float32([k1[m.trainIdx].pt for m in ms])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


def ours_path(variant_dir, scene):
    wi, dd = scene.split("_")
    return f"{ARR}/logs/eval_final/A/{variant_dir}/control_test/step05000_{wi}_{dd}_raw.mp4"


def ext_path(model, scene):
    return f"{ARR}/logs/eval_final/A_{model}/{model}_{scene}.mp4"


def members_for(scene, include_nocritic=True):
    """{name: descriptor} for one scene."""
    mem = {}
    # real reference: shared seed context frames (identical across our variants)
    rp = ours_path("pca8_8node", scene)
    if not os.path.exists(rp):
        return mem
    for ri in (2, 5, 8, 11):
        f = frame_at(rp, ri)
        if f is not None:
            mem[f"__ref{ri}__"] = desc(f)
    jobs = {f"ours_{v}": (ours_path(d, scene), OURS_CTX) for v, d in OURS.items()}
    jobs.update({m: (ext_path(m, scene), c) for m, c in EXT_CTX.items()})
    if include_nocritic:
        jobs["ours_nocritic"] = (ours_path("nocritic", scene), OURS_CTX)
    for name, (fp, ctx) in jobs.items():
        if not os.path.exists(fp):
            continue
        fps, n = meta(fp)
        ef = frame_at(fp, min(n - 2, ctx + int(round(6.0 * fps))))
        if ef is not None:
            mem[name] = desc(ef)
    return mem


def score(mem, target):
    return max((inl(mem[target], mem[b]) for b in mem if b != target), default=0)


def main():
    os.makedirs(OUT, exist_ok=True)
    scenes = sorted(os.path.basename(p).split("_r")[1].replace("_raw.mp4", "")
                    for p in glob.glob(f"{ARR}/logs/eval_final/A/nocritic/control_test/*_raw.mp4"))
    scenes = [f"r{s}" for s in scenes]
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode in ("validate", "all"):
        rel = pd.read_csv(REF)
        rel = rel[rel.model == "ours_pca8"].set_index("scene").consensus_inl
        ok = tot = 0
        for sc in scenes[:30]:
            mem = members_for(sc, include_nocritic=False)
            if "ours_pca8" not in mem or sc not in rel.index:
                continue
            got, exp = score(mem, "ours_pca8"), int(rel.loc[sc])
            tot += 1
            ok += int((got < 50) == (exp < 50))
            print(f"  {sc}: mine {got:4d}  released {exp:4d}  {'ok' if (got<50)==(exp<50) else 'LABEL DIFF'}", flush=True)
        print(f"[validate] relocation-label agreement {ok}/{tot}", flush=True)
        if mode == "validate":
            return

    rows = []
    for k, sc in enumerate(scenes):
        mem = members_for(sc)
        if "ours_nocritic" not in mem:
            continue
        rows.append(dict(scene=sc, consensus_inl=score(mem, "ours_nocritic")))
        if (k + 1) % 32 == 0:
            print(f"[nc-reloc] {k+1}/{len(scenes)}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(f"{OUT}/nocritic_reloc.csv", index=False)
    print(f"saved {OUT}/nocritic_reloc.csv ({len(d)})")


if __name__ == "__main__":
    main()
