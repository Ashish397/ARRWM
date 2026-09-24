"""Offline sibling consensus for the 30 s fleets from saved ORB descriptors (fleet30s_orb.py's orb_desc_h<H>/).
consensus_inl(scene, model) = max ORB inliers between that model's horizon frame and every OTHER model's horizon
frame at the same scene, plus the real reference (seed clip frame 8), with the same matcher as fleet30s_orb.py:
BFMatcher(NORM_HAMMING, crossCheck) -> >=8 matches -> findHomography RANSAC 5.0 inlier count (0 if <8 keypoints/matches).
Descriptor dirs from several machines are pooled (e.g. local external/ODE fleets + u6qf DMD arms); a (scene, model)
found in more than one dir must come from identical decodes (EVAL_DECODE_BITEXACT=1 on both), the first dir wins.
Usage: fleet30s_consensus.py --desc_dirs dirA,dirB --out consensus_h6.csv [--models csv]
"""
import argparse, glob, os, cv2, numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--desc_dirs", required=True); ap.add_argument("--out", required=True); ap.add_argument("--models", default="")
a = ap.parse_args()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
def load(p):
    z = np.load(p); return z["pts"], z["desc"]
def inl(kd0, kd1):
    (p0, d0), (p1, d1) = kd0, kd1
    if len(d0) == 0 or len(d1) == 0 or len(p0) < 8 or len(p1) < 8: return 0
    ms = bf.match(d0, d1)
    if len(ms) < 8: return 0
    src = np.float32([p0[m.queryIdx] for m in ms]); dst = np.float32([p1[m.trainIdx] for m in ms])
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0); return int(mask.sum()) if mask is not None else 0
keep = set(x for x in a.models.split(",") if x)
clips, refs = {}, {}
for d in [x for x in a.desc_dirs.split(",") if x]:
    for p in sorted(glob.glob(f"{d}/*.npz")):
        b = os.path.basename(p)[:-4]
        if b.startswith("ref__"): refs.setdefault(b[5:], p); continue
        sc, m = b.split("__", 1)
        if keep and m not in keep: continue
        clips.setdefault(sc, {}).setdefault(m, p)
rows = []
for sc in sorted(clips):
    uid = sc.rsplit("_", 1)[0]
    if uid not in refs: print(f"[cons] no ref for {uid}; skipping {sc}", flush=True); continue
    kd_ref = load(refs[uid]); ends = {m: load(p) for m, p in clips[sc].items()}
    for m in ends:
        peers = [inl(ends[m], ends[o]) for o in ends if o != m] + [inl(ends[m], kd_ref)]
        rows.append(dict(scene=sc, model=m, consensus_inl=max(peers), n_peer_models=len(ends) - 1))
df = pd.DataFrame(rows); df.to_csv(a.out, index=False); print(f"[cons] wrote {a.out} ({len(df)} rows, {len(clips)} scenes)")
