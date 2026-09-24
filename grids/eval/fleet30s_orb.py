"""ORB instruments for the 30 s fleets (CPU), via fleet30s_common: for every (scene, model)
  static_inl : ORB(3000)+RANSAC-homography inliers between the first generated frame and the
               horizon frame (fleet_static_inl.py rule; near-static if > CTRL_ORB_T=500 in the paper's v2 rule)
  reloc_inl  : inliers between the REAL reference (seed clip frame 8, 0.5 s in) and the horizon frame
               (fleet_scene_scan.py); relocated if < scene_reloc_threshold (43)
  consensus_inl : max inliers between this model's horizon frame and any OTHER model's horizon frame or
               the real ref at the same scene (scene_consensus.py's sibling consensus)
Env: EVAL_HORIZON_S (6), EVAL_OUT_DIR (out30s), plus fleet30s_common's FLEET30S_* filters.
Writes <EVAL_OUT_DIR>/fleet30s_orb_h<H>.csv (resumable per scene).
Also saves the ORB keypoint positions + descriptors of every horizon frame to
<EVAL_OUT_DIR>/orb_desc_h<H>/<scene>__<model>.npz and of each real reference to ref__<uid>.npz, so
consensus_inl can be recomputed offline across runs on different machines (fleet30s_consensus.py).
Rows written here keep consensus_inl over the models present in THIS run.
"""
import os, sys, cv2, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fleet30s_common as fc
HERE = os.path.dirname(os.path.abspath(__file__)); H = fc.horizon_s()
OUT = os.path.join(HERE, os.environ.get("EVAL_OUT_DIR", "out30s"), f"fleet30s_orb_h{int(H)}.csv"); os.makedirs(os.path.dirname(OUT), exist_ok=True)
THR = int(open(os.path.join(HERE, "scene_reloc_threshold.txt")).read())
DESC = os.path.join(os.path.dirname(OUT), f"orb_desc_h{int(H)}"); os.makedirs(DESC, exist_ok=True)
orb = cv2.ORB_create(3000); bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
def desc(rgb):
    g = cv2.cvtColor(cv2.resize(rgb, (640, 352)), cv2.COLOR_RGB2GRAY); return orb.detectAndCompute(g, None)
def save_kd(path, kd):
    k, d = kd
    np.savez(path, pts=np.float32([p.pt for p in k]).reshape(-1, 2), desc=(d if d is not None else np.zeros((0, 32), np.uint8)))
def inl(kd0, kd1):
    (k0, d0), (k1, d1) = kd0, kd1
    if d0 is None or d1 is None or len(k0) < 8 or len(k1) < 8: return 0
    ms = bf.match(d0, d1)
    if len(ms) < 8: return 0
    src = np.float32([k0[m.queryIdx].pt for m in ms]); dst = np.float32([k1[m.trainIdx].pt for m in ms])
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0); return int(mask.sum()) if mask is not None else 0
def seed_ref(uid):
    if fc._RD:   # EVAL_DECODE_BITEXACT=1: same bit-exact ffmpeg path as the generated clips
        r = fc._reader(fc.SEED_CLIP(uid)); f = np.asarray(r.get_data(8)); r.close(); return f
    r = cv2.VideoCapture(fc.SEED_CLIP(uid)); r.set(cv2.CAP_PROP_POS_FRAMES, 8); ok, f = r.read(); r.release()
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if ok else None
idx = fc.fleet_index(); scenes = sorted({s for s, _ in idx}); done = set(); rows = []
if os.path.exists(OUT) and os.path.getsize(OUT) > 0:
    prev = pd.read_csv(OUT); done = set(prev.scene); rows = prev.to_dict("records")
for sc in scenes:
    if sc in done: continue
    uid, d = sc.rsplit("_", 1); ref = seed_ref(uid)
    if ref is None: print(f"[orb30] no seed ref for {uid}", flush=True); continue
    kd_ref = desc(ref); ends = {}; firsts = {}
    if not os.path.exists(f"{DESC}/ref__{uid}.npz"): save_kd(f"{DESC}/ref__{uid}.npz", kd_ref)
    for s, m in idx:
        if s != sc: continue
        n, fps = fc.meta(sc, m); ctx = fc.ctx_of(m); hz = min(n - 2, ctx + int(round(H * fps)))
        fr = fc.frames_at(sc, m, [ctx, hz])
        if fr is None or len(fr) < 2: continue
        firsts[m] = desc(fr[0]); ends[m] = desc(fr[1]); save_kd(f"{DESC}/{sc}__{m}.npz", ends[m])
    for m in ends:
        peers = [inl(ends[m], ends[o]) for o in ends if o != m] + [inl(ends[m], kd_ref)]
        rows.append(dict(scene=sc, model=m, horizon_s=H, static_inl=inl(firsts[m], ends[m]), reloc_inl=inl(kd_ref, ends[m]),
                         consensus_inl=max(peers) if peers else 0))
    pd.DataFrame(rows).to_csv(OUT, index=False); print(f"[orb30] {sc} ({len(ends)} models)", flush=True)
df = pd.DataFrame(rows); df["relocated"] = (df.reloc_inl < THR).astype(int); df["static500"] = (df.static_inl > 500).astype(int)
df.to_csv(OUT, index=False); print(f"[orb30] wrote {OUT} ({len(df)} rows)")
