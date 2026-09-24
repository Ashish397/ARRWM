"""Reloc-Cal-v2: frozen reference = 6 externals + real refs (no model of ours).
Ours (flagship+ablations) scored vs R. Externals scored vs (pca8 + other ext + real).
Every system gets exactly 6 system-peers + real -> symmetric.
  python reloc_cal_v2.py  -> out/reloc_cal_v2_pairs.csv (external<->pca8 inliers)
"""
import os, numpy as np, cv2, pandas as pd

HERE=os.path.dirname(os.path.abspath(__file__))
PACK=os.path.join(HERE,"out","reloc_reference_pack_3000.npz")
EXT=["astra","matrixgame","minwm","worldcam","worldplay","yume"]
REP="ours_pca8"
bf=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def inl(p0,d0,p1,d1):
    if len(d0)<8 or len(d1)<8: return 0
    ms=bf.match(d0,d1)
    if len(ms)<8: return 0
    src=np.float32([p0[m.queryIdx] for m in ms]); dst=np.float32([p1[m.trainIdx] for m in ms])
    H,mask=cv2.findHomography(src,dst,cv2.RANSAC,5.0)
    return int(mask.sum()) if mask is not None else 0

P=np.load(PACK)
scenes=sorted(set(k.split("|")[0] for k in P.files))
rows=[]
for i,sc in enumerate(scenes,1):
    mem=set(k.split("|")[1] for k in P.files if k.startswith(sc+"|"))
    if REP not in mem: continue
    pr,dr=P[f"{sc}|{REP}|pts"],P[f"{sc}|{REP}|desc"]
    for e in EXT:
        if e not in mem: continue
        pe,de=P[f"{sc}|{e}|pts"],P[f"{sc}|{e}|desc"]
        rows.append(dict(scene=sc,model=e,inl_vs_rep=inl(pe,de,pr,dr)))
    if i%64==0: print(f"[v2] {i}/{len(scenes)}",flush=True)
d=pd.DataFrame(rows); d.to_csv(os.path.join(HERE,"out","reloc_cal_v2_pairs.csv"),index=False)
print("wrote out/reloc_cal_v2_pairs.csv",len(d))
