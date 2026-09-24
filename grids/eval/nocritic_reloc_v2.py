"""No Critic directional relocation under Reloc-Cal-v2. ONE ORB run emits both peer sets:
  inl_ext = max inliers vs 6 externals + real refs   -> deployed (ours-as-sole-representative rule)
  inl_fam = max(inl_ext, inliers vs flagship pca8)   -> kept for reference only
"""
import os, cv2, numpy as np, pandas as pd, imageio.v2 as imageio
HERE=os.path.dirname(os.path.abspath(__file__))
P=np.load(os.path.join(HERE,"out","reloc_reference_pack_3000.npz"))
NC="/home/ashish/ARRWM/logs/eval_final/A/nocritic/control_test"
EXT={"astra","matrixgame","minwm","worldcam","worldplay","yume"}; REP="ours_pca8"
bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
def desc(img):
    g=cv2.cvtColor(cv2.resize(img,(640,352)),cv2.COLOR_BGR2GRAY)
    k,d=cv2.ORB_create(3000).detectAndCompute(g,None)
    if d is None: return np.zeros((0,2),np.float32),np.zeros((0,32),np.uint8)
    return np.float32([kp.pt for kp in k]),d
def inl(p0,d0,p1,d1):
    if len(d0)<8 or len(d1)<8: return 0
    ms=bf.match(d0,d1)
    if len(ms)<8: return 0
    src=np.float32([p0[m.queryIdx] for m in ms]); dst=np.float32([p1[m.trainIdx] for m in ms])
    H,mask=cv2.findHomography(src,dst,cv2.RANSAC,5.0)
    return int(mask.sum()) if mask is not None else 0
rows=[]
for i,sc in enumerate(sorted(set(k.split("|")[0] for k in P.files)),1):
    p=os.path.join(NC,f"step05000_{sc}_raw.mp4")
    if not os.path.exists(p): continue
    r=imageio.get_reader(p); n=r.count_frames(); fps=r.get_meta_data().get("fps",16) or 16
    a=np.asarray(r.get_data(int(min(n-2,12+int(round(6.0*fps)))))); r.close()
    pn,dn=desc(cv2.cvtColor(a,cv2.COLOR_RGB2BGR))
    best_ext=0; best_rep=0
    for k in P.files:
        if not k.startswith(sc+"|") or not k.endswith("|pts"): continue
        m=k.split("|")[1]
        if m in EXT or m.startswith("__ref"):
            best_ext=max(best_ext,inl(pn,dn,P[f"{sc}|{m}|pts"],P[f"{sc}|{m}|desc"]))
        elif m==REP:
            best_rep=inl(pn,dn,P[f"{sc}|{m}|pts"],P[f"{sc}|{m}|desc"])
    fam=max(best_ext,best_rep)
    rows.append(dict(scene=sc,inl_ext=best_ext,inl_fam=fam,reloc_ext=int(best_ext<50),reloc_fam=int(fam<50)))
    if i%64==0: print(f"[nc-v2] {i}/256",flush=True)
pd.DataFrame(rows).to_csv(os.path.join(HERE,"out","nocritic_reloc_v2.csv"),index=False); print("wrote nocritic_reloc_v2.csv",len(rows))
