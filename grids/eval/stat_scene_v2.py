"""Stationary relocation under Reloc-Cal-v2 (geometry identical to stat_scene.py). ONE run emits:
  inl_ext = max vs externals(excluding self) + real ; inl_fam = max(inl_ext, vs flagship pca8)
  reloc_final: ours (incl. pca8, nocritic) -> inl_ext<50 (sole family representative);
               externals -> inl_fam<50 (flagship-inclusive panel)."""
import os, cv2, numpy as np, pandas as pd
DIR="/home/ashish/stationary_evaluation"; NC="/home/ashish/ARRWM/logs/eval_final/B_Astarts_nullinv/nocritic/control_test"
HERE=os.path.dirname(os.path.abspath(__file__))
OURS=["16node","4node","pca8","pca4","pca2","noatok","noadaln"]; EXT=["astra","matrixgame","minwm","worldcam","worldplay","yume"]
CTX={**{m:12 for m in OURS},"nocritic":12,"astra":4,"matrixgame":1,"minwm":13,"worldcam":65,"worldplay":1,"yume":1}
SIZE=(640,352); orb=cv2.ORB_create(3000); bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
def frames_at(p,idxs):
    cap=cv2.VideoCapture(p); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); out={}; want={i for i in idxs if 0<=i<n}; i=0
    while want:
        ok,f=cap.read()
        if not ok: break
        if i in want: out[i]=cv2.cvtColor(cv2.resize(f,SIZE),cv2.COLOR_BGR2GRAY); want.discard(i)
        i+=1
    cap.release(); return out,n
def meta(p):
    cap=cv2.VideoCapture(p); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS) or 16; cap.release(); return n,fps
def inliers(a,b):
    ka,da=orb.detectAndCompute(a,None); kb,db=orb.detectAndCompute(b,None)
    if da is None or db is None or len(ka)<8 or len(kb)<8: return 0
    m=bf.match(da,db)
    if len(m)<8: return 0
    src=np.float32([ka[x.queryIdx].pt for x in m]).reshape(-1,1,2); dst=np.float32([kb[x.trainIdx].pt for x in m]).reshape(-1,1,2)
    H,mask=cv2.findHomography(src,dst,cv2.RANSAC,5.0)
    return int(mask.sum()) if mask is not None else 0
rows=[]
for scene in range(32):
    rp=os.path.join(DIR,f"real_r{scene:02d}.mp4")
    if not os.path.exists(rp): continue
    rf,_=frames_at(rp,[8]); real=rf.get(8)
    end={}
    for m in OURS+EXT:
        fp=os.path.join(DIR,f"{m}_r{scene:02d}.mp4")
        if not os.path.exists(fp): continue
        n,fps=meta(fp); e=min(n-1,CTX[m]+int(round(6.0*fps))); fr,_=frames_at(fp,[e]); end[m]=fr.get(e)
    ncp=os.path.join(NC,f"step05000_r{scene:02d}_static_raw.mp4")
    if os.path.exists(ncp):
        n,fps=meta(ncp); e=min(n-1,12+int(round(6.0*fps))); fr,_=frames_at(ncp,[e]); end["nocritic"]=fr.get(e)
    for m,fm in end.items():
        if fm is None: continue
        ext_peers=[s for s in EXT if s!=m]
        cands=([real] if real is not None else [])+[end[s] for s in ext_peers if end.get(s) is not None]
        inl_ext=max((inliers(fm,c) for c in cands),default=0)
        inl_fam=max(inl_ext,inliers(fm,end["pca8"])) if (m!="pca8" and end.get("pca8") is not None) else inl_ext
        final=inl_fam if m in EXT else inl_ext
        rows.append(dict(model=m,scene=scene,inl_ext=inl_ext,inl_fam=inl_fam,
                         reloc_ext=int(inl_ext<50),reloc_fam=int(inl_fam<50),reloc_final=int(final<50)))
    print(f"scene {scene} done",flush=True)
d=pd.DataFrame(rows); d.to_csv(os.path.join(HERE,"out","stat_scene_v2.csv"),index=False)
print("\nSTATIONARY relocation final (%):"); print((d.groupby('model').reloc_final.mean()*100).round(1).to_string())
