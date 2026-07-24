"""SIFT + multi-frame-anchor consensus, validation on the 10 labeled scenes."""
import os, sys
import cv2, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__))
BASE="/home/ashish/ARRWM/grids/baselines"
MODELS={"astra":4,"matrixgame":1,"minwm":13,"worldcam":65,"worldplay":1,"yume":1}
OURS={"pca8":9,"16node":9}
sift=cv2.SIFT_create(3000)
flann=cv2.BFMatcher(cv2.NORM_L2)
def frame_at(path,i):
    c=cv2.VideoCapture(path); c.set(cv2.CAP_PROP_POS_FRAMES,i); ok,f=c.read(); c.release()
    return f if ok else None
def meta(path):
    c=cv2.VideoCapture(path); fps=c.get(cv2.CAP_PROP_FPS) or 16; n=int(c.get(cv2.CAP_PROP_FRAME_COUNT)); c.release()
    return fps,n
def desc(img):
    g=cv2.cvtColor(cv2.resize(img,(640,352)),cv2.COLOR_BGR2GRAY)
    return sift.detectAndCompute(g,None)
def inl(kd0,kd1):
    (k0,d0),(k1,d1)=kd0,kd1
    if d0 is None or d1 is None or len(k0)<8 or len(k1)<8: return 0
    ms=flann.knnMatch(d0,d1,k=2)
    good=[m for m,n in ms if m.distance<0.75*n.distance]
    if len(good)<8: return 0
    src=np.float32([k0[m.queryIdx].pt for m in good]); dst=np.float32([k1[m.trainIdx].pt for m in good])
    H,mask=cv2.findHomography(src,dst,cv2.RANSAC,5.0)
    return int(mask.sum()) if mask is not None else 0
def tile(s,v="pca8"):
    for td in ("tiles","tiles_new"):
        p=os.path.join(HERE,td,f"{s}__{v}.mp4")
        if os.path.exists(p): return p
SCENES=["r00_B","r00_BL","r00_BR","r00_F","r00_FL","r00_FR","r00_L","r00_R","r01_B","r01_BL"]
rows=[]
for sc in SCENES:
    # anchors: real ref + each member's END and MID frames
    anchors={}   # name -> list of descriptor sets
    ends={}
    rf=frame_at(tile(sc),8)
    anchors["__ref__"]=[desc(rf)]
    jobs={m:(os.path.join(BASE,f"A_{m}",f"{m}_{sc}.mp4"),c) for m,c in MODELS.items()}
    jobs.update({f"ours_{v}":(tile(sc,v),c) for v,c in OURS.items()})
    for name,(fp,ctx) in jobs.items():
        if not fp or not os.path.exists(fp): continue
        fps,n=meta(fp)
        ne=min(n-2,ctx+int(round(6.0*fps))); nm=ctx+(ne-ctx)//2
        e=frame_at(fp,ne); m_=frame_at(fp,nm)
        if e is None: continue
        ends[name]=desc(e)
        anchors[name]=[ends[name]] + ([desc(m_)] if m_ is not None else [])
    for a in ends:
        best=0
        for b,dl in anchors.items():
            if b==a: continue
            for kd in dl:
                best=max(best, inl(ends[a],kd))
        rows.append({"scene":sc,"model":a,"sift_inl":best})
        print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv("results_scene_sift.csv",index=False)
