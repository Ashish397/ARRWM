"""Fleet static-by-inliers: inl(first_gen_frame, +4s frame) > 600 -> static."""
import os
import os, glob
import cv2, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.environ.get("AF_FLEET_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "grids")), "baselines")
CTX={"astra":4,"matrixgame":1,"minwm":13,"worldcam":65,"worldplay":1,"yume":1}
orb=cv2.ORB_create(3000); bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
def frame_at(p,i):
    c=cv2.VideoCapture(p); c.set(cv2.CAP_PROP_POS_FRAMES,i); ok,f=c.read(); c.release()
    return f if ok else None
def meta(p):
    c=cv2.VideoCapture(p); fps=c.get(cv2.CAP_PROP_FPS) or 16; n=int(c.get(cv2.CAP_PROP_FRAME_COUNT)); c.release(); return fps,n
def inl(a,b):
    ga,gb=[cv2.cvtColor(cv2.resize(x,(640,352)),cv2.COLOR_BGR2GRAY) for x in (a,b)]
    k0,d0=orb.detectAndCompute(ga,None); k1,d1=orb.detectAndCompute(gb,None)
    if d0 is None or d1 is None: return 0
    ms=bf.match(d0,d1)
    if len(ms)<8: return 0
    src=np.float32([k0[m.queryIdx].pt for m in ms]); dst=np.float32([k1[m.trainIdx].pt for m in ms])
    H,mask=cv2.findHomography(src,dst,cv2.RANSAC,5.0)
    return int(mask.sum()) if mask is not None else 0
# AF_STATIC_OUT keeps a new ablation out of the shipped reference artefact.
OUT=os.environ.get("AF_STATIC_OUT", os.path.join(HERE,"fleet_static_inl.csv"))
done=set()
if os.path.exists(OUT) and os.path.getsize(OUT)>0:
    done=set(zip(*pd.read_csv(OUT)[["scene","model"]].values.T.tolist()))
f=open(OUT,"a")
if not done: f.write("scene,model,static_inl,static\n")
TILE_DIRS=[os.path.join(HERE,"tiles"),os.path.join(HERE,"tiles_new")]
TILE_DIRS+=[d for d in os.environ.get("AF_TILES_DIR","").split(os.pathsep) if d]
OURS_VARIANTS=("pca8","pca4","pca2","16node","4node","noatok","noadaln")
OURS_VARIANTS+=tuple(v.strip() for v in os.environ.get("AF_EXTRA_VARIANTS","").split(",") if v.strip())

def tile(s,v):
    for td in TILE_DIRS:
        p=os.path.join(td,f"{s}__{v}.mp4")
        if os.path.exists(p): return p
scenes=sorted(set(os.path.basename(p).split("_",1)[1][:-4] for p in glob.glob(f"{BASE}/A_astra/*.mp4")))
for sc in scenes:
    jobs={m:(os.path.join(BASE,f"A_{m}",f"{m}_{sc}.mp4"),c) for m,c in CTX.items()}
    jobs.update({f"ours_{v}":(tile(sc,v),12) for v in OURS_VARIANTS})
    _only={m.strip() for m in os.environ.get("AF_STATIC_MODELS","").split(",") if m.strip()}
    for name,(fp,ctx) in jobs.items():
        if _only and name not in _only and name.replace("ours_","") not in _only: continue
        if (sc,name) in done or not fp or not os.path.exists(fp): continue
        fps,n=meta(fp)
        f0=frame_at(fp,ctx); f1=frame_at(fp,min(n-2,ctx+int(round(6.0*fps))))
        if f0 is None or f1 is None: continue
        v=inl(f0,f1)
        f.write(f"{sc},{name},{v},{int(v>600)}\n"); f.flush()
    print(sc,flush=True)
f.close()
