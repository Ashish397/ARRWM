"""Fleet-wide scene-relocation scan: RANSAC-inlier place identity, CPU-only.
inliers between real ref frame (ours frame 8) and each model's 6s-horizon end frame.
Writes fleet_scene_reloc.csv incrementally with resume."""
import os, glob
import cv2, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("FLEET_BASE",
                      os.path.join(os.environ.get("AF_ROOT", "."), "grids", "baselines"))
MODELS={"astra":4,"matrixgame":1,"minwm":13,"worldcam":65,"worldplay":1,"yume":1}
OURS={"pca8":9,"16node":9}
THR=int(open(os.path.join(HERE,"scene_reloc_threshold.txt")).read())
OUT=os.path.join(HERE,"fleet_scene_reloc.csv")
orb=cv2.ORB_create(3000); bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
def frame_at(path,i):
    c=cv2.VideoCapture(path); c.set(cv2.CAP_PROP_POS_FRAMES,i); ok,f=c.read(); c.release()
    return f if ok else None
def meta(path):
    c=cv2.VideoCapture(path); fps=c.get(cv2.CAP_PROP_FPS) or 16; n=int(c.get(cv2.CAP_PROP_FRAME_COUNT)); c.release()
    return fps,n
def inliers(a,b):
    ga,gb=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY),cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)
    k0,d0=orb.detectAndCompute(ga,None); k1,d1=orb.detectAndCompute(gb,None)
    if d0 is None or d1 is None or len(k0)<8 or len(k1)<8: return 0
    ms=bf.match(d0,d1)
    if len(ms)<8: return 0
    src=np.float32([k0[m.queryIdx].pt for m in ms]); dst=np.float32([k1[m.trainIdx].pt for m in ms])
    H,mask=cv2.findHomography(src,dst,cv2.RANSAC,5.0)
    return int(mask.sum()) if mask is not None else 0
def tile(s):
    for td in ("tiles","tiles_new"):
        p=os.path.join(HERE,td,f"{s}__pca8.mp4")
        if os.path.exists(p): return p
def tile_v(s,v):
    for td in ("tiles","tiles_new"):
        p=os.path.join(HERE,td,f"{s}__{v}.mp4")
        if os.path.exists(p): return p
scenes=sorted(set(os.path.basename(p).split("_",1)[1][:-4] for p in glob.glob(f"{BASE}/A_astra/*.mp4")))
done=set()
if os.path.exists(OUT) and os.path.getsize(OUT)>0:
    prev=pd.read_csv(OUT); done=set(zip(prev.scene,prev.model))
f=open(OUT,"a")
if not done: f.write("scene,model,inliers,relocated\n")
for sc in scenes:
    tp=tile(sc)
    if not tp: continue
    ref=frame_at(tp,8)
    if ref is None: continue
    ref=cv2.resize(ref,(640,352))
    jobs={m:(os.path.join(BASE,f"A_{m}",f"{m}_{sc}.mp4"),c) for m,c in MODELS.items()}
    jobs.update({f"ours_{v}":(tile_v(sc,v),c) for v,c in OURS.items()})
    for name,(fp,ctx) in jobs.items():
        if (sc,name) in done or not fp or not os.path.exists(fp): continue
        fps,n=meta(fp)
        endf=frame_at(fp,min(n-2,ctx+int(round(6.0*fps))))
        if endf is None: continue
        inl=inliers(ref,cv2.resize(endf,(640,352)))
        f.write(f"{sc},{name},{inl},{int(inl<THR)}\n"); f.flush()
    print(sc, flush=True)
f.close(); print("done")
