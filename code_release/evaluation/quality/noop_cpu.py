"""No-reference no-op metrics (CPU): stillness, drift, haze.
For a no-op command the model should HOLD the scene still. So:
  stillness = median ORB inliers between first generated frame and 6s horizon
              (HIGH = correctly held still); held_still % = inliers > 600.
  drift     = the scene relocated despite no-op: consensus inliers < 50 (BAD).
  haze      = sibling-relative Laplacian sharpness loss t=1s -> end (BAD).
Reference frame for a scene = pca8 no-op frame 8 (shared start).
Writes noop_cpu_results.csv.
"""
import os
import glob, os, re
import cv2, numpy as np, pandas as pd

NOOP = os.path.join(os.environ.get("AF_FLEET_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "grids")), "noop")
OUT = os.environ.get("AF_NOOP_OUT", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "noop_cpu_results.csv"))
ABL = ["pca8","pca4","pca2","16node","4node","noatok","noadaln"]
CTX = {**{a:9 for a in ABL}, "astra":4,"matrixgame":1,"minwm":13,"worldcam":65,"worldplay":1,"yume":1}
# filename -> list of paths per (model,scene)
def paths(model, scene):
    if model in ABL:
        p = f"{NOOP}/{model}_r{scene:02d}_noop.mp4"
        return [p] if os.path.exists(p) else []
    suf = {"astra":["NOOP"],"minwm":["F"],"worldcam":["F"],"worldplay":["NOOP"],
           "yume":["NOOP"],"matrixgame":["BL","BR"]}[model]
    return [f"{NOOP}/{model}_r{scene:02d}_{s}.mp4" for s in suf
            if os.path.exists(f"{NOOP}/{model}_r{scene:02d}_{s}.mp4")]

orb = cv2.ORB_create(3000); bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
def frame(path, i):
    c=cv2.VideoCapture(path); c.set(cv2.CAP_PROP_POS_FRAMES,i); ok,f=c.read(); c.release()
    return cv2.resize(f,(640,352)) if ok else None
def meta(path):
    c=cv2.VideoCapture(path); fps=c.get(cv2.CAP_PROP_FPS) or 16; n=int(c.get(cv2.CAP_PROP_FRAME_COUNT)); c.release(); return fps,n
def inl(a,b):
    ga,gb=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY),cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)
    k0,d0=orb.detectAndCompute(ga,None); k1,d1=orb.detectAndCompute(gb,None)
    if d0 is None or d1 is None or len(k0)<8 or len(k1)<8: return 0
    ms=bf.match(d0,d1)
    if len(ms)<8: return 0
    src=np.float32([k0[m.queryIdx].pt for m in ms]); dst=np.float32([k1[m.trainIdx].pt for m in ms])
    H,mask=cv2.findHomography(src,dst,cv2.RANSAC,5.0)
    return int(mask.sum()) if mask is not None else 0
def lap(f): return cv2.Laplacian(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()

MODELS=ABL+["astra","matrixgame","minwm","worldcam","worldplay","yume"]
SCENES=range(32)

# pass 1: gather first-gen + end frames per (model,scene); reference = pca8 frame 8
rows=[]
end_frames={}   # (model,scene)->end frame (for consensus), averaged desc for matrixgame handled by first path
ref_frames={}
for sc in SCENES:
    p=paths("pca8",sc)
    if p: ref_frames[sc]=frame(p[0],8)
for model in MODELS:
    for sc in SCENES:
        ps=paths(model,sc)
        if not ps: continue
        ctx=CTX[model]
        stills=[]; ends=[]; laps_base=[]; laps_end=[]
        for p in ps:
            fps,n=meta(p)
            fg=frame(p,ctx); ne=min(n-2,ctx+int(round(6.0*fps))); ef=frame(p,ne)
            b=frame(p,ctx+int(round(1.0*fps)))  # t=1s within generation
            if fg is None or ef is None: continue
            stills.append(inl(fg,ef))
            ends.append(ef)
            if b is not None:
                laps_base.append(lap(b)); laps_end.append(lap(ef))
        if not stills: continue
        still=float(np.mean(stills))
        end_frames[(model,sc)]=ends[0]
        lap_loss = float(np.mean(laps_base)-np.mean(laps_end)) if laps_base else np.nan
        rows.append({"model":model,"scene":sc,"stillness":round(still,1),
                     "lap_loss":round(lap_loss,1)})
df=pd.DataFrame(rows)

# pass 2: drift = consensus of end frame vs ref + all sibling ends per scene
def drift_row(r):
    sc=r.scene; ef=end_frames.get((r.model,sc))
    if ef is None: return np.nan
    cands=[ref_frames[sc]] if sc in ref_frames else []
    cands+=[end_frames[(m,sc)] for m in MODELS if (m,sc) in end_frames and m!=r.model]
    return max((inl(ef,c) for c in cands if c is not None), default=0)
df["drift_inl"]=df.apply(drift_row,axis=1)
# sibling-relative haze
df["haze"]=df.groupby("scene").lap_loss.transform(lambda s: s - s.median())
df.to_csv(OUT,index=False)
print(f"wrote {OUT} ({len(df)} rows)")
