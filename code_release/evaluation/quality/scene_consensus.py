"""Sibling-consensus place identity: per scene, pairwise RANSAC inliers among all
model end-frames + real ref; each video's score = max inliers to any OTHER member.
Validate on the 10 hand-labeled scenes."""
import os
import os, sys
import cv2, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE=os.path.dirname(os.path.abspath(__file__))
STATIONARY_DIR = os.environ.get("AF_STATIONARY_DIR", "")
STATIONARY = bool(STATIONARY_DIR) and os.path.isdir(STATIONARY_DIR)
BASE = os.path.join(os.environ.get("AF_FLEET_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "grids")), "baselines")
MODELS={"astra":4,"matrixgame":1,"minwm":13,"worldcam":65,"worldplay":1,"yume":1}
OURS={"pca8":12,"pca4":12,"pca2":12,"16node":12,"4node":12,"noatok":12,"noadaln":12}
# Ablations rendered after the grid was built, supplied as standalone tiles.
for _v in os.environ.get("AF_EXTRA_VARIANTS","").split(","):
    if _v.strip(): OURS[_v.strip()]=12
orb=cv2.ORB_create(3000); bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
def frame_at(path,i):
    c=cv2.VideoCapture(path); c.set(cv2.CAP_PROP_POS_FRAMES,i); ok,f=c.read(); c.release()
    return f if ok else None
def meta(path):
    c=cv2.VideoCapture(path); fps=c.get(cv2.CAP_PROP_FPS) or 16; n=int(c.get(cv2.CAP_PROP_FRAME_COUNT)); c.release()
    return fps,n
def desc(img):
    g=cv2.cvtColor(cv2.resize(img,(640,352)),cv2.COLOR_BGR2GRAY)
    return orb.detectAndCompute(g,None)
def inl(kd0,kd1):
    (k0,d0),(k1,d1)=kd0,kd1
    if d0 is None or d1 is None or len(k0)<8 or len(k1)<8: return 0
    ms=bf.match(d0,d1)
    if len(ms)<8: return 0
    src=np.float32([k0[m.queryIdx].pt for m in ms]); dst=np.float32([k1[m.trainIdx].pt for m in ms])
    H,mask=cv2.findHomography(src,dst,cv2.RANSAC,5.0)
    return int(mask.sum()) if mask is not None else 0
# De-tiled rollouts of our variants. AF_TILES_DIR may be a colon-separated list.
TILE_DIRS=[d for d in os.environ.get("AF_TILES_DIR",
    os.pathsep.join([os.path.join(HERE,"tiles"),os.path.join(HERE,"tiles_new")])
    ).split(os.pathsep) if d]

def tile(s,v="pca8"):
    if STATIONARY:
        p=os.path.join(STATIONARY_DIR,f"{v}_{s}.mp4")
        return p if os.path.exists(p) else None
    for td in TILE_DIRS:
        p=os.path.join(td,f"{s}__{v}.mp4")
        if os.path.exists(p): return p
def run(scenes, out):
    rows=[]
    for sc in scenes:
        members={}
        rp=tile(sc)
        rf=frame_at(rp,11)
        if rf is None: continue
        members["__ref__"]=desc(rf)
        for _ri in (2,5,8):
            _rf=frame_at(rp,_ri)
            if _rf is not None: members[f"__ref{_ri}__"]=desc(_rf)
        if STATIONARY:
            jobs={m:(os.path.join(STATIONARY_DIR,f"{m}_{sc}.mp4"),c) for m,c in MODELS.items()}
        else:
            jobs={m:(os.path.join(BASE,f"A_{m}",f"{m}_{sc}.mp4"),c) for m,c in MODELS.items()}
        jobs.update({f"ours_{v}":(tile(sc,v),c) for v,c in OURS.items()})
        for name,(fp,ctx) in jobs.items():
            if not fp or not os.path.exists(fp): continue
            fps,n=meta(fp)
            ef=frame_at(fp,min(n-2,ctx+int(round(6.0*fps))))
            if ef is not None: members[name]=desc(ef)
        names=[k for k in members if not k.startswith("__ref")]
        for a in names:
            best=max(inl(members[a],members[b]) for b in members if b!=a)
        # full pairwise (cache): compute matrix
        keys=list(members)
        M={}
        for i,a in enumerate(keys):
            for b in keys[i+1:]:
                M[(a,b)]=M[(b,a)]=inl(members[a],members[b])
        for a in names:
            best=max(M[(a,b)] for b in keys if b!=a)
            rows.append({"scene":sc,"model":a,"consensus_inl":best})
        print(sc,flush=True)
    pd.DataFrame(rows).to_csv(out,index=False)
if __name__=="__main__":
    import glob
    if len(sys.argv)>1 and sys.argv[1]=="fleet":
        if STATIONARY:
            scenes=sorted({os.path.basename(p)[:-4].rsplit("_r",1)[1] for p in glob.glob(f"{STATIONARY_DIR}/real_r*.mp4")})
            scenes=[f"r{x}" for x in scenes]
        else:
            scenes=sorted(set(os.path.basename(p).split("_",1)[1][:-4] for p in glob.glob(f"{BASE}/A_astra/*.mp4")))
        run(scenes, os.environ.get("AF_CONSENSUS_OUT", "fleet_scene_consensus.csv"))
    else:
        run(["r00_B","r00_BL","r00_BR","r00_F","r00_FL","r00_FR","r00_L","r00_R","r01_B","r01_BL"],
            os.environ.get("AF_CONSENSUS_OUT", "results_scene_consensus_val.csv"))
