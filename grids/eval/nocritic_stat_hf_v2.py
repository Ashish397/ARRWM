"""No Critic stationary HF under HF-Cal-v2 (median over 6 externals). Replicates stat_hf.py exactly."""
import os, cv2, numpy as np, pandas as pd
NC="/home/ashish/ARRWM/logs/eval_final/B_Astarts_nullinv/nocritic/control_test"
HERE=os.path.dirname(os.path.abspath(__file__)); SIZE=(832,448); CTX=12
def lap_var(rgb):
    g=cv2.cvtColor(cv2.resize(rgb,SIZE),cv2.COLOR_RGB2GRAY); return float(cv2.Laplacian(g,cv2.CV_64F).var())
def frames_at(p,idxs):
    cap=cv2.VideoCapture(p); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); want={i for i in idxs if 0<=i<n}
    got={}; i=0
    while len(got)<len(want):
        ok,f=cap.read()
        if not ok: break
        if i in want: got[i]=cv2.cvtColor(f,cv2.COLOR_BGR2RGB)
        i+=1
    cap.release(); return got,n
rows=[]
for sc in range(32):
    fp=os.path.join(NC,f"step05000_r{sc:02d}_static_raw.mp4")
    if not os.path.exists(fp): continue
    cap=cv2.VideoCapture(fp); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS) or 16; cap.release()
    b0=CTX+int(round(fps)); end=min(n-1,CTX+int(round(6.0*fps)))
    bi=[i for i in range(b0,b0+4) if i<n]; ei=list(range(max(CTX,end-14),end+1,2))
    if not bi or not ei: continue
    got,_=frames_at(fp,bi+ei)
    base=np.mean([lap_var(got[i]) for i in bi if i in got]); endv=np.mean([lap_var(got[i]) for i in ei if i in got])
    rows.append(dict(scene=sc,base_blur=round(base,1),end_blur=round(endv,1),d_blur=round(endv-base,1)))
d=pd.DataFrame(rows)
ref=pd.read_csv(os.path.join(HERE,"out","hf_reference_median_v2_stationary.csv"))
m=d.merge(ref,on='scene'); m['hf_v2']=((m.med_v2-m.d_blur)>150).astype(int)
m.to_csv(os.path.join(HERE,"out","nocritic_stat_hf_v2.csv"),index=False)
print(f"No Critic stationary: n={len(m)}  HF v2 = {m.hf_v2.mean()*100:.1f}%  (published v1 3.1%)")
