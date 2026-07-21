"""Motion calibration of minWM smoke videos with OUR robust metrics:
incremental-SIFT recession% + cumulative yaw rot, vs our model's numbers."""
import os, numpy as np, imageio, cv2
det = cv2.SIFT_create(); bf = cv2.BFMatcher()
def gray(fr): return cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY)
def robust(path, stride=3):
    r = imageio.get_reader(path)
    frames = [np.asarray(f) for f in r]; r.close()
    logs, rots = [], []
    prev = None
    for i in range(0, len(frames), stride):
        g = gray(frames[i])
        if prev is not None:
            k0,d0 = det.detectAndCompute(prev,None); kt,dt = det.detectAndCompute(g,None)
            if d0 is not None and dt is not None:
                good=[a for a,b in bf.knnMatch(d0,dt,k=2) if a.distance<0.75*b.distance]
                if len(good)>=8:
                    src=np.float32([k0[x.queryIdx].pt for x in good]).reshape(-1,1,2)
                    dst=np.float32([kt[x.trainIdx].pt for x in good]).reshape(-1,1,2)
                    M,_=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=3)
                    if M is not None:
                        s=float(np.sqrt(M[0,0]**2+M[0,1]**2))
                        if 0.5<s<2.0:
                            logs.append(np.log(s)); rots.append(np.degrees(np.arctan2(M[1,0],M[0,0])))
        prev = g
    return 100*(1-np.exp(np.sum(logs))), float(np.sum(rots))
print(f"{'video':22} {'recession%':>10} {'cum_rot':>8}")
for wi in (8,1,2):
    for d in ("F","B","L","R"):
        p=f"logs/eval_final/minwm_smoke/minwm_r{wi:02d}_{d}.mp4"
        if os.path.exists(p):
            rec,rot=robust(p)
            print(f"minwm_r{wi:02d}_{d:2}        {rec:10.1f} {rot:8.1f}")
# reference: OUR model on r08 (from r08_robust_motion.csv)
import pandas as pd
df=pd.read_csv("analysis/eval_final/r08_robust_motion.csv")
s=df[df.run=="pca8_8node"]
print("\nOURS pca8_8node r08 (0.5 cmd): F rec=%.1f%% | B rec=%.1f%% | L rot=%.1f | R rot=%.1f" % (
    s[s.branch=="F"].recession.iloc[0], s[s.branch=="B"].recession.iloc[0],
    s[s.branch=="L"].rot.iloc[0], s[s.branch=="R"].rot.iloc[0]))
