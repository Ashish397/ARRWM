"""Exact binomial CIs and two-proportion tests for the paper's quality columns.

The columns are proportions over known denominators, so dispersion follows from
the counts already published - no extra runs. Wilson 95% intervals per cell,
two-sided two-proportion z between ours and each other model.
"""
import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import norm
Q = Path(__file__).resolve().parents[1] / "evaluation" / "quality"
ALIAS = {"ours_pca8":"pca8","ours_16node":"16node","ours_4node":"4node","ours_pca4":"pca4",
         "ours_pca2":"pca2","ours_noadaln":"noadaln","ours_noatok":"noatok"}
def csv(n):
    d = pd.read_csv(Q/n)
    if "model" in d: d["m"] = d.model.replace(ALIAS)
    return d
def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return 100*(c-h),100*(c+h)
def twoprop(k1,n1,k2,n2):
    p1,p2=k1/n1,k2/n2; p=(k1+k2)/(n1+n2); se=np.sqrt(p*(1-p)*(1/n1+1/n2))
    return 2*(1-norm.cdf(abs((p1-p2)/se)))
mask=csv("canonical_static_mask.csv")
act=mask[(mask.feature_valid==True)&(mask.active==1)][["m","scene"]]
tables={}
vlm=csv("results_external_vlm.csv")
tables['GEOMETRIC CORRUPTION']={m:(int((r.p_uncanny>0.5).sum()),len(r)) for m,r in vlm.groupby('m')}
j=csv("fleet_scene_consensus.csv").merge(act,on=["m","scene"])
tables['SCENE RELOCATION']={m:(int((r.consensus_inl<50).sum()),len(r)) for m,r in j.groupby('m')}
PUB={'worldplay','matrixgame','worldcam','astra','yume','minwm','pca8','16node'}
for name,T in tables.items():
    T={k:v for k,v in T.items() if k in PUB}
    print(f'\n{name}')
    for m,(k,n) in sorted(T.items(), key=lambda x:x[1][0]/x[1][1]):
        lo,hi=wilson(k,n); print(f'  {m:12} {100*k/n:5.1f}%  95% CI [{lo:5.1f}, {hi:5.1f}]   k={k}/{n}')
    ko,no=T['pca8']
    print('  ours (pca8) vs each, two-proportion z:')
    for m,(k,n) in sorted(T.items(), key=lambda x:x[1][0]/x[1][1]):
        if m=='pca8': continue
        p=twoprop(ko,no,k,n)
        print(f'    vs {m:12} p = {p:<10.3g} {"significant" if p<0.05 else "NOT significant"}')
