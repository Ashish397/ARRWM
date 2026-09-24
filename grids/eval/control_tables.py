"""Per-command control-fail table from the canonical flags (+ no-op from the stationary readout at the same threshold)."""
import os, pandas as pd, numpy as np
O=os.path.join(os.path.dirname(os.path.abspath(__file__)),"out")+"/"; THR=float(os.environ.get("CTRL_THR","0.10"))
F=pd.read_csv(O+'final_flags_v2.csv'); F=F[F.feature_valid.astype(bool)].copy(); F['dir']=F.scene.str.split('_').str[1]
DIRS=['F','B','L','R','FL','FR','BL','BR']
C=F.pivot_table(index='model',columns='dir',values='f_dis',aggfunc='mean')[DIRS]*100; C['All']=F.groupby('model').f_dis.mean()*100
s=pd.read_csv(O+'stationary_cotracker_pca.csv'); s['mag']=np.hypot(s.z0,s.z1); noop=s.groupby('model').mag.apply(lambda x:(x>=THR).mean()*100)
nc=pd.read_csv(O+'nocritic_stat_ctrl.csv'); noop['nocritic']=(nc['mag']>=THR).mean()*100
name={'pca8':'ours_pca8','pca4':'ours_pca4','pca2':'ours_pca2','16node':'ours_16node','4node':'ours_4node','noatok':'ours_noatok','noadaln':'ours_noadaln','nocritic':'ours_nocritic'}
noop.index=[name.get(i,i) for i in noop.index]; C['Noop']=noop.reindex(C.index); C.loc['real']=[np.nan]*9+[noop['real']]
C.round(1).to_csv(O+'control_fail_by_direction_v2.csv'); print(C.round(0).fillna(-1).astype(int).replace(-1,'—').to_string())
