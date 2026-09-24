"""Stationary numbers under the FINAL v2 rule (same rule as build_final_flags.py)."""
import os, numpy as np, pandas as pd
O=os.path.join(os.path.dirname(os.path.abspath(__file__)),"out")+"/"
EXT=['astra','matrixgame','minwm','worldcam','worldplay','yume']
s=pd.read_csv(O+'stat_hf.csv')[['model','scene','d_blur','hf_flag']]
nc=pd.read_csv(O+'nocritic_stat_hf_v2.csv')[['scene','d_blur']].assign(model='nocritic',hf_flag=np.nan)
s=pd.concat([s,nc],ignore_index=True)
extd=s[s.model.isin(EXT)].pivot(index='scene',columns='model',values='d_blur')[EXT]
extd.to_csv(O+'hf_reference_external_dblur_v2_stationary.csv')
rep=s[s.model=='pca8'].set_index('scene').d_blur
s['B_v2']=[float(np.median(list(extd.loc[sc].values)+[d if m not in EXT else rep[sc]]))-d for m,sc,d in zip(s.model,s.scene,s.d_blur)]
s['hf_v2']=(s.B_v2>150).astype(int)
r=pd.read_csv(O+'stat_scene_v2.csv')
T=pd.DataFrame({'hf_v1':s.groupby('model').hf_flag.mean()*100,'hf':s.groupby('model').hf_v2.mean()*100,
                'reloc':r.groupby('model').reloc_final.mean()*100})
print(T.round(1).to_string()); T.round(1).to_csv(O+'final_stationary_v2.csv')
print("wrote final_stationary_v2.csv, hf_reference_external_dblur_v2_stationary.csv")
