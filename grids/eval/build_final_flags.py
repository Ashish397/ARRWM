"""Canonical per-rollout flags under the FINAL v2 rule (directional set), one file for
tables + figures so they cannot drift:
  externals : relocation vs flagship + other 5 externals + real ; HF median over {6 ext, pca8}
  ours (every variant incl. flagship and No Critic) : scored as the SOLE representative of
              our family -> relocation vs 6 externals + real ; HF median over {6 ext, itself}
Writes out/final_flags_v2.csv, out/hf_reference_external_dblur_v2_directional.csv, out/final_v2_numbers.json"""
import os, json, numpy as np, pandas as pd
from math import sqrt, erf
from scipy.stats import spearmanr
O=os.path.join(os.path.dirname(os.path.abspath(__file__)),"out")+"/"
EXT=['astra','matrixgame','minwm','worldcam','worldplay','yume']; REP='ours_pca8'
leg=pd.read_csv(O+'fleet_legit.csv'); leg['feature_valid']=leg.feature_valid.astype(bool); leg['static']=leg.static.astype(bool)
leg['active']=leg.feature_valid & ~leg.static
leg=leg.drop(columns=['consensus_inl','B','f_scene','f_hf','legit'])   # v1 columns, re-derived below
fb=pd.read_csv(O+'reloc_family_bias.csv')                       # single ORB run: c_extreal & c_onerep
fb['consensus_inl']=np.where(fb.model.str.startswith('ours_'),fb.c_extreal,fb.c_onerep)
hf=pd.read_csv(O+'fleet_hf.csv')
extd=hf[hf.model.isin(EXT)].pivot(index='scene',columns='model',values='d_blur')[EXT]
extd.to_csv(O+'hf_reference_external_dblur_v2_directional.csv')
rep_d=hf[hf.model==REP].set_index('scene').d_blur
def B_of(scene,model,d):
    seventh = d if model.startswith('ours_') else rep_d[scene]
    return float(np.median(list(extd.loc[scene].values)+[seventh]))-d
hf['B_v2']=[B_of(s,m,d) for s,m,d in zip(hf.scene,hf.model,hf.d_blur)]
x=leg.merge(fb[['scene','model','consensus_inl','c_any']],on=['scene','model'],how='left') \
     .merge(hf[['scene','model','d_blur','B','B_v2']],on=['scene','model'],how='left')
# ---- control rule: 'strength' (user rule) = wrong direction (cos<0) OR realised strength |z|<THR ; 'orb' = paper's original ORB near-static test
RULE=os.environ.get("CTRL_RULE","orb"); THR=float(os.environ.get("CTRL_THR","0.10")); ORB_T=int(os.environ.get("CTRL_ORB_T","500"))
ob=pd.read_csv(O+'fleet_obey_pca.csv'); ob['mag']=np.hypot(ob.g0,ob.g1)
x=x.merge(ob[['scene','model','mag']],on=['scene','model'],how='left')
si=pd.read_csv(os.path.join(os.path.dirname(O.rstrip('/')),'fleet_static_inl.csv'))[['scene','model','static_inl']]
x=x.merge(si,on=['scene','model'],how='left'); x['f_dis_orb600']=x.f_dis
import math
COS_T=math.cos(math.radians(float(os.environ.get("CTRL_COS_DEG","60"))))-1e-9   # direction test: fail if angle to command > CTRL_COS_DEG
x['f_dis']=((x.cos<COS_T)|(x.static_inl>ORB_T)).astype(float) if RULE=="orb" else ((x.cos<COS_T)|(x.mag<THR)).astype(float)
print(f"control rule = {RULE}  direction fail if angle > {os.environ.get('CTRL_COS_DEG','60')} deg; ORB first-vs-6s-frame inliers > {ORB_T} => near-static")
x['f_scene']=(x.consensus_inl<50).astype(float); x['f_hf']=(x.B_v2>150).astype(float)
x['f_scene_v1']=(x.c_any<50).astype(float);      x['f_hf_v1']=(x.B>150).astype(float)
# No Critic
nl=pd.read_csv(O+'nocritic_legit.csv'); nr=pd.read_csv(O+'nocritic_reloc_v2.csv'); nf=pd.read_csv(O+'nocritic_final_eval.csv')[['scene','d_blur','static_inl']]
n=nl.merge(nr,on='scene').merge(nf,on='scene')
ncd=pd.read_csv(O+'nocritic_dir_ctrl.csv')[['scene','mag']]; n=n.merge(ncd,on='scene',how='left')
n['f_dis_new']=((n.cos<COS_T)|(n.static_inl>ORB_T)).astype(float) if RULE=="orb" else ((n.cos<COS_T)|(n['mag']<THR)).astype(float)
n['B_v2']=[B_of(s,'ours_nocritic',d) for s,d in zip(n.scene,n.d_blur)]
nc=pd.DataFrame(dict(scene=n.scene,model='ours_nocritic',feature_valid=n.feature_valid.astype(bool),active=n.active.astype(bool),
    f_dis=n.f_dis_new,f_style=n.style_flag,f_geom=n.geom_flag,f_conj=n.conj_flag,
    consensus_inl=n.inl_ext,mag=n['mag'],f_scene=n.reloc_ext.astype(float),d_blur=n.d_blur,B_v2=n.B_v2,f_hf=(n.B_v2>150).astype(float),
    f_scene_v1=n.reloc_flag.astype(float),f_hf_v1=n.hf_flag.astype(float)))
cols=['scene','model','feature_valid','active','f_dis','mag','f_style','f_geom','f_scene','f_conj','f_hf','consensus_inl','d_blur','B_v2','f_scene_v1','f_hf_v1']
F=pd.concat([x[cols],nc[cols]],ignore_index=True)
F['legit']=(1-F.f_dis)*(1-F.f_style)*(1-F.f_geom)*(1-F.f_scene)*(1-F.f_conj)*(1-F.f_hf)
F['legit_v1']=(1-F.f_dis)*(1-F.f_style)*(1-F.f_geom)*(1-F.f_scene_v1)*(1-F.f_conj)*(1-F.f_hf_v1)
F.to_csv(O+'final_flags_v2.csv',index=False)
# ---- report ----
A=F[F.active]; V=F[F.feature_valid]
T=pd.DataFrame({'reloc_v1':A.groupby('model').f_scene_v1.mean()*100,'reloc':A.groupby('model').f_scene.mean()*100,
                'hf_v1':A.groupby('model').f_hf_v1.mean()*100,'hf':A.groupby('model').f_hf.mean()*100,
                'ctrl':V.groupby('model').f_dis.mean()*100,'legit_v1':V.groupby('model').legit_v1.mean()*100,'legit':V.groupby('model').legit.mean()*100,
                'clean':V.groupby('model').legit.sum(),'n_fv':V.groupby('model').size()})
ours=[m for m in T.index if m.startswith('ours_')]; ab=[m for m in ours if m!='ours_nocritic']; ext=[m for m in T.index if m not in ours]
print(T.round(1).to_string())
sh={'reloc_ours':(T.loc[ab,'reloc']-T.loc[ab,'reloc_v1']).mean(),'reloc_ext':(T.loc[ext,'reloc']-T.loc[ext,'reloc_v1']).mean(),
    'hf_ours':(T.loc[ab,'hf']-T.loc[ab,'hf_v1']).mean(),'hf_ext':(T.loc[ext,'hf']-T.loc[ext,'hf_v1']).mean()}
print("\nmean shift v1->v2 (7 ours variants / 6 externals):",{k:round(v,1) for k,v in sh.items()})
rs,rp=spearmanr(T.loc[ab,'reloc_v1'],T.loc[ab,'reloc']); hs,hp=spearmanr(T.loc[ab,'hf_v1'],T.loc[ab,'hf'])
print(f"ablation ordering v1 vs v2: RELOC rho={rs:.3f} p={rp:.3f} | HF rho={hs:.3f} p={hp:.3f}")
print("  reloc v1:", " < ".join(T.loc[ab].sort_values('reloc_v1').index.str.replace('ours_','')))
print("  reloc v2:", " < ".join(T.loc[ab].sort_values('reloc').index.str.replace('ours_','')))
print("  hf    v2:", " < ".join(T.loc[ab].sort_values('hf').index.str.replace('ours_','')))
def mc(m1,m2):
    a=V[V.model==m1].set_index('scene').legit; b=V[V.model==m2].set_index('scene').legit; s=a.index.intersection(b.index); a,b=a[s],b[s]
    n01=int(((a==1)&(b==0)).sum()); n10=int(((a==0)&(b==1)).sum()); z=(n01-n10)/sqrt(n01+n10) if n01+n10 else 0
    return dict(p1=round(a.mean()*100,1),p2=round(b.mean()*100,1),n01=n01,n10=n10,z=round(z,2),p=round(2*(1-0.5*(1+erf(abs(z)/1.4142))),3))
tests={'pca8_vs_minwm':mc('ours_pca8','minwm'),'pca8_vs_16node':mc('ours_pca8','ours_16node'),'16node_vs_minwm':mc('ours_16node','minwm')}
print("\nMcNemar:",tests)
json.dump({'table':T.round(1).to_dict(),'shift':{k:round(v,1) for k,v in sh.items()},
           'spearman':{'reloc':round(rs,3),'reloc_p':round(rp,3),'hf':round(hs,3),'hf_p':round(hp,3)},'mcnemar':tests},
          open(O+'final_v2_numbers.json','w'),indent=1)
print("wrote final_flags_v2.csv, final_v2_numbers.json, hf_reference_external_dblur_v2_directional.csv")
