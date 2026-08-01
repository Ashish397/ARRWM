"""Is no-critic's quality advantage real, or a survivorship effect?

Its active population is 92 of 256; Default's is 239. Quality percentages are
computed over each model's own actives, so the two columns describe different
rollout sets. Recompute both on the scenes where BOTH are active.
"""
import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import norm
R = Path('/scratch/u6ex/as1748.u6ex/ARRWM/code_release/evaluation/quality/reference')
N = Path('/scratch/u6ex/as1748.u6ex/ARRWM/analysis/nocritic_ablation/results')
mask = pd.read_csv(R/'canonical_static_mask.csv')
mask['m'] = mask.model.replace({'ours_pca8':'pca8'})
d_act = set(mask[(mask.feature_valid==True)&(mask.active==1)&(mask.m=='pca8')].scene)
nst = pd.read_csv(N/'static_nocritic.csv')
n_act = set(nst[nst.static==0].scene)
both = d_act & n_act
print(f'Default active      {len(d_act)}')
print(f'no-critic active    {len(n_act)}')
print(f'both active         {len(both)}\n')

def twoprop(k1,n1,k2,n2):
    if min(n1,n2)==0: return float('nan')
    p1,p2=k1/n1,k2/n2; p=(k1+k2)/(n1+n2); se=np.sqrt(p*(1-p)*(1/n1+1/n2))
    return 2*(1-norm.cdf(abs((p1-p2)/se))) if se else float('nan')

AX = [
 ('geometric corruption', pd.read_csv(R/'results_external_vlm.csv'), pd.read_csv(N/'vlm_nocritic.csv'), lambda d: d.p_uncanny>0.5),
 ('scene relocation',     pd.read_csv(R/'fleet_scene_consensus.csv'), pd.read_csv(N/'consensus_nocritic.csv'), lambda d: d.consensus_inl<50),
 ('style shift',          pd.read_csv(R/'fleet_style_6s.csv'), pd.read_csv(N/'style_nocritic.csv'), lambda d: d.dino_drift>0.72),
 ('high-freq degradation',pd.read_csv(R/'fleet_hf.csv'), pd.read_csv(N/'hf_nocritic.csv'), lambda d: d.B>150),
 ('conjuration',          pd.read_csv(R/'popin_fleet_all.csv'), pd.read_csv(N/'popin_nocritic.csv'), lambda d: d.flag==1),
]
print(f'{"axis":24}{"Default":>22}{"no-critic":>22}{"p":>10}')
print(f'{"":24}{"own act.  both-act.":>22}{"own act.  both-act.":>22}')
for name, dref, dn, rule in AX:
    dref = dref.copy(); dref['m'] = dref.model.replace({'ours_pca8':'pca8'})
    D = dref[dref.m=='pca8']
    def pct(df, scenes):
        s = df[df.scene.isin(scenes)]
        return (int(rule(s).sum()), len(s))
    dk,dn_own = pct(D, d_act); db,db_n = pct(D, both)
    nk,nn_own = pct(dn, n_act); nb,nb_n = pct(dn, both)
    fmt=lambda k,n: f'{100*k/n:5.1f}%' if n else '   -- '
    p = twoprop(db,db_n,nb,nb_n)
    print(f'{name:24}{fmt(dk,dn_own):>11}{fmt(db,db_n):>11}{fmt(nk,nn_own):>11}{fmt(nb,nb_n):>11}{p:>10.3g}')
print(f'\n(both-active n: Default {db_n}, no-critic {nb_n}; p = two-proportion z on the both-active columns)')
