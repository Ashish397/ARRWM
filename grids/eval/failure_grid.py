"""Per-rollout failure map (final v2 rule; reads out/final_flags_v2.csv): each cell = one directional rollout.
240 evaluated rollouts per model laid out as a 4-wide x 60-tall rectangle.
Colour = which axis failed (priority order); neutral = passes all six.
The two wet-lens contexts are excluded upstream and are not shown."""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE=os.path.dirname(os.path.abspath(__file__)); O=os.path.join(HERE,"out")
d=pd.read_csv(f"{O}/final_flags_v2.csv")               # canonical flags: same file feeds the tables
leg=d[d.model!='ours_nocritic']
d['f_scene_v2']=d.f_scene; d['f_hf_v2']=d.f_hf
d=d[d.feature_valid.astype(bool)]                      # 240 evaluated only
AX=[('f_dis','Control','#111111'),('f_geom','Geometry','#d1495b'),
    ('f_scene_v2','Relocation','#1f77b4'),('f_hf_v2','HF','#ff8c1a'),
    ('f_style','Style','#7b3fa0'),('f_conj','Conjuration','#2a9d3f')]
PASS='#f2f2f2'
LABEL_FS=22
disp={'ours_pca8':'Ours (Default)','ours_pca4':'pca4','ours_pca2':'pca2','ours_16node':'Batch64',
 'ours_4node':'Batch16','ours_noatok':'No Act Tok','ours_noadaln':'No AdaLN','ours_nocritic':'No Critic',
 'minwm':'minWM','matrixgame':'Matrix-Game','worldplay':'WorldPlay','worldcam':'WorldCam',
 'astra':'Astra','yume':'Yume'}
order=['ours_pca8','ours_pca4','ours_pca2','ours_16node','ours_4node','ours_noatok','ours_noadaln','ours_nocritic',
       'minwm','matrixgame','worldplay','worldcam','astra','yume']
W,H=4,60                                               # 4 wide x 60 tall = 240
scenes=sorted(leg.scene.unique()); assert len(scenes)==W*H, len(scenes)
pos={s:i for i,s in enumerate(scenes)}

def render(models,stem,figw):
    fig,axes=plt.subplots(1,len(models),figsize=(figw,10.5))
    if len(models)==1: axes=[axes]
    for ax,mod in zip(axes,models):
        sub=d[d.model==mod]
        img=np.zeros((H,W,3)); img[:]=matplotlib.colors.to_rgb(PASS)
        nfail=0
        for _,r in sub.iterrows():
            i=pos.get(r.scene)
            if i is None: continue
            y,x=divmod(i,W)
            for c,_l,hexc in AX:
                if float(r.get(c,0))==1:
                    img[y,x]=matplotlib.colors.to_rgb(hexc); nfail+=1; break
        ax.imshow(img,interpolation='nearest',aspect='equal')
        ax.set_xticks([]); ax.set_yticks([])
        # label beside the bar (reads upward along its left edge) so it can be large and the bar can use the full height
        ax.annotate(f"{disp[mod]} \u2013 {len(sub)-nfail}/{len(sub)}",xy=(0,0.5),xycoords='axes fraction',
                    xytext=(-20,0),textcoords='offset points',rotation=90,ha='center',va='center',
                    fontsize=LABEL_FS)
        for sp in ax.spines.values(): sp.set_linewidth(0.4); sp.set_color('#888')
    h=[Patch(facecolor=PASS,edgecolor='#999',label='passes all')]+[Patch(facecolor=c,label=l) for _,l,c in AX]
    fig.legend(handles=h,loc='lower center',ncol=4,fontsize=18,frameon=False,bbox_to_anchor=(0.5,0.0),handlelength=1.6,columnspacing=1.6,handletextpad=0.6)
    fig.subplots_adjust(top=0.985,bottom=0.10,left=0.06,right=0.995,wspace=0.95)
    for ext in ('pdf','png'):
        fig.savefig(os.path.join(O,f"{stem}.{ext}"),dpi=200,bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {stem}.pdf/.png  ({len(models)} panels)")
    for mod in models:
        sub=d[d.model==mod]; cols=[c for c,_,_ in AX]
        print(f"    {disp[mod]:15s} clean {int((sub[cols].sum(axis=1)==0).sum()):3d}/{len(sub)}")

ABL=['ours_pca8','ours_pca4','ours_pca2','ours_16node','ours_4node','ours_noatok','ours_noadaln','ours_nocritic']
render(order,"failure_grid_all",19.5)
render(ABL,"failure_grid_ablations",11.5)
EXTF=["ours_pca8","minwm","matrixgame","worldplay","worldcam","astra","yume"]
render(EXTF,"failure_grid_external",10.2)
