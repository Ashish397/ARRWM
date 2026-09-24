"""No Critic stationary control readout: CoTracker->PCA z, same teacher as
stationary_cotracker_pca.py. Control fail (no-op) = |z| >= 0.05 (moved when told to hold)."""
import os, glob
import numpy as np, torch, cv2, imageio, pandas as pd
NC="/home/ashish/ARRWM/logs/eval_final/B_Astarts_nullinv/nocritic/control_test"
HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.path.join(HERE,"out","nocritic_stat_ctrl.csv")
CK=os.path.join(os.path.dirname(os.path.dirname(HERE)),"action_query","checkpoints","ss_vae_8free.pt")
GRID=10; N=100; OUT_CHUNK=12; COMPUTE_T=48; SIZE=(832,448); CTX=12
SCALES=np.array([93.7,57.7,22.5,21.2,18.1,14.5,12.6,10.8],np.float32); DEV="cuda"
def read_gen(p,ctx):
    r=imageio.get_reader(p); n=r.count_frames()
    fr=[cv2.resize(np.asarray(r.get_data(int(i))),SIZE) for i in range(min(ctx,n-2),n)]
    r.close(); return np.stack(fr)
@torch.no_grad()
def teacher_read(vid,cot,mean,comp_T):
    out=[]
    for cs in range(0,vid.shape[1],COMPUTE_T):
        ch=vid[:,cs:min(cs+COMPUTE_T,vid.shape[1])]; n_out=ch.shape[1]//OUT_CHUNK
        if n_out==0: continue
        ch=ch[:,:n_out*OUT_CHUNK].clone()
        with torch.amp.autocast(device_type="cuda",enabled=True):
            tracks,_=cot(ch,grid_size=GRID)
        tw=tracks.reshape(1,n_out,OUT_CHUNK,N,2)
        mo=(tw[:,:,1:]-tw[:,:,:-1]).mean(dim=2).squeeze(0)
        out.append((mo.reshape(mo.shape[0],200).float()-mean)@comp_T)
    return torch.cat(out,0)[:,:8] if out else None
def main():
    cot=torch.hub.load("facebookresearch/co-tracker","cotracker3_offline").to(DEV).eval()
    for p in cot.parameters(): p.requires_grad_(False)
    ck=torch.load(CK,map_location="cpu",weights_only=False)
    mean=torch.tensor(np.asarray(ck["pca_mean"]),dtype=torch.float32,device=DEV)
    comp_T=torch.tensor(np.asarray(ck["pca_comp"]).T,dtype=torch.float32,device=DEV)
    sc=torch.tensor(SCALES,device=DEV); rows=[]
    for f in sorted(glob.glob(os.path.join(NC,"*_static_raw.mp4"))):
        scene=os.path.basename(f).split("_")[1]
        frames=read_gen(f,CTX)
        vid=torch.from_numpy(frames).permute(0,3,1,2)[None].float().to(DEV)
        P=teacher_read(vid,cot,mean,comp_T)
        if P is None: print("short",scene); continue
        z=torch.tanh(P/sc).mean(0).cpu().numpy()
        rows.append(dict(model="nocritic",scene=scene,z0=float(z[0]),z1=float(z[1])))
        print(f"{scene} z=({z[0]:+.4f},{z[1]:+.4f})",flush=True)
    d=pd.DataFrame(rows); d["mag"]=np.hypot(d.z0,d.z1); d["ctrl_fail"]=(d["mag"]>=0.05).astype(int)
    d.to_csv(OUT,index=False)
    print(f"\nNo Critic stationary control fail = {d.ctrl_fail.mean()*100:.1f}%  (n={len(d)}), mean|z|={d['mag'].mean():.4f}")
main()
