import os, numpy as np, torch
from utils.wan_wrapper import WanVAEWrapper
from utils.zarr_dataset import ZarrRideDataset
import pyiqa, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
dev="cuda"; RIDE="/projects/u6ex/fbots/frodobots_encoded_weunz/20240216101235.zarr"
NLAT=460  # ~3 blocks (>2 boundaries)
vae=WanVAEWrapper().eval().requires_grad_(False).to(dev)
musiq=pyiqa.create_metric("musiq",device=dev)
frames=[]
for s in range(0,NLAT,20):  # decode in chunks to avoid OOM
    lat=ZarrRideDataset.load_latent_chunk(RIDE,s,min(s+20,NLAT)).unsqueeze(0).to(dev).float()
    latwd=torch.cat([lat[:,0:1],lat],1)
    px=vae.decode_to_pixel(latwd)[:,1:,...]
    v=(0.5*(px.float()+1)).clamp(0,1)[0]
    if v.shape[-1]!=3: v=v.permute(0,2,3,1)
    frames.append(v.cpu())
vid=torch.cat(frames,0)  # [F,H,W,3]
print("decoded frames:",vid.shape)
sc=[]
with torch.no_grad():
    for i in range(vid.shape[0]):
        t=vid[i].permute(2,0,1).unsqueeze(0).to(dev)
        sc.append(float(musiq(t).item()))
sc=np.array(sc)
np.save("analysis/glitch_musiq.npy",sc)
fig,ax=plt.subplots(figsize=(16,4)); ax.plot(sc,lw=0.8)
# mark predicted block boundaries (latent 0,151,302 -> pixel frame ~lat*4)
for b in (0,151,302,453):
    ax.axvline(b*4,color="r",ls="--",alpha=0.5)
ax.set_xlabel("pixel frame (20fps)"); ax.set_ylabel("MUSIQ (dips=glitch)"); ax.set_title("per-frame MUSIQ over a decoded ride (red=predicted block boundary @151 latents)")
fig.tight_layout(); fig.savefig("analysis/glitch_period.png",dpi=110)
# find dips: frames in bottom 5%
thr=np.percentile(sc,5); dips=np.where(sc<thr)[0]
print("MUSIQ p5 dip frames:",list(dips))
print("saved analysis/glitch_period.png")
