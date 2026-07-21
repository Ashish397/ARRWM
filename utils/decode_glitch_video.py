import os, numpy as np, torch, cv2
from utils.wan_wrapper import WanVAEWrapper
from utils.zarr_dataset import ZarrRideDataset
dev="cuda"
RIDE=os.environ.get("RIDE","/projects/u6ex/fbots/frodobots_encoded_weunz/20240216101235.zarr")
NLAT=int(os.environ.get("NLAT","600"))   # 600 latents = 2400 frames = 120s @20fps
OUT=os.environ.get("OUT","analysis/glitch_decode_2min.mp4")
FPS=20
vae=WanVAEWrapper().eval().requires_grad_(False).to(dev)
import zarr
g=zarr.open(RIDE,'r'); navail=g['latents'].shape[0]; NLAT=min(NLAT,navail)
print("ride latents:",navail,"decoding",NLAT,"-> ~",NLAT*4,"frames =",round(NLAT*4/FPS),"s")
frames=[]
for s in range(0,NLAT,20):
    lat=ZarrRideDataset.load_latent_chunk(RIDE,s,min(s+20,NLAT)).unsqueeze(0).to(dev).float()
    latwd=torch.cat([lat[:,0:1],lat],1)
    px=vae.decode_to_pixel(latwd)[:,1:,...]
    v=(0.5*(px.float()+1)).clamp(0,1)[0]
    if v.shape[-1]!=3: v=v.permute(0,2,3,1)
    frames.append((v.cpu().numpy()*255).astype(np.uint8))
vid=np.concatenate(frames,0)  # [F,H,W,3] RGB
H,W=vid.shape[1:3]
vw=cv2.VideoWriter(OUT,cv2.VideoWriter_fourcc(*'mp4v'),FPS,(W,H))
for i,f in enumerate(vid):
    bgr=cv2.cvtColor(f,cv2.COLOR_RGB2BGR).copy()
    lat=i//4; t=i/FPS
    txt=f"f{i} lat{lat} {int(t//60)}:{t%60:05.2f}"
    cv2.rectangle(bgr,(0,0),(330,28),(0,0,0),-1)
    cv2.putText(bgr,txt,(6,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
    vw.write(bgr)
vw.release()
print("saved",OUT,"|",len(vid),"frames | block boundaries at lat 0,151,302,453 -> ~0s,30s,60s,90s")
