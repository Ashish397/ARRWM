import os, sys
import cv2, numpy as np, torch, pandas as pd
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm_external import read_video, CTX_FRAMES, OURS_CTX, MODELS, OURS, BASE_DIR, HERE
from PIL import Image
SCENES = ["r00_B","r00_BL","r00_BR","r00_F","r00_FL","r00_FR","r00_L","r00_R","r01_B","r01_BL"]

def ransac_inliers(ref_g, end_g, orb, bf):
    k0,d0=orb.detectAndCompute(ref_g,None); k1,d1=orb.detectAndCompute(end_g,None)
    if d0 is None or d1 is None or len(k0)<8 or len(k1)<8: return 0
    ms=bf.match(d0,d1)
    if len(ms)<8: return 0
    src=np.float32([k0[m.queryIdx].pt for m in ms]); dst=np.float32([k1[m.trainIdx].pt for m in ms])
    H,mask=cv2.findHomography(src,dst,cv2.RANSAC,5.0)
    return int(mask.sum()) if mask is not None else 0

def main():
    import timm
    dino = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=518).to("cuda").eval()
    m_=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); s_=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
    @torch.no_grad()
    def patches(img):
        x=torch.from_numpy(img.copy()).permute(2,0,1)[None].float()/255
        x=F.interpolate(x,(518,518),mode="bilinear"); x=((x-m_)/s_).to("cuda")
        t=dino.forward_features(x)
        return F.normalize(t[0,1:],dim=-1).cpu()   # 1369 x C (37x37)
    from transformers import AutoProcessor
    from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    vlm = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", dtype=torch.bfloat16, device_map="cuda").eval()
    tok=proc.tokenizer; a_id=tok.encode("A",add_special_tokens=False)[0]; b_id=tok.encode("B",add_special_tokens=False)[0]
    PAIR_L="""The LEFT half is the last real frame of a driving video (the true place). The RIGHT half is where an AI continuation ended up. Same place = A (driving progress and visual damage still count as the same place - focus on whether the STREET LAYOUT and BUILDINGS are the same ones). Different place = B (different street, different buildings, even if similar in style).
Answer with ONLY {"answer": "A"} or {"answer": "B"}."""
    PAIR_R=PAIR_L.replace("LEFT half is the last real frame","RIGHT half is the last real frame").replace("The RIGHT half is where an AI continuation ended up","The LEFT half is where an AI continuation ended up")
    @torch.no_grad()
    def p_pair(ref_np, end_np):
        ps=[]
        for order,(l,r,pr) in enumerate([(ref_np,end_np,PAIR_L),(end_np,ref_np,PAIR_R)]):
            comp=np.concatenate([cv2.resize(l,(448,240)),np.full((240,8,3),255,np.uint8),cv2.resize(r,(448,240))],axis=1)
            im=Image.fromarray(comp)
            content=[{"type":"image","image":im},{"type":"text","text":pr}]
            text=proc.apply_chat_template([{"role":"user","content":content}],add_generation_prompt=True,tokenize=False)+'{"answer": "'
            inputs=proc(text=[text],images=[im],return_tensors="pt").to("cuda")
            lg=vlm(**inputs).logits[0,-1]
            ps.append(float(torch.softmax(lg[torch.tensor([a_id,b_id],device=lg.device)],0)[1]))
        return float(np.mean(ps))
    orb=cv2.ORB_create(3000); bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
    G=37
    yy,xx=np.meshgrid(np.arange(G),np.arange(G),indexing="ij")
    rows=[]
    for sc in SCENES:
        for td in ("tiles","tiles_new"):
            p0=os.path.join(HERE,td,f"{sc}__pca8.mp4")
            if os.path.exists(p0):
                fr0,_=read_video(p0); break
        ref_np=fr0[8]; ref_g=cv2.cvtColor(ref_np,cv2.COLOR_RGB2GRAY)
        ref_pat=patches(ref_np)
        vids={m: os.path.join(BASE_DIR,f"A_{m}",f"{m}_{sc}.mp4") for m in MODELS}
        for v in OURS:
            for td in ("tiles","tiles_new"):
                p1=os.path.join(HERE,td,f"{sc}__{v}.mp4")
                if os.path.exists(p1): vids[f"ours_{v}"]=p1; break
        for name,fp in vids.items():
            frames,fps=read_video(fp)
            n1=OURS_CTX if name.startswith("ours_") else CTX_FRAMES[name]
            ne=min(len(frames)-1,n1+int(round(6.0*fps)))
            nm=n1+(ne-n1)//2
            end_np=frames[ne]; end_g=cv2.cvtColor(end_np,cv2.COLOR_RGB2GRAY)
            inl=ransac_inliers(ref_g,end_g,orb,bf)
            # spatially-consistent patch matching
            ep=patches(end_np)
            sim=ep@ref_pat.T                    # 1369x1369
            best=sim.max(1).values; idx=sim.argmax(1).numpy()
            good=(best>0.55).numpy()
            if good.sum()>=20:
                ex,ey=xx.flatten()[good],yy.flatten()[good]
                rxm,rym=idx[good]%G, idx[good]//G
                cx=np.corrcoef(ex,rxm)[0,1]; cy=np.corrcoef(ey,rym)[0,1]
                spat=float(np.nan_to_num((cx+cy)/2))
            else:
                spat=0.0
            pp=max(p_pair(ref_np,end_np), p_pair(ref_np,frames[nm]))
            rows.append({"scene":sc,"model":name,"inliers":inl,"spat":round(spat,4),
                         "p_pair2":round(pp,4)})
            print(rows[-1],flush=True)
    pd.DataFrame(rows).to_csv("results_scene_round3.csv",index=False)
if __name__=="__main__": main()
