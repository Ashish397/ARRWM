import os, sys
import cv2, numpy as np, torch, pandas as pd
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm_external import read_video, CTX_FRAMES, OURS_CTX, MODELS, OURS, BASE_DIR, HERE
from PIL import Image

SCENES = ["r00_B","r00_BL","r00_BR","r00_F","r00_FL","r00_FR","r00_L","r00_R","r01_B","r01_BL"]

def main():
    import timm
    dino = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=518).to("cuda").eval()
    m_=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); s_=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
    @torch.no_grad()
    def feats(img):  # img RGB np -> (cls_norm, patches_norm)
        x=torch.from_numpy(img.copy()).permute(2,0,1)[None].float()/255
        x=F.interpolate(x,(518,518),mode="bilinear"); x=((x-m_)/s_).to("cuda")
        t=dino.forward_features(x)  # 1, 1+N, C
        cls=F.normalize(t[:,0],dim=-1); pt=F.normalize(t[0,1:],dim=-1)
        return cls[0].cpu(), pt.cpu()
    from transformers import AutoProcessor
    from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    vlm = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", dtype=torch.bfloat16, device_map="cuda").eval()
    tok=proc.tokenizer; a_id=tok.encode("A",add_special_tokens=False)[0]; b_id=tok.encode("B",add_special_tokens=False)[0]
    PAIR="""The LEFT half of the image is the last real frame of a driving video (the true place). The RIGHT half is where an AI continuation ended up. Is the RIGHT half the SAME place as the left (allowing driving progress and any amount of visual damage, distortion or artifacts - a damaged version of the same place is still the same place), or a DIFFERENT place (different street, different buildings)?
Answer with ONLY {"answer": "A"} for same place or {"answer": "B"} for different place."""
    @torch.no_grad()
    def p_pair(ref_np, end_np):
        comp = np.concatenate([cv2.resize(ref_np,(448,240)), np.full((240,8,3),255,np.uint8), cv2.resize(end_np,(448,240))],axis=1)
        im=Image.fromarray(comp)
        content=[{"type":"image","image":im},{"type":"text","text":PAIR}]
        text=proc.apply_chat_template([{"role":"user","content":content}],add_generation_prompt=True,tokenize=False)+'{"answer": "'
        inputs=proc(text=[text],images=[im],return_tensors="pt").to("cuda")
        lg=vlm(**inputs).logits[0,-1]
        return float(torch.softmax(lg[torch.tensor([a_id,b_id],device=lg.device)],0)[1])

    # reference features per scene (all 9 real frames)
    refs={}
    for sc in SCENES:
        for td in ("tiles","tiles_new"):
            p0=os.path.join(HERE,td,f"{sc}__pca8.mp4")
            if os.path.exists(p0):
                fr0,_=read_video(p0); break
        cls_list=[]; pat_list=[]
        for i in range(0,9,2):
            c,p=feats(fr0[i]); cls_list.append(c); pat_list.append(p)
        refs[sc]={"np":fr0[8], "cls":torch.stack(cls_list), "pat":torch.cat(pat_list)}
    rows=[]
    for sc in SCENES:
        vids={m: os.path.join(BASE_DIR,f"A_{m}",f"{m}_{sc}.mp4") for m in MODELS}
        for v in OURS:
            for td in ("tiles","tiles_new"):
                p1=os.path.join(HERE,td,f"{sc}__{v}.mp4")
                if os.path.exists(p1): vids[f"ours_{v}"]=p1; break
        for name,fp in vids.items():
            frames,fps=read_video(fp)
            n1=OURS_CTX if name.startswith("ours_") else CTX_FRAMES[name]
            ne=min(len(frames)-1, n1+int(round(6.0*fps)))
            late_idx=[n1+int((ne-n1)*f) for f in (0.5,0.75,1.0)]
            cls_l=[]; patch_match=[]
            for i in late_idx:
                c,p=feats(frames[i]); cls_l.append(c)
                sim = p @ refs[sc]["pat"].T          # N_end x N_ref
                patch_match.append(float((sim.max(1).values>0.6).float().mean()))
            cls_end=torch.stack(cls_l)
            own = float((cls_end @ refs[sc]["cls"].T).max(1).values.mean())
            others=[]
            for sc2 in SCENES:
                if sc2==sc: continue
                others.append(float((cls_end @ refs[sc2]["cls"].T).max(1).values.mean()))
            margin = own - max(others)
            sim_sustained = float((cls_end @ refs[sc]["cls"].T).max(1).values.min())
            ppair = p_pair(refs[sc]["np"], frames[ne])
            rows.append({"scene":sc,"model":name,"patch_match":round(np.mean(patch_match),4),
                         "retr_margin":round(margin,4),"sim_sustained":round(sim_sustained,4),
                         "p_pair":round(ppair,4)})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv("results_scene_round2.csv", index=False)
if __name__=="__main__": main()
