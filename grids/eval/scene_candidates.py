"""Scene-relocation candidates vs hand-labeled GT. 3-way VLM + ORB + DINO."""
import os, sys, json
import cv2, numpy as np, torch, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm_external import INTRO, read_video, label_img, CTX_FRAMES, OURS_CTX, MODELS, OURS, BASE_DIR, HERE

GT = {  # relocation labels from visual inspection of ref-vs-end sheets
 "astra":      dict(zip(["r00_B","r00_BL","r00_BR","r00_F","r00_FL","r00_FR","r00_L","r00_R","r01_B","r01_BL"],[0,0,1,0,0,1,0,0,0,0])),
 "matrixgame": dict(zip(["r00_B","r00_BL","r00_BR","r00_F","r00_FL","r00_FR","r00_L","r00_R","r01_B","r01_BL"],[0,0,0,0,0,1,0,0,1,0])),
 "minwm":      dict(zip(["r00_B","r00_BL","r00_BR","r00_F","r00_FL","r00_FR","r00_L","r00_R","r01_B","r01_BL"],[0,1,0,0,0,0,1,0,0,0])),
 "worldcam":   dict(zip(["r00_B","r00_BL","r00_BR","r00_F","r00_FL","r00_FR","r00_L","r00_R","r01_B","r01_BL"],[0,1,1,0,1,1,1,1,1,1])),
 "worldplay":  dict(zip(["r00_B","r00_BL","r00_BR","r00_F","r00_FL","r00_FR","r00_L","r00_R","r01_B","r01_BL"],[0,0,1,0,1,1,1,1,1,1])),
 "yume":       dict(zip(["r00_B","r00_BL","r00_BR","r00_F","r00_FL","r00_FR","r00_L","r00_R","r01_B","r01_BL"],[0,1,1,0,1,1,0,0,1,1])),
 "ours_pca8":  {s:0 for s in ["r00_B","r00_BL","r00_BR","r00_F","r00_FL","r00_FR","r00_L","r00_R","r01_B","r01_BL"]},
 "ours_16node":{s:0 for s in ["r00_B","r00_BL","r00_BR","r00_F","r00_FL","r00_FR","r00_L","r00_R","r01_B","r01_BL"]},
}
SCENES = ["r00_B","r00_BL","r00_BR","r00_F","r00_FL","r00_FR","r00_L","r00_R","r01_B","r01_BL"]

TRIWAY = """Compare the GENERATED frames' final location with the REFERENCE. Choose ONE:
A = the video stays in the SAME place as the reference (allowing normal driving progress and any amount of visual corruption, distortion or artifacts - a damaged version of the same place is still the same place)
B = the video RELOCATES to a clearly DIFFERENT, coherent place (a different street, different buildings, a different kind of location)
Answer with ONLY {"answer": "A"} or {"answer": "B"}."""

def endframe(frames, fps, n1):
    return min(len(frames)-1, n1+int(round(6.0*fps)))

def main():
    from transformers import AutoProcessor
    from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", dtype=torch.bfloat16, device_map="cuda").eval()
    tok = proc.tokenizer
    a_id = tok.encode("A", add_special_tokens=False)[0]; b_id = tok.encode("B", add_special_tokens=False)[0]
    import timm
    dino = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224).to("cuda").eval()
    inet_m = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); inet_s = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
    orb = cv2.ORB_create(2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    rows=[]
    for scene in SCENES:
        for td in ("tiles","tiles_new"):
            p0=os.path.join(HERE,td,f"{scene}__pca8.mp4")
            if os.path.exists(p0):
                fr0,_=read_video(p0); break
        ref_np = fr0[8]
        ref_im = label_img(ref_np,"REFERENCE")
        kp0, des0 = orb.detectAndCompute(cv2.cvtColor(ref_np, cv2.COLOR_RGB2GRAY), None)
        with torch.no_grad():
            x=torch.from_numpy(ref_np.copy()).permute(2,0,1)[None].float()/255
            x=torch.nn.functional.interpolate(x,(224,224));  x=((x-inet_m)/inet_s).to("cuda")
            e_ref=torch.nn.functional.normalize(dino(x),dim=-1)
        vids={m: os.path.join(BASE_DIR,f"A_{m}",f"{m}_{scene}.mp4") for m in MODELS}
        for v in OURS:
            for td in ("tiles","tiles_new"):
                p1=os.path.join(HERE,td,f"{scene}__{v}.mp4")
                if os.path.exists(p1): vids[f"ours_{v}"]=p1; break
        for name,fp in vids.items():
            frames,fps=read_video(fp)
            n1=OURS_CTX if name.startswith("ours_") else CTX_FRAMES[name]
            ne=endframe(frames,fps,n1)
            endf = frames[ne]
            # VLM 3-way (A same incl. corrupted / B different)
            gi=np.linspace(n1,ne,16).round().astype(int)
            ims=[ref_im]+[label_img(frames[i],"GENERATED") for i in gi]
            content=[{"type":"image","image":im} for im in ims]+[{"type":"text","text":INTRO+"\n\n"+TRIWAY}]
            text=proc.apply_chat_template([{"role":"user","content":content}],add_generation_prompt=True,tokenize=False)+'{"answer": "'
            with torch.no_grad():
                inputs=proc(text=[text],images=ims,return_tensors="pt").to("cuda")
                lg=model(**inputs).logits[0,-1]
                p_reloc=float(torch.softmax(lg[torch.tensor([a_id,b_id],device=lg.device)],0)[1])
            # ORB matches ref->end
            g=cv2.cvtColor(endf,cv2.COLOR_RGB2GRAY)
            kp1,des1=orb.detectAndCompute(g,None)
            nm = len(bf.match(des0,des1)) if des0 is not None and des1 is not None else 0
            # DINO sim
            with torch.no_grad():
                x=torch.from_numpy(endf.copy()).permute(2,0,1)[None].float()/255
                x=torch.nn.functional.interpolate(x,(224,224)); x=((x-inet_m)/inet_s).to("cuda")
                e=torch.nn.functional.normalize(dino(x),dim=-1)
                dsim=float((e_ref@e.T))
            rows.append({"scene":scene,"model":name,"gt":GT[name][scene],
                         "p_reloc":round(p_reloc,4),"orb":nm,"dino":round(dsim,4)})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv("results_scene_cand.csv", index=False)
if __name__=="__main__": main()
