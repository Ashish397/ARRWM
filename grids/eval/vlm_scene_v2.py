import json, os, sys
import numpy as np, torch, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm_external import INTRO, read_video, label_img, CTX_FRAMES, OURS_CTX, MODELS, OURS, BASE_DIR, HERE

SCENE_V2 = """Does the generated video RELOCATE to a clearly DIFFERENT, coherent place than the reference - a different street with different buildings, a different type of location? Corrupted, distorted, or artifact-ridden views of the SAME place do NOT count as a scene change; normal driving progress does NOT count. Answer Yes or No."""

def main():
    scenes = sys.argv[1:]
    from transformers import AutoProcessor
    from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", dtype=torch.bfloat16, device_map="cuda").eval()
    tok = proc.tokenizer
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]; no_id = tok.encode("No", add_special_tokens=False)[0]
    rows=[]
    for scene in scenes:
        ref=None
        for td in ("tiles","tiles_new"):
            p0=os.path.join(HERE,td,f"{scene}__pca8.mp4")
            if os.path.exists(p0):
                fr0,_=read_video(p0); ref=label_img(fr0[8],"REFERENCE"); break
        vids={m: os.path.join(BASE_DIR,f"A_{m}",f"{m}_{scene}.mp4") for m in MODELS}
        for v in OURS:
            for td in ("tiles","tiles_new"):
                p1=os.path.join(HERE,td,f"{scene}__{v}.mp4")
                if os.path.exists(p1): vids[f"ours_{v}"]=p1; break
        for name,fp in vids.items():
            if not os.path.exists(fp): continue
            frames,fps=read_video(fp)
            n1=OURS_CTX if name.startswith("ours_") else CTX_FRAMES[name]
            if len(frames)<=n1+int(fps): continue
            n_end=min(len(frames)-1,n1+int(round(6.0*fps)))
            gi=np.linspace(n1,n_end,16).round().astype(int)
            ims=[ref]+[label_img(frames[i],"GENERATED") for i in gi]
            content=[{"type":"image","image":im} for im in ims]+[{"type":"text","text":INTRO+"\n\n"+SCENE_V2+'\n\nAnswer with ONLY {"answer": "Yes"} or {"answer": "No"}.'}]
            text=proc.apply_chat_template([{"role":"user","content":content}],add_generation_prompt=True,tokenize=False)+'{"answer": "'
            with torch.no_grad():
                inputs=proc(text=[text],images=ims,return_tensors="pt").to("cuda")
                lg=model(**inputs).logits[0,-1]
                p=float(torch.softmax(lg[torch.tensor([yes_id,no_id],device=lg.device)],0)[0])
            rows.append({"scene":scene,"model":name,"p_scene_v2":round(p,4)})
            print(f"{scene} {name}: {p:.2f}", flush=True)
    pd.DataFrame(rows).to_csv("results_scene_v2.csv", index=False)
if __name__=="__main__": main()
