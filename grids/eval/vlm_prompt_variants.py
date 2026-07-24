"""Compare probe prompt variants on a small scene subset. Writes results_prompt_variants.csv."""
import json, os, sys
import numpy as np, torch, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm_external import INTRO, read_video, label_img, CTX_FRAMES, OURS_CTX, MODELS, OURS, BASE_DIR, HERE

VARIANTS = {
  "v1_uncanny": """Is there anything UNCANNY or reality-breaking in the generated frames - impossible geometry, surfaces dissolving into abstract patterns or polygons, objects or people that could not physically exist? Answer Yes or No.""",
  "v2_strict":  """Is there a SIGNIFICANT uncanny or reality-breaking failure in the generated frames - impossible geometry, surfaces dissolving into abstract patterns, large corrupted regions - that a casual viewer would notice within one second? Ignore small local artifacts and minor blur. Answer Yes or No.""",
  "v3_holistic": """Considering everything, could the generated frames pass as REAL dashcam footage of this scene? Answer Yes if they could pass as real, No if they could not.""",
}

def main():
    scenes = sys.argv[1:]
    from transformers import AutoProcessor
    from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", dtype=torch.bfloat16, device_map="cuda").eval()
    tok = proc.tokenizer
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]; no_id = tok.encode("No", add_special_tokens=False)[0]
    @torch.no_grad()
    def p_yes(ims, q):
        content=[{"type":"image","image":im} for im in ims]+[{"type":"text","text":INTRO+"\n\n"+q+'\n\nAnswer with ONLY {"answer": "Yes"} or {"answer": "No"}.'}]
        text=proc.apply_chat_template([{"role":"user","content":content}],add_generation_prompt=True,tokenize=False)+'{"answer": "'
        inputs=proc(text=[text],images=ims,return_tensors="pt").to("cuda")
        lg=model(**inputs).logits[0,-1]
        return float(torch.softmax(lg[torch.tensor([yes_id,no_id],device=lg.device)],0)[0])
    rows=[]
    fout=open("results_prompt_variants.csv","a")
    if os.path.getsize("results_prompt_variants.csv") if os.path.exists("results_prompt_variants.csv") else 0 == 0:
        fout.write("scene,model,variant,p\n")
    for scene in scenes:
        ref=None
        for td in ("tiles","tiles_new"):
            p0=os.path.join(HERE,td,f"{scene}__pca8.mp4")
            if os.path.exists(p0):
                fr0,_=read_video(p0); ref=label_img(fr0[8],"REFERENCE"); break
        if ref is None: continue
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
            for vn,q in VARIANTS.items():
                p=p_yes(ims,q)
                pv = 1-p if vn=="v3_holistic" else p   # v3: score = P(not real)
                fout.write(f"{scene},{name},{vn},{round(pv,4)}\n"); fout.flush()
            print(f"{scene} {name} done", flush=True)
    fout.close()

if __name__=="__main__": main()
