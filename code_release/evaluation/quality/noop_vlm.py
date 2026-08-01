"""No-op VLM probes: style shift + implausible, vs the shared real reference frame.
Writes noop_vlm_results.csv."""
import os
import os, numpy as np, torch, pandas as pd
from PIL import Image
import cv2
NOOP = os.path.join(os.environ.get("AF_FLEET_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "grids")), "noop")
ABL=["pca8","pca4","pca2","16node","4node","noatok","noadaln"]
CTX={**{a:9 for a in ABL},"astra":4,"matrixgame":1,"minwm":13,"worldcam":65,"worldplay":1,"yume":1}
def paths(model,sc):
    if model in ABL:
        p=f"{NOOP}/{model}_r{sc:02d}_noop.mp4"; return [p] if os.path.exists(p) else []
    suf={"astra":["NOOP"],"minwm":["F"],"worldcam":["F"],"worldplay":["NOOP"],"yume":["NOOP"],"matrixgame":["BL","BR"]}[model]
    return [f"{NOOP}/{model}_r{sc:02d}_{s}.mp4" for s in suf if os.path.exists(f"{NOOP}/{model}_r{sc:02d}_{s}.mp4")]
def readv(p):
    c=cv2.VideoCapture(p); fps=c.get(cv2.CAP_PROP_FPS) or 16; fr=[]
    while True:
        ok,f=c.read()
        if not ok: break
        fr.append(cv2.resize(cv2.cvtColor(f,cv2.COLOR_BGR2RGB),(640,352)))
    c.release(); return np.stack(fr),fps
def lab(fr,t):
    h=np.full((26,fr.shape[1],3),32,np.uint8); cv2.putText(h,t,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
    return Image.fromarray(np.concatenate([h,fr],0))
INTRO="The first image labelled REFERENCE is the last real frame of a driving video - the true scene. The remaining images labelled GENERATED are an AI world model's continuation, in temporal order. The command was to HOLD STILL, so the scene should stay the same place and style."
PROBES=[("style","Does the visual STYLE of the generated frames depart from the reference - painted, game-like, cartoonish, oversaturated, or a different rendering style? Lighting changes do NOT count. Answer Yes or No."),
        ("imp","Is there a SIGNIFICANT uncanny or reality-breaking failure in the generated frames - impossible geometry, surfaces dissolving into abstract patterns, large corrupted regions - a casual viewer would notice within one second? Ignore small artifacts and minor blur. Answer Yes or No.")]
from transformers import AutoProcessor
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
proc=AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
model=Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-8B-Instruct",dtype=torch.bfloat16,device_map="cuda").eval()
tok=proc.tokenizer; yid=tok.encode("Yes",add_special_tokens=False)[0]; nid=tok.encode("No",add_special_tokens=False)[0]
def pyes(ims,q):
    content=[{"type":"image","image":im} for im in ims]+[{"type":"text","text":INTRO+"\n\n"+q+'\n\nAnswer with ONLY {"answer": "Yes"} or {"answer": "No"}.'}]
    t=proc.apply_chat_template([{"role":"user","content":content}],add_generation_prompt=True,tokenize=False)+'{"answer": "'
    with torch.no_grad():
        i=proc(text=[t],images=ims,return_tensors="pt").to("cuda"); lg=model(**i).logits[0,-1]
        return float(torch.softmax(lg[torch.tensor([yid,nid],device=lg.device)],0)[0])
MODELS=ABL+["astra","matrixgame","minwm","worldcam","worldplay","yume"]
refs={}
for sc in range(32):
    p=paths("pca8",sc)
    if p: fr,_=readv(p[0]); refs[sc]=lab(fr[8],"REFERENCE")
out=open("noop_vlm_results.csv","a")
if os.path.getsize("noop_vlm_results.csv") if os.path.exists("noop_vlm_results.csv") else 0==0: out.write("model,scene,p_style,p_imp\n")
done=set()
if os.path.exists("noop_vlm_results.csv"):
    for l in open("noop_vlm_results.csv"):
        if l.startswith("model"): continue
        c=l.split(","); done.add((c[0],int(c[1])))
for m in MODELS:
    for sc in range(32):
        if (m,sc) in done or sc not in refs: continue
        ps=paths(m,sc)
        if not ps: continue
        vals={"style":[],"imp":[]}
        for p in ps:
            fr,fps=readv(p); n1=CTX[m]; ne=min(len(fr)-1,n1+int(round(6.0*fps)))
            gi=np.linspace(n1,ne,16).round().astype(int)
            ims=[refs[sc]]+[lab(fr[i],"GENERATED") for i in gi]
            for k,q in PROBES: vals[k].append(pyes(ims,q))
        out.write(f"{m},{sc},{round(np.mean(vals['style']),4)},{round(np.mean(vals['imp']),4)}\n"); out.flush()
    print(m,"done",flush=True)
out.close()
