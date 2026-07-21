"""Prompt-framing probe: does PLAIN 'AI-generated?' prompting grade mangle better
than melt-vocabulary prompts? Qwen2.5-VL P(Yes) on full frames (no crops), one
row per (prompt, window, run). Windows: 3 anchors + r02_R; runs: our 7 + reals.
Output: analysis/eval_final/qwen_prompt_probe.csv
"""
import os, glob
import numpy as np, pandas as pd, imageio, torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
DEV = "cuda"
PROMPTS = {
    "plain_ai": "Is this image AI-generated? Answer only Yes or No.",
    "plain_model": "Was this image produced by a video generation model rather than a real camera? Answer only Yes or No.",
    "plain_fake_region": "Does any part of this image look computer-generated or fake rather than a real photograph? Answer only Yes or No.",
    "melt_control": ("Are any buildings, walls or structures in this image melted, warped, smeared "
                     "or geometrically impossible, like a corrupted AI-generated image? Answer only Yes or No."),
}
RUNS = ["pca8_8node", "pca4", "pca2", "16node", "4node", "noatok", "noadaln"]
WINDOWS = ["r08_B", "r08_BL", "r01_R", "r02_R"]

mid = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(mid, dtype=torch.bfloat16, device_map=DEV)
proc = AutoProcessor.from_pretrained(mid)
yes_ids = [proc.tokenizer.encode(t, add_special_tokens=False)[0] for t in ("Yes", " Yes", "yes")]
no_ids = [proc.tokenizer.encode(t, add_special_tokens=False)[0] for t in ("No", " No", "no")]


def p_yes(img, q):
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": q}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], return_tensors="pt").to(DEV)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    py = torch.logsumexp(logits[yes_ids], 0); pn = torch.logsumexp(logits[no_ids], 0)
    return float(torch.sigmoid(py - pn).item())


def score(path, q, start=13, stride=12):
    r = imageio.get_reader(path)
    fr = [np.asarray(f) for i, f in enumerate(r) if i >= start and (i - start) % stride == 0]
    r.close()
    vals = [p_yes(Image.fromarray(f), q) for f in fr]
    return float(np.mean(sorted(vals)[-4:])), float(np.mean(vals))


rows = []
for pname, q in PROMPTS.items():
    for w in WINDOWS:
        for run in RUNS:
            p = f"{ARR}/logs/eval_final/A/{run}/control_test/step05000_{w}_raw.mp4"
            if not os.path.exists(p):
                continue
            top4, mean = score(p, q)
            rows.append(dict(prompt=pname, window=w, run=run, top4=round(top4, 4), mean=round(mean, 4)))
            print(f"[{pname}] {w} {run}: top4={top4:.3f}", flush=True)
    n = 0
    for p in sorted(glob.glob(f"{ARR}/analysis/eval_final/real_refs/*.mp4")):
        r = imageio.get_reader(p); f0 = np.asarray(r.get_data(5)); r.close()
        if float(f0.mean()) < 60:
            continue
        top4, mean = score(p, q, start=0)
        rows.append(dict(prompt=pname, window="REAL", run=os.path.basename(p)[:22], top4=round(top4, 4), mean=round(mean, 4)))
        print(f"[{pname}] REAL {os.path.basename(p)[:22]}: top4={top4:.3f}", flush=True)
        n += 1
        if n >= 5:
            break

df = pd.DataFrame(rows)
df.to_csv(f"{ARR}/analysis/eval_final/qwen_prompt_probe.csv", index=False)
for pname in PROMPTS:
    print(f"\n===== {pname} =====")
    sub = df[df.prompt == pname]
    for w in WINDOWS:
        s = sub[sub.window == w].sort_values("top4", ascending=False)
        print(w, ":", " ".join(f"{r.run}:{r.top4:.2f}" for r in s.itertuples()))
    reals = sub[sub.window == "REAL"]["top4"]
    if len(reals):
        print("REAL max:", reals.max())
