"""Plausibility / novel-actor / style probes on blind100 (Qwen3-VL-8B, pointwise).

Reproduces the plausibility agent's EXACT protocol (verbatim INTRO + PROBES + logit
P(Yes) scoring from vlm_external.py). Pointwise: one forward pass per (video, probe),
4 REFERENCE + 16 GENERATED composited frames -> P(Yes). Higher = more of the failure.
  p_style   : rendering-style departs from reference
  p_novel   : NEW OBJECT / actor appears (the skier/robot/car detector)
  p_uncanny : significant reality-breaking / implausible failure

Reference frames come from each video's OWN real context (per-model ctx), so short-
context externals (yume/matrixgame/worldplay, ctx=1) anchor on real frames. ctx>=12
uses the code's (2,5,8,11); else 4 frames evenly in [0,ctx-1]. Generated =
linspace(ctx, min(n-1, ctx+6s), 16). Note: p_novel/p_uncanny are stable across the
reference-version variants; p_style (4-ref) may be inflated vs the single-ref version.

Writes out/blind_plausibility.csv.
"""
import os
import numpy as np, torch, pandas as pd
import blind100_common as bc
from vlm_external import INTRO, PROBES, read_video, label_img

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "blind_plausibility.csv")


def ref_indices(ctx):
    if ctx >= 12:
        return [2, 5, 8, 11]
    return list(np.linspace(0, max(0, ctx - 1), 4).round().astype(int))


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    from transformers import AutoProcessor
    from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct", dtype=torch.bfloat16, device_map="cuda").eval()
    tok = proc.tokenizer
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id = tok.encode("No", add_special_tokens=False)[0]

    @torch.no_grad()
    def p_yes(ims, question, nref=4):
        content = [{"type": "image", "image": im} for im in ims] + [
            {"type": "text", "text": INTRO.format(nref=nref) + "\n\n" + question +
             '\n\nAnswer with ONLY {"answer": "Yes"} or {"answer": "No"}.'}]
        text = proc.apply_chat_template([{"role": "user", "content": content}],
                                        add_generation_prompt=True, tokenize=False)
        text += '{"answer": "'
        inputs = proc(text=[text], images=ims, return_tensors="pt").to("cuda")
        logits = model(**inputs).logits[0, -1]
        p = torch.softmax(logits[torch.tensor([yes_id, no_id], device=logits.device)], 0)
        return float(p[0].item())

    rows = []
    for r in bc.refs():
        frames, fps = read_video(r["path"])
        n = len(frames)
        ctx = r["ctx"]
        ridx = ref_indices(ctx)
        n_end = min(n - 1, ctx + int(round(6.0 * fps)))
        gidx = np.linspace(ctx, n_end, 16).round().astype(int)
        ims = ([label_img(frames[i], "REFERENCE") for i in ridx]
               + [label_img(frames[i], "GENERATED") for i in gidx])
        row = dict(blind_id=r["blind_id"], vid=r["vid"], model=r["model"], scene=r["scene"])
        for key, q in PROBES:
            row[f"p_{key}"] = round(p_yes(ims, q), 4)
        rows.append(row)
        print(f"[plaus] {r['blind_id']} {r['vid']:20s} "
              f"style={row['p_style']:.3f} novel={row['p_novel']:.3f} uncanny={row['p_uncanny']:.3f}",
              flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"[plaus] wrote {OUT} ({len(rows)} videos)")


if __name__ == "__main__":
    main()
