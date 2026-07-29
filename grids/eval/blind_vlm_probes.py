"""Multi-VLM plausibility/geometry probe bakeoff on blind100.

Runs the EXACT vlm_external pointwise protocol (verbatim INTRO + PROBES, 4 REFERENCE +
16 GENERATED composited frames, forced-prefix logit P(Yes)) but across three VLMs so
geometry (p_uncanny) and plausibility (p_novel) get the same bakeoff melt already has.

Env PLAUS_MODEL: qwen3vl8b | cosmos_reason1_7b | internvl3_8b
Writes out/blind_vlmprobes_<model>.csv (p_style, p_novel, p_uncanny per video).
"""
import os
import numpy as np, torch, pandas as pd
import blind100_common as bc
from vlm_external import INTRO, PROBES, read_video, label_img
from blind_plausibility import ref_indices

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = {"qwen3vl8b": "Qwen/Qwen3-VL-8B-Instruct",
            "cosmos_reason1_7b": "nvidia/Cosmos-Reason1-7B",
            "internvl3_8b": "OpenGVLab/InternVL3-8B-HF"}
PLAUS_MODEL = os.environ.get("PLAUS_MODEL", "qwen3vl8b")
OUT = os.path.join(HERE, "out", f"blind_vlmprobes_{PLAUS_MODEL}.csv")
DEV = "cuda"


def main():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    mid = MODEL_ID[PLAUS_MODEL]
    proc = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        mid, dtype=torch.bfloat16, device_map=DEV, trust_remote_code=True).eval()
    tok = proc.tokenizer

    def ids_for(strs):
        out = []
        for t in strs:
            e = tok.encode(t, add_special_tokens=False)
            if e:
                out.append(e[0])
        return sorted(set(out))
    yes = torch.tensor(ids_for(["Yes", " Yes", "yes"]), device=DEV)
    no = torch.tensor(ids_for(["No", " No", "no"]), device=DEV)
    # InternVL dynamic-tiles each of 20 images into ~7 patches (36k tokens >> 8192 ctx);
    # crop_to_patches=False -> 1 tile/image (~5.3k tokens) so the 20-image protocol fits.
    proc_kw = {"crop_to_patches": False} if PLAUS_MODEL == "internvl3_8b" else {}

    @torch.no_grad()
    def p_yes(ims, question):
        content = [{"type": "image", "image": im} for im in ims] + [
            {"type": "text", "text": INTRO + "\n\n" + question +
             '\n\nAnswer with ONLY {"answer": "Yes"} or {"answer": "No"}.'}]
        text = proc.apply_chat_template([{"role": "user", "content": content}],
                                        add_generation_prompt=True, tokenize=False)
        text += '{"answer": "'
        inputs = proc(text=[text], images=ims, return_tensors="pt", **proc_kw).to(DEV)
        lg = model(**inputs).logits[0, -1]
        y = torch.logsumexp(lg[yes], 0); n = torch.logsumexp(lg[no], 0)
        return float(torch.softmax(torch.stack([y, n]), 0)[0].item())

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print(f"[vlm] model={PLAUS_MODEL} ({mid})", flush=True)
    rows = []
    refs = bc.refs()
    mx = int(os.environ.get("BLIND_MAX", "0"))
    if mx:
        refs = refs[:mx]
    for r in refs:
        frames, fps = read_video(r["path"]); n = len(frames); ctx = r["ctx"]
        ridx = ref_indices(ctx)
        n_end = min(n - 1, ctx + int(round(6.0 * fps)))
        gidx = np.linspace(ctx, n_end, 16).round().astype(int)
        ims = ([label_img(frames[i], "REFERENCE") for i in ridx]
               + [label_img(frames[i], "GENERATED") for i in gidx])
        row = dict(blind_id=r["blind_id"], vid=r["vid"], model=r["model"], scene=r["scene"])
        for key, q in PROBES:
            try:
                row[f"p_{key}"] = round(p_yes(ims, q), 4)
            except Exception as e:
                print(f"[vlm] {r['blind_id']} {key} failed: {str(e)[:70]}", flush=True)
                row[f"p_{key}"] = np.nan
        rows.append(row)
        print(f"[vlm] {r['blind_id']} {r['vid']:20s} "
              f"novel={row['p_novel']} uncanny={row['p_uncanny']} style={row['p_style']}", flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"[vlm] wrote {OUT}")


if __name__ == "__main__":
    main()
