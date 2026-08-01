"""Run the geometric-corruption (uncanny) probe with InternVL3-8B and Cosmos on the
specific fleet rollouts shown in the fleet_2/fleet_3 geometry reels, so the reels can
show all three VLMs' scores side by side. Qwen3-VL scores come from
results_external_vlm.csv. Env PLAUS_MODEL: internvl3_8b | cosmos_reason1_7b.
Writes out/reel_uncanny_<model>.csv (scene,model,p_uncanny)."""
import os
import numpy as np, torch, pandas as pd
import fleet_common as fc
from vlm_external import INTRO, PROBES, read_video, label_img
from blind_plausibility import ref_indices

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = {"internvl3_8b": "OpenGVLab/InternVL3-8B-HF", "cosmos_reason1_7b": "nvidia/Cosmos-Reason1-7B"}
VLM = os.environ.get("PLAUS_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
OUT = os.path.join(HERE, "out", f"reel_uncanny_{VLM}.csv")
UNC_Q = dict(PROBES)["uncanny"]
DEV = "cuda"

# the exact rollouts in the two reels (fleet_2 internal + fleet_3 external)
REELS = [("r12_BL", "ours_4node"), ("r03_BL", "ours_pca4"), ("r10_L", "ours_pca2"),
         ("r03_L", "ours_noadaln"), ("r15_B", "ours_4node"), ("r16_FR", "ours_4node"),
         ("r03_R", "worldplay"), ("r20_R", "worldplay"), ("r10_L", "matrixgame"),
         ("r29_BL", "astra"), ("r00_B", "yume"), ("r27_BL", "yume")]


def path_of(scene, model):
    return fc._path(scene, model)


def main():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    mid = MODEL_ID[VLM]
    proc = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(mid, dtype=torch.bfloat16, device_map=DEV, trust_remote_code=True).eval()
    tok = proc.tokenizer

    def ids(strs):
        out = [tok.encode(t, add_special_tokens=False)[0] for t in strs if tok.encode(t, add_special_tokens=False)]
        return sorted(set(out))
    yes = torch.tensor(ids(["Yes", " Yes", "yes"]), device=DEV); no = torch.tensor(ids(["No", " No", "no"]), device=DEV)
    pkw = {"crop_to_patches": False} if VLM == "internvl3_8b" else {}

    @torch.no_grad()
    def p_yes(ims):
        content = [{"type": "image", "image": im} for im in ims] + [
            {"type": "text", "text": INTRO + "\n\n" + UNC_Q + '\n\nAnswer with ONLY {"answer": "Yes"} or {"answer": "No"}.'}]
        text = proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False) + '{"answer": "'
        inp = proc(text=[text], images=ims, return_tensors="pt", **pkw).to(DEV)
        lg = model(**inp).logits[0, -1]
        y = torch.logsumexp(lg[yes], 0); n = torch.logsumexp(lg[no], 0)
        return float(torch.softmax(torch.stack([y, n]), 0)[0])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    for scene, model_ in REELS:
        frames, fps = read_video(path_of(scene, model_)); n = len(frames); ctx = fc.ctx_of(model_)
        n_end = min(n - 1, ctx + int(round(6.0 * fps)))          # 6s cap
        gidx = np.linspace(ctx, n_end, 16).round().astype(int)
        ims = [label_img(frames[i], "REFERENCE") for i in ref_indices(ctx)] + [label_img(frames[i], "GENERATED") for i in gidx]
        u = round(p_yes(ims), 3)
        rows.append(dict(scene=scene, model=model_, p_uncanny=u)); print(f"[{VLM}] {model_}_{scene} unc={u}", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False); print("wrote", OUT)


if __name__ == "__main__":
    main()
