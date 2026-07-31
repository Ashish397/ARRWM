"""Matched melt-VLM benchmark on the 84 labelled ablation videos (LOCAL).

Reproduces, on this machine, the cluster testbench melt component so three VLMs can
be compared under ONE identical protocol (prompt, crops, frame sampling, aggregation
AND scoring) -- the reviewer asked for a matched candidate table, so any AUC gap is
the model, not tuning.

Protocol (from utils/vlm_bench.py, the deployed Qwen2.5-VL melt judge):
  prompt      : MELT_Q, one shared yes/no question
  crops       : top-60% of frame, left & right halves  (upper_crops)
  sampling    : GEN_IDX generated frames (default 6, deep in the rollout)
  score/frame : P(Yes) = softmax over {Yes-tokens, No-tokens} logits, pooled by
                logsumexp. For Qwen this is IDENTICAL to the deployed adapter's
                sigmoid(logsumexp(yes)-logsumexp(no)); applied unchanged to all three
                models so the scoring is matched, not just the prompt.
  aggregation : mean of the top-4 crop-scores per video

Boundary note (reviewer's frame-11/12 question): GEN_IDX starts at 24, far inside the
generated region under both the legacy and corrected (ref=11, gen-start=12) boundary,
so no melt score or classification changes with the boundary correction. The boundary
correction only bites the PAL4VST/depth geometry component, which is a separate script.

Env:
  MELT_MODEL   qwen25vl7b | internvl3_8b | cosmos_reason1_7b   (default qwen25vl7b)
  MELT_GEN_IDX comma list of frame indices (default 24,40,56,72,88,104)
  MELT_OUT     output csv (default out/melt_vlm_<model>.csv)
Writes per-video p_yes: columns scene,model,tier,note,p_yes.
"""
import os, sys, gc
import numpy as np, pandas as pd, imageio, torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda"
TILE_DIRS = [os.path.join(HERE, "tiles_new"), os.path.join(HERE, "tiles")]
LABELS = os.path.join(HERE, "human_tiers.csv")


def vid_path(scene, model):
    for d in TILE_DIRS:
        p = os.path.join(d, f"{scene}__{model}.mp4")
        if os.path.exists(p):
            return p
    return None

MODEL_ID = {
    "qwen25vl7b":      "Qwen/Qwen2.5-VL-7B-Instruct",
    "internvl3_8b":    "OpenGVLab/InternVL3-8B-HF",
    "cosmos_reason1_7b": "nvidia/Cosmos-Reason1-7B",
}
MELT_MODEL = os.environ.get("MELT_MODEL", "qwen25vl7b")
GEN_IDX = [int(x) for x in os.environ.get("MELT_GEN_IDX", "24,40,56,72,88,104").split(",")]

MELT_Q = ("Are any buildings, walls or structures in this image melted, warped, smeared "
          "or geometrically impossible, like a corrupted AI-generated image? "
          "Answer only Yes or No.")


def upper_crops(img):
    """Top-60% of the frame, left and right halves -- where mangling shows first."""
    H, W = img.shape[:2]
    h = int(H * 0.6); w2 = W // 2
    return [img[:h, :w2], img[:h, w2:]]


def frames_of(path, idxs):
    r = imageio.get_reader(path)
    out = []
    for i in idxs:
        try:
            out.append(np.asarray(r.get_data(i)))
        except Exception:
            break
    r.close()
    return out


def load_judge(name):
    """One unified adapter: AutoModelForImageTextToText + AutoProcessor, last-token
    Yes/No logit -> P(Yes). Same code path for all three models."""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    mid = MODEL_ID[name]
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

    @torch.no_grad()
    def p_yes_crop(crop):
        im = Image.fromarray(crop)
        msgs = [{"role": "user", "content": [{"type": "image", "image": im},
                                             {"type": "text", "text": MELT_Q}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[im], return_tensors="pt").to(DEV)
        lg = model(**inputs).logits[0, -1]
        y = torch.logsumexp(lg[yes], 0)
        n = torch.logsumexp(lg[no], 0)
        return float(torch.softmax(torch.stack([y, n]), 0)[0].item())

    def score(frames):
        vals = [p_yes_crop(c) for f in frames for c in upper_crops(f)]
        return float(np.mean(sorted(vals)[-4:]))   # top-4 mean P(Yes)

    return model, score


def main():
    lab = pd.read_csv(LABELS)
    out_csv = os.environ.get("MELT_OUT", os.path.join(HERE, "out", f"melt_vlm_{MELT_MODEL}.csv"))
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    print(f"[melt] model={MELT_MODEL} ({MODEL_ID[MELT_MODEL]})  frames={GEN_IDX}", flush=True)
    model, scorer = load_judge(MELT_MODEL)

    rows = []
    for _, r in lab.iterrows():
        vid = vid_path(r.scene, r.model)
        if vid is None:
            print(f"[melt] MISSING {r.scene}__{r.model}.mp4", flush=True); continue
        try:
            p = scorer(frames_of(vid, GEN_IDX))
        except Exception as e:
            print(f"[melt] {r.scene} {r.model} failed: {str(e)[:100]}", flush=True); p = np.nan
        rows.append(dict(scene=r.scene, model=r.model, tier=r.tier, note=r.note, p_yes=p))
        print(f"[melt] {r.scene:8s} {r.model:8s} tier={r.tier:<4} p_yes={p:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[melt] wrote {out_csv}  ({df.p_yes.notna().sum()}/{len(df)} scored)", flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
