"""Temporal novelty via FRAME-TO-FRAME probing: flag SUDDEN object appearance
(confabulation) and ignore GRADUAL entry / camera-pan reveal.

Cumulative novelty vs the original context saturates (a slow pan eventually looks
"new" too). Instead we ask, for consecutive frame pairs a fraction of a second apart,
whether a new object POPPED into the later frame:

  INTRO_T: the two images are consecutive frames of a driving video, ~0.5s apart.
  Q_T    : did a NEW distinct object suddenly appear in the second frame (popped into
           existence) rather than gradually entering from the edge or being revealed by
           the camera moving? Gradual entry / newly revealed background do NOT count.

spike_k = P(Yes) for pair k. A smooth pan -> all spikes ~0; a spawn -> a sharp spike.
  suddenness = max_k spike_k ;  n_spikes = #(spike_k > 0.5)
Flag confabulation when suddenness >= tau (default 0.5).

Usage: python temporal_novelty.py <model> <scene> [...]   Env TAU (0.5), T (12).
"""
import os, sys
import numpy as np, torch
import fleet_common as fc
from vlm_external import label_img

DEV = "cuda"
T = int(os.environ.get("T", "12"))
TAU = float(os.environ.get("TAU", "0.5"))

INTRO_T = ("The two images labelled A and B are consecutive frames from an AI-generated "
           "driving video, about half a second apart, in order (A then B).")
Q_T = ("Did a NEW distinct object - a vehicle, person, animal, or large item - SUDDENLY "
       "appear in frame B that was not present anywhere in frame A, as if it popped into "
       "existence? An object that gradually enters from the edge of the frame, grows as it "
       "approaches, or is newly revealed because the camera moved does NOT count - only a "
       "genuinely new object that materialized. Answer Yes or No.")


def build_probe():
    from transformers import AutoProcessor
    from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct", dtype=torch.bfloat16, device_map=DEV).eval()
    tok = proc.tokenizer
    yid = tok.encode("Yes", add_special_tokens=False)[0]
    nid = tok.encode("No", add_special_tokens=False)[0]

    @torch.no_grad()
    def p_yes(a_img, b_img):
        ims = [label_img(a_img, "A"), label_img(b_img, "B")]
        content = [{"type": "image", "image": im} for im in ims] + [
            {"type": "text", "text": INTRO_T + "\n\n" + Q_T +
             '\n\nAnswer with ONLY {"answer": "Yes"} or {"answer": "No"}.'}]
        text = proc.apply_chat_template([{"role": "user", "content": content}],
                                        add_generation_prompt=True, tokenize=False) + '{"answer": "'
        inp = proc(text=[text], images=ims, return_tensors="pt").to(DEV)
        lg = model(**inp).logits[0, -1]
        return float(torch.softmax(lg[torch.tensor([yid, nid], device=DEV)], 0)[0])
    return p_yes


def spikes(p_yes, model, scene):
    n, fps = fc.meta(scene, model); ctx = fc.ctx_of(model)
    ts = np.linspace(ctx, n - 1, T + 1).round().astype(int)
    fr = {i: f for i, f in zip(ts, fc.frames_at(scene, model, list(ts)))}
    return np.array([round(p_yes(fr[ts[k]], fr[ts[k + 1]]), 2) for k in range(T)]), ts


def main():
    p_yes = build_probe()
    a = sys.argv[1:]
    pairs = [(a[i], a[i + 1]) for i in range(0, len(a), 2)]
    print(f"tau={TAU} T={T}\n{'model/scene':20s} {'frame-to-frame novelty spikes':44s} {'max':>4s} {'#sp':>3s}  flag")
    for model, scene in pairs:
        sp, ts = spikes(p_yes, model, scene)
        mx = float(sp.max()); ns = int((sp > 0.5).sum())
        flag = "SUDDEN(confab)" if mx >= TAU else "gradual/clean"
        bar = " ".join(f"{v:.1f}" for v in sp)
        print(f"{model+' '+scene:20s} [{bar}] {mx:4.1f} {ns:3d}  {flag}", flush=True)


if __name__ == "__main__":
    main()
