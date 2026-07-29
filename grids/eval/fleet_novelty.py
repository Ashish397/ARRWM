"""Confabulation / sudden-novelty axis on the full 256x13 fleet.

Frame-to-frame temporal novelty (validated in temporal_novelty.py): for consecutive
generated frames ~0.5s apart, probe whether a NEW object POPPED into the later frame
(not gradual entry / camera-pan reveal). A smooth pan -> ~0 everywhere; a spawned object
-> a sharp isolated spike. Per rollout we record the spike series and:
  novel_sudden = max spike        (higher = clearer sudden appearance)
  n_spikes     = # pairs > 0.5
  confab_flag  = novel_sudden >= 0.5
Writes out/fleet_novelty.csv (resumable).
"""
import os
import numpy as np, torch, pandas as pd
import fleet_common as fc
from vlm_external import label_img

DEV = "cuda"
T = int(os.environ.get("T", "12"))
TAU = float(os.environ.get("TAU", "0.5"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "fleet_novelty.csv")

INTRO_T = ("The two images labelled A and B are consecutive frames from an AI-generated "
           "driving video, about half a second apart, in order (A then B).")
Q_T = ("Did a NEW distinct object - a vehicle, person, animal, or large item - SUDDENLY "
       "appear in frame B that was not present anywhere in frame A, as if it popped into "
       "existence? An object that gradually enters from the edge of the frame, grows as it "
       "approaches, or is newly revealed because the camera moved does NOT count - only a "
       "genuinely new object that materialized. Answer Yes or No.")


def main():
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

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    idx = fc.fleet_index()
    done = set(); rows = []
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT); done = set(zip(prev.scene, prev.model)); rows = prev.to_dict("records")
    for k, (scene, model_) in enumerate(idx):
        if (scene, model_) in done:
            continue
        try:
            n, fps = fc.meta(scene, model_); ctx = fc.ctx_of(model_)
            end = min(n - 1, ctx + int(round(6.0 * fps)))   # 6s horizon cap (was n-1)
            ts = np.linspace(ctx, end, T + 1).round().astype(int)
            fr = {i: f for i, f in zip(ts, fc.frames_at(scene, model_, list(ts)))}
            sp = np.array([p_yes(fr[ts[j]], fr[ts[j + 1]]) for j in range(T)])
            rows.append(dict(scene=scene, model=model_, novel_sudden=round(float(sp.max()), 3),
                             n_spikes=int((sp > 0.5).sum()), confab_flag=int(sp.max() >= TAU)))
        except Exception as e:
            print(f"[fnov] {scene} {model_} FAIL {str(e)[:70]}", flush=True); continue
        if (k + 1) % 50 == 0:
            pd.DataFrame(rows).to_csv(OUT, index=False); print(f"[fnov] {k+1}/{len(idx)}", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"[fnov] wrote {OUT} ({len(rows)})")


if __name__ == "__main__":
    main()
