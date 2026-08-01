"""Multi-VLM validation of the NEW frame-to-frame confabulation probe on blind100.

Same temporal metric as the deployed pop-in detector (does a new object POP into
frame B vs A), run
across Qwen3-VL-8B / Cosmos-Reason1-7B / InternVL3-8B so the Qwen3-VL choice for the
confab axis is justified (the earlier 3-VLM bakeoff validated the OLD cumulative p_novel,
not this one). Per rollout: novel_sudden = max frame-to-frame spike.

Env PLAUS_MODEL: qwen3vl8b | cosmos_reason1_7b | internvl3_8b
Writes out/blind_temporal_<model>.csv.
"""
import os
import numpy as np, torch, pandas as pd
import blind100_common as bc
from vlm_external import label_img

DEV = "cuda"
T = int(os.environ.get("T", "12"))
MODEL_ID = {"qwen3vl8b": "Qwen/Qwen3-VL-8B-Instruct",
            "cosmos_reason1_7b": "nvidia/Cosmos-Reason1-7B",
            "internvl3_8b": "OpenGVLab/InternVL3-8B-HF"}
PLAUS_MODEL = os.environ.get("PLAUS_MODEL", "qwen3vl8b")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", f"blind_temporal_{PLAUS_MODEL}.csv")

INTRO_T = ("The two images labelled A and B are consecutive frames from an AI-generated "
           "driving video, about half a second apart, in order (A then B).")
Q_T = ("Did a NEW distinct object - a vehicle, person, animal, or large item - SUDDENLY "
       "appear in frame B that was not present anywhere in frame A, as if it popped into "
       "existence? An object that gradually enters from the edge of the frame, grows as it "
       "approaches, or is newly revealed because the camera moved does NOT count - only a "
       "genuinely new object that materialized. Answer Yes or No.")


def main():
    import imageio
    from transformers import AutoModelForImageTextToText, AutoProcessor
    mid = MODEL_ID[PLAUS_MODEL]
    proc = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        mid, dtype=torch.bfloat16, device_map=DEV, trust_remote_code=True).eval()
    tok = proc.tokenizer

    def ids(strs):
        out = [tok.encode(t, add_special_tokens=False)[0] for t in strs if tok.encode(t, add_special_tokens=False)]
        return sorted(set(out))
    yes = torch.tensor(ids(["Yes", " Yes", "yes"]), device=DEV)
    no = torch.tensor(ids(["No", " No", "no"]), device=DEV)
    pkw = {"crop_to_patches": False} if PLAUS_MODEL == "internvl3_8b" else {}

    @torch.no_grad()
    def p_yes(a_img, b_img):
        ims = [label_img(a_img, "A"), label_img(b_img, "B")]
        content = [{"type": "image", "image": im} for im in ims] + [
            {"type": "text", "text": INTRO_T + "\n\n" + Q_T +
             '\n\nAnswer with ONLY {"answer": "Yes"} or {"answer": "No"}.'}]
        text = proc.apply_chat_template([{"role": "user", "content": content}],
                                        add_generation_prompt=True, tokenize=False) + '{"answer": "'
        inp = proc(text=[text], images=ims, return_tensors="pt", **pkw).to(DEV)
        lg = model(**inp).logits[0, -1]
        y = torch.logsumexp(lg[yes], 0); n = torch.logsumexp(lg[no], 0)
        return float(torch.softmax(torch.stack([y, n]), 0)[0])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print(f"[btnov] model={PLAUS_MODEL}", flush=True)
    rows = []
    for r in bc.refs():
        rd = imageio.get_reader(r["path"]); fr = [np.asarray(f) for f in rd]; rd.close()
        n = len(fr); ctx = r["ctx"]
        ts = np.linspace(ctx, n - 1, T + 1).round().astype(int)
        sp = np.array([p_yes(fr[ts[j]], fr[ts[j + 1]]) for j in range(T)])
        rows.append(dict(blind_id=r["blind_id"], vid=r["vid"], model=r["model"],
                         novel_sudden=round(float(sp.max()), 3), n_spikes=int((sp > 0.5).sum())))
        print(f"[btnov] {r['blind_id']} {r['vid']:20s} sudden={sp.max():.2f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"[btnov] wrote {OUT}")


if __name__ == "__main__":
    main()
