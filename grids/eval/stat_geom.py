"""Geometric-corruption probe (deployed Qwen3-VL-8B uncanny probe) on the stationary
null-command set. 4 real REFERENCE frames (from real_rXX) + 16 generated frames over the 6s
horizon; p_uncanny = softmax P(Yes); flag when >0.5. Writes out/stat_geom.csv (resumable)."""
import os
import cv2, numpy as np, torch, pandas as pd
from PIL import Image

DIR = "/home/ashish/stationary_evaluation"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "stat_geom.csv")
MODELS = ["16node", "4node", "pca8", "pca4", "pca2", "noatok", "noadaln",
          "astra", "matrixgame", "minwm", "worldcam", "worldplay", "yume"]
CTX = {"16node": 12, "4node": 12, "pca8": 12, "pca4": 12, "pca2": 12, "noatok": 12, "noadaln": 12,
       "astra": 4, "matrixgame": 1, "minwm": 13, "worldcam": 65, "worldplay": 1, "yume": 1}
INTRO = ("The first four images labelled REFERENCE are real frames of a driving video, in temporal "
         "order - the true scene and style. The remaining images labelled GENERATED are an AI world "
         "model's continuation of that exact scene, in temporal order. The model was supposed to "
         "continue the SAME scene in the SAME visual style with plausible content.")
UNC = ("Is there a SIGNIFICANT uncanny or reality-breaking failure in the generated frames - "
       "impossible geometry, surfaces dissolving into abstract patterns, large corrupted regions - "
       "that a casual viewer would notice within one second? Ignore small local artifacts and minor "
       "blur. Answer Yes or No.")


def read_video(path):
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS) or 16; frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (640, 352)))
    cap.release(); return np.stack(frames), fps


def label_img(fr, label):
    hdr = np.full((26, fr.shape[1], 3), 32, np.uint8)
    cv2.putText(hdr, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return Image.fromarray(np.concatenate([hdr, fr], 0))


def main():
    from transformers import AutoProcessor
    from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct", dtype=torch.bfloat16, device_map="cuda").eval()
    tok = proc.tokenizer
    yid = tok.encode("Yes", add_special_tokens=False)[0]; nid = tok.encode("No", add_special_tokens=False)[0]

    @torch.no_grad()
    def p_yes(ims):
        content = [{"type": "image", "image": im} for im in ims] + [
            {"type": "text", "text": INTRO + "\n\n" + UNC + '\n\nAnswer with ONLY {"answer": "Yes"} or {"answer": "No"}.'}]
        text = proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False) + '{"answer": "'
        inp = proc(text=[text], images=ims, return_tensors="pt").to("cuda")
        lg = model(**inp).logits[0, -1]
        return float(torch.softmax(lg[torch.tensor([yid, nid], device=lg.device)], 0)[0])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = set(); rows = []
    if os.path.exists(OUT) and os.path.getsize(OUT) > 0:
        prev = pd.read_csv(OUT); done = set(zip(prev.model, prev.scene)); rows = prev.to_dict("records")
    for scene in range(32):
        rp = os.path.join(DIR, f"real_r{scene:02d}.mp4")
        if not os.path.exists(rp):
            print("no real ref", scene, flush=True); continue
        rf, _ = read_video(rp)
        ref = [label_img(rf[i], "REFERENCE") for i in (2, 5, 8, 11)]
        for m in MODELS:
            if (m, scene) in done:
                continue
            fp = os.path.join(DIR, f"{m}_r{scene:02d}.mp4")
            if not os.path.exists(fp):
                continue
            fr, fps = read_video(fp); ctx = CTX[m]
            end = min(len(fr) - 1, ctx + int(round(6.0 * fps)))
            gi = np.linspace(ctx, end, 16).round().astype(int)
            ims = ref + [label_img(fr[i], "GENERATED") for i in gi]
            u = round(p_yes(ims), 4)
            rows.append(dict(model=m, scene=scene, p_uncanny=u, flag=int(u > 0.5)))
            print(f"{m}_r{scene:02d} uncanny={u:.3f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("wrote", OUT, len(rows))


if __name__ == "__main__":
    main()
