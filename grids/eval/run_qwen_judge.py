"""Qwen2.5-VL-7B rubric judge on tile videos (reference-free).

Samples 12 frames per tile, asks for JSON scores on artifact dimensions
matching the human eval vocabulary. Writes results_qwen.csv.
"""
import glob, json, os, re
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

HERE = os.path.dirname(os.path.abspath(__file__))
TILE_DIR = os.path.join(HERE, "tiles")
OUT_CSV = os.path.join(HERE, "results_qwen.csv")
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
N_FRAMES = 12

RUBRIC = """You are judging the output of an autoregressive driving world model (AI-generated first-person driving video). The frames below are uniformly sampled in temporal order from a ~7 second rollout. The first frames come from real context, later frames are generated.

Score the video on each dimension from 0 (terrible) to 10 (perfect), judging only what you can see:
- "melt_warp": absence of melting/warping artifacts (straight lines staying straight, objects keeping rigid shapes; 10 = no melt/warp, 0 = severe melting or curved space warping)
- "blur_haze": absence of blur, haze, or washed-out veils in the generated part (10 = crisp throughout)
- "black_regions": absence of unnatural black/dark regions or missing content (10 = none)
- "structure": plausibility of generated scene structure - buildings, trees, roads look real and complete, nothing hallucinated, mangled, or uncanny (10 = fully plausible)
- "temporal": temporal stability - no shimmer, flicker, or objects popping/morphing between frames (10 = stable)
- "style_consistent": the visual style (colors, tone, sharpness) of the LAST frames matches the FIRST frames (10 = same style, 0 = completely different look)
- "overall": overall human-eye quality of the generated video (10 = looks like real dashcam footage)

Answer with ONLY a JSON object, e.g.:
{"melt_warp": 7, "blur_haze": 5, "black_regions": 10, "structure": 6, "temporal": 8, "style_consistent": 9, "overall": 6}"""


def read_frames(path, k=N_FRAMES):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    idx = np.linspace(0, len(frames) - 1, k).round().astype(int)
    return [Image.fromarray(frames[i]) for i in idx]


def main():
    processor = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    rows = []
    for fp in sorted(glob.glob(os.path.join(TILE_DIR, "*.mp4"))):
        name = os.path.basename(fp)[: -len(".mp4")]
        grid, variant = name.split("__")
        images = read_frames(fp)
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": RUBRIC}]
        messages = [{"role": "user", "content": content}]
        chat_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        # keep vision token budget sane: ~448px longest side per frame
        small = [im.resize((448, 240)) for im in images]
        inputs = processor(text=chat_text, images=small, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        text = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            print(f"{name}: PARSE FAIL: {text[:200]}", flush=True)
            continue
        try:
            scores = json.loads(m.group(0))
        except json.JSONDecodeError:
            print(f"{name}: JSON FAIL: {text[:200]}", flush=True)
            continue
        for k, v in scores.items():
            rows.append({"grid": grid, "variant": variant, "metric": f"qwen_{k}", "value": float(v)})
        print(f"{name}: {scores}", flush=True)

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
