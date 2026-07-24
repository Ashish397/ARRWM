"""VideoScore-v1.1 (TIGER-Lab) on tile videos: 5 regression dims
(visual quality, temporal consistency, dynamic degree, T2V alignment, factual consistency).
Writes results_videoscore.csv.
"""
import glob, os
import cv2
import numpy as np
import torch
import pandas as pd
from PIL import Image
from transformers import AutoProcessor
from mantis.models.idefics2 import Idefics2ForSequenceClassification

HERE = os.path.dirname(os.path.abspath(__file__))
TILE_DIR = os.path.join(HERE, "tiles")
OUT_CSV = os.path.join(HERE, "results_videoscore.csv")

MAX_FRAMES = 48

PROMPT_TMPL = """
Suppose you are an expert in judging and evaluating the quality of AI-generated videos,
please watch the following frames of a given video and see the text prompt for generating the video,
then give scores from 5 different dimensions:
(1) visual quality: the quality of the video in terms of clearness, resolution, brightness, and color
(2) temporal consistency, both the consistency of objects or humans and the smoothness of motion or movements
(3) dynamic degree, the degree of dynamic changes
(4) text-to-video alignment, the alignment between the text prompt and the video content
(5) factual consistency, the consistency of the video content with the common-sense and factual knowledge

For each dimension, output a float number from 1.0 to 4.0,
the higher the number is, the better the video performs in that sub-score,
the lowest 1.0 means Bad, the highest 4.0 means Perfect/Real (the video is like a real video)
Here is an output example:
visual quality: 3.2
temporal consistency: 2.7
dynamic degree: 4.0
text-to-video alignment: 2.3
factual consistency: 1.8

For this video, the text prompt is "{text_prompt}",
all the frames of video are as follows:
"""

DIMS = ["vs_visual_quality", "vs_temporal_consistency", "vs_dynamic_degree", "vs_t2v_alignment", "vs_factual_consistency"]


def read_frames(path, k=MAX_FRAMES):
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
    model_name = "TIGER-Lab/VideoScore-v1.1"
    processor = AutoProcessor.from_pretrained(model_name)
    model = Idefics2ForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.bfloat16).eval().to("cuda")

    text_prompt = "first person driving footage on a road"
    rows = []
    for fp in sorted(glob.glob(os.path.join(TILE_DIR, "*.mp4"))):
        name = os.path.basename(fp)[: -len(".mp4")]
        grid, variant = name.split("__")
        images = read_frames(fp)
        frames_str = "<image>" * len(images)
        prompt = PROMPT_TMPL.format(text_prompt=text_prompt) + frames_str
        inputs = processor(text=prompt, images=images, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        scores = [round(out.logits[0, i].item(), 3) for i in range(5)]
        for d, s in zip(DIMS, scores):
            rows.append({"grid": grid, "variant": variant, "metric": d, "value": s})
        print(f"{name}: {dict(zip(DIMS, scores))}", flush=True)

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
