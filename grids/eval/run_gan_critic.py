"""GAN/fake-image detector critics as artifact scorers.

Runs pretrained AI-image detectors per frame on start/end windows:
  <name>_fake_start, <name>_fake_end, <name>_fake_drift (end - start; more fake-looking over time)
Writes results_gan.csv.
"""
import glob, os
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

DEV = "cuda"
HERE = os.path.dirname(os.path.abspath(__file__))
TILE_DIR = os.path.join(HERE, "tiles")
OUT_CSV = os.path.join(HERE, "results_gan.csv")
W = 16

DETECTORS = {
    "sdxldet": "Organika/sdxl-detector",
    "ummdet": "umm-maybe/AI-image-detector",
}


def read_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


@torch.no_grad()
def fake_prob(frames, proc, model, fake_idx):
    ims = [Image.fromarray(f) for f in frames[::2]]
    probs = []
    for i in range(0, len(ims), 8):
        inputs = proc(images=ims[i : i + 8], return_tensors="pt").to(DEV)
        logits = model(**inputs).logits
        probs.append(torch.softmax(logits, -1)[:, fake_idx].cpu())
    return float(torch.cat(probs).mean())


def main():
    files = sorted(glob.glob(os.path.join(TILE_DIR, "*.mp4")))
    vids = {}
    for fp in files:
        name = os.path.basename(fp)[: -len(".mp4")]
        grid, variant = name.split("__")
        vids[(grid, variant)] = read_frames(fp)

    rows = []
    for key, repo in DETECTORS.items():
        try:
            proc = AutoImageProcessor.from_pretrained(repo)
            model = AutoModelForImageClassification.from_pretrained(repo).to(DEV).eval()
        except Exception as e:
            print(f"SKIP {key}: {e}", flush=True)
            continue
        labels = {v.lower(): k for k, v in model.config.id2label.items()}
        fake_idx = next((i for l, i in labels.items() if "artificial" in l or "ai" in l or "fake" in l), 0)
        print(f"{key}: labels={model.config.id2label} fake_idx={fake_idx}", flush=True)
        for (grid, variant), frames in vids.items():
            ps = fake_prob(frames[:W], proc, model, fake_idx)
            pe = fake_prob(frames[-W:], proc, model, fake_idx)
            for k2, v2 in [(f"{key}_fake_start", ps), (f"{key}_fake_end", pe), (f"{key}_fake_drift", pe - ps)]:
                rows.append({"grid": grid, "variant": variant, "metric": k2, "value": v2})
            print(f"{key} {grid} {variant}: start={ps:.3f} end={pe:.3f}", flush=True)
        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    if os.path.exists(OUT_CSV):
        old = pd.read_csv(OUT_CSV)
        old = old[~old.metric.isin(df.metric.unique())]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV} ({len(df)} rows)")


if __name__ == "__main__":
    main()
