"""Stage 1 of the pop-in detector: cache dense per-frame object detections.

Runs RT-DETR on EVERY frame of every video at a deliberately LOW score threshold.
The low threshold matters: the temporal logic needs the weak-detection tail to tell
"object faded in gradually from far away" (real) from "object snapped into existence"
(conjured). Filtering to a confident set happens later, in the tracker.

Usage: python popin_cache.py [--dir DIR] [--glob PAT]  ->  out/popin_dets_<tag>.json
"""
import os, sys, json, glob
import numpy as np, torch, imageio.v3 as iio
from transformers import AutoModelForObjectDetection, AutoImageProcessor

DEV = "cuda"
MODEL_ID = "PekingU/rtdetr_r50vd"
LOW = 0.10          # cache everything above this; downstream picks its own thresholds
BATCH = 8

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "out"); os.makedirs(OUTD, exist_ok=True)


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main():
    vdir = arg("--dir", "/home/ashish/stationary_evaluation")
    pat = arg("--glob", "minwm_r*.mp4")
    tag = arg("--tag", pat.split("_")[0])
    paths = sorted(glob.glob(os.path.join(vdir, pat)))
    assert paths, f"no videos match {vdir}/{pat}"

    proc = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForObjectDetection.from_pretrained(MODEL_ID).eval().to(DEV)
    id2label = model.config.id2label

    @torch.no_grad()
    def detect_batch(frames):
        H, W = frames[0].shape[:2]
        inp = proc(images=list(frames), return_tensors="pt").to(DEV)
        out = model(**inp)
        res = proc.post_process_object_detection(out, target_sizes=[(H, W)] * len(frames), threshold=LOW)
        per = []
        for r in res:
            per.append([{"box": [round(float(x), 1) for x in b], "cls": id2label[int(l)], "score": round(float(s), 3)}
                        for b, l, s in zip(r["boxes"].cpu().numpy(), r["labels"].cpu().numpy(), r["scores"].cpu().numpy())])
        return per

    cache = {}
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        vid = iio.imread(p, plugin="pyav")
        dets = []
        for i in range(0, len(vid), BATCH):
            dets.extend(detect_batch(vid[i:i + BATCH]))
        cache[name] = {"n": int(len(vid)), "h": int(vid.shape[1]), "w": int(vid.shape[2]), "dets": dets}
        print(f"{name}: {len(vid)} frames, {sum(len(d) for d in dets)} dets", flush=True)

    fp = os.path.join(OUTD, f"popin_dets_{tag}.json")
    with open(fp, "w") as f:
        json.dump(cache, f)
    print("wrote", fp)


if __name__ == "__main__":
    main()
