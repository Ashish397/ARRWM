"""Qwen gate for ORB-proposed long-horizon relocation boundaries.

The pilot mode scores each real proposal and an obvious synthetic scene cut at
the same timestamp.  The full mode scores only real proposals.  Qwen is used
as a boundary classifier, not as a free-form evaluator; the reported value is
the forced Yes/No probability at the first output token.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
from transformers import AutoProcessor
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration


QUESTION = """The images are ordered samples from one generated driving video.
The last BEFORE image and first AFTER image are the native frames immediately
adjacent to the proposed boundary. Does that exact boundary replace the
physical environment with a different, unrelated place or world?

Answer Yes only for a persistent scene replacement: the street, surrounding
buildings, or overall place identity changes discontinuously and cannot be
reached by continuous camera motion. Answer No for an ordinary turn, forward
travel, a close object temporarily occluding the camera, motion blur, lighting
change, entering or leaving darkness, recovery from corruption, progressive
geometric damage, or featureless corruption. A large appearance change is not
enough if the adjacent native frames show a continuous transition. When the
evidence is ambiguous, answer No. Those may be failures on other axes, but
they are not relocation. Answer Yes or No."""


def resolve(path: str, local_prefix: str, remote_prefix: str) -> str:
    if Path(path).exists():
        return path
    candidate = remote_prefix + path[len(local_prefix):]
    if Path(candidate).exists():
        return candidate
    raise FileNotFoundError(path)


def read_times(path: str, context: int, fps: float,
               times: list[float]) -> list[Image.Image]:
    cap = cv2.VideoCapture(path)
    images = []
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for time_s in times:
            index = min(count - 1, max(0, context + int(round(time_s * fps))))
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError((path, index))
            images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        cap.release()
    return images


def label(image: Image.Image, text: str) -> Image.Image:
    canvas = image.copy().resize((640, 352))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 185, 26), fill="white")
    draw.text((6, 5), text, fill="black")
    return canvas


def bank_orb_scores(before: list[Image.Image], after: list[Image.Image]) -> dict[str, float]:
    orb = cv2.ORB_create(3000)

    def desc(image: Image.Image):
        array = cv2.cvtColor(np.asarray(image.resize((640, 352))), cv2.COLOR_RGB2GRAY)
        points, values = orb.detectAndCompute(array, None)
        return np.float32([point.pt for point in points]), values

    def match(left, right):
        points_l, desc_l = left
        points_r, desc_r = right
        if desc_l is None or desc_r is None or len(points_l) < 8 or len(points_r) < 8:
            return 0
        matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(desc_l, desc_r)
        if len(matches) < 8:
            return 0
        src = np.float32([points_l[item.queryIdx] for item in matches])
        dst = np.float32([points_r[item.trainIdx] for item in matches])
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        return int(mask.sum()) if mask is not None else 0

    pre = [desc(image) for image in before]
    post = [desc(image) for image in after]
    cross = max(match(left, right) for left in pre for right in post)
    pre_coherence = float(np.median([match(pre[i], pre[i + 1]) for i in range(3)]))
    post_coherence = float(np.median([match(post[i], post[i + 1]) for i in range(3)]))
    pre_keypoints = float(np.median([len(item[0]) for item in pre]))
    post_keypoints = float(np.median([len(item[0]) for item in post]))
    return {
        "bank_cross_inliers": cross,
        "bank_pre_coherence": pre_coherence,
        "bank_post_coherence": post_coherence,
        "bank_pre_keypoints": pre_keypoints,
        "bank_post_keypoints": post_keypoints,
        "orb_proposal": int(cross < 20 and pre_coherence >= 35 and
                            post_coherence >= 35 and pre_keypoints >= 100 and
                            post_keypoints >= 100),
    }


class Judge:
    def __init__(self) -> None:
        self.proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct", dtype=torch.bfloat16,
            device_map="cuda").eval()
        tokenizer = self.proc.tokenizer
        self.yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.no_id = tokenizer.encode("No", add_special_tokens=False)[0]

    @torch.no_grad()
    def score(self, before: list[Image.Image], after: list[Image.Image]) -> float:
        images = ([label(image, f"BEFORE {i + 1}") for i, image in enumerate(before)] +
                  [label(image, f"AFTER {i + 1}") for i, image in enumerate(after)])
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": QUESTION +
                        '\nAnswer with ONLY {"answer": "Yes"} or {"answer": "No"}.'})
        prompt = self.proc.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True, tokenize=False) + '{"answer": "'
        inputs = self.proc(text=[prompt], images=images,
                           return_tensors="pt").to("cuda")
        logits = self.model(**inputs).logits[0, -1]
        ids = torch.tensor([self.yes_id, self.no_id], device=logits.device)
        return float(torch.softmax(logits[ids], 0)[0])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--events", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pilot-synthetic", action="store_true")
    p.add_argument("--local-prefix", default="/home/ashish/ARRWM")
    p.add_argument("--remote-prefix", default="")
    a = p.parse_args()
    events = pd.read_csv(a.events)
    manifest = pd.read_csv(a.manifest)
    lookup = manifest.set_index(["scene", "model"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    prior = pd.read_csv(a.out) if a.out.exists() and a.out.stat().st_size else pd.DataFrame()
    completed = set(zip(prior.get("scene", []), prior.get("model", []),
                        prior.get("time_s", []), prior.get("case", [])))
    judge = Judge()
    rows = prior.to_dict("records")
    for event in events.itertuples():
        meta = lookup.loc[(event.scene, event.model)]
        path = resolve(str(meta.path), a.local_prefix, a.remote_prefix)
        source_fps = float(meta.fps)
        offsets_before = [-1.0, -0.5, -0.25, -2.0 / source_fps, -1.0 / source_fps]
        before = read_times(path, int(meta.context_frames), source_fps,
                            [event.time_s + value for value in offsets_before])
        cases = [("real", path, meta, event.time_s)]
        if a.pilot_synthetic:
            alternatives = manifest[(manifest.model.eq(event.model)) &
                                    (~manifest.scene.eq(event.scene))].sort_values("scene")
            synthetic = alternatives.iloc[0]
            synthetic_path = resolve(str(synthetic.path), a.local_prefix, a.remote_prefix)
            # Use a clear early continuation from the other scene.  Matching
            # the late timestamp can turn a deliberately different-place
            # control into two equally corrupted, place-less banks.
            cases.append(("synthetic_cut", synthetic_path, synthetic, 6.0))
        for case, after_path, after_meta, after_time in cases:
            key = (event.scene, event.model, event.time_s, case)
            if key in completed:
                continue
            after_fps = float(after_meta.fps)
            offsets_after = [0.0, 1.0 / after_fps, 2.0 / after_fps,
                             0.25, 0.5, 1.0]
            after = read_times(after_path, int(after_meta.context_frames),
                               after_fps, [after_time + value for value in offsets_after])
            orb_scores = bank_orb_scores(before, after)
            probability = judge.score(before, after)
            rows.append({
                "scene": event.scene, "model": event.model,
                "time_s": event.time_s, "case": case,
                "after_scene": (event.scene if case == "real" else synthetic.scene),
                "p_relocation": probability,
                "relocation_flag": int(probability > 0.5),
                "cross_inliers": event.cross_inliers,
                **orb_scores,
            })
            pd.DataFrame(rows).to_csv(a.out, index=False)
            print(event.scene, event.model, event.time_s, case,
                  f"p={probability:.4f}", flush=True)


if __name__ == "__main__":
    main()
