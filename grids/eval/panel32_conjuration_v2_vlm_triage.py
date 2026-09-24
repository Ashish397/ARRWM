#!/usr/bin/env python3
"""Independent VLM triage for the human conjuration-v2 evidence review.

This is deliberately not a metric producer and it never writes an
``adjudication`` column.  It assigns a forced Yes/No probability to every
already-locked audit strip so likely positives and ambiguous cases can be
reviewed first.  Final labels remain the output of
``panel32_conjuration_v2_review.py finalize`` and require a reason for every
candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import pandas as pd
from PIL import Image


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
PROMPT = """The seven images are consecutive temporal evidence from one generated
video.  They are ordered from before to after a detector's proposed object
birth.  The red box in the BIRTH image identifies the candidate.

Does a new physical object abruptly materialize inside the scene at BIRTH,
despite being absent from every earlier image, and remain present afterward?

Answer Yes only for an abrupt, persistent interior appearance.  Answer No if
the object was already visible, enters naturally from any image boundary
(including a hand or body entering from the bottom), is revealed by camera
motion or an occluder, is only a detector/class change, is a reflection or
shadow, or is visual corruption without a stable new object.  If ambiguous,
answer No.

Answer with ONLY {"answer": "Yes"} or {"answer": "No"}."""


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def evidence_frames(path: Path) -> list[Image.Image]:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    if width % 7:
        raise ValueError(f"evidence strip does not contain seven equal cells: {path} {image.size}")
    cell = width // 7
    if cell <= 0 or height <= 0:
        raise ValueError(f"empty evidence strip: {path}")
    return [image.crop((index * cell, 0, (index + 1) * cell, height))
            for index in range(7)]


class Judge:
    def __init__(self) -> None:
        import torch
        from transformers import AutoProcessor
        from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
        ).eval()
        tokenizer = self.processor.tokenizer
        self.yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.no_id = tokenizer.encode("No", add_special_tokens=False)[0]

    def score(self, images: list[Image.Image]) -> float:
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": PROMPT})
        prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True, tokenize=False,
        ) + '{"answer": "'
        inputs = self.processor(
            text=[prompt], images=images, return_tensors="pt",
        ).to("cuda")
        with self.torch.no_grad():
            logits = self.model(**inputs).logits[0, -1]
            indices = self.torch.tensor(
                [self.yes_id, self.no_id], device=logits.device,
            )
            return float(self.torch.softmax(logits[indices], 0)[0])


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")

    candidates = pd.read_csv(args.candidates, keep_default_na=False)
    required = {"candidate_id", "evidence_path"}
    if missing := required - set(candidates.columns):
        raise ValueError(f"candidate CSV missing columns: {sorted(missing)}")
    if candidates.candidate_id.astype(str).duplicated().any():
        raise ValueError("candidate IDs are not unique")
    candidates = candidates.iloc[
        [index for index in range(len(candidates))
         if index % args.shard_count == args.shard_index]
    ]
    expected_ids = candidates.candidate_id.astype(str).tolist()

    columns = [
        "candidate_id", "p_conjuration_vlm", "vlm_flag", "model_id",
        "prompt_sha256", "evidence_path", "evidence_sha256",
    ]
    rows: list[dict[str, object]] = []
    if args.output.is_file() and args.output.stat().st_size:
        previous = pd.read_csv(args.output, keep_default_na=False)
        if list(previous.columns) != columns:
            raise ValueError(f"incompatible prior output: {args.output}")
        prior_ids = previous.candidate_id.astype(str).tolist()
        if prior_ids != expected_ids[:len(prior_ids)]:
            raise ValueError(f"prior output is not a prefix of this shard: {args.output}")
        rows = previous.to_dict("records")

    judge = Judge()
    prompt_sha = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
    for row in candidates.iloc[len(rows):].itertuples(index=False):
        evidence = Path(str(row.evidence_path)).resolve()
        if not evidence.is_file() or evidence.stat().st_size == 0:
            raise FileNotFoundError(evidence)
        probability = judge.score(evidence_frames(evidence))
        rows.append({
            "candidate_id": str(row.candidate_id),
            "p_conjuration_vlm": probability,
            "vlm_flag": int(probability > 0.5),
            "model_id": MODEL_ID,
            "prompt_sha256": prompt_sha,
            "evidence_path": str(evidence),
            "evidence_sha256": digest(evidence),
        })
        atomic_csv(pd.DataFrame(rows, columns=columns), args.output)
    if [str(item["candidate_id"]) for item in rows] != expected_ids:
        raise RuntimeError("completed shard does not contain its exact candidate set")
    print(f"CONJURATION_V2_VLM_TRIAGE_COMPLETE shard={args.shard_index} rows={len(rows)}")


if __name__ == "__main__":
    main()
