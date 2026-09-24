#!/usr/bin/env python3
"""Render dense visual-review sheets for a small Ego4D candidate pool."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def sampled_row(video: Path, samples: int, width: int, height: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for fraction in np.linspace(0.05, 0.95, samples):
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, round((frame_count - 1) * fraction)))
        ok, frame = capture.read()
        if not ok:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    capture.release()
    return np.concatenate(frames, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--thumb-width", type=int, default=240)
    parser.add_argument("--thumb-height", type=int, default=135)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.manifest.open(newline="") as handle:
        metadata = list(csv.DictReader(handle))
    videos = sorted(args.raw_dir.glob("*.mp4"))
    if not args.allow_partial and len(videos) != len(metadata):
        raise SystemExit(f"expected {len(metadata)} videos, found {len(videos)}")

    rows = []
    for video in videos:
        index = int(video.name.split("_", 1)[0])
        item = metadata[index - 1]
        strip = sampled_row(video, args.samples, args.thumb_width, args.thumb_height)
        label = np.full((36, strip.shape[1], 3), 245, dtype=np.uint8)
        text = f"{index:03d} score={item['score']} {item['q_uid']}"
        cv2.putText(label, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 1, cv2.LINE_AA)
        sheet = np.concatenate([label, strip], axis=0)
        cv2.imwrite(str(args.output_dir / f"{index:03d}.jpg"), sheet)
        rows.append((index, sheet))

    rows.sort(key=lambda item: item[0])
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset : offset + args.batch_size]
        combined = np.concatenate([item[1] for item in batch], axis=0)
        first = batch[0][0]
        last = batch[-1][0]
        cv2.imwrite(str(args.output_dir / f"batch_{first:03d}_{last:03d}.jpg"), combined)


if __name__ == "__main__":
    main()
