#!/usr/bin/env python3
"""Render compact, score-ranked contact pages for conjuration adjudication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def render_pages(frame: pd.DataFrame, output: Path, *, per_page: int = 12) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for page, offset in enumerate(range(0, len(frame), per_page), 1):
        strips = []
        for rank, row in enumerate(
                frame.iloc[offset:offset + per_page].to_dict("records"), offset + 1):
            image = cv2.imread(str(row["evidence_path"]))
            if image is None:
                raise FileNotFoundError(row["evidence_path"])
            target_width = 1600
            target_height = max(1, round(image.shape[0] * target_width / image.shape[1]))
            image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
            header = np.full((38, target_width, 3), 20, np.uint8)
            text = (
                f"rank={rank}/{len(frame)} id={row['candidate_id']} "
                f"p={float(row['p_conjuration_vlm']):.4f} "
                f"legacy={int(row['legacy_positive'])} {row['model']}/{row['scene']} "
                f"class={row['class']} birth={float(row['birth_s']):.3f}s"
            )
            cv2.putText(header, text[:210], (6, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.54, (255, 255, 255), 1, cv2.LINE_AA)
            strips.append(np.concatenate([header, image], axis=0))
        destination = output / f"contact_{page:04d}.jpg"
        if not cv2.imwrite(str(destination), np.concatenate(strips, axis=0),
                           [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise RuntimeError(f"failed to write {destination}")
    metadata = {
        "rows": len(frame), "pages": (len(frame) + per_page - 1) // per_page,
        "candidate_ids": frame.candidate_id.astype(str).tolist(),
    }
    (output / "index.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranked", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--include-legacy", action="store_true")
    parser.add_argument("--model")
    args = parser.parse_args()
    frame = pd.read_csv(args.ranked, keep_default_na=False)
    if args.model:
        frame = frame[frame.model.astype(str).eq(args.model)]
    else:
        selected = frame.p_conjuration_vlm.astype(float).ge(args.threshold)
        if args.include_legacy:
            selected |= frame.legacy_positive.astype(int).eq(1)
        frame = frame[selected]
    frame = frame.sort_values(
        ["p_conjuration_vlm", "legacy_positive", "peak", "candidate_id"],
        ascending=[False, False, False, True], kind="stable",
    ).reset_index(drop=True)
    render_pages(frame, args.output)
    print({"rows": len(frame), "output": str(args.output)})


if __name__ == "__main__":
    main()
