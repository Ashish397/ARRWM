#!/usr/bin/env python3
"""Copy a complete, non-overlapping Ego4D candidate round for author review."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--unique-source", action="store_true")
    args = parser.parse_args()

    with args.candidate_manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    chosen: list[tuple[dict, Path]] = []
    seen_sources: set[str] = set()
    for original_index, row in enumerate(rows, 1):
        source = args.candidate_dir / f"{original_index:03d}_{row['q_uid']}.mp4"
        if not source.exists() or source.stat().st_size <= 1_000_000:
            continue
        if args.unique_source and row["ego4d_video_uid"] in seen_sources:
            continue
        seen_sources.add(row["ego4d_video_uid"])
        chosen.append((row, source))
        if len(chosen) == args.count:
            break
    if len(chosen) != args.count:
        raise SystemExit(f"only {len(chosen)} complete eligible clips; need {args.count}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for review_index, (row, source) in enumerate(chosen, 1):
        output = args.output_dir / f"{review_index:03d}_{row['q_uid']}.mp4"
        shutil.copy2(source, output)
        manifest_rows.append({"review_file": output.name, **row})

    manifest = args.output_dir / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    readme = args.output_dir / "README.md"
    readme.write_text(
        f"# Ego4D review round\n\n"
        f"This folder contains {args.count} individually downloaded 180-second "
        "EgoSchema/Ego4D excerpts for manual review. It is a candidate pool, not "
        "an accepted evaluation set. Reject any indoor, hand-centric, object-task, "
        "person-doing, or unsuitable-viewpoint clip. No dataset archive was downloaded.\n"
    )
    print(f"materialized {len(chosen)} clips in {args.output_dir}")


if __name__ == "__main__":
    main()
