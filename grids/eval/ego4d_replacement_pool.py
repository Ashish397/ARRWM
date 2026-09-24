#!/usr/bin/env python3
"""Build a small, reviewable Ego4D replacement pool from EgoSchema clips.

This intentionally works with individually addressable 180-second EgoSchema
clips rather than downloading an Ego4D dataset archive.  The automatic stage
only ranks likely outdoor/world-facing clips.  Contact sheets are generated
for visual acceptance before any clip can enter the evaluation manifest.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from remotezip import RemoteZip


POSITIVE = {
    "sidewalk": 16,
    "street": 14,
    "road": 15,
    "traffic": 14,
    "driving": 15,
    "drive": 10,
    "crossing": 12,
    "pedestrian": 10,
    "forest": 14,
    "trail": 14,
    "hiking": 14,
    "outdoor": 13,
    "outside": 10,
    "scenery": 8,
    "landscape": 8,
    "city": 8,
    "park": 8,
    "neighborhood": 10,
    "campus": 8,
    "journey": 6,
    "traverses": 8,
    "exploring": 7,
    "surroundings": 4,
    "vehicles": 6,
    "vehicle": 5,
    "walking": 6,
    "walk": 4,
}

# These penalties reduce bandwidth spent on obvious hand-task footage.  They
# do not constitute acceptance: every retained candidate is visually checked.
NEGATIVE = {
    "cook": -30,
    "kitchen": -30,
    "ingredient": -25,
    "chop": -25,
    "cleaning": -20,
    "repair": -20,
    "assemble": -20,
    "install": -20,
    "lawnmower": -25,
    "grocery": -20,
    "shopping": -18,
    "shop": -14,
    "phone": -16,
    "laptop": -20,
    "computer": -20,
    "desk": -20,
    "office": -20,
    "dish": -18,
    "sewing": -25,
    "workshop": -20,
    "painting": -18,
    "paint": -14,
    "cards": -18,
    "game": -14,
    "tools": -16,
    "tool": -14,
    "bucket": -15,
    "mortar": -20,
    "harvest": -20,
    "garden": -12,
    "golf": -15,
    "tennis": -12,
    "basketball": -12,
    "soccer": -10,
    "baby": -15,
    "table": -14,
    "counter": -14,
    "project": -12,
    "washing": -18,
    "wash": -14,
    "fix": -16,
    "hand": -12,
    "holding": -10,
    "hold": -8,
    "grab": -12,
    "pick up": -12,
}


def contains(text: str, phrase: str, words: set[str]) -> bool:
    return phrase in text if " " in phrase else phrase in words


def manifest_q_uids(manifests: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for manifest in manifests:
        with manifest.open(newline="") as handle:
            for row in csv.DictReader(handle):
                q_uid = row.get("q_uid")
                if q_uid:
                    excluded.add(q_uid)
    return excluded


def manifest_source_videos(manifests: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for manifest in manifests:
        with manifest.open(newline="") as handle:
            for row in csv.DictReader(handle):
                video_uid = row.get("ego4d_video_uid")
                if video_uid:
                    excluded.add(video_uid)
    return excluded


def rank(
    metadata: Path,
    limit: int,
    exclude: set[str] | None = None,
    exclude_source_videos: set[str] | None = None,
    unique_source_video: bool = False,
) -> list[dict]:
    exclude = exclude or set()
    excluded_sources = set(exclude_source_videos or set())
    questions = json.loads((metadata / "questions.json").read_text())
    mapping = json.loads((metadata / "uid_to_ego4d.json").read_text())
    rows = []
    for question in questions:
        text = " ".join(
            [str(question.get("question", ""))]
            + [str(question.get(f"option {i}", "")) for i in range(5)]
        ).lower()
        words = set(re.findall(r"[a-z]+", text))
        positive = [p for p in POSITIVE if contains(text, p, words)]
        if not positive:
            continue
        negative = [p for p in NEGATIVE if contains(text, p, words)]
        score = sum(POSITIVE[p] for p in positive) + sum(NEGATIVE[p] for p in negative)
        source = mapping[question["q_uid"]]
        rows.append(
            {
                "score": score,
                "q_uid": question["q_uid"],
                "google_drive_id": question["google_drive_id"],
                "ego4d_video_uid": source["video_uid"],
                "ego4d_starting_sec": source["starting_sec"],
                "ego4d_ending_sec": source["ending_sec"],
                "positive_terms": ",".join(positive),
                "negative_terms": ",".join(negative),
                "question": question.get("question", ""),
            }
        )
    rows.sort(key=lambda row: (-row["score"], row["q_uid"]))
    unique = []
    seen = set()
    seen_sources = set(excluded_sources)
    for row in rows:
        if row["q_uid"] in exclude:
            continue
        if row["ego4d_video_uid"] in excluded_sources:
            continue
        if unique_source_video and row["ego4d_video_uid"] in seen_sources:
            continue
        key = (row["ego4d_video_uid"], row["ego4d_starting_sec"])
        if key in seen:
            continue
        seen.add(key)
        seen_sources.add(row["ego4d_video_uid"])
        unique.append(row)
        if len(unique) == limit:
            break
    return unique


def reuse_downloads(rows: list[dict], raw_dir: Path, reuse_dirs: list[Path]) -> None:
    """Reflink already-downloaded clips into a newly numbered review pool."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows, 1):
        output = raw_dir / f"{index:03d}_{row['q_uid']}.mp4"
        if output.exists() and output.stat().st_size > 1_000_000:
            continue
        matches = [
            candidate
            for directory in reuse_dirs
            for candidate in directory.glob(f"*_{row['q_uid']}.mp4")
            if candidate.stat().st_size > 1_000_000
        ]
        if not matches:
            continue
        subprocess.run(
            ["cp", "--reflink=auto", "--preserve=timestamps", str(matches[0]), str(output)],
            check=True,
        )


def write_manifest(rows: list[dict], manifest: Path) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def download(rows: list[dict], raw_dir: Path, workers: int) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for index, row in enumerate(rows, 1):
        output = raw_dir / f"{index:03d}_{row['q_uid']}.mp4"
        if output.exists() and output.stat().st_size > 1_000_000:
            continue
        jobs.append((row["google_drive_id"], output))

    active: list[subprocess.Popen] = []
    for drive_id, output in jobs:
        while len(active) >= workers:
            active[0].wait()
            active = [proc for proc in active if proc.poll() is None]
        active.append(
            subprocess.Popen(
                [str(Path.home() / ".local/bin/gdown"), drive_id, "-q", "-O", str(output), "--retries", "5"]
            )
        )
    failures = [proc.wait() for proc in active]
    if any(failures):
        raise SystemExit(f"download failures: {failures}")


HF_CHUNKS = [
    f"https://huggingface.co/datasets/lmms-eval/egoschema/resolve/main/"
    f"videos_chunked_{index:02d}.zip?download=true"
    for index in range(1, 6)
]


def download_hf(rows: list[dict], raw_dir: Path, workers: int) -> None:
    """Range-read only requested clips from the five remote ZIPs."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    wanted = {
        row["q_uid"]: raw_dir / f"{index:03d}_{row['q_uid']}.mp4"
        for index, row in enumerate(rows, 1)
        if not (
            (raw_dir / f"{index:03d}_{row['q_uid']}.mp4").exists()
            and (raw_dir / f"{index:03d}_{row['q_uid']}.mp4").stat().st_size > 1_000_000
        )
    }
    if not wanted:
        return

    groups: dict[str, list[tuple[str, Path]]] = {}
    for url in HF_CHUNKS:
        with RemoteZip(url) as archive:
            names = {Path(info.filename).stem: info.filename for info in archive.infolist()}
        matches = [(names[q_uid], output) for q_uid, output in wanted.items() if q_uid in names]
        if matches:
            groups[url] = matches
            for member, _ in matches:
                wanted.pop(Path(member).stem)
    if wanted:
        raise SystemExit(f"clips absent from Hugging Face mirror: {sorted(wanted)}")

    def fetch_group(url: str, members: list[tuple[str, Path]]) -> None:
        with RemoteZip(url) as archive:
            for member, output in members:
                temporary = output.with_suffix(output.suffix + ".part")
                with archive.open(member) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
                os.replace(temporary, output)

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(groups))) as pool:
        futures = [pool.submit(fetch_group, url, members) for url, members in groups.items()]
        for future in futures:
            future.result()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--exclude-manifest", action="append", type=Path, default=[])
    parser.add_argument("--exclude-source-manifest", action="append", type=Path, default=[])
    parser.add_argument("--unique-source-video", action="store_true")
    parser.add_argument("--reuse-dir", action="append", type=Path, default=[])
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--download-hf", action="store_true")
    args = parser.parse_args()

    rows = rank(
        args.metadata,
        args.limit,
        exclude=manifest_q_uids(args.exclude_manifest),
        exclude_source_videos=manifest_source_videos(args.exclude_source_manifest),
        unique_source_video=args.unique_source_video,
    )
    if len(rows) != args.limit:
        raise SystemExit(f"requested {args.limit} candidates but ranked {len(rows)}")
    write_manifest(rows, args.manifest)
    if args.raw_dir is not None and args.reuse_dir:
        reuse_downloads(rows, args.raw_dir, args.reuse_dir)
    if args.download:
        if args.raw_dir is None:
            raise SystemExit("--raw-dir is required with --download")
        download(rows, args.raw_dir, args.workers)
    if args.download_hf:
        if args.raw_dir is None:
            raise SystemExit("--raw-dir is required with --download-hf")
        download_hf(rows, args.raw_dir, args.workers)


if __name__ == "__main__":
    main()
