#!/usr/bin/env python3
"""Build a broad EgoSchema pool for *visual* outdoor-traversal screening.

Ego4D scenario labels describe an entire long recording and cannot establish
that an individual 180-second EgoSchema excerpt is suitable.  This script only
creates an internal review pool.  No row is accepted until its dense temporal
contact strip has been inspected for outdoor, world-facing traversal and the
absence of hand-centric or person-doing activity.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from ego4d_replacement_pool import download_hf, reuse_downloads, write_manifest


ALLOWED_SCENARIOS = {
    "Walking on street",
    "Tourism",
    "Going to the park",
    "Going to park",
    "Hiking",
    "Car - commuting, road trip",
    "Cycling / jogging",
    "Bus",
}

POSITIVE = {
    "driving": 20,
    "drive": 15,
    "road": 14,
    "street": 14,
    "walking": 14,
    "walk": 11,
    "traffic": 11,
    "traverse": 11,
    "journey": 9,
    "route": 9,
    "path": 9,
    "scenery": 9,
    "landscape": 9,
    "city": 8,
    "urban": 8,
    "outdoor": 9,
    "outside": 8,
    "surroundings": 6,
    "pedestrian": 9,
    "hiking": 14,
    "park": 7,
    "travel": 7,
    "commute": 10,
    "vehicle": 6,
}

NEGATIVE = {
    "hand": -18,
    "hold": -10,
    "pick": -18,
    "grab": -18,
    "put": -10,
    "place": -8,
    "cook": -24,
    "eat": -24,
    "food": -20,
    "restaurant": -22,
    "phone": -18,
    "laptop": -22,
    "computer": -22,
    "table": -16,
    "shoe": -18,
    "clothes": -20,
    "clean": -18,
    "work": -12,
    "task": -12,
    "object": -14,
    "dog": -12,
    "pet": -12,
    "golf": -22,
    "game": -16,
    "shop": -18,
    "store": -14,
    "gym": -18,
    "camera": -12,
    "talk": -9,
    "conversation": -9,
    "prepare": -12,
    "interact": -10,
    "activity": -7,
    "person": -6,
    "man": -5,
    "woman": -5,
    "lady": -5,
    "room": -18,
    "house": -18,
    "indoor": -22,
}

BAD_SOURCE_SCENARIOS = {
    "Cooking",
    "Cleaning / laundry",
    "Eating",
    "Eating at a restaurant",
    "Working at desk",
    "Crafting/knitting/sewing/drawing/painting",
    "Grocery shopping indoors",
    "Play with cellphone",
    "Playing board games",
    "Golfing",
    "Daily hygiene",
}


def terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def has_term(text: str, words: set[str], term: str) -> bool:
    return term in text if " " in term else term in words


def read_ids(manifests: list[Path], field: str) -> set[str]:
    result: set[str] = set()
    for manifest in manifests:
        with manifest.open(newline="") as handle:
            result.update(row[field] for row in csv.DictReader(handle) if row.get(field))
    return result


def build_rows(
    metadata: Path,
    master_metadata: Path,
    exclude_q_manifests: list[Path],
    exclude_source_manifests: list[Path],
    per_source: int,
    limit: int,
) -> list[dict]:
    questions = json.loads((metadata / "questions.json").read_text())
    mapping = json.loads((metadata / "uid_to_ego4d.json").read_text())
    master = json.loads(master_metadata.read_text())
    videos = {row["video_uid"]: row for row in master["videos"]}
    excluded_q = read_ids(exclude_q_manifests, "q_uid")
    excluded_sources = read_ids(exclude_source_manifests, "ego4d_video_uid")

    candidates: list[dict] = []
    for question in questions:
        q_uid = question["q_uid"]
        source = mapping[q_uid]
        source_uid = source["video_uid"]
        scenarios = set(videos[source_uid].get("scenarios") or [])
        if q_uid in excluded_q or source_uid in excluded_sources:
            continue
        if not scenarios.intersection(ALLOWED_SCENARIOS):
            continue

        question_text = str(question.get("question", "")).lower()
        words = terms(question_text)
        positive = [term for term in POSITIVE if has_term(question_text, words, term)]
        negative = [term for term in NEGATIVE if has_term(question_text, words, term)]
        score = sum(POSITIVE[term] for term in positive)
        score += sum(NEGATIVE[term] for term in negative)
        score += 14 * bool(
            scenarios.intersection(
                {"Walking on street", "Tourism", "Hiking", "Going to the park", "Going to park"}
            )
        )
        score += 9 * bool(scenarios.intersection({"Car - commuting, road trip", "Cycling / jogging", "Bus"}))
        score -= 4 * len(scenarios.intersection(BAD_SOURCE_SCENARIOS))
        candidates.append(
            {
                "score": score,
                "q_uid": q_uid,
                "google_drive_id": question["google_drive_id"],
                "ego4d_video_uid": source_uid,
                "ego4d_starting_sec": source["starting_sec"],
                "ego4d_ending_sec": source["ending_sec"],
                "scenarios": ",".join(sorted(scenarios)),
                "positive_terms": ",".join(positive),
                "negative_terms": ",".join(negative),
                "question": question.get("question", ""),
                "review_status": "UNSCREENED",
                "review_reason": "",
            }
        )

    candidates.sort(key=lambda row: (-int(row["score"]), row["q_uid"]))
    selected: list[dict] = []
    source_counts: dict[str, int] = {}
    for row in candidates:
        source_uid = row["ego4d_video_uid"]
        if source_counts.get(source_uid, 0) >= per_source:
            continue
        source_counts[source_uid] = source_counts.get(source_uid, 0) + 1
        selected.append(row)
        if limit and len(selected) >= limit:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--master-metadata", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--exclude-q-manifest", action="append", type=Path, default=[])
    parser.add_argument("--exclude-source-manifest", action="append", type=Path, default=[])
    parser.add_argument("--reuse-dir", action="append", type=Path, default=[])
    parser.add_argument("--per-source", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="zero means all eligible rows")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    rows = build_rows(
        args.metadata,
        args.master_metadata,
        args.exclude_q_manifest,
        args.exclude_source_manifest,
        args.per_source,
        args.limit,
    )
    if not rows:
        raise SystemExit("no candidates matched")
    write_manifest(rows, args.manifest)
    reuse_downloads(rows, args.raw_dir, args.reuse_dir)
    if args.download:
        download_hf(rows, args.raw_dir, args.workers)
    print(f"prepared {len(rows)} visual-screening candidates")


if __name__ == "__main__":
    main()
