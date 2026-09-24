#!/usr/bin/env python3
"""Reuse human conjuration decisions only for an identical candidate queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


DECISION_COLUMNS = ("adjudication", "reason")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reuse(previous_path: Path, current_path: Path) -> dict[str, object]:
    previous = pd.read_csv(previous_path, keep_default_na=False, dtype=str)
    current = pd.read_csv(current_path, keep_default_na=False, dtype=str)
    if list(previous.columns) != list(current.columns):
        raise ValueError("previous/current adjudication schemas differ")
    if "candidate_id" not in current or any(column not in current for column in DECISION_COLUMNS):
        raise ValueError("adjudication CSV is missing required columns")
    if previous.candidate_id.duplicated().any() or current.candidate_id.duplicated().any():
        raise ValueError("candidate IDs must be unique")
    if set(previous.candidate_id) != set(current.candidate_id):
        raise ValueError("candidate ID sets differ; fresh human review is required")
    immutable = [
        column for column in current.columns
        if column not in ("candidate_id", *DECISION_COLUMNS)
    ]
    old = previous.set_index("candidate_id").sort_index()
    new = current.set_index("candidate_id").sort_index()
    if not old[immutable].equals(new[immutable]):
        raise ValueError("immutable candidate evidence differs; fresh human review is required")
    if not old.adjudication.isin(("0", "1")).all() or old.reason.str.strip().eq("").any():
        raise ValueError("previous adjudication is incomplete")
    current = current.copy()
    decisions = old[list(DECISION_COLUMNS)]
    current["adjudication"] = current.candidate_id.map(decisions.adjudication)
    current["reason"] = current.candidate_id.map(decisions.reason)
    temporary = current_path.with_name(f".{current_path.name}.tmp.{os.getpid()}")
    current.to_csv(temporary, index=False)
    temporary.replace(current_path)
    return {
        "schema_version": 1,
        "status": "pass",
        "candidate_count": len(current),
        "accepted_candidates": int(current.adjudication.eq("1").sum()),
        "previous": str(previous_path.resolve()),
        "previous_sha256": sha256(previous_path),
        "current": str(current_path.resolve()),
        "current_sha256": sha256(current_path),
        "reuse_basis": "candidate IDs and every immutable evidence column are identical",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = reuse(args.previous.resolve(), args.current.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_name(f".{args.report.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
