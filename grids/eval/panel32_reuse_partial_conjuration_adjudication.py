#!/usr/bin/env python3
"""Reuse reviewed conjuration decisions for unchanged candidate evidence.

Unlike the all-or-nothing queue reuse helper, this tool is intended for a
locked panel revision.  It copies a decision only when the candidate ID,
video hash, detector evidence fields, and evidence-strip bytes are identical.
New or changed candidates remain blank and therefore still fail closed in the
normal finalizer until they are reviewed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


DECISION_COLUMNS = ("adjudication", "reason")
PATH_COLUMNS = {"evidence_path", "record_path"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    previous = pd.read_csv(args.previous, keep_default_na=False, dtype=str)
    current = pd.read_csv(args.current, keep_default_na=False, dtype=str)
    required = {"candidate_id", "video_sha256", *DECISION_COLUMNS, *PATH_COLUMNS}
    for frame, label in ((previous, "previous"), (current, "current")):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} queue missing columns: {sorted(missing)}")
        if frame.candidate_id.duplicated().any():
            raise ValueError(f"{label} queue contains duplicate candidate IDs")
    if not previous.adjudication.isin(("0", "1")).all():
        raise ValueError("previous adjudication is incomplete")
    if previous.reason.str.strip().eq("").any():
        raise ValueError("previous reasons are incomplete")

    shared_columns = [
        column for column in current.columns
        if column in previous.columns
        and column not in {"candidate_id", *DECISION_COLUMNS, *PATH_COLUMNS}
    ]
    old = previous.set_index("candidate_id", verify_integrity=True)
    reused = 0
    changed = 0
    absent = 0
    evidence_cache: dict[Path, str] = {}
    for index, row in current.iterrows():
        candidate = str(row.candidate_id)
        if candidate not in old.index:
            absent += 1
            continue
        prior = old.loc[candidate]
        if any(str(row[column]) != str(prior[column]) for column in shared_columns):
            changed += 1
            continue
        current_evidence = Path(str(row.evidence_path)).resolve()
        prior_evidence = Path(str(prior.evidence_path)).resolve()
        if not current_evidence.is_file() or not prior_evidence.is_file():
            raise FileNotFoundError(current_evidence if not current_evidence.is_file() else prior_evidence)
        for path in (current_evidence, prior_evidence):
            if path not in evidence_cache:
                evidence_cache[path] = sha256(path)
        if evidence_cache[current_evidence] != evidence_cache[prior_evidence]:
            changed += 1
            continue
        current.at[index, "adjudication"] = str(prior.adjudication)
        current.at[index, "reason"] = str(prior.reason)
        reused += 1

    temporary = args.current.with_name(f".{args.current.name}.tmp.{os.getpid()}")
    current.to_csv(temporary, index=False)
    temporary.replace(args.current)
    undecided = ~current.adjudication.isin(("0", "1")) | current.reason.str.strip().eq("")
    report = {
        "schema_version": 1,
        "status": "pass",
        "previous_candidates": len(previous),
        "current_candidates": len(current),
        "reused_decisions": reused,
        "changed_matching_ids": changed,
        "new_candidate_ids": absent,
        "remaining_for_review": int(undecided.sum()),
        "previous": str(args.previous.resolve()),
        "current": str(args.current.resolve()),
        "current_sha256": sha256(args.current),
        "reuse_basis": (
            "candidate ID, video hash, detector fields, and evidence bytes are identical"
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_name(f".{args.report.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
