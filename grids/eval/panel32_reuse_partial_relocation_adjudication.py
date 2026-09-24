#!/usr/bin/env python3
"""Reuse relocation decisions outside explicitly changed panel contexts.

The current native-frame gate defines the review queue.  Decisions are copied
only for exact proposal keys whose context was not regenerated and whose dense
evidence columns agree with the prior finalized scan.  Candidates from changed
contexts remain blank for fresh review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["scene", "model", "time_s"]
DECISION = ["relocation_flag", "adjudication_reason"]
EVIDENCE = [
    "sample_index", "frame_index", "cross_inliers", "pre_coherence",
    "post_coherence", "pre_keypoints", "post_keypoints", "abrupt_cut",
    "seam_mad", "seam_mad_ratio", "seam_hist", "seam_hist_ratio",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--previous-adjudication", type=Path, required=True)
    parser.add_argument("--previous-events", type=Path, required=True)
    parser.add_argument("--changed-context", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    dense = pd.read_csv(args.dense, keep_default_na=False)
    current = dense[pd.to_numeric(dense.abrupt_cut, errors="raise").eq(1)].copy()
    previous = pd.read_csv(args.previous_adjudication, keep_default_na=False)
    events = pd.read_csv(args.previous_events, keep_default_na=False)
    for frame, label in ((current, "current"), (previous, "previous"), (events, "events")):
        missing = set(KEY) - set(frame.columns)
        if missing or frame.duplicated(KEY).any():
            raise ValueError(f"{label} has missing/duplicate keys: {sorted(missing)}")
    if set(DECISION) - set(previous.columns):
        raise ValueError("previous adjudication is missing decision columns")
    if not pd.to_numeric(previous.relocation_flag, errors="coerce").isin((0, 1)).all():
        raise ValueError("previous relocation decisions are incomplete")

    old_decisions = previous.set_index(KEY)
    old_events = events.set_index(KEY)
    changed = tuple(f"{context}_" for context in args.changed_context)
    output_rows = []
    reused = 0
    fresh = 0
    for row in current.itertuples(index=False):
        key = (str(row.scene), str(row.model), row.time_s)
        item = {column: getattr(row, column) for column in KEY}
        item.update({"relocation_flag": "", "adjudication_reason": ""})
        is_changed = bool(changed) and str(row.scene).startswith(changed)
        if not is_changed and key in old_decisions.index and key in old_events.index:
            prior_event = old_events.loc[key]
            evidence_matches = True
            for column in EVIDENCE:
                if column not in current.columns or column not in events.columns:
                    raise ValueError(f"missing evidence column: {column}")
                left = float(getattr(row, column))
                right = float(prior_event[column])
                if not np.isclose(left, right, rtol=0, atol=1e-12):
                    evidence_matches = False
                    break
            if evidence_matches:
                decision = old_decisions.loc[key]
                item["relocation_flag"] = str(int(decision.relocation_flag))
                item["adjudication_reason"] = str(decision.adjudication_reason)
                reused += 1
        if item["relocation_flag"] == "":
            fresh += 1
        output_rows.append(item)

    output = pd.DataFrame(output_rows, columns=KEY + DECISION)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    output.to_csv(temporary, index=False)
    temporary.replace(args.output)
    report = {
        "schema_version": 1,
        "status": "pass",
        "current_abrupt_candidates": len(current),
        "reused_decisions": reused,
        "remaining_for_review": fresh,
        "changed_contexts": args.changed_context,
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_name(f".{args.report.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
