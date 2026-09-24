#!/usr/bin/env python3
"""Reuse native-frame relocation gates outside regenerated contexts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["scene", "model", "time_s"]
BASE_COLUMNS = [
    "sample_index", "frame_index", "cross_inliers", "pre_coherence",
    "post_coherence", "pre_keypoints", "post_keypoints",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def prepare(args: argparse.Namespace) -> None:
    events = pd.read_csv(args.events, keep_default_na=False)
    prior = pd.read_csv(args.previous_dense, keep_default_na=False)
    prior = prior[prior.case.eq("real")].copy()
    for frame, label in ((events, "events"), (prior, "previous dense")):
        missing = set(KEY + BASE_COLUMNS) - set(frame.columns)
        if missing or frame.duplicated(KEY).any():
            raise ValueError(f"{label} has missing/duplicate keys: {sorted(missing)}")
    old = prior.set_index(KEY)
    changed = tuple(f"{context}_" for context in args.changed_context)
    reuse_keys = []
    pending_rows = []
    for row in events.itertuples(index=False):
        key = (str(row.scene), str(row.model), row.time_s)
        reusable = not (changed and str(row.scene).startswith(changed)) and key in old.index
        if reusable:
            old_row = old.loc[key]
            for column in BASE_COLUMNS:
                left = float(getattr(row, column))
                right = float(old_row[column])
                if not np.isclose(left, right, rtol=0, atol=1e-12):
                    reusable = False
                    break
        if reusable:
            reuse_keys.append(key)
        else:
            pending_rows.append(row._asdict())
    reused = old.loc[reuse_keys].reset_index() if reuse_keys else prior.iloc[:0].copy()
    pending = pd.DataFrame(pending_rows, columns=events.columns)
    atomic_csv(reused, args.reused_dense)
    atomic_csv(pending, args.pending_events)
    report = {
        "schema_version": 1,
        "status": "pass",
        "event_rows": len(events),
        "reused_rows": len(reused),
        "pending_rows": len(pending),
        "changed_contexts": args.changed_context,
        "events_sha256": sha256(args.events),
        "previous_dense_sha256": sha256(args.previous_dense),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_name(f".{args.report.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.report)
    print(json.dumps(report, indent=2, sort_keys=True))


def merge(args: argparse.Namespace) -> None:
    events = pd.read_csv(args.events, keep_default_na=False)
    reused = pd.read_csv(args.reused_dense, keep_default_na=False)
    fresh = pd.read_csv(args.fresh_dense, keep_default_na=False)
    fresh = fresh[fresh.case.eq("real")].copy()
    output = pd.concat([reused, fresh], ignore_index=True)
    if output.duplicated(KEY).any():
        raise ValueError("merged dense rows contain duplicate keys")
    expected = set(map(tuple, events[KEY].itertuples(index=False, name=None)))
    actual = set(map(tuple, output[KEY].itertuples(index=False, name=None)))
    if actual != expected:
        raise ValueError(
            f"merged dense key mismatch: missing={len(expected - actual)} "
            f"extra={len(actual - expected)}"
        )
    output = output.sort_values(["model", "scene", "time_s", "case"])
    atomic_csv(output, args.output)
    print(f"rows={len(output)} reused={len(reused)} fresh={len(fresh)} output={args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--events", type=Path, required=True)
    prepare_parser.add_argument("--previous-dense", type=Path, required=True)
    prepare_parser.add_argument("--changed-context", action="append", default=[])
    prepare_parser.add_argument("--reused-dense", type=Path, required=True)
    prepare_parser.add_argument("--pending-events", type=Path, required=True)
    prepare_parser.add_argument("--report", type=Path, required=True)
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--events", type=Path, required=True)
    merge_parser.add_argument("--reused-dense", type=Path, required=True)
    merge_parser.add_argument("--fresh-dense", type=Path, required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
