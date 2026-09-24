#!/usr/bin/env python3
"""Reuse relocation review only when the regenerated proposal set is unchanged."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["scene", "model", "time_s"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reuse(
    dense_path: Path,
    previous_adjudication_path: Path,
    output_path: Path,
    previous_events_path: Path | None = None,
) -> dict[str, object]:
    dense = pd.read_csv(dense_path, keep_default_na=False)
    previous = pd.read_csv(previous_adjudication_path, keep_default_na=False)
    for frame, label in ((dense, "dense proposals"), (previous, "adjudication")):
        missing = set(KEY) - set(frame.columns)
        if missing or frame.duplicated(KEY).any():
            raise ValueError(f"{label} has missing/duplicate keys: {sorted(missing)}")
    abrupt = dense[pd.to_numeric(dense.abrupt_cut, errors="raise").eq(1)]
    abrupt_keys = set(map(tuple, abrupt[KEY].astype(str).to_numpy()))
    reviewed_keys = set(map(tuple, previous[KEY].astype(str).to_numpy()))
    if abrupt_keys != reviewed_keys:
        raise ValueError("abrupt proposal keys differ; fresh human review is required")
    flags = pd.to_numeric(previous.relocation_flag, errors="coerce")
    if flags.isna().any() or not flags.isin((0, 1)).all():
        raise ValueError("previous relocation decisions are incomplete")
    if previous.adjudication_reason.astype(str).str.strip().eq("").any():
        raise ValueError("previous relocation reasons are incomplete")

    if previous_events_path is not None:
        events = pd.read_csv(previous_events_path, keep_default_na=False)
        if events.duplicated(KEY).any():
            raise ValueError("previous finalized events contain duplicate keys")
        left = dense.sort_values(KEY).reset_index(drop=True)
        right = events.sort_values(KEY).reset_index(drop=True)
        if len(left) != len(right) or not left[KEY].astype(str).equals(right[KEY].astype(str)):
            raise ValueError("dense event keys differ from the previously reviewed scan")
        shared = [column for column in left.columns if column in right.columns and column not in KEY]
        for column in shared:
            a = pd.to_numeric(left[column], errors="coerce")
            b = pd.to_numeric(right[column], errors="coerce")
            if a.notna().all() and b.notna().all():
                if not np.allclose(a.to_numpy(float), b.to_numpy(float), rtol=0, atol=1e-12):
                    raise ValueError(f"dense evidence column changed: {column}")
            elif not left[column].astype(str).equals(right[column].astype(str)):
                raise ValueError(f"dense evidence column changed: {column}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    previous.to_csv(temporary, index=False)
    temporary.replace(output_path)
    return {
        "schema_version": 1,
        "status": "pass",
        "proposal_rows": len(dense),
        "abrupt_candidates": len(abrupt),
        "accepted_relocations": int(flags.eq(1).sum()),
        "dense": str(dense_path.resolve()),
        "dense_sha256": sha256(dense_path),
        "previous_adjudication": str(previous_adjudication_path.resolve()),
        "previous_adjudication_sha256": sha256(previous_adjudication_path),
        "output": str(output_path.resolve()),
        "output_sha256": sha256(output_path),
        "reuse_basis": "proposal keys and shared dense evidence are unchanged",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--previous-adjudication", type=Path, required=True)
    parser.add_argument("--previous-events", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = reuse(
        args.dense.resolve(), args.previous_adjudication.resolve(),
        args.output.resolve(),
        args.previous_events.resolve() if args.previous_events else None,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_name(f".{args.report.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
