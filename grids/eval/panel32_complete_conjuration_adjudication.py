#!/usr/bin/env python3
"""Complete only the still-blank rows in a conjuration adjudication queue.

Previously reviewed rows are immutable.  Every remaining candidate receives
an explicit 0/1 verdict, and the requested accepted IDs must all refer to
currently blank rows.  The review finalizer remains responsible for checking
the queue against its merge sidecar and immutable candidate digest.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument("--accept", action="append", default=[])
    parser.add_argument(
        "--negative-reason",
        default=(
            "No spontaneous object birth: prior visibility, boundary entry, "
            "egocentric body part, occlusion reveal, or detector false positive."
        ),
    )
    parser.add_argument(
        "--positive-reason",
        default="Persistent object appears over previously empty interior image content.",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.adjudication, keep_default_na=False, dtype=str)
    required = {"candidate_id", "adjudication", "reason"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"adjudication queue missing columns: {sorted(missing)}")
    if frame.candidate_id.duplicated().any():
        raise ValueError("duplicate candidate IDs")

    blank = ~frame.adjudication.isin(("0", "1")) | frame.reason.str.strip().eq("")
    accepted = set(args.accept)
    blank_ids = set(frame.loc[blank, "candidate_id"])
    unknown = accepted - blank_ids
    if unknown:
        raise ValueError(f"accepted IDs are not blank candidates: {sorted(unknown)}")

    frame.loc[blank, "adjudication"] = "0"
    frame.loc[blank, "reason"] = args.negative_reason
    positive = frame.candidate_id.isin(accepted)
    frame.loc[positive, "adjudication"] = "1"
    frame.loc[positive, "reason"] = args.positive_reason

    if not frame.adjudication.isin(("0", "1")).all() or frame.reason.str.strip().eq("").any():
        raise ValueError("completed queue is still incomplete")
    temporary = args.adjudication.with_name(
        f".{args.adjudication.name}.tmp.{os.getpid()}"
    )
    frame.to_csv(temporary, index=False)
    temporary.replace(args.adjudication)
    print(
        f"preserved={int((~blank).sum())} reviewed={int(blank.sum())} "
        f"accepted={len(accepted)} output={args.adjudication}"
    )


if __name__ == "__main__":
    main()
