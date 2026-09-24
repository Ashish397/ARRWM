#!/usr/bin/env python3
"""Complete the still-blank native-frame relocation review rows."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


KEY = ["scene", "model", "time_s"]


def parse_key(value: str) -> tuple[str, str, float]:
    pieces = value.split("|")
    if len(pieces) != 3:
        raise argparse.ArgumentTypeError("keys must be scene|model|time_s")
    return pieces[0], pieces[1], float(pieces[2])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument("--accept", action="append", type=parse_key, default=[])
    args = parser.parse_args()

    frame = pd.read_csv(args.adjudication, keep_default_na=False)
    required = {*KEY, "relocation_flag", "adjudication_reason"}
    missing = required - set(frame.columns)
    if missing or frame.duplicated(KEY).any():
        raise ValueError(f"missing/duplicate review keys: {sorted(missing)}")
    flags = pd.to_numeric(frame.relocation_flag, errors="coerce")
    blank = flags.isna() | frame.adjudication_reason.astype(str).str.strip().eq("")
    blank_keys = {
        (str(row.scene), str(row.model), float(row.time_s))
        for row in frame[blank].itertuples(index=False)
    }
    accepted = set(args.accept)
    unknown = accepted - blank_keys
    if unknown:
        raise ValueError(f"accepted keys are not blank candidates: {sorted(unknown)}")

    frame.loc[blank, "relocation_flag"] = 0
    frame.loc[blank, "adjudication_reason"] = (
        "No persistent replacement of physical place identity; continuous "
        "travel/occlusion or progressive corruption/collapse."
    )
    for scene, model, time_s in accepted:
        match = (
            frame.scene.astype(str).eq(scene)
            & frame.model.astype(str).eq(model)
            & pd.to_numeric(frame.time_s).eq(time_s)
        )
        frame.loc[match, "relocation_flag"] = 1
        frame.loc[match, "adjudication_reason"] = (
            "Persistent replacement of physical place identity."
        )
    completed_flags = pd.to_numeric(frame.relocation_flag, errors="coerce")
    if not completed_flags.isin((0, 1)).all():
        raise ValueError("review remains incomplete")

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
