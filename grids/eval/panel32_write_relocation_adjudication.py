#!/usr/bin/env python3
"""Write the reviewed temporal-relocation decisions for the balanced panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.dense)
    reviewed = frame[(frame["case"] == "real") & frame["abrupt_cut"].eq(1)].copy()
    keys = ["scene", "model", "time_s"]
    if len(reviewed) != 36 or reviewed.duplicated(keys).any():
        raise ValueError(f"unexpected relocation review queue: {len(reviewed)}")
    # All 36 native-frame survivors were inspected in temporal order. They are
    # continuous travel/turns/occlusions or progressive corruption/collapse;
    # none persistently replaces physical place identity.
    reviewed["relocation_flag"] = 0
    reviewed["adjudication_reason"] = (
        "No persistent replacement of physical place identity; continuous "
        "travel/occlusion or progressive corruption/collapse."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reviewed[keys + ["relocation_flag", "adjudication_reason"]].to_csv(
        args.output, index=False,
    )
    print(f"reviewed={len(reviewed)} accepted=0 output={args.output}")


if __name__ == "__main__":
    main()
