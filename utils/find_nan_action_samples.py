#!/usr/bin/env python3
"""
Utility to scan action CSV files and blacklist directories whose actions never
provide a NaN-free window of the required length (default 21 frames).

Example:
    python utils/find_nan_action_samples.py \
        --actions-root /projects/u5dk/as1748/frodobots_actions/train \
        --blacklist-file data_blacklist.txt \
        --window-size 21
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable


def _row_has_all_finite(row: dict[str, str], value_keys: Iterable[str]) -> bool:
    """Return True if *all* columns listed in value_keys are finite floats."""
    for key in value_keys:
        try:
            val = float(row[key])
        except (KeyError, ValueError, TypeError):
            return False
        if not math.isfinite(val):
            return False
    return True


def has_valid_window(actions_path: Path, window_size: int) -> bool:
    """
    Check whether an actions CSV file contains at least one contiguous block
    of `window_size` rows whose action values are all finite.
    """
    with actions_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "frame_id" not in reader.fieldnames:
            raise ValueError(f"{actions_path}: missing 'frame_id' column")
        value_keys = [name for name in reader.fieldnames if name != "frame_id"]
        if not value_keys:
            raise ValueError(f"{actions_path}: no action columns found")

        run = 0
        for row in reader:
            if _row_has_all_finite(row, value_keys):
                run += 1
                if run >= window_size:
                    return True
            else:
                run = 0

    return False


def load_blacklist(path: Path) -> list[str]:
    if not path.exists():
        return []
    lines = []
    for raw in path.read_text().splitlines():
        entry = raw.strip()
        if entry:
            lines.append(entry)
    return lines


def write_blacklist(path: Path, entries: Iterable[str]) -> None:
    path.write_text("\n".join(sorted(entries)) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Blacklist action folders with NaN-only windows.")
    parser.add_argument(
        "--actions-root",
        type=Path,
        required=True,
        help="Root directory containing action CSV files (e.g. frodobots_actions/train).",
    )
    parser.add_argument(
        "--blacklist-file",
        type=Path,
        default=Path("data_blacklist.txt"),
        help="Path to the data blacklist file to update.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=21,
        help="Minimum number of consecutive frames that must be NaN-free.",
    )
    args = parser.parse_args()

    actions_root = args.actions_root.resolve()
    blacklist_path = args.blacklist_file
    existing_entries = set(load_blacklist(blacklist_path))

    invalid_dirs: set[str] = set()
    csv_pattern = "input_actions_*.csv"
    for csv_path in actions_root.rglob(csv_pattern):
        try:
            valid = has_valid_window(csv_path, args.window_size)
        except Exception as exc:  # pragma: no cover - debug helper
            print(f"[WARN] Failed to parse {csv_path}: {exc}")
            valid = False

        if not valid:
            rel_dir = csv_path.parent.relative_to(actions_root)
            invalid_dirs.add(rel_dir.as_posix())

    if not invalid_dirs:
        print("No new invalid action directories found.")
        return

    updated_entries = existing_entries.union(invalid_dirs)
    write_blacklist(blacklist_path, updated_entries)

    newly_added = sorted(invalid_dirs - existing_entries)
    if newly_added:
        print("Added the following entries to the blacklist:")
        for entry in newly_added:
            print(f"  {entry}")
    else:
        print("All invalid entries were already blacklisted.")


if __name__ == "__main__":
    main()
