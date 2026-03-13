#!/usr/bin/env python3
"""
Scan the entire zarr path column to find true ride boundaries (where the
video path / ride_id changes) and write them to actual_episode_markers.csv.

This bypasses the (buggy) safetensors episode index entirely.

Streaming & resumable:
  - Rows are flushed to CSV as each ride boundary is found.
  - On resume, reads the last row in the CSV to determine where to continue.
  - Safe to Ctrl+C at any time.

Usage:
  python overlap_checks/build_actual_episode_markers.py [dataset_root]
  python overlap_checks/build_actual_episode_markers.py [dataset_root] --resume

Output: overlap_checks/actual_episode_markers.csv
  Columns: ride_idx, ride_id, ride_ts, start_row, end_row, num_rows, path
"""

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

FIELDNAMES = ["ride_idx", "ride_id", "ride_ts", "start_row", "end_row", "num_rows", "path"]
TS_RE = re.compile(r"(\d{14})")


def _rid_from_path(p_str: str) -> str:
    return Path(p_str).stem.replace("_front_camera", "").replace("_rear_camera", "")


def _resume_state(out_path: Path):
    """Read existing CSV and return (resume_row, next_ride_idx, last_rid, last_path).

    resume_row:     the global row to start scanning from (= end_row of last
                    complete segment; the ride starting there is still open).
    next_ride_idx:  ride_idx to assign to the next *new* ride.
    last_rid:       ride_id of the still-open segment (its end_row is unknown).
    last_path:      path of that open segment.

    If the CSV is empty / doesn't exist, returns (0, 0, None, None).
    """
    if not out_path.exists() or out_path.stat().st_size == 0:
        return 0, 0, None, None

    last_row = None
    with open(out_path, newline="") as f:
        reader = csv.DictReader(f)
        for last_row in reader:
            pass

    if last_row is None:
        return 0, 0, None, None

    # The last written row is a *completed* segment.  The ride that starts at
    # its end_row is still open (we haven't found its end yet).
    resume_row = int(last_row["end_row"])
    next_ride_idx = int(last_row["ride_idx"]) + 1
    return resume_row, next_ride_idx, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root", nargs="?", default=None)
    ap.add_argument("--batch", type=int, default=50_000,
                    help="Rows to read per zarr batch (higher = faster, more RAM)")
    args = ap.parse_args()

    root = Path(
        args.dataset_root
        or os.environ.get(
            "FRODOBOTS_DATASET_ROOT",
            str(Path.home() / "fbots7k" / "extracted" / "frodobots_dataset"),
        )
    ).resolve()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.frodobots_zarr_loader import load_full_dataset

    ds = load_full_dataset(str(root))
    path_arr = ds.zarr["observation.images.front.path"]
    total = ds.total_steps
    print(f"Total rows: {total:,}", flush=True)

    out_path = Path(__file__).resolve().parent / "actual_episode_markers.csv"

    # --- Resume logic ---
    resume_row, ride_idx, current_rid, seg_path = _resume_state(out_path)

    if resume_row > 0:
        print(f"Resuming from row {resume_row:,}  (rides written so far: {ride_idx:,})", flush=True)
        # Re-read the ride_id at resume_row so we know the open segment
        p_str = str(path_arr[resume_row])
        current_rid = _rid_from_path(p_str)
        seg_path = p_str
        seg_start = resume_row
        mode = "a"  # append
    else:
        current_rid = None
        seg_path = None
        seg_start = 0
        mode = "w"

    # --- Scan ---
    t0 = time.time()
    file_is_new = (mode == "w")

    with open(out_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if file_is_new:
            writer.writeheader()
        f.flush()

        def flush_segment(end_row):
            nonlocal ride_idx
            if current_rid is None:
                return
            m = TS_RE.search(current_rid)
            writer.writerow({
                "ride_idx": ride_idx,
                "ride_id": current_rid,
                "ride_ts": m.group(1) if m else "",
                "start_row": seg_start,
                "end_row": end_row,
                "num_rows": end_row - seg_start,
                "path": seg_path,
            })
            f.flush()
            ride_idx += 1

        row = resume_row
        try:
            while row < total:
                end = min(row + args.batch, total)
                batch = path_arr[row:end]
                for i, p in enumerate(batch):
                    p_str = str(p)
                    rid = _rid_from_path(p_str)
                    if rid != current_rid:
                        if current_rid is not None:
                            flush_segment(row + i)
                        current_rid = rid
                        seg_start = row + i
                        seg_path = p_str
                row = end

                elapsed = time.time() - t0
                pct = row / total * 100
                print(
                    f"\r  {row:>12,} / {total:,}  ({pct:5.1f}%)  "
                    f"rides: {ride_idx:,}  elapsed: {elapsed:.0f}s",
                    end="", flush=True,
                )

            # Flush last segment
            flush_segment(total)

        except KeyboardInterrupt:
            # Flush the current open segment so we don't lose it
            if current_rid is not None:
                flush_segment(row)
            print(
                f"\n\nInterrupted at row {row:,}. CSV is safe to resume "
                f"(just re-run the same command).",
                flush=True,
            )
            sys.exit(1)

    elapsed = time.time() - t0
    print(f"\n\nDone in {elapsed:.1f}s", flush=True)
    print(f"Total rides found: {ride_idx:,}", flush=True)
    print(f"Written to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
