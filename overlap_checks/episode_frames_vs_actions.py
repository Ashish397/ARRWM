#!/usr/bin/env python3
"""
Per-ride analysis: first/last timestamps, elapsed duration, time-skip check,
num_actions, actions_hz, frames (20 fps), and % of video that has actions.

Looks up ride boundaries from actual_episode_markers.csv (built by
build_actual_episode_markers.py) instead of the buggy safetensors index.

Usage:
  python overlap_checks/episode_frames_vs_actions.py [dataset_root] <ride_timestamp>

  ride_timestamp  e.g. 20240521042959  (or integer index into matches.csv)
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

FRAMES_HZ = 20
MARKERS_CSV = Path(__file__).resolve().parent / "actual_episode_markers.csv"


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout.strip(), r.stderr


def find_ride_rows(ride_ts: str):
    """Look up a ride by its 14-digit timestamp in actual_episode_markers.csv.

    Returns (start_row, end_row, ride_id, path_rel) or None.
    """
    if not MARKERS_CSV.exists():
        print(f"Error: {MARKERS_CSV} not found. "
              f"Run build_actual_episode_markers.py first.", file=sys.stderr)
        sys.exit(1)

    with open(MARKERS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["ride_ts"] == ride_ts:
                return (
                    int(row["start_row"]),
                    int(row["end_row"]),
                    row["ride_id"],
                    row["path"],
                )
    return None


def find_video_path(root: Path, path_rel: str):
    """Resolve mp4: try root/path_rel, then videos dirs outside extracted."""
    name = Path(path_rel).name
    candidates = [
        root / path_rel,
        root.parent / "videos" / name,
        root.parent.parent / "videos" / name,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def ffprobe_duration_and_frames(path: Path) -> tuple[float | None, int | None]:
    """Return (duration_sec, frame_count). Uses nb_frames then format duration."""
    # nb_frames (may be missing)
    rc, out, _ = _run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames", "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ])
    frame_count = int(out) if rc == 0 and out and out.isdigit() else None
    # duration
    rc, out, _ = _run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ])
    duration = float(out) if rc == 0 and out else None
    if duration is None:
        rc, out, _ = _run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)
        ])
        duration = float(out) if rc == 0 and out else None
    if frame_count is None and duration and duration > 0:
        # Fallback: count frames (slow)
        rc, out, _ = _run([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=nb_read_frames", "-of", "default=noprint_wrappers=1:nokey=1", str(path)
        ])
        if rc == 0 and out and out.isdigit():
            frame_count = int(out)
    return duration, frame_count


def resolve_ride_ts(raw: str) -> str:
    """If raw is a 14-digit number, return it directly as the ride timestamp.
    Otherwise treat it as a 0-based row index into matches.csv and return the
    'ts' column value for that row."""
    if len(raw) == 14 and raw.isdigit():
        return raw

    try:
        idx = int(raw)
    except ValueError:
        print(f"Error: '{raw}' is neither a 14-digit ride timestamp nor an integer index.", file=sys.stderr)
        sys.exit(1)

    matches_path = Path(__file__).resolve().parent / "matches.csv"
    if not matches_path.exists():
        print(f"Error: matches.csv not found at {matches_path}", file=sys.stderr)
        sys.exit(1)

    with open(matches_path, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i == idx:
                ts = row["ts"]
                print(f"matches.csv row {idx} -> ts={ts}")
                return ts

    print(f"Error: index {idx} out of range for matches.csv", file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root", nargs="?", default=None)
    ap.add_argument("ride", type=str,
                    help="14-digit ride timestamp (e.g. 20240521042959) or "
                         "row index into matches.csv")
    args = ap.parse_args()

    ride_ts = resolve_ride_ts(args.ride)

    root = Path(args.dataset_root or os.environ.get("FRODOBOTS_DATASET_ROOT", str(Path.home() / "fbots7k" / "extracted" / "frodobots_dataset"))).resolve()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.frodobots_zarr_loader import load_full_dataset

    result = find_ride_rows(ride_ts)
    if result is None:
        print(f"No rows found matching '{ride_ts}' in {MARKERS_CSV.name}", file=sys.stderr)
        sys.exit(1)

    ds = load_full_dataset(str(root))

    start, end, ride_id, path_rel = result
    n_actions = end - start

    ts_arr = ds.zarr["observation.images.front.timestamp"]
    try:
        ts = np.asarray(ts_arr[start:end], dtype=np.float64)
    except Exception:
        ts = np.array([float(ts_arr[i]) for i in range(start, end)], dtype=np.float64)

    # first / last by row order (ts is already sliced, so use local indices)
    t_first = float(ts[0])
    t_last = float(ts[-1])
    elapsed = t_last - t_first

    dt = np.diff(ts)
    valid = np.isfinite(dt)
    sum_dt = float(np.sum(dt[valid]))
    match = abs(sum_dt - elapsed) < 1e-6 if elapsed >= 0 else False
    time_skips = "no" if match else "yes (sum_dt != last - first or rows not time-ordered)"

    actions_hz = (n_actions - 1) / elapsed if elapsed > 0 else None
    n_frames = int(round(elapsed * FRAMES_HZ)) if elapsed > 0 else 0

    video_path = find_video_path(root, path_rel)
    video_duration_sec = video_frame_count = None
    actions_available_percent = None
    if video_path:
        video_duration_sec, video_frame_count = ffprobe_duration_and_frames(video_path)
        if video_duration_sec and video_duration_sec > 0:
            actions_available_percent = (elapsed / video_duration_sec) * 100.0

    print(f"ride_id: {ride_id}")
    print(f"row range: [{start}, {end})")
    print(f"first_ts: {t_first}")
    print(f"last_ts: {t_last}")
    print(f"elapsed_sec: {elapsed}")
    print(f"sum(dt): {sum_dt}")
    print(f"time_skips: {time_skips}")
    print(f"num_actions: {n_actions}")
    print(f"actions_hz: {actions_hz}")
    print(f"frames_hz: {FRAMES_HZ}")
    print(f"frames (elapsed * frames_hz): {n_frames}")
    if video_path:
        print(f"video_path: {video_path}")
        if video_duration_sec is not None:
            print(f"video_duration_sec: {video_duration_sec}")
        if video_frame_count is not None:
            print(f"video_frame_count: {video_frame_count}")
        if actions_available_percent is not None:
            print(f"actions_available_percent: {actions_available_percent:.2f}%")
    else:
        print(f"video_path: not found (tried {path_rel} under root and parent videos dirs)")


if __name__ == "__main__":
    main()
