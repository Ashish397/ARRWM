#!/usr/bin/env python3
"""Build the rides CSV that pre_encode_direct.py consumes.

Walks an extracted FrodoBots-2K tree and emits one row per ride:

    ride_ts,video_path,ride_dir

``ride_ts`` is the timestamp suffix of the ride directory (``ride_17788_20240202090154``
-> ``20240202090154``); it becomes the name of the ride's zarr store, so it must
match what the training manifest expects. ``video_path`` is the ride's playlist
or the first video segment, and ``ride_dir`` is the directory holding the
recordings and control logs.

    python preprocessing/build_rides_csv.py \\
        --data_root $DATA_ROOT/frodobots_data --out rides.csv
    python preprocessing/pre_encode_direct.py --rides_csv rides.csv \\
        --output_root $DATA_ROOT/frodobots_encoded

Rides with no usable video are skipped and counted, not silently dropped.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

# ride_<id>_<14-digit timestamp>
RIDE_RE = re.compile(r"^ride_\d+_(\d{14})$")


def find_video(ride_dir: Path) -> Path | None:
    """The front-camera playlist, and only that.

    ``recordings/`` also holds the rear camera and the audio stream, both under
    ``uid_s_1001``. Matching on ``*.m3u8`` alone would pick the rear camera for
    any ride whose front stream is missing — a wrong-camera dataset that looks
    perfectly healthy downstream. The front camera is ``uid_s_1000``, matching
    ``pre_encode_motion.py``; a ride without one is skipped, not substituted.
    """
    rec = ride_dir / "recordings"
    if not rec.is_dir():
        return None
    hits = sorted(rec.glob("*uid_s_1000*video*.m3u8"))
    return hits[0] if hits else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", required=True,
                   help="the extracted corpus, containing output_rides_*/ride_*/")
    p.add_argument("--out", required=True, help="CSV to write")
    p.add_argument("--relative", action="store_true",
                   help="write paths relative to --data_root, for use with "
                        "pre_encode_direct.py --base_dir")
    args = p.parse_args()

    root = Path(os.path.expandvars(args.data_root)).expanduser()
    if not root.is_dir():
        raise SystemExit(f"--data_root does not exist: {root}")

    rows, skipped = [], []
    for ride_dir in sorted(root.glob("output_rides_*/ride_*")):
        m = RIDE_RE.match(ride_dir.name)
        if not m:
            continue
        video = find_video(ride_dir)
        if video is None:
            skipped.append(ride_dir.name)
            continue
        vp, rd = video, ride_dir
        if args.relative:
            vp, rd = video.relative_to(root), ride_dir.relative_to(root)
        rows.append({"ride_ts": m.group(1), "video_path": str(vp), "ride_dir": str(rd)})

    if not rows:
        raise SystemExit(
            f"no rides found under {root}.\n"
            "Expected output_rides_*/ride_<id>_<timestamp>/ — check that the "
            "archives were extracted rather than left as .zip/.tar.")

    out = Path(args.out)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ride_ts", "video_path", "ride_dir"])
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} rides -> {out}")
    if skipped:
        print(f"{len(skipped)} skipped, no video found (first few: {skipped[:3]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
