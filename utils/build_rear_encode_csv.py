#!/usr/bin/env python3
"""Build a rides CSV for rear/reverse-camera VAE encoding (Study 1 "AOO").

For each existing FORWARD weu zarr we read its ``source_video_2k`` attr, remap the
stale ``/home/ashish/frodobots/`` prefix to the canonical raw root, swap the front
camera id ``uid_s_1000`` -> rear ``uid_s_1001``, and keep only rides whose rear
HLS playlist still exists on disk. The output CSV is consumed verbatim by
``utils/pre_encode_direct.py`` (columns: ride_ts, video_path, ride_dir), so the
rear latents are produced by the *identical* Wan-VAE encode path as the forward
latents and land in a parallel zarr named by the SAME ride_ts -> 1:1 alignable.

Note: ~397 of the 858 weu rides have lost their raw footage (whole ride dir gone),
so rear encoding is only possible for the intersection that still has raw video.
This script reports that coverage and silently drops the unavailable rides.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path

import zarr

STALE_PREFIX = "/home/ashish/frodobots/"
RAW_ROOT = "/lus/lfs1aip2/projects/u6ex/fbots/frodobots_raw/"
FRONT_ID = "uid_s_1000"
REAR_ID = "uid_s_1001"


def remap(p: str) -> str:
    return p.replace(STALE_PREFIX, RAW_ROOT) if p.startswith(STALE_PREFIX) else p


def main() -> None:
    ap = argparse.ArgumentParser(description="Build rear-camera encode CSV from forward zarrs.")
    ap.add_argument("--forward_root", default="/projects/u6ex/fbots/frodobots_encoded_weu",
                    help="Root of existing FORWARD zarrs (selects which rides to mirror).")
    ap.add_argument("--out_csv", required=True, help="Output CSV path.")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, keep at most this many available rides (smoke subset).")
    args = ap.parse_args()

    zs = sorted(glob.glob(os.path.join(args.forward_root, "*.zarr")))
    rows = []
    no_attr = no_raw = 0
    for p in zs:
        try:
            a = dict(zarr.open(p, "r").attrs)
        except Exception:
            no_attr += 1
            continue
        ride_ts = str(a.get("ride_ts") or Path(p).stem)
        fwd = remap(str(a.get("source_video_2k", "")))
        ride_dir = remap(str(a.get("ride_dir_2k", "")))
        if not fwd or FRONT_ID not in fwd:
            no_attr += 1
            continue
        rear = fwd.replace(FRONT_ID, REAR_ID)
        if not os.path.exists(rear):
            no_raw += 1
            continue
        rows.append((ride_ts, rear, ride_dir))

    total = len(zs)
    avail = len(rows)
    if args.limit and avail > args.limit:
        rows = rows[: args.limit]

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ride_ts", "video_path", "ride_dir"])
        w.writerows(rows)

    print(f"forward zarrs scanned : {total}")
    print(f"  dropped (bad attrs) : {no_attr}")
    print(f"  dropped (no raw rear): {no_raw}")
    print(f"  rear available       : {avail}")
    print(f"  written to CSV       : {len(rows)}  -> {args.out_csv}")


if __name__ == "__main__":
    main()
