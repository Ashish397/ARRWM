#!/usr/bin/env python3
"""
frodobots_overlap_check_streaming.py

Streaming + resumable overlap checker for FrodoBots 2K-style (HLS m3u8) vs FrodoBots 7K (mp4).

For each overlapping ride timestamp TS and each camera in 7K (front/rear):
  - Find best matching 2K HLS video playlist within the ride directory
  - Compare:
      * Duration match within --dur_tol seconds
      * First-frame similarity via aHash (after resizing to --resize WxH) within --hash_tol

Key improvements vs "batch then write":
  ✅ Writes CSV rows continuously (safe if interrupted)
  ✅ Resume mode: skips rows already present in CSV
  ✅ Progress output every --progress_every rows
  ✅ Graceful Ctrl+C handling

Dependencies:
  pip install pillow numpy
Requires ffmpeg + ffprobe in PATH.
"""

import argparse
import csv
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed


MP4_RE = re.compile(r"ride_[^_]+_(\d{14})_(front|rear)_camera\.mp4$", re.IGNORECASE)


@dataclass(frozen=True)
class VidRef:
    path: str
    kind: str              # "7k_mp4" or "2k_m3u8"
    camera: Optional[str]  # "front"/"rear"/None
    ts: str                # 14-digit timestamp


def run_text(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out, err


def ffprobe_duration_seconds(path: str) -> Optional[float]:
    # format duration first
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ]
    rc, out, _ = run_text(cmd)
    if rc == 0:
        s = out.strip()
        if s:
            try:
                return float(s)
            except ValueError:
                pass

    # stream duration fallback
    cmd2 = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ]
    rc, out, _ = run_text(cmd2)
    if rc == 0:
        s = out.strip()
        if s:
            try:
                return float(s)
            except ValueError:
                pass
    return None


def ffmpeg_first_frame_ppm(path: str, resize_w: int, resize_h: int) -> Optional[bytes]:
    # Extract first decoded frame, scale, output ppm to stdout
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", path,
        "-frames:v", "1",
        "-vf", f"scale={resize_w}:{resize_h}",
        "-f", "image2pipe",
        "-vcodec", "ppm",
        "pipe:1"
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = p.communicate()
    if p.returncode != 0 or not out:
        return None
    return out


def ahash_from_ppm(ppm_bytes: bytes, hash_size: int = 16) -> Optional[List[int]]:
    try:
        from PIL import Image
        import numpy as np
        import io
    except Exception:
        return None

    try:
        img = Image.open(io.BytesIO(ppm_bytes)).convert("L").resize((hash_size, hash_size), Image.BILINEAR)
        a = np.array(img)
        bits = (a > a.mean()).astype("uint8").flatten().tolist()
        return bits
    except Exception:
        return None


def hamming(a: List[int], b: List[int]) -> int:
    return sum(x != y for x, y in zip(a, b))


def index_7k_mp4s(videos_dir: str) -> Dict[Tuple[str, str], VidRef]:
    out: Dict[Tuple[str, str], VidRef] = {}
    for root, _, files in os.walk(videos_dir):
        for fn in files:
            m = MP4_RE.match(fn)
            if not m:
                continue
            ts, cam = m.group(1), m.group(2).lower()
            out[(ts, cam)] = VidRef(path=os.path.join(root, fn), kind="7k_mp4", camera=cam, ts=ts)
    return out


def index_2k_rides(r2k_base: str) -> Dict[str, str]:
    """
    Map timestamp -> ride_dir by scanning output_rides_* directories.
    """
    ts_to_dir: Dict[str, str] = {}
    for name in os.listdir(r2k_base):
        if not name.startswith("output_rides_"):
            continue
        odir = os.path.join(r2k_base, name)
        if not os.path.isdir(odir):
            continue
        for ride in os.listdir(odir):
            if not ride.startswith("ride_"):
                continue
            parts = ride.split("_")
            if len(parts) >= 3 and parts[-1].isdigit() and len(parts[-1]) == 14:
                ts = parts[-1]
                ts_to_dir[ts] = os.path.join(odir, ride)
    return ts_to_dir


def list_2k_video_playlists(ride_dir: str, ts: str) -> List[VidRef]:
    rec = os.path.join(ride_dir, "recordings")
    if not os.path.isdir(rec):
        return []

    playlists: List[VidRef] = []
    for fn in os.listdir(rec):
        if not fn.endswith("_uid_e_video.m3u8"):
            continue
        path = os.path.join(rec, fn)
        cam = None
        low = fn.lower()
        if "front" in low:
            cam = "front"
        elif "rear" in low or "back" in low:
            cam = "rear"
        playlists.append(VidRef(path=path, kind="2k_m3u8", camera=cam, ts=ts))
    return playlists


def best_match_2k_to_7k(
    ts: str,
    cam: str,
    v7: VidRef,
    playlists2k: List[VidRef],
    resize_w: int,
    resize_h: int,
    hash_size: int,
) -> Optional[Tuple[VidRef, float, int, float, float]]:
    """
    Returns best 2k playlist match:
      (v2, dur_diff, ahash_dist, dur2, dur7)
    """
    dur7 = ffprobe_duration_seconds(v7.path)
    if dur7 is None:
        return None

    ppm7 = ffmpeg_first_frame_ppm(v7.path, resize_w, resize_h)
    if ppm7 is None:
        return None
    h7 = ahash_from_ppm(ppm7, hash_size=hash_size)
    if h7 is None:
        return None

    best = None
    best_score = None

    for v2 in playlists2k:
        if v2.camera is not None and v2.camera != cam:
            continue

        dur2 = ffprobe_duration_seconds(v2.path)
        if dur2 is None:
            continue

        ppm2 = ffmpeg_first_frame_ppm(v2.path, resize_w, resize_h)
        if ppm2 is None:
            continue
        h2 = ahash_from_ppm(ppm2, hash_size=hash_size)
        if h2 is None:
            continue

        dd = abs(dur2 - dur7)
        hd = hamming(h2, h7)

        # Score: heavily weight frame similarity; then duration
        score = hd * 10 + dd
        if best_score is None or score < best_score:
            best_score = score
            best = (v2, dd, hd, dur2, dur7)

    return best


def load_completed_keys(out_csv: str) -> Set[Tuple[str, str]]:
    """
    Resume support: read existing CSV and return set of (ts,camera) already processed.
    """
    done: Set[Tuple[str, str]] = set()
    if not os.path.exists(out_csv):
        return done
    try:
        with open(out_csv, "r", newline="") as f:
            r = csv.DictReader(f)
            if "ts" not in r.fieldnames or "camera" not in r.fieldnames:
                return done
            for row in r:
                ts = (row.get("ts") or "").strip()
                cam = (row.get("camera") or "").strip()
                if ts and cam:
                    done.add((ts, cam))
    except Exception:
        # If the CSV is partially written/corrupted, don’t crash—just treat as no resume.
        return set()
    return done


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r2k_base", required=True, help="Path to frodobots_data (contains output_rides_*)")
    ap.add_argument("--v7k_dir", required=True, help="Path to 7K videos directory (contains MP4s)")
    ap.add_argument("--out_csv", required=True, help="Output CSV path (written incrementally)")
    ap.add_argument("--max_ts", type=int, default=0, help="Limit number of overlapping timestamps (0 = no limit)")
    ap.add_argument("--workers", type=int, default=4, help="Parallel workers (ffmpeg/ffprobe calls)")
    ap.add_argument("--dur_tol", type=float, default=0.10, help="Duration tolerance seconds for match")
    ap.add_argument("--hash_tol", type=int, default=10, help="aHash Hamming tolerance for first-frame match")
    ap.add_argument("--resize", default="224x128", help="Resize WxH used before hashing (e.g. 224x128)")
    ap.add_argument("--hash_size", type=int, default=16, help="aHash size (16 => 256-bit)")
    ap.add_argument("--progress_every", type=int, default=100, help="Print progress every N completed items")
    args = ap.parse_args()

    if "x" not in args.resize:
        raise SystemExit("--resize must be like 224x128")
    resize_w, resize_h = map(int, args.resize.split("x"))

    ensure_parent_dir(args.out_csv)

    # Resume: load completed (ts,camera) keys
    completed = load_completed_keys(args.out_csv)
    if completed:
        print(f"Resume: found {len(completed)} already-completed (ts,camera) rows in {args.out_csv}")

    # Index (fast)
    mp4_map = index_7k_mp4s(args.v7k_dir)          # (ts,cam)->VidRef
    ts_to_ride = index_2k_rides(args.r2k_base)     # ts->ride_dir

    ts7 = {ts for (ts, _cam) in mp4_map.keys()}
    ts2 = set(ts_to_ride.keys())
    overlap_ts = sorted(ts7 & ts2)
    if args.max_ts and args.max_ts > 0:
        overlap_ts = overlap_ts[: args.max_ts]

    print(f"7K timestamps: {len(ts7)}")
    print(f"2K timestamps: {len(ts2)}")
    print(f"Overlapping timestamps: {len(overlap_ts)}")

    # Build tasks for 7K cameras present
    tasks: List[Tuple[str, str, VidRef]] = []
    for ts in overlap_ts:
        cam = "front"
        key = (ts, cam)
        if key in completed:
            continue
        v7 = mp4_map.get(key)
        if v7:
            tasks.append((ts, cam, v7))

    total_tasks = len(tasks)
    print(f"To process (after resume/max_ts): {total_tasks}")

    fieldnames = [
        "ts", "camera",
        "dur_2k", "dur_7k", "dur_diff", "dur_match",
        "ahash_dist", "frame_match",
        "path_2k", "path_7k",
        "ride_dir_2k",
        "note",
    ]

    file_exists = os.path.exists(args.out_csv)
    f = open(args.out_csv, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fieldnames)
    if not file_exists:
        w.writeheader()
        f.flush()

    done = 0
    dur_ok = 0
    frame_ok = 0
    both_ok = 0

    def process_one(ts: str, cam: str, v7: VidRef) -> Dict[str, object]:
        ride_dir = ts_to_ride.get(ts)
        if not ride_dir:
            return {
                "ts": ts, "camera": cam,
                "dur_7k": ffprobe_duration_seconds(v7.path),
                "dur_2k": None, "dur_diff": None,
                "dur_match": False,
                "ahash_dist": None,
                "frame_match": False,
                "path_7k": v7.path,
                "path_2k": None,
                "ride_dir_2k": None,
                "note": "missing_2k_ride_dir",
            }

        playlists2k = list_2k_video_playlists(ride_dir, ts)
        if not playlists2k:
            return {
                "ts": ts, "camera": cam,
                "dur_7k": ffprobe_duration_seconds(v7.path),
                "dur_2k": None, "dur_diff": None,
                "dur_match": False,
                "ahash_dist": None,
                "frame_match": False,
                "path_7k": v7.path,
                "path_2k": None,
                "ride_dir_2k": ride_dir,
                "note": "no_2k_playlists_found",
            }

        best = best_match_2k_to_7k(
            ts, cam, v7, playlists2k,
            resize_w=resize_w, resize_h=resize_h,
            hash_size=args.hash_size
        )
        if best is None:
            return {
                "ts": ts, "camera": cam,
                "dur_7k": ffprobe_duration_seconds(v7.path),
                "dur_2k": None, "dur_diff": None,
                "dur_match": False,
                "ahash_dist": None,
                "frame_match": False,
                "path_7k": v7.path,
                "path_2k": None,
                "ride_dir_2k": ride_dir,
                "note": "failed_to_decode_or_match_2k",
            }

        v2, dur_diff, ahd, dur2, dur7 = best
        dur_match = abs(dur2 - dur7) <= args.dur_tol
        frame_match = ahd <= args.hash_tol

        return {
            "ts": ts, "camera": cam,
            "dur_2k": dur2, "dur_7k": dur7,
            "dur_diff": dur_diff,
            "dur_match": dur_match,
            "ahash_dist": ahd,
            "frame_match": frame_match,
            "path_2k": v2.path,
            "path_7k": v7.path,
            "ride_dir_2k": ride_dir,
            "note": "",
        }

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = [ex.submit(process_one, ts, cam, v7) for (ts, cam, v7) in tasks]
            for fut in as_completed(futs):
                row = fut.result()

                # write row immediately
                w.writerow(row)
                f.flush()

                done += 1
                if row.get("dur_match"):
                    dur_ok += 1
                if row.get("frame_match"):
                    frame_ok += 1
                if row.get("dur_match") and row.get("frame_match"):
                    both_ok += 1

                if args.progress_every and (done % args.progress_every == 0 or done == total_tasks):
                    print(
                        f"Progress {done}/{total_tasks} | dur_ok={dur_ok} frame_ok={frame_ok} both_ok={both_ok}",
                        flush=True
                    )
    except KeyboardInterrupt:
        print("\nInterrupted (Ctrl+C). CSV is up-to-date; you can resume by rerunning the same command.", flush=True)
    finally:
        f.close()

    # Final summary
    print(f"Finished/wrote: {args.out_csv}")
    print(f"Processed: {done}/{total_tasks}")
    print(f"Duration matches: {dur_ok}/{done if done else 0}")
    print(f"First-frame matches: {frame_ok}/{done if done else 0}")
    print(f"Both match: {both_ok}/{done if done else 0}")


if __name__ == "__main__":
    main()
