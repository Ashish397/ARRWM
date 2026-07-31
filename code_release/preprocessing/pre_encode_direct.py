#!/usr/bin/env python3
"""Pre-encode FrodoBots 2K videos with Wan 2.1 VAE — no 7K dependency.

Takes a CSV with columns: ride_ts, video_path, ride_dir
Encodes the full video (no action trimming) into per-episode Zarr stores.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

# Reuse all the heavy machinery from pre_encode_local
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent))

from pre_encode_local import (
    VideoLoader,
    VAEEncoder,
    latents_to_time_major_numpy,
    zarr_has_latents,
    _zarr_blosc,
    make_blosc,
    parse_dtype,
    expand_path,
    require_ffmpeg,
    safe_mkdir,
)


@dataclass
class DirectRideEntry:
    ride_ts: str
    video_path: str
    ride_dir: str


def load_direct_schedule(csv_path: Path, base_dir: Optional[Path] = None) -> List[DirectRideEntry]:
    schedule = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            ts = row["ride_ts"]
            video_path = row["video_path"]
            ride_dir = row.get("ride_dir", "")
            if base_dir:
                if video_path and not os.path.isabs(video_path):
                    video_path = str((base_dir / video_path).resolve())
                if ride_dir and not os.path.isabs(ride_dir):
                    ride_dir = str((base_dir / ride_dir).resolve())
            schedule.append(DirectRideEntry(ride_ts=ts, video_path=video_path, ride_dir=ride_dir))
    return schedule


def process_ride_direct(
    entry: DirectRideEntry,
    encoder: VAEEncoder,
    video_loader: VideoLoader,
    out_root: Path,
    overwrite: bool,
    encode_stride: int,
    zarr_latent_chunk: int,
    latent_out_dtype: np.dtype,
    compressor_cfg: dict,
) -> None:
    episode_id = entry.ride_ts
    out_zarr = out_root / f"{episode_id}.zarr"
    if not overwrite and zarr_has_latents(out_zarr):
        print(f"[SKIP] {episode_id}: already encoded at {out_zarr}")
        return
    video_path = entry.video_path
    if not video_path or not os.path.exists(video_path):
        print(f"[SKIP] {episode_id}: video not found at {video_path}")
        return

    if overwrite and out_zarr.exists():
        shutil.rmtree(out_zarr, ignore_errors=True)

    zarr, Blosc = _zarr_blosc()
    comp = make_blosc(Blosc, **compressor_cfg)
    g = zarr.open_group(str(out_zarr), mode="w")
    block_size = encode_stride + 1

    for k, v in [
        ("episode_id", episode_id), ("ride_ts", entry.ride_ts),
        ("source_video_2k", video_path), ("ride_dir_2k", entry.ride_dir),
        ("encode_stride", encode_stride), ("video_block_size", block_size),
        ("zarr_latent_chunk", zarr_latent_chunk),
        ("expected_latents_per_full_block", 1 + encode_stride // 4),
        ("latent_dtype", str(latent_out_dtype)),
    ]:
        g.attrs[k] = v

    # Encode full video — no start/duration trimming
    info, blocks = video_loader.stream_blocks(video_path, block_size=encode_stride)
    g.attrs["fps"] = float(info.fps)
    print(f"[RIDE] {episode_id} full video {info.width}x{info.height}")

    lat_arr = None
    ts_arr = None
    blk_i = 0
    ride_t0 = time.monotonic()
    for frames_block, ts_block in blocks:
        t_decode = time.monotonic() - (t_write_end if blk_i > 0 else ride_t0)
        print(f"  block {blk_i}: {frames_block.shape[0]}f {frames_block.dtype} "
              f"pts=[{ts_block[0]:.3f}..{ts_block[-1]:.3f}] decode={t_decode:.1f}s", flush=True)

        t0 = time.monotonic()
        lat = encoder.encode_block(frames_block)
        lat_np = latents_to_time_major_numpy(lat, latent_out_dtype)
        t_enc = time.monotonic() - t0

        t_lat, t_ts = lat_np.shape[0], ts_block.shape[0]
        if blk_i == 0:
            print(f"  latent shape={lat_np.shape}, dtype={lat_np.dtype}", flush=True)
        if lat_arr is None:
            lat_arr = g.create_dataset(
                "latents", shape=(0, *lat_np.shape[1:]), chunks=(zarr_latent_chunk, *lat_np.shape[1:]),
                dtype=latent_out_dtype, compressor=comp, overwrite=True,
            )
            ts_arr = g.create_dataset(
                "timestamps", shape=(0,), chunks=(block_size,), dtype=np.float64, compressor=comp, overwrite=True,
            )

        t0 = time.monotonic()
        lat_arr.resize(lat_arr.shape[0] + t_lat, *lat_arr.shape[1:])
        lat_arr[-t_lat:] = lat_np
        ts_arr.resize(ts_arr.shape[0] + t_ts)
        ts_arr[-t_ts:] = ts_block
        t_write = time.monotonic() - t0
        t_write_end = time.monotonic()

        print(f"  block {blk_i}: +{t_lat} lat (total {lat_arr.shape[0]}) "
              f"enc={t_enc:.1f}s write={t_write:.1f}s", flush=True)
        blk_i += 1

    elapsed = time.monotonic() - ride_t0
    if lat_arr is not None:
        g.attrs["action_start_sec"] = float(ts_arr[0])
        g.attrs["action_end_sec"] = float(ts_arr[-1])
        print(f"[DONE] {episode_id}: {lat_arr.shape[0]} latents ({lat_arr.dtype}), "
              f"{ts_arr.shape[0]} ts, {blk_i} blocks in {elapsed:.1f}s -> {out_zarr}")


def main() -> None:
    p = argparse.ArgumentParser(description="Pre-encode FrodoBots 2K (direct, no 7K dependency).")
    p.add_argument("--rides_csv", required=True, help="CSV with ride_ts, video_path, ride_dir columns")
    p.add_argument("--base_dir", default=None, help="Base dir for resolving relative paths in CSV")
    p.add_argument("--output_root", default=os.path.join(os.environ.get("DATA_ROOT", ""), "frodobots_encoded"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--vae_path", default=os.path.join(os.environ.get("DATA_ROOT", ""), "Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"))
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--model_dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--latent_dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--scale", default="832x480", help="WxH")
    p.add_argument("--encode_stride", type=int, default=600)
    p.add_argument("--zarr_latent_chunk", type=int, default=32)
    p.add_argument("--compressor_cname", default="zstd")
    p.add_argument("--compressor_level", type=int, default=1)
    p.add_argument("--compressor_shuffle", default="bitshuffle", choices=["none", "shuffle", "bitshuffle"])
    args = p.parse_args()

    if "x" not in args.scale:
        raise ValueError("--scale must be WxH")
    scale_w, scale_h = map(int, args.scale.split("x"))

    out_root = expand_path(args.output_root)
    safe_mkdir(out_root)
    base_dir = expand_path(args.base_dir) if args.base_dir else None

    schedule = load_direct_schedule(expand_path(args.rides_csv), base_dir=base_dir)
    if not schedule:
        print("No rides to process.")
        return
    print(f"Schedule: {len(schedule)} rides")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("WARNING: using CPU")
    require_ffmpeg()

    # Import path for WanVAEWrapper
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    encoder = VAEEncoder(device=device, model_dtype=torch.float16 if args.model_dtype == "float16" else torch.float32, vae_path=args.vae_path)
    video_loader = VideoLoader(scale=(scale_w, scale_h))
    comp_cfg = {"cname": args.compressor_cname, "clevel": args.compressor_level, "shuffle": args.compressor_shuffle}

    done = errors = 0
    for entry in tqdm(schedule, desc="Rides"):
        try:
            process_ride_direct(
                entry, encoder, video_loader, out_root,
                args.overwrite, args.encode_stride, args.zarr_latent_chunk,
                parse_dtype(args.latent_dtype), comp_cfg,
            )
            done += 1
        except Exception as e:
            print(f"[ERROR] {entry.ride_ts}: {e}")
            errors += 1
    print(f"Done: {done}, Errors: {errors}")


if __name__ == "__main__":
    main()
