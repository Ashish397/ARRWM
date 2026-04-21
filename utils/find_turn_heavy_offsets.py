#!/usr/bin/env python3
"""Find turn-heavy 21-latent windows in rides, using the ss_vae z7 dim.

Given one or more zarr basenames, encodes their z-actions along the ride in
sliding 21-latent windows (stride configurable) and prints the offset with
the highest turn magnitude (sum_i |z7[i]|) for each.

Used as a one-shot pre-flight to pick latent_start_offset values for causal
chain runs on OOD cities where we want "turny" windows for qualitative eval.

Usage:
    python -m utils.find_turn_heavy_offsets \\
        --zarrs 20240408112440.zarr 20240209152725.zarr \\
        --encoded_root /projects/u6ex/fbots/frodobots_encoded \\
        --caption_root /projects/u6ex/fbots/frodobots_captions/train \\
        --motion_root  /projects/u6ex/fbots/frodobots_motion \\
        --ss_vae_checkpoint action_query/checkpoints/ss_vae_8free.pt \\
        --window 21 --stride 25 --min_offset 100 \\
        --output_format env

Output formats:
    env   -> lines like `BRIGHTON_OFFSET=730` (shell-sourceable).
    json  -> a single JSON object mapping zarr_basename -> offset.
    plain -> `zarr_basename offset magnitude` per line.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def find_turn_heavy_offset(
    z_ds,
    zarr_path: str,
    n_lat: int,
    window: int,
    stride: int,
    min_offset: int,
) -> tuple[int, float]:
    """Scan sliding windows and return (best_offset, best_magnitude)."""
    best_off = min_offset
    best_mag = -1.0
    max_start = n_lat - window
    if max_start < min_offset:
        raise RuntimeError(
            f"Ride {zarr_path} has {n_lat} latents; min_offset={min_offset} + "
            f"window={window} leaves no room."
        )
    offsets = list(range(min_offset, max_start + 1, stride))
    log.info(
        "Scanning %s: %d latents, %d candidate offsets (stride=%d, window=%d)",
        Path(zarr_path).name, n_lat, len(offsets), stride, window,
    )
    for off in offsets:
        z = z_ds.encode_z_actions_window(zarr_path, n_lat, off, off + window)
        # z7 is index 7 of the 8D ss_vae latent. It's tanh-squashed to (-1, 1).
        mag = float(z[:, 7].abs().sum())
        if mag > best_mag:
            best_mag = mag
            best_off = off
    return best_off, best_mag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarrs", nargs="+", required=True,
                        help="Zarr basenames (e.g. 20240408112440.zarr).")
    parser.add_argument("--encoded_root", required=True)
    parser.add_argument("--caption_root", required=True)
    parser.add_argument("--motion_root", required=True)
    parser.add_argument("--ss_vae_checkpoint", required=True)
    parser.add_argument("--window", type=int, default=21)
    parser.add_argument("--stride", type=int, default=25,
                        help="Stride between candidate offsets. Lower = finer search.")
    parser.add_argument("--min_offset", type=int, default=100,
                        help="Minimum latent_start_offset to consider (mirrors "
                             "EVAL_LATENT_START_OFFSET default).")
    parser.add_argument("--output_format", choices=["env", "json", "plain"], default="plain")
    parser.add_argument("--env_names", nargs="*", default=None,
                        help="With --output_format env: ENV var names (one per zarr). "
                             "Default: ZARR_<basename>_OFFSET (uppercased, dots-stripped).")
    args = parser.parse_args()

    from utils.eval_chain import _build_ts_to_ride_dir, _load_ride_entry_from_disk
    from utils.zarr_dataset import ZarrRideDataset

    caption_root = Path(args.caption_root)
    encoded_root = Path(args.encoded_root)
    ts_map = _build_ts_to_ride_dir(caption_root)

    rides_data = []
    for bn in args.zarrs:
        ride = _load_ride_entry_from_disk(bn, encoded_root, caption_root, ts_map)
        rides_data.append(ride)

    z_ds = ZarrRideDataset.from_manifest(
        rides_data=rides_data,
        motion_root=args.motion_root,
        ss_vae_checkpoint=args.ss_vae_checkpoint,
        device="cpu",
        ss_vae_device="cpu",
    )

    results = {}
    for bn, ride in zip(args.zarrs, rides_data):
        off, mag = find_turn_heavy_offset(
            z_ds, ride["zarr_path"], ride["n_latent_frames"],
            args.window, args.stride, args.min_offset,
        )
        log.info("  %s  best_offset=%d  sum|z7|=%.3f", bn, off, mag)
        results[bn] = {"offset": off, "turn_magnitude": mag}

    if args.output_format == "json":
        print(json.dumps(results))
    elif args.output_format == "env":
        names = args.env_names
        if names is None:
            names = [
                "ZARR_" + Path(b).stem.upper().replace(".", "_") + "_OFFSET"
                for b in args.zarrs
            ]
        if len(names) != len(args.zarrs):
            raise SystemExit("--env_names length must match --zarrs length")
        for name, bn in zip(names, args.zarrs):
            print(f"{name}={results[bn]['offset']}")
    else:
        for bn, r in results.items():
            print(f"{bn} {r['offset']} {r['turn_magnitude']:.3f}")


if __name__ == "__main__":
    main()
