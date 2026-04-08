#!/usr/bin/env python3
"""Test ZarrRideDataset with the lockstep ride batcher.

Mirrors the context-shifted teacher-forcing training semantics:
  - ``batch_size`` rides are selected per group.
  - All rides in the group are advanced in lockstep through non-overlapping
    ``window_size``-frame windows, each prepended with ``context_frames``
    leading context (default 3).
  - Each window loads ``window_size + context_frames`` latent frames.
  - Each slot's VAE decoder cache is reset only when the ride changes.

For each slot × window, decodes latents through the Wan VAE, overlays
motion arrows + ss_vae z_action bars + steering dial + slot/window/ride
annotations, and writes annotated MP4 clips.

Usage:
    python testing/test_zarr_dataloader.py \\
        --num_rides 4 --windows_per_ride 3 --batch_size 2 \\
        --output_dir testing/outputs \\
        [--context_frames 3] \\
        [--seed 42] \\
        [--wan_model_path /path/to/Wan2.1/]
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
import zarr as zarr_lib

from model.causal_teacher_streaming import LockstepRideBatcher
from utils.zarr_dataset import (
    ZarrRideDataset,
    _load_aligned_motion_for_zarr,
    _LATENT_TO_VIDEO,
)

# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def draw_motion_overlay(
    frame: np.ndarray,
    motion_vecs: np.ndarray,
    mag_scale: float = 30.0,
) -> np.ndarray:
    h, w = frame.shape[:2]
    out = frame.copy()
    if motion_vecs.ndim != 2 or motion_vecs.shape[1] != 3:
        return out
    N = motion_vecs.shape[0]
    grid_size = int(round(np.sqrt(N)))
    if grid_size * grid_size != N or grid_size <= 0:
        return out
    for gy in range(grid_size):
        for gx in range(grid_size):
            idx = gy * grid_size + gx
            dx, dy, vis = motion_vecs[idx]
            if vis < 0.2:
                continue
            cx = int((gx + 0.5) * w / grid_size)
            cy = int((gy + 0.5) * h / grid_size)
            end_x = int(cx - dx * mag_scale)
            end_y = int(cy - dy * mag_scale)
            color = (0, 255, 0) if vis >= 0.5 else (0, 200, 255)
            cv2.arrowedLine(out, (cx, cy), (end_x, end_y), color, 1, tipLength=0.3)
    return out


def draw_latent_overlay(
    frame: np.ndarray,
    latent: np.ndarray,
    clip: float = 1.0,
) -> np.ndarray:
    h, w = frame.shape[:2]
    out = frame.copy()
    n = len(latent)
    panel_w = 90
    slot_h = h // n
    out[:, w - panel_w :] = (out[:, w - panel_w :].astype(np.float32) * 0.35).astype(np.uint8)
    half = (panel_w - 10) // 2
    cx = w - panel_w + half + 5
    for i, v in enumerate(latent):
        y_mid = i * slot_h + slot_h // 2
        bar_len = int(min(abs(float(v)), clip) / clip * half)
        bar_top = y_mid - max(slot_h // 5, 3)
        bar_bot = y_mid + max(slot_h // 5, 3)
        color = (80, 220, 80) if v >= 0 else (80, 80, 220)
        if v >= 0:
            cv2.rectangle(out, (cx, bar_top), (cx + bar_len, bar_bot), color, -1)
        else:
            cv2.rectangle(out, (cx - bar_len, bar_top), (cx, bar_bot), color, -1)
        cv2.line(out, (cx, bar_top - 1), (cx, bar_bot + 1), (200, 200, 200), 1)
        cv2.putText(
            out,
            f"z{i}:{float(v):+.2f}",
            (w - panel_w + 2, y_mid + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
    return out


def draw_latent_dial(
    frame: np.ndarray,
    latent: np.ndarray,
    turn_idx: int = 2,
    fwd_idx: int = 7,
) -> np.ndarray:
    h, w = frame.shape[:2]
    out = frame
    bar_h = 60
    cx, cy = w - 110, h - bar_h + 30
    cv2.circle(out, (cx, cy), 22, (60, 60, 80), -1)
    cv2.circle(out, (cx, cy), 22, (180, 180, 255), 1)
    z_turn = float(latent[turn_idx])
    z_fwd = float(latent[fwd_idx])
    arrow_len = 18
    dx = int(z_turn * arrow_len)
    dy = int(-z_fwd * arrow_len)
    dx = max(-arrow_len, min(arrow_len, dx))
    dy = max(-arrow_len, min(arrow_len, dy))
    color = (255, 200, 100) if z_fwd >= 0 else (100, 150, 255)
    cv2.arrowedLine(out, (cx, cy), (cx + dx, cy + dy), color, 2, tipLength=0.35)
    cv2.putText(
        out, "z", (cx - 5, cy - 26),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 255), 1, cv2.LINE_AA,
    )
    return out


def draw_frame_info(
    frame: np.ndarray,
    slot_idx: int,
    window_idx: int,
    frame_idx: int,
    latent_idx: int,
    window_start: int,
    zarr_name: str,
) -> np.ndarray:
    h, w = frame.shape[:2]
    out = frame.copy()
    bar_h = 60
    region = out[h - bar_h : h, :, :].astype(np.float32)
    out[h - bar_h : h, :, :] = (region * 0.4).astype(np.uint8)
    y0 = h - bar_h + 18
    cv2.putText(
        out,
        f"slot={slot_idx}  win={window_idx}  frame={frame_idx}  lat={latent_idx}  start={window_start}",
        (10, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        zarr_name,
        (10, y0 + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (180, 255, 180),
        1,
        cv2.LINE_AA,
    )
    return out


def _latent_to_video_frames(n_latent: int) -> int:
    """Number of video frames produced by n_latent latent frames."""
    return 1 + _LATENT_TO_VIDEO * (n_latent - 1) if n_latent > 0 else 0


def _save_mp4(frames_arr: np.ndarray, path: Path, fps: float) -> int:
    fh_l, fw_l = frames_arr.shape[1], frames_arr.shape[2]
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{fw_l}x{fh_l}", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    proc.stdin.write(frames_arr.tobytes())
    proc.stdin.close()
    proc.wait()
    return proc.returncode


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Lockstep ride-batcher debug visualiser. "
                    "Mirrors the training dataloader semantics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=0,
                        help="Determines starting zarr index via seed %% num_zarrs.")
    parser.add_argument("--num_rides", type=int, default=4,
                        help="Total rides to index from dataset.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Rides per lockstep group (mirrors training batch_size).")
    parser.add_argument("--windows_per_ride", type=int, default=3,
                        help="Max non-overlapping windows to decode per ride.")
    parser.add_argument("--window_size", type=int, default=21,
                        help="Supervised latent frames per window (matches training).")
    parser.add_argument("--context_frames", type=int, default=3,
                        help="Leading context frames prepended to each window "
                             "(loaded total = window_size + context_frames).")
    parser.add_argument("--encoded_root", default="/projects/u6ej/fbots/frodobots_encoded")
    parser.add_argument("--caption_root", default="/projects/u6ej/fbots/frodobots_captions/train")
    parser.add_argument("--motion_root", default="/projects/u6ej/fbots/frodobots_motion")
    parser.add_argument("--ss_vae_checkpoint", default="action_query/checkpoints/ss_vae_8free.pt")
    parser.add_argument("--wan_model_path", default=None,
                        help="Path to Wan2.1 root dir (should contain Wan2.1-T2V-1.3B/).")
    parser.add_argument("--output_dir", default="testing/outputs")
    parser.add_argument("--combined", action="store_true",
                        help="Save all windows for a slot as one combined video "
                             "instead of separate per-window clips.")
    parser.add_argument("--device", default=None,
                        help="Force a single device (e.g. cuda:0). "
                             "If omitted, one GPU per batch slot is used.")
    parser.add_argument("--num_gpus", type=int, default=None,
                        help="GPUs to use (default: min(batch_size, available)).")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Build ride-level dataset ──────────────────────────────────────
    zarr_paths = sorted(Path(args.encoded_root).glob("*.zarr"))
    if not zarr_paths:
        raise RuntimeError(f"No zarr files found in {args.encoded_root}")

    start_zarr_index = args.seed % len(zarr_paths)
    logging.info(
        "Building ZarrRideDataset (start_zarr=%d, max_rides=%d, "
        "min_ride_frames=%d [window=%d + context=%d]) ...",
        start_zarr_index, args.num_rides,
        args.window_size + args.context_frames,
        args.window_size, args.context_frames,
    )

    dataset = ZarrRideDataset(
        encoded_root=args.encoded_root,
        caption_root=args.caption_root,
        motion_root=args.motion_root,
        ss_vae_checkpoint=args.ss_vae_checkpoint,
        min_ride_frames=args.window_size + args.context_frames,
        device="cpu",
        start_zarr_index=start_zarr_index,
        max_rides=args.num_rides,
    )
    logging.info("Dataset ready: %d rides.", len(dataset))

    if len(dataset) < args.batch_size:
        raise RuntimeError(
            f"Need at least batch_size={args.batch_size} rides, "
            f"but dataset has only {len(dataset)}"
        )

    # ── Resolve devices ───────────────────────────────────────────────
    if args.device is not None:
        devices = [torch.device(args.device)]
    elif torch.cuda.is_available():
        n_avail = torch.cuda.device_count()
        n_use = min(args.num_gpus or args.batch_size, n_avail)
        devices = [torch.device(f"cuda:{i}") for i in range(n_use)]
    else:
        devices = [torch.device("cpu")]

    logging.info("Using %d device(s): %s", len(devices), devices)

    def slot_device(slot_idx: int) -> torch.device:
        return devices[slot_idx % len(devices)]

    # ── Load one VAE per device ───────────────────────────────────────
    from utils.wan_wrapper import WanVAEWrapper  # noqa: PLC0415

    vae_kwargs: dict = {}
    if args.wan_model_path:
        vae_kwargs["model_root"] = Path(args.wan_model_path) / "Wan2.1-T2V-1.3B"

    vaes: dict = {}
    for dev in devices:
        logging.info("Loading Wan VAE on %s ...", dev)
        v = WanVAEWrapper(**vae_kwargs)
        v = v.to(device=dev, dtype=torch.float16).eval()
        vaes[dev] = v
    logging.info("VAE(s) ready on %d device(s).", len(vaes))

    motion_root = Path(args.motion_root)
    saved_count = 0

    # ── Build batcher ─────────────────────────────────────────────────
    batcher = LockstepRideBatcher(
        window_size=args.window_size,
        num_frame_per_block=3,
        batch_size=args.batch_size,
        max_windows_per_ride=args.windows_per_ride,
        context_frames=args.context_frames,
    )

    ride_cursor = 0
    group_counter = 0

    # ── Helper: load motion data for a ride ───────────────────────────
    def _load_motion(zarr_path_str: str, n_video: int):
        try:
            g = zarr_lib.open_group(zarr_path_str, mode="r")
            attrs = dict(g.attrs)
            motion = _load_aligned_motion_for_zarr(attrs, n_video, motion_root)
            if motion.shape[0] < n_video:
                pad = n_video - motion.shape[0]
                motion = np.concatenate(
                    [motion, np.repeat(motion[-1:], pad, axis=0)], axis=0
                )
            return motion
        except Exception as exc:
            logging.warning("  Could not load motion: %s", exc)
            return None

    # ── Main loop: iterate groups ─────────────────────────────────────
    while ride_cursor + args.batch_size <= len(dataset):
        ride_dicts = [dataset[ride_cursor + s] for s in range(args.batch_size)]
        batcher.load_group(ride_dicts)
        ride_cursor += args.batch_size
        group_counter += 1

        logging.info(
            "═══ Group %d  %s ═══",
            group_counter, batcher.summary(),
        )

        # Per-slot state: VAE cache, video cursor, motion, combined frames
        per_slot_vid_cursor = [0] * args.batch_size
        per_slot_motion: list = []
        per_slot_combined: list = [[] for _ in range(args.batch_size)]
        per_slot_zarr_name: list = []
        per_slot_fps: list = []

        def _refresh_slot_state(slot_idx: int) -> None:
            slot = batcher.get_slot_info()[slot_idx]
            per_slot_vid_cursor[slot_idx] = 0
            per_slot_combined[slot_idx] = []
            per_slot_zarr_name[slot_idx] = Path(slot.zarr_path).name

            g = zarr_lib.open_group(slot.zarr_path, mode="r")
            per_slot_fps[slot_idx] = float(dict(g.attrs).get("fps", 20.0))

            n_video = _latent_to_video_frames(slot.n_latent_frames)
            per_slot_motion[slot_idx] = _load_motion(slot.zarr_path, n_video)

        def _flush_combined_slot(slot_idx: int, slot) -> None:
            nonlocal saved_count
            if not args.combined or not per_slot_combined[slot_idx]:
                return
            all_arr = np.concatenate(per_slot_combined[slot_idx])
            stem = Path(slot.zarr_path).stem
            fps = per_slot_fps[slot_idx]
            out_path = (
                out_dir
                / f"g{group_counter:02d}_s{slot_idx}_{stem}"
                  f"_{len(per_slot_combined[slot_idx])}x{args.window_size}lat.mp4"
            )
            rc = _save_mp4(all_arr, out_path, fps)
            if rc == 0:
                logging.info(
                    "  Saved %s  (%d frames, %.1f s @ %.0f fps)",
                    out_path, all_arr.shape[0],
                    all_arr.shape[0] / fps, fps,
                )
                saved_count += 1
            else:
                logging.error("  ffmpeg failed with exit code %d", rc)
            per_slot_combined[slot_idx] = []
            del all_arr

        per_slot_motion.extend([None] * args.batch_size)
        per_slot_zarr_name.extend([""] * args.batch_size)
        per_slot_fps.extend([20.0] * args.batch_size)

        for s in range(args.batch_size):
            _refresh_slot_state(s)

        # Reset VAE cache on every device at the start of each group
        for v in vaes.values():
            v.model.clear_cache()

        dataset_exhausted = False
        while True:
            exhausted = batcher.exhausted_slot_indices()
            if exhausted and not dataset_exhausted:
                available = len(dataset) - ride_cursor
                n_fill = min(len(exhausted), max(available, 0))
                fill_indices = exhausted[:n_fill]
                if fill_indices:
                    for s in fill_indices:
                        _flush_combined_slot(s, batcher.get_slot_info()[s])
                    ride_dicts = [dataset[ride_cursor + i] for i in range(n_fill)]
                    ride_cursor += n_fill
                    batcher.refill_slots(fill_indices, ride_dicts)
                    for s in fill_indices:
                        _refresh_slot_state(s)
                    for v in vaes.values():
                        v.model.clear_cache()
                if n_fill < len(exhausted):
                    dataset_exhausted = True

            slots = batcher.get_slot_info()
            active_slots = [
                s for s, slot in enumerate(slots)
                if slot.loaded and slot.window_idx < slot.n_windows
            ]
            if not active_slots:
                break

            bounds = batcher.get_window_bounds()

            for s in active_slots:
                lat_s, lat_e = bounds[s]
                slot = slots[s]
                win_idx = slot.window_idx
                zarr_name = per_slot_zarr_name[s]
                fps = per_slot_fps[s]

                z_act_window = dataset.encode_z_actions_window(
                    slot.zarr_path, slot.n_latent_frames, lat_s, lat_e,
                ).numpy()

                dev = slot_device(s)
                vae = vaes[dev]

                chunk = ZarrRideDataset.load_latent_chunk(
                    slot.zarr_path, lat_s, lat_e,
                )
                lat_dev = chunk.unsqueeze(0).to(
                    device=dev, dtype=torch.float16,
                )

                logging.info(
                    "  slot=%d  win=%d  latents=%d-%d  %s  [%s]",
                    s, win_idx, lat_s, lat_e, tuple(lat_dev.shape), dev,
                )

                with torch.no_grad():
                    pixels = vae.decode_to_pixel(lat_dev, use_cache=True)
                pixels = pixels[0]
                batch_frames = (
                    (pixels.clamp(-1, 1) + 1) / 2 * 255
                ).to(torch.uint8).cpu()
                batch_frames = batch_frames.permute(0, 2, 3, 1).contiguous().numpy()
                n_vid = batch_frames.shape[0]

                del lat_dev, pixels, chunk
                if dev.type == "cuda":
                    torch.cuda.empty_cache()

                vid_cursor = per_slot_vid_cursor[s]
                annotated_frames = []
                for j in range(n_vid):
                    f = batch_frames[j].copy()
                    abs_vid_j = vid_cursor + j
                    lat_j_abs = min(abs_vid_j // _LATENT_TO_VIDEO, lat_e - 1)
                    lat_j_win = lat_j_abs - lat_s
                    lat_j_win = max(0, min(lat_j_win, z_act_window.shape[0] - 1))
                    motion = per_slot_motion[s]
                    if motion is not None and abs_vid_j < motion.shape[0]:
                        f = draw_motion_overlay(f, motion[abs_vid_j])
                    f = draw_frame_info(
                        f, s, win_idx, j, lat_j_abs, lat_s, zarr_name,
                    )
                    f = draw_latent_overlay(f, z_act_window[lat_j_win])
                    f = draw_latent_dial(f, z_act_window[lat_j_win])
                    annotated_frames.append(f)
                annotated = np.stack(annotated_frames)
                per_slot_vid_cursor[s] += n_vid

                if args.combined:
                    per_slot_combined[s].append(annotated)
                else:
                    stem = Path(slot.zarr_path).stem
                    out_path = (
                        out_dir
                        / f"g{group_counter:02d}_s{s}_win{win_idx:02d}"
                          f"_{stem}_lat{lat_s}-{lat_e}.mp4"
                    )
                    rc = _save_mp4(annotated, out_path, fps)
                    if rc == 0:
                        logging.info(
                            "    Saved %s  (%d frames, %.1f s @ %.0f fps)",
                            out_path, n_vid, n_vid / fps, fps,
                        )
                        saved_count += 1
                    else:
                        logging.error("    ffmpeg failed with exit code %d", rc)

                del batch_frames, annotated_frames, annotated

            batcher.advance()

        # Write combined videos for each slot if requested
        if args.combined:
            for s in range(args.batch_size):
                _flush_combined_slot(s, batcher.get_slot_info()[s])

        for v in vaes.values():
            v.model.clear_cache()
        logging.info("  Group %d done.", group_counter)

    logging.info("Done. %d clip(s) written to %s", saved_count, out_dir)


if __name__ == "__main__":
    main()
