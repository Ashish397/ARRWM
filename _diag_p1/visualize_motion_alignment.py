"""Visualise motion ↔ latent alignment by decoding a ride window with
the Wan VAE and overlaying the per-chunk z_actions as horizontal bars
on every frame.

Standalone — no DDP, no training, no trainer dependencies. Run on the
local 5090 to visually verify that the dataset's chunk-grain motion
loader (post-2d897aa, post-774456b) produces actions that semantically
match the GT video at the same ride positions.

What it does:
  1. Loads one ride from /home/ashish/frodobots/frodobots_encoded.
  2. Pulls the head-dropped, motion-capped latent slice
     [start : start + n_frames_latent] from the zarr.
  3. Encodes z_actions for the same slice via the dataset's
     ``encode_z_actions_window`` (chunk-grain, broadcast within chunks).
  4. Slices z_actions to ``action_dims=[2, 7]`` (linear / angular).
  5. VAE-decodes the latents to pixel video (Wan VAE).
  6. Overlays the per-chunk action bars at the bottom of every pixel
     frame: top bar = z[0], bottom bar = z[1]; red right for >= 0,
     blue left for < 0; length proportional to |z| (clamped at 1.0
     → half frame width).
  7. Encodes mp4 via ffmpeg subprocess. Writes to
     ``_diag_p1/out/motion_overlay_<ride>_s<start>.mp4``.

Usage:
  cd /home/ashish/ARRWM
  python _diag_p1/visualize_motion_alignment.py --start 0 --n-latents 60

The bars step at chunk boundaries (every 3 latents = every 12 video
frames at fps=20 = 0.6s of video). You should see the bar values
correlate with what the GT video is doing — left/right turns visible
as angular deflection, forward speed visible as linear bar length.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import zarr as zarr_lib

# Make repo importable + force the local Wan VAE path before importing
# anything that touches wan_wrapper module-level globals.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
os.environ.setdefault("WAN_MODEL_PATH", "/home/ashish/Wan2.1/")

import utils.wan_wrapper as _ww   # noqa: E402
_ww._default_wan_model_path = "/home/ashish/Wan2.1/"

from utils.zarr_dataset import (   # noqa: E402
    ZarrRideDataset,
    _LATENT_HEAD_DROP,
    _LATENTS_PER_MOTION_CHUNK,
)
from utils.wan_wrapper import WanVAEWrapper   # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("motion_viz")


ENCODED_ROOT = "/home/ashish/frodobots/frodobots_encoded"
CAPTION_ROOT = "/home/ashish/frodobots/frodobots_captions/train"
MOTION_ROOT = "/home/ashish/frodobots/frodobots_motion"
SS_VAE_CKPT = "/home/ashish/ARRWM/action_query/checkpoints/ss_vae_8free.pt"
ACTION_DIMS = [2, 7]   # matches configs/action_forcing_phase1_freeze.yaml


def _draw_action_overlay(
    vid_np: np.ndarray,
    z_per_chunk: np.ndarray,
    frames_per_latent: int,
    npb: int,
    label_action_dims: list[int],
) -> None:
    """In-place overlay. Mirrors ActionForcingDMDTrainer._draw_action_overlay
    but without the trainer dependency.

    ``vid_np``: ``[T_video, H, W, 3]`` uint8.
    ``z_per_chunk``: ``[n_chunks, A]`` post-tanh-squash z values.
    """
    if z_per_chunk.size == 0:
        return
    n_chunks, A = z_per_chunk.shape
    T, H, W, _ = vid_np.shape
    if H < 8 or W < 16 or n_chunks == 0:
        return

    max_strip = max(36, H // 4)
    gap = 3
    label_w = 60
    bar_h = max(6, (max_strip - 6 - gap * (A - 1)) // A)
    strip_h = 6 + bar_h * A + gap * (A - 1)
    cx = W // 2
    max_bar = (W // 2) - 6 - label_w

    # Black underlay (75% darken) so bars are readable on bright frames.
    vid_np[:, -strip_h:, :, :] = vid_np[:, -strip_h:, :, :] // 4

    frames_per_chunk = npb * frames_per_latent
    for t in range(T):
        chunk_idx = min(t // frames_per_chunk, n_chunks - 1)
        z = z_per_chunk[chunk_idx]
        for d in range(A):
            y0 = H - strip_h + 4 + d * (bar_h + gap)
            y1 = y0 + bar_h
            a = float(z[d])
            length = int(min(abs(a), 1.0) * max_bar)
            # Center marker (gray 1-px line).
            vid_np[t, y0:y1, cx - 1:cx + 1, :] = 200
            if length > 0:
                if a >= 0:
                    vid_np[t, y0:y1, cx:cx + length, 0] = 240
                    vid_np[t, y0:y1, cx:cx + length, 1] = 80
                    vid_np[t, y0:y1, cx:cx + length, 2] = 80
                else:
                    vid_np[t, y0:y1, cx - length:cx, 0] = 80
                    vid_np[t, y0:y1, cx - length:cx, 1] = 80
                    vid_np[t, y0:y1, cx - length:cx, 2] = 240


def _frames_to_mp4(frames: np.ndarray, out_path: Path, fps: float) -> None:
    """Encode [T, H, W, 3] uint8 to mp4 via ffmpeg pipe."""
    T, H, W, C = frames.shape
    assert C == 3, f"expected 3-channel frames, got C={C}"
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{W}x{H}", "-r", f"{fps}",
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "veryfast", "-crf", "20",
            tmp_path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        proc.stdin.write(frames.tobytes())
        proc.stdin.close()
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg returned exit code {rc}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ride-name", type=str, default=None,
                        help="zarr basename without .zarr; default = first indexed ride")
    parser.add_argument("--start", type=int, default=0,
                        help="dataset latent index to start at (must be a multiple of npb=3)")
    parser.add_argument("--n-latents", type=int, default=60,
                        help="number of latents to decode (multiple of npb=3)")
    parser.add_argument("--max-rides", type=int, default=4)
    parser.add_argument("--fps", type=float, default=20.0,
                        help="output video fps (frodobots native = 20)")
    parser.add_argument("--out-dir", type=str,
                        default=str(_REPO / "_diag_p1" / "out"))
    args = parser.parse_args()

    # Snap start + n_latents to npb=3.
    npb = _LATENTS_PER_MOTION_CHUNK
    if args.start % npb != 0:
        new_start = (args.start // npb) * npb
        log.warning("--start %d not a multiple of npb=%d; snapping to %d",
                    args.start, npb, new_start)
        args.start = new_start
    if args.n_latents % npb != 0:
        new_n = (args.n_latents // npb) * npb
        log.warning("--n-latents %d not a multiple of npb=%d; snapping to %d",
                    args.n_latents, npb, new_n)
        args.n_latents = new_n
    if args.n_latents <= 0:
        raise ValueError("n_latents must be > 0")

    log.info("Building ZarrRideDataset (max_rides=%d)...", args.max_rides)
    ds = ZarrRideDataset(
        encoded_root=ENCODED_ROOT,
        caption_root=CAPTION_ROOT,
        motion_root=MOTION_ROOT,
        ss_vae_checkpoint=SS_VAE_CKPT,
        min_ride_frames=21,
        device="cpu",
        ss_vae_device="cuda:0" if torch.cuda.is_available() else "cpu",
        max_rides=args.max_rides,
    )

    if args.ride_name:
        idx = next(
            (i for i, r in enumerate(ds._rides) if r[0].stem == args.ride_name),
            None,
        )
        if idx is None:
            log.error("ride %s not in indexed set; available: %s",
                      args.ride_name, [r[0].stem for r in ds._rides])
            sys.exit(1)
    else:
        idx = 0
    zpath, prompt_embeds, attrs, n_latent_frames = ds._rides[idx]
    log.info("ride: %s  n_latent_frames=%d", zpath.name, n_latent_frames)

    if args.start + args.n_latents > n_latent_frames:
        new_n = ((n_latent_frames - args.start) // npb) * npb
        if new_n <= 0:
            log.error("start=%d leaves no room in ride of %d latents",
                      args.start, n_latent_frames)
            sys.exit(1)
        log.warning("requested window exceeds ride; truncating n_latents %d -> %d",
                    args.n_latents, new_n)
        args.n_latents = new_n

    # ----- Load latents (head-dropped already by ZarrRideDataset.load_latent_chunk) -----
    log.info("Loading latents [%d : %d) from zarr (head-drop=%d already applied) ...",
             args.start, args.start + args.n_latents, _LATENT_HEAD_DROP)
    latents = ZarrRideDataset.load_latent_chunk(
        str(zpath), args.start, args.start + args.n_latents,
    )   # [F, 16, h, w], fp32
    log.info("latents shape=%s dtype=%s", tuple(latents.shape), latents.dtype)

    # ----- Encode z_actions for the same window -----
    z_window = ds.encode_z_actions_window(
        str(zpath), n_latent_frames, args.start, args.start + args.n_latents,
    )   # [n_latents, 8]
    z_window = z_window.cpu().numpy()
    log.info("z_actions shape=%s; slicing to action_dims=%s for visualisation",
             z_window.shape, ACTION_DIMS)
    z_window = z_window[..., ACTION_DIMS]   # [n_latents, 2]

    # Reduce to per-chunk (within-chunk identity verified by Test 2 of the loader diag).
    z_per_chunk = z_window[::npb]
    n_chunks = z_per_chunk.shape[0]
    log.info("per-chunk z: shape=%s ; min=%.3f max=%.3f",
             z_per_chunk.shape, float(z_per_chunk.min()), float(z_per_chunk.max()))
    log.info("per-chunk z (linear, angular):")
    for c in range(min(n_chunks, 30)):
        log.info("  chunk %3d  z=[% .3f, % .3f]",
                 c, float(z_per_chunk[c, 0]), float(z_per_chunk[c, 1]))
    if n_chunks > 30:
        log.info("  ... (%d more)", n_chunks - 30)

    # ----- Build & load Wan VAE -----
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    log.info("Loading Wan VAE on %s (dtype=%s)...", device, dtype)
    vae = WanVAEWrapper().to(device=device, dtype=dtype)
    vae.eval()

    # ----- Decode -----
    # Wan VAE wants [batch_size, num_frames, num_channels, height, width]
    # and decode_to_pixel returns the same layout. Add a leading dummy
    # frame so the temporal-conv padding doesn't eat our first frame.
    lat_b = latents.unsqueeze(0).to(device=device, dtype=dtype)   # [1, F, 16, h, w]
    dummy = lat_b[:, 0:1]
    lat_wd = torch.cat([dummy, lat_b], dim=1)
    log.info("VAE decode input shape=%s", tuple(lat_wd.shape))
    with torch.no_grad():
        pixels = vae.decode_to_pixel(lat_wd)   # [1, F+1, 3, H, W] in [-1, 1]
    pixels = pixels[:, 1:, ...]   # drop dummy
    log.info("decoded pixels shape=%s", tuple(pixels.shape))

    # To uint8 [T, H, W, 3]
    video = (0.5 * (pixels[0].float() + 1.0)).clamp(0.0, 1.0)
    vid_np = (video.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
    T, H, W, _ = vid_np.shape
    log.info("video frames %d   resolution %dx%d   chunks %d   frames/chunk %d",
             T, W, H, n_chunks, npb * 4)

    # ----- Overlay -----
    log.info("rendering action overlay (red right = +z, blue left = -z, length ∝ |z|)...")
    _draw_action_overlay(
        vid_np, z_per_chunk, frames_per_latent=4, npb=npb,
        label_action_dims=ACTION_DIMS,
    )

    # ----- Save mp4 -----
    out_dir = Path(args.out_dir)
    out_path = out_dir / f"motion_overlay_{zpath.stem}_s{args.start:05d}_n{args.n_latents:04d}.mp4"
    log.info("encoding mp4 -> %s", out_path)
    _frames_to_mp4(vid_np, out_path, fps=args.fps)
    log.info("DONE. open %s", out_path)


if __name__ == "__main__":
    main()
