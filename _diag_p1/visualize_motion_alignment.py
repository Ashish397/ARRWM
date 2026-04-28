"""Visualise motion ↔ latent alignment using the v14 overlay style:
per-pixel-frame CoTracker grid arrows (raw motion.npy) plus an
ss_vae latent dial in the corner.

Standalone — no DDP, no training. Run on the local 5090 to visually
verify that the dataset's chunk-grain motion loader produces motion
fields that semantically match the GT video at the same ride positions.

What it does:
  1. Loads one ride from /home/ashish/frodobots/frodobots_encoded.
  2. Pulls the head-dropped, motion-capped latent slice
     [start : start + n_latents] from the zarr.
  3. Loads the raw motion.npy and slices the chunks that cover the
     same window. Each chunk = mean displacement for one 12-pixel-frame
     window = one [N=100, 3] tensor of (dx, dy, visibility) values.
  4. Encodes z_actions for the same slice via the dataset's chunk-grain
     ``encode_z_actions_window`` (broadcast within chunks) and slices
     to ``action_dims=[2, 7]`` for the latent dial.
  5. VAE-decodes the latents to pixel video (Wan VAE).
  6. Overlays the v14 motion grid (10×10 green arrows showing per-pixel
     displacement; one arrow per grid point, length ∝ |dx, dy|) AND a
     small bottom-right dial (`z[2]` left/right, `z[7]` up/down).
  7. Encodes mp4. Writes to ``_diag_p1/out/<...>.mp4``.

Usage:
  cd /home/ashish/ARRWM
  python _diag_p1/visualize_motion_alignment.py --start 0 --n-latents 60

The arrows are constant for 12 consecutive video frames (= one chunk),
then jump. They should track the visible motion in the GT pixels —
forward translation shows arrows pointing radially outward (= scene
flowing past the camera), turns show arrows tilting left or right.
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

import cv2
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


def draw_motion_overlay(frame: np.ndarray, motion_vecs: np.ndarray, mag_scale: float = 30.0) -> np.ndarray:
    """v14-style motion grid arrows. Ported from utils/test_zarr_chunks.py.

    motion_vecs: ``[N, 3]`` where N = grid_size**2, columns = (dx, dy, vis).
    Draws one arrow per grid point on top of ``frame`` (modified in place
    via cv2.arrowedLine). Color: green for vis>=0.5, orange for
    0.2 <= vis < 0.5, skipped if vis < 0.2.
    """
    h, w = frame.shape[:2]
    if motion_vecs.ndim != 2 or motion_vecs.shape[1] != 3:
        return frame
    N = motion_vecs.shape[0]
    grid_size = int(round(np.sqrt(N)))
    if grid_size * grid_size != N or grid_size <= 0:
        return frame

    for gy in range(grid_size):
        for gx in range(grid_size):
            idx = gy * grid_size + gx
            dx, dy, vis = motion_vecs[idx]
            if vis < 0.2:
                continue
            cx = int((gx + 0.5) * w / grid_size)
            cy = int((gy + 0.5) * h / grid_size)
            # Negate to point toward where pixels are FROM
            # (matches v14's "flow toward camera" convention; the
            # CoTracker delta is forward-frame displacement, so negating
            # gives the visual flow direction).
            end_x = int(cx - dx * mag_scale)
            end_y = int(cy - dy * mag_scale)
            color = (0, 255, 0) if vis >= 0.5 else (0, 200, 255)
            cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), color, 1, tipLength=0.3)
    return frame


def draw_latent_dial(
    frame: np.ndarray, z_action: np.ndarray,
    label: str = "z[2,7]",
) -> np.ndarray:
    """v14-style dial for the ss_vae action latent: a circle in the
    bottom-right corner with a single arrow whose horizontal component
    encodes z[0] (turn) and vertical component encodes z[1] (forward).

    z_action: ``[2,]`` already sliced to action_dims.
    """
    h, w = frame.shape[:2]
    bar_h = 60
    cx, cy = w - 50, h - bar_h + 30
    cv2.circle(frame, (cx, cy), 22, (60, 60, 80), -1)
    cv2.circle(frame, (cx, cy), 22, (180, 180, 255), 1)

    z_turn = float(z_action[0])
    z_fwd = float(z_action[1])

    # z is post-tanh-squash, so values are roughly in [-1, 1]. Scale
    # arrow length to the dial radius.
    arrow_len = 18
    dx = int(np.clip(z_turn * arrow_len * 2.0, -arrow_len, arrow_len))
    dy = int(np.clip(-z_fwd * arrow_len * 2.0, -arrow_len, arrow_len))
    color = (255, 200, 100) if z_fwd >= 0 else (100, 150, 255)
    cv2.arrowedLine(frame, (cx, cy), (cx + dx, cy + dy), color, 2, tipLength=0.35)
    cv2.putText(
        frame, label, (cx - 22, cy - 26),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 255), 1, cv2.LINE_AA,
    )
    return frame


def draw_chunk_label(
    frame: np.ndarray, chunk_idx: int, ride_frame_lo: int, ride_frame_hi: int,
    z_action: np.ndarray,
) -> np.ndarray:
    """Bottom-left text overlay: chunk index, ride video-frame range,
    and the ss_vae z values for the active chunk.
    """
    h, w = frame.shape[:2]
    bar_h = 40
    overlay = frame[h - bar_h:h, :, :].astype(np.float32)
    frame[h - bar_h:h, :, :] = (overlay * 0.5).astype(np.uint8)
    y0 = h - bar_h + 16
    cv2.putText(
        frame,
        f"chunk {chunk_idx}  vid_frames [{ride_frame_lo}:{ride_frame_hi})",
        (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"z[2]={z_action[0]:+.3f}   z[7]={z_action[1]:+.3f}",
        (10, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1, cv2.LINE_AA,
    )
    return frame


def _frames_to_mp4(frames: np.ndarray, out_path: Path, fps: float) -> None:
    """Encode [T, H, W, 3] uint8 RGB to mp4 via ffmpeg pipe.

    NOTE: cv2 returns BGR for its drawing primitives; we keep the array
    in BGR throughout the overlay path and let ffmpeg interpret it as
    rgb24 — the resulting mp4 has R/B swapped vs source. To get correct
    colors, we swap R↔B BEFORE the ffmpeg pipe.
    """
    T, H, W, C = frames.shape
    assert C == 3, f"expected 3-channel frames, got C={C}"
    # cv2 drew on a BGR-interpreted array (since cv2's arrowedLine draws
    # color tuples as B,G,R). Our source from VAE is RGB. We choose to
    # keep the cv2 calls' visual intent — reds/greens/blues — by writing
    # the array as-is (= treating it as BGR for ffmpeg too). To do that:
    # tell ffmpeg the input is bgr24.
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
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
                        help="dataset latent index to start at (snapped to multiple of npb=3)")
    parser.add_argument("--n-latents", type=int, default=60,
                        help="number of latents to decode (snapped to multiple of npb=3)")
    parser.add_argument("--max-rides", type=int, default=4)
    parser.add_argument("--fps", type=float, default=20.0,
                        help="output video fps (frodobots native = 20)")
    parser.add_argument("--mag-scale", type=float, default=30.0,
                        help="motion arrow magnitude scale (v14 default = 30)")
    parser.add_argument("--out-dir", type=str,
                        default=str(_REPO / "_diag_p1" / "out"))
    args = parser.parse_args()

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

    # ----- Load raw motion.npy chunks for this window -----
    motion_path = (
        Path(MOTION_ROOT)
        / Path(attrs["ride_dir_2k"]).relative_to("/home/ashish/frodobots/frodobots_data")
        / "motion.npy"
    )
    if not motion_path.exists():
        log.error("motion.npy missing at %s", motion_path)
        sys.exit(1)
    raw_motion = np.load(motion_path)   # [n_motion_chunks, N=100, 3]
    log.info("raw motion.npy shape=%s", raw_motion.shape)

    # v14 alignment (utils/test_zarr_chunks.py:117): motion.npy is
    # encoded from SOURCE VIDEO frame 0 (pre_encode_motion.py drops only
    # the first frame), but latents are encoded starting at
    # ``action_start_sec`` of the source video, with a 0.8s causal lag.
    # offset_frames = (action_start_sec - 0.8) * fps. Chunk-grain rounds
    # to the nearest whole motion window.
    action_start_sec = float(attrs.get("action_start_sec", 0.0))
    fps_attr = float(attrs.get("fps", 20.0))
    motion_offset_chunks = int(round((action_start_sec - 0.8) * fps_attr / 12.0))
    log.info(
        "alignment offset: action_start_sec=%.2f fps=%.1f -> motion_offset=%d chunks (%.2fs)",
        action_start_sec, fps_attr, motion_offset_chunks,
        motion_offset_chunks * 12.0 / fps_attr,
    )

    chunk_lo = motion_offset_chunks + args.start // npb
    chunk_hi = motion_offset_chunks + (args.start + args.n_latents) // npb
    if chunk_hi > raw_motion.shape[0]:
        log.error("motion offset (%d) + window exceeds available chunks (%d)",
                  chunk_hi, raw_motion.shape[0])
        sys.exit(1)
    motion_chunks = raw_motion[chunk_lo:chunk_hi]   # [n_chunks, 100, 3]
    n_chunks = motion_chunks.shape[0]
    log.info("motion chunks for this window: [%d:%d) -> shape=%s",
             chunk_lo, chunk_hi, motion_chunks.shape)
    log.info("per-chunk |dx,dy| mean (raw motion magnitude):")
    raw_mag = np.linalg.norm(motion_chunks[:, :, :2], axis=-1).mean(axis=-1)
    for c in range(min(n_chunks, 30)):
        log.info("  chunk %3d  |motion|=%.3f", c, float(raw_mag[c]))
    if n_chunks > 30:
        log.info("  ... (%d more)", n_chunks - 30)

    # ----- Load latents (head-dropped already by load_latent_chunk) -----
    log.info("Loading latents [%d : %d) ...", args.start, args.start + args.n_latents)
    latents = ZarrRideDataset.load_latent_chunk(
        str(zpath), args.start, args.start + args.n_latents,
    )

    # ----- Encode z_actions for the dial -----
    z_window = ds.encode_z_actions_window(
        str(zpath), n_latent_frames, args.start, args.start + args.n_latents,
    ).cpu().numpy()
    z_window = z_window[..., ACTION_DIMS]   # [n_latents, 2]
    z_per_chunk = z_window[::npb]   # [n_chunks, 2]

    # ----- Build & run Wan VAE -----
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    log.info("Loading Wan VAE on %s (dtype=%s)...", device, dtype)
    vae = WanVAEWrapper().to(device=device, dtype=dtype).eval()

    lat_b = latents.unsqueeze(0).to(device=device, dtype=dtype)
    dummy = lat_b[:, 0:1]
    lat_wd = torch.cat([dummy, lat_b], dim=1)
    log.info("VAE decode input shape=%s", tuple(lat_wd.shape))
    with torch.no_grad():
        pixels = vae.decode_to_pixel(lat_wd)   # [1, F+1, 3, H, W] in [-1, 1]
    pixels = pixels[:, 1:, ...]
    log.info("decoded pixels shape=%s", tuple(pixels.shape))

    video = (0.5 * (pixels[0].float() + 1.0)).clamp(0.0, 1.0)
    vid_np = (video.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
    # Convert RGB -> BGR for cv2 drawing primitives (cv2 uses BGR by convention).
    # We'll feed the BGR array to ffmpeg as bgr24.
    vid_np = vid_np[..., ::-1].copy()   # RGB → BGR, write-safe

    T, H, W, _ = vid_np.shape
    log.info("video frames %d   resolution %dx%d   chunks %d   frames/chunk %d (npb=%d × _LATENT_TO_VIDEO=4)",
             T, W, H, n_chunks, npb * 4, npb)

    # ----- Overlay arrows + dial + label per frame -----
    frames_per_chunk = npb * 4
    log.info("rendering v14 motion overlay (grid arrows + dial)...")
    for t in range(T):
        chunk_idx = min(t // frames_per_chunk, n_chunks - 1)
        ride_lo = (chunk_lo + chunk_idx) * npb * 4
        ride_hi = ride_lo + frames_per_chunk
        draw_motion_overlay(
            vid_np[t], motion_chunks[chunk_idx], mag_scale=args.mag_scale,
        )
        draw_latent_dial(vid_np[t], z_per_chunk[chunk_idx])
        draw_chunk_label(
            vid_np[t], chunk_idx + chunk_lo,
            ride_lo, ride_hi, z_per_chunk[chunk_idx],
        )

    out_dir = Path(args.out_dir)
    out_path = (
        out_dir
        / f"motion_overlay_v14_{zpath.stem}_s{args.start:05d}_n{args.n_latents:04d}.mp4"
    )
    log.info("encoding mp4 -> %s", out_path)
    _frames_to_mp4(vid_np, out_path, fps=args.fps)
    log.info("DONE. open %s", out_path)


if __name__ == "__main__":
    main()
