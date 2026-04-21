#!/usr/bin/env python3
"""Visualise secant decomposition: decode latent bundles to pixels and render
ground truth, actual prediction, counterfactual prediction, and noise heatmaps.

Reads the .pt tensor bundles saved by:
  noise_analysis_backbone.py --mode secant --save_vis N

For each bundle, produces a PNG grid with:
  Row 1: GT chunk-0 frames | Actual pred chunk-0 | Counterfactual pred chunk-0
  Row 2: |e| total error   | |e_act| action err  | |e_noise| rollout noise
  Row 3: Sensitivity map S | e_noise overlaid on GT

Requires GPU + VAE for latent decode. Does NOT need the full diffusion model.

Usage:
  python vis_noise_secant.py --bundle_dir vis/noise_analysis/vis_bundles --device cuda:0
  python vis_noise_secant.py --bundle_dir vis/noise_analysis/vis_bundles --pick 3  # first 3 only
"""

import sys
import os
import argparse
import logging
import subprocess
import numpy as np
import torch
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

NUM_FRAME_PER_BLOCK = 3
NUM_FRAMES = 21
OUT_DIR = "vis/noise_analysis/renders"


# -------------------------------------------------------------------
# VAE decode
# -------------------------------------------------------------------

def build_vae(device):
    """Load just the Wan VAE (no diffusion model needed)."""
    os.environ.setdefault("HF_HOME", "/scratch/u6ex/as1748.u6ex/frodobots/hf_cache")
    from utils.wan_wrapper import WanVAEWrapper
    vae = WanVAEWrapper()
    vae.to(device).eval()
    return vae


def decode_latents(vae, latents, device):
    """Decode [F, C, H, W] latents to [T_video, H, W, 3] uint8 numpy.

    Prepends a dummy frame (Wan VAE requirement), decodes, strips the dummy
    from pixel output.
    """
    lat = latents.unsqueeze(0).to(device, dtype=torch.float32)  # [1, F, C, H, W]
    dummy = lat[:, 0:1]
    lat_wd = torch.cat([dummy, lat], dim=1)
    with torch.no_grad():
        px = vae.decode_to_pixel(lat_wd.float())[:, 1:, ...]
    vid = (0.5 * (px.float() + 1.0)).clamp(0, 1)
    vid_np = (vid[0].cpu().numpy() * 255).astype(np.uint8)
    if vid_np.shape[-1] != 3:
        vid_np = vid_np.transpose(0, 2, 3, 1)
    return vid_np


# -------------------------------------------------------------------
# Heatmap helpers
# -------------------------------------------------------------------

def latent_to_spatial_heatmap(tensor_4d, reduce="rms"):
    """Collapse [F, C, H, W] latent tensor to [F, H, W] spatial heatmap.

    reduce: 'rms' = root mean square over channels, 'mean' = mean abs.
    """
    t = tensor_4d.float()
    if reduce == "rms":
        return t.pow(2).mean(dim=1).sqrt()  # [F, H, W]
    else:
        return t.abs().mean(dim=1)


def normalise_heatmap(heatmap, percentile=99):
    """Normalise to [0, 1] clipping at given percentile."""
    vmax = np.percentile(heatmap, percentile)
    if vmax < 1e-8:
        return np.zeros_like(heatmap)
    return np.clip(heatmap / vmax, 0, 1)


def apply_colormap(heatmap_2d, cmap="inferno"):
    """Convert [H, W] float in [0,1] to [H, W, 3] uint8 via matplotlib colormap."""
    import matplotlib.cm as cm
    mapper = cm.get_cmap(cmap)
    rgba = mapper(heatmap_2d)  # [H, W, 4]
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def overlay_heatmap_on_frame(frame_rgb, heatmap_2d, alpha=0.5, cmap="inferno"):
    """Overlay a [H, W] heatmap on a [H, W, 3] uint8 frame."""
    import cv2
    hm_rgb = apply_colormap(heatmap_2d, cmap)
    # Resize heatmap to frame size (latent is 60x104, pixels are larger)
    if hm_rgb.shape[:2] != frame_rgb.shape[:2]:
        hm_rgb = cv2.resize(hm_rgb, (frame_rgb.shape[1], frame_rgb.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
    blended = (frame_rgb.astype(np.float32) * (1 - alpha) +
               hm_rgb.astype(np.float32) * alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def add_text(img, text, position=(10, 25), scale=0.6, color=(255, 255, 255)):
    """Put white text with dark outline on image."""
    import cv2
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)
    return img


# -------------------------------------------------------------------
# Rendering
# -------------------------------------------------------------------

def pick_representative_frame(n_pixel_frames):
    """Pick 3 evenly-spaced frame indices from the decoded pixel frames."""
    if n_pixel_frames <= 3:
        return list(range(n_pixel_frames))
    step = n_pixel_frames // 3
    return [0, step, 2 * step]


def render_bundle(bundle, vae, device, out_path):
    """Render one vis bundle as a PNG grid.

    Grid layout (each cell is one pixel frame, we pick 3 from chunk 0):
      Row 1: GT | Actual pred | Counterfactual pred
      Row 2: |e| heatmap on GT | |e_act| on GT | |e_noise| on GT
      Row 3: Sensitivity S on GT | e_noise standalone heatmap | (info panel)
    """
    import cv2

    ts = bundle["ts"]
    city = bundle["city"]
    widx = bundle["window_idx"]
    act_frac = bundle["act_fraction"]
    alpha_val = bundle["alpha"]
    z2 = bundle["z2_chunk0"]
    z7 = bundle["z7_chunk0"]

    # Decode chunk-0 latents (first 3 frames) for GT, actual, counterfactual
    L_gt = bundle["L_gt"]
    L_a = bundle["L_actual"]
    L_c = bundle["L_counterfactual"]

    if L_gt is None:
        log.warning("Skipping %s w%d: no GT latents", ts, widx)
        return

    gt_c0 = L_gt[:NUM_FRAME_PER_BLOCK]
    la_c0 = L_a[:NUM_FRAME_PER_BLOCK]
    lc_c0 = L_c[:NUM_FRAME_PER_BLOCK]

    log.info("Decoding %s w%04d [%s]...", ts, widx, city)
    frames_gt = decode_latents(vae, gt_c0, device)
    frames_a = decode_latents(vae, la_c0, device)
    frames_c = decode_latents(vae, lc_c0, device)

    # Pick representative frames
    n_pix = frames_gt.shape[0]
    pick = pick_representative_frame(n_pix)
    # Use middle frame for heatmap overlays
    mid = pick[len(pick) // 2]
    gt_frame = frames_gt[mid]
    a_frame = frames_a[mid]
    c_frame = frames_c[mid]

    H, W = gt_frame.shape[:2]

    # Compute spatial heatmaps from decomposition tensors (latent space)
    e_chunk0 = bundle["e_chunk0"].float()
    e_act = bundle["e_act_chunk0"].float()
    e_noise = bundle["e_noise_chunk0"].float()
    d_raw = bundle["d_raw_chunk0"].float()
    S_normed = bundle.get("S_normed", None)

    # Collapse to spatial [F, H_lat, W_lat], take middle frame
    mid_lat = NUM_FRAME_PER_BLOCK // 2
    hm_e = latent_to_spatial_heatmap(e_chunk0)[mid_lat].numpy()
    hm_e_act = latent_to_spatial_heatmap(e_act)[mid_lat].numpy()
    hm_e_noise = latent_to_spatial_heatmap(e_noise)[mid_lat].numpy()
    hm_d_raw = latent_to_spatial_heatmap(d_raw)[mid_lat].numpy()

    # Normalise heatmaps (shared scale for e, e_act, e_noise so they're comparable)
    shared_max = max(np.percentile(hm_e, 99), 1e-8)
    hm_e_n = np.clip(hm_e / shared_max, 0, 1)
    hm_e_act_n = np.clip(hm_e_act / shared_max, 0, 1)
    hm_e_noise_n = np.clip(hm_e_noise / shared_max, 0, 1)
    hm_d_raw_n = normalise_heatmap(hm_d_raw)

    # Sensitivity map
    if S_normed is not None:
        S_spatial = S_normed[:NUM_FRAME_PER_BLOCK].float().mean(dim=1)  # [3, 60, 104]
        hm_S = S_spatial[mid_lat].numpy()
        hm_S_n = normalise_heatmap(hm_S)
    else:
        hm_S_n = np.zeros((60, 104), dtype=np.float32)

    # Build overlays on GT frame
    ov_e = overlay_heatmap_on_frame(gt_frame, hm_e_n, alpha=0.55, cmap="hot")
    ov_e_act = overlay_heatmap_on_frame(gt_frame, hm_e_act_n, alpha=0.55, cmap="cool")
    ov_e_noise = overlay_heatmap_on_frame(gt_frame, hm_e_noise_n, alpha=0.55, cmap="inferno")
    ov_S = overlay_heatmap_on_frame(gt_frame, hm_S_n, alpha=0.55, cmap="viridis")

    # Standalone e_noise heatmap (no overlay)
    e_noise_standalone = apply_colormap(hm_e_noise_n, cmap="inferno")
    e_noise_standalone = cv2.resize(e_noise_standalone, (W, H), interpolation=cv2.INTER_LINEAR)

    # Standalone counterfactual diff heatmap
    d_raw_standalone = apply_colormap(hm_d_raw_n, cmap="magma")
    d_raw_standalone = cv2.resize(d_raw_standalone, (W, H), interpolation=cv2.INTER_LINEAR)

    # Add labels
    add_text(gt_frame, "GT (ground truth)")
    add_text(a_frame, "Actual pred (v12)")
    add_text(c_frame, "Counterfactual pred")

    add_text(ov_e, "|e| total error", color=(255, 200, 200))
    add_text(ov_e_act, "|e_act| action error", color=(200, 200, 255))
    add_text(ov_e_noise, "|e_noise| rollout noise", color=(255, 200, 100))

    add_text(ov_S, "Sensitivity map S", color=(200, 255, 200))
    add_text(e_noise_standalone, "e_noise heatmap", color=(255, 200, 100))
    add_text(d_raw_standalone, "|L_cf - L_a| cf diff", color=(255, 200, 255))

    # Info panel
    info = np.zeros((H, W, 3), dtype=np.uint8)
    lines = [
        f"Ride: {ts}",
        f"City: {city}  Window: {widx}",
        f"z2={z2:.3f}  z7={z7:.3f}",
        f"act_fraction: {act_frac:.4f}",
        f"alpha: {alpha_val:.4f}",
        f"gamma: {bundle['gamma']:.2f}",
        "",
        "Row 1: GT | Actual | CF",
        "Row 2: |e| | |e_act| | |e_noise|",
        "Row 3: Sens. | e_noise | CF diff",
    ]
    for li, line in enumerate(lines):
        add_text(info, line, position=(10, 25 + li * 22), scale=0.45)

    # Assemble grid: 3 rows x 3 columns
    row1 = np.concatenate([gt_frame, a_frame, c_frame], axis=1)
    row2 = np.concatenate([ov_e, ov_e_act, ov_e_noise], axis=1)
    row3 = np.concatenate([ov_S, e_noise_standalone, d_raw_standalone], axis=1)
    grid = np.concatenate([row1, row2, row3], axis=0)

    # Save
    cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    log.info("  Saved: %s (%dx%d)", out_path, grid.shape[1], grid.shape[0])


def render_bundle_video(bundle, vae, device, out_path):
    """Render full 21-frame actual vs GT as side-by-side MP4 with e_noise overlay.

    Layout: [GT | Actual pred | e_noise overlay on GT]
    All 21 frames decoded.
    """
    import cv2

    L_gt = bundle["L_gt"]
    L_a = bundle["L_actual"]
    if L_gt is None:
        return

    log.info("Decoding full 21-frame window for video...")
    frames_gt = decode_latents(vae, L_gt, device)
    frames_a = decode_latents(vae, L_a, device)

    # Compute per-frame error heatmap in latent space, interpolate to pixel size
    e_full = (L_a.float() - L_gt.float())
    hm_full = latent_to_spatial_heatmap(e_full)  # [21, 60, 104]

    n_pix = frames_gt.shape[0]
    n_lat = hm_full.shape[0]
    H, W = frames_gt.shape[1], frames_gt.shape[2]

    # Determine shared heatmap scale
    hm_np = hm_full.numpy()
    vmax = max(np.percentile(hm_np, 99), 1e-8)

    out_frames = []
    for fi in range(n_pix):
        gt_f = frames_gt[fi]
        a_f = frames_a[fi]

        # Map pixel frame to latent frame
        lat_idx = min(fi * n_lat // n_pix, n_lat - 1)
        hm = np.clip(hm_np[lat_idx] / vmax, 0, 1)
        ov = overlay_heatmap_on_frame(gt_f, hm, alpha=0.5, cmap="inferno")

        add_text(gt_f, "GT", position=(5, 20), scale=0.5)
        add_text(a_f, "v12 pred", position=(5, 20), scale=0.5)
        add_text(ov, "|error| on GT", position=(5, 20), scale=0.5)

        row = np.concatenate([gt_f, a_f, ov], axis=1)
        out_frames.append(row)

    out_frames = np.stack(out_frames)
    frames_to_mp4(out_frames, out_path, fps=5.0)
    log.info("  Saved video: %s (%d frames)", out_path, len(out_frames))


def frames_to_mp4(frames, path, fps=5.0):
    h, w = frames.shape[1], frames.shape[2]
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.communicate(input=frames.tobytes(), timeout=120)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Render secant decomposition vis bundles to pixel grids + videos")
    parser.add_argument("--bundle_dir", default="vis/noise_analysis/vis_bundles",
                        help="Directory containing .pt vis bundles from noise_analysis_backbone.py")
    parser.add_argument("--pick", type=int, default=None,
                        help="Only render first N bundles")
    parser.add_argument("--video", action="store_true",
                        help="Also render full 21-frame side-by-side videos (slower)")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir)
    if not bundle_dir.exists():
        log.error("Bundle dir not found: %s", bundle_dir)
        log.error("Run: noise_analysis_backbone.py --mode secant --save_vis N")
        return

    bundles = sorted(bundle_dir.glob("*.pt"))
    if not bundles:
        log.error("No .pt bundles in %s", bundle_dir)
        return

    if args.pick:
        bundles = bundles[:args.pick]

    log.info("Found %d bundles in %s", len(bundles), bundle_dir)

    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device(args.device)

    log.info("Loading VAE...")
    vae = build_vae(device)
    log.info("VAE ready.")

    for bp in bundles:
        bundle = torch.load(bp, map_location="cpu", weights_only=False)
        stem = bp.stem  # e.g. 20240316144555_w0003

        # PNG grid
        png_path = os.path.join(OUT_DIR, f"{stem}_decomp.png")
        render_bundle(bundle, vae, device, png_path)

        # Optional video
        if args.video:
            mp4_path = os.path.join(OUT_DIR, f"{stem}_sidebyside.mp4")
            render_bundle_video(bundle, vae, device, mp4_path)

    log.info("Done! Renders saved to %s", OUT_DIR)


if __name__ == "__main__":
    main()
