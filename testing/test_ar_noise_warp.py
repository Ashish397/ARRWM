"""
Proof-of-concept test for utils/ar_noise_warp.py.

Pick an existing pair of clean_x_fake.mp4 (student rollout) and
clean_x_real.mp4 (GT) from a training run's samples directory. Encode
both to WanVAE latent space, time-warp GT to align with rollout via
DTW on per-frame latent L2, decode the warped GT for visualisation,
and write a side-by-side mp4 + a warp-path plot + a JSON dump of
residual statistics so the warp can be eyeballed before we commit to
plugging it into teacher training.

Usage:
    cd /scratch/u6ex/as1748.u6ex/ARRWM
    conda activate arrwm
    python testing/test_ar_noise_warp.py \\
        --rollout logs/.../samples/step_0000091_clean_x_fake.mp4 \\
        --gt      logs/.../samples/step_0000091_clean_x_real.mp4 \\
        --out_dir testing/outputs/ar_noise_warp_step91 \\
        --max_skew 2

If --rollout/--gt are omitted, defaults point at v7 step 91.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.ar_noise_warp import (
    compute_pairwise_cost,
    dtw_monotone,
    residual_stats,
    time_warp_align,
)
from utils.wan_wrapper import WanVAEWrapper


_DEFAULT_SAMPLES = (
    PROJECT_ROOT
    / "logs/action_forcing_phase3_online_teacher_NR_G_v7/"
    "action_forcing_phase3_online_teacher_NR_G_v7_j4585669/samples"
)
_DEFAULT_ROLLOUT = _DEFAULT_SAMPLES / "step_0000091_clean_x_fake.mp4"
_DEFAULT_GT = _DEFAULT_SAMPLES / "step_0000091_clean_x_real.mp4"


# --------------------------------------------------------------------------- #
# Video I/O
# --------------------------------------------------------------------------- #


def load_mp4_to_tensor(path: Path) -> torch.Tensor:
    """Read an mp4 to a [T, C=3, H, W] tensor in [-1, 1], fp32, CPU."""
    if not path.is_file():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # cv2 returns BGR; convert to RGB.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    arr = np.stack(frames, axis=0).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()  # [T, C, H, W]


def pixel_to_uint8_bgr(pixel: torch.Tensor) -> np.ndarray:
    """[T, C=3, H, W] in [-1, 1] -> [T, H, W, 3] uint8 BGR (for cv2)."""
    x = pixel.detach().float().clamp(-1.0, 1.0).cpu().numpy()
    x = (x + 1.0) * 127.5
    x = np.clip(x, 0.0, 255.0).astype(np.uint8)
    x = x.transpose(0, 2, 3, 1)  # [T, H, W, C] RGB
    x = x[..., ::-1]              # RGB -> BGR
    return np.ascontiguousarray(x)


def write_mp4(path: Path, frames_bgr: np.ndarray, fps: int = 5) -> None:
    """frames_bgr: [T, H, W, 3] uint8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    T, H, W, _ = frames_bgr.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"cv2 VideoWriter failed for {path}")
    for t in range(T):
        writer.write(frames_bgr[t])
    writer.release()


# --------------------------------------------------------------------------- #
# VAE encode / decode helpers
# --------------------------------------------------------------------------- #


def encode_latents(
    vae: WanVAEWrapper,
    pixel_tchw: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """[T, C, H, W] pixel -> [T_lat, C_lat, H/8, W/8] latent (fp32, on CPU).

    Wraps to [B=1, C, T, H, W], encodes, returns first-batch latents
    permuted to [T_lat, C_lat, h, w].
    """
    pix = pixel_tchw.to(device=device, dtype=dtype).unsqueeze(0)  # [1, T, C, H, W]
    pix = pix.permute(0, 2, 1, 3, 4).contiguous()                  # [1, C, T, H, W]
    with torch.no_grad():
        lat = vae.encode_to_latent(pix)        # [1, T_lat, C_lat, h, w]
    return lat.squeeze(0).float().cpu()


def decode_latents(
    vae: WanVAEWrapper,
    lat_tchw: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """[T_lat, C_lat, h, w] latent -> [T_px, 3, H, W] pixel in [-1, 1] (CPU)."""
    lat = lat_tchw.to(device=device, dtype=dtype).unsqueeze(0)  # [1, T_lat, C, h, w]
    with torch.no_grad():
        pix = vae.decode_to_pixel(lat)                          # [1, T_px, 3, H, W]
    return pix.squeeze(0).float().cpu()


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #


def make_side_by_side(
    rollout_px: torch.Tensor,
    gt_orig_px: torch.Tensor,
    warped_gt_px: torch.Tensor,
    residual_px: torch.Tensor,
    labels: Tuple[str, str, str, str] = (
        "student rollout",
        "GT (orig)",
        "GT (warped)",
        "residual",
    ),
) -> np.ndarray:
    """Concatenate four pixel streams horizontally with text labels.

    Each input is [T, 3, H, W] in [-1, 1]. Residual is rescaled to be
    visible: it lives in [-2, 2] worst-case, so we map +/-1 mid-grey
    via residual / 2 -> [-0.5, 0.5] + 0.0 visualisation centred.
    Returns [T, H, W*4, 3] uint8 BGR with labels burned in.
    """
    T = min(rollout_px.shape[0], gt_orig_px.shape[0], warped_gt_px.shape[0])
    rollout = pixel_to_uint8_bgr(rollout_px[:T])
    gt_orig = pixel_to_uint8_bgr(gt_orig_px[:T])
    warped = pixel_to_uint8_bgr(warped_gt_px[:T])
    resid = pixel_to_uint8_bgr(residual_px[:T].clamp(-1.0, 1.0))

    H, W = rollout.shape[1:3]
    panel = np.concatenate([rollout, gt_orig, warped, resid], axis=2)  # [T, H, 4W, 3]

    # Burn labels into top-left of each panel.
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thick = 2
    for t in range(T):
        frame = panel[t]
        for k, lbl in enumerate(labels):
            x0 = k * W + 10
            y0 = 25
            cv2.putText(frame, lbl, (x0, y0), font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
            cv2.putText(frame, lbl, (x0, y0), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
        cv2.putText(frame, f"frame {t}", (10, H - 12), font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(frame, f"frame {t}", (10, H - 12), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
        panel[t] = frame
    return panel


def plot_warp_path(path: torch.Tensor, cost: torch.Tensor, out_png: Path) -> None:
    """Save a PNG showing the DTW path overlaid on the cost matrix."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = path.detach().cpu().numpy()
    c = cost.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(c, aspect="auto", origin="lower", cmap="viridis")
    ax.plot(p, np.arange(len(p)), color="red", linewidth=2, label="DTW path")
    # Identity reference (no warp) — only meaningful if T_r == T_g.
    if c.shape[0] == c.shape[1]:
        ax.plot(np.arange(c.shape[0]), np.arange(c.shape[0]), color="white",
                linestyle="--", linewidth=1, alpha=0.6, label="identity")
    ax.set_xlabel("GT frame index")
    ax.set_ylabel("rollout frame index")
    ax.set_title("DTW cost (L2 in latent space) + warp path")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.legend(loc="lower right")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Unit checks (cheap sanity tests on the algorithm itself)
# --------------------------------------------------------------------------- #


def _sanity_identity_warp():
    """Same sequence on both sides -> identity warp."""
    torch.manual_seed(0)
    x = torch.randn(8, 4, 3, 3)
    warped, path, residual, _ = time_warp_align(x, x, max_skew=2)
    assert torch.equal(path, torch.arange(8)), f"expected identity path, got {path.tolist()}"
    assert residual.abs().max().item() == 0.0
    assert torch.equal(warped, x)


def _sanity_shifted_warp():
    """Rollout = gt rolled forward by 1 (rollout is 1 step ahead) -> path should reflect shift."""
    torch.manual_seed(0)
    gt = torch.randn(10, 4)
    rollout = gt.roll(shifts=-1, dims=0)  # rollout[i] == gt[i+1]
    rollout[-1] = gt[-1]                  # last frame: keep gt[-1] so a valid alignment exists
    warped, path, residual, _ = time_warp_align(rollout, gt, max_skew=2)
    # First few rollout frames should map to gt index = i + 1.
    expected_prefix = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert torch.equal(path[: len(expected_prefix)], expected_prefix), (
        f"shifted alignment failed: got {path.tolist()}"
    )


def _sanity_max_skew_zero_holds():
    """max_skew=0 forbids forward progress: path must be constant."""
    torch.manual_seed(1)
    rollout = torch.randn(6, 3)
    gt = torch.randn(6, 3)
    _, path, _, _ = time_warp_align(rollout, gt, max_skew=0)
    assert torch.all(path == path[0]), f"max_skew=0 should be constant, got {path.tolist()}"


def _sanity_max_skew_one_allows_identity():
    """max_skew=1 with rollout==gt should pick the diagonal."""
    torch.manual_seed(2)
    x = torch.randn(7, 5)
    _, path, residual, _ = time_warp_align(x, x, max_skew=1)
    assert torch.equal(path, torch.arange(7)), f"got {path.tolist()}"
    assert residual.abs().max().item() == 0.0


def run_sanity_checks() -> None:
    _sanity_identity_warp()
    _sanity_shifted_warp()
    _sanity_max_skew_zero_holds()
    _sanity_max_skew_one_allows_identity()
    print("[ar_noise_warp] sanity checks PASSED")


# --------------------------------------------------------------------------- #
# Main proof-of-concept
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", type=Path, default=_DEFAULT_ROLLOUT,
                    help="path to student rollout mp4 (clean_x_fake)")
    ap.add_argument("--gt", type=Path, default=_DEFAULT_GT,
                    help="path to GT mp4 (clean_x_real)")
    ap.add_argument("--out_dir", type=Path,
                    default=PROJECT_ROOT / "testing/outputs/ar_noise_warp",
                    help="directory to write visualisations + stats")
    ap.add_argument("--max_skew", type=int, default=2,
                    help="DTW forward-skew bound (0=rigid, 2=2x speed)")
    ap.add_argument("--metric", type=str, default="l2", choices=["l2", "cos"],
                    help="per-frame distance metric for DTW cost")
    ap.add_argument("--device", type=str, default="cuda",
                    help="device for VAE forward (use 'cpu' if no GPU)")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"],
                    help="VAE dtype")
    ap.add_argument("--fps", type=int, default=5,
                    help="output mp4 fps")
    ap.add_argument("--skip_sanity", action="store_true",
                    help="skip the algorithmic sanity checks")
    args = ap.parse_args()

    if not args.skip_sanity:
        run_sanity_checks()

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    print(f"[load] rollout = {args.rollout}")
    print(f"[load]      gt = {args.gt}")
    rollout_px = load_mp4_to_tensor(args.rollout)
    gt_px = load_mp4_to_tensor(args.gt)
    print(f"[load] rollout pixel shape {tuple(rollout_px.shape)}, "
          f"gt pixel shape {tuple(gt_px.shape)}")

    print(f"[vae ] loading WanVAE on {device} ({dtype})")
    vae = WanVAEWrapper().to(device=device, dtype=dtype)
    vae.eval()

    print("[vae ] encoding rollout to latent ...")
    rollout_lat = encode_latents(vae, rollout_px, device, dtype)
    print("[vae ] encoding gt to latent ...")
    gt_lat = encode_latents(vae, gt_px, device, dtype)
    print(f"[vae ] rollout latent {tuple(rollout_lat.shape)}, "
          f"gt latent {tuple(gt_lat.shape)}")

    print(f"[warp] running DTW (max_skew={args.max_skew}, metric={args.metric}) "
          "in latent space ...")
    warped_gt_lat, path, residual_lat, cost = time_warp_align(
        rollout_lat, gt_lat, max_skew=args.max_skew, metric=args.metric,
    )
    print(f"[warp] path (rollout idx -> gt idx): {path.tolist()}")
    skew = (path[1:] - path[:-1]).tolist()
    print(f"[warp] step deltas: {skew}")

    print("[stat] computing residual stats (latent) ...")
    lat_stats = residual_stats(residual_lat)
    print(f"[stat] latent residual overall L2 norm: {lat_stats['overall_l2_norm']:.4f}")
    print(f"[stat] latent residual overall std:     {lat_stats['overall_std']:.4f}")

    # For visualisation we need pixels. Decode the warped GT and the
    # rollout/orig-gt pixel-residual is the model-free sanity check.
    print("[vae ] decoding warped gt latents back to pixels ...")
    warped_gt_px = decode_latents(vae, warped_gt_lat, device, dtype)

    # Make the four panels comparable in T. Decoding may yield a
    # slightly different frame count than the original mp4 (VAE
    # temporal compression). Use the minimum.
    T_min = min(rollout_px.shape[0], gt_px.shape[0], warped_gt_px.shape[0])
    rollout_px = rollout_px[:T_min]
    gt_px = gt_px[:T_min]
    warped_gt_px = warped_gt_px[:T_min]
    residual_px = rollout_px - warped_gt_px

    pix_stats = residual_stats(residual_px)
    print(f"[stat] pixel residual overall L2 norm:  {pix_stats['overall_l2_norm']:.4f}")
    print(f"[stat] pixel residual overall std:      {pix_stats['overall_std']:.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[viz ] building side-by-side mp4 ...")
    panel = make_side_by_side(rollout_px, gt_px, warped_gt_px, residual_px)
    panel_path = args.out_dir / "compare.mp4"
    write_mp4(panel_path, panel, fps=args.fps)
    print(f"[viz ] wrote {panel_path}")

    print("[viz ] plotting cost matrix + warp path ...")
    plot_path = args.out_dir / "warp_path.png"
    plot_warp_path(path, cost, plot_path)
    print(f"[viz ] wrote {plot_path}")

    # Dump everything quantitative for off-line review.
    dump = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "rollout_pixel_shape": list(rollout_px.shape),
        "gt_pixel_shape": list(gt_px.shape),
        "rollout_latent_shape": list(rollout_lat.shape),
        "gt_latent_shape": list(gt_lat.shape),
        "path": path.tolist(),
        "step_deltas": skew,
        "latent_residual_stats": lat_stats,
        "pixel_residual_stats": pix_stats,
    }
    dump_path = args.out_dir / "stats.json"
    with dump_path.open("w") as f:
        json.dump(dump, f, indent=2)
    print(f"[stat] wrote {dump_path}")

    print("[done] open compare.mp4 to verify the warp visually.")


if __name__ == "__main__":
    main()
