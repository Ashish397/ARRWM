"""
Proof-of-concept for AR-noise extraction via the teacher.

The training pipeline already logs `pred_image.mp4` (the student's
noisy AR rollout) and `pred_real_lora.mp4` (the LoRA real_score
teacher's forward pass ON that rollout — i.e. the teacher's clean
estimate of the rollout). Their difference IS the AR-style noise we
want to use when training the teacher, with no DTW required.

This test:
  1. Loads `pred_image.mp4` and `pred_real_lora.mp4` from a training
     run's samples directory.
  2. VAE-encodes both to latent space (where teacher training lives).
  3. Computes ar_noise = rollout_lat - teacher_clean_lat, with
     optional unit-variance standardisation so the result is a
     drop-in for torch.randn.
  4. Loads `clean_x_real.mp4` (GT) to demonstrate the swap-in:
       x_t_ar  = alpha_t * GT + sigma_t * ar_noise
       x_t_gauss = alpha_t * GT + sigma_t * torch.randn_like(GT)
     Decodes both and writes side-by-side mp4 for visual comparison.
  5. Dumps stats (mean, std, per-channel, per-frame) of ar_noise vs
     torch.randn so we can see how far the AR distribution diverges
     from Gaussian.

Usage:
    cd /scratch/u6ex/as1748.u6ex/ARRWM
    conda activate arrwm
    python testing/test_ar_noise_from_teacher.py \\
        --rollout       logs/.../samples/step_0000091_pred_image.mp4 \\
        --teacher_clean logs/.../samples/step_0000091_pred_real_lora.mp4 \\
        --gt            logs/.../samples/step_0000091_clean_x_real.mp4 \\
        --out_dir       testing/outputs/ar_noise_teacher_step91

Defaults point at v7 step 91.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.ar_noise_warp import (
    extract_ar_noise_from_teacher,
    make_ar_noised_gt,
    residual_stats,
)
from utils.wan_wrapper import WanVAEWrapper


_DEFAULT_SAMPLES = (
    PROJECT_ROOT
    / "logs/action_forcing_phase3_online_teacher_NR_G_v7/"
    "action_forcing_phase3_online_teacher_NR_G_v7_j4585669/samples"
)
_DEFAULT_ROLLOUT = _DEFAULT_SAMPLES / "step_0000091.mp4"  # pred_image (no suffix)
_DEFAULT_TEACHER = _DEFAULT_SAMPLES / "step_0000091_pred_real_lora.mp4"
_DEFAULT_GT = _DEFAULT_SAMPLES / "step_0000091_clean_x_real.mp4"


# --------------------------------------------------------------------------- #
# Video I/O (same patterns as test_ar_noise_warp.py — small enough to inline) #
# --------------------------------------------------------------------------- #


def load_mp4_to_tensor(path: Path) -> torch.Tensor:
    """[T, 3, H, W] in [-1, 1], fp32 on CPU."""
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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    arr = np.stack(frames, axis=0).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


def pixel_to_uint8_bgr(pixel: torch.Tensor) -> np.ndarray:
    x = pixel.detach().float().clamp(-1.0, 1.0).cpu().numpy()
    x = (x + 1.0) * 127.5
    x = np.clip(x, 0.0, 255.0).astype(np.uint8)
    x = x.transpose(0, 2, 3, 1)
    x = x[..., ::-1]
    return np.ascontiguousarray(x)


def write_mp4(path: Path, frames_bgr: np.ndarray, fps: int = 5) -> None:
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
# VAE encode / decode helpers                                                 #
# --------------------------------------------------------------------------- #


def encode_latents(vae, pixel_tchw, device, dtype):
    pix = pixel_tchw.to(device=device, dtype=dtype).unsqueeze(0)  # [1, T, C, H, W]
    pix = pix.permute(0, 2, 1, 3, 4).contiguous()                  # [1, C, T, H, W]
    with torch.no_grad():
        lat = vae.encode_to_latent(pix)                            # [1, T_lat, C, h, w]
    return lat.squeeze(0).float().cpu()


def decode_latents(vae, lat_tchw, device, dtype):
    lat = lat_tchw.to(device=device, dtype=dtype).unsqueeze(0)
    with torch.no_grad():
        pix = vae.decode_to_pixel(lat)
    return pix.squeeze(0).float().cpu()


# --------------------------------------------------------------------------- #
# Sanity checks on the extraction algorithm (no GPU needed)                   #
# --------------------------------------------------------------------------- #


def _sanity_zero_residual_when_identical():
    x = torch.randn(4, 3, 5, 5)
    out = extract_ar_noise_from_teacher(x, x, standardise=False)
    assert out.abs().max().item() == 0.0


def _sanity_standardise_gives_unit_variance():
    torch.manual_seed(0)
    rollout = torch.randn(6, 4, 8, 8)
    teacher = torch.randn(6, 4, 8, 8) * 0.3 + rollout
    noise = extract_ar_noise_from_teacher(rollout, teacher, standardise=True)
    # Per-sample (per-frame) std should be ~1.
    flat = noise.reshape(6, -1)
    stds = flat.std(dim=1, unbiased=False)
    assert torch.allclose(stds, torch.ones_like(stds), atol=1e-3), stds
    means = flat.mean(dim=1)
    assert means.abs().max().item() < 1e-3, means


def _sanity_make_ar_noised_gt_matches_formula():
    torch.manual_seed(1)
    gt = torch.randn(3, 2, 4, 4)
    noise = torch.randn(3, 2, 4, 4)
    alpha = torch.tensor([0.2, 0.5, 0.9]).view(3, 1, 1, 1)
    sigma = torch.tensor([0.98, 0.86, 0.43]).view(3, 1, 1, 1)
    x_t = make_ar_noised_gt(gt, noise, alpha, sigma)
    assert torch.allclose(x_t, alpha * gt + sigma * noise)


def run_sanity_checks():
    _sanity_zero_residual_when_identical()
    _sanity_standardise_gives_unit_variance()
    _sanity_make_ar_noised_gt_matches_formula()
    print("[ar_noise_teacher] sanity checks PASSED")


# --------------------------------------------------------------------------- #
# Visualisation                                                               #
# --------------------------------------------------------------------------- #


def stack_panels(panels, labels, font_scale=0.55):
    """panels: list of [T, C, H, W] in [-1, 1] -> [T, H, W*K, 3] uint8 BGR."""
    T = min(p.shape[0] for p in panels)
    arrs = [pixel_to_uint8_bgr(p[:T]) for p in panels]
    H, W = arrs[0].shape[1:3]
    cat = np.concatenate(arrs, axis=2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thick = 2
    for t in range(T):
        frame = cat[t]
        for k, lbl in enumerate(labels):
            x0 = k * W + 10
            y0 = 25
            cv2.putText(frame, lbl, (x0, y0), font, font_scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
            cv2.putText(frame, lbl, (x0, y0), font, font_scale, (255, 255, 255), thick, cv2.LINE_AA)
        cv2.putText(frame, f"frame {t}", (10, H - 12), font, font_scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(frame, f"frame {t}", (10, H - 12), font, font_scale, (255, 255, 255), thick, cv2.LINE_AA)
        cat[t] = frame
    return cat


def plot_distribution_diagnostics(ar_noise_lat: torch.Tensor, out_png: Path):
    """Histogram of ar_noise values vs unit-Gaussian, plus per-channel stats."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    flat = ar_noise_lat.reshape(-1).cpu().numpy()
    rng = np.random.default_rng(0)
    gauss = rng.standard_normal(min(flat.size, 200_000))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(flat, bins=200, density=True, alpha=0.6, label="ar_noise")
    axes[0].hist(gauss, bins=200, density=True, alpha=0.5, label="N(0,1)")
    axes[0].set_xlim(-5, 5)
    axes[0].set_title("AR noise vs Gaussian — latent value histogram")
    axes[0].set_xlabel("value")
    axes[0].set_ylabel("density")
    axes[0].legend()

    # Per-channel std (channels are dim 1 of [T, C, h, w]).
    if ar_noise_lat.dim() == 4:
        per_c_std = ar_noise_lat.reshape(ar_noise_lat.shape[0], ar_noise_lat.shape[1], -1).std(dim=(0, 2), unbiased=False).cpu().numpy()
        axes[1].bar(np.arange(per_c_std.size), per_c_std)
        axes[1].axhline(1.0, color="red", linestyle="--", label="unit std")
        axes[1].set_title("AR noise — per-channel std")
        axes[1].set_xlabel("latent channel")
        axes[1].set_ylabel("std")
        axes[1].legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", type=Path, default=_DEFAULT_ROLLOUT,
                    help="path to student rollout mp4 (pred_image)")
    ap.add_argument("--teacher_clean", type=Path, default=_DEFAULT_TEACHER,
                    help="path to teacher denoised mp4 (pred_real_lora)")
    ap.add_argument("--gt", type=Path, default=_DEFAULT_GT,
                    help="path to GT mp4 (clean_x_real) for swap-in demo")
    ap.add_argument("--out_dir", type=Path,
                    default=PROJECT_ROOT / "testing/outputs/ar_noise_teacher",
                    help="directory for visualisations + stats")
    ap.add_argument("--alpha_t", type=float, default=0.5,
                    help="diffusion alpha_t for the swap-in demo")
    ap.add_argument("--sigma_t", type=float, default=0.5,
                    help="diffusion sigma_t for the swap-in demo")
    ap.add_argument("--no_standardise", action="store_true",
                    help="skip unit-variance standardisation of ar_noise")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--skip_sanity", action="store_true")
    args = ap.parse_args()

    if not args.skip_sanity:
        run_sanity_checks()

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    print(f"[load] rollout        = {args.rollout}")
    print(f"[load] teacher_clean  = {args.teacher_clean}")
    print(f"[load] gt             = {args.gt}")
    rollout_px = load_mp4_to_tensor(args.rollout)
    teacher_px = load_mp4_to_tensor(args.teacher_clean)
    gt_px = load_mp4_to_tensor(args.gt)
    print(f"[load] rollout {tuple(rollout_px.shape)}, teacher {tuple(teacher_px.shape)}, "
          f"gt {tuple(gt_px.shape)}")

    print(f"[vae ] loading WanVAE on {device} ({dtype})")
    vae = WanVAEWrapper().to(device=device, dtype=dtype)
    vae.eval()

    print("[vae ] encoding rollout / teacher / gt ...")
    rollout_lat = encode_latents(vae, rollout_px, device, dtype)
    teacher_lat = encode_latents(vae, teacher_px, device, dtype)
    gt_lat = encode_latents(vae, gt_px, device, dtype)
    print(f"[vae ] rollout_lat {tuple(rollout_lat.shape)}, "
          f"teacher_lat {tuple(teacher_lat.shape)}, gt_lat {tuple(gt_lat.shape)}")

    # If shapes mismatch (frame count drift between mp4s), trim to common T.
    T_lat = min(rollout_lat.shape[0], teacher_lat.shape[0], gt_lat.shape[0])
    rollout_lat = rollout_lat[:T_lat]
    teacher_lat = teacher_lat[:T_lat]
    gt_lat = gt_lat[:T_lat]

    print(f"[xtr ] extracting AR noise (standardise={not args.no_standardise}) ...")
    ar_noise_lat = extract_ar_noise_from_teacher(
        rollout_lat, teacher_lat, standardise=not args.no_standardise,
    )
    print(f"[xtr ] ar_noise_lat {tuple(ar_noise_lat.shape)}")

    print("[stat] ar_noise stats vs unit Gaussian ...")
    ar_stats = residual_stats(ar_noise_lat)
    print(f"        overall_mean    = {ar_stats['overall_mean']:+.4f}  (Gauss: 0)")
    print(f"        overall_std     = {ar_stats['overall_std']:.4f}    (Gauss: 1)")
    print(f"        overall_l2_norm = {ar_stats['overall_l2_norm']:.2f}")

    # Reference Gaussian noise with the same shape, for swap-in compare.
    torch.manual_seed(0)
    gauss_noise = torch.randn_like(ar_noise_lat)

    alpha = torch.tensor(args.alpha_t).view(1, 1, 1, 1)
    sigma = torch.tensor(args.sigma_t).view(1, 1, 1, 1)
    x_t_ar = make_ar_noised_gt(gt_lat, ar_noise_lat, alpha, sigma)
    x_t_gauss = make_ar_noised_gt(gt_lat, gauss_noise, alpha, sigma)
    print(f"[mix ] alpha_t={args.alpha_t} sigma_t={args.sigma_t} -> "
          f"x_t_ar {tuple(x_t_ar.shape)}, x_t_gauss {tuple(x_t_gauss.shape)}")

    print("[vae ] decoding panels for visualisation ...")
    rollout_dec = decode_latents(vae, rollout_lat, device, dtype)
    teacher_dec = decode_latents(vae, teacher_lat, device, dtype)
    # For the noise panel, decode the ar_noise as-if it were a latent. The
    # decoder isn't trained on noise — output is just a visual heuristic of
    # where the residual concentrates spatially.
    ar_noise_dec = decode_latents(vae, ar_noise_lat.clamp(-3.0, 3.0), device, dtype)
    x_t_ar_dec = decode_latents(vae, x_t_ar, device, dtype)
    x_t_gauss_dec = decode_latents(vae, x_t_gauss, device, dtype)

    T_min = min(
        rollout_dec.shape[0], teacher_dec.shape[0], ar_noise_dec.shape[0],
        x_t_ar_dec.shape[0], x_t_gauss_dec.shape[0],
    )

    print("[viz ] writing comparison mp4 ...")
    panel_a = stack_panels(
        panels=[rollout_dec[:T_min], teacher_dec[:T_min], ar_noise_dec[:T_min]],
        labels=("student rollout", "teacher (denoised)", "ar_noise (decoded)"),
    )
    panel_b = stack_panels(
        panels=[gt_px[:T_min], x_t_ar_dec[:T_min], x_t_gauss_dec[:T_min]],
        labels=("GT clean", "GT + ar_noise", "GT + Gaussian (ref)"),
    )
    out_a = args.out_dir / "extraction.mp4"
    out_b = args.out_dir / "swap_in_demo.mp4"
    write_mp4(out_a, panel_a, fps=args.fps)
    write_mp4(out_b, panel_b, fps=args.fps)
    print(f"[viz ] wrote {out_a}")
    print(f"[viz ] wrote {out_b}")

    print("[viz ] plotting distribution diagnostics ...")
    plot_distribution_diagnostics(ar_noise_lat, args.out_dir / "noise_distribution.png")

    print("[stat] writing stats.json ...")
    dump = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "rollout_pixel_shape": list(rollout_px.shape),
        "rollout_latent_shape": list(rollout_lat.shape),
        "ar_noise_latent_shape": list(ar_noise_lat.shape),
        "ar_noise_stats": ar_stats,
        "gauss_reference_stats": residual_stats(gauss_noise),
    }
    dump_path = args.out_dir / "stats.json"
    with dump_path.open("w") as f:
        json.dump(dump, f, indent=2)
    print(f"[stat] wrote {dump_path}")
    print("[done] open extraction.mp4 / swap_in_demo.mp4 to verify visually.")


if __name__ == "__main__":
    main()
