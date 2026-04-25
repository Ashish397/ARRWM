#!/usr/bin/env python3
"""Run AR_refresh on multiple rides and evaluate the 3-chunk-median
RMS collapse trigger on each.

For each ride:
  1. Roll out AR_refresh for ``--ar_gen_chunks`` chunks.
  2. Save latents (student + GT) to ``<ride>/latents.pt``.
  3. Decode both student + GT to ``<ride>/ar_refresh.mp4`` and
     ``<ride>/gt.mp4``.
  4. Compute per-chunk RMS and the 3-chunk rolling median. Write a
     per-ride plot with the user's chosen threshold drawn.

Finally writes a combined plot with all rides overlaid so you can see
whether the trigger time lines up across rides.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_AF_ROOT = _REPO_ROOT / "action-forcing"
if str(_AF_ROOT) not in sys.path:
    sys.path.insert(0, str(_AF_ROOT))

from utils.eval_causal_AR import (
    load_per_rank_ride_ar,
    BASE_CHUNK_FRAMES,
)
from utils.eval_causal_AR_chain import ODEARRefreshPipeline
from utils.eval_chain import frames_to_mp4

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def chunk_rms(chunk: torch.Tensor) -> float:
    return chunk.float().pow(2).mean().sqrt().item()


def rolling_median(vals: List[float], window: int = 3) -> List[float]:
    out = []
    for i in range(len(vals)):
        lo = max(0, i - window + 1)
        out.append(float(np.median(vals[lo:i + 1])))
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--student_ckpt", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--rides", nargs="+", required=True,
                   help="List of zarr basenames (e.g. 20240408152948.zarr).")
    p.add_argument("--encoded_root", required=True)
    p.add_argument("--caption_root", required=True)
    p.add_argument("--motion_root", required=True)
    p.add_argument("--ss_vae_checkpoint", required=True)
    p.add_argument("--ar_initial_chunks", type=int, default=1)
    p.add_argument("--ar_gen_chunks", type=int, default=30)
    p.add_argument("--fifo_size", type=int, default=3)
    p.add_argument("--denoising_steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trigger_threshold", type=float, default=0.70,
                   help="Absolute-RMS threshold for the 3-chunk-median trigger.")
    p.add_argument("--seed_rel_threshold", type=float, default=0.85,
                   help="Seed-relative threshold: fire when median(RMS_c/RMS_seed) < this.")
    p.add_argument("--trigger_window", type=int, default=3,
                   help="Rolling-median window size (chunks).")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--video_fps", type=int, default=20)
    p.add_argument("--vae_temporal_upsample", type=int, default=4)
    p.add_argument("--skip_mp4", action="store_true")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    root = Path(args.output_dir); root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    pipe = ODEARRefreshPipeline(device, dtype=dtype)
    pipe.build(args.config, use_action_tokens=True)
    pipe.load_checkpoint(args.student_ckpt)
    pipe.set_denoising_steps(int(args.denoising_steps))

    num_frame_per_block = BASE_CHUNK_FRAMES
    seed_frames = args.ar_initial_chunks * num_frame_per_block
    total_frames = seed_frames + args.ar_gen_chunks * num_frame_per_block
    secs_per_chunk = num_frame_per_block * args.vae_temporal_upsample / args.video_fps

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)
    action_dims = list(cfg.get("action_dims", [2, 7]))

    all_rms: dict[str, list[float]] = {}
    all_rms_gt: dict[str, list[float]] = {}
    all_med: dict[str, list[float]] = {}
    fires: dict[str, dict] = {}

    for ride in args.rides:
        ride_tag = ride.replace(".zarr", "")
        ride_out = root / ride_tag
        ride_out.mkdir(parents=True, exist_ok=True)
        log.info("=== Ride %s ===", ride)

        try:
            initial_latents, prompt_embeds, noisy_fa_full, _ = load_per_rank_ride_ar(
                zarr_basename=ride,
                latent_start_offset=0,
                total_frames=total_frames,
                manifest_path=None,
                encoded_root=args.encoded_root,
                caption_root=args.caption_root,
                motion_root=args.motion_root,
                ss_vae_checkpoint=args.ss_vae_checkpoint,
                action_dims=action_dims,
                device=device,
            )
        except Exception as e:
            log.error("Ride %s skipped (load failed): %s", ride, e)
            continue

        initial_latents_dev = initial_latents[:, :seed_frames].to(device=device, dtype=dtype)
        prompt_embeds_dev = prompt_embeds.to(device=device, dtype=dtype)
        noisy_fa_full_dev = noisy_fa_full.to(device=device, dtype=dtype)
        gt_latents = initial_latents[:, :total_frames].to("cpu", dtype=torch.float32).clone()

        torch.manual_seed(int(args.seed) + 1_000_003)
        torch.cuda.manual_seed(int(args.seed) + 1_000_003)
        t0 = time.time()
        lat = pipe.generate_ar_refresh(
            prompt_embeds=prompt_embeds_dev,
            noisy_fa_full=noisy_fa_full_dev,
            initial_latents=initial_latents_dev,
            num_gen_chunks=int(args.ar_gen_chunks),
            fifo_size=int(args.fifo_size),
            context_noise_timestep=0.0,
        )
        log.info("ride=%s ar_refresh wall=%.1fs", ride, time.time() - t0)
        lat_cpu = lat.to("cpu", dtype=torch.float32).clone()

        # Per-chunk RMS (student + GT).
        n_chunks = total_frames // num_frame_per_block
        rms = [chunk_rms(lat_cpu[0, c * num_frame_per_block : (c + 1) * num_frame_per_block])
               for c in range(n_chunks)]
        rms_gt = [chunk_rms(gt_latents[0, c * num_frame_per_block : (c + 1) * num_frame_per_block])
                  for c in range(n_chunks)]
        med = rolling_median(rms, window=args.trigger_window)
        med_gt = rolling_median(rms_gt, window=args.trigger_window)

        # Seed-relative median: median of (RMS_c / RMS_seed) over the window.
        seed = rms[0] if rms[0] > 1e-6 else 1e-6
        seed_gt = rms_gt[0] if rms_gt[0] > 1e-6 else 1e-6
        seed_ratio    = [r / seed    for r in rms]
        seed_ratio_gt = [r / seed_gt for r in rms_gt]
        med_rel    = rolling_median(seed_ratio,    window=args.trigger_window)
        med_rel_gt = rolling_median(seed_ratio_gt, window=args.trigger_window)

        fire_c = next((c for c in range(args.trigger_window, n_chunks)
                       if med[c] < args.trigger_threshold), None)
        fire_gt = next((c for c in range(args.trigger_window, n_chunks)
                        if med_gt[c] < args.trigger_threshold), None)
        fire_rel    = next((c for c in range(args.trigger_window, n_chunks)
                            if med_rel[c]    < args.seed_rel_threshold), None)
        fire_rel_gt = next((c for c in range(args.trigger_window, n_chunks)
                            if med_rel_gt[c] < args.seed_rel_threshold), None)

        all_rms[ride_tag] = rms
        all_rms_gt[ride_tag] = rms_gt
        all_med[ride_tag] = med
        fires[ride_tag] = {
            "seed_rms": float(seed),
            "abs_fire_chunk": fire_c,
            "abs_fire_time_s": (fire_c * secs_per_chunk) if fire_c is not None else None,
            "abs_gt_fire_chunk": fire_gt,
            "rel_fire_chunk": fire_rel,
            "rel_fire_time_s": (fire_rel * secs_per_chunk) if fire_rel is not None else None,
            "rel_gt_fire_chunk": fire_rel_gt,
            "rel_gt_fire_time_s": (fire_rel_gt * secs_per_chunk) if fire_rel_gt is not None else None,
            "threshold_abs": float(args.trigger_threshold),
            "threshold_rel": float(args.seed_rel_threshold),
            "window": int(args.trigger_window),
        }

        log.info(
            "ride=%s seed_RMS=%.3f | ABS<%.2f: student@%s (%s)  GT@%s  |  "
            "REL<%.2f: student@%s (%s)  GT@%s",
            ride, seed,
            args.trigger_threshold,
            fire_c, f"{fire_c * secs_per_chunk:.1f}s" if fire_c is not None else "never",
            fire_gt,
            args.seed_rel_threshold,
            fire_rel, f"{fire_rel * secs_per_chunk:.1f}s" if fire_rel is not None else "never",
            fire_rel_gt,
        )

        # Save per-ride payload.
        payload = {
            "meta": {
                "ride": ride, "seed": int(args.seed),
                "ar_gen_chunks": int(args.ar_gen_chunks),
                "fifo_size": int(args.fifo_size),
                "denoising_steps": int(args.denoising_steps),
                "num_frame_per_block": num_frame_per_block,
                "secs_per_chunk": secs_per_chunk,
            },
            "gt": gt_latents,
            "ar_refresh": lat_cpu,
            "rms": rms, "rms_gt": rms_gt, "median_rms": med, "median_rms_gt": med_gt,
            "fire": fires[ride_tag],
        }
        torch.save(payload, ride_out / "latents.pt")

        # Per-ride plot: two panels — absolute RMS and seed-relative RMS.
        times = [c * secs_per_chunk for c in range(n_chunks)]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(times, rms,    "-o", color="#1f77b4", label="ar_refresh RMS", lw=1.3, ms=3, alpha=0.5)
        ax1.plot(times, med,    "-",  color="#1f77b4", label=f"ar_refresh median{args.trigger_window}", lw=2)
        ax1.plot(times, rms_gt, "-o", color="#000000", label="GT RMS",       lw=1.3, ms=3, alpha=0.5)
        ax1.plot(times, med_gt, "-",  color="#000000", label=f"GT median{args.trigger_window}", lw=2)
        ax1.axhline(args.trigger_threshold, color="red", ls="--", lw=1.1,
                    label=f"abs trigger @ {args.trigger_threshold}")
        if fire_c is not None:
            ax1.axvline(fire_c * secs_per_chunk, color="red", ls=":", lw=1.3,
                        label=f"abs fires @ {fire_c * secs_per_chunk:.1f}s")
        ax1.set_ylabel("chunk RMS (absolute)")
        ax1.set_title(f"Ride {ride_tag} | seed_RMS={seed:.3f}")
        ax1.legend(fontsize=7, loc="best"); ax1.grid(alpha=0.3)

        ax2.plot(times, seed_ratio,    "-o", color="#1f77b4", label="ar_refresh RMS/seed", lw=1.3, ms=3, alpha=0.5)
        ax2.plot(times, med_rel,       "-",  color="#1f77b4", label=f"ar_refresh median{args.trigger_window}", lw=2)
        ax2.plot(times, seed_ratio_gt, "-o", color="#000000", label="GT RMS/seed", lw=1.3, ms=3, alpha=0.5)
        ax2.plot(times, med_rel_gt,    "-",  color="#000000", label=f"GT median{args.trigger_window}", lw=2)
        ax2.axhline(args.seed_rel_threshold, color="orange", ls="--", lw=1.1,
                    label=f"rel trigger @ {args.seed_rel_threshold}")
        ax2.axhline(1.0, color="grey", ls="-", lw=0.6)
        if fire_rel is not None:
            ax2.axvline(fire_rel * secs_per_chunk, color="orange", ls=":", lw=1.3,
                        label=f"rel fires @ {fire_rel * secs_per_chunk:.1f}s")
        ax2.set_xlabel("video time (s)"); ax2.set_ylabel("RMS / RMS_seed")
        ax2.legend(fontsize=7, loc="best"); ax2.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(ride_out / "rms_trigger.png", dpi=130); plt.close(fig)

        # Decode mp4s.
        if not args.skip_mp4:
            lat_dev = lat_cpu.to(device=device, dtype=dtype)
            gt_dev  = gt_latents.to(device=device, dtype=dtype)
            ar_mp4  = pipe.decode_latents(lat_dev)
            gt_mp4  = pipe.decode_latents(gt_dev)
            frames_to_mp4(ar_mp4, str(ride_out / "ar_refresh.mp4"), fps=args.video_fps)
            frames_to_mp4(gt_mp4, str(ride_out / "gt.mp4"), fps=args.video_fps)
            log.info("ride=%s mp4 written", ride)

        del lat, lat_cpu, gt_latents, initial_latents_dev, prompt_embeds_dev, noisy_fa_full_dev
        torch.cuda.empty_cache()

    # Combined overlay plot: absolute AND seed-relative on separate axes.
    n_chunks_plot = max(len(v) for v in all_rms.values())
    times = [c * secs_per_chunk for c in range(n_chunks_plot)]
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    cmap = plt.get_cmap("tab10")
    for i, (tag, rms) in enumerate(all_rms.items()):
        c = cmap(i % 10)
        axes[0].plot(times[:len(rms)], all_rms[tag],    "-", color=c, lw=2,   label=f"{tag} student")
        axes[0].plot(times[:len(rms)], all_rms_gt[tag], "--", color=c, lw=1.2, alpha=0.7, label=f"{tag} GT")
        seed = all_rms[tag][0] if all_rms[tag][0] > 1e-6 else 1e-6
        seed_gt = all_rms_gt[tag][0] if all_rms_gt[tag][0] > 1e-6 else 1e-6
        rel    = [r / seed    for r in all_rms[tag]]
        rel_gt = [r / seed_gt for r in all_rms_gt[tag]]
        axes[1].plot(times[:len(rms)], rel,    "-", color=c, lw=2,   label=f"{tag} student")
        axes[1].plot(times[:len(rms)], rel_gt, "--", color=c, lw=1.2, alpha=0.7, label=f"{tag} GT")
        if fires[tag]["rel_fire_chunk"] is not None:
            axes[1].axvline(fires[tag]["rel_fire_time_s"], color=c, ls=":", lw=1.2, alpha=0.8)
    axes[0].axhline(args.trigger_threshold, color="red", ls="--", lw=1.1,
                    label=f"abs trigger @ {args.trigger_threshold}")
    axes[0].set_ylabel("chunk RMS (absolute)")
    axes[0].set_title("Absolute RMS — seed magnitude varies per ride → threshold unreliable")
    axes[0].legend(fontsize=6, loc="best", ncol=2); axes[0].grid(alpha=0.3)
    axes[1].axhline(args.seed_rel_threshold, color="orange", ls="--", lw=1.1,
                    label=f"rel trigger @ {args.seed_rel_threshold}")
    axes[1].axhline(1.0, color="grey", lw=0.6)
    axes[1].set_xlabel("video time (s)"); axes[1].set_ylabel("RMS / RMS_seed")
    axes[1].set_title("Seed-relative — auto-calibrated to each ride's own seed chunk")
    axes[1].legend(fontsize=6, loc="best", ncol=2); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(root / "combined_rms.png", dpi=130); plt.close(fig)

    with (root / "summary.json").open("w") as fh:
        json.dump({
            "fires": fires,
            "threshold": float(args.trigger_threshold),
            "window": int(args.trigger_window),
            "secs_per_chunk": secs_per_chunk,
        }, fh, indent=2)
    log.info("Combined plot -> %s/combined_rms.png", root)
    log.info("Summary: %s", json.dumps(fires, indent=2))


if __name__ == "__main__":
    main()
