#!/usr/bin/env python3
"""Run only ``append_baseline`` (1-slot AR with cache_refresh="append")
for any student checkpoint over ``--ar_gen_chunks`` chunks. Saves
latents.pt + ar_refresh.mp4 + gt.mp4. No annotation, no extra heads
required — bypasses ``eval_causal_AR.py``'s annotation pipeline that
asserts ``f == NUM_FRAMES``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

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

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--student_ckpt", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--rank_zarr", required=True)
    p.add_argument("--rank_offset", type=int, default=0)
    p.add_argument("--encoded_root", required=True)
    p.add_argument("--caption_root", required=True)
    p.add_argument("--motion_root", required=True)
    p.add_argument("--ss_vae_checkpoint", required=True)
    p.add_argument("--ar_initial_chunks", type=int, default=1)
    p.add_argument("--ar_gen_chunks", type=int, default=30)
    p.add_argument("--cache_chunks", type=int, default=12)
    p.add_argument("--denoising_steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--video_fps", type=int, default=20)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    torch.manual_seed(int(args.seed)); torch.cuda.manual_seed_all(int(args.seed))
    pipe = ODEARRefreshPipeline(device, dtype=dtype)
    pipe.build(args.config, use_action_tokens=True)
    pipe.load_checkpoint(args.student_ckpt)
    pipe.set_denoising_steps(int(args.denoising_steps))

    npb = BASE_CHUNK_FRAMES
    seed_frames = args.ar_initial_chunks * npb
    total_frames = seed_frames + args.ar_gen_chunks * npb

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)
    action_dims = list(cfg.get("action_dims", [2, 7]))

    initial_latents, prompt_embeds, noisy_fa_full, _ = load_per_rank_ride_ar(
        zarr_basename=args.rank_zarr, latent_start_offset=int(args.rank_offset),
        total_frames=total_frames, manifest_path=None,
        encoded_root=args.encoded_root, caption_root=args.caption_root,
        motion_root=args.motion_root, ss_vae_checkpoint=args.ss_vae_checkpoint,
        action_dims=action_dims, device=device,
    )
    initial_latents_dev = initial_latents[:, :seed_frames].to(device=device, dtype=dtype)
    prompt_embeds_dev = prompt_embeds.to(device=device, dtype=dtype)
    noisy_fa_full_dev = noisy_fa_full.to(device=device, dtype=dtype)
    gt_latents = initial_latents[:, :total_frames].to("cpu", dtype=torch.float32).clone()

    noise_seed = int(args.seed) + 1_000_003
    torch.manual_seed(noise_seed); torch.cuda.manual_seed(noise_seed)

    t0 = time.time()
    lat = pipe.generate_ar(
        prompt_embeds=prompt_embeds_dev,
        noisy_fa_full=noisy_fa_full_dev,
        initial_latents=initial_latents_dev,
        num_gen_chunks=int(args.ar_gen_chunks),
        cache_chunks=int(args.cache_chunks),
        chunks_per_step=1,
        context_noise_timestep=0.0,
        ar_cache=True,
        cache_refresh="append",
    )
    wall = time.time() - t0
    lat_cpu = lat.to("cpu", dtype=torch.float32).clone()
    log.info("append_baseline latents=%s wall=%.1fs", tuple(lat_cpu.shape), wall)

    torch.save({
        "meta": {
            "student_ckpt": args.student_ckpt, "rank_zarr": args.rank_zarr,
            "rank_offset": args.rank_offset, "seed": args.seed,
            "ar_gen_chunks": args.ar_gen_chunks, "ar_initial_chunks": args.ar_initial_chunks,
            "cache_chunks": args.cache_chunks, "denoising_steps": args.denoising_steps,
            "shape": list(lat_cpu.shape), "wall_s": wall,
            "variant": "append_baseline",
        },
        "gt": gt_latents,
        "append_baseline": lat_cpu,
    }, out / "latents.pt")
    log.info("saved %s", out / "latents.pt")

    lat_dev = lat_cpu.to(device=device, dtype=dtype)
    video_np = pipe.decode_latents(lat_dev)
    frames_to_mp4(video_np, str(out / "append_baseline.mp4"), fps=args.video_fps)
    log.info("wrote %s", out / "append_baseline.mp4")

    gt_dev = gt_latents.to(device=device, dtype=dtype)
    gt_video = pipe.decode_latents(gt_dev)
    frames_to_mp4(gt_video, str(out / "gt.mp4"), fps=args.video_fps)
    log.info("wrote %s", out / "gt.mp4")

    with (out / "manifest.json").open("w") as fh:
        json.dump({
            "student_ckpt": args.student_ckpt, "ar_gen_chunks": args.ar_gen_chunks,
            "wall_s": wall, "status": "ok",
        }, fh, indent=2)


if __name__ == "__main__":
    main()
