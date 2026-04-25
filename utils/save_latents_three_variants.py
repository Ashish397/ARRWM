#!/usr/bin/env python3
"""Run all three inference variants on the same ride, same seed, same
student, and save the generated latents side-by-side for FFT / drift
analysis.

Variants:

  1. ``ar_refresh``       — AR_refresh (per-pass recompute). Uses
                             ``ODEARRefreshPipeline.generate_ar_refresh``.
                             This is the "good" reference.
  2. ``ar_refresh_once``  — rebuild-once-per-step hybrid (one joint
                             forward over the full FIFO at chunk start,
                             then 4 denoise passes on the current chunk
                             reusing that cache). Produced the worst
                             stagger in visual inspection.
  3. ``append_baseline``  — classic AR append KV cache. Uses
                             ``ODEChainPipeline.generate_ar`` with
                             ``cache_refresh='append'``.

Each variant's output latent tensor has shape ``[1, seed_frames +
ar_gen_chunks * npb, C, H, W]`` and is saved alongside the ground-truth
latents for exactly that ride-frame range, so you can diff / FFT /
compare.

Output layout::

    <output_dir>/
      latents.pt            # {
                            #   "meta": {...},
                            #   "gt": [1, T, C, H, W] fp32 cpu,
                            #   "ar_refresh": [1, T, C, H, W] fp32 cpu,
                            #   "ar_refresh_once": [1, T, C, H, W],
                            #   "append_baseline": [1, T, C, H, W],
                            # }
      manifest.json         # pointers + meta

Noise-seed policy: each variant is re-seeded with the same value
immediately before its rollout. Because the three variants call
``torch.randn``/``torch.randn_like`` the same number of times per chunk
(1 initial draw + 3 re-noise draws), chunk-0's noise is bitwise
identical across variants and subsequent chunks stay very close.
Sufficient for looking for systematic drift patterns in the frequency
domain.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    _initialize_kv_cache,
    _initialize_crossattn_cache,
    _set_attention_window,
    FRAME_SPATIAL_TOKENS,
    BASE_CHUNK_FRAMES,
)
from utils.eval_causal_AR_chain import ODEARRefreshPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


def _reset_kv_cache(kv_cache) -> None:
    for blk in kv_cache:
        blk["k"].zero_()
        blk["v"].zero_()
        blk["global_end_index"].zero_()
        blk["local_end_index"].zero_()


@torch.no_grad()
def run_ar_refresh_once(
    pipe, *,
    prompt_embeds_dev, noisy_fa_full, initial_latents_dev,
    num_gen_chunks, fifo_size, num_frame_per_block, device, dtype,
) -> torch.Tensor:
    """Rebuild-once-per-step variant: joint forward over FIFO at t=0 to
    populate KV cache, then 4 denoise passes on current chunk only.
    Returns [1, seed_frames + num_gen_chunks*npb, C, H, W] fp32.
    """
    B = 1
    base_dit = pipe.wrapper.model
    if hasattr(base_dit, "get_base_model"):
        try:
            base_dit = base_dit.get_base_model()
        except Exception:
            pass

    seed_frames = int(initial_latents_dev.shape[1])
    seed_chunks = seed_frames // num_frame_per_block
    max_window_blocks = fifo_size + 1
    action_tokens_per_frame = int(getattr(base_dit, "action_tokens_per_frame", 1))
    frame_seq_length = FRAME_SPATIAL_TOKENS + action_tokens_per_frame
    local_attn_size_frames = max_window_blocks * BASE_CHUNK_FRAMES
    kv_cache_tokens = local_attn_size_frames * frame_seq_length
    required_chunk_tokens = max_window_blocks * num_frame_per_block * frame_seq_length

    base_dit.num_frame_per_block = num_frame_per_block
    base_dit.block_mask = None
    pipe.wrapper.seq_len = max(int(pipe.wrapper.seq_len), required_chunk_tokens)
    _set_attention_window(
        base_dit, local_attn_size_frames=local_attn_size_frames, max_tokens=kv_cache_tokens,
    )

    num_transformer_blocks = len(base_dit.blocks)
    C = int(initial_latents_dev.shape[2])
    H = int(initial_latents_dev.shape[3])
    W = int(initial_latents_dev.shape[4])

    kv_cache = _initialize_kv_cache(
        num_transformer_blocks=num_transformer_blocks, batch_size=B,
        kv_cache_size_tokens=kv_cache_tokens, dtype=dtype, device=device,
    )
    crossattn_cache = _initialize_crossattn_cache(
        num_transformer_blocks=num_transformer_blocks, batch_size=B,
        dtype=dtype, device=device,
    )
    scheduler = pipe.scheduler
    scheduler.sigmas = scheduler.sigmas.to(device)
    ts = pipe.denoising_step_list

    # Seed the FIFO with real GT chunks.
    fifo_lat: List[torch.Tensor] = []
    fifo_frame_lo: List[int] = []
    for c in range(seed_chunks):
        lo = c * num_frame_per_block
        hi = lo + num_frame_per_block
        fifo_lat.append(initial_latents_dev[:, lo:hi])
        fifo_frame_lo.append(lo)

    generated: List[torch.Tensor] = []
    for chunk_idx in range(int(num_gen_chunks)):
        n_ctx_blocks = min(len(fifo_lat), fifo_size)
        cur_global_frame_lo = (seed_chunks + chunk_idx) * num_frame_per_block
        cur_global_frame_hi = cur_global_frame_lo + num_frame_per_block

        _reset_kv_cache(kv_cache)
        current_start_frame = 0
        ctx_entries = fifo_lat[-n_ctx_blocks:] if n_ctx_blocks > 0 else []
        ctx_frame_los = fifo_frame_lo[-n_ctx_blocks:] if n_ctx_blocks > 0 else []
        if ctx_entries:
            ctx_cat = torch.cat(ctx_entries, dim=1)
            ctx_fa = torch.cat(
                [noisy_fa_full[:, fl:fl + num_frame_per_block] for fl in ctx_frame_los], dim=1,
            )
            ctx_cond = pipe._build_action_cond_chunk(
                prompt_embeds_dev, ctx_fa, num_frames=n_ctx_blocks * num_frame_per_block,
            )
            ctx_t = torch.full(
                [B, n_ctx_blocks * num_frame_per_block], 0.0,
                device=device, dtype=torch.float32,
            )
            with torch.amp.autocast("cuda", dtype=dtype):
                pipe.wrapper(
                    noisy_image_or_video=ctx_cat,
                    conditional_dict=ctx_cond,
                    timestep=ctx_t,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start_frame * frame_seq_length,
                )
            current_start_frame += n_ctx_blocks * num_frame_per_block

        block_fa = noisy_fa_full[:, cur_global_frame_lo:cur_global_frame_hi]
        cond = pipe._build_action_cond_chunk(
            prompt_embeds_dev, block_fa, num_frames=num_frame_per_block,
        )
        noise = torch.randn(
            [B, num_frame_per_block, C, H, W],
            dtype=torch.float32, device=device,
        )
        x = noise.to(dtype)
        pred_x0: Optional[torch.Tensor] = None
        for d_idx in range(int(ts.shape[0])):
            t_val = float(ts[d_idx].item())
            tt = torch.full(
                [B, num_frame_per_block], t_val,
                device=device, dtype=torch.float32,
            )
            with torch.amp.autocast("cuda", dtype=dtype):
                out = pipe.wrapper(
                    noisy_image_or_video=x,
                    conditional_dict=cond,
                    timestep=tt,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start_frame * frame_seq_length,
                )
            pred_x0 = out[1]
            if d_idx < int(ts.shape[0]) - 1:
                next_t = float(ts[d_idx + 1].item())
                flat = pred_x0.flatten(0, 1).float()
                flat_noise = torch.randn_like(flat)
                flat_t = torch.full(
                    (flat.shape[0],), next_t,
                    device=device, dtype=torch.float32,
                )
                x = (
                    scheduler.add_noise(flat, flat_noise, flat_t)
                    .view(B, num_frame_per_block, C, H, W)
                    .to(dtype)
                )
        assert pred_x0 is not None
        generated.append(pred_x0.detach().to(torch.float32))

        if len(fifo_lat) >= fifo_size:
            fifo_lat.pop(0)
            fifo_frame_lo.pop(0)
        fifo_lat.append(pred_x0.detach().to(dtype))
        fifo_frame_lo.append(cur_global_frame_lo)

    seed_real = initial_latents_dev[:, :seed_frames].to(torch.float32)
    return torch.cat([seed_real] + generated, dim=1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--student_ckpt", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--rank_zarr", type=str, required=True)
    p.add_argument("--rank_offset", type=int, default=0)
    p.add_argument("--encoded_root", type=str, required=True)
    p.add_argument("--caption_root", type=str, required=True)
    p.add_argument("--motion_root", type=str, required=True)
    p.add_argument("--ss_vae_checkpoint", type=str, required=True)
    p.add_argument("--ar_initial_chunks", type=int, default=1)
    p.add_argument("--ar_gen_chunks", type=int, default=7)
    p.add_argument("--fifo_size", type=int, default=3,
                   help="AR_refresh's fifo_size and append's cache_chunks.")
    p.add_argument("--denoising_steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    # Build pipeline ONCE — reused across all three variants.
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    pipe = ODEARRefreshPipeline(device, dtype=dtype)
    pipe.build(args.config, use_action_tokens=True)
    pipe.load_checkpoint(args.student_ckpt)
    pipe.set_denoising_steps(int(args.denoising_steps))

    num_frame_per_block = BASE_CHUNK_FRAMES
    seed_frames = args.ar_initial_chunks * num_frame_per_block
    total_frames = seed_frames + args.ar_gen_chunks * num_frame_per_block

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)
    action_dims = list(cfg.get("action_dims", [2, 7]))

    initial_latents, prompt_embeds, noisy_fa_full, ride_meta = load_per_rank_ride_ar(
        zarr_basename=args.rank_zarr,
        latent_start_offset=int(args.rank_offset),
        total_frames=total_frames,
        manifest_path=None,
        encoded_root=args.encoded_root,
        caption_root=args.caption_root,
        motion_root=args.motion_root,
        ss_vae_checkpoint=args.ss_vae_checkpoint,
        action_dims=action_dims,
        device=device,
    )

    initial_latents_dev = initial_latents[:, :seed_frames].to(device=device, dtype=dtype)
    prompt_embeds_dev = prompt_embeds.to(device=device, dtype=dtype)
    noisy_fa_full_dev = noisy_fa_full.to(device=device, dtype=dtype)
    # GT latents for the full rollout frame range, kept on CPU fp32.
    gt_latents = initial_latents[:, :total_frames].to(device="cpu", dtype=torch.float32).clone()

    def _reseed():
        noise_seed = int(args.seed) + 1_000_003
        torch.manual_seed(noise_seed)
        torch.cuda.manual_seed(noise_seed)

    # --- Variant 1: AR_refresh (per-pass). ---
    _reseed()
    t0 = time.time()
    lat_ar = pipe.generate_ar_refresh(
        prompt_embeds=prompt_embeds_dev,
        noisy_fa_full=noisy_fa_full_dev,
        initial_latents=initial_latents_dev,
        num_gen_chunks=int(args.ar_gen_chunks),
        fifo_size=int(args.fifo_size),
        context_noise_timestep=0.0,
    )
    log.info("[ar_refresh] latents shape=%s | wall=%.1fs",
             tuple(lat_ar.shape), time.time() - t0)
    lat_ar_cpu = lat_ar.to(device="cpu", dtype=torch.float32).clone()
    del lat_ar
    torch.cuda.empty_cache()

    # --- Variant 2: AR_refresh_once. ---
    _reseed()
    t0 = time.time()
    lat_once = run_ar_refresh_once(
        pipe,
        prompt_embeds_dev=prompt_embeds_dev,
        noisy_fa_full=noisy_fa_full_dev,
        initial_latents_dev=initial_latents_dev,
        num_gen_chunks=int(args.ar_gen_chunks),
        fifo_size=int(args.fifo_size),
        num_frame_per_block=num_frame_per_block,
        device=device,
        dtype=dtype,
    )
    log.info("[ar_refresh_once] latents shape=%s | wall=%.1fs",
             tuple(lat_once.shape), time.time() - t0)
    lat_once_cpu = lat_once.to(device="cpu", dtype=torch.float32).clone()
    del lat_once
    torch.cuda.empty_cache()

    # --- Variant 3: append baseline. ``generate_ar`` is inherited. ---
    _reseed()
    t0 = time.time()
    lat_ap = pipe.generate_ar(
        prompt_embeds=prompt_embeds_dev,
        noisy_fa_full=noisy_fa_full_dev,
        initial_latents=initial_latents_dev,
        num_gen_chunks=int(args.ar_gen_chunks),
        cache_chunks=int(args.fifo_size),
        chunks_per_step=1,
        context_noise_timestep=0.0,
        ar_cache=True,
        cache_refresh="append",
    )
    log.info("[append_baseline] latents shape=%s | wall=%.1fs",
             tuple(lat_ap.shape), time.time() - t0)
    lat_ap_cpu = lat_ap.to(device="cpu", dtype=torch.float32).clone()
    del lat_ap
    torch.cuda.empty_cache()

    # --- Save. ---
    payload = {
        "meta": {
            "student_ckpt": args.student_ckpt,
            "rank_zarr": args.rank_zarr,
            "rank_offset": int(args.rank_offset),
            "seed": int(args.seed),
            "ar_initial_chunks": int(args.ar_initial_chunks),
            "ar_gen_chunks": int(args.ar_gen_chunks),
            "fifo_size": int(args.fifo_size),
            "denoising_steps": int(args.denoising_steps),
            "num_frame_per_block": num_frame_per_block,
            "seed_frames": seed_frames,
            "total_frames": total_frames,
            "shape": list(lat_ar_cpu.shape),
            "dtype": "float32",
            "device": "cpu",
            "variants": ["ar_refresh", "ar_refresh_once", "append_baseline"],
        },
        "gt": gt_latents,
        "ar_refresh": lat_ar_cpu,
        "ar_refresh_once": lat_once_cpu,
        "append_baseline": lat_ap_cpu,
    }
    out_pt = out_dir / "latents.pt"
    torch.save(payload, out_pt)
    log.info("Saved -> %s (%.1f MB)", out_pt, out_pt.stat().st_size / (1024 * 1024))

    # Quick diagnostic: mean absolute difference per variant vs GT and
    # per variant pair.
    def _mad(a, b): return (a - b).abs().mean().item()
    diagnostics = {
        "mad_gt_ar_refresh":        _mad(gt_latents, lat_ar_cpu),
        "mad_gt_ar_refresh_once":   _mad(gt_latents, lat_once_cpu),
        "mad_gt_append_baseline":   _mad(gt_latents, lat_ap_cpu),
        "mad_ar_vs_once":           _mad(lat_ar_cpu, lat_once_cpu),
        "mad_ar_vs_append":         _mad(lat_ar_cpu, lat_ap_cpu),
        "mad_once_vs_append":       _mad(lat_once_cpu, lat_ap_cpu),
        # Per-chunk (post-seed) MAD — see whether drift compounds per chunk.
        "per_chunk_mad_ar_vs_append": [
            _mad(
                lat_ar_cpu[:, seed_frames + c * num_frame_per_block
                            : seed_frames + (c + 1) * num_frame_per_block],
                lat_ap_cpu[:, seed_frames + c * num_frame_per_block
                            : seed_frames + (c + 1) * num_frame_per_block],
            ) for c in range(int(args.ar_gen_chunks))
        ],
        "per_chunk_mad_ar_vs_once": [
            _mad(
                lat_ar_cpu[:, seed_frames + c * num_frame_per_block
                            : seed_frames + (c + 1) * num_frame_per_block],
                lat_once_cpu[:, seed_frames + c * num_frame_per_block
                              : seed_frames + (c + 1) * num_frame_per_block],
            ) for c in range(int(args.ar_gen_chunks))
        ],
    }
    log.info("Diagnostics: %s", json.dumps(diagnostics, indent=2))

    with (out_dir / "manifest.json").open("w") as fh:
        json.dump({
            "latents_pt": str(out_pt),
            "meta": payload["meta"],
            "diagnostics": diagnostics,
        }, fh, indent=2, default=str)
    log.info("Wrote manifest.json")


if __name__ == "__main__":
    main()
