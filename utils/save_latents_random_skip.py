#!/usr/bin/env python3
"""Random-skip AR refresh variant.

Baseline behaviour: for every denoise pass of every generated chunk,
rebuild the KV cache by forwarding the FIFO context chunks through the
student (``_forward_inference`` path). With probability ``skip_prob``
(default 0.10), skip that rebuild for that pass and keep the cache
from the previous pass — so ~10% of "recaches" are stochastically
dropped. Pass d=0 of each chunk always refreshes, because the cache
starts empty.

Motivation: AR_refresh_once refreshes only once per chunk (1/4 passes)
and produces severe stagger. AR_refresh refreshes every pass (4/4)
and is visually clean. This variant sits between them at ~3.6/4 on
average and tests whether refresh frequency alone drives quality, or
whether *which* specific passes get refreshed matters.

Output layout::

    <output_dir>/
      latents.pt            # {
                            #   "meta": {..., "skip_prob": 0.1},
                            #   "gt": [1, T, C, H, W] fp32 cpu,
                            #   "ar_refresh_random_skip": [1, T, C, H, W],
                            # }
      manifest.json
      ar_refresh_random_skip.mp4   # decoded video

Noise-seed policy: the chunk's noise draws are reseeded deterministically
before rollout, matching ``save_latents_three_variants.py``. The
per-pass skip pattern uses a separate ``numpy.random.Generator`` seeded
by ``args.skip_seed`` so the same skip pattern is reproducible across
runs (and independent from the noise seed).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

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
from utils.eval_chain import frames_to_mp4

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
def run_ar_refresh_random_skip(
    pipe, *,
    prompt_embeds_dev, noisy_fa_full, initial_latents_dev,
    num_gen_chunks, fifo_size, num_frame_per_block, device, dtype,
    skip_prob: float, skip_rng: np.random.Generator,
) -> tuple[torch.Tensor, dict]:
    """Per-pass-refresh variant with stochastic skips.

    For each generated chunk:
      - Reset cache at the start of the chunk (cold start).
      - For each of the ``D`` denoise passes:
          * If pass d==0 OR not a skip draw: clear the cache and forward
            the FIFO context chunks through the wrapper — this populates
            the per-layer K/V in ``kv_cache`` for positions
            ``[0, n_ctx_tokens)``.
          * Forward the current chunk with ``current_start`` pointing
            past the context (position ``n_ctx_tokens``). The wrapper
            reads K/V for context from ``[0, n_ctx_tokens)`` (fresh or
            stale) and writes current-chunk K/V at positions
            ``[n_ctx_tokens, n_ctx_tokens + cur_tokens)``.
          * Extract pred_x0 for the current chunk.
          * Renoise current chunk for next pass.

    Returns
    -------
    latents : Tensor
        [1, seed_frames + num_gen_chunks*npb, C, H, W] fp32.
    skip_log : dict
        Per-chunk skip pattern + totals.
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

    fifo_lat: List[torch.Tensor] = []
    fifo_frame_lo: List[int] = []
    for c in range(seed_chunks):
        lo = c * num_frame_per_block
        hi = lo + num_frame_per_block
        fifo_lat.append(initial_latents_dev[:, lo:hi])
        fifo_frame_lo.append(lo)

    skip_log = {
        "skip_prob": float(skip_prob),
        "per_chunk_skip_mask": [],
        "total_passes": 0,
        "total_refreshes": 0,
        "total_skips": 0,
    }

    generated: List[torch.Tensor] = []
    for chunk_idx in range(int(num_gen_chunks)):
        n_ctx_blocks = min(len(fifo_lat), fifo_size)
        cur_global_frame_lo = (seed_chunks + chunk_idx) * num_frame_per_block
        cur_global_frame_hi = cur_global_frame_lo + num_frame_per_block

        ctx_entries = fifo_lat[-n_ctx_blocks:] if n_ctx_blocks > 0 else []
        ctx_frame_los = fifo_frame_lo[-n_ctx_blocks:] if n_ctx_blocks > 0 else []
        ctx_cat = torch.cat(ctx_entries, dim=1) if ctx_entries else None
        ctx_fa = (
            torch.cat(
                [noisy_fa_full[:, fl:fl + num_frame_per_block] for fl in ctx_frame_los], dim=1,
            ) if ctx_entries else None
        )
        ctx_cond = (
            pipe._build_action_cond_chunk(
                prompt_embeds_dev, ctx_fa, num_frames=n_ctx_blocks * num_frame_per_block,
            ) if ctx_entries else None
        )
        ctx_t = (
            torch.full(
                [B, n_ctx_blocks * num_frame_per_block], 0.0,
                device=device, dtype=torch.float32,
            ) if ctx_entries else None
        )

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

        chunk_skip_mask: List[bool] = []
        for d_idx in range(int(ts.shape[0])):
            # Decide refresh vs skip. Pass 0 always refreshes.
            if d_idx == 0:
                do_skip = False
            elif ctx_entries:
                do_skip = bool(skip_rng.random() < skip_prob)
            else:
                do_skip = False  # no ctx -> nothing to refresh anyway
            chunk_skip_mask.append(do_skip)
            skip_log["total_passes"] += 1
            if do_skip:
                skip_log["total_skips"] += 1
            else:
                skip_log["total_refreshes"] += 1

            current_start_frame = 0
            if not do_skip and ctx_entries:
                _reset_kv_cache(kv_cache)
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
            elif ctx_entries:
                # Skipped refresh: cache retains context K/V from prior
                # pass. current_start must still skip past the context.
                current_start_frame = n_ctx_blocks * num_frame_per_block

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
        skip_log["per_chunk_skip_mask"].append(chunk_skip_mask)
        log.info(
            "[random_skip] chunk %d/%d committed | n_ctx=%d | skip_mask=%s",
            chunk_idx + 1, num_gen_chunks, n_ctx_blocks, chunk_skip_mask,
        )

        if len(fifo_lat) >= fifo_size:
            fifo_lat.pop(0)
            fifo_frame_lo.pop(0)
        fifo_lat.append(pred_x0.detach().to(dtype))
        fifo_frame_lo.append(cur_global_frame_lo)

    seed_real = initial_latents_dev[:, :seed_frames].to(torch.float32)
    return torch.cat([seed_real] + generated, dim=1), skip_log


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
    p.add_argument("--fifo_size", type=int, default=3)
    p.add_argument("--denoising_steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_prob", type=float, default=0.10,
                   help="Per-pass (d>=1) probability of skipping the cache refresh.")
    p.add_argument("--skip_seed", type=int, default=12345,
                   help="Seed for the skip-mask RNG (independent from noise seed).")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--write_mp4", action="store_true", default=True)
    p.add_argument("--no_mp4", dest="write_mp4", action="store_false")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    gt_latents = initial_latents[:, :total_frames].to(device="cpu", dtype=torch.float32).clone()

    noise_seed = int(args.seed) + 1_000_003
    torch.manual_seed(noise_seed)
    torch.cuda.manual_seed(noise_seed)

    skip_rng = np.random.default_rng(int(args.skip_seed))

    t0 = time.time()
    lat, skip_log = run_ar_refresh_random_skip(
        pipe,
        prompt_embeds_dev=prompt_embeds_dev,
        noisy_fa_full=noisy_fa_full_dev,
        initial_latents_dev=initial_latents_dev,
        num_gen_chunks=int(args.ar_gen_chunks),
        fifo_size=int(args.fifo_size),
        num_frame_per_block=num_frame_per_block,
        device=device,
        dtype=dtype,
        skip_prob=float(args.skip_prob),
        skip_rng=skip_rng,
    )
    log.info(
        "[random_skip] latents=%s wall=%.1fs refreshes=%d skips=%d "
        "(skip_rate=%.3f, target=%.3f)",
        tuple(lat.shape), time.time() - t0,
        skip_log["total_refreshes"], skip_log["total_skips"],
        skip_log["total_skips"] / max(1, skip_log["total_passes"]),
        float(args.skip_prob),
    )
    lat_cpu = lat.to(device="cpu", dtype=torch.float32).clone()

    payload = {
        "meta": {
            "student_ckpt": args.student_ckpt,
            "rank_zarr": args.rank_zarr,
            "rank_offset": int(args.rank_offset),
            "seed": int(args.seed),
            "skip_seed": int(args.skip_seed),
            "skip_prob": float(args.skip_prob),
            "ar_initial_chunks": int(args.ar_initial_chunks),
            "ar_gen_chunks": int(args.ar_gen_chunks),
            "fifo_size": int(args.fifo_size),
            "denoising_steps": int(args.denoising_steps),
            "num_frame_per_block": num_frame_per_block,
            "seed_frames": seed_frames,
            "total_frames": total_frames,
            "shape": list(lat_cpu.shape),
            "dtype": "float32",
            "device": "cpu",
            "variants": ["ar_refresh_random_skip"],
            "skip_log": skip_log,
        },
        "gt": gt_latents,
        "ar_refresh_random_skip": lat_cpu,
    }
    out_pt = out_dir / "latents.pt"
    torch.save(payload, out_pt)
    log.info("Saved -> %s (%.1f MB)", out_pt, out_pt.stat().st_size / (1024 * 1024))

    # MP4 decode (full rollout = seed + generated).
    if args.write_mp4:
        lat_dev = lat_cpu.to(device=device, dtype=dtype)
        video_np = pipe.decode_latents(lat_dev)
        mp4_path = out_dir / "ar_refresh_random_skip.mp4"
        frames_to_mp4(video_np, str(mp4_path), fps=20)
        log.info("Wrote mp4 -> %s (frames=%d)", mp4_path, video_np.shape[0])

    def _mad(a, b): return (a - b).abs().mean().item()
    per_chunk = [
        _mad(
            gt_latents[:, seed_frames + c * num_frame_per_block
                       : seed_frames + (c + 1) * num_frame_per_block],
            lat_cpu[:, seed_frames + c * num_frame_per_block
                    : seed_frames + (c + 1) * num_frame_per_block],
        ) for c in range(int(args.ar_gen_chunks))
    ]
    diagnostics = {
        "mad_gt_vs_random_skip": _mad(gt_latents, lat_cpu),
        "per_chunk_mad_gt_vs_random_skip": per_chunk,
        "skip_log": skip_log,
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
