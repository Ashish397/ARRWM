#!/usr/bin/env python3
"""AR_refresh-path emulation of append_baseline's persistent ctx K/V.

Behaviour (per rolling step):
  1. **Commit refresh** (after the 4 denoise passes for the current
     chunk finish): run one extra forward at ``t=0`` over the current
     window with the just-committed clean chunk in the
     current-chunk slot. Replay older ctx K/V from cache; capture the
     just-committed chunk's per-layer K/V into the cache for use in
     subsequent steps.
  2. **Denoise passes** (4 per step): replay every ctx chunk's
     per-layer K/V from cache; compute the current chunk's K/V fresh
     every pass against its renoised state.
  3. Cache eviction follows the FIFO — once a chunk is no longer in
     the window, its cache entry is evicted.

This is the AR_refresh-path equivalent of append_baseline's persistent
KV cache: ctx K/V values are "stamped at commit time" and stay fixed
across all subsequent steps the chunk participates in. It differs from
true append in one detail: RoPE is re-applied at the *current* window
position when the cached K is consumed downstream — so the K content
is locked but the rotational position is current. (True append also
freezes the rotation. To match exactly we'd need to hook post-RoPE,
which would require touching the attention module; this script avoids
that by hooking only the ``self.k`` / ``self.v`` Linear projections.)

Saves latents.pt + ar_refresh_appendlike.mp4 + gt.mp4.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_AF_ROOT = _REPO_ROOT / "action-forcing"
if str(_AF_ROOT) not in sys.path:
    sys.path.insert(0, str(_AF_ROOT))

from utils.eval_causal_AR import (
    load_per_rank_ride_ar,
    FRAME_SPATIAL_TOKENS,
    BASE_CHUNK_FRAMES,
)
from utils.eval_causal_AR_chain import ODEARRefreshPipeline
from utils.eval_chain import frames_to_mp4

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


class _CtxState:
    """Shared mutable state for the wrappers.

    chunk_position_map : list of ``(chunk_id, token_lo, token_hi)`` for
        the current forward — tells the wrappers which window token
        spans correspond to which chunk identities.
    capture_chunk_id   : if not None, capture K/V of this chunk during
        the current forward (used during commit refresh).
    """
    def __init__(self):
        self.chunk_position_map: List[Tuple[int, int, int]] = []
        self.use_cache: bool = False
        self.capture_chunk_id: Optional[int] = None
        self.capture_lo: int = 0
        self.capture_hi: int = 0
        # Per-chunk per-layer cache: cache[chunk_id][layer_idx] = {"k": ..., "v": ...}
        self.cache: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}

    def evict_chunk(self, chunk_id: int):
        self.cache.pop(chunk_id, None)


class CtxCacheWrapper(nn.Module):
    def __init__(self, inner: nn.Linear, state: _CtxState, layer_idx: int, kind: str):
        super().__init__()
        self.inner = inner
        self.state = state
        self.layer_idx = layer_idx
        self.kind = kind  # "k" or "v"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.inner(x)  # always compute fresh; we may overwrite slices below

        # Override ctx slices with cached values if available.
        if self.state.use_cache and self.state.chunk_position_map:
            for chunk_id, lo, hi in self.state.chunk_position_map:
                if self.state.capture_chunk_id is not None and chunk_id == self.state.capture_chunk_id:
                    # The capturing chunk is treated as "current" — don't override.
                    continue
                layer_cache = self.state.cache.get(chunk_id, {}).get(self.layer_idx)
                if layer_cache is None:
                    continue
                t = layer_cache.get(self.kind)
                if t is None or t.shape != out[:, lo:hi, :].shape:
                    continue
                out[:, lo:hi, :] = t.to(out.dtype)

        # Capture the target chunk's K/V if requested.
        if self.state.capture_chunk_id is not None:
            cid = self.state.capture_chunk_id
            lo, hi = self.state.capture_lo, self.state.capture_hi
            slot = self.state.cache.setdefault(cid, {}).setdefault(self.layer_idx, {})
            slot[self.kind] = out[:, lo:hi, :].detach().clone()
        return out


def install_appendlike_hooks(base_dit, state: _CtxState):
    layer_idx = 0
    wrappers: List[CtxCacheWrapper] = []
    for name, module in base_dit.named_modules():
        if hasattr(module, "self_attn"):
            sa = module.self_attn
            if hasattr(sa, "k") and isinstance(sa.k, nn.Linear):
                wk = CtxCacheWrapper(sa.k, state, layer_idx, "k"); sa.k = wk; wrappers.append(wk)
            if hasattr(sa, "v") and isinstance(sa.v, nn.Linear):
                wv = CtxCacheWrapper(sa.v, state, layer_idx, "v"); sa.v = wv; wrappers.append(wv)
            layer_idx += 1
    log.info("appendlike installed on %d projections (q+k actually k+v) over %d layers",
             len(wrappers), layer_idx)
    return wrappers


def restore_hooks(base_dit, wrappers: List[CtxCacheWrapper]):
    for name, module in base_dit.named_modules():
        if hasattr(module, "self_attn"):
            sa = module.self_attn
            if hasattr(sa, "k") and isinstance(sa.k, CtxCacheWrapper):
                sa.k = sa.k.inner
            if hasattr(sa, "v") and isinstance(sa.v, CtxCacheWrapper):
                sa.v = sa.v.inner


@torch.no_grad()
def run_ar_refresh_appendlike(
    pipe, *,
    prompt_embeds_dev, noisy_fa_full, initial_latents_dev,
    num_gen_chunks, fifo_size, num_frame_per_block, device, dtype,
) -> torch.Tensor:
    base_dit = pipe.wrapper.model
    if hasattr(base_dit, "get_base_model"):
        try: base_dit = base_dit.get_base_model()
        except Exception: pass

    B = 1
    seed_frames = int(initial_latents_dev.shape[1])
    seed_chunks = seed_frames // num_frame_per_block
    max_window_blocks = fifo_size + 1
    max_window_frames = max_window_blocks * num_frame_per_block
    action_tokens_per_frame = int(getattr(base_dit, "action_tokens_per_frame", 1))
    frame_seq_length = FRAME_SPATIAL_TOKENS + action_tokens_per_frame
    pipe.wrapper.seq_len = max(int(pipe.wrapper.seq_len), max_window_frames * frame_seq_length)
    base_dit.num_frame_per_block = num_frame_per_block

    prev_local_attn_size = getattr(base_dit, "local_attn_size", -1)
    base_dit.local_attn_size = -1
    for _, module in base_dit.named_modules():
        if hasattr(module, "local_attn_size"):
            try: module.local_attn_size = -1
            except Exception: pass

    scheduler = pipe.scheduler
    scheduler.sigmas = scheduler.sigmas.to(device)
    ts = pipe.denoising_step_list

    state = _CtxState()
    wrappers = install_appendlike_hooks(base_dit, state)

    C = int(initial_latents_dev.shape[2])
    H = int(initial_latents_dev.shape[3])
    W = int(initial_latents_dev.shape[4])

    # Tokens per chunk in the window.
    tokens_per_frame = frame_seq_length
    tokens_per_chunk = num_frame_per_block * tokens_per_frame

    # FIFO of (chunk_id, latent_tensor). chunk_id is the global chunk
    # index in the rollout (seed = chunk 0 here).
    fifo: List[Tuple[int, torch.Tensor]] = [(0, initial_latents_dev)]
    next_chunk_id = 1                  # seed is id 0
    next_commit_chunk_idx = seed_chunks  # frame-position of next commit
    current_window_blocks = -1
    generated: List[torch.Tensor] = []

    log.info("AR_refresh appendlike rollout: seed=%d frames  gen=%d chunks  fifo=%d",
             seed_frames, num_gen_chunks, fifo_size)

    def _build_position_map_for_window(ctx_chunks: List[Tuple[int, torch.Tensor]],
                                       current_chunk_id: Optional[int]) -> List[Tuple[int, int, int]]:
        """Return [(chunk_id, token_lo, token_hi), ...] for the window
        composed of ctx_chunks (in order, oldest first) followed by the
        current chunk (if given)."""
        out = []
        pos = 0
        for cid, _ in ctx_chunks:
            out.append((cid, pos, pos + tokens_per_chunk))
            pos += tokens_per_chunk
        if current_chunk_id is not None:
            out.append((current_chunk_id, pos, pos + tokens_per_chunk))
        return out

    try:
        for chunk_idx in range(int(num_gen_chunks)):
            n_ctx_blocks = min(len(fifo), fifo_size)
            window_blocks = n_ctx_blocks + 1
            window_frames = window_blocks * num_frame_per_block

            cur_global_lo = (seed_chunks + chunk_idx) * num_frame_per_block
            cur_global_hi = cur_global_lo + num_frame_per_block
            ctx_global_lo = cur_global_lo - n_ctx_blocks * num_frame_per_block

            if window_blocks != current_window_blocks:
                base_dit.block_mask = None
                current_window_blocks = window_blocks

            ctx_entries = fifo[-n_ctx_blocks:]
            ctx_cat = torch.cat([t for _, t in ctx_entries], dim=1)
            fa_window = noisy_fa_full[:, ctx_global_lo:cur_global_hi].contiguous()
            t_ctx_vec = torch.zeros(
                [B, n_ctx_blocks * num_frame_per_block], device=device, dtype=torch.float32,
            )

            current_chunk_id = next_chunk_id
            next_chunk_id += 1

            # Position map for denoise passes — current_chunk in last slot.
            denoise_pos_map = _build_position_map_for_window(ctx_entries, current_chunk_id)

            current_noise = torch.randn(
                [B, num_frame_per_block, C, H, W], dtype=torch.float32, device=device,
            )
            x_cur = current_noise.to(dtype)
            pred_x0_window: Optional[torch.Tensor] = None

            # 4 denoise passes — replay ctx K/V from cache, compute current fresh.
            for d_idx in range(int(ts.shape[0])):
                state.chunk_position_map = denoise_pos_map
                state.use_cache = True
                state.capture_chunk_id = None  # no capture during denoise

                t_val = float(ts[d_idx].item())
                t_cur_vec = torch.full(
                    [B, num_frame_per_block], t_val, device=device, dtype=torch.float32,
                )
                tt = torch.cat([t_ctx_vec, t_cur_vec], dim=1)
                x_full = torch.cat([ctx_cat, x_cur], dim=1)
                cond = pipe._build_action_cond_chunk(
                    prompt_embeds_dev, fa_window, num_frames=window_frames,
                )
                with torch.amp.autocast("cuda", dtype=dtype):
                    out = pipe.wrapper(
                        noisy_image_or_video=x_full,
                        conditional_dict=cond,
                        timestep=tt,
                        clean_x=None, aug_t=None,
                    )
                pred_x0_window = out[1]
                if d_idx < int(ts.shape[0]) - 1:
                    next_t = float(ts[d_idx + 1].item())
                    cur_pred_x0 = pred_x0_window[:, n_ctx_blocks * num_frame_per_block:]
                    flat = cur_pred_x0.flatten(0, 1).float()
                    flat_noise = torch.randn_like(flat)
                    flat_t = torch.full(
                        (flat.shape[0],), next_t, device=device, dtype=torch.float32,
                    )
                    x_cur = (
                        scheduler.add_noise(flat, flat_noise, flat_t)
                        .view(B, num_frame_per_block, C, H, W)
                        .to(dtype)
                    )

            assert pred_x0_window is not None
            cur_pred = pred_x0_window[:, n_ctx_blocks * num_frame_per_block:]
            generated.append(cur_pred.detach().to(torch.float32))

            # COMMIT REFRESH: forward at t=0 with the just-committed clean
            # chunk in the current slot, replaying ctx K/V from cache
            # *and* capturing the new chunk's K/V into the cache.
            ctx_t_vec_cr = torch.zeros(
                [B, n_ctx_blocks * num_frame_per_block + num_frame_per_block],
                device=device, dtype=torch.float32,
            )
            state.chunk_position_map = denoise_pos_map  # same window
            state.use_cache = True
            state.capture_chunk_id = current_chunk_id
            state.capture_lo = n_ctx_blocks * num_frame_per_block * tokens_per_frame
            state.capture_hi = state.capture_lo + tokens_per_chunk
            x_full_clean = torch.cat([ctx_cat, cur_pred.to(dtype)], dim=1)
            cond_cr = pipe._build_action_cond_chunk(
                prompt_embeds_dev, fa_window, num_frames=window_frames,
            )
            with torch.amp.autocast("cuda", dtype=dtype):
                pipe.wrapper(
                    noisy_image_or_video=x_full_clean,
                    conditional_dict=cond_cr,
                    timestep=ctx_t_vec_cr,
                    clean_x=None, aug_t=None,
                )
            # Done capturing.
            state.capture_chunk_id = None
            state.use_cache = False

            # Push to FIFO; evict oldest if needed (and clear its cache).
            if len(fifo) >= fifo_size:
                evicted_id, _ = fifo.pop(0)
                state.evict_chunk(evicted_id)
            fifo.append((current_chunk_id, cur_pred.to(dtype)))

            log.info("[appendlike] chunk %d/%d committed (id=%d) | cached chunks=%s",
                     chunk_idx + 1, num_gen_chunks, current_chunk_id,
                     sorted(state.cache.keys()))
    finally:
        restore_hooks(base_dit, wrappers)
        base_dit.local_attn_size = prev_local_attn_size
        for _, module in base_dit.named_modules():
            if hasattr(module, "local_attn_size"):
                try: module.local_attn_size = prev_local_attn_size
                except Exception: pass
        base_dit.block_mask = None

    seed_real = initial_latents_dev[:, :seed_frames].to(torch.float32)
    return torch.cat([seed_real] + generated, dim=1)


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
    p.add_argument("--fifo_size", type=int, default=3)
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
    lat = run_ar_refresh_appendlike(
        pipe,
        prompt_embeds_dev=prompt_embeds_dev,
        noisy_fa_full=noisy_fa_full_dev,
        initial_latents_dev=initial_latents_dev,
        num_gen_chunks=int(args.ar_gen_chunks),
        fifo_size=int(args.fifo_size),
        num_frame_per_block=npb,
        device=device, dtype=dtype,
    )
    wall = time.time() - t0
    lat_cpu = lat.to("cpu", dtype=torch.float32).clone()
    log.info("ar_refresh_appendlike latents=%s wall=%.1fs", tuple(lat_cpu.shape), wall)

    torch.save({
        "meta": {
            "student_ckpt": args.student_ckpt,
            "rank_zarr": args.rank_zarr, "rank_offset": args.rank_offset,
            "seed": args.seed, "ar_gen_chunks": args.ar_gen_chunks,
            "fifo_size": args.fifo_size, "denoising_steps": args.denoising_steps,
            "shape": list(lat_cpu.shape), "wall_s": wall,
            "variant": "ar_refresh_appendlike",
            "note": "ctx K/V captured at commit-refresh and replayed across all subsequent steps the chunk participates in; current chunk K/V fresh per pass",
        },
        "gt": gt_latents,
        "ar_refresh_appendlike": lat_cpu,
    }, out / "latents.pt")
    log.info("saved %s", out / "latents.pt")

    lat_dev = lat_cpu.to(device=device, dtype=dtype)
    video_np = pipe.decode_latents(lat_dev)
    frames_to_mp4(video_np, str(out / "ar_refresh_appendlike.mp4"), fps=args.video_fps)
    log.info("wrote %s", out / "ar_refresh_appendlike.mp4")

    gt_dev = gt_latents.to(device=device, dtype=dtype)
    gt_video = pipe.decode_latents(gt_dev)
    frames_to_mp4(gt_video, str(out / "gt.mp4"), fps=args.video_fps)
    log.info("wrote %s", out / "gt.mp4")

    with (out / "manifest.json").open("w") as fh:
        json.dump({"variant": "ar_refresh_appendlike", "wall_s": wall, "status": "ok"}, fh, indent=2)


if __name__ == "__main__":
    main()
