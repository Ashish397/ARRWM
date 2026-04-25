#!/usr/bin/env python3
"""AR_refresh-path append-like emulation that locks K **post-RoPE**.

Sister script to ``save_latents_ar_refresh_appendlike.py``. The
behaviour at the rollout level is identical (commit-refresh forward
captures ctx K/V; subsequent steps replay them). The difference: this
version intercepts at the ``flex_attention`` call site and overrides
the ``key`` tensor *after* RoPE has been applied — i.e. it caches the
chunk's K with the rotation it had at commit time and reuses it
without re-rotating to the current window position.

That matches what append_baseline actually does (its ctx K in the
persistent cache was rotated once at commit time and never touched
again). If freezing K's content alone (the appendlike script) keeps
quality clean while this post-RoPE freeze stutters, the failure mode
is in the rotation, not the content.

Implementation: monkey-patch
``wan.modules.causal_model.flex_attention`` for the duration of the
rollout. The patch keeps a per-forward-call layer counter (reset
before every wrapper call) so each block gets its own ctx K cache
slot. V is not cached here — it's recomputed every pass for ctx
positions too (i.e. V *content* is fresh every pass for ctx, just
like normal AR_refresh). If you want V also frozen at commit time,
pass ``--freeze_v`` and we cache the ``value`` tensor as well.
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

import wan.modules.causal_model as cm

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


class _PostRopeState:
    def __init__(self, freeze_v: bool):
        self.layer_counter: int = 0
        self.chunk_position_map: List[Tuple[int, int, int]] = []
        self.use_cache: bool = False
        self.capture_chunk_id: Optional[int] = None
        self.capture_lo: int = 0
        self.capture_hi: int = 0
        # cache[chunk_id][layer_idx] = {"k": tensor, "v": tensor (if freeze_v)}
        self.cache: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}
        self.freeze_v: bool = freeze_v

    def evict_chunk(self, chunk_id: int):
        self.cache.pop(chunk_id, None)


def install_postrope_patch(state: _PostRopeState):
    """Monkey-patch ``flex_attention`` in causal_model with our wrapper."""
    orig = cm.flex_attention

    def patched(query, key, value, block_mask=None, **kwargs):
        layer_idx = state.layer_counter
        state.layer_counter += 1

        # key/value shape: [B, n_heads, padded_seq, head_dim].
        # We index along dim=2 (sequence axis) by token positions.
        new_key = key
        new_value = value

        if state.use_cache and state.chunk_position_map:
            need_clone_k = False
            need_clone_v = False
            # Determine if any override applies; clone lazily.
            for chunk_id, lo, hi in state.chunk_position_map:
                if state.capture_chunk_id is not None and chunk_id == state.capture_chunk_id:
                    continue
                layer_cache = state.cache.get(chunk_id, {}).get(layer_idx)
                if layer_cache is None:
                    continue
                k_cached = layer_cache.get("k")
                if k_cached is not None and k_cached.shape[2] == hi - lo:
                    if not need_clone_k:
                        new_key = key.clone(); need_clone_k = True
                    new_key[:, :, lo:hi, :] = k_cached.to(new_key.dtype)
                if state.freeze_v:
                    v_cached = layer_cache.get("v")
                    if v_cached is not None and v_cached.shape[2] == hi - lo:
                        if not need_clone_v:
                            new_value = value.clone(); need_clone_v = True
                        new_value[:, :, lo:hi, :] = v_cached.to(new_value.dtype)

        if state.capture_chunk_id is not None:
            cid = state.capture_chunk_id
            lo, hi = state.capture_lo, state.capture_hi
            slot = state.cache.setdefault(cid, {}).setdefault(layer_idx, {})
            slot["k"] = new_key[:, :, lo:hi, :].detach().clone()
            if state.freeze_v:
                slot["v"] = new_value[:, :, lo:hi, :].detach().clone()

        return orig(query, new_key, new_value, block_mask=block_mask, **kwargs)

    cm.flex_attention = patched
    return orig


def restore_postrope_patch(orig_flex):
    cm.flex_attention = orig_flex


@torch.no_grad()
def run_ar_refresh_appendlike_postrope(
    pipe, *,
    prompt_embeds_dev, noisy_fa_full, initial_latents_dev,
    num_gen_chunks, fifo_size, num_frame_per_block, device, dtype,
    freeze_v: bool,
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

    state = _PostRopeState(freeze_v=freeze_v)
    orig_flex = install_postrope_patch(state)

    C = int(initial_latents_dev.shape[2])
    H = int(initial_latents_dev.shape[3])
    W = int(initial_latents_dev.shape[4])

    tokens_per_frame = frame_seq_length
    tokens_per_chunk = num_frame_per_block * tokens_per_frame

    fifo: List[Tuple[int, torch.Tensor]] = [(0, initial_latents_dev)]
    next_chunk_id = 1
    current_window_blocks = -1
    generated: List[torch.Tensor] = []

    log.info("AR_refresh appendlike-postrope rollout: seed=%d  gen=%d  fifo=%d  freeze_v=%s",
             seed_frames, num_gen_chunks, fifo_size, freeze_v)

    def _build_position_map(ctx_chunks: List[Tuple[int, torch.Tensor]],
                            current_chunk_id: Optional[int]) -> List[Tuple[int, int, int]]:
        out = []
        pos = 0
        for cid, _ in ctx_chunks:
            out.append((cid, pos, pos + tokens_per_chunk))
            pos += tokens_per_chunk
        if current_chunk_id is not None:
            out.append((current_chunk_id, pos, pos + tokens_per_chunk))
        return out

    def _wrapper_forward(*, x_full, cond, tt):
        """Reset layer counter then call the wrapper."""
        state.layer_counter = 0
        with torch.amp.autocast("cuda", dtype=dtype):
            return pipe.wrapper(
                noisy_image_or_video=x_full,
                conditional_dict=cond,
                timestep=tt,
                clean_x=None, aug_t=None,
            )

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
            denoise_pos_map = _build_position_map(ctx_entries, current_chunk_id)

            current_noise = torch.randn(
                [B, num_frame_per_block, C, H, W], dtype=torch.float32, device=device,
            )
            x_cur = current_noise.to(dtype)
            pred_x0_window: Optional[torch.Tensor] = None

            for d_idx in range(int(ts.shape[0])):
                state.chunk_position_map = denoise_pos_map
                state.use_cache = True
                state.capture_chunk_id = None

                t_val = float(ts[d_idx].item())
                t_cur_vec = torch.full(
                    [B, num_frame_per_block], t_val, device=device, dtype=torch.float32,
                )
                tt = torch.cat([t_ctx_vec, t_cur_vec], dim=1)
                x_full = torch.cat([ctx_cat, x_cur], dim=1)
                cond = pipe._build_action_cond_chunk(
                    prompt_embeds_dev, fa_window, num_frames=window_frames,
                )
                out = _wrapper_forward(x_full=x_full, cond=cond, tt=tt)
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

            # Commit refresh forward at t=0 with the just-committed clean
            # chunk in the current slot. Capture post-RoPE K (and V if
            # requested) for this chunk.
            ctx_t_vec_cr = torch.zeros(
                [B, n_ctx_blocks * num_frame_per_block + num_frame_per_block],
                device=device, dtype=torch.float32,
            )
            state.chunk_position_map = denoise_pos_map
            state.use_cache = True
            state.capture_chunk_id = current_chunk_id
            state.capture_lo = n_ctx_blocks * num_frame_per_block * tokens_per_frame
            state.capture_hi = state.capture_lo + tokens_per_chunk
            x_full_clean = torch.cat([ctx_cat, cur_pred.to(dtype)], dim=1)
            cond_cr = pipe._build_action_cond_chunk(
                prompt_embeds_dev, fa_window, num_frames=window_frames,
            )
            _wrapper_forward(x_full=x_full_clean, cond=cond_cr, tt=ctx_t_vec_cr)
            state.capture_chunk_id = None
            state.use_cache = False

            if len(fifo) >= fifo_size:
                evicted_id, _ = fifo.pop(0)
                state.evict_chunk(evicted_id)
            fifo.append((current_chunk_id, cur_pred.to(dtype)))

            log.info("[postrope] chunk %d/%d committed (id=%d) | cached chunks=%s",
                     chunk_idx + 1, num_gen_chunks, current_chunk_id,
                     sorted(state.cache.keys()))
    finally:
        restore_postrope_patch(orig_flex)
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
    p.add_argument("--freeze_v", action="store_true",
                   help="Also lock V (post-attention values) at commit time. "
                        "Default: V remains fresh per pass for ctx positions too.")
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
    lat = run_ar_refresh_appendlike_postrope(
        pipe,
        prompt_embeds_dev=prompt_embeds_dev,
        noisy_fa_full=noisy_fa_full_dev,
        initial_latents_dev=initial_latents_dev,
        num_gen_chunks=int(args.ar_gen_chunks),
        fifo_size=int(args.fifo_size),
        num_frame_per_block=npb,
        device=device, dtype=dtype,
        freeze_v=args.freeze_v,
    )
    wall = time.time() - t0
    lat_cpu = lat.to("cpu", dtype=torch.float32).clone()
    log.info("ar_refresh_appendlike_postrope latents=%s wall=%.1fs", tuple(lat_cpu.shape), wall)

    suffix = "postrope_freezeKV" if args.freeze_v else "postrope_freezeK"
    torch.save({
        "meta": {
            "student_ckpt": args.student_ckpt,
            "rank_zarr": args.rank_zarr, "rank_offset": args.rank_offset,
            "seed": args.seed, "ar_gen_chunks": args.ar_gen_chunks,
            "fifo_size": args.fifo_size, "denoising_steps": args.denoising_steps,
            "shape": list(lat_cpu.shape), "wall_s": wall,
            "variant": f"ar_refresh_{suffix}",
            "freeze_v": args.freeze_v,
            "note": "ctx K (and V if freeze_v) captured POST-RoPE at commit refresh; reused across all subsequent steps without re-rotation",
        },
        "gt": gt_latents,
        f"ar_refresh_{suffix}": lat_cpu,
    }, out / "latents.pt")
    log.info("saved %s", out / "latents.pt")

    lat_dev = lat_cpu.to(device=device, dtype=dtype)
    video_np = pipe.decode_latents(lat_dev)
    mp4_name = f"ar_refresh_{suffix}.mp4"
    frames_to_mp4(video_np, str(out / mp4_name), fps=args.video_fps)
    log.info("wrote %s", out / mp4_name)

    gt_dev = gt_latents.to(device=device, dtype=dtype)
    gt_video = pipe.decode_latents(gt_dev)
    frames_to_mp4(gt_video, str(out / "gt.mp4"), fps=args.video_fps)
    log.info("wrote %s", out / "gt.mp4")

    with (out / "manifest.json").open("w") as fh:
        json.dump({
            "variant": f"ar_refresh_{suffix}",
            "freeze_v": args.freeze_v,
            "wall_s": wall, "status": "ok",
        }, fh, indent=2)


if __name__ == "__main__":
    main()
