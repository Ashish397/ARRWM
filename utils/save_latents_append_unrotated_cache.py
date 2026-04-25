#!/usr/bin/env python3
"""append_baseline variant where K/V cache stores **un-rotated** values.

Default append: K is RoPE-rotated at commit-time global position and
stored that way; the rotation is permanent. As more chunks accumulate,
each cached chunk has its rotation locked while live Q's rotation
keeps advancing — the relative offset between Q and a cached K is
stable, but the absolute rotation drift across chunks may produce a
chunk-period spectral peak.

This variant:
  * Stores ``norm_k(k_proj(x))`` and ``v_proj(x)`` directly in the
    cache (no RoPE rotation applied before write).
  * On every attention call, re-rotates the entire (cache + new) K
    region as one contiguous frame sequence with ``start_frame=0``.
    Q is rotated at ``start_frame = local_start_index/frame_seqlen``
    so its relative offset against the joint K sequence matches the
    "first frame at index 0" convention.
  * Cached contents never get a permanent rotation stamped in;
    rotations are recomputed fresh every attention call.

Test: if ``|0.3|`` peak in append is caused by stale/static RoPE
rotation on cached K, this variant should remove or reduce the peak.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist

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

import wan.modules.causal_model as cm
from wan.modules.causal_model import (
    CausalWanSelfAttention, _separate_action_tokens, _merge_action_tokens,
    causal_rope_apply,
)
from wan.modules.attention import attention as flash_attention_fn

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def patched_forward(
    self, x, seq_lens, grid_sizes, freqs, block_mask,
    kv_cache=None, current_start=0, cache_start=None,
):
    """Override of ``CausalWanSelfAttention.forward`` for the cached
    path that stores un-rotated K/V and rotates at attention time."""
    if kv_cache is None:
        # Joint forward path — fall back to original behaviour.
        return CausalWanSelfAttention._original_forward(
            self, x, seq_lens, grid_sizes, freqs, block_mask,
            kv_cache=None, current_start=current_start, cache_start=cache_start,
        )

    if cache_start is None:
        cache_start = current_start

    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

    q = self.norm_q(self.q(x)).view(b, s, n, d)
    k = self.norm_k(self.k(x)).view(b, s, n, d)
    v = self.v(x).view(b, s, n, d)

    a_per_f = self.action_tokens_per_frame
    frame_seqlen = math.prod(grid_sizes[0][1:]).item() + a_per_f
    current_start_frame = current_start // frame_seqlen
    num_new_tokens = q.shape[1]
    num_new_frames = num_new_tokens // frame_seqlen

    sink_tokens = self.sink_size * frame_seqlen
    kv_cache_size = kv_cache["k"].shape[1]

    _frozen_le = getattr(self, "_frozen_local_end_index", None)
    _frozen_ge = getattr(self, "_frozen_global_end_index", None)
    if _frozen_le is not None:
        _cached_local_end_index = int(_frozen_le)
    else:
        _cached_local_end_index = int(kv_cache["local_end_index"].item())
    if _frozen_ge is not None:
        _cached_global_end_index = int(_frozen_ge)
    else:
        _cached_global_end_index = int(kv_cache["global_end_index"].item())

    current_end = current_start + num_new_tokens
    is_recompute = current_end <= _cached_global_end_index and current_start > 0

    cache_update_info = None
    if self.local_attn_size != -1 and (current_end > _cached_global_end_index) and (
            num_new_tokens + _cached_local_end_index > kv_cache_size):
        # Roll the cache: shift older entries left, evict oldest.
        num_evicted_tokens = num_new_tokens + _cached_local_end_index - kv_cache_size
        num_rolled_tokens = _cached_local_end_index - num_evicted_tokens - sink_tokens
        local_end_index = _cached_local_end_index + current_end - \
            _cached_global_end_index - num_evicted_tokens
        local_start_index = local_end_index - num_new_tokens
        temp_k = kv_cache["k"].clone()
        temp_v = kv_cache["v"].clone()
        if num_rolled_tokens > 0:
            temp_k[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                temp_k[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            temp_v[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                temp_v[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
        write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
        roped_offset = max(0, write_start_index - local_start_index)
        write_len = max(0, local_end_index - write_start_index)
        if write_len > 0:
            # Write UN-rotated K and V.
            temp_k[:, write_start_index:local_end_index] = k[:, roped_offset:roped_offset + write_len]
            temp_v[:, write_start_index:local_end_index] = v[:, roped_offset:roped_offset + write_len]
        cache_update_info = {
            "action": "roll_and_insert",
            "sink_tokens": sink_tokens,
            "num_rolled_tokens": num_rolled_tokens,
            "num_evicted_tokens": num_evicted_tokens,
            "local_start_index": local_start_index,
            "local_end_index": local_end_index,
            "write_start_index": write_start_index,
            "write_end_index": local_end_index,
            "new_k": k[:, roped_offset:roped_offset + write_len].detach() if write_len > 0 else None,
            "new_v": v[:, roped_offset:roped_offset + write_len].detach() if write_len > 0 else None,
            "current_end": current_end,
            "is_recompute": is_recompute,
        }
    else:
        local_end_index = _cached_local_end_index + current_end - _cached_global_end_index
        local_start_index = local_end_index - num_new_tokens
        temp_k = kv_cache["k"].clone()
        temp_v = kv_cache["v"].clone()
        write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
        roped_offset = max(0, write_start_index - local_start_index)
        write_len = max(0, local_end_index - write_start_index)
        if write_len > 0:
            temp_k[:, write_start_index:local_end_index] = k[:, roped_offset:roped_offset + write_len]
            temp_v[:, write_start_index:local_end_index] = v[:, roped_offset:roped_offset + write_len]
        cache_update_info = {
            "action": "direct_insert",
            "local_start_index": local_start_index,
            "local_end_index": local_end_index,
            "write_start_index": write_start_index,
            "write_end_index": local_end_index,
            "new_k": k[:, roped_offset:roped_offset + write_len].detach() if write_len > 0 else None,
            "new_v": v[:, roped_offset:roped_offset + write_len].detach() if write_len > 0 else None,
            "current_end": current_end,
            "is_recompute": is_recompute,
        }

    # ----------- Re-rotate temp_k (cache + new) at attention time -----------
    # The temp_k contains un-rotated K from frames [0..local_end_index/frame_seqlen).
    # Treat it as one contiguous frame sequence and rotate with start_frame=0.
    total_frames_in_kv = local_end_index // frame_seqlen
    F_total, H, W = grid_sizes[0]
    grid_sizes_full = grid_sizes.clone()
    grid_sizes_full[:, 0] = total_frames_in_kv
    # Pull out only the [0..local_end_index] slice of temp_k (the rest is zeros).
    temp_k_full = temp_k[:, :local_end_index]
    if a_per_f > 0:
        k_sp, k_act = _separate_action_tokens(temp_k_full, grid_sizes_full, a_per_f)
        rk_sp = causal_rope_apply(k_sp, grid_sizes_full, freqs, start_frame=0)
        rotated_temp_k = _merge_action_tokens(rk_sp, k_act, grid_sizes_full, a_per_f)
    else:
        rotated_temp_k = causal_rope_apply(temp_k_full, grid_sizes_full, freqs, start_frame=0)
    rotated_temp_k = rotated_temp_k.type_as(v)

    # Q: rotate at the live chunk's position in the joint sequence.
    q_start_frame = local_start_index // frame_seqlen
    if a_per_f > 0:
        q_sp, q_act = _separate_action_tokens(q, grid_sizes, a_per_f)
        rq_sp = causal_rope_apply(q_sp, grid_sizes, freqs, start_frame=q_start_frame)
        roped_query = _merge_action_tokens(rq_sp, q_act, grid_sizes, a_per_f)
    else:
        roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frame=q_start_frame)
    roped_query = roped_query.type_as(v)

    # Attention. Same window-truncation logic as original.
    temp_v_full = temp_v[:, :local_end_index]
    if sink_tokens > 0:
        local_budget = self.max_attention_size - sink_tokens
        k_sink = rotated_temp_k[:, :sink_tokens]
        v_sink = temp_v_full[:, :sink_tokens]
        if local_budget > 0:
            local_start_for_window = max(sink_tokens, local_end_index - local_budget)
            k_local = rotated_temp_k[:, local_start_for_window:local_end_index]
            v_local = temp_v_full[:, local_start_for_window:local_end_index]
            k_cat = torch.cat([k_sink, k_local], dim=1)
            v_cat = torch.cat([v_sink, v_local], dim=1)
        else:
            k_cat = k_sink; v_cat = v_sink
        attn_out = flash_attention_fn(roped_query, k_cat, v_cat)
    else:
        window_start = max(0, local_end_index - self.max_attention_size)
        attn_out = flash_attention_fn(
            roped_query,
            rotated_temp_k[:, window_start:local_end_index],
            temp_v_full[:, window_start:local_end_index],
        )

    x = attn_out.flatten(2)
    x = self.o(x)

    # Return in the same format as the original: (x, cache_update_info_tuple)
    return x, (current_end, local_end_index, cache_update_info)


def install_unrotated_cache_patch():
    """Replace ``CausalWanSelfAttention.forward`` with our patched version."""
    if not hasattr(CausalWanSelfAttention, "_original_forward"):
        CausalWanSelfAttention._original_forward = CausalWanSelfAttention.forward
    CausalWanSelfAttention.forward = patched_forward
    log.info("installed unrotated_cache patch on CausalWanSelfAttention.forward")


def restore_unrotated_cache_patch():
    if hasattr(CausalWanSelfAttention, "_original_forward"):
        CausalWanSelfAttention.forward = CausalWanSelfAttention._original_forward
        del CausalWanSelfAttention._original_forward


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

    install_unrotated_cache_patch()
    try:
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
    finally:
        restore_unrotated_cache_patch()
    wall = time.time() - t0
    lat_cpu = lat.to("cpu", dtype=torch.float32).clone()
    log.info("append_unrotated_cache latents=%s wall=%.1fs", tuple(lat_cpu.shape), wall)

    torch.save({
        "meta": {
            "student_ckpt": args.student_ckpt, "rank_zarr": args.rank_zarr,
            "seed": args.seed, "ar_gen_chunks": args.ar_gen_chunks,
            "cache_chunks": args.cache_chunks, "denoising_steps": args.denoising_steps,
            "shape": list(lat_cpu.shape), "wall_s": wall,
            "variant": "append_unrotated_cache",
            "note": "append_baseline with K cached pre-RoPE; rotation re-applied to entire cache+new at attention time with start_frame=0",
        },
        "gt": gt_latents,
        "append_unrotated_cache": lat_cpu,
    }, out / "latents.pt")
    log.info("saved %s", out / "latents.pt")

    lat_dev = lat_cpu.to(device=device, dtype=dtype)
    video_np = pipe.decode_latents(lat_dev)
    frames_to_mp4(video_np, str(out / "append_unrotated_cache.mp4"), fps=args.video_fps)
    log.info("wrote %s", out / "append_unrotated_cache.mp4")

    gt_dev = gt_latents.to(device=device, dtype=dtype)
    gt_video = pipe.decode_latents(gt_dev)
    frames_to_mp4(gt_video, str(out / "gt.mp4"), fps=args.video_fps)
    log.info("wrote %s", out / "gt.mp4")

    with (out / "manifest.json").open("w") as fh:
        json.dump({"variant": "append_unrotated_cache", "wall_s": wall, "status": "ok"}, fh, indent=2)


if __name__ == "__main__":
    main()
