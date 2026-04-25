#!/usr/bin/env python3
"""append with LongLive Block-Relativistic RoPE (Infinity-RoPE).

Adopts the optimisations from
``NVlabs/LongLive/wan/modules/causal_model_infinity.py``:

  1. **Window-relative indexing**, bounded once the cache fills:
      * cache K rotated at indices ``[0..num_cache_frames-1]``
      * Q rotated at ``[local_attn_size - num_new_frames, local_attn_size]``
        once cache is full (so Q-rotation magnitude is bounded — gives the
        "infinite" property for long rollouts)
      * Q rotated at ``[local_start_index/frame_seqlen, ...]`` before cache
        fills (matches their direct-insert path)
  2. **Un-roped K stored in cache**, RoPE applied at attention time.
  3. **No fp64 cast** in the rotation — uses fp32 complex arithmetic.
     The original ``causal_rope_apply`` upcasts to fp64 for numerical
     headroom but the bf16/fp32 path is plenty for inference.
  4. **freqs_i cache** keyed by (relative_frame_indices, h, w) — built
     once per cache configuration, reused across the 4 denoise passes.
  5. **Rotated cache K cached** across the 4 passes within a chunk
     (only needs recompute when ``local_start_index`` changes).
  6. **Action tokens** preserved unchanged through the rotation.

Compared to default append: same caching but RoPE is applied to cached K
at attention time. Compared to the upstream
``causal_model_infinity.py``: adds action-token splitting/merging and
the per-pass rotation cache.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_AF_ROOT = _REPO_ROOT / "action-forcing"
if str(_AF_ROOT) not in sys.path:
    sys.path.insert(0, str(_AF_ROOT))

from utils.eval_causal_AR import load_per_rank_ride_ar, BASE_CHUNK_FRAMES
from utils.eval_causal_AR_chain import ODEARRefreshPipeline
from utils.eval_chain import frames_to_mp4

import wan.modules.causal_model as cm
from wan.modules.causal_model import (
    CausalWanSelfAttention, _separate_action_tokens, _merge_action_tokens,
)
from wan.modules.attention import attention as flash_attention_fn

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# Per-module freqs_i cache and rotated-prefix cache. Keyed by:
#   freqs_i_cache[(id(freqs), tuple(rel_idx), h, w, head_dim)] = freqs_i tensor
# freqs_i is the complex per-position rotation tensor; depends only on indices
# and grid (h, w) and head_dim.
_FREQS_I_CACHE: dict = {}


def _build_freqs_i(freqs: torch.Tensor, rel_indices: torch.Tensor,
                   h: int, w: int, head_dim_half: int) -> torch.Tensor:
    """Return the per-position complex freqs tensor for the given relative
    frame indices and (h, w) grid. Cached.

    freqs is split into [temp_dim, h_dim, w_dim] = [c-2*(c//3), c//3, c//3]
    where c = head_dim_half. temporal indexed by rel_indices, h indexed by
    [:h], w indexed by [:w].
    """
    rel_tup = tuple(rel_indices.tolist())
    key = (id(freqs), rel_tup, h, w, head_dim_half)
    cached = _FREQS_I_CACHE.get(key)
    if cached is not None:
        return cached
    f = rel_indices.shape[0]
    seq_len = f * h * w
    c = head_dim_half
    temp_dim = c - 2 * (c // 3)
    h_dim = c // 3
    w_dim = c // 3
    f_temp = freqs[rel_indices, :temp_dim].view(f, 1, 1, -1).expand(f, h, w, -1)
    f_h = freqs[:h, temp_dim:temp_dim + h_dim].view(1, h, 1, -1).expand(f, h, w, -1)
    f_w = freqs[:w, temp_dim + h_dim:].view(1, 1, w, -1).expand(f, h, w, -1)
    freqs_i = torch.cat([f_temp, f_h, f_w], dim=-1).reshape(seq_len, 1, -1).contiguous()
    _FREQS_I_CACHE[key] = freqs_i
    return freqs_i


def _block_relativistic_rope_fast(
    x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor,
    rel_indices: torch.Tensor, action_tokens_per_frame: int,
) -> torch.Tensor:
    """LongLive's block-relativistic RoPE, action-token-aware, fp32 path.

    x:           [B, L, H, D] where L = f * (h*w + action_tokens_per_frame)
    rel_indices: [f] frame indices
    """
    B, L, H, D = x.shape
    f = rel_indices.shape[0]
    h, w = int(grid_sizes[0, 1].item()), int(grid_sizes[0, 2].item())
    spatial_per_frame = h * w
    a_per_f = action_tokens_per_frame
    expected_per_frame = spatial_per_frame + a_per_f

    if a_per_f > 0:
        x_per_frame = x.view(B, f, expected_per_frame, H, D)
        x_sp = x_per_frame[:, :, :spatial_per_frame, :, :].contiguous().view(
            B, f * spatial_per_frame, H, D)
        x_act = x_per_frame[:, :, spatial_per_frame:, :, :]  # [B, f, a_per_f, H, D]
    else:
        x_sp = x[:, : f * spatial_per_frame]
        x_act = None

    seq_len = f * spatial_per_frame
    head_dim_half = D // 2
    freqs_i = _build_freqs_i(freqs, rel_indices, h, w, head_dim_half)

    x_sp_c = torch.view_as_complex(x_sp.float().reshape(B, seq_len, H, head_dim_half, 2))
    rotated_c = x_sp_c * freqs_i
    rotated = torch.view_as_real(rotated_c).reshape(B, seq_len, H, D).to(x.dtype)

    if x_act is not None:
        rotated_per_frame = rotated.view(B, f, spatial_per_frame, H, D)
        out_per_frame = torch.cat([rotated_per_frame, x_act], dim=2)
        return out_per_frame.contiguous().view(B, L, H, D)
    return rotated


def patched_forward(
    self, x, seq_lens, grid_sizes, freqs, block_mask,
    kv_cache=None, current_start=0, cache_start=None,
):
    if kv_cache is None:
        return CausalWanSelfAttention._original_forward(
            self, x, seq_lens, grid_sizes, freqs, block_mask,
            kv_cache=None, current_start=current_start, cache_start=cache_start,
        )
    if cache_start is None:
        cache_start = current_start

    a_per_f = self.action_tokens_per_frame
    h_grid, w_grid = int(grid_sizes[0, 1].item()), int(grid_sizes[0, 2].item())
    frame_seqlen = h_grid * w_grid + a_per_f
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    q = self.norm_q(self.q(x)).view(b, s, n, d)
    k = self.norm_k(self.k(x)).view(b, s, n, d)
    v = self.v(x).view(b, s, n, d)

    sink_tokens = self.sink_size * frame_seqlen
    kv_cache_size = kv_cache["k"].shape[1]
    _frozen_le = getattr(self, "_frozen_local_end_index", None)
    _frozen_ge = getattr(self, "_frozen_global_end_index", None)
    _cached_local_end_index = int(_frozen_le if _frozen_le is not None else kv_cache["local_end_index"].item())
    _cached_global_end_index = int(_frozen_ge if _frozen_ge is not None else kv_cache["global_end_index"].item())
    num_new_tokens = q.shape[1]
    num_new_frames = num_new_tokens // frame_seqlen
    current_end = current_start + num_new_tokens
    is_recompute = current_end <= _cached_global_end_index and current_start > 0

    cache_update_info = None
    rolled = False
    if self.local_attn_size != -1 and (current_end > _cached_global_end_index) and (
            num_new_tokens + _cached_local_end_index > kv_cache_size):
        rolled = True
        num_evicted_tokens = num_new_tokens + _cached_local_end_index - kv_cache_size
        num_rolled_tokens = _cached_local_end_index - num_evicted_tokens - sink_tokens
        local_end_index = _cached_local_end_index + current_end - _cached_global_end_index - num_evicted_tokens
        local_start_index = local_end_index - num_new_tokens
        temp_k = kv_cache["k"].clone(); temp_v = kv_cache["v"].clone()
        if num_rolled_tokens > 0:
            temp_k[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                temp_k[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            temp_v[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                temp_v[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
        write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
        roped_offset = max(0, write_start_index - local_start_index)
        write_len = max(0, local_end_index - write_start_index)
        if write_len > 0:
            temp_k[:, write_start_index:local_end_index] = k[:, roped_offset:roped_offset + write_len]
            temp_v[:, write_start_index:local_end_index] = v[:, roped_offset:roped_offset + write_len]
        cache_update_info = {
            "action": "roll_and_insert",
            "sink_tokens": sink_tokens, "num_rolled_tokens": num_rolled_tokens,
            "num_evicted_tokens": num_evicted_tokens,
            "local_start_index": local_start_index, "local_end_index": local_end_index,
            "write_start_index": write_start_index, "write_end_index": local_end_index,
            "new_k": k[:, roped_offset:roped_offset + write_len].detach() if write_len > 0 else None,
            "new_v": v[:, roped_offset:roped_offset + write_len].detach() if write_len > 0 else None,
            "current_end": current_end, "is_recompute": is_recompute,
        }
        # rolling invalidates rotated prefix cache
        self._rot_prefix_k = None
        self._rot_prefix_local_start = -1
    else:
        local_end_index = _cached_local_end_index + current_end - _cached_global_end_index
        local_start_index = local_end_index - num_new_tokens
        temp_k = kv_cache["k"].clone(); temp_v = kv_cache["v"].clone()
        write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
        roped_offset = max(0, write_start_index - local_start_index)
        write_len = max(0, local_end_index - write_start_index)
        if write_len > 0:
            temp_k[:, write_start_index:local_end_index] = k[:, roped_offset:roped_offset + write_len]
            temp_v[:, write_start_index:local_end_index] = v[:, roped_offset:roped_offset + write_len]
        cache_update_info = {
            "action": "direct_insert",
            "local_start_index": local_start_index, "local_end_index": local_end_index,
            "write_start_index": write_start_index, "write_end_index": local_end_index,
            "new_k": k[:, roped_offset:roped_offset + write_len].detach() if write_len > 0 else None,
            "new_v": v[:, roped_offset:roped_offset + write_len].detach() if write_len > 0 else None,
            "current_end": current_end, "is_recompute": is_recompute,
        }

    # ---- Block-Relativistic RoPE ----
    # Q indices: bounded after cache fills, growing before.
    if rolled:
        # Cache is full; Q anchored at end of local_attn_size window.
        q_start_idx = self.local_attn_size - num_new_frames
        query_rel_indices = torch.arange(
            q_start_idx, q_start_idx + num_new_frames, device=q.device,
        )
    else:
        # Pre-roll: Q at its growing position.
        current_frame_in_window = local_start_index // frame_seqlen
        query_rel_indices = torch.arange(
            current_frame_in_window, current_frame_in_window + num_new_frames,
            device=q.device,
        )

    # Cache K indices: 0 .. num_cache_frames-1 (always).
    num_cache_frames = local_end_index // frame_seqlen
    cache_rel_indices = torch.arange(0, num_cache_frames, device=k.device)

    # Cache the rotated prefix (the cache portion before the live region)
    # across the 4 passes within a chunk.
    cached_rot_local_start = getattr(self, "_rot_prefix_local_start", -1)
    cached_rot_k = getattr(self, "_rot_prefix_k", None)
    if cached_rot_local_start == local_start_index and cached_rot_k is not None and not rolled:
        rotated_prefix = cached_rot_k
    else:
        if local_start_index > 0:
            prefix_frames = local_start_index // frame_seqlen
            grid_prefix = grid_sizes.clone(); grid_prefix[:, 0] = prefix_frames
            prefix_k = temp_k[:, :local_start_index]
            prefix_rel_indices = torch.arange(0, prefix_frames, device=k.device)
            rotated_prefix = _block_relativistic_rope_fast(
                prefix_k, grid_prefix, freqs, prefix_rel_indices, a_per_f,
            )
        else:
            rotated_prefix = temp_k[:, :0]
        self._rot_prefix_k = rotated_prefix
        self._rot_prefix_local_start = local_start_index

    # Rotate the live region (per-pass).
    live_k = temp_k[:, local_start_index:local_end_index]
    live_frames = (local_end_index - local_start_index) // frame_seqlen
    if live_frames > 0:
        grid_live = grid_sizes.clone(); grid_live[:, 0] = live_frames
        live_start = num_cache_frames - live_frames
        live_rel_indices = torch.arange(live_start, num_cache_frames, device=k.device)
        rotated_live = _block_relativistic_rope_fast(
            live_k, grid_live, freqs, live_rel_indices, a_per_f,
        )
    else:
        rotated_live = live_k

    rotated_temp_k = torch.cat([rotated_prefix, rotated_live], dim=1)

    # Q rotation
    roped_query = _block_relativistic_rope_fast(
        q, grid_sizes, freqs, query_rel_indices, a_per_f,
    )

    temp_v_full = temp_v[:, :local_end_index]
    if sink_tokens > 0:
        local_budget = self.max_attention_size - sink_tokens
        k_sink = rotated_temp_k[:, :sink_tokens]; v_sink = temp_v_full[:, :sink_tokens]
        if local_budget > 0:
            local_start_for_window = max(sink_tokens, local_end_index - local_budget)
            k_local = rotated_temp_k[:, local_start_for_window:local_end_index]
            v_local = temp_v_full[:, local_start_for_window:local_end_index]
            k_cat = torch.cat([k_sink, k_local], dim=1); v_cat = torch.cat([v_sink, v_local], dim=1)
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
    return x, (current_end, local_end_index, cache_update_info)


def install():
    if not hasattr(CausalWanSelfAttention, "_original_forward"):
        CausalWanSelfAttention._original_forward = CausalWanSelfAttention.forward
    CausalWanSelfAttention.forward = patched_forward
    log.info("installed Block-Relativistic RoPE patch (Infinity-RoPE + action tokens + rotation cache)")


def restore():
    if hasattr(CausalWanSelfAttention, "_original_forward"):
        CausalWanSelfAttention.forward = CausalWanSelfAttention._original_forward
        del CausalWanSelfAttention._original_forward
    _FREQS_I_CACHE.clear()


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

    install()
    try:
        t0 = time.time()
        lat = pipe.generate_ar(
            prompt_embeds=prompt_embeds_dev, noisy_fa_full=noisy_fa_full_dev,
            initial_latents=initial_latents_dev,
            num_gen_chunks=int(args.ar_gen_chunks),
            cache_chunks=int(args.cache_chunks), chunks_per_step=1,
            context_noise_timestep=0.0, ar_cache=True, cache_refresh="append",
        )
    finally:
        restore()
    wall = time.time() - t0
    lat_cpu = lat.to("cpu", dtype=torch.float32).clone()
    log.info("append_infinity_rope latents=%s wall=%.1fs", tuple(lat_cpu.shape), wall)

    torch.save({
        "meta": {"variant": "append_infinity_rope", "wall_s": wall,
                 "shape": list(lat_cpu.shape), "seed": args.seed,
                 "cache_chunks": args.cache_chunks},
        "gt": gt_latents,
        "append_infinity_rope": lat_cpu,
    }, out / "latents.pt")
    log.info("saved %s", out / "latents.pt")

    lat_dev = lat_cpu.to(device=device, dtype=dtype)
    video_np = pipe.decode_latents(lat_dev)
    frames_to_mp4(video_np, str(out / "append_infinity_rope.mp4"), fps=args.video_fps)
    gt_video = pipe.decode_latents(gt_latents.to(device=device, dtype=dtype))
    frames_to_mp4(gt_video, str(out / "gt.mp4"), fps=args.video_fps)

    with (out / "manifest.json").open("w") as fh:
        json.dump({"variant": "append_infinity_rope", "wall_s": wall, "status": "ok"}, fh, indent=2)


if __name__ == "__main__":
    main()
