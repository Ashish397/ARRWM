#!/usr/bin/env python3
"""2-slot append rollout with un-rotated cache + per-slot RoPE.

Two slots are denoised in parallel (NS=2, P=2 staircase). Slot 0 is
the chunk closest to commit (lowest ladder rung), slot 1 is one chunk
ahead (higher rung). Cache holds older committed chunks, stored
un-rotated; RoPE is re-applied at attention time at each slot's
intended global frame index.

Two attention modes (``--mask_mode``):
  * ``bidirectional`` — slot 0 sees slot 1's K/V and vice versa.
    Single attention call with Q = [slot_0, slot_1], K/V = [cache,
    slot_0, slot_1].
  * ``causal``        — slot 1 sees slot 0's K/V; slot 0 does NOT see
    slot 1's. Implemented as two attention calls: slot 0's output
    uses K/V = [cache, slot_0]; slot 1's output uses K/V = [cache,
    slot_0, slot_1].

Per-slot RoPE: slot 0's Q/K rotated at ``start_frame = G_0`` (its
global frame index), slot 1's at ``G_1 = G_0 + npb``. Cache K rotated
at ``start_frame = 0`` (its frame indices map directly to physical
positions).

Slot's pred_x0 from each pass is fed into the next pass via
re-noising — this is what "give them the past outputs before
renoising" means: the pred from this pass becomes the input for the
next pass via ``scheduler.add_noise(pred, noise, next_t)``.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_AF_ROOT = _REPO_ROOT / "action-forcing"
if str(_AF_ROOT) not in sys.path:
    sys.path.insert(0, str(_AF_ROOT))

from utils.eval_causal_AR import (
    load_per_rank_ride_ar, _initialize_kv_cache, _initialize_crossattn_cache,
    _set_attention_window, FRAME_SPATIAL_TOKENS, BASE_CHUNK_FRAMES,
)
from utils.eval_causal_AR_chain import ODEARRefreshPipeline
from utils.eval_chain import frames_to_mp4

import wan.modules.causal_model as cm
from wan.modules.causal_model import (
    CausalWanSelfAttention, _separate_action_tokens, _merge_action_tokens,
    causal_rope_apply,
)
from wan.modules.attention import attention as flash_attention_fn

# Per-config freqs_i cache, keyed by (id(freqs), tuple(rel_idx), h, w, head_dim/2).
_FREQS_I_CACHE: dict = {}


def _build_freqs_i(freqs: torch.Tensor, rel_indices: torch.Tensor,
                   h: int, w: int, head_dim_half: int) -> torch.Tensor:
    """Per-position complex rotation tensor; depends only on indices and grid."""
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
    """LongLive's block-relativistic RoPE, action-token-aware, fp32 path."""
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
        x_act = x_per_frame[:, :, spatial_per_frame:, :, :]
    else:
        x_sp = x[:, : f * spatial_per_frame]
        x_act = None

    seq_len = f * spatial_per_frame
    head_dim_half = D // 2
    freqs_i = _build_freqs_i(freqs, rel_indices, h, w, head_dim_half)

    x_sp_c = torch.view_as_complex(x_sp.float().reshape(B, seq_len, H, head_dim_half, 2))
    rotated_c = x_sp_c * freqs_i
    rotated = torch.view_as_real(rotated_c).reshape(B, seq_len, H, D).to(x.dtype)
    del x_sp_c, rotated_c

    if x_act is not None:
        rotated_per_frame = rotated.view(B, f, spatial_per_frame, H, D)
        out_per_frame = torch.cat([rotated_per_frame, x_act], dim=2)
        return out_per_frame.contiguous().view(B, L, H, D)
    return rotated


def _rotate_chunked(
    x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor,
    rel_indices: torch.Tensor, action_tokens_per_frame: int,
    batch_frames: int = 6,
) -> torch.Tensor:
    """Wrap _block_relativistic_rope_fast to rotate in batches of frames,
    bounding peak memory for very long prefixes (the rotation upcasts
    spatial tokens to fp32/complex64, costing 4× the bf16 footprint)."""
    F = rel_indices.shape[0]
    if F <= batch_frames or x.shape[1] == 0:
        return _block_relativistic_rope_fast(
            x, grid_sizes, freqs, rel_indices, action_tokens_per_frame,
        )
    h, w = int(grid_sizes[0, 1].item()), int(grid_sizes[0, 2].item())
    fs = h * w + action_tokens_per_frame
    out_parts = []
    for f_start in range(0, F, batch_frames):
        f_end = min(f_start + batch_frames, F)
        f_count = f_end - f_start
        sub_idx = rel_indices[f_start:f_end]
        gs_sub = grid_sizes.clone(); gs_sub[:, 0] = f_count
        x_sub = x[:, f_start * fs : f_end * fs]
        out_parts.append(_block_relativistic_rope_fast(
            x_sub, gs_sub, freqs, sub_idx, action_tokens_per_frame,
        ))
    return torch.cat(out_parts, dim=1)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


class _State:
    """Shared state for the 2-slot patched forward.

    The model is called with `x` representing the live tokens. Two
    distinct call shapes:
      * "2-slot pass" — x has 2*npb frames worth of tokens (both
        slots). state.mode = "2slot".
      * "commit refresh" — x has 1*npb frames (just-committed slot 0
        being written to cache). state.mode = "commit".
    """
    def __init__(self, mask_mode: str):
        self.mode: str = "2slot"  # or "commit"
        self.mask_mode: str = mask_mode  # "bidirectional" or "causal"
        self.npb: int = 3
        self.frame_seqlen: int = 1561
        self.G_0: int = 0  # global frame index of slot 0
        self.G_1: int = 0  # global frame index of slot 1


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

    state: _State = CausalWanSelfAttention._state
    npb = state.npb
    frame_seqlen = state.frame_seqlen
    a_per_f = self.action_tokens_per_frame

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
    current_end = current_start + num_new_tokens
    is_recompute = current_end <= _cached_global_end_index and current_start > 0

    # ---- Cache update (write un-rotated K, V for the live tokens) ----
    cache_update_info = None
    if self.local_attn_size != -1 and (current_end > _cached_global_end_index) and (
            num_new_tokens + _cached_local_end_index > kv_cache_size):
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

    # ---- Block-Relativistic RoPE (window-relative, LongLive Infinity-RoPE) ----
    # Convention: cache prefix at indices [0..prefix_frames-1], live slots at
    # [prefix_frames..num_cache_frames-1]. Q rotated at the same indices its
    # corresponding K occupies, so Q-K relative offsets always match.
    # Once the cache fills (rolled=True), Q and live-slot K indices anchor
    # at the END of local_attn_size — gives the bounded "infinite" property.

    rolled = (cache_update_info["action"] == "roll_and_insert")
    prefix_frames = local_start_index // frame_seqlen
    num_cache_frames = local_end_index // frame_seqlen
    live_frames = num_cache_frames - prefix_frames  # 2*npb in 2slot mode, npb in commit

    if rolled:
        # Bounded: live K at end of local_attn_size; cache prefix at [0..prefix_frames-1]
        live_start_idx = self.local_attn_size - live_frames
    else:
        live_start_idx = prefix_frames

    # ---- Rotated prefix (cache up to [0..prefix_frames-1]). Computed every
    # call (no per-layer cache — that would accumulate 30 × growing-prefix
    # tensors and OOM well before the underlying kv_cache rolls).
    if prefix_frames > 0:
        grid_prefix = grid_sizes.clone(); grid_prefix[:, 0] = prefix_frames
        prefix_k = temp_k[:, :local_start_index]
        prefix_rel_indices = torch.arange(0, prefix_frames, device=k.device)
        # Chunk the prefix rotation to bound peak memory for long caches.
        rotated_prefix = _rotate_chunked(
            prefix_k, grid_prefix, freqs, prefix_rel_indices, a_per_f,
            batch_frames=6,
        )
    else:
        rotated_prefix = temp_k[:, :0]

    # ---- Live region rotation ----
    if state.mode == "commit":
        # Just-committed slot 0 alone, npb frames at indices [live_start_idx..+npb-1]
        live_k_unrot = temp_k[:, local_start_index:local_end_index]
        grid_live = grid_sizes.clone(); grid_live[:, 0] = npb
        live_rel_idx = torch.arange(live_start_idx, live_start_idx + npb, device=k.device)
        live_k_rot = _block_relativistic_rope_fast(
            live_k_unrot, grid_live, freqs, live_rel_idx, a_per_f,
        )
        rotated_temp_k = torch.cat([rotated_prefix, live_k_rot], dim=1)
        # Q rotated at same indices
        roped_q = _block_relativistic_rope_fast(
            q, grid_live, freqs, live_rel_idx, a_per_f,
        )
        v_full = temp_v[:, :local_end_index]
        attn_out = flash_attention_fn(roped_q, rotated_temp_k, v_full)
    else:
        # 2-slot pass: slot 0 at [live_start_idx..+npb-1], slot 1 at [live_start_idx+npb..+2*npb-1]
        live_k_full = temp_k[:, local_start_index:local_end_index]  # 2*npb frames
        slot0_k = live_k_full[:, :npb * frame_seqlen]
        slot1_k = live_k_full[:, npb * frame_seqlen:]
        grid_one_slot = grid_sizes.clone(); grid_one_slot[:, 0] = npb
        slot0_rel_idx = torch.arange(live_start_idx, live_start_idx + npb, device=k.device)
        slot1_rel_idx = torch.arange(live_start_idx + npb, live_start_idx + 2 * npb, device=k.device)
        rotated_slot0_k = _block_relativistic_rope_fast(
            slot0_k, grid_one_slot, freqs, slot0_rel_idx, a_per_f,
        )
        rotated_slot1_k = _block_relativistic_rope_fast(
            slot1_k, grid_one_slot, freqs, slot1_rel_idx, a_per_f,
        )
        rotated_temp_k = torch.cat([rotated_prefix, rotated_slot0_k, rotated_slot1_k], dim=1)
        v_full = temp_v[:, :local_end_index]
        # Q: slot_0 at slot0_rel_idx, slot_1 at slot1_rel_idx
        slot0_q = q[:, :npb * frame_seqlen]
        slot1_q = q[:, npb * frame_seqlen:]
        roped_slot0_q = _block_relativistic_rope_fast(
            slot0_q, grid_one_slot, freqs, slot0_rel_idx, a_per_f,
        )
        roped_slot1_q = _block_relativistic_rope_fast(
            slot1_q, grid_one_slot, freqs, slot1_rel_idx, a_per_f,
        )

        if state.mask_mode == "bidirectional":
            # Single attention call: Q = [slot0_q, slot1_q], K = full, V = full
            roped_q = torch.cat([roped_slot0_q, roped_slot1_q], dim=1)
            attn_out = flash_attention_fn(roped_q, rotated_temp_k, v_full)
        else:
            # Causal: slot 0 attends to (cache + slot 0); slot 1 attends to (cache + slot 0 + slot 1)
            slot0_kv_end = rotated_prefix.shape[1] + rotated_slot0_k.shape[1]
            attn_slot0 = flash_attention_fn(
                roped_slot0_q,
                rotated_temp_k[:, :slot0_kv_end],
                v_full[:, :slot0_kv_end],
            )
            attn_slot1 = flash_attention_fn(
                roped_slot1_q,
                rotated_temp_k,
                v_full,
            )
            attn_out = torch.cat([attn_slot0, attn_slot1], dim=1)

    x = attn_out.flatten(2)
    x = self.o(x)
    return x, (current_end, local_end_index, cache_update_info)


def install(state: _State):
    if not hasattr(CausalWanSelfAttention, "_original_forward"):
        CausalWanSelfAttention._original_forward = CausalWanSelfAttention.forward
    CausalWanSelfAttention.forward = patched_forward
    CausalWanSelfAttention._state = state
    log.info("installed 2slot un-rotated cache patch (mask_mode=%s)", state.mask_mode)


def restore():
    if hasattr(CausalWanSelfAttention, "_original_forward"):
        CausalWanSelfAttention.forward = CausalWanSelfAttention._original_forward
        del CausalWanSelfAttention._original_forward
    if hasattr(CausalWanSelfAttention, "_state"):
        del CausalWanSelfAttention._state
    _FREQS_I_CACHE.clear()


@torch.no_grad()
def run(
    pipe, *,
    prompt_embeds_dev, noisy_fa_full, initial_latents_dev,
    num_gen_chunks, cache_chunks, num_frame_per_block, device, dtype,
    state: _State,
):
    base_dit = pipe.wrapper.model
    if hasattr(base_dit, "get_base_model"):
        try: base_dit = base_dit.get_base_model()
        except Exception: pass

    B = 1
    # LOCKSTEP: both slots denoised at the SAME ladder rung, no staircase.
    NS = 2; total_denoise = 4

    seed_frames = int(initial_latents_dev.shape[1])
    seed_chunks = seed_frames // num_frame_per_block

    action_tokens_per_frame = int(getattr(base_dit, "action_tokens_per_frame", 1))
    frame_seq_length = FRAME_SPATIAL_TOKENS + action_tokens_per_frame
    state.frame_seqlen = frame_seq_length
    state.npb = num_frame_per_block

    # local_attn_size needs to fit cache_chunks + 2 slots.
    local_attn_size_frames = (cache_chunks + 2) * num_frame_per_block
    kv_cache_tokens = local_attn_size_frames * frame_seq_length
    base_dit.num_frame_per_block = num_frame_per_block
    pipe.wrapper.seq_len = max(int(pipe.wrapper.seq_len),
                               (cache_chunks + 2) * num_frame_per_block * frame_seq_length)
    _set_attention_window(base_dit, local_attn_size_frames=local_attn_size_frames, max_tokens=kv_cache_tokens)

    num_transformer_blocks = len(base_dit.blocks)
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
    ladder = torch.linspace(1000.0, 50.0, total_denoise).tolist()

    C = int(initial_latents_dev.shape[2]); H = int(initial_latents_dev.shape[3]); W = int(initial_latents_dev.shape[4])

    install(state)
    try:
        # Prefill: write seed chunk's K/V into cache via commit refresh forward.
        log.info("2slot LOCKSTEP rollout: NS=%d steps=%d ladder=%s", NS, total_denoise, [round(t,1) for t in ladder])
        seed_block_fa = noisy_fa_full[:, 0:num_frame_per_block]
        seed_cond = pipe._build_action_cond_chunk(prompt_embeds_dev, seed_block_fa,
                                                   num_frames=num_frame_per_block)
        ts0 = torch.zeros([B, num_frame_per_block], device=device, dtype=torch.float32)
        state.mode = "commit"; state.G_0 = 0
        with torch.amp.autocast("cuda", dtype=dtype):
            pipe.wrapper(
                noisy_image_or_video=initial_latents_dev[:, :num_frame_per_block],
                conditional_dict=seed_cond, timestep=ts0,
                kv_cache=kv_cache, crossattn_cache=crossattn_cache,
                current_start=0,
            )
        committed_frames = num_frame_per_block

        generated: List[torch.Tensor] = []

        # LOCKSTEP rollout: both slots start fresh at idx=0 each cycle, run
        # all `total_denoise` passes together, then BOTH commit. No staircase,
        # no warmup. Both slots see identical noise levels each pass.
        while len(generated) < num_gen_chunks:
            slot_0_chunk = committed_frames // num_frame_per_block + 0
            slot_1_chunk = committed_frames // num_frame_per_block + 1
            slot_0_fa = noisy_fa_full[:, slot_0_chunk*num_frame_per_block:(slot_0_chunk+1)*num_frame_per_block]
            f1_lo = slot_1_chunk * num_frame_per_block
            f1_hi = f1_lo + num_frame_per_block
            if f1_hi <= noisy_fa_full.shape[1]:
                slot_1_fa = noisy_fa_full[:, f1_lo:f1_hi]
            else:
                slot_1_fa = noisy_fa_full[:, -num_frame_per_block:]
            joint_fa = torch.cat([slot_0_fa, slot_1_fa], dim=1)
            cond_2slot = pipe._build_action_cond_chunk(
                prompt_embeds_dev, joint_fa, num_frames=2 * num_frame_per_block,
            )

            slot0_lat = torch.randn([B, num_frame_per_block, C, H, W],
                                     dtype=torch.float32, device=device).to(dtype)
            slot1_lat = torch.randn([B, num_frame_per_block, C, H, W],
                                     dtype=torch.float32, device=device).to(dtype)

            for step_idx in range(total_denoise):
                t_val = float(ladder[step_idx])
                tt = torch.full([B, 2 * num_frame_per_block], t_val,
                                device=device, dtype=torch.float32)
                state.mode = "2slot"
                state.G_0 = slot_0_chunk * num_frame_per_block
                state.G_1 = slot_1_chunk * num_frame_per_block
                joint_lat = torch.cat([slot0_lat, slot1_lat], dim=1)
                with torch.amp.autocast("cuda", dtype=dtype):
                    out = pipe.wrapper(
                        noisy_image_or_video=joint_lat,
                        conditional_dict=cond_2slot, timestep=tt,
                        kv_cache=kv_cache, crossattn_cache=crossattn_cache,
                        current_start=committed_frames * frame_seq_length,
                    )
                pred_x0_joint = out[1]
                pred_slot0 = pred_x0_joint[:, :num_frame_per_block]
                pred_slot1 = pred_x0_joint[:, num_frame_per_block:]

                if step_idx + 1 < total_denoise:
                    next_t = float(ladder[step_idx + 1])
                    flat0 = pred_slot0.flatten(0, 1).float()
                    n0 = torch.randn_like(flat0)
                    ft = torch.full((flat0.shape[0],), next_t, device=device, dtype=torch.float32)
                    slot0_lat = scheduler.add_noise(flat0, n0, ft).view(
                        B, num_frame_per_block, C, H, W).to(dtype)
                    flat1 = pred_slot1.flatten(0, 1).float()
                    n1 = torch.randn_like(flat1)
                    slot1_lat = scheduler.add_noise(flat1, n1, ft).view(
                        B, num_frame_per_block, C, H, W).to(dtype)

            # BOTH slots commit, in order: slot 0 first, then slot 1.
            for slot_pred, slot_chunk, slot_fa in (
                (pred_slot0, slot_0_chunk, slot_0_fa),
                (pred_slot1, slot_1_chunk, slot_1_fa),
            ):
                if len(generated) >= num_gen_chunks:
                    break
                generated.append(slot_pred.detach().to(torch.float32))
                log.info("[2slot:%s:lockstep] committed chunk %d/%d (id=%d)",
                         state.mask_mode, len(generated), num_gen_chunks, slot_chunk)
                state.mode = "commit"
                state.G_0 = slot_chunk * num_frame_per_block
                ts_commit = torch.zeros([B, num_frame_per_block], device=device, dtype=torch.float32)
                cond_commit = pipe._build_action_cond_chunk(
                    prompt_embeds_dev, slot_fa, num_frames=num_frame_per_block,
                )
                with torch.amp.autocast("cuda", dtype=dtype):
                    pipe.wrapper(
                        noisy_image_or_video=slot_pred.to(dtype),
                        conditional_dict=cond_commit, timestep=ts_commit,
                        kv_cache=kv_cache, crossattn_cache=crossattn_cache,
                        current_start=committed_frames * frame_seq_length,
                    )
                committed_frames += num_frame_per_block
    finally:
        restore()

    seed_real = initial_latents_dev[:, :seed_frames].to(torch.float32)
    return torch.cat([seed_real] + generated[:num_gen_chunks], dim=1)


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
    p.add_argument("--mask_mode", choices=["bidirectional", "causal"], required=True)
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

    state = _State(mask_mode=args.mask_mode)
    t0 = time.time()
    lat = run(
        pipe,
        prompt_embeds_dev=prompt_embeds_dev, noisy_fa_full=noisy_fa_full_dev,
        initial_latents_dev=initial_latents_dev,
        num_gen_chunks=int(args.ar_gen_chunks),
        cache_chunks=int(args.cache_chunks),
        num_frame_per_block=npb, device=device, dtype=dtype, state=state,
    )
    wall = time.time() - t0
    lat_cpu = lat.to("cpu", dtype=torch.float32).clone()
    log.info("2slot %s latents=%s wall=%.1fs", args.mask_mode, tuple(lat_cpu.shape), wall)

    var_name = f"append_unrotated_2slot_lockstep_{args.mask_mode}"
    torch.save({
        "meta": {"variant": var_name, "wall_s": wall, "shape": list(lat_cpu.shape),
                 "seed": args.seed, "mask_mode": args.mask_mode,
                 "cache_chunks": args.cache_chunks},
        "gt": gt_latents,
        var_name: lat_cpu,
    }, out / "latents.pt")

    lat_dev = lat_cpu.to(device=device, dtype=dtype)
    video_np = pipe.decode_latents(lat_dev)
    frames_to_mp4(video_np, str(out / f"{var_name}.mp4"), fps=args.video_fps)
    gt_video = pipe.decode_latents(gt_latents.to(device=device, dtype=dtype))
    frames_to_mp4(gt_video, str(out / "gt.mp4"), fps=args.video_fps)

    with (out / "manifest.json").open("w") as fh:
        json.dump({"variant": var_name, "wall_s": wall, "status": "ok"}, fh, indent=2)


if __name__ == "__main__":
    main()
