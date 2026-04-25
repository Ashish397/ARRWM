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

    # ---- Apply RoPE ----
    # Cache portion (positions [0, local_start_index)) gets rotated with start_frame=0.
    # The live portion (slot_0 + slot_1 in 2slot mode; just slot_0_clean in commit mode)
    # gets rotated per-slot at their respective global frame indices.

    def _rotate_one(t, gs, start_frame):
        if a_per_f > 0:
            t_sp, t_act = _separate_action_tokens(t, gs, a_per_f)
            r_sp = causal_rope_apply(t_sp, gs, freqs, start_frame=start_frame)
            return _merge_action_tokens(r_sp, t_act, gs, a_per_f)
        return causal_rope_apply(t, gs, freqs, start_frame=start_frame)

    def _rotate(t, gs, start_frame, batch_frames=9):
        """Rotate in BATCH_FRAMES-sized chunks to keep peak memory bounded
        (causal_rope_apply casts to fp64/complex which is 4× memory of bf16)."""
        if t.shape[1] == 0:
            return t
        F = int(gs[0, 0].item())
        if F <= batch_frames:
            return _rotate_one(t, gs, start_frame)
        token_per_frame = t.shape[1] // F
        out_parts = []
        for f_start in range(0, F, batch_frames):
            f_end = min(f_start + batch_frames, F)
            f_count = f_end - f_start
            t_sub = t[:, f_start * token_per_frame : f_end * token_per_frame]
            gs_sub = gs.clone(); gs_sub[:, 0] = f_count
            r = _rotate_one(t_sub, gs_sub, start_frame=start_frame + f_start)
            out_parts.append(r)
        return torch.cat(out_parts, dim=1)

    # ---- Rotate cache prefix every call (no caching to keep memory bounded). ----
    if local_start_index > 0:
        prefix_frames = local_start_index // frame_seqlen
        grid_prefix = grid_sizes.clone(); grid_prefix[:, 0] = prefix_frames
        prefix_k = temp_k[:, :local_start_index]
        rotated_prefix = _rotate(prefix_k, grid_prefix, start_frame=0).type_as(v)
    else:
        rotated_prefix = temp_k[:, :0]

    # ---- Rotate live K and Q ----
    if state.mode == "commit":
        # Just-committed slot_0 at G_0; slot_0 alone, npb frames.
        live_k_unrot = temp_k[:, local_start_index:local_end_index]
        grid_live = grid_sizes.clone(); grid_live[:, 0] = npb
        live_k_rot = _rotate(live_k_unrot, grid_live, start_frame=state.G_0).type_as(v)
        rotated_temp_k = torch.cat([rotated_prefix, live_k_rot], dim=1)
        # Q: slot_0 alone at G_0
        roped_q = _rotate(q, grid_live, start_frame=state.G_0).type_as(v)
        # Single attention call
        v_full = temp_v[:, :local_end_index]
        attn_out = flash_attention_fn(roped_q, rotated_temp_k, v_full)
    else:
        # 2-slot pass: live has slot_0 and slot_1, each npb frames.
        # K layout: cache prefix + slot_0_K + slot_1_K
        live_k_full = temp_k[:, local_start_index:local_end_index]  # 2*npb frames worth
        slot0_k = live_k_full[:, :npb * frame_seqlen]
        slot1_k = live_k_full[:, npb * frame_seqlen:]
        grid_one_slot = grid_sizes.clone(); grid_one_slot[:, 0] = npb
        rotated_slot0_k = _rotate(slot0_k, grid_one_slot, start_frame=state.G_0).type_as(v)
        rotated_slot1_k = _rotate(slot1_k, grid_one_slot, start_frame=state.G_1).type_as(v)
        rotated_temp_k = torch.cat([rotated_prefix, rotated_slot0_k, rotated_slot1_k], dim=1)
        # V (un-rotated; same layout)
        v_full = temp_v[:, :local_end_index]
        # Q: slot_0 (npb frames) + slot_1 (npb frames)
        slot0_q = q[:, :npb * frame_seqlen]
        slot1_q = q[:, npb * frame_seqlen:]
        roped_slot0_q = _rotate(slot0_q, grid_one_slot, start_frame=state.G_0).type_as(v)
        roped_slot1_q = _rotate(slot1_q, grid_one_slot, start_frame=state.G_1).type_as(v)

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
    NS = 2; P = 2; total_denoise = 4
    assert NS * P == total_denoise

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
        log.info("2slot rollout: NS=%d P=%d ladder=%s", NS, P, [round(t,1) for t in ladder])
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

        # live_slots: list of (latent, ladder_idx). Slot 0 first.
        # Initialize in steady-state pattern: slot 0 at idx P (about to finish
        # in P passes), slot 1 at idx 0 (just started). Slot 0's first commit
        # is technically a warmup artefact (fresh noise pseudo-denoised through
        # only P rungs) but lets us avoid all warmup branching.
        live_slots: List[Tuple[torch.Tensor, int]] = []
        for ladder_idx_init in (P, 0):
            t_init = ladder[ladder_idx_init]
            fresh = torch.randn(
                [B, num_frame_per_block, C, H, W],
                dtype=torch.float32, device=device,
            )
            # The slot's "input" at idx P is what an in-progress denoise looks
            # like at that ladder rung. We sample from the noise distribution
            # at that t (random N(0,I) treated as a noised image at t).
            live_slots.append((fresh.to(dtype), ladder_idx_init))
        generated: List[torch.Tensor] = []

        while len(generated) < num_gen_chunks:
            # Top up to NS slots: after a commit + shift the live list drops
            # to NS-1 entries; we add a fresh slot at idx=0 here.
            while len(live_slots) < NS:
                fresh = torch.randn(
                    [B, num_frame_per_block, C, H, W],
                    dtype=torch.float32, device=device,
                ).to(dtype)
                live_slots.append((fresh, 0))

            # Run P forward passes per rolling step.
            for _ in range(P):
                if len(generated) >= num_gen_chunks:
                    break

                # Build joint input: slot_0 + slot_1
                slot_0_lat, slot_0_idx = live_slots[0]
                slot_1_lat, slot_1_idx = live_slots[1]
                joint_lat = torch.cat([slot_0_lat, slot_1_lat], dim=1)

                # Build per-slot timestep
                t0 = float(ladder[slot_0_idx])
                t1 = float(ladder[slot_1_idx])
                tt = torch.cat([
                    torch.full([B, num_frame_per_block], t0, device=device, dtype=torch.float32),
                    torch.full([B, num_frame_per_block], t1, device=device, dtype=torch.float32),
                ], dim=1)

                # Per-slot global frame indices
                slot_0_chunk = committed_frames // num_frame_per_block + 0  # next-to-commit
                slot_1_chunk = committed_frames // num_frame_per_block + 1
                state.mode = "2slot"
                state.G_0 = slot_0_chunk * num_frame_per_block
                state.G_1 = slot_1_chunk * num_frame_per_block

                # Per-slot actions: slot 0 uses action of chunk it'll commit; slot 1 uses next chunk's action.
                # Use real per-frame actions from noisy_fa_full where available.
                slot_0_fa = noisy_fa_full[:, slot_0_chunk*num_frame_per_block:(slot_0_chunk+1)*num_frame_per_block]
                # For slot 1, action might not be available if we're at the end; use last-known.
                f1_lo = slot_1_chunk * num_frame_per_block
                f1_hi = f1_lo + num_frame_per_block
                if f1_hi <= noisy_fa_full.shape[1]:
                    slot_1_fa = noisy_fa_full[:, f1_lo:f1_hi]
                else:
                    slot_1_fa = noisy_fa_full[:, -num_frame_per_block:]
                joint_fa = torch.cat([slot_0_fa, slot_1_fa], dim=1)
                cond = pipe._build_action_cond_chunk(
                    prompt_embeds_dev, joint_fa, num_frames=2 * num_frame_per_block,
                )

                with torch.amp.autocast("cuda", dtype=dtype):
                    out = pipe.wrapper(
                        noisy_image_or_video=joint_lat,
                        conditional_dict=cond, timestep=tt,
                        kv_cache=kv_cache, crossattn_cache=crossattn_cache,
                        current_start=committed_frames * frame_seq_length,
                    )
                pred_x0_joint = out[1]  # [B, 2*npb, C, H, W]
                pred_slot0 = pred_x0_joint[:, :num_frame_per_block]
                pred_slot1 = pred_x0_joint[:, num_frame_per_block:]

                # Advance ladder for both slots
                new_slot0_idx = slot_0_idx + 1
                new_slot1_idx = slot_1_idx + 1

                if new_slot0_idx >= total_denoise:
                    # Slot 0 commits.
                    generated.append(pred_slot0.detach().to(torch.float32))
                    log.info("[2slot:%s] committed chunk %d/%d (id=%d, G_0=%d)",
                             state.mask_mode, len(generated), num_gen_chunks,
                             slot_0_chunk, state.G_0)
                    # Run commit refresh forward: chunk_0 alone, t=0, write K/V to cache.
                    state.mode = "commit"
                    state.G_0 = slot_0_chunk * num_frame_per_block
                    ts_commit = torch.zeros([B, num_frame_per_block], device=device, dtype=torch.float32)
                    cond_commit = pipe._build_action_cond_chunk(
                        prompt_embeds_dev, slot_0_fa, num_frames=num_frame_per_block,
                    )
                    with torch.amp.autocast("cuda", dtype=dtype):
                        pipe.wrapper(
                            noisy_image_or_video=pred_slot0.to(dtype),
                            conditional_dict=cond_commit, timestep=ts_commit,
                            kv_cache=kv_cache, crossattn_cache=crossattn_cache,
                            current_start=committed_frames * frame_seq_length,
                        )
                    committed_frames += num_frame_per_block
                    # Shift slots: drop slot 0, slot 1 becomes new slot 0.
                    next_t_for_slot1 = ladder[new_slot1_idx]
                    flat = pred_slot1.flatten(0, 1).float()
                    flat_noise = torch.randn_like(flat)
                    flat_t = torch.full((flat.shape[0],), next_t_for_slot1, device=device, dtype=torch.float32)
                    new_slot0_lat = scheduler.add_noise(flat, flat_noise, flat_t).view(
                        B, num_frame_per_block, C, H, W).to(dtype)
                    live_slots = [(new_slot0_lat, new_slot1_idx)]
                else:
                    # Renoise both slots to next ladder rung
                    next_t_0 = ladder[new_slot0_idx]
                    flat = pred_slot0.flatten(0, 1).float()
                    fn = torch.randn_like(flat); ft = torch.full((flat.shape[0],), next_t_0, device=device, dtype=torch.float32)
                    new_slot0_lat = scheduler.add_noise(flat, fn, ft).view(B, num_frame_per_block, C, H, W).to(dtype)
                    next_t_1 = ladder[new_slot1_idx]
                    flat1 = pred_slot1.flatten(0, 1).float()
                    fn1 = torch.randn_like(flat1); ft1 = torch.full((flat1.shape[0],), next_t_1, device=device, dtype=torch.float32)
                    new_slot1_lat = scheduler.add_noise(flat1, fn1, ft1).view(B, num_frame_per_block, C, H, W).to(dtype)
                    live_slots = [(new_slot0_lat, new_slot0_idx), (new_slot1_lat, new_slot1_idx)]
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

    var_name = f"append_unrotated_2slot_{args.mask_mode}"
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
