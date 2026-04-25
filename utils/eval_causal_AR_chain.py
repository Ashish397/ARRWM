#!/usr/bin/env python3
"""Autoregressive eval with **chain-style K/V freshness** (AR-refresh mode).

Hypothesis under test: the "stagger at chunk boundaries" seen in
``utils/eval_causal_AR.py``'s ``--mode ar`` relative to ``--mode chain``
comes from the KV cache being **frozen** — context chunks' K/V are
computed once per chunk (during the cache-refresh forward) and then
reused verbatim for all 4 denoise passes of the next chunk, and for
every chunk thereafter until the cache rolls them out. Chain never has
this problem because it re-runs the context tokens through the whole
transformer at every denoise step, in lockstep with the noisy tokens.

This script replaces AR's persistent KV cache with a **latent FIFO**.
Two refresh modes are supported (``--refresh_mode``):

  * ``per_pass`` (default, the original AR_refresh recipe):
    concatenates the FIFO latents with the current 3-frame noisy
    chunk into a single sequence and feeds the full window through
    the TRAINING forward (``_forward_train``, ``clean_x=None``,
    ``kv_cache=None``) at every denoise step. The model recomputes
    K/V for the context tokens from raw latents at every transformer
    layer, every denoise step — exactly what chain does. Cost: 4
    large forwards/chunk (~4.00× a single-chunk forward).

  * ``per_chunk`` (LongLive-inspired recache, "Recommendation #1"):
    at the start of every chunk, rebuilds a real ``kv_cache`` from
    the FIFO via N sequential 3-frame refresh forwards through the
    INFERENCE path (``_forward_inference``, ``current_start``
    advancing by 3 frames per refresh, ``t=context_noise_timestep``).
    Then runs the 4 denoise passes as 3-frame ``_forward_inference``
    calls at ``current_start = N*3*fsl``. Passes 2-4 append/overwrite
    only the current 3 frames' K/V. Cost: N refresh + 4 denoise
    small forwards per chunk (~1.5× a single-chunk forward for
    N=3); much faster than ``per_pass`` but shares numerics with
    ``eval_causal_AR.py --ar_cache_refresh full_fifo`` so it may or
    may not retain ``per_pass`` 's visual quality. Use an A/B
    comparison to decide.
  * Per-frame timestep: context frames at ``context_noise_timestep``
    (default 0.0, matching AR's cache-refresh t), current chunk at the
    live denoising timestep.
  * Attention mask: ``_prepare_blockwise_causal_attn_mask`` with
    ``num_frame_per_block=3``. Block-wise causal between context
    chunks; the current (last) block has bidirectional attention
    within itself and causal-past to all context blocks. Exactly the
    same semantics as chain's teacher-forcing mask for the "current"
    slot, without the waste of computing denoise outputs for slots we
    don't use.
  * RoPE positions: **compact per-window** (0..11 when the FIFO is
    full). The train forward doesn't cleanly accept a per-chunk
    position offset (chain uses compact positions 0..F-1 too), so we
    match chain's behaviour rather than AR's ``current_start``-based
    sliding. If absolute-position sliding turns out to matter it can
    be added as a separate flag later.
  * GT seed: 1 chunk (3 frames), matching ``eval_causal_AR`` 's
    ``ar_initial_chunks=1`` default. The GT chunk is pushed into the
    FIFO at init so chunk 0 has one context chunk to attend to. As
    chunks are generated, the FIFO fills up to 3 and then evicts the
    oldest before pushing the newest.

Usage (single-GPU per-rank mode, mirrors ``eval_causal_AR.py``)::

  CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 LOCAL_RANK=0 \\
      python utils/eval_causal_AR_chain.py \\
          --student_ckpt logs/action_ode_distill_10h/latest_0001000.pt \\
          --config       configs/action_ode_distill.yaml \\
          --rank_zarr    20240216101235.zarr \\
          --rank_offset  100 \\
          --rank_mode    dataset \\
          --output_dir   eval/eval_AR_chain_<ts>/gpu0_... \\
          --fifo_size    3

This script only implements the AR-refresh rollout. For the plain AR
and chain baselines, use ``eval_causal_AR.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_AF_ROOT = _REPO_ROOT / "action-forcing"
if str(_AF_ROOT) not in sys.path:
    sys.path.insert(0, str(_AF_ROOT))

from utils.eval_chain import (
    NUM_ACTION_CHUNKS,
    NUM_FRAME_PER_BLOCK,
    NUM_FRAMES,
    RAW_ACTION_DIM,
    STREAM_LATENT_SPAN,
    EVAL_LATENT_START_OFFSET,
    VIDEO_NOISE_BASE,
    DEFAULT_CAPTION_ROOT,
    DEFAULT_ENCODED_ROOT,
    frame_actions_to_chunk_actions,
    annotate_video,
    CRITIC_ACTION_DIMS,
    frames_to_mp4,
)
from utils.eval_causal_AR import (
    ODEChainPipeline,
    load_per_rank_ride_ar,
    FRAME_SPATIAL_TOKENS,
    BASE_CHUNK_FRAMES,
    DEFAULT_DENOISING_STEPS,
    _initialize_kv_cache,
    _initialize_crossattn_cache,
    _reset_kv_cache,
    _set_attention_window,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

DEFAULT_FIFO_SIZE = 3        # last 3 committed chunks (= 9 latent frames).
DEFAULT_AR_INITIAL_CHUNKS = 1  # 1 GT chunk primed into the FIFO (matches eval_causal_AR).
DEFAULT_AR_GEN_CHUNKS = 7      # AR chunks to generate per rollout (matches eval_causal_AR).


# ---------------------------------------------------------------------------
# Pipeline: ODE student with AR-refresh rollout (chain-style K/V every pass).
# ---------------------------------------------------------------------------


class ODEARRefreshPipeline(ODEChainPipeline):
    """ODE student rollout where every denoise step recomputes context K/V.

    Inherits ``build``, ``load_checkpoint``, ``set_denoising_steps``,
    ``generate``, ``generate_ar``, ``decode_latents``,
    ``compute_teacher_visuals``, ``run_critic`` from
    :class:`ODEChainPipeline`. Adds :meth:`generate_ar_refresh`.
    """

    # ---- AR-refresh streaming generation --------------------------------
    @torch.no_grad()
    def generate_ar_refresh(
        self,
        *,
        prompt_embeds: torch.Tensor,
        noisy_fa_full: torch.Tensor,    # [1, total_frames, RAW_ACTION_DIM]
        initial_latents: torch.Tensor,   # [1, seed_frames, C, H, W] (1 GT chunk = 3 frames at default)
        num_gen_chunks: int,
        fifo_size: int = DEFAULT_FIFO_SIZE,
        context_noise_timestep: float = 0.0,
        refresh_mode: str = "per_pass",
    ) -> torch.Tensor:
        """AR rollout with per-step context K/V recomputation.

        Algorithm (per generated chunk K):
          1. Let the FIFO hold the last ``N = min(len(FIFO), fifo_size)``
             committed latent chunks — 3 frames each. The "window" for
             chunk K is ``[ctx_{K-N}, ctx_{K-N+1}, ..., ctx_{K-1}, noisy_K]``
             — ``(N+1) * 3`` frames total. At init the FIFO holds 1 GT
             chunk, so chunk 0 sees a 6-frame window; chunk 1 sees 9;
             chunk 2 onwards sees 12 (fifo_size=3 steady state).
          2. Sample ``noisy_K`` from N(0, I) as a fresh 3-frame latent.
          3. For each ``t`` in ``denoising_step_list`` (noisiest →
             cleanest):
               * Concatenate FIFO latents and the current (possibly
                 renoised) ``noisy_K`` into a ``(N+1)*3`` frame
                 sequence.
               * Per-frame timestep: ``context_noise_timestep`` for the
                 FIFO frames, ``t`` for the 3 current frames.
               * Build per-frame action modulation + action tokens from
                 ``noisy_fa_full`` at the corresponding temporal slice.
               * Forward through the student with
                 ``_prepare_blockwise_causal_attn_mask`` (set each time
                 the window size changes).
               * Extract ``pred_x0`` for the last 3 frames — the chunk
                 K prediction.
               * If not the last denoise step, renoise the current
                 chunk via ``scheduler.add_noise`` at the next ``t``.
          4. Commit: evict oldest FIFO entry (if len == fifo_size),
             then push ``pred_x0`` for chunk K.

        Returns ``[1, seed_frames + num_gen_chunks*3, C, H, W]`` with
        the GT seed at the front (to match ``eval_causal_AR`` 's
        ``generate_ar`` output layout).
        """
        assert self.ode_model is not None and self.wrapper is not None
        assert self.denoising_step_list is not None, "call set_denoising_steps() first"
        assert fifo_size >= 1
        if refresh_mode not in ("per_pass", "per_chunk"):
            raise SystemExit(
                f"AR_refresh: unknown refresh_mode={refresh_mode!r}; "
                "expected 'per_pass' or 'per_chunk'."
            )

        base_dit = self.wrapper.model
        if hasattr(base_dit, "get_base_model"):
            base_dit = base_dit.get_base_model()

        B = 1
        num_frame_per_block = BASE_CHUNK_FRAMES  # 3
        max_window_blocks = fifo_size + 1        # e.g. 4 for fifo_size=3
        max_window_frames = max_window_blocks * num_frame_per_block  # 12

        # Both paths patch ``base_dit.num_frame_per_block``; ``per_pass``
        # additionally rebuilds ``base_dit.block_mask`` per window-size
        # change (driven inside ``_run_ar_refresh``). For ``per_chunk``
        # we keep ``block_mask=None`` — ``_forward_inference`` does not
        # consume the train-time block_mask (it relies on the kv_cache
        # structure + causal flag in plain flash attention).
        base_dit.num_frame_per_block = num_frame_per_block
        action_tokens_per_frame = int(getattr(base_dit, "action_tokens_per_frame", 1))
        frame_seq_length = FRAME_SPATIAL_TOKENS + action_tokens_per_frame

        # wrapper.seq_len bounds the max sequence length the model can
        # produce — must accommodate the full window (``per_pass``) or
        # a single 3-frame chunk (``per_chunk`` uses cache so each
        # wrapper call only sees its own 3-frame queries).
        if refresh_mode == "per_pass":
            required_tokens = max_window_frames * frame_seq_length
        else:
            required_tokens = num_frame_per_block * frame_seq_length
        self.wrapper.seq_len = max(int(self.wrapper.seq_len), required_tokens)

        prev_local_attn_size = getattr(base_dit, "local_attn_size", -1)
        prev_max_attention_size = getattr(base_dit, "max_attention_size", None)

        if refresh_mode == "per_pass":
            # Set local_attn_size to -1 (global attention within the
            # window). ``_prepare_blockwise_causal_attn_mask`` honours
            # ``local_attn_size``; for a ``max_window_blocks``-block
            # window we want every block to see everything allowed by
            # the block-causal structure, so global is fine.
            base_dit.local_attn_size = -1
            for _, module in base_dit.named_modules():
                if hasattr(module, "local_attn_size"):
                    try:
                        module.local_attn_size = -1
                    except Exception:
                        pass
        else:
            # ``per_chunk`` uses the kv_cache path. We size the cache
            # and the attention window to exactly the steady-state
            # window (``max_window_frames`` frames = N context + 1 cur).
            # ``_set_attention_window`` propagates these to every
            # attention block so ``_forward_inference``'s window-clip
            # arithmetic does the right thing.
            _set_attention_window(
                base_dit,
                local_attn_size_frames=max_window_frames,
                max_tokens=max_window_frames * frame_seq_length,
            )
            # ``_forward_inference`` ignores ``block_mask`` (relies on
            # kv_cache + causal flag), but clear it defensively so
            # stale state from a prior ``per_pass`` call doesn't leak.
            base_dit.block_mask = None

        try:
            if refresh_mode == "per_pass":
                return self._run_ar_refresh(
                    prompt_embeds=prompt_embeds,
                    noisy_fa_full=noisy_fa_full,
                    initial_latents=initial_latents,
                    num_gen_chunks=num_gen_chunks,
                    fifo_size=fifo_size,
                    context_noise_timestep=context_noise_timestep,
                    base_dit=base_dit,
                    num_frame_per_block=num_frame_per_block,
                    frame_seq_length=frame_seq_length,
                    max_window_blocks=max_window_blocks,
                )
            return self._run_ar_refresh_cached(
                prompt_embeds=prompt_embeds,
                noisy_fa_full=noisy_fa_full,
                initial_latents=initial_latents,
                num_gen_chunks=num_gen_chunks,
                fifo_size=fifo_size,
                context_noise_timestep=context_noise_timestep,
                base_dit=base_dit,
                num_frame_per_block=num_frame_per_block,
                frame_seq_length=frame_seq_length,
                max_window_frames=max_window_frames,
            )
        finally:
            base_dit.local_attn_size = prev_local_attn_size
            if prev_max_attention_size is not None:
                base_dit.max_attention_size = prev_max_attention_size
            for _, module in base_dit.named_modules():
                if hasattr(module, "local_attn_size"):
                    try:
                        module.local_attn_size = prev_local_attn_size
                    except Exception:
                        pass
                if hasattr(module, "max_attention_size") and prev_max_attention_size is not None:
                    try:
                        module.max_attention_size = prev_max_attention_size
                    except Exception:
                        pass
            # Invalidate the block mask we built — chain/AR paths expect
            # to own it.
            base_dit.block_mask = None

    def _run_ar_refresh(
        self,
        *,
        prompt_embeds: torch.Tensor,
        noisy_fa_full: torch.Tensor,
        initial_latents: torch.Tensor,
        num_gen_chunks: int,
        fifo_size: int,
        context_noise_timestep: float,
        base_dit,
        num_frame_per_block: int,
        frame_seq_length: int,
        max_window_blocks: int,
    ) -> torch.Tensor:
        """Core loop; split out so the wrapper can handle attn-window
        save/restore via ``try/finally``."""
        B = 1
        scheduler = self.scheduler
        scheduler.sigmas = scheduler.sigmas.to(self.device)
        ts = self.denoising_step_list

        seed_frames = int(initial_latents.shape[1])
        if seed_frames != num_frame_per_block:
            raise SystemExit(
                f"AR_refresh: expected exactly 1 GT chunk "
                f"({num_frame_per_block} frames) as the seed, got {seed_frames} "
                "frames. Adjust --ar_initial_chunks if you want more."
            )

        C = int(initial_latents.shape[2])
        H = int(initial_latents.shape[3])
        W = int(initial_latents.shape[4])

        total_frames = seed_frames + num_gen_chunks * num_frame_per_block
        if noisy_fa_full.shape[1] < total_frames:
            raise SystemExit(
                f"AR_refresh: noisy_fa_full has {noisy_fa_full.shape[1]} frames; "
                f"need {total_frames} (seed={seed_frames} + gen={num_gen_chunks}*"
                f"{num_frame_per_block})."
            )

        noisy_fa_full = noisy_fa_full.to(device=self.device, dtype=self.dtype)
        prompt_embeds_dev = prompt_embeds.to(device=self.device, dtype=self.dtype)
        initial_latents_dev = initial_latents.to(device=self.device, dtype=self.dtype)

        # FIFO of committed chunks. Each entry: [B, 3, C, H, W]. The
        # GT seed is entry 0; subsequent generated chunks are appended.
        # When ``len(fifo) > fifo_size`` the oldest entry is evicted
        # after a new chunk is committed (matches the user-specified
        # "evict, push, then denoise next" ordering — we do "denoise,
        # commit, evict-if-full, push" which is equivalent because the
        # FIFO state observed by chunk K+1 is the same either way).
        fifo: List[torch.Tensor] = [initial_latents_dev]

        # We rebuild ``base_dit.block_mask`` any time the window size
        # (= current number of context blocks + 1) changes. Once the
        # FIFO is full the size is fixed.
        current_window_blocks = -1

        log.info(
            "[AR_refresh] seed=%d frames  gen=%d chunks  fifo=%d chunks  "
            "max_window=%d blocks (%d frames)  t_ctx=%.3f",
            seed_frames, num_gen_chunks, fifo_size, max_window_blocks,
            max_window_blocks * num_frame_per_block, context_noise_timestep,
        )

        generated: List[torch.Tensor] = []

        seed_chunks = seed_frames // num_frame_per_block

        for chunk_idx in range(num_gen_chunks):
            n_ctx_blocks = min(len(fifo), fifo_size)
            window_blocks = n_ctx_blocks + 1
            window_frames = window_blocks * num_frame_per_block

            # Current chunk's absolute global temporal position. Generated
            # chunk K (0-indexed) begins at global frame
            # ``(seed_chunks + K) * num_frame_per_block`` — this is
            # independent of FIFO saturation (``len(fifo)`` plateaus at
            # ``fifo_size`` once full, so we can't use it as a counter).
            cur_global_frame_lo = (seed_chunks + chunk_idx) * num_frame_per_block
            cur_global_frame_hi = cur_global_frame_lo + num_frame_per_block
            ctx_global_frame_lo = cur_global_frame_lo - n_ctx_blocks * num_frame_per_block
            assert ctx_global_frame_lo >= 0, (
                ctx_global_frame_lo, cur_global_frame_lo, n_ctx_blocks,
            )

            # Rebuild block mask if window size changed (only happens
            # during warm-up while FIFO is filling).
            if window_blocks != current_window_blocks:
                base_dit.block_mask = None  # force rebuild next forward
                current_window_blocks = window_blocks
                log.info(
                    "[AR_refresh] chunk %d: rebuilding block_mask for "
                    "%d-block window (%d context + 1 current)",
                    chunk_idx, window_blocks, n_ctx_blocks,
                )

            # Assemble the context latents (FIFO, oldest->newest, taking
            # last n_ctx_blocks). Each FIFO entry is [B, 3, C, H, W].
            ctx_chunks = fifo[-n_ctx_blocks:]
            ctx_cat = torch.cat(ctx_chunks, dim=1)  # [B, n_ctx*3, C, H, W]
            assert ctx_cat.shape[1] == n_ctx_blocks * num_frame_per_block

            # Frame-actions: the 12-frame window's actions come from
            # ``noisy_fa_full`` at the temporal positions of the
            # context + current chunks. (Both context and current use
            # the "noisy" branch here — there's only one stream.)
            fa_window = noisy_fa_full[:, ctx_global_frame_lo:cur_global_frame_hi].contiguous()
            assert fa_window.shape[1] == window_frames, (
                fa_window.shape[1], window_frames,
            )

            # Per-frame timestep: context at t_ctx, current at the live t.
            t_ctx_vec = torch.full(
                [B, n_ctx_blocks * num_frame_per_block],
                float(context_noise_timestep),
                device=self.device, dtype=torch.float32,
            )

            # Sample fresh noise for the current chunk. One draw per
            # chunk; renoised between the 4 denoise steps.
            current_noise = torch.randn(
                [B, num_frame_per_block, C, H, W],
                dtype=torch.float32, device=self.device,
            )
            x_cur = current_noise.to(self.dtype)

            pred_x0_window: Optional[torch.Tensor] = None
            for d_idx in range(int(ts.shape[0])):
                t_val = float(ts[d_idx].item())
                t_cur_vec = torch.full(
                    [B, num_frame_per_block], t_val,
                    device=self.device, dtype=torch.float32,
                )
                tt = torch.cat([t_ctx_vec, t_cur_vec], dim=1)  # [B, window_frames]

                # Build the 12-frame noisy input. Context frames at t=0
                # act as "clean" input (the model emits pred_x0 ≈ input
                # for those), and their K/V are recomputed from raw
                # latents at every transformer layer of every denoise
                # step. Current frames are the evolving noisy_K.
                x_full = torch.cat([ctx_cat, x_cur], dim=1)  # [B, window_frames, C, H, W]

                # Build conditional dict using the "noisy" branch keys
                # only — this is what the plain (kv_cache=None,
                # clean_x=None) wrapper path consumes. Action
                # projection processes per-frame, so passing
                # ``window_frames`` actions is fine.
                cond = self._build_action_cond_chunk(
                    prompt_embeds_dev, fa_window, num_frames=window_frames,
                )

                with torch.amp.autocast("cuda", dtype=self.dtype):
                    out = self.wrapper(
                        noisy_image_or_video=x_full,
                        conditional_dict=cond,
                        timestep=tt,
                        clean_x=None,
                        aug_t=None,
                    )
                # wrapper returns (flow_pred, pred_x0) for the plain path.
                pred_x0_window = out[1]  # [B, window_frames, C, H, W]

                if d_idx < int(ts.shape[0]) - 1:
                    next_t = float(ts[d_idx + 1].item())
                    # Renoise ONLY the current chunk's 3 frames.
                    cur_pred_x0 = pred_x0_window[:, n_ctx_blocks * num_frame_per_block:]
                    flat = cur_pred_x0.flatten(0, 1).float()
                    flat_noise = torch.randn_like(flat)
                    flat_t = torch.full(
                        (flat.shape[0],), next_t,
                        device=self.device, dtype=torch.float32,
                    )
                    x_cur = (
                        scheduler.add_noise(flat, flat_noise, flat_t)
                        .view(B, num_frame_per_block, C, H, W)
                        .to(self.dtype)
                    )

            assert pred_x0_window is not None
            cur_pred = pred_x0_window[:, n_ctx_blocks * num_frame_per_block:]
            generated.append(cur_pred.detach().to(torch.float32))

            # Commit: evict oldest if FIFO is at max, then push new.
            if len(fifo) >= fifo_size:
                fifo.pop(0)
            fifo.append(cur_pred.to(self.dtype))

            log.info(
                "[AR_refresh] chunk %d/%d committed | window=%d blocks "
                "(%d ctx + 1 cur) | global frames [%d:%d] | fifo_len=%d",
                chunk_idx + 1, num_gen_chunks, window_blocks,
                n_ctx_blocks, cur_global_frame_lo, cur_global_frame_hi,
                len(fifo),
            )

        # Assemble: GT seed ++ generated chunks.
        seed_real = initial_latents_dev[:, :seed_frames].to(torch.float32)
        full = torch.cat([seed_real] + generated, dim=1)
        expected = seed_frames + num_gen_chunks * num_frame_per_block
        assert int(full.shape[1]) == expected, (full.shape, expected)
        torch.cuda.empty_cache()
        return full

    def _run_ar_refresh_cached(
        self,
        *,
        prompt_embeds: torch.Tensor,
        noisy_fa_full: torch.Tensor,
        initial_latents: torch.Tensor,
        num_gen_chunks: int,
        fifo_size: int,
        context_noise_timestep: float,
        base_dit,
        num_frame_per_block: int,
        frame_seq_length: int,
        max_window_frames: int,
    ) -> torch.Tensor:
        """AR_refresh with LongLive-style per-CHUNK cache rebuild.

        For each to-be-generated chunk:
          1. Zero the kv_cache (``_reset_kv_cache``).
          2. Walk the FIFO oldest->newest and, for each 3-frame entry,
             call the inference forward at ``t=context_noise_timestep``
             with ``current_start`` advancing by 3 frames. After N
             refresh forwards the cache holds fresh K/V for
             ``cache[0 : 3N]``, block-causal across context blocks
             (each block's K/V is computed with the prior blocks'
             freshly-cached K/V in scope).
          3. Run the 4 denoise passes as 3-frame inference forwards at
             ``current_start = 3N * fsl``. Pass 1 appends the current
             chunk's K/V into ``cache[3N : 3N+3]``; passes 2-4 trigger
             the ``is_recompute`` branch and overwrite those 3 positions
             with fresh K/V from the re-noised queries.
          4. Commit ``pred_x0`` of the current chunk into the FIFO.

        Per-chunk cost: ``N + 4`` forwards of 3 frames each. For the
        default ``fifo_size=3`` steady state: 7 small forwards/chunk,
        matching ``eval_causal_AR.py --ar_cache_refresh full_fifo``.
        """
        B = 1
        scheduler = self.scheduler
        scheduler.sigmas = scheduler.sigmas.to(self.device)
        ts = self.denoising_step_list

        seed_frames = int(initial_latents.shape[1])
        if seed_frames != num_frame_per_block:
            raise SystemExit(
                f"AR_refresh (cached): expected exactly 1 GT chunk "
                f"({num_frame_per_block} frames) as the seed, got "
                f"{seed_frames} frames. Adjust --ar_initial_chunks if "
                "you want more."
            )

        C = int(initial_latents.shape[2])
        H = int(initial_latents.shape[3])
        W = int(initial_latents.shape[4])

        total_frames = seed_frames + num_gen_chunks * num_frame_per_block
        if noisy_fa_full.shape[1] < total_frames:
            raise SystemExit(
                f"AR_refresh (cached): noisy_fa_full has "
                f"{noisy_fa_full.shape[1]} frames; need {total_frames} "
                f"(seed={seed_frames} + gen={num_gen_chunks}*"
                f"{num_frame_per_block})."
            )

        noisy_fa_full = noisy_fa_full.to(device=self.device, dtype=self.dtype)
        prompt_embeds_dev = prompt_embeds.to(device=self.device, dtype=self.dtype)
        initial_latents_dev = initial_latents.to(device=self.device, dtype=self.dtype)

        # Allocate kv_cache + crossattn_cache. Size: the steady-state
        # window (``max_window_frames`` frames = fifo_size+1 blocks).
        num_transformer_blocks = len(base_dit.blocks)
        kv_cache_tokens = max_window_frames * frame_seq_length
        kv_cache = _initialize_kv_cache(
            num_transformer_blocks=num_transformer_blocks,
            batch_size=B,
            kv_cache_size_tokens=kv_cache_tokens,
            dtype=self.dtype,
            device=self.device,
        )
        crossattn_cache = _initialize_crossattn_cache(
            num_transformer_blocks=num_transformer_blocks,
            batch_size=B,
            dtype=self.dtype,
            device=self.device,
        )

        # FIFO of committed chunks; entries are [B, 3, C, H, W]. The
        # seed chunk is entry 0.
        fifo: List[torch.Tensor] = [initial_latents_dev]

        refresh_t_block = torch.full(
            [B, num_frame_per_block], float(context_noise_timestep),
            device=self.device, dtype=torch.float32,
        )

        seed_chunks = seed_frames // num_frame_per_block

        log.info(
            "[AR_refresh][per_chunk] seed=%d frames  gen=%d chunks  "
            "fifo=%d chunks  window=%d frames  kv_cache_tokens=%d  "
            "t_ctx=%.3f",
            seed_frames, num_gen_chunks, fifo_size, max_window_frames,
            kv_cache_tokens, context_noise_timestep,
        )

        generated: List[torch.Tensor] = []

        for chunk_idx in range(num_gen_chunks):
            n_ctx_blocks = min(len(fifo), fifo_size)
            window_blocks = n_ctx_blocks + 1

            cur_global_frame_lo = (seed_chunks + chunk_idx) * num_frame_per_block
            cur_global_frame_hi = cur_global_frame_lo + num_frame_per_block
            ctx_global_frame_lo = cur_global_frame_lo - n_ctx_blocks * num_frame_per_block
            assert ctx_global_frame_lo >= 0, (
                ctx_global_frame_lo, cur_global_frame_lo, n_ctx_blocks,
            )

            # -------- 1) Rebuild the cache from the FIFO ---------
            _reset_kv_cache(kv_cache)
            # Rebuild crossattn too to guarantee fresh-prompt K/V (same
            # prompt, but be defensive; LongLive does the same).
            for entry in crossattn_cache:
                entry["is_init"] = False

            current_start_frame = 0
            # Iterate FIFO oldest->newest. Each entry advances
            # ``current_start_frame`` by ``num_frame_per_block``.
            ctx_chunks = fifo[-n_ctx_blocks:] if n_ctx_blocks > 0 else []
            for block_idx, f_lat in enumerate(ctx_chunks):
                f_lo = ctx_global_frame_lo + block_idx * num_frame_per_block
                f_hi = f_lo + num_frame_per_block
                f_block_fa = noisy_fa_full[:, f_lo:f_hi]
                f_cond = self._build_action_cond_chunk(
                    prompt_embeds_dev, f_block_fa,
                    num_frames=num_frame_per_block,
                )
                with torch.amp.autocast("cuda", dtype=self.dtype):
                    self.wrapper(
                        noisy_image_or_video=f_lat,
                        conditional_dict=f_cond,
                        timestep=refresh_t_block,
                        kv_cache=kv_cache,
                        crossattn_cache=crossattn_cache,
                        current_start=current_start_frame * frame_seq_length,
                    )
                current_start_frame += num_frame_per_block
            # ``current_start_frame`` is now ``n_ctx_blocks*npb`` — the
            # position where the current chunk's 3 frames will live.

            # -------- 2) Denoise the current chunk ---------------
            block_fa = noisy_fa_full[:, cur_global_frame_lo:cur_global_frame_hi]
            cond = self._build_action_cond_chunk(
                prompt_embeds_dev, block_fa,
                num_frames=num_frame_per_block,
            )

            current_noise = torch.randn(
                [B, num_frame_per_block, C, H, W],
                dtype=torch.float32, device=self.device,
            )
            x = current_noise.to(self.dtype)

            pred_x0: Optional[torch.Tensor] = None
            for d_idx in range(int(ts.shape[0])):
                t_val = float(ts[d_idx].item())
                tt = torch.full(
                    [B, num_frame_per_block], t_val,
                    device=self.device, dtype=torch.float32,
                )
                with torch.amp.autocast("cuda", dtype=self.dtype):
                    out = self.wrapper(
                        noisy_image_or_video=x,
                        conditional_dict=cond,
                        timestep=tt,
                        kv_cache=kv_cache,
                        crossattn_cache=crossattn_cache,
                        current_start=current_start_frame * frame_seq_length,
                    )
                pred_x0 = out[1]  # [B, 3, C, H, W]

                if d_idx < int(ts.shape[0]) - 1:
                    next_t = float(ts[d_idx + 1].item())
                    flat = pred_x0.flatten(0, 1).float()
                    flat_noise = torch.randn_like(flat)
                    flat_t = torch.full(
                        (flat.shape[0],), next_t,
                        device=self.device, dtype=torch.float32,
                    )
                    x = (
                        scheduler.add_noise(flat, flat_noise, flat_t)
                        .view(B, num_frame_per_block, C, H, W)
                        .to(self.dtype)
                    )

            assert pred_x0 is not None
            generated.append(pred_x0.detach().to(torch.float32))

            # -------- 3) Commit to FIFO --------------------------
            if len(fifo) >= fifo_size:
                fifo.pop(0)
            fifo.append(pred_x0.detach().to(self.dtype))

            log.info(
                "[AR_refresh][per_chunk] chunk %d/%d committed | "
                "window=%d blocks (%d ctx + 1 cur) | global frames "
                "[%d:%d] | fifo_len=%d",
                chunk_idx + 1, num_gen_chunks, window_blocks,
                n_ctx_blocks, cur_global_frame_lo, cur_global_frame_hi,
                len(fifo),
            )

        # Assemble: GT seed ++ generated chunks.
        seed_real = initial_latents_dev[:, :seed_frames].to(torch.float32)
        full = torch.cat([seed_real] + generated, dim=1)
        expected = seed_frames + num_gen_chunks * num_frame_per_block
        assert int(full.shape[1]) == expected, (full.shape, expected)

        del kv_cache, crossattn_cache
        torch.cuda.empty_cache()
        return full


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="eval/eval_AR_chain_out")
    parser.add_argument("--config", type=str, default="configs/action_ode_distill.yaml",
                        help="ODE-distillation config matching the student checkpoint.")
    parser.add_argument("--student_ckpt", type=str, required=True,
                        help="Path to the ODE-distilled student snapshot.")
    parser.add_argument("--manifest", type=str,
                        default="logs/z_critic_v10_state_tokens/.ride_manifest.pt",
                        help="Optional ride manifest for fast lookup.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--denoising_steps", type=int, default=DEFAULT_DENOISING_STEPS,
                        help="Student denoising iterations per chunk (default 4). "
                             "Match the trained pool length for best quality.")
    parser.add_argument("--label", type=str, default=None,
                        help="Output-subdir / video-title tag. Defaults to "
                             "'ode_student_step<S>_<K>step_ar_refresh'.")

    # ---- AR-refresh knobs ----
    parser.add_argument("--fifo_size", type=int, default=DEFAULT_FIFO_SIZE,
                        help="Number of committed 3-frame chunks retained as the "
                             "context FIFO (default 3). The current chunk sees the "
                             "last ``fifo_size`` committed chunks, re-projected from "
                             "raw latents at every denoise step.")
    parser.add_argument("--ar_initial_chunks", type=int, default=DEFAULT_AR_INITIAL_CHUNKS,
                        help="GT chunks primed into the FIFO at init (default 1 — "
                             "matches eval_causal_AR). Values >1 are allowed but "
                             "change the GT context length.")
    parser.add_argument("--ar_gen_chunks", type=int, default=DEFAULT_AR_GEN_CHUNKS,
                        help="Chunks to generate autoregressively (default 7 = 21 frames).")
    parser.add_argument("--context_noise_timestep", type=float, default=0.0,
                        help="Per-frame timestep applied to context (FIFO) frames in "
                             "the joint forward. Training uses 0. Matches AR's "
                             "cache-refresh t.")
    parser.add_argument("--refresh_mode", choices=["per_pass", "per_chunk"],
                        default="per_pass",
                        help="How often context K/V are recomputed from FIFO latents. "
                             "'per_pass' (default, the original AR_refresh recipe) "
                             "does one 12-frame train-path forward per denoise step "
                             "(4 per chunk). 'per_chunk' (Recommendation #1, "
                             "LongLive-inspired) rebuilds a real kv_cache once per "
                             "chunk via N sequential 3-frame refresh forwards, then "
                             "runs the 4 denoise passes as 3-frame inference forwards "
                             "that reuse the cached context K/V. Much faster but "
                             "shares numerics with "
                             "'eval_causal_AR.py --ar_cache_refresh full_fifo'.")
    parser.add_argument("--infinity_rope", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="When --refresh_mode=per_chunk, install LongLive-style "
                             "Block-Relativistic RoPE (Infinity-RoPE) on the KV cache "
                             "path: un-roped K stored in cache, Q/K rotated with "
                             "bounded window-relative indices at attention time. "
                             "Default: read 'infinity_rope' from the config (true if "
                             "absent). No effect when --refresh_mode=per_pass (that "
                             "path doesn't use kv_cache).")

    # ---- Per-rank data-path overrides ----
    parser.add_argument("--motion_root", type=str, default=None,
                        help="Override the config's motion_root (required).")
    parser.add_argument("--ss_vae_checkpoint", type=str, default=None,
                        help="Override the config's ss_vae_checkpoint path.")

    # ---- Per-rank ride selection (matches eval_causal_AR) ----
    parser.add_argument("--rank_zarr", type=str, required=True,
                        help="Zarr basename for THIS rank.")
    parser.add_argument("--rank_offset", type=int, required=True,
                        help="latent_start_offset for THIS rank.")
    parser.add_argument("--rank_mode", choices=["dataset", "counterfactual"], required=True,
                        help="'dataset' = noisy = clean = dataset z-actions. "
                             "'counterfactual' = noisy = -dataset.")
    parser.add_argument("--rank_tag", type=str, default=None,
                        help="Optional short tag to embed in output filenames.")
    parser.add_argument("--encoded_root", type=str, default=DEFAULT_ENCODED_ROOT,
                        help="Fallback encoded-zarr root for rides not in manifest.")
    parser.add_argument("--caption_root", type=str, default=DEFAULT_CAPTION_ROOT,
                        help="Caption root (for prompt embeds + ts→ride_dir map).")
    args = parser.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    if world != 1:
        log.warning(
            "AR_refresh is designed for single-process-per-GPU launches "
            "(WORLD_SIZE=1). world=%d may produce unexpected results.", world,
        )

    # Load cfg for motion_root / ss_vae_checkpoint / action_dims.
    from omegaconf import OmegaConf
    _cfg = OmegaConf.load(args.config)
    motion_root = args.motion_root or str(_cfg.get("motion_root", "") or "")
    ss_vae_ckpt = args.ss_vae_checkpoint or str(
        _cfg.get("ss_vae_checkpoint", "action_query/checkpoints/ss_vae_8free.pt")
    )
    action_dims = list(_cfg.get("action_dims", [2, 7]))
    if args.infinity_rope is None:
        infinity_rope_enabled = bool(_cfg.get("infinity_rope", True))
    else:
        infinity_rope_enabled = bool(args.infinity_rope)
    log.info(
        "[config] motion_root=%s  ss_vae_ckpt=%s  action_dims=%s  infinity_rope=%s",
        motion_root, ss_vae_ckpt, action_dims, infinity_rope_enabled,
    )
    if not motion_root:
        raise SystemExit(
            "AR_refresh needs motion_root; set it via --motion_root. Example: "
            "--motion_root /projects/u6ex/fbots/frodobots_motion"
        )

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Ride loading — fetch the full AR temporal span.
    ar_total_frames = (args.ar_initial_chunks + args.ar_gen_chunks) * BASE_CHUNK_FRAMES
    (
        initial_latents_full,
        ar_prompt_embeds,
        ar_noisy_fa_full,
        ride_meta,
    ) = load_per_rank_ride_ar(
        zarr_basename=args.rank_zarr,
        latent_start_offset=args.rank_offset,
        total_frames=ar_total_frames,
        manifest_path=args.manifest,
        encoded_root=args.encoded_root,
        caption_root=args.caption_root,
        motion_root=motion_root,
        ss_vae_checkpoint=ss_vae_ckpt,
        action_dims=action_dims,
        device=device,
    )
    ride_meta.update({
        "rank_mode": args.rank_mode,
        "rank_tag": args.rank_tag or "",
        "ar_initial_chunks": args.ar_initial_chunks,
        "ar_gen_chunks": args.ar_gen_chunks,
        "fifo_size": args.fifo_size,
        "mode": "ar_refresh",
        "refresh_mode": args.refresh_mode,
        "infinity_rope": infinity_rope_enabled,
    })

    meta_path = out_root / f"rank{rank}_ride.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(ride_meta, fh, indent=2, default=str)

    # Build pipeline + load student.
    pipe = ODEARRefreshPipeline(device)
    pipe.build(args.config, use_action_tokens=True)
    student_step = pipe.load_checkpoint(args.student_ckpt)
    pipe.set_denoising_steps(args.denoising_steps)

    step_tag = str(student_step) if student_step >= 0 else "unknown"
    label = args.label or (
        f"ode_student_step{step_tag}_{args.denoising_steps}step_ar_refresh_"
        f"{args.refresh_mode}"
    )

    if args.rank_tag:
        out_dir = out_root / f"rank{rank}_{label}_{args.rank_tag}"
    else:
        out_dir = out_root / f"rank{rank}_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "Rank %d/%d: ar_refresh | student=%s (step=%s) | denoising_steps=%d | "
        "refresh_mode=%s | fifo=%d | gen_chunks=%d | initial_chunks=%d | out=%s",
        rank, world, args.student_ckpt, student_step, args.denoising_steps,
        args.refresh_mode, args.fifo_size, args.ar_gen_chunks,
        args.ar_initial_chunks, out_dir.name,
    )

    pe_dtype = pipe.dtype
    ar_prompt_embeds = ar_prompt_embeds.to(dtype=pe_dtype)
    clean_fa_ar = ar_noisy_fa_full.to(dtype=pe_dtype)
    if args.rank_mode == "dataset":
        noisy_fa_ar = clean_fa_ar.clone()
        cond_tag = "dataset"
    elif args.rank_mode == "counterfactual":
        noisy_fa_ar = -clean_fa_ar
        cond_tag = "counterfactual"
    else:
        raise SystemExit(f"Unknown rank_mode {args.rank_mode!r}")

    prefill_frames = args.ar_initial_chunks * BASE_CHUNK_FRAMES
    prefill_lat = initial_latents_full[:, :prefill_frames].to(dtype=pe_dtype)

    log.info(
        "[AR_refresh][%s] rank=%d mode=%s  prefill_frames=%d  gen_chunks=%d  "
        "fifo_size=%d  total_frames=%d  (z2 [%.3f,%.3f], z7 [%.3f,%.3f])",
        label, rank, cond_tag, prefill_frames, args.ar_gen_chunks, args.fifo_size,
        prefill_frames + args.ar_gen_chunks * BASE_CHUNK_FRAMES,
        float(clean_fa_ar[..., 0].min()), float(clean_fa_ar[..., 0].max()),
        float(clean_fa_ar[..., 1].min()), float(clean_fa_ar[..., 1].max()),
    )

    noise_seed = args.seed + VIDEO_NOISE_BASE
    torch.manual_seed(noise_seed)
    torch.cuda.manual_seed(noise_seed)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated() / (1024 ** 2)

    from utils.infinity_rope import infinity_rope_active
    base_dit_ar = pipe.wrapper.model
    if hasattr(base_dit_ar, "get_base_model"):
        base_dit_ar = base_dit_ar.get_base_model()
    # ``per_pass`` doesn't use kv_cache (it runs the train forward with
    # full re-projection every step), so the patch is a no-op there.
    rerope_for_run = infinity_rope_enabled and (args.refresh_mode == "per_chunk")

    t0 = time.time()
    with infinity_rope_active(rerope_for_run, base_dit_ar):
        full_latents = pipe.generate_ar_refresh(
            prompt_embeds=ar_prompt_embeds,
            noisy_fa_full=noisy_fa_ar,
            initial_latents=prefill_lat,
            num_gen_chunks=args.ar_gen_chunks,
            fifo_size=args.fifo_size,
            context_noise_timestep=args.context_noise_timestep,
            refresh_mode=args.refresh_mode,
        )
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
    log.info(
        "[AR_refresh][%s] rank=%d rollout done in %.1fs (K=%d, fifo=%d, "
        "refresh_mode=%s, infinity_rope=%s, seed=%d) | mem before=%.0f MiB  "
        "peak_alloc=%.0f MiB  peak_reserved=%.0f MiB",
        label, rank, time.time() - t0, args.denoising_steps, args.fifo_size,
        args.refresh_mode, rerope_for_run, noise_seed, mem_before, peak_mb,
        reserved_mb,
    )

    full_latents = full_latents.detach()
    gen_latents = full_latents[:, prefill_frames:]
    context_latents = full_latents[:, :prefill_frames]

    video_np = pipe.decode_latents(gen_latents)
    context_np = pipe.decode_latents(context_latents)

    motion, teacher_z_8d = pipe.compute_teacher_visuals(gen_latents)
    n_c = teacher_z_8d.shape[1]
    teacher_z2z7 = teacher_z_8d[:, :, CRITIC_ACTION_DIMS]
    chunk_dev_gen = frame_actions_to_chunk_actions(noisy_fa_ar[:, prefill_frames:])
    critic_z2z7 = None
    if pipe.action_critic is not None:
        cp = pipe.run_critic(gen_latents, chunk_dev_gen)
        if cp is not None:
            critic_z2z7 = cp[:, :, CRITIC_ACTION_DIMS]
    target_z_ar = chunk_dev_gen[:, :n_c].contiguous()

    rerope_tag = "_rerope" if rerope_for_run else ""
    tag = (
        f"{label}_{cond_tag}_ar_refresh_{args.refresh_mode}_fifo{args.fifo_size}"
        f"_init{args.ar_initial_chunks}{rerope_tag}"
    )
    raw_path = out_dir / f"{tag}_rollout_raw.mp4"
    annot_path = out_dir / f"{tag}_rollout_annotated.mp4"

    full_rollout_np = np.concatenate([context_np, video_np], axis=0)
    frames_to_mp4(full_rollout_np, str(raw_path), fps=20)

    ann = annotate_video(
        video_np, teacher_z2z7, critic_z2z7, target_z_ar, motion,
        f"{tag}  rank{rank}",
    )
    ann_cat = np.concatenate([context_np, ann], axis=0)
    frames_to_mp4(ann_cat, str(annot_path), fps=20)

    log.info(
        "[AR_refresh][%s] rank=%d wrote %s and %s (shape=%s)",
        label, rank, raw_path.name, annot_path.name, full_rollout_np.shape,
    )

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    log.info("[AR_refresh][%s] rank=%d done.", label, rank)


if __name__ == "__main__":
    main()
