#!/usr/bin/env python3
"""Autoregressive (ODE few-step student) variant of ``eval_causal_chain.py``.

Same chained rolling-context eval protocol as ``eval_causal_chain.py``, but:

* DiT is the **full-rank ODE-distilled student** (no PEFT wrap at eval
  time; the teacher's LoRA was already ``merge_and_unload()``-ed at
  distillation time, and the student trained the full 1.3B-param DiT).
* Generation uses the **student's few-step denoise loop** (default 4
  steps, same pattern as ``ODERegression.generate_eval`` and
  ``bin/bench_streaming_inference.py``), re-noising ``pred_x0`` via
  ``FlowMatchScheduler.add_noise`` between iterations.

Checkpoint format (matches ``action-forcing/bin/bench_streaming_inference.py``):
  * ``generator``: full-rank ``CausalWanModel.state_dict()``. Strict
    overlay onto the teacher-merged base built by ``ODERegression``.
  * ``action_projection`` / ``action_token_projection``: full-rank.
  * ``step`` (optional, informational).

Usage (single-GPU per-rank mode, mirrors ``eval_causal_chain.py``)::

  CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 LOCAL_RANK=0 \\
      python utils/eval_causal_AR.py \\
          --student_ckpt logs/action_ode_distill_10h/latest_0001000.pt \\
          --config       configs/action_ode_distill.yaml \\
          --rank_zarr    20240216101235.zarr \\
          --rank_offset  100 \\
          --rank_mode    dataset \\
          --output_dir   eval/eval_causal_AR_<ts>/gpu0_... \\
          --denoising_steps 4

The non-ODE control script (``eval_causal_chain.py``) runs the teacher
at 48 FlowMatch steps. This one runs the student at ``--denoising_steps``
(default 4), so the expected speedup per video is ~12×.
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
# ``af_model`` and the streaming helpers live under ``action-forcing/``.
_AF_ROOT = _REPO_ROOT / "action-forcing"
if str(_AF_ROOT) not in sys.path:
    sys.path.insert(0, str(_AF_ROOT))

# Dependencies reused wholesale: the chain math, ride loaders, and all
# annotation / VAE / cotracker / ss_vae / critic helpers. Only the DiT
# and its denoising loop change vs. ``eval_causal_chain.py``.
from utils.eval_chain import (
    ChainPipeline,
    NUM_ACTION_CHUNKS,
    NUM_FRAME_PER_BLOCK,
    NUM_FRAMES,
    RAW_ACTION_DIM,
    STREAM_LATENT_SPAN,
    EVAL_LATENT_START_OFFSET,
    VIDEO_NOISE_BASE,
    VIDEO_NOISE_SEED_STRIDE,
    DEFAULT_CAPTION_ROOT,
    DEFAULT_ENCODED_ROOT,
    frame_actions_to_chunk_actions,
    annotate_video,
    CRITIC_ACTION_DIMS,
    frames_to_mp4,
)
from utils.eval_causal_chain import (
    NUM_CAUSAL_VIDEOS,
    FIXED_ACTION_Z2,
    FIXED_ACTION_Z7,
    build_causal_clean_x,
    build_rollout_latents,
    load_per_rank_ride,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

DEFAULT_DENOISING_STEPS = 4     # student was trained at this budget (CF pool + tail).
DEFAULT_CACHE_CHUNKS = 12       # max KV cache budget that fits with grads on (bench 4150099).
DEFAULT_AR_INITIAL_CHUNKS = 3   # real-data prefill chunks for AR mode.
DEFAULT_AR_GEN_CHUNKS = 7       # AR-generated chunks per rollout.
FRAME_SPATIAL_TOKENS = 1560     # tokens per latent frame at 60x104 spatial (matches bench).
BASE_CHUNK_FRAMES = NUM_FRAME_PER_BLOCK  # 3


# ---------------------------------------------------------------------------
# KV cache + attention-window helpers (copied inline from
# ``bin/bench_streaming_inference.py`` to keep this file self-contained;
# see that script for the full rationale and the streaming benchmark path
# they were originally written for).
# ---------------------------------------------------------------------------


def _initialize_kv_cache(
    *,
    num_transformer_blocks: int,
    batch_size: int,
    kv_cache_size_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
) -> list:
    kv_cache = []
    for _ in range(num_transformer_blocks):
        kv_cache.append({
            "k": torch.zeros(
                [batch_size, kv_cache_size_tokens, 12, 128],
                dtype=dtype, device=device,
            ),
            "v": torch.zeros(
                [batch_size, kv_cache_size_tokens, 12, 128],
                dtype=dtype, device=device,
            ),
            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index":  torch.tensor([0], dtype=torch.long, device=device),
        })
    return kv_cache


def _initialize_crossattn_cache(
    *,
    num_transformer_blocks: int,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> list:
    crossattn_cache = []
    for _ in range(num_transformer_blocks):
        crossattn_cache.append({
            "k":       torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
            "v":       torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
            "is_init": False,
        })
    return crossattn_cache


def _reset_kv_cache(kv_cache: list) -> None:
    """Zero out every per-layer K/V buffer and rewind the end-index
    counters so the cache behaves like a freshly allocated one.

    Used by the ``cache_refresh="full_fifo"`` path in ``generate_ar``,
    which rebuilds the entire KV cache from raw FIFO latents at every
    chunk boundary. Also clears any ``_frozen_*_end_index`` attributes
    that may have been left dangling on per-block attention modules by
    earlier forwards (these are used by the train-time "freeze cache"
    hack in ``CausalWanModel`` and would otherwise override the reset
    values on the next forward).
    """
    for entry in kv_cache:
        entry["k"].zero_()
        entry["v"].zero_()
        entry["global_end_index"].fill_(0)
        entry["local_end_index"].fill_(0)


def _set_attention_window(base_dit, *, local_attn_size_frames: int, max_tokens: int) -> None:
    """Propagate ``local_attn_size`` (frames) and ``max_attention_size``
    (tokens) to every attention block + top-level module."""
    base_dit.local_attn_size = local_attn_size_frames
    base_dit.max_attention_size = max_tokens
    for _, module in base_dit.named_modules():
        if hasattr(module, "max_attention_size"):
            try:
                module.max_attention_size = max_tokens
            except Exception:
                pass
        if hasattr(module, "local_attn_size"):
            try:
                module.local_attn_size = local_attn_size_frames
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Pipeline: ODE student on top of the teacher-merged base.
# ---------------------------------------------------------------------------


class ODEChainPipeline(ChainPipeline):
    """ChainPipeline variant that runs a full-rank ODE-distilled student.

    * ``build()`` constructs the DiT via ``ODERegression`` (teacher's LoRA
      is merged in-place during ``ODERegression.__init__``) and disables
      its internal motion pipeline — we build VAE/CoTracker/ss_vae
      separately, identical to the teacher ``ChainPipeline``.
    * ``load_checkpoint()`` overlays the student's ``generator`` +
      ``action_projection`` + ``action_token_projection`` tensors (strict
      on the DiT, non-strict on the action heads so older checkpoints
      without them still load).
    * ``generate()`` runs ``self.denoising_step_list`` in noisiest-to-
      cleanest order, re-noising ``pred_x0`` between iterations with
      ``self.scheduler.add_noise`` (the distilled student's inference
      path). The step list can be the one baked into the checkpoint's
      config (via ``ODERegression.denoising_step_list``) or an explicit
      user-specified count expanded to ``torch.linspace(1000, 50, N)``.

    Inherited unchanged: ``build_conditional``, ``decode_latents``,
    ``compute_teacher_visuals``, ``run_critic``,
    ``_attach_shared_v12_eval_critic``.
    """

    def __init__(self, device, dtype=torch.bfloat16):
        super().__init__(device, dtype=dtype)
        self.ode_model = None
        self.scheduler = None
        self.denoising_step_list: Optional[torch.Tensor] = None
        self._student_step: int = -1

    # ---- build --------------------------------------------------------
    def build(  # type: ignore[override]
        self,
        config_path: str = "configs/action_ode_distill.yaml",
        *,
        use_action_tokens: bool = True,  # kept for CLI parity; always True for ODE student.
    ) -> None:
        from omegaconf import OmegaConf
        from utils.wan_wrapper import WanVAEWrapper

        log.info("[%s] Loading config %s", self.device, config_path)
        cfg = OmegaConf.load(config_path)
        OmegaConf.set_struct(cfg, False)

        # ODERegression constructs: wrapper (WanDiffusionWrapper), action
        # heads, state_probe, action_critic, scheduler — and in-place
        # merges the teacher's LoRA via ``PeftModel.merge_and_unload``.
        log.info("[%s] Building ODERegression (teacher merge)...", self.device)
        from af_model.ode_regression import ODERegression
        self.ode_model = ODERegression(cfg, device=self.device).eval()
        # Never build ODERegression's internal motion pipeline — we build
        # VAE/CoTracker/ss_vae ourselves, matching ``ChainPipeline``.
        self.ode_model.use_motion_pipeline = False

        self.wrapper = self.ode_model.generator
        self.action_projection = self.ode_model.action_projection
        self.action_token_projection = self.ode_model.action_token_projection
        self.scheduler = self.ode_model.scheduler
        # Default step list = whatever the config resolved. load_checkpoint /
        # CLI override replaces this.
        self.denoising_step_list = self.ode_model.denoising_step_list.detach().clone()

        if not use_action_tokens:
            log.warning(
                "[%s] use_action_tokens=False requested, but the ODE student "
                "was trained with action tokens. Forcing True.", self.device,
            )

        # Chain eval runs at 21-frame ``NUM_FRAMES``. The ODE student also
        # trains at 21 frames, so wrapper.seq_len is already correct;
        # nothing to adjust here (unlike ``ChainPipeline.build``).

        log.info("[%s] Loading frozen VAE, CoTracker, ss_vae...", self.device)
        self.vae = WanVAEWrapper(); self.vae.to(self.device).eval()
        self.cotracker = torch.hub.load(
            "facebookresearch/co-tracker", "cotracker2", skip_validation=True,
        ).to(self.device).eval()
        ss_ckpt = cfg.get("ss_vae_checkpoint", "action_query/checkpoints/ss_vae_8free.pt")
        from action_query.ss_vae_model import load_ss_vae
        self.ss_vae, self.ss_vae_scale = load_ss_vae(ss_ckpt, device=str(self.device))

        # Shared v12 eval critic (drawn on annotated video; same as ChainPipeline).
        self._attach_shared_v12_eval_critic()
        log.info("[%s] ODEChainPipeline ready.", self.device)

    # ---- load_checkpoint ---------------------------------------------
    def load_checkpoint(self, student_ckpt: str, vinfo=None):  # type: ignore[override]
        """Overlay the student weights onto the teacher-merged base.

        ``vinfo`` is ignored (kept for signature parity with ``ChainPipeline``).
        Returns the embedded ``step`` from the checkpoint (or -1).
        """
        log.info("[%s] Overlaying student %s", self.device, student_ckpt)
        raw = torch.load(student_ckpt, map_location="cpu", weights_only=False)
        try:
            generator_sd = raw["generator"]
        except KeyError as exc:
            raise RuntimeError(
                f"Student checkpoint {student_ckpt} missing 'generator' key; "
                "expected a full-rank trainer snapshot from action_ode_distill."
            ) from exc

        self.ode_model.generator.model.load_state_dict(generator_sd, strict=True)

        if (
            self.ode_model.action_projection is not None
            and "action_projection" in raw
        ):
            self.ode_model.action_projection.load_state_dict(raw["action_projection"])
        if (
            self.ode_model.action_token_projection is not None
            and "action_token_projection" in raw
        ):
            self.ode_model.action_token_projection.load_state_dict(raw["action_token_projection"])

        self.ode_model.to(self.device).eval()

        step = int(raw.get("step", -1))
        self._student_step = step
        log.info(
            "[%s] ODE student overlay OK (embedded step=%s).",
            self.device, step if step >= 0 else "?",
        )
        del raw
        return step

    # ---- denoising step list -----------------------------------------
    def set_denoising_steps(self, n: int) -> torch.Tensor:
        """Build a length-``n`` descending timestep tensor.

        Stays inside the student's TRAINED pool whenever possible —
        feeding the model timesteps it never saw at distillation
        (e.g. ``linspace(1000, 50, 4)`` = ``[1000, 683, 367, 50]``)
        produces sub-distribution outputs because the timestep
        embedding is OOD.

        Resolution order:
          1. ``n == len(trained)``: use the trained list verbatim.
          2. ``1 <= n < len(trained)``: pick the ``n`` LARGEST trained
             timesteps in descending order. With the default
             ``random_steps: [0, 36, 40, 44, 46]`` (= timesteps
             ``[1000, 625, 500, 312.5, 178.6]``) and ``n=4`` this
             returns ``[1000, 625, 500, 312.5]`` — the canonical
             4-step CF chunkwise ladder mapped onto the 48-step
             FlowMatch(shift=5) grid the student was distilled on.
             The "polish" step (178.6) is dropped first because
             it's the lowest-noise / smallest-quality-delta entry.
          3. ``n > len(trained)``: fall back to
             ``linspace(1000, 50, n)`` (matches
             ``bin/bench_streaming_inference.py``); flag in the log
             that we're feeding OOD timesteps to the student.
        """
        assert self.ode_model is not None
        trained = self.ode_model.denoising_step_list.detach().to(
            device=self.device, dtype=torch.float32,
        )
        trained, _ = torch.sort(trained, descending=True)
        K = int(trained.shape[0])
        if n == K:
            ts = trained
            src = "trained"
        elif 1 <= n < K:
            ts = trained[:n].contiguous()
            src = f"trained_top{n}_of_{K}"
        else:
            ts = torch.linspace(1000.0, 50.0, steps=n, device=self.device)
            src = "linspace(1000,50)__OOD"
            log.warning(
                "[%s] Denoising step count %d > trained pool size %d; "
                "falling back to linspace(1000, 50, %d). The student was "
                "NOT distilled at these timesteps; expect quality drop.",
                self.device, n, K, n,
            )
        self.denoising_step_list = ts
        log.info(
            "[%s] Denoising schedule (%s, %d steps): %s",
            self.device, src, int(ts.shape[0]),
            [round(float(x), 2) for x in ts.tolist()],
        )
        return ts

    # ---- generate -----------------------------------------------------
    @torch.no_grad()
    def generate(self, conditional, clean_x):  # type: ignore[override]
        """Few-step student rollout of ``NUM_FRAMES`` latents.

        Mirrors ``ODERegression.generate_eval`` and the inner loop of
        ``bin/bench_streaming_inference.py``. ``clean_x`` is the 21-frame
        teacher-forced attention context built by ``build_causal_clean_x``.
        """
        assert self.denoising_step_list is not None, "call set_denoising_steps() first"
        scheduler = self.scheduler
        scheduler.sigmas = scheduler.sigmas.to(self.device)
        ts = self.denoising_step_list

        B = clean_x.shape[0]
        C, H, W = clean_x.shape[2], clean_x.shape[3], clean_x.shape[4]
        x = torch.randn(
            [B, NUM_FRAMES, C, H, W], dtype=torch.float32, device=self.device,
        ).to(self.dtype)
        clean_x_cast = clean_x.to(self.dtype)

        pred_x0: Optional[torch.Tensor] = None
        for i in range(int(ts.shape[0])):
            t = float(ts[i].item())
            tt = torch.full(
                (B, NUM_FRAMES), t, device=self.device, dtype=torch.float32,
            )
            with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                out = self.wrapper(
                    noisy_image_or_video=x,
                    conditional_dict=conditional,
                    timestep=tt,
                    clean_x=clean_x_cast,
                    aug_t=None,
                )
            pred_x0 = out[1]
            if i < int(ts.shape[0]) - 1:
                next_t = float(ts[i + 1].item())
                flat = pred_x0.flatten(0, 1).float()
                flat_noise = torch.randn_like(flat)
                flat_t = torch.full(
                    (flat.shape[0],), next_t,
                    device=self.device, dtype=torch.float32,
                )
                x = (
                    scheduler.add_noise(flat, flat_noise, flat_t)
                    .view(B, NUM_FRAMES, C, H, W)
                    .to(self.dtype)
                )
        assert pred_x0 is not None
        return pred_x0.float()

    # ---- AR streaming conditioning helper ----------------------------
    def _build_action_cond_chunk(
        self,
        prompt_embeds: torch.Tensor,
        noisy_fa_chunk: torch.Tensor,
        num_frames: int,
    ) -> Dict[str, torch.Tensor]:
        """Conditional dict for a single streaming block (KV-cache path).

        Streaming inference only uses the "noisy branch" conditioning keys
        (``_action_modulation`` / ``_action_tokens``), never the ``_clean``
        variants — there is no separate clean context branch at AR time,
        the context lives in the KV cache. Matches
        ``bin/bench_streaming_inference._build_zero_conditional``.
        """
        cond: Dict[str, torch.Tensor] = {
            "prompt_embeds": prompt_embeds.to(dtype=self.dtype, device=self.device),
        }
        if self.action_projection is not None:
            cond["_action_modulation"] = self.action_projection(
                noisy_fa_chunk, num_frames=num_frames,
            )
        if self.action_token_projection is not None:
            cond["_action_tokens"] = self.action_token_projection(noisy_fa_chunk)
        return cond

    # ---- AR streaming generation -------------------------------------
    @torch.no_grad()
    def generate_ar(
        self,
        *,
        prompt_embeds: torch.Tensor,
        noisy_fa_full: torch.Tensor,   # [1, total_frames, RAW_ACTION_DIM]
        initial_latents: torch.Tensor,  # [1, initial_frames, C, H, W] — real-data prefill (only used when ar_cache=False)
        num_gen_chunks: int,
        cache_chunks: int = DEFAULT_CACHE_CHUNKS,
        chunks_per_step: int = 1,
        context_noise_timestep: float = 0.0,
        ar_cache: bool = True,
        cache_refresh: str = "append",
    ) -> torch.Tensor:
        """Streaming AR rollout with a KV cache.

        ``cache_refresh`` selects how the KV cache is maintained as
        new chunks are committed:

          * ``"append"`` (default) — original AR-baseline behaviour.
            After each chunk is denoised, a single cache-refresh forward
            pushes the newly committed chunk's K/V into the next free
            slot of the ring buffer. K/V for older context chunks are
            computed ONCE (when each was first committed) and never
            touched again. Over the course of the rollout the cache
            holds progressively staler K/V for older context.

          * ``"full_fifo"`` — chain-style freshness at chunk boundary
            (Variant A). The cache is treated as a FIFO of the last
            ``cache_chunks`` committed latents. At the *start* of every
            new chunk the cache is zero'd and re-populated from the FIFO
            by running a clean cache-refresh forward for each FIFO
            entry in temporal order. Every chunk's 4 denoise passes
            therefore see K/V that were freshly computed, in sequence,
            from the *current* FIFO state — exactly what chain does at
            the context-K/V level, but done once per chunk rather than
            once per denoise step (so the 4 denoise passes share the
            refreshed cache, unlike ``eval_causal_AR_chain.py``'s
            ``generate_ar_refresh`` which recomputes on every step).
            Costs roughly 1.75× of a 3-frame forward per chunk at
            steady state (3 refresh + 4 denoise passes, each at
            Q=3 frames).

        **Chain-equivalent real seed (always on).** Regardless of
        ``ar_cache`` or ``cache_refresh``, the FIRST block of
        ``initial_latents`` (= the first ``num_frame_per_block`` latent
        frames, which is 3 at the canonical ``chunks_per_step=1`` —
        matching ``eval_causal_chain`` 's ``CONTEXT_FRAMES`` /
        ``seed_lat``) is ALWAYS pushed into the KV cache through a
        clean cache-refresh forward at ``context_noise_timestep``. This
        guarantees the student sees the same 3 real seed frames that
        chain evaluates with, which is essential for head-to-head
        AR-vs-chain comparisons.

        After the mandatory real seed, ``ar_cache`` controls what
        populates the rest of the cache:

          * ``ar_cache=True`` (default) — **Generated-fill**: any prefill
            slots beyond the first block are filled by AR-generated
            "bootstrap" chunks (so the cache holds ``[real seed] + [gen
            KVs]``). With ``ar_initial_chunks == chunks_per_step`` there
            are zero bootstrap chunks and we go straight from real seed
            to main rollout — the simplest realistic deployment setting.

          * ``ar_cache=False`` — **Clean-fill**: prefill slots beyond the
            first block are pushed through cache-refresh with the
            remaining real ``initial_latents`` (oracle context KVs), so
            the cache holds ``initial_frames`` worth of real KV before
            AR-gen starts. Tells us how much the student benefits from
            clean ground-truth context beyond the seed.

        Algorithm:
          1. Configure the DiT for block size ``chunks_per_step * 3`` and
             attention window ``(cache_chunks + chunks_per_step) * 3``
             frames. Allocate KV + cross-attn caches sized for the window.
          2. **Seed-prefill (always):** cache-refresh the first block of
             ``initial_latents`` at ``context_noise_timestep``. Advance
             ``current_start_frame`` by one block.
          3. Depending on ``ar_cache``:
               * ``True``  → set ``bootstrap_chunks = initial_chunks - seed_chunks``
                             (additional AR-generated chunks before main).
               * ``False`` → cache-refresh the remaining real
                             ``initial_latents`` blocks, advancing
                             ``current_start_frame`` past them.
                             ``bootstrap_chunks = 0``.
          4. AR-generate ``bootstrap_chunks + num_gen_chunks`` chunks.
             Each streaming step:
               a. Sample fresh noise for the block.
               b. Run the student's ``denoising_step_list`` (noisiest →
                  cleanest), re-noising pred_x0 via ``scheduler.add_noise``
                  between iterations.
               c. Cache-refresh on the final clean ``pred_x0`` at
                  ``context_noise_timestep`` so the cache holds clean KVs
                  for the new block.
               d. Advance ``current_start_frame`` by ``num_frame_per_block``.
          5. Return ``[1, total_frames, C, H, W]`` where
             ``total_frames = initial_frames + num_gen_chunks * 3``.
             In both modes the first ``num_frame_per_block`` frames are
             the real zarr seed. Beyond that: real (clean-fill) or
             AR-generated (generated-fill).

        ``noisy_fa_full`` must cover the *entire* temporal range
        (prefill/bootstrap + main-gen). All forwards use the slice of
        ``noisy_fa_full`` corresponding to the frames being processed.
        """
        assert self.ode_model is not None and self.wrapper is not None
        assert self.denoising_step_list is not None, "call set_denoising_steps() first"
        assert chunks_per_step >= 1
        if cache_refresh not in ("append", "full_fifo"):
            raise SystemExit(
                f"[AR] unknown cache_refresh={cache_refresh!r} "
                "(expected 'append' or 'full_fifo')"
            )
        if cache_refresh == "full_fifo" and not ar_cache:
            raise SystemExit(
                "[AR] cache_refresh='full_fifo' is incompatible with "
                "--no-ar_cache (clean-fill). full_fifo always rebuilds "
                "the cache from a FIFO of committed latents, so the "
                "cache-fill strategy must be generated-fill."
            )

        base_dit = self.wrapper.model
        if hasattr(base_dit, "get_base_model"):
            base_dit = base_dit.get_base_model()

        B = 1
        num_frame_per_block = chunks_per_step * BASE_CHUNK_FRAMES

        initial_frames = int(initial_latents.shape[1])
        if initial_frames % num_frame_per_block != 0:
            raise SystemExit(
                f"AR prefill: initial_frames={initial_frames} is not a multiple of "
                f"num_frame_per_block={num_frame_per_block} (chunks_per_step={chunks_per_step})."
            )
        if num_gen_chunks % chunks_per_step != 0:
            raise SystemExit(
                f"AR gen: num_gen_chunks={num_gen_chunks} is not a multiple of "
                f"chunks_per_step={chunks_per_step}. Use whole streaming steps."
            )

        # Attention-window / seq_len bookkeeping (mirrors bench_streaming_inference).
        base_dit.num_frame_per_block = num_frame_per_block
        base_dit.block_mask = None
        action_tokens_per_frame = int(getattr(base_dit, "action_tokens_per_frame", 1))
        frame_seq_length = FRAME_SPATIAL_TOKENS + action_tokens_per_frame
        local_attn_size_frames = (cache_chunks + chunks_per_step) * BASE_CHUNK_FRAMES
        kv_cache_tokens = local_attn_size_frames * frame_seq_length
        required_chunk_tokens = num_frame_per_block * frame_seq_length
        self.wrapper.seq_len = max(int(self.wrapper.seq_len), required_chunk_tokens)
        _set_attention_window(
            base_dit,
            local_attn_size_frames=local_attn_size_frames,
            max_tokens=kv_cache_tokens,
        )

        num_transformer_blocks = len(base_dit.blocks)
        total_frames = initial_frames + num_gen_chunks * BASE_CHUNK_FRAMES
        if noisy_fa_full.shape[1] < total_frames:
            raise SystemExit(
                f"AR: noisy_fa_full has only {noisy_fa_full.shape[1]} frames; "
                f"need {total_frames} (={initial_frames} prefill + "
                f"{num_gen_chunks} * {BASE_CHUNK_FRAMES} generated)."
            )

        C = int(initial_latents.shape[2])
        H = int(initial_latents.shape[3])
        W = int(initial_latents.shape[4])

        log.info(
            "[AR] prefill=%d frames  gen=%d chunks  cache_chunks=%d  "
            "chunks_per_step=%d  local_attn=%d frames  kv_tokens=%d  "
            "wrapper.seq_len=%d",
            initial_frames, num_gen_chunks, cache_chunks, chunks_per_step,
            local_attn_size_frames, kv_cache_tokens, int(self.wrapper.seq_len),
        )

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

        scheduler = self.scheduler
        scheduler.sigmas = scheduler.sigmas.to(self.device)
        ts = self.denoising_step_list

        noisy_fa_full = noisy_fa_full.to(device=self.device, dtype=self.dtype)
        prompt_embeds_dev = prompt_embeds.to(device=self.device, dtype=self.dtype)
        initial_latents_dev = initial_latents.to(device=self.device, dtype=self.dtype)

        current_start_frame = 0
        refresh_t_block = torch.full(
            [B, num_frame_per_block], float(context_noise_timestep),
            device=self.device, dtype=torch.float32,
        )

        # ------------------------------------------------------------------
        # 1) Cache prefill.
        #
        # ALWAYS prefill the FIRST block (``num_frame_per_block`` frames —
        # typically the 3 frames of ``CONTEXT_FRAMES``) of ``initial_latents``
        # into the KV cache via a clean cache-refresh forward at
        # ``context_noise_timestep``. This is the chain-equivalent seed:
        # the same real latents that ``eval_causal_chain`` feeds as
        # ``seed_lat`` into its non-streaming causal rollout. Without this
        # the AR student generates its first chunk from a completely
        # empty cache, which is strictly worse than what chain sees and
        # makes AR-vs-chain comparisons unfair.
        #
        # After the mandatory real seed:
        #
        #   ``ar_cache=False`` (clean-fill) → additionally push the
        #       remaining ``initial_frames - num_frame_per_block`` real
        #       frames through cache-refresh forwards so the whole
        #       ``initial_latents`` span holds clean ground-truth KVs.
        #
        #   ``ar_cache=True``  (default, generated-fill) → leave the rest
        #       of the prefill span empty; we'll AR-generate
        #       ``bootstrap_chunks`` chunks below before starting the main
        #       rollout. Cache after bootstrap: [real seed] + [gen KVs].
        # ------------------------------------------------------------------
        initial_chunks = initial_frames // BASE_CHUNK_FRAMES
        if initial_frames < num_frame_per_block:
            raise SystemExit(
                f"AR: initial_frames={initial_frames} < num_frame_per_block="
                f"{num_frame_per_block}; need at least one full block of "
                "real seed latents (chain-equivalent CONTEXT_FRAMES)."
            )

        # --- always: real-seed prefill for the first block ---
        seed_lo, seed_hi = 0, num_frame_per_block
        seed_lat_block = initial_latents_dev[:, seed_lo:seed_hi]
        seed_fa_block = noisy_fa_full[:, seed_lo:seed_hi]
        seed_cond = self._build_action_cond_chunk(
            prompt_embeds_dev, seed_fa_block, num_frames=num_frame_per_block,
        )
        with torch.amp.autocast("cuda", dtype=self.dtype):
            self.wrapper(
                noisy_image_or_video=seed_lat_block,
                conditional_dict=seed_cond,
                timestep=refresh_t_block,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start_frame * frame_seq_length,
            )
        current_start_frame += num_frame_per_block
        seed_chunks = num_frame_per_block // BASE_CHUNK_FRAMES
        log.info(
            "[AR] real-seed prefill: %d frame(s) = %d chunk(s) from zarr "
            "pushed into KV cache (chain-equivalent seed, t=%.3f).",
            num_frame_per_block, seed_chunks, float(context_noise_timestep),
        )

        if ar_cache:
            # After the mandatory real seed, the remaining prefill span is
            # filled by AR-generated chunks (bootstrap) so the cache holds
            # [real seed] + [gen KVs] going into the main rollout.
            bootstrap_chunks = initial_chunks - seed_chunks
            log.info(
                "[AR] cache-fill strategy: GENERATED (real seed: %d chunk(s)"
                " + bootstrap: %d chunk(s) AR-gen + main: %d chunk(s) AR-gen).",
                seed_chunks, bootstrap_chunks, num_gen_chunks,
            )
        else:
            # Clean-fill: push the rest of initial_latents through
            # cache-refresh as well, so the whole initial_frames span is
            # real KV in the cache.
            bootstrap_chunks = 0
            for frame_lo in range(num_frame_per_block, initial_frames, num_frame_per_block):
                frame_hi = frame_lo + num_frame_per_block
                block_lat = initial_latents_dev[:, frame_lo:frame_hi]
                block_fa = noisy_fa_full[:, frame_lo:frame_hi]
                cond = self._build_action_cond_chunk(
                    prompt_embeds_dev, block_fa, num_frames=num_frame_per_block,
                )
                with torch.amp.autocast("cuda", dtype=self.dtype):
                    self.wrapper(
                        noisy_image_or_video=block_lat,
                        conditional_dict=cond,
                        timestep=refresh_t_block,
                        kv_cache=kv_cache,
                        crossattn_cache=crossattn_cache,
                        current_start=current_start_frame * frame_seq_length,
                    )
                current_start_frame += num_frame_per_block
            log.info(
                "[AR] cache-fill strategy: CLEAN (real seed: %d chunk(s)"
                " + real prefill: %d chunk(s) + main: %d chunk(s) AR-gen).",
                seed_chunks, initial_chunks - seed_chunks, num_gen_chunks,
            )

        # ------------------------------------------------------------------
        # 2) AR generation.
        #    ``total_gen_chunks`` covers the bootstrap (if ar_cache=True)
        #    plus the main-gen; each streaming step emits ``chunks_per_step``
        #    chunks.
        #
        #    Behaviour branches on ``cache_refresh``:
        #      * ``"append"`` — incremental KV cache (original AR logic).
        #                       Each chunk's cache-refresh writes only the
        #                       newly committed chunk's K/V, appended at
        #                       ``current_start`` which advances by
        #                       ``num_frame_per_block`` per chunk.
        #      * ``"full_fifo"`` — at the START of every chunk, zero the
        #                         cache and re-run refresh forwards for
        #                         each entry of a FIFO of the last
        #                         ``cache_chunks`` committed latent
        #                         blocks, in temporal order. The FIFO is
        #                         seeded with the real seed block and
        #                         grows one chunk per step until it
        #                         saturates; thereafter the oldest entry
        #                         is evicted before the newest pred_x0 is
        #                         pushed. The 4 denoise passes of a chunk
        #                         share the refreshed cache — only the 3
        #                         current-chunk frames get fresh K/V per
        #                         denoise step.
        # ------------------------------------------------------------------
        generated: List[torch.Tensor] = []

        # FIFO state used only by cache_refresh="full_fifo". Entries are
        # (latent_block [B, num_frame_per_block, C, H, W], global frame
        # lo index into noisy_fa_full). The seed-prefill above already
        # pushed the first block into the cache; full_fifo does not
        # reuse those K/V — it rebuilds the cache from raw latents at
        # the start of every chunk — but the seed latents themselves
        # go into the FIFO as the initial context for chunk 0.
        fifo_lat: List[torch.Tensor] = []
        fifo_frame_lo: List[int] = []
        if cache_refresh == "full_fifo":
            # Seed the FIFO with every block of ``initial_latents`` that
            # was already pushed into the cache above (seed + clean-fill
            # bootstrap). For ``ar_cache=True`` that is just the first
            # block (the chain-equivalent 3-frame seed); ``ar_cache=False``
            # is explicitly rejected at entry.
            fifo_lat.append(initial_latents_dev[:, :num_frame_per_block])
            fifo_frame_lo.append(0)
            log.info(
                "[AR] cache-fill strategy: FULL_FIFO (FIFO size=%d chunks, "
                "seed=%d chunk(s), main=%d chunk(s) AR-gen; cache is "
                "rebuilt from raw FIFO latents before every chunk).",
                cache_chunks, seed_chunks, num_gen_chunks,
            )

        total_gen_chunks = bootstrap_chunks + num_gen_chunks
        if total_gen_chunks % chunks_per_step != 0:
            raise SystemExit(
                f"AR gen: total_gen_chunks={total_gen_chunks} "
                f"(bootstrap={bootstrap_chunks} + main={num_gen_chunks}) is "
                f"not a multiple of chunks_per_step={chunks_per_step}."
            )
        num_steps_needed = total_gen_chunks // chunks_per_step
        for step_idx in range(num_steps_needed):
            if cache_refresh == "full_fifo":
                # Rebuild the cache from raw FIFO latents. Each block is
                # refreshed in temporal order so the K/V at higher
                # transformer layers see freshly-refreshed K/V of their
                # causal predecessors — matching what a single
                # full-window chain forward would produce.
                _reset_kv_cache(kv_cache)
                current_start_frame = 0
                for f_lat, f_lo in zip(fifo_lat, fifo_frame_lo):
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
                # current_start_frame is now exactly where the incoming
                # chunk's 3 frames will be queried from.
                cur_frame_lo = (seed_chunks + step_idx) * num_frame_per_block
            else:
                cur_frame_lo = current_start_frame

            frame_lo = cur_frame_lo
            frame_hi = frame_lo + num_frame_per_block
            block_fa = noisy_fa_full[:, frame_lo:frame_hi]
            cond = self._build_action_cond_chunk(
                prompt_embeds_dev, block_fa, num_frames=num_frame_per_block,
            )

            noise = torch.randn(
                [B, num_frame_per_block, C, H, W],
                dtype=torch.float32, device=self.device,
            )
            x = noise.to(self.dtype)

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
                pred_x0 = out[1]
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

            if cache_refresh == "append":
                # Cache-refresh on the clean pred_x0 so the next step sees
                # clean-context KVs (matches CausalInferencePipeline).
                with torch.amp.autocast("cuda", dtype=self.dtype):
                    self.wrapper(
                        noisy_image_or_video=pred_x0,
                        conditional_dict=cond,
                        timestep=refresh_t_block,
                        kv_cache=kv_cache,
                        crossattn_cache=crossattn_cache,
                        current_start=current_start_frame * frame_seq_length,
                    )
                current_start_frame += num_frame_per_block
                log.info(
                    "[AR] step %d/%d emitted chunks [%d:%d] at frames [%d:%d]",
                    step_idx + 1, num_steps_needed,
                    (current_start_frame - num_frame_per_block) // BASE_CHUNK_FRAMES,
                    current_start_frame // BASE_CHUNK_FRAMES,
                    current_start_frame - num_frame_per_block, current_start_frame,
                )
            else:  # full_fifo
                # Commit pred_x0 to the FIFO; next iteration's refresh
                # phase will re-populate the cache from raw latents. We
                # deliberately skip the AR-style cache-refresh-on-commit
                # because it would be wasted work — the next chunk
                # starts with a full reset.
                if len(fifo_lat) >= cache_chunks:
                    fifo_lat.pop(0)
                    fifo_frame_lo.pop(0)
                fifo_lat.append(pred_x0.detach())
                fifo_frame_lo.append(cur_frame_lo)
                log.info(
                    "[AR/full_fifo] step %d/%d emitted chunk at frames "
                    "[%d:%d] | fifo=%d/%d",
                    step_idx + 1, num_steps_needed,
                    cur_frame_lo, cur_frame_lo + num_frame_per_block,
                    len(fifo_lat), cache_chunks,
                )

        if ar_cache:
            # Real seed block (chain-equivalent CONTEXT_FRAMES) followed
            # by bootstrap + main AR-generated chunks. The real seed is
            # the same latents chain uses; everything after is the
            # model's own output.
            seed_real = initial_latents_dev[:, :num_frame_per_block].to(torch.float32)
            full = torch.cat([seed_real] + generated, dim=1)
        else:
            # Full real prefill latents + AR-generated main portion.
            full = torch.cat(
                [initial_latents_dev.to(torch.float32)] + generated, dim=1,
            )
        assert int(full.shape[1]) == total_frames, (full.shape, total_frames)
        del kv_cache, crossattn_cache
        torch.cuda.empty_cache()
        return full


# ---------------------------------------------------------------------------
# AR-mode ride loader: extended temporal window.
# ---------------------------------------------------------------------------


def load_per_rank_ride_ar(
    zarr_basename: str,
    latent_start_offset: int,
    total_frames: int,
    manifest_path: Optional[str],
    encoded_root: str,
    caption_root: str,
    motion_root: str,
    ss_vae_checkpoint: str,
    action_dims: List[int],
    device: torch.device,
) -> tuple:
    """AR-mode variant of ``load_per_rank_ride`` that fetches
    ``total_frames`` of latents AND z-actions (chain mode fetches only 21).

    Required by AR streaming eval because the rollout covers
    ``initial_chunks*3 + gen_chunks*3`` temporal frames, which exceeds
    the chain mode's fixed 21-frame window. Returns ``(initial_latents,
    prompt_embeds, noisy_fa_full, meta)`` where ``initial_latents`` has
    shape ``[1, total_frames, C, H, W]`` (the AR caller slices off the
    prefill portion) and ``noisy_fa_full`` has shape
    ``[1, total_frames, len(action_dims)]``.
    """
    import zarr as zarr_lib
    from utils.zarr_dataset import ZarrRideDataset
    from utils.eval_chain import (
        _unwrap_manifest_rides,
        _count_latent_frames,
        _build_ts_to_ride_dir,
        _load_ride_entry_from_disk,
    )
    from utils.eval_causal_chain import _find_ride_in_manifest

    need = latent_start_offset + total_frames
    ride_dict: Optional[dict] = None
    zpath: Optional[str] = None
    n_lat: Optional[int] = None
    m_idx = -1

    if manifest_path and Path(manifest_path).exists():
        try:
            manifest = torch.load(manifest_path, map_location="cpu", weights_only=False)
            rides, src = _unwrap_manifest_rides(manifest)
            hit = _find_ride_in_manifest(rides, zarr_basename)
            if hit is not None:
                m_idx, cand, zpath_h, nl_h = hit
                if nl_h >= need:
                    log.info(
                        "[AR] manifest hit (src=%s, idx=%d): %s (%d latents, need %d)",
                        src, m_idx, Path(zpath_h).name, nl_h, need,
                    )
                    g = zarr_lib.open_group(zpath_h, mode="r")
                    attrs = cand.get("attrs") or dict(g.attrs)
                    ride_dict = {
                        "zarr_path": zpath_h,
                        "prompt_embeds": cand["prompt_embeds"].cpu(),
                        "attrs": attrs,
                        "n_latent_frames": nl_h,
                    }
                    zpath, n_lat = zpath_h, nl_h
                else:
                    log.warning(
                        "[AR] manifest hit has only %d latents (need %d); "
                        "falling back to disk.", nl_h, need,
                    )
            del manifest, rides
        except Exception as exc:
            log.warning("[AR] Manifest lookup failed for %s: %s", zarr_basename, exc)

    if ride_dict is None:
        log.info(
            "[AR] Disk-fallback: building ride entry for %s from encoded_root=%s",
            zarr_basename, encoded_root,
        )
        ts_map = _build_ts_to_ride_dir(Path(caption_root))
        ride_dict = _load_ride_entry_from_disk(
            zarr_basename, Path(encoded_root), Path(caption_root), ts_map,
        )
        n_lat = int(ride_dict["n_latent_frames"])
        if n_lat < need:
            raise RuntimeError(
                f"Disk ride {zarr_basename} has {n_lat} latents < required {need}."
            )
        zpath = str(ride_dict["zarr_path"])
        m_idx = -1

    assert zpath is not None and n_lat is not None and ride_dict is not None

    z_ds = ZarrRideDataset.from_manifest(
        rides_data=[ride_dict],
        motion_root=motion_root,
        ss_vae_checkpoint=ss_vae_checkpoint,
        device="cpu",
        ss_vae_device="cpu",
    )
    z_win = z_ds.encode_z_actions_window(
        zpath, n_lat, latent_start_offset, latent_start_offset + total_frames,
    )
    noisy_fa_full = z_win[:total_frames, action_dims].unsqueeze(0).float()

    g = zarr_lib.open_group(zpath, mode="r")
    lat_np = g["latents"][latent_start_offset : latent_start_offset + total_frames]
    assert lat_np.shape[0] == total_frames, lat_np.shape
    lat = torch.from_numpy(lat_np.astype(np.float32)).cpu()
    pe = ride_dict["prompt_embeds"].cpu()

    initial_latents = lat[:total_frames].unsqueeze(0).to(device)  # [1, total_frames, C, H, W]
    prompt_embeds = pe.unsqueeze(0).to(device)
    noisy_fa_full = noisy_fa_full.to(device)

    meta = {
        "manifest_idx": m_idx,
        "zarr_path": zpath,
        "n_latent_frames": n_lat,
        "latent_start_offset": latent_start_offset,
        "ar_total_frames": total_frames,
    }
    log.info(
        "[AR] Loaded %s slice [%d:%d); noisy_fa_full shape=%s",
        Path(zpath).name, latent_start_offset, latent_start_offset + total_frames,
        tuple(noisy_fa_full.shape),
    )
    return initial_latents, prompt_embeds, noisy_fa_full, meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="eval/eval_causal_AR_out")
    parser.add_argument("--config", type=str, default="configs/action_ode_distill.yaml",
                        help="ODE-distillation config whose geometry (num_frame_per_block, "
                             "action_conditioning_mode, teacher generator_ckpt path) matches "
                             "the student checkpoint.")
    parser.add_argument("--student_ckpt", type=str, required=True,
                        help="Path to the ODE-distilled student snapshot "
                             "(e.g. logs/action_ode_distill_10h/latest_0001000.pt).")
    parser.add_argument("--manifest", type=str,
                        default="logs/z_critic_v10_state_tokens/.ride_manifest.pt",
                        help="Optional: used in per-rank mode for fast ride lookup.")
    parser.add_argument("--latent_start_offset", type=int, default=EVAL_LATENT_START_OFFSET,
                        help="Legacy mode only — single offset shared across ranks.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_causal_videos", type=int, default=NUM_CAUSAL_VIDEOS,
                        help="Number of chained videos per rank (default 7).")
    parser.add_argument("--denoising_steps", type=int, default=DEFAULT_DENOISING_STEPS,
                        help="Student denoising iterations per video (default 4). "
                             "Match the trained pool length for best quality, or set to "
                             "any N to use linspace(1000, 50, N).")
    parser.add_argument("--label", type=str, default=None,
                        help="Output-subdir / video-title tag. Defaults to "
                             "'ode_student_step<S>_<K>step_<mode>' where "
                             "K=--denoising_steps and <mode>=--mode.")

    # ---- Mode selection + AR-specific knobs ----
    parser.add_argument("--mode", choices=["chain", "ar"], default="chain",
                        help="chain: original eval_causal_chain protocol — 7 separate "
                             "videos per rank, each a non-streaming 21-frame rollout with "
                             "a causal teacher-forced clean_x that swaps in committed "
                             "generated chunks in earlier slots. "
                             "ar: true autoregressive streaming inference — prefill a KV "
                             "cache with --ar_initial_chunks real-data chunks, then "
                             "AR-generate --ar_gen_chunks chunks one streaming step at a "
                             "time, pushing each into the cache.")
    parser.add_argument("--cache_chunks", type=int, default=DEFAULT_CACHE_CHUNKS,
                        help="AR mode only: max number of past 3-frame chunks in the "
                             "KV cache (default 12, the bench-verified max on one GH200 "
                             "at training-mode memory).")
    parser.add_argument("--chunks_per_step", type=int, default=1,
                        help="AR mode only: 3-frame chunks emitted per streaming forward "
                             "(default 1, matches training num_frame_per_block=3).")
    parser.add_argument("--ar_initial_chunks", type=int, default=DEFAULT_AR_INITIAL_CHUNKS,
                        help="AR mode only: real-data chunks prefilled into the KV cache "
                             "before streaming generation starts (default 3 = 9 frames).")
    parser.add_argument("--ar_gen_chunks", type=int, default=DEFAULT_AR_GEN_CHUNKS,
                        help="AR mode only: chunks to generate autoregressively "
                             "(default 7 = 21 frames).")
    parser.add_argument("--ar_cache", action=argparse.BooleanOptionalAction, default=True,
                        help="AR mode only: cache-fill strategy. "
                             "--ar_cache (default) fills the KV cache with the "
                             "model's OWN generated outputs — the first "
                             "--ar_initial_chunks chunks are bootstrap AR "
                             "generations from an empty cache; the remaining "
                             "--ar_gen_chunks are the main rollout. Most "
                             "realistic deployment setting. "
                             "--no-ar_cache prefills the cache with "
                             "--ar_initial_chunks real-data chunks from the "
                             "zarr (clean-context KVs) before AR-generating "
                             "--ar_gen_chunks chunks on top — tests how much "
                             "the student benefits from a real context prior.")
    parser.add_argument("--context_noise_timestep", type=float, default=0.0,
                        help="AR mode only: timestep passed on the post-denoise / prefill "
                             "cache-refresh forwards. Training uses 0.")
    parser.add_argument("--ar_cache_refresh", choices=["append", "full_fifo"],
                        default="append",
                        help="AR mode only: KV cache maintenance strategy. "
                             "'append' (default) = original AR behaviour: each "
                             "chunk's cache-refresh writes only the newly "
                             "committed chunk's K/V, older context K/V are "
                             "computed once and never touched. "
                             "'full_fifo' = chain-style freshness at chunk "
                             "boundary (Variant A): at the start of every "
                             "chunk the cache is zero'd and re-populated from "
                             "a FIFO of the last --cache_chunks committed "
                             "latents by running clean cache-refresh forwards "
                             "for each FIFO entry in temporal order. The 4 "
                             "denoise passes of the current chunk share this "
                             "refreshed cache. Requires --ar_cache.")

    # ---- Per-rank data-path overrides (config doesn't always define these) ----
    parser.add_argument("--motion_root", type=str, default=None,
                        help="Override the config's motion_root. Required for "
                             "per-rank / AR modes because the ODE training config "
                             "(configs/action_ode_distill.yaml) doesn't set one; "
                             "the chain eval and SS-VAE both need it. Example: "
                             "/projects/u6ex/fbots/frodobots_motion")
    parser.add_argument("--ss_vae_checkpoint", type=str, default=None,
                        help="Override the config's ss_vae_checkpoint path.")

    # ---- Per-rank mode (matches eval_causal_chain.py) ----
    parser.add_argument("--rank_zarr", type=str, default=None,
                        help="Zarr basename for THIS rank. Triggers per-rank mode.")
    parser.add_argument("--rank_offset", type=int, default=None,
                        help="latent_start_offset for THIS rank. Required with --rank_zarr.")
    parser.add_argument("--rank_mode", choices=["dataset", "counterfactual"], default=None,
                        help="Action mode for THIS rank. 'dataset' = noisy = clean = "
                             "ss_vae-encoded dataset z-actions. 'counterfactual' = noisy = "
                             "-dataset, clean = dataset.")
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

    per_rank_mode = args.rank_zarr is not None
    if per_rank_mode:
        if args.rank_offset is None or args.rank_mode is None:
            raise SystemExit("--rank_zarr requires --rank_offset and --rank_mode")
        if world != 1:
            log.warning(
                "rank_zarr mode is designed for single-process-per-GPU launches "
                "(WORLD_SIZE=1). world=%d may produce unexpected results.", world,
            )

    mode = args.mode
    if mode == "ar" and not per_rank_mode:
        raise SystemExit(
            "--mode ar requires per-rank ride config (--rank_zarr / --rank_offset "
            "/ --rank_mode). AR streaming needs a real zarr slice for the cache "
            "prefill and ride-derived dataset actions for the generated chunks."
        )

    # Load cfg once for motion_root / ss_vae_checkpoint / action_dims (per-rank mode).
    # CLI overrides win — the ODE training config doesn't set motion_root, so AR /
    # chain evals must supply it via --motion_root.
    from omegaconf import OmegaConf
    _cfg = OmegaConf.load(args.config)
    motion_root = args.motion_root or str(_cfg.get("motion_root", "") or "")
    ss_vae_ckpt = args.ss_vae_checkpoint or str(
        _cfg.get("ss_vae_checkpoint", "action_query/checkpoints/ss_vae_8free.pt")
    )
    action_dims = list(_cfg.get("action_dims", [2, 7]))
    log.info(
        "[config] motion_root=%s  ss_vae_ckpt=%s  action_dims=%s",
        motion_root, ss_vae_ckpt, action_dims,
    )

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Initialise ride-side locals the chain path will fill in; AR path uses a
    # different set (``initial_latents_30`` etc.) loaded below.
    seed_lat = zarr_clean_21 = prompt_embeds = clean_fa = None
    initial_latents_30: Optional[torch.Tensor] = None
    ar_prompt_embeds: Optional[torch.Tensor] = None
    ar_noisy_fa_full: Optional[torch.Tensor] = None

    # ---- Ride loading ----
    if mode == "ar":
        if not motion_root:
            raise SystemExit(
                "AR mode needs motion_root; set it via --motion_root "
                "(the ODE training config configs/action_ode_distill.yaml "
                "doesn't define one). Example: "
                "--motion_root /projects/u6ex/fbots/frodobots_motion"
            )
        ar_total_frames = (args.ar_initial_chunks + args.ar_gen_chunks) * BASE_CHUNK_FRAMES
        (
            initial_latents_30,
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
        ride_meta["rank_mode"] = args.rank_mode
        ride_meta["rank_tag"] = args.rank_tag or ""
        ride_meta["ar_initial_chunks"] = args.ar_initial_chunks
        ride_meta["ar_gen_chunks"] = args.ar_gen_chunks
        ride_meta["cache_chunks"] = args.cache_chunks
        ride_meta["chunks_per_step"] = args.chunks_per_step
    elif per_rank_mode:
        if not motion_root:
            raise SystemExit(
                "per-rank chain mode needs motion_root; set it via --motion_root "
                "(the ODE training config doesn't define one). Example: "
                "--motion_root /projects/u6ex/fbots/frodobots_motion"
            )
        seed_lat, zarr_clean_21, prompt_embeds, clean_fa, ride_meta = load_per_rank_ride(
            zarr_basename=args.rank_zarr,
            latent_start_offset=args.rank_offset,
            manifest_path=args.manifest,
            encoded_root=args.encoded_root,
            caption_root=args.caption_root,
            motion_root=motion_root,
            ss_vae_checkpoint=ss_vae_ckpt,
            action_dims=action_dims,
            device=device,
        )
        ride_meta["rank_mode"] = args.rank_mode
        ride_meta["rank_tag"] = args.rank_tag or ""
    else:
        # Legacy fixed-action mode: reuse eval_causal_chain's picker.
        from utils.eval_chain import _unwrap_manifest_rides
        from utils.eval_causal_chain import (
            pick_distinct_eligible_rides,
            load_one_ride_tensors,
        )
        need = args.latent_start_offset + STREAM_LATENT_SPAN
        manifest = torch.load(args.manifest, map_location="cpu", weights_only=False)
        rides, src = _unwrap_manifest_rides(manifest)
        if not rides:
            raise RuntimeError(f"No rides in manifest {args.manifest}")
        eligible = pick_distinct_eligible_rides(rides, world, need)
        if rank >= len(eligible):
            raise RuntimeError(f"LOCAL_RANK {rank} >= {len(eligible)} eligible rides")
        slot = eligible[rank]
        log.info(
            "Rank %d/%d: manifest source=%s, ride idx=%d, zarr=%s",
            rank, world, src, slot[0], Path(slot[2]).name,
        )
        seed_lat, zarr_clean_21, prompt_embeds, ride_meta = load_one_ride_tensors(
            slot, args.latent_start_offset, device,
        )
        clean_fa = None  # built from fixed values later
        ride_meta["latent_start_offset"] = args.latent_start_offset

    meta_path = out_root / f"rank{rank}_ride.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(ride_meta, fh, indent=2, default=str)

    # ---- Build pipeline + load student ----
    pipe = ODEChainPipeline(device)
    pipe.build(args.config, use_action_tokens=True)
    student_step = pipe.load_checkpoint(args.student_ckpt)
    pipe.set_denoising_steps(args.denoising_steps)

    # ---- Label (drives output subdir + on-video title) ----
    if args.label:
        label = args.label
    else:
        step_tag = str(student_step) if student_step >= 0 else "unknown"
        label = f"ode_student_step{step_tag}_{args.denoising_steps}step_{args.mode}"

    # ---- Output subdir ----
    if per_rank_mode and args.rank_tag:
        out_dir = out_root / f"rank{rank}_{label}_{args.rank_tag}"
    else:
        out_dir = out_root / f"rank{rank}_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "Rank %d/%d: mode=%s, student=%s (step=%s), denoising_steps=%d -> %s",
        rank, world, "per_rank" if per_rank_mode else "legacy_fixed",
        args.student_ckpt, student_step, args.denoising_steps, out_dir.name,
    )

    pe_dtype = pipe.dtype

    # ------------------------------------------------------------------
    # AR-mode branch: one streaming rollout per rank. Writes a single
    # 30-frame video (``ar_initial_chunks * 3`` real context + gen) to
    # disk and exits before the chain loop runs.
    # ------------------------------------------------------------------
    if mode == "ar":
        assert initial_latents_30 is not None
        assert ar_prompt_embeds is not None and ar_noisy_fa_full is not None
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
        total_frames_ar = prefill_frames + args.ar_gen_chunks * BASE_CHUNK_FRAMES
        prefill_lat = initial_latents_30[:, :prefill_frames].to(dtype=pe_dtype)
        cache_fill_tag = "genfill" if args.ar_cache else "cleanfill"
        log.info(
            "[AR][%s] rank=%d mode=%s cache_fill=%s  prefill_frames=%d  "
            "gen_chunks=%d  cache_chunks=%d  chunks_per_step=%d  "
            "total_frames=%d  (z2 [%.3f,%.3f], z7 [%.3f,%.3f])",
            label, rank, cond_tag, cache_fill_tag, prefill_frames,
            args.ar_gen_chunks, args.cache_chunks, args.chunks_per_step,
            total_frames_ar,
            float(clean_fa_ar[..., 0].min()), float(clean_fa_ar[..., 0].max()),
            float(clean_fa_ar[..., 1].min()), float(clean_fa_ar[..., 1].max()),
        )

        noise_seed = args.seed + VIDEO_NOISE_BASE
        torch.manual_seed(noise_seed)
        torch.cuda.manual_seed(noise_seed)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / (1024 ** 2)

        t0 = time.time()
        full_latents = pipe.generate_ar(
            prompt_embeds=ar_prompt_embeds,
            noisy_fa_full=noisy_fa_ar,
            initial_latents=prefill_lat,
            num_gen_chunks=args.ar_gen_chunks,
            cache_chunks=args.cache_chunks,
            chunks_per_step=args.chunks_per_step,
            context_noise_timestep=args.context_noise_timestep,
            ar_cache=args.ar_cache,
            cache_refresh=args.ar_cache_refresh,
        )
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
        log.info(
            "[AR][%s] rank=%d streaming rollout done in %.1fs "
            "(K=%d, fill=%s, refresh=%s, bootstrap=%d, main=%d, seed=%d) | "
            "mem before=%.0f MiB  peak_alloc=%.0f MiB  peak_reserved=%.0f MiB",
            label, rank, time.time() - t0, args.denoising_steps,
            cache_fill_tag, args.ar_cache_refresh,
            args.ar_initial_chunks if args.ar_cache else 0,
            args.ar_gen_chunks, noise_seed,
            mem_before, peak_mb, reserved_mb,
        )

        full_latents = full_latents.detach()
        # full_latents is [1, prefill_frames + ar_gen_chunks*3, C, H, W].
        # At the canonical ar_initial_chunks=1 / ar_gen_chunks=7 /
        # chunks_per_step=1 setting this is [1, 3 + 21, C, H, W] = [1, 24,
        # C, H, W], matching chain's STREAM_LATENT_SPAN = CONTEXT_FRAMES
        # (3) + NUM_FRAMES (21).
        gen_latents = full_latents[:, prefill_frames:]
        context_latents = full_latents[:, :prefill_frames]

        video_np = pipe.decode_latents(gen_latents)
        context_np = pipe.decode_latents(context_latents)

        motion, teacher_z_8d = pipe.compute_teacher_visuals(gen_latents)
        n_c = teacher_z_8d.shape[1]
        teacher_z2z7 = teacher_z_8d[:, :, CRITIC_ACTION_DIMS]
        # Chunk-granularity commanded (target) actions covering only the
        # generated portion of the rollout — annotate_video displays these
        # as the "target" bars next to the teacher/critic bars.
        chunk_dev_gen = frame_actions_to_chunk_actions(noisy_fa_ar[:, prefill_frames:])
        critic_z2z7 = None
        if pipe.action_critic is not None:
            cp = pipe.run_critic(gen_latents, chunk_dev_gen)
            if cp is not None:
                critic_z2z7 = cp[:, :, CRITIC_ACTION_DIMS]
        target_z_ar = chunk_dev_gen[:, :n_c].contiguous()

        refresh_tag = "" if args.ar_cache_refresh == "append" else f"_{args.ar_cache_refresh}"
        tag = (
            f"{label}_{cond_tag}_ar_c{args.cache_chunks}_cps{args.chunks_per_step}"
            f"_{cache_fill_tag}{refresh_tag}"
        )
        raw_path = out_dir / f"{tag}_rollout_raw.mp4"
        annot_path = out_dir / f"{tag}_rollout_annotated.mp4"

        # Raw video: prefill-context ++ AR-generated, unannotated.
        # NB: pipe.decode_latents returns a 3D array [T_video, H, W, 3] —
        # there's NO batch dim to index into (matches eval_causal_chain's
        # usage). Indexing ``[0]`` here would collapse to a single H-row
        # slice and corrupt the mp4.
        full_rollout_np = np.concatenate([context_np, video_np], axis=0)
        frames_to_mp4(full_rollout_np, str(raw_path), fps=20)

        # Annotated video: overlay teacher/critic/target bars + motion on
        # the generated portion only (the prefill is raw ground truth and
        # has no "generated" metrics to display).
        ann = annotate_video(
            video_np, teacher_z2z7, critic_z2z7, target_z_ar, motion,
            f"{tag}  rank{rank}",
        )
        ann_cat = np.concatenate([context_np, ann], axis=0)
        frames_to_mp4(ann_cat, str(annot_path), fps=20)
        log.info(
            "[AR][%s] rank=%d wrote %s and %s (shape=%s)",
            label, rank, raw_path.name, annot_path.name, full_rollout_np.shape,
        )

        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        log.info("[AR][%s] rank=%d done.", label, rank)
        return

    # Chain-mode path continues below ---------------------------------------
    prompt_embeds = prompt_embeds.to(dtype=pe_dtype)

    # ---- Build clean / noisy frame actions ----
    if per_rank_mode:
        clean_fa = clean_fa.to(dtype=pe_dtype)  # [1, NUM_FRAMES, len(action_dims)]
        if args.rank_mode == "dataset":
            noisy_fa = clean_fa.clone()
            cond_tag = "dataset"
        elif args.rank_mode == "counterfactual":
            noisy_fa = -clean_fa
            cond_tag = "counterfactual"
        else:  # defensive
            raise SystemExit(f"Unknown rank_mode {args.rank_mode!r}")
        log.info(
            "[%s] rank=%d mode=%s  clean_fa shape=%s  (z2 range [%.3f,%.3f], z7 [%.3f,%.3f])",
            label, rank, cond_tag, tuple(clean_fa.shape),
            float(clean_fa[..., 0].min()), float(clean_fa[..., 0].max()),
            float(clean_fa[..., 1].min()), float(clean_fa[..., 1].max()),
        )
    else:
        fixed = torch.empty(1, NUM_FRAMES, RAW_ACTION_DIM, device=device, dtype=pe_dtype)
        fixed[..., 0] = FIXED_ACTION_Z2
        fixed[..., 1] = FIXED_ACTION_Z7
        clean_fa = fixed
        noisy_fa = fixed
        cond_tag = "fixed"

    chunk_dev_all = frame_actions_to_chunk_actions(noisy_fa)

    # ODE student was trained with both AdaLN + token conditioning on.
    use_adaln = True
    use_tokens = True

    prev_gens: List[torch.Tensor] = []
    prev_video_raws: List[np.ndarray] = []

    total_videos = 0
    for vid in range(1, args.num_causal_videos + 1):
        chunk_dev = chunk_dev_all

        clean_x = build_causal_clean_x(vid, zarr_clean_21, prev_gens)

        cond = pipe.build_conditional(
            prompt_embeds, noisy_fa, clean_fa, use_adaln, use_tokens,
        )

        noise_seed = args.seed + VIDEO_NOISE_BASE + vid * VIDEO_NOISE_SEED_STRIDE
        torch.manual_seed(noise_seed)
        torch.cuda.manual_seed(noise_seed)

        t0 = time.time()
        gen_latents = pipe.generate(cond, clean_x)
        log.info(
            "[%s] rank=%d video=%d/%d cond=%s clean_x built | gen %.1fs (K=%d) | noise_seed=%s",
            label, rank, vid, args.num_causal_videos, cond_tag,
            time.time() - t0, args.denoising_steps, noise_seed,
        )
        gen_latents = gen_latents.detach()

        context_latents_3 = seed_lat.unsqueeze(0)
        context_np = pipe.decode_latents(context_latents_3)
        video_np = pipe.decode_latents(gen_latents)

        motion, teacher_z_8d = pipe.compute_teacher_visuals(gen_latents)
        n_c = teacher_z_8d.shape[1]
        teacher_z2z7 = teacher_z_8d[:, :, CRITIC_ACTION_DIMS]
        critic_z2z7 = None
        if pipe.action_critic is not None:
            cp = pipe.run_critic(gen_latents, chunk_dev)
            if cp is not None:
                critic_z2z7 = cp[:, :, CRITIC_ACTION_DIMS]
        target_z = chunk_dev[:, :n_c].contiguous()

        title_bits = [label, f"r{rank}", f"v{vid}", cond_tag]
        if per_rank_mode and args.rank_tag:
            title_bits.append(args.rank_tag)
        title = " ".join(title_bits)
        ann = annotate_video(video_np, teacher_z2z7, critic_z2z7, target_z, motion, title)

        # -----------------------------------------------------------------
        # Rollout-so-far display (identical semantics to eval_causal_chain).
        # -----------------------------------------------------------------
        rollout_latents = build_rollout_latents(vid, gen_latents, prev_gens)
        rollout_motion, rollout_teacher_z_8d = pipe.compute_teacher_visuals(rollout_latents)
        rollout_teacher_z2z7 = rollout_teacher_z_8d[:, :, CRITIC_ACTION_DIMS]
        rollout_critic_z2z7 = None
        if pipe.action_critic is not None:
            rollout_cp = pipe.run_critic(rollout_latents, chunk_dev)
            if rollout_cp is not None:
                rollout_critic_z2z7 = rollout_cp[:, :, CRITIC_ACTION_DIMS]
        rollout_target_z = chunk_dev[:, : rollout_teacher_z_8d.shape[1]].contiguous()

        chunk_px = video_np.shape[0] // NUM_ACTION_CHUNKS
        assert video_np.shape[0] == chunk_px * NUM_ACTION_CHUNKS, (
            video_np.shape[0], chunk_px, NUM_ACTION_CHUNKS,
        )
        assert context_np.shape[0] == chunk_px, (context_np.shape[0], chunk_px)
        rollout_parts = [context_np]
        for j in range(NUM_ACTION_CHUNKS):
            lo = j * chunk_px
            hi = lo + chunk_px
            if j < vid - 1:
                src = prev_video_raws[j]
            else:
                src = video_np
            rollout_parts.append(src[lo:hi])
        rollout_video_cat = np.concatenate(rollout_parts, axis=0)

        rollout_title_bits = [label, f"r{rank}", f"v{vid}", "rollout", cond_tag]
        if per_rank_mode and args.rank_tag:
            rollout_title_bits.append(args.rank_tag)
        rollout_title = " ".join(rollout_title_bits)
        rollout_ann = annotate_video(
            rollout_video_cat[chunk_px:],
            rollout_teacher_z2z7,
            rollout_critic_z2z7,
            rollout_target_z,
            rollout_motion,
            rollout_title,
        )
        rollout_ann_cat = np.concatenate([context_np, rollout_ann], axis=0)

        if per_rank_mode:
            zarr_ts = Path(ride_meta["zarr_path"]).stem
            pieces = [f"video_{vid:02d}", zarr_ts, cond_tag]
            if args.rank_tag:
                pieces.append(args.rank_tag)
            fname_stem = "_".join(pieces)
        else:
            fname_stem = f"video_{vid:02d}"

        frames_to_mp4(rollout_ann_cat, str(out_dir / f"{fname_stem}_annotated.mp4"), fps=20)
        frames_to_mp4(rollout_video_cat, str(out_dir / f"{fname_stem}_raw.mp4"), fps=20)
        frames_to_mp4(
            np.concatenate([context_np, ann], axis=0),
            str(out_dir / f"{fname_stem}_pass_annotated.mp4"),
            fps=20,
        )
        frames_to_mp4(
            np.concatenate([context_np, video_np], axis=0),
            str(out_dir / f"{fname_stem}_pass_raw.mp4"),
            fps=20,
        )

        prev_gens.append(gen_latents)
        prev_video_raws.append(video_np)
        total_videos += 1
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------------
    # Final causal rollout: seed ++ [g1[0], g2[1], ..., g7[6]].
    # Raw pixels stitched from per-pass decodes to avoid VAE boundary
    # contamination across chunk sources.
    # ---------------------------------------------------------------------
    if len(prev_gens) >= NUM_ACTION_CHUNKS:
        committed_chunks: List[torch.Tensor] = []
        for i in range(NUM_ACTION_CHUNKS):
            g = prev_gens[i]
            lo = i * NUM_FRAME_PER_BLOCK
            hi = lo + NUM_FRAME_PER_BLOCK
            committed_chunks.append(g[:, lo:hi])
        rollout_gen = torch.cat(committed_chunks, dim=1)
        assert rollout_gen.shape[1] == NUM_FRAMES, rollout_gen.shape

        context_np = pipe.decode_latents(seed_lat.unsqueeze(0))
        if not prev_video_raws:
            raise RuntimeError("Expected cached decoded videos for final rollout.")
        chunk_px = prev_video_raws[0].shape[0] // NUM_ACTION_CHUNKS
        rollout_parts = [context_np]
        for i in range(NUM_ACTION_CHUNKS):
            lo = i * chunk_px
            hi = lo + chunk_px
            rollout_parts.append(prev_video_raws[i][lo:hi])
        rollout_cat = np.concatenate(rollout_parts, axis=0)

        motion_r, teacher_z_8d_r = pipe.compute_teacher_visuals(rollout_gen)
        n_c_r = teacher_z_8d_r.shape[1]
        teacher_z2z7_r = teacher_z_8d_r[:, :, CRITIC_ACTION_DIMS]
        critic_z2z7_r = None
        if pipe.action_critic is not None:
            cp_r = pipe.run_critic(rollout_gen, chunk_dev_all)
            if cp_r is not None:
                critic_z2z7_r = cp_r[:, :, CRITIC_ACTION_DIMS]
        target_z_r = chunk_dev_all[:, :n_c_r].contiguous()

        title_bits_r = [label, f"r{rank}", "rollout", cond_tag]
        if per_rank_mode and args.rank_tag:
            title_bits_r.append(args.rank_tag)
        title_r = " ".join(title_bits_r)
        ann_r = annotate_video(
            rollout_cat[chunk_px:], teacher_z2z7_r, critic_z2z7_r,
            target_z_r, motion_r, title_r,
        )
        ann_cat_r = np.concatenate([context_np, ann_r], axis=0)

        if per_rank_mode:
            zarr_ts = Path(ride_meta["zarr_path"]).stem
            pieces = ["final_causal_rollout", zarr_ts, cond_tag]
            if args.rank_tag:
                pieces.append(args.rank_tag)
            rollout_stem = "_".join(pieces)
        else:
            rollout_stem = "final_causal_rollout"
        frames_to_mp4(ann_cat_r, str(out_dir / f"{rollout_stem}_annotated.mp4"), fps=20)
        frames_to_mp4(rollout_cat, str(out_dir / f"{rollout_stem}_raw.mp4"), fps=20)
        log.info(
            "[%s] rank=%d final rollout written: %s (8 chunks = 24 frames, %s)",
            label, rank, rollout_stem, cond_tag,
        )
    else:
        log.warning(
            "[%s] rank=%d only %d gens (need %d) — skipping final causal rollout.",
            label, rank, len(prev_gens), NUM_ACTION_CHUNKS,
        )

    log.info("[%s] rank=%d done: %d chained videos → %s", label, rank, total_videos, out_dir)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
