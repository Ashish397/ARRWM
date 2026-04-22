
"""Streaming-inference benchmark for the action-forcing ODE-distilled student.

Measures wall-clock time and peak memory for KV-cache streaming inference
on a single GPU. Designed to answer the two questions in
``.claude/training_inference_mismatch.md``:

  1. How does per-chunk cost scale with ``chunks_per_step`` (the number
     of 3-frame chunks emitted per forward) at a fixed cache budget?
  2. How does per-chunk cost scale with the KV-cache size (and what is
     the max cache that fits on one GPU)?

Base unit for the ``cache_chunks`` CLI flag is a 3-frame chunk (matches
the training ``num_frame_per_block=3``). Internally we program the
``local_attn_size`` of the DiT (measured in frames) as

    local_attn_size = (cache_chunks + chunks_per_step) * 3

so the attention window at each forward = ``cache_chunks`` past chunks
+ ``chunks_per_step`` current chunks, as per scenario A in the doc.

Actions are zeroed throughout: the goal is pure compute timing; a real
inference path would splice in true commanded actions per step. We still
build action_projection / action_token_projection modulation and tokens
so the model sees the same sequence layout (action_tokens_per_frame=1)
it saw during training.

Outputs:
  * per-chunk wall-clock times (mean / p50 / p95)
  * per-forward mean time (one forward = one denoising step)
  * per-output-frame mean time
  * peak CUDA memory during the run

The script also supports ``--find_max_cache`` which binary-searches
the largest ``cache_chunks`` that fits + runs one streaming step.

Usage (single GPU, inside an interactive shell)::

    python action-forcing/bin/bench_streaming_inference.py \\
        --ckpt logs/v14_balanced_weunz/causal_lora_step0006600.pt \\
        --student-ckpt logs/action_ode_distill_10h/latest_0000300.pt \\
        --config configs/action_ode_distill.yaml \\
        --cache_chunks 3 --chunks_per_step 1 --denoising_steps 4 \\
        --total_chunks 8

The srun launcher in ``sbatch/bench_ode_distill_streaming.sbatch`` runs
the full parameter sweep requested for the 10h step-300 checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

_THIS_DIR = Path(__file__).resolve().parents[1]
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
_WORKSPACE = _THIS_DIR.parent
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))


log = logging.getLogger("bench_streaming")


def _reclaim_cuda(device: torch.device) -> None:
    """Drop Python refs to GPU tensors, run GC, return allocator segments.

    Required between ``find_max_cache`` trials (and after each streaming
    bench when grads are enabled): autograd graphs + KV buffers otherwise
    linger until the next allocation blows the same GPU.
    """
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()


# Wan2.1-T2V-1.3B geometry (must match training).
FRAME_SPATIAL_TOKENS = 1560      # 60 * 26 tokens per frame latent
BASE_CHUNK_FRAMES = 3             # num_frame_per_block trained into the DiT
RAW_ACTION_DIM = 2                # (z2, z7) — raw commanded action scalars


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Streaming-inference benchmark for the ODE-distilled student.")
    p.add_argument("--ckpt", type=str, required=True,
                   help="Teacher .pt used to init the merged DiT + heads "
                        "(must contain a 'lora' key). Required because "
                        "ODERegression loads the teacher and folds LoRA "
                        "before we overlay the student weights on top.")
    p.add_argument("--student-ckpt", type=str, required=True,
                   help="Student checkpoint written by af_trainer.ode "
                        "(e.g. logs/action_ode_distill_10h/latest_0000300.pt).")
    p.add_argument("--config", type=str, default="configs/action_ode_distill.yaml",
                   help="Training config whose model geometry matches the checkpoint.")
    p.add_argument("--output", type=str, default=None,
                   help="Optional JSON path to append the benchmark result(s) to.")

    p.add_argument("--cache_chunks", type=int, default=3,
                   help="Number of 3-frame chunks retained in the past KV cache.")
    p.add_argument("--chunks_per_step", type=int, default=1,
                   help="Number of 3-frame chunks generated per forward pass "
                        "(=> num_frame_per_block = chunks_per_step * 3).")
    p.add_argument("--denoising_steps", type=int, default=4,
                   help="Number of denoising iterations per chunk (student default: 4).")
    p.add_argument("--total_chunks", type=int, default=8,
                   help="Total base-chunk-equivalents to generate. Each streaming "
                        "step emits chunks_per_step chunks; we stop once we have "
                        "emitted >= total_chunks.")
    p.add_argument("--warmup_chunks", type=int, default=1,
                   help="Warmup base-chunks before timing (one GPU warmup step "
                        "is usually enough to stabilise autotuner / cudnn).")

    p.add_argument("--find_max_cache", action="store_true",
                   help="Binary-search the max cache_chunks that fits on the "
                        "current GPU (overrides --cache_chunks). By default the "
                        "reported timings come from the winning probe run (no "
                        "second full bench — avoids OOM after failed probes "
                        "fragment the allocator). See --find_max_cache_rerun_timed.")
    p.add_argument("--find_max_cache_rerun_timed", action="store_true",
                   help="After --find_max_cache, run a separate _run_streaming_bench "
                        "with --total_chunks/--warmup_chunks (legacy; can OOM if the "
                        "GPU is near capacity after probe failures).")
    p.add_argument("--max_cache_ceiling", type=int, default=200,
                   help="Upper bound (in chunks) for --find_max_cache.")

    p.add_argument("--context_noise_timestep", type=float, default=0.0,
                   help="Timestep passed on the post-denoise cache-refresh "
                        "forward (training uses context_noise=0).")
    p.add_argument("--skip_cache_refresh", action="store_true",
                   help="Skip the post-denoise cache-refresh forward. This "
                        "is NOT the production path (it breaks clean-context "
                        "KV-cache semantics) but is useful to compare raw "
                        "denoise-only wall-clock cost.")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", type=str, default="",
                   help="Free-form label written into the result JSON.")
    p.add_argument("--no_grad", action="store_true",
                   help="Wrap the bench (and find_max_cache probes) in "
                        "torch.inference_mode(). Reports realistic "
                        "inference-only peak memory — no autograd graph, no "
                        "activation saved tensors. Training still needs "
                        "grads, so the default stays grad-enabled.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading (reuses ODERegression for the assembly path)
# ---------------------------------------------------------------------------


def _build_model(args: argparse.Namespace, device: torch.device):
    """Build an ODERegression model, overlay the student, strip motion pipeline.

    Returns the model plus a bundle of references we'll need during inference:
      base_dit: the CausalWanModel (underneath PeftModel / DDP, if any).
      wrapper:  the WanDiffusionWrapper (exposes .forward with kv_cache).
      action_projection, action_token_projection: for per-chunk conditioning.
    """
    from omegaconf import OmegaConf

    log.info("Loading config %s", args.config)
    cfg = OmegaConf.load(args.config)
    OmegaConf.set_struct(cfg, False)
    cfg.generator_ckpt = args.ckpt
    # NOTE: we do NOT touch cfg.use_motion_pipeline here. Setting it to
    # False makes ODERegression.__init__ raise (it's a hard dependency
    # for training). Instead we flip the runtime flag on the model
    # *after* init so ``ensure_motion_pipeline()`` becomes a no-op —
    # the motion pipeline is only actually built on demand, never in
    # the ctor.

    log.info("Building ODERegression (ctx from teacher=%s) ...", args.ckpt)
    from af_model.ode_regression import ODERegression
    model = ODERegression(cfg, device=device)
    model.eval()
    # Never build the motion pipeline: it pulls CoTracker + ss_vae from
    # disk/HF and isn't needed for a pure inference-timing run.
    model.use_motion_pipeline = False

    # Overlay the student checkpoint on top of the teacher-merged DiT.
    sp = Path(args.student_ckpt)
    if not sp.exists():
        raise FileNotFoundError(f"Student checkpoint not found: {sp}")
    log.info("Overlaying student checkpoint %s", sp.name)
    sk = torch.load(sp, map_location="cpu", weights_only=False)
    if "generator" not in sk:
        raise RuntimeError(
            f"Student checkpoint {sp} missing 'generator' key; expected a "
            "full-rank trainer-format snapshot."
        )
    model.generator.model.load_state_dict(sk["generator"], strict=True)
    if model.action_projection is not None and "action_projection" in sk:
        model.action_projection.load_state_dict(sk["action_projection"])
    if model.action_token_projection is not None and "action_token_projection" in sk:
        model.action_token_projection.load_state_dict(sk["action_token_projection"])
    model.to(device).eval()
    log.info("Student overlay OK (embedded step=%d).", int(sk.get("step", -1)))

    wrapper = model.generator
    base_dit = wrapper.model
    # Unwrap PeftModel (if any LoRA wrapper leaked through — merge_and_unload
    # should have stripped it, but we'd rather touch the underlying DiT).
    if hasattr(base_dit, "get_base_model"):
        base_dit = base_dit.get_base_model()

    # Action-patch sanity: the wrapper should already have action patches
    # applied by ODERegression.__init__. Re-apply defensively in case a
    # future reorganisation drops that step.
    from model.action_model_patch import apply_action_patches
    apply_action_patches(wrapper)

    return model, wrapper, base_dit


# ---------------------------------------------------------------------------
# KV cache helpers (mirrors CausalInferencePipeline._initialize_kv_cache)
# ---------------------------------------------------------------------------


def _initialize_kv_cache(
    *,
    num_transformer_blocks: int,
    batch_size: int,
    kv_cache_size_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
) -> List[Dict[str, torch.Tensor]]:
    # Wan2.1-T2V-1.3B: 12 heads, 128 head-dim (d_model=1536).
    kv_cache: List[Dict[str, torch.Tensor]] = []
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
) -> List[Dict[str, Any]]:
    crossattn_cache: List[Dict[str, Any]] = []
    for _ in range(num_transformer_blocks):
        crossattn_cache.append({
            "k":       torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
            "v":       torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
            "is_init": False,
        })
    return crossattn_cache


def _set_attention_window(base_dit, *, local_attn_size_frames: int, max_tokens: int) -> None:
    """Propagate ``local_attn_size`` (frames) and ``max_attention_size``
    (tokens) to every attention block + top-level module.

    ``CausalWanSelfAttention`` uses ``self.local_attn_size`` for the
    sliding-window truncation / rolling logic in the KV-cache path, and
    ``self.max_attention_size`` as the upper bound on the temp buffers
    it concatenates during the "sink + window" attention pass. Both
    must match the cache we just allocated.
    """
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
# Streaming benchmark core
# ---------------------------------------------------------------------------


@dataclass
class ChunkTiming:
    """Per-streaming-step measurement."""
    step_idx: int
    chunk_start: int              # base-chunk index of the first chunk emitted
    chunks_this_step: int         # chunks_per_step
    denoise_ms: float             # sum of all denoising_steps forwards
    refresh_ms: float             # post-denoise cache-refresh forward (or 0)
    total_ms: float               # = denoise_ms + refresh_ms
    peak_mem_gib: float


@dataclass
class BenchResult:
    cache_chunks: int
    chunks_per_step: int
    denoising_steps: int
    total_chunks_emitted: int
    frames_emitted: int
    per_chunk_ms_mean: float
    per_chunk_ms_p50: float
    per_chunk_ms_p95: float
    per_frame_ms_mean: float
    per_forward_ms_mean: float    # averaged over denoise forwards only
    peak_mem_gib: float
    local_attn_size_frames: int
    kv_cache_tokens: int
    note: str = ""
    label: str = ""
    per_step_timings: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _build_zero_conditional(
    *, model, batch_size: int, num_frames_current: int,
    prompt_embeds: torch.Tensor, dtype: torch.dtype, device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Zero-action conditional dict for the current chunk.

    Matches the shapes expected by the forward wrapper / action patches:
      prompt_embeds:          [B, 512, 4096]
      _action_modulation:     [B, F_current, 6, dim]
      _action_tokens:         [B, F_current, dim]
    """
    zero_actions = torch.zeros(
        [batch_size, num_frames_current, RAW_ACTION_DIM],
        dtype=dtype, device=device,
    )
    cond: Dict[str, torch.Tensor] = {
        "prompt_embeds": prompt_embeds.to(dtype=dtype, device=device),
    }
    if model.action_projection is not None:
        cond["_action_modulation"] = model.action_projection(
            zero_actions, num_frames=num_frames_current,
        )
    if model.action_token_projection is not None:
        cond["_action_tokens"] = model.action_token_projection(zero_actions)
    return cond


def _run_streaming_bench(
    *,
    model,
    wrapper,
    base_dit,
    device: torch.device,
    dtype: torch.dtype,
    cache_chunks: int,
    chunks_per_step: int,
    denoising_steps: int,
    total_chunks: int,
    warmup_chunks: int,
    context_noise_timestep: float,
    skip_cache_refresh: bool,
) -> BenchResult:
    """Run a streaming rollout and time each streaming step.

    The parameterisation follows ``.claude/training_inference_mismatch.md``:
      * cache_chunks = past 3-frame chunks retained
      * chunks_per_step = chunks emitted per forward
      * local_attn_size (frames) = (cache_chunks + chunks_per_step) * 3
    """
    batch_size = 1
    num_frame_per_block = chunks_per_step * BASE_CHUNK_FRAMES
    # Ensure the model uses the requested block size for the causal mask.
    # The cached block_mask must be invalidated whenever the block size /
    # seq layout changes — without this the kv-cache path would try to
    # reuse a mask that was built for a different num_frame_per_block.
    base_dit.num_frame_per_block = num_frame_per_block
    base_dit.block_mask = None

    # action_tokens_per_frame must match what the DiT was trained with
    # (ODE config: action_conditioning_mode='both' -> 1 token per frame).
    # ODERegression.__init__ already set this; read it back here for
    # downstream seq_len arithmetic.
    action_tokens_per_frame = int(getattr(base_dit, "action_tokens_per_frame", 1))
    frame_seq_length = FRAME_SPATIAL_TOKENS + action_tokens_per_frame

    # local_attn_size is measured in FRAMES (not chunks or tokens).
    local_attn_size_frames = (cache_chunks + chunks_per_step) * BASE_CHUNK_FRAMES
    kv_cache_tokens = local_attn_size_frames * frame_seq_length

    # ``seq_len`` in the wrapper is only used as the zero-padding target
    # for the *current* chunk's token sequence (see causal_model.py,
    # around the `torch.cat([u, u.new_zeros(1, seq_len - u.size(1), ...)`
    # call). It does NOT need to span the KV cache. It just needs to be
    # ≥ num_frame_per_block * frame_seq_length. Keep whatever ODE init
    # set (typically 32781 for 21 frames) and only bump if the block is
    # larger than that (never happens for chunks_per_step ≤ 7).
    required_chunk_tokens = num_frame_per_block * frame_seq_length
    wrapper.seq_len = max(int(wrapper.seq_len), required_chunk_tokens)
    _set_attention_window(
        base_dit,
        local_attn_size_frames=local_attn_size_frames,
        max_tokens=kv_cache_tokens,
    )

    log.info(
        "Streaming bench: cache_chunks=%d chunks_per_step=%d num_frame_per_block=%d "
        "local_attn_size=%d frames kv_cache_tokens=%d (wrapper.seq_len=%d)",
        cache_chunks, chunks_per_step, num_frame_per_block,
        local_attn_size_frames, kv_cache_tokens, int(wrapper.seq_len),
    )

    num_transformer_blocks = len(base_dit.blocks)

    # Large tensors (KV + cross-attn caches) must be explicitly dropped and
    # the allocator reclaimed between successive runs — otherwise
    # ``find_max_cache`` stacks trials on the same GPU until OOM.
    prompt_embeds: Optional[torch.Tensor] = None
    kv_cache: Optional[List[Dict[str, torch.Tensor]]] = None
    crossattn_cache: Optional[List[Dict[str, Any]]] = None

    try:
        # Fake prompt embedding (T5 umt5-xxl is 4096-d).
        prompt_embeds = torch.zeros([batch_size, 512, 4096], dtype=torch.float32, device=device)

        # Denoising step list — 4 entries by default. Just pick equally-spaced
        # descending timesteps so add_noise has something sensible between
        # iterations. The student's real schedule lives in
        # model.denoising_step_list but its length is config-driven; here we
        # honour the CLI's --denoising_steps for a clean comparison across
        # rollouts.
        if denoising_steps == int(model.denoising_step_list.shape[0]):
            ts_list = model.denoising_step_list.detach().to(device=device, dtype=torch.float32)
            ts_list, _ = torch.sort(ts_list, descending=True)
        else:
            # Uniformly-spaced descending timesteps in [~50, 1000].
            edges = torch.linspace(1000.0, 50.0, steps=denoising_steps, device=device)
            ts_list = edges
        log.info("Denoising timesteps used: %s", [round(float(x), 2) for x in ts_list.tolist()])

        scheduler = model.scheduler
        scheduler.sigmas = scheduler.sigmas.to(device)

        # Channel / spatial shape — match WanVAE latent geometry for the training clip.
        # 480x832 pixels with 8x8 VAE downsample → 60x104 latent; 16 channels.
        C, H, W = 16, 60, 104

        # Allocate fresh KV cache sized for the window.
        kv_cache = _initialize_kv_cache(
            num_transformer_blocks=num_transformer_blocks,
            batch_size=batch_size,
            kv_cache_size_tokens=kv_cache_tokens,
            dtype=dtype,
            device=device,
        )
        crossattn_cache = _initialize_crossattn_cache(
            num_transformer_blocks=num_transformer_blocks,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )

        current_start_frame = 0
        per_step_timings: List[ChunkTiming] = []

        num_steps_needed = math.ceil(total_chunks / chunks_per_step)
        num_warmup_steps = max(1, math.ceil(warmup_chunks / chunks_per_step))
        total_iters = num_warmup_steps + num_steps_needed

        torch.cuda.reset_peak_memory_stats(device)
        run_peak_gib = 0.0

        for it in range(total_iters):
            is_warmup = it < num_warmup_steps
            step_idx = it - num_warmup_steps

            # Sample fresh input noise for this chunk.
            noise = torch.randn(
                [batch_size, num_frame_per_block, C, H, W],
                dtype=torch.float32, device=device,
            )
            noisy_input = noise.to(dtype)

            # Build the action conditional for this chunk (zero actions).
            cond = _build_zero_conditional(
                model=model, batch_size=batch_size,
                num_frames_current=num_frame_per_block,
                prompt_embeds=prompt_embeds, dtype=dtype, device=device,
            )

            denoise_evt_start = torch.cuda.Event(enable_timing=True)
            denoise_evt_end   = torch.cuda.Event(enable_timing=True)
            refresh_evt_start = torch.cuda.Event(enable_timing=True)
            refresh_evt_end   = torch.cuda.Event(enable_timing=True)

            denoise_evt_start.record()
            denoised_pred: Optional[torch.Tensor] = None
            for d_idx in range(denoising_steps):
                t_val = float(ts_list[d_idx].item())
                timestep = torch.full(
                    [batch_size, num_frame_per_block], t_val,
                    device=device, dtype=torch.float32,
                )
                with torch.amp.autocast("cuda", dtype=dtype):
                    out = wrapper(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond,
                        timestep=timestep,
                        kv_cache=kv_cache,
                        crossattn_cache=crossattn_cache,
                        current_start=current_start_frame * frame_seq_length,
                    )
                # out[1] == pred_x0
                denoised_pred = out[1]
                if d_idx < denoising_steps - 1:
                    next_t = float(ts_list[d_idx + 1].item())
                    flat = denoised_pred.flatten(0, 1).float()
                    flat_noise = torch.randn_like(flat)
                    flat_t = torch.full(
                        (flat.shape[0],), next_t, device=device, dtype=torch.float32,
                    )
                    noisy_input = (
                        scheduler.add_noise(flat, flat_noise, flat_t)
                        .view(batch_size, num_frame_per_block, C, H, W)
                        .to(dtype)
                    )
            denoise_evt_end.record()
            assert denoised_pred is not None

            # Cache refresh: re-run the DiT on the clean pred_x0 with
            # context_noise timestep, so the KV cache holds clean-context
            # keys/values for the next chunk. Matches CausalInferencePipeline.
            refresh_ms = 0.0
            if not skip_cache_refresh:
                refresh_t = torch.full(
                    [batch_size, num_frame_per_block], float(context_noise_timestep),
                    device=device, dtype=torch.float32,
                )
                refresh_evt_start.record()
                with torch.amp.autocast("cuda", dtype=dtype):
                    wrapper(
                        noisy_image_or_video=denoised_pred,
                        conditional_dict=cond,
                        timestep=refresh_t,
                        kv_cache=kv_cache,
                        crossattn_cache=crossattn_cache,
                        current_start=current_start_frame * frame_seq_length,
                    )
                refresh_evt_end.record()

            torch.cuda.synchronize(device)
            denoise_ms = denoise_evt_start.elapsed_time(denoise_evt_end)
            if not skip_cache_refresh:
                refresh_ms = refresh_evt_start.elapsed_time(refresh_evt_end)
            total_ms = denoise_ms + refresh_ms
            peak_mem_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            run_peak_gib = max(run_peak_gib, peak_mem_gib)

            if not is_warmup:
                per_step_timings.append(ChunkTiming(
                    step_idx=step_idx,
                    chunk_start=current_start_frame // BASE_CHUNK_FRAMES,
                    chunks_this_step=chunks_per_step,
                    denoise_ms=denoise_ms,
                    refresh_ms=refresh_ms,
                    total_ms=total_ms,
                    peak_mem_gib=peak_mem_gib,
                ))
                log.info(
                    "  step=%d chunk=%d  denoise=%.1fms refresh=%.1fms total=%.1fms  peak=%.2fGiB",
                    step_idx, current_start_frame // BASE_CHUNK_FRAMES,
                    denoise_ms, refresh_ms, total_ms, peak_mem_gib,
                )
            else:
                log.info(
                    "  [warmup %d/%d] denoise=%.1fms refresh=%.1fms total=%.1fms  peak=%.2fGiB",
                    it + 1, num_warmup_steps,
                    denoise_ms, refresh_ms, total_ms, peak_mem_gib,
                )

            current_start_frame += num_frame_per_block

        # ------------------------------------------------------------------
        # Summarise
        # ------------------------------------------------------------------
        per_step_total = [s.total_ms for s in per_step_timings]
        # "Per chunk" = total_ms of a streaming step divided by chunks_per_step,
        # aggregated across streaming steps.
        per_chunk_list = [s.total_ms / max(s.chunks_this_step, 1) for s in per_step_timings]
        # "Per forward" = denoise_ms / denoising_steps (cache-refresh excluded).
        per_forward_list = [s.denoise_ms / max(denoising_steps, 1) for s in per_step_timings]

        def _p(xs: List[float], q: float) -> float:
            if not xs:
                return 0.0
            xs_sorted = sorted(xs)
            k = int(round(q * (len(xs_sorted) - 1)))
            return xs_sorted[k]

        chunks_emitted = len(per_step_timings) * chunks_per_step
        frames_emitted = chunks_emitted * BASE_CHUNK_FRAMES

        per_chunk_mean = statistics.fmean(per_chunk_list) if per_chunk_list else 0.0
        per_chunk_p50 = _p(per_chunk_list, 0.5)
        per_chunk_p95 = _p(per_chunk_list, 0.95)
        per_forward_mean = statistics.fmean(per_forward_list) if per_forward_list else 0.0
        per_frame_mean = per_chunk_mean / BASE_CHUNK_FRAMES if per_chunk_mean else 0.0

        result = BenchResult(
            cache_chunks=cache_chunks,
            chunks_per_step=chunks_per_step,
            denoising_steps=denoising_steps,
            total_chunks_emitted=chunks_emitted,
            frames_emitted=frames_emitted,
            per_chunk_ms_mean=per_chunk_mean,
            per_chunk_ms_p50=per_chunk_p50,
            per_chunk_ms_p95=per_chunk_p95,
            per_frame_ms_mean=per_frame_mean,
            per_forward_ms_mean=per_forward_mean,
            peak_mem_gib=run_peak_gib,
            local_attn_size_frames=local_attn_size_frames,
            kv_cache_tokens=kv_cache_tokens,
            per_step_timings=[
                {
                    "step": s.step_idx, "denoise_ms": s.denoise_ms,
                    "refresh_ms": s.refresh_ms, "total_ms": s.total_ms,
                    "peak_mem_gib": s.peak_mem_gib,
                }
                for s in per_step_timings
            ],
        )
        return result
    finally:
        if prompt_embeds is not None:
            del prompt_embeds
        if kv_cache is not None:
            del kv_cache
        if crossattn_cache is not None:
            del crossattn_cache
        _reclaim_cuda(device)


def _find_max_cache(
    *, model, wrapper, base_dit, device, dtype,
    chunks_per_step: int, denoising_steps: int,
    ceiling: int, warmup_chunks: int,
) -> Tuple[int, str, BenchResult]:
    """Binary-search the largest ``cache_chunks`` that runs 2 streaming steps
    without OOM.

    We require 2 streaming steps (not just 1) so the sliding/eviction
    code path actually runs — a 1-step trial only exercises the direct-
    insert branch and can pass when a full sweep would OOM on eviction.

    Returns the :class:`BenchResult` from the successful probe at the
    winning ``cache_chunks`` so callers do not need a second timed run
    (which often OOMs after failed probes fragment the allocator).
    """
    lo, hi = 1, max(1, int(ceiling))
    best: Optional[int] = None
    best_note = ""
    # Map cache_chunks -> last successful bench at that size (overwrite on repeat).
    results_at: Dict[int, BenchResult] = {}

    def _try(cache_chunks: int) -> Tuple[bool, str]:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            br = _run_streaming_bench(
                model=model, wrapper=wrapper, base_dit=base_dit,
                device=device, dtype=dtype,
                cache_chunks=cache_chunks,
                chunks_per_step=chunks_per_step,
                denoising_steps=denoising_steps,
                total_chunks=2 * chunks_per_step,
                warmup_chunks=warmup_chunks,
                context_noise_timestep=0.0,
                skip_cache_refresh=False,
            )
            results_at[cache_chunks] = br
            return True, ""
        except torch.cuda.OutOfMemoryError as oom:
            return False, f"OOM: {oom}"
        except RuntimeError as exc:
            msg = str(exc)
            if "out of memory" in msg.lower() or "CUDA" in msg and "memory" in msg.lower():
                return False, f"OOM-ish: {msg[:200]}"
            raise
        finally:
            # Critical: without this the next probe sees 80+ GiB still resident.
            _reclaim_cuda(device)

    log.info("=== find_max_cache: binary search over [%d, %d] chunks ===", lo, hi)
    # Quick exponential ramp first, then binary refine. This saves time
    # when the cap is very far from the ceiling.
    probe = lo
    last_ok = 0
    while probe <= hi:
        log.info("[find_max_cache] probing cache_chunks=%d (exponential ramp)", probe)
        ok, note = _try(probe)
        log.info("[find_max_cache]   -> %s%s", "OK" if ok else "FAIL", (" | " + note) if note else "")
        if ok:
            last_ok = probe
            if probe == hi:
                break
            probe = min(probe * 2, hi)
        else:
            hi = probe - 1
            break

    lo = max(last_ok, 1)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        log.info("[find_max_cache] binary: lo=%d hi=%d  trying mid=%d", lo, hi, mid)
        ok, note = _try(mid)
        log.info("[find_max_cache]   -> %s%s", "OK" if ok else "FAIL", (" | " + note) if note else "")
        if ok:
            lo = mid
        else:
            hi = mid - 1

    best = lo if lo > 0 else None
    if best is None:
        raise RuntimeError("find_max_cache: even cache_chunks=1 failed.")
    best_note = f"binary-search max cache_chunks that runs 2 streaming steps, ceiling={ceiling}"
    if best not in results_at:
        raise RuntimeError(
            f"find_max_cache: internal error — no bench result stored for winning cache_chunks={best}."
        )
    return best, best_note, results_at[best]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------
    model, wrapper, base_dit = _build_model(args, device)
    dtype = model.dtype

    # Either a real torch.inference_mode context or a no-op nullcontext.
    # Using inference_mode (not just no_grad) prevents any version-bump /
    # autograd-metadata overhead on cached KV tensors, which is the
    # production eval path.
    from contextlib import nullcontext
    grad_ctx = torch.inference_mode() if args.no_grad else nullcontext()
    if args.no_grad:
        log.info("Running under torch.inference_mode() (no autograd graph).")
    else:
        log.info("Running WITH autograd enabled (grad tensors allocated).")

    with grad_ctx:
        # --------------------------------------------------------------
        # Optional: binary-search max cache
        # --------------------------------------------------------------
        find_max_note = ""
        probe_result: Optional[BenchResult] = None
        if args.find_max_cache:
            chosen, find_max_note, probe_result = _find_max_cache(
                model=model, wrapper=wrapper, base_dit=base_dit,
                device=device, dtype=dtype,
                chunks_per_step=args.chunks_per_step,
                denoising_steps=args.denoising_steps,
                ceiling=args.max_cache_ceiling,
                warmup_chunks=args.warmup_chunks,
            )
            log.info("[find_max_cache] winner: cache_chunks=%d  (%s)", chosen, find_max_note)
            args.cache_chunks = chosen
            # Probe trials already call ``_reclaim_cuda``; one extra pass here
            # before any optional rerun.
            _reclaim_cuda(device)

        # --------------------------------------------------------------
        # Timed run (skipped after find_max unless --find_max_cache_rerun_timed:
        # a second full bench OOMs easily when prior failed probes leave the
        # allocator near full despite ``_reclaim_cuda``.)
        # --------------------------------------------------------------
        if args.find_max_cache and not args.find_max_cache_rerun_timed:
            if probe_result is None:
                raise RuntimeError("find_max_cache expected a probe BenchResult.")
            result = probe_result
        else:
            torch.cuda.empty_cache()
            result = _run_streaming_bench(
                model=model, wrapper=wrapper, base_dit=base_dit,
                device=device, dtype=dtype,
                cache_chunks=args.cache_chunks,
                chunks_per_step=args.chunks_per_step,
                denoising_steps=args.denoising_steps,
                total_chunks=args.total_chunks,
                warmup_chunks=args.warmup_chunks,
                context_noise_timestep=args.context_noise_timestep,
                skip_cache_refresh=args.skip_cache_refresh,
            )
    result.label = args.label
    if find_max_note:
        result.note = find_max_note

    # --------------------------------------------------------------
    # Report
    # --------------------------------------------------------------
    print("\n=== Streaming-inference benchmark ===")
    print(f"  label                    : {result.label or '(none)'}")
    print(f"  cache_chunks             : {result.cache_chunks}")
    print(f"  chunks_per_step          : {result.chunks_per_step}")
    print(f"  denoising_steps          : {result.denoising_steps}")
    print(f"  local_attn_size (frames) : {result.local_attn_size_frames}")
    print(f"  kv_cache_tokens          : {result.kv_cache_tokens:,}")
    print(f"  total_chunks_emitted     : {result.total_chunks_emitted}")
    print(f"  frames_emitted           : {result.frames_emitted}")
    print(f"  per_chunk_ms  (mean)     : {result.per_chunk_ms_mean:.2f}")
    print(f"  per_chunk_ms  (p50)      : {result.per_chunk_ms_p50:.2f}")
    print(f"  per_chunk_ms  (p95)      : {result.per_chunk_ms_p95:.2f}")
    print(f"  per_frame_ms  (mean)     : {result.per_frame_ms_mean:.2f}")
    print(f"  per_forward_ms (denoise) : {result.per_forward_ms_mean:.2f}")
    print(f"  peak_mem_gib             : {result.peak_mem_gib:.2f}")
    if result.note:
        print(f"  note                     : {result.note}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        existing: List[Dict[str, Any]] = []
        if out_path.exists():
            try:
                loaded = json.loads(out_path.read_text())
                if isinstance(loaded, list):
                    existing = loaded
            except Exception:
                log.warning("Could not parse existing %s; overwriting.", out_path)
        existing.append(result.to_dict())
        out_path.write_text(json.dumps(existing, indent=2))
        log.info("Appended result to %s (now %d entries).", out_path, len(existing))

    print("\nDone.")


if __name__ == "__main__":
    main()
