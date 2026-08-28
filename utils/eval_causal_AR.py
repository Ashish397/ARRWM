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


def _apply_carn_seam_affine(
    chunk: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    """Apply the training-time CARN commit transform to one AR chunk."""
    strength = float(strength)
    if strength == 0.0:
        return chunk
    if not 0.0 <= strength <= 1.0:
        raise ValueError(
            f"carn_seam_affine_lambda must be in [0, 1], got {strength}"
        )

    source_dtype = chunk.dtype
    x = chunk.float()
    mean = x.mean(dim=(0, 1, 3, 4), keepdim=True)
    std = x.std(dim=(0, 1, 3, 4), keepdim=True).clamp_min(1e-4)
    target_mean = target_mean.to(x).view(1, 1, -1, 1, 1)
    target_std = target_std.to(x).view(1, 1, -1, 1, 1)
    blended_mean = strength * target_mean + (1.0 - strength) * mean
    blended_std = strength * target_std + (1.0 - strength) * std
    return ((x - mean) / std * blended_std + blended_mean).to(source_dtype)


def _apply_reverse_noiser_dedrift(
    chunk: torch.Tensor,
    noiser: torch.nn.Module,
    level: int,
    steps: int = 1,
    alpha0: float = 1.0,
    alpha_decay: float = 0.5,
    min_level: int = 1,
    _telemetry: Optional[dict] = None,
) -> torch.Tensor:
    """Eval-time port of ``ActionForcingDMD._dedrift_with_reverse_noiser``.

    De-drift one committed AR chunk toward the GT manifold with a FROZEN
    reverse-trained ``ForwardNoiser`` (a network trained under
    ``forward_noiser_reverse=true``, i.e. it maps CARN level n -> n-1).

    Same relaxed multi-step Euler loop as the training-side helper::

        cur <- cur + alpha_k * N(cur, level_k, residual=False)
        alpha_k = alpha0 * alpha_decay**k,  level_k = level - k

    The network was trained with ``residual=True``, so ``residual=False``
    returns the raw increment ``delta`` and ``cur + alpha*delta`` is a
    decelerating (no-overshoot) step toward the cleaner manifold.

    Returns ``chunk`` unchanged when ``level < min_level`` (near the
    manifold the corrector is ~identity thanks to the zero-init
    ``out_proj``, so the ~23M-param conv is skipped).

    ``noiser``'s params are temporarily ``requires_grad_(False)`` (saved
    and restored) purely to mirror the training-side discipline; the whole
    eval script already runs under ``@torch.no_grad()`` so this is
    belt-and-braces, not load-bearing.
    """
    level = int(level)
    min_level = int(min_level)
    if noiser is None or level < min_level:
        return chunk

    inner = noiser.module if hasattr(noiser, "module") else noiser
    n_steps = max(1, int(steps))
    a0 = float(alpha0)
    decay = float(alpha_decay)

    params = list(inner.parameters())
    saved = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(False)
    try:
        n_dtype = next(inner.parameters()).dtype
        cur = chunk
        for k in range(n_steps):
            lvl = max(0, level - k)
            if lvl < min_level:
                break
            alpha = a0 * (decay ** k)
            cs = torch.full(
                (cur.shape[0],), lvl, dtype=torch.long, device=cur.device,
            )
            delta = inner(cur.to(dtype=n_dtype), cs, residual=False)
            cur = cur + alpha * delta.to(dtype=cur.dtype)
        if _telemetry is not None:
            _telemetry["calls"] = int(_telemetry.get("calls", 0)) + 1
            _telemetry["rel_dz"] = float(
                (cur - chunk).norm() / max(float(chunk.norm()), 1e-8)
            )
        return cur
    finally:
        for p, r in zip(params, saved):
            p.requires_grad_(r)


def _resolve_trained_attn_window(base_dit, fallback_frames: int):
    """Resolve the student's TRAINED local attention window, in FRAMES.

    FIX 2 support (2026-08-25). Prefer what the loaded model/config
    actually says over a hardcoded constant, so a student trained at a
    different window is served at ITS window, not at 21.

    Probes, in order:
      1. ``base_dit.local_attn_size``          (the DiT's own config)
      2. any sub-module's ``local_attn_size``  (attention blocks)
      3. ``fallback_frames``                   (TEACHER_ATTN_FRAMES = 21)

    A value of ``-1`` / ``None`` / non-int means "global attention, not
    configured" and is SKIPPED -- it is not a trained window. Returns
    ``(frames, source_string)``; the source is printed at eval start so
    a scored number is never ambiguous about where its span came from.
    """
    def _ok(v):
        if v is None or isinstance(v, (list, tuple, bool)):
            return None
        try:
            iv = int(v)
        except (TypeError, ValueError):
            return None
        return iv if iv > 0 else None

    v = _ok(getattr(base_dit, "local_attn_size", None))
    if v is not None:
        return v, "model.local_attn_size"
    try:
        for name, module in base_dit.named_modules():
            v = _ok(getattr(module, "local_attn_size", None))
            if v is not None:
                return v, f"model.{name}.local_attn_size"
    except Exception:
        pass
    return int(fallback_frames), f"fallback:TEACHER_ATTN_FRAMES={fallback_frames}"


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
        # VRFM: the eval config has no ode_vrfm key, so without this the probe
        # would build vrfm=None and serve a VRFM-trained student with the
        # z-AdaLN contribution identically ZERO -- reading exactly like "the
        # method did nothing" instead of erroring. Same env-var handshake the
        # rollout stage uses for ODE_ROLLOUT.
        if os.environ.get("ODE_VRFM"):
            cfg.ode_vrfm = True
            cfg.ode_vrfm_zdim = int(os.environ.get("ODE_VRFM_ZDIM", 128))
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
    def load_checkpoint(
        self,
        student_ckpt: str,
        vinfo=None,
        *,
        use_ema: bool = False,
    ):  # type: ignore[override]
        """Overlay the student weights onto the teacher-merged base.

        ``vinfo`` is ignored (kept for signature parity with ``ChainPipeline``).
        Returns the embedded ``step`` from the checkpoint (or -1).
        """
        log.info("[%s] Overlaying student %s", self.device, student_ckpt)
        raw = torch.load(student_ckpt, map_location="cpu", weights_only=False)
        generator_key = "generator_ema" if use_ema else "generator"
        try:
            generator_sd = raw[generator_key]
        except KeyError as exc:
            raise RuntimeError(
                f"Student checkpoint {student_ckpt} missing {generator_key!r}; "
                "expected a full-rank trainer snapshot from action_ode_distill."
            ) from exc

        self.ode_model.generator.model.load_state_dict(generator_sd, strict=True)
        log.info(
            "[%s] Loaded %s student weights.",
            self.device,
            "EMA" if use_ema else "online",
        )

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
        # VRFM: the CONDITIONAL PRIOR p(z|x0,xt,t,a) is what inference samples
        # from, so it must come out of the checkpoint. Without it the student
        # would be evaluated with z=0 -- i.e. with the very conditioning it was
        # trained to use switched off, which silently looks like "VRFM did
        # nothing" rather than an error.
        for _vn in ("vrfm", "z_modulation"):
            _vm = getattr(self.ode_model, _vn, None)
            if _vm is not None:
                if _vn not in raw:
                    raise RuntimeError(
                        f"checkpoint has no '{_vn}' but the eval model was "
                        f"built with ode_vrfm=true; refusing to evaluate a "
                        f"VRFM student with an untrained prior.")
                _vm.load_state_dict(raw[_vn])
                _vm.eval()

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
        z_lat: Optional[torch.Tensor] = None,
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
        if z_lat is not None:
            _zm = self.ode_model.z_modulation(z_lat, num_frames=num_frames)
            cond["_action_modulation"] = (
                cond["_action_modulation"] + _zm if "_action_modulation" in cond else _zm)
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
        carn_seam_affine_lambda: float = 0.0,
        reverse_noiser: Optional[torch.nn.Module] = None,
        reverse_noiser_dedrift_level: int = 1,
        reverse_noiser_dedrift_steps: int = 1,
        reverse_noiser_dedrift_alpha0: float = 1.0,
        reverse_noiser_dedrift_alpha_decay: float = 0.5,
        reverse_noiser_dedrift_min_level: int = 1,
        reverse_noiser_dedrift_level_auto: bool = False,
        reverse_noiser_dedrift_level_max: int = 16,
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
        # TEACHER PARITY (utils/causal_chain_rollout.py: local_attn_chunks=7,
        # LOCAL_ATTN_F=21, target_max=21*FRAME_SEQ). The cache may be deeper than
        # the attention span -- the teacher allocates 27 frames but HARD-CAPS
        # attention at 21 -- but the span itself must not exceed what the student
        # was trained at (21-frame clean window), or late chunks attend across
        # query-key distances never seen in training and degrade per chunk.
        TEACHER_ATTN_FRAMES = 21
        _cache_frames = (cache_chunks + chunks_per_step) * BASE_CHUNK_FRAMES
        # ===============================================================
        # FIX 2 -- INFERENCE ATTENTION SPAN (researcher-ordered, 2026-08-25)
        # DEFAULT-BEHAVIOUR CHANGE. ``eval_span_match_training`` DEFAULTS
        # TO TRUE (aligned); set EVAL_SPAN_MATCH_TRAINING=0 for the legacy
        # behaviour.
        #
        # WHAT WAS WRONG. ``max_tokens`` was ``kv_cache_tokens`` -- the
        # size of the ALLOCATED KV BUFFER (cache_chunks+chunks_per_step =
        # 24 frames), not the window the student was TRAINED at (21
        # frames). Training attends 21
        # (pipeline/action_forcing_training.py:299-306, local_attn_size *
        # frame_seq_length) and so does the 14e ODE stage
        # (pipeline/ode_rollout.py). Eval was the only outlier, and the
        # comment directly above -- "the span itself must not exceed what
        # the student was trained at" -- already stated the correct
        # intent while the code did the opposite. Late chunks therefore
        # attended across query-key distances never seen in training.
        #
        # THE FIX. The cache is STILL ALLOCATED at ``kv_cache_tokens``
        # (the deeper buffer is what the rolling/eviction bookkeeping
        # expects); only the ATTENTION SPAN is capped at the trained
        # window. Teacher parity is preserved: the teacher likewise
        # allocates 27 frames and hard-caps attention at 21.
        #
        # THIS CHANGES PREVIOUSLY-SCORED NUMBERS. Any eval run before
        # 2026-08-25 attended 24 frames. The effective span and where it
        # came from are printed once at eval start (below) so no result
        # is ever ambiguous about which contract produced it.
        # ===============================================================
        _win_frames, _win_src = _resolve_trained_attn_window(
            base_dit, TEACHER_ATTN_FRAMES,
        )
        if "ODE_ATTN_FRAMES" in os.environ:
            local_attn_size_frames = int(os.environ["ODE_ATTN_FRAMES"])
            _win_src = "env:ODE_ATTN_FRAMES"
        else:
            local_attn_size_frames = int(_win_frames)
        kv_cache_tokens = max(_cache_frames, local_attn_size_frames) * frame_seq_length
        required_chunk_tokens = num_frame_per_block * frame_seq_length
        self.wrapper.seq_len = max(int(self.wrapper.seq_len), required_chunk_tokens)
        _span_match = os.environ.get(
            "EVAL_SPAN_MATCH_TRAINING", "1",
        ).strip().lower() not in ("0", "false", "no", "off", "")
        _span_tokens = (
            local_attn_size_frames * frame_seq_length
            if _span_match
            else kv_cache_tokens
        )
        # REQUIRED one-shot announcement (see the block comment above):
        # the effective span AND its provenance, every eval, at start.
        log.info(
            "[AR][span] eval_span_match_training=%s | attention span = "
            "%d frames (%d tokens) from %s | KV buffer allocated at "
            "%d frames (%d tokens) | frame_seq_length=%d%s",
            _span_match,
            _span_tokens // frame_seq_length, _span_tokens, _win_src,
            kv_cache_tokens // frame_seq_length, kv_cache_tokens,
            frame_seq_length,
            "" if _span_match else
            "  <-- LEGACY: span follows the BUFFER, not the trained window",
        )
        _set_attention_window(
            base_dit,
            local_attn_size_frames=local_attn_size_frames,
            max_tokens=_span_tokens,
        )
        # TEACHER PARITY, and it MUST precede the cache prefill below:
        # the seed/clean-fill forwards write K/V into the cache, so if the
        # flag is set after them the seed context is cached with the
        # sheared rotation. causal_chain_rollout.py sets this before any
        # forward and RESTORES it in a finally; do both.
        _apf = int(getattr(base_dit, "action_tokens_per_frame", 0) or 0)
        _cra_prev = []
        if _apf > 0:
            for _m in base_dit.modules():
                if hasattr(_m, "local_attn_size"):
                    _cra_prev.append((_m, getattr(_m, "cached_rope_action_aware", None)))
                    _m.cached_rope_action_aware = True

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

        carn_seam_affine_lambda = float(carn_seam_affine_lambda)
        if not 0.0 <= carn_seam_affine_lambda <= 1.0:
            raise SystemExit(
                "[AR] --carn_seam_affine_lambda must be in [0, 1], got "
                f"{carn_seam_affine_lambda}"
            )
        carn_target = None
        if carn_seam_affine_lambda > 0.0:
            seed_float = initial_latents_dev.float()
            carn_target = (
                seed_float.mean(dim=(0, 1, 3, 4)),
                seed_float.std(dim=(0, 1, 3, 4)),
            )
            log.info(
                "[AR] CARN seam affine enabled at commit: lambda=%.3f, "
                "target=real prefill per-channel mean/std",
                carn_seam_affine_lambda,
            )

        # Reverse-noiser de-drift at commit. Off (byte-identical to the
        # pre-existing path) whenever ``reverse_noiser is None``.
        dedrift_tel = None
        if reverse_noiser is not None:
            dedrift_tel = {"calls": 0, "rel_dz": 0.0}
            log.info(
                "[AR] reverse-noiser de-drift enabled at commit: level=%d "
                "steps=%d alpha0=%.3f decay=%.3f min_level=%d "
                "level_auto=%s level_max=%d",
                int(reverse_noiser_dedrift_level),
                int(reverse_noiser_dedrift_steps),
                float(reverse_noiser_dedrift_alpha0),
                float(reverse_noiser_dedrift_alpha_decay),
                int(reverse_noiser_dedrift_min_level),
                bool(reverse_noiser_dedrift_level_auto),
                int(reverse_noiser_dedrift_level_max),
            )

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
        # ODE_STAT_LOCK=1: serve-time inversion of the measured per-chunk
        # affine variance contraction (s_c = s_GT * k^(c+1), k~0.917 for
        # the 4-rung ladder; see flow_viz/STAT_FORENSICS_REPORT.txt).
        # Reference = the REAL seed context's per-channel mean/std; each
        # committed chunk is re-pinned to it BEFORE emission + cache
        # refresh, so the contraction cannot compound through self-context.
        _sl_ref = None
        if os.environ.get("ODE_STAT_LOCK"):
            _r = initial_latents_dev.to(torch.float32)
            _sl_ref = (_r.mean(dim=(0, 1, 3, 4)), _r.std(dim=(0, 1, 3, 4)))
        _fr_steps: list = []                       # ARRWM flow-viz hook records

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

            # ARRWM flow-viz hook (env-gated ODE_FLOW_REC): pin the block
            # noise to the SAME generator stream as utils/flow_record.py's
            # teacher recording (seed_base + block index) so student and
            # teacher trajectories share their initial noise exactly, and
            # record x_t / pred_x0 at every few-step iteration.
            _fr_dir = os.environ.get("ODE_FLOW_REC")
            _fr_gen = None
            if _fr_dir:
                _fr_gen = torch.Generator(device=self.device).manual_seed(
                    int(os.environ.get("ODE_FLOW_SEED", "1234")) + step_idx)
            noise = torch.randn(
                [B, num_frame_per_block, C, H, W],
                dtype=torch.float32, device=self.device, generator=_fr_gen,
            )
            x = noise.to(self.dtype)
            if _fr_dir:
                _fr_steps.append((step_idx, -1, float(ts[0].item()),
                                  x.detach().float().to(torch.float16).cpu().numpy()))

            pred_x0: Optional[torch.Tensor] = None
            # VRFM: x_0 is this chunk's INITIAL noise, the same quantity the
            # encoders saw at training time.
            _x0_lat = x
            for d_idx in range(int(ts.shape[0])):
                t_val = float(ts[d_idx].item())
                tt = torch.full(
                    [B, num_frame_per_block], t_val,
                    device=self.device, dtype=torch.float32,
                )
                _cond = cond
                _vrfm = getattr(self.ode_model, "vrfm", None)
                if _vrfm is not None:
                    # z ~ p(.|x0, xt, t, a): the PRIOR only. x1 (the clean
                    # target) does not exist here, which is exactly why the
                    # prior was made conditional on everything else.
                    _z, _, _ = _vrfm(
                        x0=_x0_lat.float(), xt=x.float(),
                        t=tt.reshape(-1)[:1],
                        action=block_fa.reshape(1, -1)[:, :_vrfm.action_dim], x1=None)
                    _cond = self._build_action_cond_chunk(
                        prompt_embeds_dev, block_fa,
                        num_frames=num_frame_per_block, z_lat=_z)
                with torch.amp.autocast("cuda", dtype=self.dtype):
                    out = self.wrapper(
                        noisy_image_or_video=x,
                        conditional_dict=_cond,
                        timestep=tt,
                        kv_cache=kv_cache,
                        crossattn_cache=crossattn_cache,
                        current_start=current_start_frame * frame_seq_length,
                    )
                pred_x0 = out[1]
                # KLTS serve parity: apply the LEARNED temperature on the same
                # chord the training applied it to — pred' = x + tau*(pred - x)
                # — or the probe would measure a different sampler than was
                # trained (the mismatch this campaign exists to kill). Table
                # loaded lazily once from ODE_KLTS_CKPT. Chunk index is
                # clamped into the 8-row table; serve counts bootstrap blocks
                # that training's chunk index does not — a <=2-row shift on a
                # slowly-varying table, accepted for v1.
                _ktf = float(os.environ.get("ODE_KLTS_FIXED", "0") or 0)
                if _ktf and _ktf != 1.0:
                    # serve-time fixed-temperature override (tau sweep): no
                    # table, no ckpt — one scalar on the same chord.
                    _xf = x.float()
                    pred_x0 = (_xf + _ktf * (pred_x0.float() - _xf)
                               ).to(pred_x0.dtype)
                if os.environ.get("ODE_KLTS_CKPT"):
                    if not hasattr(self, "_klts_theta"):
                        _sd = torch.load(os.environ["ODE_KLTS_CKPT"],
                                         map_location="cpu", weights_only=False)
                        _t = _sd.get("klts_theta")
                        self._klts_theta = (None if _t is None
                                            else _t.to(self.device).float())
                        del _sd
                    if self._klts_theta is not None:
                        _th = self._klts_theta[
                            min(step_idx, self._klts_theta.shape[0] - 1),
                            min(d_idx, self._klts_theta.shape[1] - 1)]
                        _xf = x.float()
                        pred_x0 = (_xf + (1.0 + _th) * (pred_x0.float() - _xf)
                                   ).to(pred_x0.dtype)
                _acfg = float(os.environ.get("ODE_ACTION_CFG", "0") or 0)
                if _acfg and _acfg != 1.0:
                    # Action classifier-free guidance: second forward with
                    # the NULL (no-op, z=0) action, extrapolate the action
                    # effect: pred = pred_null + s * (pred_act - pred_null).
                    # Meaningful when trained with action-cfg dropout.
                    if "_cfg_null_cond" not in locals() or _cfg_null_cond_frame != cur_frame_lo:
                        _null_fa = torch.zeros_like(block_fa)
                        _cfg_null_cond = self._build_action_cond_chunk(
                            prompt_embeds_dev, _null_fa,
                            num_frames=num_frame_per_block)
                        _cfg_null_cond_frame = cur_frame_lo
                    with torch.amp.autocast("cuda", dtype=self.dtype):
                        out_n = self.wrapper(
                            noisy_image_or_video=x,
                            conditional_dict=_cfg_null_cond,
                            timestep=tt,
                            kv_cache=kv_cache,
                            crossattn_cache=crossattn_cache,
                            current_start=current_start_frame * frame_seq_length,
                        )
                    pred_x0 = out_n[1] + _acfg * (pred_x0 - out_n[1])
                if _fr_dir:                        # x0 prediction at this rung
                    _fr_steps.append((step_idx, d_idx, t_val,
                                      pred_x0.detach().float().to(torch.float16).cpu().numpy()))
                _samp = os.environ.get("ODE_SAMPLER", "euler").lower()
                if _samp in ("midpoint", "heun") and d_idx < int(ts.shape[0]) - 1:
                    # Higher-order DETERMINISTIC flow step between rungs
                    # instead of the x0-restart re-noise. Velocity is
                    # derived from the model's own x0 prediction,
                    #     x_s = (1-s) x0 + s eps  =>  v = dx/ds = (x-x0)/s,
                    # so this is independent of the wrapper's flow-pred
                    # sign convention. Costs one extra forward per rung.
                    next_t = float(ts[d_idx + 1].item())
                    s_a, s_b = t_val / 1000.0, next_t / 1000.0
                    _xa = x.float()
                    _va = (_xa - pred_x0.float()) / max(s_a, 1e-6)

                    def _pred_at(_xin, _tv):
                        _tt = torch.full([B, num_frame_per_block], _tv,
                                         device=self.device, dtype=torch.float32)
                        with torch.amp.autocast("cuda", dtype=self.dtype):
                            _o = self.wrapper(
                                noisy_image_or_video=_xin.to(self.dtype),
                                conditional_dict=cond,
                                timestep=_tt,
                                kv_cache=kv_cache,
                                crossattn_cache=crossattn_cache,
                                current_start=current_start_frame * frame_seq_length,
                            )
                        return _o[1].float()

                    if _samp == "midpoint":
                        s_m = 0.5 * (s_a + s_b)
                        _xm = _xa + (s_m - s_a) * _va
                        _vm = (_xm - _pred_at(_xm, s_m * 1000.0)) / max(s_m, 1e-6)
                        _xb = _xa + (s_b - s_a) * _vm
                    else:                                  # heun (trapezoid)
                        _xe = _xa + (s_b - s_a) * _va
                        _ve = (_xe - _pred_at(_xe, next_t)) / max(s_b, 1e-6)
                        _xb = _xa + (s_b - s_a) * 0.5 * (_va + _ve)
                    x = _xb.to(self.dtype)
                    if _fr_dir:
                        _fr_steps.append((step_idx, d_idx, -next_t,
                                          x.detach().float().to(torch.float16).cpu().numpy()))
                elif d_idx < int(ts.shape[0]) - 1:
                    next_t = float(ts[d_idx + 1].item())
                    flat = pred_x0.flatten(0, 1).float()
                    if _fr_dir and os.environ.get("ODE_FLOW_DET"):
                        # deterministic (rectified-flow straight-path) chaining:
                        # re-noise with the block's ORIGINAL noise instead of a
                        # fresh draw, so rung k+1 stays on rung k's path
                        flat_noise = noise.flatten(0, 1).float()
                    elif _fr_gen is not None:
                        flat_noise = torch.randn(flat.shape, device=flat.device,
                                                 dtype=flat.dtype, generator=_fr_gen)
                    else:
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
                    if _fr_dir:                    # re-noised input for next rung
                        _fr_steps.append((step_idx, d_idx, -next_t,
                                          x.detach().float().to(torch.float16).cpu().numpy()))
            assert pred_x0 is not None

            _sl_mode = os.environ.get("ODE_STAT_LOCK")
            if _sl_mode == "repnudge":
                # Closed-form gaussian-signature repulsor at serve: one
                # gradient step of the inverse-square potential in moment
                # coordinates. D = ||mu||^2 + ||sigma-1||^2; update
                # mu *= 1 + 2*eta/D^2, sigma += 2*eta*(sigma-1)/D^2 —
                # strength grows as the state nears the noise signature.
                _eta = float(os.environ.get("ODE_REPNUDGE_ETA", "0.5"))
                _p = pred_x0.to(torch.float32)
                _mu = _p.mean(dim=(0, 1, 3, 4))
                _sd = _p.std(dim=(0, 1, 3, 4))
                _D = (_mu.pow(2).sum() + (_sd - 1.0).pow(2).sum()).clamp(min=1e-3)
                _g = 2.0 * _eta / _D.pow(2)
                _mu2 = _mu * (1.0 + _g)
                _sd2 = (_sd + _g * (_sd - 1.0)).clamp(min=1e-4)
                _muv = _mu.view(1, 1, -1, 1, 1)
                pred_x0 = (
                    (_p - _muv) / (_sd.view(1, 1, -1, 1, 1) + 1e-6)
                    * _sd2.view(1, 1, -1, 1, 1) + _mu2.view(1, 1, -1, 1, 1)
                ).to(pred_x0.dtype)
            elif _sl_mode == "chroma" and _sl_ref is not None:
                # Chroma-subspace lock (lit-review-guided): correct latent
                # channel MEANS only along the fitted decoder-visible
                # chroma directions (Cb/Cr; R^2 ~0.96), leaving the ~14
                # motion/luminance-carrying mean directions untouched.
                # Per-channel std pinned to seed (proven motion-safe).
                if not hasattr(self, "_chroma_map"):
                    import numpy as _np
                    _cz = _np.load(os.environ.get(
                        "ODE_SL_CHROMA",
                        "/scratch/u6ex/as1748.u6ex/ARRWM/analysis/"
                        "eval_final/flow_viz/chroma_subspace.npz"))
                    _Mc = torch.from_numpy(_cz["Mc"]).float()      # [2,16]
                    self._chroma_map = (
                        _Mc.to(self.device),
                        torch.linalg.pinv(_Mc).to(self.device))    # [16,2]
                _Mc, _Mp = self._chroma_map
                _p = pred_x0.to(torch.float32)
                _mu = _p.mean(dim=(0, 1, 3, 4))                    # [16]
                _sd = _p.std(dim=(0, 1, 3, 4)).view(1, 1, -1, 1, 1)
                _mu_ref = _sl_ref[0]
                # move means only along chroma pseudo-inverse directions
                _dmu = (_Mp @ (_Mc @ (_mu_ref - _mu))).view(1, 1, -1, 1, 1)
                pred_x0 = (
                    (_p - _p.mean(dim=(0, 1, 3, 4)).view(1, 1, -1, 1, 1))
                    / (_sd + 1e-6) * _sl_ref[1].view(1, 1, -1, 1, 1)
                    + _mu.view(1, 1, -1, 1, 1) + _dmu
                ).to(pred_x0.dtype)
            elif _sl_mode == "hyb" and _sl_ref is not None:
                # Hybrid: hard-pin per-channel STD to the seed anchor
                # (contrast is where the contraction is visible and is
                # near-stationary in GT) but leave channel MEANS free with
                # only the inverse-gain correction (egomotion partly lives
                # in mean shifts; anchoring them halves realized motion).
                _kmu = min(1.0 / float(os.environ.get("ODE_SL_KMU", "0.894")), 1.2)
                _p = pred_x0.to(torch.float32)
                _mu = _p.mean(dim=(0, 1, 3, 4)).view(1, 1, -1, 1, 1)
                _sd = _p.std(dim=(0, 1, 3, 4)).view(1, 1, -1, 1, 1)
                pred_x0 = (
                    _mu * _kmu + (_p - _mu) / (_sd + 1e-6)
                    * _sl_ref[1].view(1, 1, -1, 1, 1)
                ).to(pred_x0.dtype)
            elif _sl_mode == "inv":
                # Inverse-gain mode: cancel the MEASURED per-chunk affine
                # contraction (std x k_sd, mean x k_mu toward zero) without
                # anchoring content to any fixed reference — hard-pinning to
                # the seed stats was shown to also cancel legitimate scene
                # evolution (halved realized motion). Gains are measured per
                # checkpoint (arm-D: k_sd 0.9565, k_mu 0.894) and clamped.
                _ksd = min(1.0 / float(os.environ.get("ODE_SL_KSTD", "0.9565")), 1.1)
                _kmu = min(1.0 / float(os.environ.get("ODE_SL_KMU", "0.894")), 1.2)
                _p = pred_x0.to(torch.float32)
                _mu = _p.mean(dim=(0, 1, 3, 4)).view(1, 1, -1, 1, 1)
                pred_x0 = (_mu * _kmu + (_p - _mu) * _ksd).to(pred_x0.dtype)
            elif _sl_ref is not None:
                # Hard-pin mode: per-channel affine re-pin to seed stats.
                _p = pred_x0.to(torch.float32)
                _mu = _p.mean(dim=(0, 1, 3, 4)).view(1, 1, -1, 1, 1)
                _sd = _p.std(dim=(0, 1, 3, 4)).view(1, 1, -1, 1, 1)
                pred_x0 = (
                    (_p - _mu) / (_sd + 1e-6)
                    * _sl_ref[1].view(1, 1, -1, 1, 1)
                    + _sl_ref[0].view(1, 1, -1, 1, 1)
                ).to(pred_x0.dtype)

            if os.environ.get("ODE_GEXCL"):
                # AXIS 2 at serve: BOUNDED gaussian exclusion on the
                # committed chunk. Push the moment vector radially away
                # from the gaussian signature only while it is inside r0,
                # and never past r0 — so unlike the inverse-square
                # repnudge there is no unbounded expansion.
                _eta = float(os.environ["ODE_GEXCL"])
                _r0 = float(os.environ.get("ODE_GEXCL_R0", "1.0"))
                _p = pred_x0.to(torch.float32)
                _mu = _p.mean(dim=(0, 1, 3, 4))
                _sd = _p.std(dim=(0, 1, 3, 4))
                _dev = torch.cat([_mu, _sd - 1.0])
                _d = _dev.norm().clamp(min=1e-6)
                if float(_d) < _r0:
                    _dt = min(_r0, float(_d) * (1.0 + _eta))
                    _k = _dt / float(_d)
                    _mu2 = _mu * _k
                    _sd2 = (1.0 + (_sd - 1.0) * _k).clamp(min=1e-4)
                    pred_x0 = (
                        (_p - _mu.view(1, 1, -1, 1, 1))
                        / (_sd.view(1, 1, -1, 1, 1) + 1e-6)
                        * _sd2.view(1, 1, -1, 1, 1)
                        + _mu2.view(1, 1, -1, 1, 1)
                    ).to(pred_x0.dtype)

            if os.environ.get("ODE_EMDHEAD"):
                # Zero-init LEARNED transport head (AdaLN-zero analog):
                # per-channel affine y = x*(1+s)+b trained at commit-rung
                # by the EMD objectives (ode_emdhead_* arms); applied to
                # every committed chunk before emission + cache refresh.
                if not hasattr(self, "_emdhead_sb"):
                    _hck = torch.load(os.environ["ODE_EMDHEAD_CKPT"],
                                      map_location="cpu")
                    _eh = _hck.get("emd_head")
                    if _eh is None:
                        raise RuntimeError(
                            "ODE_EMDHEAD set but checkpoint has no emd_head")
                    self._emdhead_sb = (
                        _eh["scale"].float().to(self.device),
                        _eh["shift"].float().to(self.device))
                    log.info("[AR] EMD head loaded: scale %s shift %s",
                             self._emdhead_sb[0].tolist(),
                             self._emdhead_sb[1].tolist())
                _es, _eb = self._emdhead_sb
                _p = pred_x0.to(torch.float32)
                pred_x0 = (_p * (1.0 + _es.view(1, 1, -1, 1, 1))
                           + _eb.view(1, 1, -1, 1, 1)).to(pred_x0.dtype)

            if os.environ.get("ODE_EMDREMAP"):
                # Closed-form serve-time EMD transport: per-channel
                # monotone quantile remap of the committed chunk toward
                # the REAL seed context's quantile profile (exact 1-D OT
                # map, blend lambda in [0,1]). Chunk-clock by
                # construction — applied where d exists, flow map never
                # touched.
                _lamenv = os.environ["ODE_EMDREMAP"]
                _p = pred_x0.to(torch.float32)
                _B, _F, _C, _H, _W = _p.shape
                _v = _p.permute(2, 0, 1, 3, 4).reshape(_C, -1)
                if (not hasattr(self, "_emdremap_ref")
                        or self._emdremap_ref.shape[1] != _v.shape[1]):
                    _r = initial_latents_dev.to(torch.float32)
                    _rv = _r.permute(2, 0, 1, 3, 4).reshape(_C, -1)
                    _n = _v.shape[1]
                    _pq = (torch.arange(_n, device=_v.device,
                                        dtype=torch.float32) + 0.5) / _n
                    self._emdremap_ref = torch.quantile(
                        _rv, _pq, dim=1).T.contiguous()
                    self._emdremap_refsd = _rv.std(dim=1)
                # Optional per-channel action-sensitivity protection
                # (lit-review 2026-08-07): scale correction inversely
                # with each channel's measured same-context action-fan
                # variance so action-carrying channels are remapped less.
                if (os.environ.get("ODE_EMDREMAP_ASENS")
                        and not hasattr(self, "_emdremap_asens")):
                    import numpy as _np
                    self._emdremap_asens = torch.from_numpy(
                        _np.load(os.environ["ODE_EMDREMAP_ASENS"])["w"]
                    ).float().to(_v.device).view(_C, 1)
                if _lamenv.startswith("auto"):
                    # Adaptive per-channel blend: strength proportional to
                    # the channel's measured contraction deficit vs the
                    # seed reference — healthy channels untouched, fully
                    # collapsed channels fully remapped. auto[:eta]
                    # scales the response (default 1).
                    _eta = float(_lamenv.split(":")[1]) \
                        if ":" in _lamenv else 1.0
                    _sd_c = _v.std(dim=1).clamp(min=1e-6)
                    _lam = ((self._emdremap_refsd / _sd_c - 1.0) * _eta) \
                        .clamp(0.0, 1.0).view(_C, 1)
                else:
                    _lam = float(_lamenv)
                if hasattr(self, "_emdremap_asens"):
                    _lam = _lam * self._emdremap_asens
                _srt, _idx = _v.sort(dim=1)
                _blend = (1.0 - _lam) * _srt + _lam * self._emdremap_ref
                _out = torch.empty_like(_v).scatter_(1, _idx, _blend)
                pred_x0 = (_out.view(_C, _B, _F, _H, _W)
                           .permute(1, 2, 0, 3, 4).to(pred_x0.dtype))

            # Reverse-noiser de-drift runs BEFORE the seam affine, on the
            # same tensor that is both emitted and pushed into the KV cache
            # (this is the AR feedback point). Ordering rationale: the
            # de-drift is a learned, content-aware correction whose training
            # inputs were RAW student rollout chunks, so it must see a raw
            # chunk; the seam affine is a cheap moment re-anchor that is
            # correct to apply last (it restores the target mean/std no
            # matter what preceded it). NOTE: stacking both is NOT a
            # validated combination -- the smoke exercises exactly one at a
            # time.
            if reverse_noiser is not None:
                # Level = the CARN drift level of the tensor being fed IN.
                # The trainer's reverse pairing conditions on the ROLLOUT2
                # (= input) level, counting the first generated chunk as 1
                # (trainer/causal_action_forcing_train.py:11838-11851), so
                # ``level_auto`` reproduces that indexing here: the k-th
                # committed chunk is at level k+1, clamped to the FiLM
                # embedding's range. Default OFF = the fixed level below.
                if reverse_noiser_dedrift_level_auto:
                    _dd_level = min(
                        len(generated) + 1,
                        int(reverse_noiser_dedrift_level_max),
                    )
                else:
                    _dd_level = int(reverse_noiser_dedrift_level)
                pred_x0 = _apply_reverse_noiser_dedrift(
                    pred_x0,
                    reverse_noiser,
                    level=_dd_level,
                    steps=int(reverse_noiser_dedrift_steps),
                    alpha0=float(reverse_noiser_dedrift_alpha0),
                    alpha_decay=float(reverse_noiser_dedrift_alpha_decay),
                    min_level=int(reverse_noiser_dedrift_min_level),
                    _telemetry=dedrift_tel,
                )
                if dedrift_tel is not None:
                    log.info(
                        "[AR][DEDRIFT] call=%d level=%d rel|dz|=%.6f",
                        dedrift_tel["calls"], _dd_level,
                        dedrift_tel["rel_dz"],
                    )

            if carn_target is not None:
                pred_x0 = _apply_carn_seam_affine(
                    pred_x0,
                    target_mean=carn_target[0],
                    target_std=carn_target[1],
                    strength=carn_seam_affine_lambda,
                )

            generated.append(pred_x0.detach().to(torch.float32))

            # Cheap per-chunk drift telemetry (log only -- no behaviour
            # change). Contraction of ``std`` over the rollout is the AR
            # variance-contraction signature we want to compare between the
            # de-drift-on and de-drift-off arms.
            _gc = generated[-1]
            log.info(
                "[AR][STATS] chunk=%d mean=%.5f std=%.5f absmax=%.4f",
                len(generated) - 1, float(_gc.mean()), float(_gc.std()),
                float(_gc.abs().max()),
            )

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
        if _fr_steps:                              # ARRWM flow-viz hook save
            import numpy as _np
            _d = os.environ["ODE_FLOW_REC"]
            os.makedirs(_d, exist_ok=True)
            _np.savez_compressed(
                os.path.join(_d, "steps.npz"),
                sdt=_np.array([(s, d, t) for s, d, t, _ in _fr_steps], dtype=_np.float64),
                **{f"x{j}": x for j, (_, _, _, x) in enumerate(_fr_steps)})
            log.info("[flowrec] saved %s/steps.npz (%d records)", _d, len(_fr_steps))
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
    # ``encode_z_actions_window`` aligns motion for the WHOLE ``n_latent_frames``
    # it is handed and raises if that exceeds the motion-file coverage
    # (head-drop + chunk_offset). The manifest path is pre-capped via
    # ``_index_single_zarr``; the disk-fallback path (manifest_path="") returns
    # the UNcapped full ride length, so cap it here to the motion-capped length.
    # We only ever slice ``[latent_start_offset : latent_start_offset+total_frames)``
    # (= ``need``), which is far below the cap, so this never starves the window.
    from utils.zarr_dataset import _motion_capped_latents
    _mcap = _motion_capped_latents(ride_dict["attrs"], Path(motion_root))
    n_lat_enc = min(n_lat, _mcap) if _mcap > 0 else n_lat
    if n_lat_enc < need:
        raise RuntimeError(
            f"Motion-capped ride length {n_lat_enc} (cap={_mcap}) < required "
            f"window {need} for {zarr_basename}; cannot encode z-actions."
        )
    z_win = z_ds.encode_z_actions_window(
        zpath, n_lat_enc, latent_start_offset, latent_start_offset + total_frames,
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
    parser.add_argument(
        "--use_ema", action="store_true",
        help="Evaluate the checkpoint's generator_ema weights instead of its "
             "online generator weights.",
    )
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
    parser.add_argument("--carn_seam_affine_lambda", type=float, default=0.0,
                        help="AR mode only: apply the CARN per-channel mean/std "
                             "re-anchor to each generated chunk before emission and "
                             "KV-cache commit. The target is measured once from the "
                             "real prefill; 0 disables it and the precedent uses 0.5.")
    parser.add_argument("--reverse_noiser_checkpoint", type=str, default="",
                        help="AR mode only: path to a ForwardNoiser state_dict "
                             "trained with forward_noiser_reverse=true (a "
                             "de-CARN / de-drift network, e.g. a fn_rev_step*.pt "
                             "written by the trainer). When set, each generated "
                             "chunk is de-drifted through the frozen network "
                             "before emission and KV-cache commit. Empty "
                             "(default) = disabled, no new code path runs.")
    parser.add_argument("--reverse_noiser_latent_channels", type=int, default=16,
                        help="ForwardNoiser latent_channels (Wan2.1 VAE = 16).")
    parser.add_argument("--reverse_noiser_hidden_dim", type=int, default=512,
                        help="ForwardNoiser hidden_dim (must match the "
                             "checkpoint; trainer default 512).")
    parser.add_argument("--reverse_noiser_num_blocks", type=int, default=4,
                        help="ForwardNoiser num_blocks (must match the "
                             "checkpoint; trainer default 4).")
    parser.add_argument("--reverse_noiser_max_carn_step", type=int, default=16,
                        help="ForwardNoiser max_carn_step (trainer default 16). "
                             "Not a parameter shape, but keep it matched.")
    parser.add_argument("--reverse_noiser_dedrift_level", type=int, default=1,
                        help="CARN level the committed chunk is assumed to sit "
                             "at (the level the de-drift starts FROM).")
    parser.add_argument("--reverse_noiser_dedrift_steps", type=int, default=1,
                        help="Euler steps of the de-drift loop (level "
                             "decrements by 1 each step).")
    parser.add_argument("--reverse_noiser_dedrift_alpha0", type=float, default=1.0,
                        help="Step size of the first de-drift Euler step.")
    parser.add_argument("--reverse_noiser_dedrift_alpha_decay", type=float,
                        default=0.5,
                        help="Geometric decay of the de-drift step size.")
    parser.add_argument("--reverse_noiser_dedrift_min_level", type=int, default=1,
                        help="Skip the de-drift entirely below this level.")
    parser.add_argument("--reverse_noiser_dedrift_level_auto",
                        action="store_true",
                        help="Ignore --reverse_noiser_dedrift_level and use the "
                             "committed chunk's own AR index instead (k-th "
                             "generated chunk -> level k+1), which is the "
                             "indexing the reverse FN was CONDITIONED ON during "
                             "training (cond = rollout2/input level, first "
                             "generated chunk = 1). Clamped by "
                             "--reverse_noiser_dedrift_level_max.")
    parser.add_argument("--reverse_noiser_dedrift_level_max", type=int, default=16,
                        help="Upper clamp for --reverse_noiser_dedrift_level_auto "
                             "(keep <= the checkpoint's max_carn_step).")
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
    parser.add_argument("--infinity_rope", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="AR mode only: install LongLive-style "
                             "Block-Relativistic RoPE (Infinity-RoPE) on the KV "
                             "cache path. Stores un-roped K in the cache and "
                             "rotates Q/K with bounded window-relative indices "
                             "at attention time, fixing the chunk-boundary "
                             "stutter caused by stale absolute rotations on long "
                             "rollouts. Default: read 'infinity_rope' from the "
                             "config (true if absent). Pass --no-infinity_rope "
                             "to force the original RoPE path for A/B testing.")

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
    # TEACHER PARITY: the ODE LMDB was built with pca_raw + dims [0,1]
    # (gen_lmdb_14e.py sets ARRWM_ACTION_ENCODER from the teacher config).
    # zarr_dataset defaults to ss_vae, so without this the eval encodes a
    # DIFFERENT physical action than the model was trained on.
    os.environ.setdefault("ARRWM_ACTION_ENCODER",
                          str(_cfg.get("teacher_action_encoder", "pca_raw")))
    action_dims = list(_cfg.get("action_dims", [0, 1]))
    # Default ``infinity_rope`` to the config value (if any) — falls back
    # to True so the rerope fix is on for fresh runs without an explicit
    # CLI flag. ``--no-infinity_rope`` overrides the config to False.
    if args.infinity_rope is None:
        infinity_rope_enabled = bool(_cfg.get("infinity_rope", True))
    else:
        infinity_rope_enabled = bool(args.infinity_rope)
    log.info(
        "[config] motion_root=%s  ss_vae_ckpt=%s  action_dims=%s  infinity_rope=%s",
        motion_root, ss_vae_ckpt, action_dims, infinity_rope_enabled,
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
    student_step = pipe.load_checkpoint(
        args.student_ckpt, use_ema=args.use_ema,
    )
    pipe.set_denoising_steps(args.denoising_steps)

    # ---- Optional auxiliary de-drift network (reverse-trained CARN) ----
    # Loaded as a STANDALONE module: this eval script has no
    # ActionForcingDMD instance, so the noiser is not attached to the
    # student in any way -- it is applied post-hoc to each committed chunk.
    reverse_noiser = None
    if getattr(args, "reverse_noiser_checkpoint", ""):
        if mode != "ar":
            # Fail loudly rather than silently ignoring the flag: the
            # de-drift only has a commit point in the AR streaming loop.
            raise SystemExit(
                "--reverse_noiser_checkpoint is AR-mode only; got "
                f"--mode {mode!r}."
            )
        from model.forward_noiser import ForwardNoiser
        rn_path = Path(args.reverse_noiser_checkpoint)
        if not rn_path.is_file():
            raise SystemExit(
                f"[AR] --reverse_noiser_checkpoint not found: {rn_path}"
            )
        rn_obj = torch.load(str(rn_path), map_location="cpu", weights_only=False)
        # Accept either a bare state_dict (what the trainer writes for
        # fn_rev_step*.pt) or a wrapper dict holding one.
        rn_sd = rn_obj
        if isinstance(rn_obj, dict):
            for key in ("forward_noiser", "reverse_noiser", "state_dict", "model"):
                if key in rn_obj and isinstance(rn_obj[key], dict):
                    rn_sd = rn_obj[key]
                    break
        if not isinstance(rn_sd, dict) or "out_proj.weight" not in rn_sd:
            raise SystemExit(
                f"[AR] {rn_path} does not look like a ForwardNoiser "
                f"state_dict (no 'out_proj.weight'); keys="
                f"{list(rn_sd)[:8] if isinstance(rn_sd, dict) else type(rn_sd)}"
            )
        reverse_noiser = ForwardNoiser(
            latent_channels=int(args.reverse_noiser_latent_channels),
            hidden_dim=int(args.reverse_noiser_hidden_dim),
            num_blocks=int(args.reverse_noiser_num_blocks),
            max_carn_step=int(args.reverse_noiser_max_carn_step),
        )
        # strict=True: any shape/name mismatch against the constructed
        # ForwardNoiser raises here, so a hidden_dim/num_blocks mismatch
        # cannot be silently absorbed. (Dtype is NOT checked by
        # load_state_dict -- a bf16 checkpoint loads into the fp32 module
        # by cast, which is what the trainer writes.)
        reverse_noiser.load_state_dict(rn_sd, strict=True)
        reverse_noiser = reverse_noiser.to(
            device=device, dtype=pipe.dtype).eval()
        reverse_noiser.requires_grad_(False)
        _w = float(sum(
            p.detach().float().pow(2).sum() for p in reverse_noiser.parameters()
        ).sqrt())
        _op = float(
            reverse_noiser.out_proj.weight.detach().float().pow(2).sum().sqrt()
        )
        log.info(
            "[AR] reverse-noiser loaded strict from %s | params=%d wnorm=%.4f "
            "out_proj_wnorm=%.6f dtype=%s device=%s",
            rn_path, reverse_noiser.num_params(), _w, _op, pipe.dtype, device,
        )
        if _op == 0.0:
            log.warning(
                "[AR] reverse-noiser out_proj is EXACTLY zero -- this is the "
                "zero-init identity; the de-drift will be a strict no-op."
            )

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

        from utils.infinity_rope import infinity_rope_active
        base_dit_ar = pipe.wrapper.model
        if hasattr(base_dit_ar, "get_base_model"):
            base_dit_ar = base_dit_ar.get_base_model()

        t0 = time.time()
        with infinity_rope_active(infinity_rope_enabled, base_dit_ar):
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
                carn_seam_affine_lambda=args.carn_seam_affine_lambda,
                reverse_noiser=reverse_noiser,
                reverse_noiser_dedrift_level=args.reverse_noiser_dedrift_level,
                reverse_noiser_dedrift_steps=args.reverse_noiser_dedrift_steps,
                reverse_noiser_dedrift_alpha0=args.reverse_noiser_dedrift_alpha0,
                reverse_noiser_dedrift_alpha_decay=(
                    args.reverse_noiser_dedrift_alpha_decay),
                reverse_noiser_dedrift_min_level=(
                    args.reverse_noiser_dedrift_min_level),
                reverse_noiser_dedrift_level_auto=(
                    args.reverse_noiser_dedrift_level_auto),
                reverse_noiser_dedrift_level_max=(
                    args.reverse_noiser_dedrift_level_max),
            )
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
        log.info(
            "[AR][%s] rank=%d streaming rollout done in %.1fs "
            "(K=%d, fill=%s, refresh=%s, infinity_rope=%s, bootstrap=%d, "
            "main=%d, seed=%d) | mem before=%.0f MiB  peak_alloc=%.0f MiB  "
            "peak_reserved=%.0f MiB",
            label, rank, time.time() - t0, args.denoising_steps,
            cache_fill_tag, args.ar_cache_refresh, infinity_rope_enabled,
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
        rerope_tag = "_rerope" if infinity_rope_enabled else ""
        tag = (
            f"{label}_{cond_tag}_ar_c{args.cache_chunks}_cps{args.chunks_per_step}"
            f"_{cache_fill_tag}{refresh_tag}{rerope_tag}"
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
