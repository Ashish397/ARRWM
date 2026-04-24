#!/usr/bin/env python3
"""Rolling-staircase inference for the ODE-distilled student.

What this script does is exactly what Phase-1 training does, with the
loss + teacher + fake-score stack removed:

  1. Build the ODE student (``ODERegression``, which also merges the
     teacher's LoRA into the DiT and builds ``action_projection`` /
     ``action_token_projection`` / ``state_probe`` / ``action_critic``).
  2. Overlay a Phase-1-compatible student snapshot (``generator`` +
     ``action_projection`` + ``action_token_projection`` keys). If the
     caller passes the 1000-step ODE init checkpoint (the default
     when ``--student_ckpt`` points at
     ``action_ode_distill_E/action_ode_step0001000.pt``), the only
     overlay is the ODE init itself and the student runs exactly as it
     would at the very first iteration of Phase-1 training.
  3. Wrap the student's generator with ``RollingStaircaseTrainingPipeline``
     using the same topology knobs as the Phase-1 config
     (``num_live_slots``, ``passes_per_step``, ``denoising_step_list``,
     ``action_decay_per_slot``, ``local_attn_size``, etc.).
  4. For one ride per rank, run ``pipeline.rollout_ride`` under
     ``torch.inference_mode()`` and capture the slot-0 committed chunk
     yielded by each rolling step.
  5. Decode the committed chunks through the Wan VAE, annotate each
     frame (ride label + chunk index + commanded action bars) in the
     same style ``eval_chain.annotate_video`` uses, and write an mp4
     per rank via ``frames_to_mp4`` (ffmpeg).

Launch model: single-GPU-per-process, one zarr per process, so you can
spread 4 GPUs across 4 distinct rides in a single sbatch. Mirrors
``utils/eval_causal_AR.py``'s layout:

  CUDA_VISIBLE_DEVICES=$GPU WORLD_SIZE=1 LOCAL_RANK=0 \\
      python utils/eval_rolling_staircase.py \\
          --config configs/longlive_phase1_rolling_staircase.yaml \\
          --student_ckpt logs/action_ode_distill_E/action_ode_step0001000.pt \\
          --rank_zarr 20240216101235.zarr \\
          --rank_offset 100 \\
          --rank_tag madrid_held \\
          --output_dir eval/eval_rolling_staircase_<ts>/gpu0_madrid

The ODE config referenced by ``--ode_config`` (defaults to
``configs/action_ode_distill.yaml``) is used ONLY to build the
``ODERegression`` shell. All rolling-staircase knobs are read from the
Phase-1 config ``--config`` (so the two live in exactly the same
representation that Phase-1 training reads).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# af_model + streaming helpers live under action-forcing/.
_AF_ROOT = _REPO_ROOT / "action-forcing"
if str(_AF_ROOT) not in sys.path:
    sys.path.insert(0, str(_AF_ROOT))

from utils.eval_chain import (
    frames_to_mp4,
)
from utils.eval_causal_AR import load_per_rank_ride_ar

# ODE-training RoPE table size. The Phase-1 stack bumped this to 10_000 in
# ``wan/modules/{causal_model,model}.py`` so rollouts longer than 1024
# frames wouldn't index out-of-bounds. For evaluation we want to match
# the *ODE student's* training-time positional encoding exactly, which
# used the upstream 1024-entry table. Both the 1024 and 10_000 tables
# agree bit-for-bit at positions [0, 1024) (same theta, same
# ``torch.arange`` formula) — so reverting is free for any rollout shorter
# than that (≈ 340 rolling steps at npb=3). We revert anyway so this
# script is a faithful reproduction of how the ODE checkpoint actually
# sees the world.
ODE_ROPE_MAX_SEQ_LEN = 1024

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Student + pipeline construction
# ---------------------------------------------------------------------------


def _load_omega(path: str):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(path)
    OmegaConf.set_struct(cfg, False)
    return cfg


def _revert_rope_to_ode_size(inner_model: Any, max_seq_len: int = ODE_ROPE_MAX_SEQ_LEN) -> None:
    """Replace the inner DiT's RoPE temporal-frequency table with the
    ODE-training-size table (``max_seq_len=1024`` by default).

    Phase-1 training bumped the temporal RoPE table to 10_000 entries in
    ``wan/modules/{causal_model,model}.py`` so long rides wouldn't index
    past the end, but ALL the ODE distillation was done against the
    original 1024-entry table. Positions [0, 1024) are bit-identical
    between the two tables (same theta=10_000, same
    ``torch.arange(max_seq_len)`` formula), so the revert is a no-op for
    any rollout that never reaches position 1024 — but it makes the
    intent explicit and the error path loud (if someone ever rolls past
    1024 frames with this script they'll see the index-out-of-range
    that the ODE student would have seen at train time).

    Operates in-place on ``inner_model``. Safe to call on either
    ``CausalWanModel`` (the student's inner DiT) or ``WanModel`` (the
    v14 teacher's inner DiT) — both expose a ``freqs`` buffer and a
    ``dim``/``num_heads`` pair.
    """
    from wan.modules.model import rope_params
    dim = int(getattr(inner_model, "dim"))
    num_heads = int(getattr(inner_model, "num_heads"))
    d = dim // num_heads
    new_freqs = torch.cat([
        rope_params(max_seq_len, d - 4 * (d // 6)),
        rope_params(max_seq_len, 2 * (d // 6)),
        rope_params(max_seq_len, 2 * (d // 6)),
    ], dim=1)
    old_freqs = getattr(inner_model, "freqs")
    inner_model.freqs = new_freqs.to(device=old_freqs.device, dtype=old_freqs.dtype)
    inner_model.rope_max_seq_len = max_seq_len
    log.info(
        "RoPE: reverted temporal table to ODE-training size (max_seq_len=%d); "
        "table now covers positions [0, %d).",
        max_seq_len, max_seq_len,
    )


def build_student_and_pipeline(
    phase1_cfg_path: str,
    ode_cfg_path: str,
    student_ckpt_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Any, Any, Any]:
    """Build and return ``(ode_model, pipeline, vae)``.

    ``ode_model`` is a live ``ODERegression`` with the student overlay
    applied. ``pipeline`` is a ``RollingStaircaseTrainingPipeline``
    wrapping the student's generator. ``vae`` is a fresh ``WanVAEWrapper``
    on ``device`` (kept separate from ``ode_model`` because
    ``ODERegression`` does not expose it directly — we build our own
    copy for the decode path).
    """
    phase1_cfg = _load_omega(phase1_cfg_path)
    ode_cfg = _load_omega(ode_cfg_path)

    # NOTE: we deliberately do NOT flip ``use_motion_pipeline`` off in
    # the config — ODERegression.__init__ raises on that path. Instead
    # we simply never call ``ensure_motion_pipeline()``; the
    # VAE/CoTracker/SS-VAE stack is only exercised when that method is
    # invoked, and it's gated by ``self._motion_pipeline_ready`` flag.
    # We're not training or computing action-teacher targets here so
    # the pipeline is unused.
    from af_model.ode_regression import ODERegression
    log.info("Building ODERegression (merges teacher LoRA into DiT)...")
    ode_model = ODERegression(ode_cfg, device=device).eval()

    # Revert RoPE to the ODE-training table (1024). See note on
    # ``ODE_ROPE_MAX_SEQ_LEN`` at the top of this file.
    _revert_rope_to_ode_size(ode_model.generator.model)

    # Overlay the student checkpoint (full-rank DiT + action heads).
    log.info("Overlaying student snapshot: %s", student_ckpt_path)
    raw = torch.load(student_ckpt_path, map_location="cpu", weights_only=False)
    if "generator" not in raw:
        raise RuntimeError(
            f"Student checkpoint {student_ckpt_path} missing 'generator' key; "
            "expected a full-rank ODE-distill snapshot (train.py writes "
            "'generator' / 'action_projection' / 'action_token_projection')."
        )
    missing, unexpected = ode_model.generator.model.load_state_dict(
        raw["generator"], strict=False,
    )
    if missing:
        log.warning("generator: %d missing keys (first 5: %s)",
                    len(missing), list(missing)[:5])
    if unexpected:
        log.warning("generator: %d unexpected keys (first 5: %s)",
                    len(unexpected), list(unexpected)[:5])
    if ode_model.action_projection is not None and "action_projection" in raw:
        ode_model.action_projection.load_state_dict(raw["action_projection"])
        log.info("Loaded action_projection.")
    if (
        ode_model.action_token_projection is not None
        and "action_token_projection" in raw
    ):
        ode_model.action_token_projection.load_state_dict(
            raw["action_token_projection"]
        )
        log.info("Loaded action_token_projection.")
    embedded_step = int(raw.get("step", -1))
    del raw
    log.info(
        "Student overlay OK (embedded step=%s).",
        embedded_step if embedded_step >= 0 else "?",
    )

    # Cast to the eval dtype. The VAE stays in fp32 (its convs have fp32
    # biases and overflow silently in bf16 — matches ``BaseModel`` behavior
    # in the main trainer).
    ode_model.generator.model.to(device=device, dtype=dtype)
    if ode_model.action_projection is not None:
        ode_model.action_projection.to(device=device, dtype=dtype)
    if ode_model.action_token_projection is not None:
        ode_model.action_token_projection.to(device=device, dtype=dtype)
    # state_probe / action_critic are NOT exercised here; skip their
    # dtype casts to keep memory down.

    # Build the rolling-staircase pipeline from the same knobs Phase-1
    # training reads. We route through the Phase-1 YAML so the two are
    # guaranteed to match.
    from pipeline.rolling_staircase_training import RollingStaircaseTrainingPipeline

    denoising_step_list = list(
        getattr(phase1_cfg, "denoising_step_list", [1000, 750, 500, 250])
    )
    num_live_slots = int(getattr(phase1_cfg, "num_live_slots", 4))
    passes_per_step = int(getattr(phase1_cfg, "passes_per_step", 1))
    default_decay = (
        [1.0, 0.75, 0.5, 0.25] if num_live_slots == 4 else [1.0, 0.5]
    )
    action_decay_slot = tuple(
        getattr(phase1_cfg, "action_decay_per_slot", default_decay)
    )

    pipeline = RollingStaircaseTrainingPipeline(
        denoising_step_list=denoising_step_list,
        scheduler=ode_model.scheduler,
        generator=ode_model.generator,
        num_frame_per_block=int(getattr(phase1_cfg, "num_frame_per_block", 3)),
        num_slots=num_live_slots,
        passes_per_step=passes_per_step,
        action_decay_per_slot=action_decay_slot,
        prime_kv_frames=int(getattr(phase1_cfg, "prime_kv_frames", 9)),
        kv_frames_total=int(getattr(phase1_cfg, "kv_frames_total", 21)),
        kv_committed_max_frames=int(
            getattr(phase1_cfg, "kv_committed_max_frames", 9)
        ),
        local_attn_size=int(getattr(phase1_cfg, "local_attn_size", 21)),
        action_projection=ode_model.action_projection,
        action_token_projection=ode_model.action_token_projection,
        real_score_num_gt_chunks=int(
            getattr(phase1_cfg, "real_score_num_gt_chunks", 2)
        ),
    )
    log.info(
        "Pipeline: num_live_slots=%d passes_per_step=%d denoising_step_list=%s "
        "action_decay_per_slot=%s kv_committed_max_frames=%d local_attn_size=%d",
        num_live_slots, passes_per_step, denoising_step_list,
        list(action_decay_slot),
        int(getattr(phase1_cfg, "kv_committed_max_frames", 9)),
        int(getattr(phase1_cfg, "local_attn_size", 21)),
    )

    # Separate VAE copy for decode (ODERegression doesn't expose one).
    from utils.wan_wrapper import WanVAEWrapper
    log.info("Loading Wan VAE (fp32) for decode...")
    vae = WanVAEWrapper()
    vae.to(device=device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    return ode_model, pipeline, vae


# ---------------------------------------------------------------------------
# v14 teacher (bidirectional WanModel + LoRA, merged) — for --debug_dmd
# ---------------------------------------------------------------------------


def _collect_wan_lora_target_modules(model: Any) -> List[str]:
    """Mirrors ``DMD2B2BLAM_Staircase._collect_target_modules``: every
    ``nn.Linear`` under a Wan attention block is a LoRA target. The v14
    teacher was trained with exactly this coverage."""
    import torch.nn as nn
    out = set()
    for module_name, module in model.named_modules():
        if module.__class__.__name__ in {
            "WanAttentionBlock", "CausalWanAttentionBlock",
        }:
            for full_name, sub in module.named_modules(prefix=module_name):
                if isinstance(sub, nn.Linear):
                    out.add(full_name)
    return sorted(out)


def build_v14_teacher(
    *,
    v14_ckpt_path: str,
    v14_lora_rank: int,
    v14_lora_alpha: float,
    v14_lora_dropout: float,
    num_context_frames: int,
    action_per_frame: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Any:
    """Build the v14 teacher scorer as a bidirectional ``WanDiffusionWrapper``
    with action patches applied and the v14 LoRA *merged* into the base
    weights (so it has no PEFT overhead at inference time).

    Mirrors the ``model/base.py`` + ``_load_real_score_with_v14_lora``
    recipe from the Phase-1 trainer, but standalone: we don't need the
    BaseModel machinery or the fake_score — just the teacher DiT.

    Args:
      v14_ckpt_path:   Path to ``causal_lora_step0006600.pt`` (or any
                       v14 snapshot with a ``"lora"`` key).
      v14_lora_rank:   Must match v14's training config (default 256).
      v14_lora_alpha:  Default = rank.
      v14_lora_dropout: Default 0.0.
      num_context_frames: Number of frames the teacher will process in
                       a single forward. Sets ``seq_len`` upper bound.
                       For debug_dmd we use (4 GT + 1 prev + 1 live)*npb
                       = 18 frames by default.
      action_per_frame: Stream-B tokens per frame. The v14 teacher was
                       trained with 1.
      device/dtype:    Final placement.

    Returns:
      A frozen, eval-mode ``WanDiffusionWrapper`` with its inner DiT
      holding v14 LoRA merged into base weights and its RoPE table
      reverted to the ODE-training 1024 entries (see
      ``_revert_rope_to_ode_size``).
    """
    from utils.wan_wrapper import WanDiffusionWrapper
    from model.action_model_patch import apply_action_patches_critic
    import peft
    from peft import LoraConfig, set_peft_model_state_dict

    log.info("Building v14 teacher (bidirectional WanModel)...")
    teacher = WanDiffusionWrapper(is_causal=False)
    # Resize seq_len to the exact context window this teacher will be
    # called with. The bidirectional block's ``unflatten(1, (num_frames,
    # frame_seqlen))`` inside ``WanModel._forward`` requires seq_len to
    # match input tokens exactly, not just bound them from above — so we
    # MUST size this to the actual debug_dmd window (6 chunks * npb).
    teacher._base_seq_len = num_context_frames * 1560
    teacher.seq_len = teacher._base_seq_len

    # Apply bidirectional action patches (Stream A + Stream B), then set
    # the per-frame action token budget and regrow seq_len to account
    # for the extra tokens.
    apply_action_patches_critic(teacher)
    teacher.model.action_tokens_per_frame = int(action_per_frame)
    teacher.adjust_seq_len_for_action_tokens(
        num_frames=num_context_frames, action_per_frame=int(action_per_frame),
    )
    log.info(
        "v14 teacher: num_context_frames=%d action_per_frame=%d seq_len=%d",
        num_context_frames, action_per_frame, teacher.seq_len,
    )

    # Wrap in PEFT with v14's LoRA config, load LoRA weights, merge +
    # unload. Matches ``_load_real_score_with_v14_lora`` exactly.
    target_modules = _collect_wan_lora_target_modules(teacher.model) or [
        "q", "k", "v", "o",
    ]
    log.info(
        "v14 teacher: applying LoRA rank=%d alpha=%s drop=%s (%d target modules)",
        v14_lora_rank, v14_lora_alpha, v14_lora_dropout, len(target_modules),
    )
    lora_cfg = LoraConfig(
        r=int(v14_lora_rank),
        lora_alpha=float(v14_lora_alpha),
        lora_dropout=float(v14_lora_dropout),
        target_modules=target_modules,
        bias="none",
    )
    peft_model = peft.get_peft_model(teacher.model, lora_cfg)
    ckpt = torch.load(v14_ckpt_path, map_location="cpu", weights_only=False)
    if "lora" not in ckpt:
        raise KeyError(
            f"v14 checkpoint {v14_ckpt_path} missing 'lora' key; "
            f"have {list(ckpt.keys())[:10]}..."
        )
    try:
        set_peft_model_state_dict(peft_model, ckpt["lora"])
    except Exception as e:  # noqa: BLE001
        from peft import get_peft_model_state_dict
        log.warning(
            "v14 LoRA strict load failed (%s); cross-loading matched keys.", e,
        )
        current_sd = get_peft_model_state_dict(peft_model)
        matched = 0
        for k in current_sd:
            if k in ckpt["lora"] and current_sd[k].shape == ckpt["lora"][k].shape:
                current_sd[k] = ckpt["lora"][k]
                matched += 1
        set_peft_model_state_dict(peft_model, current_sd)
        log.info("v14 LoRA cross-load matched %d/%d", matched, len(current_sd))
    del ckpt

    merged = peft_model.merge_and_unload()
    # Sanity check: no residual LoraLayer.
    try:
        from peft.tuners.lora import LoraLayer
        for _, m in merged.named_modules():
            if isinstance(m, LoraLayer):
                raise RuntimeError(
                    "LoRA layers still present on v14 teacher after "
                    "merge_and_unload(); refusing to run."
                )
    except ImportError:
        pass
    teacher.model = merged.to(device=device, dtype=dtype)

    # Match ODE-training RoPE.
    _revert_rope_to_ode_size(teacher.model)

    teacher.to(device=device, dtype=dtype)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    log.info("v14 teacher ready: dtype=%s device=%s", dtype, device)
    return teacher


# ---------------------------------------------------------------------------
# Rollout + capture
# ---------------------------------------------------------------------------


def run_rollout_and_capture(
    *,
    pipeline: Any,
    gt_latents: torch.Tensor,     # [1, T, C, H, W]
    gt_actions: torch.Tensor,     # [1, T, action_dim]
    prompt_embeds: torch.Tensor,  # [1, L, C_txt]
    num_frame_per_block: int,
    prime_kv_frames: int,
    max_rolling_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the rolling-staircase rollout in inference mode and collect
    the slot-0 committed latents for visualization.

    Returns a dict with:
      - ``prime_latents``:    [1, prime_kv_frames, C, H, W] — the clean
                              GT latents pushed into the KV cache by
                              ``warmup_prime_kv_cache`` (same frames the
                              student's "past" sees at the start of the
                              rollout). Drawn at the front of the mp4
                              with a ``prime`` label.
      - ``commits``:          list of dict entries, each containing:
                                * ``latent``: [1, npb, C, H, W] fp32 CPU
                                * ``action``: [action_dim] fp32 CPU
                                  (mean-pooled over the 3 frames)
                                * ``chunk_index``: int
                                * ``global_frame_start``: int
                                * ``slot_timestep``: int
                                * ``phase``: str ("transition"/"steady")
                                * ``rolling_steps_done``: int
      - ``stats``: dict of per-step wall-clock and per-ride counters.
    """
    device = gt_latents.device
    dtype = gt_latents.dtype
    npb = int(num_frame_per_block)

    # Seed the prime view from the front of the ride — same slice the
    # pipeline's ``warmup_prime_kv_cache`` will commit internally. We
    # snapshot it HERE so the mp4 can show a ``prime`` header before the
    # generated commits even though the pipeline doesn't yield records
    # for the prime forward.
    prime = gt_latents[:, :prime_kv_frames].detach().clone().to(
        device="cpu", dtype=torch.float32,
    )

    commits: List[Dict[str, Any]] = []
    transition_captured = False
    t0 = time.time()
    last_log = t0

    with torch.inference_mode():
        it = pipeline.rollout_ride(
            gt_latents=gt_latents,
            gt_actions=gt_actions,
            prompt_embeds=prompt_embeds,
            max_rolling_steps=max_rolling_steps,
        )
        for step_idx, step_records in enumerate(it):
            if not step_records:
                continue
            # First yielded step exposes the transition-commit anchor
            # (kv_anchor_chunk) on its first record — this is the clean
            # slot-0 output produced by the warmup+transition path,
            # committed into the cache BEFORE the first steady-state
            # rolling step. Push it once so it appears between prime
            # and the first steady commit.
            if not transition_captured:
                first = step_records[0]
                kv_anchor = getattr(first, "kv_anchor_chunk", None)
                if torch.is_tensor(kv_anchor):
                    commits.append({
                        "latent": kv_anchor.detach().to(
                            device="cpu", dtype=torch.float32,
                        ).clone(),
                        "action": None,
                        "chunk_index": -1,
                        "global_frame_start": -1,
                        "slot_timestep": 0,
                        "phase": "transition",
                        "rolling_steps_done": 0,
                    })
                transition_captured = True

            # Find the slot-0 record for this step (one per rolling step
            # regardless of how many slots the pipeline was configured
            # with — slot 0 is always the cleanest and is always the
            # chunk being committed).
            slot0 = None
            for r in step_records:
                if int(getattr(r, "slot_idx", -1)) == 0:
                    slot0 = r
                    break
            if slot0 is None:
                continue

            pred = slot0.pred_x0.detach().to(
                device="cpu", dtype=torch.float32,
            ).clone()
            action_frame = getattr(slot0, "action_frame", None)
            if torch.is_tensor(action_frame) and action_frame.numel() > 0:
                action_vec = (
                    action_frame[0].detach().float().mean(dim=0).cpu().clone()
                )
            else:
                action_vec = None

            gfs = int(getattr(slot0, "global_frame_start", -1))
            commits.append({
                "latent": pred,
                "action": action_vec,
                "chunk_index": gfs // npb if gfs >= 0 else -1,
                "global_frame_start": gfs,
                "slot_timestep": int(getattr(slot0, "slot_timestep", -1)),
                "phase": str(getattr(slot0, "phase", "steady")),
                "rolling_steps_done": step_idx + 1,
            })

            # Lightweight progress log every 15 s.
            now = time.time()
            if now - last_log >= 15.0:
                log.info(
                    "rollout: %d rolling steps captured | wall=%.1fs | last chunk=%d",
                    len(commits), now - t0, commits[-1]["chunk_index"],
                )
                last_log = now

    t1 = time.time()
    stats = {
        "n_commits": len(commits),
        "wall_seconds": t1 - t0,
        "rolling_steps_per_sec": (
            (len(commits) - (1 if transition_captured else 0))
            / max(t1 - t0, 1e-6)
        ),
    }
    log.info(
        "rollout done: %d commits (transition=%s) | %.1fs | %.3f steps/s",
        len(commits), transition_captured, stats["wall_seconds"],
        stats["rolling_steps_per_sec"],
    )
    return {
        "prime_latents": prime,
        "commits": commits,
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# debug_dmd: 4 side-by-side streams for DMD-teacher sanity-checking
# ---------------------------------------------------------------------------


def _noise_block(
    scheduler: Any,
    clean: torch.Tensor,       # [B, F, C, H, W]
    timestep: int,
) -> torch.Tensor:
    """Return ``scheduler.add_noise(clean, randn, t)`` broadcast to ``clean``'s
    shape. Convention: ``t=0`` is clean, ``t=1000`` is pure noise — matches
    the flow-match scheduler used throughout the repo.

    We sample a fresh ``randn`` each call because the debug script is
    meant to show what the student / teacher do under a typical noise
    realisation; seeding is handled by the caller via ``torch.manual_seed``.
    """
    if int(timestep) <= 0:
        return clean
    B, F = clean.shape[:2]
    device = clean.device
    dtype = clean.dtype
    noise = torch.randn_like(clean)
    t = torch.full((B, F), int(timestep), device=device, dtype=torch.long)
    return scheduler.add_noise(
        clean.flatten(0, 1), noise.flatten(0, 1), t.flatten(0, 1),
    ).unflatten(0, (B, F)).to(dtype=dtype)


def _student_single_step(
    *,
    pipeline: Any,
    noisy_block: torch.Tensor,      # [B, npb, C, H, W] at timestep t_in
    action_block: torch.Tensor,     # [B, npb, action_dim]
    prompt_embeds: torch.Tensor,
    timestep_in: int,
    current_start_frame: int,
) -> torch.Tensor:
    """Run the student's CausalWanModel on a single ``npb``-frame chunk
    that has been noised to ``timestep_in``, with the pipeline's KV cache
    as left-context. Returns ``pred_x0`` ([B, npb, C, H, W]).

    IMPORTANT: this is the "before-renoising" output the user asked
    for — the raw x_0 prediction the student emits at this rolling step,
    before the next step re-adds noise on top of it. We pass
    ``skip_cache_update=True`` so the noisy-side keys do NOT leak into
    the persistent KV cache; the caller commits ``pred_x0`` (via a
    separate clean-side t=0 forward) to advance the cache.
    """
    device = noisy_block.device
    dtype = noisy_block.dtype
    B, npb = noisy_block.shape[:2]

    modulation = pipeline._compute_live_modulation(action_block, device=device, dtype=dtype)
    action_tokens = pipeline._compute_live_action_tokens(action_block, device=device, dtype=dtype)
    cond = pipeline._prepare_conditional(prompt_embeds, modulation, action_tokens)

    t_in = torch.full((B, npb), int(timestep_in), device=device, dtype=torch.int64)

    pipeline._set_skip_cache_update(True)
    with torch.inference_mode():
        out = pipeline.generator(
            noisy_image_or_video=noisy_block,
            conditional_dict=cond,
            timestep=t_in,
            kv_cache=pipeline.kv_cache1,
            crossattn_cache=pipeline.crossattn_cache,
            current_start=current_start_frame * pipeline.frame_seq_length,
        )
    return out[1]  # pred_x0


def _student_commit_clean(
    *,
    pipeline: Any,
    clean_block: torch.Tensor,      # [B, npb, C, H, W] — pred_x0 or GT
    action_block: torch.Tensor,
    prompt_embeds: torch.Tensor,
    current_start_frame: int,
) -> None:
    """Run a t=0 forward on ``clean_block`` with ``skip_cache_update=False``
    to write its KV into the persistent cache. This advances the student's
    left-context for the next rolling step. Mirrors the commit half of
    ``RollingStaircaseTrainingPipeline.transition_to_steady_state`` +
    ``rolling_step``.
    """
    device = clean_block.device
    dtype = clean_block.dtype
    B, npb = clean_block.shape[:2]

    modulation = pipeline._compute_live_modulation(action_block, device=device, dtype=dtype)
    action_tokens = pipeline._compute_live_action_tokens(action_block, device=device, dtype=dtype)
    cond = pipeline._prepare_conditional(prompt_embeds, modulation, action_tokens)
    t_zero = torch.zeros((B, npb), device=device, dtype=torch.int64)

    pipeline._set_skip_cache_update(False)
    with torch.inference_mode():
        pipeline.generator(
            noisy_image_or_video=clean_block,
            conditional_dict=cond,
            timestep=t_zero,
            kv_cache=pipeline.kv_cache1,
            crossattn_cache=pipeline.crossattn_cache,
            current_start=current_start_frame * pipeline.frame_seq_length,
        )


def _teacher_score(
    *,
    teacher: Any,
    action_projection: Any,
    action_token_projection: Any,
    ctx_clean_chunks: List[torch.Tensor],    # list of [B, npb, C, H, W] CLEAN
    ctx_clean_actions: List[torch.Tensor],    # list of [B, npb, action_dim]
    noisy_target_chunk: torch.Tensor,         # [B, npb, C, H, W] at target_t
    target_action: torch.Tensor,              # [B, npb, action_dim]
    target_timestep: int,
    prompt_embeds: torch.Tensor,
) -> torch.Tensor:
    """Single forward of the v14 bidirectional teacher.

    Builds the same input layout the DMD staircase's real_score sees —
    ``[ctx_chunks..., noisy_target_chunk]`` concatenated along the frame
    dim, with timesteps ``[0...0, target_t...target_t]`` — so the
    teacher runs exactly as it does inside DMD training (modulo the
    fact that here we're using it as a single-step denoiser rather
    than as a score function). Returns ``pred_x0`` for ALL frames in
    the input window; the caller slices off the target chunk.

    Conditioning is built from the *student's* action_projection and
    action_token_projection — this mirrors DMD's implementation, where
    both scorers share the generator's action heads so they all see
    the same Stream-A/Stream-B encoding of the commanded actions.
    """
    device = noisy_target_chunk.device
    dtype = noisy_target_chunk.dtype
    B = noisy_target_chunk.shape[0]
    npb = noisy_target_chunk.shape[1]
    n_ctx = len(ctx_clean_chunks)
    assert n_ctx >= 1, "teacher needs at least one context chunk"
    assert len(ctx_clean_actions) == n_ctx

    ctx_frames = torch.cat(ctx_clean_chunks, dim=1).contiguous()  # [B, n_ctx*npb, C, H, W]
    ctx_actions = torch.cat(ctx_clean_actions, dim=1).contiguous()  # [B, n_ctx*npb, A]

    full_input = torch.cat([ctx_frames, noisy_target_chunk], dim=1).contiguous()
    full_actions = torch.cat([ctx_actions, target_action], dim=1).contiguous()

    # Timesteps: 0 for all clean context, target_t for the last chunk.
    n_ctx_f = ctx_frames.shape[1]
    t_ctx = torch.zeros((B, n_ctx_f), device=device, dtype=torch.long)
    t_tgt = torch.full((B, npb), int(target_timestep), device=device, dtype=torch.long)
    full_t = torch.cat([t_ctx, t_tgt], dim=1).contiguous()

    modulation = action_projection(full_actions, num_frames=full_actions.shape[1])
    action_tokens = action_token_projection(full_actions)
    cond = {
        "prompt_embeds": prompt_embeds,
        "_action_modulation": modulation.to(dtype=dtype),
        "_action_tokens": action_tokens.to(dtype=dtype),
    }

    with torch.inference_mode():
        out = teacher(
            noisy_image_or_video=full_input,
            conditional_dict=cond,
            timestep=full_t,
        )
    pred_x0 = out[1]  # [B, n_ctx_f + npb, C, H, W]
    return pred_x0[:, -npb:]  # just the target chunk


def run_debug_dmd(
    *,
    pipeline: Any,
    ode_model: Any,
    teacher: Any,
    gt_latents: torch.Tensor,       # [B, T, C, H, W]
    gt_actions: torch.Tensor,       # [B, T, action_dim]
    prompt_embeds: torch.Tensor,
    num_blocks: int = 10,
    student_t_hi: int = 750,
    student_t_lo: int = 250,
    teacher_t: int = 750,
    cleanup_t: int = 750,
    teacher_num_gt_chunks: int = 4,
) -> Dict[str, Any]:
    """Produce 4 side-by-side commit streams for a single ride. Runs
    SYNCHRONOUSLY in one rolling loop, so all four modes share the exact
    same base GT ride content, the same seed, and the same KV-cache
    rollout history — the only thing that changes is what's visualized.

    Modes:

      * student_hi (default 750→0):
          Each block, the clean GT chunk is noised to t_hi, the student
          runs a single denoise forward (with its KV cache + previous
          student commits as left-context), and we record ``pred_x0``
          BEFORE any renoising. The student's pred_x0 is also committed
          into the KV cache so the next block's left-context is
          student-rolled (matches what Phase-1 would see in steady state).

      * student_lo (default 250→0):
          Same as student_hi but at a lower noise floor. Uses a FRESH
          KV cache (doesn't share state with student_hi) so the two
          student streams are independent single-step evaluators at
          different input noise levels.

      * teacher_raw (default 750→0):
          For each block the v14 teacher sees
              [ GT_{b-k}, ..., GT_{b-1}, student_hi.commit_{b-1}, noisy_GT_b@t_hi ]
          as input (``teacher_num_gt_chunks`` GT chunks + 1 student
          commit + 1 noisy live). Records the teacher's ``pred_x0`` for
          the live (last) chunk. This is what DMD's real_score sees.

      * teacher_cleaned (default 750→0, cleanup_t=750):
          Same as teacher_raw but the student commit is first "cleaned
          up" by a separate teacher no-grad forward of
              [ GT_{b-k}, ..., GT_{b-1}, noisy(student_hi.commit_{b-1}, cleanup_t) ]
          i.e. teacher re-denoises the student's commit chunk given the
          GT history. That cleaned chunk then replaces the raw student
          commit in the main teacher forward. This lets us see whether
          the raw student commit is degrading what the teacher predicts
          (if cleaned >> raw, the student's rolled context is the
          culprit; if cleaned ≈ raw, the teacher is fine with the
          student's context).

    Args:
      num_blocks: How many blocks to roll. Default 10 as per request.
      student_t_hi / student_t_lo: Student input noise levels (clean=0).
      teacher_t: Teacher's target-chunk noise level.
      cleanup_t: The teacher-cleanup no-grad forward's target-chunk noise
        level. Defaults to ``teacher_t`` so the teacher is run at the
        same operating point both times.
      teacher_num_gt_chunks: Number of GT context chunks fed to the
        teacher per forward. Matches Phase-1's
        ``real_score_num_gt_chunks``; default 4.

    Returns:
      A dict keyed by mode name with a list of per-block commit entries,
      plus the prime-cache latents (same for every mode — GT primed).
    """
    device = gt_latents.device
    dtype = gt_latents.dtype
    npb = pipeline.num_frame_per_block
    B = gt_latents.shape[0]
    scheduler = pipeline.scheduler
    prime_frames = pipeline.prime_kv_frames

    if gt_latents.shape[1] < prime_frames + num_blocks * npb:
        raise SystemExit(
            f"debug_dmd: ride has {gt_latents.shape[1]} latents, "
            f"need at least {prime_frames} + {num_blocks}*{npb} = "
            f"{prime_frames + num_blocks * npb}."
        )
    if teacher_num_gt_chunks < 1:
        raise SystemExit(
            f"teacher_num_gt_chunks must be >= 1; got {teacher_num_gt_chunks}."
        )

    action_proj = ode_model.action_projection
    action_tok_proj = ode_model.action_token_projection
    assert action_proj is not None and action_tok_proj is not None

    # ------------------------------------------------------------------
    # Prime the primary (student_hi) KV cache with 9 clean GT frames.
    # ------------------------------------------------------------------
    pipeline._initialize_kv_cache(batch_size=B, dtype=dtype, device=device)
    pipeline._initialize_crossattn_cache(batch_size=B, dtype=dtype, device=device)
    n_primed = pipeline.warmup_prime_kv_cache(
        gt_latents=gt_latents[:, :prime_frames],
        gt_actions=gt_actions[:, :prime_frames],
        prompt_embeds=prompt_embeds,
    )
    assert n_primed == prime_frames, (n_primed, prime_frames)
    prime_latents_snapshot = gt_latents[:, :n_primed].detach().to(
        device="cpu", dtype=torch.float32,
    ).clone()

    # Track student_hi's own cache state.
    current_start_hi = n_primed

    # ------------------------------------------------------------------
    # Build an INDEPENDENT second KV cache for student_lo. We snapshot
    # the primed cache's contents so both student streams start from
    # the same "3 GT chunks primed" state — then diverge as each one
    # commits its own student output.
    # ------------------------------------------------------------------
    # Easiest way to duplicate: re-prime from scratch on a fresh cache
    # each time we want to use student_lo. That's cheap (one forward
    # per primed block, no grad).
    # To avoid shuffling the pipeline's cache state twice per block we
    # use a simpler approach: keep only the student_hi cache live, and
    # for student_lo we call the student WITHOUT advancing the cache,
    # using the cache built up by student_hi as left-context. That's
    # slightly different semantics (student_lo sees student_hi's
    # previous commits rather than its own), but it's actually CLOSER
    # to what Phase-1 DMD sees at runtime — every slot's left context
    # is whatever the rolling pipeline committed, not the slot's own
    # history. The user's comparison ("student at 750 vs student at
    # 250") is meaningful under this reading because it isolates the
    # effect of the input-noise level on the SAME left-context.

    records: Dict[str, List[Dict[str, Any]]] = {
        "student_hi": [],
        "student_lo": [],
        "teacher_raw": [],
        "teacher_cleaned": [],
    }

    t_rollout_start = time.time()
    t_last_log = t_rollout_start
    for b in range(num_blocks):
        f0 = n_primed + b * npb
        f1 = f0 + npb
        clean_block = gt_latents[:, f0:f1].contiguous()
        action_block = gt_actions[:, f0:f1].contiguous()

        # --- Student_hi: noise + forward + record + commit --------------
        noisy_hi = _noise_block(scheduler, clean_block, student_t_hi)
        pred_hi = _student_single_step(
            pipeline=pipeline,
            noisy_block=noisy_hi,
            action_block=action_block,
            prompt_embeds=prompt_embeds,
            timestep_in=student_t_hi,
            current_start_frame=current_start_hi,
        )
        records["student_hi"].append({
            "latent": pred_hi.detach().to(device="cpu", dtype=torch.float32).clone(),
            "action": action_block[0].detach().float().mean(dim=0).cpu().clone(),
            "chunk_index": f0 // npb,
            "global_frame_start": f0,
            "slot_timestep": int(student_t_hi),
            "phase": "debug_student_hi",
            "rolling_steps_done": b + 1,
        })

        # --- Student_lo: noise + forward + record (NO commit) -----------
        noisy_lo = _noise_block(scheduler, clean_block, student_t_lo)
        pred_lo = _student_single_step(
            pipeline=pipeline,
            noisy_block=noisy_lo,
            action_block=action_block,
            prompt_embeds=prompt_embeds,
            timestep_in=student_t_lo,
            current_start_frame=current_start_hi,  # same left-context as hi
        )
        records["student_lo"].append({
            "latent": pred_lo.detach().to(device="cpu", dtype=torch.float32).clone(),
            "action": action_block[0].detach().float().mean(dim=0).cpu().clone(),
            "chunk_index": f0 // npb,
            "global_frame_start": f0,
            "slot_timestep": int(student_t_lo),
            "phase": "debug_student_lo",
            "rolling_steps_done": b + 1,
        })

        # --- Teacher context assembly ----------------------------------
        # "Previous 4 GT blocks" = the 4 GT chunks ending one chunk before
        # the current block. Indices [b + prime_chunks - 4, ..., b +
        # prime_chunks - 1) (in ride-chunk space). Clamp to >=0.
        prime_chunks = n_primed // npb
        gt_ctx_chunks: List[torch.Tensor] = []
        gt_ctx_actions: List[torch.Tensor] = []
        for k in range(teacher_num_gt_chunks, 0, -1):
            c_idx = (prime_chunks + b) - k  # chunk index in ride
            if c_idx < 0:
                # Not enough history — pad with the first available chunk.
                c_idx = 0
            cf0 = c_idx * npb
            cf1 = cf0 + npb
            gt_ctx_chunks.append(gt_latents[:, cf0:cf1].contiguous())
            gt_ctx_actions.append(gt_actions[:, cf0:cf1].contiguous())

        # "Last student generated block" = previous student_hi commit. For
        # b == 0 we don't have a student commit yet → use the most recent
        # GT chunk as a stand-in (matches what Phase-1's transition
        # step does: kv_anchor starts as a GT-primed chunk).
        # Always use the RIDE's per-frame actions for the prev chunk's
        # conditioning — those are what the student *was commanded with*
        # when it produced the commit, so Stream A/B see the same actions
        # they'd see inside Phase-1's real_score context assembly.
        if b == 0:
            prev_f0 = (prime_chunks - 1) * npb
            prev_f1 = prime_chunks * npb
            prev_chunk = gt_latents[:, prev_f0:prev_f1].contiguous()
        else:
            prev_entry = records["student_hi"][-2]
            prev_chunk = prev_entry["latent"].to(device=device, dtype=dtype)
            # Map back to ride-chunk coords: student_hi commit for block
            # (b-1) was generated from clean GT at frames
            # [n_primed + (b-1)*npb, n_primed + b*npb).
            prev_f0 = n_primed + (b - 1) * npb
            prev_f1 = prev_f0 + npb
        prev_action = gt_actions[:, prev_f0:prev_f1].contiguous()

        # Teacher target noise.
        noisy_tgt = _noise_block(scheduler, clean_block, teacher_t)

        # --- Teacher_raw: 4 GT + raw prev student + noisy live -----------
        pred_t_raw = _teacher_score(
            teacher=teacher,
            action_projection=action_proj,
            action_token_projection=action_tok_proj,
            ctx_clean_chunks=gt_ctx_chunks + [prev_chunk],
            ctx_clean_actions=gt_ctx_actions + [prev_action],
            noisy_target_chunk=noisy_tgt,
            target_action=action_block,
            target_timestep=teacher_t,
            prompt_embeds=prompt_embeds,
        )
        records["teacher_raw"].append({
            "latent": pred_t_raw.detach().to(device="cpu", dtype=torch.float32).clone(),
            "action": action_block[0].detach().float().mean(dim=0).cpu().clone(),
            "chunk_index": f0 // npb,
            "global_frame_start": f0,
            "slot_timestep": int(teacher_t),
            "phase": "debug_teacher_raw",
            "rolling_steps_done": b + 1,
        })

        # --- Teacher_cleaned: pre-denoise prev student, then score -------
        # Step 1: teacher denoises prev_chunk at cleanup_t given 4 GT +
        # no extra context — exactly the "cleanup pass" the user asked
        # for ("run it through the teacher model once to denoise it
        # with a no grad").
        cleanup_noisy = _noise_block(scheduler, prev_chunk, cleanup_t)
        prev_cleaned = _teacher_score(
            teacher=teacher,
            action_projection=action_proj,
            action_token_projection=action_tok_proj,
            ctx_clean_chunks=gt_ctx_chunks,   # 4 GT only
            ctx_clean_actions=gt_ctx_actions,
            noisy_target_chunk=cleanup_noisy,
            target_action=prev_action,
            target_timestep=cleanup_t,
            prompt_embeds=prompt_embeds,
        )
        # Step 2: main teacher forward with cleaned prev in place.
        pred_t_clean = _teacher_score(
            teacher=teacher,
            action_projection=action_proj,
            action_token_projection=action_tok_proj,
            ctx_clean_chunks=gt_ctx_chunks + [prev_cleaned],
            ctx_clean_actions=gt_ctx_actions + [prev_action],
            noisy_target_chunk=noisy_tgt,
            target_action=action_block,
            target_timestep=teacher_t,
            prompt_embeds=prompt_embeds,
        )
        records["teacher_cleaned"].append({
            "latent": pred_t_clean.detach().to(device="cpu", dtype=torch.float32).clone(),
            "action": action_block[0].detach().float().mean(dim=0).cpu().clone(),
            "chunk_index": f0 // npb,
            "global_frame_start": f0,
            "slot_timestep": int(teacher_t),
            "phase": "debug_teacher_cleaned",
            "rolling_steps_done": b + 1,
        })

        # --- Commit student_hi's pred_x0 into the KV cache to roll -----
        _student_commit_clean(
            pipeline=pipeline,
            clean_block=pred_hi,
            action_block=action_block,
            prompt_embeds=prompt_embeds,
            current_start_frame=current_start_hi,
        )
        current_start_hi += npb

        now = time.time()
        if now - t_last_log >= 10.0:
            log.info(
                "debug_dmd: block %d/%d | wall=%.1fs",
                b + 1, num_blocks, now - t_rollout_start,
            )
            t_last_log = now

    t_total = time.time() - t_rollout_start
    log.info(
        "debug_dmd done: %d blocks × 4 streams | %.1fs total (%.2fs / block)",
        num_blocks, t_total, t_total / max(num_blocks, 1),
    )
    return {
        "prime_latents": prime_latents_snapshot,
        "records": records,
        "stats": {
            "num_blocks": num_blocks,
            "wall_seconds": t_total,
            "student_t_hi": int(student_t_hi),
            "student_t_lo": int(student_t_lo),
            "teacher_t": int(teacher_t),
            "cleanup_t": int(cleanup_t),
            "teacher_num_gt_chunks": int(teacher_num_gt_chunks),
        },
    }


# ---------------------------------------------------------------------------
# Decode + annotate + write mp4
# ---------------------------------------------------------------------------


def _pixels_to_uint8_hwc(pixels: torch.Tensor) -> np.ndarray:
    """Wan VAE output → ``[T, H, W, 3] uint8`` in [0, 255]. Same shape
    coercion the Phase-1 vis uses; see ``utils.multislot_vis``."""
    if pixels.dim() == 5:
        pixels = pixels[0]
    if pixels.dim() != 4:
        raise RuntimeError(f"unexpected VAE output shape: {pixels.shape}")
    if pixels.shape[0] == 3 and pixels.shape[1] != 3:
        pixels = pixels.permute(1, 0, 2, 3)
    pixels = pixels.permute(0, 2, 3, 1).contiguous()
    if pixels.min().item() < -0.1:
        arr = ((pixels.clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
    else:
        arr = (pixels.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    arr_np = arr.numpy()
    if arr_np.shape[-1] == 1:
        arr_np = np.repeat(arr_np, 3, axis=-1)
    return arr_np


def _decode_latents(vae: Any, latents: torch.Tensor) -> torch.Tensor:
    """Decode ``[1, T, C, H, W]`` → ``[1, T_pix, 3, H_px, W_px]`` fp32.

    Uses the v14 / eval_chain convention: prepend one dummy latent frame
    so temporal upsampling emits a contiguous clip, then drop the first
    decoded frame so pixels line up 1:1 with input latents. VAE runs
    in fp32 with autocast disabled — the Wan VAE's convs have fp32
    biases and overflow silently in bf16."""
    device = next(vae.parameters()).device if hasattr(vae, "parameters") else latents.device
    lat = latents.to(device=device, dtype=torch.float32)
    dummy = lat[:, 0:1]
    with torch.no_grad():
        if lat.is_cuda:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                px = vae.decode_to_pixel(torch.cat([dummy, lat], dim=1))
        else:
            px = vae.decode_to_pixel(torch.cat([dummy, lat], dim=1))
    if px.dim() == 5 and px.shape[1] > 1:
        px = px[:, 1:, ...]
    return px.detach().float().cpu()


def _annotate_and_write_mp4(
    *,
    prime_latents: torch.Tensor,          # [1, prime_frames, C, H, W] fp32 CPU
    commits: List[Dict[str, Any]],
    vae: Any,
    device: torch.device,
    fps: int,
    out_path: Path,
    ride_tag: str,
    ride_basename: str,
    num_frame_per_block: int,
    embedded_step: int = -1,
) -> Dict[str, Any]:
    """Decode prime + commits to pixels, annotate each frame, write mp4.

    Returns a stats dict with timings + frame counts.
    """
    import cv2
    t_start = time.time()

    # ------------------------------------------------------------------
    # 1) Concatenate latents in display order:
    #      prime (9 frames) + transition commit (if present) + commits
    # ------------------------------------------------------------------
    pieces: List[torch.Tensor] = []
    piece_labels: List[Dict[str, Any]] = []

    npb = int(num_frame_per_block)
    # Prime chunks: we show the prime_latents as ``prime_frames // npb``
    # "prime" chunks with chunk indices 0..n-1 so the mp4's chronology
    # is clean.
    prime_frames = int(prime_latents.shape[1])
    prime_chunks = prime_frames // npb
    for c in range(prime_chunks):
        lo = c * npb
        hi = lo + npb
        pieces.append(prime_latents[:, lo:hi])
        piece_labels.append({
            "label_top": f"{ride_basename} | chunk {c} (prime)",
            "label_bot": "t=0 | GT-primed",
            "action": None,
        })

    for e in commits:
        pieces.append(e["latent"])
        if e["phase"] == "transition":
            top = f"{ride_basename} | chunk {prime_chunks} (xition)"
            bot = "transition commit | t=0"
        else:
            gfs = int(e["global_frame_start"])
            chunk_idx = gfs // npb if gfs >= 0 else -1
            top = f"{ride_basename} | chunk {chunk_idx}"
            bot = (
                f"step {e['rolling_steps_done']} | "
                f"t_in={e['slot_timestep']} | steady"
            )
        piece_labels.append({
            "label_top": top,
            "label_bot": bot,
            "action": e.get("action"),
        })

    lat_all = torch.cat(pieces, dim=1)
    log.info("decode: %d total latents (%d prime + %d commits) on %s",
             int(lat_all.shape[1]), prime_frames, len(commits), device)

    # ------------------------------------------------------------------
    # 2) Decode through VAE (fp32).
    # ------------------------------------------------------------------
    t_before_decode = time.time()
    # Chunk the decode to keep peak memory bounded on longer rollouts.
    # We decode up to ``max_decode_frames`` latent frames at a time,
    # re-apply the ``prepend-dummy-drop-first-pixel`` convention per
    # chunk, and concatenate pixel tensors. Longer rollouts (200+
    # commits = 600+ latent frames = ~1 min video @8fps) would otherwise
    # exceed the VAE's peak memory envelope on an 80 GB H100.
    max_decode_frames = 180  # ~8 GB peak at 16x60x104 latents.
    if lat_all.shape[1] <= max_decode_frames:
        pixels = _decode_latents(vae, lat_all.to(device=device))
    else:
        pieces_px: List[torch.Tensor] = []
        n = int(lat_all.shape[1])
        for lo in range(0, n, max_decode_frames):
            hi = min(lo + max_decode_frames, n)
            chunk = lat_all[:, lo:hi].to(device=device)
            pieces_px.append(_decode_latents(vae, chunk))
        pixels = torch.cat(pieces_px, dim=1)
    t_after_decode = time.time()

    arr = _pixels_to_uint8_hwc(pixels)
    t_after_cast = time.time()

    # ------------------------------------------------------------------
    # 3) Annotate (cv2 — v14 style, see ``eval_chain.annotate_video``).
    # ------------------------------------------------------------------
    n_pixel_frames = int(arr.shape[0])
    # Distribute labels across pixel frames proportionally to latent span.
    # Each ``pieces[i]`` has ``lat_pieces_frames[i]`` latent frames; after
    # VAE upsampling the same proportion maps to pixel frames.
    lat_pieces_frames = [int(p.shape[1]) for p in pieces]
    total_lat = sum(lat_pieces_frames)
    # Compute pixel-frame boundaries so rounding errors don't accumulate.
    bounds: List[int] = [0]
    acc = 0
    for lf in lat_pieces_frames:
        acc += lf
        bounds.append(int(round(acc / max(total_lat, 1) * n_pixel_frames)))

    out = np.empty_like(arr)
    for pi in range(len(pieces)):
        fa = bounds[pi]
        fb = bounds[pi + 1]
        if fb <= fa:
            continue
        lbl = piece_labels[pi]
        for fi in range(fa, fb):
            f = arr[fi].copy()
            _draw_header(f, lbl["label_top"], lbl["label_bot"])
            act = lbl.get("action")
            if act is not None:
                _draw_action_panel(f, act)
            # Bottom-right tag: run-level metadata so multi-mp4 grids
            # from the same sbatch are self-describing.
            _draw_tag(f, ride_tag, embedded_step)
            out[fi] = f
    t_after_ann = time.time()

    # ------------------------------------------------------------------
    # 4) Encode via ffmpeg.
    # ------------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_to_mp4(out, str(out_path), fps=float(fps))
    t_done = time.time()

    log.info(
        "write: %s | %d pixel frames | decode=%.2fs cast=%.2fs annotate=%.2fs "
        "encode=%.2fs total=%.2fs",
        out_path, n_pixel_frames,
        t_after_decode - t_before_decode,
        t_after_cast - t_after_decode,
        t_after_ann - t_after_cast,
        t_done - t_after_ann,
        t_done - t_start,
    )

    return {
        "out_path": str(out_path),
        "n_pixel_frames": n_pixel_frames,
        "seconds_decode": t_after_decode - t_before_decode,
        "seconds_annotate": t_after_ann - t_after_cast,
        "seconds_encode": t_done - t_after_ann,
        "seconds_total": t_done - t_start,
    }


def _draw_header(f: np.ndarray, title: str, subtitle: str) -> None:
    import cv2
    h, w = f.shape[:2]
    hdr_h = 42
    hdr_w = min(w, 360)
    f[:hdr_h, :hdr_w] = (
        f[:hdr_h, :hdr_w].astype(np.float32) * 0.22
    ).astype(np.uint8)
    cv2.putText(
        f, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
        (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        f, subtitle, (6, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
        (200, 220, 255), 1, cv2.LINE_AA,
    )


def _draw_action_panel(f: np.ndarray, action: torch.Tensor) -> None:
    import cv2
    vals = [float(v) for v in action.detach().float().cpu().numpy().tolist()]
    n_dims = len(vals)
    if n_dims == 0:
        return
    h, w = f.shape[:2]
    pw = 180
    px0 = max(0, w - pw)
    panel_h = min(h - 4, 18 + 22 * n_dims)
    f[:panel_h, px0:] = (
        f[:panel_h, px0:].astype(np.float32) * 0.22
    ).astype(np.uint8)
    cv2.putText(
        f, "action", (px0 + 6, 14),
        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 120), 1, cv2.LINE_AA,
    )
    labels = ["z2", "z7"] + [f"d{i}" for i in range(2, n_dims)]
    clip = 1.0
    cx = px0 + 100
    for i, v in enumerate(vals):
        y = 34 + 22 * i
        if y + 10 > h:
            break
        bl = int(abs(max(-clip, min(clip, v))) / clip * 60)
        cv2.line(f, (cx, y - 8), (cx, y + 8), (120, 120, 120), 1)
        col = (80, 220, 80) if v >= 0 else (80, 80, 220)
        if v >= 0:
            cv2.rectangle(f, (cx, y - 6), (cx + bl, y + 6), col, -1)
        else:
            cv2.rectangle(f, (cx - bl, y - 6), (cx, y + 6), col, -1)
        cv2.putText(
            f, f"{labels[i]} {v:+.2f}", (px0 + 6, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.34, col, 1, cv2.LINE_AA,
        )


def _draw_tag(f: np.ndarray, tag: str, embedded_step: int) -> None:
    import cv2
    h, w = f.shape[:2]
    txt = f"{tag} | step={embedded_step}" if embedded_step >= 0 else tag
    # Tiny tag in the bottom-left so you can tell runs apart at a glance.
    tag_h = 20
    tag_w = min(w, 260)
    f[h - tag_h:, :tag_w] = (
        f[h - tag_h:, :tag_w].astype(np.float32) * 0.28
    ).astype(np.uint8)
    cv2.putText(
        f, txt, (6, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
        (255, 240, 200), 1, cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Ride loader (delegates to eval_causal_AR.load_per_rank_ride_ar)
# ---------------------------------------------------------------------------


def load_ride_for_rollout(
    *,
    zarr_basename: str,
    latent_start_offset: int,
    total_frames: Optional[int],
    manifest_path: Optional[str],
    encoded_root: str,
    caption_root: str,
    motion_root: str,
    ss_vae_checkpoint: str,
    action_dims: List[int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Load ``(gt_latents, gt_actions, prompt_embeds, meta)`` for rollout.

    Uses ``eval_causal_AR.load_per_rank_ride_ar`` to handle manifest vs
    disk-fallback; if ``total_frames`` is None we auto-probe the
    ride's latent count via the disk path and use ``n_lat -
    latent_start_offset`` (i.e. roll to natural end, as Phase-1 training
    does).
    """
    # Phase-1 rolls to natural end-of-ride. To respect that, we need the
    # caller to tell us the ride length, or we probe it. Easiest is to
    # call ``load_per_rank_ride_ar`` with a large-but-safe upper bound
    # the first time to get the metadata, then re-call with the actual
    # cap if needed. In practice we know the ride length from the
    # manifest / disk attrs; do a quick probe:
    import zarr as zarr_lib
    from utils.eval_chain import _count_latent_frames, _load_ride_entry_from_disk, _build_ts_to_ride_dir
    from utils.eval_causal_chain import _find_ride_in_manifest

    zpath: Optional[str] = None
    n_lat: Optional[int] = None
    if manifest_path and Path(manifest_path).exists():
        try:
            manifest = torch.load(
                manifest_path, map_location="cpu", weights_only=False,
            )
            if isinstance(manifest, dict) and "rides" in manifest:
                rides = manifest["rides"]
            else:
                rides = manifest
            hit = _find_ride_in_manifest(rides, zarr_basename)
            if hit is not None:
                _, _, zpath, n_lat = hit
            del manifest
        except Exception as e:  # noqa: BLE001
            log.warning("manifest probe failed: %s", e)
    if zpath is None or n_lat is None:
        # Disk fallback to get the actual ride length.
        ts_map = _build_ts_to_ride_dir(Path(caption_root))
        ride_dict = _load_ride_entry_from_disk(
            zarr_basename, Path(encoded_root), Path(caption_root), ts_map,
        )
        zpath = str(ride_dict["zarr_path"])
        n_lat = int(ride_dict["n_latent_frames"])

    if total_frames is None:
        total_frames = max(0, int(n_lat) - int(latent_start_offset))
    else:
        total_frames = min(int(total_frames), int(n_lat) - int(latent_start_offset))
    if total_frames < 30:
        raise SystemExit(
            f"Ride {zarr_basename}: only {total_frames} latents available "
            f"after offset {latent_start_offset} (need >= 30 for warmup)."
        )

    log.info(
        "ride %s: n_lat=%d, start=%d, rolling for %d latent frames",
        zarr_basename, n_lat, latent_start_offset, total_frames,
    )

    initial_latents, prompt_embeds, noisy_fa_full, meta = load_per_rank_ride_ar(
        zarr_basename=zarr_basename,
        latent_start_offset=latent_start_offset,
        total_frames=total_frames,
        manifest_path=manifest_path,
        encoded_root=encoded_root,
        caption_root=caption_root,
        motion_root=motion_root,
        ss_vae_checkpoint=ss_vae_checkpoint,
        action_dims=action_dims,
        device=device,
    )
    # ``initial_latents`` is the whole slice [1, total_frames, C, H, W]
    # — exactly the ``gt_latents`` the pipeline needs.
    return initial_latents, prompt_embeds, noisy_fa_full, meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config", type=str,
        default="configs/longlive_phase1_rolling_staircase.yaml",
        help="Phase-1 config; pipeline topology (ladder / decay / kv / "
             "attn) is read from here.",
    )
    p.add_argument(
        "--ode_config", type=str,
        default="configs/action_ode_distill.yaml",
        help="ODE-distill config for ODERegression's shell (teacher "
             "merge, action head dims, scheduler). Not used for "
             "pipeline topology — that comes from --config.",
    )
    p.add_argument(
        "--student_ckpt", type=str, required=True,
        help="Path to the ODE-distilled student snapshot (default "
             "Phase-1 init: "
             "logs/action_ode_distill_E/action_ode_step0001000.pt).",
    )
    p.add_argument(
        "--output_dir", type=str, required=True,
        help="Where to write the per-rank mp4 and stats.json.",
    )
    p.add_argument(
        "--ride_tag", type=str, required=True,
        help="Human-readable tag drawn into the bottom-left of every "
             "frame (e.g. 'madrid_held', 'brighton_turny').",
    )
    p.add_argument(
        "--rank_zarr", type=str, required=True,
        help="Basename of the zarr file to roll out on this rank "
             "(e.g. '20240216101235.zarr').",
    )
    p.add_argument(
        "--rank_offset", type=int, default=0,
        help="Latent frame index to start the rollout at (default 0 "
             "= very first frame of the ride).",
    )
    p.add_argument(
        "--max_ride_frames", type=int, default=None,
        help="Optional cap on the number of latent frames to roll out "
             "(per-rank, after --rank_offset). Default = natural end "
             "of ride, like Phase-1 training.",
    )
    p.add_argument(
        "--max_rolling_steps", type=int, default=None,
        help="Optional cap on the number of rolling steps to run after "
             "warmup. Default = no cap.",
    )
    p.add_argument(
        "--manifest", type=str, default=None,
        help="Optional path to a ride manifest (.ride_manifest.pt). "
             "Used first; disk fallback otherwise.",
    )
    p.add_argument(
        "--encoded_root", type=str,
        default="/projects/u6ex/fbots/frodobots_encoded_weu",
    )
    p.add_argument(
        "--caption_root", type=str,
        default="/projects/u6ex/fbots/frodobots_captions/train",
    )
    p.add_argument(
        "--motion_root", type=str,
        default="/projects/u6ex/fbots/frodobots_motion",
    )
    p.add_argument(
        "--ss_vae_checkpoint", type=str,
        default="action_query/checkpoints/ss_vae_8free.pt",
    )
    p.add_argument(
        "--fps", type=int, default=8,
        help="Output mp4 fps.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Seed for the pipeline's noise samples inside rolling_step.",
    )
    p.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )

    # ---------------- debug_dmd mode ----------------
    p.add_argument(
        "--debug_dmd", action="store_true",
        help="Run the 4-way debug DMD visualizer instead of the regular "
             "rolling-staircase rollout. Produces 4 annotated mp4s per "
             "ride: student_hi, student_lo, teacher_raw, teacher_cleaned. "
             "Requires --v14_teacher_checkpoint.",
    )
    p.add_argument(
        "--v14_teacher_checkpoint", type=str,
        default="/scratch/u6ex/as1748.u6ex/ARRWM/logs/v14_balanced_weunz/causal_lora_step0006600.pt",
        help="Path to the v14 LoRA checkpoint for the debug_dmd teacher.",
    )
    p.add_argument(
        "--v14_lora_rank", type=int, default=256,
        help="LoRA rank (= v14 training rank). Must match checkpoint.",
    )
    p.add_argument(
        "--v14_lora_alpha", type=float, default=256.0,
    )
    p.add_argument(
        "--v14_lora_dropout", type=float, default=0.0,
    )
    p.add_argument(
        "--debug_num_blocks", type=int, default=10,
        help="Number of rolling blocks the debug_dmd mode runs.",
    )
    p.add_argument(
        "--debug_student_t_hi", type=int, default=750,
        help="Student input noise level for the 'student_hi' video.",
    )
    p.add_argument(
        "--debug_student_t_lo", type=int, default=250,
        help="Student input noise level for the 'student_lo' video.",
    )
    p.add_argument(
        "--debug_teacher_t", type=int, default=750,
        help="Teacher target-chunk noise level for teacher_{raw,cleaned}.",
    )
    p.add_argument(
        "--debug_cleanup_t", type=int, default=750,
        help="Noise level used by the teacher's cleanup pass on the "
             "student's previous commit (teacher_cleaned mode).",
    )
    p.add_argument(
        "--debug_teacher_num_gt_chunks", type=int, default=4,
        help="Number of GT chunks the teacher sees as left-context "
             "per forward. Mirrors Phase-1's real_score_num_gt_chunks.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Device.
    gpu_id = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    log.info(
        "Device=%s | dtype=%s | config=%s | ode_config=%s | student=%s",
        device, dtype, args.config, args.ode_config, args.student_ckpt,
    )

    # Seeds. We only set CPU + CUDA seeds; pipeline RNG will pull from
    # this pool inside its add_noise / randn calls.
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    # ------------------------------------------------------------------
    # Build student + pipeline + VAE.
    # ------------------------------------------------------------------
    ode_model, pipeline, vae = build_student_and_pipeline(
        phase1_cfg_path=args.config,
        ode_cfg_path=args.ode_config,
        student_ckpt_path=args.student_ckpt,
        device=device,
        dtype=dtype,
    )

    # Freeze everything; this script is pure inference.
    for m in (ode_model,):
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()

    # ------------------------------------------------------------------
    # Load a ride.
    # ------------------------------------------------------------------
    phase1_cfg = _load_omega(args.config)
    action_dims = list(getattr(phase1_cfg, "action_dims", [2, 7]))
    num_frame_per_block = int(getattr(phase1_cfg, "num_frame_per_block", 3))
    prime_kv_frames = int(getattr(phase1_cfg, "prime_kv_frames", 9))

    gt_latents, prompt_embeds, gt_actions, meta = load_ride_for_rollout(
        zarr_basename=args.rank_zarr,
        latent_start_offset=int(args.rank_offset),
        total_frames=args.max_ride_frames,
        manifest_path=args.manifest,
        encoded_root=args.encoded_root,
        caption_root=args.caption_root,
        motion_root=args.motion_root,
        ss_vae_checkpoint=args.ss_vae_checkpoint,
        action_dims=action_dims,
        device=device,
    )
    # Cast to pipeline dtype.
    gt_latents = gt_latents.to(dtype=dtype)
    gt_actions = gt_actions.to(dtype=dtype)
    prompt_embeds = prompt_embeds.to(dtype=dtype)

    log.info(
        "ride %s: latents=%s actions=%s prompt=%s",
        args.rank_zarr, tuple(gt_latents.shape),
        tuple(gt_actions.shape), tuple(prompt_embeds.shape),
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Recover the embedded step from the ckpt name if possible (for the
    # corner tag — cheap cosmetic).
    embedded_step = -1
    import re as _re
    mm = _re.search(r"step(\d+)", Path(args.student_ckpt).name)
    if mm:
        try:
            embedded_step = int(mm.group(1))
        except ValueError:
            embedded_step = -1

    if args.debug_dmd:
        # -------------------- debug_dmd mode --------------------------
        if not Path(args.v14_teacher_checkpoint).exists():
            raise SystemExit(
                f"--debug_dmd requires --v14_teacher_checkpoint to exist; "
                f"given {args.v14_teacher_checkpoint}"
            )
        # Teacher window = (teacher_num_gt_chunks + 1 prev + 1 live) * npb.
        teacher_window_chunks = int(args.debug_teacher_num_gt_chunks) + 2
        teacher_num_context_frames = teacher_window_chunks * num_frame_per_block
        teacher = build_v14_teacher(
            v14_ckpt_path=args.v14_teacher_checkpoint,
            v14_lora_rank=int(args.v14_lora_rank),
            v14_lora_alpha=float(args.v14_lora_alpha),
            v14_lora_dropout=float(args.v14_lora_dropout),
            num_context_frames=teacher_num_context_frames,
            action_per_frame=1,
            device=device,
            dtype=dtype,
        )
        log.info(
            "debug_dmd: rolling %d blocks | student t_hi=%d t_lo=%d | "
            "teacher t=%d | cleanup t=%d | %d GT chunks",
            args.debug_num_blocks, args.debug_student_t_hi,
            args.debug_student_t_lo, args.debug_teacher_t,
            args.debug_cleanup_t, args.debug_teacher_num_gt_chunks,
        )
        debug_out = run_debug_dmd(
            pipeline=pipeline,
            ode_model=ode_model,
            teacher=teacher,
            gt_latents=gt_latents,
            gt_actions=gt_actions,
            prompt_embeds=prompt_embeds,
            num_blocks=int(args.debug_num_blocks),
            student_t_hi=int(args.debug_student_t_hi),
            student_t_lo=int(args.debug_student_t_lo),
            teacher_t=int(args.debug_teacher_t),
            cleanup_t=int(args.debug_cleanup_t),
            teacher_num_gt_chunks=int(args.debug_teacher_num_gt_chunks),
        )

        # Write 4 mp4s, one per mode.
        mode_metadata = {
            "student_hi": (
                f"student {args.debug_student_t_hi}",
                int(args.debug_student_t_hi),
            ),
            "student_lo": (
                f"student {args.debug_student_t_lo}",
                int(args.debug_student_t_lo),
            ),
            "teacher_raw": (
                f"teacher raw {args.debug_teacher_t}",
                int(args.debug_teacher_t),
            ),
            "teacher_cleaned": (
                f"teacher cleaned {args.debug_teacher_t} (cu {args.debug_cleanup_t})",
                int(args.debug_teacher_t),
            ),
        }
        write_stats_all: Dict[str, Dict[str, Any]] = {}
        for mode_key, (mode_label, _t) in mode_metadata.items():
            out_mp4 = out_dir / (
                f"{Path(args.rank_zarr).stem}_{args.ride_tag}_{mode_key}.mp4"
            )
            mode_ride_tag = f"{args.ride_tag}_{mode_key}"
            ws = _annotate_and_write_mp4(
                prime_latents=debug_out["prime_latents"],
                commits=debug_out["records"][mode_key],
                vae=vae,
                device=device,
                fps=int(args.fps),
                out_path=out_mp4,
                ride_tag=mode_ride_tag,
                ride_basename=Path(args.rank_zarr).stem,
                num_frame_per_block=num_frame_per_block,
                embedded_step=embedded_step,
            )
            write_stats_all[mode_key] = ws
            log.info("debug_dmd: wrote %s (%s)", out_mp4, mode_label)

        # Stats JSON (combined).
        stats = {
            "mode": "debug_dmd",
            "student_ckpt": args.student_ckpt,
            "v14_teacher_checkpoint": args.v14_teacher_checkpoint,
            "embedded_step": embedded_step,
            "rank_zarr": args.rank_zarr,
            "rank_offset": int(args.rank_offset),
            "ride_tag": args.ride_tag,
            "debug_dmd_stats": debug_out["stats"],
            "writes": write_stats_all,
            "ride_meta": meta,
        }
        stats_path = out_dir / (
            f"{Path(args.rank_zarr).stem}_{args.ride_tag}_debug_dmd_stats.json"
        )
        with stats_path.open("w") as fh:
            json.dump(stats, fh, indent=2, default=str)
        log.info(
            "debug_dmd: all 4 mp4s + stats written to %s", out_dir,
        )
        return

    # -------------------- regular rolling-staircase mode --------------
    capture = run_rollout_and_capture(
        pipeline=pipeline,
        gt_latents=gt_latents,
        gt_actions=gt_actions,
        prompt_embeds=prompt_embeds,
        num_frame_per_block=num_frame_per_block,
        prime_kv_frames=prime_kv_frames,
        max_rolling_steps=(
            int(args.max_rolling_steps)
            if args.max_rolling_steps is not None else None
        ),
    )
    if not capture["commits"]:
        log.error("No rolling-step records captured — ride may be too short.")
        return

    out_mp4 = out_dir / f"{Path(args.rank_zarr).stem}_{args.ride_tag}.mp4"
    write_stats = _annotate_and_write_mp4(
        prime_latents=capture["prime_latents"],
        commits=capture["commits"],
        vae=vae,
        device=device,
        fps=int(args.fps),
        out_path=out_mp4,
        ride_tag=args.ride_tag,
        ride_basename=Path(args.rank_zarr).stem,
        num_frame_per_block=num_frame_per_block,
        embedded_step=embedded_step,
    )

    stats = {
        "student_ckpt": args.student_ckpt,
        "embedded_step": embedded_step,
        "rank_zarr": args.rank_zarr,
        "rank_offset": int(args.rank_offset),
        "ride_tag": args.ride_tag,
        "n_commits": capture["stats"]["n_commits"],
        "wall_seconds_rollout": capture["stats"]["wall_seconds"],
        "rolling_steps_per_sec": capture["stats"]["rolling_steps_per_sec"],
        "out_path": write_stats["out_path"],
        "n_pixel_frames": write_stats["n_pixel_frames"],
        "seconds_decode": write_stats["seconds_decode"],
        "seconds_annotate": write_stats["seconds_annotate"],
        "seconds_encode": write_stats["seconds_encode"],
        "ride_meta": meta,
    }
    stats_path = out_dir / f"{Path(args.rank_zarr).stem}_{args.ride_tag}_stats.json"
    with stats_path.open("w") as fh:
        json.dump(stats, fh, indent=2, default=str)
    log.info("Wrote %s (mp4 + stats).", out_mp4)


if __name__ == "__main__":
    main()
