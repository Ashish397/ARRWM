# pyright: reportGeneralTypeIssues=false
"""DMD2 model for Phase-1 Action-Forcing training (NO STAIRCASE).

This model is a thin, action-aware port of
``Causal-Forcing/model/dmd.py`` onto our infrastructure. The core
training mechanics — generator_loss / critic_loss / _compute_kl_grad /
compute_distribution_matching_loss / _run_generator — mirror CF
1:1 with two action-aware additions and the DMD2 upgrades retained
from the staircase variant:

  * ``_run_generator`` builds a per-frame action conditioning dict
    (Stream A modulation + Stream B tokens) from the input ``gt_actions``
    and threads it through to the Action-Forcing training pipeline.
  * Real and fake scorers receive the same per-frame action conditioning
    on the *student's* predicted 21-frame video — no GT context, no KV
    cache, fully symmetric. The grad direction is purely
    ``(pred_fake - pred_real)`` on a noisy version of the student's
    output (CF parity).
  * ODE checkpoint loading + v14 LoRA loading + generator->fake_score
    mirroring + non-strict load with optional strict gate (DMD2 retained
    upgrades from the staircase model).
  * Auxiliary action heads (``action_critic`` + ``state_probe``) are
    instantiated and loaded from the ODE checkpoint so the existing
    trainer plumbing (`action_token_projection`, etc.) keeps working,
    but their AUX LOSSES ARE NOT WIRED IN by this model — the user's
    plan for Phase-1 Action-Forcing is "DMD almost exactly like CF" so we keep
    only the DMD path. The heads remain frozen and unused by the
    losses; they live on the model only so the ODE checkpoint loads
    cleanly and downstream tooling that introspects `.action_critic`
    / `.state_probe` does not break.

The Action-Forcing training pipeline
``pipeline.action_forcing_training.ActionForcingTrainingPipeline``
runs the truncated random-exit denoise loop with persistent KV cache
and exit-flag gradient gating; ``_run_generator`` matches CF's
"slice last N frames" and ``start_gradient_frame_index`` semantics.

CF-style long rollouts (``rollout_frames > num_training_frames``):
the trainer rolls out ``rollout_frames`` total but only the LAST
``num_training_frames`` frames carry gradient (the gate inside
``ActionForcingTrainingPipeline.inference_with_trajectory`` runs the leading
``rollout_frames - num_training_frames`` exit-flag forwards under
``no_grad`` to warm the KV cache). After the rollout, this model
slices the pred to the LAST ``num_training_frames`` frames AND
slices the per-frame action streams in ``conditional_dict`` /
``unconditional_dict`` to the same window before passing to the
bidirectional scorer (whose seq_len is sized to
``num_training_frames``). When ``rollout_frames == num_training_frames``
(default), all of this is a no-op — every frame backprops.

CF-style first-chunk BOUNDARY mask (long-rollout mode only): the
first ``num_frame_per_block`` frames of the scoring window read
their attention context from the warmup KV cache (frames generated
under ``no_grad``); they are the "boundary" between warmup and
gradient-active regions. CF
(``Causal-Forcing/long_video/model/base.py:169-177``) zeros the
gradient on this first chunk so the student is not penalised for
the warmup→grad transition pattern. We do the same. We
deliberately SKIP CF's decode→re-encode anchoring of the boundary
frame: CF replaces the boundary latent with a freshly VAE-encoded
pixel frame; with the gradient masked off, the boundary frame's
contents cannot affect the loss, so the round-trip is wasted
compute. Mask is None when ``rollout_frames == num_training_frames``
(classic and pure-extension modes) — every scoring frame backprops.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from model.base import SelfForcingModel
from pipeline.action_forcing_training import (
    _ACTION_STREAM_KEYS,
    _slice_per_frame_streams,
)
from utils.debug_option import DEBUG

try:
    import peft  # type: ignore
    from peft import (  # type: ignore
        LoraConfig,
        set_peft_model_state_dict,
    )
    _HAS_PEFT = True
except Exception:
    peft = None  # type: ignore
    LoraConfig = None  # type: ignore
    set_peft_model_state_dict = None  # type: ignore
    _HAS_PEFT = False


def _is_main() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def _slice_baseline_scoring_window(
    cond_dict: Dict[str, torch.Tensor],
    rollout_frames: int,
    num_training_frames: int,
    seed_frames: int = 0,
) -> Dict[str, torch.Tensor]:
    """Slice ``cond_dict``'s per-frame action streams to the LAST
    ``num_training_frames`` of the BASELINE rollout window.

    ``seed_frames`` accounts for an optional KV-cache prefill prefix
    on the front of ``cond_dict``. When the trainer feeds the pipeline
    a ``conditional_dict`` covering ``seed_frames + rollout_frames``
    frames (seed actions first, then rollout actions — the layout the
    pipeline indexes via absolute ``current_start_frame``), the scorer
    still wants the LAST ``num_training_frames`` of the ROLLOUT half.
    The slice becomes
    ``[seed_frames + rollout_frames - num_training_frames :
       seed_frames + rollout_frames]``.

    With ``seed_frames == 0`` (no prefill / pre-2026-04-26 path) the
    slice is ``[rollout_frames - num_training_frames : rollout_frames]``,
    which collapses correctly across all three legacy regimes:

      - classic           (rollout_frames == num_training_frames):
            ``[0:num_training_frames]``
      - long-rollout      (rollout_frames >  num_training_frames):
            ``[rollout_frames - num_training_frames : rollout_frames]``
            = LAST num_training_frames of cond_dict
      - extension mode    (cond_dict longer than rollout_frames):
            same; extension frames are never scored.

    Non-action keys (``prompt_embeds`` etc.) are passed through.
    """
    start = seed_frames + rollout_frames - num_training_frames
    if start < seed_frames:
        raise ValueError(
            f"rollout_frames ({rollout_frames}) must be >= "
            f"num_training_frames ({num_training_frames}); the "
            f"scoring window is the LAST num_training_frames of the "
            f"baseline rollout."
        )
    return _slice_per_frame_streams(
        cond_dict, frame_start=start, frame_count=num_training_frames,
    )


class ActionForcingDMD(SelfForcingModel):
    """DMD2 trainer module for the Action-Forcing (no-staircase) Phase-1 recipe."""

    def __init__(self, args, device):
        super().__init__(args, device)

        self.num_frame_per_block = int(getattr(args, "num_frame_per_block", 3))
        if self.num_frame_per_block > 1 and hasattr(self.generator, "model"):
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = False
        self.num_training_frames = int(getattr(args, "num_training_frames", 21))
        # ``rollout_frames`` is the actual rollout length per training
        # iter. When None / unset it defaults to ``num_training_frames``
        # (= classic CF chunkwise behavior, every rolled frame
        # backprops). When set greater, the leading
        # ``rollout_frames - num_training_frames`` frames warm the KV
        # cache under ``no_grad`` and only the trailing
        # ``num_training_frames`` carry gradient. The bidirectional
        # scorer is always fed exactly ``num_training_frames`` frames
        # (the trailing slice), since its seq_len was sized to that
        # window above and feeding it longer sequences would either
        # silently truncate or break attention.
        rollout_frames_raw = getattr(args, "rollout_frames", None)
        self.rollout_frames = int(
            rollout_frames_raw if rollout_frames_raw is not None
            else self.num_training_frames
        )
        if self.rollout_frames < self.num_training_frames:
            raise ValueError(
                f"rollout_frames ({self.rollout_frames}) must be >= "
                f"num_training_frames ({self.num_training_frames}); "
                f"the gradient/scoring window is the LAST "
                f"num_training_frames frames of the rollout."
            )

        # ``dmd_context``: string-valued, ALWAYS SET. Selects what real_score
        # sees as ``clean_x`` during DMD scoring; fake_score's clean_x is
        # always the "self" (student-rolled) view regardless of mode, so
        # the (fake - real) gradient is computed under matched fake-side
        # conditioning across modes.
        #
        #   "self": real_score's clean_x = student-rolled view = the
        #           21-frame window shifted back by 1 chunk (3 frames)
        #           from the noisy half. For batch i=1: [last seed chunk
        #           GT, sdn[:18]]. For i>=2: sdn[(i-1)*21-3 : i*21-3]
        #           entirely from the cumulative student rollout. No
        #           noise added — both scorers see the same view.
        #
        #   "GT":   real_score's clean_x = ride GT at the same shifted
        #           positions, with a small ``clean_x_aug_t`` of noise
        #           applied to keep the reference from being perfectly
        #           clean (symmetry-breaking; without aug_t the gradient
        #           is dominated by a degenerate "real is perfect"
        #           direction). v14 teacher-forcing parity. fake_score's
        #           clean_x stays the "self" view (no noise).
        #
        # Default ``"GT"`` — the v14 teacher-forcing contract that the
        # LoRA was actually trained against gives a tight DMD signal.
        self.dmd_context = str(getattr(args, "dmd_context", "GT")).strip().lower()
        if self.dmd_context not in ("self", "gt"):
            raise ValueError(
                f"dmd_context must be 'self' or 'GT' (case-insensitive); "
                f"got {getattr(args, 'dmd_context', None)!r}."
            )
        # Normalise canonical case for downstream comparisons.
        self.dmd_context = "GT" if self.dmd_context == "gt" else "self"

        # Number of leading GT frames that seed the KV cache before the
        # student's rolling rollout starts (= the model's KV-cache size,
        # default 9 = 3 chunks of ``num_frame_per_block=3``). This is
        # the prefill amount and ONLY the prefill amount: the
        # clean/noisy SHIFT used by the bidirectional scorer is fixed
        # at ``num_frame_per_block`` (= 1 chunk), independent of the
        # seed size. Must be a multiple of ``num_frame_per_block``
        # (seed loop runs in chunks) and at least ``num_frame_per_block``
        # (so the batch-1 clean_x mix can pull the last 1 chunk from
        # seed when the rollout has only N=21 frames).
        self.dmd_context_clean_frames = int(
            getattr(args, "dmd_context_clean_frames", 9)
        )
        if self.dmd_context_clean_frames < self.num_frame_per_block:
            raise ValueError(
                "dmd_context_clean_frames must be >= num_frame_per_block "
                f"({self.num_frame_per_block}); got "
                f"{self.dmd_context_clean_frames}."
            )
        if self.dmd_context_clean_frames % self.num_frame_per_block != 0:
            raise ValueError(
                f"dmd_context_clean_frames "
                f"({self.dmd_context_clean_frames}) must be a multiple "
                f"of num_frame_per_block ({self.num_frame_per_block}) "
                f"so the seed prefill loop runs in whole chunks."
            )
        if self.dmd_context_clean_frames >= self.num_training_frames:
            raise ValueError(
                f"dmd_context_clean_frames "
                f"({self.dmd_context_clean_frames}) must be < "
                f"num_training_frames ({self.num_training_frames})."
            )

        # ``clean_x_aug_t``: noise level applied to real_score's clean_x
        # in ``"GT"`` mode. Default ``0`` = no noise on the GT clean half
        # (which the diagnostic ``_diag_p1/test_dmd_inference.py`` 3-roll
        # mask-OFF + anchor sweep settled on). fake_score's clean_x is
        # never noised. Ignored in ``"self"`` mode.
        self.clean_x_aug_t = int(getattr(args, "clean_x_aug_t", 0))
        if self.clean_x_aug_t < 0 or self.clean_x_aug_t >= int(
            getattr(args, "num_train_timestep", 1000)
        ):
            raise ValueError(
                f"clean_x_aug_t ({self.clean_x_aug_t}) must be in "
                f"[0, num_train_timestep="
                f"{getattr(args, 'num_train_timestep', 1000)}). Use 0 to "
                "disable, or a SMALL value (≲ 50) for symmetry-breaking."
            )

        # ``dmd_clean_x_anchor_frames``: number of EXTRA frames the
        # student rolls at the START of every ride to anchor batch-1's
        # ``clean_x_self`` window — fixes the seed/student quality
        # discontinuity that used to OOD the bidir scorer when
        # ``clean_x_self`` was assembled from ``[seed_last_chunk,
        # pred[:N-shift]]``. New geometry:
        #   pipeline rolls (num_training_frames + anchor_frames) frames
        #   pred_for_scoring = pred[:, anchor_frames:]   (= last N, noisy_x)
        #   clean_x_self     = pred[:, :num_training_frames]  (= first N)
        # Hardcoded to ``num_frame_per_block`` (= the clean/noisy shift)
        # — the model needs exactly ``shift`` extra frames at the front
        # to slide noisy_x forward by one chunk while keeping clean_x_self
        # entirely in the rolled-out range. Once per ride at the FRONT,
        # NOT once per batch.
        self.dmd_clean_x_anchor_frames = int(self.num_frame_per_block)

        # ``teacher_freeze_detect_enabled``: per-frame outlier-rejection
        # on the teacher (real_score) prediction. Two modes — selected
        # by ``teacher_freeze_mode``:
        #
        #  * ``"mae"`` (legacy default): per-frame MAE between
        #    ``pred_real_image`` and the GT video at the same time
        #    positions; frames with MAE > ``teacher_freeze_threshold``
        #    × median(MAE) get masked. Calibration sweep
        #    (``_diag_p1/freeze_threshold_calibration.py`` over 16
        #    motion-rich windows) shows ratios cap at ~1.4× in the
        #    middle and ~1.8× at the structural boundary, so
        #    ``threshold=2.0`` is "above natural variance" and only
        #    fires on severe freezes.
        #
        #  * ``"action"``: run the real_score's predicted latent AND
        #    the student's predicted latent through the frozen action
        #    teacher (CoTracker → ss_vae → 8-d z). Per slot, compute
        #    cosine similarity between ``z_real`` and ``z_student``.
        #    Flag slots where cos_sim < ``teacher_freeze_action_threshold``
        #    (default 0.0 → opposite direction = "very wrong / backwards"),
        #    broadcast to per-frame mask, AND into ``gradient_mask``.
        #    Lets the student learn better actions than the teacher
        #    when the teacher is action-wrong but visually plausible
        #    (which the MAE mode would miss).
        #    Requires the trainer to attach
        #    ``self._action_teacher_fn = self._compute_teacher_z_per_slot``
        #    — done in ``_build_action_teacher`` setup.
        self.teacher_freeze_detect_enabled = bool(
            getattr(args, "teacher_freeze_detect_enabled", False)
        )
        self.teacher_freeze_mode = str(
            getattr(args, "teacher_freeze_mode", "mae")
        ).strip().lower()
        if self.teacher_freeze_mode not in ("mae", "action"):
            raise ValueError(
                f"teacher_freeze_mode must be 'mae' or 'action'; "
                f"got {self.teacher_freeze_mode!r}."
            )
        self.teacher_freeze_threshold = float(
            getattr(args, "teacher_freeze_threshold", 2.0)
        )
        if self.teacher_freeze_threshold <= 1.0:
            raise ValueError(
                f"teacher_freeze_threshold ({self.teacher_freeze_threshold}) "
                "must be > 1.0 (= multiplier on median MAE; <= 1.0 would "
                "flag the median frame itself, masking ≥ half the chunk)."
            )
        # Cosine-similarity cutoff for the "action" mode. Default 0.0:
        # flag a slot only when teacher's predicted action and student's
        # predicted action point in OPPOSITE directions (cos < 0). Tune
        # higher (e.g. 0.3) to be stricter — flag any meaningful
        # disagreement; lower (e.g. -0.3) to only flag near-180°
        # reversals.
        self.teacher_freeze_action_threshold = float(
            getattr(args, "teacher_freeze_action_threshold", 0.0)
        )
        if self.teacher_freeze_action_threshold < -1.0 or self.teacher_freeze_action_threshold > 1.0:
            raise ValueError(
                f"teacher_freeze_action_threshold "
                f"({self.teacher_freeze_action_threshold}) must be in [-1, 1] "
                "(cosine similarity cutoff)."
            )
        # Filled by the trainer once the frozen action teacher is built;
        # see ``_build_action_teacher`` in
        # ``trainer/causal_action_forcing_train.py``.
        self._action_teacher_fn = None

        if getattr(args, "gradient_checkpointing", False):
            try:
                self.generator.enable_gradient_checkpointing()
                self.fake_score.enable_gradient_checkpointing()
            except Exception as e:
                if _is_main():
                    logging.warning("gradient_checkpointing enable failed: %s", e)

        # Resize the bidirectional scorer wrappers. ``BaseModel.
        # _initialize_models`` sized them to the staircase batched
        # window (4*npb), which is too small for our symmetric scoring.
        # ``num_training_frames`` covers BOTH the legacy "no context"
        # path and the ``dmd_context=true`` path — the wrapper's
        # ``seq_len`` covers the noisy half only; the model doubles
        # the kv length internally via the teacher-forcing block mask
        # when ``clean_x`` is passed.
        n_score_frames = self.num_training_frames
        for scorer_name in ("real_score", "fake_score"):
            scorer = getattr(self, scorer_name)
            scorer._base_seq_len = n_score_frames * 1560
            scorer.seq_len = scorer._base_seq_len
            if getattr(self, "_action_patch_enabled", False):
                # adjust_seq_len_for_action_tokens reads from
                # _base_seq_len, so we redo it after the override.
                scorer.adjust_seq_len_for_action_tokens(
                    num_frames=n_score_frames, action_per_frame=1,
                )

        # DMD hyperparameters (mirrors CF exactly).
        self.num_train_timestep = int(getattr(args, "num_train_timestep", 1000))
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        if hasattr(args, "real_guidance_scale"):
            self.real_guidance_scale = float(args.real_guidance_scale)
            self.fake_guidance_scale = float(getattr(args, "fake_guidance_scale", 0.0))
        else:
            self.real_guidance_scale = float(getattr(args, "guidance_scale", 1.0))
            self.fake_guidance_scale = 0.0
        self.timestep_shift = float(getattr(args, "timestep_shift", 1.0))
        self.ts_schedule = bool(getattr(args, "ts_schedule", True))
        self.ts_schedule_max = bool(getattr(args, "ts_schedule_max", False))
        self.min_score_timestep = int(getattr(args, "min_score_timestep", 0))

        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None

        self.dmd_loss_weight = float(getattr(args, "dmd_loss_weight", 1.0))

        # Fake-score updates: ON by default for DMD2.
        self.fake_score_updates_enabled = bool(
            getattr(args, "fake_score_updates_enabled", True)
        )

        # Auxiliary action-supervision heads — instantiated for ODE
        # checkpoint compatibility. NOT consumed by the loss path in
        # this model (Phase-1 Action-Forcing is pure DMD).
        self._build_action_aux_heads_compat(args, device)

        # ----- Load checkpoints -----
        self._load_generator_from_ode_checkpoint(args, device)
        self._load_real_score_with_v14_lora(args, device)
        self._mirror_generator_into_fake_score()

        # dmd_context is always set. Configure both bidirectional
        # scorers for teacher-forcing input (clean_x + noisy_x joint
        # window). The clean/noisy SHIFT is fixed at one chunk
        # (= ``num_frame_per_block`` frames), NOT the seed size:
        #   * ``model.context_shift`` (chunks) = 1
        #   * ``model.tf_rope_offset_frames`` (frames) = ``num_frame_per_block``
        # The bidir self-attn patch reads ``tf_rope_offset_frames`` to
        # offset the noisy half's RoPE positions so clean is at [0, F)
        # and noisy is at [shift, shift+F).
        # Must run AFTER ``_load_real_score_with_v14_lora`` (peft.merge_
        # and_unload rebuilds real_score.model and would drop attrs set
        # before) and AFTER ``_mirror_generator_into_fake_score``.
        # ``tf_rope_offset_frames`` knob — overridable via env for A/B
        # diagnostics. Default = ``num_frame_per_block`` (= 1-chunk
        # shift). Setting ``DIAG_TF_ROPE_OFFSET=9`` (= dmd_context_clean_frames)
        # tests v14's "clean half = cf frames, noisy half stacked
        # after" alternative training layout.
        _tf_rope_off = int(os.environ.get(
            "DIAG_TF_ROPE_OFFSET",
            str(int(self.num_frame_per_block)),
        ))
        for scorer_name in ("real_score", "fake_score"):
            m = getattr(self, scorer_name).model
            m.context_shift = 1
            m.tf_rope_offset_frames = _tf_rope_off
            # Pass the chunk size to the bidir TF block_mask builder
            # (action_model_patch._prepare_tf_block_mask_cached). The v14
            # LoRA was fine-tuned with the same chunk size as the causal
            # student (typically 3); match here so the bidir scorer's
            # joint self-attn uses v14's training-contract block_mask.
            m.num_frame_per_block = int(self.num_frame_per_block)
            # Force any cached block_mask rebuild (causal path only;
            # bidir doesn't use block_mask but harmless to clear).
            m.block_mask = None
            # Drop any stale TF block_mask cache entries from a prior
            # config so the next forward rebuilds with the new dims.
            if hasattr(m, "_tf_block_mask_cache"):
                m._tf_block_mask_cache = {}

        if _is_main():
            logging.info(
                "[ActionForcingDMD] dmd_context=%r: set "
                "context_shift=1 chunk (= tf_rope_offset_frames=%d) on "
                "real_score and fake_score. KV-cache seed prefill = "
                "%d frames. clean_x_aug_t=%d (only used in 'GT' mode).",
                self.dmd_context,
                self.num_frame_per_block,
                self.dmd_context_clean_frames,
                self.clean_x_aug_t,
            )

        # Finalize freezes.
        for p in self.real_score.parameters():
            p.requires_grad_(False)
        if not self.fake_score_updates_enabled:
            for p in self.fake_score.parameters():
                p.requires_grad_(False)
            self._fake_score_trainable = False
        else:
            self._fake_score_trainable = True

        # ``teacher_freeze_mode='action'`` requires the action-aux
        # stack to be live so the student keeps a learning signal when
        # we cut the DMD gradient on a slot:
        #
        #   * action_teacher_mode != "off" — so the trainer's
        #     ``_build_action_teacher`` actually loads the CoTracker +
        #     ss_vae and attaches ``self._action_teacher_fn``;
        #   * action_critic_aux_enabled = true — so the action-critic
        #     z-guidance loss runs on the student's pred (= the
        #     replacement supervision when DMD is masked off);
        #   * state_probe_aux_enabled = true — so the state probe head
        #     runs (CF parity + helps the student converge on
        #     coherent action-conditioned latents).
        #
        # The per-frame state-TOKEN branch is a separate mechanism set
        # by the v14 LoRA's training-time config; if v14 was trained
        # with state tokens, ``real_score.model.state_tokens_per_frame``
        # is bumped at LoRA load time and ``action_model_patch`` will
        # raise loudly at forward if state_tokens aren't passed —
        # surfaced naturally without an extra assert here.
        if (
            self.teacher_freeze_detect_enabled
            and self.teacher_freeze_mode == "action"
        ):
            ac_aux = bool(getattr(args, "action_critic_aux_enabled", False))
            sp_aux = bool(getattr(args, "state_probe_aux_enabled", False))
            t_mode = str(
                getattr(args, "action_teacher_mode", "off") or "off"
            ).strip().lower()
            missing = []
            if t_mode == "off":
                missing.append(
                    "action_teacher_mode != 'off' "
                    f"(got {t_mode!r}) — needed to load CoTracker + ss_vae"
                )
            if not ac_aux:
                missing.append(
                    "action_critic_aux_enabled: true — needed to give "
                    "the student a replacement gradient when DMD is masked"
                )
            if not sp_aux:
                missing.append(
                    "state_probe_aux_enabled: true — needed for state "
                    "probe action supervision"
                )
            if missing:
                raise RuntimeError(
                    "teacher_freeze_mode='action' requires the full "
                    "action-aux stack so the student keeps a learning "
                    "signal when DMD is gated off. Missing prerequisites:\n  - "
                    + "\n  - ".join(missing)
                )
            real_state_tokens = int(getattr(
                getattr(self.real_score, "model", None),
                "state_tokens_per_frame", 0,
            ))
            if _is_main():
                logging.info(
                    "[ActionForcingDMD] teacher_freeze_mode='action' "
                    "prerequisites verified: action_teacher_mode=%r, "
                    "action_critic_aux_enabled=%s, "
                    "state_probe_aux_enabled=%s, "
                    "teacher_freeze_action_threshold=%.3f. "
                    "(real_score.state_tokens_per_frame=%d — set by v14 "
                    "LoRA's training-time state-token branch; the "
                    "scorer forward checks this for itself.)",
                    t_mode, ac_aux, sp_aux,
                    self.teacher_freeze_action_threshold,
                    real_state_tokens,
                )

        # Pipeline is set later by the trainer (after DDP wrap).
        self.inference_pipeline = None

        # Streaming-mode persistent state. ``None`` when no sequence
        # is open; a dict tracking sequence-level state when active.
        # See ``setup_sequence`` + ``generate_next_chunk`` for the
        # contract. LongLive parity:
        # ``LongLive/model/streaming_training.py``.
        self.streaming_state: Optional[Dict[str, Any]] = None
        self.streaming_chunk_size: int = int(getattr(args, "streaming_chunk_size", self.num_training_frames))
        self.streaming_min_new_frame: int = int(getattr(args, "streaming_min_new_frame", self.streaming_chunk_size - self.num_frame_per_block))
        self.streaming_max_length: int = int(getattr(args, "streaming_max_length", 57))

        # SC-DMD (Salt paper, 2604.03118v1) — semigroup defect
        # regularizer L_SC = E[||Ψ_θ^{ts→te}(x_ts) - Ψ_θ^{tm→te}(Ψ_θ^{ts→tm}(x_ts))||²]
        # default DISABLED. Single-chunk (chunk-0, fresh KV cache) for
        # cost: 2 extra DiT forwards on a 3-frame chunk per gen step
        # ≈ 1-2% extra wallclock on the 7-chunk × 21-frame full
        # rollout. ``_sc_kv_cache`` / ``_sc_crossattn_cache`` are
        # lazily allocated on first ``sc_dmd_loss`` call.
        self.sc_dmd_enabled = bool(getattr(args, "sc_dmd_enabled", False))
        self._sc_kv_cache: Optional[list] = None
        self._sc_crossattn_cache: Optional[list] = None

    # ------------------------------------------------------------------
    # Auxiliary head compat shim (heads instantiated, frozen, unused)
    # ------------------------------------------------------------------
    def _build_action_aux_heads_compat(self, args, device) -> None:
        """Instantiate aux heads ONLY for ODE checkpoint compatibility.

        Phase-1 Action-Forcing's loss path does NOT use ``action_critic`` /
        ``state_probe`` aux losses (CF-parity DMD only). But the ODE
        checkpoint stores their state_dicts under ``action_critic`` /
        ``state_probe`` keys, so we instantiate matching modules
        whenever the corresponding ``*_aux_enabled`` flag is True so
        that ``_load_generator_from_ode_checkpoint`` can populate them
        without losing keys (loud non-strict load logs would fire on
        every run otherwise).

        Both heads get ``requires_grad_(False)`` and ``.eval()`` so
        they're truly inert. The trainer's optimizer never sees them.
        """
        self.action_critic = None
        self.state_probe = None

        # action_critic
        if bool(getattr(args, "action_critic_aux_enabled", False)):
            try:
                from model.action_critic import ActionCritic
                critic = ActionCritic(
                    latent_channels=16,
                    action_dim=int(getattr(args, "raw_action_dim", 2)),
                    z_out_dim=int(getattr(args, "action_critic_z_out_dim", 8)),
                    base_channels=int(getattr(args, "action_critic_base_channels", 128)),
                    num_res_blocks=int(getattr(args, "action_critic_num_blocks", 4)),
                    chunk_frames=self.num_frame_per_block,
                ).to(device=device)
                critic.requires_grad_(False)
                critic.eval()
                self.action_critic = critic
            except Exception as exc:
                if _is_main():
                    logging.warning(
                        "[ActionForcingDMD] action_critic compat instantiation failed: %s",
                        exc,
                    )
                self.action_critic = None

        # state_probe (attaches to generator wrapper)
        if (
            bool(getattr(args, "state_probe_aux_enabled", False))
            and hasattr(self.generator, "adding_state_probe_branch")
        ):
            try:
                self.generator.adding_state_probe_branch(
                    n_chunks=max(1, self.num_training_frames // self.num_frame_per_block),
                    z_out_dim=int(getattr(args, "state_head_out_dim", 8)),
                    dim=int(getattr(self.generator.model, "dim", 2048)),
                    probe_dim=int(getattr(args, "state_probe_dim", 256)),
                    num_heads=int(getattr(args, "state_probe_num_heads", 8)),
                    n_taps=int(getattr(args, "state_probe_n_taps", 6)),
                    num_frame_per_block=self.num_frame_per_block,
                )
                probe = getattr(self.generator, "_state_probe", None)
                if probe is not None:
                    probe.to(device=device)
                    probe.requires_grad_(False)
                    probe.eval()
                    self.state_probe = probe
            except Exception as exc:
                if _is_main():
                    logging.warning(
                        "[ActionForcingDMD] state_probe compat instantiation failed: %s",
                        exc,
                    )
                self.state_probe = None

    # ------------------------------------------------------------------
    # Checkpoint loading (ported from staircase model, simplified)
    # ------------------------------------------------------------------
    def _load_generator_from_ode_checkpoint(self, args, device) -> None:
        ckpt_path = getattr(args, "ode_generator_checkpoint", None)
        if not ckpt_path:
            if _is_main():
                logging.warning(
                    "[ActionForcingDMD] No ode_generator_checkpoint provided; generator "
                    "starts from raw Wan weights (not recommended).",
                )
            return
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"ode_generator_checkpoint not found: {ckpt_path}")

        if _is_main():
            logging.info("[ActionForcingDMD] Loading generator from %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "generator" not in ckpt:
            raise KeyError(
                f"ODE checkpoint missing 'generator' key: {list(ckpt.keys())}"
            )

        strict = bool(getattr(args, "strict_ode_load", False))
        missing, unexpected = self.generator.model.load_state_dict(
            ckpt["generator"], strict=False,
        )
        if _is_main():
            logging.info(
                "[ActionForcingDMD] generator load: missing=%d unexpected=%d",
                len(missing), len(unexpected),
            )
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"[ActionForcingDMD] strict_ode_load=True but generator load has "
                f"missing={len(missing)} unexpected={len(unexpected)} keys."
            )

        # Action heads: action_projection + action_token_projection live on
        # BaseModel; load if present.
        if "action_projection" in ckpt and self.action_projection is not None:
            ap_missing, ap_unexpected = self.action_projection.load_state_dict(
                ckpt["action_projection"], strict=False,
            )
            if _is_main():
                logging.info(
                    "[ActionForcingDMD] action_projection load: missing=%d unexpected=%d",
                    len(ap_missing), len(ap_unexpected),
                )
            if strict and (ap_missing or ap_unexpected):
                raise RuntimeError(
                    "[ActionForcingDMD] strict_ode_load=True but action_projection has "
                    "missing/unexpected keys."
                )

        for key, attr in (
            ("action_token_projection", "action_token_projection"),
            ("action_critic", "action_critic"),
            ("state_probe", "state_probe"),
        ):
            if key in ckpt and getattr(self, attr, None) is not None:
                try:
                    h_missing, h_unexpected = getattr(self, attr).load_state_dict(
                        ckpt[key], strict=False,
                    )
                    if _is_main():
                        logging.info(
                            "[ActionForcingDMD] %s load: missing=%d unexpected=%d",
                            key, len(h_missing), len(h_unexpected),
                        )
                except Exception as exc:
                    if _is_main():
                        logging.warning("[ActionForcingDMD] %s load failed: %s", key, exc)
                    if strict:
                        raise

    def _load_real_score_with_v14_lora(self, args, device) -> None:
        v14_ckpt_path = getattr(args, "v14_teacher_checkpoint", None)
        if not v14_ckpt_path:
            if _is_main():
                logging.warning(
                    "[ActionForcingDMD] No v14_teacher_checkpoint provided; real_score "
                    "is base Wan weights (DMD grads will degenerate).",
                )
            return
        if not os.path.exists(v14_ckpt_path):
            raise FileNotFoundError(
                f"v14_teacher_checkpoint not found: {v14_ckpt_path}"
            )
        if not _HAS_PEFT:
            raise RuntimeError(
                "peft required to load v14 LoRA into real_score but is not installed."
            )

        lora_cfg = getattr(args, "v14_lora", None) or {
            "rank": 256, "alpha": 256, "dropout": 0.0,
        }
        rank = int(
            lora_cfg.get("rank", 256) if isinstance(lora_cfg, dict)
            else getattr(lora_cfg, "rank", 256)
        )
        alpha = float(
            lora_cfg.get("alpha", rank) if isinstance(lora_cfg, dict)
            else getattr(lora_cfg, "alpha", rank)
        )
        dropout = float(
            lora_cfg.get("dropout", 0.0) if isinstance(lora_cfg, dict)
            else getattr(lora_cfg, "dropout", 0.0)
        )

        target_modules = self._collect_target_modules(self.real_score.model)
        if not target_modules:
            target_modules = ["q", "k", "v", "o"]

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
        )
        if _is_main():
            logging.info(
                "[ActionForcingDMD] Applying v14 LoRA (rank=%d alpha=%s drop=%s) to "
                "real_score over %d linear modules",
                rank, alpha, dropout, len(target_modules),
            )
        peft_model = peft.get_peft_model(self.real_score.model, lora_config)
        ckpt = torch.load(v14_ckpt_path, map_location="cpu")
        lora_sd = ckpt.get("lora")
        if lora_sd is None:
            raise KeyError(
                f"v14 checkpoint missing 'lora' key: have {list(ckpt.keys())}"
            )
        try:
            set_peft_model_state_dict(peft_model, lora_sd)
        except Exception:
            from peft import get_peft_model_state_dict
            current_sd = get_peft_model_state_dict(peft_model)
            matched = 0
            for key in current_sd:
                if key in lora_sd and current_sd[key].shape == lora_sd[key].shape:
                    current_sd[key] = lora_sd[key]
                    matched += 1
            set_peft_model_state_dict(peft_model, current_sd)
            if _is_main():
                logging.info(
                    "[ActionForcingDMD] real_score LoRA cross-load matched %d/%d",
                    matched, len(current_sd),
                )

        merged = peft_model.merge_and_unload()
        try:
            from peft.tuners.lora import LoraLayer
            for _, m in merged.named_modules():
                if isinstance(m, LoraLayer):
                    raise RuntimeError("LoRA layers remain after merge_and_unload on real_score.")
        except Exception:
            pass
        self.real_score.model = merged.to(device=device, dtype=self.dtype)

    @staticmethod
    def _collect_target_modules(model) -> list:
        target_modules = set()
        for module_name, module in model.named_modules():
            if module.__class__.__name__ in {"WanAttentionBlock", "CausalWanAttentionBlock"}:
                for full_name, submodule in module.named_modules(prefix=module_name):
                    if isinstance(submodule, nn.Linear):
                        target_modules.add(full_name)
        return sorted(target_modules)

    def _mirror_generator_into_fake_score(self) -> None:
        """DMD2 upgrade: fake_score starts as a clone of the generator's
        ODE-init state. Without this, the DMD subtraction kicks off as
        ``real - random`` and the student gets a noisy unhelpful gradient
        for the first ~hundreds of steps."""
        try:
            gen_sd = self.generator.model.state_dict()
            missing, unexpected = self.fake_score.model.load_state_dict(
                gen_sd, strict=False,
            )
            if _is_main() and DEBUG:
                logging.info(
                    "[ActionForcingDMD] fake_score mirror: missing=%d unexpected=%d",
                    len(missing), len(unexpected),
                )
        except Exception as e:
            if _is_main():
                logging.warning("[ActionForcingDMD] fake_score mirror failed: %s", e)

    # ------------------------------------------------------------------
    # Action conditioning helpers
    # ------------------------------------------------------------------
    def build_action_conditional(
        self,
        prompt_embeds: torch.Tensor,
        gt_actions: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Build (conditional_dict, unconditional_dict) for the full
        ``num_training_frames`` window.

        Args:
            prompt_embeds: ``[B, L, C_txt]``.
            gt_actions: ``[B, F, action_dim]`` per-frame ground-truth
                actions (no decay, exact GT — Phase-1 Action-Forcing contract).

        Both Stream A (AdaLN modulation) and Stream B (per-frame action
        tokens) are populated for every frame. ``unconditional`` zeroes
        prompt+actions; CF parity for the *real_score* always consumes
        it (uncond forward runs on every iter regardless of
        ``real_guidance_scale``), the *fake_score* uncond is gated on
        ``fake_guidance_scale != 0``. We use zero embeds rather than a
        Chinese negative prompt (CF's choice) — this is an intentional
        delta, see the comparison table.
        """
        if self.action_projection is None:
            raise RuntimeError(
                "action_projection missing; required for ActionForcingDMD action "
                "conditioning."
            )
        if self.action_token_projection is None:
            raise RuntimeError(
                "action_token_projection missing; required for Stream-B "
                "conditioning."
            )

        device = prompt_embeds.device
        dtype = prompt_embeds.dtype
        gt_actions = gt_actions.to(device=device, dtype=dtype)
        modulation = self.action_projection(
            gt_actions, num_frames=gt_actions.shape[1],
        )
        action_tokens = self.action_token_projection(gt_actions)

        conditional = {
            "prompt_embeds": prompt_embeds,
            "_action_modulation": modulation,
            "_action_tokens": action_tokens,
        }
        unconditional = {
            "prompt_embeds": torch.zeros_like(prompt_embeds),
            "_action_modulation": torch.zeros_like(modulation),
            "_action_tokens": torch.zeros_like(action_tokens),
        }
        return conditional, unconditional

    # ------------------------------------------------------------------
    # CF-style _run_generator. Backward-simulates the input from noise
    # and returns the student's predicted 21-frame video.
    # ------------------------------------------------------------------
    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        clean_latent=None,
        initial_latent: Optional[torch.Tensor] = None,
        enable_mae_extension: bool = False,
        seed_latents: Optional[torch.Tensor] = None,
        requires_grad: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[int], Optional[int]]:
        """Match CF's ``SelfForcingModel._run_generator`` behavior, plus
        CF-style long-rollout support and MAE-driven extension support:
          1. Sample fresh noise of shape ``[B, F, C, H, W]`` with
             F = ``rollout_frames`` (= ``num_training_frames`` by
             default; can be larger for long-rollout / "warmup the
             cache without grad" mode).
          2. Run ``inference_pipeline.inference_with_trajectory``. The
             pipeline's gradient gate
             ``start_gradient_frame_index = num_output_frames - num_max_frames``
             ensures only the LAST ``num_training_frames`` rolled
             frames carry gradient when ``rollout_frames >
             num_training_frames``. ``gt_latents=clean_latent`` enables
             the pipeline's MAE-extension loop when
             ``enable_mae_extension`` is True (caller-controlled —
             only the generator step opts in; the critic step skips
             extensions to avoid duplicating the metric collection).
          3. Slice the returned pred to its LAST ``num_training_frames``
             frames so the bidirectional scorer (whose seq_len is
             sized to ``num_training_frames``) sees only the
             gradient-active window. When ``rollout_frames ==
             num_training_frames`` this slice is a no-op.
          4. CF-style first-chunk boundary mask (long-rollout mode
             only): when ``rollout_frames > num_training_frames`` the
             leading ``num_frame_per_block`` frames of the
             ``num_training_frames`` scoring window receive context
             from the warmup KV cache (frames generated under
             ``no_grad`` to "warm" the cache). CF treats those
             leading scoring frames as a BOUNDARY: the gradient is
             masked off there so the student is not penalised for
             whatever transition pattern emerges between warmup-
             generated frames and the first grad-active block. We
             follow CF's mask logic byte-for-byte
             (``Causal-Forcing/long_video/model/base.py:169-177``)
             but DELIBERATELY SKIP CF's decode→re-encode round-trip
             (CF replaces the boundary frame with a fresh VAE-encoded
             pixel latent; we leave it as the student's raw latent
             since the gradient mask makes the boundary frame's
             content irrelevant to the loss anyway, and the round-
             trip is expensive — a full VAE decode + encode per
             training step). When ``rollout_frames ==
             num_training_frames`` (classic and pure-extension modes)
             the mask is None and every scoring frame backprops.
          5. Surface the pipeline's
             ``_last_extension_metrics`` dict (which is populated even
             when extensions are disabled, as long as ``gt_latents`` is
             provided — gives ``baseline_last_chunk_mae`` /
             ``last_chunk_mae`` always, ``mae_extension_count`` only
             on the generator step).

        Note on extensions: MAE extensions append no_grad chunks PAST
        the baseline rollout; they never enter the scoring window.
        So whether extensions fire or not is independent of the
        boundary mask — the mask is gated solely on
        ``rollout_frames > num_training_frames``.
        """
        assert getattr(self.args, "backward_simulation", True), (
            "ActionForcingDMD requires backward_simulation=True"
        )
        if self.args.i2v:
            raise RuntimeError(
                "ActionForcingDMD does not support i2v in Phase-1 Action-Forcing."
            )

        noise_shape = list(image_or_video_shape)
        # Phase-1 Action-Forcing: total rolled length must equal
        # ``self.rollout_frames + self.dmd_clean_x_anchor_frames``.
        # The +``anchor_frames`` is the leading chunk that anchors
        # batch-1's ``clean_x_self`` window — once per ride at the
        # FRONT, NOT once per batch.
        anchor_frames = int(self.dmd_clean_x_anchor_frames)
        expected_rollout = self.rollout_frames + anchor_frames
        if noise_shape[1] != expected_rollout:
            raise RuntimeError(
                f"Phase-1 Action-Forcing expects rollout of "
                f"{expected_rollout} latent frames (= rollout_frames="
                f"{self.rollout_frames} + anchor_frames={anchor_frames}); "
                f"got noise_shape[1]={noise_shape[1]}. Trainer must size "
                f"the noise tensor to {expected_rollout} frames so the "
                f"first {anchor_frames} frames anchor clean_x_self."
            )

        if self.inference_pipeline is None:
            raise RuntimeError(
                "ActionForcingDMD.inference_pipeline is None — the trainer must set "
                "this BEFORE calling generator_loss / critic_loss. Set "
                "`model.inference_pipeline = ActionForcingTrainingPipeline(...)` in "
                "trainer._build_pipeline."
            )

        noise = torch.randn(
            noise_shape, device=self.device, dtype=self.dtype,
        )

        pred_image_or_video, denoised_timestep_from, denoised_timestep_to = (
            self.inference_pipeline.inference_with_trajectory(
                noise=noise,
                clean_image_or_video=None,
                gt_latents=clean_latent,
                enable_mae_extension=enable_mae_extension,
                seed_latents=seed_latents,
                requires_grad=requires_grad,
                **conditional_dict,
            )
        )

        # Slice the pred into:
        #   pred_for_scoring   = pred[:, -num_training_frames:]   (last N
        #                         frames = noisy_x scoring window)
        #   clean_x_self_anchor = pred[:, :num_training_frames]   (first
        #                         N frames = clean_x_self anchor;
        #                         purely student-rolled, no seed mix)
        # The two slices overlap by N - anchor_frames frames in absolute
        # rolled-frame indexing but live at different RoPE positions in
        # the bidir scorer joint sequence (clean at [0,N), noisy at
        # [shift,N+shift)).
        #
        # Gradient mask:
        #   * ALWAYS mask the LAST ``num_frame_per_block`` frames of the
        #     scoring window. In v14's TF joint sequence the noisy half's
        #     last block lives at RoPE positions [N, N+shift) — these
        #     positions have NO clean-half counterpart (clean half is at
        #     [0, N)), so the bidir scorer cannot condition them on clean
        #     context. Empirically (``_diag_p1/test_dmd_inference.py``
        #     3-rollout sweep): F_real_GT shows ~2× the mid-window MAE
        #     in the last 3 frames of every 21-frame batch — an
        #     unavoidable end-of-window boundary the DMD gradient should
        #     not learn from.
        #   * In long-rollout mode (``rollout_frames > num_training_frames``)
        #     ALSO mask the first ``num_frame_per_block`` frames as the
        #     CF-parity boundary between cache-warmup and grad-active
        #     frames (CF: long_video/model/base.py:169-177). We
        #     deliberately skip CF's decode→re-encode round-trip.
        block = int(self.num_frame_per_block)
        full_rollout = pred_image_or_video
        clean_x_self_anchor = full_rollout[:, : self.num_training_frames].contiguous()
        pred_for_scoring = full_rollout[:, -self.num_training_frames:].contiguous()
        # ``_dmd_score_grad_mask`` returns a freshly-allocated tensor,
        # so we can mutate it in place safely.
        gradient_mask = self._dmd_score_grad_mask(pred_for_scoring.shape, pred_for_scoring.device)
        # Long-rollout warmup boundary (only when rollout > N): also
        # mask the FIRST ``block`` frames as the CF-parity
        # cache-warmup → grad-active boundary. Intentional asymmetry
        # vs the critic step: the critic's bidir forward has no
        # cache-warmup→grad-active transition, so it doesn't need
        # this leading mask. The trailing last-chunk mask remains
        # uniform across gen and critic.
        if self.rollout_frames > self.num_training_frames:
            gradient_mask[:, :block] = False

        return (
            pred_for_scoring.to(self.dtype),
            clean_x_self_anchor.to(self.dtype),
            gradient_mask,
            denoised_timestep_from,
            denoised_timestep_to,
        )

    # ------------------------------------------------------------------
    # Canonical "drop the last chunk" mask for every DMD-score loss.
    # ------------------------------------------------------------------
    def _dmd_score_grad_mask(
        self, shape: Tuple[int, ...], device: torch.device
    ) -> torch.Tensor:
        """Return a ``[B, F, C, H, W]`` bool mask with the last
        ``num_frame_per_block`` frames set to False. EVERY DMD-score
        loss in this module routes through this helper so the last-
        chunk boundary stays masked uniformly across:

          * gen-step DMD MSE (``compute_distribution_matching_loss``)
          * critic-step denoising MSE (``critic_loss``)
          * streaming gen step (``compute_generator_loss_streaming``)
          * streaming critic step (``compute_critic_loss_streaming``)

        Boundary rationale: in v14's TF joint sequence the noisy half
        lives at RoPE [shift, N+shift). The last ``shift`` noisy
        positions [N, N+shift) have no clean-half counterpart (clean
        half is at [0, N)) — the bidir scorer can't condition them on
        clean context, producing structurally OOD predictions. Verified
        empirically (``_diag_p1/test_dmd_inference.py`` 3-rollout sweep,
        F_real_GT MAE ~2× the mid-window value at the last 3 frames of
        every 21-frame batch). Hardcoded — NOT config-settable — because
        the boundary is a property of the v14 LoRA's training contract,
        not of any tunable choice.
        """
        block = int(self.num_frame_per_block)
        if len(shape) < 2:
            raise ValueError(
                f"_dmd_score_grad_mask expects shape with B and F dims, "
                f"got {shape}"
            )
        if shape[1] <= block:
            raise ValueError(
                f"scoring window has F={shape[1]} <= block={block}; the "
                f"last-chunk boundary mask would zero ALL frames. Bump "
                "num_training_frames or shrink num_frame_per_block."
            )
        mask = torch.ones(shape, dtype=torch.bool, device=device)
        mask[:, -block:] = False
        return mask

    # ------------------------------------------------------------------
    # CF-parity DMD core
    # ------------------------------------------------------------------
    def _compute_kl_grad(
        self,
        noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        normalization: bool = True,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        clean_x_real: Optional[torch.Tensor] = None,
        aug_t_real: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Eq. (7) of the DMD paper, byte-for-byte CF parity.

        Both real_score and fake_score score THE SAME ``noisy_image_or_
        video`` (derived from the student's pred via add_noise). CFG
        branches are gated on the relevant guidance scale being
        non-zero (saves a forward when off).

        When ``clean_x`` / ``aug_t`` are provided (dmd_context=True,
        v14 teacher-forcing parity), they are forwarded to
        ``fake_score`` — the underlying causal Wan model's ``clean_x``
        branch handles building the teacher-forcing block mask +
        clean-half RoPE offset (gated by ``model.context_shift``).

        When ``clean_x_real`` / ``aug_t_real`` are also provided
        (``dmd_context='GT'`` mode), real_score gets a separately
        prepared view: the GT-version of the same time positions,
        lightly noised at ``self.clean_x_aug_t``. When they are
        ``None`` (``dmd_context='self'`` mode or critic step), real_
        score falls back to the same ``(clean_x, aug_t)`` as fake_
        score (matched conditioning). The ``noisy_image_or_video``
        driving the score is always the SAME for both scorers, so
        the DMD subtraction stays well defined; only the
        conditioning ``clean_x`` differs.
        """
        tf_kwargs_fake: Dict[str, Any] = {}
        tf_kwargs_real: Dict[str, Any] = {}
        if clean_x is not None:
            tf_kwargs_fake["clean_x"] = clean_x
            tf_kwargs_fake["aug_t"] = aug_t
            tf_kwargs_real["clean_x"] = (
                clean_x_real if clean_x_real is not None else clean_x
            )
            tf_kwargs_real["aug_t"] = (
                aug_t_real if aug_t_real is not None else aug_t
            )

        # Step 1: fake score
        _, pred_fake_image_cond = self.fake_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep,
            **tf_kwargs_fake,
        )
        if self.fake_guidance_scale != 0.0:
            _, pred_fake_image_uncond = self.fake_score(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=unconditional_dict,
                timestep=timestep,
                **tf_kwargs_fake,
            )
            pred_fake_image = pred_fake_image_cond + (
                pred_fake_image_cond - pred_fake_image_uncond
            ) * self.fake_guidance_scale
        else:
            pred_fake_image = pred_fake_image_cond

        # Step 2: real score (CF parity — ALWAYS run both cond + uncond
        # forwards, no gate on real_guidance_scale). With scale=0 the
        # math collapses to ``pred_real_image_cond`` exactly, so the
        # extra forward is "free" semantically; the cost is one
        # additional real_score forward per training iter relative to a
        # gated implementation. CF code path matches this byte-for-byte
        # at ``Causal-Forcing/model/dmd.py:98-112``.
        _, pred_real_image_cond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep,
            **tf_kwargs_real,
        )
        _, pred_real_image_uncond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=unconditional_dict,
            timestep=timestep,
            **tf_kwargs_real,
        )
        pred_real_image = pred_real_image_cond + (
            pred_real_image_cond - pred_real_image_uncond
        ) * self.real_guidance_scale

        # Step 3: DMD grad = (fake - real). CF normalizes by
        # |x0 - real|.mean() (eq. 8). Match exactly.
        grad = pred_fake_image - pred_real_image
        if normalization:
            p_real = estimated_clean_image_or_video - pred_real_image
            normalizer = torch.abs(p_real).mean(
                dim=[1, 2, 3, 4], keepdim=True,
            )
            grad = grad / normalizer.clamp_min(1e-6)
        grad = torch.nan_to_num(grad)

        log_dict: Dict[str, Any] = {
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.detach(),
        }
        # Return the detached teacher prediction alongside the
        # gradient so the caller can run teacher-freeze detection
        # (per-frame MAE vs GT) without duplicating the real_score
        # forward.
        return grad, pred_real_image.detach(), log_dict

    def _sample_dmd_timestep(
        self,
        batch_size: int,
        num_frame: int,
        denoised_timestep_from: Optional[int],
        denoised_timestep_to: Optional[int],
        device: torch.device,
    ) -> torch.Tensor:
        """Sample DMD timestep with CF's ``ts_schedule`` clamp + shift."""
        min_timestep = (
            denoised_timestep_to
            if (self.ts_schedule and denoised_timestep_to is not None)
            else self.min_score_timestep
        )
        max_timestep = (
            denoised_timestep_from
            if (self.ts_schedule_max and denoised_timestep_from is not None)
            else self.num_train_timestep
        )
        # Same-step-across-frames: one timestep per sample, broadcast.
        timestep = self._get_timestep(
            min_timestep,
            max_timestep,
            batch_size,
            num_frame,
            self.num_frame_per_block,
            uniform_timestep=True,
        )
        if self.timestep_shift > 1:
            t_norm = timestep.float() / 1000.0
            t_shifted = self.timestep_shift * t_norm / (
                1 + (self.timestep_shift - 1) * t_norm
            )
            timestep = (t_shifted * 1000.0).long()
        timestep = timestep.clamp(self.min_step, self.max_step)
        return timestep

    def compute_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: Optional[int] = 0,
        denoised_timestep_to: Optional[int] = 0,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        clean_x_real: Optional[torch.Tensor] = None,
        aug_t_real: Optional[torch.Tensor] = None,
        gt_target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """CF-parity DMD loss (eq. 7).

        Inputs:
          image_or_video: student's pred_x0 [B, F, C, H, W] (graph-on).
          conditional_dict / unconditional_dict: full-21-frame action +
            prompt conditioning (built by ``build_action_conditional``).
          denoised_timestep_from/to: sampled by the pipeline; clamps the
            DMD timestep distribution to the rolling-step's denoise rung.
          clean_x: when ``dmd_context=True``, GT clean-half latents
            (``[B, num_training_frames, C, H, W]``) preceding the
            student's noisy half by ``dmd_context_clean_frames``.
            Forwarded to ``fake_score`` (and to ``real_score`` when
            ``clean_x_real`` is None).
          aug_t: low/zero noise level applied to ``clean_x`` (dtype
            int64, shape ``[B, num_training_frames]``). For the v14
            parity case the scorers see ``clean_x`` as PURE clean
            (``aug_t=0``), matching v14's training contract.
          clean_x_real / aug_t_real: optional separate view for
            real_score — used in ``dmd_context='GT'`` mode where
            real sees lightly-noised GT latents (via
            ``scheduler.add_noise`` at ``self.clean_x_aug_t``) while
            fake keeps the unnoised self-view. ``None`` in
            ``dmd_context='self'`` mode and on the critic step;
            both scorers then share ``(clean_x, aug_t)``.
        """
        original_latent = image_or_video
        batch_size, num_frame = image_or_video.shape[:2]

        with torch.no_grad():
            timestep = self._sample_dmd_timestep(
                batch_size, num_frame,
                denoised_timestep_from, denoised_timestep_to,
                device=image_or_video.device,
            )
            noise = torch.randn_like(image_or_video)
            noisy_latent = self.scheduler.add_noise(
                image_or_video.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1),
            ).detach().unflatten(0, (batch_size, num_frame))

            grad, pred_real_image_detached, dmd_log_dict = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent,
                estimated_clean_image_or_video=original_latent,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_x=clean_x,
                aug_t=aug_t,
                clean_x_real=clean_x_real,
                aug_t_real=aug_t_real,
            )

        # ``teacher_freeze_detect``: per-frame outlier rejection on the
        # teacher prediction. Compute per-frame MAE between
        # ``pred_real_image_detached`` and ``gt_target`` (= GT video
        # frames at the noisy_x positions). Frames whose MAE exceeds
        # ``threshold * median(MAE)`` are flagged as teacher freezes
        # and added to ``gradient_mask`` so DMD skips them. Robust to
        # the all-frames-similar case (median ≈ MAE on every frame ⇒
        # no frame trips the threshold ⇒ no false positives).
        if bool(getattr(self, "teacher_freeze_detect_enabled", False)):
            mode = str(getattr(self, "teacher_freeze_mode", "mae"))
            freeze_mask_bf = None  # [B, F] bool, set below

            if mode == "mae" and gt_target is not None:
                with torch.no_grad():
                    gt_t = gt_target.to(
                        dtype=pred_real_image_detached.dtype,
                        device=pred_real_image_detached.device,
                    )
                    if gt_t.shape != pred_real_image_detached.shape:
                        raise RuntimeError(
                            "teacher_freeze_detect: gt_target shape "
                            f"{tuple(gt_t.shape)} does not match "
                            f"pred_real_image shape "
                            f"{tuple(pred_real_image_detached.shape)}."
                        )
                    # Per-frame MAE [B, F].
                    per_frame_mae = (
                        pred_real_image_detached.float() - gt_t.float()
                    ).abs().mean(dim=[2, 3, 4])
                    # Median across frames (per-batch, [B, 1]) — robust
                    # to the freeze frame itself being the outlier.
                    median_mae = per_frame_mae.median(dim=1, keepdim=True).values
                    freeze_mask_bf = per_frame_mae > (
                        self.teacher_freeze_threshold * median_mae
                    )  # [B, F]
                    # Telemetry.
                    dmd_log_dict["teacher_freeze_max_mae"] = (
                        per_frame_mae.max().detach()
                    )
                    dmd_log_dict["teacher_freeze_median_mae"] = (
                        median_mae.mean().detach()
                    )

            elif (
                mode == "action"
                and self._action_teacher_fn is not None
                and gt_target is not None
            ):
                with torch.no_grad():
                    npb = int(self.num_frame_per_block)
                    F = pred_real_image_detached.shape[1]
                    if F % npb != 0:
                        raise RuntimeError(
                            f"teacher_freeze_detect (mode=action): F={F} "
                            f"not divisible by num_frame_per_block={npb}."
                        )
                    n_slots = F // npb
                    # Decision metric: cos(z_real, z_gt). The teacher is
                    # "wrong" iff its predicted action direction is opposite
                    # to the GT action direction. Comparing to the STUDENT
                    # would conflate "teacher wrong" with "student wrong",
                    # masking gradient on slots where the teacher is
                    # actually right and the student is the one drifting —
                    # exactly the slots we want DMD to keep training on.
                    # Cosine sim is magnitude-invariant, so a teacher that
                    # gets the magnitude wrong but the direction right
                    # stays in the high-cos region and is NOT masked.
                    # ``z_student`` is also computed for telemetry only.
                    z_real = self._action_teacher_fn(pred_real_image_detached)
                    z_gt = self._action_teacher_fn(gt_target)
                    z_student = self._action_teacher_fn(original_latent.detach())
                    if z_real is None or z_gt is None:
                        # Teacher temporarily unavailable (cotracker /
                        # ss_vae failure). Skip — the same iter's aux
                        # loss path also no-ops on None.
                        dmd_log_dict["teacher_freeze_unavailable"] = 1.0
                    else:
                        if z_real.shape != z_gt.shape:
                            raise RuntimeError(
                                f"teacher_freeze_detect (mode=action): "
                                f"z_real shape {tuple(z_real.shape)} != "
                                f"z_gt shape {tuple(z_gt.shape)}."
                            )
                        # Per-slot cosine similarity = the freeze metric.
                        cos_RG = torch.nn.functional.cosine_similarity(
                            z_real.float(), z_gt.float(), dim=-1,
                        )  # [B, n_slots]
                        slot_freeze = cos_RG < self.teacher_freeze_action_threshold
                        freeze_mask_bf = slot_freeze.repeat_interleave(npb, dim=1)
                        # Telemetry — surface both cos(R,GT) (the
                        # decision metric) and cos(R,student) (context).
                        dmd_log_dict["teacher_freeze_action_cos_RG_min"] = (
                            cos_RG.min().detach()
                        )
                        dmd_log_dict["teacher_freeze_action_cos_RG_mean"] = (
                            cos_RG.mean().detach()
                        )
                        dmd_log_dict["teacher_freeze_action_cos_RG_median"] = (
                            cos_RG.median().detach()
                        )
                        if z_student is not None:
                            cos_RS = torch.nn.functional.cosine_similarity(
                                z_real.float(), z_student.float(), dim=-1,
                            )
                            dmd_log_dict["teacher_freeze_action_cos_RS_mean"] = (
                                cos_RS.mean().detach()
                            )

            # AND-merge the freeze mask into the running gradient_mask.
            # No-op when the mode-specific branch couldn't compute a mask.
            if freeze_mask_bf is not None and freeze_mask_bf.any():
                if gradient_mask is None:
                    gradient_mask = torch.ones(
                        original_latent.shape, dtype=torch.bool,
                        device=original_latent.device,
                    )
                else:
                    gradient_mask = gradient_mask.clone()
                freeze_full = freeze_mask_bf.view(
                    *freeze_mask_bf.shape, 1, 1, 1,
                ).expand_as(gradient_mask)
                gradient_mask[freeze_full] = False
            # Always-on count / rate telemetry (when feature is enabled
            # and the mask was computed).
            if freeze_mask_bf is not None:
                dmd_log_dict["teacher_freeze_count"] = (
                    freeze_mask_bf.float().sum().detach()
                )
                dmd_log_dict["teacher_freeze_rate"] = (
                    freeze_mask_bf.float().mean().detach()
                )

        # FUNDAMENTAL: every DMD-score loss must arrive here with a
        # non-None ``gradient_mask`` (= at least the canonical last-chunk
        # boundary mask from ``_dmd_score_grad_mask``). The loss API
        # doesn't silently fall back to an unmasked MSE — that path was
        # removed deliberately so a future regression in any caller
        # cannot bypass the boundary mask.
        if gradient_mask is None:
            raise RuntimeError(
                "compute_distribution_matching_loss requires "
                "``gradient_mask`` (every caller must build it from "
                "``self._dmd_score_grad_mask`` so the structurally-OOD "
                "last-chunk boundary stays masked uniformly)."
            )
        if not gradient_mask.any():
            # All-False mask → MSE on empty tensor is NaN. Surface a
            # clean zero loss + telemetry so the trainer can detect /
            # gate on it instead of silently propagating NaN through
            # the optimizer step. The zero-loss is connected to the
            # autograd graph through ``original_latent`` so
            # ``.backward()`` succeeds (gradient is zero on every
            # parameter) — critical because the trainer calls
            # ``loss.backward()`` unconditionally.
            dmd_log_dict["dmd_empty_mask"] = 1.0
            zero_loss = (original_latent.double() * 0.0).sum()
            return zero_loss, dmd_log_dict
        dmd_loss = 0.5 * F.mse_loss(
            original_latent.double()[gradient_mask],
            (original_latent.double() - grad.double()).detach()[gradient_mask],
            reduction="mean",
        )
        return dmd_loss, dmd_log_dict

    # ------------------------------------------------------------------
    # Public losses (CF interface)
    # ------------------------------------------------------------------
    def _build_dmd_context_kwargs(
        self,
        clean_x_self: Optional[torch.Tensor],
        clean_x_GT: Optional[torch.Tensor],
        clean_conditional_dict: Optional[dict],
        clean_unconditional_dict: Optional[dict],
        cond_for_scoring: dict,
        uncond_for_scoring: dict,
        device: torch.device,
        dtype: torch.dtype,
        build_real_view: bool = False,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        dict,
        dict,
    ]:
        """Prepare per-scorer ``clean_x`` / ``aug_t`` and merge the clean
        action streams into the scoring cond dicts. Returns:
            (clean_x_fake, aug_t_fake,
             clean_x_real, aug_t_real,
             new_cond_for_scoring, new_uncond_for_scoring)

        ``clean_x_fake`` / ``aug_t_fake`` are ALWAYS the "self" view
        (= ``clean_x_self``, ``aug_t=0``) regardless of ``self.dmd_context``
        mode. fake_score thus sees consistent conditioning across modes.
        ``clean_x_self`` is the 21-frame view shifted back by one chunk
        from the noisy_x window, assembled by the caller from
        ``[seed_last_chunk, sdn[:18]]`` (batch 1) or
        ``sdn[(i-1)*21-3 : i*21-3]`` (batch ≥ 2).

        ``clean_x_real`` / ``aug_t_real`` are mode-dependent:
          * ``"self"``: same as fake (no noise; both scorers see the
            same view; (fake-real) gradient on matched conditioning).
          * ``"GT"``: ``scheduler.add_noise(clean_x_GT, n,
            clean_x_aug_t)`` — real_score sees the GT version of the
            same time positions, lightly noised at ``self.clean_x_aug_t``
            for symmetry-breaking. ``clean_x_GT`` is required in this
            mode; trainer assembles it from
            ``ride[s+6+(i-1)*21 : s+6+i*21]`` per batch.

        When ``build_real_view=False`` (critic step — only trains fake)
        the real-side outputs are ``None`` so callers can skip the
        noise pass entirely.

        The cond dicts are SHALLOW-COPIED (never mutate the caller's
        dict) and have ``_action_modulation_clean`` /
        ``_action_tokens_clean`` injected from ``clean_*_dict``. The
        action streams correspond to the time positions covered by
        ``clean_x_self`` / ``clean_x_GT`` (which are the same — only the
        latent values differ between self and GT views), so a SINGLE
        clean cond dict is correct for both fake and real.
        """
        if clean_x_self is None:
            raise RuntimeError(
                "_build_dmd_context_kwargs requires clean_x_self (the "
                "21-frame self-view assembled by the trainer)."
            )
        if clean_x_self.shape[1] != self.num_training_frames:
            raise RuntimeError(
                f"clean_x_self.shape[1]={clean_x_self.shape[1]} must "
                f"equal num_training_frames={self.num_training_frames}."
            )
        if self.dmd_context == "GT" and build_real_view:
            # ``clean_x_GT`` is only consumed for the real_score side
            # in "GT" mode. The critic step (``build_real_view=False``)
            # only trains fake_score, which always uses the self-view
            # regardless of dmd_context — clean_x_GT is irrelevant
            # there, so we don't require it.
            if clean_x_GT is None:
                raise RuntimeError(
                    "dmd_context='GT' requires clean_x_GT (the trainer "
                    "must assemble the shifted GT clean window and pass "
                    "it here for the gen-step real_score view)."
                )
            if clean_x_GT.shape[1] != self.num_training_frames:
                raise RuntimeError(
                    f"clean_x_GT.shape[1]={clean_x_GT.shape[1]} must "
                    f"equal num_training_frames={self.num_training_frames}."
                )

        sc_clean_x = clean_x_self.to(dtype=dtype, device=device)
        sc_aug_t = torch.zeros(
            (sc_clean_x.shape[0], self.num_training_frames),
            device=device, dtype=torch.long,
        )

        sc_clean_x_real: Optional[torch.Tensor] = None
        sc_aug_t_real: Optional[torch.Tensor] = None
        if build_real_view and self.dmd_context == "GT":
            sc_aug_t_real = torch.full(
                (sc_clean_x.shape[0], self.num_training_frames),
                fill_value=int(self.clean_x_aug_t),
                device=device, dtype=torch.long,
            )
            gt_view = clean_x_GT.to(dtype=dtype, device=device)
            real_noise = torch.randn_like(gt_view)
            sc_clean_x_real = self.scheduler.add_noise(
                gt_view.flatten(0, 1),
                real_noise.flatten(0, 1),
                sc_aug_t_real.flatten(0, 1),
            ).unflatten(0, gt_view.shape[:2]).to(dtype=dtype)
        # In "self" mode (or when build_real_view=False) we leave
        # sc_clean_x_real/sc_aug_t_real as None; _compute_kl_grad's
        # fallback uses (sc_clean_x, sc_aug_t) for real_score.

        new_cond = dict(cond_for_scoring)
        new_uncond = dict(uncond_for_scoring)
        if clean_conditional_dict is not None:
            am_c = clean_conditional_dict.get("_action_modulation", None)
            at_c = clean_conditional_dict.get("_action_tokens", None)
            if am_c is not None:
                new_cond["_action_modulation_clean"] = am_c
            if at_c is not None:
                new_cond["_action_tokens_clean"] = at_c
        if clean_unconditional_dict is not None:
            am_u = clean_unconditional_dict.get("_action_modulation", None)
            at_u = clean_unconditional_dict.get("_action_tokens", None)
            if am_u is not None:
                new_uncond["_action_modulation_clean"] = am_u
            if at_u is not None:
                new_uncond["_action_tokens_clean"] = at_u
        return sc_clean_x, sc_aug_t, sc_clean_x_real, sc_aug_t_real, new_cond, new_uncond

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: Optional[torch.Tensor] = None,
        initial_latent: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        seed_latents: Optional[torch.Tensor] = None,
        clean_x_self: Optional[torch.Tensor] = None,
        clean_x_GT: Optional[torch.Tensor] = None,
        clean_conditional_dict: Optional[dict] = None,
        clean_unconditional_dict: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """CF-parity generator loss: roll the student, compute DMD.

        ``image_or_video_shape`` is the BASELINE ROLLOUT shape
        (``[B, rollout_frames, C, H, W]``) — extensions are sampled
        inline by the pipeline, not from this noise. ``_run_generator``
        rolls the pipeline (potentially extending past ``rollout_frames``
        via the MAE-extension loop, all under no_grad), then slices the
        pred to the LAST ``num_training_frames`` of the BASELINE.
        ``conditional_dict`` / ``unconditional_dict`` per-frame streams
        may be longer than ``rollout_frames`` (sized to cover the max
        possible extended length); we slice them positionally to the
        baseline scoring window
        ``[rollout_frames - num_training_frames : rollout_frames]``
        before passing to the bidirectional scorer.

        Generator step opts INTO the MAE extension (passes
        ``enable_mae_extension=True``) and surfaces the per-call
        metrics from ``inference_pipeline._last_extension_metrics``
        into the log dict.

        ``clean_x_self`` / ``clean_x_GT`` / ``clean_conditional_dict`` /
        ``clean_unconditional_dict`` are the teacher-forcing inputs.
        The trainer assembles ``clean_x_self`` from the cumulative
        student rollout (= 21 frames shifted back one chunk from the
        noisy_x window) and, in ``dmd_context='GT'`` mode, also
        ``clean_x_GT`` from the ride at the same shifted positions.
        ``clean_conditional_dict`` carries action streams covering
        the same 21-frame clean window. The scorer attends over
        the joint [clean, noisy] sequence with the noisy half RoPE-
        shifted by ``cf`` frames to recreate v14's time-alignment.

        When ``return_aux=True`` we additionally return an ``aux``
        dict carrying the rolled student prediction and the rung-
        exit timesteps, so a caller (the trainer) can plumb them
        into auxiliary loss paths (action critic / teacher z-
        guidance) without re-rolling the student. ``pred_image``
        is graph-carrying — its gradients are exactly what the
        DMD loss already feeds back through; the trainer should
        ``.detach()`` before passing into the critic-update branch
        and use the live tensor only for the gen-side guidance
        loss.
        """
        # ``image_or_video_shape[1]`` = total rolled length =
        # ``self.rollout_frames + anchor_frames`` (= scoring window
        # length + leading clean_x_self anchor). The scoring slicer
        # below uses this full length and picks the LAST
        # ``num_training_frames`` of the rollout half.
        rollout_frames = int(image_or_video_shape[1])
        (
            pred_image,
            clean_x_self_anchor,
            gradient_mask,
            denoised_timestep_from,
            denoised_timestep_to,
        ) = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            clean_latent=clean_latent,
            initial_latent=initial_latent,
            enable_mae_extension=True,
            seed_latents=seed_latents,
        )
        scoring_frames = self.num_training_frames

        # Trainer feeds the pipeline a ``conditional_dict`` covering
        # ``seed_frames + rollout_frames`` frames (seed first, then
        # rollout) so the seed-prefill loop indexes by absolute
        # ``current_start_frame``. The scorer wants the LAST
        # ``num_training_frames`` of the ROLLOUT half — slice with the
        # ``seed_frames`` offset to skip the seed actions.
        seed_frames = int(seed_latents.shape[1]) if seed_latents is not None else 0
        cond_for_scoring = _slice_baseline_scoring_window(
            conditional_dict,
            rollout_frames=rollout_frames,
            num_training_frames=scoring_frames,
            seed_frames=seed_frames,
        )
        uncond_for_scoring = _slice_baseline_scoring_window(
            unconditional_dict,
            rollout_frames=rollout_frames,
            num_training_frames=scoring_frames,
            seed_frames=seed_frames,
        )

        # ``clean_x_self`` defaults to ``clean_x_self_anchor`` returned
        # by ``_run_generator`` — the FIRST ``num_training_frames`` of
        # the (anchor + scoring) rollout. Purely student-rolled, no
        # seed mix. Per-ride anchor: the +``anchor_frames`` chunk at
        # the front of the rollout exists exclusively so this slice
        # is well-defined for batch 1 without dipping into the seed.
        # Caller can override (rare — only multi-batch streaming
        # consumers might want a custom slice).
        if clean_x_self is None:
            clean_x_self = clean_x_self_anchor

        (
            sc_clean_x,
            sc_aug_t,
            sc_clean_x_real,
            sc_aug_t_real,
            cond_for_scoring,
            uncond_for_scoring,
        ) = self._build_dmd_context_kwargs(
            clean_x_self=clean_x_self,
            clean_x_GT=clean_x_GT,
            clean_conditional_dict=clean_conditional_dict,
            clean_unconditional_dict=clean_unconditional_dict,
            cond_for_scoring=cond_for_scoring,
            uncond_for_scoring=uncond_for_scoring,
            device=pred_image.device,
            dtype=pred_image.dtype,
            build_real_view=True,
        )

        # Teacher-freeze gt_target: GT video at the noisy_x positions
        # (= the same time positions ``pred_image`` was rolled at). In
        # the legacy single-batch path, ``clean_latent`` covers the
        # FULL ride window (seed + rollout); the rollout's gt is
        # ``clean_latent[seed_frames : seed_frames + scoring_frames]``.
        # When teacher-freeze detection is off this is an O(slice)
        # no-op.
        gt_target = None
        if (
            bool(getattr(self, "teacher_freeze_detect_enabled", False))
            and clean_latent is not None
            and clean_latent.shape[1] >= seed_frames + scoring_frames
        ):
            gt_target = clean_latent[
                :, seed_frames : seed_frames + scoring_frames
            ]

        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=cond_for_scoring,
            unconditional_dict=uncond_for_scoring,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
            clean_x=sc_clean_x,
            aug_t=sc_aug_t,
            clean_x_real=sc_clean_x_real,
            aug_t_real=sc_aug_t_real,
            gt_target=gt_target,
        )
        dmd_loss = dmd_loss * self.dmd_loss_weight

        # Surface MAE-extension metrics (may be NaN-marked if
        # gt_latents was None or shorter than the baseline rollout).
        ext_metrics = getattr(
            self.inference_pipeline, "_last_extension_metrics", None,
        ) or {}
        for k, v in ext_metrics.items():
            dmd_log_dict[k] = v
        if return_aux:
            aux: Dict[str, Any] = {
                "pred_image": pred_image,
                "gradient_mask": gradient_mask,
                "denoised_timestep_from": denoised_timestep_from,
                "denoised_timestep_to": denoised_timestep_to,
                "scoring_frames": scoring_frames,
                "rollout_frames": rollout_frames,
            }
            return dmd_loss, dmd_log_dict, aux
        return dmd_loss, dmd_log_dict

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: Optional[torch.Tensor] = None,
        initial_latent: Optional[torch.Tensor] = None,
        seed_latents: Optional[torch.Tensor] = None,
        clean_x_self: Optional[torch.Tensor] = None,
        clean_x_GT: Optional[torch.Tensor] = None,
        clean_conditional_dict: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """CF-parity critic (fake_score) loss: train fake_score to denoise
        the student's generated video.

        Same long-rollout / extension handling as ``generator_loss``:
        roll the student through the full ``rollout_frames`` (no_grad
        here), slice pred and cond dict to the BASELINE last
        ``num_training_frames``, and run the fake_score denoise loss
        on that slice. The MAE extension is DISABLED on the critic
        step (``enable_mae_extension=False``) to avoid duplicating the
        metric collection the generator step already does — extensions
        are pure no_grad forwards on the same student weights, so
        running them again here would just produce the same numbers
        at extra compute cost.

        ``clean_x_self`` / ``clean_conditional_dict``: fake_score's
        clean_x is ALWAYS the self-view (same as in the gen step),
        so fake learns to denoise the student's pred under matched
        teacher-forcing conditioning. ``clean_x_GT`` is unused on
        the critic step (real_score isn't trained here); accepted
        for signature compatibility with ``generator_loss``.
        """
        rollout_frames = int(image_or_video_shape[1])
        # LongLive parity: detach any K/V tensors held in the
        # persistent caches before the critic rollout so any autograd
        # graph from a prior gen-step rollout doesn't chain through
        # the critic's no_grad forwards. No-op when the caches are
        # freshly initialised (Phase-A) or when no K/V required grad
        # to begin with.
        if self.inference_pipeline is not None:
            self.inference_pipeline._clear_cache_gradients()
        with torch.no_grad():
            (
                generated_image,
                clean_x_self_anchor,
                _gradient_mask,
                denoised_timestep_from,
                denoised_timestep_to,
            ) = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                clean_latent=clean_latent,
                initial_latent=initial_latent,
                enable_mae_extension=False,
                seed_latents=seed_latents,
                requires_grad=False,
            )

        # ``generated_image`` is sliced by ``_run_generator`` to the
        # LAST ``num_training_frames`` of the rollout (= noisy_x scoring
        # window). ``clean_x_self_anchor`` is the FIRST
        # ``num_training_frames`` (= clean_x_self anchor; pure student-
        # rolled, no seed mix). ``conditional_dict`` covers seed+rollout
        # when seed_latents is provided; ``seed_frames`` shifts the
        # scoring slice past the seed prefix.
        scoring_shape = list(generated_image.shape)
        seed_frames = int(seed_latents.shape[1]) if seed_latents is not None else 0
        cond_for_scoring = _slice_baseline_scoring_window(
            conditional_dict,
            rollout_frames=rollout_frames,
            num_training_frames=scoring_shape[1],
            seed_frames=seed_frames,
        )

        # ``clean_x_self`` defaults to the anchor slice — same path as
        # generator_loss. fake_score's clean_x is the unnoised self-view
        # so critic step matches gen step's TF conditioning.
        if clean_x_self is None:
            clean_x_self = clean_x_self_anchor

        # dmd_context: build clean_x / aug_t for fake_score and merge
        # ``_action_modulation_clean`` / ``_action_tokens_clean`` into
        # cond_for_scoring (uncond not needed here — fake_score is
        # trained without CFG on the critic step). ``build_real_view=
        # False`` because critic_loss only trains fake_score (which
        # always sees the unnoised self-view, regardless of mode);
        # skipping the real-side noised view saves one
        # ``scheduler.add_noise`` call per critic step.
        (
            sc_clean_x,
            sc_aug_t,
            _unused_clean_x_real,
            _unused_aug_t_real,
            cond_for_scoring,
            _unused_uncond,
        ) = self._build_dmd_context_kwargs(
            clean_x_self=clean_x_self,
            clean_x_GT=clean_x_GT,
            clean_conditional_dict=clean_conditional_dict,
            clean_unconditional_dict=None,
            cond_for_scoring=cond_for_scoring,
            uncond_for_scoring={},
            device=generated_image.device,
            dtype=generated_image.dtype,
            build_real_view=False,
        )

        critic_timestep = self._sample_dmd_timestep(
            batch_size=scoring_shape[0],
            num_frame=scoring_shape[1],
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
            device=generated_image.device,
        )

        critic_noise = torch.randn_like(generated_image)
        noisy_generated_image = self.scheduler.add_noise(
            generated_image.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1),
        ).unflatten(0, scoring_shape[:2])

        tf_kwargs: Dict[str, Any] = {}
        if sc_clean_x is not None:
            tf_kwargs["clean_x"] = sc_clean_x
            tf_kwargs["aug_t"] = sc_aug_t

        _, pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_generated_image,
            conditional_dict=cond_for_scoring,
            timestep=critic_timestep,
            **tf_kwargs,
        )

        if self.args.denoising_loss_type == "flow":
            from utils.wan_wrapper import WanDiffusionWrapper
            flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1),
            )
            pred_fake_noise = None
        else:
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1),
            ).unflatten(0, scoring_shape[:2])

        # Critic step honours the SAME last-chunk boundary mask as the
        # gen step's DMD MSE — fake_score's prediction at noisy RoPE
        # positions [N, N+shift) is structurally OOD (no clean-half
        # counterpart in v14's TF joint sequence), so don't train
        # fake_score against those frames.
        critic_grad_mask = self._dmd_score_grad_mask(
            generated_image.shape, generated_image.device,
        ).flatten(0, 1)
        denoising_loss = self.denoising_loss_func(
            x=generated_image.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred,
            gradient_mask=critic_grad_mask,
        )

        # Surface the rollout's last-chunk MAE (computed by the pipeline
        # in ``_last_extension_metrics``) so the trainer can collapse-
        # gate the critic step the same way it does the gen step.
        critic_log: Dict[str, Any] = {
            "critic_timestep": critic_timestep.detach(),
        }
        ext_metrics = getattr(
            self.inference_pipeline, "_last_extension_metrics", None,
        ) or {}
        for k, v in ext_metrics.items():
            critic_log[k] = v
        return denoising_loss, critic_log

    # ==================================================================
    # STREAMING MODE — LongLive-style persistent-state DMD
    # ==================================================================
    # The training trainer uses these methods when ``streaming_mode=True``
    # in the YAML. They keep one rolling sequence open across many
    # iters, advancing by 18-21 new frames per iter (random, broadcast
    # from rank 0) while reusing the persistent KV/crossattn caches
    # initialised at ``setup_sequence``. Iter k's gradient flows only
    # through the new frames (per-frame ``gradient_mask``); overlap
    # frames carry no gradient. When the sequence is exhausted (or the
    # collapse gate fires), the trainer calls ``setup_sequence`` again
    # with a fresh ride.
    #
    # State dict layout:
    #   current_length      : int  - frames generated so far in this sequence
    #   max_length          : int  - cap for can_generate_more
    #   chunk_size          : int  - DMD scoring window (=21)
    #   shift               : int  - clean/noisy shift (= num_frame_per_block)
    #   seed_latents        : tensor [B, cf, C, H, W]  - GT prefill
    #   ride_latents_window : tensor [B, cf+max_length, ...]  - clean_x_GT source
    #   ride_actions_window : tensor [B, cf+max_length, A]
    #   prompt_embeds       : tensor [B, T, D]
    #   conditional_dict    : full-window action streams (pipeline indexes by absolute frame)
    #   unconditional_dict  : full-window unconditional streams
    #   clean_conditional_dict / clean_unconditional_dict : per-frame
    #         clean-half streams (built ONCE over the WHOLE sequence;
    #         the per-iter ``compute_*_loss_streaming`` slices to the
    #         shifted window before passing to the scorer).
    #   previous_chunk      : tensor [B, chunk_size, ...]  - last full_chunk
    #         (for clean_x assembly on iter k≥2)
    # ------------------------------------------------------------------
    def setup_sequence(
        self,
        seed_latents: torch.Tensor,
        ride_latents_window: torch.Tensor,
        ride_actions_window: torch.Tensor,
        prompt_embeds: torch.Tensor,
        max_length: int,
    ) -> None:
        """Open a new streaming sequence: re-initialise the persistent
        KV/crossattn caches, prefill the seed (cf=9 GT frames at t=0
        with the existing context-noise commit), build conditional
        dicts over the FULL sequence window, and initialise the
        per-iter state.

        Caller (trainer) is responsible for picking the random offset
        ``s`` and slicing the ride; this method just consumes the
        sliced tensors.

        Args:
            seed_latents:        ``[B, cf, C, H, W]`` GT KV prefill.
            ride_latents_window: ``[B, cf+rollout, C, H, W]`` clean_x_GT source.
                                 ``cf`` leading frames are the seed.
            ride_actions_window: ``[B, cf+rollout, A]`` per-frame action streams.
            prompt_embeds:       ``[B, T, D]`` cross-attn prompt.
            max_length:          frames generated cap (``can_generate_more``).
        """
        if self.inference_pipeline is None:
            raise RuntimeError(
                "setup_sequence requires inference_pipeline; trainer must "
                "set ``model.inference_pipeline = ActionForcingTrainingPipeline(...)``"
                " before opening a sequence."
            )
        cf = int(seed_latents.shape[1])
        if cf != self.dmd_context_clean_frames:
            raise ValueError(
                f"seed_latents has {cf} frames; expected "
                f"dmd_context_clean_frames={self.dmd_context_clean_frames}."
            )
        if ride_latents_window.shape[1] < cf:
            raise ValueError(
                f"ride_latents_window has only {ride_latents_window.shape[1]} "
                f"frames; need >= cf={cf}."
            )

        device = seed_latents.device
        dtype = seed_latents.dtype
        batch_size = seed_latents.shape[0]
        npb = self.num_frame_per_block
        pipe = self.inference_pipeline

        # Cap max_length to a multiple of npb so chunk advances stay clean.
        if max_length % npb != 0:
            max_length = (max_length // npb) * npb

        # Clean-half action streams cover ride[s+cf-shift :
        # s+cf+rollout-shift] in absolute frames; relative to
        # ``ride_*_window`` (which starts at ride[s]), this is
        # ``ride_*_window[cf-shift : cf-shift+max_length]``. We store
        # only the RAW action tensors on state and rebuild the
        # cond/uncond dicts (= action_projection + action_token_projection
        # forward) per-iter inside ``_streaming_build_cond_dicts`` so the
        # autograd graph for those projections has a single backward
        # lifecycle per gen step (matching the legacy single-iter
        # path's ``build_action_conditional`` per ``_fwdbwd_one_step``).
        # Stashing the dicts on state would freeze the graph, and the
        # 2nd backward in the same sequence would fail with "Trying to
        # backward through the graph a second time".
        #
        # NB: even with the +npb anchor (rolled below) clean_actions_window
        # still starts at cf - shift in absolute ride coords because
        # iter k≥2's clean_x is shifted back from its noisy_x by ``shift``
        # frames in the cumulative-sdn coord system; the anchor consumes
        # the leading ``shift`` frames of clean_actions_window for iter 1.
        clean_actions_window = ride_actions_window[
            :, cf - npb : cf - npb + max_length
        ]

        # Reset + initialise persistent caches.
        pipe.reset_cache_state()
        pipe._initialize_kv_cache(
            batch_size=batch_size, dtype=dtype, device=device,
        )
        pipe._initialize_crossattn_cache(
            batch_size=batch_size, dtype=dtype, device=device,
        )

        # Seed prefill: cf GT frames at t=0 + context-noise commit
        # (mirrors the seed-prefill block inside inference_with_trajectory).
        # All seed-prefill forwards run inside ``with torch.no_grad():``,
        # so the temporary cond_dict's grad_fn is orphaned and GC'd
        # once setup_sequence returns. action_projection.weight.grad
        # is unaffected — these no_grad forwards never backward through it.
        with torch.no_grad():
            seed_cond_dict, _ = self.build_action_conditional(
                prompt_embeds=prompt_embeds,
                gt_actions=ride_actions_window,
            )
        num_seed_chunks = cf // npb
        current_start_frame = 0
        for sc in range(num_seed_chunks):
            seed_chunk = seed_latents[:, sc * npb : (sc + 1) * npb]
            seed_t = torch.zeros(
                [batch_size, npb], device=device, dtype=torch.int64,
            )
            seed_block_cond = _slice_per_frame_streams(
                seed_cond_dict, frame_start=current_start_frame, frame_count=npb,
            )
            with torch.no_grad():
                pipe.generator(
                    noisy_image_or_video=seed_chunk,
                    conditional_dict=seed_block_cond,
                    timestep=seed_t,
                    kv_cache=pipe.kv_cache1,
                    crossattn_cache=pipe.crossattn_cache,
                    current_start=current_start_frame * pipe.frame_seq_length,
                )
            # Context-noise commit (LongLive parity: keeps seed K/V at
            # the same noise level as rollout chunks').
            ctx_t = torch.full_like(seed_t, pipe.context_noise)
            seed_ctx_in = self.scheduler.add_noise(
                seed_chunk.flatten(0, 1),
                torch.randn_like(seed_chunk.flatten(0, 1)),
                ctx_t.flatten(0, 1),
            ).unflatten(0, seed_chunk.shape[:2])
            with torch.no_grad():
                pipe.generator(
                    noisy_image_or_video=seed_ctx_in,
                    conditional_dict=seed_block_cond,
                    timestep=ctx_t,
                    kv_cache=pipe.kv_cache1,
                    crossattn_cache=pipe.crossattn_cache,
                    current_start=current_start_frame * pipe.frame_seq_length,
                )
            current_start_frame += npb
        del seed_cond_dict  # release the no_grad cond dict before iter 1

        # +npb leading-anchor chunk: roll a single ``shift``-frame
        # rollout chunk under no_grad so iter 1's ``clean_x_self``
        # window has purely-student-rolled content (no seed/student
        # quality discontinuity that used to OOD the bidir scorer in
        # legacy mode). Mirrors ``_run_generator``'s +npb anchor at the
        # FRONT of the rollout (legacy single-batch contract). The
        # anchor's KV/cross-attn caches advance to ``cf + npb``;
        # ``current_length`` initialises to ``npb`` so iter 1's
        # ``noisy_start_sdn = npb`` (= the absolute position right
        # after the anchor) and the geometry of clean_lo / clean_x_GT
        # slicing collapses to the uniform iter k≥2 formula.
        anchor_noise = torch.randn(
            [batch_size, npb, *seed_latents.shape[2:]],
            device=device, dtype=dtype,
        )
        with torch.no_grad():
            anchor_full_cond, _ = self.build_action_conditional(
                prompt_embeds=prompt_embeds,
                gt_actions=ride_actions_window,
            )
            anchor_chunk, _, _ = pipe.generate_chunk_with_cache(
                noise=anchor_noise,
                current_start_frame=cf,
                requires_grad=False,
                prefer_cache_pred_in_output=False,
                gt_latents=None,  # no MAE on the anchor
                **anchor_full_cond,
            )
        del anchor_full_cond
        anchor_chunk = anchor_chunk.detach()

        self.streaming_state = {
            "current_length": int(npb),  # anchor counts toward the cumulative sdn
            "max_length": int(max_length),
            "chunk_size": int(self.streaming_chunk_size),
            "shift": int(npb),
            "cf": int(cf),
            "seed_latents": seed_latents,
            "ride_latents_window": ride_latents_window,
            "ride_actions_window": ride_actions_window,
            "clean_actions_window": clean_actions_window,
            "prompt_embeds": prompt_embeds,
            "previous_chunk": None,  # last full_chunk (chunk_size frames)
            "abs_frame_after_seed": cf,  # absolute pipeline frame index after seed prefill (anchor adds npb on top)
            "anchor_chunk": anchor_chunk,  # [B, npb, C, H, W] — iter 1's clean_x_self anchor
        }

    def _streaming_build_cond_dicts(
        self,
    ) -> Tuple[dict, dict, dict, dict]:
        """Rebuild the noisy-half + clean-half conditional / unconditional
        dicts FRESH per call by running ``action_projection`` /
        ``action_token_projection`` on the persistent raw action
        tensors stored in ``streaming_state``. The dicts must NOT be
        cached on state across iters: ``build_action_conditional``
        produces grad_fn-attached tensors, and reusing them across
        backward passes raises "Trying to backward through the graph
        a second time" (the projection's saved-tensor buffer was
        freed by the first backward) or, after an in-place optimizer
        step, "one of the variables needed for gradient computation
        has been modified by an inplace operation". Rebuilding gives
        each gen-step iter its own clean grad_fn lifecycle, matching
        the legacy ``_fwdbwd_one_step`` semantics.
        """
        s = self.streaming_state
        if s is None:
            raise RuntimeError(
                "_streaming_build_cond_dicts called with no open sequence"
            )
        cond_dict, uncond_dict = self.build_action_conditional(
            prompt_embeds=s["prompt_embeds"],
            gt_actions=s["ride_actions_window"],
        )
        clean_cond_dict, clean_uncond_dict = self.build_action_conditional(
            prompt_embeds=s["prompt_embeds"],
            gt_actions=s["clean_actions_window"],
        )
        return cond_dict, uncond_dict, clean_cond_dict, clean_uncond_dict

    def can_generate_more(self) -> bool:
        """Returns True iff the streaming sequence is open AND has room
        to advance by at least ``min_new_frame`` more frames within
        ``max_length``."""
        if self.streaming_state is None:
            return False
        s = self.streaming_state
        return (s["current_length"] + self.streaming_min_new_frame) <= s["max_length"]

    def reset_streaming_state(self) -> None:
        """Tear down the open sequence (typically when collapse gate
        fires or ``can_generate_more`` returns False)."""
        self.streaming_state = None
        if self.inference_pipeline is not None:
            self.inference_pipeline.reset_cache_state()

    def _streaming_pick_new_frames(
        self, device: torch.device,
    ) -> int:
        """Pick the number of new frames to generate this iter, broadcast
        from rank 0 for DDP lockstep. Always a multiple of npb in
        ``[min_new_frame, chunk_size]``, capped to remaining sequence
        room."""
        s = self.streaming_state
        npb = s["shift"]
        chunk_size = s["chunk_size"]
        room = s["max_length"] - s["current_length"]
        max_new = min(room, chunk_size)
        # Snap to multiple of npb.
        max_new = (max_new // npb) * npb
        min_new = (self.streaming_min_new_frame // npb) * npb
        if max_new <= min_new:
            return max(npb, max_new)
        # Candidate values: min_new, min_new+npb, ..., max_new.
        candidates = list(range(min_new, max_new + 1, npb))
        if not candidates:
            return max(npb, max_new)
        if dist.is_initialized():
            if dist.get_rank() == 0:
                import random as _py_random
                idx = _py_random.randint(0, len(candidates) - 1)
            else:
                idx = 0
            t = torch.tensor([idx], device=device, dtype=torch.long)
            dist.broadcast(t, src=0)
            idx = int(t.item())
        else:
            import random as _py_random
            idx = _py_random.randint(0, len(candidates) - 1)
        return int(candidates[idx])

    def generate_next_chunk(
        self, requires_grad: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Advance the open sequence by ``new_frames`` (∈ [min_new_frame,
        chunk_size], multiple of npb). Build a chunk_size-length
        ``full_chunk`` for DMD scoring = ``[previous_chunk[-overlap:],
        new_frames]`` (overlap = chunk_size - new_frames; 0 on first
        iter or when new_frames == chunk_size). Returns
        ``(full_chunk, info)`` where ``info`` carries the
        ``gradient_mask`` (True only on new frames), the per-chunk MAE,
        and the metadata DMD scoring needs.
        """
        if self.streaming_state is None:
            raise RuntimeError("generate_next_chunk called with no open sequence")
        if not self.can_generate_more():
            raise RuntimeError("generate_next_chunk: sequence exhausted")

        s = self.streaming_state
        device = s["seed_latents"].device
        dtype = s["seed_latents"].dtype
        batch_size = s["seed_latents"].shape[0]
        npb = s["shift"]
        chunk_size = s["chunk_size"]
        cf = s["cf"]
        pipe = self.inference_pipeline

        new_frames = self._streaming_pick_new_frames(device)
        if s["previous_chunk"] is None:
            # First iter of the sequence: roll a full chunk_size, no overlap.
            new_frames = chunk_size
            overlap = 0
        else:
            overlap = chunk_size - new_frames

        # Absolute frame position in the pipeline cache where the new
        # frames will be written.
        abs_frame_start = s["abs_frame_after_seed"] + s["current_length"]

        noise_chunk = torch.randn(
            [batch_size, new_frames, *s["seed_latents"].shape[2:]],
            device=device, dtype=dtype,
        )
        # GT slice (for per-chunk MAE) covering the new-frame range.
        gt_slice_lo = cf + s["current_length"]
        gt_slice_hi = gt_slice_lo + new_frames
        if gt_slice_hi <= s["ride_latents_window"].shape[1]:
            gt_chunk = s["ride_latents_window"][:, gt_slice_lo:gt_slice_hi]
        else:
            gt_chunk = None

        # Rebuild cond/uncond/clean_cond/clean_uncond FRESH for this
        # iter (see ``_streaming_build_cond_dicts`` for the why) and
        # stash on ``info`` so ``compute_*_loss_streaming`` reuses the
        # same dicts the rollout used — single grad_fn lifecycle per
        # iter, matching legacy ``_fwdbwd_one_step`` semantics.
        cond_dict, uncond_dict, clean_cond_dict, clean_uncond_dict = (
            self._streaming_build_cond_dicts()
        )

        new_chunk, denoised_t_from, denoised_t_to = pipe.generate_chunk_with_cache(
            noise=noise_chunk,
            current_start_frame=abs_frame_start,
            requires_grad=requires_grad,
            prefer_cache_pred_in_output=False,
            gt_latents=gt_chunk,
            **cond_dict,
        )

        # Snapshot OLD previous_chunk BEFORE we overwrite — clean_x_self
        # assembly on iter k≥2 needs the iter (k-1) chunk.
        prev_chunk_for_clean = s["previous_chunk"]

        # Build chunk_size-length full_chunk via overlap.
        if overlap > 0:
            full_chunk = torch.cat(
                [s["previous_chunk"][:, -overlap:], new_chunk], dim=1,
            )
        else:
            full_chunk = new_chunk

        # gradient_mask: True only on new frames within the full_chunk.
        gradient_mask = torch.zeros_like(full_chunk, dtype=torch.bool)
        gradient_mask[:, overlap : overlap + new_frames] = True

        # Save full_chunk as previous_chunk (detached) for the NEXT iter.
        s["previous_chunk"] = full_chunk.detach()
        s["current_length"] += new_frames

        info: Dict[str, Any] = {
            "denoised_timestep_from": denoised_t_from,
            "denoised_timestep_to": denoised_t_to,
            "new_frames": int(new_frames),
            "overlap": int(overlap),
            "current_length": int(s["current_length"]),
            "max_length": int(s["max_length"]),
            "abs_frame_start": int(abs_frame_start),
            "gradient_mask": gradient_mask,
            "prev_chunk_for_clean": prev_chunk_for_clean,
            # Per-iter cond dicts (NOT cached on state — see
            # ``_streaming_build_cond_dicts`` docstring). Pass through
            # to ``compute_*_loss_streaming`` so the scorer's slice
            # reuses the rollout's cond_dict and the backward graph
            # hits action_projection / action_token_projection once
            # per iter via a single (fresh-this-iter) grad_fn chain.
            "conditional_dict": cond_dict,
            "unconditional_dict": uncond_dict,
            "clean_conditional_dict": clean_cond_dict,
            "clean_unconditional_dict": clean_uncond_dict,
        }
        # Surface MAE from pipeline.
        ext = getattr(pipe, "_last_extension_metrics", None) or {}
        for k, v in ext.items():
            info[k] = v
        return full_chunk, info

    def _streaming_build_clean_x_self(
        self, full_chunk: torch.Tensor, info: Dict[str, Any],
    ) -> torch.Tensor:
        """Assemble the 21-frame ``clean_x_self`` view = noisy window
        shifted back by ``shift`` frames in cumulative-sdn coords.

        For iter 1 (no previous_chunk available before this call):
            clean_x = [anchor_chunk, full_chunk[:N-shift]]
            (anchor_chunk = the +npb chunk rolled in setup_sequence —
            entirely student-rolled, NOT seed-mixed. Replaces the legacy
            ``[seed[-shift:], full_chunk[:N-shift]]`` seed-mix that put
            GT-quality frames in the leading positions of clean_x.)
        For iter k≥2 (previous_chunk available):
            clean_x = [prev_chunk[chunk_size-overlap-shift :
                                   chunk_size-overlap],
                       full_chunk[:N-shift]]
        Both halves are entirely model-predicted at every iter — no
        seed/student quality discontinuity anywhere.
        """
        s = self.streaming_state
        shift = s["shift"]
        chunk_size = s["chunk_size"]
        # Read previous_chunk BEFORE generate_next_chunk overwrote it.
        # The caller (compute_*_loss_streaming) is invoked AFTER
        # generate_next_chunk so previous_chunk is now full_chunk;
        # we don't have access to the iter k-1 chunk anymore.
        # Workaround: stash the pre-overwrite copy in info.
        prev_for_clean = info.get("prev_chunk_for_clean")
        overlap = info["overlap"]
        if prev_for_clean is None:
            # Iter 1: anchor chunk (rolled in setup_sequence) replaces
            # the legacy seed-tail. Anchor has length ``shift``.
            anchor = s["anchor_chunk"].to(
                dtype=full_chunk.dtype, device=full_chunk.device,
            )
            clean_x = torch.cat(
                [anchor, full_chunk[:, : chunk_size - shift]], dim=1,
            )
        else:
            head_lo = chunk_size - overlap - shift
            head_hi = chunk_size - overlap
            clean_head = prev_for_clean[:, head_lo:head_hi].to(
                dtype=full_chunk.dtype, device=full_chunk.device,
            )
            clean_x = torch.cat(
                [clean_head, full_chunk[:, : chunk_size - shift]], dim=1,
            )
        return clean_x

    def _streaming_build_clean_x_GT(
        self, info: Dict[str, Any],
    ) -> torch.Tensor:
        """GT clean-half slice = ``ride_latents_window[abs_start - shift :
        abs_start - shift + chunk_size]`` where ``abs_start`` is the
        cumulative-sdn position of the noisy half's first frame
        (= ``cf + (current_length - new_frames - overlap)``).

        Equivalently in cumulative coords with abs_frame_after_seed=cf:
            noisy_start_in_sdn = current_length - new_frames - overlap
            clean_start_in_ride_coords = cf + noisy_start_in_sdn - shift
        """
        s = self.streaming_state
        shift = s["shift"]
        chunk_size = s["chunk_size"]
        cf = s["cf"]
        # In cumulative sdn coords, the noisy half's first frame is at
        # position ``current_length - new_frames - overlap``.
        noisy_start_sdn = s["current_length"] - info["new_frames"] - info["overlap"]
        clean_start_in_ride = cf + noisy_start_sdn - shift
        clean_end_in_ride = clean_start_in_ride + chunk_size
        return s["ride_latents_window"][:, clean_start_in_ride:clean_end_in_ride]

    def _streaming_clean_cond_slice(
        self, info: Dict[str, Any],
    ) -> Tuple[dict, dict]:
        """Slice the per-iter clean_conditional_dict / clean_unconditional_dict
        (built fresh by ``generate_next_chunk`` and stashed on
        ``info``) to the iter's 21-frame window. The clean cond dict
        covers ``ride_actions[cf-shift : cf-shift+max_length]``;
        iter k's clean window is at cumulative sdn positions
        ``[noisy_start_sdn - shift, noisy_start_sdn - shift + 21)``,
        which maps to clean_cond positions
        ``[noisy_start_sdn, noisy_start_sdn + 21)`` (frame 0 of the
        clean dict = ride frame cf-shift).
        """
        s = self.streaming_state
        chunk_size = s["chunk_size"]
        noisy_start_sdn = s["current_length"] - info["new_frames"] - info["overlap"]
        clean_lo = noisy_start_sdn  # = (cf + noisy_start_sdn - shift) - (cf - shift)
        clean_hi = clean_lo + chunk_size
        clean_cond = _slice_per_frame_streams(
            info["clean_conditional_dict"],
            frame_start=clean_lo, frame_count=chunk_size,
        )
        clean_uncond = _slice_per_frame_streams(
            info["clean_unconditional_dict"],
            frame_start=clean_lo, frame_count=chunk_size,
        )
        return clean_cond, clean_uncond

    def _streaming_noisy_cond_slice(
        self, info: Dict[str, Any],
    ) -> Tuple[dict, dict]:
        """Slice the per-iter conditional_dict / unconditional_dict
        (built fresh by ``generate_next_chunk`` and stashed on
        ``info``) to the noisy-half 21-frame window. The full cond
        dict covers ride[s : s+cf+max_length] (sdn frame i lives at
        conditional_dict frame ``cf + i``). Noisy half is sdn[start :
        start+21] where start = current_length - new_frames - overlap.
        """
        s = self.streaming_state
        chunk_size = s["chunk_size"]
        cf = s["cf"]
        noisy_start_sdn = s["current_length"] - info["new_frames"] - info["overlap"]
        cond = _slice_per_frame_streams(
            info["conditional_dict"],
            frame_start=cf + noisy_start_sdn, frame_count=chunk_size,
        )
        uncond = _slice_per_frame_streams(
            info["unconditional_dict"],
            frame_start=cf + noisy_start_sdn, frame_count=chunk_size,
        )
        return cond, uncond

    def compute_generator_loss_streaming(
        self,
        chunk: torch.Tensor,
        info: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """DMD generator loss on the streaming chunk. Reuses
        ``compute_distribution_matching_loss`` + ``_build_dmd_context_kwargs``
        — same scorer math as the per-iter ``generator_loss``, just
        with the per-frame ``gradient_mask`` from streaming and the
        clean_x_self / clean_x_GT slices assembled from the persistent
        sequence state.
        """
        s = self.streaming_state
        # ``info["gradient_mask"]`` is the per-iter overlap-chunk mask
        # (True only on the new frames). It MUST be a tensor — if a
        # future change makes it None, the silent fallback would
        # quietly drop the new-frames mask and leave only the last-
        # chunk mask, retraining the overlap region every iter. Same
        # contract as ``compute_distribution_matching_loss``: raise
        # rather than fall back.
        per_iter_mask = info.get("gradient_mask")
        if per_iter_mask is None:
            raise RuntimeError(
                "compute_generator_loss_streaming: info['gradient_mask'] "
                "must be a tensor (built by ``_streaming_generate_chunk_with_grad``)."
            )
        # AND-in the canonical last-chunk boundary mask. Same mask
        # used in the legacy gen step — zeroes out the structurally-
        # OOD positions [N, N+shift) of the noisy half.
        last_chunk_mask = self._dmd_score_grad_mask(
            chunk.shape, chunk.device,
        )
        gradient_mask_eff = per_iter_mask & last_chunk_mask

        clean_x_self = self._streaming_build_clean_x_self(chunk, info)
        clean_x_GT = (
            self._streaming_build_clean_x_GT(info) if self.dmd_context == "GT" else None
        )
        cond_for_scoring, uncond_for_scoring = self._streaming_noisy_cond_slice(info)
        clean_cond, clean_uncond = self._streaming_clean_cond_slice(info)

        (
            sc_clean_x, sc_aug_t,
            sc_clean_x_real, sc_aug_t_real,
            cond_for_scoring, uncond_for_scoring,
        ) = self._build_dmd_context_kwargs(
            clean_x_self=clean_x_self,
            clean_x_GT=clean_x_GT,
            clean_conditional_dict=clean_cond,
            clean_unconditional_dict=clean_uncond,
            cond_for_scoring=cond_for_scoring,
            uncond_for_scoring=uncond_for_scoring,
            device=chunk.device, dtype=chunk.dtype,
            build_real_view=True,
        )

        # Teacher-freeze gt_target for streaming: GT video at the
        # chunk's noisy_x positions (= ride_latents_window indices
        # [cf + noisy_start_sdn : cf + noisy_start_sdn + chunk_size]).
        # ``noisy_start_sdn = current_length - new_frames - overlap``
        # already accounts for the iter's overlap region. No-op when
        # the feature is off.
        gt_target = None
        if bool(getattr(self, "teacher_freeze_detect_enabled", False)):
            cf_state = int(s["cf"])
            chunk_size_state = int(s["chunk_size"])
            noisy_start_sdn = int(
                s["current_length"]
                - info["new_frames"]
                - info["overlap"]
            )
            ride_window = s["ride_latents_window"]
            chunk_lo = cf_state + noisy_start_sdn
            chunk_hi = chunk_lo + chunk_size_state
            if ride_window.shape[1] >= chunk_hi:
                gt_target = ride_window[:, chunk_lo:chunk_hi]

        dmd_loss, dmd_log = self.compute_distribution_matching_loss(
            image_or_video=chunk,
            conditional_dict=cond_for_scoring,
            unconditional_dict=uncond_for_scoring,
            gradient_mask=gradient_mask_eff,
            denoised_timestep_from=info.get("denoised_timestep_from"),
            denoised_timestep_to=info.get("denoised_timestep_to"),
            clean_x=sc_clean_x, aug_t=sc_aug_t,
            gt_target=gt_target,
            clean_x_real=sc_clean_x_real, aug_t_real=sc_aug_t_real,
        )
        dmd_loss = dmd_loss * self.dmd_loss_weight

        for k in ("baseline_last_chunk_mae", "baseline_avg_rollout_mae", "last_chunk_mae", "mae_extension_count"):
            if k in info:
                dmd_log[k] = info[k]
        dmd_log["streaming_new_frames"] = float(info["new_frames"])
        dmd_log["streaming_current_length"] = float(info["current_length"])
        return dmd_loss, dmd_log

    def compute_critic_loss_streaming(
        self,
        chunk: torch.Tensor,
        info: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Streaming critic step. Detaches chunk + caches before the
        fake_score forward, masks the denoising loss with the
        per-frame ``gradient_mask`` so only new frames train fake_score.
        """
        if self.inference_pipeline is not None:
            self.inference_pipeline._clear_cache_gradients()
        if chunk.requires_grad:
            chunk = chunk.detach()

        s = self.streaming_state
        # Strict contract: per-iter mask must be present. See the
        # matching comment in compute_generator_loss_streaming — silent
        # fallback would retrain the overlap region every iter.
        per_iter_mask = info.get("gradient_mask")
        if per_iter_mask is None:
            raise RuntimeError(
                "compute_critic_loss_streaming: info['gradient_mask'] "
                "must be a tensor (built by ``_streaming_generate_chunk_with_grad``)."
            )
        # AND-in the canonical last-chunk boundary mask.
        last_chunk_mask = self._dmd_score_grad_mask(
            chunk.shape, chunk.device,
        )
        gradient_mask = per_iter_mask & last_chunk_mask
        cond_for_scoring, _uncond = self._streaming_noisy_cond_slice(info)

        # Build clean_x for the fake_score's TF half — same self-view
        # as the gen step (fake never sees GT clean_x).
        clean_x_self = self._streaming_build_clean_x_self(chunk, info)
        clean_cond, _ = self._streaming_clean_cond_slice(info)
        (
            sc_clean_x, sc_aug_t,
            _r1, _r2, cond_for_scoring, _u,
        ) = self._build_dmd_context_kwargs(
            clean_x_self=clean_x_self,
            clean_x_GT=None,
            clean_conditional_dict=clean_cond,
            clean_unconditional_dict=None,
            cond_for_scoring=cond_for_scoring,
            uncond_for_scoring={},
            device=chunk.device, dtype=chunk.dtype,
            build_real_view=False,
        )

        denoised_timestep_from = info.get("denoised_timestep_from")
        denoised_timestep_to = info.get("denoised_timestep_to")
        critic_timestep = self._sample_dmd_timestep(
            batch_size=chunk.shape[0], num_frame=chunk.shape[1],
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
            device=chunk.device,
        )

        critic_noise = torch.randn_like(chunk)
        noisy_chunk = self.scheduler.add_noise(
            chunk.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1),
        ).unflatten(0, chunk.shape[:2])

        tf_kwargs: Dict[str, Any] = {}
        if sc_clean_x is not None:
            tf_kwargs["clean_x"] = sc_clean_x
            tf_kwargs["aug_t"] = sc_aug_t

        _, pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_chunk,
            conditional_dict=cond_for_scoring,
            timestep=critic_timestep,
            **tf_kwargs,
        )

        if self.args.denoising_loss_type == "flow":
            from utils.wan_wrapper import WanDiffusionWrapper
            flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake_image.flatten(0, 1),
                xt=noisy_chunk.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1),
            )
            pred_fake_noise = None
        else:
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1),
                xt=noisy_chunk.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1),
            ).unflatten(0, chunk.shape[:2])

        critic_log: Dict[str, Any] = {
            "critic_timestep": critic_timestep.detach(),
            "streaming_new_frames": float(info["new_frames"]),
            "streaming_current_length": float(info["current_length"]),
        }
        if not gradient_mask.any():
            # End-of-sequence iter where ``new_frames`` (= npb) lands
            # entirely inside the last-chunk-masked tail → AND is all
            # False. Short-circuit to a zero loss + telemetry so the
            # trainer can collapse-gate / reset on this iter instead
            # of NaNing the optimizer. Zero-loss is connected to the
            # autograd graph through ``pred_fake_image`` so
            # ``.backward()`` succeeds (zero gradient on every
            # fake_score parameter).
            critic_log["critic_empty_mask"] = 1.0
            zero_loss = (pred_fake_image.double() * 0.0).sum()
            return zero_loss, critic_log
        gradient_mask_flat = gradient_mask.flatten(0, 1)
        denoising_loss = self.denoising_loss_func(
            x=chunk.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred,
            gradient_mask=gradient_mask_flat,
        )
        for k in ("baseline_last_chunk_mae", "baseline_avg_rollout_mae", "last_chunk_mae", "mae_extension_count"):
            if k in info:
                critic_log[k] = info[k]
        return denoising_loss, critic_log

    # ------------------------------------------------------------------
    # SC-DMD: Self-Consistent Distribution Matching Distillation.
    # Reference: "Salt: Self-Consistent Distribution Matching with
    # Cache-Aware Training for Fast Video Generation" (arXiv 2604.03118v1).
    # ------------------------------------------------------------------
    def _ensure_sc_kv_cache(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[list, list]:
        """Lazily allocate a small fresh KV cache for SC-DMD chunk-0
        forwards. Sized to exactly ``num_frame_per_block`` frames (no
        rolling, no extension headroom) so SC-DMD's memory footprint
        is bounded by ~1 chunk's worth of K/V — negligible vs the
        rollout cache (which is sized to ``rollout_frames``).

        The cache is reset (positions zeroed, ``is_init=False``) on
        every call so consecutive SC-DMD invocations get a fresh
        chunk-0 view (no cross-iter leakage).
        """
        if self.inference_pipeline is None:
            raise RuntimeError(
                "ActionForcingDMD.inference_pipeline must be set before "
                "sc_dmd_loss; the trainer assigns this in _build_pipeline."
            )
        npb = self.num_frame_per_block
        pipe = self.inference_pipeline
        fsl = pipe.frame_seq_length
        n_blocks = pipe.num_transformer_blocks
        sc_size = npb * fsl

        cache = self._sc_kv_cache
        cross = self._sc_crossattn_cache
        need_realloc = (
            cache is None
            or cross is None
            or len(cache) != n_blocks
            or cache[0]["k"].shape[0] != batch_size
            or cache[0]["k"].shape[1] != sc_size
            or cache[0]["k"].dtype != dtype
            or cache[0]["k"].device != device
        )
        if need_realloc:
            cache = []
            for _ in range(n_blocks):
                cache.append({
                    "k": torch.zeros(
                        [batch_size, sc_size, 12, 128],
                        dtype=dtype, device=device,
                    ),
                    "v": torch.zeros(
                        [batch_size, sc_size, 12, 128],
                        dtype=dtype, device=device,
                    ),
                    "global_end_index": torch.tensor(
                        [0], dtype=torch.long, device=device,
                    ),
                    "local_end_index": torch.tensor(
                        [0], dtype=torch.long, device=device,
                    ),
                })
            cross = []
            for _ in range(n_blocks):
                cross.append({
                    "k": torch.zeros(
                        [batch_size, 512, 12, 128],
                        dtype=dtype, device=device,
                    ),
                    "v": torch.zeros(
                        [batch_size, 512, 12, 128],
                        dtype=dtype, device=device,
                    ),
                    "is_init": False,
                })
            self._sc_kv_cache = cache
            self._sc_crossattn_cache = cross
        else:
            for layer in cache:
                layer["global_end_index"].zero_()
                layer["local_end_index"].zero_()
            for layer in cross:
                layer["is_init"] = False
        return cache, cross

    def _sample_sc_triplet(
        self, device: torch.device,
    ) -> Optional[Tuple[float, float, float]]:
        """Pick three rungs ``(t_s > t_m > t_e)`` from the inference
        denoising step grid, broadcast from rank 0 so DDP ranks stay
        in lockstep on the sampled triplet (otherwise different ranks
        would compute different SC losses and DDP all-reduce would
        average inconsistent gradients).

        Returns ``None`` if there are fewer than 3 rungs (SC-DMD
        needs a strict ``t_s > t_m > t_e`` triple). Caller must skip
        the SC pass when ``None`` is returned.
        """
        ds = self.inference_pipeline.denoising_step_list
        n = len(ds)
        if n < 3:
            return None
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            indices = torch.randperm(n, device=device)[:3].sort().values
        else:
            indices = torch.empty(3, dtype=torch.long, device=device)
        if dist.is_initialized():
            dist.broadcast(indices, src=0)
        # ``denoising_step_list`` is high-to-low (timestep 999 first,
        # 250 last). ASCENDING indices map to t_s (highest) > t_m > t_e.
        i_s = int(indices[0].item())
        i_m = int(indices[1].item())
        i_e = int(indices[2].item())
        return float(ds[i_s]), float(ds[i_m]), float(ds[i_e])

    def _flow_partial_denoise(
        self,
        x_ts: torch.Tensor,
        x0_hat: torch.Tensor,
        t_s_tensor: torch.Tensor,
        t_e_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """One-step Euler partial-denoise from ``t_s`` to ``t_e`` using
        the model's clean-space prediction ``x0_hat`` of ``x_ts``.

        For ``FlowMatchScheduler`` (linear/shifted sigma):
            x_ts = (1 - σ_s) * x0 + σ_s * ε
            ⇒ ε_implicit = (x_ts - (1 - σ_s) * x0_hat) / σ_s
            ⇒ x_te = (1 - σ_e) * x0_hat + σ_e * ε_implicit
                   = (σ_e/σ_s) * x_ts
                     + ((1 - σ_e) - (σ_e/σ_s) * (1 - σ_s)) * x0_hat

        Sigmas are looked up from ``scheduler.sigmas`` so this respects
        ``timestep_shift`` (non-linear sigma mapping). All sigma math
        is fp32 for numerical stability; final cast back to ``x_ts.dtype``.

        Inputs:
          - ``x_ts``: ``[B, F, C, H, W]``
          - ``x0_hat``: ``[B, F, C, H, W]`` (carries gradient)
          - ``t_s_tensor``, ``t_e_tensor``: ``[B, F]`` int64
        """
        B, F = t_s_tensor.shape
        sched = self.scheduler
        sched.sigmas = sched.sigmas.to(x_ts.device)
        sched.timesteps = sched.timesteps.to(x_ts.device)

        flat_s = t_s_tensor.flatten(0, 1)
        flat_e = t_e_tensor.flatten(0, 1)
        id_s = torch.argmin(
            (sched.timesteps.unsqueeze(0) - flat_s.float().unsqueeze(1)).abs(),
            dim=1,
        )
        id_e = torch.argmin(
            (sched.timesteps.unsqueeze(0) - flat_e.float().unsqueeze(1)).abs(),
            dim=1,
        )
        sigma_s = sched.sigmas[id_s].float().view(B, F, 1, 1, 1)
        sigma_e = sched.sigmas[id_e].float().view(B, F, 1, 1, 1)

        ratio = sigma_e / sigma_s.clamp(min=1e-8)
        coef_x0 = (1.0 - sigma_e) - ratio * (1.0 - sigma_s)
        ratio = ratio.to(x_ts.dtype)
        coef_x0 = coef_x0.to(x_ts.dtype)

        return ratio * x_ts + coef_x0 * x0_hat

    def sc_dmd_loss(
        self,
        conditional_dict: dict,
        clean_latent: torch.Tensor,
        seed_frames: int = 0,
    ) -> Tuple[torch.Tensor, dict]:
        """Salt's Self-Consistent DMD regularizer (single chunk, fresh cache).

        Implements ``L_SC = E[||x_te^(1) - x_te^(2)||²]`` (Eq. 8 of
        the Salt paper) where:
            x_te^(1) = Ψ_θ^{ts→te}(x_ts)
            x_te^(2) = Ψ_θ^{tm→te}(Ψ_θ^{ts→tm}(x_ts))

        and ``Ψ_θ^{a→b}`` is one Euler step of the student's
        velocity field from noise level ``a`` to ``b`` under the
        ``FlowMatchScheduler``'s sigma mapping.

        Implementation choices (kept deliberately MINIMAL — paper's
        Section 3.2 plain-SC variant, no mixed-step training, no
        cache-conditioned feature alignment):

          * Single chunk: we always run SC at chunk 0 with a FRESH
            KV cache (sized to ``num_frame_per_block`` frames). This
            avoids polluting the main rollout's cache and keeps the
            cost bounded to 2 DiT forwards on a single 3-frame chunk
            (~1-2% extra wallclock per gen step). The model's chunk-0
            forwards attend to NO prior context (empty cache), so they
            isolate the velocity field's local consistency without
            confounds from earlier-chunk conditioning.

          * Triplet sampled from ``denoising_step_list``: ``t_s, t_m,
            t_e`` are three distinct rungs from the inference grid in
            descending order, broadcast from rank 0 for DDP lockstep.
            Both ``t_s → t_e`` (direct) and ``t_s → t_m → t_e``
            (composed) paths use the model's CLEAN-SPACE prediction
            x0_hat to derive the velocity step (math is exact for
            flow matching with arbitrary ``timestep_shift``; see
            ``_flow_partial_denoise``).

          * Reference clean latent: we noise the FIRST GT chunk
            (``clean_latent[:, :npb]``) to level ``t_s`` to get
            ``x_ts``. The noise is fresh per call. The clean latent
            is detached and never gradient-flowed — it's only the
            base point of the SC defect, not a target.

          * Both forward passes BACKPROP to ``θ`` (no stop-grad on
            either path); the symmetric formulation pulls both
            predictions toward agreement, which is the local
            convergence property the Salt paper proves (Theorem 1).

        Returns ``(sc_loss, log_dict)``. Log dict keys:
            sc_dmd_t_s, sc_dmd_t_m, sc_dmd_t_e: the sampled triple
            sc_dmd_loss_raw: the raw scalar (for wandb)
            sc_dmd_skipped: 1 if SC was skipped (too few rungs), else 0
        """
        npb = self.num_frame_per_block
        device = self.device
        dtype = self.dtype

        triplet = self._sample_sc_triplet(device=device)
        if triplet is None:
            zero = torch.zeros((), device=device, dtype=dtype)
            return zero, {
                "sc_dmd_skipped": 1.0,
                "sc_dmd_loss_raw": 0.0,
            }
        t_s, t_m, t_e = triplet

        if clean_latent is None:
            raise RuntimeError(
                "sc_dmd_loss requires clean_latent (GT slice) to noise to t_s"
            )
        if clean_latent.shape[1] < npb:
            raise RuntimeError(
                f"sc_dmd_loss: clean_latent has only {clean_latent.shape[1]} "
                f"frames, need at least {npb} (one chunk)."
            )

        x0_chunk = clean_latent[:, :npb].to(
            dtype=dtype, device=device,
        ).detach()
        # ``conditional_dict`` may include leading seed-prefill action
        # streams (when the trainer runs with dmd_context KV-cache
        # prefill). The first chunk of ``clean_latent`` is the first
        # rollout chunk (not a seed chunk), so we slice the cond dict
        # at ``seed_frames`` to skip the seed actions and pick up the
        # first rollout chunk's conditioning.
        chunk_cond = _slice_per_frame_streams(
            conditional_dict, frame_start=int(seed_frames), frame_count=npb,
        )

        B = x0_chunk.shape[0]
        eps = torch.randn_like(x0_chunk)

        t_s_int = int(round(t_s))
        t_m_int = int(round(t_m))
        t_s_tensor = torch.full(
            [B, npb], t_s_int, device=device, dtype=torch.int64,
        )
        t_m_tensor = torch.full(
            [B, npb], t_m_int, device=device, dtype=torch.int64,
        )
        t_e_tensor = torch.full(
            [B, npb], int(round(t_e)), device=device, dtype=torch.int64,
        )

        x_ts = self.scheduler.add_noise(
            x0_chunk.flatten(0, 1),
            eps.flatten(0, 1),
            t_s_tensor.flatten(0, 1),
        ).unflatten(0, x0_chunk.shape[:2]).contiguous()

        # First forward: vθ(x_ts, t_s, c) → x0_hat_ts.
        kv_cache, crossattn_cache = self._ensure_sc_kv_cache(
            batch_size=B, dtype=dtype, device=device,
        )
        _, x0_hat_ts = self.generator(
            noisy_image_or_video=x_ts,
            conditional_dict=chunk_cond,
            timestep=t_s_tensor,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=0,
        )

        # Direct path: x_te^(1) = Ψ_θ^{ts→te}(x_ts).
        x_te_1 = self._flow_partial_denoise(
            x_ts=x_ts, x0_hat=x0_hat_ts,
            t_s_tensor=t_s_tensor, t_e_tensor=t_e_tensor,
        )

        # Composed path step A: x_tm = Ψ_θ^{ts→tm}(x_ts).
        x_tm = self._flow_partial_denoise(
            x_ts=x_ts, x0_hat=x0_hat_ts,
            t_s_tensor=t_s_tensor, t_e_tensor=t_m_tensor,
        )

        # Reset cache (positions zeroed; tensors reused) for the
        # second forward at t_m. This is critical: the first forward
        # wrote K/V at positions [0:npb] for noise level t_s; the
        # second forward must NOT attend to that (it's a different
        # noise level). Resetting positions effectively makes the
        # cache empty again for chunk 0.
        for layer in kv_cache:
            layer["global_end_index"].zero_()
            layer["local_end_index"].zero_()
        for layer in crossattn_cache:
            layer["is_init"] = False

        # Second forward: vθ(x_tm, t_m, c) → x0_hat_tm.
        _, x0_hat_tm = self.generator(
            noisy_image_or_video=x_tm,
            conditional_dict=chunk_cond,
            timestep=t_m_tensor,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=0,
        )

        # Composed path step B: x_te^(2) = Ψ_θ^{tm→te}(x_tm).
        x_te_2 = self._flow_partial_denoise(
            x_ts=x_tm, x0_hat=x0_hat_tm,
            t_s_tensor=t_m_tensor, t_e_tensor=t_e_tensor,
        )

        # SC defect (fp32 for numerical stability of small differences).
        sc_loss = (x_te_1.float() - x_te_2.float()).pow(2).mean()
        sc_loss = sc_loss.to(dtype)

        return sc_loss, {
            "sc_dmd_t_s": float(t_s),
            "sc_dmd_t_m": float(t_m),
            "sc_dmd_t_e": float(t_e),
            "sc_dmd_loss_raw": float(sc_loss.detach().item()),
            "sc_dmd_skipped": 0.0,
        }
