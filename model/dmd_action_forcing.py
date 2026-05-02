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

    # Sentinel log key the trainer reads to decide whether to skip the
    # post-critic teacher backward (end-of-ride: no gt_target available
    # on at least one rank → all ranks return a non-grad zero loss via
    # the all_reduce(MAX) skip path). Hoisted to a class-level constant
    # so a rename doesn't silently break the trainer's skip gate. The
    # trainer references this attribute, not the literal string.
    REAL_TEACHER_SKIP_KEY = "real_teacher_skipped_short_ride"

    def __init__(self, args, device):
        super().__init__(args, device)

        self.num_frame_per_block = int(getattr(args, "num_frame_per_block", 3))
        if self.num_frame_per_block > 1 and hasattr(self.generator, "model"):
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        # ``action_dims``: which ss_vae output dims feed the action
        # conditioning streams. Stashed for the action-mode freeze
        # block — it slices the teacher's full-8-d ``z_real`` down to
        # the same dims as ``gt_z_per_slot`` (which the trainer
        # pre-computes from cond_dict's per-frame z_actions, already
        # sliced to ``action_dims`` at dataset load time).
        action_dims_cfg = getattr(args, "action_dims", None)
        self.action_dims: Optional[List[int]] = (
            list(action_dims_cfg) if action_dims_cfg is not None else None
        )

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
        if self.dmd_context not in ("self", "gt", "mix"):
            raise ValueError(
                f"dmd_context must be 'self', 'GT', or 'mix' "
                f"(case-insensitive); got "
                f"{getattr(args, 'dmd_context', None)!r}."
            )
        # Normalise canonical case for downstream comparisons.
        if self.dmd_context == "gt":
            self.dmd_context = "GT"
        # ``dmd_context_mix_p``: per-CHUNK probability of routing
        # real_score's clean_x through the GT view (vs the self view).
        # Only consulted when ``dmd_context == "mix"``. The 21-frame
        # clean_x window is divided into ``num_training_frames /
        # num_frame_per_block`` chunks; each chunk independently flips
        # a Bernoulli(``dmd_context_mix_p``) coin to decide if it's a
        # "GT-dominant" or "self-dominant" chunk. 0.5 = balanced;
        # 0.0 collapses to pure self; 1.0 collapses to pure GT. The
        # per-chunk vector is sampled on rank 0 and broadcast across
        # DDP so every rank takes the same per-chunk decision (DMD's
        # (fake-real) subtraction needs matched conditioning across
        # ranks; otherwise the gradient sum is malformed).
        self.dmd_context_mix_p = float(
            getattr(args, "dmd_context_mix_p", 0.5)
        )
        if not (0.0 <= self.dmd_context_mix_p <= 1.0):
            raise ValueError(
                f"dmd_context_mix_p must be in [0, 1]; got "
                f"{self.dmd_context_mix_p!r}."
            )
        # ``"mix"`` mode is now a uniform linear blend across ALL
        # frames:
        #   clean_x_real = (1 - mix_p) * clean_x_self + mix_p * noised_GT
        # No per-chunk Bernoulli, no chunk-level dominance — the
        # previous yinyang_random shuffling was removed because the
        # chunk-level dice introduced step-discontinuities along the
        # time axis of the clean_x context (visible as disjoint
        # artefacts in the student's reconstruction). The mix knob
        # (``dmd_context_mix_p``) alone now fully determines the
        # blend.

        # ---------------------------------------------------------------
        # Online real_teacher (v14-LoRA trained online vs GT video).
        # ---------------------------------------------------------------
        # When ``real_teacher_train_online`` is True, the v14 LoRA load
        # path (``_load_real_score_with_v14_lora``) skips
        # ``merge_and_unload``, the rank-256 adapter stays alive on top
        # of base Wan, only the LoRA params receive gradient, and the
        # trainer wires a separate optimizer + DDP wrap + per-iter
        # teacher step (``compute_real_teacher_loss_streaming``). When
        # False (default), real_score remains the frozen merged oracle
        # — legacy behavior preserved.
        self.real_teacher_train_online = bool(
            getattr(args, "real_teacher_train_online", False)
        )
        # Causal mask flag on the joint [clean | noisy] TF sequence
        # (v14 parity = True). When False, the inner CausalWanModel's
        # ``_prepare_teacher_forcing_mask`` returns a full-bidirectional
        # block mask. Plumbed onto ``self.real_score.model
        # .tf_use_causal_mask`` immediately after the LoRA load.
        self.real_teacher_causal_mask = bool(
            getattr(args, "real_teacher_causal_mask", True)
        )
        # List of trainable LoRA params, populated by
        # ``_load_real_score_with_v14_lora`` after the peft wrap.
        # The trainer reads this to build the teacher optimizer.
        self._real_teacher_trainable_params: List[torch.nn.Parameter] = []

        # ``dmd_frozen_teacher_pass_enabled``: when True (default when
        # online teacher is on), the model holds TWO real-score
        # modules:
        #   * ``self.real_score``       — base Wan + ACTIVE v14 LoRA
        #     (trainable). Used for the auxiliary teacher pass that
        #     trains the LoRA AND, optionally, flows gradient back to
        #     the student (option 3 from the dual-teacher design).
        #   * ``self.real_score_frozen`` — base Wan with v14 LoRA
        #     MERGED into the base weights (frozen, no_grad). Used in
        #     the DMD scoring path so the (fake - real) gradient is
        #     against a clean ``p_real`` model, not the moving-target
        #     online LoRA. Preserves the DMD theoretical guarantee.
        # When False, the legacy single-teacher behavior holds:
        # ``self.real_score`` is used in BOTH the DMD path and the
        # auxiliary training pass.
        self.dmd_frozen_teacher_pass_enabled = bool(
            getattr(args, "dmd_frozen_teacher_pass_enabled", True)
        )
        # Holds the frozen merged-v14 teacher when the dual-teacher
        # path is enabled. None otherwise. Loaded by
        # ``_load_real_score_with_v14_lora``.
        self.real_score_frozen: Optional[nn.Module] = None

        # ``aux_teacher_loss_weight``: weight on the auxiliary
        # online-teacher flow loss in the gen-step total loss. When
        # ``real_teacher_train_online`` is True, the auxiliary pass
        # forwards ``self.real_score`` (LoRA-active, with grad on the
        # LoRA params AND on the student's chunk via the noise base)
        # and computes ``(flow_pred - (ε - GT))²``. The loss
        # contributes:
        #   * Always: gradient to the LoRA → trains the online teacher.
        #   * When the noise base is the student chunk (input_source
        #     "student" / coin lands student): gradient also to the
        #     student via the noise base.
        # Set to 0.0 to keep the LoRA-training pass alive but not fold
        # the loss into the gen-step total (legacy behavior with the
        # post-critic separate teacher step).
        self.aux_teacher_loss_weight = float(
            getattr(args, "aux_teacher_loss_weight", 1.0)
        )

        # ``real_teacher_input_source`` ∈ {"student", "gt", "mix"}
        # controls how ``compute_real_teacher_loss_streaming`` builds
        # ``noisy_input``:
        #   "student" : add_noise(student_chunk_detached, ε, t)
        #               — current default; an amortized refinement
        #               target. Strictly speaking NOT score matching
        #               for ``p_real(x_t)`` because the input is
        #               drawn from a noised-student joint, not the
        #               noised-data marginal. As ``student → GT`` the
        #               bias decays and it converges back to a true
        #               score.
        #   "gt"      : add_noise(gt_target, ε, t) — proper DMD score
        #               matching: the teacher learns
        #               ``∇ log p_real(x_t)`` exactly, no curriculum
        #               bias.
        #   "mix"     : per-iter Bernoulli(``real_teacher_input_mix_gt_p``);
        #               True = GT, False = student. DDP-synced from
        #               rank 0 so all ranks pick the same source per
        #               iter.
        # Whatever source is chosen, the LOSS TARGET is always GT —
        # the source only changes what the noise is added to.
        self.real_teacher_input_source = str(
            getattr(args, "real_teacher_input_source", "student")
        )
        if self.real_teacher_input_source not in ("student", "gt", "mix"):
            raise ValueError(
                f"real_teacher_input_source must be one of "
                f"'student' / 'gt' / 'mix'; got "
                f"{self.real_teacher_input_source!r}."
            )
        self.real_teacher_input_mix_gt_p = float(
            getattr(args, "real_teacher_input_mix_gt_p", 0.5)
        )
        if not (0.0 <= self.real_teacher_input_mix_gt_p <= 1.0):
            raise ValueError(
                f"real_teacher_input_mix_gt_p must be in [0, 1]; got "
                f"{self.real_teacher_input_mix_gt_p!r}."
            )

        # ``flash_dmd_split_timestep`` — Flash-DMD-style decoupling of
        # DMD vs GAN gradients per iter (paper arXiv:2511.20549, §3.3).
        # When None (default) the legacy DMD2 summing of both losses
        # every iter is used. When set to an int (e.g. 500), the
        # trainer rolls a single DDP-synced scalar t per gen iter and
        # writes ``info["flash_dmd_regime"]`` ∈ {"high", "low"}:
        #   * High-noise iter (t > split): DMD active, constrained to
        #     [split, num_train_timestep]; trainer skips the
        #     generator's GAN-loss contribution (discriminator still
        #     trains independently).
        #   * Low-noise iter (t <= split): DMD skipped (zero-loss
        #     through chunk so backward still flows zero gradient
        #     through the generator); generator's GAN gradient is the
        #     only signal this iter.
        # The aux-teacher pass is orthogonal — it always fires when
        # ``aux_teacher_loss_weight > 0`` regardless of regime.
        # Critic (fake_score) training is unaffected.
        _split = getattr(args, "flash_dmd_split_timestep", None)
        self.flash_dmd_split_timestep = (
            int(_split) if _split is not None else None
        )

        # ``flash_dmd_paper_aligned_adv`` (default false) — Flash-DMD
        # paper §3.3 Eq. 8-9 path: every gen iter does BOTH DMD and
        # adv losses, summed (L_Gθ = L_DMD + λ · L_adv). Decoupling
        # is by timestep:
        #   * DMD: gen forward 1 at high-noise t (the existing rolling
        #     rollout, with grad through the random exit rung).
        #   * adv: ONE EXTRA gen forward 2 at LOW-noise ˆt on
        #     re-noised pred_x0, with grad ONLY through this single
        #     forward. The adv gradient never reaches the rolling
        #     rollout's high-noise denoising steps, so the GAN only
        #     updates the gen's texture-refinement behavior — exactly
        #     the paper's spec.
        # When True, the per-iter alternation (flash_dmd_split_timestep)
        # gate is bypassed in compute_generator_loss_streaming —
        # both losses fire every iter.
        # Memory: extra ~3-6 GB per rank (one extra gen forward + a
        # 21-frame KV cache); the cache is cached on self after
        # first allocation.
        self.flash_dmd_paper_aligned_adv = bool(
            getattr(args, "flash_dmd_paper_aligned_adv", False)
        )

        # Warm-start init: subsequent rolling chunks denoise from the
        # prior chunk's clean pred re-noised at
        # ``denoising_step_list[warm_start_rung_idx]`` via the shortened
        # ladder ``denoising_step_list[warm_start_rung_idx:]``. The
        # very first chunk in a rollout (anchor in streaming mode;
        # block 0 in non-streaming) uses the standard cold init (pure
        # noise + full ladder). Default off; flip via config to A/B.
        # ``warm_start_rung_idx`` defaults to 1 (= second-noisiest
        # rung in the standard 4-step ladder); move it later in the
        # ladder for less-aggressive warm-start. Validation against
        # the actual denoising_step_list length lives in the pipeline.
        self.warm_start_init = bool(getattr(args, "warm_start_init", False))
        self.warm_start_rung_idx = int(getattr(args, "warm_start_rung_idx", 1))

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
        # ``cfg_uncond_keep_actions``: when True, ``build_action_conditional``
        # zeros only ``prompt_embeds`` in the unconditional dict and
        # keeps the action streams (``_action_modulation`` /
        # ``_action_tokens``) identical to the conditional. This makes
        # the uncond forward IN-DISTRIBUTION for v14 LoRA scorers (which
        # were trained with non-zero action conditioning everywhere)
        # while still letting CFG sharpen the prompt direction. Default
        # False = legacy zero-everything CF/Wan convention.
        self.cfg_uncond_keep_actions = bool(
            getattr(args, "cfg_uncond_keep_actions", False)
        )
        # ``dmd_debug_step``: when set, the eval-stash branch of
        # ``_compute_kl_grad`` runs an EXTRA pair of real_score /
        # fake_score forwards at this fixed timestep and overwrites
        # the stash's pred_real / pred_fake / dmd_timestep / noisy_input
        # entries with the debug-t versions. Diagnostic only — does NOT
        # affect the training-time DMD gradient (random t still used
        # for ``grad = pred_fake - pred_real``).
        # Accepted values:
        #   * int → fixed timestep in [0, num_train_timestep)
        #   * "last_rung" → last entry of ``denoising_step_list`` (= the
        #     rollout's last-rung exit timestep)
        #   * None / unset → no override, eval stash uses random t
        _dbg = getattr(args, "dmd_debug_step", None)
        if _dbg is None or (isinstance(_dbg, str) and _dbg.lower() in {"none", "off", ""}):
            self.dmd_debug_step = None
        elif isinstance(_dbg, str) and _dbg.lower() in {"last_rung", "last", "last_step"}:
            _dsl = list(getattr(args, "denoising_step_list", []) or [])
            self.dmd_debug_step = int(round(float(_dsl[-1]))) if _dsl else None
        else:
            self.dmd_debug_step = int(_dbg)

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
        # Apply the TF context attrs to every scorer DiT — including
        # ``real_score_frozen`` if the dual-teacher path built it. peft
        # does not forward attribute SETs to the base model, so we
        # unwrap before assigning.
        scorer_wrappers = [self.real_score.model, self.fake_score.model]
        if self.real_score_frozen is not None:
            scorer_wrappers.append(self.real_score_frozen.model)
        for wrapper in scorer_wrappers:
            m = (
                wrapper.get_base_model()
                if hasattr(wrapper, "get_base_model") else wrapper
            )
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

        # Finalize freezes. When real_teacher_train_online=True, the
        # v14 LoRA load (_load_real_score_with_v14_lora) deliberately
        # leaves the LoRA params with requires_grad=True so the trainer
        # can build an optimizer + DDP-wrap real_score. Don't clobber
        # that here — only freeze if the teacher is the offline merged
        # oracle.
        if not self.real_teacher_train_online:
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
        # ``boundary_vae_roundtrip``: Causal-Forcing/long_video parity
        # for the rolling-cache boundary (CF: long_video/model/base.py:
        # 155-167). When True and the streaming chunk has overlap (iter
        # k>=2), the first frame of the chunk is replaced by a VAE
        # decode->encode round-trip — temporal context is the prior
        # iter's full chunk, and the re-encoded LAST pixel frame
        # becomes the fresh image-manifold latent at the seam. Default
        # False (legacy behavior); flip via config when long-horizon
        # AR drift starts compounding into blur.
        self.boundary_vae_roundtrip: bool = bool(
            getattr(args, "boundary_vae_roundtrip", False)
        )
        # Deterministic stride for slide-and-train: when > 0, every
        # ``_streaming_pick_new_frames`` call returns exactly this many
        # ``num_frame_per_block``-chunks (* npb frames) instead of the
        # legacy random pick from {min_new, max_new}. Phase-1 freeze
        # YAML sets ``num_chunks_roll_forward: 6`` (= 18 frames per
        # slide) so the slide-and-train helper's per-iter MAE is
        # comparable across iters. Iter 1's full chunk_size advance is
        # unaffected (it's forced by the previous_chunk=None branch in
        # ``generate_next_chunk``).
        self.streaming_force_new_frame_chunks: int = int(
            getattr(args, "num_chunks_roll_forward", 0)
        )

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

        # Load v14 LoRA state dict ONCE, off disk, here. The dual-
        # teacher path calls ``_apply_v14_lora`` twice (frozen + LoRA);
        # without this cache the same checkpoint would be re-read from
        # /scratch on each call.
        _v14_ckpt_blob = torch.load(v14_ckpt_path, map_location="cpu")
        v14_lora_sd = _v14_ckpt_blob.get("lora")
        if v14_lora_sd is None:
            raise KeyError(
                f"v14 checkpoint missing 'lora' key: "
                f"have {list(_v14_ckpt_blob.keys())}"
            )
        del _v14_ckpt_blob

        # Helper: apply v14 LoRA to a wrapper's ``.model`` field.
        # ``merge=True`` returns the merged base (LoRA folded in,
        # peft removed); ``merge=False`` returns the peft-wrapped
        # model with the LoRA params marked trainable. Mutates
        # ``wrapper.model`` in place.
        def _apply_v14_lora(wrapper, *, merge: bool) -> List[nn.Parameter]:
            target_modules = self._collect_target_modules(wrapper.model)
            if not target_modules:
                target_modules = ["q", "k", "v", "o"]
            lora_config = LoraConfig(
                r=rank,
                lora_alpha=alpha,
                lora_dropout=dropout,
                target_modules=target_modules,
                bias="none",
            )
            peft_model = peft.get_peft_model(wrapper.model, lora_config)
            lora_sd = v14_lora_sd
            try:
                set_peft_model_state_dict(peft_model, lora_sd)
            except Exception:
                from peft import get_peft_model_state_dict
                current_sd = get_peft_model_state_dict(peft_model)
                matched = 0
                for key in current_sd:
                    if (
                        key in lora_sd
                        and current_sd[key].shape == lora_sd[key].shape
                    ):
                        current_sd[key] = lora_sd[key]
                        matched += 1
                set_peft_model_state_dict(peft_model, current_sd)
                if _is_main():
                    logging.info(
                        "[ActionForcingDMD] real_score LoRA cross-load "
                        "matched %d/%d", matched, len(current_sd),
                    )
            if merge:
                merged = peft_model.merge_and_unload()
                try:
                    from peft.tuners.lora import LoraLayer
                    for _, m in merged.named_modules():
                        if isinstance(m, LoraLayer):
                            raise RuntimeError(
                                "LoRA layers remain after merge_and_unload."
                            )
                except Exception:
                    pass
                wrapper.model = merged.to(device=device, dtype=self.dtype)
                return []
            wrapper.model = peft_model.to(device=device, dtype=self.dtype)
            trainable: List[nn.Parameter] = []
            for n, p in wrapper.model.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True
                    trainable.append(p)
                else:
                    p.requires_grad = False
            return trainable

        if _is_main():
            logging.info(
                "[ActionForcingDMD] Applying v14 LoRA (rank=%d alpha=%s drop=%s) "
                "to real_score (online=%s, frozen_pass=%s).",
                rank, alpha, dropout,
                self.real_teacher_train_online,
                self.dmd_frozen_teacher_pass_enabled
                and self.real_teacher_train_online,
            )

        # ---------------------------------------------------------------
        # Dual-teacher path: build the FROZEN merged-v14 copy FIRST
        # (deepcopy of the base wrapper before mutating the live one).
        # The live ``self.real_score`` then becomes the LoRA-online
        # teacher in the second branch below.
        # ---------------------------------------------------------------
        if (
            self.real_teacher_train_online
            and self.dmd_frozen_teacher_pass_enabled
        ):
            import copy as _copy
            frozen_wrapper = _copy.deepcopy(self.real_score)
            _apply_v14_lora(frozen_wrapper, merge=True)
            for p in frozen_wrapper.parameters():
                p.requires_grad = False
            frozen_wrapper.model.eval()
            inner_frozen = frozen_wrapper.model
            inner_frozen.tf_use_causal_mask = bool(self.real_teacher_causal_mask)
            self.real_score_frozen = frozen_wrapper
            if _is_main():
                logging.info(
                    "[ActionForcingDMD] dual-teacher: built frozen merged-v14 "
                    "real_score_frozen (eval, no_grad, dtype=%s).",
                    self.dtype,
                )

        if self.real_teacher_train_online:
            # Online teacher: apply v14 LoRA to ``self.real_score``
            # WITHOUT merging — the peft adapter stays alive, base is
            # frozen, LoRA params are trainable. Trainer builds an
            # optimizer over ``_real_teacher_trainable_params`` below;
            # DDP-wraps ``real_score.model`` separately.
            trainable_lora = _apply_v14_lora(self.real_score, merge=False)
            self._real_teacher_trainable_params = trainable_lora
            if not trainable_lora:
                raise RuntimeError(
                    "[ActionForcingDMD] real_teacher_train_online=True but "
                    "no LoRA params were marked trainable after the peft "
                    "wrap — check peft naming convention or LoRA config."
                )
            # Causal mask flag — consumed by
            # ``CausalWanModel._prepare_teacher_forcing_mask`` (see
            # ``wan/modules/causal_model.py``). Default True (v14 parity).
            inner = self.real_score.model
            base_for_flag = (
                inner.get_base_model()
                if hasattr(inner, "get_base_model") else inner
            )
            base_for_flag.tf_use_causal_mask = bool(self.real_teacher_causal_mask)

            # Activation checkpointing on real_score's WanBlocks (memory
            # — see prior comment block; trade is ~10 GB saved for ~10%
            # wallclock from re-forward on backward).
            gc_on = bool(getattr(args, "gradient_checkpointing", True))
            if gc_on:
                base_for_flag.gradient_checkpointing = True

            if _is_main():
                logging.info(
                    "[ActionForcingDMD] real_teacher_train_online=True: "
                    "v14 LoRA adapter kept (no merge), %d LoRA params "
                    "trainable, tf_use_causal_mask=%s, "
                    "gradient_checkpointing=%s.",
                    len(trainable_lora),
                    self.real_teacher_causal_mask,
                    gc_on,
                )
            return

        # Legacy single-teacher path: merge LoRA into ``self.real_score``
        # and freeze. The merged model is the v14 oracle that DMD scoring
        # forwards through.
        _apply_v14_lora(self.real_score, merge=True)
        self.real_score.model.tf_use_causal_mask = bool(self.real_teacher_causal_mask)

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
        # ``cfg_uncond_keep_actions``: if True, the unconditional dict
        # ZEROS ONLY the prompt embeds and KEEPS the action streams
        # identical to the conditional. This avoids feeding the v14
        # LoRA-merged real_score an OOD all-zero action input (the LoRA
        # was trained with non-zero action conditioning everywhere; an
        # all-zero uncond pushes pred_real toward an OOD direction when
        # CFG extrapolates ``cond + scale*(cond - uncond)``). Default
        # False = legacy zero-everything CF/Wan convention.
        if getattr(self, "cfg_uncond_keep_actions", False):
            unconditional = {
                "prompt_embeds": torch.zeros_like(prompt_embeds),
                "_action_modulation": modulation,
                "_action_tokens": action_tokens,
            }
        else:
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

        # ``dual_grad_rollout`` is gated on the paper-aligned-adv knob
        # AND ``requires_grad=True``. The critic step (requires_grad=
        # False) doesn't need a last-rung grad forward, so disable the
        # two-grad-point path there to avoid wasting one extra forward
        # per chunk on a pred we'll never backprop through.
        dual_grad_rollout = bool(self.flash_dmd_paper_aligned_adv) and bool(
            requires_grad
        )
        # Warm-start init applies to BOTH gen and critic rollouts so
        # the critic sees the same denoising trajectory shape as the
        # gen (otherwise fake_score would learn a different
        # distribution than the gen produces).
        pred_image_or_video, denoised_timestep_from, denoised_timestep_to = (
            self.inference_pipeline.inference_with_trajectory(
                noise=noise,
                clean_image_or_video=None,
                gt_latents=clean_latent,
                enable_mae_extension=enable_mae_extension,
                seed_latents=seed_latents,
                requires_grad=requires_grad,
                dual_grad_rollout=dual_grad_rollout,
                warm_start_init=self.warm_start_init,
                warm_start_rung_idx=self.warm_start_rung_idx,
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
        # Optional eval-time stash so the trainer can decode the
        # scorers' denoised x0 estimates as sample videos. ``None``
        # = capture disabled (default); a dict means the trainer
        # has armed the stash for this iter. Read-only side effect
        # on the loss math (just .detach() copies into a dict).
        stash = getattr(self, "_dmd_eval_stash", None)
        if isinstance(stash, dict):
            stash["pred_real"] = pred_real_image.detach()
            stash["pred_fake"] = pred_fake_image.detach()
            stash["dmd_timestep"] = int(timestep.flatten()[0].item())
            stash["noisy_input"] = noisy_image_or_video.detach()

            # ``dmd_debug_step`` override: re-noise the student's x0 at
            # a FIXED timestep (e.g. the last denoising rung) and re-run
            # the scorers, overwriting the stash entries with the
            # debug-t versions. Lets the eval video render pred_real /
            # pred_fake at a low, comparable noise level instead of the
            # randomly-sampled training-time ``timestep`` which is
            # often very high. Training-time DMD gradient above is
            # untouched.
            if self.dmd_debug_step is not None:
                with torch.no_grad():
                    debug_t_int = int(self.dmd_debug_step)
                    debug_t = torch.full_like(timestep, debug_t_int)
                    debug_noise = torch.randn_like(estimated_clean_image_or_video)
                    _b, _f = estimated_clean_image_or_video.shape[:2]
                    noisy_debug = self.scheduler.add_noise(
                        estimated_clean_image_or_video.flatten(0, 1),
                        debug_noise.flatten(0, 1),
                        debug_t.flatten(0, 1),
                    ).unflatten(0, (_b, _f))

                    _, dbg_pred_fake_cond = self.fake_score(
                        noisy_image_or_video=noisy_debug,
                        conditional_dict=conditional_dict,
                        timestep=debug_t,
                        **tf_kwargs_fake,
                    )
                    if self.fake_guidance_scale != 0.0:
                        _, dbg_pred_fake_uncond = self.fake_score(
                            noisy_image_or_video=noisy_debug,
                            conditional_dict=unconditional_dict,
                            timestep=debug_t,
                            **tf_kwargs_fake,
                        )
                        dbg_pred_fake = dbg_pred_fake_cond + (
                            dbg_pred_fake_cond - dbg_pred_fake_uncond
                        ) * self.fake_guidance_scale
                    else:
                        dbg_pred_fake = dbg_pred_fake_cond

                    _, dbg_pred_real_cond = self.real_score(
                        noisy_image_or_video=noisy_debug,
                        conditional_dict=conditional_dict,
                        timestep=debug_t,
                        **tf_kwargs_real,
                    )
                    _, dbg_pred_real_uncond = self.real_score(
                        noisy_image_or_video=noisy_debug,
                        conditional_dict=unconditional_dict,
                        timestep=debug_t,
                        **tf_kwargs_real,
                    )
                    dbg_pred_real = dbg_pred_real_cond + (
                        dbg_pred_real_cond - dbg_pred_real_uncond
                    ) * self.real_guidance_scale

                    stash["pred_real"] = dbg_pred_real.detach()
                    stash["pred_fake"] = dbg_pred_fake.detach()
                    stash["dmd_timestep"] = debug_t_int
                    stash["noisy_input"] = noisy_debug.detach()
                    stash["dmd_debug_step_active"] = 1.0
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
        flash_dmd_t_min: Optional[int] = None,
        flash_dmd_t_max: Optional[int] = None,
    ) -> torch.Tensor:
        """Sample DMD timestep with CF's ``ts_schedule`` clamp + shift.

        ``flash_dmd_t_min`` / ``flash_dmd_t_max`` are HARD bounds set
        by the trainer's Flash-DMD regime gate. Unlike
        ``denoised_timestep_from/to`` (which are gated by the
        ``ts_schedule`` / ``ts_schedule_max`` config flags), the
        Flash-DMD bounds always take effect when supplied — they
        shouldn't be ignored just because the ts_schedule flags are
        off in the active config. They override the corresponding
        endpoint; the other endpoint follows the existing
        ts_schedule-or-default logic.
        """
        # Lower bound: Flash-DMD override > ts_schedule from-pipeline
        # > config default.
        if flash_dmd_t_min is not None:
            min_timestep = int(flash_dmd_t_min)
        elif self.ts_schedule and denoised_timestep_to is not None:
            min_timestep = denoised_timestep_to
        else:
            min_timestep = self.min_score_timestep
        # Upper bound: Flash-DMD override > ts_schedule from-pipeline
        # > config default.
        if flash_dmd_t_max is not None:
            max_timestep = int(flash_dmd_t_max)
        elif self.ts_schedule_max and denoised_timestep_from is not None:
            max_timestep = denoised_timestep_from
        else:
            max_timestep = self.num_train_timestep
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
        gt_z_per_slot: Optional[torch.Tensor] = None,
        flash_dmd_t_min: Optional[int] = None,
        flash_dmd_t_max: Optional[int] = None,
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
                flash_dmd_t_min=flash_dmd_t_min,
                flash_dmd_t_max=flash_dmd_t_max,
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
        # FUNDAMENTAL: gradient_mask arrives non-None at this loss
        # API (every caller routes through ``_dmd_score_grad_mask``
        # which always produces a tensor). Enforce the contract here
        # rather than only at the MSE boundary below — the freeze
        # block below mutates ``gradient_mask`` in place via
        # ``gradient_mask.clone()``, which would AttributeError on
        # None. Fail loud here so any caller regression that bypasses
        # the boundary mask gets a clear error.
        if gradient_mask is None:
            raise RuntimeError(
                "compute_distribution_matching_loss requires "
                "``gradient_mask`` (every caller must build it from "
                "``self._dmd_score_grad_mask`` so the structurally-OOD "
                "last-chunk boundary stays masked uniformly)."
            )

        # Teacher-freeze detection. The freeze mask we build below
        # AND-merges into ``gradient_mask`` and ONLY affects this
        # function's DMD MSE — aux losses (action critic z-guidance,
        # R3GAN, SC-DMD) run on the unmasked ``pred_image`` from
        # ``_run_generator`` and keep training the student through
        # frames where the teacher is gated off. That's the design:
        # when DMD is wrong on a slot, the action critic should still
        # carry the gradient there.
        if bool(getattr(self, "teacher_freeze_detect_enabled", False)):
            mode = self.teacher_freeze_mode  # already lowercased at init
            freeze_mask_bf = None  # [B, F] bool, set below

            # Fail-loud on the configured-but-not-actionable case:
            # mode=='action' but the trainer never attached the teacher
            # callable. The init-time assert in __init__ already gates
            # on action_teacher_mode != 'off', but a future code path
            # that bypasses the trainer's _build_action_teacher hookup
            # would hit this.
            if mode == "action" and self._action_teacher_fn is None:
                raise RuntimeError(
                    "teacher_freeze_mode='action' but the trainer did not "
                    "attach ``self._action_teacher_fn``. Verify that "
                    "``_build_action_teacher`` ran (action_teacher_mode != "
                    "'off') and that the model has the attribute."
                )

            if gt_target is None:
                # gt_target is supplied by the caller (generator_loss /
                # streaming gen) when teacher_freeze is enabled. BOTH
                # modes need it: 'mae' compares pred_real against
                # gt_target directly; 'action' runs the action teacher
                # on gt_target to get z_gt for the cos(R,GT) decision.
                # Currently unreachable on the locked-in geometry
                # (min_ride_frames=69 covers cf+rollout) but worth
                # surfacing if a future config change makes it
                # reachable — silent fallback in either mode would
                # silently disable the freeze gate.
                dmd_log_dict["teacher_freeze_unavailable"] = 1.0

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
                    n_F = pred_real_image_detached.shape[1]
                    if n_F % npb != 0:
                        raise RuntimeError(
                            f"teacher_freeze_detect (mode=action): F={n_F} "
                            f"not divisible by num_frame_per_block={npb}."
                        )
                    n_slots = n_F // npb

                    # Decision metric: cos(z_real, z_gt) per slot.
                    # ``z_real`` runs the teacher pipeline (CoTracker +
                    # ss_vae) on real_score's pred — necessary, no
                    # shortcut. For ``z_gt`` we accept a pre-computed
                    # tensor sourced from the dataset's per-frame
                    # z_actions (which were encoded offline by the
                    # SAME ss_vae from the SAME motion data the
                    # teacher pipeline would re-derive via CoTracker)
                    # — no second CoTracker forward on GT latents.
                    # Falls back to running the teacher on gt_target
                    # only if the trainer didn't pre-compute, so the
                    # behaviour is preserved when callers don't yet
                    # pass the kwarg.
                    z_real = self._action_teacher_fn(pred_real_image_detached)
                    if z_real is None:
                        dmd_log_dict["teacher_freeze_unavailable"] = 1.0
                    else:
                        # Resolve z_gt — prefer pre-computed.
                        if gt_z_per_slot is not None:
                            z_gt = gt_z_per_slot.to(
                                dtype=z_real.dtype, device=z_real.device,
                            )
                            # ``z_real`` is the teacher's full ss_vae
                            # output (8-d). ``gt_z_per_slot`` may be
                            # sliced to ``action_dims`` (= 2-d) at
                            # dataset load time. Slice z_real to match
                            # so cos sim is on the same axes.
                            if (
                                z_gt.shape[-1] != z_real.shape[-1]
                                and self.action_dims is not None
                                and len(self.action_dims) == z_gt.shape[-1]
                            ):
                                z_real = z_real[..., self.action_dims]
                            dmd_log_dict["teacher_freeze_z_gt_source"] = 1.0  # = pre-computed
                        else:
                            # Back-compat fallback: re-derive via teacher.
                            z_gt = self._action_teacher_fn(gt_target)
                            if z_gt is None:
                                dmd_log_dict["teacher_freeze_unavailable"] = 1.0
                                z_real = None  # short-circuit below
                            else:
                                dmd_log_dict["teacher_freeze_z_gt_source"] = 0.0  # = cotracker
                        if z_real is not None and z_gt is not None:
                            if z_real.shape != z_gt.shape:
                                raise RuntimeError(
                                    f"teacher_freeze_detect (mode=action): "
                                    f"z_real shape {tuple(z_real.shape)} != "
                                    f"z_gt shape {tuple(z_gt.shape)}. "
                                    f"action_dims={self.action_dims}."
                                )
                            cos_RG = torch.nn.functional.cosine_similarity(
                                z_real.float(), z_gt.float(), dim=-1,
                            )  # [B, n_slots]
                            slot_freeze = cos_RG < self.teacher_freeze_action_threshold
                            freeze_mask_bf = slot_freeze.repeat_interleave(npb, dim=1)
                            dmd_log_dict["teacher_freeze_action_cos_RG_min"] = (
                                cos_RG.min().detach()
                            )
                            dmd_log_dict["teacher_freeze_action_cos_RG_mean"] = (
                                cos_RG.mean().detach()
                            )
                            dmd_log_dict["teacher_freeze_action_cos_RG_median"] = (
                                cos_RG.median().detach()
                            )

            # AND-merge the freeze mask into the running gradient_mask.
            # ``gradient_mask`` arrives non-None by contract (the loss
            # boundary's required-arg check below; matches Fix 1) so we
            # always have a tensor to clone-and-mutate.
            if freeze_mask_bf is not None and freeze_mask_bf.any():
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

        # gradient_mask is non-None by the contract enforced at the
        # top of this function. After the freeze AND-merge it can be
        # all-False (= every slot flagged); short-circuit to a
        # connected zero-loss so .backward() succeeds.
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
          * ``"self"``: ``None`` (same as fake; ``_compute_kl_grad``
            falls back to ``sc_clean_x``).
          * ``"GT"``: ``scheduler.add_noise(clean_x_GT, n,
            clean_x_aug_t)`` — real_score sees the GT version of the
            same time positions, lightly noised at ``self.clean_x_aug_t``
            for symmetry-breaking. ``clean_x_GT`` is required.
          * ``"mix"``: ``(1 - mix_p) * sc_clean_x + mix_p * noised_gt``
            uniformly across all frames. No per-chunk shuffle. The
            mix knob alone determines the blend; aug_t stays at 0
            (matches the dominant ``self`` contribution).

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

        sc_clean_x = clean_x_self.to(dtype=dtype, device=device)
        sc_aug_t = torch.zeros(
            (sc_clean_x.shape[0], self.num_training_frames),
            device=device, dtype=torch.long,
        )

        # Decide whether the real-side clean_x mixes in any GT this
        # iter. "self" and the critic-step short-circuit both produce
        # None (caller falls back to sc_clean_x).
        sc_clean_x_real: Optional[torch.Tensor] = None
        sc_aug_t_real: Optional[torch.Tensor] = None
        needs_gt = build_real_view and self.dmd_context in ("GT", "mix")
        if needs_gt and self.dmd_context == "mix":
            # mix_p == 0 collapses to pure self; skip the GT branch
            # entirely (cheaper, identical result).
            if float(self.dmd_context_mix_p) <= 0.0:
                needs_gt = False
        if needs_gt:
            if clean_x_GT is None:
                raise RuntimeError(
                    f"dmd_context={self.dmd_context!r} requires "
                    "clean_x_GT this iter but the caller did not "
                    "provide it. Trainer must assemble clean_x_GT for "
                    "every gen-step iter when dmd_context is 'GT' or "
                    "'mix' (mix_p > 0)."
                )
            if clean_x_GT.shape[1] != self.num_training_frames:
                raise RuntimeError(
                    f"clean_x_GT.shape[1]={clean_x_GT.shape[1]} must "
                    f"equal num_training_frames={self.num_training_frames}."
                )

            aug_t_full_gt = torch.full(
                (sc_clean_x.shape[0], self.num_training_frames),
                fill_value=int(self.clean_x_aug_t),
                device=device, dtype=torch.long,
            )
            gt_view = clean_x_GT.to(dtype=dtype, device=device)
            real_noise = torch.randn_like(gt_view)
            noised_gt = self.scheduler.add_noise(
                gt_view.flatten(0, 1),
                real_noise.flatten(0, 1),
                aug_t_full_gt.flatten(0, 1),
            ).unflatten(0, gt_view.shape[:2]).to(dtype=dtype)

            if self.dmd_context == "GT":
                # Pure GT: real_score sees noised_gt with full aug_t.
                sc_clean_x_real = noised_gt
                sc_aug_t_real = aug_t_full_gt
            else:
                # mix: uniform linear blend across all frames. Lower
                # variance than the per-chunk Bernoulli yinyang and
                # without the chunk-level step-discontinuities along
                # the time axis. aug_t stays 0 because the blended
                # frame is dominated by sc_clean_x (which is clean).
                p = float(self.dmd_context_mix_p)
                sc_clean_x_real = (1.0 - p) * sc_clean_x + p * noised_gt
                sc_aug_t_real = sc_aug_t

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
        z_actions_for_scoring: Optional[torch.Tensor] = None,
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

        # Pre-compute gt_z_per_slot from the dataset's per-frame
        # z_actions for the noisy_x scoring window — saves a CoTracker
        # forward on GT latents in the action-mode freeze block.
        # ``z_actions_for_scoring`` is [B, scoring_frames, A] (already
        # sliced to ``action_dims`` at dataset load time). Pool to
        # [B, n_slots, A] by mean across each ``num_frame_per_block``
        # slot — these per-frame z's were encoded by the SAME ss_vae
        # the teacher pipeline uses, so the cosine sim is meaningful
        # in the action-relevant subspace.
        gt_z_per_slot = None
        if (
            self.teacher_freeze_detect_enabled
            and self.teacher_freeze_mode == "action"
            and z_actions_for_scoring is not None
        ):
            npb = int(self.num_frame_per_block)
            B, F_act, A = z_actions_for_scoring.shape
            if F_act != scoring_frames:
                raise RuntimeError(
                    f"z_actions_for_scoring has F={F_act} but "
                    f"scoring_frames={scoring_frames}."
                )
            if F_act % npb != 0:
                raise RuntimeError(
                    f"z_actions_for_scoring frames ({F_act}) must be "
                    f"divisible by num_frame_per_block ({npb})."
                )
            n_slots = F_act // npb
            gt_z_per_slot = z_actions_for_scoring.reshape(
                B, n_slots, npb, A,
            ).mean(dim=2)  # [B, n_slots, A]

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
            gt_z_per_slot=gt_z_per_slot,
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
            # Anchor is the very first chunk in the rollout — always
            # cold-start (pure noise + full ladder), regardless of
            # ``warm_start_init``. Per user spec: "first chunk in the
            # rollout: unchanged".
            anchor_chunk, _, _ = pipe.generate_chunk_with_cache(
                noise=anchor_noise,
                current_start_frame=cf,
                requires_grad=False,
                prefer_cache_pred_in_output=False,
                gt_latents=None,  # no MAE on the anchor
                warm_start_init=False,
                **anchor_full_cond,
            )
        del anchor_full_cond
        anchor_chunk = anchor_chunk.detach()
        # Capture the anchor's clean pred to seed iter 1's warm-start
        # (when warm_start_init is on). The pipeline ALWAYS populates
        # ``_last_clean_pred`` after a successful rollout call (see
        # ``generate_chunk_with_cache``'s end-of-call stash); a None
        # value here signals an upstream contract violation. Failing
        # loud here prevents iter 1 from silently warm-starting from
        # the noisy exit-rung pred (anchor_chunk), which would
        # ``add_noise(noisy_x, ε, warm_start_t)`` and produce a sample
        # at ~2× the intended noise level — a quiet quality regression.
        if getattr(pipe, "_last_clean_pred", None) is None:
            raise RuntimeError(
                "ActionForcingTrainingPipeline did not populate "
                "_last_clean_pred after the anchor rollout. The "
                "warm-start carry would silently fall back to the "
                "noisy exit-rung pred, producing over-noised seeds "
                "for iter 1. Check that generate_chunk_with_cache "
                "stashes _last_clean_pred at end-of-call."
            )
        anchor_clean = pipe._last_clean_pred.detach()

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
            "previous_last_rung_chunk": None,  # last_rung view of full_chunk (Flash-DMD §3.3)
            # Warm-start carry: the prior call's last block's clean
            # pred. Initialised to the anchor's clean pred so iter 1
            # warm-starts from the anchor (when warm_start_init=True).
            # Updated to ``pipe._last_clean_pred`` after each
            # ``generate_chunk_with_cache`` call.
            "previous_clean_chunk": anchor_clean.detach(),
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
        # Deterministic-stride mode (slide-and-train): when set, every
        # call returns the same number of frames (capped to room +
        # chunk_size). No DDP broadcast needed because the value is
        # rank-invariant. Iter 1's chunk_size advance is still applied
        # by ``generate_next_chunk``'s previous_chunk=None branch.
        if self.streaming_force_new_frame_chunks > 0:
            forced = self.streaming_force_new_frame_chunks * npb
            capped = min(forced, room, chunk_size)
            capped = (capped // npb) * npb
            return max(npb, capped)
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
        compute_baseline_mae: bool = True,
        sync_exit_flags: bool = True,
        force_exit_step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Advance the open sequence by ``new_frames`` (∈ [min_new_frame,
        chunk_size], multiple of npb). Build a chunk_size-length
        ``full_chunk`` for DMD scoring = ``[previous_chunk[-overlap:],
        new_frames]`` (overlap = chunk_size - new_frames; 0 on first
        iter or when new_frames == chunk_size). Returns
        ``(full_chunk, info)`` where ``info`` carries the
        ``gradient_mask`` (True only on new frames), the per-chunk MAE,
        and the metadata DMD scoring needs.

        ``compute_baseline_mae`` (default True): when True, the pipeline
        computes ``baseline_last_chunk_mae`` via ``_compute_chunk_mae``,
        which fires a ``dist.all_reduce`` across DDP ranks. Set to False
        from per-rank-divergent call sites (= the slide-and-train
        helper, where each rank rolls a different number of chunks
        based on its own ride's MAE). With this False, no ``gt_chunk``
        is passed to ``generate_chunk_with_cache`` so ``_compute_chunk_mae``
        is not called and no DDP collective fires per slide. The slide
        helper computes its own per-rank MAE locally; the baseline
        telemetry is irrelevant on that path.

        ``force_exit_step`` (optional): when set, every rolling block
        in this call uses this exit-rung index. Skips
        ``generate_and_sync_list``'s per-call broadcast entirely. Used
        by the slide-and-train helper, which pre-broadcasts ONE index
        before the slide loop and reuses it across all slides — keeps
        cross-rank exit-rung lockstep while collapsing N per-slide
        broadcasts into 1 per training step.

        ``sync_exit_flags`` (default True): forwarded to
        ``generate_chunk_with_cache``. Only consulted when
        ``force_exit_step`` is None. With sync=True rank 0 samples and
        broadcasts the per-block exit indices; with sync=False each
        rank samples independently (per-rank gradient variance, but no
        DDP collective).
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
        # Built only when the caller wants baseline_last_chunk_mae
        # populated; passing ``gt_chunk=None`` to the pipeline below
        # short-circuits ``_compute_chunk_mae`` (and its DDP all_reduce).
        gt_chunk = None
        if compute_baseline_mae:
            gt_slice_lo = cf + s["current_length"]
            gt_slice_hi = gt_slice_lo + new_frames
            if gt_slice_hi <= s["ride_latents_window"].shape[1]:
                gt_chunk = s["ride_latents_window"][:, gt_slice_lo:gt_slice_hi]

        # Rebuild cond/uncond/clean_cond/clean_uncond FRESH for this
        # iter (see ``_streaming_build_cond_dicts`` for the why) and
        # stash on ``info`` so ``compute_*_loss_streaming`` reuses the
        # same dicts the rollout used — single grad_fn lifecycle per
        # iter, matching legacy ``_fwdbwd_one_step`` semantics.
        cond_dict, uncond_dict, clean_cond_dict, clean_uncond_dict = (
            self._streaming_build_cond_dicts()
        )

        # Two-grad-point rollout (Flash-DMD §3.3): when paper-aligned-
        # adv is on AND we're in a grad-active iter, run an extra
        # grad-active forward at the LAST rung in the same rollout.
        # The last-rung K/V is committed to the cache as-is (no
        # separate context_noise commit). The pipeline stashes the
        # last-rung pred on ``pipe._last_rung_output``; we re-stitch
        # it into a chunk_size-frame slab below for the GAN.
        dual_grad_rollout = bool(
            self.flash_dmd_paper_aligned_adv and requires_grad
        )
        # Warm-start init: when on, the iter's first block warm-starts
        # from the prior call's clean pred (= ``previous_clean_chunk``)
        # instead of pure noise. Subsequent blocks within the same
        # call (iter 1's chunk_size>npb path, multiple blocks per call)
        # warm-start from the preceding block's clean pred — handled
        # internally by ``generate_chunk_with_cache``.
        warm_start_init = bool(self.warm_start_init)
        initial_prev_clean = (
            s.get("previous_clean_chunk") if warm_start_init else None
        )
        new_chunk, denoised_t_from, denoised_t_to = pipe.generate_chunk_with_cache(
            noise=noise_chunk,
            current_start_frame=abs_frame_start,
            requires_grad=requires_grad,
            prefer_cache_pred_in_output=False,
            gt_latents=gt_chunk,
            sync_exit_flags=sync_exit_flags,
            force_exit_step=force_exit_step,
            dual_grad_rollout=dual_grad_rollout,
            warm_start_init=warm_start_init,
            warm_start_rung_idx=self.warm_start_rung_idx,
            initial_prev_clean=initial_prev_clean,
            **cond_dict,
        )
        # Pull the last-rung output (None when dual_grad_rollout=False).
        # Shape ``[B, new_frames, C, H, W]`` — the SAME npb-aligned slab
        # the pipeline rolled this iter.
        new_last_rung_chunk = pipe._last_rung_output if dual_grad_rollout else None
        # Capture this call's last-block clean pred for the NEXT call's
        # warm-start seed. ``_last_clean_pred`` is the post-finish-
        # denoise cache_pred of the rollout's final block (already
        # detached by the pipeline). Always update so flipping
        # warm_start_init mid-run picks up the latest clean.
        last_clean = getattr(pipe, "_last_clean_pred", None)
        if last_clean is not None:
            s["previous_clean_chunk"] = last_clean.detach()

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

        # Two-grad-point: build the chunk_size-length last-rung view
        # the same way as ``full_chunk`` above. Overlap region comes
        # from the prior iter's stashed last-rung chunk (detached, no
        # gradient), the new region from this iter's grad-active last
        # rung. On iter 1 (no previous_last_rung_chunk yet) we mirror
        # the ``full_chunk = new_chunk`` shortcut: overlap=0, so the
        # whole slab is the new last-rung chunk.
        full_last_rung_chunk = None
        if new_last_rung_chunk is not None:
            if overlap > 0:
                prev_last_rung = s.get("previous_last_rung_chunk")
                if prev_last_rung is None:
                    # Cold path: paper-aligned-adv was just enabled
                    # mid-sequence, OR iter 1 has overlap>0 (shouldn't
                    # happen — iter 1 forces overlap=0 above). Fall
                    # back to ``previous_chunk[:, -overlap:]``: it's
                    # detached and not the last-rung pred, but the
                    # G-side overlap region carries no gradient anyway
                    # so the disc just sees a slightly different
                    # texture for those frames.
                    overlap_slab = s["previous_chunk"][:, -overlap:]
                else:
                    overlap_slab = prev_last_rung[:, -overlap:]
                full_last_rung_chunk = torch.cat(
                    [overlap_slab.detach(), new_last_rung_chunk], dim=1,
                )
            else:
                full_last_rung_chunk = new_last_rung_chunk

        # gradient_mask: True only on new frames within the full_chunk.
        gradient_mask = torch.zeros_like(full_chunk, dtype=torch.bool)
        gradient_mask[:, overlap : overlap + new_frames] = True

        # Boundary VAE round-trip (Causal-Forcing/long_video parity, see
        # Causal-Forcing/long_video/model/base.py:155-167). When the iter
        # has overlap, ``full_chunk[:, 0:1]`` is the seam between the
        # previously-committed cache and the current scoring window —
        # analog of CF's "first frame of the trailing-21". Decode the
        # prior chunk + that boundary latent for VAE temporal context,
        # take the last pixel frame, and re-encode it to a fresh
        # single-frame latent on the image manifold. The replacement is
        # no_grad; the boundary lives at gradient_mask[:, 0] = False
        # already (overlap region) so this does NOT break gradient flow
        # on the new frames. Saved into ``s["previous_chunk"]`` below
        # so next iter's clean_x_self / overlap inherits the on-manifold
        # boundary instead of needing to re-anchor.
        if (
            self.boundary_vae_roundtrip
            and overlap > 0
            and prev_chunk_for_clean is not None
        ):
            with torch.no_grad():
                from einops import rearrange as _rearrange
                ctx_latents = torch.cat(
                    [prev_chunk_for_clean, full_chunk[:, 0:1]], dim=1,
                ).to(dtype)
                pixels = self.vae.decode_to_pixel(ctx_latents)
                last_frame_btchw = pixels[:, -1:, ...].to(dtype)
                last_frame_bcthw = _rearrange(
                    last_frame_btchw, "b t c h w -> b c t h w",
                )
                image_latent = self.vae.encode_to_latent(
                    last_frame_bcthw,
                ).to(dtype)
                full_chunk = torch.cat(
                    [image_latent, full_chunk[:, 1:]], dim=1,
                )

        # Save full_chunk as previous_chunk (detached) for the NEXT iter.
        s["previous_chunk"] = full_chunk.detach()
        # Mirror the same stash for the last-rung view so future iters'
        # overlap region carries last-rung pred (paper-aligned semantics
        # — the disc sees an apples-to-apples chunk_size slab of last-
        # rung pred, not a mix of exit-rung overlap + last-rung new).
        if full_last_rung_chunk is not None:
            s["previous_last_rung_chunk"] = full_last_rung_chunk.detach()
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
            # Two-grad-point last-rung view (Flash-DMD §3.3). None when
            # paper_aligned_adv is off OR this is a no-grad iter. The
            # generator-loss path stashes this onto
            # ``info["paper_aligned_x0_for_adv"]`` so the trainer's
            # ``_compute_r3gan_losses`` consumes it as the G-side fake.
            "last_rung_full_chunk": full_last_rung_chunk,
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
        # In "mix" mode the per-iter coin (sampled inside
        # ``_build_dmd_context_kwargs``) may resolve to GT, so we MUST
        # build clean_x_GT for every gen-step iter regardless of which
        # branch this iter ends up on. Building is cheap (slice +
        # add_noise on existing tensors).
        clean_x_GT = (
            self._streaming_build_clean_x_GT(info)
            if self.dmd_context in ("GT", "mix") else None
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

        # Eval-time stash for sample-video diagnostics. Mirrors the
        # ``clean_x`` views handed to fake_score / real_score (after
        # any ``add_noise`` round-trip in ``_build_dmd_context_kwargs``)
        # so a side-by-side decode shows exactly what each scorer
        # was conditioned on. No effect when not armed by the trainer.
        stash = getattr(self, "_dmd_eval_stash", None)
        if isinstance(stash, dict):
            stash["clean_x_fake"] = sc_clean_x.detach()
            stash["clean_x_real"] = (
                sc_clean_x_real.detach()
                if sc_clean_x_real is not None
                else sc_clean_x.detach()
            )
            stash["dmd_context"] = str(self.dmd_context)
            stash["clean_x_aug_t"] = int(self.clean_x_aug_t)
            stash["chunk"] = chunk.detach()
            # Raw clean-half z's (one z per chunk, broadcast to per-
            # latent by the dataset's ``encode_z_actions_window``).
            # Stashed for the video logger to overlay onto the
            # clean_x_real eval mp4 so the visual content can be
            # cross-checked against the actions the bidir scorer's
            # clean half was conditioned on.
            cf_state_eval = int(s["cf"])
            chunk_size_eval = int(s["chunk_size"])
            shift_eval = int(s["shift"])
            noisy_start_sdn_eval = int(
                s["current_length"]
                - info["new_frames"]
                - info["overlap"]
            )
            clean_lo_ride = cf_state_eval + noisy_start_sdn_eval - shift_eval
            clean_hi_ride = clean_lo_ride + chunk_size_eval
            ride_actions_eval = s["ride_actions_window"]
            if (
                clean_lo_ride >= 0
                and ride_actions_eval.shape[1] >= clean_hi_ride
            ):
                stash["clean_z_actions"] = (
                    ride_actions_eval[:, clean_lo_ride:clean_hi_ride].detach()
                )
                # Per-frame index annotations for the clean_x_real video
                # logger. ``ride_offset_s`` and ``motion_chunk_offset``
                # were stashed by the trainer at setup time. The clean
                # window's first dataset latent has zarr-absolute index
                #     ride_offset_s + clean_lo_ride
                # and lives in motion.npy at chunk
                #     motion_chunk_offset + (zarr_lat_idx // npb)
                # See utils/zarr_dataset.py:_motion_chunk_offset for the
                # offset definition.
                ride_offset_s = int(s.get("ride_offset_s", 0))
                motion_chunk_offset = int(s.get("motion_chunk_offset", 0))
                stash["clean_x_real_zarr_lat_lo"] = (
                    ride_offset_s + clean_lo_ride
                )
                stash["clean_x_real_motion_chunk_offset"] = motion_chunk_offset
                stash["clean_x_real_npb"] = shift_eval
                # Ride identifier (zarr file stem) so the overlay can
                # show which underlying recording the clean_x_real
                # window came from. Stem only — full paths are too long
                # to fit on a video frame.
                _zarr_path = s.get("zarr_path", "")
                if _zarr_path:
                    from pathlib import Path as _Path
                    stash["clean_x_real_zarr_name"] = _Path(_zarr_path).stem
                else:
                    stash["clean_x_real_zarr_name"] = ""

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

        # Pre-compute gt_z_per_slot for the noisy_x window from the
        # ride's per-frame z_actions (already sliced to action_dims at
        # dataset load time, stored on streaming_state). Pool by slot.
        gt_z_per_slot = None
        if (
            self.teacher_freeze_detect_enabled
            and self.teacher_freeze_mode == "action"
        ):
            npb = int(self.num_frame_per_block)
            ride_actions_window = s.get("ride_actions_window")
            if (
                ride_actions_window is not None
                and ride_actions_window.shape[1] >= chunk_hi
                and chunk_size_state % npb == 0
            ):
                z_acts_for_scoring = ride_actions_window[
                    :, chunk_lo:chunk_hi
                ]  # [B, F=chunk_size, A]
                B, F_act, A = z_acts_for_scoring.shape
                n_slots = F_act // npb
                gt_z_per_slot = z_acts_for_scoring.reshape(
                    B, n_slots, npb, A,
                ).mean(dim=2)

        # Flash-DMD timestep gate: when the trainer rolls a low-noise
        # iter, skip the entire DMD compute (no real_score / fake_score
        # forward, no DMD gradient). The student gets only the GAN
        # gradient this iter (gated trainer-side). Backward still works
        # because we return a zero-loss bound to ``chunk`` so the
        # autograd graph stays intact through the gen forward.
        # ``flash_dmd_t_min/max`` (info-supplied by the trainer's
        # regime gate) are HARD bounds for the DMD's per-frame
        # timestep distribution, applied unconditionally inside
        # ``_sample_dmd_timestep`` (i.e. they bypass the ``ts_schedule``
        # / ``ts_schedule_max`` config flags).
        flash_dmd_regime = info.get("flash_dmd_regime")  # "high"|"low"|None
        flash_dmd_t_min = info.get("flash_dmd_t_min")
        flash_dmd_t_max = info.get("flash_dmd_t_max")
        # Paper-aligned adv path bypasses the per-iter regime gate —
        # both DMD and adv fire every iter (paper Algorithm 1, lines
        # 13-16). The DMD path uses high-noise t (constrained via
        # the existing flash_dmd_t_min/max plumbing); the adv path
        # uses an EXTRA single gen forward at low-noise ˆt computed
        # below.
        if self.flash_dmd_paper_aligned_adv:
            flash_dmd_regime = None
        if flash_dmd_regime == "low":
            # Skip DMD entirely. Surface a sparse log dict (no
            # ``timestep`` / ``dmdtrain_gradient_norm`` keys) so wandb
            # plots aren't polluted with 0s on every low-noise iter —
            # the regime indicator (``flash_dmd_regime``) below tells
            # operators why a step is missing those metrics.
            dmd_loss = (chunk * 0.0).sum()
            dmd_log: Dict[str, Any] = {
                "flash_dmd_skipped_low_noise": 1.0,
            }
        else:
            # Dual-teacher routing: when a frozen merged-v14 teacher is
            # held, the DMD scoring forwards through THAT module (a clean
            # ``p_real`` model — no GT contamination, no moving target),
            # not through the LoRA-active ``self.real_score``. We swap
            # ``self.real_score`` ↔ ``self.real_score_frozen`` for the
            # duration of the DMD call and pass clean_x_real=None so the
            # frozen teacher sees only the self-view clean_x (proper DMD,
            # no GT in the conditioning either). The aux pass below uses
            # the LoRA teacher with the GT-mixed clean_x_real, gradient
            # to BOTH the LoRA AND the student.
            dual_teacher_active = self.real_score_frozen is not None
            if dual_teacher_active:
                _saved_real_score = self.real_score
                self.real_score = self.real_score_frozen
                try:
                    dmd_loss, dmd_log = self.compute_distribution_matching_loss(
                        image_or_video=chunk,
                        conditional_dict=cond_for_scoring,
                        unconditional_dict=uncond_for_scoring,
                        gradient_mask=gradient_mask_eff,
                        denoised_timestep_from=info.get("denoised_timestep_from"),
                        denoised_timestep_to=info.get("denoised_timestep_to"),
                        clean_x=sc_clean_x, aug_t=sc_aug_t,
                        gt_target=gt_target,
                        gt_z_per_slot=gt_z_per_slot,
                        clean_x_real=None, aug_t_real=None,
                        flash_dmd_t_min=flash_dmd_t_min,
                        flash_dmd_t_max=flash_dmd_t_max,
                    )
                finally:
                    self.real_score = _saved_real_score
            else:
                dmd_loss, dmd_log = self.compute_distribution_matching_loss(
                    image_or_video=chunk,
                    conditional_dict=cond_for_scoring,
                    unconditional_dict=uncond_for_scoring,
                    gradient_mask=gradient_mask_eff,
                    denoised_timestep_from=info.get("denoised_timestep_from"),
                    denoised_timestep_to=info.get("denoised_timestep_to"),
                    clean_x=sc_clean_x, aug_t=sc_aug_t,
                    gt_target=gt_target,
                    gt_z_per_slot=gt_z_per_slot,
                    clean_x_real=sc_clean_x_real, aug_t_real=sc_aug_t_real,
                    flash_dmd_t_min=flash_dmd_t_min,
                    flash_dmd_t_max=flash_dmd_t_max,
                )
            dmd_loss = dmd_loss * self.dmd_loss_weight
            if flash_dmd_regime == "high":
                dmd_log["flash_dmd_high_noise"] = 1.0
        if flash_dmd_regime is not None:
            dmd_log["flash_dmd_regime"] = (
                1.0 if flash_dmd_regime == "high" else 0.0
            )

        # Flash-DMD paper §3.3 Eq. 8-9: when paper_aligned_adv is on,
        # the rolling rollout's two-grad-point mode emitted a grad-
        # active LAST-RUNG forward in the SAME rollout (no extra gen
        # forward; the last-rung pred replaces the standard cache-
        # commit forward). ``info["last_rung_full_chunk"]`` carries
        # the chunk_size-aligned slab (overlap from prior iter's
        # last-rung pred + this iter's grad-active last-rung pred).
        # Grad path:
        #   adv_loss → last_rung_full_chunk → last-rung gen forward
        #     in rollout → gen params
        # The gradient does NOT traverse the high-noise denoising
        # rungs (those stayed no_grad); the paper's claim that adv
        # supervises only the gen's texture-refinement (low-noise)
        # behavior is preserved.
        if self.flash_dmd_paper_aligned_adv:
            last_rung_chunk = info.get("last_rung_full_chunk")
            if last_rung_chunk is None:
                raise RuntimeError(
                    "flash_dmd_paper_aligned_adv=True but "
                    "info['last_rung_full_chunk'] is None — the "
                    "rollout must run with dual_grad_rollout=True so "
                    "the pipeline emits the last-rung output."
                )
            info["paper_aligned_x0_for_adv"] = last_rung_chunk
            dmd_log["flash_dmd_paper_aligned_adv"] = 1.0
            dmd_log["flash_dmd_adv_low_noise_t"] = float(
                int(round(float(
                    self.inference_pipeline.denoising_step_list[-1]
                )))
            )

        # Auxiliary online-teacher pass (option 3 of the dual-teacher
        # design). When ``real_teacher_train_online`` is True we run a
        # SECOND real-side forward — through ``self.real_score`` (the
        # LoRA teacher, possibly the same module as DMD's when dual-
        # teacher is off) — with the GT-mixed clean_x context and a
        # noisy_input built from the student chunk (or GT, or a coin).
        # The flow loss ``(flow_pred - (ε - GT))²`` produces gradient:
        #   * To the LoRA params (trains the online teacher).
        #   * To the student chunk via the noise base, when the noise
        #     base is the chunk (input_source 'student' or coin lands
        #     student).
        # Folded into the gen-step total so a single backward
        # populates both. ``aux_teacher_loss_weight=0`` short-circuits
        # the pass entirely.
        aux_log: Dict[str, Any] = {}
        total_loss = dmd_loss
        if (
            self.real_teacher_train_online
            and self.aux_teacher_loss_weight > 0.0
        ):
            aux_loss, aux_log = self._compute_aux_teacher_loss_streaming(
                chunk=chunk,
                gradient_mask_eff=gradient_mask_eff,
                cond_for_scoring=cond_for_scoring,
                sc_clean_x_real=sc_clean_x_real,
                sc_aug_t_real=sc_aug_t_real,
                info=info,
            )
            if aux_loss is not None:
                total_loss = total_loss + self.aux_teacher_loss_weight * aux_loss

        for k in ("baseline_last_chunk_mae", "baseline_avg_rollout_mae", "last_chunk_mae", "mae_extension_count"):
            if k in info:
                dmd_log[k] = info[k]
        dmd_log["streaming_new_frames"] = float(info["new_frames"])
        dmd_log["streaming_current_length"] = float(info["current_length"])
        # Replacement for the (removed) ``dmd_context_branch_gt`` key.
        # Under the uniform-mix regime the GT contribution is constant
        # = ``dmd_context_mix_p`` on every chunk (no per-chunk dice),
        # so this surfaces the actual blend weight in wandb for any
        # dashboard/alert that previously tracked the per-iter
        # realized GT-chunk fraction.
        dmd_log["dmd_context_mix_p"] = (
            float(self.dmd_context_mix_p)
            if self.dmd_context == "mix" else (
                1.0 if self.dmd_context == "GT" else 0.0
            )
        )
        for k, v in aux_log.items():
            dmd_log[k] = v
        return total_loss, dmd_log

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
        # Surface MAE-extension metrics BEFORE any short-circuit so the
        # critic-side collapse gate (which reads
        # ``baseline_avg_rollout_mae`` off ``critic_log``) never goes
        # blind on the empty-mask early return below — same
        # telemetry-vs-loss-path independence Fix 1 enforced for the
        # gen step.
        for k in (
            "baseline_last_chunk_mae",
            "baseline_avg_rollout_mae",
            "last_chunk_mae",
            "mae_extension_count",
        ):
            if k in info:
                critic_log[k] = info[k]
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
        return denoising_loss, critic_log

    # ------------------------------------------------------------------
    # Auxiliary online-teacher pass at the gen step. Forwards the
    # LoRA-active ``self.real_score`` with grad on both the LoRA params
    # (trains the online teacher) AND the student chunk (when the
    # noise base is the chunk). Folded into ``compute_generator_loss_
    # streaming``'s total loss so a single gen.backward() populates
    # both gradients. The post-critic ``compute_real_teacher_loss_
    # streaming`` (below) stays alive — it provides the every-iter
    # LoRA training cadence; this function provides the every-gen-iter
    # student-gradient channel.
    # ------------------------------------------------------------------
    def _compute_aux_teacher_loss_streaming(
        self,
        *,
        chunk: torch.Tensor,
        gradient_mask_eff: torch.Tensor,
        cond_for_scoring: dict,
        sc_clean_x_real: Optional[torch.Tensor],
        sc_aug_t_real: Optional[torch.Tensor],
        info: Dict[str, Any],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """Aux pass: forward ``self.real_score`` (LoRA) on
        ``add_noise(<chunk or GT>, ε, t)`` with the GT-mixed clean_x
        context, return the FlowPredLoss against GT.

        ``chunk`` is the GENERATOR'S undetached output — the loss
        gradient flows to the student via the noise base (when
        ``real_teacher_input_source`` selects the student) AND to the
        LoRA params via the real_score forward.

        Returns ``(None, log)`` when the ride is too short for a
        gt_target slice (DDP-synced skip across all ranks).
        """
        s = self.streaming_state
        if sc_clean_x_real is None:
            # No GT in clean_x this iter (mix_p=0 or "self" mode).
            # Aux pass has no GT-pull contract — skip.
            return None, {}

        # GT target slice at the noisy_x positions (= 3 frames forward
        # of clean_x). Same indexing the teacher_freeze_detect block
        # uses; reused unchanged so the lag math is locked.
        chunk_size = s["chunk_size"]
        cf = s["cf"]
        noisy_start_sdn = (
            s["current_length"] - info["new_frames"] - info["overlap"]
        )
        ride_window = s.get("ride_latents_window")
        if ride_window is None:
            raise RuntimeError(
                "_compute_aux_teacher_loss_streaming: streaming_state "
                "missing ride_latents_window — cannot build gt_target."
            )
        chunk_lo = cf + noisy_start_sdn
        chunk_hi = chunk_lo + chunk_size
        # End-of-ride DDP-sync: same all_reduce(MAX) pattern as the
        # post-critic teacher step (model/dmd_action_forcing.py:
        # compute_real_teacher_loss_streaming). Without this, a rank
        # whose ride is too short would skip the LoRA forward while
        # peer ranks proceed → DDP AllReduce hang on the LoRA's
        # gradient sync. With dual-teacher mode the LoRA is also
        # DDP-wrapped (trainer real_score_ddp), so the hang is real.
        local_skip_int = 1 if ride_window.shape[1] < chunk_hi else 0
        if dist.is_available() and dist.is_initialized():
            flag = torch.tensor(
                [float(local_skip_int)],
                device=chunk.device, dtype=torch.float32,
            )
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            skip_all_ranks = bool(flag.item() > 0.5)
        else:
            skip_all_ranks = bool(local_skip_int)
        if skip_all_ranks:
            return None, {"aux_teacher_skipped_short_ride": 1.0}
        gt_target = ride_window[:, chunk_lo:chunk_hi].to(
            dtype=chunk.dtype, device=chunk.device,
        )

        # Sample fresh ε and t at the DMD timestep distribution.
        t = self._sample_dmd_timestep(
            batch_size=chunk.shape[0],
            num_frame=chunk.shape[1],
            denoised_timestep_from=info.get("denoised_timestep_from"),
            denoised_timestep_to=info.get("denoised_timestep_to"),
            device=chunk.device,
        )
        eps = torch.randn_like(chunk)

        # Decide noise base per ``real_teacher_input_source``. "mix"
        # picks a Bernoulli coin on rank 0 and broadcasts so every
        # rank uses the same source per iter.
        if self.real_teacher_input_source == "gt":
            use_gt = True
        elif self.real_teacher_input_source == "student":
            use_gt = False
        else:
            p_gt = float(self.real_teacher_input_mix_gt_p)
            if p_gt <= 0.0:
                use_gt = False
            elif p_gt >= 1.0:
                use_gt = True
            elif dist.is_available() and dist.is_initialized():
                if dist.get_rank() == 0:
                    coin = torch.tensor(
                        [1.0 if torch.rand(1).item() < p_gt else 0.0],
                        device=chunk.device, dtype=torch.float32,
                    )
                else:
                    coin = torch.zeros(
                        1, device=chunk.device, dtype=torch.float32,
                    )
                dist.broadcast(coin, src=0)
                use_gt = bool(coin.item() > 0.5)
            else:
                use_gt = bool(torch.rand(1).item() < p_gt)
        # IMPLICIT GRADIENT CHANNEL: the loss flows back to the
        # student generator THROUGH ``noise_base`` when it equals
        # ``chunk`` (which carries autograd from the rollout). When
        # ``noise_base = gt_target`` (sourced from the dataset's
        # ride_window) the path is dead-ended — gradient lands only
        # on the LoRA params. This is the documented mechanism by
        # which the aux pass "feeds gradient back to the student"
        # WITHOUT a direct ``MSE(student, GT)`` term. The
        # student-bound gradient comes from
        #   ∂loss/∂chunk = (∂loss/∂flow_pred) · (∂flow_pred/∂noisy_input) · α_t
        # with α_t the scheduler's add_noise scaling — i.e. a
        # learned-distillation gradient through the LoRA's flow
        # function, NOT a mean-seeking L2 to GT.
        noise_base = gt_target if use_gt else chunk
        if not use_gt:
            assert chunk.requires_grad, (
                "_compute_aux_teacher_loss_streaming: noise_base=chunk "
                "but chunk has no grad — student-gradient channel is "
                "broken. Caller must pass an undetached chunk from "
                "compute_generator_loss_streaming."
            )
        noisy_input = self.scheduler.add_noise(
            noise_base.flatten(0, 1),
            eps.flatten(0, 1),
            t.flatten(0, 1),
        ).unflatten(0, chunk.shape[:2])

        # Teacher forward — full grad on LoRA params (and on chunk
        # via noisy_input when use_gt=False).
        flow_pred, _x0 = self.real_score(
            noisy_image_or_video=noisy_input,
            conditional_dict=cond_for_scoring,
            timestep=t,
            clean_x=sc_clean_x_real,
            aug_t=sc_aug_t_real,
        )

        # Eval-time stash: surface the LoRA aux teacher's denoised x0
        # estimate for the sample-video logger. The DMD pass already
        # stashes ``pred_real`` (= the FROZEN teacher's CFG-extrapolated
        # x0); this adds ``pred_real_lora`` (= the LoRA aux teacher's
        # raw x0, no CFG since the aux pass runs cond-only). The
        # trainer's video logger iteration list includes both keys so
        # they decode side-by-side per sample step.
        stash = getattr(self, "_dmd_eval_stash", None)
        if isinstance(stash, dict):
            stash["pred_real_lora"] = _x0.detach()
            stash["aux_teacher_timestep"] = int(t.flatten()[0].item())
            stash["aux_teacher_input_was_gt"] = 1.0 if use_gt else 0.0

        gradient_mask_flat = gradient_mask_eff.flatten(0, 1)
        loss = self.denoising_loss_func(
            x=gt_target.flatten(0, 1),
            x_pred=None,
            noise=eps.flatten(0, 1),
            noise_pred=None,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=t.flatten(0, 1),
            flow_pred=flow_pred.flatten(0, 1),
            gradient_mask=gradient_mask_flat,
        )

        log: Dict[str, Any] = {
            "aux_teacher_loss": loss.detach(),
            "aux_teacher_input_was_gt": 1.0 if use_gt else 0.0,
        }
        return loss, log

    # ------------------------------------------------------------------
    # Online real_teacher: flow-loss training of the v14 LoRA against
    # GT video.
    # ------------------------------------------------------------------
    def compute_real_teacher_loss_streaming(
        self,
        chunk: torch.Tensor,
        info: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Train the v14 LoRA online: given a noisy input + the same
        clean_x mix the DMD scoring uses, predict a flow that points
        to GT.

        Math (v14-parity FlowPredLoss):
            noisy_input = scheduler.add_noise(<source>, ε, t)
            flow_pred   = real_score(noisy_input, cond, t,
                                     clean_x = mix(clean_x_GT,
                                                   clean_x_self),
                                     aug_t   = ...)
            loss        = (flow_pred - (ε - GT))²

        ``<source>`` is selected by ``self.real_teacher_input_source``:
          * "student": ``chunk_detached`` — amortized refinement
            target. Default; the bias decays as student → GT.
          * "gt": ``gt_target`` — proper DMD score matching for
            ``p_real(x_t)``. No curriculum bias.
          * "mix": per-iter Bernoulli(``real_teacher_input_mix_gt_p``);
            DDP-synced from rank 0. True → GT, False → student.

        The LOSS TARGET is always GT regardless of source — the source
        only decides what the noise is added to.

        Both halves of the clean_x context are CLEAN latents
        (clean_x_aug_t=0). The only place noise enters is the
        noisy_input half. v14's training contract preserved.

        Reuses the chunk + info from the critic step — the streaming KV
        cache state is NOT advanced again. ``chunk`` is detached so the
        teacher's backward never reaches the student.
        """
        s = self.streaming_state
        # Strict contract — same as gen / critic streaming methods.
        per_iter_mask = info.get("gradient_mask")
        if per_iter_mask is None:
            raise RuntimeError(
                "compute_real_teacher_loss_streaming: info['gradient_mask'] "
                "must be a tensor (built by ``_streaming_generate_chunk_with_grad``)."
            )
        last_chunk_mask = self._dmd_score_grad_mask(
            chunk.shape, chunk.device,
        )
        # ``per_iter_mask`` is True only on new frames; we apply that
        # here so the teacher only learns on the same frames the
        # scorer scores. AND with last-chunk boundary mask.
        gradient_mask_eff = per_iter_mask & last_chunk_mask

        chunk_detached = chunk.detach()

        clean_x_self = self._streaming_build_clean_x_self(chunk_detached, info)
        clean_x_GT = self._streaming_build_clean_x_GT(info)
        clean_cond, clean_uncond = self._streaming_clean_cond_slice(info)
        cond_for_scoring, uncond_for_scoring = self._streaming_noisy_cond_slice(info)

        # Detach all conditional dicts. ``info["conditional_dict"]`` /
        # ``info["unconditional_dict"]`` carry autograd state from the
        # per-iter action-embedding projection inside generate_next_chunk
        # (action MLP outputs etc.). The trainer just called
        # ``critic_loss.backward()`` on the line before this one, which
        # already consumed those saved tensors — reusing them here
        # raises "Trying to backward through the graph a second time".
        # The teacher only learns its own LoRA params; cond is a pure
        # input — no grad needs to flow back through it.
        def _detach_dict(d):
            if d is None:
                return None
            return {
                k: (v.detach() if torch.is_tensor(v) else v)
                for k, v in d.items()
            }
        clean_cond = _detach_dict(clean_cond)
        clean_uncond = _detach_dict(clean_uncond)
        cond_for_scoring = _detach_dict(cond_for_scoring)
        uncond_for_scoring = _detach_dict(uncond_for_scoring)
        (
            _sc_clean_x_fake,         # = clean_x_self; unused on real side
            _sc_aug_t_fake,           # = zeros; unused on real side
            sc_clean_x_real,          # = uniform GT/self blend
            sc_aug_t_real,            # = aug_t (uniform 0 in mix mode)
            cond_for_scoring,         # action streams at noisy positions
            _uncond_for_scoring,      # unused (no CFG on training step)
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

        # GT target slice at the noisy_x positions (= 3 frames forward
        # of clean_x; v14's target geometry). Same indexing the
        # teacher_freeze_detect block in compute_distribution_matching_loss
        # uses for `gt_target` — reused unchanged so the lag math is
        # locked.
        chunk_size = s["chunk_size"]
        cf = s["cf"]
        noisy_start_sdn = (
            s["current_length"] - info["new_frames"] - info["overlap"]
        )
        ride_window = s.get("ride_latents_window")
        if ride_window is None:
            # Streaming contract should always provide this; surface
            # loud rather than silently zero out.
            raise RuntimeError(
                "compute_real_teacher_loss_streaming: streaming_state "
                "missing ride_latents_window — cannot build gt_target."
            )
        chunk_lo = cf + noisy_start_sdn
        chunk_hi = chunk_lo + chunk_size
        # End-of-ride detection. The check is rank-local — different
        # ranks pull different rides — so we MUST DDP-sync before
        # acting on it. Two failure modes if we don't:
        #   1. The previous return path was ``(chunk_detached * 0.0)``
        #      which has no ``grad_fn`` (chunk came from generate_next_
        #      chunk(requires_grad=False)), so .backward() raises.
        #   2. Even with a grad-bearing zero, a RANK-LOCAL skip means
        #      the skipping rank never forwards through the DDP-wrapped
        #      real_score, while peer ranks do — peer backward fires
        #      AllReduce that hangs forever waiting for the skipper.
        # Fix: all_reduce(MAX) the skip flag so EVERY rank skips
        # together when ANY rank can't build a gt_target this iter.
        # Trainer must then ALSO skip the .backward() call (the
        # returned zero is a leaf with no grad — a defensive no-op,
        # not a backward-able loss).
        local_skip_int = 1 if ride_window.shape[1] < chunk_hi else 0
        if dist.is_available() and dist.is_initialized():
            flag = torch.tensor(
                [float(local_skip_int)],
                device=chunk.device, dtype=torch.float32,
            )
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            skip_all_ranks = bool(flag.item() > 0.5)
        else:
            skip_all_ranks = bool(local_skip_int)
        if skip_all_ranks:
            # Non-grad zero. Trainer reads ``REAL_TEACHER_SKIP_KEY``
            # from the model class to decide whether to call
            # ``.backward()``; the constant lives on the class
            # (``ActionForcingDMD.REAL_TEACHER_SKIP_KEY``) so a rename
            # propagates to both sides without a string-match break.
            zero = torch.zeros(
                (), device=chunk.device, dtype=chunk.dtype,
            )
            return zero, {self.REAL_TEACHER_SKIP_KEY: 1.0}
        gt_target = ride_window[:, chunk_lo:chunk_hi].to(
            dtype=chunk.dtype, device=chunk.device,
        )

        # Sample fresh ε and t at the DMD timestep distribution.
        # ``denoised_timestep_from/to`` come from info if the pipeline
        # populated them (they gate the timestep range when ts_schedule
        # is on); we use the same hooks the critic step does.
        t = self._sample_dmd_timestep(
            batch_size=chunk.shape[0],
            num_frame=chunk.shape[1],
            denoised_timestep_from=info.get("denoised_timestep_from"),
            denoised_timestep_to=info.get("denoised_timestep_to"),
            device=chunk.device,
        )
        eps = torch.randn_like(chunk_detached)

        # Decide the noise base for THIS iter based on the input-
        # source knob. "mix" picks a Bernoulli coin on rank 0 and
        # broadcasts so every rank uses the same source per iter
        # (DDP grads need matched conditioning).
        if self.real_teacher_input_source == "gt":
            use_gt = True
        elif self.real_teacher_input_source == "student":
            use_gt = False
        else:
            p_gt = float(self.real_teacher_input_mix_gt_p)
            if p_gt <= 0.0:
                use_gt = False
            elif p_gt >= 1.0:
                use_gt = True
            elif dist.is_available() and dist.is_initialized():
                if dist.get_rank() == 0:
                    coin = torch.tensor(
                        [1.0 if torch.rand(1).item() < p_gt else 0.0],
                        device=chunk.device, dtype=torch.float32,
                    )
                else:
                    coin = torch.zeros(
                        1, device=chunk.device, dtype=torch.float32,
                    )
                dist.broadcast(coin, src=0)
                use_gt = bool(coin.item() > 0.5)
            else:
                use_gt = bool(torch.rand(1).item() < p_gt)
        noise_base = gt_target if use_gt else chunk_detached
        noisy_input = self.scheduler.add_noise(
            noise_base.flatten(0, 1),
            eps.flatten(0, 1),
            t.flatten(0, 1),
        ).unflatten(0, chunk.shape[:2])

        # Teacher forward — full grad on the LoRA params.
        flow_pred, _x0 = self.real_score(
            noisy_image_or_video=noisy_input,
            conditional_dict=cond_for_scoring,
            timestep=t,
            clean_x=sc_clean_x_real,
            aug_t=sc_aug_t_real,
        )

        # FlowPredLoss(x=GT, noise=ε, flow_pred=teacher_out).
        # err = (flow_pred - (ε - GT))²; gradient_mask zeroes the
        # overlap region + last-chunk boundary.
        gradient_mask_flat = gradient_mask_eff.flatten(0, 1)
        loss = self.denoising_loss_func(
            x=gt_target.flatten(0, 1),
            x_pred=None,                       # unused by FlowPredLoss
            noise=eps.flatten(0, 1),
            noise_pred=None,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=t.flatten(0, 1),
            flow_pred=flow_pred.flatten(0, 1),
            gradient_mask=gradient_mask_flat,
        )

        log: Dict[str, Any] = {
            "real_teacher_loss": loss.detach(),
            "real_teacher_timestep": t.detach(),
            "real_teacher_input_was_gt": 1.0 if use_gt else 0.0,
        }
        return loss, log

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
