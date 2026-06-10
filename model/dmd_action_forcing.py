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
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import swap_tensors as _swap_tensors
from torch.utils.checkpoint import checkpoint as _ckpt

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
        # conditioning streams.
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

        # DMD context is hardcoded to the "self" view (= student-rolled
        # 21-frame window shifted back by 1 chunk). real_score and
        # fake_score both see the same clean_x. No knob, no GT / mix
        # alternatives.

        # ``dmd_asymmetric_scoring_enabled``: v14-faithful DMD on a
        # single student chunk against 21 GT clean frames. When True,
        # the streaming gen path feeds the scorers:
        #   clean_x = 21 GT frames at ride positions preceding the
        #             newly rolled chunk (NOT the self-view).
        #   noisy_x = the new student chunk only (3 frames).
        # The scorers' forward goes from 42-frame TF (21+21) down to a
        # 24-frame TF (21+3), which is the source of the memory win.
        # The teacher (v14) sees its in-distribution clean half (all
        # GT, no student-rolled drift) and a single noisy chunk to
        # denoise. The DMD MSE is computed on the 3 new frames only.
        self.dmd_asymmetric_scoring_enabled = bool(
            getattr(args, "dmd_asymmetric_scoring_enabled", True)
        )

        # ``dmd_42f_enabled`` (gt-context fix): keep v14's EXACT training
        # shape — 21 clean + 21 noisy at tf_rope_offset=npb — but make the
        # noisy half mostly GT (18 GT context frames + the 3 new student
        # frames), masking the DMD gradient to the 3 student frames. This
        # restores the in-distribution noisy-half SHAPE that the asymmetric
        # (3-frame) path broke, so pred_real/pred_fake denoise cleanly.
        # Takes precedence over ``dmd_asymmetric_scoring_enabled``. See
        # ``_build_42f_scoring_inputs``.
        self.dmd_42f_enabled = bool(
            getattr(args, "dmd_42f_enabled", False)
        )
        # ``dmd_42f_2chunk``: roll TWO student chunks and supervise the
        # FIRST (which then sits at the reliable mid-window slot RoPE
        # [18,21) with a clean counterpart), masking the newest chunk at
        # the structurally-OOD slot [21,24). Mitigates the collapse the
        # 1-chunk 42f hit (gradient taken at the unreliable last slot).
        # Only consulted when ``dmd_42f_enabled`` is True.
        self.dmd_42f_2chunk = bool(getattr(args, "dmd_42f_2chunk", False))
        # ``dmd_42f_num_chunks`` (>=1): roll this many student chunks into
        # the noisy half and supervise all but the newest (which is masked
        # at the OOD slot). 1 = original 42f; 2 = the 2-chunk fix (supervise
        # 1); 3 = supervise 2 reliable chunks (denser DMD signal). Overrides
        # ``dmd_42f_2chunk`` when > 0.
        self.dmd_42f_num_chunks = int(getattr(args, "dmd_42f_num_chunks", 0))
        # ``dmd_42f_seed_last``: replace the masked (newest) STUDENT
        # chunk(s) in the noisy half with GT, so the supervised student
        # chunk is scaffolded by clean GT on BOTH temporal sides (GT
        # context before + GT after) instead of a drifted student chunk
        # after it. e.g. ns=2 -> noisy = [5 GT | student | 1 GT].
        self.dmd_42f_seed_last = bool(getattr(args, "dmd_42f_seed_last", False))
        # ``dmd_42f_gt_after_chunks`` (>0): number of chunks placed AFTER
        # the supervised student chunk, overriding the default (ns -
        # num_sup). Moves the supervised chunk further from the
        # structurally-OOD newest slot RoPE [21,24). e.g. with one
        # supervised chunk, =1 -> 2nd-from-last (RoPE [18,21)); =2 ->
        # 3rd-from-last (RoPE [15,18)). ``seed_last`` sets the after-block
        # CONTENT: True -> GT scaffold (noisy = [4 GT | S | 2 GT]); False
        # -> the student's own rolled chunks (noisy = [4 GT | S | s s]).
        self.dmd_42f_gt_after_chunks = int(
            getattr(args, "dmd_42f_gt_after_chunks", 0)
        )
        # ``dmd_42f_fix_clean_counterpart``: replace the GT clean-half
        # counterpart of each STUDENT noisy chunk with the student content
        # (detached) so the teacher's bidirectional clean/noisy halves
        # AGREE at each world position (v14's contract). Kills the
        # GT-vs-student conflict the bidir scorer otherwise sees. See
        # ``_build_42f_scoring_inputs``.
        self.dmd_42f_fix_clean_counterpart = bool(
            getattr(args, "dmd_42f_fix_clean_counterpart", False)
        )
        # ``dmd_42f_clean_self``: fill the clean-half AFTER-region (the
        # supervised chunk's clean counterpart + the gt_after slots, i.e.
        # clean[n_ctx+npb : N]) with the student's OWN rolled chunks taken
        # straight from ``chunk`` (detached) — independent of whether the
        # NOISY after-region is GT (seed_last) or student. Unlike
        # ``fix_clean_counterpart`` (which copies the noisy half's content
        # for the supervised counterpart only), this always sources the
        # student rollout and covers the whole after-region. Used for the
        # clean_self / all_self OOD-ablation configs.
        self.dmd_42f_clean_self = bool(
            getattr(args, "dmd_42f_clean_self", False)
        )
        # ``dmd_42f_rand_sup_slot`` (seed_last=false only): instead of
        # always grading the FIRST after-student chunk (the slot right
        # after the GT context), randomly pick WHICH student chunk carries
        # the DMD gradient among those that still have a clean counterpart
        # (every noisy slot except the last, counterpart-less one). The
        # chosen chunk is graph-on; the rest of the student region stays
        # detached context, exactly like the fixed-slot path. Step-seeded
        # so all ranks agree and the choice sweeps over training. Spreads
        # the supervision across rollout depth (slots 4/5/6 for the midp
        # gt_after=3 layout). See ``_build_42f_scoring_inputs``.
        self.dmd_42f_rand_sup_slot = bool(
            getattr(args, "dmd_42f_rand_sup_slot", False)
        )
        # ``dmd_42f_allsup`` (seed_last=false only): DMD-supervise EVERY
        # student after-chunk that has a clean counterpart (noisy slots
        # [n_ctx, N-npb)) instead of one (fixed/random) slot — e.g. slots
        # 4,5,6 for gt_after=3. The last slot (no counterpart) stays
        # detached. See ``_build_42f_scoring_inputs``.
        self.dmd_42f_allsup = bool(
            getattr(args, "dmd_42f_allsup", False)
        )
        # ``dmd_42f_rolling_sup_new`` (phase-2 rolling): on streaming rolls
        # k>=2 (info["overlap"] > 0) anchor the 42f noisy window on the NEW
        # frames instead of the window head. Without this, the supervised
        # block ``chunk[:, :sup_span]`` lands on the DETACHED overlap
        # frames on every k>=2 roll -> the DMD "loss" backwards into
        # nothing (silent zero-gradient). With it, the layout becomes
        #   [GT ctx | new chunks (graph-on, supervised) | GT future chunk]
        # where the GT future chunk (ride GT at the next abs position)
        # fills the structurally-OOD newest slot RoPE [N, N+npb) (masked
        # from gradient, content only). All new chunks then sit at
        # counterpart-valid slots -> every rolled chunk is DMD-supervised
        # exactly once when the per-roll advance equals the supervised
        # count (streaming_force_new_frame_chunks = num supervised chunks).
        # Iter 1 (overlap=0) keeps the standard layout unchanged.
        self.dmd_42f_rolling_sup_new = bool(
            getattr(args, "dmd_42f_rolling_sup_new", False)
        )

        # ``dmd_42f_gt_anchor``: make the +npb leading-anchor chunk (rolled
        # in ``setup_sequence`` before the first supervised chunk) a CLEAN
        # GT chunk (seed-prefilled) instead of a student rollout. The
        # student anchor writes a drifted student chunk into the KV slot
        # the supervised chunks attend to, while the 42f teacher scores
        # that slot with GT (a generate-vs-score mismatch); a GT anchor
        # removes the mismatch and gives the supervised chunks clean GT
        # context (= the 7-chunk inference eval). Default OFF (legacy
        # student anchor). See ``setup_sequence``. NOTE: candidate to
        # become the default if it validates.
        self.dmd_42f_gt_anchor = bool(
            getattr(args, "dmd_42f_gt_anchor", False)
        )

        # ``dmd_only_first_chunk_per_ride``: pair with
        # ``max_rolls_per_ride > 1`` to roll the student N causal
        # chunks per ride but only DMD-supervise the first. Subsequent
        # rolls within the ride still advance the KV cache (the trainer
        # call to ``generate_next_chunk`` is unchanged) so the student
        # gets practice at long-horizon autoregressive generation, but
        # only chunk 0's prediction carries a learning signal — keeps
        # the v14 scorer fed an in-distribution single-chunk noisy_x
        # at the same position in every ride. Default ``False`` so
        # legacy single-roll configs behave unchanged.
        self.dmd_only_first_chunk_per_ride = bool(
            getattr(args, "dmd_only_first_chunk_per_ride", False)
        )

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
        # Online aux teacher LoRA rank / alpha / dropout. Configurable
        # SEPARATELY from v14's rank: when ``real_teacher_train_online``
        # is True we MERGE v14's rank-256 LoRA into the base model
        # (folds v14's learned weights into the WAN base permanently),
        # then apply a FRESH trainable adapter at ``aux_teacher_lora_
        # rank`` (default 32) on top. Reducing rank cuts LoRA params +
        # Adam state + grad buffers by ``256/rank`` and saves ~1 GB at
        # rank 32. ``alpha`` should track rank to preserve the LoRA
        # scaling factor ``α/r`` (default ``α=rank`` → ``α/r=1.0``;
        # changing only one of them is almost always a bug). When
        # ``real_teacher_train_online=False`` these knobs are unused
        # (v14 is still merged in by the legacy path).
        self.aux_teacher_lora_rank = int(
            getattr(args, "aux_teacher_lora_rank", 32)
        )
        self.aux_teacher_lora_alpha = float(
            getattr(args, "aux_teacher_lora_alpha", self.aux_teacher_lora_rank)
        )
        self.aux_teacher_lora_dropout = float(
            getattr(args, "aux_teacher_lora_dropout", 0.0)
        )
        if self.aux_teacher_lora_rank <= 0:
            raise ValueError(
                f"aux_teacher_lora_rank={self.aux_teacher_lora_rank} "
                f"must be > 0"
            )

        # v28G_6+: optional LoRA wrap on fake_score.model. Default OFF
        # = legacy full-fine-tune of fake_score (all params trainable).
        # When enabled, peft wraps fake_score.model with the
        # ``fake_score_lora_*`` adapter config; only LoRA params are
        # marked trainable, the base WAN stays frozen. This collapses
        # fake_score's per-rank optimizer + gradient memory from
        # ~15 GB (full bf16 params + grad + Adam state on 1.3B WAN)
        # down to ~3 GB (frozen base + thin LoRA), freeing memory for
        # larger LADD pair budgets / gradient-chunk windows.
        # ``head_alt`` (v21 fake_alt) is explicitly re-enabled to
        # requires_grad=True after the peft wrap so the alt loss
        # gradient still flows; LoRA itself sits on QKV/O attention
        # projections only, separate from head_alt's CausalHead.
        self.fake_score_lora_enabled = bool(
            getattr(args, "fake_score_lora_enabled", False)
        )
        self.fake_score_lora_rank = int(
            getattr(args, "fake_score_lora_rank", 32)
        )
        self.fake_score_lora_alpha = float(
            getattr(args, "fake_score_lora_alpha", self.fake_score_lora_rank)
        )
        self.fake_score_lora_dropout = float(
            getattr(args, "fake_score_lora_dropout", 0.0)
        )
        if self.fake_score_lora_rank <= 0:
            raise ValueError(
                f"fake_score_lora_rank={self.fake_score_lora_rank} "
                f"must be > 0"
            )

        # Phased DMD K-LoRA on the student. When enabled, K separate
        # named peft adapters ('rung_0'..'rung_{K-1}') are added to
        # ``generator.model``. The active adapter is selected per
        # denoise step via ``set_adapter()`` (see pipeline). All K
        # share the frozen WAN base; only LoRA params are trainable
        # when ``freeze_base=True``. Default OFF = legacy full-FT.
        self.student_phase_lora_enabled = bool(
            getattr(args, "student_phase_lora_enabled", False)
        )
        self.student_phase_lora_K = int(
            getattr(args, "student_phase_lora_K", 0)
        )
        self.student_phase_lora_rank = int(
            getattr(args, "student_phase_lora_rank", 32)
        )
        self.student_phase_lora_alpha = float(
            getattr(args, "student_phase_lora_alpha", self.student_phase_lora_rank)
        )
        self.student_phase_lora_dropout = float(
            getattr(args, "student_phase_lora_dropout", 0.0)
        )
        self.student_phase_lora_freeze_base = bool(
            getattr(args, "student_phase_lora_freeze_base", True)
        )
        # When True, also create a dedicated ``rung_flash`` adapter
        # (separate from the K ODE rungs). The pipeline routes the
        # Flash-DMD t=60 forward to this adapter so the K ODE rungs
        # don't absorb flash GAN gradient (clean per-rung
        # specialization on the ODE chain).
        self.student_phase_lora_flash_adapter_enabled = bool(
            getattr(args, "student_phase_lora_flash_adapter_enabled", False)
        )
        if self.student_phase_lora_enabled:
            if self.student_phase_lora_K <= 0:
                raise ValueError(
                    f"student_phase_lora_enabled=True requires "
                    f"student_phase_lora_K > 0; got "
                    f"{self.student_phase_lora_K}"
                )
            if self.student_phase_lora_rank <= 0:
                raise ValueError(
                    f"student_phase_lora_rank="
                    f"{self.student_phase_lora_rank} must be > 0"
                )
        self._student_phase_lora_names: list = []
        self._active_phase_rung_idx: int = 0

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

        # ``real_score_ema_weight``: target-network EMA on the LoRA
        # adapter of ``self.real_score``. When > 0, the DMD scoring
        # path (compute_distribution_matching_loss) forwards through a
        # slow-moving EMA copy of the LoRA weights instead of the live
        # LoRA. The aux teacher loss and the standalone real_teacher
        # loss continue to forward through the live LoRA (so gradient
        # still trains it). Mirror of fake_score_ema_weight on the
        # other side of the DMD subtraction; same RL target-network
        # idea (DQN/DDPG/TD3).
        #
        # Mutually exclusive with dmd_frozen_teacher_pass_enabled: if
        # both are on, the frozen merged-v14 teacher path wins (it's
        # checked first in compute_generator_loss_streaming).
        #
        # Memory cost: a dict of cloned LoRA-adapter tensors (~tens of
        # MB for rank-32), held on each rank. NOT a full module copy.
        # The EMA is applied IN PLACE on the existing real_score's
        # LoRA params during the DMD forward via a swap-and-restore
        # context manager, so no second WAN transformer is allocated.
        #
        # Value range: 0.0 (default = disabled) to <1.0. Typical
        # values: 0.95-0.99. With teacher_cadence='fake' the EMA
        # update fires after EVERY LoRA optimizer step (5x per outer
        # iter), so 0.99 → half-life ~70 LoRA-steps ~14 outer steps.
        self.real_score_ema_weight = float(
            getattr(args, "real_score_ema_weight", 0.0)
        )
        if not (0.0 <= self.real_score_ema_weight < 1.0):
            raise ValueError(
                "real_score_ema_weight must be in [0, 1); got "
                f"{self.real_score_ema_weight!r}."
            )
        # Lazy-initialised dict of {param_name -> EMA tensor} for the
        # LoRA adapter of self.real_score. None until the first call
        # to ``ema_update_real_score_lora`` — first call snapshots the
        # current LoRA state and exits without an EMA update (so the
        # very first DMD-with-EMA pass sees an EMA == live).
        self._real_score_ema_lora_state: Optional[Dict[str, torch.Tensor]] = None
        # Diagnostic flag: emit a one-shot audit log on the first EMA
        # update (mirror of _fake_ema_audit_done in the trainer) so a
        # silent param-name drift is loud rather than invisible.
        self._real_score_ema_audit_done: bool = False
        # Diagnostic scalar: relative L2 distance between live LoRA and
        # EMA-LoRA after each EMA update, ‖live - ema‖₂ / ‖live‖₂.
        # Trainer reads this and surfaces it as
        # ``train/real_score_ema_rel_l2`` so wandb shows the gap.
        # Saturating-near-zero = EMA isn't buying decoupling (live not
        # moving); growing-unboundedly = live diverging from EMA in a
        # way that signals upcoming instability. None until the first
        # post-init EMA update.
        self._real_score_ema_rel_l2: Optional[float] = None

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

        # Aux + standalone teacher noisy_input is ALWAYS noised(GT) —
        # the v14 training contract. The "student" / "mix" / "blend"
        # alternatives previously selectable via ``real_teacher_input_
        # source`` have been removed: they leaked student-distribution
        # structure into the teacher's score and were observed to break
        # training. The loss target is also GT, so the teacher learns
        # ``∇ log p_real(x_t)`` exactly with no curriculum bias.

        # Stash for the EMA-swapped CFG-extrapolated teacher x0
        # prediction produced by the gen-step DMD scoring pass.
        # Detached, overwritten each gen-step DMD pass; None until
        # first set. Consumed by downstream alt-head plumbing.
        self._latest_pred_real_image: Optional[torch.Tensor] = None

        # ===== v21 fake-score alt head =====
        # When enabled, builds a second small projection head on
        # ``self.fake_score.model`` (CausalWanModel). The alt head
        # shares the WAN backbone forward pass but has a separate
        # output projection trained to predict ``ema_real_x0`` (the
        # EMA-swapped real_score's x0 estimate) instead of
        # ``pred_image`` (what the main head predicts).
        #
        # Training contract (mirrors fake_score's main critic loss
        # exactly — same flow-space FlowPredLoss, same noised input):
        #     flow_alt = fake_score.head_alt(features)
        #     target = critic_noise - pred_real_EMA   (= ε - ema_real_x0)
        #     loss   = FlowPredLoss(flow_alt, target)
        # The alt head's input is detached inside CausalWanModel.forward,
        # so its loss propagates only into alt-head params — backbone
        # supervision stays with the main critic loss.
        #
        # Application contract (consumed in
        # ``_compute_aux_teacher_loss_streaming``):
        #     noisy_gt = add_noise(gt_target, ε_aux, t_aux)
        #     _, x0_alt = fake_score.alt(noisy_gt, t_aux, cond)   # no_grad
        #     causal_AR_GT = x0_alt
        #     noisy_for_real = add_noise(causal_AR_GT, ε_aux, t_aux)
        #     real_score(noisy_for_real) → flow_pred
        #     target_flow = ε_aux - gt_target_TRUE  (still true GT)
        # The teacher learns to denoise AR-noise back to true GT.
        self.fake_alt_head_enabled = bool(
            getattr(args, "fake_alt_head_enabled", False)
        )
        # Step at which alt-head training begins. Before this step,
        # alt loss is not computed (waiting for ema_real to diverge
        # from live real_score so the target is non-trivial).
        self.fake_alt_head_start_step = int(
            getattr(args, "fake_alt_head_start_step", 0)
        )
        # Step at which the AR-aux application begins consuming the
        # alt head's output. Before this step, the aux teacher pass
        # runs as if alt were disabled (clean GT, Gaussian-only noise).
        self.fake_alt_apply_start_step = int(
            getattr(args, "fake_alt_apply_start_step", 0)
        )
        # ===== v24 alt-head target source =====
        # ``ema_real_x0``  (v23 default): alt head trained to predict
        #     the EMA-swapped real_score's x0 estimate. Target is built
        #     via an extra no_grad EMA-real_score forward per critic
        #     iter. Marks ``_real_score_ema_swap`` and related EMA
        #     plumbing as live consumers.
        # ``rollout2_student`` (v24): alt head trained to predict the
        #     student's x0 from a NOISIER auxiliary rollout that uses
        #     ``fake_alt_rollout2_num_seed_chunks`` seed chunks (vs the
        #     normal ``dmd_context_clean_frames/npb`` count). That
        #     rollout has 1 more AR step of drift, so the alt head
        #     learns the worst-case causal-AR noise distribution.
        #     ``real_score_ema`` infrastructure becomes a no-op
        #     consumer under this mode — marked deprecated.
        self.fake_alt_target_mode = str(
            getattr(args, "fake_alt_target_mode", "ema_real_x0")
        ).lower().strip()
        if self.fake_alt_target_mode not in (
            "ema_real_x0", "rollout2_student",
        ):
            raise ValueError(
                "fake_alt_target_mode must be 'ema_real_x0' or "
                f"'rollout2_student'; got {self.fake_alt_target_mode!r}."
            )
        # Number of seed chunks for the v24 rollout-2 prebuild. Must be
        # < ``dmd_context_clean_frames / num_frame_per_block`` so
        # rollout 2 has 1 fewer seed chunk than rollout 1 (= 1 extra AR
        # step of drift). Default 2 (= 6 latent frames seed), matching
        # the canonical 3-seed-chunks rollout-1 configuration.
        self.fake_alt_rollout2_num_seed_chunks = int(
            getattr(args, "fake_alt_rollout2_num_seed_chunks", 2)
        )
        if self.fake_alt_rollout2_num_seed_chunks < 1:
            raise ValueError(
                "fake_alt_rollout2_num_seed_chunks must be >= 1."
            )
        # ===== v25 critic clean_x source =====
        # ``self`` (default, v23 behavior): fake_score's critic step
        #     trains on noised STUDENT x0 (``chunk``) with TF context
        #     drawn from the student's own previous frames. Loss target
        #     = ``chunk`` (student's own x0).
        # ``gt_causal_ar``: mirrors the v21 aux-teacher pattern but on
        #     fake_score's critic step. Per-iter:
        #       gt_chunk    = ride_latents_window at chunk's abs positions
        #       noisy_gt    = add_noise(gt_chunk, ε_aux, critic_timestep)
        #       causal_AR_GT = fake_score.alt(noisy_gt, no_grad)
        #       noisy_input  = add_noise(causal_AR_GT, ε, critic_timestep)
        #       clean_x      = causal_AR_GT  (un-noised same source)
        #       loss target  = true gt_chunk (NOT student chunk)
        #     Fake_score thus learns to denoise causal-AR-noised GT
        #     back to true GT. Falls back to ``self`` for iters whose
        #     chunk abs positions exceed the ride_latents_window range.
        self.critic_clean_x_source = str(
            getattr(args, "critic_clean_x_source", "self")
        ).lower().strip()
        if self.critic_clean_x_source not in ("self", "gt_causal_ar"):
            raise ValueError(
                "critic_clean_x_source must be 'self' or 'gt_causal_ar';"
                f" got {self.critic_clean_x_source!r}."
            )
        # ===== Aux-teacher clean_x source =====
        # ``gt`` (a.k.a. ``default``): clean_x_for_real = pure GT slice
        #     covering the chunk's clean-half window in ride coords.
        #     This is the canonical v14 contract — teacher denoises
        #     (noisy GT, ε, t) back to GT with a clean GT reference.
        #     The pre-v29 default was a GT-seed + student-snapshot
        #     mix, which leaked student-distribution structure into
        #     the teacher's score and was observed to break training.
        # ``causal_AR_GT``: clean_x_for_real = causal_AR_x0 (the alt-
        #     head's no_grad output on noised GT) or fn_causal_AR_GT
        #     (forward noiser's output on GT). Pairs the real_score's
        #     TF context with its noisy_input source (both derived from
        #     causal_AR_GT), so the LoRA learns "denoise causal-AR-noised
        #     input -> GT" with a consistent un-noised reference.
        #     Requires fake_alt_apply_active or the forward noiser
        #     to be active (else falls back to the pure-GT default).
        _aux_clean_x_raw = str(
            getattr(args, "aux_real_clean_x_source", "gt")
        ).lower().strip()
        # ``default`` is accepted as a legacy alias for ``gt`` so old
        # configs/sbatches keep working; semantics are identical.
        if _aux_clean_x_raw == "default":
            _aux_clean_x_raw = "gt"
        self.aux_real_clean_x_source = _aux_clean_x_raw
        if self.aux_real_clean_x_source not in ("gt", "causal_ar_gt"):
            raise ValueError(
                "aux_real_clean_x_source must be 'gt' (a.k.a. "
                "'default') or 'causal_AR_GT'; got "
                f"{self.aux_real_clean_x_source!r}."
            )
        # Decouple ``noisy_input`` from any CARN / alt-head shaping
        # applied to ``noise_base``. When True, the aux teacher's
        # ``noisy_input = scheduler.add_noise(<base>, ε, t)`` is built
        # from the **raw** ``gt_target`` instead of the (potentially
        # CARN-noised / alt-head-shaped) ``noise_base``.
        # ``clean_x_for_real`` (the LoRA's TF context) keeps whatever
        # the ``aux_real_clean_x_source`` setting put there — so this
        # knob lets you have an asymmetric setup:
        #   * clean_x = CARN-noised GT (current causal_AR_GT regime)
        #   * noisy_input = raw GT + Gaussian noise
        # The LoRA then learns "given a clean reference that's CARN-
        # noised GT, denoise a Gaussian-noised version of the TRUE GT
        # back to the TRUE GT". Useful when you want to keep CARN
        # shaping the conditioning while training the score on
        # vanilla (Gaussian-noise-only) inputs.
        # Default False = current behaviour (noisy_input built from
        # noise_base, which CARN / alt_head may have replaced).
        self.aux_noisy_from_raw_gt = bool(
            getattr(args, "aux_noisy_from_raw_gt", False)
        )
        # Upper bound for the per-chunk random renoise applied to
        # the pure-GT clean_x_aux slice. Each chunk of ``npb`` frames
        # is renoised at a SINGLE timestep sampled uniformly from
        # ``[0, clean_x_gt_noise_t]`` via ``scheduler.add_noise``,
        # then handed to the teacher. Without this the teacher's
        # task collapses to identity (crisp GT clean_x + noisy GT
        # noisy_x → trivially recover GT from the clean side).
        # Default upper bound 60 (roughly 3/10 of the denoising
        # ladder; the lowest non-zero rung sits at ~178). Set 0 to
        # disable entirely (teacher sees crisp GT — risks identity-
        # copy collapse).
        self.clean_x_gt_noise_t = int(
            getattr(args, "clean_x_gt_noise_t", 60)
        )
        if self.clean_x_gt_noise_t < 0 or self.clean_x_gt_noise_t >= 1000:
            raise ValueError(
                "clean_x_gt_noise_t must be in [0, 999]; got "
                f"{self.clean_x_gt_noise_t}"
            )
        # ===== v29 DMD lookback =====
        # When > 0, gradients from chunk_k's DMD loss flow back through
        # chunk_(k-1)'s random-exit forward via the overlap region. This
        # is truncated BPTT for the AR rollout: chunk N's quality
        # depends on chunk N-1's KV context, so DMD on chunk N should
        # also assign credit to chunk N-1's params (via student weights
        # shared across both forwards).
        #
        # Implementation: in generate_next_chunk, the previous_chunk
        # stash partially detaches — the overlap region (early frames)
        # is detached to break the cascade past 1 step back, while the
        # last new_frames region stays un-detached so the next iter's
        # cat-into-overlap carries chunk_k's exit-rung graph. With
        # generator_loss.backward(retain_graph=True) already in place,
        # the previous iter's graph survives long enough for the next
        # iter's backward to walk through it.
        #
        # Memory cost: ~+1 chunk of activations held at any time. With
        # activation checkpointing already on the exit rung, this is
        # ~5-7 GB per rank at our settings. Compute cost: ~+30-50% per
        # gen backward (re-walks chunk_(k-1)'s exit-rung graph each iter).
        #
        # 0 = lookback OFF (current/default behavior). 1 = 1-step
        # lookback. Higher values not yet supported (would require
        # multi-chunk retain_graph chaining + memory-bound).
        self.dmd_lookback_chunks = int(
            getattr(args, "dmd_lookback_chunks", 0)
        )
        if self.dmd_lookback_chunks < 0 or self.dmd_lookback_chunks > 1:
            raise ValueError(
                "dmd_lookback_chunks must be 0 or 1 (multi-step "
                "lookback not yet supported); got "
                f"{self.dmd_lookback_chunks}."
            )
        # ===== v27B forward noiser =====
        # Dedicated small ConvNet trained as a 1-step CARN forward-
        # noiser (rollout-1 chunk → rollout-2 chunk = +1 CARN step).
        # At apply time it's iteratively applied to GT chunks (chunkwise,
        # different CARN levels per chunk position) to synthesize a
        # CARN-shaped GT video that the online real_score's aux-teacher
        # pass uses as its noisy_input source. Replaces the v21 alt-head
        # application path (causal_AR_dir_rms=0 pathology).
        # 0 = off (default). Requires fake_alt_target_mode=rollout2_student
        # (or any mode where rollout2 chunks are available in the stash).
        self.forward_noiser_enabled = bool(
            getattr(args, "forward_noiser_enabled", False)
        )
        # Validate ``aux_real_clean_x_source=causal_ar_gt`` has a
        # source available. The override only fires inside the
        # ``_strategy_can_run`` block of
        # ``_compute_aux_teacher_loss_streaming`` (= when the forward
        # noiser actually applied a CARN-GT) or inside the analogous
        # ``fake_alt_apply_active`` block. With both off, clean_x
        # silently falls back to the gaussian-renoised pure GT (the
        # "gt" path) — a footgun for the user who thinks they're
        # getting CARN-GT clean_x. Raise here so misconfig fails fast.
        if self.aux_real_clean_x_source == "causal_ar_gt":
            if not (
                self.forward_noiser_enabled or self.fake_alt_head_enabled
            ):
                raise ValueError(
                    "aux_real_clean_x_source='causal_ar_gt' requires "
                    "either ``forward_noiser_enabled=True`` or "
                    "``fake_alt_head_enabled=True`` (to produce the "
                    "causal_AR_GT signal). With both off, clean_x "
                    "would silently fall back to pure-GT — this used "
                    "to be a silent footgun. Either enable one of "
                    "those producers or set ``aux_real_clean_x_source"
                    "='gt'`` explicitly."
                )
        self.forward_noiser_hidden_dim = int(
            getattr(args, "forward_noiser_hidden_dim", 512)
        )
        self.forward_noiser_num_blocks = int(
            getattr(args, "forward_noiser_num_blocks", 4)
        )
        self.forward_noiser_max_carn_step = int(
            getattr(args, "forward_noiser_max_carn_step", 16)
        )
        # Loss weight for the forward noiser's training MSE (predicted
        # rollout2 chunk vs actual rollout2 chunk at +1 CARN level).
        self.forward_noiser_loss_weight = float(
            getattr(args, "forward_noiser_loss_weight", 1.0)
        )
        # When True, the aux-teacher pass consumes the forward noiser's
        # iteratively-applied output as its causal_AR_GT source,
        # overriding the v21 alt-head application path.
        self.forward_noiser_apply_in_aux = bool(
            getattr(args, "forward_noiser_apply_in_aux", True)
        )
        # v27E: temporal low-freq REPLACEMENT on the FN-noised GT.
        # After applying the FN we compute the noise FN added
        #     residual = fn_output - gt_target
        # then low-pass that residual along the temporal axis with a
        # length-W boxcar, and REPLACE the FN output with
        #     boosted = gt_target + alpha * lowpass(residual)
        # i.e. GT with only the smoothed/low-frequency component of the
        # FN-added noise. window=0 disables (no-op). Default window=3
        # nulls f=1/3, 2/3 (one full period of the per-chunk
        # high-frequency element).
        self.forward_noiser_lowfreq_smooth_window = int(
            getattr(args, "forward_noiser_lowfreq_smooth_window", 0)
        )
        self.forward_noiser_lowfreq_alpha = float(
            getattr(args, "forward_noiser_lowfreq_alpha", 1.0)
        )
        if self.forward_noiser_lowfreq_smooth_window < 0:
            raise ValueError(
                "forward_noiser_lowfreq_smooth_window must be >= 0; got "
                f"{self.forward_noiser_lowfreq_smooth_window}"
            )
        # Per-pass apply strategy for the iterative FN application.
        # Only the learned strategy is supported now — the blur_noise
        # and sum variants depended on a 2D spatial Gaussian blur op
        # that has been removed from the codebase (it was unsafe / a
        # source of bugs and is no longer applied anywhere).
        #   "learned" (default): FN module only.
        #   "off":               skip iterative application; aux teacher
        #                        sees clean GT (debug only).
        self.forward_noiser_apply_strategy = str(
            getattr(args, "forward_noiser_apply_strategy", "learned")
        ).lower()
        if self.forward_noiser_apply_strategy not in ("learned", "off"):
            raise ValueError(
                "forward_noiser_apply_strategy must be 'learned' or "
                f"'off'; got {self.forward_noiser_apply_strategy!r} "
                "(legacy 'blur_noise' and 'sum' strategies removed)."
            )
        # ``aux_carn_level_mode``: how the per-chunk CARN level is assigned
        # in ``_apply_forward_noiser_to_gt`` (the aux-teacher GT corruption).
        #   "position" (default): level = max(0, chunk_abs_idx -
        #       (num_seed - 1)) — drift-faithful to the student rollout's
        #       actual CARN at each absolute ride position. The first
        #       ``num_seed`` chunks (the clean seed region) get level 0,
        #       then it ramps 1, 2, ...
        #   "local_index": level = the chunk's index WITHIN the aux window
        #       (chunk 0 -> 0 CARN, chunk 1 -> 1, chunk 2 -> 2, ...). Treats
        #       only the window's first chunk as the clean anchor, so the
        #       teacher sees the full 0..(n_chunks-1) drift ladder every
        #       aux step regardless of where the window sits in the ride.
        self.aux_carn_level_mode = str(
            getattr(args, "aux_carn_level_mode", "position")
        ).lower()
        if self.aux_carn_level_mode not in ("position", "local_index"):
            raise ValueError(
                "aux_carn_level_mode must be 'position' or 'local_index'; "
                f"got {self.aux_carn_level_mode!r}."
            )
        # ``carn_recurse``: how the per-chunk CARN level is REALISED.
        #   True (default): the FN is a +1 stepper, applied RECURSIVELY k
        #       times to reach level k (the input is fed back each call).
        #       Trained rollout1->rollout2 (a single +1 drift).
        #   False: the FN is a CUMULATIVE jumper conditioned on the target
        #       level. To reach level k we call it ONCE with carn_step=k on
        #       the CLEAN input (no recursion). This requires the FN to be
        #       trained clean(GT)->student-drifted-at-level-k with
        #       carn_step=k (handled in _train_forward_noiser_tf). The SAME
        #       choice drives the GAN-former CARN so the model stays
        #       consistent. carn_recurse=False implies the step IS used (the
        #       forward_noiser_step_unconditioned flag is ignored here).
        self.carn_recurse = bool(getattr(args, "carn_recurse", True))
        # ``ladd_gt_transition_carn_former``: in the transition GAN's GT
        # pairs (former_chunk, latter_chunk), push the FORMER through the
        # (trained) forward noiser so the disc sees REAL = "CARN-degraded ->
        # clean" (a self-correcting transition). The student, fooling the
        # disc, learns to clean up its own causal-AR drift each step. Does
        # NOT touch the real teacher (unlike the aux-pass CARN path) —
        # shapes the GAN target distribution instead. ``carn_steps`` = how
        # many FN steps to apply to the former (more = more degraded).
        self.ladd_gt_transition_carn_former = bool(
            getattr(args, "ladd_gt_transition_carn_former", False)
        )
        self.ladd_gt_transition_carn_steps = int(
            getattr(args, "ladd_gt_transition_carn_steps", 1)
        )
        # FIX A (gen-side former detach): the gt_transition real pair is
        # [CARN(GT_former) -> clean GT_latter]; the disc rewards a CARN-
        # degraded former, so the gen-side GAN loss (which flows to BOTH
        # student former and latter) pulls a GOOD/clean student former
        # back UP toward the fixed CARN level. When True, the gen-side
        # fake pair DETACHES the former so the GAN gradient cleans only
        # the LATTER (the transition target) and never degrades the
        # former. The disc D-update still sees both members (texture
        # learning + transition intact). Default off (= e8 behaviour).
        self.ladd_gt_transition_gen_detach_former = bool(
            getattr(args, "ladd_gt_transition_gen_detach_former", False)
        )
        # FIX B (random real-former CARN level): instead of a FIXED
        # ``carn_steps`` on the GT former, draw the CARN level uniformly in
        # [0, ladd_gt_transition_carn_max_level] PER PAIR (0 = clean
        # former). The disc then sees real formers at every degradation
        # level (incl. clean), so it cannot pull the student toward any
        # single fixed CARN level — it must key on the transition (clean
        # latter | any former). Default off; max_level defaults to
        # carn_steps.
        self.ladd_gt_transition_carn_random_level = bool(
            getattr(args, "ladd_gt_transition_carn_random_level", False)
        )
        self.ladd_gt_transition_carn_max_level = int(
            getattr(
                args, "ladd_gt_transition_carn_max_level",
                self.ladd_gt_transition_carn_steps,
            )
        )
        # e11: apply CARN inside the MATCHED candidate pool (the path e8
        # actually uses). e8's _carn_former only touched real_chunks_det,
        # which is UNUSED when ladd_gt_transition_match=true -> CARN was
        # inert. With this flag, the disc's matched real formers ARE
        # CARN'd, and:
        #   * Req 1: the per-row CARN level is drawn uniformly in [1, cap],
        #     cap = (min drift of the fake pairs this real serves) - 1, so
        #     the real former is always noised LESS than the student former
        #     it's compared against (and 0/clean when cap<1). Non-trivial
        #     (can exceed 1), per-row.
        #   * Req 2: the D-update sees the POST-CARN former (teaches the
        #     transition); the gen-side sees the PRE-CARN (clean) former
        #     (so the student is pulled toward clean, never toward CARN).
        # Default off (e8/g unchanged).
        self.ladd_gt_transition_carn_match_pool = bool(
            getattr(args, "ladd_gt_transition_carn_match_pool", False)
        )
        # Fixed CARN level for the matched real formers (Req-1 drift-capped).
        # 0 (default) => random level in [1, cap]; >0 => exactly this many
        # CARN steps per eligible former (still capped below served-fake drift).
        self.ladd_gt_transition_carn_match_pool_level = int(
            getattr(args, "ladd_gt_transition_carn_match_pool_level", 0)
        )
        # ``forward_noiser_loss_mode``: how the FN is trained.
        #   "mse" (default, legacy): value-match FN(rollout1) -> rollout2
        #       per chunk (folded into the critic backward — couples to
        #       fake_score).
        #   "teacher_feat": DISTRIBUTION-match FN(rollout1) -> rollout2 in
        #       the FROZEN teacher's feature space (the LADD
        #       WanFeatureProjector backbone, NOT its trained heads), as a
        #       SEPARATE backward stepping only the FN optimizer (fully
        #       decoupled from fake_score). The teacher is a fixed,
        #       comprehensive critic, so the FN can't exploit an adversary's
        #       blind spots. Computed trainer-side (where the projector +
        #       FN optimizer live). See trainer ``_train_forward_noiser_tf``.
        self.forward_noiser_loss_mode = str(
            getattr(args, "forward_noiser_loss_mode", "mse")
        ).lower()
        if self.forward_noiser_loss_mode not in ("mse", "teacher_feat"):
            raise ValueError(
                "forward_noiser_loss_mode must be 'mse' or 'teacher_feat'; "
                f"got {self.forward_noiser_loss_mode!r}."
            )
        # ``forward_noiser_reverse`` (h1): flip the FN training direction.
        #   False (default): FN(rollout1) -> rollout2  (learn to ADD a drift
        #                    step = a learned CARN).
        #   True:            FN(rollout2) -> rollout1  (learn to REMOVE a
        #                    drift step = a learned de-CARN denoiser).
        # Only wired for the teacher_feat path (_train_forward_noiser_tf).
        self.forward_noiser_reverse = bool(
            getattr(args, "forward_noiser_reverse", False)
        )
        # ``forward_noiser_apply_gt_both`` (h1): apply the FN ONCE to BOTH
        # GT latents (former + latter) of each matched real transition pair
        # (texture-only, moment-preserving). With forward_noiser_reverse it
        # de-CARNs the disc's real target, suppressing the rollout's
        # explode/harmonise modes from the real data distribution.
        self.forward_noiser_apply_gt_both = bool(
            getattr(args, "forward_noiser_apply_gt_both", False)
        )
        # ``forward_noiser_apply_gt_level`` (h1): the carn_step passed when
        # applying a STEP-CONDITIONED reverse FN to the GT pair. The de-CARN
        # FN learned (drifted @ level L -> clean GT) conditioned on L; level 1
        # removes one drift-level's worth of modes. Ignored when the FN is
        # step-unconditioned (then carn_step=0 is used, matching training).
        self.forward_noiser_apply_gt_level = int(
            getattr(args, "forward_noiser_apply_gt_level", 1)
        )
        # ``forward_noiser_apply_gt_former`` (h3): apply the FN ONCE to the
        # FORMER GT latent only of each matched real transition pair —
        # NAIVE variant: same CARN'd former on BOTH the D-update and the
        # gen-side real views (no drift cap, no gen-side clean former).
        # With a FORWARD-trained FN this makes the real anchor a
        # [1-step-drifted GT former -> clean GT latter] self-correcting
        # transition. Mutually exclusive with apply_gt_both (former wins
        # if both set... they should not both be set).
        self.forward_noiser_apply_gt_former = bool(
            getattr(args, "forward_noiser_apply_gt_former", False)
        )
        # ``forward_noiser_apply_gt_level_max`` (h4): when > apply_gt_level
        # (and the FN is step-conditioned), the application level is drawn
        # PER ROW uniformly in [apply_gt_level, level_max] instead of the
        # fixed level — a stronger, varied de-CARN/CARN. 0 (default) =
        # fixed level.
        self.forward_noiser_apply_gt_level_max = int(
            getattr(args, "forward_noiser_apply_gt_level_max", 0)
        )
        # ``forward_noiser_apply_gt_drift_cap`` (h5): per-row level drawn
        # uniformly in [1, cap] where cap = (min drift step of the student
        # formers this real row is served against) - 1 — i.e. the real
        # former is CARN'd strictly LESS than the student former it is
        # discriminated against (e11's Req-1 cap, here in the naive
        # step-conditioned application). Rows whose served fakes are at
        # drift <= 1 keep a CLEAN former (level 0 = no FN). Takes
        # precedence over apply_gt_level / level_max. Step-conditioned
        # FN only.
        self.forward_noiser_apply_gt_drift_cap = bool(
            getattr(args, "forward_noiser_apply_gt_drift_cap", False)
        )
        # ``fn_frontier_pairs`` (phase-2 rolling): at every ride RESET,
        # generate a fresh (rollout1', rollout2') chunk pair AT THE RIDE'S
        # FRONTIER — two short no_grad rollouts seeded from the student's
        # own newest chunks (6 vs 5 seeds -> +1 AR step of drift at the
        # same world position, mirroring the setup-time r1/r2 geometry) —
        # and train the FN on it. Keeps the FN looking at data from the
        # depths the generator actually reaches as rolling advances.
        self.fn_frontier_pairs = bool(
            getattr(args, "fn_frontier_pairs", False)
        )
        # ``forward_noiser_step_unconditioned``: when True, the FN never
        # sees the CARN level — every call passes carn_step=0, so it learns
        # a single generic "+1 shift" transform (input chunk -> output
        # chunk) regardless of how drifted the input is. Used by the
        # teacher_feat regime (and the carn_former GAN application). The
        # FiLM conditioning still exists but receives a constant, so it
        # collapses to a fixed modulation.
        self.forward_noiser_step_unconditioned = bool(
            getattr(args, "forward_noiser_step_unconditioned", False)
        )
        # Timestep at which the frozen teacher extracts features for the
        # teacher_feat distribution match (both FN(rollout1) and rollout2
        # are noised to this t before the projector forward). 0 = clean.
        self.forward_noiser_feat_t = int(
            getattr(args, "forward_noiser_feat_t", 0)
        )
        # Sliced-Wasserstein knobs for the teacher_feat FN loss.
        #   sw_n_proj: number of random 1D projections per feature tap
        #     (more = lower-variance Wasserstein estimate, linear cost).
        #   sw_max_tokens: cap on the per-tap sample count (random subsample
        #     per side) to bound sort cost/memory; 0 = use all tokens.
        self.forward_noiser_sw_n_proj = int(
            getattr(args, "forward_noiser_sw_n_proj", 64)
        )
        self.forward_noiser_sw_max_tokens = int(
            getattr(args, "forward_noiser_sw_max_tokens", 4096)
        )
        self.forward_noiser = None
        if self.forward_noiser_enabled:
            from model.forward_noiser import ForwardNoiser
            # Infer latent channels from the wrapped generator's
            # in_dim; default to 16 (Wan2.1 VAE).
            latent_ch = int(
                getattr(getattr(self.generator, "model", None), "in_dim", 16)
            )
            self.forward_noiser = ForwardNoiser(
                latent_channels=latent_ch,
                hidden_dim=self.forward_noiser_hidden_dim,
                num_blocks=self.forward_noiser_num_blocks,
                max_carn_step=self.forward_noiser_max_carn_step,
            )
            if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                logging.info(
                    "[ActionForcingDMD] ForwardNoiser ENABLED "
                    "(latent_ch=%d, hidden=%d, blocks=%d, params=%.2fM, "
                    "max_carn=%d, loss_w=%.2f, apply_in_aux=%s, "
                    "lowfreq_window=%d, lowfreq_alpha=%.2f, "
                    "strategy=%s).",
                    latent_ch,
                    self.forward_noiser_hidden_dim,
                    self.forward_noiser_num_blocks,
                    self.forward_noiser.num_params() / 1e6,
                    self.forward_noiser_max_carn_step,
                    self.forward_noiser_loss_weight,
                    str(self.forward_noiser_apply_in_aux),
                    self.forward_noiser_lowfreq_smooth_window,
                    self.forward_noiser_lowfreq_alpha,
                    self.forward_noiser_apply_strategy,
                )

        # Hard start-step gate for the aux teacher pass. Below this
        # step the aux pass does not fire at all (no real_score
        # forward, no LoRA gradient accumulated). Lets the student
        # produce reasonable outputs first so the LoRA isn't trained
        # on early-step student garbage. Independent of
        # ``real_teacher_warmup_steps`` (which is an LR-warmup ramp
        # only, applied AFTER this gate opens). Default 0 = no delay
        # (current behavior).
        self.aux_teacher_start_step = int(
            getattr(args, "aux_teacher_start_step", 0)
        )
        if self.aux_teacher_start_step < 0:
            raise ValueError(
                f"aux_teacher_start_step={self.aux_teacher_start_step} "
                f"must be >= 0"
            )

        # Linear warmup on top of the start-step gate: from
        # ``aux_teacher_start_step`` the effective aux loss weight
        # ramps from 0 → ``aux_teacher_loss_weight`` over
        # ``aux_teacher_loss_warmup_steps`` steps, then holds at
        # ``aux_teacher_loss_weight``. Independent of
        # ``real_teacher_warmup_steps`` (LR warmup). Default 0 =
        # instantaneous full weight at start_step (current behavior).
        self.aux_teacher_loss_warmup_steps = int(
            getattr(args, "aux_teacher_loss_warmup_steps", 0)
        )
        if self.aux_teacher_loss_warmup_steps < 0:
            raise ValueError(
                f"aux_teacher_loss_warmup_steps="
                f"{self.aux_teacher_loss_warmup_steps} must be >= 0"
            )
        # ``aux_teacher_separate_backward``: when True, the aux teacher is
        # NOT fused into the generator loss. Instead the trainer runs it as
        # a standalone forward+backward at END-OF-STEP (after the gen+critic
        # graphs are freed), so the 1.3B teacher's activation graph never
        # coexists with the GAN R1 double-backward. Same training signal
        # (fresh eps,t via run_extra_aux_pass), ~10-15 GB lower gen-step
        # peak. The real_teacher_optimizer step is unchanged (outer loop).
        self.aux_teacher_separate_backward = bool(
            getattr(args, "aux_teacher_separate_backward", False)
        )

        # GAN-disc-borrowed regularisers on the LoRA aux teacher's x0
        # estimate. The disc already encodes a high-quality "what does
        # real look like" judgement against the student; reusing it as
        # a second supervision channel for the LoRA gives a stronger
        # gradient than the flow-MSE alone, which struggles under
        # data-dependent CARN/forward-noiser noise. Two variants:
        #
        #   * ``aux_teacher_disc_adv_weight`` > 0: feed lora_x0 + gt
        #     through the same LADD disc the gen-side uses, take
        #     RpGAN gen-loss ``softplus(d_real - d_fake).mean()`` to
        #     push lora_x0 toward "real". Disc params detached — used
        #     as a critic, not trained on this signal.
        #
        #   * ``aux_teacher_disc_feat_weight`` > 0: intermediate-feature
        #     matching. Run the disc's projector on lora_x0 (grad)
        #     and gt (no_grad), compute per-block L2 of the captured
        #     teacher features. Smaller-risk LPIPS-style signal — no
        #     adversarial framing, no disc-collapse feedback concern.
        #
        # Both default to 0 (off). Independent — you can enable both.
        # Gated on top of the aux-teacher start gate by
        # ``aux_teacher_disc_warmup_steps`` so the disc has converged
        # before we trust its signal.
        self.aux_teacher_disc_adv_weight = float(
            getattr(args, "aux_teacher_disc_adv_weight", 0.0)
        )
        self.aux_teacher_disc_feat_weight = float(
            getattr(args, "aux_teacher_disc_feat_weight", 0.0)
        )
        # Default matches ``gan_critic_warmup_steps`` (~100) so the
        # disc has had enough D-updates to be a useful critic before
        # we read its score for the aux LoRA gradient. Set explicitly
        # higher when the gen-side adv path starts firing later.
        self.aux_teacher_disc_warmup_steps = int(
            getattr(args, "aux_teacher_disc_warmup_steps", 100)
        )

        # Flash DMD: when enabled, every block's rollout adds ONE extra
        # graph-on gen forward at ``flash_dmd_gan_t`` (default 60, raw
        # post-warp timestep) on top of the standard random-exit-rung
        # DMD pass. The extra forward's output feeds the GAN's adv
        # loss (and other gen-side aux losses); DMD scoring continues
        # to use the random-exit-rung output. Gradient through the
        # t=60 forward flows ONLY through that forward's gen weights —
        # the K/V it writes into the rolling cache is overwritten by
        # the per-block context-noise commit, preserving the paper's
        # cross-timestep decoupling. Default ON. ``flash_dmd_gan_t``
        # is the post-warp timestep (matches the convention of the
        # existing rungs 1000/625/312.5/178.57); ``60`` corresponds to
        # a near-clean image.
        self.flash_dmd_enabled = bool(
            getattr(args, "flash_dmd_enabled", True)
        )
        self.flash_dmd_gan_t = int(
            getattr(args, "flash_dmd_gan_t", 60)
        )

        # ===== LADD adjacent-chunk discriminator (v28) =====
        # Teacher-feature discriminator that taps WAN's intermediate
        # transformer blocks. Adversarial pairs are adjacent CHUNKS
        # within a single causal rollout
        # (chunk_i = real-anchor, chunk_{i+1} = fake / push-target).
        # See ``model/ladd_disc.py`` for the architecture; the loss
        # math lives in the trainer's
        # ``_compute_ladd_losses_distilled``. Activated by setting
        # ``gan_backbone=ladd_teacher_feat`` in config; otherwise the
        # legacy 3D-conv path runs unchanged.
        self.ladd_feature_blocks = list(
            getattr(args, "ladd_feature_blocks", []) or []
        )
        self.ladd_proj_dim = int(getattr(args, "ladd_proj_dim", 256))
        self.ladd_disc_head_kernel = int(
            getattr(args, "ladd_disc_head_kernel", 9)
        )
        self.ladd_use_csm = bool(getattr(args, "ladd_use_csm", True))
        self.ladd_use_lateral_proj = bool(
            getattr(args, "ladd_use_lateral_proj", False)
        )
        self.ladd_use_prompt_cond = bool(
            getattr(args, "ladd_use_prompt_cond", True)
        )
        self.ladd_cmap_dim = int(getattr(args, "ladd_cmap_dim", 64))
        self.ladd_disc_loss_weight = float(
            getattr(args, "ladd_disc_loss_weight", 1.0)
        )
        self.ladd_r1_gamma = float(getattr(args, "ladd_r1_gamma", 1.0))
        # Lazy R1: compute R1 every N disc updates instead of every
        # update. Default 1 (every step, original behaviour). Mainly
        # useful when ``ladd_r1_mode="autograd"`` (the expensive path).
        self.ladd_r1_every_n_steps = int(
            getattr(args, "ladd_r1_every_n_steps", 1)
        )
        # R1 estimator mode:
        #   "fd"       — finite-difference stochastic estimator
        #                (Causal-Forcing pattern, references at
        #                ``Causal-Forcing/model/gan.py:258-271`` and
        #                ``RollingForcing/model/gan.py``). One extra
        #                disc forward at ``x + sigma * eps``; no
        #                second-order autograd graph. ~half the memory
        #                of the autograd path.
        #   "autograd" — exact second-order ``torch.autograd.grad(...,
        #                create_graph=True)`` (legacy). Holds two
        #                graphs simultaneously and defeats the disc's
        #                gradient checkpointing — the dominant memory
        #                cost of the disc step.
        # Default "fd" (was "autograd" before this knob existed).
        self.ladd_r1_mode = str(
            getattr(args, "ladd_r1_mode", "fd")
        ).lower()
        if self.ladd_r1_mode not in ("fd", "autograd"):
            raise ValueError(
                "ladd_r1_mode must be 'fd' or 'autograd'; got "
                f"{self.ladd_r1_mode!r}."
            )
        # Sigma for finite-difference R1 perturbation. Same default as
        # Causal-Forcing's r1_sigma (0.01). Small enough that
        # ``D(x + sigma*eps) - D(x)`` is well-approximated by a
        # first-order Taylor expansion.
        self.ladd_r1_sigma = float(
            getattr(args, "ladd_r1_sigma", 0.01)
        )
        # --- R2 (zero-centered gradient penalty on the FAKE side) ---
        # The R3GAN ("GAN is dead, long live the GAN") recipe pairs R1
        # (penalty on real) with R2 (penalty on fake) — both are needed
        # for the local-convergence guarantee. The LADD path only had
        # R1; these knobs add the symmetric R2. Computed in the SAME
        # finite-difference style as R1 (one extra disc forward on
        # ``fake + sigma*eps``) on the DETACHED fake, so only D is
        # regularized (no gradient flows to the generator).
        #
        # Default ``ladd_r2_gamma=0.0`` -> OFF, so existing runs are
        # bit-identical until R2 is explicitly enabled.
        self.ladd_r2_gamma = float(getattr(args, "ladd_r2_gamma", 0.0))
        # Lazy R2 cadence (every N disc updates). Falls back to the R1
        # cadence when unset.
        self.ladd_r2_every_n_steps = int(
            getattr(args, "ladd_r2_every_n_steps",
                    self.ladd_r1_every_n_steps)
        )
        # Phase offset (in steps) between R1 and R2 firing. With equal
        # cadences (e.g. both every 2), an offset of 1 makes R1 fire on
        # even steps and R2 on odd steps, so the two perturbed-input
        # forwards NEVER stack in the same D-update — avoids the OOM of
        # carrying both extra segments at once. R1 fires on
        # ``step % r1_every == 0``; R2 fires on
        # ``(step - r2_offset) % r2_every == 0``. Default 1.
        self.ladd_r2_phase_offset = int(
            getattr(args, "ladd_r2_phase_offset", 1)
        )
        # Sigma for the finite-difference R2 perturbation. Falls back to
        # the R1 sigma when unset.
        self.ladd_r2_sigma = float(
            getattr(args, "ladd_r2_sigma", self.ladd_r1_sigma)
        )
        _diff_aug_raw = str(
            getattr(args, "ladd_diff_aug_policy", "flip,cutout,translation")
        )
        # Accept "none" / "off" / "" / "false" as the explicit-disabled
        # sentinel so command-line overrides (which can't easily pass an
        # empty string through OmegaConf dotlist syntax) have a clean
        # opt-out.
        if _diff_aug_raw.lower() in ("none", "off", "false", ""):
            _diff_aug_raw = ""
        self.ladd_diff_aug_policy = _diff_aug_raw
        self.ladd_pair_start_seed_boundary = bool(
            getattr(args, "ladd_pair_start_seed_boundary", True)
        )
        self.ladd_pairs_per_step = int(
            getattr(args, "ladd_pairs_per_step", 0)
        )  # 0 = use all pairs
        # ``ladd_pair_selection``: when ladd_pairs_per_step caps the pairs,
        # how to pick them. "random" (default) = random subset seeded by
        # the step (varies each step, same on all ranks). "first" = the
        # FIRST N pairs = the first N student-rolled chunks (lowest drift,
        # deterministic) — so pairs=2 supervises the first 2 rolled chunks,
        # pairs=3 the first 3, etc.
        self.ladd_pair_selection = str(
            getattr(args, "ladd_pair_selection", "random")
        ).lower()
        if self.ladd_pair_selection not in ("random", "first"):
            raise ValueError(
                f"ladd_pair_selection must be 'random' or 'first', got "
                f"{self.ladd_pair_selection!r}."
            )
        # LADD pair-mode toggles. v28A wants only gt_vs_fake; v28B
        # turns on adjacent_chunks as an additional adversarial term.
        # Defaults preserve v28 behaviour (adjacent on, gt_vs_fake off).
        self.ladd_gt_vs_fake_enabled = bool(
            getattr(args, "ladd_gt_vs_fake_enabled", False)
        )
        # All-pairs (style, not position) gt_vs_fake: compare EVERY GT
        # chunk to EVERY generated chunk in the disc loss (B_real x B_fake
        # relativistic terms) instead of position-matched (GT_i vs gen_i).
        # The disc forward is unchanged (still B unique reals + B unique
        # fakes); only the RpGAN reduction becomes a 7x7-style outer
        # product, so the extra signal is ~free in memory. Trains a
        # position-invariant "GT-style vs generated-style" discriminator.
        self.ladd_gt_vs_fake_all_pairs = bool(
            getattr(args, "ladd_gt_vs_fake_all_pairs", False)
        )
        # All-pairs for the gt_transition mode: compare EVERY GT
        # transition-pair (chunk i->i+1) to EVERY student transition-pair
        # (chunk j->j+1) in the disc loss (N_trans x N_trans relativistic
        # terms) instead of position-matched. Decorrelates transition
        # STYLE from position. Same outer-product loss as the gt_vs_fake
        # variant; the forward is unchanged (each transition already
        # carries its own position's action tokens, so every logit stays
        # correctly self-conditioned).
        self.ladd_gt_transition_all_pairs = bool(
            getattr(args, "ladd_gt_transition_all_pairs", False)
        )
        # Mismatched all-pairs: feed N_real GT chunks (sampled fresh from
        # the full ride window) vs the N_fake student chunks, giving an
        # N_real x N_fake all-pairs comparison (more decorrelated GT). The
        # GT side is RESAMPLED every D-update (hardcoded), so updates x
        # N_real distinct GT chunks per step. >0 enables it (and requires
        # the matching all_pairs knob). gt_vs_fake uses single chunks;
        # gt_transition uses transition-pairs. Texture-only disc, so the
        # co-located real actions are sufficient. Separate real/fake/
        # perturbed disc forwards make R1 exact for unequal counts.
        self.ladd_gt_vs_fake_n_real = int(
            getattr(args, "ladd_gt_vs_fake_n_real", 0)
        )
        self.ladd_gt_transition_n_real = int(
            getattr(args, "ladd_gt_transition_n_real", 0)
        )
        # Content-MATCHED wide GT (gt_transition). For each fake transition,
        # pick the K GT transitions from the loaded ride that are CLOSEST by
        # MAE — measured on the MEAN-EQUALIZED representation, so the match
        # is on texture/structure, NOT brightness (and so a brightness-
        # drifted fake can't cherry-pick a drifted GT). Each fake is then
        # scored ONLY against its own K matched GT (block-diagonal RpGAN,
        # NOT all_pairs), giving multiple RELEVANT real views per fake
        # without the content-confounded noise of all_pairs. Position is
        # never used (the student rollout drifts in time, so position-
        # matching is unreliable — content-matching is the point). Mutually
        # exclusive with all_pairs / n_real. ``match_k`` = GT per fake.
        # The candidate pool is the full loaded ride (stashed on
        # streaming_state as ``gt_match_latents`` / ``gt_match_actions``),
        # i.e. wider than ride_latents_window.
        self.ladd_gt_transition_match = bool(
            getattr(args, "ladd_gt_transition_match", False)
        )
        self.ladd_gt_transition_match_k = int(
            getattr(args, "ladd_gt_transition_match_k", 4)
        )
        # Relevant pool the per-update sampler draws the K matched GT from
        # (preserves 5b's resample-fresh-each-D-update GT decorrelation
        # without widening to the irrelevant whole-ride). 0 = auto (2*K);
        # ==K disables resampling (deterministic top-K).
        self.ladd_gt_transition_match_pool = int(
            getattr(args, "ladd_gt_transition_match_pool", 0)
        )
        # Hard cap on the number of DISTINCT GT transitions forwarded through
        # the disc per D-update (the combined real+fake+R1-perturbed forward
        # is 2*n_uniq + n_fake rows). Decouples the wide match pool from
        # disc-forward memory; over the cap, a fake's extra pick is remapped
        # to one of its own already-included nearer GT. 0 = uncapped.
        self.ladd_gt_transition_match_max_real = int(
            getattr(args, "ladd_gt_transition_match_max_real", 12)
        )
        # Magnitude-equalize the two members (former / latter chunk) of
        # every gt_transition pair — for BOTH real and fake — to their
        # common average MEAN MAGNITUDE (mean of |x|, the brightness/energy
        # proxy; NOT the signed mean, which can cancel to ~0). Done by
        # SCALING each member (lower magnitude up, higher down), preserving
        # the pair's overall magnitude. Removes the inter-chunk magnitude
        # (brightness) DRIFT across the transition from the disc's view, so
        # the transition GAN can no longer reward ever-increasing
        # brightness (the failure the gt_vs_fake disc doesn't have).
        # Differentiable on the grad-on fake side, so it also kills the
        # gen-side gradient that would push the inter-member magnitude.
        self.ladd_gt_transition_mean_equalize = bool(
            getattr(args, "ladd_gt_transition_mean_equalize", False)
        )
        # Cross-equalization variant for mean_equalize on gt_transition
        # pairs. Both variants equalize the OVERALL level across the real
        # and fake pools so the disc gets NO absolute-brightness cue (kills
        # the white-collapse feedback). They differ on the WITHIN-pair
        # former->latter brightness delta:
        #   False (default, "flatten"): scale each of the 4 members
        #     independently to the common level a* -> the within-pair delta
        #     is removed too; the disc is FULLY brightness-blind.
        #   True ("preserve_delta"): scale each PAIR by ONE shared factor to
        #     a* -> real/fake absolute level is equalized but the within-
        #     pair brightness TRANSITION ratio is preserved, so the disc can
        #     still see (and push the student to match GT's) transition
        #     brightness behaviour. No collapse because the absolute level
        #     is still pinned. Intended to become the hardcoded default.
        self.ladd_gt_transition_xeq_preserve_delta = bool(
            getattr(args, "ladd_gt_transition_xeq_preserve_delta", False)
        )
        # Cross-equalization granularity. The magnitude(s) used to equalize
        # are computed either as ONE scalar over all channels, or PER
        # CHANNEL.
        #   False (default): reduce over [F, C, H, W] -> one scalar per
        #     member; only the aggregate magnitude is equalized, so per-
        #     channel brightness/colour differences survive as a disc cue.
        #   True ("per_channel"): reduce over [F, H, W] keeping C -> a
        #     per-channel magnitude; each channel is equalized
        #     independently, so per-channel brightness/colour is fully
        #     non-discriminative (composes with preserve_delta: the shared
        #     factor becomes per-channel, still preserving the within-pair
        #     transition per channel). Intended to become the default.
        self.ladd_gt_transition_xeq_per_channel = bool(
            getattr(args, "ladd_gt_transition_xeq_per_channel", False)
        )
        # STD equalization — same cross-eq mechanism as the mean(-magnitude)
        # equalization above, but on the STD (spread/contrast) instead of
        # the magnitude. Applied AFTER the mean-eq, centered (scale the
        # deviations-from-mean to a common std, re-add the mean) so it sets
        # the 2nd moment WITHOUT disturbing the level. Honours the same
        # xeq_per_channel and xeq_preserve_delta options. Together with the
        # mean-eq this makes the disc input first+second-moment-free (per
        # channel) = a per-channel standardization -> the GAN keys on pure
        # texture, never brightness OR contrast. Requires mean_equalize on.
        self.ladd_gt_transition_std_equalize = bool(
            getattr(args, "ladd_gt_transition_std_equalize", False)
        )
        # Force the LADD disc to operate on CLEAN (t=0) latents. Normally the
        # disc timestep follows the wavelet stage (clean) or, with wavelet
        # off, the flash-DMD t (=flash_dmd_gan_t, ~60, NOISED). When the disc
        # is meant to learn brightness/contrast (wavelet off), noised input
        # swamps that signal — set this True to keep the disc on clean
        # latents without touching flash_dmd_gan_t (which the main DMD path
        # shares). Default False = legacy behaviour.
        self.ladd_disc_force_clean = bool(
            getattr(args, "ladd_disc_force_clean", False)
        )
        # Per-pair, per-channel magnitude normalization of gt_transition
        # pairs (both GT and student) — each pair self-normalizes using
        # ITS OWN per-channel stats (over F,H,W; C kept), so different
        # videos' different distributions are each handled correctly (no
        # cross-pair / cross-video reference). Makes per-channel magnitude
        # non-discriminative so the disc can't set brightness; DMD /
        # stat-anchor (M1) own brightness instead. Modes:
        #   ""     : off (default).
        #   "m1"   : divide each channel by its RMS -> unit per-channel
        #            ENERGY (M1). Removes per-channel brightness/magnitude;
        #            keeps per-channel contrast (std) for the disc.
        #   "m1m2" : per-channel standardize (subtract mean, divide std) ->
        #            unit per-channel energy AND contrast (M1 + M2). Removes
        #            brightness AND per-channel contrast.
        # Differentiable + scale(/shift)-invariant per channel, so the
        # gen-side gradient on the normalized stat is zeroed. Supersedes
        # mean_equalize when set.
        self.ladd_gt_transition_mag_norm = str(
            getattr(args, "ladd_gt_transition_mag_norm", "")
        ).lower()
        # Wide-real: draw the REAL (GT) chunks for the all-pairs gt_vs_fake
        # disc from the FULL loaded ride window (~25 chunks: seed + rollout
        # + post-window) instead of only the 21-frame scored slice — more
        # varied/decorrelated GT in the style comparison. Counts stay equal
        # to the fake side (so no forward/R1 change); only WHICH GT chunks
        # are sliced changes. Requires all_pairs (position no longer
        # matters). Real latent + its action are sliced from the same
        # absolute position in the same-coordinate ride windows, so they
        # stay co-located. Off by default.
        self.ladd_gt_vs_fake_wide_real = bool(
            getattr(args, "ladd_gt_vs_fake_wide_real", False)
        )
        self.ladd_adjacent_chunks_enabled = bool(
            getattr(args, "ladd_adjacent_chunks_enabled", True)
        )
        # gt_transition mode enable flag — was MISSING from the model
        # (the trainer's _compute_ladd_losses reads
        # getattr(self.model, "ladd_gt_transition_enabled", False), so
        # without this it ALWAYS resolved False and the gt_transition GAN
        # mode never ran). Wire it through here like the other modes.
        self.ladd_gt_transition_enabled = bool(
            getattr(args, "ladd_gt_transition_enabled", False)
        )
        # Per-mode gen-side weights (multiplied on top of the standard
        # gan_loss_weight ramp).
        self.ladd_gt_vs_fake_weight = float(
            getattr(args, "ladd_gt_vs_fake_weight", 1.0)
        )
        self.ladd_adjacent_chunks_weight = float(
            getattr(args, "ladd_adjacent_chunks_weight", 1.0)
        )
        self.ladd_gt_transition_weight = float(
            getattr(args, "ladd_gt_transition_weight", 1.0)
        )
        # Wavelet-HF pre-stage (v28B, WGSR-style frequency-band
        # restriction). When True, the disc's input latent is passed
        # through a single-level Haar SWT, LL is dropped, and the
        # remaining HF sub-bands are channel-adapted back to 16 channels
        # before the projector forward. This restricts the GAN gradient
        # to only push HF detail; content anchoring stays DMD's job.
        # 0 = off (default). See model/wavelet_hf.py.
        self.ladd_wavelet_hf_enabled = bool(
            getattr(args, "ladd_wavelet_hf_enabled", False)
        )
        self.ladd_wavelet_hf_drop_ll = bool(
            getattr(args, "ladd_wavelet_hf_drop_ll", True)
        )
        self.ladd_wavelet_hf_adapter_init_gain = float(
            getattr(args, "ladd_wavelet_hf_adapter_init_gain", 0.1)
        )

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
        if self.dmd_asymmetric_scoring_enabled:
            # Asymmetric DMD scores the FIRST rolled chunk against the
            # ``num_training_frames`` GT frames immediately preceding it
            # (ride positions [chunk_lo - N, chunk_lo) where the first
            # chunk sits at chunk_lo = cf + npb). For that GT history to
            # exist without underflowing the ride window, the seed must
            # satisfy ``cf >= num_training_frames - npb``. The legacy
            # ``cf < num_training_frames`` upper bound does NOT apply
            # here (the symmetric clean_x_self assembly that bound
            # protected is unused on the asymmetric path).
            min_cf = self.num_training_frames - self.num_frame_per_block
            if self.dmd_context_clean_frames < min_cf:
                raise ValueError(
                    f"dmd_asymmetric_scoring_enabled requires "
                    f"dmd_context_clean_frames ({self.dmd_context_clean_frames}) "
                    f">= num_training_frames - num_frame_per_block ({min_cf}) "
                    f"so the first rolled chunk has {self.num_training_frames} "
                    f"GT frames of clean context before it."
                )
        elif self.dmd_context_clean_frames >= self.num_training_frames:
            raise ValueError(
                f"dmd_context_clean_frames "
                f"({self.dmd_context_clean_frames}) must be < "
                f"num_training_frames ({self.num_training_frames})."
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

        if getattr(args, "gradient_checkpointing", False):
            try:
                self.generator.enable_gradient_checkpointing()
                self.fake_score.enable_gradient_checkpointing()
            except Exception as e:
                if _is_main():
                    logging.warning("gradient_checkpointing enable failed: %s", e)

        # ===== v21: enable alt head on fake_score =====
        # Build a parallel projection head on fake_score's WAN backbone
        # so a single forward can produce both pred_image-style x0 (main
        # head, trained as usual) and ema_real_x0-style x0 (alt head,
        # trained to predict the EMA-swapped real_score's output).
        # Idempotent; warm-inits the alt head from the main head's
        # weights so it starts equivalent and diverges with training.
        if self.fake_alt_head_enabled:
            try:
                self.fake_score.enable_alt_head()
                if _is_main():
                    n_alt = sum(
                        p.numel()
                        for p in self.fake_score.model.head_alt.parameters()
                    )
                    logging.info(
                        "[ActionForcingDMD] fake_score alt head ENABLED "
                        "(%d params, warm-init from main head). "
                        "train_start_step=%d apply_start_step=%d.",
                        n_alt,
                        self.fake_alt_head_start_step,
                        self.fake_alt_apply_start_step,
                    )
            except Exception as e:
                raise RuntimeError(
                    "fake_alt_head_enabled=True but enable_alt_head() "
                    f"failed: {e!r}. The underlying fake_score model "
                    "must support an alt head."
                )

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
        # Upper bound on the DMD timestep sampler. Default = full schedule
        # (num_train_timestep). Cap to <num_train_timestep to suppress the
        # very-high-noise regime where x0 estimates from both scorers are
        # noisy + biased toward the data mean (gray collapse risk).
        # ``ts_schedule_max`` still takes precedence (pipeline-driven cap
        # via ``denoised_timestep_from``).
        self.max_score_timestep = int(
            getattr(args, "max_score_timestep", self.num_train_timestep)
        )
        if not (
            self.min_score_timestep < self.max_score_timestep
            <= self.num_train_timestep
        ):
            raise ValueError(
                f"max_score_timestep ({self.max_score_timestep}) must be "
                f"in ({self.min_score_timestep}, {self.num_train_timestep}]."
            )
        # ``dmd_normalization_enabled``: when True (default), the DMD
        # gradient is divided per-sample by ``|x0 - pred_real|.mean()``
        # to give CausVid-style scale invariance. Pathology: as the
        # student approaches the teacher (pred_real -> x0), the
        # denominator shrinks and the gradient is artificially amplified
        # at the cusp of convergence. Setting this False uses the raw
        # ``pred_fake - pred_real`` gradient, so the DMD signal decays
        # naturally as the teacher agrees with the student — preferred
        # in v11+ where we want no power when teacher matches student.
        self.dmd_normalization_enabled = bool(
            getattr(args, "dmd_normalization_enabled", True)
        )
        # Denominator FLOOR for the CausVid normalizer. The DMD grad is
        # divided by ``max(|x0 - pred_real|.mean(), floor)``. The legacy
        # value 0.05 only guards the extreme cusp (caps amplification at
        # 20x) but still amplifies the gradient whenever the per-sample
        # normalizer < 1. Setting the floor to 1.0 makes the denominator
        # ``max(|f|, 1)``: for normalizers >= 1 it behaves like the
        # standard normalizer (down-scaling large grads), but once the
        # student converges and |f| drops below 1 it divides by 1 — i.e.
        # NO amplification near convergence, while still keeping the
        # scale-invariant down-weighting of over-large grads. Default
        # 0.05 preserves the prior behaviour exactly.
        self.dmd_normalization_denom_floor = float(
            getattr(args, "dmd_normalization_denom_floor", 0.05)
        )
        # --- MAE-based student-vs-teacher DMD downweighting ---
        # The DMD gradient pulls the student toward the frozen teacher
        # (real_score). That is only helpful while the teacher is the
        # better oracle. Once the student's per-chunk prediction is as
        # good as — or better than — the teacher's denoise on the SAME
        # scored chunk, the DMD pull is at best uninformative and at
        # worst actively drags the (better) student back toward the
        # (worse) teacher. This gate measures both errors against GT at
        # the scoring site and scales the DMD loss down as they converge:
        #   m_real = |pred_real    - gt|   (teacher denoise error)
        #   m_fake = |pred_student - gt|   (student pred_x0 error)
        #   r      = m_fake / m_real       (EMA-smoothed)
        #   w      = clamp((r - 1) / (r_full - 1), min_weight, 1)
        # so w=1 when the student is >= r_full x worse than the teacher,
        # ramping to ``min_weight`` (default 0) as the student reaches
        # teacher parity (r->1) or beats it (r<1). Default OFF.
        self.dmd_mae_gate_enabled = bool(
            getattr(args, "dmd_mae_gate_enabled", False)
        )
        # Ratio at/above which the gate passes the full DMD signal.
        # Must be > 1. At r = r_full -> w = 1; at r = 1 (parity) -> w =
        # min_weight.
        self.dmd_mae_gate_r_full = float(
            getattr(args, "dmd_mae_gate_r_full", 2.0)
        )
        if self.dmd_mae_gate_enabled and self.dmd_mae_gate_r_full <= 1.0:
            raise ValueError(
                "dmd_mae_gate_r_full must be > 1.0 (it is the fake/real "
                f"MAE ratio at which DMD reaches full weight); got "
                f"{self.dmd_mae_gate_r_full}."
            )
        # EMA on the ratio so a single noisy step (the per-step MAE is
        # jumpy) can't slam the gate shut. 0 = no smoothing (use the raw
        # per-step ratio); closer to 1 = heavier smoothing.
        self.dmd_mae_gate_ema = float(
            getattr(args, "dmd_mae_gate_ema", 0.9)
        )
        # Floor on the gate weight. Default 0.0 (DMD can be fully gated
        # off once the student matches/beats the teacher). Set > 0 to
        # always retain a residual DMD pull.
        self.dmd_mae_gate_min_weight = float(
            getattr(args, "dmd_mae_gate_min_weight", 0.0)
        )
        # Shape of the gate ramp. 1.0 = linear (w = (r-1)/(r_full-1));
        # <1 = CONCAVE — DMD stays near full strength across most of
        # (1, r_full) and only drops sharply toward parity, raising the
        # effective DMD strength under the gate (compensates the gate's
        # attenuation). >1 = convex (gentler, weaker). Default 1.0.
        self.dmd_mae_gate_exponent = float(
            getattr(args, "dmd_mae_gate_exponent", 1.0)
        )
        if self.dmd_mae_gate_exponent <= 0.0:
            raise ValueError(
                "dmd_mae_gate_exponent must be > 0; got "
                f"{self.dmd_mae_gate_exponent}."
            )
        # Per-rank EMA state for the gate ratio (None until first update).
        self._dmd_mae_gate_ratio_ema: Optional[float] = None
        # Anti-collapse variance floor. Penalizes the student's
        # per-frame latent std falling below the GT frame's std
        # (one-sided ReLU gap). Counters the loss-shape pull toward
        # the teacher's gray-biased prior; only active when
        # ``gt_target`` is available at the DMD call site (it is for
        # every gen-step loss path that supplies gt_latents). 0.0 = off.
        self.anti_collapse_loss_weight = float(
            getattr(args, "anti_collapse_loss_weight", 0.0)
        )
        # ``anti_collapse_std_floor_ratio``: scales the GT std before
        # the floor compare. The penalty becomes
        # ``ReLU(s_gt * ratio - s_pred)^2`` so it only fires when the
        # student's per-frame std drops below ``ratio * s_gt``, not
        # all the way up to ``s_gt``. Default 0.5 → student is free
        # to settle anywhere in ``[0.5 * s_gt, +inf]`` without the
        # floor pulling on it. Catches dramatic gray-collapse without
        # constantly pushing variance upward (which combined with the
        # one-sided floor was driving exponential std growth).
        self.anti_collapse_std_floor_ratio = float(
            getattr(args, "anti_collapse_std_floor_ratio", 0.5)
        )
        if self.anti_collapse_std_floor_ratio < 0.0:
            raise ValueError(
                "anti_collapse_std_floor_ratio must be >= 0.0; got "
                f"{self.anti_collapse_std_floor_ratio}"
            )
        # Anti-collapse mean anchor. Companion to the std floor above:
        # the std-only term lets the optimizer satisfy the variance
        # floor by inflating latent magnitude, which the VAE decodes
        # as saturated/white pixels (white-collapse — the overshoot of
        # gray-collapse). Penalizes ``pred_mean^2`` per-(batch, frame)
        # channel-pooled mean, pulling the latent offset toward 0 (the
        # VAE's zero-centered prior) so std cannot be satisfied by
        # translation. 0.0 = off (default).
        self.anti_collapse_mean_weight = float(
            getattr(args, "anti_collapse_mean_weight", 0.0)
        )
        # ``anti_collapse_mean_target``: selects what the mean anchor
        # pulls pred_x0's per-(batch, frame) channel-pooled mean
        # toward. "zero" (default, legacy) penalises ``pred_mean^2`` —
        # assumes the VAE prior is zero-centered. "gt" penalises
        # ``(pred_mean - gt_mean)^2`` — pulls toward the GT frame's
        # actual per-(batch, frame) mean, which is the more natural
        # anchor when GT is not zero-centered in latent space. Only
        # consulted when ``anti_collapse_mean_weight > 0``.
        self.anti_collapse_mean_target = str(
            getattr(args, "anti_collapse_mean_target", "zero")
        ).strip().lower()
        if self.anti_collapse_mean_target not in ("zero", "gt"):
            raise ValueError(
                f"anti_collapse_mean_target must be 'zero' or 'gt'; "
                f"got {self.anti_collapse_mean_target!r}."
            )
        # ``anti_collapse_type``: select the anti-collapse formulation.
        #   "std_floor" (default): legacy one-sided
        #       ``ReLU(floor_ratio * s_gt - s_pred)^2`` — penalises
        #       only shrinkage, never overshoot. Anchored at a fraction
        #       of GT std.
        #   "std_corridor": bidirectional log-ratio bands on std + RMS
        #       + mean, plus chunk-to-chunk drift penalty. More
        #       structured than std_mse but more knobs to tune.
        #   "std_mse": simplest possible — symmetric
        #       ``(s_pred - s_gt)^2``, per-frame averaged. Pulls the
        #       student std to *match* GT (both up and down), no
        #       corridor, no log space. Weight via
        #       ``anti_collapse_loss_weight``.
        # See ``model/anti_collapse.py`` for all three formulations.
        self.anti_collapse_type = str(
            getattr(args, "anti_collapse_type", "std_floor")
        ).strip().lower()
        if self.anti_collapse_type not in (
            "std_floor", "std_corridor", "std_mse",
            "std_graded_mse_constant",
        ):
            raise ValueError(
                "anti_collapse_type must be 'std_floor', 'std_corridor', "
                "'std_mse', or 'std_graded_mse_constant'; got "
                f"{self.anti_collapse_type!r}."
            )
        # std_graded_mse_constant knobs — fixed scalar target + MSE
        # floor that zeroes the gradient once close enough.
        self.anti_collapse_std_target = float(
            getattr(args, "anti_collapse_std_target", 0.875)
        )
        self.anti_collapse_std_mse_floor = float(
            getattr(args, "anti_collapse_std_mse_floor", 0.015)
        )
        # Per-rung gating for the anti-collapse term. Two independent
        # call sites exist:
        #   * DMD random-exit rung (``_compute_kl_grad``) — the rung the
        #     pipeline sampled this step. Always graph-attached to gen.
        #   * Flash-DMD t=gan_t rung (``info["flash_dmd_gan_chunk"]``) —
        #     the near-clean rung the GAN supervises. Only exists when
        #     ``flash_dmd_enabled=True``; without anti-collapse here the
        #     GAN-supervised rung is unconstrained and provides a
        #     degenerate-mode escape hatch (gen satisfies disc at gan_t
        #     while letting the random-exit rung drift toward gray).
        # Both default True to preserve historical behaviour. Set to
        # False to skip the corresponding rung's anti-collapse add.
        self.anti_collapse_apply_to_dmd_rung = bool(
            getattr(args, "anti_collapse_apply_to_dmd_rung", True)
        )
        self.anti_collapse_apply_to_flash_rung = bool(
            getattr(args, "anti_collapse_apply_to_flash_rung", True)
        )

        # v28E_12+: multi-horizon stat-anchor hinge loss config. Pins
        # M2 (Σ σ²) and TV (total variation) per-frame stats against
        # the seed window's stat values. Hinge bounds come from GT
        # rolling-band analysis (rolling_bands.json W=16 p95). Only
        # fires when the student's drift exceeds the GT p95 budget,
        # which DMD's per-pixel MSE cannot see (the per-pixel
        # ``fake - real`` cancels the marginal). See
        # ``model.anti_collapse.compute_stat_anchor_loss`` for the
        # implementation. Default OFF (weight=0) so existing v28E
        # configs are unaffected.
        self.stat_anchor_loss_weight = float(
            getattr(args, "stat_anchor_loss_weight", 0.0)
        )
        # --- Matched-GT stat anchor (stat_anchor_mode='target_matching') ---
        # In target_matching mode the stat anchor sources ALL its stats
        # (STD/M2/TV/SOS/M1) from the K closest GT chunks the GAN's matcher
        # would pick (same MAE-on-mean-equalized texture metric + the same
        # gt_match_latents pool), PER ROLLED CHUNK, instead of the positional
        # GT seed. See ``_matched_gt_stat_anchors``; the anchors feed the
        # normal ``compute_stat_anchor_loss`` (one code path, per-stat
        # weights as usual). ``stat_anchor_match_k``: how many closest GT
        # chunks to average (the "k" — like the GAN's match_k).
        self.stat_anchor_match_k = int(
            getattr(args, "stat_anchor_match_k", 3)
        )
        # NOTE: the matched anchor searches the WHOLE gt_match_latents pool
        # EXHAUSTIVELY (same scheme as the GAN matcher) — no subsampling, no
        # window. Memory is bounded by the pool size (= max_ride_frames) and
        # by computing stats for the K selected chunks only. (No knob.)
        # Per-stat per-horizon weights for the MSE-with-floor regulariser
        # against the seed anchor (STD, M2, TV) × (short = per-frame,
        # long = causal cumavg). See ``compute_stat_anchor_loss``.
        self.stat_anchor_STD_short_weight = float(
            getattr(args, "stat_anchor_STD_short_weight", 0.1)
        )
        self.stat_anchor_STD_long_weight = float(
            getattr(args, "stat_anchor_STD_long_weight", 0.1)
        )
        self.stat_anchor_M2_short_weight = float(
            getattr(args, "stat_anchor_M2_short_weight", 0.1)
        )
        self.stat_anchor_M2_long_weight = float(
            getattr(args, "stat_anchor_M2_long_weight", 0.1)
        )
        self.stat_anchor_TV_short_weight = float(
            getattr(args, "stat_anchor_TV_short_weight", 0.1)
        )
        self.stat_anchor_TV_long_weight = float(
            getattr(args, "stat_anchor_TV_long_weight", 0.1)
        )
        # SOS (raw Σ_{C,H,W} x²) and M1 (per-channel Σ_{H,W} x²) anchors.
        # Default OFF (0.0) — these are raw second moments with large
        # magnitudes, so set small weights relative to STD/TV.
        self.stat_anchor_SOS_short_weight = float(
            getattr(args, "stat_anchor_SOS_short_weight", 0.0)
        )
        self.stat_anchor_SOS_long_weight = float(
            getattr(args, "stat_anchor_SOS_long_weight", 0.0)
        )
        self.stat_anchor_M1_short_weight = float(
            getattr(args, "stat_anchor_M1_short_weight", 0.0)
        )
        self.stat_anchor_M1_long_weight = float(
            getattr(args, "stat_anchor_M1_long_weight", 0.0)
        )
        # Tolerance band as a fraction of the anchor. Deviations inside
        # the band produce zero gradient (floor = (rel_tol * anchor)^2).
        # Long horizon uses a tighter tolerance because cumulative
        # averaging smooths out per-frame noise.
        self.stat_anchor_rel_tol_short = float(
            getattr(args, "stat_anchor_rel_tol_short", 0.20)
        )
        self.stat_anchor_rel_tol_long = float(
            getattr(args, "stat_anchor_rel_tol_long", 0.10)
        )
        # Optional linear ramp-down on ``stat_anchor_loss_weight``.
        # When ``stat_anchor_rampdown_steps > 0``: starting at step
        # ``stat_anchor_rampdown_start_step``, the effective weight
        # ramps linearly from the static knob value down to 0 over
        # ``stat_anchor_rampdown_steps`` steps, then stays at 0.
        # Used to anchor early-training stats while letting the
        # student find its natural distribution later.
        # ``rampdown_steps == 0`` (default) disables the schedule.
        self.stat_anchor_rampdown_start_step = int(
            getattr(args, "stat_anchor_rampdown_start_step", 0)
        )
        self.stat_anchor_rampdown_steps = int(
            getattr(args, "stat_anchor_rampdown_steps", 0)
        )
        # Floor (as a FRACTION of the full weight) the rampdown decays TO,
        # instead of all the way to 0. Default 0.0 = ramp to zero (legacy).
        # Set e.g. 0.1 to keep a thin stat-anchor safety net (a tenth of
        # full) after the GAN takes over brightness/contrast.
        self.stat_anchor_rampdown_floor = float(
            getattr(args, "stat_anchor_rampdown_floor", 0.0)
        )
        # v28G_5+: optional EMA on the LONG-HORIZON stat_anchor.
        # Short-horizon (per-frame) anchor keeps the per-batch seed
        # value so rollouts stay pinned to the ride they started in.
        # Long-horizon anchor uses a running EMA of cross-rank-averaged
        # seed anchors, converging to GT-population stats over ~100
        # steps at ema_weight=0.99. Eliminates the "edge-seed lock-in"
        # failure where a single seed at the tail of the GT
        # distribution blocks long-horizon self-correction toward
        # the population mean (rel_tol_long forces the rollout to
        # match the seed indefinitely, even when the seed is an
        # edge sample).
        # Default OFF; flip on via sbatch.
        self.stat_anchor_long_ema_enabled = bool(
            getattr(args, "stat_anchor_long_ema_enabled", False)
        )
        self.stat_anchor_long_ema_weight = float(
            getattr(args, "stat_anchor_long_ema_weight", 0.99)
        )
        # Running-EMA state. ``None`` until the first stat_anchor
        # call populates them with the first step's cross-rank mean
        # (cold start). Identical across ranks by construction
        # (the helper all-reduces before the EMA update). Held as
        # plain Tensor attributes (not registered buffers) so
        # checkpoint resumes cold-start; that's acceptable given
        # the EMA converges in ~100 steps at w=0.99.
        self._stat_anchor_STD_long_ema: Optional[torch.Tensor] = None
        self._stat_anchor_M2_long_ema: Optional[torch.Tensor] = None
        self._stat_anchor_TV_long_ema: Optional[torch.Tensor] = None

        # v28E_17+: stat_anchor mode dispatch.
        # ``seed_anchor`` (default): the original MSE-with-floor
        #     comparison against per-batch seed anchors (optionally
        #     EMA-driven for the long horizon).
        # ``target_matching``: the anchor SOURCE for ALL stats becomes the
        #     per-rolled-chunk K-CLOSEST GT chunks (K=``stat_anchor_match_k``)
        #     selected from ``streaming_state['gt_match_latents']`` by MAE on
        #     the mean-equalized rep — the same selection the LADD GAN
        #     matcher uses. The resulting per-frame anchors feed the SAME
        #     ``compute_stat_anchor_loss`` (per-frame-anchor aware). Requires
        #     ``ladd_gt_transition_match=true`` (populates the pool); falls
        #     back to the seed anchor with a one-time warning otherwise.
        #     See ``_matched_gt_stat_anchors``.
        self.stat_anchor_mode = str(
            getattr(args, "stat_anchor_mode", "seed_anchor")
        ).lower().strip()
        if self.stat_anchor_mode not in (
            "seed_anchor", "target_matching", "gt_window",
        ):
            raise ValueError(
                "stat_anchor_mode must be 'seed_anchor', 'target_matching', "
                f"or 'gt_window'; got {self.stat_anchor_mode!r}."
            )
        # ``gt_window`` mode: the anchor SOURCE is the POSITIONALLY-ALIGNED
        # GT chunk (the actual GT the student should produce at this rolled
        # position) plus/minus ``stat_anchor_match_k`` neighbour chunks,
        # averaged -> a smoothed local-GT anchor. The SAME ``stat_anchor_
        # match_k`` knob serves both modes: in ``target_matching`` it is the
        # number of content-CLOSEST GT chunks; in ``gt_window`` it is the
        # +/-k window RADIUS of temporally-adjacent GT around the chunk's own
        # ride position. k=0 -> exactly the aligned GT chunk; k=2 -> mean over
        # a 5-chunk window centred on it.
        # Target-matching knobs (consulted only when mode ==
        # ``target_matching``). M2 target / band correspond to the
        # user's externally-named "STD" (Σ σ², typical Wan range 6-9).
        self.stat_anchor_target_M2 = float(
            getattr(args, "stat_anchor_target_M2", 9.0)
        )
        self.stat_anchor_target_M2_band_low = float(
            getattr(args, "stat_anchor_target_M2_band_low", 5.0)
        )
        self.stat_anchor_target_M2_band_high = float(
            getattr(args, "stat_anchor_target_M2_band_high", 13.0)
        )
        self.stat_anchor_target_M2_weight = float(
            getattr(args, "stat_anchor_target_M2_weight", 0.1)
        )
        self.stat_anchor_target_TV = float(
            getattr(args, "stat_anchor_target_TV", 7.8)
        )
        self.stat_anchor_target_TV_band_low = float(
            getattr(args, "stat_anchor_target_TV_band_low", 6.0)
        )
        self.stat_anchor_target_TV_band_high = float(
            getattr(args, "stat_anchor_target_TV_band_high", 9.0)
        )
        self.stat_anchor_target_TV_weight = float(
            getattr(args, "stat_anchor_target_TV_weight", 0.1)
        )
        self.stat_anchor_target_rank = float(
            getattr(args, "stat_anchor_target_rank", 1.5)
        )
        self.stat_anchor_target_rank_band_high = float(
            getattr(args, "stat_anchor_target_rank_band_high", 2.25)
        )
        # Optional lower bound for stable rank. -1 (or 0/negative) =
        # use the user-spec upper-only gate (default contract).
        # Set positive to enable a two-sided gate.
        _rank_low_raw = float(
            getattr(args, "stat_anchor_target_rank_band_low", -1.0)
        )
        self.stat_anchor_target_rank_band_low = (
            _rank_low_raw if _rank_low_raw > 0.0 else None
        )
        self.stat_anchor_target_rank_weight = float(
            getattr(args, "stat_anchor_target_rank_weight", 0.1)
        )

        # Corridor-mode weights (consulted only when
        # ``anti_collapse_type == "std_corridor"``). Defaults match the
        # discussion: moment=0.1 (per-frame std/RMS/mean corridor),
        # drift=0.03 (chunk-to-chunk std-ratio drift). The corridor
        # bounds are slightly wider than the original spec
        # (0.85/1.15 vs 0.90/1.10) to avoid spurious early-training
        # firing while still anchoring the steady-state distribution.
        self.latent_moment_corridor_loss_weight = float(
            getattr(args, "latent_moment_corridor_loss_weight", 0.1)
        )
        self.latent_contrast_drift_loss_weight = float(
            getattr(args, "latent_contrast_drift_loss_weight", 0.03)
        )
        self.latent_moment_corridor_std_low = float(
            getattr(args, "latent_moment_corridor_std_low", 0.85)
        )
        self.latent_moment_corridor_std_high = float(
            getattr(args, "latent_moment_corridor_std_high", 1.15)
        )
        self.latent_moment_corridor_rms_low = float(
            getattr(args, "latent_moment_corridor_rms_low", 0.85)
        )
        self.latent_moment_corridor_rms_high = float(
            getattr(args, "latent_moment_corridor_rms_high", 1.15)
        )
        self.latent_moment_corridor_mean_tol_ratio = float(
            getattr(args, "latent_moment_corridor_mean_tol_ratio", 0.05)
        )
        self.latent_contrast_drift_tol = float(
            getattr(args, "latent_contrast_drift_tol", 0.02)
        )
        # ``max_gradient_chunks``: cap on how many of the leading
        # chunks (each ``num_frame_per_block`` frames) carry gradient
        # in DMD-score and aux-teacher losses. 0 = no cap (CF default,
        # only the trailing last-chunk boundary is masked). N>0 means
        # only chunks [0, N) receive gradient — every chunk at index
        # N or later is force-masked. Use this to restrict learning
        # to the seed-adjacent, drift-free rolls and discard the
        # noise-dominated tail. Applies uniformly to all losses that
        # route through ``_dmd_score_grad_mask`` (gen DMD score, aux
        # teacher, critic update, streaming critic).
        self.max_gradient_chunks = int(
            getattr(args, "max_gradient_chunks", 0)
        )
        if self.max_gradient_chunks < 0:
            raise ValueError(
                f"max_gradient_chunks must be >= 0; got "
                f"{self.max_gradient_chunks}."
            )
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

        # DMD loss start-step + linear warmup gate. Below ``dmd_loss_
        # start_step`` the gen-side DMD loss is zero (no gradient to
        # the student from DMD); from ``dmd_loss_start_step`` to
        # ``dmd_loss_start_step + dmd_loss_warmup_steps`` the
        # effective weight ramps linearly from 0 to ``dmd_loss_weight``.
        # Past that, the full weight applies. Default
        # ``dmd_loss_start_step=0, dmd_loss_warmup_steps=0`` reproduces
        # the prior "DMD full from step 0" behavior bit-identically.
        # Use case: let the student rollout warm up on GAN
        # signal first (steps 0-49), then phase DMD in (50-100), then
        # full from step 100. Independent of GAN's own
        # ``gan_warmup_steps`` / ``gan_critic_warmup_steps`` ramps.
        self.dmd_loss_start_step = int(
            getattr(args, "dmd_loss_start_step", 0)
        )
        self.dmd_loss_warmup_steps = int(
            getattr(args, "dmd_loss_warmup_steps", 0)
        )
        if self.dmd_loss_start_step < 0:
            raise ValueError(
                f"dmd_loss_start_step={self.dmd_loss_start_step} "
                f"must be >= 0"
            )
        if self.dmd_loss_warmup_steps < 0:
            raise ValueError(
                f"dmd_loss_warmup_steps={self.dmd_loss_warmup_steps} "
                f"must be >= 0"
            )

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
        # v28G_6+: optional LoRA wrap on fake_score. Default OFF
        # (full-FT). MUST run AFTER the mirror (mirror loads
        # state_dict directly into the bare WAN; peft wrap on top
        # would intercept the load path). See ``_apply_fake_score_lora``
        # for the memory-savings rationale and head_alt handling.
        self._apply_fake_score_lora(device)
        # Phased DMD K-LoRA on the student. Default OFF. MUST run
        # AFTER the ODE checkpoint load AND AFTER the fake_score
        # mirror (mirror copies generator.model.state_dict(); a peft
        # wrap on top would rename keys and break that path).
        self._apply_student_phase_lora(device)

        # Attach the (now weight-loaded) state_probe to the real_score
        # wrapper too. Must run AFTER the v14 LoRA peft wrap so the
        # ``get_base_model`` walk finds the bare DiT, and AFTER the
        # state_probe weights have been loaded into ``self.state_probe``
        # by ``_load_generator_from_ode_checkpoint`` so the shared
        # module on real_score reads from the trained weights from
        # iter 1. No-op when state_probe is disabled.
        self._attach_state_probe_to_real_score(args)

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
        # v14 LoRA was trained with context_shift=1 chunk → tf_rope_offset
        # = num_frame_per_block. Non-negotiable: the LoRA's weights are
        # tuned for clean at RoPE [0, F) and noisy at RoPE [npb, npb+F),
        # giving a (F + npb)-frame total RoPE span where each chunk
        # index shares the same RoPE position across the clean and
        # noisy halves (in the overlap region). DO NOT change without
        # retraining the v14 LoRA.
        _tf_rope_off = int(self.num_frame_per_block)
        assert _tf_rope_off == 3, (
            f"v14 LoRA was trained with num_frame_per_block=3; got "
            f"{_tf_rope_off}. The teacher's joint-TF RoPE convention is "
            f"hardcoded to v14's training contract."
        )
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

        # Verify the value propagated all the way to each scorer's
        # base model at construction time (before the first forward).
        # The bidir patch only sets per-block ``tf_rope_offset`` on the
        # first patched forward; we read ``tf_rope_offset_frames`` at
        # the model level here to verify the trainer-side setter took.
        for wrapper in scorer_wrappers:
            m = (
                wrapper.get_base_model()
                if hasattr(wrapper, "get_base_model") else wrapper
            )
            actual = int(getattr(m, "tf_rope_offset_frames", 0))
            assert actual == _tf_rope_off, (
                f"scorer {type(m).__name__} has "
                f"tf_rope_offset_frames={actual}, expected "
                f"{_tf_rope_off}. v14 LoRA training contract violated."
            )

        if _is_main():
            logging.info(
                "[ActionForcingDMD] dmd_context=self (hardcoded): set "
                "context_shift=1 chunk (= tf_rope_offset_frames=%d) on "
                "real_score and fake_score. KV-cache seed prefill = "
                "%d frames.",
                self.num_frame_per_block,
                self.dmd_context_clean_frames,
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

        if _is_main():
            logging.info(
                "[ActionForcingDMD] LoRA aux-pass clean_x and noisy_input "
                "are both pure noised(GT). DMD scoring uses the self-view "
                "clean_x (hardcoded)."
            )

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

    def _attach_state_probe_to_real_score(self, args) -> None:
        """Mirror ``adding_state_probe_branch`` onto ``self.real_score``
        without instantiating a second StateProbeModule. The probe nn.
        Module stays SHARED with the generator (one set of weights, one
        optimizer). What the real_score wrapper needs to fire its
        forward-time probe readout:

          * ``self.real_score._state_probe`` — the shared module.
          * ``self.real_score._state_n_chunks``,
            ``self.real_score._state_z_out_dim`` — used by the wrapper
            to gate the probe forward (skips when frame count mismatches).
          * ``self.real_score.model._state_probe_tap_set`` /
            ``_state_probe_tap_indices`` — the inner DiT reads these
            to know which transformer-block depths to snapshot during
            its forward.

        The aux pass operates on a chunk_size = num_training_frames =
        21-frame window (= 7 chunks of 3), exactly the geometry the
        probe was init'd at. So ``state_preds`` fires on every aux
        forward with full graph-on coverage.

        No-op when state_probe is disabled or when the gen-side attach
        in ``_build_action_aux_heads_compat`` failed silently.
        """
        if self.state_probe is None or self.real_score is None:
            return
        gen_n = int(getattr(self.generator, "_state_n_chunks", 0))
        gen_z = int(getattr(self.generator, "_state_z_out_dim", 0))
        gen_set = getattr(self.generator.model, "_state_probe_tap_set", None)
        gen_idx = getattr(self.generator.model, "_state_probe_tap_indices", None)
        if not gen_n or not gen_z or gen_set is None or gen_idx is None:
            if _is_main():
                logging.warning(
                    "[ActionForcingDMD] _attach_state_probe_to_real_score: "
                    "generator-side state_probe attach incomplete; skipping "
                    "real_score attach."
                )
            return
        self.real_score._state_probe = self.state_probe
        self.real_score._state_n_chunks = gen_n
        self.real_score._state_z_out_dim = gen_z
        # Walk to the inner DiT (LoRA-wrapped → unwrap with get_base_model)
        # and stamp the tap-set on EVERY candidate hop so the actual forward
        # path (which may traverse peft.LoraModel → base_model → DiT
        # depending on the peft version's dispatch) reads it from
        # whichever ``self`` it sees during ``_forward_train``. Setting
        # on extra hops is harmless — only the DiT's forward consults
        # ``_state_probe_tap_set``; the LoraModel/PeftModel layers
        # don't read it at all.
        # Walk EVERY layer of wrapping the real_score's model can
        # acquire and stamp the tap_set on each hop. The actual layering
        # at runtime can be (worst case):
        #   wrapper.model = DDP(peft.PeftModel(LoraModel(DiT)))
        # so we need to unwrap DDP first, then peft, then LoraModel,
        # to hit the bare DiT that ``_forward_train`` reads from. We
        # also stamp the intermediate hops because some forward paths
        # propagate through them and a redundant set is harmless.
        try:
            from torch.nn.parallel import DistributedDataParallel as _DDP
        except Exception:
            _DDP = None
        candidates = []
        cur = self.real_score.model
        # Hop 0: outermost (may be DDP).
        candidates.append(("outer", cur))
        # Hop 1: unwrap DDP if present.
        if _DDP is not None and isinstance(cur, _DDP):
            cur = cur.module
            candidates.append(("ddp.module", cur))
        # Hop 2: peft.PeftModel.get_base_model() if present.
        if hasattr(cur, "get_base_model"):
            g = cur.get_base_model()
            if g is not cur:
                candidates.append(("get_base_model()", g))
                cur_after_peft = g
            else:
                cur_after_peft = cur
        else:
            cur_after_peft = cur
        # Hop 3: peft.LoraModel intermediate (peft_model.base_model
        # is the LoraModel; its .model is the bare DiT).
        outer_for_lora = candidates[1][1] if len(candidates) > 1 else cur
        if hasattr(outer_for_lora, "base_model"):
            bm = getattr(outer_for_lora, "base_model")
            candidates.append(("base_model", bm))
            if hasattr(bm, "model"):
                candidates.append(("base_model.model", bm.model))
        # Stamp all unique objects.
        seen = set()
        unique = []
        for label, obj in candidates:
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            unique.append((label, obj))
            obj._state_probe_tap_set = set(gen_set)
            obj._state_probe_tap_indices = list(gen_idx)
        if _is_main():
            logging.info(
                "[ActionForcingDMD] state_probe attached to real_score "
                "(shared module, taps at %s, stamped on %d unique hops: %s).",
                list(gen_idx), len(unique),
                [label for label, _ in unique],
            )

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
            # Online teacher: TWO-STAGE LoRA setup.
            #
            # Stage 1 — MERGE v14 into the base: fold v14's rank-256
            # LoRA weights into the WAN base permanently. v14's learned
            # behavior is now baked into the base parameters; the peft
            # wrapper is removed.
            _apply_v14_lora(self.real_score, merge=True)
            # Stage 2 — apply a FRESH rank-``aux_teacher_lora_rank``
            # LoRA on top of the v14-merged base. This new adapter
            # starts from zero output (random init), is what the
            # online aux teacher trains, and has 8x fewer params than
            # v14 at rank 32 (~1 GB savings on params + Adam state +
            # grads). Target modules are re-collected from the merged
            # base (peft strips the wrapper after merge_and_unload).
            target_modules = self._collect_target_modules(self.real_score.model)
            if not target_modules:
                target_modules = ["q", "k", "v", "o"]
            fresh_lora_config = LoraConfig(
                r=int(self.aux_teacher_lora_rank),
                lora_alpha=float(self.aux_teacher_lora_alpha),
                lora_dropout=float(self.aux_teacher_lora_dropout),
                target_modules=target_modules,
                bias="none",
            )
            self.real_score.model = peft.get_peft_model(
                self.real_score.model, fresh_lora_config,
            )
            # Mark fresh LoRA params trainable; base stays frozen.
            trainable_lora: List[nn.Parameter] = []
            for name, p in self.real_score.model.named_parameters():
                if "lora_" in name:
                    p.requires_grad_(True)
                    trainable_lora.append(p)
                else:
                    p.requires_grad_(False)
            self._real_teacher_trainable_params = trainable_lora
            if not trainable_lora:
                raise RuntimeError(
                    "[ActionForcingDMD] real_teacher_train_online=True but "
                    "no LoRA params were marked trainable after the fresh "
                    "rank-%d peft wrap — check peft naming convention or "
                    "LoRA config." % int(self.aux_teacher_lora_rank)
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
                    "v14 LoRA MERGED into base; fresh rank-%d LoRA "
                    "(alpha=%.1f, dropout=%.3f) applied on top, %d LoRA "
                    "params trainable, tf_use_causal_mask=%s, "
                    "gradient_checkpointing=%s.",
                    int(self.aux_teacher_lora_rank),
                    float(self.aux_teacher_lora_alpha),
                    float(self.aux_teacher_lora_dropout),
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

    def _apply_fake_score_lora(self, device) -> None:
        """Wrap ``fake_score.model`` with peft LoRA when
        ``fake_score_lora_enabled``. Marks LoRA params trainable,
        freezes the base WAN, and re-enables ``head_alt`` (v21 alt
        head, NOT an attention LoRA target) so the alt loss can
        still train it via the critic step.

        Memory effect: full-FT fake_score holds ~15 GB of optimizer
        state (params + grad + Adam m/v at bf16/fp32 mix on a 1.3B
        WAN); LoRA at rank 32 drops that to ~3 GB (frozen base +
        thin adapter). The DDP gradient buffer also shrinks
        proportionally. Frees ~10-12 GB per rank for other uses
        (larger ladd_pairs_per_step, max_gradient_chunks, etc.).

        Must run AFTER ``_mirror_generator_into_fake_score`` (the
        mirror loads state_dict directly into the bare WAN; peft
        wrap on top would break the direct attribute access) and
        AFTER ``enable_alt_head`` (so head_alt is present in the
        base WAN before peft wraps it).
        """
        if not self.fake_score_lora_enabled:
            return
        if not _HAS_PEFT:
            raise RuntimeError(
                "fake_score_lora_enabled=True but peft is not installed."
            )
        rank = int(self.fake_score_lora_rank)
        alpha = float(self.fake_score_lora_alpha)
        dropout = float(self.fake_score_lora_dropout)
        target_modules = self._collect_target_modules(self.fake_score.model)
        if not target_modules:
            target_modules = ["q", "k", "v", "o"]
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
        )
        peft_model = peft.get_peft_model(self.fake_score.model, lora_config)
        self.fake_score.model = peft_model.to(
            device=device, dtype=self.dtype,
        )
        trainable_lora = 0
        for name, p in self.fake_score.model.named_parameters():
            if "lora_" in name:
                p.requires_grad = True
                trainable_lora += 1
            else:
                p.requires_grad = False
        # Re-enable head_alt (v21 fake_alt) — NOT a LoRA target, but
        # must remain trainable so the alt loss in the critic step
        # has a gradient path. Use the wrapper's ``_unwrapped_model``
        # to walk past peft + any other wrapping to the bare WAN
        # that holds the actual head_alt submodule.
        alt_head_params = 0
        try:
            base = self.fake_score._unwrapped_model()
            head_alt = getattr(base, "head_alt", None)
            if head_alt is not None:
                for p in head_alt.parameters():
                    p.requires_grad = True
                    alt_head_params += 1
        except Exception as exc:
            if _is_main():
                logging.warning(
                    "[ActionForcingDMD] fake_score LoRA: failed to "
                    "re-enable head_alt grad (%s); alt loss may not "
                    "train.", exc,
                )
        if _is_main():
            logging.info(
                "[ActionForcingDMD] fake_score LoRA wrapped "
                "(rank=%d alpha=%s drop=%s). %d LoRA params "
                "trainable; %d head_alt params re-enabled.",
                rank, alpha, dropout,
                trainable_lora, alt_head_params,
            )

    def _apply_student_phase_lora(self, device) -> None:
        """Wrap ``generator.model`` with K named peft LoRA adapters
        ('rung_0'..'rung_{K-1}'). One frozen WAN base, K shared-base
        adapters. Active adapter is selected per generator forward by
        the pipeline via ``set_student_phase_rung``. Must run AFTER
        ``_load_generator_from_ode_checkpoint`` (loader writes to the
        bare WAN) and AFTER ``_mirror_generator_into_fake_score``
        (mirror reads ``generator.model.state_dict()`` and expects
        bare keys).
        """
        if not self.student_phase_lora_enabled:
            return
        if not _HAS_PEFT:
            raise RuntimeError(
                "student_phase_lora_enabled=True but peft is not installed."
            )
        K = int(self.student_phase_lora_K)
        rank = int(self.student_phase_lora_rank)
        alpha = float(self.student_phase_lora_alpha)
        dropout = float(self.student_phase_lora_dropout)
        target_modules = self._collect_target_modules(self.generator.model)
        if not target_modules:
            target_modules = ["q", "k", "v", "o"]
        names = [f"rung_{k}" for k in range(K)]

        def _mk_cfg() -> "LoraConfig":
            return LoraConfig(
                r=rank, lora_alpha=alpha, lora_dropout=dropout,
                target_modules=target_modules, bias="none",
            )

        peft_model = peft.get_peft_model(
            self.generator.model, _mk_cfg(), adapter_name=names[0],
        )
        for nm in names[1:]:
            peft_model.add_adapter(nm, _mk_cfg())
        # Optional dedicated flash adapter (K+1 design). When present,
        # the pipeline's _maybe_set_phase_lora_for_flash routes the
        # Flash-DMD t=60 forward HERE so the K ODE rungs stay clean.
        flash_added = False
        if self.student_phase_lora_flash_adapter_enabled:
            peft_model.add_adapter("rung_flash", _mk_cfg())
            flash_added = True
        self.generator.model = peft_model.to(device=device, dtype=self.dtype)

        # All K adapters trainable BEFORE the DDP wrap so the reducer
        # sees the union. set_adapter() at forward time only flips
        # which one contributes to grads on a given step; DDP marks
        # the inactive K-1 as unused (find_unused_parameters=True is
        # required and set by the trainer).
        trainable_lora = 0
        for n, p in self.generator.model.named_parameters():
            if "lora_" in n:
                p.requires_grad = True
                trainable_lora += 1
            elif self.student_phase_lora_freeze_base:
                p.requires_grad = False
            # else: leave base requires_grad as-is (full-FT + LoRA hybrid)

        self._student_phase_lora_names = names
        # NOTE: do NOT call set_adapter(names[0]) here — peft's
        # set_adapter forces requires_grad=False on the non-active
        # adapters, which would break the optimizer build that
        # follows (rung_k>0 would see 0 trainable params). The pipeline
        # dispatcher calls set_adapter per rung at forward time; we
        # re-enable all lora_ requires_grad AFTER each dispatch so
        # DDP's static graph stays consistent across iters.

        if _is_main():
            logging.info(
                "[ActionForcingDMD] student phase LoRA wrapped: K=%d "
                "(rank=%d alpha=%s drop=%s freeze_base=%s flash_adapter=%s). "
                "%d LoRA params trainable across %d adapters; "
                "target_modules=%d.",
                K, rank, alpha, dropout,
                self.student_phase_lora_freeze_base,
                flash_added,
                trainable_lora, K + (1 if flash_added else 0),
                len(target_modules),
            )

    def set_student_phase_rung(self, rung_idx: int) -> None:
        """Dispatcher: select which student LoRA contributes to the
        next generator forward. Safe under ``torch.no_grad()``.
        Unsafe under ``torch.compile`` (mutates active adapter Python
        attr; would force recompile). Call BEFORE each per-rung
        generator forward in the pipeline."""
        if not getattr(self, "student_phase_lora_enabled", False):
            return
        names = getattr(self, "_student_phase_lora_names", None)
        if not names:
            return
        idx = int(rung_idx) % len(names)
        self._active_phase_rung_idx = idx
        try:
            # set_adapter is on the PeftModel; if DDP-wrapped, peek
            # through .module.
            gen_model = self.generator.model
            if hasattr(gen_model, "module"):
                gen_model = gen_model.module
            gen_model.set_adapter(names[idx])
        except Exception as exc:
            if _is_main():
                logging.warning(
                    "[ActionForcingDMD] set_student_phase_rung(%d) failed: %s",
                    idx, exc,
                )

    def _phase_lora_idx_for_step(
        self, denoise_idx: int, num_denoising_steps: int,
    ) -> int:
        """Map a denoise rung index in ``[0, num_denoising_steps)`` to
        a phase-LoRA adapter index in ``[0, K)``. K=4 over 4 rungs is
        identity; K=2 over 4 rungs is {0,1}->0, {2,3}->1; K=1 always 0.
        No-op (returns 0) when phase LoRA is disabled.
        """
        if not getattr(self, "student_phase_lora_enabled", False):
            return 0
        K = max(1, int(self.student_phase_lora_K))
        bucket = max(1, int(num_denoising_steps) // K)
        return min(K - 1, int(denoise_idx) // bucket)

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

        # Flash-DMD: when ``self.flash_dmd_enabled`` is True, the
        # pipeline adds a per-block t=flash_dmd_gan_t forward. Originally
        # this was gated on ``requires_grad=True`` so the critic step
        # could skip the extra forward. v27I onward: flash_dmd is
        # COMPULSORY whenever the master switch is on — both rollout 1
        # (gen step, grad-on) AND rollout 2 (prebuild, no_grad) must see
        # the same 5-forward (4 rungs + t=gan_t) trajectory so the FN
        # training pairs are symmetric.
        flash_dmd_enabled = bool(self.flash_dmd_enabled)
        # Warm-start init applies to BOTH gen and critic rollouts so
        # the critic sees the same denoising trajectory shape as the
        # gen (otherwise fake_score would learn a different
        # distribution than the gen produces).
        pred_image_or_video, denoised_timestep_from, denoised_timestep_to = (
            self.inference_pipeline.inference_with_trajectory(
                noise=noise,
                clean_image_or_video=None,
                gt_latents=clean_latent,
                seed_latents=seed_latents,
                requires_grad=requires_grad,
                flash_dmd_enabled=flash_dmd_enabled,
                flash_dmd_gan_t=int(self.flash_dmd_gan_t),
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
        # ``max_gradient_chunks`` cap: limit gradient to the leading
        # N chunks of the scoring window. Frames at chunk index >= N
        # are force-masked. This is composed AFTER the trailing
        # last-chunk boundary mask (which is always applied for
        # v14-LoRA RoPE-boundary reasons), so the effective mask is
        # the INTERSECTION of "first N chunks" and "not last chunk".
        if self.max_gradient_chunks > 0:
            cap_frames = int(self.max_gradient_chunks) * block
            if cap_frames < shape[1]:
                mask[:, cap_frames:] = False
        return mask

    def _matched_gt_stat_anchors(self, pred_image: torch.Tensor):
        """Per-frame stat anchors from the per-chunk k-CLOSEST GT chunks.

        Used by ``stat_anchor_mode='target_matching'``: for each rolled
        chunk, find the K nearest GT chunks in the match pool
        (``streaming_state['gt_match_latents']``) by MAE on the mean-
        equalized (brightness-removed, texture) rep — the same selection
        the LADD matcher uses, K = ``stat_anchor_match_k`` — and set that
        chunk's anchor for EVERY stat (STD, M2, TV, SOS, M1) to the mean of
        those K matched GT chunks. Returns a dict of per-frame anchors
        ({STD,M2,TV,SOS: [B,F]; M1: [B,F,C]}, fully detached) to feed
        ``compute_stat_anchor_loss``'s precomputed-anchor path — so ALL
        stats flow through the SAME function, no separate loss term.
        Returns None when the pool is unavailable (e.g.
        ladd_gt_transition_match off) -> caller falls back to the seed.
        """
        s = getattr(self, "streaming_state", None)
        pool = s.get("gt_match_latents") if isinstance(s, dict) else None
        if pool is None or pred_image is None:
            if pool is None and not getattr(self, "_matched_gt_warned", False):
                self._matched_gt_warned = True
                if _is_main():
                    import logging as _logging
                    _logging.warning(
                        "[ActionForcingDMD] stat_anchor_mode='target_matching' "
                        "but gt_match_latents is missing -> falling back to the "
                        "seed anchor. Set ladd_gt_transition_match=true to "
                        "populate the match pool."
                    )
            return None
        from model.anti_collapse import (
            _per_frame_STD, _per_frame_M2, _per_frame_TV,
            _per_frame_SOS, _per_frame_M1,
        )
        npb = int(self.num_frame_per_block)
        if pred_image.dim() != 5 or int(pred_image.shape[1]) < npb:
            return None
        B, F_, C, H, W = pred_image.shape
        pool = pool.detach().to(device=pred_image.device, dtype=torch.float32)
        n_pool = int(pool.shape[1]) // npb
        if n_pool < 1:
            return None
        eps = 1e-6
        pred = pred_image.float().detach()  # anchors are TARGETS -> detached
        n_chunks = F_ // npb

        def _meq(x):  # abs-mean normalize per chunk over (npb,C,H,W)
            a = x.abs().mean(dim=(-4, -3, -2, -1), keepdim=True).clamp_min(eps)
            return x / a

        with torch.no_grad():
            pool_c = pool[:, : n_pool * npb].reshape(B, n_pool, npb, C, H, W)
            # SAME SCHEME AS THE GAN MATCHER: each rolled chunk searches the
            # WHOLE pool EXHAUSTIVELY (torch.cdist over all n_pool chunks,
            # p=1 on the mean-equalized rep) for its k-nearest GT — NO
            # subsampling, NO window. Memory stays light because we only
            # materialize (a) the mean-equalized search reps for the cdist and
            # (b) the raw latents of the K SELECTED chunks for stats (just
            # like the GAN only forwards its selected reals) — never the stats
            # of the whole pool. The pool size is bounded by max_ride_frames.
            pool_meq = _meq(pool_c).reshape(B, n_pool, -1)           # [B,n_pool,D]
            _D = float(pool_meq.shape[-1])
            K = max(1, min(int(getattr(self, "stat_anchor_match_k", 3)), n_pool))

            a_std = pred.new_zeros((B, F_))
            a_m2 = pred.new_zeros((B, F_))
            a_tv = pred.new_zeros((B, F_))
            a_sos = pred.new_zeros((B, F_))
            a_m1 = pred.new_zeros((B, F_, C))
            for c in range(n_chunks):
                fsl = slice(c * npb, (c + 1) * npb)
                pc_meq = _meq(pred[:, fsl]).reshape(B, 1, -1)
                # Exhaustive cdist over the WHOLE pool (no subsample/window).
                mae = torch.cdist(pc_meq, pool_meq, p=1).squeeze(1) / _D  # [B,n_pool]
                top = torch.topk(mae, K, dim=1, largest=False).indices    # [B,K]
                # Stats for the K SELECTED chunks only (gather raw latents).
                sel = pool_c.gather(
                    1, top.view(B, K, 1, 1, 1, 1).expand(B, K, npb, C, H, W),
                )                                                        # [B,K,npb,...]
                sf = sel.reshape(B * K, npb, C, H, W)
                a_std[:, fsl] = _per_frame_STD(sf).reshape(
                    B, K, npb).mean(-1).mean(1, keepdim=True)
                a_m2[:, fsl] = _per_frame_M2(sf).reshape(
                    B, K, npb).mean(-1).mean(1, keepdim=True)
                a_tv[:, fsl] = _per_frame_TV(sf).reshape(
                    B, K, npb).mean(-1).mean(1, keepdim=True)
                a_sos[:, fsl] = _per_frame_SOS(sf).reshape(
                    B, K, npb).mean(-1).mean(1, keepdim=True)
                a_m1[:, fsl] = _per_frame_M1(sf).reshape(
                    B, K, npb, C).mean(2).mean(1, keepdim=True)
            rem = F_ - n_chunks * npb
            if rem > 0:  # remainder frames (F not a multiple of npb): repeat last
                j = n_chunks * npb
                a_std[:, j:] = a_std[:, j - 1:j]
                a_m2[:, j:] = a_m2[:, j - 1:j]
                a_tv[:, j:] = a_tv[:, j - 1:j]
                a_sos[:, j:] = a_sos[:, j - 1:j]
                a_m1[:, j:] = a_m1[:, j - 1:j]
        return {"STD": a_std, "M2": a_m2, "TV": a_tv, "SOS": a_sos, "M1": a_m1}

    def _gt_window_stat_anchors(self, pred_image: torch.Tensor, chunk_lo: int):
        """Per-frame stat anchors from the POSITIONALLY-ALIGNED GT chunk
        +/- ``stat_anchor_match_k`` neighbour chunks (smoothed local GT).

        Used by ``stat_anchor_mode='gt_window'``. For each rolled chunk ``c``
        in ``pred_image`` (whose GT lives at abs ride frame
        ``chunk_lo + c*npb`` in ``streaming_state['ride_latents_window']``),
        the anchor for EVERY stat (STD, M2, TV, SOS, M1) is the mean over the
        GT frames in the window ``[p - k*npb, p + (k+1)*npb)`` clamped to the
        ride. This is the actual GT the student should match at that position
        — not a content search (``target_matching``) and not the seed prefix
        (``seed_anchor``) — smoothed over +/- k chunks to suppress per-chunk
        GT noise. Returns ``{STD,M2,TV,SOS:[B,F]; M1:[B,F,C]}`` detached, or
        ``None`` if the ride window is unavailable (caller falls back).
        """
        s = getattr(self, "streaming_state", None)
        ride = s.get("ride_latents_window") if isinstance(s, dict) else None
        if ride is None or pred_image is None:
            return None
        from model.anti_collapse import (
            _per_frame_STD, _per_frame_M2, _per_frame_TV,
            _per_frame_SOS, _per_frame_M1,
        )
        npb = int(self.num_frame_per_block)
        if pred_image.dim() != 5 or int(pred_image.shape[1]) < npb:
            return None
        B, F_, C, H, W = pred_image.shape
        ride = ride.detach().to(device=pred_image.device, dtype=torch.float32)
        T = int(ride.shape[1])
        if T < npb:
            return None
        k = int(getattr(self, "stat_anchor_match_k", 3))
        n_chunks = F_ // npb
        a_std = pred_image.new_zeros((B, F_))
        a_m2 = pred_image.new_zeros((B, F_))
        a_tv = pred_image.new_zeros((B, F_))
        a_sos = pred_image.new_zeros((B, F_))
        a_m1 = pred_image.new_zeros((B, F_, C))
        with torch.no_grad():
            for c in range(n_chunks):
                p = int(chunk_lo) + c * npb
                lo = p - k * npb
                hi = p + (k + 1) * npb
                # Clamp the +/-k window to the ride; if it falls entirely
                # outside, snap to the nearest in-bounds single chunk.
                lo = max(0, lo)
                hi = min(T, hi)
                if hi - lo < npb:
                    lo = max(0, min(p, T - npb))
                    hi = lo + npb
                win = ride[:, lo:hi]                 # [B, w, C, H, W]
                fsl = slice(c * npb, (c + 1) * npb)
                # Per-frame stats over the window, averaged over its frames;
                # [B,1] broadcasts across the chunk's npb frames ([B,1,C] for M1).
                a_std[:, fsl] = _per_frame_STD(win).mean(1, keepdim=True)
                a_m2[:, fsl] = _per_frame_M2(win).mean(1, keepdim=True)
                a_tv[:, fsl] = _per_frame_TV(win).mean(1, keepdim=True)
                a_sos[:, fsl] = _per_frame_SOS(win).mean(1, keepdim=True)
                a_m1[:, fsl] = _per_frame_M1(win).mean(1, keepdim=True)
            rem = F_ - n_chunks * npb
            if rem > 0:  # remainder frames: repeat the last chunk's anchor
                j = n_chunks * npb
                a_std[:, j:] = a_std[:, j - 1:j]
                a_m2[:, j:] = a_m2[:, j - 1:j]
                a_tv[:, j:] = a_tv[:, j - 1:j]
                a_sos[:, j:] = a_sos[:, j - 1:j]
                a_m1[:, j:] = a_m1[:, j - 1:j]
        return {"STD": a_std, "M2": a_m2, "TV": a_tv, "SOS": a_sos, "M1": a_m1}

    def _dmd_mae_gate_weight(
        self,
        pred_real: torch.Tensor,
        pred_student: torch.Tensor,
        gt_target: Optional[torch.Tensor],
        gradient_mask: torch.Tensor,
        log_dict: Dict[str, Any],
    ) -> float:
        """Scalar in ``[min_weight, 1]`` that scales the DMD loss down as
        the student's per-chunk error approaches/beats the teacher's.

        Both errors are measured against GT on the SAME scored chunk and
        on the SAME supervised slots (``gradient_mask``) — the only
        apples-to-apples comparison of "is the teacher still the better
        oracle here". Returns 1.0 (no gating) when the gate is disabled,
        GT is unavailable, shapes mismatch, or the teacher error is
        degenerate.

        DDP note: the weight is per-rank (each rank gates on its OWN
        ride's student-vs-teacher ratio, with its own EMA state on
        ``self``). DDP then averages the scaled gradients, so the
        effective signal is ``mean_r(w_r · g_r)`` — each ride is
        downweighted by how good ITS student is, which is the intended
        behaviour (not a bug). The EMA momentum makes the per-rank
        ``w_r`` converge over time. No-op steps (early-returns below)
        deliberately do NOT touch the EMA, so an occasional empty-mask /
        missing-GT step leaves the smoothed ratio untouched rather than
        resetting it.
        """
        if not self.dmd_mae_gate_enabled:
            return 1.0
        if gt_target is None:
            log_dict["dmd_mae_gate_skipped"] = 1.0
            return 1.0
        gt = gt_target.to(dtype=pred_real.dtype, device=pred_real.device)
        if not (gt.shape == pred_real.shape == pred_student.shape):
            log_dict["dmd_mae_gate_skipped"] = 1.0
            return 1.0
        if not gradient_mask.any():
            log_dict["dmd_mae_gate_skipped"] = 1.0
            return 1.0
        with torch.no_grad():
            m = gradient_mask
            gtf = gt.float()
            m_real = (pred_real.float() - gtf)[m].abs().mean()
            m_fake = (pred_student.float() - gtf)[m].abs().mean()
            eps = 1e-8
            # Degenerate teacher error -> don't gate (avoid div blow-up).
            if float(m_real.item()) <= eps:
                log_dict["dmd_mae_gate_skipped"] = 1.0
                return 1.0
            r = float((m_fake / (m_real + eps)).item())
            # EMA-smooth the ratio (per-rank state).
            a = self.dmd_mae_gate_ema
            if self._dmd_mae_gate_ratio_ema is None or a <= 0.0:
                r_s = r
            else:
                r_s = a * self._dmd_mae_gate_ratio_ema + (1.0 - a) * r
            self._dmd_mae_gate_ratio_ema = r_s
            # w ramps 0->1 over r in [1, r_full]; clamped to [min_w, 1].
            # ``dmd_mae_gate_exponent`` shapes the ramp: exponent=1 is the
            # linear ramp; exponent<1 makes it CONCAVE — DMD stays near full
            # strength across most of (1, r_full) and only drops sharply
            # toward parity (r->1). Use exponent<1 to keep the DMD signal
            # strong despite the gate, with the loss "going away" only very
            # close to where the student matches the teacher.
            r_full = self.dmd_mae_gate_r_full
            base = (r_s - 1.0) / (r_full - 1.0)
            base = max(0.0, min(1.0, base))
            exponent = float(self.dmd_mae_gate_exponent)
            w = base ** exponent if exponent != 1.0 else base
            w = max(self.dmd_mae_gate_min_weight, min(1.0, w))
            log_dict["dmd_mae_gate_m_real"] = float(m_real.item())
            log_dict["dmd_mae_gate_m_fake"] = float(m_fake.item())
            log_dict["dmd_mae_gate_ratio"] = r
            log_dict["dmd_mae_gate_ratio_ema"] = r_s
            log_dict["dmd_mae_gate_weight"] = float(w)
            # Always-on companion to ``real_score_mae_vs_gt`` so the
            # student-vs-teacher crossover is finally visible on the
            # SAME (per-chunk, masked) basis.
            log_dict["student_mae_vs_gt"] = float(m_fake.item())
            return float(w)

    def _sigma_at_timestep(
        self, timestep: torch.Tensor, like: torch.Tensor
    ) -> torch.Tensor:
        """Look up the scheduler's σ_t for each timestep entry.

        ``timestep`` : long tensor, any shape (typically [B, F]).
        ``like``     : reference tensor providing target device/dtype
                       and the desired broadcast shape (e.g. chunk
                       [B, F, C, H, W]). The returned σ tensor is
                       broadcast-compatible with ``like`` — same B/F
                       and singleton trailing dims.

        Mirrors the lookup performed by ``add_noise`` and
        ``_convert_flow_pred_to_x0`` so the σ used for the AR-noise
        flow-space scaling is byte-identical to the σ used inside
        the noising operation.
        """
        sched_t = self.scheduler.timesteps.to(timestep.device)
        sched_s = self.scheduler.sigmas.to(
            device=like.device, dtype=like.dtype,
        )
        flat_t = timestep.reshape(-1)
        idx = torch.argmin(
            (sched_t.unsqueeze(0) - flat_t.unsqueeze(1).float()).abs(),
            dim=1,
        )
        sigma = sched_s[idx].reshape(timestep.shape)
        # Right-pad singleton dims to broadcast against ``like``.
        while sigma.dim() < like.dim():
            sigma = sigma.unsqueeze(-1)
        return sigma

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

        RoPE convention boundary
        ────────────────────────
        The DMD scorers (real_score = v14 LoRA, fake_score = student
        copy) use the v14 LoRA's training-time RoPE convention:

            Joint-TF input: [clean_half (F frames), noisy_half (F frames)]
            Clean half at RoPE [0, F)
            Noisy half at RoPE [npb, npb + F)
            Total RoPE span: F + npb = 24 frames = 8 chunks (v25)

        The student (Phase-3 always-roll causal cache) uses a different
        convention: window-relative RoPE where every newest chunk lands
        at ``[local_attn_size - npb, local_attn_size)``. The two
        conventions do NOT align — the teacher's bidir joint-TF needs
        unique RoPE positions per chunk to function, while the
        student's causal always-roll deliberately puts every newest
        chunk at the same RoPE position.

        Aligning them strictly would require either (a) retraining v14
        in the student's window-relative convention, or (b) running the
        teacher causally (one forward per noisy chunk) so each chunk's
        RoPE matches its student-generation-time position. Option (b)
        costs 7x teacher forwards and is currently out of scope.

        Until then, the teacher scores at v14's training-distribution
        RoPE (this function), and the student generates at its own
        RoPE. The DMD gradient direction is approximately right
        despite the positional offset between conventions; this is a
        known, bounded inconsistency.
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

        # Step 2: real score. CF parity used to ALWAYS run cond + uncond
        # forwards (no gate on real_guidance_scale) and rely on
        # ``scale=0`` collapsing the math to ``pred_real_image_cond``.
        # That cost ~7 GB of activation graph per gen step on this rig
        # (one full TF 42-frame DiT forward). When ``real_guidance_scale
        # == 0`` we now skip the uncond forward entirely — the math is
        # identical (``pred_real = cond + 0 * (cond - uncond) = cond``)
        # and the saved activation graph is the cheapest single memory
        # win available. CF parity preserved by-value; only the
        # forward count differs.
        _, pred_real_image_cond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep,
            **tf_kwargs_real,
        )
        if self.real_guidance_scale != 0.0:
            _, pred_real_image_uncond = self.real_score(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=unconditional_dict,
                timestep=timestep,
                **tf_kwargs_real,
            )
            pred_real_image = pred_real_image_cond + (
                pred_real_image_cond - pred_real_image_uncond
            ) * self.real_guidance_scale
        else:
            pred_real_image = pred_real_image_cond

        # Unconditionally stash pred_real_image for downstream
        # consumers (alt-head plumbing). The CFG-extrapolated
        # EMA-swapped teacher x0 prediction (or just cond when
        # real_guidance_scale == 0). Detached so no autograd
        # connection to anything that reads the stash; overwritten
        # each gen-step DMD pass.
        self._latest_pred_real_image = pred_real_image.detach()

        # Step 3: DMD grad = (fake - real). CF normalizes by
        # |x0 - real|.mean() (eq. 8). Gated by ``normalization`` AND
        # ``self.dmd_normalization_enabled`` — the caller flag is the
        # legacy local override (defaults True), the self-flag is the
        # config knob that lets a run disable the normaliser globally
        # (raw ``pred_fake - pred_real`` so the gradient decays
        # naturally as the teacher converges with the student).
        grad = pred_fake_image - pred_real_image
        if normalization and self.dmd_normalization_enabled:
            p_real = estimated_clean_image_or_video - pred_real_image
            normalizer = torch.abs(p_real).mean(
                dim=[1, 2, 3, 4], keepdim=True,
            )
            # clamp_min bounds the cusp amplification when the student
            # converges to the teacher: small p_real => big 1/normalizer
            # => DMD gradient pulled hard toward the teacher's prior
            # (gray-collapse signature with foreign teachers). The floor
            # is ``dmd_normalization_denom_floor`` (default 0.05 caps the
            # per-sample amplification at 20x). Set it to 1.0 to divide by
            # ``max(|f|, 1)`` — no amplification once |f| < 1 (near
            # convergence) while still down-scaling over-large grads.
            grad = grad / normalizer.clamp_min(
                self.dmd_normalization_denom_floor
            )
        grad = torch.nan_to_num(grad)

        # Diagnostics: pred_real / pred_fake L2 norms (RMS) for the gen
        # and teacher score outputs at the sampled DMD timestep.
        with torch.no_grad():
            _pred_real_l2 = float(
                pred_real_image.float().pow(2).mean().sqrt().item()
            )
            _pred_fake_l2 = float(
                pred_fake_image.float().pow(2).mean().sqrt().item()
            )
        log_dict: Dict[str, Any] = {
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.detach(),
            "pred_real_rms": _pred_real_l2,
            "pred_fake_rms": _pred_fake_l2,
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
    ) -> torch.Tensor:
        """Sample DMD timestep with CF's ``ts_schedule`` clamp + shift."""
        # Lower bound: ts_schedule from-pipeline > config default.
        if self.ts_schedule and denoised_timestep_to is not None:
            min_timestep = denoised_timestep_to
        else:
            min_timestep = self.min_score_timestep
        # Upper bound: ts_schedule from-pipeline > config default
        # (``max_score_timestep``, which defaults to num_train_timestep).
        if self.ts_schedule_max and denoised_timestep_from is not None:
            max_timestep = denoised_timestep_from
        else:
            max_timestep = self.max_score_timestep
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

        # gradient_mask arrives non-None at this API (every caller
        # routes through ``_dmd_score_grad_mask`` which always produces
        # a tensor).
        if gradient_mask is None:
            raise RuntimeError(
                "compute_distribution_matching_loss requires "
                "``gradient_mask`` (every caller must build it from "
                "``self._dmd_score_grad_mask`` so the structurally-OOD "
                "last-chunk boundary stays masked uniformly)."
            )

        # Always-on: real-score MAE vs GT diagnostic.
        if gt_target is not None:
            with torch.no_grad():
                gt_t_dbg = gt_target.to(
                    dtype=pred_real_image_detached.dtype,
                    device=pred_real_image_detached.device,
                )
                if gt_t_dbg.shape == pred_real_image_detached.shape:
                    real_mae_vs_gt = float(
                        (pred_real_image_detached.float() - gt_t_dbg.float())
                        .abs().mean().item()
                    )
                    dmd_log_dict["real_score_mae_vs_gt"] = real_mae_vs_gt

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

        # MAE-based student-vs-teacher gate: scale the DMD loss (hence
        # its gradient) down as the student's per-chunk error approaches
        # / beats the teacher's on the SAME scored chunk. ``w`` is a
        # detached scalar so it only reweights the gradient — it does not
        # add a gradient path of its own. No-op (w=1) when disabled.
        # Applied to the DMD term ONLY; anti-collapse below is untouched.
        _gate_w = self._dmd_mae_gate_weight(
            pred_real=pred_real_image_detached,
            pred_student=original_latent.detach(),
            gt_target=gt_target,
            gradient_mask=gradient_mask,
            log_dict=dmd_log_dict,
        )
        if _gate_w != 1.0:
            dmd_loss = dmd_loss * _gate_w

        # Anti-collapse: stashed unscaled on ``self`` so the caller adds
        # it AFTER the dmd_loss_weight multiplication. See helper
        # ``_compute_anti_collapse_term`` for the math + rationale.
        # Gated by ``anti_collapse_apply_to_dmd_rung`` (default True) so
        # the random-exit rung's anti-collapse contribution can be
        # disabled independently of the flash-DMD rung's.
        if self.anti_collapse_apply_to_dmd_rung:
            self._latest_anti_collapse_total = self._compute_anti_collapse_term(
                original_latent=original_latent,
                gt_target=gt_target,
                log_dict=dmd_log_dict,
                log_prefix="",
                ref_dtype=dmd_loss.dtype,
            )
        else:
            self._latest_anti_collapse_total = None

        return dmd_loss, dmd_log_dict

    def _compute_anti_collapse_term(
        self,
        original_latent: torch.Tensor,
        gt_target: Optional[torch.Tensor],
        log_dict: Dict[str, Any],
        log_prefix: str,
        ref_dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Compute the unscaled anti-collapse loss term for one x0
        chunk. Returns ``None`` when both anti_collapse weights are 0,
        when ``gt_target`` is missing, or when shapes don't match.

        Used at two sites:
          * Random-exit rung's x0 (= ``chunk`` consumed by DMD scoring)
            — called from ``compute_distribution_matching_loss``.
          * Flash-DMD t=gan_t rung's x0 (= ``flash_dmd_gan_chunk``)
            — called from ``compute_generator_loss_streaming`` so the
            same std/mean floors that protect the DMD-supervised rung
            also protect the GAN-supervised rung. Both rungs feed
            different grad-on paths into the student; constraining
            both prevents collapse routes through whichever path the
            optimizer would otherwise exploit.

        ``log_prefix`` (e.g. ``""`` or ``"flash_"``) disambiguates the
        log keys so wandb shows both sites' stats side-by-side.
        """
        if gt_target is None:
            return None
        gt_for_std = gt_target.to(
            dtype=original_latent.dtype,
            device=original_latent.device,
        )
        if gt_for_std.shape != original_latent.shape:
            return None

        # Dispatch on ``anti_collapse_type``. ``std_floor`` keeps the
        # legacy one-sided std-deficit (+ optional mean anchor) below.
        # ``std_corridor`` uses the new bidirectional log-ratio
        # corridor + chunk-drift penalty from ``model.anti_collapse``.
        if self.anti_collapse_type == "std_corridor":
            moment_w = float(self.latent_moment_corridor_loss_weight)
            drift_w = float(self.latent_contrast_drift_loss_weight)
            if moment_w <= 0.0 and drift_w <= 0.0:
                return None
            from model.anti_collapse import compute_std_corridor_anti_collapse
            # The corridor helper expects fp32 inputs for stable log-
            # ratio + reduction stats. Up-cast both tensors locally
            # (gen path's bf16 cast stays at the caller boundary).
            total_corridor, corridor_logs = (
                compute_std_corridor_anti_collapse(
                    pred_x0=original_latent.float(),
                    gt_target=gt_for_std.float(),
                    num_frame_per_block=int(self.num_frame_per_block),
                    moment_weight=moment_w,
                    drift_weight=drift_w,
                    corridor_std_low=self.latent_moment_corridor_std_low,
                    corridor_std_high=self.latent_moment_corridor_std_high,
                    corridor_rms_low=self.latent_moment_corridor_rms_low,
                    corridor_rms_high=self.latent_moment_corridor_rms_high,
                    corridor_mean_tol_ratio=(
                        self.latent_moment_corridor_mean_tol_ratio
                    ),
                    drift_tol=self.latent_contrast_drift_tol,
                )
            )
            for k, v in corridor_logs.items():
                log_dict[f"{k}_{log_prefix}"] = v
            # Telemetry: also log pred/gt std-mean so dashboards keep
            # the same plot keys as the std_floor path.
            with torch.no_grad():
                s_pred = original_latent.float().std(dim=[2, 3, 4])
                s_gt = gt_for_std.float().std(dim=[2, 3, 4])
                log_dict[f"anti_collapse_{log_prefix}pred_std_mean"] = (
                    s_pred.mean()
                )
                log_dict[f"anti_collapse_{log_prefix}gt_std_mean"] = s_gt.mean()
            return total_corridor.to(ref_dtype)

        # ----- std_graded_mse_constant: MSE vs fixed target with floor -----
        # Pulls per-frame std toward ``anti_collapse_std_target`` until
        # the mean MSE drops below ``anti_collapse_std_mse_floor`` — at
        # that point the loss clamps to the floor (gradient zero) and
        # stops fighting micro-deviations. Acts as a barrier against
        # both zero-power collapse (gray) and infinite-power collapse;
        # does NOT track GT's natural step-to-step variation.
        if self.anti_collapse_type == "std_graded_mse_constant":
            if self.anti_collapse_loss_weight <= 0.0:
                return None
            from model.anti_collapse import (
                latent_std_graded_mse_constant_loss,
            )
            loss_std_graded = latent_std_graded_mse_constant_loss(
                pred_x0=original_latent.float(),
                target_std=self.anti_collapse_std_target,
                mse_floor=self.anti_collapse_std_mse_floor,
            )
            log_dict[f"anti_collapse_{log_prefix}std_graded_mse_raw"] = (
                loss_std_graded.detach()
            )
            log_dict[f"anti_collapse_{log_prefix}std_target"] = float(
                self.anti_collapse_std_target
            )
            log_dict[f"anti_collapse_{log_prefix}std_mse_floor"] = float(
                self.anti_collapse_std_mse_floor
            )
            with torch.no_grad():
                s_pred = original_latent.float().std(dim=[2, 3, 4])
                s_gt = gt_for_std.float().std(dim=[2, 3, 4])
                log_dict[f"anti_collapse_{log_prefix}pred_std_mean"] = (
                    s_pred.mean()
                )
                log_dict[f"anti_collapse_{log_prefix}gt_std_mean"] = s_gt.mean()
            return (
                self.anti_collapse_loss_weight
                * loss_std_graded.to(ref_dtype)
            )

        # ----- std_mse: simplest possible (s_pred - s_gt)^2 -----
        if self.anti_collapse_type == "std_mse":
            if self.anti_collapse_loss_weight <= 0.0:
                return None
            from model.anti_collapse import latent_std_mse_loss
            # fp32 reductions for numerical stability — gen path keeps
            # the bf16 cast at the caller boundary.
            loss_std_mse = latent_std_mse_loss(
                pred_x0=original_latent.float(),
                gt_target=gt_for_std.float(),
            )
            log_dict[f"anti_collapse_{log_prefix}std_mse_raw"] = (
                loss_std_mse.detach()
            )
            # Telemetry parity with std_floor / std_corridor.
            with torch.no_grad():
                s_pred = original_latent.float().std(dim=[2, 3, 4])
                s_gt = gt_for_std.float().std(dim=[2, 3, 4])
                log_dict[f"anti_collapse_{log_prefix}pred_std_mean"] = (
                    s_pred.mean()
                )
                log_dict[f"anti_collapse_{log_prefix}gt_std_mean"] = s_gt.mean()
            return (
                self.anti_collapse_loss_weight
                * loss_std_mse.to(ref_dtype)
            )

        # ----- legacy std_floor + optional mean anchor -----
        anti_collapse_any = (
            self.anti_collapse_loss_weight > 0.0
            or self.anti_collapse_mean_weight > 0.0
        )
        if not anti_collapse_any:
            return None
        total: Optional[torch.Tensor] = None
        if self.anti_collapse_loss_weight > 0.0:
            s_pred = original_latent.float().std(dim=[2, 3, 4])
            s_gt = gt_for_std.float().std(dim=[2, 3, 4]).detach()
            # Relaxed floor: only penalise when student std drops
            # below ``ratio * s_gt`` (default ratio=0.5). Leaves a
            # wider band where the floor is silent so it doesn't keep
            # pulling variance up toward GT once the student is
            # already in a reasonable range.
            floor_ratio = float(self.anti_collapse_std_floor_ratio)
            deficit = F.relu(s_gt * floor_ratio - s_pred)
            anti_collapse_loss = deficit.pow(2).mean()
            log_dict[f"anti_collapse_{log_prefix}loss_raw"] = (
                anti_collapse_loss.detach()
            )
            log_dict[f"anti_collapse_{log_prefix}pred_std_mean"] = (
                s_pred.detach().mean()
            )
            log_dict[f"anti_collapse_{log_prefix}gt_std_mean"] = s_gt.mean()
            total = (
                self.anti_collapse_loss_weight
                * anti_collapse_loss.to(ref_dtype)
            )
        if self.anti_collapse_mean_weight > 0.0:
            m_pred = original_latent.float().mean(dim=[2, 3, 4])
            if self.anti_collapse_mean_target == "gt":
                m_gt = gt_for_std.float().mean(dim=[2, 3, 4]).detach()
                mean_sq_loss = (m_pred - m_gt).pow(2).mean()
                log_dict[f"anti_collapse_{log_prefix}gt_mean_mean"] = (
                    m_gt.mean()
                )
            else:
                mean_sq_loss = m_pred.pow(2).mean()
            log_dict[f"anti_collapse_{log_prefix}mean_sq_raw"] = (
                mean_sq_loss.detach()
            )
            log_dict[f"anti_collapse_{log_prefix}pred_mean_mean"] = (
                m_pred.detach().mean()
            )
            mean_term = (
                self.anti_collapse_mean_weight
                * mean_sq_loss.to(ref_dtype)
            )
            total = mean_term if total is None else total + mean_term
        return total

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
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Prepare per-scorer ``clean_x`` / ``aug_t`` and merge the clean
        action streams into the scoring cond dicts. Returns:
            (clean_x_fake, aug_t_fake,
             clean_x_real, aug_t_real,
             new_cond_for_scoring, new_uncond_for_scoring,
             clean_x_aux, aug_t_aux)

        ``dmd_context`` is hardcoded to "self" on this branch:
          * ``clean_x_fake`` / ``aug_t_fake`` are the self view
            (``clean_x_self`` at ``aug_t=0``).
          * ``clean_x_real`` / ``aug_t_real`` are ``None`` so
            ``_compute_kl_grad`` falls back to the self view for the
            real branch.
          * ``clean_x_aux`` / ``aug_t_aux`` are the self view when
            ``build_real_view=True`` so the aux pass's gate (
            ``sc_clean_x_real is not None``) opens; the aux function
            then builds its own pure-GT clean half from the ride window.

        When ``build_real_view=False`` (critic step — only trains fake)
        all real/aux outputs are ``None``.

        The cond dicts are SHALLOW-COPIED (never mutate the caller's
        dict) and have ``_action_modulation_clean`` /
        ``_action_tokens_clean`` injected from ``clean_*_dict``.
        """
        if clean_x_self is None:
            raise RuntimeError(
                "_build_dmd_context_kwargs requires clean_x_self (the "
                "21-frame self-view assembled by the trainer)."
            )
        # The 42f path OVERRIDES every output of this function with its own
        # GT-built clean_x/noisy_x/gt_target (see compute_*_loss_streaming),
        # so clean_x_self is dead here. That lets streaming_chunk_size (the
        # student's causal-roll count) be DECOUPLED from num_training_frames
        # (the 21-frame teacher window) — e.g. roll 3 chunks (chunk_size=9)
        # while the teacher still scores a 21-frame window filled from GT +
        # the 1 supervised chunk. Only enforce the ==N shape when NOT 42f
        # (symmetric / asymmetric paths genuinely consume clean_x_self).
        if not bool(getattr(self, "dmd_42f_enabled", False)):
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

        # dmd_context=self: real-side DMD view = None (falls back to
        # self in _compute_kl_grad). Aux view = self (gates the aux
        # pass on; the aux function builds its own GT clean half).
        sc_clean_x_real: Optional[torch.Tensor] = None
        sc_aug_t_real: Optional[torch.Tensor] = None
        if build_real_view:
            sc_clean_x_aux: Optional[torch.Tensor] = sc_clean_x
            sc_aug_t_aux: Optional[torch.Tensor] = sc_aug_t
        else:
            sc_clean_x_aux = None
            sc_aug_t_aux = None

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
        return (
            sc_clean_x, sc_aug_t,
            sc_clean_x_real, sc_aug_t_real,
            new_cond, new_uncond,
            sc_clean_x_aux, sc_aug_t_aux,
        )

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
            _sc_clean_x_aux,  # aux fields unused on this DMD-scoring path
            _sc_aug_t_aux,
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
        # GT slice for the always-on real_score_mae_vs_gt diagnostic.
        gt_target = None
        if (
            clean_latent is not None
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
        # v28+: add the anti-collapse term UNSCALED by dmd_loss_weight,
        # so the user's ``anti_collapse_*_weight`` knobs mean what they
        # say regardless of where DMD is in its warmup ramp.
        if getattr(self, "_latest_anti_collapse_total", None) is not None:
            dmd_loss = dmd_loss + self._latest_anti_collapse_total
            self._latest_anti_collapse_total = None

        # v28E_12+: multi-horizon stat-anchor hinge loss. Pins per-frame
        # M2 (Σ σ²) and TV (total variation) against the seed window's
        # stat values, with hinge bounds derived from GT analysis
        # (rolling_bands.json W=16 p95). Only fires when the student's
        # drift exceeds the GT p95 budget. See
        # ``model.anti_collapse.compute_stat_anchor_loss`` for details.
        # Non-streaming generator_loss path doesn't carry ``current_step``
        # in its signature. Resolver returns the full static weight when
        # called with step 0 (rampdown hasn't started yet), so safe
        # fallback. Streaming-mode runs use the resolver-with-step at the
        # streaming application site.
        stat_anchor_w_resolved = self._resolved_stat_anchor_loss_weight(
            int(getattr(self, "_last_current_step", 0))
        )
        if stat_anchor_w_resolved > 0.0 and pred_image is not None:
            from model.anti_collapse import compute_stat_anchor_loss
            try:
                _wk = dict(
                    STD_short_weight=self.stat_anchor_STD_short_weight,
                    STD_long_weight=self.stat_anchor_STD_long_weight,
                    M2_short_weight=self.stat_anchor_M2_short_weight,
                    M2_long_weight=self.stat_anchor_M2_long_weight,
                    TV_short_weight=self.stat_anchor_TV_short_weight,
                    TV_long_weight=self.stat_anchor_TV_long_weight,
                    SOS_short_weight=self.stat_anchor_SOS_short_weight,
                    SOS_long_weight=self.stat_anchor_SOS_long_weight,
                    M1_short_weight=self.stat_anchor_M1_short_weight,
                    M1_long_weight=self.stat_anchor_M1_long_weight,
                    rel_tol_short=self.stat_anchor_rel_tol_short,
                    rel_tol_long=self.stat_anchor_rel_tol_long,
                )
                # Anchor SOURCE by mode: 'target_matching' -> per-chunk
                # k-closest-GT anchors for ALL stats; else -> the GT seed.
                _matched = (
                    self._matched_gt_stat_anchors(pred_image.float())
                    if getattr(self, "stat_anchor_mode", "seed_anchor")
                    == "target_matching" else None
                )
                if _matched is not None:
                    stat_loss, stat_logs = compute_stat_anchor_loss(
                        pred_x0=pred_image.float(), seed_latents=None,
                        seed_STD_anchor=_matched["STD"],
                        seed_M2_anchor=_matched["M2"],
                        seed_TV_anchor=_matched["TV"],
                        seed_SOS_anchor=_matched["SOS"],
                        seed_M1_anchor=_matched["M1"], **_wk,
                    )
                elif seed_latents is not None:
                    stat_loss, stat_logs = compute_stat_anchor_loss(
                        pred_x0=pred_image.float(),
                        seed_latents=seed_latents.float(), **_wk,
                    )
                else:
                    stat_loss, stat_logs = None, {}
                if stat_loss is not None:
                    stat_loss = (
                        stat_anchor_w_resolved * stat_loss.to(dmd_loss.dtype)
                    )
                    dmd_loss = dmd_loss + stat_loss
                    dmd_log_dict["stat_anchor_total"] = stat_loss.detach()
                    dmd_log_dict["stat_anchor_weight_resolved"] = float(
                        stat_anchor_w_resolved
                    )
                    dmd_log_dict["stat_anchor_mode"] = float(
                        1.0 if _matched is not None else 0.0
                    )
                    for k, v in stat_logs.items():
                        dmd_log_dict[k] = v
            except Exception as exc:
                # Fail-soft: log and continue. If the loss is genuinely
                # broken the operator will see the wandb key go away.
                if _is_main():
                    import logging as _logging
                    _logging.warning(
                        "[ActionForcingDMD] stat_anchor_loss failed (%s); "
                        "skipping this iteration.", exc,
                    )

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
            _unused_clean_x_aux,
            _unused_aug_t_aux,
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
        critic_log: Dict[str, Any] = {}
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
        # New streaming sequence = new ride / new video. Clear the
        # WAN VAE's temporal feat_map so the first decode of this
        # sequence doesn't inherit the previous sequence's tail as
        # left-context (would corrupt this video's first frame).
        # Within the same sequence, subsequent decodes (sample-video
        # render, boundary roundtrip, perceptual loss decodes, ...)
        # SHARE the cache so they flow smoothly through each other's
        # left-context — no per-call init-frame brightness anomaly.
        if (
            hasattr(self, "vae")
            and getattr(self, "vae", None) is not None
            and hasattr(self.vae, "model")
            and hasattr(self.vae.model, "clear_cache")
        ):
            self.vae.model.clear_cache()
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

        # v24: pre-roll rollout 2 (no_grad, fewer seed chunks) BEFORE
        # rollout 1's cache initialization. The prebuild reuses the
        # SAME pipe.kv_cache1 / crossattn_cache; its finally clause
        # resets them so the rollout-1 prefill below starts clean.
        # Result is stashed in streaming_state["rollout2_x0"] for the
        # critic step's alt-head loss to consume.
        rollout2_x0 = None
        rollout2_abs_frame_start = None
        # Prebuild rollout2 (one fewer seed chunk = +1 CARN drift step) when
        # EITHER the fake-score alt head consumes it (legacy) OR the FN's
        # teacher_feat regime needs the rollout1->rollout2 pairs. The latter
        # is independent of the alt head (so the FN can be trained without
        # any fake_score coupling).
        _need_rollout2 = (
            self.fake_alt_head_enabled
            and self.fake_alt_target_mode == "rollout2_student"
        ) or (
            self.forward_noiser_enabled
            and self.forward_noiser_loss_mode == "teacher_feat"
        )
        if _need_rollout2:
            rollout2_x0, rollout2_abs_frame_start = (
                self._prebuild_rollout2_for_v24(
                    seed_latents=seed_latents,
                    ride_latents_window=ride_latents_window,
                    ride_actions_window=ride_actions_window,
                    prompt_embeds=prompt_embeds,
                    max_length=max_length,
                )
            )

        # Reset + initialise persistent caches.
        pipe.reset_cache_state()
        pipe._initialize_kv_cache(
            batch_size=batch_size, dtype=dtype, device=device,
        )
        pipe._initialize_crossattn_cache(
            batch_size=batch_size, dtype=dtype, device=device,
        )

        # Seed prefill: cf GT frames through the model's denoise→commit
        # sequence (via ``pipe._seed_prefill_chunk``) so the KV cache
        # ends up populated with model-style cache_pred traces matching
        # what training rollouts produce. The legacy raw-GT-at-t=0
        # prefill placed OOD attention traces into the cache, causing
        # the first generated chunk to start at a low-energy boundary
        # state. See ``_seed_prefill_chunk`` for the rationale.
        # All forwards run under no_grad inside the helper, so the
        # temporary cond_dict's grad_fn is orphaned and GC'd once
        # setup_sequence returns.
        with torch.no_grad():
            seed_cond_dict, _ = self.build_action_conditional(
                prompt_embeds=prompt_embeds,
                gt_actions=ride_actions_window,
            )
        num_seed_chunks = cf // npb
        current_start_frame = 0
        for sc in range(num_seed_chunks):
            seed_chunk = seed_latents[:, sc * npb : (sc + 1) * npb]
            seed_block_cond = _slice_per_frame_streams(
                seed_cond_dict, frame_start=current_start_frame, frame_count=npb,
            )
            pipe._seed_prefill_chunk(
                seed_chunk=seed_chunk,
                seed_block_cond=seed_block_cond,
                current_start_frame=current_start_frame,
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
        # ``dmd_42f_gt_anchor``: make the +npb leading-anchor chunk a CLEAN
        # GT chunk (seed-prefilled, same denoise->commit path as the seed
        # loop) instead of a student-rolled chunk. The student anchor
        # writes a drifted student chunk into KV frames [cf, cf+npb) that
        # the supervised rollout chunks attend to AND that the 42f teacher
        # scores with GT in that slot (n_ctx is GT) — a generate-vs-score
        # mismatch. A GT anchor makes the supervised chunk's KV context
        # purely clean GT (matching the 7-chunk inference eval) and aligns
        # the rollout context with the GT the teacher scores against, while
        # keeping the EXACT iter-1 geometry (``current_length=npb`` headroom
        # the 42f slice bounds need). Default OFF (legacy student anchor).
        if bool(getattr(self, "dmd_42f_gt_anchor", False)):
            anchor_gt = ride_latents_window[:, cf:cf + npb]
            if int(anchor_gt.shape[1]) < npb:
                raise RuntimeError(
                    "dmd_42f_gt_anchor: ride_latents_window too short for a "
                    f"GT anchor at [cf:cf+npb]=[{cf}:{cf + npb}] (have "
                    f"{int(ride_latents_window.shape[1])} frames)."
                )
            with torch.no_grad():
                anchor_cond_dict, _ = self.build_action_conditional(
                    prompt_embeds=prompt_embeds,
                    gt_actions=ride_actions_window,
                )
                anchor_block_cond = _slice_per_frame_streams(
                    anchor_cond_dict, frame_start=cf, frame_count=npb,
                )
                # Same model-style denoise->commit prefill as the seed loop,
                # so KV frames [cf, cf+npb) carry a GT trace in the same
                # representation regime as the rest of the seed.
                pipe._seed_prefill_chunk(
                    seed_chunk=anchor_gt,
                    seed_block_cond=anchor_block_cond,
                    current_start_frame=cf,
                )
            del anchor_cond_dict
            anchor_chunk = anchor_gt.detach()
        else:
            anchor_noise = torch.randn(
                [batch_size, npb, *seed_latents.shape[2:]],
                device=device, dtype=dtype,
            )
            with torch.no_grad():
                anchor_full_cond, _ = self.build_action_conditional(
                    prompt_embeds=prompt_embeds,
                    gt_actions=ride_actions_window,
                )
                # Anchor cold-start (pure noise + full ladder).
                anchor_chunk, _, _ = pipe.generate_chunk_with_cache(
                    noise=anchor_noise,
                    current_start_frame=cf,
                    requires_grad=False,
                    prefer_cache_pred_in_output=False,
                    gt_latents=None,  # no MAE on the anchor
                    flash_dmd_enabled=bool(self.flash_dmd_enabled),
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
            "previous_last_rung_chunk": None,  # last_rung view of full_chunk (Flash-DMD §3.3)
            # Warm-start removed: ``previous_clean_chunk`` field is no
            # longer populated or read. Each chunk now cold-starts from
            # pure noise + the full denoising ladder.
            "abs_frame_after_seed": cf,  # absolute pipeline frame index after seed prefill (anchor adds npb on top)
            "anchor_chunk": anchor_chunk,  # [B, npb, C, H, W] — iter 1's clean_x_self anchor
            # v24: pre-rolled noisier student rollout (fewer seed chunks).
            # ``None`` when fake_alt_target_mode != "rollout2_student".
            # When set: tensor [B, n_gen, C, H, W] with abs frame index
            # ``rollout2_abs_frame_start`` for slot 0.
            "rollout2_x0": rollout2_x0,
            "rollout2_abs_frame_start": rollout2_abs_frame_start,
        }

    def _forward_noiser_ddp_anchor(
        self, src: torch.Tensor, npb: int,
    ) -> torch.Tensor:
        """Zero-valued FN loss that STILL routes through ``self.forward_noiser``.

        Used at every data-dependent bail in ``_compute_forward_noiser_loss``
        so the forward_noiser module participates in the critic backward on
        EVERY rank, even when this rank's ride yielded no aligned (rollout1,
        rollout2) pairs. Without it, the FN grad-bucket all-reduce fires on a
        per-rank-data-dependent subset of ranks -> the others block forever ->
        silent DDP hang (no NCCL error, no OOM). The ``* 0.0`` keeps the loss
        value zero (no spurious signal) while giving the DDP reducer identical
        participation across ranks. ``find_unused_parameters=True`` does not
        suffice — it only reconciles params unused within a forward that ran,
        not a forward that was skipped entirely.
        """
        carn0 = torch.zeros(
            (src.shape[0],), dtype=torch.long, device=src.device,
        )
        pred = self.forward_noiser(
            src[:, :npb].detach(), carn0, residual=True,
        )
        return pred.sum() * 0.0

    def _compute_forward_noiser_loss(
        self,
        chunk: torch.Tensor,
        info: Dict[str, Any],
        critic_log: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        """Train the forward noiser: predict rollout-2 chunks (at CARN
        level n+1) from rollout-1 chunks (at CARN level n) at the same
        abs frame positions, with per-chunk CARN-step conditioning.

        Returns the unscaled MSE loss (None if no aligned pair was
        available this iter — e.g. chunk fell entirely outside
        rollout-2's coverage).
        """
        if self.forward_noiser is None:
            return None
        # teacher_feat mode trains the FN trainer-side in a SEPARATE backward
        # (decoupled from fake_score), so the critic-folded MSE path is off.
        if self.forward_noiser_loss_mode == "teacher_feat":
            return None
        npb = int(self.num_frame_per_block)
        # DDP anchor on every data-dependent bail: each of the 32 ranks
        # rides a DIFFERENT video, so the aligned-pair count below is
        # per-rank. If a rank returns None here, self.forward_noiser never
        # enters its critic backward graph, so its grad-bucket all-reduce
        # never fires while other ranks' does -> silent DDP hang (no NCCL
        # error, no OOM). ``_forward_noiser_ddp_anchor`` returns a zero loss
        # that STILL routes through forward_noiser, keeping participation
        # identical across ranks. (find_unused_parameters=True does NOT
        # cover a skipped-forward; only a real forward does.)
        s = self.streaming_state
        if s is None:
            return self._forward_noiser_ddp_anchor(chunk, npb)
        r2_x0 = s.get("rollout2_x0")
        r2_abs_start = s.get("rollout2_abs_frame_start")
        if r2_x0 is None or r2_abs_start is None:
            return self._forward_noiser_ddp_anchor(chunk, npb)
        r2_abs_start = int(r2_abs_start)
        r2_total = int(r2_x0.shape[1])

        # Prefer the t=60 refined chunk for FN's rollout1 input when
        # available (flash_dmd_enabled). Rollout2's r2_x0 was already
        # built from the t=60 stash in the prebuild path. Falls back to
        # the random-rung chunk when flash_dmd is off (legacy parity).
        flash_chunk = info.get("flash_dmd_gan_x0")
        fn_input_chunk = (
            flash_chunk.detach() if flash_chunk is not None else chunk
        )

        chunk_size_critic = int(fn_input_chunk.shape[1])
        if chunk_size_critic % npb != 0:
            return self._forward_noiser_ddp_anchor(fn_input_chunk, npb)
        n_chunks = chunk_size_critic // npb
        abs_new_start = int(info.get("abs_frame_start", 0))
        overlap_critic = int(info.get("overlap", 0))
        chunk_abs_start = abs_new_start - overlap_critic
        # Streaming invariant: abs positions are npb-aligned. Floor-div
        # would silently drop a remainder otherwise, producing wrong
        # CARN levels. Assert to surface any violation loudly.
        if chunk_abs_start % npb != 0:
            raise RuntimeError(
                f"_compute_forward_noiser_loss: chunk_abs_start="
                f"{chunk_abs_start} not divisible by npb={npb}. Streaming "
                "alignment invariant violated — investigate "
                "abs_frame_start ({abs_new_start}) and overlap "
                "({overlap_critic}) sources."
            )
        num_seed_r1 = int(self.dmd_context_clean_frames // npb)

        # Iterate per chunk in the window. Each chunk has its own CARN
        # level (based on its abs position) and its own rollout-2
        # counterpart looked up by abs frame position.
        losses: list = []
        for c in range(n_chunks):
            f_start = c * npb
            f_end = f_start + npb
            abs_f_start = chunk_abs_start + f_start
            abs_f_end = chunk_abs_start + f_end
            if (abs_f_start < r2_abs_start
                    or abs_f_end > r2_abs_start + r2_total):
                continue
            slot_start = abs_f_start - r2_abs_start
            slot_end = slot_start + npb
            target_r2 = r2_x0[:, slot_start:slot_end].to(
                dtype=fn_input_chunk.dtype, device=fn_input_chunk.device,
            ).detach()
            input_r1 = fn_input_chunk[:, f_start:f_end].detach()
            chunk_abs_idx = abs_f_start // npb
            carn_r1 = max(0, chunk_abs_idx - (num_seed_r1 - 1))
            carn_step = torch.full(
                (fn_input_chunk.shape[0],), carn_r1,
                dtype=torch.long, device=fn_input_chunk.device,
            )
            predicted = self.forward_noiser(
                input_r1, carn_step, residual=True,
            )
            losses.append(F.mse_loss(predicted, target_r2))

        # v27B audit-fix #2: explicit CARN_step=0 training pair.
        # The streaming critic window only covers the ROLLED region
        # (positions >= cf, chunk_idx >= num_seed_r1). At those
        # positions rollout 1 has CARN_step ∈ [1, 7] — never 0. But at
        # application time, every chunk starts at CARN_step=0 (clean
        # GT input) and gets noised iteratively. Without a CARN_step=0
        # training pair the noiser's first iteration is uncontrolled.
        #
        # Add the missing pair: rollout 1's LAST seed chunk (positions
        # [cf-npb, cf), clean GT) maps to rollout 2's anchor chunk
        # (positions [cf_r2, cf_r2+npb) — the first AR-generated chunk
        # of rollout 2). When cf_r2 = cf - npb (default v24 config with
        # num_seed_chunks_r1=3, num_seed_chunks_r2=2), the abs positions
        # align: r1 seed-last @ [6, 9) ↔ r2 anchor @ [6, 9).
        cf_r1 = int(self.dmd_context_clean_frames)
        if (
            r2_abs_start == cf_r1 - npb
            and r2_total >= npb
            and "ride_latents_window" in s
        ):
            ride_window = s["ride_latents_window"]
            seed_last_lo = cf_r1 - npb
            seed_last_hi = cf_r1
            if int(ride_window.shape[1]) >= seed_last_hi:
                seed_last_r1 = ride_window[
                    :, seed_last_lo:seed_last_hi
                ].to(dtype=chunk.dtype, device=chunk.device).detach()
                anchor_r2 = r2_x0[:, 0:npb].to(
                    dtype=chunk.dtype, device=chunk.device,
                ).detach()
                carn_step_0 = torch.zeros(
                    (chunk.shape[0],), dtype=torch.long, device=chunk.device,
                )
                predicted_anchor = self.forward_noiser(
                    seed_last_r1, carn_step_0, residual=True,
                )
                losses.append(F.mse_loss(predicted_anchor, anchor_r2))

        if not losses:
            # No aligned pair for THIS rank's ride this step. Do NOT return
            # None — that skips the forward_noiser forward and desyncs the
            # FN grad-bucket all-reduce across ranks (silent DDP hang). Run
            # the zero-anchor so participation is identical on every rank.
            return self._forward_noiser_ddp_anchor(fn_input_chunk, npb)
        fn_loss = torch.stack(losses).mean()
        critic_log["forward_noiser_loss_raw"] = fn_loss.detach()
        critic_log["forward_noiser_n_pairs"] = float(len(losses))
        # One-time stderr confirmation that the forward noiser is actually
        # TRAINING (n_pairs>0 + non-trivial loss). The wandb-only metrics
        # don't reach .err, so this gives an offline FN-alive signal.
        _fc = getattr(self, "_fn_train_dbg", 0)
        if _fc < 2:
            self._fn_train_dbg = _fc + 1
            import sys as _sys
            print(
                f"[FN-TRAIN] forward_noiser ALIVE: n_pairs={len(losses)} "
                f"loss_raw={float(fn_loss.detach().item()):.5f}",
                file=_sys.stderr, flush=True,
            )
        return fn_loss

    @staticmethod
    def _temporal_lowpass_5d(
        x: torch.Tensor, window: int,
    ) -> torch.Tensor:
        """Temporal moving-average low-pass along F for [B,F,C,H,W].

        Boxcar of length ``window`` with reflection padding so output F
        matches input F. Length-W boxcar has nulls at f=k/W, so for
        period-3 high-freq content, W>=3 nulls it (W=6 nulls it and
        also strongly attenuates anything with period<6).
        """
        if window <= 1:
            return torch.zeros_like(x)
        B, F_, C, H, W_ = x.shape
        # [B*C*H*W, 1, F] for conv1d-style avg_pool1d.
        x_perm = x.permute(0, 2, 3, 4, 1).contiguous().view(-1, 1, F_)
        pad_l = (window - 1) // 2
        pad_r = window - 1 - pad_l
        # reflect pad requires pad < F_; fall back to replicate when F is
        # small (rare; defensive).
        if pad_l >= F_ or pad_r >= F_:
            x_padded = F.pad(x_perm, (pad_l, pad_r), mode="replicate")
        else:
            x_padded = F.pad(x_perm, (pad_l, pad_r), mode="reflect")
        smoothed = F.avg_pool1d(
            x_padded, kernel_size=window, stride=1,
        )
        smoothed = smoothed.view(B, C, H, W_, F_).permute(
            0, 4, 1, 2, 3
        ).contiguous()
        return smoothed

    def _apply_forward_noiser_to_gt(
        self,
        gt_target: torch.Tensor,
        abs_frame_start_gt: int,
    ) -> Optional[torch.Tensor]:
        """Build causal_AR_GT by iteratively applying the forward
        noiser to gt_target chunks. Each chunk in the window is noised
        up to its target CARN level (matching the rollout-1 student's
        CARN level at that abs position).

        Args:
            gt_target: [B, F, C, H, W] clean GT chunks at consecutive
                abs frame positions starting at abs_frame_start_gt.
            abs_frame_start_gt: abs frame index of gt_target[:, 0].

        Returns:
            [B, F, C, H, W] with each chunk noised to its target CARN
            level. Returns ``None`` if forward_noiser is unavailable or
            if shapes are misaligned (caller falls back to non-noised
            gt_target).
        """
        strategy = self.forward_noiser_apply_strategy
        if strategy == "off":
            return None
        if self.forward_noiser is None:
            return None
        B, F_total, C, H, W = gt_target.shape
        npb = int(self.num_frame_per_block)
        if F_total % npb != 0:
            return None
        # Same npb-alignment invariant as _compute_forward_noiser_loss.
        if int(abs_frame_start_gt) % npb != 0:
            raise RuntimeError(
                f"_apply_forward_noiser_to_gt: abs_frame_start_gt="
                f"{abs_frame_start_gt} not divisible by npb={npb}. "
                "Floor-div on chunk_abs_idx would drop a remainder and "
                "produce incorrect CARN levels."
            )
        n_chunks = F_total // npb
        num_seed_r1 = int(self.dmd_context_clean_frames // npb)
        device = gt_target.device

        target_carn_per_chunk: list = []
        for c in range(n_chunks):
            if self.aux_carn_level_mode == "local_index":
                # Window-local ladder: chunk 0 -> 0 CARN, chunk 1 -> 1, ...
                # Only the first chunk is the clean anchor; the rest
                # accumulate +1 CARN application each, recursively.
                target_carn = c
            else:
                # "position": drift-faithful to the rollout's CARN at the
                # chunk's absolute ride position (seed region = level 0).
                chunk_abs_idx = (int(abs_frame_start_gt) + c * npb) // npb
                target_carn = max(0, chunk_abs_idx - (num_seed_r1 - 1))
            target_carn_per_chunk.append(int(target_carn))

        max_carn = max(target_carn_per_chunk) if target_carn_per_chunk else 0
        if max_carn == 0:
            return gt_target

        current = gt_target.clone()
        if not self.carn_recurse:
            # SINGLE-CALL mode: one conditioned call per chunk straight from
            # the CLEAN GT, carn_step = the chunk's target level. No
            # recursion / no feeding the output back. Matches the cumulative
            # FN training (clean -> drifted-at-level-k). Level-0 chunks are
            # left untouched.
            with torch.no_grad():
                for c, tc in enumerate(target_carn_per_chunk):
                    if tc <= 0:
                        continue
                    f_start = c * npb
                    f_end = f_start + npb
                    chunk_in = gt_target[:, f_start:f_end].contiguous()
                    carn_step = torch.full(
                        (B,), int(tc), dtype=torch.long, device=device,
                    )
                    current[:, f_start:f_end] = self.forward_noiser(
                        chunk_in, carn_step, residual=True,
                    )
            return current
        with torch.no_grad():
            for k in range(max_carn):
                for c, tc in enumerate(target_carn_per_chunk):
                    if tc <= k:
                        continue
                    f_start = c * npb
                    f_end = f_start + npb
                    chunk_in = current[:, f_start:f_end].contiguous()
                    # Only the learned FN strategy is supported. The
                    # blur_noise and sum strategies have been removed —
                    # they depended on a 2D spatial Gaussian blur op
                    # that is no longer in the codebase.
                    # Step-unconditioned FN -> carn_step=0 every apply.
                    carn_step = torch.full(
                        (B,),
                        0 if bool(getattr(
                            self, "forward_noiser_step_unconditioned", False,
                        )) else k,
                        dtype=torch.long, device=device,
                    )
                    chunk_out = self.forward_noiser(
                        chunk_in, carn_step, residual=True,
                    )
                    current[:, f_start:f_end] = chunk_out
        return current

    def _prebuild_rollout2_for_v24(
        self,
        seed_latents: torch.Tensor,
        ride_latents_window: torch.Tensor,
        ride_actions_window: torch.Tensor,
        prompt_embeds: torch.Tensor,
        max_length: int,
    ) -> Tuple[torch.Tensor, int]:
        """v24 pre-roll: run a SECOND student rollout under no_grad with
        ``fake_alt_rollout2_num_seed_chunks`` seed chunks (vs rollout 1's
        ``dmd_context_clean_frames / npb`` count). With 1 fewer seed
        chunk, rollout 2 carries 1 extra AR step of drift; its x0
        estimates capture the worst-case causal-AR noise distribution
        that the fake-score alt head learns to predict as the v24
        replacement for the v23 ``ema_real_x0`` target.

        Shares ``inference_pipeline.kv_cache1`` / ``crossattn_cache``
        with rollout 1, so caches are reset in the finally clause to
        leave the pipeline clean for the rollout-1 prefill that
        immediately follows in setup_sequence.

        Returns ``(rollout2_x0, abs_frame_start)`` where
        ``rollout2_x0`` has shape ``[B, n_gen, C, H, W]`` and
        ``abs_frame_start = num_seed_r2 * npb`` (the absolute ride
        frame index of slot 0).
        """
        pipe = self.inference_pipeline
        npb = int(self.num_frame_per_block)
        n_seed_r2 = int(self.fake_alt_rollout2_num_seed_chunks)
        cf_r2 = n_seed_r2 * npb
        # Rollout 2 needs 1 extra generated chunk to reach the same end
        # absolute position as rollout 1 (it started one seed-chunk
        # earlier in the ride).
        # CAP at the standard scoring window: under phase-2 rolling,
        # ``max_length`` is the RIDE cap (hundreds of frames) — without
        # this cap the prebuild would pre-roll the whole ride no_grad at
        # every setup (~300 chunks, looks like a silent multi-minute
        # hang at 100% GPU). FN pairs only ever consume the first-window
        # span, so nothing is lost. Stationary configs (max_length ==
        # num_training_frames) are unaffected.
        max_length_r2 = min(
            int(max_length), int(self.num_training_frames)
        ) + npb
        device = seed_latents.device
        dtype = seed_latents.dtype
        batch_size = int(seed_latents.shape[0])
        if ride_latents_window.shape[1] < cf_r2:
            raise ValueError(
                f"ride_latents_window has {ride_latents_window.shape[1]} "
                f"frames; rollout 2 needs >= cf_r2={cf_r2}."
            )

        seed_r2 = seed_latents[:, :cf_r2]
        # Temporarily blank streaming_state so generate_next_chunk
        # operates on rollout 2's transient state (built below).
        saved_streaming_state = self.streaming_state
        self.streaming_state = None
        # Pin the stride knobs to legacy prebuild behavior for the
        # duration: phase-2 rolling sets a deterministic multi-chunk
        # stride (num_chunks_roll_forward) + a large min_new_frame, which
        # would otherwise gate ``can_generate_more`` / the picker inside
        # this short prebuild and skip its streamed chunks entirely.
        saved_min_new = self.streaming_min_new_frame
        saved_force = self.streaming_force_new_frame_chunks
        self.streaming_min_new_frame = npb
        self.streaming_force_new_frame_chunks = 0

        rollout2_chunks: list = []
        try:
            with torch.no_grad():
                # 1) Reset + (re-)init caches for rollout 2.
                pipe.reset_cache_state()
                pipe._initialize_kv_cache(
                    batch_size=batch_size, dtype=dtype, device=device,
                )
                pipe._initialize_crossattn_cache(
                    batch_size=batch_size, dtype=dtype, device=device,
                )

                # 2) Seed prefill: n_seed_r2 chunks via the model's
                # denoise→commit sequence (helper). Same fix as the
                # rollout-1 seed loop in setup_sequence — replaces the
                # legacy raw-GT-at-t=0 prefill with model-style
                # cache_pred KV traces.
                seed_cond_dict, _ = self.build_action_conditional(
                    prompt_embeds=prompt_embeds,
                    gt_actions=ride_actions_window,
                )
                current_start_frame = 0
                for sc in range(n_seed_r2):
                    seed_chunk = seed_r2[:, sc * npb : (sc + 1) * npb]
                    seed_block_cond = _slice_per_frame_streams(
                        seed_cond_dict,
                        frame_start=current_start_frame, frame_count=npb,
                    )
                    pipe._seed_prefill_chunk(
                        seed_chunk=seed_chunk,
                        seed_block_cond=seed_block_cond,
                        current_start_frame=current_start_frame,
                    )
                    current_start_frame += npb
                del seed_cond_dict

                # 3) Anchor chunk at cf_r2.
                anchor_noise = torch.randn(
                    [batch_size, npb, *seed_latents.shape[2:]],
                    device=device, dtype=dtype,
                )
                anchor_full_cond, _ = self.build_action_conditional(
                    prompt_embeds=prompt_embeds,
                    gt_actions=ride_actions_window,
                )
                anchor_chunk, _, _ = pipe.generate_chunk_with_cache(
                    noise=anchor_noise,
                    current_start_frame=cf_r2,
                    requires_grad=False,
                    # Return the t=denoising_step_list[-1] (~178.6)
                    # finish-denoised cache_pred instead of the random-
                    # exit-rung output. The rollout already denoises
                    # all the way to populate the KV cache, so the
                    # cleanest x0 estimate is free to surface here.
                    # When flash_dmd_enabled is True, Step 3.2.b
                    # reassigns cache_pred to the t=60 refined output,
                    # so anchor_chunk is the t=60 version in that
                    # branch — matches the previous flash-on behavior.
                    prefer_cache_pred_in_output=True,
                    gt_latents=None,
                    flash_dmd_enabled=bool(self.flash_dmd_enabled),
                    **anchor_full_cond,
                )
                del anchor_full_cond
                # ``anchor_chunk`` is now the cleanest x0 estimate
                # the rollout produced (t=60 refined when flash on,
                # t=~178.6 finish-denoised when flash off). No need
                # to override via ``_flash_dmd_gan_output`` — it would
                # be the same tensor when flash is on, and a no-op
                # fallback when flash is off.
                rollout2_chunks.append(anchor_chunk.detach())

                # 4) Transient streaming_state for the streaming loop.
                clean_actions_window_r2 = ride_actions_window[
                    :, cf_r2 - npb : cf_r2 - npb + max_length_r2
                ]
                self.streaming_state = {
                    "current_length": int(npb),
                    "max_length": int(max_length_r2),
                    "chunk_size": int(self.streaming_chunk_size),
                    "shift": int(npb),
                    "cf": int(cf_r2),
                    "seed_latents": seed_r2,
                    "ride_latents_window": ride_latents_window,
                    "ride_actions_window": ride_actions_window,
                    "clean_actions_window": clean_actions_window_r2,
                    "prompt_embeds": prompt_embeds,
                    "previous_chunk": None,
                    "previous_last_rung_chunk": None,
                    "abs_frame_after_seed": int(cf_r2),
                    "anchor_chunk": anchor_chunk.detach(),
                    "rollout2_x0": None,
                    "rollout2_abs_frame_start": None,
                }

                # 5) Streaming loop: each iter yields chunk_size frames.
                # Snapshot the t=60 refined tail (info["flash_dmd_gan_x0"])
                # when flash_dmd_enabled; fall back to the random-rung
                # chunk tail otherwise.
                while self.can_generate_more():
                    full_chunk, info = self.generate_next_chunk(
                        requires_grad=False,
                        compute_baseline_mae=False,
                    )
                    new_frames_count = int(info.get("new_frames", npb))
                    flash_slab = info.get("flash_dmd_gan_x0")
                    if flash_slab is not None:
                        rollout2_chunks.append(
                            flash_slab[:, -new_frames_count:].detach()
                        )
                    else:
                        rollout2_chunks.append(
                            full_chunk[:, -new_frames_count:].detach()
                        )
        finally:
            # Leave pipeline caches clean for the caller's rollout-1
            # prefill. Restore the prior streaming_state (typically None
            # at this stage of setup_sequence).
            pipe.reset_cache_state()
            self.streaming_state = saved_streaming_state
            self.streaming_min_new_frame = saved_min_new
            self.streaming_force_new_frame_chunks = saved_force
            # Defrag — prebuild's transient activations leave the heap
            # fragmented, and the upcoming GAN engage at step 80 spikes
            # memory by ~35GB. Without this, v24 OOMs where v25 (no
            # prebuild) fits.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        rollout2_x0 = torch.cat(rollout2_chunks, dim=1)
        # Park on CPU; critic step ships a per-iter slice back to GPU
        # via the abs-position lookup. Saves ~5 MB/sample * batch_size
        # of permanent GPU residency for the duration of the ride.
        rollout2_x0 = rollout2_x0.detach().cpu()
        return rollout2_x0, int(cf_r2)

    def generate_fn_frontier_pair(
        self,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """FN frontier pair (phase-2 rolling): two short no_grad rollouts
        seeded from the student's NEWEST chunks, producing a chunk pair
        at the ride's frontier position F (= the next ungenerated chunk):

          r1' : seeds = last 6 student chunks -> 1 AR step lands [F, F+npb)
          r2' : seeds = last 5 of those (one chunk earlier) -> 2 AR steps
                land [F, F+npb) = +1 AR step of drift vs r1'

        Mirrors the setup-time rollout1/rollout2 geometry (6 vs 5 GT seed
        chunks, delta = cf/npb - fake_alt_rollout2_num_seed_chunks) but on
        STUDENT seed material at the depth the generator actually
        reached. Both mini-rides are rebased to abs 0 (rides start at
        arbitrary offsets in training, so this is in-distribution); the
        action stream is the TRUE ride actions for those positions.

        CLOBBERS the pipeline KV caches (via the prebuild) — call ONLY
        when the open sequence is about to be reset.

        Returns (r1_chunk, r2_chunk) on CPU, both [B, npb, C, H, W]
        detached, or None when the geometry doesn't fit (caller must
        keep DDP participation balanced with a zero-anchor).
        """
        s = self.streaming_state
        if s is None:
            return None
        npb = int(self.num_frame_per_block)
        prev = s.get("previous_chunk")
        if prev is None or int(prev.shape[1]) < 6 * npb:
            return None
        cf = int(s["cf"])
        cur = int(s["current_length"])
        if cur < 6 * npb:
            return None
        F = cf + cur                                # frontier abs frame
        lat = s["ride_latents_window"]
        act = s["ride_actions_window"]
        if F - 6 * npb < 0 or F + npb > int(act.shape[1]) \
                or F + npb > int(lat.shape[1]):
            return None
        pe = s["prompt_embeds"]
        n1 = 6
        delta = max(1, (cf // npb) - int(self.fake_alt_rollout2_num_seed_chunks))
        n2 = n1 - delta
        if n2 < 1:
            return None
        seed1 = prev[:, -n1 * npb:].detach()        # chunks [F-6 .. F)
        seed2 = prev[:, -n1 * npb:-delta * npb].detach()  # [F-6 .. F-delta)
        # Mini windows rebased to 0 at world frame F - n1*npb. Latent
        # windows only need shape (seed check + state plumbing); action
        # window must be the TRUE actions for these positions.
        mini_act = act[:, F - n1 * npb : F + npb]
        mini_lat1 = torch.cat(
            [seed1, lat[:, F : F + npb].to(seed1.dtype)], dim=1,
        )
        mini_lat2 = torch.cat(
            [seed2, lat[:, F - delta * npb : F + npb].to(seed2.dtype)],
            dim=1,
        )
        saved_nseed = self.fake_alt_rollout2_num_seed_chunks
        try:
            # r1': 6 seed chunks, anchor only (max_length=0) -> 1 chunk
            # at rebased [n1*npb, n1*npb + npb) = world [F, F+npb).
            self.fake_alt_rollout2_num_seed_chunks = n1
            r1_x0, _ = self._prebuild_rollout2_for_v24(
                seed_latents=seed1,
                ride_latents_window=mini_lat1,
                ride_actions_window=mini_act,
                prompt_embeds=pe,
                max_length=0,
            )
            # r2': n2 seed chunks, anchor (regen of an existing chunk) +
            # delta streamed chunks -> the LAST lands at world [F, F+npb)
            # with +delta AR steps of drift vs r1'.
            self.fake_alt_rollout2_num_seed_chunks = n2
            r2_x0, _ = self._prebuild_rollout2_for_v24(
                seed_latents=seed2,
                ride_latents_window=mini_lat2,
                ride_actions_window=mini_act,
                prompt_embeds=pe,
                max_length=delta * npb,
            )
        finally:
            self.fake_alt_rollout2_num_seed_chunks = saved_nseed
        if int(r1_x0.shape[1]) < npb or int(r2_x0.shape[1]) < npb:
            return None
        r1c = r1_x0[:, :npb].detach()
        r2c = r2_x0[:, -npb:].detach()
        # Probe: prove the FN pair ADVANCES with the generator. World
        # positions are ride-window abs frames; F grows with ride depth
        # (deeper rides -> deeper frontier pairs).
        try:
            import torch.distributed as _dist
            _is_r0 = (not _dist.is_initialized()) or _dist.get_rank() == 0
        except Exception:
            _is_r0 = True
        if _is_r0:
            import sys as _sys
            print(
                f"[FN-PAIR] frontier F={F} (ride chunk {F // npb}): "
                f"r1' seeds world [{F - n1 * npb},{F}) -> pair chunk "
                f"[{F},{F + npb}) @1 AR step; r2' seeds "
                f"[{F - n1 * npb},{F - delta * npb}) -> same chunk "
                f"@{1 + delta} AR steps (ride_offset_s="
                f"{int(s.get('ride_offset_s', 0))})",
                file=_sys.stderr, flush=True,
            )
        return r1c, r2c

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

    def ema_update_real_score_lora(self, compute_rel_l2: bool = True) -> None:
        """[DEPRECATED in v24] Update the EMA snapshot of
        self.real_score's LoRA adapter from the live LoRA params. No-op
        when ``real_score_ema_weight == 0`` (default).

        DEPRECATION: under ``fake_alt_target_mode == "rollout2_student"``
        (v24) the EMA-real_score is no longer consumed by the alt-head
        training path. ``_real_score_ema_swap`` becomes a no-op
        consumer in that mode. Slated for removal in the next big
        refactor once v23/v25 are fully retired.

        ``compute_rel_l2`` controls whether the diagnostic
        ‖live - ema‖₂ / ‖live‖₂ scalar is computed this call. With
        ``teacher_cadence='fake'`` the trainer fires this method 5×
        per outer step (once per LoRA-step inside the inner loop);
        only the LAST inner-loop call needs the diagnostic since
        wandb only logs once per outer step. Skipping the rel_l2
        computation on the first N-1 inner calls saves ~4× the
        per-param transient allocations and a GPU→CPU sync. Default
        True preserves backward-compat for callers that don't care
        about cadence.

        Iterates LoRA-tagged named_parameters (those whose name
        contains ``lora_`` — matches the same selector used at LoRA
        wrap time). On first invocation, snapshots the current live
        state into ``_real_score_ema_lora_state`` and returns (so the
        EMA starts at the live values, not at 0). Subsequent
        invocations EMA-pull:

            ema = w * ema + (1 - w) * live

        Memory: tensors are allocated on the same device as the live
        params, and cloned with .detach() so they hold no autograd
        graph.

        DDP-safe: live params are already grad-synced by the time the
        trainer calls this (post real_teacher_optimizer.step()), so
        each rank sees the same live values → each rank's EMA update
        produces the same result. No additional collective needed.
        """
        if self.real_score_ema_weight <= 0.0:
            return
        w = self.real_score_ema_weight
        # Resolve the inner module that holds the LoRA-tagged params.
        # In the codebase real_score is a wrapper whose .model attr
        # holds the WAN transformer; the LoRA adapter lives inside
        # that.
        inner = getattr(self.real_score, "model", None)
        if inner is None:
            return
        with torch.no_grad():
            if self._real_score_ema_lora_state is None:
                # First call: snapshot the live LoRA state into the
                # EMA buffer. NO EMA update yet — the very next DMD
                # pass with EMA active should see EMA == live, which
                # is bit-identical to running without EMA.
                self._real_score_ema_lora_state = {
                    name: p.detach().clone()
                    for name, p in inner.named_parameters()
                    if "lora_" in name
                }
                if (
                    not self._real_score_ema_audit_done
                    and _is_main()
                ):
                    n = len(self._real_score_ema_lora_state)
                    total_numel = sum(
                        t.numel() for t in self._real_score_ema_lora_state.values()
                    )
                    logging.info(
                        "[ActionForcingDMD] real_score EMA initialised: "
                        "%d LoRA-tagged params, %.2fM total elements "
                        "(weight=%.4f).",
                        n, total_numel / 1e6, w,
                    )
                    self._real_score_ema_audit_done = True
                return
            # Steady-state. Optionally accumulate ‖live - ema‖² and
            # ‖live‖² as we walk the params for the EMA pull. The
            # accumulators are persistent fp32 scalars on the same
            # device as the params; in-place .add_() avoids allocating
            # a new scalar tensor per param-iter. Per-param sums use
            # .sum(dtype=torch.float32) to upcast at reduction time
            # without materialising a full-tensor fp32 copy (which
            # was costing ~8 MB transient per param × 600 params on
            # the previous implementation).
            diff_sq_total: Optional[torch.Tensor] = None
            live_sq_total: Optional[torch.Tensor] = None
            if compute_rel_l2:
                # Lazy-init persistent accumulators on the first
                # rel_l2 call (any subsequent rel_l2 call zeroes
                # them in place — no new allocation).
                if getattr(self, "_real_score_ema_diff_sq_acc", None) is None:
                    # Use the first LoRA param's device — guaranteed
                    # to exist because we already enter the loop.
                    _device_probe = next(
                        (p.device for n, p in inner.named_parameters()
                         if "lora_" in n),
                        None,
                    )
                    if _device_probe is not None:
                        self._real_score_ema_diff_sq_acc = torch.zeros(
                            (), dtype=torch.float32, device=_device_probe,
                        )
                        self._real_score_ema_live_sq_acc = torch.zeros(
                            (), dtype=torch.float32, device=_device_probe,
                        )
                diff_sq_total = self._real_score_ema_diff_sq_acc
                live_sq_total = self._real_score_ema_live_sq_acc
                if diff_sq_total is not None:
                    diff_sq_total.zero_()
                    live_sq_total.zero_()
            for name, p in inner.named_parameters():
                ema_t = self._real_score_ema_lora_state.get(name)
                if ema_t is None:
                    continue  # skip any param that wasn't in the
                              # snapshot (e.g. mid-run architectural
                              # change). Should not normally happen.
                if ema_t.shape != p.shape:
                    # Shape drift — log once and skip. Avoids silent
                    # corruption.
                    if _is_main():
                        logging.warning(
                            "[ActionForcingDMD] real_score EMA shape "
                            "mismatch for %s (ema=%s vs live=%s); "
                            "skipping this param.",
                            name, tuple(ema_t.shape), tuple(p.shape),
                        )
                    continue
                if compute_rel_l2 and diff_sq_total is not None:
                    # Compute diagnostic BEFORE the EMA pull. The
                    # metric reflects "the gap that the just-completed
                    # live step opened up" — after the EMA pull the
                    # gap would look smaller by exactly the (1-w)
                    # factor, which carries no new information.
                    # ``.sum(dtype=torch.float32)`` upcasts at reduction,
                    # avoiding the full-tensor fp32 transient.
                    diff_sq_total.add_(
                        (p.detach() - ema_t).pow(2).sum(dtype=torch.float32)
                    )
                    live_sq_total.add_(
                        p.detach().pow(2).sum(dtype=torch.float32)
                    )
                # EMA pull.
                ema_t.mul_(w).add_(p.detach(), alpha=1.0 - w)
            if compute_rel_l2 and diff_sq_total is not None and live_sq_total is not None:
                rel = (
                    diff_sq_total.clamp_min(0.0)
                    / live_sq_total.clamp_min(1e-12)
                ).sqrt()
                self._real_score_ema_rel_l2 = float(rel.item())

    @contextmanager
    def _real_score_ema_swap(self):
        """Context manager: temporarily swap self.real_score's LoRA
        params to the EMA values, restore on exit. Used to wrap the
        DMD scoring forward so DMD reads a slow-moving target while
        the aux teacher loss continues to forward through the live
        LoRA outside this context.

        Implementation: ``torch.utils.swap_tensors`` exchanges the
        underlying storage of two Tensors with zero allocation. After
        the first swap, ``p.data`` holds the EMA values and the
        ``ema_t`` entry in the dict temporarily holds the live
        values. The DMD forward (no_grad) reads ``p.data`` as
        normal. On exit the swap-back restores both — ``p.data``
        back to live values, ``ema_t`` back to EMA values — with no
        additional memory churn. Avoids the ~87 MB live_backup
        allocation the clone-and-copy approach required.

        Safe to swap ``p.data`` because:
          * Both tensors have identical shape/dtype/device (the EMA
            buffer was constructed via ``.clone()`` from the live
            params).
          * The swap happens inside no_grad — DDP isn't running
            collectives on these params during the swap window.
          * We swap the underlying tensors, not the Parameter
            wrappers, so DDP's parameter refs are unaffected.

        No-op when EMA is disabled (weight == 0) or not yet
        initialised (pre-first LoRA optim step). In those cases the
        live params are used unchanged.

        Restore is unconditional via try/finally — if the wrapped
        forward raises, the live LoRA state is still restored before
        the exception propagates.
        """
        if (
            self.real_score_ema_weight <= 0.0
            or self._real_score_ema_lora_state is None
        ):
            yield
            return
        inner = getattr(self.real_score, "model", None)
        if inner is None:
            yield
            return
        # Build the swap list once. Only includes LoRA-tagged params
        # whose EMA entry is shape-compatible (defensive against
        # mid-run param-shape drift, which shouldn't happen but is
        # cheap to guard against).
        swap_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for name, p in inner.named_parameters():
            ema_t = self._real_score_ema_lora_state.get(name)
            if ema_t is None or ema_t.shape != p.shape:
                continue
            swap_pairs.append((p.data, ema_t))
        with torch.no_grad():
            for p_data, ema_t in swap_pairs:
                _swap_tensors(p_data, ema_t)
        try:
            yield
        finally:
            with torch.no_grad():
                for p_data, ema_t in swap_pairs:
                    _swap_tensors(p_data, ema_t)

    @contextmanager
    def _maybe_asymmetric_tf_rope_offset(self, offset: Optional[int]):
        """Context manager: temporarily override ``tf_rope_offset_frames``
        on every scorer's base DiT for an asymmetric DMD scoring forward.

        v14's symmetric training contract sets ``tf_rope_offset_frames``
        = ``num_frame_per_block`` at init time (clean at RoPE [0, F),
        noisy at RoPE [npb, npb+F)). For the dmd_one_step asymmetric
        path we want the single noisy chunk to sit just after the clean
        window ends — at RoPE [F_clean, F_clean + F_noisy) — so v14
        treats it as "the next chunk after preceding 7 GT chunks".
        Passing ``offset = F_clean`` here mutates the attribute on all
        scorer DiTs (real, fake, optional frozen-v14) for the duration
        of the wrapped block and restores the v14 value on exit.

        When ``offset is None`` this is a no-op (legacy symmetric path).
        Restore is unconditional via try/finally so a raise inside the
        wrapped forward still leaves the v14 contract attribute intact.
        """
        if offset is None:
            yield
            return
        scorer_wrappers = [self.real_score.model, self.fake_score.model]
        if self.real_score_frozen is not None:
            scorer_wrappers.append(self.real_score_frozen.model)
        saved: List[Tuple[Any, Optional[int]]] = []
        for wrapper in scorer_wrappers:
            m = (
                wrapper.get_base_model()
                if hasattr(wrapper, "get_base_model") else wrapper
            )
            saved.append((m, getattr(m, "tf_rope_offset_frames", None)))
            m.tf_rope_offset_frames = int(offset)
            # NB: the block_mask is invalidated automatically inside
            # ``_forward_train`` via the (num_clean, num_noisy,
            # frame_seqlen) signature check — no manual clear needed
            # here, and clearing would force a wasteful rebuild every
            # call.
        try:
            yield
        finally:
            for m, prev in saved:
                m.tf_rope_offset_frames = prev

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
        # Flash-DMD t=flash_dmd_gan_t pred on ``pipe._flash_dmd_gan_output``;
        # we re-stitch it into a chunk_size-frame slab below for the GAN.
        # v27I onward: flash_dmd is compulsory whenever the master
        # switch is on, REGARDLESS of requires_grad. Ensures rollout 2
        # (no_grad prebuild) runs the same 5-forward trajectory as
        # rollout 1 (gen step), so the FN training sees symmetric
        # rollout1/rollout2 chunks.
        flash_dmd_enabled = bool(self.flash_dmd_enabled)
        # Warm-start init: when on, the iter's first block warm-starts
        # from the prior call's clean pred (= ``previous_clean_chunk``)
        # instead of pure noise. Subsequent blocks within the same
        # call (iter 1's chunk_size>npb path, multiple blocks per call)
        # warm-start from the preceding block's clean pred — handled
        # internally by ``generate_chunk_with_cache``.
        new_chunk, denoised_t_from, denoised_t_to = pipe.generate_chunk_with_cache(
            noise=noise_chunk,
            current_start_frame=abs_frame_start,
            requires_grad=requires_grad,
            prefer_cache_pred_in_output=False,
            gt_latents=gt_chunk,
            sync_exit_flags=sync_exit_flags,
            force_exit_step=force_exit_step,
            flash_dmd_enabled=flash_dmd_enabled,
            flash_dmd_gan_t=int(self.flash_dmd_gan_t),
            **cond_dict,
        )
        # Pull the Flash-DMD t=gan_t output (None when flash_dmd_enabled=False).
        # Shape ``[B, new_frames, C, H, W]`` — the SAME npb-aligned slab
        # the pipeline rolled this iter. After Step 3.2.b/3.3.5
        # unification, this is the single grad-on t=60 forward's output
        # (consumed by both GAN and — via detach — FN training + eval
        # logging).
        new_last_rung_chunk = (
            pipe._flash_dmd_gan_output if flash_dmd_enabled else None
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
                # WAN VAE Conv3d kernels are fp32; the streaming
                # latents are bf16 (`dtype`). Cast to fp32 for the
                # VAE forwards and back to bf16 after — preserves
                # downstream contract while satisfying the conv's
                # weight/input dtype invariant.
                ctx_latents = torch.cat(
                    [prev_chunk_for_clean, full_chunk[:, 0:1]], dim=1,
                ).to(torch.float32)
                # Cached-decode path: keeps the WAN VAE's temporal
                # feat_map populated across calls so the unconditioned
                # init-frame brightness anomaly isn't re-injected
                # every iter (would otherwise leak into the boundary
                # latent via the re-encode and compound across iters).
                pixels = self.vae.decode_to_pixel(
                    ctx_latents, use_cache=True,
                )
                last_frame_btchw = pixels[:, -1:, ...].to(torch.float32)
                last_frame_bcthw = _rearrange(
                    last_frame_btchw, "b t c h w -> b c t h w",
                )
                image_latent = self.vae.encode_to_latent(
                    last_frame_bcthw,
                ).to(dtype)
                full_chunk = torch.cat(
                    [image_latent, full_chunk[:, 1:]], dim=1,
                )

        # Save full_chunk as previous_chunk for the NEXT iter.
        #
        # Default (dmd_lookback_chunks=0): full detach — the next iter's
        # overlap region starts from a leaf tensor, no graph crosses
        # chunks.
        #
        # v29 (dmd_lookback_chunks=1): partial detach — the OVERLAP
        # region (early frames, sourced from PRIOR previous_chunk and
        # therefore from chunk_(k-2)'s graph) is detached to prevent
        # cascading. The LAST new_frames region stays un-detached so
        # chunk_k's exit-rung forward graph is reachable from chunk_
        # (k+1)'s overlap on the next iter. Combined with
        # generator_loss.backward(retain_graph=True), chunk_(k+1)'s
        # DMD backward walks back through chunk_k's exit-rung forward.
        if int(self.dmd_lookback_chunks) > 0 and overlap > 0:
            overlap_part = full_chunk[:, :overlap].detach()
            new_part = full_chunk[:, overlap:]
            s["previous_chunk"] = torch.cat(
                [overlap_part, new_part], dim=1,
            )
        else:
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
            # Flash-DMD t=flash_dmd_gan_t view. None when
            # ``flash_dmd_enabled`` is off OR this is a no-grad iter.
            # The generator-loss path stashes this onto
            # ``info["flash_dmd_gan_x0"]`` so the trainer's
            # ``_compute_r3gan_losses`` and the gen-side aux losses
            # (action_critic) consume it
            # as the G-side fake.
            "flash_dmd_gan_chunk": full_last_rung_chunk,
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

    def _build_asymmetric_scoring_inputs(
        self, chunk: torch.Tensor, info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """v14-faithful asymmetric DMD scoring inputs for the FIRST
        rolled chunk of this iter.

        Layout (npb=3, N=num_training_frames=21):
          * noisy_x = the student's FIRST npb-frame chunk of the rolled
            window (``chunk[:, :npb]``) — lowest-drift, conditioned on
            the GT seed. Graph state from ``chunk`` is preserved (the
            generator path needs grad; the critic path passes a
            detached ``chunk``).
          * clean_x = the N GT frames at ride positions
            ``[chunk_lo - N, chunk_lo)`` that immediately precede the
            noisy chunk (chunk_lo = cf + noisy_start_sdn = the rolled
            window's absolute start = the first chunk's position).
          * rope_offset = N → the single noisy chunk lands at v14's
            LAST noisy-slot RoPE position ``[N, N+npb)``; combined with
            the asymmetric mask (noisy attends to all N clean frames)
            this reproduces v14's training contract for its final chunk.

        Action conditioning for BOTH windows is rebuilt fresh from
        ``ride_actions_window`` (which spans the whole ride), so the
        clean window's earlier positions are available even though the
        symmetric ``clean_actions_window`` does not reach back this far.

        Raises (NO silent fallback) when the GT history or future the
        slices need is outside the ride window — the streaming setup
        guarantees enough on the first roll, so a violation is a real
        bug, not an expected edge.
        """
        s = self.streaming_state
        npb = int(self.num_frame_per_block)
        N = int(self.num_training_frames)
        cf = int(s["cf"])
        # clean = N GT frames at [0, N), single noisy chunk at RoPE
        # [N, N+npb) (tf_rope_offset = N) = v14's LAST noisy-slot contract.
        n_ctx = N
        rope_offset = N
        noisy_start_sdn = int(
            s["current_length"] - info["new_frames"] - info["overlap"]
        )
        chunk_lo = cf + noisy_start_sdn          # first chunk's abs ride position
        clean_lo = chunk_lo - n_ctx               # n_ctx GT frames before it
        ride_lat = s["ride_latents_window"]
        ride_act = s["ride_actions_window"]
        if clean_lo < 0:
            raise RuntimeError(
                f"asymmetric DMD: clean_lo={clean_lo} < 0 (chunk_lo={chunk_lo}, "
                f"n_ctx={n_ctx}). Need a larger dmd_context_clean_frames."
            )
        if chunk_lo + npb > int(ride_lat.shape[1]):
            raise RuntimeError(
                f"asymmetric DMD: ride window too short — need "
                f">= {chunk_lo + npb} latent frames, have {int(ride_lat.shape[1])}."
            )
        if ride_act.shape[1] < chunk_lo + npb:
            raise RuntimeError(
                f"asymmetric DMD: ride_actions_window too short — need "
                f">= {chunk_lo + npb} frames, have {int(ride_act.shape[1])}."
            )
        clean_x = ride_lat[:, clean_lo:chunk_lo].to(
            dtype=chunk.dtype, device=chunk.device,
        ).detach()
        gt_target = ride_lat[:, chunk_lo:chunk_lo + npb].to(
            dtype=chunk.dtype, device=chunk.device,
        )
        noisy_x = chunk[:, :npb]
        prompt = s["prompt_embeds"]
        noisy_cond, noisy_uncond = self.build_action_conditional(
            prompt_embeds=prompt,
            gt_actions=ride_act[:, chunk_lo:chunk_lo + npb],
        )
        clean_cond, clean_uncond = self.build_action_conditional(
            prompt_embeds=prompt,
            gt_actions=ride_act[:, clean_lo:chunk_lo],
        )
        cond = dict(noisy_cond)
        cond["_action_modulation_clean"] = clean_cond.get("_action_modulation")
        cond["_action_tokens_clean"] = clean_cond.get("_action_tokens")
        uncond = dict(noisy_uncond)
        uncond["_action_modulation_clean"] = clean_uncond.get("_action_modulation")
        uncond["_action_tokens_clean"] = clean_uncond.get("_action_tokens")
        aug_t = torch.zeros(
            (clean_x.shape[0], n_ctx), device=chunk.device, dtype=torch.long,
        )
        gradient_mask = torch.ones_like(noisy_x, dtype=torch.bool)
        return {
            "noisy_x": noisy_x,
            "clean_x": clean_x,
            "aug_t": aug_t,
            "cond": cond,
            "uncond": uncond,
            "gt_target": gt_target,
            "gradient_mask": gradient_mask,
            "rope_offset": rope_offset,
        }

    def _build_42f_scoring_inputs(
        self, chunk: torch.Tensor, info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """v14-faithful gt-context DMD inputs (the "42f" fix).

        Keeps v14's EXACT training shape — a 21-clean + 21-noisy joint at
        tf_rope_offset=npb (=3) — so the teacher denoises IN-DISTRIBUTION,
        but makes the noisy half mostly GT so only the supervised student
        chunk carries the DMD signal.

        Two layouts, selected by ``dmd_42f_2chunk``:

        * 1-chunk (default): noisy_x = [18 GT | 1 student], the student's
          first chunk at the NEWEST slot RoPE [21,24). That slot has NO
          clean counterpart → v14's denoising there is structurally OOD
          (~2x MAE; the position the canonical DMD mask throws away). The
          DMD gradient taken there is noisy → can spike / collapse.

        * 2-chunk (``dmd_42f_2chunk=True``): roll TWO student chunks; put
          them at the LAST two noisy slots. The 2nd (newest) lands at
          RoPE [21,24) (no clean counterpart) and is MASKED OUT. The 1st
          lands at RoPE [18,21), which DOES have a clean counterpart →
          v14's reliable mid-window region → clean gradient. We supervise
          ONLY that 1st chunk. noisy_x = [15 GT | student_A | student_B];
          student_B is detached (masked context only). This avoids the
          unreliable last slot.

        clean_x = 21 GT frames shifted back npb from the noisy half (v14's
        teacher-forcing alignment). rope_offset is None → scorers keep
        their init tf_rope_offset_frames = npb (=3).
        """
        s = self.streaming_state
        npb = int(self.num_frame_per_block)
        N = int(self.num_training_frames)          # 21
        # ``ns`` = number of student chunks rolled into the noisy half.
        # ``dmd_42f_num_chunks`` (>=1) takes precedence; the legacy
        # ``dmd_42f_2chunk`` boolean maps to ns=2.
        ns = int(getattr(self, "dmd_42f_num_chunks", 0))
        if ns <= 0:
            ns = 2 if bool(getattr(self, "dmd_42f_2chunk", False)) else 1
        ns = max(1, ns)
        # Supervise ALL student chunks EXCEPT the newest (which lands at
        # the structurally-OOD slot RoPE [21,24) and is masked). For ns=1
        # there is only one chunk, so it IS supervised (the original 42f).
        num_sup = 1 if ns == 1 else (ns - 1)
        sup_frames = num_sup * npb
        student_frames = ns * npb                  # rolled student frames
        seed_last = bool(getattr(self, "dmd_42f_seed_last", False))
        # Number of chunks placed AFTER the supervised student block.
        # Default: the (ns - num_sup) newest chunks (masked context).
        # ``dmd_42f_gt_after_chunks`` (>0) overrides the COUNT, moving the
        # supervised chunk further from the structurally-OOD newest slot
        # RoPE [21,24): e.g. =2 places the single student chunk
        # 3rd-from-last (RoPE [15,18)) with TWO chunks after it.
        #
        # ``seed_last`` decides the CONTENT of the after-block (handled in
        # the parts assembly below): True -> detached GT scaffold; False ->
        # the student's OWN rolled chunks (detached, masked context). So
        # ``gt_after_chunks=2`` + ``seed_last=false`` = [4 GT | S | s s],
        # leaving the student rollout in instead of GT.
        gt_after_override = int(getattr(self, "dmd_42f_gt_after_chunks", 0))
        if gt_after_override > 0:
            gt_after_frames = gt_after_override * npb
        else:
            gt_after_frames = student_frames - sup_frames
        n_ctx = N - sup_frames - gt_after_frames   # GT context frames
        if n_ctx < 0:
            raise RuntimeError(
                f"42f DMD: n_ctx={n_ctx} < 0 (sup_frames={sup_frames}, "
                f"gt_after_frames={gt_after_frames}, N={N}). Reduce "
                f"dmd_42f_gt_after_chunks or num_chunks."
            )
        cf = int(s["cf"])
        noisy_start_sdn = int(
            s["current_length"] - info["new_frames"] - info["overlap"]
        )
        chunk_lo = cf + noisy_start_sdn            # first student chunk's abs pos
        ride_lat = s["ride_latents_window"]
        ride_act = s["ride_actions_window"]
        # ---- Phase-2 rolling (dmd_42f_rolling_sup_new): on k>=2 rolls
        # (overlap > 0) the WHOLE 42f window rolls forward with the
        # student. Layout becomes
        #   [n_ctx STUDENT ctx (overlap, detached) | new frames (graph-on,
        #    supervised) | npb GT future scaffold]
        # i.e. the noisy half is the student's own rolled window — the
        # GT-context former half is SPECIAL to iter 1 only (per design).
        # The GT future chunk occupies the structurally-OOD newest slot
        # RoPE [N, N+npb) (gradient-masked, content only): a chunk newer
        # than the newest rolled chunk does not exist yet, and that slot
        # must be filled for v14's exact 21f joint geometry. clean_x
        # stays positional GT and rolls forward with the window.
        # Without this branch the supervised block = chunk[:, :sup_span]
        # = the DETACHED overlap frames -> silent zero-gradient DMD on
        # every roll after the first.
        _rolling = (
            bool(getattr(self, "dmd_42f_rolling_sup_new", False))
            and int(info.get("overlap", 0)) > 0
        )
        if _rolling:
            _ovl = int(info["overlap"])
            _nf = int(info["new_frames"])
            if _nf % npb != 0 or _nf <= 0:
                raise RuntimeError(
                    f"42f rolling: new_frames={_nf} must be a positive "
                    f"multiple of npb={npb}."
                )
            sup_frames = _nf                       # supervise ALL new chunks
            gt_after_frames = npb                  # GT scaffold at the OOD slot
            n_ctx = N - sup_frames - gt_after_frames
            if n_ctx < 0:
                raise RuntimeError(
                    f"42f rolling: n_ctx={n_ctx} < 0 — new_frames={_nf} too "
                    f"large for N={N} (reduce num_chunks_roll_forward)."
                )
            if _ovl < n_ctx:
                raise RuntimeError(
                    f"42f rolling: overlap={_ovl} < n_ctx={n_ctx} — the "
                    f"student window cannot supply the context half."
                )
            # Rebind ``chunk`` to [n_ctx student ctx | new frames]: drop
            # the overlap frames older than the context window. The
            # context slice comes from the previous iters' chunks
            # (already detached); the new frames stay graph-on.
            # ``chunk_lo`` keeps its downstream meaning = abs pos of the
            # first SUPERVISED frame (ctx occupies [chunk_lo - n_ctx,
            # chunk_lo), exactly the overlap frames preceding the new
            # ones), so noisy_lo / noisy_hi / clean_lo math is unchanged.
            chunk = torch.cat(
                [
                    chunk[:, _ovl - n_ctx:_ovl].detach(),
                    chunk[:, _ovl:],
                ],
                dim=1,
            )
            chunk_lo = chunk_lo + _ovl             # abs pos of first NEW frame
        # noisy half abs span [chunk_lo - n_ctx, chunk_lo + sup_frames +
        # gt_after_frames) (21 frames); clean half shifted back npb.
        noisy_lo = chunk_lo - n_ctx
        noisy_hi = chunk_lo + sup_frames + gt_after_frames
        clean_lo = noisy_lo - npb                  # = chunk_lo - n_ctx - npb
        if clean_lo < 0:
            raise RuntimeError(
                f"42f DMD: clean_lo={clean_lo} < 0 (chunk_lo={chunk_lo}, "
                f"N={N}, ns={ns}). Need a larger dmd_context_clean_frames."
            )
        need = max(noisy_hi, clean_lo + N)
        if need > int(ride_lat.shape[1]) or need > int(ride_act.shape[1]):
            raise RuntimeError(
                f"42f DMD: ride window too short — need >= {need} frames, "
                f"have lat={int(ride_lat.shape[1])} act={int(ride_act.shape[1])}."
            )
        # The student ``chunk`` must supply the supervised frames, plus the
        # detached tail frames when NOT seed_last (seed_last sources its
        # after-block from GT ``ride_lat`` instead). With rand_sup_slot the
        # supervised chunk can be drawn from any after-offset (up to the
        # last counterpart slot), so the rollout must cover the whole
        # after-region even when seed_last. Guard explicitly so a mis-set
        # num_chunks / short rollout fails loud, not via an opaque
        # out-of-bounds slice below.
        _rand_need = bool(getattr(self, "dmd_42f_rand_sup_slot", False))
        if _rolling:
            # Rolling: the (rebound) chunk holds [n_ctx student ctx |
            # new frames]; the after-slot is sourced from ride GT.
            chunk_need = n_ctx + sup_frames
        else:
            chunk_need = (
                sup_frames
                if (seed_last and not _rand_need)
                else (sup_frames + gt_after_frames)
            )
        if int(chunk.shape[1]) < chunk_need:
            raise RuntimeError(
                f"42f DMD: student chunk has {int(chunk.shape[1])} frames "
                f"but {chunk_need} required (sup_frames={sup_frames}, "
                f"seed_last={seed_last}, gt_after_frames={gt_after_frames})."
            )
        # noisy_x: n_ctx context (detached) + the ``sup_frames`` student
        # frames (graph-on, supervised) + the newest chunk(s) (detached,
        # masked context). The supervised chunks sit at the reliable
        # mid-window slots; the newest is at [21,24) (masked).
        # Context source: ride GT normally; the STUDENT's own overlap
        # frames on rolling k>=2 (iter 1's GT former half is special).
        if _rolling:
            gt_ctx = chunk[:, :n_ctx].detach()
        else:
            gt_ctx = ride_lat[:, noisy_lo:chunk_lo].to(
                dtype=chunk.dtype, device=chunk.device,
            ).detach()
        # Random supervised slot (rand_sup_slot): pick WHICH after-slot
        # carries the DMD gradient. Valid slots are those whose noisy frames
        # still have a clean counterpart — every slot except the last
        # ([N-npb, N)). The chosen chunk is graph-on student content; the
        # rest of the after-region is detached context (GT when seed_last —
        # a scaffold on both sides — else the student rollout). Step-seeded
        # -> all ranks agree and r sweeps over training. World-consistent:
        # after-slot at frame-offset o aligns with ride world [chunk_lo+o,…],
        # so the supervised slot holds chunk[:, o:o+npb] (the o//npb-th roll).
        sup_offset = 0
        _rand_slot = bool(getattr(self, "dmd_42f_rand_sup_slot", False))
        if _rand_slot:
            n_after = (sup_frames + gt_after_frames) // npb
            max_r = min(n_after - 1, ((N - npb) - n_ctx) // npb - 1)
            if max_r >= 1:
                _step = int(info.get("current_step", 0))
                _g = torch.Generator(device="cpu").manual_seed(int(_step))
                sup_offset = int(
                    torch.randint(0, max_r + 1, (1,), generator=_g).item()
                ) * npb
        # allsup: supervise ALL student after-chunks that have a clean
        # counterpart — noisy slots [n_ctx, N-npb) (every after-chunk
        # except the last, counterpart-less one). Those frames are graph-on
        # and the gradient_mask covers all of them (vs one slot). Only for
        # seed_last=false (student after-region). sup_span spans the
        # supervised chunks; the single-slot path keeps sup_span=sup_frames.
        _allsup = (not seed_last) and bool(
            getattr(self, "dmd_42f_allsup", False)
        )
        if _rolling:
            # Rolling overrides allsup/rand_slot geometry: every new chunk
            # is supervised; the after-slot is a GT scaffold.
            sup_span = sup_frames
            sup_offset = 0
            _allsup = False
            _rand_slot = False
        elif _allsup:
            sup_span = (N - npb) - n_ctx
            sup_offset = 0
        else:
            sup_span = sup_frames
        if _rolling:
            gt_future = ride_lat[
                :,
                chunk_lo + sup_frames : chunk_lo + sup_frames + gt_after_frames,
            ].to(dtype=chunk.dtype, device=chunk.device).detach()
            _sup_block = chunk[:, n_ctx:n_ctx + sup_frames]
            parts = [gt_ctx, _sup_block, gt_future]
            if getattr(self, "_rolling_42f_dbg", 0) < 2:
                self._rolling_42f_dbg = getattr(
                    self, "_rolling_42f_dbg", 0) + 1
                import sys as _sys
                print(
                    f"[42F-ROLLING] ACTIVE: full student window — "
                    f"ctx=STUDENT overlap [{chunk_lo - n_ctx},{chunk_lo}) "
                    f"sup=[{chunk_lo},{chunk_lo + sup_frames}) "
                    f"(graph-on={bool(_sup_block.requires_grad)}) "
                    f"gt_future=[{chunk_lo + sup_frames},"
                    f"{chunk_lo + sup_frames + gt_after_frames}) "
                    f"clean_lo={chunk_lo - n_ctx - npb} (GT, rolls fwd)",
                    file=_sys.stderr, flush=True,
                )
        elif _allsup:
            stu_tot = sup_frames + gt_after_frames
            parts = [
                gt_ctx,
                chunk[:, :sup_span],                   # slots [n_ctx..N-npb): graph-on
                chunk[:, sup_span:stu_tot].detach(),   # last slot: no counterpart
            ]
        elif seed_last and _rand_slot:
            # GT scaffold for the WHOLE after-region, with the supervised
            # student chunk inserted at the random slot (sup_offset). Only
            # that slot is student (graph-on); every other after-slot is GT.
            after_parts = []
            n_after = (sup_frames + gt_after_frames) // npb
            for _o_idx in range(n_after):
                _o = _o_idx * npb
                if _o == sup_offset:
                    after_parts.append(chunk[:, _o:_o + sup_frames])
                else:
                    _w = chunk_lo + _o
                    after_parts.append(
                        ride_lat[:, _w:_w + npb].to(
                            dtype=chunk.dtype, device=chunk.device,
                        ).detach()
                    )
            parts = [gt_ctx] + after_parts
        elif (not seed_last) and sup_offset > 0:
            # seed_last=false + random: student after-region with ONE
            # graph-on chunk at sup_offset; rest detached student context.
            stu_tot = sup_frames + gt_after_frames
            parts = [
                gt_ctx,
                chunk[:, :sup_offset].detach(),
                chunk[:, sup_offset:sup_offset + sup_frames],
                chunk[:, sup_offset + sup_frames:stu_tot].detach(),
            ]
        else:
            parts = [gt_ctx, chunk[:, :sup_frames]]
            if gt_after_frames > 0:
                if seed_last:
                    # GT scaffold AFTER the supervised chunk (clean both
                    # sides) instead of the masked newest STUDENT chunk(s).
                    gt_after = ride_lat[
                        :, chunk_lo + sup_frames : chunk_lo + sup_frames + gt_after_frames,
                    ].to(dtype=chunk.dtype, device=chunk.device).detach()
                    parts.append(gt_after)
                else:
                    parts.append(
                        chunk[:, sup_frames:sup_frames + gt_after_frames].detach()
                    )
        noisy_x = torch.cat(parts, dim=1)          # [B, 21, ...]
        clean_x = ride_lat[:, clean_lo:clean_lo + N].to(
            dtype=chunk.dtype, device=chunk.device,
        ).detach()
        # Clean-counterpart fix: in v14's joint TF, clean frame at array
        # index (a+npb) is the SAME world frame as noisy frame at index a
        # (clean shifted back npb). Where the noisy half holds a STUDENT
        # chunk but its clean counterpart is GT, the teacher's full
        # bidirectional attention sees a GT "what it should have been" that
        # disagrees with the student noisy frame -> OOD (v14 was trained
        # with the two halves identical at each world position). Replace
        # each such clean counterpart with the student content (detached)
        # so clean and noisy AGREE -> kills the conflict, keeps all N clean
        # frames (more v14-faithful than dropping a key, which the flash
        # kernel can't do for a mid-sequence frame anyway).
        if bool(getattr(self, "dmd_42f_fix_clean_counterpart", False)):
            stu_lo = n_ctx
            # Student frames in noisy_x = supervised block, plus the
            # after-block when it is student-rolled (not seed_last GT).
            stu_hi = n_ctx + sup_frames + (
                0 if seed_last else gt_after_frames
            )
            dst_lo = stu_lo + npb
            dst_hi = min(stu_hi + npb, N)
            n_rep = dst_hi - dst_lo
            if n_rep > 0:
                clean_x = clean_x.clone()
                clean_x[:, dst_lo:dst_hi] = (
                    noisy_x[:, stu_lo:stu_lo + n_rep].detach()
                )
        # Clean-self: put the student's OWN rolled chunks into the clean
        # after-region (clean[n_ctx+npb : N] = supervised chunk's clean
        # counterpart + the gt_after slots), sourced from ``chunk`` directly
        # so it works whether the noisy after-region is GT (seed_last=true)
        # or student (seed_last=false). OOD ablation: makes the clean half
        # follow the rollout in its later slots.
        if bool(getattr(self, "dmd_42f_clean_self", False)):
            cs_lo = n_ctx + npb
            cs_n = N - cs_lo
            if cs_n > 0 and int(chunk.shape[1]) >= cs_n:
                clean_x = clean_x.clone()
                clean_x[:, cs_lo:N] = chunk[:, :cs_n].detach()
        gt_target = ride_lat[:, noisy_lo:noisy_hi].to(
            dtype=chunk.dtype, device=chunk.device,
        )
        prompt = s["prompt_embeds"]
        noisy_cond, noisy_uncond = self.build_action_conditional(
            prompt_embeds=prompt, gt_actions=ride_act[:, noisy_lo:noisy_hi],
        )
        clean_cond, clean_uncond = self.build_action_conditional(
            prompt_embeds=prompt, gt_actions=ride_act[:, clean_lo:clean_lo + N],
        )
        cond = dict(noisy_cond)
        cond["_action_modulation_clean"] = clean_cond.get("_action_modulation")
        cond["_action_tokens_clean"] = clean_cond.get("_action_tokens")
        uncond = dict(noisy_uncond)
        uncond["_action_modulation_clean"] = clean_uncond.get("_action_modulation")
        uncond["_action_tokens_clean"] = clean_uncond.get("_action_tokens")
        aug_t = torch.zeros(
            (clean_x.shape[0], N), device=chunk.device, dtype=torch.long,
        )
        # Gradient on the supervised student frames only (noisy positions
        # [n_ctx, n_ctx + sup_frames)): the single chunk for ns=1, the
        # first (ns-1) chunks for ns>=2 (the newest is masked).
        gradient_mask = torch.zeros_like(noisy_x, dtype=torch.bool)
        gradient_mask[
            :, n_ctx + sup_offset : n_ctx + sup_offset + sup_span
        ] = True
        import os as _os
        if _os.environ.get("ARRWM_ROPE_DEBUG"):
            try:
                _c = getattr(self, "_rope_dbg_42f", 0)
                if _c < 3:
                    self._rope_dbg_42f = _c + 1
                    ntot = N // npb
                    ncc = n_ctx // npb
                    supc = sup_span // npb   # # of supervised chunks (allsup>1)
                    # Per noisy chunk-slot label: G=GT, S=sup-student,
                    # s=ctx-student (masked).
                    _supc0 = ncc + (sup_offset // npb)   # first supervised slot
                    nlab = []
                    for _sl in range(ntot):
                        if _sl < ncc:
                            nlab.append("G")
                        elif _supc0 <= _sl < _supc0 + supc:
                            nlab.append("S")
                        else:
                            nlab.append("G" if seed_last else "s")
                    # Bidir counterpart: noisy slot k <-> clean slot k+1
                    # (clean shifted back npb). A student noisy chunk whose
                    # clean counterpart slot exists+is GT = the OOD conflict
                    # the teacher's bidir attention sees.
                    conflict = [
                        _sl + 1 for _sl, _l in enumerate(nlab)
                        if _l in ("S", "s") and (_sl + 1) < ntot
                    ]
                    _fix = bool(getattr(
                        self, "dmd_42f_fix_clean_counterpart", False))
                    clab = ["G"] * ntot
                    if _fix:
                        for _cs in conflict:
                            clab[_cs] = "S"  # replaced with student content
                    if bool(getattr(self, "dmd_42f_clean_self", False)):
                        _cs_slot_lo = (n_ctx + npb) // npb
                        for _cs in range(_cs_slot_lo, ntot):
                            clab[_cs] = "S"  # clean_self: student content
                    print(
                        f"[ROPE-DBG 42f] seed_last={seed_last} n_ctx={n_ctx} "
                        f"sup_frames={sup_frames} gt_after={gt_after_frames} "
                        f"fix_clean_counterpart={_fix}\n"
                        f"  noisy slots[0..{ntot-1}]: {nlab}  "
                        f"(S=sup-student, s=ctx-student, G=GT)\n"
                        f"  clean slots[0..{ntot-1}]: {clab}  "
                        f"(S=student-replaced)\n"
                        f"  supervised chunk: sup_offset={sup_offset} "
                        f"slot={_supc0} noisy_array=[{n_ctx + sup_offset},"
                        f"{n_ctx + sup_offset + sup_frames}) -> scoring_rope="
                        f"[{n_ctx + sup_offset + npb},"
                        f"{n_ctx + sup_offset + sup_frames + npb})\n"
                        f"  CONFLICT clean slots (GT counterpart of a student "
                        f"noisy chunk): {conflict}  "
                        f"-> {'REPLACED w/ student' if _fix else 'left GT (OOD)'}\n"
                        f"  SHAPES: chunk={tuple(chunk.shape[:2])} "
                        f"clean_x={tuple(clean_x.shape[:2])} "
                        f"noisy_x={tuple(noisy_x.shape[:2])} | "
                        f"WORLD: chunk_lo={chunk_lo} noisy_lo={noisy_lo} "
                        f"clean_lo={clean_lo} N={N} npb={npb} "
                        f"streaming_chunk_size={int(s['chunk_size'])} "
                        f"sup_chunk_world=[{chunk_lo},{chunk_lo + sup_frames})",
                        flush=True,
                    )
            except Exception:
                pass
        return {
            "noisy_x": noisy_x,
            "clean_x": clean_x,
            "aug_t": aug_t,
            "cond": cond,
            "uncond": uncond,
            "gt_target": gt_target,
            "gradient_mask": gradient_mask,
            "rope_offset": None,   # keep v14's init offset (npb=3)
        }

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

        When ``dmd_only_first_chunk_per_ride`` is True (dmd_one_step
        contract — supervise only the first rolled chunk per ride),
        non-first rolls within the ride return a connected zero loss
        and skip the scorer forward entirely. The chunk's autograd
        graph still anchors the returned scalar so the trainer's
        ``loss.backward()`` succeeds without exception.
        """
        s = self.streaming_state
        chunks_in_ride = int(info.get("chunks_in_current_ride", 1))
        only_first = bool(
            getattr(self, "dmd_only_first_chunk_per_ride", False)
        )
        if only_first and chunks_in_ride > 1:
            zero = (chunk.float() * 0.0).sum().to(chunk.dtype)
            return zero, {
                "dmd_skipped_non_first_chunk_of_ride": 1.0,
                "streaming_chunks_in_ride": float(chunks_in_ride),
            }

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
        # Pre-resolve the unified aux GT/student blend ratio ONCE per
        # iter from the schedule (or static fallback). Both halves of
        # the LoRA aux pass — clean half (built inside
        # ``_build_dmd_context_kwargs``) and noisy half (consumed in
        # ``_compute_aux_teacher_loss_streaming``) — must see the
        # SAME value, otherwise the LoRA's joint-TF input is
        # incoherent. Resolving once and threading through avoids
        # double-resolving with potentially different values (race-
        # safe even though step doesn't change within an iter; clearer
        # invariant).
        current_step = int(info.get("current_step", 0))
        # Stash for the non-streaming ``generator_loss`` fallback —
        # that path doesn't carry ``current_step`` in its signature,
        # so any step-dependent resolver it touches (e.g.
        # ``_resolved_stat_anchor_loss_weight``) reads this stash.
        # Streaming path passes ``current_step`` explicitly so this
        # stash is purely a safety net for the non-streaming path.
        self._last_current_step = current_step

        # dmd_context is hardcoded to "self". The DMD scoring path does
        # not need clean_x_GT; the aux pass builds its own GT clean
        # half internally from the ride window.
        clean_x_GT = None
        cond_for_scoring, uncond_for_scoring = self._streaming_noisy_cond_slice(info)
        clean_cond, clean_uncond = self._streaming_clean_cond_slice(info)

        (
            sc_clean_x, sc_aug_t,
            sc_clean_x_real, sc_aug_t_real,
            cond_for_scoring, uncond_for_scoring,
            sc_clean_x_aux, sc_aug_t_aux,
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
            stash["dmd_context"] = "self"
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

        # gt_target for streaming: GT video at the chunk's noisy_x
        # positions (= ride_latents_window indices [cf + noisy_start_sdn
        # : cf + noisy_start_sdn + chunk_size]). ``noisy_start_sdn =
        # current_length - new_frames - overlap`` already accounts for
        # the iter's overlap region.
        # Always build it (cheap slice) so the MAE-vs-GT diagnostics
        # GT slice for the always-on real_score_mae_vs_gt diagnostic.
        gt_target = None
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

        # Asymmetric DMD scoring path (dmd_one_step). Supervise the
        # student's FIRST rolled chunk (npb frames, lowest drift)
        # against the ``num_training_frames`` GT frames immediately
        # preceding it. The scorers see v14's training contract for its
        # final chunk: 21 clean GT + 1 noisy chunk at RoPE [21, 24).
        # ``_build_asymmetric_scoring_inputs`` raises (no silent
        # fallback) if the ride lacks the required GT history.
        asymmetric = bool(self.dmd_asymmetric_scoring_enabled)
        if bool(self.dmd_42f_enabled):
            f42 = self._build_42f_scoring_inputs(chunk, info)
            score_image = f42["noisy_x"]
            score_cond = f42["cond"]
            score_uncond = f42["uncond"]
            score_clean_x = f42["clean_x"]
            score_aug_t = f42["aug_t"]
            score_gt_target = f42["gt_target"]
            score_grad_mask = f42["gradient_mask"]
            score_clean_x_real = None
            score_aug_t_real = None
            asym_rope_offset = f42["rope_offset"]   # None → keep offset=npb
        elif asymmetric:
            asym = self._build_asymmetric_scoring_inputs(chunk, info)
            score_image = asym["noisy_x"]
            score_cond = asym["cond"]
            score_uncond = asym["uncond"]
            score_clean_x = asym["clean_x"]
            score_aug_t = asym["aug_t"]
            score_gt_target = asym["gt_target"]
            score_grad_mask = asym["gradient_mask"]
            score_clean_x_real = None
            score_aug_t_real = None
            asym_rope_offset = asym["rope_offset"]
        else:
            score_image = chunk
            score_cond = cond_for_scoring
            score_uncond = uncond_for_scoring
            score_clean_x = sc_clean_x
            score_aug_t = sc_aug_t
            score_gt_target = gt_target
            score_grad_mask = gradient_mask_eff
            score_clean_x_real = sc_clean_x_real
            score_aug_t_real = sc_aug_t_real
            asym_rope_offset = None

        # Standard DMD: random exit-rung output is graph-on; DMD
        # samples a timestep from the rung-bounded range and computes
        # the score-matching loss. Dual-teacher routing applies when a
        # frozen merged-v14 teacher is held — DMD scoring forwards
        # through THAT module (clean p_real, no LoRA-active drift)
        # and passes clean_x_real=None so the frozen teacher sees only
        # the self-view clean_x. The aux pass below uses the LoRA
        # teacher with the GT-mixed clean_x_real.
        dual_teacher_active = self.real_score_frozen is not None
        ema_real_score_active = (
            not dual_teacher_active
            and self.real_score_ema_weight > 0.0
            and self._real_score_ema_lora_state is not None
        )
        with self._maybe_asymmetric_tf_rope_offset(asym_rope_offset):
            if dual_teacher_active:
                _saved_real_score = self.real_score
                self.real_score = self.real_score_frozen
                try:
                    dmd_loss, dmd_log = self.compute_distribution_matching_loss(
                        image_or_video=score_image,
                        conditional_dict=score_cond,
                        unconditional_dict=score_uncond,
                        gradient_mask=score_grad_mask,
                        denoised_timestep_from=info.get("denoised_timestep_from"),
                        denoised_timestep_to=info.get("denoised_timestep_to"),
                        clean_x=score_clean_x, aug_t=score_aug_t,
                        gt_target=score_gt_target,
                        clean_x_real=None, aug_t_real=None,
                    )
                finally:
                    self.real_score = _saved_real_score
            elif ema_real_score_active:
                # Target-network EMA path: swap the live LoRA params to
                # their EMA values for the DMD scoring forward, restore
                # afterwards. The aux teacher loss (which runs LATER in
                # this same compute_generator_loss_streaming call) sees
                # the LIVE params unchanged. See _real_score_ema_swap for
                # the swap/restore mechanics.
                with self._real_score_ema_swap():
                    dmd_loss, dmd_log = self.compute_distribution_matching_loss(
                        image_or_video=score_image,
                        conditional_dict=score_cond,
                        unconditional_dict=score_uncond,
                        gradient_mask=score_grad_mask,
                        denoised_timestep_from=info.get("denoised_timestep_from"),
                        denoised_timestep_to=info.get("denoised_timestep_to"),
                        clean_x=score_clean_x, aug_t=score_aug_t,
                        clean_x_real=score_clean_x_real,
                        aug_t_real=score_aug_t_real,
                        gt_target=score_gt_target,
                    )
            else:
                dmd_loss, dmd_log = self.compute_distribution_matching_loss(
                    image_or_video=score_image,
                    conditional_dict=score_cond,
                    unconditional_dict=score_uncond,
                    gradient_mask=score_grad_mask,
                    denoised_timestep_from=info.get("denoised_timestep_from"),
                    denoised_timestep_to=info.get("denoised_timestep_to"),
                    clean_x=score_clean_x, aug_t=score_aug_t,
                    gt_target=score_gt_target,
                    clean_x_real=score_clean_x_real,
                    aug_t_real=score_aug_t_real,
                )
        # Resolved DMD loss weight: applies the start-step gate +
        # linear warmup ramp on top of the static ``dmd_loss_weight``.
        # ``current_step`` was already plumbed via ``info`` (also used
        # by the aux teacher schedule resolver above).
        dmd_weight_resolved = self._resolved_dmd_loss_weight(current_step)
        dmd_loss = dmd_loss * dmd_weight_resolved
        # v28+: add the anti-collapse term UNSCALED by dmd_weight_resolved,
        # so the std/mean floors fire at their nominal weight even when
        # the DMD warmup ramp is still small (e.g. step 60 with 150-step
        # warmup → 13% effective DMD weight; old code damped anti_collapse
        # to the same fraction, defeating the gray/black collapse guard).
        if getattr(self, "_latest_anti_collapse_total", None) is not None:
            dmd_loss = dmd_loss + self._latest_anti_collapse_total
            self._latest_anti_collapse_total = None

        # v28E_12+: multi-horizon stat-anchor hinge (M2 + TV against
        # the cf-prefix seed window). See ``compute_stat_anchor_loss``
        # in model.anti_collapse for the math. Streaming twin of the
        # equivalent block in ``generator_loss``. Seed window is the
        # first ``cf_state`` frames of ``ride_latents_window``; pred is
        # the streaming chunk (= the rolled student x0 for this iter).
        stat_anchor_w_resolved = self._resolved_stat_anchor_loss_weight(
            current_step
        )
        dmd_log["stat_anchor_weight_resolved"] = float(stat_anchor_w_resolved)
        # ``target_matching`` mode does not require seed_latents; we
        # only need cf_state > 0 / ride_window check in the seed_anchor
        # branch. The mode dispatch lives inside the gate so the call
        # sequence stays identical (one ``compute_*`` call, same total
        # add to dmd_loss).
        # Stat anchor (streaming). Anchor SOURCE by mode:
        #   seed_anchor (default): the GT seed window (ride_window[:cf]).
        #   target_matching: per-chunk k-CLOSEST GT chunks for ALL stats
        #     (``_matched_gt_stat_anchors`` -> compute_stat_anchor_loss's
        #     precomputed-anchor path). ONE compute call either way.
        _mode = getattr(self, "stat_anchor_mode", "seed_anchor")
        if stat_anchor_w_resolved > 0.0:
            try:
                from model.anti_collapse import compute_stat_anchor_loss
                _wk = dict(
                    STD_short_weight=self.stat_anchor_STD_short_weight,
                    STD_long_weight=self.stat_anchor_STD_long_weight,
                    M2_short_weight=self.stat_anchor_M2_short_weight,
                    M2_long_weight=self.stat_anchor_M2_long_weight,
                    TV_short_weight=self.stat_anchor_TV_short_weight,
                    TV_long_weight=self.stat_anchor_TV_long_weight,
                    SOS_short_weight=self.stat_anchor_SOS_short_weight,
                    SOS_long_weight=self.stat_anchor_SOS_long_weight,
                    M1_short_weight=self.stat_anchor_M1_short_weight,
                    M1_long_weight=self.stat_anchor_M1_long_weight,
                    rel_tol_short=self.stat_anchor_rel_tol_short,
                    rel_tol_long=self.stat_anchor_rel_tol_long,
                )
                if _mode == "target_matching":
                    _matched = self._matched_gt_stat_anchors(chunk.float())
                elif _mode == "gt_window":
                    _matched = self._gt_window_stat_anchors(
                        chunk.float(), chunk_lo,
                    )
                else:
                    _matched = None
                if _matched is not None:
                    stat_loss, stat_logs = compute_stat_anchor_loss(
                        pred_x0=chunk.float(), seed_latents=None,
                        seed_STD_anchor=_matched["STD"],
                        seed_M2_anchor=_matched["M2"],
                        seed_TV_anchor=_matched["TV"],
                        seed_SOS_anchor=_matched["SOS"],
                        seed_M1_anchor=_matched["M1"], **_wk,
                    )
                elif cf_state > 0 and ride_window.shape[1] >= cf_state:
                    seed_latents_stream = ride_window[:, :cf_state].detach()
                    # Long-horizon EMA anchor when enabled.
                    long_STD_ov = long_M2_ov = long_TV_ov = None
                    if self.stat_anchor_long_ema_enabled:
                        (
                            long_STD_ov,
                            long_M2_ov,
                            long_TV_ov,
                        ) = self._update_stat_anchor_long_ema(
                            seed_latents_stream
                        )
                    stat_loss, stat_logs = compute_stat_anchor_loss(
                        pred_x0=chunk.float(),
                        seed_latents=seed_latents_stream.float(),
                        long_STD_anchor_override=long_STD_ov,
                        long_M2_anchor_override=long_M2_ov,
                        long_TV_anchor_override=long_TV_ov,
                        **_wk,
                    )
                else:
                    stat_loss, stat_logs = None, {}
                if stat_loss is not None:
                    stat_loss = (
                        stat_anchor_w_resolved * stat_loss.to(dmd_loss.dtype)
                    )
                    dmd_loss = dmd_loss + stat_loss
                    dmd_log["stat_anchor_total"] = stat_loss.detach()
                    dmd_log["stat_anchor_mode"] = float(
                        1.0 if _matched is not None else 0.0
                    )
                    for k, v in stat_logs.items():
                        dmd_log[k] = v
            except Exception as exc:
                if _is_main():
                    import logging as _logging
                    _logging.warning(
                        "[ActionForcingDMD] stat_anchor_loss "
                        "(streaming) failed (%s); skipping this "
                        "iteration.", exc,
                    )

        # Flash-DMD: when enabled, the rolling rollout emitted a
        # per-block t=flash_dmd_gan_t grad-on forward. The chunk_size-
        # aligned slab (overlap from prior iter's t=gan_t pred + this
        # iter's grad-active t=gan_t preds) lives in
        # ``info["flash_dmd_gan_chunk"]``. We surface it as
        # ``info["flash_dmd_gan_x0"]`` so the trainer's GAN and aux
        # losses can consume it. Grad path:
        #   adv_loss → flash_dmd_gan_x0 → t=gan_t gen forward → gen.
        # The gradient does NOT traverse the high-noise denoising
        # rungs (no_grad); GAN supervision is restricted to texture
        # refinement at near-clean noise.
        if self.flash_dmd_enabled:
            last_rung_chunk = info.get("flash_dmd_gan_chunk")
            if last_rung_chunk is None:
                raise RuntimeError(
                    "flash_dmd_enabled=True but "
                    "info['flash_dmd_gan_chunk'] is None — the "
                    "rollout must run with flash_dmd_enabled=True so "
                    "the pipeline emits the t=flash_dmd_gan_t output."
                )
            info["flash_dmd_gan_x0"] = last_rung_chunk
            # v29: also anchor std/mean on the flash-DMD t=gan_t rung's
            # x0. Without this, the GAN-supervised rung is unconstrained
            # by anti-collapse and provides a degenerate-mode escape
            # hatch (gen learns to satisfy disc at t=gan_t while letting
            # the random-exit rung drift toward gray/black). Same loss
            # math, same weights, unscaled add (parallel to the
            # random-exit anti_collapse contribution above).
            # Gated by ``anti_collapse_apply_to_flash_rung`` (default
            # True) so the flash-rung contribution can be disabled
            # independently of the DMD random-exit rung's.
            if self.anti_collapse_apply_to_flash_rung:
                flash_anti_collapse_total = self._compute_anti_collapse_term(
                    original_latent=last_rung_chunk,
                    gt_target=gt_target,
                    log_dict=dmd_log,
                    log_prefix="flash_",
                    ref_dtype=dmd_loss.dtype,
                )
                if flash_anti_collapse_total is not None:
                    dmd_loss = dmd_loss + flash_anti_collapse_total

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
        # Hard gate: aux pass fires only when (a) real_teacher_train_online,
        # (b) loss weight > 0, AND (c) we're past
        # ``aux_teacher_start_step``. Below the start step the real_score
        # forward is skipped entirely so no LoRA gradient accumulates.
        # Belt-and-braces companion gate exists in the trainer's
        # real_teacher_optimizer.step() block.
        aux_active = (
            self.real_teacher_train_online
            and self.aux_teacher_loss_weight > 0.0
            and int(current_step) >= int(self.aux_teacher_start_step)
            # When separate_backward is on, the aux teacher runs as its OWN
            # backward at end-of-step (trainer) so its 1.3B forward graph
            # never coexists with the GAN R1 backward — do NOT fuse it here.
            and not getattr(self, "aux_teacher_separate_backward", False)
        )
        if aux_active:
            aux_loss, aux_log = self._compute_aux_teacher_loss_streaming(
                chunk=chunk,
                gradient_mask_eff=gradient_mask_eff,
                cond_for_scoring=cond_for_scoring,
                # ``sc_clean_x_aux`` is the self-view (= ``sc_clean_x``);
                # the aux function builds its own GT clean half from
                # the ride window. Param name kept for legacy reasons.
                sc_clean_x_real=sc_clean_x_aux,
                sc_aug_t_real=sc_aug_t_aux,
                info=info,
            )
            if aux_loss is not None:
                # Resolved aux teacher loss weight applies the start-
                # step gate + linear warmup ramp on top of the static
                # ``aux_teacher_loss_weight``. Mirrors the DMD weight
                # ramp; ``warmup_steps=0`` reproduces instantaneous
                # full weight at start_step.
                aux_weight_resolved = self._resolved_aux_teacher_loss_weight(
                    int(current_step)
                )
                total_loss = total_loss + aux_weight_resolved * aux_loss
                aux_log["aux_teacher_loss_weight_resolved"] = float(
                    aux_weight_resolved
                )
        elif (
            self.real_teacher_train_online
            and self.aux_teacher_loss_weight > 0.0
        ):
            # Online aux is configured but the start-step gate is closed
            # this iter. Surface a dedicated diagnostic so the wandb
            # plot shows the frozen-pre-start window distinctly from
            # "aux is off entirely".
            aux_log["aux_teacher_frozen_pre_start"] = 1.0
            aux_log["aux_teacher_loss_weight_resolved"] = 0.0
        else:
            aux_log["aux_teacher_loss_weight_resolved"] = 0.0
        # Visible-in-wandb gate state every iter (1.0 when fired this
        # step, 0.0 when skipped — including the no-online case).
        aux_log["aux_teacher_active"] = 1.0 if aux_active else 0.0

        dmd_log["streaming_new_frames"] = float(info["new_frames"])
        dmd_log["streaming_current_length"] = float(info["current_length"])
        # Visible-in-wandb DMD-weight ramp state. Traces 0 → full over
        # ``[dmd_loss_start_step, dmd_loss_start_step + dmd_loss_warmup_steps]``.
        dmd_log["dmd_loss_weight_resolved"] = float(dmd_weight_resolved)
        dmd_log["dmd_loss_active"] = 1.0 if dmd_weight_resolved > 0.0 else 0.0
        # dmd_context is hardcoded to "self" — GT contribution is 0.
        dmd_log["dmd_context_mix_p"] = 0.0
        for k, v in aux_log.items():
            dmd_log[k] = v

        # UNCONDITIONAL graph anchor through the rollout chunk.
        # When the start-step gates are all closed (early production
        # steps with dmd_loss_start_step=50, aux_teacher_start_step
        # =10, gan_critic_warmup_steps=10) the gen-side losses sum to
        # 0-weighted contributions and leaf-zero placeholders, leaving
        # ``total_loss.requires_grad=False`` and
        # ``total_loss.grad_fn=None`` → ``.backward()`` raises
        # "element 0 of tensors does not require grad".
        #
        # The anchor flows through ``chunk`` (the rollout output)
        # rather than a single generator parameter for a critical
        # DDP reason: the gen rollout makes MANY forwards per iter
        # (random-exit rung, post-exit no_grad chain, flash-DMD t=60
        # forward, context-noise commit), and ``chunk`` carries the
        # autograd subgraph from EVERY grad-on forward via the
        # cat+write-into-output chain. Anchoring through ``chunk``
        # therefore registers every gen-forward output's grad chain
        # in the backward pass. A parameter-based anchor only feeds
        # grad to one param, which DDP rejects (find_unused_parameters
        # =True helps for unused PARAMS but not unused forward
        # OUTPUTS).
        #
        # Unconditional (not gated on ``not requires_grad``) because
        # the cost is negligible (~5 MB transient + one cast +
        # multiply + sum, all bf16/fp32 cheap ops) and being
        # unconditional eliminates the ambiguity of WHEN exactly the
        # downstream pieces lose grad. Numerical impact: zero (the
        # multiplier is 0.0; the optimizer step is bit-identical to
        # without the anchor).
        # Anchor when chunk has grad: ``((x - x.detach()) ** 2).mean()``
        # is mathematically zero (x - x.detach() = 0) but forces
        # PyTorch to construct the autograd graph through 4 ops.
        # When chunk has no grad (observed at 32-rank early-iter
        # production runs — under investigation), the anchor itself
        # is detached and can't save the graph. In that case the
        # trainer-side skip (in ``_streaming_train_one_chunk``) will
        # detect ``not generator_loss.requires_grad`` and skip
        # backward in DDP-lockstep — bit-identical to running
        # backward on a sum of 0-weighted losses (zero gradient
        # either way).
        if chunk.requires_grad:
            anchor = ((chunk - chunk.detach()).float().pow(2).mean()) * 0.0
            total_loss = total_loss + anchor
            dmd_log["gen_loss_graph_anchor_used"] = 1.0
        else:
            dmd_log["gen_loss_graph_anchor_used"] = 0.0
        return total_loss, dmd_log

    def run_extra_aux_pass(
        self,
        chunk: torch.Tensor,
        info: Dict[str, Any],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """Compute a fresh aux teacher loss on a detached chunk with
        a new (ε, t). Used by the trainer's ``teacher_cadence='fake'``
        path to run extra LoRA training steps per outer iter without
        re-rolling a new student chunk.

        Mirrors the aux-pass setup in
        ``compute_generator_loss_streaming`` but produces only the
        aux loss (no DMD / GAN). Returned loss is already
        scaled by the resolved ``aux_teacher_loss_weight``.

        ``chunk`` must be detached — gradient never flows back to the
        student through these extra passes (semantically matches
        ``aux_teacher_send_student_grad=False`` regardless of config).
        Returns ``(None, log)`` when aux is gated off, missing inputs,
        or all ranks agreed to skip via the short-ride sync inside
        ``_compute_aux_teacher_loss_streaming``.
        """
        current_step = int(info.get("current_step", 0))
        aux_active = (
            self.real_teacher_train_online
            and self.aux_teacher_loss_weight > 0.0
            and current_step >= int(self.aux_teacher_start_step)
        )
        if not aux_active:
            return None, {"aux_teacher_active_extra": 0.0}
        if self.streaming_state is None:
            return None, {"aux_teacher_active_extra": 0.0}
        per_iter_mask = info.get("gradient_mask")
        if per_iter_mask is None:
            return None, {"aux_teacher_active_extra": 0.0}
        last_chunk_mask = self._dmd_score_grad_mask(chunk.shape, chunk.device)
        gradient_mask_eff = per_iter_mask & last_chunk_mask

        clean_x_self = self._streaming_build_clean_x_self(chunk, info)
        # dmd_context=self (hardcoded). The aux function builds its
        # own GT clean half internally from the ride window.
        clean_x_GT = None
        cond_for_scoring, uncond_for_scoring = self._streaming_noisy_cond_slice(info)
        clean_cond, clean_uncond = self._streaming_clean_cond_slice(info)
        # Detach every tensor reachable through the cond dicts and the
        # clean_x views. The first gen iter's backward already consumed
        # the autograd graph attached to ``info["conditional_dict"]`` /
        # ``info["clean_conditional_dict"]`` (which carry grad into
        # action-projection params via ``build_action_conditional``)
        # and to ``streaming_state["anchor_chunk"]`` (already detached
        # in setup, but defensive). Without this, re-using the same
        # tensor objects in a fresh aux forward triggers a "Trying to
        # backward through the graph a second time" RuntimeError on
        # the second ``aux_loss_extra.backward()``. The aux pass needs
        # only the LoRA-side grad anyway (and the chunk arg is already
        # detached by the caller), so dropping the rest is correct.
        def _detach_cond(d):
            return {
                k: (v.detach() if torch.is_tensor(v) else v)
                for k, v in d.items()
            }
        cond_for_scoring = _detach_cond(cond_for_scoring)
        uncond_for_scoring = _detach_cond(uncond_for_scoring)
        clean_cond = _detach_cond(clean_cond)
        clean_uncond = _detach_cond(clean_uncond)
        clean_x_self = clean_x_self.detach()
        if clean_x_GT is not None:
            clean_x_GT = clean_x_GT.detach()
        (
            sc_clean_x, sc_aug_t,
            sc_clean_x_real, sc_aug_t_real,
            cond_for_scoring, uncond_for_scoring,
            sc_clean_x_aux, sc_aug_t_aux,
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

        aux_loss, aux_log = self._compute_aux_teacher_loss_streaming(
            chunk=chunk,
            gradient_mask_eff=gradient_mask_eff,
            cond_for_scoring=cond_for_scoring,
            sc_clean_x_real=sc_clean_x_aux,
            sc_aug_t_real=sc_aug_t_aux,
            info=info,
        )
        if aux_loss is None:
            aux_log["aux_teacher_active_extra"] = 0.0
            return None, aux_log
        weight = self._resolved_aux_teacher_loss_weight(current_step)
        aux_loss = weight * aux_loss
        aux_log["aux_teacher_active_extra"] = 1.0
        aux_log["aux_teacher_loss_weight_resolved_extra"] = float(weight)
        return aux_loss, aux_log

    def _compute_critic_loss_streaming_gtfix(
        self,
        chunk: torch.Tensor,
        info: Dict[str, Any],
        *,
        mode: str,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Critic step for the gt-context DMD modes (asymmetric or 42f).
        Trains fake_score under the EXACT same (clean_x, noisy_x, cond,
        rope_offset) view the generator scores under, so the (fake-real)
        DMD gradient is computed against a fake_score trained on matched
        conditioning. ``chunk`` arrives detached.

        The cond action streams are detached so the critic's backward
        trains ONLY fake_score params, never the shared action
        projections (those are trained by the generator rollout).
        """
        if mode == "42f":
            inp = self._build_42f_scoring_inputs(chunk, info)
        else:
            inp = self._build_asymmetric_scoring_inputs(chunk, info)
        noisy_x = inp["noisy_x"].detach()
        clean_x = inp["clean_x"]
        aug_t = inp["aug_t"]
        cond = {
            k: (v.detach() if torch.is_tensor(v) else v)
            for k, v in inp["cond"].items()
        }
        grad_mask = inp["gradient_mask"]
        rope_offset = inp["rope_offset"]

        critic_timestep = self._sample_dmd_timestep(
            batch_size=noisy_x.shape[0], num_frame=noisy_x.shape[1],
            denoised_timestep_from=info.get("denoised_timestep_from"),
            denoised_timestep_to=info.get("denoised_timestep_to"),
            device=noisy_x.device,
        )
        critic_noise = torch.randn_like(noisy_x)
        noised = self.scheduler.add_noise(
            noisy_x.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1),
        ).unflatten(0, noisy_x.shape[:2])

        with self._maybe_asymmetric_tf_rope_offset(rope_offset):
            _, pred_fake_image = self.fake_score(
                noisy_image_or_video=noised,
                conditional_dict=cond,
                timestep=critic_timestep,
                clean_x=clean_x,
                aug_t=aug_t,
            )

        if self.args.denoising_loss_type == "flow":
            from utils.wan_wrapper import WanDiffusionWrapper
            flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake_image.flatten(0, 1),
                xt=noised.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1),
            )
            pred_fake_noise = None
        else:
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1),
                xt=noised.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1),
            ).unflatten(0, noisy_x.shape[:2])

        gradient_mask_flat = grad_mask.flatten(0, 1)
        denoising_loss = self.denoising_loss_func(
            x=noisy_x.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred,
            gradient_mask=gradient_mask_flat,
        )
        critic_log: Dict[str, Any] = {
            "streaming_new_frames": float(info["new_frames"]),
            "streaming_current_length": float(info["current_length"]),
            "critic_gtfix_mode": (1.0 if mode == "42f" else 2.0),
            # Populate the display key (the per-iter stdout line reads
            # ``critic_log_dict.get('critic_loss')``); without it the log
            # shows a misleading critic_loss=0.0000 even though
            # fake_score IS training (see fake_grad_norm).
            "critic_loss": float(denoising_loss.detach().item()),
        }
        # Train the forward noiser on aligned (rollout1, rollout2) pairs.
        # The 42f/asymmetric critic path returns here BEFORE the FN block
        # in ``compute_critic_loss_streaming``, so the FN was never trained
        # under dmd_42f_enabled — wire it in explicitly (mirrors the non-42f
        # block). Folded into the critic loss so one backward lights up both
        # fake_score and forward_noiser params; the trainer's separate FN
        # optimizer steps the noiser grads.
        if self.forward_noiser_enabled and self.forward_noiser is not None:
            fn_loss = self._compute_forward_noiser_loss(
                chunk=chunk, info=info, critic_log=critic_log,
            )
            if fn_loss is not None:
                denoising_loss = denoising_loss + (
                    self.forward_noiser_loss_weight
                    * fn_loss.to(denoising_loss.dtype)
                )
        return denoising_loss, critic_log

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

        # gt-context critics: train fake_score on the EXACT conditioning
        # the generator's DMD uses. Keeping the critic's fake_score
        # forward identical to the generator's is mandatory — a
        # (fake - real) DMD gradient computed under one conditioning while
        # fake_score was trained under another is a silent inconsistency.
        if bool(self.dmd_42f_enabled):
            return self._compute_critic_loss_streaming_gtfix(
                chunk, info, mode="42f",
            )
        if bool(self.dmd_asymmetric_scoring_enabled):
            return self._compute_critic_loss_streaming_gtfix(
                chunk, info, mode="asym",
            )

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
            _aux1, _aux2,
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

        # ===== v25 GT + causal-AR clean_x source replacement =====
        # When ``critic_clean_x_source == "gt_causal_ar"``, fake_score's
        # main critic loss flips from "denoise student x0 -> student x0"
        # to "denoise causal-AR-noised GT -> true GT". Mirrors the v21
        # aux-teacher pattern but on fake_score's critic step.
        #
        # Requires the fake-score alt head to be present (used in the
        # auxiliary no_grad forward to produce causal_AR_GT). When alt
        # head is unavailable OR the chunk's abs positions exceed the
        # ride_latents_window range, falls back silently to the legacy
        # self path.
        critic_use_gt_causal_ar = (
            self.critic_clean_x_source == "gt_causal_ar"
            and self.fake_alt_head_enabled
            and getattr(self.fake_score, "has_alt_head", False)
        )
        critic_main_target = chunk
        if critic_use_gt_causal_ar:
            abs_new_start = int(info.get("abs_frame_start", 0))
            overlap_critic = int(info.get("overlap", 0))
            chunk_abs_start = abs_new_start - overlap_critic
            chunk_size_critic = int(chunk.shape[1])
            ride = s["ride_latents_window"]
            if (
                chunk_abs_start < 0
                or chunk_abs_start + chunk_size_critic > int(ride.shape[1])
            ):
                critic_use_gt_causal_ar = False
            else:
                gt_chunk = ride[
                    :, chunk_abs_start : chunk_abs_start + chunk_size_critic
                ].to(dtype=chunk.dtype, device=chunk.device).detach()
                # Build causal_AR_GT via a no_grad fake_score.alt forward
                # on noised GT at critic_timestep. Uses the same self-
                # derived TF context built above (alt head was trained
                # under that context).
                aux_noise = torch.randn_like(gt_chunk)
                noisy_gt_for_alt = self.scheduler.add_noise(
                    gt_chunk.flatten(0, 1),
                    aux_noise.flatten(0, 1),
                    critic_timestep.flatten(0, 1),
                ).unflatten(0, gt_chunk.shape[:2])
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                with torch.no_grad():
                    fs_alt_out = self.fake_score(
                        noisy_image_or_video=noisy_gt_for_alt,
                        conditional_dict=cond_for_scoring,
                        timestep=critic_timestep,
                        compute_alt_head=True,
                        **tf_kwargs,
                    )
                if isinstance(fs_alt_out, tuple) and len(fs_alt_out) >= 4:
                    causal_AR_GT = fs_alt_out[3].detach()
                else:
                    causal_AR_GT = None
                if causal_AR_GT is None:
                    critic_use_gt_causal_ar = False
                else:
                    # Replace noisy_chunk source: noise(causal_AR_GT).
                    critic_noise = torch.randn_like(causal_AR_GT)
                    noisy_chunk = self.scheduler.add_noise(
                        causal_AR_GT.flatten(0, 1),
                        critic_noise.flatten(0, 1),
                        critic_timestep.flatten(0, 1),
                    ).unflatten(0, causal_AR_GT.shape[:2])
                    # Replace clean_x with causal_AR_GT (un-noised) and
                    # zero its aug_t.
                    tf_kwargs["clean_x"] = causal_AR_GT
                    tf_kwargs["aug_t"] = torch.zeros(
                        (causal_AR_GT.shape[0], causal_AR_GT.shape[1]),
                        device=causal_AR_GT.device, dtype=torch.long,
                    )
                    # Flip the main loss target from student chunk to
                    # true GT.
                    critic_main_target = gt_chunk

        # v21: when fake_alt head is BUILT, run fake_score with
        # compute_alt_head=True every critic iter so a single forward
        # produces BOTH the main x0 estimate (for the critic's normal
        # training loss) and the alt-head x0 estimate (for the alt loss
        # against pred_real_EMA). The alt-head's input is detached
        # inside the model, so its gradient never reaches the backbone.
        #
        # Why no start_step gating on alt training: the alt's target
        # (ema_real_x0, the GT-supervised teacher's clean estimate) is
        # already meaningfully different from the main head's target
        # (pred_image, the student distribution) FROM STEP 0 — they
        # don't converge to the same thing. So alt trains from iter 1.
        # The ``fake_alt_apply_start_step`` knob still gates when the
        # aux teacher pass CONSUMES the alt's output (that's where
        # divergence quality matters); alt-head TRAINING is always-on
        # to keep fake_score's DDP gradient bucket consistent
        # (``find_unused_parameters=False`` would hang otherwise).
        current_step = int(info.get("current_step", 0))
        alt_head_present = (
            self.fake_alt_head_enabled
            and getattr(self.fake_score, "has_alt_head", False)
        )
        if alt_head_present:
            fs_out = self.fake_score(
                noisy_image_or_video=noisy_chunk,
                conditional_dict=cond_for_scoring,
                timestep=critic_timestep,
                compute_alt_head=True,
                **tf_kwargs,
            )
            if isinstance(fs_out, tuple) and len(fs_out) >= 4:
                _flow_main, pred_fake_image, _flow_alt, pred_fake_image_alt = (
                    fs_out[0], fs_out[1], fs_out[2], fs_out[3]
                )
            else:
                # has_alt_head was True but the model didn't return alt
                # (shouldn't happen). Fall back to no-alt.
                _, pred_fake_image = fs_out[:2]
                pred_fake_image_alt = None
        else:
            _, pred_fake_image = self.fake_score(
                noisy_image_or_video=noisy_chunk,
                conditional_dict=cond_for_scoring,
                timestep=critic_timestep,
                **tf_kwargs,
            )
            pred_fake_image_alt = None

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
            "streaming_new_frames": float(info["new_frames"]),
            "streaming_current_length": float(info["current_length"]),
        }
        # Surface MAE-extension metrics BEFORE any short-circuit so the
        # critic-side collapse gate (which reads
        # ``baseline_avg_rollout_mae`` off ``critic_log``) never goes
        # blind on the empty-mask early return below — same
        # telemetry-vs-loss-path independence Fix 1 enforced for the
        # gen step.
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
            if pred_fake_image_alt is not None:
                # Keep alt-head params in the DDP gradient bucket so
                # find_unused_parameters=False stays happy. Adding a
                # 0-coefficient term preserves the graph but contributes
                # no gradient magnitude.
                zero_loss = zero_loss + (pred_fake_image_alt.double() * 0.0).sum()
            return zero_loss, critic_log
        gradient_mask_flat = gradient_mask.flatten(0, 1)
        # critic_main_target = chunk under default (self mode); = gt_chunk
        # when ``critic_clean_x_source == "gt_causal_ar"`` and the
        # auxiliary alt forward succeeded (v25 contract).
        denoising_loss = self.denoising_loss_func(
            x=critic_main_target.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred,
            gradient_mask=gradient_mask_flat,
        )
        # ===== v21 alt-head training (v24 target swap) =====
        # Train fake_score's alt head to predict either:
        #   v23 ``ema_real_x0``    : EMA-swapped real_score's x0 estimate
        #                            (computed via an extra no_grad forward
        #                             at the SAME critic_noise/timestep).
        #   v24 ``rollout2_student``: the pre-rolled noisier-AR student x0
        #                            looked up by absolute frame position
        #                            from streaming_state["rollout2_x0"].
        # The alt loss flows only into alt-head params (input detached
        # inside the model); backbone supervision stays with the main
        # critic denoising loss. Always-on training keeps head_alt's
        # params in the DDP gradient bucket.
        if pred_fake_image_alt is not None:
            s_state = self.streaming_state
            use_rollout2_target = (
                self.fake_alt_target_mode == "rollout2_student"
                and s_state is not None
                and s_state.get("rollout2_x0") is not None
            )

            valid_mask_alt = None
            if use_rollout2_target:
                # v24: look up rollout 2's x0 at the same absolute frame
                # positions as ``chunk``. Skip the EMA-real_score forward.
                r2_x0 = s_state["rollout2_x0"]
                r2_abs_start = int(s_state["rollout2_abs_frame_start"])
                r2_total = int(r2_x0.shape[1])
                chunk_size_critic = int(chunk.shape[1])
                # chunk = [previous_chunk[-overlap:], new_frames]; the
                # NEW frames sit at info["abs_frame_start"], overlap
                # frames precede them.
                abs_new_start = int(info.get("abs_frame_start", 0))
                overlap_critic = int(info.get("overlap", 0))
                chunk_abs_start = abs_new_start - overlap_critic
                # Build aligned target tensor + per-frame valid mask.
                target_aligned = chunk.detach().clone()
                valid_mask_alt = torch.zeros(
                    (chunk.shape[0], chunk.shape[1]),
                    dtype=torch.bool, device=chunk.device,
                )
                r2_x0_cast = r2_x0.to(
                    dtype=chunk.dtype, device=chunk.device,
                )
                for f in range(chunk_size_critic):
                    abs_pos = chunk_abs_start + f
                    if r2_abs_start <= abs_pos < r2_abs_start + r2_total:
                        slot = abs_pos - r2_abs_start
                        target_aligned[:, f] = r2_x0_cast[:, slot]
                        valid_mask_alt[:, f] = True
                ema_real_x0_detached = target_aligned.detach()
                # AND-in the v24 valid-position mask so out-of-coverage
                # frames contribute zero to the alt loss.
                gradient_mask_alt = gradient_mask & valid_mask_alt
                gradient_mask_alt_flat = gradient_mask_alt.flatten(0, 1)
                target_name_for_log = "rollout2_student"
            else:
                # v23: EMA-swapped real_score forward (no_grad). Defrag
                # the allocator before this forward — the critic step
                # already holds fake_score's grad-on activations.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                with torch.no_grad():
                    with self._real_score_ema_swap():
                        # Use the SAME clean_x context the fake_score
                        # saw — tf_kwargs already has the right entries.
                        _, ema_real_x0 = self.real_score(
                            noisy_image_or_video=noisy_chunk,
                            conditional_dict=cond_for_scoring,
                            timestep=critic_timestep,
                            **tf_kwargs,
                        )
                    ema_real_x0_detached = ema_real_x0.detach()
                gradient_mask_alt_flat = gradient_mask_flat
                target_name_for_log = "ema_real_x0"

            # Convert alt's x0 prediction to flow space for the
            # FlowPredLoss. Same conversion as the main head's loss
            # uses (see denoising_loss_type=="flow" branch above).
            if self.args.denoising_loss_type == "flow":
                from utils.wan_wrapper import WanDiffusionWrapper
                flow_pred_alt = WanDiffusionWrapper._convert_x0_to_flow_pred(
                    scheduler=self.scheduler,
                    x0_pred=pred_fake_image_alt.flatten(0, 1),
                    xt=noisy_chunk.flatten(0, 1),
                    timestep=critic_timestep.flatten(0, 1),
                )
                pred_fake_alt_noise = None
            else:
                flow_pred_alt = None
                pred_fake_alt_noise = self.scheduler.convert_x0_to_noise(
                    x0=pred_fake_image_alt.flatten(0, 1),
                    xt=noisy_chunk.flatten(0, 1),
                    timestep=critic_timestep.flatten(0, 1),
                ).unflatten(0, chunk.shape[:2])
            alt_loss = self.denoising_loss_func(
                x=ema_real_x0_detached.flatten(0, 1),
                x_pred=pred_fake_image_alt.flatten(0, 1),
                noise=critic_noise.flatten(0, 1),
                noise_pred=pred_fake_alt_noise,
                alphas_cumprod=self.scheduler.alphas_cumprod,
                timestep=critic_timestep.flatten(0, 1),
                flow_pred=flow_pred_alt,
                gradient_mask=gradient_mask_alt_flat,
            )
            denoising_loss = denoising_loss + alt_loss
            with torch.no_grad():
                critic_log["fake_alt_head_loss"] = float(alt_loss.detach().item())
                _diff_rms = (
                    (pred_fake_image_alt.float() - pred_fake_image.float())
                    .pow(2).mean().sqrt().item()
                )
                critic_log["fake_alt_vs_main_rms"] = float(_diff_rms)
                _target_rms = (
                    (ema_real_x0_detached.float() - pred_fake_image.detach().float())
                    .pow(2).mean().sqrt().item()
                )
                critic_log["fake_alt_target_rms"] = float(_target_rms)
                # v24 telemetry: which target source fed the alt loss,
                # and what fraction of chunk frames had rollout-2
                # coverage (1.0 = full overlap; 0.0 = nothing aligned).
                critic_log["fake_alt_target_is_rollout2"] = (
                    1.0 if use_rollout2_target else 0.0
                )
                if valid_mask_alt is not None:
                    critic_log["fake_alt_valid_frac"] = float(
                        valid_mask_alt.float().mean().item()
                    )
                else:
                    critic_log["fake_alt_valid_frac"] = 1.0
        # v25 telemetry: which clean_x source the critic step used this
        # iter (1.0 = gt_causal_ar took effect; 0.0 = self path).
        critic_log["critic_clean_x_is_gt_causal_ar"] = (
            1.0 if critic_use_gt_causal_ar else 0.0
        )

        # v27B: train the forward noiser on aligned (rollout1, rollout2)
        # chunk pairs. The loss is added to the critic step's loss so
        # the same backward call lights up both fake_score and
        # forward_noiser params; the trainer's separate optimizer for
        # forward_noiser will step the noiser-specific grads. Off when
        # forward_noiser_enabled=False (default).
        if (
            self.forward_noiser_enabled
            and self.forward_noiser is not None
        ):
            fn_loss = self._compute_forward_noiser_loss(
                chunk=chunk, info=info, critic_log=critic_log,
            )
            if fn_loss is not None:
                denoising_loss = denoising_loss + (
                    self.forward_noiser_loss_weight
                    * fn_loss.to(denoising_loss.dtype)
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
    def _update_stat_anchor_long_ema(
        self, seed_latents: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cross-rank-average the current step's seed anchors and EMA-
        update the long-horizon anchor buffers. Returns the current
        EMA values as scalar tensors (detached) for use as the
        long-horizon anchor in ``compute_stat_anchor_loss``.

        The EMA buffer state is kept identical across ranks by
        ``all_reduce``-ing the per-rank seed scalars before the EMA
        update. Cold-start initialises buffers to the first step's
        cross-rank mean.

        Only called when ``stat_anchor_long_ema_enabled`` is True;
        the caller is responsible for the gate.
        """
        from model.anti_collapse import (
            _per_frame_STD, _per_frame_M2, _per_frame_TV,
        )
        with torch.no_grad():
            seed = seed_latents.detach().float()
            # Mean across (B, F_seed) → scalar per stat (one rank).
            STD_step = _per_frame_STD(seed).mean()
            M2_step = _per_frame_M2(seed).mean()
            TV_step = _per_frame_TV(seed).mean()
            # Cross-rank average so every rank holds the same EMA
            # buffer state (loss computations must be identical).
            if (
                dist.is_available()
                and dist.is_initialized()
                and dist.get_world_size() > 1
            ):
                ws = float(dist.get_world_size())
                STD_step = STD_step.clone()
                M2_step = M2_step.clone()
                TV_step = TV_step.clone()
                dist.all_reduce(STD_step, op=dist.ReduceOp.SUM)
                dist.all_reduce(M2_step, op=dist.ReduceOp.SUM)
                dist.all_reduce(TV_step, op=dist.ReduceOp.SUM)
                STD_step.div_(ws)
                M2_step.div_(ws)
                TV_step.div_(ws)
            # EMA update. First call cold-starts the buffer at the
            # current step's cross-rank mean (1-sample estimate).
            w = float(self.stat_anchor_long_ema_weight)
            if self._stat_anchor_STD_long_ema is None:
                self._stat_anchor_STD_long_ema = STD_step.detach().clone()
                self._stat_anchor_M2_long_ema = M2_step.detach().clone()
                self._stat_anchor_TV_long_ema = TV_step.detach().clone()
            else:
                self._stat_anchor_STD_long_ema = (
                    w * self._stat_anchor_STD_long_ema
                    + (1.0 - w) * STD_step
                ).detach()
                self._stat_anchor_M2_long_ema = (
                    w * self._stat_anchor_M2_long_ema
                    + (1.0 - w) * M2_step
                ).detach()
                self._stat_anchor_TV_long_ema = (
                    w * self._stat_anchor_TV_long_ema
                    + (1.0 - w) * TV_step
                ).detach()
        return (
            self._stat_anchor_STD_long_ema,
            self._stat_anchor_M2_long_ema,
            self._stat_anchor_TV_long_ema,
        )

    def _resolved_stat_anchor_loss_weight(self, current_step: int) -> float:
        """Return the effective stat_anchor loss weight at this step.

        Below ``stat_anchor_rampdown_start_step``: full weight.
        In ``[start, start + rampdown_steps)``: linear ramp from full
        weight down to 0.
        From ``start + rampdown_steps`` onward: 0.

        Defaults (``rampdown_steps=0``) collapse to a constant
        ``stat_anchor_loss_weight`` for backward compatibility.
        """
        full = float(self.stat_anchor_loss_weight)
        rampdown_steps = int(self.stat_anchor_rampdown_steps)
        if rampdown_steps <= 0 or full == 0.0:
            return full
        s = int(current_step)
        start = int(self.stat_anchor_rampdown_start_step)
        # Floor weight the rampdown decays TO (fraction of full; 0 = legacy).
        floor = full * float(self.stat_anchor_rampdown_floor)
        if s < start:
            return full
        if s >= start + rampdown_steps:
            return floor
        # Linear ramp from full at s=start to floor at s=start+rampdown_steps.
        progress = float(s - start) / float(rampdown_steps)
        return full * (1.0 - progress) + floor * progress

    def _resolved_dmd_loss_weight(self, current_step: int) -> float:
        """Return the effective DMD loss weight at this step.

        Below ``dmd_loss_start_step``: 0.
        In ``[start_step, start_step + warmup_steps)``: linear ramp
        from 0 to ``dmd_loss_weight``.
        From ``start_step + warmup_steps`` onward: ``dmd_loss_weight``.

        Defaults (``start_step=0, warmup_steps=0``) collapse to a
        constant ``dmd_loss_weight`` for backward compatibility.
        """
        s = int(current_step)
        start = int(self.dmd_loss_start_step)
        warmup = int(self.dmd_loss_warmup_steps)
        full = float(self.dmd_loss_weight)
        if s < start:
            return 0.0
        if warmup <= 0:
            return full
        if s >= start + warmup:
            return full
        # Linear ramp from 0 at s=start to full at s=start+warmup.
        return full * float(s - start) / float(warmup)

    def _resolved_aux_teacher_loss_weight(self, current_step: int) -> float:
        """Return the effective aux-teacher loss weight at this step.

        Below ``aux_teacher_start_step``: 0.
        In ``[start_step, start_step + warmup_steps)``: linear ramp
        from 0 to ``aux_teacher_loss_weight``.
        From ``start_step + warmup_steps`` onward:
        ``aux_teacher_loss_weight``.

        Defaults (``warmup_steps=0``) collapse to instantaneous full
        weight at start_step for backward compatibility.
        """
        s = int(current_step)
        start = int(self.aux_teacher_start_step)
        warmup = int(self.aux_teacher_loss_warmup_steps)
        full = float(self.aux_teacher_loss_weight)
        if s < start:
            return 0.0
        if warmup <= 0:
            return full
        if s >= start + warmup:
            return full
        return full * float(s - start) / float(warmup)

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
        ``add_noise(GT, ε, t)`` with a pure-GT clean_x context built
        internally from the ride window, return the FlowPredLoss against
        GT. The teacher input source is always GT.

        ``sc_clean_x_real`` / ``sc_aug_t_real`` are passed by the caller
        as a gate signal: when ``sc_clean_x_real`` is None, the aux
        pass is skipped (build_real_view=False at the call site). The
        actual clean_x used by the teacher is built fresh from
        ``ride_window`` inside this function.

        Returns ``(None, log)`` when the ride is too short for a
        gt_target slice (DDP-synced skip across all ranks).
        """
        s = self.streaming_state
        if sc_clean_x_real is None:
            # build_real_view was False (e.g. critic step). Aux pass
            # disabled this iter — skip.
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

        # Aux teacher noisy_input is ALWAYS noised(GT). The
        # student-grad implicit channel through ``noise_base`` is
        # therefore always closed; the aux loss trains only the LoRA
        # params. ``current_step`` is consumed downstream by the
        # AR-noise burn-in factor.
        current_step = int(info.get("current_step", 0))
        use_gt = True
        noise_base = gt_target

        # ===== v21 fake_alt application =====
        # When the alt head is enabled AND past apply_start_step,
        # shape the clean reference (= noise_base) through fake_score's
        # alt head BEFORE noising. The alt head predicts what the
        # EMA-real teacher would estimate as the clean version of a
        # noised input — i.e. an AR-flavored x0 estimate. Replacing
        # noise_base with this estimate (= "causal_AR_GT") means the
        # real_score is trained to denoise AR-shaped noise back to TRUE
        # GT (target stays gt_target below). The alt forward runs
        # no_grad so neither alt-head nor backbone params receive
        # gradient from the aux teacher loss; alt-head training is
        # exclusively driven by the critic step's alt loss against
        # pred_real_EMA.
        fake_alt_apply_active = (
            self.fake_alt_head_enabled
            and current_step >= int(self.fake_alt_apply_start_step)
            and getattr(self.fake_score, "has_alt_head", False)
        )
        causal_AR_dir_rms = 0.0
        # ``noisy_input`` is built below; the noise vector used for
        # ``add_noise`` is the same vector used as the FlowPredLoss
        # target ``noise`` arg.

        # clean_x for the aux teacher: ALWAYS the pure GT slice
        # covering the chunk's clean-half window. The aux teacher is
        # trained to denoise (noisy GT, ε, t) back to GT — the v14
        # contract — and any student or student/GT-mixed clean_x leaks
        # student-distribution structure into the teacher's
        # conditioning, which was observed to break training (chunk
        # discontinuities in clean_x_aux videos, teacher drift). The
        # only sanctioned non-GT clean_x is ``causal_AR_GT`` (GT with
        # learned forward-noise / alt-head AR shaping), handled
        # downstream when ``aux_real_clean_x_source == "causal_ar_gt"``
        # and the noiser is active.
        #
        # clean_x_GT slice math: covers ride frames
        # ``[cf + noisy_start_sdn - shift, cf + noisy_start_sdn -
        # shift + chunk_size)`` — same window
        # ``_streaming_build_clean_x_GT`` returns. We inline it here
        # to avoid an extra info-dict round-trip; identical math.
        npb = int(s["shift"])
        cf_state = int(s["cf"])
        _clean_start = cf_state + noisy_start_sdn - npb
        _clean_end = _clean_start + chunk_size
        if _clean_end > ride_window.shape[1]:
            # Defensive: if the ride window is too short for the clean
            # slice this iter, skip the aux pass. Shouldn't fire in
            # practice (the noisy gt_target slice above already
            # all-reduce-skips on short rides), but guard anyway.
            return None, {"aux_teacher_skipped_short_clean_ride": 1.0}
        clean_x_for_real = ride_window[
            :, _clean_start:_clean_end,
        ].to(dtype=chunk.dtype, device=chunk.device).detach()
        # Renoise the pure-GT clean_x via the scheduler. Each chunk
        # of ``npb`` frames gets a SINGLE random timestep sampled
        # uniformly from ``[0, clean_x_gt_noise_t]`` (default upper
        # bound 60 — strength ~3/10 of the ladder, lowest rung ~178).
        # Per-chunk random (not per-frame, not progressive) so the
        # teacher sees a noise level that varies stochastically along
        # the rollout, preventing identity collapse without baking in
        # a position-dependent prior. Skipped when
        # ``clean_x_gt_noise_t <= 0``.
        clean_x_noise_t = int(getattr(self, "clean_x_gt_noise_t", 0))
        if clean_x_noise_t > 0:
            B_eff = clean_x_for_real.shape[0]
            F_eff = clean_x_for_real.shape[1]
            npb_local = int(self.num_frame_per_block)
            if F_eff % npb_local != 0:
                raise RuntimeError(
                    "_compute_aux_teacher_loss_streaming: clean_x_for_real "
                    f"has {F_eff} frames which is not divisible by npb="
                    f"{npb_local}. Per-chunk random noise sampling "
                    "requires npb-aligned frame count."
                )
            n_chunks_local = F_eff // npb_local
            t_per_chunk = torch.randint(
                low=0,
                high=clean_x_noise_t + 1,
                size=(B_eff, n_chunks_local),
                device=chunk.device,
                dtype=torch.long,
            )
            _t = t_per_chunk.repeat_interleave(npb_local, dim=1)
            _eps = torch.randn_like(clean_x_for_real)
            _noised = self.scheduler.add_noise(
                clean_x_for_real.flatten(0, 1),
                _eps.flatten(0, 1),
                _t.flatten(0, 1),
            ).unflatten(0, clean_x_for_real.shape[:2]).to(
                dtype=clean_x_for_real.dtype,
            )
            clean_x_for_real = _noised.detach()
            aug_t_for_real = _t
        else:
            # v14's clean half is at zero noise.
            aug_t_for_real = torch.zeros(
                (clean_x_for_real.shape[0], clean_x_for_real.shape[1]),
                device=chunk.device, dtype=torch.long,
            )

        # ===== v27B forward-noiser application =====
        # When enabled, build causal_AR_GT by iteratively applying the
        # learned forward noiser to gt_target chunkwise (per-chunk CARN
        # level matching the student rollout's CARN at that position).
        # This REPLACES the v21 alt-head's causal_AR_x0 application path
        # (which had causal_AR_dir_rms=0 across v21-v28 due to the
        # alt-head's training-vs-application distribution mismatch).
        fn_applied_this_iter = False
        # Gate: enter the apply path only when the learned FN module
        # is built and the strategy is not 'off'. (The legacy
        # 'blur_noise' and 'sum' strategies depended on a 2D Gaussian
        # blur op that has been removed.)
        _strategy = self.forward_noiser_apply_strategy
        _strategy_can_run = (
            _strategy != "off"
            and self.forward_noiser_enabled
            and self.forward_noiser is not None
        )
        if (
            self.forward_noiser_apply_in_aux
            and _strategy_can_run
        ):
            # gt_target's abs frame start = chunk_lo (= cf + noisy_start_sdn).
            # We reconstruct it here to pass to the noiser application
            # helper (the variable isn't held in scope past gt_target
            # construction).
            abs_frame_start_gt = int(cf_state + (
                s["current_length"] - info["new_frames"] - info["overlap"]
            ))
            fn_causal_AR_GT = self._apply_forward_noiser_to_gt(
                gt_target=noise_base.detach()
                if isinstance(noise_base, torch.Tensor) else None,
                abs_frame_start_gt=abs_frame_start_gt,
            )
            if fn_causal_AR_GT is not None:
                # v27E: REPLACE the FN output with GT + low-pass of the
                # FN-added noise. Length-W boxcar (W=3 nulls f=1/3,
                # 2/3, removing the per-chunk high-frequency element);
                # window=0 disables and leaves the FN output alone.
                fn_window = int(
                    self.forward_noiser_lowfreq_smooth_window
                )
                if (
                    fn_window > 0
                    and isinstance(gt_target, torch.Tensor)
                ):
                    with torch.no_grad():
                        residual = (
                            fn_causal_AR_GT.float()
                            - gt_target.float()
                        )
                        smoothed = self._temporal_lowpass_5d(
                            residual, window=fn_window,
                        )
                        boosted = (
                            gt_target.float()
                            + float(self.forward_noiser_lowfreq_alpha)
                            * smoothed
                        ).to(dtype=fn_causal_AR_GT.dtype)
                    fn_causal_AR_GT = boosted
                # Replace noise_base with the CARN-noised version.
                # Detached — no autograd flow from real_score loss back
                # to the noiser (the noiser trains via its own loss in
                # the critic step).
                fn_causal_AR_GT_det = fn_causal_AR_GT.detach().to(
                    dtype=chunk.dtype, device=chunk.device,
                )
                noise_base = fn_causal_AR_GT_det
                # v26/v27B contract: when aux_real_clean_x_source is
                # "causal_ar_gt", clean_x_for_real should match the
                # source noise_base was built from (un-noised). For v21
                # this was causal_AR_x0; for v27B it's fn_causal_AR_GT.
                # Without this, real_score sees noisy_input from
                # causal_AR_GT but TF context from sc_clean_x_real —
                # breaking the "noisy_x and clean_x share source"
                # contract and meaning the eval video stash's
                # ``clean_x_aux`` doesn't actually reflect what the
                # online teacher consumed.
                if self.aux_real_clean_x_source == "causal_ar_gt":
                    clean_x_for_real = fn_causal_AR_GT_det
                    aug_t_for_real = torch.zeros(
                        (clean_x_for_real.shape[0],
                         clean_x_for_real.shape[1]),
                        device=chunk.device, dtype=torch.long,
                    )
                fn_applied_this_iter = True
                # Set causal_AR_dir_rms diagnostic so we can see the
                # forward noiser's effective shift (vs gt_target).
                with torch.no_grad():
                    if isinstance(gt_target, torch.Tensor):
                        causal_AR_dir_rms = float(
                            (fn_causal_AR_GT.float()
                             - gt_target.float())
                            .pow(2).mean().sqrt().item()
                        )

        # ===== v21 fake_alt forward (deferred) =====
        # Now that ``clean_x_for_real`` and ``aug_t_for_real`` are
        # finalised (pure-GT clean_x slice, optionally replaced with
        # causal_AR_GT when the forward noiser is active), run the
        # no_grad fake_alt forward to shape the clean reference into
        # a ``causal_AR`` x0 estimate. Replaces
        # ``noise_base`` so the downstream ``add_noise(noise_base,
        # eps, t)`` call produces an AR-flavored noised input for the
        # real_score's training; the FlowPredLoss target stays
        # ``eps - gt_target`` so the LoRA learns to denoise AR-noise
        # back to TRUE GT.
        # v27B: skip the v21 alt-head application when the forward
        # noiser already replaced noise_base — they serve the same
        # purpose, and running both would double-noise the input.
        if fake_alt_apply_active and fn_applied_this_iter:
            fake_alt_apply_active = False
        if fake_alt_apply_active:
            # Memory hygiene: the aux teacher pass already holds the
            # real_score's grad-on activations; the upcoming fake_alt
            # no_grad forward spikes peak by another WAN-forward's
            # worth (~5-7 GB transient). Defrag the allocator first
            # so the spike fits without fragmentation-driven OOM.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            with torch.no_grad():
                noisy_for_alt = self.scheduler.add_noise(
                    noise_base.detach().flatten(0, 1),
                    eps.flatten(0, 1),
                    t.flatten(0, 1),
                ).unflatten(0, chunk.shape[:2])
                fs_alt_kwargs = {
                    "noisy_image_or_video": noisy_for_alt,
                    "conditional_dict": cond_for_scoring,
                    "timestep": t,
                    "compute_alt_head": True,
                }
                if clean_x_for_real is not None:
                    fs_alt_kwargs["clean_x"] = clean_x_for_real
                if aug_t_for_real is not None:
                    fs_alt_kwargs["aug_t"] = aug_t_for_real
                fs_alt_out = self.fake_score(**fs_alt_kwargs)
                if isinstance(fs_alt_out, tuple) and len(fs_alt_out) >= 4:
                    causal_AR_x0 = fs_alt_out[3]
                else:
                    causal_AR_x0 = None
            if causal_AR_x0 is not None:
                with torch.no_grad():
                    causal_AR_dir_rms = float(
                        (causal_AR_x0.float() - noise_base.detach().float())
                        .pow(2).mean().sqrt().item()
                    )
                # Replace noise_base with the AR-flavored clean
                # reference. Detached — no grad to alt_head or
                # backbone from the aux teacher loss.
                noise_base = causal_AR_x0.detach().to(
                    dtype=chunk.dtype, device=chunk.device,
                )
                # v26: also use causal_AR_x0 as the real_score's
                # clean_x TF context, so the teacher's un-noised
                # reference matches the source the noisy_input was
                # built from. Aug_t is zeros — causal_AR_x0 is the
                # alt head's x0 estimate at zero noise.
                if self.aux_real_clean_x_source == "causal_ar_gt":
                    clean_x_for_real = causal_AR_x0.detach().to(
                        dtype=chunk.dtype, device=chunk.device,
                    )
                    aug_t_for_real = torch.zeros(
                        (clean_x_for_real.shape[0],
                         clean_x_for_real.shape[1]),
                        device=chunk.device, dtype=torch.long,
                    )

        # Teacher forward — full grad on LoRA params (and on chunk
        # via noisy_input when use_gt=False). When state_probe is
        # attached to real_score (= aux training enabled), the wrapper
        # also returns ``state_preds, probe_hidden`` from the probe
        # readout. The 21-frame aux window matches the probe's
        # ``n_chunks * num_frame_per_block`` gate, so state_preds is
        # graph-bearing through both LoRA params (via the taps) and
        # state_probe params (the probe's own weights). We unpack
        # both arities so the same code works with the probe on or off.
        #
        # Activation-checkpoint the OUTER call. The inner per-
        # transformer-block ``real_score_gradient_checkpointing=True``
        # already discards per-layer activations on backward, but the
        # outer call still holds ~30 layer-input tensors (~75 MB each
        # at the aux pass's 21-frame × hidden_dim × bf16 shape =
        # ~2-3 GB total). Wrapping in ``torch.utils.checkpoint`` makes
        # the backward re-run the full real_score forward instead of
        # holding those layer inputs — frees ~3 GB on aux iters.
        # Cost: +15-20% backward wallclock on those iters.
        #
        # Closure capture via default args (pattern identical to the
        # flash-DMD ckpt in the pipeline): ``cond_for_scoring`` /
        # ``t`` / ``clean_x_for_real`` / ``aug_t_for_real`` are
        # loop-local-ish here (rebound per iter) but the closure
        # snapshot below is defensive against any future move into a
        # loop. ``noisy_input`` is passed positionally — it carries
        # the autograd graph back to ``chunk`` when
        # ``aux_teacher_send_student_grad=True``, so it must be a
        # ``Tensor`` arg (not a closed-over name) for checkpoint to
        # plumb backward correctly.
        def _aux_real_score_fn(
            x,
            _gen=self.real_score,
            _cond=cond_for_scoring,
            _t=t,
            _clean=clean_x_for_real,
            _aug=aug_t_for_real,
        ):
            return _gen(
                noisy_image_or_video=x,
                conditional_dict=_cond,
                timestep=_t,
                clean_x=_clean,
                aug_t=_aug,
            )

        # ----- Single Gaussian-only teacher forward -----
        # Build the noisy input from ``noise_base`` (= gt_target /
        # chunk / blend per ``real_teacher_input_source``) and the
        # Gaussian sample ``eps`` at timestep ``t``. The FlowPredLoss
        # target uses the SAME ``eps`` so target = eps - gt_target.
        # ``aux_noisy_from_raw_gt=True`` overrides this and uses the
        # raw ``gt_target`` instead of the (CARN-noised / alt-head-
        # shaped) ``noise_base``, decoupling the noisy_input from any
        # shaping applied to ``clean_x_for_real`` upstream. The clean
        # reference (``clean_x_for_real``) keeps its CARN/alt-head
        # shaping per ``aux_real_clean_x_source``. See the knob's
        # registration in ``__init__`` for the full rationale.
        gradient_mask_flat = gradient_mask_eff.flatten(0, 1)
        _noisy_base = (
            gt_target if getattr(self, "aux_noisy_from_raw_gt", False)
            else noise_base
        )
        noisy_input = self.scheduler.add_noise(
            _noisy_base.flatten(0, 1),
            eps.flatten(0, 1),
            t.flatten(0, 1),
        ).unflatten(0, chunk.shape[:2])

        _real_score_out = _ckpt(
            _aux_real_score_fn, noisy_input, use_reentrant=False,
        )
        if isinstance(_real_score_out, tuple) and len(_real_score_out) >= 4:
            flow_pred, _x0, lora_state_preds, _probe_hidden = (
                _real_score_out[0], _real_score_out[1],
                _real_score_out[2], _real_score_out[3],
            )
        else:
            flow_pred, _x0 = _real_score_out
            lora_state_preds = None

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

        # Eval-time stash: surface the LoRA aux teacher's denoised x0
        # estimate for the sample-video logger. The DMD pass already
        # stashes ``pred_real`` (= the FROZEN teacher's CFG-extrapolated
        # x0); this adds ``pred_real_lora`` (= the LoRA aux teacher's
        # raw x0, no CFG since the aux pass runs cond-only). The
        # trainer's video logger iteration list includes both keys so
        # they decode side-by-side per sample step. Uses the LAST
        # pass's outputs (in two_pass mode this is the AR-augmented
        # pass, which is the more diagnostic of the two).
        stash = getattr(self, "_dmd_eval_stash", None)
        if isinstance(stash, dict):
            stash["pred_real_lora"] = _x0.detach()
            stash["aux_teacher_timestep"] = int(t.flatten()[0].item())
            stash["aux_teacher_input_was_gt"] = 1.0
            # v27B: clean_x_aux = the TF context fed to real_score in
            # the aux teacher pass (the value of clean_x_for_real AFTER
            # any v27B forward-noiser / v21 alt-head override). Lets
            # the eval video logger decode and visually verify what
            # context the online teacher's training step actually
            # consumed each iter.
            if isinstance(clean_x_for_real, torch.Tensor):
                stash["clean_x_aux"] = clean_x_for_real.detach()
            # Also stash the aux noisy_input source (= noise_base) so
            # the side-by-side comparison shows GT vs causal_AR_GT.
            if isinstance(noise_base, torch.Tensor):
                stash["aux_noise_base"] = noise_base.detach()

        # Diagnostics: MAE form of FlowPredLoss target, gradient_mask-
        # weighted. The target is ``eps - gt_target`` (FlowPredLoss
        # contract for the single Gaussian forward).
        with torch.no_grad():
            target_dbg = (eps - gt_target).flatten(0, 1)
            err_dbg = (
                flow_pred.flatten(0, 1).float() - target_dbg.float()
            ).abs()
            mask_f_dbg = gradient_mask_flat.float()
            denom_dbg = mask_f_dbg.sum().clamp_min(1.0)
            err_masked_dbg = (err_dbg * mask_f_dbg).sum() / denom_dbg
            aux_teacher_pred_mae_v = float(err_masked_dbg.item())
            aux_t_mean_v = float(t.float().mean().item())
        # float, not a raw bf16 tensor — see real_teacher log note: a bf16
        # value kills the trainer's whole wandb.log call for the step.
        log: Dict[str, Any] = {
            "aux_teacher_loss": float(loss.detach().item()),
            "aux_teacher_input_was_gt": 1.0,
            "aux_teacher_pred_mae": aux_teacher_pred_mae_v,
            "aux_teacher_t_mean": aux_t_mean_v,
            # v21 fake_alt diagnostics. ``fake_alt_apply_active``: 1.0
            # this iter if alt was applied; ``causal_AR_dir_rms``: RMS
            # of (alt_x0 - clean_reference) — measures how much
            # AR-flavor the alt is injecting. Zero before apply_start
            # (and at first apply iters before alt has trained).
            "fake_alt_apply_active": (
                1.0 if fake_alt_apply_active else 0.0
            ),
            "fake_alt_causal_AR_dir_rms": causal_AR_dir_rms,
        }
        # Expose the graph-bearing LoRA-side outputs so the trainer can
        # fold action_critic z-guidance and state_probe supervision into
        # the gen step's total loss before the single backward. These
        # MUST be tensor-valued (not detached) so gradient flows back
        # into LoRA params via _x0 and into both LoRA + state_probe
        # params via lora_state_preds. Keys are nested under "_aux_teacher_
        # tensors" so the standard ``isinstance(v, dict)`` filter in the
        # trainer's wandb-log unpack skips them (else .item() would fail).
        log["_aux_teacher_tensors"] = {
            "lora_x0": _x0,
            "lora_state_preds": lora_state_preds,
            # Detached GT slice the LoRA was trained to predict this
            # iter. Surfaced so the trainer can compute disc-borrowed
            # regularisers (``aux_teacher_disc_adv_weight`` /
            # ``aux_teacher_disc_feat_weight``) on (lora_x0, gt_target)
            # without re-slicing the ride window.
            "gt_target": gt_target.detach(),
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
            _sc_clean_x_aux,          # aux fields unused on this path
            _sc_aug_t_aux,            # (legacy compute_real_teacher_loss_streaming
                                      # uses dmd-scoring's mix; only the
                                      # gen-step _compute_aux_teacher_
                                      # loss_streaming caller routes the
                                      # aux clean_x.)
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
        # source knob. Mirrors the dispatch in
        # Standalone teacher noisy_input is ALWAYS noised(GT). The K=1
        # path never feeds gradient back to the student regardless;
        # gt_target is the only v14-faithful source.
        use_gt = True
        noise_base = gt_target
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

        # Diagnostics: MAE form of FlowPredLoss target (= |flow_pred -
        # (eps - GT)|), with the same gradient_mask the loss uses. Two
        # purposes:
        #   1. Comparable across LoRA timesteps in MAE units (loss is
        #      MSE so it's quadratic-biased toward outlier frames).
        #   2. Lets us split the LoRA's training error by DMD-rung
        #      bin (high/low noise) — diverging high-rung MAE while
        #      low-rung stays flat is a leading collapse indicator
        #      (the teacher loses high-noise score-matching first).
        with torch.no_grad():
            target = (eps - gt_target).flatten(0, 1)
            err = (flow_pred.flatten(0, 1).float() - target.float()).abs()
            mask_f = gradient_mask_flat.float()
            denom = mask_f.sum().clamp_min(1.0)
            err_masked = (err * mask_f).sum() / denom
            real_teacher_pred_mae_v = float(err_masked.item())
            t_mean = float(t.float().mean().item())
        # Scalars/float32 only: a raw bf16 tensor here propagates into the
        # trainer's wandb payload and kills the WHOLE wandb.log call
        # ("Got unsupported ScalarType BFloat16") — every metric for the
        # step is silently dropped.
        log: Dict[str, Any] = {
            "real_teacher_loss": float(loss.detach().item()),
            "real_teacher_timestep": t.detach().float(),
            "real_teacher_input_was_gt": 1.0 if use_gt else 0.0,
            "real_teacher_pred_mae": real_teacher_pred_mae_v,
            "real_teacher_t_mean": t_mean,
        }
        # Per-rung bin keys (high-noise t > 500 vs low-noise t <= 500).
        # The LoRA online-training loss + MAE split into the same two
        # buckets the gen-side DMD push uses (see _compute_kl_grad), so
        # gen-side push and teacher-side training error can be cross-
        # plotted at each end of the DMD ladder individually.
        _hi = t_mean > 500.0
        _loss_v = float(loss.detach().item())
        log["real_teacher_loss_t_high"] = _loss_v if _hi else 0.0
        log["real_teacher_loss_t_low"] = _loss_v if not _hi else 0.0
        log["real_teacher_pred_mae_t_high"] = (
            real_teacher_pred_mae_v if _hi else 0.0
        )
        log["real_teacher_pred_mae_t_low"] = (
            real_teacher_pred_mae_v if not _hi else 0.0
        )
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
