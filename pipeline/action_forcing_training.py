"""Phase-1 Action-Forcing training pipeline (NO STAIRCASE).

Model after Causal-Forcing's ``SelfForcingTrainingPipeline`` 1:1, with
the following additions for our action-aware student:

  1. Per-block slicing of action conditioning streams. The trainer
     pre-computes the full-video action_modulation [B, F, ...] +
     action_tokens [B, F, ...] (where F = rollout_frames or
     extended_rollout_length when MAE-extension is enabled) and packs
     them into ``conditional_dict``. For each block forward we slice
     the appropriate [start:start+count] window so the DiT sees one
     action stream entry per noisy frame, exactly matching how the ODE
     student was trained.

  2. ``chunks_per_rolling_step`` (default 1, mirroring CF). At = 2 we
     denoise 2 chunks (= 2*npb frames) at the same timestep within a
     single block — both with their own GT actions, no decay.

  3. CF-style decoupling of rollout length and gradient window:
     ``rollout_frames`` (= ``noise.shape[1]`` per call) controls the
     full per-iter rollout; ``num_max_frames`` controls the LAST-N
     slice that gets a grad-enabled exit-flag forward. With
     ``rollout_frames > num_max_frames`` the leading
     ``rollout_frames - num_max_frames`` frames warm the KV cache
     under ``no_grad`` and only the trailing ``num_max_frames`` carry
     gradient. This is the gate at
     ``Causal-Forcing/pipeline/self_forcing_training.py:120``
     (``start_gradient_frame_index = num_output_frames - 21``),
     parameterised so callers don't have to hardcode 21/24.

  4. MAE-driven dynamic rollout extension. After the baseline rollout
     completes, if the last 3 generated latent frames are within
     ``mae_extension_threshold`` of the GT latents (mean-absolute-
     error in latent space), the pipeline rolls one more 3-frame
     chunk under ``no_grad`` (full denoise, all rungs run; the
     resulting frames are committed to the KV cache via the same
     t=context_noise forward used by the baseline blocks). The
     extension keeps firing as long as the last-chunk MAE stays below
     threshold AND we haven't hit ``mae_extension_max_extra_chunks``
     extras AND every DDP rank still has GT to compare against (a
     synced ``MIN`` reduce of remaining GT chunks bounds the loop so
     ranks never drift out of lockstep on the in-loop ``all_reduce``).
     Extensions are NOT returned in ``output`` — the returned tensor
     is always exactly ``rollout_frames`` long, so the DMD loss path
     is unchanged. The extensions are pure forward passes that exist
     to (a) collect a "how far can the student sustain quality"
     metric and (b) bump ``last_chunk_mae`` to a more meaningful
     value (the MAE at the point where the student finally degrades,
     or ride end). Per-call metrics land on
     ``self._last_extension_metrics`` for the caller to log.

     IMPORTANT: extensions do NOT grow the KV buffer. The cache is
     always sized to ``max(num_max_frames, rollout_frames)`` and
     extension chunks commit through the model-side rolling cache
     (sink-pinned + FIFO local window). This is the "we always do
     max 21 frames of cache" invariant — there is no memory cost for
     enabling extensions beyond what the baseline rollout already
     pays.

Otherwise the recipe is byte-for-byte the CF pipeline:

  * Persistent KV cache primed by recursively recording each committed
    chunk via a t=0 (or context_noise) clean forward. Buffer is sized
    to ``max(num_max_frames, rollout_frames)`` so the BASELINE
    rollout fits without triggering eviction. MAE-extension chunks,
    when enabled, commit past that point and the model-side rolling
    cache (``utils/infinity_rope.py``, ``wan/modules/causal_model.py``)
    sink-pins the first ``sink_size`` frames and FIFO-evicts the
    oldest non-sink frames. The cache size is INDEPENDENT of
    ``mae_extension_max_extra_chunks``; that cap bounds compute
    only.
  * Per-block truncated random-exit denoising:
        T -> tau_1 -> ... -> tau_{exit-1}  (no_grad)
            then ONE grad-enabled forward at tau_exit
    with ``exit_flag`` broadcast from rank 0 so every DDP rank picks
    the same denoising index per block (lockstep DMD).
  * ``start_gradient_frame_index = num_output_frames - num_max_frames``
    so only the last ``num_max_frames`` frames receive gradient
    (CF hardcodes 21; we plumb it from ``num_max_frames`` so a future
    change of the fixed scoring window size doesn't drift the gate).
  * After each block's grad forward, the prediction is committed to the
    KV cache via a t=context_noise forward run under ``no_grad``.
  * Returns ``(output, denoised_timestep_from, denoised_timestep_to)``,
    consumed by the DMD loss to ts-schedule the noise injection.

This pipeline is action-aware but otherwise faithful to CF: real_score
and fake_score never see GT context (they score the student's pred
directly via the model's bidirectional forward — no KV cache). The
caller is expected to slice the rolled-out pred to the LAST
``num_max_frames`` frames (matching ``start_gradient_frame_index``)
before passing to the bidirectional scorer, since the scorer's seq_len
is sized to ``num_max_frames`` and the leading warmup frames carry no
gradient anyway.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint as _ckpt

from utils.scheduler import SchedulerInterface
from utils.wan_wrapper import WanDiffusionWrapper


_ACTION_STREAM_KEYS: Tuple[str, ...] = (
    "_action_modulation",
    "_action_tokens",
    "_action_modulation_clean",
    "_action_tokens_clean",
)


def _slice_per_frame_streams(
    conditional_dict: dict,
    frame_start: int,
    frame_count: int,
) -> dict:
    """Return a shallow copy of ``conditional_dict`` with per-frame action
    streams sliced to ``[frame_start:frame_start+frame_count]``.

    Action streams are assumed to be ``[B, F, ...]`` tensors. Other keys
    (``prompt_embeds`` etc.) are passed through unchanged.
    """
    out = {}
    for key, value in conditional_dict.items():
        if (
            key in _ACTION_STREAM_KEYS
            and isinstance(value, torch.Tensor)
            and value.dim() >= 2
        ):
            sliced = value[:, frame_start:frame_start + frame_count]
            if sliced.shape[1] != frame_count:
                raise ValueError(
                    f"Per-frame stream {key!r} sliced to {sliced.shape[1]} "
                    f"frames but block requested {frame_count}. Streams "
                    f"must cover the full ``num_output_frames`` window."
                )
            out[key] = sliced.contiguous()
        else:
            out[key] = value
    return out


class ActionForcingTrainingPipeline:
    """CF-style self-forcing training pipeline with action conditioning."""

    def __init__(
        self,
        denoising_step_list: List[int],
        scheduler: SchedulerInterface,
        generator: WanDiffusionWrapper,
        num_frame_per_block: int = 3,
        chunks_per_rolling_step: int = 1,
        same_step_across_blocks: bool = True,
        last_step_only: bool = False,
        num_max_frames: int = 21,
        rollout_frames: Optional[int] = None,
        context_noise: int = 0,
        exit_flag_weights: Optional[List[float]] = None,
        seed_prefill_frames: int = 0,
        seed_prefill_mode: str = "estimate",
        kv_cache_seed_headroom: bool = False,
        **kwargs,
    ):
        self.scheduler = scheduler
        self.generator = generator

        ds_list = list(denoising_step_list)
        if len(ds_list) > 0 and float(ds_list[-1]) == 0.0:
            ds_list = ds_list[:-1]
        self.denoising_step_list = ds_list

        self.num_transformer_blocks = 30
        self._spatial_frame_seq_length = 1560
        self.num_frame_per_block = int(num_frame_per_block)
        self.chunks_per_rolling_step = int(chunks_per_rolling_step)
        if self.chunks_per_rolling_step < 1:
            raise ValueError(
                f"chunks_per_rolling_step must be >= 1; got "
                f"{self.chunks_per_rolling_step}"
            )
        self.context_noise = int(context_noise)
        self.same_step_across_blocks = bool(same_step_across_blocks)
        self.last_step_only = bool(last_step_only)
        # Optional non-uniform exit-flag sampling. When set, a list of K
        # non-negative floats (K = num_denoising_steps) reweights the
        # per-iter random exit rung. Used to concentrate training on
        # middle-t rungs (which sit in the highest-SNR / lowest-
        # gradient-variance regime — analogous to min-SNR-γ / P2
        # weighting in diffusion training literature) while preserving
        # some training on the extremes. None / empty preserves uniform
        # sampling (default behavior — identical to torch.randint).
        if exit_flag_weights is None or len(exit_flag_weights) == 0:
            self.exit_flag_weights = None
        else:
            w = [float(v) for v in exit_flag_weights]
            if any(v < 0 for v in w):
                raise ValueError(
                    f"exit_flag_weights must be non-negative; got {w}"
                )
            if sum(w) <= 0:
                raise ValueError(
                    f"exit_flag_weights must sum > 0; got {w}"
                )
            self.exit_flag_weights = w
        self.num_max_frames = int(num_max_frames)
        # ``rollout_frames`` decouples the per-iter rollout length from
        # the gradient/scoring window (``num_max_frames``). When unset
        # (or equal to ``num_max_frames``), the pipeline behaves
        # identically to the original CF chunkwise recipe: every rolled
        # frame is inside the gradient window. When greater, the
        # LEADING ``rollout_frames - num_max_frames`` frames run the
        # exit-flag forward under ``no_grad`` (warming the KV cache)
        # and only the TRAILING ``num_max_frames`` frames carry
        # gradient — exactly the gate at
        # ``Causal-Forcing/pipeline/self_forcing_training.py:120``,
        # parameterised. Caller (the DMD model) is responsible for
        # slicing the returned full-rollout pred to the last
        # ``num_max_frames`` before passing to the bidirectional
        # scorer (whose seq_len is sized to ``num_max_frames``).
        self.rollout_frames = int(
            rollout_frames if rollout_frames is not None else num_max_frames
        )
        if self.rollout_frames < self.num_max_frames:
            raise ValueError(
                f"rollout_frames ({self.rollout_frames}) must be >= "
                f"num_max_frames ({self.num_max_frames}); the gradient "
                f"window is the LAST num_max_frames frames of the "
                f"rollout, so the rollout must be at least that long."
            )

        # Legacy per-call metrics dict, retained as an empty stash for
        # back-compat with any downstream consumers; the MAE-extension
        # path was removed and no longer populates it.
        # Keys (always present after a call):
        #   - mae_extension_count (int): number of extension chunks rolled
        #   - last_chunk_mae (float): MAE of the FINAL evaluated chunk
        #     (= baseline last chunk if no extensions, else the chunk
        #     where MAE first hit/exceeded threshold or the last
        #     extension if we hit the cap or ran out of GT)
        #   - baseline_last_chunk_mae (float): MAE of the baseline
        #     rollout's last chunk (always populated when gt_latents
        #     was provided and extension is enabled)
        # NaN signals "metric not computed for this call" (e.g.
        # gt_latents was None, or extension was disabled).
        self._last_extension_metrics: dict = {}

        # -------------------------------------------------------------
        # Seed-prefill contract (14e alignment, 2026-08-17).
        #
        # ``seed_prefill_frames``: how many CONTEXT frames the caller
        # writes into the KV cache before any generated chunk (=
        # ``dmd_context_clean_frames`` on the DMD path). Purely
        # informational for ``kv_cache_size`` — the prefill itself is
        # driven by the caller. 0 (default) = the pipeline knows of no
        # prefill and every size/occupancy computation below collapses
        # to the pre-existing behaviour.
        #
        # ``seed_prefill_mode``: what ``_seed_prefill_chunk`` writes.
        #   * "estimate" (default, legacy): noise the seed to the last
        #     rung, run the generator, commit the model's OWN x0
        #     ESTIMATE at ``context_noise``.
        #   * "real": write the REAL seed latents at t=0, one forward
        #     per chunk — the contract the 14e ODE student was trained
        #     under (``action-forcing/af_model/ode_rollout.py:315-325``).
        #
        # ``kv_cache_seed_headroom``: size the cache for seed + rollout
        # (+ one block) instead of the rollout alone. Implied by
        # ``seed_prefill_mode="real"`` — a real-seed prefill is
        # pointless if the buffer then evicts it mid-rollout.
        # -------------------------------------------------------------
        self.seed_prefill_frames = int(seed_prefill_frames)
        mode = str(seed_prefill_mode).strip().lower()
        if mode not in ("estimate", "real"):
            raise ValueError(
                f"seed_prefill_mode must be 'estimate' or 'real'; got "
                f"{seed_prefill_mode!r}."
            )
        self.seed_prefill_mode = mode
        self.kv_cache_seed_headroom = bool(kv_cache_seed_headroom) or (
            self.seed_prefill_mode == "real"
        )
        # One-shot dedupe for the cache-contract / eviction notices.
        self._cache_occupancy_warned: set = set()
        self._cache_contract_logged: set = set()

        self.kv_cache1: Optional[list] = None
        self.crossattn_cache: Optional[list] = None

        # max_attention_size propagation (2026-08-16): every sibling pipeline
        # (self_forcing_training.py:564, rolling_staircase_training.py:570,
        # ode_rollout.py:204) aligns the attention window to WHOLE frames:
        # local_attn_size * frame_seq_length (action-token AWARE). This
        # pipeline was the only one relying on causal_model.py's own default
        # local_attn_size * 1560 (token-UNAWARE), which starts the window 21
        # tokens INSIDE the oldest live frame — permanently non-frame-aligned.
        _gen_mod = getattr(generator, "model", generator)
        _las = getattr(_gen_mod, "local_attn_size", -1)
        if _las is not None and not isinstance(_las, (list, tuple)) \
                and int(_las) != -1:
            _target = int(_las) * int(self.frame_seq_length)
            for _m in generator.modules():
                if hasattr(_m, "max_attention_size"):
                    _m.max_attention_size = _target

    # -----------------------------------------------------------------
    # Token-shape helpers (KV cache + current_start computation must
    # account for per-frame action tokens — frame_seq_length grows by
    # ``action_tokens_per_frame`` when the action patch is on).
    # -----------------------------------------------------------------
    def _inner_model(self):
        model = self.generator.model
        if hasattr(model, "module"):
            try:
                from torch.nn.parallel import DistributedDataParallel as _DDP
                if isinstance(model, _DDP):
                    model = model.module
            except Exception:
                pass
        if hasattr(model, "get_base_model"):
            try:
                model = model.get_base_model()
            except Exception:
                pass
        return model

    def _peft_model_or_none(self):
        """Walk DDP -> PeftModel without unwrapping further. Returns
        the PeftModel when the generator was wrapped by phase LoRA,
        else None. Used by ``_maybe_set_phase_lora_for_step`` to
        dispatch active adapter per denoising rung."""
        model = self.generator.model
        if hasattr(model, "module"):
            try:
                from torch.nn.parallel import DistributedDataParallel as _DDP
                if isinstance(model, _DDP):
                    model = model.module
            except Exception:
                pass
        if hasattr(model, "set_adapter") and hasattr(model, "peft_config"):
            return model
        return None

    def _maybe_set_phase_lora_for_step(
        self,
        step_idx: int,
        num_steps: int,
        re_arm: bool = True,
    ) -> None:
        """If the generator is peft-wrapped with named ``rung_*``
        adapters (Phased DMD K-LoRA), select the active adapter for
        the current rung. No-op when no such wrap is present.

        Mapping: with K=4 over 4 rungs the mapping is identity;
        K=2 over 4 rungs is ``{0,1}->0, {2,3}->1``; K=1 always 0.
        K is read from the live ``peft_config`` so this auto-syncs
        with whatever the model's ``_apply_student_phase_lora`` set up.

        ``re_arm`` (default True): after ``set_adapter``, force
        ``requires_grad=True`` on ALL ``lora_*`` params so the set of
        params DDP tracks is stationary across iters. Required for
        the grad-on dispatch points (ckpt-internal _exit_fn / _flash_fn
        and the top-of-ODE-loop pre-call) — without re-arm, peft's
        set_adapter side-effect would leave non-active adapters with
        requires_grad=False, breaking the optimizer build and making
        DDP's per-iter tracked-param set non-stationary.

        Pass ``re_arm=False`` from no_grad call sites (seed prefill,
        post-exit finish-denoise loop, non-ckpt flash branch, Step 3.4
        context_noise commit). The forward there is no_grad, so
        gradient routing isn't affected by requires_grad. Skipping the
        re-arm reduces DDP-state churn between forwards — Fix B for
        the ``unmarked_param_indices.empty() ASSERT FAILED`` crash on
        v8e8 + K+1: too many re-arms across the new no_grad dispatch
        sites destabilised DDP's bucket-rebuild metadata. The grad-on
        path's re-arm at iter boundary still restores the canonical
        all-True state before the next backward.
        """
        peft_model = self._peft_model_or_none()
        if peft_model is None:
            return
        # Exclude the optional ``rung_flash`` adapter from the ODE
        # rung count — it's a sibling adapter used only by the flash
        # forward, not part of the K-rung ODE chain.
        names = [
            n for n in peft_model.peft_config.keys()
            if n.startswith("rung_") and n != "rung_flash"
        ]
        if not names:
            return
        K = len(names)
        bucket = max(1, int(num_steps) // K)
        idx = min(K - 1, int(step_idx) // bucket)
        target = f"rung_{idx}"
        if target not in peft_model.peft_config:
            return
        try:
            peft_model.set_adapter(target)
            if re_arm:
                for n, p in peft_model.named_parameters():
                    if "lora_" in n:
                        p.requires_grad = True
        except Exception:
            pass

    def _maybe_set_phase_lora_for_flash(self, re_arm: bool = True) -> None:
        """Phase-LoRA dispatch for the Flash-DMD t=60 forward.

        Routing rule (in priority order):

        1. If a dedicated ``rung_flash`` adapter exists (created by
           ``student_phase_lora_flash_adapter_enabled=true``), route
           flash here. This is the K+1 design: K ODE rungs train
           cleanly on their t-bracket DMD loss, and rung_flash trains
           solely on the t=60 GAN adversarial loss. No inter-task
           interference within any single adapter.

        2. Else, fall back to env var ``FLASH_PHASE_LORA_ROUTE``:
             - unset or ``exit``: no-op. Flash uses whichever adapter
               the ODE loop's exit step left active (uniform-random
               across ODE rungs).
             - ``last``: set active adapter to ``rung_{K-1}``.
               Concentrates flash on the lowest-t ODE rung; freeing
               rungs 0..K-2 but overloading rung_{K-1} with two
               heterogeneous loss signals (DMD + GAN).

        ``re_arm`` semantics: see ``_maybe_set_phase_lora_for_step``.

        No-op when no phase-LoRA wrap is present.
        """
        peft_model = self._peft_model_or_none()
        if peft_model is None:
            return
        target: Optional[str] = None
        if "rung_flash" in peft_model.peft_config:
            target = "rung_flash"
        else:
            import os
            route = (
                os.environ.get("FLASH_PHASE_LORA_ROUTE") or "exit"
            ).strip().lower()
            if route != "last":
                return
            ode_names = sorted(
                n for n in peft_model.peft_config.keys()
                if n.startswith("rung_") and n != "rung_flash"
            )
            if not ode_names:
                return
            target = ode_names[-1]
        try:
            peft_model.set_adapter(target)
            if re_arm:
                for n, p in peft_model.named_parameters():
                    if "lora_" in n:
                        p.requires_grad = True
        except Exception:
            pass

    @property
    def frame_seq_length(self) -> int:
        extra = int(getattr(self._inner_model(), "action_tokens_per_frame", 0))
        return self._spatial_frame_seq_length + extra

    @property
    def attn_window_frames(self) -> int:
        """The attention window in FRAMES (``local_attn_size``).

        ``-1`` means "no local attention" (attend the whole cache), and
        is also what we return when the attribute is a list/schedule or
        otherwise unreadable — i.e. "unknown, do not enforce". Read from
        the live model, so a ``local_attn_size_schedule`` transition
        (``trainer/causal_action_forcing_train.py::
        _apply_attn_size_if_changed``) is picked up on the next read.
        Step-derived, therefore identical on every rank.
        """
        las = getattr(self._inner_model(), "local_attn_size", -1)
        if las is None or isinstance(las, (list, tuple)):
            return -1
        try:
            return int(las)
        except (TypeError, ValueError):
            return -1

    @property
    def kv_cache_frames(self) -> int:
        """Cache size in FRAMES.

        Legacy (``kv_cache_seed_headroom=False``): the BASELINE rollout
        window with no headroom. LongLive's streaming pipeline sizes its
        cache as ``(local_attn + slice_last) * frame_seq_length`` to give
        the rolling cache headroom for seed + in-flight rollout
        simultaneously, but that doubles per-layer memory which OOMs on a
        32GB 5090.

        ``kv_cache_seed_headroom=True`` (14e alignment): the seed prefill
        is counted. The legacy formula ignored it entirely, so on the
        streaming DMD path the buffer held ``max(num_max, rollout)``
        frames while the sequence actually wrote
        ``seed + anchor_block + rollout`` — the cache silently ROLLED
        partway through the very first rollout, evicting the clean GT
        context the student was conditioned on. Size:

            seed_prefill_frames + max(num_max_frames, rollout_frames)
                                + num_frame_per_block

        The ``+ npb`` block is the same one-block headroom
        ``ode_rollout.py:236-237`` allocates; on the DMD streaming path
        it is exactly consumed by the leading anchor chunk that
        ``setup_sequence`` rolls between the seed and the first
        supervised chunk (``dmd_42f_gt_anchor``), so the final cache
        pointer lands on the last slot rather than past it.

        SIZING INVARIANT (2026-08-17), enforced on the seed-headroom
        path: ``kv_cache_frames >= local_attn_size + num_frame_per_block``
        — the buffer must always hold the whole attention window plus
        one block of headroom, so the model can actually attend the
        window it is configured for. The floor is applied ONLY when
        ``kv_cache_seed_headroom`` is on. On the legacy path the buffer
        is deliberately left equal to the window
        (``max(num_max_frames, rollout_frames)``, which the schedule
        keeps in lockstep with ``local_attn_size``) purely to avoid
        changing every legacy config's memory footprint.

        CORRECTION (2026-08-17): this note used to claim buffer == window
        is "the one sizing at which the block-relative rotation indices
        are CONTINUOUS across the cache-fill boundary". That was true of
        the OLD window-relative query anchor; ``utils/infinity_rope.py``
        now anchors the query buffer-relative
        (``num_cache_frames - num_new_frames``), so the offsets are
        continuous at ANY buffer depth and a deeper buffer introduces no
        jump. See ``_check_cache_contract``.
        """
        rollout = max(self.num_max_frames, self.rollout_frames)
        if not self.kv_cache_seed_headroom:
            return rollout
        frames = self.seed_prefill_frames + rollout + self.num_frame_per_block
        win = self.attn_window_frames
        if win > 0:
            frames = max(frames, win + self.num_frame_per_block)
        return frames

    @property
    def kv_cache_size(self) -> int:
        return self.kv_cache_frames * self.frame_seq_length

    def _check_cache_contract(self) -> None:
        """One-shot startup line describing the KV-cache steady state,
        plus one sizing sanity warning.

        Contract (block-relative / infinity RoPE, the single convention
        for both stationary and rolling generation): the attention
        window and its rotation slots stay fixed while data flows
        through them; the newest chunk takes the newest slot, older
        chunks shift back, and whatever falls out of the window is
        evicted. Eviction beyond the window is EXPECTED and legitimate —
        RoPE attention only sees the relative offset (i-j) and the
        student's attention was already capped at ``local_attn_size``
        frames, so an evicted frame is one it could not have attended to
        anyway.

        ROTATION CONTINUITY (corrected 2026-08-17). The earlier text here
        (and in the log line, and in the ``window > buffer`` raise below)
        asserted that offsets are continuous across the cache-fill
        boundary ONLY when ``kv_cache_frames == local_attn_size``, on the
        grounds that the query anchors at ``local_attn_size -
        num_new_frames``. That WAS true, and it is what motivated the
        raise — but ``utils/infinity_rope.py`` has since been fixed: the
        query anchor is now BUFFER-relative,

            q_start_idx = num_cache_frames - num_new_frames

        (``utils/infinity_rope.py``, "Q is anchored to the *buffer* frame
        count, exactly like K"), which is the same origin K is rotated
        against. Offsets are therefore continuous at ANY buffer depth,
        rolling or not, and a buffer deeper than the window is simply
        deeper — no shear, no jump. The `local_attn_size == buffer`
        coincidence is no longer load-bearing.

        WHY ``window > buffer`` IS NOW ONLY A WARNING. With a
        buffer-relative anchor, ``num_cache_frames`` is bounded by the
        buffer, so an over-large window cannot shift any RoPE offset; the
        attention slice ``temp_k[local_end - max_attention_size :
        local_end]`` just saturates at the whole buffer. The real
        consequence is a CAPABILITY shortfall, not corruption: the model
        is told it may attend ``local_attn_size`` frames while the buffer
        can only ever hold ``kv_cache_frames`` of them, so the effective
        context is silently capped at the buffer. That is worth saying
        out loud but it is not worth hard-failing a config that is
        otherwise correct — and the old raise would kill exactly such a
        config on a premise that no longer holds. Downgraded to a
        warning; the sizing facts are printed either way.
        """
        win = self.attn_window_frames
        cap = self.kv_cache_frames
        npb = self.num_frame_per_block
        if win > 0 and win > cap:
            import logging as _logging
            _logging.warning(
                "[ActionForcing][KV-CACHE] attention window "
                "local_attn_size=%d frames exceeds the KV buffer (%d "
                "frames; seed_prefill_frames=%d, num_max_frames=%d, "
                "rollout_frames=%d, npb=%d, seed_headroom=%s). Under the "
                "buffer-relative RoPE anchor this does NOT shift any "
                "query/key offset, but the effective context is silently "
                "capped at the buffer: the extra %d frame(s) of window "
                "can never be stored, so the model attends less history "
                "than the config advertises. Enlarge the cache "
                "(kv_cache_seed_headroom / rollout_frames) or shrink "
                "local_attn_size to make the two agree.",
                win, cap, self.seed_prefill_frames, self.num_max_frames,
                self.rollout_frames, npb, self.kv_cache_seed_headroom,
                win - cap,
            )
        key = (win, cap, npb, self.seed_prefill_frames)
        if key in self._cache_contract_logged:
            return
        self._cache_contract_logged.add(key)
        import logging as _logging
        if win > 0:
            when = (
                f"rolling begins once the sequence passes frame {cap} "
                f"(evicting {npb} frames per chunk); frames older than the "
                f"{win}-frame window are never attended to, so eviction "
                "beyond the window is expected and safe"
            )
            if cap == win:
                cont = "buffer == window"
            elif cap > win:
                cont = (
                    f"buffer > window by {cap - win} frame(s) of slack "
                    "(deliberate headroom)"
                )
            else:
                cont = (
                    f"buffer < window by {win - cap} frame(s): effective "
                    "context is capped at the buffer"
                )
            cont += (
                " | rotation offsets are continuous at any buffer depth "
                "(query anchored buffer-relative at num_cache_frames - "
                "num_new_frames)"
            )
        else:
            when = "local attention disabled (local_attn_size=-1); no rolling"
            cont = "n/a"
        _logging.info(
            "[ActionForcing][KV-CACHE] window=%s frames  buffer=%s frames  "
            "npb=%d  seed_prefill=%d (%s)  seed_headroom=%s | %s | %s",
            win, cap, npb, self.seed_prefill_frames, self.seed_prefill_mode,
            self.kv_cache_seed_headroom, when, cont,
        )

    def _check_cache_occupancy(self, end_frame: int, where: str) -> None:
        """Informational note the first time a call site's sequence runs
        past the end of the buffer.

        Downgraded from a warning (2026-08-17): under the block-relative
        RoPE contract eviction is the DESIGNED steady state, not a
        defect, so a per-call-site warning here was pure noise. The
        expected steady state is printed once by
        ``_check_cache_contract`` at cache-allocation time; this adds one
        INFO line naming the first call site that actually reaches the
        rolling regime.

        The window/buffer sizing mismatch is reported by
        ``_check_cache_contract``. It is re-checked here because
        ``local_attn_size`` can be mutated AFTER the buffers were
        allocated (``_apply_attn_size_if_changed``), and a warning cannot
        deadlock DDP even if some ranks' geometry differs.

        Never raises: ``end_frame`` is derived from per-rank ride
        geometry (``roll_cap`` depends on each rank's own ride length),
        so a raise here could fire on some ranks and not others.
        """
        cap = self.kv_cache_frames
        win = self.attn_window_frames
        if int(end_frame) <= cap:
            return
        import logging as _logging
        if win > 0 and win > cap and "undersized_buffer" not in self._cache_occupancy_warned:
            self._cache_occupancy_warned.add("undersized_buffer")
            import sys as _sys
            # Corrected 2026-08-17: this used to claim "queries anchor
            # past the last stored rotation slot, shifting every RoPE
            # offset". That premise died with the buffer-relative anchor
            # fix in ``utils/infinity_rope.py`` (q_start_idx =
            # num_cache_frames - num_new_frames, bounded by the buffer).
            # The real symptom is a truncated context, not a sheared one.
            msg = (
                f"[ActionForcing][KV-CACHE] {where}: attention window "
                f"({win} frames) is LARGER than the KV buffer ({cap} "
                f"frames), so the last {win - cap} frame(s) of window can "
                "never be stored and the effective context is capped at "
                "the buffer. RoPE offsets are NOT affected (the query "
                "anchor is buffer-relative), but the model attends less "
                "history than the config advertises — fix the sizing."
            )
            _logging.warning(msg)
            print(msg, file=_sys.stderr, flush=True)
        if "evict" in self._cache_occupancy_warned:
            return
        self._cache_occupancy_warned.add("evict")
        _logging.info(
            "[ActionForcing][KV-CACHE] %s: sequence reaches frame %d > "
            "buffer %d frames -> the cache rolls from here on (expected: "
            "only frames older than the %s-frame attention window are "
            "dropped).",
            where, int(end_frame), cap, win,
        )

    # -----------------------------------------------------------------
    # Lockstep exit-flag selection (rank 0 rolls, broadcast to all)
    # -----------------------------------------------------------------
    def generate_and_sync_list(
        self, num_blocks: int, num_denoising_steps: int, device: torch.device,
        sync: bool = True, force_exit_step: Optional[int] = None,
        exclude_last_rung: bool = False,
        low: int = 0,
    ) -> List[int]:
        """Pick a random exit step per rolling block.

        ``force_exit_step`` (highest priority): when not None, return
        ``[force_exit_step] * num_blocks`` — no NCCL traffic, no random
        sampling. Used by per-rank-divergent call sites (the slide-and
        -train helper) that pre-broadcast a single index ONCE before
        the loop and reuse it across slides; this keeps cross-rank
        lockstep on the exit rung while collapsing N per-slide
        broadcasts into 1 per training step.

        ``sync=True`` (default): rank 0 samples and broadcasts so all
        ranks pick the same exit rung. Required when the call count is
        matched across ranks (e.g. the warmup rollout, the per-iter
        generator step).

        ``sync=False``: each rank samples independently, no NCCL
        traffic. Per-rank-different exit flags add gradient variance
        but the per-rank backward + DDP all-reduce still converges.

        ``exclude_last_rung=True``: sample from ``[0, num_denoising_steps - 1)``
        so the last rung is reserved for a separate grad-active forward
        in the rollout's two-grad-point mode (Flash-DMD §3.3 — DMD
        grad at the random exit, GAN grad at the last rung). Mutually
        exclusive with ``last_step_only=True``.

        ``low=k>0``: shift the lower bound of the sample range so
        ``low <= sample < sample_high``. Used by warm-start init to
        skip the noisiest rung (rung 0 = pure-noise level) for chunks
        that get a warm-start seed; these chunks denoise from rung 1
        onward and must NOT exit at rung 0 (which they don't run).
        """
        if force_exit_step is not None:
            idx = int(force_exit_step)
            if not (0 <= idx < num_denoising_steps):
                raise ValueError(
                    f"force_exit_step={idx} out of range [0, "
                    f"{num_denoising_steps})"
                )
            if exclude_last_rung and idx == num_denoising_steps - 1:
                raise ValueError(
                    f"force_exit_step={idx} (last rung) is incompatible "
                    f"with exclude_last_rung=True."
                )
            if idx < low:
                raise ValueError(
                    f"force_exit_step={idx} below low={low}."
                )
            return [idx] * num_blocks

        if exclude_last_rung:
            if num_denoising_steps < 2:
                raise ValueError(
                    "exclude_last_rung=True requires num_denoising_steps>=2; "
                    f"got {num_denoising_steps}."
                )
            if self.last_step_only:
                raise ValueError(
                    "last_step_only=True is incompatible with "
                    "exclude_last_rung=True (the former forces the last "
                    "rung, the latter forbids it)."
                )
        sample_high = (num_denoising_steps - 1) if exclude_last_rung else num_denoising_steps
        if low < 0 or low >= sample_high:
            raise ValueError(
                f"low={low} out of range [0, sample_high={sample_high})."
            )

        # When ``exit_flag_weights`` is set on the pipeline, sample
        # weighted (multinomial) over the [low, sample_high) range
        # instead of uniform randint. Weights are sliced/padded to match
        # the active range so ``low`` and ``exclude_last_rung`` still
        # work correctly. ``last_step_only`` still short-circuits
        # everything (returns the last rung).
        weights: Optional[torch.Tensor] = None
        if self.exit_flag_weights is not None and not self.last_step_only:
            full_w = self.exit_flag_weights
            if len(full_w) != num_denoising_steps:
                raise ValueError(
                    f"exit_flag_weights has length {len(full_w)} but "
                    f"num_denoising_steps={num_denoising_steps}"
                )
            sliced = full_w[low:sample_high]
            if sum(sliced) <= 0:
                raise ValueError(
                    f"exit_flag_weights slice [{low}:{sample_high}] has "
                    f"non-positive sum: {sliced}"
                )
            weights = torch.tensor(
                sliced, device=device, dtype=torch.float32,
            )

        def _sample(n: int) -> torch.Tensor:
            if weights is not None:
                # multinomial with replacement → independent samples per
                # block. Returns indices in [0, len(weights)); shift by
                # ``low`` to get back into the original rung space.
                idx_local = torch.multinomial(
                    weights, num_samples=n, replacement=True,
                )
                return (idx_local + low).to(device=device, dtype=torch.long)
            return torch.randint(
                low=low, high=sample_high, size=(n,), device=device,
            )

        if sync:
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                indices = _sample(num_blocks)
                if self.last_step_only:
                    indices = torch.ones_like(indices) * (num_denoising_steps - 1)
            else:
                indices = torch.empty(num_blocks, dtype=torch.long, device=device)

            if dist.is_initialized():
                dist.broadcast(indices, src=0)
        else:
            indices = _sample(num_blocks)
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        return indices.tolist()

    # -----------------------------------------------------------------
    # Seed prefill helper (shared across all three call sites)
    # -----------------------------------------------------------------
    def _seed_prefill_chunk(
        self,
        seed_chunk: torch.Tensor,
        seed_block_cond: dict,
        current_start_frame: int,
    ) -> None:
        """KV-cache prefill one seed chunk through the model's denoise
        process. Replaces the legacy "raw GT at t=0" prefill, which put
        OOD attention traces into the cache (the model is trained on its
        own cache_pred outputs, not on raw GT, so a raw-GT t=0 forward
        wrote KV the model had never seen during training — producing a
        low-energy boundary state at the first generated chunk that
        recovered only as recency-biased attention shifted onto
        self-rolled cache_pred KV from subsequent chunks).

        New sequence (matches training rollouts' per-block KV pattern):
          1. Noise the seed at the cleanest training rung
             (``denoising_step_list[-1]``) and forward the generator —
             yields ``cache_pred`` = the model's x0 estimate of the seed,
             which sits in the same representation space as every
             training rollout's per-chunk cache_pred.
          2. Commit ``cache_pred`` at ``t=context_noise`` — the same
             commit forward the rollout's Step 3.4 uses. The KV cache
             now contains model-style x0 traces, eliminating the
             seed→rollout boundary OOD effect.

        All forwards run under ``no_grad``. The seed chunk itself
        (raw GT) remains written to the OUTPUT buffer at the seed
        positions by the caller — only the KV cache content changes.

        ``seed_prefill_mode="real"`` replaces the two forwards above
        with ONE forward of the REAL seed latents at ``t=0``. That is
        the contract the 14e ODE student was trained under
        (``action-forcing/af_model/ode_rollout.py:315-325``: "write the
        REAL seed into the cache, one chunk at a time, in order, at
        t=0"), and it is also what the teacher used to produce every
        LMDB target. The "estimate" rationale above was measured against
        the 14d student, which was teacher-forced with NO cache at all
        — for 14e the model-style-trace argument is inverted: its cache
        traces for CONTEXT frames are real-latent traces.
        """
        batch_size, npb = seed_chunk.shape[:2]
        device = seed_chunk.device
        # Phase-LoRA dispatch: both seed-prefill forwards run at low t
        # (denoising_step_list[-1] for the prediction; context_noise for
        # the commit), so route through the LAST ODE rung's adapter (=
        # K-1) — the rung specialized for the lowest-t bracket. This
        # is canonical for cold-start prefill at the bottom of the
        # denoising ladder. Also resets adapter state from any prior
        # rollout (e.g. rollout2's leftover rung_flash), so the next
        # grad-on rollout's first dispatch starts from a known state.
        num_rungs = len(self.denoising_step_list)
        prefill_step_idx = max(0, num_rungs - 1)
        self._check_cache_occupancy(
            current_start_frame + npb, "seed_prefill")

        if self.seed_prefill_mode == "real":
            # ODE parity: ONE forward, real latents, t=0, at the same
            # ``current_start`` the caller advances by ``npb``. No
            # noising, no estimate, no context_noise commit — the
            # student's trained contract has none of them for context
            # frames.
            with torch.no_grad():
                zero_t = torch.zeros(
                    [batch_size, npb], device=device, dtype=torch.int64,
                )
                self._maybe_set_phase_lora_for_step(
                    prefill_step_idx, num_rungs, re_arm=False)
                self.generator(
                    noisy_image_or_video=seed_chunk.detach(),
                    conditional_dict=seed_block_cond,
                    timestep=zero_t,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
            return

        with torch.no_grad():
            seed_t_int = int(round(float(self.denoising_step_list[-1])))
            seed_t = torch.full(
                [batch_size, npb], seed_t_int,
                device=device, dtype=torch.int64,
            )
            seed_chunk_flat = seed_chunk.detach().flatten(0, 1)
            seed_noised = self.scheduler.add_noise(
                seed_chunk_flat,
                torch.randn_like(seed_chunk_flat),
                seed_t.flatten(0, 1),
            ).unflatten(0, seed_chunk.shape[:2])
            self._maybe_set_phase_lora_for_step(prefill_step_idx, num_rungs, re_arm=False)
            _, cache_pred = self.generator(
                noisy_image_or_video=seed_noised,
                conditional_dict=seed_block_cond,
                timestep=seed_t,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )
            del seed_noised, seed_chunk_flat

            commit_input_clean = cache_pred.detach()
            del cache_pred
            ctx_t = torch.full_like(seed_t, self.context_noise)
            commit_input_flat = commit_input_clean.flatten(0, 1)
            if int(self.context_noise) == 0:
                # Third of three commit sites (see generate_chunk_with_cache
                # and inference_with_trajectory): add_noise at t=0 resolves to
                # the grid's min sigma 0.00498, not zero, so it noises clean
                # context. Dormant under seed_prefill_mode="real" (which
                # returns before this branch) but live for "estimate".
                cache_commit_input = commit_input_clean
            else:
                cache_commit_input = self.scheduler.add_noise(
                    commit_input_flat,
                    torch.randn_like(commit_input_flat),
                    ctx_t.flatten(0, 1),
                ).unflatten(0, commit_input_clean.shape[:2])
            del commit_input_clean, commit_input_flat
            self._maybe_set_phase_lora_for_step(prefill_step_idx, num_rungs, re_arm=False)
            self.generator(
                noisy_image_or_video=cache_commit_input,
                conditional_dict=seed_block_cond,
                timestep=ctx_t,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )
            del cache_commit_input

    # -----------------------------------------------------------------
    # Main entry: inference_with_trajectory
    # -----------------------------------------------------------------
    def inference_with_trajectory(
        self,
        noise: torch.Tensor,
        clean_image_or_video: Optional[torch.Tensor] = None,
        initial_latent: Optional[torch.Tensor] = None,
        gt_latents: Optional[torch.Tensor] = None,
        return_sim_step: bool = False,
        seed_latents: Optional[torch.Tensor] = None,
        prefer_cache_pred_in_output: bool = False,
        requires_grad: bool = True,
        flash_dmd_enabled: bool = False,
        flash_dmd_gan_t: int = 60,
        **conditional_dict,
    ) -> Tuple[torch.Tensor, Optional[int], Optional[int]]:
        """Run the Action-Forcing chunkwise rollout.

        Args:
            noise: ``[B, rollout_frames, C, H, W]`` initial noise.
            clean_image_or_video: unused (kept for API compatibility).
            initial_latent: optional i2v anchor (1-frame). Mutually
                exclusive with ``seed_latents``.
            gt_latents: ``[B, F_gt, C, H, W]`` ground-truth latents
                covering the BASELINE rollout AND any frames the MAE-
                extension loop might roll into (legacy; unused now
                that the MAE-extension path is removed). Pass
                ``None`` to skip MAE metric collection.
            return_sim_step: if True, also return the exit step index
                (legacy interface, unused by Phase-1 Action-Forcing).
            seed_latents: ``[B, cf, C, H, W]`` clean GT latents to
                pre-populate the KV cache before the actual rollout
                starts. Run through the generator at ``timestep=0``
                in chunks of ``num_frame_per_block`` frames each, with
                a context-noise commit forward per chunk (mirrors the
                regular rolling cache-update pattern). The seed frames
                are NOT included in the returned ``output`` — the
                caller's scoring window is unaffected. Mutually
                exclusive with ``initial_latent``.
            conditional_dict: per-frame action streams + prompt embed.
                When ``seed_latents`` is provided the streams must
                cover ``cf + rollout_frames`` frames (seed first,
                then rollout). Otherwise must cover at least
                ``rollout_frames``.

        Returns:
            ``(output, denoised_timestep_from, denoised_timestep_to)``
            where ``output`` is shape ``[B, rollout_frames, C, H, W]``.
        """
        if seed_latents is not None and initial_latent is not None:
            raise ValueError(
                "seed_latents and initial_latent are mutually exclusive; "
                "Phase-1 Action-Forcing uses seed_latents (cf-frame KV "
                "prefill), the i2v initial_latent path is legacy."
            )
        self._last_extension_metrics = {}
        # Per-channel stat imposition target: set BEFORE any student forward
        # in this rollout, from the seed window the student is conditioned
        # on (the imposition rescales every pred_x0 toward these per-channel
        # seed stats). No-op when imposition is disabled on the generator.
        # Falls back to ``initial_latent`` for the legacy i2v path; if
        # neither is present the generator retains its previous target (and
        # warns once if it has none). This is the single place the target is
        # bound for the action-forcing rollout, so it follows the student
        # through every forward below.
        if hasattr(self.generator, "set_impose_stat_target"):
            _impose_seed = seed_latents if seed_latents is not None \
                else initial_latent
            self.generator.set_impose_stat_target(_impose_seed)
        # Reset per-call Flash-DMD t=flash_dmd_gan_t output. Populated
        # below when ``flash_dmd_enabled=True``; remains None otherwise
        # so callers can fail-fast if they expect it but the rollout
        # wasn't run with Flash DMD active.
        self._flash_dmd_gan_output: Optional[torch.Tensor] = None
        # Per-block CLEAN PRED buffer for the aux teacher's clean_x.
        # Populated post-Step-3.3.5 (= refined cache_pred when flash
        # is enabled; post-rung cache_pred otherwise). Detached,
        # graph-free. Read by ``_compute_aux_teacher_loss_streaming``
        # to assemble the LoRA's clean half as
        # ``cat(GT_seed_last, refined_cache_pred[: chunk_size-npb])``.
        # Sized to the rollout window (post-slice); ``None`` when
        # ``flash_dmd_enabled=False`` so the aux pass falls back to
        # the legacy ``_streaming_build_clean_x_self`` path.
        self._clean_chunk: Optional[torch.Tensor] = None
        # A23: this path never builds the grad twin, so clear any stale
        # buffer left by a previous ``generate_chunk_with_cache`` call
        # rather than letting a consumer read a graph from a different
        # rollout. (No behavioural change: nothing reads this attribute
        # unless the A23 gate is on.)
        self._clean_chunk_grad: Optional[torch.Tensor] = None
        self._clean_chunk_grad_mask: Optional[torch.Tensor] = None
        # Reset per-call last-block clean pred (= cache_pred at the
        batch_size, num_frames, num_channels, height, width = noise.shape

        # No independent first frame in our setup.
        npb = self.num_frame_per_block
        cps = self.chunks_per_rolling_step
        block_frames = npb * cps

        if num_frames % npb != 0:
            raise RuntimeError(
                f"num_frames ({num_frames}) must be divisible by "
                f"num_frame_per_block ({npb})."
            )
        num_blocks_total = num_frames // npb
        # Group blocks into rolling steps. If chunks_per_rolling_step does not
        # divide num_blocks_total evenly, the trailing rolling step processes
        # fewer chunks (last_step_remainder). E.g. cps=1, num_blocks_total=7
        # (CF-parity default) gives uniform sizes [3, 3, 3, 3, 3, 3, 3];
        # cps=2, num_blocks_total=7 would give [6, 6, 6, 3].
        all_num_frames: List[int] = []
        remaining_blocks = num_blocks_total
        while remaining_blocks > 0:
            chunks_this_step = min(cps, remaining_blocks)
            all_num_frames.append(chunks_this_step * npb)
            remaining_blocks -= chunks_this_step

        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_seed_frames = seed_latents.shape[1] if seed_latents is not None else 0
        # Output covers seed + rollout so we can write to ``output[:,
        # current_start_frame:...]`` using ABSOLUTE indices that match
        # the KV-cache positions. The seed slice is sliced off before
        # return so the caller's scoring window is unchanged.
        num_output_frames = num_frames + num_input_frames + num_seed_frames
        self._check_cache_occupancy(
            num_output_frames, "inference_with_trajectory")
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype,
        )
        if num_seed_frames > 0:
            output[:, num_input_frames: num_input_frames + num_seed_frames] = seed_latents

        # Step 1: Initialize KV / crossattn caches.
        self._initialize_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )

        # Step 2: Cache initial latent if provided (i2v path; unused for us).
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.zeros(
                [batch_size, 1], device=noise.device, dtype=torch.int64
            )
            output[:, :1] = initial_latent
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=_slice_per_frame_streams(
                        conditional_dict,
                        frame_start=current_start_frame,
                        frame_count=1,
                    ),
                    timestep=timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
            current_start_frame += 1

        # Step 2b: KV-cache prefill from seed_latents. The seed is the
        # leading ``cf`` frames of the ride (= dmd_context_clean_frames),
        # = clean GT context the ODE student was trained to see as
        # ``clean_x``. Without this prefill the rolling rollout starts
        # cold and produces garbage. We seed in chunks of ``npb`` frames
        # each via ``_seed_prefill_chunk``, which runs the model's
        # denoise→commit sequence on each seed chunk so the KV cache
        # ends up in the SAME representation regime as the rollout's
        # per-block commits (= model-style cache_pred, not raw GT). See
        # ``_seed_prefill_chunk`` for the rationale.
        if seed_latents is not None:
            cf = int(seed_latents.shape[1])
            if cf <= 0:
                raise ValueError(f"seed_latents must have >= 1 frame; got cf={cf}")
            if cf % npb != 0:
                raise ValueError(
                    f"seed_latents has {cf} frames; must be a multiple of "
                    f"num_frame_per_block ({npb})."
                )
            num_seed_chunks = cf // npb
            for sc in range(num_seed_chunks):
                seed_start = sc * npb
                seed_chunk = seed_latents[:, seed_start: seed_start + npb]
                seed_block_cond = _slice_per_frame_streams(
                    conditional_dict,
                    frame_start=current_start_frame,
                    frame_count=npb,
                )
                self._seed_prefill_chunk(
                    seed_chunk=seed_chunk,
                    seed_block_cond=seed_block_cond,
                    current_start_frame=current_start_frame,
                )
                current_start_frame += npb

        # Step 3: Per-rolling-step denoise loop with truncated random-exit.
        num_denoising_steps = len(self.denoising_step_list)
        # ``flash_dmd_enabled=True``: every block adds ONE extra graph-on
        # gen forward at ``flash_dmd_gan_t`` (default 60, raw post-warp
        # timestep) AFTER the standard denoise chain finishes. The extra
        # forward's output (= ``flash_dmd_gan_pred`` per block, assembled
        # into ``flash_dmd_gan_output``) is consumed by the GAN adv loss
        # and the gen-side aux losses (LPIPS / MS-SSIM /
        # action_critic). DMD scoring continues to use the random-exit-
        # rung output (= ``denoised_pred``). The K/V slots written by
        # the flash_dmd forward are overwritten by Step 3.4's context-
        # noise commit so the next block's exit-rung forward reads
        # graph-free K/V (paper §3.3 cross-timestep decoupling).
        # The random exit pool now spans ALL rungs (including the last)
        # since GAN no longer reserves the last rung.
        # Cold-start only. Every block uses the full ladder (low=0).
        exit_flags = self.generate_and_sync_list(
            len(all_num_frames), num_denoising_steps, device=noise.device,
            exclude_last_rung=False,
        )
        # Buffer for accumulating Flash-DMD t=flash_dmd_gan_t grad-active
        # outputs across blocks (one [B, current_num_frames, C, H, W]
        # slab per block, zero-init for warmup blocks where the forward
        # stays no_grad). Sized to num_output_frames so block writes
        # use absolute current_start_frame indexing, mirroring ``output``.
        if flash_dmd_enabled:
            flash_dmd_gan_output = torch.zeros(
                [batch_size, num_output_frames, num_channels, height, width],
                device=noise.device,
                dtype=noise.dtype,
            )
        else:
            flash_dmd_gan_output = None
        # Aux-teacher clean_x buffer: per-block post-Step-3.3.5
        # cache_pred. Detached, no autograd graph. Sized to
        # num_output_frames so block writes use absolute
        # current_start_frame indexing. Allocated UNCONDITIONALLY so
        # downstream consumers (aux teacher, fake_alt_head) always
        # receive the cleanest available x0 estimate:
        #   * flash_dmd_enabled=True  → t=flash_dmd_gan_t refined
        #     ``cache_pred`` (Step 3.2.b reassigns ``cache_pred`` to
        #     the t=60 grad-on output)
        #   * flash_dmd_enabled=False → t=denoising_step_list[-1]
        #     (~178.6) finish-denoised ``cache_pred`` from Step 3.2
        # Either way the per-block stash at the end of the rollout
        # loop fills this buffer with the cleanest x0 estimate we have
        # without spending an extra forward.
        clean_chunk = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype,
        )
        # CF-parity #11: gradient-window gate. CF hardcodes a literal
        # 21 here (``Causal-Forcing/pipeline/self_forcing_training.py:
        # 120``: ``start_gradient_frame_index = num_output_frames - 21``)
        # because CF's design pins the SCORING window at exactly 21
        # latents. We pin our scoring window at ``num_max_frames``
        # (default 21, matching CF) and parameterise the rollout
        # length via ``rollout_frames`` (= ``num_output_frames``
        # here). When ``rollout_frames == num_max_frames`` (default),
        # the gate gives ``start_gradient_frame_index = 0`` and every
        # block's exit-flag forward fires WITH grad. When
        # ``rollout_frames > num_max_frames`` (long-rollout mode), the
        # leading ``rollout_frames - num_max_frames`` frames warm the
        # cache via no-grad exit-flag forwards and only the trailing
        # ``num_max_frames`` frames backprop — byte-for-byte CF
        # behavior, generalised. This matches the user's contract:
        # "rollouts longer than num_training_frames, train on the last
        # num_training_frames only."
        start_gradient_frame_index = num_output_frames - self.num_max_frames
        # ``requires_grad=False`` (LongLive parity: the critic step
        # passes False so the rollout produces no autograd graph at
        # all). Setting the gradient-start index past the end of the
        # rollout disables the grad-active branch in the exit-flag
        # forward — every block stays in the ``with torch.no_grad():``
        # path. Also redundant with the trainer wrapping the critic-
        # path call in an outer ``no_grad`` context, but kept
        # explicit for self-documenting behaviour.
        if not requires_grad:
            start_gradient_frame_index = num_output_frames + 1

        denoised_pred = None
        timestep = None
        for block_index, current_num_frames in enumerate(all_num_frames):
            # Slice the noisy input + per-frame conditioning for this block.
            # ``noise`` covers ROLLOUT frames only (no seed, no i2v anchor),
            # so we offset the absolute current_start_frame back to the
            # noise tensor's 0-based index by subtracting both prefixes.
            block_start_in_noise = (
                current_start_frame - num_input_frames - num_seed_frames
            )
            # Cold-start always: pure noise + full denoising ladder.
            noisy_input = noise[
                :,
                block_start_in_noise: block_start_in_noise + current_num_frames,
            ]
            block_cond = _slice_per_frame_streams(
                conditional_dict,
                frame_start=current_start_frame,
                frame_count=current_num_frames,
            )

            # Step 3.1: Truncated denoise loop over the full ladder.
            for index in range(0, num_denoising_steps):
                current_timestep = self.denoising_step_list[index]
                # Phase-LoRA dispatch: select which student adapter
                # contributes to this rung's forward (no-op when phase
                # LoRA is not wrapped). Must run BEFORE the generator
                # call below.
                self._maybe_set_phase_lora_for_step(index, num_denoising_steps)
                if self.same_step_across_blocks:
                    exit_flag = (index == exit_flags[0])
                else:
                    exit_flag = (index == exit_flags[block_index])

                ts_value = int(round(float(current_timestep)))
                timestep = torch.full(
                    [batch_size, current_num_frames],
                    ts_value,
                    device=noise.device,
                    dtype=torch.int64,
                )

                if not exit_flag:
                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=block_cond,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                        )
                        next_t_value = int(round(float(
                            self.denoising_step_list[index + 1]
                        )))
                        flat = denoised_pred.flatten(0, 1)
                        noisy_input = self.scheduler.add_noise(
                            flat,
                            torch.randn_like(flat),
                            next_t_value
                            * torch.ones(
                                [batch_size * current_num_frames],
                                device=noise.device,
                                dtype=torch.long,
                            ),
                        ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # Skip the grad-active random-exit rung forward on
                    # the LAST block of MULTI-BLOCK calls. The DMD
                    # scoring mask zeros the trailing
                    # ``num_frame_per_block`` frames structurally (v14
                    # joint-TF OOD region) — so the last block's grad
                    # activations are pure waste. ``is_multi_block``
                    # gate is critical: in single-block calls (e.g.
                    # streaming iter k>=2 with ``new_frames=npb``),
                    # the only block IS marked last; skipping its grad
                    # would detach the entire rollout output and break
                    # the gen backward. See
                    # ``generate_chunk_with_cache`` for the full
                    # rationale.
                    is_last_block = (block_index == len(all_num_frames) - 1)
                    is_multi_block = (len(all_num_frames) > 1)
                    skip_last_block_grad = is_last_block and is_multi_block
                    if current_start_frame < start_gradient_frame_index or skip_last_block_grad:
                        with torch.no_grad():
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=block_cond,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length,
                            )
                    else:
                        # Activation-checkpoint the grad-active random-
                        # exit rung forward. This is the dominant
                        # held-activation source per block (~30 layer-
                        # inputs at 14 MB each = ~420 MB per block,
                        # ~3 GB across 7 blocks). Same default-args
                        # closure trick as the flash-DMD ckpt to capture
                        # by value; same cache-safety story (per-block
                        # K/V slots get overwritten by Step 3.4
                        # context-noise commits and stay stable until
                        # next rollout reset).
                        #
                        # Phase-LoRA dispatch MUST happen INSIDE the
                        # checkpoint function so backward-time recompute
                        # (use_reentrant=False re-runs forward) re-sets
                        # the same active adapter as the original
                        # forward. Without this, if set_adapter is called
                        # between original forward (e.g. for the flash
                        # forward in the K+1 design) and backward, the
                        # recompute would use the wrong active adapter —
                        # gradient flows to the wrong rung's lora
                        # matrices, manifesting as DDP's "marked ready
                        # twice" error.
                        def _exit_fn(
                            x,
                            _self=self,
                            _gen=self.generator,
                            _cond=block_cond,
                            _t=timestep,
                            _kv=self.kv_cache1,
                            _xa=self.crossattn_cache,
                            _start=current_start_frame * self.frame_seq_length,
                            _idx=index,
                            _ns=num_denoising_steps,
                        ):
                            _self._maybe_set_phase_lora_for_step(_idx, _ns)
                            return _gen(
                                noisy_image_or_video=x,
                                conditional_dict=_cond,
                                timestep=_t,
                                kv_cache=_kv,
                                crossattn_cache=_xa,
                                current_start=_start,
                            )

                        _, denoised_pred = _ckpt(
                            _exit_fn, noisy_input, use_reentrant=False,
                        )
                    exit_index = index
                    break

            # Step 3.2: Finish the denoising chain past the random exit
            # rung — fully no_grad. ``cache_pred`` ends up at the last
            # rung's clean x0 estimate. ``denoised_pred`` (the random
            # exit-rung's grad-active output) is what DMD scoring
            # consumes; it's preserved unchanged.
            cache_pred = denoised_pred.detach()
            num_rungs = len(self.denoising_step_list)
            for j in range(exit_index + 1, num_rungs):
                # Phase-LoRA dispatch for the post-exit no_grad
                # finish-denoise step: each j is a distinct ODE rung,
                # so route through that rung's adapter even though
                # the forward is no_grad. Without this, all post-exit
                # forwards inherit the exit_index rung's adapter and
                # write KV traces that won't match the next chunk's
                # rung-X forward — subtle quality drift over a
                # multi-chunk rollout.
                self._maybe_set_phase_lora_for_step(j, num_rungs, re_arm=False)
                next_t_value = int(round(float(
                    self.denoising_step_list[j]
                )))
                flat = cache_pred.flatten(0, 1)
                cache_input = self.scheduler.add_noise(
                    flat,
                    torch.randn_like(flat),
                    next_t_value
                    * torch.ones(
                        [batch_size * current_num_frames],
                        device=noise.device,
                        dtype=torch.long,
                    ),
                ).unflatten(0, denoised_pred.shape[:2])
                step_t = torch.full_like(timestep, next_t_value)
                with torch.no_grad():
                    _, cache_pred = self.generator(
                        noisy_image_or_video=cache_input,
                        conditional_dict=block_cond,
                        timestep=step_t,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )

            # Step 3.2.b: Flash-DMD t=flash_dmd_gan_t grad-on forward.
            # When ``flash_dmd_enabled``, take the post-chain clean x0
            # (= ``cache_pred``, no_grad), noise to ``flash_dmd_gan_t``,
            # forward graph-on (in the gradient-active window) or
            # no_grad (warmup blocks). The output's gradient flows ONLY
            # through this forward's gen weights — the input is detached
            # upstream. Step 3.4 below overwrites the K/V slots with
            # no_grad context-noise K/V so the NEXT block's exit-rung
            # forward reads graph-free K/V (paper §3.3 cross-timestep
            # decoupling). Fires on EVERY block in the grad window
            # (no final-block restriction).
            flash_dmd_pred: Optional[torch.Tensor] = None
            if flash_dmd_enabled:
                # Phase-LoRA dispatch for the flash forward. No-op
                # when phase LoRA isn't wrapped, or when env var
                # FLASH_PHASE_LORA_ROUTE is unset/"exit" (preserves
                # the historical uniform-across-rung gradient
                # distribution). Set FLASH_PHASE_LORA_ROUTE=last to
                # route flash GAN gradient to rung_{K-1} only.
                self._maybe_set_phase_lora_for_flash()
                flash_t_value = int(flash_dmd_gan_t)
                flash_flat = cache_pred.flatten(0, 1)
                flash_input = self.scheduler.add_noise(
                    flash_flat,
                    torch.randn_like(flash_flat),
                    flash_t_value * torch.ones(
                        [batch_size * current_num_frames],
                        device=noise.device, dtype=torch.long,
                    ),
                ).unflatten(0, cache_pred.shape[:2])
                flash_t_step = torch.full_like(timestep, flash_t_value)
                # Skip the grad-active flash-DMD forward on the LAST
                # block of MULTI-BLOCK calls. Matches the streaming
                # path's gating in ``generate_chunk_with_cache`` so
                # both functions agree on what "skip last chunk"
                # means: drop only the trailing block of multi-block
                # rollouts (= iter 1 in streaming; full-rollout calls
                # here). Single-block calls keep their flash forward
                # grad-active. Saves the trailing block's per-block
                # transformer activations.
                is_last_block = (block_index == len(all_num_frames) - 1)
                is_multi_block = (len(all_num_frames) > 1)
                flash_grad_active = (
                    requires_grad
                    and current_start_frame >= start_gradient_frame_index
                    and not (is_last_block and is_multi_block)
                )
                if flash_grad_active:
                    # Activation-checkpoint the per-block grad-on
                    # flash-DMD forward. Saves ~9 GB peak by
                    # recomputing the forward in backward instead of
                    # holding all 7 blocks' transformer activations
                    # in flight. Cost: +15-20% wallclock per gen step.
                    #
                    # Closure capture via default args: ``block_cond``
                    # / ``flash_t_step`` / ``current_start_frame`` are
                    # loop-local and rebind each iter. Without the
                    # default-arg trick, every checkpoint's recompute
                    # would use the FINAL iter's bindings at backward
                    # time. Default-arg evaluation captures by value
                    # at function-definition time → each iter's
                    # closure is correctly pinned to that iter's
                    # values. ``self.kv_cache1`` /
                    # ``self.crossattn_cache`` are stable references
                    # to the same dict objects across the rollout
                    # (their contents mutate, but the references
                    # don't), so accessing them via ``self`` is safe.
                    #
                    # Cache safety: chunk K's flash forward reads
                    # prior chunks' K/V (slots 1..K-1). Those slots
                    # are committed to context_noise K/V by their
                    # Step 3.4 (no_grad, not checkpointed) and never
                    # subsequently overwritten. So at backward-time
                    # recompute, the prior-slot K/V state is
                    # bit-identical to the original-forward state.
                    # ✓ Mathematically clean.
                    # Phase-LoRA dispatch INSIDE the ckpt fn so
                    # backward-time recompute (use_reentrant=False
                    # re-runs forward) also routes to the same active
                    # adapter (rung_flash in K+1 design). Otherwise the
                    # recompute would inherit whichever adapter was set
                    # by the most-recent external call, sending the
                    # gradient to the wrong rung's lora matrices —
                    # manifests as DDP's "marked ready twice" error.
                    def _flash_fn(
                        x,
                        _self=self,
                        _gen=self.generator,
                        _cond=block_cond,
                        _t=flash_t_step,
                        _kv=self.kv_cache1,
                        _xa=self.crossattn_cache,
                        _start=current_start_frame * self.frame_seq_length,
                    ):
                        _self._maybe_set_phase_lora_for_flash()
                        return _gen(
                            noisy_image_or_video=x,
                            conditional_dict=_cond,
                            timestep=_t,
                            kv_cache=_kv,
                            crossattn_cache=_xa,
                            current_start=_start,
                        )

                    _, flash_dmd_pred = _ckpt(
                        _flash_fn, flash_input, use_reentrant=False,
                    )
                else:
                    # Phase-LoRA dispatch for the no_grad flash branch
                    # (warmup / final-block-of-multi-block skip). Must
                    # dispatch even no_grad so the rung_flash adapter is
                    # the one running through the model — without this,
                    # the flash forward inherits whatever adapter the
                    # post-exit no_grad loop left active.
                    # re_arm=False: this is a no_grad path; skipping
                    # the requires_grad re-arm reduces DDP-state churn
                    # (Fix B for the unmarked_param_indices crash).
                    self._maybe_set_phase_lora_for_flash(re_arm=False)
                    with torch.no_grad():
                        _, flash_dmd_pred = self.generator(
                            noisy_image_or_video=flash_input,
                            conditional_dict=block_cond,
                            timestep=flash_t_step,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                        )

            # Step 3.3: Record the model's output. By default this is
            # the grad-active ``denoised_pred`` (x0 at the random exit
            # rung) so DMD's gradient flows through. When
            # ``prefer_cache_pred_in_output=True`` (visualization-only
            # mode used by the diagnostic test) we write the fully-
            # denoised ``cache_pred`` instead, so the output video
            # shows what the cache K/V was built from rather than the
            # noisy exit-rung x0 estimate. This is INVALID for training
            # (no gradient signal in cache_pred) and only intended for
            # diagnosing the no_grad-finish-denoise behaviour.
            output_pred = (
                cache_pred.to(denoised_pred.dtype)
                if prefer_cache_pred_in_output
                else denoised_pred
            )
            output[
                :,
                current_start_frame: current_start_frame + current_num_frames,
            ] = output_pred

            # Step 3.2.b + 3.3.5 UNIFIED: one t=60 forward (Step 3.2.b
            # above) serves both purposes — its grad-on output goes to
            # the GAN's adv path AND becomes the new ``cache_pred`` for
            # downstream consumption (Step 3.4 commit, clean_chunk
            # stash, etc.). The previously-separate no_grad t=60
            # refinement (old Step 3.3.5) is removed: the flash forward
            # already computed at t=60, re-running it would be a wasted
            # forward. We detach when assigning to ``cache_pred`` so the
            # cache K/V commit below doesn't carry the gen's autograd
            # graph into the cache state.
            if flash_dmd_enabled:
                if flash_dmd_pred is None:
                    raise RuntimeError(
                        "flash_dmd_enabled=True but Step 3.2.b did not "
                        "produce flash_dmd_pred."
                    )
                flash_dmd_gan_output[
                    :,
                    current_start_frame: current_start_frame + current_num_frames,
                ] = flash_dmd_pred
                # Unification: cache_pred becomes the t=60 refined
                # output (detached), avoiding the second forward.
                cache_pred = flash_dmd_pred.detach()

            # Stash the post-Step-3.3.5 ``cache_pred`` (refined when
            # flash_dmd_enabled, post-rung otherwise) into the per-
            # rollout ``clean_chunk`` buffer for the aux teacher's
            # clean half. Detached — the aux pass is the LoRA's
            # training step; gradient must NOT flow back through this
            # tensor into the student.
            if clean_chunk is not None:
                clean_chunk[
                    :,
                    current_start_frame: current_start_frame + current_num_frames,
                ] = cache_pred.detach()

            # Step 3.4: Cache-update forward at t=context_noise. Runs
            # in BOTH modes (flash_dmd_enabled or not):
            #   * Standard: committing the fully-denoised cache_pred at
            #     context_noise gives the rolling cache a clean x0
            #     estimate (no noise compounding for downstream chunks).
            #   * Flash-DMD: MANDATORY for paper §3.3 K/V decoupling —
            #     overwrites the grad-attached K/V slots written by
            #     the t=flash_dmd_gan_t grad-on forward with graph-free
            #     K/V at t=context_noise so the next block's exit-rung
            #     (DMD-grad) forward reads detached K/V. ``cache_pred``
            #     here is the t=60 refined output (reassigned at the
            #     ``flash_dmd_enabled`` branch above) — the KV cache
            #     therefore carries the cleaner t=60 x0 estimate forward.
            # The input is always detached (cache_pred came from a
            # no_grad chain anyway, but the explicit detach releases
            # any autograd nodes early — memory hygiene).
            commit_input_clean = cache_pred.detach()
            # CARN seam affine (latent CARN v0): gated per-channel mean/std
            # re-anchor of the committed context toward the ride seed's latent
            # stats -- counters the AR drift walk (DC/color haze, contraction)
            # INSIDE the loop; affine correction measured ~80% sufficient in
            # the contraction-law study. lambda=0 (default) = byte-identical.
            # Temperature scaling (carn_seam_temp, default 1.0 = off): scale
            # per-channel deviations of the committed context by T to counter
            # the per-chunk variance contraction (k~0.917 for the 4-rung
            # sampler => T ~ 1/k). Pure re-inflation, no target stats needed;
            # composes with (runs before) the affine re-anchor below.
            _tT = float(getattr(self, "carn_seam_temp", 1.0) or 1.0)
            if abs(_tT - 1.0) > 1e-6:
                _x = commit_input_clean.float()
                _mu = _x.mean(dim=(0, 1, 3, 4), keepdim=True)
                commit_input_clean = (
                    _mu + _tT * (_x - _mu)
                ).to(commit_input_clean.dtype)
            # Global drift counter-bias (CARN repulsor v1): subtract
            # lambda_d * d_hat (per-channel-mean drift/roll, fitted from the
            # frozen-model drift probe -- transition cos +0.53, ride cos
            # +0.74) from every committed chunk. mu-only by design (the
            # grep-sign study: the mu-half is right-signed; sigma-half is
            # handled by temp/affine instead).
            _dl = float(getattr(self, "carn_seam_drift_lambda", 0.0) or 0.0)
            _dv = getattr(self, "_carn_seam_drift_vec", None)
            if _dl > 0.0 and _dv is not None:
                commit_input_clean = (
                    commit_input_clean.float()
                    - _dl * _dv.view(1, 1, -1, 1, 1).to(
                        commit_input_clean.device)
                ).to(commit_input_clean.dtype)
            _cl = float(getattr(self, "carn_seam_affine_lambda", 0.0) or 0.0)
            _ct = getattr(self, "_carn_seam_target", None)
            if _cl > 0.0 and _ct is not None:
                _tm, _tsd = _ct
                _x = commit_input_clean.float()
                _mu = _x.mean(dim=(0, 1, 3, 4), keepdim=True)
                _sd = _x.std(dim=(0, 1, 3, 4), keepdim=True).clamp_min(1e-4)
                _tm_ = _tm.view(1, 1, -1, 1, 1).to(_x)
                _tsd_ = _tsd.view(1, 1, -1, 1, 1).to(_x)
                commit_input_clean = (
                    (_x - _mu) / _sd * (_cl * _tsd_ + (1.0 - _cl) * _sd)
                    + (_cl * _tm_ + (1.0 - _cl) * _mu)
                ).to(commit_input_clean.dtype)
            context_timestep = torch.full_like(timestep, self.context_noise)
            if int(self.context_noise) == 0:
                # See the matching note in generate_chunk_with_cache: the
                # FlowMatchScheduler grid has NO t=0 entry (min sigma
                # 0.00498), so add_noise at t=0 silently writes
                # 0.995*pred + 0.005*eps into what is supposed to be clean
                # context. Pass it through verbatim, as ode_rollout.py does.
                cache_commit_input = commit_input_clean
            else:
                cache_commit_input = self.scheduler.add_noise(
                    commit_input_clean.flatten(0, 1),
                    torch.randn_like(commit_input_clean.flatten(0, 1)),
                    context_timestep.flatten(0, 1),
                ).unflatten(0, commit_input_clean.shape[:2])
            # Phase-LoRA dispatch for the context_noise commit forward.
            # This forward writes the KV cache at t=context_noise (~0)
            # — same logical position as _seed_prefill_chunk's commit
            # forward, so route through the lowest-t ODE rung (= K-1).
            # Without this, the commit inherits whichever adapter the
            # preceding flash forward left active (e.g. rung_flash),
            # writing KV that doesn't correspond to the canonical low-t
            # ODE rung's output distribution.
            self._maybe_set_phase_lora_for_step(
                max(0, len(self.denoising_step_list) - 1),
                len(self.denoising_step_list),
                re_arm=False,
            )
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=cache_commit_input,
                    conditional_dict=block_cond,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )

            current_start_frame += current_num_frames

        # Step 3.5: Engineering trick — derive the timestep range we
        # supervised in this rollout (used by DMD's ts_schedule clamp).
        # All blocks cold-start now (warm-start removed) so just report
        # block 0's exit rung.
        denoised_timestep_from: Optional[int]
        denoised_timestep_to: Optional[int]
        ts_block_idx = 0
        if not self.same_step_across_blocks:
            denoised_timestep_from = None
            denoised_timestep_to = None
        elif exit_flags[ts_block_idx] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = self._round_to_grid(
                self.denoising_step_list[exit_flags[ts_block_idx]]
            )
        else:
            denoised_timestep_to = self._round_to_grid(
                self.denoising_step_list[exit_flags[ts_block_idx] + 1]
            )
            denoised_timestep_from = self._round_to_grid(
                self.denoising_step_list[exit_flags[ts_block_idx]]
            )

        # Slice off the seed prefix (KV prefill window) before
        # returning. The caller's scoring window is the rollout-only
        # region, frames [num_input_frames + num_seed_frames :].
        if num_seed_frames > 0 or num_input_frames > 0:
            output = output[:, num_input_frames + num_seed_frames:]
            if flash_dmd_gan_output is not None:
                flash_dmd_gan_output = flash_dmd_gan_output[
                    :, num_input_frames + num_seed_frames:
                ]
            if clean_chunk is not None:
                clean_chunk = clean_chunk[
                    :, num_input_frames + num_seed_frames:
                ]

        # Stash the Flash-DMD t=flash_dmd_gan_t output on the pipeline
        # instance so the model can read it without a return-tuple
        # signature change. After Step 3.2.b/3.3.5 unification, this is
        # the SINGLE t=60 grad-on forward's output used by both the GAN
        # adv path AND consumers that want a cleaner-than-random-rung
        # chunk (FN training, eval video logging, KV cache state via
        # the Step 3.4 context-noise commit). ``None`` when
        # ``flash_dmd_enabled=False``.
        self._flash_dmd_gan_output = flash_dmd_gan_output
        # Same stash for the aux-teacher clean_chunk buffer.
        self._clean_chunk = clean_chunk

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1
        return output, denoised_timestep_from, denoised_timestep_to

    def _round_to_grid(self, t_value) -> int:
        ts = self.scheduler.timesteps
        if ts.is_cuda:
            grid = ts
        else:
            grid = ts
        target = float(t_value)
        # Mirror CF's "1000 - argmin(|grid - target|)" map. For Wan's
        # FlowMatchScheduler the timesteps are stored ordered, and
        # 1000 - argmin index gives the *time* (since timesteps[0]=999
        # maps to t=1000 etc. in CF's convention).
        diffs = (grid.float() - target).abs()
        idx = int(torch.argmin(diffs).item())
        return 1000 - idx

    # -----------------------------------------------------------------
    # MAE-extension helpers.
    # -----------------------------------------------------------------
    @staticmethod
    def _compute_chunk_mae(
        pred_chunk: torch.Tensor,
        gt_chunk: torch.Tensor,
    ) -> float:
        """Mean-absolute-error between a generated chunk and GT in
        latent space, averaged over all dims AND all DDP ranks.

        Casts to fp32 to avoid bf16/fp16 underflow on small diffs;
        all-reduces with ``ReduceOp.AVG`` so every rank ends up with
        the same scalar — required so the extension decision stays in
        lockstep across DDP ranks (otherwise some ranks would extend
        and others wouldn't, drifting their KV cache fill levels).
        Returns a Python float (zero-dim tensor materialised).
        """
        mae = (pred_chunk.detach().float() - gt_chunk.float()).abs().mean()
        if dist.is_initialized():
            dist.all_reduce(mae, op=dist.ReduceOp.AVG)
        return float(mae.item())

    # -----------------------------------------------------------------
    # Cache initializers (CF-shape: [B, kv_cache_size, 12, 128]).
    # -----------------------------------------------------------------
    def _initialize_kv_cache(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> None:
        # Config-derived sizing contract: logs the steady state once and
        # raises on window > buffer. Runs on every (re-)allocation, so a
        # local_attn_size_schedule transition (which forces a re-alloc)
        # is re-validated at its new window.
        self._check_cache_contract()
        kv_cache1: list = []
        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros(
                    [batch_size, self.kv_cache_size, 12, 128],
                    dtype=dtype, device=device,
                ),
                "v": torch.zeros(
                    [batch_size, self.kv_cache_size, 12, 128],
                    dtype=dtype, device=device,
                ),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            })
        self.kv_cache1 = kv_cache1

    def _initialize_crossattn_cache(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> None:
        crossattn_cache: list = []
        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
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
        self.crossattn_cache = crossattn_cache

    def reset_cache_state(self) -> None:
        """Force the next ``_initialize_kv_cache``/``_initialize_crossattn_cache``
        call to re-allocate by setting the existing caches to None. Used
        by the streaming flow to release GPU memory between sequences
        (each new sequence calls ``setup_sequence`` which re-initialises).
        """
        self.kv_cache1 = None
        self.crossattn_cache = None

    # -----------------------------------------------------------------
    # KV-cache CPU snapshot / restore  (FT_v3 post-build, "option 1").
    # The rolling KV cache is captured to CPU RAM at a roll boundary so a
    # POST-roll rollout2 can restore it and regenerate only the TAIL (with
    # one fewer seed chunk = +1 drift) — instead of prebuilding a full
    # rollout2 at setup before the dynamic depth is known. CPU RAM (not
    # GPU) so deep rides can't OOM the device. ``max_frames`` windows the
    # snapshot to the last N frames' worth of tokens (defaults to the full
    # buffer); a windowed snapshot keeps the per-snapshot RAM bounded so a
    # small ring of them (one per recent roll boundary) stays cheap.
    # -----------------------------------------------------------------
    def snapshot_kv_cache_cpu(
        self, max_frames: Optional[int] = None
    ) -> Optional[dict]:
        from pipeline.kv_snapshot import snapshot_kv_cache_cpu as _snap
        return _snap(self.kv_cache1, int(self.frame_seq_length), max_frames)

    def restore_kv_cache_cpu(
        self, snap: dict, *, batch_size: int, dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Re-materialise ``kv_cache1`` on ``device`` from a CPU snapshot.
        Reallocates the full (zero) buffers then copies the snapshot's
        captured window back per block. ``crossattn_cache`` is the text
        cross-attn (stable across the roll) and is left as-is.
        """
        from pipeline.kv_snapshot import restore_into_kv_cache as _restore
        self._initialize_kv_cache(
            batch_size=batch_size, dtype=dtype, device=device)
        _restore(self.kv_cache1, snap, device=device, dtype=dtype)

    def generate_chunk_with_cache(
        self,
        noise: torch.Tensor,
        current_start_frame: int,
        *,
        requires_grad: bool = True,
        prefer_cache_pred_in_output: bool = False,
        gt_latents: Optional[torch.Tensor] = None,
        sync_exit_flags: bool = True,
        force_exit_step: Optional[int] = None,
        flash_dmd_enabled: bool = False,
        flash_dmd_gan_t: int = 60,
        **conditional_dict,
    ) -> Tuple[torch.Tensor, Optional[int], Optional[int]]:
        """Streaming variant of ``inference_with_trajectory`` — rolls a
        single chunk against the EXISTING ``kv_cache1`` /
        ``crossattn_cache`` (caller is responsible for initialising
        them via ``setup_sequence`` and seed-prefilling). No
        cache re-init, no seed prefill, no MAE-extension loop. Per-block
        rolling denoise + no-grad finish-denoise (LongLive parity for
        clean K/V) + cache-update at ``context_noise``.

        Args:
            noise: ``[B, F, C, H, W]`` initial noise for this chunk.
                ``F`` must be a multiple of ``num_frame_per_block``.
            current_start_frame: absolute starting frame index in the
                sequence (= the persistent KV cache position to write
                into). Caller advances this between calls.
            requires_grad: when False, force the rollout's grad-active
                exit branch into ``no_grad`` (the trainer's critic
                rollout already wraps in ``with torch.no_grad():``;
                this is belt-and-suspenders LongLive parity).
            prefer_cache_pred_in_output: visualisation only — write
                the post-finish-denoise ``cache_pred`` to the per-chunk
                output instead of the grad-active exit-rung pred.
            gt_latents: ``[B, F_gt, C, H, W]`` GT covering this chunk's
                absolute window for the per-chunk MAE metric.
                Optional; pass ``None`` to skip MAE.
            conditional_dict: per-frame action streams covering the
                ``num_frame_per_block``-length slice STARTING at
                ``current_start_frame``. Caller is responsible for
                slicing.

        Returns:
            (output, denoised_timestep_from, denoised_timestep_to),
            where ``output`` is shape ``[B, F, C, H, W]`` — the rolled
            chunk only (no seed, no prior frames).
        """
        if self.kv_cache1 is None or self.crossattn_cache is None:
            raise RuntimeError(
                "generate_chunk_with_cache requires pre-allocated caches; "
                "call ``_initialize_kv_cache`` + ``_initialize_crossattn_cache`` "
                "(typically via ``setup_sequence``) before this method."
            )

        self._last_extension_metrics = {}
        # Reset per-call Flash-DMD t=flash_dmd_gan_t output.
        self._flash_dmd_gan_output: Optional[torch.Tensor] = None
        # Reset per-call clean_chunk buffer (per-block post-Step-3.3.5
        # cache_pred, detached). See ``__init__`` docstring for
        # consumer details.
        self._clean_chunk: Optional[torch.Tensor] = None
        # A23 / WP-PIXGAN. Reset per-call GRAD twin of ``_clean_chunk``
        # (gate ``pix_finish_grad_enabled``). Holds the LADDER-ENDPOINT
        # x0 -- the tensor ``utils/eval_causal_AR.py`` commits and
        # renders -- WITH an autograd path back to the generator
        # weights, so a pixel-space critic can be fed an
        # inference-parity fake instead of the t=60 flash tensor.
        # ``None`` whenever the gate is off.
        self._clean_chunk_grad: Optional[torch.Tensor] = None
        # Frame-level attachment mask for the buffer above, ``[F]``
        # bool. Published only together with the buffer.
        self._clean_chunk_grad_mask: Optional[torch.Tensor] = None
        # Per-call A23 telemetry; populated ONLY when the gate is on.
        self._pix_finish_grad_stats: dict = {}

        # A23 gate. Read the way every other knob on this class is read
        # (plain attribute set by the trainer; absent => OFF). Default
        # False keeps this method byte-identical to the pre-A23 code:
        # no new tensor allocation, no new forward, no new RNG draw, no
        # new key on anything the caller can observe.
        pix_finish_grad = bool(
            getattr(self, "pix_finish_grad_enabled", False)
        )

        # A24 / WP-PIXGAN KV-commit tripwire gate. ``pix_kv_commit_check_every``
        # (int, default 0 = OFF) — read as a plain attribute on the PIPELINE,
        # exactly like ``pix_finish_grad_enabled`` above. 0 keeps this method
        # byte-identical: no fingerprint, no tensor, no RNG draw, no new key.
        _kv_check_every = int(
            getattr(self, "pix_kv_commit_check_every", 0) or 0
        )
        kv_commit_check = False
        if _kv_check_every > 0:
            self._pix_kv_commit_stats = {}
            # Verify any record left pending by a PREVIOUS armed call. By
            # the time the next rollout starts, that step's ``.backward()``
            # has run. This is the FALLBACK site (site_code=2) and is
            # weaker than the primary one — other forwards may have touched
            # the slots in between — so it is labelled, never conflated.
            # PRIMARY site: the trainer calling ``pix_kv_commit_verify()``
            # right after ``generator_loss.backward()`` (site_code=1).
            _kv_deferred = self._pix_kv_commit_run_verify(site_code=2)
            if _kv_deferred:
                self._last_extension_metrics.update(_kv_deferred)
            _kv_n = int(getattr(self, "_pix_kv_commit_calls", 0)) + 1
            self._pix_kv_commit_calls = _kv_n
            kv_commit_check = ((_kv_n - 1) % _kv_check_every) == 0
            self._pix_kv_commit_pending = [] if kv_commit_check else None
            if kv_commit_check:
                self._pix_kv_commit_cache_tag = self._pix_kv_cache_tag()

        batch_size, num_frames, _, _, _ = noise.shape
        npb = self.num_frame_per_block
        cps = self.chunks_per_rolling_step

        if num_frames % npb != 0:
            raise RuntimeError(
                f"num_frames ({num_frames}) must be divisible by "
                f"num_frame_per_block ({npb})."
            )
        self._check_cache_occupancy(
            current_start_frame + num_frames, "generate_chunk_with_cache")
        num_blocks_total = num_frames // npb
        all_num_frames: List[int] = []
        remaining_blocks = num_blocks_total
        while remaining_blocks > 0:
            chunks_this_step = min(cps, remaining_blocks)
            all_num_frames.append(chunks_this_step * npb)
            remaining_blocks -= chunks_this_step

        output = torch.zeros_like(noise)
        # Flash-DMD t=flash_dmd_gan_t buffer. See
        # ``inference_with_trajectory`` for full rationale.
        flash_dmd_gan_output = (
            torch.zeros_like(noise) if flash_dmd_enabled else None
        )
        # Aux-teacher clean_x buffer (per-block post-Step-3.3.5
        # cache_pred, detached). Allocated UNCONDITIONALLY — see
        # ``inference_with_trajectory`` for rationale: the rollout
        # always denoises to the last rung (~t=178.6) to populate the
        # KV cache, so exposing that ``cache_pred`` to downstream
        # consumers is free.
        clean_chunk = torch.zeros_like(noise)
        # A23 grad twin of ``clean_chunk``. Allocated the same way but
        # ONLY under the gate (the retained one-rung graph is real
        # memory). Written with the GRAD-CARRYING ladder-endpoint
        # tensor at the same index slice; ``clean_chunk`` itself keeps
        # its ``.detach()`` and is untouched.
        #
        # NOTE on values: with ``flash_dmd_enabled=True`` the two
        # buffers hold DIFFERENT tensors by design --
        # ``clean_chunk`` = the t=flash_dmd_gan_t flash x0 (what
        # training commits), ``clean_chunk_grad`` = the PRE-FLASH
        # ladder endpoint (what inference commits). That divergence is
        # the entire point of A23; see the one-shot notice below.
        #
        # AND: only the frames whose ``clean_chunk_grad_mask`` entry is
        # True are written at all. Frames with no grad rung to attach
        # (trailing block of a multi-block call; a block that exited at
        # the last rung) keep this buffer's ZEROS init -- they are NOT
        # backfilled with ``cache_pred``, because with flash on that is
        # the flash tensor and writing it presents the exact thing A23
        # excludes as the ladder endpoint. Zeros here mean "not
        # measured", and the mask says which.
        clean_chunk_grad = (
            torch.zeros_like(noise) if pix_finish_grad else None
        )
        # A23 FRAME-LEVEL attachment mask, ``[F]`` bool over the SAME
        # frame axis as ``clean_chunk_grad``. ``True`` = that frame's
        # slice was written with a graph-carrying tensor.
        #
        # This exists because ``clean_chunk_grad.requires_grad`` is a
        # WHOLE-BUFFER property: it flips True as soon as ONE block
        # attaches. On a multi-block call the TRAILING block is
        # deliberately detached (see ``finish_grad_active`` below), and
        # so is any block that had no post-exit rung to attach to. A
        # consumer that reduced its generator loss over the whole
        # chunk would then silently average in frames with no path to
        # the generator -- diluting the adversarial signal by
        # 1/num_blocks with nothing in the logs saying so. Consumers
        # MUST select on this mask instead of assuming the whole chunk
        # is live.
        clean_chunk_grad_mask = (
            torch.zeros(num_frames, dtype=torch.bool, device=noise.device)
            if pix_finish_grad else None
        )
        # A23 per-call counters (blocks that attached a grad rung /
        # blocks that could not because the ladder had no post-exit
        # rung left to run).
        _pix_fg_attached = 0
        _pix_fg_no_rung = 0
        _pix_fg_nograd = 0

        num_denoising_steps = len(self.denoising_step_list)
        # Cold-start only (warm_start removed). Every block uses the
        # full denoising ladder; exit rung sampled per block.
        exit_flags = self.generate_and_sync_list(
            len(all_num_frames), num_denoising_steps, device=noise.device,
            sync=sync_exit_flags, force_exit_step=force_exit_step,
            # Trainer-set attr (default False = the old hardcoded value).
            # True = KV-probe regime (researcher-approved 14:0x): reserves
            # a finish rung so the A23 grad path can attach.
            exclude_last_rung=bool(
                getattr(self, "exit_exclude_last_rung", False)),
        )
        # In streaming mode the generator's gradient gate is not the
        # rollout-vs-warmup split — it's a single flag from the caller.
        # ``requires_grad=False`` ⇒ no grad anywhere; True ⇒ grad on the
        # exit-rung forward.
        start_gradient_frame_index = 0 if requires_grad else (num_frames + 1)

        denoised_pred = None
        timestep = None
        sequence_start = current_start_frame
        for block_index, current_num_frames in enumerate(all_num_frames):
            block_start_in_noise = current_start_frame - sequence_start
            block_cond = _slice_per_frame_streams(
                conditional_dict,
                frame_start=current_start_frame,
                frame_count=current_num_frames,
            )
            if kv_commit_check:
                # A24: this block's K/V slot window, and the fingerprints of
                # the checkpointed forwards that write into it.
                _kv_h_exit = None
                _kv_h_finish = None
                _kv_h_flash = None
                _kv_tok_a = current_start_frame * self.frame_seq_length
                _kv_tok_b = _kv_tok_a + (
                    current_num_frames * self.frame_seq_length
                )

            # Cold-start: pure noise input + full denoising ladder.
            noisy_input = noise[
                :, block_start_in_noise: block_start_in_noise + current_num_frames,
            ]

            # Rolling denoise loop with truncated random exit, full
            # ladder.
            exit_index = num_denoising_steps - 1
            for index in range(0, num_denoising_steps):
                current_timestep = self.denoising_step_list[index]
                # Phase-LoRA dispatch — see comment at the matching
                # site in the streaming rollout above.
                self._maybe_set_phase_lora_for_step(index, num_denoising_steps)
                if self.same_step_across_blocks:
                    exit_flag = (index == exit_flags[0])
                else:
                    exit_flag = (index == exit_flags[block_index])

                ts_value = int(round(float(current_timestep)))
                timestep = torch.full(
                    [batch_size, current_num_frames], ts_value,
                    device=noise.device, dtype=torch.int64,
                )

                if not exit_flag:
                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=block_cond,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                        )
                        next_t_value = int(round(float(
                            self.denoising_step_list[index + 1]
                        )))
                        flat = denoised_pred.flatten(0, 1)
                        noisy_input = self.scheduler.add_noise(
                            flat,
                            torch.randn_like(flat),
                            next_t_value
                            * torch.ones(
                                [batch_size * current_num_frames],
                                device=noise.device, dtype=torch.long,
                            ),
                        ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # Skip the grad-active random-exit rung forward on
                    # the LAST block of MULTI-BLOCK calls. The DMD
                    # scoring mask (``_dmd_score_grad_mask``) zeros
                    # the trailing ``num_frame_per_block`` frames of
                    # the ``chunk_size``-length scoring window
                    # structurally (v14 joint-TF OOD region). The
                    # ``is_multi_block`` gate is CRITICAL: in single-
                    # block calls (streaming iter k>=2 with
                    # ``new_frames=npb``), the only block IS marked
                    # last; skipping its grad would detach the entire
                    # rollout output → ``full_chunk`` cat returns a
                    # no-grad tensor → DMD empty-mask short-circuit's
                    # ``zero_loss`` has no graph → gen backward fails
                    # with "element 0 of tensors does not require
                    # grad" (observed at production iter 6).
                    is_last_block = (block_index == len(all_num_frames) - 1)
                    is_multi_block = (len(all_num_frames) > 1)
                    skip_last_block_grad = is_last_block and is_multi_block
                    if (not requires_grad) or skip_last_block_grad:
                        with torch.no_grad():
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=block_cond,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length,
                            )
                    else:
                        # Activation-checkpoint the grad-active random-
                        # exit rung forward (the dominant per-block
                        # held activation; ~3 GB across the iter-1
                        # rollout's 7 blocks). Mirrors the flash-DMD
                        # ckpt below: default-args closure for
                        # by-value capture; cache safety guaranteed
                        # by Step 3.4 context-noise commits being
                        # outside the checkpoint scope.
                        #
                        # Phase-LoRA dispatch INSIDE the ckpt fn so
                        # backward-time recompute re-sets the same
                        # active adapter (rung_{exit_index}). See the
                        # parallel _exit_fn in inference_with_trajectory
                        # for the full rationale.
                        def _exit_fn(
                            x,
                            _self=self,
                            _gen=self.generator,
                            _cond=block_cond,
                            _t=timestep,
                            _kv=self.kv_cache1,
                            _xa=self.crossattn_cache,
                            _start=current_start_frame * self.frame_seq_length,
                            _idx=index,
                            _ns=num_denoising_steps,
                        ):
                            _self._maybe_set_phase_lora_for_step(_idx, _ns)
                            return _gen(
                                noisy_image_or_video=x,
                                conditional_dict=_cond,
                                timestep=_t,
                                kv_cache=_kv,
                                crossattn_cache=_xa,
                                current_start=_start,
                            )

                        _, denoised_pred = _ckpt(
                            _exit_fn, noisy_input, use_reentrant=False,
                        )
                        if kv_commit_check:
                            _kv_h_exit = self._pix_kv_hash_slots(
                                _kv_tok_a, _kv_tok_b)
                    exit_index = index
                    break

            # Post-exit no_grad chain through remaining rungs. Ends
            # with ``cache_pred`` = clean x0 estimate at the last rung.
            #
            # A23 (``pix_finish_grad_enabled``, default OFF): the LAST
            # rung of this loop -- and only the last -- runs WITH grad
            # under ``_ckpt(..., use_reentrant=False)``, so the ladder
            # endpoint (the tensor inference commits and renders) can
            # be handed to a pixel critic with a path back to the
            # generator weights. Every earlier rung stays ``no_grad``,
            # and the grad rung's INPUT is built from the DETACHED
            # previous rung output, so exactly ONE rung's worth of
            # activation graph is retained per block.
            #
            # The grad output is captured into ``finish_grad_pred`` and
            # ``cache_pred`` is IMMEDIATELY re-detached, so the flash
            # forward, ``output``, the ``clean_chunk`` buffer and the
            # Step-3.4 K/V commit below all see exactly the tensor they
            # saw before this change (paper 3.3 cross-timestep
            # decoupling: the commit must stay graph-free).
            _last_blk = (block_index == len(all_num_frames) - 1)
            _multi_blk = (len(all_num_frames) > 1)
            # Mirror ``flash_grad_active`` below: honour the caller's
            # grad gate, and skip the trailing block of MULTI-block
            # calls (the heavy iter-1 rollout) for the same memory
            # reason the flash forward does. Single-block calls (the
            # 99% streaming case) always keep the grad rung.
            finish_grad_active = (
                pix_finish_grad
                and requires_grad
                and not (_last_blk and _multi_blk)
            )
            finish_grad_pred: Optional[torch.Tensor] = None
            cache_pred = denoised_pred.detach()
            for j in range(exit_index + 1, num_denoising_steps):
                # Phase-LoRA dispatch per rung; see the equivalent
                # block in inference_with_trajectory for rationale.
                self._maybe_set_phase_lora_for_step(j, num_denoising_steps, re_arm=False)
                next_t_value = int(round(float(
                    self.denoising_step_list[j]
                )))
                flat = cache_pred.flatten(0, 1)
                cache_input = self.scheduler.add_noise(
                    flat,
                    torch.randn_like(flat),
                    next_t_value * torch.ones(
                        [batch_size * current_num_frames],
                        device=noise.device, dtype=torch.long,
                    ),
                ).unflatten(0, denoised_pred.shape[:2])
                step_t = torch.full_like(timestep, next_t_value)
                if finish_grad_active and j == num_denoising_steps - 1:
                    # Grad-on FINAL finish rung. Same idiom as the
                    # ``_exit_fn`` / ``_flash_fn`` checkpoints: default
                    # -args closure for by-value capture, Phase-LoRA
                    # dispatch INSIDE the fn (with re_arm, this is a
                    # grad-on site) so backward-time recompute routes
                    # to the same adapter. Input explicitly detached:
                    # the previous rungs were ``no_grad`` so it is
                    # already graph-free, but the detach makes the
                    # one-rung bound structural rather than incidental.
                    def _finish_fn(
                        x,
                        _self=self,
                        _gen=self.generator,
                        _cond=block_cond,
                        _t=step_t,
                        _kv=self.kv_cache1,
                        _xa=self.crossattn_cache,
                        _start=current_start_frame * self.frame_seq_length,
                        _idx=j,
                        _ns=num_denoising_steps,
                    ):
                        _self._maybe_set_phase_lora_for_step(_idx, _ns)
                        return _gen(
                            noisy_image_or_video=x,
                            conditional_dict=_cond,
                            timestep=_t,
                            kv_cache=_kv,
                            crossattn_cache=_xa,
                            current_start=_start,
                        )

                    _, finish_grad_pred = _ckpt(
                        _finish_fn, cache_input.detach(),
                        use_reentrant=False,
                    )
                    # Everything downstream keeps the pre-A23 tensor.
                    cache_pred = finish_grad_pred.detach()
                    if kv_commit_check:
                        _kv_h_finish = self._pix_kv_hash_slots(
                            _kv_tok_a, _kv_tok_b)
                else:
                    with torch.no_grad():
                        _, cache_pred = self.generator(
                            noisy_image_or_video=cache_input,
                            conditional_dict=block_cond,
                            timestep=step_t,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                        )

            if finish_grad_active:
                # Explicit, LOUD handling of the "no finish rung to
                # attach" case: when the block's random exit rung IS
                # the last rung, ``range(exit_index + 1, K)`` is empty
                # and there is no post-exit forward to hang a graph on.
                # ``cache_pred`` is then the exit rung's own output --
                # which IS the ladder endpoint numerically, but we
                # deliberately do NOT reuse the exit-rung graph here
                # (it is already owned by the DMD path; sharing it
                # would make a second .backward() on the pixel term a
                # double-backward error). The slice is written
                # detached and counted, never silently passed off as
                # grad-carrying: ``_publish`` below drops the whole
                # buffer to ``None`` if NOTHING in it carries grad.
                if finish_grad_pred is None:
                    _pix_fg_no_rung += 1
                    self._warn_once_pix_finish_grad_no_rung(
                        exit_index, num_denoising_steps,
                    )
                elif not finish_grad_pred.requires_grad:
                    # Should not happen (requires_grad=True caller +
                    # trainable generator), but a grad-less tensor here
                    # would silently defeat the whole feature.
                    _pix_fg_nograd += 1
                    self._warn_once_pix_finish_grad_nograd()
                else:
                    _pix_fg_attached += 1

            # Flash-DMD t=flash_dmd_gan_t grad-on forward (per block,
            # no final-block restriction). Inputs: noised cache_pred
            # at flash_dmd_gan_t. Output goes to ``flash_dmd_gan_output``
            # for the GAN / aux losses; K/V slots overwritten by
            # Step 3.4 below for paper §3.3 cross-timestep decoupling.
            flash_dmd_pred: Optional[torch.Tensor] = None
            if flash_dmd_enabled:
                # Phase-LoRA dispatch for the flash forward. See
                # _maybe_set_phase_lora_for_flash for routing rules
                # (env var FLASH_PHASE_LORA_ROUTE: "exit" no-op vs
                # "last" → rung_{K-1}).
                self._maybe_set_phase_lora_for_flash()
                flash_t_value = int(flash_dmd_gan_t)
                flash_flat = cache_pred.flatten(0, 1)
                flash_input = self.scheduler.add_noise(
                    flash_flat,
                    torch.randn_like(flash_flat),
                    flash_t_value * torch.ones(
                        [batch_size * current_num_frames],
                        device=noise.device, dtype=torch.long,
                    ),
                ).unflatten(0, cache_pred.shape[:2])
                flash_t_step = torch.full_like(timestep, flash_t_value)
                # Skip the grad-active flash-DMD forward on the LAST
                # block of MULTI-BLOCK calls only. In streaming mode
                # iter 1 rolls all 7 chunks in one call (multi-block:
                # skip block 6 = the trailing chunk of the 21-frame
                # rollout); iters k>=2 roll 1 chunk per call (single-
                # block: keep the only-block grad-active so GAN signal
                # survives in subsequent iters). Matches the user's
                # "6 chunks instead of 7" intent for the heavy iter 1
                # without nuking GAN supervision in the 99% of calls
                # that are single-block.
                is_last_block = (block_index == len(all_num_frames) - 1)
                is_multi_block = (len(all_num_frames) > 1)
                flash_grad_active = (
                    requires_grad
                    and not (is_last_block and is_multi_block)
                )
                if flash_grad_active:
                    # Activation-checkpoint via default-args closure;
                    # see ``inference_with_trajectory`` for the cache
                    # safety analysis (prior chunks' K/V are committed
                    # to context_noise by Step 3.4 and stable until
                    # next rollout reset, so backward-time recompute
                    # reads bit-identical state to the original
                    # forward).
                    # Phase-LoRA dispatch INSIDE the ckpt fn so
                    # backward-time recompute (use_reentrant=False
                    # re-runs forward) also routes to the same active
                    # adapter (rung_flash in K+1 design). Otherwise the
                    # recompute would inherit whichever adapter was set
                    # by the most-recent external call, sending the
                    # gradient to the wrong rung's lora matrices —
                    # manifests as DDP's "marked ready twice" error.
                    def _flash_fn(
                        x,
                        _self=self,
                        _gen=self.generator,
                        _cond=block_cond,
                        _t=flash_t_step,
                        _kv=self.kv_cache1,
                        _xa=self.crossattn_cache,
                        _start=current_start_frame * self.frame_seq_length,
                    ):
                        _self._maybe_set_phase_lora_for_flash()
                        return _gen(
                            noisy_image_or_video=x,
                            conditional_dict=_cond,
                            timestep=_t,
                            kv_cache=_kv,
                            crossattn_cache=_xa,
                            current_start=_start,
                        )

                    _, flash_dmd_pred = _ckpt(
                        _flash_fn, flash_input, use_reentrant=False,
                    )
                    if kv_commit_check:
                        _kv_h_flash = self._pix_kv_hash_slots(
                            _kv_tok_a, _kv_tok_b)
                else:
                    # Phase-LoRA dispatch for the no_grad flash branch.
                    # See inference_with_trajectory's twin for rationale.
                    # re_arm=False (Fix B): no_grad path; skipping
                    # the re-arm reduces DDP-state churn.
                    self._maybe_set_phase_lora_for_flash(re_arm=False)
                    with torch.no_grad():
                        _, flash_dmd_pred = self.generator(
                            noisy_image_or_video=flash_input,
                            conditional_dict=block_cond,
                            timestep=flash_t_step,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                        )

            output_pred = (
                cache_pred.to(denoised_pred.dtype)
                if prefer_cache_pred_in_output
                else denoised_pred
            )
            output[
                :, block_start_in_noise: block_start_in_noise + current_num_frames,
            ] = output_pred

            # Step 3.2.b + 3.3.5 UNIFIED: single grad-on t=60 forward
            # serves both the GAN adv buffer AND the cache_pred state.
            # No separate no_grad t=60 refinement (one forward saved
            # per block).
            if flash_dmd_enabled:
                if flash_dmd_pred is None:
                    raise RuntimeError(
                        "flash_dmd_enabled=True but Flash-DMD step did "
                        "not produce flash_dmd_pred."
                    )
                flash_dmd_gan_output[
                    :, block_start_in_noise: block_start_in_noise + current_num_frames,
                ] = flash_dmd_pred
                cache_pred = flash_dmd_pred.detach()

            # Stash post-Step-3.3.5 ``cache_pred`` into the per-rollout
            # ``clean_chunk`` buffer. Detached. Indexing mirrors the
            # ``output`` write above (``block_start_in_noise`` for the
            # streaming path's noise-tensor-relative offset).
            if clean_chunk is not None:
                clean_chunk[
                    :,
                    block_start_in_noise: block_start_in_noise + current_num_frames,
                ] = cache_pred.detach()

            # A23 grad twin. Written at the SAME index slice, and ONLY
            # ever with ``finish_grad_pred`` -- the PRE-FLASH ladder
            # endpoint. Never ``cache_pred``, which by this point has
            # been overwritten with the t=flash_dmd_gan_t flash output
            # when ``flash_dmd_enabled`` is on. That is the whole A23
            # fix: the critic's fake must be the tensor inference
            # commits (t~208 ladder endpoint), not the t=60 flash
            # prediction.
            #
            # NO FALLBACK WRITE. This branch used to fall back to
            # ``cache_pred.detach()`` when there was no finish rung to
            # attach, with a comment claiming the slice is written with
            # ``finish_grad_pred`` "never ``cache_pred``". The code did
            # the opposite of its comment, and with flash ON it wrote
            # the FLASH tensor: measured on a 9-frame / 3-block flash-ON
            # call, ``clean_chunk_grad[:, 6:9]`` came back bit-identical
            # (max|diff| = 0.0) to the t=60 flash output -- precisely
            # the tensor A23 exists to exclude. The mask read False, so
            # a compliant consumer was safe, but any consumer or viz
            # decoding the whole buffer saw the wrong tensor presented
            # as the ladder endpoint.
            #
            # The fallback slices are therefore LEFT AT THEIR ZEROS
            # INIT. One invariant, no flash-dependence: a frame of
            # ``clean_chunk_grad`` is the ladder endpoint iff its mask
            # entry is True, and is exactly zeros otherwise. Zeros
            # cannot be mistaken for a prediction; the flash tensor can.
            # Consumers that want a value for the detached frames must
            # read ``clean_chunk`` (which still holds ``cache_pred`` for
            # every frame) and know what they are getting.
            if clean_chunk_grad is not None:
                _grad_live = False
                if finish_grad_pred is not None:
                    assert (
                        flash_dmd_pred is None
                        or finish_grad_pred is not flash_dmd_pred
                    ), (
                        "pix_finish_grad: the grad buffer would hold the "
                        "FLASH output instead of the ladder endpoint -- "
                        "this defeats the A23 inference-parity fix."
                    )
                    if flash_dmd_enabled:
                        self._warn_once_pix_finish_grad_flash_on()
                    _grad_live = bool(finish_grad_pred.requires_grad)
                    clean_chunk_grad[
                        :,
                        block_start_in_noise:
                        block_start_in_noise + current_num_frames,
                    ] = finish_grad_pred
                # else: ``finish_grad_active`` was False for this block
                # (trailing block of a multi-block call, or a no-grad
                # caller) OR the block exited at the last rung so there
                # was no finish rung to attach to. Nothing is written --
                # the slice stays zeros and the mask stays False.
                if clean_chunk_grad_mask is not None:
                    clean_chunk_grad_mask[
                        block_start_in_noise:
                        block_start_in_noise + current_num_frames
                    ] = _grad_live

            # Cache-update commit at t=context_noise. Runs in BOTH
            # modes — see ``inference_with_trajectory`` for the full
            # rationale (paper §3.3 K/V decoupling: must overwrite
            # the grad-attached K/V slots written by the Flash-DMD
            # forward with graph-free K/V so the next block's exit-
            # rung forward doesn't pull DMD's grad through the prior
            # block's Flash-DMD gen forward).
            commit_input_clean = cache_pred.detach()
            # CARN seam affine (latent CARN v0): gated per-channel mean/std
            # re-anchor of the committed context toward the ride seed's latent
            # stats -- counters the AR drift walk (DC/color haze, contraction)
            # INSIDE the loop; affine correction measured ~80% sufficient in
            # the contraction-law study. lambda=0 (default) = byte-identical.
            # Temperature scaling (carn_seam_temp, default 1.0 = off): scale
            # per-channel deviations of the committed context by T to counter
            # the per-chunk variance contraction (k~0.917 for the 4-rung
            # sampler => T ~ 1/k). Pure re-inflation, no target stats needed;
            # composes with (runs before) the affine re-anchor below.
            _tT = float(getattr(self, "carn_seam_temp", 1.0) or 1.0)
            if abs(_tT - 1.0) > 1e-6:
                _x = commit_input_clean.float()
                _mu = _x.mean(dim=(0, 1, 3, 4), keepdim=True)
                commit_input_clean = (
                    _mu + _tT * (_x - _mu)
                ).to(commit_input_clean.dtype)
            # Global drift counter-bias (CARN repulsor v1): subtract
            # lambda_d * d_hat (per-channel-mean drift/roll, fitted from the
            # frozen-model drift probe -- transition cos +0.53, ride cos
            # +0.74) from every committed chunk. mu-only by design (the
            # grep-sign study: the mu-half is right-signed; sigma-half is
            # handled by temp/affine instead).
            _dl = float(getattr(self, "carn_seam_drift_lambda", 0.0) or 0.0)
            _dv = getattr(self, "_carn_seam_drift_vec", None)
            if _dl > 0.0 and _dv is not None:
                commit_input_clean = (
                    commit_input_clean.float()
                    - _dl * _dv.view(1, 1, -1, 1, 1).to(
                        commit_input_clean.device)
                ).to(commit_input_clean.dtype)
            _cl = float(getattr(self, "carn_seam_affine_lambda", 0.0) or 0.0)
            _ct = getattr(self, "_carn_seam_target", None)
            if _cl > 0.0 and _ct is not None:
                _tm, _tsd = _ct
                _x = commit_input_clean.float()
                _mu = _x.mean(dim=(0, 1, 3, 4), keepdim=True)
                _sd = _x.std(dim=(0, 1, 3, 4), keepdim=True).clamp_min(1e-4)
                _tm_ = _tm.view(1, 1, -1, 1, 1).to(_x)
                _tsd_ = _tsd.view(1, 1, -1, 1, 1).to(_x)
                commit_input_clean = (
                    (_x - _mu) / _sd * (_cl * _tsd_ + (1.0 - _cl) * _sd)
                    + (_cl * _tm_ + (1.0 - _cl) * _mu)
                ).to(commit_input_clean.dtype)
            context_timestep = torch.full_like(timestep, self.context_noise)
            if int(self.context_noise) == 0:
                # context_noise=0 did NOT mean "clean". FlowMatchScheduler's
                # grid (shift=5, sigma_min=0, extra_one_step) has NO t=0
                # entry: its smallest sigma is 5*0.001/1.004 = 0.00498, and
                # add_noise resolves t by argmin|timesteps - t|. So every
                # commit wrote 0.995*pred + 0.005*eps while telling the DiT
                # t=0 — i.e. it silently noised the clean context, against
                # the standing rule, and compounding over 7 commits. The ODE
                # stage commits verbatim (ode_rollout.py: commit_lat passed
                # straight through at t=0). Match it.
                cache_commit = commit_input_clean
            else:
                cache_commit = self.scheduler.add_noise(
                    commit_input_clean.flatten(0, 1),
                    torch.randn_like(commit_input_clean.flatten(0, 1)),
                    context_timestep.flatten(0, 1),
                ).unflatten(0, commit_input_clean.shape[:2])
            # Phase-LoRA dispatch for the context_noise commit forward.
            # See the equivalent block in inference_with_trajectory for
            # rationale: route through the lowest-t ODE rung (= K-1)
            # so the KV commit traces match the canonical low-t output
            # distribution, not the flash-adapter (rung_flash) output
            # left active by the preceding flash forward.
            self._maybe_set_phase_lora_for_step(
                max(0, len(self.denoising_step_list) - 1),
                len(self.denoising_step_list),
                re_arm=False,
            )
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=cache_commit,
                    conditional_dict=block_cond,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )

            if kv_commit_check:
                # A24: Step 3.4's commit is the LAST forward writer of these
                # slots and the state that must survive backward. Fingerprint
                # it now; the post-backward comparison happens in
                # ``pix_kv_commit_verify`` / the deferred site above.
                if self._pix_kv_commit_pending is None:
                    self._pix_kv_commit_pending = []
                self._pix_kv_commit_pending.append({
                    "block_index": int(block_index),
                    "frame_start": int(current_start_frame),
                    "tok_a": int(_kv_tok_a),
                    "tok_b": int(_kv_tok_b),
                    "commit": self._pix_kv_hash_slots(_kv_tok_a, _kv_tok_b),
                    "exit": _kv_h_exit,
                    "finish": _kv_h_finish,
                    "flash": _kv_h_flash,
                })

            current_start_frame += current_num_frames

        # Compute denoised_t_from / denoised_t_to from block 0's
        # exit_flag (warm_start removed — all blocks cold-start).
        denoised_t_from, denoised_t_to = None, None
        try:
            if self.same_step_across_blocks:
                idx = exit_flags[0]
                from_t = int(round(float(self.denoising_step_list[idx])))
                to_t = (
                    0
                    if idx == num_denoising_steps - 1
                    else int(round(float(self.denoising_step_list[idx + 1])))
                )
                denoised_t_from, denoised_t_to = from_t, to_t
        except Exception:
            pass

        # Stash the unified t=60 flash output (None when
        # ``flash_dmd_enabled=False``) for downstream consumers.
        self._flash_dmd_gan_output = flash_dmd_gan_output
        # Same stash for the aux-teacher clean_chunk buffer.
        self._clean_chunk = clean_chunk
        # A23 stash. Published ONLY when the buffer actually carries a
        # graph -- a grad-less buffer here would be silently useless to
        # the pixel critic (its G-loss would have no path to the
        # generator), so it is dropped to ``None`` and the reason is in
        # ``_pix_finish_grad_stats`` / the one-shot warnings above.
        if clean_chunk_grad is not None and clean_chunk_grad.requires_grad:
            self._clean_chunk_grad = clean_chunk_grad
            # The mask travels WITH the buffer and is never published
            # without it. ``clean_chunk_grad[:, mask]`` is the only
            # slice a consumer may treat as adversarially live.
            self._clean_chunk_grad_mask = clean_chunk_grad_mask
        else:
            self._clean_chunk_grad = None
            self._clean_chunk_grad_mask = None
        if pix_finish_grad:
            _live_frames = (
                int(clean_chunk_grad_mask.sum().item())
                if clean_chunk_grad_mask is not None else 0
            )
            self._pix_finish_grad_stats = {
                "pix_finish_grad_blocks": float(_pix_fg_attached),
                "pix_finish_grad_no_rung": float(_pix_fg_no_rung),
                "pix_finish_grad_nograd": float(_pix_fg_nograd),
                "pix_finish_grad_published": float(
                    self._clean_chunk_grad is not None
                ),
                # Frame-level attachment. The DILUTION a consumer would
                # silently eat if it ignored the mask is exactly
                # 1 - frames/frames_total; log it from day one.
                "pix_finish_grad_frames": float(_live_frames),
                "pix_finish_grad_frames_total": float(num_frames),
            }
            # Surface through the existing per-call metrics channel so
            # the counters land in the trainer's info dict / logs with
            # no consumer-side change. Gated: when the A23 flag is off
            # this dict is left exactly as the pre-A23 code left it.
            self._last_extension_metrics.update(self._pix_finish_grad_stats)

        return output, denoised_t_from, denoised_t_to

    # -----------------------------------------------------------------
    # A24 / WP-PIXGAN — KV-COMMIT TRIPWIRE  (``pix_kv_commit_check_every``)
    #
    # THE HAZARD. ``_exit_fn`` / ``_finish_fn`` / ``_flash_fn`` are all
    # run under ``_ckpt(..., use_reentrant=False)``, and every one of them
    # WRITES the block's K/V slots as a side effect of the generator
    # forward. In FORWARD order the last writer is Step 3.4's
    # context-noise commit (``commit_input_clean = cache_pred.detach()``),
    # which is exactly the graph-free state paper §3.3 requires. At
    # BACKWARD each of those checkpoints is RECOMPUTED, in REVERSE order,
    # and each recompute re-writes the SAME slots. Nothing re-runs the
    # Step-3.4 commit afterwards, so the slots can end the step holding a
    # recompute's K/V instead of the commit's. The corruption is finite,
    # well-scaled and silent.
    #
    # The existing safety notes on those checkpoints reason about
    # recompute READS being stable; they say nothing about recompute
    # WRITES. A23 added a SECOND KV-writing checkpoint (``_finish_fn``)
    # next to the pre-existing ``_flash_fn``, which is why this tripwire
    # exists.
    #
    # WHAT IT DOES. On an armed call the pipeline fingerprints the K/V
    # slot window of every transformer block at four points per block:
    # after the exit-rung checkpoint, after the finish-rung checkpoint,
    # after the flash checkpoint, and after Step 3.4's commit. The commit
    # fingerprint is the reference. A SECOND fingerprint of the same
    # window, taken AFTER ``.backward()``, is compared against it; the
    # three intermediate fingerprints let a mismatch be attributed to the
    # recompute that produced it.
    #
    # OMIT-NEVER-FAKE. ``pix_kv_commit_match`` is emitted ONLY when two
    # real fingerprints of the same live buffer were actually compared.
    # If the cache was reset/reallocated in between, or there was nothing
    # to hash, the key is ABSENT and ``pix_kv_commit_skipped`` is emitted
    # instead. A forged 1.0 here is precisely the failure this package
    # is hunting, so there is no code path that can produce one.
    # -----------------------------------------------------------------

    # [present, sum, sum-of-squares, token-position moment,
    #  channel-position moment]
    _PIX_KV_STATS = 5

    def _pix_kv_cache_tag(self):
        """Identity of the CURRENT ``kv_cache1`` buffers. Used to refuse a
        comparison across a ``reset_cache_state`` / re-allocation (which
        would compare fingerprints of two different tensors). ``id()``
        alone is not enough (ids get recycled), so the layer-0 storage
        pointer and element count are folded in."""
        c = self.kv_cache1
        if c is None or len(c) == 0:
            return None
        k0 = c[0].get("k") if isinstance(c[0], dict) else None
        if torch.is_tensor(k0):
            return (id(c), len(c), int(k0.data_ptr()), int(k0.numel()))
        return (id(c), len(c), -1, -1)

    def _pix_kv_hash_slots(self, tok_a: int, tok_b: int):
        """Cheap deterministic fingerprint of the K/V token window
        ``[tok_a, tok_b)`` across every transformer block.

        Returns ``[num_blocks, 2, 5]`` float64 (k and v; presence, sum,
        sum-of-squares, token-position moment, channel-position moment)
        on the cache's device, or ``None`` when there is nothing to hash.

        Cost: two reductions plus two tiny weighted sums per (block, k/v).
        No full-size temporaries (``torch.dot`` instead of ``(x*x).sum()``)
        and NO RNG — ``arange``/``sin``/``cos`` only — so arming the
        tripwire cannot shift the sampler's RNG stream for the frames it
        is measuring.
        """
        cache = self.kv_cache1
        if cache is None or len(cache) == 0:
            return None
        rows = []
        dev = None
        any_present = False
        for blk in cache:
            for key in ("k", "v"):
                t = blk.get(key) if isinstance(blk, dict) else None
                if not torch.is_tensor(t) or t.dim() < 2:
                    rows.append(None)
                    continue
                a = max(0, int(tok_a))
                b = min(int(t.shape[1]), int(tok_b))
                if b <= a:
                    rows.append(None)
                    continue
                with torch.no_grad():
                    x = t[:, a:b].detach().to(torch.float32)
                    dev = x.device
                    flat = x.reshape(-1)
                    s_sum = flat.sum()
                    s_sq = torch.dot(flat, flat)
                    t_axes = tuple(
                        i for i in range(x.dim()) if i != 1
                    )
                    per_t = x.sum(dim=t_axes)
                    per_d = x.sum(dim=tuple(range(x.dim() - 1)))
                    r_t = torch.arange(
                        per_t.numel(), device=x.device, dtype=torch.float32)
                    r_d = torch.arange(
                        per_d.numel(), device=x.device, dtype=torch.float32)
                    s_t = (
                        per_t * torch.sin(r_t * 0.7548776662466927 + 0.5)
                    ).sum()
                    s_d = (
                        per_d * torch.cos(r_d * 0.5698402909980532 + 1.5)
                    ).sum()
                    rows.append(
                        torch.stack([
                            torch.ones_like(s_sum), s_sum, s_sq, s_t, s_d,
                        ]).to(torch.float64)
                    )
                    any_present = True
        if not any_present:
            return None
        absent = torch.zeros(
            self._PIX_KV_STATS, dtype=torch.float64, device=dev)
        rows = [r if r is not None else absent for r in rows]
        return torch.stack(rows).reshape(len(cache), 2, self._PIX_KV_STATS)

    @staticmethod
    def _pix_kv_rel_delta(a, b):
        """Per-block max RELATIVE fingerprint delta -> ``[num_blocks]``."""
        r = (a - b).abs() / b.abs().clamp_min(1e-8)
        return r.reshape(r.shape[0], -1).max(dim=1).values

    def _pix_kv_blame(self, post, rec, tol: float) -> float:
        """Attribute a mismatch to the recompute that produced it.

        1 = exit-rung recompute, 2 = finish-rung recompute (A23),
        3 = flash recompute, -1 = matches none of them (unknown).
        Probed in the order they would SURVIVE: backward recomputes in
        reverse forward order, so the last writer standing is the
        EARLIEST forward that touched the slots (the exit rung).
        """
        for code, key in ((1, "exit"), (2, "finish"), (3, "flash")):
            cand = rec.get(key)
            if cand is None or cand.shape != post.shape:
                continue
            if float(self._pix_kv_rel_delta(post, cand).max().item()) <= tol:
                return float(code)
        return -1.0

    def _pix_kv_commit_run_verify(self, *, site_code: int) -> dict:
        """Second half of the tripwire: re-fingerprint every recorded slot
        window and compare against the Step-3.4 commit fingerprint.

        Consumes the pending record (one-shot). Returns a metrics dict;
        empty when nothing was pending.
        """
        recs = getattr(self, "_pix_kv_commit_pending", None)
        if not recs:
            return {}
        self._pix_kv_commit_pending = None
        tol = float(getattr(self, "pix_kv_commit_check_tol", 1e-5) or 0.0)
        tag = getattr(self, "_pix_kv_commit_cache_tag", None)
        now = self._pix_kv_cache_tag()
        skipped = {
            "pix_kv_commit_skipped": 1.0,
            "pix_kv_commit_site_code": float(site_code),
        }
        if tag is None or now is None or tag != now:
            # Cache reset / re-allocated between the commit and here.
            # No comparison is possible; emit NO match key.
            self._pix_kv_commit_stats = skipped
            return skipped
        checked = 0
        mismatched = 0
        worst = 0.0
        first_blk = -1
        first_layer = -1
        blame = -1.0
        for rec in recs:
            ref = rec.get("commit")
            if ref is None:
                continue
            post = self._pix_kv_hash_slots(rec["tok_a"], rec["tok_b"])
            if post is None or post.shape != ref.shape:
                continue
            checked += 1
            d = self._pix_kv_rel_delta(post, ref)
            m = float(d.max().item())
            if m > worst:
                worst = m
            if m > tol:
                mismatched += 1
                if first_blk < 0:
                    first_blk = int(rec["block_index"])
                    first_layer = int(torch.argmax(d).item())
                    blame = self._pix_kv_blame(post, rec, tol)
        if checked == 0:
            self._pix_kv_commit_stats = skipped
            return skipped
        out = {
            "pix_kv_commit_match": 1.0 if mismatched == 0 else 0.0,
            "pix_kv_commit_checked_blocks": float(checked),
            "pix_kv_commit_mismatch_blocks": float(mismatched),
            "pix_kv_commit_max_rel_delta": float(worst),
            "pix_kv_commit_site_code": float(site_code),
        }
        if mismatched:
            out["pix_kv_commit_first_mismatch_block"] = float(first_blk)
            out["pix_kv_commit_first_mismatch_layer"] = float(first_layer)
            out["pix_kv_commit_blame"] = blame
            self._pix_warn_once(
                "kv_commit_mismatch",
                "[ActionForcing][A24] pix_kv_commit tripwire FIRED: the K/V "
                "slots written by Step 3.4's context-noise commit did NOT "
                "survive backward. first_mismatch_block=%d layer=%d "
                "blame=%d (1=exit-rung recompute, 2=finish-rung recompute, "
                "3=flash recompute, -1=unknown) max_rel_delta=%.3e "
                "site_code=%d (1=post_backward, 2=deferred_next_call). "
                "Paper 3.3 cross-timestep decoupling is violated for this "
                "step: the next block/step reads a recompute's K/V."
                % (first_blk, first_layer, int(blame), worst, site_code),
            )
        self._pix_kv_commit_stats = out
        return out

    def pix_kv_commit_verify(self) -> dict:
        """POST-BACKWARD half of the KV-commit tripwire — PUBLIC HOOK.

        The pipeline cannot reach a post-backward point on its own, so the
        TRAINER must call this. EXACT CALL SITE (the only correct one):

            ``trainer/causal_action_forcing_train.py``, in
            ``_streaming_step``, IMMEDIATELY after the generator backward

                if gen_should_backward:
                    generator_loss.backward(retain_graph=True)
                out.update(                                   # <-- ADD
                    self.model.inference_pipeline                # <-- ADD
                        .pix_kv_commit_verify())                # <-- ADD

        NOTE the receiver: ``self.model.inference_pipeline`` — the
        PIPELINE object, which is also where the trainer must set
        ``pix_kv_commit_check_every``. Setting the flag on ``self.model``
        is a silent no-op.

        Returns ``{}`` when the flag is off or nothing is pending, so the
        call is safe to leave in unconditionally.
        """
        if int(getattr(self, "pix_kv_commit_check_every", 0) or 0) <= 0:
            return {}
        return self._pix_kv_commit_run_verify(site_code=1)

    # -----------------------------------------------------------------
    # A23 / WP-PIXGAN one-shot notices. Deduped per process so a 600-
    # step run cannot drown in them, but LOUD (warning + stderr) the
    # first time, because every one of these means the pixel critic's
    # fake is not what the design says it is.
    # -----------------------------------------------------------------
    def _pix_warn_once(self, key: str, msg: str) -> None:
        seen = getattr(self, "_pix_finish_grad_warned", None)
        if seen is None:
            seen = set()
            self._pix_finish_grad_warned = seen
        if key in seen:
            return
        seen.add(key)
        import logging as _logging
        import sys as _sys
        _logging.warning(msg)
        print(msg, file=_sys.stderr, flush=True)

    def _warn_once_pix_finish_grad_no_rung(
        self, exit_index: int, num_denoising_steps: int
    ) -> None:
        self._pix_warn_once(
            "no_rung",
            "[ActionForcing][A23] pix_finish_grad_enabled=True but this "
            f"block exited at rung {exit_index} of {num_denoising_steps} "
            "(the LAST rung), so the post-exit finish loop was empty and "
            "there was no finish rung to attach a gradient to. That "
            "block's slice of ``_clean_chunk_grad`` is left at its ZEROS "
            "init (mask False) — it is NOT backfilled with the detached "
            "``cache_pred``, which with flash on would be the t=60 flash "
            "tensor. This "
            "happens ~1/K of the time with a uniform exit-rung draw; to "
            "eliminate it, have the caller reserve the last rung "
            "(generate_and_sync_list(exclude_last_rung=True)) or pin "
            "force_exit_step < K-1. Counted in "
            "``_pix_finish_grad_stats['pix_finish_grad_no_rung']``.",
        )

    def _warn_once_pix_finish_grad_nograd(self) -> None:
        self._pix_warn_once(
            "nograd",
            "[ActionForcing][A23] pix_finish_grad_enabled=True and a "
            "finish rung ran, but its output does NOT require grad — the "
            "generator has no trainable parameters reachable from this "
            "forward. The pixel critic's generator term would be a no-op. "
            "Check that the student (or its LoRA adapters) is unfrozen.",
        )

    def _warn_once_pix_finish_grad_flash_on(self) -> None:
        self._pix_warn_once(
            "flash_on",
            "[ActionForcing][A23] pix_finish_grad_enabled=True together "
            "with flash_dmd_enabled=True. ``_clean_chunk`` (detached) "
            "holds the t=flash_dmd_gan_t FLASH tensor as before, while "
            "``_clean_chunk_grad`` holds the PRE-FLASH ladder endpoint — "
            "the tensor utils/eval_causal_AR.py actually commits and "
            "renders — on its MASK-TRUE frames only, and exact zeros "
            "elsewhere. The two buffers therefore DIFFER in value by "
            "design. Any consumer that wants inference parity (the A23 "
            "fake) must read ``finish_denoised_chunk_grad`` AND select "
            "on ``finish_denoised_chunk_grad_mask``; decoding the whole "
            "buffer will decode zeros for the detached frames. Not "
            "``finish_denoised_chunk`` and not ``flash_dmd_gan_x0``.",
        )

    def _clear_cache_gradients(self) -> None:
        """Detach K/V tensors in the persistent caches so any autograd
        graph held by the gen step's rollout doesn't chain back through
        the critic's no_grad rollout. LongLive parity:
        ``LongLive/model/streaming_training.py:601-626``. Required when
        the cache persists across gen→critic boundaries (Phase-B
        streaming) and cheap insurance for Phase-A's per-iter cache:
        if any K/V tensor was written under ``torch.enable_grad`` (e.g.
        the grad-active exit-rung forward), the autograd graph stays
        attached until the cache slot is overwritten — detaching it
        explicitly releases that graph deterministically.
        """
        if self.kv_cache1 is not None:
            for cache_block in self.kv_cache1:
                k = cache_block.get("k")
                v = cache_block.get("v")
                if k is not None and k.requires_grad:
                    cache_block["k"] = k.detach()
                if v is not None and v.requires_grad:
                    cache_block["v"] = v.detach()
        if self.crossattn_cache is not None:
            for cache_block in self.crossattn_cache:
                k = cache_block.get("k")
                v = cache_block.get("v")
                if k is not None and k.requires_grad:
                    cache_block["k"] = k.detach()
                if v is not None and v.requires_grad:
                    cache_block["v"] = v.detach()
