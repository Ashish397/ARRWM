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
        mae_extension_threshold: Optional[float] = None,
        mae_extension_max_extra_chunks: int = 0,
        context_noise: int = 0,
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

        # MAE-driven dynamic rollout extension. ``mae_extension_threshold``
        # is the upper bound on the latent-space mean-absolute-error of
        # the last 3 generated frames vs GT; below this we roll one
        # more 3-frame chunk (no_grad, full denoise). ``max_extra_chunks``
        # bounds how many extra chunks we'll roll per call (a COMPUTE
        # cap, not a memory cap — extension chunks commit through the
        # rolling cache, see ``kv_cache_size`` below). Set the threshold
        # to ``None`` to disable extensions entirely; this is genuinely
        # free since the cache buffer never grows for extensions.
        self.mae_extension_threshold: Optional[float] = (
            float(mae_extension_threshold)
            if mae_extension_threshold is not None
            else None
        )
        self.mae_extension_max_extra_chunks = int(mae_extension_max_extra_chunks)
        if self.mae_extension_max_extra_chunks < 0:
            raise ValueError(
                f"mae_extension_max_extra_chunks must be >= 0; got "
                f"{self.mae_extension_max_extra_chunks}"
            )

        # Per-call metrics dict, populated by ``inference_with_trajectory``.
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

        self.kv_cache1: Optional[list] = None
        self.crossattn_cache: Optional[list] = None

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

    @property
    def frame_seq_length(self) -> int:
        extra = int(getattr(self._inner_model(), "action_tokens_per_frame", 0))
        return self._spatial_frame_seq_length + extra

    @property
    def kv_cache_size(self) -> int:
        # KV buffer is sized to the BASELINE rollout window only —
        # ``max(num_max_frames, rollout_frames)``. We deliberately do
        # NOT add headroom for MAE-driven extensions: the model-side
        # rolling cache (``utils/infinity_rope.py:213-219``,
        # ``wan/modules/causal_model.py:302-306``) implements
        # sink-preserving FIFO eviction, so extension chunks just
        # commit through the rolling buffer (the first
        # ``sink_size`` frames stay pinned, oldest non-sink frames
        # are evicted as new ones arrive). This keeps memory cost
        # FIXED regardless of ``mae_extension_max_extra_chunks``,
        # makes the documented "set threshold=null to disable" path
        # genuinely free, and matches the user's invariant: "we
        # always do max 21 frames of cache, that's why we have the
        # tail config".
        return max(self.num_max_frames, self.rollout_frames) * self.frame_seq_length

    # -----------------------------------------------------------------
    # Lockstep exit-flag selection (rank 0 rolls, broadcast to all)
    # -----------------------------------------------------------------
    def generate_and_sync_list(
        self, num_blocks: int, num_denoising_steps: int, device: torch.device
    ) -> List[int]:
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(num_blocks,),
                device=device,
            )
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        if dist.is_initialized():
            dist.broadcast(indices, src=0)
        return indices.tolist()

    # -----------------------------------------------------------------
    # Main entry: inference_with_trajectory
    # -----------------------------------------------------------------
    def inference_with_trajectory(
        self,
        noise: torch.Tensor,
        clean_image_or_video: Optional[torch.Tensor] = None,
        initial_latent: Optional[torch.Tensor] = None,
        gt_latents: Optional[torch.Tensor] = None,
        enable_mae_extension: bool = False,
        return_sim_step: bool = False,
        seed_latents: Optional[torch.Tensor] = None,
        prefer_cache_pred_in_output: bool = False,
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
                extension loop might roll into. ``F_gt`` must be at
                least ``rollout_frames``; extensions are bounded
                additionally by ``F_gt - rollout_frames`` (i.e. they
                stop when GT runs out). Pass ``None`` to disable
                extensions and skip MAE metric collection.
            enable_mae_extension: gates the extension loop. The
                generator step passes ``True``; the critic step
                passes ``False`` to avoid a duplicate (gradient-free)
                rollout that would only produce the same metrics
                the generator step already logged.
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
                ``rollout_frames``; extension mode covers up to
                ``rollout_frames + max_extra_chunks*npb``.

        Returns:
            ``(output, denoised_timestep_from, denoised_timestep_to)``
            where ``output`` is shape ``[B, rollout_frames, C, H, W]``
            (BASELINE only — extensions are not included so the DMD
            scoring path is unaffected). MAE-extension metrics live on
            ``self._last_extension_metrics`` after the call.
        """
        if seed_latents is not None and initial_latent is not None:
            raise ValueError(
                "seed_latents and initial_latent are mutually exclusive; "
                "Phase-1 Action-Forcing uses seed_latents (cf-frame KV "
                "prefill), the i2v initial_latent path is legacy."
            )
        # Reset per-call metrics (NaN = not computed; gets overwritten
        # in the extension path when gt_latents is available).
        nan_f = float("nan")
        self._last_extension_metrics = {
            "mae_extension_count": 0,
            "last_chunk_mae": nan_f,
            "baseline_last_chunk_mae": nan_f,
        }

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
        # = clean GT context that the ODE student was trained to see as
        # ``clean_x``. Without this prefill the rolling rollout starts
        # cold and the ODE student (trained ONLY teacher-forced) produces
        # garbage. We seed in chunks of ``npb`` frames each, with one
        # forward at ``timestep=0`` per chunk plus the standard
        # ``context_noise`` cache-update forward — same shape as the
        # regular per-block path so the KV cache state ends up identical
        # to "rolling at clean GT input" through the seed window.
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
                seed_t = torch.zeros(
                    [batch_size, npb], device=noise.device, dtype=torch.int64,
                )
                seed_block_cond = _slice_per_frame_streams(
                    conditional_dict,
                    frame_start=current_start_frame,
                    frame_count=npb,
                )
                with torch.no_grad():
                    self.generator(
                        noisy_image_or_video=seed_chunk,
                        conditional_dict=seed_block_cond,
                        timestep=seed_t,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )
                # Context-noise commit forward, mirroring the rollout
                # loop's unconditional commit at line ~635 so the seed
                # window's K/V ends up at the same noise level as every
                # rollout chunk's K/V. At ``context_noise=0`` this is a
                # no-op repetition (``add_noise(seed, n, 0) == seed``,
                # so the t=0 forward writes the same K/V already
                # produced by the line-491 forward) — kept unconditional
                # for parity with the rollout, so bumping
                # ``context_noise > 0`` later doesn't silently leave the
                # seed's K/V at t=0 while the rollout's is at t=context_noise.
                ctx_t = torch.full_like(seed_t, self.context_noise)
                seed_ctx_in = self.scheduler.add_noise(
                    seed_chunk.flatten(0, 1),
                    torch.randn_like(seed_chunk.flatten(0, 1)),
                    ctx_t.flatten(0, 1),
                ).unflatten(0, seed_chunk.shape[:2])
                with torch.no_grad():
                    self.generator(
                        noisy_image_or_video=seed_ctx_in,
                        conditional_dict=seed_block_cond,
                        timestep=ctx_t,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )
                current_start_frame += npb

        # Step 3: Per-rolling-step denoise loop with truncated random-exit.
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(
            len(all_num_frames), num_denoising_steps, device=noise.device
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
            noisy_input = noise[
                :,
                block_start_in_noise: block_start_in_noise + current_num_frames,
            ]
            block_cond = _slice_per_frame_streams(
                conditional_dict,
                frame_start=current_start_frame,
                frame_count=current_num_frames,
            )

            # Step 3.1: Truncated denoise loop.
            for index, current_timestep in enumerate(self.denoising_step_list):
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
                    if current_start_frame < start_gradient_frame_index:
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
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=block_cond,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                        )
                    exit_index = index
                    break

            # Step 3.2: Finish the denoising chain past the random exit
            # rung under ``no_grad``. ``denoised_pred`` from the exit
            # forward is the model's x0 estimate AT the random exit
            # timestep — used by DMD as the grad-active output. For the
            # cache-update we want a CLEAN x0 estimate so subsequent
            # chunks attend to clean K/V (no noise compounding across
            # the rolling cache). Continue stepping noisiest→cleanest
            # from the exit rung's ``next_t`` to the last rung, all
            # under ``no_grad``. ``cache_pred`` ends up at the last
            # rung's x0 estimate (= same as ``last_step_only=True``
            # would have produced for the cache, but with the random-
            # exit gradient signal preserved in ``denoised_pred``).
            cache_pred = denoised_pred.detach()
            num_rungs = len(self.denoising_step_list)
            for j in range(exit_index + 1, num_rungs):
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

            # Step 3.4: Cache-update forward at context_noise, using
            # the FULLY-denoised ``cache_pred`` so the K/V committed
            # to the rolling cache is a clean x0 estimate (no
            # noise compounding for downstream chunks).
            context_timestep = torch.full_like(timestep, self.context_noise)
            cache_pred = self.scheduler.add_noise(
                cache_pred.flatten(0, 1),
                torch.randn_like(cache_pred.flatten(0, 1)),
                context_timestep.flatten(0, 1),
            ).unflatten(0, cache_pred.shape[:2])
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=cache_pred,
                    conditional_dict=block_cond,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )

            current_start_frame += current_num_frames

        # Step 3.4: MAE metrics + dynamic rollout extension.
        # We always compute the baseline last-chunk MAE if ``gt_latents``
        # is available (caller wants this on wandb regardless of
        # whether extensions are enabled). Extensions then fire ONLY
        # when ``enable_mae_extension`` is True AND the threshold is
        # set AND there's GT room to grow into AND the extension cap
        # isn't already 0.
        npb = self.num_frame_per_block
        baseline_can_compute_mae = (
            gt_latents is not None
            and denoised_pred is not None
            and gt_latents.shape[1] >= current_start_frame
        )
        if baseline_can_compute_mae:
            baseline_last_mae_value = self._compute_chunk_mae(
                pred_chunk=output[
                    :, current_start_frame - npb: current_start_frame
                ].detach(),
                gt_chunk=gt_latents[
                    :, current_start_frame - npb: current_start_frame
                ],
            )
            self._last_extension_metrics["baseline_last_chunk_mae"] = (
                baseline_last_mae_value
            )
            self._last_extension_metrics["last_chunk_mae"] = (
                baseline_last_mae_value
            )

        extension_active = (
            enable_mae_extension
            and self.mae_extension_threshold is not None
            and self.mae_extension_max_extra_chunks > 0
            and gt_latents is not None
            and baseline_can_compute_mae
        )
        if extension_active:
            threshold = float(self.mae_extension_threshold)
            extension_count = 0
            last_chunk_mae_value = (
                self._last_extension_metrics["baseline_last_chunk_mae"]
            )

            # DDP-safe extension cap. ``_compute_chunk_mae`` performs an
            # ``all_reduce`` inside the loop, so EVERY rank must enter and
            # exit it the same number of times. Different ranks can have
            # DIFFERENT GT lengths (rides aren't truncated to a fixed
            # length — see ``causal_action_forcing_train._load_ride_tensors``
            # called with ``max_frames=None``), so a rank-local stop
            # condition like ``current_start + npb <= gt_latents.shape[1]``
            # would let one rank exit while peers wait forever on the next
            # all-reduce.
            #
            # Fix: reduce the per-rank "how many more 3-frame chunks of GT
            # do I have left" with ``ReduceOp.MIN`` BEFORE the loop, then
            # use that synced count as the only loop bound. The MAE-vs-
            # threshold check is already lockstep (averaged across ranks
            # in ``_compute_chunk_mae``).
            local_available = max(
                0, (gt_latents.shape[1] - current_start_frame) // npb
            )
            avail_t = torch.tensor(
                [local_available],
                device=gt_latents.device,
                dtype=torch.long,
            )
            if dist.is_initialized():
                dist.all_reduce(avail_t, op=dist.ReduceOp.MIN)
            synced_available = int(avail_t.item())
            max_iters = min(
                self.mae_extension_max_extra_chunks, synced_available
            )

            while (
                extension_count < max_iters
                and last_chunk_mae_value == last_chunk_mae_value  # NaN guard
                and last_chunk_mae_value < threshold
            ):
                ext_denoised = self._run_extension_block(
                    batch_size=batch_size,
                    num_channels=num_channels,
                    height=height,
                    width=width,
                    current_start_frame=current_start_frame,
                    conditional_dict=conditional_dict,
                    device=noise.device,
                    dtype=noise.dtype,
                )
                last_chunk_mae_value = self._compute_chunk_mae(
                    pred_chunk=ext_denoised,
                    gt_chunk=gt_latents[
                        :, current_start_frame: current_start_frame + npb
                    ],
                )
                extension_count += 1
                current_start_frame += npb

            self._last_extension_metrics["mae_extension_count"] = (
                extension_count
            )
            self._last_extension_metrics["last_chunk_mae"] = (
                last_chunk_mae_value
            )

        # Step 3.5: Engineering trick — derive the timestep range we
        # supervised in this rollout (used by DMD's ts_schedule clamp).
        denoised_timestep_from: Optional[int]
        denoised_timestep_to: Optional[int]
        if not self.same_step_across_blocks:
            denoised_timestep_from = None
            denoised_timestep_to = None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = self._round_to_grid(
                self.denoising_step_list[exit_flags[0]]
            )
        else:
            denoised_timestep_to = self._round_to_grid(
                self.denoising_step_list[exit_flags[0] + 1]
            )
            denoised_timestep_from = self._round_to_grid(
                self.denoising_step_list[exit_flags[0]]
            )

        # Slice off the seed prefix (KV prefill window) before
        # returning. The caller's scoring window is the rollout-only
        # region, frames [num_input_frames + num_seed_frames :].
        if num_seed_frames > 0 or num_input_frames > 0:
            output = output[:, num_input_frames + num_seed_frames:]

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

    def _run_extension_block(
        self,
        batch_size: int,
        num_channels: int,
        height: int,
        width: int,
        current_start_frame: int,
        conditional_dict: dict,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Run one ``num_frame_per_block``-frame extension chunk.

        Always under ``no_grad`` (extensions never carry gradient — the
        DMD scoring window is the BASELINE rollout's last
        ``num_max_frames`` frames). Full denoise: every rung in
        ``denoising_step_list`` runs, no random exit. After the final
        denoise, a t=context_noise forward commits the K/V to the
        cache exactly like the baseline blocks, so subsequent extensions
        see the new chunk as committed context.

        Returns the final denoised prediction
        ``[B, num_frame_per_block, C, H, W]`` for the caller's MAE
        check.
        """
        npb = self.num_frame_per_block
        block_cond = _slice_per_frame_streams(
            conditional_dict,
            frame_start=current_start_frame,
            frame_count=npb,
        )

        noisy_input = torch.randn(
            [batch_size, npb, num_channels, height, width],
            device=device, dtype=dtype,
        )

        num_denoising_steps = len(self.denoising_step_list)
        denoised_pred: Optional[torch.Tensor] = None
        timestep: Optional[torch.Tensor] = None

        with torch.no_grad():
            for index, current_timestep in enumerate(self.denoising_step_list):
                ts_value = int(round(float(current_timestep)))
                timestep = torch.full(
                    [batch_size, npb], ts_value,
                    device=device, dtype=torch.int64,
                )
                _, denoised_pred = self.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=block_cond,
                    timestep=timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                if index < num_denoising_steps - 1:
                    next_t_value = int(round(float(
                        self.denoising_step_list[index + 1]
                    )))
                    flat = denoised_pred.flatten(0, 1)
                    noisy_input = self.scheduler.add_noise(
                        flat,
                        torch.randn_like(flat),
                        next_t_value * torch.ones(
                            [batch_size * npb],
                            device=device, dtype=torch.long,
                        ),
                    ).unflatten(0, denoised_pred.shape[:2])

            assert denoised_pred is not None and timestep is not None
            context_timestep = torch.full_like(timestep, self.context_noise)
            cache_pred = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep.flatten(0, 1),
            ).unflatten(0, denoised_pred.shape[:2])
            self.generator(
                noisy_image_or_video=cache_pred,
                conditional_dict=block_cond,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

        return denoised_pred.detach()

    # -----------------------------------------------------------------
    # Cache initializers (CF-shape: [B, kv_cache_size, 12, 128]).
    # -----------------------------------------------------------------
    def _initialize_kv_cache(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> None:
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
