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
        # Per-iter pipeline: cache sized to the BASELINE rollout
        # window (no headroom). LongLive's streaming pipeline sizes
        # its cache as ``(local_attn + slice_last) * frame_seq_length``
        # to give the rolling cache headroom for seed + in-flight
        # rollout simultaneously, but that doubles per-layer memory
        # which OOMs on a 32GB 5090. Defer the bigger cache to
        # Phase B (persistent streaming state), where it can be gated
        # on a streaming-mode flag.
        return max(self.num_max_frames, self.rollout_frames) * self.frame_seq_length

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

        if sync:
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                indices = torch.randint(
                    low=low,
                    high=sample_high,
                    size=(num_blocks,),
                    device=device,
                )
                if self.last_step_only:
                    indices = torch.ones_like(indices) * (num_denoising_steps - 1)
            else:
                indices = torch.empty(num_blocks, dtype=torch.long, device=device)

            if dist.is_initialized():
                dist.broadcast(indices, src=0)
        else:
            indices = torch.randint(
                low=low,
                high=sample_high,
                size=(num_blocks,),
                device=device,
            )
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
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
        # ``flash_dmd_enabled=True``: every block adds ONE extra graph-on
        # gen forward at ``flash_dmd_gan_t`` (default 60, raw post-warp
        # timestep) AFTER the standard denoise chain finishes. The extra
        # forward's output (= ``flash_dmd_gan_pred`` per block, assembled
        # into ``flash_dmd_gan_output``) is consumed by the GAN adv loss
        # and the gen-side aux losses (LPIPS / MS-SSIM / MANIQA /
        # action_critic). DMD scoring continues to use the random-exit-
        # rung output (= ``denoised_pred``). The K/V slots written by
        # the flash_dmd forward are overwritten by Step 3.4's context-
        # noise commit so the next block's exit-rung forward reads
        # graph-free K/V (paper §3.3 cross-timestep decoupling).
        # The random exit pool now spans ALL rungs (including the last)
        # since GAN no longer reserves the last rung.
        # ``warm_start_init=True``: blocks that have a prior chunk's
        # clean pred (block_index >= 1 within this call, OR block 0
        # when ``initial_prev_clean`` is supplied by the caller) skip
        # rung 0 (= the noisiest level, t≈1000). Their initial state
        # is built by re-noising the prior chunk's clean pred at rung
        # 1's t (= second-noisiest, t≈625) and they denoise via the
        # shortened ladder rungs [1..N-1]. The very first block in
        # the rollout (block 0 with no caller-provided seed) keeps
        # the cold init: pure Gaussian noise + full ladder starting
        # at rung 0. See user spec — "first chunk in the rollout:
        # unchanged; second chunk onward warm-starts".
        # Sample exit flags. Cold-start only — every block uses the
        # full ladder (low=0). Warm-start removed.
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
        # cache_pred (refined when flash_dmd_enabled, else post-rung).
        # Detached, no autograd graph. Sized to num_output_frames so
        # block writes use absolute current_start_frame indexing.
        # Allocated only when ``flash_dmd_enabled`` (the regime where
        # the aux pass consumes it); ``None`` otherwise lets the aux
        # pass fall back to the legacy ``_streaming_build_clean_x_self``
        # path on baselines without flash.
        if flash_dmd_enabled:
            clean_chunk = torch.zeros(
                [batch_size, num_output_frames, num_channels, height, width],
                device=noise.device,
                dtype=noise.dtype,
            )
        else:
            clean_chunk = None
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
                        def _exit_fn(
                            x,
                            _gen=self.generator,
                            _cond=block_cond,
                            _t=timestep,
                            _kv=self.kv_cache1,
                            _xa=self.crossattn_cache,
                            _start=current_start_frame * self.frame_seq_length,
                        ):
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
                    def _flash_fn(
                        x,
                        _gen=self.generator,
                        _cond=block_cond,
                        _t=flash_t_step,
                        _kv=self.kv_cache1,
                        _xa=self.crossattn_cache,
                        _start=current_start_frame * self.frame_seq_length,
                    ):
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
            #     (DMD-grad) forward reads detached K/V. The input
            #     ``cache_pred`` here has been refined by Step 3.3.5
            #     when ``flash_dmd_enabled``.
            # The input is always detached (cache_pred came from a
            # no_grad chain anyway, but the explicit detach releases
            # any autograd nodes early — memory hygiene).
            commit_input_clean = cache_pred.detach()
            context_timestep = torch.full_like(timestep, self.context_noise)
            cache_commit_input = self.scheduler.add_noise(
                commit_input_clean.flatten(0, 1),
                torch.randn_like(commit_input_clean.flatten(0, 1)),
                context_timestep.flatten(0, 1),
            ).unflatten(0, commit_input_clean.shape[:2])
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

        batch_size, num_frames, _, _, _ = noise.shape
        npb = self.num_frame_per_block
        cps = self.chunks_per_rolling_step

        if num_frames % npb != 0:
            raise RuntimeError(
                f"num_frames ({num_frames}) must be divisible by "
                f"num_frame_per_block ({npb})."
            )
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
        # Aux-teacher clean_x buffer (per-block post-Step-3.3.5 cache_pred,
        # detached). See ``inference_with_trajectory`` for rationale.
        clean_chunk = (
            torch.zeros_like(noise) if flash_dmd_enabled else None
        )

        num_denoising_steps = len(self.denoising_step_list)
        # Cold-start only (warm_start removed). Every block uses the
        # full denoising ladder; exit rung sampled per block.
        exit_flags = self.generate_and_sync_list(
            len(all_num_frames), num_denoising_steps, device=noise.device,
            sync=sync_exit_flags, force_exit_step=force_exit_step,
            exclude_last_rung=False,
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

            # Cold-start: pure noise input + full denoising ladder.
            noisy_input = noise[
                :, block_start_in_noise: block_start_in_noise + current_num_frames,
            ]

            # Rolling denoise loop with truncated random exit, full
            # ladder.
            exit_index = num_denoising_steps - 1
            for index in range(0, num_denoising_steps):
                current_timestep = self.denoising_step_list[index]
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
                        def _exit_fn(
                            x,
                            _gen=self.generator,
                            _cond=block_cond,
                            _t=timestep,
                            _kv=self.kv_cache1,
                            _xa=self.crossattn_cache,
                            _start=current_start_frame * self.frame_seq_length,
                        ):
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

            # Post-exit no_grad chain through remaining rungs. Ends
            # with ``cache_pred`` = clean x0 estimate at the last rung.
            cache_pred = denoised_pred.detach()
            for j in range(exit_index + 1, num_denoising_steps):
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
                with torch.no_grad():
                    _, cache_pred = self.generator(
                        noisy_image_or_video=cache_input,
                        conditional_dict=block_cond,
                        timestep=step_t,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )

            # Flash-DMD t=flash_dmd_gan_t grad-on forward (per block,
            # no final-block restriction). Inputs: noised cache_pred
            # at flash_dmd_gan_t. Output goes to ``flash_dmd_gan_output``
            # for the GAN / aux losses; K/V slots overwritten by
            # Step 3.4 below for paper §3.3 cross-timestep decoupling.
            flash_dmd_pred: Optional[torch.Tensor] = None
            if flash_dmd_enabled:
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
                    def _flash_fn(
                        x,
                        _gen=self.generator,
                        _cond=block_cond,
                        _t=flash_t_step,
                        _kv=self.kv_cache1,
                        _xa=self.crossattn_cache,
                        _start=current_start_frame * self.frame_seq_length,
                    ):
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

            # Cache-update commit at t=context_noise. Runs in BOTH
            # modes — see ``inference_with_trajectory`` for the full
            # rationale (paper §3.3 K/V decoupling: must overwrite
            # the grad-attached K/V slots written by the Flash-DMD
            # forward with graph-free K/V so the next block's exit-
            # rung forward doesn't pull DMD's grad through the prior
            # block's Flash-DMD gen forward).
            commit_input_clean = cache_pred.detach()
            context_timestep = torch.full_like(timestep, self.context_noise)
            cache_commit = self.scheduler.add_noise(
                commit_input_clean.flatten(0, 1),
                torch.randn_like(commit_input_clean.flatten(0, 1)),
                context_timestep.flatten(0, 1),
            ).unflatten(0, commit_input_clean.shape[:2])
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=cache_commit,
                    conditional_dict=block_cond,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )

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

        return output, denoised_t_from, denoised_t_to

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
