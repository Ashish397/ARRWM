"""Phase-1 Action-Forcing DMD trainer (NO STAIRCASE).

Subclasses ``RollingStaircaseDMDTrainer`` to reuse the heavy
infrastructure (DDP setup, ZarrRideDataset, action teacher plumbing,
W&B init, EMA, checkpoint save/resume, vis recorder), but overrides
the model/pipeline construction and the training loop to a CF-style
recipe:

  Per training iter (CF parity):
    - Pull a ``num_training_frames``-long ride (default 21 frames)
      from the dataset.
    - Build per-frame action conditioning (Stream A modulation + Stream
      B tokens) from the ride's GT actions. NO decay, NO attenuation —
      every frame gets its actual GT action, exactly mirroring the
      ODE student's training distribution.
    - Run ``model.generator_loss(...)`` (CF-parity DMD on the student's
      pred, no GT context, real/fake symmetric on noisy student).
    - ``generator_loss.backward()`` -> generator optimizer step + EMA.
    - Pull another ride.
    - Run ``model.critic_loss(...)`` (CF-parity fake_score denoise loss
      on the student's pred). ``critic_loss.backward()`` -> fake_score
      optimizer step.

The dataset is constrained to rides with at least
``num_training_frames`` latent frames; we slice the leading
``num_training_frames`` frames + corresponding actions for every ride.
"""
from __future__ import annotations

import gc
import logging
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import wandb  # type: ignore
    _HAS_WANDB = True
except Exception:
    wandb = None  # type: ignore
    _HAS_WANDB = False

from trainer.causal_rolling_staircase_train import (
    RollingStaircaseDMDTrainer,
    _load_ride_tensors,
)
from model.dmd_action_forcing import ActionForcingDMD
from model.r3gan import (
    R3GANDiscriminator3D,
    rpgan_d_loss,
    rpgan_g_loss,
    r1_penalty,
    r2_penalty,
)
from pipeline.action_forcing_training import ActionForcingTrainingPipeline
from utils.infinity_rope import install as _install_infinity_rope


def _frames_to_mp4_bytes(frames: np.ndarray, fps: float = 5.0) -> Optional[bytes]:
    """Encode uint8ndarray ``[T, H, W, 3]`` to mp4 bytes via ffmpeg.

    Mirrors the helper in ``trainer/causal_diffusion_teacher_train.py``.
    Returns ``None`` on any subprocess / encode failure so the caller
    can skip the wandb upload silently. Best-effort: never raises.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        return None
    h, w = int(frames.shape[1]), int(frames.shape[2])
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-f", "mp4",
        "-movflags", "frag_keyframe+empty_moov",
        "pipe:1",
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out_bytes, _ = proc.communicate(
            input=frames.tobytes(), timeout=120,
        )
        if proc.returncode == 0 and len(out_bytes) > 0:
            return out_bytes
    except Exception:
        pass
    return None


def _chunk_actions(per_frame: torch.Tensor, chunk_frames: int) -> torch.Tensor:
    """Mean-pool ``[B, F, D]`` into ``[B, n_chunks, D]`` (drop tail).

    Mirrors ``trainer.causal_diffusion_teacher_train._chunk_actions``;
    duplicated rather than imported to keep the trainer's dependency
    surface tight (``causal_diffusion_teacher_train.py`` pulls in
    motion-pipeline imports we don't want at module load time when
    aux losses are disabled).
    """
    B, F_len, D = per_frame.shape
    n_chunks = F_len // chunk_frames
    trimmed = per_frame[:, : n_chunks * chunk_frames]
    return trimmed.reshape(B, n_chunks, chunk_frames, D).mean(dim=2)


class ActionForcingDMDTrainer(RollingStaircaseDMDTrainer):
    """CF-style DMD2 trainer for the Action-Forcing (no-staircase) Phase-1 recipe.

    Auxiliary losses (action critic + frozen-teacher z guidance) are
    OFF by default. Enable with ``action_critic_aux_enabled: true``
    AND ``action_teacher_mode: all`` (or ``slot0``) in the YAML.
    When enabled the trainer:

      - un-freezes ``self.model.action_critic`` (the model
        instantiates it frozen for ODE checkpoint compatibility),
        wraps it in DDP, and builds a separate AdamW for it;
      - inherits the parent staircase trainer's
        ``_build_action_teacher`` so ``_frozen_cotracker`` /
        ``_frozen_ss_vae`` are loaded; and
      - on every generator step computes teacher z via
        ``_compute_teacher_z_per_slot`` (parent), runs
        ``critic_updates_per_step`` independent critic optim steps,
        then adds the gen-side z-guidance loss to the DMD loss
        before the generator backward.

    State-probe aux losses are NOT wired here on purpose — Phase-1
    Action-Forcing's contract is "DMD + action critic only". The
    state probe head is still instantiated (for ODE checkpoint
    parity) and remains frozen.

    R3GAN (RpGAN + R1 + R2) is OPT-IN via ``gan_enabled: true``. When
    enabled the trainer:

      - builds ``model.r3gan.R3GANDiscriminator3D`` on the latent
        video shape and DDP-wraps it (separate from the ``fake_score``
        / ``action_critic`` DDPs);
      - builds a dedicated AdamW (``gan_lr``, ``gan_betas``,
        ``gan_weight_decay``);
      - on every generator step computes:
            real = ride["latents"][:, gen_window]   (GT latent video)
            fake = aux["pred_image"]                (student rollout
                                                     last num_training_
                                                     frames)
        and runs:
            * D-update (RpGAN-D + R1 on real + R2 on fake; backward
              into D only — fake is detached so no flow into G);
            * G's RpGAN term added to the generator loss (graph flows
              through ``pred_image`` -> generator);
      - persists / resumes ``r3gan_discriminator`` +
        ``r3gan_optimizer`` state in the checkpoint file.

    See ``model/r3gan.py`` for the loss / discriminator code and
    ``configs/action_forcing_phase1_aux_gan.yaml`` for the
    intended hyper-parameters.

    SC-DMD (Salt paper, arXiv 2604.03118v1) is OPT-IN via
    ``sc_dmd_enabled: true``. When enabled the trainer adds the
    semigroup defect regularizer
        L_SC = E[||Ψ_θ^{ts→te}(x_ts) - Ψ_θ^{tm→te}(Ψ_θ^{ts→tm}(x_ts))||²]
    to the generator loss. This penalizes the gap between the model's
    direct one-step denoising (``t_s → t_e``) and the same denoising
    composed through an intermediate rung ``t_m``, addressing DMD's
    compositionality deficit in multi-step inference.

    Implementation lives entirely on the model
    (``ActionForcingDMD.sc_dmd_loss`` — single chunk, fresh KV cache,
    2 extra DiT forwards on a 3-frame chunk per gen step ≈ 1-2%
    extra wallclock). The trainer threads weight + warmup config and
    adds the weighted SC loss to the unified generator backward
    alongside DMD / aux / GAN. SC works with or without aux/GAN
    enabled (independent loss).
    """

    # ------------------------------------------------------------------
    # Model & pipeline construction (replace staircase model + pipeline).
    # ------------------------------------------------------------------
    def _build_model(self) -> None:
        args = self.config
        if not hasattr(args, "text_pre_encoded"):
            OmegaConf.update(args, "text_pre_encoded", True, merge=True)
        if not hasattr(args, "mixed_precision"):
            OmegaConf.update(args, "mixed_precision", self.use_mixed_precision, merge=True)

        model = ActionForcingDMD(args=args, device=self.device)
        # Move sub-modules to the target device + dtype (mirroring the
        # staircase trainer's setup).
        model.generator.model.to(device=self.device, dtype=self.dtype)
        model.fake_score.model.to(device=self.device, dtype=self.dtype)
        model.real_score.model.to(device=self.device, dtype=self.dtype)
        if model.action_projection is not None:
            model.action_projection.to(device=self.device, dtype=self.dtype)
        if model.action_token_projection is not None:
            model.action_token_projection.to(device=self.device, dtype=self.dtype)
        if getattr(model, "state_probe", None) is not None:
            model.state_probe.to(device=self.device, dtype=self.dtype)
        if getattr(model, "action_critic", None) is not None:
            model.action_critic.to(device=self.device, dtype=self.dtype)
        if getattr(model, "vae", None) is not None:
            try:
                model.vae.to(device=self.device)
            except AttributeError:
                inner_vae = getattr(model.vae, "model", None)
                if inner_vae is not None:
                    inner_vae.to(device=self.device)
        self.model = model

        debug_fup = bool(getattr(self.config, "debug_find_unused_parameters", False))
        self.generator_ddp: Optional[DDP] = None
        self.fake_score_ddp: Optional[DDP] = None
        if self.world_size > 1:
            self.generator_ddp = DDP(
                model.generator.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=debug_fup,
                broadcast_buffers=False,
            )
            model.generator.model = self.generator_ddp  # type: ignore

            if bool(getattr(self.config, "fake_score_updates_enabled", True)):
                self.fake_score_ddp = DDP(
                    model.fake_score.model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=debug_fup,
                    broadcast_buffers=False,
                )
                model.fake_score.model = self.fake_score_ddp  # type: ignore

        # ------------------------------------------------------------------
        # Auxiliary action critic (CF-parity ActionCritic, frozen teacher
        # supervised). The model's ``_build_action_aux_heads_compat``
        # instantiated the critic in ``eval()`` + ``requires_grad_(False)``
        # mode purely so the ODE checkpoint's critic state_dict can be
        # loaded without losing keys. When aux losses are TURNED ON via
        # config we un-freeze it here, set train(), and DDP-wrap it. The
        # critic's optimizer is built in ``_build_optimizer``.
        #
        # The "aux losses active" flag requires BOTH ``action_critic_
        # aux_enabled=true`` AND ``action_teacher_mode != "off"``. The
        # base ``configs/action_forcing_phase1.yaml`` has the head
        # enabled (for ODE checkpoint compat) but the teacher off, so
        # the aux loss path stays cold there — no DDP-wrap, no critic
        # optimizer, no behavioural change vs the pre-aux trainer.
        # The aux variant ``configs/action_forcing_phase1_aux.yaml``
        # enables both and gets the full critic + teacher path.
        # ------------------------------------------------------------------
        ac_aux_enabled_cfg = bool(
            getattr(self.config, "action_critic_aux_enabled", False)
        )
        teacher_mode_cfg = str(
            getattr(self.config, "action_teacher_mode", "off") or "off"
        ).strip().lower()
        self.action_critic_aux_enabled = ac_aux_enabled_cfg
        self.action_critic_loss_active = (
            ac_aux_enabled_cfg and teacher_mode_cfg != "off"
        )
        self.action_critic_ddp: Optional[DDP] = None
        if self.action_critic_loss_active:
            ac = getattr(model, "action_critic", None)
            if ac is None:
                raise RuntimeError(
                    "action_critic_aux_enabled=true but "
                    "model.action_critic is None — set "
                    "action_critic_aux_enabled=true in the YAML so "
                    "ActionForcingDMD instantiates the head, or set "
                    "action_critic_aux_enabled=false to disable the "
                    "aux loss path."
                )
            ac.requires_grad_(True)
            ac.train()
            if self.world_size > 1:
                self.action_critic_ddp = DDP(
                    ac,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=False,
                    broadcast_buffers=False,
                )
                # Keep ``model.action_critic`` pointing at the DDP
                # wrapper so all call sites (loss path, optim, EMA,
                # checkpoint save) see the same handle. The wrapped
                # ``.module`` still exposes the underlying ActionCritic
                # for state_dict save.
                model.action_critic = self.action_critic_ddp  # type: ignore

        # ------------------------------------------------------------------
        # R3GAN — RpGAN + R1 + R2 discriminator (opt-in via ``gan_enabled``).
        # Built in fp32 to keep the gradient-penalty (second-order)
        # arithmetic numerically clean. The discriminator operates on
        # the same latent video tensor the student predicts; both real
        # (GT) and fake (``pred_image``) come from the trainer's
        # ride / aux dict, so the disc never has to know about the
        # RoPE-cache / scheduler state of the generator.
        # ------------------------------------------------------------------
        self.gan_enabled = bool(getattr(self.config, "gan_enabled", False))
        self.r3gan_disc: Optional[torch.nn.Module] = None
        self.r3gan_disc_ddp: Optional[DDP] = None
        if self.gan_enabled:
            in_channels = int(
                getattr(self.config, "gan_disc_in_channels", 16)
            )
            base_channels = int(
                getattr(self.config, "gan_disc_base_channels", 64)
            )
            num_blocks = int(
                getattr(self.config, "gan_disc_num_blocks", 4)
            )
            disc = R3GANDiscriminator3D(
                in_channels=in_channels,
                base_channels=base_channels,
                num_blocks=num_blocks,
            )
            disc.to(device=self.device, dtype=torch.float32)
            disc.train()
            self.r3gan_disc = disc
            if self.world_size > 1:
                self.r3gan_disc_ddp = DDP(
                    disc,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=False,
                    broadcast_buffers=False,
                )
            if self.is_main_process:
                n_params = sum(p.numel() for p in disc.parameters())
                logging.info(
                    "[ActionForcing] R3GAN discriminator built: "
                    "in_channels=%d base_channels=%d num_blocks=%d "
                    "params=%.2fM (DDP=%s)",
                    in_channels, base_channels, num_blocks,
                    n_params / 1e6,
                    self.r3gan_disc_ddp is not None,
                )

        # ------------------------------------------------------------------
        # SC-DMD (Salt) — semigroup defect regularizer. Default OFF.
        # No new modules / optimizers needed; the SC pass shares the
        # existing generator's parameters via the same DDP-wrapped
        # ``model.generator``. We only thread a config flag, weight,
        # and warmup into the gen-loss compute path. Triplet sampling
        # and the two extra DiT forwards live on
        # ``ActionForcingDMD.sc_dmd_loss``; this trainer just calls it
        # and adds the weighted loss to the unified generator backward.
        # ------------------------------------------------------------------
        self.sc_dmd_enabled = bool(
            getattr(self.config, "sc_dmd_enabled", False)
        )
        self.sc_dmd_loss_weight = float(
            getattr(self.config, "sc_dmd_loss_weight", 0.05)
        )
        self.sc_dmd_warmup_steps = int(
            getattr(self.config, "sc_dmd_warmup_steps", 0)
        )
        if self.sc_dmd_enabled and self.is_main_process:
            logging.info(
                "[ActionForcing] SC-DMD (Salt) ENABLED: weight=%.4f "
                "warmup_steps=%d (loss multiplier ramps linearly from "
                "0 to ``weight`` over ``warmup_steps`` steps; SC pass "
                "costs ~2 extra DiT forwards on a ``num_frame_per_block``-"
                "frame chunk per gen step).",
                self.sc_dmd_loss_weight,
                self.sc_dmd_warmup_steps,
            )

        # ------------------------------------------------------------------
        # dmd_context (v14 teacher-forcing parity) — single source of truth
        # is the model. ``self.model.dmd_context_clean_frames`` is the
        # KV-cache seed prefill size (= 9 by default, configurable via
        # the ``dmd_context_clean_frames`` knob the model reads from
        # ``args``). The trainer just reads it back to size its ride
        # slices. The clean/noisy shift used inside DMD scoring is fixed
        # at ``num_frame_per_block`` (= 1 chunk) and lives entirely in
        # the model — the trainer never sees it.
        # ------------------------------------------------------------------

        # Collapse gating: the threshold is consulted when deciding
        # whether to EXTEND a rollout (multi-batch on the same ride)
        # vs swap to a fresh ride next iter. ``MAE > threshold`` ⇒
        # student has collapsed on this ride, so don't extend; just
        # train on the current rollout and let the next iter pull a
        # new ride. ``MAE <= threshold`` ⇒ keep rolling on this ride.
        # In streaming mode this gates the per-iter "advance same
        # sequence vs setup fresh sequence" decision; in legacy
        # single-batch mode it's parsed and stored but unused.
        # ``None`` disables the gate.
        _collapse_t = getattr(self.config, "collapse_mae_threshold", 0.5)
        self.collapse_mae_threshold: Optional[float] = (
            None if _collapse_t is None else float(_collapse_t)
        )

        # Streaming-mode flag (LongLive-style persistent KV cache +
        # rolling-sequence training). When True, ``_fwdbwd_one_step``
        # uses ``model.setup_sequence`` / ``generate_next_chunk`` /
        # ``compute_*_loss_streaming`` instead of the legacy single-
        # iter ``generator_loss`` / ``critic_loss`` flow.
        # FUNDAMENTAL: defaults to True. Set ``streaming_mode: false``
        # in the YAML to fall back to the legacy single-batch path.
        self.streaming_mode: bool = bool(
            getattr(self.config, "streaming_mode", True)
        )
        self.streaming_max_length: int = int(
            getattr(self.config, "streaming_max_length", 57)
        )
        # Phase-B streaming-mode aux wiring is in
        # ``_fwdbwd_streaming_step`` (action-critic z-guidance, R3GAN,
        # SC-DMD all mirror the legacy gen-with-aux block; sliced to
        # the per-iter chunk's ride window via the streaming state).

        self.sample_interval = int(
            getattr(self.config, "sample_interval", 0) or 0
        )
        self.sample_fps = int(getattr(self.config, "sample_fps", 5) or 5)
        self.sample_max_frames = int(
            getattr(self.config, "sample_max_frames", 0) or 0
        )
        # Optional explicit step list (in addition to the periodic interval).
        # Semantics: each entry fires on the FIRST eligible step at or
        # after that value. Necessary because dfake_gen_update_ratio>1
        # makes even-indexed steps critic-only (no rollout to emit), so
        # asking for an emission at e.g. step 10 with ratio=2 has to be
        # interpreted as "at the next gen-step", which is step 11.
        _sas = getattr(self.config, "sample_at_steps", None)
        self._sample_at_steps_pending: list = (
            sorted({int(s) for s in _sas}) if _sas else []
        )
        # When true, ALSO save mp4 to <log_dir>/samples/step_<n>.mp4 — this
        # works even when wandb is disabled, so smoke runs can inspect early
        # rollouts.
        self.vis_save_local = bool(
            getattr(self.config, "vis_save_local", False)
        )
        self._pending_video_latents: Optional[torch.Tensor] = None
        self._video_logger_warned_no_vae: bool = False
        # Diagnostic stash populated by the DMD model when a sample
        # video is due. Keys: ``pred_real``, ``pred_fake``,
        # ``clean_x_fake``, ``clean_x_real`` (latent tensors), plus
        # scalar metadata (``dmd_timestep``, ``clean_x_aug_t``,
        # ``dmd_context``). Decoded alongside ``_pending_video_latents``
        # at the video-logger callsite.
        self._pending_dmd_eval_latents: Optional[Dict[str, Any]] = None
        # Sticky "due" bits — set when the cadence boundary is crossed
        # on a step that can't actually emit (critic-only iter under
        # dfake_gen_update_ratio>1). The next iter that CAN emit
        # (gen-iter with a fresh chunk / generator_log_dict) consumes
        # the bit. Without these, log_interval / sample_interval values
        # whose parity collides with dfake_gen_update_ratio silently
        # never log — what bit us at the start of phase_1_b2b.
        self._wandb_log_due: bool = False
        self._video_sample_due_bit: bool = False
        if self.sample_interval > 0 and self.is_main_process:
            logging.info(
                "[ActionForcing] Video sampling ENABLED: every %d "
                "generator steps, decode pred_image -> mp4 -> wandb "
                "(fps=%d, max_frames=%s)",
                self.sample_interval,
                self.sample_fps,
                ("all" if self.sample_max_frames <= 0
                 else str(self.sample_max_frames)),
            )
        if self.is_main_process:
            cf = int(getattr(self.model, "dmd_context_clean_frames", 0))
            N = int(getattr(self.config, "num_training_frames", 21))
            logging.info(
                "[ActionForcing] dmd_context: KV-cache seed prefill "
                "= %d frames (= dmd_context_clean_frames). Per iter, "
                "trainer picks random offset s ~ U[0, ride_len/2] "
                "(broadcast from rank 0 in DDP), then slices "
                "seed=ride[s:s+%d] and rollout=ride[s+%d:s+%d]; "
                "scorers see clean/noisy shift of one chunk "
                "(= num_frame_per_block).",
                cf, cf, cf, cf + N,
            )

    def _build_pipeline(self) -> None:
        cfg = self.config
        denoising_step_list = list(getattr(
            cfg, "denoising_step_list", [1000, 750, 500, 250],
        ))
        num_frame_per_block = int(getattr(cfg, "num_frame_per_block", 3))
        chunks_per_rolling_step = int(getattr(cfg, "chunks_per_rolling_step", 1))
        same_step_across_blocks = bool(getattr(cfg, "same_step_across_blocks", True))
        last_step_only = bool(getattr(cfg, "last_step_only", False))
        num_max_frames = int(getattr(cfg, "num_training_frames", 21))
        # CF-style long rollout: ``rollout_frames`` (default = unset =
        # ``num_training_frames``) controls the per-iter rollout
        # length. When > num_training_frames, the pipeline rolls that
        # many frames but only the LAST ``num_training_frames`` get
        # gradient — matches CF's
        # ``start_gradient_frame_index = num_output_frames - 21``.
        rollout_frames_raw = getattr(cfg, "rollout_frames", None)
        rollout_frames = (
            int(rollout_frames_raw) if rollout_frames_raw is not None
            else num_max_frames
        )
        if rollout_frames < num_max_frames:
            raise ValueError(
                f"cfg.rollout_frames ({rollout_frames}) must be >= "
                f"num_training_frames ({num_max_frames}). The gradient "
                f"window is the last num_training_frames frames of the "
                f"rollout, so the rollout must be at least that long."
            )
        if rollout_frames % num_frame_per_block != 0:
            raise ValueError(
                f"cfg.rollout_frames ({rollout_frames}) must be divisible "
                f"by num_frame_per_block ({num_frame_per_block})."
            )
        context_noise = int(getattr(cfg, "context_noise", 0))

        # MAE-driven dynamic rollout extension. ``threshold`` is the
        # upper bound on last-3-frames latent-MAE below which the
        # pipeline rolls one more 3-frame chunk (no_grad, full
        # denoise). ``max_extra_chunks`` caps how many extras we'll
        # roll per iter (each extra chunk costs cache memory; the
        # cache is sized to the sum of baseline + max extras).
        # Threshold can be ``null`` to disable the extension loop
        # entirely (still computes baseline_last_chunk_mae for
        # logging when ``gt_latents`` is passed; never extends).
        mae_extension_threshold_raw = getattr(cfg, "mae_extension_threshold", None)
        mae_extension_threshold = (
            float(mae_extension_threshold_raw)
            if mae_extension_threshold_raw is not None
            else None
        )
        mae_extension_max_extra_chunks = int(
            getattr(cfg, "mae_extension_max_extra_chunks", 0)
        )

        self.pipeline = ActionForcingTrainingPipeline(
            denoising_step_list=denoising_step_list,
            scheduler=self.model.scheduler,
            generator=self.model.generator,
            num_frame_per_block=num_frame_per_block,
            chunks_per_rolling_step=chunks_per_rolling_step,
            same_step_across_blocks=same_step_across_blocks,
            last_step_only=last_step_only,
            num_max_frames=num_max_frames,
            rollout_frames=rollout_frames,
            mae_extension_threshold=mae_extension_threshold,
            mae_extension_max_extra_chunks=mae_extension_max_extra_chunks,
            context_noise=context_noise,
        )
        # The ActionForcingDMD model needs the pipeline reference for backward
        # simulation inside generator_loss / critic_loss.
        self.model.inference_pipeline = self.pipeline

        if self.is_main_process:
            logging.info(
                "[ActionForcing] Pipeline: num_frame_per_block=%d chunks_per_rolling_step=%d "
                "denoising_step_list=%s context_noise=%d num_max_frames=%d "
                "rollout_frames=%d (gradient window = last %d frames; "
                "warmup = %d frames) mae_extension_threshold=%s "
                "mae_extension_max_extra_chunks=%d",
                num_frame_per_block, chunks_per_rolling_step,
                denoising_step_list, context_noise, num_max_frames,
                rollout_frames, num_max_frames,
                rollout_frames - num_max_frames,
                mae_extension_threshold, mae_extension_max_extra_chunks,
            )

        # ------------------------------------------------------------------
        # Infinity-RoPE (Block-Relativistic RoPE) on the KV-cache path.
        # ``utils.infinity_rope.install()`` monkey-patches
        # ``CausalWanSelfAttention.forward`` to store un-roped K in the
        # cache and rotate Q / cached-K with bounded window-relative
        # indices at attention time. Default ON so that the Action-
        # Forcing student is trained with the SAME RoPE convention
        # that ``utils/eval_causal_AR.py`` runs with at AR rollout
        # time (its ``infinity_rope`` flag also defaults true). Within
        # an Action-Forcing training iter the cache never rolls
        # (num_training_frames ==
        # local_attn_size, sink_size=0), so the patch is mathematically
        # equivalent to the original RoPE path during training; the
        # point is purely train/inference parity. Set
        # ``infinity_rope=false`` in the YAML / via ``--override`` to
        # restore the original RoPE training path for A/B studies.
        infinity_rope_enabled = bool(getattr(cfg, "infinity_rope", True))
        if infinity_rope_enabled:
            _install_infinity_rope(self._inner_dit_for_rope())
            if self.is_main_process:
                logging.info(
                    "[ActionForcing] Infinity-RoPE patch installed on "
                    "CausalWanSelfAttention.forward (train/inference "
                    "parity with utils/eval_causal_AR.py defaults)."
                )
        else:
            if self.is_main_process:
                logging.info(
                    "[ActionForcing] Infinity-RoPE patch DISABLED "
                    "(infinity_rope=false). Training will run the "
                    "original RoPE path; AR-eval defaults to "
                    "infinity_rope=true so a train/inference mismatch "
                    "is expected on long-horizon rollouts."
                )

        # ------------------------------------------------------------------
        # Step-scheduled local_attn_size (KV cache window). Lets training
        # start cheap (small attention window = fast iters, modest cache)
        # then grow context as the student's predictions stabilise and
        # long-horizon coherence becomes the bottleneck. Format:
        #
        #   local_attn_size_schedule: [[step_threshold, frames], ...]
        #
        # At step S, the active window is the size for the LARGEST
        # threshold ≤ S. Omit the key (or set null) to disable the
        # schedule and use the static ``model_kwargs.local_attn_size``
        # for the whole run. The schedule first waypoint should match
        # ``model_kwargs.local_attn_size`` (the model's construction-
        # time value) so step-0 attention behaviour is unchanged.
        sched_raw = getattr(cfg, "local_attn_size_schedule", None)
        self._attn_size_schedule: List[Tuple[int, int]] = []
        if sched_raw:
            try:
                pairs = [(int(s), int(n)) for s, n in sched_raw]
                pairs.sort(key=lambda p: p[0])
                self._attn_size_schedule = pairs
            except Exception as e:
                if self.is_main_process:
                    logging.warning(
                        "[ActionForcing] Bad local_attn_size_schedule %r: %s "
                        "— ignoring schedule.", sched_raw, e,
                    )
        # Last applied frames count, set to None until first apply so the
        # first sequence open at step 0 always seeds with the right size.
        self._current_attn_frames: Optional[int] = None
        if self._attn_size_schedule and self.is_main_process:
            logging.info(
                "[ActionForcing] local_attn_size_schedule active: %s",
                self._attn_size_schedule,
            )

    # ------------------------------------------------------------------
    # Optimizer (parent builds gen + fake_score; we add the critic).
    # ------------------------------------------------------------------
    def _build_optimizer(self) -> None:
        super()._build_optimizer()
        cfg = self.config
        # Aux-loss hyperparameters. Defaults match the ODE distill
        # teacher trainer (``causal_diffusion_teacher_train.py``):
        # critic_lr=3e-4, weight=0.25, gen z-guidance=0.125 (warmup
        # 500), 2 critic updates per gen step, dims=[2,7].
        critic_dims_cfg = getattr(cfg, "action_critic_dims", None)
        self.action_critic_dims: List[int] = (
            list(critic_dims_cfg) if critic_dims_cfg is not None else [2, 7]
        )
        self.action_critic_z_loss_weight = float(
            getattr(cfg, "action_critic_z_loss_weight", 0.25)
        )
        self.generator_action_z_guidance_weight = float(
            getattr(cfg, "generator_action_z_guidance_weight", 0.0)
        )
        self.z_guidance_warmup_steps = int(
            getattr(cfg, "z_guidance_warmup_steps", 500)
        )
        self.warmup_steps_for_guidance = int(getattr(cfg, "warmup_steps", 0))
        self.critic_updates_per_step = int(
            getattr(cfg, "critic_updates_per_step", 1)
        )
        self.critic_optimizer: Optional[torch.optim.Optimizer] = None
        if self.action_critic_loss_active and self.model.action_critic is not None:
            critic_lr = float(getattr(cfg, "critic_lr", 3e-4))
            critic_betas = tuple(
                getattr(cfg, "critic_betas", getattr(cfg, "betas", [0.9, 0.999]))
            )
            critic_eps = float(getattr(cfg, "critic_eps", getattr(cfg, "eps", 1e-8)))
            critic_wd = float(
                getattr(cfg, "critic_weight_decay", getattr(cfg, "weight_decay", 0.0))
            )
            critic_params = [
                p for p in self.model.action_critic.parameters() if p.requires_grad
            ]
            if not critic_params:
                raise RuntimeError(
                    "action_critic has no trainable parameters even after "
                    "un-freeze; check ActionCritic.__init__ + the trainer's "
                    "requires_grad_(True) call in _build_model."
                )
            self.critic_optimizer = torch.optim.AdamW(
                critic_params,
                lr=critic_lr,
                betas=critic_betas,
                eps=critic_eps,
                weight_decay=critic_wd,
            )
            self.critic_max_grad_norm = float(
                getattr(cfg, "critic_max_grad_norm", self.max_grad_norm)
            )
            if self.is_main_process:
                n_params = sum(p.numel() for p in critic_params)
                logging.info(
                    "[ActionForcing] Action critic optimizer built: "
                    "AdamW lr=%.2e wd=%.4f params=%d (z_loss_weight=%.4f, "
                    "gen_z_guidance_weight=%.4f, warmup_steps=%d, "
                    "z_guidance_warmup_steps=%d, critic_updates_per_step=%d, "
                    "dims=%s)",
                    critic_lr, critic_wd, n_params,
                    self.action_critic_z_loss_weight,
                    self.generator_action_z_guidance_weight,
                    self.warmup_steps_for_guidance,
                    self.z_guidance_warmup_steps,
                    self.critic_updates_per_step,
                    self.action_critic_dims,
                )

        # ------------------------------------------------------------------
        # R3GAN discriminator optimizer + hyperparameters. Defaults
        # follow the StyleGAN/R3GAN convention: AdamW with
        # ``betas=(0.0, 0.9)`` (high-momentum first-order beta is
        # *unsafe* for GAN D — keep first-moment beta near 0 so D
        # doesn't lock into pre-G's-update statistics).
        # ------------------------------------------------------------------
        self.gan_loss_weight = float(getattr(cfg, "gan_loss_weight", 0.05))
        self.gan_r1_gamma = float(getattr(cfg, "gan_r1_gamma", 1.0))
        self.gan_r2_gamma = float(getattr(cfg, "gan_r2_gamma", 1.0))
        # ``gan_warmup_steps``: ramp the *generator-side* RpGAN-G loss
        # weight from 0 → ``gan_loss_weight`` over this many steps so
        # the gen doesn't see noise from a freshly-initialised D.
        # The D-side update runs from step 0 (it has to learn before
        # it can give G a signal). Default 500 (== z_guidance_warmup_steps).
        self.gan_warmup_steps = int(getattr(cfg, "gan_warmup_steps", 500))
        self.gan_updates_per_step = int(
            getattr(cfg, "gan_updates_per_step", 1)
        )
        self.gan_max_grad_norm = float(
            getattr(cfg, "gan_max_grad_norm", self.max_grad_norm)
        )
        self.r3gan_optimizer: Optional[torch.optim.Optimizer] = None
        if self.gan_enabled and self.r3gan_disc is not None:
            disc_lr = float(getattr(cfg, "gan_lr", 2e-4))
            disc_betas = tuple(
                getattr(cfg, "gan_betas", [0.0, 0.9])
            )
            disc_eps = float(getattr(cfg, "gan_eps", 1e-8))
            disc_wd = float(getattr(cfg, "gan_weight_decay", 0.0))
            disc_params = [
                p for p in self.r3gan_disc.parameters() if p.requires_grad
            ]
            if not disc_params:
                raise RuntimeError(
                    "r3gan_disc has no trainable parameters; check the "
                    "constructor."
                )
            self.r3gan_optimizer = torch.optim.AdamW(
                disc_params,
                lr=disc_lr,
                betas=disc_betas,
                eps=disc_eps,
                weight_decay=disc_wd,
            )
            if self.is_main_process:
                n_params = sum(p.numel() for p in disc_params)
                logging.info(
                    "[ActionForcing] R3GAN optimizer built: AdamW "
                    "lr=%.2e betas=%s wd=%.4f params=%.2fM "
                    "(loss_weight=%.4f, R1_gamma=%.3f, R2_gamma=%.3f, "
                    "warmup_steps=%d, gan_updates_per_step=%d)",
                    disc_lr, disc_betas, disc_wd, n_params / 1e6,
                    self.gan_loss_weight, self.gan_r1_gamma,
                    self.gan_r2_gamma, self.gan_warmup_steps,
                    self.gan_updates_per_step,
                )

    def _inner_dit_for_rope(self):
        """Return the bare DiT module under the (possibly DDP / LoRA)
        generator so ``infinity_rope.install()`` can clear any
        per-attention-module rotated-prefix state on its
        ``CausalWanSelfAttention`` children. The class-level monkey
        patch is global, so this argument is only used for state
        clearing — passing the wrapper is fine since
        ``_clear_module_state`` walks ``module.modules()``.
        """
        gen = self.model.generator.model
        return gen

    # ------------------------------------------------------------------
    # Training loop (CF-parity).
    # ------------------------------------------------------------------
    def train(self) -> None:
        cfg = self.config
        max_steps = int(getattr(cfg, "max_steps", 10000))
        ckpt_interval = int(getattr(cfg, "checkpoint_interval", 500))
        log_interval = int(getattr(cfg, "log_interval", 10))
        dfake_gen_update_ratio = int(getattr(cfg, "dfake_gen_update_ratio", 1))
        num_training_frames = int(getattr(cfg, "num_training_frames", 21))
        num_frame_per_block = int(getattr(cfg, "num_frame_per_block", 3))
        rollout_frames_raw = getattr(cfg, "rollout_frames", None)
        rollout_frames = (
            int(rollout_frames_raw) if rollout_frames_raw is not None
            else num_training_frames
        )
        # MAE-extension cap (must mirror what was passed to the pipeline
        # in ``_build_pipeline``). The trainer uses this to size the
        # ride slice + action conditioning streams so that even if the
        # extension fires the maximum number of times, the per-frame
        # streams cover the full extended window.
        #
        # Gating MUST mirror the pipeline's gating exactly
        # (``ActionForcingTrainingPipeline.inference_with_trajectory``:
        # ``extension_active = enable_mae_extension AND threshold is not
        # None AND max_extra_chunks > 0``). Otherwise threshold=null with
        # cap>0 (or vice-versa) would still have the trainer slice an
        # extended ride + build action conditioning over the extended
        # window, even though the pipeline never extends — that's the
        # "set threshold=null to disable" path failing to be free.
        mae_extension_threshold_for_alloc = getattr(
            cfg, "mae_extension_threshold", None
        )
        mae_extension_max_extra_chunks = int(
            getattr(cfg, "mae_extension_max_extra_chunks", 0)
        )
        extensions_active = (
            mae_extension_threshold_for_alloc is not None
            and mae_extension_max_extra_chunks > 0
        )
        max_extension_frames = (
            mae_extension_max_extra_chunks * num_frame_per_block
            if extensions_active
            else 0
        )
        max_total_rollout_frames = rollout_frames + max_extension_frames

        # dmd_context shifts the student's rollout window forward by
        # ``cf`` frames in the ride (= ride frames [cf, cf +
        # max_total_rollout_frames)) so the leading ``cf`` ride frames
        # become the KV-cache seed prefill. ``cf`` is the model's
        # single source of truth (= ``self.model.dmd_context_clean_frames``,
        # default 9 = KV-cache size).
        cf_dmdctx = int(getattr(self.model, "dmd_context_clean_frames", 0))
        # CF-parity #6: periodic memory hygiene.
        # ``empty_cache_interval``: how often to release PyTorch's caching
        # allocator pool back to the driver (CF: every 20 steps).
        # ``gc_interval``: how often to run a full Python ``gc.collect``
        # (CF: every ``gc_interval``, default 100). Both reduce peak
        # memory and fragmentation on long runs without affecting losses.
        empty_cache_interval = int(getattr(cfg, "empty_cache_interval", 20))
        gc_interval = int(getattr(cfg, "gc_interval", 100))

        if self.is_main_process:
            logging.info(
                "[ActionForcing] Starting CF-style DMD training: max_steps=%d "
                "dfake_gen_update_ratio=%d num_training_frames=%d "
                "rollout_frames=%d (gradient on last %d frames; "
                "warmup_frames=%d) mae_extension_max_extra_chunks=%d "
                "(=> max_total_rollout_frames=%d when ride is long enough "
                "and MAE stays under threshold)",
                max_steps, dfake_gen_update_ratio, num_training_frames,
                rollout_frames, num_training_frames,
                rollout_frames - num_training_frames,
                mae_extension_max_extra_chunks, max_total_rollout_frames,
            )
            logging.info(
                "[ActionForcing] memory hygiene: empty_cache every %d steps, "
                "gc.collect every %d steps",
                empty_cache_interval, gc_interval,
            )
            logging.info(
                "[ActionForcing] EMA: weight=%.4f start_step=%d (lazy-init at "
                "start_step; EMA shadow not created until then)",
                float(getattr(cfg, "ema_weight", 0.0) or 0.0),
                int(getattr(cfg, "ema_start_step", 0) or 0),
            )

        previous_time: Optional[float] = None
        self._epoch = 0
        self._ride_iter = self._fresh_ride_iter(self._epoch)

        # Apply the schedule's step-0 value before the first sequence
        # opens (no-op when no schedule configured).
        self._apply_attn_size_if_changed()

        while self.step < max_steps:
            # CF-parity #5: put the whole DMD module in eval mode at the
            # start of each iter to suppress any latent randomness
            # (Dropout / BN running stats). Our DiT has no Dropout/BN
            # so this is a no-op in math, but we match CF's
            # ``self.model.eval()`` at ``Causal-Forcing/trainer/
            # distillation.py:230`` to keep the code path identical.
            self.model.eval()
            # Cheap per-iter check for an attn-size schedule transition.
            # Only does work on the iter that crosses a threshold (rest
            # are dict-lookup + int-compare). On a transition this also
            # closes the open streaming sequence so the next sequence
            # re-allocates the KV cache at the new size.
            self._apply_attn_size_if_changed()
            # Decide whether this iter trains the generator or the critic.
            train_generator = (self.step % dfake_gen_update_ratio == 0)

            if train_generator:
                generator_log_dict = self._fwdbwd_one_step(
                    train_generator=True,
                    rollout_frames=rollout_frames,
                    max_total_rollout_frames=max_total_rollout_frames,
                    cf_dmdctx=cf_dmdctx,
                )

            # Always run the critic step (CF parity).
            critic_log_dict = self._fwdbwd_one_step(
                train_generator=False,
                rollout_frames=rollout_frames,
                max_total_rollout_frames=max_total_rollout_frames,
                cf_dmdctx=cf_dmdctx,
            )

            # Step optimizers.
            if train_generator:
                self._all_reduce_extra_trainable_grads()
                gen_grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.optimizer.param_groups[0]["params"]
                     if p.grad is not None],
                    max_norm=self.max_grad_norm,
                )
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self._maybe_update_generator_ema()
            else:
                gen_grad_norm = torch.tensor(0.0, device=self.device)

            fake_grad_norm_val = 0.0
            if self.fake_optimizer is not None:
                fake_params_with_grad = [
                    p for p in self.fake_optimizer.param_groups[0]["params"]
                    if p.grad is not None
                ]
                if fake_params_with_grad:
                    fgn = torch.nn.utils.clip_grad_norm_(
                        fake_params_with_grad,
                        max_norm=self.fake_max_grad_norm,
                    )
                    fake_grad_norm_val = (
                        float(fgn.item()) if torch.is_tensor(fgn) else float(fgn)
                    )
                    self.fake_optimizer.step()
                self.fake_optimizer.zero_grad(set_to_none=True)

            self.step += 1

            # Mark wandb log as DUE if the cadence boundary just
            # crossed. With ``dfake_gen_update_ratio>1`` the boundary
            # often lands on a critic-only iter (no fresh gen data),
            # so we just stash a sticky bit and emit on the next iter
            # that has both gen and critic data — meeting the spec
            # "if it is not the right step then it needs to happen in
            # the subsequent step." Without this, e.g. log_interval=10
            # with dfake=2 silently skips every gen log because
            # boundary post-step parity always lands on a critic iter.
            if self.step % log_interval == 0:
                self._wandb_log_due = True

            # Mark video sample as DUE on the same cadence-boundary
            # principle. The ``_video_sample_due`` callsite (inside
            # the gen rollout) already runs only on gen iters, so the
            # bit is consumed there; we just need to set it whenever
            # the boundary crossed but no chunk got stashed this iter
            # (== ``_pending_video_latents is None`` at this point,
            # since stash happens during the gen-step BEFORE step++).
            if (
                self.sample_interval > 0
                and (self.step % self.sample_interval) == 0
                and self._pending_video_latents is None
            ):
                self._video_sample_due_bit = True

            # Logging — emit only when we have BOTH fresh gen and
            # critic data this iter (i.e. ``train_generator`` was True
            # and the gen step actually produced a dict). On a
            # critic-only iter we leave ``_wandb_log_due`` set so the
            # next gen iter emits the deferred log.
            gen_ready = (train_generator and generator_log_dict is not None)
            critic_ready = (critic_log_dict is not None)
            if (
                self.is_main_process
                and self._wandb_log_due
                and gen_ready
                and critic_ready
            ):
                msg_parts = [f"step={self.step}/{max_steps}"]
                msg_parts.append(
                    f"gen_loss={generator_log_dict.get('generator_loss', 0.0):.4f}"
                )
                if "dmdtrain_gradient_norm" in generator_log_dict:
                    msg_parts.append(
                        f"dmd_grad={generator_log_dict['dmdtrain_gradient_norm']:.4f}"
                    )
                msg_parts.append(
                    f"critic_loss={critic_log_dict.get('critic_loss', 0.0):.4f}"
                )
                msg_parts.append(
                    f"gen_grad_norm={float(gen_grad_norm.item()) if torch.is_tensor(gen_grad_norm) else float(gen_grad_norm):.4f}"
                )
                msg_parts.append(f"fake_grad_norm={fake_grad_norm_val:.4f}")
                # MAE diagnostics: surface baseline_last_chunk_mae +
                # baseline_avg_rollout_mae to stdout so we can detect
                # student collapse from the iter trace alone (the
                # primary "is the student degrading" signal).
                _mae_last = generator_log_dict.get("baseline_last_chunk_mae")
                if _mae_last is not None:
                    msg_parts.append(f"mae_last={float(_mae_last):.4f}")
                _mae_avg = generator_log_dict.get("baseline_avg_rollout_mae")
                if _mae_avg is not None:
                    msg_parts.append(f"mae_avg={float(_mae_avg):.4f}")
                logging.info("[ActionForcing] " + " ".join(msg_parts))
                if (
                    _HAS_WANDB
                    and getattr(self, "wandb_enabled", False)
                ):
                    log_payload: Dict[str, Any] = {}
                    for k, v in generator_log_dict.items():
                        if torch.is_tensor(v):
                            v = float(v.item())
                        log_payload[f"gen/{k}"] = v
                    for k, v in critic_log_dict.items():
                        if torch.is_tensor(v):
                            v = float(v.item())
                        log_payload[f"critic/{k}"] = v
                    log_payload["gen/grad_norm"] = (
                        float(gen_grad_norm.item())
                        if torch.is_tensor(gen_grad_norm)
                        else float(gen_grad_norm)
                    )
                    log_payload["critic/grad_norm"] = fake_grad_norm_val
                    if previous_time is not None:
                        log_payload["per_iter_time"] = time.time() - previous_time
                    try:
                        wandb.log(log_payload, step=self.step)
                    except Exception as e:
                        logging.warning("wandb.log failed: %s", e)
                previous_time = time.time()
                self._wandb_log_due = False

            if (
                self.is_main_process
                and self._pending_video_latents is not None
            ):
                self._log_pred_image_video(
                    self._pending_video_latents, int(self.step),
                )
                self._pending_video_latents = None

                # DMD scorer + clean_x diagnostic videos. ``pred_real``
                # / ``pred_fake`` are the scorers' denoised x0 estimates
                # on the SAME noisy student input — divergence between
                # them is what DMD is asking the student to close;
                # ``clean_x_fake`` / ``clean_x_real`` are what each
                # scorer was conditioned on. Same VAE decode + ffmpeg
                # encode path as ``pred_image``; best-effort, never
                # raises.
                eval_latents = getattr(self, "_pending_dmd_eval_latents", None)
                if isinstance(eval_latents, dict) and eval_latents:
                    _t = eval_latents.get("dmd_timestep")
                    _aug = eval_latents.get("clean_x_aug_t")
                    _ctx = eval_latents.get("dmd_context")
                    _suffix_real = (
                        f"t={_t} aug_t={_aug} ctx={_ctx}"
                        if _t is not None
                        else ""
                    )
                    for _key, _name, _cap in (
                        ("pred_real", "pred_real", _suffix_real or ""),
                        ("pred_fake", "pred_fake", _suffix_real or ""),
                        ("clean_x_fake", "clean_x_fake", "fake_score conditioning"),
                        ("clean_x_real", "clean_x_real", f"real_score conditioning ({_ctx})"),
                    ):
                        _t_lat = eval_latents.get(_key)
                        if _t_lat is None or not torch.is_tensor(_t_lat):
                            continue
                        # Action overlay disabled — see commit notes;
                        # the per-frame numpy bar render was suspected
                        # of stalling the gen-loss tail. Leave the
                        # ``clean_z_actions`` stash in the model so we
                        # can re-enable here once the slowdown is
                        # diagnosed (the stash itself is just a tensor
                        # detach, sub-millisecond).
                        _overlay = None
                        try:
                            self._log_pred_image_video(
                                _t_lat.to(torch.float32),
                                int(self.step),
                                name=_name,
                                caption_suffix=_cap,
                                action_overlay=_overlay,
                            )
                        except Exception as _exc:
                            logging.warning(
                                "[ActionForcing] DMD eval video '%s' "
                                "failed at step=%d: %s",
                                _name, int(self.step), _exc,
                            )
                self._pending_dmd_eval_latents = None

            # Checkpoint.
            if self.step > 0 and self.step % ckpt_interval == 0:
                if self.is_main_process:
                    self._save_checkpoint()
                if self.world_size > 1:
                    dist.barrier()

            # CF-parity #6: periodic memory hygiene. Release PyTorch's
            # caching allocator pool back to the driver every
            # ``empty_cache_interval`` steps (CF: 20) and run a full
            # Python ``gc.collect`` every ``gc_interval`` steps (CF: 100).
            # Both are cheap; the empty_cache call returns memory to
            # the driver but PyTorch will re-grow the pool on the next
            # alloc.
            if (
                empty_cache_interval > 0
                and self.step % empty_cache_interval == 0
            ):
                torch.cuda.empty_cache()
            if gc_interval > 0 and self.step % gc_interval == 0:
                gc.collect()
                if self.is_main_process:
                    logging.debug("[ActionForcing] gc.collect at step=%d", self.step)

        if self.is_main_process:
            logging.info("[ActionForcing] Training complete (step=%d).", self.step)
            self._save_checkpoint()

    # ------------------------------------------------------------------
    # Checkpoint save/resume — extend the parent to persist the
    # action_critic state_dict + critic optimizer state when aux
    # losses are enabled. The parent's path covers the generator /
    # action_projection / action_token_projection / fake_score /
    # fake_optimizer / EMA / step. We append the critic-side state
    # AFTER the parent has written the rest, then re-save the same
    # file. This avoids forking the parent's serialization layout.
    # ------------------------------------------------------------------
    def _save_checkpoint(self) -> None:
        super()._save_checkpoint()
        if not self.is_main_process:
            return
        # Either the action critic OR the GAN may need state appended;
        # fast-path-skip if neither is active.
        if not (self.action_critic_loss_active or self.gan_enabled):
            return
        path = self._checkpoint_path(self.step)
        if not os.path.exists(path):
            return
        try:
            state = torch.load(path, map_location="cpu")
        except Exception as exc:
            logging.warning(
                "[ActionForcing] Could not re-open checkpoint for "
                "aux/GAN append: %s. Skipping.", exc,
            )
            return
        appended = []
        if self.action_critic_loss_active:
            ac = self.model.action_critic
            ac_module = ac.module if isinstance(ac, DDP) else ac
            if ac_module is not None:
                state["action_critic"] = ac_module.state_dict()
                appended.append("action_critic")
            if self.critic_optimizer is not None:
                state["critic_optimizer"] = self.critic_optimizer.state_dict()
                appended.append("critic_optimizer")
        if self.gan_enabled and self.r3gan_disc is not None:
            disc_module = (
                self.r3gan_disc_ddp.module if self.r3gan_disc_ddp is not None
                else self.r3gan_disc
            )
            state["r3gan_discriminator"] = disc_module.state_dict()
            appended.append("r3gan_discriminator")
            if self.r3gan_optimizer is not None:
                state["r3gan_optimizer"] = self.r3gan_optimizer.state_dict()
                appended.append("r3gan_optimizer")
        if not appended:
            return
        torch.save(state, path)
        logging.info(
            "[ActionForcing] Appended state to %s: %s",
            path, ", ".join(appended),
        )

    def _maybe_resume(self) -> None:
        super()._maybe_resume()
        if not (self.action_critic_loss_active or self.gan_enabled):
            return
        if not bool(getattr(self.config, "auto_resume", False)):
            return
        ckpts = sorted(self.log_dir.glob("phase1_step*.pt"))
        if not ckpts:
            return
        path = ckpts[-1]
        try:
            state = torch.load(path, map_location="cpu")
        except Exception as exc:
            if self.is_main_process:
                logging.warning(
                    "[ActionForcing] Could not load checkpoint for "
                    "aux/GAN resume: %s", exc,
                )
            return
        if self.action_critic_loss_active:
            ac = self.model.action_critic
            ac_module = ac.module if isinstance(ac, DDP) else ac
            if ac_module is not None and "action_critic" in state:
                ac_missing, ac_unexpected = ac_module.load_state_dict(
                    state["action_critic"], strict=False,
                )
                if self.is_main_process:
                    logging.info(
                        "resume: action_critic missing=%d unexpected=%d",
                        len(ac_missing), len(ac_unexpected),
                    )
            if self.critic_optimizer is not None and "critic_optimizer" in state:
                try:
                    self.critic_optimizer.load_state_dict(state["critic_optimizer"])
                    if self.is_main_process:
                        logging.info("resume: critic_optimizer state restored")
                except Exception as exc:
                    if self.is_main_process:
                        logging.warning(
                            "resume: critic_optimizer load failed: %s. "
                            "Starting critic optim from fresh state.", exc,
                        )
        if self.gan_enabled and self.r3gan_disc is not None:
            disc_module = (
                self.r3gan_disc_ddp.module if self.r3gan_disc_ddp is not None
                else self.r3gan_disc
            )
            if "r3gan_discriminator" in state:
                d_missing, d_unexpected = disc_module.load_state_dict(
                    state["r3gan_discriminator"], strict=False,
                )
                if self.is_main_process:
                    logging.info(
                        "resume: r3gan_discriminator missing=%d unexpected=%d",
                        len(d_missing), len(d_unexpected),
                    )
            if self.r3gan_optimizer is not None and "r3gan_optimizer" in state:
                try:
                    self.r3gan_optimizer.load_state_dict(state["r3gan_optimizer"])
                    if self.is_main_process:
                        logging.info("resume: r3gan_optimizer state restored")
                except Exception as exc:
                    if self.is_main_process:
                        logging.warning(
                            "resume: r3gan_optimizer load failed: %s. "
                            "Starting r3gan optim from fresh state.", exc,
                        )

    # ------------------------------------------------------------------
    # Auxiliary action-critic losses (ported from
    # ``trainer.causal_diffusion_teacher_train._compute_action_critic_losses``,
    # adapted for the action-forcing rollout shape: ``pred_x0`` is the
    # student's last ``num_training_frames``-frame baseline window
    # rather than a per-frame flow-matching pred, so we use a single
    # broadcast timestep — the rollout's rung-exit timestep — instead
    # of the per-frame random timestep that the teacher trainer has).
    # ------------------------------------------------------------------
    def _weighted_z_mse(self, pred_z: torch.Tensor, target_z: torch.Tensor) -> torch.Tensor:
        w = torch.ones(pred_z.shape[-1], device=pred_z.device, dtype=pred_z.dtype)
        for dim_idx in self.action_critic_dims:
            w[dim_idx] = 2.0
        return (w * (pred_z - target_z) ** 2).mean()

    def _slice_actions_for_critic(self, actions: torch.Tensor) -> torch.Tensor:
        """Slice the action stream's last dim down to ``action_critic_dims``.

        The data loader (``_load_ride_tensors``) pre-slices ``z_actions``
        by ``self.action_dims`` before storing them on the ride dict, so
        ``actions[..., -1]`` may already be the critic-target slice.
        We therefore have three valid configurations to handle:

          1. ``action_dims is None`` AND ``actions.shape[-1] == 8``
             (or whatever the SS-VAE z-dim is): no pre-slicing happened,
             slice with ``action_critic_dims`` directly.
          2. ``action_dims is not None`` AND ``actions.shape[-1] ==
             len(action_dims)``: pre-slicing happened. Map
             ``action_critic_dims`` (absolute indices into the original
             8-dim z space) into RELATIVE indices within the
             ``action_dims``-sliced stream, then slice. If
             ``action_dims == action_critic_dims`` this is just
             ``[0..len(action_critic_dims)-1]``, i.e. a no-op slice.
          3. Anything else: fall back to the absolute-index slice and
             let the index op surface the bug as an out-of-bounds error.
        """
        last = int(actions.shape[-1])
        crit_dims = list(self.action_critic_dims)
        if self.action_dims is None:
            return actions[..., crit_dims]
        if last == len(self.action_dims):
            try:
                rel = [int(self.action_dims.index(int(d))) for d in crit_dims]
            except ValueError as exc:
                raise RuntimeError(
                    f"action_critic_dims={crit_dims} contains an index "
                    f"not present in action_dims={list(self.action_dims)}. "
                    "The action critic z-target stream is pre-sliced by "
                    "action_dims in the data loader; action_critic_dims "
                    "must be a subset of action_dims."
                ) from exc
            return actions[..., rel]
        return actions[..., crit_dims]

    def _compute_action_critic_losses(
        self,
        pred_x0: torch.Tensor,
        target_action_z: torch.Tensor,
        chunk_t: torch.Tensor,
        current_step: int,
    ):
        """Train the action critic and return generator z-guidance loss.

        Args:
            pred_x0: ``[B, F, C, H, W]`` student rollout pred (graph-
                carrying; we detach internally for the critic-update
                branch and use the live tensor for the gen-side
                guidance branch).
            target_action_z: ``[B, F, K]`` per-frame commanded action
                (already sliced to ``self.action_critic_dims``).
            chunk_t: ``[B, n_chunks]`` per-chunk diffusion timestep
                (broadcast from the rollout's rung-exit timestep).
            current_step: training step (for guidance warmup ramp).

        Returns:
            ``(generator_action_loss, logs, teacher_z_8d)`` —
            ``generator_action_loss`` is graph-carrying and should
            be added to the DMD loss before backward; ``logs`` is a
            flat ``str -> float`` dict for wandb.
        """
        # DDP-wrap-aware handles. ``critic_for_update`` goes through
        # the DDP wrapper so backward triggers all-reduce on critic
        # gradients — required for distributed correctness on the
        # critic optim. ``critic_for_guidance`` is the UNWRAPPED
        # ActionCritic; we call it directly in the gen-side branch
        # so DDP isn't involved (we set ``requires_grad_(False)``
        # there anyway, and the gradient we care about is on the
        # generator's pred_x0, which flows through the outer
        # generator-DDP backward).
        critic_for_update = self.model.action_critic
        critic_for_guidance = (
            self.model.action_critic.module
            if isinstance(self.model.action_critic, DDP)
            else self.model.action_critic
        )
        chunk_frames = int(self.config.num_frame_per_block)
        B = pred_x0.shape[0]
        n_chunks = pred_x0.shape[1] // chunk_frames

        chunk_actions = _chunk_actions(target_action_z, chunk_frames)[:, :n_chunks]
        # If the student rollout went off-window relative to the
        # action stream we fed the loader, we slice both to the
        # shorter min — defensive, the trainer already aligns them.
        if chunk_actions.shape[1] < n_chunks:
            n_chunks = chunk_actions.shape[1]

        zero = torch.tensor(0.0, device=pred_x0.device, dtype=pred_x0.dtype)

        teacher_z_8d = self._compute_teacher_z_per_slot(pred_x0.detach())
        if teacher_z_8d is None:
            # Teacher unavailable (cotracker / VAE / ss_vae failure) —
            # silently skip aux loss this iter; the failure was already
            # counted by the parent ``_compute_teacher_z_per_slot``.
            logs = {
                "train/critic_z_loss": 0.0,
                "train/critic_loss": 0.0,
                "train/critic_z2_mse": 0.0,
                "train/critic_z7_mse": 0.0,
                "train/gen_z_loss": 0.0,
                "train/gen_action_loss": 0.0,
                "train/teacher_z2_mean": 0.0,
                "train/teacher_z7_mean": 0.0,
                "train/z_guidance_scale": 0.0,
                "train/teacher_unavailable": 1.0,
            }
            return zero, logs, None
        teacher_z_8d = teacher_z_8d[:, :n_chunks]
        chunk_actions = chunk_actions[:, :n_chunks]
        chunk_t = chunk_t[:, :n_chunks]

        pred_x0_detached = pred_x0.detach()

        # The action critic is built in fp32 (its internal Conv3d /
        # Linear blocks default to fp32 weights, see
        # ``ActionForcingDMD._build_aux_heads``). The DMD generator
        # produces ``pred_x0`` in bf16, so a direct call into the
        # critic raises ``RuntimeError: mat1 and mat2 must have the
        # same dtype``. The v14 trainer hides this by wrapping the
        # critic call in ``autocast(bf16)``; we do the same for both
        # the critic-update and the gen-side z-guidance path so the
        # critic transparently picks up the same bf16 mat-mul kernels
        # as the rest of training. Grad flow:
        #   * critic-update path: backward stays on the critic's
        #     fp32 params (autocast unscales + casts back); optim
        #     step happens in fp32.
        #   * gen-side path: grad flows through ``pred_x0`` (bf16)
        #     into the generator's DDP backward; the critic's
        #     params are frozen for this branch.
        # --- Multi-step critic training (independent optim loop) ---
        critic_z_loss = zero
        critic_loss_k = zero
        pred_z = None
        for _k in range(max(1, self.critic_updates_per_step)):
            self.critic_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=True,
            ):
                pred_z = critic_for_update(
                    pred_x0_detached, chunk_t, chunk_actions,
                )
                pred_z = pred_z[:, :n_chunks]
                critic_z_loss = self._weighted_z_mse(
                    pred_z, teacher_z_8d.to(pred_z.dtype),
                )
                critic_loss_k = (
                    self.action_critic_z_loss_weight * critic_z_loss
                )
            critic_loss_k.backward()
            if self.critic_max_grad_norm is not None and self.critic_max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    critic_for_guidance.parameters(), self.critic_max_grad_norm,
                )
            self.critic_optimizer.step()

        # --- Generator z-guidance (frozen critic, grad through pred) ---
        warmup_start = self.warmup_steps_for_guidance
        if current_step < warmup_start:
            guidance_scale = 0.0
        elif self.z_guidance_warmup_steps > 0:
            ramp = min(
                1.0,
                (current_step - warmup_start) / max(1, self.z_guidance_warmup_steps),
            )
            guidance_scale = ramp * self.generator_action_z_guidance_weight
        else:
            guidance_scale = self.generator_action_z_guidance_weight

        if guidance_scale > 0:
            critic_for_guidance.requires_grad_(False)
            try:
                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=True,
                ):
                    gen_pred_z = critic_for_guidance(
                        pred_x0, chunk_t, chunk_actions,
                    )
                    gen_pred_z = gen_pred_z[:, :n_chunks]
                    gen_z2z7 = gen_pred_z[:, :, self.action_critic_dims]
                    target_z2z7 = chunk_actions
                    gen_z_loss = F.mse_loss(
                        gen_z2z7, target_z2z7.to(gen_z2z7.dtype),
                    )
                    generator_action_loss = guidance_scale * gen_z_loss
            finally:
                critic_for_guidance.requires_grad_(True)
        else:
            gen_z_loss = zero
            generator_action_loss = zero

        with torch.no_grad():
            z2_idx, z7_idx = (
                self.action_critic_dims[0],
                self.action_critic_dims[1] if len(self.action_critic_dims) > 1
                else self.action_critic_dims[0],
            )
            z2_mse = (
                F.mse_loss(pred_z[:, :, z2_idx].float(), teacher_z_8d[:, :, z2_idx].float()).item()
                if pred_z is not None else 0.0
            )
            z7_mse = (
                F.mse_loss(pred_z[:, :, z7_idx].float(), teacher_z_8d[:, :, z7_idx].float()).item()
                if pred_z is not None else 0.0
            )
        logs = {
            "train/critic_z_loss": float(critic_z_loss.detach().item()) if torch.is_tensor(critic_z_loss) else 0.0,
            "train/critic_loss": float(critic_loss_k.detach().item()) if torch.is_tensor(critic_loss_k) else 0.0,
            "train/critic_z2_mse": z2_mse,
            "train/critic_z7_mse": z7_mse,
            "train/gen_z_loss": float(gen_z_loss.detach().item()) if torch.is_tensor(gen_z_loss) else 0.0,
            "train/gen_action_loss": float(generator_action_loss.detach().item()) if torch.is_tensor(generator_action_loss) else 0.0,
            "train/teacher_z2_mean": float(teacher_z_8d[:, :, z2_idx].mean().item()),
            "train/teacher_z7_mean": float(teacher_z_8d[:, :, z7_idx].mean().item()),
            "train/z_guidance_scale": float(guidance_scale),
            "train/teacher_unavailable": 0.0,
        }
        return generator_action_loss, logs, teacher_z_8d

    # ------------------------------------------------------------------
    # R3GAN — RpGAN + R1 + R2.
    # Trains a separate 3D-conv discriminator on
    #   real = ride["latents"][:, gen_window_start:gen_window_end]
    #   fake = aux["pred_image"]
    # Returns the gen-side RpGAN-G term (graph-carrying through
    # ``pred_image`` -> generator) for the caller to add to the DMD
    # loss before the generator backward. The D-side update +
    # backward + optim step happens inside this method.
    #
    # DDP correctness:
    #   * D-update: uses ``disc_for_update`` (DDP-wrapped). Backward
    #     fires on D's own graph; ``fake_detached``/``real_detached``
    #     are not part of G's graph so no grad escapes.
    #   * G-update: uses ``disc_for_guidance`` (UNWRAPPED + frozen).
    #     We ``requires_grad_(False)`` D's params, so the only path
    #     for grad is through ``fake = pred_image`` -> generator's
    #     own DDP backward, which is fired in the outer caller.
    # ------------------------------------------------------------------
    def _compute_r3gan_losses(
        self,
        pred_image: torch.Tensor,
        gt_latents_window: torch.Tensor,
        current_step: int,
    ) -> tuple:
        """Run a D-update on (real, fake) and return the G-side RpGAN
        term (graph-carrying).

        Args:
            pred_image: ``[B, F, C, H, W]`` student rollout pred (graph
                attached). The G-side branch uses this as ``fake``;
                the D-side branch detaches it.
            gt_latents_window: ``[B, F, C, H, W]`` GT latent slice
                covering the same frame indices as ``pred_image``.
                Detached for the D-update; never used for the
                G-side path (G never sees real samples directly in
                R3GAN — only D's pairwise score).
            current_step: training step (for G-side warmup ramp).

        Returns:
            ``(generator_gan_loss, logs)`` — ``generator_gan_loss``
            is graph-carrying; ``logs`` is a flat ``str -> float``
            dict for wandb.
        """
        device = pred_image.device
        zero = torch.zeros((), device=device, dtype=torch.float32)
        if not self.gan_enabled or self.r3gan_disc is None:
            return zero, {}

        disc_for_update = (
            self.r3gan_disc_ddp if self.r3gan_disc_ddp is not None else self.r3gan_disc
        )
        disc_for_guidance = self.r3gan_disc

        # D's own arithmetic stays in fp32 for the second-order
        # gradient penalty. The cost is one extra cast.
        real_detached = gt_latents_window.detach().to(torch.float32)
        fake_detached = pred_image.detach().to(torch.float32)

        # --- D-update (multi-step) -----------------------------------
        d_loss_value = 0.0
        d_real_value = 0.0
        d_fake_detached_value = 0.0
        r1_value = 0.0
        r2_value = 0.0
        for _k in range(max(1, self.gan_updates_per_step)):
            self.r3gan_optimizer.zero_grad(set_to_none=True)
            real_in = real_detached.clone().requires_grad_(True)
            fake_in = fake_detached.clone().requires_grad_(True)
            r1, d_real = r1_penalty(disc_for_update, real_in, gamma=self.gan_r1_gamma)
            r2, d_fake_d = r2_penalty(disc_for_update, fake_in, gamma=self.gan_r2_gamma)
            d_main = rpgan_d_loss(d_real, d_fake_d)
            d_total = d_main + r1 + r2
            d_total.backward()
            if self.gan_max_grad_norm is not None and self.gan_max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in (
                        self.r3gan_disc_ddp.parameters() if self.r3gan_disc_ddp is not None
                        else self.r3gan_disc.parameters()
                    ) if p.grad is not None],
                    self.gan_max_grad_norm,
                )
            self.r3gan_optimizer.step()
            d_loss_value = float(d_main.detach().item())
            d_real_value = float(d_real.detach().mean().item())
            d_fake_detached_value = float(d_fake_d.detach().mean().item())
            r1_value = float(r1.detach().item())
            r2_value = float(r2.detach().item())

        # --- G-side RpGAN term (frozen D) ----------------------------
        # Warmup: ramp G's GAN weight 0 -> gan_loss_weight over
        # gan_warmup_steps steps. D learns from step 0 regardless.
        if self.gan_warmup_steps > 0 and current_step < self.gan_warmup_steps:
            ramp = current_step / max(1, self.gan_warmup_steps)
            gen_gan_weight = ramp * self.gan_loss_weight
        else:
            gen_gan_weight = self.gan_loss_weight

        if gen_gan_weight > 0:
            disc_for_guidance.requires_grad_(False)
            try:
                d_real_for_g = disc_for_guidance(real_detached).detach()
                d_fake_for_g = disc_for_guidance(pred_image.to(torch.float32))
                gen_gan_main = rpgan_g_loss(d_real_for_g, d_fake_for_g)
                generator_gan_loss = gen_gan_weight * gen_gan_main.to(pred_image.dtype)
            finally:
                disc_for_guidance.requires_grad_(True)
            d_fake_for_g_value = float(d_fake_for_g.detach().mean().item())
            gen_gan_main_value = float(gen_gan_main.detach().item())
        else:
            generator_gan_loss = zero
            d_fake_for_g_value = 0.0
            gen_gan_main_value = 0.0

        logs = {
            "train/r3gan_d_loss": d_loss_value,
            "train/r3gan_r1": r1_value,
            "train/r3gan_r2": r2_value,
            "train/r3gan_d_real": d_real_value,
            "train/r3gan_d_fake_detached": d_fake_detached_value,
            "train/r3gan_d_fake_for_g": d_fake_for_g_value,
            "train/r3gan_g_loss_raw": gen_gan_main_value,
            "train/r3gan_g_loss_weighted": (
                float(generator_gan_loss.detach().item())
                if torch.is_tensor(generator_gan_loss)
                and generator_gan_loss.requires_grad
                else 0.0
            ),
            "train/r3gan_g_weight": float(gen_gan_weight),
        }
        return generator_gan_loss, logs

    # ------------------------------------------------------------------
    # SC-DMD warmup helper.
    # ------------------------------------------------------------------
    def _sc_dmd_current_weight(self, current_step: int) -> float:
        """Linear-ramp the SC-DMD weight from 0 → ``sc_dmd_loss_weight``
        over the first ``sc_dmd_warmup_steps`` training steps.

        The SC defect is a SECOND-ORDER consistency regularizer (the
        generator's velocity field appears in both the direct and
        composed paths), so its initial gradient signal can be very
        large at random init and destabilize early DMD training. The
        ramp keeps SC's contribution sub-DMD until the student is
        producing meaningful x0_hat predictions, then phases SC in
        gradually. Mirrors the ``z_guidance_warmup_steps`` recipe used
        for action-critic z-guidance.

        With ``sc_dmd_warmup_steps == 0`` (default) the weight is
        applied at full strength from step 0.
        """
        if not self.sc_dmd_enabled:
            return 0.0
        if self.sc_dmd_warmup_steps <= 0:
            return self.sc_dmd_loss_weight
        if current_step >= self.sc_dmd_warmup_steps:
            return self.sc_dmd_loss_weight
        ramp = float(current_step) / float(max(1, self.sc_dmd_warmup_steps))
        return ramp * self.sc_dmd_loss_weight

    # ------------------------------------------------------------------
    # Periodic pred_image -> wandb.Video sampling.
    # ------------------------------------------------------------------
    def _video_sample_due(self, current_step: int) -> bool:
        """Return True iff this step should produce a sample video.

        Three gates can fire it:
          (1) explicit ``sample_at_steps`` membership (uses ``>=`` so a
              missed gen-step rolls forward to the next eligible iter),
          (2) ``sample_interval > 0`` and ``current_step`` is a positive
              multiple thereof, OR
          (3) ``self._video_sample_due_bit`` is set — meaning a previous
              iter crossed the cadence boundary while no rollout was
              available (critic-only iter under
              dfake_gen_update_ratio>1) and stashed the request for the
              next eligible gen iter to consume.
        All require main rank and at least one viable sink (wandb
        enabled OR ``vis_save_local`` true). ``current_step <= 0`` is
        skipped because no rollout has happened yet at iter 0 entry
        (the first eligible step is ``current_step == 1``).
        """
        if not self.is_main_process:
            return False
        if not (
            (_HAS_WANDB and getattr(self, "wandb_enabled", False))
            or self.vis_save_local
        ):
            return False
        if current_step <= 0:
            return False
        if (
            self._sample_at_steps_pending
            and current_step >= self._sample_at_steps_pending[0]
        ):
            return True
        if self._video_sample_due_bit:
            return True
        if self.sample_interval <= 0:
            return False
        return (current_step % self.sample_interval) == 0

    @torch.no_grad()
    def _log_pred_image_video(
        self,
        pred_image: torch.Tensor,
        step: int,
        name: str = "pred_image",
        caption_suffix: str = "",
        action_overlay: Optional[torch.Tensor] = None,
    ) -> None:
        """Decode ``pred_image`` (a student rollout in latent space) to
        pixel mp4 bytes and upload to wandb under ``sample/<name>``.

        ``name`` controls both the wandb key and the local filename
        suffix: ``step_NNN.mp4`` for the default ``pred_image``, else
        ``step_NNN_<name>.mp4``. ``caption_suffix`` is appended to the
        wandb caption.

        Cheap: re-uses the generator's already-loaded ``WanVAEWrapper``
        (no second VAE copy), runs under ``no_grad``, encodes mp4 in
        memory via ffmpeg subprocess. Best-effort: any failure is
        warned and never raises.

        Only the main rank ever enters this function (the call site
        gates on ``is_main_process``); other ranks are not synchronised
        with this work. There is no NCCL collective inside.

        Caller contract: ``pred_image`` must be a ``[B, F, C, H, W]``
        latent tensor on the GPU. We slice to the first batch element
        and (optionally) the trailing ``sample_max_frames`` frames.

        ``action_overlay``: optional ``[B, F_lat, A]`` (or ``[F_lat, A]``)
        per-latent action stream. When supplied, two horizontal bars
        are rendered along the bottom of every video frame — one per
        action dim — magnitude proportional to ``|z|`` (red for ``z>=0``,
        blue for ``z<0``). The dataset enforces one z per chunk
        (``num_frame_per_block`` latents share the same z), so the
        overlay is constant within each chunk and steps at chunk
        boundaries. Used only for ``clean_x_real`` to cross-check the
        bidir scorer's clean-half action conditioning against the
        decoded pixel content.
        """
        if pred_image is None or pred_image.numel() == 0:
            return
        vae = getattr(self.model, "vae", None)
        if vae is None:
            if not self._video_logger_warned_no_vae:
                logging.warning(
                    "[ActionForcing] sample_interval set but "
                    "self.model.vae is None; sample videos disabled."
                )
                self._video_logger_warned_no_vae = True
            return
        try:
            latents = pred_image.detach().to(torch.float32)
            if latents.dim() != 5:
                return
            if self.sample_max_frames > 0:
                F_total = latents.shape[1]
                if F_total > self.sample_max_frames:
                    latents = latents[:, -self.sample_max_frames:]
            lat = latents[0:1]
            dummy = lat[:, 0:1]
            lat_wd = torch.cat([dummy, lat], dim=1)
            pixels = vae.decode_to_pixel(lat_wd)[:, 1:, ...]
            video = (0.5 * (pixels.float() + 1.0)).clamp(0.0, 1.0)
            vid_np = (video[0].cpu().numpy() * 255.0).astype(np.uint8)
            if vid_np.ndim != 4:
                return
            if vid_np.shape[-1] != 3:
                vid_np = vid_np.transpose(0, 2, 3, 1)
        except Exception as exc:
            logging.warning(
                "[ActionForcing] sample-video VAE decode failed at "
                "step=%d: %s", step, exc,
            )
            return

        # Render the action overlay (best-effort, never raises).
        if action_overlay is not None:
            try:
                npb = int(getattr(self.config, "num_frame_per_block", 3))
                self._draw_action_overlay(
                    vid_np, action_overlay, frames_per_latent=4, npb=npb,
                )
            except Exception as exc:
                logging.warning(
                    "[ActionForcing] action overlay failed at step=%d "
                    "(name=%s): %s", step, name, exc,
                )

        mp4_bytes = _frames_to_mp4_bytes(
            vid_np, fps=float(self.sample_fps),
        )
        if mp4_bytes is None:
            logging.warning(
                "[ActionForcing] sample-video ffmpeg encode failed at "
                "step=%d (frames=%d)", step, vid_np.shape[0],
            )
            return

        if self.vis_save_local:
            try:
                samples_dir = Path(self.log_dir) / "samples"
                samples_dir.mkdir(parents=True, exist_ok=True)
                _suffix = "" if name == "pred_image" else f"_{name}"
                out_path = samples_dir / f"step_{int(step):07d}{_suffix}.mp4"
                with open(out_path, "wb") as fh:
                    fh.write(mp4_bytes)
                logging.info(
                    "[ActionForcing] Saved sample video to %s "
                    "(frames=%d, fps=%d)",
                    out_path, vid_np.shape[0], int(self.sample_fps),
                )
            except Exception as exc:
                logging.warning(
                    "[ActionForcing] sample-video local save failed at "
                    "step=%d: %s", step, exc,
                )

        if not (_HAS_WANDB and getattr(self, "wandb_enabled", False)):
            return

        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".mp4", delete=False,
            ) as fh:
                fh.write(mp4_bytes)
                tmp_path = fh.name
            _caption = f"step {step}"
            if caption_suffix:
                _caption = f"{_caption} {caption_suffix}"
            wandb.log(
                {
                    f"sample/{name}": wandb.Video(
                        tmp_path,
                        fps=int(self.sample_fps),
                        format="mp4",
                        caption=_caption,
                    ),
                    "sample/step": int(step),
                    f"sample/{name}_num_frames": int(vid_np.shape[0]),
                },
                step=step,
            )
            logging.info(
                "[ActionForcing] Logged sample/%s video at step=%d "
                "(frames=%d, fps=%d)",
                name, step, vid_np.shape[0], int(self.sample_fps),
            )
        except Exception as exc:
            logging.warning(
                "[ActionForcing] sample-video wandb upload failed at "
                "step=%d: %s", step, exc,
            )
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    @staticmethod
    def _draw_action_overlay(
        vid_np: np.ndarray,
        action_overlay,
        frames_per_latent: int = 4,
        npb: int = 3,
    ) -> None:
        """Draw per-chunk action bars at the bottom of every frame.

        ``vid_np``: ``[T_video, H, W, 3]`` uint8, edited in place.
        ``action_overlay``: ``[B, F_lat, A]`` or ``[F_lat, A]`` tensor /
        ndarray of post-tanh-squash z values in roughly ``[-1, 1]``.

        After Wan VAE temporal upsampling, ``T_video = F_lat *
        frames_per_latent`` (the leading dummy frame was already
        stripped off by the caller). One motion entry covers ``npb``
        consecutive latents; we render one bar value per chunk so the
        overlay is constant across the chunk's ``npb *
        frames_per_latent`` video frames and steps at chunk
        boundaries.

        Two stacked horizontal bars: top = z[0], bottom = z[1] (or up
        to ``A`` bars if A>2). Red bar to the right for ``z>=0``,
        blue bar to the left for ``z<0``; length scales with ``|z|``
        (clamped to 1.0 → half the frame width).
        """
        if action_overlay is None:
            return
        if torch.is_tensor(action_overlay):
            arr = action_overlay.detach().float().cpu().numpy()
        else:
            arr = np.asarray(action_overlay, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]  # batch 0 — rank-local rollout uses batch dim 0
        if arr.ndim != 2:
            return
        F_lat, A = arr.shape
        if F_lat == 0 or A == 0:
            return
        # Per-chunk z (one row per chunk; the dataset already
        # broadcasts within-chunk so all 3 latents in a chunk share
        # the same row — we just take every npb-th).
        z_per_chunk = arr[::npb]
        n_chunks = z_per_chunk.shape[0]
        T, H, W, _ = vid_np.shape
        if H < 8 or W < 16 or n_chunks == 0:
            return

        # Strip layout: 4-px top padding then A bars of equal height
        # with 2-px gaps; total bound by ~25% of frame height.
        max_strip = max(20, H // 4)
        gap = 2
        bar_h = max(3, (max_strip - 4 - gap * (A - 1)) // A)
        strip_h = 4 + bar_h * A + gap * (A - 1)
        cx = W // 2
        max_bar = (W // 2) - 4

        # Black underlay so the bars are readable on bright frames.
        vid_np[:, -strip_h:, :, :] = vid_np[:, -strip_h:, :, :] // 4

        frames_per_chunk = npb * frames_per_latent
        for t in range(T):
            chunk_idx = min(t // frames_per_chunk, n_chunks - 1)
            z = z_per_chunk[chunk_idx]
            for d in range(A):
                y0 = H - strip_h + 4 + d * (bar_h + gap)
                y1 = y0 + bar_h
                a = float(z[d])
                length = int(min(abs(a), 1.0) * max_bar)
                # Center marker (gray 1-px line).
                vid_np[t, y0:y1, cx - 1:cx + 1, :] = 128
                if length <= 0:
                    continue
                if a >= 0:
                    vid_np[t, y0:y1, cx:cx + length, 0] = 255
                    vid_np[t, y0:y1, cx:cx + length, 1] = 0
                    vid_np[t, y0:y1, cx:cx + length, 2] = 0
                else:
                    vid_np[t, y0:y1, cx - length:cx, 0] = 0
                    vid_np[t, y0:y1, cx - length:cx, 1] = 0
                    vid_np[t, y0:y1, cx - length:cx, 2] = 255

    # ------------------------------------------------------------------
    # Step-scheduled local_attn_size (KV cache window) helpers.
    # ------------------------------------------------------------------
    def _current_attn_size_frames(self) -> Optional[int]:
        """Return the active ``local_attn_size`` (in frames) for
        ``self.step`` per the configured schedule, or None if no
        schedule is set."""
        if not self._attn_size_schedule:
            return None
        active = self._attn_size_schedule[0][1]
        for thr, frames in self._attn_size_schedule:
            if self.step >= thr:
                active = frames
            else:
                break
        return int(active)

    def _apply_attn_size_if_changed(self) -> bool:
        """If the schedule says the active ``local_attn_size`` differs
        from what we last applied, push the new value through the
        model's attention layers + the pipeline's cache-sizing knob,
        and force a streaming-state reset so the next sequence open
        re-allocates the KV cache at the new size. Returns True iff
        anything changed.

        Why reset instead of resizing in place: ``_initialize_kv_cache``
        allocates fixed-size ``[B, kv_cache_size, 12, 128]`` buffers per
        layer. There's no in-place resize for those buffers, and growing
        the attention window past the current cache size mid-sequence
        would let the model attend to slots that don't exist yet. A
        sequence reset is the cheapest atomic transition: closes the
        old sequence (releases old buffers next setup), then the next
        ``setup_sequence`` allocates at the new size.
        """
        target = self._current_attn_size_frames()
        if target is None or target == self._current_attn_frames:
            return False

        # Walk the bare DiT (under DDP / wrapper) and update each
        # CausalWanSelfAttention's local_attn_size. The infinity_rope
        # patch reads ``self.local_attn_size`` per attention forward,
        # so updating the attribute takes effect on the next forward
        # without re-installing the patch.
        dit = self._inner_dit_for_rope()
        if hasattr(dit, "local_attn_size"):
            dit.local_attn_size = target
        if hasattr(dit, "blocks"):
            for blk in dit.blocks:
                if hasattr(blk, "local_attn_size"):
                    blk.local_attn_size = target
                if hasattr(blk, "self_attn") and hasattr(blk.self_attn, "local_attn_size"):
                    blk.self_attn.local_attn_size = target

        # Bump the pipeline's rollout_frames so the next
        # ``_initialize_kv_cache`` call sizes the buffer to ``target``.
        # ``kv_cache_size = max(num_max_frames, rollout_frames) *
        # frame_seq_length`` (pipeline.kv_cache_size property).
        if self.pipeline is not None:
            self.pipeline.rollout_frames = max(
                int(self.pipeline.num_max_frames), int(target)
            )

        # Force the NEXT _fwdbwd_streaming_step to open a new sequence
        # (which calls setup_sequence -> _initialize_kv_cache with the
        # new size). reset_streaming_state is a no-op when state is
        # already None, so safe to call unconditionally.
        if hasattr(self.model, "reset_streaming_state"):
            self.model.reset_streaming_state()

        prev = self._current_attn_frames
        self._current_attn_frames = int(target)
        if self.is_main_process:
            logging.info(
                "[ActionForcing] local_attn_size schedule transition at "
                "step=%d: %s -> %d frames (KV cache buffers will "
                "re-allocate on next sequence open).",
                int(self.step), str(prev), int(target),
            )
        return True

    # ------------------------------------------------------------------
    # Motion-aware rollout offset picker.
    # ------------------------------------------------------------------
    def _pick_motion_aware_offset(
        self,
        ride: Dict[str, Any],
        s_local_max: int,
        window_lo_offset: int,
        window_len: int,
    ) -> int:
        """Pick a starting offset ``s ∈ [0, s_local_max]`` that gives a
        motion-positive rollout window when possible.

        With probability ``cfg.motion_start_prob`` (default 1.0),
        restricts the candidate set to offsets whose rollout window
        ``ride[s + window_lo_offset : s + window_lo_offset + window_len]``
        has mean per-frame motion magnitude ≥ ``cfg.motion_start_threshold``
        (default 0.5 — empirically separates parked / idling windows
        from clearly-moving ones across the dataset). With probability
        1 - prob, samples uniformly. If motion is unavailable for this
        ride or no candidate meets the threshold, falls back to uniform
        sampling on ``[0, s_local_max]``.

        Per-rank: each rank picks independently from its own ride's
        motion. The legacy DDP pattern broadcast a rank-0 pick (so
        every rank used the same ``s``) — that broke as soon as we
        wanted motion-aware selection because rank 0's motion-positive
        offset is meaningless on rank 1's ride. Cross-rank sync is
        instead preserved via per-rank ``s_local_max`` ensuring
        ``s + cf + cap <= ride_len_r`` on every rank, so ``actual_cap
        = cap`` is invariant without any all_reduce.

        Chunk-boundary snap: the dataset enforces per-chunk identity on
        ``z_actions`` (one z per ``num_frame_per_block``-frame chunk;
        see utils/zarr_dataset.py:_LATENTS_PER_MOTION_CHUNK). Slicing
        the ride at an ``s`` that isn't a multiple of npb would put a
        chunk boundary mid-window in the model's view, producing a
        mid-chunk action transition that's OOD for v14's training
        contract (constant action within each chunk). We snap every
        return value to a multiple of npb here so all downstream
        slices are chunk-aligned by construction.
        """
        npb = int(getattr(self.config, "num_frame_per_block", 3))
        # Cap s_local_max to a multiple of npb so the snap below never
        # produces a value that exceeds the per-rank max.
        s_local_max = (max(0, int(s_local_max)) // npb) * npb
        if s_local_max <= 0:
            return 0

        def _snap(val: int) -> int:
            return (max(0, int(val)) // npb) * npb

        # ``random.randint(a, b)`` is INCLUSIVE on b; with b = s_local_max
        # already npb-aligned, snapping is idempotent on b. Other values
        # in (0, b) snap DOWN, which is fine — uniform-on-multiples-of-npb
        # is what we want.
        prob = float(getattr(self.config, "motion_start_prob", 1.0))
        threshold = float(getattr(self.config, "motion_start_threshold", 0.5))
        motion_mag = ride.get("motion_mag")
        use_motion = (
            motion_mag is not None
            and prob > 0.0
            and (prob >= 1.0 or random.random() < prob)
        )
        if not use_motion:
            return _snap(random.randint(0, s_local_max))
        mag_np = (
            motion_mag.cpu().numpy()
            if torch.is_tensor(motion_mag)
            else np.asarray(motion_mag)
        )
        n_lat = mag_np.shape[0]
        win_hi_max = s_local_max + window_lo_offset + window_len
        if win_hi_max > n_lat or window_len <= 0:
            return _snap(random.randint(0, s_local_max))
        # Sliding-window mean via cumulative sum.
        # window_means[s] = mean of mag_np[s+lo : s+lo+win] for
        # s ∈ [0, s_local_max].
        csum = np.concatenate(
            [[0.0], np.cumsum(mag_np, dtype=np.float64)],
        )
        lo = window_lo_offset
        n_candidates = s_local_max + 1
        sums = (
            csum[lo + window_len : lo + window_len + n_candidates]
            - csum[lo : lo + n_candidates]
        )
        means = sums / float(window_len)
        valid_idx = np.flatnonzero(means >= threshold)
        if valid_idx.size == 0:
            return _snap(random.randint(0, s_local_max))
        # Restrict to npb-aligned candidates among valid_idx so we
        # don't bias toward pseudo-aligned offsets via post-hoc snap.
        # ``valid_idx`` is sorted (np.flatnonzero output), so the
        # filter just keeps the multiples-of-npb entries; if none are
        # valid, fall back to snapping a random valid offset.
        aligned_valid = valid_idx[valid_idx % npb == 0]
        if aligned_valid.size > 0:
            return int(aligned_valid[random.randrange(aligned_valid.size)])
        return _snap(int(valid_idx[random.randrange(valid_idx.size)]))

    # ------------------------------------------------------------------
    # Streaming slide-and-train (Phase-1 default). Replaces the legacy
    # _streaming_maybe_extend "base + extensions" scheme. Per ride:
    # slide a 21-frame window forward (one ``generate_next_chunk`` per
    # slide; iter-1 advances chunk_size frames, subsequent iters
    # advance by ``num_chunks_roll_forward * num_frame_per_block``
    # frames per the model's deterministic-stride mode). Compute avg
    # MAE on the FULL 21-frame window vs the corresponding GT slice.
    # The FIRST window whose avg MAE crosses ``mae_extension_threshold``
    # is the trained window — DMD pulls the student toward real_score
    # exactly where the student is wrong. A ride that stays under the
    # threshold across the cap / end-of-ride is "too easy": no optim
    # signal, the next iter pulls a fresh ride.
    # ------------------------------------------------------------------
    def _streaming_roll_and_train_one_window(
        self,
        rollout_frames: int,
        cf_dmdctx: int,
    ) -> Optional[Dict[str, Any]]:
        """Slide-and-train: roll until avg MAE > threshold, then train
        gen+critic on that window. Returns ``None`` if the ride loader
        is exhausted; otherwise returns a flat ``str -> float`` log
        dict (with ``streaming_window_too_easy: 1.0`` on the no-op
        exit). Always closes the streaming sequence on exit so the
        next iter pulls a fresh ride.
        """
        cfg = self.config
        threshold = float(cfg.mae_extension_threshold)
        max_slides = int(cfg.mae_extension_max_extra_chunks)

        # Open a fresh sequence — single trained window per ride means
        # we never carry streaming state across iters.
        self.model.reset_streaming_state()
        if not self._streaming_setup_sequence_from_ride(
            rollout_frames=rollout_frames,
            max_total_rollout_frames=rollout_frames,
            cf_dmdctx=cf_dmdctx,
        ):
            return None

        state = self.model.streaming_state
        npb = int(state["shift"])
        chunk_size = int(state["chunk_size"])
        cf_state = int(state["cf"])

        # Pre-pick an exit-rung index ONCE per training step and
        # broadcast from rank 0. Every slide reuses this same index
        # (passed as ``force_exit_step`` to ``generate_next_chunk``),
        # so all ranks denoise to the same exit step on every slide
        # without firing a per-slide ``dist.broadcast``. This collapses
        # the original per-call collective (which deadlocked on per-
        # rank-divergent slide counts) into 1 collective per training
        # step. ``last_step_only=True`` is honoured by the same shortcut
        # the per-call sampler uses.
        pipe = self.model.inference_pipeline
        n_steps = len(pipe.denoising_step_list)
        if pipe.last_step_only:
            forced_exit_step: int = n_steps - 1
        elif dist.is_initialized() and dist.get_world_size() > 1:
            if dist.get_rank() == 0:
                idx_t = torch.randint(
                    0, n_steps, (1,),
                    device=self.device, dtype=torch.long,
                )
            else:
                idx_t = torch.empty(1, dtype=torch.long, device=self.device)
            dist.broadcast(idx_t, src=0)
            forced_exit_step = int(idx_t.item())
        else:
            forced_exit_step = int(torch.randint(0, n_steps, (1,)).item())

        chunks_rolled = 0
        stop_reason: str = "unknown"
        train_chunk: Optional[torch.Tensor] = None
        train_info: Optional[Dict[str, Any]] = None
        train_avg_mae: float = float("nan")
        # Hold onto the most recently rolled chunk. Used (a) as the
        # synthetic-backward chunk on ranks whose own slide loop ended
        # without a trained window but at least one peer did train, and
        # (b) discarded outright when no rank trained.
        prev_chunk: Optional[torch.Tensor] = None
        prev_info: Optional[Dict[str, Any]] = None
        # Per-slide MAE history for stdout reporting at the end of the
        # slide loop. One entry per generate_next_chunk call. Helps
        # diagnose "the helper extends too far before training" cases.
        slide_maes: List[float] = []

        while True:
            if chunks_rolled >= max_slides:
                stop_reason = "cap"
                break
            if not self.model.can_generate_more():
                stop_reason = "end_of_ride"
                break

            # Per-rank-divergent slide loop — each rank rolls a
            # different number of chunks based on its own ride's MAE,
            # so EVERY DDP collective inside ``generate_next_chunk``
            # must be bypassed or the slide counts mismatch and NCCL
            # deadlocks at the divergence point. Two collectives reach
            # this hot path:
            #   * ``_compute_chunk_mae`` ⇒ gated by
            #     ``compute_baseline_mae=False`` (no gt_chunk → no
            #     all_reduce). The local MAE we compute below is what
            #     drives slide-and-train; baseline telemetry isn't
            #     read on this path.
            #   * ``generate_and_sync_list`` ⇒ gated by
            #     ``force_exit_step=forced_exit_step`` (the index was
            #     broadcast ONCE before the loop; every slide reuses it
            #     so no per-call collective fires). All ranks still
            #     denoise to the same exit rung — lockstep preserved.
            chunk, info = self.model.generate_next_chunk(
                requires_grad=True,
                compute_baseline_mae=False,
                force_exit_step=forced_exit_step,
            )
            chunks_rolled += 1

            # GT slice covering the chunk's noisy half (= the full 21
            # frames the chunk represents in cumulative-sdn coords).
            noisy_start_sdn = int(
                info["current_length"]
                - info["new_frames"]
                - info["overlap"]
            )
            chunk_lo = cf_state + noisy_start_sdn
            chunk_hi = chunk_lo + chunk_size
            ride_window = state["ride_latents_window"]
            # Geometry invariant (proof): ``_streaming_setup_sequence_
            # from_ride`` MIN-reduces ``actual_cap`` across ranks, then
            # builds ``ride_latents_window`` of length cf + actual_cap
            # (= cf_state + max_length). ``can_generate_more()`` keeps
            # ``current_length ≤ max_length``. With chunk_size =
            # new_frames + overlap and noisy_start_sdn = current_length
            # - new_frames - overlap, chunk_hi simplifies to cf_state +
            # current_length ≤ cf_state + max_length =
            # ride_window.shape[1]. Symmetric across ranks because every
            # term is rank-invariant under the MIN-reduce + the
            # streaming_force_new_frame_chunks deterministic-stride. If
            # this ever fires it's a real bug, not a corner case —
            # surface with a hard error on every rank simultaneously.
            assert ride_window.shape[1] >= chunk_hi, (
                f"slide-loop geometry violation: ride_window.shape[1]="
                f"{ride_window.shape[1]} < chunk_hi={chunk_hi}. "
                f"setup_sequence MIN-reduces actual_cap and "
                f"can_generate_more() keeps current_length ≤ max_length."
            )
            gt_slice = ride_window[:, chunk_lo:chunk_hi]
            # Per-rank MAE on the full 21-frame window. No all_reduce —
            # each rank decides locally based on its own ride. The
            # post-loop MAX-reduce on local_trained handles cross-rank
            # DDP coordination.
            avg_mae = float(
                (chunk.detach().float() - gt_slice.float())
                .abs().mean().item()
            )
            slide_maes.append(avg_mae)

            # Stage as the most-recent leftover, releasing the previous.
            if prev_chunk is not None:
                del prev_chunk, prev_info
            prev_chunk, prev_info = chunk, info

            if avg_mae == avg_mae and avg_mae > threshold:
                stop_reason = "mae_threshold"
                train_chunk = chunk
                train_info = info
                train_avg_mae = avg_mae
                break

        out: Dict[str, Any] = {
            "streaming_chunks_rolled": float(chunks_rolled),
            f"streaming_stop_reason_{stop_reason}": 1.0,
        }

        # Per-slide MAE printout — log on EVERY rank (with rank tag) so
        # we can verify the per-rank decisions diverge as expected
        # (each rank slides on its own ride and decides locally based
        # on its own MAE). Helps diagnose "is the helper extending too
        # far before training" + cross-rank divergence. Uses WARNING
        # level because the parent trainer sets non-main ranks to
        # logging.WARNING; INFO would be silently dropped on rank>0.
        rank = dist.get_rank() if dist.is_initialized() else 0
        if slide_maes:
            mae_str = ", ".join(f"{m:.4f}" for m in slide_maes)
            logging.warning(
                "[ActionForcing] slide_loop rank=%d step=%d threshold=%.3f "
                "stop_reason=%s slide_maes=[%s]",
                rank, int(self.step) + 1, threshold, stop_reason, mae_str,
            )

        # Cross-rank coordination: did ANY rank find a hard window? If
        # so, ranks that didn't cross threshold fall back to ``prev_chunk``
        # (the last rolled chunk) as their training target. This keeps
        # generator_ddp + fake_score_ddp all_reduces in lockstep AND
        # gives stragglers a real gradient contribution instead of a
        # zero-loss no-op (the chunks are sub-threshold but still carry
        # signal — the slide loop's forward compute is no longer wasted).
        local_trained = 1 if train_chunk is not None else 0
        if dist.is_initialized() and dist.get_world_size() > 1:
            flag_t = torch.tensor(
                [local_trained], device=self.device, dtype=torch.long,
            )
            dist.all_reduce(flag_t, op=dist.ReduceOp.MAX)
            any_trained = int(flag_t.item()) == 1
        else:
            any_trained = bool(local_trained)

        if not any_trained:
            # Every rank's slide loop finished without crossing the
            # threshold. No backward fires anywhere; the trainer's
            # outer optim.step() runs on zero grads (safe path).
            out["streaming_window_too_easy"] = 1.0
            if prev_chunk is not None:
                del prev_chunk, prev_info
            self.model.reset_streaming_state()
            return out

        if train_chunk is None:
            # This rank didn't cross threshold but at least one peer did.
            # Promote ``prev_chunk`` (the last rolled chunk on this rank)
            # to train_chunk and continue through the unified backward
            # path. Real loss, real gradient — the rank still contributes
            # to the DDP-averaged update, just on a sub-threshold window.
            #
            # ``prev_chunk is not None`` here: setup_sequence rejects
            # rides too small for ``anchor_frames + min_new``, so the
            # loop always rolls at least one chunk and stages it as
            # prev_chunk before any in-body break. The geometry-guard
            # assert above eliminates the only path that could leave
            # the loop with prev_chunk unset.
            assert prev_chunk is not None, (
                "straggler-fallback invariant broken: peer rank trained "
                "but this rank has no staged prev_chunk. The slide loop "
                "should have rolled at least one chunk before exiting."
            )
            train_chunk = prev_chunk
            train_info = prev_info
            train_avg_mae = slide_maes[-1] if slide_maes else float("nan")
            out["streaming_straggler_fallback"] = 1.0

        out["streaming_window_avg_mae"] = float(train_avg_mae)
        out["streaming_window_start_chunk"] = float(chunks_rolled)

        aux_active = (
            self.action_critic_loss_active
            and self.critic_optimizer is not None
        )
        gan_active = (
            self.gan_enabled
            and self.r3gan_disc is not None
            and self.r3gan_optimizer is not None
        )
        sc_dmd_active = bool(self.sc_dmd_enabled)

        # Stash the trained chunk for the periodic wandb sample-video
        # logger (parity with _fwdbwd_streaming_step). DMD-eval stash
        # arming is identical; the eval video logger reads it after
        # compute_generator_loss_streaming populates the views.
        _sample_due_now = self._video_sample_due(int(self.step) + 1)
        if _sample_due_now:
            try:
                self._pending_video_latents = (
                    train_chunk.detach().to(torch.float32)
                )
                _cs = int(self.step) + 1
                while (
                    self._sample_at_steps_pending
                    and self._sample_at_steps_pending[0] <= _cs
                ):
                    self._sample_at_steps_pending.pop(0)
                self._video_sample_due_bit = False
            except Exception as _exc:
                logging.warning(
                    "[ActionForcing] failed to stash slide-and-train "
                    "chunk for video at step=%d: %s",
                    int(self.step) + 1, _exc,
                )
                self._pending_video_latents = None
            self.model._dmd_eval_stash = {}
        else:
            self.model._dmd_eval_stash = None

        gen_loss_dmd, gen_log = self.model.compute_generator_loss_streaming(
            train_chunk, train_info,
        )

        if _sample_due_now:
            eval_stash = getattr(self.model, "_dmd_eval_stash", None)
            if isinstance(eval_stash, dict) and eval_stash:
                self._pending_dmd_eval_latents = eval_stash
            self.model._dmd_eval_stash = None

        out["generator_dmd_loss"] = float(gen_loss_dmd.detach().item())
        out.update({
            k: (float(v.detach().float().mean().item())
                if torch.is_tensor(v) else v)
            for k, v in gen_log.items()
            if not isinstance(v, dict)
        })

        generator_loss = gen_loss_dmd

        # Aux / GAN / SC-DMD chunk geometry. ``state["current_length"]``
        # has already advanced past train_chunk by the time we get
        # here; the chunk's noisy half lives at
        # ``cf + (train_info[current_length] - new_frames - overlap) :
        #  ... + chunk_size``.
        noisy_start_sdn = int(
            train_info["current_length"]
            - train_info["new_frames"]
            - train_info["overlap"]
        )
        chunk_lo = cf_state + noisy_start_sdn
        chunk_hi = chunk_lo + chunk_size

        if aux_active:
            actions_chunk = state["ride_actions_window"][:, chunk_lo:chunk_hi]
            actions_for_critic = self._slice_actions_for_critic(
                actions_chunk
            ).to(train_chunk.dtype)
            ts_value = train_info.get("denoised_timestep_from", None)
            ts_int = int(ts_value) if ts_value is not None else 0
            B = train_chunk.shape[0]
            n_chunks = train_chunk.shape[1] // int(self.config.num_frame_per_block)
            chunk_t = torch.full(
                (B, n_chunks), ts_int,
                device=train_chunk.device, dtype=torch.long,
            )
            gen_action_loss, critic_logs, _teacher_z = (
                self._compute_action_critic_losses(
                    pred_x0=train_chunk,
                    target_action_z=actions_for_critic,
                    chunk_t=chunk_t,
                    current_step=int(self.step),
                )
            )
            generator_loss = generator_loss + gen_action_loss
            out.update(critic_logs)

        if gan_active:
            gt_window = state["ride_latents_window"][:, chunk_lo:chunk_hi]
            gen_gan_loss, gan_logs = self._compute_r3gan_losses(
                pred_image=train_chunk,
                gt_latents_window=gt_window,
                current_step=int(self.step),
            )
            generator_loss = generator_loss + gen_gan_loss
            out.update(gan_logs)

        if sc_dmd_active:
            sc_loss_raw, sc_logs = self.model.sc_dmd_loss(
                conditional_dict=train_info["conditional_dict"],
                clean_latent=state["ride_latents_window"][:, cf_state:],
                seed_frames=cf_state,
            )
            sc_weight = self._sc_dmd_current_weight(int(self.step))
            weighted_sc = sc_loss_raw * sc_weight
            generator_loss = generator_loss + weighted_sc
            out.update(sc_logs)
            out["sc_dmd_weight_effective"] = float(sc_weight)
            out["sc_dmd_loss_weighted"] = float(weighted_sc.detach().item())

        out["generator_loss"] = float(generator_loss.detach().item())
        # retain_graph=True so the critic backward can walk the shared
        # cond_dict / action_projection subgraph that both losses use.
        generator_loss.backward(retain_graph=True)

        critic_loss, critic_log = self.model.compute_critic_loss_streaming(
            train_chunk, train_info,
        )
        out["critic_loss"] = float(critic_loss.detach().item())
        out.update({
            k: (float(v.detach().float().mean().item())
                if torch.is_tensor(v) else v)
            for k, v in critic_log.items()
            if not isinstance(v, dict)
        })
        critic_loss.backward()

        # Close the sequence — exactly one trained window per ride.
        self.model.reset_streaming_state()
        return out

    # ------------------------------------------------------------------
    # Streaming-mode helpers (LongLive parity).
    # ------------------------------------------------------------------
    def _streaming_setup_sequence_from_ride(
        self,
        rollout_frames: int,
        max_total_rollout_frames: int,
        cf_dmdctx: int,
    ) -> bool:
        """Pull a ride, pick a random offset s, slice the seed +
        rollout window, and call ``model.setup_sequence`` to open a
        new streaming sequence. Returns True on success, False if no
        ride was available.
        """
        npb = int(getattr(self.config, "num_frame_per_block", 3))
        cap = int(self.streaming_max_length)
        if cap % npb != 0:
            cap = (cap // npb) * npb

        # Need at least cf + cap frames of ride post-offset.
        ride = self._next_ride(rollout_frames + cf_dmdctx)
        if ride is None:
            return False
        ride_len = int(ride["latents"].shape[1])

        # Per-rank motion-aware s pick. Each rank biases toward its
        # OWN ride's motion (the legacy MIN-reduce + rank-0 broadcast
        # pattern would force every rank to the same offset, which is
        # meaningless when each rank has a different ride and motion
        # profile). Per-rank ``s_local_max`` already ensures
        # ``s + cf + cap ≤ ride_len_r`` so ``actual_cap = cap`` on every
        # rank without coordination. ``rollout_frames`` is the window
        # we want motion in (= the first valid scoring window of the
        # streaming sequence).
        s_local_max = max(
            0, min(ride_len // 2, ride_len - cf_dmdctx - cap),
        )
        s = self._pick_motion_aware_offset(
            ride,
            s_local_max=s_local_max,
            window_lo_offset=cf_dmdctx,
            window_len=int(rollout_frames),
        )

        # Cap the actual rollout length to what fits this ride
        # (s + cf + cap ≤ ride_len). ``ride_len`` is per-rank
        # (DistributedSampler hands different rides to different ranks),
        # so ``actual_cap`` diverges across ranks → DDP all-reduces
        # inside ``_streaming_pick_new_frames`` / per-chunk MAE / etc.
        # would get out of step. MIN-reduce to the lowest-fitting cap
        # so all ranks set up sequences of the same max_length and
        # advance in lockstep. (Single-rank path skips the reduce.)
        actual_cap = min(cap, ride_len - cf_dmdctx - s)
        if actual_cap % npb != 0:
            actual_cap = (actual_cap // npb) * npb
        if dist.is_initialized() and dist.get_world_size() > 1:
            t = torch.tensor(
                [actual_cap], device=self.device, dtype=torch.long,
            )
            dist.all_reduce(t, op=dist.ReduceOp.MIN)
            actual_cap = int(t.item())
        # Reject if the ride can't fit the +npb anchor + at least one
        # valid ``generate_next_chunk`` call. ``setup_sequence`` rolls
        # an anchor chunk that initialises ``current_length = npb``, so
        # ``can_generate_more()`` requires ``npb + min_new_frame ≤ max_length``
        # → reject if ``actual_cap < npb + min_new_frame``.
        min_new = int(getattr(self.model, "streaming_min_new_frame", npb))
        anchor_frames = int(getattr(self.model, "dmd_clean_x_anchor_frames", npb))
        if actual_cap < anchor_frames + min_new:
            return False

        prompt_embeds = ride["prompt_embeds"]
        seed_latents = ride["latents"][:, s : s + cf_dmdctx].contiguous()
        # Window covers seed + rollout: ride_*[s : s + cf + actual_cap].
        ride_lat_window = ride["latents"][:, s : s + cf_dmdctx + actual_cap].contiguous()
        ride_act_window = ride["z_actions"][:, s : s + cf_dmdctx + actual_cap].contiguous()

        self.model.setup_sequence(
            seed_latents=seed_latents,
            ride_latents_window=ride_lat_window,
            ride_actions_window=ride_act_window,
            prompt_embeds=prompt_embeds,
            max_length=int(actual_cap),
        )
        return True

    def _fwdbwd_streaming_step(
        self,
        train_generator: bool,
        rollout_frames: int,
        max_total_rollout_frames: int,
        cf_dmdctx: int,
    ) -> Optional[Dict[str, Any]]:
        """Streaming-mode per-iter step: open a sequence if none open,
        advance by ``new_frames``, score DMD on the chunk, backward.
        Collapse gate at end of step decides whether to keep rolling
        next iter.

        When ``mae_extension_threshold is not None and
        mae_extension_max_extra_chunks > 0`` (the Phase-1 freeze-config
        default), the gen-iter path routes to
        ``_streaming_roll_and_train_one_window`` (slide-and-train) and
        the standalone critic-iter call is a no-op (the helper already
        ran the critic backward on the same window). The legacy
        per-chunk-trains-immediately + ``_streaming_maybe_extend``
        chain is unreachable in that mode.
        """
        cfg = self.config
        extension_active = (
            getattr(cfg, "mae_extension_threshold", None) is not None
            and int(getattr(cfg, "mae_extension_max_extra_chunks", 0)) > 0
        )
        if extension_active:
            if train_generator:
                return self._streaming_roll_and_train_one_window(
                    rollout_frames=rollout_frames,
                    cf_dmdctx=cf_dmdctx,
                )
            # Critic-iter no-op: the helper already trained the critic
            # on the same window during the gen-iter call. Running
            # another critic update here would either re-train on a
            # fresh ride (= different chunk than the gen update — DMD2
            # decoupling) or roll into a closed sequence. Either is
            # incorrect; skip with telemetry.
            return {"streaming_critic_skipped": 1.0}

        # Legacy per-chunk path (extension_active=False). Open a
        # sequence if needed.
        if (
            self.model.streaming_state is None
            or not self.model.can_generate_more()
        ):
            self.model.reset_streaming_state()
            if not self._streaming_setup_sequence_from_ride(
                rollout_frames=rollout_frames,
                max_total_rollout_frames=max_total_rollout_frames,
                cf_dmdctx=cf_dmdctx,
            ):
                return None

        if train_generator:
            aux_active = (
                self.action_critic_loss_active
                and self.critic_optimizer is not None
            )
            gan_active = (
                self.gan_enabled
                and self.r3gan_disc is not None
                and self.r3gan_optimizer is not None
            )
            sc_dmd_active = bool(self.sc_dmd_enabled)

            chunk, info = self.model.generate_next_chunk(requires_grad=True)

            # Stash the chunk for the periodic wandb sample-video logger
            # (parity with the legacy aux/plain branches). ``chunk``
            # includes the overlap so the rendered mp4 reflects what
            # DMD just scored. Detach + float32 to release the
            # autograd graph (logger never backwards through this).
            _sample_due_now = self._video_sample_due(int(self.step) + 1)
            if _sample_due_now:
                try:
                    self._pending_video_latents = (
                        chunk.detach().to(torch.float32)
                    )
                    _cs = int(self.step) + 1
                    while (
                        self._sample_at_steps_pending
                        and self._sample_at_steps_pending[0] <= _cs
                    ):
                        self._sample_at_steps_pending.pop(0)
                    # Consume the sticky deferral bit (set when a
                    # previous critic-only iter crossed the cadence
                    # boundary). Safe to clear unconditionally — if it
                    # was already False the periodic ``%==0`` gate or
                    # ``sample_at_steps`` triggered us, and clearing has
                    # no effect.
                    self._video_sample_due_bit = False
                except Exception as _exc:
                    logging.warning(
                        "[ActionForcing] failed to stash streaming chunk "
                        "for video at step=%d: %s",
                        int(self.step) + 1, _exc,
                    )
                    self._pending_video_latents = None

            # Arm the DMD-scorer eval stash: when set to a dict,
            # ``compute_generator_loss_streaming`` (and its inner
            # ``_compute_kl_grad``) populate the scorers' denoised x0
            # estimates and the clean_x conditioning views so the
            # video logger can decode them as side-by-side diagnostic
            # videos. Cleared after harvest. No effect on training
            # math (read-only ``.detach()`` copies into a dict).
            if _sample_due_now:
                self.model._dmd_eval_stash = {}
            else:
                self.model._dmd_eval_stash = None

            gen_loss_dmd, gen_log = self.model.compute_generator_loss_streaming(
                chunk, info,
            )

            # Harvest scorer outputs + clean_x views into a parallel
            # ``self._pending_dmd_eval_latents`` so the video logger
            # can decode them without the metrics path having to skip
            # non-scalar values.
            if _sample_due_now:
                eval_stash = getattr(self.model, "_dmd_eval_stash", None)
                if isinstance(eval_stash, dict) and eval_stash:
                    self._pending_dmd_eval_latents = eval_stash
                self.model._dmd_eval_stash = None
            merged: Dict[str, Any] = {
                "generator_dmd_loss": float(gen_loss_dmd.detach().item()),
            }
            merged.update({
                k: (float(v.detach().float().mean().item())
                    if torch.is_tensor(v) else v)
                for k, v in gen_log.items()
                if not isinstance(v, dict)
            })

            generator_loss = gen_loss_dmd

            # ---- Auxiliary losses (mirror the legacy aux block) ----
            # Streaming chunk's ride window in
            # ``state["ride_*_window"]`` indices: the noisy-half lives
            # at cumulative-sdn positions ``[noisy_start_sdn,
            # noisy_start_sdn + chunk_size)``, which translates to
            # ``ride_*_window[cf + noisy_start_sdn : cf + noisy_start_sdn
            # + chunk_size]``. ``noisy_start_sdn = current_length -
            # new_frames - overlap`` (info-supplied).
            state = self.model.streaming_state
            cf = int(state["cf"])
            chunk_size = int(state["chunk_size"])
            noisy_start_sdn = int(
                state["current_length"]
                - info["new_frames"]
                - info["overlap"]
            )
            chunk_lo = cf + noisy_start_sdn
            chunk_hi = chunk_lo + chunk_size

            if aux_active:
                actions_chunk = state["ride_actions_window"][
                    :, chunk_lo:chunk_hi,
                ]
                actions_for_critic = self._slice_actions_for_critic(
                    actions_chunk
                ).to(chunk.dtype)
                ts_value = info.get("denoised_timestep_from", None)
                ts_int = int(ts_value) if ts_value is not None else 0
                B = chunk.shape[0]
                n_chunks = chunk.shape[1] // int(self.config.num_frame_per_block)
                chunk_t = torch.full(
                    (B, n_chunks), ts_int,
                    device=chunk.device, dtype=torch.long,
                )
                gen_action_loss, critic_logs, _teacher_z = (
                    self._compute_action_critic_losses(
                        pred_x0=chunk,
                        target_action_z=actions_for_critic,
                        chunk_t=chunk_t,
                        current_step=int(self.step),
                    )
                )
                generator_loss = generator_loss + gen_action_loss
                merged.update(critic_logs)

            if gan_active:
                gt_window = state["ride_latents_window"][
                    :, chunk_lo:chunk_hi,
                ]
                gen_gan_loss, gan_logs = self._compute_r3gan_losses(
                    pred_image=chunk,
                    gt_latents_window=gt_window,
                    current_step=int(self.step),
                )
                generator_loss = generator_loss + gen_gan_loss
                merged.update(gan_logs)

            if sc_dmd_active:
                # SC regularizer runs on a fresh chunk-0 KV cache, so
                # the absolute frame position doesn't matter — pass
                # the rollout-only ``ride_latents_window[cf:]`` as
                # ``clean_latent`` (same role as ``latents`` in the
                # legacy path) and the iter's fresh full-window cond
                # dict (from ``info``, NOT from state — stashed cond
                # dicts on state would leak the action-projection
                # graph across iters) with ``seed_frames=cf`` so
                # sc_dmd_loss skips the seed streams when slicing
                # chunk-0 actions.
                sc_loss_raw, sc_logs = self.model.sc_dmd_loss(
                    conditional_dict=info["conditional_dict"],
                    clean_latent=state["ride_latents_window"][:, cf:],
                    seed_frames=cf,
                )
                sc_weight = self._sc_dmd_current_weight(int(self.step))
                weighted_sc = sc_loss_raw * sc_weight
                generator_loss = generator_loss + weighted_sc
                merged.update(sc_logs)
                merged["sc_dmd_weight_effective"] = float(sc_weight)
                merged["sc_dmd_loss_weighted"] = float(
                    weighted_sc.detach().item()
                )

            merged["generator_loss"] = float(generator_loss.detach().item())
            generator_loss.backward()

            # Collapse gate: if the last-chunk MAE > threshold, the
            # student collapsed on this ride — close the sequence so
            # next iter pulls a fresh ride. Current chunk still trained
            # (gradient already accumulated via .backward()).
            mae = float(gen_log.get("baseline_avg_rollout_mae", float("nan")))
            collapsed = (
                self.collapse_mae_threshold is not None
                and mae == mae
                and mae > self.collapse_mae_threshold
            )
            if collapsed:
                merged["streaming_reset_for_collapse"] = 1.0
                self.model.reset_streaming_state()
            return merged
        else:
            chunk, info = self.model.generate_next_chunk(requires_grad=False)
            critic_loss, critic_log = self.model.compute_critic_loss_streaming(
                chunk, info,
            )
            critic_loss.backward()
            merged: Dict[str, Any] = {
                "critic_loss": float(critic_loss.detach().item()),
            }
            merged.update({
                k: (float(v.detach().float().mean().item())
                    if torch.is_tensor(v) else v)
                for k, v in critic_log.items()
                if not isinstance(v, dict)
            })
            mae = float(critic_log.get("baseline_avg_rollout_mae", float("nan")))
            if (
                self.collapse_mae_threshold is not None
                and mae == mae
                and mae > self.collapse_mae_threshold
            ):
                merged["streaming_reset_for_collapse"] = 1.0
                self.model.reset_streaming_state()
            return merged

    # ------------------------------------------------------------------
    # Per-iter forward/backward.
    # ------------------------------------------------------------------
    def _fwdbwd_one_step(
        self,
        train_generator: bool,
        rollout_frames: int,
        max_total_rollout_frames: int,
        cf_dmdctx: int = 0,
    ) -> Optional[Dict[str, Any]]:
        # Streaming mode (LongLive parity) — single rolling sequence
        # spans many iters, advancing by ``new_frames`` ∈ [min_new,
        # chunk_size] frames per iter using a persistent KV cache.
        # Falls back to the legacy single-batch path below when
        # ``streaming_mode=False``.
        if self.streaming_mode:
            return self._fwdbwd_streaming_step(
                train_generator=train_generator,
                rollout_frames=rollout_frames,
                max_total_rollout_frames=max_total_rollout_frames,
                cf_dmdctx=cf_dmdctx,
            )
        # Pull next valid ride; if loader exhausted, restart. Note that
        # ``rollout_frames >= num_training_frames`` (validated in
        # ``_build_pipeline``); for long-rollout mode we slice the ride
        # to ``rollout_frames`` here, build action streams over the
        # full window, and let the pipeline + DMD model gate the
        # gradient to the last ``num_training_frames``.
        #
        # MAE-extension support: if extensions are enabled
        # (``mae_extension_max_extra_chunks > 0``,
        # ``max_total_rollout_frames > rollout_frames``), we slice the
        # ride to ``min(ride_len, max_total_rollout_frames)`` and build
        # the per-frame action conditioning streams over THAT length.
        # The model passes those streams through to the pipeline,
        # which uses ``[0:rollout_frames]`` for the baseline rollout
        # and any of ``[rollout_frames:extended_length]`` slices for
        # extension chunks (each extension consumes ``num_frame_per_
        # block`` frames). The latents themselves cover the same
        # extended window so the pipeline's MAE check has GT to
        # compare against. ``image_or_video_shape`` is held at
        # ``rollout_frames`` (BASELINE) — the noise tensor for
        # extension chunks is sampled inline by the pipeline, not
        # from the trainer.
        #
        # dmd_context window layout (single source of truth, all lengths
        # in latent frames; `cf` = ``self.model.dmd_context_clean_frames``
        # = KV-cache seed prefill, default 9; `shift` = num_frame_per_block
        # = clean/noisy shift, hardcoded to 1 chunk; `anchor` = ``shift`` =
        # leading clean_x_self anchor chunk, ONCE per ride at the FRONT):
        #
        #   ride[s : s+cf]                          → seed_latents (KV prefill at t=0)
        #   ride[s+cf : s+cf+anchor]                → leading anchor chunk (rolled, scored later)
        #   ride[s+cf+anchor : s+cf+anchor+N]       → noisy_x target window
        #   ride[s+cf : s+cf+N]                     → clean_x_GT (= same as
        #                                              the rollout's leading
        #                                              N frames; pure
        #                                              student-rolled in
        #                                              clean_x_self mode,
        #                                              GT in clean_x_GT mode)
        #
        # ``s`` is sampled uniformly from ``[0, ride_len // 2]`` per iter
        # (clamped down so ``s + cf + rollout_window ≤ ride_len``). DDP
        # ranks pick independent s values — DDP grad-averaging handles
        # the resulting cross-rank diversity.
        npb = int(getattr(self.config, "num_frame_per_block", 3))
        shift = npb  # clean/noisy shift; always 1 chunk, never configurable
        anchor = int(
            getattr(self.model, "dmd_clean_x_anchor_frames", npb)
        )  # leading clean_x_self anchor; hardcoded to npb (= shift)

        ride = self._next_ride(rollout_frames + cf_dmdctx)
        if ride is None:
            return None

        ride_len = int(ride["latents"].shape[1])
        # Total rolled length includes the leading anchor chunk
        # (= ``anchor`` = npb frames at the front so batch-1 clean_x_self
        # is purely student-rolled, no seed dip). Once per ride.
        max_rollout_window = max_total_rollout_frames + anchor

        # Per-rank motion-aware s pick. Each rank biases toward its
        # OWN ride's motion. The MAE-extension all_reduce inside the
        # pipeline operates on a SCALAR (averaged MAE) so rank-local
        # absolute frame positions don't break the reduce — what used
        # to need broadcasting was only the rank-0 sampled ``s``,
        # which is meaningless when each rank has a different ride.
        # ``window_len = rollout_frames`` (= num_training_frames):
        # the gradient-bearing scoring window of the iter, which is
        # what we want loaded with motion.
        s_local_max = max(
            0, min(ride_len // 2, ride_len - cf_dmdctx - max_rollout_window)
        )
        s = self._pick_motion_aware_offset(
            ride,
            s_local_max=s_local_max,
            window_lo_offset=cf_dmdctx,
            window_len=int(rollout_frames),
        )

        rollout_end = s + cf_dmdctx + max_rollout_window
        rollout_window = max_rollout_window
        if rollout_window < rollout_frames:
            raise RuntimeError(
                f"Ride too short for rollout: ride_len={ride_len}, "
                f"rollout_frames={rollout_frames}, cf_dmdctx={cf_dmdctx}, "
                f"s={s}. ``_next_ride`` should have skipped this ride."
            )

        prompt_embeds = ride["prompt_embeds"]
        # Rollout-only slices (= ride[s+cf : s+cf+rollout_window]) — used
        # by aux/gan paths whose slicing predates the seed prefill design.
        latents = ride["latents"][:, s + cf_dmdctx : rollout_end]
        actions = ride["z_actions"][:, s + cf_dmdctx : rollout_end]
        # ``clean_latent_full`` covers the seed+rollout window so the
        # pipeline's MAE indexing at absolute ``current_start_frame`` works.
        clean_latent_full = ride["latents"][:, s : rollout_end]

        # Pipeline ``conditional_dict`` covers the FULL window ``ride[s :
        # s+cf+rollout_window]`` so the seed-prefill loop reads frames
        # ``[0, cf)`` and the rollout loop reads frames ``[cf, cf+rollout)``
        # of these streams (pipeline indexes by absolute current_start_frame
        # which starts at 0 for the seed and ``cf`` for the rollout). The
        # model's scoring slicer takes a ``seed_frames=cf`` offset so the
        # scorer still gets the rollout half.
        full_actions = ride["z_actions"][:, s : rollout_end]
        conditional_dict, unconditional_dict = self.model.build_action_conditional(
            prompt_embeds=prompt_embeds,
            gt_actions=full_actions,
        )

        # KV-cache prefill seed: cf frames of GT latent fed at t=0 to
        # populate the KV cache before the rollout starts.
        seed_latents = None
        if cf_dmdctx > 0:
            seed_latents = ride["latents"][:, s : s + cf_dmdctx].contiguous()

        # ``clean_x_GT`` window = the leading N frames of the rollout
        # (= ``ride[s+cf : s+cf+N]``). With the +anchor leading chunk
        # at the FRONT of the rollout, this slice is fully inside the
        # rolled-out range — clean leads noisy by ``shift=npb`` frames
        # WITHIN the rollout (clean=[s+cf : s+cf+N], noisy=[s+cf+anchor :
        # s+cf+anchor+N]) so no seed dip is needed.
        clean_context_latents = None
        clean_conditional_dict = None
        clean_unconditional_dict = None
        if cf_dmdctx > 0:
            num_training_frames = int(getattr(
                self.config, "num_training_frames", 21,
            ))
            clean_start = s + cf_dmdctx
            clean_end = clean_start + num_training_frames
            clean_context_latents = ride["latents"][:, clean_start : clean_end]
            clean_actions = ride["z_actions"][:, clean_start : clean_end]
            clean_conditional_dict, clean_unconditional_dict = (
                self.model.build_action_conditional(
                    prompt_embeds=prompt_embeds,
                    gt_actions=clean_actions,
                )
            )

        # ``image_or_video_shape`` is the TOTAL rolled shape (= scoring
        # window + leading anchor). ``_run_generator`` validates against
        # ``self.rollout_frames + self.dmd_clean_x_anchor_frames`` and
        # slices the pred into noisy_x (last N) and clean_x_self anchor
        # (first N).
        image_or_video_shape = [
            latents.shape[0], rollout_frames + anchor, *latents.shape[2:]
        ]

        if train_generator:
            aux_active = (
                self.action_critic_loss_active
                and self.critic_optimizer is not None
            )
            gan_active = (
                self.gan_enabled
                and self.r3gan_disc is not None
                and self.r3gan_optimizer is not None
            )
            sc_dmd_active = bool(self.sc_dmd_enabled)
            need_aux_artifacts = aux_active or gan_active

            if need_aux_artifacts:
                # Unified path: aux losses (action critic z-guidance)
                # and/or R3GAN both consume ``aux["pred_image"]`` so we
                # only need ONE generator forward. Each downstream
                # branch is gated separately on its own ``*_active``
                # flag, and adds its term to the running generator
                # loss before the single backward at the bottom.
                # ``z_actions_for_scoring`` = per-frame z_actions
                # (already sliced to ``action_dims`` at dataset load
                # time) for the noisy_x scoring window. Used by the
                # action-mode freeze block to skip a CoTracker forward
                # on GT — the dataset's offline pipeline already
                # encoded the same motion through the same ss_vae.
                z_actions_scoring = actions[:, -int(self.config.num_training_frames):].contiguous()
                gen_loss_dmd, gen_log_dict, aux = self.model.generator_loss(
                    image_or_video_shape=image_or_video_shape,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict,
                    clean_latent=clean_latent_full,
                    initial_latent=None,
                    return_aux=True,
                    seed_latents=seed_latents,
                    clean_x_GT=clean_context_latents,
                    clean_conditional_dict=clean_conditional_dict,
                    clean_unconditional_dict=clean_unconditional_dict,
                    z_actions_for_scoring=z_actions_scoring,
                )
                pred_image = aux["pred_image"]
                scoring_frames = int(aux["scoring_frames"])
                gen_window_end = int(aux["rollout_frames"])
                gen_window_start = gen_window_end - scoring_frames

                if self._video_sample_due(int(self.step) + 1):
                    try:
                        self._pending_video_latents = (
                            pred_image.detach().to(torch.float32)
                        )
                        _cs = int(self.step) + 1
                        while (
                            self._sample_at_steps_pending
                            and self._sample_at_steps_pending[0] <= _cs
                        ):
                            self._sample_at_steps_pending.pop(0)
                        # Consume the sticky deferral bit (see
                        # streaming-path twin for rationale).
                        self._video_sample_due_bit = False
                    except Exception as _exc:
                        logging.warning(
                            "[ActionForcing] failed to stash "
                            "pred_image for video at step=%d: %s",
                            int(self.step) + 1, _exc,
                        )
                        self._pending_video_latents = None

                merged: Dict[str, Any] = {
                    "generator_dmd_loss": float(gen_loss_dmd.detach().item()),
                }
                merged.update({
                    k: (float(v.detach().float().mean().item())
                        if torch.is_tensor(v) else v)
                    for k, v in gen_log_dict.items()
                    if not isinstance(v, dict)
                })

                generator_loss = gen_loss_dmd

                if aux_active:
                    actions_for_critic = self._slice_actions_for_critic(
                        actions[:, gen_window_start:gen_window_end]
                    ).to(pred_image.dtype)
                    ts_value = aux.get("denoised_timestep_from", None)
                    ts_int = int(ts_value) if ts_value is not None else 0
                    B = pred_image.shape[0]
                    n_chunks = pred_image.shape[1] // int(self.config.num_frame_per_block)
                    chunk_t = torch.full(
                        (B, n_chunks), ts_int,
                        device=pred_image.device, dtype=torch.long,
                    )
                    gen_action_loss, critic_logs, _teacher_z = (
                        self._compute_action_critic_losses(
                            pred_x0=pred_image,
                            target_action_z=actions_for_critic,
                            chunk_t=chunk_t,
                            current_step=int(self.step),
                        )
                    )
                    generator_loss = generator_loss + gen_action_loss
                    merged.update(critic_logs)

                if gan_active:
                    gt_window = latents[:, gen_window_start:gen_window_end]
                    gen_gan_loss, gan_logs = self._compute_r3gan_losses(
                        pred_image=pred_image,
                        gt_latents_window=gt_window,
                        current_step=int(self.step),
                    )
                    generator_loss = generator_loss + gen_gan_loss
                    merged.update(gan_logs)

                if sc_dmd_active:
                    sc_loss_raw, sc_logs = self.model.sc_dmd_loss(
                        conditional_dict=conditional_dict,
                        clean_latent=latents,
                        seed_frames=cf_dmdctx,
                    )
                    sc_weight = self._sc_dmd_current_weight(int(self.step))
                    weighted_sc = sc_loss_raw * sc_weight
                    generator_loss = generator_loss + weighted_sc
                    merged.update(sc_logs)
                    merged["sc_dmd_weight_effective"] = float(sc_weight)
                    merged["sc_dmd_loss_weighted"] = float(
                        weighted_sc.detach().item()
                    )

                merged["generator_loss"] = float(generator_loss.detach().item())
                generator_loss.backward()
                return merged
            else:
                z_actions_scoring = actions[:, -int(self.config.num_training_frames):].contiguous()
                generator_loss, gen_log_dict = self.model.generator_loss(
                    image_or_video_shape=image_or_video_shape,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict,
                    clean_latent=clean_latent_full,
                    initial_latent=None,
                    seed_latents=seed_latents,
                    clean_x_GT=clean_context_latents,
                    clean_conditional_dict=clean_conditional_dict,
                    clean_unconditional_dict=clean_unconditional_dict,
                    z_actions_for_scoring=z_actions_scoring,
                )
                merged_plain: Dict[str, Any] = {
                    "generator_dmd_loss": float(generator_loss.detach().item()),
                    **{
                        k: (float(v.detach().float().mean().item())
                            if torch.is_tensor(v) else v)
                        for k, v in gen_log_dict.items()
                        if not isinstance(v, dict)
                    },
                }

                if sc_dmd_active:
                    sc_loss_raw, sc_logs = self.model.sc_dmd_loss(
                        conditional_dict=conditional_dict,
                        clean_latent=latents,
                        seed_frames=cf_dmdctx,
                    )
                    sc_weight = self._sc_dmd_current_weight(int(self.step))
                    weighted_sc = sc_loss_raw * sc_weight
                    generator_loss = generator_loss + weighted_sc
                    merged_plain.update(sc_logs)
                    merged_plain["sc_dmd_weight_effective"] = float(sc_weight)
                    merged_plain["sc_dmd_loss_weighted"] = float(
                        weighted_sc.detach().item()
                    )

                merged_plain["generator_loss"] = float(
                    generator_loss.detach().item()
                )
                generator_loss.backward()
                return merged_plain
        else:
            critic_loss, critic_log_dict = self.model.critic_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent_full,
                initial_latent=None,
                seed_latents=seed_latents,
                clean_x_GT=clean_context_latents,
                clean_conditional_dict=clean_conditional_dict,
            )
            critic_loss.backward()
            return {
                "critic_loss": float(critic_loss.detach().item()),
                **{
                    k: (float(v.detach().float().mean().item())
                        if torch.is_tensor(v) else v)
                    for k, v in critic_log_dict.items()
                    if not isinstance(v, dict)
                },
            }

    # ------------------------------------------------------------------
    # Ride iteration helpers.
    # ------------------------------------------------------------------
    def _fresh_ride_iter(self, epoch: int):
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)
        return iter(self.dataloader)

    def _next_ride(
        self,
        rollout_frames: int,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Pull a ride from ``self._ride_iter``, skipping rides that don't
        have enough latent frames to cover the full rollout. Auto-
        replenishes the iterator (and bumps the epoch) when exhausted
        so the curriculum order resumes from the start of the sorted
        ride list on the next pass.

        The threshold is ``rollout_frames`` (= num_training_frames in
        classic mode, larger in CF-style long-rollout mode). The
        dataset's own ``min_ride_frames`` filter should ideally
        already exclude too-short rides at index time; this is a
        belt-and-suspenders check in case the YAML's
        ``min_ride_frames`` is out of sync with ``rollout_frames``.
        """
        # Ride-level motion filter. The frodobots dataset is bimodal:
        # ~25% of rides are parked the entire time, and the per-ride
        # mean motion magnitude has median ~3.3 across the 857
        # available rides. Filtering by per-ride MEAN at the median
        # keeps only the upper-half — the consistently-moving rides
        # that actually carry the action-conditioning signal. Skip
        # happens BEFORE the expensive ``_load_ride_tensors`` call
        # (which runs an ss_vae forward on every frame to encode
        # z_actions); the pre-check is ~3 ms (npy load + mean).
        # ``motion_ride_min_mean`` is the threshold on per-ride mean
        # magnitude; default 3.0 keeps roughly the upper half.
        skip_dead = bool(getattr(self.config, "motion_skip_dead_rides", True))
        ride_min_mean = float(
            getattr(self.config, "motion_ride_min_mean", 3.0)
        )
        attempts = 0
        max_attempts = 200
        skipped_dead = 0
        while attempts < max_attempts:
            try:
                batch = next(self._ride_iter)
            except StopIteration:
                self._epoch += 1
                self._ride_iter = self._fresh_ride_iter(self._epoch)
                continue
            if not batch:
                attempts += 1
                continue
            meta = batch[0]
            n_latent_frames = int(meta.get("n_latent_frames", 0))
            if n_latent_frames < rollout_frames:
                attempts += 1
                continue
            # Pre-check motion BEFORE the expensive ride load. Best-
            # effort: if the dataset doesn't expose load_motion_
            # magnitudes (older dataset class) we fall through and
            # accept the ride.
            if skip_dead:
                loader = getattr(
                    self.dataset, "load_motion_magnitudes", None,
                )
                if loader is not None:
                    try:
                        mag = loader(meta["zarr_path"], n_latent_frames)
                        if float(mag.mean()) < ride_min_mean:
                            skipped_dead += 1
                            attempts += 1
                            continue
                    except Exception as e:
                        logging.warning(
                            "_next_ride motion pre-check failed for %s: %s — "
                            "accepting ride without filter",
                            meta.get("zarr_path", "?"), e,
                        )
            # Cap the loaded ride at ``cfg.max_ride_frames`` (saved on
            # ``self.max_ride_frames`` by the parent trainer). This
            # bounds memory/compute when the dataset has very long
            # rides — extensions only need ``max_total_rollout_frames``
            # frames anyway, so loading more is wasteful. The trainer's
            # downstream ``[:extended_length]`` slice further trims to
            # exactly the per-iter window.
            ride = _load_ride_tensors(
                self.dataset, meta, self.device, self.dtype,
                action_dims=self.action_dims,
                max_frames=getattr(self, "max_ride_frames", None),
            )
            if ride is None or ride["latents"].shape[1] < rollout_frames:
                attempts += 1
                continue
            if skipped_dead > 0 and self.is_main_process:
                logging.debug(
                    "[ActionForcing] _next_ride: skipped %d low-motion "
                    "ride(s) before finding one with mean motion >= %.3f.",
                    skipped_dead, ride_min_mean,
                )
            return ride
        if self.is_main_process:
            logging.warning(
                "[ActionForcing] _next_ride: gave up after %d attempts "
                "(skipped %d rides with mean motion < %.3f); the dataset "
                "may have no rides with >=%d latent frames AND non-trivial "
                "motion. Consider lowering rollout_frames, raising the "
                "dataset's min_ride_frames filter, lowering "
                "motion_ride_min_mean, or setting motion_skip_dead_rides=false.",
                max_attempts, skipped_dead, ride_min_mean, rollout_frames,
            )
        return None

    # ------------------------------------------------------------------
    # Override grad-uniformity (parent expects multi-slot path).
    # The parent's ``_all_reduce_extra_trainable_grads`` uses dist
    # reductions on action_projection / action_token_projection grads;
    # that part still applies here. We reuse it as-is.
    # ------------------------------------------------------------------


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Phase-1 Action-Forcing DMD trainer"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--override", type=str, nargs="*", default=[])
    args = parser.parse_args()

    if args.override:
        base = OmegaConf.load(args.config)
        override = OmegaConf.from_dotlist(list(args.override))
        cfg = OmegaConf.merge(base, override)
    else:
        cfg = OmegaConf.load(args.config)

    trainer = ActionForcingDMDTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
