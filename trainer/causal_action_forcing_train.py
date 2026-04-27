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
from typing import Any, Dict, List, Optional

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

        while self.step < max_steps:
            # CF-parity #5: put the whole DMD module in eval mode at the
            # start of each iter to suppress any latent randomness
            # (Dropout / BN running stats). Our DiT has no Dropout/BN
            # so this is a no-op in math, but we match CF's
            # ``self.model.eval()`` at ``Causal-Forcing/trainer/
            # distillation.py:230`` to keep the code path identical.
            self.model.eval()
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

            # Logging.
            if self.is_main_process and (self.step % log_interval == 0):
                msg_parts = [f"step={self.step}/{max_steps}"]
                if train_generator and generator_log_dict is not None:
                    msg_parts.append(
                        f"gen_loss={generator_log_dict.get('generator_loss', 0.0):.4f}"
                    )
                    if "dmdtrain_gradient_norm" in generator_log_dict:
                        msg_parts.append(
                            f"dmd_grad={generator_log_dict['dmdtrain_gradient_norm']:.4f}"
                        )
                if critic_log_dict is not None:
                    msg_parts.append(
                        f"critic_loss={critic_log_dict.get('critic_loss', 0.0):.4f}"
                    )
                msg_parts.append(
                    f"gen_grad_norm={float(gen_grad_norm.item()) if torch.is_tensor(gen_grad_norm) else float(gen_grad_norm):.4f}"
                )
                msg_parts.append(f"fake_grad_norm={fake_grad_norm_val:.4f}")
                logging.info("[ActionForcing] " + " ".join(msg_parts))
                if (
                    _HAS_WANDB
                    and getattr(self, "wandb_enabled", False)
                ):
                    log_payload: Dict[str, Any] = {}
                    if train_generator and generator_log_dict is not None:
                        for k, v in generator_log_dict.items():
                            if torch.is_tensor(v):
                                v = float(v.item())
                            log_payload[f"gen/{k}"] = v
                    if critic_log_dict is not None:
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

            if (
                self.is_main_process
                and self._pending_video_latents is not None
            ):
                self._log_pred_image_video(
                    self._pending_video_latents, int(self.step),
                )
                self._pending_video_latents = None

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

        Two gates can fire it:
          (1) explicit ``sample_at_steps`` membership, or
          (2) ``sample_interval > 0`` and step is a positive multiple
              thereof.
        Both also require main rank and at least one viable sink
        (wandb enabled OR ``vis_save_local`` true). ``current_step <= 0``
        is skipped because no rollout has happened yet at iter 0 entry
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
        if self.sample_interval <= 0:
            return False
        return (current_step % self.sample_interval) == 0

    @torch.no_grad()
    def _log_pred_image_video(
        self,
        pred_image: torch.Tensor,
        step: int,
    ) -> None:
        """Decode ``pred_image`` (a student rollout in latent space) to
        pixel mp4 bytes and upload to wandb under ``sample/pred_image``.

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
                out_path = samples_dir / f"step_{int(step):07d}.mp4"
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
            wandb.log(
                {
                    "sample/pred_image": wandb.Video(
                        tmp_path,
                        fps=int(self.sample_fps),
                        format="mp4",
                        caption=f"step {step}",
                    ),
                    "sample/step": int(step),
                    "sample/num_frames": int(vid_np.shape[0]),
                },
                step=step,
            )
            logging.info(
                "[ActionForcing] Logged sample video at step=%d "
                "(frames=%d, fps=%d)",
                step, vid_np.shape[0], int(self.sample_fps),
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

        # Random s ∈ [0, min(ride_len/2, ride_len - cf - cap)] with
        # cross-rank MIN-reduce + rank-0 broadcast (mirrors the legacy
        # single-batch path).
        s_local_max = max(
            0, min(ride_len // 2, ride_len - cf_dmdctx - cap),
        )
        if dist.is_initialized() and dist.get_world_size() > 1:
            t = torch.tensor([s_local_max], device=self.device, dtype=torch.long)
            dist.all_reduce(t, op=dist.ReduceOp.MIN)
            s_global_max = int(t.item())
            if dist.get_rank() == 0:
                t.fill_(random.randint(0, s_global_max) if s_global_max > 0 else 0)
            dist.broadcast(t, src=0)
            s = int(t.item())
        else:
            s = random.randint(0, s_local_max) if s_local_max > 0 else 0

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
        """
        # Open a sequence if needed.
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
            if self._video_sample_due(int(self.step) + 1):
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
                except Exception as _exc:
                    logging.warning(
                        "[ActionForcing] failed to stash streaming chunk "
                        "for video at step=%d: %s",
                        int(self.step) + 1, _exc,
                    )
                    self._pending_video_latents = None

            gen_loss_dmd, gen_log = self.model.compute_generator_loss_streaming(
                chunk, info,
            )
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
            if (
                self.collapse_mae_threshold is not None
                and mae == mae
                and mae > self.collapse_mae_threshold
            ):
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

        # Random offset s ∈ [0, min(ride_len // 2, ride_len - cf - max_rollout_window)].
        # Each rank loads a different ride with a different ride_len,
        # so the upper bound on s is rank-local. To keep the
        # extension-loop all_reduces (MAE / synced_available) describing
        # the same relative offset on every rank, we MIN-reduce the
        # local bound, sample s on rank 0 within that global bound, and
        # broadcast it. (Single-rank code paths skip the all_reduce/
        # broadcast and just sample locally.)
        s_local_max = max(
            0, min(ride_len // 2, ride_len - cf_dmdctx - max_rollout_window)
        )
        if dist.is_initialized() and dist.get_world_size() > 1:
            s_t = torch.tensor(
                [s_local_max], device=self.device, dtype=torch.long,
            )
            dist.all_reduce(s_t, op=dist.ReduceOp.MIN)
            s_global_max = int(s_t.item())
            if dist.get_rank() == 0:
                s_t.fill_(
                    random.randint(0, s_global_max) if s_global_max > 0 else 0
                )
            dist.broadcast(s_t, src=0)
            s = int(s_t.item())
        else:
            s = random.randint(0, s_local_max) if s_local_max > 0 else 0

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
        attempts = 0
        max_attempts = 200
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
            return ride
        if self.is_main_process:
            logging.warning(
                "[ActionForcing] _next_ride: gave up after %d attempts; the dataset "
                "may have no rides with >=%d latent frames. Consider "
                "lowering rollout_frames or raising the dataset's "
                "min_ride_frames filter.",
                max_attempts, rollout_frames,
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
