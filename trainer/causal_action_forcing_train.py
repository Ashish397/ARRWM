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

import atexit
import gc
import logging
import math
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

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from trainer.causal_rolling_staircase_train import (
    RollingStaircaseDMDTrainer,
    _finalize_ride_to_gpu,
    _load_ride_tensors,
    _load_ride_tensors_cpu_part,
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
        # Gradient checkpointing — recompute transformer-block
        # activations during backward to free forward-time memory.
        # WAN ships with native checkpoint support
        # (``wan/modules/model.py:767-772``); each block routes through
        # ``torch.utils.checkpoint.checkpoint`` when
        # ``self.gradient_checkpointing`` is True. Saves ~80% of the
        # forward-activation memory on each grad-active gen/real/fake
        # forward at the cost of ~33% extra compute (one extra forward
        # per block during backward).
        # Defaults: gen ON when distilled-critic mode is on (the
        # dual_grad path stacks per-block last-rung grad-active
        # forwards on top of the rolling rollout, OOMing without
        # checkpointing); real_score / fake_score OFF by default
        # (their forwards are detached at the gen-grad path).
        # Configurable per knob below.
        gen_grad_ckpt = bool(
            getattr(args, "gen_gradient_checkpointing",
                    bool(getattr(args, "gan_sam2_distilled_critic", False)))
        )
        rs_grad_ckpt = bool(
            getattr(args, "real_score_gradient_checkpointing", False)
        )
        fs_grad_ckpt = bool(
            getattr(args, "fake_score_gradient_checkpointing", False)
        )
        if gen_grad_ckpt:
            model.generator.model.gradient_checkpointing = True
        if rs_grad_ckpt:
            model.real_score.model.gradient_checkpointing = True
        if fs_grad_ckpt:
            model.fake_score.model.gradient_checkpointing = True
        if self.is_main_process:
            logging.info(
                "[ActionForcing] gradient_checkpointing: gen=%s "
                "real_score=%s fake_score=%s",
                gen_grad_ckpt, rs_grad_ckpt, fs_grad_ckpt,
            )
        if model.action_projection is not None:
            model.action_projection.to(device=self.device, dtype=self.dtype)
        if model.action_token_projection is not None:
            model.action_token_projection.to(device=self.device, dtype=self.dtype)
        if getattr(model, "state_probe", None) is not None:
            model.state_probe.to(device=self.device, dtype=self.dtype)
        if getattr(model, "action_critic", None) is not None:
            model.action_critic.to(device=self.device, dtype=self.dtype)
        # noise_aux: small fp32 ConvNet (~8M params). fp32 keeps the
        # MSE residual stable across the rectified-flow noise levels.
        if getattr(model, "noise_aux", None) is not None:
            model.noise_aux.to(device=self.device, dtype=torch.float32)
        if getattr(model, "vae", None) is not None:
            try:
                model.vae.to(device=self.device)
            except AttributeError:
                inner_vae = getattr(model.vae, "model", None)
                if inner_vae is not None:
                    inner_vae.to(device=self.device)
        self.model = model

        # ------------------------------------------------------------------
        # torch.compile (optional, config-gated). Wraps the inner DiT
        # modules BEFORE DDP wrap so the compiled graph is what DDP all-
        # reduces over. Real_score is frozen but still benefits from
        # compile on its inference forward (3 forwards per gen step).
        #
        # Knobs:
        #   ``compile_dit`` (bool, default False): turn it on.
        #   ``compile_dit_mode`` (str, default "default"): forwarded to
        #       ``torch.compile(mode=...)``. ``"default"`` is safe with
        #       in-place KV cache mutation; ``"reduce-overhead"`` enables
        #       CUDA graphs which conflict with our cache writes — avoid.
        #   ``compile_dit_dynamic`` (bool|None, default None = auto):
        #       ``True`` = single shape-polymorphic graph (faster compile,
        #       slightly slower runtime); ``False`` = recompile per shape
        #       (slower compile, fastest runtime). ``None`` lets dynamo
        #       decide. Our forward shapes vary (rolling 3-frame chunks
        #       vs scoring 21-frame chunks vs local_attn_size_schedule
        #       transitions), so the dynamo cache holds multiple graphs.
        #   ``compile_dynamo_cache_size`` (int, default 64): bump
        #       dynamo's cache_size_limit so shape-polymorphic recompiles
        #       don't fall back to eager.
        #
        # First-step wallclock spikes during compilation (~minutes for
        # a 1.3B DiT). Steady-state speedup is typically 1.5-3x on the
        # forward path. If compile fails (dynamic-shape edge case, DDP
        # interaction), we log + fall back to eager — training continues.
        # ------------------------------------------------------------------
        compile_dit = bool(getattr(self.config, "compile_dit", False))
        if compile_dit:
            compile_mode = str(getattr(self.config, "compile_dit_mode", "default"))
            compile_dynamic = getattr(self.config, "compile_dit_dynamic", None)
            cache_size_limit = int(getattr(self.config, "compile_dynamo_cache_size", 64))
            try:
                import torch._dynamo as _dynamo
                _dynamo.config.cache_size_limit = max(
                    int(_dynamo.config.cache_size_limit), cache_size_limit,
                )
            except Exception as _exc:
                logging.warning(
                    "[ActionForcing] torch._dynamo cache_size_limit bump failed: %s",
                    _exc,
                )
            if self.is_main_process:
                logging.info(
                    "[ActionForcing] torch.compile DiT modules (mode=%s, "
                    "dynamic=%s, dynamo_cache=%d). First-step wallclock will "
                    "spike during compilation; steady-state should be ~1.5-3x "
                    "faster on forward.",
                    compile_mode, compile_dynamic, cache_size_limit,
                )
            try:
                model.generator.model = torch.compile(
                    model.generator.model,
                    mode=compile_mode, dynamic=compile_dynamic,
                )
                model.fake_score.model = torch.compile(
                    model.fake_score.model,
                    mode=compile_mode, dynamic=compile_dynamic,
                )
                model.real_score.model = torch.compile(
                    model.real_score.model,
                    mode=compile_mode, dynamic=compile_dynamic,
                )
            except Exception as _exc:
                logging.warning(
                    "[ActionForcing] torch.compile failed (%s) — falling "
                    "back to eager.", _exc,
                )

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

            # Online real_teacher: peft adapter is alive on
            # model.real_score.model. Wrap with DDP using
            # find_unused_parameters=True (peft only touches LoRA-target
            # modules — non-LoRA layers see no gradient per step).
            self.real_score_ddp: Optional[DDP] = None
            if bool(getattr(self.config, "real_teacher_train_online", False)):
                self.real_score_ddp = DDP(
                    model.real_score.model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=True,
                    broadcast_buffers=False,
                )
                model.real_score.model = self.real_score_ddp  # type: ignore
        else:
            self.real_score_ddp = None

        # ------------------------------------------------------------------
        # noise_aux DDP wrap. NoiseAuxLite is a small all-rank-trained
        # ConvNet — wrap with find_unused_parameters=False since every
        # param sees a gradient on every iter the noise_aux loss fires.
        # ------------------------------------------------------------------
        self.noise_aux_ddp: Optional[DDP] = None
        if (
            self.world_size > 1
            and getattr(model, "noise_aux", None) is not None
        ):
            self.noise_aux_ddp = DDP(
                model.noise_aux,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
                broadcast_buffers=False,
            )
            # Keep ``model.noise_aux`` pointing at the DDP wrapper so
            # loss and optim see the same handle. The underlying
            # NoiseAuxLite is reachable via ``.module``.
            model.noise_aux = self.noise_aux_ddp  # type: ignore

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
        # State-probe (LoRA-side z-supervision). Mirror of the action_
        # critic DDP-wrap+optim plumbing, fired only when the LoRA-side
        # state_probe loss is requested (= state_probe_aux_enabled AND
        # action_teacher_mode != off AND state_probe_z_loss_weight > 0).
        # The probe was instantiated frozen by ``_build_action_aux_heads_
        # compat`` for ckpt-load compat; flip to trainable here when the
        # loss is wired in. Heads share weights between gen-side
        # (attached to generator wrapper, currently no loss) and
        # LoRA-side (attached to real_score wrapper by
        # ``_attach_state_probe_to_real_score``). Single optimizer.
        # ------------------------------------------------------------------
        self.state_probe_ddp: Optional[DDP] = None
        self.state_probe_optimizer: Optional[torch.optim.Optimizer] = None
        self.state_probe_z_loss_weight = float(
            getattr(self.config, "state_probe_z_loss_weight", 0.0)
        )
        self.state_probe_aux_active = (
            self.action_critic_loss_active
            and bool(getattr(self.config, "state_probe_aux_enabled", False))
            and self.state_probe_z_loss_weight > 0.0
            and getattr(model, "state_probe", None) is not None
        )
        self.state_probe_max_grad_norm = float(
            getattr(self.config, "state_probe_max_grad_norm", 1.0)
        )
        if self.state_probe_aux_active:
            sp = model.state_probe
            sp.requires_grad_(True)
            sp.train()
            if self.world_size > 1:
                self.state_probe_ddp = DDP(
                    sp,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=False,
                    broadcast_buffers=False,
                )
                model.state_probe = self.state_probe_ddp  # type: ignore
                # Keep the shared wrapper-side references in sync so the
                # real_score wrapper's forward calls the DDP-wrapped
                # module (matched all_reduce across ranks). Also point
                # generator._state_probe at the DDP wrap so any gen-side
                # forward that does fire the probe (rare under streaming
                # geometry) routes through DDP rather than the raw
                # module, avoiding a desync.
                if hasattr(model, "real_score"):
                    setattr(model.real_score, "_state_probe", model.state_probe)
                if hasattr(model, "generator"):
                    setattr(model.generator, "_state_probe", model.state_probe)
                # The wrapper's probe-firing gate reads
                # ``self._state_probe.num_frame_per_block`` directly
                # (utils/wan_wrapper.py:624) without unwrapping DDP.
                # DDP doesn't proxy arbitrary attrs, so the lookup
                # returns 0 and the gate fails — state_preds never
                # gets returned, the LoRA-side state_probe loss never
                # fires, no params get gradient. Mirror the attribute
                # onto the DDP wrap explicitly. Same goes for the
                # other init-time attrs the wrapper reads from the
                # probe in older code paths (defensive).
                self.state_probe_ddp.num_frame_per_block = (
                    sp.num_frame_per_block
                )
            sp_lr = float(getattr(self.config, "state_probe_lr", 1.0e-04))
            sp_betas = tuple(
                getattr(self.config, "state_probe_betas", [0.9, 0.999])
            )
            sp_eps = float(getattr(self.config, "state_probe_eps", 1.0e-08))
            sp_wd = float(
                getattr(self.config, "state_probe_weight_decay", 0.0)
            )
            sp_params = [p for p in sp.parameters() if p.requires_grad]
            self.state_probe_optimizer = torch.optim.AdamW(
                sp_params,
                lr=sp_lr, betas=sp_betas, eps=sp_eps, weight_decay=sp_wd,
            )
            if self.is_main_process:
                logging.info(
                    "[ActionForcing] state_probe optimizer built: "
                    "AdamW lr=%.2e wd=%.4f weight=%.3f params=%.2fM",
                    sp_lr, sp_wd, self.state_probe_z_loss_weight,
                    sum(p.numel() for p in sp_params) / 1e6,
                )

        # LoRA-side action_critic z-guidance weight (mirror of the
        # gen-side ``generator_action_z_guidance_weight`` knob, applied
        # to the LoRA's denoised x0 instead of the student's pred). Set
        # to 0.0 to disable. Reads from ``self.config`` so it can be
        # overridden per-sbatch.
        self.lora_action_critic_z_guidance_weight = float(
            getattr(
                self.config,
                "lora_action_critic_z_guidance_weight",
                0.0,
            )
        )

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
        # Distilled-critic mode wraps ``disc.heads_module`` instead of
        # the full disc to give DDP a clean, fully-trainable submodule
        # (the frozen SAM2 encoder is left un-wrapped). Only set when
        # ``gan_sam2_distilled_critic=True`` and world_size > 1.
        self.r3gan_heads_ddp: Optional[DDP] = None
        # ``gan_backbone`` selects the discriminator architecture:
        #   * "latent_3d_conv" (default, legacy) — 3D ConvNet on the
        #     student's latent video. Cheap, no VAE decode required.
        #   * "sam2_pixel" — Flash-DMD-style frozen SAM2 image encoder
        #     on VAE-decoded pixel video, with multiple trainable
        #     heads on the Hiera FPN. Paper-aligned, video-aware via
        #     SAM2's segmentation pretraining; catches structural
        #     failure modes (road edges, vehicle outlines, lane
        #     markings) the latent disc misses.
        self.gan_backbone = str(
            getattr(self.config, "gan_backbone", "latent_3d_conv")
        )
        if self.gan_backbone not in ("latent_3d_conv", "sam2_pixel"):
            raise ValueError(
                f"gan_backbone must be 'latent_3d_conv' or 'sam2_pixel'; "
                f"got {self.gan_backbone!r}."
            )
        if self.gan_enabled:
            if self.gan_backbone == "latent_3d_conv":
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
            elif self.gan_backbone == "sam2_pixel":
                from model.r3gan_sam2 import R3GANDiscriminatorSAM2Pixel
                sam2_ckpt = getattr(
                    self.config, "gan_sam2_checkpoint_path", None,
                )
                sam2_cfg = getattr(
                    self.config, "gan_sam2_config_path", None,
                )
                if sam2_ckpt is None or sam2_cfg is None:
                    raise ValueError(
                        "gan_backbone=sam2_pixel requires "
                        "gan_sam2_checkpoint_path and gan_sam2_config_path "
                        "to be set in the config."
                    )
                resolution = int(
                    getattr(self.config, "gan_sam2_resolution", 512)
                )
                # All-fp32 SAM2 path: R1/R2 second-order penalties are
                # numerically unstable under bf16 (the original latent
                # disc was explicitly all-fp32 for the same reason).
                # Memory cost is +~80 MB params + ~2× activation vs
                # bf16 — comfortably within budget on Hiera-B+.
                preserve_aspect = bool(
                    getattr(self.config, "gan_sam2_preserve_aspect", True)
                )
                pad_to_square = bool(
                    getattr(self.config, "gan_sam2_pad_to_square", False)
                )
                frame_pool = str(
                    getattr(self.config, "gan_sam2_frame_pool", "mean")
                )
                frame_pool_topk = int(
                    getattr(self.config, "gan_sam2_frame_pool_topk", 4)
                )
                # Per-frame chunking of the SAM2 encoder forward. With
                # 84 pixel frames per gen step at 512×512 input, the
                # encoder's transient activations were the largest
                # contributor to the GAN distilled-critic peak segment.
                # Splitting into ``encoder_chunk_size``-sized chunks
                # along the batched-frame axis cuts that transient
                # ~84/chunk×; semantically transparent (SAM2 encoder
                # is per-frame only). Default 0 = no chunking.
                encoder_chunk_size = int(
                    getattr(self.config, "gan_sam2_encoder_chunk_size", 0)
                )
                disc = R3GANDiscriminatorSAM2Pixel(
                    sam2_checkpoint_path=str(sam2_ckpt),
                    sam2_config_path=str(sam2_cfg),
                    image_resolution=resolution,
                    device=self.device,
                    dtype=torch.float32,
                    preserve_aspect=preserve_aspect,
                    pad_to_square=pad_to_square,
                    frame_pool=frame_pool,
                    frame_pool_topk=frame_pool_topk,
                    encoder_chunk_size=encoder_chunk_size,
                )
                disc.train()  # heads → train; encoder pinned to eval
                              # via overridden train() in the class
                self.r3gan_disc = disc
                # Distilled-critic mode determines DDP-wrap topology:
                #   * Legacy mode (False): wrap the FULL disc with
                #     find_unused_parameters=True (frozen SAM2 encoder
                #     has no grad → "unused").
                #   * Distilled mode (True): the distilled D-update
                #     calls heads-only forward; if we wrapped the full
                #     disc, the heads forward would bypass the wrapper's
                #     forward-tracking and DDP all-reduce would NOT
                #     fire on backward (silent rank divergence). Wrap
                #     ONLY ``heads_module`` instead. The frozen encoder
                #     stays un-wrapped; no DDP needed (no trainable
                #     params).
                self.gan_sam2_distilled_critic = bool(
                    getattr(self.config, "gan_sam2_distilled_critic", False)
                )
                self.r3gan_heads_ddp = None
                if self.world_size > 1:
                    if self.gan_sam2_distilled_critic:
                        # Heads-only DDP wrap. find_unused_parameters
                        # can be False here because every head is
                        # always exercised in the distilled D-update.
                        self.r3gan_heads_ddp = DDP(
                            disc.heads_module,
                            device_ids=[self.local_rank],
                            output_device=self.local_rank,
                            find_unused_parameters=False,
                            broadcast_buffers=False,
                        )
                    else:
                        # Legacy: full-disc DDP wrap (frozen encoder
                        # forces find_unused_parameters=True).
                        self.r3gan_disc_ddp = DDP(
                            disc,
                            device_ids=[self.local_rank],
                            output_device=self.local_rank,
                            find_unused_parameters=True,
                            broadcast_buffers=False,
                        )
                if self.is_main_process:
                    n_total = sum(p.numel() for p in disc.parameters())
                    n_train = sum(
                        p.numel() for p in disc.parameters()
                        if p.requires_grad
                    )
                    ddp_kind = (
                        "heads-only"
                        if self.r3gan_heads_ddp is not None
                        else (
                            "full-disc"
                            if self.r3gan_disc_ddp is not None
                            else "no"
                        )
                    )
                    logging.info(
                        "[ActionForcing] R3GAN-SAM2 discriminator built (ADM 2D heads): "
                        "ckpt=%s cfg=%s resolution=%d "
                        "params_total=%.2fM params_trainable=%.2fM (DDP=%s)",
                        sam2_ckpt, sam2_cfg, resolution,
                        n_total / 1e6, n_train / 1e6, ddp_kind,
                    )

                # SAM2-distilled latent critic (Sobolev-style value+gradient
                # distillation). When enabled, the pixel-space disc is no
                # longer in the gen's autograd graph — instead this small
                # latent-space critic is trained to match BOTH the disc's
                # logit values AND its gradient field w.r.t. the input
                # latent, and the gen gets its GAN gradient from the critic.
                # See ``model/latent_sam2_critic.py`` for the architecture.
                self.latent_critic = None
                self.latent_critic_ddp = None
                self.latent_critic_optimizer = None
                if self.gan_sam2_distilled_critic:
                    from model.latent_sam2_critic import LatentSAM2Critic
                    critic_hidden = int(
                        getattr(self.config, "gan_critic_hidden", 512)
                    )
                    critic_num_blocks = int(
                        getattr(self.config, "gan_critic_num_blocks", 4)
                    )
                    critic_in_channels = 16  # WAN VAE latent channel count
                    self.latent_critic = LatentSAM2Critic(
                        in_channels=critic_in_channels,
                        d_model=critic_hidden,
                        num_blocks=critic_num_blocks,
                        frame_pool=frame_pool,
                        frame_pool_topk=frame_pool_topk,
                    ).to(device=self.device, dtype=torch.float32)
                    self.latent_critic.train()
                    if self.world_size > 1:
                        self.latent_critic_ddp = DDP(
                            self.latent_critic,
                            device_ids=[self.local_rank],
                            output_device=self.local_rank,
                            find_unused_parameters=False,
                            broadcast_buffers=False,
                        )
                    if self.is_main_process:
                        n_critic = self.latent_critic.num_params
                        logging.info(
                            "[ActionForcing] LatentSAM2Critic built: "
                            "d_model=%d num_blocks=%d num_params=%.2fM "
                            "frame_pool=%s (DDP=%s)",
                            critic_hidden, critic_num_blocks,
                            n_critic / 1e6, frame_pool,
                            self.latent_critic_ddp is not None,
                        )

                # ----- Dense perceptual approximators (mse / lpips /
                # gan_d_approx) — all share the PerceptualApprox 3D-CNN
                # architecture. Each takes (gen_lat, gt_lat) and outputs
                # a dense per-frame-per-spatial-token field; the three
                # differ only in their training target (per-token MSE,
                # per-token LPIPS, per-token disc logit map).
                # When ``gan_d_approx_loss_weight > 0``, gan_d_approx
                # REPLACES LatentSAM2Critic in the gen-side path.
                self.mse_approx = None
                self.mse_approx_ddp = None
                self.mse_approx_optimizer = None
                self.lpips_approx = None
                self.lpips_approx_ddp = None
                self.lpips_approx_optimizer = None
                self.msssim_approx = None
                self.msssim_approx_ddp = None
                self.msssim_approx_optimizer = None
                self.maniqa_approx = None
                self.maniqa_approx_ddp = None
                self.maniqa_approx_optimizer = None
                self.gan_d_approx = None
                self.gan_d_approx_ddp = None
                self.gan_d_approx_optimizer = None
                self._lpips_target_model = None  # lazy-built no_grad LPIPS
                self._maniqa_target_model = None  # lazy-built no_grad MANIQA
                # Read directly from cfg here so we avoid the order-of-
                # init issue (the knob assignments to self happen later
                # in this same __init__, after the disc/critic build).
                _approx_d_model = int(
                    getattr(self.config, "perceptual_approx_d_model", 256)
                )
                _approx_num_blocks = int(
                    getattr(self.config, "perceptual_approx_num_blocks", 4)
                )
                _need_mse_approx = float(
                    getattr(self.config, "mse_approx_loss_weight", 0.0)
                ) > 0
                _need_lpips_approx = float(
                    getattr(self.config, "lpips_approx_loss_weight", 0.0)
                ) > 0
                _need_msssim_approx = float(
                    getattr(self.config, "msssim_approx_loss_weight", 0.0)
                ) > 0
                _need_maniqa_approx = float(
                    getattr(self.config, "maniqa_approx_loss_weight", 0.0)
                ) > 0
                _need_gan_d_approx = False  # gan_d_approx is now LatentSAM2Critic (existing path); PerceptualApprox-as-disc-approx is dropped per user realignment
                if (
                    _need_mse_approx or _need_lpips_approx
                    or _need_msssim_approx or _need_maniqa_approx
                    or _need_gan_d_approx
                ):
                    from model.perceptual_approx import PerceptualApprox

                    def _build_approx(name: str, single_input: bool = False):
                        m = PerceptualApprox(
                            in_channels=16,
                            d_model=_approx_d_model,
                            num_blocks=_approx_num_blocks,
                            single_input=single_input,
                        ).to(device=self.device, dtype=torch.float32)
                        m.train()
                        ddp = None
                        if self.world_size > 1:
                            ddp = DDP(
                                m, device_ids=[self.local_rank],
                                output_device=self.local_rank,
                                find_unused_parameters=False,
                                broadcast_buffers=False,
                            )
                        if self.is_main_process:
                            logging.info(
                                "[ActionForcing] %s built: d_model=%d "
                                "num_blocks=%d num_params=%.2fM "
                                "(DDP=%s)",
                                name,
                                _approx_d_model,
                                _approx_num_blocks,
                                m.num_params / 1e6,
                                ddp is not None,
                            )
                        return m, ddp

                    if _need_mse_approx:
                        self.mse_approx, self.mse_approx_ddp = _build_approx(
                            "MSEApprox"
                        )
                    if _need_lpips_approx:
                        self.lpips_approx, self.lpips_approx_ddp = (
                            _build_approx("LPIPSApprox")
                        )
                    if _need_msssim_approx:
                        (
                            self.msssim_approx,
                            self.msssim_approx_ddp,
                        ) = _build_approx("MSSSIMApprox")
                    if _need_maniqa_approx:
                        (
                            self.maniqa_approx,
                            self.maniqa_approx_ddp,
                        ) = _build_approx(
                            "MANIQAApprox", single_input=True,
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

        # dmd_context_mix_p schedule — linear ramp from the config-time
        # start to ``dmd_context_mix_p_target`` over the first
        # ``dmd_context_mix_p_ramp_steps`` outer steps. With ramp_steps
        # <= 0 (default), the knob is held at the start value forever.
        # Mutates ``self.model.dmd_context_mix_p`` once per iter via
        # ``_apply_dmd_context_mix_p_schedule``.
        self._dmd_context_mix_p_start = float(
            getattr(self.config, "dmd_context_mix_p", 0.5)
        )
        self._dmd_context_mix_p_target = float(
            getattr(
                self.config,
                "dmd_context_mix_p_target",
                self._dmd_context_mix_p_start,
            )
        )
        self._dmd_context_mix_p_ramp_steps = int(
            getattr(self.config, "dmd_context_mix_p_ramp_steps", 0)
        )
        if (
            self._dmd_context_mix_p_ramp_steps > 0
            and self._dmd_context_mix_p_start != self._dmd_context_mix_p_target
            and self.is_main_process
        ):
            logging.info(
                "[ActionForcing] dmd_context_mix_p schedule active: "
                "linear ramp %.3f -> %.3f over the first %d outer steps.",
                self._dmd_context_mix_p_start,
                self._dmd_context_mix_p_target,
                self._dmd_context_mix_p_ramp_steps,
            )

        # dmd_context_mix_p sensor-gate — overrides the linear ramp when
        # ``dmd_context_mix_p_sensor_enabled=true``. Reads the most
        # recent ``gen/dmd_pf_minus_pr_mae`` value (= |pred_fake -
        # pred_real|, the unnormalised DMD push direction). When this
        # drops below ``dmd_context_mix_p_sensor_threshold`` the
        # scorers are converging — student is catching up to teacher
        # and the LoRA needs more GT context to remain meaningful as a
        # supervisory target. The sensor flips ``model.dmd_context_
        # mix_p`` from ``_low`` to ``_high`` until the metric rises
        # back above the threshold, then flips back. Stateless gate
        # (no hysteresis), reads previous gen-iter's metric. Mutates
        # ``self.model.dmd_context_mix_p`` once per outer iter via
        # ``_apply_dmd_context_mix_p_schedule``.
        self._dmd_context_mix_p_sensor_enabled = bool(
            getattr(self.config, "dmd_context_mix_p_sensor_enabled", False)
        )
        self._dmd_context_mix_p_sensor_threshold = float(
            getattr(self.config, "dmd_context_mix_p_sensor_threshold", 0.1)
        )
        self._dmd_context_mix_p_sensor_low = float(
            getattr(self.config, "dmd_context_mix_p_sensor_low", 0.15)
        )
        self._dmd_context_mix_p_sensor_high = float(
            getattr(self.config, "dmd_context_mix_p_sensor_high", 0.7)
        )
        # State updated after each gen step. ``None`` until the first
        # gen step has produced a metric — sensor falls back to
        # ``_low`` while None to avoid spuriously flipping high before
        # we have a real reading.
        self._latest_dmd_pf_minus_pr_mae: Optional[float] = None
        if (
            self._dmd_context_mix_p_sensor_enabled
            and self.is_main_process
        ):
            logging.info(
                "[ActionForcing] dmd_context_mix_p sensor active: "
                "threshold=%.3f, low=%.3f, high=%.3f. Reads "
                "gen/dmd_pf_minus_pr_mae each gen iter; flips mix_p "
                "high when below threshold, low when above. Overrides "
                "linear ramp.",
                self._dmd_context_mix_p_sensor_threshold,
                self._dmd_context_mix_p_sensor_low,
                self._dmd_context_mix_p_sensor_high,
            )

        # dmd_context_mix_p HARD step-switch — fires once at outer
        # ``self.step == dmd_context_mix_p_step_switch_at`` and stays
        # at ``_target`` thereafter. Default 0 = disabled (legacy
        # behavior: ramp + optional sensor). When > 0, this overrides
        # the linear ramp (but is still overridden by the sensor when
        # ``_sensor_enabled``). Use case: pre-warmup with low GT
        # context, then step-switch to high GT once the LoRA has
        # stabilized — discrete, no interpolation.
        self._dmd_context_mix_p_step_switch_at = int(
            getattr(self.config, "dmd_context_mix_p_step_switch_at", 0)
        )
        if (
            self._dmd_context_mix_p_step_switch_at > 0
            and not self._dmd_context_mix_p_sensor_enabled
            and self.is_main_process
        ):
            logging.info(
                "[ActionForcing] dmd_context_mix_p step-switch active: "
                "step < %d -> %.3f, step >= %d -> %.3f (overrides "
                "linear ramp).",
                self._dmd_context_mix_p_step_switch_at,
                self._dmd_context_mix_p_start,
                self._dmd_context_mix_p_step_switch_at,
                self._dmd_context_mix_p_target,
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
            context_noise=context_noise,
        )
        # The ActionForcingDMD model needs the pipeline reference for backward
        # simulation inside generator_loss / critic_loss.
        self.model.inference_pipeline = self.pipeline

        if self.is_main_process:
            logging.info(
                "[ActionForcing] Pipeline: num_frame_per_block=%d "
                "chunks_per_rolling_step=%d denoising_step_list=%s "
                "context_noise=%d num_max_frames=%d rollout_frames=%d "
                "(gradient window = last %d frames; warmup = %d frames)",
                num_frame_per_block, chunks_per_rolling_step,
                denoising_step_list, context_noise, num_max_frames,
                rollout_frames, num_max_frames,
                rollout_frames - num_max_frames,
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
        # ``gan_warmup_shape``: shape of the gen-side GAN weight ramp
        # over ``gan_warmup_steps``. ``"linear"`` (default, legacy
        # behaviour): t/T constant derivative — first non-zero step
        # delivers (1/T) × plateau weight. ``"quadratic"``: (t/T)² —
        # zero derivative at gate-open; first non-zero step delivers
        # (1/T²) × plateau, two orders of magnitude softer than
        # linear at small t. ``"cosine"``: ½(1-cos(πt/T)) — S-curve
        # with zero derivative at both ends. Recommended "quadratic"
        # to suppress fish-scale artifacts at GAN onset.
        self.gan_warmup_shape = str(
            getattr(cfg, "gan_warmup_shape", "linear")
        ).lower()
        if self.gan_warmup_shape not in ("linear", "quadratic", "cosine"):
            raise ValueError(
                "gan_warmup_shape must be one of 'linear' | 'quadratic' "
                f"| 'cosine'; got {self.gan_warmup_shape!r}."
            )
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

        # Distilled-critic config + optimizer (only when the disc was
        # built with ``gan_sam2_distilled_critic=True``). Read knobs
        # unconditionally so they're available for log surfacing even
        # when the critic isn't built.
        self.gan_critic_warmup_steps = int(
            getattr(cfg, "gan_critic_warmup_steps", 500)
        )
        # Defers D-side training (the SAM2 disc heads + LatentSAM2Critic
        # value-distillation step) until this outer step. Pre-start the
        # entire D pass (forwards, R1/R2 penalty, D backward, D optimizer
        # step) is skipped — no D activations live, no D gradient, no
        # D-side compute. Decouples from ``gan_critic_warmup_steps``,
        # which gates only the GEN-side gradient (Path 3). Default 0
        # preserves the original behaviour (D trains every iter from
        # step 0). DDP-safe: the gate reads ``current_step`` which is
        # rank-invariant, so every rank skips/runs in lockstep — no
        # NCCL collective count mismatch.
        self.gan_disc_start_step = int(
            getattr(cfg, "gan_disc_start_step", 0)
        )
        # Multi-step critic training (mirrors action_critic's
        # ``critic_updates_per_step``): step the critic optimizer K_c
        # times per gen step. Default 1 = legacy single-step behavior.
        self.gan_critic_updates_per_step = int(
            getattr(cfg, "gan_critic_updates_per_step", 1)
        )
        # Dense per-token value distillation. When True, Path 2's
        # value-distillation MSE matches the critic's PER-TOKEN logit
        # map (LatentSAM2Critic.forward(dense=True)) against the disc's
        # PER-TOKEN logit map (forward_dense_heads, pooled 4-pix-frames
        # → 1-latent-frame). 8x13 ≈ 104x more constraints per sample
        # than the scalar/per-frame MSE the legacy path used. Default
        # False = legacy scalar MSE (for back-compat).
        self.gan_critic_dense_distillation = bool(
            getattr(cfg, "gan_critic_dense_distillation", False)
        )
        # ---------- Pixel-space perceptual losses (v13) ----------
        # LPIPS (Zhang et al., CVPR 2018) — VGG-based perceptual
        # distance between gen pixels and GT pixels. Direct anti-blur
        # signal that doesn't go through the GAN distillation chain.
        # Operates on a small frame subset per iter (memory budget).
        # ``lpips_loss_weight=0`` disables. Lazy-loaded on first use.
        self.lpips_loss_weight = float(
            getattr(cfg, "lpips_loss_weight", 0.0)
        )
        self.lpips_n_frames = int(
            getattr(cfg, "lpips_n_frames", 2)
        )
        # Pixel-space L1/MSE (the "(3)" term). Anti-streaming-drift
        # signal — direct pixel supervision that DMD doesn't provide.
        # Low weight (0.01-0.1) is the standard.
        self.pixel_recon_loss_weight = float(
            getattr(cfg, "pixel_recon_loss_weight", 0.0)
        )
        # ``mse | l1`` — L1 is more outlier-robust, MSE is smoother.
        self.pixel_recon_loss_type = str(
            getattr(cfg, "pixel_recon_loss_type", "l1")
        )
        # Optional random crop for LPIPS / pixel-recon — keeps memory
        # bounded when full-resolution decode (480×832) is too large.
        # 0 disables cropping (use full decoded resolution).
        self.lpips_crop_size = int(
            getattr(cfg, "lpips_crop_size", 0)
        )

        # ---------- Dense perceptual approximators (v13 redesign) ---------
        # Train cheap latent-space approximators that predict the
        # per-token MSE / LPIPS distance between gen and GT pixels —
        # gen-side loss flows gradient through these (no VAE backprop)
        # so we get pixel-level supervision without OOM. The approxes
        # are trained on dense per-token targets computed via no_grad
        # VAE decode (which the disc training path already does), so
        # the only extra cost per iter is the LPIPS forward + per-token
        # mean-pool. See ``model/perceptual_approx.py`` for design.
        self.mse_approx_loss_weight = float(
            getattr(cfg, "mse_approx_loss_weight", 0.0)
        )
        # The "MSE" approx actually predicts a per-token combined
        # ``l2_w * (gen-gt)² + l1_w * |gen-gt|`` field. Both terms are
        # GRANULAR (per-pixel reduced to per-latent-token by mean-pool),
        # not just per-frame scalars. Setting ``l1_w=0`` recovers the
        # legacy pure-MSE target.
        self.mse_approx_l2_weight = float(
            getattr(cfg, "mse_approx_l2_weight", 1.0)
        )
        self.mse_approx_l1_weight = float(
            getattr(cfg, "mse_approx_l1_weight", 0.0)
        )
        self.lpips_approx_loss_weight = float(
            getattr(cfg, "lpips_approx_loss_weight", 0.0)
        )
        # MS-SSIM approx — per-frame MS-SSIM scalar broadcast across
        # the latent token grid. Gen-side loss is sign-flipped (gen
        # wants MS-SSIM HIGH = "looks similar to GT in structure").
        # Less blur-prone than MSE because MS-SSIM preserves local
        # structure / edges instead of penalizing pixel-wise diffs.
        self.msssim_approx_loss_weight = float(
            getattr(cfg, "msssim_approx_loss_weight", 0.0)
        )
        # MS-SSIM input domain. Default ``image`` matches the standard
        # pytorch_msssim contract (raw normalised pixels). Setting to
        # ``edges`` runs MS-SSIM on per-channel Sobel gradient
        # magnitudes — defeats the gray-collapse failure mode where
        # uniformly-low-contrast outputs still match locally on raw
        # pixel windows but fail on edge maps. See
        # ``_compute_dense_perceptual_targets`` for the implementation.
        self.msssim_target_domain = str(
            getattr(cfg, "msssim_target_domain", "image")
        ).lower().strip()
        if self.msssim_target_domain not in ("image", "edges"):
            raise ValueError(
                f"msssim_target_domain must be 'image' or 'edges'; got "
                f"{self.msssim_target_domain!r}."
            )
        # MANIQA NR-IQA approx (no GT — gen-only). Predicts a sparse
        # per-token quality score map. Targets are built every iter by:
        #   1. VAE-decoding the gen latents (no_grad) → pixel frames.
        #   2. Sampling N random 224×224 patches from selected frames
        #      (DDP-synced offsets, ``maniqa_n_patches_per_frame``).
        #   3. Forwarding each patch through a frozen pretrained
        #      MANIQA model → per-patch scalar quality score in [0, 1].
        #   4. Mapping each patch's pixel region to the latent-token
        #      coordinates it covers; assigning the score there. All
        #      other tokens NaN → masked from the approx training loss.
        # Gen-side loss is sign-flipped (gen wants quality HIGH).
        # ``maniqa_n_frames``: number of latent frames per iter to
        # supervise (rest of frames carry NaN this iter; supervised on
        # later iters via random subsampling). Default 4 (matches LPIPS-
        # pixel pattern). Set equal to F_lat to supervise every frame
        # every iter ("option 1" — slower target build).
        # ``maniqa_n_patches_per_frame``: random 224×224 patches per
        # selected frame. Default 1.
        self.maniqa_approx_loss_weight = float(
            getattr(cfg, "maniqa_approx_loss_weight", 0.0)
        )
        self.maniqa_n_frames = int(
            getattr(cfg, "maniqa_n_frames", 4)
        )
        self.maniqa_n_patches_per_frame = int(
            getattr(cfg, "maniqa_n_patches_per_frame", 1)
        )
        # Pretrained MANIQA variant (pyiqa metric_name). 'maniqa-pipal'
        # was trained on PIPAL (which includes GAN-distortion images);
        # the default 'maniqa' is KonIQ-10k-trained.
        self.maniqa_metric_name = str(
            getattr(cfg, "maniqa_metric_name", "maniqa-pipal")
        )
        # gan_d_approx replaces LatentSAM2Critic when > 0. Same arch
        # (PerceptualApprox), takes (gen_lat, gt_lat), trained against
        # the pixel-disc's DENSE per-token output map for the gen
        # latent. Gen-side loss = -gan_d_approx(gen, gt).mean() (same
        # sign convention as the legacy ``-critic(fake).mean()``).
        self.gan_d_approx_loss_weight = float(
            getattr(cfg, "gan_d_approx_loss_weight", 0.0)
        )
        # Dense target supervision (the "(2)" in user's spec): the
        # approxes are trained on per-token MSE/LPIPS targets PLUS
        # a scalar mean-alignment term. ``mean_align_weight`` weights
        # the second term in the approx training loss; default 0.1
        # (mean is a soft constraint relative to the dense target).
        self.perceptual_approx_mean_align_weight = float(
            getattr(cfg, "perceptual_approx_mean_align_weight", 0.1)
        )
        # Approx model architecture knobs.
        self.perceptual_approx_d_model = int(
            getattr(cfg, "perceptual_approx_d_model", 256)
        )
        self.perceptual_approx_num_blocks = int(
            getattr(cfg, "perceptual_approx_num_blocks", 4)
        )
        self.perceptual_approx_lr = float(
            getattr(cfg, "perceptual_approx_lr", 2e-4)
        )
        self.perceptual_approx_warmup_steps = int(
            getattr(cfg, "perceptual_approx_warmup_steps", 25)
        )
        # Optional linear ramp on the perceptual-approx gen-side
        # contribution AFTER the warmup gate fires. Default 0 = hard
        # switch at warmup_steps (legacy behaviour). When > 0, the
        # combined ramp_factor multiplies every per-approx weight in
        # _compute_gen_side_perceptual_loss, growing linearly from 0
        # to 1.0 over ``perceptual_approx_ramp_steps`` outer steps
        # starting at ``perceptual_approx_warmup_steps``. Useful when
        # MANIQA's full-weight onset would otherwise destabilise
        # nearby losses (the additive "bias" effect on the bundled
        # r3gan_g_loss_weighted seen in v3/v4 runs).
        self.perceptual_approx_ramp_steps = int(
            getattr(cfg, "perceptual_approx_ramp_steps", 0)
        )
        # Constant-weight L1 (MAE) and L2 (MSE) between the student's
        # pred_image latent and the GT latent window. Direct anti-drift
        # anchor, applied EVERY step with no ramp or warmup. Default 0
        # for both keeps earlier configs unchanged. Used by v10 onwards
        # as a small constant pull toward GT to complement DMD+GAN.
        self.gt_latent_mae_loss_weight = float(
            getattr(cfg, "gt_latent_mae_loss_weight", 0.0)
        )
        self.gt_latent_mse_loss_weight = float(
            getattr(cfg, "gt_latent_mse_loss_weight", 0.0)
        )
        # ----- v11-style multi-noise (Lipschitz-by-density) -----
        # When > 0, sample K interpolation points between
        # ``fake_lat`` and ``real_lat`` each iter, compute the dense
        # per-token target metrics on each, and add their MSE to each
        # approx's training loss. The denser supervision constrains
        # the approx's value field across the (fake → real) line so
        # its gradient (what the gen consumes via ``.mean()``) is
        # meaningful by Lipschitz interpolation.
        # Each interpolation costs: 1 no_grad VAE decode + (optionally)
        # 1 SAM2 forward + 1 LPIPS forward + K small approx forwards.
        # Sequential per-alpha processing bounds peak memory.
        self.perceptual_approx_n_interp_samples = int(
            getattr(cfg, "perceptual_approx_n_interp_samples", 0)
        )
        if (
            self.gan_enabled
            and getattr(self, "latent_critic", None) is not None
        ):
            critic_lr = float(getattr(cfg, "gan_critic_lr", 2e-4))
            critic_betas = tuple(
                getattr(cfg, "gan_critic_betas", [0.0, 0.9])
            )
            critic_eps = float(getattr(cfg, "gan_critic_eps", 1e-8))
            critic_wd = float(getattr(cfg, "gan_critic_weight_decay", 0.0))
            critic_params = [
                p for p in self.latent_critic.parameters()
                if p.requires_grad
            ]
            if not critic_params:
                raise RuntimeError(
                    "latent_critic has no trainable parameters."
                )
            self.latent_critic_optimizer = torch.optim.AdamW(
                critic_params,
                lr=critic_lr,
                betas=critic_betas,
                eps=critic_eps,
                weight_decay=critic_wd,
            )
            if self.is_main_process:
                n_params = sum(p.numel() for p in critic_params)
                logging.info(
                    "[ActionForcing] LatentSAM2Critic optimizer built: "
                    "AdamW lr=%.2e betas=%s wd=%.4f params=%.2fM "
                    "(warmup=%d, updates_per_step=%d, dense=%s)",
                    critic_lr, critic_betas, critic_wd, n_params / 1e6,
                    self.gan_critic_warmup_steps,
                    self.gan_critic_updates_per_step,
                    self.gan_critic_dense_distillation,
                )

        # ----- Perceptual approx optimizers (mse / lpips) -----
        if getattr(self, "mse_approx", None) is not None:
            self.mse_approx_optimizer = torch.optim.AdamW(
                [p for p in self.mse_approx.parameters() if p.requires_grad],
                lr=self.perceptual_approx_lr,
                betas=(0.0, 0.9),
                eps=1e-8,
                weight_decay=0.0,
            )
            if self.is_main_process:
                logging.info(
                    "[ActionForcing] MSEApprox optimizer built: "
                    "AdamW lr=%.2e weight=%.3f warmup=%d "
                    "mean_align_w=%.3f",
                    self.perceptual_approx_lr,
                    self.mse_approx_loss_weight,
                    self.perceptual_approx_warmup_steps,
                    self.perceptual_approx_mean_align_weight,
                )
        if getattr(self, "lpips_approx", None) is not None:
            self.lpips_approx_optimizer = torch.optim.AdamW(
                [
                    p for p in self.lpips_approx.parameters()
                    if p.requires_grad
                ],
                lr=self.perceptual_approx_lr,
                betas=(0.0, 0.9),
                eps=1e-8,
                weight_decay=0.0,
            )
            if self.is_main_process:
                logging.info(
                    "[ActionForcing] LPIPSApprox optimizer built: "
                    "AdamW lr=%.2e weight=%.3f warmup=%d "
                    "mean_align_w=%.3f",
                    self.perceptual_approx_lr,
                    self.lpips_approx_loss_weight,
                    self.perceptual_approx_warmup_steps,
                    self.perceptual_approx_mean_align_weight,
                )
        if getattr(self, "msssim_approx", None) is not None:
            self.msssim_approx_optimizer = torch.optim.AdamW(
                [
                    p for p in self.msssim_approx.parameters()
                    if p.requires_grad
                ],
                lr=self.perceptual_approx_lr,
                betas=(0.0, 0.9),
                eps=1e-8,
                weight_decay=0.0,
            )
            if self.is_main_process:
                logging.info(
                    "[ActionForcing] MSSSIMApprox optimizer built: "
                    "AdamW lr=%.2e weight=%.3f warmup=%d "
                    "(gen-side sign FLIPPED — gen wants MS-SSIM HIGH)",
                    self.perceptual_approx_lr,
                    self.msssim_approx_loss_weight,
                    self.perceptual_approx_warmup_steps,
                )
        if getattr(self, "maniqa_approx", None) is not None:
            self.maniqa_approx_optimizer = torch.optim.AdamW(
                [
                    p for p in self.maniqa_approx.parameters()
                    if p.requires_grad
                ],
                lr=self.perceptual_approx_lr,
                betas=(0.0, 0.9),
                eps=1e-8,
                weight_decay=0.0,
            )
            if self.is_main_process:
                logging.info(
                    "[ActionForcing] MANIQAApprox optimizer built: "
                    "AdamW lr=%.2e weight=%.3f warmup=%d "
                    "n_frames=%d n_patches/frame=%d metric=%s "
                    "(NR-IQA, single-input, sign FLIPPED — gen wants "
                    "quality HIGH)",
                    self.perceptual_approx_lr,
                    self.maniqa_approx_loss_weight,
                    self.perceptual_approx_warmup_steps,
                    self.maniqa_n_frames,
                    self.maniqa_n_patches_per_frame,
                    self.maniqa_metric_name,
                )

        # ------------------------------------------------------------------
        # Online real_teacher (v14-LoRA flow training vs GT).
        # ------------------------------------------------------------------
        # Builds an AdamW over the LoRA params kept alive on
        # ``model.real_score.model`` by ``_load_real_score_with_v14_lora``
        # (which skipped ``merge_and_unload`` because the YAML flag
        # ``real_teacher_train_online`` was True). Stays None when the
        # flag is False.
        self.real_teacher_optimizer: Optional[torch.optim.Optimizer] = None
        self.real_teacher_train_online = bool(
            getattr(cfg, "real_teacher_train_online", False)
        )
        if self.real_teacher_train_online:
            rt_params = list(
                getattr(self.model, "_real_teacher_trainable_params", []) or []
            )
            if not rt_params:
                # Fallback: discover by `requires_grad`. Should never
                # trigger if the model-side load went well, but a
                # missing stash here means the LoRA wrap broke silently
                # — fail loud.
                rt_params = [
                    p for p in self.model.real_score.model.parameters()
                    if p.requires_grad
                ]
            if not rt_params:
                raise RuntimeError(
                    "real_teacher_train_online=True but real_score has "
                    "no trainable params; check "
                    "_load_real_score_with_v14_lora's online path."
                )
            rt_lr = float(getattr(cfg, "real_teacher_lr", 5.0e-05))
            rt_betas = tuple(getattr(cfg, "real_teacher_betas", [0.9, 0.999]))
            rt_eps = float(getattr(cfg, "real_teacher_eps", 1.0e-08))
            rt_wd = float(getattr(cfg, "real_teacher_weight_decay", 0.01))
            self.real_teacher_optimizer = torch.optim.AdamW(
                rt_params,
                lr=rt_lr,
                betas=rt_betas,
                eps=rt_eps,
                weight_decay=rt_wd,
            )
            self.real_teacher_max_grad_norm = float(
                getattr(cfg, "real_teacher_max_grad_norm", 1.0)
            )
            self.real_teacher_warmup_steps = int(
                getattr(cfg, "real_teacher_warmup_steps", 200)
            )
            self._real_teacher_base_lr = rt_lr
            if self.is_main_process:
                n_params = sum(p.numel() for p in rt_params)
                logging.info(
                    "[ActionForcing] real_teacher optimizer built: "
                    "AdamW lr=%.2e betas=%s wd=%.4f params=%.2fM "
                    "(warmup_steps=%d, max_grad_norm=%.2f)",
                    rt_lr, rt_betas, rt_wd, n_params / 1e6,
                    self.real_teacher_warmup_steps,
                    self.real_teacher_max_grad_norm,
                )
        else:
            self.real_teacher_max_grad_norm = 1.0
            self.real_teacher_warmup_steps = 0
            self._real_teacher_base_lr = 0.0

        # ------------------------------------------------------------------
        # noise_aux optimizer (small forward-noising ConvNet). Built only
        # when ``noise_aux_enabled`` on the model. Trains independently
        # of the LoRA — its own AdamW + grad clip + zero. Loss is set up
        # in ``model.compute_noise_aux_loss_streaming``; the trainer
        # fires it once per gen step after the gen backward.
        # ------------------------------------------------------------------
        self.noise_aux_optimizer: Optional[torch.optim.Optimizer] = None
        self.noise_aux_enabled = bool(
            getattr(self.model, "noise_aux_enabled", False)
            and getattr(self.model, "noise_aux", None) is not None
        )
        if self.noise_aux_enabled:
            na_params = [
                p for p in self.model.noise_aux.parameters() if p.requires_grad
            ]
            if not na_params:
                raise RuntimeError(
                    "noise_aux_enabled=True but noise_aux has no trainable "
                    "params — check NoiseAuxLite construction."
                )
            na_lr = float(getattr(cfg, "noise_aux_lr", 1.0e-04))
            na_betas = tuple(getattr(cfg, "noise_aux_betas", [0.9, 0.999]))
            na_eps = float(getattr(cfg, "noise_aux_eps", 1.0e-08))
            na_wd = float(getattr(cfg, "noise_aux_weight_decay", 0.0))
            self.noise_aux_optimizer = torch.optim.AdamW(
                na_params,
                lr=na_lr,
                betas=na_betas,
                eps=na_eps,
                weight_decay=na_wd,
            )
            self.noise_aux_max_grad_norm = float(
                getattr(cfg, "noise_aux_max_grad_norm", 1.0)
            )
            if self.is_main_process:
                n_na_params = sum(p.numel() for p in na_params)
                logging.info(
                    "[ActionForcing] noise_aux optimizer built: "
                    "AdamW lr=%.2e betas=%s wd=%.4f params=%.2fM",
                    na_lr, na_betas, na_wd, n_na_params / 1e6,
                )
        else:
            self.noise_aux_max_grad_norm = 1.0

        # ``fake_score_ema_weight`` (0.0 = off, current behavior; e.g.
        # 0.95 = engage). After each generator optimizer.step(), the
        # fake_score's params get EMA-pulled toward the generator's
        # newly-updated params:
        #     ψ ← w · ψ + (1 - w) · θ
        # This lets the fake_score track p_gen with much fewer dedicated
        # diffusion-loss updates, so dfake_gen_update_ratio can be
        # dropped from 5 to 1 or 2 with no quality loss (paper §3.3).
        self.fake_score_ema_weight = float(
            getattr(cfg, "fake_score_ema_weight", 0.0)
        )
        if not (0.0 <= self.fake_score_ema_weight < 1.0):
            raise ValueError(
                f"fake_score_ema_weight must be in [0, 1); got "
                f"{self.fake_score_ema_weight!r}."
            )
        if self.is_main_process and self.fake_score_ema_weight > 0.0:
            logging.info(
                "[ActionForcing] fake_score_ema_weight=%.4f",
                self.fake_score_ema_weight,
            )

    def _maybe_ema_fake_score_from_generator(self) -> None:
        """Flash-DMD §3.3 EMA: after each generator optimizer.step(),
        pull the fake_score's params toward the generator's params
        with weight ``self.fake_score_ema_weight``. No-op when the
        weight is 0.

        Iterates by ``named_parameters()`` and matches by name +
        shape. On the first call we audit the match coverage and log
        a one-shot summary so silent param-structure drift (rename,
        shape mismatch, accidental architectural divergence) is
        loud — without that audit a fake_score param could quietly
        decouple from the generator forever.
        """
        if self.fake_score_ema_weight <= 0.0:
            return
        # Both modules may be DDP-wrapped; unwrap to the inner module.
        gen_inner = (
            self.generator_ddp.module
            if getattr(self, "generator_ddp", None) is not None
            else self.model.generator.model
        )
        fake_inner = (
            self.fake_score_ddp.module
            if getattr(self, "fake_score_ddp", None) is not None
            else self.model.fake_score.model
        )
        gen_params = dict(gen_inner.named_parameters())
        w = self.fake_score_ema_weight

        # One-shot startup audit. Counts matched / skipped on the
        # first invocation only; logs the summary so operators see
        # whether 100% of fake_score params are EMA-tracked. If the
        # match rate is < 100%, the SKIPPED ones drift untouched and
        # that's almost always wrong (architectural drift, not
        # intentional).
        if not getattr(self, "_fake_ema_audit_done", False):
            matched = 0
            skipped = 0
            skipped_examples = []
            for name, p_fake in fake_inner.named_parameters():
                p_gen = gen_params.get(name)
                if p_gen is None or p_gen.shape != p_fake.shape:
                    skipped += 1
                    if len(skipped_examples) < 5:
                        reason = (
                            "missing-in-generator" if p_gen is None
                            else f"shape-mismatch ({tuple(p_gen.shape)} vs {tuple(p_fake.shape)})"
                        )
                        skipped_examples.append(f"{name}: {reason}")
                else:
                    matched += 1
            total = matched + skipped
            if self.is_main_process:
                logging.info(
                    "[ActionForcing] fake_score EMA audit: matched=%d "
                    "skipped=%d total=%d (%.1f%% coverage). EMA "
                    "weight=%.4f.",
                    matched, skipped, total,
                    100.0 * matched / max(1, total),
                    w,
                )
                if skipped > 0:
                    logging.warning(
                        "[ActionForcing] fake_score EMA: %d params NOT "
                        "EMA-tracked. First %d examples: %s",
                        skipped, len(skipped_examples), skipped_examples,
                    )
            self._fake_ema_audit_done = True

        with torch.no_grad():
            for name, p_fake in fake_inner.named_parameters():
                p_gen = gen_params.get(name)
                if p_gen is None or p_gen.shape != p_fake.shape:
                    continue
                p_fake.data.mul_(w).add_(p_gen.data, alpha=1.0 - w)

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
        # Memory audit: pre-training snapshot. Captures static param +
        # buffer footprint AFTER all heads / DDP wraps / optimizers
        # are built but BEFORE any step's activation memory has hit
        # the allocator. Call again after step 1 + step 5 below
        # (gated by ``memory_audit_enabled`` so prod runs aren't
        # spammed). Default off; smoke turns on.
        if bool(getattr(cfg, "memory_audit_enabled", False)):
            self._dump_memory_audit("pre_train")
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
        # MAE-extension code path was removed; rollouts no longer grow
        # beyond ``rollout_frames``. ``max_total_rollout_frames`` stays
        # as a name for downstream slicing but equals ``rollout_frames``.
        max_rolls_per_ride = int(
            getattr(cfg, "max_rolls_per_ride", 0)
        )
        max_total_rollout_frames = rollout_frames

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
                "warmup_frames=%d) max_rolls_per_ride=%d "
                "(=> max_total_rollout_frames=%d when ride is long enough "
                "and MAE stays under threshold)",
                max_steps, dfake_gen_update_ratio, num_training_frames,
                rollout_frames, num_training_frames,
                rollout_frames - num_training_frames,
                max_rolls_per_ride, max_total_rollout_frames,
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
            # Linear ramp on ``dmd_context_mix_p`` (no-op when the
            # target == start or ramp_steps <= 0). Mutates
            # ``self.model.dmd_context_mix_p`` so the next forward
            # picks up the new value without further plumbing.
            _dmd_ctx_mix_p_now = self._apply_dmd_context_mix_p_schedule()
            # Decide whether this iter trains the generator or the critic.
            train_generator = (self.step % dfake_gen_update_ratio == 0)

            if train_generator:
                generator_log_dict = self._fwdbwd_one_step(
                    train_generator=True,
                    rollout_frames=rollout_frames,
                    max_total_rollout_frames=max_total_rollout_frames,
                    cf_dmdctx=cf_dmdctx,
                )
                # Cache the most recent ``dmd_pf_minus_pr_mae`` so the
                # NEXT iter's schedule call can read it. Sensor-gated
                # mix_p mode (``dmd_context_mix_p_sensor_enabled``)
                # uses this value to decide whether to flip mix_p
                # high (scorers converging → bolster teacher with GT
                # context) or low (scorers disagreeing → standard).
                if (
                    isinstance(generator_log_dict, dict)
                    and "dmd_pf_minus_pr_mae" in generator_log_dict
                ):
                    _v = generator_log_dict["dmd_pf_minus_pr_mae"]
                    try:
                        self._latest_dmd_pf_minus_pr_mae = float(_v)
                    except (TypeError, ValueError):
                        pass

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

            # ----- State-probe optimizer step (LoRA-side aux loss) -----
            # Mirrors the fake/real_teacher pattern: clip → step → zero.
            # Probe gets gradient ONLY when the LoRA aux pass populated
            # state_preds AND the gen-step backward fired (= gen iter).
            # On critic-only iters the probe params have no grad, so
            # the params_with_grad check elides the step naturally.
            state_probe_grad_norm_val = 0.0
            if self.state_probe_optimizer is not None:
                sp_params_with_grad = [
                    p for p in self.state_probe_optimizer.param_groups[0]["params"]
                    if p.grad is not None
                ]
                if sp_params_with_grad:
                    sgn = torch.nn.utils.clip_grad_norm_(
                        sp_params_with_grad,
                        max_norm=self.state_probe_max_grad_norm,
                    )
                    state_probe_grad_norm_val = (
                        float(sgn.item()) if torch.is_tensor(sgn) else float(sgn)
                    )
                    self.state_probe_optimizer.step()
                self.state_probe_optimizer.zero_grad(set_to_none=True)

            # Flash-DMD §3.3: EMA fake_score toward generator AFTER
            # the fake_optimizer step. Order matters: applying Adam's
            # accumulated momentum/variance to fake's pre-step params
            # is the intended math; the EMA pull then becomes the
            # FINAL mutation of fake's params for the iter, with no
            # stale-momentum interaction. No-op when
            # ``fake_score_ema_weight == 0`` (default).
            if (
                train_generator
                and self.fake_score_ema_weight > 0.0
            ):
                self._maybe_ema_fake_score_from_generator()

            # ----- Online real_teacher: clip + step + zero -----
            real_teacher_grad_norm_val = 0.0
            # Hard gate: skip the LoRA optimizer step entirely until
            # ``aux_teacher_start_step``. The aux pass is also gated
            # in ``compute_generator_loss_streaming`` (so no aux grad
            # accumulates pre-start), but defensively zero any stray
            # gradient here too in case some other path populates it.
            _aux_start_step = int(
                getattr(self.config, "aux_teacher_start_step", 0)
            )
            _aux_step_open = self.step >= _aux_start_step
            if (
                self.real_teacher_optimizer is not None
                and _aux_step_open
            ):
                rt_params_with_grad = [
                    p for p in self.real_teacher_optimizer.param_groups[0]["params"]
                    if p.grad is not None
                ]
                if rt_params_with_grad:
                    # Linear LR warmup over the first
                    # ``real_teacher_warmup_steps`` steps.
                    if (
                        self.real_teacher_warmup_steps > 0
                        and self.step < self.real_teacher_warmup_steps
                    ):
                        warm_factor = (
                            (self.step + 1) / self.real_teacher_warmup_steps
                        )
                        for pg in self.real_teacher_optimizer.param_groups:
                            pg["lr"] = self._real_teacher_base_lr * warm_factor
                    elif self.real_teacher_warmup_steps > 0:
                        # Restore base LR once warmup completes (no-op
                        # after first post-warmup step but harmless).
                        for pg in self.real_teacher_optimizer.param_groups:
                            pg["lr"] = self._real_teacher_base_lr
                    rtgn = torch.nn.utils.clip_grad_norm_(
                        rt_params_with_grad,
                        max_norm=self.real_teacher_max_grad_norm,
                    )
                    real_teacher_grad_norm_val = (
                        float(rtgn.item()) if torch.is_tensor(rtgn) else float(rtgn)
                    )
                    self.real_teacher_optimizer.step()
                    # Target-network EMA pull on the LoRA adapter
                    # (no-op when real_score_ema_weight == 0). Fires
                    # AFTER optim.step on every LoRA update so the
                    # EMA tracks the post-step weights, including
                    # the just-clipped gradient's effect.
                    self.model.ema_update_real_score_lora()
                self.real_teacher_optimizer.zero_grad(set_to_none=True)
            elif self.real_teacher_optimizer is not None:
                # Below start_step (or aux gate closed for any other
                # reason): zero any stray gradient so the next iter
                # starts clean. No optimizer step taken.
                self.real_teacher_optimizer.zero_grad(set_to_none=True)

            self.step += 1

            # Memory audit at steps 1 and 5 (steady-state activation
            # memory after the warm-up forward + grad path is fully
            # established). Gated by ``memory_audit_enabled`` so prod
            # logs don't get spammed.
            if (
                bool(getattr(cfg, "memory_audit_enabled", False))
                and self.step in (1, 5)
            ):
                self._dump_memory_audit(f"after_step_{self.step}")
                self._dump_step_mem_breakdown(f"step_{self.step}")

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
                # GAN diagnostics: surface the 3 most-watched signals
                # so terminal trace alone is enough to spot disc/critic
                # health (full set on wandb under train/*).
                #   d_real   = disc score on real (should rise > 0)
                #   d_fake   = disc score on fake-detached (D-update)
                #   c_corr   = critic↔disc value Pearson (should → 1)
                _d_real = generator_log_dict.get("train/r3gan_d_real")
                if _d_real is not None:
                    msg_parts.append(f"d_real={float(_d_real):+.3f}")
                _d_fake = generator_log_dict.get("train/r3gan_d_fake_detached")
                if _d_fake is not None:
                    msg_parts.append(f"d_fake={float(_d_fake):+.3f}")
                _c_corr = generator_log_dict.get("train/critic_disc_corr")
                if _c_corr is not None:
                    msg_parts.append(f"c_corr={float(_c_corr):+.3f}")
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
                # Peak GPU memory this step (resets the high-water
                # mark each iter). Useful for OOM-margin diagnostics.
                try:
                    if torch.cuda.is_available():
                        peak_gb = (
                            torch.cuda.max_memory_allocated() / (1024 ** 3)
                        )
                        msg_parts.append(f"peak_gb={peak_gb:.2f}")
                        torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass
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
                    # Online real_teacher (LoRA) gets the same treatment
                    # as the gen/critic optimizers: log the post-clip
                    # grad_norm and the warmup-aware effective LR. Both
                    # are always-defined locals (init'd to 0.0 / read
                    # off param_groups[0]["lr"] above) so the keys
                    # appear every step regardless of whether the
                    # optimizer fired this iter.
                    log_payload["critic/real_teacher_grad_norm"] = real_teacher_grad_norm_val
                    if self.real_teacher_optimizer is not None:
                        log_payload["critic/real_teacher_lr"] = float(
                            self.real_teacher_optimizer.param_groups[0]["lr"]
                        )
                    # EMA-LoRA drift diagnostic (None when
                    # real_score_ema_weight==0 or before first LoRA
                    # optim step). Picks up whatever the most recent
                    # EMA-update computed.
                    _ema_rel_l2 = getattr(
                        self.model, "_real_score_ema_rel_l2", None,
                    )
                    if _ema_rel_l2 is not None:
                        log_payload["critic/real_score_ema_rel_l2"] = float(_ema_rel_l2)
                    log_payload["critic/state_probe_grad_norm"] = (
                        state_probe_grad_norm_val
                    )
                    if self.state_probe_optimizer is not None:
                        log_payload["critic/state_probe_lr"] = float(
                            self.state_probe_optimizer.param_groups[0]["lr"]
                        )
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
                    # ``pred_real`` is the DMD-path teacher's x0 (frozen
                    # merged-v14 in dual-teacher mode; the LoRA in
                    # single-teacher mode). ``pred_real_lora`` is the
                    # aux-pass LoRA teacher's x0 — only present when
                    # dual-teacher is on (frozen pass + aux pass both
                    # populate the stash). Operators get a side-by-
                    # side comparison: the frozen oracle (DMD's
                    # gradient source) vs the moving-target online
                    # LoRA (aux pass's training target).
                    _aux_t = eval_latents.get("aux_teacher_timestep")
                    _aux_was_gt = eval_latents.get("aux_teacher_input_was_gt")
                    _suffix_aux = (
                        f"t={_aux_t} input={'GT' if _aux_was_gt else 'student'}"
                        if _aux_t is not None
                        else ""
                    )
                    for _key, _name, _cap in (
                        ("pred_real", "pred_real", _suffix_real or ""),
                        ("pred_real_lora", "pred_real_lora", _suffix_aux or ""),
                        ("pred_fake", "pred_fake", _suffix_real or ""),
                        ("clean_x_fake", "clean_x_fake", "fake_score conditioning"),
                        ("clean_x_real", "clean_x_real", f"real_score conditioning ({_ctx})"),
                    ):
                        _t_lat = eval_latents.get(_key)
                        if _t_lat is None or not torch.is_tensor(_t_lat):
                            continue
                        # Overlay enabled on ``clean_x_real`` only. The
                        # consolidated ``_draw_action_overlay`` draws
                        # both the action z-value bars (from
                        # ``clean_z_actions``) and the per-frame
                        # zarr_lat / motion-chunk text (from
                        # ``index_overlay``) in a single pass; both
                        # are gated on action_overlay being non-None,
                        # so passing the clean_z tensor turns the
                        # whole overlay on.
                        _clean_z_overlay = eval_latents.get(
                            "clean_z_actions"
                        )
                        _overlay = (
                            _clean_z_overlay
                            if _key == "clean_x_real"
                            and torch.is_tensor(_clean_z_overlay)
                            else None
                        )
                        # Per-frame zarr-latent + motion.npy chunk
                        # annotations on clean_x_real ONLY (other views
                        # are pred_*/clean_x_fake which don't trace back
                        # to a specific ride window). The model stashes
                        # the absolute zarr-latent index of clean_x_real's
                        # first frame plus the ride's motion-chunk offset
                        # in ``compute_generator_loss_streaming``.
                        _index_overlay = None
                        if _key == "clean_x_real":
                            _zarr_lo = eval_latents.get("clean_x_real_zarr_lat_lo")
                            _mco = eval_latents.get("clean_x_real_motion_chunk_offset")
                            _ovl_npb = eval_latents.get("clean_x_real_npb")
                            _zarr_name = eval_latents.get(
                                "clean_x_real_zarr_name", ""
                            )
                            if (
                                _zarr_lo is not None
                                and _mco is not None
                                and _ovl_npb is not None
                            ):
                                _index_overlay = (
                                    int(_zarr_lo),
                                    int(_mco),
                                    int(_ovl_npb),
                                    str(_zarr_name),
                                )
                        try:
                            self._log_pred_image_video(
                                _t_lat.to(torch.float32),
                                int(self.step),
                                name=_name,
                                caption_suffix=_cap,
                                action_overlay=_overlay,
                                index_overlay=_index_overlay,
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
        if not (
            self.action_critic_loss_active
            or self.gan_enabled
            or self.real_teacher_train_online
            or self.state_probe_aux_active
            or self.noise_aux_enabled
        ):
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
        if self.state_probe_aux_active:
            sp = self.model.state_probe
            sp_module = sp.module if isinstance(sp, DDP) else sp
            if sp_module is not None:
                state["state_probe"] = sp_module.state_dict()
                appended.append("state_probe")
            if self.state_probe_optimizer is not None:
                state["state_probe_optimizer"] = (
                    self.state_probe_optimizer.state_dict()
                )
                appended.append("state_probe_optimizer")
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
        if self.real_teacher_train_online:
            # FAIL-LOUD on save: silently dropping the LoRA state from
            # the checkpoint pairs with the resume path's silent
            # restart-from-warm-init to erase hours of online teacher
            # training without operator notice. Better to crash the
            # save and require investigation than to leave a
            # checkpoint that quietly discards progress on the next
            # resume. (action_critic / r3gan retain fail-soft because
            # they are bootstrapped from fixed checkpoints; only the
            # online-trained LoRA has this asymmetric risk profile.)
            try:
                from peft import get_peft_model_state_dict
                rs_inner = self.model.real_score.model
                if isinstance(rs_inner, DDP):
                    rs_inner = rs_inner.module
                state["real_teacher_lora"] = get_peft_model_state_dict(rs_inner)
                appended.append("real_teacher_lora")
            except Exception as exc:
                raise RuntimeError(
                    "[ActionForcing] real_teacher LoRA save failed: "
                    f"{exc!r}. Refusing to silently drop trained LoRA "
                    "state from the checkpoint — the next auto-resume "
                    "would rewind the teacher to v14 warm-start without "
                    "any operator-visible signal. Investigate the save "
                    "failure (most likely a peft version mismatch or a "
                    "DDP-wrap-shape issue on real_score.model) and rerun."
                ) from exc
            if self.real_teacher_optimizer is not None:
                state["real_teacher_optimizer"] = (
                    self.real_teacher_optimizer.state_dict()
                )
                appended.append("real_teacher_optimizer")
        if self.noise_aux_enabled and self.noise_aux_optimizer is not None:
            na = self.model.noise_aux
            na_module = na.module if isinstance(na, DDP) else na
            if na_module is not None:
                state["noise_aux"] = na_module.state_dict()
                appended.append("noise_aux")
            state["noise_aux_optimizer"] = self.noise_aux_optimizer.state_dict()
            appended.append("noise_aux_optimizer")
        if not appended:
            return
        torch.save(state, path)
        logging.info(
            "[ActionForcing] Appended state to %s: %s",
            path, ", ".join(appended),
        )

    def _maybe_resume(self) -> None:
        super()._maybe_resume()
        if not (
            self.action_critic_loss_active
            or self.gan_enabled
            or self.real_teacher_train_online
            or self.state_probe_aux_active
            or self.noise_aux_enabled
        ):
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
        if self.state_probe_aux_active:
            sp = self.model.state_probe
            sp_module = sp.module if isinstance(sp, DDP) else sp
            if sp_module is not None and "state_probe" in state:
                sp_missing, sp_unexpected = sp_module.load_state_dict(
                    state["state_probe"], strict=False,
                )
                if self.is_main_process:
                    logging.info(
                        "resume: state_probe missing=%d unexpected=%d",
                        len(sp_missing), len(sp_unexpected),
                    )
            if (
                self.state_probe_optimizer is not None
                and "state_probe_optimizer" in state
            ):
                try:
                    self.state_probe_optimizer.load_state_dict(
                        state["state_probe_optimizer"]
                    )
                    if self.is_main_process:
                        logging.info("resume: state_probe_optimizer state restored")
                except Exception as exc:
                    if self.is_main_process:
                        logging.warning(
                            "resume: state_probe_optimizer load failed: %s. "
                            "Starting state_probe optim from fresh state.", exc,
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
        if self.real_teacher_train_online:
            # FAIL-LOUD on resume: silently rolling the LoRA back to
            # v14 warm-start because the load errored — or because the
            # checkpoint just doesn't contain it — would erase every
            # step of online teacher training while the student
            # KEPT its optimizer state. That asymmetric rewind is
            # almost never what the operator wants. To force a fresh
            # teacher start, delete the checkpoint or set
            # auto_resume=False (which short-circuits this whole
            # block above).
            if "real_teacher_lora" not in state:
                raise RuntimeError(
                    "[ActionForcing] real_teacher_train_online=True but "
                    f"the resume checkpoint at {path} does not contain "
                    "'real_teacher_lora' state. This means either the "
                    "previous run wasn't training the teacher (expected: "
                    "the checkpoint predates this feature), or a save "
                    "failed silently in an earlier step. To proceed from "
                    "v14 warm-start instead of resuming, delete the "
                    "checkpoint or set auto_resume=False."
                )
            try:
                from peft import set_peft_model_state_dict
                rs_inner = self.model.real_score.model
                if isinstance(rs_inner, DDP):
                    rs_inner = rs_inner.module
                set_peft_model_state_dict(rs_inner, state["real_teacher_lora"])
                if self.is_main_process:
                    logging.info("resume: real_teacher LoRA state restored")
            except Exception as exc:
                raise RuntimeError(
                    "[ActionForcing] real_teacher LoRA load failed: "
                    f"{exc!r}. Refusing to silently restart from v14 "
                    "warm-start — would discard every step of online "
                    "teacher training. Investigate the checkpoint "
                    "(likely a peft / config mismatch with the "
                    "current LoRA setup) and rerun."
                ) from exc
            if self.real_teacher_optimizer is not None:
                if "real_teacher_optimizer" not in state:
                    raise RuntimeError(
                        "[ActionForcing] real_teacher_optimizer is built "
                        "but resume checkpoint lacks 'real_teacher_optimizer' "
                        "state. Same reasoning as the LoRA-state check above: "
                        "silently re-warming-up Adam moments + the LR "
                        "schedule on a checkpoint past the warmup phase "
                        "is almost never wanted."
                    )
                try:
                    self.real_teacher_optimizer.load_state_dict(
                        state["real_teacher_optimizer"]
                    )
                    if self.is_main_process:
                        logging.info(
                            "resume: real_teacher_optimizer state restored"
                        )
                except Exception as exc:
                    raise RuntimeError(
                        "[ActionForcing] real_teacher_optimizer load "
                        f"failed: {exc!r}. Refusing to silently start "
                        "from fresh optimizer state."
                    ) from exc
        if self.noise_aux_enabled:
            na = self.model.noise_aux
            na_module = na.module if isinstance(na, DDP) else na
            if na_module is not None and "noise_aux" in state:
                na_missing, na_unexpected = na_module.load_state_dict(
                    state["noise_aux"], strict=False,
                )
                if self.is_main_process:
                    logging.info(
                        "resume: noise_aux missing=%d unexpected=%d",
                        len(na_missing), len(na_unexpected),
                    )
            if (
                self.noise_aux_optimizer is not None
                and "noise_aux_optimizer" in state
            ):
                try:
                    self.noise_aux_optimizer.load_state_dict(
                        state["noise_aux_optimizer"]
                    )
                    if self.is_main_process:
                        logging.info("resume: noise_aux_optimizer state restored")
                except Exception as exc:
                    if self.is_main_process:
                        logging.warning(
                            "resume: noise_aux_optimizer load failed: %s. "
                            "Starting noise_aux optim from fresh state.", exc,
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
        # Cross-rank coordination: if ANY rank's teacher failed
        # (cotracker / VAE / ss_vae error on this rank's pred_x0), ALL
        # ranks must skip aux training together. Otherwise the failing
        # ranks early-return without firing the action_critic_ddp
        # forward (a DDP-wrapped collective) while the succeeding ranks
        # do — NCCL collective op count mismatches across ranks → hang.
        # This is the action-critic analog of the slide-loop's
        # any_crossed MAX-reduce.
        local_unavailable = 1 if teacher_z_8d is None else 0
        if dist.is_initialized() and dist.get_world_size() > 1:
            flag_t = torch.tensor(
                [local_unavailable], device=pred_x0.device, dtype=torch.long,
            )
            dist.all_reduce(flag_t, op=dist.ReduceOp.MAX)
            any_unavailable = bool(int(flag_t.item()))
        else:
            any_unavailable = bool(local_unavailable)
        if any_unavailable:
            # Skip aux loss on every rank this iter; teacher_unavailable
            # log fires only on the rank(s) that actually failed.
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
                "train/teacher_unavailable": float(local_unavailable),
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
    # LoRA-side (real_score) action losses: action_critic z-guidance on
    # the LoRA's denoised x0 + state_probe MSE on the per-chunk z the
    # probe reads from real_score's internal taps. Mirror of the
    # gen-side z-guidance formulation; SHARES the action_critic +
    # state_probe heads (Option B). Critic is treated as frozen during
    # this branch (its gradient already came from the gen-side inner
    # loop in ``_compute_action_critic_losses``); state_probe is
    # treated as trainable here (the probe gets its only gradient from
    # this loss, since gen-side currently has no probe loss).
    # ------------------------------------------------------------------
    def _compute_lora_action_losses(
        self,
        lora_x0: Optional[torch.Tensor],
        lora_state_preds: Optional[torch.Tensor],
        target_action_z: torch.Tensor,
        teacher_z_8d: torch.Tensor,
        chunk_t: torch.Tensor,
        current_step: int,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Dict[str, float],
    ]:
        """Returns ``(lora_action_loss, lora_state_probe_loss, logs)``.

        Each loss tensor is graph-bearing (grad flows back into LoRA
        params via the upstream forwards). ``None`` is returned for
        a loss when its required input is missing or its weight is 0.
        """
        logs: Dict[str, float] = {}
        zero = (
            torch.tensor(0.0, device=target_action_z.device,
                         dtype=target_action_z.dtype)
        )

        # Frozen-critic z-guidance on the LoRA's x0. Same warmup / ramp
        # contract as the gen-side z-guidance — pulls gen-side and
        # LoRA-side together so they activate at the same step. Read the
        # ramp config (``warmup_steps_for_guidance`` /
        # ``z_guidance_warmup_steps``) from existing trainer attrs.
        action_loss: Optional[torch.Tensor] = None
        if (
            lora_x0 is not None
            and self.action_critic_loss_active
            and self.lora_action_critic_z_guidance_weight > 0.0
        ):
            critic_for_guidance = (
                self.model.action_critic.module
                if isinstance(self.model.action_critic, DDP)
                else self.model.action_critic
            )
            warmup_start = self.warmup_steps_for_guidance
            if current_step < warmup_start:
                guidance_scale = 0.0
            elif self.z_guidance_warmup_steps > 0:
                ramp = min(
                    1.0,
                    (current_step - warmup_start)
                    / max(1, self.z_guidance_warmup_steps),
                )
                guidance_scale = ramp * self.lora_action_critic_z_guidance_weight
            else:
                guidance_scale = self.lora_action_critic_z_guidance_weight
            if guidance_scale > 0.0:
                critic_for_guidance.requires_grad_(False)
                try:
                    with torch.amp.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                        enabled=True,
                    ):
                        chunk_frames = int(self.config.num_frame_per_block)
                        n_chunks = lora_x0.shape[1] // chunk_frames
                        # Critic expects chunk-aligned actions
                        # ``[B, n_chunks, K]`` (mean over fpb frames per
                        # chunk), NOT the per-frame ``[B, F, K]`` stream.
                        # Mirror the gen-side prep in
                        # ``_compute_action_critic_losses``.
                        chunk_actions = _chunk_actions(
                            target_action_z, chunk_frames,
                        )[:, :n_chunks]
                        lora_pred_z = critic_for_guidance(
                            lora_x0, chunk_t, chunk_actions,
                        )
                        lora_pred_z = lora_pred_z[:, :n_chunks]
                        lora_z2z7 = lora_pred_z[:, :, self.action_critic_dims]
                        lora_z_mse = F.mse_loss(
                            lora_z2z7, chunk_actions.to(lora_z2z7.dtype),
                        )
                        action_loss = guidance_scale * lora_z_mse
                finally:
                    critic_for_guidance.requires_grad_(True)
                logs["train/lora_critic_z_loss"] = float(lora_z_mse.detach().item())
                logs["train/lora_action_loss"] = float(action_loss.detach().item())
                logs["train/lora_z_guidance_scale"] = float(guidance_scale)

        # State-probe MSE supervision on the LoRA-side per-chunk z. Target
        # is the SAME ``teacher_z_8d`` the gen-side critic trained
        # against (= cotracker+ss_vae extracted from the student's pred).
        # Loss flows to LoRA params via real_score's internal taps and
        # to state_probe params directly. ``state_probe`` is configured
        # as TRAINABLE in this branch (built via DDP at trainer init).
        # Diagnostic logging: surface whether the wrapper actually
        # produced state_preds this iter so a missing state_probe loss
        # is debuggable from wandb alone.
        logs["train/lora_state_preds_present"] = (
            1.0 if lora_state_preds is not None else 0.0
        )
        state_probe_loss: Optional[torch.Tensor] = None
        if (
            lora_state_preds is not None
            and self.state_probe_z_loss_weight > 0.0
        ):
            n_chunks = lora_state_preds.shape[1]
            tz = teacher_z_8d[:, :n_chunks].to(lora_state_preds.dtype)
            sp_mse = F.mse_loss(lora_state_preds, tz)
            state_probe_loss = self.state_probe_z_loss_weight * sp_mse
            logs["train/lora_state_probe_z_loss"] = float(sp_mse.detach().item())
            logs["train/lora_state_probe_loss"] = float(
                state_probe_loss.detach().item()
            )
            logs["train/state_probe_z_loss_weight"] = float(
                self.state_probe_z_loss_weight
            )

        return action_loss, state_probe_loss, logs

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
    def _sample_critic_grad_frame_indices(
        self, F: int, k: int, device: torch.device,
    ) -> List[int]:
        """Pick ``k`` distinct frame indices in [0, F) for the
        gradient-distillation subset, broadcast from rank 0 so all
        DDP ranks pick the same subset (otherwise the per-rank
        gradient targets would differ and the all-reduce on the
        critic backward would average inconsistent gradients).
        """
        k = max(1, min(int(k), int(F)))
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            idx = torch.randperm(F, device=device)[:k].sort().values
        else:
            idx = torch.empty(k, dtype=torch.long, device=device)
        if dist.is_initialized():
            dist.broadcast(idx, src=0)
        return idx.tolist()

    def _get_maniqa_target_model(self) -> Optional["torch.nn.Module"]:
        """Lazy-build a pyiqa MANIQA NR-IQA metric for use as a frozen
        no_grad target. Forward signature: ``model(x) -> [B, 1]``
        scalar quality score in [0, 1] (higher = better) per image.
        Input ``x`` ``[B, 3, H, W]`` in [0, 1].

        We force ``test_sample=1`` on the inner net so MANIQA returns
        the score from ONE crop. With our pre-cropped 224×224 input,
        the crop is an identity (uniform_crop with crop_num=1 picks
        offset (0, 0) on a 224×224 image = pass-through). This gives
        us a per-patch score we can map back to specific latent token
        regions, which is the whole point of patch-sampled NR-IQA.
        """
        cached = getattr(self, "_maniqa_target_model", None)
        if cached is not None:
            return cached
        try:
            import pyiqa as _pyiqa
        except ImportError as e:
            raise RuntimeError(
                "maniqa_approx_loss_weight>0 requires pyiqa: pip "
                f"install pyiqa. Original error: {e}"
            )
        metric_name = self.maniqa_metric_name
        model = _pyiqa.create_metric(
            metric_name, as_loss=False, device=self.device,
        )
        # Force single-crop forward; default test_sample=20 averages
        # 20 uniform crops which (a) is wasteful when our input is
        # already 224×224 and (b) destroys the per-patch granularity
        # we need for spatial token mapping.
        try:
            model.net.test_sample = 1
        except Exception:
            pass
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._maniqa_target_model = model
        if self.is_main_process:
            n_params = sum(p.numel() for p in model.parameters())
            logging.info(
                "[ActionForcing] MANIQA-target lazy-built: metric=%s "
                "params=%.2fM (used for no_grad target only; "
                "test_sample forced to 1)",
                metric_name, n_params / 1e6,
            )
        return model

    def _compute_dense_maniqa_target(
        self,
        gen_pix: torch.Tensor,
        F_lat: int,
        target_h: int,
        target_w: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], float]:
        """Build a sparse per-token MANIQA quality target.

        Args:
            gen_pix: ``[B, F_pix, 3, H_pix, W_pix]`` in [-1, 1] (VAE
                output range). F_pix = pix_per_lat * F_lat.
            F_lat: latent temporal length.
            target_h, target_w: approx-output token grid spatial dims.

        Returns:
            ``(target, valid_mask, target_build_ms)`` where:
              * ``target`` ``[B, F_lat, target_h, target_w]`` float32:
                MANIQA score in [0, 1] for tokens covered by a
                sampled patch this iter; NaN elsewhere.
              * ``valid_mask`` ``[B, F_lat, target_h, target_w]``
                bool: True where target is valid (not NaN).
              * ``target_build_ms``: wall-clock for the target build
                (for the 4-vs-84 frame timing diagnostic).

        Returns ``(None, None, 0.0)`` when MANIQA approx weight is 0.

        Patch sampling: per iter, picks ``maniqa_n_frames`` random
        latent frames (DDP-synced), and per selected frame, samples
        ``maniqa_n_patches_per_frame`` random 224×224 patches at
        DDP-synced offsets. Each patch gets one MANIQA score, and the
        score is broadcast to all approx-tokens whose pixel coverage
        overlaps the patch.
        """
        if self.maniqa_approx_loss_weight <= 0:
            return None, None, 0.0
        if (
            self.maniqa_approx is None
            or self.maniqa_approx_optimizer is None
        ):
            return None, None, 0.0

        import time as _time
        t0 = _time.time()

        device = gen_pix.device
        B, F_pix, C, H_pix, W_pix = gen_pix.shape
        if F_pix % F_lat != 0:
            raise RuntimeError(
                f"_compute_dense_maniqa_target: F_pix={F_pix} not "
                f"divisible by F_lat={F_lat}."
            )
        pix_per_lat = F_pix // F_lat
        crop = 224
        if H_pix < crop or W_pix < crop:
            raise RuntimeError(
                f"_compute_dense_maniqa_target: pixel resolution "
                f"({H_pix}, {W_pix}) smaller than MANIQA crop {crop}."
            )

        # Pick latent-frame indices (DDP-synced).
        n_frames = max(1, min(int(self.maniqa_n_frames), int(F_lat)))
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            frame_idx = torch.randperm(
                F_lat, device=device,
            )[:n_frames].sort().values
        else:
            frame_idx = torch.empty(
                n_frames, dtype=torch.long, device=device,
            )
        if dist.is_initialized():
            dist.broadcast(frame_idx, src=0)
        frame_list = frame_idx.tolist()

        # Pick a representative pixel-frame per selected latent — the
        # CENTER pixel of the 4-frame group (index pix_per_lat // 2 of
        # each group). Avoids averaging across pixel-frames (which
        # would smear motion artefacts) while still being representative.
        pix_center_offset = pix_per_lat // 2
        pix_indices = [
            f * pix_per_lat + pix_center_offset for f in frame_list
        ]
        # Patch offsets — DETERMINISTIC ORDERED tiling that covers the
        # whole image. Random placement was previously used here; it's
        # been removed because random patches sample the same regions
        # repeatedly across iters (and miss other regions entirely),
        # producing a more variable / less complete approx target.
        # Ordered tiling guarantees that every iter the approximator
        # sees the SAME spatial coverage, so its dense per-token
        # supervision is uniform.
        #
        # Layout: factor n_patches into (rows, cols) closest to the
        # image's aspect ratio (H_pix / W_pix). For our 480×832 input
        # at n_patches=8, that picks (2, 4) — 2 rows × 4 cols. Each
        # axis uses even spacing from offset=0 to offset=L-P with
        # ``stride = (L-P)/(n-1)``, so the corner patches sit exactly
        # at (0, 0) and (H-P, W-P).
        # Coverage at 480×832 / n=8 / patch=224:
        #   * rows: 2 patches at h=0, h=256 cover 0..223 ∪ 256..479
        #     — a 32-row gap at 224..255 (rows alone = 93.3%).
        #   * cols: 4 patches at w=0, 203, 405, 608 overlap into full
        #     coverage of 0..831 (100% width).
        # Net area coverage ≈ 93.3% — matches the "~90%" target.
        n_patches = max(1, int(self.maniqa_n_patches_per_frame))

        def _factor_grid(n: int, h: int, w: int):
            target = float(h) / float(w)
            best_rc = (1, n)
            best_diff = abs((1.0 / n) - target)
            for r in range(1, n + 1):
                if n % r == 0:
                    c = n // r
                    diff = abs((r / c) - target)
                    if diff < best_diff:
                        best_rc, best_diff = (r, c), diff
            return best_rc

        def _tile_positions(L: int, P: int, n: int):
            if n <= 1:
                return [(L - P) // 2]
            stride = (L - P) / (n - 1)
            return [int(round(i * stride)) for i in range(n)]

        rows, cols = _factor_grid(n_patches, H_pix, W_pix)
        h_pos = _tile_positions(H_pix, crop, rows)
        w_pos = _tile_positions(W_pix, crop, cols)
        # Build per-(frame, patch) (h_o, w_o) lists; same offsets for
        # every selected frame and every batch sample. Deterministic →
        # no DDP broadcast needed.
        h_offsets: list = []
        w_offsets: list = []
        for _ in range(n_frames):
            for r in range(rows):
                for c in range(cols):
                    h_offsets.append(h_pos[r])
                    w_offsets.append(w_pos[c])
        n_total_patches = len(h_offsets)

        # Crop all patches into a single batch for MANIQA forward.
        # Shape: [B * n_total_patches, 3, 224, 224].
        # Convert [-1, 1] → [0, 1] (clamp first).
        patches = []
        patch_meta = []  # (frame_in_F_lat, h_o, w_o)
        for k_p, f_lat_idx in enumerate(frame_list):
            pix_f = pix_indices[k_p]
            for q in range(n_patches):
                p_idx = k_p * n_patches + q
                h_o = h_offsets[p_idx]
                w_o = w_offsets[p_idx]
                # All B samples at this frame, this patch position.
                pix_slice = gen_pix[
                    :, pix_f, :, h_o:h_o + crop, w_o:w_o + crop,
                ]
                patches.append(pix_slice)  # [B, 3, 224, 224]
                patch_meta.append((f_lat_idx, h_o, w_o))
        # Stack: [n_total_patches, B, 3, 224, 224] → [B, n_total, ...]
        patches_t = torch.stack(patches, dim=1).contiguous()
        patches_t = patches_t.view(
            B * n_total_patches, 3, crop, crop,
        )
        patches_t = (patches_t.clamp(-1.0, 1.0) * 0.5 + 0.5).float()

        # MANIQA forward (no_grad).
        model = self._get_maniqa_target_model()
        with torch.no_grad():
            scores = model(patches_t).float()  # [B * n_total, 1]
        scores = scores.view(B, n_total_patches)

        # Build sparse target. NaN-init, fill the patch-covered tokens.
        target = torch.full(
            (B, F_lat, target_h, target_w),
            float("nan"), device=device, dtype=torch.float32,
        )
        pix_per_token_h = H_pix / float(target_h)
        pix_per_token_w = W_pix / float(target_w)
        for q_idx, (f_lat_idx, h_o, w_o) in enumerate(patch_meta):
            t_row_lo = int(max(0, h_o // pix_per_token_h))
            t_row_hi = int(min(
                target_h,
                int((h_o + crop - 1) // pix_per_token_h) + 1,
            ))
            t_col_lo = int(max(0, w_o // pix_per_token_w))
            t_col_hi = int(min(
                target_w,
                int((w_o + crop - 1) // pix_per_token_w) + 1,
            ))
            score_q = scores[:, q_idx]  # [B]
            target[
                :, f_lat_idx, t_row_lo:t_row_hi, t_col_lo:t_col_hi,
            ] = score_q.view(B, 1, 1)

        valid_mask = ~torch.isnan(target)
        # Replace NaN in target with 0 so the approx training step
        # can do a masked-MSE without NaN propagation. Actual target
        # values at masked-out positions are ignored via valid_mask.
        target = torch.where(
            valid_mask, target, torch.zeros_like(target),
        )
        target_build_ms = (_time.time() - t0) * 1000.0
        return target, valid_mask, target_build_ms

    def _get_lpips_target_model(self) -> Optional["torch.nn.Module"]:
        """Lazy-build the LPIPS model used to compute DENSE TARGETS for
        the LPIPS approx (no_grad, never trained). Spatial=True returns
        per-position distance maps so we can mean-pool to a per-token
        target grid for the approx.

        Separate from ``_get_lpips_model`` (which would have been used
        for direct LPIPS-as-loss; that path was abandoned due to OOM
        from VAE backprop). This one always sets ``spatial=True``.
        """
        cached = getattr(self, "_lpips_target_model", None)
        if cached is not None:
            return cached
        try:
            import lpips as _lpips
        except ImportError as e:
            raise RuntimeError(
                "lpips_approx_loss_weight>0 requires the 'lpips' "
                f"package: pip install lpips. Original error: {e}"
            )
        model = _lpips.LPIPS(
            net="vgg", verbose=False, spatial=True,
        )
        model = model.to(device=self.device, dtype=torch.float32)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._lpips_target_model = model
        if self.is_main_process:
            n_params = sum(p.numel() for p in model.parameters())
            logging.info(
                "[ActionForcing] LPIPS-target(vgg, spatial) lazy-built: "
                "params=%.2fM (used for no_grad target only)",
                n_params / 1e6,
            )
        return model

    def _compute_dense_perceptual_targets(
        self,
        gen_pix: torch.Tensor,
        gt_pix: torch.Tensor,
        F_lat: int,
        target_h: int,
        target_w: int,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Build per-token MSE-or-MSE+MAE / LPIPS / MS-SSIM dense
        targets from VAE-decoded pixel pairs.

        Args:
            gen_pix / gt_pix: ``[B, F_pix, 3, H_pix, W_pix]`` in [-1, 1].
                F_pix = 4 * F_lat (WAN VAE temporal expansion).
            F_lat: latent temporal length (target frame axis).
            target_h, target_w: token grid spatial dims (=8 by default,
                matching the approx's post-stem output).

        Returns:
            ``(target_mse, target_lpips, target_msssim)`` each
            ``[B, F_lat, target_h, target_w]``. Any may be None when
            its weight is 0 (skip compute).

        ``target_mse`` is actually a combined ``l2_w * (gen-gt)² +
        l1_w * |gen-gt|`` per-token field — both terms are per-pixel
        and granular (mean-pooled to the latent token grid). Set
        ``mse_approx_l1_weight=0`` for legacy pure-MSE.

        ``target_msssim`` is per-frame MS-SSIM (computed on each
        pixel frame, then averaged across the 4 pixel-frames-per-
        latent) BROADCAST across all (target_h * target_w) tokens.
        Per-frame scalar supervision but dense approx output, so
        the approx learns its own spatial pattern.
        """
        B, F_pix, C, H_pix, W_pix = gen_pix.shape
        if F_pix % F_lat != 0:
            raise RuntimeError(
                f"_compute_dense_perceptual_targets: F_pix={F_pix} "
                f"not divisible by F_lat={F_lat}."
            )
        pix_per_lat = F_pix // F_lat
        target_mse: Optional[torch.Tensor] = None
        target_lpips: Optional[torch.Tensor] = None
        target_msssim: Optional[torch.Tensor] = None

        with torch.no_grad():
            if self.mse_approx_loss_weight > 0:
                # Combined L2 + L1 per-token target.
                # Per-pixel l2 = (gen - gt)², l1 = |gen - gt|.
                diff = gen_pix - gt_pix  # [B, F_pix, 3, H, W]
                l2_w = float(self.mse_approx_l2_weight)
                l1_w = float(self.mse_approx_l1_weight)
                combined = l2_w * (diff ** 2)
                if l1_w > 0:
                    combined = combined + l1_w * diff.abs()
                del diff
                # Reshape to [B, F_lat, pix_per_lat, 3, H, W] then mean
                # over (pix_per_lat, 3) → [B, F_lat, H_pix, W_pix].
                combined = combined.view(
                    B, F_lat, pix_per_lat, C, H_pix, W_pix,
                )
                combined = combined.mean(dim=(2, 3))
                # Spatial mean-pool to token grid.
                combined = combined.reshape(
                    B * F_lat, 1, H_pix, W_pix,
                )
                target_mse = torch.nn.functional.adaptive_avg_pool2d(
                    combined, (target_h, target_w),
                ).reshape(B, F_lat, target_h, target_w).float()
                del combined
            if self.lpips_approx_loss_weight > 0:
                lpips_model = self._get_lpips_target_model()
                # LPIPS expects [N, 3, H, W] in [-1, 1]. Forward on
                # all 84 frames at 480×832 blows ~8 GB activation
                # workspace. Process in small chunks (4 frames each)
                # and accumulate the spatial-pooled distance.
                gen_flat = gen_pix.reshape(B * F_pix, C, H_pix, W_pix)
                gt_flat = gt_pix.reshape(B * F_pix, C, H_pix, W_pix)
                chunk = 4
                pooled_chunks = []
                for s in range(0, gen_flat.shape[0], chunk):
                    e = min(s + chunk, gen_flat.shape[0])
                    d_chunk = lpips_model(
                        gen_flat[s:e], gt_flat[s:e],
                    ).float()  # [chunk, 1, h_d, w_d]
                    d_chunk = torch.nn.functional.adaptive_avg_pool2d(
                        d_chunk, (target_h, target_w),
                    )
                    pooled_chunks.append(d_chunk)
                dist = torch.cat(pooled_chunks, dim=0)
                dist = dist.reshape(
                    B, F_lat, pix_per_lat, target_h, target_w,
                )
                target_lpips = dist.mean(dim=2).float()
            if self.msssim_approx_loss_weight > 0:
                # MS-SSIM expects [N, C, H, W] in non-negative range
                # (default data_range=1.0). Our pixels are in
                # [-1, 1] so shift to [0, 1] and use data_range=1.
                # Per-frame scalar (size_average=False), then average
                # over the 4 pixel-frames per latent → [B, F_lat].
                from pytorch_msssim import ms_ssim as _ms_ssim_fn
                gen_norm = (
                    gen_pix.reshape(
                        B * F_pix, C, H_pix, W_pix,
                    ).float().clamp(-1, 1) + 1.0
                ) * 0.5
                gt_norm = (
                    gt_pix.reshape(
                        B * F_pix, C, H_pix, W_pix,
                    ).float().clamp(-1, 1) + 1.0
                ) * 0.5
                # Edge-domain switch: when ``msssim_target_domain ==
                # 'edges'`` the MS-SSIM input is the per-pixel Sobel
                # gradient magnitude rather than the raw image. Defeats
                # the gray-collapse failure mode where standard MS-SSIM
                # stays high on a uniformly low-contrast image (the
                # "gray attack": local-window contrast statistics still
                # match the GT's locally-low-contrast statistics, even
                # though the image has lost all detail). On gradient
                # maps, gray pred ⇒ near-zero edges; GT ⇒ rich edges;
                # the SSIM crashes to ~0 and the gen receives a strong
                # push to recover detail. Output is renormalised to
                # [0, 1] and clamped so MS-SSIM's data_range=1.0
                # contract is preserved.
                if self.msssim_target_domain == "edges":
                    # 3×3 Sobel kernels, applied per-channel via grouped
                    # conv so the gradient stays per-channel (no cross-
                    # channel coupling). Reflection pad keeps shape.
                    sobel_kx = torch.tensor(
                        [[1.0, 0.0, -1.0],
                         [2.0, 0.0, -2.0],
                         [1.0, 0.0, -1.0]],
                        dtype=gen_norm.dtype, device=gen_norm.device,
                    ).view(1, 1, 3, 3).expand(C, 1, 3, 3).contiguous()
                    sobel_ky = torch.tensor(
                        [[1.0, 2.0, 1.0],
                         [0.0, 0.0, 0.0],
                         [-1.0, -2.0, -1.0]],
                        dtype=gen_norm.dtype, device=gen_norm.device,
                    ).view(1, 1, 3, 3).expand(C, 1, 3, 3).contiguous()

                    def _grad_mag(x: torch.Tensor) -> torch.Tensor:
                        x_pad = F.pad(x, (1, 1, 1, 1), mode="reflect")
                        gx = F.conv2d(x_pad, sobel_kx, groups=C)
                        gy = F.conv2d(x_pad, sobel_ky, groups=C)
                        # Magnitude in [0, ~4*sqrt(2)*max_pixel] before
                        # normalisation. Sobel max-response on a step
                        # edge from 0→1 is 4 per kernel, so |grad| ∈
                        # [0, ~5.66]. Divide by 5.66 to roughly land in
                        # [0, 1], then clamp for data_range=1.0.
                        mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
                        return (mag / 5.66).clamp(0.0, 1.0)

                    gen_norm = _grad_mag(gen_norm)
                    gt_norm = _grad_mag(gt_norm)
                # Chunk to bound memory (MS-SSIM does multi-scale
                # Gaussian filtering — ~1-2 GB transient per 4-frame
                # batch at 480x832).
                chunk = 4
                msssim_chunks = []
                for s in range(0, gen_norm.shape[0], chunk):
                    e = min(s + chunk, gen_norm.shape[0])
                    msssim_chunks.append(
                        _ms_ssim_fn(
                            gen_norm[s:e],
                            gt_norm[s:e],
                            data_range=1.0,
                            size_average=False,
                        ).float()
                    )
                msssim_per_pix_frame = torch.cat(msssim_chunks, dim=0)
                # [B*F_pix] → [B, F_lat] (mean over pix_per_lat).
                msssim_per_lat = msssim_per_pix_frame.view(
                    B, F_lat, pix_per_lat,
                ).mean(dim=2)
                # Broadcast to dense token grid.
                target_msssim = (
                    msssim_per_lat
                    .unsqueeze(-1).unsqueeze(-1)
                    .expand(B, F_lat, target_h, target_w)
                    .contiguous()
                    .float()
                )
                del gen_norm, gt_norm, msssim_per_pix_frame
        return target_mse, target_lpips, target_msssim

    def _run_perceptual_approx_update(
        self,
        gen_lat: torch.Tensor,
        gt_lat: torch.Tensor,
        target_mse: Optional[torch.Tensor],
        target_lpips: Optional[torch.Tensor],
        target_msssim: Optional[torch.Tensor] = None,
        target_maniqa: Optional[torch.Tensor] = None,
        target_maniqa_mask: Optional[torch.Tensor] = None,
        interp_data: Optional[List[Dict[str, torch.Tensor]]] = None,
        out: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Train all dense per-token approxes against their targets.

        All approxes operate on ``(input_lat, gt_lat)`` (detached) and
        produce ``[B, F_lat, target_h, target_w]`` predictions. Loss is
        per-token MSE + scalar mean-alignment. Mutates ``out`` in place.

        ``gan_d_approx`` is trained on TWO forward passes per iter:
        ``(gen_lat, gt_lat) → target_disc_fake`` and
        ``(gt_lat, gt_lat) → target_disc_real``. The two-sided training
        forces the approx to USE its gt_lat input (otherwise the same
        input gives two different targets, contradicting). This
        anchors the gt-conditioning so the approx learns "how the disc
        would score this CANDIDATE given this REFERENCE", which is
        what the gen actually wants in Path 3.
        """
        if out is None:
            out = {}
        gen_lat_d = gen_lat.detach()
        gt_lat_d = gt_lat.detach()

        # Paired-input approxes (mse / lpips / msssim). Each takes
        # ``(candidate=gen_lat, reference=gt_lat)`` and predicts the
        # per-token metric value. The reference is essential — these
        # metrics are inherently comparative; without it the score
        # is unanchored.
        for (
            name, model_ddp, model, optim, target,
            interp_key, weight,
        ) in [
            (
                "mse_approx",
                self.mse_approx_ddp,
                self.mse_approx,
                self.mse_approx_optimizer,
                target_mse,
                "mse",
                self.mse_approx_loss_weight,
            ),
            (
                "lpips_approx",
                self.lpips_approx_ddp,
                self.lpips_approx,
                self.lpips_approx_optimizer,
                target_lpips,
                "lpips",
                self.lpips_approx_loss_weight,
            ),
            (
                "msssim_approx",
                self.msssim_approx_ddp,
                self.msssim_approx,
                self.msssim_approx_optimizer,
                target_msssim,
                "msssim",
                self.msssim_approx_loss_weight,
            ),
        ]:
            if model is None or optim is None or target is None or weight <= 0:
                continue
            optim.zero_grad(set_to_none=True)
            model_for_update = model_ddp if model_ddp is not None else model
            pred = model_for_update(gen_lat_d, gt_lat_d).float()
            # Dense per-token loss on the original (gen, gt) anchor.
            L_dense = ((pred - target) ** 2).mean()
            L_mean_align = (pred.mean() - target.mean()) ** 2
            L_total = (
                L_dense
                + self.perceptual_approx_mean_align_weight * L_mean_align
            )
            # v11-style multi-noise: extra anchor points along the
            # (fake → real) line. Each contributes a per-token MSE
            # term to the same training loss. Densifies the value
            # field so the approx's gradient (what gen consumes) is
            # meaningful by Lipschitz interpolation across anchors.
            L_interp_value = 0.0
            if interp_data is not None and len(interp_data) > 0:
                interp_terms = []
                for d in interp_data:
                    interp_target = d.get(interp_key)
                    if interp_target is None:
                        continue
                    interp_pred = model_for_update(
                        d["lat"], gt_lat_d,
                    ).float()
                    interp_terms.append(
                        ((interp_pred - interp_target) ** 2).mean()
                    )
                if interp_terms:
                    L_interp = sum(interp_terms) / len(interp_terms)
                    L_total = L_total + L_interp
                    L_interp_value = float(L_interp.detach().item())
            L_total.backward()
            if (
                self.gan_max_grad_norm is not None
                and self.gan_max_grad_norm > 0
            ):
                params_iter = (
                    model_ddp.parameters() if model_ddp is not None
                    else model.parameters()
                )
                torch.nn.utils.clip_grad_norm_(
                    [p for p in params_iter if p.grad is not None],
                    self.gan_max_grad_norm,
                )
            optim.step()
            out[f"train/{name}_dense_loss"] = float(L_dense.detach().item())
            out[f"train/{name}_mean_align_loss"] = float(
                L_mean_align.detach().item()
            )
            out[f"train/{name}_pred_mean"] = float(pred.detach().mean().item())
            out[f"train/{name}_target_mean"] = float(target.mean().item())
            out[f"train/{name}_interp_loss"] = L_interp_value

        # ----- maniqa_approx training -----
        # Single-input (NR-IQA, no GT). The target is sparse — only
        # tokens covered by a sampled MANIQA patch this iter have a
        # valid target; others are NaN-masked via ``target_maniqa_mask``.
        # Loss = masked-MSE + masked-mean-align (over valid positions
        # only). The mean-align term anchors the approx's average
        # prediction to the sampled patches' mean quality so the model
        # learns the correct absolute scale even though dense MSE only
        # covers a fraction of tokens per iter.
        if (
            self.maniqa_approx is not None
            and self.maniqa_approx_optimizer is not None
            and target_maniqa is not None
            and target_maniqa_mask is not None
            and self.maniqa_approx_loss_weight > 0
        ):
            optim = self.maniqa_approx_optimizer
            optim.zero_grad(set_to_none=True)
            model_for_update = (
                self.maniqa_approx_ddp if self.maniqa_approx_ddp is not None
                else self.maniqa_approx
            )
            pred = model_for_update(gen_lat_d).float()
            mask_f = target_maniqa_mask.float()
            mask_sum = mask_f.sum().clamp_min(1.0)
            sq = (pred - target_maniqa) ** 2 * mask_f
            L_dense = sq.sum() / mask_sum
            # Masked mean-align: align mean over VALID tokens only.
            pred_masked_mean = (
                (pred * mask_f).sum() / mask_sum
            )
            target_masked_mean = (
                (target_maniqa * mask_f).sum() / mask_sum
            )
            L_mean_align = (pred_masked_mean - target_masked_mean) ** 2
            L_total = (
                L_dense
                + self.perceptual_approx_mean_align_weight * L_mean_align
            )
            L_total.backward()
            if (
                self.gan_max_grad_norm is not None
                and self.gan_max_grad_norm > 0
            ):
                params_iter = (
                    self.maniqa_approx_ddp.parameters()
                    if self.maniqa_approx_ddp is not None
                    else self.maniqa_approx.parameters()
                )
                torch.nn.utils.clip_grad_norm_(
                    [p for p in params_iter if p.grad is not None],
                    self.gan_max_grad_norm,
                )
            optim.step()
            out["train/maniqa_approx_dense_loss"] = float(
                L_dense.detach().item()
            )
            out["train/maniqa_approx_mean_align_loss"] = float(
                L_mean_align.detach().item()
            )
            out["train/maniqa_approx_pred_mean"] = float(
                pred_masked_mean.detach().item()
            )
            out["train/maniqa_approx_target_mean"] = float(
                target_masked_mean.detach().item()
            )
            out["train/maniqa_approx_n_valid_tokens"] = float(
                mask_sum.item()
            )

        # ----- gan_d_approx training -----
        # gan_d_approx training was here (paired PerceptualApprox-based
        # disc approximator). REMOVED — replaced by the existing
        # LatentSAM2Critic (single-input) whose distillation runs in
        # ``_run_distilled_disc_critic_update`` Path 2 with multi-noise
        # via ``gan_critic_n_interp_samples``. ``self.latent_critic``
        # IS the gan_d_approx now.

    def _compute_gt_latent_recon_loss(
        self,
        pred_image: torch.Tensor,
        gt_latents_window: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Direct latent-space MAE+MSE between pred_image and GT.

        Gated by ``gt_latent_mae_loss_weight`` and
        ``gt_latent_mse_loss_weight``. Both apply at constant weight
        every step — no warmup, no ramp. Returns ``(loss, logs)``;
        ``loss`` is a graph-attached scalar to add to ``generator_loss``.
        Zero scalar + empty logs when both weights are 0.
        """
        device = pred_image.device
        zero = torch.zeros((), device=device, dtype=torch.float32)
        w_mae = self.gt_latent_mae_loss_weight
        w_mse = self.gt_latent_mse_loss_weight
        if w_mae <= 0.0 and w_mse <= 0.0:
            return zero, {}
        gt_det = gt_latents_window.detach().to(pred_image.dtype)
        diff = pred_image - gt_det
        loss = zero
        logs: Dict[str, float] = {}
        if w_mae > 0.0:
            mae_raw = diff.abs().mean()
            loss = loss + w_mae * mae_raw.to(pred_image.dtype)
            logs["train/gt_latent_mae_raw"] = float(mae_raw.detach().item())
            logs["train/gt_latent_mae_weighted"] = float(
                (w_mae * mae_raw).detach().item()
            )
        if w_mse > 0.0:
            mse_raw = diff.pow(2).mean()
            loss = loss + w_mse * mse_raw.to(pred_image.dtype)
            logs["train/gt_latent_mse_raw"] = float(mse_raw.detach().item())
            logs["train/gt_latent_mse_weighted"] = float(
                (w_mse * mse_raw).detach().item()
            )
        return loss, logs

    def _compute_gen_side_perceptual_loss(
        self,
        pred_image: torch.Tensor,
        gt_latents_window: torch.Tensor,
        current_step: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Gen-side perceptual loss via the trained approxes.

        ``pred_image``: graph-on gen latent ``[B, F, C, H, W]``.
        ``gt_latents_window``: GT latent (detached).

        Returns ``(loss, logs)`` where ``loss`` is a graph-attached
        scalar to add to gen_loss; backward flows through the small
        approxes (no VAE) into the gen.

        Gated by warmup and per-approx weights. Returns zero scalar
        + empty logs when nothing fires.
        """
        device = pred_image.device
        zero = torch.zeros((), device=device, dtype=torch.float32)
        if current_step < self.perceptual_approx_warmup_steps:
            return zero, {}
        # Linear ramp factor after the warmup gate opens. Multiplies
        # every per-approx weight uniformly so MANIQA/LPIPS/MS-SSIM/MSE
        # all share the same on-ramp shape. ramp_steps == 0 preserves
        # the legacy hard-switch behaviour.
        if self.perceptual_approx_ramp_steps > 0:
            ramp_in = current_step - self.perceptual_approx_warmup_steps
            ramp_factor = min(
                1.0,
                max(0.0, float(ramp_in) / float(self.perceptual_approx_ramp_steps)),
            )
        else:
            ramp_factor = 1.0
        loss = zero
        logs: Dict[str, float] = {
            "train/perc_ramp_factor": float(ramp_factor),
        }
        gt_lat_det = gt_latents_window.detach().to(pred_image.dtype)
        # ``sign``: +1 for distance approxes (gen wants pred LOW); -1
        # for similarity / GAN approxes (gen wants pred HIGH).
        # MS-SSIM is in [0, 1] with 1 = identical → sign = -1 so the
        # gen MAXIMIZES it.
        for name, model, weight, sign, single_input in [
            (
                "mse_approx",
                self.mse_approx,
                self.mse_approx_loss_weight,
                +1.0,
                False,
            ),
            (
                "lpips_approx",
                self.lpips_approx,
                self.lpips_approx_loss_weight,
                +1.0,
                False,
            ),
            (
                "msssim_approx",
                self.msssim_approx,
                self.msssim_approx_loss_weight,
                -1.0,
                False,
            ),
            (
                "maniqa_approx",
                self.maniqa_approx,
                self.maniqa_approx_loss_weight,
                -1.0,  # gen wants MANIQA quality HIGH
                True,  # NR-IQA → no GT input
            ),
            # NOTE: gan_d_approx (PerceptualApprox-based) entry removed.
            # The disc-side gradient is delivered to gen via
            # ``self.latent_critic`` (LatentSAM2Critic) in the existing
            # Path 3 of ``_compute_r3gan_losses_distilled``.
        ]:
            if model is None or weight <= 0:
                continue
            effective_weight = weight * ramp_factor
            # Skip the forward entirely if the ramp factor has zeroed
            # the contribution (saves the approx forward+backward when
            # the ramp hasn't engaged yet).
            if effective_weight == 0.0:
                logs[f"train/{name}_gen_pred_mean"] = 0.0
                logs[f"train/{name}_gen_loss_weighted"] = 0.0
                continue
            # Freeze approx params for this forward — gen-side
            # backward only flows into pred_image, not the approx.
            model.requires_grad_(False)
            try:
                if single_input:
                    pred = model(pred_image.to(pred_image.dtype)).float()
                else:
                    pred = model(
                        pred_image.to(pred_image.dtype), gt_lat_det,
                    ).float()
                pred_mean = pred.mean()
                loss = loss + sign * effective_weight * pred_mean.to(pred_image.dtype)
            finally:
                model.requires_grad_(True)
            logs[f"train/{name}_gen_pred_mean"] = float(
                pred_mean.detach().item()
            )
            logs[f"train/{name}_gen_loss_weighted"] = float(
                (sign * effective_weight * pred_mean).detach().item()
            )
        return loss, logs

    def _get_lpips_model(self) -> Optional["torch.nn.Module"]:
        """Lazy-build (and cache) the LPIPS-VGG perceptual distance
        model. Returns None when ``lpips_loss_weight == 0``.

        VGG16 backbone is downloaded once into ``~/.cache/torch/hub`` —
        all ranks already have it after the first run. Model is
        ``eval()`` + ``requires_grad_(False)`` so its parameters are
        not trained — only its forward gradient flows back into the
        gen via the decoded pixels.
        """
        if self.lpips_loss_weight <= 0:
            return None
        cached = getattr(self, "_lpips_model", None)
        if cached is not None:
            return cached
        try:
            import lpips as _lpips
        except ImportError as e:
            raise RuntimeError(
                "lpips_loss_weight>0 requires the 'lpips' python "
                f"package: pip install lpips. Original error: {e}"
            )
        model = _lpips.LPIPS(net="vgg", verbose=False)
        model = model.to(device=self.device, dtype=torch.float32)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._lpips_model = model
        if self.is_main_process:
            n_params = sum(p.numel() for p in model.parameters())
            logging.info(
                "[ActionForcing] LPIPS(vgg) lazy-built: params=%.2fM "
                "(weight=%.3f, n_frames=%d, crop_size=%d)",
                n_params / 1e6, self.lpips_loss_weight,
                self.lpips_n_frames, self.lpips_crop_size,
            )
        return model

    def _vae_decode_grad(
        self,
        latent: torch.Tensor,
        use_checkpoint: bool = True,
    ) -> torch.Tensor:
        """Graph-on VAE decode using the dummy-leading-frame trick.

        ``latent``: ``[B, F_lat, C, H_lat, W_lat]`` (with grad).
        Returns: ``[B, F_pix, 3, H_pix, W_pix]`` in ``[-1, 1]``.

        The WAN VAE single-shot decode produces a "special first
        latent" output (1 pixel frame) and 4× temporal expansion for
        the rest. We prepend a dummy frame and slice ``[:, 1:]`` so
        the dummy absorbs the special-first behavior and our actual
        latents get the full 4× expansion. Caller controls grad
        context — used here for graph-on decode that flows gradient
        back to the gen.

        ``use_checkpoint=True`` wraps the VAE forward in
        ``torch.utils.checkpoint`` so activations are recomputed
        during backward instead of cached. ~30% extra compute but
        ~5-10 GB activation savings — required to fit graph-on decode
        alongside the gen rollout's persistent activations.
        """
        vae = getattr(self.model, "vae", None)
        if vae is None:
            raise RuntimeError(
                "_vae_decode_grad requires self.model.vae."
            )
        dummy = latent[:, 0:1]
        lat_pad = torch.cat([dummy, latent], dim=1)
        if use_checkpoint:
            from torch.utils.checkpoint import checkpoint as _ckpt

            def _decode(z):
                return vae.decode_to_pixel(z)

            pix = _ckpt(_decode, lat_pad, use_reentrant=False)
        else:
            pix = vae.decode_to_pixel(lat_pad)
        return pix[:, 1:, ...]

    def _compute_pixel_perceptual_losses(
        self,
        pred_image: torch.Tensor,
        gt_latents_window: torch.Tensor,
        current_step: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """LPIPS + pixel reconstruction losses on a graph-on subset.

        Decodes a small (``lpips_n_frames``) random subset of the gen
        prediction through the VAE WITH GRAD, decodes the matching
        GT latents WITHOUT GRAD (target), optionally crops to a
        smaller resolution, and computes:

          * LPIPS-VGG distance per frame (anti-blur, perceptual)
          * Pixel L1 / MSE per frame (anti-drift, direct supervision)

        Returns ``(perceptual_loss, logs)`` where ``perceptual_loss``
        is graph-attached scalar (to be added to generator_loss);
        ``logs`` is a flat ``str -> float`` dict.

        Gated by ``self.lpips_loss_weight > 0`` AND/OR
        ``self.pixel_recon_loss_weight > 0``. Returns zero scalar +
        empty logs when both are zero.
        """
        device = pred_image.device
        zero = torch.zeros((), device=device, dtype=torch.float32)
        lpips_active = self.lpips_loss_weight > 0
        recon_active = self.pixel_recon_loss_weight > 0
        if not (lpips_active or recon_active):
            return zero, {}

        # Pick a small random frame subset (DDP-synced).
        F_lat = pred_image.shape[1]
        n_pick = max(1, min(int(self.lpips_n_frames), int(F_lat)))
        frame_idx = self._sample_critic_grad_frame_indices(
            F_lat, n_pick, device,
        )

        # Slice latents to the subset and decode.
        pred_lat_sub = pred_image[:, frame_idx].to(torch.float32)
        gt_lat_sub = (
            gt_latents_window[:, frame_idx].detach().to(torch.float32)
        )
        # Graph-on for gen, no_grad for GT.
        pred_pix = self._vae_decode_grad(pred_lat_sub).to(torch.float32)
        with torch.no_grad():
            gt_pix = self._vae_decode_grad(gt_lat_sub).to(torch.float32)

        # ``pred_pix`` / ``gt_pix`` are ``[B, F_pix, 3, H, W]`` in
        # ``[-1, 1]``. The WAN VAE decode produces 4× temporal
        # expansion; both sides have the same F_pix so they're
        # frame-aligned for per-pixel loss computation.
        B, F_pix, C, H, W = pred_pix.shape

        # Optional random spatial crop. Single crop per iter (DDP-
        # synced) so all ranks compute the loss on the same region.
        crop = int(self.lpips_crop_size)
        if 0 < crop < min(H, W):
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                h_off = torch.randint(0, H - crop + 1, (1,), device=device)
                w_off = torch.randint(0, W - crop + 1, (1,), device=device)
                offsets = torch.cat([h_off, w_off]).long()
            else:
                offsets = torch.empty(2, dtype=torch.long, device=device)
            if dist.is_initialized():
                dist.broadcast(offsets, src=0)
            h_o = int(offsets[0].item())
            w_o = int(offsets[1].item())
            pred_pix = pred_pix[..., h_o:h_o + crop, w_o:w_o + crop]
            gt_pix = gt_pix[..., h_o:h_o + crop, w_o:w_o + crop]

        # Flatten frames into batch for both losses (frame-independent).
        pred_flat = pred_pix.reshape(B * F_pix, C, pred_pix.shape[-2], pred_pix.shape[-1])
        gt_flat = gt_pix.reshape(B * F_pix, C, gt_pix.shape[-2], gt_pix.shape[-1])

        loss = zero
        logs: Dict[str, float] = {}
        if lpips_active:
            lpips_model = self._get_lpips_model()
            # LPIPS expects ``[N, 3, H, W]`` in ``[-1, 1]`` — already
            # in that range from the WAN VAE clamp.
            lpips_dist = lpips_model(pred_flat, gt_flat)
            lpips_loss = lpips_dist.mean()
            loss = loss + self.lpips_loss_weight * lpips_loss
            logs["train/lpips_loss_raw"] = float(lpips_loss.detach().item())
            logs["train/lpips_loss_weighted"] = float(
                (self.lpips_loss_weight * lpips_loss).detach().item()
            )
        if recon_active:
            if self.pixel_recon_loss_type == "mse":
                recon_loss = ((pred_flat - gt_flat) ** 2).mean()
            else:  # default l1
                recon_loss = (pred_flat - gt_flat).abs().mean()
            loss = loss + self.pixel_recon_loss_weight * recon_loss
            logs["train/pixel_recon_loss_raw"] = float(
                recon_loss.detach().item()
            )
            logs["train/pixel_recon_loss_weighted"] = float(
                (self.pixel_recon_loss_weight * recon_loss).detach().item()
            )
        logs["train/lpips_n_frames"] = float(n_pick)
        return loss.to(pred_image.dtype), logs

    def _gan_warmup_shape_apply(self, t_normalized: float) -> float:
        """Apply ``self.gan_warmup_shape`` to a normalised ramp position
        ``t ∈ [0, 1]``. Returns a value in ``[0, 1]`` that the caller
        multiplies by ``gan_loss_weight`` to get the effective gen-side
        weight. See ``__init__`` for shape definitions.
        """
        t = max(0.0, min(1.0, float(t_normalized)))
        if self.gan_warmup_shape == "quadratic":
            return t * t
        if self.gan_warmup_shape == "cosine":
            return 0.5 * (1.0 - math.cos(math.pi * t))
        return t  # "linear"

    def _compute_r3gan_losses_distilled(
        self,
        pred_image: torch.Tensor,
        gt_latents_window: torch.Tensor,
        current_step: int,
        flash_dmd_gan_x0: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Distilled-critic R3GAN flow with action_critic-style boosts.

        Order of operations (mirrors ``_compute_action_critic_losses``):

          1. Path 1 (D-update on heads + R1/R2) — trains the SAM2 disc
             heads on this iter's (real, fake) features.
          2. Path 2 (critic value/grad distillation) — trains the
             ``LatentSAM2Critic`` to match the freshly-updated disc.
          3. Path 3 (gen-side ``-critic(fake_lat_grad)``) — uses the
             FRESHLY-UPDATED critic so the generator sees current
             gradients, not the previous iter's stale critic.

        This mirrors the action-critic pattern: train the critic just
        before the generator consumes it. Paths 1+2 run INLINE up-front
        so Path 3 sees the freshly-updated critic.
        """
        device = pred_image.device
        zero = torch.zeros((), device=device, dtype=torch.float32)

        critic = self.latent_critic

        # ----- Latents (fp32 for R1/R2 stability — Paths 1 + 2 both fp32).
        real_lat = gt_latents_window.detach().to(torch.float32)
        # G-side fake latent: Flash-DMD = flash_dmd_gan_x0 (paper §3.3
        # Eq. 9); else the rolling rollout's pred_image. Path 3 uses
        # the GRAD-attached version; Paths 1+2 use detached.
        if flash_dmd_gan_x0 is not None:
            fake_lat_grad = flash_dmd_gan_x0.to(torch.float32)
        else:
            fake_lat_grad = pred_image.to(torch.float32)
        fake_lat = fake_lat_grad.detach()

        logs: Dict[str, float] = {}

        # ===== Paths 1+2: D-update + critic value-distillation =======
        # Gated by ``gan_disc_start_step`` (default 0 = train from step
        # 0). When deferred, the entire D pass is skipped — no D
        # activations live, no D backward, no D optim step. Surfaces a
        # ``train/r3gan_disc_skipped`` flag so wandb can mark the gap.
        if current_step >= self.gan_disc_start_step:
            self._run_distilled_disc_critic_update(
                real_lat=real_lat,
                fake_lat=fake_lat,
                current_step=current_step,
                out=logs,
            )
            logs["train/r3gan_disc_skipped"] = 0.0
        else:
            logs["train/r3gan_disc_skipped"] = 1.0

        # ===== Path 3: Gen-side via critic (after warmup) =============
        critic_warmup_done = (
            current_step >= self.gan_critic_warmup_steps
        )
        # Gen GAN weight ramp: 0 until critic warmup, then linear
        # 0 → gan_loss_weight over the next ``gan_warmup_steps``.
        if (
            self.gan_warmup_steps > 0
            and current_step < (
                self.gan_critic_warmup_steps + self.gan_warmup_steps
            )
            and current_step >= self.gan_critic_warmup_steps
        ):
            ramp_steps_in = current_step - self.gan_critic_warmup_steps
            t_norm = ramp_steps_in / max(1, self.gan_warmup_steps)
            ramp = self._gan_warmup_shape_apply(t_norm)
            gen_gan_weight = ramp * self.gan_loss_weight
        elif current_step >= (
            self.gan_critic_warmup_steps + self.gan_warmup_steps
        ):
            gen_gan_weight = self.gan_loss_weight
        else:
            gen_gan_weight = 0.0

        if critic_warmup_done and gen_gan_weight > 0:
            # Freeze critic params so the gen backward doesn't write
            # critic-side gradients into the critic optim (the critic
            # has already been updated above for this iter, OR will be
            # updated in the deferred step for Sobolev). Mirrors the
            # legacy ``disc_for_guidance.requires_grad_(False)`` pattern
            # and matches ``_compute_action_critic_losses``'s
            # ``critic_for_guidance.requires_grad_(False)`` block.
            critic.requires_grad_(False)
            try:
                gen_critic_logit = critic(fake_lat_grad).float()
                gen_gan_main = -gen_critic_logit.mean()
                generator_gan_loss = (
                    gen_gan_weight * gen_gan_main.to(pred_image.dtype)
                )
            finally:
                critic.requires_grad_(True)
            d_fake_for_g_value = float(gen_critic_logit.detach().mean().item())
            gen_gan_main_value = float(gen_gan_main.detach().item())
        else:
            generator_gan_loss = zero
            d_fake_for_g_value = 0.0
            gen_gan_main_value = 0.0
        # Capture the GAN-only weighted value BEFORE the perceptual
        # losses are summed in below. This is what
        # ``train/r3gan_g_loss_weighted`` should reflect — the actual
        # gen-side GAN push on the student — without the perceptual-
        # approx bias term that's been contaminating the metric. The
        # student's total gradient is unchanged (still the sum), but
        # the wandb plot now reads cleanly.
        gan_only_weighted_value = (
            gen_gan_weight * gen_gan_main_value
            if (critic_warmup_done and gen_gan_weight > 0)
            else 0.0
        )

        # Gen-side perceptual loss via the trained approxes. Adds to
        # generator_gan_loss so the same downstream summing path works.
        # Backward flows through the (small) approxes into pred_image
        # — no VAE in autograd graph.
        perc_loss_weighted_value = 0.0
        if (
            self.mse_approx is not None
            or self.lpips_approx is not None
            or self.msssim_approx is not None
            or self.maniqa_approx is not None
        ):
            perc_loss, perc_logs = self._compute_gen_side_perceptual_loss(
                pred_image=fake_lat_grad,
                gt_latents_window=real_lat,
                current_step=current_step,
            )
            if (
                torch.is_tensor(perc_loss)
                and perc_loss.requires_grad
            ):
                perc_loss_weighted_value = float(perc_loss.detach().item())
            generator_gan_loss = generator_gan_loss + perc_loss
            logs.update(perc_logs)

        logs.update({
            "train/r3gan_d_fake_for_g": d_fake_for_g_value,
            "train/r3gan_g_loss_raw": gen_gan_main_value,
            # GAN-only weighted value (NOT bundled with perceptual
            # losses any more). Captured pre-summation above so the
            # plot shows ``gan_loss_weight × gen_gan_main`` cleanly.
            "train/r3gan_g_loss_weighted": gan_only_weighted_value,
            # Perceptual-approx weighted contribution (MANIQA + others)
            # — separate from the GAN-only key. Sum of these two
            # equals the previous (bundled) r3gan_g_loss_weighted.
            "train/perc_loss_weighted": perc_loss_weighted_value,
            "train/gan_plus_perc_loss_weighted": (
                gan_only_weighted_value + perc_loss_weighted_value
            ),
            "train/r3gan_g_weight": float(gen_gan_weight),
            "train/critic_warmup_done": 1.0 if critic_warmup_done else 0.0,
        })
        return generator_gan_loss, logs

    def _run_distilled_disc_critic_update(
        self,
        real_lat: torch.Tensor,
        fake_lat: torch.Tensor,
        current_step: int,
        out: Dict[str, Any],
    ) -> None:
        """Train the SAM2 disc heads (Path 1) + LatentSAM2Critic (Path 2).

        Operates on detached real/fake latents. Mutates ``out`` in
        place with Path 1 + Path 2 diagnostics. Runs INLINE before
        Path 3 so the gen consumes a freshly-trained critic.
        """
        from model.r3gan import rpgan_d_loss

        # R1/R2 gradient penalty on the SAM2-feature manifold (Option A).
        # The ADM 2D heads end with ``GAP → Linear(C → 1)`` so the
        # gradient at each input feature element is shrunk by the head's
        # spatial averaging factor: ``∂D / ∂feat[c, h, w] ≈ w[c] / (H*W)``
        # at the post-conv stage. The naive penalty
        # ``mean_batch(||∇D||²)`` therefore SHRINKS with feature spatial
        # size: per scale, ``Σ |∂D/∂feat|² ≈ HW * (1/HW)² = 1/HW``. With
        # 3 SAM2 scales of {128², 64², 32²}, the 32² scale dominates by
        # ~16x while the 128² scale contributes ~1/16 as much, and the
        # absolute magnitude is so small (~1e-3 to 1e-5 with γ=1) that
        # the penalty is effectively zero — disc saturates unconstrained.
        #
        # We restore paper-faithful magnitude by multiplying each scale's
        # per-sample squared-norm by its own ``H*W`` (undoing the GAP
        # shrinkage) before averaging across scales. With this, γ=1 puts
        # the penalty at the same scale as the paper's pixel-manifold
        # γ=1 result and the disc is properly regularized.
        def _gap_unscaled_grad_penalty(grads):
            terms = []
            for g in grads:
                # ``g.shape == [B*F, C, H, W]`` — multi-scale features
                # come out of SAM2 at different spatial dims per scale.
                spatial_size = g.shape[-2] * g.shape[-1]
                per_sample_sq = (g.flatten(1) ** 2).sum(dim=1)  # [B*F]
                terms.append((per_sample_sq * spatial_size).mean())
            return sum(terms) / max(1, len(terms))

        disc = self.r3gan_disc
        # Heads-only DDP wrap (or un-wrapped fallback). Routing the
        # heads forward through this wrapper is critical for DDP
        # all-reduce to fire on the heads' params (see
        # ``model/r3gan_sam2.py:_R3GANDiscHeads`` docstring).
        heads_for_update = (
            self.r3gan_heads_ddp
            if self.r3gan_heads_ddp is not None else disc.heads_module
        )
        critic = self.latent_critic
        critic_for_update = (
            self.latent_critic_ddp
            if self.latent_critic_ddp is not None else critic
        )

        vae = getattr(self.model, "vae", None)
        if vae is None:
            raise RuntimeError(
                "gan_sam2_distilled_critic=True requires self.model.vae."
            )

        # ``real_lat`` and ``fake_lat`` come from the gen-step's
        # stash. Both are detached (no graph from the gen rollout
        # remains since gen.backward already ran).
        B, F_, _, _, _ = real_lat.shape
        device = real_lat.device

        def _decode_no_grad(lat: torch.Tensor) -> torch.Tensor:
            """VAE decode with the dummy-leading-frame trick. Caller
            controls grad context (this fn assumes torch.no_grad).

            We chunk the input temporally and run the single-shot
            ``decode`` per chunk to bound peak workspace. The WAN VAE
            ``decode`` has a special "first latent" behavior (1
            output frame) and 4× expansion for the rest; the
            dummy-leading-frame trick + ``[:, 1:]`` slice is
            calibrated for that. ``cached_decode`` has different
            temporal semantics (4× per latent, no special-first) so
            we don't use it here.

            Chunk size: ``decode_chunk_size`` latent frames. Each
            chunk after the first prepends one frame from the prior
            chunk's tail to seed the temporal Conv3d's left-context
            (single-shot decode's first frame is the special
            short-output one). The first chunk's first latent is
            our dummy frame, so the left-context bootstrap matches
            the original full-clip decode semantics byte-for-byte.
            """
            dummy = lat[:, 0:1]
            lat_pad = torch.cat([dummy, lat], dim=1)
            pix = vae.decode_to_pixel(lat_pad)  # use_cache=False
            return pix[:, 1:, ...]

        def _decode_grad(lat: torch.Tensor) -> torch.Tensor:
            """VAE decode that PRESERVES the autograd graph through
            ``lat`` for the gradient-distillation target. Caller
            should keep input small (subset = 3 frames) to bound
            workspace. Same single-shot ``decode`` path as
            ``_decode_no_grad`` for consistent output shape.
            """
            dummy = lat[:, 0:1]
            lat_pad = torch.cat([dummy, lat], dim=1)
            pix = vae.decode_to_pixel(lat_pad)
            return pix[:, 1:, ...]

        # ===== Path 1: D-update (no_grad V+SAM2; R1/R2 on features) =====
        with torch.no_grad():
            # GT pixels: prefer the pre-decoded uint8 cache (loaded once
            # per sequence in ``_streaming_setup_sequence_from_ride``)
            # to skip the WAN VAE decode of GT every iter. Cached
            # tensor is uint8 [1, F_pix_window, 3, H, W] at the WAN VAE's
            # native pixel resolution (480x832); slice to the active
            # chunk window via the chunk_lo stash, dequantize to fp32
            # in [-1, 1] (matches ``_decode_no_grad`` output range).
            ss = getattr(self.model, "streaming_state", None) or {}
            cached_pixels_uint8 = ss.get("ride_pixels_window_uint8")
            chunk_lo_in_ride = ss.get("last_chunk_lo_in_ride_window")
            chunk_size_for_pix = ss.get("last_chunk_size")
            real_pixel_d: Optional[torch.Tensor] = None
            if (
                cached_pixels_uint8 is not None
                and chunk_lo_in_ride is not None
                and chunk_size_for_pix is not None
            ):
                pix_lo = int(chunk_lo_in_ride) * 4
                pix_hi = (int(chunk_lo_in_ride) + int(chunk_size_for_pix)) * 4
                if pix_hi <= cached_pixels_uint8.shape[1]:
                    sl = cached_pixels_uint8[:, pix_lo:pix_hi].to(
                        device=device, non_blocking=True,
                    )
                    real_pixel_d = (
                        sl.to(torch.float32) / 127.5 - 1.0
                    ).clamp_(-1.0, 1.0)
                    if not getattr(self, "_gt_pixel_cache_hit_logged", False):
                        logging.info(
                            "[gt-pixel-cache] using cached GT pixels "
                            "(chunk_lo=%d, %d→%d frames)",
                            int(chunk_lo_in_ride), pix_lo, pix_hi,
                        )
                        self._gt_pixel_cache_hit_logged = True
            if real_pixel_d is None:
                real_pixel_d = _decode_no_grad(real_lat).to(torch.float32)
            fake_pixel_d = _decode_no_grad(fake_lat).to(torch.float32)
            # The WAN VAE has 4× temporal expansion + the dummy-frame
            # trick produces ``F_pix = 4 * F_lat``. SAM2 features are
            # batched at the PIXEL frame count, so heads.forward needs
            # F_pix (not F_=21 from the latent shape).
            B_pix, F_pix = real_pixel_d.shape[0], real_pixel_d.shape[1]
            real_feats_raw = disc.forward_features(real_pixel_d)
            fake_feats_raw = disc.forward_features(fake_pixel_d)

        # ===== Path 1.5: Train dense per-token approxes (mse / lpips /
        # gan_d_approx). Reuses the no_grad pixel decodes above. Targets
        # are dense per-token at the latent-token grid (matches each
        # approx's post-stem output shape). Approxes train against
        # ``MSE(approx_pred, target)`` + ``β * MSE(approx_pred.mean(),
        # target.mean())``. Gen-side loss flows gradient through the
        # small approxes (no VAE in autograd graph) — see
        # ``_compute_gen_side_perceptual_loss``.
        if (
            self.mse_approx is not None
            or self.lpips_approx is not None
            or self.msssim_approx is not None
            or self.maniqa_approx is not None
        ):
            # Target token grid: latent spatial dim ceil-divided by 8
            # (matches the approx's post-stem shape — 3× stride-2
            # conv with padding=1 produces ``ceil(H/8)`` tokens, not
            # ``floor`` — e.g. H=60 → 30 → 15 → 8, not 7).
            target_h = max(1, (real_lat.shape[-2] + 7) // 8)
            target_w = max(1, (real_lat.shape[-1] + 7) // 8)
            (
                target_mse,
                target_lpips,
                target_msssim,
            ) = self._compute_dense_perceptual_targets(
                gen_pix=fake_pixel_d,
                gt_pix=real_pixel_d,
                F_lat=int(F_),
                target_h=target_h,
                target_w=target_w,
            )
            # MANIQA target — sparse spatial (only patch-covered
            # tokens). NaN at unsupervised positions; approx training
            # masks them out. Built no_grad on the gen pixels alone
            # (no GT — NR-IQA). Wall-clock recorded for the timing
            # smoke (option 1 vs middle option).
            (
                target_maniqa,
                target_maniqa_mask,
                maniqa_target_build_ms,
            ) = self._compute_dense_maniqa_target(
                gen_pix=fake_pixel_d,
                F_lat=int(F_),
                target_h=target_h,
                target_w=target_w,
            )
            if maniqa_target_build_ms > 0:
                out["train/maniqa_target_build_ms"] = float(
                    maniqa_target_build_ms
                )
            # NOTE: disc dense target computation removed — the
            # PerceptualApprox-based gan_d_approx is gone. The
            # LatentSAM2Critic distillation in Path 2 below uses
            # per-frame disc logit targets (its existing API).

            # ----- (2) Diagnostic quality logs (no_grad, lightweight).
            # Reuse the dense targets we already computed so .mean() is
            # essentially free — we don't need to recompute heavy
            # pixel-space subtractions.
            with torch.no_grad():
                if target_mse is not None:
                    out["diag/pixel_mse"] = float(target_mse.mean().item())
                if target_lpips is not None:
                    out["diag/pixel_lpips"] = float(
                        target_lpips.mean().item()
                    )
                if target_msssim is not None:
                    out["diag/pixel_msssim"] = float(
                        target_msssim.mean().item()
                    )
            # ----- v11-style multi-noise interpolations (Lipschitz). -----
            # Sample K interpolation points along the (fake_lat → real_lat)
            # line. For each, compute dense per-token targets via the
            # no_grad pipeline (VAE decode + LPIPS + SAM2/disc-dense
            # if needed). Process sequentially so peak transient
            # memory is bounded to one interpolation's no_grad cost.
            # Approxes train on these K extra anchor points in addition
            # to the original (fake, gt) pair — the densified value-
            # field supervision is what makes the approx's gradient
            # (gen-side ``-approx.mean()``) actually meaningful.
            n_interp = max(0, int(self.perceptual_approx_n_interp_samples))
            interp_data: List[Dict[str, torch.Tensor]] = []
            if n_interp > 0:
                alphas = torch.linspace(
                    0.0, 1.0, n_interp + 2, device=device,
                )[1:-1]
                for alpha_t in alphas:
                    a = float(alpha_t.item())
                    interp_lat = (1.0 - a) * fake_lat + a * real_lat
                    with torch.no_grad():
                        interp_pixel = _decode_no_grad(interp_lat).to(
                            torch.float32,
                        )
                        (
                            i_target_mse,
                            i_target_lpips,
                            i_target_msssim,
                        ) = self._compute_dense_perceptual_targets(
                            gen_pix=interp_pixel,
                            gt_pix=real_pixel_d,
                            F_lat=int(F_),
                            target_h=target_h,
                            target_w=target_w,
                        )
                        del interp_pixel
                    interp_data.append({
                        "lat": interp_lat.detach(),
                        "mse": i_target_mse,
                        "lpips": i_target_lpips,
                        "msssim": i_target_msssim,
                    })
            self._run_perceptual_approx_update(
                gen_lat=fake_lat,
                gt_lat=real_lat,
                target_mse=target_mse,
                target_lpips=target_lpips,
                target_msssim=target_msssim,
                target_maniqa=target_maniqa,
                target_maniqa_mask=target_maniqa_mask,
                interp_data=interp_data,
                out=out,
            )
        # Detach-and-leaf the features for R1/R2 (Option A: penalty on
        # the feature manifold, not pixel manifold).
        real_feats = [
            f.detach().requires_grad_(True) for f in real_feats_raw
        ]
        fake_feats = [
            f.detach().requires_grad_(True) for f in fake_feats_raw
        ]
        d_loss_value = 0.0
        d_real_value = 0.0
        d_fake_detached_value = 0.0
        r1_value = 0.0
        r2_value = 0.0
        for _k in range(max(1, self.gan_updates_per_step)):
            self.r3gan_optimizer.zero_grad(set_to_none=True)
            # Re-detach + re-leaf each iter so R1/R2 grads chain only
            # through the current iter's heads forward.
            real_feats_iter = [
                f.detach().requires_grad_(True) for f in real_feats
            ]
            fake_feats_iter = [
                f.detach().requires_grad_(True) for f in fake_feats
            ]
            # Route through the heads-DDP wrapper so DDP's forward-time
            # tracking fires (essential — if we routed through the
            # un-wrapped ``disc.heads_module`` here, the backward
            # below would NOT trigger DDP all-reduce on the heads'
            # params and ranks would silently diverge).
            # Use PIXEL frame count (B_pix, F_pix) — features are
            # batched at the post-VAE-decode pixel rate, not the
            # latent rate.
            d_real = heads_for_update(
                real_feats_iter, B_pix, F_pix,
            )
            r1_grads = torch.autograd.grad(
                d_real.sum(), real_feats_iter,
                create_graph=True, retain_graph=True,
            )
            r1 = self.gan_r1_gamma * _gap_unscaled_grad_penalty(r1_grads)
            d_fake_d = heads_for_update(
                fake_feats_iter, B_pix, F_pix,
            )
            r2_grads = torch.autograd.grad(
                d_fake_d.sum(), fake_feats_iter,
                create_graph=True, retain_graph=True,
            )
            r2 = self.gan_r2_gamma * _gap_unscaled_grad_penalty(r2_grads)
            d_main = rpgan_d_loss(d_real, d_fake_d)
            d_total = d_main + r1 + r2
            d_total.backward()
            if self.gan_max_grad_norm is not None and self.gan_max_grad_norm > 0:
                # In distilled mode the only trainable disc params are
                # the heads (the SAM2 encoder is frozen). Clip those.
                heads_params_iter = (
                    self.r3gan_heads_ddp.parameters()
                    if self.r3gan_heads_ddp is not None
                    else disc.heads_module.parameters()
                )
                torch.nn.utils.clip_grad_norm_(
                    [p for p in heads_params_iter if p.grad is not None],
                    self.gan_max_grad_norm,
                )
            self.r3gan_optimizer.step()
            d_loss_value = float(d_main.detach().item())
            d_real_value = float(d_real.detach().mean().item())
            d_fake_detached_value = float(d_fake_d.detach().mean().item())
            r1_value = float(r1.detach().item())
            r2_value = float(r2.detach().item())

        # ===== Path 2: Latent critic value+grad distillation ==========
        # Value targets — full-frame, no_grad teacher forward. Reuse
        # Path 1's already-computed features (avoids two redundant
        # SAM2 forwards per iter, saving ~6 GB transient workspace).
        # When dense distillation is enabled, additionally compute the
        # per-token logit maps (forward_dense_heads at the critic's
        # post-stem grid) and pool 4 pixel-frames → 1 latent-frame.
        from model.latent_sam2_critic import LatentSAM2Critic as _LSAM2C
        dense_distill = bool(self.gan_critic_dense_distillation)
        # Compute the post-stem grid the critic will produce so we can
        # ask the disc for a matching dense target. Defensive fallback
        # to the latent's spatial dims floor-divided by 8 (which is what
        # the critic's stem produces by construction).
        target_h, target_w = _LSAM2C.post_stem_grid(
            int(real_lat.shape[-2]), int(real_lat.shape[-1]),
        )
        with torch.no_grad():
            teacher_val_real = disc.forward_heads(
                real_feats_raw, batch_size=B_pix, num_frames=F_pix,
            ).float()
            teacher_val_fake = disc.forward_heads(
                fake_feats_raw, batch_size=B_pix, num_frames=F_pix,
            ).float()
            if dense_distill:
                teacher_dense_real = disc.forward_dense_heads(
                    real_feats_raw,
                    batch_size=B_pix,
                    num_frames=F_pix,
                    target_h=target_h,
                    target_w=target_w,
                ).float()  # [B, F_pix, target_h, target_w]
                teacher_dense_fake = disc.forward_dense_heads(
                    fake_feats_raw,
                    batch_size=B_pix,
                    num_frames=F_pix,
                    target_h=target_h,
                    target_w=target_w,
                ).float()
                # Pool F_pix → F_lat by averaging 4 pixel-frames per
                # latent (WAN VAE temporal expansion). Result:
                # [B, F_lat, target_h, target_w].
                pix_per_lat = F_pix // int(F_)
                if pix_per_lat * int(F_) != F_pix:
                    raise RuntimeError(
                        f"Dense distillation: F_pix={F_pix} not "
                        f"divisible by F_lat={F_}."
                    )
                teacher_dense_real = teacher_dense_real.view(
                    B, int(F_), pix_per_lat, target_h, target_w,
                ).mean(dim=2)
                teacher_dense_fake = teacher_dense_fake.view(
                    B, int(F_), pix_per_lat, target_h, target_w,
                ).mean(dim=2)
            else:
                teacher_dense_real = None
                teacher_dense_fake = None

        # ----- Frame-rate alignment for distillation -----
        # In ``frame_pool=none`` mode the disc returns ``[B*F_pix]``
        # per-pixel-frame logits and the critic returns ``[B*F_lat]``
        # per-latent-frame logits — different rates because the WAN
        # VAE has 4× temporal expansion (F_pix = 4·F_lat). To match
        # them for the value-distillation MSE, we average each latent's
        # 4 pixel-frame teacher logits down to a single per-latent
        # logit. Other pool modes already produce ``[B]`` per clip
        # so no alignment needed.
        def _align_teacher_to_critic(t: torch.Tensor) -> torch.Tensor:
            if t.dim() == 1 and t.shape[0] == B * F_pix:
                # Per-pixel-frame → per-latent-frame averaging.
                pix_per_lat = F_pix // int(F_)
                if pix_per_lat * int(F_) != F_pix:
                    raise RuntimeError(
                        f"Cannot align disc teacher: F_pix={F_pix} not "
                        f"divisible by F_lat={F_}."
                    )
                return t.view(B, F_, pix_per_lat).mean(dim=2).reshape(
                    B * F_,
                )
            return t

        teacher_val_real = _align_teacher_to_critic(teacher_val_real)
        teacher_val_fake = _align_teacher_to_critic(teacher_val_fake)

        # ----- Multi-step critic update loop. -----
        # Mirrors action_critic's ``critic_updates_per_step`` pattern:
        # step the critic optimizer K_c times per gen step so the
        # critic actually converges to the disc within one outer iter.
        # Default K_c=1 is the legacy single-step behavior.
        critic_updates = max(1, int(self.gan_critic_updates_per_step))
        # Track only the LAST iter's diagnostics for logging.
        L_value_value = 0.0
        L_value_dense_value = 0.0
        critic_val_real_last: Optional[torch.Tensor] = None
        critic_val_fake_last: Optional[torch.Tensor] = None
        for _kc in range(critic_updates):
            if self.latent_critic_optimizer is not None:
                self.latent_critic_optimizer.zero_grad(set_to_none=True)
            # Critic forward on FULL clip — value-only distillation.
            real_lat_critic_in = real_lat.clone().detach()
            fake_lat_critic_in = fake_lat.clone().detach()
            critic_val_real = critic_for_update(real_lat_critic_in).float()
            critic_val_fake = critic_for_update(fake_lat_critic_in).float()
            # Value loss on (real, fake) — scalar/per-frame MSE.
            L_value = (
                ((critic_val_real - teacher_val_real.detach()) ** 2).mean()
                + ((critic_val_fake - teacher_val_fake.detach()) ** 2).mean()
            )
            # Dense per-token value loss. Routes through ``forward``
            # with the ``dense=True`` kwarg so DDP intercepts the call
            # correctly (calling ``critic.forward_dense`` directly on
            # the DDP wrapper would skip grad-sync hooks and silently
            # break param synchronization).
            if dense_distill and teacher_dense_real is not None:
                critic_val_real_dense = critic_for_update(
                    real_lat_critic_in, dense=True,
                ).float()  # [B, F_lat, target_h, target_w]
                critic_val_fake_dense = critic_for_update(
                    fake_lat_critic_in, dense=True,
                ).float()
                L_value_dense = (
                    (
                        (critic_val_real_dense - teacher_dense_real.detach())
                        ** 2
                    ).mean()
                    + (
                        (critic_val_fake_dense - teacher_dense_fake.detach())
                        ** 2
                    ).mean()
                )
                L_value = L_value + L_value_dense
                L_value_dense_value = float(L_value_dense.detach().item())
                del critic_val_real_dense, critic_val_fake_dense
            if self.latent_critic_optimizer is not None:
                L_value.backward()
                if self.gan_max_grad_norm is not None and self.gan_max_grad_norm > 0:
                    critic_params_iter = (
                        self.latent_critic_ddp.parameters()
                        if self.latent_critic_ddp is not None
                        else critic.parameters()
                    )
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in critic_params_iter if p.grad is not None],
                        self.gan_max_grad_norm,
                    )
                self.latent_critic_optimizer.step()
            L_value_value = float(L_value.detach().item())
            critic_val_real_last = critic_val_real.detach()
            critic_val_fake_last = critic_val_fake.detach()
        # Diagnostic: critic↔disc value correlation on (real, fake).
        with torch.no_grad():
            cv_all = torch.cat([critic_val_real_last, critic_val_fake_last], dim=0)
            tv_all = torch.cat([teacher_val_real, teacher_val_fake], dim=0)
            if cv_all.numel() >= 2:
                cv_c = cv_all - cv_all.mean()
                tv_c = tv_all - tv_all.mean()
                denom = (cv_c.norm() * tv_c.norm()).clamp_min(1e-8)
                critic_disc_corr = float((cv_c * tv_c).sum() / denom)
            else:
                critic_disc_corr = 0.0

        # Diagnostics for ``out``.
        out["train/r3gan_d_loss"] = d_loss_value
        out["train/r3gan_r1"] = r1_value
        out["train/r3gan_r2"] = r2_value
        out["train/r3gan_d_real"] = d_real_value
        out["train/r3gan_d_fake_detached"] = d_fake_detached_value
        out["train/critic_value_loss"] = L_value_value
        out["train/critic_value_dense_loss"] = L_value_dense_value
        out["train/critic_logit_mean"] = float(
            (
                (critic_val_real_last.mean() + critic_val_fake_last.mean())
                / 2.0
            ).item()
        ) if critic_val_real_last is not None else 0.0
        out["train/disc_logit_mean"] = float(
            ((teacher_val_real.mean() + teacher_val_fake.mean()) / 2.0)
            .detach().item()
        )
        out["train/critic_disc_corr"] = critic_disc_corr
        out["train/critic_updates_per_step"] = float(critic_updates)

    def _compute_r3gan_losses(
        self,
        pred_image: torch.Tensor,
        gt_latents_window: torch.Tensor,
        current_step: int,
        flash_dmd_gan_x0: Optional[torch.Tensor] = None,
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
            flash_dmd_gan_x0: optional graph-on output of the Flash-DMD
                t=flash_dmd_gan_t gen forward. When provided, the G-side
                path uses this as the fake (so the GAN supervises only
                the gen's near-clean texture-refinement behavior).

        Returns:
            ``(generator_gan_loss, logs)`` — ``generator_gan_loss``
            is graph-carrying; ``logs`` is a flat ``str -> float``
            dict for wandb.
        """
        device = pred_image.device
        zero = torch.zeros((), device=device, dtype=torch.float32)
        if not self.gan_enabled or self.r3gan_disc is None:
            return zero, {}

        # Dispatch to the distilled-critic flow when enabled. Keeps
        # the legacy pixel-disc-in-gen-graph path as the fallback.
        if (
            getattr(self, "gan_sam2_distilled_critic", False)
            and self.latent_critic is not None
            and self.gan_backbone == "sam2_pixel"
        ):
            return self._compute_r3gan_losses_distilled(
                pred_image=pred_image,
                gt_latents_window=gt_latents_window,
                current_step=current_step,
                flash_dmd_gan_x0=flash_dmd_gan_x0,
            )

        disc_for_update = (
            self.r3gan_disc_ddp if self.r3gan_disc_ddp is not None else self.r3gan_disc
        )
        disc_for_guidance = self.r3gan_disc

        # D's own arithmetic stays in fp32 for the second-order
        # gradient penalty. The cost is one extra cast.
        # ``pred_image_for_g`` is the graph-carrying version used in
        # the G-side branch; ``*_detached`` versions feed the D-update.
        real_detached_lat = gt_latents_window.detach().to(torch.float32)
        fake_detached_lat = pred_image.detach().to(torch.float32)
        # Flash-DMD: the G-side fake is the t=flash_dmd_gan_t forward
        # output (a separate near-clean gen forward) when available;
        # otherwise the rolling rollout's pred_image is used. Grad
        # path is restricted to the t=gan_t forward — the high-noise
        # denoising rungs were no_grad.
        if flash_dmd_gan_x0 is not None:
            pred_image_for_g_lat = flash_dmd_gan_x0.to(torch.float32)
        else:
            pred_image_for_g_lat = pred_image.to(torch.float32)
        skip_g_side = False

        if self.gan_backbone == "sam2_pixel":
            # Pixel-space discriminator: decode the latent video to
            # pixels via the WAN VAE. The decode runs in fp32 (the
            # VAE's internal Conv3d kernels require fp32) and mirrors
            # the dummy-frame trick from ``_log_pred_image_video`` to
            # avoid the WAN VAE's first-frame artifact. Two D-side
            # decodes per iter (real + fake) ALWAYS fire; the third
            # G-side decode is skipped on Flash-DMD high-noise iters
            # (where the gen GAN loss would be discarded anyway).
            #
            # Deviation from paper §4.1 (auditor's #11): the paper
            # feeds RAW pixel reals directly (x_real ∼ D_real) and
            # only fakes go through V (V(z_fake)). Our dataset is
            # preprocessed to latent — no raw pixel videos on disk —
            # so real = V(encode(real_pixel)) goes through the same
            # VAE roundtrip as fake. Both real and fake live on the
            # V-decode manifold (no V-roundtrip artifacts to detect,
            # which is good), but the disc learns distinguishing on
            # the V-decode manifold rather than natural images (mild
            # distribution shift). Acceptable trade-off given data
            # constraints; would require a parallel pixel-video data
            # path to fix paper-faithfully.
            vae = getattr(self.model, "vae", None)
            if vae is None:
                raise RuntimeError(
                    "gan_backbone=sam2_pixel requires self.model.vae "
                    "to be set; the WAN VAE wrapper is built by the "
                    "parent SelfForcingModel init. Check the model "
                    "construction path."
                )

            def _decode(lat: torch.Tensor) -> torch.Tensor:
                # ``lat`` is [B, F, C, H, W] in fp32. Apply the
                # dummy-leading-frame trick (matches video logger).
                dummy = lat[:, 0:1]
                lat_pad = torch.cat([dummy, lat], dim=1)
                pix = vae.decode_to_pixel(lat_pad)
                return pix[:, 1:, ...]

            real_detached = _decode(real_detached_lat)
            fake_detached = _decode(fake_detached_lat)
            pred_image_for_g = (
                None if skip_g_side else _decode(pred_image_for_g_lat)
            )
        else:
            real_detached = real_detached_lat
            fake_detached = fake_detached_lat
            pred_image_for_g = (
                None if skip_g_side else pred_image_for_g_lat
            )

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
            t_norm = current_step / max(1, self.gan_warmup_steps)
            ramp = self._gan_warmup_shape_apply(t_norm)
            gen_gan_weight = ramp * self.gan_loss_weight
        else:
            gen_gan_weight = self.gan_loss_weight

        if gen_gan_weight > 0 and not skip_g_side:
            # Freeze the disc for the G-side path. On the SAM2
            # backbone the encoder stays frozen at all times — we
            # restore ``requires_grad=True`` only on the trainable
            # heads after the G-update.
            disc_for_guidance.requires_grad_(False)
            try:
                d_real_for_g = disc_for_guidance(real_detached).detach()
                d_fake_for_g = disc_for_guidance(pred_image_for_g)
                gen_gan_main = rpgan_g_loss(d_real_for_g, d_fake_for_g)
                generator_gan_loss = gen_gan_weight * gen_gan_main.to(pred_image.dtype)
            finally:
                # Restore trainability on the trainable params only.
                # For SAM2 backbone, the frozen encoder must stay
                # frozen — flip ALL params back to True then re-freeze
                # the encoder. For latent backbone, all params are
                # trainable so True everywhere is correct.
                disc_for_guidance.requires_grad_(True)
                if self.gan_backbone == "sam2_pixel":
                    enc = getattr(disc_for_guidance, "image_encoder", None)
                    if enc is not None:
                        for p in enc.parameters():
                            p.requires_grad_(False)
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
        index_overlay: Optional[Tuple[int, int, int]] = None,
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

        ``index_overlay``: optional
        ``(zarr_lat_lo, motion_chunk_offset, npb)`` tuple. When supplied,
        per-frame text "zarr_lat=N motion=M" is rendered top-left of
        each video frame so the displayed content can be cross-checked
        against the dataset (which exact zarr latent index this frame
        corresponds to, and which motion.npy chunk encodes its motion).
        Used only for ``clean_x_real`` (the only view that traces back
        to a specific ride window). For video frame ``t`` (after VAE
        4× temporal upsampling): zarr_lat = zarr_lat_lo + (t // 4),
        motion_chunk = motion_chunk_offset + (zarr_lat // npb).
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

        # Render the action overlay (best-effort, never raises). The
        # ``index_overlay`` indices (when supplied) are drawn in the
        # same pass — top-left "zarr_lat=N motion=M" text per frame so
        # we don't iterate the video twice.
        if action_overlay is not None:
            try:
                npb = int(getattr(self.config, "num_frame_per_block", 3))
                _zarr_lo = None
                _mco = None
                _zarr_name = ""
                if index_overlay is not None:
                    # 4-tuple from the trainer call site:
                    # (zarr_lat_lo, motion_chunk_offset, npb, zarr_name).
                    # Tolerate the legacy 3-tuple shape too.
                    if len(index_overlay) == 4:
                        _zarr_lo, _mco, _, _zarr_name = index_overlay
                    else:
                        _zarr_lo, _mco, _ = index_overlay
                self._draw_action_overlay(
                    vid_np, action_overlay, frames_per_latent=4, npb=npb,
                    zarr_lat_lo=_zarr_lo, motion_chunk_offset=_mco,
                    zarr_name=_zarr_name,
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
        zarr_lat_lo: Optional[int] = None,
        motion_chunk_offset: Optional[int] = None,
        zarr_name: str = "",
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
        # cv2 needs a contiguous frame buffer; if the caller passed in
        # a transposed (= non-contiguous) ``vid_np`` we use a per-frame
        # contiguous copy and write it back.
        cv2 = None
        do_index = (
            zarr_lat_lo is not None and motion_chunk_offset is not None
        )
        if do_index:
            try:
                import cv2 as _cv2
                cv2 = _cv2
            except Exception:
                cv2 = None
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

            # Per-frame zarr-latent + motion.npy chunk annotation
            # (top-left). Mapping: video frame t -> dataset latent
            # zarr_lat_lo + (t // frames_per_latent); dataset latent k
            # -> motion.npy chunk motion_chunk_offset + (k // npb).
            # Two stacked lines:
            #   line 1 (y=14): zarr file stem (which recording)
            #   line 2 (y=28): per-frame indices into that recording
            if cv2 is not None:
                zarr_lat = int(zarr_lat_lo) + (t // frames_per_latent)
                motion_chunk = int(motion_chunk_offset) + (zarr_lat // npb)
                strip_top_h = 34 if zarr_name else 20
                vid_np[t, :strip_top_h, :, :] = vid_np[t, :strip_top_h, :, :] // 4
                frame = np.ascontiguousarray(vid_np[t])
                if zarr_name:
                    cv2.putText(
                        frame,
                        f"zarr={zarr_name}",
                        (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                _idx_y = 28 if zarr_name else 14
                cv2.putText(
                    frame,
                    f"lat={zarr_lat} motion={motion_chunk}",
                    (4, _idx_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                vid_np[t] = frame


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

    def _mem_step_snapshot(self, label: str) -> None:
        """Capture (allocated, peak-since-last-snapshot) at a labeled
        boundary inside one training step. Cleared at the start of each
        step. Dumped from ``_dump_step_mem_breakdown`` (called at end of
        step 1 / step 5).
        """
        if not (
            torch.cuda.is_available()
            and getattr(self, "is_main_process", True)
            and bool(getattr(self.config, "memory_audit_enabled", False))
        ):
            return
        if not hasattr(self, "_mem_step_snaps"):
            self._mem_step_snaps: List[Tuple[str, int, int]] = []
        try:
            alloc = torch.cuda.memory_allocated()
            peak = torch.cuda.max_memory_allocated()
            self._mem_step_snaps.append((label, alloc, peak))
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def _dump_step_mem_breakdown(self, step_label: str) -> None:
        """Print the full sequence of (label, allocated, peak-since-prev)
        snapshots collected during one step. Reveals which boundary in
        the step's code path drives each chunk of the transient peak.
        """
        if not (
            torch.cuda.is_available()
            and getattr(self, "is_main_process", True)
            and bool(getattr(self.config, "memory_audit_enabled", False))
        ):
            return
        snaps = getattr(self, "_mem_step_snaps", None)
        if not snaps:
            return

        def _gb(x: int) -> float:
            return x / (1024 ** 3)

        prev_alloc = snaps[0][1]
        logging.info(
            "[mem-step %s] === Per-boundary alloc + peak-since-prev ===",
            step_label,
        )
        logging.info(
            "[mem-step %s]   %-32s  alloc=%6.2f GB  peak_in_segment=%6.2f GB  "
            "delta_alloc=%+6.2f GB",
            step_label, snaps[0][0], _gb(snaps[0][1]), _gb(snaps[0][2]), 0.0,
        )
        for i in range(1, len(snaps)):
            label, alloc, peak = snaps[i]
            delta = alloc - prev_alloc
            logging.info(
                "[mem-step %s]   %-32s  alloc=%6.2f GB  peak_in_segment=%6.2f GB  "
                "delta_alloc=%+6.2f GB",
                step_label, label, _gb(alloc), _gb(peak), _gb(delta),
            )
            prev_alloc = alloc

    def _dump_memory_audit(self, label: str) -> None:
        """Print a structured memory audit so the operator can see
        WHERE the GPU memory is going. Three sections per call:

          1. Per-top-level-module parameter + buffer byte tally on the
             current rank. Reveals which subsystems (gen, fake_score,
             real_score, action_critic, state_probe, perceptual approxes,
             SAM2 disc, etc.) are the heavy hitters.
          2. ``torch.cuda.memory_stats`` snapshot: allocated / reserved
             / fragmentation / peak. Shows how much PyTorch is holding
             vs. how much it could free.
          3. Free-vs-total via ``torch.cuda.mem_get_info``: includes the
             non-PyTorch overhead (NCCL buffers, cuDNN workspace,
             CUDA context) so we can size the actual usable PyTorch
             ceiling.

        Only rank 0 logs (so the run.log isn't 32×repeated). Other
        ranks compute their own peaks for local inspection if needed
        but stay quiet.
        """
        if not (
            torch.cuda.is_available()
            and getattr(self, "is_main_process", True)
        ):
            return
        try:
            dev = torch.cuda.current_device()
            # ---- (1) Per-module param/buffer tally -----------------
            buckets: List[Tuple[str, int, int]] = []  # (name, n_params, bytes)

            def _bucket(name: str, mod: Optional[torch.nn.Module]) -> None:
                if mod is None:
                    return
                # Unwrap DDP for honest counting.
                inner = (
                    mod.module if isinstance(mod, DDP) else mod
                )
                n_param = 0
                n_bytes = 0
                for p in inner.parameters():
                    if p is None:
                        continue
                    n_param += p.numel()
                    n_bytes += p.numel() * p.element_size()
                for b in inner.buffers():
                    if b is None:
                        continue
                    n_bytes += b.numel() * b.element_size()
                buckets.append((name, n_param, n_bytes))

            m = self.model
            _bucket("generator", getattr(m, "generator", None))
            _bucket("fake_score", getattr(m, "fake_score", None))
            _bucket("real_score", getattr(m, "real_score", None))
            _bucket("real_score_frozen", getattr(m, "real_score_frozen", None))
            _bucket("action_projection", getattr(m, "action_projection", None))
            _bucket("action_token_projection", getattr(m, "action_token_projection", None))
            _bucket("action_critic", getattr(m, "action_critic", None))
            _bucket("state_probe", getattr(m, "state_probe", None))
            _bucket("mse_approx", getattr(self, "mse_approx", None))
            _bucket("lpips_approx", getattr(self, "lpips_approx", None))
            _bucket("msssim_approx", getattr(self, "msssim_approx", None))
            _bucket("latent_critic", getattr(self, "latent_critic", None))
            _bucket("r3gan_disc", getattr(self, "r3gan_disc", None))
            _bucket(
                "_frozen_cotracker",
                getattr(self, "_frozen_cotracker", None),
            )
            _bucket("_frozen_ss_vae", getattr(self, "_frozen_ss_vae", None))
            _bucket(
                "_frozen_vae",
                getattr(self, "_frozen_vae", None),
            )

            total_bytes = sum(b for _, _, b in buckets)
            buckets.sort(key=lambda t: -t[2])

            # ---- (2) PyTorch alloc / reserved / peak ---------------
            stats = torch.cuda.memory_stats(device=dev)
            alloc = stats.get("allocated_bytes.all.current", 0)
            reserved = stats.get("reserved_bytes.all.current", 0)
            peak_alloc = stats.get("allocated_bytes.all.peak", 0)
            peak_reserved = stats.get("reserved_bytes.all.peak", 0)

            # ---- (3) Free vs total (includes non-PyTorch overhead) -
            free_b, total_b = torch.cuda.mem_get_info(dev)
            non_pt = max(0, (total_b - free_b) - reserved)

            def _gb(x: int) -> float:
                return x / (1024 ** 3)

            logging.info(
                "[mem-audit %s] === Per-module param+buffer bytes ===",
                label,
            )
            for name, np_, nb in buckets:
                if nb == 0:
                    continue
                logging.info(
                    "[mem-audit %s]   %-22s  params=%9.2fM  bytes=%6.2f GB",
                    label, name, np_ / 1e6, _gb(nb),
                )
            logging.info(
                "[mem-audit %s]   %-22s  bytes=%6.2f GB  (sum of above)",
                label, "TOTAL_MODULE_BYTES", _gb(total_bytes),
            )
            logging.info(
                "[mem-audit %s] === PyTorch caching allocator ===",
                label,
            )
            logging.info(
                "[mem-audit %s]   allocated_now=%.2f GB  reserved_now=%.2f GB  "
                "fragmentation=%.1f%%",
                label, _gb(alloc), _gb(reserved),
                100.0 * (1.0 - alloc / max(reserved, 1)),
            )
            logging.info(
                "[mem-audit %s]   peak_allocated=%.2f GB  peak_reserved=%.2f GB",
                label, _gb(peak_alloc), _gb(peak_reserved),
            )
            logging.info(
                "[mem-audit %s] === GPU as a whole (rank 0) ===",
                label,
            )
            logging.info(
                "[mem-audit %s]   total=%.2f GB  free=%.2f GB  "
                "pytorch_reserved=%.2f GB  non_pytorch=%.2f GB",
                label, _gb(total_b), _gb(free_b),
                _gb(reserved), _gb(non_pt),
            )
        except Exception as exc:
            logging.warning(
                "[mem-audit %s] failed: %s", label, exc,
            )

    def _apply_dmd_context_mix_p_schedule(self) -> float:
        """Compute and apply ``dmd_context_mix_p`` for this outer iter.

        Three modes; sensor takes priority, then step-switch, then
        linear ramp:

        1. **Sensor-gated** (``dmd_context_mix_p_sensor_enabled=True``):
           Reads the most recent ``gen/dmd_pf_minus_pr_mae`` value
           cached on ``self._latest_dmd_pf_minus_pr_mae`` (updated by
           ``_streaming_train_one_chunk`` after each gen step). When
           the metric is below ``..._sensor_threshold``, sets mix_p to
           ``..._sensor_high`` (more GT context — bolster the teacher
           when it's about to be insufficient as a target). When above,
           sets to ``..._sensor_low``. Stateless single-threshold gate.
           No-op until the first gen iter has produced a reading;
           defaults to ``..._sensor_low`` until then.

        2. **Hard step-switch** (sensor off,
           ``dmd_context_mix_p_step_switch_at>0``): mix_p = start when
           ``self.step < step_switch_at``; mix_p = target thereafter.
           No interpolation — a single discrete flip at that step.

        3. **Linear ramp** (sensor off, step_switch_at<=0): from the
           config-time start (= ``dmd_context_mix_p``) to ``..._target``
           over the first ``..._ramp_steps`` outer steps. With
           ramp_steps <= 0 the knob is held at start (legacy behavior).

        Mutates ``self.model.dmd_context_mix_p`` directly. The model
        reads it every iter inside ``_build_dmd_context_kwargs`` so the
        next forward picks up the new value without any other plumbing.
        Returns the value applied this iter so callers can log it.
        """
        if self._dmd_context_mix_p_sensor_enabled:
            sensor_low = float(self._dmd_context_mix_p_sensor_low)
            sensor_high = float(self._dmd_context_mix_p_sensor_high)
            sensor_thr = float(self._dmd_context_mix_p_sensor_threshold)
            latest = self._latest_dmd_pf_minus_pr_mae
            if latest is None:
                p = sensor_low
            elif latest < sensor_thr:
                p = sensor_high
            else:
                p = sensor_low
        elif self._dmd_context_mix_p_step_switch_at > 0:
            start = float(self._dmd_context_mix_p_start)
            target = float(self._dmd_context_mix_p_target)
            if self.step < self._dmd_context_mix_p_step_switch_at:
                p = start
            else:
                p = target
        else:
            start = float(self._dmd_context_mix_p_start)
            target = float(self._dmd_context_mix_p_target)
            ramp_steps = int(self._dmd_context_mix_p_ramp_steps)
            if ramp_steps <= 0 or start == target:
                p = start
            elif self.step >= ramp_steps:
                p = target
            else:
                frac = float(self.step) / float(ramp_steps)
                p = start + (target - start) * frac
        # Clamp to [0, 1] defensively (the model raises on out-of-range
        # values at init; mirror the same bound here so a bad config
        # doesn't surface mid-run).
        p = max(0.0, min(1.0, p))
        default_for_compare = float(self._dmd_context_mix_p_start)
        if p != float(getattr(self.model, "dmd_context_mix_p", default_for_compare)):
            self.model.dmd_context_mix_p = p
        return p

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
    # CPU-only ride prefetch (executor + queued future). Worker thread
    # does the disk reads in ``_load_ride_tensors_cpu_part`` — no CUDA
    # work, no ss_vae forward. Main thread does ss_vae + H2D inline at
    # consume time on the default CUDA stream. Why: multi-thread CUDA
    # on the default stream serializes, so a worker doing GPU work
    # wouldn't overlap; disk I/O is the dominant cost (~50-200ms) and
    # threads cleanly amortize it across the previous step's compute.
    # ------------------------------------------------------------------
    # Hard timeout on prefetch wait. If the worker stalls (NFS hiccup,
    # slow disk, deadlocked dataset code) the entire trainer would
    # hang silently without this — no log, no error. 60s is well
    # above any healthy disk-read time but well below NCCL's default
    # 30-min collective timeout, so we surface the stall as a clean
    # exception (with traceback on every rank that hits it) before
    # downstream collectives time out and produce confusing errors.
    _PREFETCH_FUTURE_TIMEOUT_S: float = 60.0

    # Consecutive-None guard. ``_streaming_step`` returns None when
    # the ride loader can't produce a usable ride (all rides too
    # short, motion-filter rejects everything, prefetch CPU stage
    # exhausted attempts). The outer trainer treats None as "skip
    # optim, increment step, continue" — so a permanent loader fault
    # would silently waste the entire training run. Raise after this
    # many consecutive None returns so the run dies loudly with
    # diagnostics instead.
    _MAX_CONSECUTIVE_STREAMING_NONE: int = 10

    def _meta_passes_filters(
        self, meta: dict, rollout_frames: int,
    ) -> bool:
        """Shared min-frames + motion-magnitude filter for both the
        prefetch worker (``_next_ride_cpu_only``) and the legacy
        synchronous loader (``_next_ride``). Single source of truth
        so the two paths can't drift apart on filter logic.
        """
        n_latent_frames = int(meta.get("n_latent_frames", 0))
        if n_latent_frames < rollout_frames:
            return False
        skip_dead = bool(getattr(self.config, "motion_skip_dead_rides", True))
        if not skip_dead:
            return True
        ride_min_mean = float(getattr(self.config, "motion_ride_min_mean", 3.0))
        mag_loader = getattr(self.dataset, "load_motion_magnitudes", None)
        if mag_loader is None:
            return True
        try:
            mag = mag_loader(meta["zarr_path"], n_latent_frames)
        except Exception as e:
            # Best-effort: missing motion file shouldn't block the
            # ride; log and accept. (Same semantics as the legacy
            # ``_next_ride`` filter, kept centralized.)
            logging.warning(
                "motion pre-check failed for %s: %s — accepting ride "
                "without filter", meta.get("zarr_path", "?"), e,
            )
            return True
        return float(mag.mean()) >= ride_min_mean

    def _ensure_prefetch_executor(self) -> None:
        """Lazy-init the single-worker thread pool that does CPU ride
        prefetch, plus the slot that holds the pending Future. Also
        registers an ``atexit`` shutdown so the worker thread doesn't
        leak on SIGKILL / unhandled exception in another thread (CI
        produces a clean process exit instead of "process won't exit"
        zombies)."""
        if not hasattr(self, "_prefetch_executor") or self._prefetch_executor is None:
            self._prefetch_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="ride_prefetch",
            )
            atexit.register(self._prefetch_executor.shutdown, wait=False)
        if not hasattr(self, "_prefetched_ride_future"):
            self._prefetched_ride_future = None

    def _kick_ride_prefetch_if_idle(self, rollout_frames: int) -> None:
        """Submit a CPU-side ride load to the worker thread if no
        prefetch is already in flight. Idempotent — no-op if a future
        is already queued. Respects ``self.max_ride_frames`` so the
        prefetched RAM is bounded by the actually-reachable size, not
        the raw zarr length.
        """
        self._ensure_prefetch_executor()
        if self._prefetched_ride_future is not None:
            return
        max_frames = getattr(self, "max_ride_frames", None)
        self._prefetched_ride_future = self._prefetch_executor.submit(
            self._next_ride_cpu_only, rollout_frames, max_frames,
        )

    def _next_ride_cpu_only(
        self, rollout_frames: int, max_frames: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        """Run on the worker thread. Pulls the next ride from
        ``self._ride_iter`` (single-writer invariant: only this method
        ever advances the iterator) and runs the CPU-side disk reads.
        Returns the cpu_part dict that ``_finalize_ride_to_gpu`` will
        consume on the main thread, or None if the loader is exhausted
        after ``max_attempts`` rejected rides.
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
            if not self._meta_passes_filters(meta, rollout_frames):
                attempts += 1
                continue
            cpu_part = _load_ride_tensors_cpu_part(
                self.dataset, meta, max_frames=max_frames,
            )
            if cpu_part is None or cpu_part["latents_cpu"].shape[0] < rollout_frames:
                attempts += 1
                continue
            return cpu_part
        return None

    @staticmethod
    def _is_dataloader_worker_death(exc: BaseException) -> bool:
        """Heuristic: did this exception come from a PyTorch DataLoader
        fork-worker dying mid-fetch? PyTorch's signal_handling raises a
        plain ``RuntimeError`` whose message starts with "DataLoader
        worker (pid …) exited unexpectedly". The traceback may surface
        the error wherever the SIGCHLD handler fires (often deep inside
        a flash-attention forward), so we match on the message rather
        than on the call stack.
        """
        if not isinstance(exc, RuntimeError):
            return False
        msg = str(exc)
        return (
            "DataLoader worker" in msg
            and "exited unexpectedly" in msg
        )

    def _consume_or_load_ride(
        self, rollout_frames: int,
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], float]:
        """Block on the queued prefetch (kick one synchronously if
        none queued — cold-start path), run ss_vae + H2D on the main
        thread, return ``(ride_dict, prefetch_wait_ms)``.

        ``prefetch_wait_ms`` is the time spent BLOCKED on the worker's
        future (= time the prefetch couldn't keep up with the reset
        rate). Caller logs it as ``streaming_prefetch_wait_ms`` so we
        can tell apart "ss_vae + H2D dominates" (high setup_ms, low
        wait_ms) from "prefetch can't keep up" (high wait_ms).

        Hard 60s timeout on the future. If the worker stalls, we raise
        a clear ``TimeoutError`` instead of letting the trainer hang
        silently until NCCL collectives time out 30 min later.

        Defensive single-retry on PyTorch DataLoader fork-worker death.
        With ``num_workers > 0`` the dataloader fork-spawns CPU helpers
        that occasionally die from transient OS signals (page-cache
        eviction race during zarr open, NFS hiccup, OOM-killer drive-by,
        ...). PyTorch surfaces those deaths as
        ``RuntimeError("DataLoader worker (pid N) exited unexpectedly")``
        from the next ``next(self._ride_iter)`` call. We catch that
        once, recreate ``self._ride_iter`` (which tears down the dead
        worker pool and forks fresh), kick a new prefetch, and retry.
        Cap at one retry so a persistent fault still surfaces.

        ``self._ride_iter`` is touched ONLY by the executor's worker
        thread, never from the main thread directly — even on cold
        start (we ``submit`` and immediately await rather than calling
        ``next()`` ourselves) to preserve the single-writer invariant.
        On retry, the iterator-reset itself runs on the main thread but
        only AFTER the executor's task has already returned (raised),
        so single-writer is preserved.
        """
        self._ensure_prefetch_executor()
        if self._prefetched_ride_future is None:
            self._kick_ride_prefetch_if_idle(rollout_frames)
        fut = self._prefetched_ride_future
        self._prefetched_ride_future = None
        t0 = time.monotonic()
        try:
            cpu_part = fut.result(timeout=self._PREFETCH_FUTURE_TIMEOUT_S)
        except FuturesTimeoutError as exc:
            raise TimeoutError(
                f"ride prefetch stalled for >"
                f"{self._PREFETCH_FUTURE_TIMEOUT_S:.0f}s — disk hiccup, "
                f"deadlocked worker, or dataset code. Surfacing before "
                f"NCCL collective timeouts produce a confusing error."
            ) from exc
        except RuntimeError as exc:
            if not self._is_dataloader_worker_death(exc):
                raise
            logging.warning(
                "[ActionForcing] DataLoader worker died (transient "
                "subprocess fault): %s. Recreating ride iterator and "
                "retrying once.",
                exc,
            )
            try:
                self._ride_iter = self._fresh_ride_iter(self._epoch)
            except Exception as reset_exc:  # pragma: no cover
                logging.error(
                    "[ActionForcing] Failed to recreate ride iterator "
                    "after worker death: %s", reset_exc,
                )
                raise
            self._kick_ride_prefetch_if_idle(rollout_frames)
            retry_fut = self._prefetched_ride_future
            self._prefetched_ride_future = None
            try:
                cpu_part = retry_fut.result(
                    timeout=self._PREFETCH_FUTURE_TIMEOUT_S,
                )
            except FuturesTimeoutError as exc2:
                raise TimeoutError(
                    f"ride prefetch retry stalled for >"
                    f"{self._PREFETCH_FUTURE_TIMEOUT_S:.0f}s after "
                    f"DataLoader-worker recovery."
                ) from exc2
            except RuntimeError as exc2:
                if self._is_dataloader_worker_death(exc2):
                    raise RuntimeError(
                        "DataLoader worker died TWICE in a row "
                        "(after iterator recreation). This is no "
                        "longer a transient fault — investigate "
                        "dmesg, CPU RAM pressure, or a corrupted "
                        "ride file."
                    ) from exc2
                raise
        wait_ms = (time.monotonic() - t0) * 1000.0
        if cpu_part is None:
            return None, wait_ms
        ride = _finalize_ride_to_gpu(
            cpu_part, self.dataset, self.device, self.dtype,
            action_dims=self.action_dims,
        )
        return ride, wait_ms

    # ------------------------------------------------------------------
    # K=1-per-step state machine. Each call rolls one chunk on the
    # current ride (or sets up a fresh ride if reset was triggered last
    # step), trains every active head, and decides whether to reset
    # before the next call. Persistent ``self.model.streaming_state``
    # carries KV cache + ``ride_*_window`` across calls.
    # ------------------------------------------------------------------
    def _streaming_step(
        self,
        rollout_frames: int,
        cf_dmdctx: int,
    ) -> Optional[Dict[str, Any]]:
        """One outer training step in K=1 mode.

        Returns a flat ``str -> float`` log dict (or None if the ride
        loader is exhausted on a setup-required step). The trainer's
        outer ``optim.step()`` runs after this returns.
        """
        cfg = self.config
        max_rolls = int(getattr(cfg, "max_rolls_per_ride", 60))
        force_exit_step_enabled = bool(
            getattr(cfg, "force_exit_step_enabled", False)
        )

        out: Dict[str, Any] = {}
        t_step_start = time.monotonic()
        t_setup_ms = 0.0
        prefetch_wait_ms = 0.0

        # ---- Stage 1: set up a fresh ride if needed ------------------
        needs_setup = self.model.streaming_state is None
        if needs_setup:
            t_setup_start = time.monotonic()
            ride, prefetch_wait_ms = self._consume_or_load_ride(
                rollout_frames + cf_dmdctx,
            )
            if ride is None:
                # Loader exhausted on this step. Outer trainer treats
                # None as "skip optim, increment step, continue". The
                # consecutive-None guard below catches a permanent
                # loader fault before silently wasting the entire run.
                return self._handle_streaming_step_none(
                    "ride_loader_exhausted_on_setup",
                )
            ok = self._streaming_setup_sequence_from_ride(
                rollout_frames=rollout_frames,
                max_total_rollout_frames=rollout_frames,
                cf_dmdctx=cf_dmdctx,
                ride=ride,
            )
            if not ok:
                return self._handle_streaming_step_none(
                    "setup_sequence_rejected_ride",
                )
            self._chunks_in_current_ride = 0
            t_setup_ms = (time.monotonic() - t_setup_start) * 1000.0
        # Reset the consecutive-None counter on any successful setup
        # OR continuing-from-existing-state path.
        self._consecutive_streaming_step_none = 0

        # ---- Kick prefetch for the NEXT ride (overlaps this step's
        # backward with the next ride's disk I/O). Idempotent.
        self._kick_ride_prefetch_if_idle(rollout_frames + cf_dmdctx)

        state = self.model.streaming_state
        chunk_size = int(state["chunk_size"])
        cf_state = int(state["cf"])

        # ---- Optional per-step force_exit_step broadcast (config-gated).
        # Default OFF — under K=1 the inner ``generate_and_sync_list(sync=True)``
        # already broadcasts once per call on every rank, matched
        # count, so cross-rank exit-rung lockstep is preserved without
        # outer plumbing. Kept as a knob for variance-reduction
        # experiments that want to pin a single rank-0-picked rung.
        forced_exit_step: Optional[int] = None
        if force_exit_step_enabled:
            pipe = self.model.inference_pipeline
            n_steps = len(pipe.denoising_step_list)
            if pipe.last_step_only:
                forced_exit_step = n_steps - 1
            elif dist.is_initialized() and dist.get_world_size() > 1:
                if dist.get_rank() == 0:
                    idx_t = torch.randint(
                        0, n_steps, (1,),
                        device=self.device, dtype=torch.long,
                    )
                else:
                    idx_t = torch.empty(
                        1, dtype=torch.long, device=self.device,
                    )
                dist.broadcast(idx_t, src=0)
                forced_exit_step = int(idx_t.item())
            else:
                forced_exit_step = int(torch.randint(0, n_steps, (1,)).item())

        # ---- Stage 2: roll one chunk ---------------------------------
        # ``compute_baseline_mae=False`` skips the pipeline's internal
        # gt_chunk slice + ``_compute_chunk_mae`` (which would also
        # fire a DDP all_reduce). Under K=1-per-step that all_reduce
        # would be matched-count-safe, but it's redundant work — we
        # compute the per-rank MAE on this same window in Stage 3
        # below using the ride_latents_window slice we already hold.
        # Kept to skip the redundant compute, NOT for deadlock-avoidance.
        t_rollout_start = time.monotonic()
        chunk, info = self.model.generate_next_chunk(
            requires_grad=True,
            compute_baseline_mae=False,
            force_exit_step=forced_exit_step,
        )
        self._chunks_in_current_ride = (
            getattr(self, "_chunks_in_current_ride", 0) + 1
        )

        # ---- Stage 3: per-rank MAE on the chunk_size-frame trained window.
        noisy_start_sdn = int(
            info["current_length"] - info["new_frames"] - info["overlap"]
        )
        chunk_lo = cf_state + noisy_start_sdn
        chunk_hi = chunk_lo + chunk_size
        ride_window = state["ride_latents_window"]
        # Geometry invariant: setup_sequence builds ride_latents_window of
        # length cf + actual_cap, ``can_generate_more()`` keeps
        # current_length ≤ max_length=actual_cap, and chunk_hi simplifies
        # to cf_state + current_length ≤ ride_window.shape[1].
        assert ride_window.shape[1] >= chunk_hi, (
            f"streaming-step geometry violation: ride_window.shape[1]="
            f"{ride_window.shape[1]} < chunk_hi={chunk_hi}."
        )
        gt_slice = ride_window[:, chunk_lo:chunk_hi]
        avg_mae = float(
            (chunk.detach().float() - gt_slice.float()).abs().mean().item()
        )
        t_rollout_ms = (time.monotonic() - t_rollout_start) * 1000.0

        out["streaming_chunks_in_ride"] = float(self._chunks_in_current_ride)
        out["streaming_window_avg_mae"] = float(avg_mae)
        out["streaming_did_setup_this_step"] = 1.0 if needs_setup else 0.0
        out["streaming_window_start_chunk"] = float(self._chunks_in_current_ride)

        # ---- Stage 4: train every active head -----------------------
        t_train_start = time.monotonic()
        self._streaming_train_one_chunk(
            chunk, info, avg_mae,
            state=state, cf_state=cf_state, chunk_size=chunk_size,
            out=out,
        )
        t_train_ms = (time.monotonic() - t_train_start) * 1000.0

        # ---- Stage 5: reset decision for NEXT step ------------------
        # The reset decision MUST be lockstep across ranks. If even one
        # rank resets, that rank's NEXT step opens a fresh sequence —
        # which fires an anchor forward inside ``setup_sequence``
        # (= one extra ``generate_chunk_with_cache`` call → one extra
        # ``generate_and_sync_list`` broadcast). Other ranks that DIDN'T
        # reset would skip the anchor's broadcast, leaving NCCL
        # collectives mismatched in op-count → user-code-paired
        # broadcasts cross-pair the wrong calls → hang on the next
        # mismatched-size collective. We MAX-reduce the per-rank
        # ``should_reset`` flag so ANY-rank-reset triggers ALL-ranks-
        # reset; the cost is a few resets that some ranks didn't
        # individually need (their ride state still gets thrown away),
        # but that's strictly better than a hang.
        # MAE-collapse reset removed (the MAE-extension path was
        # deleted; we no longer monitor MAE for ride resets).
        # The cap + exhausted gates still fire normally.
        local_hit_cap = self._chunks_in_current_ride >= max_rolls
        local_exhausted = not self.model.can_generate_more()
        local_should_reset = local_hit_cap or local_exhausted
        if dist.is_initialized() and dist.get_world_size() > 1:
            flag_t = torch.tensor(
                [1 if local_should_reset else 0],
                device=self.device, dtype=torch.long,
            )
            dist.all_reduce(flag_t, op=dist.ReduceOp.MAX)
            should_reset = bool(int(flag_t.item()))
        else:
            should_reset = local_should_reset

        # Telemetry: log the LOCAL reason so we can tell apart "this
        # rank actually wanted to reset" from "peer triggered the
        # reset". Both ranks always agree on the boolean now.
        if should_reset:
            out["streaming_did_reset"] = 1.0
            # MAE-collapse reset removed alongside the MAE-extension
            # path. Only cap / end-of-ride / peer-triggered reasons
            # remain.
            if local_hit_cap:
                out["streaming_reset_reason_cap"] = 1.0
            elif local_exhausted:
                out["streaming_reset_reason_end_of_ride"] = 1.0
            else:
                out["streaming_reset_reason_peer_triggered"] = 1.0
            self.model.reset_streaming_state()
        else:
            out["streaming_did_reset"] = 0.0

        # Wall-clock telemetry. Lets us tell apart two failure modes
        # of the prefetch design:
        #   * High setup_ms with small prefetch_wait_ms → ss_vae forward
        #     / H2D dominates. The "premature optimization to use
        #     CUDA streams" call was right; revisit if it's bad enough.
        #   * High setup_ms WITH high prefetch_wait_ms → the worker
        #     can't keep up with the reset rate (disk too slow, or
        #     ride pulls too expensive). Revisit prefetch design.
        out["streaming_step_setup_ms"] = float(t_setup_ms)
        out["streaming_prefetch_wait_ms"] = float(prefetch_wait_ms)
        out["streaming_step_rollout_ms"] = float(t_rollout_ms)
        out["streaming_step_train_ms"] = float(t_train_ms)
        out["streaming_step_wallclock_ms"] = float(
            (time.monotonic() - t_step_start) * 1000.0
        )

        return out

    def _handle_streaming_step_none(self, reason: str) -> None:
        """Account for a None-return from ``_streaming_step``. The
        outer trainer interprets None as "skip optim, increment step,
        continue", which is harmless on a single bad ride but masks a
        permanent loader fault (e.g. dataset config wrong, all rides
        too short, motion filter rejecting everything). Track
        consecutive Nones and raise after ``_MAX_CONSECUTIVE_STREAMING_NONE``
        so the run dies loudly instead of silently wasting time.
        """
        self._consecutive_streaming_step_none = (
            getattr(self, "_consecutive_streaming_step_none", 0) + 1
        )
        if self._consecutive_streaming_step_none > self._MAX_CONSECUTIVE_STREAMING_NONE:
            raise RuntimeError(
                f"_streaming_step returned None for "
                f"{self._consecutive_streaming_step_none} consecutive "
                f"calls (last reason: {reason!r}). Ride loader appears "
                f"permanently exhausted — check dataset config, "
                f"min_ride_frames, motion_ride_min_mean, and ensure "
                f"streaming_max_length / max_ride_frames align with "
                f"max_rolls_per_ride."
            )
        return None

    # ------------------------------------------------------------------
    # Per-chunk training block. Called once per outer training step
    # from ``_streaming_step``: every active head (gen, DMD critic,
    # aux action critic, R3GAN, SC-DMD) fires on every step. K=1 per
    # step makes every collective matched-count across ranks; no
    # no_sync wrapping, no loss scaling, no gating needed.
    # ------------------------------------------------------------------
    def _streaming_train_one_chunk(
        self,
        train_chunk: torch.Tensor,
        train_info: Dict[str, Any],
        train_avg_mae: float,
        *,
        state: Dict[str, Any],
        cf_state: int,
        chunk_size: int,
        out: Dict[str, Any],
    ) -> None:
        """Run gen DMD + aux gen-side + R3GAN gen-side + SC-DMD weighted
        + gen.backward + DMD critic + critic.backward on the trained
        chunk. Aux's inner action-critic optim and GAN's inner D
        optim each fire once per call (= once per outer trainer step)
        — same all-reduce count on every rank.
        """
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

        _sample_due_now = self._video_sample_due(int(self.step) + 1)
        if _sample_due_now:
            try:
                # Prefer the post-Step-3.3.5 refined cache_pred from
                # the pipeline's ``_clean_chunk`` buffer — cleanest
                # available student state at t=60 (vs ``train_chunk``
                # which is the random-exit-rung output at a noisier,
                # variable t). Falls back to ``train_chunk`` when
                # ``flash_dmd_enabled=False`` (buffer is None).
                _clean = getattr(
                    self.model.inference_pipeline, "_clean_chunk", None,
                )
                if _clean is not None:
                    self._pending_video_latents = (
                        _clean.detach().to(torch.float32)
                    )
                else:
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
                    "[ActionForcing] failed to stash trained chunk "
                    "for video at step=%d: %s",
                    int(self.step) + 1, _exc,
                )
                self._pending_video_latents = None
            self.model._dmd_eval_stash = {}
        else:
            self.model._dmd_eval_stash = None

        # ---- memory audit boundary 0: entry to per-chunk training ----
        self._mem_step_snaps = []
        self._mem_step_snapshot("0_entry")

        # Plumb the trainer's current step into ``info`` so the model
        # can resolve step-dependent schedules (e.g. the aux teacher's
        # piecewise-linear ``real_teacher_input_mix_gt_p`` schedule)
        # without separately threading the step through the call
        # signature.
        train_info["current_step"] = int(self.step)
        gen_loss_dmd, gen_log = self.model.compute_generator_loss_streaming(
            train_chunk, train_info,
        )
        self._mem_step_snapshot("1_after_compute_gen_loss_streaming")

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
        # Per-rung bin: split the actual gen-side DMD push (= the
        # signal that backprops into the student) by the active DMD
        # timestep. ``dmd_t_mean`` is set by ``_compute_kl_grad`` and
        # Student-pred latent stats — abs-max / RMS over the chunk the
        # gen step was just trained on. If RMS climbs unboundedly or
        # abs_max saturates near the VAE's latent range, the student
        # is collapsing in latent space — a leading indicator that
        # decode-time output will go grey/blocky in the next few iters.
        # Prefer the post-Step-3.3.5 refined cache_pred (cleanest
        # student state at t=60) when available; fall back to
        # ``train_chunk`` (random-exit-rung output) when
        # ``flash_dmd_enabled=False``. More representative of
        # inference-time output quality.
        with torch.no_grad():
            _clean = getattr(
                self.model.inference_pipeline, "_clean_chunk", None,
            )
            _tc = (_clean if _clean is not None else train_chunk).detach().float()
            out["student_pred_rms"] = float(_tc.pow(2).mean().sqrt().item())
            out["student_pred_abs_max"] = float(_tc.abs().max().item())
            out["student_pred_mean"] = float(_tc.mean().item())

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
            # Route action_critic to flash_dmd_gan_x0 (=
            # flash_dmd_gan_x0, the t=flash_dmd_gan_t gen forward) when
            # available. Without this, action_critic supervises the gen
            # at whatever random exit-rung was sampled this iter (75%
            # of iters land at high-noise rungs, where pred_x0 is a
            # blurry/uncertain estimate). Routing to the final-step
            # forward gives the critic a clean, structurally-coherent
            # input every iter — same trick the GAN already uses via
            # fake_lat_grad. Falls back to train_chunk when paper_
            # aligned is off or this iter didn't emit it. Gated by
            # ``gen_aux_losses_use_paper_aligned_x0`` (default True).
            ac_pred_x0 = train_chunk
            if bool(getattr(
                self.config, "gen_aux_losses_use_paper_aligned_x0", True,
            )):
                _pa_x0 = train_info.get("flash_dmd_gan_x0")
                if _pa_x0 is not None:
                    ac_pred_x0 = _pa_x0
            gen_action_loss, critic_logs, teacher_z_8d = (
                self._compute_action_critic_losses(
                    pred_x0=ac_pred_x0,
                    target_action_z=actions_for_critic,
                    chunk_t=chunk_t,
                    current_step=int(self.step),
                )
            )
            generator_loss = generator_loss + gen_action_loss
            out.update(critic_logs)
            self._mem_step_snapshot("2_after_action_critic")

            # LoRA-side action losses (action_critic z-guidance on the
            # LoRA's denoised x0 + state_probe MSE on the per-chunk z
            # the probe reads from real_score's internal taps). Both
            # graph-bearing tensors come from the model's aux pass via
            # ``_aux_teacher_tensors`` in the gen_log dict. Gradient
            # flows: action_critic → LoRA params via lora_x0; state_probe
            # → LoRA params via taps + state_probe params via the probe
            # readout. Skipped when aux pass produced no tensors (= aux
            # was skipped this iter, e.g. end-of-ride).
            aux_tensors = gen_log.get("_aux_teacher_tensors") if isinstance(
                gen_log, dict,
            ) else None
            if aux_tensors is not None:
                lora_action_loss, lora_state_probe_loss, lora_logs = (
                    self._compute_lora_action_losses(
                        lora_x0=aux_tensors.get("lora_x0"),
                        lora_state_preds=aux_tensors.get("lora_state_preds"),
                        target_action_z=actions_for_critic,
                        teacher_z_8d=teacher_z_8d,
                        chunk_t=chunk_t,
                        current_step=int(self.step),
                    )
                )
                if lora_action_loss is not None:
                    generator_loss = generator_loss + lora_action_loss
                if lora_state_probe_loss is not None:
                    generator_loss = generator_loss + lora_state_probe_loss
                out.update(lora_logs)
                self._mem_step_snapshot("3_after_lora_action")

        # Constant-weight latent-space MAE + MSE between student's
        # pred_image (= ``train_chunk``) and the GT latent window. Fires
        # every step when the weights are set, no warmup/ramp.
        if (
            self.gt_latent_mae_loss_weight > 0.0
            or self.gt_latent_mse_loss_weight > 0.0
        ):
            _gt_window_for_recon = (
                state["ride_latents_window"][:, chunk_lo:chunk_hi]
            )
            recon_loss, recon_logs = self._compute_gt_latent_recon_loss(
                pred_image=train_chunk,
                gt_latents_window=_gt_window_for_recon,
            )
            generator_loss = generator_loss + recon_loss
            out.update(recon_logs)

        if gan_active:
            gt_window = state["ride_latents_window"][:, chunk_lo:chunk_hi]
            # Stash the latent chunk_lo on streaming_state so
            # ``_run_distilled_disc_critic_update`` can index into the
            # cached uint8 pixel window (when present) without re-
            # plumbing the chunk geometry through 3 function signatures.
            self.model.streaming_state["last_chunk_lo_in_ride_window"] = chunk_lo
            self.model.streaming_state["last_chunk_size"] = chunk_size
            # Flash-DMD: the model stashes the per-block t=flash_dmd_gan_t
            # gen forward output on ``info["flash_dmd_gan_x0"]``; pass it
            # through so the disc and gen-side aux losses consume the
            # near-clean output as the G-side fake.
            _flash_dmd_gan_x0 = train_info.get("flash_dmd_gan_x0")
            gen_gan_loss, gan_logs = self._compute_r3gan_losses(
                pred_image=train_chunk,
                gt_latents_window=gt_window,
                current_step=int(self.step),
                flash_dmd_gan_x0=_flash_dmd_gan_x0,
            )
            generator_loss = generator_loss + gen_gan_loss
            out.update(gan_logs)
            self._mem_step_snapshot("4_after_gan_pre_backward")

        # Pixel-space perceptual losses (LPIPS + pixel reconstruction).
        # Direct anti-blur + anti-drift signals on a graph-on decoded
        # frame subset. Independent of the GAN distillation chain
        # (no critic involved). Gated by their own weight knobs;
        # zero-weight = no compute. See ``_compute_pixel_perceptual_losses``.
        if (
            self.lpips_loss_weight > 0
            or self.pixel_recon_loss_weight > 0
        ):
            # Defragment the allocator before the graph-on VAE decode
            # — the decoder peaks at a large contiguous workspace
            # (~1 GB) and the gen rollout's persistent activations
            # leave only fragmented gaps. Without this, the allocation
            # can fail at step 2+ even though total free memory is
            # sufficient.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            perc_loss, perc_logs = self._compute_pixel_perceptual_losses(
                pred_image=train_chunk,
                gt_latents_window=gt_window,
                current_step=int(self.step),
            )
            generator_loss = generator_loss + perc_loss
            out.update(perc_logs)

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
        #
        # DDP-safe skip when generator_loss has no autograd graph.
        # Happens at production early steps (DMD/aux/GAN all gated off
        # via dmd_loss_start_step / aux_teacher_start_step /
        # gan_critic_warmup_steps) — the sum of 0-weighted losses
        # collapses to a leaf zero tensor with no grad_fn. Backward
        # would raise "element 0 of tensors does not require grad".
        # Mathematically there's no gen gradient to compute (every
        # contributor is zero-weighted), so skipping ``.backward()``
        # produces a bit-identical optimizer step (gen params get no
        # update either way). DDP-lockstep enforced via all_reduce —
        # if ANY rank has grad, ALL ranks must run backward together
        # to keep the gradient bucket reduction count matched.
        gen_has_grad_local = (
            1.0 if generator_loss.requires_grad else 0.0
        )
        if dist.is_initialized() and dist.get_world_size() > 1:
            flag = torch.tensor(
                [gen_has_grad_local],
                device=generator_loss.device, dtype=torch.float32,
            )
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            gen_should_backward = bool(flag.item() > 0.5)
        else:
            gen_should_backward = bool(gen_has_grad_local > 0.5)
        out["gen_backward_skipped"] = (
            0.0 if gen_should_backward else 1.0
        )
        if gen_should_backward:
            generator_loss.backward(retain_graph=True)
        self._mem_step_snapshot("5_after_gen_backward")

        # ----- noise_aux training step -----
        # Independent of LoRA/gen optimizers. Uses the gen step's stashes
        # (pred_image, pred_real) and trains noise_aux to learn the
        # AR-noise direction. Fires every gen step; tiny model (~8M
        # params), small memory footprint vs the rest of the graph.
        # Skipped silently when stashes aren't available (e.g. first
        # iter) or when noise_aux is disabled.
        if (
            self.noise_aux_enabled
            and self.noise_aux_optimizer is not None
        ):
            na_loss, na_log = self.model.compute_noise_aux_loss_streaming(
                current_step=int(self.step),
            )
            if na_loss is not None and na_loss.requires_grad:
                na_loss.backward()
                na_params_with_grad = [
                    p for p in self.noise_aux_optimizer.param_groups[0]["params"]
                    if p.grad is not None
                ]
                if na_params_with_grad:
                    nagn = torch.nn.utils.clip_grad_norm_(
                        na_params_with_grad,
                        max_norm=self.noise_aux_max_grad_norm,
                    )
                    out["noise_aux_grad_norm"] = (
                        float(nagn.item()) if torch.is_tensor(nagn) else float(nagn)
                    )
                    self.noise_aux_optimizer.step()
                self.noise_aux_optimizer.zero_grad(set_to_none=True)
            out.update({
                k: (float(v.detach().float().mean().item())
                    if torch.is_tensor(v) else v)
                for k, v in na_log.items()
                if not isinstance(v, dict)
            })
        self._mem_step_snapshot("5b_after_noise_aux")

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
        self._mem_step_snapshot("6_after_critic_backward")

        # teacher_cadence='fake': run ``dfake_gen_update_ratio`` total
        # LoRA optimizer steps per outer iter (vs the default 1 under
        # 'student'). Cycle i=0 consumes the gradient already on the
        # LoRA params from the main gen backward above (which included
        # the gen-iter's aux pass). Cycles i=1..n-1 each do a fresh
        # aux forward+backward on the SAME detached chunk with new
        # (ε, t), then step and zero. After this loop LoRA grads are
        # None, so the outer loop's real_teacher_optimizer.step() at
        # ~line 2084 finds nothing to apply and short-circuits.
        # DDP-safe: every rank runs the same number of cycles; the
        # short-ride skip inside ``_compute_aux_teacher_loss_streaming``
        # is already all_reduce(MAX)-synced across ranks.
        teacher_cadence = str(
            getattr(self.config, "teacher_cadence", "student")
        ).lower()
        if (
            teacher_cadence == "fake"
            and getattr(self, "real_teacher_optimizer", None) is not None
        ):
            n_total = int(
                getattr(self.config, "dfake_gen_update_ratio", 1)
            )
            _aux_start = int(getattr(self.config, "aux_teacher_start_step", 0))
            _aux_open = self.step >= _aux_start
            if _aux_open and n_total >= 1:
                # Match the outer loop's LR-warmup schedule so every
                # in-loop step uses the same LR that the outer loop
                # would have applied.
                warm_lr = self._real_teacher_base_lr
                if (
                    self.real_teacher_warmup_steps > 0
                    and self.step < self.real_teacher_warmup_steps
                ):
                    warm_lr = self._real_teacher_base_lr * (
                        (self.step + 1) / self.real_teacher_warmup_steps
                    )
                for pg in self.real_teacher_optimizer.param_groups:
                    pg["lr"] = warm_lr

                _fired = 0
                for i in range(n_total):
                    if i > 0:
                        aux_loss_extra, aux_log_extra = (
                            self.model.run_extra_aux_pass(
                                train_chunk.detach(), train_info,
                            )
                        )
                        if (
                            aux_loss_extra is None
                            or not aux_loss_extra.requires_grad
                        ):
                            continue
                        aux_loss_extra.backward()
                    rt_params = [
                        p
                        for p in self.real_teacher_optimizer
                            .param_groups[0]["params"]
                        if p.grad is not None
                    ]
                    if rt_params:
                        torch.nn.utils.clip_grad_norm_(
                            rt_params,
                            max_norm=self.real_teacher_max_grad_norm,
                        )
                        self.real_teacher_optimizer.step()
                        # Target-network EMA pull after every LoRA
                        # update inside the teacher_cadence='fake'
                        # inner loop. With dfake_gen_update_ratio=5
                        # this fires 5x per outer step. No-op when
                        # real_score_ema_weight == 0. The rel_l2
                        # diagnostic only fires on the LAST inner
                        # iter to avoid 4× redundant per-param sums
                        # and the GPU→CPU .item() sync; wandb only
                        # reads the metric once per outer step
                        # anyway, so only the latest value matters.
                        self.model.ema_update_real_score_lora(
                            compute_rel_l2=(i == n_total - 1),
                        )
                        _fired += 1
                    self.real_teacher_optimizer.zero_grad(set_to_none=True)
                out["teacher_cadence_steps_fired"] = float(_fired)
                self._mem_step_snapshot("7_after_teacher_cadence_fake_loop")

        # Distilled-critic deferred step. Path 1 (D-update) + Path 2
        # (critic value+grad distillation) were stashed by the gen
        # step's ``_compute_r3gan_losses_distilled``. Run them here
        # AFTER both gen and action-critic backwards. Note: the gen
        # backward uses ``retain_graph=True`` (so the action-critic
        # backward can walk the shared cond_dict / action_projection
        # subgraph) which keeps gen activations alive even after the
        # action-critic backward completes. Explicitly empty the
        self._mem_step_snapshot("8_exit")

    # ------------------------------------------------------------------
    # Streaming-mode helpers (LongLive parity).
    # ------------------------------------------------------------------
    def _streaming_setup_sequence_from_ride(
        self,
        rollout_frames: int,
        max_total_rollout_frames: int,
        cf_dmdctx: int,
        *,
        ride: Optional[Dict[str, torch.Tensor]] = None,
    ) -> bool:
        """Set up a fresh streaming sequence on this rank.

        ``ride``: when provided, use this pre-loaded ride directly (the
        K=1-per-step path passes the prefetched ride consumed via the
        executor). When None, do a synchronous ``_next_ride`` pull
        (legacy code paths).

        Per-rank ``actual_cap`` is each rank's own — no MIN-reduce.
        Under K=1-per-step there are no per-call DDP collectives that
        depend on a shared max_length across ranks. Each rank's reset
        decision (MAE / cap / end-of-ride) is independent, so ride
        lengths and rolling-window caps can diverge freely.

        Returns True on success, False if no ride was available or if
        the loaded ride is too short for one valid trained chunk.
        """
        npb = int(getattr(self.config, "num_frame_per_block", 3))
        cap = int(self.streaming_max_length)
        if cap % npb != 0:
            cap = (cap // npb) * npb

        if ride is None:
            ride = self._next_ride(rollout_frames + cf_dmdctx)
            if ride is None:
                return False
        ride_len = int(ride["latents"].shape[1])

        # Per-rank motion-aware s pick. Each rank biases toward its
        # OWN ride's motion. ``rollout_frames`` is the window we want
        # motion in (= the first valid scoring window of the streaming
        # sequence).
        s_local_max = max(
            0, min(ride_len // 2, ride_len - cf_dmdctx - cap),
        )
        s = self._pick_motion_aware_offset(
            ride,
            s_local_max=s_local_max,
            window_lo_offset=cf_dmdctx,
            window_len=int(rollout_frames),
        )

        # Per-rank rolling-window cap: how many post-seed frames fit
        # in this rank's ride. Snap to npb. NO MIN-reduce — each rank
        # rolls until its own ride's MAE crosses or its own cap is
        # reached. Cross-rank lockstep is not required because every
        # collective in the per-step training (gen / fake_score backward
        # all-reduces, the inner pipeline's exit-flag broadcast) fires
        # exactly once per outer trainer iter on every rank by
        # construction.
        actual_cap = min(cap, ride_len - cf_dmdctx - s)
        if actual_cap % npb != 0:
            actual_cap = (actual_cap // npb) * npb
        # Reject if the ride can't fit the +npb anchor + at least one
        # valid ``generate_next_chunk`` call.
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

        # Telemetry: stash the ride's absolute zarr-latent offset ``s``
        # and its motion.npy chunk offset on streaming_state so the
        # clean_x_real video logger can render frame-level "zarr_lat=N
        # motion_chunk=M" annotations alongside the action-bar overlay.
        # Read motion offset from the dataset's per-ride attrs cache so
        # we don't re-open the zarr (zero disk I/O).
        zarr_path = ride.get("zarr_path", "")
        motion_chunk_offset = 0
        attrs_by_path = getattr(self.dataset, "_attrs_by_path", None)
        if attrs_by_path is not None and zarr_path in attrs_by_path:
            try:
                from utils.zarr_dataset import _motion_chunk_offset
                motion_chunk_offset = int(
                    _motion_chunk_offset(attrs_by_path[zarr_path])
                )
            except Exception as _exc:  # pragma: no cover
                logging.warning(
                    "[ActionForcing] motion-chunk offset lookup failed "
                    "for %s: %s — falling back to 0",
                    zarr_path, _exc,
                )
        if self.model.streaming_state is not None:
            self.model.streaming_state["ride_offset_s"] = int(s)
            self.model.streaming_state["zarr_path"] = zarr_path
            self.model.streaming_state["motion_chunk_offset"] = motion_chunk_offset
            # Pre-decoded GT pixels: when ``gt_pixel_cache_root`` is set
            # AND a cached zarr exists for this ride, load the pixel
            # slice corresponding to the active ride window. Stash on
            # streaming_state so ``_run_distilled_disc_critic_update``
            # can short-circuit the live VAE decode of GT. uint8 →
            # fp32 dequantization is deferred to the consumer (so we
            # store ~5 GB compressed instead of ~20 GB fp32). Falls
            # silently to live-decode when the cache file is missing.
            self.model.streaming_state["ride_pixels_window_uint8"] = (
                self._maybe_load_cached_gt_pixels(
                    zarr_path=zarr_path,
                    ride_offset_s=int(s),
                    cf_dmdctx=cf_dmdctx,
                    actual_cap=int(actual_cap),
                )
            )
        return True

    def _maybe_load_cached_gt_pixels(
        self,
        zarr_path: str,
        ride_offset_s: int,
        cf_dmdctx: int,
        actual_cap: int,
    ) -> Optional[torch.Tensor]:
        """Try to load a pre-decoded uint8 pixel slice for the active
        ride window. Returns ``[1, F_pix, 3, H, W]`` uint8 on cache hit,
        ``None`` on miss or when the cache root isn't configured.

        Indexing contract: the precompute script (``bin/precompute_gt_
        pixels.py``) writes ``pixels[F_pix, 3, H, W]`` indexed by
        DATASET latent indices (post-_LATENT_HEAD_DROP shift), with
        4× temporal expansion via the WAN VAE dummy-leading-frame
        trick. Latent index ``i`` (in the dataset's space) decodes to
        pixel frames ``[4*i, 4*(i+1))``. The active ride window is
        ``[ride_offset_s : ride_offset_s + cf_dmdctx + actual_cap]``,
        so the pixel slice we want is ``[ride_offset_s*4 : (ride_
        offset_s + cf + cap)*4]``.
        """
        cache_root = getattr(self.config, "gt_pixel_cache_root", None)
        if not cache_root:
            return None
        if not zarr_path:
            return None
        ride_name = Path(zarr_path).stem
        cache_zarr = Path(cache_root) / f"{ride_name}.zarr"
        if not cache_zarr.exists():
            return None
        try:
            import zarr as _zarr_lib
            g = _zarr_lib.open_group(str(cache_zarr), mode="r")
            pixels_arr = g["pixels"]  # uint8 [F_pix_total, 3, H, W]
            n_lat_total = int(g.attrs.get(
                "n_latent_frames",
                pixels_arr.shape[0] // 4,
            ))
            window_lat_lo = ride_offset_s
            window_lat_hi = ride_offset_s + cf_dmdctx + actual_cap
            if window_lat_hi > n_lat_total:
                if self.is_main_process:
                    logging.warning(
                        "[gt-pixel-cache] %s window [%d:%d) extends "
                        "past cached n_lat=%d — cache miss for safety.",
                        ride_name, window_lat_lo, window_lat_hi, n_lat_total,
                    )
                return None
            pix_lo = window_lat_lo * 4
            pix_hi = window_lat_hi * 4
            arr = pixels_arr[pix_lo:pix_hi]  # uint8 numpy
            t = torch.from_numpy(arr).unsqueeze(0)  # [1, F_pix, 3, H, W]
            t = t.contiguous()
            if self.is_main_process:
                logging.info(
                    "[gt-pixel-cache] HIT %s [lat %d:%d) → pix %d:%d) "
                    "(%.2f MB uint8)",
                    ride_name, window_lat_lo, window_lat_hi,
                    pix_lo, pix_hi,
                    t.numel() / (1024 ** 2),
                )
            return t
        except Exception as exc:
            if self.is_main_process:
                logging.warning(
                    "[gt-pixel-cache] load failed for %s: %s — "
                    "falling back to live decode.",
                    ride_name, exc,
                )
            return None

    def _fwdbwd_streaming_step(
        self,
        train_generator: bool,
        rollout_frames: int,
        max_total_rollout_frames: int,
        cf_dmdctx: int,
    ) -> Optional[Dict[str, Any]]:
        """Streaming-mode per-iter step (K=1 per step).

        The gen-iter path routes to ``_streaming_step`` which rolls
        one chunk on the persistent ride state, trains every active
        head (gen / DMD critic / aux / R3GAN / SC-DMD), and decides
        whether to reset the ride for the next step. The standalone
        critic-iter call is a no-op — the K=1 path already trained
        the critic on the same chunk.
        """
        if train_generator:
            return self._streaming_step(
                rollout_frames=rollout_frames,
                cf_dmdctx=cf_dmdctx,
            )
        # Critic-iter no-op: the K=1 _streaming_step already trained
        # the critic on this chunk. Running another critic update
        # here would either re-roll into a closed sequence or
        # re-train on a fresh ride (DMD2-decoupling violation).
        return {"streaming_critic_skipped": 1.0}

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

                # Constant-weight latent-space MAE + MSE between
                # pred_image and the GT latent window. Fires every step
                # when the weights are set, no warmup/ramp.
                if (
                    self.gt_latent_mae_loss_weight > 0.0
                    or self.gt_latent_mse_loss_weight > 0.0
                ):
                    _gt_window_for_recon = (
                        latents[:, gen_window_start:gen_window_end]
                    )
                    recon_loss, recon_logs = (
                        self._compute_gt_latent_recon_loss(
                            pred_image=pred_image,
                            gt_latents_window=_gt_window_for_recon,
                        )
                    )
                    generator_loss = generator_loss + recon_loss
                    merged.update(recon_logs)

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
        # Ride-level motion + min-frames filter shared with the K=1
        # prefetch path (``_next_ride_cpu_only``). See
        # ``_meta_passes_filters`` for the filter logic.
        attempts = 0
        max_attempts = 200
        ride_min_mean = float(
            getattr(self.config, "motion_ride_min_mean", 3.0)
        )
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
            if not self._meta_passes_filters(meta, rollout_frames):
                attempts += 1
                continue
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
                "[ActionForcing] _next_ride: gave up after %d attempts "
                "(filter threshold motion_ride_min_mean=%.3f); the "
                "dataset may have no rides with >=%d latent frames AND "
                "non-trivial motion. Consider lowering rollout_frames, "
                "raising the dataset's min_ride_frames filter, lowering "
                "motion_ride_min_mean, or setting motion_skip_dead_rides=false.",
                max_attempts, ride_min_mean, rollout_frames,
            )
        return None

    # ------------------------------------------------------------------
    # Override grad-uniformity (parent expects multi-slot path).
    # The parent's ``_all_reduce_extra_trainable_grads`` uses dist
    # reductions on action_projection / action_token_projection grads;
    # that part still applies here. We reuse it as-is.
    # ------------------------------------------------------------------


def _load_config_with_extends(path: str) -> "OmegaConf":
    """Load a yaml config, recursively resolving a top-level
    ``_extends: <path>`` directive.

    Behavior:
      * If the yaml has ``_extends: <other.yaml>`` at top level, the
        extended file is loaded first (recursively) and the current
        file's keys merge ON TOP via ``OmegaConf.merge``. This lets a
        variant config carry only the deltas from a base.
      * ``_extends`` is stripped from the final config so consumers
        never see it.
      * The path is resolved relative to the directory of the file
        containing the directive (Hydra-style).
      * Cycles are detected and raise.

    Used by the action_forcing config family to collapse
    near-identical 600-line yamls into a single base + thin overlays.
    """
    # ``stack`` tracks the active recursion path (push on entry, pop on
    # exit). Catches actual cycles A->B->...->A without false-positive-
    # ing on a future diamond pattern (A->B and A->C both extending D
    # — D is touched twice along non-cyclic branches).
    stack: list = []

    def _load(p: str) -> "OmegaConf":
        ap = os.path.abspath(p)
        if ap in stack:
            raise RuntimeError(
                f"Circular _extends in config chain: "
                f"{' -> '.join(stack + [ap])}"
            )
        stack.append(ap)
        try:
            cfg = OmegaConf.load(ap)
            # ``_extends`` is optional and consumed here.
            ext = None
            if isinstance(cfg, type(OmegaConf.create({}))) and "_extends" in cfg:
                ext = cfg["_extends"]
                del cfg["_extends"]
            if ext is None:
                return cfg
            ext_path = os.path.normpath(
                os.path.join(os.path.dirname(ap), str(ext))
            )
            base = _load(ext_path)
            return OmegaConf.merge(base, cfg)
        finally:
            stack.pop()

    return _load(path)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Phase-1 Action-Forcing DMD trainer"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--override", type=str, nargs="*", default=[])
    args = parser.parse_args()

    base = _load_config_with_extends(args.config)
    if args.override:
        override = OmegaConf.from_dotlist(list(args.override))
        cfg = OmegaConf.merge(base, override)
    else:
        cfg = base

    trainer = ActionForcingDMDTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
