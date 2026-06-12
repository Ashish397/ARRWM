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

import ast
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
    rpgan_d_loss_allpairs,
    rpgan_g_loss_allpairs,
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
            getattr(args, "gen_gradient_checkpointing", False)
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
        _phase_lora_no_ddp = bool(
            getattr(self.config, "student_phase_lora_enabled", False)
        )
        if self.world_size > 1:
            if _phase_lora_no_ddp:
                # Phase LoRA: the student is deliberately NOT DDP-
                # wrapped. Every DDP reducer configuration was tried
                # and failed — the per-iter-varying autograd graph
                # (random exit rung selects a different lora branch
                # each iter) and the ckpt-recompute forwards (which
                # call DDP _pre_forward MID-backward) together break
                # all of the reducer's modes:
                #   * find_unused=True: deferred bucket rebuild fires
                #     mid-recompute at step 21 (= gan_disc_start_step
                #     + 1, when rung_flash's first real grad completes
                #     the rebuild precondition) -> INTERNAL ASSERT
                #     (j5147634/5/6, j5151298+).
                #   * find_unused=False + ghost: ghost is a shallow
                #     loss node, all lora grads arrive at backward
                #     START, the rebuild precondition completes mid-
                #     backward at iter 1 -> "Expected to have finished
                #     reduction" (j5173579).
                #   * + static_graph: per-iter graph-structure
                #     variation (different lora branch per exit rung)
                #     violates the static-graph contract -> "training
                #     graph has changed in this iteration" at iter 2
                #     (j5176005/11).
                # With freeze_base=true the ONLY trainable student
                # params are the lora matrices (~tens of MB), so
                # manual gradient sync is cheap and removes the entire
                # reducer state machine from the problem. Mirrors the
                # existing ``action_projection`` manual-sync pattern
                # (see ``_all_reduce_extra_trainable_grads``, which
                # also syncs the lora grads in this mode).
                #
                # DDP's construction-time param broadcast is replaced
                # by an explicit one-time broadcast: peft initialises
                # lora_A from per-rank RNG, so ranks MUST be aligned
                # before the first forward.
                if dist.is_initialized():
                    _n_bcast = 0
                    for _n, _p in model.generator.model.named_parameters():
                        if "lora_" in _n:
                            dist.broadcast(_p.data, src=0)
                            _n_bcast += 1
                    if self.is_main_process:
                        logging.info(
                            "[ActionForcing] phase LoRA no-DDP mode: "
                            "broadcast %d lora params from rank 0; "
                            "student grads sync manually in "
                            "_all_reduce_extra_trainable_grads.",
                            _n_bcast,
                        )
            else:
                gen_fup = debug_fup
                self.generator_ddp = DDP(
                    model.generator.model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=gen_fup,
                    broadcast_buffers=False,
                )
                model.generator.model = self.generator_ddp  # type: ignore

            if bool(getattr(self.config, "fake_score_updates_enabled", True)):
                # v21: fake_alt_head_enabled adds head_alt params that
                # see no gradient on iters where compute_alt_head=False
                # (e.g. the DMD gen-step's fake_score call). DDP with
                # find_unused_parameters=False would hang on this. Flip
                # the flag to True when alt head is active so the
                # reducer skips alt-head params that weren't touched.
                fake_score_fup = debug_fup or bool(
                    getattr(self.config, "fake_alt_head_enabled", False)
                )
                self.fake_score_ddp = DDP(
                    model.fake_score.model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=fake_score_fup,
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

            # v27B: ForwardNoiser DDP wrap. The noiser is a separate
            # small ConvNet trained on rollout1→rollout2 chunk pairs
            # with CARN-step conditioning.
            # find_unused_parameters=True (defensive): the critic-step
            # loss path has multiple early-return guards (no rollout2
            # stash, no aligned pairs, chunk alignment edge cases). In
            # principle these are rank-symmetric, but with =False any
            # rank-divergent return path would AllReduce-hang. =True
            # makes DDP traverse the autograd graph to find used params
            # per iter — small perf cost on a 30M-param noiser, zero
            # hang risk.
            self.forward_noiser_ddp: Optional[DDP] = None
            if (
                bool(getattr(self.config, "forward_noiser_enabled", False))
                and getattr(model, "forward_noiser", None) is not None
            ):
                # Match the rest of the model's dtype (bf16) so 3D conv
                # bias matches input dtype. Without this, forward_noiser
                # stays in fp32 (default nn.Module dtype) while inputs
                # arrive as bf16 → RuntimeError on Conv3d bias.
                # Derive the dtype from fake_score's parameters so we
                # automatically follow whatever precision the run uses.
                noiser_dtype = torch.float32
                fs_model = getattr(model.fake_score, "model", None)
                if fs_model is not None:
                    fs_param = next(fs_model.parameters(), None)
                    if fs_param is not None:
                        noiser_dtype = fs_param.dtype
                model.forward_noiser = model.forward_noiser.to(
                    device=self.device, dtype=noiser_dtype,
                )
                self.forward_noiser_ddp = DDP(
                    model.forward_noiser,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=True,
                    broadcast_buffers=False,
                )
                model.forward_noiser = self.forward_noiser_ddp  # type: ignore
        else:
            self.real_score_ddp = None
            self.forward_noiser_ddp = None

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
        # ``gan_backbone`` selects the discriminator architecture:
        #   * "latent_3d_conv" (default, legacy) — 3D ConvNet on the
        #     student's latent video. Cheap, no VAE decode required.
        #   * "ladd_teacher_feat" — LADD adjacent-chunk discriminator
        #     on WAN teacher intermediate features.
        self.gan_backbone = str(
            getattr(self.config, "gan_backbone", "latent_3d_conv")
        )
        if self.gan_backbone not in (
            "latent_3d_conv", "ladd_teacher_feat",
        ):
            raise ValueError(
                "gan_backbone must be one of 'latent_3d_conv' | "
                "'ladd_teacher_feat'; got "
                f"{self.gan_backbone!r}."
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
            elif self.gan_backbone == "ladd_teacher_feat":
                # ===== LADD adjacent-chunk discriminator (v28) =====
                # Frozen WAN-teacher feature taps + trainable CCM/CSM
                # + 1D SpectralConv heads. See model/ladd_disc.py for
                # the architecture and the design rationale (LADD,
                # Projected GAN, ASD-style adjacent-chunk loss).
                from model.ladd_disc import build_ladd_disc
                # Tap indices: per-teacher-size defaults if config left
                # empty. 1.3B has 30 transformer blocks; 14B has 40.
                ladd_blocks = list(
                    getattr(self.config, "ladd_feature_blocks", []) or []
                )
                # Pull the underlying WanModel's transformer blocks from
                # real_score. Walk all submodules (handles arbitrary
                # wrap depth: DDP / WanDiffusionWrapper / v14 LoRA /
                # alt-head plumbing). Pick the first nn.ModuleList of
                # WAN transformer blocks we find.
                _real_score = self.model.real_score
                _blocks_iter = None
                _wm_for_blocks = _real_score
                for _candidate in [_real_score] + list(
                    _real_score.modules()
                ):
                    for _attr in ("transformer_blocks", "blocks"):
                        _b = getattr(_candidate, _attr, None)
                        if (
                            isinstance(_b, torch.nn.ModuleList)
                            and len(_b) >= 10  # WAN has 30+ blocks
                        ):
                            _blocks_iter = _b
                            _wm_for_blocks = _candidate
                            break
                    if _blocks_iter is not None:
                        break
                if _blocks_iter is None:
                    raise AttributeError(
                        "LADD: could not find a transformer-block "
                        "ModuleList anywhere under real_score. real_score "
                        f"type={type(_real_score).__name__}. Inspect the "
                        "wrap chain and either patch the search loop or "
                        "specify ``ladd_feature_blocks`` explicitly."
                    )
                _n_blocks = len(_blocks_iter)
                if not ladd_blocks:
                    # Span the depth: ~5 taps evenly distributed.
                    if _n_blocks <= 8:
                        ladd_blocks = list(range(_n_blocks))
                    else:
                        ladd_blocks = [
                            max(0, int(_n_blocks * f / 5))
                            for f in (1, 2, 3, 4, 4.95)
                        ]
                        # Dedup + sort; trim to <= 5.
                        ladd_blocks = sorted(set(min(b, _n_blocks - 1) for b in ladd_blocks))[:5]
                # Infer teacher hidden dim from a tap block. WAN
                # transformer block has .hidden_size or we read it
                # from any Linear layer in the block.
                _tap_block = _blocks_iter[ladd_blocks[0]]
                _dim_teacher = None
                for _name in ("hidden_size", "dim", "inner_dim"):
                    if hasattr(_tap_block, _name):
                        _dim_teacher = int(getattr(_tap_block, _name))
                        break
                if _dim_teacher is None:
                    # Fallback: scan block's parameters for a square
                    # weight (linear self-projection often has it).
                    for _p in _tap_block.parameters():
                        if _p.dim() == 2 and _p.shape[0] == _p.shape[1]:
                            _dim_teacher = int(_p.shape[0])
                            break
                if _dim_teacher is None:
                    raise RuntimeError(
                        "LADD: could not infer dim_teacher from a tap "
                        f"block at index {ladd_blocks[0]}. Specify "
                        "ladd_feature_blocks explicitly."
                    )
                # Prompt-conditioning dim from real_score's text embed.
                _prompt_embed_dim = 0
                if bool(getattr(self.config, "ladd_use_prompt_cond", True)):
                    _prompt_embed_dim = int(
                        getattr(self.config, "text_embed_dim", 4096)
                    )
                _wavelet_hf_enabled = bool(
                    getattr(self.config, "ladd_wavelet_hf_enabled", False)
                )
                _wavelet_in_channels = int(
                    getattr(self.config, "gan_disc_in_channels", 16)
                )
                # WAN tokenisation params for the 2D head reshape.
                # patch_size: read off the WAN model attribute if
                # present (default (1, 2, 2) for Wan-1.3B/14B);
                # action_tokens_per_frame: needed to strip per-frame
                # action tokens before the spatial reshape.
                _ps_attr = getattr(_wm_for_blocks, "patch_size", None)
                if _ps_attr is None:
                    _ps_attr = (1, 2, 2)
                _patch_size = tuple(int(p) for p in _ps_attr)
                _a_per_f = 0
                for _cand in [_real_score] + list(_real_score.modules()):
                    if hasattr(_cand, "action_tokens_per_frame"):
                        _v = int(getattr(_cand, "action_tokens_per_frame", 0))
                        if _v > 0:
                            _a_per_f = _v
                            break
                disc = build_ladd_disc(
                    real_score=_real_score,
                    block_indices=ladd_blocks,
                    dim_teacher=_dim_teacher,
                    dim_proj=int(
                        getattr(self.config, "ladd_proj_dim", 256)
                    ),
                    use_csm=bool(
                        getattr(self.config, "ladd_use_csm", True)
                    ),
                    use_lateral_proj=bool(
                        getattr(self.config, "ladd_use_lateral_proj", False)
                    ),
                    head_kernel_size=int(
                        getattr(self.config, "ladd_disc_head_kernel", 3)
                    ),
                    cmap_dim=int(
                        getattr(self.config, "ladd_cmap_dim", 64)
                    ) if _prompt_embed_dim > 0 else 0,
                    prompt_embed_dim=_prompt_embed_dim,
                    wavelet_hf_enabled=_wavelet_hf_enabled,
                    wavelet_hf_in_channels=_wavelet_in_channels,
                    wavelet_hf_drop_ll=bool(
                        getattr(
                            self.config,
                            "ladd_wavelet_hf_drop_ll",
                            False,
                        )
                    ),
                    wavelet_hf_adapter_init_gain=float(
                        getattr(
                            self.config,
                            "ladd_wavelet_hf_adapter_init_gain",
                            0.1,
                        )
                    ),
                    wavelet_hf_ll_weight=float(
                        getattr(
                            self.config,
                            "ladd_wavelet_hf_ll_weight",
                            0.15,
                        )
                    ),
                    patch_size=_patch_size,
                    action_tokens_per_frame=_a_per_f,
                    # Parallel stat-head sideband: distribution-match
                    # std statistics adversarially. Opt-in; default
                    # OFF so existing runs are unaffected.
                    stat_head_enabled=bool(
                        getattr(
                            self.config,
                            "ladd_stat_head_enabled",
                            False,
                        )
                    ),
                    stat_head_frames_per_window=int(
                        getattr(
                            self.config,
                            "ladd_stat_head_frames_per_window",
                            int(getattr(self.config, "num_frame_per_block", 3)),
                        )
                    ),
                    stat_head_pool_size=int(
                        getattr(
                            self.config,
                            "ladd_stat_head_pool_size",
                            4,
                        )
                    ),
                    stat_head_hidden_dim=int(
                        getattr(
                            self.config,
                            "ladd_stat_head_hidden_dim",
                            256,
                        )
                    ),
                )
                # All-fp32 for R1 stability.
                disc.to(device=self.device, dtype=torch.float32)
                disc.train()
                self.r3gan_disc = disc
                self.ladd_block_indices = ladd_blocks
                if self.world_size > 1:
                    # Trainable params = CCM + CSM + heads + cmapper.
                    # Teacher params are not in disc.parameters() (the
                    # projector is a plain attribute, not a submodule).
                    self.r3gan_disc_ddp = DDP(
                        disc,
                        device_ids=[self.local_rank],
                        output_device=self.local_rank,
                        find_unused_parameters=False,
                        broadcast_buffers=False,
                    )
                if self.is_main_process:
                    n_total = sum(p.numel() for p in disc.parameters())
                    n_train = sum(
                        p.numel() for p in disc.parameters()
                        if p.requires_grad
                    )
                    logging.info(
                        "[ActionForcing] LADD discriminator built: "
                        "blocks=%s dim_teacher=%d dim_proj=%d "
                        "use_csm=%s cmap_dim=%d wavelet_hf=%s "
                        "params_total=%.2fM params_trainable=%.2fM "
                        "(DDP=%s)",
                        ladd_blocks, _dim_teacher,
                        int(getattr(self.config, "ladd_proj_dim", 256)),
                        bool(getattr(self.config, "ladd_use_csm", True)),
                        int(getattr(self.config, "ladd_cmap_dim", 64))
                        if _prompt_embed_dim > 0 else 0,
                        bool(_wavelet_hf_enabled),
                        n_total / 1e6, n_train / 1e6,
                        self.r3gan_disc_ddp is not None,
                    )

                # R1/R2 cadence sanity. The R2 (fake-side) penalty is
                # supposed to fire on steps OFFSET from R1's so their two
                # perturbed-input disc forwards never stack in the same
                # D-update (OOM guard). That guarantee only holds when
                # BOTH cadences are >= 2: if either ``ladd_r1_every_n_steps``
                # or ``ladd_r2_every_n_steps`` is 1 (fires every step), the
                # offset cannot separate them and both perturbed forwards
                # land together. Detect any real collision over the LCM
                # period and warn loudly — silent OOM is the failure mode
                # this whole offset machinery exists to avoid.
                _r2_g = float(getattr(self.model, "ladd_r2_gamma", 0.0))
                if _r2_g > 0.0 and self.is_main_process:
                    import math as _m
                    _r1_e = max(1, int(getattr(
                        self.model, "ladd_r1_every_n_steps", 1)))
                    _r2_e = max(1, int(getattr(
                        self.model, "ladd_r2_every_n_steps", _r1_e)))
                    _r2_o = int(getattr(self.model, "ladd_r2_phase_offset", 1))
                    _period = _r1_e * _r2_e // _m.gcd(_r1_e, _r2_e)
                    _collisions = [
                        s for s in range(_period)
                        if s % _r1_e == 0 and (s - _r2_o) % _r2_e == 0
                    ]
                    if _collisions:
                        logging.warning(
                            "[ActionForcing] LADD R2 enabled "
                            "(ladd_r2_gamma=%.4g) but R1 and R2 CO-FIRE on "
                            "steps %s of every %d (r1_every=%d, r2_every=%d, "
                            "r2_phase_offset=%d). On those steps BOTH the "
                            "perturbed-real and perturbed-fake disc forwards "
                            "run in one D-update — the OOM the offset is "
                            "meant to prevent. For a clean offset set BOTH "
                            "cadences >= 2 (e.g. r1_every=2, r2_every=2, "
                            "offset=1 -> R1 even / R2 odd).",
                            _r2_g, _collisions, _period, _r1_e, _r2_e, _r2_o,
                        )
                    else:
                        logging.info(
                            "[ActionForcing] LADD R2 enabled "
                            "(ladd_r2_gamma=%.4g); R1/R2 fire on disjoint "
                            "steps (r1_every=%d, r2_every=%d, offset=%d) — "
                            "no perturbed-forward stacking.",
                            _r2_g, _r1_e, _r2_e, _r2_o,
                        )


        # ------------------------------------------------------------------
        # Moment-GAN — distribution-matching alternative to the std-MSE
        # anti-collapse loss. Trained independently of the main wavelet
        # LADD disc (own optimizer, own warmup schedule) but uses the same
        # R3GAN softplus + FD-R1 recipe. See model/ladd_disc.py for the
        # MomentDiscriminator architecture. Built in fp32 for R1 stability.
        # ------------------------------------------------------------------
        self.moment_disc: Optional[torch.nn.Module] = None
        self.moment_disc_ddp: Optional[DDP] = None
        self.moment_disc_optimizer: Optional[torch.optim.Optimizer] = None
        self.moment_gan_enabled = bool(
            getattr(self.config, "moment_gan_enabled", False)
        )
        if self.moment_gan_enabled:
            from model.ladd_disc import build_moment_disc
            mdisc = build_moment_disc(
                in_channels=int(
                    getattr(self.config, "moment_gan_in_channels", 16)
                ),
                hidden_dim=int(
                    getattr(self.config, "moment_gan_hidden_dim", 128)
                ),
                num_blocks=int(
                    getattr(self.config, "moment_gan_num_blocks", 2)
                ),
                include_mean=bool(
                    getattr(self.config, "moment_gan_include_mean", True)
                ),
                include_rms=bool(
                    getattr(self.config, "moment_gan_include_rms", True)
                ),
                clip_logit_enabled=bool(
                    getattr(
                        self.config,
                        "moment_gan_clip_logit_enabled",
                        False,
                    )
                ),
                clip_hidden_dim=int(
                    getattr(
                        self.config,
                        "moment_gan_clip_hidden_dim",
                        128,
                    )
                ),
            )
            mdisc.to(device=self.device, dtype=torch.float32)
            mdisc.train()
            self.moment_disc = mdisc
            if self.world_size > 1:
                self.moment_disc_ddp = DDP(
                    mdisc,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=False,
                    broadcast_buffers=False,
                )
            if self.is_main_process:
                n_total = sum(p.numel() for p in mdisc.parameters())
                logging.info(
                    "[ActionForcing] MomentDiscriminator built: "
                    "in_ch=%d hidden=%d blocks=%d include_mean=%s "
                    "include_rms=%s clip_logit=%s params=%.2fK (DDP=%s)",
                    mdisc.in_channels, mdisc.hidden_dim, mdisc.num_blocks,
                    mdisc.include_mean, mdisc.include_rms,
                    mdisc.clip_logit_enabled,
                    n_total / 1e3, self.moment_disc_ddp is not None,
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
        # dmd_context (hardcoded "self") — single source of truth
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
        # pred_image_7_chunk eval video: seed the student with 7 GT
        # context chunks, roll 7 more, log the full 14-chunk rollout.
        # Default ON; disable via ``sample_7chunk_enabled=false`` if the
        # all-ranks inference rollout ever misbehaves on the cluster.
        self.sample_7chunk_enabled = bool(
            getattr(self.config, "sample_7chunk_enabled", True)
        )
        self._sample_7chunk_ride: Optional[Dict[str, torch.Tensor]] = None
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
        # Optional non-uniform exit-flag sampling for phase-LoRA (or
        # plain DMD) training. List of K non-negative floats reweights
        # the random exit-rung selection; None / empty preserves uniform
        # behavior. Use to concentrate training on middle-t rungs
        # (min-SNR-γ-style) where score-matching has the best signal-
        # to-gradient-variance trade-off — extremes (rung 0 = highest
        # noise, rung K-1 = lowest noise) get less weight.
        exit_flag_weights_raw = getattr(cfg, "exit_flag_weights", None)
        exit_flag_weights: Optional[List[float]] = None
        if exit_flag_weights_raw is not None:
            if isinstance(exit_flag_weights_raw, str):
                # OmegaConf override "[0.1,0.4,0.4,0.1]" may arrive as
                # a string when the launcher does not parse the list
                # literal. Defensive parse handles both forms.
                try:
                    exit_flag_weights = list(
                        ast.literal_eval(exit_flag_weights_raw)
                    )
                except Exception:
                    exit_flag_weights = None
            else:
                exit_flag_weights = [float(v) for v in exit_flag_weights_raw]
            if exit_flag_weights is not None and (
                len(exit_flag_weights) != len(denoising_step_list)
            ):
                raise ValueError(
                    f"cfg.exit_flag_weights has length "
                    f"{len(exit_flag_weights)} but denoising_step_list "
                    f"has length {len(denoising_step_list)}"
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
            context_noise=context_noise,
            exit_flag_weights=exit_flag_weights,
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

        # Moment-GAN hyperparams + optimizer. Independent of the main
        # GAN/wavelet disc — has its own loss weight, lr, warmup
        # schedule and R1 cadence. All knobs default to safe values
        # that mirror the wavelet LADD disc's defaults so the moment
        # disc engages on a similar timeline.
        self.moment_gan_loss_weight = float(
            getattr(cfg, "moment_gan_loss_weight", 0.01)
        )
        self.moment_gan_r1_gamma = float(
            getattr(cfg, "moment_gan_r1_gamma", 0.01)
        )
        self.moment_gan_r1_every_n_steps = max(
            1, int(getattr(cfg, "moment_gan_r1_every_n_steps", 10))
        )
        self.moment_gan_r1_sigma = float(
            getattr(cfg, "moment_gan_r1_sigma", 0.01)
        )
        self.moment_gan_disc_start_step = int(
            getattr(cfg, "moment_gan_disc_start_step", 20)
        )
        self.moment_gan_critic_warmup_steps = int(
            getattr(cfg, "moment_gan_critic_warmup_steps", 20)
        )
        self.moment_gan_warmup_steps = int(
            getattr(cfg, "moment_gan_warmup_steps", 100)
        )
        self.moment_gan_updates_per_step = int(
            getattr(cfg, "moment_gan_updates_per_step", 1)
        )
        self.moment_gan_max_grad_norm = float(
            getattr(cfg, "moment_gan_max_grad_norm", self.max_grad_norm)
        )
        if self.moment_gan_enabled and self.moment_disc is not None:
            mdisc_lr = float(getattr(cfg, "moment_gan_lr", 2e-5))
            mdisc_betas = tuple(
                getattr(cfg, "moment_gan_betas", [0.0, 0.9])
            )
            mdisc_eps = float(getattr(cfg, "moment_gan_eps", 1e-8))
            mdisc_wd = float(getattr(cfg, "moment_gan_weight_decay", 0.0))
            mdisc_params = [
                p for p in self.moment_disc.parameters() if p.requires_grad
            ]
            if not mdisc_params:
                raise RuntimeError(
                    "moment_disc has no trainable parameters; check the "
                    "MomentDiscriminator constructor."
                )
            self.moment_disc_optimizer = torch.optim.AdamW(
                mdisc_params,
                lr=mdisc_lr,
                betas=mdisc_betas,
                eps=mdisc_eps,
                weight_decay=mdisc_wd,
            )
            if self.is_main_process:
                n_params = sum(p.numel() for p in mdisc_params)
                logging.info(
                    "[ActionForcing] MomentGAN optimizer built: AdamW "
                    "lr=%.2e betas=%s wd=%.4f params=%.2fK "
                    "(loss_weight=%.4f, R1_gamma=%.4f, warmup_steps=%d, "
                    "critic_warmup=%d, disc_start=%d)",
                    mdisc_lr, mdisc_betas, mdisc_wd, n_params / 1e3,
                    self.moment_gan_loss_weight, self.moment_gan_r1_gamma,
                    self.moment_gan_warmup_steps,
                    self.moment_gan_critic_warmup_steps,
                    self.moment_gan_disc_start_step,
                )

        # GAN warmup / disc-start schedule (used by the LADD path).
        # ``gan_critic_warmup_steps``: defers the gen-side GAN gradient
        # until this outer step; D trains from step 0 (or from
        # ``gan_disc_start_step`` if non-zero). ``gan_disc_start_step``:
        # defers the entire D-side pass until this step.
        self.gan_critic_warmup_steps = int(
            getattr(cfg, "gan_critic_warmup_steps", 500)
        )
        self.gan_disc_start_step = int(
            getattr(cfg, "gan_disc_start_step", 0)
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
            self._dump_module_inventory()
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
                # Phase LoRA: step the K-1 remaining per-rung
                # optimizers (rung_0 IS self.optimizer above; the
                # rest each own only their adapter's LoRA params).
                # Each adapter is clipped INDEPENDENTLY at the same
                # max_grad_norm as rung_0: per-adapter granularity is
                # the right unit (a spike on one rung — most likely
                # rung_flash, which alone absorbs the full
                # gan_loss_weight adversarial gradient — must not
                # shrink another rung's legitimate update, which a
                # single global clip over all lora params would do).
                for _opt in getattr(self, "phase_lora_optimizers", [])[1:]:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in _opt.param_groups[0]["params"]
                         if p.grad is not None],
                        max_norm=self.max_grad_norm,
                    )
                    _opt.step()
                    _opt.zero_grad(set_to_none=True)
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

            # v27B forward-noiser optimizer step.
            forward_noiser_grad_norm_val = 0.0
            if getattr(self, "forward_noiser_optimizer", None) is not None:
                fn_params_with_grad = [
                    p for p in
                    self.forward_noiser_optimizer.param_groups[0]["params"]
                    if p.grad is not None
                ]
                if fn_params_with_grad:
                    fngn = torch.nn.utils.clip_grad_norm_(
                        fn_params_with_grad,
                        max_norm=self.forward_noiser_max_grad_norm,
                    )
                    forward_noiser_grad_norm_val = (
                        float(fngn.item()) if torch.is_tensor(fngn)
                        else float(fngn)
                    )
                    self.forward_noiser_optimizer.step()
                self.forward_noiser_optimizer.zero_grad(set_to_none=True)
                # Surface grad norm to wandb via out dict if available.
                if isinstance(generator_log_dict, dict):
                    generator_log_dict["forward_noiser_grad_norm"] = (
                        forward_noiser_grad_norm_val
                    )

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
            # Per-step boundary breakdown: dump every step so we can see
            # which named boundary's ``delta_alloc`` is non-zero across
            # steps (= the leak's residence). Stays gated by
            # ``memory_audit_enabled``. Smoke turns this on.
            if bool(getattr(cfg, "memory_audit_enabled", False)):
                self._dump_step_mem_breakdown(f"step_{self.step}")
                if (
                    torch.cuda.is_available()
                    and getattr(self, "is_main_process", True)
                ):
                    _alloc_gb = (
                        torch.cuda.memory_allocated() / (1024 ** 3)
                    )
                    _reserved_gb = (
                        torch.cuda.memory_reserved() / (1024 ** 3)
                    )
                    logging.info(
                        "[mem-end step=%d] alloc=%6.2f GB  reserved=%6.2f GB",
                        self.step, _alloc_gb, _reserved_gb,
                    )

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
                # Streaming mode: the standalone critic iter is a no-op
                # (its dict has no critic_loss) — the REAL diffusion loss
                # comes from the gtfix path merged into the gen-step
                # dict. Without this fallback the line printed a
                # misleading 0.0000 for weeks.
                _cl_disp = critic_log_dict.get(
                    "critic_loss",
                    generator_log_dict.get("critic_loss", 0.0),
                )
                if not _cl_disp:
                    _cl_disp = generator_log_dict.get("critic_loss", 0.0)
                msg_parts.append(f"critic_loss={float(_cl_disp):.4f}")
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
                    # Skip private/internal keys (e.g. ``_aux_teacher_tensors``
                    # — a dict of bf16 graph tensors stashed for the LoRA
                    # action/probe losses, NOT a metric). Leaving it in
                    # poisons the payload: wandb can't serialize the nested
                    # bf16 and drops the WHOLE step, so every gen/critic curve
                    # silently vanishes. Also drop multi-element tensors here.
                    for k, v in generator_log_dict.items():
                        if str(k).startswith("_"):
                            continue
                        if torch.is_tensor(v):
                            if v.numel() != 1:
                                continue
                            v = float(v.item())
                        log_payload[f"gen/{k}"] = v
                    for k, v in critic_log_dict.items():
                        if str(k).startswith("_"):
                            continue
                        if torch.is_tensor(v):
                            if v.numel() != 1:
                                continue
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
                        # Final guard: wandb chokes on bf16/fp16 tensors
                        # ("Got unsupported ScalarType BFloat16") and the
                        # exception drops the WHOLE payload for the step.
                        # Coerce scalar tensors to float and DROP anything
                        # that isn't a plain number/bool (containers, strings,
                        # multi-element tensors) so one bad value can never
                        # nuke the curves again.
                        clean_payload: Dict[str, Any] = {}
                        for _k, _v in log_payload.items():
                            if torch.is_tensor(_v):
                                if _v.numel() == 1:
                                    clean_payload[_k] = float(_v.item())
                                continue
                            if isinstance(_v, (int, float, bool)):
                                clean_payload[_k] = _v
                        wandb.log(clean_payload, step=self.step)
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

                # Rolling: also emit the FULL ride rollout accumulated
                # across rolls (the 21f pred_image window only shows the
                # latest slice once rides run deep). cap_frames=False —
                # the whole rollout is the point.
                _acc = getattr(self, "_rollout_video_acc", None)
                if _acc is not None and int(_acc.shape[1]) > 21:
                    try:
                        self._log_pred_image_video(
                            _acc.to(device=self.device, dtype=self.dtype),
                            int(self.step),
                            name="pred_image_rollout",
                            caption_suffix=(
                                f"full ride rollout "
                                f"({int(_acc.shape[1])} latent frames)"
                            ),
                            cap_frames=False,
                        )
                    except Exception as _exc:
                        logging.warning(
                            "[ActionForcing] pred_image_rollout decode "
                            "failed (len=%d): %s",
                            int(_acc.shape[1]), _exc,
                        )

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
                        # v27B: aux-teacher TF context + noisy_input source
                        ("clean_x_aux", "clean_x_aux",
                         "aux_teacher TF context (post forward-noiser/alt-head override)"),
                        ("aux_noise_base", "aux_noise_base",
                         "aux_teacher noise_base (= causal_AR_GT when fn/alt active, else gt_target)"),
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
        if self.moment_gan_enabled and self.moment_disc is not None:
            md_module = (
                self.moment_disc_ddp.module
                if self.moment_disc_ddp is not None
                else self.moment_disc
            )
            state["moment_discriminator"] = md_module.state_dict()
            appended.append("moment_discriminator")
            if self.moment_disc_optimizer is not None:
                state["moment_disc_optimizer"] = (
                    self.moment_disc_optimizer.state_dict()
                )
                appended.append("moment_disc_optimizer")
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
        # ForwardNoiser (CARN) + its optimizer.
        _fn = getattr(self.model, "forward_noiser", None)
        if _fn is not None and "forward_noiser" in state:
            _fn_mod = _fn.module if hasattr(_fn, "module") else _fn
            fn_missing, fn_unexpected = _fn_mod.load_state_dict(
                state["forward_noiser"], strict=False,
            )
            if self.is_main_process:
                logging.info(
                    "resume: forward_noiser missing=%d unexpected=%d",
                    len(fn_missing), len(fn_unexpected),
                )
            _fn_opt = getattr(self, "forward_noiser_optimizer", None)
            if _fn_opt is not None and "forward_noiser_optimizer" in state:
                try:
                    _fn_opt.load_state_dict(state["forward_noiser_optimizer"])
                    if self.is_main_process:
                        logging.info("resume: forward_noiser_optimizer restored")
                except Exception as exc:
                    if self.is_main_process:
                        logging.warning(
                            "resume: forward_noiser_optimizer load failed: %s. "
                            "Starting FN optim from fresh state.", exc,
                        )
        if self.moment_gan_enabled and self.moment_disc is not None:
            md_module = (
                self.moment_disc_ddp.module
                if self.moment_disc_ddp is not None
                else self.moment_disc
            )
            if "moment_discriminator" in state:
                md_missing, md_unexpected = md_module.load_state_dict(
                    state["moment_discriminator"], strict=False,
                )
                if self.is_main_process:
                    logging.info(
                        "resume: moment_discriminator missing=%d unexpected=%d",
                        len(md_missing), len(md_unexpected),
                    )
            if (
                self.moment_disc_optimizer is not None
                and "moment_disc_optimizer" in state
            ):
                try:
                    self.moment_disc_optimizer.load_state_dict(
                        state["moment_disc_optimizer"]
                    )
                    if self.is_main_process:
                        logging.info(
                            "resume: moment_disc_optimizer state restored"
                        )
                except Exception as exc:
                    if self.is_main_process:
                        logging.warning(
                            "resume: moment_disc_optimizer load failed: %s. "
                            "Starting moment_disc optim from fresh state.",
                            exc,
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
    def _compute_aux_teacher_disc_losses(
        self,
        *,
        lora_x0: Optional[torch.Tensor],
        gt_target: Optional[torch.Tensor],
        chunk_lo: int,
        current_step: int,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        """Borrow the GAN disc's signal as a regulariser on the LoRA
        aux teacher's x0 estimate.

        Two paths, each gated by its own weight knob on the model:

          * ``aux_teacher_disc_adv_weight`` > 0 — RpGAN gen-loss style.
            Concat ``[gt_target, lora_x0]`` along batch, forward the
            LADD disc once, split the logits, apply
            ``rpgan_g_loss(d_real.detach(), d_fake)`` to push lora_x0
            toward "real" from the disc's perspective. Disc params
            are detached for this forward (we use the disc as a critic;
            it's still trained on the main gen-side path).

          * ``aux_teacher_disc_feat_weight`` > 0 — feature matching.
            Run the disc's projector (only) on both lora_x0 (grad-on)
            and gt_target (no_grad), L2 between per-block teacher
            features. LPIPS-style signal — no adversarial framing.

        Both default to 0 (off). When both > 0, both signals are
        summed. Gated on top of the aux teacher's own start gate by
        ``aux_teacher_disc_warmup_steps`` so the disc has had enough
        D-updates to be a useful critic before we read its score.

        The disc forward operates on a SINGLE per-block (npb-frame)
        chunk slice (``lora_x0[:, :npb]``) — keeps the forward at the
        disc's training distribution (it trains on per-chunk slices,
        not the full aux teacher window). One slice per iter is enough
        to give the LoRA a useful gradient.

        ``chunk_lo`` is the absolute ride-window frame index of
        ``lora_x0[:, 0]``, used to slice the matching action tokens.
        The aux teacher computes ``gt_target = ride_window[:, chunk_lo:
        chunk_lo + chunk_size]`` so ``lora_x0[:, :npb]`` corresponds
        to ride frames ``[chunk_lo, chunk_lo + npb)``. Caller MUST
        pass the chunk_lo it computed for this iter — reading the
        ``streaming_state["last_chunk_lo_in_ride_window"]`` stash
        would race with the main GAN path (set after this helper
        runs, so we'd see the PREVIOUS iter's value → action tokens
        misaligned with lora_x0's actual ride position).

        Returns ``(total_loss, logs)`` where ``total_loss`` is a
        graph-bearing scalar (gradient flows to LoRA params via
        ``lora_x0``), or ``None`` when the path is fully gated off.
        """
        # DDP-safety note: all early-return paths below are derivable
        # from globally-consistent state — config knobs (weights,
        # warmup), aux-teacher's own DDP-synced skip (which makes
        # lora_x0 / gt_target None on ALL ranks together when the
        # ride is too short), or deterministic shape/state derived
        # from those. Ride-actions length is tied to ride-window
        # length (same loader output), so its shape check inherits
        # the aux-teacher's sync. So every rank reaches the disc
        # forward together, or none do — no explicit all_reduce
        # needed here. If any of these conditions ever DOES diverge
        # per-rank in a future code change, the LoRA's DDP allreduce
        # at backward will detect the unused-param mismatch.
        adv_w = float(getattr(self.model, "aux_teacher_disc_adv_weight", 0.0))
        feat_w = float(
            getattr(self.model, "aux_teacher_disc_feat_weight", 0.0)
        )
        if adv_w == 0.0 and feat_w == 0.0:
            return None, {}
        if (
            not getattr(self, "gan_enabled", False)
            or self.r3gan_disc is None
        ):
            return None, {}
        if lora_x0 is None or gt_target is None:
            return None, {}
        warmup = int(
            getattr(self.model, "aux_teacher_disc_warmup_steps", 100)
        )
        if current_step < warmup:
            return None, {"train/aux_disc_skipped_warmup": 1.0}

        device = lora_x0.device
        disc = self.r3gan_disc
        npb = int(getattr(self.model, "num_frame_per_block", 3))
        if lora_x0.shape[1] < npb:
            return None, {"train/aux_disc_skipped_too_short": 1.0}

        # First npb frames — deterministic across ranks (DDP-safe).
        lora_chunk = lora_x0[:, :npb].to(torch.float32)
        gt_chunk = gt_target[:, :npb].to(torch.float32).detach()
        B = lora_chunk.shape[0]

        # Disc at t=0 (wavelet_hf path keeps clean inputs; mirrors
        # what ``_ladd_run_pair_mode`` does when wavelet_on).
        t_disc = torch.zeros((B, npb), dtype=torch.long, device=device)

        # ----- Prompt embeddings + pooled prompt -----
        s_state = getattr(self.model, "streaming_state", None) or {}
        prompt_embeds = (
            s_state.get("prompt_embeds")
            if isinstance(s_state, dict) else None
        )
        if prompt_embeds is None:
            return None, {"train/aux_disc_skipped_no_prompt": 1.0}
        # Batch broadcast. In practice the aux teacher and the ride
        # share the same B, so the shapes match. If they ever don't
        # (e.g. multi-prompt ride), require a clean integer ratio so
        # we don't silently produce a misaligned slice.
        if prompt_embeds.shape[0] != B:
            if B % max(1, prompt_embeds.shape[0]) != 0:
                return None, {"train/aux_disc_skipped_prompt_shape": 1.0}
            reps = B // prompt_embeds.shape[0]
            prompt_embeds = prompt_embeds.repeat_interleave(reps, dim=0)
        pooled_prompt = (
            prompt_embeds.float().mean(dim=1)
            if getattr(disc, "cmap_dim", 0) > 0 else None
        )

        # ----- Per-chunk action conditioning -----
        # ``chunk_lo`` is passed by the caller — it's the absolute
        # ride frame index of ``lora_x0[:, 0]``. Slicing the action
        # window at ``[chunk_lo : chunk_lo + npb)`` matches the
        # frames lora_x0[:, :npb] cover; the disc's action-aware
        # WAN forward then sees actions consistent with the latents.
        a_per_f = 0
        _rs = self.model.real_score
        for _cand in [_rs] + list(_rs.modules()):
            if hasattr(_cand, "action_tokens_per_frame"):
                a_per_f = int(getattr(_cand, "action_tokens_per_frame", 0))
                if a_per_f > 0:
                    break
        cond_extra: Optional[Dict[str, torch.Tensor]] = None
        if a_per_f > 0:
            ride_actions = (
                s_state.get("ride_actions_window")
                if isinstance(s_state, dict) else None
            )
            atp = getattr(self.model, "action_token_projection", None)
            ap = getattr(self.model, "action_projection", None)
            if (
                ride_actions is None
                or atp is None
            ):
                return None, {"train/aux_disc_skipped_no_actions": 1.0}
            lo, hi = int(chunk_lo), int(chunk_lo) + npb
            if lo < 0 or ride_actions.shape[1] < hi:
                return None, {"train/aux_disc_skipped_no_actions": 1.0}
            acts = ride_actions[:, lo:hi].to(
                device=device, dtype=prompt_embeds.dtype,
            )
            with torch.no_grad():
                act_tokens = atp(acts).detach()
                act_mod = (
                    ap(acts, num_frames=npb).detach()
                    if ap is not None else None
                )
            cond_extra = {"_action_tokens": act_tokens}
            if act_mod is not None:
                cond_extra["_action_modulation"] = act_mod

        logs: Dict[str, float] = {}
        total_loss: Optional[torch.Tensor] = None

        # Disc in eval (no spectral_norm buffer mutation), params
        # detached (critic mode — not trained on this signal).
        disc_was_training = disc.training
        disc.eval()
        disc.requires_grad_(False)
        # CRITICAL: the LADD disc's WanFeatureProjector holds
        # ``self.real_score`` as a plain Python attribute, not as an
        # nn.Module submodule (model/ladd_disc.py:99). So
        # ``disc.requires_grad_(False)`` above does NOT touch the
        # LoRA params inside real_score. Without freezing them
        # explicitly here, the disc/projector forward on ``lora_chunk``
        # would open a feedback path:
        #   disc_loss → projector's real_score(lora_chunk) → LoRA params
        # which is anti-correlated with the wanted signal: instead of
        # making ``lora_chunk`` closer to GT, it would tune the LoRA
        # so the FEATURE EXTRACTOR maps any input to GT-like features.
        # For the FM path this is especially toxic (no adversarial
        # counterbalance). Freezing the LoRA params for the duration
        # of the disc forward kills the feedback loop while
        # preserving the wanted path (disc_loss → lora_chunk → aux
        # teacher's autograd graph → LoRA params).
        # Memory bonus: the projector's activation checkpoint no
        # longer needs to save LoRA-bearing layer activations for a
        # backward that will never reach those params.
        _rs = self.model.real_score
        _lora_params_to_restore = [
            p for p in _rs.parameters() if p.requires_grad
        ]
        for _p in _lora_params_to_restore:
            _p.requires_grad_(False)
        try:
            if adv_w > 0.0:
                from model.r3gan import rpgan_g_loss
                combined = torch.cat([gt_chunk, lora_chunk], dim=0)
                _t = torch.cat([t_disc, t_disc], dim=0)
                _pe = torch.cat([prompt_embeds, prompt_embeds], dim=0)
                _pp = (
                    torch.cat([pooled_prompt, pooled_prompt], dim=0)
                    if pooled_prompt is not None else None
                )
                _ce = None
                if cond_extra is not None:
                    _ce = {
                        k: torch.cat([v, v], dim=0)
                        for k, v in cond_extra.items()
                    }
                combined_logits = disc(
                    x_noisy=combined,
                    timestep=_t,
                    prompt_embeds=_pe,
                    pooled_prompt=_pp,
                    conditional_extra=_ce,
                )
                d_real = combined_logits[:B]
                d_fake = combined_logits[B:]
                K_stat = int(getattr(disc, "stat_logit_count", 0))
                W_stat = float(getattr(
                    self.model, "ladd_stat_head_loss_weight", 1.0,
                ))
                if K_stat > 0 and W_stat > 0.0:
                    g_visual = rpgan_g_loss(
                        d_real[:, :-K_stat].detach(),
                        d_fake[:, :-K_stat],
                    )
                    g_stat = rpgan_g_loss(
                        d_real[:, -K_stat:].detach(),
                        d_fake[:, -K_stat:],
                    )
                    g_rp = g_visual + W_stat * g_stat
                else:
                    g_rp = rpgan_g_loss(d_real.detach(), d_fake)
                adv_loss = adv_w * g_rp.to(lora_x0.dtype)
                total_loss = (
                    adv_loss if total_loss is None
                    else total_loss + adv_loss
                )
                logs["train/aux_disc_adv_raw"] = float(
                    g_rp.detach().item()
                )
                logs["train/aux_disc_adv_weighted"] = float(
                    adv_loss.detach().item()
                )

            if feat_w > 0.0:
                projector = disc.projector
                # IMPORTANT: ``LADDDiscriminator.forward`` applies
                # ``self.wavelet_hf`` to x_noisy BEFORE the projector
                # call (model/ladd_disc.py:810-811). The projector was
                # trained to receive wavelet-HF-transformed inputs.
                # Mirror that pre-stage manually here — feeding raw
                # latents to the projector would be OOD relative to
                # the disc's training distribution.
                wavelet_hf = getattr(disc, "wavelet_hf", None)
                if wavelet_hf is not None:
                    lora_for_proj = wavelet_hf(lora_chunk)
                    with torch.no_grad():
                        gt_for_proj = wavelet_hf(gt_chunk)
                else:
                    lora_for_proj = lora_chunk
                    gt_for_proj = gt_chunk
                feats_fake = projector(
                    x_noisy=lora_for_proj,
                    timestep=t_disc,
                    prompt_embeds=prompt_embeds,
                    conditional_extra=cond_extra,
                )
                with torch.no_grad():
                    feats_real = projector(
                        x_noisy=gt_for_proj,
                        timestep=t_disc,
                        prompt_embeds=prompt_embeds,
                        conditional_extra=cond_extra,
                    )
                feat_l2 = None
                n_blocks = 0
                for idx in feats_fake:
                    diff_sq = (
                        feats_fake[idx]
                        - feats_real[idx].detach()
                    ).pow(2).mean()
                    feat_l2 = (
                        diff_sq if feat_l2 is None
                        else feat_l2 + diff_sq
                    )
                    n_blocks += 1
                if feat_l2 is not None and n_blocks > 0:
                    feat_l2 = feat_l2 / float(n_blocks)
                    feat_loss = feat_w * feat_l2.to(lora_x0.dtype)
                    total_loss = (
                        feat_loss if total_loss is None
                        else total_loss + feat_loss
                    )
                    logs["train/aux_disc_feat_raw"] = float(
                        feat_l2.detach().item()
                    )
                    logs["train/aux_disc_feat_weighted"] = float(
                        feat_loss.detach().item()
                    )
        finally:
            disc.requires_grad_(True)
            if disc_was_training:
                disc.train()
            # Restore the LoRA params' requires_grad state. The aux
            # teacher's autograd graph (recorded BEFORE this disc
            # forward) is independent of this restore — its backward
            # path through real_score still works because the
            # recorded ops captured grad-state at forward time.
            for _p in _lora_params_to_restore:
                _p.requires_grad_(True)

        if total_loss is not None:
            logs["train/aux_disc_total"] = float(total_loss.detach().item())
        return total_loss, logs

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
        # Cached-decode path: avoids the init-frame brightness anomaly
        # that plain decode injects at every call (see ``_decode_no_grad``
        # docstring above for the full rationale).
        if use_checkpoint:
            from torch.utils.checkpoint import checkpoint as _ckpt

            def _decode(z):
                return vae.decode_to_pixel(z, use_cache=True)

            pix = _ckpt(_decode, latent, use_reentrant=False)
        else:
            pix = vae.decode_to_pixel(latent, use_cache=True)
        return pix

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

    # ==================================================================
    # LADD discriminator (v28)
    # ==================================================================
    # Supports two pair construction modes, independently toggleable:
    #   * "gt_vs_fake" (LADD canonical): real = GT chunks at the same
    #     temporal positions as the gen output. Standard adversarial
    def _compute_moment_gan_losses(
        self,
        pred_image: torch.Tensor,
        gt_latents_window: torch.Tensor,
        current_step: int,
    ) -> tuple:
        """Distribution-matching GAN on per-frame latent moments.

        Trains a tiny MLP discriminator (``self.moment_disc``) to tell
        GT per-frame moment vectors apart from the student's, then
        backprops the symmetric RpGAN gen loss into ``pred_image``.

        The semantic shift vs ``latent_std_mse_loss``: that loss pulls
        ``s_pred`` toward ``s_gt`` *per frame* (point-wise match); this
        loss pulls the student's *distribution* of per-frame moments
        toward GT's, leaving room for natural per-frame variation. See
        ``model.ladd_disc.MomentDiscriminator`` for the architecture.

        Mirrors ``_ladd_run_pair_mode``'s recipe: combined real+fake
        D-update forward, FD-R1 calibrated to the autograd magnitude
        (``0.5 · γ · ‖∇_x Σ_i D_i‖²``), gen-side forward in disc.eval()
        to silence spectral_norm power-iter mutation.

        Returns ``(generator_gan_loss, logs)``. ``generator_gan_loss``
        is graph-attached through ``pred_image``.
        """
        device = pred_image.device
        zero = torch.zeros((), device=device, dtype=torch.float32)
        if (
            not self.moment_gan_enabled
            or self.moment_disc is None
        ):
            return zero, {}

        # Shape parity is assumed downstream — both sides feed the same
        # disc forward and pair their logits 1-to-1. A mismatched
        # window slice would fail deep inside ``extract_moments`` with
        # a confusing reshape error; fail loud at the boundary instead.
        if pred_image.shape != gt_latents_window.shape:
            raise ValueError(
                "_compute_moment_gan_losses: pred_image and "
                "gt_latents_window must have identical shapes; got "
                f"pred_image={tuple(pred_image.shape)} vs "
                f"gt_latents_window={tuple(gt_latents_window.shape)}."
            )

        fake_grad = pred_image.to(torch.float32)
        fake_det = fake_grad.detach()
        real_det = gt_latents_window.detach().to(torch.float32)

        disc_for_update = (
            self.moment_disc_ddp
            if self.moment_disc_ddp is not None
            else self.moment_disc
        )
        disc_for_guidance = self.moment_disc

        # ----- D-update -----
        disc_skipped = (
            current_step < int(self.moment_gan_disc_start_step)
        )
        last_d_loss = 0.0
        last_d_real = 0.0
        last_d_fake = 0.0
        last_r1 = 0.0
        last_r1_grad_sq = float("nan")
        last_r1_fired = 0.0
        if not disc_skipped and self.moment_disc_optimizer is not None:
            for _ in range(int(self.moment_gan_updates_per_step)):
                self.moment_disc_optimizer.zero_grad(set_to_none=True)
                _do_r1 = (
                    current_step % self.moment_gan_r1_every_n_steps == 0
                )
                B_d = real_det.shape[0]
                if _do_r1:
                    # FD-R1: perturb real only, batch [real, fake,
                    # real_perturbed] through a single disc forward.
                    sigma = float(self.moment_gan_r1_sigma)
                    real_part = real_det.detach()
                    fake_part = fake_det.detach()
                    eps_real = sigma * torch.randn_like(real_part)
                    real_perturbed = real_part + eps_real
                    combined = torch.cat(
                        [real_part, fake_part, real_perturbed], dim=0,
                    ).requires_grad_(False)
                    logits = disc_for_update(combined)
                    d_real = logits[:B_d]
                    d_fake = logits[B_d:2 * B_d]
                    d_real_pert = logits[2 * B_d:]
                    # Match the LADD FD-R1 calibration: sum the per-token
                    # logits per sample BEFORE FD so the penalty
                    # magnitude tracks ``‖∇_x Σ_i D_i‖²`` (same as
                    # autograd). Here per-frame logits are [B, F]; sum
                    # over F to get one scalar per sample.
                    d_real_sum = d_real.sum(dim=1)
                    d_real_pert_sum = d_real_pert.sum(dim=1)
                    r1_grad_fd = (d_real_pert_sum - d_real_sum) / sigma
                    _r1_grad_sq_raw = r1_grad_fd.pow(2).mean()
                    r1 = 0.5 * self.moment_gan_r1_gamma * _r1_grad_sq_raw
                    last_r1_grad_sq = float(_r1_grad_sq_raw.detach().item())
                    last_r1_fired = 1.0
                else:
                    combined = torch.cat([real_det, fake_det], dim=0)
                    logits = disc_for_update(combined)
                    d_real = logits[:B_d]
                    d_fake = logits[B_d:]
                    # Graph-connected zero so backward is safe.
                    r1 = logits.sum() * 0.0
                d_rp = rpgan_d_loss(d_real, d_fake)
                d_total = d_rp + r1
                d_total.backward()
                if (
                    self.moment_gan_max_grad_norm
                    and self.moment_gan_max_grad_norm > 0
                ):
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.moment_disc.parameters()
                         if p.grad is not None],
                        self.moment_gan_max_grad_norm,
                    )
                self.moment_disc_optimizer.step()
                last_d_loss = float(d_rp.detach().item())
                last_d_real = float(d_real.detach().mean().item())
                last_d_fake = float(d_fake.detach().mean().item())
                last_r1 = float(r1.detach().item())

        # ----- Gen-side -----
        critic_warmup_done = (
            current_step >= int(self.moment_gan_critic_warmup_steps)
        )
        if (
            self.moment_gan_warmup_steps > 0
            and current_step < (
                self.moment_gan_critic_warmup_steps
                + self.moment_gan_warmup_steps
            )
            and current_step >= self.moment_gan_critic_warmup_steps
        ):
            ramp_steps_in = current_step - self.moment_gan_critic_warmup_steps
            t_norm = ramp_steps_in / max(1, self.moment_gan_warmup_steps)
            ramp = self._gan_warmup_shape_apply(t_norm)
            gen_gan_weight = ramp * self.moment_gan_loss_weight
        elif current_step >= (
            self.moment_gan_critic_warmup_steps
            + self.moment_gan_warmup_steps
        ):
            gen_gan_weight = self.moment_gan_loss_weight
        else:
            gen_gan_weight = 0.0

        gen_gan_main_value = 0.0
        if critic_warmup_done and gen_gan_weight > 0:
            disc_for_guidance.requires_grad_(False)
            disc_was_training = disc_for_guidance.training
            disc_for_guidance.eval()
            try:
                B_g = real_det.shape[0]
                combined_g = torch.cat([real_det, fake_grad], dim=0)
                g_logits = disc_for_guidance(combined_g)
                d_real_g = g_logits[:B_g]
                d_fake_g = g_logits[B_g:]
                g_rp = rpgan_g_loss(d_real_g.detach(), d_fake_g)
                generator_gan_loss = (
                    gen_gan_weight * g_rp.to(pred_image.dtype)
                )
                gen_gan_main_value = float(g_rp.detach().item())
            finally:
                disc_for_guidance.requires_grad_(True)
                if disc_was_training:
                    disc_for_guidance.train()
        else:
            generator_gan_loss = zero

        logs = {
            "train/moment_gan_disc_skipped": 1.0 if disc_skipped else 0.0,
            "train/moment_gan_d_loss": last_d_loss,
            "train/moment_gan_d_real": last_d_real,
            "train/moment_gan_d_fake_detached": last_d_fake,
            "train/moment_gan_r1": last_r1,
            "train/moment_gan_r1_grad_sq": last_r1_grad_sq,
            "train/moment_gan_r1_fired": last_r1_fired,
            "train/moment_gan_r1_gamma": float(self.moment_gan_r1_gamma),
            "train/moment_gan_g_loss_raw": gen_gan_main_value,
            "train/moment_gan_g_loss_weighted": (
                gen_gan_weight * gen_gan_main_value
            ),
            "train/moment_gan_g_weight": float(gen_gan_weight),
            "train/moment_gan_critic_warmup_done": (
                1.0 if critic_warmup_done else 0.0
            ),
        }
        return generator_gan_loss, logs

    # ==================================================================
    # ForwardNoiser (CARN) — teacher_feat training
    # ==================================================================
    # Train the FN to map rollout1 chunks -> rollout2 chunks (+1 CARN
    # drift step) by DISTRIBUTION-matching them in the FROZEN teacher's
    # feature space (the LADD WanFeatureProjector backbone, NOT its
    # trained heads). The teacher is a fixed, comprehensive critic, so the
    # FN can't exploit an adversary's blind spots. SEPARATE backward
    # (FN grads only; the train() loop steps the FN optimizer) — fully
    # decoupled from fake_score. The FN is step-UNCONDITIONED (carn_step=0
    # always): it learns one generic "+1 shift" transform.
    def _fn_action_blind_cond(self, n_rows: int, n_frames: int,
                              device, dtype):
        """Build action-blind (zeroed) conditioning of the shape the
        action-aware WAN teacher requires, so the projector forward runs.
        Returns None when the teacher has no action tokens."""
        m = self.model
        a_per_f = 0
        _rs = getattr(m, "real_score", None)
        if _rs is None:
            return None
        for _cand in [_rs] + list(_rs.modules()):
            if hasattr(_cand, "action_tokens_per_frame"):
                a_per_f = int(getattr(_cand, "action_tokens_per_frame", 0))
                if a_per_f > 0:
                    break
        if a_per_f <= 0:
            return None
        atp = getattr(m, "action_token_projection", None)
        ap = getattr(m, "action_projection", None)
        s = getattr(m, "streaming_state", None)
        ride_act = s.get("ride_actions_window") if isinstance(s, dict) else None
        a_dim = int(ride_act.shape[-1]) if ride_act is not None else int(
            getattr(m, "raw_action_dim", 2)
        )
        acts_zero = torch.zeros(
            (n_rows, n_frames, a_dim), device=device, dtype=dtype,
        )
        cond: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            if atp is not None:
                cond["_action_tokens"] = atp(acts_zero).detach()
            if ap is not None:
                cond["_action_modulation"] = ap(
                    acts_zero, num_frames=n_frames,
                ).detach()
        return cond if cond else None

    def _fn_teacher_feat_loss(self, fn_out, r2_target, projector):
        """Sliced-Wasserstein per-token distribution match between
        FN(rollout1) [grad-on] and rollout2 [detached] in the frozen
        teacher's feature space. Returns a scalar loss graph-attached
        through ``fn_out`` only.

        NOT a moment (mean+std) match: GT<->student moments barely differ
        and are pinned by the stat anchor, so a moment match is inert (v8d
        fn_tf_loss ~1e-6). The full sliced-Wasserstein marginal captures
        the texture/cartoon distribution shift the FN must learn."""
        m = self.model
        feat_t = int(getattr(m, "forward_noiser_feat_t", 0))
        # Match the disc's input pipeline: when the disc applies a wavelet-HF
        # pre-stage, the projector was conditioned on wavelet'd CLEAN latents
        # (disc forces t=0 when wavelet_on). Replicate both here so the FN is
        # matched in the SAME feature space its output is consumed in
        # (the carn_former feeds FN(GT) through this same wavelet'd disc).
        wavelet = getattr(self.r3gan_disc, "wavelet_hf", None)
        if wavelet is not None:
            feat_t = 0
        n_rows, n_frames = int(fn_out.shape[0]), int(fn_out.shape[1])
        device = fn_out.device
        s = m.streaming_state
        pe = s.get("prompt_embeds")
        if pe is None:
            return fn_out.sum() * 0.0
        # repeat (not repeat_interleave): rows are chunk-major (torch.cat of
        # per-chunk slices) and there is a single global prompt, so a tiled
        # repeat aligns per-row. (For B>1 with per-sample prompts this would
        # need revisiting — not the case in these configs.)
        reps = max(1, n_rows // int(pe.shape[0]))
        pe_eff = pe.repeat(reps, 1, 1) if int(pe.shape[0]) != n_rows else pe
        t = torch.full(
            (n_rows, n_frames), feat_t, dtype=torch.long, device=device,
        )

        # The wavelet-HF adapter is a DISC-owned trainable module; the FN
        # backward must NOT update it (only FN params). Freeze its params
        # for the duration of this forward (the transform stays
        # differentiable w.r.t. the INPUT, so the FN gradient still flows
        # through it). Restored in ``finally``.
        _wav_params = list(wavelet.parameters()) if wavelet is not None else []
        _wav_req = [p.requires_grad for p in _wav_params]

        def _prep(x):
            xn = _noise(x)
            if wavelet is not None:
                xn = wavelet(xn)
            return xn

        def _noise(x):
            if feat_t <= 0:
                return x
            sched = getattr(m, "scheduler", None)
            if sched is None or not hasattr(sched, "add_noise"):
                return x
            eps = torch.randn_like(x)
            xf, ef = x.flatten(0, 1), eps.flatten(0, 1)
            tpf = torch.full(
                (xf.shape[0],), feat_t, dtype=torch.long, device=device,
            )
            return sched.add_noise(xf, ef, tpf).unflatten(0, x.shape[:2])

        cond_extra = self._fn_action_blind_cond(
            n_rows, n_frames, device, pe_eff.dtype,
        )
        for _p in _wav_params:
            _p.requires_grad_(False)
        try:
            feats_fake = projector(
                x_noisy=_prep(fn_out), timestep=t, prompt_embeds=pe_eff,
                conditional_extra=cond_extra,
            )
            with torch.no_grad():
                feats_real = projector(
                    x_noisy=_prep(r2_target), timestep=t,
                    prompt_embeds=pe_eff, conditional_extra=cond_extra,
                )
        finally:
            for _p, _r in zip(_wav_params, _wav_req):
                _p.requires_grad_(_r)
        # Sliced-Wasserstein per-token: treat the LAST feature dim as the
        # channel and every (row, token) as a sample; project onto random
        # unit directions and L2 the SORTED 1D marginals (= 1D Wasserstein-2
        # per direction, averaged). Captures the full distribution shape
        # (all moments), not just the first two -> the texture/cartoon shift
        # survives where mean+std collapsed to ~0.
        n_proj = int(getattr(m, "forward_noiser_sw_n_proj", 64))
        max_tok = int(getattr(m, "forward_noiser_sw_max_tokens", 4096))
        idxs = list(feats_fake.keys())
        loss = fn_out.new_zeros(())
        for idx in idxs:
            ff = feats_fake[idx].float().reshape(-1, int(feats_fake[idx].shape[-1]))
            fr = feats_real[idx].float().reshape(
                -1, int(feats_real[idx].shape[-1])).detach()
            D = ff.shape[-1]
            M = min(int(ff.shape[0]), int(fr.shape[0]))
            if M == 0:
                # Empty tap (no tokens) -> skip; mean() over empty = NaN.
                # Unreachable for real latents, but cheap to bulletproof.
                continue
            if max_tok > 0 and M > max_tok:
                # Independent subsample per side: a sorted-marginal match
                # needs only equal COUNTS, not paired indices. Caps sort
                # cost + variance. FN grad flows through the gathered fake
                # tokens (differentiable index_select).
                sel_f = torch.randperm(int(ff.shape[0]), device=ff.device)[:max_tok]
                sel_r = torch.randperm(int(fr.shape[0]), device=fr.device)[:max_tok]
                ff = ff[sel_f]
                fr = fr[sel_r]
            elif int(ff.shape[0]) != int(fr.shape[0]):
                # Shapes should match (fn_out and r2_target are same shape);
                # guard anyway so the sorted L2 has equal lengths.
                ff = ff[:M]
                fr = fr[:M]
            dirs = torch.randn(D, n_proj, device=ff.device, dtype=ff.dtype)
            dirs = dirs / dirs.norm(dim=0, keepdim=True).clamp_min(1e-8)
            pf_s, _ = torch.sort(ff @ dirs, dim=0)
            pr_s, _ = torch.sort(fr @ dirs, dim=0)
            loss = loss + (pf_s - pr_s).pow(2).mean()
        return loss / max(1, len(idxs))

    def _train_fn_frontier_pair(self) -> dict:
        """FN frontier training (phase-2 rolling): consume the
        (rollout1', rollout2') chunk pair generated at the ride's
        frontier (``model.generate_fn_frontier_pair``) and backprop the
        teacher-feat SW loss into the FN. Called ONLY on reset steps
        (rank-lockstep), immediately before ``reset_streaming_state``.
        Mirrors ``_train_forward_noiser_tf``'s DDP discipline: every
        rank runs exactly one FN forward+backward (zero-anchor on any
        bail) so the FN DDP reducer stays matched."""
        m = self.model
        if (getattr(self, "is_main_process", True)
                and getattr(self, "_fn_frontier_entry_dbg", 0) < 5):
            self._fn_frontier_entry_dbg = getattr(
                self, "_fn_frontier_entry_dbg", 0) + 1
            import sys as _sys
            print(
                f"[FN-FRONTIER] invoked at ride depth "
                f"{self._chunks_in_current_ride}",
                file=_sys.stderr, flush=True,
            )

        def _anchor(reason: str) -> dict:
            if (getattr(self, "is_main_process", True)
                    and getattr(self, "_fn_frontier_anchor_dbg", 0) < 5):
                self._fn_frontier_anchor_dbg = getattr(
                    self, "_fn_frontier_anchor_dbg", 0) + 1
                import sys as _sys
                print(
                    f"[FN-FRONTIER] skipped: {reason}",
                    file=_sys.stderr, flush=True,
                )
            fn = m.forward_noiser
            fn_inner = fn.module if hasattr(fn, "module") else fn
            p0 = next(fn.parameters())
            C = int(getattr(fn_inner, "latent_channels", 16))
            x0 = torch.zeros(
                (1, int(getattr(m, "num_frame_per_block", 3)), C, 8, 8),
                device=p0.device, dtype=p0.dtype,
            )
            cz = torch.zeros((1,), dtype=torch.long, device=x0.device)
            (fn(x0, cz, residual=True).sum() * 0.0).backward()
            return {
                "train/fn_frontier_skipped": 1.0,
            }

        proj = (
            getattr(self.r3gan_disc, "projector", None)
            if self.r3gan_disc is not None else None
        )
        if proj is None:
            return _anchor("no_projector")
        pair = m.generate_fn_frontier_pair()
        if pair is None:
            return _anchor("no_pair_geometry")
        r1c, r2c = pair
        p0 = next(m.forward_noiser.parameters())
        r1c = r1c.to(device=p0.device, dtype=p0.dtype)
        r2c = r2c.to(device=p0.device, dtype=p0.dtype)
        cs = torch.zeros(
            (r1c.shape[0],), dtype=torch.long, device=r1c.device,
        )
        fn_out = m.forward_noiser(r1c, cs, residual=True)  # grad-on
        loss = self._fn_teacher_feat_loss(fn_out, r2c, proj)
        loss.backward()
        if (getattr(self, "is_main_process", True)
                and getattr(self, "_fn_frontier_dbg", 0) < 3):
            self._fn_frontier_dbg = getattr(self, "_fn_frontier_dbg", 0) + 1
            import sys as _sys
            print(
                f"[FN-FRONTIER] trained on frontier pair at ride depth "
                f"{self._chunks_in_current_ride} (loss="
                f"{float(loss.detach().item()):.5f})",
                file=_sys.stderr, flush=True,
            )
        return {
            "train/fn_frontier_loss": float(loss.detach().item()),
            "train/fn_frontier_pairs": 1.0,
        }

    def _train_forward_noiser_tf(self, rollout1_chunk, info) -> dict:
        """teacher_feat FN training step (separate FN backward). Builds the
        flattened (rollout1_chunk, rollout2_chunk) pairs, runs FN on the
        rollout1 chunks (step-unconditioned), and backprops the teacher-
        feature distribution-match into the FN. DDP-balanced: every rank
        runs at least one FN forward+backward (zero-anchor on bail)."""
        m = self.model
        if not (
            getattr(m, "forward_noiser_enabled", False)
            and getattr(m, "forward_noiser_loss_mode", "mse") == "teacher_feat"
            and getattr(m, "forward_noiser", None) is not None
        ):
            return {}
        npb = int(m.num_frame_per_block)
        proj = (
            getattr(self.r3gan_disc, "projector", None)
            if self.r3gan_disc is not None else None
        )

        def _anchor(reason: str) -> dict:
            # Keep FN DDP participation identical across ranks: ALWAYS run a
            # zero FN forward+backward so its grad-bucket all-reduce fires on
            # every rank, regardless of why this rank bailed. If a rank ran
            # the real backward while another ran nothing, the FN DDP reducer
            # would hang — so the anchor is unconditional. The FN is fully
            # convolutional, so a minimal synthetic input still produces a
            # gradient on every parameter (bucket shapes are param-shaped and
            # input-size-independent; batch size need not match across ranks).
            fn = m.forward_noiser
            fn_inner = fn.module if hasattr(fn, "module") else fn
            p0 = next(fn.parameters())
            if (rollout1_chunk is not None
                    and int(rollout1_chunk.shape[1]) >= npb):
                x0 = rollout1_chunk[:, :npb].detach()
            else:
                # No usable rollout1 chunk — synthesize a minimal zero input.
                C = int(getattr(fn_inner, "latent_channels", 16))
                B = int(rollout1_chunk.shape[0]) if rollout1_chunk is not None else 1
                x0 = torch.zeros(
                    (B, npb, C, 8, 8), device=p0.device, dtype=p0.dtype,
                )
            cz = torch.zeros(
                (x0.shape[0],), dtype=torch.long, device=x0.device,
            )
            z = fn(x0, cz, residual=True)
            (z.sum() * 0.0).backward()
            return {"train/fn_tf_loss": 0.0, "train/fn_tf_pairs": 0.0,
                    "train/fn_tf_skipped": 1.0}

        if proj is None:
            return _anchor("no_projector")
        s = getattr(m, "streaming_state", None)
        if not isinstance(s, dict):
            return _anchor("no_state")
        r2 = s.get("rollout2_x0")
        r2_abs = s.get("rollout2_abs_frame_start")
        if r2 is None or r2_abs is None:
            return _anchor("no_rollout2")
        r2_abs = int(r2_abs)
        r2_total = int(r2.shape[1])
        # rollout1 input: prefer the t=60 flash chunk (cleaner) like the
        # MSE path; both are detached (FN input is never grad-on upstream).
        flash = info.get("flash_dmd_gan_x0")
        r1 = flash.detach() if flash is not None else rollout1_chunk
        if r1 is None or int(r1.shape[1]) % npb != 0:
            return _anchor("bad_r1")
        n_chunks = int(r1.shape[1]) // npb
        abs_new_start = int(info.get("abs_frame_start", 0))
        overlap = int(info.get("overlap", 0))
        chunk_abs_start = abs_new_start - overlap

        # ``forward_noiser_reverse`` (h1): flip the mapping the FN learns.
        #   forward (default): FN(clean/rollout1) -> drifted/rollout2  (ADD a
        #                      drift step; a learned CARN that reproduces the
        #                      rollout's per-step explode/harmonise modes).
        #   reverse  (=true):  FN(drifted/rollout2) -> clean/rollout1  (REMOVE
        #                      a drift step; a learned de-CARN denoiser).
        #                      Applied to GT it suppresses those modes from
        #                      the real distribution (matched-pool gt_both).
        _reverse = bool(getattr(m, "forward_noiser_reverse", False))

        # ===== carn_recurse=False: CUMULATIVE single-call FN training =====
        # forward: FN(clean GT chunk, carn_step=L) -> student-drifted chunk @L,
        # in ONE conditioned call (no rollout2 / no +1 composition). The
        # student's rollout1 chunk at abs position p IS the level-L(p) drift of
        # the clean GT at p; we pair them and condition on L(p).
        # reverse:  FN(student-drifted chunk @L, carn_step=L) -> clean GT — a
        # de-CARN denoiser conditioned on the input's drift level. Application
        # (_apply_forward_noiser_to_gt / matched-pool gt_both) passes a real
        # level (forward_noiser_apply_gt_level, default 1) so the FN removes
        # one drift-level's worth of modes. Level-0 chunks carry no drift, so
        # they are skipped (the FN stays identity at step 0) either direction.
        if not bool(getattr(m, "carn_recurse", True)):
            ride_win = s.get("ride_latents_window")
            if ride_win is None:
                return _anchor("no_ride_window_cumulative")
            num_seed = int(m.dmd_context_clean_frames // npb)
            ride_T = int(ride_win.shape[1])
            Bc = int(r1.shape[0])
            cum_inputs, cum_targets, carn_levels = [], [], []
            for c in range(n_chunks):
                f0, f1 = c * npb, c * npb + npb
                a0, a1 = chunk_abs_start + f0, chunk_abs_start + f1
                if a0 < 0 or a1 > ride_T:
                    continue
                lvl = max(0, (a0 // npb) - (num_seed - 1))
                if lvl <= 0:
                    continue  # level-0: no drift to learn (FN identity).
                gt_chunk = ride_win[:, a0:a1].to(
                    dtype=r1.dtype, device=r1.device).detach()
                drift_chunk = r1[:, f0:f1].detach()
                if _reverse:
                    cum_inputs.append(drift_chunk)   # drifted @L  -> input
                    cum_targets.append(gt_chunk)     # clean GT    -> target
                else:
                    cum_inputs.append(gt_chunk)
                    cum_targets.append(drift_chunk)
                carn_levels.append(int(lvl))
            if not cum_inputs:
                return _anchor("no_pairs_cumulative")
            fn_in = torch.cat(cum_inputs, dim=0)          # [N*B, npb, C, H, W]
            cum_tg = torch.cat(cum_targets, dim=0)
            carn = torch.cat([
                torch.full((Bc,), lvl, dtype=torch.long, device=fn_in.device)
                for lvl in carn_levels
            ])
            fn_out = m.forward_noiser(fn_in, carn, residual=True)  # grad-on
            loss = self._fn_teacher_feat_loss(fn_out, cum_tg, proj)
            loss.backward()
            return {
                "train/fn_tf_loss": float(loss.detach().item()),
                "train/fn_tf_pairs": float(len(cum_inputs)),
                "train/fn_tf_skipped": 0.0,
                "train/fn_tf_cumulative": 1.0,
                "train/fn_tf_reverse": 1.0 if _reverse else 0.0,
            }

        fn_inputs, fn_targets = [], []
        _pair_abs: list = []
        for c in range(n_chunks):
            f0, f1 = c * npb, c * npb + npb
            a0, a1 = chunk_abs_start + f0, chunk_abs_start + f1
            if a0 < r2_abs or a1 > r2_abs + r2_total:
                continue
            s0 = a0 - r2_abs
            r1_chunk = r1[:, f0:f1].detach()
            r2_chunk = r2[:, s0:s0 + npb].to(
                dtype=r1.dtype, device=r1.device).detach()
            _pair_abs.append(int(a0))
            if _reverse:
                fn_inputs.append(r2_chunk)   # input  = rollout2 (drifted)
                fn_targets.append(r1_chunk)  # target = rollout1 (cleaner)
            else:
                fn_inputs.append(r1_chunk)
                fn_targets.append(r2_chunk)
        if not fn_inputs:
            return _anchor("no_pairs")
        # Probe: where the SETUP-WINDOW FN pairs live (abs ride frames).
        # Expected under rolling: only each ride's FIRST window yields
        # pairs (rollout2 covers the setup span) — the frontier pairs
        # ([FN-PAIR]) are the ones that advance with the generator.
        if (getattr(self, "is_main_process", True)
                and getattr(self, "_fn_tf_pos_dbg", 0) < 6):
            self._fn_tf_pos_dbg = getattr(self, "_fn_tf_pos_dbg", 0) + 1
            import sys as _sys
            print(
                f"[FN-TF] setup-window pairs n={len(_pair_abs)} "
                f"r1_abs={_pair_abs} r2_span=[{r2_abs},{r2_abs + r2_total})",
                file=_sys.stderr, flush=True,
            )
        fn_in = torch.cat(fn_inputs, dim=0)       # [N*B, npb, C, H, W]
        fn_tg = torch.cat(fn_targets, dim=0)
        # Step-UNCONDITIONED: carn_step=0 always (generic +/-1 shift).
        carn = torch.zeros(
            (fn_in.shape[0],), dtype=torch.long, device=fn_in.device,
        )
        fn_out = m.forward_noiser(fn_in, carn, residual=True)  # grad-on
        loss = self._fn_teacher_feat_loss(fn_out, fn_tg, proj)
        loss.backward()
        return {
            "train/fn_tf_loss": float(loss.detach().item()),
            "train/fn_tf_pairs": float(len(fn_inputs)),
            "train/fn_tf_skipped": 0.0,
            "train/fn_tf_reverse": 1.0 if _reverse else 0.0,
        }

    #     "student vs teacher's clean data".
    #   * "adjacent_chunks" (ASD-style): real = chunk_i, fake =
    #     chunk_{i+1} from the same gen rollout. Pushes
    #     chunk_{i+1}'s distribution toward chunk_i's.
    # When both flags are on, two D updates run per iter and both
    # gen-side losses sum into the generator's total.
    def _compute_ladd_losses(
        self,
        pred_image: torch.Tensor,
        gt_latents_window: torch.Tensor,
        current_step: int,
        flash_dmd_gan_x0: Optional[torch.Tensor] = None,
    ) -> tuple:
        device = pred_image.device
        zero = torch.zeros((), device=device, dtype=torch.float32)
        if not self.gan_enabled or self.r3gan_disc is None:
            return zero, {}

        # Source latent for the FAKE side (gradient-bearing for the gen).
        if flash_dmd_gan_x0 is not None:
            src_grad = flash_dmd_gan_x0.to(torch.float32)
        else:
            src_grad = pred_image.to(torch.float32)
        src_detached = src_grad.detach()
        # GT latent for the REAL side in gt_vs_fake mode. Always detached.
        gt_detached = gt_latents_window.detach().to(torch.float32)

        # Mode toggles. Defaults preserve backward compat with v28:
        # adjacent on by default, gt_vs_fake off by default.
        gt_vs_fake_enabled = bool(
            getattr(self.model, "ladd_gt_vs_fake_enabled", False)
        )
        adj_enabled = bool(
            getattr(self.model, "ladd_adjacent_chunks_enabled", True)
        )
        # gt_transition (v28G): action-conditioned GT-supervised adjacent-
        # chunk transition matching. Concatenates (chunk_n, chunk_{n+1})
        # along the F dim, feeds GT on the real side and student on the
        # fake side. The WAN teacher projector already conditions on per-
        # frame action_modulation, so the disc sees the action context
        # naturally — no extra action-projector needed in the disc.
        # Strictly more informative than adjacent_chunks (which is
        # student-vs-student, action-blind); typically run with
        # adjacent_chunks disabled.
        gt_xn_enabled = bool(
            getattr(self.model, "ladd_gt_transition_enabled", False)
        )
        if getattr(self, "_ladd_diag_n", 0) < 4:
            self._ladd_diag_n = getattr(self, "_ladd_diag_n", 0) + 1
            import sys as _sys
            print(
                f"[LADD-DIAG] _compute_ladd_losses REACHED: gt_vs_fake={gt_vs_fake_enabled} "
                f"adj={adj_enabled} gt_xn={gt_xn_enabled} "
                f"(model.ladd_gt_transition_enabled="
                f"{getattr(self.model, 'ladd_gt_transition_enabled', 'MISSING')})",
                file=_sys.stderr, flush=True,
            )
        if not (gt_vs_fake_enabled or adj_enabled or gt_xn_enabled):
            return zero, {}

        gen_loss_total = zero
        logs: dict = {}

        # Two-phase orchestration when MORE THAN ONE pair-mode is
        # enabled (gt_vs_fake + adjacent_chunks): we must run ALL
        # D-updates before ANY gen-side forward, otherwise mode-2's
        # D-update mutates ``spectral_norm._u`` between mode-1's
        # gen-side forward and the outer ``generator_loss.backward()``
        # → version-counter mismatch on saved tensors. The
        # ``_ladd_run_pair_mode`` ``phase`` parameter implements the
        # split: "d_only" runs only the D-update, "g_only" runs only
        # the gen-side, "both" (default) runs both in sequence. The
        # G-side ALSO runs in disc.eval() mode (set inside
        # ``_ladd_run_pair_mode``) so spectral_norm doesn't mutate
        # the running buffers during its forward.
        enabled_modes: List[Tuple[str, torch.Tensor, str]] = []
        if gt_vs_fake_enabled:
            enabled_modes.append(("gt_vs_fake", gt_detached, "_gt"))
        if adj_enabled:
            enabled_modes.append(("adjacent_chunks", src_detached, "_adj"))
        if gt_xn_enabled:
            # Real source is GT for gt_transition (the whole point).
            enabled_modes.append(("gt_transition", gt_detached, "_gtxn"))

        two_phase = len(enabled_modes) > 1

        if two_phase:
            # Phase 1: all D-updates (each backward + step immediately).
            for mode_name, real_src_t, _suffix in enabled_modes:
                self._ladd_run_pair_mode(
                    real_src=real_src_t,
                    fake_src_grad=src_grad,
                    fake_src_detached=src_detached,
                    pred_image_dtype=pred_image.dtype,
                    pair_mode=mode_name,
                    current_step=current_step,
                    phase="d_only",
                )

        # Phase 2: per-mode gen-side forwards (or single combined
        # call when only one mode is enabled).
        for mode_name, real_src_t, suffix in enabled_modes:
            g_loss, partial_logs = self._ladd_run_pair_mode(
                real_src=real_src_t,
                fake_src_grad=src_grad,
                fake_src_detached=src_detached,
                pred_image_dtype=pred_image.dtype,
                pair_mode=mode_name,
                current_step=current_step,
                phase="g_only" if two_phase else "both",
            )
            if mode_name == "gt_vs_fake":
                w = float(getattr(self.model, "ladd_gt_vs_fake_weight", 1.0))
            elif mode_name == "gt_transition":
                w = float(
                    getattr(self.model, "ladd_gt_transition_weight", 1.0)
                )
            else:
                w = float(getattr(self.model, "ladd_adjacent_chunks_weight", 1.0))
            gen_loss_total = gen_loss_total + w * g_loss
            for k, v in partial_logs.items():
                logs[k + suffix] = v

        # Top-level (non-suffixed) R1 traces so the wandb plot is
        # findable as ``train/r3gan_r1*`` without the ``_gt`` / ``_adj``
        # suffix dance. Aggregate via mean across pair modes — both
        # modes share the same disc weights so their R1 estimates
        # should be comparable. The fired flag is OR'd (firing on EITHER
        # mode is enough to count this iter as "R1 fired"). ``grad_sq``
        # uses nan-aware mean so a non-firing mode doesn't pull the
        # combined value to ~half.
        import math as _math
        r1_keys = [k for k in logs if "r3gan_r1_grad_sq_" in k]
        if r1_keys:
            valid_vals = [
                logs[k] for k in r1_keys
                if isinstance(logs[k], float) and not _math.isnan(logs[k])
            ]
            if valid_vals:
                logs["train/r3gan_r1_grad_sq"] = sum(valid_vals) / len(valid_vals)
            else:
                logs["train/r3gan_r1_grad_sq"] = float("nan")
        fired_keys = [k for k in logs if "r3gan_r1_fired_" in k]
        if fired_keys:
            logs["train/r3gan_r1_fired"] = max(logs[k] for k in fired_keys)
        # Top-level r3gan_r1 = mean of the γ-weighted per-mode losses.
        r1_loss_keys = [
            k for k in logs
            if k in (
                "train/r3gan_r1_gt",
                "train/r3gan_r1_adj",
                "train/r3gan_r1_gtxn",
            )
        ]
        if r1_loss_keys:
            logs["train/r3gan_r1"] = sum(logs[k] for k in r1_loss_keys) / len(r1_loss_keys)
        # R2 top-level traces — mirror the R1 aggregation above so the
        # ``train/r3gan_r2*`` keys are findable without the per-mode
        # suffix dance.
        r2_keys = [k for k in logs if "r3gan_r2_grad_sq_" in k]
        if r2_keys:
            valid_vals = [
                logs[k] for k in r2_keys
                if isinstance(logs[k], float) and not _math.isnan(logs[k])
            ]
            if valid_vals:
                logs["train/r3gan_r2_grad_sq"] = sum(valid_vals) / len(valid_vals)
            else:
                logs["train/r3gan_r2_grad_sq"] = float("nan")
        r2_fired_keys = [k for k in logs if "r3gan_r2_fired_" in k]
        if r2_fired_keys:
            logs["train/r3gan_r2_fired"] = max(logs[k] for k in r2_fired_keys)
        r2_loss_keys = [
            k for k in logs
            if k in (
                "train/r3gan_r2_gt",
                "train/r3gan_r2_adj",
                "train/r3gan_r2_gtxn",
            )
        ]
        if r2_loss_keys:
            logs["train/r3gan_r2"] = sum(logs[k] for k in r2_loss_keys) / len(r2_loss_keys)
        return gen_loss_total, logs

    def _ladd_r2_knobs(self, r1_every_n: int, r1_sigma: float):
        """Read the R2 (fake-side gradient penalty) knobs off the model.

        Returns ``(gamma, every_n, phase_offset, sigma)``. ``every_n``
        and ``sigma`` fall back to the R1 values when unset. ``gamma=0``
        (default) means R2 is OFF. The phase offset desynchronizes R1
        and R2 firing so their perturbed-input forwards don't stack in
        the same D-update (OOM guard) — see ``ladd_r2_phase_offset``.
        """
        gamma = float(getattr(self.model, "ladd_r2_gamma", 0.0))
        every_n = max(1, int(
            getattr(self.model, "ladd_r2_every_n_steps", r1_every_n)))
        offset = int(getattr(self.model, "ladd_r2_phase_offset", 1))
        sigma = float(getattr(self.model, "ladd_r2_sigma", r1_sigma))
        return gamma, every_n, offset, sigma

    @staticmethod
    def _ladd_r2_fires(
        gamma: float, every_n: int, offset: int, current_step: int,
    ) -> bool:
        """R2 fires when its gamma is on AND the step is on R2's
        (offset) lazy cadence. Kept identical across all three D-update
        branches so the offset-vs-R1 OOM guarantee holds everywhere."""
        return gamma > 0.0 and ((current_step - offset) % every_n == 0)

    def _ladd_run_pair_mode(
        self,
        real_src: torch.Tensor,
        fake_src_grad: torch.Tensor,
        fake_src_detached: torch.Tensor,
        pred_image_dtype: torch.dtype,
        pair_mode: str,
        current_step: int,
        phase: str = "both",
    ) -> tuple:
        if getattr(self, "_carn_diag_n", 0) < 6:
            self._carn_diag_n = getattr(self, "_carn_diag_n", 0) + 1
            import sys as _sys
            print(
                f"[CARN-DIAG] _ladd_run_pair_mode mode={pair_mode!r} phase={phase} "
                f"carn_knob_model={getattr(self.model, 'ladd_gt_transition_carn_former', 'MISSING')} "
                f"carn_knob_cfg={getattr(self.config, 'ladd_gt_transition_carn_former', 'MISSING')} "
                f"fn={getattr(self.model, 'forward_noiser', None) is not None}",
                file=_sys.stderr, flush=True,
            )
        """Single D-update + gen-side loss for one pair-construction mode.

        Args:
            real_src: detached tensor that supplies the REAL chunks.
                For ``gt_vs_fake`` this is the GT latent window; for
                ``adjacent_chunks`` it is the (detached) source latent.
            fake_src_grad: gradient-bearing source for the FAKE chunks
                on the gen-side forward.
            fake_src_detached: detached source for the FAKE chunks on
                the D-side forward.
            pair_mode: ``"gt_vs_fake"`` or ``"adjacent_chunks"``.
            current_step: training step (for warmup ramp + RNG seed).

        Returns ``(gen_gan_loss, logs)``. ``gen_gan_loss`` is graph-
        attached through ``fake_src_grad``.
        """
        from model.r3gan import rpgan_d_loss, rpgan_g_loss, r1_penalty
        from model.ladd_disc import latent_diff_augment

        device = fake_src_grad.device
        zero = torch.zeros((), device=device, dtype=torch.float32)
        disc_for_update = (
            self.r3gan_disc_ddp
            if self.r3gan_disc_ddp is not None
            else self.r3gan_disc
        )
        disc_for_guidance = self.r3gan_disc

        npb = int(getattr(self.model, "num_frame_per_block", 3))
        F_total = int(fake_src_grad.shape[1])
        if F_total < npb:
            return zero, {"train/ladd_n_pairs": 0.0}
        n_chunks = F_total // npb

        # Per-mode pair index lists. Each pair = (real_idx, fake_idx)
        # where the indices are chunk positions in real_src / fake_src.
        # ``chunks_per_pair`` controls whether the disc sees one chunk
        # (gt_vs_fake / adjacent_chunks) or a pair-of-chunks concatenated
        # along F (gt_transition).
        chunks_per_pair = 1
        if pair_mode == "gt_vs_fake":
            all_pairs = [(i, i) for i in range(n_chunks)]
        elif pair_mode == "adjacent_chunks":
            if n_chunks < 2:
                return zero, {"train/ladd_n_pairs": 0.0}
            all_pairs = [(i, i + 1) for i in range(n_chunks - 1)]
        elif pair_mode == "gt_transition":
            # Action-conditioned GT-supervised adjacent-chunk matching.
            # Each "pair" is the chunk-pair (i, i+1); the disc input is
            # concat(chunk_i, chunk_{i+1}) along the F dim, doubling the
            # frames-per-sample from npb to 2*npb. Real side uses GT,
            # fake side uses student. The pair indices store (i, i+1)
            # so the action-token slicing below can use the same lo_r /
            # lo_f layout (and we cover both chunks per pair).
            if n_chunks < 2:
                return zero, {"train/ladd_n_pairs": 0.0}
            all_pairs = [(i, i + 1) for i in range(n_chunks - 1)]
            chunks_per_pair = 2
        else:
            raise ValueError(
                f"_ladd_run_pair_mode: unknown pair_mode={pair_mode!r}."
            )

        # All-pairs (style, not position) RpGAN: compare EVERY real to
        # EVERY fake in the loss (B_real x B_fake relativistic terms)
        # rather than position-matched. The disc forward is unchanged
        # (same B reals + B fakes); only the loss reduction becomes an
        # outer product, so the extra signal is ~free in memory. Enabled
        # per-mode: gt_vs_fake (chunk vs chunk) via
        # ``ladd_gt_vs_fake_all_pairs``, gt_transition (transition-pair vs
        # transition-pair) via ``ladd_gt_transition_all_pairs``. Each
        # logit stays bound to its own chunk/transition's action through
        # the forward, so the outer product only recombines correctly
        # self-conditioned scalars. adjacent_chunks keeps positional
        # pairing. ``_d_loss_fn`` / ``_g_loss_fn`` are used at every
        # RpGAN reduction site below.
        _all_pairs_mode = (
            (
                pair_mode == "gt_vs_fake"
                and bool(getattr(
                    self.model, "ladd_gt_vs_fake_all_pairs", False))
            )
            or (
                pair_mode == "gt_transition"
                and bool(getattr(
                    self.model, "ladd_gt_transition_all_pairs", False))
            )
        )
        _d_loss_fn = rpgan_d_loss_allpairs if _all_pairs_mode else rpgan_d_loss
        _g_loss_fn = rpgan_g_loss_allpairs if _all_pairs_mode else rpgan_g_loss

        # Mismatched all-pairs (N_real != N_fake) + resample-per-update.
        # N_real GT chunks (sampled fresh each D-update from the full ride
        # window) vs the N_fake student chunks. Only meaningful with
        # all-pairs (the loss is a non-square outer product). Handled by a
        # dedicated self-contained branch below (the matched path is left
        # untouched). 0 = off.
        _n_real = 0
        if _all_pairs_mode and pair_mode == "gt_vs_fake":
            _n_real = int(getattr(self.model, "ladd_gt_vs_fake_n_real", 0))
        elif _all_pairs_mode and pair_mode == "gt_transition":
            _n_real = int(getattr(self.model, "ladd_gt_transition_n_real", 0))
        _mismatched = _n_real > 0

        # Optional pair budget. ``ladd_pair_selection`` decides which pairs
        # when capped: "first" = the first N rolled chunks (deterministic,
        # lowest drift); "random" = random subset seeded by the step (same
        # on all ranks, varies each step). all_pairs is ordered by chunk
        # index, so all_pairs[:N] == the first N rolled chunks.
        ladd_pairs_per_step = int(
            getattr(self.model, "ladd_pairs_per_step", 0)
        )
        pair_selection = str(
            getattr(self.model, "ladd_pair_selection", "random")
        ).lower()
        if (
            ladd_pairs_per_step > 0
            and ladd_pairs_per_step < len(all_pairs)
        ):
            if pair_selection == "first":
                pairs = all_pairs[:ladd_pairs_per_step]
            else:
                g = torch.Generator(device="cpu").manual_seed(int(current_step))
                idx = torch.randperm(len(all_pairs), generator=g)[
                    :ladd_pairs_per_step
                ].tolist()
                pairs = [all_pairs[i] for i in sorted(idx)]
        else:
            pairs = all_pairs
        n_pairs = len(pairs)

        # Wide-real (all-pairs gt_vs_fake only): draw the REAL GT chunks
        # from the FULL loaded ride window (seed + rollout + post-window)
        # rather than the position-matched scored slice, for more varied
        # GT in the style comparison. ``real_positions`` (one absolute
        # chunk index per pair) + ``real_src_eff`` drive BOTH the real-
        # latent slice and the real-action slice below; both index the
        # same-coordinate ride windows (ride_latents_window /
        # ride_actions_window share the ride offset s) at the same
        # absolute frame, so latent and action stay co-located. Counts
        # stay == n_pairs (== fake count), so the batched forward and the
        # R1 finite-diff split are unaffected. Non-wide path reproduces
        # the original behaviour exactly (real_src_eff=real_src,
        # real_positions=[i for (i,_) in pairs]).
        _wide_real = (
            _all_pairs_mode
            and pair_mode == "gt_vs_fake"
            and chunks_per_pair == 1
            and bool(getattr(self.model, "ladd_gt_vs_fake_wide_real", False))
        )
        real_src_eff = real_src
        real_positions = [i for (i, _) in pairs]
        if _wide_real:
            _ss = getattr(self.model, "streaming_state", None)
            _wide_win = (
                _ss.get("ride_latents_window")
                if isinstance(_ss, dict) else None
            )
            if _wide_win is not None and int(_wide_win.shape[1]) >= npb:
                real_src_eff = _wide_win.detach().to(real_src.dtype)
                _w_chunks = int(_wide_win.shape[1] // npb)
                # Step-seeded sample (same positions across ranks is fine —
                # each rank holds a different ride, so GT content still
                # differs; the COUNT is fixed = n_pairs so disc-forward
                # shapes match across ranks → DDP-safe). Cycle if the
                # window holds fewer chunks than n_pairs.
                _g = torch.Generator(device="cpu").manual_seed(
                    int(current_step) * 131 + 17
                )
                _perm = torch.randperm(_w_chunks, generator=_g).tolist()
                real_positions = [
                    _perm[k % _w_chunks] for k in range(n_pairs)
                ]
            else:
                _wide_real = False  # window unavailable → fall back

        def _slice(t, i):
            return t[:, i * npb:(i + 1) * npb]

        def _slice_pair(t, i, j):
            # F-concat both indices: [B, 2*npb, C, H, W].
            return torch.cat([_slice(t, i), _slice(t, j)], dim=1)

        # Magnitude-equalize the two members of a gt_transition pair
        # (former [:, :npb], latter [:, npb:]) to their common average
        # MEAN MAGNITUDE (mean of |x| — the energy/brightness proxy, NOT
        # the signed mean which is direction+magnitude and can cancel to
        # ~0). Equalizing a magnitude requires SCALING (not shifting):
        # scale the lower-magnitude chunk UP and the higher one DOWN by
        # the matching factor so both reach the pair's average magnitude,
        # preserving the pair's overall magnitude. Removes the inter-chunk
        # magnitude (brightness) drift across the transition from the
        # disc's view — counters the transition GAN's progressive
        # brightening. Differentiable, so on the grad-on fake side it also
        # zeroes the gen-side gradient pushing the inter-member magnitude.
        _mag_mode = (
            str(getattr(self.model, "ladd_gt_transition_mag_norm", "")).lower()
            if chunks_per_pair == 2 else ""
        )
        _mean_eq = (_mag_mode in ("m1", "m1m2")) or (
            chunks_per_pair == 2
            and bool(getattr(
                self.model, "ladd_gt_transition_mean_equalize", False))
        )

        def _mean_equalize_pair(pair):
            eps = 1e-6
            if _mag_mode in ("m1", "m1m2"):
                # Per-pair, per-channel normalization (reduce over F,H,W;
                # keep C). Each pair uses its OWN stats -> per-video
                # distributions handled correctly. m1: divide each channel
                # by its RMS (unit energy, keeps contrast). m1m2:
                # standardize each channel (unit energy + contrast).
                if _mag_mode == "m1":
                    rms = pair.pow(2).mean(
                        dim=[1, 3, 4], keepdim=True).add(eps).sqrt()
                    return pair / rms
                mu = pair.mean(dim=[1, 3, 4], keepdim=True)
                sd = pair.std(
                    dim=[1, 3, 4], unbiased=False, keepdim=True).add(eps)
                return (pair - mu) / sd
            # mean_equalize: scale the two members (former/latter) to the
            # pair's common mean-magnitude (mean of |x|); preserves the
            # pair's overall magnitude, removes only the inner drift.
            fmr = pair[:, :npb]
            ltr = pair[:, npb:]
            a_f = fmr.abs().mean(dim=[1, 2, 3, 4], keepdim=True)
            a_l = ltr.abs().mean(dim=[1, 2, 3, 4], keepdim=True)
            a = 0.5 * (a_f + a_l)
            return torch.cat(
                [fmr * (a / (a_f + eps)), ltr * (a / (a_l + eps))], dim=1)

        # CARN-former (ladd_gt_transition_carn_former): degrade the FORMER
        # chunk of each GT transition pair through the trained forward
        # noiser, leaving the latter clean -> the disc sees REAL as a
        # "CARN-degraded -> clean" (self-correcting) transition. Applied
        # under no_grad on the detached real side (does NOT train the FN
        # here; the FN trains via the rollout2 critic loss). Identity until
        # the FN has trained (zero-init), so the effect ramps in.
        # Robust knob read: prefer self.model, fall back to self.config
        # (the override lives on the config object; self.model may not
        # surface every arg).
        _carn_knob = bool(
            getattr(self.model, "ladd_gt_transition_carn_former", None)
            or getattr(self.config, "ladd_gt_transition_carn_former", False)
        )
        # carn_former applies the learned ForwardNoiser (FN) to the GT
        # former chunk: the FN has learned the rollout1->rollout2 "+1
        # cartoon shift" (sliced-Wasserstein teacher-feature match), so
        # FN(GT_former) = "GT with one AR step of texture drift". The real
        # anchor becomes [cartoon-shifted GT former -> clean GT latter] = a
        # self-correcting (de-cartoon) transition the student must match.
        # Latent-space (no wavelet); requires the FN to be present.
        _carn_former_on = (
            pair_mode == "gt_transition"
            and _carn_knob
            and getattr(self.model, "forward_noiser", None) is not None
        )
        if _carn_former_on and getattr(self, "_carn_former_dbg", 0) < 2:
            self._carn_former_dbg = getattr(self, "_carn_former_dbg", 0) + 1
            import sys as _sys
            print(
                "[CARN-FORMER] ON: gt_transition GT former chunk pushed +1 "
                "cartoon step by the learned ForwardNoiser.",
                file=_sys.stderr, flush=True,
            )

        def _carn_former(x):
            n_steps = int(getattr(self.model, "ladd_gt_transition_carn_steps", 1))
            # FIX B (random real-former CARN level): draw the level per PAIR
            # uniformly in [0, max_level] (0 = clean former). The disc then
            # sees real formers at every degradation level (incl. clean), so
            # it can't pull the student toward a single fixed CARN level —
            # it must key on the transition (clean latter | any former).
            if bool(getattr(
                self.model, "ladd_gt_transition_carn_random_level", False)):
                _maxlvl = int(getattr(
                    self.model, "ladd_gt_transition_carn_max_level", n_steps))
                n_steps = int(torch.randint(0, max(1, _maxlvl) + 1, (1,)).item())
                if n_steps <= 0:
                    return x.detach()  # level 0 -> clean GT former, no CARN
            # Step-unconditioned FN: every apply passes carn_step=0 (the FN
            # learned one generic "+1 shift"). Otherwise pass the iteration
            # index as the CARN level (legacy).
            _uncond = bool(getattr(
                self.model, "forward_noiser_step_unconditioned", False))
            _recurse = bool(getattr(self.model, "carn_recurse", True))
            x0 = x  # original former (pre-carn) for moment restoration
            with torch.no_grad():
                if not _recurse:
                    # SINGLE-CALL mode (consistent with the aux CARN): one
                    # conditioned call from the clean former, carn_step =
                    # the target level (= n_steps, default 1). No recursion.
                    cs = torch.full(
                        (x.shape[0],), int(max(1, n_steps)),
                        dtype=torch.long, device=x.device,
                    )
                    x = self.model.forward_noiser(x0, cs, residual=True)
                else:
                    for _s in range(max(1, n_steps)):
                        cs = torch.full(
                            (x.shape[0],), 0 if _uncond else _s,
                            dtype=torch.long, device=x.device,
                        )
                        x = self.model.forward_noiser(x, cs, residual=True)
                # MOMENT-PRESERVING carn (texture only, NO stats — see the
                # "CARN = texture, not stats" directive). The FN trains on
                # un-normalized rollout1->rollout2, where rollout2 runs hot
                # (one more drift step), so its output drifts BRIGHTER. If
                # that reaches the GT former, ``_mean_equalize_pair`` below
                # rescales BOTH members to their common magnitude a=0.5*(a_f
                # +a_l) -> a hot former drags the clean latter UP and lifts
                # the whole REAL pair above the fake pair -> the disc learns
                # "brighter=real" -> student brightens -> rollout2 brighter
                # -> FN brighter -> WHITE COLLAPSE. Restore the original
                # former's (a) per-channel DC mean and (b) per-sample mean-
                # magnitude (the exact stat _mean_equalize_pair keys on), so
                # carn changes only HF/texture STRUCTURE and the equalize
                # sees an unchanged former. Scalar rescale preserves the
                # texture pattern.
                eps = 1e-6
                mc_in = x0.mean(dim=[1, 3, 4], keepdim=True)
                mc_out = x.mean(dim=[1, 3, 4], keepdim=True)
                x = x - mc_out + mc_in
                a_in = x0.abs().mean(dim=[1, 2, 3, 4], keepdim=True)
                a_out = x.abs().mean(dim=[1, 2, 3, 4], keepdim=True)
                x = x * (a_in / (a_out + eps))
            return x.detach()

        def _slice_pair_carn(t, i, j):
            former = _carn_former(_slice(t, i))
            return torch.cat([former, _slice(t, j)], dim=1)

        if chunks_per_pair == 2:
            # gt_transition: real and fake BOTH cover the chunk-pair
            # (i, i+1). Real uses GT (real_src=gt_detached), fake uses
            # student (fake_src_*). Same i,j indices on both sides — the
            # disc compares "GT's chunk pair" vs "student's chunk pair".
            _real_pair_fn = _slice_pair_carn if _carn_former_on else _slice_pair
            real_chunks_det = torch.cat(
                [_real_pair_fn(real_src, i, j) for (i, j) in pairs], dim=0,
            )
            fake_chunks_det = torch.cat(
                [_slice_pair(fake_src_detached, i, j) for (i, j) in pairs],
                dim=0,
            )
            _gen_detach_former = bool(getattr(
                self.model, "ladd_gt_transition_gen_detach_former", False))

            def _slice_pair_gen(t, i, j):
                # FIX A: on the gen-side fake pair, optionally DETACH the
                # former so the GAN gradient flows only to the LATTER (the
                # transition target) — the student learns "given my drifted
                # former, make the next chunk clean" without being pushed to
                # degrade its own former toward the real-pair's CARN level.
                former = _slice(t, i)
                if _gen_detach_former:
                    former = former.detach()
                return torch.cat([former, _slice(t, j)], dim=1)

            fake_chunks_grad_tensor = torch.cat(
                [_slice_pair_gen(fake_src_grad, i, j) for (i, j) in pairs],
                dim=0,
            )
            if _mean_eq and _mag_mode in ("m1", "m1m2"):
                # Per-channel magnitude mode (opt-in): unchanged, per-tensor.
                real_chunks_det = _mean_equalize_pair(real_chunks_det)
                fake_chunks_det = _mean_equalize_pair(fake_chunks_det)
                fake_chunks_grad_tensor = _mean_equalize_pair(
                    fake_chunks_grad_tensor)
            elif _mean_eq:
                # CROSS-equalize the OVERALL level across the real and fake
                # pools so the disc gets NO absolute-brightness cue (kills
                # the white-collapse feedback). a* is the common per-row
                # target level, built from DETACHED magnitudes so the grad-on
                # fake gets a constant target. Two variants
                # (ladd_gt_transition_xeq_preserve_delta):
                #   * flatten (False, default): scale each of the 4 members
                #     INDEPENDENTLY to a* -> within-pair former<->latter
                #     brightness delta removed too; disc fully brightness-
                #     blind.
                #   * preserve_delta (True): scale each PAIR by ONE shared
                #     factor a*/a_pair -> real/fake absolute level equalized
                #     but the within-pair brightness TRANSITION ratio is
                #     preserved, so the disc still sees (and pushes the
                #     student to match GT's) transition brightness. No
                #     collapse because absolute level is still pinned.
                # Either way the rescale is a scalar per member/pair, so
                # texture STRUCTURE is preserved. real-row-i and fake-row-i
                # are the same chunk-pair position (GT vs student).
                _eps = 1e-6
                _preserve_delta = bool(
                    getattr(self.model,
                            "ladd_gt_transition_xeq_preserve_delta", None)
                    or getattr(self.config,
                               "ladd_gt_transition_xeq_preserve_delta", False)
                )
                _per_channel = bool(
                    getattr(self.model,
                            "ladd_gt_transition_xeq_per_channel", None)
                    or getattr(self.config,
                               "ladd_gt_transition_xeq_per_channel", False)
                )
                # Reduce over [F,H,W] keeping C (per-channel) or over
                # [F,C,H,W] (one scalar). keepdim=True so a_star / the
                # factors broadcast either way.
                _red = [1, 3, 4] if _per_channel else [1, 2, 3, 4]

                def _mags(t):
                    return (
                        t[:, :npb].abs().mean(dim=_red, keepdim=True),
                        t[:, npb:].abs().mean(dim=_red, keepdim=True),
                    )

                _da_rf, _da_rl = _mags(real_chunks_det.detach())
                _da_ff, _da_fl = _mags(fake_chunks_det.detach())
                if real_chunks_det.shape[0] == fake_chunks_det.shape[0]:
                    a_star = 0.25 * (_da_rf + _da_rl + _da_ff + _da_fl)  # per row
                else:
                    # Defensive (row-count mismatch): one shared target
                    # (per-channel if enabled, else scalar) so real and fake
                    # still share a level. mean over the stacked rows, keep C.
                    a_star = torch.cat(
                        [_da_rf, _da_rl, _da_ff, _da_fl], dim=0,
                    ).mean(dim=0, keepdim=True)

                if _preserve_delta:
                    def _xeq(t):
                        af, al = _mags(t)  # in-graph for the grad tensor
                        a_pair = 0.5 * (af + al)            # this pair's level
                        factor = a_star / (a_pair + _eps)   # ONE factor, both members
                        return t * factor                   # ratio (transition) kept
                else:
                    def _xeq(t):
                        af, al = _mags(t)  # in-graph for the grad tensor
                        return torch.cat(
                            [t[:, :npb] * (a_star / (af + _eps)),
                             t[:, npb:] * (a_star / (al + _eps))], dim=1,
                        )

                real_chunks_det = _xeq(real_chunks_det)
                fake_chunks_det = _xeq(fake_chunks_det)
                fake_chunks_grad_tensor = _xeq(fake_chunks_grad_tensor)

                # STD equalization (same cross-eq mechanism, on the 2nd
                # moment). Applied AFTER the mean-eq, CENTERED: scale each
                # member's deviations-from-mean to the common std s* and
                # re-add the mean, so it sets spread/contrast WITHOUT
                # disturbing the level. Honours the same per-channel /
                # preserve-delta options. With both mean+std eq on, the disc
                # input is per-channel first+second-moment-free -> the GAN
                # keys on pure texture, never brightness OR contrast.
                _std_eq = bool(
                    getattr(self.model,
                            "ladd_gt_transition_std_equalize", None)
                    or getattr(self.config,
                               "ladd_gt_transition_std_equalize", False)
                )
                if _std_eq:
                    def _stds(t):
                        return (
                            t[:, :npb].std(dim=_red, unbiased=False,
                                           keepdim=True),
                            t[:, npb:].std(dim=_red, unbiased=False,
                                           keepdim=True),
                        )

                    def _means(t):
                        return (
                            t[:, :npb].mean(dim=_red, keepdim=True),
                            t[:, npb:].mean(dim=_red, keepdim=True),
                        )

                    _ds_rf, _ds_rl = _stds(real_chunks_det.detach())
                    _ds_ff, _ds_fl = _stds(fake_chunks_det.detach())
                    if real_chunks_det.shape[0] == fake_chunks_det.shape[0]:
                        s_star = 0.25 * (_ds_rf + _ds_rl + _ds_ff + _ds_fl)
                    else:
                        s_star = torch.cat(
                            [_ds_rf, _ds_rl, _ds_ff, _ds_fl], dim=0,
                        ).mean(dim=0, keepdim=True)

                    if _preserve_delta:
                        def _xeq_std(t):
                            sf, sl = _stds(t)
                            mf, ml = _means(t)
                            s_pair = 0.5 * (sf + sl)
                            fac = s_star / (s_pair + _eps)  # ONE factor, both
                            return torch.cat(
                                [(t[:, :npb] - mf) * fac + mf,
                                 (t[:, npb:] - ml) * fac + ml], dim=1,
                            )
                    else:
                        def _xeq_std(t):
                            sf, sl = _stds(t)
                            mf, ml = _means(t)
                            return torch.cat(
                                [(t[:, :npb] - mf) * (s_star / (sf + _eps)) + mf,
                                 (t[:, npb:] - ml) * (s_star / (sl + _eps)) + ml],
                                dim=1,
                            )

                    real_chunks_det = _xeq_std(real_chunks_det)
                    fake_chunks_det = _xeq_std(fake_chunks_det)
                    fake_chunks_grad_tensor = _xeq_std(fake_chunks_grad_tensor)
        else:
            real_chunks_det = torch.cat(
                [_slice(real_src_eff, real_positions[k])
                 for k in range(n_pairs)], dim=0,
            )
            fake_chunks_det = torch.cat(
                [_slice(fake_src_detached, j) for (_, j) in pairs], dim=0,
            )
            fake_chunks_grad_tensor = torch.cat(
                [_slice(fake_src_grad, j) for (_, j) in pairs], dim=0,
            )

        # ----- Disc timestep -----
        # WAN with action-aware forward expects per-frame timestep
        # ``[B, F]`` so the time_projection output matches the per-frame
        # ``_action_modulation``'s ``[B*F, ...]`` reshape. Without F in
        # the timestep shape, time_projection outputs ``[B, hidden]``
        # while action_modulation is ``[B*F, hidden]`` and the
        # ``am_flat != e0`` shape check raises. Match the main DMD
        # path's convention.
        flash_on = bool(getattr(self.model, "flash_dmd_enabled", False))
        flash_t = int(getattr(self.model, "flash_dmd_gan_t", 60))
        # When the wavelet-HF stage is enabled, force the disc to
        # operate on CLEAN latents (disc_t_int=0). The wavelet step
        # decomposes its input into LL+LH+HL+HH sub-bands; if we noise
        # the input first, the diffusion noise leaks broadband into
        # the HF sub-bands and the disc ends up partly discriminating
        # noise rather than student-vs-GT HF structure (the WGSR
        # frequency-band restriction we're trying to enforce becomes
        # noise-band restriction). Match the original WGSR setup by
        # keeping the wavelet input clean.
        wavelet_on = bool(
            getattr(self.r3gan_disc, "wavelet_hf_enabled", False)
        )
        # Force clean disc input when requested (e.g. wavelet off but the
        # disc is meant to learn brightness/contrast, which t=flash_t noise
        # would swamp). Does NOT touch flash_dmd_gan_t (shared with main DMD).
        _disc_force_clean = bool(
            getattr(self.model, "ladd_disc_force_clean", None)
            or getattr(self.config, "ladd_disc_force_clean", False)
        )
        if wavelet_on or _disc_force_clean:
            disc_t_int = 0
        else:
            disc_t_int = flash_t if flash_on else 0
        bsz_eff = real_chunks_det.shape[0]
        # Per-frame timestep matches the disc input's F dim — which is
        # ``npb`` for single-chunk modes and ``2 * npb`` for gt_transition.
        t_frames = chunks_per_pair * npb
        t_disc = torch.full(
            (bsz_eff, t_frames), disc_t_int, dtype=torch.long, device=device,
        )

        def _add_disc_noise(x):
            if disc_t_int <= 0:
                return x
            scheduler = getattr(self.model, "scheduler", None)
            if scheduler is None or not hasattr(scheduler, "add_noise"):
                raise RuntimeError(
                    "_add_disc_noise: disc_t_int > 0 but the model has "
                    "no scheduler with ``add_noise`` available. Either "
                    "set ``self.model.scheduler`` to a usable scheduler "
                    "or set disc_t_int=0 (e.g. ``flash_dmd_enabled="
                    "False``) so the disc operates on clean inputs. "
                    "Silent fallback removed — this used to return ``x`` "
                    "unchanged and mask a misconfigured model."
                )
            eps = torch.randn_like(x)
            x_flat = x.flatten(0, 1)
            eps_flat = eps.flatten(0, 1)
            # t_disc is now [B, F]; flatten matches x_flat's [B*F] axis.
            t_per_frame = t_disc.flatten(0, 1)
            noisy_flat = scheduler.add_noise(x_flat, eps_flat, t_per_frame)
            return noisy_flat.unflatten(0, x.shape[:2])

        real_chunks_det_noisy = _add_disc_noise(real_chunks_det)
        fake_chunks_det_noisy = _add_disc_noise(fake_chunks_det)

        # DiffAugment — same per-sample randomness on real and fake.
        # Different seeds per mode so the augmentations decorrelate.
        diff_aug_policy = str(
            getattr(self.model, "ladd_diff_aug_policy", "")
        )
        seed_offset = 0 if pair_mode == "gt_vs_fake" else 7919
        if diff_aug_policy:
            real_chunks_det_noisy, fake_chunks_det_noisy = (
                latent_diff_augment(
                    real_chunks_det_noisy,
                    fake_chunks_det_noisy,
                    policy=diff_aug_policy,
                    seed=int(current_step) * 31 + seed_offset,
                )
            )

        # ----- Prompt embedding (broadcast across pairs) -----
        prompt_embeds = None
        pooled_prompt = None
        s_state = getattr(self.model, "streaming_state", None)
        if isinstance(s_state, dict):
            prompt_embeds = s_state.get("prompt_embeds")
        if prompt_embeds is None:
            raise RuntimeError(
                "_ladd_run_pair_mode: prompt_embeds not in "
                "streaming_state. setup_sequence must have run before "
                "the gen step."
            )
        if prompt_embeds.shape[0] != bsz_eff:
            reps = bsz_eff // prompt_embeds.shape[0]
            prompt_embeds_eff = prompt_embeds.repeat_interleave(reps, dim=0)
        else:
            prompt_embeds_eff = prompt_embeds
        if disc_for_guidance.cmap_dim > 0:
            pooled_prompt = prompt_embeds_eff.float().mean(dim=1)

        # ----- Action tokens + modulation (action-aware WAN configs) -----
        # When the WAN model has action_tokens_per_frame > 0, its
        # forward refuses to run without action_tokens, AND the
        # transformer blocks use a per-frame ``_action_modulation``
        # whose frame count MUST match the input's frame count (else
        # "size of tensor a (X) must match (Y)" in the per-block FiLM
        # multiply). Build BOTH streams from per-chunk slices of
        # streaming_state[ride_actions_window].
        real_action_tokens = None
        fake_action_tokens = None
        real_action_modulation = None
        fake_action_modulation = None
        a_per_f = 0
        # Look up the WAN model under real_score to read a_per_f.
        _rs = self.model.real_score
        for _cand in [_rs] + list(_rs.modules()):
            if hasattr(_cand, "action_tokens_per_frame"):
                a_per_f = int(getattr(_cand, "action_tokens_per_frame", 0))
                if a_per_f > 0:
                    break
        ride_actions = (
            s_state.get("ride_actions_window")
            if isinstance(s_state, dict) else None
        )
        atp = getattr(self.model, "action_token_projection", None)
        ap = getattr(self.model, "action_projection", None)
        if a_per_f > 0 and ride_actions is not None and atp is not None:
            cf_state = int(s_state.get("cf", 0))
            # Build per-pair action slices for real and fake sides.
            real_acts = []
            fake_acts = []
            for k, (i_real, j_fake) in enumerate(pairs):
                # ride_actions: [B, cf+rollout, A]. Chunk i covers frames
                # [cf + i*npb, cf + (i+1)*npb) within the rollout. For
                # gt_vs_fake, real and fake use the same chunk index
                # (i_real == j_fake). For adjacent_chunks they differ
                # by one (j_fake = i_real + 1).
                # Wide-real: the real chunk k was sliced from absolute
                # window position real_positions[k] (NOT cf_state-relative);
                # its action is co-located at the same absolute frame in
                # ride_actions_window (latents/actions share the ride
                # offset s), so use real_positions[k]*npb. Only fires for
                # all-pairs gt_vs_fake (chunks_per_pair==1).
                if _wide_real:
                    lo_r = real_positions[k] * npb
                else:
                    lo_r = cf_state + i_real * npb
                lo_f = cf_state + j_fake * npb
                if chunks_per_pair == 2:
                    # gt_transition: both sides cover the (i, i+1)
                    # chunk-pair. Concat the two consecutive action
                    # slices so the WAN teacher's per-frame action
                    # modulation has a slice for every one of the
                    # 2*npb input frames. Real and fake actions are
                    # identical (same time positions, same actions —
                    # only the latents differ).
                    real_acts.append(torch.cat([
                        ride_actions[:, lo_r:lo_r + npb],
                        ride_actions[:, lo_f:lo_f + npb],
                    ], dim=1))
                    fake_acts.append(torch.cat([
                        ride_actions[:, lo_r:lo_r + npb],
                        ride_actions[:, lo_f:lo_f + npb],
                    ], dim=1))
                else:
                    real_acts.append(ride_actions[:, lo_r:lo_r + npb])
                    fake_acts.append(ride_actions[:, lo_f:lo_f + npb])
            real_acts_t = torch.cat(real_acts, dim=0).to(
                device=device, dtype=prompt_embeds_eff.dtype,
            )
            fake_acts_t = torch.cat(fake_acts, dim=0).to(
                device=device, dtype=prompt_embeds_eff.dtype,
            )
            # No_grad: action projections are conditioning-only; we
            # never backward through them on the disc path, so the
            # autograd graph they'd build is dead weight (~1-2 GB).
            with torch.no_grad():
                real_action_tokens = atp(real_acts_t).detach()
                fake_action_tokens = atp(fake_acts_t).detach()
                if ap is not None:
                    real_action_modulation = ap(
                        real_acts_t, num_frames=real_acts_t.shape[1],
                    ).detach()
                    fake_action_modulation = ap(
                        fake_acts_t, num_frames=fake_acts_t.shape[1],
                    ).detach()

        # v28G_2: action-blind variant of gt_transition. Zeroes out the
        # action tokens and modulation BEFORE they hit the disc forward,
        # while preserving tensor shapes so the WAN teacher (which
        # requires action_tokens when ``a_per_f > 0``) still runs. The
        # disc then learns "GT chunk-pair transitions look like this
        # distribution" *independent of which action drove the
        # transition* — useful when the action signal is noisy or when
        # you want to ablate the action-conditioning contribution.
        # Only applies to gt_transition; gt_vs_fake / adjacent_chunks
        # keep their full action conditioning.
        if (
            pair_mode == "gt_transition"
            and bool(getattr(
                self.model, "ladd_gt_transition_action_blind", False,
            ))
        ):
            if real_action_tokens is not None:
                real_action_tokens = torch.zeros_like(real_action_tokens)
            if fake_action_tokens is not None:
                fake_action_tokens = torch.zeros_like(fake_action_tokens)
            if real_action_modulation is not None:
                real_action_modulation = torch.zeros_like(
                    real_action_modulation
                )
            if fake_action_modulation is not None:
                fake_action_modulation = torch.zeros_like(
                    fake_action_modulation
                )

        self._mem_step_snapshot(f"ladd_{pair_mode}_a_cond_built")

        # ----- D-update -----
        # ``phase`` allows the caller to split the D-update from the
        # gen-side forward across multiple pair modes (two-phase
        # orchestration). With "d_only" we run only the D-step + step
        # and skip the gen-side; with "g_only" we run only the
        # gen-side forward; with "both" (default) we do D then G in a
        # single call (correct for single-mode runs).
        disc_skipped = (
            current_step < int(getattr(self, "gan_disc_start_step", 0))
        )
        skip_d = (phase == "g_only")
        skip_g = (phase == "d_only")
        n_disc_updates = (
            0 if (disc_skipped or skip_d) else int(self.gan_updates_per_step)
        )
        last_d_loss = 0.0
        last_d_real = 0.0
        last_d_fake = 0.0
        last_r1 = 0.0
        # Stat-head diagnostic accumulators — non-zero only when
        # ``stat_head_enabled``. Separate D-side stat loss + per-sample
        # real / fake logit means so the user can see the stat side's
        # behaviour independently of the per-token visual side. 0 when
        # stat head is off OR ``ladd_stat_head_loss_weight`` is 0.
        last_d_loss_stat = 0.0
        last_d_real_stat = 0.0
        last_d_fake_stat = 0.0
        # Diagnostic R1 stats (independent of γ + lazy schedule):
        #   * ``last_r1_grad_sq``: raw ‖∇_x Σ_i D_i‖² estimate from the
        #     finite-difference or autograd path. This is the quantity
        #     γ multiplies — log it separately so the gradient norm
        #     itself is visible regardless of how small γ is. NaN on
        #     iters where R1 didn't fire (lazy schedule skip or disc
        #     warmup) so wandb plots gaps rather than misleading 0s.
        #   * ``last_r1_fired``: 1.0 if R1 was actually computed this
        #     iter, 0.0 otherwise. Lets the user filter the trace to
        #     only the iters where R1 has a real value.
        last_r1_grad_sq = float("nan")
        last_r1_fired = 0.0
        # R2 (fake-side penalty) counterparts — same NaN-when-not-fired
        # convention as R1 so wandb plots gaps rather than 0s.
        last_r2 = 0.0
        last_r2_grad_sq = float("nan")
        last_r2_fired = 0.0

        # ===== Content-MATCHED wide GT (gt_transition; all_pairs=false) =====
        # For each fake transition, score it ONLY against its K GT
        # transitions from the FULL loaded ride that are closest by MAE on
        # the MEAN-EQUALIZED rep (texture, not brightness — so a drifted
        # fake can't cherry-pick a drifted GT). Block-diagonal RpGAN (each
        # fake vs its own K matched GT), NOT all_pairs — gives multiple
        # RELEVANT real views per fake without the content-confounded noise
        # of all_pairs. Position is never used (the rollout drifts in time).
        # Efficient: each UNIQUE matched GT is forwarded ONCE; the per-fake-K
        # grouping is done at the LOGIT level (index_select + repeat), so
        # neither fakes nor reals are re-forwarded per K. Self-contained
        # (own D-loop + gen-side + R1). Assumes per-rank B==1 (single ride),
        # like the mismatched branch's prompt tiling. Mutually exclusive
        # with all_pairs / mismatched n_real.
        _match = (
            pair_mode == "gt_transition"
            and chunks_per_pair == 2
            and bool(getattr(self.model, "ladd_gt_transition_match", False))
        )
        # Mutually exclusive with the all_pairs matrix and the mismatched
        # n_real sampler — the match branch is physically first and returns,
        # so a co-set config would SILENTLY ignore those. Fail loud instead.
        if _match and (_all_pairs_mode or _mismatched):
            raise RuntimeError(
                "ladd_gt_transition_match is mutually exclusive with "
                "ladd_gt_transition_all_pairs / ladd_gt_transition_n_real>0; "
                "set all_pairs=false and n_real=0 for the matched mode."
            )
        _match_lat = None
        if _match:
            _ss_m = getattr(self.model, "streaming_state", None)
            _match_lat = (
                _ss_m.get("gt_match_latents")
                if isinstance(_ss_m, dict) else None
            )
        if _match and _match_lat is not None and int(_match_lat.shape[1]) >= 2 * npb:
            K = max(1, int(getattr(self.model, "ladd_gt_transition_match_k", 3)))
            _need_pp = pooled_prompt is not None
            _r1_gamma = float(getattr(self.model, "ladd_r1_gamma", 1.0))
            _r1_every_n = max(1, int(
                getattr(self.model, "ladd_r1_every_n_steps", 1)))
            _r1_sigma = float(getattr(self.model, "ladd_r1_sigma", 0.01))
            _r2_gamma, _r2_every_n, _r2_offset, _r2_sigma = self._ladd_r2_knobs(
                _r1_every_n, _r1_sigma)
            _K_stat = int(getattr(self.r3gan_disc, "stat_logit_count", 0))
            _W_stat = float(getattr(
                self.model, "ladd_stat_head_loss_weight", 1.0))
            _action_blind = bool(getattr(
                self.model, "ladd_gt_transition_action_blind", False))
            B = int(prompt_embeds.shape[0])
            pool_lat = _match_lat.detach()                       # [B, RL, C,H,W]
            pool_act_full = (
                _ss_m.get("gt_match_actions")
                if isinstance(_ss_m, dict) else None
            )
            pool_chunks = int(pool_lat.shape[1] // npb)
            n_cand = max(1, pool_chunks - 1)                     # (u, u+1)
            Kk = min(K, n_cand)
            # Hard cap on distinct reals forwarded per D-update (memory): the
            # combined disc forward is 2*n_uniq + n_fake rows, so this bounds
            # it independent of the (possibly whole-ride) match pool. 0 = no
            # cap. Default keeps it near 5b's n_real footprint.
            _cap = int(getattr(self.model, "ladd_gt_transition_match_max_real", 12))
            _fdt = fake_chunks_det.dtype

            def _pool_pair_lat(b, u):
                return torch.cat([
                    pool_lat[b:b + 1, u * npb:(u + 1) * npb],
                    pool_lat[b:b + 1, (u + 1) * npb:(u + 2) * npb],
                ], dim=1)                                        # [1, 2*npb, ...]

            # Candidate transitions per batch elem, mean-equalized (the disc
            # input AND the match key — match on the same rep the disc sees).
            cand_disc = []          # over b: [n_cand, 2*npb, C,H,W] (equalized)
            for b in range(B):
                cl = torch.cat(
                    [_pool_pair_lat(b, u) for u in range(n_cand)], dim=0)
                if _mean_eq:
                    cl = _mean_equalize_pair(cl)
                cand_disc.append(cl.to(_fdt))

            # Fake queries: fake_chunks_det is [n_pairs*B, 2*npb, ...]
            # pair-major batch-minor, already mean-equalized. View [n_pairs,B].
            fq = fake_chunks_det.detach().view(
                n_pairs, B, *fake_chunks_det.shape[1:])

            # Per-(pair, batch) TOP-M nearest candidate indices by MAE
            # (no_grad; the fake is fixed for the step so this is computed
            # ONCE). ``M`` is the RELEVANT pool each D-update then samples K
            # from — this is how we keep 5b's "resample fresh GT every
            # D-update" decorrelation (the disc never sees the same K reals
            # twice across the D-loop) WITHOUT widening to the irrelevant
            # all-ride noise: only each fake's M nearest are ever eligible.
            # M defaults to 2*K; M==K reduces to deterministic top-K (no
            # resample). Override via ``ladd_gt_transition_match_pool``.
            _M = int(getattr(self.model, "ladd_gt_transition_match_pool", 0))
            if _M <= 0:
                _M = 2 * Kk
            _M = max(Kk, min(_M, n_cand))
            top_idx = [[None] * B for _ in range(n_pairs)]
            with torch.no_grad():
                for b in range(B):
                    ce = cand_disc[b].float().flatten(1)         # [n_cand, D]
                    fb = fq[:, b].float().flatten(1)             # [n_pairs, D]
                    # L1 mean distance via cdist (p=1 = sum|.|) / D — avoids
                    # materializing the [n_pairs, n_cand, D] difference tensor
                    # (the dominant transient for long rides / large K).
                    mae = torch.cdist(fb, ce, p=1) / float(fb.shape[1])
                    idx = torch.topk(
                        mae, _M, dim=1, largest=False).indices.tolist()
                    for p in range(n_pairs):
                        top_idx[p][b] = idx[p]

            def _match_select(salt, carn=False):
                # Sample Kk of each fake's M nearest GT — FRESH per call
                # (seeded step+salt) — then DEDUP: forward each unique GT
                # ONCE and recover the per-fake-K block-diagonal at the logit
                # level via ``group_flat`` (gather). Dedup is essential for
                # memory: the rolled fakes are temporally adjacent so their
                # nearest-GT sets overlap heavily, keeping the disc-forward
                # row count near 5b's n_real instead of n_pairs*Kk. Count
                # varies per draw/rank, which DDP tolerates (fwd/bwd-pass
                # count is global-step driven, grads are per-param) — exactly
                # as 5b's rank-variable n_pairs already does.
                g = torch.Generator(device="cpu").manual_seed(
                    int(current_step) * 977 + int(salt) * 131 + 17)
                sel = [[None] * B for _ in range(n_pairs)]
                for p in range(n_pairs):
                    for b in range(B):
                        pool = top_idx[p][b]
                        if _M > Kk:
                            perm = torch.randperm(_M, generator=g).tolist()
                            sel[p][b] = [pool[i] for i in perm[:Kk]]
                        else:
                            sel[p][b] = list(pool[:Kk])
                # Build the UNIQUE real set + group_map, with a hard CAP on
                # the number of distinct reals forwarded (``_cap``) so the
                # wide match pool (n_cand can be ~the whole ride) is decoupled
                # from the disc-forward / R1 memory (combined forward is
                # 2*n_uniq + n_fake rows). When the cap is hit, a fake's new
                # pick is remapped to one of ITS OWN already-included nearer
                # picks (still its own relevant GT, just shared) — the
                # block-diagonal stays valid, each fake keeps Kk reals.
                key_to_row = {}
                rows_bu = []
                gm = torch.empty(n_pairs * B, Kk, dtype=torch.long)
                for p in range(n_pairs):
                    for b in range(B):
                        fr = p * B + b
                        for k in range(Kk):
                            u = sel[p][b][k]
                            if (b, u) in key_to_row:
                                gm[fr, k] = key_to_row[(b, u)]
                            elif _cap > 0 and len(rows_bu) >= _cap:
                                alt = next((uu for uu in sel[p][b]
                                            if (b, uu) in key_to_row), None)
                                if alt is None:
                                    alt = next((uu for (bb, uu) in rows_bu
                                                if bb == b), None)
                                gm[fr, k] = (key_to_row[(b, alt)] if alt is not None
                                             else 0)
                            else:
                                key_to_row[(b, u)] = len(rows_bu)
                                rows_bu.append((b, u))
                                gm[fr, k] = key_to_row[(b, u)]
                real_rows = [cand_disc[b][u] for (b, u) in rows_bu]
                acts_list = []
                if a_per_f > 0 and pool_act_full is not None and atp is not None:
                    for (b, u) in rows_bu:
                        acts_list.append(torch.cat([
                            pool_act_full[b:b + 1, u * npb:(u + 1) * npb],
                            pool_act_full[b:b + 1, (u + 1) * npb:(u + 2) * npb],
                        ], dim=1))
                ru = torch.stack(real_rows, dim=0).to(
                    device=device, dtype=_fdt)                   # [n_uniq, ...]
                # e11: CARN the matched real FORMERS (Req 1 + Req 2). Only on
                # the D-update side (carn=True); the gen-side keeps clean
                # formers (Req 2), so the student is pulled toward clean GT.
                if carn and bool(getattr(
                        self.model, "ladd_gt_transition_carn_match_pool", False)):
                    n_uniq = ru.shape[0]
                    # Per-row cap = (min served-fake drift) - 1. A row's
                    # served fakes come from gm; fake pair p's former drift
                    # = pairs[p][0] (rollout chunk index). The cap keeps the
                    # real former noised LESS than every student former it is
                    # compared against (Req 1).
                    BIG = 1 << 30
                    row_mindrift = [BIG] * n_uniq
                    for _p in range(n_pairs):
                        _fd = int(pairs[_p][0])
                        for _b in range(B):
                            _fr = _p * B + _b
                            for _k in range(Kk):
                                _r = int(gm[_fr, _k].item())
                                if _fd < row_mindrift[_r]:
                                    row_mindrift[_r] = _fd
                    # Fixed level (>0): degrade every eligible real former by
                    # exactly this many CARN steps (still drift-capped by
                    # Req 1). 0 (default) => random level in [1, cap].
                    _fixed_lvl = int(getattr(
                        self.model,
                        "ladd_gt_transition_carn_match_pool_level", 0))
                    _levels = []
                    for _r in range(n_uniq):
                        _lvlcap = (row_mindrift[_r] - 1
                                   if row_mindrift[_r] < BIG else 0)
                        if _lvlcap < 1:
                            _levels.append(0)
                        elif _fixed_lvl > 0:
                            _levels.append(min(_fixed_lvl, _lvlcap))
                        else:
                            _levels.append(
                                int(torch.randint(
                                    1, _lvlcap + 1, (1,)).item()))
                    _maxlvl = max(_levels) if _levels else 0
                    if (getattr(self, "is_main_process", True)
                            and getattr(self, "_carn_match_pool_dbg", 0) < 3):
                        self._carn_match_pool_dbg = getattr(
                            self, "_carn_match_pool_dbg", 0) + 1
                        import sys as _sys
                        print(
                            "[CARN-MATCH-POOL] ACTIVE: n_uniq=%d "
                            "row_mindrift=%s levels=%s maxlvl=%d "
                            "(real formers get recursive CARN < served-fake "
                            "drift; gen-side stays clean)" % (
                                n_uniq, row_mindrift[:8], _levels[:8],
                                _maxlvl,
                            ),
                            file=_sys.stderr, flush=True,
                        )
                    if _maxlvl > 0:
                        _x0 = ru[:, :npb].clone()
                        _cur = _x0.clone()
                        with torch.no_grad():
                            for _kk in range(_maxlvl):
                                _idx = [r for r in range(n_uniq)
                                        if _levels[r] > _kk]
                                if not _idx:
                                    continue
                                _ii = torch.tensor(_idx, device=_cur.device)
                                _sub = _cur.index_select(0, _ii)
                                _cs = torch.zeros(
                                    (_sub.shape[0],), dtype=torch.long,
                                    device=_sub.device)
                                _sub = self.model.forward_noiser(
                                    _sub, _cs, residual=True)
                                _cur = _cur.index_copy(0, _ii, _sub)
                            # Moment-preserving (texture-only) restore.
                            _e = 1e-6
                            _mci = _x0.mean(dim=[1, 3, 4], keepdim=True)
                            _mco = _cur.mean(dim=[1, 3, 4], keepdim=True)
                            _cur = _cur - _mco + _mci
                            _ai = _x0.abs().mean(dim=[1, 2, 3, 4], keepdim=True)
                            _ao = _cur.abs().mean(dim=[1, 2, 3, 4], keepdim=True)
                            _cur = _cur * (_ai / (_ao + _e))
                        ru = torch.cat([_cur.detach(), ru[:, npb:]], dim=1)
                # FN application to the matched real pairs (both naive: the
                # SAME transformed pair feeds the D-update and the gen-side
                # real view — no Req-1 drift cap, no Req-2 gen-side shield):
                #   * apply_gt_both (h1): ONE FN call per half (former AND
                #     latter). With a REVERSE-trained FN this de-CARNs the
                #     real target (suppresses the rollout's explode/
                #     harmonise modes from the real distribution).
                #   * apply_gt_former (h3): ONE FN call on the FORMER only.
                #     With a FORWARD-trained FN the real anchor becomes
                #     [1-step-drifted GT former -> clean GT latter], the
                #     self-correcting transition.
                # Moment-preserving (texture-only) restore per transformed
                # half so only texture STRUCTURE changes (no brightness/
                # colour shift).
                _gt_both = bool(getattr(
                    self.model, "forward_noiser_apply_gt_both", False))
                _gt_former = bool(getattr(
                    self.model, "forward_noiser_apply_gt_former", False))
                if ((_gt_both or _gt_former)
                        and getattr(self.model, "forward_noiser", None)
                        is not None):
                    # Conditioning level for the application. A step-
                    # UNCONDITIONED FN trained only at carn_step=0, so apply
                    # at 0. A step-CONDITIONED FN (give it the numbers)
                    # trained at the input's drift level L; applying at
                    # forward_noiser_apply_gt_level (default 1) adds/removes
                    # one drift-level's worth depending on the FN direction.
                    _step_uncond = bool(getattr(
                        self.model,
                        "forward_noiser_step_unconditioned", False))
                    _apply_lvl = 0 if _step_uncond else int(getattr(
                        self.model, "forward_noiser_apply_gt_level", 1))
                    # h4: per-row random level in [apply_lvl, lvl_max]
                    # (step-conditioned only). 0/<=apply_lvl = fixed level.
                    _lvl_max = 0 if _step_uncond else int(getattr(
                        self.model, "forward_noiser_apply_gt_level_max", 0))
                    # h5: drift-capped per-row level — rand[1, mindrift-1]
                    # where mindrift = min drift step of the student
                    # formers this real row is served against (via gm).
                    # Rows served only by drift<=1 fakes keep a CLEAN
                    # former (level 0 -> FN output discarded below).
                    _drift_cap = (not _step_uncond) and bool(getattr(
                        self.model,
                        "forward_noiser_apply_gt_drift_cap", False))
                    _fn_offsets = (0, npb) if _gt_both else (0,)
                    with torch.no_grad():
                        if _drift_cap:
                            _BIGc = 1 << 30
                            _mind = [_BIGc] * int(ru.shape[0])
                            for _p in range(n_pairs):
                                _fd = int(pairs[_p][0])
                                for _b in range(B):
                                    _fr = _p * B + _b
                                    for _k in range(Kk):
                                        _r = int(gm[_fr, _k].item())
                                        if _fd < _mind[_r]:
                                            _mind[_r] = _fd
                            _lvls = []
                            for _r in range(int(ru.shape[0])):
                                _cp = (_mind[_r] - 1
                                       if _mind[_r] < _BIGc else 0)
                                _lvls.append(
                                    int(torch.randint(
                                        1, _cp + 1, (1,)).item())
                                    if _cp >= 1 else 0)
                            _cs0 = torch.tensor(
                                _lvls, dtype=torch.long, device=ru.device)
                        elif _lvl_max > _apply_lvl:
                            _cs0 = torch.randint(
                                _apply_lvl, _lvl_max + 1, (ru.shape[0],),
                                dtype=torch.long, device=ru.device)
                        else:
                            _cs0 = torch.full(
                                (ru.shape[0],), _apply_lvl,
                                dtype=torch.long, device=ru.device)
                        _halves = []
                        for _h0 in (0, npb):
                            _x0 = ru[:, _h0:_h0 + npb]
                            if _h0 not in _fn_offsets:
                                _halves.append(_x0)
                                continue
                            _cur = self.model.forward_noiser(
                                _x0, _cs0, residual=True)
                            _e = 1e-6
                            _mci = _x0.mean(dim=[1, 3, 4], keepdim=True)
                            _mco = _cur.mean(dim=[1, 3, 4], keepdim=True)
                            _cur = _cur - _mco + _mci
                            _ai = _x0.abs().mean(
                                dim=[1, 2, 3, 4], keepdim=True)
                            _ao = _cur.abs().mean(
                                dim=[1, 2, 3, 4], keepdim=True)
                            _cur = _cur * (_ai / (_ao + _e))
                            # Level-0 rows (drift-cap mode: served fakes at
                            # drift <= 1) keep the ORIGINAL clean half — the
                            # FN was never trained at level 0, so its output
                            # there is undefined; discard it.
                            _keep = (_cs0 > 0).view(-1, 1, 1, 1, 1)
                            _cur = torch.where(_keep, _cur, _x0)
                            _halves.append(_cur)
                        ru = torch.cat(_halves, dim=1).detach()
                    if (getattr(self, "is_main_process", True)
                            and getattr(self, "_fn_gt_both_dbg", 0) < 2):
                        self._fn_gt_both_dbg = getattr(
                            self, "_fn_gt_both_dbg", 0) + 1
                        import sys as _sys
                        if _drift_cap:
                            _lvl_desc = (
                                "drift-capped rand[1,mindrift-1] "
                                "(levels=%s)" % (
                                    _cs0.tolist()[:8],))
                        elif _lvl_max > _apply_lvl:
                            _lvl_desc = "rand[%d,%d]" % (
                                _apply_lvl, _lvl_max)
                        else:
                            _lvl_desc = str(_apply_lvl)
                        print(
                            "[FN-GT-%s] ACTIVE: FN applied to %s of %d "
                            "matched real pairs at carn_step=%s "
                            "(step_uncond=%s, naive both-sides)" % (
                                "BOTH" if _gt_both else "FORMER",
                                "BOTH GT latents" if _gt_both
                                else "the FORMER GT latent",
                                ru.shape[0], _lvl_desc, _step_uncond),
                            file=_sys.stderr, flush=True,
                        )
                gflat = gm.reshape(-1).to(device)                # [n_pairs*B*Kk]
                rat = ram = None
                if acts_list:
                    _ra = torch.cat(acts_list, dim=0).to(
                        device=device, dtype=prompt_embeds_eff.dtype)
                    with torch.no_grad():
                        rat = atp(_ra).detach()
                        if ap is not None:
                            ram = ap(_ra, num_frames=_ra.shape[1]).detach()
                if _action_blind:
                    if rat is not None:
                        rat = torch.zeros_like(rat)
                    if ram is not None:
                        ram = torch.zeros_like(ram)
                return ru, rat, ram, gflat

            def _m_noise(x):
                if disc_t_int <= 0:
                    return x
                scheduler = getattr(self.model, "scheduler", None)
                if scheduler is None or not hasattr(scheduler, "add_noise"):
                    raise RuntimeError(
                        "_m_noise: disc_t_int>0 but no scheduler.add_noise")
                eps = torch.randn_like(x)
                xf, ef = x.flatten(0, 1), eps.flatten(0, 1)
                tpf = torch.full(
                    (xf.shape[0],), disc_t_int, dtype=torch.long, device=x.device)
                return scheduler.add_noise(xf, ef, tpf).unflatten(0, x.shape[:2])

            def _m_fwd(disc, segs):
                xs = [s[0] for s in segs]
                counts = [int(s[0].shape[0]) for s in segs]
                x = torch.cat(xs, dim=0)
                n_rows = x.shape[0]
                t = torch.full(
                    (n_rows, t_frames), disc_t_int, dtype=torch.long, device=device)
                reps = max(1, n_rows // int(prompt_embeds.shape[0]))
                pe = prompt_embeds.repeat(reps, 1, 1)
                pp = (
                    prompt_embeds.float().mean(dim=1).repeat(reps, 1)
                    if _need_pp else None)
                ce = None
                if any(s[1] is not None for s in segs):
                    ce = {"_action_tokens": torch.cat(
                        [s[1] for s in segs], dim=0)}
                    if all(s[2] is not None for s in segs):
                        ce["_action_modulation"] = torch.cat(
                            [s[2] for s in segs], dim=0)
                logits = disc(
                    x_noisy=x, timestep=t, prompt_embeds=pe,
                    pooled_prompt=pp, conditional_extra=ce)
                out, o = [], 0
                for c in counts:
                    out.append(logits[o:o + c])
                    o += c
                return out

            def _m_rp(d_real_uniq, d_fake, detach_real, group_flat):
                # Block-diagonal RpGAN. ``group_flat`` gathers each unique
                # real's logit into the (pair, batch, k)-major layout; the
                # fakes are repeat_interleave'd to match (fake row (p,b)
                # repeated Kk times lines up with its Kk matched reals). Each
                # fake is scored ONLY against its own K matched GT (NOT
                # all_pairs): standard MATCHED RpGAN on the aligned rows.
                rr = d_real_uniq.detach() if detach_real else d_real_uniq
                rg = rr.index_select(0, group_flat)              # [Nf*Kk, tok]
                fg = d_fake.repeat_interleave(Kk, dim=0)         # [Nf*Kk, tok]
                mfn = rpgan_g_loss if detach_real else rpgan_d_loss
                if _K_stat > 0 and _W_stat > 0.0:
                    rv, rs = rg[:, :-_K_stat], rg[:, -_K_stat:]
                    fv, fs = fg[:, :-_K_stat], fg[:, -_K_stat:]
                    lv = mfn(rv, fv)
                    ls = mfn(rs, fs)
                    return lv + _W_stat * ls, float(ls.detach().item())
                return mfn(rg, fg), 0.0

            disc_skipped = (
                current_step < int(getattr(self, "gan_disc_start_step", 0)))
            last_d_loss = last_d_real = last_d_fake = 0.0
            last_r1 = last_r1_grad_sq = last_r1_fired = 0.0
            last_r2 = last_r2_grad_sq = last_r2_fired = 0.0
            last_d_loss_stat = last_d_real_stat = last_d_fake_stat = 0.0
            last_n_real = 0.0
            _fseg_act = (fake_action_tokens, fake_action_modulation)

            # ---- D-updates (resample each fake's K matched GT FRESH per
            # update — salt=_it — to preserve 5b's GT decorrelation) ----
            if n_disc_updates > 0 and self.r3gan_optimizer is not None:
                for _it in range(n_disc_updates):
                    self.r3gan_optimizer.zero_grad(set_to_none=True)
                    # D-update: POST-CARN matched real formers (Req 2).
                    real_m, real_m_rat, real_m_ram, group_flat = _match_select(
                        _it, carn=True)
                    last_n_real = float(real_m.shape[0])
                    _rn = _m_noise(real_m)
                    if diff_aug_policy:
                        _rn, _ = latent_diff_augment(
                            _rn, _rn, policy=diff_aug_policy,
                            seed=int(current_step) * 31 + seed_offset + _it * 101)
                    _rn = _rn.detach()
                    _fk = _m_noise(fake_chunks_det).detach()
                    _fseg = (_fk, _fseg_act[0], _fseg_act[1])
                    # R1 (real-side) + R2 (fake-side) gradient penalties,
                    # each on its own lazy cadence. The perturbed
                    # segments are appended to the SINGLE batched disc
                    # forward only on the steps they fire (offset so the
                    # two never stack — OOM guard).
                    _do_r1 = (current_step % _r1_every_n == 0)
                    _do_r2 = self._ladd_r2_fires(
                        _r2_gamma, _r2_every_n, _r2_offset, current_step)
                    _segs = [(_rn, real_m_rat, real_m_ram), _fseg]
                    if _do_r1:
                        _eps_r = _r1_sigma * torch.randn_like(_rn)
                        _segs.append((_rn + _eps_r, real_m_rat, real_m_ram))
                    if _do_r2:
                        _eps_f = _r2_sigma * torch.randn_like(_fk)
                        _segs.append(
                            (_fk + _eps_f, _fseg_act[0], _fseg_act[1]))
                    _outs = _m_fwd(disc_for_update, _segs)
                    d_r, d_f = _outs[0], _outs[1]
                    _nxt = 2
                    if _do_r1:
                        d_r_pert = _outs[_nxt]
                        _nxt += 1
                        _gsq = (
                            ((d_r_pert.sum(dim=1) - d_r.sum(dim=1)) / _r1_sigma)
                            .pow(2).mean())
                        r1 = 0.5 * _r1_gamma * _gsq
                        last_r1_grad_sq = float(_gsq.detach().item())
                        last_r1_fired = 1.0
                    else:
                        r1 = d_r.sum() * 0.0
                    if _do_r2:
                        d_f_pert = _outs[_nxt]
                        _nxt += 1
                        _gsq2 = (
                            ((d_f_pert.sum(dim=1) - d_f.sum(dim=1)) / _r2_sigma)
                            .pow(2).mean())
                        r2 = 0.5 * _r2_gamma * _gsq2
                        last_r2_grad_sq = float(_gsq2.detach().item())
                        last_r2_fired = 1.0
                    else:
                        r2 = d_f.sum() * 0.0
                    d_rp, _dstat = _m_rp(d_r, d_f, False, group_flat)
                    (d_rp + r1 + r2).backward()
                    if self.gan_max_grad_norm and self.gan_max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.r3gan_disc.parameters()
                             if p.grad is not None],
                            self.gan_max_grad_norm)
                    self.r3gan_optimizer.step()
                    last_d_loss = float(d_rp.detach().item())
                    last_d_real = float(d_r.detach().mean().item())
                    last_d_fake = float(d_f.detach().mean().item())
                    last_r1 = float(r1.detach().item())
                    last_r2 = float(r2.detach().item())
                    last_d_loss_stat = _dstat

            # ---- Gen-side ----
            critic_warmup_done = (
                current_step >= int(getattr(self, "gan_critic_warmup_steps", 0)))
            if (
                self.gan_warmup_steps > 0
                and current_step < (
                    self.gan_critic_warmup_steps + self.gan_warmup_steps)
                and current_step >= self.gan_critic_warmup_steps
            ):
                _ramp_in = current_step - self.gan_critic_warmup_steps
                gen_gan_weight = self._gan_warmup_shape_apply(
                    _ramp_in / max(1, self.gan_warmup_steps)
                ) * self.gan_loss_weight
            elif current_step >= (
                self.gan_critic_warmup_steps + self.gan_warmup_steps
            ):
                gen_gan_weight = self.gan_loss_weight
            else:
                gen_gan_weight = 0.0
            gen_gan_weight = gen_gan_weight * float(
                getattr(self.model, "ladd_disc_loss_weight", 1.0))

            generator_gan_loss = zero
            gen_gan_main_value = 0.0
            gen_gan_stat_value = 0.0
            if critic_warmup_done and gen_gan_weight > 0 and not skip_g:
                disc_for_guidance.requires_grad_(False)
                _disc_was_training = disc_for_guidance.training
                disc_for_guidance.eval()
                try:
                    real_m, real_m_rat, real_m_ram, group_flat = _match_select(7919)
                    _rn = _m_noise(real_m).detach()
                    _fg = _m_noise(fake_chunks_grad_tensor)
                    if diff_aug_policy:
                        _rn, _ = latent_diff_augment(
                            _rn, _rn, policy=diff_aug_policy,
                            seed=int(current_step) * 31 + seed_offset + 99)
                        _, _fg = latent_diff_augment(
                            _fg, _fg, policy=diff_aug_policy,
                            seed=int(current_step) * 31 + seed_offset + 199)
                    d_rg, d_fg = _m_fwd(disc_for_guidance, [
                        (_rn, real_m_rat, real_m_ram),
                        (_fg, _fseg_act[0], _fseg_act[1]),
                    ])
                    g_rp, gen_gan_stat_value = _m_rp(d_rg, d_fg, True, group_flat)
                    generator_gan_loss = (
                        gen_gan_weight * g_rp.to(pred_image_dtype))
                    gen_gan_main_value = float(g_rp.detach().item())
                finally:
                    disc_for_guidance.requires_grad_(True)
                    if _disc_was_training:
                        disc_for_guidance.train()

            logs = {
                "train/r3gan_disc_skipped": 1.0 if disc_skipped else 0.0,
                "train/r3gan_d_loss": last_d_loss,
                "train/r3gan_d_real": last_d_real,
                "train/r3gan_d_fake_detached": last_d_fake,
                "train/r3gan_r1": last_r1,
                "train/r3gan_r1_grad_sq": last_r1_grad_sq,
                "train/r3gan_r1_fired": last_r1_fired,
                "train/r3gan_r1_gamma": float(_r1_gamma),
                "train/r3gan_r2": last_r2,
                "train/r3gan_r2_grad_sq": last_r2_grad_sq,
                "train/r3gan_r2_fired": last_r2_fired,
                "train/r3gan_r2_gamma": float(_r2_gamma),
                "train/r3gan_g_loss_raw": gen_gan_main_value,
                "train/r3gan_g_loss_weighted": gen_gan_weight * gen_gan_main_value,
                "train/r3gan_g_weight": float(gen_gan_weight),
                "train/critic_warmup_done": 1.0 if critic_warmup_done else 0.0,
                "train/ladd_n_pairs": float(n_pairs),
                "train/ladd_match_k": float(Kk),
                "train/ladd_match_pool_m": float(_M),
                "train/ladd_match_n_real": last_n_real,
                "train/r3gan_d_loss_stat": last_d_loss_stat,
                "train/r3gan_g_loss_raw_stat": gen_gan_stat_value,
                "train/r3gan_stat_loss_weight": float(
                    getattr(self.model, "ladd_stat_head_loss_weight", 1.0)),
            }
            return generator_gan_loss, logs

        # ===== Mismatched all-pairs (N_real != N_fake) + resample-per-update
        # =====
        # Feed N_real GT chunks (gt_vs_fake) / transition-pairs
        # (gt_transition) — RESAMPLED fresh from the full ride window every
        # D-update — against the N_fake student chunks, for an
        # N_real x N_fake all-pairs comparison with maximal GT
        # decorrelation. Real / fake / perturbed are forwarded SEPARATELY
        # (no equal-count combined split), so the R1 finite-diff is exact
        # for unequal counts. Real latent + its action are co-located (same
        # absolute window position). Self-contained: runs its own D-loop +
        # gen-side and returns the standard (generator_gan_loss, logs).
        # Matched configs never enter here (gated on n_real knob > 0).
        _mm_win = None
        if _mismatched:
            _ss_mm = getattr(self.model, "streaming_state", None)
            _mm_win = (
                _ss_mm.get("ride_latents_window")
                if isinstance(_ss_mm, dict) else None
            )
        if _mismatched and _mm_win is not None and int(_mm_win.shape[1]) >= npb:
            _win = _mm_win.detach().to(torch.float32)
            _w_chunks = int(_win.shape[1] // npb)
            _n_units = _w_chunks if chunks_per_pair == 1 else max(1, _w_chunks - 1)
            _K_stat = int(getattr(self.r3gan_disc, "stat_logit_count", 0))
            _W_stat = float(getattr(
                self.model, "ladd_stat_head_loss_weight", 1.0))
            _r1_gamma = float(getattr(self.model, "ladd_r1_gamma", 1.0))
            _r1_every_n = max(1, int(
                getattr(self.model, "ladd_r1_every_n_steps", 1)))
            _r1_sigma = float(getattr(self.model, "ladd_r1_sigma", 0.01))
            _r2_gamma, _r2_every_n, _r2_offset, _r2_sigma = self._ladd_r2_knobs(
                _r1_every_n, _r1_sigma)
            _need_pp = pooled_prompt is not None

            def _resample_real(salt):
                # Sample _n_real fresh window positions (cycle if short);
                # slice co-located latent + action.
                g = torch.Generator(device="cpu").manual_seed(
                    int(current_step) * 977 + int(salt) * 131 + 17)
                perm = torch.randperm(_n_units, generator=g).tolist()
                pos = [perm[k % _n_units] for k in range(_n_real)]
                if chunks_per_pair == 1:
                    rc = torch.cat([_slice(_win, p) for p in pos], dim=0)
                else:
                    rc = torch.cat(
                        [_slice_pair(_win, p, p + 1) for p in pos], dim=0)
                    if _mean_eq:
                        rc = _mean_equalize_pair(rc)
                rat = None
                ram = None
                if a_per_f > 0 and ride_actions is not None and atp is not None:
                    acts = []
                    for p in pos:
                        if chunks_per_pair == 1:
                            lo = p * npb
                            acts.append(ride_actions[:, lo:lo + npb])
                        else:
                            lo0, lo1 = p * npb, (p + 1) * npb
                            acts.append(torch.cat([
                                ride_actions[:, lo0:lo0 + npb],
                                ride_actions[:, lo1:lo1 + npb],
                            ], dim=1))
                    acts_t = torch.cat(acts, dim=0).to(
                        device=device, dtype=prompt_embeds_eff.dtype)
                    with torch.no_grad():
                        rat = atp(acts_t).detach()
                        if ap is not None:
                            ram = ap(
                                acts_t, num_frames=acts_t.shape[1]).detach()
                return rc.detach(), rat, ram

            def _mm_noise(x):
                # Count-correct disc noise (the outer _add_disc_noise reuses
                # the matched-sized t_disc, so it can't take a differently-
                # sized batch).
                if disc_t_int <= 0:
                    return x
                scheduler = getattr(self.model, "scheduler", None)
                if scheduler is None or not hasattr(scheduler, "add_noise"):
                    raise RuntimeError(
                        "_mm_noise: disc_t_int > 0 but model has no scheduler "
                        "with add_noise. Set flash_dmd_enabled=False (disc_t=0) "
                        "or provide a scheduler.")
                eps = torch.randn_like(x)
                x_flat = x.flatten(0, 1)
                eps_flat = eps.flatten(0, 1)
                t_pf = torch.full(
                    (x_flat.shape[0],), disc_t_int,
                    dtype=torch.long, device=x.device)
                noisy = scheduler.add_noise(x_flat, eps_flat, t_pf)
                return noisy.unflatten(0, x.shape[:2])

            def _mm_fwd_segments(disc, segs):
                # ONE disc forward over concatenated segments — DDP-safe
                # (single forward -> single backward; 3 separate forwards
                # before a backward would desync DDP's grad reducer).
                # segs: list of (chunks, action_tokens, action_mod). Returns
                # the per-segment logit slices in order. Prompt/pooled are
                # tiled position-major to match the chunk concat ordering.
                xs = [s[0] for s in segs]
                counts = [int(s[0].shape[0]) for s in segs]
                x = torch.cat(xs, dim=0)
                n_rows = x.shape[0]
                t = torch.full(
                    (n_rows, t_frames), disc_t_int,
                    dtype=torch.long, device=device)
                reps = max(1, n_rows // int(prompt_embeds.shape[0]))
                pe = prompt_embeds.repeat(reps, 1, 1)
                pp = (
                    prompt_embeds.float().mean(dim=1).repeat(reps, 1)
                    if _need_pp else None)
                ce = None
                if any(s[1] is not None for s in segs):
                    ce = {"_action_tokens": torch.cat(
                        [s[1] for s in segs], dim=0)}
                    if all(s[2] is not None for s in segs):
                        ce["_action_modulation"] = torch.cat(
                            [s[2] for s in segs], dim=0)
                logits = disc(
                    x_noisy=x, timestep=t, prompt_embeds=pe,
                    pooled_prompt=pp, conditional_extra=ce)
                out = []
                o = 0
                for c in counts:
                    out.append(logits[o:o + c])
                    o += c
                return out

            def _mm_rp(d_real, d_fake, loss_fn, detach_real):
                if _K_stat > 0 and _W_stat > 0.0:
                    rv, rs = d_real[:, :-_K_stat], d_real[:, -_K_stat:]
                    fv, fs = d_fake[:, :-_K_stat], d_fake[:, -_K_stat:]
                    if detach_real:
                        rv, rs = rv.detach(), rs.detach()
                    lv = loss_fn(rv, fv)
                    ls = loss_fn(rs, fs)
                    return lv + _W_stat * ls, float(ls.detach().item())
                rr = d_real.detach() if detach_real else d_real
                return loss_fn(rr, d_fake), 0.0

            # ---- D-updates (resample real each iter) ----
            if n_disc_updates > 0 and self.r3gan_optimizer is not None:
                for _iter_idx in range(n_disc_updates):
                    self.r3gan_optimizer.zero_grad(set_to_none=True)
                    _rc, _rat, _ram = _resample_real(_iter_idx)
                    _rn = _mm_noise(_rc)
                    if diff_aug_policy:
                        _rn, _ = latent_diff_augment(
                            _rn, _rn, policy=diff_aug_policy,
                            seed=int(current_step) * 31 + seed_offset
                            + _iter_idx * 101)
                    _rn = _rn.detach()
                    _fk = fake_chunks_det_noisy.detach()
                    _fseg = (_fk, fake_action_tokens, fake_action_modulation)
                    # R1 (real-side) + R2 (fake-side) gradient penalties.
                    # Append the perturbed segment(s) to the SINGLE
                    # DDP-safe disc forward only on the steps each fires
                    # (offset so the two perturbed forwards never stack —
                    # OOM guard).
                    _do_r1 = (current_step % _r1_every_n == 0)
                    _do_r2 = self._ladd_r2_fires(
                        _r2_gamma, _r2_every_n, _r2_offset, current_step)
                    _segs = [(_rn, _rat, _ram), _fseg]
                    if _do_r1:
                        _eps_r = _r1_sigma * torch.randn_like(_rn)
                        _segs.append((_rn + _eps_r, _rat, _ram))
                    if _do_r2:
                        _eps_f = _r2_sigma * torch.randn_like(_fk)
                        _segs.append((
                            _fk + _eps_f, fake_action_tokens,
                            fake_action_modulation))
                    _outs = _mm_fwd_segments(disc_for_update, _segs)
                    d_real_logits, d_fake_logits = _outs[0], _outs[1]
                    _nxt = 2
                    if _do_r1:
                        d_real_pert = _outs[_nxt]
                        _nxt += 1
                        d_real_sum = d_real_logits.sum(dim=1)
                        d_real_pert_sum = d_real_pert.sum(dim=1)
                        _r1_grad_sq_raw = (
                            ((d_real_pert_sum - d_real_sum) / _r1_sigma)
                            .pow(2).mean())
                        r1 = 0.5 * _r1_gamma * _r1_grad_sq_raw
                        last_r1_grad_sq = float(_r1_grad_sq_raw.detach().item())
                        last_r1_fired = 1.0
                    else:
                        r1 = d_real_logits.sum() * 0.0
                    if _do_r2:
                        d_fake_pert = _outs[_nxt]
                        _nxt += 1
                        d_fake_sum = d_fake_logits.sum(dim=1)
                        d_fake_pert_sum = d_fake_pert.sum(dim=1)
                        _r2_grad_sq_raw = (
                            ((d_fake_pert_sum - d_fake_sum) / _r2_sigma)
                            .pow(2).mean())
                        r2 = 0.5 * _r2_gamma * _r2_grad_sq_raw
                        last_r2_grad_sq = float(_r2_grad_sq_raw.detach().item())
                        last_r2_fired = 1.0
                    else:
                        r2 = d_fake_logits.sum() * 0.0
                    d_rp, _dstat = _mm_rp(
                        d_real_logits, d_fake_logits, _d_loss_fn, False)
                    (d_rp + r1 + r2).backward()
                    if self.gan_max_grad_norm and self.gan_max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.r3gan_disc.parameters()
                             if p.grad is not None],
                            self.gan_max_grad_norm)
                    self.r3gan_optimizer.step()
                    last_d_loss = float(d_rp.detach().item())
                    last_d_real = float(d_real_logits.detach().mean().item())
                    last_d_fake = float(d_fake_logits.detach().mean().item())
                    last_r1 = float(r1.detach().item())
                    last_r2 = float(r2.detach().item())
                    last_d_loss_stat = _dstat
            self._mem_step_snapshot(f"ladd_{pair_mode}_b_post_d_update")

            # ---- Gen-side (resample real once) ----
            critic_warmup_done = (
                current_step >= int(getattr(self, "gan_critic_warmup_steps", 0)))
            if (
                self.gan_warmup_steps > 0
                and current_step < (
                    self.gan_critic_warmup_steps + self.gan_warmup_steps)
                and current_step >= self.gan_critic_warmup_steps
            ):
                _ramp_in = current_step - self.gan_critic_warmup_steps
                gen_gan_weight = self._gan_warmup_shape_apply(
                    _ramp_in / max(1, self.gan_warmup_steps)
                ) * self.gan_loss_weight
            elif current_step >= (
                self.gan_critic_warmup_steps + self.gan_warmup_steps
            ):
                gen_gan_weight = self.gan_loss_weight
            else:
                gen_gan_weight = 0.0
            gen_gan_weight = gen_gan_weight * float(
                getattr(self.model, "ladd_disc_loss_weight", 1.0))

            generator_gan_loss = zero
            gen_gan_main_value = 0.0
            gen_gan_stat_value = 0.0
            if critic_warmup_done and gen_gan_weight > 0 and not skip_g:
                disc_for_guidance.requires_grad_(False)
                _disc_was_training = disc_for_guidance.training
                disc_for_guidance.eval()
                try:
                    _rc, _rat, _ram = _resample_real(7919)
                    _rn = _mm_noise(_rc).detach()
                    _fg = _mm_noise(fake_chunks_grad_tensor)
                    if diff_aug_policy:
                        _rn, _ = latent_diff_augment(
                            _rn, _rn, policy=diff_aug_policy,
                            seed=int(current_step) * 31 + seed_offset + 99)
                        _, _fg = latent_diff_augment(
                            _fg, _fg, policy=diff_aug_policy,
                            seed=int(current_step) * 31 + seed_offset + 199)
                    d_real_g, d_fake_g = _mm_fwd_segments(
                        disc_for_guidance, [
                            (_rn, _rat, _ram),
                            (_fg, fake_action_tokens, fake_action_modulation),
                        ])
                    g_rp, gen_gan_stat_value = _mm_rp(
                        d_real_g, d_fake_g, _g_loss_fn, True)
                    generator_gan_loss = (
                        gen_gan_weight * g_rp.to(pred_image_dtype))
                    gen_gan_main_value = float(g_rp.detach().item())
                finally:
                    disc_for_guidance.requires_grad_(True)
                    if _disc_was_training:
                        disc_for_guidance.train()
            self._mem_step_snapshot(f"ladd_{pair_mode}_c_post_g_forward")

            logs = {
                "train/r3gan_disc_skipped": 1.0 if disc_skipped else 0.0,
                "train/r3gan_d_loss": last_d_loss,
                "train/r3gan_d_real": last_d_real,
                "train/r3gan_d_fake_detached": last_d_fake,
                "train/r3gan_r1": last_r1,
                "train/r3gan_r1_grad_sq": last_r1_grad_sq,
                "train/r3gan_r1_fired": last_r1_fired,
                "train/r3gan_r1_gamma": float(_r1_gamma),
                "train/r3gan_r2": last_r2,
                "train/r3gan_r2_grad_sq": last_r2_grad_sq,
                "train/r3gan_r2_fired": last_r2_fired,
                "train/r3gan_r2_gamma": float(_r2_gamma),
                "train/r3gan_g_loss_raw": gen_gan_main_value,
                "train/r3gan_g_loss_weighted": gen_gan_weight * gen_gan_main_value,
                "train/r3gan_g_weight": float(gen_gan_weight),
                "train/critic_warmup_done": 1.0 if critic_warmup_done else 0.0,
                "train/ladd_n_pairs": float(n_pairs),
                "train/ladd_n_real": float(_n_real),
                "train/ladd_disc_t": float(disc_t_int),
                "train/r3gan_d_loss_stat": last_d_loss_stat,
                "train/r3gan_d_real_stat": last_d_real_stat,
                "train/r3gan_d_fake_detached_stat": last_d_fake_stat,
                "train/r3gan_g_loss_raw_stat": gen_gan_stat_value,
                "train/r3gan_stat_loss_weight": float(
                    getattr(self.model, "ladd_stat_head_loss_weight", 1.0)),
            }
            return generator_gan_loss, logs

        if n_disc_updates > 0 and self.r3gan_optimizer is not None:
            for _ in range(n_disc_updates):
                self.r3gan_optimizer.zero_grad(set_to_none=True)
                # ----- Batched real+fake forward (v28A OOM mitigation) -----
                # Stack real + fake into ONE disc forward; halves the
                # teacher-class forwards (2 → 1) for the D-update.
                # R1 penalty stays correct: ``d_real_logits.sum()`` only
                # depends on the real rows, so the gradient w.r.t. fake
                # rows is identically zero.
                B_pair_d = real_chunks_det_noisy.shape[0]
                combined_in = torch.cat(
                    [real_chunks_det_noisy.detach(),
                     fake_chunks_det_noisy.detach()],
                    dim=0,
                ).requires_grad_(True)
                _disc_t_d = torch.cat([t_disc, t_disc], dim=0)
                _pe_d = torch.cat([prompt_embeds_eff, prompt_embeds_eff], dim=0)
                _pp_d = (
                    torch.cat([pooled_prompt, pooled_prompt], dim=0)
                    if pooled_prompt is not None else None
                )
                _combined_cond_extra = None
                if real_action_tokens is not None:
                    _combined_cond_extra = {
                        "_action_tokens": torch.cat(
                            [real_action_tokens, fake_action_tokens], dim=0,
                        ),
                    }
                    if real_action_modulation is not None:
                        _combined_cond_extra["_action_modulation"] = torch.cat(
                            [real_action_modulation, fake_action_modulation],
                            dim=0,
                        )
                # R1 regularization. Two estimator modes
                # (``ladd_r1_mode`` on the model):
                #   "fd" (default): finite-difference stochastic
                #         estimator. ``r1_grad = (D(x+sigma*eps) -
                #         D(x)) / sigma``; ``r1 = mean(r1_grad**2)``.
                #         ONE extra disc forward, no second-order
                #         autograd graph. Cheap. Reference impl:
                #         Causal-Forcing/model/gan.py:258-271.
                #   "autograd": exact ``torch.autograd.grad(...,
                #         create_graph=True)``. Holds the second-order
                #         graph; defeats disc gradient checkpointing;
                #         ~2× the memory of "fd". Legacy.
                # Lazy R1 (``ladd_r1_every_n_steps`` > 1) skips R1 on
                # off-iters regardless of mode.
                _r1_mode = str(getattr(self.model, "ladd_r1_mode", "fd"))
                _r1_every_n = max(1, int(
                    getattr(self.model, "ladd_r1_every_n_steps", 1)
                ))
                _do_r1 = (current_step % _r1_every_n == 0)
                _r1_gamma = float(
                    getattr(self.model, "ladd_r1_gamma", 1.0)
                )
                _r1_sigma = float(
                    getattr(self.model, "ladd_r1_sigma", 0.01)
                )
                # R2 (fake-side penalty) — on its own offset cadence so
                # its perturbed-fake forward never stacks with R1's
                # perturbed-real forward in the same D-update (OOM guard).
                _r2_gamma, _r2_every_n, _r2_offset, _r2_sigma = (
                    self._ladd_r2_knobs(_r1_every_n, _r1_sigma)
                )
                _do_r2 = self._ladd_r2_fires(
                    _r2_gamma, _r2_every_n, _r2_offset, current_step)

                if _do_r1 and _r1_mode == "autograd":
                    combined_logits = disc_for_update(
                        x_noisy=combined_in,
                        timestep=_disc_t_d,
                        prompt_embeds=_pe_d,
                        pooled_prompt=_pp_d,
                        conditional_extra=_combined_cond_extra,
                    )
                    d_real_logits = combined_logits[:B_pair_d]
                    d_fake_logits = combined_logits[B_pair_d:]
                    grads = torch.autograd.grad(
                        d_real_logits.sum(),
                        combined_in,
                        create_graph=True,
                        retain_graph=True,
                    )[0]
                    real_grads_only = grads[:B_pair_d]
                    # Raw ‖∇_x Σ_i D_i‖² (per-sample, then averaged) —
                    # the quantity γ multiplies. Captured BEFORE the
                    # 0.5·γ scaling so wandb shows the gradient norm
                    # independent of the γ knob.
                    _r1_grad_sq_raw = (
                        real_grads_only.flatten(1).pow(2).sum(dim=1).mean()
                    )
                    r1 = 0.5 * _r1_gamma * _r1_grad_sq_raw
                    last_r1_grad_sq = float(_r1_grad_sq_raw.detach().item())
                    last_r1_fired = 1.0
                    # Autograd R2: separate ∇ of the FAKE logits w.r.t.
                    # the (fake half of the) combined input. d_fake only
                    # depends on the fake rows, so the real-row grads are
                    # zero — we slice the fake half explicitly.
                    if _do_r2:
                        grads_f = torch.autograd.grad(
                            d_fake_logits.sum(),
                            combined_in,
                            create_graph=True,
                            retain_graph=True,
                        )[0]
                        fake_grads_only = grads_f[B_pair_d:]
                        _r2_grad_sq_raw = (
                            fake_grads_only.flatten(1).pow(2).sum(dim=1).mean()
                        )
                        r2 = 0.5 * _r2_gamma * _r2_grad_sq_raw
                        last_r2_grad_sq = float(_r2_grad_sq_raw.detach().item())
                        last_r2_fired = 1.0
                    else:
                        r2 = combined_logits.sum() * 0.0
                elif (_do_r1 and _r1_mode == "fd") or _do_r2:
                    # Finite-difference R1 and/or R2. Detach inputs (no
                    # second-order graph). Append the perturbed real
                    # (R1) and/or perturbed fake (R2) segment to ONE
                    # batched disc forward — only the segment(s) that
                    # fire this step are appended (offset → at most one
                    # of R1/R2 here per step under equal cadences).
                    # NOTE: reaching this branch with ``_do_r1`` True
                    # implies ``_r1_mode == "fd"`` — the autograd+R1 case
                    # is fully handled above, so ``_do_r1`` here always
                    # means an FD R1.
                    real_part = combined_in[:B_pair_d].detach()
                    fake_part = combined_in[B_pair_d:].detach()
                    _fd_segs = [real_part, fake_part]
                    if _do_r1:
                        eps_real = _r1_sigma * torch.randn_like(real_part)
                        _fd_segs.append(real_part + eps_real)
                    if _do_r2:
                        eps_fake = _r2_sigma * torch.randn_like(fake_part)
                        _fd_segs.append(fake_part + eps_fake)
                    _n_seg = len(_fd_segs)
                    combined_in_fd = torch.cat(_fd_segs, dim=0)
                    _disc_t_d_fd = torch.cat([t_disc] * _n_seg, dim=0)
                    _pe_d_fd = torch.cat(
                        [prompt_embeds_eff] * _n_seg, dim=0,
                    )
                    _pp_d_fd = (
                        torch.cat([pooled_prompt] * _n_seg, dim=0)
                        if pooled_prompt is not None else None
                    )
                    _cond_extra_fd = None
                    if _combined_cond_extra is not None:
                        # The combined dict has [real, fake] sub-batches
                        # along dim 0. Append the real slice for the R1
                        # segment and/or the fake slice for the R2
                        # segment, matching ``_fd_segs`` order.
                        _cond_extra_fd = {}
                        for k, v in _combined_cond_extra.items():
                            real_v = v[:B_pair_d]
                            fake_v = v[B_pair_d:]
                            _parts = [v]
                            if _do_r1:
                                _parts.append(real_v)
                            if _do_r2:
                                _parts.append(fake_v)
                            _cond_extra_fd[k] = torch.cat(_parts, dim=0)
                    combined_logits = disc_for_update(
                        x_noisy=combined_in_fd,
                        timestep=_disc_t_d_fd,
                        prompt_embeds=_pe_d_fd,
                        pooled_prompt=_pp_d_fd,
                        conditional_extra=_cond_extra_fd,
                    )
                    d_real_logits = combined_logits[:B_pair_d]
                    d_fake_logits = combined_logits[B_pair_d:2 * B_pair_d]
                    _off = 2 * B_pair_d
                    # Finite-difference penalties — calibrated to match
                    # the autograd path: ``(γ/2) · E_x[‖∇_x Σ_i D_i‖²]``.
                    # SUM the per-token logits per sample FIRST, then take
                    # the finite difference: ``(D_sum(x+σε) − D_sum(x))/σ
                    # ≈ ∇_x D_sum · ε``; squaring + expectation gives
                    # ``‖Σ_i ∇_x D_i‖²``, the quantity γ multiplies. The
                    # 0.5 factor mirrors the autograd path so γ tunes
                    # uniformly across modes.
                    if _do_r1:
                        d_real_perturbed = combined_logits[_off:_off + B_pair_d]
                        _off += B_pair_d
                        d_real_sum = d_real_logits.sum(dim=1)
                        d_real_pert_sum = d_real_perturbed.sum(dim=1)
                        _r1_grad_sq_raw = (
                            (d_real_pert_sum - d_real_sum) / _r1_sigma
                        ).pow(2).mean()
                        r1 = 0.5 * _r1_gamma * _r1_grad_sq_raw
                        last_r1_grad_sq = float(_r1_grad_sq_raw.detach().item())
                        last_r1_fired = 1.0
                    else:
                        r1 = d_real_logits.sum() * 0.0
                    if _do_r2:
                        d_fake_perturbed = combined_logits[_off:_off + B_pair_d]
                        _off += B_pair_d
                        d_fake_sum = d_fake_logits.sum(dim=1)
                        d_fake_pert_sum = d_fake_perturbed.sum(dim=1)
                        _r2_grad_sq_raw = (
                            (d_fake_pert_sum - d_fake_sum) / _r2_sigma
                        ).pow(2).mean()
                        r2 = 0.5 * _r2_gamma * _r2_grad_sq_raw
                        last_r2_grad_sq = float(_r2_grad_sq_raw.detach().item())
                        last_r2_fired = 1.0
                    else:
                        r2 = d_fake_logits.sum() * 0.0
                else:
                    # Neither R1 nor R2 fires this iter (lazy). Detached
                    # forward for stability + memory.
                    combined_logits = disc_for_update(
                        x_noisy=combined_in.detach(),
                        timestep=_disc_t_d,
                        prompt_embeds=_pe_d,
                        pooled_prompt=_pp_d,
                        conditional_extra=_combined_cond_extra,
                    )
                    d_real_logits = combined_logits[:B_pair_d]
                    d_fake_logits = combined_logits[B_pair_d:]
                    r1 = combined_logits.sum() * 0.0  # graph-connected zero
                    r2 = combined_logits.sum() * 0.0
                # Per-token RpGAN: feed the full [B, K·T'·H'·W']
                # logit tensors directly. ``rpgan_d_loss = softplus(
                # d_fake - d_real).mean()`` is elementwise, so this
                # gives each (frame, spatial-position, tap) token its
                # own per-position contribution to the disc loss
                # (and, symmetrically, its own per-position gradient
                # signal back to the disc weights). The previous
                # ``.mean(dim=1)`` collapsed positions to a single
                # scalar per sample BEFORE the softplus, which made
                # the disc see one global score per sample and
                # erased per-position discrimination.
                #
                # Visual / stat split (when ``stat_head_enabled``):
                # the disc forward appends K stat-logits at the END
                # of the per-token visual logits. With ~thousands of
                # visual tokens vs K stat-logits, lumping them into
                # one ``.mean()`` reduction would drown the stat side
                # (contribution K / (N_visual + K)). Instead we split,
                # reduce each side independently (each mean is O(1)),
                # and combine with ``ladd_stat_head_loss_weight``.
                # Default weight 1.0 → ~50/50 stat:visual at the
                # gradient level. Set weight=0 to neuter the stat side.
                _K_stat = int(getattr(self.r3gan_disc, "stat_logit_count", 0))
                _W_stat = float(getattr(
                    self.model, "ladd_stat_head_loss_weight", 1.0,
                ))
                if _K_stat > 0 and _W_stat > 0.0:
                    d_real_v = d_real_logits[:, :-_K_stat]
                    d_fake_v = d_fake_logits[:, :-_K_stat]
                    d_real_s = d_real_logits[:, -_K_stat:]
                    d_fake_s = d_fake_logits[:, -_K_stat:]
                    d_rp_visual = _d_loss_fn(d_real_v, d_fake_v)
                    d_rp_stat = _d_loss_fn(d_real_s, d_fake_s)
                    d_rp = d_rp_visual + _W_stat * d_rp_stat
                    last_d_loss_stat = float(d_rp_stat.detach().item())
                    last_d_real_stat = float(d_real_s.detach().mean().item())
                    last_d_fake_stat = float(d_fake_s.detach().mean().item())
                else:
                    d_rp = _d_loss_fn(d_real_logits, d_fake_logits)
                    last_d_loss_stat = 0.0
                    last_d_real_stat = 0.0
                    last_d_fake_stat = 0.0
                d_total = d_rp + r1 + r2
                d_total.backward()
                if self.gan_max_grad_norm and self.gan_max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.r3gan_disc.parameters()
                         if p.grad is not None],
                        self.gan_max_grad_norm,
                    )
                self.r3gan_optimizer.step()
                last_d_loss = float(d_rp.detach().item())
                last_d_real = float(d_real_logits.detach().mean().item())
                last_d_fake = float(d_fake_logits.detach().mean().item())
                last_r1 = float(r1.detach().item())
                last_r2 = float(r2.detach().item())
            self._mem_step_snapshot(f"ladd_{pair_mode}_b_post_d_update")

        # ----- Gen-side -----
        critic_warmup_done = (
            current_step >= int(getattr(self, "gan_critic_warmup_steps", 0))
        )
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
        gen_gan_weight = gen_gan_weight * float(
            getattr(self.model, "ladd_disc_loss_weight", 1.0)
        )

        gen_gan_main_value = 0.0
        gen_gan_stat_value = 0.0
        if critic_warmup_done and gen_gan_weight > 0 and not skip_g:
            # Eval mode disables spectral_norm's power-iter mutation
            # of the ``_u`` / ``_sigma`` buffers during this forward.
            # Required to support multi-mode LADD (gt_vs_fake +
            # adjacent_chunks): if mode-1's gen-side forward saved
            # ``_u`` at version V, ANY subsequent forward in train
            # mode (e.g. mode-2's D-update) bumps the version and
            # the eventual outer ``generator_loss.backward()`` would
            # raise a version-counter mismatch on the saved tensor.
            # Eval mode uses the cached ``_sigma`` from the most
            # recent D-update training forward; the D-updates still
            # run in train mode so ``_sigma`` stays current.
            disc_for_guidance.requires_grad_(False)
            disc_was_training = disc_for_guidance.training
            disc_for_guidance.eval()
            try:
                real_chunks_grad_det = real_chunks_det.detach()
                fake_chunks_grad = fake_chunks_grad_tensor
                real_g = _add_disc_noise(real_chunks_grad_det)
                fake_g = _add_disc_noise(fake_chunks_grad)
                if diff_aug_policy:
                    real_g, fake_g = latent_diff_augment(
                        real_g, fake_g,
                        policy=diff_aug_policy,
                        seed=int(current_step) * 31 + seed_offset,
                    )
                # ----- Batched gen-side forward (v28A OOM mitigation) -----
                # Stack real (detached anchor) + fake (grad-on) into ONE
                # disc forward; halves the teacher-class forwards (2 → 1)
                # on the gen side. d_fake_g still gets its gen-side
                # gradient via the fake rows.
                B_pair_g = real_g.shape[0]
                combined_g = torch.cat([real_g.detach(), fake_g], dim=0)
                _disc_t_g = torch.cat([t_disc, t_disc], dim=0)
                _pe_g = torch.cat([prompt_embeds_eff, prompt_embeds_eff], dim=0)
                _pp_g = (
                    torch.cat([pooled_prompt, pooled_prompt], dim=0)
                    if pooled_prompt is not None else None
                )
                _gen_combined_cond_extra = None
                if real_action_tokens is not None:
                    _gen_combined_cond_extra = {
                        "_action_tokens": torch.cat(
                            [real_action_tokens, fake_action_tokens], dim=0,
                        ),
                    }
                    if real_action_modulation is not None:
                        _gen_combined_cond_extra["_action_modulation"] = (
                            torch.cat(
                                [real_action_modulation, fake_action_modulation],
                                dim=0,
                            )
                        )
                combined_g_logits = disc_for_guidance(
                    x_noisy=combined_g,
                    timestep=_disc_t_g,
                    prompt_embeds=_pe_g,
                    pooled_prompt=_pp_g,
                    conditional_extra=_gen_combined_cond_extra,
                )
                d_real_g = combined_g_logits[:B_pair_g]
                d_fake_g = combined_g_logits[B_pair_g:]
                # Per-token RpGAN on the gen side (mirrors the D-side
                # change above). ``rpgan_g_loss = softplus(d_real -
                # d_fake).mean()`` is elementwise — feeding the full
                # [B, K·T'·H'·W'] logit tensors gives each
                # (frame, spatial-position, tap) token its own
                # ``-sigmoid(d_real_i - d_fake_i) / N`` upstream
                # gradient. Combined with the 2D-conv heads, this
                # delivers per-frame-per-spatial-position-per-tap
                # signal back to the generator instead of a single
                # uniformly-magnitude'd scalar gradient per sample.
                #
                # Visual / stat split — same logic as the D-side: the
                # stat head's K appended logits get their own RpGAN
                # mean reduction, then sum with ``ladd_stat_head_loss
                # _weight``. Default 1.0 → stat side at parity with
                # visual side at the gen-gradient level. Without this
                # split, the stat side's contribution to the gen
                # gradient would be K / (N_visual + K) ≈ 0.
                _K_stat_g = int(getattr(self.r3gan_disc, "stat_logit_count", 0))
                _W_stat_g = float(getattr(
                    self.model, "ladd_stat_head_loss_weight", 1.0,
                ))
                if _K_stat_g > 0 and _W_stat_g > 0.0:
                    d_real_g_v = d_real_g[:, :-_K_stat_g]
                    d_fake_g_v = d_fake_g[:, :-_K_stat_g]
                    d_real_g_s = d_real_g[:, -_K_stat_g:]
                    d_fake_g_s = d_fake_g[:, -_K_stat_g:]
                    g_rp_visual = _g_loss_fn(d_real_g_v.detach(), d_fake_g_v)
                    g_rp_stat = _g_loss_fn(d_real_g_s.detach(), d_fake_g_s)
                    g_rp = g_rp_visual + _W_stat_g * g_rp_stat
                    gen_gan_stat_value = float(g_rp_stat.detach().item())
                else:
                    g_rp = _g_loss_fn(d_real_g.detach(), d_fake_g)
                    gen_gan_stat_value = 0.0
                generator_gan_loss = (
                    gen_gan_weight * g_rp.to(pred_image_dtype)
                )
                gen_gan_main_value = float(g_rp.detach().item())
            finally:
                disc_for_guidance.requires_grad_(True)
                if disc_was_training:
                    disc_for_guidance.train()
        else:
            generator_gan_loss = zero
        self._mem_step_snapshot(f"ladd_{pair_mode}_c_post_g_forward")

        logs = {
            "train/r3gan_disc_skipped": 1.0 if disc_skipped else 0.0,
            "train/r3gan_d_loss": last_d_loss,
            "train/r3gan_d_real": last_d_real,
            "train/r3gan_d_fake_detached": last_d_fake,
            # γ-weighted R1 penalty actually added to d_total. NOTE: on
            # lazy-skipped iters this is 0.0 (the graph-connected zero);
            # the trace will look almost-flat with γ=0.001. Use the
            # ``r3gan_r1_grad_sq`` key below to see the underlying
            # gradient norm regardless of γ + lazy schedule.
            "train/r3gan_r1": last_r1,
            # Raw ‖∇_x Σ_i D_i‖² estimate — γ-free, lazy-aware (NaN on
            # iters where R1 didn't actually fire so wandb plots gaps).
            # This is the right knob to watch when calibrating
            # ``ladd_r1_gamma``: if it stays ~0, R1 has nothing to bite
            # on; if it climbs into the thousands, the disc is becoming
            # sharp and γ should go up.
            "train/r3gan_r1_grad_sq": last_r1_grad_sq,
            # 1.0 iff R1 was actually computed this disc-update iter
            # (i.e. ``current_step % ladd_r1_every_n_steps == 0`` AND
            # the disc isn't in warmup). Lets the user filter the
            # ``r3gan_r1`` / ``r3gan_r1_grad_sq`` traces to only the
            # iters where the values are meaningful.
            "train/r3gan_r1_fired": last_r1_fired,
            "train/r3gan_r1_gamma": float(_r1_gamma) if "_r1_gamma" in locals() else float(getattr(self.model, "ladd_r1_gamma", 1.0)),
            # R2 (fake-side penalty) counterparts of the R1 traces above.
            "train/r3gan_r2": last_r2,
            "train/r3gan_r2_grad_sq": last_r2_grad_sq,
            "train/r3gan_r2_fired": last_r2_fired,
            "train/r3gan_r2_gamma": float(getattr(self.model, "ladd_r2_gamma", 0.0)),
            "train/r3gan_g_loss_raw": gen_gan_main_value,
            "train/r3gan_g_loss_weighted": (
                gen_gan_weight * gen_gan_main_value
            ),
            "train/r3gan_g_weight": float(gen_gan_weight),
            "train/critic_warmup_done": 1.0 if critic_warmup_done else 0.0,
            "train/ladd_n_pairs": float(n_pairs),
            "train/ladd_disc_t": float(disc_t_int),
            # Stat-head diagnostics (0.0 when stat head is off or
            # ``ladd_stat_head_loss_weight=0``). The "_stat" suffix lets
            # the user compare stat-side and visual-side dynamics
            # independently in wandb.
            "train/r3gan_d_loss_stat": last_d_loss_stat,
            "train/r3gan_d_real_stat": last_d_real_stat,
            "train/r3gan_d_fake_detached_stat": last_d_fake_stat,
            "train/r3gan_g_loss_raw_stat": gen_gan_stat_value,
            "train/r3gan_stat_loss_weight": float(
                getattr(self.model, "ladd_stat_head_loss_weight", 1.0)
            ),
        }
        return generator_gan_loss, logs

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

        # Dispatch to the LADD adjacent-chunk flow when configured.
        if self.gan_backbone == "ladd_teacher_feat":
            return self._compute_ladd_losses(
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
            # Freeze the disc for the G-side path.
            disc_for_guidance.requires_grad_(False)
            try:
                d_real_for_g = disc_for_guidance(real_detached).detach()
                d_fake_for_g = disc_for_guidance(pred_image_for_g)
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
        index_overlay: Optional[Tuple[int, int, int]] = None,
        cap_frames: bool = True,
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
            if cap_frames and self.sample_max_frames > 0:
                F_total = latents.shape[1]
                if F_total > self.sample_max_frames:
                    latents = latents[:, -self.sample_max_frames:]
            lat = latents[0:1]
            # Cached-decode path: WAN VAE's plain ``decode`` re-runs
            # its unconditioned init at the first latent of every
            # call, leaving a brightness anomaly in the rendered
            # video. ``cached_decode`` keeps the temporal-conv
            # feat_map populated across calls so the init is encoded
            # at most once per streaming sequence. The cache is
            # cleared at ``setup_sequence`` time (= when a new ride
            # / video is loaded) — within the same sequence, calls
            # SHARE cache so successive renders / boundary decodes
            # of the same video flow smoothly through one another's
            # left-context.
            pixels = vae.decode_to_pixel(lat, use_cache=True)
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

    @torch.no_grad()
    def _build_holdout_eval_dataset(self):
        """Lazily build a small ZarrRideDataset over ``holdout_eval_root``,
        used to SEED the pred_image_7_chunk eval from rides the model never
        trains on (true held-out generalization eval). Built with NO cache
        (a 20-ride scan is fast) so all ranks can build concurrently with no
        manifest-write race. Returns None when holdout_eval_root is unset."""
        if getattr(self, "_holdout_eval_dataset_built", False):
            return self._holdout_eval_dataset
        self._holdout_eval_dataset_built = True
        self._holdout_eval_dataset = None
        root = getattr(self.config, "holdout_eval_root", None)
        if not root:
            return None
        from utils.zarr_dataset import ZarrRideDataset
        cfg = self.config
        try:
            self._holdout_eval_dataset = ZarrRideDataset(
                encoded_root=str(root),
                caption_root=str(cfg.caption_root),
                motion_root=str(cfg.motion_root),
                ss_vae_checkpoint=str(cfg.ss_vae_checkpoint),
                min_ride_frames=int(getattr(cfg, "min_ride_frames", 64)),
                device="cpu",
                max_rides=None,
                sort_by_length=None,
                cache_path=None,  # no shared cache -> no cross-rank write race
            )
            if self.is_main_process:
                logging.info(
                    "[holdout-eval] 7chunk eval seeds from %d held-out rides "
                    "in %s", len(self._holdout_eval_dataset), str(root),
                )
        except Exception as e:  # never let eval-setup kill training
            logging.warning(
                "[holdout-eval] failed to build holdout dataset from %s: %r "
                "-> falling back to training-ride 7chunk seed.", str(root), e,
            )
            self._holdout_eval_dataset = None
        return self._holdout_eval_dataset

    def _holdout_eval_ride_for_step(self, step: int):
        """Return a {latents, z_actions, prompt_embeds} 14-chunk slice from
        the holdout set to seed the 7chunk eval, or None. DETERMINISTIC per
        step (same ride on every rank -> DDP-balanced + runs comparable).
        ``holdout_eval_mode``: 'cycle' (default) rotates a different held-out
        ride per eval step; 'fixed' always uses ride 0 (watch one scene)."""
        ds = self._build_holdout_eval_dataset()
        if ds is None or len(ds) == 0:
            return None
        npb = int(getattr(self.config, "num_frame_per_block", 3))
        need = 14 * npb
        mode = str(getattr(self.config, "holdout_eval_mode", "cycle")).lower()
        interval = max(1, int(getattr(self, "sample_interval", 1)))
        ordinal = int(step) // interval
        n = len(ds)
        for off in range(n):  # skip any too-short ride (rare; all >=69 frames)
            idx = 0 if mode == "fixed" else (ordinal + off) % n
            meta = ds[idx]
            if int(meta.get("n_latent_frames", 0)) < need:
                continue
            ride = _load_ride_tensors(
                ds, meta, self.device, self.dtype,
                action_dims=self.action_dims, max_frames=need,
            )
            if ride is None or int(ride["latents"].shape[1]) < need:
                continue
            return {
                "latents": ride["latents"][:, :need].contiguous(),
                "z_actions": ride["z_actions"][:, :need].contiguous(),
                "prompt_embeds": ride["prompt_embeds"],
            }
        return None

    def _log_pred_image_7chunk_sample(self, step: int) -> None:
        """Eval rollout: seed the causal student with 7 GT-context chunks
        (21 latent frames) and roll out 7 more chunks (21 frames), then
        log the full 14-chunk (42-frame) video at ``sample_fps`` under
        ``sample/pred_image_7_chunk``. GT actions drive the whole window.

        Runs on ALL ranks (the underlying ``inference_with_trajectory``
        does a cross-rank exit-flag broadcast, so every rank must call it
        to stay collective-balanced) but only the main rank decodes and
        uploads its own ride's video. A readiness MIN-reduce guarantees
        all ranks agree to run-or-skip together, so the broadcast never
        deadlocks. The training gen+critic backward for this step is
        already complete and ``max_rolls_per_ride=1`` re-seeds next step,
        so transiently overwriting the inference KV cache here is safe.
        """
        rd = getattr(self, "_sample_7chunk_ride", None)
        ready_local = 1 if rd is not None else 0
        if dist.is_initialized() and dist.get_world_size() > 1:
            flag = torch.tensor(
                [ready_local], device=self.device, dtype=torch.long,
            )
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            if int(flag.item()) == 0:
                if self.is_main_process:
                    logging.info(
                        "[ActionForcing] 7chunk SKIP at step=%d: a rank had "
                        "no stashed ride (local_ready=%d).",
                        int(step), ready_local,
                    )
                return
        elif not ready_local:
            return
        if self.is_main_process:
            logging.info(
                "[ActionForcing] 7chunk RUN at step=%d: rolling 7 GT ctx + "
                "7 student chunks.", int(step),
            )

        npb = int(getattr(self.config, "num_frame_per_block", 3))
        seed_frames = 7 * npb
        roll_frames = 7 * npb
        gt = rd["latents"].to(device=self.device, dtype=self.dtype)
        act = rd["z_actions"].to(device=self.device)
        prompt = rd["prompt_embeds"].to(device=self.device)
        seed = gt[:, :seed_frames]
        B, _, C, H, W = gt.shape
        noise = torch.randn(
            B, roll_frames, C, H, W, device=self.device, dtype=self.dtype,
        )
        # cond must cover seed (cf) + rollout frames, seed first.
        cond, _uncond = self.model.build_action_conditional(
            prompt_embeds=prompt,
            gt_actions=act[:, : seed_frames + roll_frames],
        )
        pipe = self.model.inference_pipeline
        pipe.reset_cache_state()
        out, _, _ = pipe.inference_with_trajectory(
            noise=noise,
            seed_latents=seed,
            requires_grad=False,
            **cond,
        )
        pipe.reset_cache_state()
        # The training ride's persistent streaming KV state was just
        # clobbered by the inference rollout above; force a fresh setup
        # on the next step rather than reusing a now-inconsistent cache.
        self.model.reset_streaming_state()

        if self.is_main_process:
            try:
                video_latents = torch.cat([seed.to(out.dtype), out], dim=1)
                self._log_pred_image_video(
                    video_latents, int(step), name="pred_image_7_chunk",
                    caption_suffix="7 GT ctx + 7 student rolled (GT actions)",
                    cap_frames=False,
                )
            except Exception as exc:  # pragma: no cover - logging only
                logging.warning(
                    "[ActionForcing] pred_image_7_chunk log failed at "
                    "step=%d: %s", int(step), exc,
                )

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

    def _dump_module_inventory(self) -> None:
        """Print a one-shot startup inventory of which trainer-side
        modules + optimizers are actually built (vs ``None``). Reveals
        whether any feature the operator believes is off (e.g.
        forward_noiser, alt_head, state_probe) is in fact allocating
        GPU memory. Only main process logs.
        """
        if not getattr(self, "is_main_process", True):
            return
        m = getattr(self, "model", None)
        names_trainer = [
            "r3gan_disc", "r3gan_disc_ddp", "r3gan_optimizer",
            "forward_noiser_optimizer", "state_probe_optimizer",
            "real_teacher_optimizer", "critic_optimizer",
            "fake_optimizer", "optimizer",
            "_frozen_cotracker", "_frozen_ss_vae", "_frozen_vae",
        ]
        names_model = [
            "generator", "fake_score", "real_score", "real_score_frozen",
            "action_projection", "action_token_projection",
            "action_critic", "state_probe", "forward_noiser",
            "vae",
        ]
        logging.info("[mem-inventory] === Trainer-side attributes ===")
        for n in names_trainer:
            val = getattr(self, n, None)
            tag = "ON" if val is not None else "off"
            type_name = type(val).__name__ if val is not None else "-"
            logging.info(
                "[mem-inventory]   self.%-32s  %-4s  type=%s",
                n, tag, type_name,
            )
        if m is not None:
            logging.info("[mem-inventory] === Model-side attributes ===")
            for n in names_model:
                val = getattr(m, n, None)
                tag = "ON" if val is not None else "off"
                type_name = type(val).__name__ if val is not None else "-"
                logging.info(
                    "[mem-inventory]   model.%-32s %-4s  type=%s",
                    n, tag, type_name,
                )
            # Booleans / config-derived flags that toggle subsystems
            flags = [
                "gan_enabled", "gan_backbone",
                "flash_dmd_enabled", "boundary_vae_roundtrip",
                "dmd_frozen_teacher_pass_enabled",
                "real_teacher_train_online",
                "fake_alt_head_enabled", "forward_noiser_enabled",
                "dmd_lookback_chunks",
            ]
            logging.info("[mem-inventory] === Config flags ===")
            for f in flags:
                v_self = getattr(self, f, None)
                v_model = getattr(m, f, None)
                v_config = getattr(getattr(self, "config", None), f, None)
                logging.info(
                    "[mem-inventory]   %-40s self=%r model=%r config=%r",
                    f, v_self, v_model, v_config,
                )

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
        (code default 0.5, but the phase-3 DMD config sets 5.0 to keep
        only clearly high-motion windows; 0.5 merely excludes parked /
        idling windows). Among qualifying windows it samples uniformly.
        With probability 1 - prob, samples uniformly over all offsets.
        If motion is unavailable for this ride (or the window bounds
        don't fit), falls back to uniform sampling on ``[0, s_local_max]``.
        If motion IS available but NO window clears the threshold, picks
        the ride's single highest-motion window (argmax) — never a random
        low-motion offset — so the high-motion guarantee holds per ride.

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
        # Work directly on npb-aligned candidate offsets so we never bias
        # toward pseudo-aligned offsets via a post-hoc snap.
        aligned_all = np.arange(0, s_local_max + 1, npb)
        aligned_means = means[aligned_all]
        aligned_valid = aligned_all[aligned_means >= threshold]
        if aligned_valid.size > 0:
            # Uniform among the high-motion (>= threshold) windows.
            return int(aligned_valid[random.randrange(aligned_valid.size)])
        # No aligned window clears the threshold for this ride. Rather
        # than a uniform-random (low-motion) offset, pick the ride's
        # single HIGHEST-motion window (argmax) so "high motion only"
        # still holds even on rides whose best window is below cutoff.
        return int(aligned_all[int(np.argmax(aligned_means))])

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
                random_window=bool(
                    getattr(self.config, "max_ride_frames_random", False)
                ),
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
        # Rolling-depth CURRICULUM (phase-2 rolling): cap the rolls per
        # ride at ``base + (step - start) // every`` so early training
        # re-seeds often (short rides) and later training rolls deeper.
        # E.g. base=2, every=50: steps [start, start+50) -> first window
        # + 1 extra roll; +1 extra roll per 50 steps thereafter. 0 = off
        # (plain max_rolls_per_ride). The schedule only LOWERS the cap.
        _sched_every = int(getattr(
            cfg, "rolling_rolls_schedule_every", 0))
        if _sched_every > 0:
            _sched_base = int(getattr(
                cfg, "rolling_rolls_schedule_base", 2))
            _sched_start = int(getattr(
                cfg, "rolling_rolls_schedule_start_step", 0))
            _cur_step = int(getattr(self, "step", 0))
            max_rolls = min(
                max_rolls,
                _sched_base
                + max(0, _cur_step - _sched_start) // _sched_every,
            )
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

        # Rolling: accumulate the student's WHOLE ride rollout (rank 0,
        # CPU, detached) so sample videos can show the full generated
        # sequence instead of just the last 21f window. Restarted at
        # each ride setup; kept across the ride's reset so the sampler
        # can still emit the completed ride's full video.
        if (
            self.is_main_process
            and int(getattr(
                self.model, "streaming_force_new_frame_chunks", 0)) > 0
        ):
            _nf_acc = int(info["new_frames"])
            _src_acc = info.get("flash_dmd_gan_x0")
            _new_acc = (
                _src_acc if _src_acc is not None else chunk
            )[:, -_nf_acc:].detach().float().cpu()
            if needs_setup or getattr(self, "_rollout_video_acc", None) is None:
                self._rollout_video_acc = _new_acc
            else:
                self._rollout_video_acc = torch.cat(
                    [self._rollout_video_acc, _new_acc], dim=1)

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
        # MAE-collapse reset (phase-2 rolling): reset the ride when the
        # trained window's MAE vs GT crosses ``streaming_mae_collapse_
        # threshold`` (0 = off). ``avg_mae`` is this step's per-rank MAE
        # on the chunk_size window (computed in Stage 3). ``min_chunks``
        # skips the gate on the first roll(s) of a ride, where the MAE
        # is dominated by the fresh-window rollout rather than drift.
        # Per-rank decision; the MAX-reduce below keeps resets lockstep.
        _mae_thr = float(getattr(
            cfg, "streaming_mae_collapse_threshold", 0.0))
        _mae_min_chunks = int(getattr(
            cfg, "streaming_mae_collapse_min_chunks", 2))
        local_mae_collapse = (
            _mae_thr > 0.0
            and self._chunks_in_current_ride >= _mae_min_chunks
            and avg_mae > _mae_thr
        )
        local_hit_cap = self._chunks_in_current_ride >= max_rolls
        local_exhausted = not self.model.can_generate_more()
        local_should_reset = (
            local_hit_cap or local_exhausted or local_mae_collapse
        )
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
            if local_mae_collapse:
                out["streaming_reset_reason_mae_collapse"] = 1.0
            elif local_hit_cap:
                out["streaming_reset_reason_cap"] = 1.0
            elif local_exhausted:
                out["streaming_reset_reason_end_of_ride"] = 1.0
            else:
                out["streaming_reset_reason_peer_triggered"] = 1.0
            # FN frontier pair (phase-2 rolling): train the FN on a fresh
            # rollout1'/rollout2' pair generated AT THE RIDE'S FRONTIER,
            # right before the sequence is torn down (the pair generation
            # clobbers the pipeline KV caches, which is safe ONLY here).
            # Reset is rank-lockstep, so every rank runs exactly one FN
            # backward per reset step (DDP bucket counts stay matched;
            # bail-outs run the zero-anchor).
            if (
                bool(getattr(self.model, "fn_frontier_pairs", False))
                and getattr(self.model, "forward_noiser", None) is not None
                and str(getattr(
                    self.model, "forward_noiser_loss_mode", "mse",
                )) == "teacher_feat"
            ):
                out.update(self._train_fn_frontier_pair())
            self.model.reset_streaming_state()
            # Deferred 7chunk eval (see the gate in
            # ``_streaming_train_one_chunk``): fire ONLY at ride teardown
            # so the eval's KV-cache clobber can't truncate a live
            # multi-roll ride. ``_pending_7chunk_sample`` was set via a
            # rank-0 broadcast and ``should_reset`` is MAX-reduced, so
            # every rank takes this branch together (the eval rollout is
            # an all-ranks collective).
            if (
                getattr(self, "_pending_7chunk_sample", False)
                and bool(getattr(self, "sample_7chunk_enabled", True))
            ):
                self._pending_7chunk_sample = False
                if getattr(self.config, "holdout_eval_root", None):
                    _hr = self._holdout_eval_ride_for_step(
                        int(self.step) + 1)
                    if _hr is not None:
                        self._sample_7chunk_ride = _hr
                self._log_pred_image_7chunk_sample(int(self.step) + 1)
        else:
            out["streaming_did_reset"] = 0.0

        # Phase-2 rolling stderr telemetry (rank 0, only when the
        # deterministic-stride rolling mode is active): one compact line
        # per step so ride depth / MAE / resets are visible offline
        # (wandb-only metrics don't reach .err).
        if (
            self.is_main_process
            and int(getattr(
                self.model, "streaming_force_new_frame_chunks", 0)) > 0
        ):
            import sys as _sys
            if should_reset:
                _rr = (
                    "mae_collapse" if local_mae_collapse
                    else "cap" if local_hit_cap
                    else "end_of_ride" if local_exhausted
                    else "peer"
                )
            else:
                _rr = "-"
            _alloc_gb = _peakstep_gb = 0.0
            if torch.cuda.is_available():
                _alloc_gb = torch.cuda.memory_allocated() / 2**30
                _peakstep_gb = torch.cuda.max_memory_allocated() / 2**30
                # Per-gen-step peak window (rolling mode only — this
                # branch is gated on the deterministic-stride knob).
                torch.cuda.reset_peak_memory_stats()
            print(
                f"[ROLL] step={int(getattr(self, 'step', -1))} "
                f"ride_chunk={self._chunks_in_current_ride}/{max_rolls} "
                f"mae={avg_mae:.4f} reset={int(bool(should_reset))}({_rr}) "
                f"alloc={_alloc_gb:.2f} step_peak={_peakstep_gb:.2f}",
                file=_sys.stderr, flush=True,
            )

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
        # 1-indexed roll counter within the current ride. Read by the
        # dmd_one_step "first chunk only" gate inside
        # compute_generator_loss_streaming.
        train_info["chunks_in_current_ride"] = int(
            getattr(self, "_chunks_in_current_ride", 1)
        )
        gen_loss_dmd, gen_log = self.model.compute_generator_loss_streaming(
            train_chunk, train_info,
        )
        self._mem_step_snapshot("1_after_compute_gen_loss_streaming")

        if _sample_due_now:
            eval_stash = getattr(self.model, "_dmd_eval_stash", None)
            if isinstance(eval_stash, dict) and eval_stash:
                self._pending_dmd_eval_latents = eval_stash
            # Do NOT disarm the stash here: under aux_teacher_separate_
            # backward the aux pass runs LATER in this step and writes
            # ``pred_real_lora`` / ``clean_x_aux`` into the SAME dict
            # (shared reference with _pending_dmd_eval_latents). The
            # early ``= None`` here was why those videos vanished in
            # every separate-backward run (g5+, j2). Disarmed at the
            # end of the step instead.

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

                # Disc-borrowed regulariser on the LoRA's aux x0.
                # See ``_compute_aux_teacher_disc_losses`` for the two
                # variants (RpGAN adv vs feature matching). Both are
                # off when their weight knobs are 0 — no-op otherwise.
                # Pass this iter's ``chunk_lo`` directly (rather than
                # reading the streaming_state stash) — the stash is
                # set further down by the main GAN block and would
                # carry the PREVIOUS iter's value at this call site.
                aux_disc_loss, aux_disc_logs = (
                    self._compute_aux_teacher_disc_losses(
                        lora_x0=aux_tensors.get("lora_x0"),
                        gt_target=aux_tensors.get("gt_target"),
                        chunk_lo=int(chunk_lo),
                        current_step=int(self.step),
                    )
                )
                if aux_disc_loss is not None:
                    generator_loss = generator_loss + aux_disc_loss
                out.update(aux_disc_logs)
                self._mem_step_snapshot("3b_after_aux_disc")

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

        # Moment-GAN runs independently of the main GAN gate (its own
        # ``moment_gan_enabled`` flag). Trained against GT moments
        # (always available); supplies a distribution-matching signal
        # on per-frame std (+ optional mean / RMS) that replaces the
        # point-wise ``latent_std_mse_loss`` anti-collapse signal.
        if self.moment_gan_enabled and self.moment_disc is not None:
            _mgt_window = state["ride_latents_window"][:, chunk_lo:chunk_hi]
            moment_gen_loss, moment_logs = self._compute_moment_gan_losses(
                pred_image=train_chunk,
                gt_latents_window=_mgt_window,
                current_step=int(self.step),
            )
            generator_loss = generator_loss + moment_gen_loss
            out.update(moment_logs)

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

        # Phase-LoRA ghost anchor — see ``_phase_lora_ghost_anchor``
        # for the full rationale (K+1 DDP rebuild fix).
        generator_loss = self._phase_lora_ghost_anchor(generator_loss)

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

        # ForwardNoiser teacher_feat training (separate FN backward, fully
        # decoupled from fake_score). Runs AFTER the gen+critic backwards so
        # the gen/critic graphs are freed before the FN's own teacher
        # forward (memory hygiene). Populates FN grads; the outer train()
        # loop clips + steps the FN optimizer. No-op unless
        # forward_noiser_loss_mode=="teacher_feat".
        if (
            getattr(self.model, "forward_noiser_enabled", False)
            and getattr(self.model, "forward_noiser_loss_mode", "mse")
            == "teacher_feat"
        ):
            fn_tf_logs = self._train_forward_noiser_tf(train_chunk, train_info)
            out.update(fn_tf_logs)
            self._mem_step_snapshot("6b_after_fn_tf_backward")

        # aux_teacher_separate_backward: run the aux teacher as its OWN
        # forward+backward HERE — after the gen+critic (+FN) graphs are
        # freed — so the 1.3B teacher's activation graph never coexists with
        # the GAN R1 double-backward (drops the gen-step peak ~10-15 GB).
        # The fused aux in compute_generator_loss_streaming is gated OFF in
        # this mode. run_extra_aux_pass does a fresh (eps,t) pass and
        # returns the weighted loss; the outer loop clips + steps
        # real_teacher_optimizer and runs the EMA pull. DDP-safe: the
        # all_reduce(MAX) short-ride skip lives inside
        # _compute_aux_teacher_loss_streaming, so all ranks agree on
        # whether a usable loss exists.
        if (
            getattr(self.model, "aux_teacher_separate_backward", False)
            and getattr(self, "real_teacher_optimizer", None) is not None
            and self.step >= int(
                getattr(self.config, "aux_teacher_start_step", 0)
            )
        ):
            aux_loss_sep, aux_log_sep = self.model.run_extra_aux_pass(
                train_chunk.detach(), train_info,
            )
            if aux_loss_sep is not None and aux_loss_sep.requires_grad:
                aux_loss_sep.backward()
            if isinstance(aux_log_sep, dict):
                out.update(aux_log_sep)
            self._mem_step_snapshot("6c_after_aux_separate_backward")

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

        # pred_image_7_chunk eval video (all ranks; balanced collective).
        # Fires on the sample cadence AFTER this step's gen+critic
        # backward. Seeds the student with 7 GT-context chunks and rolls
        # 7 more, logging the full 14-chunk video on the main rank. The
        # inference rollout clobbers the KV cache + streaming state, but
        # ``max_rolls_per_ride=1`` re-seeds next step, so it is reset
        # inside the helper. Gated on a pure-step condition so every
        # rank takes the same branch (collective-safe).
        # Fire the pred_image_7_chunk eval video on the SAME cadence as
        # the existing pred_image sample (``_sample_due_now``, computed
        # at the top of this method). That signal is main-rank-only
        # (``_video_sample_due`` returns False off-rank), so broadcast it
        # from rank 0 to keep the downstream all-ranks inference rollout
        # collective-balanced. Tying to ``_sample_due_now`` (rather than
        # a pure-step gate) is necessary because this gen step only runs
        # once every ``dfake_gen_update_ratio`` steps, so a raw
        # ``step % interval`` rarely aligns with the gen cadence.
        _do7 = bool(getattr(self, "sample_7chunk_enabled", True))
        if _do7:
            _due_flag = torch.tensor(
                [1 if _sample_due_now else 0],
                device=self.device, dtype=torch.long,
            )
            if dist.is_initialized() and dist.get_world_size() > 1:
                dist.broadcast(_due_flag, src=0)
            _do7 = bool(int(_due_flag.item()) > 0)
        if self.is_main_process:
            logging.info(
                "[ActionForcing] 7chunk gate: step=%d sample_due=%s "
                "pending=%s ride_stashed=%s",
                int(self.step), bool(_sample_due_now), _do7,
                getattr(self, "_sample_7chunk_ride", None) is not None,
            )
        if _do7:
            # DEFERRED FIRE (phase-2 rolling fix): the 7chunk eval rollout
            # clobbers the pipeline KV cache + streaming state, which under
            # multi-roll rides would silently TRUNCATE the live ride at the
            # sample cadence (the "end_of_ride at every 15th step" bug).
            # Mark the sample as pending (all ranks, broadcast above keeps
            # it lockstep); ``_streaming_step`` fires it on the next RESET
            # step, right after the sequence is torn down — where the
            # clobber is free. Stationary configs (max_rolls_per_ride=1)
            # reset every step, so the eval still fires the same step.
            self._pending_7chunk_sample = True

        # Disarm the eval stash only now — after the (possibly separate-
        # backward) aux pass had its chance to add pred_real_lora /
        # clean_x_aux. See the harvest comment above.
        self.model._dmd_eval_stash = None

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
        # Phase-2 rolling (dmd_42f_rolling_sup_new): the 42f rolling
        # layout needs one GT chunk BEYOND the newest rolled frame (the
        # scaffold at the masked OOD slot). Keep the ride window sliced
        # at the full ``actual_cap`` but stop the ROLLING ``max_length``
        # one chunk short, so ride GT at [new_hi, new_hi + npb) always
        # exists for the last roll.
        _rolling_slack = (
            npb if bool(getattr(
                self.config, "dmd_42f_rolling_sup_new", False)) else 0
        )
        # Forward clean drift (e-framework) places clean_x up to +npb AHEAD of
        # the noisy window at frac=1, so the 42f clean slice (clean_lo+N) needs
        # npb MORE real GT beyond the rolling scaffold. Reserve it so the last
        # roll's forward-clean read stays in-bounds.
        if bool(getattr(self.config, "dmd_42f_clean_drift_enabled", False)):
            _rolling_slack += npb
        roll_cap = actual_cap - _rolling_slack
        # Reject if the ride can't fit the +npb anchor + at least one
        # valid ``generate_next_chunk`` call.
        min_new = int(getattr(self.model, "streaming_min_new_frame", npb))
        anchor_frames = int(getattr(self.model, "dmd_clean_x_anchor_frames", npb))
        if roll_cap < anchor_frames + min_new:
            return False

        prompt_embeds = ride["prompt_embeds"]
        seed_latents = ride["latents"][:, s : s + cf_dmdctx].contiguous()
        # Window covers seed + rollout: ride_*[s : s + cf + actual_cap].
        ride_lat_window = ride["latents"][:, s : s + cf_dmdctx + actual_cap].contiguous()
        ride_act_window = ride["z_actions"][:, s : s + cf_dmdctx + actual_cap].contiguous()

        # Stash a 42-frame (14-chunk) GT slice for the pred_image_7_chunk
        # eval video (7 GT-context chunks + 7 student-rolled chunks).
        # Stashed on EVERY rank (min_ride_frames guarantees >= 42 so the
        # later all-ranks sample rollout stays collective-balanced). The
        # 7-chunk context starts at the ride origin (not the motion-aware
        # ``s``) so the slice is guaranteed in-bounds. Cheap (~tens of MB)
        # and overwritten each ride.
        _need7 = 14 * npb
        if int(ride["latents"].shape[1]) >= _need7:
            self._sample_7chunk_ride = {
                "latents": ride["latents"][:, :_need7].detach().clone(),
                "z_actions": ride["z_actions"][:, :_need7].detach().clone(),
                "prompt_embeds": prompt_embeds,
            }
        else:
            self._sample_7chunk_ride = None

        self.model.setup_sequence(
            seed_latents=seed_latents,
            ride_latents_window=ride_lat_window,
            ride_actions_window=ride_act_window,
            prompt_embeds=prompt_embeds,
            max_length=int(roll_cap),
        )

        # GT match pool for the content-matched transition GAN: the FULL
        # loaded ride (latents + co-located actions, absolute frame 0..),
        # wider than ride_latents_window so the per-fake MAE matcher has a
        # rich candidate set. Moved to the rollout device once per ride
        # (~tens of MB; ride is small). Only when matching is enabled.
        if bool(getattr(self.model, "ladd_gt_transition_match", False)):
            _ss = self.model.streaming_state
            _dev = _ss["ride_latents_window"].device
            _gm_lat = ride["latents"]
            _gm_act = ride["z_actions"]
            if bool(getattr(self.config, "dmd_42f_rolling_sup_new", False)):
                # Phase-2 rolling: bound the matcher pool to the ACTIVE
                # ride window. Rolling rides are full-length (no random
                # truncation), and a whole-ride pool scales the per-fake
                # MAE matching transients with ride length (the 91-GiB
                # ceiling has no room for that). The window (<= cf +
                # streaming_max_length frames) is still hundreds of
                # chunks AND positionally tracks where the student
                # actually rolls. Stationary configs keep the legacy
                # whole-ride pool.
                _gm_lat = _gm_lat[:, s : s + cf_dmdctx + actual_cap]
                _gm_act = _gm_act[:, s : s + cf_dmdctx + actual_cap]
            _ss["gt_match_latents"] = _gm_lat.detach().to(_dev)
            _ss["gt_match_actions"] = _gm_act.detach().to(_dev)

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

    def _phase_lora_ghost_anchor(
        self, loss: torch.Tensor,
    ) -> torch.Tensor:
        """Zero-weighted touch on EVERY student phase-LoRA param.

        K+1 DDP rebuild fix: each gen step only the active adapter
        gets a REAL gradient. The dedicated ``rung_flash`` adapter's
        only real signal is the GAN adversarial loss, which is gated
        off until ``gan_disc_start_step`` — so for the first ~20 iters
        its params are NEVER marked ready in DDP's reducer. DDP defers
        its one-time bucket rebuild until every tracked param has been
        marked at least once; at the first GAN-active iter (step 21)
        rung_flash finally fires, the deferred rebuild triggers MID-
        CKPT-RECOMPUTE (the recompute's ``_pre_forward`` runs inside
        the outer backward), and the reducer dies with
        ``!unmarked_param_indices.empty() INTERNAL ASSERT``
        (reducer.cpp:2035; observed at step=21 on j5147634/5/6 and the
        j5151298+ Fix-B resubmits — both crashes exactly one step
        after gan_disc_start_step=20).

        Adding ``0.0 * sum(lora params)`` to the gen loss guarantees
        every adapter receives an (exactly-zero) grad on every
        backward from iter 1 → the rebuild completes cleanly at iter 2
        the same way it does on the no-LoRA baseline. The added
        gradient is identically zero: per-rung training signal,
        optimizer state, and rung independence are untouched
        (weight_decay=0 → idle-rung AdamW steps are pure momentum
        decay, no shrinkage, no cross-rung pollution).

        In the current no-DDP phase-LoRA mode (the student is not
        DDP-wrapped; lora grads sync manually in
        ``_all_reduce_extra_trainable_grads``), the ghost's job is to
        guarantee every lora param has a non-None grad on every iter,
        so the manual all-reduce param list is identical across ranks
        and iters — no divergence risk, no conditional sync logic.

        No-op (returns ``loss`` unchanged) when phase LoRA is off.
        """
        if not bool(
            getattr(self.config, "student_phase_lora_enabled", False)
        ):
            return loss
        gen_mod = self.model.generator.model
        if hasattr(gen_mod, "module"):
            gen_mod = gen_mod.module
        # Re-arm FIRST, then ghost over ALL lora params. The no_grad
        # dispatch sites use re_arm=False, so whichever dispatch ran
        # last (typically the Step 3.4 context_noise commit) leaves
        # only ONE adapter's params requires_grad=True via peft's
        # set_adapter side-effect. Filtering the ghost by the CURRENT
        # requires_grad would then cover a single adapter and leave
        # the other buckets unreduced — find_unused=False DDP dies
        # with "Expected to have finished reduction in the prior
        # iteration" at the next iter's first grad-on forward
        # (observed on j5169482/3/4, ~iter 1-2). Re-arming here also
        # restores the canonical all-True state before backward, which
        # both the ckpt recomputes and DDP's tracked-param set expect.
        ghost: Optional[torch.Tensor] = None
        for n, p in gen_mod.named_parameters():
            if "lora_" in n:
                p.requires_grad = True
                ghost = p.sum() if ghost is None else ghost + p.sum()
        if ghost is not None:
            loss = loss + 0.0 * ghost
        return loss

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
                if gan_active:
                    gt_window = latents[:, gen_window_start:gen_window_end]
                    gen_gan_loss, gan_logs = self._compute_r3gan_losses(
                        pred_image=pred_image,
                        gt_latents_window=gt_window,
                        current_step=int(self.step),
                    )
                    generator_loss = generator_loss + gen_gan_loss
                    merged.update(gan_logs)

                # Moment-GAN runs independently of the main GAN gate
                # (own ``moment_gan_enabled`` flag). Same idea as the
                # streaming path above.
                if self.moment_gan_enabled and self.moment_disc is not None:
                    _mgt_window = latents[
                        :, gen_window_start:gen_window_end
                    ]
                    moment_gen_loss, moment_logs = (
                        self._compute_moment_gan_losses(
                            pred_image=pred_image,
                            gt_latents_window=_mgt_window,
                            current_step=int(self.step),
                        )
                    )
                    generator_loss = generator_loss + moment_gen_loss
                    merged.update(moment_logs)

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

                generator_loss = self._phase_lora_ghost_anchor(
                    generator_loss
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

                generator_loss = self._phase_lora_ghost_anchor(
                    generator_loss
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
                random_window=bool(
                    getattr(self.config, "max_ride_frames_random", False)
                ),
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
