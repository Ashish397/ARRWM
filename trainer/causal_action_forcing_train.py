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
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from pipeline.action_forcing_training import ActionForcingTrainingPipeline
from utils.infinity_rope import install as _install_infinity_rope


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
                )

            # Always run the critic step (CF parity).
            critic_log_dict = self._fwdbwd_one_step(
                train_generator=False,
                rollout_frames=rollout_frames,
                max_total_rollout_frames=max_total_rollout_frames,
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
        if not self.action_critic_loss_active:
            return
        path = self._checkpoint_path(self.step)
        if not os.path.exists(path):
            return
        try:
            state = torch.load(path, map_location="cpu")
        except Exception as exc:
            logging.warning(
                "[ActionForcing] Could not re-open checkpoint for "
                "action_critic append: %s. Skipping.", exc,
            )
            return
        ac = self.model.action_critic
        ac_module = ac.module if isinstance(ac, DDP) else ac
        if ac_module is not None:
            state["action_critic"] = ac_module.state_dict()
        if self.critic_optimizer is not None:
            state["critic_optimizer"] = self.critic_optimizer.state_dict()
        torch.save(state, path)
        logging.info(
            "[ActionForcing] Appended action_critic + critic_optimizer "
            "state to %s", path,
        )

    def _maybe_resume(self) -> None:
        super()._maybe_resume()
        if not self.action_critic_loss_active:
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
                    "action_critic resume: %s", exc,
                )
            return
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

        # --- Multi-step critic training (independent optim loop) ---
        critic_z_loss = zero
        critic_loss_k = zero
        pred_z = None
        for _k in range(max(1, self.critic_updates_per_step)):
            self.critic_optimizer.zero_grad(set_to_none=True)
            pred_z = critic_for_update(pred_x0_detached, chunk_t, chunk_actions)
            pred_z = pred_z[:, :n_chunks]
            critic_z_loss = self._weighted_z_mse(pred_z, teacher_z_8d)
            critic_loss_k = self.action_critic_z_loss_weight * critic_z_loss
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
                gen_pred_z = critic_for_guidance(pred_x0, chunk_t, chunk_actions)
                gen_pred_z = gen_pred_z[:, :n_chunks]
                gen_z2z7 = gen_pred_z[:, :, self.action_critic_dims]
                target_z2z7 = chunk_actions
                gen_z_loss = F.mse_loss(gen_z2z7, target_z2z7.to(gen_z2z7.dtype))
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
    # Per-iter forward/backward.
    # ------------------------------------------------------------------
    def _fwdbwd_one_step(
        self,
        train_generator: bool,
        rollout_frames: int,
        max_total_rollout_frames: int,
    ) -> Optional[Dict[str, Any]]:
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
        ride = self._next_ride(rollout_frames)
        if ride is None:
            return None

        ride_len = int(ride["latents"].shape[1])
        extended_length = min(ride_len, max_total_rollout_frames)
        # Defensive: extension streams must be at least ``rollout_frames``
        # (validated in ``_next_ride`` already, but this is the final
        # gate before slicing).
        if extended_length < rollout_frames:
            raise RuntimeError(
                f"Ride too short for rollout: ride_len={ride_len}, "
                f"rollout_frames={rollout_frames}. ``_next_ride`` should "
                f"have skipped this ride."
            )

        prompt_embeds = ride["prompt_embeds"]
        latents = ride["latents"][:, :extended_length]
        actions = ride["z_actions"][:, :extended_length]

        conditional_dict, unconditional_dict = self.model.build_action_conditional(
            prompt_embeds=prompt_embeds,
            gt_actions=actions,
        )

        # ``image_or_video_shape`` is the BASELINE rollout shape — the
        # pipeline allocates noise from this and then samples extension
        # noise inline. ``clean_latent`` covers the full extended
        # window so the pipeline's MAE check (at frames
        # [rollout_frames - npb : rollout_frames] for the baseline,
        # and [rollout_frames + k*npb : rollout_frames + (k+1)*npb]
        # for the k-th extension) has GT available.
        image_or_video_shape = [
            latents.shape[0], rollout_frames, *latents.shape[2:]
        ]

        if train_generator:
            aux_active = (
                self.action_critic_loss_active
                and getattr(self, "action_teacher_enabled", False)
                and self.critic_optimizer is not None
            )
            if aux_active:
                gen_loss_dmd, gen_log_dict, aux = self.model.generator_loss(
                    image_or_video_shape=image_or_video_shape,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict,
                    clean_latent=latents,
                    initial_latent=None,
                    return_aux=True,
                )
                # ``aux["pred_image"]`` is the LAST ``num_training_frames``
                # of the BASELINE rollout (already sliced by
                # ``_run_generator``). Slice the per-frame action stream
                # to that same window so the chunk-pooled targets line
                # up frame-for-frame with the critic's per-chunk preds.
                pred_image = aux["pred_image"]
                scoring_frames = int(aux["scoring_frames"])
                gen_window_end = int(aux["rollout_frames"])
                gen_window_start = gen_window_end - scoring_frames
                actions_for_critic = actions[
                    :, gen_window_start:gen_window_end
                ][..., self.action_critic_dims].to(pred_image.dtype)
                # Per-chunk timestep tensor: rung-exit timestep
                # broadcast across all chunks. ``denoised_timestep_from``
                # is None when ``same_step_across_blocks=False`` (per-
                # block exit) — fall back to 0 in that case (the
                # ActionCritic's time conditioning is robust to the
                # specific value; the dominant signal is the action
                # condition + latent content).
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
                generator_loss = gen_loss_dmd + gen_action_loss
                generator_loss.backward()
                merged: Dict[str, Any] = {
                    "generator_loss": float(generator_loss.detach().item()),
                    "generator_dmd_loss": float(gen_loss_dmd.detach().item()),
                }
                merged.update({
                    k: float(v.detach().mean().item()) if torch.is_tensor(v) else v
                    for k, v in gen_log_dict.items()
                    if not isinstance(v, dict)
                })
                merged.update(critic_logs)
                return merged
            else:
                generator_loss, gen_log_dict = self.model.generator_loss(
                    image_or_video_shape=image_or_video_shape,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict,
                    clean_latent=latents,
                    initial_latent=None,
                )
                generator_loss.backward()
                return {
                    "generator_loss": float(generator_loss.detach().item()),
                    **{
                        k: float(v.detach().mean().item()) if torch.is_tensor(v) else v
                        for k, v in gen_log_dict.items()
                        if not isinstance(v, dict)
                    },
                }
        else:
            critic_loss, critic_log_dict = self.model.critic_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=latents,
                initial_latent=None,
            )
            critic_loss.backward()
            return {
                "critic_loss": float(critic_loss.detach().item()),
                **{
                    k: float(v.detach().mean().item()) if torch.is_tensor(v) else v
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
