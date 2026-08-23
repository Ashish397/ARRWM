"""Action-Forcing ODE distillation student model (random-timestep recipe).

The student mirrors the v14 teacher's DiT plus the full action
apparatus (``action_projection`` + ``action_token_projection`` +
``state_probe`` + ``action_critic``). Training is **full-rank** — the
teacher's rank-256 LoRA is loaded once at checkpoint load time and
immediately folded into the base weights via ``PeftModel.merge_and_unload()``
so every training step updates the entire DiT directly (no PEFT wrapper
at train time). See ``load_teacher_checkpoint``.

At every training step we do a **paired packed B=2 forward** per rank:
the clean and CF branches of the same dataset pair are stacked along
the batch dim and pushed through a single DiT call. ``generator_loss``
slices the output back into per-branch tensors for the loss heads.
There is no two-pass fallback — the packed path is the sole path, and
OOM (if it happens) surfaces loudly. See ``generator_loss``.

At every training step we:

1. Pick a per-block random timestep for each chunk from a user-configurable
   pool ``random_steps`` (default ``[0, 36, 44, 46]``). ``-1`` aliases the
   teacher_x0 snapshot and is intentionally excluded from the default
   pool — at timestep 0 the wrapper's flow-matching conversion yields
   ``pred_x0 == xt`` and the loss is zero by construction.
2. Gather the corresponding latent snapshot from the stored 7-step
   trajectory as the noisy input for each chunk.
3. Feed the truly-clean preceding 21 latent frames (``clean_x_gt``) as the
   teacher-forced attention context (``aug_t=None``).
4. Regress the student's ``pred_x0`` against the teacher's final x0
   snapshot (``trajectory[:, -1]``) — same for the clean and CF branches.
   The CF branch regresses onto ``trajectory_cf[:, -1]`` (teacher's CF
   x0) automatically by construction.

The ``_build_conditional`` routine mirrors the teacher's four-stream
action routing (``_action_modulation`` + ``_action_modulation_clean`` +
``_action_tokens`` + ``_action_tokens_clean``). Checkpoints written here
store the full DiT ``state_dict`` (not LoRA deltas) and are still
compatible with ``utils/eval_chain.py`` because the heads use the same
byte layout as the teacher's.

Critic chunk selection: chunks whose sampled pool index falls in the
top-``critic_num_chunks`` cleanest bin (``pool_idx >= K - k``). The
critic is never supervised on high-noise chunks, protecting its reader.

The motion pipeline (VAE + CoTracker + ss_vae) is called **once per
step** over the packed ``[2*B_pair, F, C, H, W]`` DiT output. The VAE
decode and ss_vae encoder both run as single batched calls over the
whole pack, but CoTracker is looped per-sample — its internal
``coords.view(B*T, N, 2)`` fails on non-contiguous strides when B>1, so
B=1 is the only shape it tolerates safely. See
``_compute_action_teacher_targets``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import peft
from peft import LoraConfig, set_peft_model_state_dict

from utils.wan_wrapper import WanDiffusionWrapper, WanVAEWrapper
from model.action_modulation import ActionModulationProjection, ActionTokenProjection
from model.action_model_patch import apply_action_patches
from model.action_critic import ActionCritic
from utils.zarr_dataset import _tanh_squash

from af_utils.schedule import (
    DEFAULT_RANDOM_STEPS,
    NUM_CHUNKS,
    SNAPSHOT_STEPS,
    resolve_denoising_step_list,
)


log = logging.getLogger(__name__)


def _collect_target_modules(model) -> list:
    """Mirror teacher's ``_collect_target_modules``: every ``nn.Linear`` inside
    a ``WanAttentionBlock`` / ``CausalWanAttentionBlock`` becomes a LoRA target.
    """
    target_modules = set()
    for module_name, module in model.named_modules():
        cls = module.__class__.__name__
        if cls in {"WanAttentionBlock", "CausalWanAttentionBlock"}:
            for full_name, sub in module.named_modules(prefix=module_name):
                if isinstance(sub, nn.Linear):
                    target_modules.add(full_name)
    return sorted(target_modules)


def _chunk_actions(z: torch.Tensor, chunk_frames: int) -> torch.Tensor:
    """``[B, F, D]`` -> ``[B, F//chunk_frames, D]`` via mean-pool per chunk."""
    B, F_, D = z.shape
    assert F_ % chunk_frames == 0
    return z.reshape(B, F_ // chunk_frames, chunk_frames, D).mean(dim=2)


class ODERegression(nn.Module):
    """Student generator + full teacher action apparatus (random-timestep)."""

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.dtype = torch.bfloat16 if bool(getattr(config, "mixed_precision", True)) else torch.float32

        # ------------------------------------------------------------------
        # Hyper-params (teacher v14 defaults)
        # ------------------------------------------------------------------
        self.num_frame_per_block: int = int(getattr(config, "num_frame_per_block", 3))
        self.context_frames: int = int(getattr(config, "context_frames", 3))
        self.num_training_frames: int = int(getattr(config, "num_training_frames", 21))
        self.n_chunks: int = self.num_training_frames // self.num_frame_per_block

        self.model_variant: str = str(getattr(config, "model_variant", "action-injection"))
        self.raw_action_dim: int = int(getattr(config, "raw_action_dim", 2))
        self.action_activation: str = str(getattr(config, "action_activation", "silu"))
        self.enable_adaln_zero: bool = bool(getattr(config, "enable_adaln_zero", True))
        acm = str(getattr(config, "action_conditioning_mode", "both"))
        self.use_adaln: bool = acm in ("adaln", "both")
        self.use_action_tokens: bool = acm in ("tokens", "both")
        self.use_action_conditioning: bool = True
        # Both conditioning paths must be live: the teacher was trained
        # with ``action_conditioning_mode="both"`` and the student inherits
        # that same 4-stream routing (``_action_modulation``,
        # ``_action_action_tokens`` for the noisy window and
        # ``_action_modulation_clean``, ``_action_token_projection`` for
        # the clean_x context). Dropping either silently starves one of
        # those streams and corrupts CF supervision.
        if acm != "both":
            raise ValueError(
                f"action_conditioning_mode must be 'both' for action-forcing "
                f"(got {acm!r}). The teacher checkpoint's action projections "
                f"and the _build_conditional routing both assume 4 streams."
            )

        self.action_critic_enabled: bool = bool(getattr(config, "action_critic_enabled", True))
        # ------------------------------------------------------------------
        # Action-forcing is *fundamentally* an action-conditioned ODE
        # distillation. Disabling any of the action pillars (critic,
        # projections, state probe, motion pipeline) would reduce this to
        # plain ODE distillation AND silently break the paired dual-forward
        # CF branch (which depends on the critic reading CF pred_x0 against
        # the CF-commanded action). We fail loud rather than quietly
        # degrade.
        # ------------------------------------------------------------------
        if not self.action_critic_enabled:
            raise ValueError(
                "config.action_critic_enabled=False is not supported for "
                "action-forcing ODE distillation. The critic is the entire "
                "mechanism by which the CF branch learns anything. If you "
                "want plain ODE distillation without action supervision, "
                "use Causal-Forcing/trainer/ode.py instead."
            )
        self.action_critic_dims: list = list(getattr(config, "action_critic_dims", [2, 7]))
        self.action_critic_z_out_dim: int = int(getattr(config, "action_critic_z_out_dim", 8))
        self.action_critic_z_loss_weight: float = float(getattr(config, "action_critic_z_loss_weight", 0.5))
        self.generator_action_z_guidance_weight: float = float(
            getattr(config, "generator_action_z_guidance_weight", 0.25),
        )
        # Dormant warmup knob: set either >0 to re-enable the teacher's ramp.
        # At the default of 0 the guidance weight is applied at full strength
        # from step 0, which matches how we distill from the warm-started v14
        # critic + LoRA seed.
        self.warmup_steps: int = int(getattr(config, "warmup_steps", 0))
        self.z_guidance_warmup_steps: int = int(getattr(config, "z_guidance_warmup_steps", 0))

        # Which chunks the action critic should supervise.
        # Default: the three cleanest noise levels.
        self.critic_num_chunks: int = int(getattr(config, "critic_num_chunks", 3))

        self.state_head_enabled: bool = bool(getattr(config, "state_head_enabled", True))
        self.state_probe_mode: bool = bool(getattr(config, "state_probe_mode", True))
        if not (self.state_head_enabled and self.state_probe_mode):
            raise ValueError(
                "state_head_enabled and state_probe_mode must both be True. "
                "The state probe is the continuous action-signal reader "
                "during distillation; disabling it silently downgrades the "
                "student's action fidelity without raising the ODE MSE."
            )
        self.state_probe_dim: int = int(getattr(config, "state_probe_dim", 256))
        self.state_probe_n_taps: int = int(getattr(config, "state_probe_n_taps", 6))
        self.state_probe_num_heads: int = int(getattr(config, "state_probe_num_heads", 8))
        self.state_head_out_dim: int = int(getattr(config, "state_head_out_dim", 8))
        self.state_head_loss_weight: float = float(getattr(config, "state_head_loss_weight", 0.1))
        self.state_head_action_dim_weight: float = float(
            getattr(config, "state_head_action_dim_weight", 3.0),
        )

        # Frozen-readout state guidance (teacher parity). At each step we
        # feed the probe's pooled hidden features through a *frozen* copy
        # of the state_probe readout and regress the 2-D action-dim
        # outputs toward the commanded z. Gradients flow through the
        # DiT/probe backbone but the readout weights are frozen (via
        # ``.detach()``), so this loss nudges the generator's hidden
        # state toward "readable" action content without letting the
        # readout collapse. The weight defaults to 0 so adding the loss
        # has no effect until the student config enables it.
        # See trainer/causal_diffusion_teacher_train.py:2038-2056 for the
        # teacher's reference implementation.
        self.state_guidance_weight: float = float(
            getattr(config, "state_guidance_weight", 0.0),
        )
        self.state_guidance_warmup_steps: int = int(
            getattr(config, "state_guidance_warmup_steps", 0),
        )
        self.state_guidance_action_scale: float = float(
            getattr(config, "state_guidance_action_scale", 1.1),
        )

        self.use_motion_pipeline: bool = bool(getattr(config, "use_motion_pipeline", True))
        if not self.use_motion_pipeline:
            raise ValueError(
                "config.use_motion_pipeline=False is not supported. The "
                "motion pipeline (VAE + CoTracker + ss_vae) produces "
                "teacher_z_8d which is the *only* supervision signal for "
                "the action critic and state probe during distillation. "
                "Without it, ``_compute_action_teacher_targets`` returns "
                "nothing and the critic/state losses become no-ops, which "
                "silently drops the entire action-forcing training signal."
            )

        # ------------------------------------------------------------------
        # 1) Build the DiT wrapper (pre-LoRA, pre-patches)
        # ------------------------------------------------------------------
        model_kwargs = dict(getattr(config, "model_kwargs", {}) or {})
        model_name = str(getattr(config, "model_name", "Wan2.1-T2V-1.3B"))

        wrapper = WanDiffusionWrapper(
            model_name=model_name,
            is_causal=True,
            **model_kwargs,
        )
        if bool(getattr(config, "gradient_checkpointing", True)):
            wrapper.model.enable_gradient_checkpointing()

        self.scheduler = wrapper.get_scheduler()
        num_train_timestep = int(getattr(config, "num_train_timestep", 1000))
        self.scheduler.set_timesteps(num_inference_steps=num_train_timestep, denoising_strength=1.0)

        wrapper.model.num_frame_per_block = self.num_frame_per_block
        wrapper.model.context_shift = self.context_frames // self.num_frame_per_block

        # ------------------------------------------------------------------
        # 2) Apply action patches + build action projections
        # ------------------------------------------------------------------
        self.action_projection: Optional[ActionModulationProjection] = None
        self.action_token_projection: Optional[ActionTokenProjection] = None
        if self.model_variant == "action-injection":
            apply_action_patches(wrapper)
            model_dim = int(getattr(wrapper.model, "dim", 2048))

            if self.use_adaln:
                self.action_projection = ActionModulationProjection(
                    action_dim=self.raw_action_dim,
                    activation=self.action_activation,
                    hidden_dim=model_dim,
                    num_frames=1,
                    zero_init=self.enable_adaln_zero,
                )
                self.action_projection.to(device)
                self.action_projection.train()

            if self.use_action_tokens:
                self.action_token_projection = ActionTokenProjection(
                    action_dim=self.raw_action_dim,
                    activation=self.action_activation,
                    hidden_dim=model_dim,
                    zero_init=self.enable_adaln_zero,
                )
                self.action_token_projection.to(device)
                self.action_token_projection.train()
                wrapper.model.action_tokens_per_frame = 1
                wrapper.adjust_seq_len_for_action_tokens(
                    num_frames=self.num_training_frames, action_per_frame=1,
                )

        # ------------------------------------------------------------------
        # VARIATIONAL RECTIFIED FLOW MATCHING (arXiv:2502.09616), adapted.
        # The GT velocity field is multi-modal: many (x0, x1) pairs cross the
        # same (x_t, t), so plain MSE regresses their MEAN and the motion comes
        # out direction-averaged. A per-sample latent z says WHICH mode this
        # sample takes, making v = v(x_t, t, z) so no averaging is needed.
        # Our prior is CONDITIONAL on (x0, x_t, t, a) -- everything inference
        # has -- while only the posterior additionally sees x1 (the teacher's
        # clean chunk). See af_model/vrfm.py.
        # ------------------------------------------------------------------
        self.ode_vrfm = bool(getattr(config, "ode_vrfm", False))
        self.ode_vrfm_beta = float(getattr(config, "ode_vrfm_beta", 1e-3))
        self.vrfm = None
        self.z_modulation = None
        if self.ode_vrfm:
            from af_model.vrfm import VRFMLatent, ZModulation
            if not self.use_adaln:
                raise RuntimeError(
                    "ode_vrfm injects z through the AdaLN modulation stream "
                    "(the same tensor _action_modulation feeds), so it needs "
                    "enable_adaln_zero/use_adaln. Got use_adaln=False.")
            _zdim = int(getattr(config, "ode_vrfm_zdim", 128))
            _lc = int(getattr(wrapper.model, "in_dim", 16))
            _md = int(getattr(wrapper.model, "dim", 1536))
            self.vrfm = VRFMLatent(
                latent_channels=_lc, z_dim=_zdim,
                hidden=int(getattr(config, "ode_vrfm_hidden", 128)),
                action_dim=self.raw_action_dim).to(device)
            self.z_modulation = ZModulation(
                z_dim=_zdim, hidden_dim=_md).to(device)
            self.vrfm.train()
            self.z_modulation.train()
            log.info(
                "VRFM enabled: z_dim=%d beta=%.2e latent_ch=%d model_dim=%d "
                "(%.2fM params)", _zdim, self.ode_vrfm_beta, _lc, _md,
                (sum(p.numel() for p in self.vrfm.parameters())
                 + sum(p.numel() for p in self.z_modulation.parameters())) / 1e6)

        # ------------------------------------------------------------------
        # 3) Apply a *transient* LoRA wrapper (post action patches, pre
        #    state probe) so we can load the teacher's rank-256 adapter
        #    state in ``load_teacher_checkpoint``. The adapter is folded
        #    into the base weights via ``merge_and_unload()`` immediately
        #    after the state load; from that point on the model is
        #    full-rank and has no PEFT structure.
        # ------------------------------------------------------------------
        lora_cfg = getattr(config, "teacher_lora", None)
        if lora_cfg is None:
            lora_cfg = getattr(config, "adapter", None)  # back-compat
        if lora_cfg is None:
            raise ValueError(
                "config.teacher_lora (or legacy config.adapter) is required — "
                "we need it to size the PEFT wrapper when loading the teacher "
                "adapter weights prior to merge_and_unload()."
            )
        rank = int(lora_cfg.get("rank", 256)) if isinstance(lora_cfg, dict) else int(getattr(lora_cfg, "rank", 256))
        alpha = float(lora_cfg.get("alpha", rank)) if isinstance(lora_cfg, dict) else float(getattr(lora_cfg, "alpha", rank))
        dropout = float(lora_cfg.get("dropout", 0.0)) if isinstance(lora_cfg, dict) else float(getattr(lora_cfg, "dropout", 0.0))
        target_modules = _collect_target_modules(wrapper.model) or ["q", "k", "v", "o"]
        peft_cfg = LoraConfig(
            r=rank, lora_alpha=alpha, lora_dropout=dropout,
            target_modules=target_modules, bias="none",
        )
        wrapper.model = peft.get_peft_model(wrapper.model, peft_cfg)
        self.lora_rank = rank
        self.lora_alpha = alpha
        log.info(
            "ODERegression: LoRA applied rank=%d alpha=%.1f targets=%d modules",
            rank, alpha, len(target_modules),
        )

        # ------------------------------------------------------------------
        # 4) Attach cross-attention state probe (post-LoRA)
        # ------------------------------------------------------------------
        self._state_head_built = False
        if self.state_head_enabled and self.state_probe_mode:
            wrapper.adding_state_probe_branch(
                n_chunks=self.n_chunks,
                z_out_dim=self.state_head_out_dim,
                dim=int(getattr(wrapper.model, "dim", 2048)),
                probe_dim=self.state_probe_dim,
                num_heads=self.state_probe_num_heads,
                n_taps=self.state_probe_n_taps,
                num_frame_per_block=self.num_frame_per_block,
            )
            self._state_head_built = True

        wrapper.to(device)
        wrapper.train()
        self.generator = wrapper

        # ------------------------------------------------------------------
        # 5) Build the action critic
        # ------------------------------------------------------------------
        self.action_critic: Optional[ActionCritic] = None
        if self.action_critic_enabled:
            action_dim = len(self.action_critic_dims)
            self.action_critic = ActionCritic(
                latent_channels=16,
                action_dim=action_dim,
                z_out_dim=self.action_critic_z_out_dim,
                base_channels=int(getattr(config, "action_critic_base_channels", 128)),
                num_res_blocks=int(getattr(config, "action_critic_num_blocks", 4)),
                chunk_frames=self.num_frame_per_block,
            ).to(device)
            self.action_critic.train()

        # ------------------------------------------------------------------
        # 6) Resolve denoising_step_list from the user's ``random_steps`` pool
        # ------------------------------------------------------------------
        self.ode_chunked_supervision = bool(
            getattr(config, "ode_chunked_supervision",
                    getattr(config, "chunked_lmdb", False))
        )
        snapshot_steps = list(getattr(config, "snapshot_steps", SNAPSHOT_STEPS))
        random_steps = list(getattr(config, "random_steps", DEFAULT_RANDOM_STEPS))
        if len(random_steps) == 0:
            raise ValueError("random_steps must contain at least one entry")
        self.random_steps: List[int] = [int(s) for s in random_steps]
        try:
            usable_idx = [snapshot_steps.index(s) for s in self.random_steps]
        except ValueError as exc:
            raise ValueError(
                f"random_steps={self.random_steps} must be a subset of "
                f"snapshot_steps={snapshot_steps}"
            ) from exc
        self.usable_stored_indices: List[int] = list(usable_idx)

        ds_list = resolve_denoising_step_list(
            usable_stored_indices=usable_idx,
            num_inference_steps=int(getattr(config, "eval_inference_steps", 48)),
            shift=float(model_kwargs.get("timestep_shift", 5.0)),
            snapshot_steps=snapshot_steps,
        )
        self.register_buffer(
            "denoising_step_list",
            ds_list.to(torch.float32).to(device),
            persistent=False,
        )
        log.info(
            "ODERegression[random]: random_steps=%s (snap_idx=%s) timesteps=%s",
            self.random_steps, usable_idx,
            [round(float(x), 2) for x in self.denoising_step_list.tolist()],
        )

        # CD rung list = ALL saved snapshot rungs (the dual-CD steps between
        # adjacent saved rungs, independent of the random training subset).
        # The student-CD/teacher-CD sample an adjacent (n, n+1) pair from
        # this list so they cover the checkpoint's full recorded path.
        cd_list = resolve_denoising_step_list(
            usable_stored_indices=list(range(len(snapshot_steps))),
            num_inference_steps=int(getattr(config, "eval_inference_steps", 48)),
            shift=float(model_kwargs.get("timestep_shift", 5.0)),
            snapshot_steps=snapshot_steps,
        )
        self.register_buffer(
            "cd_rung_list", cd_list.to(torch.float32).to(device), persistent=False,
        )
        log.info(
            "ODERegression[CD]: cd_rung_list (all %d saved rungs) timesteps=%s",
            len(snapshot_steps),
            [round(float(x), 2) for x in self.cd_rung_list.tolist()],
        )

        # ------------------------------------------------------------------
        # 7) Frozen motion pipeline (VAE + co-tracker + ss_vae)
        # ------------------------------------------------------------------
        self._motion_pipeline_ready = False
        # Lazily built on first call to avoid download races before DDP
        # rendezvous. See ``ensure_motion_pipeline``.

        # ------------------------------------------------------------------
        # 8) Load all heads + LoRA from the v14 teacher checkpoint
        # ------------------------------------------------------------------
        self.loaded_from_step: int = -1
        ckpt_path = getattr(config, "generator_ckpt", None)
        if ckpt_path:
            self.load_teacher_checkpoint(str(ckpt_path))

        # ------------------------------------------------------------------
        # 9) Online CD losses (ODE-F dual-CD). Two consistency regularizers
        #    riding on the GT-anchored ODE loss (collapse-safe):
        #      * teacher-CD: a FROZEN-14d teacher takes the Euler step
        #        x_n -> x_{n+1} on the 4-step schedule; the live student's
        #        x0 at x_n must match the EMA-student's x0 at x_{n+1}.
        #      * student-CD: same, but the STUDENT's own x0 drives the step
        #        (self-consistency: "denoise from rung n == rung n+1").
        #    The frozen-14d teacher == a deepcopy of the student taken at
        #    the FIRST CD call (the student is initialized AS merged-14d by
        #    load_teacher_checkpoint above, before any optimizer step). The
        #    EMA-student is a second deepcopy, EMA-updated in place. Both
        #    are held OFF the nn.Module registry (object.__setattr__) so
        #    they never enter the optimizer / DDP wrap / checkpoint. Default
        #    OFF -> byte-identical. See [[project_ode_distill_dual_cd]].
        self.cd_teacher_loss_enabled = bool(
            getattr(config, "cd_teacher_loss_enabled", False))
        self.cd_student_loss_enabled = bool(
            getattr(config, "cd_student_loss_enabled", False))
        self.cd_teacher_loss_weight = float(
            getattr(config, "cd_teacher_loss_weight", 0.5))
        self.cd_student_loss_weight = float(
            getattr(config, "cd_student_loss_weight", 0.25))
        self.cd_loss_warmup_steps = int(
            getattr(config, "cd_loss_warmup_steps", 100))
        self.cd_ema_decay = float(getattr(config, "cd_ema_decay", 0.99))
        if not (0.0 <= self.cd_ema_decay <= 1.0):
            raise ValueError(
                f"cd_ema_decay must be in [0, 1]; got {self.cd_ema_decay}")
        # When False, the CD *target* readout uses the LIVE student (stop-grad
        # via the no_grad block in _compute_cd_losses) instead of an EMA copy.
        # The stop-grad is what prevents collapse; the EMA is only an extra
        # smoother. For teacher-CD (target input anchored by the frozen-14d
        # teacher's step + the ODE regression anchor) the EMA is largely
        # redundant, so this saves a resident DiT copy (~5GB) + its update.
        self.cd_ema_enabled = bool(getattr(config, "cd_ema_enabled", True))
        # --- On-policy teacher supervision (chunked mode) ---------------
        # The student's OWN target-block x0 prediction is re-noised to a
        # mid rung and handed to the FROZEN teacher (same forward, same
        # pinned committed context); the teacher's one-step x0 correction
        # of the student's content becomes an extra regression target.
        # Differs from ODE regression (pregenerated teacher rungs) and
        # dual-CD (noising anchored on the TEACHER's x0): here the anchor
        # is the student's x0, so gradients target the student's actual
        # error modes. At step 0 student == teacher, so the loss starts
        # ~0 and grows only as generations drift — self-calibrating, no
        # warmup needed. Default OFF -> byte-identical.
        self.ode_teachersup_enabled = bool(
            getattr(config, "ode_teachersup_enabled", False))
        self.ode_teachersup_weight = float(
            getattr(config, "ode_teachersup_weight", 0.5))
        self.ode_teachersup_rungs = [float(v) for v in getattr(
            config, "ode_teachersup_rungs",
            [625.0, 357.142857, 208.333333])]
        # steps=1: single teacher x0 readout (a conditional MEAN — measured
        # to halve stat drift but blur actions, see flip2ts A/B). steps>1:
        # the teacher WALKS the real 20-step shift-5 grid tail below the
        # renoise rung (Euler steps via _cd_partial_denoise), so the target
        # is an actual teacher SAMPLE — sharp — still anchored on the
        # student's content. steps caps the number of teacher forwards.
        self.ode_teachersup_steps = int(
            getattr(config, "ode_teachersup_steps", 1))
        # Arm-D structural change: replace the committed-x0 regression
        # target on rung frames with the progressive-distillation target
        # implied by the teacher's stored NEXT trajectory state (see
        # _nextrung_targets). Default OFF -> byte-identical.
        self.ode_nextrung_targets = bool(
            getattr(config, "ode_nextrung_targets", False))
        # Moment-preservation loss (root fix for few-step moment
        # contraction, 2026-08-03 ledger): pointwise MSE is minimised by
        # the conditional mean — correct per-channel mean, strictly
        # UNDER-DISPERSED per-channel std (measured: committed std -40%
        # over 6 chunks at 4 rungs vs -4% dense, same weights). This term
        # matches the prediction's per-channel (mu, sigma) on the target
        # block to the TARGET state's own moments, making the learned
        # jump map moment-preserving. Default 0 -> byte-identical.
        self.ode_moment_loss_weight = float(
            getattr(config, "ode_moment_loss_weight", 0.0))
        # Action-plane separation loss (user prescription 2026-08-05):
        # match the DISTANCE between the clean/cf branch predictions to
        # the distance between their committed teacher samples — fights
        # the collapse of action branches toward the no-op/conditional
        # mean without unbounded repulsion. Default 0 -> byte-identical.
        self.ode_sep_loss_weight = float(
            getattr(config, "ode_sep_loss_weight", 0.0))
        # Action-effect vector matching (user-directed 2026-08-05): the
        # clean/cf pair shares context AND noise seed, differing only in
        # action — match the student's action-induced displacement VECTOR
        # (p_c - p_cf) to the teacher's (t_c - t_cf), relative to its
        # norm. Trains the action-effect map itself (direction+magnitude
        # of divergence at fixed noise). Default 0 -> byte-identical.
        self.ode_actdelta_loss_weight = float(
            getattr(config, "ode_actdelta_loss_weight", 0.0))
        # Gaussian-signature repulsor (user-directed 2026-08-05): the
        # noise/attractor signature in latent moment space is (mu=0,
        # sigma=1) per channel. One-sided hinge: penalize predictions
        # whose signature distance D = ||mu||^2 + ||sigma-1||^2 falls
        # BELOW their own target's (i.e., closer to the gaussian than
        # the data is); being farther is free. Default 0 -> identical.
        self.ode_noiserep_loss_weight = float(
            getattr(config, "ode_noiserep_loss_weight", 0.0))
        # EMD repulsor family (user-directed 2026-08-06): d = per-channel
        # 1-D earth-mover (W2) distance between the predicted block's
        # empirical value distribution and N(0,1), exact via sorted-value
        # vs gaussian-quantile matching (captures full-shape deviation,
        # not just the first two moments). v1: + w / d^2 (always-on
        # inverse-square, same law as ode_noiserep). v2: + w / (d_cur -
        # d_prevchunk) — penalizes chunk-over-chunk CONTRACTION toward
        # the gaussian; delta clamped at ode_emdrep_delta_eps (1/x is
        # singular at 0 and REWARDS contraction when negative — clamp
        # makes contraction pay the max penalty instead). Default 0 ->
        # byte-identical.
        self.ode_emdrep_loss_weight = float(
            getattr(config, "ode_emdrep_loss_weight", 0.0))
        self.ode_emdrep_delta_weight = float(
            getattr(config, "ode_emdrep_delta_weight", 0.0))
        self.ode_emdrep_delta_eps = float(
            getattr(config, "ode_emdrep_delta_eps", 0.05))
        # Commit-clock restriction (user-directed 2026-08-07): d is a
        # property of the COMMITTED chunk — it only exists at the last
        # rung. With commit_only=true the EMD terms are applied ONLY on
        # samples drawn at the final rung (t=208.33, whose pred_x0 IS the
        # chunk the 4-rung sampler commits), so the high-rung flow map
        # (conditional-mean regime) is never fought by a chunk-clock
        # statistical constraint.
        self.ode_emdrep_commit_only = bool(
            getattr(config, "ode_emdrep_commit_only", False))
        # Zero-init learned transport head (AdaLN-zero analog, user-
        # directed 2026-08-07): per-channel affine correction
        # y = x*(1+s)+b with s,b initialized to EXACT ZERO — at step 0
        # nothing is perturbed. Trained ONLY by the commit-clock EMD
        # objectives on the DETACHED committed prediction (the flow map
        # never receives repulsor gradient); the correction grows from
        # zero as the EMD gradient teaches it. At serve the head is
        # applied to each committed chunk (ODE_EMDHEAD=1) — a LEARNED
        # serve-time EMD transport.
        self.ode_emdhead_weight = float(
            getattr(config, "ode_emdhead_weight", 0.0))
        self.ode_emdhead_delta_weight = float(
            getattr(config, "ode_emdhead_delta_weight", 0.0))
        # Head objective (lit-review 2026-08-07): "invd" = original
        # 1/d^2 + 1/dd EMD pressures; "deadband" = VICReg-style
        # satisfiable constraint — per-channel log-ratio of the corrected
        # chunk's std vs the PREVIOUS committed chunk, squared hinge below
        # 0 (contraction) and above +band_hi (runaway), ZERO gradient
        # inside; mean tracked the same way; + identity-minimality term
        # so the correction is minimum-necessary. Target = zero per-chunk
        # decay (beats the teacher's k~0.986, per user requirement).
        self.ode_emdhead_objective = str(
            getattr(config, "ode_emdhead_objective", "invd"))
        self.ode_emdhead_band_hi = float(
            getattr(config, "ode_emdhead_band_hi", 0.05))
        self.ode_emdhead_mu_band = float(
            getattr(config, "ode_emdhead_mu_band", 0.03))
        self.ode_emdhead_id_weight = float(
            getattr(config, "ode_emdhead_id_weight", 0.1))
        # AXIS 2 (lit-review 2026-08-08): BOUNDED gaussian exclusion zone,
        # replacing the divergent 1/d^2 repulsor. R = softplus((r0-d)/tau):
        # a real force while the committed chunk is nearer the gaussian
        # signature than r0, and ~0 beyond it, so there is no incentive
        # for unbounded expansion (the emd1/emd2/emdc failure).
        # MEASURED 2026-08-09 on real dir8n committed chunks: their
        # per-channel sigma is ~0.5, so they sit at distance ~2.2 from the
        # N(0,1) signature, and COLLAPSE MOVES THEM FURTHER AWAY (k=1.0 ->
        # d 2.30, k=0.0 -> d 4.00). A "repel from the gaussian" term is
        # therefore both inert here (0.2% of max force at r0=1.0) and
        # WRONGLY SIGNED — it rewards contraction. The collapse mode is not
        # near N(0,1) at all. So the barrier is referenced to the TEACHER's
        # own committed sigma instead: a satisfiable deadband that fires
        # only when the student contracts below the teacher (or runs away
        # above it), which is the failure actually being fought.
        self.ode_gexcl_weight = float(getattr(config, "ode_gexcl_weight", 0.0))
        self.ode_gexcl_band = float(getattr(config, "ode_gexcl_band", 0.05))
        # Gaussian repulsor (N(0,1) reference, inverse-square, no deadband).
        # Separate from ode_gexcl_*, which is the teacher-referenced barrier.
        self.ode_grep_weight = float(getattr(config, "ode_grep_weight", 0.0))
        self.ode_grep_components = str(
            getattr(config, "ode_grep_components", "both"))
        # DIRECTIONAL repulsor (v2 of the mu-repulsor, 2026-08-15): distance
        # measured as the PROJECTION onto the teacher's per-chunk mean
        # direction, not the isotropic distance from mu=0. The isotropic form
        # was measured gaming itself: it held Sum mu^2 up (4.2->5.2) while
        # pushing means in wrong-sign directions (error 1.68 vs roll's 1.09,
        # the purple tinge) because ANY escape from the origin satisfied it.
        # Orthogonal/wrong-sign offsets buy ZERO relief from the projection.
        self.ode_grepdir_weight = float(
            getattr(config, "ode_grepdir_weight", 0.0))
        self.ode_grepdir_eps = float(
            getattr(config, "ode_grepdir_eps", 0.5))
        # KLTS — KL Temperature Scaling (user design 2026-08-15): a learned
        # per-(AR-chunk, rung) temperature on the denoising CHORD,
        #   pred_x0' = x_t + tau[c, i] * (pred_x0 - x_t),  tau = 1 + theta.
        # The step direction is the model's OWN prediction for this content —
        # no external reference to game (the repulsor's purple failure is
        # impossible by construction). Zero-init theta => tau=1 => byte-
        # identical model at step 0 (AdaLN-zero philosophy, emd-head
        # precedent). 8 chunk rows x 4 rungs; measured contraction law
        # (k~0.917/chunk) predicts learned tau ~1.05-1.15 late — a
        # falsifiable mechanism check. Trains in the emd-head optimizer
        # group (high lr, no decay: base 2e-6 would freeze 32 scalars at
        # init — measured on jobs 5941221/5946996).
        self.ode_klts = bool(getattr(config, "ode_klts", False))
        if self.ode_klts:
            self.klts_theta = nn.Parameter(
                torch.zeros(8, 4, device=self.device))
        # ONLINE ATTRACTOR TRACKING + REPULSION (af_model/attractor_tracker.py):
        # estimate the per-direction collapse attractor in stat space from the
        # training rollouts, repel from a target-network-frozen copy, gated to
        # auto-shutoff when the attractor merges with the data manifold.
        self.ode_attractor_enabled = bool(
            getattr(config, "ode_attractor_enabled", False))
        if self.ode_attractor_enabled:
            if self.ode_grep_weight > 0.0 or self.ode_gexcl_weight > 0.0:
                raise RuntimeError(
                    "ode_attractor_enabled shares the per-chunk slot with "
                    "ode_grep/ode_gexcl; enable exactly one of the three.")
            from af_model.attractor_tracker import AttractorTracker
            # .to(device): ODERegression never moves itself wholesale (each
            # submodule is moved individually), so without this the tracker's
            # buffers stay on CPU -> device-mismatch crash at the first
            # observe(), and sync() would hand NCCL a CPU tensor (review C1).
            self.attractor = AttractorTracker(
                weight=float(getattr(config, "ode_attractor_weight", 0.1)),
                freeze_k=int(getattr(config, "ode_attractor_freeze_k", 200)),
                warmup=int(getattr(config, "ode_attractor_warmup", 300)),
                use_sigma=bool(getattr(config, "ode_attractor_sigma", False)),
            ).to(self.device)
        else:
            self.attractor = None
        self.ode_gexcl_r0 = float(getattr(config, "ode_gexcl_r0", 1.0))
        self.ode_gexcl_tau = float(getattr(config, "ode_gexcl_tau", 0.25))
        # AXIS 3: distribution matching instead of pure mean matching.
        # ENERGY DISTANCE between the student's and teacher's committed
        # chunks in a feature space phi(z) (moments + quantiles + temporal
        # deltas -- NOT channel-Gaussian moments alone, which is what made
        # the KL arms action-blind):
        #     D_E = 2 E||X-Y|| - E||X-X'|| - E||Y-Y'||
        # The -E||Y-Y'|| term is a BOUNDED anti-collapse repulsion whose
        # optimum is the teacher distribution rather than infinite
        # separation. Samples come from the 8-action fan batch, gathered
        # across ranks (features are tiny), so the set spans the whole fan
        # and producing one answer for every action is directly penalized.
        self.ode_edist_weight = float(getattr(config, "ode_edist_weight", 0.0))
        # PROPER energy score (2026-08-16): two noise branches per chunk,
        # strictly proper scoring rule vs the single teacher realization.
        # Exclusive with the other slot losses (enforced in rollout_ode_loss).
        self.ode_es = bool(getattr(config, "ode_es", False))
        # 1.0 = stop-grad doubling: matches the symmetric m=2 energy score's
        # expected repulsion gradient (review 2026-08-16; 0.5 is improper).
        self.ode_es_repw = float(getattr(config, "ode_es_repw", 1.0))
        if self.ode_es and str(getattr(config, "ode_loss_type", "mse")) != "mse":
            # The es ladder replaces the pointwise loss entirely; a configured
            # kl_local would be silently ignored — fail fast instead.
            raise RuntimeError("ode_es replaces the rung loss; set ode_loss_type=mse")
        self.ode_edist_commit_only = bool(
            getattr(config, "ode_edist_commit_only", True))
        # KV-CACHE ROLLOUT STAGE (af_model/ode_rollout.py). The teacher made
        # every target with a cache-driven AR rollout and NO clean_x; the
        # teacher-forced window path above trains a mechanism the teacher never
        # used and the student never serves. With ode_rollout=true the student
        # is rolled out through the teacher's own cache path at its 4 rungs and
        # EVERY generated frame is supervised.
        self.ode_rollout = bool(getattr(config, "ode_rollout", False))
        # The PROBE builds this class from the EVAL config, which has no
        # ode_rollout key, while ODE_ROLLOUT is still exported by the arm.
        # Only enforce the guard for the TRAINING model.
        if (not os.environ.get("ODE_EVAL_BUILD")
                and self.ode_rollout != bool(os.environ.get("ODE_ROLLOUT"))):
            raise RuntimeError(
                "ode_rollout and the ODE_ROLLOUT env var must agree: the\n"
                "MODEL switches on the config key but the DATASET switches\n"
                "on the env var. Mismatch gives a bare KeyError "
                "('seed_lat' or 'trajectory_clean'). "
                f"config={self.ode_rollout} env={bool(os.environ.get('ODE_ROLLOUT'))}")
        self.ode_rollout_commit = str(getattr(config, "ode_rollout_commit", "teacher"))
        self.ode_rollout_commit_p = float(getattr(config, "ode_rollout_commit_p", 0.0))
        # ACTION-ERROR LOSS WEIGHTING (user-directed): measure the REALISED
        # action of the rolled-out chain with the teacher's own CoTracker->PCA
        # pipeline and upweight the chain's loss in proportion to how far it is
        # from the COMMANDED action. Applies to whichever pointwise base is in
        # use (mse / kl_local) because it scales the accumulated total.
        # Chain-level, not per-chunk: CoTracker needs >=12 frames and a chunk is 3.
        self.ode_actw_enabled = bool(getattr(config, "ode_actw_enabled", False))
        self.ode_actw_alpha = float(getattr(config, "ode_actw_alpha", 1.0))
        # 0.25 was a GUESS made before calibration. utils/calibrate_actw.py
        # then measured 360 TEACHER chains -- i.e. the achievable floor --
        # and found median err 0.914 (analysis/eval_final/flow_viz/
        # actw_calibration.json). With err_ref=0.25 the ratio pinned at
        # max_ratio for ~19% of chains and averaged 4.2x, making the
        # "weight" a flat LR multiplier carrying no per-chain signal --
        # the exact failure the weighting was meant to avoid.
        self.ode_actw_ref = float(getattr(config, "ode_actw_ref", 0.914))
        self.ode_actw_max = float(getattr(config, "ode_actw_max", 4.0))
        self._actw_cotracker = None
        # KL variant: weight by DISPERSION error as well as action error.
        self.ode_varw_enabled = bool(getattr(config, "ode_varw_enabled", False))
        self.ode_varw_alpha = float(getattr(config, "ode_varw_alpha", 1.0))
        self.ode_varw_ref = float(getattr(config, "ode_varw_ref", 0.10))
        if self.ode_emdhead_weight > 0.0 or self.ode_emdhead_delta_weight > 0.0:
            self.emd_head_scale = nn.Parameter(torch.zeros(16))
            self.emd_head_shift = nn.Parameter(torch.zeros(16))
        self._emd_quantiles = None
        # Action-CFG training (user-directed 2026-08-05): with prob p per
        # sample, replace the noisy-branch action stream with the NULL
        # action (zeros) so the model learns an unconditional-action
        # branch -> enables classifier-free guidance on actions at serve
        # (ODE_ACTION_CFG). Default 0 -> byte-identical.
        self.ode_action_cfg_dropout = float(
            getattr(config, "ode_action_cfg_dropout", 0.0))
        # Off-registry placeholders (object.__setattr__ keeps them out of
        # ._modules so .parameters()/.state_dict()/DDP never see them).
        object.__setattr__(self, "_cd_teacher", None)
        object.__setattr__(self, "_cd_ema", None)
        # Capture the FROZEN-14d teacher NOW — right after load_teacher_checkpoint
        # (above), while the generator == merged-14d and on device — NOT lazily
        # at the first CD step. A lazy capture would, on a --requeue/resume,
        # deepcopy the RESUMED (drifted) weights instead of 14d, silently
        # corrupting the teacher-CD anchor (it is supposed to be a FIXED 14d
        # reference). The trainer's _try_resume overwrites the LIVE generator
        # but never this separate frozen copy, so it stays 14d across restarts.
        # (EMA-student stays lazy: it is meant to track the live student, so
        # seeding it from the resumed student on restart is acceptable.)
        if self.cd_teacher_loss_enabled or self.ode_teachersup_enabled:
            import copy as _copy
            # self.generator is still the BARE wrapper here (the trainer wraps
            # it in DDP only after model construction), so deepcopy it directly.
            # (Shared by teacher-CD and on-policy teacher supervision.)
            _tea = _copy.deepcopy(self.generator)
            _tea.requires_grad_(False)
            _tea.eval()
            object.__setattr__(self, "_cd_teacher", _tea)
        if (self.cd_teacher_loss_enabled or self.cd_student_loss_enabled):
            logging.info(
                "[ODE-F] dual-CD ENABLED: teacher=%s(w=%.3f) student=%s(w=%.3f) "
                "warmup=%d ema_decay=%.4f (frozen-14d teacher + EMA-student, "
                "resident, off-registry).",
                self.cd_teacher_loss_enabled, self.cd_teacher_loss_weight,
                self.cd_student_loss_enabled, self.cd_student_loss_weight,
                self.cd_loss_warmup_steps, self.cd_ema_decay,
            )

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def load_teacher_checkpoint(self, ckpt_path: str) -> None:
        """Load teacher LoRA + action heads + state probe + action critic, then
        *fold* the LoRA weights into the base DiT so training happens full-rank.

        Matches the save layout used by ``trainer/causal_diffusion_teacher_train.py``:
          ``{'lora': ..., 'action_projection': ..., 'action_token_projection': ...,
             'state_probe': ..., 'action_critic': ..., 'step': ..., 'config_name': ...}``

        Post-load invariants:
          * The LoRA wrapper is fully stripped (no ``LoraLayer`` anywhere in
            ``self.generator.model``).
          * Every parameter of the merged DiT has ``requires_grad=True``.
          * The teacher-LoRA deltas have been folded into the base weights via
            ``PeftModel.merge_and_unload()`` (which computes
            ``W += scaling * B @ A`` in place on each targeted ``Linear``).

        Raises:
          FileNotFoundError: checkpoint path doesn't exist.
          RuntimeError:       checkpoint is missing the ``lora`` key — without
                              it we'd silently be training a fresh Wan2.1 base
                              with no teacher warmstart.
        """
        p = Path(ckpt_path)
        if not p.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {p}")
        ck = torch.load(p, map_location="cpu", weights_only=False)

        if "lora" not in ck:
            raise RuntimeError(
                f"Teacher checkpoint {p} is missing the 'lora' key. "
                "The action-forcing ODE distill loads the teacher's LoRA "
                "adapter and folds it into the base DiT via merge_and_unload(). "
                "Without it we'd train a fresh Wan2.1-T2V-1.3B with no warm "
                "start, which is never what we want at this stage."
            )
        set_peft_model_state_dict(self.generator.model, ck["lora"])
        log.info("Loaded LoRA adapters from %s", p.name)

        # ---------------------------------------------------------------
        # Fold the teacher LoRA into the base weights and strip the PEFT
        # wrapper. After this the model is a plain ``CausalWanModel`` with
        # no LoRA structure — the teacher's rank-256 delta has been baked
        # into the base ``Linear.weight`` buffers and the A/B matrices no
        # longer exist. Flip the whole DiT to trainable.
        # ---------------------------------------------------------------
        self.generator.model = self.generator.model.merge_and_unload()
        from peft.tuners.lora import LoraLayer  # local import to avoid top-level peft.lora cost
        residual_lora = [
            n for n, m in self.generator.model.named_modules()
            if isinstance(m, LoraLayer)
        ]
        if residual_lora:
            raise RuntimeError(
                "merge_and_unload() failed to strip LoRA layers; residuals: "
                f"{residual_lora[:5]}{' …' if len(residual_lora) > 5 else ''}"
            )
        self.generator.model.requires_grad_(True)
        n_trainable = sum(
            p.numel() for p in self.generator.model.parameters() if p.requires_grad
        )
        log.info(
            "Merged LoRA rank=%d into base DiT; full-rank training enabled "
            "(%.2fM trainable DiT params).",
            self.lora_rank, n_trainable / 1e6,
        )

        if self.action_projection is not None and "action_projection" in ck:
            self.action_projection.load_state_dict(ck["action_projection"])
            log.info("Loaded action_projection")
        if self.action_token_projection is not None and "action_token_projection" in ck:
            self.action_token_projection.load_state_dict(ck["action_token_projection"])
            log.info("Loaded action_token_projection")

        if self._state_head_built and "state_probe" in ck and hasattr(self.generator, "_state_probe"):
            missing, unexpected = self.generator._state_probe.load_state_dict(
                ck["state_probe"], strict=False,
            )
            if missing or unexpected:
                log.warning(
                    "state_probe partial load: %d missing, %d unexpected",
                    len(missing), len(unexpected),
                )
            else:
                log.info("Loaded state_probe")

        if self.action_critic is not None and "action_critic" in ck:
            # Shape-filtered load: the teacher checkpoint may carry a critic
            # whose ``action_embed`` input width differs from ours (e.g. the
            # v14d ``critic8`` has action_dim=8 because it conditions on the
            # full 8-dim action-z, whereas the ODE student commands only the
            # 2-dim [z2,z7] stream -> action_dim=2). ``load_state_dict`` with
            # strict=False still *raises* on a size mismatch, so we drop the
            # shape-incompatible tensors here. Everything action-dim-INDEPENDENT
            # -- the visual trunk (stem/down1/down2/trunk), time_embed and
            # z_head -- is identical shape and warm-starts from the teacher;
            # only ``action_embed.0.weight`` (the action input projection)
            # re-initialises fresh and is learned during distillation.
            ck_critic = ck["action_critic"]
            own = self.action_critic.state_dict()
            filtered = {
                k: v for k, v in ck_critic.items()
                if k in own and own[k].shape == v.shape
            }
            skipped = [
                k for k, v in ck_critic.items()
                if not (k in own and own[k].shape == v.shape)
            ]
            missing, unexpected = self.action_critic.load_state_dict(
                filtered, strict=False,
            )
            if skipped:
                log.warning(
                    "action_critic: skipped %d shape-mismatched key(s) %s "
                    "(re-init fresh -- expected when teacher critic action_dim "
                    "!= student's); loaded %d/%d tensors.",
                    len(skipped), skipped[:4], len(filtered), len(ck_critic),
                )
            if missing or unexpected:
                log.warning(
                    "action_critic partial load: %d missing, %d unexpected",
                    len(missing), len(unexpected),
                )
            if not skipped and not missing and not unexpected:
                log.info("Loaded action_critic")

        self.loaded_from_step = int(ck.get("step", -1))
        log.info("Teacher checkpoint loaded (step=%d).", self.loaded_from_step)

    # ------------------------------------------------------------------
    # Frozen motion pipeline (VAE + co-tracker + ss_vae)
    # ------------------------------------------------------------------

    def ensure_motion_pipeline(self, distributed: bool = True) -> None:
        """Lazily build the frozen evaluator modules.

        Call this once after distributed rendezvous so the rank-0 cotracker
        download can populate ``torch.hub`` cache first, and other ranks
        reuse it.
        """
        if self._motion_pipeline_ready or not self.use_motion_pipeline:
            return

        self._frozen_vae = WanVAEWrapper()
        self._frozen_vae.eval()
        self._frozen_vae.requires_grad_(False)
        self._frozen_vae.to(self.device)

        import torch.distributed as dist  # noqa: F401 (optional import guard)

        def _load_cotracker():
            """torch.hub.load with a pinned ref + retries.

            Without an explicit ref, torch.hub queries the GitHub API for the
            default branch EVEN WHEN the local cache is warm — a transient
            network drop on any single rank then kills the whole 32-rank job
            at init (job 6015319: rank 7 RemoteDisconnected -> SIGTERM x31).
            Pinning ':main' + skip_validation makes the warm-cache path fully
            offline; the retry covers the cold-cache case on flaky links.
            """
            last = None
            for attempt in range(3):
                try:
                    return torch.hub.load(
                        "facebookresearch/co-tracker:main", "cotracker3_offline",
                        skip_validation=True,
                    ).to(self.device)
                except Exception as e:              # noqa: BLE001
                    last = e
                    import time as _t
                    _t.sleep(20 * (attempt + 1))
            raise last

        if distributed and hasattr(torch.distributed, "is_initialized") and torch.distributed.is_initialized():
            is_main = (torch.distributed.get_rank() == 0)
            if is_main:
                self._frozen_cotracker = _load_cotracker()
            torch.distributed.barrier()
            if not is_main:
                self._frozen_cotracker = _load_cotracker()
            torch.distributed.barrier()
        else:
            self._frozen_cotracker = _load_cotracker()

        self._frozen_cotracker.eval()
        for p in self._frozen_cotracker.parameters():
            p.requires_grad_(False)

        from action_query.ss_vae_model import load_ss_vae
        ss_vae_ckpt = str(getattr(
            self.config, "ss_vae_checkpoint", "action_query/checkpoints/ss_vae_8free.pt",
        ))
        ss_vae, scale = load_ss_vae(ss_vae_ckpt, device=str(self.device))
        ss_vae.eval()
        ss_vae.requires_grad_(False)
        self._frozen_ss_vae = ss_vae
        self._frozen_ss_vae_scale = scale

        self._motion_pipeline_ready = True
        log.info("Frozen motion pipeline ready (VAE + cotracker + ss_vae)")

    @staticmethod
    def _reduce_to_segments(per_frame: torch.Tensor, n_seg: int) -> torch.Tensor:
        """Mean-pool ``[F, D]`` into ``[n_seg, D]`` (teacher parity)."""
        F_len = per_frame.shape[0]
        seg_size = max(1, F_len // n_seg)
        segs = []
        for i in range(n_seg):
            s = i * seg_size
            e = min(s + seg_size, F_len)
            segs.append(per_frame[s:e].mean(dim=0))
        return torch.stack(segs)

    @torch.no_grad()
    def _compute_action_teacher_targets(self, pred_x0: torch.Tensor) -> torch.Tensor:
        """Ported from ``trainer/causal_diffusion_teacher_train.py``.

        Decodes ``pred_x0 [B, F_full, C, H, W]`` via the frozen VAE, runs
        CoTracker *per sample* (CoTracker's internal
        ``coords.view(B*T, N, 2)`` breaks for B>1 because of stride
        mismatches in its intermediate buffers — so we loop over B). The
        resulting point tracks + visibility are then encoded through
        ``ss_vae`` in a single batched ``[B*raw_n, 2, grid, grid]`` call.
        Returns ``[B, n_chunks, 8]``.

        With ``T_total = 21`` and ``compute_T = 48`` the inner time-sub-
        chunk loop runs exactly once and ``n_out == 1``; we keep the
        loop so the implementation generalises if either value is
        bumped in the future.

        Assumes every sample produces the same ``raw_n`` (true in
        practice because the input frame count is deterministic). We
        verify it explicitly before the batched ss_vae reshape.
        """
        if not self._motion_pipeline_ready:
            raise RuntimeError("motion pipeline not built; call ensure_motion_pipeline().")

        B, F_full, C, H, W = pred_x0.shape
        n_chunks = F_full // self.num_frame_per_block

        # Frozen VAE decode (fp32) → [B, T_total, 3, H_px, W_px] uint8-scaled.
        dummy = pred_x0[:, 0:1]
        latents_with_dummy = torch.cat([dummy, pred_x0], dim=1)
        pixels = self._frozen_vae.decode_to_pixel(
            latents_with_dummy.float(),
        )[:, 1:, ...]
        video = (255.0 * 0.5 * (pixels + 1.0)).clamp(0, 255).float()

        grid_size = 10
        output_chunk_size = 12
        compute_T = 48
        N = grid_size ** 2

        T_total = video.shape[1]

        # CoTracker does not safely support B>1 (its internal
        # ``coords.view(B*T, N, 2)`` fails when the batched queries produce
        # non-contiguous strides). Loop over B so each CoTracker call is
        # B=1, then stack. ss_vae below is still run as a single batched
        # encoder call over [B*raw_n, 2, grid, grid].
        mw_sub_chunks: List[torch.Tensor] = []
        for cs in range(0, T_total, compute_T):
            ce = min(cs + compute_T, T_total)
            ch_full = video[:, cs:ce]
            n_out = ch_full.shape[1] // output_chunk_size
            if n_out == 0:
                continue
            used = n_out * output_chunk_size
            ch_full = ch_full[:, :used]

            per_b_results: List[torch.Tensor] = []
            for b_idx in range(B):
                # [1, used, 3, H_px, W_px]; ``.clone()`` matches the
                # teacher-training path and guarantees a contiguous buffer.
                ch_b = ch_full[b_idx : b_idx + 1].clone()
                with torch.amp.autocast(device_type="cuda", enabled=True):
                    tracks_b, vis_b = self._frozen_cotracker(ch_b, grid_size=grid_size)
                tw_b = tracks_b.reshape(1, n_out, output_chunk_size, N, 2)
                if vis_b.dim() == 3:
                    vw_b = vis_b.reshape(1, n_out, output_chunk_size, N).unsqueeze(-1)
                else:
                    vw_b = vis_b.reshape(1, n_out, output_chunk_size, N, 1)
                dw_b = tw_b[:, :, 1:] - tw_b[:, :, :-1]
                mo_b = dw_b.mean(dim=2)                           # [1, n_out, N, 2]
                vo_b = vw_b.to(dtype=mo_b.dtype).mean(dim=2)      # [1, n_out, N, 1]
                per_b_results.append(torch.cat([mo_b, vo_b], dim=-1))
            mw_sub_chunks.append(torch.cat(per_b_results, dim=0))  # [B, n_out, N, 3]

        if not mw_sub_chunks:
            # Defensive: cannot happen for our standard F_full = 21 input,
            # but emit a well-defined zero tensor rather than crash.
            return torch.zeros(B, n_chunks, 8, device=pred_x0.device)

        # Stack sub-chunks along the chunk dim to form [B, raw_n, N, 3].
        est_motion = torch.cat(mw_sub_chunks, dim=1)
        raw_n = est_motion.shape[1]
        assert est_motion.shape[0] == B and est_motion.shape[-1] == 3, (
            f"Batched CoTracker expected [B={B}, raw_n={raw_n}, N={N}, 3]; "
            f"got {tuple(est_motion.shape)}"
        )

        # Batched ss_vae encoder call over [B*raw_n, 2, grid, grid].
        xy = est_motion[..., :2].reshape(B * raw_n, grid_size, grid_size, 2)
        x_in = xy.permute(0, 3, 1, 2).float() / self._frozen_ss_vae_scale
        mu, _ = self._frozen_ss_vae.encoder(x_in.to(self.device))  # [B*raw_n, 8, 1, 1]
        z8 = _tanh_squash(mu.squeeze(-1).squeeze(-1))              # [B*raw_n, 8]

        # Reduce each sample's raw_n rows to ``n_chunks`` via the same
        # mean-pool rule as ``_reduce_to_segments`` — applied per sample
        # so we keep deterministic parity with the old path.
        z8_per_b = z8.reshape(B, raw_n, 8)
        teacher_z_per_b: List[torch.Tensor] = [
            self._reduce_to_segments(z8_per_b[b], n_chunks) for b in range(B)
        ]
        teacher_z = torch.stack(teacher_z_per_b, dim=0)
        return teacher_z.detach()

    # ------------------------------------------------------------------
    # Conditional-dict construction (mirrors teacher lines 1370-1377)
    # ------------------------------------------------------------------

    def _build_conditional(
        self,
        prompt_embeds: torch.Tensor,
        z_noisy: torch.Tensor,
        z_clean: torch.Tensor,
        num_frames: int,
    ) -> Dict[str, torch.Tensor]:
        conditional: Dict[str, torch.Tensor] = {"prompt_embeds": prompt_embeds}
        if self.use_adaln and self.action_projection is not None:
            conditional["_action_modulation"] = self.action_projection(z_noisy, num_frames=num_frames)
            conditional["_action_modulation_clean"] = self.action_projection(z_clean, num_frames=num_frames)
        if self.use_action_tokens and self.action_token_projection is not None:
            conditional["_action_tokens"] = self.action_token_projection(z_noisy)
            conditional["_action_tokens_clean"] = self.action_token_projection(z_clean)
        return conditional

    # ------------------------------------------------------------------
    # Input preparation — per-block random timestep from the usable pool
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _get_random_blockwise_index(
        self,
        batch_size: int,
        num_frames: int,
        num_choices: int,
    ) -> torch.Tensor:
        """Per-block random index into the usable pool (``uniform_timestep=False``)."""
        assert num_frames % self.num_frame_per_block == 0
        idx = torch.randint(
            0, num_choices, [batch_size, num_frames],
            device=self.device, dtype=torch.long,
        )
        idx = idx.reshape(batch_size, -1, self.num_frame_per_block)
        idx[:, :, 1:] = idx[:, :, 0:1]
        return idx.reshape(batch_size, num_frames)

    @torch.no_grad()
    def _prep_random(
        self,
        trajectory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-block random sampling from the usable pool.

        Returns:
          noisy_input[B, F, C, H, W], timestep[B, F], pool_idx[B, F]
        """
        B, T_snap, F_, C, H, W = trajectory.shape
        K = int(self.denoising_step_list.shape[0])
        pool_idx = self._get_random_blockwise_index(B, F_, K)
        # ode_chunked_supervision (chained-LMDB mode): context frames are
        # COMMITTED/clean states tiled across snapshot slots — labeling them
        # with noisy rungs teaches an identity shortcut (observed pilot
        # collapse). The post-gather block below pins context to the clean
        # committed state at t=0; only the last block carries a sampled rung.

        pool_table = torch.tensor(
            self.usable_stored_indices, device=self.device, dtype=torch.long,
        )
        snap_idx = pool_table[pool_idx]

        noisy_input = torch.gather(
            trajectory, dim=1,
            index=snap_idx.reshape(B, 1, F_, 1, 1, 1).expand(-1, -1, -1, C, H, W),
        ).squeeze(1)

        timestep = self.denoising_step_list[pool_idx]
        if getattr(self, "ode_chunked_supervision", False):
            nfb = self.num_frame_per_block
            ctx = F_ - nfb
            committed = trajectory[:, -1, :ctx]            # final snapshot = clean
            noisy_input = torch.cat([committed, noisy_input[:, ctx:]], dim=1)
            timestep = timestep.clone()
            timestep[:, :ctx] = 0.0
            pool_idx = pool_idx.clone()
            pool_idx[:, :ctx] = -1        # exclude context from critic chunk masks
        return noisy_input, timestep, pool_idx

    @torch.no_grad()
    def _resolve_target(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Final teacher-x0 snapshot for every frame.

        The stored trajectory has ``T_snap = 7`` entries ordered
        ``[noise, 18, 36, 40, 44, 46, x0]``; the last entry is always the
        teacher's fully-denoised output. We regress against that for both
        the clean and CF branches — for CF this is automatically the
        teacher's CF x0 because the caller passes ``trajectory_cf``.
        """
        B, T_snap, F_, C, H, W = trajectory.shape
        final = int(T_snap) - 1
        tgt_snap = torch.full(
            (B, F_), final, dtype=torch.long, device=trajectory.device,
        )
        return torch.gather(
            trajectory, dim=1,
            index=tgt_snap.reshape(B, 1, F_, 1, 1, 1).expand(-1, -1, -1, C, H, W),
        ).squeeze(1)

    @staticmethod
    def _dist_features(x: torch.Tensor) -> torch.Tensor:
        """phi(z) for distribution matching: per-channel moments, quantiles
        and TEMPORAL deltas. The temporal/quantile parts are what a
        channel-Gaussian KL throws away — the omission that let the KL
        arms satisfy the loss while discarding the action fan.
        """
        xf = x.to(torch.float32)                       # [B, F, C, H, W]
        B, F_, C_, H_, W_ = xf.shape
        v = xf.permute(0, 2, 1, 3, 4).reshape(B, C_, -1)
        mu = v.mean(dim=-1)
        sd = v.std(dim=-1).clamp(min=1e-6).log()
        q = torch.quantile(
            v, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=v.device),
            dim=-1)                                    # [5, B, C]
        q = q.permute(1, 0, 2).reshape(B, -1)
        if F_ > 1:                                     # temporal structure
            d = (xf[:, 1:] - xf[:, :-1]).abs().mean(dim=(1, 3, 4))
        else:
            d = torch.zeros_like(mu)
        # SPATIALLY-SENSITIVE terms. Everything above is a per-channel
        # value histogram and is therefore INVARIANT to permuting voxels
        # — exactly the blindness that made ode_loss_type=kl action-blind
        # (verified: scrambling a prediction leaves that loss bit-identical
        # while MSE rises 66x). Without these a distribution loss can be
        # satisfied by a scrambled image, so add a coarse spatial grid of
        # (a) content and (b) SIGNED frame-to-frame motion — the latter is
        # where a left turn differs from a right turn.
        gh, gw = 3, 4
        pool = torch.nn.functional.adaptive_avg_pool2d
        grid = pool(xf.mean(dim=1), (gh, gw)).reshape(B, -1)
        if F_ > 1:
            dmap = (xf[:, 1:] - xf[:, :-1]).mean(dim=1)   # signed motion
            dgrid = pool(dmap, (gh, gw)).reshape(B, -1)
        else:
            dgrid = torch.zeros_like(grid)
        return torch.cat([mu, sd, q, d, grid, dgrid], dim=1)

    def _energy_distance_loss(self, p_c, t_c, p_f, t_f, tt_c, tt_f):
        """D_E(student, teacher) over the fan batch, gathered across ranks.

        Only commit-rung samples participate (the statistic is a property
        of the committed chunk). Returns None when the global sample count
        is too small for the U-statistic to mean anything.
        """
        import torch.distributed as dist
        sel_c = (tt_c < 250.0) if self.ode_edist_commit_only else torch.ones_like(tt_c, dtype=torch.bool)
        sel_f = (tt_f < 250.0) if self.ode_edist_commit_only else torch.ones_like(tt_f, dtype=torch.bool)
        xs, ys = [], []
        for _p, _t, _s in ((p_c, t_c, sel_c), (p_f, t_f, sel_f)):
            if int(_s.sum()) == 0:
                continue
            xs.append(self._dist_features(_p[_s]))
            ys.append(self._dist_features(_t[_s].float()).detach())
        # Every rank must reach the collectives below, so build (possibly
        # empty) local tensors and let the gather decide if there is data.
        dev = p_c.device
        # phi width: C*(1 mu + 1 sd + 5 quantiles + 1 temporal-mag)
        #          + C*gh*gw content grid + C*gh*gw signed-motion grid
        D = p_c.shape[2] * (8 + 2 * 3 * 4)
        X = torch.cat(xs, 0) if xs else torch.zeros(0, D, device=dev)
        Y = torch.cat(ys, 0) if ys else torch.zeros(0, D, device=dev)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            world = dist.get_world_size()
            n_loc = torch.tensor([X.shape[0]], device=dev)
            counts = [torch.zeros_like(n_loc) for _ in range(world)]
            dist.all_gather(counts, n_loc)
            n_max = int(max(int(c.item()) for c in counts))
            if n_max == 0:
                return None
            def _gather(T):
                pad = torch.zeros(n_max, D, device=dev, dtype=T.dtype)
                if T.shape[0]:
                    pad[:T.shape[0]] = T
                buf = [torch.zeros_like(pad) for _ in range(world)]
                dist.all_gather(buf, pad)
                buf[dist.get_rank()] = pad              # keep LOCAL grad path
                return torch.cat([b[:int(counts[i].item())]
                                  for i, b in enumerate(buf)], 0)
            X, Y = _gather(X), _gather(Y)
        return self._edist_core(X, Y)

    def _edist_core(self, X, Y, gid=None):
        """Energy distance between two GATHERED feature sets.

        D_E = 2E||X-Y|| - E||X-X'|| - E||Y-Y'||; the last term is constant
        (Y is detached) so it is dropped. The -E||X-X'|| term is a BOUNDED
        anti-collapse repulsion whose optimum is the teacher distribution
        rather than infinite separation.
        """
        import torch.distributed as dist
        if X.shape[0] < 2 or Y.shape[0] < 2:
            return None
        # Scale each feature by the teacher's spread so no single family
        # (e.g. the 80 quantile dims) dominates the distance.
        _sd = Y.std(dim=0, keepdim=True).detach()
        # A degenerate set (all rows identical -> median spread 0) would floor
        # the scale at 1e-8 and amplify features by 1e8. Y is identical on every
        # rank here, so returning None is collective-safe.
        if float(_sd.median()) <= 1e-8:
            return None
        s = _sd.clamp(min=1e-2 * _sd.median().clamp(min=1e-6))
        Xn, Yn = X / s, Y / s
        _cx = torch.cdist(Xn, Yn)
        _xx = torch.cdist(Xn, Xn)
        n = Xn.shape[0]
        if gid is not None:
            # WITHIN-GROUP pairs only = the action fan of one context.
            same = (gid.view(-1, 1) == gid.view(1, -1)).float()
            eye = torch.eye(n, device=X.device)
            n_cross = same.sum().clamp(min=1.0)
            n_self = (same - same * eye).sum().clamp(min=1.0)
            cross = (_cx * same).sum() / n_cross
            xx = (_xx * (same - same * eye)).sum() / n_self
        else:
            cross = _cx.mean()
            xx = _xx.sum() / max(n * (n - 1), 1)        # exclude diagonal
        # DDP AVERAGES rank gradients, but each rank contributes only
        # its own 1/W rows of X, so the effective weight would be 1/W
        # (1/32 here). Scale back so the flag means what it says.
        # Each rank contributes only its own 1/W rows of X, so the effective
        # gradient weight after DDP's AVG would be 1/W; scale back so the flag
        # means what it says. This holds with grouping too: a rank's row takes
        # part in G of the W*G same-group pairs, i.e. 1/W of `cross` either
        # way, so the rescale is world-size-INVARIANT as written.
        _w = float(dist.get_world_size()) if (
            dist.is_available() and dist.is_initialized()) else 1.0
        return _w * (2.0 * cross - xx)   # -E||Y-Y'|| is constant

    def edist_chunk(self, pred, target, group_id=None):
        """Set-level energy distance for ONE committed chunk, across ranks.

        This is the loss the user's distribution-matching theory actually
        calls for: each rank holds a DIFFERENT commanded direction of the
        same context (see alldir_batches / _CurriculumBatches), so gathering
        the committed chunks across ranks gives the OUTPUT DISTRIBUTION
        induced by the ACTION DISTRIBUTION, and D_E compares it to the
        teacher's. Contrast with the per-sample KL, which matches one
        prediction to one target via per-channel moments only.

        Collective-safe: every rank calls the same gathers unconditionally.
        """
        import torch.distributed as dist
        X = self._dist_features(pred)
        Y = self._dist_features(target.float()).detach()
        gid = None
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            world, rank = dist.get_world_size(), dist.get_rank()
            def _g(T):
                buf = [torch.zeros_like(T) for _ in range(world)]
                dist.all_gather(buf, T.contiguous())
                buf[rank] = T                     # keep the LOCAL grad path
                return torch.cat(buf, 0)
            X, Y = _g(X), _g(Y)
            # Group id per row, so D_E is computed WITHIN a context (the action
            # fan) instead of across the 4-8 contexts a global batch spans.
            # Without this 90-97% of the pairs compare different contexts and
            # the term measures mixture spread, not the conditional fan.
            if group_id is not None:
                gid = _g(torch.full((1,), float(group_id), device=X.device))
        return self._edist_core(X, Y, gid)

    def _nextrung_targets(
        self,
        trajectory: torch.Tensor,     # [B, T_snap, F, C, H, W]
        base_target: torch.Tensor,    # [B, F, C, H, W] committed x0
        noisy_input: torch.Tensor,    # [B, F, C, H, W] gathered rung states
        pool_idx: torch.Tensor,       # [B, F] rung pool index (-1 = context)
    ) -> torch.Tensor:
        """Arm-D (next-rung state regression): on rung-labeled frames, swap
        the committed-x0 target for the progressive-distillation target the
        teacher's stored NEXT trajectory state implies:

            x0_hat_k = x_k - sigma_k * (x_k - x_{k+1}) / (sigma_k - sigma_{k+1})

        x_{k+1} is the next stored snapshot of the SAME chain (final slot =
        the committed sample, sigma=0). Because stored states straddle
        MULTI-step segments of the teacher's 20-step schedule, x0_hat is a
        segment extrapolation — a sample-consistent tangent of the teacher's
        actual path — not a one-step conditional mean. At the last rung the
        formula reduces exactly to the committed target (v1 continuity);
        context frames (pool_idx = -1) keep base_target.
        """
        B, T_snap, F_, C, H, W = trajectory.shape
        K = int(self.denoising_step_list.shape[0])
        stored = list(self.usable_stored_indices)
        if stored != list(range(K)) or T_snap != K + 1:
            raise RuntimeError(
                "ode_nextrung_targets assumes contiguous rung slots "
                f"[0..{K-1}] + committed final; got usable_stored_indices="
                f"{stored}, T_snap={T_snap}")
        pool = pool_idx.clamp(min=0)
        next_slot = (pool + 1).clamp(max=T_snap - 1)
        x_next = torch.gather(
            trajectory, dim=1,
            index=next_slot.reshape(B, 1, F_, 1, 1, 1).expand(-1, -1, -1, C, H, W),
        ).squeeze(1).float()
        sig = self.denoising_step_list.to(
            device=pool.device, dtype=torch.float32) / 1000.0
        sig_pad = torch.cat([sig, torch.zeros(1, device=pool.device)])
        sig_k = sig_pad[pool]
        sig_n = sig_pad[(pool + 1).clamp(max=K)]
        x_k = noisy_input.float()
        _solver = str(getattr(self.config, "ode_nextrung_solver", "euler")).lower()
        if _solver in ("midpoint", "heun"):
            # Higher-order tangent from the teacher's stored path. The
            # Euler form uses the single secant over [k, k+1]; these use a
            # second stored segment [k+1, k+2] so the x0 extrapolation
            # follows the CURVATURE of the teacher's trajectory instead of
            # its local chord (the chord systematically under-shoots on a
            # curved path, which is one source of the mean-like target).
            #   midpoint: central secant over [k, k+2]  (slope at k+1)
            #   heun    : mean of secant[k,k+1] and secant[k+1,k+2]
            nxt2 = (pool + 2).clamp(max=T_snap - 1)
            x_next2 = torch.gather(
                trajectory, dim=1,
                index=nxt2.reshape(B, 1, F_, 1, 1, 1).expand(-1, -1, -1, C, H, W),
            ).squeeze(1).float()
            sig_n2 = sig_pad[(pool + 2).clamp(max=K)]
            d1 = (sig_k - sig_n).clamp(min=1e-6).reshape(B, F_, 1, 1, 1)
            d2 = (sig_n - sig_n2).clamp(min=1e-6).reshape(B, F_, 1, 1, 1)
            v1 = (x_k - x_next) / d1
            v2 = (x_next - x_next2) / d2
            # Where the second segment does not exist (last rung: k+1 is
            # already the committed slot) fall back to the Euler secant,
            # preserving the exact v1 continuity property.
            has2 = ((pool + 2) <= K).reshape(B, F_, 1, 1, 1)
            if _solver == "heun":
                v = torch.where(has2, 0.5 * (v1 + v2), v1)
            else:
                dtot = (sig_k - sig_n2).clamp(min=1e-6).reshape(B, F_, 1, 1, 1)
                v = torch.where(has2, (x_k - x_next2) / dtot, v1)
            x0_hat = x_k - sig_k.reshape(B, F_, 1, 1, 1) * v
        else:
            w = (sig_k / (sig_k - sig_n).clamp(min=1e-6)).reshape(B, F_, 1, 1, 1)
            x0_hat = x_k - w * (x_k - x_next)
        mask = (pool_idx >= 0).reshape(B, F_, 1, 1, 1)
        return torch.where(
            mask, x0_hat, base_target.float()).to(base_target.dtype)

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

    def _compute_ode_loss(
        self,
        pred_x0: torch.Tensor,
        target: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """``MSE(pred_x0, target)`` averaged over frames where ``timestep != 0``.

        Mirrors the Causal-Forcing teacher (``Causal-Forcing/model/
        ode_regression.py:124``): at timestep 0 the flow-matching wrapper
        returns ``pred_x0 == xt == target`` by construction, so those
        frames contribute zero signal and would only dilute the mean. If
        every frame is at ``t == 0`` we fall back to the plain MSE to
        avoid an empty ``reduction="mean"`` division.
        """
        mask = timestep != 0
        _lt = getattr(self.config, "ode_loss_type", "mse")
        if _lt == "kl_local":
            # LOCALISED Gaussian KL. The global variant reduces over
            # dims=(1,3,4), i.e. one mean+var per CHANNEL over the whole
            # 18,720-voxel chunk. Measured consequences of that reduction:
            #   * gradient rank 2 per channel -> 32 constrained scalars out of
            #     299,520 (0.011%); it can shift and rescale the existing
            #     pattern but never change WHICH pixels move;
            #   * bit-exact invariance to spatially scrambling the prediction;
            #   * a "moment cheat" -- take the OPPOSITE direction's chunk and
            #     affine-rescale it to the target's channel moments -- scores
            #     MACHINE ZERO while its MSE is 91% of a fully wrong answer.
            # Reducing per (channel x frame x spatial cell) on a gh x gw grid
            # keeps the log(st/sp) anti-collapse term (which genuinely resists
            # blur: KL prefers a crisp wrong answer where MSE prefers the blur
            # by 2.5x) while taking the constrained count to 2*C*F*gh*gw and
            # destroying the global permutation invariance.
            gh = int(getattr(self.config, "ode_kl_grid_h", 3))
            gw = int(getattr(self.config, "ode_kl_grid_w", 4))
            m = mask if mask.any() else torch.ones_like(mask)
            _m = m[..., None, None, None]
            p_ = (pred_x0.float() * _m)
            t_ = (target.float() * _m)
            B, F_, C_, H_, W_ = p_.shape
            pool = torch.nn.functional.adaptive_avg_pool2d
            def _cells(x):
                y = x.reshape(B * F_, C_, H_, W_)
                mu = pool(y, (gh, gw))
                var = (pool(y * y, (gh, gw)) - mu * mu).clamp(min=1e-6)
                return mu.reshape(B, F_, C_, gh, gw), var.reshape(B, F_, C_, gh, gw)
            mp, sp2 = _cells(p_)
            mt, st2 = _cells(t_)
            kl = (0.5 * torch.log(st2 / sp2)
                  + (sp2 + (mp - mt).pow(2)) / (2.0 * st2) - 0.5)
            return kl.mean().to(pred_x0.dtype)
        if _lt == "kl":
            # Per-channel Gaussian KL(pred || target) over the masked
            # frames: log(st/sp) + (sp^2 + (mp-mt)^2) / (2 st^2) - 1/2.
            # log(st/sp) -> +inf as sp -> 0: under-dispersion is punished
            # asymmetrically, the anti-collapse property MSE lacks.
            m = mask if mask.any() else torch.ones_like(mask)
            p = pred_x0.float() * m[..., None, None, None]
            t = target.float() * m[..., None, None, None]
            nfrm = m.sum(dim=1).clamp(min=1)
            dims = (1, 3, 4)
            scale = (pred_x0.shape[1] / nfrm).view(-1, 1)
            mp = p.mean(dim=dims) * scale
            mt = t.mean(dim=dims) * scale
            # centred second moments over masked region only
            vp = (p - (mp / scale).view(p.shape[0], 1, -1, 1, 1) * m[..., None, None, None]).pow(2).mean(dim=dims) * scale
            vt = (t - (mt / scale).view(t.shape[0], 1, -1, 1, 1) * m[..., None, None, None]).pow(2).mean(dim=dims) * scale
            sp2 = vp.clamp(min=1e-6); st2 = vt.clamp(min=1e-6)
            kl = 0.5 * torch.log(st2 / sp2) + (sp2 + (mp - mt).pow(2)) / (2.0 * st2) - 0.5
            return kl.mean().to(pred_x0.dtype)
        if not mask.any():
            return F.mse_loss(pred_x0.float(), target.float())
        # mask is [B, F]; broadcast to the full [B, F, C, H, W] shape.
        m = mask[..., None, None, None]
        diff = (pred_x0.float() - target.float()) ** 2
        return (diff * m).sum() / (m.expand_as(diff).sum().clamp_min(1.0))

    def _state_probe_loss(
        self,
        state_preds: torch.Tensor,        # [B, n_c, D]
        teacher_z_8d: Optional[torch.Tensor],  # [B, n_c, 8] or None
        z_clean_chunked: torch.Tensor,    # [B, n_c, 2]
    ) -> torch.Tensor:
        """Supervise the probe against teacher_z_8d when available, else commanded z."""
        n_c = state_preds.shape[1]
        D = state_preds.shape[-1]
        if teacher_z_8d is not None:
            target = teacher_z_8d[:, :n_c]
            if D <= len(self.action_critic_dims):
                # Probe is 2-D; slice teacher z down to action_critic_dims.
                target = target[:, :, self.action_critic_dims]
                return F.mse_loss(state_preds.float(), target.float())
            # 8-D supervision, up-weight z2/z7.
            w = torch.ones(D, device=state_preds.device, dtype=state_preds.dtype)
            for d in self.action_critic_dims:
                w[d] = self.state_head_action_dim_weight
            return (w * (state_preds.float() - target.float()) ** 2).mean()

        # Fallback: commanded-z supervision (no motion pipeline).
        if D <= len(self.action_critic_dims):
            return F.mse_loss(state_preds.float(), z_clean_chunked.float())
        target = torch.zeros_like(state_preds)
        for i, d in enumerate(self.action_critic_dims):
            target[:, :, d] = z_clean_chunked[:, :, i]
        w = torch.ones(D, device=state_preds.device, dtype=state_preds.dtype)
        for d in self.action_critic_dims:
            w[d] = self.state_head_action_dim_weight
        return (w * (state_preds.float() - target.float()) ** 2).mean()

    def _critic_chunk_mask(
        self,
        B: int,
        pool_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Chunks the critic should supervise: the top-``critic_num_chunks``
        cleanest pool levels (``pool_idx >= K - k``). Returns ``[B, n_chunks]``.
        """
        K = int(self.denoising_step_list.shape[0])
        k = min(self.critic_num_chunks, self.n_chunks)
        # pool_idx is [B, F]; one entry per frame. All frames in a chunk share
        # the same pool index by construction of _get_random_blockwise_index.
        chunk_pool = pool_idx[:, :: self.num_frame_per_block][:, : self.n_chunks]
        return chunk_pool >= (K - k)

    def _state_guidance_scale(self, step: int) -> float:
        """Teacher-parity schedule: warmup then constant ``state_guidance_weight``."""
        if self.state_guidance_weight <= 0:
            return 0.0
        warmup_start = self.warmup_steps
        if step < warmup_start:
            return 0.0
        if self.state_guidance_warmup_steps > 0:
            ramp = min(
                1.0, (step - warmup_start) / float(self.state_guidance_warmup_steps),
            )
            return ramp * self.state_guidance_weight
        return self.state_guidance_weight

    def _compute_state_guidance_loss(
        self,
        probe_hidden: torch.Tensor,        # [B, n_c, probe_dim]
        chunk_actions: torch.Tensor,       # [B, n_c, 2]  commanded action (z2/z7)
        step: int,
    ) -> Tuple[torch.Tensor, float]:
        """Frozen-readout generator guidance (teacher parity).

        Runs the probe's pooled hidden features through the state_probe
        readout with *detached* weights, then regresses the action-dim
        outputs toward ``state_guidance_action_scale * commanded_z``.
        Gradients flow only through ``probe_hidden`` (i.e. back into the
        DiT + probe backbone), never into the readout.

        Returns ``(loss, scale)``. ``scale == 0`` means the weight was
        zero or warmup hadn't completed — we return ``torch.zeros(())``
        in that case so the caller can still ``.detach()`` for logging.
        """
        scale = self._state_guidance_scale(step)
        if scale <= 0.0 or probe_hidden is None:
            return torch.zeros((), device=self.device), 0.0

        base_module = self._gen_base_module()
        probe = getattr(base_module, "_state_probe", None)
        if probe is None:
            return torch.zeros((), device=self.device), 0.0

        n_c = probe_hidden.shape[1]
        readout = probe.readout
        # Detach both weight and bias so the readout is frozen for this
        # gradient path. Gradients flow back only through ``probe_hidden``.
        frozen_preds_raw = F.linear(
            probe_hidden[:, :n_c].float(),
            readout.weight.detach(),
            readout.bias.detach(),
        )                                                               # [B, n_c, D_out]
        if self.state_head_out_dim <= len(self.action_critic_dims):
            frozen_preds = frozen_preds_raw
        else:
            frozen_preds = frozen_preds_raw[:, :, self.action_critic_dims]

        cmd_target = self.state_guidance_action_scale * chunk_actions.float()
        loss = F.mse_loss(frozen_preds, cmd_target) * scale
        return loss, float(scale)

    def _gen_base_module(self):
        """Return the underlying WanDiffusionWrapper, unwrapping any DDP."""
        g = self.generator
        try:
            from torch.nn.parallel import DistributedDataParallel as _DDP
            if isinstance(g, _DDP):
                return g.module
        except Exception:
            pass
        return g

    def _compute_critic_guidance_loss(
        self,
        pred_x0: torch.Tensor,                 # [B, F, C, H, W]
        chunk_t: torch.Tensor,                 # [B, n_c]
        chunk_actions: torch.Tensor,           # [B, n_c, 2]
        chunk_mask: torch.Tensor,              # [B, n_c] bool
        step: int = 0,
    ) -> Tuple[torch.Tensor, float]:
        """Frozen-critic generator guidance (gradient through ``pred_x0``).

        Returns ``(loss, guidance_scale)``.

        The ``step`` argument is honoured only when a dormant warmup ramp is
        configured (both ``warmup_steps > 0`` and/or
        ``z_guidance_warmup_steps > 0``). By default both are 0, which makes
        the weight constant from step 0 — the intended distillation regime
        now that the critic is warm-started from v14.
        """
        if self.action_critic is None or not chunk_mask.any():
            return torch.zeros((), device=self.device), 0.0

        if self.warmup_steps > 0 and step < self.warmup_steps:
            scale = 0.0
        elif self.z_guidance_warmup_steps > 0 and step < (self.warmup_steps + self.z_guidance_warmup_steps):
            ramp = min(1.0, (step - self.warmup_steps) / self.z_guidance_warmup_steps)
            scale = ramp * self.generator_action_z_guidance_weight
        else:
            scale = self.generator_action_z_guidance_weight
        if scale <= 0.0:
            return torch.zeros((), device=self.device), 0.0

        # Use the *unwrapped* critic here: we backprop only through pred_x0,
        # critic params are frozen for this call, and we don't want DDP
        # forward tracking (the critic has its own DDP cycle via the
        # ``_critic_update_loop``).
        from torch.nn.parallel import DistributedDataParallel as _DDP
        critic_mod = (
            self.action_critic.module
            if isinstance(self.action_critic, _DDP)
            else self.action_critic
        )

        # Per-parameter ``requires_grad`` save/restore. A coarse "assume
        # all params share the same flag" pattern breaks if any critic
        # parameter was ever individually frozen (e.g. by a downstream
        # finetune strategy). This form is cheap and robust.
        saved_req_grad = [p.requires_grad for p in critic_mod.parameters()]
        for p in critic_mod.parameters():
            p.requires_grad_(False)
        try:
            gen_pred_z = critic_mod(pred_x0, chunk_t, chunk_actions)
            gen_pred_z = gen_pred_z[:, : self.n_chunks]                              # [B, n_c, 8]
            gen_z27 = gen_pred_z[:, :, self.action_critic_dims]                      # [B, n_c, 2]
            tgt_z27 = chunk_actions                                                  # commanded z
            per_chunk = F.mse_loss(gen_z27, tgt_z27, reduction="none").mean(dim=-1)  # [B, n_c]
            if chunk_mask.any():
                loss = per_chunk[chunk_mask].mean() * scale
            else:
                loss = torch.zeros((), device=self.device)
        finally:
            for p, req in zip(critic_mod.parameters(), saved_req_grad):
                p.requires_grad_(req)
        return loss, float(scale)

    # ------------------------------------------------------------------
    # Distilled inference — used by the in-loop evaluator and smoke test.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_eval(
        self,
        conditional: Dict[str, torch.Tensor],
        clean_x: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Few-step student inference.

        Iterates ``self.denoising_step_list`` in noisiest-to-cleanest order,
        re-noising ``pred_x0`` to the next timestep between iterations via
        ``self.scheduler.add_noise`` (pattern from
        ``pipeline/action_inference.py``).

        IMPORTANT — DDP SAFETY: this method MUST remain ``@torch.no_grad``.
        At training time the trainer replaces ``self.generator`` with a
        ``DistributedDataParallel`` wrapper. DDP.forward without grad is
        a pass-through (no backward hooks, no collective expectation), so
        rank-0-only eval is safe. Dropping the decorator would have DDP
        register ``prepare_for_backward`` expectations on rank 0 while the
        other ranks are elsewhere, which would deadlock or diverge on the
        next training step.

        Args:
          conditional: four-stream action conditioning dict from
              ``_build_conditional`` (bf16-cast if ``mixed_precision``).
          clean_x: ``[B, F, C, H, W]`` teacher-forced attention context
              (ground-truth preceding 21 latent frames).
          noise: ``[B, F, C, H, W]`` starting latent (fp32 on device).

        Returns:
          ``pred_x0`` in fp32, shape matches ``noise``.
        """
        ts = self.denoising_step_list.detach().to(
            device=noise.device, dtype=torch.float32,
        )
        ts, _ = torch.sort(ts, descending=True)

        B, F_, C, H, W = noise.shape
        x = noise.clone().to(self.dtype)
        clean_x_cast = clean_x.to(self.dtype)
        scheduler = self.scheduler
        # NOTE: in-place mutation of ``scheduler.sigmas``. This is safe
        # today because the scheduler is only used here (for ``add_noise``
        # between few-step iterations); it has no other live users. If
        # you ever share the scheduler across training + eval paths or
        # across ranks, move to a local copy instead:
        #     scheduler = copy.deepcopy(self.scheduler)
        scheduler.sigmas = scheduler.sigmas.to(noise.device)

        pred_x0: Optional[torch.Tensor] = None
        for i in range(ts.shape[0]):
            t = ts[i].item()
            tt = torch.full((B, F_), t, device=noise.device, dtype=torch.float32)
            with torch.amp.autocast("cuda", dtype=self.dtype):
                out = self.generator(
                    noisy_image_or_video=x,
                    conditional_dict=conditional,
                    timestep=tt,
                    clean_x=clean_x_cast,
                    aug_t=None,
                )
            pred_x0 = out[1]
            if i < ts.shape[0] - 1:
                next_t = float(ts[i + 1].item())
                flat_pred = pred_x0.flatten(0, 1).float()
                flat_noise = torch.randn_like(flat_pred)
                flat_t = torch.full(
                    (flat_pred.shape[0],), next_t,
                    device=noise.device, dtype=torch.float32,
                )
                x = scheduler.add_noise(
                    flat_pred, flat_noise, flat_t,
                ).view(B, F_, C, H, W).to(self.dtype)
        assert pred_x0 is not None
        return pred_x0.float()

    # ------------------------------------------------------------------
    # Main training entrypoint (packed clean+CF forward).
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Online dual-CD losses (ODE-F): frozen-14d teacher-CD + student self-CD
    # ------------------------------------------------------------------
    def _cd_unwrapped_generator(self):
        """The bare WanDiffusionWrapper under ``self.generator`` (which the
        ODE trainer replaces with a DDP wrapper). CD forwards MUST use the
        bare module: a second DDP forward would re-arm the reducer and trip
        'marked ready only once'. Grads from the bare student-CD forward
        still land on the shared leaf params and are all-reduced by the
        reducer armed during the MAIN DDP forward (single combined
        backward)."""
        g = self.generator
        return g.module if hasattr(g, "module") else g

    def _cd_current_weight(self, step: int) -> float:
        """Linear 0->1 warmup ramp over ``cd_loss_warmup_steps`` (then 1.0).
        Holds the consistency regularizers sub-ODE until the student makes
        meaningful x0 predictions."""
        w = int(self.cd_loss_warmup_steps)
        if w <= 0:
            return 1.0
        return min(1.0, max(0, int(step)) / float(w))

    def _cd_sample_rung_pair(self):
        """Adjacent rung pair (t_n > t_next) from ``cd_rung_list`` — ALL saved
        snapshot rungs (high->low), index broadcast from rank 0 for DDP
        lockstep. Returns (t_n, t_next) float timesteps, or None if <2 rungs."""
        import torch.distributed as dist
        ds = self.cd_rung_list
        K = int(ds.shape[0])
        if K < 2:
            return None
        idx = torch.empty(1, dtype=torch.long, device=self.device)
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            idx[0] = int(torch.randint(0, K - 1, (1,), device=self.device).item())
        if dist.is_initialized():
            dist.broadcast(idx, src=0)
        i = int(idx[0].item())
        return float(ds[i].item()), float(ds[i + 1].item())

    def _cd_partial_denoise(self, x_ts, x0_hat, t_s_tensor, t_e_tensor):
        """One Euler step from level ``t_s`` to ``t_e`` using the model's
        clean-space x0 prediction, in the FlowMatchScheduler's sigma space
        (respects timestep_shift). Mirrors the DMD ``_flow_partial_denoise``.
        ``t_*_tensor``: [B, F] timestep values."""
        sched = self.scheduler
        sched.sigmas = sched.sigmas.to(x_ts.device)
        sched.timesteps = sched.timesteps.to(x_ts.device)
        B, F = t_s_tensor.shape
        flat_s = t_s_tensor.flatten().float()
        flat_e = t_e_tensor.flatten().float()
        id_s = torch.argmin((sched.timesteps.unsqueeze(0) - flat_s.unsqueeze(1)).abs(), dim=1)
        id_e = torch.argmin((sched.timesteps.unsqueeze(0) - flat_e.unsqueeze(1)).abs(), dim=1)
        sigma_s = sched.sigmas[id_s].float().view(B, F, 1, 1, 1)
        sigma_e = sched.sigmas[id_e].float().view(B, F, 1, 1, 1)
        ratio = sigma_e / sigma_s.clamp(min=1e-8)
        coef_x0 = (1.0 - sigma_e) - ratio * (1.0 - sigma_s)
        return ratio.to(x_ts.dtype) * x_ts + coef_x0.to(x_ts.dtype) * x0_hat

    @torch.no_grad()
    def _update_cd_ema_and_teacher(self) -> None:
        """Lazily create the frozen-14d teacher + EMA-student (first call),
        then EMA-update the EMA-student toward the live generator. Both are
        deepcopies of the bare (merged-14d at first call) generator, frozen,
        eval, and held off the module registry."""
        # No EMA target requested: the frozen teacher is already captured in
        # __init__, and the CD target uses the live student (stop-grad) — so
        # there is nothing to create/update here. Skip (saves a resident DiT).
        if not self.cd_ema_enabled:
            return
        import copy as _copy
        base = self._cd_unwrapped_generator()
        if self._cd_ema is None:
            # Teacher is normally captured in __init__ (frozen-14d, resume-safe);
            # only fall back to a lazy capture here if it is somehow still None.
            if self.cd_teacher_loss_enabled and self._cd_teacher is None:
                tea = _copy.deepcopy(base)
                tea.requires_grad_(False)
                tea.eval()
                object.__setattr__(self, "_cd_teacher", tea)
            ema = _copy.deepcopy(base)
            ema.requires_grad_(False)
            ema.eval()
            object.__setattr__(self, "_cd_ema", ema)
            return  # first call: EMA == live; nothing to blend yet
        d = self.cd_ema_decay
        live = dict(base.named_parameters())
        for n, p_ema in self._cd_ema.named_parameters():
            p_live = live.get(n, None)
            if p_live is not None and p_live.shape == p_ema.shape:
                p_ema.mul_(d).add_(p_live.detach().to(p_ema.dtype), alpha=1.0 - d)
        live_buf = dict(base.named_buffers())
        for n, b_ema in self._cd_ema.named_buffers():
            b_live = live_buf.get(n, None)
            if b_live is not None and b_live.shape == b_ema.shape:
                b_ema.copy_(b_live)

    def _cd_forward_x0(self, module, x_t, t_tensor, conditional, clean_x, aug_t):
        """Forward a CD module (bare student / frozen teacher / EMA) and
        return its x0 prediction (element [1] of the 4-tuple)."""
        out = module(
            noisy_image_or_video=x_t,
            conditional_dict=conditional,
            timestep=t_tensor,
            clean_x=clean_x,
            aug_t=aug_t,
        )
        return out[1] if isinstance(out, tuple) else out

    def _compute_cd_losses(self, x0_cd, conditional, clean_x, aug_t):
        """Teacher-CD + student-CD on one branch.

        x0_cd: [B, F, C, H, W] clean target (teacher final x0). Both CDs:
          x_n = noise(x0_cd, t_n); p_n = student(x_n, t_n)            [grad]
          teacher-CD: x_{n+1} = teacher-step(x_n); target EMA(x_{n+1}) [no_grad]
          student-CD: x_{n+1} = student-step(x_n); target EMA(x_{n+1}) [no_grad]
          loss = MSE(p_n, target)
        Returns (cd_teacher, cd_student, logs); zeros for disabled halves.
        """
        device, dtype = self.device, self.dtype
        zero = torch.zeros((), device=device, dtype=dtype)
        pair = self._cd_sample_rung_pair()
        if pair is None:
            return zero, zero, {"cd_skipped": 1.0}
        t_n, t_next = pair
        B, F_ = x0_cd.shape[:2]
        x0_cd = x0_cd.to(dtype=dtype, device=device).detach()
        eps = torch.randn_like(x0_cd)
        # Use the EXACT fp32 rung timesteps (e.g. 312.5, 178.6) — NOT int64.
        # The main ODE path feeds float ``denoising_step_list`` values to the
        # DiT + scheduler.add_noise; rounding to int here (312.5->312) would
        # make the CD student see a different timestep than it is trained on at
        # the same rung, biasing the consistency target. add_noise and the DiT
        # both accept float timesteps (and _cd_partial_denoise floats internally).
        t_n_t = torch.full([B, F_], float(t_n), device=device, dtype=torch.float32)
        t_next_t = torch.full([B, F_], float(t_next), device=device, dtype=torch.float32)
        x_n = self.scheduler.add_noise(
            x0_cd.flatten(0, 1), eps.flatten(0, 1), t_n_t.flatten(0, 1),
        ).unflatten(0, x0_cd.shape[:2]).to(dtype).contiguous()

        bare = self._cd_unwrapped_generator()
        # Shared grad-carrying student x0 at (x_n, t_n).
        p_n = self._cd_forward_x0(bare, x_n, t_n_t, conditional, clean_x, aug_t)

        # CD *target* network: EMA-student if enabled, else the LIVE student.
        # Either way the readout runs inside ``torch.no_grad()`` below, so the
        # target branch is stop-gradient (the property that actually prevents
        # consistency collapse) regardless of which module is used.
        tgt_mod = self._cd_ema if self.cd_ema_enabled else bare

        cd_teacher, cd_student = zero, zero
        logs = {"cd_t_n": float(t_n), "cd_t_next": float(t_next), "cd_skipped": 0.0}

        if self.cd_teacher_loss_enabled and self._cd_teacher is not None:
            with torch.no_grad():
                tea_x0 = self._cd_forward_x0(
                    self._cd_teacher, x_n, t_n_t, conditional, clean_x, aug_t)
                x_next_T = self._cd_partial_denoise(x_n, tea_x0, t_n_t, t_next_t)
                p_next_T = self._cd_forward_x0(
                    tgt_mod, x_next_T, t_next_t, conditional, clean_x, aug_t)
            cd_teacher = (p_n.float() - p_next_T.float()).pow(2).mean().to(dtype)
            logs["cd_teacher_loss_raw"] = float(cd_teacher.detach().item())

        if self.cd_student_loss_enabled:
            with torch.no_grad():
                x_next_S = self._cd_partial_denoise(
                    x_n, p_n.detach(), t_n_t, t_next_t)
                p_next_S = self._cd_forward_x0(
                    tgt_mod, x_next_S, t_next_t, conditional, clean_x, aug_t)
            cd_student = (p_n.float() - p_next_S.float()).pow(2).mean().to(dtype)
            logs["cd_student_loss_raw"] = float(cd_student.detach().item())

        return cd_teacher, cd_student, logs

    def _grep_chunk(self, pred, target=None):
        """GAUSSIAN REPULSOR for one committed chunk. Reference is N(0,1).

        Purpose is EXPOSURE BIAS, not dispersion matching: over an AR rollout
        the student's committed chunks drift toward the noise point, and the
        drift COMPOUNDS chunk over chunk. A standing repulsion from N(0,1)
        makes each step drift less, so the compounding is damped.

        Deliberately NOT the teacher-referenced deadband: the student has to be
        free to be better than the teacher, so the reference is the noise point
        itself, and there is no satisfied region -- the term is always on.

        Inverse-square in the signature distance (same law as the original
        `ode_noiserep`), so the force is strong close to the noise point and
        falls away as the prediction gets clear of it:

            d2 = sum_c mu_c^2 + sum_c (sigma_c - 1)^2      (per channel)
            L  = w / d2

        Minimising L maximises d2, i.e. repels. `target` is accepted and
        ignored so this can be dropped into the same call site as the barrier.
        """
        p_ = pred.float()
        mu = p_.mean(dim=(1, 3, 4))
        sd = p_.std(dim=(1, 3, 4))
        comp = getattr(self, "ode_grep_components", "both")
        d2 = torch.zeros(p_.shape[0], device=p_.device, dtype=torch.float32)
        if comp in ("both", "mu"):
            d2 = d2 + mu.pow(2).sum(dim=1)
        if comp in ("both", "sigma"):
            d2 = d2 + (sd - 1.0).pow(2).sum(dim=1)
        # ADDITIVE softening, not a clamp (2026-08-14, noise-point campaign):
        # clamp(min) makes the loss FLAT inside the floor — zero gradient
        # exactly at the attractor, where a collapsed chunk needs the rescue
        # force most (defect caught by the tracker unit tests). The additive
        # form keeps the same bound (max L = 1/0.25 = 4 at d=0, so max dose
        # w*4 = 0.4) but stays smooth with nonzero gradient off-centre.
        # Healthy chunks (Sum mu^2 ~ 1.5-3.2 measured) pay ~w*0.03-0.06 —
        # regulariser scale — with force growing as the chunk nears mu=0.
        # RUNG GATING (user directive): the slot fires at EVERY rung EXCEPT
        # the first (see ode_rollout.py, `if i >= 1` on the gexcl fold) —
        # final-rung-only was "too little too late". Rung 0 stays excluded:
        # its pred_x0 comes from near-pure noise, closest to mu=0, where the
        # inverse-square force would be outsized.
        return (1.0 / (d2 + 0.25)).mean()

    def _grepdir_chunk(self, pred, target):
        """DIRECTIONAL mean repulsor: barrier on the projection of the chunk's
        per-channel means onto the TEACHER's mean direction.

            p = <mu_student, mu_teacher/||mu_teacher||>
            L = 1 / (max(p, 0)^2 + eps)

        Semantics: the collapse mode drains mean energy along the teacher's
        direction (measured: Sum mu^2 3.3->1.0 over the rollout); L blows up
        as that projection approaches zero -- the barrier kl_local's quadratic
        tracking lacks -- and is INDIFFERENT to components orthogonal to the
        teacher direction, so the isotropic form's color-offset cheat earns
        nothing. Wrong-sign (anti-aligned) projections clamp to 0 = maximum
        force. eps bounds the worst case (max L = 1/eps; early rungs sit
        closest to collapse and ran ~3x hot under the isotropic form).
        `target` is the teacher's committed chunk from the batch: a per-chunk
        MOVING reference with zero estimation machinery.
        """
        mu_s = pred.float().mean(dim=(1, 3, 4))            # [B, C]
        mu_t = target.float().mean(dim=(1, 3, 4))          # [B, C]
        that = mu_t / mu_t.norm(dim=1, keepdim=True).clamp(min=1e-6)
        p = (mu_s * that).sum(dim=1).clamp(min=0.0)        # [B]
        return (1.0 / (p.pow(2) + self.ode_grepdir_eps)).mean()

    def _gexcl_chunk(self, pred, target):
        """Teacher-referenced contraction barrier for one committed chunk.

        Per-channel log(sigma_pred / sigma_teacher) with a SQUARED HINGE
        outside [-band, +band]: exactly zero loss and zero gradient while the
        student's dispersion sits inside the teacher's band, real force once it
        contracts below it (and once it runs away above it). Satisfiable by
        construction -- unlike the 1/d^2 repulsors earlier in this campaign,
        which had no satisfied region and diverged, and unlike the N(0,1)
        reference, which we measured to be inert AND wrongly signed on this data
        (committed chunks sit at sigma~0.5, so contraction moves them AWAY from
        the gaussian and the old term rewarded it).
        """
        band = self.ode_gexcl_band
        p = pred.float()
        with torch.no_grad():
            sd_t = target.float().std(dim=(1, 3, 4)).clamp(min=1e-6)
        lr = (p.std(dim=(1, 3, 4)).clamp(min=1e-6) / sd_t).log()
        return (((-band) - lr).clamp(min=0).pow(2)
                + (lr - band).clamp(min=0).pow(2)).sum(dim=1).mean()

    def _action_error_weight(self, chain, z_actions):
        """Detached weight >= 1 from the CoTracker->PCA realised action.

        Built lazily so a run that never enables it pays nothing, and it fails
        SOFT: any problem (no CoTracker, chain too short, missing PCA table)
        returns None and the loss is left unweighted rather than crashing a
        multi-hour job.
        """
        from af_model.action_weight import realised_action_z, action_error_weight
        try:
            if self._actw_cotracker is None:
                import torch.hub
                self._actw_cotracker = torch.hub.load(
                    "facebookresearch/co-tracker", "cotracker3_offline"
                ).to(chain.device).eval()
                for _p in self._actw_cotracker.parameters():
                    _p.requires_grad_(False)
            if getattr(self, "_actw_pca", None) is None:
                import numpy as _np
                _ck = torch.load(str(self.config.ss_vae_checkpoint),
                                 map_location="cpu", weights_only=False)
                # The checkpoint has NO "pca_scales" key (it holds a single
                # scalar "scale"), so the old [1.0]*8 fallback ALWAYS fired:
                # tanh(P/1.0) with |P|~37 saturates to +-1 for every window,
                # so the error pinned at max_ratio and the "action weight"
                # was a constant 5.0 -- an LR change, not a weight. Use the
                # teacher's own per-component scales (2.5*std), the same
                # numbers 14e trains with (pca_raw_scales in its config).
                _sc = list(getattr(self.config, "pca_raw_scales",
                                   [93.7, 57.7, 22.5, 21.2, 18.1, 14.5, 12.6, 10.8]))[:8]
                self._actw_pca = (
                    torch.tensor(_np.asarray(_ck["pca_mean"]), dtype=torch.float32,
                                 device=chain.device),
                    torch.tensor(_np.asarray(_ck["pca_comp"]).T, dtype=torch.float32,
                                 device=chain.device),
                    torch.tensor(_sc, dtype=torch.float32, device=chain.device))
            pm, pc, ps = self._actw_pca
            rz = realised_action_z(
                chain_latents=chain, frozen_vae=self._frozen_vae,
                cotracker=self._actw_cotracker,
                pca_mean=pm, pca_comp_T=pc, pca_scales=ps)
            # commanded action over the GENERATED frames only
            cmd = z_actions[0, -chain.shape[1]:]
            return action_error_weight(
                realised_z=rz, commanded_z=cmd,
                # self.action_dims does not exist on this class -- reading it
                # raised AttributeError, which the broad except below caught,
                # so EVERY actw arm silently ran unweighted. Read the config.
                action_dims=tuple(
                    (getattr(self.config, 'action_dims', None) or (0, 1))[:2]),
                alpha=self.ode_actw_alpha, err_ref=self.ode_actw_ref,
                max_ratio=self.ode_actw_max)
        except Exception as e:                       # fail SOFT, never kill a run
            if not getattr(self, "_actw_warned", False):
                logging.warning("action-error weighting disabled (%s: %s)",
                                type(e).__name__, e)
                self._actw_warned = True
            return None

    def _probe_touch(self):
        """Zero-weighted touch of the state-probe params.

        DDP runs with find_unused_parameters=False, so every parameter in the
        wrapped module must receive a gradient. The state probe only fires at
        21/42 frames and this path forwards 3 at a time, so without this the
        run aborts with "Expected to have finished reduction in the prior
        iteration". Folded into the rollout's LAST backward.
        """
        _touch = None
        _probe = getattr(self._gen_base_module(), "_state_probe", None)
        if _probe is not None:
            for _pp in _probe.parameters():
                _t = 0.0 * _pp.sum()
                _touch = _t if _touch is None else _touch + _t
        return _touch

    def rollout_loss(self, batch, step: int = 0):
        """KV-cache AR rollout stage — see af_model/ode_rollout.py.

        Runs the clean (no-op) and cf (action) chains through the SAME cache
        path the teacher used to generate their targets, at the student's 4
        rungs, supervising every generated frame of every chunk.
        """
        from af_model.ode_rollout import rollout_ode_loss
        # The rollout path SUMS the branch losses; lambda_cf is not wired in
        # (each branch backwards internally, so a late multiply would be a
        # silent no-op). Fail fast rather than silently ignore a non-default
        # value (review M1).
        _lam = float(getattr(self.config, "lambda_cf", 1.0))
        if _lam != 1.0:
            raise RuntimeError(
                f"ode_rollout ignores lambda_cf (got {_lam}); it must be 1.0. "
                "To weight the cf branch, scale it inside rollout_ode_loss "
                "via loss_scale instead.")
        wrapper = self.generator
        dev, dt = self.device, self.dtype
        pe = batch["prompt_embeds"].to(dev, dt)
        seed = batch["seed_lat"].to(dev)
        # Assert rather than slice: [:1] would silently discard 50-75% of a
        # batch whenever the 8-direction group does not split 1-per-rank.
        _B = int(seed.shape[0])
        if _B != 1:
            raise RuntimeError(
                f"ode_rollout supports a per-rank batch of 1; got {_B}. The "
                "grouped sampler yields max(len(group)//world,1), so this "
                "means world_size < group size (need >= 8 ranks for dir8n).")
        logs, total, n = {}, None, 0
        per_chunk_all = []
        _scale_next = 1.0
        # Direction bin for the attractor tracker: clean branch is always the
        # no-op chain (bin 8 = cN, chunked_ode_dataset._ROLL_DIRS); cf uses the
        # dataset's dir_idx (defensive extraction mirrors the trainer's).
        _ad = batch.get("meta", {}).get("dir_idx", -1)
        _cf_bin = (int(torch.as_tensor(_ad).reshape(-1)[0].item())
                   if _ad is not None else -1)
        for tag in ("clean", "cf"):
            z = batch[f"z_{tag}"].to(dev, dt)
            tgt = batch[f"committed_{tag}"].to(dev)
            sb = int(batch[f"noise_seed_{tag}"].reshape(-1)[0].item())
            # edist ONLY on the cf branch: every rank of a group shares the
            # SAME clean file, so the clean branch's gathered set is identical
            # across ranks -> zero spread -> 1e8 feature amplification.
            _use_ed = (tag == "cf") and self.ode_edist_weight > 0.0
            _bin = 8 if tag == "clean" else _cf_bin
            # Per-chunk slot precedence: attractor repulsor > grep > gexcl
            # (mutual exclusion enforced at init). The lambda binds THIS
            # branch's direction bin; target arg accepted and ignored to match
            # the slot signature.
            if self.attractor is not None:
                _gfn = (lambda p_, t_, _b=_bin: self.attractor.repulsor(p_, _b))
                _gw = self.attractor.weight
            elif self.ode_grepdir_weight > 0.0:
                _gfn, _gw = self._grepdir_chunk, self.ode_grepdir_weight
            elif self.ode_grep_weight > 0.0:
                _gfn, _gw = self._grep_chunk, self.ode_grep_weight
            elif self.ode_gexcl_weight > 0.0:
                _gfn, _gw = self._gexcl_chunk, self.ode_gexcl_weight
            else:
                _gfn, _gw = None, 0.0
            out = rollout_ode_loss(
                wrapper, self.action_projection, self.action_token_projection,
                wrapper_call=self.generator,
                prompt_embeds=pe[:1], seed_lat=seed[:1], z_actions=z[:1],
                committed=tgt[:1],
                denoising_step_list=self.denoising_step_list,
                scheduler=self.scheduler, dtype=dt, device=dev,
                nfb=self.num_frame_per_block,
                seed_base=sb,
                commit_mode=self.ode_rollout_commit,
                commit_p=self.ode_rollout_commit_p,
                # group_id is REQUIRED, no sample_idx fallback (review M4):
                # unique-per-sample ids degenerate the within-group mask to
                # the identity and the repulsion term silently vanishes.
                edist_fn=((lambda p_, t_: self.edist_chunk(
                    p_, t_, group_id=int(
                        batch['group_id'].reshape(-1)[0].item())))
                    if _use_ed else None),
                edist_weight=(self.ode_edist_weight if _use_ed else 0.0),
                gexcl_fn=_gfn,
                gexcl_weight=_gw,
                klts_theta=getattr(self, "klts_theta", None),
                # Keep the target in fp32: t.to(p.dtype) would round the
                # teacher's committed latents to bf16 under autocast before the
                # loss casts both back to float.
                loss_fn=(lambda p, t: self._compute_ode_loss(
                    p.float(), t.float(),
                    torch.full(p.shape[:2], 1.0, device=p.device))),
                ddp_module=(self.generator if hasattr(self.generator, "no_sync")
                            else None),
                vrfm=self.vrfm, z_modulation=self.z_modulation,
                vrfm_beta=(self.ode_vrfm_beta if self.ode_vrfm else 0.0),
                # Only the LAST backward of the LAST branch may all-reduce.
                sync_last=(tag == "cf"),
                # The action/variance weights need the FULL chain, which does
                # not exist until the rollout has finished -- but the per-rung
                # backwards happen DURING it. Apply the previous step's weight
                # instead: it is a slowly-varying scalar with a rolling-mean
                # reference, so a one-step lag is immaterial. Multiplying the
                # returned scalar (as before) would now be a silent no-op,
                # because the gradients are already applied.
                loss_scale=float(getattr(self, "_rollout_scale_prev", 1.0)),
                tail_loss_fn=self._probe_touch,   # every armed backward must mark all params ready
                es=self.ode_es, es_repw=self.ode_es_repw,
            )
            # Attractor estimator feed: the committed chain (detached) plus the
            # teacher targets, binned by this branch's direction. The clean
            # branch is the SAME no-op chain on all ranks of a curriculum
            # group (max group size 8), so it is down-weighted to avoid the
            # SUM all-reduce counting the identical observation 8x (review M4).
            if self.attractor is not None and out.get("chain") is not None:
                self.attractor.observe(out["chain"], tgt[:1], _bin,
                                       weight=(0.125 if tag == "clean" else 1.0))
            if self.ode_es:
                # mean rms pair-distance between the two noise branches — the
                # spread the score is defending; watch it against the teacher's
                # own draw-to-draw spread rather than letting it run dark.
                logs[f"es_rep_{tag}"] = float(out.get("es_rep", 0.0))
            _w_act = None
            if self.ode_actw_enabled and tag == "cf" and out.get("chain") is not None:
                _w_act = self._action_error_weight(out["chain"], z[:1])
                if _w_act is not None:
                    _scale_next = _scale_next * float(_w_act)
                    logs["actw"] = float(_w_act)
                    logs["act_residual"] = float(_w_act) - 1.0
            if (self.ode_varw_enabled and out.get("chain") is not None
                    and tag == "cf"):
                from af_model.action_weight import variance_error_weight
                _tc = tgt[:1].reshape(1, -1, *tgt.shape[-3:])
                _n = min(out["chain"].shape[1], _tc.shape[1])
                _w_var = variance_error_weight(
                    pred_chain=out["chain"][:, :_n], teacher_chain=_tc[:, :_n],
                    alpha=self.ode_varw_alpha, ref=self.ode_varw_ref)
                _scale_next = _scale_next * float(_w_var)
                logs["varw"] = float(_w_var)
            total = out["loss"] if total is None else total + out["loss"]
            n += 1
            per_chunk_all.append(out["per_chunk"])
            logs[f"ode_loss_{tag}"] = out["loss"].detach()
            if out.get("kl_z"):
                logs[f"vrfm_kl_{tag}"] = float(out["kl_z"])
            for _k, _v in (out.get("vrfm_stats") or {}).items():
                logs[f"vrfm_{_k}_{tag}"] = float(_v)
        # Fold this step's attractor observations + (periodically) re-freeze
        # the actuation reference. Runs on EVERY rank every step (collective).
        if self.attractor is not None:
            self.attractor.sync(step)
            logs.update(self.attractor.log_summary())
        loss = total          # SUM (matches loss_clean + lambda_cf*loss_cf), not a mean
        # DDP find_unused_parameters=False: every parameter in the wrapped
        # module must receive a gradient. The state probe only fires at 21/42
        # frames, and this path forwards 3 at a time, so touch its params with a
        # zero-weighted term (same trick the packed path uses) or DDP aborts
        # with "Expected to have finished reduction in the prior iteration".
        # NOTE: under the rollout the probe touch is handed to the rollout as
        # `tail_loss_fn` and folded into the LAST backward (see below) -- adding
        # it here would be a no-op, since `loss` is already detached, and DDP
        # would abort on the untouched probe params.
        # Carry the action/variance weight to the NEXT step (see loss_scale).
        logs["backwarded"] = True
        # Log the scale that was ACTUALLY applied to this step's gradients
        # (the previous step's weight), BEFORE overwriting it for the next.
        logs["rollout_scale_applied"] = float(
            getattr(self, "_rollout_scale_prev", 1.0))
        self._rollout_scale_prev = float(_scale_next)
        logs["rollout_per_chunk_clean"] = per_chunk_all[0]
        logs["rollout_per_chunk_cf"] = per_chunk_all[-1]
        # curriculum hook: difficulty of THIS (context, direction) = mean error
        # of the cf chain's chunks
        logs["per_sample_err_cf"] = torch.tensor(
            [sum(per_chunk_all[-1]) / max(len(per_chunk_all[-1]), 1)], device=dev)
        logs["per_sample_rung_cf"] = torch.tensor(
            [float(self.denoising_step_list[-1])], device=dev)
        return loss, logs

    def generator_loss(
        self,
        trajectory_clean: torch.Tensor,      # [B_pair, T_snap, F, C, H, W]
        trajectory_cf: torch.Tensor,         # [B_pair, T_snap, F, C, H, W]
        prompt_embeds: torch.Tensor,         # [B_pair, 512, 4096]
        z_clean: torch.Tensor,               # [B_pair, F, 2]   (shared context action)
        z_noisy: torch.Tensor,               # [B_pair, F, 2]   (clean-branch action)
        z_noisy_cf: torch.Tensor,            # [B_pair, F, 2]   (CF-branch action)
        clean_x_gt: torch.Tensor,            # [B_pair, F, C, H, W]
        clean_x_gt_cf: torch.Tensor = None,  # optional CF-chain clean window
        step: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Run a full paired training-forward pass (random-timestep recipe).

        Clean and CF branches are *packed* into a single DiT call of batch
        size ``2 * B_pair`` (slot 0..B_pair-1 = clean; slot B_pair..2*B_pair-1
        = CF). This is the only path — there is no two-pass fallback.
        OOM at B_pair=1 raises loudly rather than silently falling back.

        Shared tensors (``prompt_embeds``, ``z_clean``, ``clean_x_gt``) are
        duplicated along the batch dim so the DiT sees a coherent B=2*B_pair
        mini-batch. The noisy-window streams (``_action_modulation`` +
        ``_action_tokens``) are built from ``stack([z_noisy; z_noisy_cf])``.

        Critic reading, state-probe supervision, motion-pipeline targets,
        and per-chunk ODE diagnostics are computed per-slot after slicing
        and then combined into a single ``L = L_clean + lambda_cf * L_cf``.

        Returns:
          ``(loss, log_dict)`` where ``log_dict`` carries branch-prefixed
          ``ode_loss_{clean,cf}`` / ``state_loss_{clean,cf}`` /
          ``critic_guidance_loss_{clean,cf}`` etc. as well as
          ``pred_x0_detached_{clean,cf}`` + ``teacher_z_8d_{clean,cf}``
          so the trainer can drive the critic's own mini-loop.
        """
        assert trajectory_clean.shape == trajectory_cf.shape, (
            f"clean/CF trajectory shape mismatch: "
            f"{trajectory_clean.shape} vs {trajectory_cf.shape}"
        )
        B_pair = trajectory_clean.shape[0]
        F_ = trajectory_clean.shape[2]
        lam_cf = float(getattr(self.config, "lambda_cf", 1.0))

        # ---- Per-branch independent noise-level draw.
        # Clean and CF are separate training signals; sampling their
        # timesteps independently gives the student broader coverage of
        # the (clean-level x CF-level) joint space. The alternative
        # (shared ``pool_idx`` across branches) would tie the two
        # together and halve that coverage.
        noisy_clean, t_clean, pool_idx_clean = self._prep_random(trajectory_clean)
        noisy_cf,    t_cf,    pool_idx_cf    = self._prep_random(trajectory_cf)

        target_clean = self._resolve_target(trajectory_clean)
        target_cf    = self._resolve_target(trajectory_cf)
        # Committed-sample copies kept for the moment loss: nr tangent
        # targets at high rungs are extrapolations whose OWN std is
        # mean-like (~0.63 vs sample 0.88) — matching moments to them
        # reproduces the collapse (measured: nrmom rung-0 std 0.714 =
        # nr's 0.719). ode_moment_ref=committed matches moments to the
        # SAMPLE-level committed block instead.
        target_clean_committed = target_clean
        target_cf_committed    = target_cf
        if getattr(self, "ode_nextrung_targets", False):
            target_clean = self._nextrung_targets(
                trajectory_clean, target_clean, noisy_clean, pool_idx_clean)
            target_cf = self._nextrung_targets(
                trajectory_cf, target_cf, noisy_cf, pool_idx_cf)

        # ---- Casts (every cast happens exactly once; these are the
        # tensors the DiT actually consumes).
        dtype = self.dtype
        noisy_clean = noisy_clean.to(dtype)
        noisy_cf    = noisy_cf.to(dtype)
        clean_x_in  = clean_x_gt.to(dtype)
        # The teacher-forced context is ALWAYS truly clean. The clean context is
        # NEVER noised under any config (hard user rule) — there is no
        # context-augmentation path. ``aug_t`` is always None.
        prompt_cast = prompt_embeds.to(dtype)
        z_clean_c   = z_clean.to(dtype)
        z_noisy_c   = z_noisy.to(dtype)
        z_noisy_cfc = z_noisy_cf.to(dtype)

        # ---- Pack along batch dim ----
        # ``_build_conditional`` is batch-agnostic so we can stack the
        # *per-branch* action streams before calling it; the projections
        # run on the packed [2*B_pair, ...] in one shot.
        prompt_pack  = torch.cat([prompt_cast,  prompt_cast],  dim=0)
        clean_x_cf_in = (clean_x_gt_cf.to(dtype)
                         if clean_x_gt_cf is not None else clean_x_in)
        clean_x_pack = torch.cat([clean_x_in,   clean_x_cf_in], dim=0)
        if self.ode_action_cfg_dropout > 0.0:
            _keep = (torch.rand(z_noisy_c.shape[0], 1, 1,
                                device=z_noisy_c.device)
                     >= self.ode_action_cfg_dropout).to(z_noisy_c.dtype)
            z_noisy_c = z_noisy_c * _keep
            _keep_cf = (torch.rand(z_noisy_cfc.shape[0], 1, 1,
                                   device=z_noisy_cfc.device)
                        >= self.ode_action_cfg_dropout).to(z_noisy_cfc.dtype)
            z_noisy_cfc = z_noisy_cfc * _keep_cf
        z_noisy_pack = torch.cat([z_noisy_c,    z_noisy_cfc],  dim=0)
        z_clean_pack = torch.cat([z_clean_c,    z_clean_c],    dim=0)
        noisy_pack   = torch.cat([noisy_clean,  noisy_cf],     dim=0)
        t_pack       = torch.cat([t_clean,      t_cf],         dim=0)
        aug_t_pack   = None   # clean context is never noised

        conditional = self._build_conditional(
            prompt_pack, z_noisy_pack, z_clean_pack, num_frames=F_,
        )

        # ---- Single DiT forward of batch 2*B_pair ----
        model_out = self.generator(
            noisy_image_or_video=noisy_pack,
            conditional_dict=conditional,
            timestep=t_pack,
            clean_x=clean_x_pack,
            aug_t=aug_t_pack,
        )
        # Dead-code removal (teacher parity): ``__init__`` enforces
        # ``state_probe_mode=True``, so the wrapper must return a 4-tuple
        # ``(flow_pred, pred_x0, state_preds, probe_hidden)``. Anything
        # shorter indicates a silent wrapper drift and is fatal — better
        # to crash now than to train for hours with ``state_guidance_loss``
        # / ``state_probe_loss`` mysteriously muted.
        if not (isinstance(model_out, tuple) and len(model_out) == 4):
            raise RuntimeError(
                f"Expected 4-tuple (flow_pred, pred_x0, state_preds, "
                f"probe_hidden) from generator; got "
                f"{type(model_out).__name__} of length "
                f"{len(model_out) if isinstance(model_out, tuple) else 'N/A'}. "
                "The student enforces state_probe_mode=True — a non-4 "
                "return means the state probe attachment regressed."
            )
        flow_pred, pred_x0, state_preds, probe_hidden = model_out

        pred_clean = pred_x0[:B_pair]
        pred_cf    = pred_x0[B_pair:]
        state_clean = state_preds[:B_pair]
        state_cf    = state_preds[B_pair:]
        probe_hidden_clean = probe_hidden[:B_pair]
        probe_hidden_cf    = probe_hidden[B_pair:]

        # ---- ODE regression per slot ----
        if getattr(self, "ode_chunked_supervision", False):
            _nfb = self.num_frame_per_block
            ode_loss_clean = self._compute_ode_loss(
                pred_clean[:, -_nfb:], target_clean[:, -_nfb:].to(dtype),
                t_clean[:, -_nfb:],
            )
            ode_loss_cf = self._compute_ode_loss(
                pred_cf[:, -_nfb:],    target_cf[:, -_nfb:].to(dtype),
                t_cf[:, -_nfb:],
            )
            if self.ode_moment_loss_weight > 0.0:
                # Per-channel moments over (frames, H, W) of the target
                # block; grad flows through pred only.
                def _mml(p, tgt):
                    pf = p.to(torch.float32)
                    tf = tgt.to(torch.float32)
                    dims = (1, 3, 4)
                    dmu = pf.mean(dim=dims) - tf.mean(dim=dims)
                    dsd = pf.std(dim=dims) - tf.std(dim=dims)
                    return (dmu.pow(2).mean() + dsd.pow(2).mean()).to(dtype)
                if getattr(self.config, "ode_moment_ref", "target") == "committed":
                    _mm_c = _mml(pred_clean[:, -_nfb:], target_clean_committed[:, -_nfb:])
                    _mm_f = _mml(pred_cf[:, -_nfb:],    target_cf_committed[:, -_nfb:])
                else:
                    _mm_c = _mml(pred_clean[:, -_nfb:], target_clean[:, -_nfb:])
                    _mm_f = _mml(pred_cf[:, -_nfb:],    target_cf[:, -_nfb:])
                ode_loss_clean = ode_loss_clean + self.ode_moment_loss_weight * _mm_c
                ode_loss_cf    = ode_loss_cf    + self.ode_moment_loss_weight * _mm_f
            if self.ode_noiserep_loss_weight > 0.0:
                # ALWAYS-ON inverse-square repulsor from the gaussian
                # signature (mu=0, sigma=1): L = w / D with D = the
                # SQUARED signature distance, so the repulsion grows as
                # 1/d^2 the closer the prediction sits to the noise
                # point and never switches off (user-specified form).
                def _sigdist2(x):
                    xf = x.to(torch.float32)
                    mu = xf.mean(dim=(1, 3, 4))
                    sd = xf.std(dim=(1, 3, 4))
                    return (mu.pow(2).sum(dim=1)
                            + (sd - 1.0).pow(2).sum(dim=1)).clamp(min=1e-4)
                _rep = ((1.0 / _sigdist2(pred_clean[:, -_nfb:])).mean()
                        + (1.0 / _sigdist2(pred_cf[:, -_nfb:])).mean())
                ode_loss_clean = ode_loss_clean + self.ode_noiserep_loss_weight * (_rep / 2.0).to(dtype)
            if (self.ode_emdrep_loss_weight > 0.0
                    or self.ode_emdrep_delta_weight > 0.0
                    or self.ode_emdhead_weight > 0.0
                    or self.ode_emdhead_delta_weight > 0.0):
                def _emd2(x):
                    # Squared per-channel 1-D W2 to N(0,1), summed over
                    # channels: sort values over (frames, H, W), match
                    # against standard-normal quantiles. Exact 1-D OT;
                    # differentiable through the sort.
                    xf = x.to(torch.float32)
                    b, f, c, h, w = xf.shape
                    v = xf.permute(0, 2, 1, 3, 4).reshape(b, c, -1)
                    v, _ = v.sort(dim=-1)
                    n = v.shape[-1]
                    if (self._emd_quantiles is None
                            or self._emd_quantiles.shape[0] != n
                            or self._emd_quantiles.device != v.device):
                        p = (torch.arange(n, device=v.device,
                                          dtype=torch.float32) + 0.5) / n
                        self._emd_quantiles = (
                            torch.erfinv(2.0 * p - 1.0) * (2.0 ** 0.5))
                    q = self._emd_quantiles
                    return (v - q).pow(2).mean(dim=-1).sum(dim=1) \
                        .clamp(min=1e-4)
                # Commit-rung masks: True where the sample's rung is the
                # final one (t=208.33), i.e. where pred_x0 IS the chunk
                # the 4-rung sampler would commit.
                _cm_c = t_clean[:, -1] < 250.0
                _cm_f = t_cf[:, -1] < 250.0
                def _msel(x, m):
                    return x[m] if self.ode_emdrep_commit_only else x
                if self.ode_emdrep_loss_weight > 0.0:
                    # v1: inverse-square EMD repulsor (commit-only when
                    # ode_emdrep_commit_only; else every rung).
                    _terms = []
                    for _pp, _mm in ((pred_clean[:, -_nfb:], _cm_c),
                                     (pred_cf[:, -_nfb:], _cm_f)):
                        _ps = _msel(_pp, _mm)
                        if _ps.shape[0] > 0:
                            _terms.append((1.0 / _emd2(_ps)).mean())
                    if _terms:
                        _er = sum(_terms) / len(_terms)
                        ode_loss_clean = ode_loss_clean \
                            + self.ode_emdrep_loss_weight * _er.to(dtype)
                if self.ode_emdrep_delta_weight > 0.0:
                    # v2: penalize contraction of d across chunks. d_prev
                    # from the COMMITTED context chunk directly preceding
                    # the target block (constant — grad flows only
                    # through the current chunk's d).
                    _eps = self.ode_emdrep_delta_eps
                    _terms = []
                    for _pp, _tc, _mm in (
                            (pred_clean[:, -_nfb:],
                             target_clean_committed[:, -2 * _nfb:-_nfb], _cm_c),
                            (pred_cf[:, -_nfb:],
                             target_cf_committed[:, -2 * _nfb:-_nfb], _cm_f)):
                        _ps = _msel(_pp, _mm)
                        if _ps.shape[0] == 0:
                            continue
                        with torch.no_grad():
                            _dp = _emd2(_msel(_tc, _mm)).sqrt()
                        _dd = (_emd2(_ps).sqrt() - _dp).clamp(min=_eps)
                        _terms.append((1.0 / _dd).mean())
                    if _terms:
                        _ed = sum(_terms) / len(_terms)
                        ode_loss_clean = ode_loss_clean \
                            + self.ode_emdrep_delta_weight * _ed.to(dtype)
            if self.ode_gexcl_weight > 0.0:
                # Bounded gaussian exclusion zone on the COMMITTED chunk
                # (commit clock: applied where the statistic exists).
                _cmg_c = t_clean[:, -1] < 250.0
                _cmg_f = t_cf[:, -1] < 250.0
                _band = self.ode_gexcl_band
                _gt = []
                for _pp, _tt, _mm in (
                        (pred_clean[:, -_nfb:],
                         target_clean_committed[:, -_nfb:], _cmg_c),
                        (pred_cf[:, -_nfb:],
                         target_cf_committed[:, -_nfb:], _cmg_f)):
                    if int(_mm.sum()) == 0:
                        continue
                    _p = _pp[_mm].to(torch.float32)
                    with torch.no_grad():
                        _sd_t = _tt[_mm].to(torch.float32).std(
                            dim=(1, 3, 4)).clamp(min=1e-6)
                    _lr = (_p.std(dim=(1, 3, 4)).clamp(min=1e-6) / _sd_t).log()
                    # squared hinge OUTSIDE [-band, +band]; exactly zero
                    # loss and zero gradient inside -> satisfiable.
                    _gt.append((((-_band) - _lr).clamp(min=0).pow(2)
                                + (_lr - _band).clamp(min=0).pow(2)
                                ).sum(dim=1).mean())
                if _gt:
                    ode_loss_clean = ode_loss_clean + self.ode_gexcl_weight \
                        * (sum(_gt) / len(_gt)).to(dtype)
            if self.ode_edist_weight > 0.0:
                _ed_loss = self._energy_distance_loss(
                    pred_clean[:, -_nfb:], target_clean_committed[:, -_nfb:],
                    pred_cf[:, -_nfb:], target_cf_committed[:, -_nfb:],
                    t_clean[:, -1], t_cf[:, -1])
                if _ed_loss is not None:
                    ode_loss_clean = ode_loss_clean \
                        + self.ode_edist_weight * _ed_loss.to(dtype)
            if (self.ode_emdhead_weight > 0.0
                    or self.ode_emdhead_delta_weight > 0.0):
                # Zero-init transport head: EMD objectives on the head-
                # corrected DETACHED commit-rung prediction. Flow map is
                # untouched; only s,b learn. The 0.0* touch guarantees
                # head grads exist on EVERY rank each step (manual DDP
                # all-reduce would desync otherwise).
                _cm_c = t_clean[:, -1] < 250.0
                _cm_f = t_cf[:, -1] < 250.0
                _hs = self.emd_head_scale.to(torch.float32)
                _hb = self.emd_head_shift.to(torch.float32)
                def _hcorr(x):
                    return (x.to(torch.float32)
                            * (1.0 + _hs.view(1, 1, -1, 1, 1))
                            + _hb.view(1, 1, -1, 1, 1))
                _hterms = []
                for _pp, _tc, _mm in (
                        (pred_clean[:, -_nfb:],
                         target_clean_committed[:, -2 * _nfb:-_nfb], _cm_c),
                        (pred_cf[:, -_nfb:],
                         target_cf_committed[:, -2 * _nfb:-_nfb], _cm_f)):
                    if int(_mm.sum()) == 0:
                        continue
                    _base = _pp[_mm].detach()
                    _ch = _hcorr(_base)
                    if self.ode_emdhead_objective == "deadband":
                        _dims = (1, 3, 4)
                        with torch.no_grad():
                            _prev = _tc[_mm].to(torch.float32)
                            _sd_r = _prev.std(dim=_dims).clamp(min=1e-6)
                            _mu_r = _prev.mean(dim=_dims)
                        _lr = (_ch.std(dim=_dims).clamp(min=1e-6)
                               / _sd_r).log()
                        _lsd = ((-_lr).clamp(min=0).pow(2)
                                + (_lr - self.ode_emdhead_band_hi)
                                .clamp(min=0).pow(2)).sum(dim=1).mean()
                        _lmu = ((_ch.mean(dim=_dims) - _mu_r).abs()
                                - self.ode_emdhead_mu_band) \
                            .clamp(min=0).pow(2).sum(dim=1).mean()
                        _lid = (_ch - _base.to(torch.float32)).pow(2).mean()
                        _hterms.append(
                            self.ode_emdhead_weight * (_lsd + _lmu)
                            + self.ode_emdhead_id_weight * _lid)
                        continue
                    _d2 = _emd2(_ch)
                    if self.ode_emdhead_weight > 0.0:
                        _hterms.append(
                            self.ode_emdhead_weight * (1.0 / _d2).mean())
                    if self.ode_emdhead_delta_weight > 0.0:
                        with torch.no_grad():
                            _dp = _emd2(_tc[_mm]).sqrt()
                        _dd = (_d2.sqrt() - _dp).clamp(
                            min=self.ode_emdrep_delta_eps)
                        _hterms.append(
                            self.ode_emdhead_delta_weight * (1.0 / _dd).mean())
                _htouch = 0.0 * (_hs.sum() + _hb.sum())
                ode_loss_clean = ode_loss_clean \
                    + (_htouch + (sum(_hterms) if _hterms else 0.0)).to(dtype)
            if self.ode_actdelta_loss_weight > 0.0:
                _dp = (pred_clean[:, -_nfb:].to(torch.float32)
                       - pred_cf[:, -_nfb:].to(torch.float32))
                _dt = (target_clean_committed[:, -_nfb:].to(torch.float32)
                       - target_cf_committed[:, -_nfb:].to(torch.float32))
                _num = (_dp - _dt).flatten(1).norm(dim=1)
                _den = _dt.flatten(1).norm(dim=1) + 1e-6
                _ad = (_num / _den).pow(2).mean().to(dtype)
                ode_loss_clean = ode_loss_clean + self.ode_actdelta_loss_weight * _ad
            if self.ode_sep_loss_weight > 0.0:
                _pc = pred_clean[:, -_nfb:].to(torch.float32)
                _pf = pred_cf[:, -_nfb:].to(torch.float32)
                _tc = target_clean_committed[:, -_nfb:].to(torch.float32)
                _tf_ = target_cf_committed[:, -_nfb:].to(torch.float32)
                _dp = (_pc - _pf).flatten(1).norm(dim=1)
                _dt = (_tc - _tf_).flatten(1).norm(dim=1)
                _sep = ((_dp - _dt) / (_dt + 1e-6)).pow(2).mean().to(dtype)
                ode_loss_clean = ode_loss_clean + self.ode_sep_loss_weight * _sep
        else:
            ode_loss_clean = self._compute_ode_loss(
                pred_clean, target_clean.to(dtype), t_clean,
            )
            ode_loss_cf = self._compute_ode_loss(
                pred_cf,    target_cf.to(dtype),    t_cf,
            )

        # ---- Motion pipeline: batched call over the full 2*B_pair pack.
        # Slicing afterward keeps the state-probe and critic targets per slot.
        # We *require* the motion pipeline to be live here: with it off
        # the state probe silently falls back to commanded-z supervision
        # and the critic update loop returns an empty dict, so the ODE
        # MSE curve keeps going down while the action signal trains on
        # nothing. ``__init__`` already hard-fails on
        # ``use_motion_pipeline=False``; this assert catches the only
        # remaining failure mode — ``ensure_motion_pipeline()`` never got
        # called or silently short-circuited before the first step.
        if not (self.use_motion_pipeline and self._motion_pipeline_ready):
            raise RuntimeError(
                "Motion pipeline is not ready at generator_loss time "
                f"(use_motion_pipeline={self.use_motion_pipeline}, "
                f"_motion_pipeline_ready={self._motion_pipeline_ready}). "
                "Call ``ensure_motion_pipeline()`` after distributed "
                "rendezvous — without it, ``_compute_action_teacher_targets`` "
                "never runs, the state probe silently falls back to "
                "commanded-z supervision, and the action critic sees no "
                "update. The ODE MSE would continue to decrease while the "
                "entire action-forcing signal is a no-op."
            )
        teacher_z_8d_all = self._compute_action_teacher_targets(pred_x0.detach())
        teacher_z_8d_clean = teacher_z_8d_all[:B_pair]
        teacher_z_8d_cf    = teacher_z_8d_all[B_pair:]

        n_c = self.n_chunks
        chunk_t_clean = t_clean[:, :: self.num_frame_per_block][:, :n_c]
        chunk_t_cf    = t_cf[:, :: self.num_frame_per_block][:, :n_c]

        # Critic conditioning = the commanded action for the *noisy window*
        # being denoised. Clean branch uses ``z_noisy``; CF branch uses
        # ``z_noisy_cf``. Using ``z_clean`` here was a bug that actively
        # penalised the generator for following the CF edit.
        chunk_actions_clean = _chunk_actions(z_noisy_c,   self.num_frame_per_block)[:, :n_c]
        chunk_actions_cf    = _chunk_actions(z_noisy_cfc, self.num_frame_per_block)[:, :n_c]

        # ---- State probe loss per slot ----
        state_loss_clean = self._state_probe_loss(
            state_clean[:, :n_c], teacher_z_8d_clean, chunk_actions_clean,
        )
        state_loss_cf = self._state_probe_loss(
            state_cf[:, :n_c], teacher_z_8d_cf, chunk_actions_cf,
        )

        # ---- Frozen-readout state guidance loss per slot (teacher parity) ----
        # Dormant when ``state_guidance_weight <= 0`` (the config default);
        # pushes the DiT's pooled hidden features toward action-readable
        # content once enabled. Mirrors v14's
        # ``trainer/causal_diffusion_teacher_train.py:2038-2056``.
        state_g_loss_clean, state_g_scale_clean = self._compute_state_guidance_loss(
            probe_hidden_clean[:, :n_c], chunk_actions_clean, step=step,
        )
        state_g_loss_cf, state_g_scale_cf = self._compute_state_guidance_loss(
            probe_hidden_cf[:, :n_c], chunk_actions_cf, step=step,
        )

        # ---- Critic-guidance loss per slot (gradient through pred_x0) ----
        critic_mask_clean = self._critic_chunk_mask(B_pair, pool_idx_clean)
        critic_mask_cf    = self._critic_chunk_mask(B_pair, pool_idx_cf)
        critic_guidance_loss_clean, guidance_scale_clean = self._compute_critic_guidance_loss(
            pred_clean, chunk_t_clean, chunk_actions_clean, critic_mask_clean, step=step,
        )
        critic_guidance_loss_cf, guidance_scale_cf = self._compute_critic_guidance_loss(
            pred_cf,    chunk_t_cf,    chunk_actions_cf,    critic_mask_cf,    step=step,
        )

        # ---- Combine ----
        loss_clean = (
            ode_loss_clean
            + self.state_head_loss_weight * state_loss_clean
            + state_g_loss_clean
            + critic_guidance_loss_clean
        )
        loss_cf = (
            ode_loss_cf
            + self.state_head_loss_weight * state_loss_cf
            + state_g_loss_cf
            + critic_guidance_loss_cf
        )
        loss = loss_clean + lam_cf * loss_cf

        # ---- Online dual-CD (ODE-F): frozen-14d teacher-CD + student
        # self-CD, computed on the CLEAN branch (the GT-anchored target).
        # Riding on the ODE loss above so neither can collapse. Warmup-ramped.
        _cd_logs: Dict[str, Any] = {}
        if self.cd_teacher_loss_enabled or self.cd_student_loss_enabled:
            self._update_cd_ema_and_teacher()
            cond_cd = self._build_conditional(
                prompt_cast, z_noisy_c, z_clean_c, num_frames=F_,
            )
            cd_teacher, cd_student, _cd_logs = self._compute_cd_losses(
                x0_cd=target_clean, conditional=cond_cd,
                clean_x=clean_x_in, aug_t=None,
            )
            cd_w = self._cd_current_weight(step)
            cd_term = (
                self.cd_teacher_loss_weight * cd_teacher
                + self.cd_student_loss_weight * cd_student
            )
            loss = loss + cd_w * cd_term
            _cd_logs["cd_weight_effective"] = float(cd_w)
            _cd_logs["cd_term_weighted"] = float((cd_w * cd_term).detach().item())

        # ---- On-policy teacher supervision (chunked mode only) ----
        # Re-noise the student's own target-block x0 to a sampled mid rung,
        # keep the committed context + its pinned t=0 timesteps untouched,
        # and let the FROZEN teacher produce a one-step x0 correction of the
        # student's content under the same commanded actions. The student
        # regresses toward that correction — gradients flow only through
        # p_pack (the main forward's pred), so this costs one extra no-grad
        # DiT forward per step.
        ts_sup_clean = torch.zeros((), device=self.device, dtype=dtype)
        ts_sup_cf = torch.zeros((), device=self.device, dtype=dtype)
        ts_t_corr = 0.0
        if (self.ode_teachersup_enabled
                and getattr(self, "ode_chunked_supervision", False)
                and self._cd_teacher is not None):
            _nfb = self.num_frame_per_block
            _ri = int(torch.randint(len(self.ode_teachersup_rungs), (1,)).item())
            ts_t_corr = self.ode_teachersup_rungs[_ri]
            p_pack = pred_x0[:, -_nfb:]                    # [2B, nfb, C, H, W], grad
            B2 = p_pack.shape[0]
            with torch.no_grad():
                eps_ts = torch.randn_like(p_pack.float())
                t_blk = torch.full(
                    (B2, _nfb), float(ts_t_corr),
                    device=p_pack.device, dtype=torch.float32)
                x_ren = self.scheduler.add_noise(
                    p_pack.detach().float().flatten(0, 1),
                    eps_ts.flatten(0, 1), t_blk.flatten(0, 1),
                ).unflatten(0, (B2, _nfb)).to(dtype)
                corr_pack = noisy_pack.clone()
                t_corr_pack = t_pack.clone()
                # steps=1: single x0 readout (conditional mean). steps>1:
                # walk the teacher down the REAL 20-step shift-5 grid tail
                # below t_corr — the final x0 readout is then an actual
                # teacher sample of the student's content, sharp.
                _grid = [1000.0 * (5.0 * u) / (1.0 + 4.0 * u)
                         for u in (1.0 - i / 20.0 for i in range(20))]
                _tail = [t for t in _grid if t < ts_t_corr - 1e-4]
                _tail = _tail[: max(self.ode_teachersup_steps - 1, 0)]
                x_cur, t_cur = x_ren, float(ts_t_corr)
                tea_x0 = None
                for t_nxt in _tail + [None]:
                    corr_pack[:, -_nfb:] = x_cur
                    t_corr_pack[:, -_nfb:] = torch.full(
                        (B2, _nfb), t_cur,
                        device=p_pack.device, dtype=torch.float32,
                    ).to(t_pack.dtype)
                    tea_out = self._cd_teacher(
                        noisy_image_or_video=corr_pack,
                        conditional_dict=conditional,
                        timestep=t_corr_pack,
                        clean_x=clean_x_pack,
                        aug_t=aug_t_pack,
                    )
                    tea_x0 = tea_out[1] if isinstance(tea_out, tuple) else tea_out
                    if t_nxt is None:
                        break
                    _tf = torch.full((B2, _nfb), t_cur,
                                     device=p_pack.device, dtype=torch.float32)
                    _tt = torch.full((B2, _nfb), float(t_nxt),
                                     device=p_pack.device, dtype=torch.float32)
                    x_cur = self._cd_partial_denoise(
                        x_cur, tea_x0[:, -_nfb:].to(x_cur.dtype), _tf, _tt)
                    t_cur = float(t_nxt)
                tea_tgt = tea_x0[:, -_nfb:].float()
            ts_all = (p_pack.float() - tea_tgt).pow(2).mean(dim=(1, 2, 3, 4))
            ts_sup_clean = ts_all[:B_pair].mean().to(dtype)
            ts_sup_cf = ts_all[B_pair:].mean().to(dtype)
            loss = loss + self.ode_teachersup_weight * (
                ts_sup_clean + lam_cf * ts_sup_cf)

        # ---- Per-chunk ODE MSE diagnostic (for dashboard) — clean branch.
        with torch.no_grad():
            per_frame_sq = (pred_clean.float() - target_clean.float()).pow(2).mean(dim=(2, 3, 4))  # [B_pair, F]
            per_chunk_sq = per_frame_sq.reshape(
                B_pair, n_c, self.num_frame_per_block,
            ).mean(dim=2).mean(dim=0)                                                              # [n_c]

        # ---- Branch-tagged log dict ----
        if isinstance(flow_pred, torch.Tensor):
            flow_norm = flow_pred.detach().float().norm().item()
        else:
            flow_norm = 0.0

        # PER-SAMPLE cf error for the hard-direction curriculum: the same
        # masked-frame squared error the ODE loss averages, but kept
        # per batch element so the trainer can attribute difficulty to
        # this (context, target-chunk, DIRECTION) triple. Always emitted
        # (cheap); the trainer ignores it unless the curriculum is on.
        with torch.no_grad():
            if getattr(self, "ode_chunked_supervision", False):
                _nfb_e = self.num_frame_per_block
                _pe = pred_cf[:, -_nfb_e:].float()
                _te = target_cf[:, -_nfb_e:].float().to(_pe.device)
                _per_sample_err = (_pe - _te).pow(2).mean(dim=(1, 2, 3, 4))
            else:
                _per_sample_err = (pred_cf.float() - target_cf.float()
                                   ).pow(2).flatten(1).mean(dim=1)

        log_dict: Dict[str, Any] = {
            # ODE losses
            "ode_loss_clean": ode_loss_clean.detach(),
            "ode_loss_cf":    ode_loss_cf.detach(),
            "per_sample_err_cf": _per_sample_err.detach(),
            # The rung this sample was drawn at. The raw error is dominated
            # by WHICH RUNG was sampled (t=1000 errors dwarf t=208 ones),
            # not by which direction is hard, so the curriculum MUST divide
            # by a per-rung reference before ranking directions — otherwise
            # "hardest direction" just means "unluckiest timestep draw".
            "per_sample_rung_cf": t_cf[:, -1].detach(),
            # On-policy teacher supervision (zeros when disabled)
            "ode_teachersup_clean": ts_sup_clean.detach(),
            "ode_teachersup_cf":    ts_sup_cf.detach(),
            "ode_teachersup_t":     ts_t_corr,
            # State-probe losses
            "state_loss_clean": state_loss_clean.detach() if isinstance(state_loss_clean, torch.Tensor) else torch.tensor(0.0),
            "state_loss_cf":    state_loss_cf.detach()    if isinstance(state_loss_cf, torch.Tensor)    else torch.tensor(0.0),
            # Frozen-readout state guidance (teacher parity)
            "state_guidance_loss_clean": state_g_loss_clean.detach() if isinstance(state_g_loss_clean, torch.Tensor) else torch.tensor(0.0),
            "state_guidance_loss_cf":    state_g_loss_cf.detach()    if isinstance(state_g_loss_cf, torch.Tensor)    else torch.tensor(0.0),
            "state_guidance_scale_clean": state_g_scale_clean,
            "state_guidance_scale_cf":    state_g_scale_cf,
            # Critic-guidance losses (gradient-through-pred path)
            "critic_guidance_loss_clean": critic_guidance_loss_clean.detach(),
            "critic_guidance_loss_cf":    critic_guidance_loss_cf.detach(),
            # Per-branch total losses (for logging parity with old trainer)
            "loss_clean": loss_clean.detach(),
            "loss_cf":    loss_cf.detach(),
            "guidance_scale":       guidance_scale_clean,
            "guidance_scale_cf":    guidance_scale_cf,
            # Timestep means
            "timestep_clean": t_clean.float().mean(dim=1).detach(),
            "timestep_cf":    t_cf.float().mean(dim=1).detach(),
            # Inputs to the critic's own update loop (per slot)
            "pred_x0_detached_clean": pred_clean.detach(),
            "pred_x0_detached_cf":    pred_cf.detach(),
            "teacher_z_8d_clean":     teacher_z_8d_clean,   # may be None
            "teacher_z_8d_cf":        teacher_z_8d_cf,      # may be None
            "chunk_t_clean":          chunk_t_clean.detach(),
            "chunk_t_cf":             chunk_t_cf.detach(),
            "chunk_actions_clean":    chunk_actions_clean.detach(),
            "chunk_actions_cf":       chunk_actions_cf.detach(),
            "critic_chunk_mask_clean": critic_mask_clean.detach(),
            "critic_chunk_mask_cf":    critic_mask_cf.detach(),
            # Targets (mostly for debugging / parity probes)
            "target_clean": target_clean.detach(),
            "target_cf":    target_cf.detach(),
            # Per-chunk diagnostic on clean branch
            "per_chunk_ode_mse": per_chunk_sq.detach(),
            "flow_pred_norm":    flow_norm,
        }
        log_dict.update(_cd_logs)
        return loss, log_dict
