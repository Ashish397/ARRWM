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
    step_value_to_snap_idx,
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
        random_steps = list(getattr(config, "random_steps", DEFAULT_RANDOM_STEPS))
        if len(random_steps) == 0:
            raise ValueError("random_steps must contain at least one entry")
        self.random_steps: List[int] = [int(s) for s in random_steps]
        usable_idx = [step_value_to_snap_idx(s) for s in self.random_steps]
        self.usable_stored_indices: List[int] = list(usable_idx)

        ds_list = resolve_denoising_step_list(
            usable_stored_indices=usable_idx,
            num_inference_steps=int(getattr(config, "eval_inference_steps", 48)),
            shift=float(model_kwargs.get("timestep_shift", 5.0)),
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
        # ADJACENT saved rungs, independent of the 4 ODE training rungs). 7
        # rungs: SNAPSHOT_STEPS timesteps ~= [1000, 893, 625, 500, 312.5,
        # 178.6, 0]. The student-CD/teacher-CD sample an adjacent (n, n+1)
        # pair from this list so they cover the full denoising path at the
        # finest saved granularity.
        cd_list = resolve_denoising_step_list(
            usable_stored_indices=list(range(len(SNAPSHOT_STEPS))),
            num_inference_steps=int(getattr(config, "eval_inference_steps", 48)),
            shift=float(model_kwargs.get("timestep_shift", 5.0)),
        )
        self.register_buffer(
            "cd_rung_list", cd_list.to(torch.float32).to(device), persistent=False,
        )
        log.info(
            "ODERegression[CD]: cd_rung_list (all %d saved rungs) timesteps=%s",
            len(SNAPSHOT_STEPS),
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
        if self.cd_teacher_loss_enabled:
            import copy as _copy
            # self.generator is still the BARE wrapper here (the trainer wraps
            # it in DDP only after model construction), so deepcopy it directly.
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

        if distributed and hasattr(torch.distributed, "is_initialized") and torch.distributed.is_initialized():
            is_main = (torch.distributed.get_rank() == 0)
            if is_main:
                self._frozen_cotracker = torch.hub.load(
                    "facebookresearch/co-tracker", "cotracker3_offline",
                ).to(self.device)
            torch.distributed.barrier()
            if not is_main:
                self._frozen_cotracker = torch.hub.load(
                    "facebookresearch/co-tracker", "cotracker3_offline",
                ).to(self.device)
            torch.distributed.barrier()
        else:
            self._frozen_cotracker = torch.hub.load(
                "facebookresearch/co-tracker", "cotracker3_offline",
            ).to(self.device)

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

        pool_table = torch.tensor(
            self.usable_stored_indices, device=self.device, dtype=torch.long,
        )
        snap_idx = pool_table[pool_idx]

        noisy_input = torch.gather(
            trajectory, dim=1,
            index=snap_idx.reshape(B, 1, F_, 1, 1, 1).expand(-1, -1, -1, C, H, W),
        ).squeeze(1)

        timestep = self.denoising_step_list[pool_idx]
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

    def generator_loss(
        self,
        trajectory_clean: torch.Tensor,      # [B_pair, T_snap, F, C, H, W]
        trajectory_cf: torch.Tensor,         # [B_pair, T_snap, F, C, H, W]
        prompt_embeds: torch.Tensor,         # [B_pair, 512, 4096]
        z_clean: torch.Tensor,               # [B_pair, F, 2]   (shared context action)
        z_noisy: torch.Tensor,               # [B_pair, F, 2]   (clean-branch action)
        z_noisy_cf: torch.Tensor,            # [B_pair, F, 2]   (CF-branch action)
        clean_x_gt: torch.Tensor,            # [B_pair, F, C, H, W]
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
        clean_x_pack = torch.cat([clean_x_in,   clean_x_in],   dim=0)
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

        log_dict: Dict[str, Any] = {
            # ODE losses
            "ode_loss_clean": ode_loss_clean.detach(),
            "ode_loss_cf":    ode_loss_cf.detach(),
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
