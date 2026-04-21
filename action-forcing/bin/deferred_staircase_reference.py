"""FROZEN REFERENCE — do not import, do not edit.

Verbatim snapshot of ``action-forcing/af_model/ode_regression.py`` captured
just before the staircase / GT-target / warmup paths were stripped from the
main module. Kept here so we can resurrect the rolling-forcing staircase
recipe without digging through git history. Everything below this banner
reflects the state of the code at that moment and will drift over time.

Original docstring follows.

---

Action-Forcing ODE distillation student model.

Two orthogonal config choices drive the training recipe:

* ``training_mode``: controls *input noise scheduling*.

  * ``"random"``   — Causal-Forcing-style per-block random timestep sampling
    from ``random_steps``, a user-configurable list in ODE-step-value
    vocabulary (default ``[0, 36, 44, 46, -1]`` — 5 entries spanning the
    full ODE arc, from pure noise ``0`` through teacher_x0 ``-1``).
    ``clean_x`` is supplied fully clean (``aug_t=None``).

  * ``"staircase"`` — Fully-configurable staircase over 7 chunks. Two
    7-member lists parameterise the schedule:

      * ``staircase_input_steps``  (default ``[-1, 46, 44, 40, 36, 18, 0]``)
        — the input noise level for each chunk. Valid values: any entry of
        ``SNAPSHOT_STEPS = [0, 18, 36, 40, 44, 46, -1]``.

      * ``staircase_target_steps`` (default ``["GT", -1, 46, 44, 40, 36, 18]``)
        — the target noise level for each chunk. Valid values: any entry of
        ``SNAPSHOT_STEPS`` **or** the string ``"GT"`` (real zarr ground
        truth, implying target-timestep = 0).

    ``clean_x`` is per-chunk noised to the target timestep (``aug_t``),
    which matches the level the chunk is *about to* reach after this
    forward pass.

* ``target_kind``: globally controls *what the student regresses against*.

  * ``"teacher_x0"`` — stay on the teacher manifold (honours the
    per-chunk ``staircase_target_steps`` list in staircase mode; uses
    ``trajectory[-1]`` uniformly in random mode).

  * ``"gt"``         — regress onto real zarr frames
    (``target_gt = zarr[o+3 : o+24]``). Overrides any ``"teacher_x0"``
    entries in the staircase target list.

Regardless of mode, ``_build_conditional`` mirrors the teacher's routing
of ``z_noisy`` / ``z_clean`` into four conditioning tensors, the generator
carries LoRA + action projections + state probe, and an optional frozen
motion pipeline (VAE + co-tracker + ss_vae) supplies teacher_z_8d for
state-probe and action-critic supervision.

Critic chunk selection:
  * ``staircase``: positions ``[0 .. critic_num_chunks-1]`` (the cleanest
    input noise levels).
  * ``random``:    chunks whose sampled pool index is among the top-K
    cleanest (``pool_idx >= K - critic_num_chunks``).
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
    DEFAULT_STAIR_INPUT_STEPS,
    DEFAULT_STAIR_TARGET_STEPS,
    GT_SENTINEL,
    NUM_CHUNKS,
    SNAPSHOT_STEPS,
    resolve_denoising_step_list,
    staircase_schedule,
    staircase_tensors,
    step_value_to_snap_idx,
    target_aug_timesteps,
    teacher_x0_timestep,
)


log = logging.getLogger(__name__)


TrainingMode = str  # "random" | "staircase"


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


def _weighted_z_mse(
    pred_z: torch.Tensor,
    target_z: torch.Tensor,
    weight_dims: Optional[List[int]] = None,
    weight: float = 2.0,
) -> torch.Tensor:
    """Teacher's weighted MSE over 8-D z with 2x weight on action-relevant dims."""
    D = pred_z.shape[-1]
    w = torch.ones(D, device=pred_z.device, dtype=pred_z.dtype)
    if weight_dims is not None:
        for d in weight_dims:
            w[d] = weight
    return (w * (pred_z - target_z) ** 2).mean()


class ODERegression(nn.Module):
    """Student generator + full teacher action apparatus (both training modes)."""

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.dtype = torch.bfloat16 if bool(getattr(config, "mixed_precision", True)) else torch.float32

        # ------------------------------------------------------------------
        # Mode + hyper-params (teacher v14 defaults)
        # ------------------------------------------------------------------
        self.training_mode: TrainingMode = str(getattr(config, "training_mode", "random")).lower()
        if self.training_mode not in ("random", "staircase"):
            raise ValueError(
                f"config.training_mode must be 'random' or 'staircase' (got {self.training_mode!r})"
            )

        self.target_kind: str = str(getattr(config, "target_kind", "gt")).lower()
        if self.target_kind not in ("teacher_x0", "gt"):
            raise ValueError(
                f"config.target_kind must be 'teacher_x0' or 'gt' (got {self.target_kind!r})"
            )

        self.num_frame_per_block: int = int(getattr(config, "num_frame_per_block", 3))
        self.context_frames: int = int(getattr(config, "context_frames", 3))
        self.num_training_frames: int = int(getattr(config, "num_training_frames", 21))
        self.n_chunks: int = self.num_training_frames // self.num_frame_per_block
        if self.training_mode == "staircase" and self.n_chunks != NUM_CHUNKS:
            raise ValueError(
                f"staircase mode requires n_chunks={NUM_CHUNKS}, got {self.n_chunks}"
            )

        self.model_variant: str = str(getattr(config, "model_variant", "action-injection"))
        self.raw_action_dim: int = int(getattr(config, "raw_action_dim", 2))
        self.action_activation: str = str(getattr(config, "action_activation", "silu"))
        self.enable_adaln_zero: bool = bool(getattr(config, "enable_adaln_zero", True))
        acm = str(getattr(config, "action_conditioning_mode", "both"))
        self.use_adaln: bool = acm in ("adaln", "both")
        self.use_action_tokens: bool = acm in ("tokens", "both")
        self.use_action_conditioning: bool = True

        self.action_critic_enabled: bool = bool(getattr(config, "action_critic_enabled", True))
        self.action_critic_dims: list = list(getattr(config, "action_critic_dims", [2, 7]))
        self.action_critic_z_out_dim: int = int(getattr(config, "action_critic_z_out_dim", 8))
        self.action_critic_z_loss_weight: float = float(getattr(config, "action_critic_z_loss_weight", 0.5))
        self.generator_action_z_guidance_weight: float = float(
            getattr(config, "generator_action_z_guidance_weight", 0.25),
        )
        self.z_guidance_warmup_steps: int = int(getattr(config, "z_guidance_warmup_steps", 200))
        self.warmup_steps: int = int(getattr(config, "warmup_steps", 200))

        # Which chunks the action critic should supervise.
        # Default: the three cleanest noise levels.
        self.critic_num_chunks: int = int(getattr(config, "critic_num_chunks", 3))

        self.state_head_enabled: bool = bool(getattr(config, "state_head_enabled", True))
        self.state_probe_mode: bool = bool(getattr(config, "state_probe_mode", True))
        self.state_probe_dim: int = int(getattr(config, "state_probe_dim", 256))
        self.state_probe_n_taps: int = int(getattr(config, "state_probe_n_taps", 6))
        self.state_probe_num_heads: int = int(getattr(config, "state_probe_num_heads", 8))
        self.state_head_out_dim: int = int(getattr(config, "state_head_out_dim", 8))
        self.state_head_loss_weight: float = float(getattr(config, "state_head_loss_weight", 0.1))
        self.state_head_action_dim_weight: float = float(
            getattr(config, "state_head_action_dim_weight", 3.0),
        )

        self.use_motion_pipeline: bool = bool(getattr(config, "use_motion_pipeline", True))

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
        # 3) Apply LoRA adapter (post action patches, pre state probe)
        # ------------------------------------------------------------------
        lora_cfg = getattr(config, "adapter", None)
        if lora_cfg is None:
            raise ValueError("config.adapter (LoRA) is required.")
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
        # 6) Resolve denoising_step_list
        # ------------------------------------------------------------------
        # Staircase mode always needs the full 7-snapshot pool (any chunk may
        # reference any snap index). Random mode uses the user-configurable
        # ``random_steps`` list, expressed in ODE-step-value vocabulary
        # (same alphabet as the staircase lists; ``-1`` = teacher_x0).
        if self.training_mode == "staircase":
            usable_idx = [0, 1, 2, 3, 4, 5, 6]
            self.random_steps: List[int] = [SNAPSHOT_STEPS[i] for i in usable_idx]
        else:
            random_steps = list(getattr(config, "random_steps", DEFAULT_RANDOM_STEPS))
            if len(random_steps) == 0:
                raise ValueError("random_steps must contain at least one entry")
            self.random_steps = [int(s) for s in random_steps]
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
        self.teacher_x0_ts: float = teacher_x0_timestep(
            num_inference_steps=int(getattr(config, "eval_inference_steps", 48)),
            shift=float(model_kwargs.get("timestep_shift", 5.0)),
        )
        log.info(
            "ODERegression[mode=%s target=%s]: random_steps=%s (snap_idx=%s) timesteps=%s",
            self.training_mode, self.target_kind,
            self.random_steps, usable_idx,
            [round(float(x), 2) for x in self.denoising_step_list.tolist()],
        )

        # ------------------------------------------------------------------
        # 7) Staircase-mode constants (cached tensors)
        # ------------------------------------------------------------------
        if self.training_mode == "staircase":
            input_steps = list(getattr(
                config, "staircase_input_steps", DEFAULT_STAIR_INPUT_STEPS,
            ))
            target_steps = list(getattr(
                config, "staircase_target_steps", DEFAULT_STAIR_TARGET_STEPS,
            ))
            self.staircase_input_steps = input_steps
            self.staircase_target_steps = target_steps

            # Canonicalised schedule (sanity-check + reuse below).
            canon_input, canon_target = staircase_schedule(
                input_steps=input_steps,
                target_steps=target_steps,
                num_chunks=self.n_chunks,
            )

            inp_idx, tgt_idx, tgt_gt_mask = staircase_tensors(
                input_steps=input_steps,
                target_steps=target_steps,
                num_chunks=self.n_chunks,
                num_frame_per_block=self.num_frame_per_block,
                device=device,
            )
            # Per-frame stored-trajectory snapshot indices for input/target.
            self.register_buffer("_stair_input_snap", inp_idx, persistent=False)   # [F]
            self.register_buffer("_stair_target_snap", tgt_idx, persistent=False)  # [F]
            # Per-frame GT-target mask: True where target is zarr GT.
            self.register_buffer("_stair_target_gt_mask", tgt_gt_mask, persistent=False)
            # Per-frame timesteps resolved via the teacher's FlowMatchScheduler.
            stair_timestep = self._snap_to_timestep(inp_idx)        # [F]
            stair_aug_t = target_aug_timesteps(
                canon_target,
                num_frame_per_block=self.num_frame_per_block,
                num_inference_steps=int(getattr(config, "eval_inference_steps", 48)),
                shift=float(model_kwargs.get("timestep_shift", 5.0)),
                device=device,
            )
            self.register_buffer("_stair_timestep", stair_timestep, persistent=False)
            self.register_buffer("_stair_aug_t", stair_aug_t, persistent=False)

            log.info(
                "staircase schedule: input=%s target=%s",
                input_steps, target_steps,
            )

        # ------------------------------------------------------------------
        # 8) Frozen motion pipeline (VAE + co-tracker + ss_vae)
        # ------------------------------------------------------------------
        self._motion_pipeline_ready = False
        if self.use_motion_pipeline:
            # Lazily built on first call to avoid download races before DDP
            # rendezvous. See ``ensure_motion_pipeline``.
            self._motion_pipeline_ready = False

        # ------------------------------------------------------------------
        # 9) Load all heads + LoRA from the v14 teacher checkpoint
        # ------------------------------------------------------------------
        self.loaded_from_step: int = -1
        ckpt_path = getattr(config, "generator_ckpt", None)
        if ckpt_path:
            self.load_teacher_checkpoint(str(ckpt_path))

    # ------------------------------------------------------------------
    # Snapshot-index -> teacher-timestep conversion
    # ------------------------------------------------------------------

    def _snap_to_timestep(self, snap_idx: torch.Tensor) -> torch.Tensor:
        """Map stored-trajectory snapshot indices (0..6) to teacher timesteps.

        Because ``denoising_step_list`` is pre-sliced to the *usable* pool
        (which in staircase mode is the full pool ``[0,1,2,3,4,5,6]``),
        the snapshot index can be used directly as the pool index.
        """
        pool_table = {p: i for i, p in enumerate(self.usable_stored_indices)}
        out = torch.empty_like(snap_idx, dtype=torch.float32, device=self.denoising_step_list.device)
        for p, i in pool_table.items():
            out[snap_idx == p] = self.denoising_step_list[i]
        return out

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def load_teacher_checkpoint(self, ckpt_path: str) -> None:
        """Load LoRA + action heads + state probe + action critic from ``ckpt_path``.

        Matches the save layout used by ``trainer/causal_diffusion_teacher_train.py``:
          ``{'lora': ..., 'action_projection': ..., 'action_token_projection': ...,
             'state_probe': ..., 'action_critic': ..., 'step': ..., 'config_name': ...}``
        """
        p = Path(ckpt_path)
        if not p.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {p}")
        ck = torch.load(p, map_location="cpu", weights_only=False)

        if "lora" in ck:
            set_peft_model_state_dict(self.generator.model, ck["lora"])
            log.info("Loaded LoRA adapters from %s", p.name)

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
            missing, unexpected = self.action_critic.load_state_dict(
                ck["action_critic"], strict=False,
            )
            if missing or unexpected:
                log.warning(
                    "action_critic partial load: %d missing, %d unexpected",
                    len(missing), len(unexpected),
                )
            else:
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

        Decodes ``pred_x0`` via the frozen VAE, runs cotracker for point
        tracks + visibility, and encodes the motion through ``ss_vae`` to
        produce the 8-D teacher latent ``z``. Returns ``[B, n_chunks, 8]``.
        """
        if not self._motion_pipeline_ready:
            raise RuntimeError("motion pipeline not built; call ensure_motion_pipeline().")

        B, F_full, C, H, W = pred_x0.shape
        n_chunks = F_full // self.num_frame_per_block

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

        all_teacher_z = []
        for b in range(B):
            vid = video[b].unsqueeze(0)
            T_total = vid.shape[1]
            mws = []
            for cs in range(0, T_total, compute_T):
                ce = min(cs + compute_T, T_total)
                ch = vid[:, cs:ce]
                n_out = ch.shape[1] // output_chunk_size
                if n_out == 0:
                    continue
                used = n_out * output_chunk_size
                ch = ch[:, :used].clone()
                with torch.amp.autocast(device_type="cuda", enabled=True):
                    tracks, vis = self._frozen_cotracker(ch, grid_size=grid_size)
                tw = tracks.reshape(1, n_out, output_chunk_size, N, 2)
                if vis.dim() == 3:
                    vw = vis.reshape(1, n_out, output_chunk_size, N).unsqueeze(-1)
                else:
                    vw = vis.reshape(1, n_out, output_chunk_size, N, 1)
                dw = tw[:, :, 1:] - tw[:, :, :-1]
                mo = dw.mean(dim=2)
                vo = vw.to(dtype=mo.dtype).mean(dim=2)
                mws.append(torch.cat([mo, vo], dim=-1).squeeze(0))

            if not mws:
                all_teacher_z.append(torch.zeros(n_chunks, 8, device=pred_x0.device))
                continue

            est_motion = torch.cat(mws, dim=0)
            raw_n = est_motion.shape[0]
            xy = est_motion[:, :, :2].reshape(raw_n, 10, 10, 2)
            x_in = xy.permute(0, 3, 1, 2).float() / self._frozen_ss_vae_scale
            mu, _ = self._frozen_ss_vae.encoder(x_in.to(self.device))
            z8 = _tanh_squash(mu.squeeze(-1).squeeze(-1))
            all_teacher_z.append(self._reduce_to_segments(z8, n_chunks))

        teacher_z = torch.stack(all_teacher_z, dim=0)
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
    # Input preparation — mode-specific
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
    def _prep_staircase(
        self,
        trajectory: torch.Tensor,
        clean_x_gt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fixed rolling-forcing staircase; noise clean_x to the target level.

        Returns:
          noisy_input[B, F, C, H, W], timestep[B, F],
          aug_t[B, F], clean_x_noised[B, F, C, H, W]
        """
        B, T_snap, F_, C, H, W = trajectory.shape

        inp_snap = self._stair_input_snap.unsqueeze(0).expand(B, -1)   # [B, F]
        noisy_input = torch.gather(
            trajectory, dim=1,
            index=inp_snap.reshape(B, 1, F_, 1, 1, 1).expand(-1, -1, -1, C, H, W),
        ).squeeze(1)

        timestep = self._stair_timestep.unsqueeze(0).expand(B, -1).contiguous()  # [B, F]
        aug_t = self._stair_aug_t.unsqueeze(0).expand(B, -1).contiguous()         # [B, F]

        noise = torch.randn_like(clean_x_gt)
        clean_x_noised = self.scheduler.add_noise(
            clean_x_gt.flatten(0, 1), noise.flatten(0, 1), aug_t.flatten(0, 1),
        ).view_as(clean_x_gt)

        return noisy_input, timestep, aug_t, clean_x_noised

    @torch.no_grad()
    def _resolve_target(
        self,
        trajectory: torch.Tensor,  # [B, T_snap, F, C, H, W]
        target_gt: torch.Tensor,   # [B, F, C, H, W]
    ) -> torch.Tensor:
        """Pick the regression target per frame.

        * ``target_kind="gt"``          -> ``target_gt`` for every frame.
        * ``target_kind="teacher_x0"``:
            * ``staircase`` mode -> per-frame blend of
              ``trajectory[stair_target_snap]`` and ``target_gt`` (the latter
              for frames whose target schedule entry is ``"GT"``).
            * ``random`` mode    -> ``trajectory[snap_idx=6]`` (the final
              teacher x0 snapshot, uniform across all frames).
        """
        if self.target_kind == "gt":
            return target_gt

        B, T_snap, F_, C, H, W = trajectory.shape
        if self.training_mode == "staircase":
            tgt_snap = self._stair_target_snap.unsqueeze(0).expand(B, -1)  # [B, F]
            traj_target = torch.gather(
                trajectory, dim=1,
                index=tgt_snap.reshape(B, 1, F_, 1, 1, 1).expand(-1, -1, -1, C, H, W),
            ).squeeze(1)
            gt_mask = self._stair_target_gt_mask                             # [F]
            if gt_mask.any():
                # Broadcast mask to [B, F, 1, 1, 1]
                m = gt_mask.view(1, F_, 1, 1, 1).to(device=traj_target.device)
                target_gt_cast = target_gt.to(traj_target.dtype)
                return torch.where(m, target_gt_cast, traj_target)
            return traj_target

        # random mode, target=teacher_x0: final teacher x0 snapshot.
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
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """``MSE(pred_x0, target)`` averaged over all frames.

        The previous ``timestep != 0`` mask was an identity-prediction
        short-circuit that no longer holds under configurable schedules
        (e.g. staircase chunk 0 has ``timestep=0`` input targeting GT).
        Identity samples contribute exactly 0 loss/grad so an unmasked MSE
        is both correct and simpler.
        """
        _ = timesteps  # kept for call-site parity; may be used by diagnostics
        return F.mse_loss(pred_x0.float(), target.float())

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
        pool_idx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Which chunks the critic should train on.

        * staircase mode: positions ``[0 .. critic_num_chunks-1]`` (the
          three cleanest).
        * random mode: chunks whose pool index is among the top-K cleanest
          levels (i.e. ``>= K - critic_num_chunks``).

        Returns a bool mask of shape ``[B, n_chunks]``.
        """
        K = int(self.denoising_step_list.shape[0])
        k = min(self.critic_num_chunks, self.n_chunks)

        if self.training_mode == "staircase":
            mask = torch.zeros(B, self.n_chunks, dtype=torch.bool, device=self.device)
            mask[:, :k] = True
            return mask
        # random mode
        if pool_idx is None:
            return torch.zeros(B, self.n_chunks, dtype=torch.bool, device=self.device)
        # pool_idx is [B, F]; take first frame per chunk.
        chunk_pool = pool_idx[:, :: self.num_frame_per_block][:, : self.n_chunks]
        return chunk_pool >= (K - k)

    def _compute_critic_guidance_loss(
        self,
        pred_x0: torch.Tensor,                 # [B, F, C, H, W]
        chunk_t: torch.Tensor,                 # [B, n_c]
        chunk_actions: torch.Tensor,           # [B, n_c, 2]
        chunk_mask: torch.Tensor,              # [B, n_c] bool
        step: int,
    ) -> Tuple[torch.Tensor, float]:
        """Frozen-critic generator guidance (gradient through ``pred_x0``).

        Returns ``(loss, guidance_scale)``.
        """
        if self.action_critic is None or not chunk_mask.any():
            return torch.zeros((), device=self.device), 0.0

        # Warmup ramp (identical to teacher).
        if step < self.warmup_steps:
            scale = 0.0
        elif self.z_guidance_warmup_steps > 0:
            ramp = min(1.0, (step - self.warmup_steps) / self.z_guidance_warmup_steps)
            scale = ramp * self.generator_action_z_guidance_weight
        else:
            scale = self.generator_action_z_guidance_weight
        if scale <= 0.0:
            return torch.zeros((), device=self.device), 0.0

        # Use the *unwrapped* critic here: we backprop only through pred_x0,
        # critic params are frozen, and we don't want DDP forward tracking.
        try:
            from torch.nn.parallel import DistributedDataParallel as _DDP
            critic_mod = self.action_critic.module if isinstance(self.action_critic, _DDP) else self.action_critic
        except Exception:
            critic_mod = self.action_critic

        was_trainable = any(p.requires_grad for p in critic_mod.parameters())
        critic_mod.requires_grad_(False)
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
            if was_trainable:
                critic_mod.requires_grad_(True)
        return loss, float(scale)

    # ------------------------------------------------------------------
    # Main training entrypoint (per-sample, per-branch).
    # ------------------------------------------------------------------

    def generator_loss(
        self,
        trajectory: torch.Tensor,            # [B, T_snap, F, C, H, W]
        prompt_embeds: torch.Tensor,         # [B, 512, 4096]
        z_noisy: torch.Tensor,               # [B, F, 2]
        z_clean: torch.Tensor,               # [B, F, 2]
        clean_x_gt: torch.Tensor,            # [B, F, C, H, W]  — truly clean zarr window
        target_gt: torch.Tensor,             # [B, F, C, H, W]  — zarr[o+3:o+24]
        step: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Run a full training-forward pass under the selected mode.

        The returned ``log_dict`` carries ``pred_x0_detached`` and
        ``teacher_z_8d`` so the trainer can run the critic-training loop
        (mirroring the teacher's ``critic_updates_per_step`` pattern).
        """
        B, _T, F_ = trajectory.shape[0], trajectory.shape[1], trajectory.shape[2]

        # ---- Mode-specific input prep ----
        pool_idx: Optional[torch.Tensor] = None
        if self.training_mode == "staircase":
            noisy_input, timestep, aug_t, clean_x_in = self._prep_staircase(
                trajectory, clean_x_gt,
            )
        else:
            noisy_input, timestep, pool_idx = self._prep_random(trajectory)
            aug_t = None
            clean_x_in = clean_x_gt

        # ---- Target selection ----
        target = self._resolve_target(trajectory, target_gt)

        # ---- Casts ----
        noisy_input = noisy_input.to(self.dtype)
        clean_x_in = clean_x_in.to(self.dtype)
        prompt_embeds = prompt_embeds.to(self.dtype)
        z_noisy = z_noisy.to(self.dtype)
        z_clean = z_clean.to(self.dtype)

        conditional = self._build_conditional(prompt_embeds, z_noisy, z_clean, num_frames=F_)

        model_out = self.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=conditional,
            timestep=timestep,
            clean_x=clean_x_in,
            aug_t=aug_t,
        )
        if isinstance(model_out, tuple) and len(model_out) == 4:
            _flow_pred, pred_x0, state_preds, _probe_hidden = model_out
        elif isinstance(model_out, tuple) and len(model_out) == 3:
            _flow_pred, pred_x0, state_preds = model_out
        else:
            _flow_pred, pred_x0 = model_out[:2]
            state_preds = None

        # ---- ODE regression ----
        ode_loss = self._compute_ode_loss(pred_x0, target.to(self.dtype), timestep)
        loss = ode_loss

        # ---- State probe + action critic via motion pipeline ----
        n_c = self.n_chunks
        chunk_t = timestep[:, :: self.num_frame_per_block][:, :n_c]
        chunk_actions = _chunk_actions(z_clean, self.num_frame_per_block)[:, :n_c]  # commanded z (2-D)

        teacher_z_8d: Optional[torch.Tensor] = None
        if self.use_motion_pipeline and self._motion_pipeline_ready:
            teacher_z_8d = self._compute_action_teacher_targets(pred_x0.detach())

        state_loss = torch.zeros((), device=self.device)
        if state_preds is not None and self._state_head_built:
            state_loss = self._state_probe_loss(
                state_preds[:, :n_c], teacher_z_8d, chunk_actions,
            )
            loss = loss + self.state_head_loss_weight * state_loss

        critic_mask = self._critic_chunk_mask(B, pool_idx)
        critic_guidance_loss, guidance_scale = self._compute_critic_guidance_loss(
            pred_x0, chunk_t, chunk_actions, critic_mask, step,
        )
        loss = loss + critic_guidance_loss

        log_dict: Dict[str, Any] = {
            "mode": self.training_mode,
            "target_kind": self.target_kind,
            "ode_loss": ode_loss.detach(),
            "state_loss": state_loss.detach() if isinstance(state_loss, torch.Tensor) else torch.tensor(0.0),
            "critic_guidance_loss": critic_guidance_loss.detach(),
            "guidance_scale": guidance_scale,
            "timestep": timestep.float().mean(dim=1).detach(),
            "pred_x0_detached": pred_x0.detach(),
            "teacher_z_8d": teacher_z_8d,           # may be None
            "chunk_t": chunk_t.detach(),
            "chunk_actions": chunk_actions.detach(),
            "critic_chunk_mask": critic_mask.detach(),
            "target": target.detach(),
        }
        return loss, log_dict
