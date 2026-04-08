"""CausalLoRADiffusionTrainer: causal (autoregressive) diffusion teacher training.

Streaming-only trainer for the Causal-Forcing Stage 1 pipeline:
  - CausalWanModel (is_causal=True) with block-wise causal attention
  - ZarrRideDataset + LockstepRideBatcher: ride-level streaming windows
  - Per-block independent timestep sampling (BSMNTW-weighted flow loss)
  - Teacher forcing: model sees clean context (clean_x + aug_t)
  - Action conditioning: tanh-squashed z2/z7 ss_vae latent via adaLN-Zero + tokens
  - Learned action critic for action-aware guidance
  - Periodic held-out evaluation with W&B video logging
"""

import argparse
import logging
import math
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import wandb
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from torch.cuda.amp import autocast, GradScaler
from omegaconf import OmegaConf
import peft
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict

from utils.zarr_dataset import ZarrRideDataset, build_ride_manifest, _tanh_squash
from utils.dataset import cycle
from utils.distributed import barrier, launch_distributed_job
from utils.misc import set_seed
from utils.memory import log_gpu_memory
from utils.debug_option import LOG_GPU_MEMORY
from utils.wan_wrapper import WanDiffusionWrapper, WanVAEWrapper

from model.action_modulation import ActionModulationProjection, ActionTokenProjection
from model.action_model_patch import apply_action_patches
from model.causal_teacher_streaming import LockstepRideBatcher
from model.action_critic import ActionCritic


def _is_distributed() -> bool:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    return world > 1


# ======================================================================
# Eval / overlay helpers (factored from testing/test_actual_trainer.py)
# ======================================================================

def _temporal_pool(per_frame: torch.Tensor, n_seg: int) -> torch.Tensor:
    """Average-pool ``[B, F, D]`` into ``[B, n_seg, D]``."""
    B, F_len, D = per_frame.shape
    seg_size = max(1, F_len // n_seg)
    segs = []
    for i in range(n_seg):
        s = i * seg_size
        e = min(s + seg_size, F_len)
        segs.append(per_frame[:, s:e].mean(dim=1))
    return torch.stack(segs, dim=1)


def _chunk_actions(per_frame: torch.Tensor, chunk_frames: int) -> torch.Tensor:
    """Mean-pool ``[B, F, D]`` into ``[B, n_chunks, D]`` with exact chunk size."""
    B, F_len, D = per_frame.shape
    n_chunks = F_len // chunk_frames
    trimmed = per_frame[:, :n_chunks * chunk_frames]
    return trimmed.reshape(B, n_chunks, chunk_frames, D).mean(dim=2)


def _safe_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation between flattened tensors, safe against zero variance."""
    a_flat = a.detach().float().flatten()
    b_flat = b.detach().float().flatten()
    if a_flat.numel() < 2:
        return 0.0
    a_c = a_flat - a_flat.mean()
    b_c = b_flat - b_flat.mean()
    denom = a_c.norm() * b_c.norm()
    if denom < 1e-8:
        return 0.0
    return (a_c @ b_c / denom).item()


def _asymmetric_action_diff(
    gen: torch.Tensor,
    target: torch.Tensor,
    over_weight: float = 0.5,
    under_weight: float = 1.0,
) -> torch.Tensor:
    """Compute asymmetric action error that penalises undershoot more than overshoot.

    Overshoot = same direction as target but greater magnitude (e.g. gen=-5
    for target=-4).  Undershoot = less magnitude or wrong direction.

    Near-zero targets (``|target| < 1e-3``) fall back to symmetric absolute error.
    """
    raw_err = gen - target
    dir_sign = torch.sign(target)
    signed_err = raw_err * dir_sign

    over_err = torch.relu(signed_err)
    under_err = torch.relu(-signed_err)
    asym = over_weight * over_err + under_weight * under_err

    sym = raw_err.abs()
    return torch.where(target.abs() > 1e-3, asym, sym)


def _draw_z_action_overlay(
    frame: np.ndarray,
    teacher_z: np.ndarray,
    critic_z: np.ndarray,
    target_z: np.ndarray,
    title: str,
    frame_idx: int,
    seg_idx: int,
    clip: float = 1.0,
    state_z: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Draw teacher / critic / state / target action bars for z2 and z7."""
    h, w = frame.shape[:2]
    out = frame.copy()
    panel_w = 220
    panel_x0 = w - panel_w
    out[:, panel_x0:] = (out[:, panel_x0:].astype(np.float32) * 0.22).astype(np.uint8)

    def draw_bar(y_mid: int, value: float, color: tuple, label: str):
        cx = panel_x0 + 120
        half = 80
        bar_h = 7
        v = float(max(-clip, min(clip, value)))
        bar_len = int(abs(v) / clip * half)
        cv2.line(out, (cx, y_mid - 10), (cx, y_mid + 10), (120, 120, 120), 1)
        if v >= 0:
            cv2.rectangle(out, (cx, y_mid - bar_h), (cx + bar_len, y_mid + bar_h), color, -1)
        else:
            cv2.rectangle(out, (cx - bar_len, y_mid - bar_h), (cx, y_mid + bar_h), color, -1)
        cv2.putText(
            out, f"{label} {value:+.3f}", (panel_x0 + 8, y_mid + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA,
        )

    cv2.putText(out, title, (panel_x0 + 8, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"frame={frame_idx} seg={seg_idx}", (panel_x0 + 8, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 200), 1, cv2.LINE_AA)

    has_state = state_z is not None
    if has_state:
        cv2.putText(out, "teach(g) crit(c) state(o) tgt(b)", (panel_x0 + 8, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.26, (180, 180, 180), 1, cv2.LINE_AA)
        step = 16
        z2_y0, z7_y0 = 74, 146
    else:
        cv2.putText(out, "teacher(g) critic(c) target(b)", (panel_x0 + 8, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180, 180, 180), 1, cv2.LINE_AA)
        step = 18
        z2_y0, z7_y0 = 78, 138

    rows = [
        ("z2", float(teacher_z[0]), float(critic_z[0]), float(target_z[0]),
         float(state_z[0]) if has_state else None, z2_y0),
        ("z7", float(teacher_z[1]), float(critic_z[1]), float(target_z[1]),
         float(state_z[1]) if has_state else None, z7_y0),
    ]
    for z_name, teacher_val, critic_val, target_val, state_val, y0 in rows:
        cv2.putText(out, z_name, (panel_x0 + 8, y0 - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 120), 1, cv2.LINE_AA)
        y = y0
        draw_bar(y, teacher_val, (80, 220, 80), "t")
        y += step
        draw_bar(y, critic_val, (100, 200, 255), "c")
        y += step
        if has_state:
            draw_bar(y, state_val, (50, 170, 255), "s")
            y += step
        draw_bar(y, target_val, (80, 80, 220), "y")

    return out


def _annotate_action_video(
    video_np: np.ndarray,
    est_motion: torch.Tensor,
    teacher_z2z7: torch.Tensor,
    critic_z2z7: torch.Tensor,
    target_action_z: torch.Tensor,
    title: str,
    grid_size: int = 10,
    output_chunk_size: int = 12,
    state_z2z7: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """Overlay evaluator motion + teacher/critic/state/target z2/z7 bars.

    ``target_action_z`` should already be chunked to ``[B, n_chunks, D]``,
    matching the dimensionality of ``teacher_z2z7``.
    ``state_z2z7``, when provided, should be ``[B, n_chunks, 2]``.
    """
    teacher_np = teacher_z2z7[0].detach().cpu().numpy()
    critic_np = critic_z2z7[0].detach().cpu().numpy()
    n_seg = teacher_z2z7.shape[1]
    target_seg_np = target_action_z[0, :n_seg].detach().float().cpu().numpy()
    motion_np = est_motion[0].detach().cpu().numpy()
    state_np = state_z2z7[0].detach().cpu().numpy() if state_z2z7 is not None else None

    n_frames = video_np.shape[0]
    annotated = []
    panel_w = 220
    for frame_idx in range(n_frames):
        f = video_np[frame_idx].copy()
        seg_idx = min(frame_idx // output_chunk_size, teacher_np.shape[0] - 1)
        h, w = f.shape[:2]

        if seg_idx < motion_np.shape[0]:
            mvecs = motion_np[seg_idx]
            drawable_w = max(1, w - panel_w)
            for gy in range(grid_size):
                for gx in range(grid_size):
                    idx = gy * grid_size + gx
                    dx, dy, vis_val = mvecs[idx]
                    if vis_val < 0.2:
                        continue
                    cx_pt = int((gx + 0.5) * drawable_w / grid_size)
                    cy_pt = int((gy + 0.5) * h / grid_size)
                    ex_pt = int(cx_pt - dx * 3)
                    ey_pt = int(cy_pt - dy * 3)
                    color = (0, 255, 0) if vis_val >= 0.5 else (0, 200, 255)
                    cv2.arrowedLine(f, (cx_pt, cy_pt), (ex_pt, ey_pt), color, 1, tipLength=0.3)

        state_seg = state_np[seg_idx] if state_np is not None and seg_idx < state_np.shape[0] else None
        f = _draw_z_action_overlay(
            f, teacher_np[seg_idx], critic_np[seg_idx], target_seg_np[seg_idx],
            title=title, frame_idx=frame_idx, seg_idx=seg_idx,
            state_z=state_seg,
        )
        annotated.append(f)
    return np.stack(annotated)


def _frames_to_mp4_bytes(frames: np.ndarray, fps: float = 5.0) -> Optional[bytes]:
    """Encode uint8 [T, H, W, 3] to mp4 bytes in memory (for wandb.Video)."""
    h, w = frames.shape[1], frames.shape[2]
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-f", "mp4", "-movflags", "frag_keyframe+empty_moov",
        "pipe:1",
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out_bytes, _ = proc.communicate(input=frames.tobytes(), timeout=120)
        if proc.returncode == 0 and len(out_bytes) > 0:
            return out_bytes
    except Exception:
        pass
    return None


# ======================================================================
# Trainer
# ======================================================================

class CausalLoRADiffusionTrainer:
    """LoRA finetuning for the causal (autoregressive) Wan diffusion model.

    Trains Stage 1 of the Causal-Forcing pipeline: an AR diffusion model with
    teacher forcing, block-wise causal attention, and action conditioning via
    the ss_vae z2/z7 motion latent.
    """

    def __init__(self, config):
        self.config = config
        self.is_distributed = _is_distributed()
        if self.is_distributed and not dist.is_initialized():
            launch_distributed_job()
        self.global_rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.is_main_process = self.global_rank == 0
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)
        self.allow_checkpoint_config_mismatch = bool(
            getattr(config, "allow_checkpoint_config_mismatch", False)
        )

        self.max_steps = int(getattr(config, "max_iters", 30000))
        self.log_interval = max(1, int(getattr(config, "log_interval", 10)))
        self.ckpt_interval = int(getattr(config, "ckpt_interval", 300))
        self.gradient_accumulation = max(1, int(getattr(config, "gradient_accumulation_steps", 1)))
        self.grad_clip = float(getattr(config, "max_grad_norm", 1.0))
        self.num_workers = int(getattr(config, "num_workers", 4))

        self.use_mixed_precision = bool(getattr(config, "mixed_precision", True))
        dtype_str = getattr(config, "autocast_dtype", "bf16")
        self.autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(
            dtype_str, torch.bfloat16
        )
        self.dtype = self.autocast_dtype

        self.disable_wandb = bool(getattr(config, "disable_wandb", False))
        self.wandb_project = getattr(config, "wandb_project", None)
        self.wandb_entity = getattr(config, "wandb_entity", None)
        self.wandb_group = getattr(config, "wandb_group", None)
        tags_value = getattr(config, "wandb_tags", None)
        if tags_value is not None:
            try:
                self.wandb_tags = list(tags_value)
            except TypeError:
                self.wandb_tags = [tags_value]
        else:
            self.wandb_tags = []
        self.wandb_run_name = getattr(config, "wandb_run_name", None)
        self._wandb_login_key = getattr(config, "wandb_key", None)

        self.teacher_forcing = bool(getattr(config, "teacher_forcing", True))
        self.num_frame_per_block = int(getattr(config, "num_frame_per_block", 3))
        self.context_frames = int(getattr(config, "context_frames", self.num_frame_per_block))
        self.noise_augmentation_max_timestep = int(
            getattr(config, "noise_augmentation_max_timestep", 0)
        )

        action_dims_cfg = getattr(config, "action_dims", None)
        self.action_dims = list(action_dims_cfg) if action_dims_cfg is not None else None

        self.streaming_chunk_size = int(getattr(config, "streaming_chunk_size", 21))

        set_seed(int(getattr(config, "seed", 0)) + self.global_rank)

        if self.is_main_process:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
            logging.info("Initialising CausalLoRADiffusionTrainer...")

        self.logdir = self._resolve_logdir(getattr(config, "logdir", ""))
        self.config_name = getattr(config, "config_name", None)
        self.load_checkpoint_config_name = getattr(config, "load_checkpoint_config_name", None)
        if self.config_name is None:
            raise ValueError("config_name is required")
        if self.is_main_process:
            logging.info("Resolved logdir: %s", self.logdir)
            self.logdir.mkdir(parents=True, exist_ok=True)
        if self.is_distributed:
            barrier()

        adapter_cfg = getattr(config, "adapter", {})
        try:
            self.adapter_rank = adapter_cfg.get("rank")
            self.adapter_alpha = adapter_cfg.get("alpha")
        except AttributeError:
            self.adapter_rank = None
            self.adapter_alpha = None

        self.train_dataset_size: Optional[int] = None
        self.eval_dataset_size: Optional[int] = None
        self.wandb_run = None

        hf_cache = self.scratch_root / "frodobots" / "hf_cache"
        for env_key in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
            os.environ.setdefault(env_key, str(hf_cache))
        if self.is_main_process:
            hf_cache.mkdir(parents=True, exist_ok=True)
        if self.is_distributed:
            barrier()

        self.model = None
        self.scheduler = None
        self.optimizer = None
        self.scaler = GradScaler(
            enabled=self.use_mixed_precision and self.autocast_dtype == torch.float16
        )
        self.start_step = 0
        self.action_projection = None
        self.action_token_projection = None
        self.use_action_conditioning = True

        # Action critic config
        self.action_critic_enabled = bool(getattr(config, "action_critic_enabled", False))
        self.critic_updates_per_step = int(getattr(config, "critic_updates_per_step", 1))
        critic_dims_cfg = getattr(config, "action_critic_dims", None)
        self.action_critic_dims = list(critic_dims_cfg) if critic_dims_cfg is not None else [2, 7]
        self.action_critic_z_loss_weight = float(
            getattr(config, "action_critic_z_loss_weight",
                    getattr(config, "action_critic_reward_loss_weight", 0.1)))
        self.generator_action_z_guidance_weight = float(
            getattr(config, "generator_action_z_guidance_weight",
                    getattr(config, "generator_action_reward_guidance_weight", 0.0)))
        self.z_guidance_warmup_steps = int(
            getattr(config, "z_guidance_warmup_steps",
                    getattr(config, "reward_guidance_warmup_steps", 500)))
        self.critic_lr = float(getattr(config, "critic_lr", 3e-4))
        self.action_critic = None
        self.critic_optimizer = None
        self._frozen_vae = None
        self._frozen_cotracker = None
        self._frozen_ss_vae = None
        self._frozen_ss_vae_scale = 1.0

        # State-token action head config
        self.state_head_enabled = bool(getattr(config, "state_head_enabled", False))
        self.state_head_loss_weight = float(getattr(config, "state_head_loss_weight", 0.005))
        self.state_head_out_dim = int(getattr(config, "state_head_out_dim", 2))
        self._state_head_built = False

        # Cross-attention state probe config
        self.state_probe_mode = bool(getattr(config, "state_probe_mode", False))
        self.state_probe_dim = int(getattr(config, "state_probe_dim", 256))
        self.state_probe_n_taps = int(getattr(config, "state_probe_n_taps", 6))
        self.state_probe_num_heads = int(getattr(config, "state_probe_num_heads", 8))

        # Frozen-readout state guidance config
        self.state_guidance_weight = float(getattr(config, "state_guidance_weight", 0.0))
        self.state_guidance_warmup_steps = int(getattr(config, "state_guidance_warmup_steps", 500))
        self.state_guidance_action_scale = float(getattr(config, "state_guidance_action_scale", 1.1))

        # LR schedule
        self.warmup_steps = int(getattr(config, "warmup_steps", 0))
        self.base_lr = float(getattr(config, "lr", 1e-4))

        acm = getattr(config, "action_conditioning_mode", "adaln")
        if acm not in ("adaln", "action_tokens", "both"):
            raise ValueError(f"action_conditioning_mode must be adaln|action_tokens|both, got {acm}")
        self.action_conditioning_mode = acm
        self.use_adaln = acm in ("adaln", "both")
        self.use_action_tokens = acm in ("action_tokens", "both")

        # Held-out eval config
        self.test_start_index = int(getattr(config, "test_start_index", 0))
        self.test_num_rides = int(getattr(config, "test_num_rides", 20))
        self.eval_interval = int(getattr(config, "eval_interval", 50))
        self.eval_inference_steps = int(getattr(config, "eval_inference_steps", 48))
        self.grad_norm_interval = int(getattr(config, "grad_norm_interval", 50))

        if self.is_main_process:
            logging.info("Building training dataloader...")
        self._build_dataloader()
        if self.is_main_process:
            logging.info("Training dataloader ready (world_size=%s)", self.world_size)

        self._init_wandb()

        if self.is_main_process:
            logging.info("Building model...")
        self._build_model()

        if self.action_critic_enabled:
            if self.is_main_process:
                logging.info("Building action critic and frozen evaluator modules...")
            self._build_action_critic()

        if self.is_main_process:
            logging.info("Building optimizer...")
        self._build_optimizer()

        pretrained_lora_ckpt = getattr(config, "pretrained_lora_ckpt", None)
        if pretrained_lora_ckpt:
            self._load_pretrained_lora_weights(pretrained_lora_ckpt)

        self._maybe_resume()

        if self.is_main_process:
            logging.info("CausalLoRADiffusionTrainer ready.  start_step=%d", self.start_step)

    @property
    def scratch_root(self) -> Path:
        return Path(os.environ.get("SCRATCH", os.path.expanduser("~")))

    # ------------------------------------------------------------------
    # Dataloader
    # ------------------------------------------------------------------

    def _build_dataloader(self) -> None:
        ss_vae_ckpt = getattr(self.config, "ss_vae_checkpoint", "action_query/checkpoints/ss_vae_8free.pt")
        min_ride_frames = self.streaming_chunk_size + self.context_frames
        max_rides = getattr(self.config, "max_rides", None)
        if max_rides is not None:
            max_rides = int(max_rides)

        manifest_cache = str(self.logdir / ".ride_manifest.pt") if self.logdir else None

        # --- rank-0 builds (or loads) ONE master manifest, other ranks wait ---
        if self.is_main_process:
            all_rides = build_ride_manifest(
                encoded_root=self.config.encoded_root,
                caption_root=self.config.caption_root,
                min_ride_frames=min_ride_frames,
                cache_path=manifest_cache,
            )
            if max_rides is not None:
                all_rides = all_rides[:max_rides]
            if manifest_cache:
                _shared = str(self.logdir / ".ride_manifest_shared.pt")
                torch.save(all_rides, _shared)
                logging.info("Manifest broadcast file written (%d rides).", len(all_rides))
        barrier()

        if not self.is_main_process:
            _shared = str(self.logdir / ".ride_manifest_shared.pt")
            all_rides = torch.load(_shared, map_location="cpu")
            logging.info("Rank %d loaded manifest (%d rides) from broadcast file.",
                         self.global_rank, len(all_rides))
        barrier()

        # --- single-scan split into train / eval ---
        train_start = self.test_start_index + self.test_num_rides
        eval_rides = all_rides[self.test_start_index : self.test_start_index + self.test_num_rides]
        train_rides = all_rides[train_start:] + all_rides[:self.test_start_index]

        self.dataset = ZarrRideDataset.from_manifest(
            rides_data=train_rides,
            motion_root=self.config.motion_root,
            ss_vae_checkpoint=ss_vae_ckpt,
        )
        self.train_dataset_size = len(self.dataset)
        if self.is_main_process:
            logging.info("Training dataset: %d rides (skipping %d held-out)",
                         len(self.dataset), self.test_num_rides)

        sampler = self._make_sampler(self.dataset)
        dl_kwargs = self._dataloader_kwargs(sampler)
        dl_kwargs["batch_size"] = 1
        self.dataloader = DataLoader(self.dataset, **dl_kwargs)
        self.data_iter = cycle(self.dataloader)

        self.eval_dataset = None
        if eval_rides:
            try:
                self.eval_dataset = ZarrRideDataset.from_manifest(
                    rides_data=eval_rides,
                    motion_root=self.config.motion_root,
                    ss_vae_checkpoint=ss_vae_ckpt,
                    _share_ss_vae=self.dataset,
                )
                self.eval_dataset_size = len(self.eval_dataset)
                if self.is_main_process:
                    logging.info("Eval dataset: %d rides (indices %d..%d)",
                                 len(self.eval_dataset), self.test_start_index,
                                 self.test_start_index + self.test_num_rides)
            except Exception as exc:
                if self.is_main_process:
                    logging.warning("Failed to build eval dataset: %s", exc)

    def _make_sampler(self, dataset):
        if self.is_distributed:
            return DistributedSampler(dataset, shuffle=True, drop_last=True)
        from torch.utils.data import RandomSampler
        return RandomSampler(dataset)

    def _dataloader_kwargs(self, sampler) -> dict:
        return {
            "batch_size": int(getattr(self.config, "batch_size", 1)),
            "sampler": sampler,
            "num_workers": self.num_workers,
            "pin_memory": True,
            "drop_last": True,
            "prefetch_factor": 2 if self.num_workers > 0 else None,
        }

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def _build_model(self) -> None:
        model_name = getattr(self.config, "model_name", "Wan2.1-T2V-1.3B")
        num_train_timestep = int(getattr(self.config, "num_train_timestep", 1000))
        num_train_frames = int(getattr(self.config, "num_training_frames", 21))
        gradient_checkpointing = bool(getattr(self.config, "gradient_checkpointing", True))
        model_kwargs = dict(getattr(self.config, "model_kwargs", {}))

        wrapper = WanDiffusionWrapper(
            model_name=model_name,
            is_causal=True,
            **model_kwargs,
        )
        if gradient_checkpointing:
            wrapper.model.enable_gradient_checkpointing()

        self.scheduler = wrapper.get_scheduler()
        self.scheduler.set_timesteps(num_inference_steps=num_train_timestep, denoising_strength=1.0)

        wrapper.model.num_frame_per_block = self.num_frame_per_block
        wrapper.model.context_shift = self.context_frames // self.num_frame_per_block

        model_variant = getattr(self.config, "model_variant", None)
        if model_variant == "action-injection":
            if self.is_main_process:
                logging.info("Applying action injection (mode=%s)", self.action_conditioning_mode)
            apply_action_patches(wrapper)

            model_dim = getattr(wrapper.model, "dim", 2048)
            action_dim = int(getattr(self.config, "raw_action_dim", 8))
            enable_adaln_zero = bool(getattr(self.config, "enable_adaln_zero", True))
            activation = getattr(self.config, "action_activation", "silu")

            if self.use_adaln:
                self.action_projection = ActionModulationProjection(
                    action_dim=action_dim, activation=activation,
                    hidden_dim=model_dim, num_frames=1, zero_init=enable_adaln_zero,
                )
                self.action_projection.to(self.device)
                self.action_projection.train()

            if self.use_action_tokens:
                self.action_token_projection = ActionTokenProjection(
                    action_dim=action_dim, activation=activation,
                    hidden_dim=model_dim, zero_init=enable_adaln_zero,
                )
                self.action_token_projection.to(self.device)
                self.action_token_projection.train()
                wrapper.model.action_tokens_per_frame = 1
                wrapper.adjust_seq_len_for_action_tokens(
                    num_frames=num_train_frames, action_per_frame=1,
                )

            self.use_action_conditioning = True

        lora_cfg = getattr(self.config, "adapter", None)
        if lora_cfg is None:
            raise ValueError("LoRA adapter configuration (config.adapter) is required.")
        self.lora_config = lora_cfg
        wrapper.model = self._apply_lora(wrapper.model, lora_cfg)

        # State-token or cross-attention probe (before DDP wrap, after action tokens)
        if self.state_head_enabled:
            n_chunks = num_train_frames // self.num_frame_per_block
            if self.state_probe_mode:
                wrapper.adding_state_probe_branch(
                    n_chunks=n_chunks,
                    z_out_dim=self.state_head_out_dim,
                    dim=wrapper.model.dim,
                    probe_dim=self.state_probe_dim,
                    num_heads=self.state_probe_num_heads,
                    n_taps=self.state_probe_n_taps,
                    num_frame_per_block=self.num_frame_per_block,
                )
                self._state_head_built = True
                if self.is_main_process:
                    base_m = wrapper.model.get_base_model() if hasattr(wrapper.model, 'get_base_model') else wrapper.model
                    logging.info(
                        "State cross-attention probes: %d chunks, out_dim=%d, "
                        "probe_dim=%d, n_taps=%d, taps=%s, weight=%.4f",
                        n_chunks, self.state_head_out_dim, self.state_probe_dim,
                        self.state_probe_n_taps,
                        getattr(base_m, '_state_probe_tap_indices', []),
                        self.state_head_loss_weight,
                    )
            else:
                wrapper.adding_state_token_branch(
                    n_chunks=n_chunks,
                    z_out_dim=self.state_head_out_dim,
                    dim=wrapper.model.dim,
                    num_frame_per_block=self.num_frame_per_block,
                )
                self._state_head_built = True
                if self.is_main_process:
                    logging.info(
                        "State-token action head: %d chunks, out_dim=%d, weight=%.4f",
                        n_chunks, self.state_head_out_dim, self.state_head_loss_weight,
                    )

        wrapper.to(self.device)
        wrapper.train()
        if self.is_distributed:
            self.model = DDP(
                wrapper, device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False, find_unused_parameters=True,
            )
        else:
            self.model = wrapper

        if self.is_main_process and LOG_GPU_MEMORY:
            log_gpu_memory("After model build", device=self.device, rank=self.global_rank)

    # ------------------------------------------------------------------
    # Action critic helpers
    # ------------------------------------------------------------------

    def _build_action_critic(self) -> None:
        action_dim = len(self.action_critic_dims)
        base_channels = int(getattr(self.config, "action_critic_base_channels", 64))
        num_res_blocks = int(getattr(self.config, "action_critic_num_blocks", 3))
        z_out_dim = int(getattr(self.config, "action_critic_z_out_dim", 8))
        self.action_critic = ActionCritic(
            latent_channels=16,
            action_dim=action_dim,
            z_out_dim=z_out_dim,
            base_channels=base_channels,
            num_res_blocks=num_res_blocks,
            chunk_frames=self.num_frame_per_block,
        ).to(self.device)
        self.action_critic.train()

        if self.is_distributed:
            self.action_critic = DDP(
                self.action_critic, device_ids=[torch.cuda.current_device()],
                output_device=torch.cuda.current_device(),
                broadcast_buffers=False, find_unused_parameters=False,
            )

        self._build_frozen_evaluator_modules()

        if self.is_main_process:
            critic_mod = self.action_critic.module if isinstance(self.action_critic, DDP) else self.action_critic
            n_params = sum(p.numel() for p in critic_mod.parameters())
            logging.info("Action critic: %d params, action_dim=%d", n_params, action_dim)

    def _build_frozen_evaluator_modules(self) -> None:
        self._frozen_vae = WanVAEWrapper()
        self._frozen_vae.eval()
        self._frozen_vae.requires_grad_(False)
        self._frozen_vae.to(self.device)

        # Rank 0 downloads first (populates hub cache), then other ranks
        # load from cache to avoid download races on multi-node.
        if self.is_main_process:
            self._frozen_cotracker = torch.hub.load(
                "facebookresearch/co-tracker", "cotracker3_offline",
            ).to(self.device)
        if self.is_distributed:
            barrier()
        if not self.is_main_process:
            self._frozen_cotracker = torch.hub.load(
                "facebookresearch/co-tracker", "cotracker3_offline",
            ).to(self.device)
        if self.is_distributed:
            barrier()
        self._frozen_cotracker.eval()
        for p in self._frozen_cotracker.parameters():
            p.requires_grad_(False)

        from action_query.ss_vae_model import load_ss_vae
        ss_vae_ckpt = getattr(self.config, "ss_vae_checkpoint", "action_query/checkpoints/ss_vae_8free.pt")
        ss_vae, scale = load_ss_vae(ss_vae_ckpt, device=str(self.device))
        ss_vae.eval()
        ss_vae.requires_grad_(False)
        self._frozen_ss_vae = ss_vae
        self._frozen_ss_vae_scale = scale

        if self.is_main_process:
            logging.info("Frozen evaluator modules ready (VAE, CoTracker, ss_vae)")

    @torch.no_grad()
    def _compute_action_teacher_targets(
        self,
        pred_x0: torch.Tensor,
    ):
        """Produce full 8-D teacher z from pred_x0 via the motion pipeline.

        Returns exactly ``n_chunks = F // chunk_frames`` segments so that
        the outputs align 1-to-1 with the chunkwise critic's predictions.

        Returns:
            teacher_z_8d: ``[B, n_chunks, 8]``
        """
        B, F, C, H, W = pred_x0.shape
        n_chunks = F // self.num_frame_per_block

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
            vid = video[b]
            T_total = vid.shape[0]
            vid_5d = vid.unsqueeze(0)

            motion_windows = []
            for chunk_start in range(0, T_total, compute_T):
                chunk_end = min(chunk_start + compute_T, T_total)
                chunk = vid_5d[:, chunk_start:chunk_end]
                n_out = chunk.shape[1] // output_chunk_size
                if n_out == 0:
                    continue
                used = n_out * output_chunk_size
                chunk = chunk[:, :used].clone()

                with torch.amp.autocast(device_type="cuda", enabled=True):
                    pred_tracks, pred_vis = self._frozen_cotracker(chunk, grid_size=grid_size)

                tracks_w = pred_tracks.reshape(1, n_out, output_chunk_size, N, 2)
                if pred_vis.dim() == 3:
                    vis_w = pred_vis.reshape(1, n_out, output_chunk_size, N).unsqueeze(-1)
                else:
                    vis_w = pred_vis.reshape(1, n_out, output_chunk_size, N, 1)

                d_w = tracks_w[:, :, 1:] - tracks_w[:, :, :-1]
                motion_out = d_w.mean(dim=2)
                vis_out = vis_w.to(dtype=motion_out.dtype).mean(dim=2)
                mw = torch.cat([motion_out, vis_out], dim=-1).squeeze(0)
                motion_windows.append(mw)

            if not motion_windows:
                all_teacher_z.append(torch.zeros(n_chunks, 8, device=pred_x0.device))
                continue

            est_motion = torch.cat(motion_windows, dim=0)
            raw_n = est_motion.shape[0]
            xy = est_motion[:, :, :2].reshape(raw_n, 10, 10, 2)
            x_in = xy.permute(0, 3, 1, 2).float() / self._frozen_ss_vae_scale
            mu, _ = self._frozen_ss_vae.encoder(x_in.to(self.device))
            z_full_8d = mu.squeeze(-1).squeeze(-1)
            z_full_8d = _tanh_squash(z_full_8d)

            z_chunked = self._reduce_to_segments(z_full_8d, n_chunks)
            all_teacher_z.append(z_chunked)

        teacher_z = torch.stack(all_teacher_z, dim=0)
        return teacher_z.detach()

    @torch.no_grad()
    def _compute_teacher_visuals(
        self,
        pred_x0: torch.Tensor,
    ):
        """Like teacher targets but also returns motion for overlays.

        Aligns outputs to the chunkwise contract: one prediction per
        ``chunk_frames``-frame block, giving ``n_chunks = F // chunk_frames``.

        Returns:
            motion: ``[B, n_chunks, N, 3]``
            teacher_z_8d: ``[B, n_chunks, 8]``
        """
        B, F, C, H, W = pred_x0.shape
        n_chunks = F // self.num_frame_per_block

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

        all_motion = []
        all_z = []

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
                mw = torch.cat([mo, vo], dim=-1).squeeze(0)
                mws.append(mw)

            if not mws:
                all_motion.append(torch.zeros(n_chunks, N, 3, device=pred_x0.device))
                all_z.append(torch.zeros(n_chunks, 8, device=pred_x0.device))
                continue

            est_motion = torch.cat(mws, dim=0)
            raw_n = est_motion.shape[0]
            xy = est_motion[:, :, :2].reshape(raw_n, 10, 10, 2)
            x_in = xy.permute(0, 3, 1, 2).float() / self._frozen_ss_vae_scale
            mu, _ = self._frozen_ss_vae.encoder(x_in.to(self.device))
            z8 = _tanh_squash(mu.squeeze(-1).squeeze(-1))

            z8_chunked = self._reduce_to_segments(z8, n_chunks)
            motion_chunked = self._reduce_to_segments(
                est_motion.reshape(raw_n, -1), n_chunks,
            ).reshape(n_chunks, N, 3)

            all_motion.append(motion_chunked)
            all_z.append(z8_chunked)

        motion = torch.stack(all_motion)
        tz = torch.stack(all_z)
        return motion.detach(), tz.detach()

    @staticmethod
    def _reduce_to_segments(per_frame: torch.Tensor, n_seg: int) -> torch.Tensor:
        """Mean-pool ``[F, D]`` into ``[n_seg, D]``."""
        F_len = per_frame.shape[0]
        seg_size = max(1, F_len // n_seg)
        segs = []
        for i in range(n_seg):
            s = i * seg_size
            e = min(s + seg_size, F_len)
            segs.append(per_frame[s:e].mean(dim=0))
        return torch.stack(segs)

    # ------------------------------------------------------------------
    # LoRA helpers
    # ------------------------------------------------------------------

    def _load_pretrained_lora_weights(self, ckpt_path: str) -> None:
        path = Path(ckpt_path)
        if not path.exists():
            if self.is_main_process:
                logging.warning("pretrained_lora_ckpt not found: %s", path)
            return
        if self.is_main_process:
            logging.info("Loading pretrained LoRA weights from %s", path)

        ckpt = torch.load(path, map_location="cpu")
        lora_sd = ckpt.get("lora") or ckpt.get("generator_lora")
        if lora_sd is None:
            if self.is_main_process:
                logging.warning("No 'lora'/'generator_lora' key in %s", path)
            return

        base = self.model.module if isinstance(self.model, DDP) else self.model
        try:
            set_peft_model_state_dict(base.model, lora_sd)
            if self.is_main_process:
                logging.info("Loaded pretrained LoRA (%d keys)", len(lora_sd))
        except Exception as exc:
            if self.is_main_process:
                logging.warning("Failed to load pretrained LoRA: %s (cross-loading)", exc)
            current_sd = get_peft_model_state_dict(base.model)
            matched = 0
            for key in current_sd:
                if key in lora_sd and current_sd[key].shape == lora_sd[key].shape:
                    current_sd[key] = lora_sd[key]
                    matched += 1
            set_peft_model_state_dict(base.model, current_sd)
            if self.is_main_process:
                logging.info("Cross-loaded %d / %d LoRA keys", matched, len(current_sd))

    def _collect_target_modules(self, model) -> list:
        target_modules = set()
        for module_name, module in model.named_modules():
            if module.__class__.__name__ in {"WanAttentionBlock", "CausalWanAttentionBlock"}:
                for full_name, submodule in module.named_modules(prefix=module_name):
                    if isinstance(submodule, torch.nn.Linear):
                        target_modules.add(full_name)
        return sorted(target_modules)

    def _apply_lora(self, model, lora_cfg):
        rank = lora_cfg.get("rank", 128) if isinstance(lora_cfg, dict) else getattr(lora_cfg, "rank", 128)
        alpha = lora_cfg.get("alpha", rank) if isinstance(lora_cfg, dict) else getattr(lora_cfg, "alpha", rank)
        dropout = lora_cfg.get("dropout", 0.0) if isinstance(lora_cfg, dict) else getattr(lora_cfg, "dropout", 0.0)

        target_modules = self._collect_target_modules(model)
        if not target_modules:
            target_modules = ["q", "k", "v", "o"]

        lora_config = LoraConfig(
            r=int(rank), lora_alpha=float(alpha), lora_dropout=float(dropout),
            target_modules=target_modules, bias="none",
        )
        model = peft.get_peft_model(model, lora_config)
        if self.is_main_process:
            logging.info("LoRA applied: rank=%s alpha=%s targets=%s", rank, alpha, target_modules)
        return model

    # ------------------------------------------------------------------
    # Logdir resolution
    # ------------------------------------------------------------------

    def _resolve_logdir(self, logdir_str: str) -> Path:
        def _broadcast_path(p: Path) -> Path:
            if self.is_distributed:
                obj_list = [str(p) if self.is_main_process else None]
                dist.broadcast_object_list(obj_list, src=0)
                return Path(obj_list[0])
            return p

        if logdir_str:
            return _broadcast_path(Path(logdir_str))

        scratch = Path(os.environ.get("SCRATCH", os.path.expanduser("~")))
        return _broadcast_path(scratch / "logs" / "causal_lora_teacher" / datetime.now().strftime("%Y%m%d_%H%M%S"))

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def _build_optimizer(self) -> None:
        params = [p for p in self.model.parameters() if p.requires_grad]

        if self.use_action_conditioning and self.action_projection is not None:
            proj_module = self.action_projection.module if isinstance(self.action_projection, DDP) else self.action_projection
            proj_params = [p for p in proj_module.parameters() if p.requires_grad]
            params.extend(proj_params)
            if self.is_main_process:
                logging.info("Added %d adaLN action projection params to optimizer", len(proj_params))

        if self.use_action_conditioning and self.action_token_projection is not None:
            tok_module = self.action_token_projection.module if isinstance(self.action_token_projection, DDP) else self.action_token_projection
            tok_params = [p for p in tok_module.parameters() if p.requires_grad]
            params.extend(tok_params)
            if self.is_main_process:
                logging.info("Added %d action token projection params to optimizer", len(tok_params))

        if not params:
            raise RuntimeError("No trainable parameters found after applying LoRA.")

        lr = float(getattr(self.config, "lr", 1e-4))
        beta1 = float(getattr(self.config, "beta1", 0.9))
        beta2 = float(getattr(self.config, "beta2", 0.999))
        weight_decay = float(getattr(self.config, "weight_decay", 0.01))
        self.optimizer = torch.optim.AdamW(params, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
        if self.is_main_process:
            logging.info("Generator optimizer: AdamW lr=%.2e betas=(%.3f,%.3f) wd=%.4f params=%d",
                         lr, beta1, beta2, weight_decay, len(params))

        if self.action_critic is not None:
            critic_module = self.action_critic.module if isinstance(self.action_critic, DDP) else self.action_critic
            critic_params = [p for p in critic_module.parameters() if p.requires_grad]
            self.critic_optimizer = torch.optim.AdamW(
                critic_params, lr=self.critic_lr,
                betas=(beta1, beta2), weight_decay=weight_decay,
            )
            if self.is_main_process:
                logging.info("Critic optimizer: AdamW lr=%.2e params=%d",
                             self.critic_lr, len(critic_params))

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------

    def _init_wandb(self) -> None:
        if not self.is_main_process or self.disable_wandb:
            return
        if not self.wandb_project:
            logging.warning("wandb_project not set; disabling W&B logging.")
            self.disable_wandb = True
            return
        if self._wandb_login_key:
            try:
                wandb.login(key=self._wandb_login_key)
            except Exception as exc:
                logging.warning("Failed to login to W&B: %s", exc)

        init_kwargs: Dict[str, Any] = dict(
            project=self.wandb_project,
            name=self.wandb_run_name or self.config_name,
            dir=getattr(self.config, "wandb_save_dir", str(self.logdir)),
        )
        if self.wandb_entity:
            init_kwargs["entity"] = self.wandb_entity
        if self.wandb_group:
            init_kwargs["group"] = self.wandb_group
        if self.wandb_tags:
            init_kwargs["tags"] = self.wandb_tags

        try:
            self.wandb_run = wandb.init(**init_kwargs)
            wandb.config.update({
                "config_name": self.config_name,
                "lr": getattr(self.config, "lr", None),
                "batch_size": getattr(self.config, "batch_size", None),
                "num_frame_per_block": self.num_frame_per_block,
                "teacher_forcing": self.teacher_forcing,
                "noise_aug_max_t": self.noise_augmentation_max_timestep,
                "world_size": self.world_size,
                "max_steps": self.max_steps,
                "train_dataset_size": self.train_dataset_size,
                "eval_dataset_size": self.eval_dataset_size,
                "lora_rank": self.adapter_rank,
                "lora_alpha": self.adapter_alpha,
                "action_conditioning_mode": self.action_conditioning_mode,
                "action_critic_enabled": self.action_critic_enabled,
                "test_num_rides": self.test_num_rides,
                "eval_interval": self.eval_interval,
            }, allow_val_change=True)
            wandb.define_metric("train/step")
            wandb.define_metric("train/*", step_metric="train/step")
            wandb.define_metric("eval/step")
            wandb.define_metric("eval/*", step_metric="eval/step")
        except Exception as exc:
            logging.warning("Failed to initialise W&B: %s", exc)
            self.disable_wandb = True
            self.wandb_run = None

    def _wandb_log(self, payload: Dict[str, Any], step: int) -> None:
        if self.is_main_process and not self.disable_wandb and self.wandb_run is not None:
            wandb.log(payload, step=step)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _checkpoint_path(self, step: int) -> Optional[Path]:
        if self.logdir is None:
            return None
        self.logdir.mkdir(parents=True, exist_ok=True)
        return self.logdir / f"causal_lora_step{step:07d}.pt"

    def _latest_checkpoint(self) -> Optional[Path]:
        if self.logdir is None:
            return None
        checkpoints = sorted(self.logdir.glob("causal_lora_step*.pt"))
        return checkpoints[-1] if checkpoints else None

    def _maybe_resume(self) -> None:
        resume = getattr(self.config, "resume_from", None)
        auto_resume = bool(getattr(self.config, "auto_resume", True))

        checkpoint_path = None
        if resume:
            checkpoint_path = Path(resume)
        elif auto_resume:
            checkpoint_path = self._latest_checkpoint()

        if checkpoint_path is None or not checkpoint_path.exists():
            return

        if self.is_main_process:
            logging.info("Resuming from %s", checkpoint_path)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        expected_config_name = self.load_checkpoint_config_name or self.config_name
        ckpt_config_name = checkpoint.get("config_name")
        mismatch = ckpt_config_name is not None and ckpt_config_name != expected_config_name
        if mismatch and not self.allow_checkpoint_config_mismatch:
            if self.is_main_process:
                logging.warning("Config name mismatch (%s vs %s). Skipping resume.",
                                ckpt_config_name, expected_config_name)
            return

        base = self.model.module if isinstance(self.model, DDP) else self.model
        set_peft_model_state_dict(base.model, checkpoint["lora"])

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        self.start_step = int(checkpoint.get("step", 0))

        if self.action_critic is not None and "action_critic" in checkpoint:
            critic_mod = self.action_critic.module if isinstance(self.action_critic, DDP) else self.action_critic
            missing, unexpected = critic_mod.load_state_dict(checkpoint["action_critic"], strict=False)
            if self.is_main_process:
                if missing or unexpected:
                    logging.warning(
                        "Action critic checkpoint had %d missing, %d unexpected keys (architecture change?)",
                        len(missing), len(unexpected),
                    )
                else:
                    logging.info("Restored action critic from checkpoint")

        if self.critic_optimizer is not None and "critic_optimizer" in checkpoint:
            try:
                self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
                for state in self.critic_optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
                if self.is_main_process:
                    logging.info("Restored critic optimizer from checkpoint")
            except Exception as exc:
                if self.is_main_process:
                    logging.warning("Failed to restore critic optimizer: %s", exc)

        if self.action_projection is not None and "action_projection" in checkpoint:
            self.action_projection.load_state_dict(checkpoint["action_projection"])
            if self.is_main_process:
                logging.info("Restored action modulation projection from checkpoint")

        if self.action_token_projection is not None and "action_token_projection" in checkpoint:
            self.action_token_projection.load_state_dict(checkpoint["action_token_projection"])
            if self.is_main_process:
                logging.info("Restored action token projection from checkpoint")

        if self._state_head_built:
            if hasattr(base, '_state_probe') and "state_probe" in checkpoint:
                missing, unexpected = base._state_probe.load_state_dict(
                    checkpoint["state_probe"], strict=False,
                )
                if self.is_main_process:
                    if missing or unexpected:
                        logging.warning(
                            "State probe: %d missing, %d unexpected keys",
                            len(missing), len(unexpected),
                        )
                    else:
                        logging.info("Restored state_probe from checkpoint")
            elif hasattr(base, '_state_probe'):
                if self.is_main_process:
                    logging.info("No state_probe in checkpoint (training from scratch)")
            else:
                for key, attr_name in [
                    ("state_token_init", "_state_token_init"),
                    ("state_readout", "_state_readout"),
                ]:
                    if key in checkpoint:
                        obj = getattr(base, attr_name)
                        if isinstance(obj, nn.Parameter):
                            saved = checkpoint[key]
                            if isinstance(saved, dict) and "weight" in saved:
                                saved = saved["weight"]
                            if isinstance(saved, torch.Tensor) and saved.shape == obj.shape:
                                obj.data.copy_(saved)
                                if self.is_main_process:
                                    logging.info("Restored %s from checkpoint", key)
                            else:
                                if self.is_main_process:
                                    logging.warning("Shape mismatch for %s, training from scratch", key)
                        else:
                            missing, unexpected = obj.load_state_dict(checkpoint[key], strict=False)
                            if self.is_main_process:
                                if missing or unexpected:
                                    logging.warning(
                                        "State-token %s: %d missing, %d unexpected keys",
                                        key, len(missing), len(unexpected),
                                    )
                                else:
                                    logging.info("Restored %s from checkpoint", key)
                    else:
                        if self.is_main_process:
                            logging.info("No %s in checkpoint (new modules will be trained from scratch)", key)

    def _save_checkpoint(self, step: int, keep_last: int = 3) -> None:
        path = self._checkpoint_path(step)
        if path is None:
            return
        base = self.model.module if isinstance(self.model, DDP) else self.model
        state: Dict[str, Any] = {
            "step": step,
            "lora": get_peft_model_state_dict(base.model),
            "optimizer": self.optimizer.state_dict(),
            "config_name": self.config_name,
        }
        if self.action_critic is not None:
            critic_mod = self.action_critic.module if isinstance(self.action_critic, DDP) else self.action_critic
            state["action_critic"] = critic_mod.state_dict()
        if self.critic_optimizer is not None:
            state["critic_optimizer"] = self.critic_optimizer.state_dict()
        if self.action_projection is not None:
            state["action_projection"] = self.action_projection.state_dict()
        if self.action_token_projection is not None:
            state["action_token_projection"] = self.action_token_projection.state_dict()
        if self._state_head_built:
            if hasattr(base, '_state_probe'):
                state["state_probe"] = base._state_probe.state_dict()
            else:
                state["state_token_init"] = base._state_token_init.data
                state["state_readout"] = base._state_readout.state_dict()
        if self.is_main_process:
            torch.save(state, path)
            logging.info("Saved checkpoint to %s", path)
            self._cleanup_old_checkpoints(keep_last=keep_last)

    def _cleanup_old_checkpoints(self, keep_last: int = 3) -> None:
        if self.logdir is None:
            return
        checkpoints = sorted(self.logdir.glob("causal_lora_step*.pt"))
        if len(checkpoints) <= keep_last:
            return
        for old_ckpt in checkpoints[:-keep_last]:
            try:
                old_ckpt.unlink()
                logging.info("Removed old checkpoint %s", old_ckpt.name)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------

    def _make_blockwise_index(self, batch_size: int, num_frames: int, high: int) -> torch.Tensor:
        index = torch.randint(0, high, (batch_size, num_frames), device=self.device)
        index = index.reshape(batch_size, -1, self.num_frame_per_block)
        index[:, :, 1:] = index[:, :, 0:1]
        return index.reshape(batch_size, num_frames)

    def _sample_timesteps(self, batch_size: int, num_frames: int) -> torch.Tensor:
        timesteps = self.scheduler.timesteps.to(self.device)
        index = self._make_blockwise_index(batch_size, num_frames, len(timesteps))
        return timesteps[index]

    def _make_teacher_context(self, latents, noise, bsz, num_frames):
        if self.teacher_forcing and self.noise_augmentation_max_timestep > 0:
            aug_index = self._make_blockwise_index(bsz, num_frames, self.noise_augmentation_max_timestep)
            aug_timestep = self.scheduler.timesteps[aug_index]
            clean_latent_aug = self.scheduler.add_noise(
                latents.flatten(0, 1), noise.flatten(0, 1), aug_timestep.flatten(0, 1),
            ).view_as(latents)
            return clean_latent_aug, aug_timestep
        return (latents if self.teacher_forcing else None), None

    def _build_conditional(self, prompt_embeds, z_noisy, z_clean, num_frames):
        conditional = {"prompt_embeds": prompt_embeds}
        if self.use_action_conditioning and self.action_projection is not None:
            conditional["_action_modulation"] = self.action_projection(z_noisy, num_frames=num_frames)
            conditional["_action_modulation_clean"] = self.action_projection(z_clean, num_frames=num_frames)
        if self.use_action_conditioning and self.action_token_projection is not None:
            conditional["_action_tokens"] = self.action_token_projection(z_noisy)
            conditional["_action_tokens_clean"] = self.action_token_projection(z_clean)
        return conditional

    def _compute_flow_loss(self, flow_pred, training_target, timesteps, bsz, num_frames):
        flow_loss = F.mse_loss(flow_pred.float(), training_target.float(), reduction="none")
        flow_loss = flow_loss.mean(dim=(2, 3, 4))
        weights = self.scheduler.training_weight(timesteps.flatten(0, 1)).view(bsz, num_frames)
        return (flow_loss * weights).mean()

    def _weighted_z_mse(self, pred_z, target_z):
        """Weighted MSE over 8-D z with 2x weight on action-relevant dims."""
        w = torch.ones(pred_z.shape[-1], device=pred_z.device, dtype=pred_z.dtype)
        for dim_idx in self.action_critic_dims:
            w[dim_idx] = 2.0
        return (w * (pred_z - target_z) ** 2).mean()

    def _compute_action_critic_losses(self, pred_x0, target_action_z, timesteps, current_step):
        """Compute z-predictor training and generator guidance losses.

        Runs ``critic_updates_per_step`` independent critic optimiser steps
        (each with its own zero-grad / backward / clip / step) so the critic
        can track the evolving generator more tightly.  Then computes the
        generator guidance loss (frozen critic, gradients through pred_x0).

        Args:
            pred_x0: ``[B, F, C, H, W]`` predicted clean latents.
            target_action_z: ``[B, F, z_dim]`` commanded action per frame (z2/z7).
            timesteps: ``[B, F]`` diffusion timesteps (same within each block).
            current_step: current training step (for guidance warmup).

        Returns:
            generator_action_loss, logs, teacher_z_8d
        """
        critic_mod = self.action_critic.module if isinstance(self.action_critic, DDP) else self.action_critic
        chunk_frames = self.num_frame_per_block
        B = pred_x0.shape[0]
        n_chunks = pred_x0.shape[1] // chunk_frames

        chunk_t = timesteps[:, ::chunk_frames][:, :n_chunks]
        chunk_actions = _chunk_actions(target_action_z, chunk_frames)[:, :n_chunks]  # [B, n_chunks, 2]

        zero = torch.tensor(0.0, device=pred_x0.device)

        # Teacher targets: full 8D z from motion pipeline (computed once)
        teacher_z_8d = self._compute_action_teacher_targets(pred_x0.detach())
        teacher_z_8d = teacher_z_8d[:, :n_chunks]  # [B, n_chunks, 8]

        pred_x0_detached = pred_x0.detach()

        # --- Multi-step critic training (self-contained optimiser loop) ---
        for _k in range(self.critic_updates_per_step):
            self.critic_optimizer.zero_grad(set_to_none=True)
            pred_z = self.action_critic(pred_x0_detached, chunk_t, chunk_actions)
            pred_z = pred_z[:, :n_chunks]

            critic_z_loss = self._weighted_z_mse(pred_z, teacher_z_8d)
            critic_loss_k = self.action_critic_z_loss_weight * critic_z_loss

            if self.scaler.is_enabled():
                self.scaler.scale(critic_loss_k).backward()
                self.scaler.unscale_(self.critic_optimizer)
            else:
                critic_loss_k.backward()

            if self.grad_clip is not None and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(critic_mod.parameters(), self.grad_clip)

            if self.scaler.is_enabled():
                self.scaler.step(self.critic_optimizer)
            else:
                self.critic_optimizer.step()

        # --- Generator guidance: frozen critic, gradient through pred_x0 ---
        warmup_start = self.warmup_steps
        if current_step < warmup_start:
            guidance_scale = 0.0
        elif self.z_guidance_warmup_steps > 0:
            ramp = min(1.0, (current_step - warmup_start) / self.z_guidance_warmup_steps)
            guidance_scale = ramp * self.generator_action_z_guidance_weight
        else:
            guidance_scale = self.generator_action_z_guidance_weight

        if guidance_scale > 0:
            critic_mod.requires_grad_(False)
            gen_pred_z = critic_mod(pred_x0, chunk_t, chunk_actions)
            gen_pred_z = gen_pred_z[:, :n_chunks]  # [B, n_chunks, 8]

            gen_z2z7 = gen_pred_z[:, :, self.action_critic_dims]  # [B, n_chunks, 2]
            target_z2z7 = 1.1 * chunk_actions  # [B, n_chunks, 2]
            gen_z_loss = F.mse_loss(gen_z2z7, target_z2z7)
            generator_action_loss = guidance_scale * gen_z_loss
            critic_mod.requires_grad_(True)
        else:
            gen_z_loss = zero
            generator_action_loss = zero

        # Per-dim MSE for logging (from the last critic training step)
        with torch.no_grad():
            z2_idx, z7_idx = self.action_critic_dims[0], self.action_critic_dims[1]
            z2_mse = F.mse_loss(pred_z[:, :, z2_idx], teacher_z_8d[:, :, z2_idx]).item()
            z7_mse = F.mse_loss(pred_z[:, :, z7_idx], teacher_z_8d[:, :, z7_idx]).item()

        logs = {
            "train/critic_z_loss": critic_z_loss.detach().item(),
            "train/critic_loss": critic_loss_k.detach().item(),
            "train/critic_z2_mse": z2_mse,
            "train/critic_z7_mse": z7_mse,
            "train/gen_z_loss": gen_z_loss.detach().item() if torch.is_tensor(gen_z_loss) else 0.0,
            "train/gen_action_loss": generator_action_loss.detach().item() if torch.is_tensor(generator_action_loss) else 0.0,
            "train/teacher_z2_mean": teacher_z_8d[:, :, z2_idx].mean().item(),
            "train/teacher_z7_mean": teacher_z_8d[:, :, z7_idx].mean().item(),
            "train/z_guidance_scale": guidance_scale,
        }
        return generator_action_loss, logs, teacher_z_8d

    def _optim_step(self, base_module: torch.nn.Module) -> None:
        # Sync gradients for modules not wrapped in DDP (projections are
        # called outside the DDP-wrapped model, so backward doesn't trigger
        # DDP all-reduce for their parameters).
        if self.is_distributed:
            for mod in (self.action_projection, self.action_token_projection):
                if mod is not None:
                    for p in mod.parameters():
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

        if self.grad_clip is not None and self.grad_clip > 0:
            if self.scaler.is_enabled():
                self.scaler.unscale_(self.optimizer)

            gen_params = list(base_module.parameters())
            if self.action_projection is not None:
                gen_params.extend(self.action_projection.parameters())
            if self.action_token_projection is not None:
                gen_params.extend(self.action_token_projection.parameters())
            torch.nn.utils.clip_grad_norm_(gen_params, self.grad_clip)

        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

    # ------------------------------------------------------------------
    # Gradient norm logging
    # ------------------------------------------------------------------

    def _compute_grad_norms(self) -> Dict[str, float]:
        norms = {}
        base = self.model.module if isinstance(self.model, DDP) else self.model

        lora_grads = [p.grad for p in base.parameters() if p.grad is not None and p.requires_grad]
        if lora_grads:
            norms["grad_norm/lora"] = torch.norm(torch.stack([g.norm() for g in lora_grads])).item()

        if self.action_projection is not None:
            grads = [p.grad for p in self.action_projection.parameters() if p.grad is not None]
            if grads:
                norms["grad_norm/action_modulation"] = torch.norm(torch.stack([g.norm() for g in grads])).item()

        if self.action_token_projection is not None:
            grads = [p.grad for p in self.action_token_projection.parameters() if p.grad is not None]
            if grads:
                norms["grad_norm/action_tokens"] = torch.norm(torch.stack([g.norm() for g in grads])).item()

        if self.action_critic is not None:
            c = self.action_critic.module if isinstance(self.action_critic, DDP) else self.action_critic
            grads = [p.grad for p in c.parameters() if p.grad is not None]
            if grads:
                norms["grad_norm/critic"] = torch.norm(torch.stack([g.norm() for g in grads])).item()

        if self._state_head_built:
            state_params = []
            if hasattr(base, "_state_probe"):
                state_params.extend([p.grad for p in base._state_probe.parameters() if p.grad is not None])
            else:
                if hasattr(base, "_state_token_init") and base._state_token_init.grad is not None:
                    state_params.append(base._state_token_init.grad)
                if hasattr(base, "_state_readout"):
                    state_params.extend([p.grad for p in base._state_readout.parameters() if p.grad is not None])
            if state_params:
                norms["grad_norm/state_tokens"] = torch.norm(torch.stack([g.norm() for g in state_params])).item()

        return norms

    # ------------------------------------------------------------------
    # LR schedule
    # ------------------------------------------------------------------

    def _get_lr_scale(self, step: int) -> float:
        """Linear warmup then cosine decay to zero."""
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    def _update_lr(self, step: int) -> None:
        scale = self._get_lr_scale(step)
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.base_lr * scale
        if self.critic_optimizer is not None:
            for pg in self.critic_optimizer.param_groups:
                pg["lr"] = self.critic_lr * scale

    def _state_guidance_scale(self, step: int) -> float:
        """Compute state-guidance weight with linear warmup after LR warmup."""
        if self.state_guidance_weight <= 0:
            return 0.0
        warmup_start = self.warmup_steps
        if step < warmup_start:
            return 0.0
        if self.state_guidance_warmup_steps > 0:
            ramp = min(1.0, (step - warmup_start) / self.state_guidance_warmup_steps)
            return ramp * self.state_guidance_weight
        return self.state_guidance_weight

    # ------------------------------------------------------------------
    # Periodic evaluation – memory management
    # ------------------------------------------------------------------

    def _offload_training_state(self) -> None:
        """Move optimizer states to CPU and free gradients to reclaim GPU
        memory for the heavier teacher-forced eval pass."""
        base = self.model.module if isinstance(self.model, DDP) else self.model
        base.zero_grad(set_to_none=True)
        if self.action_projection is not None:
            self.action_projection.zero_grad(set_to_none=True)
        if self.action_token_projection is not None:
            self.action_token_projection.zero_grad(set_to_none=True)
        if self.action_critic is not None:
            critic = self.action_critic.module if isinstance(self.action_critic, DDP) else self.action_critic
            critic.zero_grad(set_to_none=True)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()
        if self.critic_optimizer is not None:
            for state in self.critic_optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cpu()

        torch.cuda.empty_cache()

    def _restore_training_state(self) -> None:
        """Move optimizer states back to GPU after eval."""
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        if self.critic_optimizer is not None:
            for state in self.critic_optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

    # ------------------------------------------------------------------
    # Periodic evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_eval(self, wrapper, conditional, context_latents, num_frames):
        """Generate eval frames with teacher-forced clean context.

        Passes ``clean_x=context_latents`` so the model can condition on
        real visual context, matching the training regime.  The caller
        must free enough GPU memory beforehand (see ``_offload_training_state``).

        After the denoising loop finishes, runs one extra forward pass on the
        final generated latents so the returned state-token predictions
        correspond to the actual generated sample, not a stale mid-loop readout.

        Returns:
            latents, state_preds (state_preds is None when state-token branch is absent)
        """
        from utils.scheduler import FlowMatchScheduler

        scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
        scheduler.set_timesteps(num_inference_steps=self.eval_inference_steps, denoising_strength=1.0)
        scheduler.sigmas = scheduler.sigmas.to(self.device)

        B = context_latents.shape[0]
        C, H, W = context_latents.shape[2], context_latents.shape[3], context_latents.shape[4]
        latents = torch.randn([B, num_frames, C, H, W], dtype=torch.float32, device=self.device)

        for t in scheduler.timesteps:
            timestep = t * torch.ones([B, num_frames], device=self.device, dtype=torch.float32)
            with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                model_out = wrapper(
                    latents, conditional, timestep,
                    clean_x=context_latents, aug_t=None,
                )
                flow_pred = model_out[0]
            latents = scheduler.step(
                flow_pred.flatten(0, 1), timestep.flatten(0, 1), latents.flatten(0, 1),
            ).unflatten(dim=0, sizes=flow_pred.shape[:2])

        # Fresh state-token readout on the final generated sample (t=0).
        state_preds = None
        if self._state_head_built:
            t_zero = torch.zeros([B, num_frames], device=self.device, dtype=torch.float32)
            with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                final_out = wrapper(
                    latents, conditional, t_zero,
                    clean_x=context_latents, aug_t=None,
                )
                if isinstance(final_out, tuple) and len(final_out) >= 3:
                    state_preds = final_out[2]

        return latents, state_preds

    @torch.no_grad()
    def _decode_latents(self, latents):
        dummy = latents[:, 0:1]
        lat_wd = torch.cat([dummy, latents], dim=1)
        pixels = self._frozen_vae.decode_to_pixel(lat_wd.float())[:, 1:, ...]
        video = (0.5 * (pixels.float() + 1.0)).clamp(0, 1)
        vid_np = (video[0].cpu().numpy() * 255).astype(np.uint8)
        if vid_np.shape[-1] != 3:
            vid_np = vid_np.transpose(0, 2, 3, 1)
        return vid_np

    def _maybe_eval(self, step: int) -> None:
        if not self.is_main_process:
            return
        if self.eval_dataset is None:
            return
        if self._frozen_vae is None:
            return
        is_first_step = (step == self.start_step)
        if not is_first_step and (step + 1) % self.eval_interval != 0:
            return

        logging.info("Running held-out eval at step %d...", step + 1)

        self._offload_training_state()

        wrapper = self.model.module if isinstance(self.model, DDP) else self.model
        was_training = wrapper.training
        wrapper.eval()

        causal_model = wrapper.model
        if hasattr(causal_model, "base_model"):
            causal_model = causal_model.base_model.model
        saved_mask = getattr(causal_model, "block_mask", None)
        causal_model.block_mask = None

        try:
            ride_idx = step % len(self.eval_dataset)
            ride = self.eval_dataset[ride_idx]
            zarr_path = ride["zarr_path"]
            prompt_embeds_eval = ride["prompt_embeds"].unsqueeze(0).to(self.device, dtype=self.dtype)
            n_lat = ride["n_latent_frames"]

            num_frames = self.streaming_chunk_size
            cf = self.context_frames
            window_total = num_frames + cf

            if n_lat < window_total:
                logging.warning("Eval ride too short (%d < %d), skipping.", n_lat, window_total)
                return

            full_latents = ZarrRideDataset.load_latent_chunk(zarr_path, 0, window_total)
            full_latents = full_latents.unsqueeze(0).to(self.device, dtype=torch.float32)

            z_actions = self.eval_dataset.encode_z_actions_window(
                zarr_path, n_lat, 0, window_total,
            ).unsqueeze(0).to(self.device, dtype=self.dtype)

            context_latents = full_latents[:, :num_frames]

            z_sliced = z_actions
            if self.action_dims is not None:
                z_sliced = z_actions[..., self.action_dims]
            z_noisy = z_sliced[:, cf:]
            z_clean = z_sliced[:, :num_frames]
            target_action_z = z_actions[:, cf:][..., self.action_critic_dims][:, :num_frames]

            conditional = self._build_conditional(prompt_embeds_eval, z_noisy, z_clean, num_frames)

            gen_latents, state_preds = self._generate_eval(wrapper, conditional, context_latents, num_frames)
            video_np = self._decode_latents(gen_latents)

            eval_log: Dict[str, Any] = {"eval/step": step + 1}

            state_z2z7 = None
            if state_preds is not None and self._state_head_built:
                _eval_need_slice = (self.state_head_out_dim <= len(self.action_critic_dims))
                state_z2z7 = state_preds.float() if _eval_need_slice else state_preds.float()[:, :, self.action_critic_dims]

            if self.action_critic is not None:
                critic_mod = self.action_critic.module if isinstance(self.action_critic, DDP) else self.action_critic
                critic_mod.eval()

                motion, teacher_z_8d = self._compute_teacher_visuals(gen_latents)
                n_chunks = teacher_z_8d.shape[1]
                target_chunk = _chunk_actions(target_action_z, self.num_frame_per_block)[:, :n_chunks]

                eval_t = torch.zeros(1, n_chunks, device=self.device)
                critic_pred_z = critic_mod(gen_latents, eval_t, target_chunk)
                critic_pred_z = critic_pred_z[:, :n_chunks]  # [1, n_chunks, 8]

                teacher_z2z7 = teacher_z_8d[:, :, self.action_critic_dims]
                critic_z2z7 = critic_pred_z[:, :, self.action_critic_dims]

                annotated = _annotate_action_video(
                    video_np, motion, teacher_z2z7, critic_z2z7,
                    target_chunk,
                    title=f"eval step {step + 1}",
                    state_z2z7=state_z2z7,
                )
                eval_log["eval/critic_z_mse"] = self._weighted_z_mse(
                    critic_pred_z.float(), teacher_z_8d.float()
                ).item()
                z2_idx, z7_idx = self.action_critic_dims[0], self.action_critic_dims[1]
                eval_log["eval/teacher_z2_mean"] = teacher_z_8d[:, :, z2_idx].mean().item()
                eval_log["eval/teacher_z7_mean"] = teacher_z_8d[:, :, z7_idx].mean().item()
                eval_log["eval/critic_z2_mse"] = F.mse_loss(
                    critic_pred_z[:, :, z2_idx].float(), teacher_z_8d[:, :, z2_idx].float()
                ).item()
                eval_log["eval/critic_z7_mse"] = F.mse_loss(
                    critic_pred_z[:, :, z7_idx].float(), teacher_z_8d[:, :, z7_idx].float()
                ).item()

                if state_z2z7 is not None:
                    teacher_z27_chunked = teacher_z2z7[:, :n_chunks]
                    state_z27_trimmed = state_z2z7[:, :n_chunks]
                    eval_log["eval/state_z2_mse"] = F.mse_loss(
                        state_z27_trimmed[:, :, 0], teacher_z27_chunked[:, :, 0]
                    ).item()
                    eval_log["eval/state_z7_mse"] = F.mse_loss(
                        state_z27_trimmed[:, :, 1], teacher_z27_chunked[:, :, 1]
                    ).item()
                if state_preds is not None and self._state_head_built:
                    state_trimmed = state_preds.float()[:, :n_chunks]
                    if _eval_need_slice:
                        teacher_matched = teacher_z2z7[:, :n_chunks]
                    else:
                        teacher_matched = teacher_z_8d[:, :n_chunks]
                    eval_log["eval/state_mse"] = F.mse_loss(
                        state_trimmed, teacher_matched.float()
                    ).item()

                    state_z27_eval = state_trimmed if _eval_need_slice else state_trimmed[:, :, self.action_critic_dims]
                    teacher_z27_eval = teacher_z2z7[:, :n_chunks]
                    cmd_z27_eval = target_chunk
                    eval_log["eval/corr_state_teacher_z27"] = _safe_corr(state_z27_eval, teacher_z27_eval)
                    eval_log["eval/corr_state_cmd_z27"] = _safe_corr(state_z27_eval, cmd_z27_eval)
                    eval_log["eval/mse_teacher_cmd_z27"] = F.mse_loss(
                        teacher_z27_eval.float(), cmd_z27_eval.float()
                    ).item()

                critic_mod.train()
            else:
                annotated = video_np

            tmp_path = None
            mp4_bytes = _frames_to_mp4_bytes(annotated, fps=5.0)
            if mp4_bytes is not None:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                    f.write(mp4_bytes)
                    tmp_path = f.name
                eval_log["eval/video"] = wandb.Video(tmp_path, fps=5, format="mp4",
                                                     caption=f"step {step + 1}")

            self._wandb_log(eval_log, step=step + 1)
            logging.info("Eval video logged to W&B at step %d", step + 1)

            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        except Exception as exc:
            logging.warning("Eval failed at step %d: %s", step + 1, exc, exc_info=True)
        finally:
            causal_model.block_mask = saved_mask
            if was_training:
                wrapper.train()
            self._restore_training_state()
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        if self.start_step >= self.max_steps:
            if self.is_main_process:
                logging.info("start_step >= max_iters, nothing to train.")
            return
        self._train_streaming()

    def _train_streaming(self) -> None:
        forward_model = self.model
        base_module = self.model.module if isinstance(self.model, DDP) else self.model
        base_module.train()

        micro_batch = int(getattr(self.config, "batch_size", 1))
        max_windows_per_ride = getattr(self.config, "max_windows_per_ride", None)
        if max_windows_per_ride is not None:
            max_windows_per_ride = int(max_windows_per_ride)

        batcher = LockstepRideBatcher(
            window_size=self.streaming_chunk_size,
            num_frame_per_block=self.num_frame_per_block,
            batch_size=micro_batch,
            max_windows_per_ride=max_windows_per_ride,
            context_frames=self.context_frames,
        )

        total_batch_size = micro_batch * self.gradient_accumulation * self.world_size
        if self.is_main_process:
            logging.info(
                "Starting training (lockstep streaming): "
                "global_batch=%d (micro=%d x accum=%d x world=%d), window=%d, block=%d",
                total_batch_size, micro_batch, self.gradient_accumulation,
                self.world_size, self.streaming_chunk_size, self.num_frame_per_block,
            )

        def _encode_z_window(zarr_path, n_latent_frames, start, end):
            return self.dataset.encode_z_actions_window(zarr_path, n_latent_frames, start, end)

        def _next_ride() -> dict:
            raw = next(self.data_iter)
            return {
                "zarr_path": raw["zarr_path"][0],
                "prompt_embeds": raw["prompt_embeds"][0],
                "n_latent_frames": int(raw["n_latent_frames"][0].item()),
            }

        def _refill_exhausted() -> None:
            """Replace only the slots whose rides have run out."""
            needs = batcher.exhausted_slot_indices()
            if not needs:
                return
            rides = [_next_ride() for _ in needs]
            batcher.refill_slots(needs, rides)
            if self.is_main_process:
                logging.info("Refilled %d slot(s): %s", len(needs), batcher.summary())

        for step in range(self.start_step, self.max_steps):
            if self.is_distributed:
                sampler = self.dataloader.sampler
                if isinstance(sampler, DistributedSampler):
                    sampler.set_epoch(step)

            self._update_lr(step)
            self.optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            accumulated_flow_loss = 0.0
            accumulated_state_loss = 0.0
            accumulated_state_guidance_loss = 0.0
            diag_corr_state_teacher = 0.0
            diag_corr_state_cmd = 0.0
            diag_mse_teacher_cmd = 0.0
            diag_count = 0
            windows_this_step = 0
            critic_logs: Dict[str, Any] = {}

            cf = self.context_frames
            num_frames = self.streaming_chunk_size

            for _ in range(self.gradient_accumulation):
                _refill_exhausted()

                bsz = micro_batch

                full_latents = batcher.load_latent_batch(self.device)
                z_actions_full = batcher.load_z_actions_batch(
                    self.device, dtype=self.dtype, encode_fn=_encode_z_window,
                )
                prompt_embeds = batcher.load_prompt_embeds_batch(self.device, dtype=self.dtype)

                context_latents = full_latents[:, :num_frames]
                target_latents = full_latents[:, cf:]

                z_actions_full_raw = z_actions_full
                z_sliced = z_actions_full
                if self.action_dims is not None:
                    z_sliced = z_actions_full[..., self.action_dims]
                z_noisy = z_sliced[:, cf:]
                z_clean = z_sliced[:, :num_frames]

                conditional = self._build_conditional(prompt_embeds, z_noisy, z_clean, num_frames)

                timesteps = self._sample_timesteps(bsz, num_frames)

                noise = torch.randn_like(target_latents)
                noisy_latents = self.scheduler.add_noise(
                    target_latents.flatten(0, 1), noise.flatten(0, 1), timesteps.flatten(0, 1),
                ).view_as(target_latents)

                training_target = self.scheduler.training_target(
                    target_latents.flatten(0, 1), noise.flatten(0, 1), timesteps.flatten(0, 1),
                ).view_as(target_latents)

                clean_latent_aug, aug_timestep = self._make_teacher_context(
                    context_latents, noise, bsz, num_frames,
                )

                with autocast(dtype=self.autocast_dtype, enabled=self.use_mixed_precision):
                    model_out = forward_model(
                        noisy_latents, conditional, timesteps,
                        clean_x=clean_latent_aug, aug_t=aug_timestep,
                    )
                    if isinstance(model_out, tuple) and len(model_out) == 4:
                        flow_pred, pred_x0, state_preds, state_pooled = model_out
                    elif isinstance(model_out, tuple) and len(model_out) == 3:
                        flow_pred, pred_x0, state_preds = model_out
                        state_pooled = None
                    else:
                        flow_pred, pred_x0 = model_out[:2]
                        state_preds = None
                        state_pooled = None

                    flow_loss = self._compute_flow_loss(flow_pred, training_target, timesteps, bsz, num_frames)
                    loss = flow_loss

                    teacher_z_8d = None
                    if self.action_critic_enabled and self.action_critic is not None:
                        target_action_z = z_actions_full_raw[:, cf:][..., self.action_critic_dims]
                        gen_loss, critic_logs, teacher_z_8d = self._compute_action_critic_losses(
                            pred_x0, target_action_z, timesteps, step,
                        )
                        loss = loss + gen_loss

                    # State-token head loss: supervise against teacher z-features.
                    # When state_head_out_dim == 2, target is z2/z7 only;
                    # when 8, target is the full 8D teacher latent.
                    state_guidance_loss_val = 0.0
                    if state_preds is not None and self._state_head_built:
                        n_c_g = num_frames // self.num_frame_per_block
                        _need_z_slice = (self.state_head_out_dim <= len(self.action_critic_dims))
                        if teacher_z_8d is not None:
                            t_z = teacher_z_8d[:, :n_c_g]
                            state_target = (t_z[:, :, self.action_critic_dims] if _need_z_slice else t_z).detach()
                        else:
                            target_z_pf = z_actions_full_raw[:, cf:]
                            t_z = _chunk_actions(target_z_pf, self.num_frame_per_block)[:, :n_c_g]
                            state_target = t_z[:, :, self.action_critic_dims] if _need_z_slice else t_z
                        state_z = state_preds[:, :n_c_g].float()
                        state_loss = F.mse_loss(state_z, state_target.float())
                        loss = loss + self.state_head_loss_weight * state_loss

                        # Frozen-readout generator guidance: compare readout
                        # predictions to commanded actions (z2/z7 only).
                        # Gradients flow through the generator (via pooled hidden
                        # states) but not through the readout head weights.
                        state_g_scale = self._state_guidance_scale(step)
                        if state_g_scale > 0 and state_pooled is not None:
                            readout = (base_module._state_probe.readout
                                       if hasattr(base_module, '_state_probe')
                                       else base_module._state_readout)
                            frozen_preds_raw = F.linear(
                                state_pooled[:, :n_c_g].float(),
                                readout.weight.detach(),
                                readout.bias.detach(),
                            )
                            frozen_preds = frozen_preds_raw if _need_z_slice else frozen_preds_raw[:, :, self.action_critic_dims]
                            cmd_z = _chunk_actions(
                                z_actions_full_raw[:, cf:][..., self.action_critic_dims],
                                self.num_frame_per_block,
                            )[:, :n_c_g]
                            cmd_target = self.state_guidance_action_scale * cmd_z
                            state_guidance_loss = F.mse_loss(frozen_preds, cmd_target.float())
                            loss = loss + state_g_scale * state_guidance_loss
                            state_guidance_loss_val = state_guidance_loss.detach().item()

                        # Diagnostics: correlation & MSE between state preds, teacher, and commands
                        if teacher_z_8d is not None and self.is_main_process:
                            teacher_z27 = teacher_z_8d[:, :n_c_g, self.action_critic_dims].detach()
                            state_z27 = state_z if _need_z_slice else state_z[:, :, self.action_critic_dims]
                            cmd_z27 = _chunk_actions(
                                z_actions_full_raw[:, cf:][..., self.action_critic_dims],
                                self.num_frame_per_block,
                            )[:, :n_c_g]
                            diag_corr_state_teacher += _safe_corr(state_z27, teacher_z27)
                            diag_corr_state_cmd += _safe_corr(state_z27, cmd_z27)
                            diag_mse_teacher_cmd += F.mse_loss(teacher_z27.float(), cmd_z27.float()).item()
                            diag_count += 1

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {step + 1}")

                accumulated_loss += loss.detach().item()
                accumulated_flow_loss += flow_loss.detach().item()
                if state_preds is not None and self._state_head_built:
                    accumulated_state_loss += state_loss.detach().item()
                    accumulated_state_guidance_loss += state_guidance_loss_val
                scaled_loss = loss / self.gradient_accumulation

                if self.scaler.is_enabled():
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                batcher.advance()
                windows_this_step += 1

            # Gradient norms (before optimizer step clips them)
            grad_norms = {}
            if self.is_main_process and (step + 1) % self.grad_norm_interval == 0:
                grad_norms = self._compute_grad_norms()

            self._optim_step(base_module)

            # Logging
            if (step + 1) % self.log_interval == 0:
                avg_loss = accumulated_loss / self.gradient_accumulation
                avg_flow = accumulated_flow_loss / self.gradient_accumulation
                loss_tensor = torch.tensor(avg_loss, device=self.device)
                flow_tensor = torch.tensor(avg_flow, device=self.device)
                if self.is_distributed:
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                    dist.all_reduce(flow_tensor, op=dist.ReduceOp.AVG)

                if self.is_main_process:
                    logging.info("step %d | loss %.6f | flow %.6f", step + 1, loss_tensor.item(), flow_tensor.item())

                    try:
                        current_lr = self.optimizer.param_groups[0]["lr"]
                    except (IndexError, KeyError):
                        current_lr = None
                    try:
                        critic_current_lr = self.critic_optimizer.param_groups[0]["lr"] if self.critic_optimizer else None
                    except (IndexError, KeyError):
                        critic_current_lr = None

                    payload: Dict[str, Any] = {
                        "train/step": step + 1,
                        "train/total_loss": loss_tensor.item(),
                        "train/flow_loss": flow_tensor.item(),
                        "train/learning_rate": current_lr,
                        "train/critic_learning_rate": critic_current_lr,
                        "train/windows": windows_this_step,
                        "train/window_idx": batcher.current_window_idx,
                    }
                    if self._state_head_built and accumulated_state_loss > 0:
                        avg_state = accumulated_state_loss / self.gradient_accumulation
                        payload["train/state_z_loss"] = avg_state
                    if accumulated_state_guidance_loss > 0:
                        payload["train/state_guidance_loss"] = accumulated_state_guidance_loss / self.gradient_accumulation
                        payload["train/state_guidance_scale"] = self._state_guidance_scale(step)
                    if diag_count > 0:
                        payload["train/corr_state_teacher_z27"] = diag_corr_state_teacher / diag_count
                        payload["train/corr_state_cmd_z27"] = diag_corr_state_cmd / diag_count
                        payload["train/mse_teacher_cmd_z27"] = diag_mse_teacher_cmd / diag_count
                    if critic_logs:
                        critic_logs.pop("train/flow_loss", None)
                        payload.update(critic_logs)
                    if grad_norms:
                        payload.update(grad_norms)

                    self._wandb_log(payload, step=step + 1)

            # Checkpoint
            if self.ckpt_interval > 0 and (step + 1) % self.ckpt_interval == 0:
                barrier()
                self._save_checkpoint(step + 1)
                barrier()

            # Periodic eval
            self._maybe_eval(step)

            if step == self.start_step:
                torch.cuda.empty_cache()

        barrier()
        self._save_checkpoint(self.max_steps)
        barrier()

        self._maybe_eval(self.max_steps - 1)


def main():
    parser = argparse.ArgumentParser(description="LoRA finetuning for the causal Wan diffusion model.")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--logdir", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--no-auto-resume", action="store_true")
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    default_config_path = Path(__file__).parent.parent / "configs" / "default_config.yaml"
    if default_config_path.exists():
        default_config = OmegaConf.load(default_config_path)
        config = OmegaConf.merge(default_config, config)
    if args.logdir:
        config.logdir = args.logdir
    if args.resume:
        config.resume_from = args.resume
    if args.no_auto_resume:
        config.auto_resume = False

    trainer = CausalLoRADiffusionTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
