"""CausalLoRADiffusionTrainer: causal (autoregressive) diffusion teacher training.

Forked from trainer/diffusion_train.py with the following key changes:
  - CausalWanModel (is_causal=True) with block-wise causal attention
  - ZarrSequentialDataset: zarr latents iterated sequentially, actions from ss_vae
  - Per-block independent timestep sampling (BSMNTW-weighted flow loss)
  - Teacher forcing: model sees clean context (clean_x + aug_t)
  - I2V first-frame conditioning: first latent frame passed as image_latent
  - Action conditioning: 8D tanh-squashed ss_vae z-latent via adaLN-Zero
  - pretrained_lora_ckpt: loads LoRA weights only (no optimizer, no step)

GAN loss and domain-shift evaluation are removed; they are not needed for the
AR diffusion teacher stage.
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

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

from utils.zarr_dataset import ZarrSequentialDataset, ZarrRideDataset
from utils.dataset import cycle
from utils.distributed import barrier, launch_distributed_job
from utils.misc import set_seed
from utils.memory import log_gpu_memory
from utils.debug_option import LOG_GPU_MEMORY
from utils.wan_wrapper import WanDiffusionWrapper

from model.action_modulation import ActionModulationProjection
from model.action_model_patch import apply_action_patches
from model.causal_teacher_streaming import LockstepRideBatcher


def _is_distributed() -> bool:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    return world > 1


class CausalLoRADiffusionTrainer:
    """LoRA finetuning for the causal (autoregressive) Wan diffusion model.

    Trains Stage 1 of the Causal-Forcing pipeline: an AR diffusion model with
    teacher forcing, block-wise causal attention, and action conditioning via
    the ss_vae 8D motion latent.
    """

    def __init__(self, config):
        self.config = config
        self.world_size = 1
        self.global_rank = 0
        self.is_distributed = _is_distributed()

        if self.is_distributed:
            launch_distributed_job()
            self.global_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            torch.cuda.set_device(torch.cuda.current_device())

        self.is_main_process = self.global_rank == 0
        self.scratch_root = Path("/scratch/u5as/as1748.u5as")

        self.device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.use_mixed_precision = bool(getattr(config, "mixed_precision", False))
        self.autocast_dtype = (
            torch.bfloat16
            if getattr(config, "autocast_dtype", "bf16") == "bf16"
            else torch.float16
        )
        self.dtype = torch.bfloat16 if self.use_mixed_precision else torch.float32
        self.grad_clip = getattr(config, "max_grad_norm", None)
        self.gradient_accumulation = max(1, getattr(config, "gradient_accumulation_steps", 1))
        self.max_steps = int(getattr(config, "max_iters", 1000))
        self.log_interval = int(getattr(config, "log_interval", 50))
        self.ckpt_interval = int(getattr(config, "ckpt_interval", 100))
        self.num_workers = int(getattr(config, "num_workers", 4))
        self.pin_memory = bool(getattr(config, "pin_memory", True))
        self.prefetch_factor = getattr(config, "prefetch_factor", None)
        self.allow_checkpoint_config_mismatch = bool(
            getattr(config, "allow_checkpoint_config_mismatch", True)
        )
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

        # Causal-specific
        self.teacher_forcing = bool(getattr(config, "teacher_forcing", True))
        self.num_frame_per_block = int(getattr(config, "num_frame_per_block", 3))
        self.noise_augmentation_max_timestep = int(
            getattr(config, "noise_augmentation_max_timestep", 0)
        )

        # Streaming training
        self.streaming_training = bool(getattr(config, "streaming_training", False))
        self.streaming_chunk_size = int(getattr(config, "streaming_chunk_size", 21))
        self.streaming_min_new_frame = int(
            getattr(config, "streaming_min_new_frame",
                    self.streaming_chunk_size - self.num_frame_per_block)
        )
        self.streaming_max_length = getattr(config, "streaming_max_length", None)
        if self.streaming_max_length is not None:
            self.streaming_max_length = int(self.streaming_max_length)

        set_seed(int(getattr(config, "seed", 0)) + self.global_rank)

        if self.is_main_process:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)s | %(message)s",
            )
            logging.info("Initialising CausalLoRADiffusionTrainer...")
            logging.info("Config (keys only): %s", list(config.keys()))

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
        self.use_action_conditioning = True

        if self.is_main_process:
            logging.info("Building training dataloader...")
        self._build_dataloader()
        if self.is_main_process:
            logging.info("Training dataloader ready (world_size=%s)", self.world_size)

        self._init_wandb()

        if self.is_main_process:
            logging.info("Building model...")
        self._build_model()

        # Load pretrained bidirectional LoRA weights BEFORE building optimizer
        self._load_pretrained_lora_weights()

        if self.is_main_process:
            logging.info("Initialising optimizer (fresh)...")
        self._build_optimizer()

        if self.is_main_process:
            logging.info("Checking for resume checkpoints...")
        self._maybe_resume()
        if self.is_main_process:
            logging.info("Resume logic completed (start_step=%s)", self.start_step)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _build_dataloader(self) -> None:
        ss_vae_ckpt = getattr(self.config, "ss_vae_checkpoint", "action_query/checkpoints/ss_vae_8free.pt")

        if self.streaming_training:
            self._build_streaming_dataloader(ss_vae_ckpt)
        else:
            self._build_window_dataloader(ss_vae_ckpt)

    def _build_window_dataloader(self, ss_vae_ckpt: str) -> None:
        """Fixed-window dataloader (original behaviour)."""
        window_size = int(getattr(self.config, "window_size", 21))
        window_stride = int(getattr(self.config, "window_stride", 1))

        dataset = ZarrSequentialDataset(
            encoded_root=self.config.encoded_root,
            caption_root=self.config.caption_root,
            motion_root=self.config.motion_root,
            ss_vae_checkpoint=ss_vae_ckpt,
            window_size=window_size,
            window_stride=window_stride,
            device="cpu",
        )
        self.train_dataset_size = len(dataset)
        if self.is_main_process:
            logging.info("Training dataset size: %d windows", len(dataset))

        sampler = self._make_sampler(dataset)
        self.dataloader = DataLoader(
            dataset, **self._dataloader_kwargs(sampler),
        )
        self.data_iter = cycle(self.dataloader)

    def _build_streaming_dataloader(self, ss_vae_ckpt: str) -> None:
        """Ride-level dataloader for streaming training.

        The dataloader yields individual rides (batch_size=1 at the
        DataLoader level).  Grouping into ``[B, ...]`` batches is
        handled by ``LockstepRideBatcher`` inside ``_train_streaming``.
        """
        min_ride_frames = self.streaming_chunk_size
        max_rides = getattr(self.config, "max_rides", None)
        if max_rides is not None:
            max_rides = int(max_rides)

        dataset = ZarrRideDataset(
            encoded_root=self.config.encoded_root,
            caption_root=self.config.caption_root,
            motion_root=self.config.motion_root,
            ss_vae_checkpoint=ss_vae_ckpt,
            min_ride_frames=min_ride_frames,
            device="cpu",
        )
        self.train_dataset_size = len(dataset)
        if self.is_main_process:
            logging.info("Training dataset size: %d rides (streaming)", len(dataset))

        sampler = self._make_sampler(dataset)
        dl_kwargs = self._dataloader_kwargs(sampler)
        dl_kwargs["batch_size"] = 1
        self.dataloader = DataLoader(dataset, **dl_kwargs)
        self.data_iter = cycle(self.dataloader)

    def _make_sampler(self, dataset):
        if self.is_distributed:
            return DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.global_rank,
                shuffle=False,
                drop_last=True,
            )
        return SequentialSampler(dataset)

    def _dataloader_kwargs(self, sampler) -> Dict[str, Any]:
        kw: Dict[str, Any] = dict(
            batch_size=int(getattr(self.config, "batch_size", 1)),
            sampler=sampler,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )
        if self.prefetch_factor is not None:
            kw["prefetch_factor"] = self.prefetch_factor
        return kw

    def _build_model(self) -> None:
        model_name = getattr(self.config, "model_name", None)
        if model_name is None:
            model_name = getattr(self.config, "real_name", "Wan2.1-T2V-1.3B")
        wrapper_kwargs = dict(getattr(self.config, "model_kwargs", {}))

        # Causal model
        wrapper = WanDiffusionWrapper(
            model_name=model_name,
            is_causal=True,
            **wrapper_kwargs,
        )

        if getattr(self.config, "gradient_checkpointing", False):
            wrapper.enable_gradient_checkpointing()

        # Set num_frame_per_block on the underlying CausalWanModel
        wrapper.model.num_frame_per_block = self.num_frame_per_block

        # Apply LoRA FIRST (before action patches and DDP)
        lora_cfg = getattr(self.config, "adapter", None)
        if lora_cfg is None:
            raise ValueError("LoRA adapter configuration (config.adapter) is required.")
        self.lora_config = lora_cfg
        wrapper.model = self._apply_lora(wrapper.model, lora_cfg)

        # Action injection
        model_variant = getattr(self.config, "model_variant", None)
        if model_variant == "action-injection":
            if self.is_main_process:
                logging.info("Applying action injection to causal model")

            apply_action_patches(wrapper)

            model_dim = getattr(wrapper.model, "dim", 2048)
            action_dim = int(getattr(self.config, "raw_action_dim", 8))
            enable_adaln_zero = bool(getattr(self.config, "enable_adaln_zero", True))
            activation = getattr(self.config, "action_activation", "silu")

            self.action_projection = ActionModulationProjection(
                action_dim=action_dim,
                activation=activation,
                hidden_dim=model_dim,
                num_frames=1,
                zero_init=enable_adaln_zero,
            )
            self.action_projection.to(self.device)
            self.action_projection.train()
            self.use_action_conditioning = True

            if self.is_main_process:
                logging.info(
                    "Action-injection enabled: action_dim=%d, activation=%s, adaln_zero=%s",
                    action_dim,
                    activation,
                    enable_adaln_zero,
                )

        wrapper.to(self.device)
        wrapper.train()

        self.model = wrapper
        self.scheduler = self.model.get_scheduler()
        self.scheduler.set_timesteps(
            int(getattr(self.config, "num_train_timestep", 1000)),
            training=True,
        )

        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[torch.cuda.current_device()],
                output_device=torch.cuda.current_device(),
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
            if self.use_action_conditioning and self.action_projection is not None:
                self.action_projection = DDP(
                    self.action_projection,
                    device_ids=[torch.cuda.current_device()],
                    output_device=torch.cuda.current_device(),
                    broadcast_buffers=False,
                    find_unused_parameters=False,
                )

        if self.is_main_process and LOG_GPU_MEMORY:
            log_gpu_memory("After model build", device=self.device, rank=self.global_rank)

    def _load_pretrained_lora_weights(self) -> None:
        """Load LoRA weights from a pretrained (bidirectional) checkpoint.

        Only the 'lora' (or 'generator_lora') key is loaded.
        The optimizer and training step are NOT restored -- this is purely a
        weight initialisation step.  Auto-resume of causal training runs is
        handled separately by _maybe_resume().
        """
        path = getattr(self.config, "pretrained_lora_ckpt", None)
        if not path:
            if self.is_main_process:
                logging.info("No pretrained_lora_ckpt specified; starting LoRA from scratch.")
            return

        if self.is_main_process:
            logging.info("Loading pretrained LoRA weights from %s", path)

        ckpt = torch.load(path, map_location="cpu")
        lora_state = ckpt.get("lora") or ckpt.get("generator_lora")
        if lora_state is None:
            raise ValueError(
                f"pretrained_lora_ckpt {path} has no 'lora' or 'generator_lora' key. "
                f"Found keys: {list(ckpt.keys())}"
            )

        base = self.model.module if isinstance(self.model, DDP) else self.model
        set_peft_model_state_dict(base.model, lora_state)

        if self.is_main_process:
            logging.info(
                "Loaded pretrained LoRA weights (%d tensors) from %s "
                "(optimizer NOT loaded; starting fresh)",
                len(lora_state),
                path,
            )

    def _collect_target_modules(self, transformer: torch.nn.Module) -> list:
        target_modules: set = set()
        for module_name, module in transformer.named_modules():
            cls_name = module.__class__.__name__
            if cls_name in {"WanAttentionBlock", "CausalWanAttentionBlock"}:
                for full_name, sub_module in module.named_modules(prefix=module_name):
                    if isinstance(sub_module, torch.nn.Linear):
                        target_modules.add(full_name)
        if not target_modules:
            raise RuntimeError("Failed to locate target Linear modules for LoRA.")
        if self.is_main_process:
            logging.info("Applying LoRA to %d Linear layers in attention blocks", len(target_modules))
        return sorted(target_modules)

    def _apply_lora(self, transformer: torch.nn.Module, lora_cfg) -> torch.nn.Module:
        target_modules = self._collect_target_modules(transformer)
        rank = int(lora_cfg.get("rank", 16))
        alpha = int(lora_cfg.get("alpha", rank))
        dropout = float(lora_cfg.get("dropout", 0.0))
        init_type = lora_cfg.get("init_lora_weights", "gaussian")

        peft_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            init_lora_weights=init_type,
        )
        lora_model = peft.get_peft_model(transformer, peft_config)
        if self.is_main_process:
            lora_model.print_trainable_parameters()
        return lora_model

    def _resolve_logdir(self, logdir_value: str) -> Path:
        base = self.scratch_root.resolve()
        default_root = base / "frodobots" / "causal_lora_runs"

        def _broadcast_path(path_str: str) -> Path:
            if self.is_distributed and dist.is_initialized():
                obj = [path_str]
                dist.broadcast_object_list(obj, src=0)
                path_str = obj[0]
            return Path(path_str).resolve()

        if logdir_value:
            if self.is_distributed and dist.is_initialized():
                if self.is_main_process:
                    candidate = Path(logdir_value).expanduser()
                    if not candidate.is_absolute():
                        candidate = base / candidate
                    resolved_str = str(candidate.resolve())
                else:
                    resolved_str = ""
                resolved = _broadcast_path(resolved_str)
            else:
                candidate = Path(logdir_value).expanduser()
                if not candidate.is_absolute():
                    candidate = base / candidate
                resolved = candidate.resolve()
        else:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            if self.is_distributed and dist.is_initialized():
                if self.is_main_process:
                    config_name = getattr(self.config, "config_name", "unnamed")
                    resolved_str = str((default_root / f"{config_name}_{timestamp}").resolve())
                else:
                    resolved_str = ""
                resolved = _broadcast_path(resolved_str)
            else:
                config_name = getattr(self.config, "config_name", "unnamed")
                resolved = (default_root / f"{config_name}_{timestamp}").resolve()

        return resolved

    def _build_optimizer(self) -> None:
        params = [p for p in self.model.parameters() if p.requires_grad]

        if self.use_action_conditioning and self.action_projection is not None:
            proj_module = (
                self.action_projection.module
                if isinstance(self.action_projection, DDP)
                else self.action_projection
            )
            proj_params = [p for p in proj_module.parameters() if p.requires_grad]
            params.extend(proj_params)
            if self.is_main_process:
                logging.info("Added %d action projection parameters to optimizer", len(proj_params))

        if not params:
            raise RuntimeError("No trainable parameters found after applying LoRA.")

        lr = float(getattr(self.config, "lr", 1e-4))
        beta1 = float(getattr(self.config, "beta1", 0.9))
        beta2 = float(getattr(self.config, "beta2", 0.999))
        weight_decay = float(getattr(self.config, "weight_decay", 0.01))
        self.optimizer = torch.optim.AdamW(
            params, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay
        )
        if self.is_main_process:
            logging.info(
                "Optimizer: AdamW lr=%.2e betas=(%.3f,%.3f) wd=%.4f params=%d",
                lr, beta1, beta2, weight_decay, len(params),
            )

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
            wandb.config.update(
                {
                    "config_name": self.config_name,
                    "lr": getattr(self.config, "lr", None),
                    "batch_size": getattr(self.config, "batch_size", None),
                    "num_frame_per_block": self.num_frame_per_block,
                    "teacher_forcing": self.teacher_forcing,
                    "noise_aug_max_t": self.noise_augmentation_max_timestep,
                    "world_size": self.world_size,
                    "max_steps": self.max_steps,
                    "train_dataset_size": self.train_dataset_size,
                    "lora_rank": self.adapter_rank,
                    "lora_alpha": self.adapter_alpha,
                },
                allow_val_change=True,
            )
            wandb.define_metric("train/step")
            wandb.define_metric("train/*", step_metric="train/step")
        except Exception as exc:
            logging.warning("Failed to initialise W&B: %s", exc)
            self.disable_wandb = True
            self.wandb_run = None

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
        """Auto-resume from the latest causal training checkpoint in logdir."""
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
            logging.info("Resuming causal training from %s", checkpoint_path)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        expected_config_name = self.load_checkpoint_config_name or self.config_name
        ckpt_config_name = checkpoint.get("config_name")
        mismatch = ckpt_config_name is not None and ckpt_config_name != expected_config_name

        if mismatch and not self.allow_checkpoint_config_mismatch:
            if self.is_main_process:
                logging.warning(
                    "Checkpoint config name %s does not match %s. Skipping auto-resume.",
                    ckpt_config_name,
                    expected_config_name,
                )
            return

        base = self.model.module if isinstance(self.model, DDP) else self.model
        set_peft_model_state_dict(base.model, checkpoint["lora"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        self.start_step = int(checkpoint.get("step", 0))

    def _save_checkpoint(self, step: int) -> None:
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
        if self.is_main_process:
            torch.save(state, path)
            logging.info("Saved causal LoRA checkpoint to %s", path)

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------

    def _prepare_prompt_embeds(self, prompt_batch) -> torch.Tensor:
        if prompt_batch is None:
            raise ValueError("Batch is missing 'prompt_embeds'.")
        if isinstance(prompt_batch, torch.Tensor):
            tensor = prompt_batch
        elif isinstance(prompt_batch, (list, tuple)):
            tensor = torch.stack([torch.as_tensor(x) for x in prompt_batch])
        else:
            tensor = torch.as_tensor(prompt_batch)
        return tensor.to(self.device, dtype=self.dtype)

    def _prepare_latents(self, latent_batch) -> torch.Tensor:
        if isinstance(latent_batch, torch.Tensor):
            latents = latent_batch
        elif isinstance(latent_batch, (list, tuple)):
            latents = torch.stack([torch.as_tensor(x) for x in latent_batch])
        else:
            raise TypeError(f"Unsupported latent batch type: {type(latent_batch)}")
        if latents.dim() == 4:
            latents = latents.unsqueeze(0)
        return latents.to(self.device, dtype=torch.float32)

    def _make_blockwise_index(self, batch_size: int, num_frames: int, high: int) -> torch.Tensor:
        """Sample a ``[B, F]`` random-index tensor that is constant within blocks.

        Each block of ``num_frame_per_block`` consecutive frames shares the
        same index (drawn independently per block).  This matches the
        Causal-Forcing blockwise timestep / augmentation recipe.
        """
        index = torch.randint(0, high, (batch_size, num_frames), device=self.device)
        index = index.reshape(batch_size, -1, self.num_frame_per_block)
        index[:, :, 1:] = index[:, :, 0:1]
        return index.reshape(batch_size, num_frames)

    def _sample_timesteps(self, batch_size: int, num_frames: int) -> torch.Tensor:
        """Sample per-block independent timesteps."""
        timesteps = self.scheduler.timesteps.to(self.device)
        index = self._make_blockwise_index(batch_size, num_frames, len(timesteps))
        return timesteps[index]  # [B, F]

    def _make_teacher_context(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        bsz: int,
        num_frames: int,
    ):
        """Build teacher-forcing clean context with optional noise augmentation.

        Returns ``(clean_latent_aug, aug_timestep)`` — either augmented
        clean latents and their timesteps, or ``(latents, None)`` when
        augmentation is disabled.
        """
        if self.teacher_forcing and self.noise_augmentation_max_timestep > 0:
            aug_index = self._make_blockwise_index(
                bsz, num_frames, self.noise_augmentation_max_timestep,
            )
            aug_timestep = self.scheduler.timesteps[aug_index]
            clean_latent_aug = self.scheduler.add_noise(
                latents.flatten(0, 1),
                noise.flatten(0, 1),
                aug_timestep.flatten(0, 1),
            ).view_as(latents)
            return clean_latent_aug, aug_timestep

        return (latents if self.teacher_forcing else None), None

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        if self.start_step >= self.max_steps:
            if self.is_main_process:
                logging.info("start_step >= max_iters, nothing to train.")
            return

        if self.streaming_training:
            self._train_streaming()
        else:
            self._train_window()

    # ------------------------------------------------------------------
    # Fixed-window training (original behaviour)
    # ------------------------------------------------------------------

    def _train_window(self) -> None:
        forward_model = self.model
        base_module = self.model.module if isinstance(self.model, DDP) else self.model
        base_module.train()

        total_batch_size = (
            getattr(self.config, "batch_size", 1) * self.gradient_accumulation * self.world_size
        )
        if self.is_main_process:
            logging.info(
                "Starting causal LoRA diffusion training (window mode): "
                "global_batch=%d (micro=%d x accum=%d x world=%d)",
                total_batch_size,
                getattr(self.config, "batch_size", 1),
                self.gradient_accumulation,
                self.world_size,
            )

        for step in range(self.start_step, self.max_steps):
            if self.is_distributed:
                sampler = self.dataloader.sampler
                if isinstance(sampler, DistributedSampler):
                    sampler.set_epoch(step)

            self.optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0

            for _ in range(self.gradient_accumulation):
                batch = next(self.data_iter)

                prompt_embeds = self._prepare_prompt_embeds(batch.get("prompt_embeds"))
                latents = self._prepare_latents(batch.get("real_latents"))

                bsz, num_frames = latents.shape[:2]
                image_latent = latents[:, 0:1]

                action_modulation = None
                if self.use_action_conditioning and self.action_projection is not None:
                    z_actions = batch.get("z_actions")
                    if z_actions is None:
                        raise RuntimeError("z_actions missing from batch")
                    if not torch.is_tensor(z_actions):
                        raise TypeError("z_actions must be a torch.Tensor")
                    z_actions = z_actions.to(self.device, dtype=self.dtype)
                    action_modulation = self.action_projection(z_actions, num_frames=num_frames)

                timesteps = self._sample_timesteps(bsz, num_frames)

                noise = torch.randn_like(latents)
                noisy_latents = self.scheduler.add_noise(
                    latents.flatten(0, 1),
                    noise.flatten(0, 1),
                    timesteps.flatten(0, 1),
                ).view_as(latents)

                training_target = self.scheduler.training_target(
                    latents.flatten(0, 1),
                    noise.flatten(0, 1),
                    timesteps.flatten(0, 1),
                ).view_as(latents)

                clean_latent_aug, aug_timestep = self._make_teacher_context(
                    latents, noise, bsz, num_frames,
                )

                conditional = {"prompt_embeds": prompt_embeds}
                if action_modulation is not None:
                    conditional["_action_modulation"] = action_modulation

                with autocast(dtype=self.autocast_dtype, enabled=self.use_mixed_precision):
                    flow_pred, pred_x0 = forward_model(
                        noisy_latents,
                        conditional,
                        timesteps,
                        clean_x=clean_latent_aug,
                        aug_t=aug_timestep,
                    )

                    loss = F.mse_loss(
                        flow_pred.float(),
                        training_target.float(),
                        reduction="none",
                    )
                    loss = loss.mean(dim=(2, 3, 4))
                    weights = self.scheduler.training_weight(
                        timesteps.flatten(0, 1)
                    ).view(bsz, num_frames)
                    loss = (loss * weights).mean()

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {step + 1}")

                accumulated_loss += loss.detach().item()
                scaled_loss = loss / self.gradient_accumulation

                if self.scaler.is_enabled():
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

            self._optim_step(base_module)
            self._maybe_log(step, accumulated_loss)
            self._maybe_checkpoint(step)

        barrier()
        self._save_checkpoint(self.max_steps)
        barrier()

    # ------------------------------------------------------------------
    # Streaming training (lockstep ride batcher)
    # ------------------------------------------------------------------

    def _train_streaming(self) -> None:
        """Train using the lockstep ride batcher.

        Iterates non-overlapping ``window_size``-frame windows in lockstep
        across ``batch_size`` rides.  Each window is trained with teacher
        forcing (same doubled-sequence path as fixed-window training).
        When the group of rides is exhausted, a new group is loaded.
        """
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
        )

        total_batch_size = micro_batch * self.gradient_accumulation * self.world_size
        if self.is_main_process:
            logging.info(
                "Starting causal LoRA diffusion training (lockstep streaming): "
                "global_batch=%d (micro=%d × accum=%d × world=%d), "
                "window=%d, block=%d",
                total_batch_size,
                micro_batch,
                self.gradient_accumulation,
                self.world_size,
                self.streaming_chunk_size,
                self.num_frame_per_block,
            )

        def _fill_group() -> None:
            """Load batch_size rides from the data iterator into the batcher."""
            rides: list = []
            for _ in range(micro_batch):
                raw = next(self.data_iter)
                rides.append({
                    "zarr_path": raw["zarr_path"][0],
                    "prompt_embeds": raw["prompt_embeds"][0],
                    "z_actions": raw["z_actions"][0],
                    "n_latent_frames": int(raw["n_latent_frames"][0].item()),
                })
            batcher.load_group(rides)
            if self.is_main_process:
                logging.info(
                    "Loaded ride group: %s", batcher.summary(),
                )

        for step in range(self.start_step, self.max_steps):
            if self.is_distributed:
                sampler = self.dataloader.sampler
                if isinstance(sampler, DistributedSampler):
                    sampler.set_epoch(step)

            self.optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            windows_this_step = 0

            for _ in range(self.gradient_accumulation):
                if batcher.needs_new_group():
                    _fill_group()

                num_frames = self.streaming_chunk_size
                bsz = micro_batch

                latents = batcher.load_latent_batch(self.device)
                z_actions = batcher.load_z_actions_batch(self.device, dtype=self.dtype)
                prompt_embeds = batcher.load_prompt_embeds_batch(self.device, dtype=self.dtype)

                action_modulation = None
                if self.use_action_conditioning and self.action_projection is not None:
                    action_modulation = self.action_projection(
                        z_actions, num_frames=num_frames,
                    )

                timesteps = self._sample_timesteps(bsz, num_frames)

                noise = torch.randn_like(latents)
                noisy_latents = self.scheduler.add_noise(
                    latents.flatten(0, 1),
                    noise.flatten(0, 1),
                    timesteps.flatten(0, 1),
                ).view_as(latents)

                training_target = self.scheduler.training_target(
                    latents.flatten(0, 1),
                    noise.flatten(0, 1),
                    timesteps.flatten(0, 1),
                ).view_as(latents)

                clean_latent_aug, aug_timestep = self._make_teacher_context(
                    latents, noise, bsz, num_frames,
                )

                conditional = {"prompt_embeds": prompt_embeds}
                if action_modulation is not None:
                    conditional["_action_modulation"] = action_modulation

                with autocast(dtype=self.autocast_dtype, enabled=self.use_mixed_precision):
                    flow_pred, pred_x0 = forward_model(
                        noisy_latents,
                        conditional,
                        timesteps,
                        clean_x=clean_latent_aug,
                        aug_t=aug_timestep,
                    )

                    per_frame_loss = F.mse_loss(
                        flow_pred.float(),
                        training_target.float(),
                        reduction="none",
                    ).mean(dim=(2, 3, 4))  # [B, F]

                    weights = self.scheduler.training_weight(
                        timesteps.flatten(0, 1)
                    ).view(bsz, num_frames)

                    loss = (per_frame_loss * weights).mean()

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {step + 1}")

                accumulated_loss += loss.detach().item()
                scaled_loss = loss / self.gradient_accumulation

                if self.scaler.is_enabled():
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                batcher.advance()
                windows_this_step += 1

            self._optim_step(base_module)
            self._maybe_log(
                step, accumulated_loss,
                extra={
                    "train/windows": windows_this_step,
                    "train/window_idx": batcher.current_window_idx,
                },
            )
            self._maybe_checkpoint(step)

        barrier()
        self._save_checkpoint(self.max_steps)
        barrier()

    # ------------------------------------------------------------------
    # Shared step helpers
    # ------------------------------------------------------------------

    def _optim_step(self, base_module: torch.nn.Module) -> None:
        if self.grad_clip is not None and self.grad_clip > 0:
            if self.scaler.is_enabled():
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(base_module.parameters(), self.grad_clip)

        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

    def _maybe_log(
        self, step: int, accumulated_loss: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if (step + 1) % self.log_interval != 0:
            return
        avg_loss = accumulated_loss / self.gradient_accumulation
        loss_tensor = torch.tensor(avg_loss, device=self.device)
        if self.is_distributed:
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)

        if self.is_main_process:
            logging.info("step %d | loss %.6f", step + 1, loss_tensor.item())
            if not self.disable_wandb and self.wandb_run is not None:
                try:
                    current_lr = self.optimizer.param_groups[0]["lr"]
                except (IndexError, KeyError):
                    current_lr = None
                payload = {
                    "train/step": step + 1,
                    "train/loss": loss_tensor.item(),
                    "train/learning_rate": current_lr,
                }
                if extra:
                    payload.update(extra)
                wandb.log(payload, step=step + 1)

    def _maybe_checkpoint(self, step: int) -> None:
        if self.ckpt_interval > 0 and (step + 1) % self.ckpt_interval == 0:
            barrier()
            self._save_checkpoint(step + 1)
            barrier()


def main():
    parser = argparse.ArgumentParser(
        description="LoRA finetuning for the causal Wan diffusion model."
    )
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
