import argparse
import json
import logging
import os
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import wandb
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from omegaconf import OmegaConf
import peft
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict

from utils.dataset import VideoLatentCaptionDataset, cycle
from utils.distributed import barrier, launch_distributed_job
from utils.misc import set_seed
from utils.memory import log_gpu_memory
from utils.debug_option import LOG_GPU_MEMORY
from utils.wan_wrapper import WanDiffusionWrapper
from utils.loss import get_denoising_loss

from model.action_modulation import ActionModulationProjection
from model.action_model_patch import apply_action_patches

class PatchDiscriminator3D(nn.Module):
    def __init__(self, in_channels: int = 16, base_channels: int = 64, num_layers: int = 4, channel_multiplier: int = 2):
        super().__init__()
        layers = []
        ch_in = in_channels
        for layer_idx in range(num_layers):
            ch_out = base_channels * (channel_multiplier ** layer_idx)
            stride = (1, 2, 2) if layer_idx > 0 else (1, 1, 1)
            layers.append(nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            ch_in = ch_out
        self.features = nn.Sequential(*layers)
        self.head = nn.Conv3d(ch_in, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError("Expected 5D input (batch, channels, frames, height, width)")
        x = self.features(x)
        return self.head(x)


def hinge_discriminator_loss(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 - real_scores).mean() + F.relu(1.0 + fake_scores).mean()


def hinge_generator_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    return -fake_scores.mean()


def toggle_module_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(requires_grad)


def _is_distributed() -> bool:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    return world > 1


class LoRADiffusionTrainer:
    """LoRA finetuning for the bidirectional Wan 14B diffusion model."""

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

        self.device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        self.use_mixed_precision = bool(getattr(config, "mixed_precision", False))
        self.autocast_dtype = torch.bfloat16 if getattr(config, "autocast_dtype", "bf16") == "bf16" else torch.float16
        self.dtype = torch.bfloat16 if self.use_mixed_precision else torch.float32
        self.grad_clip = getattr(config, "max_grad_norm", None)
        self.gradient_accumulation = max(1, getattr(config, "gradient_accumulation_steps", 1))
        self.max_steps = int(getattr(config, "max_iters", 1000))
        self.log_interval = int(getattr(config, "log_interval", 50))
        self.ckpt_interval = int(getattr(config, "ckpt_interval", 100))
        self.num_workers = int(getattr(config, "num_workers", 8))
        self.pin_memory = bool(getattr(config, "pin_memory", True))
        self.prefetch_factor = getattr(config, "prefetch_factor", None)
        self.sequential_sampling = bool(getattr(config, "sequential_sampling", False))
        self.allow_checkpoint_config_mismatch = bool(getattr(config, "allow_checkpoint_config_mismatch", False))
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
        self.enable_gan_loss = bool(getattr(config, "enable_gan_loss", False))
        self.gan_loss_weight = float(getattr(config, "gan_loss_weight", 0.1))
        self.gan_loss_type = getattr(config, "gan_loss_type", "hinge")
        self.gan_discriminator_steps = max(1, int(getattr(config, "gan_discriminator_steps", 1)))
        self.use_adaptive_gan_weight = bool(getattr(config, "gan_use_adaptive_weight", True))
        self.gan_adaptive_eps = float(getattr(config, "gan_adaptive_eps", 1e-4))
        adaptive_max = getattr(config, "gan_adaptive_max_weight", 1e4)
        self.gan_adaptive_max_weight = float(adaptive_max) if adaptive_max is not None else None
        self._gan_discriminator_cfg = getattr(config, "gan_discriminator_kwargs", {})
        if self.enable_gan_loss and self.gan_loss_type not in {"hinge"}:
            raise ValueError(f"Unsupported gan_loss_type: {self.gan_loss_type}")
        if self.enable_gan_loss and self.gan_loss_weight <= 0:
            raise ValueError("gan_loss_weight must be positive when enable_gan_loss is true")
        set_seed(int(getattr(config, "seed", 0)) + self.global_rank)

        if self.is_main_process:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)s | %(message)s"
            )
            logging.info('Initialising LoRADiffusionTrainer...')
            logging.info('Config (keys only): %s', list(config.keys()))

        self.logdir = self._resolve_logdir(getattr(config, "logdir", ""))
        self.config_name = getattr(config, "config_name", None)
        self.load_checkpoint_config_name = getattr(config, "load_checkpoint_config_name", None)
        if self.config_name is None:
            raise ValueError("config_name is required")
        if self.is_main_process:
            logging.info('Resolved logdir: %s', self.logdir)
        if self.is_main_process:
            self.logdir.mkdir(parents=True, exist_ok=True)
        if self.is_distributed:
            barrier()
        self.metrics_path = self.logdir / f"{self.config_name}_domain_shift_metrics.jsonl"
        if self.is_main_process:
            logging.info('Metrics path: %s', self.metrics_path)
        if self.is_main_process and not self.metrics_path.exists():
            self.metrics_path.touch()

        adapter_cfg = getattr(config, "adapter", {})
        try:
            adapter_rank = adapter_cfg.get("rank")
            adapter_alpha = adapter_cfg.get("alpha")
        except AttributeError:
            adapter_rank = None
            adapter_alpha = None
        self.adapter_rank = adapter_rank
        self.adapter_alpha = adapter_alpha

        self.train_dataset_size: Optional[int] = None
        self.eval_dataset_size: Optional[int] = None
        self.wandb_run = None

        hf_cache = self.scratch_root / "frodobots" / "hf_cache"
        if self.is_main_process:
            logging.info('HF cache root: %s', hf_cache)
        for env_key in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
            os.environ.setdefault(env_key, str(hf_cache))
        if self.is_main_process:
            hf_cache.mkdir(parents=True, exist_ok=True)
        if self.is_distributed:
            barrier()

        self.model = None
        self.scheduler = None
        self.optimizer = None
        self.scaler = GradScaler(enabled=self.use_mixed_precision and self.autocast_dtype == torch.float16)
        self.discriminator = None
        self.disc_optimizer = None
        self.disc_scaler = (
            GradScaler(enabled=self.use_mixed_precision and self.autocast_dtype == torch.float16)
            if self.enable_gan_loss else None
        )
        self.gan_expected_channels = None
        self.start_step = 0

        self.action_projection = None
        self.use_action_conditioning = None

        self.denoising_loss_type = getattr(config, "denoising_loss_type", "noise")
        try:
            loss_cls = get_denoising_loss(self.denoising_loss_type)
        except KeyError as err:
            raise ValueError(f"Unsupported denoising loss type: {self.denoising_loss_type}") from err
        self.denoising_loss_func = loss_cls()
        if self.is_main_process:
            logging.info('Using denoising loss: %s', self.denoising_loss_type)
        if self.enable_gan_loss and self.is_main_process:
            adaptive_note = (
                f"adaptive=True eps={self.gan_adaptive_eps:.1e} max="
                f"{self.gan_adaptive_max_weight if self.gan_adaptive_max_weight is not None else 'None'}"
                if self.use_adaptive_gan_weight
                else "adaptive=False"
            )
            logging.info('GAN loss enabled (type=%s, base_weight=%.4f, disc_steps=%d, %s)',
                         self.gan_loss_type, self.gan_loss_weight, self.gan_discriminator_steps, adaptive_note)

        if self.is_main_process:
            logging.info('Validating dataset roots...')
        self._validate_dataset_roots()
        if self.is_main_process:
            logging.info('Building training dataloader...')
        self._build_dataloader()
        if self.is_main_process:
            logging.info('Training dataloader ready (world_size=%s)', self.world_size)
        self.eval_loader = self._build_eval_dataloader()
        self.eval_interval = max(1, int(getattr(self.config, "eval_interval", self.ckpt_interval or self.log_interval or 1)))
        self.eval_batches = max(1, int(getattr(self.config, "eval_batches", 32)))
        self._init_wandb()

        if self.is_main_process:
            logging.info('Building model...')
        self._build_model()
        if self.is_main_process:
            logging.info('Model build complete')
        if self.is_main_process:
            logging.info('Initialising optimizer...')
        self._build_optimizer()
        if self.is_main_process:
            logging.info('Optimizer initialised')
        if self.is_main_process:
            logging.info('Checking for resume checkpoints...')
        self._maybe_resume()
        if self.is_main_process:
            logging.info('Resume logic completed (start_step=%s)', self.start_step)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _build_dataloader(self) -> None:
        blacklist_path = getattr(self.config, "data_blacklist_path", None)

        dataset = VideoLatentCaptionDataset(
            latent_root=self.config.real_latent_root,
            caption_root=self.config.caption_root,
            num_frames=getattr(self.config, "num_training_frames", 21),
            text_pre_encoded=bool(getattr(self.config, "text_pre_encoded", False)),
            include_dir_substrings=getattr(self.config, "include_dir_substrings", None),
            blacklist_path=blacklist_path,
        )
        self.train_dataset_size = len(dataset)
        if self.is_main_process:
            logging.info('Training dataset size: %d', len(dataset))

        if self.is_distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.global_rank,
                shuffle=not self.sequential_sampling,
                drop_last=True,
            )
        else:
            sampler = None

        dataloader_kwargs: Dict[str, Any] = dict(
            batch_size=int(getattr(self.config, "batch_size", 1)),
            sampler=sampler,
            shuffle=(sampler is None) and not self.sequential_sampling,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )
        if self.prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = self.prefetch_factor

        self.dataloader = DataLoader(dataset, **dataloader_kwargs)
        self.data_iter = cycle(self.dataloader)

    def _build_model(self) -> None:
        model_name = getattr(self.config, "model_name", None)
        if model_name is None:
            model_name = getattr(self.config, "real_name", "Wan2.1-T2V-14B")
        wrapper_kwargs = dict(getattr(self.config, "model_kwargs", {}))
        wrapper = WanDiffusionWrapper(
            model_name=model_name,
            is_causal=False,
            **wrapper_kwargs,
        )

        if getattr(self.config, "gradient_checkpointing", False):
            wrapper.enable_gradient_checkpointing()

        model_variant = getattr(self.config, "model_variant", None)

        # Apply LoRA FIRST
        lora_cfg = getattr(self.config, "adapter", None)
        if lora_cfg is None:
            raise ValueError("LoRA adapter configuration (config.adapter) is required for diffusion finetuning.")
        self.lora_config = lora_cfg
        wrapper.model = self._apply_lora(wrapper.model, lora_cfg)

        if model_variant == "action-injection":
            if self.is_main_process:
                logging.info("Applying action injection to model")

            apply_action_patches(wrapper.model)

            if hasattr(wrapper.model, "dim"):
                model_dim = wrapper.model.dim
            else:
                model_dim = 2048

            resolved_action_dim = getattr(self.config, "raw_action_dim", None)
            if resolved_action_dim is None:
                resolved_action_dim = getattr(self.config, "action_dim", 2)
            enable_adaln_zero = getattr(self.config, "enable_adaln_zero", True)

            self.action_projection = ActionModulationProjection(
                action_dim=resolved_action_dim,
                hidden_dim=model_dim,
                num_frames=1,
                zero_init=enable_adaln_zero,
            )

            # Move to device BEFORE DDP wrapping
            self.action_projection.to(self.device)
            self.action_projection.train()
            
            self.use_action_conditioning = True

            if self.is_main_process:
                logging.info(
                    "Action-injection enabled: action_dim=%s, enable_adaln_zero=%s",
                    resolved_action_dim,
                    enable_adaln_zero,
                )

        wrapper.to(self.device)
        wrapper.train()

        self.model = wrapper
        self.scheduler = self.model.get_scheduler()
        self.scheduler.set_timesteps(
            getattr(self.config, "num_train_timestep", 1000),
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

            if self.use_action_conditioning:
                if self.action_projection is not None:
                    self.action_projection = DDP(
                        self.action_projection,
                        device_ids=[torch.cuda.current_device()],
                        output_device=torch.cuda.current_device(),
                        broadcast_buffers=False,
                        find_unused_parameters=False,
                    )

        if self.enable_gan_loss:
            self._build_gan_components()

        if self.is_main_process and LOG_GPU_MEMORY:
            log_gpu_memory("After model build", device=self.device, rank=self.global_rank)

    def _build_gan_components(self) -> None:
        if not self.enable_gan_loss:
            return
        if self.discriminator is not None:
            return

        cfg_obj = self._gan_discriminator_cfg
        if cfg_obj:
            if isinstance(cfg_obj, dict):
                cfg_dict = dict(cfg_obj)
            else:
                cfg_dict = OmegaConf.to_container(cfg_obj, resolve=True)
        else:
            cfg_dict = {}

        in_channels = int(cfg_dict.get("in_channels", getattr(self.config, "gan_discriminator_in_channels", 16)))
        base_channels = int(cfg_dict.get("base_channels", getattr(self.config, "gan_discriminator_base_channels", 64)))
        num_layers = int(cfg_dict.get("num_layers", getattr(self.config, "gan_discriminator_layers", 4)))
        channel_multiplier = int(cfg_dict.get("channel_multiplier", getattr(self.config, "gan_discriminator_channel_multiplier", 2)))

        self.gan_expected_channels = in_channels
        discriminator = PatchDiscriminator3D(
            in_channels=in_channels,
            base_channels=base_channels,
            num_layers=num_layers,
            channel_multiplier=channel_multiplier,
        )
        discriminator.to(self.device)
        discriminator.train()

        if self.is_distributed:
            discriminator = DDP(
                discriminator,
                device_ids=[torch.cuda.current_device()],
                output_device=torch.cuda.current_device(),
                broadcast_buffers=False,
                find_unused_parameters=False,
            )

        self.discriminator = discriminator

        if self.is_main_process:
            logging.info(
                "Initialised GAN discriminator (in_channels=%d, base_channels=%d, num_layers=%d, channel_multiplier=%d)",
                in_channels,
                base_channels,
                num_layers,
                channel_multiplier,
            )

    def _collect_target_modules(self, transformer: torch.nn.Module) -> list[str]:
        """
        Locate Linear layers that should receive LoRA adapters.
        """
        target_modules: set[str] = set()

        for module_name, module in transformer.named_modules():
            cls_name = module.__class__.__name__

            # Apply LoRA to all Linear layers inside attention blocks
            if cls_name in {"WanAttentionBlock", "CausalWanAttentionBlock"}:
                for full_name, sub_module in module.named_modules(prefix=module_name):
                    if isinstance(sub_module, torch.nn.Linear):
                        target_modules.add(full_name)

        if not target_modules:
            raise RuntimeError("Failed to locate target Linear modules for LoRA.")
        
        if self.is_main_process:
            logging.info(f"Applying LoRA to {len(target_modules)} Linear layers in attention blocks")

        return sorted(target_modules)

    def _resolve_logdir(self, logdir_value: str) -> Path:
        base = self.scratch_root.resolve()
        default_root = base / "frodobots" / "lora_runs"

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
                    resolved = candidate.resolve()
                    resolved_str = str(resolved)
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

        if resolved != base and base not in resolved.parents:
            raise ValueError("Log directory must reside under /scratch/u5as/as1748.u5as")
        return resolved

    def _validate_dataset_roots(self) -> None:
        def _check_path(path_like: str, label: str) -> None:
            path = Path(path_like).expanduser()
            lowered_parts = {part.lower() for part in path.parts}
            if "test" in lowered_parts:
                raise ValueError(f"{label} appears to reference the test split: {path}")

        _check_path(self.config.real_latent_root, "real_latent_root")
        _check_path(self.config.caption_root, "caption_root")

    def _build_eval_dataloader(self) -> Optional[DataLoader]:
        val_latent_root = getattr(self.config, "val_real_latent_root", None)
        val_caption_root = getattr(self.config, "val_caption_root", None)
        if not val_latent_root or not val_caption_root:
            return None

        dataset = VideoLatentCaptionDataset(
            latent_root=val_latent_root,
            caption_root=val_caption_root,
            num_frames=getattr(self.config, "num_training_frames", 21),
            text_pre_encoded=bool(getattr(self.config, "text_pre_encoded", False)),
            include_dir_substrings=getattr(self.config, "val_include_dir_substrings", None),
            blacklist_path=getattr(self.config, "data_blacklist_path", None),
        )
        self.eval_dataset_size = len(dataset)
        if self.is_main_process:
            logging.info('Validation dataset size: %d', len(dataset))

        if self.is_distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.global_rank,
                shuffle=False,
                drop_last=False,
            )
        else:
            sampler = None

        dataloader_kwargs: Dict[str, Any] = dict(
            batch_size=int(getattr(self.config, "val_batch_size", getattr(self.config, "batch_size", 1))),
            sampler=sampler,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
        if self.prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = self.prefetch_factor

        return DataLoader(dataset, **dataloader_kwargs)

    def _init_wandb(self) -> None:
        if not self.is_main_process or self.disable_wandb:
            return

        if not self.wandb_project:
            logging.warning("wandb_project not set; disabling Weights & Biases logging.")
            self.disable_wandb = True
            return

        if self._wandb_login_key:
            try:
                wandb.login(key=self._wandb_login_key)
            except Exception as exc:
                logging.warning("Failed to login to Weights & Biases: %s", exc)

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
        except Exception as exc:
            logging.warning("Failed to initialise Weights & Biases run: %s", exc)
            self.disable_wandb = True
            self.wandb_run = None
            return

        train_filters = getattr(self.config, "include_dir_substrings", None)
        val_filters = getattr(self.config, "val_include_dir_substrings", None)
        try:
            train_filters = list(train_filters) if train_filters is not None else None
        except TypeError:
            train_filters = None
        try:
            val_filters = list(val_filters) if val_filters is not None else None
        except TypeError:
            val_filters = None

        global_batch_size = getattr(self.config, "batch_size", 1) * self.gradient_accumulation * self.world_size
        update_payload = {
            "config_name": self.config_name,
            "load_checkpoint_config_name": self.load_checkpoint_config_name,
            "lr": getattr(self.config, "lr", None),
            "weight_decay": getattr(self.config, "weight_decay", None),
            "batch_size": getattr(self.config, "batch_size", None),
            "gradient_accumulation_steps": self.gradient_accumulation,
            "global_batch_size": global_batch_size,
            "world_size": self.world_size,
            "max_steps": self.max_steps,
            "enable_gan_loss": self.enable_gan_loss,
            "gan_loss_weight": self.gan_loss_weight if self.enable_gan_loss else 0.0,
            "gan_use_adaptive_weight": self.use_adaptive_gan_weight and self.enable_gan_loss,
            "gan_discriminator_steps": self.gan_discriminator_steps if self.enable_gan_loss else 0,
            "train_dataset_size": self.train_dataset_size,
            "eval_dataset_size": self.eval_dataset_size,
            "num_workers": self.num_workers,
            "lora_rank": self.adapter_rank,
            "lora_alpha": self.adapter_alpha,
            "include_dir_substrings": train_filters,
            "val_include_dir_substrings": val_filters,
        }
        try:
            wandb.config.update({k: v for k, v in update_payload.items() if v is not None}, allow_val_change=True)
            wandb.define_metric("train/step")
            wandb.define_metric("eval/step")
            wandb.define_metric("train/*", step_metric="train/step")
            wandb.define_metric("eval/*", step_metric="eval/step")
        except Exception as exc:
            logging.warning("Failed to configure Weights & Biases run: %s", exc)

    @torch.no_grad()
    def _evaluate_domain_shift(self, step: int, final: bool = False) -> Optional[float]:
        if self.eval_loader is None:
            return None

        module = self.model.module if isinstance(self.model, DDP) else self.model
        was_training = module.training
        module.eval()

        sampler = getattr(self.eval_loader, "sampler", None)
        if self.is_distributed and isinstance(sampler, DistributedSampler):
            sampler.set_epoch(step)

        total_loss = torch.tensor(0.0, device=self.device)
        total_kl = torch.tensor(0.0, device=self.device)
        total_distribution = torch.tensor(0.0, device=self.device)
        total_batches = torch.tensor(0, device=self.device, dtype=torch.long)

        iterator = iter(self.eval_loader)
        for idx in range(self.eval_batches):
            try:
                batch = next(iterator)
            except StopIteration:
                break

            prompt_embeds = self._prepare_prompt_embeds(batch.get("prompt_embeds"))
            latents = self._prepare_latents(batch.get("real_latents"))

            bsz, num_frames = latents.shape[:2]
            timesteps = self._sample_timesteps(bsz, num_frames)
            noise = torch.randn_like(latents)

            noisy_latents = self.scheduler.add_noise(
                latents.flatten(0, 1),
                noise.flatten(0, 1),
                timesteps.flatten(0, 1),
            ).view_as(latents)

            conditional = {"prompt_embeds": prompt_embeds}

            with autocast(dtype=self.autocast_dtype, enabled=self.use_mixed_precision):
                flow_pred, pred_x0 = module(
                    noisy_latents,
                    conditional,
                    timesteps,
                )
                batch_loss = self._compute_denoising_loss(
                    latents=latents,
                    noise=noise,
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    flow_pred=flow_pred,
                    pred_x0=pred_x0,
                )
                kl_loss, distribution_loss = self._compute_distribution_metrics(
                    real_latents=latents,
                    predicted_latents=pred_x0,
                )

            total_loss = total_loss + batch_loss.detach()
            total_kl = total_kl + kl_loss.detach()
            total_distribution = total_distribution + distribution_loss.detach()
            total_batches = total_batches + 1

        if self.is_distributed and dist.is_initialized():
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_kl, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_distribution, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_batches, op=dist.ReduceOp.SUM)

        total_batches_value = int(total_batches.item())
        avg_loss = None
        if total_batches_value > 0:
            avg_loss = (total_loss / float(total_batches_value)).item()
            avg_kl = (total_kl / float(total_batches_value)).item()
            avg_distribution = (total_distribution / float(total_batches_value)).item()
        else:
            avg_kl = None
            avg_distribution = None

        if was_training:
            module.train()

        if avg_loss is not None and self.is_main_process:
            loss_name = "flow_loss" if self.denoising_loss_type == "flow" else "epsilon_mse"
            record = {
                "step": step,
                loss_name: avg_loss,
                "latent_kl": avg_kl,
                "latent_distribution": avg_distribution,
                "final": final,
            }
            with self.metrics_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
            logging.info(
                "Domain shift eval (step %s): %s %.6f | KL %.6f | Dist %.6f",
                step,
                loss_name,
                avg_loss,
                avg_kl if avg_kl is not None else float('nan'),
                avg_distribution if avg_distribution is not None else float('nan'),
            )
            if not self.disable_wandb and self.wandb_run is not None:
                eval_log = {
                    "eval/step": step,
                    f"eval/{loss_name}": avg_loss,
                    "eval/kl_loss": avg_kl,
                    "eval/distribution_loss": avg_distribution,
                    "eval/is_final": int(final),
                }
                wandb.log(eval_log, step=step)

        return avg_loss

    def _apply_lora(self, transformer: torch.nn.Module, lora_cfg) -> torch.nn.Module:
        target_modules = self._collect_target_modules(transformer)
        rank = int(lora_cfg.get("rank", 16))
        alpha = int(lora_cfg.get("alpha", rank))
        dropout = float(lora_cfg.get("dropout", 0.0))
        init_type = lora_cfg.get("init_lora_weights", "gaussian")

        if self.is_main_process:
            logging.info("Applying LoRA to %d Linear modules", len(target_modules))

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

    def _build_optimizer(self) -> None:
        params = [p for p in self.model.parameters() if p.requires_grad]

        if self.use_action_conditioning:
            if self.action_projection is not None:
                proj_params = [p for p in self.action_projection.parameters() if p.requires_grad]
                params.extend(proj_params)
                if self.is_main_process:
                    logging.info(f"Added {len(proj_params)} action projection parameters to optimizer")

        if not params:
            raise RuntimeError("No trainable parameters found after applying LoRA.")

        lr = float(getattr(self.config, "lr", 1e-4))
        beta1 = float(getattr(self.config, "beta1", 0.9))
        beta2 = float(getattr(self.config, "beta2", 0.999))
        betas = (beta1, beta2)
        weight_decay = float(getattr(self.config, "weight_decay", 0.01))
        self.optimizer = torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)

        if self.enable_gan_loss and self.discriminator is not None:
            disc_module = self.discriminator.module if isinstance(self.discriminator, DDP) else self.discriminator
            disc_params = [p for p in disc_module.parameters() if p.requires_grad]
            if disc_params:
                gan_lr = float(getattr(self.config, "gan_lr", lr))
                gan_beta1 = float(getattr(self.config, "gan_beta1", beta1))
                gan_beta2 = float(getattr(self.config, "gan_beta2", beta2))
                gan_weight_decay = float(getattr(self.config, "gan_weight_decay", weight_decay))
                self.disc_optimizer = torch.optim.AdamW(
                    disc_params,
                    lr=gan_lr,
                    betas=(gan_beta1, gan_beta2),
                    weight_decay=gan_weight_decay,
                )
                if self.is_main_process:
                    logging.info(
                        "Initialised discriminator optimizer (lr=%.5f, betas=(%.3f, %.3f), weight_decay=%.5f)",
                        gan_lr,
                        gan_beta1,
                        gan_beta2,
                        gan_weight_decay,
                    )
            else:
                self.disc_optimizer = None

    def _checkpoint_path(self, step: int) -> Optional[Path]:
        if self.logdir is None:
            return None
        self.logdir.mkdir(parents=True, exist_ok=True)
        return self.logdir / f"diffusion_lora_step{step:07d}.pt"

    def _latest_checkpoint(self) -> Optional[Path]:
        if self.logdir is None:
            return None
        checkpoints = sorted(self.logdir.glob("diffusion_lora_step*.pt"))
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
            logging.info("Resuming LoRA diffusion training from %s", checkpoint_path)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        expected_config_name = self.load_checkpoint_config_name or self.config_name
        checkpoint_config_name = checkpoint.get("config_name", None)
        mismatch = checkpoint_config_name is not None and checkpoint_config_name != expected_config_name
        missing_name = checkpoint_config_name is None

        if (mismatch or missing_name) and not self.allow_checkpoint_config_mismatch:
            if self.is_main_process:
                if mismatch:
                    logging.warning(
                        f"Checkpoint config name {checkpoint_config_name} does not match expected name {expected_config_name}. Skipping auto-resume."
                    )
                else:
                    logging.warning(
                        f"Checkpoint config name is not found. Expected {expected_config_name}. Skipping auto-resume."
                    )
            return
        if mismatch and self.is_main_process:
            logging.warning(
                f"Checkpoint config name {checkpoint_config_name} does not match expected name {expected_config_name}. Continuing because allow_checkpoint_config_mismatch is true."
            )
        if missing_name and self.is_main_process:
            logging.warning(
                f"Checkpoint config name is not found. Expected {expected_config_name}. Continuing because allow_checkpoint_config_mismatch is true."
            )
        base_module = self.model.module if isinstance(self.model, DDP) else self.model
        set_peft_model_state_dict(base_module.model, checkpoint["lora"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        if self.enable_gan_loss and self.discriminator is not None:
            disc_state = checkpoint.get("discriminator")
            if disc_state is not None:
                disc_module = self.discriminator.module if isinstance(self.discriminator, DDP) else self.discriminator
                disc_module.load_state_dict(disc_state)

        if self.enable_gan_loss and self.disc_optimizer is not None:
            disc_opt_state = checkpoint.get("discriminator_optimizer")
            if disc_opt_state is not None:
                self.disc_optimizer.load_state_dict(disc_opt_state)
                for state in self.disc_optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)

        if self.enable_gan_loss and self.disc_scaler is not None:
            disc_scaler_state = checkpoint.get("discriminator_scaler")
            if disc_scaler_state is not None:
                self.disc_scaler.load_state_dict(disc_scaler_state)

        self.start_step = int(checkpoint.get("step", 0))

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _prepare_prompt_embeds(self, prompt_batch: Any) -> torch.Tensor:
        if prompt_batch is None:
            raise ValueError("Batch is missing 'prompt_embeds'. Enable text_pre_encoded in the config.")
        if isinstance(prompt_batch, torch.Tensor):
            tensor = prompt_batch
        elif isinstance(prompt_batch, (list, tuple)):
            tensor = torch.stack([torch.as_tensor(x) for x in prompt_batch])
        else:
            tensor = torch.as_tensor(prompt_batch)
        return tensor.to(self.device, dtype=self.dtype)

    def _prepare_latents(self, latent_batch: Any) -> torch.Tensor:
        if isinstance(latent_batch, torch.Tensor):
            latents = latent_batch
        elif isinstance(latent_batch, (list, tuple)):
            latents = torch.stack([torch.as_tensor(x) for x in latent_batch])
        else:
            raise TypeError(f"Unsupported latent batch type: {type(latent_batch)}")
        if latents.dim() == 4:
            latents = latents.unsqueeze(0)
        return latents.to(self.device, dtype=torch.float32)

    def _sample_timesteps(self, batch_size: int, num_frames: int) -> torch.Tensor:
        timesteps = self.scheduler.timesteps.to(self.device)
        step_idx = torch.randint(0, timesteps.shape[0], (batch_size,), device=self.device)
        sampled = timesteps[step_idx].unsqueeze(1).repeat(1, num_frames)
        return sampled

    def _compute_distribution_metrics(
        self,
        real_latents: torch.Tensor,
        predicted_latents: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        eps = 1e-6
        dims = (0, 1, 3, 4)

        real = real_latents.float()
        pred = predicted_latents.float()

        real_mean = real.mean(dim=dims)
        pred_mean = pred.mean(dim=dims)
        real_var = real.var(dim=dims, unbiased=False)
        pred_var = pred.var(dim=dims, unbiased=False)

        var_ratio = (pred_var + eps) / (real_var + eps)
        mean_diff_sq = (real_mean - pred_mean) ** 2
        kl_elements = torch.log(var_ratio) + (real_var + mean_diff_sq) / (pred_var + eps) - 1.0
        kl_div = 0.5 * kl_elements.mean()

        std_real = torch.sqrt(real_var + eps)
        std_pred = torch.sqrt(pred_var + eps)
        distribution_loss = F.mse_loss(pred_mean, real_mean) + F.mse_loss(std_pred, std_real)

        return kl_div, distribution_loss

    def _compute_denoising_loss(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        flow_pred: torch.Tensor,
        pred_x0: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the configured denoising loss for a batch."""
        alphas_cumprod = getattr(self.scheduler, 'alphas_cumprod', None)

        if self.denoising_loss_type == 'flow':
            # FlowPredLoss needs flow_pred in kwargs
            loss_kwargs = {'flow_pred': flow_pred.flatten(0, 1)}
            noise_pred = None
        else:
            # NoisePredLoss needs noise_pred
            noise_pred = self.scheduler.convert_x0_to_noise(
                pred_x0.flatten(0, 1),
                noisy_latents.flatten(0, 1),
                timesteps.flatten(0, 1),
            ).view_as(latents)
            loss_kwargs = {}

        return self.denoising_loss_func(
            x=latents.flatten(0, 1),
            x_pred=pred_x0.flatten(0, 1),
            noise=noise.flatten(0, 1),
            noise_pred=noise_pred.flatten(0, 1) if noise_pred is not None else None,
            alphas_cumprod=alphas_cumprod,
            timestep=timesteps.flatten(0, 1),
            **loss_kwargs,
        )

    def _prepare_discriminator_input(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.dim() != 5:
            raise ValueError("Expected latents with shape [B, F, C, H, W]")
        return latents.permute(0, 2, 1, 3, 4).contiguous()

    def _compute_gan_losses(
        self,
        real_latents: torch.Tensor,
        fake_latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.enable_gan_loss or self.discriminator is None:
            raise RuntimeError("GAN loss requested without initialised discriminator.")

        real_input = self._prepare_discriminator_input(real_latents)
        fake_input = self._prepare_discriminator_input(fake_latents)

        if self.gan_expected_channels is None:
            self.gan_expected_channels = real_input.shape[1]
        elif real_input.shape[1] != self.gan_expected_channels:
            raise ValueError(
                f"Discriminator expects {self.gan_expected_channels} channels but received {real_input.shape[1]}"
            )

        fake_detached = fake_input.detach()

        toggle_module_grad(self.discriminator, True)
        disc_loss = None
        for _ in range(self.gan_discriminator_steps):
            with autocast(dtype=self.autocast_dtype, enabled=self.use_mixed_precision):
                real_scores = self.discriminator(real_input)
                fake_scores = self.discriminator(fake_detached)
                if self.gan_loss_type == "hinge":
                    step_loss = hinge_discriminator_loss(real_scores, fake_scores)
                else:
                    raise ValueError(f"Unsupported gan_loss_type: {self.gan_loss_type}")
            disc_loss = step_loss if disc_loss is None else disc_loss + step_loss
        disc_loss = disc_loss / float(self.gan_discriminator_steps)

        toggle_module_grad(self.discriminator, False)
        with autocast(dtype=self.autocast_dtype, enabled=self.use_mixed_precision):
            fake_scores_for_gen = self.discriminator(fake_input)
            if self.gan_loss_type == "hinge":
                gen_loss = hinge_generator_loss(fake_scores_for_gen)
            else:
                raise ValueError(f"Unsupported gan_loss_type: {self.gan_loss_type}")
        toggle_module_grad(self.discriminator, True)

        return gen_loss, disc_loss

    def _compute_adaptive_gan_weight(
        self,
        base_loss: torch.Tensor,
        gan_loss: torch.Tensor,
        fake_latents: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        base_value = float(self.gan_loss_weight)
        device = base_loss.device
        base_tensor = torch.tensor(base_value, device=device, dtype=base_loss.dtype)

        if (
            not self.use_adaptive_gan_weight
            or fake_latents is None
            or not torch.is_floating_point(fake_latents)
            or not fake_latents.requires_grad
            or base_loss.grad_fn is None
            or gan_loss.grad_fn is None
        ):
            return base_tensor, base_value

        with torch.enable_grad():
            recon_grad = torch.autograd.grad(
                base_loss, fake_latents, retain_graph=True, create_graph=False, allow_unused=True
            )[0]
            gan_grad = torch.autograd.grad(
                gan_loss, fake_latents, retain_graph=True, create_graph=False, allow_unused=True
            )[0]

        if recon_grad is None or gan_grad is None:
            return base_tensor, base_value

        recon_norm = recon_grad.float().norm()
        gan_norm = gan_grad.float().norm()

        if (
            not torch.isfinite(recon_norm)
            or not torch.isfinite(gan_norm)
            or gan_norm.item() < self.gan_adaptive_eps
        ):
            return base_tensor, base_value

        ratio = (recon_norm / gan_norm.clamp(min=self.gan_adaptive_eps)).detach()
        if self.gan_adaptive_max_weight is not None:
            ratio = ratio.clamp(max=self.gan_adaptive_max_weight)

        adaptive_value = float(ratio.item() * self.gan_loss_weight)
        adaptive_tensor = torch.tensor(adaptive_value, device=device, dtype=base_loss.dtype)
        return adaptive_tensor, adaptive_value

    def _get_action_weight(self, step: int) -> float:
        """
        Compute action conditioning weight with warmup schedule.
        
        Returns weight in [0, 1] that scales action contribution.
        """
        warmup_steps = getattr(self.config, "action_warmup_steps", 1000)
        warmup_type = getattr(self.config, "action_warmup_type", "linear")  # linear, cosine, or none
        
        if warmup_type == "none" or warmup_steps == 0:
            return 1.0
        
        if step >= warmup_steps:
            return 1.0
        
        progress = step / warmup_steps
        
        if warmup_type == "linear":
            return progress
        elif warmup_type == "cosine":
            # Cosine schedule: smooth acceleration
            return 0.5 * (1 - math.cos(math.pi * progress))
        elif warmup_type == "exponential":
            # Exponential: slow start, fast finish
            return progress ** 2
        
        return progress

    def _save_checkpoint(self, step: int) -> None:
        path = self._checkpoint_path(step)
        if path is None:
            return
        base_module = self.model.module if isinstance(self.model, DDP) else self.model
        state = {
            "step": step,
            "lora": get_peft_model_state_dict(base_module.model),
            "optimizer": self.optimizer.state_dict(),
            "config_name": self.config_name,
        }
        if self.enable_gan_loss and self.discriminator is not None:
            disc_module = self.discriminator.module if isinstance(self.discriminator, DDP) else self.discriminator
            state["discriminator"] = disc_module.state_dict()
        if self.enable_gan_loss and self.disc_optimizer is not None:
            state["discriminator_optimizer"] = self.disc_optimizer.state_dict()
        if self.enable_gan_loss and self.disc_scaler is not None:
            state["discriminator_scaler"] = self.disc_scaler.state_dict()
        if self.is_main_process:
            torch.save(state, path)
            logging.info("Saved LoRA checkpoint to %s", path)

    def train(self) -> None:
        if self.start_step >= self.max_steps:
            if self.is_main_process:
                logging.info("Start step >= max_iters, nothing to train.")
            return

        forward_model = self.model
        base_module = self.model.module if isinstance(self.model, DDP) else self.model
        base_module.train()

        total_batch_size = self.config.batch_size * self.gradient_accumulation * self.world_size
        if self.is_main_process:
            logging.info("Starting LoRA diffusion finetuning")
            logging.info("global_batch_size=%d (micro=%d x accum=%d x world=%d)",
                         total_batch_size, self.config.batch_size, self.gradient_accumulation, self.world_size)

        for step in range(self.start_step, self.max_steps):
            if self.is_distributed:
                sampler = self.dataloader.sampler
                if isinstance(sampler, DistributedSampler):
                    sampler.set_epoch(step)

            self.optimizer.zero_grad(set_to_none=True)
            if self.enable_gan_loss and self.disc_optimizer is not None:
                self.disc_optimizer.zero_grad(set_to_none=True)
            accumulated_base_loss = 0.0
            accumulated_loss = 0.0
            accumulated_gan_loss = 0.0
            accumulated_disc_loss = 0.0
            accumulated_gan_weight = 0.0
            accumulated_kl_loss = 0.0
            accumulated_distribution_loss = 0.0

            for _ in range(self.gradient_accumulation):
                batch = next(self.data_iter)
                prompt_embeds = self._prepare_prompt_embeds(batch.get("prompt_embeds"))
                latents = self._prepare_latents(batch.get("real_latents"))

                action_modulation = None
                if self.use_action_conditioning and self.action_projection is not None:
                    raw_actions = batch.get("raw_actions", None)
                    if raw_actions is not None:
                        if not torch.is_tensor(raw_actions):
                            raise TypeError("raw_actions must be a torch.Tensor when action conditioning is enabled")
                        action_features = raw_actions.to(self.device)
                        proj_module = self.action_projection.module if isinstance(self.action_projection, DDP) else self.action_projection
                        proj_dtype = next(proj_module.parameters()).dtype
                        action_features = action_features.to(dtype=proj_dtype)
                        num_frames = latents.shape[1]

                        if action_features.dim() == 2:
                            prepared_features = action_features
                        elif action_features.dim() == 3:
                            if action_features.shape[1] < num_frames:
                                raise ValueError(
                                    f"raw_actions has {action_features.shape[1]} frames, but {num_frames} are required"
                                )
                            prepared_features = action_features[:, :num_frames]
                        else:
                            raise ValueError("raw_actions must have shape [B, action_dim] or [B, T, action_dim]")

                        action_weight = self._get_action_weight(step)
                        action_modulation = self.action_projection(
                            prepared_features,
                            num_frames=num_frames
                        ) * action_weight
                        if self.is_main_process and step % self.log_interval == 0:
                            logging.info(f"Action weight at step {step}: {action_weight:.4f}")

                bsz, num_frames = latents.shape[:2]
                timesteps = self._sample_timesteps(bsz, num_frames)
                noise = torch.randn_like(latents)

                noisy_latents = self.scheduler.add_noise(
                    latents.flatten(0, 1),
                    noise.flatten(0, 1),
                    timesteps.flatten(0, 1),
                ).view_as(latents)

                conditional = {"prompt_embeds": prompt_embeds}

                with autocast(dtype=self.autocast_dtype, enabled=self.use_mixed_precision):
                    flow_pred, pred_x0 = forward_model(
                        noisy_latents,
                        conditional,
                        timesteps,
                    )
                    base_loss = self._compute_denoising_loss(
                        latents=latents,
                        noise=noise,
                        noisy_latents=noisy_latents,
                        timesteps=timesteps,
                        flow_pred=flow_pred,
                        pred_x0=pred_x0,
                    )

                if not torch.isfinite(base_loss):
                    raise RuntimeError(f"Non-finite loss encountered at global step {step + 1}")

                accumulated_base_loss += base_loss.detach().item()
                gan_gen_loss = None
                disc_loss = None
                if self.enable_gan_loss:
                    gan_gen_loss, disc_loss = self._compute_gan_losses(
                        real_latents=latents,
                        fake_latents=pred_x0,
                    )
                    if not torch.isfinite(gan_gen_loss):
                        raise RuntimeError(f"Non-finite GAN generator loss at global step {step + 1}")
                    if not torch.isfinite(disc_loss):
                        raise RuntimeError(f"Non-finite GAN discriminator loss at global step {step + 1}")

                kl_loss, distribution_loss = self._compute_distribution_metrics(
                    real_latents=latents.detach(),
                    predicted_latents=pred_x0.detach(),
                )

                total_loss = base_loss
                applied_gan_weight_value = None
                if self.enable_gan_loss and gan_gen_loss is not None:
                    gan_weight_tensor, applied_gan_weight_value = self._compute_adaptive_gan_weight(
                        base_loss=base_loss,
                        gan_loss=gan_gen_loss,
                        fake_latents=pred_x0,
                    )
                    total_loss = total_loss + gan_weight_tensor * gan_gen_loss

                accumulated_loss += total_loss.detach().item()
                accumulated_kl_loss += float(kl_loss.item())
                accumulated_distribution_loss += float(distribution_loss.item())
                if self.enable_gan_loss and gan_gen_loss is not None:
                    accumulated_gan_loss += gan_gen_loss.detach().item()
                    accumulated_disc_loss += disc_loss.detach().item()
                    if applied_gan_weight_value is None:
                        applied_gan_weight_value = self.gan_loss_weight
                    accumulated_gan_weight += applied_gan_weight_value

                loss = total_loss / self.gradient_accumulation

                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if self.enable_gan_loss and disc_loss is not None and self.disc_optimizer is not None:
                    disc_step_loss = disc_loss / self.gradient_accumulation
                    if self.disc_scaler is not None and self.disc_scaler.is_enabled():
                        self.disc_scaler.scale(disc_step_loss).backward()
                    else:
                        disc_step_loss.backward()

            if self.grad_clip is not None and self.grad_clip > 0:
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(base_module.parameters(), self.grad_clip)

            if self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            if self.enable_gan_loss and self.disc_optimizer is not None:
                if self.disc_scaler is not None and self.disc_scaler.is_enabled():
                    self.disc_scaler.step(self.disc_optimizer)
                    self.disc_scaler.update()
                else:
                    self.disc_optimizer.step()

            if (step + 1) % self.log_interval == 0:
                avg_total_loss = accumulated_loss / self.gradient_accumulation
                avg_base_loss = accumulated_base_loss / self.gradient_accumulation
                avg_gan_loss = accumulated_gan_loss / self.gradient_accumulation
                avg_disc_loss = accumulated_disc_loss / self.gradient_accumulation
                avg_gan_weight = accumulated_gan_weight / self.gradient_accumulation
                avg_kl_loss = accumulated_kl_loss / self.gradient_accumulation
                avg_distribution_loss = accumulated_distribution_loss / self.gradient_accumulation

                loss_tensor = torch.tensor(avg_total_loss, device=self.device)
                base_tensor = torch.tensor(avg_base_loss, device=self.device)
                gan_tensor = torch.tensor(avg_gan_loss, device=self.device)
                disc_tensor = torch.tensor(avg_disc_loss, device=self.device)
                weight_tensor = torch.tensor(avg_gan_weight, device=self.device)
                kl_tensor = torch.tensor(avg_kl_loss, device=self.device)
                distribution_tensor = torch.tensor(avg_distribution_loss, device=self.device)

                if self.is_distributed:
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                    dist.all_reduce(base_tensor, op=dist.ReduceOp.AVG)
                    dist.all_reduce(gan_tensor, op=dist.ReduceOp.AVG)
                    dist.all_reduce(disc_tensor, op=dist.ReduceOp.AVG)
                    dist.all_reduce(weight_tensor, op=dist.ReduceOp.AVG)
                    dist.all_reduce(kl_tensor, op=dist.ReduceOp.AVG)
                    dist.all_reduce(distribution_tensor, op=dist.ReduceOp.AVG)

                if self.is_main_process:
                    if self.enable_gan_loss:
                        logging.info(
                            "step %d | loss %.6f | base %.6f | kl %.6f | dist %.6f | gan_gen %.6f | gan_disc %.6f | gan_w %.6f",
                            step + 1,
                            loss_tensor.item(),
                            base_tensor.item(),
                            kl_tensor.item(),
                            distribution_tensor.item(),
                            gan_tensor.item(),
                            disc_tensor.item(),
                            weight_tensor.item(),
                        )
                    else:
                        logging.info(
                            "step %d | loss %.6f | base %.6f | kl %.6f | dist %.6f",
                            step + 1,
                            loss_tensor.item(),
                            base_tensor.item(),
                            kl_tensor.item(),
                            distribution_tensor.item(),
                        )

                if self.is_main_process and not self.disable_wandb and self.wandb_run is not None:
                    try:
                        current_lr = self.optimizer.param_groups[0]["lr"]
                    except (IndexError, KeyError):
                        current_lr = None
                    try:
                        batch_size_value = int(getattr(self.config, "batch_size", 1))
                    except (TypeError, ValueError):
                        batch_size_value = 1
                    global_batch = batch_size_value * self.gradient_accumulation * self.world_size
                    log_dict: Dict[str, Any] = {
                        "train/step": step + 1,
                        "train/loss": loss_tensor.item(),
                        "train/base_loss": base_tensor.item(),
                        "train/kl_loss": kl_tensor.item(),
                        "train/distribution_loss": distribution_tensor.item(),
                        "train/gan_gen_loss": gan_tensor.item(),
                        "train/gan_disc_loss": disc_tensor.item(),
                        "train/gan_weight": weight_tensor.item(),
                        "train/gan_enabled": 1 if self.enable_gan_loss else 0,
                        "train/learning_rate": current_lr,
                        "train/world_size": self.world_size,
                        "train/batch_size": batch_size_value,
                        "train/global_batch_size": global_batch,
                        "train/gradient_accumulation": self.gradient_accumulation,
                    }
                    wandb.log(log_dict, step=step + 1)

            if self.eval_loader is not None and (step + 1) % self.eval_interval == 0:
                barrier()
                self._evaluate_domain_shift(step + 1)
                barrier()

            if self.ckpt_interval > 0 and (step + 1) % self.ckpt_interval == 0:
                barrier()
                self._save_checkpoint(step + 1)
                barrier()

        barrier()
        self._save_checkpoint(self.max_steps)
        if self.eval_loader is not None:
            barrier()
            self._evaluate_domain_shift(self.max_steps, final=True)
        barrier()


def main():
    parser = argparse.ArgumentParser(description="LoRA finetuning for the bidirectional Wan diffusion model.")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--logdir", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--no-auto-resume", action="store_true")
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    if args.logdir:
        config.logdir = args.logdir
    if args.resume:
        config.resume_from = args.resume
    if args.no_auto_resume:
        config.auto_resume = False

    trainer = LoRADiffusionTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
