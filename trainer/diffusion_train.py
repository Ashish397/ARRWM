import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
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
from model.action_encoder import ActionEncoder

def _is_distributed() -> bool:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    return world > 1


class LoRADiffusionTrainer:
    """LoRA finetuning for the bidirectional Wan 1.3B diffusion model."""

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
        self.ckpt_interval = int(getattr(config, "checkpoint_interval", 1000))
        self.num_workers = int(getattr(config, "num_workers", 8))
        self.pin_memory = bool(getattr(config, "pin_memory", True))
        self.prefetch_factor = getattr(config, "prefetch_factor", None)
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
        self.start_step = 0

        self.action_encoder = None
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
        dataset = VideoLatentCaptionDataset(
            latent_root=self.config.real_latent_root,
            caption_root=self.config.caption_root,
            num_frames=getattr(self.config, "num_training_frames", 21),
            text_pre_encoded=bool(getattr(self.config, "text_pre_encoded", False)),
        )
        if self.is_main_process:
            logging.info('Training dataset size: %d', len(dataset))

        if self.is_distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.global_rank,
                shuffle=True,
                drop_last=True,
            )
        else:
            sampler = None

        dataloader_kwargs: Dict[str, Any] = dict(
            batch_size=int(getattr(self.config, "batch_size", 1)),
            sampler=sampler,
            shuffle=(sampler is None),
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
            model_name = getattr(self.config, "real_name", "Wan2.1-T2V-1.3B")
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

            action_dim = getattr(self.config, "action_dim", 512)
            raw_action_dim = getattr(self.config, "raw_action_dim", 2)
            enable_adaln_zero = getattr(self.config, "enable_adaln_zero", True)

            self.action_encoder = ActionEncoder(
                action_dim=raw_action_dim,      # Input: 2D raw actions
                feature_dim=action_dim,          # Output: 512D features
                hidden_dim=256,
                use_sinusoidal=True,
            )
            self.action_encoder.to(self.device)
            self.action_encoder.train()

            self.action_projection = ActionModulationProjection(
                action_dim=action_dim,
                hidden_dim=model_dim,
                num_frames=1,
                zero_init=enable_adaln_zero,
            )

            # Move to device BEFORE DDP wrapping
            self.action_projection.to(self.device)
            self.action_projection.train()
            
            self.use_action_conditioning = True

            if self.is_main_process:
                logging.info(f"Action-injection enabled: raw_action_dim={raw_action_dim}, "
                    f"feature_dim={action_dim}, enable_adaln_zero={enable_adaln_zero}")

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
                if self.action_encoder is not None:
                    self.action_encoder = DDP(
                        self.action_encoder,
                        device_ids=[torch.cuda.current_device()],
                        output_device=torch.cuda.current_device(),
                        broadcast_buffers=False,
                        find_unused_parameters=False,
                    )
                if self.action_projection is not None:
                    self.action_projection = DDP(
                        self.action_projection,
                        device_ids=[torch.cuda.current_device()],
                        output_device=torch.cuda.current_device(),
                        broadcast_buffers=False,
                        find_unused_parameters=False,
                    )

        if self.is_main_process and LOG_GPU_MEMORY:
            log_gpu_memory("After model build", device=self.device, rank=self.global_rank)

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
        )
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

            total_loss = total_loss + batch_loss.detach()
            total_batches = total_batches + 1

        if self.is_distributed and dist.is_initialized():
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_batches, op=dist.ReduceOp.SUM)

        total_batches_value = int(total_batches.item())
        avg_loss = None
        if total_batches_value > 0:
            avg_loss = (total_loss / float(total_batches_value)).item()

        if was_training:
            module.train()

        if avg_loss is not None and self.is_main_process:
            loss_name = "flow_loss" if self.denoising_loss_type == "flow" else "epsilon_mse"
            record = {"step": step, loss_name: avg_loss, "final": final}
            with self.metrics_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
            logging.info("Domain shift eval (step %s): epsilon MSE %.6f", step, avg_loss)

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

        # Add action_projection parameters if using action conditioning
        if self.use_action_conditioning:
            if self.action_encoder is not None:
                encoder_params = [p for p in self.action_encoder.parameters() if p.requires_grad]
                params.extend(encoder_params)
                if self.is_main_process:
                    logging.info(f"Added {len(encoder_params)} action encoder parameters to optimizer")
            
            if self.action_projection is not None:
                proj_params = [p for p in self.action_projection.parameters() if p.requires_grad]
                params.extend(proj_params)
                if self.is_main_process:
                    logging.info(f"Added {len(proj_params)} action projection parameters to optimizer")

        if not params:
            raise RuntimeError("No trainable parameters found after applying LoRA.")
        lr = float(getattr(self.config, "lr", 1e-4))
        betas = (
            float(getattr(self.config, "beta1", 0.9)),
            float(getattr(self.config, "beta2", 0.999)),
        )
        weight_decay = float(getattr(self.config, "weight_decay", 0.01))
        self.optimizer = torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)

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

        checkpoint_config_name = checkpoint.get("config_name", None)
        if checkpoint_config_name is not None and checkpoint_config_name != self.config_name:
            if self.is_main_process:
                logging.warning(
                    f"Checkpoint config name {checkpoint_config_name} does not match current config name {self.config_name}. Skipping auto-resume."
                )
            return
        elif checkpoint_config_name is None:
            if self.is_main_process:
                logging.warning(
                    f"Checkpoint config name is not found. Skipping auto-resume."
                )
            return
        base_module = self.model.module if isinstance(self.model, DDP) else self.model
        set_peft_model_state_dict(base_module.model, checkpoint["lora"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
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
            accumulated_loss = 0.0

            for _ in range(self.gradient_accumulation):
                batch = next(self.data_iter)
                prompt_embeds = self._prepare_prompt_embeds(batch.get("prompt_embeds"))
                latents = self._prepare_latents(batch.get("real_latents"))

                action_modulation = None
                if self.use_action_conditioning and self.action_projection is not None:
                    raw_actions = batch.get("raw_actions", None)
                    if raw_actions is not None:
                        action_features = self.action_encoder(raw_actions.to(self.device))
                        action_weight = self._get_action_weight(step)
                        num_frames = latents.shape[1]

                        action_modulation = self.action_projection(
                            action_features,
                            num_frames=num_frames
                        ) * action_weight
                        if self.is_main_process and step % self.log_interval == 0:
                            logging.info(f"Action weight at step {step}: {action_weight:.4f}")
                    else:
                        action_modulation = None
                else:
                    action_modulation = None

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

                accumulated_loss += base_loss.detach().item()
                loss = base_loss / self.gradient_accumulation

                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

            if self.grad_clip is not None and self.grad_clip > 0:
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(base_module.parameters(), self.grad_clip)

            if self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            if (step + 1) % self.log_interval == 0:
                avg_loss = accumulated_loss / self.gradient_accumulation
                loss_tensor = torch.tensor(avg_loss, device=self.device)
                if self.is_distributed:
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                if self.is_main_process:
                    logging.info("step %d | loss %.6f", step + 1, loss_tensor.item())

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
