# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
import gc
import logging
import random
import math
import re
from pathlib import Path
import json

from utils.dataset import TextDataset, TwoTextDataset, VideoLatentCaptionDataset, cycle
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import (
    set_seed,
    merge_dict_list
)
# from utils.action_loss import compute_action_faithfulness_loss
import torch.distributed as dist
from omegaconf import OmegaConf, DictConfig
from model import (
    DMD,
    DMDSwitch,
    DMD2,
    DMD2MSE,
    DMD2Real,
    MSE_DMD,
    MSE_DMD_LAM,
    MSE_DMD_LAM_ACTION,
    DMD2RealMSE,
    DMD2RealMSELAM,
    DMD2RealMSELAM_Actions,
)
from model.streaming_training import StreamingTrainingModel
from latent_actions.train_latent_action_model import Action3DCNN
import torch
import torch.nn as nn
import wandb
import time
import os
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp._common_utils import TrainingState
from torch.distributed.fsdp._flat_param import HandleTrainingState
from torchvision.io import write_video

# LoRA related imports
import peft
from peft import get_peft_model_state_dict
import safetensors.torch

from utils.memory import gpu, get_cuda_free_memory_gb, log_gpu_memory
from pipeline import (
    ActionCausalInferencePipeline,
    CausalInferencePipeline,
    SwitchCausalInferencePipeline
)
from utils.debug_option import DEBUG, LOG_GPU_MEMORY, DEBUG_GRADIENT
# # from one_logger_utils import OneLoggerUtils  # Commented out - module not available
import time

class Trainer:
    
    def __init__(self, config):
        self.config = config
        self.step = 0
        self._zero_grad_reported_steps: set[int] = set()

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb
        self.text_pre_encoded = bool(getattr(config, "text_pre_encoded", False))

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        self.use_one_logger = False #getattr(config, "use_one_logger", False)
        if self.is_main_process and not self.disable_wandb:
            wandb.login(
                # host=config.wandb_host,
                key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir
        app_start_time = time.time_ns() / 1_000_000 
        
        # ------------------------------------- One Logger Setup ----------------------------------------------
        if self.use_one_logger and dist.get_rank() == 0 and not self.disable_wandb:
            app_tag_run_name = f"dmd_{config.real_name[:6]}_local_attn_size_{config.model_kwargs.local_attn_size}_lr_{config.lr}"
            app_tag_run_version = "0.0.0"
            app_tag = f"{app_tag_run_name}_{app_tag_run_version}_{config.batch_size}_{dist.get_world_size()}"
            one_logger_config = {
                "enable_for_current_rank": True,
                "one_logger_async": True,
                "one_logger_project": getattr(config, "one_logger_project", "self-forcing"),
                "log_every_n_train_iterations": getattr(config, "log_iters", 10),
                "app_tag_run_version": app_tag_run_version,
                "summary_data_schema_version": "1.0.0",
                "app_run_type": "training",
                "app_tag": app_tag,
                "app_tag_run_name": app_tag_run_name,
                "one_logger_run_name": app_tag_run_name,
                "world_size": dist.get_world_size(),
                "global_batch_size": config.batch_size * getattr(config, "gradient_accumulation_steps", 1) * dist.get_world_size(),
                "batch_size": config.batch_size,
                "train_iterations_target": getattr(config, "max_iters", 0),
                "train_samples_target": (getattr(config, "max_iters", 0) * config.batch_size) if getattr(config, "max_iters", 0) else 0,
                "is_train_iterations_enabled": True,
                "is_baseline_run": False,
                "is_test_iterations_enabled": False,
                "is_validation_iterations_enabled": True,
                "is_save_checkpoint_enabled": True,
                "is_log_throughput_enabled": False,
                "micro_batch_size": config.batch_size,
                "seq_length": getattr(config, "image_or_video_shape")[1] * getattr(config, "image_or_video_shape")[3] * getattr(config, "image_or_video_shape")[4],
                "save_checkpoint_strategy": "sync",
            }
            # self.one_logger = OneLoggerUtils(one_logger_config)  # Commented out - module not available
            self.one_logger = None  # Disable one_logger functionality
            # self.one_logger.on_app_start(app_start_time = app_start_time)  # Commented out - one_logger disabled  
        else:
            self.one_logger = None

        cfg_flag = getattr(config, "action_patch_enabled", None)
        self._action_patch_enabled = bool(cfg_flag)

        # Step 2: Initialize the model
        if self.one_logger is not None:
            self.one_logger.on_model_init_start()

        if config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        elif config.distribution_loss == "dmd2":
            self.model = DMD2(config, device=self.device)
        elif config.distribution_loss == "dmd_switch":
            self.model = DMDSwitch(config, device=self.device)
        elif config.distribution_loss == "dmd2real":
            self.model = DMD2Real(config, device=self.device)
        elif config.distribution_loss == "dmd2mse":
            self.model = DMD2MSE(config, device=self.device)
        elif config.distribution_loss == "dmd2realmse":
            self.model = DMD2RealMSE(config, device=self.device)
        elif config.distribution_loss == "dmd2realmselam":
            self.model = DMD2RealMSELAM(config, device=self.device)
        elif config.distribution_loss == "dmd2realmselam_actions":
            self.model = DMD2RealMSELAM_Actions(config, device=self.device)
        elif config.distribution_loss == "mse_dmd":
            self.model = MSE_DMD(config, device=self.device)
        elif config.distribution_loss == "mse_dmd_lam":
            self.model = MSE_DMD_LAM(config, device=self.device)
        elif config.distribution_loss == "mse_dmd_lam_action":
            latent_action_model = self._build_latent_action_model(config)
            self.model = MSE_DMD_LAM_ACTION(
                config,
                device=self.device,
                latent_action_model=latent_action_model,
            )
        elif config.distribution_loss in ("dmd2realmselam_actions"):
            latent_action_model = self._build_latent_action_model(config)
            self.model = DMD2RealMSELAM_Actions(
                config,
                device=self.device,
                latent_action_model=latent_action_model,
            )
        else:
            raise ValueError("Invalid distribution matching loss")

        # Save pretrained model state_dicts to CPU
        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

        # Auto resume configuration (needed for LoRA checkpoint loading)
        auto_resume = getattr(config, "auto_resume", True)  # Default to True

        # ================================= LoRA Configuration =================================
        self.is_lora_enabled = False
        self.lora_config = None
        # Track which sub-models have LoRA applied for loading/saving logic
        self.apply_lora_to_critic = False
        self.apply_lora_to_teacher = False
        self.teacher_lora_rank = None
        self.teacher_lora_weights = None
        self.teacher_adapter_name = None
        if hasattr(config, 'adapter') and config.adapter is not None:
            self.is_lora_enabled = True
            self.lora_config = config.adapter
            if isinstance(self.lora_config, DictConfig):
                self.lora_config = OmegaConf.to_container(self.lora_config, resolve=True)
            else:
                # Ensure we have a dict-like structure we can safely query
                self.lora_config = dict(self.lora_config)

            self.apply_lora_to_critic = bool(self.lora_config.get('apply_to_critic', True))

            teacher_lora_value = getattr(config, "teacher_lora", None)
            if isinstance(teacher_lora_value, str):
                teacher_lora_value = teacher_lora_value.strip()
                if teacher_lora_value.lower() in {"", "none"}:
                    teacher_lora_value = None
            if teacher_lora_value is not None:
                try:
                    candidate_rank = int(teacher_lora_value)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"teacher_lora must be an integer rank or 'none', got {teacher_lora_value!r}"
                    )
                if candidate_rank <= 0:
                    raise ValueError("teacher_lora rank must be a positive integer.")
                self.teacher_lora_rank = candidate_rank
                self.teacher_lora_weights = getattr(config, "teacher_lora_weights", None)
                if not self.teacher_lora_weights:
                    raise ValueError(
                        "teacher_lora is set, but teacher_lora_weights was not provided in the config."
                    )
                self.teacher_adapter_name = "default"
                self.apply_lora_to_teacher = True

            generator_cfg = self._get_adapter_config('generator_adapter', default_adapter_name="default")
            if generator_cfg is None:
                raise ValueError("LoRA adapter configuration missing generator settings")
            self.generator_adapter_name = generator_cfg["adapter_name"]
            self.critic_adapter_name = None
            if self.apply_lora_to_critic:
                critic_cfg = self._get_adapter_config('critic_adapter', default_adapter_name="critic")
                if critic_cfg is None:
                    critic_cfg = self._get_adapter_config(None, default_adapter_name="critic")
                self.critic_adapter_name = critic_cfg["adapter_name"]
            
            if self.is_main_process:
                print(f"LoRA enabled with config: {self.lora_config}")
                print("Loading base model and applying LoRA before FSDP wrapping...")
            
            # 1. Load base model first (config.generator_ckpt) - before applying LoRA and FSDP
            base_checkpoint_path = getattr(config, "generator_ckpt", None)
            if base_checkpoint_path:
                if self.is_main_process:
                    print(f"Loading base model from {base_checkpoint_path} (before applying LoRA)")
                base_checkpoint = torch.load(base_checkpoint_path, map_location="cpu")
                
                # Load generator (directly; no key alignment needed since LoRA not applied yet)
                if "generator" in base_checkpoint:
                    if self.is_main_process:
                        print(f"Loading pretrained generator from {base_checkpoint_path}")
                    result = self.model.generator.load_state_dict(base_checkpoint["generator"], strict=True)
                    if self.is_main_process:
                        print("Generator weights loaded successfully")
                elif "model" in base_checkpoint:
                    if self.is_main_process:
                        print(f"Loading pretrained generator from {base_checkpoint_path}")
                    result = self.model.generator.load_state_dict(base_checkpoint["model"], strict=True)
                    if self.is_main_process:
                        print("Generator weights loaded successfully")
                else:
                    if self.is_main_process:
                        print("Warning: Generator checkpoint not found in base model.")
                
                # Load critic
                if "critic" in base_checkpoint:
                    if self.is_main_process:
                        print(f"Loading pretrained critic from {base_checkpoint_path}")
                    result = self.model.fake_score.load_state_dict(base_checkpoint["critic"], strict=True)
                    if self.is_main_process:
                        print("Critic weights loaded successfully")
                else:
                    if self.is_main_process:
                        print("Warning: Critic checkpoint not found in base model.")
            else:
                if self.is_main_process:
                    raise ValueError("No base model checkpoint specified for LoRA training.")
            
            # Load training step
            if "step" in base_checkpoint:
                self.step = base_checkpoint["step"]
                if self.is_main_process:
                    print(f"base_checkpoint step: {self.step}")
            else:
                if self.is_main_process:
                    print("Warning: Step not found in checkpoint, starting from step 0.")
            
            # 2. Apply LoRA wrapping now (after loading base model, before FSDP wrapping)
            if self.is_main_process:
                print("Applying LoRA to models...")
            self.model.generator.model = self._configure_lora_for_model(
                self.model.generator.model, "generator", trainable=True)

            applied_targets = ["generator"]

            if self.apply_lora_to_teacher:
                self.model.real_score.model = self._configure_teacher_lora(self.model.real_score.model)
                self._load_pretrained_teacher_lora(self.model.real_score.model)
                applied_targets.append("teacher")
            elif self.is_main_process:
                print("LoRA not applied to teacher (real_score)")

            if self.apply_lora_to_critic:
                self.model.fake_score.model = self._configure_lora_for_model(
                    self.model.fake_score.model, "fake_score", trainable=True)
                applied_targets.append("critic")
            elif self.is_main_process:
                print("LoRA not applied to critic (fake_score)")

            if self.is_main_process:
                print(f"LoRA applied to: {', '.join(applied_targets)}")
            
            # 3. Load LoRA weights before FSDP wrapping (if a checkpoint is available)
            lora_checkpoint_path = None
            if auto_resume and self.output_path:
                # Find the latest checkpoint and verify it is a LoRA checkpoint
                latest_checkpoint = self.find_latest_checkpoint(self.output_path)
                if latest_checkpoint:
                    try:
                        checkpoint = torch.load(latest_checkpoint, map_location="cpu")
                        expected_lora_keys = {"generator_lora"}
                        if self.apply_lora_to_critic:
                            expected_lora_keys.add("critic_lora")
                        if self.apply_lora_to_teacher:
                            expected_lora_keys.add("teacher_lora")

                        missing_keys = [key for key in expected_lora_keys if key not in checkpoint]
                        if "generator_lora" in checkpoint:
                            lora_checkpoint_path = latest_checkpoint
                            if self.is_main_process:
                                print(f"Auto resume: Found LoRA checkpoint at {lora_checkpoint_path}")
                                if missing_keys:
                                    print(f"Auto resume warning: checkpoint missing LoRA keys {missing_keys}")
                        else:
                            raise ValueError(f"Checkpoint {latest_checkpoint} is not a LoRA checkpoint. "
                                           f"Found keys: {list(checkpoint.keys())}")
                    except Exception as e:
                        if self.is_main_process:
                            print(f"Error validating checkpoint: {e}")
                        raise e
                else:
                    if self.is_main_process:
                        print("Auto resume: No LoRA checkpoint found in logdir")
            elif auto_resume:
                if self.is_main_process:
                    print("Auto resume enabled but no logdir specified for LoRA")
            else:
                if self.is_main_process:
                    print("Auto resume disabled for LoRA")
            
            # If no auto-resumed LoRA checkpoint found, try config.lora_ckpt
            if lora_checkpoint_path is None:
                lora_ckpt_path = getattr(config, "lora_ckpt", None)
                if lora_ckpt_path:
                    try:
                        checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
                        expected_lora_keys = {"generator_lora"}
                        if self.apply_lora_to_critic:
                            expected_lora_keys.add("critic_lora")
                        if self.apply_lora_to_teacher:
                            expected_lora_keys.add("teacher_lora")

                        missing_keys = [key for key in expected_lora_keys if key not in checkpoint]
                        if "generator_lora" in checkpoint:
                            lora_checkpoint_path = lora_ckpt_path
                            if self.is_main_process:
                                print(f"Using explicit LoRA checkpoint: {lora_checkpoint_path}")
                                if missing_keys:
                                    print(f"Explicit LoRA checkpoint warning: missing LoRA keys {missing_keys}")
                        else:
                            raise ValueError(f"Explicit LoRA checkpoint {lora_ckpt_path} is not a valid LoRA checkpoint. "
                                           f"Found keys: {list(checkpoint.keys())}")
                    except Exception as e:
                        if self.is_main_process:
                            print(f"Error loading explicit LoRA checkpoint: {e}")
                        raise e
                else:
                    if self.is_main_process:
                        print("No LoRA checkpoint specified, starting LoRA training from scratch")
            
            # Load LoRA checkpoint (before FSDP wrapping)
            if lora_checkpoint_path:
                if self.is_main_process:
                    print(f"Loading LoRA checkpoint from {lora_checkpoint_path} (before FSDP wrapping)")
                lora_checkpoint = torch.load(lora_checkpoint_path, map_location="cpu")
                
                # Load LoRA weights using PEFT's standard method
                if "generator_lora" in lora_checkpoint:
                    if self.is_main_process:
                        print(f"Loading LoRA generator weights: {len(lora_checkpoint['generator_lora'])} keys in checkpoint")
                    # Use PEFT's set_peft_model_state_dict; it automatically aligns key names
                    peft.set_peft_model_state_dict(
                        self.model.generator.model,
                        lora_checkpoint["generator_lora"],
                        adapter_name=getattr(self, "generator_adapter_name", None),
                    )
                
                if "teacher_lora" in lora_checkpoint:
                    if self.apply_lora_to_teacher:
                        if self.is_main_process:
                            print(f"Loading LoRA teacher weights: {len(lora_checkpoint['teacher_lora'])} keys in checkpoint")
                        peft.set_peft_model_state_dict(
                            self.model.real_score.model,
                            lora_checkpoint["teacher_lora"],
                            adapter_name=self.teacher_adapter_name,
                        )
                    elif self.is_main_process:
                        print("Teacher LoRA weights found in checkpoint, but apply_to_teacher=False; skipping load.")
                elif self.apply_lora_to_teacher and self.is_main_process:
                    print("Warning: LoRA checkpoint missing teacher_lora weights; continuing without loading teacher LoRA.")

                if "critic_lora" in lora_checkpoint:
                    if self.apply_lora_to_critic:
                        if self.is_main_process:
                            print(f"Loading LoRA critic weights: {len(lora_checkpoint['critic_lora'])} keys in checkpoint")
                        
                        # Use PEFT's set_peft_model_state_dict; it automatically aligns key names
                        peft.set_peft_model_state_dict(
                            self.model.fake_score.model,
                            lora_checkpoint["critic_lora"],
                            adapter_name=self.critic_adapter_name if self.critic_adapter_name else None,
                        )
                    elif self.is_main_process:
                        print("Critic LoRA weights found in checkpoint, but apply_to_critic=False; skipping load.")
                elif self.apply_lora_to_critic and self.is_main_process:
                    print("Warning: LoRA checkpoint missing critic_lora weights; continuing without loading critic LoRA.")

                # Load training step
                if "step" in lora_checkpoint:
                    self.step = lora_checkpoint["step"]
                    if self.is_main_process:
                        print(f"Resuming LoRA training from step {self.step}")
            else:
                if self.is_main_process:
                    print("No LoRA checkpoint to load, starting from scratch")

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy
        )

        if self.model.text_encoder is not None:
            self.model.text_encoder = fsdp_wrap(
                self.model.text_encoder,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
                cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
            )
        self.model.vae = self.model.vae.to(
            device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        if getattr(self.model, "action_projection", None) is not None:
            self.model.action_projection = fsdp_wrap(
                self.model.action_projection,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy="size",
                min_num_params=1,
            )

        # Lazily initialize the action-conditioned pipeline now that FSDP wrapping is complete.
        self.model._initialize_inference_pipeline()

        self.extra_generator_modules: list[nn.Module] = []
        if getattr(self.model, "action_projection", None) is not None:
            self.extra_generator_modules.append(self.model.action_projection)

        # if not config.no_visualize or config.load_raw_video:
        #     print("Moving vae to device 2, self.device: ", self.device)
        #     self.model.vae = self.model.vae.to(
        #         device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        # Step 3: Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            if self.is_lora_enabled:
                if self.is_main_process:
                    print(f"EMA disabled in LoRA mode (LoRA provides efficient parameter updates without EMA)")
                self.generator_ema = None
            else:
                print(f"Setting up EMA with weight {ema_weight}")
                self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        
        if self.one_logger is not None:
            self.one_logger.on_model_init_end()
        
        # Step 4: Initialize the optimizer
        if self.one_logger is not None:
            self.one_logger.on_optimizer_init_start()

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters() if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        extra_param_list = []
        for module in self.extra_generator_modules:
            extra_param_list.extend(
                param for param in module.parameters() if param.requires_grad
            )
        self.generator_aux_optimizer = torch.optim.AdamW(
            extra_param_list,
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        ) if extra_param_list else None

        critic_params = [param for param in self.model.fake_score.parameters()
                         if param.requires_grad]
        if (
            getattr(config, "distribution_loss", None) in ("dmd2realmselam", "dmd2realmselam_actions")
            and getattr(self.model, "latent_action_model", None) is not None
        ):
            critic_params.extend(
                param
                for param in self.model.latent_action_model.parameters()
                if param.requires_grad
            )
        self.critic_optimizer = torch.optim.AdamW(
            critic_params,
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        if self.one_logger is not None:
            self.one_logger.on_optimizer_init_end() 

        # Step 5: Initialize the dataloader
        if self.one_logger is not None:
            self.one_logger.on_dataloader_init_start()
        if self.config.i2v:
            dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))
        elif self.config.distribution_loss == "dmd_switch":
            dataset = TwoTextDataset(config.data_path, config.switch_prompt_path)
        elif self.config.distribution_loss in ("dmd2", "dmd2mse", "dmd2real", "dmd2realmse", "dmd2realmselam", "dmd2realmselam_actions", "mse_dmd", "mse_dmd_lam", "mse_dmd_lam_action"):
            dataset = VideoLatentCaptionDataset(
                config.real_latent_root,
                config.caption_root,
                num_frames=getattr(config, "num_training_frames", 21),
                text_pre_encoded=self.text_pre_encoded,
                include_actions=self._action_patch_enabled,
            )
        else:
            dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        # Step 6: Initialize the validation dataloader for visualization (fixed prompts)
        self.fixed_vis_batch = None
        self.vis_interval = getattr(config, "vis_interval", -1)
        if self.vis_interval > 0 and len(getattr(config, "vis_video_lengths", [])) > 0:
            # Determine validation data path
            val_data_path = getattr(config, "val_data_path", None) or config.data_path

            if self.config.i2v:
                val_dataset = ShardingLMDBDataset(val_data_path, max_pair=int(1e8))
            elif self.config.distribution_loss == "dmd_switch":
                val_dataset = TwoTextDataset(val_data_path, config.val_switch_prompt_path)
            elif self.config.distribution_loss in ("dmd2", "dmd2mse", "dmd2real", "dmd2realmse", "dmd2realmselam", "dmd2realmselam_actions", "dmd2realmselam_action", "mse_dmd", "mse_dmd_lam", "mse_dmd_lam_action"):
                val_latent_root = getattr(config, "val_real_latent_root", None)
                val_caption_root = getattr(config, "val_caption_root", None)

                val_dataset = VideoLatentCaptionDataset(
                    val_latent_root,
                    val_caption_root,
                    num_frames=getattr(config, "num_training_frames", 21),
                    text_pre_encoded=self.text_pre_encoded,
                    include_actions=self._action_patch_enabled,
                )
            else:
                val_dataset = TextDataset(val_data_path)

            if dist.get_rank() == 0:
                print("VAL DATASET SIZE %d" % len(val_dataset))

            sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset, shuffle=False, drop_last=False)
            # streaming sampling to keep prompts fixed
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=getattr(config, "val_batch_size", 1),
                sampler=sampler,
                num_workers=8,
            )

            # Take the first batch as fixed visualization batch
            try:
                self.fixed_vis_batch = next(iter(val_dataloader))
            except StopIteration:
                self.fixed_vis_batch = None
            
            # ----------------------------------------------------------------------------------------------------------
            # Visualization settings
            # ----------------------------------------------------------------------------------------------------------
            # List of video lengths to visualize, e.g. [8, 16, 32]
            self.vis_video_lengths = getattr(config, "vis_video_lengths", [])

            if self.vis_interval > 0 and len(self.vis_video_lengths) > 0:
                self._setup_visualizer()
            
        if self.one_logger is not None:
            self.one_logger.on_dataloader_init_end() 

        if self.one_logger is not None:
            self.one_logger.on_load_checkpoint_start()
        if not self.is_lora_enabled:
            # ================================= Standard (non-LoRA) model logic =================================
            checkpoint_path = None
            
            if auto_resume and self.output_path:
                # Auto resume: find latest checkpoint in logdir
                latest_checkpoint = self.find_latest_checkpoint(self.output_path)
                if latest_checkpoint:
                    checkpoint_path = latest_checkpoint
                    if self.is_main_process:
                        print(f"Auto resume: Found latest checkpoint at {checkpoint_path}")
                else:
                    if self.is_main_process:
                        print("Auto resume: No checkpoint found in logdir, starting from scratch")
            elif auto_resume:
                if self.is_main_process:
                    print("Auto resume enabled but no logdir specified, starting from scratch")
            else:
                if self.is_main_process:
                    print("Auto resume disabled, starting from scratch")
            
            if checkpoint_path is None:
                if getattr(config, "generator_ckpt", False):
                    # Explicit checkpoint path provided
                    checkpoint_path = config.generator_ckpt
                    if self.is_main_process:
                        print(f"Using explicit checkpoint: {checkpoint_path}")

            if checkpoint_path:
                if self.is_main_process:
                    print(f"Loading checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                
                # Load generator
                if "generator" in checkpoint:
                    if self.is_main_process:
                        print(f"Loading pretrained generator from {checkpoint_path}")
                    self.model.generator.load_state_dict(checkpoint["generator"], strict=True)
                elif "model" in checkpoint:
                    if self.is_main_process:
                        print(f"Loading pretrained generator from {checkpoint_path}")
                    self.model.generator.load_state_dict(checkpoint["model"], strict=True)
                else:
                    if self.is_main_process:
                        print("Warning: Generator checkpoint not found.")
                
                # Load critic
                if "critic" in checkpoint:
                    if self.is_main_process:
                        print(f"Loading pretrained critic from {checkpoint_path}")
                    self.model.fake_score.load_state_dict(checkpoint["critic"], strict=True)
                else:
                    if self.is_main_process:
                        print("Warning: Critic checkpoint not found.")
                
                # Load EMA
                if "generator_ema" in checkpoint and self.generator_ema is not None:
                    if self.is_main_process:
                        print(f"Loading pretrained EMA from {checkpoint_path}")
                    self.generator_ema.load_state_dict(checkpoint["generator_ema"])
                else:
                    if self.is_main_process:
                        print("Warning: EMA checkpoint not found or EMA not initialized.")

                if getattr(self.model, "action_projection", None) is not None:
                    expects_projection = (
                        "step" in checkpoint
                        or "generator_optimizer" in checkpoint
                        or "critic_optimizer" in checkpoint
                    )
                    if "action_projection" not in checkpoint:
                        if expects_projection:
                            raise RuntimeError(
                                "Checkpoint is missing 'action_projection' weights required for action-conditioned training."
                            )
                        elif self.is_main_process:
                            print("No action projection weights in checkpoint; starting from freshly initialized module.")
                    else:
                        if self.is_main_process:
                            print("Loading action modulation projection weights from checkpoint")
                        cfg = FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
                        with FSDP.state_dict_type(
                            self.model.action_projection,
                            StateDictType.FULL_STATE_DICT,
                            cfg,
                        ):
                            self.model.action_projection.load_state_dict(checkpoint["action_projection"], strict=True)
                
                # For auto resume, always resume full training state
                # Load optimizers
                if "generator_optimizer" in checkpoint:
                    if self.is_main_process:
                        print("Resuming generator optimizer...")
                    gen_osd = FSDP.optim_state_dict_to_load(
                        self.model.generator,              # FSDP root module
                        self.generator_optimizer,          # newly created optimizer
                        checkpoint["generator_optimizer"]  # optimizer state dict at save time
                    )
                    self.generator_optimizer.load_state_dict(gen_osd)
                else:
                    if self.is_main_process:
                        print("Warning: Generator optimizer checkpoint not found.")

                if (
                    self.generator_aux_optimizer is not None
                    and "action_optimizer" in checkpoint
                    and getattr(self.model, "action_projection", None) is not None
                ):
                    if self.is_main_process:
                        print("Resuming action modulation optimizer...")
                    action_osd = FSDP.optim_state_dict_to_load(
                        self.model.action_projection,
                        self.generator_aux_optimizer,
                        checkpoint["action_optimizer"],
                    )
                    self.generator_aux_optimizer.load_state_dict(action_osd)
                elif self.generator_aux_optimizer is not None and self.is_main_process:
                    print("Warning: Action modulation optimizer state not found in checkpoint.")
                
                if "critic_optimizer" in checkpoint:
                    if self.is_main_process:
                        print("Resuming critic optimizer...")
                    crit_osd = FSDP.optim_state_dict_to_load(
                        self.model.fake_score,
                        self.critic_optimizer,
                        checkpoint["critic_optimizer"]
                    )
                    self.critic_optimizer.load_state_dict(crit_osd)
                else:
                    if self.is_main_process:
                        print("Warning: Critic optimizer checkpoint not found.")
                
                # Load training step
                if "step" in checkpoint:
                    self.step = checkpoint["step"]
                    if self.is_main_process:
                        print(f"Resuming from step {self.step}")
                else:
                    if self.is_main_process:
                        print("Warning: Step not found in checkpoint, starting from step 0.")

        if self.one_logger is not None:
            self.one_logger.on_load_checkpoint_end()
        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        # Note: This should be done after potential resume to avoid accidentally deleting resumed EMA
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.gradient_accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)
        self.previous_time = None

        self._setup_weight_schedules()
        
        # streaming training configuration
        self.streaming_training = getattr(config, "streaming_training", False)
        self.streaming_chunk_size = getattr(config, "streaming_chunk_size", 21)
        self.streaming_max_length = getattr(config, "streaming_max_length", 63)
        
        # Create streaming training model if enabled
        if self.streaming_training:
            self.streaming_model = StreamingTrainingModel(self.model, config)
            if self.is_main_process:
                print(f"streaming training enabled: chunk_size={self.streaming_chunk_size}, max_length={self.streaming_max_length}")
        else:
            self.streaming_model = None
        
        # streaming training state (simplified)
        self.streaming_active = False  # Whether we're currently in a sequence
        
        if self.is_main_process:
            print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
            if self.gradient_accumulation_steps > 1:
                print(f"Effective batch size: {config.batch_size * self.gradient_accumulation_steps * self.world_size}")
            if self.streaming_training:
                print(f"streaming training enabled: chunk_size={self.streaming_chunk_size}, max_length={self.streaming_max_length}")
            if LOG_GPU_MEMORY:
                log_gpu_memory("After initialization", device=self.device, rank=dist.get_rank())

        
        if self.one_logger is not None:
            self.one_logger.on_train_start(train_iterations_start = self.step, train_samples_start = self.step * self.config.batch_size)
        
    def _setup_weight_schedules(self):
        """Configure generator MSE and GAN loss weight schedules."""
        total_iters = getattr(self.config, "max_iters", 0)
        self._schedule_total_iters = max(1, int(total_iters)) if total_iters else 1

        # GAN loss weight schedule (linear). "decay" represents the total change applied across training.
        self._gan_weight_start = float(getattr(self.config, "gan_loss_weight", 0.0))
        gan_decay = getattr(self.config, "gan_loss_weight_decay", None)
        if gan_decay is None:
            self._gan_weight_end = self._gan_weight_start
        else:
            self._gan_weight_end = self._gan_weight_start - float(gan_decay)
        self._gan_weight_end = max(0.0, self._gan_weight_end)
        self._gan_schedule_enabled = abs(self._gan_weight_end - self._gan_weight_start) > 1e-8
        if hasattr(self.model, "gan_loss_weight"):
            self.model.gan_loss_weight = self._gan_weight_start

        # GAN loss weight schedule (linear). "decay" represents the total change applied across training.
        self._latent_action_loss_weight_start = float(getattr(self.config, "latent_action_loss_weight", 0.0))
        latent_action_decay = getattr(self.config, "latent_action_loss_weight_decay", None)
        if latent_action_decay is None:
            self._latent_action_loss_weight_end = self._latent_action_loss_weight_start
        else:
            self._latent_action_loss_weight_end = self._latent_action_loss_weight_start - float(latent_action_decay)
        self._latent_action_loss_weight_end = max(0.0, self._latent_action_loss_weight_end)
        self._latent_action_schedule_enabled = abs(self._latent_action_loss_weight_end - self._latent_action_loss_weight_start) > 1e-8
        if hasattr(self.model, "latent_action_loss_weight"):
            self.model.latent_action_loss_weight = self._latent_action_loss_weight_start

        # Action loss weight schedule (linear). "decay" represents the total change applied across training.
        self._action_loss_weight_start = float(getattr(self.config, "action_loss_weight", 0.0))
        action_decay = getattr(self.config, "action_loss_weight_decay", None)
        if action_decay is None:
            self._action_loss_weight_end = self._action_loss_weight_start
        else:
            self._action_loss_weight_end = self._action_loss_weight_start - float(action_decay)
        self._action_loss_weight_end = max(0.0, self._action_loss_weight_end)
        self._action_schedule_enabled = abs(self._action_loss_weight_end - self._action_loss_weight_start) > 1e-8
        if hasattr(self.model, "action_loss_weight"):
            self.model.action_loss_weight = self._action_loss_weight_start

        # Generator MSE weight schedule (cosine). Positive decay reduces the weight, negative increases it.
        self._mse_weight_start = float(getattr(self.config, "generator_mse_loss_weight", 0.0))
        mse_decay = getattr(self.config, "generator_mse_loss_weight_decay", None)
        if mse_decay is None:
            self._mse_weight_end = self._mse_weight_start
        else:
            self._mse_weight_end = self._mse_weight_start - float(mse_decay)
        self._mse_weight_end = max(0.0, self._mse_weight_end)
        self._mse_schedule_enabled = abs(self._mse_weight_end - self._mse_weight_start) > 1e-8
        if hasattr(self.model, "generator_mse_loss_weight"):
            self.model.generator_mse_loss_weight = self._mse_weight_start

        # Generator DMD loss weight schedule (linear).
        self._dmd_weight_start = float(getattr(self.config, "dmd_loss_weight", 0.0))
        dmd_decay = getattr(self.config, "dmd_loss_weight_decay", None)
        if dmd_decay is None:
            self._dmd_weight_end = self._dmd_weight_start
        else:
            self._dmd_weight_end = self._dmd_weight_start - float(dmd_decay)
        self._dmd_weight_end = max(0.0, self._dmd_weight_end)
        self._dmd_schedule_enabled = abs(self._dmd_weight_end - self._dmd_weight_start) > 1e-8
        if hasattr(self.model, "dmd_loss_weight"):
            self.model.dmd_loss_weight = self._dmd_weight_start

        # Apply schedule immediately in case we are resuming mid-training
        self._update_loss_weights()

    def _update_loss_weights(self):
        """Update loss weights according to the configured schedules."""
        if self._schedule_total_iters is None or self._schedule_total_iters <= 0:
            return
        progress = min(1.0, self.step / self._schedule_total_iters)

        if self._mse_schedule_enabled and hasattr(self.model, "generator_mse_loss_weight"):
            cosine_term = 0.5 * (1.0 + math.cos(math.pi * progress))
            weight = self._mse_weight_end + (self._mse_weight_start - self._mse_weight_end) * cosine_term
            self.model.generator_mse_loss_weight = weight

        if self._dmd_schedule_enabled and hasattr(self.model, "dmd_loss_weight"):
            weight = self._dmd_weight_start + (self._dmd_weight_end - self._dmd_weight_start) * progress
            self.model.dmd_loss_weight = weight

        if self._gan_schedule_enabled and hasattr(self.model, "gan_loss_weight"):
            weight = self._gan_weight_start + (self._gan_weight_end - self._gan_weight_start) * progress
            self.model.gan_loss_weight = weight

        if self._latent_action_schedule_enabled and hasattr(self.model, "latent_action_loss_weight"):
            weight = self._latent_action_loss_weight_start + (self._latent_action_loss_weight_end - self._latent_action_loss_weight_start) * progress
            self.model.latent_action_loss_weight = weight

        if self._action_schedule_enabled and hasattr(self.model, "action_loss_weight"):
            weight = self._action_loss_weight_start + (self._action_loss_weight_end - self._action_loss_weight_start) * progress
            self.model.action_loss_weight = weight

    def _build_latent_action_model(self, config):
        checkpoint_path = getattr(config, "latent_action_checkpoint", None)
        dropout = float(getattr(config, "latent_action_dropout", 0.0))
        hidden_dims_cfg = getattr(config, "latent_action_hidden_dims", (256, 128))
        hidden_dims = list(hidden_dims_cfg) if hidden_dims_cfg else []
        action_dim_cfg = getattr(
            config,
            "latent_action_action_dim",
            getattr(config, "raw_action_dim", getattr(config, "action_dim", 2)),
        )
        in_channels_cfg = getattr(config, "latent_action_in_channels", None)

        default_checkpoint = Path("/projects/u5as/frodobots_lam/latent_actions/checkpoints/1442126/best/input_actions_best.pt")
        if not checkpoint_path:
            if default_checkpoint.exists():
                checkpoint_path = str(default_checkpoint)
                if self.is_main_process:
                    print(f"[latent_action] Using default checkpoint: {checkpoint_path}")
            else:
                checkpoint_path = None

        state_dict = None
        if checkpoint_path:
            payload = torch.load(checkpoint_path, map_location="cpu")
            state_dict = payload.get("model_state", payload)
            if isinstance(payload, dict):
                hidden_dims_payload = payload.get("hidden_dims")
                if hidden_dims_payload:
                    hidden_dims = list(hidden_dims_payload)
                metadata = payload.get("metadata", {})
                dropout = metadata.get("dropout", dropout)
                action_dim_cfg = payload.get("action_dim", action_dim_cfg)

            if "encoder.0.weight" not in state_dict:
                raise KeyError(
                    "Latent action checkpoint missing 'encoder.0.weight' needed to infer input channels."
                )
            in_channels_cfg = state_dict["encoder.0.weight"].shape[1]
            if "head.-1.weight" in state_dict:
                action_dim_cfg = state_dict["head.-1.weight"].shape[0]

        if not hidden_dims:
            hidden_dims = [256, 128]

        if in_channels_cfg is None:
            image_shape = getattr(config, "image_or_video_shape", None)
            if not image_shape or len(image_shape) < 3:
                raise ValueError(
                    "latent_action_in_channels not provided and image_or_video_shape is invalid."
                )
            in_channels_cfg = int(image_shape[2])

        action_dim = int(action_dim_cfg)
        in_channels = int(in_channels_cfg)

        latent_action_model = Action3DCNN(
            in_channels=in_channels,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

        if state_dict is not None:
            latent_action_model.load_state_dict(state_dict, strict=True)

        latent_action_model.to(device=self.device, dtype=torch.float32)
        latent_action_model.eval()
        for param in latent_action_model.parameters():
            param.requires_grad_(False)
        return latent_action_model
        
    def _move_optimizer_to_device(self, optimizer, device):
        """Move optimizer state to the specified device."""
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    def find_latest_checkpoint(self, logdir):
        """Find the latest checkpoint in the logdir."""
        if not os.path.exists(logdir):
            return None
        
        checkpoint_dirs = []
        for item in os.listdir(logdir):
            if item.startswith("checkpoint_model_") and os.path.isdir(os.path.join(logdir, item)):
                try:
                    # Extract step number from directory name
                    step_str = item.replace("checkpoint_model_", "")
                    step = int(step_str)
                    checkpoint_path = os.path.join(logdir, item, "model.pt")
                    if os.path.exists(checkpoint_path):
                        checkpoint_dirs.append((step, checkpoint_path))
                except ValueError:
                    continue
        
        if not checkpoint_dirs:
            return None
        
        # Sort by step number and return the latest one
        checkpoint_dirs.sort(key=lambda x: x[0])
        latest_step, latest_path = checkpoint_dirs[-1]
        return latest_path

    def get_all_checkpoints(self, logdir):
        """Get all checkpoints in the logdir sorted by step number."""
        if not os.path.exists(logdir):
            return []
        
        checkpoint_dirs = []
        for item in os.listdir(logdir):
            if item.startswith("checkpoint_model_") and os.path.isdir(os.path.join(logdir, item)):
                try:
                    # Extract step number from directory name
                    step_str = item.replace("checkpoint_model_", "")
                    step = int(step_str)
                    checkpoint_dir_path = os.path.join(logdir, item)
                    checkpoint_file_path = os.path.join(checkpoint_dir_path, "model.pt")
                    if os.path.exists(checkpoint_file_path):
                        checkpoint_dirs.append((step, checkpoint_dir_path, item))
                except ValueError:
                    continue
        
        # Sort by step number (ascending order)
        checkpoint_dirs.sort(key=lambda x: x[0])
        return checkpoint_dirs

    def cleanup_old_checkpoints(self, logdir, max_checkpoints):
        """Remove old checkpoints if the number exceeds max_checkpoints.
        
        Only the main process performs the actual deletion to avoid race conditions
        in distributed training.
        """
        if max_checkpoints <= 0:
            return
        
        # Only main process should perform cleanup to avoid race conditions
        if not self.is_main_process:
            return
            
        checkpoints = self.get_all_checkpoints(logdir)
        if len(checkpoints) > max_checkpoints:
            # Calculate how many to remove
            num_to_remove = len(checkpoints) - max_checkpoints
            checkpoints_to_remove = checkpoints[:num_to_remove]  # Remove oldest ones
            
            print(f"Checkpoint cleanup: Found {len(checkpoints)} checkpoints, removing {num_to_remove} oldest ones (keeping {max_checkpoints})")
            
            import shutil
            removed_count = 0
            for step, checkpoint_dir_path, dir_name in checkpoints_to_remove:
                try:
                    print(f"  Removing: {dir_name} (step {step})")
                    shutil.rmtree(checkpoint_dir_path)
                    removed_count += 1
                except Exception as e:
                    print(f"  Warning: Failed to remove checkpoint {dir_name}: {e}")
            
            print(f"Checkpoint cleanup completed: removed {removed_count}/{num_to_remove} old checkpoints")
        else:
            if len(checkpoints) > 0:
                print(f"Checkpoint cleanup: Found {len(checkpoints)} checkpoints (max: {max_checkpoints}, no cleanup needed)")

    def _get_switch_frame_index(self, max_length=None):
        if getattr(self.config, "switch_mode", "fixed") == "random":
            block = self.config.num_frame_per_block
            min_idx = self.config.min_switch_frame_index
            max_idx = self.config.max_switch_frame_index
            if min_idx == max_idx:
                switch_idx = min_idx
            else:
                choices = list(range(min_idx, max_idx, block))
                if max_length is not None:
                    choices = [choice for choice in choices if choice < max_length]
                
                if len(choices) == 0:
                    if max_length is not None:
                        raise ValueError(f"No valid switch choices available (all choices >= max_length {max_length})")
                    else:
                        switch_idx = block
                else:
                    if dist.get_rank() == 0:
                        switch_idx = random.choice(choices)
                    else:
                        switch_idx = 0  # placeholder; will be overwritten by broadcast
                switch_idx_tensor = torch.tensor(switch_idx, device=self.device)
                dist.broadcast(switch_idx_tensor, src=0)
                switch_idx = switch_idx_tensor.item()
        elif getattr(self.config, "switch_mode", "fixed") == "fixed":
            switch_idx = getattr(self.config, "fixed_switch_index", 21)
            if max_length is not None:
                assert max_length > switch_idx, f"max_length {max_length} is not greater than switch_idx {switch_idx}"
        elif getattr(self.config, "switch_mode", "fixed") == "random_choice":
            switch_choices = getattr(self.config, "switch_choices", [])
            if len(switch_choices) == 0:
                raise ValueError("switch_choices is empty")
            else:
                if max_length is not None:
                    switch_choices = [choice for choice in switch_choices if choice < max_length]
                    if len(switch_choices) == 0:
                        raise ValueError(f"No valid switch choices available (all choices >= max_length {max_length})")
                
                if dist.get_rank() == 0:
                    switch_idx = random.choice(switch_choices)
                else:
                    switch_idx = 0
            switch_idx_tensor = torch.tensor(switch_idx, device=self.device)
            dist.broadcast(switch_idx_tensor, src=0)
            switch_idx = switch_idx_tensor.item()
        else:
            raise ValueError(f"Invalid switch_mode: {getattr(self.config, 'switch_mode', 'fixed')}")
        return switch_idx

    def _warn_if_no_generator_grads(self, core_has_grad: bool, aux_has_grad: bool) -> None:
        if (core_has_grad or aux_has_grad) or not self.is_main_process:
            return
        if self.step not in self._zero_grad_reported_steps:
            print(
                f"[Warning] Generator backward at logical step {self.step} produced no gradients; "
                f"action modulation may be the only component receiving updates."
            )
            self._zero_grad_reported_steps.add(self.step)

    def _clip_auxiliary_grad_norm(self, max_norm: float, core_norm: torch.Tensor) -> torch.Tensor:
        if not self.extra_generator_modules:
            return torch.zeros((), device=self.device, dtype=core_norm.dtype)

        params: list[torch.Tensor] = []
        for module in self.extra_generator_modules:
            for param in module.parameters():
                if param.grad is not None:
                    params.append(param)

        if not params:
            return torch.zeros((), device=self.device, dtype=core_norm.dtype)

        grad_sq = torch.zeros(1, device=self.device, dtype=torch.float32)
        for param in params:
            grad_sq += torch.sum(param.grad.detach().float() ** 2)
        if dist.is_initialized():
            dist.all_reduce(grad_sq, op=dist.ReduceOp.SUM)

        grad_sq = grad_sq.clamp_min(0.0)
        aux_norm = torch.sqrt(grad_sq).squeeze(0)
        if not torch.isfinite(aux_norm):
            aux_norm = torch.zeros((), device=self.device, dtype=torch.float32)

        if max_norm is not None and max_norm > 0 and torch.isfinite(aux_norm):
            core_val = float(core_norm.detach().float().cpu())
            max_norm_sq = max_norm * max_norm
            residual_sq = max_norm_sq - min(core_val * core_val, max_norm_sq)
            if residual_sq <= 0.0:
                for param in params:
                    param.grad.zero_()
                return aux_norm.new_zeros(())
            residual = math.sqrt(residual_sq)
            aux_val = float(aux_norm.cpu())
            if aux_val > residual:
                scale = residual / (aux_val + 1e-6)
                for param in params:
                    param.grad.mul_(scale)
                aux_norm = aux_norm.new_tensor(residual)

        return aux_norm.to(device=self.device, dtype=core_norm.dtype)

    def _ensure_fsdp_idle(self, module: torch.nn.Module, module_name: str = "module") -> None:
        if not isinstance(module, FSDP):
            return
        reset_state = 0
        reset_handle = 0
        for fsdp_submodule in module.modules():
            if not isinstance(fsdp_submodule, FSDP):
                continue
            if getattr(fsdp_submodule, "training_state", None) != TrainingState.IDLE:
                fsdp_submodule.training_state = TrainingState.IDLE
                fsdp_submodule._needs_pre_forward_unshard = False
                fsdp_submodule._needs_pre_backward_unshard = False
                reset_state += 1
            handle = getattr(fsdp_submodule, "_handle", None)
            if handle is not None and getattr(handle, "_training_state", None) != HandleTrainingState.IDLE:
                handle._training_state = HandleTrainingState.IDLE
                handle._prefetched = False
                reset_handle += 1
        if self.is_main_process and (reset_state or reset_handle):
            print(f"[FSDP:{module_name}] Reset {reset_state} module states and {reset_handle} handles to IDLE prior to checkpointing.")


    def save(self):
        print("Start gathering distributed model states...")
        if getattr(self, 'one_logger', None) is not None and self.is_main_process:
            self.one_logger.on_save_checkpoint_start(global_step=self.step)

        if self.is_lora_enabled:
            gen_lora_sd = self._gather_lora_state_dict(
                self.model.generator.model,
                adapter_name=getattr(self, "generator_adapter_name", None),
            )
            state_dict = {
                "generator_lora": gen_lora_sd,
                "step": self.step,
            }

            if self.apply_lora_to_critic:
                crit_lora_sd = self._gather_lora_state_dict(
                    self.model.fake_score.model,
                    adapter_name=self.critic_adapter_name,
                )
                state_dict["critic_lora"] = crit_lora_sd
            else:
                state_dict["critic_lora"] = {}

            if self.apply_lora_to_teacher:
                teacher_lora_sd = self._gather_lora_state_dict(
                    self.model.real_score.model,
                    adapter_name=self.teacher_adapter_name,
                )
                state_dict["teacher_lora"] = teacher_lora_sd
            else:
                state_dict["teacher_lora"] = {}
        else:
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            if dist.is_initialized():
                dist.barrier()
            self._ensure_fsdp_idle(self.model.generator, module_name="generator")
            generator_state_dict: dict[str, object] = {}
            generator_opim_state_dict: dict[str, object] = {}
            with FSDP.summon_full_params(
                self.model.generator, writeback=False, rank0_only=True
            ):
                if self.is_main_process:
                    generator_state_dict = self.model.generator.module.state_dict()
            full_gen_optim = FSDP.full_optim_state_dict(
                self.model.generator,
                self.generator_optimizer,
                rank0_only=True,
            )
            if self.is_main_process and full_gen_optim is not None:
                generator_opim_state_dict = full_gen_optim
            if not self.is_main_process:
                generator_state_dict = {}
                generator_opim_state_dict = {}

            self._ensure_fsdp_idle(self.model.fake_score, module_name="critic")
            critic_state_dict: dict[str, object] = {}
            critic_opim_state_dict: dict[str, object] = {}
            with FSDP.summon_full_params(
                self.model.fake_score, writeback=False, rank0_only=True
            ):
                if self.is_main_process:
                    critic_state_dict = self.model.fake_score.module.state_dict()
            full_critic_optim = FSDP.full_optim_state_dict(
                self.model.fake_score,
                self.critic_optimizer,
                rank0_only=True,
            )
            if self.is_main_process and full_critic_optim is not None:
                critic_opim_state_dict = full_critic_optim
            if not self.is_main_process:
                critic_state_dict = {}
                critic_opim_state_dict = {}

            if self.config.ema_start_step < self.step and self.generator_ema is not None:
                state_dict = {
                    "generator": generator_state_dict,
                    "critic": critic_state_dict,
                    "generator_ema": self.generator_ema.state_dict(),
                    "generator_optimizer": generator_opim_state_dict,
                    "critic_optimizer": critic_opim_state_dict,
                    "step": self.step,
                }
            else:
                state_dict = {
                    "generator": generator_state_dict,
                    "critic": critic_state_dict,
                    "generator_optimizer": generator_opim_state_dict,
                    "critic_optimizer": critic_opim_state_dict,
                    "step": self.step,
                }

            action_opim_state_dict: dict[str, object] = {}
            if (
                getattr(self.model, "action_projection", None) is not None
                and self.generator_aux_optimizer is not None
            ):
                self._ensure_fsdp_idle(self.model.action_projection, module_name="action_projection")
                full_action_optim = FSDP.full_optim_state_dict(
                    self.model.action_projection,
                    self.generator_aux_optimizer,
                    rank0_only=True,
                )
                if self.is_main_process and full_action_optim is not None:
                    action_opim_state_dict = full_action_optim
                if not self.is_main_process:
                    action_opim_state_dict = {}
                state_dict["action_projection"] = fsdp_state_dict(self.model.action_projection)
                state_dict["action_optimizer"] = action_opim_state_dict
            elif getattr(self.model, "action_projection", None) is not None:
                state_dict["action_projection"] = fsdp_state_dict(self.model.action_projection)

        if self.is_main_process:
            checkpoint_dir = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_file = os.path.join(checkpoint_dir, "model.pt")
            torch.save(state_dict, checkpoint_file)
            print("Model saved to", checkpoint_file)
            
            # Cleanup old checkpoints if max_checkpoints is set
            max_checkpoints = getattr(self.config, "max_checkpoints", 0)
            if max_checkpoints > 0:
                self.cleanup_old_checkpoints(self.output_path, max_checkpoints)

        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
        if self.one_logger is not None:
            self.one_logger.on_save_checkpoint_success(global_step=self.step)
            self.one_logger.on_save_checkpoint_end(global_step=self.step)

    def _load_preencoded_negative_prompt(self) -> torch.Tensor:
        cached = getattr(self, '_cached_negative_prompt_embeds', None)
        if cached is not None:
            return cached

        encoded_path = getattr(self.config, 'negative_prompt_encoded_path', None)
        encoded_values = getattr(self.config, 'negative_prompt_encoded', None)
        tensor: torch.Tensor | None = None

        if encoded_path:
            try:
                with open(encoded_path, 'r', encoding='utf-8') as fh:
                    payload = json.load(fh)
            except Exception as exc:
                raise RuntimeError(f"Failed to load negative prompt embeddings from {encoded_path}: {exc}") from exc
            data = payload.get('caption_encoded')
            if data is None:
                raise ValueError(f"'caption_encoded' missing in negative prompt file {encoded_path}")
            tensor = torch.tensor(data, dtype=torch.float32)
        elif encoded_values is not None:
            tensor = torch.tensor(encoded_values, dtype=torch.float32)

        if tensor is None:
            raise RuntimeError(f"Failed to load negative prompt embeddings from {encoded_path} or {encoded_values}")
        elif tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 3:
            raise ValueError(f'Unexpected negative prompt embedding shape {tensor.shape}')

        self._cached_negative_prompt_embeds = tensor
        return tensor

    def _build_unconditional_from_preencoded(self, batch_size: int) -> dict:
        negative = self._load_preencoded_negative_prompt()
        if negative.shape[0] != batch_size:
            negative = negative[:1].repeat(batch_size, 1, 1)
        unconditional = negative.to(device=self.device, dtype=self.dtype)
        return {'prompt_embeds': unconditional}

    def fwdbwd_one_step(self, batch, train_generator):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 5 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts / embeddings
        if self.text_pre_encoded:
            prompt_embeds = batch.get('prompt_embeds')
            if prompt_embeds is None:
                raise ValueError('Batch missing pre-encoded prompt embeddings while text_pre_encoded is enabled.')
            if isinstance(prompt_embeds, list):
                prompt_embeds = torch.stack(prompt_embeds)
            elif isinstance(prompt_embeds, tuple):
                prompt_embeds = torch.stack(list(prompt_embeds))
            elif not isinstance(prompt_embeds, torch.Tensor):
                prompt_embeds = torch.tensor(prompt_embeds, dtype=torch.float32)
            prompt_embeds = prompt_embeds.to(self.device, dtype=self.dtype)
            batch_size = prompt_embeds.shape[0]
        else:
            text_prompts = batch.get('prompts')
            if text_prompts is None:
                raise ValueError('Batch is missing text prompts.')
            if isinstance(text_prompts, str):
                text_prompts = [text_prompts]
            batch_size = len(text_prompts)

        real_latents = batch.get('real_latents')
        if real_latents is not None:
            if isinstance(real_latents, torch.Tensor):
                real_latents = real_latents.to(self.device, dtype=torch.float32)
            else:
                real_latents = torch.stack(real_latents).to(self.device, dtype=torch.float32)

        actions = batch.get('actions')
        if actions is not None:
            if isinstance(actions, torch.Tensor):
                actions = actions.to(self.device, dtype=torch.float32)
            else:
                actions = torch.stack(actions).to(self.device, dtype=torch.float32)
        if (
            self._action_patch_enabled
            and getattr(self.config, 'distribution_loss', '') in ('dmd2realmselam', 'dmd2realmselam_actions', 'dmd2realmselam_action', 'mse_dmd_lam', 'mse_dmd_lam_action')
            and actions is None
        ):
            raise ValueError('Batch is missing action annotations required for action-guided training.')

        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            if self.text_pre_encoded:
                conditional_dict = {'prompt_embeds': prompt_embeds}
                unconditional_dict = self._build_unconditional_from_preencoded(batch_size)
            else:
                conditional_dict = self.model.text_encoder(
                    text_prompts=text_prompts)

                if not getattr(self, 'unconditional_dict', None):
                    unconditional_dict = self.model.text_encoder(
                        text_prompts=[self.config.negative_prompt] * batch_size)
                    unconditional_dict = {k: v.detach()
                                          for k, v in unconditional_dict.items()}
                    self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
                else:
                    unconditional_dict = self.unconditional_dict

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            generator_kwargs = dict(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=None,
                initial_latent=None,
            )
            if getattr(self.config, 'distribution_loss', '') in ('dmd2', 'dmd2mse', 'dmd2real', 'dmd2realmse', 'dmd2realmselam', 'dmd2realmselam_actions'):
                generator_kwargs['real_latents'] = real_latents
            if actions is not None and getattr(self.config, 'distribution_loss', '') in ('dmd2realmselam', 'dmd2realmselam_actions', 'mse_dmd_lam', 'mse_dmd_lam_action'):
                generator_kwargs['actions'] = actions
            if getattr(self.config, 'distribution_loss', '') in ('mse_dmd', 'mse_dmd_lam', 'mse_dmd_lam_action'):
                generator_kwargs['clean_latent'] = real_latents
            generator_loss, generator_log_dict = self.model.generator_loss(**generator_kwargs)

            # # Compute action faithfulness loss if applicable
            # if actions is not None and hasattr(self.model, 'last_generated_latents') and self.model.last_generated_latents is not None:
            #     try:
            #         action_mae, action_log_dict = compute_action_faithfulness_loss(
            #             pred_latents=self.model.last_generated_latents,
            #             gt_actions=actions,
            #             device=self.device
            #         )
            #         generator_log_dict.update(action_log_dict)
            #     except Exception as e:
            #         if dist.get_rank() == 0:
            #             print(f"Warning: Failed to compute action faithfulness loss: {e}")

            # Scale loss for gradient accumulation and backward
            scaled_generator_loss = generator_loss / self.gradient_accumulation_steps
            scaled_generator_loss.backward()
            if hasattr(self.model, 'fake_score'):
                for p in self.model.fake_score.parameters():
                    if p.grad is not None:
                        p.grad = None
            if LOG_GPU_MEMORY:
                log_gpu_memory("After train_generator backward pass", device=self.device, rank=dist.get_rank())
            core_has_grad = any(param.grad is not None for param in self.model.generator.parameters())
            aux_has_grad = False
            if self.extra_generator_modules:
                for module in self.extra_generator_modules:
                    for param in module.parameters():
                        if param.grad is not None:
                            aux_has_grad = True
                            break
                    if aux_has_grad:
                        break
            self._warn_if_no_generator_grads(core_has_grad, aux_has_grad)
            # Return original loss for logging
            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": torch.tensor(0.0, device=self.device)})  # Will be computed after accumulation

            return generator_log_dict
        else:
            generator_log_dict = {}

        critic_kwargs = dict(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=None,
            initial_latent=None,
        )
        if getattr(self.config, 'distribution_loss', '') in ('dmd2', 'dmd2mse', 'dmd2real', 'dmd2realmse', 'dmd2realmselam', 'dmd2realmselam_actions'):
            critic_kwargs['real_latents'] = real_latents
        if actions is not None and getattr(self.config, 'distribution_loss', '') in ('dmd2realmselam', 'dmd2realmselam_actions', 'mse_dmd_lam', 'mse_dmd_lam_action'):
            critic_kwargs['actions'] = actions
        if getattr(self.config, 'distribution_loss', '') in ('mse_dmd', 'mse_dmd_lam', 'mse_dmd_lam_action'):
            critic_kwargs['clean_latent'] = real_latents
        if train_generator:
            critic_kwargs['update_discriminator'] = True
            critic_loss, critic_log_dict = self.model.critic_loss(**critic_kwargs)
            critic_kwargs['update_discriminator'] = False
        else:
            critic_loss, critic_log_dict = self.model.critic_loss(**critic_kwargs)
        
        

        # Scale loss for gradient accumulation and backward
        scaled_critic_loss = critic_loss / self.gradient_accumulation_steps
        scaled_critic_loss.backward()
        if LOG_GPU_MEMORY:
            log_gpu_memory("After train_critic backward pass", device=self.device, rank=dist.get_rank())
        # Return original loss for logging
        critic_log_dict.update({"critic_loss": critic_loss,
                                "critic_grad_norm": torch.tensor(0.0, device=self.device)})  # Will be computed after accumulation

        return critic_log_dict

    def generate_video(self, pipeline, num_frames, prompts, prompt_embeds=None, image=None, action_inputs=None):
        if prompt_embeds is not None and not isinstance(prompt_embeds, torch.Tensor):
            if isinstance(prompt_embeds, list):
                prompt_embeds = torch.stack(prompt_embeds)
            elif isinstance(prompt_embeds, tuple):
                prompt_embeds = torch.stack(list(prompt_embeds))
            else:
                prompt_embeds = torch.tensor(prompt_embeds, dtype=torch.float32)
        batch_size = prompt_embeds.shape[0] if isinstance(prompt_embeds, torch.Tensor) else len(prompts)
        if image is not None:
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device="cuda", dtype=torch.bfloat16)

            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encode_to_latent(image).to(device="cuda", dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [batch_size, num_frames - 1, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )
        else:
            initial_latent = None
            sampled_noise = torch.randn(
                [batch_size, num_frames, 16, 60, 104],
                device=self.device,
                dtype=self.dtype
            )
        with torch.no_grad():
            video, _ = pipeline.inference(
                noise=sampled_noise,
                text_prompts=prompts if not self.text_pre_encoded else None,
                prompt_embeds=prompt_embeds if self.text_pre_encoded else None,
                return_latents=True,
                action_inputs=action_inputs,
            )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        pipeline.vae.model.clear_cache()
        return current_video
    

    def generate_video_with_switch(self, pipeline, num_frames, prompts, switch_prompts, switch_frame_index, image=None):
        batch_size = len(prompts)
        if image is not None:
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device="cuda", dtype=torch.bfloat16)

            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encode_to_latent(image).to(device="cuda", dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [batch_size, num_frames - 1, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )
        else:
            initial_latent = None
            sampled_noise = torch.randn(
                [batch_size, num_frames, 16, 60, 104],
                device=self.device,
                dtype=self.dtype
            )
        with torch.no_grad():
            video, _ = pipeline.inference(
                noise=sampled_noise,
                text_prompts_first=prompts,
                text_prompts_second=switch_prompts,
                switch_frame_index=switch_frame_index,
                return_latents=True
            )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        pipeline.vae.model.clear_cache()
        return current_video

    def start_new_sequence(self):
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Trainer] start_new_sequence called")
        
        if LOG_GPU_MEMORY:
            log_gpu_memory(f"streaming Training: Before start_new_sequence", device=self.device, rank=dist.get_rank())
        
        # Fetch a new batch
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Trainer] start_new_sequence: fetch new batch")
        batch = next(self.dataloader)

        # Prepare conditional information
        if self.config.i2v:
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            image_latent = None

        if self.text_pre_encoded:
            prompt_embeds = batch.get('prompt_embeds')
            if prompt_embeds is None:
                raise ValueError('Streaming batch missing prompt_embeds while text_pre_encoded is enabled.')
            if isinstance(prompt_embeds, list):
                prompt_embeds = torch.stack(prompt_embeds)
            elif isinstance(prompt_embeds, tuple):
                prompt_embeds = torch.stack(list(prompt_embeds))
            elif not isinstance(prompt_embeds, torch.Tensor):
                prompt_embeds = torch.tensor(prompt_embeds, dtype=torch.float32)
            prompt_embeds = prompt_embeds.to(self.device, dtype=self.dtype)
            batch_size = prompt_embeds.shape[0]
        else:
            text_prompts = batch.get('prompts')
            if isinstance(text_prompts, str):
                text_prompts = [text_prompts]
            batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Trainer] Setting up sequence: batch_size={batch_size}, i2v={self.config.i2v}")
            print(f"[SeqTrain-Trainer] image_or_video_shape={image_or_video_shape}")
        
        with torch.no_grad():
            if self.text_pre_encoded:
                conditional_dict = {'prompt_embeds': prompt_embeds}
                unconditional_dict = self._build_unconditional_from_preencoded(batch_size)
            else:
                conditional_dict = self.model.text_encoder(text_prompts=text_prompts)
                if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"[SeqTrain-Trainer] Created and cached conditional_dict")
                if not getattr(self, 'unconditional_dict', None):
                    unconditional_dict = self.model.text_encoder(
                        text_prompts=[self.config.negative_prompt] * batch_size)
                    unconditional_dict = {k: v.detach() for k, v in unconditional_dict.items()}
                    self.unconditional_dict = unconditional_dict
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[SeqTrain-Trainer] Created and cached unconditional_dict")
                else:
                    unconditional_dict = self.unconditional_dict
        
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"streaming Training: After text encoding", device=self.device, rank=dist.get_rank())
        
        if self.streaming_model.possible_max_length is not None:
            # Ensure all processes choose the same length
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    import random
                    selected_idx = random.randint(0, len(self.streaming_model.possible_max_length) - 1)
                else:
                    selected_idx = 0
                selected_idx_tensor = torch.tensor(selected_idx, device=self.device, dtype=torch.int32)
                dist.broadcast(selected_idx_tensor, src=0)
                selected_idx = selected_idx_tensor.item()
            else:
                import random
                selected_idx = random.randint(0, len(self.streaming_model.possible_max_length) - 1)
            
            temp_max_length = self.streaming_model.possible_max_length[selected_idx]
        else:
            temp_max_length = self.streaming_model.max_length
            
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Model] Selected temporary max length: {temp_max_length} (from {self.streaming_model.possible_max_length})")
        

        # Handle DMD Switch related information
        switch_conditional_dict = None
        switch_frame_index = None
        if isinstance(self.model, DMDSwitch) and "switch_prompts" in batch:
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] Processing DMDSwitch info")
                
            if self.text_pre_encoded:
                raise NotImplementedError('text_pre_encoded is not yet supported for DMDSwitch training.')
            with torch.no_grad():
                switch_conditional_dict = self.model.text_encoder(
                    text_prompts=batch["switch_prompts"]
                )
            switch_frame_index = self._get_switch_frame_index(temp_max_length)
            
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] switch_frame_index={switch_frame_index}")
            
            if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
                log_gpu_memory(f"streaming Training: After switch text encoding", device=self.device, rank=dist.get_rank())
        
        # Set up the sequence
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Trainer] Calling streaming_model.setup_sequence")
            
        self.streaming_model.setup_sequence(
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            initial_latent=image_latent,
            switch_conditional_dict=switch_conditional_dict,
            switch_frame_index=switch_frame_index,
            temp_max_length=temp_max_length,
        )
        
        self.streaming_active = True
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Trainer] streaming training sequence setup completed")
            
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"streaming Training: After sequence setup", device=self.device, rank=dist.get_rank())

    def fwdbwd_one_step_streaming(self, train_generator):
        """Forward/backward pass using the new StreamingTrainingModel for serialized training"""
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 5 == 0:
            torch.cuda.empty_cache()

        # If no active sequence, start a new one
        if not self.streaming_active:
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] No active sequence, starting new one")
            self.start_new_sequence()
        
        # Check whether we can generate more chunks
        if not self.streaming_model.can_generate_more():
            # Current sequence is finished; start a new one
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] Current sequence completed, starting new one")
            self.streaming_active = False
            self.start_new_sequence()
        
        self.kv_cache_before_generator_rollout = None
        self.kv_cache_after_generator_rollout = None
        self.kv_cache_after_generator_backward = None
        self.kv_cache_before_critic_rollout = None
        self.kv_cache_after_critic_rollout = None
        self.kv_cache_after_critic_backward = None
        
        if train_generator:
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] Training generator: generating next chunk")

            train_first_chunk = getattr(self.config, "train_first_chunk", False)
            if train_first_chunk:
                generated_chunk, chunk_info = self.streaming_model.generate_next_chunk(requires_grad=True)
            else:
                current_seq_length = self.streaming_model.state.get("current_length")
                if current_seq_length == 0:
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[SeqTrain-Trainer] train_first_chunk={train_first_chunk}, current_seq_length={current_seq_length}, generate first chunk")
                    generated_chunk, chunk_info = self.streaming_model.generate_next_chunk(requires_grad=False)

                generated_chunk, chunk_info = self.streaming_model.generate_next_chunk(requires_grad=True)
            
                if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"[SeqTrain-Trainer] train_first_chunk={train_first_chunk}, current_seq_length={current_seq_length}")

            # Compute generator loss
            generator_loss, generator_log_dict = self.streaming_model.compute_generator_loss(
                chunk=generated_chunk,
                chunk_info=chunk_info
            )

            # Scale loss for gradient accumulation and backward
            scaled_generator_loss = generator_loss / self.gradient_accumulation_steps
            
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[DEBUG] Scaled generator loss: {scaled_generator_loss.item()}")

            try:
                scaled_generator_loss.backward()
            except RuntimeError as e:
                raise

            generator_log_dict.update({
                "generator_loss": generator_loss,
                "generator_grad_norm": torch.tensor(0.0, device=self.device),
            })
            
            return generator_log_dict
        else:
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] Training critic: generating next chunk")

            train_first_chunk = getattr(self.config, "train_first_chunk", False)
            if train_first_chunk:
                generated_chunk, chunk_info = self.streaming_model.generate_next_chunk(requires_grad=False)
            else:
                current_seq_length = self.streaming_model.state.get("current_length")
                if current_seq_length == 0:
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[SeqTrain-Trainer] train_first_chunk={train_first_chunk}, current_seq_length={current_seq_length}, generate first chunk")
                    generated_chunk, chunk_info = self.streaming_model.generate_next_chunk(requires_grad=False)

                generated_chunk, chunk_info = self.streaming_model.generate_next_chunk(requires_grad=False)
            
                if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"[SeqTrain-Trainer] train_first_chunk={train_first_chunk}, current_seq_length={current_seq_length}")

            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] Generated chunk shape: {generated_chunk.shape}")
                print(f"[SeqTrain-Trainer] Generated chunk requires_grad: {generated_chunk.requires_grad}")
            
            if generated_chunk.requires_grad:
                generated_chunk = generated_chunk.detach()

            if TRAIN_GENERATOR:
                # Compute critic loss
                critic_loss, critic_log_dict = self.streaming_model.compute_critic_loss(
                    chunk=generated_chunk,
                    chunk_info=chunk_info,
                    update_discriminator=True
                )
            else:
                # Compute critic loss
                critic_loss, critic_log_dict = self.streaming_model.compute_critic_loss(
                    chunk=generated_chunk,
                    chunk_info=chunk_info
                )

            
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] Critic loss: {critic_loss.item()}")
            
            # Scale loss for gradient accumulation and backward
            scaled_critic_loss = critic_loss / self.gradient_accumulation_steps
            scaled_critic_loss.backward()
            
            critic_log_dict.update({
                "critic_loss": critic_loss,
                "critic_grad_norm": torch.tensor(0.0, device=self.device),
            })
            
            return critic_log_dict

    def train(self):
        start_step = self.step
        try:
            while True:
                self._update_loss_weights()
                # Check if we should train generator on this optimization step
                TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0
                if LOG_GPU_MEMORY:
                    log_gpu_memory(f"Before training", device=self.device, rank=dist.get_rank())
                
                if dist.get_rank() == 0 and DEBUG:
                    print(f"[Debug] Step {self.step}: switch_mode={getattr(self.config,'switch_mode','fixed')}")

                if self.one_logger is not None:
                    self.one_logger.on_train_batch_start()

                if self.streaming_training:
                    # Zero-out all optimizer gradients
                    if TRAIN_GENERATOR:
                        self.generator_optimizer.zero_grad(set_to_none=True)
                        if self.generator_aux_optimizer is not None:
                            self.generator_aux_optimizer.zero_grad(set_to_none=True)
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    
                    # Whole-cycle gradient accumulation loop
                    accumulated_generator_logs = []
                    accumulated_critic_logs = []
                    
                    for accumulation_step in range(self.gradient_accumulation_steps):
                        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                            print(f"[SeqTrain-Trainer] Whole-cycle accumulation step {accumulation_step + 1}/{self.gradient_accumulation_steps}")
                        
                        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY and accumulation_step == 0:
                            log_gpu_memory(f"streaming Training Step {self.step}: Before whole-cycle forward/backward", device=self.device, rank=dist.get_rank())
                        
                        # Train generator (if needed)
                        if TRAIN_GENERATOR:
                            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                                print(f"[SeqTrain-Trainer] Accumulation step {accumulation_step + 1}: Training generator")
                            extra_gen = self.fwdbwd_one_step_streaming(True)
                            accumulated_generator_logs.append(extra_gen)
                        
                        # Train critic
                        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                            print(f"[SeqTrain-Trainer] Accumulation step {accumulation_step + 1}: Training critic")
                        extra_crit = self.fwdbwd_one_step_streaming(False)
                        accumulated_critic_logs.append(extra_crit)
                        
                        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY and accumulation_step == 0:
                            log_gpu_memory(f"streaming Training Step {self.step}: After whole-cycle forward/backward", device=self.device, rank=dist.get_rank())
                    
                    # Compute grad norm and update parameters
                    if TRAIN_GENERATOR:
                        generator_core_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm_generator)
                        aux_norm = self._clip_auxiliary_grad_norm(self.max_grad_norm_generator, generator_core_norm)
                        generator_log_dict = merge_dict_list(accumulated_generator_logs)
                        core_val = generator_core_norm.detach().float()
                        aux_val = aux_norm.detach().float()
                        total_norm = torch.sqrt(core_val ** 2 + aux_val ** 2).to(device=self.device, dtype=generator_core_norm.dtype)
                        generator_log_dict["generator_grad_norm"] = total_norm
                        generator_log_dict["generator_core_grad_norm"] = generator_core_norm.detach()
                        if self.extra_generator_modules:
                            generator_log_dict["action_mod_grad_norm"] = aux_norm.detach()

                        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                            print(f"[SeqTrain-Trainer] Generator training completed, grad_norm={total_norm.item()}")

                        core_has_grad = core_val.item() > 0.0
                        aux_has_grad = aux_val.item() > 0.0 if self.extra_generator_modules else False
                        self._warn_if_no_generator_grads(core_has_grad, aux_has_grad)

                        self.generator_optimizer.step()
                        if self.generator_aux_optimizer is not None:
                            self.generator_aux_optimizer.step()
                        if self.generator_ema is not None:
                            self.generator_ema.update(self.model.generator)
                    else:
                        generator_log_dict = {}
                    
                    critic_grad_norm = self.model.fake_score.clip_grad_norm_(self.max_grad_norm_critic)
                    critic_log_dict = merge_dict_list(accumulated_critic_logs)
                    critic_log_dict["critic_grad_norm"] = critic_grad_norm
                    
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[SeqTrain-Trainer] Critic training completed, grad_norm={critic_grad_norm.item()}")
                    
                    self.critic_optimizer.step()
                    
                    if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
                        log_gpu_memory(f"streaming Training Step {self.step}: After optimizer steps", device=self.device, rank=dist.get_rank())
                    
                    # Increase step count
                    self.step += 1
                    
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[SeqTrain-Trainer] streaming training step completed: step={self.step}")
                        if hasattr(self, 'streaming_model') and self.streaming_model is not None:
                            current_seq_length = self.streaming_model.state.get("current_length", 0)
                            print(f"[SeqTrain-Trainer] Current sequence length: {current_seq_length}/{self.streaming_model.max_length}")
                            
                    if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
                        log_gpu_memory(f"streaming Training Step {self.step}: Training step completed", device=self.device, rank=dist.get_rank())
                else:
                    if TRAIN_GENERATOR:
                        self.generator_optimizer.zero_grad(set_to_none=True)
                        if self.generator_aux_optimizer is not None:
                            self.generator_aux_optimizer.zero_grad(set_to_none=True)
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    
                    # Whole-cycle gradient accumulation loop
                    accumulated_generator_logs = []
                    accumulated_critic_logs = []
                    
                    for accumulation_step in range(self.gradient_accumulation_steps):
                        batch = next(self.dataloader)
                        
                        # Train generator (if needed)
                        if TRAIN_GENERATOR:
                            extra_gen = self.fwdbwd_one_step(batch, True)
                            accumulated_generator_logs.append(extra_gen)
                        
                        # Train critic
                        extra_crit = self.fwdbwd_one_step(batch, False)
                        accumulated_critic_logs.append(extra_crit)
                    
                    # Compute grad norm and update parameters
                    if TRAIN_GENERATOR:
                        generator_core_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm_generator)
                        aux_norm = self._clip_auxiliary_grad_norm(self.max_grad_norm_generator, generator_core_norm)
                        generator_log_dict = merge_dict_list(accumulated_generator_logs)
                        core_val = generator_core_norm.detach().float()
                        aux_val = aux_norm.detach().float()
                        total_norm = torch.sqrt(core_val ** 2 + aux_val ** 2).to(device=self.device, dtype=generator_core_norm.dtype)
                        generator_log_dict["generator_grad_norm"] = total_norm
                        generator_log_dict["generator_core_grad_norm"] = generator_core_norm.detach()
                        if self.extra_generator_modules:
                            generator_log_dict["action_mod_grad_norm"] = aux_norm.detach()

                        core_has_grad = core_val.item() > 0.0
                        aux_has_grad = aux_val.item() > 0.0 if self.extra_generator_modules else False
                        self._warn_if_no_generator_grads(core_has_grad, aux_has_grad)

                        self.generator_optimizer.step()
                        if self.generator_aux_optimizer is not None:
                            self.generator_aux_optimizer.step()
                        if self.generator_ema is not None:
                            self.generator_ema.update(self.model.generator)
                    else:
                        generator_log_dict = {}
                    
                    critic_grad_norm = self.model.fake_score.clip_grad_norm_(self.max_grad_norm_critic)
                    critic_log_dict = merge_dict_list(accumulated_critic_logs)
                    critic_log_dict["critic_grad_norm"] = critic_grad_norm
                    
                    self.critic_optimizer.step()

                    # Increment the step since we finished gradient update
                    self.step += 1

                if self.one_logger is not None:
                    self.one_logger.on_train_batch_end()

                # Create EMA params (if not already created)
                if (self.step >= self.config.ema_start_step) and \
                        (self.generator_ema is None) and (self.config.ema_weight > 0):
                    if not self.is_lora_enabled:
                        self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)
                        if self.is_main_process:
                            print(f"EMA created at step {self.step} with weight {self.config.ema_weight}")
                    else:
                        if self.is_main_process:
                            print(f"EMA creation skipped at step {self.step} (disabled in LoRA mode)")

                # Save the model
                if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                    torch.cuda.empty_cache()
                    self.save()
                    torch.cuda.empty_cache()

                # Logging
                if self.is_main_process:
                    wandb_loss_dict = {}

                    def _collect_metric(log_dict, key, rename=None):
                        if not log_dict or key not in log_dict:
                            return
                        value = log_dict[key]
                        if isinstance(value, torch.Tensor):
                            value = value.detach()
                            value = value.float().mean().item()
                        elif isinstance(value, (float, int)):
                            value = float(value)
                        else:
                            return
                        wandb_loss_dict[rename or key] = value

                    if TRAIN_GENERATOR and generator_log_dict:
                        for metric in (
                            "generator_loss",
                            "generator_grad_norm",
                            "dmdtrain_gradient_norm",
                            "generator_mse_loss",
                            "generator_mse_raw",
                            "generator_dwt_loss",
                            "generator_dwt_raw",
                            "generator_gan_loss",
                            "generator_gan_logits",
                            "generator_gan_real_prob",
                            "generator_gan_logits_std",
                            "generator_adaptive_gan_weight",
                        ):
                            _collect_metric(generator_log_dict, metric)
                        _collect_metric(generator_log_dict, "latent_action_loss", rename="action_loss")
                        _collect_metric(generator_log_dict, "latent_action_loss_raw")

                    for metric in (
                        "critic_loss",
                        "critic_grad_norm",
                        "critic_denoising_loss",
                        "critic_gan_loss",
                        "critic_cls_loss",
                        "critic_real_logits",
                        "critic_fake_logits",
                        "critic_real_prob",
                        "critic_fake_prob",
                        "critic_real_accuracy",
                        "critic_fake_accuracy",
                    ):
                        _collect_metric(critic_log_dict, metric)

                    if hasattr(self.model, "generator_mse_loss_weight"):
                        wandb_loss_dict["generator_mse_loss_weight_current"] = float(self.model.generator_mse_loss_weight)
                    if hasattr(self.model, "dmd_loss_weight"):
                        wandb_loss_dict["dmd_loss_weight_current"] = float(self.model.dmd_loss_weight)
                    if hasattr(self.model, "gan_loss_weight"):
                        wandb_loss_dict["generator_gan_loss_weight_current"] = float(self.model.gan_loss_weight)
                    if hasattr(self.model, "action_loss_weight"):
                        wandb_loss_dict["generator_action_loss_weight_current"] = float(self.model.action_loss_weight)

                    if not self.disable_wandb and wandb_loss_dict:
                        wandb.log(wandb_loss_dict, step=self.step)

                if self.step % self.config.gc_interval == 0:
                    if dist.get_rank() == 0:
                        logging.info("DistGarbageCollector: Running GC.")
                    gc.collect()
                    torch.cuda.empty_cache()

                if self.is_main_process:
                    current_time = time.time()
                    iteration_time = 0 if self.previous_time is None else current_time - self.previous_time
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": iteration_time}, step=self.step)
                    self.previous_time = current_time
                    # Log training progress
                    if TRAIN_GENERATOR and generator_log_dict:
                        print(f"step {self.step}, per iteration time {iteration_time}, generator_loss {generator_log_dict['generator_loss'].mean().item()}, generator_grad_norm {generator_log_dict['generator_grad_norm'].mean().item()}, dmdtrain_gradient_norm {generator_log_dict['dmdtrain_gradient_norm'].mean().item()}, critic_loss {critic_log_dict['critic_loss'].mean().item()}, critic_grad_norm {critic_log_dict['critic_grad_norm'].mean().item()}")
                    else:
                        print(f"step {self.step}, per iteration time {iteration_time}, critic_loss {critic_log_dict['critic_loss'].mean().item()}, critic_grad_norm {critic_log_dict['critic_grad_norm'].mean().item()}")

                # ---------------------------------------- Visualization ---------------------------------------------------

                if self.vis_interval > 0 and (self.step % self.vis_interval == 0):
                    if self.one_logger is not None:
                        self.one_logger.on_validation_start()

                    try:
                        self._visualize()
                    except Exception as e:
                        print(f"[Warning] Visualization failed at step {self.step}: {e}")
                
                    if self.one_logger is not None:
                        self.one_logger.on_validation_end()
                
                if self.step > self.config.max_iters:
                    break

            if self.one_logger is not None:
                self.one_logger.on_train_end()
                self.one_logger.on_app_end()
        
        except Exception as e:
            if self.is_main_process:
                print(f"[ERROR] Training crashed at step {self.step} with exception: {e}")
                print(f"[ERROR] Exception traceback:", flush=True)
                import traceback
                traceback.print_exc()
        finally:
            # Clean up resources
            if self.one_logger is not None:
                try:
                    self.one_logger.on_train_end()
                    self.one_logger.on_app_end()
                except Exception as cleanup_e:
                    if self.is_main_process:
                        print(f"[WARNING] Failed to clean up one_logger: {cleanup_e}")


    def _get_adapter_config(self, adapter_key=None, default_adapter_name="default"):
        """Resolve adapter configuration from the global LoRA settings and optional overrides."""
        if not self.lora_config:
            return None

        base_cfg = {}
        for key, value in self.lora_config.items():
            if isinstance(value, dict):
                continue
            if key in ("apply_to_critic", "apply_to_teacher"):
                continue
            base_cfg[key] = value

        override_cfg = {}
        if adapter_key:
            override_cfg = self.lora_config.get(adapter_key, {}) or {}

        def _get(key, default=None):
            if key in override_cfg and override_cfg[key] is not None:
                return override_cfg[key]
            if key in base_cfg and base_cfg[key] is not None:
                return base_cfg[key]
            return default

        adapter_cfg = {
            "type": _get("type", "lora"),
            "rank": _get("rank", 16),
            "alpha": _get("alpha", None),
            "dropout": _get("dropout", 0.0),
            "verbose": bool(_get("verbose", False)),
            "adapter_name": _get("adapter_name", default_adapter_name),
        }

        if adapter_cfg["alpha"] is None:
            adapter_cfg["alpha"] = adapter_cfg["rank"]

        return adapter_cfg

    def _build_lora_config(self, adapter_cfg, target_modules, trainable):
        if adapter_cfg["type"] != "lora":
            raise NotImplementedError(f'Adapter type {adapter_cfg["type"]} is not implemented')

        return peft.LoraConfig(
            r=adapter_cfg["rank"],
            lora_alpha=adapter_cfg["alpha"],
            lora_dropout=adapter_cfg["dropout"],
            target_modules=target_modules,
            inference_mode=not trainable,
        )

    def _freeze_adapter_params(self, lora_model, adapter_name):
        for name, param in lora_model.named_parameters():
            if f".{adapter_name}." in name or name.endswith(f".{adapter_name}"):
                param.requires_grad_(False)

    def _configure_lora_for_model(self, transformer, model_name, trainable=True):
        """Configure LoRA for generator or critic models."""
        target_linear_modules = set()

        if model_name == 'generator':
            adapter_target_modules = ['CausalWanAttentionBlock']
        elif model_name == 'fake_score':
            adapter_target_modules = ['WanAttentionBlock']
        else:
            raise ValueError(f"Invalid model name for LoRA configuration: {model_name}")

        for name, module in transformer.named_modules():
            if module.__class__.__name__ in adapter_target_modules:
                for full_submodule_name, submodule in module.named_modules(prefix=name):
                    if isinstance(submodule, torch.nn.Linear):
                        target_linear_modules.add(full_submodule_name)

        target_linear_modules = list(target_linear_modules)

        if model_name == 'generator':
            adapter_cfg = self._get_adapter_config('generator_adapter', default_adapter_name="default")
            if adapter_cfg is None:
                raise ValueError("Generator adapter configuration is required for LoRA training")
            adapter_cfg["adapter_name"] = self.generator_adapter_name or adapter_cfg["adapter_name"]
        else:  # fake_score
            adapter_cfg = self._get_adapter_config('critic_adapter', default_adapter_name="critic")
            if adapter_cfg is None:
                adapter_cfg = self._get_adapter_config(None, default_adapter_name="critic")
            adapter_cfg["adapter_name"] = self.critic_adapter_name or adapter_cfg["adapter_name"]

        if self.is_main_process:
            print(f"LoRA target modules for {model_name}: {len(target_linear_modules)} Linear layers")
            if adapter_cfg.get("verbose"):
                for module_name in sorted(target_linear_modules):
                    print(f"  - {module_name}")

        peft_config = self._build_lora_config(adapter_cfg, target_linear_modules, trainable)
        adapter_name = adapter_cfg["adapter_name"]
        lora_model = peft.get_peft_model(transformer, peft_config, adapter_name=adapter_name)

        if not trainable:
            self._freeze_adapter_params(lora_model, adapter_name)
            if self.is_main_process:
                print(f"LoRA parameters for {model_name} frozen (inference-only).")

        if self.is_main_process:
            print(f"Configured LoRA adapter '{adapter_name}' for {model_name}")
            lora_model.print_trainable_parameters()

        return lora_model

    def _configure_teacher_lora(self, transformer):
        if self.teacher_lora_rank is None or self.teacher_adapter_name is None:
            raise RuntimeError("Teacher LoRA requested but configuration is incomplete.")

        target_linear_modules = set()
        for name, module in transformer.named_modules():
            if module.__class__.__name__ in {"WanAttentionBlock", "CausalWanAttentionBlock"}:
                for full_submodule_name, submodule in module.named_modules(prefix=name):
                    if isinstance(submodule, torch.nn.Linear):
                        target_linear_modules.add(full_submodule_name)

        target_linear_modules = list(target_linear_modules)
        if not target_linear_modules:
            raise RuntimeError("Failed to locate Linear modules for teacher LoRA.")

        adapter_cfg = {
            "type": "lora",
            "rank": self.teacher_lora_rank,
            "alpha": self.teacher_lora_rank,
            "dropout": 0.0,
            "adapter_name": self.teacher_adapter_name,
            "verbose": False,
        }

        if self.is_main_process:
            print(
                f"Teacher LoRA target modules: {len(target_linear_modules)} Linear layers "
                f"(rank={self.teacher_lora_rank})"
            )

        peft_config = self._build_lora_config(adapter_cfg, target_linear_modules, trainable=False)
        lora_model = peft.get_peft_model(transformer, peft_config, adapter_name=self.teacher_adapter_name)
        self._freeze_adapter_params(lora_model, self.teacher_adapter_name)

        if self.is_main_process:
            print("Configured teacher LoRA adapter; parameters frozen for inference.")

        return lora_model

    def _load_pretrained_teacher_lora(self, lora_model):
        if not self.teacher_lora_weights:
            return

        weights_path = Path(self.teacher_lora_weights).expanduser()
        if not weights_path.exists():
            raise FileNotFoundError(f"teacher_lora_weights path does not exist: {weights_path}")

        if self.is_main_process:
            print(f"Loading pretrained teacher LoRA weights from {weights_path}")

        checkpoint = torch.load(weights_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            if "lora" in checkpoint:
                lora_state = checkpoint["lora"]
            elif "teacher_lora" in checkpoint:
                lora_state = checkpoint["teacher_lora"]
            else:
                lora_state = checkpoint
        else:
            raise ValueError(f"Unexpected format for teacher LoRA checkpoint: {type(checkpoint)}")

        peft.set_peft_model_state_dict(
            lora_model,
            lora_state,
            adapter_name=self.teacher_adapter_name,
        )
        self._freeze_adapter_params(lora_model, self.teacher_adapter_name)

        if self.is_main_process:
            print(
                f"Loaded pretrained teacher LoRA ({len(lora_state)} tensors) "
                f"into adapter '{self.teacher_adapter_name}'."
            )


    def _gather_lora_state_dict(self, lora_model, adapter_name=None):
        "On rank-0, gather FULL_STATE_DICT, then filter only LoRA weights"
        with FSDP.state_dict_type(
            lora_model,                       # lora_model contains nested FSDP submodules
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
        ):
            full = lora_model.state_dict()
        return get_peft_model_state_dict(lora_model, state_dict=full, adapter_name=adapter_name)
    
    # --------------------------------------------------------------------------------------------------------------
    # Visualization helpers
    # --------------------------------------------------------------------------------------------------------------

    def _setup_visualizer(self):
        """Initialize the inference pipeline for visualization on CPU, to be moved to GPU only when needed."""

        # Choose pipeline class depending on causal flag
        if 'switch' in self.config.distribution_loss:
            self.vis_pipeline = SwitchCausalInferencePipeline(
                args=self.config,
                device=self.device,
                generator=self.model.generator,
                text_encoder=self.model.text_encoder,
                vae=self.model.vae)
        else:
            if self._action_patch_enabled and getattr(self.model, "action_projection", None) is None:
                raise RuntimeError(
                    "Action projection weights are missing; load a checkpoint that includes 'action_projection' before visualization."
                )
            self.vis_pipeline = ActionCausalInferencePipeline(
                args=self.config,
                device=self.device,
                generator=self.model.generator,
                text_encoder=self.model.text_encoder,
                vae=self.model.vae,
                action_projection=getattr(self.model, "action_projection", None),
            )

        # Visualization output directory (default: <logdir>/vis)
        self.vis_output_dir = os.path.join(os.path.dirname(self.output_path), "vis")
        os.makedirs(self.vis_output_dir, exist_ok=True)
        if self.config.vis_ema:
            raise NotImplementedError("Visualization with EMA is not implemented")

    def _visualize(self):
        """Generate and save sample videos to monitor training progress."""
        if self.vis_interval <= 0 or not hasattr(self, "vis_pipeline"):
            return

        # Use the fixed batch of prompts/images prepared from val_loader
        if not getattr(self, "fixed_vis_batch", None):
            print("[Warning] No fixed validation batch available for visualization.")
            return

        if self.one_logger is not None:
            self.one_logger.on_validation_batch_start()

        step_vis_dir = os.path.join(self.vis_output_dir, f"step_{self.step:07d}")
        os.makedirs(step_vis_dir, exist_ok=True)
        batch = self.fixed_vis_batch
        prompt_embeds = batch.get('prompt_embeds') if isinstance(batch, dict) else None
        action_inputs = batch.get('actions') if isinstance(batch, dict) else None
        if isinstance(self.vis_pipeline, SwitchCausalInferencePipeline):
            prompts = batch["prompts"]
            switch_prompts = batch["switch_prompts"]
            switch_frame_index = self._get_switch_frame_index()
        else:
            prompts = batch["prompts"]

        image = None
        if self.config.i2v and ("image" in batch):
            image = batch["image"]

        # Prepare model mode info for filename
        mode_info = ""
        if self.is_lora_enabled:
            mode_info = "_lora"
            if self.is_main_process:
                print(f"Generating videos in LoRA mode (step {self.step})")
        
        for vid_len in self.vis_video_lengths:
            print(f"Generating video of length {vid_len}")
            if isinstance(self.vis_pipeline, SwitchCausalInferencePipeline):
                videos = self.generate_video_with_switch(self.vis_pipeline, vid_len, prompts, switch_prompts, switch_frame_index, image=image)
            else:
                videos = self.generate_video(self.vis_pipeline, vid_len, prompts, prompt_embeds=prompt_embeds, image=image, action_inputs=action_inputs)

            # Save each sample
            for idx, video_np in enumerate(videos):
                if isinstance(self.vis_pipeline, SwitchCausalInferencePipeline):
                    video_name = f"step_{self.step:07d}_rank_{dist.get_rank()}_sample_{idx}_len_{vid_len}{mode_info}_switch_frame_{switch_frame_index}.mp4"
                else:
                    video_name = f"step_{self.step:07d}_rank_{dist.get_rank()}_sample_{idx}_len_{vid_len}{mode_info}.mp4"
                out_path = os.path.join(
                    step_vis_dir,
                    video_name,
                )
                video_tensor = torch.from_numpy(video_np.astype("uint8"))
                write_video(out_path, video_tensor, fps=16)

            # After saving current length videos, release related tensors to reduce peak memory
            del videos, video_np, video_tensor  # type: ignore
            torch.cuda.empty_cache()

        if self.one_logger is not None:
            self.one_logger.on_validation_batch_end()

        torch.cuda.empty_cache()
        import gc
        gc.collect()
