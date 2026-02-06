#!/usr/bin/env python3
"""Standalone script to replicate training-time visualization using the exact _visualize() function."""

import argparse
import os
import sys
import gc
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from torchvision.io import write_video

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.dataset import TextDataset, TwoTextDataset, VideoLatentCaptionDataset
from utils.misc import set_seed
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
    DMD2B2BLAM,
    DMD2B2BLAM_grad,
    DMD2B2BLAM_actions
)
from model.streaming_training import StreamingTrainingModel
from latent_actions.train_motion_2_action import Motion2ActionModel
from cotracker.predictor import CoTrackerPredictor
from pipeline import ActionCausalInferencePipeline, SwitchCausalInferencePipeline
import torch.distributed as dist

# Copy the checkpoint loading function
def _load_checkpoint_with_storage_fallback(path, **torch_load_kwargs):
    """
    Load a checkpoint while handling typed storages that were saved as non-resizable.
    """
    try:
        return torch.load(path, **torch_load_kwargs)
    except RuntimeError as exc:
        msg = str(exc)
        if "not resizable" not in msg:
            raise
        print("[Info] Checkpoint uses non-resizable storages; cloning for compatibility.", file=sys.stderr)

    import torch._utils as torch_utils

    original_rebuild_tensor = torch_utils._rebuild_tensor

    def _rebuild_tensor_with_clone(storage, storage_offset, size, stride):
        untyped = storage._untyped_storage
        if not untyped.resizable():
            untyped = untyped.clone()
        tensor = torch.empty((0,), dtype=storage.dtype, device=untyped.device)
        return tensor.set_(untyped, storage_offset, size, stride)

    torch_utils._rebuild_tensor = _rebuild_tensor_with_clone
    try:
        return torch.load(path, **torch_load_kwargs)
    finally:
        torch_utils._rebuild_tensor = original_rebuild_tensor


class VisualizationTrainer:
    """Minimal trainer class that replicates only the visualization functionality."""
    
    def __init__(self, config, checkpoint_path: Optional[Path] = None, step: int = 0, dataset_actions_only: bool = False, prompt_indices: Optional[list] = None):
        self.config = config
        self.step = step
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.text_pre_encoded = bool(getattr(config, "text_pre_encoded", False))
        self.is_main_process = True
        self.is_lora_enabled = False
        self.dataset_actions_only = dataset_actions_only  # If True, only use dataset actions; if False (default), use both dummy and dataset
        self.prompt_indices = prompt_indices  # List of prompt indices to process (None = all prompts)
        
        # Mock distributed settings for compatibility
        if not dist.is_initialized():
            try:
                os.environ.setdefault('RANK', '0')
                os.environ.setdefault('WORLD_SIZE', '1')
                os.environ.setdefault('MASTER_ADDR', 'localhost')
                os.environ.setdefault('MASTER_PORT', '12355')
                backend = 'nccl' if torch.cuda.is_available() else 'gloo'
                dist.init_process_group(backend=backend, init_method='env://')
            except Exception as e:
                print(f"[Warning] Could not initialize distributed process group: {e}")
                print("[Info] Continuing without distributed setup...")
        
        # Set seed
        seed = getattr(config, "seed", 42)
        set_seed(seed)
        
        # Action patch enabled
        cfg_flag = getattr(config, "action_patch_enabled", None)
        self._action_patch_enabled = bool(cfg_flag)
        print(f"Action patch enabled: {self._action_patch_enabled}")
        
        # Initialize model (same as Trainer.__init__)
        self._init_model()
        
        # Load checkpoint if provided
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)
        
        # Initialize visualization dataset and batch
        self._init_visualization_data()
        
        # Output path for compatibility (needed before _setup_visualizer)
        if hasattr(config, "logdir"):
            self.output_path = config.logdir
        elif checkpoint_path is not None:
            # Extract logdir from checkpoint path structure
            ckpt_parent = checkpoint_path.parent
            if ckpt_parent.name.startswith("checkpoint_model_"):
                self.output_path = str(ckpt_parent.parent)
            else:
                self.output_path = str(ckpt_parent)
        else:
            self.output_path = "./output"
        
        # Setup visualizer (uses self.output_path)
        self._setup_visualizer()
        
        # Disable one_logger (not needed for standalone)
        self.one_logger = None
    
    def _init_model(self):
        """Initialize the model based on distribution_loss."""
        if self.config.distribution_loss == "dmd":
            self.model = DMD(self.config, device=self.device)
        elif self.config.distribution_loss == "dmd2":
            self.model = DMD2(self.config, device=self.device)
        elif self.config.distribution_loss == "dmd_switch":
            self.model = DMDSwitch(self.config, device=self.device)
        elif self.config.distribution_loss == "dmd2real":
            self.model = DMD2Real(self.config, device=self.device)
        elif self.config.distribution_loss == "dmd2mse":
            self.model = DMD2MSE(self.config, device=self.device)
        elif self.config.distribution_loss == "dmd2realmse":
            self.model = DMD2RealMSE(self.config, device=self.device)
        elif self.config.distribution_loss == "dmd2realmselam":
            self.model = DMD2RealMSELAM(self.config, device=self.device)
        elif self.config.distribution_loss == "mse_dmd":
            self.model = MSE_DMD(self.config, device=self.device)
        elif self.config.distribution_loss == "mse_dmd_lam":
            self.model = MSE_DMD_LAM(self.config, device=self.device)
        elif self.config.distribution_loss == "mse_dmd_lam_action":
            latent_action_model = self._build_latent_action_model()
            self.model = MSE_DMD_LAM_ACTION(
                self.config,
                device=self.device,
                latent_action_model=latent_action_model,
            )
        elif self.config.distribution_loss in ("dmd2realmselam_actions"):
            latent_action_model = self._build_latent_action_model()
            self.model = DMD2RealMSELAM_Actions(
                self.config,
                device=self.device,
                latent_action_model=latent_action_model,
            )
        elif self.config.distribution_loss == "dmd2b2blam":
            latent_action_model = self._build_latent_action_model()
            motion_model = self._build_motion_model()
            self.model = DMD2B2BLAM(
                self.config,
                device=self.device,
                latent_action_model=latent_action_model,
                motion_model=motion_model,
            )
        elif self.config.distribution_loss == "dmd2b2blam_grad":
            latent_action_model = self._build_latent_action_model()
            motion_model = self._build_motion_model()
            self.model = DMD2B2BLAM_grad(
                self.config,
                device=self.device,
                latent_action_model=latent_action_model,
                motion_model=motion_model,
            )
        elif self.config.distribution_loss == "dmd2b2blam_actions":
            latent_action_model = self._build_latent_action_model()
            motion_model = self._build_motion_model()
            self.model = DMD2B2BLAM_actions(
                self.config,
                device=self.device,
                latent_action_model=latent_action_model,
                motion_model=motion_model,
            )
        else:
            raise ValueError(f"Invalid distribution matching loss: {self.config.distribution_loss}")
    
    def _build_latent_action_model(self):
        """Build Motion2ActionModel for dmd2b2blam."""
        # Get checkpoint path from config
        checkpoint_path = getattr(self.config, "latent_action_checkpoint", None)
        if checkpoint_path is None:
            raise ValueError("latent_action_checkpoint must be set in config for dmd2b2blam model")
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Latent action checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = _load_checkpoint_with_storage_fallback(str(checkpoint_path), map_location="cpu")
        checkpoint_config = checkpoint.get('config', {})
        film_hidden_dim = checkpoint_config.get('film_hidden_dim', 1024)
        head_mode = checkpoint_config.get('head_mode', 'distribution')
        
        # Initialize model
        motion2action_model = Motion2ActionModel(
            latent_channels=16,
            film_hidden_dim=film_hidden_dim,
            motion_grid_size=10,
            film_gamma_scale=0.5,
            head_out_t=2,
            head_out_h=2,
            head_out_w=4,
            action_minmax=(-1.0, 1.0),
            head_mode=head_mode,
            log_std_bounds=(-5.0, 2.0),
            dist_eps=1e-6,
            return_dist=(head_mode == "distribution"),
        )
        
        # Load model state
        motion2action_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move to device and freeze
        motion2action_model = motion2action_model.to(device=self.device).eval()
        for param in motion2action_model.parameters():
            param.requires_grad_(False)
        
        return motion2action_model
    
    def _build_motion_model(self):
        """Build MotionModel (CoTracker) for dmd2b2blam."""        
        # CoTrackerPredictor handles checkpoint loading internally
        cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(self.device)
        
        for param in cotracker.parameters():
            param.requires_grad_(False)
        
        return cotracker
    
    def _load_checkpoint(self, checkpoint_path: Path):
        """Load checkpoint into the model."""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = _load_checkpoint_with_storage_fallback(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        
        # Load generator
        generator_state = checkpoint.get("generator") or checkpoint.get("model")
        action_projection_state = checkpoint.get("action_projection")
        
        if generator_state is not None and action_projection_state is None:
            # Extract action_projection from generator_state if embedded
            extracted: dict[str, torch.Tensor] = {}
            filtered_state = generator_state.__class__()
            for key, value in generator_state.items():
                marker = ".action_projection."
                if key.startswith("action_projection.") or marker in key:
                    _, suffix = key.split("action_projection.", 1)
                    extracted[suffix] = value
                else:
                    filtered_state[key] = value
            if extracted:
                action_projection_state = extracted
                generator_state = filtered_state
        
        if generator_state is not None:
            if action_projection_state is None:
                print("Generator checkpoint missing action_projection weights; keeping freshly initialized projection.")
            incompatible = self.model.generator.load_state_dict(generator_state, strict=True)
            if incompatible.missing_keys:
                print(f"Generator load missing keys: {incompatible.missing_keys[:10]}...")
            if incompatible.unexpected_keys:
                print(f"Generator load unexpected keys: {incompatible.unexpected_keys[:10]}...")
        
        # Load action projection
        if action_projection_state is not None:
            target = getattr(self.model, "action_projection", None)
            if target is None:
                raise RuntimeError("Action projection module not initialized but found in checkpoint.")
            print("Loading action modulation projection weights from checkpoint")
            target.load_state_dict(action_projection_state, strict=True)
            
            # Print weight statistics to verify they're non-zero
            with torch.no_grad():
                all_params = []
                for name, param in target.named_parameters():
                    flat = param.data.flatten()
                    all_params.append(flat)
                    print(f"  {name}: shape={param.shape}, mean={flat.mean().item():.6f}, std={flat.std().item():.6f}, "
                          f"abs_mean={flat.abs().mean().item():.6f}, abs_max={flat.abs().max().item():.6f}")
                if all_params:
                    combined = torch.cat(all_params)
                    print(f"  Combined stats: mean={combined.mean().item():.6f}, std={combined.std().item():.6f}, "
                          f"abs_mean={combined.abs().mean().item():.6f}, abs_max={combined.abs().max().item():.6f}")
        
        # Extract step from checkpoint if available
        if "step" in checkpoint:
            self.step = checkpoint["step"]
            print(f"Checkpoint step: {self.step}")
    
    def _init_visualization_data(self):
        """Initialize validation dataset and fixed_vis_batch."""
        self.fixed_vis_batch = None
        self.vis_interval = getattr(self.config, "vis_interval", -1)
        if self.vis_interval > 0 and len(getattr(self.config, "vis_video_lengths", [])) > 0:
            # Determine validation data path
            val_data_path = getattr(self.config, "val_data_path", None) or self.config.data_path
            
            blacklist_path = getattr(self.config, "data_blacklist_path", None)
            
            if self.config.i2v:
                from utils.dataset import ShardingLMDBDataset
                val_dataset = ShardingLMDBDataset(val_data_path, max_pair=int(1e8))
            elif self.config.distribution_loss == "dmd_switch":
                val_dataset = TwoTextDataset(val_data_path, self.config.val_switch_prompt_path)
            elif self.config.distribution_loss in ("dmd2", "dmd2mse", "dmd2real", "dmd2realmse", "dmd2realmselam", "dmd2realmselam_actions", "dmd2realmselam_action", "dmd2b2blam", "dmd2b2blam_grad", "dmd2b2blam_actions", "mse_dmd", "mse_dmd_lam", "mse_dmd_lam_action"):
                val_latent_root = getattr(self.config, "val_real_latent_root", None)
                val_caption_root = getattr(self.config, "val_caption_root", None)
                
                val_dataset = VideoLatentCaptionDataset(
                    val_latent_root,
                    val_caption_root,
                    num_frames=getattr(self.config, "num_training_frames", 21),
                    text_pre_encoded=self.text_pre_encoded,
                    include_actions=self._action_patch_enabled,
                    blacklist_path=blacklist_path,
                )
            else:
                val_dataset = TextDataset(val_data_path)
            
            print(f"VAL DATASET SIZE {len(val_dataset)}")
            
            # Create dataloader with DistributedSampler if distributed is initialized (same as trainer)
            if dist.is_initialized():
                sampler = torch.utils.data.distributed.DistributedSampler(
                    val_dataset, shuffle=False, drop_last=False)
            else:
                sampler = None
            
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=getattr(self.config, "val_batch_size", 1),
                sampler=sampler,
                shuffle=False,
                drop_last=False,
                num_workers=8,
            )
            
            # Store dataloader for iteration (needed when using dataset actions)
            self.val_dataloader = val_dataloader
            
            # Take the first batch as fixed visualization batch (same as trainer)
            try:
                self.fixed_vis_batch = next(iter(val_dataloader))
                print(f"Loaded fixed_vis_batch with keys: {list(self.fixed_vis_batch.keys()) if isinstance(self.fixed_vis_batch, dict) else 'not a dict'}")
            except StopIteration:
                self.fixed_vis_batch = None
                print("[Warning] Validation dataloader is empty")
            
            # Visualization video lengths
            self.vis_video_lengths = getattr(self.config, "vis_video_lengths", [])
    
    def _setup_visualizer(self):
        """Initialize the inference pipeline for visualization."""
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
                print(
                    "[Visualizer] Action projection weights missing in checkpoint; "
                    "reinitializing default projection for visualization."
                )
                self.model._init_action_projection(self.model.args, self.device)
            self.vis_pipeline = ActionCausalInferencePipeline(
                args=self.config,
                device=self.device,
                generator=self.model.generator,
                text_encoder=self.model.text_encoder,
                vae=self.model.vae,
                action_projection=getattr(self.model, "action_projection", None),
            )
        
        # Convert pipeline and components to correct dtype (same as visualise_phase1_actions.py)
        use_bfloat16 = bool(getattr(self.config, "mixed_precision", False)) and self.device.type == "cuda"
        dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        self.vis_pipeline = self.vis_pipeline.to(device=self.device, dtype=dtype)
        self.vis_pipeline.generator.to(device=self.device, dtype=dtype)
        self.vis_pipeline.vae.to(device=self.device, dtype=dtype)
        if getattr(self.vis_pipeline, 'text_encoder', None) is not None:
            self.vis_pipeline.text_encoder.to(device=self.device)
        if getattr(self.vis_pipeline, "action_projection", None) is not None:
            self.vis_pipeline.action_projection.to(device=self.device, dtype=dtype)
        
        # Visualization output directory
        self.vis_output_dir = os.path.join(os.path.dirname(self.output_path), "vis")
        os.makedirs(self.vis_output_dir, exist_ok=True)
        print(f"Visualization output directory: {self.vis_output_dir}")
    
    def _get_switch_frame_index(self, max_length=None):
        """Get switch frame index for switch models."""
        if getattr(self.config, "switch_mode", "fixed") == "random":
            block = self.config.num_frame_per_block
            min_idx = self.config.min_switch_frame_index
            max_idx = self.config.max_switch_frame_index
            if min_idx == max_idx:
                switch_idx = min_idx
            else:
                import random
                choices = list(range(min_idx, max_idx, block))
                if max_length is not None:
                    choices = [choice for choice in choices if choice < max_length]
                if len(choices) == 0:
                    switch_idx = block
                else:
                    switch_idx = random.choice(choices)
            return switch_idx
        else:
            return getattr(self.config, "switch_frame_index", self.config.num_frame_per_block)
    
    def generate_video(self, pipeline, num_frames, prompts, prompt_embeds=None, image=None, action_inputs=None):
        """Generate video using the pipeline - exact copy from Trainer."""
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
            video, latents = pipeline.inference(
                noise=sampled_noise,
                text_prompts=prompts if not self.text_pre_encoded else None,
                prompt_embeds=prompt_embeds if self.text_pre_encoded else None,
                return_latents=True,
                action_inputs=action_inputs,
            )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        pipeline.vae.model.clear_cache()
        return current_video, latents
    
    def generate_video_with_switch(self, pipeline, num_frames, prompts, switch_prompts, switch_frame_index, image=None):
        """Generate video with switch prompts - placeholder for switch models."""
        # This would be the same as generate_video but with switch logic
        # For now, just call generate_video
        result = self.generate_video(pipeline, num_frames, prompts, image=image)
        # Handle both old (video only) and new (video, latents) return formats
        if isinstance(result, tuple):
            return result
        else:
            return result, None
    
    def _visualize(self):
        """Generate and save sample videos to monitor training progress - exact copy from Trainer."""
        if self.vis_interval <= 0 or not hasattr(self, "vis_pipeline"):
            return

        # Set model to eval mode to prevent randomness (same as trainer)
        self.model.eval()

        # Get 8 different samples from the dataloader
        if not hasattr(self, "val_dataloader") or self.val_dataloader is None:
            print("[Warning] No validation dataloader available for visualization.")
            return

        # Collect 8 different samples
        num_samples_needed = 8
        samples = []
        dataloader_iter = iter(self.val_dataloader)
        try:
            while len(samples) < num_samples_needed:
                batch = next(dataloader_iter)
                # If batch_size > 1, flatten into individual samples
                if isinstance(batch, dict):
                    batch_size = len(batch.get("prompts", [])) if isinstance(batch.get("prompts"), list) else 1
                    if batch_size == 1:
                        samples.append(batch)
                    else:
                        # Split batch into individual samples
                        for i in range(batch_size):
                            sample = {}
                            for key, value in batch.items():
                                if isinstance(value, torch.Tensor):
                                    if value.dim() > 0:
                                        sample[key] = value[i:i+1] if i < value.shape[0] else value[0:1]
                                    else:
                                        sample[key] = value
                                elif isinstance(value, list):
                                    sample[key] = [value[i]] if i < len(value) else [value[0]]
                                else:
                                    sample[key] = value
                            samples.append(sample)
                            if len(samples) >= num_samples_needed:
                                break
        except StopIteration:
            print(f"[Warning] Only found {len(samples)} samples in dataloader, need {num_samples_needed}")
            if len(samples) == 0:
                return

        if self.one_logger is not None:
            self.one_logger.on_validation_batch_start()

        step_vis_dir = os.path.join(self.vis_output_dir, f"step_{self.step:07d}")
        os.makedirs(step_vis_dir, exist_ok=True)

        # Create dummy actions for 9 directions with Gaussian noise (used if not using dataset actions)
        action_directions = [
            (0.75, 0.75),
            (0.75, 0),
            (0.75, -0.75),
            (0, -0.75),
            (0, 0.75),
            (-0.75, 0),
            (-0.75, 0.75),
            (-0.75, -0.75),
            (0, 0),
        ]
        num_frames = 21
        action_noise_std = 0.1

        # Prepare model mode info for filename
        mode_info = ""
        if self.is_lora_enabled:
            mode_info = "_lora"
            if self.is_main_process:
                print(f"Generating videos in LoRA mode (step {self.step})")
        
        for vid_len in self.vis_video_lengths:
            print(f"Generating video of length {vid_len}")
            
            # Helper function to generate and save video for a given prompt and actions
            def generate_and_save_video(sample, actions, action_source, prompt_idx, dir_idx=None):
                # Extract prompt and prompt_embeds
                prompt_embeds = sample.get('prompt_embeds') if isinstance(sample, dict) else None
                if isinstance(self.vis_pipeline, SwitchCausalInferencePipeline):
                    prompts = sample["prompts"]
                    switch_prompts = sample.get("switch_prompts", [])
                    switch_frame_index = self._get_switch_frame_index()
                    single_prompts = [prompts[0]] if isinstance(prompts, list) else prompts[:1]
                    single_switch_prompts = [switch_prompts[0]] if isinstance(switch_prompts, list) and switch_prompts else single_prompts
                else:
                    prompts = sample["prompts"]
                    single_prompts = [prompts[0]] if isinstance(prompts, list) else [prompts] if isinstance(prompts, str) else prompts[:1]

                if prompt_embeds is not None:
                    single_prompt_embeds = prompt_embeds[0:1] if prompt_embeds.dim() > 1 else prompt_embeds.unsqueeze(0)
                else:
                    single_prompt_embeds = None

                image = None
                if self.config.i2v and ("image" in sample):
                    image = sample["image"]

                # Compute and print action modulation statistics
                if hasattr(self.vis_pipeline, 'action_projection') and self.vis_pipeline.action_projection is not None:
                    with torch.no_grad():
                        # Compute modulation: [B, F, 6, hidden_dim] where F=num_frames
                        action_mod = self.vis_pipeline.action_projection(actions, num_frames=num_frames)
                        # action_mod shape: [1, 21, 6, hidden_dim]
                        action_mod_flat = action_mod.reshape(-1)  # Flatten for statistics
                        
                        print(f"\n[Prompt {prompt_idx}, {action_source}]")
                        print(f"  Prompt: {single_prompts[0] if single_prompts else 'N/A'}")
                        print(f"  Action input shape: {actions.shape}")
                        print(f"  Action input mean: {actions.mean().item():.6f}, std: {actions.std().item():.6f}")
                        print(f"  Action input min: {actions.min().item():.6f}, max: {actions.max().item():.6f}")
                        print(f"  Action modulation shape: {action_mod.shape}")
                        print(f"  Action modulation mean: {action_mod_flat.mean().item():.6f}")
                        print(f"  Action modulation std: {action_mod_flat.std().item():.6f}")
                        print(f"  Action modulation norm (overall): {action_mod_flat.norm().item():.6f}")
                        print(f"  Action modulation min: {action_mod_flat.min().item():.6f}")
                        print(f"  Action modulation max: {action_mod_flat.max().item():.6f}")
                
                # Generate video
                if isinstance(self.vis_pipeline, SwitchCausalInferencePipeline):
                    _, all_latents = self.generate_video_with_switch(self.vis_pipeline, vid_len, single_prompts, single_switch_prompts, switch_frame_index, image=image)
                else:
                    _, all_latents = self.generate_video(self.vis_pipeline, vid_len, single_prompts, prompt_embeds=single_prompt_embeds, image=image, action_inputs=actions)

                # Generate noise and add it to latents
                noise = torch.randn_like(all_latents)
                flattened_latents = all_latents.flatten(0, 1)  # [B*T, C, H, W]
                flattened_noise = noise.flatten(0, 1)  # [B*T, C, H, W]
                batch_timesteps = flattened_latents.shape[0]  # B*T
                timesteps = torch.full((batch_timesteps,), 25, dtype=torch.long, device=all_latents.device)  # [B*T]
                
                noisy_latent = self.model.scheduler.add_noise(
                    flattened_latents,
                    flattened_noise,
                    timesteps
                ).detach().unflatten(0, all_latents.shape[:2])

                videos = self.vis_pipeline.vae.decode_to_pixel(noisy_latent)
                # decode_to_pixel returns tensor in [-1, 1] range, convert to [0, 1] then [0, 255]
                videos = (videos * 0.5 + 0.5).clamp(0, 1)  # [B, F, C, H, W] in [0, 1]
                videos = (videos * 255.0).round().to(torch.uint8)  # [B, F, C, H, W] in [0, 255]
                videos = videos.permute(0, 1, 3, 4, 2).cpu().numpy()  # [B, F, H, W, C] for write_video

                # Save video
                for idx, video_np in enumerate(videos):
                    rank = dist.get_rank() if dist.is_initialized() else 0
                    # Include direction or dataset in filename
                    if dir_idx is not None:
                        if dir_idx < len(action_directions):
                            dir_x, dir_y = action_directions[dir_idx]
                            direction_suffix = f"_dir_{dir_idx}_{dir_x:+.2f}_{dir_y:+.2f}".replace(".", "p").replace("+", "pos").replace("-", "neg")
                        else:
                            # dir_9: dataset actions
                            direction_suffix = "_dir_9_dataset"
                    else:
                        direction_suffix = "_dataset"
                    
                    if isinstance(self.vis_pipeline, SwitchCausalInferencePipeline):
                        video_name = f"step_{self.step:07d}_rank_{rank}_sample_{idx}_len_{vid_len}{mode_info}_switch_frame_{switch_frame_index}_prompt_{prompt_idx}{direction_suffix}.mp4"
                    else:
                        video_name = f"step_{self.step:07d}_rank_{rank}_sample_{idx}_len_{vid_len}{mode_info}_prompt_{prompt_idx}{direction_suffix}.mp4"
                    out_path = os.path.join(
                        step_vis_dir,
                        video_name,
                    )
                    video_tensor = torch.from_numpy(video_np.astype("uint8"))
                    write_video(out_path, video_tensor, fps=16)
                    
                    # Save actions to a txt file
                    base_name = video_name.rsplit('.mp4', 1)[0]
                    action_path = os.path.join(step_vis_dir, f"{base_name}_actions.txt")
                    
                    with open(action_path, "w") as f:
                        # Write input actions if available
                        if actions is not None:
                            actions_np = actions.cpu().numpy()
                            # Handle batch dimension: [1, num_frames, 2] -> [num_frames, 2]
                            if actions_np.ndim == 3:
                                actions_np = actions_np[0]  # Remove batch dimension
                            # Write input actions (one per line: frame_idx, action_0, action_1)
                            f.write("# Input actions\n")
                            f.write("frame_idx,action_0,action_1\n")
                            for frame_idx, action in enumerate(actions_np):
                                f.write(f"{frame_idx},{action[0]:.6f},{action[1]:.6f}\n")
                        # Use pipeline outputs directly - NO encoding or decoding
                        original_num_frames = video_np.shape[0]  # Store original frame count for truncation later
                        
                        # Prepare decoded video (pixels) for CoTracker
                        # video_np is [num_frames, H, W, C] from pipeline output
                        # Pipeline: video = (decode_to_pixel(...) * 0.5 + 0.5).clamp(0, 1)  # [0, 1] range
                        # Then: current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0  # [0, 255] range
                        # So video_np is already in [0, 255] range
                        # Convert to tensor: [num_frames, H, W, C] -> [1, num_frames, C, H, W]
                        video_tensor = torch.from_numpy(video_np).float()  # [num_frames, H, W, C] in [0, 255]
                        video_tensor = video_tensor.permute(0, 3, 1, 2)  # [num_frames, C, H, W]
                        video_tensor = video_tensor.unsqueeze(0)  # [1, num_frames, C, H, W]
                        video_tensor = video_tensor.to(device=self.device)
                        
                        # Ensure values are in [0, 255] range (clamp to be safe)
                        # Match test_motion_from_scratch.py: pixels are in [0, 255] range
                        video_tensor = video_tensor.clamp(0, 255)
                        
                        # Debug: print video stats to verify normalization
                        if idx == 0:  # Only print for first video to avoid spam
                            print(f"[Debug] Video tensor stats: min={video_tensor.min().item():.2f}, max={video_tensor.max().item():.2f}, mean={video_tensor.mean().item():.2f}, shape={video_tensor.shape}")
                        
                        # Prepare original latents for motion2action - use latents directly from pipeline
                        # Motion2action expects 21 frames (7*3), NOT 84 frames
                        video_latents_reshaped = None
                        if all_latents is not None:
                            # Get latents for this video (handle batch dimension)
                            if isinstance(all_latents, (list, tuple)):
                                video_latents = all_latents[idx] if idx < len(all_latents) else all_latents[0]
                            else:
                                # If all_latents is a tensor, extract the batch index
                                if all_latents.dim() == 5:  # [B, F, C, H, W]
                                    video_latents = all_latents[idx:idx+1] if idx < all_latents.shape[0] else all_latents[0:1]
                                else:
                                    video_latents = all_latents
                            
                            # Ensure latents are on the correct device and dtype
                            # CRITICAL: Use float32 (not self.dtype) to match test_action_from_motion.py
                            if isinstance(video_latents, torch.Tensor):
                                video_latents = video_latents.to(device=self.device, dtype=torch.float32)
                                latent_num_frames = video_latents.shape[1]
                                
                                # Motion2action expects 21 frames (7 segments * 3 frames per segment)
                                # Use original frame count, don't pad to 84
                                if latent_num_frames < 21:
                                    # Pad with last frame to reach 21
                                    frames_needed = 21 - latent_num_frames
                                    last_frame = video_latents[:, -1:, ...]
                                    padding = last_frame.repeat(1, frames_needed, 1, 1, 1)
                                    video_latents = torch.cat([video_latents, padding], dim=1)
                                elif latent_num_frames > 21:
                                    # Truncate to 21 frames (motion2action expects exactly 21)
                                    video_latents = video_latents[:, :21, ...]
                                
                                # Reshape to [7, 3, 16, 60, 104] then permute to [7, 16, 3, 60, 104] as expected by motion2action
                                # This matches dmd2b2blam.py: pred_frames.reshape(7, 3, 16, 60, 104).permute(0, 2, 1, 3, 4)
                                video_latents_reshaped = video_latents.reshape(7, 3, 16, 60, 104).permute(0, 2, 1, 3, 4)  # [7, 16, 3, 60, 104]
                        
                        # Run cotracker to estimate motion (using pixels)
                        # CRITICAL: Match test_motion_from_scratch.py EXACTLY - process in chunks
                        grid_size = 10
                        output_chunk_size = 12  # frames per output window
                        compute_T = 48  # frames per compute chunk - MUST MATCH test_motion_from_scratch.py
                        N = grid_size ** 2
                        n_out_full = compute_T // output_chunk_size
                        assert compute_T % output_chunk_size == 0
                        
                        # Ensure motion_model is on device and in eval mode - EXACTLY like test_motion_from_scratch.py line 103
                        motion_model = self.model.motion_model.to(device=self.device).eval()
                        
                        # Process video in chunks matching test_motion_from_scratch.py exactly
                        all_motion_windows = []
                        T_total = video_tensor.shape[1]
                        num_chunks = (T_total + compute_T - 1) // compute_T
                        
                        print(f"Processing {T_total} frames in {num_chunks} chunks of {compute_T} frames...")
                        
                        # Process in chunks of compute_T frames - EXACTLY like test_motion_from_scratch.py
                        with torch.inference_mode():
                            dropped_first = False
                            for chunk_idx, chunk_start in enumerate(range(0, T_total, compute_T)):
                                if chunk_idx % 5 == 0 or chunk_idx == num_chunks - 1:
                                    print(f"  Processing chunk {chunk_idx + 1}/{num_chunks}...")
                                chunk_end = min(chunk_start + compute_T, T_total)
                                frames_chunk = video_tensor[:, chunk_start:chunk_end, :, :, :]  # [1, T_chunk, 3, H, W]
                                
                                # Drop first frame overall - EXACTLY like test_motion_from_scratch.py line 123-130
                                if not dropped_first:
                                    if frames_chunk.shape[1] == 0:
                                        continue
                                    frames_chunk = frames_chunk[:, 1:, :, :, :]
                                    dropped_first = True
                                    if frames_chunk.shape[1] == 0:
                                        continue
                                
                                # Keep only complete 12-frame windows - EXACTLY like test_motion_from_scratch.py line 132-138
                                n_out = frames_chunk.shape[1] // output_chunk_size
                                if n_out == 0:
                                    continue
                                
                                used_frames = n_out * output_chunk_size
                                frames_chunk = frames_chunk[:, :used_frames, :, :, :].clone()
                                
                                # Run CoTracker - EXACTLY like test_motion_from_scratch.py line 141-142
                                with torch.amp.autocast(device_type='cuda', enabled=True):
                                    pred_tracks, pred_visibility = motion_model(frames_chunk, grid_size=grid_size)  # [B,T,N,2], [B,T,N] or [B,T,N,1]
                                
                                B = pred_tracks.shape[0]
                                T_chunk = pred_tracks.shape[1]
                                
                                # Group into disjoint 12-frame windows - EXACTLY like test_motion_from_scratch.py line 148-153
                                tracks_w = pred_tracks.reshape(B, n_out, output_chunk_size, N, 2)  # [B,n_out,12,N,2]
                                # Handle visibility shape - EXACTLY like test_motion_from_scratch.py line 150-153
                                if pred_visibility.dim() == 3:  # [B,T,N]
                                    vis_w = pred_visibility.reshape(B, n_out, output_chunk_size, N).unsqueeze(-1)  # [B,n_out,12,N,1]
                                else:  # [B,T,N,1]
                                    vis_w = pred_visibility.reshape(B, n_out, output_chunk_size, N, 1)  # [B,n_out,12,N,1]
                                
                                # Compute motion: mean within-window deltas (11 deltas for 12 frames) - EXACTLY like test_motion_from_scratch.py line 156-162
                                d_w = tracks_w[:, :, 1:] - tracks_w[:, :, :-1]  # [B,n_out,11,N,2]
                                motion_out = d_w.mean(dim=2)  # [B,n_out,N,2]
                                
                                # Mean visibility over 12 frames
                                vis_out = vis_w.to(dtype=motion_out.dtype).mean(dim=2)  # [B,n_out,N,1]
                                
                                motion_window = torch.cat([motion_out, vis_out], dim=-1).squeeze(0)  # [n_out,N,3]
                                all_motion_windows.append(motion_window)
                        
                        # Check results OUTSIDE inference_mode block - EXACTLY like test_motion_from_scratch.py line 165
                        if not all_motion_windows:
                            print("ERROR: No motion windows computed!")
                            estimated_motion = None
                        else:
                            # Concatenate all windows: [total_windows, N, 3] - EXACTLY like test_motion_from_scratch.py line 169
                            estimated_motion = torch.cat(all_motion_windows, dim=0)  # [total_out, N, 3]
                            print(f"Motion computation complete. Final motion shape: {estimated_motion.shape}")
                            
                            # We need exactly 7 windows for motion2action
                            if estimated_motion.shape[0] < 7:
                                # Pad with last window
                                last_window = estimated_motion[-1:, :, :]
                                padding = last_window.repeat(7 - estimated_motion.shape[0], 1, 1)
                                estimated_motion = torch.cat([estimated_motion, padding], dim=0)
                            elif estimated_motion.shape[0] > 7:
                                # Truncate to 7 windows
                                estimated_motion = estimated_motion[:7, :, :]
                            
                            # Debug: print motion stats
                            if idx == 0:  # Only print for first video
                                print(f"[Debug] Estimated motion shape: {estimated_motion.shape}")
                                print(f"[Debug] Estimated motion stats: min={estimated_motion.min().item():.4f}, max={estimated_motion.max().item():.4f}, mean={estimated_motion.mean().item():.4f}")
                            
                            # CRITICAL: Normalize motion by dividing by 10.0 
                            # Matches test_motion_2_action.py line 372 and test_motion_2_action_scratch.py line 382
                            # This is required for the motion2action model to work correctly
                            # CRITICAL: Move motion to device and dtype (float32) before normalization, matching test_action_from_motion.py line 147
                            estimated_motion = estimated_motion.to(device=self.device, dtype=torch.float32) / 10.0
                            
                            # Run motion2action model with original latents
                            if video_latents_reshaped is not None and estimated_motion is not None:
                                model_output = self.model.latent_action_model(video_latents_reshaped, estimated_motion)
                                if isinstance(model_output, tuple) and len(model_output) == 3:
                                    _, estimated_action, _ = model_output
                                elif isinstance(model_output, tuple) and len(model_output) == 2:
                                    estimated_action, _ = model_output
                                else:
                                    estimated_action = model_output
                                
                                # Convert to numpy and write
                                estimated_action_np = estimated_action.cpu().numpy()
                                # Handle batch dimension: [7, 2] or [1, num_frames, 2] -> [num_frames, 2]
                                if estimated_action_np.ndim == 2 and estimated_action_np.shape[0] == 7:
                                    # It's [7, 2] - need to expand to frame level (7 segments -> 21 frames, 3 frames per segment)
                                    estimated_action_expanded = []
                                    for seg_idx in range(7):
                                        for _ in range(3):
                                            estimated_action_expanded.append(estimated_action_np[seg_idx])
                                    estimated_action_np = np.array(estimated_action_expanded)
                                elif estimated_action_np.ndim == 3:
                                    estimated_action_np = estimated_action_np[0]  # Remove batch dimension
                                
                                # Truncate to original frame count if we padded
                                if original_num_frames < 84:
                                    estimated_action_np = estimated_action_np[:original_num_frames]
                                
                                # Write estimated actions
                                f.write("\n# Estimated actions (from CoTracker + motion2action model)\n")
                                f.write("frame_idx,action_0,action_1\n")
                                for frame_idx, action in enumerate(estimated_action_np):
                                    f.write(f"{frame_idx},{action[0]:.6f},{action[1]:.6f}\n")
                            else:
                                f.write("\n# Estimated actions: Failed - could not obtain latents or motion\n")

                # After saving videos, release related tensors
                del videos  # type: ignore
                del actions
                torch.cuda.empty_cache()
            
            # Determine which prompts to process
            if self.prompt_indices is not None:
                prompt_list = [idx for idx in self.prompt_indices if 0 <= idx < len(samples)]
            else:
                prompt_list = list(range(min(num_samples_needed, len(samples))))
            
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(f"[Rank {rank}] Processing prompts: {prompt_list}")
            
            # Generate videos with dummy actions (10 directions: 9 fixed + 1 from dataset) for each prompt
            # This gives 8 prompts × 10 directions = 80 videos total
            if not self.dataset_actions_only:
                print(f"[Rank {rank}] Generating videos with dummy actions (10 directions: 9 fixed + 1 from dataset) for {len(prompt_list)} prompts...")
                for prompt_idx in prompt_list:
                    sample = samples[prompt_idx]
                    # First 9 directions: fixed directions with noise
                    for dir_idx, (dir_x, dir_y) in enumerate(action_directions):
                        # Generate dummy actions for this direction
                        base_action = torch.tensor([[dir_x, dir_y]], dtype=torch.float32).repeat(num_frames, 1)
                        noise = torch.randn(num_frames, 2, dtype=torch.float32) * action_noise_std
                        actions = (base_action + noise).unsqueeze(0).to(device=self.device)
                        generate_and_save_video(sample, actions, f"dummy_dir_{dir_idx}", prompt_idx, dir_idx=dir_idx)
                    
                    # 10th direction (dir_9): use actions from the sample file for this prompt
                    if "actions" in sample:
                        # Use dataset actions: shape is [num_frames*3, 2] or [1, num_frames*3, 2]
                        dataset_actions = sample["actions"]
                        if dataset_actions.dim() == 2:
                            # [num_frames*3, 2] -> [1, num_frames*3, 2]
                            dataset_actions = dataset_actions.unsqueeze(0)
                        # Downsample from repeated-per-frame actions
                        # Validate that actions are repeated 3x before downsampling
                        if dataset_actions.shape[1] % 3 != 0:
                            raise ValueError(f"Expected actions length to be a multiple of 3, got {dataset_actions.shape[1]}")
                        trip = dataset_actions.view(dataset_actions.shape[0], -1, 3, dataset_actions.shape[-1])
                        if not (torch.allclose(trip[:, :, 0, :], trip[:, :, 1, :], atol=1e-6) and torch.allclose(trip[:, :, 0, :], trip[:, :, 2, :], atol=1e-6)):
                            raise ValueError("Actions are not repeated in triplets; dataset action alignment is likely broken")
                        # Take one action per triplet: [B, T*3, 2] -> [B, T, 2]
                        dataset_actions = trip[:, :, 0, :].to(device=self.device)
                        # Ensure we have exactly num_frames
                        if dataset_actions.shape[1] > num_frames:
                            dataset_actions = dataset_actions[:, :num_frames, :]
                        elif dataset_actions.shape[1] < num_frames:
                            # Pad with last frame
                            last_frame = dataset_actions[:, -1:, :]
                            padding = last_frame.repeat(1, num_frames - dataset_actions.shape[1], 1)
                            dataset_actions = torch.cat([dataset_actions, padding], dim=1)
                        actions = dataset_actions
                        generate_and_save_video(sample, actions, "dummy_dir_9_dataset", prompt_idx, dir_idx=9)
                    else:
                        print(f"[Warning] Sample {prompt_idx} has no actions, skipping dir_9")
            else:
                # Generate videos with dataset actions instead
                print(f"[Rank {rank}] Generating videos with dataset actions for {len(prompt_list)} prompts...")
                for prompt_idx in prompt_list:
                    sample = samples[prompt_idx]
                    
                    # Use dataset actions if available
                    if "actions" in sample:
                        # Use dataset actions: shape is [num_frames*3, 2], need to reshape to [1, num_frames, 2]
                        dataset_actions = sample["actions"]  # [num_frames*3, 2] or [1, num_frames*3, 2]
                        if dataset_actions.dim() == 2:
                            # [num_frames*3, 2] -> [1, num_frames*3, 2]
                            dataset_actions = dataset_actions.unsqueeze(0)
                        # Downsample from repeated-per-frame actions
                        # Validate that actions are repeated 3x before downsampling
                        if dataset_actions.shape[1] % 3 != 0:
                            raise ValueError(f"Expected actions length to be a multiple of 3, got {dataset_actions.shape[1]}")
                        trip = dataset_actions.view(dataset_actions.shape[0], -1, 3, dataset_actions.shape[-1])
                        if not (torch.allclose(trip[:, :, 0, :], trip[:, :, 1, :], atol=1e-6) and torch.allclose(trip[:, :, 0, :], trip[:, :, 2, :], atol=1e-6)):
                            raise ValueError("Actions are not repeated in triplets; dataset action alignment is likely broken")
                        # Take one action per triplet: [B, T*3, 2] -> [B, T, 2]
                        dataset_actions = trip[:, :, 0, :].to(device=self.device)
                        # Ensure we have exactly num_frames
                        if dataset_actions.shape[1] > num_frames:
                            dataset_actions = dataset_actions[:, :num_frames, :]
                        elif dataset_actions.shape[1] < num_frames:
                            # Pad with last frame
                            last_frame = dataset_actions[:, -1:, :]
                            padding = last_frame.repeat(1, num_frames - dataset_actions.shape[1], 1)
                            dataset_actions = torch.cat([dataset_actions, padding], dim=1)
                        actions = dataset_actions
                        generate_and_save_video(sample, actions, "dataset", prompt_idx, dir_idx=None)
                    else:
                        print(f"[Warning] Sample {prompt_idx} has no actions, skipping dataset action video")

        if self.one_logger is not None:
            self.one_logger.on_validation_batch_end()

        torch.cuda.empty_cache()
        import gc
        gc.collect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replicate training-time visualization using the exact _visualize() function.")
    parser.add_argument("--config-path", type=Path, required=True, help="Path to the YAML config used for the run.")
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Path to the checkpoint file to load.")
    parser.add_argument("--step", type=int, default=None, help="Step number for visualization (default: extract from checkpoint).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for visualization (default: <checkpoint_dir>/../vis).")
    parser.add_argument("--device", default="cuda:0", help="Computation device (default: cuda:0).")
    parser.add_argument("--dataset-actions-only", action="store_true", help="Only use dataset actions, skip dummy actions (default: use both).")
    parser.add_argument("--prompt-indices", type=int, nargs="+", default=None, help="List of prompt indices to process (e.g., '0 1' for first two prompts). Default: all prompts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Load config
    default_cfg_path = REPO_ROOT / 'configs' / 'default_config.yaml'
    if default_cfg_path.is_file():
        base_cfg = OmegaConf.load(default_cfg_path)
    else:
        base_cfg = OmegaConf.create({})
    user_cfg = OmegaConf.load(args.config_path)
    config = OmegaConf.merge(base_cfg, user_cfg)

    # Force this visualization script to use a single diffusion pass instead of the
    # usual multi-step denoising schedule. We keep one timestep from the configured
    # list (by default the last / smallest timestep).
    # if hasattr(config, "denoising_step_list") and config.denoising_step_list:
    #     try:
    #         # Convert to a plain Python list in case it's an OmegaConf ListConfig
    #         _steps = list(config.denoising_step_list)
    #         # Keep just one step (use the last by default)
    #         config.denoising_step_list = _steps[:-1]
    #         print(f"[true_visualise] Forcing single diffusion pass with denoising_step_list={config.denoising_step_list}")
    #     except Exception as e:
    #         print(f"[true_visualise] Warning: failed to clamp denoising_step_list to single step: {e}")
    
    # Set device - when using SLURM with --gpus-per-task=1, CUDA_VISIBLE_DEVICES
    # is set to expose only one GPU per task, which is always cuda:0 from the task's perspective
    # So we should use cuda:0 when running under SLURM, regardless of SLURM_PROCID
    if 'SLURM_PROCID' in os.environ or 'SLURM_JOB_ID' in os.environ:
        # Under SLURM with --gpus-per-task=1, each task sees only one GPU as cuda:0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    elif args.device.startswith("cuda:"):
        device_id = int(args.device.split(":")[1])
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cuda":
        torch.cuda.set_device(device)
    
    # If prompt_indices not provided but SLURM_PROCID is set, calculate from SLURM_PROCID
    # This allows automatic distribution across GPUs when using srun
    prompt_indices = args.prompt_indices
    if prompt_indices is None and 'SLURM_PROCID' in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        # Each process handles 2 prompts
        prompt_indices = [proc_id * 2, proc_id * 2 + 1]
        print(f"[Rank {proc_id}] Auto-assigned prompt indices: {prompt_indices}")
    
    # Initialize trainer (which sets up model, loads checkpoint, and prepares visualization)
    trainer = VisualizationTrainer(
        config, 
        checkpoint_path=args.checkpoint_path, 
        step=args.step or 0, 
        dataset_actions_only=getattr(args, 'dataset_actions_only', False),
        prompt_indices=prompt_indices
    )
    
    # Override output path if specified
    if args.output_dir is not None:
        trainer.vis_output_dir = str(args.output_dir)
        os.makedirs(trainer.vis_output_dir, exist_ok=True)
    
    # Override step if specified
    if args.step is not None:
        trainer.step = args.step
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f"[Rank {rank}] Generating visualization for step {trainer.step}")
    print(f"[Rank {rank}] Output directory: {trainer.vis_output_dir}")
    print(f"[Rank {rank}] Dataset actions only: {trainer.dataset_actions_only} (default: False, generates both dummy and dataset actions)")
    print(f"[Rank {rank}] Prompt indices: {trainer.prompt_indices if trainer.prompt_indices else 'all'}")
    
    # Call the exact _visualize function
    trainer._visualize()
    
    print(f"[Rank {rank}] Visualization complete!")


if __name__ == "__main__":
    main()

