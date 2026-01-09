# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

####################NOTE####################
#NOTE: flow_discriminator is lazily initialized. The trainer must either (a) call _get_flow_discriminator(...) once before building optimizers, 
# or (b) add discriminator params to the critic optimizer after first init. Otherwise the discriminator won’t be trained.
####################NOTE####################

import torch.nn.functional as F
from typing import Optional, Tuple
import torch
import torch.nn as nn
import time
from contextlib import contextmanager

from model.base import SelfForcingModel
from utils.memory import log_gpu_memory
import torch.distributed as dist
from utils.debug_option import DEBUG, LOG_GPU_MEMORY
from utils.wan_wrapper import WanDiffusionWrapper
import math


def _safe_gn_groups(c: int, max_groups: int = 8) -> int:
    """
    Helper to find safe GroupNorm group count that divides channels.
    Avoids runtime crashes from GroupNorm divisibility issues.
    """
    for g in range(min(max_groups, c), 0, -1):
        if c % g == 0:
            return g
    return 1


def _sinusoidal_embedding(timesteps: torch.Tensor, dim: int = 64) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings (no external deps).
    Input: timesteps [N] (flattened)
    Output: embeddings [N, dim]
    """
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=timesteps.dtype) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2 == 1:  # Zero pad if dim is odd
        emb = F.pad(emb, (0, 1))
    return emb


class FlowDiscriminator(nn.Module):
    """
    Timestep-aware and action-conditioned FlowGAN discriminator.
    Takes (x, t, a) where x = cat([x_t, flow], dim=channel), t is timestep, and a is action.
    Uses sinusoidal timestep embeddings and FiLM-style fusion.
    """
    def __init__(self, in_channels: int, hidden_dim: int, time_embed_dim: int = 64, 
                 action_dim: int = 0, device=None, dtype=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        self.action_dim = action_dim
        
        # Conv backbone
        self.conv1 = nn.Conv2d(in_channels, hidden_dim * 2, kernel_size=3, padding=1)
        # Use safe GroupNorm to avoid divisibility issues
        self.gn1 = nn.GroupNorm(_safe_gn_groups(hidden_dim * 2), hidden_dim * 2)
        self.conv2 = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(_safe_gn_groups(hidden_dim), hidden_dim)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Timestep embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Action embedding MLP (only if action_dim > 0)
        if action_dim > 0:
            self.action_mlp = nn.Sequential(
                nn.Linear(action_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        else:
            self.action_mlp = None
        
        # Final classifier
        self.final = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 1)
        )
        
        if device is not None:
            self.to(device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, a: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        Input:
            - x: [N, 2C, H, W] where x = cat([x_t, flow], dim=channel)
            - t: [N] (flattened timesteps)
            - a: [N, action_dim] (optional, action vectors)
        Output:
            - logits: [N, 1]
        """
        # Conv backbone
        h = F.silu(self.gn1(self.conv1(x)))
        h = F.silu(self.gn2(self.conv2(h)))
        h = self.pool(h).flatten(1)  # [N, hidden_dim]
        
        # Timestep embedding and fusion (FiLM-style additive)
        # Fix dtype safety: _sinusoidal_embedding produces float32, but module may be bf16/fp16
        # Cast to match module's compute dtype to avoid dtype mismatch or unnecessary casts
        t_emb = _sinusoidal_embedding(t, dim=self.time_embed_dim)  # [N, time_embed_dim]
        t_emb = t_emb.to(dtype=h.dtype)  # Match module's compute dtype
        t_proj = self.time_mlp(t_emb)  # [N, hidden_dim]
        h = h + t_proj  # Additive fusion
        
        # Action embedding and fusion (additive, similar to timestep)
        if a is not None and self.action_mlp is not None:
            a_emb = a.to(dtype=h.dtype)  # [N, action_dim]
            a_proj = self.action_mlp(a_emb)  # [N, hidden_dim]
            h = h + a_proj  # Additive fusion
        
        # Final classification
        logits = self.final(h)  # [N, 1]
        return logits


class DMD2FlowSquared(SelfForcingModel):
    """
    DMD2 with FlowGAN: discriminator classifies flows at a given timestep, not final clean targets.
    
    FlowGAN modifications:
    - Discriminator input: cat([x_t, flow], dim=channel) where x_t is noised latent and flow is target/predicted flow
    - Real examples: flow_target computed using same expression as flow matching loss
    - Fake examples: flow_pred from model (with .detach() for discriminator training)
    - Losses: softplus(-D(x_t, t, flow_target)) for real, softplus(D(x_t, t, flow_pred)) for fake
    - Uses same timestep t as flow matching loss
    """
    def __init__(self, args, device):
        """
        Initialize the DMD (Distribution Matching Distillation) module.
        This class is self-contained and compute generator and fake score losses
        in the forward pass.
        """
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.same_step_across_blocks = getattr(args, "same_step_across_blocks", True)
        self.min_num_training_frames = getattr(args, "min_num_training_frames", 21)
        self.num_training_frames = getattr(args, "num_training_frames", 21)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

        # this will be init later with fsdp-wrapped modules
        self.inference_pipeline: ActionSelfForcingTrainingPipeline = None

        # Step 2: Initialize all dmd hyperparameters
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        if hasattr(args, "real_guidance_scale"):
            self.real_guidance_scale = args.real_guidance_scale
            self.fake_guidance_scale = args.fake_guidance_scale
        else:
            self.real_guidance_scale = args.guidance_scale
            self.fake_guidance_scale = 0.0
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.ts_schedule = getattr(args, "ts_schedule", True)
        self.ts_schedule_max = getattr(args, "ts_schedule_max", False)
        self.min_score_timestep = getattr(args, "min_score_timestep", 0)

        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None

        # DMD2-specific configuration
        self.gan_loss_weight = getattr(args, "gan_loss_weight", 0.0)
        self.guidance_cls_loss_weight = getattr(args, "guidance_cls_loss_weight", self.gan_loss_weight)
        self.concat_time_embeddings = getattr(args, "concat_time_embeddings", False)
        self.generator_real_flow_loss_weight = getattr(args, "generator_real_flow_loss_weight", 0.0)
        
        # FlowGAN timestep gating: only train FlowGAN on frames with timestep <= max_timestep
        # Default to 600 if not specified to ensure FlowGAN trains
        self.diffusion_gan_max_timestep = getattr(args, "diffusion_gan_max_timestep", 600)
        self.diffusion_gan_min_timestep = getattr(args, "diffusion_gan_min_timestep", 0)

        config_action_dim = getattr(args, "raw_action_dim", 2)
        if config_action_dim is not None and config_action_dim != 2:
            raise ValueError(
                f"FlowGAN requires action_dim=2, but config provides action_dim={config_action_dim}. "
                f"Either remove action_dim from config or set it to 2."
            )
        # Hardcode action_dim to 2 for FlowGAN; used by discriminator and action checks.
        self.action_dim = 2
        
        # FlowGAN: Create dedicated discriminator head that accepts [x_t, flow] without going through denoiser
        # This avoids breaking the denoiser's input channel assumptions
        # The discriminator is a lightweight ConvNet that processes concatenated [x_t, flow] directly
        self._init_flow_discriminator(args)
    
    def get_generator_parameters(self):
        """
        Get generator parameters, explicitly excluding flow_discriminator.
        
        Safety: This ensures flow_discriminator parameters are NEVER included in the generator optimizer.
        The optimizer setup in trainer/distillation.py uses self.model.generator.parameters(), which
        should be safe (flow_discriminator is not under generator), but this method provides an
        explicit exclusion for defensive programming.
        """
        # Get generator parameters (flow_discriminator lives outside generator, so nothing to filter out)
        return list(self.generator.parameters())
    
    def get_critic_parameters(self):
        """
        Get critic parameters, including fake_score and flow_discriminator.
        
        Safety: This ensures flow_discriminator parameters are included in the critic optimizer,
        not the generator optimizer.
        """
        # Get fake_score parameters
        critic_params = list(self.fake_score.parameters())
        
        # Add flow_discriminator parameters if it exists (lazily initialized, may not exist yet)
        if self.flow_discriminator is not None:
            critic_params.extend(self.flow_discriminator.parameters())
        
        return critic_params

    def _init_flow_discriminator(self, args):
        """
        Initialize a lightweight discriminator head for FlowGAN.
        This discriminator accepts [x_t, flow] concatenated along channels, timestep embedding, and actions,
        without going through the denoiser's patch embed to avoid channel mismatch.
        """
        latent_shape = getattr(args, "image_or_video_shape", None)
        if latent_shape is None or len(latent_shape) < 3:
            raise ValueError(
                "image_or_video_shape with at least 3 entries (B, F, C, ...) "
                "is required to initialize the FlowGAN discriminator."
            )
        latent_channels = int(latent_shape[2])
        disc_in_channels = latent_channels * 2
        hidden_dim = max(64, disc_in_channels // 4)
        time_embed_dim = 64

        self._flow_disc_in_channels = disc_in_channels
        self.flow_discriminator = FlowDiscriminator(
            in_channels=disc_in_channels,
            hidden_dim=hidden_dim,
            time_embed_dim=time_embed_dim,
            action_dim=self.action_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self.flow_discriminator.requires_grad_(True)

    def _get_flow_discriminator(self, x_t: torch.Tensor, flow: torch.Tensor) -> nn.Module:
        """
        Fetch the flow discriminator and validate channel consistency.
        """
        # Concatenate along channel dimension: [B, F, 2*C, H, W]
        disc_in_channels = x_t.shape[2] + flow.shape[2]  # 2 * original channels
        
        if self.flow_discriminator is None:
            raise RuntimeError("Flow discriminator has not been initialized.")

        # Discriminator already exists: assert channel count matches to prevent silent mismatch
        assert self._flow_disc_in_channels == disc_in_channels, (
            f"FlowGAN discriminator channel mismatch: expected {self._flow_disc_in_channels} "
            f"(from configuration), got {disc_in_channels}. This indicates inconsistent "
            f"input shapes (x_t channels: {x_t.shape[2]}, flow channels: {flow.shape[2]}). "
            f"Discriminator must be configured with consistent latent channel counts."
        )
        
        return self.flow_discriminator

    def _compute_kl_grad(
        self, noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict, unconditional_dict: dict,
        normalization: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the KL grad (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - noisy_image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - estimated_clean_image_or_video: a tensor with shape [B, F, C, H, W] representing the estimated clean image or video.
            - timestep: a tensor with shape [B, F] containing the randomly generated timestep.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - normalization: a boolean indicating whether to normalize the gradient.
        Output:
            - kl_grad: a tensor representing the KL grad.
            - kl_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Compute the fake score
        _, pred_fake_image_cond = self.fake_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        if self.fake_guidance_scale != 0.0:
            _, pred_fake_image_uncond = self.fake_score(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=unconditional_dict,
                timestep=timestep
            )
            pred_fake_image = pred_fake_image_cond + (
                pred_fake_image_cond - pred_fake_image_uncond
            ) * self.fake_guidance_scale
        else:
            pred_fake_image = pred_fake_image_cond

        # Step 2: Compute the real score
        # We compute the conditional and unconditional prediction
        # and add them together to achieve cfg (https://arxiv.org/abs/2207.12598)
        _, pred_real_image_cond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        _, pred_real_image_uncond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=unconditional_dict,
            timestep=timestep
        )

        pred_real_image = pred_real_image_cond + (
            pred_real_image_cond - pred_real_image_uncond
        ) * self.real_guidance_scale

        # Step 3: Compute the DMD2 gradient (difference between p_real and p_fake as in DMD2).
        p_real = estimated_clean_image_or_video - pred_real_image
        p_fake = estimated_clean_image_or_video - pred_fake_image
        grad = p_real - p_fake

        if normalization:
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / normalizer
        grad = torch.nan_to_num(grad)

        return grad, {
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.detach()
        }
    
    def _action_batch_stats(self, actions: Optional[torch.Tensor], prefix: str) -> dict:
        """Lightweight scalars describing the input action distribution for logging."""
        if actions is None:
            return {}
        stats: dict = {}
        actions_f = actions.to(dtype=torch.float32)
        for i in range(min(actions_f.shape[-1], 3)):
            stats[f"{prefix}_dim{i}_mean"] = actions_f[..., i].mean().detach()
            stats[f"{prefix}_dim{i}_std"] = actions_f[..., i].std(unbiased=False).detach()
        mag = torch.linalg.norm(actions_f, dim=-1)
        stats[f"{prefix}_mag_mean"] = mag.mean().detach()
        stats[f"{prefix}_mag_std"] = mag.std(unbiased=False).detach()
        return stats

    def _select_action_window(self, actions: Optional[torch.Tensor], num_frames: int) -> Optional[torch.Tensor]:
        """
        Slice or validate actions to match the requested number of frames.
        Accepts [B, 2] or [B, F, 2]; returns matching shape (broadcasting for [B, 2]).
        """
        if actions is None:
            return None
        if actions.dim() == 2:
            if actions.shape[-1] != self.action_dim:
                raise ValueError(
                    f"Actions last dimension must be {self.action_dim}, got {actions.shape[-1]}."
                )
            return actions
        if actions.dim() == 3:
            if actions.shape[-1] != self.action_dim:
                raise ValueError(
                    f"Actions last dimension must be {self.action_dim}, got {actions.shape[-1]}."
                )
            if actions.shape[1] < num_frames:
                raise ValueError(
                    f"Actions provide {actions.shape[1]} frames but {num_frames} required."
                )
            return actions[:, :num_frames]
        raise ValueError(f"Unsupported actions shape: {actions.shape}")

    @contextmanager
    def _freeze_fake_score_params(self):
        """Temporarily freeze fake_score parameters so gradients do not update them."""
        if not hasattr(self.fake_score, "parameters"):
            yield
            return
        requires_grad = []
        for p in self.fake_score.parameters():
            requires_grad.append(p.requires_grad)
            p.requires_grad_(False)
        try:
            yield
        finally:
            for p, rg in zip(self.fake_score.parameters(), requires_grad):
                p.requires_grad_(rg)

    @contextmanager
    def _freeze_flow_discriminator_params(self):
        """Temporarily freeze flow discriminator parameters so gradients do not update them."""
        if self.flow_discriminator is None:
            yield
            return
        requires_grad = []
        for p in self.flow_discriminator.parameters():
            requires_grad.append(p.requires_grad)
            p.requires_grad_(False)
        try:
            yield
        finally:
            for p, rg in zip(self.flow_discriminator.parameters(), requires_grad):
                p.requires_grad_(rg)

    def _build_classifier_inputs(self, latents: torch.Tensor, conditional_dict: dict, reuse_noise: Optional[torch.Tensor] = None):
        batch_size, num_frames = latents.shape[:2]
        device = latents.device
        latents = latents.float()

        timesteps = torch.zeros((batch_size,), device=device, dtype=torch.long)

        timestep_full = timesteps[:, None].repeat(1, num_frames)

        noisy_latents = latents

        return noisy_latents, timestep_full

    def _classifier_logits_flow(self, x_t: torch.Tensor, flow: torch.Tensor, timestep: torch.Tensor, actions: Optional[torch.Tensor]) -> torch.Tensor:
        """
        FlowGAN: discriminator sees (x_t, flow, t, action) with explicit timestep and action conditioning.
        This uses a dedicated discriminator head that processes [x_t, flow] directly,
        avoiding the denoiser's patch embed to prevent channel mismatch.
        
        FlowGAN matches flow distribution at timestep t, not final clean targets.
        
        Input:
            - x_t: noised latent at timestep t, shape [B, F, C, H, W]
            - flow: flow tensor (target for real, predicted for fake), shape [B, F, C, H, W]
            - timestep: timestep tensor, shape [B, F]
            - actions: action tensor, shape [B, action_dim] or [B, F, action_dim]
        Output:
            - logits: discriminator logits, shape [B, F]
        """
        batch_size, num_frames = x_t.shape[:2]
        
        # Validate actions whenever the flow discriminator is used for training
        # The flow discriminator is used when either gan_loss_weight > 0 (generator GAN loss) or 
        # guidance_cls_loss_weight > 0 (critic discriminator loss)
        flow_disc_enabled = (self.gan_loss_weight > 0.0) or (self.guidance_cls_loss_weight > 0.0)
        if flow_disc_enabled and self.action_dim == 2:
            if actions is None:
                raise ValueError(
                    f"FlowGAN discriminator requires actions (action_dim=2) when used for training losses "
                    f"(gan_loss_weight={self.gan_loss_weight}, guidance_cls_loss_weight={self.guidance_cls_loss_weight}), "
                    f"but actions=None was provided."
                )
            # Validate action_dim == 2
            if actions.dim() == 2:
                action_dim_actual = actions.shape[-1]
            elif actions.dim() == 3:
                action_dim_actual = actions.shape[-1]
            else:
                raise ValueError(
                    f"Actions must have shape [B, 2] or [B, F, 2], got {actions.shape}"
                )
            if action_dim_actual != 2:
                raise ValueError(
                    f"Action dimension mismatch: actions.shape[-1]={action_dim_actual} but FlowGAN requires action_dim=2. "
                    f"Actions must have shape [B, 2] or [B, F, 2]."
                )
        
        # FlowGAN: concatenate x_t and flow along channel dimension
        # disc_in = cat([x_t, flow], dim=channel) -> [B, F, 2*C, H, W]
        disc_in = torch.cat([x_t, flow], dim=2)  # dim=2 is channel dimension for [B, F, C, H, W]
        
        # Get or create discriminator (action_dim comes from config, not inferred)
        discriminator = self._get_flow_discriminator(x_t, flow)
        
        # Reshape to process each frame independently: [B, F, 2*C, H, W] -> [B*F, 2*C, H, W]
        disc_in_2d = disc_in.flatten(0, 1)  # [B*F, 2*C, H, W]
        
        # Flatten timestep: [B, F] -> [B*F]
        t_flat = timestep.flatten(0, 1)  # [B*F]
        
        # Prepare actions: flatten for discriminator forward (actions already validated/aligned by caller)
        a_flat = None
        if actions is not None:
            if actions.dim() == 2:
                # [B, 2] -> expand to [B, F, 2] then flatten
                a_flat = actions[:, None, :].expand(-1, num_frames, -1).flatten(0, 1)  # [B*F, 2]
            elif actions.dim() == 3:
                # [B, F, 2] -> flatten
                a_flat = actions.flatten(0, 1)  # [B*F, 2]
            else:
                raise ValueError(
                    f"Actions must have shape [B, 2] or [B, F, 2], got {actions.shape}"
                )
        
        # Apply timestep-aware and action-conditioned discriminator
        # Ensure inputs match discriminator's dtype to avoid dtype mismatch errors
        # Note: t_flat should remain as long/int for timestep embedding, only cast disc_in_2d and a_flat
        disc_dtype = next(discriminator.parameters()).dtype
        disc_in_2d = disc_in_2d.to(dtype=disc_dtype)
        # t_flat stays as long/int - the embedding function will handle dtype conversion
        if a_flat is not None:
            a_flat = a_flat.to(dtype=disc_dtype)
        logits_flat = discriminator(disc_in_2d, t_flat, a_flat)  # [B*F, 1]
        logits = logits_flat.view(batch_size, num_frames)  # [B, F]
        
        return logits

    def _classifier_logits(self, latents: torch.Tensor, conditional_dict: dict) -> torch.Tensor:
        if not hasattr(self.fake_score, "adding_cls_branch"):
            raise RuntimeError("fake_score does not support classification branch required for DMD2 GAN loss.")

        noisy_latents, timestep_full = self._build_classifier_inputs(latents, conditional_dict)
        outputs = self.fake_score(
            noisy_image_or_video=noisy_latents,
            conditional_dict=conditional_dict,
            timestep=timestep_full,
            classify_mode=True,
            concat_time_embeddings=self.concat_time_embeddings,
        )
        logits = outputs[-1]
        return logits.squeeze(-1)

    def compute_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0,
        actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the DMD loss (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - gradient_mask: a boolean tensor with the same shape as image_or_video indicating which pixels to compute loss .
        Output:
            - dmd_loss: a scalar tensor representing the DMD loss.
            - dmd_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        original_latent = image_or_video

        batch_size, num_frame = image_or_video.shape[:2]
        cond_with_action = conditional_dict
        uncond_with_action = unconditional_dict
        if self._action_patch_enabled:
            cond_with_action = self._with_action_conditioning(
                conditional_dict,
                actions,
                num_frame,
                image_or_video.device,
                image_or_video.dtype,
                target_num_frames=1,
            )
            uncond_with_action = self._with_action_conditioning(
                unconditional_dict,
                actions,
                num_frame,
                image_or_video.device,
                image_or_video.dtype,
                target_num_frames=1,
            )

        with torch.no_grad():
            # Step 1: Randomly sample timestep based on the given schedule and corresponding noise
            min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
            max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
            timestep = self._get_timestep(
                min_timestep,
                max_timestep,
                batch_size,
                num_frame,
                self.num_frame_per_block,
                uniform_timestep=True
            )

            # TODO:should we change it to `timestep = self.scheduler.timesteps[timestep]`?
            if self.timestep_shift > 1:
                timestep = self.timestep_shift * \
                    (timestep / 1000) / \
                    (1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000
            timestep = timestep.clamp(self.min_step, self.max_step)

            noise = torch.randn_like(image_or_video)
            noisy_latent = self.scheduler.add_noise(
                image_or_video.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1)
            ).detach().unflatten(0, (batch_size, num_frame))

            # Step 2: Compute the KL grad
            grad, dmd_log_dict = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent,
                estimated_clean_image_or_video=original_latent,
                timestep=timestep,
                conditional_dict=cond_with_action,
                unconditional_dict=uncond_with_action
            )

        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            )[gradient_mask], (original_latent.double() - grad.double()).detach()[gradient_mask], reduction="mean")
        else:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            ), (original_latent.double() - grad.double()).detach(), reduction="mean")
        return dmd_loss, dmd_log_dict

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        real_latents: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Freeze critic params so FSDP does not try updating the wrapped fake_score during generator step.
        self._set_fake_score_trainable(False)
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"Generator loss: Before generator unroll", device=self.device, rank=dist.get_rank())
        # Step 1: Unroll generator to obtain fake videos
        slice_last_frames = getattr(self.args, "slice_last_frames", 21)
        _t_gen_start = time.time()
        if DEBUG and dist.get_rank() == 0:
            print(f"generator_rollout")
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent,
            slice_last_frames=slice_last_frames,
            action_inputs=actions,
        )
        gen_time = time.time() - _t_gen_start
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"Generator loss: After generator unroll", device=self.device, rank=dist.get_rank())
        # Step 2: Compute the DMD loss
        _t_loss_start = time.time()
        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
            actions=actions,
        )
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"Generator loss: After compute_distribution_matching_loss", device=self.device, rank=dist.get_rank())
        loss_time = time.time() - _t_loss_start

        total_loss = dmd_loss
        dmd_log_dict.update({
            "gen_time": gen_time,
            "loss_time": loss_time,
            "generator_dmd_loss": dmd_loss.detach(),
        })
        dmd_log_dict.update(self._action_batch_stats(actions, "actions"))
        
        if self.generator_real_flow_loss_weight > 0.0:
            if real_latents is None:
                raise ValueError("generator_rollout_flow_to_real_weight > 0 requires real_latents")
            
            real_flow_loss = self._run_generator_flow(
                conditional_dict=conditional_dict,
                initial_latent=initial_latent,
                slice_last_frames=slice_last_frames,
                action_inputs=actions,
                real_latents_x0=real_latents,
            )
            weighted_real_flow_loss = real_flow_loss * self.generator_real_flow_loss_weight
            total_loss = total_loss + weighted_real_flow_loss
            
            dmd_log_dict.update({
                "generator_rollout_flow_loss": real_flow_loss.detach(),
                "generator_rollout_flow_loss_weighted": weighted_real_flow_loss.detach(),
            })

        # FlowGAN: generator loss uses flow-space GAN instead of clean-x0 GAN
        # FlowGAN matches flow distribution at timestep t, using generator's flow prediction
        # Discriminator no longer receives 2C through the denoiser; it has its own dedicated head
        if self.gan_loss_weight > 0.0:
            with self._freeze_fake_score_params():
                batch_size, num_frame = pred_image.shape[:2]
                
                # Build timestep gating mask: only train FlowGAN on frames with timestep <= max_timestep
                # diffusion_gan_max_timestep defaults to 600, so gating is always active
                mask_gan = None
                if self.diffusion_gan_max_timestep is not None:
                    # t_exit_full is [B, F] - same shape as mask we need
                    mask_gan = (t_exit_full >= self.diffusion_gan_min_timestep) & (t_exit_full <= self.diffusion_gan_max_timestep)
                    if mask_gan.sum() == 0:
                        # No eligible frames: skip FlowGAN loss for this step
                        gan_loss = torch.tensor(0.0, device=self.device)
                        dmd_log_dict.update({
                            "generator_gan_loss": gan_loss.detach(),
                            "generator_gan_logits": torch.tensor(0.0, device=self.device),
                            "gan_mask_fraction": torch.tensor(0.0, device=self.device),
                            "gan_mask_num": torch.tensor(0, device=self.device, dtype=torch.long),
                            "generator_gan_logits_masked_mean": torch.tensor(0.0, device=self.device),
                        })
                        total_loss = total_loss + gan_loss
                    else:
                        # Force-create discriminator before freeze context to ensure it's frozen on first call
                        _ = self._get_flow_discriminator(xt_exit_full, flow_exit_full)  # Create if needed
                        
                        # Freeze discriminator during generator updates
                        with self._freeze_flow_discriminator_params():
                            # Select eligible frames: flatten [B,F] -> [B*F], select indices where mask is true
                            # This avoids calling discriminator on ineligible frames
                            B, F = t_exit_full.shape
                            flat_indices = torch.arange(B * F, device=t_exit_full.device)
                            eligible_indices = flat_indices[mask_gan.flatten()]
                            
                            # Extract eligible frames
                            xt_eligible = xt_exit_full.flatten(0, 1)[eligible_indices]  # [N_eligible, C, H, W]
                            flow_eligible = flow_exit_full.flatten(0, 1)[eligible_indices]  # [N_eligible, C, H, W]
                            t_eligible = t_exit_full.flatten(0, 1)[eligible_indices]  # [N_eligible]
                            
                            # Prepare actions for eligible frames
                            actions_window = self._select_action_window(actions, num_frame)
                            if actions_window is not None:
                                if actions_window.dim() == 2:
                                    # [B, 2] -> expand to [B, F, 2] then flatten and select
                                    actions_flat = actions_window[:, None, :].expand(-1, F, -1).flatten(0, 1)
                                else:
                                    # [B, F, 2] -> flatten
                                    actions_flat = actions_window.flatten(0, 1)
                                actions_eligible = actions_flat[eligible_indices]  # [N_eligible, 2]
                            else:
                                actions_eligible = None
                            
                            # Reshape for discriminator: add frame dimension back [N_eligible, C, H, W] -> [N_eligible, 1, C, H, W]
                            xt_eligible = xt_eligible.unsqueeze(1)  # [N_eligible, 1, C, H, W]
                            flow_eligible = flow_eligible.unsqueeze(1)  # [N_eligible, 1, C, H, W]
                            t_eligible = t_eligible.unsqueeze(1)  # [N_eligible, 1]
                            
                            # Compute logits only on eligible frames
                            logits_eligible = self._classifier_logits_flow(
                                x_t=xt_eligible,
                                flow=flow_eligible,
                                timestep=t_eligible,
                                actions=actions_eligible
                            )  # [N_eligible, 1]
                            
                            # Compute loss on eligible frames only
                            logits_sel = logits_eligible.flatten()  # [N_eligible]
                            gan_loss = F.softplus(-logits_sel).mean() * self.gan_loss_weight
                            
                            # Logging
                            mask_fraction = mask_gan.float().mean()
                            mask_num = mask_gan.sum()
                            logits_masked_mean = logits_sel.mean()
                            
                            dmd_log_dict.update({
                                "generator_gan_loss": gan_loss.detach(),
                                "generator_gan_logits": logits_sel.detach().mean(),
                                "gan_mask_fraction": mask_fraction.detach(),
                                "gan_mask_num": mask_num.detach(),
                                "generator_gan_logits_masked_mean": logits_masked_mean.detach(),
                            })
                        
                        total_loss = total_loss + gan_loss
                else:
                    # No gating: compute on all frames (original behavior)
                    # Force-create discriminator before freeze context to ensure it's frozen on first call
                    _ = self._get_flow_discriminator(xt_exit_full, flow_exit_full)  # Create if needed
                                
                    # Freeze discriminator during generator updates
                    with self._freeze_flow_discriminator_params():
                        logits_fake = self._classifier_logits_flow(
                            x_t=xt_exit_full,
                            flow=flow_exit_full,
                            timestep=t_exit_full,
                            actions=actions
                        )
                        gan_loss = F.softplus(-logits_fake).mean() * self.gan_loss_weight
                        dmd_log_dict.update({
                            "generator_gan_loss": gan_loss.detach(),
                            "generator_gan_logits": logits_fake.detach().mean(),
                        })
                    
                    total_loss = total_loss + gan_loss

        dmd_log_dict.update({
            "gen_time": gen_time,
            "loss_time": loss_time
        })

        return total_loss, dmd_log_dict

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        real_latents: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and train the critic with generated samples.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - critic_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Re-enable critic params now that the critic optimizer is active.
        self._set_fake_score_trainable(True)
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"Critic loss: Before generator unroll", device=self.device, rank=dist.get_rank())
        slice_last_frames = getattr(self.args, "slice_last_frames", 21)
        # Step 1: Run generator on backward simulated noisy input
        _t_gen_start = time.time()
        with torch.no_grad():
            if DEBUG and dist.get_rank() == 0:
                print(f"critic_rollout")
        generated_image, _, denoised_timestep_from, denoised_timestep_to = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                initial_latent=initial_latent,
                slice_last_frames=slice_last_frames,
                action_inputs=actions,
            )
        if dist.get_rank() == 0 and DEBUG:
            print(f"pred_image: {generated_image.shape}")
        gen_time = time.time() - _t_gen_start
        batch_size, num_frame = generated_image.shape[:2]
        critic_conditional = self._with_action_conditioning(
            conditional_dict,
            actions,
            num_frame,
            generated_image.device,
            generated_image.dtype,
            detach_modulation=True,
            target_num_frames=1,
        )
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"Critic loss: After generator unroll", device=self.device, rank=dist.get_rank())
        _t_loss_start = time.time()

        # Step 2: Compute the fake prediction
        # Use scheduled timestep (same as flow matching loss)
        min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
        max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
        critic_timestep = self._get_timestep(
            min_timestep,
            max_timestep,
            batch_size,
            num_frame,
            self.num_frame_per_block,
            uniform_timestep=True
        )

        if self.timestep_shift > 1:
            critic_timestep = self.timestep_shift * \
                (critic_timestep / 1000) / (1 + (self.timestep_shift - 1) * (critic_timestep / 1000)) * 1000

        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

        critic_noise = torch.randn_like(generated_image)
        noisy_generated_image = self.scheduler.add_noise(
            generated_image.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frame))

        _, pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_generated_image,
            conditional_dict=critic_conditional,
            timestep=critic_timestep
        )

        # Step 3: Compute the denoising loss for the fake critic
        if self.args.denoising_loss_type == "flow":
            flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            )
            pred_fake_noise = None
        else:
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            ).unflatten(0, (batch_size, num_frame))

        denoising_loss = self.denoising_loss_func(
            x=generated_image.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred
        )

        # FlowGAN: classifier/discriminator branch classifies flows at a given timestep
        # FlowGAN matches flow distribution at timestep t, not final clean targets
        cls_loss = torch.tensor(0.0, device=self.device)
        gan_log_dict = {}


        total_loss = denoising_loss + cls_loss
        loss_time = time.time() - _t_loss_start
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"Critic loss: After denoising loss", device=self.device, rank=dist.get_rank())

        critic_log_dict = {
            "critic_timestep": critic_timestep.detach(),
            "gen_time": gen_time,
            "loss_time": loss_time
        }
        critic_log_dict.update(gan_log_dict)

        return total_loss, critic_log_dict
