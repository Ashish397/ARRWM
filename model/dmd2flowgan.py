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
    Timestep-aware FlowGAN discriminator.
    Takes (x, t) where x = cat([x_t, flow], dim=channel) and t is timestep.
    Uses sinusoidal timestep embeddings and FiLM-style fusion.
    """
    def __init__(self, in_channels: int, hidden_dim: int, time_embed_dim: int = 64, 
                 device=None, dtype=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        
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
        
        # Final classifier
        self.final = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 1)
        )
        
        if device is not None:
            self.to(device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Input:
            - x: [N, 2C, H, W] where x = cat([x_t, flow], dim=channel)
            - t: [N] (flattened timesteps)
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
        
        # Final classification
        logits = self.final(h)  # [N, 1]
        return logits


class DMD2(SelfForcingModel):
    """
    DMD2 with FlowGAN: discriminator classifies flows at a given timestep, not final clean targets.
    
    FlowGAN modifications:
    - Discriminator input: cat([x_t, flow], dim=channel) where x_t is noised latent and flow is target/predicted flow
    - Real examples: flow_target computed using same expression as flow matching loss
    - Fake examples: flow_pred from model (with .detach() for discriminator training)
    - Losses: softplus(-D(x_t, t, flow_target)) for real, softplus(D(x_t, t, flow_pred)) for fake
    - Uses same timestep t as flow matching loss (or random-t if diffusion_gan=True)
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
        self.inference_pipeline: SelfForcingTrainingPipeline = None

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
        self.cls_on_clean_image = getattr(args, "cls_on_clean_image", True)
        self.gan_loss_weight = getattr(args, "gan_loss_weight", 1.0)
        self.guidance_cls_loss_weight = getattr(args, "guidance_cls_loss_weight", self.gan_loss_weight)
        self.diffusion_gan = getattr(args, "diffusion_gan", False)
        self.diffusion_gan_max_timestep = getattr(args, "diffusion_gan_max_timestep", self.num_train_timestep)
        self.concat_time_embeddings = getattr(args, "concat_time_embeddings", False)

        # FlowGAN: Create dedicated discriminator head that accepts [x_t, flow] without going through denoiser
        # This avoids breaking the denoiser's input channel assumptions
        # The discriminator is a lightweight ConvNet that processes concatenated [x_t, flow] directly
        self._init_flow_discriminator()
    
    def get_generator_parameters(self):
        """
        Get generator parameters, explicitly excluding flow_discriminator.
        
        Safety: This ensures flow_discriminator parameters are NEVER included in the generator optimizer.
        The optimizer setup in trainer/distillation.py uses self.model.generator.parameters(), which
        should be safe (flow_discriminator is not under generator), but this method provides an
        explicit exclusion for defensive programming.
        """
        # Get generator parameters (flow_discriminator is not under generator, so this is safe)
        gen_params = list(self.generator.parameters())
        
        # Explicitly exclude flow_discriminator if it exists (defensive check)
        if self.flow_discriminator is not None:
            flow_disc_params = set(self.flow_discriminator.parameters())
            gen_params = [p for p in gen_params if p not in flow_disc_params]
        
        return gen_params
    
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

    def _init_flow_discriminator(self):
        """
        Initialize a lightweight discriminator head for FlowGAN.
        This discriminator accepts [x_t, flow] concatenated along channels and timestep embedding,
        without going through the denoiser's patch embed to avoid channel mismatch.
        """
        # Will be initialized lazily on first use to infer input channels
        self.flow_discriminator = None  # Will be initialized lazily on first use
        self._flow_disc_in_channels = None  # Will be set on first forward and must remain constant

    def ensure_flow_discriminator_initialized(self, image_or_video_shape) -> None:
        """
        Ensure the flow discriminator is initialized before optimizer creation.
        """
        if self.flow_discriminator is not None:
            return
        if image_or_video_shape is None or len(image_or_video_shape) < 5:
            raise ValueError(
                f"image_or_video_shape must provide [B, F, C, H, W], got {image_or_video_shape}"
            )
        _, num_frames, channels, height, width = [int(x) for x in image_or_video_shape[:5]]
        dummy_shape = (1, num_frames, channels, height, width)
        with torch.no_grad():
            dummy_xt = torch.zeros(dummy_shape, device=self.device, dtype=self.dtype)
            dummy_flow = torch.zeros_like(dummy_xt)
            self._get_flow_discriminator(dummy_xt, dummy_flow)

    def _get_flow_discriminator(self, x_t: torch.Tensor, flow: torch.Tensor) -> nn.Module:
        """
        Get or create the flow discriminator. Initialize it lazily to infer input channels.
        Returns a FlowDiscriminator module that is timestep-aware.
        
        Safety: Once initialized, the discriminator channel count is locked. If channels change
        across calls (e.g., due to different latent channel counts), this will fail loudly rather
        than silently recreating or mis-shaping the discriminator.
        """
        # Concatenate along channel dimension: [B, F, 2*C, H, W]
        disc_in_channels = x_t.shape[2] + flow.shape[2]  # 2 * original channels
        
        if self.flow_discriminator is None:
            # First initialization: create discriminator and lock channel count
            self._flow_disc_in_channels = disc_in_channels
            hidden_dim = max(64, disc_in_channels // 4)  # At least 64, or 1/4 of input channels
            time_embed_dim = 64  # Dimension for timestep embeddings
            
            self.flow_discriminator = FlowDiscriminator(
                in_channels=disc_in_channels,
                hidden_dim=hidden_dim,
                time_embed_dim=time_embed_dim,
                device=x_t.device,
                dtype=x_t.dtype
            )
            self.flow_discriminator.requires_grad_(True)
        else:
            # Discriminator already exists: assert channel count matches to prevent silent reinit/mismatch
            assert self._flow_disc_in_channels == disc_in_channels, (
                f"FlowGAN discriminator channel mismatch: expected {self._flow_disc_in_channels} "
                f"(from first initialization), got {disc_in_channels}. This indicates inconsistent "
                f"input shapes (x_t channels: {x_t.shape[2]}, flow channels: {flow.shape[2]}). "
                f"Discriminator must not be reinitialized with different channel counts."
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

        if self.diffusion_gan:
            timesteps = torch.randint(
                0,
                self.diffusion_gan_max_timestep,
                (batch_size,),
                device=device,
                dtype=torch.long,
            )
        else:
            timesteps = torch.zeros((batch_size,), device=device, dtype=torch.long)

        timestep_full = timesteps[:, None].repeat(1, num_frames)

        if self.diffusion_gan:
            noise = reuse_noise if reuse_noise is not None else torch.randn_like(latents)
            noisy_latents = self.scheduler.add_noise(
                latents.flatten(0, 1),
                noise.flatten(0, 1),
                timestep_full.flatten(0, 1),
            ).unflatten(0, (batch_size, num_frames))
        else:
            noisy_latents = latents

        return noisy_latents, timestep_full

    def _classifier_logits_flow(self, x_t: torch.Tensor, flow: torch.Tensor, timestep: torch.Tensor, conditional_dict: dict) -> torch.Tensor:
        """
        FlowGAN: discriminator sees (x_t, flow, t) with explicit timestep conditioning.
        This uses a dedicated discriminator head that processes [x_t, flow] directly,
        avoiding the denoiser's patch embed to prevent channel mismatch.
        
        FlowGAN matches flow distribution at timestep t, not final clean targets.
        
        Input:
            - x_t: noised latent at timestep t, shape [B, F, C, H, W]
            - flow: flow tensor (target for real, predicted for fake), shape [B, F, C, H, W]
            - timestep: timestep tensor, shape [B, F]
            - conditional_dict: conditional information dict (unused, kept for compatibility)
        Output:
            - logits: discriminator logits, shape [B, F]
        """
        batch_size, num_frames = x_t.shape[:2]
        
        # FlowGAN: concatenate x_t and flow along channel dimension
        # disc_in = cat([x_t, flow], dim=channel) -> [B, F, 2*C, H, W]
        disc_in = torch.cat([x_t, flow], dim=2)  # dim=2 is channel dimension for [B, F, C, H, W]
        
        # Get or create discriminator
        discriminator = self._get_flow_discriminator(x_t, flow)
        
        # Reshape to process each frame independently: [B, F, 2*C, H, W] -> [B*F, 2*C, H, W]
        disc_in_2d = disc_in.flatten(0, 1)  # [B*F, 2*C, H, W]
        
        # Flatten timestep: [B, F] -> [B*F]
        t_flat = timestep.flatten(0, 1)  # [B*F]
        
        # Apply timestep-aware discriminator
        logits_flat = discriminator(disc_in_2d, t_flat)  # [B*F, 1]
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
        denoised_timestep_to: int = 0
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
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict
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
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"Generator loss: Before generator unroll", device=self.device, rank=dist.get_rank())
        # Step 1: Unroll generator to obtain fake videos
        slice_last_frames = getattr(self.args, "slice_last_frames", 21)
        _t_gen_start = time.time()
        if DEBUG and dist.get_rank() == 0:
            print(f"generator_rollout")
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to, _ = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent,
            slice_last_frames=slice_last_frames
        )
        if dist.get_rank() == 0 and DEBUG:
            print(f"pred_image: {pred_image.shape}")
            if gradient_mask is not None:   
                print(f"gradient_mask: {gradient_mask[0, :, 0, 0, 0]}")
            else:
                print(f"gradient_mask: None")
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
            denoised_timestep_to=denoised_timestep_to
        )
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"Generator loss: After compute_distribution_matching_loss", device=self.device, rank=dist.get_rank())
        try:
            loss_val = dmd_loss.item()
        except Exception:
            loss_val = float('nan')
        loss_time = time.time() - _t_loss_start
        # print(f"[GeneratorLoss] loss {loss_val} | gen_time {gen_time:.3f}s | loss_time {loss_time:.3f}s")

        total_loss = dmd_loss
        # FlowGAN: generator loss uses flow-space GAN instead of clean-x0 GAN
        # FlowGAN matches flow distribution at timestep t, using generator's flow prediction
        # Discriminator no longer receives 2C through the denoiser; it has its own dedicated head
        if self.cls_on_clean_image and self.gan_loss_weight > 0.0:
            with self._freeze_fake_score_params():
                batch_size, num_frame = pred_image.shape[:2]
                
                # Sample timestep: use same strategy as critic_loss
                # If diffusion_gan=True, use random timestep; otherwise use scheduled timestep
                if self.diffusion_gan:
                    # Use random timestep strategy for diffusion_gan
                    gan_timestep = torch.randint(
                        0,
                        self.diffusion_gan_max_timestep,
                        (batch_size,),
                        device=pred_image.device,
                        dtype=torch.long,
                    )[:, None].repeat(1, num_frame)
                else:
                    # Use same timestep sampling as critic_loss
                    min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
                    max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
                    gan_timestep = self._get_timestep(
                        min_timestep,
                        max_timestep,
                        batch_size,
                        num_frame,
                        self.num_frame_per_block,
                        uniform_timestep=True
                    )
                
                if self.timestep_shift > 1:
                    gan_timestep = self.timestep_shift * \
                        (gan_timestep / 1000) / (1 + (self.timestep_shift - 1) * (gan_timestep / 1000)) * 1000
                gan_timestep = gan_timestep.clamp(self.min_step, self.max_step)
                
                # Compute x_t (noised latent at timestep t)
                gan_noise = torch.randn_like(pred_image)
                gan_x_t = self.scheduler.add_noise(
                    pred_image.flatten(0, 1),
                    gan_noise.flatten(0, 1),
                    gan_timestep.flatten(0, 1)
                ).unflatten(0, (batch_size, num_frame))
                
                # FlowGAN: compute flow_pred using the generator/student model (self.generator)
                # This is the model being optimized by DMD, not fake_score
                # In this repo, WanDiffusionWrapper returns (_, pred_x0) like real_score/fake_score
                out = self.generator(
                    noisy_image_or_video=gan_x_t,
                    conditional_dict=conditional_dict,
                    timestep=gan_timestep
                )
                
                # Extract pred_x0: handle tuple/list or single value
                if isinstance(out, (tuple, list)) and len(out) >= 2:
                    pred_x0_gen = out[1]  # pred_x0 is second element
                else:
                    pred_x0_gen = out
                
                # Convert pred_x0 to flow_pred using existing utility
                from utils.wan_wrapper import WanDiffusionWrapper
                flow_pred_gen = WanDiffusionWrapper._convert_x0_to_flow_pred(
                    scheduler=self.scheduler,
                    x0_pred=pred_x0_gen.flatten(0, 1),
                    xt=gan_x_t.flatten(0, 1),
                    timestep=gan_timestep.flatten(0, 1)
                )
                flow_pred_gen = flow_pred_gen.unflatten(0, (batch_size, num_frame))
                
                # FlowGAN: gan_loss = softplus(-D(x_t, t, flow_pred)).mean() * gan_loss_weight
                # Force-create discriminator before freeze context to ensure it's frozen on first call
                # _freeze_flow_discriminator_params is a no-op if discriminator is None, and
                # _classifier_logits_flow lazily creates it, so we must create it first
                _ = self._get_flow_discriminator(gan_x_t, flow_pred_gen)  # Create if needed
                
                # Freeze discriminator during generator updates
                with self._freeze_flow_discriminator_params():
                    logits_fake = self._classifier_logits_flow(
                        x_t=gan_x_t,
                        flow=flow_pred_gen,
                        timestep=gan_timestep,
                        conditional_dict=conditional_dict
                    )
                    gan_loss = F.softplus(-logits_fake).mean() * self.gan_loss_weight
                
                total_loss = total_loss + gan_loss
                dmd_log_dict.update({
                    "generator_gan_loss": gan_loss.detach(),
                    "generator_gan_logits": logits_fake.detach().mean(),
                })

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
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"Critic loss: Before generator unroll", device=self.device, rank=dist.get_rank())
        slice_last_frames = getattr(self.args, "slice_last_frames", 21)
        # Step 1: Run generator on backward simulated noisy input
        _t_gen_start = time.time()
        with torch.no_grad():
            if DEBUG and dist.get_rank() == 0:
                print(f"critic_rollout")
        generated_image, _, denoised_timestep_from, denoised_timestep_to, _ = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                initial_latent=initial_latent,
                slice_last_frames=slice_last_frames
            )
        if dist.get_rank() == 0 and DEBUG:
            print(f"pred_image: {generated_image.shape}")
        gen_time = time.time() - _t_gen_start
        batch_size, num_frame = generated_image.shape[:2]
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"Critic loss: After generator unroll", device=self.device, rank=dist.get_rank())
        _t_loss_start = time.time()

        # Step 2: Compute the fake prediction
        # Sample timestep: if diffusion_gan=True, use random timestep; otherwise use scheduled timestep
        if self.diffusion_gan:
            # Use random timestep strategy for diffusion_gan
            critic_timestep = torch.randint(
                0,
                self.diffusion_gan_max_timestep,
                (batch_size,),
                device=generated_image.device,
                dtype=torch.long,
            )[:, None].repeat(1, num_frame)
        else:
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
            conditional_dict=conditional_dict,
            timestep=critic_timestep
        )

        # Step 3: Compute the denoising loss for the fake critic
        if self.args.denoising_loss_type == "flow":
            from utils.wan_wrapper import WanDiffusionWrapper
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
        if (self.cls_on_clean_image and self.guidance_cls_loss_weight > 0.0 and real_latents is not None):
            # If diffusion_gan=True, use a separately sampled timestep for classifier
            # Otherwise, use the same critic_timestep as flow matching loss
            if self.diffusion_gan:
                # Sample separate timestep for classifier (same for real and fake)
                cls_timestep = torch.randint(
                    0,
                    self.diffusion_gan_max_timestep,
                    (batch_size,),
                    device=generated_image.device,
                    dtype=torch.long,
                )[:, None].repeat(1, num_frame)
                if self.timestep_shift > 1:
                    cls_timestep = self.timestep_shift * \
                        (cls_timestep / 1000) / (1 + (self.timestep_shift - 1) * (cls_timestep / 1000)) * 1000
                cls_timestep = cls_timestep.clamp(self.min_step, self.max_step)
                
                # For real: recompute inputs at cls_timestep
                real_noise_cls = torch.randn_like(real_latents)
                real_x_t_cls = self.scheduler.add_noise(
                    real_latents.to(self.device, dtype=torch.float32).flatten(0, 1),
                    real_noise_cls.flatten(0, 1),
                    cls_timestep.flatten(0, 1)
                ).unflatten(0, (batch_size, num_frame))
                
                # Compute flow_target_cls using the same target definition as flow-matching loss
                from utils.wan_wrapper import WanDiffusionWrapper
                flow_target_cls = WanDiffusionWrapper._convert_x0_to_flow_pred(
                    scheduler=self.scheduler,
                    x0_pred=real_latents.to(self.device, dtype=torch.float32).flatten(0, 1),
                    xt=real_x_t_cls.flatten(0, 1),
                    timestep=cls_timestep.flatten(0, 1)
                )
                flow_target_cls = flow_target_cls.unflatten(0, (batch_size, num_frame))
                
                # For fake: recompute inputs at cls_timestep
                fake_noise_cls = torch.randn_like(generated_image)
                fake_x_t_cls = self.scheduler.add_noise(
                    generated_image.flatten(0, 1),
                    fake_noise_cls.flatten(0, 1),
                    cls_timestep.flatten(0, 1)
                ).unflatten(0, (batch_size, num_frame))
                
                # Run fake_score to get pred_x0 at cls_timestep
                _, pred_x0_fake_cls = self.fake_score(
                    noisy_image_or_video=fake_x_t_cls,
                    conditional_dict=conditional_dict,
                    timestep=cls_timestep
                )
                
                # Convert pred_x0_fake_cls to flow_pred_fake_cls
                flow_pred_fake_cls = WanDiffusionWrapper._convert_x0_to_flow_pred(
                    scheduler=self.scheduler,
                    x0_pred=pred_x0_fake_cls.flatten(0, 1),
                    xt=fake_x_t_cls.flatten(0, 1),
                    timestep=cls_timestep.flatten(0, 1)
                )
                flow_pred_fake_cls = flow_pred_fake_cls.unflatten(0, (batch_size, num_frame))
                
                # Use cls_timestep for discriminator
                disc_timestep = cls_timestep
                real_x_t_disc = real_x_t_cls
                flow_target_disc = flow_target_cls
                fake_x_t_disc = fake_x_t_cls
                flow_pred_fake_disc = flow_pred_fake_cls
            else:
                # Use same critic_timestep as flow matching loss
                # For real examples: compute flow_target using the exact same expression as flow matching loss
                real_noise = torch.randn_like(real_latents)
                real_x_t = self.scheduler.add_noise(
                    real_latents.to(self.device, dtype=torch.float32).flatten(0, 1),
                    real_noise.flatten(0, 1),
                    critic_timestep.flatten(0, 1)
                ).unflatten(0, (batch_size, num_frame))
                
                # Compute flow_target using the same conversion as flow_pred to ensure exact match
                from utils.wan_wrapper import WanDiffusionWrapper
                flow_target = WanDiffusionWrapper._convert_x0_to_flow_pred(
                    scheduler=self.scheduler,
                    x0_pred=real_latents.to(self.device, dtype=torch.float32).flatten(0, 1),
                    xt=real_x_t.flatten(0, 1),
                    timestep=critic_timestep.flatten(0, 1)
                )
                flow_target = flow_target.unflatten(0, (batch_size, num_frame))
                
                # For fake examples: use flow_pred (already computed above for denoising loss)
                # flow_pred is already computed in Step 3 above
                flow_pred_unflattened = flow_pred.unflatten(0, (batch_size, num_frame)) if flow_pred is not None else None
                
                # Use critic_timestep for discriminator
                disc_timestep = critic_timestep
                real_x_t_disc = real_x_t
                flow_target_disc = flow_target
                fake_x_t_disc = noisy_generated_image
                flow_pred_fake_disc = flow_pred_unflattened
            
            if flow_pred_fake_disc is not None:
                # FlowGAN discriminator loss:
                # loss_real = softplus(-D(x_t, t, flow_target)).mean()
                # loss_fake = softplus(D(x_t, t, flow_pred.detach())).mean()
                # Total: L_D = loss_real + loss_fake
                real_logits = self._classifier_logits_flow(
                    x_t=real_x_t_disc,
                    flow=flow_target_disc,
                    timestep=disc_timestep,
                    conditional_dict=conditional_dict
                )
                fake_logits = self._classifier_logits_flow(
                    x_t=fake_x_t_disc,
                    flow=flow_pred_fake_disc.detach(),  # .detach() for discriminator training
                    timestep=disc_timestep,
                    conditional_dict=conditional_dict
                )
                loss_real = F.softplus(-real_logits).mean()
                loss_fake = F.softplus(fake_logits).mean()
                cls_loss = (loss_real + loss_fake) * self.guidance_cls_loss_weight
                gan_log_dict = {
                    "critic_gan_loss": cls_loss.detach(),
                    "critic_real_logits": real_logits.detach().mean(),
                    "critic_fake_logits": fake_logits.detach().mean(),
                }

        total_loss = denoising_loss + cls_loss

        try:
            loss_val = total_loss.item()
        except Exception:
            loss_val = float('nan')
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
