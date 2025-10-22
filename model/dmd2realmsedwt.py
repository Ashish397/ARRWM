# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
import torch.nn.functional as F
from typing import Optional, Tuple
import torch
import time
import math
from contextlib import contextmanager

from model.base import SelfForcingModel
from utils.memory import log_gpu_memory
import torch.distributed as dist
from utils.debug_option import DEBUG, LOG_GPU_MEMORY

class DMD2RealMSEDWT(SelfForcingModel):
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
        self.generator_mse_loss_weight = getattr(args, "generator_mse_loss_weight", 0.0)
        self.generator_diffusion_loss_weight = getattr(args, "generator_diffusion_loss_weight", 0.0)
        self.generator_dwt_loss_weight = getattr(args, "generator_dwt_loss_weight", 0.0)
        self.adaptive_d_weight = getattr(args, "adaptive_d_weight", False)
        self.adaptive_d_weight_max = getattr(args, "adaptive_d_weight_max", 1e4)

        # Precompute Haar wavelet kernels for the DWT loss (stored as buffers so they follow device/dtype)
        haar_norm = 1.0 / math.sqrt(2.0)
        low_filter = torch.tensor([haar_norm, haar_norm], dtype=torch.float32)
        high_filter = torch.tensor([haar_norm, -haar_norm], dtype=torch.float32)
        lh_kernel = torch.outer(low_filter, high_filter)
        hl_kernel = torch.outer(high_filter, low_filter)
        hh_kernel = torch.outer(high_filter, high_filter)
        self.register_buffer("dwt_kernel_lh", lh_kernel.view(1, 1, 2, 2), persistent=False)
        self.register_buffer("dwt_kernel_hl", hl_kernel.view(1, 1, 2, 2), persistent=False)
        self.register_buffer("dwt_kernel_hh", hh_kernel.view(1, 1, 2, 2), persistent=False)

        if hasattr(self.fake_score, "adding_cls_branch"):
            try:
                self.fake_score.adding_cls_branch(
                    time_embed_dim=1536 if self.concat_time_embeddings else 0
                )
            except Exception:
                if dist.get_rank() == 0:
                    print("[Warning] Failed to add classification branch to fake_score.")

    def _build_classifier_inputs(self, latents: torch.Tensor, conditional_dict: dict):
        batch_size, num_frames = latents.shape[:2]
        device = latents.device
        latents = latents.float()

        noise = None
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
            noise = torch.randn_like(latents)
            noisy_latents = self.scheduler.add_noise(
                latents.flatten(0, 1),
                noise.flatten(0, 1),
                timestep_full.flatten(0, 1),
            ).unflatten(0, (batch_size, num_frames))
        else:
            noisy_latents = latents

        return noisy_latents, timestep_full, noise

    def _build_classifier_inputs_reuse(self, latents: torch.Tensor, conditional_dict: dict, noise: torch.Tensor, timestep_full: torch.Tensor):
        batch_size, num_frames = latents.shape[:2]
        device = latents.device
        latents = latents.float()

        if self.diffusion_gan:
            noisy_latents = self.scheduler.add_noise(
                latents.flatten(0, 1),
                noise.flatten(0, 1),
                timestep_full.flatten(0, 1),
            ).unflatten(0, (batch_size, num_frames))
        else:
            noisy_latents = latents

        return noisy_latents

    def _classifier_logits(self, latents: torch.Tensor, conditional_dict: dict) -> torch.Tensor:
        if not hasattr(self.fake_score, "adding_cls_branch"):
            raise RuntimeError("fake_score does not support classification branch required for DMD2 GAN loss.")

        noisy_latents, timestep_full, noise = self._build_classifier_inputs(latents, conditional_dict)
        outputs = self.fake_score(
            noisy_image_or_video=noisy_latents,
            conditional_dict=conditional_dict,
            timestep=timestep_full,
            classify_mode=True,
            concat_time_embeddings=self.concat_time_embeddings,
        )
        logits = outputs[-1]
        return logits.squeeze(-1), noise, timestep_full
    
    def _classifier_logits_reuse(self, latents: torch.Tensor, conditional_dict: dict, noise: torch.Tensor, timestep_full: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.fake_score, "adding_cls_branch"):
            raise RuntimeError("fake_score does not support classification branch required for DMD2 GAN loss.")

        noisy_latents = self._build_classifier_inputs_reuse(latents, conditional_dict, noise, timestep_full)
        outputs = self.fake_score(
            noisy_image_or_video=noisy_latents,
            conditional_dict=conditional_dict,
            timestep=timestep_full,
            classify_mode=True,
            concat_time_embeddings=self.concat_time_embeddings,
        )
        logits = outputs[-1]
        return logits.squeeze(-1)

    def _calculate_adaptive_weight(self, reconstruction_loss, gan_loss, target):
        """Compute adaptive discriminator weight using gradient norms."""
        base_weight = torch.tensor(float(self.gan_loss_weight), device=target.device, dtype=target.dtype)
        if (reconstruction_loss is None) or (not self.adaptive_d_weight):
            return base_weight
        recon_val = reconstruction_loss.detach()
        if not torch.isfinite(recon_val).all():
            return base_weight
        if torch.allclose(recon_val, torch.zeros_like(recon_val)):
            return base_weight
        try:
            rec_grads = torch.autograd.grad(
                reconstruction_loss,
                target,
                retain_graph=True,
                allow_unused=True,
            )[0]
            gan_grads = torch.autograd.grad(
                gan_loss,
                target,
                retain_graph=True,
                allow_unused=True,
            )[0]
        except RuntimeError:
            return base_weight
        if rec_grads is None or gan_grads is None:
            return base_weight
        rec_norm = rec_grads.norm()
        gan_norm = gan_grads.norm()
        if (not torch.isfinite(rec_norm)) or (not torch.isfinite(gan_norm)):
            return base_weight
        d_weight = rec_norm / (gan_norm + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, float(self.adaptive_d_weight_max))
        return (d_weight.detach() * base_weight)

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        real_latents: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Train the generator using only the GAN objective (no teacher)."""
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory("Generator loss: Before generator unroll", device=self.device, rank=dist.get_rank())

        slice_last_frames = getattr(self.args, "slice_last_frames", 21)
        _t_gen_start = time.time()
        pred_image, _, _, _ = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent,
            slice_last_frames=slice_last_frames,
        )
        gen_time = time.time() - _t_gen_start
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory("Generator loss: After generator unroll", device=self.device, rank=dist.get_rank())

        _t_loss_start = time.time()
        log_dict = {"gen_time": gen_time}

        total_loss = None
        reconstruction_loss = None
        mse_target = None
        if real_latents is not None:
            mse_target = real_latents
        elif clean_latent is not None:
            mse_target = clean_latent

        pred_for_compare = None
        target_for_compare = None
        recon_target_tensor = None
        if mse_target is not None:
            mse_target = mse_target.to(device=pred_image.device, dtype=pred_image.dtype)
            min_frames = min(pred_image.shape[1], mse_target.shape[1])
            recon_target_tensor = pred_image[:, :min_frames].contiguous()
            target_for_compare = mse_target[:, :min_frames].contiguous()
            pred_for_compare = recon_target_tensor.float()
            target_for_compare = target_for_compare.float()

        if self.generator_mse_loss_weight > 0.0 and pred_for_compare is not None:
            mse_loss = F.mse_loss(
                pred_for_compare,
                target_for_compare
            )
            weighted_mse_loss = mse_loss * self.generator_mse_loss_weight
            total_loss = weighted_mse_loss if total_loss is None else total_loss + weighted_mse_loss
            reconstruction_loss = weighted_mse_loss if reconstruction_loss is None else reconstruction_loss + weighted_mse_loss
            log_dict.update({
                "generator_mse_loss": weighted_mse_loss.detach(),
                "generator_mse_raw": mse_loss.detach(),
            })

        if (
            real_latents is not None
            and target_for_compare is not None
            and self.generator_diffusion_loss_weight > 0.0
        ):
            latents_for_diff = target_for_compare.to(device=pred_image.device, dtype=pred_image.dtype)
            batch_size, num_frames = latents_for_diff.shape[:2]
            diffusion_timestep = self._get_timestep(
                self.min_step,
                self.max_step,
                batch_size,
                num_frames,
                self.num_frame_per_block,
                uniform_timestep=self.generator.uniform_timestep
            )
            if self.timestep_shift > 1:
                diffusion_timestep = self.timestep_shift * \
                    (diffusion_timestep / 1000) / (1 + (self.timestep_shift - 1) * (diffusion_timestep / 1000)) * 1000
            diffusion_timestep = diffusion_timestep.clamp(self.min_step, self.max_step)

            diffusion_noise = torch.randn_like(latents_for_diff)
            noisy_latents = self.scheduler.add_noise(
                latents_for_diff.flatten(0, 1),
                diffusion_noise.flatten(0, 1),
                diffusion_timestep.flatten(0, 1)
            ).unflatten(0, (batch_size, num_frames))

            if self.inference_pipeline is None:
                self._initialize_inference_pipeline()

            # Initialise fresh caches so the generator runs in inference mode
            self.inference_pipeline._initialize_kv_cache(
                batch_size=batch_size,
                dtype=latents_for_diff.dtype,
                device=latents_for_diff.device
            )
            self.inference_pipeline._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=latents_for_diff.dtype,
                device=latents_for_diff.device
            )

            flow_pred, pred_latent = self.generator(
                noisy_image_or_video=noisy_latents,
                conditional_dict=conditional_dict,
                timestep=diffusion_timestep,
                kv_cache=self.inference_pipeline.kv_cache1,
                crossattn_cache=self.inference_pipeline.crossattn_cache,
                current_start=0,
                cache_start=0
            )

            if self.args.denoising_loss_type == "flow":
                flow_pred_for_loss = flow_pred.flatten(0, 1)
                noise_pred = None
            else:
                flow_pred_for_loss = None
                noise_pred = self.scheduler.convert_x0_to_noise(
                    x0=pred_latent.flatten(0, 1),
                    xt=noisy_latents.flatten(0, 1),
                    timestep=diffusion_timestep.flatten(0, 1)
                ).unflatten(0, (batch_size, num_frames))

            diffusion_loss_raw = self.denoising_loss_func(
                x=latents_for_diff.flatten(0, 1),
                x_pred=pred_latent.flatten(0, 1),
                noise=diffusion_noise.flatten(0, 1),
                noise_pred=noise_pred,
                alphas_cumprod=self.scheduler.alphas_cumprod,
                timestep=diffusion_timestep.flatten(0, 1),
                flow_pred=flow_pred_for_loss
            )
            weighted_diffusion_loss = diffusion_loss_raw * self.generator_diffusion_loss_weight
            total_loss = weighted_diffusion_loss if total_loss is None else total_loss + weighted_diffusion_loss
            log_dict.update({
                "generator_diffusion_loss": weighted_diffusion_loss.detach(),
                "generator_diffusion_raw": diffusion_loss_raw.detach(),
            })

        if self.gan_loss_weight > 0.0:
            logits_fake, _, _ = self._classifier_logits(pred_image, conditional_dict)
            gan_loss_raw = F.softplus(-logits_fake).mean()
            gan_weight = torch.tensor(float(self.gan_loss_weight), device=pred_image.device, dtype=pred_image.dtype)
            if self.adaptive_d_weight and recon_target_tensor is not None:
                adaptive_weight = self._calculate_adaptive_weight(reconstruction_loss, gan_loss_raw, recon_target_tensor)
                gan_weight = adaptive_weight
                log_dict["generator_adaptive_gan_weight"] = adaptive_weight.detach()
            gan_loss = gan_loss_raw * gan_weight
            total_loss = gan_loss if total_loss is None else total_loss + gan_loss
            log_dict.update({
                "generator_gan_loss": gan_loss.detach(),
                "generator_gan_logits": logits_fake.detach().mean(),
                "generator_gan_real_prob": torch.sigmoid(logits_fake.detach()).mean(),
                "generator_gan_logits_std": logits_fake.detach().std(unbiased=False),
            })

        if total_loss is None:
            raise RuntimeError("GAN loss is disabled but DMD2Real has no teacher-driven objective.")

        loss_time = time.time() - _t_loss_start
        log_dict["loss_time"] = loss_time
        log_dict["dmdtrain_gradient_norm"] = torch.tensor(0.0, device=pred_image.device)

        return total_loss, log_dict

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
            generated_image, _, denoised_timestep_from, denoised_timestep_to = self._run_generator(
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

        cls_loss = torch.tensor(0.0, device=self.device)
        gan_log_dict = {}
        if (self.cls_on_clean_image and self.guidance_cls_loss_weight > 0.0 and
                real_latents is not None and hasattr(self.fake_score, "adding_cls_branch")):
            real_logits, reused_noise, reused_timestep_full = self._classifier_logits(real_latents.to(self.device, dtype=torch.float32), conditional_dict)
            fake_logits = self._classifier_logits_reuse(generated_image.detach(), conditional_dict, reused_noise, reused_timestep_full)
            loss_real = F.softplus(-real_logits).mean()
            loss_fake = F.softplus(fake_logits).mean()
            cls_loss = (loss_real + loss_fake) * self.guidance_cls_loss_weight
            gan_log_dict = {
                "critic_gan_loss": cls_loss.detach(),
                "critic_real_logits": real_logits.detach().mean(),
                "critic_fake_logits": fake_logits.detach().mean(),
                "critic_real_prob": torch.sigmoid(real_logits.detach()).mean(),
                "critic_fake_prob": torch.sigmoid(fake_logits.detach()).mean(),
                "critic_real_accuracy": (real_logits.detach() > 0).float().mean(),
                "critic_fake_accuracy": (fake_logits.detach() < 0).float().mean(),
                "critic_cls_loss": cls_loss.detach(),
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
            "loss_time": loss_time,
            "critic_denoising_loss": denoising_loss.detach(),
        }
        critic_log_dict.update(gan_log_dict)

        return total_loss, critic_log_dict
