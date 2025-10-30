# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
import torch.nn.functional as F
from typing import Optional, Tuple
import torch
import torch.nn as nn
import time
import math
from contextlib import contextmanager

from model.base import SelfForcingModel
from pipeline.action_selforcing import ActionSelfForcingTrainingPipeline
from utils.memory import log_gpu_memory
import torch.distributed as dist
from utils.debug_option import DEBUG, LOG_GPU_MEMORY


class DMD2RealMSELAM(SelfForcingModel):
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
        self.inference_pipeline: Optional[ActionSelfForcingTrainingPipeline] = None

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
        self.action_loss_weight = getattr(args, "action_loss_weight", 1.0)
        self.guidance_cls_loss_weight = getattr(args, "guidance_cls_loss_weight", self.gan_loss_weight)
        self.guidance_rgs_loss_weight = getattr(args, "guidance_rgs_loss_weight", self.action_loss_weight)
        self.diffusion_gan = getattr(args, "diffusion_gan", False)
        self.diffusion_gan_max_timestep = getattr(args, "diffusion_gan_max_timestep", self.num_train_timestep)
        self.concat_time_embeddings = getattr(args, "concat_time_embeddings", False)
        self.generator_mse_loss_weight = getattr(args, "generator_mse_loss_weight", 0.0)
        self.motion_enabled_loss = bool(getattr(args, "motion_enabled_loss", False))
        self.motion_weight_c = float(getattr(args, "motion_weight_c", 2.0))
        self.enable_adaln_zero = bool(getattr(args, "enable_adaln_zero", True))
        self.action_module = getattr(args, "action_module", None)

        if hasattr(self.fake_score, "adding_cls_branch"):
            try:
                self.fake_score.adding_cls_branch(
                    time_embed_dim=1536 if self.concat_time_embeddings else 0
                )
            except Exception:
                if dist.get_rank() == 0:
                    print("[Warning] Failed to add classification branch to fake_score.")

        # Only initialize regression branch if action loss weights are non-zero
        if (self.action_loss_weight > 0.0 or self.guidance_rgs_loss_weight > 0.0) and hasattr(self.fake_score, "adding_rgs_branch"):
            try:
                self.fake_score.adding_rgs_branch(
                    time_embed_dim=1536 if self.concat_time_embeddings else 0,
                )
            except Exception:
                if dist.get_rank() == 0:
                    print("[Warning] Failed to add regression branch to fake_score.")

        latent_shape = getattr(args, "image_or_video_shape", None)
        latent_channels = 16
        if latent_shape is not None and len(latent_shape) >= 3:
            latent_channels = int(latent_shape[2])
        self.latent_feature_dim = latent_channels
        self.action_dim = int(getattr(args, "action_dim", getattr(args, "raw_action_dim", 2)))
        self.action_head_hidden_dim = int(getattr(args, "action_head_hidden_dim", 256))
        self.action_loss_weight = float(getattr(args, "action_loss_weight", 1.0))


    def _initialize_inference_pipeline(self):
        if self.inference_pipeline is not None:
            return

        local_attn_size = getattr(self.args, "model_kwargs", {}).get("local_attn_size", -1)
        slice_last_frames = getattr(self.args, "slice_last_frames", 21)
        num_training_frames = getattr(self.args, "num_training_frames")

        self.inference_pipeline = ActionSelfForcingTrainingPipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.args.independent_first_frame,
            same_step_across_blocks=self.args.same_step_across_blocks,
            last_step_only=self.args.last_step_only,
            num_max_frames=num_training_frames,
            context_noise=self.args.context_noise,
            local_attn_size=local_attn_size,
            slice_last_frames=slice_last_frames,
            num_training_frames=num_training_frames,
            action_dim=self.action_dim,
            enable_adaln_zero=self.enable_adaln_zero,
            action_module=self.action_module,
        )


    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        initial_latent: torch.Tensor = None,
        slice_last_frames: int = 21,
        actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[int], Optional[int]]:
        action_inputs = None
        if actions is not None:
            action_inputs = {
                "action_features": actions.to(device=self.device, dtype=torch.float32),
            }

        self._initialize_inference_pipeline()

        if isinstance(self.inference_pipeline, ActionSelfForcingTrainingPipeline):
            self.inference_pipeline.set_default_action_inputs(action_inputs)

        try:
            return super()._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                initial_latent=initial_latent,
                slice_last_frames=slice_last_frames,
            )
        finally:
            if isinstance(self.inference_pipeline, ActionSelfForcingTrainingPipeline):
                self.inference_pipeline.set_default_action_inputs(None)


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

    def _regressor_preds(self, latents: torch.Tensor, conditional_dict: dict) -> torch.Tensor:
        if not hasattr(self.fake_score, "adding_rgs_branch"):
            raise RuntimeError("fake_score does not support regression branch required for Action loss.")

        noisy_latents, timestep_full = self._build_classifier_inputs(latents, conditional_dict)

        outputs = self.fake_score(
            noisy_image_or_video=noisy_latents,
            conditional_dict=conditional_dict,
            timestep=timestep_full,
            regress_mode=True,
            concat_time_embeddings=self.concat_time_embeddings,
        )

        preds = outputs[-1]
        return preds

    def _compute_motion_weights(self, reference: torch.Tensor) -> torch.Tensor:
        if (not self.motion_enabled_loss) or reference.ndim < 5:
            return torch.ones_like(reference, dtype=reference.dtype)

        batch_size, num_frames = reference.shape[:2]
        if num_frames <= 1:
            return torch.ones_like(reference, dtype=reference.dtype)

        c = max(self.motion_weight_c, 1.0)
        ref = reference.detach()
        diffs = torch.abs(ref[:, 1:] - ref[:, :-1])
        diffs_flat = diffs.reshape(batch_size, num_frames - 1, -1).to(torch.float32)
        weights_flat = torch.softmax(diffs_flat, dim=1)
        log_c = math.log(c)
        scaled_flat = torch.exp(weights_flat * log_c)
        scaled = scaled_flat.reshape_as(diffs).to(dtype=reference.dtype, device=reference.device)

        weights = torch.ones_like(reference, dtype=reference.dtype, device=reference.device)
        weights[:, 1:] = scaled
        return weights

    def _motion_weighted_mse(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        gradient_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        error = (pred - target) ** 2

        mask = None
        if gradient_mask is not None:
            mask = gradient_mask.to(torch.bool)

        if self.motion_enabled_loss:
            weights = self._compute_motion_weights(target)
            error = error * weights.to(dtype=error.dtype, device=error.device)

        if mask is not None:
            error = error[mask]

        return error.mean()

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

        target_latent = (original_latent.double() - grad.double()).detach()
        dmd_loss = 0.5 * self._motion_weighted_mse(
            pred=original_latent.double(),
            target=target_latent,
            gradient_mask=gradient_mask,
        )
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
        """Train the generator with DMD2 loss plus optional MSE and GAN terms."""
        self._set_fake_score_trainable(False)
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory("Generator loss: Before generator unroll", device=self.device, rank=dist.get_rank())

        slice_last_frames = getattr(self.args, "slice_last_frames", 21)
        _t_gen_start = time.time()
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent,
            slice_last_frames=slice_last_frames,
            actions=actions,
        )
        gen_time = time.time() - _t_gen_start
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory("Generator loss: After generator unroll", device=self.device, rank=dist.get_rank())

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
            log_gpu_memory("Generator loss: After compute_distribution_matching_loss", device=self.device, rank=dist.get_rank())
        try:
            loss_val = dmd_loss.item()
        except Exception:
            loss_val = float("nan")
        loss_time = time.time() - _t_loss_start

        total_loss = dmd_loss
        log_dict = dict(dmd_log_dict)
        log_dict.update({
            "gen_time": gen_time,
            "loss_time": loss_time,
            "generator_dmd_loss": dmd_loss.detach(),
        })

        if self.generator_mse_loss_weight > 0.0:
            mse_target = None
            if real_latents is not None:
                mse_target = real_latents
            elif clean_latent is not None:
                mse_target = clean_latent

            if mse_target is not None:
                mse_target = mse_target.to(device=pred_image.device, dtype=pred_image.dtype)

                # Align number of frames if the targets include more temporal steps than the rollout
                min_frames = min(pred_image.shape[1], mse_target.shape[1])
                pred_for_mse = pred_image[:, :min_frames]
                target_for_mse = mse_target[:, :min_frames]

                mse_loss = self._motion_weighted_mse(
                    pred=pred_for_mse.float(),
                    target=target_for_mse.float(),
                )
                weighted_mse_loss = mse_loss * self.generator_mse_loss_weight
                total_loss = weighted_mse_loss if total_loss is None else total_loss + weighted_mse_loss
                log_dict.update({
                    "generator_mse_loss": weighted_mse_loss.detach(),
                    "generator_mse_raw": mse_loss.detach(),
                })

        if self.gan_loss_weight > 0.0 and hasattr(self.fake_score, "adding_cls_branch"):
            with self._freeze_fake_score_params():
                logits_fake = self._classifier_logits(pred_image, conditional_dict)
            gan_loss = F.softplus(-logits_fake).mean() * self.gan_loss_weight
            total_loss = total_loss + gan_loss
            log_dict.update({
                "generator_gan_loss": gan_loss.detach(),
                "generator_gan_logits": logits_fake.detach().mean(),
                "generator_gan_real_prob": torch.sigmoid(logits_fake.detach()).mean(),
                "generator_gan_logits_std": logits_fake.detach().std(unbiased=False),
            })

        if self.action_loss_weight > 0.0 and hasattr(self.fake_score, "adding_rgs_branch"):
            if actions is None:
                raise ValueError("Actions tensor required for generator action regression loss.")
            with self._freeze_fake_score_params():
                action_preds = self._regressor_preds(pred_image, conditional_dict)
            act_loss = F.l1_loss(action_preds, actions.to(device=pred_image.device, dtype=pred_image.dtype)) * self.action_loss_weight
            total_loss = total_loss + act_loss
            log_dict.update({
                "generator_act_loss": act_loss.detach(),
            })

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
                actions=actions,
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
            real_logits = self._classifier_logits(real_latents.to(self.device, dtype=torch.float32), conditional_dict)
            fake_logits = self._classifier_logits(generated_image.detach(), conditional_dict)
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

        rgs_loss = torch.tensor(0.0, device=self.device)
        action_log_dict = {}
        if self.guidance_rgs_loss_weight > 0.0 and hasattr(self.fake_score, "adding_rgs_branch"):
            if real_latents is None:
                raise ValueError("Real latents required for critic action regression loss.")
            if actions is None:
                raise ValueError("Actions tensor required for critic action regression loss.")
            live_actions = self._regressor_preds(real_latents.to(self.device, dtype=torch.float32), conditional_dict)
            rgs_loss = F.l1_loss(live_actions, actions.to(device=self.device, dtype=torch.float32)) * self.guidance_rgs_loss_weight
            action_log_dict = {
                "critic_rgs_loss": rgs_loss.detach(),
                "critic_action_pred_mean": live_actions.detach().mean(),
                "critic_action_target_mean": actions.detach().mean(),
            }

        total_loss = denoising_loss + cls_loss + rgs_loss

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
        critic_log_dict.update(action_log_dict)

        return total_loss, critic_log_dict
