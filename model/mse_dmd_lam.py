import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import time
import math
import torch.distributed as dist
from contextlib import contextmanager

from model.dmd import DMD
from utils.memory import log_gpu_memory
from utils.debug_option import DEBUG, LOG_GPU_MEMORY


class MSE_DMD_LAM(DMD):
    def __init__(self, args, device):
        super().__init__(args, device)
        # weights for combined loss
        self.lambda_dmd = getattr(args, "lambda_dmd", 1.0)
        self.lambda_mse = getattr(args, "lambda_mse", 0.1)
        self.diffusion_gan = getattr(args, "diffusion_gan", False)
        self.diffusion_gan_max_timestep = getattr(args, "diffusion_gan_max_timestep", self.num_train_timestep)
        self.concat_time_embeddings = getattr(args, "concat_time_embeddings", False)
        self.action_loss_weight = float(getattr(args, "action_loss_weight", 0.0))
        self.guidance_rgs_loss_weight = float(getattr(args, "guidance_rgs_loss_weight", self.action_loss_weight))
        self.motion_enabled_loss = bool(getattr(args, "motion_enabled_loss", False))
        self.motion_weight_c = float(getattr(args, "motion_weight_c", 2.0))

        if (self.action_loss_weight > 0.0 or self.guidance_rgs_loss_weight > 0.0) and hasattr(self.fake_score, "adding_rgs_branch"):
            try:
                self.fake_score.adding_rgs_branch(
                    time_embed_dim=1536 if self.concat_time_embeddings else 0,
                )
            except Exception:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print("[Warning] Failed to add regression branch to fake_score.")

    def compute_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: Dict[str, Any],
        unconditional_dict: Dict[str, Any],
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Combine DMD loss with distillation MSE between student's x0 and teacher's x0
        at the same noisy input and timestep.
        """
        # First compute the original DMD loss and the sampled noisy input/timestep
        original_latent = image_or_video

        batch_size, num_frame = image_or_video.shape[:2]

        with torch.no_grad():
            # sample timestep range consistent with parent implementation
            min_timestep = (
                denoised_timestep_to
                if self.ts_schedule and denoised_timestep_to is not None
                else self.min_score_timestep
            )
            max_timestep = (
                denoised_timestep_from
                if self.ts_schedule_max and denoised_timestep_from is not None
                else self.num_train_timestep
            )
            timestep = self._get_timestep(
                min_timestep,
                max_timestep,
                batch_size,
                num_frame,
                self.num_frame_per_block,
                uniform_timestep=True,
            )

            if self.timestep_shift > 1:
                timestep = (
                    self.timestep_shift
                    * (timestep / 1000)
                    / (1 + (self.timestep_shift - 1) * (timestep / 1000))
                    * 1000
                )
            timestep = timestep.clamp(self.min_step, self.max_step)

            noise = torch.randn_like(image_or_video)
            noisy_latent = self.scheduler.add_noise(
                image_or_video.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1),
            ).detach().unflatten(0, (batch_size, num_frame))

        # DMD loss (reuse parent implementation but avoid recomputing noise/timestep)
        # We need grad; replicate minimal logic from parent to keep single-pass predictions for MSE
        # 1) student (fake) x0 prediction
        _, pred_fake_image_cond = self.fake_score(
            noisy_image_or_video=noisy_latent,
            conditional_dict=conditional_dict,
            timestep=timestep,
        )
        if self.fake_guidance_scale != 0.0:
            _, pred_fake_image_uncond = self.fake_score(
                noisy_image_or_video=noisy_latent,
                conditional_dict=unconditional_dict,
                timestep=timestep,
            )
            pred_fake_image = pred_fake_image_cond + (
                pred_fake_image_cond - pred_fake_image_uncond
            ) * self.fake_guidance_scale
        else:
            pred_fake_image = pred_fake_image_cond

        # 2) teacher (real) x0 prediction
        _, pred_real_image_cond = self.real_score(
            noisy_image_or_video=noisy_latent,
            conditional_dict=conditional_dict,
            timestep=timestep,
        )
        _, pred_real_image_uncond = self.real_score(
            noisy_image_or_video=noisy_latent,
            conditional_dict=unconditional_dict,
            timestep=timestep,
        )
        pred_real_image = pred_real_image_cond + (
            pred_real_image_cond - pred_real_image_uncond
        ) * self.real_guidance_scale

        # 3) DMD grad and normalization
        grad = pred_fake_image - pred_real_image
        if getattr(self, "scheduler", None) is not None:
            p_real = original_latent - pred_real_image
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / normalizer
        grad = torch.nan_to_num(grad)
        grad_norm = torch.mean(torch.abs(grad)).detach()

        # DMD loss term (same as parent)
        target_latent = (original_latent.double() - grad.double()).detach()
        dmd_loss = 0.5 * self._motion_weighted_mse(
            pred=original_latent.double(),
            target=target_latent,
            gradient_mask=gradient_mask,
        )

        # Distillation MSE term: align student's x0 with teacher's x0 (teacher-guided)
        mse_term = self._motion_weighted_mse(
            pred=pred_fake_image.double(),
            target=pred_real_image.double(),
            gradient_mask=gradient_mask,
        )

        log_dict = {
            "dmd_loss": dmd_loss.detach(),
            "teacher_mse": mse_term.detach(),
            "timestep": timestep.detach(),
            "dmdtrain_gradient_norm": grad_norm,
        }
        # 返回仅 DMD 分量；总损由 generator_loss 组合（含 latent MSE）
        return dmd_loss, log_dict

    @contextmanager
    def _freeze_fake_score_params(self):
        """Temporarily freeze fake_score parameters during generator loss computation."""
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

    def _build_classifier_inputs(
        self,
        latents: torch.Tensor,
        conditional_dict: Dict[str, Any],
        reuse_noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def _regressor_preds(self, latents: torch.Tensor, conditional_dict: Dict[str, Any]) -> torch.Tensor:
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

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: Dict[str, Any],
        unconditional_dict: Dict[str, Any],
        clean_latent: Optional[torch.Tensor] = None,
        initial_latent: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        生成器前向：先 unroll 生成 latent，再计算 DMD 损失；若提供 clean_latent，额外计算 latent MSE，并线性加权成总损失。
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
            slice_last_frames=slice_last_frames,
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

        # Step 2: Compute the DMD loss (teacher-guided)
        _t_loss_start = time.time()
        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
        )

        # Step 3: Optional latent MSE to real data (if provided)
        if clean_latent is not None:
            # 对齐 shape：按需裁剪到相同帧数
            min_frames = min(clean_latent.shape[1], pred_image.shape[1])
            clean_lat = clean_latent[:, :min_frames].to(dtype=pred_image.dtype, device=pred_image.device)
            pred_lat = pred_image[:, :min_frames]
            mask = gradient_mask[:, :min_frames] if gradient_mask is not None else None
            latents_mse = self._motion_weighted_mse(
                pred=pred_lat.double(),
                target=clean_lat.double(),
                gradient_mask=mask,
            )
        else:
            latents_mse = torch.zeros((), device=pred_image.device, dtype=pred_image.dtype)

        action_loss = torch.zeros((), device=pred_image.device, dtype=pred_image.dtype)
        if self.action_loss_weight > 0.0 and hasattr(self.fake_score, "adding_rgs_branch"):
            if actions is None:
                raise ValueError("Actions tensor required for generator action regression loss.")
            with self._freeze_fake_score_params():
                action_preds = self._regressor_preds(pred_image, conditional_dict)
            action_targets = actions.to(device=pred_image.device, dtype=pred_image.dtype)
            action_loss = F.l1_loss(action_preds, action_targets) * self.action_loss_weight

        total_loss = self.lambda_dmd * dmd_loss + self.lambda_mse * latents_mse + action_loss

        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"Generator loss: After losses", device=self.device, rank=dist.get_rank())
        try:
            _ = total_loss.item()
        except Exception:
            pass

        dmd_log_dict.update({
            "gen_time": gen_time,
            "loss_time": time.time() - _t_loss_start,
            "generator_loss": total_loss.detach(),
            "latent_mse": latents_mse.detach(),
            "generator_act_loss": action_loss.detach(),
        })
        return total_loss, dmd_log_dict

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: Dict[str, Any],
        unconditional_dict: Dict[str, Any],
        clean_latent: torch.Tensor,
        initial_latent: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Critic training step with optional action regression guidance.
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

        rgs_loss = torch.tensor(0.0, device=self.device)
        action_log_dict: Dict[str, Any] = {}
        if self.guidance_rgs_loss_weight > 0.0 and hasattr(self.fake_score, "adding_rgs_branch"):
            if clean_latent is None:
                raise ValueError("Real latents required for critic action regression loss.")
            if actions is None:
                raise ValueError("Actions tensor required for critic action regression loss.")
            live_actions = self._regressor_preds(
                clean_latent.to(self.device, dtype=torch.float32),
                conditional_dict,
            )
            target_actions = actions.to(device=self.device, dtype=torch.float32)
            rgs_loss = F.l1_loss(live_actions, target_actions) * self.guidance_rgs_loss_weight
            action_log_dict = {
                "critic_rgs_loss": rgs_loss.detach(),
                "critic_action_pred_mean": live_actions.detach().mean(),
                "critic_action_target_mean": target_actions.detach().mean(),
            }

        total_loss = denoising_loss + rgs_loss

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
        }
        critic_log_dict.update(action_log_dict)

        return total_loss, critic_log_dict
