import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import time
import torch.distributed as dist

from model.dmd import DMD
from utils.memory import log_gpu_memory
from utils.debug_option import DEBUG, LOG_GPU_MEMORY


class MSE_DMD(DMD):
    def __init__(self, args, device):
        super().__init__(args, device)
        # weights for combined loss
        self.lambda_dmd = getattr(args, "lambda_dmd", 1.0)
        self.lambda_mse = getattr(args, "lambda_mse", 0.1)

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
        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(
                original_latent.double()[gradient_mask],
                (original_latent.double() - grad.double()).detach()[gradient_mask],
                reduction="mean",
            )
        else:
            dmd_loss = 0.5 * F.mse_loss(
                original_latent.double(),
                (original_latent.double() - grad.double()).detach(),
                reduction="mean",
            )

        # Distillation MSE term: align student's x0 with teacher's x0 (teacher-guided)
        if gradient_mask is not None:
            mse_term = F.mse_loss(
                pred_fake_image.double()[gradient_mask],
                pred_real_image.double()[gradient_mask],
                reduction="mean",
            )
        else:
            mse_term = F.mse_loss(
                pred_fake_image.double(), pred_real_image.double(), reduction="mean"
            )

        log_dict = {
            "dmd_loss": dmd_loss.detach(),
            "teacher_mse": mse_term.detach(),
            "timestep": timestep.detach(),
            "dmdtrain_gradient_norm": grad_norm,
        }
        # 返回仅 DMD 分量；总损由 generator_loss 组合（含 latent MSE）
        return dmd_loss, log_dict

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: Dict[str, Any],
        unconditional_dict: Dict[str, Any],
        clean_latent: Optional[torch.Tensor] = None,
        initial_latent: Optional[torch.Tensor] = None,
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
            if gradient_mask is not None:
                mask = gradient_mask[:, :min_frames]
                latents_mse = F.mse_loss(pred_lat[mask].double(), target=clean_lat[mask].double(), reduction="mean")
            else:
                latents_mse = F.mse_loss(pred_lat.double(), clean_lat.double(), reduction="mean")
        else:
            latents_mse = torch.zeros((), device=pred_image.device, dtype=pred_image.dtype)

        total_loss = self.lambda_dmd * dmd_loss + self.lambda_mse * latents_mse

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
        })
        return total_loss, dmd_log_dict

