# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
import torch.nn.functional as F
from typing import Optional, Tuple
import torch
import torch.nn as nn
import time
import math
from contextlib import contextmanager
from pathlib import Path
import sys

from model.base import SelfForcingModel
from utils.memory import log_gpu_memory
import torch.distributed as dist
from utils.debug_option import DEBUG, LOG_GPU_MEMORY

# Import Motion2ActionModel from test_motion_2_action.py
# Add parent directory to path to allow imports from latent_actions
_latent_actions_path = Path(__file__).parent.parent / "latent_actions"
if str(_latent_actions_path) not in sys.path:
    sys.path.insert(0, str(_latent_actions_path))

from train_motion_2_action import Motion2ActionModel
from cotracker.predictor import CoTrackerPredictor

class DMD2B2BLAM_actions(SelfForcingModel):
    def __init__(self, args, device, latent_action_model: Optional[nn.Module] = None, motion_model: Optional[nn.Module] = None):
        """
        Initialize the DMD (Distribution Matching Distillation) module.
        This class is self-contained and compute generator and fake score losses
        in the forward pass.
        """
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
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
        self.indep_lat_act_weight = getattr(args, "indep_lat_act_weight", None)
        self.guidance_cls_loss_weight = getattr(args, "guidance_cls_loss_weight", self.gan_loss_weight)
        self.action_cfg_enabled = getattr(args, "action_cfg_enabled", False)
        self.action_cfg_type = getattr(args, "action_cfg_type", "framewise")
        self.action_cfg_side_multiplier = getattr(args, "action_cfg_side_multiplier", 1.25)
        object.__setattr__(self, "_latent_action_model_ref", None)
        if latent_action_model is not None:
            try:
                latent_action_model.to(device=device)
            except AttributeError:
                pass
            object.__setattr__(self, "_latent_action_model_ref", latent_action_model)

        object.__setattr__(self, "_motion_model_ref", None)
        if motion_model is not None:
            try:
                motion_model.to(device=device)
            except AttributeError:
                pass
            object.__setattr__(self, "_motion_model_ref", motion_model)

        self.action_dim = int(getattr(args, "action_dim", getattr(args, "raw_action_dim", 2)))
        axis_weights = getattr(args, "action_axis_weights", None)
        if axis_weights is not None:
            if len(axis_weights) != self.action_dim:
                raise ValueError(
                    f"action_axis_weights length {len(axis_weights)} does not match action_dim {self.action_dim}"
                )
            self._action_axis_weights = torch.tensor(axis_weights, dtype=torch.float32)
        else:
            self._action_axis_weights = None

        self.fake_score.adding_cls_branch(
            time_embed_dim=0,
        )
    
    def _estimate_action_from_motion(self, pred_frames: torch.Tensor, estimated_motion: torch.Tensor) -> torch.Tensor:
        # Use pre-initialized motion2action model (initialized in __init__ for FSDP compatibility)
        if self.latent_action_model is None:
            raise RuntimeError("motion2action model not initialized")

        latents_for_model = pred_frames.reshape(7, 3, 16, 60, 104).permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
        
        # Run model inference (model is frozen in eval mode, but gradients flow through to inputs)
        # Gradients flow: loss -> pred_actions -> model -> latents_for_model -> pred_frames
        # motion_out is detached, so no gradients flow through CoTracker
        model_output = self.latent_action_model(latents_for_model, estimated_motion)
        
        if isinstance(model_output, tuple) and len(model_output) == 3:
            # return_dist=True: (dist, mean_action, log_std)
            _, pred_actions, log_std = model_output
            # std = torch.exp(torch.clamp(log_std, -5.0, 2.0)) + 1e-6
        elif isinstance(model_output, tuple) and len(model_output) == 2:
            # return_dist=False: (mean_action, log_std)
            pred_actions, log_std = model_output
            # std = torch.exp(torch.clamp(log_std, -5.0, 2.0)) + 1e-6
        else:
            raise ValueError("Invalid model output")
        
        return pred_actions, log_std

    def _estimate_motion(self, pred_frames: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # Decode latents to pixel space
            # Use self.dtype to match VAE's dtype (bfloat16 if mixed_precision, float32 otherwise)
            # Create a modified version with dummy frame for VAE decode (preserve original pred_frames for gradients)
            pred_frames_with_dummy = torch.cat([pred_frames[:, 0:1], pred_frames], dim=1)
            pixels = self.vae.decode_to_pixel(pred_frames_with_dummy.to(dtype=self.dtype))[:, 1:, ...]  # [B, F, 3, H, W] in [-1, 1] range

            # Convert to format expected by cotracker: [B, T, C, H, W]
            # VAE decode returns pixels in [-1, 1] range, scale to [0, 255] as cotracker expects
            # Similar to test_motion_2_action_scratch.py line 454
            video_pixels = (255 * 0.5 * (pixels + 1.0)).clamp(0, 255).float()  # [B, F, 3, H, W] in [0, 255] range

            #Convert video pixels from shape [1, 84, 3, H, W] to [7, 12, 3, H, W]
            batch_size, num_frames, num_channels, height, width = video_pixels.shape
            assert num_frames == 84
            video_pixels = video_pixels.reshape(7, 12, num_channels, height, width)
            grid_size = 10
            N = grid_size ** 2

            # Use pre-initialized cotracker model (initialized in __init__ for FSDP compatibility)
            if self.motion_model is None:
                raise RuntimeError("cotracker not initialized")

            # Run cotracker on frame window (similar to test_motion_2_action_scratch.py line 493)
            with torch.amp.autocast(device_type='cuda', enabled=True):
                tracks_w, vis_w = self.motion_model(video_pixels, grid_size=grid_size)  # [B, T_window, N, 2], [B, T_window, N]

            #pred_tracks will have shape [7, 12, N, 2], pred_visibility will have shape [7, 12, N]

            # Vectorized motion computation - compute all deltas at once (similar to test_motion_2_action_scratch.py line 508)
            motion_out = (tracks_w[:, 1:] - tracks_w[:, :-1]).mean(dim=1)  # output shape [7, N, 2]

            # Visibility: single conversion and mean (similar to test_motion_2_action_scratch.py line 511)
            vis_out = vis_w.float().mean(dim=1).unsqueeze(-1)  # [7, N, 1]

            # Single concatenation (similar to test_motion_2_action_scratch.py line 514)
            motion_out = torch.cat([motion_out, vis_out], dim=-1)  # [7, N, 3]

        return motion_out.detach()

    def _compute_kl_grad(
        self, 
        noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict_pos: dict, 
        unconditional_dict: dict,
        normalization: bool = True,
        conditional_dict_neg = None,
        fake_guidance_scale = None,
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
            conditional_dict=conditional_dict_pos,
            timestep=timestep
        )

        if (conditional_dict_neg is not None) and (fake_guidance_scale is not None):
            _, pred_fake_image_uncond = self.fake_score(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=conditional_dict_neg,
                timestep=timestep
            )
            pred_fake_image = pred_fake_image_cond + (
                pred_fake_image_cond - pred_fake_image_uncond
            ) * fake_guidance_scale
        else:
            pred_fake_image = pred_fake_image_cond

        # Step 2: Compute the real score
        # We compute the conditional and unconditional prediction
        # and add them together to achieve cfg (https://arxiv.org/abs/2207.12598)
        _, pred_real_image_cond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict_neg,
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

    @property
    def latent_action_model(self) -> Optional[nn.Module]:
        return getattr(self, "_latent_action_model_ref", None)

    @property
    def motion_model(self) -> Optional[nn.Module]:
        return getattr(self, "_motion_model_ref", None)

    def _classifier_logits(self, latents: torch.Tensor, conditional_dict: dict) -> torch.Tensor:
        """
        Efficiently extract only classification logits from fake_score.
        Uses clean latents (no noise) for faster computation, matching dmd2realmselam_actions.py.
        """
        if not hasattr(self.fake_score, "adding_cls_branch"):
            raise RuntimeError("fake_score does not support classification branch required for DMD2 GAN loss.")
        
        batch_size, num_frames = latents.shape[:2]
        device = latents.device
        # Use clean latents (no noise) for classification - much faster than adding noise
        timestep_full = torch.zeros((batch_size, num_frames), device=device, dtype=torch.long)
        
        outputs = self.fake_score(
            noisy_image_or_video=latents.float(),
            conditional_dict=conditional_dict,
            timestep=timestep_full,
            classify_mode=True,
        )
        logits = outputs[-1]
        return logits.squeeze(-1)

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

    def _compute_latent_action_loss(
        self,
        pred_frames: torch.Tensor,  # [B, 21, C, H, W]
        actions: torch.Tensor,  # [B, 21, ...]
        estimated_motion: torch.Tensor,  # [B, 21, N, 2]
        estimated_action: torch.Tensor,  # [B, 21, 2]
        estimated_action_log_std: torch.Tensor,  # [B, 21, 2]
    ) -> Tuple[torch.Tensor, dict]:
        assert estimated_action.requires_grad == True, "estimated_action detached on arrival"

        # Compute loss: diff / std where diff = |pred_actions - actions_for_model|
        # Large diff/std = large loss, small diff/std = small loss
        # Gradients flow back through pred_actions -> model -> latents_for_model -> pred_frames
        diff = estimated_action - actions.reshape(actions.shape[0], 7, 3, actions.shape[2]).mean(dim=2).squeeze(2)  # [B, 2]
        # Use absolute value of diff/std as loss (or squared)
        normalized_diff = diff #/ std  # [B, 2]
        loss_per_sample = normalized_diff.abs().mean(dim=1)  # [B] - sum over action dimensions
        loss = loss_per_sample.mean()  # Scalar loss
        
        logs = {
            "latent_action_loss": loss.detach(),
            "latent_action_pred_mean": estimated_action.detach().mean(),
            "latent_action_target_mean": actions.detach().mean(),
            "latent_action_std_mean": estimated_action_log_std.detach().mean(),
        }

        return loss, logs

    def _cat(self, a: torch.Tensor, eps: float = 0.1) -> torch.Tensor:  # [B,21,2] -> [B,21] in {0..4}
        a0, a1 = a[..., 0], a[..., 1] * self.action_cfg_side_multiplier
        is_zero = (a0.abs() < eps) & (a1.abs() < eps)
        dom0 = (a0.abs() >= a1.abs()) & (~is_zero)
        dom1 = (~dom0) & (~is_zero)

        cat = torch.zeros_like(a0, dtype=torch.long)          # 0 = zero
        cat = torch.where(dom0 & (a0 > 0), 1, cat)            # 1 = forward
        cat = torch.where(dom0 & (a0 < 0), 2, cat)            # 2 = backward
        cat = torch.where(dom1 & (a1 > 0), 3, cat)            # 3 = right
        cat = torch.where(dom1 & (a1 < 0), 4, cat)            # 4 = left
        return cat

    def _build_action_cfg_framewise(
        self,
        conditional_dict: dict,
        actions: torch.Tensor,            # [B,21,2]  (given / intended)
        estimated_action: torch.Tensor,   # [B,21,2]  (estimated)
        num_frame: int,
        device,
        dtype,
    ):
        actions = actions.to(device=device, dtype=dtype)
        estimated_action = estimated_action.to(device=device, dtype=dtype).unsqueeze(0).repeat_interleave(3, axis=1)

        given_cat = self._cat(actions)            # [B,21]
        est_cat = self._cat(estimated_action)     # [B,21]
        success = (given_cat == est_cat)     # [B,21]

        # Default (NOT success): guidance=0, repel given action, nothing to attract
        pos = actions.clone()  # repel
        neg = actions.clone()  # attract (unused because guidance=0)

        # Success: attract given action; repel -actions (unless zero, then repel forward [1,0])
        is_zero_action = (given_cat == 0)  # [B,21] - check if zero action
        fwd = torch.zeros_like(actions); fwd[..., 0] = 1.0
        negated_actions = -actions  # [B,21,2] - opposite action

        pos = torch.where(
            success.unsqueeze(-1),
            torch.where(
                is_zero_action.unsqueeze(-1),
                fwd,  # forward [1,0] for zero
                negated_actions  # -actions for non-zero
            ),
            pos  # keep original actions when not success
        )
        neg = torch.where(success.unsqueeze(-1), actions, neg)

        guidance = torch.where(
            success,
            torch.tensor(float(self.fake_guidance_scale), device=device, dtype=dtype),
            torch.tensor(0.0, device=device, dtype=dtype),
        )  # [B,21]
        guidance = guidance[:, :, None, None, None]  # [B,21,1,1,1] for broadcasting with [B,21,C,H,W]

        cond_pos = self._with_action_conditioning(conditional_dict, pos, num_frame, device, dtype)
        cond_neg = self._with_action_conditioning(conditional_dict, neg, num_frame, device, dtype)
        return cond_pos, cond_neg, guidance
        
    def _build_action_cfg_samplewise(
        self,
        conditional_dict: dict,
        actions: torch.Tensor,            # [B,21,2]  (given / intended)
        estimated_action: torch.Tensor,   # [B,21,2]  (estimated)
        num_frame: int,
        device,
        dtype,
    ):
        actions = actions.to(device=device, dtype=dtype)
        estimated_action = estimated_action.to(device=device, dtype=dtype).unsqueeze(0)

        # Sample-wise: compute mean actions across all frames
        mean_actions = actions.mean(dim=1)  # [B, 2] - average across frame dimension
        mean_estimated_action = estimated_action.mean(dim=1)  # [B, 2] - average across frame dimension

        # Categorize the mean actions (sample-wise)
        mean_given_cat = self._cat(mean_actions.unsqueeze(1))  # [B, 1] -> squeeze to [B]
        mean_est_cat = self._cat(mean_estimated_action.unsqueeze(1))  # [B, 1] -> squeeze to [B]
        
        # Check if categories match (sample-wise)
        mean_given_cat = mean_given_cat.squeeze(1) if mean_given_cat.dim() > 1 else mean_given_cat
        mean_est_cat = mean_est_cat.squeeze(1) if mean_est_cat.dim() > 1 else mean_est_cat
        success = (mean_given_cat == mean_est_cat)  # [B] - sample-wise success

        # If categories match: cond_neg = actions given, cond_pos = -actions (unless zero, then [1,0])
        # If categories don't match: guidance = 0, cond_pos = actions given
        is_zero_action = (mean_given_cat == 0)  # [B] - check if zero action
        forward_action = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
        negated_actions = -actions  # [B,21,2] - opposite action
        
        # When success: use -actions unless zero (then use forward), otherwise use actions
        pos = torch.where(
            success.unsqueeze(-1).unsqueeze(-1),
            torch.where(
                is_zero_action.unsqueeze(-1).unsqueeze(-1),
                forward_action.unsqueeze(0).unsqueeze(0).expand(actions.shape[0], actions.shape[1], -1),  # [1,0] * 21 for zero
                negated_actions  # -actions for non-zero
            ),
            actions.clone()  # actions given when categories don't match
        )
        neg = actions.clone()  # actions given for cond_neg (used when categories match)

        # Set guidance scale: fake_guidance_scale if categories match, 0 otherwise
        guidance = torch.where(
            success,
            torch.tensor(float(self.fake_guidance_scale), device=device, dtype=dtype),
            torch.tensor(0.0, device=device, dtype=dtype),
        )  # [B]
        
        # Expand guidance to [B,21,1,1,1] for broadcasting with [B,21,C,H,W]
        guidance = guidance.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B,1,1,1,1]
        guidance = guidance.expand(-1, num_frame, -1, -1, -1)  # [B,21,1,1,1]

        cond_pos = self._with_action_conditioning(conditional_dict, pos, num_frame, device, dtype)
        cond_neg = self._with_action_conditioning(conditional_dict, neg, num_frame, device, dtype)
        return cond_pos, cond_neg, guidance
            
    def compute_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        estimated_action: torch.Tensor,
        estimated_action_log_std: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0,
        actions: Optional[torch.Tensor] = None,
        *,
        frame_start: int = 0,
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

        if self.action_cfg_type == "framewise":
            cond_pos, cond_neg, per_sample_fake_scale = self._build_action_cfg_framewise(
                conditional_dict, 
                actions, 
                estimated_action, 
                num_frame,
                image_or_video.device, 
                image_or_video.dtype,
            )
        elif self.action_cfg_type == "samplewise":
            cond_pos, cond_neg, per_sample_fake_scale = self._build_action_cfg_samplewise(
                conditional_dict, 
                actions, 
                estimated_action, 
                num_frame,
                image_or_video.device, 
                image_or_video.dtype,
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
                conditional_dict_pos=cond_pos,
                conditional_dict_neg=cond_neg,
                unconditional_dict=unconditional_dict,
                fake_guidance_scale=per_sample_fake_scale,
            )

        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double()[gradient_mask], (
                original_latent.double() - grad.double()
                ).detach()[gradient_mask], reduction="mean")
        else:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(), (
                original_latent.double() - grad.double()
                ).detach(), reduction="mean")

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
        """Train the generator with DMD2 loss plus optional GAN terms."""
        self._set_fake_score_trainable(False)
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory("Generator loss: Before generator unroll", device=self.device, rank=dist.get_rank())

        slice_last_frames = getattr(self.args, "slice_last_frames", 21)
        _t_gen_start = time.time()
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to, _ = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent,
            slice_last_frames=slice_last_frames,
            action_inputs=actions,
        )

        estimated_motion = self._estimate_motion(pred_image)
        estimated_action, estimated_action_log_std = self._estimate_action_from_motion(pred_image, estimated_motion)
        estimated_action_log_std = estimated_action_log_std.detach()

        assert pred_image.requires_grad, "generator rollout returned a detached tensor"
        gen_time = time.time() - _t_gen_start
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory("Generator loss: After generator unroll", device=self.device, rank=dist.get_rank())

        _t_loss_start = time.time()
        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            estimated_action=estimated_action.detach(),
            estimated_action_log_std=estimated_action_log_std,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
            actions=actions,
        )
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory("Generator loss: After compute_distribution_matching_loss", device=self.device, rank=dist.get_rank())

        total_loss = dmd_loss
        log_dict = dict(dmd_log_dict)
        log_dict.update({
            "gen_time": gen_time,
            "generator_dmd_loss": dmd_loss.detach(),
        })
        log_dict.update(self._action_batch_stats(actions, "actions"))

        with self._freeze_fake_score_params():
            cls_conditional = self._with_action_conditioning(
                conditional_dict,
                actions,
                pred_image.shape[1],
                pred_image.device,
                pred_image.dtype,
                # detach_modulation=True, #we do not detach modulation as this is equivalent to the same prompt that generated the latents
            )
            # Use optimized helper that only computes logits (no denoising prediction)
            logits_fake = self._classifier_logits(pred_image, cls_conditional)
        gan_loss = F.softplus(-logits_fake).mean() * self.gan_loss_weight
        total_loss = total_loss + gan_loss
        log_dict.update({
            "generator_gan_loss": gan_loss.detach(),
            "generator_gan_logits": logits_fake.detach().mean(),
            "generator_gan_real_prob": torch.sigmoid(logits_fake.detach()).mean(),
            "generator_gan_logits_std": logits_fake.detach().std(unbiased=False),
        })

        latent_action_loss, latent_action_logs = self._compute_latent_action_loss(
            pred_image, actions, estimated_motion, estimated_action, estimated_action_log_std
        )
        total_loss = total_loss + latent_action_loss * self.indep_lat_act_weight
        log_dict.update(latent_action_logs)

        loss_time = time.time() - _t_loss_start
        log_dict["loss_time"] = loss_time
        # dmdtrain_gradient_norm is already set in dmd_log_dict from compute_distribution_matching_loss
        # No need to overwrite it with 0.0

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
            generated_image, _, denoised_timestep_from, denoised_timestep_to, _ = self._run_generator(
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
        )
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

        # Compute denoising prediction (needs noisy input)
        _, pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_generated_image,
            conditional_dict=critic_conditional,
            timestep=critic_timestep,
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

        # For GAN classification loss, use clean latents (much faster than noisy)
        # Only compute if guidance_cls_loss_weight > 0 to avoid unnecessary computation
        if self.guidance_cls_loss_weight > 0.0 and real_latents is not None:
            # Prepare conditional dict for real latents (may have different frame count)
            real_conditional = self._with_action_conditioning(
                conditional_dict,
                actions,
                real_latents.shape[1],
                real_latents.generated_image.device,
                real_latents.generated_image.dtype,
                detach_modulation=True,
            )
            # Only extract logits, not denoising predictions (uses clean latents, no noise)
            fake_logits = self._classifier_logits(generated_image.detach(), critic_conditional)
            real_logits = self._classifier_logits(real_latents.to(generated_image.device, dtype=generated_image.dtype), real_conditional)
            
            loss_real = F.softplus(-real_logits.squeeze(-1)).mean()
            loss_fake = F.softplus(fake_logits.squeeze(-1)).mean()
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
        critic_log_dict.update(self._action_batch_stats(actions, "actions"))

        return total_loss, critic_log_dict
