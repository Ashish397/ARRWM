# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import time
import math
from contextlib import contextmanager

from model.base import SelfForcingModelActions
from utils.memory import log_gpu_memory
import torch.distributed as dist
from utils.debug_option import DEBUG, LOG_GPU_MEMORY

# Add imports for AdaLN injection
from model.action_modulation import ActionModulationProjection
from model.action_model_patch import patch_causal_wan_model_for_action


class DMD2RealMSELAM_Actions(SelfForcingModelActions):
    def __init__(self, args, device, latent_action_model: Optional[Any] = None):
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

        # AdaLN-Zero action projection and patching
        self.motion_enabled_loss = False #bool(getattr(args, "motion_enabled_loss", False))
        self.action_loss_weight = float(getattr(args, "action_loss_weight", 0.0))
        self.action_dim = int(getattr(args, "action_dim", getattr(args, "raw_action_dim", 2)))
        if self.action_loss_weight > 0.0:
            # Apply action patches to both model and wrapper so wrapper forwards _action_modulation
            try:
                from model.action_model_patch import apply_action_patches
                apply_action_patches(self.generator)
            except Exception:
                if hasattr(self.generator, "model"):
                    patch_causal_wan_model_for_action(self.generator.model)
            model_dim = self.generator.model.dim if hasattr(self.generator.model, "dim") else 2048
            self.action_projection = ActionModulationProjection(
                action_dim=self.action_dim,
                hidden_dim=model_dim,
                num_frames=1,  # set dynamically per batch
                zero_init=True
            ).to(device)
        else:
            self.action_projection = None

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


        # Latent action model integration
        self.latent_action_loss_weight = float(getattr(args, "latent_action_loss_weight", 0.0))
        self.latent_action_model = latent_action_model
        if self.latent_action_model is not None:
            try:
                self.latent_action_model.to(device=self.device)
            except AttributeError:
                pass

        latent_shape = getattr(args, "image_or_video_shape", None)
        latent_channels = 16
        if latent_shape is not None and len(latent_shape) >= 3:
            latent_channels = int(latent_shape[2])
        self.latent_feature_dim = latent_channels
        self.action_dim = int(getattr(args, "action_dim", getattr(args, "raw_action_dim", 2)))
        self.action_head_hidden_dim = int(getattr(args, "action_head_hidden_dim", 256))


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
    def _freeze_module_params(self, module: Optional[Any]):
        if module is None or not hasattr(module, "parameters"):
            yield
            return

        params = list(module.parameters())
        if not params:
            yield
            return

        requires_grad = [p.requires_grad for p in params]
        try:
            for p in params:
                p.requires_grad_(False)
            yield
        finally:
            for p, rg in zip(params, requires_grad):
                p.requires_grad_(rg)

    def _compute_latent_action_loss(
        self,
        latents: torch.Tensor,
        actions: Optional[torch.Tensor],
        conditional_dict: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero = torch.zeros((), device=latents.device, dtype=latents.dtype)
        if self.latent_action_model is None or self.latent_action_loss_weight <= 0.0:
            return zero, {}

        if actions is None:
            return zero, {}

        if actions.device != latents.device:
            actions = actions.to(device=latents.device)

        latents_for_model = latents
        actions_for_loss = actions
        if latents.dim() >= 2 and actions.dim() >= 2:
            min_frames = min(latents.shape[1], actions.shape[1])
            latents_for_model = latents[:, :min_frames]
            actions_for_loss = actions[:, :min_frames]

        model = self.latent_action_model
        assert model is not None

        def _call_latent_action_model_sequence(inputs: torch.Tensor) -> torch.Tensor:
            if inputs.dim() != 5:
                raise ValueError(
                    f"Latent action model expects 5D tensor [B, F, C, H, W], got shape {tuple(inputs.shape)}"
                )
            bsz, frames, ch, h, w = inputs.shape
            if frames < 8:
                raise ValueError("Latent action model requires at least 8 frames for a forward window.")
            num_preds = frames - 7  # targets correspond to indices [6 : 6+num_preds)

            window_batches: list[torch.Tensor] = []
            for start in range(0, num_preds):
                window = inputs[:, start:start + 8]  # [B, 8, C, H, W]
                window_batches.append(window)
            stacked = torch.stack(window_batches, dim=1)  # [B, num_preds, 8, C, H, W]
            stacked = stacked.flatten(0, 1)  # [B*num_preds, 8, C, H, W]
            model_inputs = stacked.permute(0, 2, 1, 3, 4).contiguous()  # [B*num_preds, C, 8, H, W]

            # Ensure FP32 inference regardless of global autocast
            with torch.amp.autocast('cuda', enabled=False):
                outputs = model(model_inputs.to(dtype=torch.float32))

            if isinstance(outputs, dict):
                if "logits" in outputs and torch.is_tensor(outputs["logits"]):
                    out = outputs["logits"]
                elif "predictions" in outputs and torch.is_tensor(outputs["predictions"]):
                    out = outputs["predictions"]
                elif "loss" in outputs and torch.is_tensor(outputs["loss"]):
                    out = outputs["loss"].unsqueeze(0)
                else:
                    raise TypeError("Unsupported dict outputs from latent_action_model for sequence mode.")
            elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                first = outputs[0]
                if torch.is_tensor(first):
                    out = first
                else:
                    raise TypeError("Unsupported tuple outputs from latent_action_model for sequence mode.")
            elif torch.is_tensor(outputs):
                out = outputs
            else:
                raise TypeError("Unsupported output type from latent_action_model in sequence mode.")

            out = out.to(device=inputs.device)
            # reshape back to [B, num_preds, ...]
            out = out.reshape(bsz, num_preds, *out.shape[1:])
            return out

        def _loss_from_predictions(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            if targets.dtype.is_floating_point:
                if predictions.shape != targets.shape:
                    raise ValueError(
                        f"Latent action model predictions shape {predictions.shape} does not match targets shape {targets.shape}."
                    )
                return F.mse_loss(predictions.to(dtype=targets.dtype), targets, reduction="mean")

            logits = predictions
            num_classes = logits.shape[-1]
            logits_flat = logits.reshape(-1, num_classes)
            targets_flat = targets.reshape(-1).long()
            if logits_flat.shape[0] != targets_flat.shape[0]:
                raise ValueError(
                    "Latent action model logits and target sizes do not align for cross-entropy computation."
                )
            return F.cross_entropy(logits_flat, targets_flat)

        with self._freeze_module_params(self.latent_action_model):
            predictions_seq = _call_latent_action_model_sequence(latents_for_model)

        # Align targets: actions_for_loss[:, 6 : -1] -> length num_preds (frames-7)
        num_preds = predictions_seq.shape[1]
        target_start = 6
        target_end = target_start + num_preds
        targets_seq = actions_for_loss[:, target_start:target_end]

        # Compute supervised loss over the sequence
        base_loss = _loss_from_predictions(predictions_seq, targets_seq)
        log_updates: Dict[str, torch.Tensor] = {}

        weighted_loss = base_loss * self.latent_action_loss_weight
        log_updates.update({
            "latent_action_loss": weighted_loss.detach(),
            "latent_action_loss_raw": base_loss.detach(),
        })
        return weighted_loss.to(dtype=latents.dtype), log_updates

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

    # Use base SelfForcingModelActions._run_generator which handles action modulation and backward simulation

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

        # Latent action loss
        latent_action_loss, latent_action_logs = self._compute_latent_action_loss(
            pred_image,
            actions,
            conditional_dict,
        )
        total_loss = total_loss + latent_action_loss
        if latent_action_logs:
            log_dict.update(latent_action_logs)

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

        # No GAN or action regression losses in this variant
        total_loss = denoising_loss

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
        

        return total_loss, critic_log_dict
