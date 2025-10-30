# pyright: reportGeneralTypeIssues=false
from __future__ import annotations

from typing import Any, Optional, cast

import torch

from pipeline.self_forcing_training import SelfForcingTrainingPipeline
from model.action_modulation import ActionModulationProjection
from model.action_model_patch import apply_action_patches
from utils.scheduler import SchedulerInterface
from utils.wan_wrapper import WanDiffusionWrapper


class ActionSelfForcingTrainingPipeline(SelfForcingTrainingPipeline):
    """Self-Forcing 训练版本的动作条件化流水线。

    继承 `SelfForcingTrainingPipeline`，在每次调用 Wan 生成器时将动作调制（AdaLN-Zero）
    与当前 timestep 一起注入，从而在训练/蒸馏阶段保持与推理时一致的动作条件化行为。
    """

    def __init__(
        self,
        denoising_step_list: list[int],
        scheduler: SchedulerInterface,
        generator: WanDiffusionWrapper,
        num_frame_per_block: int = 3,
        independent_first_frame: bool = False,
        same_step_across_blocks: bool = False,
        last_step_only: bool = False,
        num_max_frames: int = 21,
        context_noise: int = 0,
        *,
        action_dim: int = 512,
        enable_adaln_zero: bool = True,
        action_module: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        self.action_dim: int = action_dim
        self.enable_adaln_zero: bool = enable_adaln_zero
        self.action_module: Optional[Any] = action_module
        self.action_projection: Optional[ActionModulationProjection] = None
        self._action_conditioning_enabled: bool = enable_adaln_zero or action_module is not None
        self._default_action_inputs: Optional[dict[str, Any]] = None

        super().__init__(
            denoising_step_list=denoising_step_list,
            scheduler=scheduler,
            generator=generator,
            num_frame_per_block=num_frame_per_block,
            independent_first_frame=independent_first_frame,
            same_step_across_blocks=same_step_across_blocks,
            last_step_only=last_step_only,
            num_max_frames=num_max_frames,
            context_noise=context_noise,
            **kwargs,
        )

        if self._action_conditioning_enabled:
            _ = apply_action_patches(self.generator)

        if self.enable_adaln_zero:
            device = self._resolve_module_device(self.generator)
            model_dim = getattr(self.generator.model, "dim", 2048)
            self.action_projection = ActionModulationProjection(
                action_dim=self.action_dim,
                hidden_dim=model_dim,
                num_frames=1,
                zero_init=True,
            ).to(device)

    @staticmethod
    def _resolve_module_device(module: WanDiffusionWrapper) -> torch.device:
        try:
            param = next(module.parameters())
            return param.device
        except (StopIteration, AttributeError):
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _compute_action_modulation(
        self,
        action_inputs: Optional[dict[str, Any]],
        frame_start: int,
        num_frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not self._action_conditioning_enabled or action_inputs is None:
            return None

        batch_mod: Optional[torch.Tensor] = None

        provided_mod = action_inputs.get("action_modulation") if isinstance(action_inputs, dict) else None
        if provided_mod is not None:
            provided_mod = provided_mod.to(device=device, dtype=dtype)
            if provided_mod.dim() == 3:
                # Assume shape [B, total_frames, hidden_dim * 6]
                total_frames = provided_mod.shape[1]
                hidden = provided_mod.shape[2] // 6
                provided_mod = provided_mod.view(provided_mod.shape[0], total_frames, 6, hidden)
            if provided_mod.dim() != 4:
                raise ValueError("action_modulation 需要形状 [B, T, 6, hidden_dim] 或兼容形状")
            if frame_start + num_frames > provided_mod.shape[1]:
                raise ValueError("action_modulation 的帧长度不足以覆盖当前块")
            batch_mod = provided_mod[:, frame_start:frame_start + num_frames]
            return batch_mod

        action_features = None
        if isinstance(action_inputs, dict):
            if "action_features" in action_inputs and action_inputs["action_features"] is not None:
                action_features = action_inputs["action_features"]
            elif "actions" in action_inputs and action_inputs["actions"] is not None:
                action_features = action_inputs["actions"]

        if action_features is None:
            return None

        action_features = action_features.to(device=device, dtype=dtype)

        if action_features.dim() == 2:
            # [B, action_dim] -> expand到当前块帧数
            action_features = action_features.unsqueeze(1).expand(-1, num_frames, -1)
        elif action_features.dim() == 3:
            if frame_start + num_frames > action_features.shape[1]:
                raise ValueError("action_features 的帧长度不足以覆盖当前块")
            action_features = action_features[:, frame_start:frame_start + num_frames]
        else:
            raise ValueError("action_features 需为 [B, action_dim] 或 [B, T, action_dim]")

        if not self.enable_adaln_zero or self.action_projection is None:
            return None

        modulation = self.action_projection(action_features, num_frames=num_frames)
        return modulation.to(device=device, dtype=dtype)

    def _prepare_action_conditional(
        self,
        base_conditional: dict[str, Any],
        action_inputs: Optional[dict[str, Any]],
        frame_start: int,
        num_frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        if action_inputs is None:
            return base_conditional

        modulation = self._compute_action_modulation(action_inputs, frame_start, num_frames, device, dtype)
        if modulation is None:
            return base_conditional

        conditional_dict = dict(base_conditional)
        conditional_dict["_action_modulation"] = modulation
        if "action_features" in action_inputs:
            conditional_dict["action_features"] = action_inputs["action_features"]
        return conditional_dict

    def set_default_action_inputs(self, action_inputs: Optional[dict[str, Any]]) -> None:
        self._default_action_inputs = action_inputs

    def generate_chunk_with_cache(
        self,
        noise: torch.Tensor,
        conditional_dict: dict[str, Any],
        *,
        current_start_frame: int = 0,
        requires_grad: bool = True,
        return_sim_step: bool = False,
        action_inputs: Optional[dict[str, Any]] = None,
    ):
        if action_inputs is None:
            action_inputs = self._default_action_inputs

        batch_size, chunk_frames, num_channels, height, width = noise.shape

        if chunk_frames % self.num_frame_per_block != 0:
            if not self.independent_first_frame or chunk_frames <= 1:
                raise AssertionError("chunk_frames 必须是 num_frame_per_block 的整数倍")

        if not self.independent_first_frame or chunk_frames % self.num_frame_per_block == 0:
            num_blocks = chunk_frames // self.num_frame_per_block
            all_num_frames = [self.num_frame_per_block] * num_blocks
        else:
            num_blocks = (chunk_frames - 1) // self.num_frame_per_block
            all_num_frames = [1] + [self.num_frame_per_block] * num_blocks

        output = torch.zeros_like(noise)

        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)

        if not requires_grad:
            start_gradient_frame_index = chunk_frames
        else:
            start_gradient_frame_index = 0

        local_start_frame = 0
        if not isinstance(self.local_attn_size, (list, tuple)):
            self.generator.model.local_attn_size = int(self.local_attn_size)
            self._set_all_modules_max_attention_size(int(self.local_attn_size))

        for block_index, current_num_frames in enumerate(all_num_frames):
            noisy_input = noise[:, local_start_frame:local_start_frame + current_num_frames]

            global_start = current_start_frame + local_start_frame
            cond_with_action = self._prepare_action_conditional(
                conditional_dict,
                action_inputs,
                global_start,
                current_num_frames,
                device=noise.device,
                dtype=noise.dtype,
            )

            denoised_pred = torch.zeros_like(noisy_input)
            timestep = torch.zeros(
                [batch_size, current_num_frames],
                device=noise.device,
                dtype=torch.int64,
            )

            for step_idx, current_timestep in enumerate(self.denoising_step_list):
                if isinstance(self.local_attn_size, (list, tuple)):
                    self.generator.model.local_attn_size = int(self.local_attn_size[step_idx])
                    self._set_all_modules_max_attention_size(int(self.local_attn_size[step_idx]))

                exit_flag = (
                    step_idx == exit_flags[0]
                    if self.same_step_across_blocks
                    else step_idx == exit_flags[block_index]
                )

                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64,
                ) * current_timestep

                if not exit_flag:
                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=cond_with_action,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=(current_start_frame + local_start_frame) * self.frame_seq_length,
                        )

                        if step_idx < len(self.denoising_step_list) - 1:
                            next_timestep = self.denoising_step_list[step_idx + 1]
                            noisy_input = self.scheduler.add_noise(
                                denoised_pred.flatten(0, 1),
                                torch.randn_like(denoised_pred.flatten(0, 1)),
                                next_timestep * torch.ones(
                                    [batch_size * current_num_frames],
                                    device=noise.device,
                                    dtype=torch.long,
                                ),
                            ).unflatten(0, denoised_pred.shape[:2])
                else:
                    enable_grad = local_start_frame >= start_gradient_frame_index
                    context_manager = torch.enable_grad() if enable_grad else torch.no_grad()
                    with context_manager:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=cond_with_action,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=(current_start_frame + local_start_frame) * self.frame_seq_length,
                        )
                    break

            output[:, local_start_frame:local_start_frame + current_num_frames] = denoised_pred

            context_timestep = torch.ones_like(timestep) * self.context_noise
            context_noisy = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep.flatten(0, 1),
            ).unflatten(0, denoised_pred.shape[:2])

            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=context_noisy,
                    conditional_dict=cond_with_action,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=(current_start_frame + local_start_frame) * self.frame_seq_length,
                )

            local_start_frame += current_num_frames

        scheduler_timesteps = cast(torch.Tensor, getattr(self.scheduler, "timesteps"))

        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (scheduler_timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0,
            ).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (scheduler_timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0,
            ).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (scheduler_timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0,
            ).item()

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1

        return output, denoised_timestep_from, denoised_timestep_to

    def inference_with_trajectory(
        self,
        noise: torch.Tensor,
        initial_latent: Optional[torch.Tensor] = None,
        return_sim_step: bool = False,
        slice_last_frames: int = 21,
        action_inputs: Optional[dict[str, Any]] = None,
        **conditional_dict,
    ):
        if action_inputs is None:
            action_inputs = self._default_action_inputs

        batch_size, num_frames, num_channels, height, width = noise.shape

        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block

        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype,
        )

        self._initialize_kv_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        self._initialize_crossattn_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)

        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.zeros([batch_size, 1], device=noise.device, dtype=torch.int64)
            cond_with_action = self._prepare_action_conditional(
                conditional_dict,
                action_inputs,
                frame_start=0,
                num_frames=initial_latent.shape[1],
                device=noise.device,
                dtype=initial_latent.dtype,
            )
            output[:, : initial_latent.shape[1]] = initial_latent
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=cond_with_action,
                    timestep=timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
            current_start_frame += initial_latent.shape[1]

        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames

        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)
        start_gradient_frame_index = num_output_frames - slice_last_frames

        grad_enable_mask = torch.zeros((batch_size, sum(all_num_frames)), dtype=torch.bool)

        if not isinstance(self.local_attn_size, (list, tuple)):
            self.generator.model.local_attn_size = int(self.local_attn_size)
            self._set_all_modules_max_attention_size(int(self.local_attn_size))

        for block_index, current_num_frames in enumerate(all_num_frames):
            offset = current_start_frame - num_input_frames
            noisy_input = noise[:, offset:offset + current_num_frames]

            global_start = current_start_frame
            cond_with_action = self._prepare_action_conditional(
                conditional_dict,
                action_inputs,
                frame_start=global_start,
                num_frames=current_num_frames,
                device=noise.device,
                dtype=noise.dtype,
            )

            denoised_pred = torch.zeros_like(noisy_input)
            timestep = torch.zeros(
                [batch_size, current_num_frames],
                device=noise.device,
                dtype=torch.int64,
            )

            for index, current_timestep in enumerate(self.denoising_step_list):
                if isinstance(self.local_attn_size, (list, tuple)):
                    self.generator.model.local_attn_size = int(self.local_attn_size[index])
                    self._set_all_modules_max_attention_size(int(self.local_attn_size[index]))

                exit_flag = index == exit_flags[0] if self.same_step_across_blocks else index == exit_flags[block_index]

                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64,
                ) * current_timestep

                if not exit_flag:
                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=cond_with_action,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                        )
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames],
                                device=noise.device,
                                dtype=torch.long,
                            ),
                        ).unflatten(0, denoised_pred.shape[:2])
                else:
                    if current_start_frame < start_gradient_frame_index:
                        grad_enable_mask[:, current_start_frame:current_start_frame + current_num_frames] = False
                        with torch.no_grad():
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=cond_with_action,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length,
                            )
                    else:
                        grad_enable_mask[:, current_start_frame:current_start_frame + current_num_frames] = True
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=cond_with_action,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                        )
                    break

            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            context_timestep = torch.ones_like(timestep) * self.context_noise
            denoised_pred_for_cache = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep * torch.ones(
                    [batch_size * current_num_frames],
                    device=noise.device,
                    dtype=torch.long,
                ),
            ).unflatten(0, denoised_pred.shape[:2])

            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred_for_cache,
                    conditional_dict=cond_with_action,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )

            current_start_frame += current_num_frames

        scheduler_timesteps = cast(torch.Tensor, getattr(self.scheduler, "timesteps"))

        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (scheduler_timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0,
            ).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (scheduler_timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0,
            ).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (scheduler_timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0,
            ).item()

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1

        return output, denoised_timestep_from, denoised_timestep_to

