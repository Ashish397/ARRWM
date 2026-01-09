# pyright: reportGeneralTypeIssues=false
from __future__ import annotations

from typing import Any, Optional, cast


from pipeline.self_forcing_training import SelfForcingTrainingPipeline
from model.action_modulation import ActionModulationProjection
from model.action_model_patch import apply_action_patches
from torch.cuda.amp import autocast
from utils.scheduler import SchedulerInterface
from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
from typing import List, Optional, Tuple, cast, Any
import torch
import torch.distributed as dist
from utils.debug_option import DEBUG, LOG_GPU_MEMORY
from utils.memory import log_gpu_memory


class ActionSelfForcingTrainingPipeline(SelfForcingTrainingPipeline):
    def __init__(self,
                 denoising_step_list: List[int],
                 scheduler: SchedulerInterface,
                 generator: WanDiffusionWrapper,
                 num_frame_per_block=3,
                 independent_first_frame: bool = False,
                 same_step_across_blocks: bool = False,
                 last_step_only: bool = False,
                 num_max_frames: int = 21,
                 context_noise: int = 0,
                 *,
                 action_dim: int = 512,
                 enable_adaln_zero: bool = True,
                 action_module: Optional[Any] = None,
                 action_projection: Optional[ActionModulationProjection] = None,
                 **kwargs):
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

        self.action_dim: int = action_dim
        self.enable_adaln_zero: bool = enable_adaln_zero
        self.action_module: Optional[Any] = action_module
        self.action_projection: Optional[ActionModulationProjection] = action_projection
        self._shared_action_projection: bool = action_projection is not None
        self._action_conditioning_enabled: bool = enable_adaln_zero

        if self._action_conditioning_enabled:
            _ = apply_action_patches(self.generator)

        if self.enable_adaln_zero and self._action_conditioning_enabled and self.action_projection is None:
            # Match the generator's primary dtype/device so we avoid dtype mismatches
            first_param = next(self.generator.parameters(), None)
            if first_param is None:
                raise RuntimeError("Generator must have parameters to derive dtype/device for action projection.")
            device = first_param.device
            dtype = first_param.dtype
            self.action_projection = ActionModulationProjection(
                action_dim=self.action_dim,
                hidden_dim=self.generator.model.dim,
                num_frames=1,
                zero_init=True,
            ).to(device=device, dtype=dtype)

    def _compute_action_modulation(
        self,
        action_inputs: Any,
        frame_start: int,
        num_frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not self._action_conditioning_enabled or action_inputs is None:
            raise ValueError("action_inputs is required / action_conditioning_enabled is disabled")

        if isinstance(action_inputs, dict):
            full_cached = action_inputs.get("_cached_action_modulation_full") or action_inputs.get("_cached_action_modulation")
            provided_mod = action_inputs.get("action_modulation")
            if full_cached is not None:
                mod = full_cached.to(device=device, dtype=dtype)
                if mod.dim() != 4 or mod.shape[2] != 6:
                    raise ValueError("action_modulation must be [B, T, 6, hidden_dim]")
                stop = frame_start + num_frames
                if stop > mod.shape[1]:
                    raise ValueError("Cached action_modulation is shorter than required block")
                return mod[:, frame_start:stop]
            if provided_mod is not None:
                mod = provided_mod.to(device=device, dtype=dtype)
                if mod.dim() != 4 or mod.shape[2] != 6:
                    raise ValueError("action_modulation must be [B, T, 6, hidden_dim]")
                stop = frame_start + num_frames
                if stop > mod.shape[1]:
                    raise ValueError("action_modulation 的帧长度不足以覆盖当前块")
                action_inputs["_cached_action_modulation_full"] = mod
                return mod[:, frame_start:stop]
            action_features = action_inputs.get("action_features") or action_inputs.get("actions")
            if action_features is None:
                raise ValueError("action_inputs must provide 'action_features'/'actions'")
        else:
            action_features = action_inputs

        action_features = action_features.to(device=device, dtype=dtype)
        if action_features.dim() == 1:
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"[ActionPipeline] Detected 1-D action vector (len={action_features.shape[0]}). Expanding to batch=1.")
            action_features = action_features.unsqueeze(0)

        if action_features.dim() == 2:
            # Treat a 2D tensor as an already-selected action for the current chunk.
            # Do not broadcast it across frames; the downstream Wan patch will match
            # the single-frame modulation to whichever timesteps are active.
            action_features = action_features.unsqueeze(1)
        elif action_features.dim() == 3:
            action_features = action_features[:, frame_start:frame_start + num_frames]
        else:
            raise ValueError("action_features 需为 [B, action_dim] 或 [B, T, action_dim]")

        if not self.enable_adaln_zero or self.action_projection is None:
            raise ValueError("action_projection is unavailable; supply precomputed 'action_modulation' instead of features")

        self._ensure_action_projection_dtype(device=device, dtype=dtype)
        if (not dist.is_initialized() or dist.get_rank() == 0) and DEBUG:
            print(f"[ActionPipeline] Feeding action_features with shape {action_features.shape} into action_projection (num_frames={num_frames}).")
        modulation = self.action_projection(action_features, num_frames=num_frames)
        return modulation.to(device=device, dtype=dtype)

    def _ensure_action_projection_dtype(self, device: torch.device, dtype: torch.dtype) -> None:
        if self.action_projection is None or self._shared_action_projection:
            return
        param = next(self.action_projection.parameters(), None)
        if param is None:
            return
        if param.device == device and param.dtype == dtype:
            return
        self.action_projection = self.action_projection.to(device=device, dtype=dtype)

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
            raise ValueError("action_inputs is required")

        modulation = self._compute_action_modulation(action_inputs, frame_start, num_frames, device, dtype)
        if modulation is None:
            raise ValueError("action_modulation is required")

        conditional_dict = dict(base_conditional)
        conditional_dict["_action_modulation"] = modulation
        return conditional_dict

    def generate_chunk_with_cache(
        self,
        noise: torch.Tensor,
        conditional_dict: dict,
        *,
        current_start_frame: int = 0,
        requires_grad: bool = True,
        return_sim_step: bool = False,
        action_inputs: Optional[dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Optional[int], Optional[int]]:
        """
        Chunk generation method tailored for sequential training
        
        Args:
            noise: noise tensor for a single chunk [batch_size, chunk_frames, C, H, W]
            conditional_dict: dictionary of conditional information
            kv_cache: externally provided KV cache (defaults to self.kv_cache1 if None)
            crossattn_cache: externally provided cross-attention cache (defaults to self.crossattn_cache if None)
            current_start_frame: start frame index of the chunk in the full sequence
            requires_grad: whether gradients are required
            return_sim_step: whether to return simulation step info
            action_inputs: Dictionary of action inputs
            
        Returns:
            output: generated chunk [batch_size, chunk_frames, C, H, W]
            denoised_timestep_from: starting denoise timestep
            denoised_timestep_to: ending denoise timestep
        """
        batch_size, chunk_frames, num_channels, height, width = noise.shape
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Pipeline] generate_chunk_with_cache: batch_size={batch_size}, chunk_frames={chunk_frames}")
            print(f"[SeqTrain-Pipeline] current_start_frame={current_start_frame}, requires_grad={requires_grad}")
        
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"SeqTrain-Pipeline: Before chunk generation", device=noise.device, rank=dist.get_rank() if dist.is_initialized() else 0)
        
        # Compute block configuration
        if not self.independent_first_frame or chunk_frames % self.num_frame_per_block == 0:
            assert chunk_frames % self.num_frame_per_block == 0
            num_blocks = chunk_frames // self.num_frame_per_block
            all_num_frames = [self.num_frame_per_block] * num_blocks
        else:
            # Handle the case of an independent first frame
            assert (chunk_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (chunk_frames - 1) // self.num_frame_per_block
            all_num_frames = [1] + [self.num_frame_per_block] * num_blocks
            
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Pipeline] Block config: num_blocks={num_blocks}, all_num_frames={all_num_frames}")
            print(f"[SeqTrain-Pipeline] independent_first_frame={self.independent_first_frame}")
            
        # Prepare output tensor
        output = torch.zeros_like(noise)

        # Randomly select denoising steps (synced across ranks)
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Pipeline] Denoising steps: {num_denoising_steps}, exit_flags: {exit_flags}")
        
        # Determine gradient-enabled range — disable everywhere when requires_grad=False
        if not requires_grad:
            start_gradient_frame_index = chunk_frames  # Out of range: no gradients anywhere
        else:
            start_gradient_frame_index = 0

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Pipeline] start_gradient_frame_index={start_gradient_frame_index}")
        
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"SeqTrain-Pipeline: Before block generation loop", device=noise.device, rank=dist.get_rank() if dist.is_initialized() else 0)
        
        # Generate block by block
        local_start_frame = 0
        # If static local_attn_size, set it on the model before the step loop
        if not (isinstance(self.local_attn_size, (list, tuple)) or (hasattr(self.local_attn_size, "__iter__") and not isinstance(self.local_attn_size, (str, bytes)))):
            self.generator.model.local_attn_size = int(self.local_attn_size)
            self._set_all_modules_max_attention_size(int(self.local_attn_size))
        for block_index, current_num_frames in enumerate(all_num_frames):
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Pipeline] Processing block {block_index}: frames {local_start_frame}-{local_start_frame + current_num_frames}")
            
            if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY and block_index == 0:
                log_gpu_memory(f"SeqTrain-Pipeline: Before first block generation", device=noise.device, rank=dist.get_rank() if dist.is_initialized() else 0)
                
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
            
            # Spatial denoising loop
            for step_idx, current_timestep in enumerate(self.denoising_step_list):
                # If scheduled, set local_attn_size dynamically per timestep
                if isinstance(self.local_attn_size, (list, tuple)) or (hasattr(self.local_attn_size, "__iter__") and not isinstance(self.local_attn_size, (str, bytes))):
                    self.generator.model.local_attn_size = int(self.local_attn_size[step_idx])
                    if (not dist.is_initialized() or dist.get_rank() == 0) and DEBUG:
                        print(f"[denoise step {step_idx}] timestep={float(current_timestep)} local_attn_size={self.generator.model.local_attn_size}")
                    self._set_all_modules_max_attention_size(int(self.local_attn_size[step_idx]))
                exit_flag = (
                    step_idx == exit_flags[0]
                    if self.same_step_across_blocks
                    else step_idx == exit_flags[block_index]
                )

                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64
                ) * current_timestep

                if not exit_flag:
                    # Intermediate steps: no gradients
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[SeqTrain-Pipeline] Block {block_index} intermediate steps (no grad)")
                        
                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=cond_with_action,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=(current_start_frame + local_start_frame) * self.frame_seq_length,
                        )
                        
                        # Add noise for the next step
                        if step_idx < len(self.denoising_step_list) - 1:
                            next_timestep = self.denoising_step_list[step_idx + 1]
                            noisy_input = self.scheduler.add_noise(
                                denoised_pred.flatten(0, 1),
                                torch.randn_like(denoised_pred.flatten(0, 1)),
                                next_timestep * torch.ones(
                                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long
                                ),
                            ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # Final step may require gradients
                    enable_grad = local_start_frame >= start_gradient_frame_index
                    
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[SeqTrain-Pipeline] Block {block_index} final step: enable_grad={enable_grad}")
                    
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
            
            # Record output
            output[:, local_start_frame:local_start_frame + current_num_frames] = denoised_pred
            
            # Update cache with context noise
            context_timestep = torch.ones_like(timestep) * self.context_noise
            context_noisy = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep.flatten(0, 1),
            ).unflatten(0, denoised_pred.shape[:2])
            
            if DEBUG and block_index == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Pipeline] Updating cache with context_noise={self.context_noise}")
            
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
        
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"SeqTrain-Pipeline: After all blocks generated", device=noise.device, rank=dist.get_rank() if dist.is_initialized() else 0)
        
        # Compute returned timestep information
        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0
            ).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0
            ).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0
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
            **conditional_dict
    ) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Step 1: Initialize KV cache to all zeros
        self._initialize_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            cond_with_action = self._prepare_action_conditional(
                conditional_dict,
                action_inputs,
                frame_start=0,
                num_frames=initial_latent.shape[1],
                device=noise.device,
                dtype=initial_latent.dtype,
            )
            # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
            output[:, :1] = initial_latent
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=cond_with_action,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )
            current_start_frame += 1

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)
        start_gradient_frame_index = num_output_frames - slice_last_frames

        grad_enable_mask = torch.zeros((batch_size, sum(all_num_frames)), dtype=torch.bool)
        # If static local_attn_size, set it first
        if not isinstance(self.local_attn_size, (list, tuple)):
            self.generator.model.local_attn_size = int(self.local_attn_size)
            self._set_all_modules_max_attention_size(int(self.local_attn_size))
        # for block_index in range(num_blocks):
        for block_index, current_num_frames in enumerate(all_num_frames):
            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            global_start = current_start_frame
            cond_with_action = self._prepare_action_conditional(
                conditional_dict,
                action_inputs,
                frame_start=global_start,
                num_frames=current_num_frames,
                device=noise.device,
                dtype=noise.dtype,
            )

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                # If scheduled, set local_attn_size dynamically per timestep
                if isinstance(self.local_attn_size, (list, tuple)):
                    self.generator.model.local_attn_size = int(self.local_attn_size[index])
                    if not dist.is_initialized() or dist.get_rank() == 0 and DEBUG:
                        print(f"[denoise step {index}] timestep={float(current_timestep)} local_attn_size={self.generator.model.local_attn_size}")
                    self._set_all_modules_max_attention_size(int(self.local_attn_size[index]))
                if self.same_step_across_blocks:
                    exit_flag = (index == exit_flags[0])
                else:
                    exit_flag = (index == exit_flags[block_index])  # Only backprop at the randomly selected timestep (consistent across all ranks)
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep
                if DEBUG and dist.get_rank() == 0:
                    print(f"rank {dist.get_rank()}, current_start_frame: {current_start_frame}, current_num_frames: {current_num_frames}, current_timestep: {current_timestep}")
                if not exit_flag:
                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=cond_with_action,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    # with torch.set_grad_enabled(current_start_frame >= start_gradient_frame_index):
                    if current_start_frame < start_gradient_frame_index:
                        grad_enable_mask[:, current_start_frame:current_start_frame + current_num_frames] = False
                        with torch.no_grad():
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=cond_with_action,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length
                            )
                    else:
                        # print(f"enable grad: {current_start_frame}")
                        grad_enable_mask[:, current_start_frame:current_start_frame + current_num_frames] = True
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=cond_with_action,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                    break

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update the cache
            context_timestep = torch.ones_like(timestep) * self.context_noise
            # add context noise
            denoised_pred = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep * torch.ones(
                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
            ).unflatten(0, denoised_pred.shape[:2])
            if DEBUG and dist.get_rank() == 0:
                print(f"rank {dist.get_rank()}, current_start_frame: {current_start_frame}, current_num_frames: {current_num_frames}, current_timestep: {current_timestep}")
                print(f"rank {dist.get_rank()}, rerun_for_cache")
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=cond_with_action,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if dist.get_rank() == 0 and DEBUG:
            print(f"grad_enable_mask: {grad_enable_mask[0, :]}")
            
        # Step 3.5: Return the denoised timestep
        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1

        return output, denoised_timestep_from, denoised_timestep_to

    def inference_with_trajectory_for_flow(
            self,
            initial_latent: Optional[torch.Tensor] = None,
            slice_last_frames: int = 21,
            action_inputs: Optional[dict[str, Any]] = None,
            real_latents_x0: Optional[torch.Tensor] = None,
            **conditional_dict,
    ) -> torch.Tensor:
        flow_losses: list[torch.Tensor] = []

        if real_latents_x0 is None:
            raise ValueError("real_latents_x0 is required for inference_with_trajectory_for_flow")
        
        batch_size, num_frames, num_channels, height, width = real_latents_x0.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block

        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames

        device = torch.cuda.current_device()

        # Step 1: Initialize KV cache to all zeros
        self._initialize_kv_cache(
            batch_size=batch_size, dtype=torch.float32, device=device
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=torch.float32, device=device
        )
        # Step 2: Cache context feature
        current_start_frame = 0

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        num_denoising_steps = len(self.denoising_step_list)
        exit_indices = torch.tensor(
            self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=device),
            device=device,
            dtype=torch.long,
        )
        if self.same_step_across_blocks:
            exit_indices = exit_indices[0].repeat(len(all_num_frames))
        
        start_gradient_frame_index = num_output_frames - slice_last_frames

        grad_enable_mask = torch.zeros((batch_size, sum(all_num_frames)), dtype=torch.bool)
        # If static local_attn_size, set it first
        if not isinstance(self.local_attn_size, (list, tuple)):
            self.generator.model.local_attn_size = int(self.local_attn_size)
            self._set_all_modules_max_attention_size(int(self.local_attn_size))
        # for block_index in range(num_blocks):
                
        # Construct real_xt (xt_exit_full)
        if real_latents_x0.shape[1] < num_output_frames:
            raise ValueError(
                f"real_latents_x0 has insufficient frames: need {num_output_frames}, "
                f"got {real_latents_x0.shape[1]}."
            )
        real_latents_window = real_latents_x0.to(device=device, dtype=torch.float32)
        
        for block_index, current_num_frames in enumerate(all_num_frames):
            noiseless_input = real_latents_window[
                :, current_start_frame:current_start_frame + current_num_frames]

            global_start = current_start_frame
            cond_with_action = self._prepare_action_conditional(
                conditional_dict,
                action_inputs,
                frame_start=global_start,
                num_frames=current_num_frames,
                device=device,
                dtype=torch.float32,
            )

            this_noise = torch.randn_like(noiseless_input, dtype=torch.float32)
            noisy_input = self.scheduler.add_noise(
                noiseless_input.flatten(0, 1),
                this_noise.flatten(0, 1),
                1000 * torch.ones([batch_size * current_num_frames], device=device, dtype=torch.long),
                ).view_as(noiseless_input)
            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                # If scheduled, set local_attn_size dynamically per timestep
                if isinstance(self.local_attn_size, (list, tuple)):
                    self.generator.model.local_attn_size = int(self.local_attn_size[index])
                    self._set_all_modules_max_attention_size(int(self.local_attn_size[index]))
                exit_flag = (index == exit_indices[block_index].item())  # Only backprop at the randomly selected timestep (consistent across all ranks)
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=device,
                    dtype=torch.int64) * current_timestep
                
                if not exit_flag:
                    with torch.no_grad():
                        with autocast(dtype=torch.bfloat16, enabled=True):
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=cond_with_action,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length
                            )
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1).to(dtype=torch.float32),
                            this_noise.flatten(0, 1),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames], device=device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    # with torch.set_grad_enabled(current_start_frame >= start_gradient_frame_index):
                    if current_start_frame < start_gradient_frame_index:
                        grad_enable_mask[:, current_start_frame:current_start_frame + current_num_frames] = False
                        with torch.no_grad():
                            with autocast(dtype=torch.bfloat16, enabled=True):
                                _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=cond_with_action,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length
                                )
                    else:
                        with autocast(dtype=torch.bfloat16, enabled=True):
                            flow_pred, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=cond_with_action,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length
                            )
                        err = ((flow_pred - (this_noise - noiseless_input)) ** 2).mean()
                        flow_losses.append(err)
                    break

            # Step 3.3: rerun with timestep zero to update the cache
            context_timestep = torch.ones_like(timestep) * self.context_noise
            # add context noise
            denoised_pred = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1).to(dtype=torch.float32),
                torch.randn_like(denoised_pred.flatten(0, 1), dtype=torch.float32),
                context_timestep * torch.ones(
                    [batch_size * current_num_frames], device=device, dtype=torch.long)
            ).unflatten(0, denoised_pred.shape[:2])
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=cond_with_action,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames
            
        return torch.stack(flow_losses).mean()
