# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
from __future__ import annotations

from typing import Any, Optional
import torch

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation
from pipeline.causal_inference import CausalInferencePipeline
from model.action_modulation import ActionModulationProjection
from model.action_model_patch import apply_action_patches
import torch.distributed as dist


class ActionCausalInferencePipeline(CausalInferencePipeline):
    """
    Action-conditioned inference pipeline that extends CausalInferencePipeline.
    
    This pipeline allows inserting action modules during inference to guide video generation.
    """

    def __init__(
        self,
        args,
        device,
        *,
        generator: WanDiffusionWrapper | None = None,
        text_encoder: WanTextEncoder | None = None,
        vae: WanVAEWrapper | None = None,
        action_module=None,  # 动作模块，可以是任何模型
        action_dim: Optional[int] = None,  # 动作特征维度
        enable_adaln_zero: bool = True,  # 是否使用 adaLN-Zero 注入
        action_projection: Optional[ActionModulationProjection] = None,
    ):
        super().__init__(args, device, generator=generator, text_encoder=text_encoder, vae=vae)
        
        # 初始化动作相关的配置
        if action_dim is not None:
            resolved_action_dim = action_dim
        else:
            resolved_action_dim = int(getattr(args, "raw_action_dim", getattr(args, "action_dim", 2)))
        self.action_module = action_module
        self.enable_adaln_zero = enable_adaln_zero
        self.action_dim = resolved_action_dim
        self.action_projection: Optional[ActionModulationProjection] = action_projection
        self._shared_action_projection = action_projection is not None
        self._action_conditioning_enabled = (
            self.enable_adaln_zero or self.action_module is not None or action_projection is not None
        )
        self.action_norm_epsilon = float(getattr(args, "action_norm_epsilon", 0.0) or 0.0)
        # Backwards compatibility for external checks.
        self.use_action_conditioning = self._action_conditioning_enabled

        if self._action_conditioning_enabled:
            _ = apply_action_patches(self.generator)

        if self.enable_adaln_zero and self.action_projection is None:
            device = self._resolve_module_device(self.generator)
            model_dim = getattr(self.generator.model, "dim", 2048)
            self.action_projection = ActionModulationProjection(
                action_dim=self.action_dim,
                hidden_dim=model_dim,
                num_frames=1,
                zero_init=True,
            ).to(device)
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"[ActionPipeline] adaLN-Zero enabled: action_dim={self.action_dim}, model_dim={model_dim}")
        elif self.enable_adaln_zero and self.action_projection is not None:
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"[ActionPipeline] Reusing shared action_projection (dim={self.action_dim}).")

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"[ActionPipeline] Action conditioning: {self._action_conditioning_enabled}")

    def _canonicalize_action_inputs(self, action_inputs: Any) -> dict[str, Any]:
        if action_inputs is None:
            raise ValueError("action_inputs is required for action-conditioned inference")
        if isinstance(action_inputs, dict):
            if not action_inputs:
                raise ValueError("action_inputs dictionary is empty; provide action data")
            return dict(action_inputs)
        if torch.is_tensor(action_inputs):
            return {"action_features": action_inputs}
        raise TypeError(
            f"Unsupported action_inputs type '{type(action_inputs)}'; expected dict or torch.Tensor"
        )

    def _assert_non_zero_modulation(self, modulation: torch.Tensor, *, source: str = "action_modulation") -> None:
        if getattr(self, "debug_allow_zero_modulation", False):
            return
        if modulation.numel() == 0:
            raise ValueError(f"{source} tensor is empty")
        flat = modulation.detach().float().reshape(modulation.shape[0], -1)
        threshold = max(self.action_norm_epsilon, 0.0)
        zero_mask = flat.norm(dim=1) <= threshold
        if torch.any(zero_mask):
            idx_list = torch.nonzero(zero_mask, as_tuple=False).flatten().tolist()
            raise ValueError(
                f"{source} contains zero-norm entries for sample indices {idx_list}; "
                "action-conditioned inference requires non-zero modulation"
            )

    def _coerce_action_modulation_tensor(
        self,
        modulation: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        debug_allow_zero_modulation: bool = False,
    ) -> torch.Tensor:
        mod = modulation.to(device=device, dtype=dtype)
        if mod.dim() == 3:
            if mod.shape[2] % 6 != 0:
                raise ValueError(
                    "Flattened action_modulation last dimension must be divisible by 6 "
                    f"(got {mod.shape[2]})"
                )
            hidden = mod.shape[2] // 6
            mod = mod.view(mod.shape[0], mod.shape[1], 6, hidden)
        if mod.dim() != 4:
            raise ValueError(
                "action_modulation must be shaped as [B, num_frames, 6, hidden_dim] after coercion"
            )
        if mod.shape[2] != 6:
            raise ValueError(f"action_modulation expected 6 modulation slots, got {mod.shape[2]}")
        if not debug_allow_zero_modulation:
            self._assert_non_zero_modulation(mod)
        return mod

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
        *,
        debug_allow_zero_modulation: bool = False,
    ) -> Optional[torch.Tensor]:
        if not self._action_conditioning_enabled:
            raise RuntimeError("Action conditioning is disabled; enable AdaLN-Zero or provide an action module")
        if action_inputs is None:
            raise ValueError("action_inputs is required")

        full_cached = action_inputs.get("_cached_action_modulation_full") or action_inputs.get("_cached_action_modulation")
        if full_cached is not None:
            mod = self._coerce_action_modulation_tensor(
                full_cached,
                device,
                dtype,
                debug_allow_zero_modulation=debug_allow_zero_modulation,
            )
            if frame_start + num_frames > mod.shape[1]:
                raise ValueError("cached action_modulation is shorter than required block")
            return mod[:, frame_start:frame_start + num_frames]

        provided_mod = action_inputs.get("action_modulation")
        if provided_mod is not None:
            mod = self._coerce_action_modulation_tensor(
                provided_mod,
                device,
                dtype,
                debug_allow_zero_modulation=debug_allow_zero_modulation,
            )
            if frame_start + num_frames > mod.shape[1]:
                raise ValueError("action_modulation 的帧长度不足以覆盖当前块")
            action_inputs["_cached_action_modulation_full"] = mod
            return mod[:, frame_start:frame_start + num_frames]

        action_features = action_inputs.get("action_features")
        if action_features is None:
            action_features = action_inputs.get("actions")
        if action_features is None:
            raise ValueError("action_inputs must include 'action_features'/'actions' or 'action_modulation'")

        action_features = action_features.to(device=device, dtype=dtype)

        if action_features.dim() == 2:
            # Already-selected actions target the current chunk; keep them as single-frame
            # inputs instead of tiling across num_frames.
            action_features = action_features.unsqueeze(1)
        elif action_features.dim() == 3:
            stop = frame_start + num_frames
            if stop > action_features.shape[1]:
                raise ValueError("action_features 的帧长度不足以覆盖当前块")
            action_features = action_features[:, frame_start:stop]
        else:
            raise ValueError("action_features 需为 [B, action_dim] 或 [B, T, action_dim]")

        if action_features.shape[-1] != self.action_dim:
            raise ValueError(
                f"action_features last dimension ({action_features.shape[-1]}) != configured action_dim ({self.action_dim})"
            )

        if not self.enable_adaln_zero or self.action_projection is None:
            raise RuntimeError(
                "action_projection is unavailable; supply precomputed 'action_modulation' instead of features"
            )

        modulation = self.action_projection(action_features, num_frames=num_frames)
        modulation = modulation.to(device=device, dtype=dtype)
        if not debug_allow_zero_modulation:
            self._assert_non_zero_modulation(modulation)
        return modulation

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
            raise ValueError("action_inputs is required for conditional preparation")

        modulation = self._compute_action_modulation(action_inputs, frame_start, num_frames, device, dtype)

        conditional_dict = dict(base_conditional)
        conditional_dict["_action_modulation"] = modulation
        if "action_features" in action_inputs:
            conditional_dict["action_features"] = action_inputs["action_features"]
        return conditional_dict

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: list[str] | None = None,
        *,
        prompt_embeds: torch.Tensor | None = None,
        action_inputs: Any = None,  # 新增：动作相关的输入
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
    ) -> torch.Tensor:
        """
        执行动作条件化的推理。
        
        Args:
            noise: 输入噪声 [batch_size, num_output_frames, num_channels, height, width]
            text_prompts: 文本提示列表（当 text_pre_encoded=False 时使用）
            prompt_embeds: 预编码的文本嵌入（当 text_pre_encoded=True 时使用）
            action_inputs: 必填的动作输入。可为 torch.Tensor（形状 [B, action_dim] 或 [B, T, action_dim]）
                或包含 'action_features' / 'actions' / 'action_modulation' 的字典。
            return_latents: 是否返回潜在表示
            profile: 是否进行性能分析
            low_memory: 是否使用低内存模式
            
        Returns:
            video: 生成的视频张量 [batch_size, num_output_frames, 3, height, width]
        """
        if not self._action_conditioning_enabled:
            raise RuntimeError("Action conditioning is disabled; enable AdaLN-Zero before using this pipeline")
        base_action_inputs = self._canonicalize_action_inputs(action_inputs)

        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block

        # 编码文本提示或使用预编码的嵌入
        if not self.text_pre_encoded:
            if text_prompts is None:
                raise ValueError("text_prompts must be provided when text_pre_encoded is False")
            conditional_dict = self.text_encoder(text_prompts=text_prompts)

            if low_memory and self.text_encoder is not None:
                gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
                move_model_to_device_with_memory_preservation(
                    self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation
                )
        else:
            if prompt_embeds is None:
                raise ValueError("prompt_embeds must be provided when text_pre_encoded is True")
            if prompt_embeds.dim() == 2:
                prompt_embeds = prompt_embeds.unsqueeze(0)
            target_device = noise.device if not low_memory else torch.device("cpu")
            conditional_dict = {
                "prompt_embeds": prompt_embeds.to(device=target_device, dtype=noise.dtype)
            }

        # 决定输出设备
        output_device = torch.device('cpu') if low_memory else noise.device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=output_device,
            dtype=noise.dtype
        )

        # 初始化 KV cache
        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        if local_attn_cfg != -1:
            kv_cache_size = local_attn_cfg * self.frame_seq_length
            kv_policy = f"local, size={local_attn_cfg}"
        else:
            kv_cache_size = num_output_frames * self.frame_seq_length
            kv_policy = "global (-1)"
        
        print(f"[ActionInference] kv_cache_size: {kv_cache_size} (policy: {kv_policy})")

        self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device,
            kv_cache_size_override=kv_cache_size
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device
        )

        current_start_frame = 0
        self.generator.model.local_attn_size = self.local_attn_size
        self._set_all_modules_max_attention_size(self.local_attn_size)

        base_conditional_dict = conditional_dict

        # 时序去噪循环（逐块生成）
        all_num_frames = [self.num_frame_per_block] * num_blocks
        
        for block_idx, current_num_frames in enumerate(all_num_frames):
            print(f"\n[Block {block_idx}] Generating frames {current_start_frame} to {current_start_frame + current_num_frames}")

            block_action_inputs = dict(base_action_inputs)

            cond_with_action = self._prepare_action_conditional(
                base_conditional=base_conditional_dict,
                action_inputs=block_action_inputs,
                frame_start=current_start_frame,
                num_frames=current_num_frames,
                device=noise.device,
                dtype=noise.dtype,
            )

            # 准备输入噪声
            noisy_input = noise[:, current_start_frame:current_start_frame + current_num_frames]

            # ============ 空间去噪循环 ============
            for index, current_timestep in enumerate(self.denoising_step_list):
                # 设置当前时间步
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64
                ) * current_timestep

                # ============ 动作模块插入点 2: 动态调整去噪过程 ============
                # 你可以在这里根据动作信息动态调整条件或者噪声
                # 例如：在特定时间步应用更强的动作引导
                
                if index < len(self.denoising_step_list) - 1:
                    # 中间去噪步骤
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_with_action,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )
                    
                    # 添加噪声到下一步
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], 
                            device=noise.device, 
                            dtype=torch.long
                        )
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # 最后一步，获取干净输出
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_with_action,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )

            # 记录模型输出
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred.to(output.device)
            
            # ============ 动作模块插入点 3: 后处理生成的帧 ============
            # 你可以在这里对生成的帧进行后处理或验证
            # 例如：检查是否符合动作约束，如果不符合可以进行调整
            
            # 使用干净的context更新KV cache
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=cond_with_action,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            # 更新起始帧索引
            current_start_frame += current_num_frames

        # 解码到像素空间
        video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if return_latents:
            return video, output.to(noise.device)
        else:
            return video
