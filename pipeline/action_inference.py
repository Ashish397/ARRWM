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
        action_dim: int = 512,  # 动作特征维度
        enable_adaln_zero: bool = True,  # 是否使用 adaLN-Zero 注入
    ):
        super().__init__(args, device, generator=generator, text_encoder=text_encoder, vae=vae)
        
        # 初始化动作相关的配置
        self.action_module = action_module
        self.enable_adaln_zero = enable_adaln_zero
        self.action_dim = action_dim
        self.action_projection: Optional[ActionModulationProjection] = None
        self._action_conditioning_enabled = self.enable_adaln_zero or self.action_module is not None
        # Backwards compatibility for external checks.
        self.use_action_conditioning = self._action_conditioning_enabled

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

            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"[ActionPipeline] adaLN-Zero enabled: action_dim={self.action_dim}, model_dim={model_dim}")

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"[ActionPipeline] Action conditioning: {self._action_conditioning_enabled}")

    def _process_action(
        self,
        generated_frames: torch.Tensor,
        current_frame_idx: int,
        action_inputs: dict | None = None,
    ) -> torch.Tensor | None:
        """
        处理动作模块，基于已生成的帧预测/提取动作特征。
        
        Args:
            generated_frames: 已生成的帧 [batch_size, num_frames, channels, height, width]
            current_frame_idx: 当前帧索引
            action_inputs: 额外的动作输入（例如：目标位置、动作序列等）
            
        Returns:
            action_features: 动作特征，用于条件化后续生成
        """
        if not self.use_action_conditioning or self.action_module is None:
            return None
        
        # TODO: 在这里实现你的动作处理逻辑
        # 示例：
        # with torch.no_grad():
        #     action_features = self.action_module(
        #         frames=generated_frames,
        #         frame_idx=current_frame_idx,
        #         **action_inputs
        #     )
        # return action_features
        
        print(f"[Action Module] Processing action at frame {current_frame_idx}")
        return None

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

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: list[str] | None = None,
        *,
        prompt_embeds: torch.Tensor | None = None,
        action_inputs: dict | None = None,  # 新增：动作相关的输入
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
            action_inputs: 动作相关的输入字典，可以包含：
                - 'target_actions': 目标动作序列
                - 'action_embeddings': 预先计算的动作嵌入
                - 'control_signals': 控制信号等
            return_latents: 是否返回潜在表示
            profile: 是否进行性能分析
            low_memory: 是否使用低内存模式
            
        Returns:
            video: 生成的视频张量 [batch_size, num_output_frames, 3, height, width]
        """
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
        base_action_inputs = ({k: v for k, v in action_inputs.items()} if isinstance(action_inputs, dict) else None)

        # 时序去噪循环（逐块生成）
        all_num_frames = [self.num_frame_per_block] * num_blocks
        
        for block_idx, current_num_frames in enumerate(all_num_frames):
            print(f"\n[Block {block_idx}] Generating frames {current_start_frame} to {current_start_frame + current_num_frames}")

            block_action_inputs = (
                {k: v for k, v in base_action_inputs.items()} if base_action_inputs is not None else None
            )

            inferred_action_features: Optional[torch.Tensor] = None
            if (
                current_start_frame > 0
                and self._action_conditioning_enabled
                and self.action_module is not None
            ):
                historical_frames = output[:, :current_start_frame].to(noise.device)
                inferred_action_features = self._process_action(
                    generated_frames=historical_frames,
                    current_frame_idx=current_start_frame,
                    action_inputs=action_inputs,
                )

            if inferred_action_features is not None:
                if block_action_inputs is None:
                    block_action_inputs = {}
                block_action_inputs["action_features"] = inferred_action_features

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
