# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA-4.0).
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

"""
Action-conditioned Wan model wrapper.

Extends WanDiffusionWrapper to support action conditioning via adaLN-Zero injection.
"""

from typing import Optional, List
import torch
from utils.wan_wrapper import WanDiffusionWrapper
from model.action_modulation import ActionModulationProjection


class ActionWanDiffusionWrapper(WanDiffusionWrapper):
    """
    Extended WanDiffusionWrapper that supports action conditioning.
    
    This wrapper injects action modulation parameters into the model's
    forward pass without modifying the underlying architecture.
    """
    
    def __init__(
        self,
        action_dim: int | None = None,
        enable_action_conditioning: bool = True,
        action_modulation_scale: float = 1.0,
        **kwargs
    ):
        """
        Args:
            action_dim: dimension of action features (required if enable_action_conditioning=True)
            enable_action_conditioning: whether to enable action conditioning
            action_modulation_scale: scaling factor for action modulation
            **kwargs: arguments passed to WanDiffusionWrapper
        """
        super().__init__(**kwargs)
        
        self.enable_action_conditioning = enable_action_conditioning and (action_dim is not None)
        self.action_modulation_scale = action_modulation_scale
        
        if self.enable_action_conditioning:
            # Get model hidden dimension
            if hasattr(self.model, 'dim'):
                model_dim = self.model.dim
            else:
                model_dim = 2048  # default for Wan-1.3B
            
            # Create action modulation projection
            self.action_projection = ActionModulationProjection(
                action_dim=action_dim,
                hidden_dim=model_dim,
                num_frames=1,
                zero_init=True,  # adaLN-Zero initialization
            )
            
            print(f"[ActionWanWrapper] Enabled action conditioning: action_dim={action_dim}, model_dim={model_dim}")
        else:
            self.action_projection = None
            print(f"[ActionWanWrapper] Action conditioning disabled")
    
    def _inject_action_into_model(
        self,
        action_features: torch.Tensor | None,
        num_frames: int,
    ) -> torch.Tensor | None:
        """
        Generate action modulation parameters.
        
        Args:
            action_features: [batch_size, action_dim] or [batch_size, num_frames, action_dim]
            num_frames: number of frames
            
        Returns:
            action_modulation: [batch_size, num_frames, 6, hidden_dim] or None
        """
        if not self.enable_action_conditioning or action_features is None:
            return None
        
        # Generate modulation parameters
        action_mod = self.action_projection(action_features, num_frames=num_frames)
        
        # Apply scaling
        if self.action_modulation_scale != 1.0:
            action_mod = action_mod * self.action_modulation_scale
        
        return action_mod
    
    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        classify_mode: Optional[bool] = False,
        concat_time_embeddings: Optional[bool] = False,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        cache_start: Optional[int] = None,
        action_features: Optional[torch.Tensor] = None,  # NEW: action features
    ) -> torch.Tensor:
        """
        Forward pass with optional action conditioning.
        
        Args:
            noisy_image_or_video: [batch_size, num_frames, channels, height, width]
            conditional_dict: dictionary containing 'prompt_embeds' and optionally 'action_features'
            timestep: [batch_size, num_frames] or [batch_size]
            kv_cache: KV cache for causal inference
            crossattn_cache: cross-attention cache
            current_start: current starting frame index
            classify_mode: whether in classification mode
            concat_time_embeddings: whether to concatenate time embeddings
            clean_x: clean latent for teacher forcing
            aug_t: augmented timestep
            cache_start: cache starting position
            action_features: [batch_size, action_dim] action features (overrides conditional_dict)
            
        Returns:
            Tuple of (logits, denoised_prediction) or just denoised_prediction
        """
        # Extract action features from conditional_dict if not provided directly
        if action_features is None and 'action_features' in conditional_dict:
            action_features = conditional_dict['action_features']
        
        # Store action features for model to access
        # We'll use a hook-based approach to inject during model forward
        if self.enable_action_conditioning and action_features is not None:
            num_frames = noisy_image_or_video.shape[1]
            action_mod = self._inject_action_into_model(action_features, num_frames)
            
            # Store in conditional_dict for model to access
            conditional_dict = conditional_dict.copy()
            conditional_dict['_action_modulation'] = action_mod
        
        # Call parent forward (which will call the model)
        # Note: The actual injection happens in the model's forward pass
        # We need to modify the model to check for '_action_modulation' in conditional_dict
        return super().forward(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start,
            classify_mode=classify_mode,
            concat_time_embeddings=concat_time_embeddings,
            clean_x=clean_x,
            aug_t=aug_t,
            cache_start=cache_start,
        )
    
    def get_action_projection_parameters(self):
        """Get parameters of the action projection module for optimization."""
        if self.action_projection is not None:
            return list(self.action_projection.parameters())
        return []

