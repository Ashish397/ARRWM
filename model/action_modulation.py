# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA-4.0).
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

"""
Action Modulation Module for injecting action conditions via adaLN-Zero.

This module generates modulation parameters (scale, shift, gate) from action features
to condition the diffusion model's transformer blocks.
"""

import math
import torch
import torch.nn as nn


class ActionModulationProjection(nn.Module):
    """
    Projects action features to adaLN modulation parameters.
    
    Similar to time_projection in Wan model, but specifically for action conditioning.
    Outputs 6 modulation parameters per frame: (shift, scale, gate) × 2 (for self-attn and ffn)
    """
    
    def __init__(
        self,
        action_dim: int,
        activation: str,
        hidden_dim: int = 2048,
        mlp_dim: int = 512,
        num_frames: int = 1,
        zero_init: bool = True,
        use_fourier: bool = True,
        fourier_frequencies: int = 8,
        include_raw_actions: bool = True,
        zero_init_std: float = 1e-3,
    ):
        """
        Args:
            action_dim: dimension of input action features
            activation: activation function to use, one of "leakyrelu", "silu", or "tanh"
            hidden_dim: hidden dimension of the model (default 2048 for Wan-1.3B)
            num_frames: number of frames to generate modulation for
            zero_init: if True, initialize output layer to zero (adaLN-Zero)
        """
        super().__init__()
        
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.num_frames = num_frames
        self.use_fourier = use_fourier
        self.fourier_frequencies = fourier_frequencies
        self.include_raw_actions = include_raw_actions
        self.zero_init_std = zero_init_std
        self.activation = activation

        self.shift_scale = 1.0 #0.5
        self.gate_scale = 0.2 #0.1

        # Get activation function based on string
        activation_fn = self._get_activation(activation)

        # Fourier features (like time embeddings, but for actions in [-1, 1])
        if self.use_fourier:
            freq = (2.0 ** torch.arange(self.fourier_frequencies, dtype=torch.float32)) * math.pi
            self.register_buffer("fourier_freq_bands", freq, persistent=False)
            fourier_dim = action_dim * self.fourier_frequencies * 2  # sin+cos
            in_dim = fourier_dim + (action_dim if self.include_raw_actions else 0)
        else:
            in_dim = action_dim
        
        # Shallow, sign-symmetric-ish MLP with activation + LayerNorm
        self.action_embedding = nn.Sequential(
            nn.Linear(in_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            activation_fn(),
            nn.Linear(mlp_dim, mlp_dim),
            activation_fn(),
        )

        # Projection to modulation parameters (6 params per frame -> dim)
        self.action_projection = nn.Linear(mlp_dim, hidden_dim * 6)
        
        # Initialize with zeros for adaLN-Zero
        if zero_init:
            # "Near-zero" init so gradients flow through the whole branch immediately
            nn.init.normal_(self.action_projection.weight, mean=0.0, std=self.zero_init_std)
            nn.init.zeros_(self.action_projection.bias)
    
    def _get_activation(self, activation_str: str):
        """Get the appropriate activation class based on string."""
        activation_str = activation_str.lower()
        if activation_str == "leakyrelu":
            return lambda: nn.LeakyReLU(negative_slope=0.2)
        elif activation_str == "silu":
            return nn.SiLU
        elif activation_str == "tanh":
            return nn.Tanh
        else:
            raise ValueError(f"Unknown activation function: {activation_str}. Must be one of 'leakyrelu', 'silu', or 'tanh'")
    
    def forward(self, action_features: torch.Tensor, num_frames: int | None = None) -> torch.Tensor:
        """
        Generate modulation parameters from action features.
        
        Args:
            action_features: [batch_size, action_dim] or [batch_size, num_frames, action_dim]
            num_frames: number of frames (if action_features is 2D)
            
        Returns:
            modulation_params: [batch_size, num_frames, 6, hidden_dim]
        """
        # [B, F, action_dim]
        assert action_features.dim() == 3
        num_frames = action_features.shape[1]
        batch_size = action_features.shape[0]

        x = action_features
        # Fourier features
        if self.use_fourier:
            # x: [B, F, A] -> [B, F, A, K]
            xb = x[..., None] * self.fourier_freq_bands[None, None, None, :]
            ff = torch.cat([torch.sin(xb), torch.cos(xb)], dim=-1)  # [B,F,A,2K]
            ff = ff.reshape(batch_size, num_frames, -1)            # [B,F,2AK]
            if self.include_raw_actions:
                x = torch.cat([x, ff], dim=-1)                     # [B,F,A+2AK]
            else:
                x = ff                                             # [B,F,2AK]
        
        # Flatten for processing
        action_flat = x.flatten(0, 1)  # [B*F, in_dim]
        ref_weight = next(self.action_embedding.parameters(), None)
        if ref_weight is None:
            raise RuntimeError("action_embedding must have parameters to determine device/dtype.")
        action_flat = action_flat.to(device=ref_weight.device, dtype=ref_weight.dtype)

        # Embed action
        action_emb = self.action_embedding(action_flat)  # [B*F, hidden_dim]

        # Project to modulation parameters
        modulation = self.action_projection(action_emb)  # [B*F, hidden_dim * 6]
        # Reshape to [B, F, 6, hidden_dim]
        modulation = modulation.view(batch_size, num_frames, 6, self.hidden_dim)
        
        scale = modulation.new_tensor([
            self.shift_scale, self.shift_scale, self.gate_scale,
            self.shift_scale, self.shift_scale, self.gate_scale
        ]).view(1, 1, 6, 1)

        modulation = modulation * scale
        return modulation


class ActionConditionedWrapper:
    """
    Wrapper to inject action modulation into existing Wan model forward pass.
    
    This modifies the time embedding to include action modulation without
    changing the underlying model architecture.
    """
    
    def __init__(
        self,
        base_model,
        action_projection: ActionModulationProjection,
        device='cuda'
    ):
        """
        Args:
            base_model: the Wan model (CausalWanModel or WanModel)
            action_projection: ActionModulationProjection module
            device: device to place modules on
        """
        self.base_model = base_model
        self.action_projection = action_projection.to(device)
        self.device = device
        
        # Store original forward methods
        self._original_forward_inference = base_model._forward_inference if hasattr(base_model, '_forward_inference') else None
        self._original_forward = base_model._forward if hasattr(base_model, '_forward') else None
        
    def inject_action_modulation(self, time_modulation: torch.Tensor, action_features: torch.Tensor | None) -> torch.Tensor:
        """
        Add action modulation to time modulation.
        
        Args:
            time_modulation: [B, F, 6, dim] time modulation parameters
            action_features: [B, action_dim] or [B, F, action_dim] action features
            
        Returns:
            combined_modulation: [B, F, 6, dim] combined modulation
        """
        if action_features is None:
            return time_modulation
        
        # Generate action modulation
        num_frames = time_modulation.shape[1]
        action_mod = self.action_projection(action_features, num_frames=num_frames)
        
        # Combine: time modulation + action modulation
        # This is where adaLN-Zero helps: action starts at zero, gradually learns
        combined = time_modulation + action_mod
        
        return combined
    
    def get_modulation_injector(self):
        """
        Returns a function that can be called to inject action during forward pass.
        Use this as a hook.
        """
        def injector(time_emb, action_features):
            return self.inject_action_modulation(time_emb, action_features)
        return injector


def create_action_modulation_module(
    action_dim: int,
    activation: str,
    model_hidden_dim: int = 2048,
    num_frames: int = 1,
    zero_init: bool = True,
    device: str = 'cuda'
) -> ActionModulationProjection:
    """
    Factory function to create action modulation module.
    
    Args:
        action_dim: dimension of action features
        activation: activation function to use, one of "leakyrelu", "silu", or "tanh"
        model_hidden_dim: hidden dimension of the diffusion model
        num_frames: number of frames
        zero_init: whether to use zero initialization (adaLN-Zero)
        device: device to place the module on
        
    Returns:
        ActionModulationProjection module
    """
    module = ActionModulationProjection(
        action_dim=action_dim,
        activation=activation,
        hidden_dim=model_hidden_dim,
        num_frames=num_frames,
        zero_init=zero_init,
    )
    return module.to(device)
