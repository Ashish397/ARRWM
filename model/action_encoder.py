# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

"""
Action Encoder: Encode raw action values to action features

This module converts raw action values (e.g., [velocity, steering])
into high-dimensional features suitable for action conditioning.

Similar to how timestep is processed:
  timestep → sinusoidal_embedding → time_embedding → features
  
For actions:
  action_values → action_encoder → action_features
"""

import torch
import torch.nn as nn
import math


class ActionEncoder(nn.Module):
    """
    Encode raw action values to feature vectors.
    
    Similar to timestep encoding, but for continuous action values.
    """
    
    def __init__(
        self,
        action_dim: int = 2,           # Number of action dimensions (e.g., 2 for [velocity, steering])
        feature_dim: int = 512,        # Output feature dimension
        hidden_dim: int = 256,         # Hidden dimension for MLP
        use_sinusoidal: bool = True,   # Whether to use sinusoidal encoding (like timestep)
        freq_dim: int = 64,            # Frequency dimension for sinusoidal encoding
    ):
        """
        Args:
            action_dim: Number of action values (e.g., 2 for [forward, turn])
            feature_dim: Output feature dimension (should match action_dim in ActionModulationProjection)
            hidden_dim: Hidden dimension for MLP
            use_sinusoidal: Whether to use sinusoidal encoding (recommended for continuous actions)
            freq_dim: Frequency dimension for sinusoidal encoding per action dimension
        """
        super().__init__()
        
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.use_sinusoidal = use_sinusoidal
        self.freq_dim = freq_dim
        
        if use_sinusoidal:
            # Sinusoidal encoding (similar to timestep encoding)
            # Input: [B, action_dim] -> Output: [B, action_dim * freq_dim]
            self.sinusoidal_input_dim = action_dim * freq_dim
            
            # MLP to process sinusoidal features
            self.encoder = nn.Sequential(
                nn.Linear(self.sinusoidal_input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, feature_dim),
            )
        else:
            # Direct MLP encoding (simpler, but less effective for continuous values)
            self.encoder = nn.Sequential(
                nn.Linear(action_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, feature_dim),
            )
        
        print(f"[ActionEncoder] Created: action_dim={action_dim}, feature_dim={feature_dim}, "
              f"use_sinusoidal={use_sinusoidal}")
    
    def sinusoidal_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply sinusoidal encoding to action values (similar to positional encoding).
        
        Args:
            x: [batch_size, action_dim] raw action values
            
        Returns:
            encoded: [batch_size, action_dim * freq_dim] sinusoidal features
        """
        batch_size, action_dim = x.shape
        device = x.device
        
        # Create frequency bands (similar to timestep encoding)
        # freq = 1.0 / (10000^(2i/freq_dim))
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, self.freq_dim, 2, device=device) / self.freq_dim
        )
        
        # Apply to each action dimension
        encoded_list = []
        for i in range(action_dim):
            # x[:, i:i+1] -> [B, 1]
            # freqs -> [freq_dim/2]
            # result -> [B, freq_dim/2]
            angles = x[:, i:i+1] * freqs.unsqueeze(0)
            
            # Compute sin and cos
            sin_enc = torch.sin(angles)  # [B, freq_dim/2]
            cos_enc = torch.cos(angles)  # [B, freq_dim/2]
            
            # Interleave sin and cos
            encoded = torch.stack([sin_enc, cos_enc], dim=2).flatten(1)  # [B, freq_dim]
            encoded_list.append(encoded)
        
        # Concatenate all action dimensions
        encoded = torch.cat(encoded_list, dim=1)  # [B, action_dim * freq_dim]
        return encoded
    
    def forward(self, action_values: torch.Tensor) -> torch.Tensor:
        """
        Encode action values to feature vectors.
        
        Args:
            action_values: [batch_size, action_dim] raw action values
                          e.g., [[0.5, 0.3], [0.8, -0.2], ...]
            
        Returns:
            action_features: [batch_size, feature_dim] encoded features
        """
        assert action_values.dim() == 2, f"Expected 2D tensor, got {action_values.dim()}D"
        assert action_values.shape[1] == self.action_dim, \
            f"Expected action_dim={self.action_dim}, got {action_values.shape[1]}"
        
        if self.use_sinusoidal:
            # Apply sinusoidal encoding first
            encoded = self.sinusoidal_encoding(action_values)  # [B, action_dim * freq_dim]
        else:
            encoded = action_values
        
        # Pass through MLP
        features = self.encoder(encoded)  # [B, feature_dim]
        
        return features


class DiscreteActionEncoder(nn.Module):
    """
    Encode discrete action indices to feature vectors.
    
    Use this if your actions are discrete (e.g., [0, 1, 2] for left/forward/right).
    """
    
    def __init__(
        self,
        num_actions: int,              # Number of discrete actions
        feature_dim: int = 512,        # Output feature dimension
        embedding_dim: int = 128,      # Embedding dimension for each action
    ):
        """
        Args:
            num_actions: Number of discrete action values
            feature_dim: Output feature dimension
            embedding_dim: Intermediate embedding dimension
        """
        super().__init__()
        
        self.num_actions = num_actions
        self.feature_dim = feature_dim
        
        # Learnable embedding for each action
        self.action_embedding = nn.Embedding(num_actions, embedding_dim)
        
        # MLP to project to feature_dim
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        
        print(f"[DiscreteActionEncoder] Created: num_actions={num_actions}, feature_dim={feature_dim}")
    
    def forward(self, action_indices: torch.Tensor) -> torch.Tensor:
        """
        Encode discrete action indices to feature vectors.
        
        Args:
            action_indices: [batch_size] action indices (0 to num_actions-1)
            
        Returns:
            action_features: [batch_size, feature_dim] encoded features
        """
        assert action_indices.dim() == 1, f"Expected 1D tensor, got {action_indices.dim()}D"
        
        # Embed actions
        embedded = self.action_embedding(action_indices)  # [B, embedding_dim]
        
        # Project to feature_dim
        features = self.encoder(embedded)  # [B, feature_dim]
        
        return features


# Factory function for easy creation
def create_action_encoder(
    action_type: str = 'continuous',
    action_dim: int = 2,
    feature_dim: int = 512,
    **kwargs
) -> nn.Module:
    """
    Factory function to create appropriate action encoder.
    
    Args:
        action_type: 'continuous' or 'discrete'
        action_dim: Number of action dimensions (for continuous) or number of actions (for discrete)
        feature_dim: Output feature dimension
        **kwargs: Additional arguments passed to encoder
        
    Returns:
        action_encoder: ActionEncoder or DiscreteActionEncoder
    """
    if action_type == 'continuous':
        return ActionEncoder(
            action_dim=action_dim,
            feature_dim=feature_dim,
            **kwargs
        )
    elif action_type == 'discrete':
        return DiscreteActionEncoder(
            num_actions=action_dim,
            feature_dim=feature_dim,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown action_type: {action_type}")

