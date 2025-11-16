from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Constants from the latent action training
WINDOW_SIZE = 8
ACTION_INDEX = 6  # Predict action taken at frame index 6 (0-based)


class Action3DCNN(nn.Module):
    """3D CNN model for predicting actions from sequences of latent frames."""
    
    def __init__(self, in_channels: int, action_dim: int, hidden_dims: tuple, dropout: float = 0.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
        )

        mlp_layers = []
        prev_dim = 256
        for hidden_dim in hidden_dims:
            mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            mlp_layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                mlp_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        mlp_layers.append(nn.Linear(prev_dim, action_dim))
        self.head = nn.Sequential(*mlp_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.head(features)


class LatentActionLoss:
    """
    Wrapper for computing action faithfulness loss using a pre-trained latent action model.
    
    The model takes 8 consecutive latent frames and predicts the action at frame 6.
    For a sequence of 21 frames, we slide the window across to get 14 action predictions,
    then compute MAE against the ground truth actions.
    """
    
    def __init__(self, device: torch.device, model_path: Optional[str] = None):
        self.device = device
        if not model_path:
            raise ValueError("Latent action model path is not set.")
        self.model = None
        self.model_path = str(Path(model_path).expanduser())
        self._load_model()
    
    def _load_model(self):
        """Load the pre-trained latent action model."""
        print(f"[ActionLoss] Loading latent action model from {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location="cpu")
        in_channels = checkpoint.get("input_channels", 16)
        action_dim = checkpoint.get("action_dim", 2)
        hidden_dims = tuple(checkpoint.get("hidden_dims", [256, 128]))
        self.model = Action3DCNN(
            in_channels=in_channels,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            dropout=0.0
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        print(f"[ActionLoss] Successfully loaded latent action model (in_channels={in_channels}, action_dim={action_dim})")
    
    def compute_loss(
        self,
        pred_latents: torch.Tensor,
        gt_actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute action faithfulness loss by applying latent action model with sliding window.
        
        Args:
            pred_latents: Generated latent frames [B, T, C, H, W]
            gt_actions: Ground truth actions [B, T, action_dim]
        
        Returns:
            mae_loss: Mean absolute error between predicted and ground truth actions
            log_dict: Dictionary with additional metrics for logging
        """
        if self.model is None or gt_actions is None:
            return torch.tensor(0.0, device=self.device), {}
        
        batch_size, num_frames, c, h, w = pred_latents.shape
        
        # Need at least 8 frames to make predictions
        if num_frames < WINDOW_SIZE:
            return torch.tensor(0.0, device=self.device), {}
        
        # Sliding window: for 21 frames, we get windows at positions 0-13 (14 windows total)
        num_windows = num_frames - WINDOW_SIZE + 1
        
        predicted_actions_list = []
        gt_actions_list = []
        
        with torch.no_grad():
            for start_idx in range(num_windows):
                # Extract window of 8 frames: [B, 8, C, H, W]
                window_frames = pred_latents[:, start_idx:start_idx + WINDOW_SIZE, :, :, :]
                
                # Reshape for Action3DCNN: expects [B, C, T, H, W]
                window_input = window_frames.permute(0, 2, 1, 3, 4).contiguous()
                
                # Predict action for frame at ACTION_INDEX
                action_pred = self.model(window_input)  # [B, action_dim]
                predicted_actions_list.append(action_pred)
                
                # Get corresponding ground truth action
                gt_action_idx = start_idx + ACTION_INDEX
                if gt_action_idx < gt_actions.shape[1]:
                    gt_actions_list.append(gt_actions[:, gt_action_idx, :])
        
        if not predicted_actions_list:
            return torch.tensor(0.0, device=self.device), {}
        
        predicted_actions = torch.stack(predicted_actions_list, dim=1)  # [B, num_windows, action_dim]
        gt_actions_stacked = torch.stack(gt_actions_list, dim=1)  # [B, num_windows, action_dim]
        
        # Compute MAE
        mae_loss = F.l1_loss(predicted_actions, gt_actions_stacked, reduction="mean")
        
        log_dict = {
            "action_faithfulness_mae": mae_loss.item(),
            "action_num_windows": num_windows,
        }
        
        return mae_loss, log_dict


# Global instance (lazy initialized)
_global_action_loss_instances: Dict[Tuple[str, str], LatentActionLoss] = {}


def get_action_loss_instance(device: torch.device, model_path: Optional[str] = None) -> LatentActionLoss:
    """Get or create the global action loss instance."""
    if not model_path:
        raise ValueError("Latent action model path is not set.")
    resolved_path = str(Path(model_path).expanduser())
    key = (str(device), resolved_path)
    instance = _global_action_loss_instances.get(key)
    if instance is None:
        instance = LatentActionLoss(device, model_path=resolved_path)
        _global_action_loss_instances[key] = instance
    return instance


def compute_action_faithfulness_loss(
    pred_latents: torch.Tensor,
    gt_actions: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    model_path: Optional[str] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Convenience function to compute action faithfulness loss.
    
    Args:
        pred_latents: Generated latent frames [B, T, C, H, W]
        gt_actions: Ground truth actions [B, T, action_dim]
        device: Device to use (inferred from pred_latents if not provided)
    
    Returns:
        mae_loss: Mean absolute error between predicted and ground truth actions
        log_dict: Dictionary with additional metrics for logging
    """
    if device is None:
        device = pred_latents.device
    
    action_loss_fn = get_action_loss_instance(device, model_path=model_path)
    return action_loss_fn.compute_loss(pred_latents, gt_actions)
