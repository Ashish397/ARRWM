"""Learned action critic for guiding the causal diffusion model.

Chunkwise 3D-convolutional critic that consumes 3-frame latent cubes
from ``pred_x0`` and predicts per-chunk action latents and reward:
  - z2/z7: 2-dim action latent per 3-frame chunk  (z-head)
  - reward: scalar per 3-frame chunk               (reward-head)

Teacher targets are produced by a frozen evaluator pipeline:
  pred_x0 -> noise(t=25) -> VAE decode -> CoTracker -> ss_vae -> z2/z7, reward
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcite3d(nn.Module):
    """Channel attention via squeeze-and-excitation."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape[:2]
        w = self.pool(x).view(B, C)
        w = self.fc(w).view(B, C, 1, 1, 1)
        return x * w


class ResBlock3d(nn.Module):
    """Residual block with two 3D convolutions, GroupNorm, and optional SE."""

    def __init__(self, channels: int, groups: int = 8, use_se: bool = True):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.act = nn.SiLU(inplace=True)
        self.se = SqueezeExcite3d(channels) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.se(out)
        return self.act(out + residual)


class AttentionPool3d(nn.Module):
    """Learned attention pooling over spatial and temporal dims."""

    def __init__(self, channels: int):
        super().__init__()
        self.query = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.SiLU(inplace=True),
            nn.Linear(channels // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # [B, T*H*W, C]
        w = self.query(x_flat)  # [B, T*H*W, 1]
        w = F.softmax(w, dim=1)
        return (x_flat * w).sum(dim=1)  # [B, C]


class ActionCritic(nn.Module):
    """Chunkwise 3D-convolutional critic over latent cubes.

    Consumes ``pred_x0`` with shape ``[B, F, C, H, W]`` where
    ``C=16, H=60, W=104`` for the Wan-1.3B latent space. Internally
    reshapes frames into contiguous 3-frame chunks and processes each
    chunk through a 3D CNN with squeeze-excitation and learned attention
    pooling, producing one action prediction and one reward prediction
    per chunk.

    Args:
        latent_channels: Number of latent channels (default 16).
        z_dim: Dimension of action latent output (default 2 for z2/z7).
        base_channels: Channel width of first conv stage (default 64).
        num_res_blocks: Number of residual blocks in the trunk (default 3).
        chunk_frames: Frames per chunk (default 3, matching num_frame_per_block).
        use_se: Enable squeeze-and-excitation in residual blocks.
        use_attn_pool: Use learned attention pooling instead of average pooling.
    """

    def __init__(
        self,
        latent_channels: int = 16,
        z_dim: int = 2,
        base_channels: int = 64,
        num_res_blocks: int = 3,
        chunk_frames: int = 3,
        use_se: bool = True,
        use_attn_pool: bool = True,
        # Legacy kwargs accepted for backward-compat checkpoint loading
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.z_dim = z_dim
        self.chunk_frames = chunk_frames
        self._use_attn_pool = use_attn_pool

        bc = base_channels

        self.stem = nn.Sequential(
            nn.Conv3d(latent_channels, bc, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, bc),
            nn.SiLU(inplace=True),
        )

        self.down1 = nn.Sequential(
            nn.Conv3d(bc, bc * 2, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.GroupNorm(8, bc * 2),
            nn.SiLU(inplace=True),
        )

        self.down2 = nn.Sequential(
            nn.Conv3d(bc * 2, bc * 4, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.GroupNorm(8, bc * 4),
            nn.SiLU(inplace=True),
        )

        trunk_layers = []
        for _ in range(num_res_blocks):
            trunk_layers.append(ResBlock3d(bc * 4, groups=8, use_se=use_se))
        self.trunk = nn.Sequential(*trunk_layers)

        if use_attn_pool:
            self.pool = AttentionPool3d(bc * 4)
        else:
            self.pool = nn.AdaptiveAvgPool3d(1)

        head_in = bc * 4
        self.z_head = nn.Sequential(
            nn.Linear(head_in, head_in),
            nn.SiLU(inplace=True),
            nn.Linear(head_in, head_in // 2),
            nn.SiLU(inplace=True),
            nn.Linear(head_in // 2, z_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(head_in, head_in),
            nn.SiLU(inplace=True),
            nn.Linear(head_in, head_in // 2),
            nn.SiLU(inplace=True),
            nn.Linear(head_in // 2, 1),
        )

        self._init_heads()

    def _init_heads(self):
        for head in (self.z_head, self.reward_head):
            last = head[-1]
            nn.init.normal_(last.weight, std=1e-3)
            nn.init.zeros_(last.bias)

    def forward(self, pred_x0: torch.Tensor):
        """
        Args:
            pred_x0: ``[B, F, C, H, W]`` predicted clean latents.
                     F must be divisible by ``chunk_frames``.

        Returns:
            pred_z: ``[B, n_chunks, z_dim]``
            pred_reward: ``[B, n_chunks, 1]``
        """
        B, F, C, H, W = pred_x0.shape
        T = self.chunk_frames
        assert F % T == 0, f"F={F} not divisible by chunk_frames={T}"
        n_chunks = F // T

        # [B, n_chunks, T, C, H, W] -> [B*n_chunks, C, T, H, W]
        x = pred_x0.reshape(B, n_chunks, T, C, H, W)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * n_chunks, C, T, H, W)

        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.trunk(x)

        if self._use_attn_pool:
            x = self.pool(x)  # [B*n_chunks, head_in]
        else:
            x = self.pool(x).flatten(1)  # [B*n_chunks, head_in]

        pred_z = self.z_head(x).reshape(B, n_chunks, self.z_dim)
        pred_reward = self.reward_head(x).reshape(B, n_chunks, 1)
        return pred_z, pred_reward
