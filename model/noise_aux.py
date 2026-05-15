"""Lite AR-noise generator for the aux-teacher pipeline.

Predicts the AR-noise direction from a clean latent. Used as a
reference option in ``_compute_aux_teacher_loss_streaming`` to
replace the (chunk - pred_real) pairwise residual, which suffered
from time-warp blur because chunk and pred_real share a temporal
shift relative to GT.

Training contract:
    Input  : pred_image          [B, F, C, H, W]   (student rollout)
    Target : 2 * pred_image - pred_real             (= pred_image + AR_noise)
    Loss   : MSE(noise_aux(pred_image), target)

At inference (aux teacher training time):
    noise_aux(GT) ≈ GT + AR_noise_estimate_for_GT
    ar_residual   = noise_aux(GT) - GT
The model is trained on the forward noising direction (clean → noised)
which is well-defined per sample, avoiding the time-warp problem of
the inverse direction.

Architecture: per-frame residual ConvNet. Treats each latent frame
independently (no temporal mixing) for simplicity and memory. Small
enough (~8M params @ hidden=192, num_blocks=4) to add to the existing
training graph without OOM. Upgradeable to a temporal variant later
if per-frame noise-modelling is insufficient.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResBlock2D(nn.Module):
    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class NoiseAuxLite(nn.Module):
    """Per-frame residual ConvNet for AR-noise prediction.

    Maps a latent ``x [B, F, C, H, W]`` to ``x + delta(x)`` where
    ``delta`` is learned to approximate the AR-noise direction.
    The ``out_proj`` is zero-initialised so the model is the identity
    at init: ``noise_aux(x) = x``. The delta grows as training
    progresses.
    """

    def __init__(
        self,
        latent_channels: int = 16,
        hidden_channels: int = 192,
        num_blocks: int = 4,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.hidden_channels = hidden_channels
        self.num_blocks = num_blocks

        self.in_proj = nn.Conv2d(latent_channels, hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [_ResBlock2D(hidden_channels) for _ in range(num_blocks)]
        )
        self.out_proj = nn.Conv2d(hidden_channels, latent_channels, kernel_size=1)
        # Identity-at-init: delta = 0 initially.
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, F, C, H, W] → x + delta(x) of same shape."""
        if x.dim() != 5:
            raise ValueError(
                f"NoiseAuxLite expects 5D input [B, F, C, H, W], got {tuple(x.shape)}"
            )
        B, Fd, C, H, W = x.shape
        if C != self.latent_channels:
            raise ValueError(
                f"NoiseAuxLite latent_channels={self.latent_channels} but input has C={C}"
            )
        x_flat = x.reshape(B * Fd, C, H, W)
        h = self.in_proj(x_flat)
        for blk in self.blocks:
            h = blk(h)
        delta_flat = self.out_proj(h)
        delta = delta_flat.reshape(B, Fd, C, H, W)
        return x + delta

    def predict_delta(self, x: torch.Tensor) -> torch.Tensor:
        """Returns just the delta (= predicted AR-noise increment)."""
        return self.forward(x) - x

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
