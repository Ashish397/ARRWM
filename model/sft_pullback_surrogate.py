"""Multiscale SFT surrogate for decoder pullback experiments.

The pixel cotangent remains spatially resolved at four reverse-decoder grids:
12x192x256, 12x96x128, 6x48x64 and 3x24x32.  A separate condition pyramid
produces scale/shift maps at every grid.  ``global`` mode averages those exact
maps before modulation and is therefore a parameter-matched FiLM control;
``spatial`` mode applies them without spatial or temporal pooling.

Optional detached WAN decoder summaries are adapted at their native grids and
added to the corresponding condition feature before SFT.  No decoder graph is
accepted by this benchmark-only module.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SpatialFeatureTransform3D", "SFTPullbackSurrogate"]


class SpatialFeatureTransform3D(nn.Module):
    """Identity-initialised spatial SFT or parameter-identical global FiLM."""

    def __init__(self, feature_width: int, condition_width: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(condition_width, condition_width, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(condition_width, 2 * feature_width, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(
        self, features: torch.Tensor, condition: torch.Tensor, *, global_map: bool,
    ) -> torch.Tensor:
        if features.shape[0] != condition.shape[0] or features.shape[-3:] != condition.shape[-3:]:
            raise ValueError(
                "SFT feature/condition geometry mismatch: "
                f"{tuple(features.shape)}/{tuple(condition.shape)}"
            )
        gamma_raw, beta = self.body(condition).chunk(2, dim=1)
        if global_map:
            gamma_raw = gamma_raw.mean(dim=(-3, -2, -1), keepdim=True)
            beta = beta.mean(dim=(-3, -2, -1), keepdim=True)
        gamma = 1.0 + torch.tanh(gamma_raw)
        return gamma * features + beta


class ResidualFeatureBlock3D(nn.Module):
    def __init__(self, width: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(
            width, width, 3, padding=dilation, dilation=dilation,
        )
        self.conv2 = nn.Conv3d(
            width, width, 3, padding=dilation, dilation=dilation,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.conv2(F.silu(self.conv1(F.silu(value))))


class SFTPullbackSurrogate(nn.Module):
    """Predict ``J_decode(z)^T v`` with multiscale pixel-gradient injection.

    Public tensors use repository layout ``[B,F,C,H,W]``.  Decoder states, if
    supplied, are ordered high-to-low and must match the four cotangent grids.
    """

    def __init__(
        self,
        *,
        latent_channels: int = 16,
        latent_frames: int = 3,
        latent_height: int = 24,
        latent_width: int = 32,
        pixel_frames: int = 12,
        pixel_height: int = 192,
        pixel_width: int = 256,
        feature_widths: Sequence[int] = (24, 48, 96, 96),
        condition_widths: Sequence[int] = (16, 24, 32, 48),
        latent_blocks: int = 4,
        conditioning_mode: str = "spatial",
        decoder_state_channels: int = 0,
    ) -> None:
        super().__init__()
        if conditioning_mode not in ("spatial", "global"):
            raise ValueError("conditioning_mode must be 'spatial' or 'global'")
        if len(feature_widths) != 4 or len(condition_widths) != 4:
            raise ValueError("feature_widths and condition_widths must have four levels")
        self.conditioning_mode = conditioning_mode
        self.decoder_state_channels = int(decoder_state_channels)
        self.latent_shape = (
            int(latent_frames), int(latent_channels), int(latent_height), int(latent_width),
        )
        self.pixel_shape = (
            int(pixel_frames), 3, int(pixel_height), int(pixel_width),
        )
        if (
            self.pixel_shape[0] != 4 * self.latent_shape[0]
            or self.pixel_shape[2] != 8 * self.latent_shape[2]
            or self.pixel_shape[3] != 8 * self.latent_shape[3]
        ):
            raise ValueError("pixel geometry must be 4x temporal and 8x spatial latent geometry")
        self.level_grids: Tuple[Tuple[int, int, int], ...] = (
            (self.pixel_shape[0], self.pixel_shape[2], self.pixel_shape[3]),
            (self.pixel_shape[0], self.pixel_shape[2] // 2, self.pixel_shape[3] // 2),
            (self.pixel_shape[0] // 2, self.pixel_shape[2] // 4, self.pixel_shape[3] // 4),
            (self.latent_shape[0], self.latent_shape[2], self.latent_shape[3]),
        )
        fw = tuple(int(item) for item in feature_widths)
        cw = tuple(int(item) for item in condition_widths)
        if any(item <= 0 for item in (*fw, *cw)) or int(latent_blocks) <= 0:
            raise ValueError("all widths and latent_blocks must be positive")

        self.feature_in = nn.Conv3d(3, fw[0], 3, padding=1)
        self.condition_in = nn.Conv3d(3, cw[0], 3, padding=1)
        # Exact reverse of this repository's configured WAN decoder hierarchy:
        # spatial-only at the pixel end, then two temporal+spatial reductions.
        strides = ((1, 2, 2), (2, 2, 2), (2, 2, 2))
        self.feature_down = nn.ModuleList([
            nn.Conv3d(fw[index], fw[index + 1], 3, stride=strides[index], padding=1)
            for index in range(3)
        ])
        self.condition_down = nn.ModuleList([
            nn.Conv3d(cw[index], cw[index + 1], 3, stride=strides[index], padding=1)
            for index in range(3)
        ])
        self.modulations = nn.ModuleList([
            SpatialFeatureTransform3D(fw[index], cw[index]) for index in range(4)
        ])
        if self.decoder_state_channels > 0:
            self.state_adapters: Optional[nn.ModuleList] = nn.ModuleList([
                nn.Conv3d(self.decoder_state_channels, cw[index], 1)
                for index in range(4)
            ])
        else:
            self.state_adapters = None
        self.z_tower = nn.Sequential(
            nn.Conv3d(int(latent_channels), fw[-1], 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(fw[-1], fw[-1], 3, padding=1),
            nn.SiLU(),
        )
        self.fuse = nn.Conv3d(2 * fw[-1], fw[-1], 1)
        dilations = (1, 2, 4, 2, 1)
        self.blocks = nn.ModuleList([
            ResidualFeatureBlock3D(fw[-1], dilations[index % len(dilations)])
            for index in range(int(latent_blocks))
        ])
        self.output = nn.Conv3d(fw[-1], int(latent_channels), 3, padding=1)

    @property
    def num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _validate_state(
        self, decoder_states: Optional[Sequence[torch.Tensor]], batch: int,
    ) -> Optional[Tuple[torch.Tensor, ...]]:
        if self.state_adapters is None:
            if decoder_states is not None:
                raise ValueError("decoder states supplied to a state-free SFT model")
            return None
        if decoder_states is None or len(decoder_states) != 4:
            raise ValueError("state-conditioned SFT requires four decoder state tensors")
        parameter = next(self.parameters())
        result = []
        for index, (state, grid) in enumerate(zip(decoder_states, self.level_grids)):
            if state.requires_grad:
                raise ValueError("WAN decoder states must be detached")
            expected = (batch, grid[0], self.decoder_state_channels, grid[1], grid[2])
            if tuple(state.shape) != expected:
                raise ValueError(
                    f"decoder state {index} must have shape {expected}, got {tuple(state.shape)}"
                )
            result.append(
                state.detach().to(device=parameter.device, dtype=parameter.dtype)
                .permute(0, 2, 1, 3, 4).contiguous()
            )
        return tuple(result)

    def forward(
        self,
        latent: torch.Tensor,
        pixel_cotangent: torch.Tensor,
        decoder_states: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if latent.dim() != 5 or tuple(latent.shape[1:]) != self.latent_shape:
            raise ValueError(
                f"latent must be [B,{','.join(map(str, self.latent_shape))}], "
                f"got {tuple(latent.shape)}"
            )
        if pixel_cotangent.dim() != 5 or tuple(pixel_cotangent.shape[1:]) != self.pixel_shape:
            raise ValueError(
                f"pixel cotangent must be [B,{','.join(map(str, self.pixel_shape))}], "
                f"got {tuple(pixel_cotangent.shape)}"
            )
        if pixel_cotangent.requires_grad:
            raise ValueError("pixel cotangent must be detached")
        batch = int(latent.shape[0])
        if int(pixel_cotangent.shape[0]) != batch:
            raise ValueError("latent/pixel-cotangent batch mismatch")
        states = self._validate_state(decoder_states, batch)
        parameter = next(self.parameters())
        v = pixel_cotangent.detach().to(device=parameter.device, dtype=parameter.dtype)
        v = v.permute(0, 2, 1, 3, 4).contiguous()
        h = F.silu(self.feature_in(v))
        condition = F.silu(self.condition_in(v))
        global_map = self.conditioning_mode == "global"
        for level in range(4):
            conditioned = condition
            if states is not None:
                conditioned = conditioned + self.state_adapters[level](states[level])
            h = self.modulations[level](h, conditioned, global_map=global_map)
            if level < 3:
                h = F.silu(self.feature_down[level](h))
                condition = F.silu(self.condition_down[level](condition))
        z = latent.to(device=parameter.device, dtype=parameter.dtype)
        z = z.permute(0, 2, 1, 3, 4).contiguous()
        h = self.fuse(torch.cat([h, self.z_tower(z)], dim=1))
        for block in self.blocks:
            h = block(h)
        output = self.output(F.silu(h))
        return output.permute(0, 2, 1, 3, 4).contiguous().float()
