"""Benchmark-only surrogate for the WAN decoder pullback ``J(z)^T v``.

This module is deliberately disconnected from the trainer and production
configuration.  It tests a narrower hypothesis than the current latent field
predictor: can a student learn the decoder's vector-Jacobian product when it
is given both the latent state ``z`` and the signed pixel cotangent ``v``?

The important structural contract is exact linearity in ``v`` for fixed
``z``.  The cotangent path contains only bias-free convolutions, additions,
and multiplication by gates that depend *only* on ``z``.  Consequently, up to
floating-point roundoff,

``S(z, 0)=0``, ``S(z, -v)=-S(z, v)``, and
``S(z, a*v1+b*v2)=a*S(z,v1)+b*S(z,v2)``.

The default geometry is the full independently seeded WAN crop decode:
``[B,12,3,192,256]`` pixels for ``[B,3,16,24,32]`` latents.  A production
border-trimmed cotangent must be zero-padded back to this full support before
calling the model, exactly as it must be before an exact decoder VJP.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "SignedCotangentPyramid",
    "ZConditionedLinearPullbackBlock",
    "DecoderPullbackSurrogate",
]


class SignedCotangentPyramid(nn.Module):
    """Bias-free signed 3-D analysis pyramid at exact WAN scale factors.

    Input is ``[B,F_pixel,3,H_pixel,W_pixel]``.  Three strided convolutions
    reduce by ``(temporal, spatial) = (2,2), (2,2), (1,2)`` and therefore map
    the default ``12x192x256`` support exactly onto ``3x24x32``.  There is no
    max pooling, activation, bias, or normalisation: dark/bright signs,
    cancellation, and cotangent amplitude remain representable.  The public
    output layout is ``[B,F_latent,C_pyramid,H_latent,W_latent]``; Conv3d's
    channel-first-temporal layout remains an internal implementation detail.
    """

    def __init__(
        self,
        *,
        channels: Sequence[int] = (24, 48, 96),
        pixel_frames: int = 12,
        pixel_height: int = 192,
        pixel_width: int = 256,
        latent_frames: int = 3,
        latent_height: int = 24,
        latent_width: int = 32,
    ) -> None:
        super().__init__()
        if len(channels) != 3 or any(int(item) <= 0 for item in channels):
            raise ValueError("channels must contain three positive widths")
        self.channels = tuple(int(item) for item in channels)
        self.pixel_shape = (
            int(pixel_frames), 3, int(pixel_height), int(pixel_width),
        )
        self.latent_grid = (
            int(latent_frames), int(latent_height), int(latent_width),
        )
        if (
            self.pixel_shape[0] != 4 * self.latent_grid[0]
            or self.pixel_shape[2] != 8 * self.latent_grid[1]
            or self.pixel_shape[3] != 8 * self.latent_grid[2]
        ):
            raise ValueError(
                "pixel geometry must be exactly 4x temporal and 8x spatial "
                "relative to the latent grid; got "
                f"pixel={self.pixel_shape}, latent={self.latent_grid}"
            )

        strides: Tuple[Tuple[int, int, int], ...] = (
            (2, 2, 2), (2, 2, 2), (1, 2, 2),
        )
        self.stage_grids: Tuple[Tuple[int, int, int], ...] = (
            (
                self.pixel_shape[0] // 2,
                self.pixel_shape[2] // 2,
                self.pixel_shape[3] // 2,
            ),
            (
                self.pixel_shape[0] // 4,
                self.pixel_shape[2] // 4,
                self.pixel_shape[3] // 4,
            ),
            self.latent_grid,
        )
        widths = (3, *self.channels)
        self.stages = nn.ModuleList([
            nn.Conv3d(
                widths[index], widths[index + 1],
                kernel_size=(3, 5, 5), stride=strides[index],
                padding=(1, 2, 2), bias=False,
            )
            for index in range(3)
        ])

    @property
    def out_channels(self) -> int:
        return self.channels[-1]

    def forward(
        self,
        pixel_cotangent: torch.Tensor,
        stage_gates: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if pixel_cotangent.dim() != 5:
            raise ValueError(
                "pixel cotangent must be [B,F,3,H,W], got "
                f"{tuple(pixel_cotangent.shape)}"
            )
        if tuple(pixel_cotangent.shape[1:]) != self.pixel_shape:
            raise ValueError(
                f"pixel cotangent must have trailing shape {self.pixel_shape}, "
                f"got {tuple(pixel_cotangent.shape[1:])}"
            )
        if pixel_cotangent.requires_grad:
            raise ValueError(
                "pixel cotangent must be detached; its source graph is not "
                "part of the pullback surrogate"
            )
        parameter = next(self.parameters())
        # [B,F,3,H,W] -> Conv3d's [B,3,F,H,W].
        h = pixel_cotangent.detach().to(
            device=parameter.device, dtype=parameter.dtype,
        ).permute(0, 2, 1, 3, 4).contiguous()
        if stage_gates is not None and len(stage_gates) != len(self.stages):
            raise ValueError(
                f"expected {len(self.stages)} stage gates, got "
                f"{len(stage_gates)}"
            )
        for index, stage in enumerate(self.stages):
            h = stage(h)
            if stage_gates is not None:
                gate = stage_gates[index]
                if tuple(gate.shape) != tuple(h.shape):
                    raise ValueError(
                        f"stage {index} gate must have shape {tuple(h.shape)}, "
                        f"got {tuple(gate.shape)}"
                    )
                # gate depends only on z, so multiplication preserves exact
                # linearity of the entire branch in the pixel cotangent.
                h = h * gate
        if tuple(h.shape[-3:]) != self.latent_grid:
            raise RuntimeError(
                "signed cotangent pyramid produced the wrong latent grid: "
                f"expected {self.latent_grid}, got {tuple(h.shape[-3:])}"
            )
        # Conv3d [B,C,F,H,W] -> repository-standard [B,F,C,H,W].
        return h.permute(0, 2, 1, 3, 4).contiguous()


class ZConditionedLinearPullbackBlock(nn.Module):
    """Residual linear map in ``v`` with a nonlinear multiplicative z gate."""

    def __init__(self, width: int, z_width: int, dilation: int = 1) -> None:
        super().__init__()
        width, z_width = int(width), int(z_width)
        self.in_map = nn.Conv3d(
            width, width, 3, padding=int(dilation), dilation=int(dilation),
            bias=False,
        )
        self.out_map = nn.Conv3d(
            width, width, 3, padding=int(dilation), dilation=int(dilation),
            bias=False,
        )
        # A z-only bias is mathematically safe: it can alter the multiplier,
        # but can never create an output when the cotangent stream is zero.
        self.gate = nn.Conv3d(z_width, width, 1, bias=True)

    def forward(self, h: torch.Tensor, z_features: torch.Tensor) -> torch.Tensor:
        if h.shape[0] != z_features.shape[0] or h.shape[-3:] != z_features.shape[-3:]:
            raise ValueError(
                "cotangent/z feature geometry mismatch: "
                f"{tuple(h.shape)}/{tuple(z_features.shape)}"
            )
        gate = torch.tanh(self.gate(z_features))
        # For fixed z this is h + L2(diag(g(z)) L1(h)): exactly linear in h.
        return h + self.out_map(gate * self.in_map(h))


class DecoderPullbackSurrogate(nn.Module):
    """Predict the raw decoder pullback ``J_decoder(z)^T v``.

    The z tower may be nonlinear.  The v tower is structurally linear and
    contains no bias, activation, max-pool, or sample-dependent normaliser.
    The returned tensor has the same ``[B,F,C,H,W]`` layout as ``z``.
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
        pyramid_channels: Sequence[int] = (24, 48, 96),
        z_width: int = 96,
        num_blocks: int = 4,
        multiscale_z_gating: bool = False,
    ) -> None:
        super().__init__()
        self.latent_shape = (
            int(latent_frames), int(latent_channels),
            int(latent_height), int(latent_width),
        )
        self.cotangent = SignedCotangentPyramid(
            channels=pyramid_channels,
            pixel_frames=pixel_frames,
            pixel_height=pixel_height,
            pixel_width=pixel_width,
            latent_frames=latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
        )
        width = self.cotangent.out_channels
        z_width = int(z_width)
        if z_width <= 0 or int(num_blocks) <= 0:
            raise ValueError("z_width and num_blocks must be positive")
        self.z_tower = nn.Sequential(
            nn.Conv3d(int(latent_channels), z_width, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(z_width, z_width, 3, padding=1),
            nn.SiLU(),
        )
        dilation_cycle = (1, 2, 4, 2, 1)
        self.blocks = nn.ModuleList([
            ZConditionedLinearPullbackBlock(
                width, z_width, dilation=dilation_cycle[index % 5],
            )
            for index in range(int(num_blocks))
        ])
        self.output = nn.Conv3d(
            width, int(latent_channels), 3, padding=1, bias=False,
        )
        # Keep this allocation after every v1 module.  With the default False,
        # no module (and therefore no parameter or RNG draw) is added, so old
        # seeded initialisation and state dictionaries are preserved exactly.
        self.multiscale_z_gating = bool(multiscale_z_gating)
        self.multiscale_z_gates: Optional[nn.ModuleList]
        if self.multiscale_z_gating:
            self.multiscale_z_gates = nn.ModuleList([
                nn.Conv3d(z_width, stage_width, 1, bias=True)
                for stage_width in self.cotangent.channels
            ])
        else:
            self.multiscale_z_gates = None

    @property
    def num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def unit_pixel_cotangent(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Return the fixed cotangent for the same-parameter z-only control.

        Passing this tensor through :meth:`forward` exercises every learned
        parameter and operation in the conditional architecture, while
        deliberately removing sample-specific ``v`` identity.  It is thus a
        stricter compute/parameter-matched control than a separate z-only
        network.  The returned tensor is a fresh, graph-free constant.
        """
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        parameter = next(self.parameters())
        return torch.ones(
            (int(batch_size), *self.cotangent.pixel_shape),
            device=parameter.device if device is None else device,
            dtype=parameter.dtype if dtype is None else dtype,
        )

    def forward_unit_control(self, latent: torch.Tensor) -> torch.Tensor:
        """Run the exact model with a fixed unit cotangent instead of sample v."""
        unit = self.unit_pixel_cotangent(
            int(latent.shape[0]), device=latent.device, dtype=latent.dtype,
        )
        return self.forward(latent, unit)

    def _make_multiscale_stage_gates(
        self,
        z_features: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, ...]]:
        """Construct z-only gates in Conv3d layout for each pyramid stage."""
        if self.multiscale_z_gates is None:
            return None
        gates = []
        for projection, stage_grid in zip(
            self.multiscale_z_gates, self.cotangent.stage_grids,
        ):
            # Projection at the latent grid is materially cheaper than
            # expanding all z_width channels to the first high-res stage.
            # Resizing the projected logits still supplies a spatially varying
            # z gate at that stage and does not involve v.
            logits = projection(z_features)
            if tuple(logits.shape[-3:]) != tuple(stage_grid):
                logits = F.interpolate(
                    logits, size=stage_grid, mode="trilinear",
                    align_corners=False,
                )
            # Identity-centred gating avoids suppressing the signed v stream at
            # initialisation while allowing z-dependent amplification/sign.
            gates.append(1.0 + torch.tanh(logits))
        return tuple(gates)

    def forward(
        self,
        latent: torch.Tensor,
        pixel_cotangent: torch.Tensor,
    ) -> torch.Tensor:
        if latent.dim() != 5:
            raise ValueError(
                f"latent must be [B,F,C,H,W], got {tuple(latent.shape)}"
            )
        if tuple(latent.shape[1:]) != self.latent_shape:
            raise ValueError(
                f"latent must have trailing shape {self.latent_shape}, got "
                f"{tuple(latent.shape[1:])}"
            )
        if int(pixel_cotangent.shape[0]) != int(latent.shape[0]):
            raise ValueError(
                "latent/pixel-cotangent batch mismatch: "
                f"{int(latent.shape[0])}/{int(pixel_cotangent.shape[0])}"
            )
        parameter = next(self.parameters())
        z = latent.to(
            device=parameter.device, dtype=parameter.dtype,
        ).permute(0, 2, 1, 3, 4).contiguous()
        z_features = self.z_tower(z)
        stage_gates = self._make_multiscale_stage_gates(z_features)
        # Public [B,F,C,H,W] -> Conv3d [B,C,F,H,W].
        h = self.cotangent(
            pixel_cotangent, stage_gates=stage_gates,
        ).permute(0, 2, 1, 3, 4).contiguous()
        for block in self.blocks:
            h = block(h, z_features)
        out = self.output(h)
        return out.permute(0, 2, 1, 3, 4).contiguous().float()
