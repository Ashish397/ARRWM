"""State-conditioned local VJP operator, structurally linear in cotangent."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["IdentityAnchoredLocalVJP", "LocalLinearVJPSurrogate"]


class _StateGatedLinearBlock(nn.Module):
    def __init__(self, width: int, state_channels: int, dilation: int) -> None:
        super().__init__()
        self.in_map = nn.Conv3d(
            width, width, 3, padding=dilation, dilation=dilation, bias=False,
        )
        self.out_map = nn.Conv3d(
            width, width, 3, padding=dilation, dilation=dilation, bias=False,
        )
        self.gate = nn.Conv3d(state_channels, width, 1, bias=True)

    def forward(self, value: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        gate = 1.0 + torch.tanh(self.gate(state))
        return value + self.out_map(gate * self.in_map(value))


class LocalLinearVJPSurrogate(nn.Module):
    """Approximate one decoder stage's ``J_f(h)^T lambda``.

    State may affect only multiplicative gates. Every operation from incoming
    cotangent to output is bias-free and linear for fixed state, enforcing
    zero, oddness, homogeneity and additivity by construction.
    """

    def __init__(
        self,
        *,
        state_channels: int,
        packed_cotangent_channels: int,
        output_channels: int,
        width: int = 48,
        blocks: int = 2,
    ) -> None:
        super().__init__()
        for name, value in (
            ("state_channels", state_channels),
            ("packed_cotangent_channels", packed_cotangent_channels),
            ("output_channels", output_channels), ("width", width),
            ("blocks", blocks),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        self.state_channels = int(state_channels)
        self.packed_cotangent_channels = int(packed_cotangent_channels)
        self.output_channels = int(output_channels)
        self.input = nn.Conv3d(
            self.packed_cotangent_channels, int(width), 1, bias=False,
        )
        dilations = (1, 2, 1, 2)
        self.blocks = nn.ModuleList([
            _StateGatedLinearBlock(
                int(width), self.state_channels, dilations[index % len(dilations)],
            )
            for index in range(int(blocks))
        ])
        self.output = nn.Conv3d(
            int(width), self.output_channels, 1, bias=False,
        )

    @property
    def num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, state: torch.Tensor, packed_cotangent: torch.Tensor) -> torch.Tensor:
        if state.dim() != 5 or packed_cotangent.dim() != 5:
            raise ValueError("state/cotangent must use [B,C,F,H,W] layout")
        if int(state.shape[1]) != self.state_channels:
            raise ValueError(f"state channels {state.shape[1]} != {self.state_channels}")
        if int(packed_cotangent.shape[1]) != self.packed_cotangent_channels:
            raise ValueError(
                f"cotangent channels {packed_cotangent.shape[1]} != "
                f"{self.packed_cotangent_channels}"
            )
        if state.shape[0] != packed_cotangent.shape[0] or state.shape[-3:] != packed_cotangent.shape[-3:]:
            raise ValueError("state/cotangent geometry mismatch")
        if state.requires_grad or packed_cotangent.requires_grad:
            raise ValueError("local VJP bank inputs must be detached")
        parameter = next(self.parameters())
        state = state.detach().to(device=parameter.device, dtype=parameter.dtype)
        value = packed_cotangent.detach().to(device=parameter.device, dtype=parameter.dtype)
        value = self.input(value)
        for block in self.blocks:
            value = block(value, state)
        return self.output(value).float()


class IdentityAnchoredLocalVJP(nn.Module):
    """Learn only a residual block's non-shortcut VJP correction.

    For WAN residual blocks with equal input/output width, the shortcut VJP is
    exactly the identity.  Keeping it analytically prevents a small surrogate
    from wasting evidence and capacity rediscovering the dominant transport.
    The learned correction remains exactly linear in the cotangent.
    """

    def __init__(
        self, *, state_channels: int, width: int = 48, blocks: int = 2,
    ) -> None:
        super().__init__()
        self.state_channels = int(state_channels)
        self.correction = LocalLinearVJPSurrogate(
            state_channels=self.state_channels,
            packed_cotangent_channels=self.state_channels,
            output_channels=self.state_channels,
            width=int(width), blocks=int(blocks),
        )
        # Update zero is the exact shortcut-only diagnostic.  The output layer
        # receives gradients immediately; deeper correction weights begin
        # receiving gradients after that first update.
        nn.init.zeros_(self.correction.output.weight)

    @property
    def num_params(self) -> int:
        return self.correction.num_params

    def forward(self, state: torch.Tensor, cotangent: torch.Tensor) -> torch.Tensor:
        if int(cotangent.shape[1]) != self.state_channels:
            raise ValueError("identity shortcut requires equal boundary widths")
        return cotangent.detach().float() + self.correction(state, cotangent)
