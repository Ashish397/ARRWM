"""Structured WAN residual-block VJP with optional tied low-rank convs."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["StructuredResidualVJP", "TiedLowRankCausalTranspose"]


def _crop_causal_padding(value: torch.Tensor, conv, input_shape: Sequence[int]):
    # CausalConv3d records PyTorch F.pad order:
    # (w_left, w_right, h_left, h_right, t_left, t_right).
    w_left, _w_right, h_left, _h_right, t_left, _t_right = conv._padding
    frames, height, width = map(int, input_shape[-3:])
    return value[
        ..., t_left:t_left + frames,
        h_left:h_left + height, w_left:w_left + width,
    ]


def _exact_causal_transpose(
    gradient: torch.Tensor, conv, input_shape: Sequence[int],
) -> torch.Tensor:
    padded = F.conv_transpose3d(
        gradient, conv.weight.float(), bias=None,
        stride=conv.stride, padding=0, output_padding=0,
        groups=conv.groups, dilation=conv.dilation,
    )
    return _crop_causal_padding(padded, conv, input_shape)


class TiedLowRankCausalTranspose(nn.Module):
    """Output-channel SVD of a frozen WAN convolution, used in transpose."""

    def __init__(self, conv, rank: int, *, oversample: int = 8, niter: int = 4):
        super().__init__()
        weight = conv.weight.detach().float()
        out_channels = int(weight.shape[0])
        maximum = min(out_channels, int(weight[0].numel()))
        if not 1 <= int(rank) <= maximum:
            raise ValueError(f"rank {rank} outside [1,{maximum}]")
        matrix = weight.reshape(out_channels, -1)
        q = min(maximum, int(rank) + int(oversample))
        # This factorization is performed once.  Fixed seed makes the tied
        # approximation reproducible and does not involve audit examples.
        devices = [matrix.device] if matrix.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(28082026 + out_channels * 17 + int(rank))
            u, singular, v = torch.svd_lowrank(matrix, q=q, niter=int(niter))
        u = u[:, :int(rank)].contiguous()
        kernel = (
            singular[:int(rank), None] * v[:, :int(rank)].T
        ).reshape(int(rank), *weight.shape[1:]).contiguous()
        self.register_buffer("output_basis", u)
        self.register_buffer("kernel", kernel)
        self.stride = tuple(conv.stride)
        self.dilation = tuple(conv.dilation)
        self.groups = int(conv.groups)
        self.padding = tuple(conv._padding)

    def forward(
        self, gradient: torch.Tensor, input_shape: Sequence[int],
    ) -> torch.Tensor:
        projection = self.output_basis.T[:, :, None, None, None]
        reduced = F.conv3d(gradient.float(), projection)
        padded = F.conv_transpose3d(
            reduced, self.kernel, bias=None, stride=self.stride,
            padding=0, output_padding=0, groups=self.groups,
            dilation=self.dilation,
        )
        w_left, _wr, h_left, _hr, t_left, _tr = self.padding
        frames, height, width = map(int, input_shape[-3:])
        return padded[
            ..., t_left:t_left + frames,
            h_left:h_left + height, w_left:w_left + width,
        ]


def _rms_vjp(value: torch.Tensor, norm, gradient: torch.Tensor) -> torch.Tensor:
    value = value.float()
    scaled = gradient.float() * norm.gamma.float() * math.sqrt(value.shape[1])
    magnitude = value.square().sum(dim=1, keepdim=True).sqrt().clamp_min(1.0e-12)
    radial = (scaled * value).sum(dim=1, keepdim=True)
    return scaled / magnitude - value * radial / magnitude.pow(3)


def _silu_derivative(value: torch.Tensor) -> torch.Tensor:
    sigmoid = torch.sigmoid(value.float())
    return sigmoid * (1.0 + value.float() * (1.0 - sigmoid))


class StructuredResidualVJP(nn.Module):
    """Apply a WAN residual block's backward rule to a detached cotangent.

    With ``rank=None`` this is the exact manual VJP.  With a positive rank,
    only the two expensive convolution transposes are approximated by tied
    output-channel SVDs.  Shortcut, RMS-normalization derivatives, SiLU
    derivatives and forward state remain exact.  Both modes are exactly linear
    in the incoming cotangent for fixed state.
    """

    def __init__(self, block, *, rank: Optional[int] = None) -> None:
        super().__init__()
        self.block = block.eval().requires_grad_(False)
        self.rank = None if rank is None else int(rank)
        conv1, conv2 = self.block.residual[2], self.block.residual[6]
        self.conv1_transpose = (
            None if self.rank is None else TiedLowRankCausalTranspose(conv1, self.rank)
        )
        self.conv2_transpose = (
            None if self.rank is None else TiedLowRankCausalTranspose(conv2, self.rank)
        )

    def _transpose(self, gradient, conv, approximation, input_shape):
        if approximation is None:
            return _exact_causal_transpose(gradient, conv, input_shape)
        return approximation(gradient, input_shape)

    def forward(self, state: torch.Tensor, cotangent: torch.Tensor) -> torch.Tensor:
        if state.requires_grad or cotangent.requires_grad:
            raise ValueError("structured VJP inputs must be detached")
        state = state.detach().float()
        cotangent = cotangent.detach().float()
        norm1, conv1 = self.block.residual[0], self.block.residual[2]
        norm2, conv2 = self.block.residual[3], self.block.residual[6]

        # In deployment these are tapped from the detached WAN forward.  The
        # benchmark recomputes the one missing internal activation to avoid a
        # second large persistent state bank.
        normalized1 = norm1(state)
        activation1 = F.silu(normalized1)
        conv1_state = conv1(activation1)
        normalized2 = norm2(conv1_state)

        gradient = self._transpose(
            cotangent, conv2, self.conv2_transpose, conv1_state.shape,
        )
        gradient = gradient * _silu_derivative(normalized2)
        gradient = _rms_vjp(conv1_state, norm2, gradient)
        gradient = self._transpose(
            gradient, conv1, self.conv1_transpose, state.shape,
        )
        gradient = gradient * _silu_derivative(normalized1)
        gradient = _rms_vjp(state, norm1, gradient)

        if isinstance(self.block.shortcut, nn.Identity):
            shortcut = cotangent
        else:
            shortcut = _exact_causal_transpose(
                cotangent, self.block.shortcut, state.shape,
            )
        return (gradient + shortcut).float()
