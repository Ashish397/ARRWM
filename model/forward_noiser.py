"""ForwardNoiser — learned 1-step causal-AR-noise (CARN) increment.

Trained to map rollout-1 chunks (CARN level n) to rollout-2 chunks (CARN
level n+1) at the same temporal position. At inference time it's applied
iteratively to GT chunks to synthesize "GT noised to CARN level k", which
the online real_score's aux-teacher pass consumes as its input
distribution. This replaces the v21 alt-head's failed
"alt(noise(GT)) ≈ GT" application path (causal_AR_dir_rms=0 across v21-v28).

Design notes
------------
- Dedicated network (NOT sharing fake_score's backbone). Backbone shared
  with fake_score is trained as a STUDENT-DISTRIBUTION DENOISER; feeding
  it clean GT for forward-noising is OOD and the alt-head projection
  can't override the backbone's prior. A separate small net avoids the
  distributional fight entirely.

- CARN-step conditioning via FiLM (scale + shift per block). The forward
  noiser knows what step it's noising up FROM, so it can learn step-
  dependent drift (probably non-linear over training).

- 3D ConvNet (NOT a transformer) because: (1) the task is highly local
  (local AR-noise increment), (2) much smaller param count than a
  transformer of equivalent receptive field, (3) factorized spatial+
  temporal kernels keep compute manageable.

- Default capacity (~30M params) at hidden_dim=512, 4 blocks. Adjustable
  via config.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding for an integer step value."""

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(
                f"_SinusoidalEmbedding dim must be even; got {dim}."
            )
        self.dim = dim

    def forward(self, step: torch.Tensor) -> torch.Tensor:
        # step: [B] long; returns [B, dim].
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, half, device=step.device, dtype=torch.float32)
            / half
        )
        args = step.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


class _CARNStepConditioning(nn.Module):
    """Map CARN-step int → per-block FiLM (scale, shift) modulation."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.embed = _SinusoidalEmbedding(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
        )

    def forward(self, step: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.embed(step)
        # The sinusoidal embedding always returns fp32 for numerical
        # stability of the sin/cos math. The MLP's dtype may have been
        # set externally (e.g. trainer cast to bf16 to match the rest
        # of the model). Cast emb to match the MLP's input dtype so
        # matmul doesn't fail with mixed-precision tensors.
        mlp_dtype = self.mlp[0].weight.dtype
        if emb.dtype != mlp_dtype:
            emb = emb.to(dtype=mlp_dtype)
        out = self.mlp(emb)
        scale, shift = out.chunk(2, dim=-1)
        return scale, shift


class _ResBlock3D(nn.Module):
    """Factorized 3D residual block: spatial 1x3x3 conv + temporal 3x1x1
    conv + FiLM modulation. Smaller than a full 3x3x3 block, with
    comparable receptive field for local AR-noise increment learning."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv_spatial = nn.Conv3d(
            channels, channels, kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
        )
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv_temporal = nn.Conv3d(
            channels, channels, kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
        )
        self.norm3 = nn.GroupNorm(8, channels)
        self.conv_out = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        # x: [B, C, F, H, W]
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv_spatial(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv_temporal(h)
        # FiLM modulation. scale/shift: [B, C]; broadcast across F,H,W.
        h = self.norm3(h)
        s = scale.view(scale.shape[0], scale.shape[1], 1, 1, 1)
        b = shift.view(shift.shape[0], shift.shape[1], 1, 1, 1)
        h = h * (1.0 + s) + b
        h = F.silu(h)
        h = self.conv_out(h)
        return x + h


class ForwardNoiser(nn.Module):
    """Learned one-step CARN forward-noiser.

    Args:
        latent_channels: number of input/output latent channels (16 for
            Wan2.1 VAE latents).
        hidden_dim: internal feature dimension. Default 512.
        num_blocks: number of residual blocks. Default 4.
        max_carn_step: max CARN step value the conditioning embedding
            supports. Default 16 (covers rollouts up to 16 AR steps).

    Input:  ``[B, F, C, H, W]`` latents at CARN level n + ``carn_step``
            int (the n that the input represents).
    Output: ``[B, F, C, H, W]`` latents at CARN level n+1.
    """

    def __init__(
        self,
        latent_channels: int = 16,
        hidden_dim: int = 512,
        num_blocks: int = 4,
        max_carn_step: int = 16,
    ):
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = int(num_blocks)
        self.max_carn_step = int(max_carn_step)
        self.in_proj = nn.Conv3d(
            self.latent_channels, self.hidden_dim, kernel_size=1,
        )
        # Independent CARN conditioning per block (each block gets its
        # own scale/shift). Cheap (just embeddings).
        self.carn_cond = nn.ModuleList([
            _CARNStepConditioning(dim=128, hidden_dim=self.hidden_dim)
            for _ in range(self.num_blocks)
        ])
        self.blocks = nn.ModuleList([
            _ResBlock3D(channels=self.hidden_dim)
            for _ in range(self.num_blocks)
        ])
        self.out_norm = nn.GroupNorm(8, self.hidden_dim)
        self.out_proj = nn.Conv3d(
            self.hidden_dim, self.latent_channels, kernel_size=1,
        )
        # Zero-init the output projection so at init the noiser is the
        # identity (no noise added). This makes early-training behaviour
        # well-conditioned: F(x, 0) ≈ x.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        x: torch.Tensor,
        carn_step: torch.Tensor,
        residual: bool = True,
    ) -> torch.Tensor:
        """Apply 1-step CARN forward noising.

        Args:
            x: ``[B, F, C, H, W]`` latents at CARN level n.
            carn_step: ``[B]`` long tensor of CARN level n (= the step
                BEING noised up FROM; output is at n+1).
            residual: when True (default) returns ``x + delta`` where
                ``delta`` is the learned increment. With zero-init
                ``out_proj`` and ``residual=True``, the noiser starts as
                identity. When False, returns ``delta`` only (the raw
                increment; useful for diagnostics).
        """
        if x.dim() != 5:
            raise ValueError(
                f"ForwardNoiser expects 5D input [B,F,C,H,W]; got {x.shape}"
            )
        if carn_step.dim() != 1 or carn_step.shape[0] != x.shape[0]:
            raise ValueError(
                f"carn_step must be [B] matching x.shape[0]={x.shape[0]}; "
                f"got {carn_step.shape}"
            )
        # Clamp carn_step to embedding range.
        carn_step = carn_step.clamp(0, self.max_carn_step)
        # Match input dtype to module dtype at the boundary so Conv3d
        # bias / weight matmuls don't fail under mixed precision. The
        # rest of the model runs in bf16; the trainer casts this module
        # to bf16 during DDP wrap. Cast input accordingly, do the work,
        # then cast the residual back to the input dtype.
        in_dtype = x.dtype
        module_dtype = self.in_proj.weight.dtype
        x_cast = x.to(dtype=module_dtype) if in_dtype != module_dtype else x
        # Reshape to [B, C, F, H, W] for 3D convs.
        h = x_cast.permute(0, 2, 1, 3, 4).contiguous()
        h = self.in_proj(h)
        for i, block in enumerate(self.blocks):
            scale, shift = self.carn_cond[i](carn_step)
            h = block(h, scale, shift)
        h = self.out_norm(h)
        h = F.silu(h)
        delta = self.out_proj(h)
        # Back to [B, F, C, H, W] and to input dtype for residual add.
        delta = delta.permute(0, 2, 1, 3, 4).contiguous()
        if delta.dtype != in_dtype:
            delta = delta.to(dtype=in_dtype)
        if residual:
            return x + delta
        return delta
