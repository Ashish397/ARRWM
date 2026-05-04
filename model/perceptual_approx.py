"""Latent-space dense-output approximators for video metrics.

A single architecture used for THREE different approximators:

  1. ``mse_approx``     — predicts dense per-token MSE between
                           decoded gen pixels and decoded GT pixels.
  2. ``lpips_approx``   — predicts dense per-token LPIPS distance
                           between decoded gen pixels and decoded GT
                           pixels.
  3. ``gan_d_approx``   — predicts dense per-token pixel-discriminator
                           output map for ``decode(gen_lat)`` (the
                           classification logit per spatial position).

Why a single architecture for all three?
  * They take the same inputs (``gen_lat``, ``gt_lat``) and produce
    the same shape (dense per-frame-per-token field).
  * They differ only in the regression target. Composing them as
    instances of the same class makes the trainer plumbing uniform.

Why 3D CNN (not transformer)?
  * Video signals have strong local spatio-temporal correlation —
    3D conv kernels with explicit spatial structure preserve that
    correlation from latent → token grid.
  * No global attention means the per-token output is a function of
    a local receptive field (input spatial neighborhood), which
    matches how pixel-level metrics like MSE/LPIPS factor.
  * Future-compatible with VLPIPS and other video-native perceptual
    losses (3D feature stacks → 3D CNN regression target).

Why dense output (per-frame-per-spatial-token)?
  * The approxes are SURROGATES, not exact metrics. High-granularity
    supervision (per-token target) gives the optimizer many more
    constraints per iter, which constrains the value field densely
    enough that the GRADIENT (what the gen consumes via .mean()) is
    meaningful by Lipschitz interpolation.
  * Per-token predictions also let each spatial region of the latent
    learn its own perceptual / GAN contribution, mirroring the
    spatial structure of the pixel-space metric.

Architecture
------------
Input:   ``gen_lat: [B, F, C=16, H=60, W=104]``,
         ``gt_lat:  [B, F, C=16, H=60, W=104]``
Concat:  along channel dim → ``[B, F, 2C=32, H, W]``
Stem:    3D conv pyramid, 3× stride-(1, 2, 2) → ``[B, d_model, F, H/8, W/8]``
         (matches LatentSAM2Critic's stem so token grids are
         comparable across approxes).
Body:    ``num_blocks`` 3D residual conv blocks (Conv3d × 2 + GN +
         SiLU + residual). Spatial kernel size 3, temporal kernel
         size 3 — captures local spatio-temporal structure.
Head:    Conv3d(d_model → 1, kernel 1, zero-init) → ``[B, 1, F,
         H/8, W/8]`` → squeeze → ``[B, F, H/8, W/8]`` dense scalar.

Gen-side use
------------
After warmup, the gen consumes::

    perceptual_loss = approx(gen_lat_grad, gt_lat_no_grad).mean()

Backward through the (small) approx to the gen. No VAE involved.

Training the approx
-------------------
Every iter, compute:
  * Decode ``gen_lat`` and ``gt_lat`` to pixels via VAE (no_grad).
  * Compute the per-token target via spatial+temporal mean-pool of
    the per-pixel metric (squared-error for MSE, VGG-feature distance
    for LPIPS, disc-logit map for GAN approx).
  * Forward the approx with grad enabled.
  * Loss = ``MSE(approx_pred, target)`` (dense per-token MSE) +
    ``β * MSE(approx_pred.mean(), target.mean())`` (mean-aligning).
  * Backward + step.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _gn(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """GroupNorm with the largest divisor of ``channels`` that does
    not exceed ``max_groups``. Matches ADM's GN sizing convention."""
    g = max_groups
    while g > 1 and channels % g != 0:
        g //= 2
    return nn.GroupNorm(num_groups=g, num_channels=channels)


class _Res3DBlock(nn.Module):
    """Residual 3D conv block: Conv3d → GN → SiLU → Conv3d → GN → +res.

    Kernel size 3 in all three dims (T, H, W). Padding 1 keeps shape.
    Skip connection projects channels via 1×1×1 if needed.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_t: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        pad_t = kernel_t // 2
        self.conv1 = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(kernel_t, 3, 3),
            stride=1,
            padding=(pad_t, 1, 1),
        )
        self.norm1 = _gn(out_channels)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=(kernel_t, 3, 3),
            stride=1,
            padding=(pad_t, 1, 1),
        )
        self.norm2 = _gn(out_channels)
        self.act2 = nn.SiLU()
        self.dropout = (
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        )
        if in_channels != out_channels:
            self.skip = nn.Conv3d(
                in_channels, out_channels,
                kernel_size=1, bias=False,
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = h + self.skip(x)
        return self.act2(h)


class PerceptualApprox(nn.Module):
    """Dense per-token video-metric approximator (3D CNN).

    Architecture: 3D conv stem (downsample spatial 8×) + N residual
    3D conv blocks at the token resolution + 1×1×1 head.

    Constructor args:
        in_channels: per-latent channel count (default 16 for WAN VAE).
            The model receives ``2*in_channels`` after concat.
        d_model: 3D CNN hidden dim (default 256).
        num_blocks: number of residual 3D conv blocks at the token
            resolution (default 4).
        kernel_t: temporal conv kernel size in body blocks (default 3).
        dropout: dropout3d applied between conv1 and conv2 in each
            residual block (default 0.0).
        zero_init_head: zero-init the final 1×1×1 head so the model
            starts as the constant-zero field.

    Forward signature:
        ``forward(gen_lat, gt_lat) -> [B, F, H_token, W_token]`` dense
        scalar prediction. Both inputs ``[B, F, C, H, W]`` with
        matching shapes (gen with grad if used in gen-side loss; gt
        always detached).
    """

    def __init__(
        self,
        in_channels: int = 16,
        d_model: int = 256,
        num_blocks: int = 4,
        kernel_t: int = 3,
        dropout: float = 0.0,
        zero_init_head: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.d_model = int(d_model)
        self.num_blocks = int(num_blocks)
        self.kernel_t = int(kernel_t)

        # 3D conv stem: project (gen, gt) concat to d_model and reduce
        # spatial 8× via 3 stride-(1, 2, 2) convs. Matches the
        # LatentSAM2Critic stem so token grids are aligned.
        c_in = 2 * self.in_channels
        c1 = max(d_model // 4, 64)
        c2 = max(d_model // 2, 128)
        c3 = d_model
        pad_t = self.kernel_t // 2
        self.stem = nn.Sequential(
            nn.Conv3d(
                c_in, c1,
                kernel_size=(self.kernel_t, 3, 3),
                stride=(1, 2, 2),
                padding=(pad_t, 1, 1),
            ),
            _gn(c1),
            nn.SiLU(),
            nn.Conv3d(
                c1, c2,
                kernel_size=(self.kernel_t, 3, 3),
                stride=(1, 2, 2),
                padding=(pad_t, 1, 1),
            ),
            _gn(c2),
            nn.SiLU(),
            nn.Conv3d(
                c2, c3,
                kernel_size=(self.kernel_t, 3, 3),
                stride=(1, 2, 2),
                padding=(pad_t, 1, 1),
            ),
            _gn(c3),
            nn.SiLU(),
        )

        # 3D residual conv body at the token resolution. Each block
        # has its own receptive field grow in spatial+temporal — by
        # block N, the token effectively "sees" a (2N+1)³ neighborhood.
        self.blocks = nn.ModuleList([
            _Res3DBlock(
                in_channels=c3,
                out_channels=c3,
                kernel_t=self.kernel_t,
                dropout=dropout,
            )
            for _ in range(self.num_blocks)
        ])

        # Final per-token head: 1×1×1 conv to scalar.
        self.head = nn.Conv3d(c3, 1, kernel_size=1, bias=True)
        if zero_init_head:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        gen_lat: torch.Tensor,
        gt_lat: torch.Tensor,
    ) -> torch.Tensor:
        """Forward over (gen, gt) latent pair.

        Args:
            gen_lat: ``[B, F, C, H, W]``. Channels must match
                ``in_channels``.
            gt_lat: ``[B, F, C, H, W]``. Same shape as ``gen_lat``.

        Returns:
            ``[B, F, H_token, W_token]`` dense scalar predictions.
        """
        if gen_lat.shape != gt_lat.shape:
            raise ValueError(
                f"PerceptualApprox: gen_lat shape {gen_lat.shape} != "
                f"gt_lat shape {gt_lat.shape}."
            )
        if gen_lat.dim() != 5:
            raise ValueError(
                f"PerceptualApprox expects [B, F, C, H, W]; got "
                f"{gen_lat.shape}."
            )
        B, F_, C, H, W = gen_lat.shape
        if C != self.in_channels:
            raise ValueError(
                f"PerceptualApprox in_channels={self.in_channels} but "
                f"got C={C}."
            )
        # Concat along channel dim: [B, F, 2C, H, W].
        x = torch.cat([gen_lat, gt_lat], dim=2)
        # [B, F, 2C, H, W] → [B, 2C, F, H, W] for Conv3d.
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        # [B, d_model, F, H', W'] → [B, 1, F, H', W'] → [B, F, H', W'].
        x = self.head(x)
        return x.squeeze(1).float()
