"""Latent-space surrogate critic for SAM2-pixel disc distillation.

Architectural fix for the OOM in `_compute_r3gan_losses` — the G-side
gradient path used to cross VAE decode + SAM2 forward in graph-on
mode (~10 GB workspace × 1 grad-active forward per iter). With the
distilled-critic flow, the generator's autograd graph never crosses
VAE or SAM2: it only crosses this small latent-space module.

This is the same pattern as ``_compute_action_critic_losses`` (frozen
"expensive teacher" supervises a cheap latent-space critic that
delivers the gradient to the generator), with one extra term
beyond value-only distillation:

  * **Value distillation** (standard): ``MSE(g_φ(z), f(z).detach())``
    — the critic's logit at z matches the pixel disc's logit.
  * **Gradient distillation** (Sobolev-style): ``MSE(∇_z g_φ(z),
    ∇_z f(z).detach())`` — the critic's gradient field at z matches
    the pixel disc's gradient field. Without this, `g ≈ f` doesn't
    imply `∇g ≈ ∇f`; the generator could learn to fool the critic
    on values without improving pixel-perceptual quality.

The critic is intentionally larger than the action critic
(``~30M params`` vs ~few M) because the regression target — a
SAM2-feature disc logit and its full gradient field — is rich.

Architecture
------------
Input:  ``[B, F, C=16, H=60, W=104]`` latent video.
Stem:   3D conv pyramid, 3× stride-(1, 2, 2) convs to project the
        latent to ``d_model=512`` and reduce spatial 8× (60×104 →
        ~7×13). Each conv block is Conv3d → GroupNorm(32) → SiLU.
        The temporal dim is preserved through the stem.
Body:   ``num_blocks`` (default 4) Transformer encoder blocks with
        full space-time self-attention (treats every (frame, h, w)
        token equally). Per-block: LN → MultiHeadAttn → residual →
        LN → MLP(d_model → 4·d_model → d_model) → residual.
        Learnable per-frame and per-spatial positional embeddings.
Head:   Per-frame mean-pool over spatial tokens → LayerNorm →
        Linear(d_model, 1) (zero-init) → frame_pool (mean / max /
        topk_mean, matching the pixel disc's contract) → ``[B]``
        scalar logit.

Gen-side use
------------
After the warmup steps configured at the trainer level, the critic
provides the GAN gradient to the generator:

    pred_image_for_g (grad-attached latent)  →  critic forward  →
    ``-critic(pred_image_for_g).mean()`` (negation for G-min D-max)

Backward through the critic to ``pred_image_for_g`` to the generator.
The critic's parameters are toggled to ``requires_grad=False`` for
the duration of this forward so the gen-step's backward doesn't
accidentally write to critic-side gradients (the critic optimizer
is updated only by the value+grad distillation step).
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """GroupNorm with the largest divisor of ``channels`` that does
    not exceed ``max_groups``. Matches ADM's GN sizing convention."""
    g = max_groups
    while g > 1 and channels % g != 0:
        g //= 2
    return nn.GroupNorm(num_groups=g, num_channels=channels)


class _MultiHeadSelfAttention(nn.Module):
    """Hand-rolled multi-head self-attention.

    Used instead of ``nn.MultiheadAttention`` because the gradient
    distillation path computes ``L_grad.backward()`` which requires
    second-order grad through the attention. PyTorch's SDPA backends
    select FlashAttention by default on some devices, which lacks a
    double-backward kernel — we want the explicit softmax(QK^T/√d)V
    path which has working double-backward on every device.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by num_heads={num_heads}."
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        B, N, D = x.shape
        qkv = self.qkv(x)  # [B, N, 3D]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        # → [3, B, num_heads, N, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = self.head_dim ** -0.5
        # [B, num_heads, N, N]
        scores = torch.matmul(q * scale, k.transpose(-2, -1))
        attn = scores.softmax(dim=-1)
        if self.training and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout)
        out = torch.matmul(attn, v)  # [B, num_heads, N, head_dim]
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


class _TransformerBlock(nn.Module):
    """Standard pre-LN transformer encoder block (hand-rolled MHA so
    second-order grad works for the gradient-distillation backward)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(d_model)
        ff_hidden = d_model * max(1, int(ff_mult))
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.GELU(),
            nn.Linear(ff_hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class LatentSAM2Critic(nn.Module):
    """Latent-space surrogate critic distilled from a pixel-space
    SAM2 disc.

    See module docstring for the value+gradient distillation rationale.

    Constructor args:
        in_channels: latent channel count (default 16 for WAN VAE).
        d_model: transformer hidden dim (default 512).
        num_blocks: number of transformer blocks (default 4 ≈ 24M
            params for d_model=512; bump to 5-6 for ~30M).
        num_heads: attention heads (default 8).
        ff_mult: MLP expansion ratio (default 4).
        dropout: applied to attention weights and the MLP.
        max_frames: max temporal tokens (default 64; sized to allow
            future longer rollouts).
        max_spatial: max spatial tokens after the stem (default 256;
            sized to allow ``image_resolution`` up to ~1024 with the
            current 8× spatial reduction).
        frame_pool: ``mean | max | topk_mean`` — matches the pixel
            disc's frame_pool contract for symmetry.
        frame_pool_topk: only consulted when ``frame_pool='topk_mean'``.
    """

    def __init__(
        self,
        *,
        in_channels: int = 16,
        d_model: int = 512,
        num_blocks: int = 4,
        num_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.0,
        max_frames: int = 64,
        max_spatial: int = 256,
        frame_pool: str = "mean",
        frame_pool_topk: int = 4,
    ) -> None:
        super().__init__()
        if frame_pool not in ("mean", "max", "topk_mean"):
            raise ValueError(
                f"frame_pool must be 'mean' | 'max' | 'topk_mean'; "
                f"got {frame_pool!r}."
            )
        self.in_channels = int(in_channels)
        self.d_model = int(d_model)
        self.num_blocks = int(num_blocks)
        self.num_heads = int(num_heads)
        self.frame_pool = frame_pool
        self.frame_pool_topk = max(1, int(frame_pool_topk))
        self.max_frames = int(max_frames)
        self.max_spatial = int(max_spatial)

        # 3D conv stem: project channels to d_model and reduce spatial 8×.
        # Temporal dim is preserved (stride 1 in T) so token counts in the
        # transformer match the input frame count.
        c1 = max(d_model // 4, 64)
        c2 = max(d_model // 2, 128)
        c3 = d_model
        self.stem = nn.Sequential(
            nn.Conv3d(
                in_channels, c1, kernel_size=3,
                stride=(1, 2, 2), padding=1,
            ),
            _gn(c1),
            nn.SiLU(),
            nn.Conv3d(
                c1, c2, kernel_size=3,
                stride=(1, 2, 2), padding=1,
            ),
            _gn(c2),
            nn.SiLU(),
            nn.Conv3d(
                c2, c3, kernel_size=3,
                stride=(1, 2, 2), padding=1,
            ),
            _gn(c3),
            nn.SiLU(),
        )

        # Positional embeddings — split into per-frame and per-spatial
        # so they're sharable across (frame, spatial) factorizations.
        self.frame_pos = nn.Parameter(
            torch.zeros(1, self.max_frames, 1, d_model)
        )
        nn.init.trunc_normal_(self.frame_pos, std=0.02)
        self.spatial_pos = nn.Parameter(
            torch.zeros(1, 1, self.max_spatial, d_model)
        )
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)

        # Transformer body — full space-time self-attention.
        self.blocks = nn.ModuleList([
            _TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                ff_mult=ff_mult,
                dropout=dropout,
            )
            for _ in range(self.num_blocks)
        ])

        # Output: norm → linear → 1 scalar per frame.
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        # Zero-init the head so the critic starts as the constant-zero
        # logit (matches the disc convention; the gen warmup ramp +
        # value distillation loss train this from zero).
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Forward over a latent video.

        Args:
            latent: ``[B, F, C, H, W]``. Channels must match
                ``in_channels``. ``F`` ≤ ``max_frames``.

        Returns:
            ``[B]`` scalar logits. The frame_pool aggregation matches
            the pixel disc's contract so distilled-mode losses can
            be written as if disc and critic produced the same shape.
        """
        if latent.dim() != 5:
            raise ValueError(
                f"LatentSAM2Critic expects [B, F, C, H, W]; got {latent.shape}."
            )
        B, F_, C, H, W = latent.shape
        if C != self.in_channels:
            raise ValueError(
                f"LatentSAM2Critic in_channels={self.in_channels} but got C={C}."
            )
        if F_ > self.max_frames:
            raise ValueError(
                f"LatentSAM2Critic F={F_} exceeds max_frames={self.max_frames}."
            )
        # [B, F, C, H, W] → [B, C, F, H, W] for Conv3d.
        x = latent.permute(0, 2, 1, 3, 4).contiguous()
        x = self.stem(x)  # [B, d_model, F, H', W']
        D = x.shape[1]
        Hs = x.shape[3]
        Ws = x.shape[4]
        N_spatial = Hs * Ws
        if N_spatial > self.max_spatial:
            raise ValueError(
                f"LatentSAM2Critic spatial tokens {N_spatial} exceed "
                f"max_spatial={self.max_spatial}."
            )
        # [B, d_model, F, H', W'] → [B, F, H'·W', d_model]
        x = x.permute(0, 2, 3, 4, 1).reshape(B, F_, N_spatial, D)
        # Add positional embeddings (broadcast).
        pos_frame = self.frame_pos[:, :F_].to(dtype=x.dtype)
        pos_spatial = self.spatial_pos[:, :, :N_spatial].to(dtype=x.dtype)
        x = x + pos_frame + pos_spatial
        # Flatten to [B, F·N_spatial, d_model] for full space-time attention.
        x = x.reshape(B, F_ * N_spatial, D)
        for block in self.blocks:
            x = block(x)
        # Per-frame mean-pool over spatial tokens.
        x = x.reshape(B, F_, N_spatial, D)
        per_frame = x.mean(dim=2)  # [B, F, d_model]
        per_frame = self.norm_out(per_frame)
        per_frame_logit = self.head(per_frame).squeeze(-1)  # [B, F]
        # Frame-pool — same shapes as the pixel disc.
        if self.frame_pool == "mean":
            per_sample = per_frame_logit.mean(dim=1)
        elif self.frame_pool == "max":
            per_sample = per_frame_logit.gather(
                1,
                per_frame_logit.abs().argmax(dim=1, keepdim=True),
            ).squeeze(1)
        else:  # topk_mean
            k = min(self.frame_pool_topk, F_)
            _, idx = per_frame_logit.abs().topk(k, dim=1)
            per_sample = per_frame_logit.gather(1, idx).mean(dim=1)
        return per_sample.float()
