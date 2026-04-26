"""R3GAN — RpGAN + R1 + R2 discriminator and losses for latent video.

This module wires the **R3GAN** ("Re-GAN") training recipe from
*Huang et al., "The GAN is dead; long live the GAN! A Modern Baseline
GAN", NeurIPS 2024* (https://arxiv.org/abs/2501.05441) to the
Phase-1 Action-Forcing pipeline.

Why R3GAN here
--------------
The DMD2 paper combines its score-distillation objective with a
GAN term. The original DMD2 implementation reuses the ``fake_score``
network with a tacked-on classifier branch — efficient, but fragile:
a single weight share couples generator's and discriminator's
fitness, and the ``adding_cls_branch`` is brittle to checkpoint /
RoPE-cache changes.

R3GAN is a clean alternative: a *separate* discriminator with a
**provably locally convergent** loss (RpGAN + zero-centered R1 + R2
gradient penalties; Section 2 of the paper). The loss is
architecture-agnostic — convergence guarantees come from the loss
formulation, not the backbone — so we keep a small modernized 3D
ConvNet that operates directly on the student's latent video output.

The GAN's job is to push the generator's video distribution towards
the GT distribution (StyleGAN-style adversarial signal), as a
*complement* to:
  - DMD2's score-matching loss (already in place)
  - The action critic's z-guidance loss (aux variant)

so that we get distributional matching to GT not just per-frame
score-distillation.

Real samples
------------
Real = ``ride["latents"]`` slices from the same window the generator
predicts (frames ``[gen_window_start:gen_window_end]``). Fake =
``pred_image`` from ``model.generator_loss(..., return_aux=True)``.

Loss formulation (eq. 2 + 3 of the R3GAN paper)
-----------------------------------------------
Let ``f(x) = softplus(x) = log(1 + e^x)``. With ``D := D_ψ`` and
``G := G_θ`` and pairs ``(x_real, x_fake)``:

D loss (minimize over ψ):
    L_D = E[ f( D(x_fake) - D(x_real) ) ]
        + (γ/2) * E_{x_real}[ ||∇_{x_real} D(x_real)||² ]   <-- R1
        + (γ/2) * E_{x_fake}[ ||∇_{x_fake} D(x_fake)||² ]   <-- R2

G loss (minimize over θ):
    L_G = E[ f( D(x_real) - D(x_fake) ) ]

Notes
-----
* R1 + R2 are computed with ``torch.autograd.grad(... create_graph=
  True)`` so the second-order term flows into D's update. The fake
  gradient penalty (R2) is computed on the **detached** fake
  (``pred_image.detach()``) so no gradient flows back to G during
  the D-update.
* The G-loss path uses the **same** discriminator with frozen
  parameters (``requires_grad_(False)``). DDP doesn't fire on D
  during the G-update; D's gradients are all-reduced in its own
  D-update backward pass.
* R3GAN's recommended γ depends on dataset; for our 16-channel
  latent video we default to ``gan_r1_gamma=1.0, gan_r2_gamma=1.0``
  (the paper uses γ in [0.01, 100]; 1.0 is the median of their
  ablations). Tune via YAML.

The discriminator follows the modern-GAN guidance from R3GAN
(Section 3): GroupNorm + GELU + 1x1 inverted-residual blocks (a la
ConvNeXt), no spectral norm / no equalized LR / no minibatch std,
because R3GAN explicitly drops these "patches for weak backbones".
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Discriminator backbone.
# ---------------------------------------------------------------------------


class _R3GANBlock3D(nn.Module):
    """ConvNeXt-style 3D residual block with optional stride-2 downsample.

    Forward path (input ``x`` of shape ``[B, C_in, T, H, W]``):
        1. depthwise conv3d 3x3x3 (stride=stride)
        2. GroupNorm + GELU
        3. pointwise conv3d 1x1x1 expand to ``4 * C_out``
        4. GELU
        5. pointwise conv3d 1x1x1 squeeze to ``C_out``
        6. residual: ``+ skip(x)`` where ``skip`` is identity if
           ``C_in == C_out`` and ``stride == 1``, else a stride-`stride`
           1x1x1 conv that projects channels.

    No batch-stats layers — GroupNorm only — so the block is
    deterministic across batch sizes (R3GAN paper, Sec. 3.2).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Tuple[int, int, int] = (2, 2, 2),
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.dwconv = nn.Conv3d(
            in_channels, in_channels,
            kernel_size=3, stride=stride, padding=1, groups=in_channels,
        )
        self.norm = nn.GroupNorm(num_groups=min(groups, in_channels), num_channels=in_channels)
        self.pw_expand = nn.Conv3d(in_channels, 4 * out_channels, kernel_size=1)
        self.pw_squeeze = nn.Conv3d(4 * out_channels, out_channels, kernel_size=1)
        if stride == (1, 1, 1) and in_channels == out_channels:
            self.skip: nn.Module = nn.Identity()
        else:
            self.skip = nn.Conv3d(
                in_channels, out_channels, kernel_size=1, stride=stride,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dwconv(x)
        h = self.norm(h)
        h = F.gelu(h)
        h = self.pw_expand(h)
        h = F.gelu(h)
        h = self.pw_squeeze(h)
        return h + self.skip(x)


class R3GANDiscriminator3D(nn.Module):
    """Lightweight 3D ConvNeXt-style discriminator for latent video.

    Input shape: ``[B, F, C, H, W]`` (the trainer's per-iter latent
    layout). The forward pass permutes to ``[B, C, F, H, W]`` for
    3D convolutions. Output: a ``[B]`` tensor of scalar logits.

    Default sizing for the 16-channel Wan latent video at 21 frames:
        Block 0: stride (1, 2, 2),  C: 16  -> base
        Block 1: stride (2, 2, 2),  C: base -> 2*base
        Block 2: stride (2, 2, 2),  C: 2*base -> 4*base
        Block 3: stride (1, 2, 2),  C: 4*base -> 8*base
        AdaptiveAvgPool3d to (1, 1, 1)
        Linear(8*base -> 1)

    For ``base=64`` total params ≈ 4.6M (well under the action_critic's
    footprint and ~0.3% of the 14B generator).
    """

    def __init__(
        self,
        in_channels: int = 16,
        base_channels: int = 64,
        num_blocks: int = 4,
        block_strides: Optional[list] = None,
        groups: int = 8,
    ) -> None:
        super().__init__()
        if block_strides is None:
            block_strides = [
                (1, 2, 2),
                (2, 2, 2),
                (2, 2, 2),
                (1, 2, 2),
            ][:num_blocks]
        if len(block_strides) != num_blocks:
            raise ValueError(
                f"block_strides has {len(block_strides)} entries but "
                f"num_blocks={num_blocks}"
            )

        self.stem = nn.Conv3d(
            in_channels, base_channels, kernel_size=3, padding=1,
        )

        blocks = []
        ch_in = base_channels
        for i, stride in enumerate(block_strides):
            ch_out = base_channels * (2 ** i) if i < num_blocks - 1 else base_channels * (2 ** (num_blocks - 1))
            blocks.append(_R3GANBlock3D(ch_in, ch_out, stride=tuple(stride), groups=groups))
            ch_in = ch_out
        self.blocks = nn.ModuleList(blocks)

        self.final_norm = nn.GroupNorm(
            num_groups=min(groups, ch_in), num_channels=ch_in,
        )
        self.head = nn.Linear(ch_in, 1)

        self._final_channels = ch_in

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    @property
    def final_channels(self) -> int:
        return self._final_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(
                f"R3GANDiscriminator3D expected 5D input [B,F,C,H,W], got {tuple(x.shape)}"
            )
        h = x.permute(0, 2, 1, 3, 4).contiguous()
        h = self.stem(h)
        for blk in self.blocks:
            h = blk(h)
        h = self.final_norm(h)
        h = F.gelu(h)
        h = h.mean(dim=(2, 3, 4))
        logits = self.head(h)
        return logits.squeeze(-1)


# ---------------------------------------------------------------------------
# Losses.
# ---------------------------------------------------------------------------


def rpgan_d_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """RpGAN discriminator loss: ``E[softplus(D(fake) - D(real))]``.

    Equivalent to ``-E[log σ(D(real) - D(fake))]``, the relativistic
    non-saturating GAN's logistic loss. D minimizes this.
    """
    return F.softplus(d_fake - d_real).mean()


def rpgan_g_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """RpGAN generator loss: ``E[softplus(D(real) - D(fake))]``.

    Symmetric counterpart to ``rpgan_d_loss``. G minimizes this; the
    ``d_real`` argument is detached at the call site so no gradient
    flows into the real samples (only through ``d_fake`` on
    ``pred_image``).
    """
    return F.softplus(d_real - d_fake).mean()


def r1_penalty(
    discriminator: nn.Module,
    real_input: torch.Tensor,
    *,
    gamma: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Zero-centered gradient penalty on **real** samples (R1).

    R1(ψ) = (γ/2) * E_{x~pD}[ ||∇_x D_ψ(x)||² ]

    Returns ``(penalty, d_real)`` where ``d_real`` is the discriminator
    output on ``real_input`` (with the ``create_graph`` graph still
    attached). Caller can reuse ``d_real`` for the RpGAN-D term to
    avoid a redundant forward.

    Important: ``real_input`` MUST have ``requires_grad_(True)`` set
    by the caller, AND must be the input you actually care about
    (typically ``ride["latents"][..., :].detach().requires_grad_()``
    so the autograd graph for the second-order penalty is rooted at
    the input, not at the dataset's stored latents). Returning
    ``d_real`` from this function avoids a double forward.
    """
    if not real_input.requires_grad:
        raise RuntimeError(
            "r1_penalty: real_input must have requires_grad=True so "
            "the gradient ∇_x D(x) is well-defined."
        )
    d_real = discriminator(real_input)
    grads = torch.autograd.grad(
        outputs=d_real.sum(),
        inputs=real_input,
        create_graph=True,
        retain_graph=True,
    )[0]
    penalty = 0.5 * gamma * grads.flatten(1).pow(2).sum(dim=1).mean()
    return penalty, d_real


def r2_penalty(
    discriminator: nn.Module,
    fake_input: torch.Tensor,
    *,
    gamma: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Zero-centered gradient penalty on **fake** samples (R2).

    R2(θ, ψ) = (γ/2) * E_{x~pθ}[ ||∇_x D_ψ(x)||² ]

    Same contract as ``r1_penalty``: ``fake_input`` must have
    ``requires_grad=True``. The fake samples should be **detached**
    from the generator before being re-flagged for grad, so this
    penalty does NOT push gradients back into the generator (we
    only want it to constrain D's gradient norm on the fake
    distribution).
    """
    if not fake_input.requires_grad:
        raise RuntimeError(
            "r2_penalty: fake_input must have requires_grad=True so "
            "the gradient ∇_x D(x) is well-defined."
        )
    d_fake = discriminator(fake_input)
    grads = torch.autograd.grad(
        outputs=d_fake.sum(),
        inputs=fake_input,
        create_graph=True,
        retain_graph=True,
    )[0]
    penalty = 0.5 * gamma * grads.flatten(1).pow(2).sum(dim=1).mean()
    return penalty, d_fake
