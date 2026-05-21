"""SWT-based wavelet HF extractor for LADD-wavelet (v28B).

Implements WGSR's "disc sees only HF" principle in latent space.
Single-level Haar Stationary Wavelet Transform (SWT) preserves spatial
resolution: each of the 16 latent channels produces 4 same-size sub-
bands (LL, LH, HL, HH). We drop LL by default and feed [LH, HL, HH]
(= 48 channels) to a small learned 1x1 conv adapter that re-maps back
to 16 channels at full resolution so the downstream WAN-teacher
projector sees an in-distribution shape.

Why SWT (not DWT): DWT decimates by 2 so sub-bands are H/2 x W/2.
That's OOD for the projector and halves the token count visible to
the disc heads. SWT keeps H x W.

Why Haar: cheapest (2x2 kernels), well-localised, and matches the
default in WGSR's config. Higher-order wavelets (db2 etc.) are
overkill for the gen's HF-detail signal at this resolution.

Architectural role:
    L1-on-LL  in WGSR  ~  DMD                    (content anchor)
    GAN-on-HF in WGSR  ~  LADD disc on HF latent (detail)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Haar SWT 2D kernels (2x2). Coefficients normalised so LL is the
# spatial mean over the 2x2 window. The remaining three bands have
# zero mean by construction.
_HAAR_LL = torch.tensor(
    [[0.25, 0.25],
     [0.25, 0.25]], dtype=torch.float32,
)
_HAAR_LH = torch.tensor(
    [[0.25, -0.25],
     [0.25, -0.25]], dtype=torch.float32,
)  # high-frequency along width
_HAAR_HL = torch.tensor(
    [[0.25, 0.25],
     [-0.25, -0.25]], dtype=torch.float32,
)  # high-frequency along height
_HAAR_HH = torch.tensor(
    [[0.25, -0.25],
     [-0.25, 0.25]], dtype=torch.float32,
)  # diagonal high-frequency


class LatentWaveletHF(nn.Module):
    """SWT-based HF extractor with a learned channel adapter.

    Args:
        in_channels: number of input latent channels (16 for Wan VAE).
        drop_ll: when True (default), the LL sub-band is dropped so
            the disc only sees the 3 HF sub-bands (LH, HL, HH) per
            channel. Set False to keep all 4 (mostly diagnostic).
        adapter_init_gain: Xavier-uniform gain for the adapter's
            weight init. Small (default 0.1) so the projector sees
            roughly-zero HF for the first few steps; the adapter
            learns to amplify HF channels over training.

    Input/output shape: ``[B, F, C, H, W]`` -> ``[B, F, C, H, W]``.
    Spatial resolution is preserved.
    """

    def __init__(
        self,
        in_channels: int = 16,
        drop_ll: bool = True,
        adapter_init_gain: float = 0.1,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.drop_ll = bool(drop_ll)
        out_bands = 3 if self.drop_ll else 4

        # Stack sub-band kernels; drop LL if requested.
        if self.drop_ll:
            kernels = torch.stack([_HAAR_LH, _HAAR_HL, _HAAR_HH], dim=0)
        else:
            kernels = torch.stack(
                [_HAAR_LL, _HAAR_LH, _HAAR_HL, _HAAR_HH], dim=0,
            )
        # kernels: [out_per_group, 2, 2] -> [out_per_group, 1, 2, 2]
        kernels = kernels.unsqueeze(1)
        # Tile per channel via grouped conv: shape
        # [in_channels * out_per_group, 1, 2, 2], groups=in_channels.
        weight = kernels.unsqueeze(0).expand(
            self.in_channels, -1, -1, -1, -1,
        ).reshape(
            self.in_channels * out_bands, 1, 2, 2,
        ).contiguous()
        self.register_buffer("_haar_weight", weight)
        self._out_bands = int(out_bands)

        # 1x1 conv adapter: HF-only channels -> in_channels.
        self.adapter = nn.Conv2d(
            self.in_channels * out_bands,
            self.in_channels,
            kernel_size=1,
            bias=True,
        )
        nn.init.xavier_uniform_(
            self.adapter.weight, gain=float(adapter_init_gain),
        )
        nn.init.zeros_(self.adapter.bias)

    @property
    def out_channels(self) -> int:
        return self.in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(
                f"LatentWaveletHF expects [B, F, C, H, W]; got "
                f"{tuple(x.shape)}"
            )
        B, F_, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(
                f"LatentWaveletHF: in_channels={self.in_channels} but input "
                f"has {C} channels."
            )
        x_flat = x.reshape(B * F_, C, H, W)
        # Reflect-pad 1 pixel on right + bottom so the 2x2 Haar conv
        # with stride=1 returns same-size H, W. Reflection avoids
        # boundary artifacts at the latent edges.
        x_pad = F.pad(x_flat, (0, 1, 0, 1), mode="reflect")
        hf = F.conv2d(
            x_pad,
            self._haar_weight.to(dtype=x_pad.dtype),
            bias=None,
            stride=1,
            padding=0,
            groups=C,
        )
        # hf: [B*F, C * out_bands, H, W]
        # Adapter re-maps back to the projector's expected channel
        # count.
        adapter_in_dtype = self.adapter.weight.dtype
        out_flat = self.adapter(hf.to(dtype=adapter_in_dtype))
        return out_flat.to(dtype=x.dtype).reshape(B, F_, C, H, W)
