"""Pretrained-backbone pixel discriminators for the WP-SURROGATE teacher.

Researcher directive (2026-08-24): the surrogate branch's teacher must be
a PRETRAINED pixel discriminator, never a from-scratch one. `sam2` was the
first such teacher (``model/r3gan_sam2.py``, restored from ``835b1df``);
this module adds the alternatives so the choice can be MEASURED rather
than argued:

    surrogate_teacher_backbone: sam2 | dinov2 | convnext

Why these three (the axes are deliberately spanned, not arbitrary)
------------------------------------------------------------------
| backbone      | family            | pretraining          | scales | known bias        |
|---------------|-------------------|----------------------|--------|-------------------|
| SAM2 Hiera-B+ | hierarchical ViT  | segmentation (SA-1B) | multi  | region / boundary |
| DINOv2 ViT-S/14 | plain ViT       | self-supervised      | single | semantic / shape  |
| ConvNeXt-T    | conv              | supervised ImageNet  | multi  | TEXTURE           |

* **DINOv2** is the doc-sanctioned option: ``TEXTURE_GAN_DESIGN.md`` §4.1
  names it as the B5 escalation and records that **ADD used it for this
  exact job**. Layer taps mirror ADD's ViT hooks (evenly spaced blocks).
* **ConvNeXt** is included because ImageNet-supervised CNNs are
  *texture-biased* (Geirhos et al., ICLR 2019 — they classify by texture
  where ViTs/humans use shape). For a critic whose entire job is texture
  that bias is a feature, and it is the sharpest available contrast to
  DINOv2's self-supervised ViT: conv-vs-transformer AND
  supervised-vs-self-supervised in one swap.

The controlled-comparison property (the point of this file)
-----------------------------------------------------------
Every backbone here reuses ``r3gan_sam2._R3GANDiscHeads`` **verbatim** —
the same ADM 2D heads (arXiv:2507.18569 §B.1), the same per-scale mean,
the same frame_pool, the same zero-init final linear. So an arm-to-arm
difference is attributable to the FEATURE BASIS and nothing else. Writing
a second head stack per backbone would have quietly confounded exactly
the comparison this exists to make.

Offline by construction
-----------------------
Both backbones' weights are pre-cached under the repo's ``HF_HOME`` /
``torch.hub`` cache; compute nodes have no reliable network, so anything
requiring a live download would fail at step 0 on the holder rather than
here. ``dinov2`` loads from the cached ``facebookresearch_dinov2_main``
hub repo; ``convnext`` from the cached timm weights.

API contract — identical to ``R3GANDiscriminatorSAM2Pixel`` so the
trainer's teacher closure is backbone-agnostic:

    forward(pixel_video [B, F, 3, H, W] in [-1, 1]) -> [B] per-clip logits
    .heads_module  -> the trainable submodule (optimizer / DDP wrap this)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# REUSE, never reimplement -- see "controlled-comparison property" above.
from model.r3gan_sam2 import _R3GANDiscHeads

__all__ = ["PretrainedPixelDisc", "SUPPORTED_BACKBONES"]

SUPPORTED_BACKBONES = ("dinov2", "convnext")

# ImageNet statistics -- correct for BOTH backbones (DINOv2 and timm's
# ConvNeXt both normalise this way).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Frozen backbone builders. Each returns (module, forward_fn, patch_multiple)
# where forward_fn(x) -> List[[N, C, h, w]] feature maps.
# ---------------------------------------------------------------------------
def _build_dinov2(
    variant: str,
    layers: Optional[Sequence[int]],
    device: torch.device,
    dtype: torch.dtype,
):
    """Frozen DINOv2 ViT with ADD-style evenly-spaced layer taps.

    ADD (arXiv:2311.17042) hooks evenly-spaced ViT blocks rather than
    only the last -- shallow blocks carry the local/textural signal a
    texture critic needs, which the final semantic block has largely
    abstracted away. ``get_intermediate_layers(..., reshape=True)``
    returns each tap already as ``[N, C, h, w]``, so no manual token
    un-flattening (and no chance of getting it subtly wrong).
    """
    model = torch.hub.load(
        "facebookresearch/dinov2", variant,
        pretrained=True, source="github", trust_repo=True, verbose=False,
    )
    n_blocks = len(model.blocks)
    if layers is None:
        # Evenly spaced, last block always included -- the ADD analog for
        # whatever depth this variant has (12 for vits14/vitb14, 24 vitl14).
        layers = [
            int(round((i + 1) * n_blocks / 4.0)) - 1 for i in range(4)
        ]
    layers = [int(l) % n_blocks for l in layers]
    model = model.to(device=device, dtype=dtype).eval()
    model.requires_grad_(False)

    def _forward(x: torch.Tensor) -> List[torch.Tensor]:
        outs = model.get_intermediate_layers(
            x, n=layers, reshape=True, norm=True,
        )
        return list(outs)

    return model, _forward, 14, layers


def _build_convnext(
    variant: str,
    layers: Optional[Sequence[int]],
    device: torch.device,
    dtype: torch.dtype,
):
    """Frozen ConvNeXt via timm ``features_only`` -- a true hierarchical
    FPN (strides 4/8/16/32), structurally the closest analog to SAM2's
    Hiera and therefore the fairest multi-scale comparison against it."""
    import os

    import timm

    # OFFLINE-FIRST. A local state_dict is preferred over hub resolution
    # because hub resolution is NOT reliably offline-safe here: with
    # HF_HUB_OFFLINE=1 timm asks for ``pytorch_model.bin`` and cannot
    # discover that the cache actually holds ``model.safetensors``
    # (finding the safetensors alternative needs a HEAD request, which
    # offline mode blocks). That failed on a compute node at step 0 --
    # exactly the "needs a live download -> dies on a holder" risk this
    # module's docstring warns about, which then came true. Export once
    # with ``timm.create_model(..., pretrained=True, features_only=True)``
    # + ``torch.save(m.state_dict(), ...)``; the constructor then never
    # touches the network at all.
    _local = os.environ.get(
        "SURROGATE_CONVNEXT_WEIGHTS",
        "pretrained_backbones/convnext_tiny_fb_in1k_featonly.pt",
    )
    if os.path.isfile(_local):
        model = timm.create_model(
            variant, pretrained=False, features_only=True)
        _sd = torch.load(_local, map_location="cpu")
        _missing, _unexpected = model.load_state_dict(_sd, strict=True)
        logging.info(
            "[pretrained_pixel_disc] convnext weights loaded from %s "
            "(offline, no hub resolution)", _local,
        )
    else:
        # Fallback: hub. Kept so a fresh checkout still works ON A NODE
        # WITH NETWORK, but it is the degraded path -- the log line says
        # which one ran so a silent fallback cannot be mistaken for the
        # offline-safe route.
        logging.warning(
            "[pretrained_pixel_disc] convnext local weights not at %s -- "
            "falling back to hub resolution, which is NOT offline-safe "
            "on compute nodes.", _local,
        )
        model = timm.create_model(variant, pretrained=True, features_only=True)
    model = model.to(device=device, dtype=dtype).eval()
    model.requires_grad_(False)
    if layers is not None:
        keep = [int(l) for l in layers]
    else:
        keep = None

    def _forward(x: torch.Tensor) -> List[torch.Tensor]:
        feats = model(x)
        return [feats[i] for i in keep] if keep else list(feats)

    return model, _forward, 32, (keep or list(range(len(model.feature_info.channels()))))


_BUILDERS = {"dinov2": _build_dinov2, "convnext": _build_convnext}
_DEFAULT_VARIANT = {"dinov2": "dinov2_vits14", "convnext": "convnext_tiny.fb_in1k"}

# Minimum spatial extent of a TAPPED feature map, forced by the shared ADM
# head: it applies three Conv2d(k=4, s=2, p=1) in series, each mapping
# ``n -> floor((n-2)/2)+1``. So 8 -> 4 -> 2 -> 1 survives, but 7 -> 3 -> 1
# -> **crash** (a 1x1 input padded to 3x3 is smaller than the 4x4 kernel).
# Found the hard way: ConvNeXt's stride-32 stage at 224 px is exactly 7x7,
# which died inside the third conv with a message naming neither the head
# nor the backbone. Guarded at BUILD time below instead -- a resolution
# mistake should fail in the constructor with the fix in the message, not
# at step 0 on a holder six layers down a stack trace.
_MIN_FEAT_SPATIAL = 8

# Per-backbone default input resolution, chosen so every tapped map clears
# ``_MIN_FEAT_SPATIAL`` with room to spare (and comparable to SAM2's 512):
#   dinov2   518 = 37 x patch-14  -> 37x37 at every tap (plain ViT)
#   convnext 512                  -> 128/64/32/16 (strides 4/8/16/32)
_DEFAULT_RES = {"dinov2": 518, "convnext": 512}


class PretrainedPixelDisc(nn.Module):
    """Frozen pretrained pixel backbone + trainable ADM 2D heads.

    Drop-in for ``R3GANDiscriminatorSAM2Pixel``: same input contract,
    same ``[B]`` output, same ``heads_module`` handle. See the module
    docstring for why the heads are shared rather than per-backbone.
    """

    def __init__(
        self,
        backbone: str,
        *,
        variant: Optional[str] = None,
        layers: Optional[Sequence[int]] = None,
        image_resolution: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        pad_to_square: bool = True,
        frame_pool: str = "mean",
        frame_pool_topk: int = 4,
    ) -> None:
        super().__init__()
        backbone = str(backbone).lower()
        if backbone not in _BUILDERS:
            raise ValueError(
                f"backbone must be one of {SUPPORTED_BACKBONES}; "
                f"got {backbone!r}."
            )
        if device is None:
            device = (
                torch.device("cuda") if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.backbone_name = backbone
        self.variant = str(variant or _DEFAULT_VARIANT[backbone])
        if image_resolution is None:
            image_resolution = _DEFAULT_RES[backbone]
        self.pad_to_square = bool(pad_to_square)
        self._dtype = dtype

        enc, fwd, multiple, taps = _BUILDERS[backbone](
            self.variant, layers, device, dtype,
        )
        self.encoder = enc
        self._encoder_forward = fwd
        self._patch_multiple = int(multiple)
        self.taps = list(taps)
        # Round the requested resolution to what the backbone requires
        # (DINOv2 patch-14 REJECTS non-multiples; rounding here rather
        # than at the call site means no caller can get it wrong).
        self.image_resolution = max(
            self._patch_multiple,
            int(round(image_resolution / self._patch_multiple))
            * self._patch_multiple,
        )

        feat_channels, feat_spatial = self._probe(device, dtype)
        _too_small = [
            (i, hw) for i, hw in enumerate(feat_spatial)
            if min(hw) < _MIN_FEAT_SPATIAL
        ]
        if _too_small:
            raise ValueError(
                f"{backbone} ({self.variant}) at image_resolution="
                f"{self.image_resolution} produces tapped feature map(s) "
                f"smaller than the shared ADM head's minimum "
                f"{_MIN_FEAT_SPATIAL}x{_MIN_FEAT_SPATIAL}: "
                + ", ".join(f"tap {i} -> {hw[0]}x{hw[1]}" for i, hw in _too_small)
                + ". The head applies three stride-2 4x4 convs, so anything "
                f"below {_MIN_FEAT_SPATIAL} collapses to <1 and the third "
                f"conv raises deep inside the head. Fix: raise "
                f"image_resolution (>= "
                f"{_MIN_FEAT_SPATIAL * self._stride_of_deepest_tap(feat_spatial)}"
                f" px for this tap set), or drop the deepest tap via "
                f"`layers`."
            )
        self._feat_channels = feat_channels
        self.heads_module = _R3GANDiscHeads(
            feat_channels=feat_channels,
            frame_pool=frame_pool,
            frame_pool_topk=frame_pool_topk,
        ).to(device=device, dtype=torch.float32)

        self.register_buffer(
            "_mean",
            torch.tensor(_IMAGENET_MEAN, dtype=dtype,
                         device=device).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_std",
            torch.tensor(_IMAGENET_STD, dtype=dtype,
                         device=device).view(1, 3, 1, 1),
            persistent=False,
        )
        logging.info(
            "[PretrainedPixelDisc] %s (%s): res=%d taps=%s channels=%s "
            "spatial=%s heads=%.2fM (encoder frozen)",
            backbone, self.variant, self.image_resolution, self.taps,
            feat_channels, feat_spatial,
            sum(p.numel() for p in self.heads_module.parameters()) / 1e6,
        )

    def _stride_of_deepest_tap(self, feat_spatial) -> int:
        """Effective stride of the smallest tapped map -- used only to put
        a concrete "raise resolution to at least N px" number in the guard
        message above, rather than making the reader work it out."""
        smallest = min(min(hw) for hw in feat_spatial)
        return max(1, int(round(self.image_resolution / max(1, smallest))))

    # -- introspection -------------------------------------------------
    @torch.no_grad()
    def _probe(self, device, dtype) -> Tuple[List[int], List[Tuple[int, int]]]:
        r = self.image_resolution
        dummy = torch.zeros(1, 3, r, r, device=device, dtype=dtype)
        feats = self._encoder_forward(dummy)
        if not feats:
            raise RuntimeError(
                f"{self.backbone_name} produced no feature maps at "
                f"resolution {r}."
            )
        return (
            [int(f.shape[1]) for f in feats],
            [(int(f.shape[2]), int(f.shape[3])) for f in feats],
        )

    @property
    def feat_channels(self) -> List[int]:
        return list(self._feat_channels)

    def train(self, mode: bool = True):
        """Heads follow ``mode``; the frozen encoder is pinned to eval
        FOREVER. Same override (and same reason) as the SAM2 disc: a
        stray ``.train()`` would otherwise flip BN/stochastic-depth in a
        backbone whose statistics must stay fixed."""
        super().train(mode)
        self.encoder.eval()
        return self

    # -- forward -------------------------------------------------------
    def _prep(self, px: torch.Tensor) -> torch.Tensor:
        """``[N, 3, H, W]`` in [-1, 1] -> normalized, backbone-shaped.

        NOT wrapped in ``no_grad``: the teacher path differentiates
        through here back to the latent crop. The encoder's PARAMS are
        frozen (no weight grads allocated); the INPUT path is live.
        """
        if self.pad_to_square:
            h, w = px.shape[-2], px.shape[-1]
            side = max(h, w)
            if h != side or w != side:
                px = F.pad(
                    px,
                    (0, side - w, 0, side - h),
                    mode="constant", value=0.0,
                )
        r = self.image_resolution
        px = F.interpolate(
            px, size=(r, r), mode="bilinear", align_corners=False,
        )
        px = (px.clamp(-1.0, 1.0) + 1.0) * 0.5      # [-1,1] -> [0,1]
        return (px.to(self._dtype) - self._mean) / self._std

    def forward_features(self, pixel_video: torch.Tensor) -> List[torch.Tensor]:
        if pixel_video.dim() != 5:
            raise ValueError(
                f"expected [B, F, 3, H, W]; got {tuple(pixel_video.shape)}."
            )
        b, f = int(pixel_video.shape[0]), int(pixel_video.shape[1])
        x = self._prep(pixel_video.reshape(b * f, *pixel_video.shape[2:]))
        return self._encoder_forward(x)

    def forward(self, pixel_video: torch.Tensor) -> torch.Tensor:
        b, f = int(pixel_video.shape[0]), int(pixel_video.shape[1])
        feats = self.forward_features(pixel_video)
        feats = [t.to(torch.float32) for t in feats]
        return self.heads_module(feats, batch_size=b, num_frames=f)


if __name__ == "__main__":  # pragma: no cover - manual selftest
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="dinov2",
                    choices=list(SUPPORTED_BACKBONES))
    ap.add_argument("--res", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    dev = torch.device(a.device)
    d = PretrainedPixelDisc(a.backbone, image_resolution=a.res, device=dev)
    d.train()
    x = (torch.rand(2, 3, 3, 176, 240, device=dev) * 2 - 1)
    print("forward:", d(x).shape, d(x).detach().cpu().tolist())
    xg = x.clone().requires_grad_(True)
    g = torch.autograd.grad(d(xg).sum(), xg)[0]
    print(f"input-grad norm={g.norm().item():.4g}")
    assert not any(p.grad is not None for p in d.encoder.parameters()), \
        "encoder must receive no weight gradients"
    print("SELFTEST PASS")
