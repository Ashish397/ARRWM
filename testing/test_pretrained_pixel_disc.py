"""Tests for model/pretrained_pixel_disc.py (DINOv2 / ConvNeXt teachers).

CPU-only and OFFLINE — weights come from the repo's pre-populated caches,
which is also the property that matters in production (compute nodes have
no reliable network, so a backbone needing a live download would fail at
step 0 on a holder rather than here).

What these protect, in order of how much they would cost to miss:

* **The controlled-comparison property.** Every backbone must attach the
  SAME ADM heads. If someone adds a per-backbone head stack, an arm-to-arm
  delta stops being attributable to the feature basis and the whole
  three-way comparison silently becomes uninterpretable. Pinned
  structurally AND by identity of the head class.
* **The head's 8x8 feature minimum.** Found the hard way: ConvNeXt's
  stride-32 stage at 224 px is exactly 7x7, which died inside the third
  stride-2 conv with a message naming neither the head nor the backbone.
  The guard must fire at CONSTRUCTION with an actionable message.
* **Encoder frozen, input path live.** The teacher's whole job is to
  hand a gradient back to the latent crop; the encoder's weights must
  receive none. Both halves are asserted — a test that only checked
  "frozen" would pass on a backbone that also blocks the input gradient,
  which would make the surrogate distil from a zero field.

Run:
    OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 PYTHONPATH=. \
      pytest -q testing/test_pretrained_pixel_disc.py
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.pretrained_pixel_disc import (  # noqa: E402
    _MIN_FEAT_SPATIAL, SUPPORTED_BACKBONES, PretrainedPixelDisc,
)
from model.r3gan_sam2 import _R3GANDiscHeads  # noqa: E402

CPU = torch.device("cpu")
# Small but legal: every tapped map must still clear _MIN_FEAT_SPATIAL.
_RES = {"dinov2": 224, "convnext": 256}


def _mk(bk):
    return PretrainedPixelDisc(bk, image_resolution=_RES[bk], device=CPU)


@pytest.mark.parametrize("bk", SUPPORTED_BACKBONES)
def test_builds_and_reports_taps(bk):
    d = _mk(bk)
    assert len(d.feat_channels) >= 1
    assert len(d.taps) == len(d.feat_channels)


@pytest.mark.parametrize("bk", SUPPORTED_BACKBONES)
def test_forward_contract_matches_sam2_disc(bk):
    """[B, F, 3, H, W] in [-1,1] -> [B]. Same contract as
    R3GANDiscriminatorSAM2Pixel, so the trainer's teacher closure is
    backbone-agnostic and needs no per-backbone special-casing."""
    d = _mk(bk)
    out = d(torch.rand(2, 3, 3, 176, 240) * 2 - 1)
    assert out.shape == (2,)
    assert out.dtype is torch.float32


@pytest.mark.parametrize("bk", SUPPORTED_BACKBONES)
def test_shares_the_adm_heads_verbatim(bk):
    """THE controlled-comparison invariant. Not 'a head with the same
    shape' -- the same CLASS, so heads cannot drift per backbone."""
    d = _mk(bk)
    assert isinstance(d.heads_module, _R3GANDiscHeads)
    assert len(d.heads_module.heads) == len(d.feat_channels)


@pytest.mark.parametrize("bk", SUPPORTED_BACKBONES)
def test_zero_init_head_gives_zero_logits(bk):
    """Matches the SAM2/latent-disc convention: D starts as a constant-zero
    classifier. This is ALSO why the Sobolev term needs its degenerate-
    target guard -- at init the teacher gradient is exactly zero."""
    d = _mk(bk)
    assert torch.count_nonzero(d(torch.rand(2, 2, 3, 176, 240) * 2 - 1)) == 0


@pytest.mark.parametrize("bk", SUPPORTED_BACKBONES)
def test_encoder_frozen_but_input_gradient_flows(bk):
    """Both halves matter and they are different claims."""
    d = _mk(bk)
    torch.nn.init.normal_(d.heads_module.heads[0].head.weight, std=0.05)
    x = (torch.rand(1, 2, 3, 176, 240) * 2 - 1).requires_grad_(True)
    g = torch.autograd.grad(d(x).sum(), x)[0]
    assert torch.count_nonzero(g) > 0, "no gradient reached the input"
    assert all(not p.requires_grad for p in d.encoder.parameters())
    assert all(p.grad is None for p in d.encoder.parameters())


@pytest.mark.parametrize("bk", SUPPORTED_BACKBONES)
def test_train_mode_never_unfreezes_the_encoder(bk):
    d = _mk(bk)
    d.train()
    assert d.heads_module.training is True
    assert d.encoder.training is False, "encoder must stay pinned to eval"


def test_head_minimum_guard_fires_with_an_actionable_message():
    """ConvNeXt at 224 -> stride-32 stage is 7x7, one short of the head's
    minimum. Must raise at CONSTRUCTION naming the tap, the minimum and
    the fix -- not die inside a conv three layers down."""
    with pytest.raises(ValueError) as e:
        PretrainedPixelDisc("convnext", image_resolution=224, device=CPU)
    msg = str(e.value)
    assert "tap 3 -> 7x7" in msg
    assert str(_MIN_FEAT_SPATIAL) in msg
    assert "image_resolution" in msg


def test_head_minimum_guard_has_teeth():
    """Mutation control: the same backbone at a legal resolution must
    build, so the guard above is not simply rejecting convnext."""
    assert PretrainedPixelDisc(
        "convnext", image_resolution=256, device=CPU) is not None


def test_unknown_backbone_is_rejected_by_name():
    with pytest.raises(ValueError) as e:
        PretrainedPixelDisc("resnet50", device=CPU)
    assert "resnet50" in str(e.value)


def test_resolution_is_rounded_to_the_backbone_multiple():
    """DINOv2 is patch-14 and REJECTS non-multiples; rounding in the
    constructor means no caller can get this wrong."""
    d = PretrainedPixelDisc("dinov2", image_resolution=220, device=CPU)
    assert d.image_resolution % 14 == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
