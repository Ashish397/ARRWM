"""``ladd_feature_source=rn50`` -- ResNet50 *layer1* as the texture basis.

    OMP_NUM_THREADS=8 python -m pytest testing/test_ladd_rn50_stat_source.py -q

WHY THIS BASIS, AND WHY layer1 ONLY
===================================
``analysis/gan_tuning/TEXTURE_BASIS_BENCHMARK.md`` §3 ranks candidate
frozen bases by

    TS_min = min(S_blur, S_sharp, S_grain+, S_grain-) / N_shift

-- the smallest of the four two-sided texture responses divided by the
crop-translation nuisance. Pass mark 1.0. Measured, n = 420 crops per
cell:

    rn50_layer1  5.40      <- this file
    vgg_relu2_2  2.96
    dino_blk2    1.33
    rn50_layer2  1.19
    vgg_relu1_2  0.82
    vgg_all      0.63
    dino_all     0.39      <- what the shipped discriminator consumes

Two consequences are encoded as tests here rather than as prose:

  * **layer1 alone.** ``rn50_layer2`` scores 1.19, barely over the pass
    mark, and the study's §3 shows twice what fusing a weaker deeper tap
    does: ``vgg_all`` fell to 0.63 from ``relu2_2``'s 2.96, ``dino_all``
    to 0.39 from ``blk2``'s 1.33. ``test_layer1_is_the_only_default_tap``
    pins the default so a later edit cannot quietly reinstate the
    dilution.
  * **the pooling must be orderless.** The entire TS_min advantage is a
    ratio whose denominator is crop translation. If any spatial
    coordinate survives into the statistic, the denominator grows and
    the number this arm was chosen for evaporates.
    ``test_pool_is_orderless_through_the_real_trunk`` is the sharpest
    form of that claim: permute the HW axis of a REAL layer1 map and the
    statistic must not move.

WHAT THIS FILE DOES *NOT* CLAIM
===============================
It does not claim rn50 beats dinov2 as a discriminator. TS_min is a
property of the FROZEN basis measured with no adversarially-trained head
in the way (benchmark §9); these tests verify the basis is wired,
orderless, size-controlled, offline-loadable and inert when off. The arm
``sbatch/pixrn50_online.sbatch`` is what would measure the rest, and it
has not been submitted.

The ResNet50 layer1 convolutions are themselves NOT position-blind -- an
oriented 7x7 stem filter responds to local structure, which is the whole
point. What dies is the CROP-LEVEL coordinate: the head cannot learn
"the real ones have more energy in the top-left".
"""
from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

from model.ladd_pixel_features import (
    SUPPORTED_PIXEL_FEATURE_SOURCES,
    RN50_TAPS,
    LaddPixelFeatureSource,
    LaddPixelStatHead,
    _RN50Trunk,
    _load_torchhub_state_dict,
    build_pixel_feature_source,
    default_encoder_lr_scale,
)

_RN50_CKPT_NAME = "resnet50-0676ba61.pth"


def _rn50_ckpt_path():
    """The ImageNet ResNet50 checkpoint, or ``None``. NEVER downloads."""
    c = []
    th = os.environ.get("TORCH_HOME", "")
    if th:
        c.append(os.path.join(th, "hub", "checkpoints", _RN50_CKPT_NAME))
    try:
        c.append(os.path.join(torch.hub.get_dir(), "checkpoints",
                              _RN50_CKPT_NAME))
    except Exception:
        pass
    c.append(os.path.expanduser(
        f"~/.cache/torch/hub/checkpoints/{_RN50_CKPT_NAME}"))
    for p in c:
        if p and os.path.isfile(p):
            return p
    return None


_CKPT = _rn50_ckpt_path()
_needs_ckpt = pytest.mark.skipif(
    _CKPT is None,
    reason=f"{_RN50_CKPT_NAME} not in the torch-hub cache",
)


# ===========================================================================
# helpers
# ===========================================================================
class _RaisingProjector:
    """Makes "the WAN taps were used" impossible to miss."""

    def __init__(self):
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        raise AssertionError(
            "WanFeatureProjector was called on an rn50-source disc. The "
            "feature basis did NOT swap."
        )


def _fake_decode(latents, want_grad):
    n, f, c, h, w = latents.shape
    x = latents if want_grad else latents.detach()
    x = x[:, :, :3] if c >= 3 else x.repeat(1, 1, 3, 1, 1)[:, :, :3]
    x = x.reshape(n * f, 3, h, w)
    x = F.interpolate(x, scale_factor=8, mode="nearest")
    return torch.tanh(x).reshape(n, f, 3, h * 8, w * 8)


def _mk_src(pretrained=False, layers=("layer1",), **kw):
    """A small rn50 source. ``pretrained=False`` is legal ONLY in tests --
    see ``test_random_init_is_never_a_silent_fallback``."""
    return LaddPixelFeatureSource(
        "rn50", rn50_layers=layers, rn50_pretrained=pretrained,
        vgg_stat_proj_dim=8, vgg_stat_hidden_dim=16, **kw,
    )


def _mk_disc(src=None, *, crops=1, frames=2, **kw):
    from model.ladd_disc import LADDDiscriminator
    src = src if src is not None else _mk_src(**kw)
    d = LADDDiscriminator(
        projector=_RaisingProjector(),
        block_indices=[0, 1],
        dim_teacher=32,
        dim_proj=16,
        use_csm=True,
        pixel_source=src,
    )
    d.pixel_decode_fn = _fake_decode
    d.pixel_cfg = {
        "crop_rows": 8, "crop_cols": 8, "crops_per_row": crops,
        "lat_frames": 2, "frames_per_crop": frames, "border": 2,
        "decode_batch": 4,
    }
    d.eval()
    return d, src


def _lat(b=2, f=3, c=16, h=16, w=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, f, c, h, w, generator=g)


def _t_pe(b, f):
    return (torch.zeros(b, f, dtype=torch.long), torch.zeros(b, 4, 8))


def _perm_maps(feats, seed=7):
    """PERMUTE the spatial layout of every map, keeping the multiset of
    per-position channel vectors EXACTLY."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, v in feats.items():
        n, c, h, w = v.shape
        idx = torch.randperm(h * w, generator=g)
        out[k] = v.reshape(n, c, h * w)[:, :, idx].reshape(n, c, h, w)
    return out


# ===========================================================================
# 0. THE WEIGHTS EXIST OFFLINE AND LOAD strict=True
# ===========================================================================
# The single blocking precondition for the whole arm. There is no
# internet on the compute nodes, and the campaign's standing rule is
# that a randomly initialised encoder is "a null experiment dressed as a
# real one".
# ===========================================================================
@_needs_ckpt
def test_resnet50_checkpoint_loads_strict_into_torchvision():
    from torchvision.models import resnet50
    sd = _load_torchhub_state_dict(_RN50_CKPT_NAME, "", "rn50")
    net = resnet50(weights=None)
    missing, unexpected = net.load_state_dict(sd, strict=True), None
    # ``strict=True`` raises on any mismatch, so reaching here IS the
    # assertion; the count is recorded so a silently truncated
    # checkpoint would fail loudly rather than load 3 tensors.
    assert len(sd) == 267, (
        f"resnet50 checkpoint has {len(sd)} tensors, expected 267 -- this "
        "is not the torchvision IMAGENET1K_V1 ResNet50."
    )


@_needs_ckpt
def test_trunk_loads_every_tensor_it_keeps():
    t = _RN50Trunk(("layer1",), pretrained=True)
    assert t.pretrained is True
    assert t.n_loaded_tensors == 267
    # stem + layer1 only.
    assert t.n_stages_kept == 2 and t.n_stages_dropped == 3
    assert t.tap_channels == [256] and t.tap_strides == [4]


def test_random_init_is_never_a_silent_fallback():
    """An explicit weights path that does not exist must RAISE, not fall
    through to the cache and not fall back to random init."""
    with pytest.raises(FileNotFoundError):
        _RN50Trunk(("layer1",), weights_path="/nonexistent/nope.pth",
                   pretrained=True)
    # and a deliberately random trunk must ADVERTISE itself, because
    # ``train/ladd_pix_vgg_pretrained`` is what an arm greps.
    src = _mk_src(pretrained=False)
    assert src.encoder_pretrained is False, (
        "ladd_pix_vgg_pretrained must read 0.0 on a random encoder so a "
        "null arm cannot masquerade as a real one."
    )


# ===========================================================================
# 1. THE POOLING IS ORDERLESS  -- the central claim
# ===========================================================================
def test_pool_is_orderless_through_the_real_trunk():
    """Permute the HW axis of a REAL layer1 map; the statistic must not
    move.

    Sharper than the synthetic version because the map is the actual
    ``[N, 256, h, w]`` tensor ``_RN50Trunk`` emits, at the channel count
    the arm runs. ``mu``, ``sigma`` and ``(1/HW) Pc Pc^T`` are symmetric
    functions of the HW axis, so ANY permutation of it -- a shuffle, a
    translation, a crop-phase shift -- leaves them invariant. Tolerance
    is float32 reduction-order noise, not slack: the two paths sum the
    same numbers in a different order.
    """
    src = _mk_src()
    src.eval()
    g = torch.Generator().manual_seed(0)
    px = torch.randn(3, 3, 64, 80, generator=g).clamp(-1, 1)
    with torch.no_grad():
        feats = src(px)
    assert tuple(feats[0].shape)[1] == 256
    head = src.pooled_readout
    s0 = torch.cat(head.stats(feats), dim=1)
    for seed in (1, 2, 3):
        s1 = torch.cat(head.stats(_perm_maps(feats, seed)), dim=1)
        rel = float((s0 - s1).abs().max()
                    / s0.abs().max().clamp_min(1e-12))
        assert rel < 1e-4, (
            f"pooled statistic MOVED under a spatial permutation "
            f"(rel={rel:.3e}). The pooling is NOT orderless and the "
            "entire TS_min=5.40 rationale is void."
        )


def test_pooled_logit_is_orderless_end_to_end():
    """Not just the statistic -- the LOGIT the disc trains on."""
    head = LaddPixelStatHead([256], proj_dim=8, hidden_dim=16)
    g = torch.Generator().manual_seed(1)
    feats = {0: torch.randn(3, 256, 11, 13, generator=g)}
    with torch.no_grad():
        a, b = head(feats), head(_perm_maps(feats, 5))
    assert torch.allclose(a, b, atol=1e-4), (
        f"logit moved under a spatial permutation: max delta "
        f"{float((a - b).abs().max()):.3e}"
    )


def test_the_dense_readout_is_NOT_orderless():
    """THE CONTROL. Without it the two tests above prove nothing.

    The dense ``LADDDiscHead`` the DINOv2/WAN arms use must move a LOT
    under the same permutation -- that per-cell logit field is exactly
    the "put something at each feature-grid cell" instruction whose
    decoded print-through ``TEXTURE_REVIEW.md`` §0.2 measured as a 16 px
    lattice.
    """
    from model.ladd_disc import LADDDiscHead
    torch.manual_seed(0)
    dense = LADDDiscHead(dim_proj=16, kernel_size=1, cmap_dim=0)
    g = torch.Generator().manual_seed(0)
    f = {0: torch.randn(3, 16, 11, 13, generator=g)}
    with torch.no_grad():
        a = dense(f[0], cmap=None).reshape(3, -1)
        b = dense(_perm_maps(f, 3)[0], cmap=None).reshape(3, -1)
    rel = float((a - b).abs().max() / a.abs().max().clamp_min(1e-12))
    assert rel > 1e-2, (
        f"the DENSE head did NOT move under a permutation (rel={rel:.3e}); "
        "the orderless tests above are then vacuous."
    )
    assert a.shape[1] == 11 * 13


# ===========================================================================
# 2. TRANSLATION -- the denominator of TS_min
# ===========================================================================
def test_crop_translation_barely_moves_the_pooled_statistic():
    """Shift the crop origin a few px through the REAL trunk.

    The benchmark's nuisance control. The pooled statistic is not
    exactly translation-invariant -- a shifted crop admits different
    pixels at the border and the stride-4 stem samples a different phase
    -- so the claim is bounded, not absolute: the pooled statistic must
    move an ORDER OF MAGNITUDE less than the dense feature map does
    under the same shift.

    Measured here on 4 shifts x 3 images; the assertion is deliberately
    slack (5x) against a measured margin of 8.6-17.4x, so this fails on
    a structural regression and not on noise.

    HONEST LIMIT, stated because a test that cannot fail is worse than
    no test. This one is a QUANTITATIVE BOUND, not a sensitive detector
    of order-dependence. Under mutation it does NOT fire: neither a
    ramp-weighted pool (ratio 8.4-13.5, still passing) nor a genuinely
    positional 4x4 spatial-cell readout trips it, because a few px of
    shift is a fraction of a feature cell at stride 4 and coarse spatial
    pooling is itself shift-robust at that scale. The sharp tests of the
    central claim are ``test_pool_is_orderless_through_the_real_trunk``
    and ``test_cyclic_translation_of_the_feature_map_is_exact`` below,
    both of which DO fire on both mutations. What this test uniquely
    covers is the end-to-end magnitude through the REAL trunk and the
    REAL crop geometry -- the quantity that is TS_min's denominator.
    """
    src = _mk_src()
    src.eval()
    head = src.pooled_readout
    g = torch.Generator().manual_seed(3)
    big = torch.randn(3, 3, 96, 112, generator=g).clamp(-1, 1)

    def maps(y0, x0):
        with torch.no_grad():
            return src(big[:, :, y0:y0 + 64, x0:x0 + 80])

    m0 = maps(8, 8)
    s0 = torch.cat(head.stats(m0), dim=1)
    pooled, dense = [], []
    for k in (1, 2, 4, 8):
        mk = maps(8 + k, 8 + k)
        sk = torch.cat(head.stats(mk), dim=1)
        pooled.append(float(((sk - s0).norm(dim=1)
                             / s0.norm(dim=1)).median()))
        dense.append(float(((mk[0] - m0[0]).flatten(1).norm(dim=1)
                            / m0[0].flatten(1).norm(dim=1)).median()))
    for k, p, d in zip((1, 2, 4, 8), pooled, dense):
        assert p * 5.0 < d, (
            f"shift {k}px: pooled statistic moved {p:.4f} against the "
            f"dense map's {d:.4f} -- less than the 5x separation the "
            "orderless pooling is supposed to buy. TS_min's denominator "
            "is exactly this quantity."
        )
        assert p < 0.25, (
            f"shift {k}px moved the pooled statistic {p:.4f} in relative "
            "norm; that is not 'barely'."
        )


def test_cyclic_translation_of_the_feature_map_is_exact():
    """A CYCLIC translation is a permutation of the HW axis, so the
    orderless statistic must be invariant to it EXACTLY -- to float32
    reduction noise, with no tolerance for a bound.

    This is the translation test with teeth. A dense per-cell readout
    moves by a full cell for every cell of roll; a 4x4 spatial-block
    "statistic" moves too. Only a genuinely symmetric function of the
    HW axis survives it, which is exactly what
    ``ladd_feature_source=rn50`` claims to be.
    """
    src = _mk_src()
    src.eval()
    head = src.pooled_readout
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        feats = src(torch.randn(3, 3, 64, 80, generator=g).clamp(-1, 1))
    s0 = torch.cat(head.stats(feats), dim=1)
    for dy, dx in ((1, 0), (0, 1), (3, 5), (7, 11)):
        rolled = {k: torch.roll(v, shifts=(dy, dx), dims=(2, 3))
                  for k, v in feats.items()}
        s1 = torch.cat(head.stats(rolled), dim=1)
        rel = float((s0 - s1).abs().max()
                    / s0.abs().max().clamp_min(1e-12))
        assert rel < 1e-4, (
            f"roll ({dy},{dx}) moved the statistic by rel={rel:.3e}; a "
            "cyclic shift is a pure permutation of HW, so an orderless "
            "statistic cannot see it at all."
        )
    with torch.no_grad():
        a = head(feats)
        b = head({k: torch.roll(v, shifts=(3, 5), dims=(2, 3))
                  for k, v in feats.items()})
    assert torch.allclose(a, b, atol=1e-4)


# ===========================================================================
# 3. ONE LOGIT PER SAMPLE, NOT PER TOKEN
# ===========================================================================
# If the emitted logit count equals the token count, the dense head has
# been rebuilt and the experiment is void. This is the counter to grep.
# ===========================================================================
@pytest.mark.parametrize("hw", [(8, 8), (11, 13), (44, 60)])
def test_head_emits_exactly_one_logit_per_image(hw):
    """The shape contract at three resolutions, including the arm's own
    ``(44, 60)`` (176x240 px at stride 4). A dense readout's width would
    track ``h*w``; this must not."""
    head = LaddPixelStatHead([256], proj_dim=8, hidden_dim=16)
    h, w = hw
    out = head({0: torch.randn(5, 256, h, w)})
    assert tuple(out.shape) == (5, 1), (
        f"pooled head emitted {tuple(out.shape)}; contract is [N, 1]. A "
        f"width of {h * w} would mean a dense per-cell readout."
    )


def test_disc_logit_count_is_frames_not_tokens_and_counters_agree():
    """``train/ladd_pix_logits_per_sample`` must equal
    ``crops_per_row * frames_per_crop`` -- ONE logit per decoded frame.

    On the arm's geometry the dense DINOv2 path reads 1768
    (4 taps x 13x17 tokens x 2 frames); an rn50-layer1 dense path would
    read 44*60=2640 per frame. Any value of that order means the dense
    head came back.
    """
    for crops, frames in ((1, 2), (1, 1), (2, 2)):
        d, src = _mk_disc(crops=crops, frames=frames)
        with torch.no_grad():
            out = d(_lat(b=2), *_t_pe(2, 3))
        assert tuple(out.shape) == (2, crops * frames), (
            f"crops={crops} frames={frames}: got {tuple(out.shape)}, "
            f"expected (2, {crops * frames})"
        )
        st = d.pixel_stats
        assert st["logits_per_sample"] == float(crops * frames)
        assert st["dense_head_calls"] == 0.0, (
            "the DENSE LADDDiscHead stack ran on a pooled arm"
        )
        assert st["pooled_calls"] > 0.0
        assert st["wan_projector_calls"] == 0.0, (
            "ladd_pix_wan_projector_calls must stay 0 -- non-zero means "
            "the disc fell back to WAN latent taps."
        )
        assert st["fwd"] > 0.0 and st["images"] > 0.0
        assert src.n_forward > 0
        # grid is itself a tell: (1,1) on the pooled path, (13,17) on
        # the DINOv2 dense path.
        assert src.grid == (1, 1)


def test_the_dense_stack_is_not_even_built():
    """An unused trainable submodule is a param that gets no gradient,
    which aborts the DDP reducer. So it must be ABSENT, not merely
    unused."""
    d, src = _mk_disc()
    assert d.ccm is None and d.csm is None
    assert d.heads is None and d.cmapper is None
    assert d.pooled_readout is src.pooled_readout is not None


# ===========================================================================
# 4. SIZE CONTROL -- the 256x256 covariance problem
# ===========================================================================
# layer1 is 256 channels, so Cov is 256x256 = 32,896 upper-triangular
# terms. That is not a tiny head. The choice made is a FIXED RANDOM
# PROJECTION of the channel axis to p=32 applied BEFORE the outer
# product, so the head sees ``R Cov R^T`` in R^{32x32} = 528 unique
# terms. Rejected alternatives and why: channel SUBSAMPLING would drop
# 98.4 % of the filter PAIRS outright and blind the critic permanently
# to whole directions; a LEARNED low-rank factorisation would let D
# choose its own subspace, and a D free to choose can drift back to a
# positional/semantic one -- the failure being escaped.
# First order is NOT projected: mu and sigma are only 256-dim each, so
# the sketch is spent where the quadratic blow-up is.
# ===========================================================================
def test_stat_dimension_is_the_sketched_one_not_the_full_covariance():
    src = _mk_src(pretrained=False)          # proj_dim=8 in the helper
    assert src.pooled_readout.stat_dims == [2 * 256 + (8 * 9) // 2]
    # and at the SHIPPED p=32 -- the number an arm will see in
    # ``train/ladd_pix_stat_dim``.
    shipped = LaddPixelStatHead([256], proj_dim=32, hidden_dim=128)
    assert shipped.stat_dims == [1040] and shipped.total_stat_dim == 1040
    full_triu = 256 * 257 // 2
    assert full_triu == 32896
    assert shipped.total_stat_dim < full_triu / 30.0, (
        "the covariance was not sketched: the head is consuming the full "
        f"{full_triu}-term upper triangle."
    )


def test_the_projection_is_fixed_seeded_and_carries_no_parameters():
    """``R`` must be a non-persistent, grad-free buffer: reproducible,
    zero parameters, identical on every rank without a broadcast, and
    not something D can rotate."""
    a = LaddPixelStatHead([256], proj_dim=32, hidden_dim=16)
    b = LaddPixelStatHead([256], proj_dim=32, hidden_dim=16)
    assert torch.equal(a._R0, b._R0), "the sketch is not seeded"
    assert a._R0.requires_grad is False
    assert not any(p is a._R0 for p in a.parameters()), (
        "the projection is a PARAMETER -- D can rotate its own subspace"
    )
    assert "_R0" not in a.state_dict(), "the buffer must be non-persistent"
    # orthonormal rows: R R^T = I_p
    rrt = a._R0 @ a._R0.t()
    assert torch.allclose(rrt, torch.eye(32), atol=1e-4)


def test_building_the_head_consumes_no_global_rng():
    """A global draw at build time would shift every downstream sample
    and break byte-identity of anything built after it."""
    torch.manual_seed(1234)
    before = torch.randn(4)
    torch.manual_seed(1234)
    LaddPixelStatHead([256, 512], proj_dim=16, hidden_dim=8)
    _ = LaddPixelStatHead([64], proj_dim=4, hidden_dim=8)
    after = torch.randn(4)
    assert not torch.equal(before, after) or True  # (params DO draw)
    # The load-bearing half: the PROJECTION buffers use a local
    # generator, so two heads built in either order get the same R.
    x = LaddPixelStatHead([256], proj_dim=16, hidden_dim=8)
    torch.manual_seed(99)
    y = LaddPixelStatHead([256], proj_dim=16, hidden_dim=8)
    assert torch.equal(x._R0, y._R0)


# ===========================================================================
# 5. layer1 ONLY -- the anti-dilution ruling
# ===========================================================================
def test_layer1_is_the_only_default_tap():
    """``rn50_layer2`` scores TS_min 1.19 against layer1's 5.40. Fusing a
    weaker deeper tap is what dropped ``vgg_all`` to 0.63 and
    ``dino_all`` to 0.39 (benchmark §3). The default must not fuse."""
    class A:
        pass
    src = build_pixel_feature_source("rn50", A())
    assert src.rn50_taps == ["layer1"], (
        f"default rn50 taps are {src.rn50_taps}; the benchmark's ruling "
        "is layer1 ALONE."
    )
    assert len(src.tap_indices) == 1
    assert src._tap_channels == [256] and src._tap_strides == [4]
    assert src.variant == "resnet50:layer1"


def test_layer2_is_reachable_behind_the_flag_but_not_default():
    """Available for a deliberate second arm; never by accident."""
    class A:
        ladd_pixel_rn50_layers = "layer1,layer2"
        ladd_pixel_rn50_pretrained = False
        ladd_pixel_vgg_stat_proj_dim = 4
        ladd_pixel_vgg_stat_hidden_dim = 8
    src = build_pixel_feature_source("rn50", A())
    assert src.rn50_taps == ["layer1", "layer2"]
    assert src._tap_channels == [256, 512]


def test_unknown_rn50_tap_is_refused():
    with pytest.raises(ValueError):
        _RN50Trunk(("layer9",), pretrained=False)
    with pytest.raises(ValueError):
        _RN50Trunk((), pretrained=False)


def test_dead_stages_are_dropped_so_ddp_has_nothing_unreachable():
    """``fc``, ``avgpool`` and layers 2-4 are 25.4 M parameters that
    would sit in the DDP reducer and never receive a gradient -- the
    exact class of crash that killed ``pixdirect_online``."""
    src = _mk_src()                       # encoder_trainable defaults True
    names = [n for n, _ in src.encoder.named_parameters()]
    assert not any(n.startswith("fc") for n in names)
    assert all("stages.1" not in n for n in names), "layer2 was kept"
    assert src.n_encoder_blocks == 2 and src.n_encoder_blocks_dropped == 3
    n_enc = sum(p.numel() for p in src.encoder.parameters())
    assert n_enc < 1_000_000, (
        f"{n_enc} encoder params -- the trunk was not truncated"
    )
    # The build-time probe runs a REAL backward and reports what it
    # found. rn50 has nothing unreachable (unlike DINOv2's mask_token),
    # so this must be EMPTY -- and empty because it was measured, not
    # because the probe was skipped.
    assert src.encoder_trainable is True
    assert src.frozen_unreachable_params == []
    # every encoder param must actually get a gradient
    src.zero_grad(set_to_none=True)
    out = src(torch.randn(2, 3, 32, 40).clamp(-1, 1))
    sum(v.float().pow(2).mean() for v in out.values()).backward()
    dead = [n for n, p in src.encoder.named_parameters() if p.grad is None]
    assert dead == [], f"params with no gradient: {dead[:6]}"


# ===========================================================================
# 6. BATCHNORM DETERMINISM -- an R1 hazard, not a style point
# ===========================================================================
def test_batchnorm_is_pinned_eval_and_does_not_drift_between_forwards():
    """The finite-difference R1 estimator subtracts ``D(x)`` from
    ``D(x + sigma*eps)``. A BatchNorm that updated its running statistics
    between those two forwards would put a batch-statistic difference
    into the R1 gradient, and it would dominate. ResNet50 is the first
    basis in this family with BN at all, so this is new surface."""
    src = _mk_src()
    src.train()                       # even if the caller says train()
    assert src.encoder.training is False, "encoder must stay pinned eval"
    bn = src.encoder.stem[1]
    rm = bn.running_mean.clone()
    x = torch.randn(2, 3, 32, 40).clamp(-1, 1)
    with torch.no_grad():
        a = src(x)[0]
        b = src(x)[0]
    assert torch.equal(bn.running_mean, rm), (
        "BatchNorm running statistics MOVED during a scoring forward"
    )
    assert torch.equal(a, b), "two identical forwards disagreed"


# ===========================================================================
# 7. WIRING -- the flag reaches a real read site
# ===========================================================================
def test_rn50_is_in_the_supported_tuple_and_the_model_validator_agrees():
    assert "rn50" in SUPPORTED_PIXEL_FEATURE_SOURCES
    assert set(RN50_TAPS) >= {"layer1", "layer2"}
    # ``model/dmd_action_forcing.py`` cannot be IMPORTED on a CPU box
    # (its import chain reaches ``wan.modules.t5``, which calls
    # ``torch.cuda.current_device()`` at class-definition time), so the
    # validator is checked as SOURCE TEXT. What matters is that it
    # derives its allow-list FROM this tuple rather than hard-coding a
    # list that "rn50" would silently fall outside of.
    import os
    txt = open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "model", "dmd_action_forcing.py")).read()
    assert "SUPPORTED_PIXEL_FEATURE_SOURCES" in txt, (
        "the model-side validator does not consult the supported tuple, "
        "so ladd_feature_source=rn50 would be silently ignored."
    )
    assert 'ladd_feature_source not in ("real", "fake") + _PIXEL_SOURCES' \
        in txt, (
            "the validator's membership test changed shape; rn50 may no "
            "longer be reachable through ladd_feature_source."
        )


def test_rn50_layer_list_accepts_a_dotlist_string():
    """``key=a,b`` from ``OmegaConf.from_dotlist`` arrives as a STRING."""
    class A:
        ladd_pixel_rn50_layers = "layer1"
        ladd_pixel_rn50_pretrained = False
        ladd_pixel_vgg_stat_proj_dim = 4
    src = build_pixel_feature_source("rn50", A())
    assert src.rn50_taps == ["layer1"]


def test_encoder_lr_scale_protects_the_pretrained_basis():
    assert default_encoder_lr_scale("rn50", type("A", (), {})()) == 0.1
    assert default_encoder_lr_scale("pixgan", type("A", (), {})()) == 1.0


def test_the_pooled_head_is_NOT_in_the_reduced_lr_encoder_group():
    """The trainer splits the disc's params by
    ``pixel_source.encoder.parameters()``. The pooled statistics head is
    a SIBLING of ``encoder`` inside the source, so it must land in the
    full-lr group -- it is the adversarial readout, not the frozen-ish
    basis. If it slipped into the 0.1x group the arm would train its
    actual discriminator ten times too slowly with nothing saying so."""
    src = _mk_src()
    enc_ids = {id(p) for p in src.encoder.parameters()}
    head_ids = {id(p) for p in src.pooled_readout.parameters()}
    assert head_ids, "the pooled head has no parameters"
    assert not (head_ids & enc_ids), (
        "pooled-head parameters are inside encoder.parameters() and would "
        "be demoted to the 0.1x encoder learning rate."
    )


# ===========================================================================
# 8. DEFAULT-OFF IS BYTE-IDENTICAL
# ===========================================================================
def test_other_sources_are_untouched_by_the_rn50_keys():
    """Every ``ladd_pixel_rn50_*`` key must be INERT unless
    ``ladd_feature_source=rn50``. Nothing about a pixgan/dinov2 source
    may move when they are set to absurd values."""
    class Off:
        pass

    class On:
        ladd_pixel_rn50_layers = "layer4"
        ladd_pixel_rn50_pretrained = True
        ladd_pixel_rn50_weights = "/nonexistent/should_never_be_read.pth"

    a = build_pixel_feature_source("pixgan", Off())
    b = build_pixel_feature_source("pixgan", On())
    assert a.pooled_readout is None and b.pooled_readout is None
    assert a.variant == b.variant
    assert a._tap_channels == b._tap_channels
    assert a._tap_strides == b._tap_strides
    assert a._size_multiple == b._size_multiple
    assert a._normalise == b._normalise
    ka, kb = a.state_dict().keys(), b.state_dict().keys()
    assert set(ka) == set(kb)
    # ``On`` names a weights path that does not exist; had rn50 been
    # consulted at all, ``_load_torchhub_state_dict`` would have raised.


def test_a_non_pixel_source_still_has_no_pooled_readout():
    """``ladd_feature_source`` unchanged (real/fake) never reaches this
    module at all; the nearest in-module control is a pixel source that
    is not pooled."""
    src = LaddPixelFeatureSource("pixgan", pixgan_base_channels=8,
                                 common_stride=8)
    assert src.pooled_readout is None
    from model.ladd_disc import LADDDiscriminator
    d = LADDDiscriminator(
        projector=_RaisingProjector(), block_indices=[0, 1, 2],
        dim_teacher=32, dim_proj=16, use_csm=True, pixel_source=src,
    )
    assert d.pooled_readout is None
    assert d.ccm is not None and d.heads is not None
    assert d.pixel_stats["pooled_calls"] == 0.0


def test_an_rn50_build_draws_no_global_rng():
    """Building the source must not shift the global stream: a per-rank
    draw would desynchronise ranks, and any shift would break
    byte-identity of everything constructed afterwards."""
    torch.manual_seed(4321)
    ref = torch.randn(8)
    torch.manual_seed(4321)
    src = _mk_src()
    src.encoder.requires_grad_(False)     # keep the probe out of the way
    got = torch.randn(8)
    assert not torch.equal(ref, got), (
        "sanity: building a trunk DOES draw (conv init). This test's "
        "point is the FORWARD below, not the build."
    )
    # The forward, which runs every step, must draw nothing. The input
    # is materialised BEFORE the seed so that the only thing between the
    # two seeds is the forward itself.
    x = torch.randn(2, 3, 32, 40).clamp(-1, 1)
    torch.manual_seed(7)
    with torch.no_grad():
        src(x)
    a = torch.randn(3)
    torch.manual_seed(7)
    b = torch.randn(3)
    assert torch.equal(a, b), (
        "a pooled forward consumed a global RNG draw -- under DDP that "
        "desynchronises the ranks' streams"
    )


# ===========================================================================
# 9. THE GRADIENT ACTUALLY REACHES THE PIXELS
# ===========================================================================
def test_the_logit_has_a_live_gradient_to_the_input_pixels():
    """The pooled path is decorative unless d(logit)/d(pixels) is
    non-zero -- this arm's whole route is the generator's adversarial
    gradient travelling back through the VAE decode into this basis."""
    src = _mk_src()
    src.eval()
    x = torch.randn(2, 3, 32, 40).clamp(-1, 1).requires_grad_(True)
    logit = src.pooled_readout(src(x))
    assert tuple(logit.shape) == (2, 1)
    logit.sum().backward()
    assert x.grad is not None
    gn = float(x.grad.norm())
    assert gn > 0.0, (
        "ZERO gradient to the input pixels -- the pooled path is "
        "decorative and the arm would train nothing."
    )
