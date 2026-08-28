"""CPU tests for the VGG ORDERLESS-STATISTICS discriminator basis.

    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" PYTHONPATH=. \
        python -m pytest testing/test_ladd_vgg_stat_source.py -q

(``OMP_NUM_THREADS=8`` is MANDATORY on this box: ``nproc`` is 144 and the
default thread count makes torch's CPU ops thrash for 30+ minutes.)

WHAT IS COVERED, hardest claim first
====================================
The central claim of ``ladd_feature_source=vgg`` is that the adversarial
signal stops being a FIELD ON A FEATURE GRID ("put something at each
cell", which the VAE decoder prints as a lattice -- ``TEXTURE_REVIEW.md``
§0.2 measured it at 16 px) and becomes a statement about the
DISTRIBUTION of local filter responses. Three things must hold for that
claim to be true rather than decorative, and each has its own test:

1.  **THE POOLING IS ORDERLESS.** Permuting the spatial layout of a
    feature map must leave the pooled statistic UNCHANGED. This is the
    sharpest possible test of the central claim, and the same test run
    against the dense ``LADDDiscHead`` readout must FAIL -- otherwise
    the test proves nothing about the difference between them.
    (``test_pooling_is_orderless``,
    ``test_the_dense_readout_is_NOT_orderless``.)
2.  **THE READOUT EMITS ONE LOGIT PER SAMPLE, NOT PER TOKEN.** If the
    emitted logit count ever equals the token count, a dense head has
    been reintroduced and the experiment is void.
    (``test_head_emits_exactly_one_logit_per_image``,
    ``test_disc_pooled_logit_count_is_frames_not_tokens``.)
3.  **DEFAULT-OFF IS BYTE-IDENTICAL.** Proven by EXACT-MATCH against the
    pre-change ``model/ladd_disc.py`` loaded from git, not by reading
    the diff. (``test_default_off_is_bit_exact_vs_git_head``.)

Plus the guards that turn a silent wrong result into a crash:

4.  ``1/HW`` normalisation is real -- tiling a feature map 4x must not
    change the statistic (it would 4x a Gram that forgot the divide).
5.  No random-init fallback: a missing checkpoint RAISES. A randomly
    initialised VGG is a null experiment dressed as a real one.
6.  Every encoder AND head parameter receives gradient in one backward,
    so the DDP reducer cannot abort the way ``pixdirect_online`` did.
7.  The statistic is TWO-SIDED (the standing "too smooth is as wrong as
    too sharp" rule): blur AND grain must both move it away from real.
8.  Measured, not asserted: the pooled statistic is LESS crop-phase
    sensitive than the dense feature map. See the docstring of
    ``test_pooled_is_less_phase_sensitive_than_dense`` for the honest
    size of that effect -- it is 2x, NOT invariance, because VGG's own
    stride-2 max-pools are shift-variant.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.ladd_disc import LADDDiscriminator  # noqa: E402
from model.ladd_pixel_features import (  # noqa: E402
    SUPPORTED_PIXEL_FEATURE_SOURCES, LaddPixelFeatureSource,
    LaddPixelStatHead, VGG16_TAPS, _load_vgg16_state_dict,
    build_pixel_feature_source, default_encoder_lr_scale,
)

# The pretrained tests need the VGG16 checkpoint. VERIFIED PRESENT on
# this cluster (553,433,881 bytes, 32 state-dict keys) -- skip rather
# than fail if someone runs this elsewhere.
def _vgg_ckpt_path():
    """File-EXISTENCE probe only. Deliberately does not ``torch.load``:
    the checkpoint is 553 MB and this box has a 4 GB per-user cgroup
    cap, so an import-time load-and-discard is a real OOM risk."""
    import torch as _t
    c = []
    th = os.environ.get("TORCH_HOME", "")
    if th:
        c.append(os.path.join(th, "hub", "checkpoints", "vgg16-397923af.pth"))
    try:
        c.append(os.path.join(_t.hub.get_dir(), "checkpoints",
                              "vgg16-397923af.pth"))
    except Exception:
        pass
    c.append(os.path.expanduser(
        "~/.cache/torch/hub/checkpoints/vgg16-397923af.pth"))
    for x in c:
        if os.path.isfile(x):
            return x
    return ""


_VGG_CKPT = _vgg_ckpt_path()
_HAS_VGG = bool(_VGG_CKPT)
_vgg_only = pytest.mark.skipif(
    not _HAS_VGG, reason="ImageNet VGG16 weights not in the torch-hub cache")


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
            "WanFeatureProjector was called on a VGG-source disc. The "
            "feature basis did NOT swap."
        )


def _fake_decode(latents, want_grad):
    n, f, c, h, w = latents.shape
    x = latents if want_grad else latents.detach()
    x = x[:, :, :3] if c >= 3 else x.repeat(1, 1, 3, 1, 1)[:, :, :3]
    x = x.reshape(n * f, 3, h, w)
    x = F.interpolate(x, scale_factor=8, mode="nearest")
    return torch.tanh(x).reshape(n, f, 3, h * 8, w * 8)


def _mk_src(pretrained=False, layers=("relu1_2", "relu2_2"), **kw):
    """A SMALL VGG source. ``pretrained=False`` is legal ONLY here --
    see ``test_random_init_is_never_a_silent_fallback``."""
    return LaddPixelFeatureSource(
        "vgg", vgg_layers=layers, vgg_pretrained=pretrained,
        vgg_stat_proj_dim=8, vgg_stat_hidden_dim=16, **kw,
    )


def _mk_disc(src=None, *, crops=1, frames=2, **kw):
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
    """Randomly PERMUTE the spatial layout of every map, keeping the
    multiset of per-position channel vectors exactly."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, v in feats.items():
        n, c, h, w = v.shape
        idx = torch.randperm(h * w, generator=g)
        out[k] = v.reshape(n, c, h * w)[:, :, idx].reshape(n, c, h, w)
    return out


# ===========================================================================
# 1. THE POOLING IS ORDERLESS  -- the central claim
# ===========================================================================
def test_pooling_is_orderless():
    """PERMUTE the spatial layout; the statistic must not move.

    This is the whole design in one assertion. mean / var /
    (1/HW) Pc Pc^T are symmetric functions of the HW axis, so ANY
    permutation of it -- a shuffle, a translation, a crop-phase shift --
    leaves them invariant. Tolerance is float32 reduction-order noise,
    not slack: the two paths sum the same numbers in a different order.
    """
    head = LaddPixelStatHead([16, 24], proj_dim=8, hidden_dim=16)
    g = torch.Generator().manual_seed(0)
    feats = {0: torch.randn(3, 16, 11, 13, generator=g),
             1: torch.randn(3, 24, 6, 7, generator=g)}
    s0 = torch.cat(head.stats(feats), dim=1)
    for seed in (1, 2, 3):
        s1 = torch.cat(head.stats(_perm_maps(feats, seed)), dim=1)
        rel = (s0 - s1).abs().max() / s0.abs().max().clamp_min(1e-12)
        assert rel < 1e-4, (
            f"pooled statistic MOVED under a spatial permutation "
            f"(rel={float(rel):.3e}). The pooling is NOT orderless and "
            f"the whole experiment is void."
        )


def test_pooled_logit_is_orderless_end_to_end():
    """Not just the statistic -- the LOGIT the disc trains on."""
    head = LaddPixelStatHead([16, 24], proj_dim=8, hidden_dim=16)
    g = torch.Generator().manual_seed(1)
    feats = {0: torch.randn(3, 16, 11, 13, generator=g),
             1: torch.randn(3, 24, 6, 7, generator=g)}
    with torch.no_grad():
        a, b = head(feats), head(_perm_maps(feats, 5))
    assert torch.allclose(a, b, atol=1e-4), (
        f"logit moved under a spatial permutation: max delta "
        f"{float((a - b).abs().max()):.3e}"
    )


def test_the_dense_readout_is_NOT_orderless():
    """THE CONTROL. Without this, the test above proves nothing.

    The dense ``LADDDiscHead`` readout the DINOv2 / WAN arms use must
    move a LOT under the same permutation -- that is precisely the
    "put something at each feature-grid cell" signal being escaped.
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
        "the DENSE head did NOT move under a spatial permutation "
        f"(rel={rel:.3e}); the orderless test above is then vacuous."
    )
    # and it emits one logit PER CELL, which is the thing being escaped
    assert a.shape[1] == 11 * 13


# ===========================================================================
# 2. ONE LOGIT PER SAMPLE, NOT PER TOKEN
# ===========================================================================
@pytest.mark.parametrize("hw", [(8, 8), (11, 13), (32, 40)])
def test_head_emits_exactly_one_logit_per_image(hw):
    """The shape contract, at three resolutions. A dense head's output
    width would track ``hw``; this must not."""
    head = LaddPixelStatHead([16, 24], proj_dim=8, hidden_dim=16)
    h, w = hw
    feats = {0: torch.randn(5, 16, h, w), 1: torch.randn(5, 24, h, w)}
    out = head(feats)
    assert tuple(out.shape) == (5, 1), (
        f"pooled head emitted {tuple(out.shape)}; contract is [N, 1]. "
        f"A width of {h * w} would mean a dense per-cell readout."
    )


def test_disc_pooled_logit_count_is_frames_not_tokens():
    """The number to grep in wandb.

    ``train/ladd_pix_logits_per_sample`` must equal
    ``crops_per_row * frames_per_crop`` -- ONE logit per decoded frame.
    The DINOv2 arm reads 1768 (4 taps x 13x17 x 2 frames). Any value of
    that order means the dense head came back.
    """
    for crops, frames in ((1, 2), (1, 1), (2, 2)):
        d, src = _mk_disc(crops=crops, frames=frames)
        x = _lat(b=2)
        t, pe = _t_pe(2, 3)
        with torch.no_grad():
            out = d(x, t, pe)
        assert tuple(out.shape) == (2, crops * frames), (
            f"crops={crops} frames={frames}: got {tuple(out.shape)}, "
            f"expected (2, {crops * frames})"
        )
        assert d.pixel_stats["logits_per_sample"] == float(crops * frames)
        assert d.pixel_stats["dense_head_calls"] == 0.0, (
            "the DENSE LADDDiscHead stack ran on a pooled arm"
        )
        assert d.pixel_stats["pooled_calls"] > 0.0
        assert d.pixel_stats["wan_projector_calls"] == 0.0


def test_the_dense_stack_is_not_even_built():
    """An unused trainable submodule is a param that gets no gradient,
    which aborts the DDP reducer -- the exact class of bug that killed
    ``pixdirect_online``. So it must be absent, not merely unused."""
    d, src = _mk_disc()
    assert d.ccm is None and d.csm is None
    assert d.heads is None and d.cmapper is None
    assert d.pooled_readout is src.pooled_readout is not None
    assert src.grid == (0, 0)          # before any forward
    with torch.no_grad():
        d(_lat(b=2), *_t_pe(2, 3))
    assert src.grid == (1, 1), (
        "grid must read (1,1) on the pooled path -- ladd_pix_grid_h/w "
        "is itself a tell (the DINOv2 arm reads 13/17)."
    )


def test_dinov2_and_pixgan_keep_the_dense_stack():
    """The negative control for the branch above."""
    src = LaddPixelFeatureSource("pixgan", pixgan_base_channels=8,
                                 common_stride=8)
    assert src.pooled_readout is None
    d = LADDDiscriminator(
        projector=_RaisingProjector(), block_indices=[0, 1, 2],
        dim_teacher=32, dim_proj=16, use_csm=True, pixel_source=src,
    )
    assert d.pooled_readout is None
    assert d.ccm is not None and d.heads is not None


# ===========================================================================
# 3. DEFAULT-OFF IS BYTE-IDENTICAL -- proven against git, not the diff
# ===========================================================================
def _load_module_from_git(path, rev="HEAD", name="_old_mod"):
    import importlib.util
    import subprocess
    import tempfile
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        src = subprocess.check_output(
            ["git", "-C", root, "show", f"{rev}:{path}"],
            stderr=subprocess.DEVNULL,
        ).decode()
    except Exception:
        return None
    fd, fn = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location(name, fn)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class _FixedProjector:
    """A deterministic stand-in for the WAN feature projector, so the
    HISTORICAL (``ladd_feature_source='real'``) path can be exercised
    against the pre-change module."""

    def __init__(self, blocks, seq_len, dim, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.feats = {int(i): torch.randn(2, seq_len, dim, generator=g)
                      for i in blocks}
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        return {k: v.clone() for k, v in self.feats.items()}


def _wan_disc_from(mod, projector, seed=0):
    """The WAN-tap disc: ``pixel_source=None``, i.e. what every arm
    running right now is."""
    torch.manual_seed(seed)
    d = mod.LADDDiscriminator(
        projector=projector, block_indices=[0, 2], dim_teacher=32,
        dim_proj=16, use_csm=True, patch_size=(1, 2, 2),
        action_tokens_per_frame=1,
    )
    d.eval()
    return d


def test_wan_path_is_bit_exact_vs_git_head():
    """EXACT MATCH against the pre-change file, loaded from git.

    Reading a diff is how this campaign has produced false conclusions,
    so this builds the OLD ``LADDDiscriminator`` and the NEW one with the
    same seed, copies the old weights across, and requires BIT-IDENTICAL
    logits on the WAN-tap path -- which is what ``ladd_feature_source``
    at its historical 'real'/'fake' values takes.

    SCOPE, stated rather than implied: the whole pixel-source feature is
    UNCOMMITTED on this branch (``git status`` shows ``M
    model/ladd_disc.py``), so ``HEAD`` predates it and cannot be asked
    about ``pixgan``/``dinov2``. Those two are covered structurally by
    ``test_dinov2_and_pixgan_keep_the_dense_stack`` and
    ``test_pixgan_path_still_emits_one_logit_per_token`` instead. What
    this test does establish is that the edits made for the VGG path did
    not perturb the disc every currently-running arm uses.
    """
    import model.ladd_disc as new_mod
    old_mod = _load_module_from_git("model/ladd_disc.py")
    if old_mod is None or getattr(old_mod, "LADDDiscriminator", None) is None:
        pytest.skip("git HEAD copy of model/ladd_disc.py unavailable")
    # F=2, H=16, W=20 with patch (1,2,2) -> T'=2, H'=8, W'=10 -> 162 tokens
    seq_len = 200
    p_old = _FixedProjector([0, 2], seq_len, 32)
    p_new = _FixedProjector([0, 2], seq_len, 32)
    d_old = _wan_disc_from(old_mod, p_old)
    d_new = _wan_disc_from(new_mod, p_new)
    d_new.load_state_dict(d_old.state_dict(), strict=True)
    x = _lat(b=2, f=2, c=16, h=16, w=20)
    t, pe = _t_pe(2, 2)
    with torch.no_grad():
        a, b = d_old(x, t, pe), d_new(x, t, pe)
    assert a.shape == b.shape and a.shape[1] > 1
    assert torch.equal(a, b), (
        f"the WAN path is NOT bit-exact after the VGG edits: max delta "
        f"{float((a - b).abs().max()):.3e}"
    )
    assert p_old.calls == p_new.calls == 1


def test_pixgan_path_still_emits_one_logit_per_token():
    """The dense pixel path is structurally untouched: it still emits
    ``n_taps * gh * gw * frames`` logits and still runs the dense head.
    This is the CONTRAST the pooled arm is measured against."""
    src = LaddPixelFeatureSource("pixgan", pixgan_base_channels=8,
                                 common_stride=8)
    d = LADDDiscriminator(
        projector=_RaisingProjector(), block_indices=[0, 1, 2],
        dim_teacher=32, dim_proj=16, use_csm=True, pixel_source=src,
    )
    d.pixel_decode_fn = _fake_decode
    d.pixel_cfg = {
        "crop_rows": 8, "crop_cols": 8, "crops_per_row": 1,
        "lat_frames": 2, "frames_per_crop": 2, "border": 2,
        "decode_batch": 4,
    }
    d.eval()
    with torch.no_grad():
        out = d(_lat(b=2), *_t_pe(2, 3))
    gh, gw = src.grid
    n_taps = len(src.tap_indices)
    assert out.shape[1] == n_taps * gh * gw * 2, (
        f"dense pixel readout emitted {out.shape[1]}, expected "
        f"{n_taps} taps x {gh}x{gw} x 2 frames"
    )
    assert d.pixel_stats["dense_head_calls"] == 1.0
    assert d.pixel_stats["pooled_calls"] == 0.0
    assert out.shape[1] > 10, "the contrast case must be genuinely dense"


def test_pooled_forward_consumes_no_global_rng():
    """A pooled forward must not draw: a per-rank draw would
    desynchronise DDP and perturb every other path's byte-identity."""
    d, _ = _mk_disc()
    torch.manual_seed(1234)
    before = torch.get_rng_state()
    with torch.no_grad():
        d(_lat(b=2), *_t_pe(2, 3))
    assert torch.equal(before, torch.get_rng_state())


def test_vgg_is_in_the_supported_tuple_and_the_model_validator_agrees():
    assert "vgg" in SUPPORTED_PIXEL_FEATURE_SOURCES
    # the validator imports the SAME tuple, so they cannot drift
    import re
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    txt = open(os.path.join(root, "model/dmd_action_forcing.py")).read()
    assert re.search(
        r"SUPPORTED_PIXEL_FEATURE_SOURCES as _PIXEL_SOURCES", txt), (
        "model/dmd_action_forcing.py no longer imports the shared tuple; "
        "the validator and the builder can now drift apart."
    )
    txt2 = open(os.path.join(
        root, "trainer/causal_action_forcing_train.py")).read()
    assert "SUPPORTED_PIXEL_FEATURE_SOURCES as _PIX_SRCS" in txt2, (
        "the trainer's build gate no longer reads the shared tuple; "
        "ladd_feature_source=vgg would be silently ignored."
    )


# ===========================================================================
# 4. THE 1/HW NORMALISATION IS REAL
# ===========================================================================
def test_statistic_is_invariant_to_tiling_so_1_over_HW_is_applied():
    """Tile a map 2x2. HW quadruples; mean, std and (1/HW)FF^T do not.

    A Gram that forgot the ``/HW`` would be exactly 4x larger. This is
    the researcher's "normalise by HW so the statistic is
    scale-consistent across differently-sized crops", tested exactly
    rather than approximately.
    """
    head = LaddPixelStatHead([12], proj_dim=6, hidden_dim=8)
    g = torch.Generator().manual_seed(0)
    m = torch.randn(2, 12, 9, 11, generator=g)
    big = m.repeat(1, 1, 2, 2)
    s0 = head.pool(m, 0)
    s1 = head.pool(big, 0)
    rel = float((s0 - s1).abs().max() / s0.abs().max())
    assert rel < 1e-4, (
        f"statistic changed by {rel:.3e} under an exact 2x2 tiling; the "
        f"1/HW normalisation is missing or wrong."
    )


def test_second_order_block_is_the_projected_covariance():
    """The sketch really is ``R Cov(F) R^T``, computed independently."""
    head = LaddPixelStatHead([10], proj_dim=4, hidden_dim=8,
                             signed_sqrt=False, centered=True)
    g = torch.Generator().manual_seed(2)
    m = torch.randn(1, 10, 5, 7, generator=g)
    f = m.reshape(1, 10, 35)
    r = head._R0                                     # [4, 10]
    p = r @ f[0]                                     # [4, 35]
    p = p - p.mean(-1, keepdim=True)
    cov = (p @ p.t()) / 35.0
    ti = head._triu0
    want = cov[ti[0], ti[1]]
    got = head.pool(m, 0)[0, 20:]                    # after mu(10)+sd(10)
    assert torch.allclose(got, want, atol=1e-5), (
        f"max delta {float((got - want).abs().max()):.3e}"
    )
    # rows of R are orthonormal -> the sketch is a rotation, not a
    # subsample: every input channel contributes to every output.
    assert torch.allclose(r @ r.t(), torch.eye(4), atol=1e-5)
    assert (r.abs() > 1e-6).all(), "R is sparse; that is a subsample"


def test_mu_and_sigma_are_first_order_and_full_width():
    head = LaddPixelStatHead([10], proj_dim=4, hidden_dim=8)
    g = torch.Generator().manual_seed(3)
    m = torch.randn(2, 10, 5, 7, generator=g)
    s = head.pool(m, 0)
    f = m.reshape(2, 10, 35)
    assert torch.allclose(s[:, :10], f.mean(-1), atol=1e-6)
    assert torch.allclose(
        s[:, 10:20], (f.var(-1, unbiased=False) + head.eps).sqrt(), atol=1e-6)
    assert s.shape[1] == 10 + 10 + 4 * 5 // 2


def test_sigma_survives_a_dead_relu_channel():
    """VGG ReLU channels are often identically zero. ``sqrt(var)`` has
    an infinite derivative at 0, so eps must be INSIDE the sqrt or the
    first backward produces NaN."""
    head = LaddPixelStatHead([4], proj_dim=2, hidden_dim=4)
    m = torch.zeros(1, 4, 5, 5, requires_grad=True)
    head.pool(m, 0).sum().backward()
    assert torch.isfinite(m.grad).all(), "NaN/Inf gradient on a dead channel"


# ===========================================================================
# 5. NO SILENT RANDOM-INIT FALLBACK
# ===========================================================================
def test_random_init_is_never_a_silent_fallback():
    """A missing checkpoint must RAISE. A randomly initialised VGG is a
    null experiment dressed as a real one, and there is no internet to
    fetch one from."""
    with pytest.raises(FileNotFoundError, match="NO INTERNET"):
        _load_vgg16_state_dict("/nonexistent/definitely_not_here.pth")


def test_pretrained_flag_is_reported_not_assumed():
    src = _mk_src(pretrained=False)
    assert src.encoder_pretrained is False, (
        "ladd_pix_vgg_pretrained must read 0.0 on a random encoder so a "
        "null run is visible in wandb"
    )


@_vgg_only
def test_pretrained_weights_actually_load():
    src = build_pixel_feature_source("vgg", type("A", (), {})())
    assert src.encoder_pretrained is True
    assert src.vgg_taps == ["relu1_2", "relu2_2"], (
        "MEASURED SPEC (TEXTURE_BASIS_BENCHMARK.md §3): relu3_3 must NOT "
        "be in the default set -- fusing it drops TS_min 2.96 -> 0.63. "
        f"Got {src.vgg_taps}."
    )
    assert src.encoder.n_loaded_tensors == 2 * 4, (
        "relu1_2/relu2_2 needs convs 0,2,5,7 = 4 convs = 8 tensors; "
        f"loaded {src.encoder.n_loaded_tensors}"
    )
    # conv1_1 must match the ImageNet checkpoint BIT-FOR-BIT. Compared
    # against a stored sha256 rather than by re-loading the 553 MB file:
    # this box has a 4 GB per-user cgroup cap and a second copy of the
    # state dict OOM-kills the test process. The digest is of
    # ``features.0.weight`` in ``vgg16-397923af.pth``; if it changes, the
    # checkpoint changed and this SHOULD fail.
    import hashlib
    d = hashlib.sha256(
        src.encoder.features[0].weight.detach().contiguous()
        .numpy().tobytes()).hexdigest()
    assert d == (
        "03c02e0544ea3b2a9e832ca1d3cba452e3e4a43f13fb64a7d9c59bd024e43f48"
    ), f"conv1_1 is NOT the ImageNet VGG16 tensor (sha256 {d})"
    assert src.variant == "vgg16:relu1_2,relu2_2"
    assert src.tap_strides == [1, 2]
    assert src.pooled_readout.stat_dims == [656, 784]
    assert src.pooled_readout.total_stat_dim == 1440


@_vgg_only
def test_the_deep_semantic_blocks_are_not_even_instantiated():
    """The cutoff is structural, not a runtime skip.

    ``TEXTURE_BASIS_BENCHMARK.md`` §3 measured that fusing relu3_3 drops
    TS_min from 2.96 to 0.63 -- below the 1.0 pass mark, i.e. a crop
    shift would move the statistic MORE than a real texture change. So
    the deeper convs must not merely be unused, they must not exist."""
    src = build_pixel_feature_source("vgg", type("A", (), {})())
    widths = {int(m.out_channels) for m in src.encoder.features
              if hasattr(m, "out_channels")}
    assert widths == {64, 128}, (
        f"got widths {sorted(widths)}; 256 means relu3_3 is back and the "
        f"fused TS_min falls from 2.96 to 0.63"
    )
    n = sum(p.numel() for p in src.encoder.parameters())
    assert n == 260_160, f"trunk is {n} params, expected 260,160"


def test_unknown_vgg_tap_is_refused():
    with pytest.raises(ValueError, match="unknown VGG16 tap"):
        _mk_src(layers=("relu9_9",))


def test_vgg_layer_list_accepts_a_dotlist_string():
    """``key=a,b,c`` through OmegaConf.from_dotlist arrives as a STRING."""
    class A:
        ladd_pixel_vgg_layers = "relu1_1,relu2_1"
        ladd_pixel_vgg_pretrained = False
        ladd_pixel_vgg_stat_proj_dim = 4
    src = build_pixel_feature_source("vgg", A())
    assert src.vgg_taps == ["relu1_1", "relu2_1"]


def test_encoder_lr_scale_protects_the_pretrained_vgg_basis():
    assert default_encoder_lr_scale("vgg", type("A", (), {})()) == 0.1
    assert default_encoder_lr_scale("pixgan", type("A", (), {})()) == 1.0


# ===========================================================================
# 6. DDP REACHABILITY -- every param gets gradient
# ===========================================================================
def test_every_trainable_param_receives_gradient():
    """``pixdirect_online`` died at the first D-update because ONE
    parameter received no gradient. Assert the whole set, encoder and
    head, so the reducer cannot abort."""
    d, src = _mk_disc(pretrained=False, encoder_trainable=True)
    out = d(_lat(b=2), *_t_pe(2, 3))
    out.sum().backward()
    missing = [n for n, p in d.named_parameters()
               if p.requires_grad and p.grad is None]
    assert not missing, f"{len(missing)} params got no gradient: {missing[:6]}"
    assert sum(1 for _ in d.parameters()) > 0
    assert src.frozen_unreachable_params == [], (
        "the VGG trunk is truncated after the deepest tap, so nothing "
        "should be unreachable"
    )


def test_gradient_reaches_the_input_latent_through_decode_and_pooling():
    """The orderless pooling must not sever the generator's route."""
    d, _ = _mk_disc(pretrained=False)
    x = _lat(b=2).requires_grad_(True)
    d(x, *_t_pe(2, 3)).sum().backward()
    assert x.grad is not None
    assert float(x.grad.abs().sum()) > 0.0, (
        "ZERO gradient to the latent -- the pooled path is decorative"
    )


def test_head_params_are_in_disc_parameters_and_not_in_the_encoder_group():
    """The head must train at the FULL disc lr (it is the adapter), and
    the encoder at ``lr_scale`` -- the trainer splits on
    ``pixel_source.encoder.parameters()``."""
    d, src = _mk_disc(pretrained=False)
    enc = {id(p) for p in src.encoder.parameters()}
    head = {id(p) for p in src.pooled_readout.parameters()}
    allp = {id(p) for p in d.parameters()}
    assert head and head <= allp
    assert not (head & enc)
    assert enc <= allp


def test_the_random_projection_is_a_buffer_not_a_parameter():
    """A LEARNED projection would let D choose its own subspace, and a
    D free to choose a subspace can drift back to a positional one."""
    head = LaddPixelStatHead([16], proj_dim=8, hidden_dim=16)
    names = {n for n, _ in head.named_parameters()}
    assert not any(n.startswith("_R") for n in names)
    assert head._R0.requires_grad is False
    # seeded => reproducible across processes with no broadcast
    h2 = LaddPixelStatHead([16], proj_dim=8, hidden_dim=16)
    assert torch.equal(head._R0, h2._R0)


def test_building_the_head_consumes_no_global_rng():
    torch.manual_seed(99)
    before = torch.get_rng_state()
    LaddPixelStatHead([16, 24], proj_dim=8, hidden_dim=16)
    after = torch.get_rng_state()
    # nn.Linear/LayerNorm init DOES draw; the projections must not add
    # to it beyond that, so compare against a head built with a
    # different projection seed -- same RNG consumption either way.
    torch.manual_seed(99)
    LaddPixelStatHead([16, 24], proj_dim=8, hidden_dim=16, seed=777)
    assert torch.equal(after, torch.get_rng_state()), (
        "the fixed projection consumed global RNG; it would desynchronise "
        "ranks and perturb every other path's byte-identity"
    )


# ===========================================================================
# 7. TWO-SIDED SENSITIVITY -- "too smooth is as wrong as too sharp"
# ===========================================================================
def _gauss(x, s):
    k = int(2 * round(3 * s) + 1)
    r = k // 2
    t = torch.arange(k).float() - r
    g = torch.exp(-t ** 2 / (2 * s * s))
    g = g / g.sum()
    xp = F.pad(x[None], (r, r, r, r), mode="reflect")
    xp = F.conv2d(xp, g.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3)
    return F.conv2d(xp, g.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3)[0]


def _pink(h, w, seed, beta=1.0):
    """1/f texture with NATURAL-IMAGE spectral falloff, built in the
    Fourier domain so it tiles exactly (``torch.roll`` on it is a true
    translation with no wrap seam)."""
    g = torch.Generator().manual_seed(seed)
    fy = torch.fft.fftfreq(h)[:, None]
    fx = torch.fft.fftfreq(w)[None, :]
    r = (fy ** 2 + fx ** 2).sqrt()
    r[0, 0] = 1.0
    ph = torch.rand(3, h, w, generator=g) * 2 * 3.141592653589793
    x = torch.fft.ifft2(r.pow(-beta)[None] * torch.exp(1j * ph)).real
    return ((x - x.mean()) / x.std() * 0.35).clamp(-1, 1)


@_vgg_only
def test_statistic_is_two_sided_blur_AND_grain_both_move_it():
    """The standing rule: undershooting (too smooth) is as wrong as
    overshooting. If the statistic were monotone in only one direction
    a GAN driven by it would run away in the other.

    n = 3 texture seeds; the assertion is per-seed, so it is 3/3.
    """
    src = build_pixel_feature_source("vgg", type("A", (), {})())
    src.eval()
    head = src.pooled_readout

    def stat(x):
        with torch.no_grad():
            return torch.cat(head.stats(src(x[None], epoch=0)), dim=1)[0]

    for seed in (11, 12, 13):
        tex = _pink(96, 128, seed)
        b = stat(tex)
        sc = b.abs().clamp_min(1e-3)
        d = lambda y: float(((stat(y) - b) / sc).pow(2).mean().sqrt())
        gg = torch.Generator().manual_seed(seed)
        d_blur = d(_gauss(tex, 1.2))
        d_grain = d((tex + 0.05 * torch.randn(
            tex.shape, generator=gg)).clamp(-1, 1))
        assert d_blur > 0.05, f"seed {seed}: blur barely moved ({d_blur:.4f})"
        assert d_grain > 0.05, f"seed {seed}: grain barely moved ({d_grain:.4f})"
        # monotone within the blur arm too
        assert d(_gauss(tex, 2.0)) > d(_gauss(tex, 0.6)), (
            f"seed {seed}: heavier blur is not further from real"
        )


@_vgg_only
def test_pooled_is_less_phase_sensitive_than_dense():
    """MEASURED, and the honest size of the effect is ~2x, NOT invariance.

    The pooled statistic is EXACTLY invariant to a permutation of the
    FEATURE MAP (``test_pooling_is_orderless``). It is NOT exactly
    invariant to a translation of the INPUT, because VGG's own stride-2
    max-pools are shift-VARIANT (Zhang, ICML 2019) -- a 1 px input shift
    produces a genuinely different feature map, not a permutation of
    one. So the crop-phase nuisance survives at roughly HALF the dense
    level rather than vanishing.

    That is a benchmark caveat, not a training bug: real and fake rows
    share one crop origin and one jitter in a single forward
    (``_pixel_features``), so the phase term is common-mode and cancels
    in the RpGAN difference.

    n = 6 shifts x 2 texture seeds = 12 paired comparisons.
    """
    src = build_pixel_feature_source("vgg", type("A", (), {})())
    src.eval()
    head = src.pooled_readout
    wins = 0
    total = 0
    for seed in (21, 22):
        tex = _pink(96, 128, seed)
        with torch.no_grad():
            f0 = src(tex[None], epoch=0)
        s0 = torch.cat(head.stats(f0), dim=1)[0]
        m0 = f0[max(f0)][0]
        ssc = s0.abs().clamp_min(1e-3)
        msc = m0.abs().mean().clamp_min(1e-3)
        for sh in (1, 2, 3, 4, 8, 16):
            t2 = torch.roll(tex, shifts=(sh, sh), dims=(1, 2))
            with torch.no_grad():
                f2 = src(t2[None], epoch=0)
            dp = float((((torch.cat(head.stats(f2), dim=1)[0] - s0)
                         / ssc) ** 2).mean().sqrt())
            dd = float((((f2[max(f2)][0] - m0) / msc) ** 2).mean().sqrt())
            total += 1
            wins += int(dp < dd)
    # MEASURED on this tree: 9/12 paired shifts, and the MEDIAN ratio
    # dense/pooled is ~2x. The threshold is 8/12 because the effect is a
    # factor of two, not an order of magnitude, and asserting more than
    # was measured is how this campaign produced its false conclusions.
    assert wins >= 8, (
        f"the pooled statistic was less phase-sensitive than the dense "
        f"feature map on only {wins}/{total} shifts"
    )


def test_the_two_taps_enter_the_mlp_on_equal_footing():
    """BENCHMARK CAVEAT 1: relu1_2's raw statistics are an order of
    magnitude larger than relu2_2's, so a naive concatenation would let
    relu1_2 dominate. Each tap gets its OWN LayerNorm and its own
    Linear to the SAME hidden width, so the MLP sees equal-width,
    equal-scale slices -- 656 vs 784 raw dims become 128 vs 128."""
    head = LaddPixelStatHead([64, 128], proj_dim=8, hidden_dim=32)
    assert head.stat_dims[0] != head.stat_dims[1]
    outs = [e[-1].out_features for e in head.embeds]
    assert outs == [32, 32], f"unequal embed widths {outs}"
    assert all(isinstance(e[0], torch.nn.LayerNorm) for e in head.embeds)
    # and it works: a tap scaled 100x must not change the OTHER tap's
    # contribution scale after its own LayerNorm
    g = torch.Generator().manual_seed(0)
    a = torch.randn(4, 64, 12, 14, generator=g).abs()
    b = torch.randn(4, 128, 6, 7, generator=g).abs()
    e_small = head.embeds[0](head.pool(a, 0))
    e_big = head.embeds[0](head.pool(a * 100.0, 0))
    # LayerNorm removes the common scale, so the embeddings stay the
    # same order of magnitude rather than differing by 100x (Cov: 10^4x)
    with torch.no_grad():
        r = float(e_big.std() / e_small.std().clamp_min(1e-8))
    assert 0.2 < r < 5.0, (
        f"a 100x rescale of one tap changed its embedding scale by {r:.1f}x; "
        f"LayerNorm is not doing its job and the taps are not balanced"
    )
    del b


# ===========================================================================
# 9. THE SECOND BASIS -- ResNet50 layer1 (TS_min 5.40, best in the study)
# ===========================================================================
_RN50_CKPT_PATH = os.path.expanduser(
    "~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth")
_rn50_only = pytest.mark.skipif(
    not os.path.isfile(_RN50_CKPT_PATH),
    reason="resnet50 weights not in the torch-hub cache")


@_rn50_only
def test_rn50_source_builds_pooled_and_truncated():
    src = build_pixel_feature_source("rn50", type("A", (), {})())
    assert src.pooled_readout is not None
    assert src.rn50_taps == ["layer1"]
    assert src.tap_strides == [4]
    assert src.encoder_pretrained is True
    n = sum(p.numel() for p in src.encoder.parameters())
    assert n == 225_344, (
        f"trunk is {n} params; 25.6 M would mean layer2-4 + fc were kept "
        f"and would sit ungradiented in the DDP reducer"
    )
    assert not hasattr(src.encoder, "fc")
    with torch.no_grad():
        f = src(torch.randn(2, 3, 176, 240).clamp(-1, 1), epoch=0)
        out = src.pooled_readout(f)
    assert tuple(f[0].shape) == (2, 256, 44, 60)
    assert tuple(out.shape) == (2, 1), "rn50 readout must be pooled too"
    assert src.grid == (1, 1)


@_rn50_only
def test_rn50_pooling_is_orderless_too():
    src = build_pixel_feature_source("rn50", type("A", (), {})())
    with torch.no_grad():
        f = src(torch.randn(2, 3, 88, 120).clamp(-1, 1), epoch=0)
        a = src.pooled_readout(f)
        b = src.pooled_readout(_perm_maps(f, 4))
    assert torch.allclose(a, b, atol=1e-4)


def test_rn50_is_in_the_supported_tuple():
    assert "rn50" in SUPPORTED_PIXEL_FEATURE_SOURCES
    assert default_encoder_lr_scale("rn50", type("A", (), {})()) == 0.1
