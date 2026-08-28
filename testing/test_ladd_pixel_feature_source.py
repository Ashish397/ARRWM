"""CPU tests for the LADD PIXEL FEATURE SOURCE.

    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" PYTHONPATH=. \
        python -m pytest testing/test_ladd_pixel_feature_source.py -q

(``OMP_NUM_THREADS=8`` is MANDATORY on this box: ``nproc`` is 144 and the
default thread count makes torch's CPU ops thrash for 30+ minutes.)

WHAT IS COVERED, and why each item is here
==========================================
Ordered by what it would cost to miss. Every one of these is a
silent-no-op class this campaign has actually been bitten by
(``project_silent_failure_taxonomy``: 11+ instances, one mechanism).

1.  **DEFAULT-OFF BYTE-IDENTITY.** Production arms are queued on the tree
    this lands in. With ``ladd_feature_source`` at its historical
    "real"/"fake" values the disc must be bit-for-bit what it was:
    same modules, same parameter shapes, same logits, same RNG state,
    and NO pixel attribute consulted.
2.  **THE FEATURES ACTUALLY COME FROM THE PIXEL ENCODER.** The strong
    form, not "a log key appeared": with a pixel source installed the
    WAN projector is NEVER CALLED (asserted with a projector that
    raises), and perturbing the PIXELS changes the logits.
3.  **GRADIENT REACHES THE INPUT LATENT.** Through the decode callback,
    through the encoder, to the tensor the generator would own. A probe
    that reads exactly 0.0 is the specific defect the directive names.
4.  **THE ENCODER IS TRAINABLE AND ACTUALLY RECEIVES WEIGHT GRADIENT.**
    The researcher asked for DINO to be trainable; "the flag is set" is
    not the same claim as "the weights get gradient".
5.  **REAL AND FAKE ROWS ARE CROPPED IDENTICALLY.** If they were not,
    the disc could separate them on crop POSITION -- a perfect,
    silent, content-free tell.
6.  **THE R1 ESTIMATOR SEES ONE GEOMETRY.** ``D(x)`` and ``D(x+sigma*eps)``
    must share crop origin and phase jitter, or R1 measures the geometry
    change instead of the gradient.
7.  **BLUR-POOL IS A REAL LOW-PASS**, and the un-blurred trunk it
    replaces is measurably more phase-variant.
8.  **THE PRECONDITION GUARDS RAISE**, because each of them would
    otherwise be a wrong result rather than a crash.
"""
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.ladd_disc import (  # noqa: E402
    LADDChannelMixer, LADDDiscriminator, build_ladd_disc,
)
from model.ladd_pixel_features import (  # noqa: E402
    SUPPORTED_PIXEL_FEATURE_SOURCES, BlurPool2d, LaddPixelFeatureSource,
    default_encoder_lr_scale,
)

# The DINO tests need the torch-hub cache. On the compute nodes it is
# pre-populated; skip rather than fail if someone runs this elsewhere.
_DINO_CKPT = os.path.expanduser(
    "~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth")
_HAS_DINO = os.path.exists(_DINO_CKPT)
_dino_only = pytest.mark.skipif(
    not _HAS_DINO, reason="dinov2_vits14 weights not in the torch-hub cache")


# ===========================================================================
# helpers
# ===========================================================================
class _RaisingProjector:
    """A projector that makes "the WAN taps were used" impossible to miss.

    The whole claim of this feature is that the disc stops reading WAN
    transformer features. A counter can be forgotten to be checked; an
    exception cannot.
    """

    def __init__(self):
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        raise AssertionError(
            "WanFeatureProjector was called on a pixel-source disc. The "
            "feature basis did NOT swap."
        )


def _fake_decode(latents, want_grad):
    """Stand-in for the VAE: latent [n,F,C,h,w] -> pixels [n,F,3,8h,8w].

    Differentiable, cheap, and 8x upsampling per axis so the crop
    geometry the disc computes is the real one. ``want_grad=False``
    detaches, exactly as the production callback does.
    """
    n, f, c, h, w = latents.shape
    x = latents if want_grad else latents.detach()
    x = x[:, :, :3] if c >= 3 else x.repeat(1, 1, 3, 1, 1)[:, :, :3]
    x = x.reshape(n * f, 3, h, w)
    x = torch.nn.functional.interpolate(x, scale_factor=8, mode="nearest")
    return torch.tanh(x).reshape(n, f, 3, h * 8, w * 8)


def _mk_disc(source="pixgan", *, base=8, taps=(0, 2), dim_teacher=32,
             projector=None, **src_kw):
    src = LaddPixelFeatureSource(
        source, pixgan_base_channels=base, common_stride=8, **src_kw,
    )
    d = LADDDiscriminator(
        projector=projector if projector is not None else _RaisingProjector(),
        block_indices=list(taps),
        dim_teacher=dim_teacher,
        dim_proj=16,
        use_csm=True,
        pixel_source=src,
    )
    d.pixel_decode_fn = _fake_decode
    d.pixel_cfg = {
        "crop_rows": 8, "crop_cols": 8, "crops_per_row": 1,
        "lat_frames": 2, "frames_per_crop": 2, "border": 2,
        "decode_batch": 4,
    }
    d.eval()
    return d, src


def _lat(b=2, f=3, c=16, h=16, w=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, f, c, h, w, generator=g)


def _t_pe(b, f):
    return (torch.zeros(b, f, dtype=torch.long),
            torch.zeros(b, 4, 8))


# ===========================================================================
# 1. DEFAULT-OFF BYTE-IDENTITY
# ===========================================================================
def test_ccm_int_dim_teacher_builds_the_identical_moduledict():
    """The per-tap-dict widening must not perturb the scalar path."""
    a = LADDChannelMixer([0, 2, 4], 1536, 256)
    b = LADDChannelMixer([0, 2, 4], {0: 1536, 2: 1536, 4: 1536}, 256)
    assert a.dim_teacher == b.dim_teacher == 1536
    assert a.dim_teacher_map == b.dim_teacher_map
    for k in ("0", "2", "4"):
        assert a.proj[k].weight.shape == b.proj[k].weight.shape == (256, 1536)


def test_ccm_per_tap_dims_build_per_tap_shapes():
    m = LADDChannelMixer([0, 1, 2], {0: 64, 1: 128, 2: 256}, 16)
    assert m.proj["0"].weight.shape == (16, 64)
    assert m.proj["1"].weight.shape == (16, 128)
    assert m.proj["2"].weight.shape == (16, 256)


def test_pixel_source_none_leaves_forward_on_the_wan_path():
    """No pixel source => ``pixel_source`` is None, the WAN branch runs,
    and the projector-call counter is the thing that moves."""
    captured = {}

    class _P:
        def __call__(self, **kw):
            captured["called"] = True
            b = kw["x_noisy"].shape[0]
            return {0: torch.zeros(b, 400, 32), 2: torch.zeros(b, 400, 32)}

    d = LADDDiscriminator(
        projector=_P(), block_indices=[0, 2], dim_teacher=32, dim_proj=8,
        use_csm=False,
    )
    d.eval()
    x = _lat(b=2, f=2, c=16, h=8, w=10)
    t, pe = _t_pe(2, 2)
    out = d(x_noisy=x, timestep=t, prompt_embeds=pe)
    assert captured.get("called") is True
    assert d.pixel_source is None
    assert d.pixel_stats["wan_projector_calls"] == 1.0
    assert d.pixel_stats["fwd"] == 0.0
    assert out.shape[0] == 2


def test_default_off_touches_no_rng_and_no_pixel_attribute():
    """Constructing + forwarding a legacy disc must not consume the
    global RNG differently or read a pixel key."""
    class _P:
        def __call__(self, **kw):
            b = kw["x_noisy"].shape[0]
            return {0: torch.zeros(b, 400, 32)}

    torch.manual_seed(1234)
    d = LADDDiscriminator(projector=_P(), block_indices=[0],
                          dim_teacher=32, dim_proj=8, use_csm=False)
    d.eval()
    x = _lat(b=1, f=2, c=16, h=8, w=10)
    t, pe = _t_pe(1, 2)
    before = torch.get_rng_state()
    d(x_noisy=x, timestep=t, prompt_embeds=pe)
    assert torch.equal(before, torch.get_rng_state()), (
        "the legacy forward consumed global RNG that it did not before"
    )


# ===========================================================================
# 2. THE FEATURES COME FROM THE PIXEL ENCODER
# ===========================================================================
def test_pixel_disc_never_calls_the_wan_projector():
    proj = _RaisingProjector()
    d, src = _mk_disc(projector=proj)
    x = _lat()
    t, pe = _t_pe(2, 3)
    out = d(x_noisy=x, timestep=t, prompt_embeds=pe)
    assert proj.calls == 0
    assert d.pixel_stats["wan_projector_calls"] == 0.0
    assert d.pixel_stats["fwd"] == 1.0
    assert src.n_forward == 1
    assert out.shape[0] == 2


def test_logits_respond_to_the_decoded_pixels():
    """A disc whose logits do not move when the picture moves is not
    reading the picture."""
    d, _ = _mk_disc()
    t, pe = _t_pe(2, 3)
    a = d(x_noisy=_lat(seed=0), timestep=t, prompt_embeds=pe)
    b = d(x_noisy=_lat(seed=1), timestep=t, prompt_embeds=pe)
    assert not torch.allclose(a, b), "logits are independent of the input"


def test_block_indices_are_taken_from_the_encoder_not_ladd_feature_blocks():
    """WAN taps [0,2,4,8,29] do not exist on a 3-tap conv trunk; silently
    reusing them would index nothing."""
    d, src = _mk_disc(taps=(0, 2, 4, 8, 29))
    assert d.block_indices == src.tap_indices == [0, 1, 2]
    assert set(d.heads.keys()) == {"0", "1", "2"}


def test_taps_share_one_grid_so_the_csm_can_fuse_them():
    d, src = _mk_disc(taps=(0, 1, 2))
    x = _lat()
    t, pe = _t_pe(2, 3)
    d(x_noisy=x, timestep=t, prompt_embeds=pe)
    feats = src(torch.randn(1, 3, 48, 48).clamp(-1, 1), epoch=0)
    n_tok = {tuple(v.shape[:2]) for v in feats.values()}
    assert len(n_tok) == 1, f"taps disagree on token count: {n_tok}"


# ===========================================================================
# 3. GRADIENT REACHES THE INPUT LATENT
# ===========================================================================
def test_gradient_flows_to_the_latent_through_decode_and_encoder():
    d, _ = _mk_disc()
    x = _lat().requires_grad_(True)
    t, pe = _t_pe(2, 3)
    out = d(x_noisy=x, timestep=t, prompt_embeds=pe)
    out.sum().backward()
    assert x.grad is not None, "gradient was SEVERED at the decode"
    g = float(x.grad.norm())
    assert g > 0.0, (
        "gradient reached the latent but is EXACTLY 0.0 -- the "
        "forgeable zero this feature's probes exist to rule out"
    )


def test_grad_decode_is_used_only_when_the_input_carries_grad():
    d, _ = _mk_disc()
    t, pe = _t_pe(2, 3)
    d(x_noisy=_lat(), timestep=t, prompt_embeds=pe)          # detached
    assert d.pixel_stats["decode_nograd"] == 1.0
    assert d.pixel_stats["decode_grad"] == 0.0
    d(x_noisy=_lat().requires_grad_(True), timestep=t, prompt_embeds=pe)
    assert d.pixel_stats["decode_grad"] == 1.0


def test_no_grad_context_forces_the_nograd_decode():
    d, _ = _mk_disc()
    t, pe = _t_pe(2, 3)
    with torch.no_grad():
        d(x_noisy=_lat().requires_grad_(True), timestep=t, prompt_embeds=pe)
    assert d.pixel_stats["decode_nograd"] == 1.0
    assert d.pixel_stats["decode_grad"] == 0.0


# ===========================================================================
# 4. THE ENCODER IS TRAINABLE AND RECEIVES WEIGHT GRADIENT
# ===========================================================================
def test_trainable_encoder_receives_weight_gradient():
    d, src = _mk_disc(encoder_trainable=True)
    assert src.n_trainable_encoder_params > 0
    x = _lat()
    t, pe = _t_pe(2, 3)
    d(x_noisy=x, timestep=t, prompt_embeds=pe).sum().backward()
    got = [p for p in src.encoder.parameters()
           if p.grad is not None and float(p.grad.abs().sum()) > 0.0]
    assert got, (
        "ladd_pixel_encoder_trainable=true but NO encoder parameter "
        "received gradient -- the flag did not take"
    )


def test_frozen_encoder_receives_none_but_the_heads_still_do():
    d, src = _mk_disc(encoder_trainable=False)
    assert src.n_trainable_encoder_params == 0
    x = _lat()
    t, pe = _t_pe(2, 3)
    d(x_noisy=x, timestep=t, prompt_embeds=pe).sum().backward()
    assert all(p.grad is None for p in src.encoder.parameters())
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in d.heads.parameters()), "heads got nothing either"


def test_encoder_params_are_in_disc_parameters():
    """They must be, or the disc optimizer never steps them and
    ``r3gan_disc.state_dict()`` never saves them."""
    d, src = _mk_disc(encoder_trainable=True)
    ids = {id(p) for p in d.parameters()}
    assert all(id(p) in ids for p in src.encoder.parameters())


def test_encoder_is_pinned_to_eval_even_when_trainable():
    d, src = _mk_disc(encoder_trainable=True)
    d.train()
    assert d.heads.training is True
    assert src.encoder.training is False, (
        "a trainable encoder must still be eval-pinned: dropout / "
        "stochastic depth flipping between the D and G forwards is a "
        "difference the critic can see that is not texture"
    )


def test_encoder_lr_scale_defaults_differ_by_source():
    class _A:
        pass
    a = _A()
    assert default_encoder_lr_scale("pixgan", a) == 1.0
    assert default_encoder_lr_scale("dinov2", a) == 0.1
    a.ladd_pixel_encoder_lr_scale = 1.0
    assert default_encoder_lr_scale("dinov2", a) == 1.0
    a.ladd_pixel_encoder_lr_scale = 0.0
    assert default_encoder_lr_scale("pixgan", a) == 0.0


# ===========================================================================
# 5. REAL AND FAKE ROWS SHARE ONE CROP
# ===========================================================================
def test_all_rows_in_a_forward_share_the_crop_origin():
    """Feed a batch whose rows are IDENTICAL apart from a marker placed
    at one spatial location. If the rows were cropped independently, the
    marker would land inside some crops and outside others and the
    per-row logits would differ in a way the shared-crop path cannot
    produce.

    Constructed the other way round, which is stronger: build a batch of
    two IDENTICAL rows and require identical logits. Independent
    per-row crop draws would make them differ.
    """
    d, _ = _mk_disc()
    one = _lat(b=1, seed=7)
    x = torch.cat([one, one], dim=0)
    t, pe = _t_pe(2, 3)
    out = d(x_noisy=x, timestep=t, prompt_embeds=pe)
    assert torch.allclose(out[0], out[1], atol=1e-5), (
        "two identical rows got different logits -- the crop is being "
        "drawn PER ROW, so real and fake are not spatially aligned"
    )


def test_crop_origin_is_a_function_of_the_epoch_only():
    d, _ = _mk_disc()
    t, pe = _t_pe(2, 3)
    x = _lat(seed=11)
    d.pixel_epoch = 5
    a1 = d(x_noisy=x, timestep=t, prompt_embeds=pe)
    a2 = d(x_noisy=x, timestep=t, prompt_embeds=pe)
    d.pixel_epoch = 6
    b1 = d(x_noisy=x, timestep=t, prompt_embeds=pe)
    assert torch.allclose(a1, a2, atol=1e-6), (
        "two forwards at the same epoch disagree -- the R1 finite "
        "difference would be measuring the geometry change"
    )
    assert not torch.allclose(a1, b1, atol=1e-6), (
        "the crop/jitter never moves with the step, so the encoder's "
        "grid phase is locked -- the artefact this feature removes"
    )


def test_forward_consumes_no_global_rng():
    """Crop + jitter come from private generators. Anything else would
    desynchronise DDP ranks and perturb every other path's seeding."""
    d, _ = _mk_disc()
    t, pe = _t_pe(2, 3)
    torch.manual_seed(99)
    before = torch.get_rng_state()
    d(x_noisy=_lat(), timestep=t, prompt_embeds=pe)
    assert torch.equal(before, torch.get_rng_state())


# ===========================================================================
# 6. JITTER
# ===========================================================================
def test_jitter_is_deterministic_per_epoch_and_moves_between_epochs():
    src = LaddPixelFeatureSource("pixgan", pixgan_base_channels=8,
                                 grid_jitter=8)
    assert src.jitter_for(3) == src.jitter_for(3)
    moved = [src.jitter_for(i) for i in range(24)]
    assert len(set(moved)) > 6, f"jitter barely moves: {set(moved)}"
    assert all(-8 <= a <= 8 and -8 <= b <= 8 for a, b in moved)


def test_jitter_zero_disables_it():
    src = LaddPixelFeatureSource("pixgan", pixgan_base_channels=8,
                                 grid_jitter=0)
    assert all(src.jitter_for(i) == (0, 0) for i in range(5))


# ===========================================================================
# 7. BLUR-POOL
# ===========================================================================
def test_blurpool_kernel_is_a_normalised_binomial_low_pass():
    b = BlurPool2d(1, filt_size=4, stride=2)
    k = b.kernel[0, 0]
    assert pytest.approx(float(k.sum()), abs=1e-6) == 1.0
    row = torch.tensor([1.0, 3.0, 3.0, 1.0])
    expect = (row[:, None] * row[None, :]) / 64.0
    assert torch.allclose(k, expect, atol=1e-6)
    assert sum(p.numel() for p in b.parameters()) == 0, (
        "the low-pass must not be learnable -- a learnable one is a "
        "low-pass the D loss can undo"
    )


def test_blurpool_filt_size_one_is_refused():
    with pytest.raises(ValueError, match="filt_size"):
        BlurPool2d(4, filt_size=1)


def test_blurpool_reduces_phase_variance_of_a_striped_input():
    """THE POINT OF THE WHOLE THING. Present a periodic stripe pattern at
    every phase and ask how much the downsampled response depends on the
    phase. Anti-aliased must be markedly flatter than a bare subsample.
    """
    x = torch.zeros(1, 1, 64, 64)
    x[:, :, ::8, :] = 1.0            # period-8 stripes = the VAE cell grid

    def _resp(fn):
        return torch.tensor([
            float(fn(torch.roll(x, s, dims=2)).mean()) for s in range(8)
        ])

    bp = BlurPool2d(1, filt_size=4, stride=2)
    naive = nn.AvgPool2d(1, stride=2)          # bare subsample
    v_bp = float(_resp(bp).std())
    v_naive = float(_resp(naive).std())
    assert v_naive > 0.0
    assert v_bp < 0.35 * v_naive, (
        f"blur-pool phase variance {v_bp:.4g} is not markedly below the "
        f"bare subsample's {v_naive:.4g}"
    )


def test_blurpool_flag_off_reproduces_the_strided_conv_trunk():
    from model.ladd_pixel_features import PixGANFeatureTrunk
    on = PixGANFeatureTrunk(base_channels=8, blurpool=True)
    off = PixGANFeatureTrunk(base_channels=8, blurpool=False)
    assert on.blurpool_enabled and not off.blurpool_enabled
    x = torch.randn(1, 3, 64, 64)
    assert [tuple(t.shape) for t in on(x)] == [tuple(t.shape)
                                               for t in off(x)]


# ===========================================================================
# 8. BRIGHTNESS-INVARIANT / STATIONARY-WAVELET INPUT
# ===========================================================================
@pytest.mark.parametrize("mode", ["dc", "swt"])
def test_texture_input_filter_is_invariant_to_global_channel_offsets(mode):
    src = LaddPixelFeatureSource(
        "pixgan", pixgan_base_channels=8, input_filter=mode,
        swt_strength=1.0,
    )
    x = torch.randn(2, 3, 24, 28) * 0.1
    offset = torch.tensor([0.12, -0.08, 0.05]).view(1, 3, 1, 1)
    a = src._filter_input(x, midpoint=0.0)
    b = src._filter_input(x + offset, midpoint=0.0)
    assert torch.allclose(a, b, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("mode", ["dc", "swt"])
def test_texture_input_filter_annihilates_spatial_dc_cotangent(mode):
    src = LaddPixelFeatureSource(
        "pixgan", pixgan_base_channels=8, input_filter=mode,
        swt_strength=0.75,
    )
    x = torch.randn(2, 3, 24, 28, requires_grad=True)
    y = src._filter_input(x, midpoint=0.0)
    probe = torch.randn_like(y)
    grad = torch.autograd.grad((y * probe).sum(), x)[0]
    assert float(grad.mean(dim=(-2, -1)).abs().max()) < 2e-6


def test_stationary_wavelet_filter_preserves_shape_and_uses_no_parameters():
    src = LaddPixelFeatureSource(
        "pixgan", pixgan_base_channels=8, input_filter="swt",
    )
    x = torch.randn(2, 3, 25, 29)
    y = src._filter_input(x, midpoint=0.0)
    assert y.shape == x.shape
    assert "_swt_lowpass" not in dict(src.named_parameters())
    assert "input_filter=swt" in src.describe()


# ===========================================================================
# 9. GUARDS
# ===========================================================================
def test_unknown_source_is_refused():
    with pytest.raises(ValueError, match="source must be one of"):
        LaddPixelFeatureSource("resnet50")


def test_unknown_input_filter_is_refused():
    with pytest.raises(ValueError, match="input_filter"):
        LaddPixelFeatureSource(
            "pixgan", pixgan_base_channels=8, input_filter="haar_hh_only",
        )


def test_missing_decode_callback_raises_rather_than_falling_back():
    src = LaddPixelFeatureSource("pixgan", pixgan_base_channels=8)
    d = LADDDiscriminator(projector=_RaisingProjector(), block_indices=[0],
                          dim_teacher=32, dim_proj=8, use_csm=False,
                          pixel_source=src)
    d.eval()
    t, pe = _t_pe(1, 2)
    with pytest.raises(RuntimeError, match="pixel_decode_fn"):
        d(x_noisy=_lat(b=1, f=2), timestep=t, prompt_embeds=pe)


def test_register_readout_is_refused():
    src = LaddPixelFeatureSource("pixgan", pixgan_base_channels=8)
    with pytest.raises(ValueError, match="ladd_readout"):
        LADDDiscriminator(projector=_RaisingProjector(), block_indices=[0],
                          dim_teacher=32, dim_proj=8, pixel_source=src,
                          readout="register")


def test_wavelet_hf_is_refused():
    src = LaddPixelFeatureSource("pixgan", pixgan_base_channels=8)
    with pytest.raises(ValueError, match="wavelet"):
        LADDDiscriminator(projector=_RaisingProjector(), block_indices=[0],
                          dim_teacher=32, dim_proj=8, pixel_source=src,
                          wavelet_hf_enabled=True)


def test_model_side_preconditions():
    """``ActionForcingDMD``'s validator block, exercised without building
    a 1.3B model -- the same discipline as
    ``testing/test_ladd_fake_resolvers.py``."""
    # Read the SHIPPED SOURCE by path rather than importing it: the
    # module initialises CUDA at import and this suite is CPU-only.
    src = open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "model", "dmd_action_forcing.py")).read()
    for needle in (
        'ladd_disc_force_clean',
        "requires ladd_r1_mode='fd'",
        'ladd_wavelet_hf_enabled=true',
        "requires ladd_readout='ladd'",
    ):
        assert needle in src, (
            f"the pixel-source precondition {needle!r} is gone from "
            "ActionForcingDMD -- it was load-bearing"
        )


def test_fake_backbone_trainable_still_refuses_a_pixel_source():
    """``ladd_fake_backbone_trainable`` releases the adversarial freeze on
    the FAKE-SCORE backbone. A pixel-source disc never touches that
    backbone, so the flag would be silently inert."""
    # Same import shim as testing/test_ladd_fake_resolvers.py: the
    # module calls ``torch.cuda.current_device()`` at import time.
    from unittest.mock import patch
    with patch.object(torch.cuda, "current_device", return_value=0):
        from model.dmd_action_forcing import (
            resolve_ladd_fake_backbone_trainable,
        )

    class _A:
        ladd_fake_backbone_trainable = True
        ladd_defer_disc_update = True
        streaming_mode = True

    with pytest.raises(ValueError, match="ladd_feature_source"):
        resolve_ladd_fake_backbone_trainable(_A(), "dinov2")


def test_build_ladd_disc_passes_the_source_through():
    src = LaddPixelFeatureSource("pixgan", pixgan_base_channels=8)

    class _RS(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = nn.ModuleList(
                [nn.Identity() for _ in range(30)])

    d = build_ladd_disc(real_score=_RS(), block_indices=[0, 2, 4],
                        dim_teacher=1536, dim_proj=8, pixel_source=src)
    assert d.pixel_source is src
    assert d.block_indices == [0, 1, 2]


def test_supported_sources_tuple_matches_the_model_validator():
    src = open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "model", "dmd_action_forcing.py")).read()
    # UPDATED 2026-08-26 (VGG / RN50 orderless-statistics sources). The
    # validator used to hard-code its own copy of the tuple, which could
    # drift; it now IMPORTS the shared one, so asserting the import is
    # strictly stronger than asserting a duplicated literal.
    assert "SUPPORTED_PIXEL_FEATURE_SOURCES as _PIXEL_SOURCES" in src, (
        "model/dmd_action_forcing.py no longer imports the shared tuple; "
        "the validator and the builder can now drift apart."
    )
    assert SUPPORTED_PIXEL_FEATURE_SOURCES == (
        "pixgan", "dinov2", "vgg", "rn50")


# ===========================================================================
# 9. SURROGATE TEACHER ENTRY POINT
# ===========================================================================
def test_score_pixels_returns_one_value_per_image_with_input_gradient():
    """``compute_teacher_targets`` differentiates ``value.sum()`` w.r.t.
    the latent crop and asserts ``value.shape[0] == z.shape[0]``."""
    d, _ = _mk_disc()
    px = torch.randn(5, 3, 48, 48).clamp(-1, 1).requires_grad_(True)
    v = d.score_pixels(px)
    assert v.shape == (5,)
    v.sum().backward()
    assert px.grad is not None and float(px.grad.norm()) > 0.0


def test_score_pixels_uses_the_same_modules_the_d_update_trains():
    """If the surrogate distilled from a parallel copy of the critic the
    whole premise would fail. Assert identity, not equality."""
    d, src = _mk_disc()
    before = {n: p.detach().clone() for n, p in d.ccm.named_parameters()}
    px = torch.randn(2, 3, 48, 48).clamp(-1, 1)
    d.score_pixels(px).sum().backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in d.ccm.parameters()), (
        "score_pixels does not touch the CCM the D-update trains"
    )
    assert set(before) == {n for n, _ in d.ccm.named_parameters()}


def test_score_pixels_requires_a_pixel_source():
    class _P:
        def __call__(self, **kw):
            return {}

    d = LADDDiscriminator(projector=_P(), block_indices=[0],
                          dim_teacher=32, dim_proj=8, use_csm=False)
    with pytest.raises(RuntimeError, match="pixel feature source"):
        d.score_pixels(torch.randn(2, 3, 32, 32))


# ===========================================================================
# 10. DINOv2 (cache-gated)
# ===========================================================================
@_dino_only
def test_dinov2_source_shapes_and_uniform_dims():
    src = LaddPixelFeatureSource("dinov2", encoder_trainable=True)
    assert len(src.tap_indices) == 4
    assert len(set(src.tap_dims.values())) == 1, (
        "a plain ViT must give one width across taps, or the CSM's "
        "elementwise fusion is wrong"
    )
    f = src(torch.randn(2, 3, 176, 240).clamp(-1, 1), epoch=0)
    gh, gw = src.grid
    for v in f.values():
        assert v.shape == (2, gh * gw, 384)


@_dino_only
def test_dinov2_native_scale_is_not_a_2x_magnification():
    """``PretrainedPixelDisc._prep`` pads to square and resizes to 518 --
    a 2.2x magnification of a 240 px crop, which rescales the very
    texture being measured. This path must stay near native."""
    src = LaddPixelFeatureSource("dinov2")
    nh, nw = src._size_for(176, 240)
    assert 0.9 < nh / 176 < 1.1 and 0.9 < nw / 240 < 1.1, (
        f"176x240 -> {nh}x{nw} is a texture rescale, not a rounding"
    )
    assert nh % 14 == 0 and nw % 14 == 0


@_dino_only
def test_dinov2_ships_trainable_here_unlike_pretrained_pixel_disc():
    """The researcher's directive, asserted against the module that does
    the opposite."""
    trainable = LaddPixelFeatureSource("dinov2", encoder_trainable=True)
    assert trainable.n_trainable_encoder_params > 1e6
    frozen = LaddPixelFeatureSource("dinov2", encoder_trainable=False)
    assert frozen.n_trainable_encoder_params == 0


# ===========================================================================
# 11. TELEMETRY ORDERING — the defect that made the first GPU smoke look dead
# ===========================================================================
# The first pixel arm booted clean and reported ladd_pix_src_forwards=0,
# disc_forwards=0, images=0, grid 0x0, teacher_calls=0 — a build that
# looked completely inert. It was not (necessarily) inert: with
# ``ladd_defer_disc_update=true`` the D-update closure is STASHED inside
# ``_compute_ladd_losses`` (~:19460) and executed at the deferred flush
# site (~:20240). ``_ladd_pixel_logs()`` was called only at the END of
# ``_compute_ladd_losses`` — i.e. ~780 lines and one deferred dispatch
# BEFORE the work it counts. On an arm whose ONLY disc forward is the
# D-update (the LADD G-term is off via ladd_disc_loss_weight=0.0), every
# counter therefore lagged one logged GAN step, and the probe's last
# logged step was 21 against gan_disc_start_step=20.
#
# These tests pin the fix in the SHIPPED SOURCE, because the defect is a
# property of statement ORDER that no unit-level mock can reproduce.
def _trainer_source():
    return open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "trainer", "causal_action_forcing_train.py")).read()


def test_pixel_logs_are_republished_after_the_deferred_flush():
    src = _trainer_source()
    assert src.count("_pix_after = self._ladd_pixel_logs()") == 1, (
        "the post-flush re-publish of the pixel counters is gone; every "
        "ladd_pix_* counter would silently lag one logged GAN step again"
    )
    assert 'out["train/ladd_pix_logs_after_flush"] = 1.0' in src, (
        "the provenance key is gone — a reader can no longer tell the "
        "post-flush publish from the stale pre-flush one"
    )
    # ORDER: the re-publish must come AFTER the flush, not before.
    flush = src.index("self._mem_step_snapshot(\"6a_after_deferred_disc\")")
    republish = src.index("_pix_after = self._ladd_pixel_logs()")
    assert republish > flush, (
        "the pixel-counter re-publish sits BEFORE the deferred flush — "
        "which is the original defect, reintroduced"
    )
    # ...and after the surrogate distillation, which is what produces
    # ladd_pix_teacher_calls.
    distill = src.rindex("self._maybe_run_surrogate_distillation(")
    assert republish > distill, (
        "the re-publish precedes _maybe_run_surrogate_distillation, so "
        "ladd_pix_teacher_calls would still lag a step"
    )


def test_deferred_flush_proof_keys_are_not_gated_on_the_fake_override():
    """Every OTHER deferred-update key lives inside
    ``if _ov_proof is not None:`` — the fake-score feature-backbone
    override, which a PIXEL arm never installs. So on this config all of
    them were suppressed and the run had no evidence of its D-update in
    either direction."""
    src = _trainer_source()
    for key in ("ladd_disc_deferred_flush_events",
                "ladd_disc_deferred_flush_closures"):
        assert f'out["{key}"]' in src, f"{key} proof key is gone"
    # The publish must be at method-body indentation (8 spaces), i.e. NOT
    # nested inside the `if _ov_proof is not None:` guard (12 spaces).
    for key in ("ladd_disc_deferred_flush_events",
                "ladd_disc_deferred_flush_closures"):
        line = next(l for l in src.splitlines() if f'out["{key}"]' in l)
        indent = len(line) - len(line.lstrip())
        assert indent == 8, (
            f"{key} is indented {indent} — it has been nested back inside "
            "a guard and is suppressed on exactly the arms that need it"
        )


def test_every_probe_unavailable_path_publishes_a_reason():
    """The first smoke published
    ``surrogate_grad_probe_unavailable=1`` with NO reason key, so a
    base-unreachable read (normal in the warmup window, and STRUCTURAL on
    an arm whose surrogate fake is a different sub-graph from DMD's) was
    indistinguishable from a real severing. An alarm that cannot say what
    tripped it is the forgeable-zero failure wearing a different hat."""
    src = _trainer_source()
    for reason in ("surrogate_grad_probe_reason_no_tensor",
                   "surrogate_grad_probe_reason_base_unreachable",
                   "surrogate_grad_severed_from_chunk"):
        assert reason in src, f"reason key {reason} is gone"


def test_parameter_space_calibration_probe_exists_and_is_whole_set():
    """On these arms the surrogate's fake is ``flash_dmd_gan_x0`` — a
    DIFFERENT generator sub-graph from the tensor DMD scores — so no
    single tensor lies on both routes and the same-quantity chunk ratio
    is undefined BY CONSTRUCTION. The parameters are where the two
    gradients actually add, and a whole-set norm cannot be off-path."""
    src = _trainer_source()
    assert "def _param_grad_ratio(" in src
    assert 'out[f"{K}_grad_ratio_unweighted"]' in src
    # Whole set, not a positional pick — the artefact that made the old
    # probe read an unfalsifiable 0.0.
    assert ("params = [p for p in self.model.generator.parameters()\n"
            "                      if p.requires_grad]") in src, (
        "the parameter probe no longer takes the WHOLE trainable set; a "
        "positional pick reintroduces the off-path-zero artefact"
    )
    assert 'out[f"{K}_grad_severed"]' in src
    # The SAME instrument must serve the DIRECT route, or the
    # zero-gradient failure would only be catchable on the
    # abandoned surrogate path.
    assert 'prefix="ladd_g"' in src, (
        "the direct decode-gradient route is not wired to the "
        "parameter probe that caught the surrogate zero")


# ===========================================================================
# DDP reachability -- the 2026-08-26 blocker
# ===========================================================================
# ``pixdirect_online`` / ``pixdino_online`` died on EVERY rank at the first
# discriminator update with
#
#     RuntimeError: Expected to have finished reduction in the prior
#     iteration ... Parameter indices which did not receive grad for
#     rank 0: 2
#
# DDP registers exactly the ``requires_grad`` params of the wrapped module
# in ``parameters()`` order and aborts if one produces no gradient.
# ``LADDDiscriminator`` registers ``pixel_source`` first, so index 2 is
# DINOv2's third declared parameter: ``mask_token`` -- which
# ``prepare_tokens_with_masks`` only reads under ``if masks is not None``
# and this call site never passes masks.
#
# NOTE these tests also pin the FROZEN arm's invariance, because the frozen
# arm was already running on two holders when the fix landed and had to stay
# comparable.
@_dino_only
def test_trainable_dinov2_has_no_ddp_unreachable_params():
    """The condition DDP itself checks: every ``requires_grad`` param must
    come back with a gradient after one backward."""
    src = LaddPixelFeatureSource("dinov2", encoder_trainable=True)
    f = src(torch.randn(1, 3, 176, 240).clamp(-1, 1), epoch=0)
    sum(v.float().pow(2).mean() for v in f.values()).backward()
    unreachable = [
        n for n, p in src.encoder.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert unreachable == [], (
        "these params would abort the DDP reducer at the first disc "
        f"update: {unreachable}"
    )


@_dino_only
def test_the_frozen_mask_token_is_the_one_that_was_breaking_ddp():
    """PROOF OF FIRE, not a reading of the patch. Asserts the discovery
    actually happened AND that it found the parameter the live traceback
    named -- so a rename upstream fails this test instead of silently
    reinstating the crash."""
    src = LaddPixelFeatureSource("dinov2", encoder_trainable=True)
    assert src.frozen_unreachable_params == ["mask_token"], (
        f"expected mask_token to be discovered unreachable; got "
        f"{src.frozen_unreachable_params}"
    )
    # ... and that it is genuinely out of the reducer's index space.
    reducer = [n for n, p in src.encoder.named_parameters() if p.requires_grad]
    assert "mask_token" not in reducer
    assert reducer[2] == "patch_embed.proj.weight", (
        "index 2 is what the live traceback named; it must no longer be "
        f"mask_token, got {reducer[2]}"
    )


@_dino_only
def test_the_frozen_arm_skips_the_probe_entirely():
    """Frozen-arm invariance. With the encoder already frozen nothing can
    enter the reducer, so the probe must not run at all -- no extra
    forward, no state touched."""
    src = LaddPixelFeatureSource("dinov2", encoder_trainable=False)
    assert src.frozen_unreachable_params == []
    assert src.n_trainable_encoder_params == 0
    assert src.n_forward == 0, (
        "the build-time probe must not touch the real forward counter"
    )


@_dino_only
def test_the_fix_does_not_change_the_features_in_either_arm():
    """The frozen arm was ALREADY RUNNING when this landed. Both arms must
    produce bit-identical features to the pre-fix code -- ``mask_token`` is
    never read, and at ``n_taps=4`` the truncation drops nothing."""
    g = torch.Generator().manual_seed(20260826)
    px = torch.rand(2, 3, 176, 240, generator=g) * 2 - 1
    outs = []
    for trainable in (False, True):
        src = LaddPixelFeatureSource("dinov2", encoder_trainable=trainable)
        with torch.no_grad():
            outs.append(src(px, epoch=7))
    for k in outs[0]:
        assert torch.equal(outs[0][k], outs[1][k]), (
            f"tap {k} differs between the frozen and online arms; the "
            "encoder's trainability must not change its OUTPUT"
        )


@_dino_only
def test_truncation_is_inert_at_the_shipped_tap_count_and_says_so():
    """The onboarding's proposed fix was 'truncate after the last tapped
    block'. At ``dino_n_taps=4`` the evenly-spaced taps are [2, 5, 8, 11]
    on a 12-block ViT-S -- block 11 IS tapped, so truncation drops nothing
    and CANNOT have been the fix. Pinned so nobody re-derives it."""
    src = LaddPixelFeatureSource("dinov2", dino_n_taps=4)
    assert src.dino_taps == [2, 5, 8, 11]
    assert src.n_encoder_blocks == 12
    assert src.n_encoder_blocks_dropped == 0


@_dino_only
def test_truncation_does_fire_on_a_shallow_explicit_tap_list():
    """...but it is not dead code: a shallow explicit
    ``ladd_pixel_dino_layers`` is exactly the case the onboarding had in
    mind, and there it drops 8 of 12 blocks."""
    src = LaddPixelFeatureSource(
        "dinov2", dino_layers=[0, 1, 2, 3], encoder_trainable=True)
    assert src.dino_taps == [0, 1, 2, 3]
    assert src.n_encoder_blocks == 4
    assert src.n_encoder_blocks_dropped == 8
    f = src(torch.randn(1, 3, 176, 240).clamp(-1, 1), epoch=0)
    sum(v.float().pow(2).mean() for v in f.values()).backward()
    unreachable = [
        n for n, p in src.encoder.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert unreachable == [], (
        f"shallow taps must also leave a clean reducer; got {unreachable}"
    )
