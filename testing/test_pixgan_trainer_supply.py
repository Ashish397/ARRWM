"""WP-PIXGAN (B1) / T3-B — real+fake supply and the D-update loop.

Scope of the chunk under test (nothing else — T3-A's gate/construction/
optimizer/resume is ``testing/test_pixgan_trainer_wiring.py``, and the
G-term / §7 telemetry / config block are T3-C's):

  * ``_pix_take_crops_with_origins`` — the COMMITTED cross-package crop
    helper, ``(crops, ys, xs, bands)`` with origins in LATENT rows/cols
    (docs/WP_PIXGAN.md §4 commitment 2; WP-SURROGATE codes against it);
  * ``_pix_decode_crops_grad`` / ``_pix_decode_crops_nograd`` — two
    DIFFERENTLY NAMED helpers, never one with a flag (commitment 1);
  * ``_pix_select_fake_latents`` — flash (default) vs the A23 ladder
    endpoint, MASK-SELECTED, fail-loud when the mask is absent/all-False;
  * ``_pix_pool_fill`` / ``_pix_draw_reals`` — A20 (one independently
    SOURCED real per fake, repeat_frac 0), A21 (measured support off
    distinct identities, continuous refresh), A22 (reserved rides absent),
    A24 (coarse thirds, same-band pairing, mismatch counter 0);
  * ``_maybe_run_pixel_texture_d_updates`` — the D-loop, R1 on every
    update (``pix_r1_rate == 1.00``), rank-0 broadcast draws.

Every "must be absent / must be zero" guard here has a COMPANION PLANTED-
VIOLATION test that proves the guard actually fires. A guard nobody has
seen trip is not evidence.

CPU-ONLY: the VAE and the zarr loader are stubbed. Run WITH the thread
caps — nproc=144 here and an uncapped run looks hung for >30 min:

    cd /scratch/u6ex/as1748.u6ex/ARRWM
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONPATH=. \
      /scratch/u6ex/as1748.u6ex/miniforge3/envs/arrwm/bin/python \
      -m pytest -q testing/test_pixgan_trainer_supply.py
"""
import ast
import contextlib
import inspect
import io
import logging
import os
import shutil
import tempfile
import textwrap
import random
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as _Fnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.pixel_texture_disc import PixelTextureDisc  # noqa: E402
# Reuse T2's comment/docstring stripper (T3-A imports the same one) rather
# than writing a third copy of the same idea.
from testing.test_pixel_texture_disc import (  # noqa: E402
    code_identifiers, code_only,
)


def method_code(fn):
    """``code_only`` over a METHOD's source. ``inspect.getsource`` returns
    it still indented at class level, which ``ast.parse`` rejects, so dedent
    first. Reuses T2's stripper rather than adding a third copy."""
    return code_only(textwrap.dedent(inspect.getsource(fn)))

with patch.object(torch.cuda, "current_device", return_value=0):
    from trainer import causal_action_forcing_train as CAFT

Trainer = CAFT.ActionForcingDMDTrainer

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_AF_SRC_PATH = os.path.join(_ROOT, "trainer", "causal_action_forcing_train.py")


def _af_source():
    with open(_AF_SRC_PATH, encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# Stub trainer: the REAL shipped T3-B methods, bound to a CPU stub ``self``.
# Nothing here re-implements logic under test — only the VAE and the zarr
# loader are replaced.
# ---------------------------------------------------------------------------
LAT_C, LAT_H, LAT_W = 16, 60, 104          # the ride window's latent geometry


def _decode_impl(lat):
    """[B,F,C,h,w] -> [B, 4F, 3, 8h, 8w] in [-1,1]. Differentiable."""
    b, f, c, h, w = lat.shape
    x = lat[:, :, :3] if c >= 3 else lat[:, :, :1].expand(b, f, 3, h, w)
    x = x.reshape(b * f, 3, h, w)
    x = _Fnn.interpolate(x, scale_factor=8, mode="nearest")
    x = x.reshape(b, f, 3, h * 8, w * 8)
    x = x.repeat_interleave(4, dim=1)
    return torch.tanh(x)


class StubTrainer:
    """Carries the real T3-B methods; everything else is a stub."""

    # --- the code under test, pulled off the shipped class ---------------
    # DERIVED, never hand-maintained. A fixed list is two things that must
    # agree with nothing enforcing it: adding one trainer method and
    # forgetting to list it here silently broke 19 tests once already
    # (AttributeError, not a real failure). Everything WP-PIXGAN owns is
    # named ``_pix_*``, ``PIX_*`` or ``*pixel_texture*``, so bind by that
    # rule and let new methods arrive for free.
    for _n in dir(Trainer):
        if (_n.startswith("_pix_") or _n.startswith("PIX_")
                or "pixel_texture" in _n):
            locals()[_n] = getattr(Trainer, _n)
    del _n

    def __init__(self, **cfg):
        self.device = torch.device("cpu")
        self.world_size = 1
        self.is_main_process = True
        self.step = 0
        self.gan_updates_per_step = 5
        self.gan_disc_start_step = 0
        self.gan_warmup_steps = 500
        self.n_decodes = 0
        self.gan_pixel_texture_enabled = True
        # The A23 seam receiver: an ActionForcingTrainingPipeline stands
        # here in production. ``self.model`` is a DIFFERENT object.
        self.pipeline = SimpleNamespace()
        base = dict(
            pix_crop_lat=(24, 32), pix_crops_per_step=2,
            pix_frames_per_crop=2, pix_band_count=3, pix_reals_per_fake=1,
            # NOT 1.0. That is the module's INERT default (R1/d_loss ~
            # 5e-6 on the shipped critic) and _pix_resolve_r1_gamma refuses
            # it -- a stub that carried it would be a stub proving the
            # trainer runs in a configuration production must never launch.
            pix_loss_form="nsgan", pix_r1_gamma=200.0, pix_r1_sigma=0.01,
            pix_r1_every_n=1, pix_r1_num_samples=None,
            pix_decode_border_trim=8, pix_decode_batch=4,
            pix_lat_frames_per_crop=2, pix_gan_updates_per_step=2,
            # Smaller than spec by default purely for CPU runtime; the
            # directive shape (8 crops x 3 frames = 24-and-24, researcher
            # 2026-08-24) is exercised explicitly in
            # test_a20_directive_shape_is_twentyfour_and_twentyfour.
            pix_real_pool_windows=64, pix_real_pool_refresh=4,
            # Tiny pool + tiny warm-up so the A21 warm-up regime is short
            # enough to leave in a CPU test; production is 4096/64.
            pix_real_pool_warm_updates=8, pix_real_reuse_horizon=4,
            pix_finish_grad_enabled=False, pix_seed=7,
        )
        base.update(cfg)
        self.config = SimpleNamespace(**base)
        self.pixel_texture_disc = PixelTextureDisc().to(torch.float32)
        self.pixel_texture_disc_ddp = None
        self.pix_optimizer = torch.optim.Adam(
            list(self.pixel_texture_disc.parameters()), lr=1e-5,
            betas=(0.0, 0.9),
        )
        # Rides: 6 "train" rides, 2 reserved. ``_holdout_names`` returns the
        # cached set, so no filesystem glob is needed.
        self.dataset = SimpleNamespace(
            _rides=[(f"/fake/root/train_{i}.zarr", 0) for i in range(6)]
        )
        self._dhp_holdout_names = {"held_0.zarr", "held_1.zarr"}
        self._dhp_holdout_paths = []

    # --- stubbed VAE ------------------------------------------------------
    def _vae_decode_grad(self, latent, use_checkpoint=True):
        self.n_decodes += 1
        return _decode_impl(latent.to(torch.float32))

    def _vae_decode_nograd(self, latent):
        self.n_decodes += 1
        with torch.no_grad():
            return _decode_impl(latent.detach().to(torch.float32))


def _fake_load_latent_chunk(path, lo, hi):
    """Deterministic pseudo-GT latents keyed on (ride, start)."""
    g = torch.Generator().manual_seed((hash((str(path), int(lo))) % (2 ** 31)))
    return torch.randn(int(hi) - int(lo), LAT_C, LAT_H, LAT_W, generator=g)


@pytest.fixture
def loader_patch():
    from utils import zarr_dataset as _zd
    from model import disc_holdout_probe as _dhp
    with patch.object(_zd.ZarrRideDataset, "load_latent_chunk",
                      staticmethod(_fake_load_latent_chunk)), \
            patch.object(_dhp, "_latent_len", lambda p: 400):
        yield


def _info_flash(n_lat=6):
    return {"flash_dmd_gan_x0": torch.randn(1, n_lat, LAT_C, LAT_H, LAT_W)}


def _info_ladder(n_lat=7, mask=None):
    z = torch.randn(1, n_lat, LAT_C, LAT_H, LAT_W, requires_grad=True)
    if mask is None:
        mask = [True] * 6 + [False]        # the measured 14.3 % trailing skip
    return {
        "finish_denoised_chunk_grad": z,
        "finish_denoised_chunk_grad_mask": torch.tensor(mask),
    }


# ===========================================================================
# 1. Byte-identical when the gate is off
# ===========================================================================
def test_gate_off_consumes_no_rng_and_emits_no_key(loader_patch):
    t = StubTrainer()
    t.gan_pixel_texture_enabled = False
    info = _info_flash()               # built BEFORE the measurement window
    torch.manual_seed(0)
    random.seed(0)
    before_t = torch.random.get_rng_state().clone()
    before_p = random.getstate()
    out = {}
    t._maybe_run_pixel_texture_d_updates(info, out, current_step=0)
    assert out == {}, out
    assert torch.equal(torch.random.get_rng_state(), before_t)
    assert random.getstate() == before_p
    assert t.n_decodes == 0


def test_gate_on_also_leaves_global_rng_untouched(loader_patch):
    """Every draw comes from a PRIVATE generator, so even the ON path does
    not re-order the global stream — which is what makes the OFF path
    byte-identical rather than merely 'no new keys'."""
    t = StubTrainer()
    info = _info_flash()
    torch.manual_seed(0)
    before = torch.random.get_rng_state().clone()
    out = {}
    t._maybe_run_pixel_texture_d_updates(info, out, current_step=0)
    assert out["train/pix_d_updates"] == 2.0
    assert torch.equal(torch.random.get_rng_state(), before)


def test_call_site_is_gate_guarded():
    src = _af_source()
    i = src.index("self._maybe_run_pixel_texture_d_updates(")
    ctx = src[i - 900:i]
    assert 'if bool(getattr(self, "gan_pixel_texture_enabled", False)):' in ctx


def test_warmup_start_step_defers_d_updates(loader_patch):
    t = StubTrainer()
    t.gan_disc_start_step = 50
    out = {}
    t._maybe_run_pixel_texture_d_updates(_info_flash(), out, current_step=10)
    assert out["train/pix_d_updates"] == 0.0
    assert out["train/pix_d_warmup_skipped"] == 1.0
    assert t.n_decodes == 0


def test_updates_per_step_follows_gan_updates_per_step(loader_patch):
    t = StubTrainer()
    del t.config.pix_gan_updates_per_step
    t.gan_updates_per_step = 3
    out = {}
    t._maybe_run_pixel_texture_d_updates(_info_flash(), out, current_step=0)
    assert out["train/pix_d_updates"] == 3.0


def test_updates_per_step_null_follows_and_echoes_the_source(loader_patch):
    """RESEARCHER EXPOSURE DIRECTIVE (2026-08-24): the shipped config now
    carries ``pix_gan_updates_per_step: null``, which must FOLLOW
    ``gan_updates_per_step`` (5 in the ganfix arms) -- and the echo must
    report both the derived value AND its source, so a reader of the trace
    can tell a followed 5 from a pinned 5."""
    t = StubTrainer(pix_gan_updates_per_step=None)
    t.gan_updates_per_step = 5
    out = {}
    t._maybe_run_pixel_texture_d_updates(_info_flash(), out, current_step=0)
    assert out["train/pix_d_updates"] == 5.0
    assert out["train/pix_cfg_pix_gan_updates_per_step_derived"] == 5.0
    assert out["train/pix_cfg_pix_gan_updates_per_step_followed"] == 1.0


def test_updates_per_step_pinned_integer_still_wins(loader_patch):
    t = StubTrainer(pix_gan_updates_per_step=3)
    t.gan_updates_per_step = 5
    out = {}
    t._maybe_run_pixel_texture_d_updates(_info_flash(), out, current_step=0)
    assert out["train/pix_d_updates"] == 3.0
    assert out["train/pix_cfg_pix_gan_updates_per_step_derived"] == 3.0
    assert out["train/pix_cfg_pix_gan_updates_per_step_followed"] == 0.0


def test_negative_updates_per_step_is_refused_at_resolution():
    """PLANTED VIOLATION for the new resolver guard, exercising product
    code: a negative count has no meaning and must be refused where every
    other pix_* invariant is -- at resolution, not at the consumer."""
    t = StubTrainer(pix_gan_updates_per_step=-1)
    with pytest.raises(ValueError, match="pix_gan_updates_per_step"):
        t._pix_resolve_cfg()


# ===========================================================================
# 2. The crop helper — the committed cross-package contract
# ===========================================================================
def test_crop_helper_signature_is_the_committed_one():
    sig = inspect.signature(Trainer._pix_take_crops_with_origins)
    p = list(sig.parameters)
    assert p[:2] == ["self", "source"]
    for name in ("n_crops", "crop_rows", "crop_cols", "n_bands", "gen"):
        assert sig.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    for name in ("bands", "offs_y"):
        assert sig.parameters[name].default is None


def test_crop_helper_returns_usable_origins_in_latent_coords():
    t = StubTrainer()
    src = torch.randn(2, 3, LAT_C, LAT_H, LAT_W)
    g = torch.Generator().manual_seed(11)
    crops, ys, xs, bands = t._pix_take_crops_with_origins(
        src, n_crops=6, crop_rows=24, crop_cols=32, n_bands=3, gen=g,
    )
    assert crops.shape == (6, 3, LAT_C, 24, 32)
    assert len(ys) == len(xs) == len(bands) == 6
    # LATENT coords: origins must index the latent grid, not pixels.
    assert all(0 <= y <= LAT_H - 24 for y in ys), ys
    assert all(0 <= x <= LAT_W - 32 for x in xs), xs
    assert all(b in (0, 1, 2) for b in bands), bands
    # Each returned crop is literally source[i, :, :, y:y+24, x:x+32] for
    # SOME source index i — the origins describe the crop that came back.
    for k in range(6):
        assert any(
            torch.equal(crops[k], src[i, :, :, ys[k]:ys[k] + 24,
                                      xs[k]:xs[k] + 32])
            for i in range(src.shape[0])
        )


def test_crop_helper_origins_respect_a24_band_bins():
    t = StubTrainer()
    src = torch.randn(1, 2, LAT_C, LAT_H, LAT_W)
    g = torch.Generator().manual_seed(3)
    _, ys, _, bands = t._pix_take_crops_with_origins(
        src, n_crops=200, crop_rows=24, crop_cols=32, n_bands=3, gen=g,
    )
    max_off = LAT_H - 24                                  # 36 rows
    assert max_off == 36
    for y, b in zip(ys, bands):
        lo = round(b * max_off / 3)
        hi = round((b + 1) * max_off / 3)
        assert lo <= y <= max(hi, lo), (y, b)
    # A24 thirds of a 24-row crop over 60 rows: offsets 0-12 / 12-24 / 24-36.
    assert {(round(b * 36 / 3), round((b + 1) * 36 / 3)) for b in (0, 1, 2)} \
        == {(0, 12), (12, 24), (24, 36)}
    assert set(bands) == {0, 1, 2}


def test_crop_helper_is_grad_transparent():
    t = StubTrainer()
    src = torch.randn(1, 2, LAT_C, LAT_H, LAT_W, requires_grad=True)
    g = torch.Generator().manual_seed(5)
    crops, _, _, _ = t._pix_take_crops_with_origins(
        src, n_crops=2, crop_rows=24, crop_cols=32, n_bands=3, gen=g,
    )
    assert crops.requires_grad
    crops.sum().backward()
    assert src.grad is not None and float(src.grad.abs().sum()) > 0


def test_grad_and_nograd_decode_are_separate_named_helpers():
    """Commitment 1: not one helper with a flag."""
    assert hasattr(Trainer, "_pix_decode_crops_grad")
    assert hasattr(Trainer, "_pix_decode_crops_nograd")
    gsig = inspect.signature(Trainer._pix_decode_crops_grad)
    for bad in ("grad", "use_grad", "no_grad", "requires_grad"):
        assert bad not in gsig.parameters
    # The grad helper must empty_cache BEFORE decoding (step-2 allocator
    # failure is real) and must call _vae_decode_grad, never the probe's
    # no_grad _decode_crops.
    body = method_code(Trainer._pix_decode_crops_grad)
    # empty_cache FIRST, before the decode (the step-2 allocator failure).
    assert body.index("empty_cache") < body.index("_vae_decode_grad")
    assert "_decode_crops(" not in body
    nbody = method_code(Trainer._pix_decode_crops_nograd)
    assert "_decode_crops(" in nbody and "_vae_decode_grad" not in nbody


def test_grad_decode_flows_gradient_to_the_latent_crop():
    t = StubTrainer()
    src = torch.randn(1, 2, LAT_C, LAT_H, LAT_W, requires_grad=True)
    g = torch.Generator().manual_seed(9)
    crops, _, _, _ = t._pix_take_crops_with_origins(
        src, n_crops=2, crop_rows=24, crop_cols=32, n_bands=3, gen=g,
    )
    px = t._pix_decode_crops_grad(
        crops, border=8, n_frames=3, decode_batch=2, gen=g,
    )
    assert px.shape[1:] == (3, 24 * 8 - 16, 32 * 8 - 16) == (3, 176, 240)
    assert px.requires_grad
    px.sum().backward()
    assert src.grad is not None and float(src.grad.abs().sum()) > 0


# ---------------------------------------------------------------------------
# bf16 latent x fp32 VAE — the run-6113341 crash, reproduced on CPU.
#
# The stub VAE above is DELIBERATELY forgiving (it casts its own input), which
# is exactly why it never caught this.  The stubs below are STRICT: an fp32
# parameter module that raises the live error string on a dtype mismatch,
# the way ``F.conv3d`` does, and trainer decode wrappers that are faithful
# copies of the SHIPPED ``_vae_decode_grad`` / ``_vae_decode_nograd`` bodies
# (grad: no input cast at all; nograd: the unconditional ``.to(float32)``).
# Pre-fix the grad helper raises here and the two sides disagree on dtype.
# ---------------------------------------------------------------------------
class _StrictVae(torch.nn.Module):
    """VAE whose weights are fp32 and which REFUSES a mismatched input.

    Mirrors ``wan/modules/vae.py``'s ``self.conv2(z)``: a real Conv3d with
    fp32 weight+bias, raising the verbatim runtime error rather than
    silently upcasting.  Records every dtype it is handed so a test can
    compare the real and the fake side.
    """

    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.conv = torch.nn.Conv3d(3, 3, 1).to(dtype)
        self.seen = []

    def decode_to_pixel(self, latent, use_cache=False, seed_first=False):
        w = self.conv.weight
        self.seen.append(latent.dtype)
        if latent.dtype != w.dtype:
            raise RuntimeError(
                f"Input type ({latent.dtype}) and bias type ({w.dtype}) "
                "should be the same"
            )
        # Keep the graph: scale by the (fp32) weight sum so the decode is a
        # genuine differentiable function of the latent, then reuse the
        # shared shape contract.
        return _decode_impl(latent) * w.sum().to(latent.dtype)


class StrictVaeTrainer(StubTrainer):
    """StubTrainer + a strict VAE + FAITHFUL copies of the shipped wrappers."""

    def __init__(self, vae_dtype=torch.float32, **cfg):
        super().__init__(**cfg)
        self.model = SimpleNamespace(vae=_StrictVae(vae_dtype))

    # Verbatim shape of trainer/causal_action_forcing_train.py:4525 (the
    # use_checkpoint branch is irrelevant on CPU): NO input cast.
    def _vae_decode_grad(self, latent, use_checkpoint=True):
        self.n_decodes += 1
        return self.model.vae.decode_to_pixel(latent, seed_first=True)

    # Verbatim shape of :4675: detach + unconditional fp32.
    def _vae_decode_nograd(self, latent):
        self.n_decodes += 1
        with torch.no_grad():
            return self.model.vae.decode_to_pixel(
                latent.detach().to(torch.float32), seed_first=True,
            )


def _bf16_crops(requires_grad=False):
    src = torch.randn(
        1, 2, LAT_C, LAT_H, LAT_W, requires_grad=requires_grad,
    )
    t = StubTrainer()
    g = torch.Generator().manual_seed(9)
    crops, _, _, _ = t._pix_take_crops_with_origins(
        src.to(torch.bfloat16) if not requires_grad else src,
        n_crops=2, crop_rows=24, crop_cols=32, n_bands=3, gen=g,
    )
    return src, crops


def test_vae_dtype_is_derived_from_the_vae_not_hardcoded():
    """(a) The cast target comes from the VAE's own parameters."""
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        t = StrictVaeTrainer(vae_dtype=dt)
        assert t._pix_vae_dtype() == dt
    # A hard-coded float32 would pass the first case only.
    body = method_code(Trainer._pix_vae_dtype)
    assert "parameters()" in body
    # Both decode helpers must derive it the SAME way -- neither may carry
    # its own private notion of the VAE's precision.
    for fn in (Trainer._pix_decode_crops_grad,
               Trainer._pix_decode_crops_nograd):
        assert "_pix_vae_dtype()" in method_code(fn)


def test_bf16_latent_crop_is_accepted_by_the_grad_decode():
    """REGRESSION, run 6113341: bf16 crop + fp32 VAE must not raise.

    Fails pre-fix with the verbatim live error
    ``Input type (c10::BFloat16) and bias type (float) should be the same``.
    """
    t = StrictVaeTrainer()
    _src, crops = _bf16_crops()
    assert crops.dtype == torch.bfloat16
    g = torch.Generator().manual_seed(3)
    px = t._pix_decode_crops_grad(
        crops, border=8, n_frames=3, decode_batch=2, gen=g,
    )
    assert px.shape[1:] == (3, 176, 240)
    assert px.dtype == torch.float32
    assert t.model.vae.seen and all(
        d == torch.float32 for d in t.model.vae.seen
    )


def test_bf16_latent_crop_is_accepted_by_the_nograd_decode():
    """The same regression on the no-grad twin (the D-loop's fake side)."""
    t = StrictVaeTrainer()
    _src, crops = _bf16_crops()
    g = torch.Generator().manual_seed(3)
    px = t._pix_decode_crops_nograd(
        crops, border=8, n_frames=3, decode_batch=2, gen=g,
    )
    assert px.shape[1:] == (3, 176, 240)
    assert px.dtype == torch.float32
    assert not px.requires_grad


def test_grad_decode_flows_gradient_through_the_dtype_cast():
    """(b) ``.to(dtype)`` is differentiable — the graph still reaches the
    latent crop AFTER the cast, which is the whole point of the grad helper.

    Uses ``autograd.grad`` (not ``.backward()``) so the assertion is that
    the input is genuinely REACHABLE from the output, not merely that some
    ``.grad`` attribute got populated.
    """
    t = StrictVaeTrainer()
    lat = torch.randn(
        1, 2, LAT_C, 24, 32, dtype=torch.bfloat16,
    ).requires_grad_(True)                             # the bf16 latent crop
    g = torch.Generator().manual_seed(5)
    px = t._pix_decode_crops_grad(
        lat, border=8, n_frames=2, decode_batch=1, gen=g,
    )
    assert px.requires_grad
    (gr,) = torch.autograd.grad(px.sum(), lat, allow_unused=False)
    assert gr is not None
    assert gr.dtype == lat.dtype                       # bf16 in, bf16 grad
    assert float(gr.abs().sum()) > 0


def test_real_and_fake_decode_paths_get_identical_dtype_treatment():
    """(c) A cast on ONE side only is a real/fake tell. This test fails if
    anyone casts the fake and not the real (or vice versa).

    Production dtypes are asymmetric on entry -- the fake latents stream in
    bf16 while ``_pix_draw_reals`` hands over fp32 pool draws -- so the
    guarantee has to be that both arrive at the VAE in the SAME dtype.
    """
    g_kwargs = dict(border=8, n_frames=1, decode_batch=2)
    fake = torch.randn(2, 2, LAT_C, 24, 32, dtype=torch.bfloat16)
    real = fake.to(torch.float32)                      # the pool's dtype

    # Fake side: both the D-loop (no_grad) and the G-term (grad) helper.
    seen = {}
    for name, kind in (("fake_nograd", "n"), ("fake_grad", "g"),
                       ("real_nograd", "n")):
        t = StrictVaeTrainer()
        src = real if name.startswith("real") else fake
        gen = torch.Generator().manual_seed(11)
        if kind == "g":
            t._pix_decode_crops_grad(src, gen=gen, **g_kwargs)
        else:
            t._pix_decode_crops_nograd(src, gen=gen, **g_kwargs)
        seen[name] = list(t.model.vae.seen)

    assert seen["fake_nograd"] == seen["real_nograd"], (
        f"real/fake dtype asymmetry in the D-loop: {seen}"
    )
    assert seen["fake_grad"] == seen["fake_nograd"], (
        f"grad/no-grad dtype asymmetry on the fake side: {seen}"
    )
    assert set(seen["fake_nograd"]) == {torch.float32}

    # The comparison above is made through the SHIPPED wrappers, and
    # ``_vae_decode_nograd`` re-casts to fp32 on its own -- which would mask
    # a cast dropped from the no-grad helper ONLY.  So measure what OUR two
    # helpers actually hand over, at their own boundary: the grad helper's
    # argument to ``_vae_decode_grad`` (recorded by the strict VAE) vs the
    # no-grad helper's argument to the probe's ``_decode_crops``.  Drop the
    # cast on either side alone and these two disagree.
    from model import disc_holdout_probe as _dhp
    handed = []
    _real_dc = _dhp._decode_crops

    def _spy(trainer, crops, **kw):
        handed.append(crops.dtype)
        return _real_dc(trainer, crops, **kw)

    t = StrictVaeTrainer()
    with patch.object(_dhp, "_decode_crops", _spy):
        t._pix_decode_crops_nograd(
            fake, gen=torch.Generator().manual_seed(11), **g_kwargs,
        )
    assert handed == [torch.float32], (
        f"no-grad helper handed {handed}, grad helper handed "
        f"{seen['fake_grad'][:1]}"
    )
    assert handed[0] == seen["fake_grad"][0]

    # And the guard actually fires: cast one side only and the equality dies.
    t = StrictVaeTrainer()
    gen = torch.Generator().manual_seed(11)
    with patch.object(StrictVaeTrainer, "_pix_vae_dtype",
                      lambda self: torch.bfloat16):
        # bf16 VAE-dtype claim + the shipped nograd wrapper's own fp32
        # re-cast is precisely the "one side only" shape; the grad side
        # would then hand the decoder bf16 while the no-grad side hands it
        # fp32.
        with pytest.raises(RuntimeError):
            t._pix_decode_crops_grad(fake, gen=gen, **g_kwargs)
    assert t.model.vae.seen == [torch.bfloat16]


def test_frame_selection_stays_after_the_decode_in_both_helpers():
    """The cast must not have moved frame selection ahead of the decode --
    real and fake share geometry AND selection rule (docs/WP_PIXGAN.md §9)."""
    body = method_code(Trainer._pix_decode_crops_grad)
    assert body.index("_vae_decode_grad(") < body.index("randperm")
    assert body.index("_pix_vae_dtype()") < body.index("_vae_decode_grad(")
    # empty_cache still comes first of all (kept from the shipped contract).
    assert body.index("empty_cache") < body.index("_pix_vae_dtype()")
    nbody = method_code(Trainer._pix_decode_crops_nograd)
    assert "randperm" not in nbody          # selection lives in _decode_crops
    probe = method_code(CAFT and __import__(
        "model.disc_holdout_probe", fromlist=["_decode_crops"],
    )._decode_crops)
    assert probe.index("_vae_decode_nograd(") < probe.index("randperm")


def test_p_is_660_computed_from_the_tensor_not_a_constant():
    d = PixelTextureDisc()
    with torch.no_grad():
        logits = d(torch.zeros(1, 3, 176, 240))
    assert logits.shape == (1, 1, 22, 30)
    assert int(logits[0].numel()) == 660          # NOT the doc's 768


# ===========================================================================
# 3. THE CONFIG->PIPELINE SEAM for ``pix_finish_grad_enabled``
#
# Found by a live 60-step smoke (smoke_pixtex ran with
# pix_finish_grad_enabled=true and the A23 grad path NEVER ENGAGED).
# The flag is read off TWO DIFFERENT OBJECTS:
#   * pipeline/action_forcing_training.py reads
#     ``getattr(self, "pix_finish_grad_enabled", False)`` -- a plain
#     ATTRIBUTE ON THE PIPELINE, which is that file's convention;
#   * this trainer reads ``getattr(self.config, ...)`` -- off the CONFIG.
# Nothing connected them, so the config value never reached the pipeline,
# the grad buffer was never built, and the trainer's fail-loud check fired.
#
# BOTH HALVES WERE ALREADY TESTED AND THE SEAM WAS NOT: T1's suite sets the
# attribute directly on a stub pipeline, and T3-B's tests set the config.
# These tests start from a CONFIG VALUE and assert it ARRIVES on the
# pipeline object, executing the SHIPPED construction source.
# ===========================================================================
_PIPE_SRC_PATH = os.path.join(_ROOT, "pipeline", "action_forcing_training.py")
_FLAG = "pix_finish_grad_enabled"


def _pipe_source():
    with open(_PIPE_SRC_PATH, encoding="utf-8") as fh:
        return fh.read()


def _reading_class_name():
    """Parse pipeline/action_forcing_training.py for the class that performs
    ``getattr(self, "pix_finish_grad_enabled", ...)``.

    THE POINT: the flag has to land on an instance of THIS class. Two
    plausible receivers exist on the trainer -- ``self.model``
    (ActionForcingDMD) and ``self.pipeline`` (ActionForcingTrainingPipeline)
    -- and the trainer even cross-links them. Setting it on the wrong one
    compiles, runs, logs a cheerful confirmation and does nothing at all.
    """
    tree = ast.parse(_pipe_source())
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for node in ast.walk(cls):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "getattr"
                    and len(node.args) >= 2
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id == "self"
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value == _FLAG):
                return cls.name
    raise AssertionError(
        f"pipeline no longer reads {_FLAG} off self -- seam contract changed")


def _trainer_attr_holding(cls_name):
    """The ``self.<attr>`` the trainer assigns an instance of ``cls_name``."""
    tree = ast.parse(_af_source())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        v = node.value
        if not (isinstance(v, ast.Call) and isinstance(v.func, ast.Name)
                and v.func.id == cls_name):
            continue
        for t in node.targets:
            if (isinstance(t, ast.Attribute)
                    and isinstance(t.value, ast.Name)
                    and t.value.id == "self"):
                return t.attr
    raise AssertionError(f"trainer never constructs a {cls_name}")


def _trainer_flag_receivers():
    """Every ``self.<X>.pix_finish_grad_enabled = ...`` target in the
    trainer, plus the local-alias form ``_pipe.<flag> = ...``."""
    tree = ast.parse(_af_source())
    direct, aliased = set(), set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            if isinstance(t, ast.Attribute) and t.attr == _FLAG:
                base = t.value
                if (isinstance(base, ast.Attribute)
                        and isinstance(base.value, ast.Name)
                        and base.value.id == "self"):
                    direct.add(base.attr)
                elif isinstance(base, ast.Name):
                    aliased.add(base.id)
    return direct, aliased


def test_flag_lands_on_the_object_whose_class_reads_it():
    """THE SEAM INVARIANT, enforced structurally rather than by convention.

    Derives the receiver from the code on BOTH sides -- which class reads
    the flag, and which trainer attribute holds an instance of that class --
    then requires the assignment to target that same attribute. This is the
    check that catches "set it on self.model instead of self.pipeline",
    which is a silent no-op that every value-level test still passes.
    """
    reader = _reading_class_name()
    expected_attr = _trainer_attr_holding(reader)
    direct, _aliased = _trainer_flag_receivers()
    assert direct, f"nothing in the trainer assigns {_FLAG}"
    assert direct == {expected_attr}, (
        f"{_FLAG} is read by {reader}, which the trainer holds as "
        f"self.{expected_attr}, but the trainer assigns the flag to "
        f"self.{sorted(direct)} -- a silent no-op."
    )


def test_the_reader_is_the_pipeline_not_the_dmd_model():
    """Names the two candidates explicitly so a future reader sees the
    trap rather than re-deriving it."""
    assert _reading_class_name() == "ActionForcingTrainingPipeline"
    assert _trainer_attr_holding("ActionForcingTrainingPipeline") == "pipeline"
    assert "ActionForcingDMD" != _reading_class_name()


# --- executable half: a CONFIG value must ARRIVE on the pipeline object ---
_SEAM_ANCHOR = "        self.model.inference_pipeline = self.pipeline\n"
_SEAM_END = ('        if self.is_main_process:\n            logging.info(\n'
             '                "[ActionForcing] Pipeline: num_frame_per_block')


def _seam_block():
    import textwrap
    src = _af_source()
    i = src.index(_SEAM_ANCHOR)
    j = src.index(_SEAM_END, i)
    return textwrap.dedent(src[i:j])


class _StubPipe:
    """A bare object, exactly like the real pipeline before the trainer
    touches it -- the attribute is ABSENT."""


def _run_seam(**cfg):
    self_stub = SimpleNamespace(
        model=SimpleNamespace(), pipeline=_StubPipe(),
        config=SimpleNamespace(**cfg), is_main_process=True,
    )
    # ``sys`` is supplied because the shipped block emits the A23
    # gate-propagation proof to stderr as well as through logging: an
    # INFO-only proof-of-connection is invisible under a WARNING-level
    # root (see Trainer._pix_emit_actionable). This supplies a name the
    # PRODUCT block genuinely uses -- nothing here is stubbed out.
    ns = {"self": self_stub, "logging": CAFT.logging, "bool": bool,
          "getattr": getattr, "type": type, "sys": sys}
    exec(compile(_seam_block(), "<seam>", "exec"), ns)
    return self_stub


@pytest.mark.parametrize("value", [True, False])
def test_config_pix_finish_grad_reaches_the_pipeline_object(value):
    st = _run_seam(**{_FLAG: value})
    assert hasattr(st.pipeline, _FLAG), (
        f"CONFIG->PIPELINE SEAM BROKEN: config.{_FLAG}={value} never "
        "reached the pipeline object, so the A23 grad buffer is never "
        "built and the ladder-fake path cannot run."
    )
    assert getattr(st.pipeline, _FLAG) is value


def test_pix_finish_grad_false_propagates_as_false_not_absent():
    st = _run_seam(**{_FLAG: False})
    assert getattr(st.pipeline, _FLAG) is False


def test_pix_finish_grad_absent_from_config_propagates_as_false():
    st = _run_seam()
    assert getattr(st.pipeline, _FLAG) is False


def test_the_dmd_model_is_not_the_receiver():
    """Guards the exact regression: the flag must NOT be parked on
    self.model, where nothing reads it."""
    st = _run_seam(**{_FLAG: True})
    assert not hasattr(st.model, _FLAG)


def test_gate_re_asserts_the_flag_after_a_pipeline_rebuild(loader_patch):
    """A flag that survives construction but not a rebuild is the same bug
    with a longer fuse. ``self.pipeline`` is assigned exactly once today
    (audited: trainer/causal_action_forcing_train.py:1342, never rebound),
    so construction is what carries it in production and this re-assert is
    insurance -- but insurance that is tested."""
    t = StubTrainer(pix_finish_grad_enabled=True)
    t.pipeline = _StubPipe()                     # simulate a rebuild
    assert not hasattr(t.pipeline, _FLAG)
    info = _info_ladder()
    t._maybe_run_pixel_texture_d_updates(info, {}, current_step=0)
    assert getattr(t.pipeline, _FLAG) is True


# ===========================================================================
# 4. The FAKE — source selection and MASK selection (§6.2)
# ===========================================================================
def test_default_fake_source_is_the_flash_tensor():
    """The first adversarial arm runs pix_finish_grad_enabled=false."""
    t = StubTrainer()
    info = _info_flash()
    z, logs = t._pix_select_fake_latents(info)
    assert logs["train/pix_fake_source_ladder"] == 0.0
    assert z.shape == (1, 2, LAT_C, LAT_H, LAT_W)
    assert torch.equal(z, info["flash_dmd_gan_x0"][:, -2:])


def test_ladder_fake_selects_only_mask_valid_frames():
    t = StubTrainer(pix_finish_grad_enabled=True)
    info = _info_ladder(n_lat=7, mask=[True] * 6 + [False])
    z, logs = t._pix_select_fake_latents(info)
    assert logs["train/pix_fake_source_ladder"] == 1.0
    assert logs["train/pix_finish_grad_frames"] == 6.0
    assert logs["train/pix_finish_grad_frames_total"] == 7.0
    assert abs(logs["train/pix_finish_grad_frac"] - 6.0 / 7.0) < 1e-6
    # frames 4,5 — inside the valid run, never the detached trailing frame 6
    assert torch.equal(z, info["finish_denoised_chunk_grad"][:, 4:6])


def test_ladder_fake_skips_an_interior_detached_frame():
    t = StubTrainer(pix_finish_grad_enabled=True)
    info = _info_ladder(n_lat=8, mask=[True, True, True, False,
                                       True, True, False, False])
    z, _ = t._pix_select_fake_latents(info)
    assert torch.equal(z, info["finish_denoised_chunk_grad"][:, 4:6])


def test_missing_mask_fails_loud():
    t = StubTrainer(pix_finish_grad_enabled=True)
    info = _info_ladder()
    del info["finish_denoised_chunk_grad_mask"]
    with pytest.raises(RuntimeError, match="_mask.. is MISSING"):
        t._pix_select_fake_latents(info)


def test_all_false_mask_fails_loud():
    t = StubTrainer(pix_finish_grad_enabled=True)
    info = _info_ladder(n_lat=5, mask=[False] * 5)
    with pytest.raises(RuntimeError, match="ALL-FALSE"):
        t._pix_select_fake_latents(info)


def test_missing_grad_buffer_fails_loud_and_does_not_fall_back_to_flash():
    t = StubTrainer(pix_finish_grad_enabled=True)
    info = _info_flash()          # flash present, ladder absent
    with pytest.raises(RuntimeError, match="finish_denoised_chunk_grad"):
        t._pix_select_fake_latents(info)


def test_mask_length_mismatch_fails_loud():
    t = StubTrainer(pix_finish_grad_enabled=True)
    info = _info_ladder(n_lat=7, mask=[True] * 5)
    with pytest.raises(RuntimeError, match="does not describe this buffer"):
        t._pix_select_fake_latents(info)


# ===========================================================================
# 5. A20 — one independently SOURCED real per fake
# ===========================================================================
def test_a20_directive_shape_is_twentyfour_and_twentyfour(loader_patch):
    """The shipped shape after the researcher's exposure directive
    (2026-08-24): 8 crops x 3 frames = 24 fakes + 24 independently drawn
    band-matched reals PER D-UPDATE (was 4 x 3 = 12-and-12). A20's 1:1
    invariant scales with the shape by construction; this pins it at the
    shape production now ships."""
    t = StubTrainer(pix_crops_per_step=8, pix_frames_per_crop=3,
                    pix_gan_updates_per_step=1, pix_real_pool_windows=128)
    out = {}
    t._maybe_run_pixel_texture_d_updates(_info_flash(), out, current_step=0)
    assert out["train/pix_fake_images"] == 24.0
    assert out["train/pix_real_images"] == 24.0
    assert out["train/pix_real_repeat_frac"] == 0.0
    assert out["train/pix_real_distinct_sources"] == 24.0


def test_exposure_arithmetic_is_120_and_120_per_step(loader_patch):
    """The directive's NET arithmetic: 24 per update x 5 FOLLOWED updates
    = 120 fakes + 120 reals per training step (was 12/12) -- counted by
    the monotone exposure totals, never assumed from the config."""
    t = StubTrainer(pix_crops_per_step=8, pix_frames_per_crop=3,
                    pix_gan_updates_per_step=None,
                    pix_real_pool_windows=256)
    t.gan_updates_per_step = 5
    out = {}
    t._maybe_run_pixel_texture_d_updates(_info_flash(), out, current_step=0)
    assert out["train/pix_d_updates"] == 5.0
    assert out["train/pix_fake_images"] == 24.0
    assert out["train/pix_real_images"] == 24.0
    assert out["train/pix_fake_images_total"] == 120.0
    assert out["train/pix_real_images_total"] == 120.0


def test_exposure_totals_are_monotone_across_steps(loader_patch):
    t = StubTrainer()
    out1, out2 = {}, {}
    t._maybe_run_pixel_texture_d_updates(_info_flash(), out1, current_step=0)
    f1 = out1["train/pix_fake_images_total"]
    r1 = out1["train/pix_real_images_total"]
    assert f1 == out1["train/pix_fake_images"] * out1["train/pix_d_updates"]
    assert r1 == out1["train/pix_real_images"] * out1["train/pix_d_updates"]
    t._maybe_run_pixel_texture_d_updates(_info_flash(), out2, current_step=1)
    assert out2["train/pix_fake_images_total"] == 2 * f1
    assert out2["train/pix_real_images_total"] == 2 * r1


def test_decode_cost_is_measured_not_assumed(loader_patch):
    """Sec. 6: the real-side decodes are no_grad but the loader lands on
    the critical path -- so the D-block's decode-window count and its
    wall-time delta are LOGGED, not inferred from the knobs."""
    t = StubTrainer()
    out = {}
    t._maybe_run_pixel_texture_d_updates(_info_flash(), out, current_step=0)
    assert out["train/pix_decode_windows_step"] == (
        out["train/pix_decodes"] * out["train/pix_d_updates"]
    )
    assert out["train/pix_d_block_wall_s"] > 0.0


def test_depth_tag_feeds_the_fake_depth_histogram(loader_patch):
    """The depth tag (0-based AR chunk index, threaded from the call
    site) lands in cumulative per-bin image counts, so "the D sees
    drifted fakes" is a number. Bins are 0-1 / 2-3 / 4+; counts are
    cumulative over the run and monotone."""
    t = StubTrainer()
    out = {}
    t._maybe_run_pixel_texture_d_updates(
        _info_flash(), out, current_step=0, chunk_depth=0)
    n = out["train/pix_fake_images"] * out["train/pix_d_updates"]
    assert out["train/pix_fake_depth"] == 0.0
    assert out["train/pix_fake_depth_images_0_1"] == n
    assert out["train/pix_fake_depth_images_2_3"] == 0.0
    assert out["train/pix_fake_depth_images_4p"] == 0.0
    out2 = {}
    t._maybe_run_pixel_texture_d_updates(
        _info_flash(), out2, current_step=1, chunk_depth=6)
    assert out2["train/pix_fake_depth"] == 6.0
    assert out2["train/pix_fake_depth_images_0_1"] == n   # cumulative
    assert out2["train/pix_fake_depth_images_4p"] == n


def test_depth_histogram_is_omitted_when_no_tag_is_threaded(loader_patch):
    """OMIT-NEVER-FAKE companion: with no depth tag the histogram must be
    ABSENT -- zero-filled bins would certify "no drifted fakes seen" for
    a measurement that never happened."""
    t = StubTrainer()
    out = {}
    t._maybe_run_pixel_texture_d_updates(_info_flash(), out, current_step=0)
    assert not [k for k in out if k.startswith("train/pix_fake_depth")], (
        [k for k in out if k.startswith("train/pix_fake_depth")]
    )


def test_call_site_threads_the_depth_tag_from_chunk_identity():
    """The depth tag is DERIVED AT THE CALL SITE in
    _streaming_train_one_chunk, where the chunk identity is known --
    never guessed from trainer state inside the helper."""
    src = _af_source()
    i = src.index("self._maybe_run_pixel_texture_d_updates(")
    ctx = src[i:i + 500]
    assert "chunk_depth=" in ctx, ctx
    assert "_chunks_in_current_ride" in ctx, ctx


def test_a20_one_real_per_fake_at_stub_shape(loader_patch):
    t = StubTrainer()
    out = {}
    t._maybe_run_pixel_texture_d_updates(_info_flash(), out, current_step=0)
    assert out["train/pix_real_images"] == out["train/pix_fake_images"]
    assert out["train/pix_real_repeat_frac"] == 0.0


def test_a20_reals_come_from_distinct_source_windows(loader_patch):
    t = StubTrainer()
    g = torch.Generator().manual_seed(1)
    t._pix_pool_fill(40, gen=g, crop_rows=24, crop_cols=32, n_bands=3,
                     lat_frames=2)
    lat, bands, logs = t._pix_draw_reals(
        [0, 0, 1, 1, 2, 2], gen=g, crop_rows=24, crop_cols=32, n_bands=3,
        lat_frames=2,
    )
    assert lat.shape[0] == 6
    assert logs["train/pix_real_repeat_frac"] == 0.0
    assert logs["train/pix_real_distinct_sources"] == 6.0
    # Distinct SOURCE frames: no two reals share a (ride, start, y0, x0).
    uids = [e["uid"] for e in t._pix_real_pool]
    assert len(set(uids)) == len(uids)


def test_a20_planted_repeat_fires(loader_patch):
    """PLANTED VIOLATION: force the pool onto ONE source-frame span and
    prove pix_real_repeat_frac stops being 0 and the D-loop REFUSES.

    The plant goes into ``ride``/``start`` -- the fields
    ``_pix_draw_reals`` actually reads -- NOT into ``uid``. This test used
    to plant ``uid`` alone, which was correct against the OLD gauge
    (``len(set(uids)) < len(uids)``) and went silently vacuous the moment
    that gauge was reworked into a source-frame SPAN overlap: nothing
    reads ``uid``, so the "violation" planted nothing and repeat_frac was
    legitimately 0.0 while this companion still "passed". A
    planted-violation companion that plants into a field no product code
    reads proves nothing, so the plant is pinned to the gauge's own
    inputs: ``pool[i]["ride"]``, ``pool[i]["start"]`` and the admitted
    span width ``pool[i]["nf"]``.

    ``uid`` is deliberately left ALONE so the distinct-source build
    assertion stays honest and the RuntimeError raised below can only be
    the A20 repeat guard.
    """
    t = StubTrainer()
    g = torch.Generator().manual_seed(2)
    t._pix_pool_fill(40, gen=g, crop_rows=24, crop_cols=32, n_bands=3,
                     lat_frames=2)
    for e in t._pix_real_pool:
        e["ride"], e["start"] = "SAME", 0
    _lat, _b, logs = t._pix_draw_reals(
        [0, 0, 1, 1], gen=g, crop_rows=24, crop_cols=32, n_bands=3,
        lat_frames=2,
    )
    assert logs["train/pix_real_repeat_frac"] > 0.0
    with pytest.raises(RuntimeError, match="pix_real_repeat_frac"):
        t._maybe_run_pixel_texture_d_updates(_info_flash(), {},
                                             current_step=0)


def test_a20_frame_expansion_never_leaks_to_the_real_side():
    """The real decode must ask for exactly ONE pixel frame per window."""
    body = textwrap.dedent(
        inspect.getsource(Trainer._maybe_run_pixel_texture_d_updates))
    i = body.index("real_px = self._pix_decode_crops_nograd(")
    assert "n_frames=1" in body[i:i + 300]


# ===========================================================================
# 6. A24 — coarse thirds, same-band pairing, mismatch counter 0
# ===========================================================================
def test_a24_pairing_is_same_band_and_mismatch_is_zero(loader_patch):
    t = StubTrainer()
    out = {}
    t._maybe_run_pixel_texture_d_updates(_info_flash(), out, current_step=0)
    assert out["train/pix_band_mismatch"] == 0.0
    assert t._pix_band_mismatch == 0


def test_a24_draw_reals_always_returns_the_requested_band(loader_patch):
    t = StubTrainer()
    g = torch.Generator().manual_seed(4)
    t._pix_pool_fill(30, gen=g, crop_rows=24, crop_cols=32, n_bands=3,
                     lat_frames=2)
    req = [0, 1, 2, 2, 1, 0, 0, 2]
    _lat, got, _logs = t._pix_draw_reals(
        req, gen=g, crop_rows=24, crop_cols=32, n_bands=3, lat_frames=2,
    )
    assert got == req


def test_a24_band_count_cannot_be_silently_tightened(loader_patch):
    for n in (2, 4, 6, 12, 60):
        t = StubTrainer(pix_band_count=n)
        with pytest.raises(ValueError, match="A24 pins COARSE thirds"):
            t._maybe_run_pixel_texture_d_updates(_info_flash(), {},
                                                 current_step=0)


def test_a24_planted_band_mismatch_fires(loader_patch):
    """PLANTED VIOLATION: a real supply that ignores the requested band."""
    import types
    t = StubTrainer()
    real_draw = t._pix_draw_reals

    def bad(self_, bands, **kw):
        lat, got, logs = real_draw(bands, **kw)
        return lat, [(int(b) + 1) % 3 for b in got], logs

    t._pix_draw_reals = types.MethodType(bad, t)
    with pytest.raises(RuntimeError, match="A24 VIOLATION"):
        t._maybe_run_pixel_texture_d_updates(_info_flash(), {},
                                             current_step=0)


def test_a24_exact_y_matching_is_absent_from_the_source():
    code = code_only(_af_source())
    i = code.index("_pix_take_crops_with_origins")
    seg = code[i:code.index("_pix_disc_module")]
    for forbidden in ("tolerance", "nearest", "argmin", "cdist"):
        assert forbidden not in seg, forbidden


# ===========================================================================
# 7. A21 — support measured from DISTINCT identities
# ===========================================================================
def test_a21_support_is_counted_from_distinct_identities(loader_patch):
    t = StubTrainer()
    g = torch.Generator().manual_seed(6)
    n = t._pix_pool_fill(25, gen=g, crop_rows=24, crop_cols=32, n_bands=3,
                         lat_frames=2)
    assert n == 25
    ids = set()
    for e in t._pix_real_pool:
        for j in range(2):
            ids.add((e["ride"], e["start"] + j))
    # Measured off actual (ride, latent-frame) identities, never inferred
    # from the loader's nominal size.
    assert t._pix_real_support == ids
    assert len(t._pix_real_support) <= 25 * 2
    assert len(t._pix_real_rides) >= 1


def test_a21_support_does_not_inflate_on_a_repeated_window(loader_patch):
    t = StubTrainer()
    g = torch.Generator().manual_seed(6)
    t._pix_pool_fill(10, gen=g, crop_rows=24, crop_cols=32, n_bands=3,
                     lat_frames=2)
    before = len(t._pix_real_support)
    e = t._pix_real_pool[0]
    for j in range(2):
        t._pix_real_support.add((e["ride"], e["start"] + j))
    assert len(t._pix_real_support) == before


def test_a21_pool_is_continuously_refreshed(loader_patch):
    t = StubTrainer()
    out = {}
    t._maybe_run_pixel_texture_d_updates(_info_flash(), out, current_step=0)
    n1 = len(t._pix_real_pool)
    s1 = len(t._pix_real_support)
    t._maybe_run_pixel_texture_d_updates(_info_flash(), {}, current_step=1)
    assert len(t._pix_real_support) > s1, "support did not grow -> frozen pool"
    assert len(t._pix_real_pool) >= n1


def test_a21_reals_are_cross_ride(loader_patch):
    t = StubTrainer()
    g = torch.Generator().manual_seed(8)
    t._pix_pool_fill(60, gen=g, crop_rows=24, crop_cols=32, n_bands=3,
                     lat_frames=2)
    assert len(t._pix_real_rides) >= 2, t._pix_real_rides


def test_a21_floor_constant_is_4096():
    assert Trainer.PIX_A21_SUPPORT_FLOOR == 4096


# ===========================================================================
# 8. A22 — reserved rides provably absent
# ===========================================================================
def test_a22_holdout_rides_absent_from_the_real_supply(loader_patch):
    t = StubTrainer()
    paths = t._pix_train_ride_paths()
    names = {os.path.basename(str(p)) for p in paths}
    assert names & t._dhp_holdout_names == set()
    assert t._pix_holdout_leak == 0.0
    assert t._pix_holdout_checked == 1.0


def test_a22_telemetry_reports_the_check(loader_patch):
    t = StubTrainer()
    out = {}
    t._maybe_run_pixel_texture_d_updates(_info_flash(), out, current_step=0)
    assert out["train/pix_holdout_checked"] == 1.0
    assert out["train/pix_holdout_leak"] == 0.0
    assert out["train/pix_holdout_rides"] == 2.0


def test_a22_planted_leak_fires(loader_patch):
    """PLANTED VIOLATION: a reserved ride present in the training list."""
    t = StubTrainer()
    t._dhp_holdout_names = {"train_2.zarr", "held_0.zarr"}
    with pytest.raises(RuntimeError, match="A22 VIOLATION"):
        t._pix_train_ride_paths()
    with pytest.raises(RuntimeError, match="A22 VIOLATION"):
        t._maybe_run_pixel_texture_d_updates(_info_flash(), {},
                                             current_step=0)


def test_a22_empty_training_ride_list_fails_loud(loader_patch):
    t = StubTrainer()
    t.dataset._rides = []
    with pytest.raises(RuntimeError, match="dataset._rides"):
        t._pix_train_ride_paths()


# ===========================================================================
# 9. FORGEABLE ZEROS — assert ABSENCE, never a pinned 0.0
#
# Several keys here have 0.0 as their REQUIRED HEALTHY value
# (pix_real_repeat_frac, pix_band_mismatch, pix_holdout_leak) or as a
# SPECIFIC ALARM (pix_r1_rate < 0.99 = build bug). Emitting one on a path
# that did not run certifies a perfect result for a computation that never
# happened. These tests assert the key is MISSING on the not-run path — a
# test that pinned it to 0.0 would harden the defect against being fixed.
# ===========================================================================
_HEALTHY_ZERO_DIAGNOSTICS = (
    "train/pix_real_repeat_frac",
    "train/pix_band_mismatch",
    "train/pix_r1_rate",
    "train/pix_d_loss",
    "train/pix_r1_grad_sq",
    "train/pix_patch_count",
    "train/pix_real_images",
    "train/pix_fake_images",
    "train/pix_crop_y0_mean",
    "train/pix_crop_x0_mean",
    # RESEARCHER EXPOSURE DIRECTIVE (2026-08-24): exposure is measured,
    # never assumed -- and never zero-filled on a path that did not run.
    "train/pix_fake_images_total",
    "train/pix_real_images_total",
    "train/pix_decode_windows_step",
    "train/pix_d_block_wall_s",
)
_SAFETY_CLAIMS = (
    "train/pix_holdout_leak",
    "train/pix_a21_support_ok",
    "train/pix_real_support_frames",
)


def test_warmup_path_omits_every_healthy_zero_key(loader_patch):
    t = StubTrainer()
    t.gan_disc_start_step = 100
    out = {}
    t._maybe_run_pixel_texture_d_updates(_info_flash(), out, current_step=3)
    for k in _HEALTHY_ZERO_DIAGNOSTICS + _SAFETY_CLAIMS:
        assert k not in out, f"{k} forged on the warmup path"
    # The regime flag is what makes the absence interpretable.
    assert out["train/pix_d_warmup_skipped"] == 1.0
    assert out["train/pix_d_updates"] == 0.0


def test_zero_updates_path_omits_every_healthy_zero_key(loader_patch):
    t = StubTrainer(pix_gan_updates_per_step=0)
    out = {}
    t._maybe_run_pixel_texture_d_updates(_info_flash(), out, current_step=0)
    assert out["train/pix_d_updates"] == 0.0
    for k in _HEALTHY_ZERO_DIAGNOSTICS + _SAFETY_CLAIMS:
        assert k not in out, f"{k} forged with pix_gan_updates_per_step=0"


def test_run_path_emits_every_one_of_them(loader_patch):
    t = StubTrainer()
    out = {}
    t._maybe_run_pixel_texture_d_updates(_info_flash(), out, current_step=0)
    for k in _HEALTHY_ZERO_DIAGNOSTICS + _SAFETY_CLAIMS:
        assert k in out, f"{k} missing on the run path"
    assert "train/pix_d_warmup_skipped" not in out


def test_flash_path_omits_the_a23_grad_dilution_counters(loader_patch):
    """0 no-rung frames is the HEALTHY reading, and the flash path never
    consults the grad buffer — so the counters must be absent, with
    pix_fake_source_ladder as the regime flag."""
    t = StubTrainer()
    out = {}
    t._maybe_run_pixel_texture_d_updates(_info_flash(), out, current_step=0)
    assert out["train/pix_fake_source_ladder"] == 0.0
    for k in ("train/pix_finish_grad_frames",
              "train/pix_finish_grad_frames_total",
              "train/pix_finish_grad_frac"):
        assert k not in out, f"{k} forged on the flash path"


def test_ladder_path_emits_the_a23_grad_dilution_counters():
    t = StubTrainer(pix_finish_grad_enabled=True)
    _z, logs = t._pix_select_fake_latents(_info_ladder())
    for k in ("train/pix_finish_grad_frames",
              "train/pix_finish_grad_frames_total",
              "train/pix_finish_grad_frac"):
        assert k in logs


def test_r1_rate_is_never_a_zero_over_zero(loader_patch):
    t = StubTrainer(pix_gan_updates_per_step=0)
    out = {}
    t._maybe_run_pixel_texture_d_updates(_info_flash(), out, current_step=0)
    # PixCounters.r1_rate is fired/max(1, updates) -> 0.0 before the first
    # update, a FALSE build-bug alarm. It must not be emitted.
    assert "train/pix_r1_rate" not in out


def test_safety_claim_a22_fails_closed_when_unverifiable(loader_patch):
    """PLANTED VIOLATION: a D-loop that ran without the A22 check must
    RAISE, not omit and continue — silence and 'clean' must not be
    confusable for a correctness claim."""
    import types
    t = StubTrainer()

    def no_check(self_, *a, **kw):
        # pool fill that never runs the holdout disjointness check
        self_._pix_real_pool = getattr(self_, "_pix_real_pool", None) or []
        self_._pix_real_support = getattr(self_, "_pix_real_support", None) \
            or set()
        g = torch.Generator().manual_seed(0)
        from model.disc_holdout_probe import band_plan
        for _ in range(max(1, int(a[0]) if a else int(kw.get("n_new", 1)))):
            b, y = band_plan(1, LAT_H, 24, 3, g)
            self_._pix_real_pool.append({
                "lat": torch.randn(2, LAT_C, 24, 32).half(),
                "band": int(kw.get("force_band") if kw.get("force_band")
                            is not None else b[0]),
                "y0": int(y[0]), "x0": 0, "ride": "x.zarr",
                "start": len(self_._pix_real_pool),
                "uid": ("x.zarr", len(self_._pix_real_pool), 0, 0),
            })
            self_._pix_real_support.add(
                ("x.zarr", len(self_._pix_real_pool)))
        return 1

    t._pix_pool_fill = types.MethodType(no_check, t)
    with pytest.raises(RuntimeError, match="A22 UNVERIFIABLE"):
        t._maybe_run_pixel_texture_d_updates(_info_flash(), {},
                                             current_step=0)


def test_safety_claim_a21_fails_closed_when_unverifiable(loader_patch):
    import types
    t = StubTrainer()
    out = {}
    t._maybe_run_pixel_texture_d_updates(_info_flash(), out, current_step=0)
    assert out["train/pix_a21_support_ok"] in (0.0, 1.0)
    # Now delete the identity set the claim is measured from.
    real_draw = t._pix_draw_reals

    def drop(self_, bands, **kw):
        r = real_draw(bands, **kw)
        self_._pix_real_support = None
        return r

    t._pix_draw_reals = types.MethodType(drop, t)
    with pytest.raises(RuntimeError, match="A21 UNVERIFIABLE"):
        t._maybe_run_pixel_texture_d_updates(_info_flash(), {},
                                             current_step=1)


# ===========================================================================
# 10. R1 cadence, DDP-broadcast draws, decode counts, R2 tripwire
# ===========================================================================
def test_r1_fires_on_every_d_update_and_counters_are_monotone(loader_patch):
    t = StubTrainer()
    o1 = {}
    t._maybe_run_pixel_texture_d_updates(_info_flash(), o1, current_step=0)
    assert o1["train/pix_r1_rate"] == 1.00
    assert o1["train/pix_d_updates_total"] == 2.0
    assert o1["train/pix_r1_fires_total"] == 2.0
    o2 = {}
    t._maybe_run_pixel_texture_d_updates(_info_flash(), o2, current_step=1)
    assert o2["train/pix_d_updates_total"] == 4.0        # monotone
    assert o2["train/pix_r1_fires_total"] == 4.0
    assert o2["train/pix_r1_rate"] == 1.00


# --- the R1 running mean must actually be FED by the call site ----------
# WHY A SEPARATE TEST: PixCounters' incremental mean is unit-tested in T2
# (testing/test_pixel_texture_disc.py). What was untested -- and was in
# fact broken for the entire life of the key -- is that ANYTHING ever
# calls note_d_update WITH a gsq. The D-loop called
# ``note_d_update(r1_fired=True)`` only, so the mean averaged nothing and
# its keys stayed absent forever, while TEXTURE_GAN_DESIGN 5.3/5.4 told
# the reader to calibrate gamma off the mean and to read it on row one.
# A single draw is 126% relative sd at the spec crop, so the per-step
# ``train/pix_r1_grad_sq`` is NOT a substitute. No test of the counter
# class can see a missing caller; only a test of the CALL SITE can.
from model.pixel_texture_disc import PixCounters  # noqa: E402


class _SpyCounters(PixCounters):
    """Records exactly what the trainer hands note_d_update.

    PixCounters uses __slots__; a subclass that declares none gets a
    __dict__, which is what lets this hold the log.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.seen = []

    def note_d_update(self, r1_fired, gsq=None):
        self.seen.append(gsq)
        super().note_d_update(r1_fired=r1_fired, gsq=gsq)


def test_the_call_site_feeds_the_r1_grad_sq_running_mean(loader_patch):
    t = StubTrainer()
    c = _SpyCounters(r1_every_n=1)
    t._pix_counters_obj = c
    o1 = {}
    t._maybe_run_pixel_texture_d_updates(_info_flash(), o1, current_step=0)

    n1 = int(o1["train/pix_d_updates_total"])
    assert n1 == 2
    assert len(c.seen) == n1, "note_d_update must be called once per update"
    assert all(g is not None for g in c.seen), (
        "the D-loop called note_d_update WITHOUT the measured gsq, so the "
        "running mean averages nothing and train/pix_r1_grad_sq_mean can "
        "never be emitted -- the exact omission this test exists for"
    )
    assert "train/pix_r1_grad_sq_mean" in o1, (
        "the running mean is fed but never emitted -- unreadable is the "
        "same as unwired for a number 5.3 says to calibrate gamma from"
    )
    assert "train/pix_r1_grad_sq_n" in o1
    assert o1["train/pix_r1_grad_sq_n"] == float(n1), (
        "the sample count must equal the number of D-updates, else the "
        "mean silently covers fewer draws than the reader believes"
    )
    exp1 = sum(float(g) for g in c.seen) / len(c.seen)
    assert o1["train/pix_r1_grad_sq_mean"] == pytest.approx(exp1, rel=1e-6)
    # The single-draw key stays the LAST draw -- the mean is an ADDITION,
    # not a redefinition of the shipped key.
    assert o1["train/pix_r1_grad_sq"] == pytest.approx(float(c.seen[-1]))

    # ... and it keeps averaging ACROSS steps (the counters object is
    # per-trainer and persistent; a per-step object would reset the mean
    # to a single draw every row and defeat the whole point).
    o2 = {}
    t._maybe_run_pixel_texture_d_updates(_info_flash(), o2, current_step=1)
    n2 = int(o2["train/pix_d_updates_total"])
    assert n2 == 4
    assert len(c.seen) == n2
    assert o2["train/pix_r1_grad_sq_n"] == float(n2)
    exp2 = sum(float(g) for g in c.seen) / len(c.seen)
    assert o2["train/pix_r1_grad_sq_mean"] == pytest.approx(exp2, rel=1e-6)


def test_r1_grad_sq_mean_keys_are_absent_until_observed(loader_patch):
    """Absent, never 0.0: a 0.0 running mean reads as 'R1 measured zero
    gradient on the reals', which is a real and alarming diagnosis."""
    t = StubTrainer()
    t.gan_disc_start_step = 100
    out = {}
    t._maybe_run_pixel_texture_d_updates(_info_flash(), out, current_step=3)
    for k in ("train/pix_r1_grad_sq_mean", "train/pix_r1_grad_sq_n"):
        assert k not in out, f"{k} forged before any gsq was observed"


def test_the_unfed_running_mean_tripwire_fires_on_a_planted_omission(
        loader_patch):
    """PLANTED VIOLATION -- the pre-fix state, reproduced deliberately: a
    counter that is called on every D-update but never handed the value.
    The emission must vanish (absent, not 0.0), which is what makes the
    positive test above evidence rather than decoration."""
    class _DroppingCounters(_SpyCounters):
        def note_d_update(self, r1_fired, gsq=None):
            super().note_d_update(r1_fired=r1_fired, gsq=None)

    t = StubTrainer()
    t._pix_counters_obj = _DroppingCounters(r1_every_n=1)
    out = {}
    t._maybe_run_pixel_texture_d_updates(_info_flash(), out, current_step=0)
    assert out["train/pix_d_updates_total"] == 2.0     # loop really ran
    assert out["train/pix_r1_grad_sq"] > 0.0           # R1 really measured
    for k in ("train/pix_r1_grad_sq_mean", "train/pix_r1_grad_sq_n"):
        assert k not in out, f"{k} emitted from an unfed running mean"


def test_r1_cost_is_bounded_by_subsampling_not_by_every_n(loader_patch):
    t = StubTrainer(pix_crops_per_step=4, pix_frames_per_crop=3,
                    pix_r1_num_samples=3, pix_gan_updates_per_step=1)
    out = {}
    t._maybe_run_pixel_texture_d_updates(_info_flash(), out, current_step=0)
    assert out["train/pix_r1_n_used"] == 3.0
    assert out["train/pix_r1_rate"] == 1.00              # still 1.00


def test_r1_indices_and_eps_are_passed_explicitly_and_broadcast():
    body = method_code(Trainer._maybe_run_pixel_texture_d_updates)
    assert "dist.broadcast(idx, src=0)" in body
    assert "dist.broadcast(eps, src=0)" in body
    assert "indices=idx" in body and "eps=eps" in body
    assert "num_samples=None" in body
    # Cost must never be bounded by SKIPPING R1. Checked STRUCTURALLY: the
    # r1_penalty call must not sit under any conditional. A substring test
    # for "pix_r1_every_n" fired on the resolved-config echo, which names
    # every pix_* key because naming them is its entire job -- the third
    # time this tripwire class has flagged correct code.
    assert not _calls_under_a_conditional(
        Trainer._maybe_run_pixel_texture_d_updates, "r1_penalty")


def _calls_under_a_conditional(fn_or_src, callee):
    """True if any call to ``callee`` is nested inside an ``if``.

    Accepts a live function OR raw source text. The planted-violation
    companion builds its subject with ``exec``, which leaves the function
    with no source FILE, so ``inspect.getsource`` raises ``OSError`` on it
    -- passing the text straight through is the fix.
    """
    src = (fn_or_src if isinstance(fn_or_src, str)
           else inspect.getsource(fn_or_src))
    name = (fn_or_src if isinstance(fn_or_src, str)
            else fn_or_src.__name__)
    tree = ast.parse(textwrap.dedent(src))
    hits = []

    def walk(node, guarded):
        for child in ast.iter_child_nodes(node):
            if (isinstance(child, ast.Call)
                    and getattr(child.func, "id", None) == callee):
                hits.append(guarded)
            walk(child, guarded or isinstance(child, ast.If))

    walk(tree, False)
    assert hits, f"{callee} is never called in {name}"
    return any(hits)


def test_the_r1_cadence_tripwire_fires_on_a_planted_skip():
    """PLANTED VIOLATION: an R1 call moved under a cadence conditional."""
    src = textwrap.dedent("""
        def planted(self):
            if self.counters.should_fire_r1():
                r1 = r1_penalty(disc, real_px)
    """)
    assert _calls_under_a_conditional(src, "r1_penalty")
    clean = textwrap.dedent("""
        def clean(self):
            r1 = r1_penalty(disc, real_px)
    """)
    assert not _calls_under_a_conditional(clean, "r1_penalty")


class _FakeDist:
    """Minimal collective: broadcast copies rank 0's value to everyone."""

    def __init__(self, rank):
        self.rank = rank
        self.store = {}

    def is_initialized(self):
        return True

    def get_rank(self):
        return self.rank

    def broadcast(self, t, src=0):
        key = tuple(t.shape)
        if self.rank == src:
            self.store[key] = t.clone()
        elif key in self.store:
            t.copy_(self.store[key])
        return None


def test_sync_generator_draws_are_identical_across_simulated_ranks():
    shared = {}

    def draws(rank):
        fd = _FakeDist(rank)
        fd.store = shared
        with patch.object(CAFT, "dist", fd):
            t = StubTrainer()
            g = t._pix_sync_generator(7, 2)
            return torch.randperm(64, generator=g).tolist()

    r0 = draws(0)
    r1 = draws(1)
    r2 = draws(1)
    assert r0 == r1 == r2, "rank-divergent draw would desync the all-reduce"


def test_sync_generator_differs_across_steps_and_updates():
    t = StubTrainer()
    a = torch.randperm(64, generator=t._pix_sync_generator(0, 0)).tolist()
    b = torch.randperm(64, generator=t._pix_sync_generator(0, 1)).tolist()
    c = torch.randperm(64, generator=t._pix_sync_generator(1, 0)).tolist()
    assert a != b and a != c


def test_rank_generator_diversifies_the_real_data_draw():
    """The real-DATA draw is per-rank ON PURPOSE (identical reals on every
    rank would divide the effective real support by world_size). Nothing
    computed from it enters an all-reduced quantity."""
    def draw(rank):
        fd = _FakeDist(rank)
        with patch.object(CAFT, "dist", fd):
            t = StubTrainer()
            return torch.randperm(64,
                                  generator=t._pix_rank_generator(0, 0)
                                  ).tolist()
    assert draw(0) != draw(1)


def test_rank_generator_can_be_forced_synced_for_diagnostics():
    shared = {}

    def draw(rank):
        fd = _FakeDist(rank)
        fd.store = shared
        with patch.object(CAFT, "dist", fd):
            t = StubTrainer(pix_real_draw_rank_synced=True)
            return torch.randperm(64,
                                  generator=t._pix_rank_generator(0, 0)
                                  ).tolist()
    assert draw(0) == draw(1)


def test_decode_count_scales_with_crops_only(loader_patch):
    """WP_PIXGAN §9: _decode_crops decodes the WHOLE latent crop and only
    then draws the pixel-frame subset, so pix_frames_per_crop costs nothing
    at decode. Doubling frames must not change the decode count; the real
    side scales because A20 needs one independent SOURCE per fake."""
    def n_decodes(crops, frames):
        t = StubTrainer(pix_crops_per_step=crops, pix_frames_per_crop=frames,
                        pix_gan_updates_per_step=1, pix_decode_batch=1)
        t._maybe_run_pixel_texture_d_updates(_info_flash(), {},
                                             current_step=0)
        return t.n_decodes

    # fake crops + reals(= crops*frames, A20 1:1 on independent sources)
    assert n_decodes(2, 2) == 2 + 4
    assert n_decodes(4, 2) == 4 + 8
    assert n_decodes(2, 3) == 2 + 6


def banned_symbol_hits(src, prefix):
    """Occurrences of ``prefix`` as a SYMBOL that is defined, assigned or
    read -- not as a word in a blob.

    Reuses T2's ``code_only`` (docstrings + comments stripped) and
    ``code_identifiers`` (names / attributes / args / defs / kwargs), then
    adds the one thing identifiers miss and that genuinely matters: a banned
    name smuggled in as a STRING KEY of a config read or a log emission --
    a ``getattr`` on the run config with the banned name as its literal
    second argument, or a ``logs[...] = v`` whose key contains it.

    (Those two examples used to be spelled out here in full.  They are not any
    more, and deliberately: this docstring is SOURCE that the trainer's
    override-guard scanner walks, and spelling the banned name next to
    ``getattr(cfg,`` was one of the two places that allowlisted it.  See the
    RUNTIME-ASSEMBLED BANNED SYMBOL note near the fixtures below.)

    What it deliberately does NOT flag: a banned name merely mentioned in a
    string, a message, or -- the case that bit us three times -- a resolved-
    config echo that names every knob because naming them is its job.
    """
    code = code_only(src)
    hits = {i for i in code_identifiers(code) if prefix in i}
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and getattr(node.func, "id", None)
                == "getattr" and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
                and prefix in node.args[1].value):
            hits.add(node.args[1].value)
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if (isinstance(t, ast.Subscript)
                        and isinstance(t.slice, ast.Constant)
                        and isinstance(t.slice.value, str)
                        and prefix in t.slice.value):
                    hits.add(t.slice.value)
    return hits


# ---------------------------------------------------------------------------
# RUNTIME-ASSEMBLED BANNED SYMBOL
#
# ``trainer._scan_config_sourced_keys`` walks EVERY .py in the repo -- this
# file included -- and harvests ``<prefix><name>`` tokens that sit next to a
# config-receiver pattern (``getattr(cfg, "<name>"``, ``config.<name>``, ...)
# into the allowlist of ``--override`` keys the guard will ACCEPT.  It strips
# whole-line comments but NOT docstrings and NOT string literals.
#
# The planted-violation fixtures below exist to spell exactly such a pattern,
# so they were feeding the guard the one symbol B1 forbids outright.
# MEASURED, by running the real scanner over one file at a time: this file
# alone yielded ``{'pix_r2_gamma'}``; model/pixel_texture_disc.py,
# trainer/causal_action_forcing_train.py and the other three pixgan suites
# yielded nothing.  Harmless today -- nothing sets that key -- but it defeats
# the guard for precisely the symbol it is guarding.
#
# Fix: this file's SOURCE never contains the literal token.  It is assembled
# from fragments at import time, so the scanner's regex finds no ``pix_r2*``
# to harvest, while every fixture still contains exactly the text it needs to
# contain at RUN time.  ``test_this_file_does_not_allowlist_the_banned_key``
# pins that with the real scanner, and its planted companion proves the
# scanner does harvest the old spelling.
# ---------------------------------------------------------------------------
_R2 = "pix_" + "r2"

_R2_PLANTED_VIOLATIONS = (
    f"{_R2}_gamma = 1.0\n",                          # defined
    f'x = getattr(cfg, "{_R2}_gamma", 1.0)\n',       # read from config
    f'logs["train/{_R2}_fires"] = 1.0\n',            # emitted as a log key
    f"self.{_R2}_weight = 3\n",                      # assigned attribute
)

_R2_BENIGN_MENTIONS = (
    f'"""We deliberately have no {_R2}_gamma knob."""\nx = 1\n',
    f"# {_R2}_gamma was deleted\nx = 1\n",
    f'raise ValueError("{_R2}_gamma is not supported")\n',
)


def test_no_pix_r2_symbol_anywhere_in_this_chunk():
    """R2 is DELETED (WP_PIXGAN §2) -- no ``pix_r2_*`` symbol may be
    defined, assigned, read from config or emitted as a log key."""
    assert banned_symbol_hits(_af_source(), _R2) == set()


def test_the_r2_tripwire_fires_on_a_planted_violation():
    for planted in _R2_PLANTED_VIOLATIONS:
        assert banned_symbol_hits(planted, _R2), planted


def test_the_r2_tripwire_does_not_fire_on_prose_or_an_echo():
    """The three false positives this tripwire class has already produced:
    a docstring, a comment, and a resolved-config echo."""
    for benign in _R2_BENIGN_MENTIONS:
        assert banned_symbol_hits(benign, _R2) == set(), benign


def _scan_dir_for_banned(paths):
    """Run the REAL override-guard scanner over a throwaway dir holding
    ``paths``, and return the ``pix_r2*`` keys it allowlisted.

    The scanner returns ``None`` ("repo layout is not what we expect") on
    fewer than 10 ``.py`` files, so the dir is padded with inert ones.
    """
    d = tempfile.mkdtemp()
    try:
        for i in range(12):
            with open(os.path.join(d, f"pad{i}.py"), "w") as fh:
                fh.write("x = 1\n")
        for i, p in enumerate(paths):
            shutil.copy(p, os.path.join(d, f"subject{i}.py"))
        keys = CAFT._scan_config_sourced_keys(
            CAFT._OVERRIDE_GUARD_PREFIXES, d)
        assert keys is not None, "scanner declined to run"
        return {k for k in keys if k.startswith(_R2)}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_this_file_does_not_allowlist_the_banned_key():
    """The planted-violation fixtures must not teach the override guard to
    ACCEPT ``--override <banned>=...``.  Scanned with the real scanner."""
    assert _scan_dir_for_banned([os.path.abspath(__file__)]) == set()


def test_the_scanner_really_would_have_harvested_the_old_spelling():
    """PLANTED VIOLATION for the guard above.  Writes the fixture the way it
    used to be written -- the literal token next to ``getattr(cfg,`` -- and
    proves the scanner picks it up.  Without this, the test above would pass
    just as well if the scanner were broken."""
    d = tempfile.mkdtemp()
    try:
        for i in range(12):
            with open(os.path.join(d, f"pad{i}.py"), "w") as fh:
                fh.write("x = 1\n")
        with open(os.path.join(d, "old_spelling.py"), "w") as fh:
            fh.write(f'x = getattr(cfg, "{_R2}_gamma", 1.0)\n')
        keys = CAFT._scan_config_sourced_keys(
            CAFT._OVERRIDE_GUARD_PREFIXES, d)
        assert keys is not None
        assert f"{_R2}_gamma" in keys, sorted(keys)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ===========================================================================
# 11. Resolved-config echo — "only the resolved value settles it"
#
# From a live incident: two sessions independently concluded a knob had
# fallen back to its default because the launch script did not name it.
# Both were wrong — the holder passed it one level up. Every launcher-level
# check was accurate and the conclusion still came out wrong, because the
# effective value is assembled across two files.
# ===========================================================================
def test_resolved_config_is_echoed_once_with_band_count(loader_patch):
    t = StubTrainer()
    out = {}
    t._maybe_run_pixel_texture_d_updates(_info_flash(), out, current_step=0)
    assert out["train/pix_cfg_pix_band_count"] == 3.0      # §7: every run
    assert out["train/pix_cfg_pix_decode_batch"] == 4.0
    assert out["train/pix_cfg_gan_pixel_texture_enabled"] == 1.0
    assert out["train/pix_cfg_pix_finish_grad_enabled"] == 0.0
    assert out["train/pix_cfg_pix_gan_updates_per_step_derived"] == 2.0
    # Once per run, not once per step.
    out2 = {}
    t._maybe_run_pixel_texture_d_updates(_info_flash(), out2, current_step=1)
    assert not any(k.startswith("train/pix_cfg_") for k in out2)


def test_echo_reports_DERIVED_not_requested_when_clamped(loader_patch):
    """A crop larger than the latent grid is clamped by the crop helper.
    The echo must show what actually ran."""
    t = StubTrainer(pix_crop_lat=(80, 200), pix_crops_per_step=1,
                    pix_frames_per_crop=1, pix_gan_updates_per_step=1)
    out = {}
    t._maybe_run_pixel_texture_d_updates(_info_flash(), out, current_step=0)
    assert out["train/pix_cfg_pix_crop_lat_rows_requested"] == 80.0
    assert out["train/pix_cfg_pix_crop_lat_cols_requested"] == 200.0
    assert out["train/pix_cfg_pix_crop_lat_rows_derived"] == float(LAT_H)
    assert out["train/pix_cfg_pix_crop_lat_cols_derived"] == float(LAT_W)
    assert out["train/pix_cfg_pix_decode_border_trim_derived"] == 8.0


def test_echo_uses_an_out_of_range_sentinel_for_none(loader_patch):
    t = StubTrainer()          # pix_r1_num_samples=None
    out = {}
    t._maybe_run_pixel_texture_d_updates(_info_flash(), out, current_step=0)
    # -1 is OUTSIDE the knob's meaningful range (>= 1). 0 would read as a
    # real setting, which is the forgeable-placeholder failure again.
    assert out["train/pix_cfg_pix_r1_num_samples"] == -1.0


def test_echo_reports_the_optimizers_LIVE_lr_not_the_config(loader_patch):
    """THE DEFECT this fix removed, in one assertion.

    The echo's whole job is that a REQUESTED value and an EFFECTIVE value
    can differ and only the effective one settles anything. ``pix_gan_lr``
    was echoed straight off ``cfg`` -- but ``_build_optimizer`` is what
    turned that config into the number that actually steps D, and anything
    that moves the param group afterwards (a resume restoring an optimizer
    state dict, a scheduler, a hand poke) leaves the config untouched. The
    echo then reported a number nothing uses while looking authoritative.

    Mutating the param group is exactly what all three of those do, so this
    test fails against the pre-fix trainer: it echoed 1e-5, the config's
    value, for an optimizer stepping at 3.7e-4. A test that merely checked
    the echo equalled the config would have passed then and proved nothing.
    """
    t = StubTrainer(pix_gan_lr=1e-5)
    for g in t.pix_optimizer.param_groups:
        g["lr"] = 3.7e-4
    out = {}
    t._maybe_run_pixel_texture_d_updates(_info_flash(), out, current_step=0)
    assert t.config.pix_gan_lr == 1e-5              # the config never moved
    assert out["train/pix_cfg_pix_gan_lr"] == pytest.approx(3.7e-4)


def test_echo_reports_the_optimizers_LIVE_betas_not_the_config(loader_patch):
    """Same knob class, same source. ``pix_gan_betas`` was echoed off cfg
    too, and Adam's momentum state is as re-settable on resume as its lr."""
    t = StubTrainer(pix_gan_betas=(0.0, 0.9))
    for g in t.pix_optimizer.param_groups:
        g["betas"] = (0.5, 0.99)
    out = {}
    t._maybe_run_pixel_texture_d_updates(_info_flash(), out, current_step=0)
    assert tuple(t.config.pix_gan_betas) == (0.0, 0.9)
    assert out["train/pix_cfg_pix_gan_beta1"] == pytest.approx(0.5)
    assert out["train/pix_cfg_pix_gan_beta2"] == pytest.approx(0.99)


def test_echo_surfaces_param_group_disagreement_instead_of_the_first(
        loader_patch):
    """Two param groups at two lrs: the SPLIT is the finding.

    Reporting ``param_groups[0]`` would hide it behind a number that is
    true of exactly one group -- the forged-reading shape again, one level
    down. So the single value is OMITTED and the split is named.
    """
    t = StubTrainer()
    params = [p for p in t.pixel_texture_disc.parameters()
              if p.requires_grad]
    t.pix_optimizer = torch.optim.Adam(
        [{"params": params[:1], "lr": 1e-5},
         {"params": params[1:], "lr": 4e-4}],
        betas=(0.0, 0.9),
    )
    out = {}
    t._maybe_run_pixel_texture_d_updates(_info_flash(), out, current_step=0)
    assert "train/pix_cfg_pix_gan_lr" not in out
    assert out["train/pix_cfg_pix_gan_lr_param_group_disagreement"] == 1.0
    assert out["train/pix_cfg_pix_gan_lr_min"] == pytest.approx(1e-5)
    assert out["train/pix_cfg_pix_gan_lr_max"] == pytest.approx(4e-4)
    assert out["train/pix_cfg_pix_gan_optimizer_param_groups"] == 2.0
    # The betas still AGREE across both groups, so they keep a single
    # honest value -- the omission is per-knob, not a blanket bail.
    assert out["train/pix_cfg_pix_gan_beta1"] == pytest.approx(0.0)
    assert out["train/pix_cfg_pix_gan_beta2"] == pytest.approx(0.9)


def test_echo_omits_the_lr_rather_than_defaulting_when_none_was_built():
    """OMIT WHEN UNCOMPUTED, never default. ``PIX_GAN_LR`` is the
    constructor's fallback, not an observation; printing it when no
    optimizer exists is a reading of something that never happened.

    Exercised on the shipped method (bound by the ``_pix_*`` rule), not a
    copy: the D-loop returns before the echo when there is no optimizer, so
    the helper is where this branch is reachable."""
    t = StubTrainer()
    t.pix_optimizer = None
    h = t._pix_optimizer_hyper()
    assert "pix_gan_lr" not in h
    assert h["pix_gan_lr_unavailable"] is True
    assert h["pix_gan_beta1_unavailable"] is True
    assert h["pix_gan_optimizer_param_groups"] == 0


def test_echo_reports_the_r1_cadence_the_counters_actually_use(loader_patch):
    """``PixCounters`` floors ``every_n`` at 1, so a config ``0`` RUNS at
    cadence 1. The pre-fix echo read the config and reported 0 -- a cadence
    no part of the build ever used."""
    t = StubTrainer(pix_r1_every_n=0)
    out = {}
    t._maybe_run_pixel_texture_d_updates(_info_flash(), out, current_step=0)
    assert t.config.pix_r1_every_n == 0
    assert out["train/pix_cfg_pix_r1_every_n"] == 1.0


def test_echo_is_absent_when_the_gate_is_off(loader_patch):
    t = StubTrainer()
    t.gan_pixel_texture_enabled = False
    out = {}
    t._maybe_run_pixel_texture_d_updates(_info_flash(), out, current_step=0)
    assert not any(k.startswith("train/pix_cfg_") for k in out)


# ===========================================================================
# 11b. The echo must be VISIBLE, not merely correct
#
# The rule-4 instrument was itself an instance of the defect class it was
# built to catch: present, plausible, computed correctly -- and wired to a
# channel that does not reach the run log.
#
# ``trainer/causal_rolling_staircase_train.py`` does
# ``logging.basicConfig(level=INFO if main else WARNING, ...)`` with NO
# ``force=True``. ``basicConfig`` is a documented SILENT NO-OP once the root
# logger already has handlers, so any earlier import that installs one
# leaves root at Python's default WARNING and every ``logging.info`` in the
# trainer disappears. Two full smoke logs had 0 INFO hits with
# WARNING/ERROR unaffected.
#
# Every test in section 11 above checks that the echo COMPUTES the right
# values. All of them passed against the pre-fix code. Computing it is not
# enough -- these check it ARRIVES.
# ===========================================================================
@contextlib.contextmanager
def _root_at_warning(handler_level=logging.WARNING):
    """Put the root logger in the exact state the defect lives in.

    Root at WARNING with a single handler, and stderr captured separately.
    Yields ``(logging_text_buf, stderr_text_buf)``.

    ``handler_level=logging.CRITICAL`` additionally gags the logging leg at
    the HANDLER, which isolates the stderr leg: it also stops
    ``logging.lastResort`` (which only fires when no handler is found, and
    writes to stderr) from smuggling the logging leg into the stderr buffer
    and making the two channels indistinguishable.
    """
    root = logging.getLogger()
    old_handlers = root.handlers[:]
    old_level = root.level
    old_disable = logging.root.manager.disable
    buf_log, buf_err = io.StringIO(), io.StringIO()
    h = logging.StreamHandler(buf_log)
    h.setLevel(handler_level)
    h.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    root.handlers = [h]
    root.setLevel(logging.WARNING)
    logging.disable(logging.NOTSET)
    try:
        with contextlib.redirect_stderr(buf_err):
            yield buf_log, buf_err
    finally:
        root.handlers = old_handlers
        root.setLevel(old_level)
        logging.disable(old_disable)


_ECHO_BANNER = "WP-PIXGAN resolved config"


def test_the_harness_really_reproduces_the_defect():
    """NON-VACUITY for everything below: prove the harness DOES swallow an
    ``logging.info`` line, so a passing visibility test below is evidence
    about the echo and not about a harness that captures everything."""
    with _root_at_warning() as (buf_log, buf_err):
        logging.info("[canary] an INFO line from inside the harness")
        logging.warning("[canary] a WARNING line from inside the harness")
    assert "[canary] an INFO line" not in buf_log.getvalue(), (
        "the harness does not reproduce the production defect -- INFO "
        "survived a WARNING-level root, so no test below proves anything."
    )
    assert "[canary] a WARNING line" in buf_log.getvalue()
    assert buf_err.getvalue() == ""


def test_resolved_echo_reaches_the_log_under_a_warning_level_root(
        loader_patch):
    """THE REGRESSION GUARD. Runs the PRODUCT path -- the real D-update
    entry point, not the echo helper in isolation -- with root at WARNING,
    and requires the resolved values to still arrive.

    Against the pre-fix ``logging.info`` echo this fails on the first
    assert: the record is dropped at the level check and the buffer is
    empty, while every value-level test in section 11 keeps passing.
    """
    t = StubTrainer()
    out = {}
    with _root_at_warning() as (buf_log, buf_err):
        t._maybe_run_pixel_texture_d_updates(
            _info_flash(), out, current_step=0)
    log_text, err_text = buf_log.getvalue(), buf_err.getvalue()

    for chan, text in (("logging", log_text), ("stderr", err_text)):
        assert _ECHO_BANNER in text, (
            f"the resolved-config echo never reached {chan} under a "
            "WARNING-level root -- rule 4 has no instrument in production."
        )
        # Spot-check the two knobs the echo exists for: the memory knob
        # and the §7 banding, by VALUE, not just the banner.
        assert "pix_decode_batch=4" in text, (chan, text)
        assert "pix_band_count=3" in text, (chan, text)

    # And what arrived is the SAME resolved set that reached the metrics --
    # not a parallel re-derivation that could drift from it.
    echoed = [k for k in out if k.startswith("train/pix_cfg_")]
    assert len(echoed) > 30, f"only {len(echoed)} echo keys; scanner stale"
    for k in echoed:
        knob = k[len("train/pix_cfg_"):]
        assert f"{knob}=" in log_text, (
            f"train/{k} reached the metrics but {knob} never reached the "
            "log line -- the two halves of the echo disagree."
        )


def test_resolved_echo_reaches_stderr_even_when_logging_is_gagged(
        loader_patch):
    """The second leg, isolated. A WARNING-level root is the failure we
    KNOW about; a root whose handler drops the record (wrong level, a
    handler someone else installed, a filter) is the one we do not. The
    ``print(..., file=sys.stderr, flush=True)`` leg is what covers it --
    the same pair used at pipeline/action_forcing_training.py:701-702.
    """
    t = StubTrainer()
    out = {}
    with _root_at_warning(handler_level=logging.CRITICAL) as (
            buf_log, buf_err):
        t._maybe_run_pixel_texture_d_updates(
            _info_flash(), out, current_step=0)

    assert _ECHO_BANNER not in buf_log.getvalue(), (
        "the logging leg was supposed to be gagged here; this test is no "
        "longer isolating the stderr leg."
    )
    err_text = buf_err.getvalue()
    assert _ECHO_BANNER in err_text
    assert "pix_decode_batch=4" in err_text
    assert "pix_band_count=3" in err_text


def test_nothing_is_emitted_on_either_channel_when_the_arm_is_off(
        loader_patch):
    """BYTE-IDENTICAL-OFF extends to the new channels: promoting the echo
    to WARNING+stderr must not make an arm-off run start printing."""
    t = StubTrainer()
    t.gan_pixel_texture_enabled = False
    out = {}
    with _root_at_warning() as (buf_log, buf_err):
        t._maybe_run_pixel_texture_d_updates(
            _info_flash(), out, current_step=0)
    assert _ECHO_BANNER not in buf_log.getvalue()
    assert _ECHO_BANNER not in buf_err.getvalue()
    assert buf_err.getvalue() == ""
    assert not any(k.startswith("train/pix_cfg_") for k in out)


def test_the_arm_off_guard_would_have_caught_an_emission(loader_patch):
    """PLANTED-VIOLATION COMPANION for the absence guard above, exercising
    the PRODUCT path: identical harness, identical entry point, arm ON.
    Without this, the guard above would pass just as well if the capture
    were broken and both channels were always empty."""
    t = StubTrainer()
    out = {}
    with _root_at_warning() as (buf_log, buf_err):
        t._maybe_run_pixel_texture_d_updates(
            _info_flash(), out, current_step=0)
    assert _ECHO_BANNER in buf_log.getvalue()
    assert _ECHO_BANNER in buf_err.getvalue()


def test_the_echo_is_main_process_gated_on_the_new_channels(loader_patch):
    """Promoting to stderr must not turn every rank into a printer."""
    t = StubTrainer()
    t.is_main_process = False
    out = {}
    with _root_at_warning() as (buf_log, buf_err):
        t._maybe_run_pixel_texture_d_updates(
            _info_flash(), out, current_step=0)
    assert _ECHO_BANNER not in buf_log.getvalue()
    assert buf_err.getvalue() == ""
    # The METRICS half is rank-independent and must be untouched.
    assert out["train/pix_cfg_pix_band_count"] == 3.0


def test_the_echo_stays_once_per_run_on_the_new_channels(loader_patch):
    """A per-step stderr line on a 100k-step run is not an instrument."""
    t = StubTrainer()
    with _root_at_warning() as (buf_log, buf_err):
        for s in range(3):
            t._maybe_run_pixel_texture_d_updates(
                _info_flash(), {}, current_step=s)
    assert buf_log.getvalue().count(_ECHO_BANNER) == 1
    assert buf_err.getvalue().count(_ECHO_BANNER) == 1


# --- static half: the level cannot silently drift back to INFO ------------
def _emit_channels(src, fname):
    """``(logging levels used, count of print(..., file=sys.stderr))`` for
    the named function in ``src``."""
    for node in ast.walk(ast.parse(src)):
        if not (isinstance(node, ast.FunctionDef) and node.name == fname):
            continue
        levels, n_stderr = set(), 0
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            f = sub.func
            if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                    and f.value.id in ("logging", "_logging")):
                levels.add(f.attr)
            if isinstance(f, ast.Name) and f.id == "print":
                for kw in sub.keywords:
                    if (kw.arg == "file"
                            and isinstance(kw.value, ast.Attribute)
                            and kw.value.attr == "stderr"):
                        n_stderr += 1
        return levels, n_stderr
    raise AssertionError(f"{fname} not found in the scanned source")


def test_the_survivable_emitter_carries_both_channels():
    levels, n_stderr = _emit_channels(_trainer_src(), "_pix_emit_actionable")
    assert "warning" in levels, levels
    assert "info" not in levels, levels
    assert n_stderr == 1, (
        "the stderr leg is what survives a root whose handler drops the "
        "record; logging alone does not cover that case."
    )


def test_the_echo_does_not_emit_at_info():
    levels, _n = _emit_channels(
        _trainer_src(), "_pix_echo_resolved_config")
    assert "info" not in levels, (
        "the resolved-config echo emits at INFO again. INFO does not reach "
        "the run logs when a root handler is installed before "
        "basicConfig -- the echo would be computed and never seen."
    )


def test_the_channel_scanner_really_flags_an_info_only_echo():
    """PLANTED VIOLATION for the two guards above: the pre-fix spelling,
    verbatim, proven to be caught."""
    src = textwrap.dedent('''
        def _pix_echo_resolved_config(self, logs, *, resolved):
            logging.info(
                "[ActionForcing] WP-PIXGAN resolved config: %s", resolved,
            )
    ''')
    levels, n_stderr = _emit_channels(src, "_pix_echo_resolved_config")
    assert levels == {"info"} and n_stderr == 0


# ===========================================================================
# 12. ONE RESOLUTION POINT — no pix_* knob may be resolved twice
# ===========================================================================
# ``_pix_resolve_cfg`` exists to be the SINGLE resolve-and-validate point for
# the whole ``pix_*`` block, and the constructor calls it so a misconfigured
# arm fails early rather than inside whichever consumer reaches its own copy
# first. That guarantee is worth nothing unless something enforces it: the
# A21 blocker was a second ``getattr(cfg, "pix_real_pool_windows", 2048)``
# in ``_pix_pool_fill`` while the config said 4096, so a missing key would
# have silently halved A21's ceiling at the ONE site that enforces it while
# the validated value, the margin warning and the resolved-config echo all
# still said 4096. A knob resolved in two places is the recurring defect in
# this package, not a one-off, so it is scanned for structurally.
PIX_RESOLVERS = ("_pix_resolve_cfg", "_pix_resolve_r1_gamma")

_UNKNOWN = object()


def _pix_eval_default(node):
    """Best-effort VALUE of a ``getattr`` default expression.

    Constants, module-level names from ``model.pixel_texture_disc`` and
    ``self.PIX_*`` class constants all resolve; anything else returns
    ``_UNKNOWN`` and is simply not compared (an unknown default is not
    evidence of agreement OR disagreement).
    """
    import model.pixel_texture_disc as _ptd
    if node is None:
        return _UNKNOWN
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _pix_eval_default(node.operand)
        return _UNKNOWN if inner is _UNKNOWN else -inner
    if isinstance(node, (ast.Tuple, ast.List)):
        vals = [_pix_eval_default(e) for e in node.elts]
        return _UNKNOWN if any(v is _UNKNOWN for v in vals) else tuple(vals)
    if isinstance(node, ast.Name):
        return getattr(_ptd, node.id, _UNKNOWN)
    if isinstance(node, ast.Attribute):
        return getattr(Trainer, node.attr, _UNKNOWN)
    return _UNKNOWN


def pix_getattr_sites(src):
    """Every ``getattr(<obj>, "pix_...", <default>)`` in ``src``, tagged with
    its INNERMOST enclosing function.

    Innermost matters: ``ast.walk`` from an outer function would attribute a
    nested helper's reads to the outer one and let a duplicate hide inside a
    closure defined in the resolver.
    """
    out = []

    def _collect(node, fname):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2):
            k = node.args[1]
            if (isinstance(k, ast.Constant) and isinstance(k.value, str)
                    and k.value.startswith("pix_")):
                dflt = node.args[2] if len(node.args) > 2 else None
                out.append((fname, k.value, _pix_eval_default(dflt),
                            node.lineno))
        for ch in ast.iter_child_nodes(node):
            nf = (ch.name
                  if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef))
                  else fname)
            _collect(ch, nf)

    _collect(ast.parse(src), "<module>")
    return out


def pix_resolution_report(src):
    """``(strays, disagreements)`` for one trainer source text.

    ``strays``        -- reads of a RESOLVER-OWNED key from outside the
                         resolvers, i.e. a second resolution point for a
                         knob whose value was already resolved+validated.
    ``disagreements`` -- any key whose ``getattr`` fallbacks do not all
                         carry the same value, resolver-owned or not. This
                         is the hazard in its general form: two resolution
                         points are only dangerous because they CAN
                         disagree, and a key read in three places with
                         three defaults is the same defect without the
                         resolver being involved at all.
    """
    sites = pix_getattr_sites(src)
    owned = {k for (f, k, _d, _l) in sites if f in PIX_RESOLVERS}
    strays = sorted(
        (k, f, ln) for (f, k, _d, ln) in sites
        if k in owned and f not in PIX_RESOLVERS
    )
    by_key = {}
    for (_f, k, d, _l) in sites:
        if d is not _UNKNOWN:
            by_key.setdefault(k, set()).add(repr(d))
    disagreements = {k: sorted(v) for k, v in by_key.items() if len(v) > 1}
    return strays, disagreements


def _trainer_src():
    with open(inspect.getsourcefile(CAFT)) as fh:
        return fh.read()


def test_scanner_actually_sees_the_pix_block(loader_patch):
    """NON-VACUITY: a scanner that silently matched nothing would make every
    assertion below pass. Pin that it finds the block and that the resolver
    really owns the keys the other tests here reason about."""
    sites = pix_getattr_sites(_trainer_src())
    assert len(sites) > 20, f"scanner found only {len(sites)} pix_* getattrs"
    owned = {k for (f, k, _d, _l) in sites if f in PIX_RESOLVERS}
    for k in ("pix_real_pool_windows", "pix_lat_frames_per_crop",
              "pix_band_count", "pix_crops_per_step", "pix_r1_gamma"):
        assert k in owned, f"{k} is not resolved in {PIX_RESOLVERS}"


def test_no_pix_knob_has_a_second_resolution_point():
    strays, _ = pix_resolution_report(_trainer_src())
    assert strays == [], (
        "these pix_* knobs are resolved a SECOND time outside "
        f"{PIX_RESOLVERS}: {strays}. _pix_resolve_cfg validated a value for "
        "each of them; a consumer that re-reads the key with its own "
        "getattr default silently wins whenever the key is absent, and "
        "nothing reconciles the two. Read the resolved value instead."
    )


def test_no_pix_knob_has_disagreeing_defaults():
    _, disagreements = pix_resolution_report(_trainer_src())
    assert disagreements == {}, (
        f"these pix_* knobs carry more than one getattr default: "
        f"{disagreements}. Whichever site runs first decides."
    )


def test_each_resolver_owned_knob_is_read_exactly_once():
    """Even INSIDE the resolver, one key = one read. Two reads there could
    disagree with each other just as easily."""
    sites = pix_getattr_sites(_trainer_src())
    owned = [(f, k) for (f, k, _d, _l) in sites if f in PIX_RESOLVERS]
    dupes = sorted({k for (_f, k) in owned
                    if sum(1 for (_g, j) in owned if j == k) > 1})
    assert dupes == [], f"resolved more than once inside the resolver: {dupes}"


def test_planted_second_resolution_point_fires():
    """PLANTED VIOLATION: restore the exact A21 defect this fix removed --
    a second ``pix_real_pool_windows`` resolution inside ``_pix_pool_fill``
    -- and prove the scanner reports it. Planted into the REAL trainer
    source, through the same ``pix_resolution_report`` the guard above
    calls, so the companion cannot pass on source the guard never sees."""
    src = _trainer_src()
    good = 'cap = max(int(n_new), int(self._pix_resolve_cfg()["pool_windows"]))'
    assert src.count(good) == 1, "consolidated pool cap moved; re-aim the plant"
    bad = ('cap = max(int(n_new), int(getattr(\n'
           '            self.config, "pix_real_pool_windows", 2048)))')
    strays, disagreements = pix_resolution_report(src.replace(good, bad))
    assert any(k == "pix_real_pool_windows" and f == "_pix_pool_fill"
               for (k, f, _ln) in strays), strays
    # 2048 vs the resolver's 2048 AGREE, so the value-disagreement check
    # alone would NOT have caught this one. The two checks are independent
    # and both are needed: a second resolution point is a defect even when
    # today's defaults happen to match, because the config says 4096.
    assert "pix_real_pool_windows" not in disagreements


def test_planted_disagreeing_default_fires():
    """PLANTED VIOLATION for the other half: same key, two different
    fallbacks. Nothing about this needs a resolver to be a bug.

    RE-AIMED off ``pix_seed``. That key now has exactly ONE getattr in the
    trainer -- the resolver's -- because the two generators and the
    resolved-config echo were routed through ``rc``, and a single-site key
    cannot carry a two-default plant at all. ``pix_finish_grad_enabled`` is
    the remaining multi-site key and is deliberately still multi-site:
    ``_build_pipeline`` reads it OUTSIDE the pixel gate, where the
    validating resolver must never be reached, so it cannot become
    resolver-owned. The guard is unchanged; only the plant moved.
    """
    src = _trainer_src()
    # The FULL call text, not just the key+default: a prose comment nearby
    # quotes ``getattr(self, "pix_finish_grad_enabled", False)`` verbatim,
    # and a plant that lands in a comment changes no AST and would make
    # this companion silently vacuous -- the exact failure the companion
    # exists to rule out.
    tok = 'getattr(self.config, "pix_finish_grad_enabled", False)'
    assert src.count(tok) >= 2, src.count(tok)
    planted = src.replace(
        tok, 'getattr(self.config, "pix_finish_grad_enabled", True)', 1)
    _strays, disagreements = pix_resolution_report(planted)
    assert "pix_finish_grad_enabled" in disagreements, disagreements
    assert disagreements["pix_finish_grad_enabled"] == ["False", "True"]


# ===========================================================================
# 12b. THE ECHO MAY NOT RE-READ ``cfg``
#
# Same defect SHAPE as a second resolution point, one level meaner. A
# consumer that re-reads a knob can disagree with the resolver; the ECHO
# that re-reads a knob is the artefact a reader trusts to settle what ran,
# so a drift there is the one drift guaranteed not to be noticed. Structural,
# for the same reason section 12 is: this package keeps producing it.
# ===========================================================================
# Knobs the echo still sources from ``cfg``, each because its CONSUMER
# cannot route through the resolver. FROZEN: an addition is a new drift
# surface and must be argued for; a removal means one was routed and the
# list is stale.
#
#   pix_gan_weight    -- authoritatively resolved by
#                        model.pixel_texture_disc.resolve_gan_weight(
#                        strict=True) inside ``_pix_gen_weight``. A second,
#                        laxer copy in ``_pix_resolve_cfg`` would BE the
#                        defect section 12 forbids, and the echo reports
#                        only calibrated-ness, never a value.
#   pix_poscontrol_*  -- ``_pix_positive_control`` must do NO work when off
#                        (§8.1: not a tensor, not a decode, not an RNG
#                        draw). A whole-block resolve above its gate is work.
PIX_ECHO_CFG_READS_ALLOWED = {
    "pix_gan_weight",
    "pix_poscontrol_every",
    "pix_poscontrol_amplitude",
    "pix_poscontrol_boot",
}

# ``pix_gan_lr`` / ``pix_gan_betas`` are not resolver-owned and must not
# become so: their resolution point is the OPTIMIZER. ``_build_optimizer``
# is the one place allowed to turn the config into a number; every later
# reader takes it off ``param_groups``.
PIX_OPTIMIZER_OWNED = ("pix_gan_lr", "pix_gan_betas")


def pix_echo_report(src):
    """``(cfg_read_keys, n_entries)`` for the resolved-config echo site.

    ``cfg_read_keys`` -- every ``pix_*`` key the echo resolves with a
    ``getattr`` of its OWN rather than taking the value from the resolution
    point (``rc``), from the optimizer, or from a local the same function
    already computed.
    """
    keys, n_entries = set(), 0
    for node in ast.walk(ast.parse(src)):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_pix_echo_resolved_config"):
            continue
        for kw in node.keywords:
            if isinstance(kw.value, ast.Dict):
                n_entries += len(kw.value.keys)
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
                    and sub.func.id == "getattr" and len(sub.args) >= 2):
                k = sub.args[1]
                if (isinstance(k, ast.Constant) and isinstance(k.value, str)
                        and k.value.startswith("pix_")):
                    keys.add(k.value)
    return sorted(keys), n_entries


def test_echo_scanner_actually_sees_the_echo():
    """NON-VACUITY: a scanner that matched nothing would make both guards
    below pass forever."""
    _keys, n = pix_echo_report(_trainer_src())
    assert n > 30, f"echo scanner found only {n} entries"


def test_the_echo_does_not_resolve_pix_knobs_of_its_own():
    keys, _n = pix_echo_report(_trainer_src())
    assert set(keys) == PIX_ECHO_CFG_READS_ALLOWED, (
        "the resolved-config echo reads these pix_* knobs off cfg: "
        f"{keys}, expected exactly {sorted(PIX_ECHO_CFG_READS_ALLOWED)}. "
        "An UNEXPECTED key means the echo resolved a knob a second time -- "
        "it must read rc[...], the optimizer, or a local the D-loop already "
        "computed. A MISSING key means one was routed and this frozen list "
        "is stale; shrink it and say why in the comment above."
    )


def test_optimizer_owned_knobs_are_read_only_where_the_optimizer_is_built():
    """``pix_gan_lr``/``pix_gan_betas`` have exactly one config read: the
    construction site. Anything else must read ``param_groups``, because
    the config stops being the answer the moment the optimizer exists."""
    sites = pix_getattr_sites(_trainer_src())
    seen = [(f, k, ln) for (f, k, _d, ln) in sites
            if k in PIX_OPTIMIZER_OWNED]
    assert seen, "scanner found no pix_gan_lr/pix_gan_betas reads at all"
    stray = sorted((k, f, ln) for (f, k, ln) in seen if f != "_build_optimizer")
    assert stray == [], (
        f"these read an OPTIMIZER-owned knob off cfg outside "
        f"_build_optimizer: {stray}. After construction the live value "
        "lives in pix_optimizer.param_groups -- a resume, a scheduler or a "
        "manual edit moves it there and never touches the config."
    )


def test_planted_echo_cfg_read_fires():
    """PLANTED VIOLATION for both guards above: restore the exact pre-fix
    line this work package removed -- ``pix_gan_lr`` echoed off ``cfg`` --
    into the REAL trainer source, and prove both reports name it. Without
    this the two guards could be passing on a scanner that sees nothing."""
    src = _trainer_src()
    good = "**self._pix_optimizer_hyper(),"
    assert src.count(good) == 1, "the echo's optimizer splice moved; re-aim"
    bad = '"pix_gan_lr": float(getattr(cfg, "pix_gan_lr", 1e-5)),'
    planted = src.replace(good, bad)

    keys, _n = pix_echo_report(planted)
    assert "pix_gan_lr" in keys, keys
    assert set(keys) != PIX_ECHO_CFG_READS_ALLOWED

    sites = pix_getattr_sites(planted)
    assert any(k == "pix_gan_lr"
               and f == "_maybe_run_pixel_texture_d_updates"
               for (f, k, _d, _ln) in sites), "ownership guard missed it"
