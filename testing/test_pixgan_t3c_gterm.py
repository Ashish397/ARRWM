"""WP-PIXGAN (B1) / T3-C — ``grad_at``, the pixel G-term, the UNWEIGHTED
ratio, R1's gradient SHARE, and the byte-identical-off guarantee.

Scope of the chunk under test (nothing else):
  * module-level ``grad_at`` — VECTORS, never ratios; ``None``, never 0.0;
  * the A7 grad-telemetry block refactored onto it, byte-identically;
  * ``_pix_emit_grad_ratio`` — the unweighted ``pix_gan_grad_ratio``, and
    its OMISSION (never ``else 0.0``) when the denominator is zero;
  * ``_pix_gen_weight`` — ``resolve_gan_weight(strict=True)``, so an
    uncalibrated ``pix_gan_weight`` is LOUD;
  * ``_pix_g_snapshot_disc`` — the reason the G-term survives the D-loop's
    in-place optimizer step later in the same iteration;
  * ``_pix_r1_grad_share`` — R1's SHARE of the D-side gradient;
  * ``_pix_band_histogram``;
  * the appended ``pix_*`` config block and the emptied
    ``_OVERRIDE_GUARD_KEYS``.

CPU-ONLY.  There is no GPU on the build node and the trainer module cannot
be imported on a compute node either (``from pipeline import ...`` resolves
to "unknown location" there), so CPU is the only option for this suite and
the result is labelled as such rather than claimed as GPU-verified.

Run (the thread caps are NOT optional — nproc=144 here and an uncapped run
looks hung for >30 min):

    cd /scratch/u6ex/as1748.u6ex/ARRWM
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONPATH=. \
      /scratch/u6ex/as1748.u6ex/miniforge3/envs/arrwm/bin/python \
      -m pytest -q testing/test_pixgan_t3c_gterm.py
"""
import os
import random
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.pixel_texture_disc import (  # noqa: E402
    PixelTextureDisc, g_loss, resolve_gan_weight,
)

# Wan's T5 wrapper evaluates ``torch.cuda.current_device()`` at import time
# (idiom borrowed verbatim from testing/test_pixgan_trainer_wiring.py).
with patch.object(torch.cuda, "current_device", return_value=0):
    from trainer import causal_action_forcing_train as CAFT

grad_at = CAFT.grad_at
Trainer = CAFT.ActionForcingDMDTrainer

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_AF_SRC_PATH = os.path.join(_ROOT, "trainer", "causal_action_forcing_train.py")
_CFG_PATH = os.path.join(_ROOT, "configs", "action_forcing_phase3_dmd.yaml")


def _af_source():
    with open(_AF_SRC_PATH, encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# 1. ``grad_at`` — the signature WP-14B codes against.
# ---------------------------------------------------------------------------
def test_grad_at_returns_flattened_gradient_vector():
    """A VECTOR, flattened, matching autograd — not a norm, not a ratio."""
    p = torch.randn(3, 4, requires_grad=True)
    x = torch.randn(5, 3)
    loss = (x @ p).pow(2).sum()
    got = grad_at(loss, p, retain_graph=True)
    ref = torch.autograd.grad(loss, p, retain_graph=True)[0]
    assert got is not None
    assert got.dim() == 1 and got.numel() == p.numel()
    assert got.dtype == torch.float32
    assert torch.allclose(got, ref.flatten().float(), atol=0, rtol=0)
    # Detached: a returned vector must not extend anybody's graph.
    assert not got.requires_grad


def test_grad_at_accepts_a_single_tensor_and_an_iterable():
    """``params`` must accept a SINGLE Tensor — the A7 probe passes one."""
    a = torch.randn(2, 2, requires_grad=True)
    b = torch.randn(3, requires_grad=True)
    loss = a.sum() * 2.0 + b.sum() * 3.0
    one = grad_at(loss, a, retain_graph=True)
    many = grad_at(loss, [a, b], retain_graph=True)
    assert one.numel() == 4
    assert many.numel() == 7
    # concatenated IN THE ORDER GIVEN
    assert torch.allclose(many[:4], one)
    assert torch.allclose(many[4:], torch.full((3,), 3.0))


def test_grad_at_returns_none_not_zero_on_unused_param():
    """The forgeable-zero rule, at the helper level (WP_PIXGAN §16/§21)."""
    used = torch.randn(4, requires_grad=True)
    unused = torch.randn(4, requires_grad=True)
    loss = used.sum()
    got = grad_at(loss, unused, retain_graph=True)
    assert got is None, "must be None; 0.0 is indistinguishable from 'no contribution'"
    # and it must not be some falsy tensor either
    assert not isinstance(got, torch.Tensor)
    # ANY unused param in a multi-param request poisons the whole call:
    # a partially filled vector is not layout-comparable with the other
    # loss's vector, and zero-filling the hole re-imports the forged zero.
    assert grad_at(loss, [used, unused], retain_graph=True) is None


def test_grad_at_returns_none_on_graph_free_loss():
    assert grad_at(torch.tensor(1.0), torch.randn(2, requires_grad=True)) is None
    assert grad_at(3.0, torch.randn(2, requires_grad=True)) is None


def test_grad_at_does_not_compute_a_ratio():
    """Guard the CONTRACT: the helper must not reduce to a scalar.

    Two ratio-returning calls would each re-backprop the shared DMD term;
    on the LADD path the second pass runs a checkpointed teacher recompute
    (~11 s / ~37 GiB). Returning vectors is what lets ONE backward serve
    every ratio and every cosine.
    """
    p = torch.randn(6, requires_grad=True)
    v = grad_at((p * 2).sum(), p, retain_graph=True)
    assert v.numel() == 6, "a scalar here would mean the helper reduced"


def test_one_backward_of_the_shared_term_serves_every_consumer():
    """The DMD vector is taken ONCE and reused for both ratio and cosine."""
    calls = {"n": 0}
    real_grad = torch.autograd.grad

    def counting_grad(*a, **k):
        calls["n"] += 1
        return real_grad(*a, **k)

    p = torch.randn(8, requires_grad=True)
    dmd = (p * 3.0).sum()
    gan = (p * -1.0).sum()
    pix = (p * 0.5).sum()
    with patch.object(torch.autograd, "grad", counting_grad):
        g_dmd = grad_at(dmd, p, retain_graph=True)      # ONE backward
        g_gan = grad_at(gan, p, retain_graph=True)
        g_pix = grad_at(pix, p, retain_graph=True)
    assert calls["n"] == 3, "one backward PER LOSS, never per readout"

    # four readouts, still three backwards
    n_d, n_g, n_p = (float(v.norm()) for v in (g_dmd, g_gan, g_pix))
    assert abs(n_g / n_d - float(gan.grad_fn is not None)) >= 0.0  # smoke
    cos_gan = float(torch.dot(g_gan, g_dmd) / (n_g * n_d))
    cos_pix = float(torch.dot(g_pix, g_dmd) / (n_p * n_d))
    assert cos_gan == pytest.approx(-1.0, abs=1e-6)
    assert cos_pix == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. The A7 refactor is byte-identical.
# ---------------------------------------------------------------------------
_A7_START = "            _tel_n = int(getattr(self.config,\n"
_A7_END = '                    out["train/gan_grad_telemetry_err"] = 1.0\n'


def _a7_block():
    src = _af_source()
    i = src.index(_A7_START)
    j = src.index(_A7_END, i) + len(_A7_END)
    return textwrap.dedent(src[i:j])


def _a7_original(self, generator_loss, gen_gan_loss, out):
    """The A7 block EXACTLY as it stood before T3-C's refactor."""
    _tel_n = int(getattr(self.config, "gan_grad_telemetry_every", 25) or 0)
    if _tel_n > 0 and int(self.step) % _tel_n == 0:
        try:
            _p_last = None
            for _pn, _pp in self.model.generator.named_parameters():
                if _pp.requires_grad:
                    _p_last = _pp
            if _p_last is not None:
                _g_rest = torch.autograd.grad(
                    generator_loss, _p_last, retain_graph=True,
                    allow_unused=True)[0]
                _g_gan = torch.autograd.grad(
                    gen_gan_loss, _p_last, retain_graph=True,
                    allow_unused=True)[0]
                if _g_rest is not None and _g_gan is not None:
                    _gr = _g_rest.flatten().float()
                    _gg = _g_gan.flatten().float()
                    _nr = float(_gr.norm())
                    _ng = float(_gg.norm())
                    out["train/gan_grad_norm"] = _ng
                    out["train/dmd_grad_norm_shared"] = _nr
                    out["train/gan_dmd_grad_ratio"] = (
                        _ng / _nr if _nr > 0 else 0.0)
                    out["train/gan_dmd_grad_cos"] = float(
                        torch.dot(_gg, _gr) / max(_ng * _nr, 1e-12))
        except Exception:
            out["train/gan_grad_telemetry_err"] = 1.0


class _GenStub(nn.Module):
    def __init__(self, frozen_tail=False):
        super().__init__()
        self.a = nn.Parameter(torch.randn(3, 3))
        self.b = nn.Parameter(torch.randn(4))
        if frozen_tail:
            self.b.requires_grad_(False)


def _a7_stub(every=1, step=0, frozen_tail=False):
    stub = SimpleNamespace(
        config=SimpleNamespace(gan_grad_telemetry_every=every),
        step=step,
        model=SimpleNamespace(generator=_GenStub(frozen_tail=frozen_tail)),
    )
    # The REAL method, bound to the stub -- not a mock. Both telemetry paths
    # must publish through the same helper, so a stub that reimplemented it
    # would test nothing.
    stub._pix_emit_grad_ratio = (
        lambda *a, **k: Trainer._pix_emit_grad_ratio(stub, *a, **k)
    )
    return stub


def _run_shipped_a7(
    stub, generator_loss, gen_gan_loss, pix_raw=None, pix_folded=False,
):
    out = {}
    ns = {
        "self": stub, "out": out, "torch": torch,
        "grad_at": grad_at,
        # WP-14B: the extracted span now also calls ``ladd_unweighted_ratio``
        # (module-level, same pattern as ``grad_at`` above) -- supplied here
        # for the same reason ``grad_at`` is: the block is exec'd in an
        # isolated namespace, so any name it references at module scope
        # must be provided explicitly or it raises inside the ``try`` and
        # is swallowed into ``train/gan_grad_telemetry_err``.
        "ladd_unweighted_ratio": CAFT.ladd_unweighted_ratio,
        # ``_pix_g_folded`` is assigned BEFORE ``_A7_START`` in the real
        # function (outside this extracted span), so it too must be
        # supplied rather than computed here. Default False: none of the
        # existing calls below simulate the pixel-folded interaction.
        "_pix_g_folded": pix_folded,
        "generator_loss": generator_loss,
        "gen_gan_loss": gen_gan_loss,
        "_pix_g_raw": pix_raw,
    }
    exec(compile(_a7_block(), "<a7>", "exec"), ns)
    return out


@pytest.mark.parametrize("frozen_tail", [False, True])
def test_a7_refactor_is_byte_identical(frozen_tail):
    """The SHIPPED block vs the pre-refactor code — same keys, same bits."""
    torch.manual_seed(0)
    stub_new = _a7_stub(frozen_tail=frozen_tail)
    stub_old = _a7_stub(frozen_tail=frozen_tail)
    stub_old.model.generator.load_state_dict(
        stub_new.model.generator.state_dict())

    def losses(stub):
        # Both params enter both losses, so the probe's anchor (the LAST
        # requires_grad param) carries a gradient in BOTH parametrisations.
        g = stub.model.generator
        x = torch.ones(3, 3)
        dmd = (g.a * x).sum() * 2.0 + g.b.sum() * 3.0
        gan = (g.a * x).sum() * -0.25 + g.b.sum() * 0.5
        return dmd, gan

    d_new, g_new = losses(stub_new)
    got = _run_shipped_a7(stub_new, d_new, g_new)
    d_old, g_old = losses(stub_old)
    ref = {}
    _a7_original(stub_old, d_old, g_old, ref)

    # WP-14B added ``train/ladd_gan_grad_*`` keys AFTER T3-C's own refactor
    # (a genuinely new capability, not part of the T3-C-vs-pre-T3-C claim
    # this test pins) -- the stub never sets ``_ladd_last_mode_count`` etc.,
    # so the guard always fires and always emits the "unavailable" pair.
    # Excluded from the equality check for that reason, not asserted on
    # here at all: see testing/test_wp14b_ladd_unweighted_ratio.py for the
    # feature's own coverage, including the guard states this stub happens
    # to hit.
    ladd_new_keys = {k for k in got if k.startswith("train/ladd_gan_grad_")}
    assert set(got) - ladd_new_keys == set(ref)
    for k in ref:
        assert got[k] == ref[k], k
    assert "train/gan_dmd_grad_ratio" in got


def test_a7_error_regime_preserved_on_graph_free_loss():
    """``grad_at`` returns None where ``autograd.grad`` used to RAISE.

    The old block let that raise into the ``except`` and log
    ``gan_grad_telemetry_err=1.0``. Silently converting an error regime
    into a silent one would be a behaviour change, so the refactor
    re-raises explicitly and the recorded key still appears.
    """
    stub = _a7_stub()
    dmd = torch.tensor(3.0)            # no graph
    gan = torch.tensor(1.0)
    got = _run_shipped_a7(stub, dmd, gan)
    ref = {}
    _a7_original(stub, dmd, gan, ref)
    assert got == ref == {"train/gan_grad_telemetry_err": 1.0}


def test_a7_emits_no_pixel_key_when_the_arm_is_off():
    """BYTE-IDENTICAL OFF, at the telemetry block."""
    stub = _a7_stub()
    g = stub.model.generator
    dmd = g.a.sum() * 2.0
    gan = g.a.sum() * -0.25
    got = _run_shipped_a7(stub, dmd, gan, pix_raw=None)
    assert not [k for k in got if "pix" in k]


# ---------------------------------------------------------------------------
# 3. The UNWEIGHTED ratio — correct, and OMITTED (never zeroed).
# ---------------------------------------------------------------------------
def _emit(pix_raw, dmd_vec, param):
    out = {}
    Trainer._pix_emit_grad_ratio(
        SimpleNamespace(), out, pix_raw=pix_raw, dmd_vec=dmd_vec, param=param,
    )
    return out


def test_unweighted_ratio_is_the_ratio_at_weight_one():
    p = torch.randn(10, requires_grad=True)
    dmd = (p * 4.0).sum()
    raw = (p * 0.5).sum()                 # the PRE-weight g_loss
    dmd_vec = grad_at(dmd, p, retain_graph=True)
    out = _emit(raw, dmd_vec, p)
    assert out["train/pix_gan_grad_ratio"] == pytest.approx(0.5 / 4.0)
    assert out["train/pix_gan_grad_norm_unweighted"] == pytest.approx(
        float(grad_at(raw, p, retain_graph=True).norm()))
    assert out["train/pix_gan_grad_cos"] == pytest.approx(1.0, abs=1e-6)


def test_unweighted_ratio_is_weight_free_by_construction():
    """Scaling the APPLIED weight must not move this number (§12 step 1).

    The whole point: at ``pix_gan_weight=0`` the applied term contributes
    no gradient, so only a pre-weight readout can be calibrated against.
    """
    p = torch.randn(10, requires_grad=True)
    dmd_vec = grad_at((p * 4.0).sum(), p, retain_graph=True)
    raw = (p * 0.5).sum()
    r0 = _emit(raw, dmd_vec, p)["train/pix_gan_grad_ratio"]
    # the trainer applies ``raw * weight`` separately; ``raw`` is unchanged
    for w in (0.0, 0.03, 17.0):
        _applied = raw * w                       # noqa: F841 — not measured
        assert _emit(raw, dmd_vec, p)["train/pix_gan_grad_ratio"] == r0


def test_ratio_is_OMITTED_not_zeroed_when_the_denominator_is_zero():
    """A logged 0.0 there would mean the DMD gradient vanished, which is
    the OPPOSITE conclusion from "the GAN term contributes nothing"."""
    p = torch.randn(6, requires_grad=True)
    raw = (p * 0.5).sum()
    out = _emit(raw, torch.zeros(6), p)
    assert "train/pix_gan_grad_ratio" not in out
    assert out["train/pix_gan_grad_denom_zero"] == 1.0
    assert out["train/pix_gan_grad_cos_undefined"] == 1.0
    assert "train/pix_gan_grad_cos" not in out
    # the numerator IS still reported — it was measured
    assert out["train/pix_gan_grad_norm_unweighted"] > 0.0


def test_ratio_omitted_when_denominator_unavailable_or_pix_unused():
    p = torch.randn(6, requires_grad=True)
    raw = (p * 0.5).sum()
    out = _emit(raw, None, p)
    assert "train/pix_gan_grad_ratio" not in out
    assert out["train/pix_gan_grad_denom_unavailable"] == 1.0

    unused = torch.randn(6, requires_grad=True)
    out2 = _emit(raw, torch.ones(6), unused)
    assert out2 == {"train/pix_gan_grad_unavailable": 1.0}
    assert not any(v == 0.0 for v in out2.values())


def test_shipped_A7_line_still_has_its_else_zero_and_ours_does_not():
    """The recorded-run line is UNCHANGED; the new one does not copy it."""
    src = _af_source()
    assert 'out["train/gan_dmd_grad_ratio"] = (\n' in src
    assert "_ng / _nr if _nr > 0 else 0.0)" in src, (
        "the existing A7 line's behaviour must not change — recorded runs "
        "depend on it")
    i = src.index("def _pix_emit_grad_ratio(")
    j = src.index("def _pix_standalone_grad_telemetry(", i)
    body = src[i:j]
    assert "else 0.0" not in body.split('"""')[-1]


# ---------------------------------------------------------------------------
# 4. ``pix_gan_weight`` has no default — absence is LOUD.
# ---------------------------------------------------------------------------
def test_resolve_gan_weight_raises_on_absence():
    with pytest.raises(ValueError):
        resolve_gan_weight(None, strict=True)
    assert resolve_gan_weight(0.0, strict=True) == 0.0     # probe regime
    assert resolve_gan_weight(0.7, strict=True) == 0.7


def _weight_stub(**cfg):
    cfg.setdefault("pix_gan_weight", None)
    return SimpleNamespace(
        config=SimpleNamespace(**cfg),
        gan_disc_start_step=cfg.pop("_start", 0),
        gan_warmup_steps=cfg.pop("_warm", 0),
        gan_warmup_shape="linear",
        _gan_warmup_shape_apply=(
            lambda t: max(0.0, min(1.0, float(t)))),
    )


def test_pix_gen_weight_raises_when_uncalibrated():
    stub = _weight_stub()
    stub.config.pix_gan_weight = None
    with pytest.raises(ValueError):
        Trainer._pix_gen_weight(stub, 100)


def test_pix_gen_weight_zero_is_legal_and_is_the_probe():
    stub = _weight_stub()
    stub.config.pix_gan_weight = 0.0
    assert Trainer._pix_gen_weight(stub, 100) == 0.0


def test_pix_gen_weight_ramp_and_start():
    stub = _weight_stub()
    stub.config.pix_gan_weight = 2.0
    stub.gan_disc_start_step = 10
    stub.gan_warmup_steps = 10
    assert Trainer._pix_gen_weight(stub, 5) == 0.0        # before D starts
    assert Trainer._pix_gen_weight(stub, 15) == pytest.approx(1.0)
    assert Trainer._pix_gen_weight(stub, 25) == pytest.approx(2.0)


def test_the_withdrawn_and_the_ladd_constants_are_not_inherited():
    """§5.5 withdrew 0.03; §24 forbids copying the LADD arms' weights."""
    src = _af_source()
    i = src.index("def _pix_gen_weight(")
    j = src.index("def _compute_pixel_texture_g_loss(", i)
    body = "".join(
        ln for ln in src[i:j].splitlines(True) if not ln.strip().startswith("#")
    )
    for banned in ("0.03", "0.65", "0.80", "0.054"):
        assert f'"pix_gan_weight", {banned}' not in body
    assert 'getattr(self.config, "pix_gan_weight", None)' in body
    assert "resolve_gan_weight" in body and "strict=True" in body


# ---------------------------------------------------------------------------
# 5. The snapshot critic — the defect it exists for, reproduced.
# ---------------------------------------------------------------------------
def test_inplace_optimizer_step_breaks_a_live_critic_graph():
    """MUTATION CONTROL for the snapshot: prove the hazard is real."""
    disc = nn.Conv2d(3, 4, 3)
    opt = torch.optim.Adam(disc.parameters(), lr=1e-3)
    x = torch.randn(1, 3, 16, 16, requires_grad=True)
    loss = disc(x).mean()
    opt.zero_grad(set_to_none=True)
    disc(x.detach()).mean().backward()
    opt.step()                                  # in-place on the weights
    with pytest.raises(RuntimeError, match="inplace operation"):
        loss.backward()


def test_snapshot_survives_a_later_inplace_step_and_leaves_d_grads_clean():
    disc = PixelTextureDisc()
    stub = SimpleNamespace(pixel_texture_disc=disc)
    snap = Trainer._pix_g_snapshot_disc(stub)

    x = torch.randn(2, 3, 64, 64, requires_grad=True)
    raw = g_loss(snap(x), loss_form="nsgan")

    opt = torch.optim.Adam(disc.parameters(), lr=1e-2)
    opt.zero_grad(set_to_none=True)
    g_loss(disc(x.detach()), loss_form="nsgan").backward()
    opt.step()                                  # would poison a live graph

    raw.backward()                              # must NOT raise
    assert x.grad is not None and torch.isfinite(x.grad).all()
    # and the G backward must not have touched the REAL critic's grads
    for p in snap.parameters():
        assert p.grad is None
        assert p.requires_grad is False


def test_snapshot_tracks_the_live_critic_weights():
    disc = PixelTextureDisc()
    stub = SimpleNamespace(pixel_texture_disc=disc)
    first = Trainer._pix_g_snapshot_disc(stub)
    with torch.no_grad():
        for p in disc.parameters():
            p.add_(0.05)
    second = Trainer._pix_g_snapshot_disc(stub)
    assert second is first, "the snapshot module is reused, not reallocated"
    live = dict(disc.state_dict())
    for k, v in second.state_dict().items():
        assert torch.equal(v, live[k]), k


# ---------------------------------------------------------------------------
# 6. R1's SHARE of the D-side gradient.
# ---------------------------------------------------------------------------
class _TinyD(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(4))

    def forward(self, x):
        return x * self.w


def _share_stub(every=1, disc=None):
    return SimpleNamespace(
        config=SimpleNamespace(gan_grad_telemetry_every=every),
        pixel_texture_disc=disc if disc is not None else _TinyD(),
    )


def test_r1_grad_share_matches_the_norms_it_claims_to_compare():
    stub = _share_stub()
    w = stub.pixel_texture_disc.w
    d_term = (w * 3.0).sum()
    r1_term = (w * 1.0).sum()
    logs = Trainer._pix_r1_grad_share(stub, d_term, r1_term, current_step=0)
    n_r1 = float(grad_at(r1_term, w, retain_graph=True).norm())
    n_d = float(grad_at(d_term, w, retain_graph=True).norm())
    assert logs["train/pix_r1_grad_norm"] == pytest.approx(n_r1)
    assert logs["train/pix_d_grad_norm"] == pytest.approx(n_d)
    assert logs["train/pix_r1_grad_share"] == pytest.approx(
        n_r1 / (n_r1 + n_d))
    assert logs["train/pix_r1_d_grad_cos"] == pytest.approx(1.0, abs=1e-6)


def test_r1_grad_share_separates_the_two_opposite_failure_modes():
    """A MAGNITUDE cannot tell 'decorative' from 'pinning'; a share can."""
    stub = _share_stub()
    w = stub.pixel_texture_disc.w
    decorative = Trainer._pix_r1_grad_share(
        stub, (w * 100.0).sum(), (w * 0.01).sum(), current_step=0)
    pinning = Trainer._pix_r1_grad_share(
        stub, (w * 0.01).sum(), (w * 100.0).sum(), current_step=0)
    assert decorative["train/pix_r1_grad_share"] < 0.01
    assert pinning["train/pix_r1_grad_share"] > 0.99
    # the magnitude alone is identical in the two regimes
    assert (decorative["train/pix_r1_grad_norm"]
            == pytest.approx(pinning["train/pix_d_grad_norm"]))


def test_r1_grad_share_omitted_on_zero_denominator_and_off_cadence():
    stub = _share_stub()
    w = stub.pixel_texture_disc.w
    zero = (w * 0.0).sum()
    logs = Trainer._pix_r1_grad_share(stub, zero, zero, current_step=0)
    assert "train/pix_r1_grad_share" not in logs
    assert logs["train/pix_r1_grad_share_denom_zero"] == 1.0

    stub_off = _share_stub(every=25)
    assert Trainer._pix_r1_grad_share(
        stub_off, (w * 2).sum(), (w).sum(), current_step=7) == {}
    stub_never = _share_stub(every=0)
    assert Trainer._pix_r1_grad_share(
        stub_never, (w * 2).sum(), (w).sum(), current_step=0) == {}


def test_r1_grad_share_reports_unavailable_rather_than_zero():
    stub = _share_stub()
    logs = Trainer._pix_r1_grad_share(
        stub, torch.tensor(1.0), torch.tensor(1.0), current_step=0)
    assert logs == {"train/pix_r1_grad_share_unavailable": 1.0}


# ---------------------------------------------------------------------------
# 7. Band histogram.
# ---------------------------------------------------------------------------
def test_band_histogram_is_a_complete_set_and_echoes_the_band_count():
    logs = Trainer._pix_band_histogram(None, [0, 0, 1, 2], 3)
    assert logs["train/pix_band_frac_0"] == pytest.approx(0.5)
    assert logs["train/pix_band_frac_1"] == pytest.approx(0.25)
    assert logs["train/pix_band_frac_2"] == pytest.approx(0.25)
    assert logs["train/pix_band_count"] == 3.0
    assert sum(v for k, v in logs.items() if "band_frac" in k) == \
        pytest.approx(1.0)
    g = Trainer._pix_band_histogram(None, [1], 3, prefix="pix_g_")
    assert set(g) == {
        "train/pix_g_band_frac_0", "train/pix_g_band_frac_1",
        "train/pix_g_band_frac_2", "train/pix_g_band_count",
    }


# ---------------------------------------------------------------------------
# 8. Byte-identical OFF, including the RNG streams.
# ---------------------------------------------------------------------------
def _rng_fingerprint():
    return (
        torch.random.get_rng_state().clone(),
        random.getstate(),
        np.random.get_state()[1].copy(),
    )


def _same_rng(a, b):
    return (torch.equal(a[0], b[0]) and a[1] == b[1]
            and np.array_equal(a[2], b[2]))


def test_g_term_off_touches_nothing_and_consumes_no_rng():
    stub = SimpleNamespace(
        gan_pixel_texture_enabled=False,
        pixel_texture_disc=None,
        config=SimpleNamespace(),
    )
    before = _rng_fingerprint()
    w, raw, logs = Trainer._compute_pixel_texture_g_loss(
        stub, {"anything": 1}, current_step=1234)
    after = _rng_fingerprint()
    assert (w, raw, logs) == (None, None, {})
    assert _same_rng(before, after), "the OFF path drew from a global RNG"


def test_g_term_gate_is_checked_at_the_call_site_too():
    """No tensor is even requested when the arm is off."""
    src = _af_source()
    i = src.index("WP-PIXGAN T3-C -- the pixel G-term, computed at the OUTER")
    j = src.index("if gan_active:", i)
    block = src[i:j]
    assert 'getattr(self, "gan_pixel_texture_enabled", False)' in block
    assert "_pix_g_w = None" in block and "_pix_g_raw = None" in block


def test_g_term_reaches_the_generator_on_both_paths():
    """§22's lesson: verify the PATH BETWEEN the endpoints, not the ends.

    ``gen_gan_loss`` only exists on the ``gan_active`` branch, and T3-A's
    own stub runs the pixel arm with ``gan_enabled=False`` — so a G-term
    wired only into ``gen_gan_loss`` would apply ZERO gradient on exactly
    the configuration the arm ships with.
    """
    src = _af_source()
    assert "gen_gan_loss = gen_gan_loss + _pix_g_w" in src
    i = src.index("if not gan_active:")
    tail = src[i:i + 1200]
    assert "generator_loss = generator_loss + _pix_g_w" in tail
    assert "_pix_standalone_grad_telemetry(" in tail
    assert 'out["train/pix_g_via_gen_gan_loss"]' in src


def test_no_pix_r2_key_anywhere_in_the_trainer():
    assert _pix_r2_violations(_af_source()) == []


def _pix_r2_violations(src):
    """Every ``pix_r2_*`` mention that is CODE (comments/docstrings are
    allowed to say the knob does not exist)."""
    bad = []
    for n, line in enumerate(src.splitlines(), 1):
        s = line.strip()
        if s.startswith("#"):
            continue
        if "pix_r2_" in line:
            bad.append((n, s))
    return bad


def test_the_pix_r2_guard_actually_fires_on_a_planted_violation():
    """A guard nobody has seen trip is not evidence (WP_PIXGAN §14)."""
    planted = 'logs["train/pix_r2_gamma"] = float(x)\n'
    assert _pix_r2_violations(planted)
    assert _pix_r2_violations("# there is no pix_r2_gamma knob\n") == []


# ---------------------------------------------------------------------------
# 9. The config block and the override-guard prune.
# ---------------------------------------------------------------------------
def _cfg_text():
    with open(_CFG_PATH, encoding="utf-8") as fh:
        return fh.read()


def test_config_block_is_appended_at_the_end_and_edits_nothing_above():
    txt = _cfg_text()
    i = txt.index("# WP-PIXGAN (B1) — pixel-space texture PatchGAN.")
    j = txt.index("# WP-14B — independent frozen backbone")
    assert j < i, "the pix block must come AFTER WP-14B's block"
    # every key defined exactly once in the whole file (a duplicate at the
    # end would silently retarget an existing run — WP-14B's own warning)
    keys = [ln.split(":")[0] for ln in txt.splitlines()
            if ln and not ln.startswith((" ", "#")) and ":" in ln]
    assert len(keys) == len(set(keys)), \
        [k for k in keys if keys.count(k) > 1]


def test_config_block_contents():
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(_CFG_PATH)
    assert cfg.gan_pixel_texture_enabled is False
    assert cfg.pix_finish_grad_enabled is False
    assert "pix_gan_weight" not in cfg, \
        "§5.5: no default; absence must stay LOUD"
    assert not [k for k in cfg if str(k).startswith("pix_r2")]
    assert cfg.pix_band_count == 3            # A24 pinned
    assert cfg.pix_reals_per_fake == 1        # A20
    assert cfg.pix_r1_every_n == 1            # §5.4
    assert cfg.pix_poscontrol_every == 0      # §8.1 default OFF
    assert cfg.pix_loss_form in ("nsgan", "hinge")


def test_override_guard_table_is_empty_and_the_scan_sources_the_keys():
    """A registered key is allowlisted whether or not anything reads it."""
    assert CAFT._OVERRIDE_GUARD_KEYS == []
    found = CAFT._scan_config_sourced_keys(CAFT._OVERRIDE_GUARD_PREFIXES, _ROOT)
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(_CFG_PATH)
    for k in [str(k) for k in cfg if str(k).startswith("pix_")]:
        assert k in found, f"{k} is not sourced by any literal read site"
    assert "pix_gan_weight" in found
    assert "gan_pixel_texture_enabled" in found


# ---------------------------------------------------------------------------
# 10. The two remaining T3-C entry points, exercised rather than grepped.
# ---------------------------------------------------------------------------
def test_standalone_grad_telemetry_publishes_the_same_ratio_off_the_a7_path():
    """The GAN-off arm must get the SAME calibration read as the A7 path."""
    stub = _a7_stub(every=1)
    g = stub.model.generator
    gen_loss = g.b.sum() * 4.0
    pix_raw = g.b.sum() * 0.5
    out = {}
    Trainer._pix_standalone_grad_telemetry(
        stub, gen_loss, pix_raw, out, current_step=0)
    assert out["train/pix_gan_grad_ratio"] == pytest.approx(0.5 / 4.0)
    assert out["train/dmd_grad_norm_shared"] == pytest.approx(
        float(grad_at(gen_loss, g.b, retain_graph=True).norm()))
    assert out["train/pix_gan_grad_cos"] == pytest.approx(1.0, abs=1e-6)


def test_standalone_grad_telemetry_is_silent_off_cadence_and_when_disabled():
    stub = _a7_stub(every=25)
    g = stub.model.generator
    out = {}
    Trainer._pix_standalone_grad_telemetry(
        stub, g.b.sum() * 4.0, g.b.sum() * 0.5, out, current_step=7)
    assert out == {}
    stub0 = _a7_stub(every=0)
    out0 = {}
    Trainer._pix_standalone_grad_telemetry(
        stub0, stub0.model.generator.b.sum(), stub0.model.generator.b.sum(),
        out0, current_step=0)
    assert out0 == {}


def test_standalone_grad_telemetry_never_takes_a_run_down():
    stub = _a7_stub(every=1)
    out = {}
    Trainer._pix_standalone_grad_telemetry(
        stub, torch.tensor(1.0), torch.tensor(1.0), out, current_step=0)
    # graph-free losses -> ``grad_at`` returns None -> a DISTINCT reason key,
    # never a 0.0 ratio.
    assert "train/pix_gan_grad_ratio" not in out
    assert out.get("train/pix_gan_grad_unavailable") == 1.0


def test_positive_control_is_off_by_default_and_silent_off_cadence():
    """§8.1 default OFF: not one decode, not one key, no RNG draw."""
    stub = SimpleNamespace(config=SimpleNamespace())
    before = _rng_fingerprint()
    got = Trainer._pix_positive_control(
        stub, torch.zeros(1), torch.zeros(1), None,
        current_step=0, update_idx=0, border=0, k_frames=1, decode_batch=1,
    )
    assert got == {}
    assert _same_rng(before, _rng_fingerprint())

    stub_on = SimpleNamespace(config=SimpleNamespace(pix_poscontrol_every=50))
    # off-cadence step
    assert Trainer._pix_positive_control(
        stub_on, torch.zeros(1), torch.zeros(1), None,
        current_step=7, update_idx=0, border=0, k_frames=1, decode_batch=1,
    ) == {}
    # and only on the FIRST D-update of the step (it is a per-step readout)
    assert Trainer._pix_positive_control(
        stub_on, torch.zeros(1), torch.zeros(1), None,
        current_step=50, update_idx=1, border=0, k_frames=1, decode_batch=1,
    ) == {}
