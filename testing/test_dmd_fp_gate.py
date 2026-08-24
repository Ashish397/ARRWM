"""Tests for the DMD self-fingerprint gate (docs/DMD_FINGERPRINT_PROBE.md).

CPU-only. Every helper under test is exercised as an unbound method
against a stub `self`, so no model, VAE or GPU is constructed.

What these protect:

* **The Class-A / Class-B split.** Measurement params may have defaults;
  CALIBRATION params may not. `dmd_fp_scale=0.15`, `seeds=2`,
  `exponent=1.0`, `min_weight=0.0` were guesses that happened to run, and
  a guessed default that silently works is the failure mode this whole
  arming exercise exists to remove. Arming with any of them unset must
  RAISE and must NAME the missing keys.
* **The m_lo/m_hi response curve.** The previous code used the calibrated
  `m` directly as the weight, which silently assumes the response is
  linear in `m` across the whole of [0,1]. That is an assumption, not a
  measurement. The mapping is now explicit and pinned at both clamp ends.
* **Per-FRAME attenuation.** The measured cliff (align +0.094 at frame 9
  -> -0.530 by frame 17) lives INSIDE one band. A scalar weight averages
  it away. The whole design rests on the tail being attenuated more than
  the head, so that is asserted directly on the gradient.
* **Default-off is byte-identical.** Exact match, not approximate,
  against the pre-gate `F.mse_loss` path.
* **No forgeable zeros.** An unavailable diagnostic is an ABSENT key.
  `w = 0.0` is a real regime ("DMD fully gated off") and must never be
  emitted as a placeholder.

Every test in here was checked to FAIL under a mutation of the mechanism
it claims to cover -- see the docstring of each for the mutation used.

Run:
    OMP_NUM_THREADS=8 PYTHONPATH=. pytest -q testing/test_dmd_fp_gate.py
"""
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with patch.object(torch.cuda, "current_device", return_value=0):
    from model.dmd_action_forcing import ActionForcingDMD

INIT = ActionForcingDMD._init_dmd_fp_knobs
WFROM_M = ActionForcingDMD._dmd_fp_gate_weight_from_m
LOSS = ActionForcingDMD._dmd_loss_with_fp_gate
PROBE_PARAMS = ActionForcingDMD._dmd_fp_probe_params
MISSING = ActionForcingDMD._dmd_fp_missing_calib
CALIB_KEYS = ActionForcingDMD._DMD_FP_CALIB_KEYS
RATCHET_KEYS = ActionForcingDMD._DMD_FP_RATCHET_KEYS
RESET = ActionForcingDMD.reset_dmd_fp_ratchet
OBSERVE = ActionForcingDMD._dmd_fp_ratchet_observe_depth

SHAPE = (1, 6, 4, 8, 8)


class _Stub:
    """A bare object the unbound methods can be called against."""

    # the methods under test call each other through self
    _DMD_FP_CALIB_KEYS = CALIB_KEYS
    _DMD_FP_RATCHET_KEYS = RATCHET_KEYS
    _dmd_fp_opt_float = staticmethod(ActionForcingDMD._dmd_fp_opt_float)
    _dmd_fp_missing_calib = ActionForcingDMD._dmd_fp_missing_calib
    _dmd_fp_require_calib = ActionForcingDMD._dmd_fp_require_calib
    _dmd_fp_gate_weight_from_m = ActionForcingDMD._dmd_fp_gate_weight_from_m
    # ratchet
    reset_dmd_fp_ratchet = ActionForcingDMD.reset_dmd_fp_ratchet
    _dmd_fp_ratchet_observe_depth = (
        ActionForcingDMD._dmd_fp_ratchet_observe_depth)
    _dmd_fp_missing_ratchet_calib = (
        ActionForcingDMD._dmd_fp_missing_ratchet_calib)
    _dmd_fp_require_ratchet_calib = (
        ActionForcingDMD._dmd_fp_require_ratchet_calib)
    _dmd_fp_ratchet_apply = ActionForcingDMD._dmd_fp_ratchet_apply


def _cfg(**kw):
    """A config object shaped like the yaml (missing keys == absent)."""
    return SimpleNamespace(**kw)


def _armed(**over):
    """A stub with a COMPLETE, explicitly-supplied calibration."""
    cfg = dict(
        dmd_fp_every=1,
        dmd_fp_perturb="channel_rot",
        dmd_fp_seeds=3,
        dmd_fp_scale=0.2,
        dmd_fp_off_scale=0.8,
        dmd_fp_m_lo=0.2,
        dmd_fp_m_hi=0.8,
        dmd_fp_gate_exponent=1.0,
        dmd_fp_gate_min_weight=0.0,
        dmd_fp_gate_enabled=True,
    )
    cfg.update(over)
    s = _Stub()
    INIT(s, _cfg(**cfg))
    return s


# ---------------------------------------------------------------------------
# config -> consumer wiring
# ---------------------------------------------------------------------------
def test_class_a_knobs_reach_the_consumer_from_config():
    """`dmd_fp_perturb` / `dmd_fp_seeds` / `dmd_fp_scale` /
    `dmd_fp_off_scale` must arrive at the probe from the CONFIG.

    MUTATION: hardcode `_dmd_fp_probe_params` to return the old
    ("hf_scramble", 0.15, max(0.6, .15*4), 2) -> every assert here fails.
    """
    s = _armed(dmd_fp_perturb="patch_shuffle", dmd_fp_seeds=5,
               dmd_fp_scale=0.33, dmd_fp_off_scale=0.77)
    mode, scale, off_scale, seeds = PROBE_PARAMS(s)
    assert mode == "patch_shuffle"
    assert seeds == 5
    assert scale == pytest.approx(0.33)
    assert off_scale == pytest.approx(0.77)


def test_class_a_keeps_working_defaults_when_absent():
    """Measurement params MAY default -- the probe and the depth study
    have to be able to run before any calibration exists.

    MUTATION: make dmd_fp_every/perturb/seeds default to None -> the
    diagnostic path can no longer start, and this goes red.
    """
    s = _Stub()
    INIT(s, _cfg())
    assert s.dmd_fp_every == 0
    assert s.dmd_fp_perturb == "hf_scramble"
    assert s.dmd_fp_seeds == 2


def test_class_b_knobs_are_none_when_absent_from_config():
    """CALIBRATION params must NOT default. This is the whole point.

    MUTATION: restore `float(getattr(args, "dmd_fp_scale", 0.15))` (the
    pre-fix line) -> dmd_fp_scale comes back 0.15 and this goes red.
    """
    s = _Stub()
    INIT(s, _cfg())
    for k in CALIB_KEYS:
        assert getattr(s, k) is None, f"{k} was silently defaulted"
    assert sorted(MISSING(s)) == sorted(CALIB_KEYS)


def _ref_weight(m, m_lo, m_hi, ex, mw):
    """Independent reference for the ramp, written out longhand so the
    consumer tests are not tautological against the implementation."""
    u = min(1.0, max(0.0, (m - m_lo) / (m_hi - m_lo)))
    return mw + (1.0 - mw) * (u ** ex)


# (key, value, probe m, expected weight) -- each row is chosen so the
# expected weight DIFFERS from what the pre-fix guessed constants
# (m_lo=0, m_hi=1, exponent=1, min_weight=0) would give at that m.
@pytest.mark.parametrize("key,val,probe_m,expect", [
    ("dmd_fp_m_lo", 0.11, 0.15, _ref_weight(0.15, 0.11, 0.8, 1.0, 0.0)),
    ("dmd_fp_m_hi", 0.91, 0.85, _ref_weight(0.85, 0.2, 0.91, 1.0, 0.0)),
    ("dmd_fp_gate_exponent", 2.5, 0.5, _ref_weight(0.5, 0.2, 0.8, 2.5, 0.0)),
    ("dmd_fp_gate_min_weight", 0.37, 0.1, 0.37),
])
def test_every_class_b_curve_knob_reaches_the_consumer(key, val, probe_m, expect):
    """Each Class-B curve value must CHANGE THE GATE'S OUTPUT, not merely
    land on an attribute. Asserting the attribute alone would survive a
    consumer that ignores it -- which is the exact "wired but unconsumed"
    failure this campaign has shipped before.

    MUTATION: hardcode `m_lo, m_hi, ex, mw = 0.0, 1.0, 1.0, 0.0` inside
    `_dmd_fp_gate_weight_from_m` -> every row here goes red.
    """
    s = _armed(**{key: val})
    assert getattr(s, key) == pytest.approx(val)          # config -> attr
    assert WFROM_M(s, probe_m) == pytest.approx(expect)   # attr -> output
    # ...and the pre-fix guessed curve would NOT have produced it
    assert expect != pytest.approx(_ref_weight(probe_m, 0.0, 1.0, 1.0, 0.0))


@pytest.mark.parametrize("key,val,idx", [
    ("dmd_fp_scale", 0.44, 1),
    ("dmd_fp_off_scale", 0.66, 2),
])
def test_every_class_b_scale_knob_reaches_the_probe(key, val, idx):
    """MUTATION: return the literals 0.15 / 0.6 from
    `_dmd_fp_probe_params` -> red."""
    s = _armed(**{key: val})
    assert getattr(s, key) == pytest.approx(val)
    assert PROBE_PARAMS(s)[idx] == pytest.approx(val)


def test_gate_enabled_flag_is_read_from_config():
    """MUTATION: hardcode `self.dmd_fp_gate_enabled = False` -> red."""
    s = _Stub()
    INIT(s, _cfg(dmd_fp_gate_enabled=True))
    assert s.dmd_fp_gate_enabled is True
    s2 = _Stub()
    INIT(s2, _cfg())
    assert s2.dmd_fp_gate_enabled is False


def test_yaml_declares_every_dmd_fp_key():
    """The knobs must exist in the shipped config, null where measured.

    A flag with no yaml key is un-settable in practice: this campaign has
    already shipped consumers nobody could reach.
    """
    import re
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    txt = open(os.path.join(
        root, "configs", "action_forcing_phase3_dmd.yaml")).read()
    for k in ("dmd_fp_every", "dmd_fp_perturb", "dmd_fp_seeds",
              "dmd_fp_gate_enabled") + tuple(CALIB_KEYS):
        assert re.search(rf"^{k}:", txt, re.M), f"{k} missing from yaml"
    for k in CALIB_KEYS:
        assert re.search(rf"^{k}: *null", txt, re.M), (
            f"{k} must ship as null -- it is a MEASURED value")
    assert re.search(r"^dmd_fp_gate_enabled: *false", txt, re.M)
    # and the superseded block must say so
    assert "SUPERSEDED" in txt


# ---------------------------------------------------------------------------
# fail-loud: armed without measurement
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("missing_key", list(CALIB_KEYS))
def test_armed_with_any_class_b_unset_raises_and_names_it(missing_key):
    """Arming on an invented constant must be structurally impossible.

    MUTATION: drop the `_dmd_fp_require_calib` call from
    `_dmd_loss_with_fp_gate` -> no raise, and every parametrisation here
    goes red.
    """
    s = _armed(**{missing_key: None})
    s._last_dmd_fp_w_per_frame = torch.ones(SHAPE[1])
    x = torch.randn(*SHAPE, requires_grad=True)
    with pytest.raises(ValueError) as ei:
        LOSS(s, x, torch.randn(*SHAPE),
             torch.ones(SHAPE, dtype=torch.bool), {})
    msg = str(ei.value)
    assert missing_key in msg, f"error does not name {missing_key}: {msg}"
    assert "dmd_fp_depth_study" in msg, "error must point at the study"


def test_armed_with_nothing_measured_names_all_six():
    s = _armed(**{k: None for k in CALIB_KEYS})
    s._last_dmd_fp_w_per_frame = torch.ones(SHAPE[1])
    with pytest.raises(ValueError) as ei:
        LOSS(s, torch.randn(*SHAPE), torch.randn(*SHAPE),
             torch.ones(SHAPE, dtype=torch.bool), {})
    for k in CALIB_KEYS:
        assert k in str(ei.value)


def test_calibration_error_precedes_missing_weight_error():
    """Both failures can be live at once; the CALIBRATION one is the
    worse and is the message the operator must see."""
    s = _armed(dmd_fp_m_lo=None)
    s._last_dmd_fp_w_per_frame = None
    with pytest.raises(ValueError, match="dmd_fp_m_lo"):
        LOSS(s, torch.randn(*SHAPE), torch.randn(*SHAPE),
             torch.ones(SHAPE, dtype=torch.bool), {})


def test_armed_and_calibrated_but_no_probe_ran_still_raises():
    """A fully-calibrated gate with no measured weights would be a
    silently UN-attenuated 'attenuated' DMD."""
    s = _armed()
    s._last_dmd_fp_w_per_frame = None
    with pytest.raises(ValueError, match="no per-frame fingerprint"):
        LOSS(s, torch.randn(*SHAPE), torch.randn(*SHAPE),
             torch.ones(SHAPE, dtype=torch.bool), {})


@pytest.mark.parametrize("key", ["dmd_fp_scale", "dmd_fp_off_scale"])
def test_probe_requires_explicit_scales_and_never_substitutes(key):
    """PROBE-ONLY operation (gate off) must still refuse to invent a
    perturbation magnitude. Chosen over an internal sweep: the sweep is
    the study's job, and a trainer-side one would make the probe's own
    output depend on a grid nobody chose.

    MUTATION: `float(getattr(self, "dmd_fp_scale", 0.15))` -> no raise,
    red.
    """
    s = _armed(dmd_fp_gate_enabled=False, **{key: None})
    with pytest.raises(ValueError) as ei:
        PROBE_PARAMS(s)
    assert key in str(ei.value)
    assert "0.15" in str(ei.value)   # names the guess it refuses to reuse


@pytest.mark.parametrize("lo,hi,exp,mw,pat", [
    (0.8, 0.2, 1.0, 0.0, "must exceed"),
    (0.5, 0.5, 1.0, 0.0, "must exceed"),
    (-0.1, 0.8, 1.0, 0.0, r"\[0, 1\]"),
    (0.2, 1.5, 1.0, 0.0, r"\[0, 1\]"),
    (0.2, 0.8, 0.0, 0.0, "exponent"),
    (0.2, 0.8, -1.0, 0.0, "exponent"),
    (0.2, 0.8, 1.0, 1.5, "min_weight"),
])
def test_incoherent_calibration_raises(lo, hi, exp, mw, pat):
    """Measured-but-nonsensical values must not be accepted either."""
    s = _armed(dmd_fp_m_lo=lo, dmd_fp_m_hi=hi,
               dmd_fp_gate_exponent=exp, dmd_fp_gate_min_weight=mw)
    with pytest.raises(ValueError, match=pat):
        WFROM_M(s, 0.5)


# ---------------------------------------------------------------------------
# the response curve
# ---------------------------------------------------------------------------
def test_mapping_clamps_at_both_ends():
    """m <= m_lo -> min_weight; m >= m_hi -> 1.0. Both ends, explicitly.

    MUTATION: `return m` (the pre-fix behaviour, which used m directly as
    the weight) -> w(0.0)=0.0 != 0.25 and w(0.9)=0.9 != 1.0, red.
    """
    s = _armed(dmd_fp_m_lo=0.3, dmd_fp_m_hi=0.7,
               dmd_fp_gate_min_weight=0.25)
    assert WFROM_M(s, 0.0) == pytest.approx(0.25)
    assert WFROM_M(s, 0.3) == pytest.approx(0.25)
    assert WFROM_M(s, 0.7) == pytest.approx(1.0)
    assert WFROM_M(s, 0.9) == pytest.approx(1.0)
    assert WFROM_M(s, 1.0) == pytest.approx(1.0)


def test_mapping_is_linear_in_the_window_at_exponent_one():
    s = _armed(dmd_fp_m_lo=0.2, dmd_fp_m_hi=0.8,
               dmd_fp_gate_exponent=1.0, dmd_fp_gate_min_weight=0.0)
    assert WFROM_M(s, 0.5) == pytest.approx(0.5)
    assert WFROM_M(s, 0.35) == pytest.approx(0.25)
    assert WFROM_M(s, 0.65) == pytest.approx(0.75)


def test_exponent_bends_the_ramp():
    """MUTATION: ignore `dmd_fp_gate_exponent` -> w stays 0.5, red."""
    s = _armed(dmd_fp_m_lo=0.0, dmd_fp_m_hi=1.0,
               dmd_fp_gate_exponent=2.0, dmd_fp_gate_min_weight=0.0)
    assert WFROM_M(s, 0.5) == pytest.approx(0.25)
    s3 = _armed(dmd_fp_m_lo=0.0, dmd_fp_m_hi=1.0,
                dmd_fp_gate_exponent=0.5, dmd_fp_gate_min_weight=0.0)
    assert WFROM_M(s3, 0.25) == pytest.approx(0.5)


def test_min_weight_floor_is_affine_not_a_clamp():
    """A CLAMPED floor of 0.3 with exponent 2 would flatten every u below
    0.55 onto the floor -- over half the calibrated range dead, silently.
    The affine form keeps the ramp monotone across the whole window.

    MUTATION: `w = clamp(u**exp, min=mw)` -> w(0.5) becomes 0.30, red.
    """
    s = _armed(dmd_fp_m_lo=0.0, dmd_fp_m_hi=1.0,
               dmd_fp_gate_exponent=2.0, dmd_fp_gate_min_weight=0.3)
    assert WFROM_M(s, 0.5) == pytest.approx(0.3 + 0.7 * 0.25)
    # strictly increasing everywhere inside the window
    ws = [WFROM_M(s, v) for v in (0.1, 0.2, 0.3, 0.4, 0.5)]
    assert all(b > a for a, b in zip(ws, ws[1:])), ws


def test_mapping_is_monotone_non_decreasing_in_m():
    """Polarity: MORE manifold-localisation -> MORE DMD. Backwards would
    apply full DMD exactly on the drifted tail, the failure the gate
    exists to prevent."""
    s = _armed(dmd_fp_m_lo=0.2, dmd_fp_m_hi=0.8,
               dmd_fp_gate_exponent=1.7, dmd_fp_gate_min_weight=0.1)
    ws = [WFROM_M(s, v / 20.0) for v in range(21)]
    assert ws == sorted(ws), ws
    assert ws[0] == pytest.approx(0.1)
    assert ws[-1] == pytest.approx(1.0)


def test_mapping_accepts_a_per_frame_tensor():
    s = _armed(dmd_fp_m_lo=0.0, dmd_fp_m_hi=1.0,
               dmd_fp_gate_exponent=1.0, dmd_fp_gate_min_weight=0.0)
    m = torch.tensor([0.0, 0.25, 0.5, 1.0])
    w = WFROM_M(s, m)
    assert torch.is_tensor(w) and w.shape == m.shape
    assert torch.allclose(w, m)


# ---------------------------------------------------------------------------
# the property the design rests on: PER-FRAME attenuation
# ---------------------------------------------------------------------------
def _equal_error_band(w_per_frame):
    """A band whose RAW per-frame squared error is IDENTICAL everywhere,
    so any per-frame difference in the loss is the GATE and nothing else.
    """
    torch.manual_seed(0)
    x = torch.zeros(*SHAPE, requires_grad=True)
    # grad is the DMD update direction; constant magnitude per frame
    g = torch.full(SHAPE, 0.5)
    s = _armed()
    s._last_dmd_fp_w_per_frame = torch.as_tensor(w_per_frame)
    mask = torch.ones(SHAPE, dtype=torch.bool)
    return s, x, g, mask


def test_per_frame_weights_attenuate_the_tail_more_than_the_head():
    """THE property. The measured cliff lives INSIDE one band, so the
    tail's contribution must be strictly smaller than the head's even
    though their raw errors are identical.

    MUTATION: collapse the per-frame vector to its scalar MEAN before
    applying it (`_w = _fpw.mean().expand_as(_fpw)`) -> head and tail
    gradients become equal and this goes red. That mutation is exactly
    the scalar-gate design this replaces.
    """
    w = [1.0, 1.0, 1.0, 0.5, 0.2, 0.05]     # drifted tail
    s, x, g, mask = _equal_error_band(w)
    loss = LOSS(s, x, g, mask, {})
    loss.backward()
    per_frame = x.grad.abs().flatten(2).sum(-1).flatten()
    head, tail = float(per_frame[0]), float(per_frame[-1])
    assert tail < head, f"tail {tail} not attenuated vs head {head}"
    # and it tracks the weights, not just "smaller"
    ratios = [float(per_frame[i] / per_frame[0]) for i in range(SHAPE[1])]
    assert ratios == pytest.approx([wi / w[0] for wi in w], rel=1e-5)


def test_a_fully_closed_tail_frame_contributes_no_gradient():
    w = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    s, x, g, mask = _equal_error_band(w)
    LOSS(s, x, g, mask, {}).backward()
    per_frame = x.grad.abs().flatten(2).sum(-1).flatten()
    assert float(per_frame[-1]) == 0.0
    assert float(per_frame[0]) > 0.0


def test_uniform_weights_reproduce_the_unweighted_loss_value():
    """Sanity on the weighted reduction itself: an all-ones weight is a
    weighted MEAN, so it must equal the unweighted mean in value."""
    s, x, g, mask = _equal_error_band([1.0] * SHAPE[1])
    got = float(LOSS(s, x, g, mask, {}))
    ref = float(0.5 * F.mse_loss(
        x.double()[mask], (x.double() - g.double()).detach()[mask],
        reduction="mean"))
    assert got == pytest.approx(ref, rel=1e-12)


# ---------------------------------------------------------------------------
# default-off must be byte-identical -- the most important safety test
# ---------------------------------------------------------------------------
def test_default_off_is_bitwise_identical_to_the_old_mse_path():
    """EXACT match, not approximate. Off must touch nothing.

    MUTATIONS this bites on (all verified red): float32 instead of double
    in the off path; `.detach()` dropped (caught by the gradient half);
    the mask applied after the reduction; the 0.5 factor dropped.

    MEASURED CAVEAT, recorded rather than assumed: routing the disabled
    path through the WEIGHTED branch with an all-ones weight does NOT
    break this test -- torch's weighted sum/sum and `mse_loss`'s mean
    happen to agree bit-for-bit here. So this test does not by itself
    forbid that refactor; `test_default_off_emits_no_gate_telemetry_at_all`
    does (it goes red on exactly that mutation). The two are needed
    together. The off path is still kept literally separate in the code
    for that reason -- the agreement is an accident of this reduction,
    not a guarantee.
    """
    torch.manual_seed(3)
    s = _Stub()
    INIT(s, _cfg())                       # nothing set: the shipped default
    assert s.dmd_fp_gate_enabled is False
    x = torch.randn(*SHAPE, dtype=torch.float32, requires_grad=True)
    g = torch.randn(*SHAPE, dtype=torch.float32)
    mask = torch.rand(SHAPE) > 0.3
    log = {}
    got = LOSS(s, x, g, mask, log)
    ref = 0.5 * F.mse_loss(
        x.double()[mask],
        (x.double() - g.double()).detach()[mask],
        reduction="mean",
    )
    assert torch.equal(got.detach(), ref.detach()), (
        f"default-off path drifted: {got.item()!r} vs {ref.item()!r}")
    # gradients too, not only the value
    gx = torch.autograd.grad(got, x, retain_graph=True)[0]
    gr = torch.autograd.grad(ref, x)[0]
    assert torch.equal(gx, gr)


def test_default_off_emits_no_gate_telemetry_at_all():
    s = _Stub()
    INIT(s, _cfg())
    log = {}
    LOSS(s, torch.randn(*SHAPE, requires_grad=True), torch.randn(*SHAPE),
         torch.ones(SHAPE, dtype=torch.bool), log)
    assert not [k for k in log if k.startswith("dmd_fp")], log


def test_stale_weights_are_ignored_when_the_gate_is_off():
    """A probe-only run computes weights; with the gate OFF they must not
    reach the loss.

    MUTATION: drop the `dmd_fp_gate_enabled` check and use
    `_last_dmd_fp_w_per_frame` whenever present -> red.
    """
    s = _armed(dmd_fp_gate_enabled=False)
    s._last_dmd_fp_w_per_frame = torch.tensor([1., 1., 1., .1, .1, .1])
    x = torch.randn(*SHAPE, requires_grad=True)
    g = torch.randn(*SHAPE)
    mask = torch.ones(SHAPE, dtype=torch.bool)
    ref = 0.5 * F.mse_loss(x.double()[mask],
                           (x.double() - g.double()).detach()[mask],
                           reduction="mean")
    assert torch.equal(LOSS(s, x, g, mask, {}).detach(), ref.detach())


# ---------------------------------------------------------------------------
# telemetry: visible, and no forgeable zeros
# ---------------------------------------------------------------------------
def test_gate_telemetry_is_stashed_on_the_model_for_the_step_line():
    """dmd_log_dict is wandb-only and never reaches stderr -- the bug
    already caught twice in this campaign. The trainer reads model
    attributes via getattr, so the gate must set them.

    MUTATION: write the values only into `log_dict` -> red.
    """
    w = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    s, x, g, mask = _equal_error_band(w)
    log = {}
    LOSS(s, x, g, mask, log)
    assert s._last_dmd_fp_gate_w_mean == pytest.approx(sum(w) / len(w))
    assert s._last_dmd_fp_gate_w_min == pytest.approx(0.0)
    assert s._last_dmd_fp_gate_w_max == pytest.approx(1.0)
    # SHARE: how much DMD actually survives, as a fraction of 1.0
    assert s._last_dmd_fp_gate_share == pytest.approx(sum(w) / len(w))
    for k in ("dmd_fp_gate_w_mean", "dmd_fp_gate_w_min",
              "dmd_fp_gate_w_max", "dmd_fp_gate_share"):
        assert k in log


def test_share_reports_masked_weights_not_every_frame():
    """The unsupervised frames contribute nothing to the loss; including
    them would report an attenuation that never happened.

    MUTATION: compute the stats over the full expanded `_w` -> the mean
    becomes 0.5 instead of 1.0 and this goes red.
    """
    s = _armed()
    s._last_dmd_fp_w_per_frame = torch.tensor([1., 1., 1., 0., 0., 0.])
    mask = torch.zeros(SHAPE, dtype=torch.bool)
    mask[:, :3] = True                      # only the head is supervised
    log = {}
    LOSS(s, torch.randn(*SHAPE, requires_grad=True), torch.randn(*SHAPE),
         mask, log)
    assert log["dmd_fp_gate_share"] == pytest.approx(1.0)


def test_share_is_one_when_nothing_is_attenuated():
    s, x, g, mask = _equal_error_band([1.0] * SHAPE[1])
    log = {}
    LOSS(s, x, g, mask, log)
    assert log["dmd_fp_gate_share"] == pytest.approx(1.0)


def test_no_forgeable_zeros_before_anything_is_measured():
    """Every telemetry slot starts as None -> the trainer's `is not None`
    check omits the key. 0.0 is a MEANINGFUL value here ('DMD fully gated
    off') and must never be emitted as a placeholder."""
    s = _Stub()
    INIT(s, _cfg())
    for a in ("_last_dmd_fp_m", "_last_dmd_fp_gate_w_mean",
              "_last_dmd_fp_gate_w_min", "_last_dmd_fp_gate_w_max",
              "_last_dmd_fp_gate_share", "_last_dmd_fp_w_per_frame",
              "_last_dmd_fp_m_per_frame", "_last_dmd_fp_calc_idx"):
        assert getattr(s, a) is None, f"{a} was pre-filled"


def test_trainer_step_line_reads_the_fingerprint_attributes():
    """The consumer end of the telemetry wiring: the shared trainer must
    actually look these up. A stashed attribute nobody reads is the same
    silent failure one level along.

    MUTATION: remove the fp_* rows from the trainer's getattr tuple ->
    red.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    txt = open(os.path.join(
        root, "trainer", "causal_action_forcing_train.py")).read()
    for a in ("_last_dmd_fp_m", "_last_dmd_fp_gate_w_mean",
              "_last_dmd_fp_gate_w_min", "_last_dmd_fp_gate_w_max",
              "_last_dmd_fp_gate_share"):
        assert a in txt, f"{a} never read by the trainer step line"


def test_weight_age_is_absent_until_a_probe_has_run():
    """Staleness telemetry must not fake a zero age."""
    s = _armed()
    s._last_dmd_fp_w_per_frame = torch.ones(SHAPE[1])
    s._last_dmd_fp_calc_idx = None
    log = {}
    LOSS(s, torch.randn(*SHAPE, requires_grad=True), torch.randn(*SHAPE),
         torch.ones(SHAPE, dtype=torch.bool), log)
    assert "dmd_fp_gate_w_age" not in log
    s._last_dmd_fp_calc_idx = 4
    s._dmd_gate_call_idx = 9
    log2 = {}
    LOSS(s, torch.randn(*SHAPE, requires_grad=True), torch.randn(*SHAPE),
         torch.ones(SHAPE, dtype=torch.bool), log2)
    assert log2["dmd_fp_gate_w_age"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# the DENOISE HOOK -- `_install_dmd_fp_denoise_fn`
#
# The hook IS the probe's measurement apparatus, and its first version had
# four simultaneous defects (docs/DMD_FINGERPRINT_PROBE.md §6):
#   1. it never noised          -- guarded on `hasattr(self,
#      "_add_noise_to_x0")`, a method that exists nowhere in the repo, so
#      the guard was permanently false and x_t = x_p
#   2. it returned `[0]`        -- the FLOW prediction, not x0
#   3. it dropped tf_kwargs_real -- teacher run off-contract
#   4. it called real_score 3x   -- once to isinstance-check, once for the
#      value, once in the fallback branch
# One test per defect, each verified red against that defect restored.
# ---------------------------------------------------------------------------
INSTALL = ActionForcingDMD._install_dmd_fp_denoise_fn
SIGMA_AT = ActionForcingDMD._sigma_at_timestep

# A 4-rung-shaped grid: t=625 maps to sigma=0.625, so a correct noising is
# a LARGE, unmistakable displacement and "forgot to noise" cannot hide.
FP_TIMESTEPS = torch.tensor([1000., 625., 357., 208., 0.])
FP_SIGMAS = torch.tensor([1.0, 0.625, 0.357, 0.208, 0.0])
FP_T = 625


class _FakeSched:
    """`FlowMatchScheduler.add_noise` verbatim (utils/scheduler.py:159).

    Copied rather than imported so the expected noising is an INDEPENDENT
    reference and the test is not tautological against whatever the
    trainer happens to call.
    """

    def __init__(self):
        self.timesteps = FP_TIMESTEPS.clone()
        self.sigmas = FP_SIGMAS.clone()
        self.calls = []

    def add_noise(self, original_samples, noise, timestep):
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        idx = torch.argmin(
            (self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(),
            dim=1,
        )
        sigma = self.sigmas[idx].reshape(-1, 1, 1, 1)
        self.calls.append(timestep.clone())
        return ((1 - sigma) * original_samples + sigma * noise).type_as(noise)


class _RealScoreSpy:
    """Returns (flow, x0) as two DISTINGUISHABLE constants and records
    every call, so "which element" and "how many calls" are both
    observable."""

    def __init__(self, mode="tuple"):
        self.calls = []
        self.flow = torch.full(SHAPE, 7.0)
        self.x0 = torch.full(SHAPE, -3.0)
        self.mode = mode

    def __call__(self, *, noisy_image_or_video, conditional_dict, timestep,
                 **kw):
        self.calls.append({
            "x": noisy_image_or_video.detach().clone(),
            "cond": conditional_dict,
            "t": timestep.detach().clone(),
            "kw": dict(kw),
        })
        if self.mode == "identity":
            return self.flow, noisy_image_or_video
        return self.flow, self.x0


class _HookStub:
    """A bare `self` for the hook installer and the probe helpers."""

    _install_dmd_fp_denoise_fn = ActionForcingDMD._install_dmd_fp_denoise_fn
    _sigma_at_timestep = ActionForcingDMD._sigma_at_timestep
    _dmd_fp_perturb = ActionForcingDMD._dmd_fp_perturb
    _dmd_fingerprint = ActionForcingDMD._dmd_fingerprint
    dmd_fp_every = 1
    _dmd_fp_denoise_fn = "NOT-INSTALLED"      # sentinel, never a callable


def _hook(scheduler="ok", mode="tuple"):
    """Install the hook and hand back everything the asserts need."""
    s = _HookStub()
    s.scheduler = (
        _FakeSched() if scheduler == "ok"
        else (None if scheduler is None else SimpleNamespace())
    )
    s.real_score = _RealScoreSpy(mode=mode)
    ctx = SimpleNamespace(
        t=torch.full(SHAPE[:2], FP_T, dtype=torch.long),
        cond={"prompt_embeds": torch.zeros(1, 2)},
        # two distinct, identifiable kwargs -- the real call passes
        # clean_x / aug_t through tf_kwargs_real
        tfk={"clean_x": torch.arange(6.0).reshape(1, 6),
             "aug_t": torch.tensor([[13]])},
        ref=torch.zeros(*SHAPE),
    )
    INSTALL(s, ctx.t, ctx.cond, ctx.tfk, ctx.ref)
    return s, ctx


# --- defect 1: the sample was never noised ---------------------------------
def test_hook_noises_the_sample_by_the_schedulers_sigma():
    """The tensor handed to `real_score` must be x_p pushed toward a
    gaussian by exactly the scheduler's sigma at t.

    MUTATION (verified red): restore
    `_xt = self._add_noise_to_x0(_xp, _n, _t) if hasattr(self,
    "_add_noise_to_x0") else _xp` -> x_t == x_p, both asserts fail.
    """
    s, ctx = _hook()
    x_p = torch.randn(*SHAPE)
    torch.manual_seed(1717)
    s._dmd_fp_denoise_fn(x_p)
    got = s.real_score.calls[0]["x"]

    assert not torch.allclose(got, x_p), "x_p reached the teacher un-noised"
    # ...and it is the RIGHT noising, for the eps actually drawn
    torch.manual_seed(1717)
    eps = torch.randn(*SHAPE)
    sigma = SIGMA_AT(s, ctx.t, x_p)
    assert float(sigma.flatten()[0]) == pytest.approx(0.625)
    ref = (1.0 - sigma) * x_p + sigma * eps
    assert torch.allclose(got, ref, atol=1e-6), (got - ref).abs().max()


def test_hook_uses_the_probes_own_timestep_for_the_noising():
    """MUTATION: noise at a hardcoded t (e.g. 1000) -> sigma 1.0, the
    `sigma == 0.625` assert above and this one both fail."""
    s, ctx = _hook()
    s._dmd_fp_denoise_fn(torch.randn(*SHAPE))
    assert len(s.scheduler.calls) == 1
    assert torch.equal(
        s.scheduler.calls[0],
        ctx.t.reshape(-1),
    )


def test_noising_is_what_makes_an_identity_teacher_measurable():
    """End-to-end on `_dmd_fingerprint`: with a teacher that returns its
    own input, r = x0_hat - x_p is IDENTICALLY ZERO if the sample was not
    noised, so the fingerprint degenerates to `None` -- a probe that
    silently reports nothing.

    MUTATION (verified red): the un-noised passthrough -> `s` comes back
    None instead of a float.
    """
    s, _ = _hook(mode="identity")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    val = s._dmd_fingerprint(
        torch.randn(*SHAPE), s._dmd_fp_denoise_fn, gen,
        mode="channel_rot", scale=0.2, seeds=1,
    )
    assert val is not None and isinstance(val, float)


# --- defect 2: returned the flow, not x0 -----------------------------------
def test_hook_returns_the_x0_element_not_the_flow():
    """MUTATION (verified red): `return _flow` (i.e. the old `[0]`) ->
    both asserts flip."""
    s, _ = _hook()
    out = s._dmd_fp_denoise_fn(torch.randn(*SHAPE))
    assert torch.equal(out, s.real_score.x0)
    assert not torch.equal(out, s.real_score.flow)


# --- defect 3: tf_kwargs_real was dropped ----------------------------------
def test_hook_passes_tf_kwargs_real_through_to_the_teacher():
    """Every other real_score call in `_compute_kl_grad` passes
    `**tf_kwargs_real`; the probe running off-contract measures a teacher
    nobody else uses.

    MUTATION (verified red): drop `**_kw` from the call -> `kw` arrives
    empty.
    """
    s, ctx = _hook()
    s._dmd_fp_denoise_fn(torch.randn(*SHAPE))
    kw = s.real_score.calls[0]["kw"]
    assert set(kw) == set(ctx.tfk), kw
    for k, v in ctx.tfk.items():
        assert torch.equal(kw[k], v)
    # the rest of the contract too
    assert s.real_score.calls[0]["cond"] is ctx.cond
    assert torch.equal(s.real_score.calls[0]["t"], ctx.t)


def test_hook_snapshots_tf_kwargs_so_later_mutation_cannot_leak_in():
    """The caller's dict is rebuilt every DMD call; the hook must not be
    retro-actively re-contracted by a mutation of it."""
    s, ctx = _hook()
    ctx.tfk["clean_x"] = torch.full((1, 6), 99.0)
    ctx.tfk["injected"] = torch.zeros(1)
    s._dmd_fp_denoise_fn(torch.randn(*SHAPE))
    kw = s.real_score.calls[0]["kw"]
    assert "injected" not in kw
    assert float(kw["clean_x"][0, 0]) == 0.0


# --- defect 4: real_score called up to three times --------------------------
def test_hook_calls_real_score_exactly_once_per_invocation():
    """The old body ran the teacher for the isinstance check, again for
    the value, and again in the fallback -- 2-3 full TF forwards for one
    result.

    MUTATION (verified red): restore the isinstance-then-index form ->
    the count is 2 (tuple path) or 3 (fallback path).
    """
    s, _ = _hook()
    s._dmd_fp_denoise_fn(torch.randn(*SHAPE))
    assert len(s.real_score.calls) == 1
    assert len(s.scheduler.calls) == 1
    s._dmd_fp_denoise_fn(torch.randn(*SHAPE))
    assert len(s.real_score.calls) == 2, "one teacher forward per call"


# --- an unbuildable hook is None, never a passthrough -----------------------
@pytest.mark.parametrize("sched", [None, "no_add_noise"])
def test_unbuildable_hook_is_none_and_says_so_loudly(sched, capsys):
    """A hook that cannot noise must be ABSENT. The original code's
    `except Exception: None` was silent, and its fallback was worse than
    silent -- it ran the probe on an un-noised sample.

    MUTATION (verified red): install the closure anyway / swallow the
    condition -> `_dmd_fp_denoise_fn` is callable and nothing is printed.
    """
    s = _HookStub()
    s.scheduler = None if sched is None else SimpleNamespace()
    s.real_score = _RealScoreSpy()
    INSTALL(s, torch.full(SHAPE[:2], FP_T, dtype=torch.long),
            {}, {}, torch.zeros(*SHAPE))
    assert s._dmd_fp_denoise_fn is None
    assert not callable(s._dmd_fp_denoise_fn)
    err = capsys.readouterr().err
    assert "dmd_fp" in err and "DISABLED" in err, err
    assert "un-noised" in err


def test_a_built_hook_is_callable_and_the_sentinel_is_replaced():
    s, _ = _hook()
    assert callable(s._dmd_fp_denoise_fn)


def test_the_disabled_warning_is_loud_once_not_once_per_dmd_call(capsys):
    """`_compute_kl_grad` installs the hook EVERY gen step, so an
    unconditional print would be per-step spam and would stop being read
    -- which is how a loud failure becomes a silent one."""
    s = _HookStub()
    s.scheduler = None
    s.real_score = _RealScoreSpy()
    t = torch.full(SHAPE[:2], FP_T, dtype=torch.long)
    for _ in range(5):
        INSTALL(s, t, {}, {}, torch.zeros(*SHAPE))
        assert s._dmd_fp_denoise_fn is None
    assert capsys.readouterr().err.count("[dmd_fp]") == 1


# --- source-level regressions on the two structural defects -----------------
def _model_src():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return open(os.path.join(root, "model", "dmd_action_forcing.py")).read()


def test_the_phantom_noising_method_is_never_referenced_in_code():
    """`_add_noise_to_x0` is defined NOWHERE in the repo, so any CODE
    reference to it -- an attribute access or a hasattr/getattr probe --
    is by construction a permanently-false guard.

    AST-based, not a substring grep, so the defect can be NAMED in prose
    (docstrings, comments) without the test tripping over the history it
    is recording.

    MUTATION (verified red): restore
    `self._add_noise_to_x0(...) if hasattr(self, "_add_noise_to_x0")` ->
    both an Attribute node and a hasattr constant come back, red.
    """
    import ast
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    name = "_add_noise_to_x0"
    hits = []
    for sub in ("model", "trainer", "utils", "pipeline"):
        for dirpath, _dirs, files in os.walk(os.path.join(root, sub)):
            for f in sorted(files):
                if not f.endswith(".py"):
                    continue
                p = os.path.join(dirpath, f)
                try:
                    tree = ast.parse(open(p, errors="ignore").read())
                except SyntaxError:
                    continue
                for n in ast.walk(tree):
                    if isinstance(n, ast.Attribute) and n.attr == name:
                        hits.append(f"{p}:{n.lineno} attribute")
                    elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                            and n.name == name:
                        hits.append(f"{p}:{n.lineno} def")
                    elif (isinstance(n, ast.Call)
                          and isinstance(n.func, ast.Name)
                          and n.func.id in ("hasattr", "getattr")
                          and any(isinstance(a, ast.Constant) and a.value == name
                                  for a in n.args)):
                        hits.append(f"{p}:{n.lineno} {n.func.id}")
    assert hits == [], f"code references a method that does not exist: {hits}"


def test_compute_kl_grad_installs_the_hook_via_the_tested_helper():
    """The helper is only worth testing if the trainer path uses it."""
    src = _model_src()
    assert "self._install_dmd_fp_denoise_fn(" in src


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# ===========================================================================
# THE RATCHET -- "capacitor-diode", the stateful wrapper on the gate
#
# WHY it exists (and why these tests are shaped as they are): the depth
# study returned "PROBE CANNOT RANK OFF-MANIFOLD DISTANCE -- GATE NOT
# VIABLE", 0/8 arms, at t=250/500/750. That verdict STANDS. It is a
# verdict about a MONOTONICITY criterion, and the measured s-vs-depth
# curve is U-SHAPED (trough at depth 16 in 11 of 12 (arm, timestep)
# series): high near depth 0 because the sample is on the DATA manifold,
# high again at large depth because it has been captured by the MODEL'S
# OWN attractor -- the teacher's field is locally restoring in both
# basins. deepest-s minus depth-0-s over 12 series: mean +0.0244, sd
# 0.0709, 9/12 POSITIVE (1.78x the MEAN seed-noise floor 0.0137, but only
# 0.72x the WORST floor 0.0338), i.e. the two ends are statistically
# indistinguishable and where they differ the DEEP end scores HIGHER.
#
# So m is NON-INJECTIVE in depth and any gate that is a pure function of
# the current m is ill-posed. The ratchet adds the one thing the probe
# does not have: TIME ORDER, via a per-ride running minimum.
#
#   DIODE     w_t = min(w_{t-1}, ...)   -- the far branch can never
#                                          RE-OPEN the gate
#   CAPACITOR ema on m BEFORE the ramp  -- closes progressively
#
# The DANGEROUS failure mode is the opposite one: a running minimum that
# never resets latches shut and silently zeroes DMD for the rest of
# training. Hence the reset tests below are the load-bearing ones.
# ===========================================================================
APPLY = ActionForcingDMD._dmd_fp_ratchet_apply


def _ratchet(ema=0.0, m_lo=0.0, m_hi=1.0, ex=1.0, mw=0.0, **over):
    """An armed gate + armed ratchet, with NO ride depth observed yet."""
    return _armed(
        dmd_fp_m_lo=m_lo, dmd_fp_m_hi=m_hi,
        dmd_fp_gate_exponent=ex, dmd_fp_gate_min_weight=mw,
        dmd_fp_ratchet_enabled=True, dmd_fp_ratchet_ema=ema, **over)


def _rstep(s, m_vec, depth, log=None):
    """One DMD call at ride depth ``depth``, probe reporting ``m_vec``.

    Drives the REAL composition path (`_dmd_loss_with_fp_gate`), not the
    ratchet in isolation, so these tests also pin that the ratchet's
    output is what actually weights the loss.
    """
    OBSERVE(s, depth)
    m = torch.as_tensor(m_vec, dtype=torch.float32)
    s._last_dmd_fp_m_per_frame = m
    # what the probe stashes: the UN-ratcheted per-frame weight
    s._last_dmd_fp_w_per_frame = WFROM_M(s, m)
    x = torch.zeros(*SHAPE, requires_grad=True)
    g = torch.full(SHAPE, 0.5)
    mask = torch.ones(SHAPE, dtype=torch.bool)
    lg = {} if log is None else log
    loss = LOSS(s, x, g, mask, lg)
    return lg, loss, x


def _flat(v):
    """A uniform per-frame m vector -- the scalar-like case."""
    return [float(v)] * SHAPE[1]


# The measured shape, coarsened: falling to a trough then rising back
# ABOVE where it started. Index 4 is the trough (the study's depth 16).
U_SHAPED_M = [0.90, 0.70, 0.50, 0.35, 0.25, 0.40, 0.60, 0.85, 0.95]
U_TROUGH_IDX = 4


# ---------------------------------------------------------------------------
# DIODE: monotone non-increase within a ride
# ---------------------------------------------------------------------------
def test_ratchet_never_rises_within_a_ride():
    """THE diode property. Feed the U-shaped m the study measured and
    assert the weight never rises -- the far (attractor) branch's rising
    m must NEVER re-open the gate.

    MUTATION: `w_out = w_new` (i.e. use f(m_t) directly instead of the
    running min) in `_dmd_fp_ratchet_apply` -> the weight tracks the U
    straight back up and this goes red.
    """
    s = _ratchet(ema=0.0)
    seen = []
    for d, mv in enumerate(U_SHAPED_M):
        _rstep(s, _flat(mv), depth=d)
        seen.append(s._last_dmd_fp_ratchet_w_mean)
    assert all(b <= a + 1e-9 for a, b in zip(seen, seen[1:])), seen
    # and it is HELD at the trough, not merely non-increasing by luck
    assert seen[-1] == pytest.approx(min(U_SHAPED_M), abs=1e-6)
    # the un-ratcheted gate would have re-opened all the way to 0.95
    assert WFROM_M(s, U_SHAPED_M[-1]) == pytest.approx(0.95)


def test_ratchet_output_is_what_actually_weights_the_loss():
    """A running minimum nobody applies is the same silent failure one
    level along.

    MUTATION: return `w_raw` from `_dmd_fp_ratchet_apply` -> red.
    """
    s = _ratchet(ema=0.0)
    _rstep(s, _flat(0.2), depth=0)                 # min pinned at 0.2
    _, loss, x = _rstep(s, _flat(1.0), depth=1)    # probe says wide open
    loss.backward()
    per_frame = x.grad.abs().flatten(2).sum(-1).flatten()
    # weighted MEAN with a uniform weight is scale-free, so check the
    # telemetry the loss reduction actually used
    assert s._last_dmd_fp_gate_w_mean == pytest.approx(0.2)
    assert float(per_frame[0]) > 0.0


def test_ratchet_latch_depth_is_the_trough():
    """`fp_rt_d` = ride depth at which the running MIN last decreased. On
    a U-shaped m that stops moving AT THE TROUGH, which is the
    training-time cross-check on the offline study's depth-16 trough.

    MUTATION: set the latch depth on every call (drop the `if decreased`)
    -> it tracks the current depth to 8 and this goes red.
    """
    s = _ratchet(ema=0.0)
    for d, mv in enumerate(U_SHAPED_M):
        _rstep(s, _flat(mv), depth=d)
    assert s._last_dmd_fp_ratchet_latch_depth == pytest.approx(
        float(U_TROUGH_IDX))
    # and the diode is recorded as having BLOCKED a rise
    assert s._last_dmd_fp_ratchet_latched == pytest.approx(1.0)


def test_latched_is_zero_while_m_is_still_falling():
    """Not sticky-on from the start: the flag means "the diode refused a
    rise", which has not happened yet on a monotone descent.

    MUTATION (M27): set `latched` unconditionally -> red."""
    s = _ratchet(ema=0.0)
    for d, mv in enumerate(U_SHAPED_M[:U_TROUGH_IDX + 1]):
        _rstep(s, _flat(mv), depth=d)
    assert s._last_dmd_fp_ratchet_latched == pytest.approx(0.0)
    assert s._last_dmd_fp_ratchet_latch_depth == pytest.approx(
        float(U_TROUGH_IDX))


# ---------------------------------------------------------------------------
# RESET -- the anti-latch-forever tests. The dangerous direction.
# ---------------------------------------------------------------------------
def test_ratchet_resets_at_ride_boundary_restoring_full_weight():
    """THE anti-latch test. A ratchet that never resets latches shut and
    silently kills DMD for the rest of training.

    MUTATION: make `reset_dmd_fp_ratchet` a no-op body (`pass`) -> the
    new ride inherits the old ride's 0.1 minimum and this goes red.
    """
    s = _ratchet(ema=0.0)
    for d, mv in enumerate([0.9, 0.5, 0.1]):
        _rstep(s, _flat(mv), depth=d)
    assert s._last_dmd_fp_ratchet_w_mean == pytest.approx(0.1)
    RESET(s, reason="ride_setup")                  # <- the ride boundary
    assert s._last_dmd_fp_ratchet_w_mean is None   # state truly cleared
    _rstep(s, _flat(0.9), depth=0)                 # fresh ride, near-manifold
    assert s._last_dmd_fp_ratchet_w_mean == pytest.approx(0.9)
    assert s._last_dmd_fp_ratchet_latched == pytest.approx(0.0)


def test_reset_clears_every_piece_of_ratchet_state():
    """Partial reset is the same bug with extra steps: a surviving EMA or
    latch depth carries the old ride's history into the new one.

    MUTATION (M24): leave `_dmd_fp_ratchet_m_ema` alone in the reset ->
    red."""
    s = _ratchet(ema=0.5)
    for d, mv in enumerate([0.9, 0.5, 0.1]):
        _rstep(s, _flat(mv), depth=d)
    RESET(s, reason="ride_setup")
    for a in ("_dmd_fp_ratchet_w", "_dmd_fp_ratchet_m_ema",
              "_dmd_fp_ratchet_depth", "_dmd_fp_ratchet_latch_depth",
              "_last_dmd_fp_ratchet_w_mean", "_last_dmd_fp_ratchet_share",
              "_last_dmd_fp_ratchet_latched",
              "_last_dmd_fp_ratchet_latch_depth",
              "_last_dmd_fp_ratchet_depth"):
        assert getattr(s, a) is None, f"{a} survived the reset"
    assert s._dmd_fp_ratchet_latched is False


def test_reset_is_counted_so_a_stalled_hook_is_visible():
    """If rides turn over and this counter stops moving, the ratchet is
    latching forever. That has to be readable off the step line.

    MUTATION (M26): drop the `+ 1` from the reset counter -> red."""
    s = _ratchet(ema=0.0)
    assert s._last_dmd_fp_ratchet_resets == pytest.approx(0.0)
    RESET(s, reason="ride_setup")
    RESET(s, reason="ride_setup")
    assert s._last_dmd_fp_ratchet_resets == pytest.approx(2.0)


def test_reset_count_is_absent_when_the_ratchet_is_off():
    """Step-line hygiene: nothing to say on a default-off run.

    MUTATION (M23): publish the count unconditionally -> red, and it
    takes `test_no_forgeable_ratchet_zeros_before_anything_is_measured`
    with it. That is the point: 0.0 is a real value here."""
    s = _Stub()
    INIT(s, _cfg())
    assert s._last_dmd_fp_ratchet_resets is None
    RESET(s, reason="ride_setup")
    assert s._last_dmd_fp_ratchet_resets is None


def test_depth_backstop_resets_loudly_when_the_hook_is_missing(capsys):
    """BACKSTOP. A ride boundary that arrives without the explicit reset
    (a trainer that lost the hook) must NOT latch the gate forever. A
    strict decrease in ride depth can only mean a new ride, so reset --
    and say so on stderr, because the missing hook is still a bug.

    MUTATION: delete the backstop branch from
    `_dmd_fp_ratchet_observe_depth` -> the second ride inherits the first
    ride's 0.1 minimum forever and this goes red.
    """
    s = _ratchet(ema=0.0)
    for d, mv in enumerate([0.9, 0.5, 0.1]):
        _rstep(s, _flat(mv), depth=d)
    assert s._last_dmd_fp_ratchet_w_mean == pytest.approx(0.1)
    _rstep(s, _flat(0.9), depth=0)          # new ride, NO explicit reset
    assert s._last_dmd_fp_ratchet_w_mean == pytest.approx(0.9)
    err = capsys.readouterr().err
    assert "RATCHET BACKSTOP" in err
    assert "reset_dmd_fp_ratchet" in err


def test_repeated_depth_does_NOT_reset_the_ratchet():
    """The backstop tests a STRICT decrease on purpose. Several DMD calls
    can share one ride depth, and a `<=` test would reset on every repeat
    -- silently defeating the diode, i.e. the same bug in the opposite
    direction and much harder to see.

    MUTATION: `d <= prev` in `_dmd_fp_ratchet_observe_depth` -> the
    second call at depth 3 resets and the weight jumps back to 0.9, red.
    """
    s = _ratchet(ema=0.0)
    _rstep(s, _flat(0.2), depth=3)
    _rstep(s, _flat(0.9), depth=3)          # same depth, m recovered
    assert s._last_dmd_fp_ratchet_w_mean == pytest.approx(0.2)


def test_ratchet_refuses_to_run_with_no_ride_depth_observed():
    """No ride => no defined reset point => the latch-forever failure.
    Refuse rather than accumulate a minimum that can never be cleared
    (e.g. the non-streaming generator loss, which has no rides).

    MUTATION (M12): drop the depth check -> the ratchet happily starts
    accumulating with no defined reset point, red."""
    s = _ratchet(ema=0.0)
    s._last_dmd_fp_m_per_frame = torch.full((SHAPE[1],), 0.5)
    s._last_dmd_fp_w_per_frame = torch.full((SHAPE[1],), 0.5)
    with pytest.raises(ValueError, match="no ride depth"):
        LOSS(s, torch.randn(*SHAPE, requires_grad=True), torch.randn(*SHAPE),
             torch.ones(SHAPE, dtype=torch.bool), {})


# ---------------------------------------------------------------------------
# CAPACITOR: the EMA time constant
# ---------------------------------------------------------------------------
def test_ema_time_constant_actually_smooths():
    """The capacitor. With ema>0 a single large drop in m must NOT close
    the gate all the way in one step.

    MUTATION: ignore `dmd_fp_ratchet_ema` (use m directly) -> the smoothed
    weight equals the unsmoothed 0.1 and this goes red.
    """
    fast = _ratchet(ema=0.0)
    slow = _ratchet(ema=0.8)
    for s in (fast, slow):
        _rstep(s, _flat(0.9), depth=0)
        _rstep(s, _flat(0.1), depth=1)
    assert fast._last_dmd_fp_ratchet_w_mean == pytest.approx(0.1)
    # ema on m BEFORE the ramp: 0.8*0.9 + 0.2*0.1 = 0.74
    assert slow._last_dmd_fp_ratchet_w_mean == pytest.approx(0.74, abs=1e-6)
    assert slow._last_dmd_fp_ratchet_w_mean > fast._last_dmd_fp_ratchet_w_mean


def test_ema_still_converges_so_the_gate_can_close():
    """A capacitor that never charges is a gate that never closes.

    MUTATION (M29): `m_ema = prev_e` (a == 1 in effect) -> the gate holds
    at 0.9 forever, red."""
    s = _ratchet(ema=0.8)
    _rstep(s, _flat(0.9), depth=0)
    for d in range(1, 40):
        _rstep(s, _flat(0.0), depth=d)
    assert s._last_dmd_fp_ratchet_w_mean == pytest.approx(0.0, abs=1e-3)


def test_ema_is_applied_to_m_not_to_the_mapped_weight():
    """Smoothing AFTER the ramp cannot recover what the ramp's clamp has
    already destroyed. With m_lo=0.4 both m=0.2 and m=0.0 map to w=0, so
    a w-side EMA cannot tell them apart; an m-side EMA can.

    MUTATION: `w_ema = a*prev_w + (1-a)*f(m)` -> both stubs below agree
    and this goes red.
    """
    a, lo, hi = 0.5, 0.4, 1.0
    s1 = _ratchet(ema=a, m_lo=lo, m_hi=hi)
    s2 = _ratchet(ema=a, m_lo=lo, m_hi=hi)
    _rstep(s1, _flat(1.0), depth=0)
    _rstep(s2, _flat(1.0), depth=0)
    _rstep(s1, _flat(0.2), depth=1)
    _rstep(s2, _flat(0.0), depth=1)
    # m-side: ema(m) = 0.6 vs 0.5 -> w = 1/3 vs 1/6. Distinguishable.
    assert s1._last_dmd_fp_ratchet_w_mean == pytest.approx(
        (0.6 - lo) / (hi - lo), abs=1e-6)
    assert s2._last_dmd_fp_ratchet_w_mean == pytest.approx(
        (0.5 - lo) / (hi - lo), abs=1e-6)
    assert s1._last_dmd_fp_ratchet_w_mean != pytest.approx(
        s2._last_dmd_fp_ratchet_w_mean)


@pytest.mark.parametrize("bad", [1.0, 1.5, -0.1])
def test_incoherent_ema_raises(bad):
    """1.0 would freeze the capacitor at the ride's first reading, so the
    gate could never close at all -- inert, not conservative.

    MUTATION (M18): drop the range check -> red on all three values."""
    s = _ratchet(ema=bad)
    OBSERVE(s, 1)
    s._last_dmd_fp_m_per_frame = torch.full((SHAPE[1],), 0.5)
    with pytest.raises(ValueError, match="dmd_fp_ratchet_ema"):
        APPLY(s, torch.full((SHAPE[1],), 0.5), {})


# ---------------------------------------------------------------------------
# ELEMENTWISE per-frame semantics survive the ratchet
# ---------------------------------------------------------------------------
def test_ratchet_is_elementwise_over_frames_not_a_scalar():
    """The measured align cliff lives INSIDE one band (positive at frames
    9-13, -0.53 by 17). Frame i's running minimum must be driven only by
    frame i's history, so a band whose TAIL has drifted still attenuates
    TAIL-ONLY.

    MUTATION: collapse to a scalar before the min
    (`w_new = w_new.mean().expand_as(w_new)`) -> the head is dragged down
    with the tail and this goes red.
    """
    s = _ratchet(ema=0.0)
    _rstep(s, _flat(1.0), depth=0)
    # tail drifts, head does not
    _rstep(s, [1.0, 1.0, 1.0, 0.5, 0.2, 0.05], depth=1)
    w = s._dmd_fp_ratchet_w
    assert torch.allclose(
        w.float(), torch.tensor([1.0, 1.0, 1.0, 0.5, 0.2, 0.05]), atol=1e-6)
    # ...and the head stays open when the tail's minimum is already shut
    _, loss, x = _rstep(s, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], depth=2)
    assert torch.allclose(
        s._dmd_fp_ratchet_w.float(),
        torch.tensor([1.0, 1.0, 1.0, 0.5, 0.2, 0.05]), atol=1e-6)
    loss.backward()
    per_frame = x.grad.abs().flatten(2).sum(-1).flatten()
    head, tail = float(per_frame[0]), float(per_frame[-1])
    assert tail < head, f"tail {tail} not attenuated vs head {head}"
    ratios = [float(per_frame[i] / per_frame[0]) for i in range(SHAPE[1])]
    assert ratios == pytest.approx(
        [1.0, 1.0, 1.0, 0.5, 0.2, 0.05], rel=1e-5)


def test_ratchet_shape_change_mid_ride_raises_rather_than_resetting():
    """A reset here would RE-OPEN the gate on the deep end -- exactly what
    the diode exists to prevent. "I no longer know what to line up with
    what" must not be resolved silently in the unsafe direction.

    MUTATION (M17b): drop both shape guards and silently null the state
    (previous EMA and previous running min) instead -> no raise, red."""
    s = _ratchet(ema=0.0)
    _rstep(s, _flat(0.5), depth=0)
    OBSERVE(s, 1)
    s._last_dmd_fp_m_per_frame = torch.full((SHAPE[1] - 1,), 0.5)
    with pytest.raises(ValueError, match="changed shape mid-ride"):
        APPLY(s, torch.full((SHAPE[1] - 1,), 0.5), {})


# ---------------------------------------------------------------------------
# fail-loud arming
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("missing_key", list(RATCHET_KEYS))
def test_ratchet_armed_with_unset_knobs_raises_and_names_them(missing_key):
    """Same Class-B discipline as the gate: an unmeasured constant must
    not be silently inherited.

    MUTATION: drop the `_dmd_fp_require_ratchet_calib` call from
    `_dmd_fp_ratchet_apply` -> no raise, red.
    """
    _kw = dict(dmd_fp_ratchet_enabled=True, dmd_fp_ratchet_ema=0.5)
    _kw[missing_key] = None
    s = _armed(**_kw)
    OBSERVE(s, 1)
    s._last_dmd_fp_m_per_frame = torch.full((SHAPE[1],), 0.5)
    s._last_dmd_fp_w_per_frame = torch.full((SHAPE[1],), 0.5)
    with pytest.raises(ValueError) as ei:
        LOSS(s, torch.randn(*SHAPE, requires_grad=True), torch.randn(*SHAPE),
             torch.ones(SHAPE, dtype=torch.bool), {})
    assert missing_key in str(ei.value)


def test_ratchet_class_b_knobs_are_none_when_absent_from_config():
    """MUTATION (M22): `float(getattr(args, "dmd_fp_ratchet_ema", 0.9))`
    -> the time constant comes back invented, red."""
    s = _Stub()
    INIT(s, _cfg())
    assert s.dmd_fp_ratchet_enabled is False
    for k in RATCHET_KEYS:
        assert getattr(s, k) is None, f"{k} was silently defaulted"
    assert sorted(ActionForcingDMD._dmd_fp_missing_ratchet_calib(s)) == \
        sorted(RATCHET_KEYS)


def test_ratchet_armed_without_the_gate_raises():
    """The ratchet MODIFIES the gate's per-frame weight. With the gate off
    there is nothing to modify and the flag would be silently inert --
    the exact failure mode this campaign keeps rediscovering.

    MUTATION (M14): drop the elif branch -> silently inert, red."""
    s = _armed(dmd_fp_gate_enabled=False,
               dmd_fp_ratchet_enabled=True, dmd_fp_ratchet_ema=0.5)
    with pytest.raises(ValueError, match="dmd_fp_gate_enabled is"):
        LOSS(s, torch.randn(*SHAPE, requires_grad=True), torch.randn(*SHAPE),
             torch.ones(SHAPE, dtype=torch.bool), {})


def test_ratchet_without_a_per_frame_m_raises():
    """The ratchet smooths m BEFORE the ramp, so it needs m's pre-image.
    Ratcheting a weight whose m it does not have would silently switch
    the capacitor off.

    MUTATION (M13): fall back to `m = w_raw` -> no raise, red."""
    s = _ratchet(ema=0.5)
    OBSERVE(s, 1)
    s._last_dmd_fp_m_per_frame = None
    s._last_dmd_fp_w_per_frame = torch.full((SHAPE[1],), 0.5)
    with pytest.raises(ValueError, match="no per-frame fingerprint score m"):
        LOSS(s, torch.randn(*SHAPE, requires_grad=True), torch.randn(*SHAPE),
             torch.ones(SHAPE, dtype=torch.bool), {})


# ---------------------------------------------------------------------------
# default-off byte-identical, and telemetry
# ---------------------------------------------------------------------------
def test_ratchet_default_off_is_bitwise_identical_and_silent():
    """The whole feature must be invisible until armed.

    MUTATION (M25): default `dmd_fp_ratchet_enabled` to True -> the
    shipped-default run trips the "ratchet armed, gate off" raise, red.
    """
    torch.manual_seed(11)
    s = _Stub()
    INIT(s, _cfg())                     # shipped default: nothing set
    assert s.dmd_fp_ratchet_enabled is False
    x = torch.randn(*SHAPE, dtype=torch.float32, requires_grad=True)
    g = torch.randn(*SHAPE, dtype=torch.float32)
    mask = torch.rand(SHAPE) > 0.3
    log = {}
    got = LOSS(s, x, g, mask, log)
    ref = 0.5 * F.mse_loss(
        x.double()[mask], (x.double() - g.double()).detach()[mask],
        reduction="mean")
    assert torch.equal(got.detach(), ref.detach())
    assert torch.equal(torch.autograd.grad(got, x, retain_graph=True)[0],
                       torch.autograd.grad(ref, x)[0])
    assert not [k for k in log if "ratchet" in k], log


def test_gate_on_ratchet_off_is_unchanged_by_the_ratchet_landing():
    """The gate's own behaviour must not move because the ratchet exists.

    MUTATION: apply the ratchet whenever a per-frame m is present -> red.
    """
    w = [1.0, 1.0, 1.0, 0.5, 0.2, 0.05]
    s, x, g, mask = _equal_error_band(w)
    s._last_dmd_fp_m_per_frame = torch.as_tensor(w)   # probe HAS run
    assert s.dmd_fp_ratchet_enabled is False
    log = {}
    LOSS(s, x, g, mask, log)
    assert s._last_dmd_fp_gate_w_mean == pytest.approx(sum(w) / len(w))
    assert not [k for k in log if "ratchet" in k], log


def test_ratchet_telemetry_is_stashed_on_the_model_for_the_step_line():
    """dmd_log_dict is wandb-only. The trainer reads model attributes.

    MUTATION: write the ratchet values only into `log_dict` -> red.
    """
    s = _ratchet(ema=0.0)
    log = {}
    _rstep(s, _flat(0.8), depth=0)
    _rstep(s, _flat(0.4), depth=1, log=log)
    _rstep(s, _flat(0.9), depth=2, log=log)
    assert s._last_dmd_fp_ratchet_w_mean == pytest.approx(0.4)
    assert s._last_dmd_fp_ratchet_latched == pytest.approx(1.0)
    assert s._last_dmd_fp_ratchet_latch_depth == pytest.approx(1.0)
    assert s._last_dmd_fp_ratchet_depth == pytest.approx(2.0)
    for k in ("dmd_fp_ratchet_w_mean", "dmd_fp_ratchet_raw_w_mean",
              "dmd_fp_ratchet_share", "dmd_fp_ratchet_latched",
              "dmd_fp_ratchet_latch_depth", "dmd_fp_ratchet_depth",
              "dmd_fp_ratchet_resets"):
        assert k in log, k


def test_ratchet_share_reports_the_retained_fraction_of_dmd():
    """SHARE, not raw values -- standing campaign rule. `fp_rt_shr` is the
    ratchet's OWN marginal effect (retained / un-ratcheted); `fp_share`
    downstream is the absolute share of DMD that survives everything, and
    it must already include the ratchet.

    MUTATION (M9): `return w_raw` from `_dmd_fp_ratchet_apply` -> both
    the ratchet share and the absolute fp_share report 0.9, red.
    """
    s = _ratchet(ema=0.0)
    _rstep(s, _flat(0.3), depth=0)
    log, _, _ = _rstep(s, _flat(0.9), depth=1)
    assert log["dmd_fp_ratchet_raw_w_mean"] == pytest.approx(0.9)
    assert log["dmd_fp_ratchet_w_mean"] == pytest.approx(0.3)
    assert log["dmd_fp_ratchet_share"] == pytest.approx(0.3 / 0.9)
    # the ABSOLUTE share of DMD surviving the whole gate
    assert log["dmd_fp_gate_share"] == pytest.approx(0.3)


def test_ratchet_share_is_absent_not_forged_when_the_ramp_is_at_zero():
    """"What fraction did the ratchet remove" is UNDEFINED when the
    response curve itself is at zero. A 0.0 or 1.0 there would be a
    forgeable number inside the metric's meaningful range.

    MUTATION (M16): emit the share unconditionally (0/0 guard removed) ->
    red."""
    s = _ratchet(ema=0.0)
    log, _, _ = _rstep(s, _flat(0.0), depth=0)
    assert "dmd_fp_ratchet_share" not in log
    assert log["dmd_fp_ratchet_raw_closed"] == pytest.approx(1.0)
    assert s._last_dmd_fp_ratchet_share is None


def test_no_forgeable_ratchet_zeros_before_anything_is_measured():
    """MUTATIONS (M23, M24): publish the reset count unconditionally /
    leave the EMA behind on reset -> red."""
    s = _Stub()
    INIT(s, _cfg())
    for a in ("_last_dmd_fp_ratchet_w_mean", "_last_dmd_fp_ratchet_share",
              "_last_dmd_fp_ratchet_latched",
              "_last_dmd_fp_ratchet_latch_depth",
              "_last_dmd_fp_ratchet_depth", "_last_dmd_fp_ratchet_resets",
              "_dmd_fp_ratchet_w", "_dmd_fp_ratchet_m_ema"):
        assert getattr(s, a) is None, f"{a} was pre-filled"


# ---------------------------------------------------------------------------
# the wiring nobody can see from inside the model
# ---------------------------------------------------------------------------
def _src(*parts):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return open(os.path.join(root, *parts)).read()


def test_trainer_resets_the_ratchet_at_the_RIDE_BOUNDARY():
    """Hooked to the ride-setup line, NOT to a step counter. This is the
    anti-latch-forever wiring and it is invisible from the model side.

    MUTATION: delete the `reset_dmd_fp_ratchet` call from the trainer ->
    red (and the run would latch shut after its first ride).
    """
    txt = _src("trainer", "causal_action_forcing_train.py")
    marker = "self._chunks_in_current_ride = 0"
    assert marker in txt
    i = txt.index(marker)
    window = txt[i:i + 1500]
    assert "self.model.reset_dmd_fp_ratchet(" in window, (
        "the ratchet reset is not adjacent to the ride-boundary line")
    # exactly one reset call site -- two would mean one of them is
    # keyed on something that is not a ride
    assert txt.count("reset_dmd_fp_ratchet(") == 1


def test_model_feeds_ride_depth_to_the_ratchet_before_any_early_return():
    """The depth backstop has to see EVERY roll, including the ones the
    only_first / only_last gates skip -- otherwise a skipped roll can
    carry a stale depth across a ride boundary.

    MUTATION: move the `_dmd_fp_ratchet_observe_depth` call below the
    `only_first` early return -> red.
    """
    txt = _src("model", "dmd_action_forcing.py")
    i = txt.index("def compute_generator_loss_streaming")
    body = txt[i:i + 6000]
    assert "self._dmd_fp_ratchet_observe_depth(chunks_in_ride)" in body
    assert (body.index("self._dmd_fp_ratchet_observe_depth(chunks_in_ride)")
            < body.index("dmd_skipped_non_first_chunk_of_ride"))


def test_trainer_step_line_reads_the_ratchet_attributes():
    """MUTATION (M20): delete the fp_rt_* rows from the trainer's getattr
    tuple -> red. A stashed attribute nobody reads is the same silent
    failure one level along."""
    txt = _src("trainer", "causal_action_forcing_train.py")
    for a in ("_last_dmd_fp_ratchet_w_mean", "_last_dmd_fp_ratchet_share",
              "_last_dmd_fp_ratchet_latched",
              "_last_dmd_fp_ratchet_latch_depth",
              "_last_dmd_fp_ratchet_resets"):
        assert a in txt, f"{a} never read by the trainer step line"


def test_yaml_declares_the_ratchet_keys():
    """MUTATIONS (M28, M30): ship `dmd_fp_ratchet_ema: 0.9`, or ship
    `dmd_fp_ratchet_enabled: true` -> red. A flag with no yaml key is
    un-settable; a yaml key shipped pre-filled is the guessed default
    coming back in through the config instead of through the code."""
    import re
    txt = _src("configs", "action_forcing_phase3_dmd.yaml")
    assert re.search(r"^dmd_fp_ratchet_enabled: *false", txt, re.M)
    for k in RATCHET_KEYS:
        assert re.search(rf"^{k}: *null", txt, re.M), (
            f"{k} must ship as null -- it is a MEASURED value")
