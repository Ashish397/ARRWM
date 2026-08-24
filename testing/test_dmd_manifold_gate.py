"""Tests for the DMD manifold gate (docs/DMD_MANIFOLD_GATE.md).

CPU-only. Exercises `_dmd_error_gate_weight` as a pure function against a
stub `self`, so no model, VAE or GPU is constructed.

What these protect:

* **The sign convention of `align`.** The whole thesis is "DMD points
  away from the manifold when far from it", and `align` is the number
  that decides it. If its sign were inverted the measurement would
  confirm the thesis exactly when it is false. Pinned against
  analytically-constructed directions where the answer is +1 / -1.
* **Fail-loud on unset thresholds.** `pix_gan_weight` and `pix_r1_gamma`
  were both silently-inherited-default disasters in this campaign. A gate
  threshold is a MEASURED quantity for a specific teacher/data pair, so
  enabling without it must raise, never fall back.
* **Gate polarity.** This gate closes as error GROWS. The pre-existing
  `dmd_mae_gate` closes as the ratio SHRINKS. Getting these backwards
  would apply full DMD weight exactly on the drifted tail — the failure
  the gate exists to prevent.
* **Inertness when off.** Default-off must touch nothing.

Run:
    OMP_NUM_THREADS=8 PYTHONPATH=. pytest -q testing/test_dmd_manifold_gate.py
"""
import os
import sys
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with patch.object(torch.cuda, "current_device", return_value=0):
    from model.dmd_action_forcing import ActionForcingDMD

GATE = ActionForcingDMD._dmd_error_gate_weight
SHAPE = (1, 3, 16, 8, 8)


class _Stub:
    """Only what the gate reads."""

    def __init__(self, **kw):
        self.dmd_err_gate_enabled = False
        self.dmd_err_gate_e_lo = None
        self.dmd_err_gate_e_hi = None
        self.dmd_err_gate_min_weight = 0.0
        self.dmd_err_gate_ema = 0.0
        self.dmd_err_gate_trace_path = None
        self._dmd_err_gate_ema = None
        for k, v in kw.items():
            setattr(self, k, v)


def _fixture():
    torch.manual_seed(0)
    gt = torch.randn(*SHAPE)
    x0 = gt + 0.5
    mask = torch.ones(SHAPE, dtype=torch.bool)
    return gt, x0, mask


# ---------------------------------------------------------------------------
# align — the number the whole thesis rests on
# ---------------------------------------------------------------------------
def test_align_is_plus_one_when_update_points_at_gt():
    """`align = cos(-grad, GT - x0)`. The UPDATE direction is -grad, so a
    grad of -(GT - x0) means the update points exactly at GT."""
    gt, x0, mask = _fixture()
    toward = gt - x0
    log = {}
    GATE(_Stub(), x0=x0, pred_real=gt, grad=-toward, gt_target=gt,
         gradient_mask=mask, log_dict=log)
    assert log["dmd_align"] == pytest.approx(1.0, abs=1e-4)


def test_align_is_minus_one_when_update_points_away():
    gt, x0, mask = _fixture()
    toward = gt - x0
    log = {}
    GATE(_Stub(), x0=x0, pred_real=gt, grad=+toward, gt_target=gt,
         gradient_mask=mask, log_dict=log)
    assert log["dmd_align"] == pytest.approx(-1.0, abs=1e-4)


def test_align_absent_without_gt_but_e_still_measured():
    """`e` needs no GT — that is the point of choosing it over any
    MAE-vs-GT signal. `align` is measurement-only and does need GT, so it
    is simply omitted (never zero-filled: 0.0 is a meaningful align,
    meaning 'orthogonal')."""
    gt, x0, mask = _fixture()
    log = {}
    GATE(_Stub(), x0=x0, pred_real=gt, grad=torch.randn(*SHAPE),
         gt_target=None, gradient_mask=mask, log_dict=log)
    assert "dmd_align" not in log
    assert log["dmd_err_gate_e"] > 0.0


def test_e_measures_teacher_disagreement_not_student_error():
    """`e = |x0 - pred_real|`. A student far from GT but with a teacher
    that AGREES with it yields small `e` -- the regime where DMD is
    useful and must stay ON. A student-error signal would gate here."""
    gt, x0, mask = _fixture()
    far_x0 = gt + 5.0                     # student far from GT
    log = {}
    GATE(_Stub(), x0=far_x0, pred_real=far_x0.clone(), grad=torch.randn(*SHAPE),
         gt_target=gt, gradient_mask=mask, log_dict=log)
    assert log["dmd_err_gate_e"] == pytest.approx(0.0, abs=1e-6)
    assert log["dmd_gt_dist"] > 4.0       # ...though it IS far from GT


# ---------------------------------------------------------------------------
# fail-loud
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("lo,hi", [(None, None), (0.2, None), (None, 1.0)])
def test_armed_without_measured_thresholds_raises(lo, hi):
    gt, x0, mask = _fixture()
    s = _Stub(dmd_err_gate_enabled=True,
              dmd_err_gate_e_lo=lo, dmd_err_gate_e_hi=hi)
    with pytest.raises(ValueError, match="measured"):
        GATE(s, x0=x0, pred_real=gt, grad=torch.randn(*SHAPE), gt_target=gt,
             gradient_mask=mask, log_dict={})


def test_inverted_thresholds_raise():
    gt, x0, mask = _fixture()
    s = _Stub(dmd_err_gate_enabled=True,
              dmd_err_gate_e_lo=1.0, dmd_err_gate_e_hi=0.2)
    with pytest.raises(ValueError, match="exceed"):
        GATE(s, x0=x0, pred_real=gt, grad=torch.randn(*SHAPE), gt_target=gt,
             gradient_mask=mask, log_dict={})


# ---------------------------------------------------------------------------
# polarity — the thing that must not be backwards
# ---------------------------------------------------------------------------
def test_gate_closes_as_error_grows_not_shrinks():
    """THE polarity test. Weight must DECREASE monotonically with `e`.
    The pre-existing dmd_mae_gate does the opposite (closes as the ratio
    shrinks toward parity); applying that polarity here would give full
    DMD weight on the drifted tail, which is the failure this gate
    exists to prevent."""
    gt, _, mask = _fixture()
    ws = []
    for k in (0.1, 0.3, 0.6, 0.9, 1.5):
        s = _Stub(dmd_err_gate_enabled=True, dmd_err_gate_e_lo=0.2,
                  dmd_err_gate_e_hi=1.0)
        ws.append(GATE(s, x0=gt + k, pred_real=gt, grad=torch.randn(*SHAPE),
                       gt_target=gt, gradient_mask=mask, log_dict={}))
    assert ws == sorted(ws, reverse=True), f"not monotone decreasing: {ws}"
    assert ws[0] == pytest.approx(1.0)     # near manifold -> full DMD
    assert ws[-1] == pytest.approx(0.0)    # far -> DMD off


def test_min_weight_is_a_floor():
    gt, _, mask = _fixture()
    s = _Stub(dmd_err_gate_enabled=True, dmd_err_gate_e_lo=0.2,
              dmd_err_gate_e_hi=1.0, dmd_err_gate_min_weight=0.25)
    w = GATE(s, x0=gt + 5.0, pred_real=gt, grad=torch.randn(*SHAPE),
             gt_target=gt, gradient_mask=mask, log_dict={})
    assert w == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# inertness
# ---------------------------------------------------------------------------
def test_disabled_returns_one_and_still_measures():
    """Off must not gate — but MUST still emit the measurement, because
    the whole point of the first runs is to collect align/e with the gate
    disabled."""
    gt, x0, mask = _fixture()
    log = {}
    w = GATE(_Stub(), x0=x0, pred_real=gt, grad=torch.randn(*SHAPE),
             gt_target=gt, gradient_mask=mask, log_dict=log)
    assert w == 1.0
    assert "dmd_align" in log and "dmd_err_gate_e" in log
    assert "dmd_err_gate_weight" not in log   # no weight when not gating


def test_empty_mask_is_a_noop():
    gt, x0, _ = _fixture()
    log = {}
    w = GATE(_Stub(dmd_err_gate_enabled=True), x0=x0, pred_real=gt,
             grad=torch.randn(*SHAPE), gt_target=gt,
             gradient_mask=torch.zeros(SHAPE, dtype=torch.bool), log_dict=log)
    assert w == 1.0 and log == {}


def test_per_frame_breakdown_present_and_sized():
    """Per-frame align is what detects the effect when the bidirectional
    teacher's clean-GT anchor masks it at the band mean."""
    gt, x0, mask = _fixture()
    log = {}
    GATE(_Stub(), x0=x0, pred_real=gt, grad=torch.randn(*SHAPE), gt_target=gt,
         gradient_mask=mask, log_dict=log)
    pf = log.get("dmd_align_per_frame")
    assert pf is not None and len(pf) == SHAPE[1]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
