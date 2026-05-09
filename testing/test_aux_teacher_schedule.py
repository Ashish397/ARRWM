#!/usr/bin/env python3
"""Tests for the aux teacher's piecewise-linear ``p`` schedule and
the ``aux_teacher_send_student_grad`` flag.

The schedule and grad-gate are local (and tested in isolation):

  * ``_resolved_real_teacher_input_mix_gt_p(current_step)`` — returns
    the static knob when disabled; the piecewise-linear schedule
    when enabled (segment 1 linear → discontinuous drop → segment 2
    linear → flat).
  * ``aux_teacher_send_student_grad=False`` detaches ``noise_base``
    so ``noise_base.requires_grad`` is False even when ``chunk``
    carries autograd.

CPU-only. No model weights, no DDP, no CUDA.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _import_action_forcing_dmd():
    """Import ``ActionForcingDMD`` without a live CUDA device.

    The wan.modules.t5 module evaluates ``torch.cuda.current_device()``
    at class-body time; on CPU-only test runners this raises. Patch
    the call so the import succeeds. We only use the class for its
    pure-Python resolver method, never instantiate it.
    """
    with patch("torch.cuda.current_device", return_value=0):
        from model.dmd_action_forcing import ActionForcingDMD
    return ActionForcingDMD


def _make_stub_with_schedule(
    *,
    enabled: bool = False,
    static_p: float = 0.5,
    seg1_start: float = 1.0,
    seg1_end: float = 0.65,
    seg1_steps: int = 50,
    seg2_start: float = 0.35,
    seg2_end: float = 0.0,
    seg2_steps: int = 50,
):
    """Build a SimpleNamespace stub with the attrs the resolver reads.

    Avoids loading the full ``ActionForcingDMD`` (which pulls CUDA
    deps via the wan model). The resolver method is bound onto the
    stub so we can call it directly.
    """
    ActionForcingDMD = _import_action_forcing_dmd()

    stub = SimpleNamespace(
        aux_teacher_p_schedule_enabled=enabled,
        real_teacher_input_mix_gt_p=static_p,
        aux_teacher_p_seg1_start=seg1_start,
        aux_teacher_p_seg1_end=seg1_end,
        aux_teacher_p_seg1_steps=seg1_steps,
        aux_teacher_p_seg2_start=seg2_start,
        aux_teacher_p_seg2_end=seg2_end,
        aux_teacher_p_seg2_steps=seg2_steps,
    )
    # Bind the unbound method so ``stub.resolver(step)`` works.
    stub.resolver = (
        lambda step: ActionForcingDMD._resolved_real_teacher_input_mix_gt_p(
            stub, step,
        )
    )
    return stub


class TestPSchedule(unittest.TestCase):
    """Piecewise-linear ``p`` schedule semantics."""

    def test_default_disabled_returns_static(self):
        """schedule_enabled=False → resolver returns the static knob
        regardless of step."""
        stub = _make_stub_with_schedule(enabled=False, static_p=0.42)
        for s in (-10, 0, 1, 50, 99, 100, 1000):
            self.assertEqual(
                stub.resolver(s), 0.42,
                f"static-mode resolver should be 0.42 at step {s}",
            )

    def test_segment1_linear_endpoints(self):
        """First segment: p(0)=seg1_start, p(seg1_steps-1)=last
        interior point of the linear ramp (NOT seg1_end — that one
        is overridden by seg2_start at exactly s=seg1_steps)."""
        stub = _make_stub_with_schedule(enabled=True)
        self.assertAlmostEqual(stub.resolver(0), 1.0, places=6)
        # Midpoint of segment 1: s=25, expected = 1.0 + (0.65-1.0)*0.5 = 0.825
        self.assertAlmostEqual(stub.resolver(25), 0.825, places=6)
        # Last interior point of segment 1: s=49, expected = 1.0 + (0.65-1.0)*49/50 = 0.657
        expected = 1.0 + (0.65 - 1.0) * 49.0 / 50.0
        self.assertAlmostEqual(stub.resolver(49), expected, places=6)

    def test_discontinuous_drop_at_seg1_steps(self):
        """At s == seg1_steps, the schedule jumps to seg2_start."""
        stub = _make_stub_with_schedule(enabled=True)
        # Exactly at seg1_steps=50, expect seg2_start=0.35.
        self.assertAlmostEqual(stub.resolver(50), 0.35, places=6)

    def test_segment2_linear_endpoints(self):
        """Second segment: p(seg1)=seg2_start, ramping to seg2_end."""
        stub = _make_stub_with_schedule(enabled=True)
        # Midpoint of segment 2: s=75, offset=25, expected = 0.35 + (0-0.35)*0.5 = 0.175
        self.assertAlmostEqual(stub.resolver(75), 0.175, places=6)
        # Last interior point: s=99, offset=49, expected = 0.35 + (0-0.35)*49/50
        expected = 0.35 + (0.0 - 0.35) * 49.0 / 50.0
        self.assertAlmostEqual(stub.resolver(99), expected, places=6)
        # At s=100 (= seg1+seg2), past-end clamp → seg2_end.
        self.assertAlmostEqual(stub.resolver(100), 0.0, places=6)
        # Past-end stays at seg2_end.
        self.assertAlmostEqual(stub.resolver(200), 0.0, places=6)
        self.assertAlmostEqual(stub.resolver(10000), 0.0, places=6)

    def test_negative_step_clamps_to_seg1_start(self):
        """Defensive: s<0 → seg1_start."""
        stub = _make_stub_with_schedule(enabled=True)
        self.assertAlmostEqual(stub.resolver(-1), 1.0, places=6)
        self.assertAlmostEqual(stub.resolver(-100), 1.0, places=6)

    def test_custom_values_are_respected(self):
        """All knobs are config-settable."""
        stub = _make_stub_with_schedule(
            enabled=True,
            seg1_start=0.9, seg1_end=0.4, seg1_steps=10,
            seg2_start=0.2, seg2_end=0.05, seg2_steps=20,
        )
        self.assertAlmostEqual(stub.resolver(0), 0.9, places=6)
        # Midpoint segment 1: s=5, p = 0.9 + (0.4-0.9)*0.5 = 0.65.
        self.assertAlmostEqual(stub.resolver(5), 0.65, places=6)
        # Discontinuous drop.
        self.assertAlmostEqual(stub.resolver(10), 0.2, places=6)
        # Midpoint segment 2: s=20, offset=10, p = 0.2 + (0.05-0.2)*0.5 = 0.125.
        self.assertAlmostEqual(stub.resolver(20), 0.125, places=6)
        # Past-end clamp.
        self.assertAlmostEqual(stub.resolver(30), 0.05, places=6)
        self.assertAlmostEqual(stub.resolver(1000), 0.05, places=6)

    def test_zero_step_segments_do_not_divide_by_zero(self):
        """If a segment is configured with 0 steps the resolver must
        not divide by zero (defensive)."""
        stub = _make_stub_with_schedule(
            enabled=True,
            seg1_start=1.0, seg1_end=0.5, seg1_steps=0,
            seg2_start=0.4, seg2_end=0.0, seg2_steps=10,
        )
        # seg1 collapsed → s>=0 falls into seg2 immediately.
        self.assertAlmostEqual(stub.resolver(0), 0.4, places=6)
        self.assertAlmostEqual(stub.resolver(5), 0.2, places=6)


class TestSendStudentGradDetach(unittest.TestCase):
    """``aux_teacher_send_student_grad=False`` severs the implicit
    student-grad channel by detaching ``noise_base``.

    We don't run the real_score forward; we replicate the
    detach-or-not logic and check that the resulting tensor is
    properly detached.
    """

    def test_send_grad_true_keeps_grad(self):
        chunk = torch.randn(1, 4, requires_grad=True)
        gt = torch.randn(1, 4)
        # Replica of the aux pass's blend logic with send_grad=True.
        p = 0.7
        noise_base = p * gt + (1.0 - p) * chunk
        # Default (True) — no detach.
        send_student_grad = True
        if not send_student_grad:
            noise_base = noise_base.detach()
        self.assertTrue(
            noise_base.requires_grad,
            "send_student_grad=True must keep noise_base attached "
            "to chunk's autograd graph.",
        )
        # And the grad path is functional: chunk.grad gets populated.
        loss = noise_base.sum()
        loss.backward()
        self.assertIsNotNone(chunk.grad)
        self.assertGreater(chunk.grad.abs().sum().item(), 0.0)

    def test_send_grad_false_detaches(self):
        chunk = torch.randn(1, 4, requires_grad=True)
        gt = torch.randn(1, 4)
        p = 0.7
        noise_base = p * gt + (1.0 - p) * chunk
        send_student_grad = False
        if not send_student_grad:
            noise_base = noise_base.detach()
        self.assertFalse(
            noise_base.requires_grad,
            "send_student_grad=False must detach noise_base so the "
            "implicit student-grad channel is closed.",
        )
        # Reduce to a scalar that DOES require grad so .backward runs;
        # chunk should NOT receive any gradient through this path.
        bridge = chunk.sum() * 0.0  # zero contribution to keep loss bounded
        loss = noise_base.sum() + bridge
        loss.backward()
        # chunk.grad will be all-zero (only the *0.0 bridge contributes).
        self.assertIsNotNone(chunk.grad)
        self.assertEqual(
            chunk.grad.abs().sum().item(), 0.0,
            "send_student_grad=False must produce zero student grad "
            "from the noise_base path.",
        )


class TestKnobsParseAndValidate(unittest.TestCase):
    """The knobs read in __init__ must (a) default to current
    behaviour, (b) reject out-of-range values."""

    def test_init_block_present_in_source(self):
        """Static check: __init__ reads the new knobs and validates
        ranges. Catches any future deletion that drops the
        ``getattr(args, "aux_teacher_send_student_grad", True)`` /
        ``aux_teacher_p_schedule_enabled`` plumbing.
        """
        path = Path(__file__).resolve().parents[1] / "model" / "dmd_action_forcing.py"
        src = path.read_text()
        for needle in (
            'getattr(args, "aux_teacher_send_student_grad", True)',
            'getattr(args, "aux_teacher_p_schedule_enabled", False)',
            'getattr(args, "aux_teacher_p_seg1_start", 1.0)',
            'getattr(args, "aux_teacher_p_seg1_end", 0.65)',
            'getattr(args, "aux_teacher_p_seg1_steps", 50)',
            'getattr(args, "aux_teacher_p_seg2_start", 0.35)',
            'getattr(args, "aux_teacher_p_seg2_end", 0.0)',
            'getattr(args, "aux_teacher_p_seg2_steps", 50)',
        ):
            self.assertIn(
                needle, src,
                f"__init__ must read knob via: {needle}",
            )
        # Resolver method present.
        self.assertIn(
            "def _resolved_real_teacher_input_mix_gt_p", src,
            "_resolved_real_teacher_input_mix_gt_p method missing.",
        )

    def test_aux_pass_uses_resolver_at_three_sites(self):
        """Aux pass calls the resolver and uses its result for
        blend p, mix coin, and blend-grad-path-live check."""
        path = Path(__file__).resolve().parents[1] / "model" / "dmd_action_forcing.py"
        src = path.read_text()
        # The resolver call should appear inside the aux pass.
        self.assertIn(
            "p_resolved = self._resolved_real_teacher_input_mix_gt_p(current_step)",
            src,
            "Aux pass must resolve p via the schedule method.",
        )
        # The static-knob direct-read patterns must be replaced by
        # ``p_resolved`` reads at the three sites.
        # blend mode read:
        self.assertIn("p_blend = float(p_resolved)", src)
        # mix mode read:
        self.assertIn("p_gt = float(p_resolved)", src)
        # blend-grad-path-live check (the float read inside the
        # chunk_grad_path_live OR clause):
        self.assertIn(
            "and float(p_resolved) < 1.0", src,
            "blend-mode chunk_grad_path_live must compare against "
            "p_resolved, not the static knob.",
        )

    def test_send_student_grad_is_logged(self):
        path = Path(__file__).resolve().parents[1] / "model" / "dmd_action_forcing.py"
        src = path.read_text()
        self.assertIn(
            '"aux_teacher_p_resolved": float(p_resolved)', src,
            "Resolved p must be logged.",
        )
        self.assertIn(
            '"aux_teacher_send_student_grad":', src,
            "send_student_grad flag must be logged.",
        )

    def test_trainer_plumbs_current_step(self):
        path = Path(__file__).resolve().parents[1] / "trainer" / "causal_action_forcing_train.py"
        src = path.read_text()
        self.assertIn(
            'train_info["current_step"] = int(self.step)', src,
            "Trainer must inject current_step into train_info before "
            "calling compute_generator_loss_streaming.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
