"""Tests for the FT_v3 phase-2 rolling changes:

  * the STRUCTURAL GAN memory fix (deferring the discriminator update past
    ``generator_loss.backward()``) — verifies the *principle* that a
    detached-input optimizer step is ORDER-INDEPENDENT w.r.t. an unrelated
    backward, so deferring it cannot change the disc result;
  * the TOOTHPASTE depth-curriculum pure helpers (grow / gone / s_local_max).

Run:  pytest testing/test_toothpaste_and_defer_disc.py -q
"""
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the torch-free curriculum helpers directly (the trainer module
# CUDA-inits at import, so we test the extracted pure module — which the
# trainer delegates to verbatim).
from trainer import toothpaste as T


# --------------------------------------------------------------------------
# Structural fix: a detached-input disc update gives identical results whether
# it runs BEFORE or AFTER an unrelated gen backward. This is the exact
# property that makes ``ladd_defer_disc_update`` safe (the real disc update
# reads only detached chunks/actions/GT, so deferring it past the freed gen
# graph is result-preserving — it only changes the peak memory).
# --------------------------------------------------------------------------
def _disc_step(disc, opt, real_det, fake_det):
    """One disc update on DETACHED inputs (mirrors the real loop's shape:
    zero_grad -> loss on detached real/fake -> backward -> step)."""
    opt.zero_grad(set_to_none=True)
    loss = (disc(fake_det).mean() - disc(real_det).mean()
            + 0.1 * (disc(real_det) ** 2).mean())   # + an R1-like penalty
    loss.backward()
    opt.step()
    return float(loss.detach())


def test_disc_update_order_independence():
    torch.manual_seed(0)
    gen = torch.nn.Linear(4, 4)
    disc = torch.nn.Linear(4, 1)
    x = torch.randn(8, 4)
    real_det = torch.randn(8, 4)          # detached disc inputs
    fake_det = torch.randn(8, 4)

    # ---- Path A: disc update FIRST, then gen backward (current inline). ----
    genA, discA = copy.deepcopy(gen), copy.deepcopy(disc)
    optA = torch.optim.SGD(discA.parameters(), lr=0.1)
    _disc_step(discA, optA, real_det, fake_det)
    gen_lossA = genA(x).sum()
    gen_lossA.backward()
    gradA = [p.grad.clone() for p in genA.parameters()]
    discA_params = [p.detach().clone() for p in discA.parameters()]

    # ---- Path B: gen backward FIRST, then disc update (deferred fix). ----
    genB, discB = copy.deepcopy(gen), copy.deepcopy(disc)
    optB = torch.optim.SGD(discB.parameters(), lr=0.1)
    gen_lossB = genB(x).sum()
    gen_lossB.backward()
    _disc_step(discB, optB, real_det, fake_det)
    gradB = [p.grad.clone() for p in genB.parameters()]
    discB_params = [p.detach().clone() for p in discB.parameters()]

    # Disc params identical -> deferral did not change the disc update.
    for a, b in zip(discA_params, discB_params):
        assert torch.equal(a, b), "disc update changed under deferral"
    # Gen grads identical -> the gen backward is unaffected by disc ordering.
    for a, b in zip(gradA, gradB):
        assert torch.equal(a, b), "gen grad changed under deferral"


# --------------------------------------------------------------------------
# Toothpaste depth-growth helper.
# --------------------------------------------------------------------------
def test_grow_waits_for_full_window():
    # Window not yet full -> never grow, prev_avg unchanged.
    grow, base = T.toothpaste_grow([0.4, 0.4], win=5, prev_avg=None, tol=0.1)
    assert grow is False and base is None


def test_grow_first_depth_establishes_baseline():
    # prev_avg None + full window -> grow and set baseline = window mean.
    win = [0.4, 0.5, 0.6, 0.5, 0.5]   # mean 0.5
    grow, base = T.toothpaste_grow(win, win=5, prev_avg=None, tol=0.1)
    assert grow is True
    assert abs(base - 0.5) < 1e-9


def test_grow_when_within_tolerance():
    # mean 0.55 <= prev 0.5 * 1.1 = 0.55 -> grow (boundary inclusive).
    win = [0.55] * 5
    grow, base = T.toothpaste_grow(win, win=5, prev_avg=0.5, tol=0.1)
    assert grow is True and abs(base - 0.55) < 1e-9


def test_no_grow_when_above_tolerance():
    # mean 0.60 > prev 0.5 * 1.1 = 0.55 -> hold, baseline unchanged.
    win = [0.60] * 5
    grow, base = T.toothpaste_grow(win, win=5, prev_avg=0.5, tol=0.1)
    assert grow is False and base == 0.5


# --------------------------------------------------------------------------
# Toothpaste gone (off-manifold) helper.
# --------------------------------------------------------------------------
def test_gone_none_baseline_never_fires():
    assert T.toothpaste_gone(99.0, None, 3.0) is False


def test_gone_fires_above_threshold():
    assert T.toothpaste_gone(0.61, 0.2, 3.0) is True    # 0.61 > 0.6
    assert T.toothpaste_gone(0.59, 0.2, 3.0) is False   # 0.59 < 0.6


# --------------------------------------------------------------------------
# s_local_max floor: the rejection roll_cap < anchor+min_new must be
# unreachable for EVERY s in [0, s_local_max] (the DDP-hang fix). Replays the
# exact setup arithmetic and asserts the floor holds across the whole range.
# --------------------------------------------------------------------------
def _roll_cap_at(s, ride_len, cf, cap, slack, npb):
    actual_cap = min(cap, ride_len - cf - s)
    actual_cap = (actual_cap // npb) * npb
    return actual_cap - slack


def test_s_local_max_reserves_floor():
    npb, cf, cap = 3, 18, 300
    anchor, min_new = npb, 9          # band-3 -> min_new = 9
    for ride_len in (120, 240, 480, 900):
        for slack in (npb, 2 * npb):  # rolling, and rolling+forward-clean
            s_max = T.setup_s_local_max(
                ride_len, cf, cap, slack, anchor, min_new, npb)
            floor = anchor + min_new
            for s in range(0, s_max + 1, npb):
                assert _roll_cap_at(s, ride_len, cf, cap, slack, npb) >= floor, (
                    f"roll_cap below floor at s={s} "
                    f"(ride_len={ride_len}, slack={slack})"
                )


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))


# ---- FT_v3 post-build two-threshold going/gone gate -----------------------

def test_going_gone_gate_gone_takes_priority():
    # off-manifold collapse wins even when past the going threshold
    assert T.going_gone_gate(10.0, 0.5, 0.2, 3.0, min_depth=2, cur_depth=5) == "gone"


def test_going_gone_gate_min_depth_holds_roll():
    # below min_depth we keep rolling even if over going (gone still aborts)
    assert T.going_gone_gate(0.6, 0.5, 1.0, 3.0, min_depth=4, cur_depth=2) == "roll"


def test_going_gone_gate_going_stops():
    assert T.going_gone_gate(0.6, 0.5, 1.0, 3.0, min_depth=2, cur_depth=4) == "going"


def test_going_gone_gate_below_going_rolls():
    assert T.going_gone_gate(0.3, 0.5, 1.0, 3.0, min_depth=2, cur_depth=4) == "roll"


def test_going_gone_gate_disabled_never_going():
    assert T.going_gone_gate(0.9, 0.0, 1.0, 3.0, min_depth=2, cur_depth=4) == "roll"


def test_going_gone_gate_no_baseline_not_gone():
    assert T.going_gone_gate(5.0, 0.5, None, 3.0, min_depth=2, cur_depth=4) == "going"
