"""Coverage for the first-moment (MEAN) term of ``compute_stat_anchor_loss``.

``model/anti_collapse.py``'s ``compute_stat_anchor_loss`` had ZERO test
coverage repo-wide before this file, and the 2026-08-25 MEAN term
(``_per_frame_MEAN`` + ``MEAN_short_weight`` / ``MEAN_long_weight`` +
``seed_MEAN_anchor``) is the newest thing in it while 8-node jobs are
queued against the tree.

What is pinned here:

  1. ``_per_frame_MEAN`` is the SIGNED per-channel spatial mean, shape
     ``[B, F, C]`` -- and specifically NOT ``_per_frame_M1`` (the naming
     trap documented in the function's docstring: "M1" is per-channel
     ENERGY).
  2. DEFAULT-OFF IS FREE. With ``MEAN_*_weight == 0.0`` the returned loss
     and the gradient w.r.t. ``pred_x0`` are BIT-IDENTICAL to the call
     that predates the term (no ``seed_MEAN_anchor`` at all), the anchor
     value cannot leak into either, and ``_per_frame_MEAN`` is never even
     evaluated on the prediction.
  3. A non-zero weight adds EXACTLY ``w * MSE(mean)`` (checked at
     ``rel_tol == 0`` so the floor is out of the way).
  4. The DELIBERATELY DIFFERENT FLOOR. Every other stat in the function
     is non-negative so ``anchor.mean()**2`` is a sane deadband scale;
     the mean is SIGNED and its per-channel values largely cancel, so the
     MEAN term uses ``anchor.abs().mean()**2`` instead. Pinned two ways:
     it is provably NOT the signed form on a sign-cancelling anchor (the
     signed form would collapse the deadband to ~0), and it is exactly
     inert at ``rel_tol == 0``.
  5. Gradient reaches ``pred_x0``; the anchor never receives gradient.
  6. The ``stat/MEAN_*`` telemetry keys are all emitted, and
     ``MEAN_requested`` / ``MEAN_term_built`` separate "weight is 0" from
     "weight > 0 but the anchor arrived None" (the silent-drop case).

CPU-only, float64, no CUDA, no dataset.

Run:
    python -m pytest testing/test_stat_anchor_mean_term.py -q
or
    python testing/test_stat_anchor_mean_term.py
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model.anti_collapse as AC  # noqa: E402
from model.anti_collapse import (  # noqa: E402
    _causal_cumulative_mean,
    _per_frame_M1,
    _per_frame_MEAN,
    compute_stat_anchor_loss,
)

B, F_, C, H, W = 2, 4, 3, 5, 5
DT = torch.float64


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _pred(seed=0, requires_grad=True):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, F_, C, H, W, generator=g, dtype=DT)
    x.requires_grad_(requires_grad)
    return x


def _seed_latents(seed=1, n=3):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, n, C, H, W, generator=g, dtype=DT)


def _precomputed_anchors(seed_lat):
    """The five anchors that predate the MEAN term, exactly as the
    ``seed_latents`` branch derives them."""
    with torch.no_grad():
        return dict(
            seed_STD_anchor=AC._per_frame_STD(seed_lat).mean(dim=1),
            seed_M2_anchor=AC._per_frame_M2(seed_lat).mean(dim=1),
            seed_TV_anchor=AC._per_frame_TV(seed_lat).mean(dim=1),
            seed_SOS_anchor=AC._per_frame_SOS(seed_lat).mean(dim=1),
            seed_M1_anchor=AC._per_frame_M1(seed_lat).mean(dim=1),
        )


# Weights with every OTHER stat live, so a leak in the MEAN term has to be
# separated from a genuinely non-trivial baseline loss.
_LIVE = dict(
    STD_short_weight=0.3, STD_long_weight=0.2,
    M2_short_weight=0.1, M2_long_weight=0.05,
    TV_short_weight=0.7, TV_long_weight=0.4,
    SOS_short_weight=1e-4, SOS_long_weight=2e-4,
    M1_short_weight=1e-3, M1_long_weight=5e-4,
)


def _loss_and_grad(pred, **kw):
    loss, logs = compute_stat_anchor_loss(pred, **kw)
    if pred.grad is not None:
        pred.grad = None
    loss.backward()
    return loss.detach().clone(), pred.grad.detach().clone(), logs


# ---------------------------------------------------------------------------
# 1. _per_frame_MEAN itself
# ---------------------------------------------------------------------------
def test_per_frame_MEAN_shape_and_value():
    x = _pred(requires_grad=False)
    m = _per_frame_MEAN(x)
    assert m.shape == (B, F_, C)
    assert torch.equal(m, x.mean(dim=[3, 4]))


def test_per_frame_MEAN_is_signed_and_is_not_M1():
    """The naming trap: ``_per_frame_M1`` is per-channel ENERGY (never
    negative); ``_per_frame_MEAN`` is the SIGNED first moment."""
    x = -torch.ones(B, F_, C, H, W, dtype=DT) * 2.0
    mean = _per_frame_MEAN(x)
    m1 = _per_frame_M1(x)
    assert torch.all(mean < 0), "MEAN must keep the sign of the input"
    assert torch.all(m1 > 0), "M1 is a raw second moment and cannot be < 0"
    assert mean.shape == m1.shape == (B, F_, C)
    # ... and they are not merely scaled versions of one another.
    assert not torch.allclose(mean, m1 / (H * W))


def test_per_frame_MEAN_sign_does_not_cancel_across_channels():
    """abs()-ing inside the stat would hide a per-channel sign flip; the
    docstring says it deliberately does not."""
    x = torch.zeros(1, 1, 2, H, W, dtype=DT)
    x[0, 0, 0] = 1.0
    x[0, 0, 1] = -1.0
    m = _per_frame_MEAN(x)
    assert float(m[0, 0, 0]) == pytest.approx(1.0)
    assert float(m[0, 0, 1]) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# 2. default-off is byte-identical AND gradient-identical
# ---------------------------------------------------------------------------
def test_zero_weight_is_bit_identical_to_the_pre_term_call():
    """The call that predates the term passes NO ``seed_MEAN_anchor``.
    With the default weights (0.0) the new signature must reproduce it in
    the loss value AND in ``d loss / d pred``, bit for bit."""
    sl = _seed_latents()
    anchors = _precomputed_anchors(sl)
    mean_anchor = _per_frame_MEAN(sl).mean(dim=1)

    p0 = _pred()
    legacy_loss, legacy_grad, legacy_logs = _loss_and_grad(
        p0, **anchors, **_LIVE)              # no seed_MEAN_anchor at all

    p1 = _pred()
    new_loss, new_grad, new_logs = _loss_and_grad(
        p1, **anchors, seed_MEAN_anchor=mean_anchor, **_LIVE)

    assert torch.equal(legacy_loss, new_loss)
    assert torch.equal(legacy_grad, new_grad)
    # and the telemetry says the term was NOT built in either case
    assert float(legacy_logs["stat/MEAN_term_built"]) == 0.0
    assert float(new_logs["stat/MEAN_term_built"]) == 0.0
    assert float(new_logs["stat/MEAN_requested"]) == 0.0


def test_zero_weight_cannot_leak_the_anchor_value():
    """A term that leaked in at weight 0 would move when the anchor moves."""
    sl = _seed_latents()
    anchors = _precomputed_anchors(sl)
    a = _per_frame_MEAN(sl).mean(dim=1)

    p0 = _pred()
    l0, g0, _ = _loss_and_grad(p0, **anchors, seed_MEAN_anchor=a, **_LIVE)
    p1 = _pred()
    l1, g1, _ = _loss_and_grad(
        p1, **anchors, seed_MEAN_anchor=a * 1000.0 + 7.0, **_LIVE)

    assert torch.equal(l0, l1)
    assert torch.equal(g0, g1)


def test_zero_weight_never_evaluates_MEAN_on_the_prediction():
    """Cheapest possible proof of "the term is not built": count calls.
    One call is expected (deriving the anchor from ``seed_latents``, under
    no_grad); zero calls may touch ``pred_x0``."""
    calls = []
    real = AC._per_frame_MEAN

    def _spy(x):
        calls.append(tuple(x.shape))
        return real(x)

    AC._per_frame_MEAN = _spy
    try:
        pred = _pred()
        compute_stat_anchor_loss(pred, seed_latents=_seed_latents(), **_LIVE)
    finally:
        AC._per_frame_MEAN = real

    assert calls == [(B, 3, C, H, W)], (
        f"_per_frame_MEAN call log {calls}: expected exactly one call, on "
        f"the seed latents, and none on pred_x0"
    )


def test_weight_requested_without_anchor_is_reported_not_silent():
    """The silent-no-op guard: weight > 0 but the anchor arrived None."""
    sl = _seed_latents()
    anchors = _precomputed_anchors(sl)
    pred = _pred(requires_grad=False)
    loss, logs = compute_stat_anchor_loss(
        pred, **anchors, MEAN_short_weight=1.0, MEAN_long_weight=1.0)
    assert float(logs["stat/MEAN_requested"]) == 1.0
    assert float(logs["stat/MEAN_term_built"]) == 0.0
    assert float(logs["stat/MEAN_active_short"]) == 0.0
    assert float(logs["stat/MEAN_active_long"]) == 0.0


# ---------------------------------------------------------------------------
# 3. a non-zero weight adds exactly w * MSE(mean)
# ---------------------------------------------------------------------------
def _mean_mses(pred, anchor):
    pf = _per_frame_MEAN(pred)                    # [B, F, C]
    lg = _causal_cumulative_mean(pf)              # [B, F, C]
    a = anchor.to(pf.dtype)
    a = a if a.dim() == 3 else a.unsqueeze(1)
    return (pf - a).pow(2).mean(), (lg - a).pow(2).mean()


def test_nonzero_weight_adds_exactly_w_times_mse():
    """At ``rel_tol == 0`` the floor is 0, so the added contribution is
    the raw weighted MSE and nothing else. Every other weight is 0 here so
    the sum is exact in float64."""
    sl = _seed_latents()
    anchors = _precomputed_anchors(sl)
    a = _per_frame_MEAN(sl).mean(dim=1)
    ws, wl = 0.37, 2.5
    off = dict(
        STD_short_weight=0.0, STD_long_weight=0.0,
        M2_short_weight=0.0, M2_long_weight=0.0,
        TV_short_weight=0.0, TV_long_weight=0.0,
    )

    pred = _pred(requires_grad=False)
    loss, logs = compute_stat_anchor_loss(
        pred, **anchors, seed_MEAN_anchor=a,
        MEAN_short_weight=ws, MEAN_long_weight=wl,
        rel_tol_short=0.0, rel_tol_long=0.0, **off)

    mse_s, mse_l = _mean_mses(pred, a)
    expected = ws * mse_s + wl * mse_l
    assert torch.equal(loss, expected), (
        f"loss {loss.item()!r} != w*MSE {expected.item()!r}")
    assert float(logs["stat/MEAN_term_built"]) == 1.0
    assert float(logs["stat/MEAN_mse_short"]) == pytest.approx(
        float(mse_s), rel=0, abs=0)
    assert float(logs["stat/MEAN_mse_long"]) == pytest.approx(
        float(mse_l), rel=0, abs=0)


def test_short_and_long_weights_are_wired_to_their_own_horizons():
    """Swapping which weight is non-zero must select a DIFFERENT MSE --
    catches a copy-paste that wires both to the short horizon."""
    sl = _seed_latents()
    anchors = _precomputed_anchors(sl)
    a = _per_frame_MEAN(sl).mean(dim=1)
    off = dict(
        STD_short_weight=0.0, STD_long_weight=0.0,
        M2_short_weight=0.0, M2_long_weight=0.0,
        TV_short_weight=0.0, TV_long_weight=0.0,
        rel_tol_short=0.0, rel_tol_long=0.0)
    pred = _pred(requires_grad=False)
    mse_s, mse_l = _mean_mses(pred, a)
    assert not torch.isclose(mse_s, mse_l), "degenerate fixture"

    only_s, _ = compute_stat_anchor_loss(
        pred, **anchors, seed_MEAN_anchor=a, MEAN_short_weight=1.0, **off)
    only_l, _ = compute_stat_anchor_loss(
        pred, **anchors, seed_MEAN_anchor=a, MEAN_long_weight=1.0, **off)
    assert torch.equal(only_s, mse_s)
    assert torch.equal(only_l, mse_l)


def test_per_frame_anchor_rank3_is_used_as_is():
    """``[B, F, C]`` (matched-GT mode) must NOT be unsqueezed."""
    sl = _seed_latents()
    anchors = _precomputed_anchors(sl)
    a2 = _per_frame_MEAN(sl).mean(dim=1)                    # [B, C]
    a3 = a2.unsqueeze(1).expand(B, F_, C).contiguous()      # [B, F, C]
    off = dict(
        STD_short_weight=0.0, STD_long_weight=0.0,
        M2_short_weight=0.0, M2_long_weight=0.0,
        TV_short_weight=0.0, TV_long_weight=0.0,
        rel_tol_short=0.0, rel_tol_long=0.0, MEAN_short_weight=1.0)
    pred = _pred(requires_grad=False)
    l2, _ = compute_stat_anchor_loss(
        pred, **anchors, seed_MEAN_anchor=a2, **off)
    l3, _ = compute_stat_anchor_loss(
        pred, **anchors, seed_MEAN_anchor=a3, **off)
    assert torch.equal(l2, l3)


# ---------------------------------------------------------------------------
# 4. the deliberately different floor
# ---------------------------------------------------------------------------
def _sign_cancelling_anchor():
    """Per-channel means that cancel: ``mean() == 0`` but
    ``abs().mean() == 0.5``. This is the regime the deviation exists for."""
    a = torch.tensor([[0.5, -0.5, 0.5], [-0.5, 0.5, -0.5]], dtype=DT)
    assert float(a.mean()) == 0.0
    assert float(a.abs().mean()) == 0.5
    return a


def test_floor_uses_abs_mean_not_signed_mean():
    sl = _seed_latents()
    anchors = _precomputed_anchors(sl)
    a = _sign_cancelling_anchor()
    rs, rl = 0.20, 0.10
    pred = _pred(requires_grad=False)
    _, logs = compute_stat_anchor_loss(
        pred, **anchors, seed_MEAN_anchor=a,
        MEAN_short_weight=1.0, MEAN_long_weight=1.0,
        rel_tol_short=rs, rel_tol_long=rl)

    want_s = rs ** 2 * float(a.abs().mean()) ** 2
    want_l = rl ** 2 * float(a.abs().mean()) ** 2
    naive_s = rs ** 2 * float(a.mean()) ** 2       # == 0.0, the collapsed band
    assert float(logs["stat/MEAN_floor_short"]) == pytest.approx(want_s)
    assert float(logs["stat/MEAN_floor_long"]) == pytest.approx(want_l)
    assert naive_s == 0.0
    assert float(logs["stat/MEAN_floor_short"]) > 0.0, (
        "the signed-mean floor would be exactly zero here -- rel_tol "
        "would be inoperative, which is the whole reason for abs()")


def test_floor_creates_a_real_deadband():
    """Inside the band: raw MSE > 0 but the ACTIVE (gradient-carrying)
    loss is exactly 0. With the signed-mean floor this anchor's deadband
    would be ~0 and the term would pull."""
    sl = _seed_latents()
    anchors = _precomputed_anchors(sl)
    a = _sign_cancelling_anchor()
    # pred whose per-channel means sit a little away from the anchor
    pred = torch.zeros(B, F_, C, H, W, dtype=DT)
    for b in range(B):
        for c in range(C):
            pred[b, :, c] = a[b, c] + 0.02
    loss, logs = compute_stat_anchor_loss(
        pred, **anchors, seed_MEAN_anchor=a,
        MEAN_short_weight=1.0, MEAN_long_weight=1.0,
        rel_tol_short=0.20, rel_tol_long=0.10,
        STD_short_weight=0.0, STD_long_weight=0.0,
        M2_short_weight=0.0, M2_long_weight=0.0,
        TV_short_weight=0.0, TV_long_weight=0.0)
    assert float(logs["stat/MEAN_mse_short"]) > 0.0
    assert float(logs["stat/MEAN_active_short"]) == 0.0
    assert float(logs["stat/MEAN_active_long"]) == 0.0
    assert float(loss) == 0.0


def test_floor_is_inert_at_rel_tol_zero():
    sl = _seed_latents()
    anchors = _precomputed_anchors(sl)
    a = _sign_cancelling_anchor()
    pred = _pred(requires_grad=False)
    _, logs = compute_stat_anchor_loss(
        pred, **anchors, seed_MEAN_anchor=a,
        MEAN_short_weight=1.0, MEAN_long_weight=1.0,
        rel_tol_short=0.0, rel_tol_long=0.0)
    assert float(logs["stat/MEAN_floor_short"]) == 0.0
    assert float(logs["stat/MEAN_floor_long"]) == 0.0
    assert float(logs["stat/MEAN_active_short"]) == pytest.approx(
        float(logs["stat/MEAN_mse_short"]))
    assert float(logs["stat/MEAN_active_long"]) == pytest.approx(
        float(logs["stat/MEAN_mse_long"]))


def test_floor_clamp_never_goes_negative():
    sl = _seed_latents()
    anchors = _precomputed_anchors(sl)
    a = _per_frame_MEAN(sl).mean(dim=1)
    pred = _pred(requires_grad=False)
    _, logs = compute_stat_anchor_loss(
        pred, **anchors, seed_MEAN_anchor=a,
        MEAN_short_weight=1.0, MEAN_long_weight=1.0,
        rel_tol_short=50.0, rel_tol_long=50.0)   # absurd band -> fully inside
    assert float(logs["stat/MEAN_active_short"]) == 0.0
    assert float(logs["stat/MEAN_active_long"]) == 0.0


# ---------------------------------------------------------------------------
# 5. gradients
# ---------------------------------------------------------------------------
def test_gradient_flows_to_the_prediction():
    sl = _seed_latents()
    anchors = _precomputed_anchors(sl)
    a = _per_frame_MEAN(sl).mean(dim=1) + 3.0     # force it out of band
    pred = _pred()
    loss, grad, logs = _loss_and_grad(
        pred, **anchors, seed_MEAN_anchor=a,
        MEAN_short_weight=1.0, MEAN_long_weight=1.0,
        rel_tol_short=0.0, rel_tol_long=0.0,
        STD_short_weight=0.0, STD_long_weight=0.0,
        M2_short_weight=0.0, M2_long_weight=0.0,
        TV_short_weight=0.0, TV_long_weight=0.0)
    assert float(loss) > 0.0
    assert grad.shape == pred.shape
    assert torch.isfinite(grad).all()
    assert float(grad.abs().max()) > 0.0
    # the pull is toward the anchor: pred mean is below the anchor, so the
    # gradient (which is descended) must be negative everywhere.
    assert torch.all(grad < 0.0)


def test_anchor_receives_no_gradient():
    sl = _seed_latents()
    anchors = _precomputed_anchors(sl)
    a = (_per_frame_MEAN(sl).mean(dim=1) + 3.0).requires_grad_(True)
    pred = _pred()
    loss, _ = compute_stat_anchor_loss(
        pred, **anchors, seed_MEAN_anchor=a,
        MEAN_short_weight=1.0, MEAN_long_weight=1.0,
        rel_tol_short=0.0, rel_tol_long=0.0)
    loss.backward()
    assert a.grad is None, "the MEAN anchor must be detached"


def test_seed_latents_path_detaches_the_seed():
    sl = _seed_latents().requires_grad_(True)
    pred = _pred()
    loss, logs = compute_stat_anchor_loss(
        pred, seed_latents=sl,
        MEAN_short_weight=1.0, MEAN_long_weight=1.0,
        rel_tol_short=0.0, rel_tol_long=0.0)
    loss.backward()
    assert sl.grad is None
    assert pred.grad is not None
    assert float(logs["stat/MEAN_term_built"]) == 1.0


def test_seed_latents_path_derives_the_same_anchor():
    sl = _seed_latents()
    pred = _pred(requires_grad=False)
    _, logs = compute_stat_anchor_loss(
        pred, seed_latents=sl, MEAN_short_weight=1.0)
    a = _per_frame_MEAN(sl).mean(dim=1)
    assert float(logs["stat/MEAN_anchor"]) == pytest.approx(float(a.mean()))
    assert float(logs["stat/MEAN_anchor_abs"]) == pytest.approx(
        float(a.abs().mean()))


# ---------------------------------------------------------------------------
# 6. telemetry keys
# ---------------------------------------------------------------------------
_MEAN_KEYS = (
    "stat/MEAN_anchor", "stat/MEAN_anchor_abs",
    "stat/MEAN_mse_short", "stat/MEAN_mse_long",
    "stat/MEAN_floor_short", "stat/MEAN_floor_long",
    "stat/MEAN_active_short", "stat/MEAN_active_long",
    "stat/MEAN_requested", "stat/MEAN_term_built",
)


def test_mean_telemetry_keys_always_present():
    sl = _seed_latents()
    anchors = _precomputed_anchors(sl)
    pred = _pred(requires_grad=False)
    for kw in (
        dict(seed_latents=sl),                                  # term off
        dict(seed_latents=sl, MEAN_short_weight=1.0),           # term on
        dict(**anchors, MEAN_short_weight=1.0),                 # anchor None
        dict(**anchors),                                        # legacy call
    ):
        _, logs = compute_stat_anchor_loss(pred, **kw)
        for k in _MEAN_KEYS:
            assert k in logs, f"missing telemetry key {k} for {sorted(kw)}"
            assert isinstance(logs[k], torch.Tensor)
            assert not logs[k].requires_grad, f"{k} must be detached"


def test_anchor_abs_telemetry_separates_from_signed_anchor():
    """``stat/MEAN_anchor`` alone reads ~0 on a healthy signed anchor --
    that is exactly why ``_abs`` exists."""
    sl = _seed_latents()
    anchors = _precomputed_anchors(sl)
    a = _sign_cancelling_anchor()
    pred = _pred(requires_grad=False)
    _, logs = compute_stat_anchor_loss(
        pred, **anchors, seed_MEAN_anchor=a, MEAN_short_weight=1.0)
    assert float(logs["stat/MEAN_anchor"]) == pytest.approx(0.0)
    assert float(logs["stat/MEAN_anchor_abs"]) == pytest.approx(0.5)


def test_rank_and_shape_validation_still_raises():
    with pytest.raises(ValueError):
        compute_stat_anchor_loss(torch.randn(2, 3, 4), seed_latents=None)
    with pytest.raises(ValueError):
        compute_stat_anchor_loss(_pred(requires_grad=False))


# ---------------------------------------------------------------------------
def main():
    g = dict(globals())
    names = [n for n in g if n.startswith("test_")]
    names.sort(key=lambda n: g[n].__code__.co_firstlineno)
    for n in names:
        g[n]()
        print(f"  ok  {n}")
    print(f"\nALL {len(names)} TESTS PASSED")


if __name__ == "__main__":
    main()
