"""Coverage for ``_gt_window_stat_anchors`` and its new
``stat_anchor_use_clean_match_offset`` behaviour (default TRUE).

``model/dmd_action_forcing.py:_gt_window_stat_anchors`` builds the
per-frame GT stat anchors for ``stat_anchor_mode='gt_window'``. On
2026-08-25 it gained the matched-clean offset: when
``dmd_42f_clean_match_enabled`` is on, the GT window is shifted by
``streaming_state['clean_match_offset']`` so the statistics the generator
is trained toward come from the GT frames its output was actually matched
against -- the same ``m``, with the same SIGN and the same BASE, that
``_build_42f_scoring_inputs`` applies (``chunk_lo = chunk_lo + match_m``).

Pinned here:
  * flag OFF -> byte-identical to the positional (pre-2026-08-25) result,
    for every one of the six stat keys, whatever ``clean_match_offset``
    holds;
  * the gate is a CONJUNCTION: the offset is also inert when
    ``dmd_42f_clean_match_enabled`` is off, which is the state every
    non-matching recipe is in;
  * flag ON -> the result is EXACTLY the positional result computed at
    ``chunk_lo + m``. That is the sign and the base in one assertion (an
    inverted sign or a ``chunk_lo - m`` base fails it);
  * ``self._stat_anchor_applied_offset`` is stashed for logging and is
    never left stale;
  * ride-edge clamping: huge offsets in both directions do not crash, do
    not read out of bounds, and snap to the nearest in-bounds chunk;
  * output shapes are rank-uniform (STD/M2/TV/SOS ``[B,F]``; M1/MEAN
    ``[B,F,C]``), detached, finite;
  * every ``None`` bail-out (no ride window, no pred, short ride, wrong
    rank, too-few frames).

The method is called UNBOUND against a namespace stub -- it reads
everything through ``getattr``/``self.streaming_state``, so no 1.3B model
and no GPU are needed. The module import needs the usual
``torch.cuda.current_device`` patch (``wan.modules.t5`` evaluates it at
import time).

Run:
    python -m pytest testing/test_gt_window_stat_anchor_offset.py -q
or
    python testing/test_gt_window_stat_anchor_offset.py
"""
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with patch.object(torch.cuda, "current_device", return_value=0):
    from model.dmd_action_forcing import ActionForcingDMD  # noqa: E402

from model.anti_collapse import (  # noqa: E402
    _per_frame_M1, _per_frame_M2, _per_frame_MEAN, _per_frame_SOS,
    _per_frame_STD, _per_frame_TV,
)

ANCHORS = ActionForcingDMD._gt_window_stat_anchors

B, C, H, W = 2, 3, 4, 4
NPB = 3
T_RIDE = 60
KEYS = ("STD", "M2", "TV", "SOS", "M1", "MEAN")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _ride(seed=0, T=T_RIDE):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, C, H, W, generator=g)


def _pred(frames=9, seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, frames, C, H, W, generator=g)


def _stub(
    *,
    ride=None,
    match_offset=0,
    use_offset=True,
    match_enabled=True,
    k=3,
    npb=NPB,
):
    ss = {} if ride is None else {"ride_latents_window": ride}
    ss["clean_match_offset"] = match_offset
    return SimpleNamespace(
        streaming_state=ss,
        num_frame_per_block=npb,
        stat_anchor_match_k=k,
        stat_anchor_use_clean_match_offset=use_offset,
        dmd_42f_clean_match_enabled=match_enabled,
    )


def _same(a, b):
    assert a is not None and b is not None
    assert set(a) == set(b) == set(KEYS)
    for k in KEYS:
        assert torch.equal(a[k], b[k]), f"{k} differs"


def _differs(a, b):
    assert a is not None and b is not None
    assert any(not torch.equal(a[k], b[k]) for k in KEYS), (
        "expected the anchors to move, but every stat is identical -- the "
        "fixture is degenerate or the offset was dropped")


# ---------------------------------------------------------------------------
# 1. flag OFF == positional, byte for byte
# ---------------------------------------------------------------------------
def test_flag_off_is_byte_identical_to_positional():
    ride, pred = _ride(), _pred()
    base = ANCHORS(_stub(ride=ride, match_offset=0, use_offset=False), pred, 21)
    for m in (-7, -1, 0, 1, 4, 11):
        got = ANCHORS(
            _stub(ride=ride, match_offset=m, use_offset=False), pred, 21)
        _same(base, got)


def test_flag_off_stashes_zero_offset():
    s = _stub(ride=_ride(), match_offset=5, use_offset=False)
    ANCHORS(s, _pred(), 21)
    assert s._stat_anchor_applied_offset == 0


def test_offset_is_also_inert_when_clean_match_is_disabled():
    """The gate is a CONJUNCTION. A recipe with the (default-TRUE) flag on
    but ``dmd_42f_clean_match_enabled=false`` must stay positional."""
    ride, pred = _ride(), _pred()
    base = ANCHORS(_stub(ride=ride, match_offset=0, use_offset=False), pred, 21)
    s = _stub(ride=ride, match_offset=6, use_offset=True, match_enabled=False)
    got = ANCHORS(s, pred, 21)
    _same(base, got)
    assert s._stat_anchor_applied_offset == 0


# ---------------------------------------------------------------------------
# 2. flag ON: same sign, same base as the 42f builder
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("m", [-6, -3, -1, 1, 3, 6])
def test_flag_on_shifts_the_window_by_plus_m(m):
    """``_build_42f_scoring_inputs`` does ``chunk_lo = chunk_lo + match_m``.
    Applying ``m`` here must therefore equal evaluating the POSITIONAL
    builder at ``chunk_lo + m``: same sign, same base, one assertion."""
    ride, pred = _ride(), _pred()
    lo = 21
    shifted = ANCHORS(
        _stub(ride=ride, match_offset=m, use_offset=True), pred, lo)
    positional_at_lo_plus_m = ANCHORS(
        _stub(ride=ride, match_offset=0, use_offset=False), pred, lo + m)
    _same(shifted, positional_at_lo_plus_m)

    # sanity: the offset actually moved something (otherwise the test above
    # would pass on a resolver that silently forces m=0)
    positional = ANCHORS(
        _stub(ride=ride, match_offset=0, use_offset=False), pred, lo)
    _differs(shifted, positional)


def test_flag_on_would_fail_for_the_opposite_sign():
    """Explicit anti-sign-flip guard: +m and -m must NOT agree."""
    ride, pred = _ride(), _pred()
    plus = ANCHORS(_stub(ride=ride, match_offset=4), pred, 21)
    minus = ANCHORS(_stub(ride=ride, match_offset=-4), pred, 21)
    _differs(plus, minus)


def test_applied_offset_is_stashed_and_never_stale():
    ride, pred = _ride(), _pred()
    s = _stub(ride=ride, match_offset=5)
    ANCHORS(s, pred, 21)
    assert s._stat_anchor_applied_offset == 5
    # a later roll with no offset must clear it, not leave 5 behind
    s.streaming_state["clean_match_offset"] = 0
    ANCHORS(s, pred, 21)
    assert s._stat_anchor_applied_offset == 0


def test_missing_or_none_offset_key_reads_as_zero():
    ride, pred = _ride(), _pred()
    base = ANCHORS(_stub(ride=ride, match_offset=0), pred, 21)
    for val in (None, 0):
        s = _stub(ride=ride, match_offset=val)
        _same(base, ANCHORS(s, pred, 21))
        assert s._stat_anchor_applied_offset == 0
    s = _stub(ride=ride)
    del s.streaming_state["clean_match_offset"]
    _same(base, ANCHORS(s, pred, 21))


# ---------------------------------------------------------------------------
# 3. ride-edge clamping
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "chunk_lo,m",
    [
        (0, -1000),        # far before the ride start
        (0, -NPB),         # one chunk before the start
        (T_RIDE - NPB, +1000),   # far past the ride end
        (T_RIDE - NPB, +NPB),    # one chunk past the end
        (T_RIDE // 2, +T_RIDE),  # offset larger than the ride
        (0, 0),            # exact left edge, positional
    ],
)
def test_edge_offsets_clamp_without_crashing(chunk_lo, m):
    ride, pred = _ride(), _pred()
    out = ANCHORS(_stub(ride=ride, match_offset=m), pred, chunk_lo)
    assert out is not None
    for k in KEYS:
        assert torch.isfinite(out[k]).all(), f"{k} not finite at edge"
    # the clamp must produce a window that is a real slice of the ride, so
    # every anchor value must lie inside the ride's own stat range
    lo_std = float(_per_frame_STD(ride).min())
    hi_std = float(_per_frame_STD(ride).max())
    assert float(out["STD"].min()) >= lo_std - 1e-5
    assert float(out["STD"].max()) <= hi_std + 1e-5


def test_left_edge_snaps_to_the_first_in_bounds_chunk():
    """``lo`` clamps to 0 and the window is never shorter than npb."""
    ride, pred = _ride(), _pred(frames=NPB)
    far_left = ANCHORS(_stub(ride=ride, match_offset=-10 ** 6), pred, 0)
    # window snapped to ride[:, 0:NPB]
    win = ride[:, 0:NPB].to(torch.float32)
    assert torch.allclose(
        far_left["STD"][:, 0], _per_frame_STD(win).mean(1), atol=1e-6)
    assert torch.allclose(
        far_left["MEAN"][:, 0], _per_frame_MEAN(win).mean(1), atol=1e-6)


def test_right_edge_snaps_to_the_last_in_bounds_chunk():
    ride, pred = _ride(), _pred(frames=NPB)
    far_right = ANCHORS(_stub(ride=ride, match_offset=10 ** 6), pred, 0)
    win = ride[:, T_RIDE - NPB:T_RIDE].to(torch.float32)
    assert torch.allclose(
        far_right["STD"][:, 0], _per_frame_STD(win).mean(1), atol=1e-6)
    assert torch.allclose(
        far_right["M1"][:, 0], _per_frame_M1(win).mean(1), atol=1e-4)


# ---------------------------------------------------------------------------
# 4. shapes / dtype / detachment / values
# ---------------------------------------------------------------------------
def test_output_shapes_are_rank_uniform():
    ride = _ride()
    for frames in (NPB, 2 * NPB, 9, 12):
        out = ANCHORS(_stub(ride=ride, match_offset=2), _pred(frames), 15)
        for k in ("STD", "M2", "TV", "SOS"):
            assert out[k].shape == (B, frames), f"{k} {out[k].shape}"
        for k in ("M1", "MEAN"):
            assert out[k].shape == (B, frames, C), f"{k} {out[k].shape}"


def test_output_is_detached_and_key_set_is_exact():
    ride = _ride()
    pred = _pred().requires_grad_(True)
    out = ANCHORS(_stub(ride=ride, match_offset=1), pred, 15)
    assert set(out) == set(KEYS)
    for k in KEYS:
        assert not out[k].requires_grad, f"{k} carries a graph"


def test_anchor_values_match_the_window_mean():
    """Direct value check for chunk 0 with k=0 (window == the chunk)."""
    ride = _ride()
    pred = _pred(frames=NPB)
    lo, m = 12, 3
    out = ANCHORS(_stub(ride=ride, match_offset=m, k=0), pred, lo)
    win = ride[:, lo + m: lo + m + NPB].to(torch.float32)
    assert torch.allclose(out["STD"][:, 0], _per_frame_STD(win).mean(1),
                          atol=1e-6)
    assert torch.allclose(out["M2"][:, 0], _per_frame_M2(win).mean(1),
                          atol=1e-6)
    assert torch.allclose(out["TV"][:, 0], _per_frame_TV(win).mean(1),
                          atol=1e-6)
    assert torch.allclose(out["SOS"][:, 0], _per_frame_SOS(win).mean(1),
                          atol=1e-3)
    assert torch.allclose(out["M1"][:, 0], _per_frame_M1(win).mean(1),
                          atol=1e-4)
    assert torch.allclose(out["MEAN"][:, 0], _per_frame_MEAN(win).mean(1),
                          atol=1e-6)


def test_remainder_frames_repeat_the_last_chunk_anchor():
    ride = _ride()
    frames = 2 * NPB + 2                     # not a multiple of npb
    out = ANCHORS(_stub(ride=ride, match_offset=1), _pred(frames), 15)
    j = (frames // NPB) * NPB
    for k in KEYS:
        for f in range(j, frames):
            assert torch.equal(out[k][:, f], out[k][:, j - 1]), (
                f"{k} remainder frame {f} does not repeat the last chunk")


def test_each_chunk_gets_its_own_anchor():
    """Two chunks whose GT windows differ must not share an anchor value
    (catches a loop that writes chunk 0's anchor everywhere)."""
    ride = _ride()
    out = ANCHORS(_stub(ride=ride, match_offset=0, k=0), _pred(3 * NPB), 6)
    assert not torch.equal(out["STD"][:, 0], out["STD"][:, NPB])
    assert not torch.equal(out["MEAN"][:, 0], out["MEAN"][:, NPB])


# ---------------------------------------------------------------------------
# 5. bail-outs
# ---------------------------------------------------------------------------
def test_returns_none_without_a_ride_window():
    assert ANCHORS(_stub(ride=None), _pred(), 0) is None


def test_returns_none_without_a_prediction():
    assert ANCHORS(_stub(ride=_ride()), None, 0) is None


def test_returns_none_when_streaming_state_is_absent():
    s = SimpleNamespace(num_frame_per_block=NPB)
    assert ANCHORS(s, _pred(), 0) is None


def test_returns_none_for_wrong_rank_or_too_few_frames():
    ride = _ride()
    assert ANCHORS(_stub(ride=ride), torch.randn(B, 9, C, H), 0) is None
    assert ANCHORS(_stub(ride=ride), _pred(frames=NPB - 1), 0) is None


def test_returns_none_when_the_ride_is_shorter_than_a_chunk():
    short = _ride(T=NPB - 1)
    assert ANCHORS(_stub(ride=short), _pred(), 0) is None


# ---------------------------------------------------------------------------
def main():
    g = dict(globals())
    names = [n for n in g if n.startswith("test_")]
    names.sort(key=lambda n: g[n].__code__.co_firstlineno)
    n_run = 0
    for n in names:
        fn = g[n]
        marks = getattr(fn, "pytestmark", [])
        params = [
            m for m in marks if getattr(m, "name", "") == "parametrize"]
        if params:
            argnames = [a.strip() for a in params[0].args[0].split(",")]
            for vals in params[0].args[1]:
                vals = vals if isinstance(vals, tuple) else (vals,)
                fn(**dict(zip(argnames, vals)))
                n_run += 1
            print(f"  ok  {n}  x{len(params[0].args[1])}")
        else:
            fn()
            n_run += 1
            print(f"  ok  {n}")
    print(f"\nALL {n_run} TEST CASES PASSED")


if __name__ == "__main__":
    main()
