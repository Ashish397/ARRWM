"""Regression test for matched-clean-x offset selection
(ActionForcingDMD.compute_clean_match_offset + _build_42f_scoring_inputs).

The feature: instead of placing the 42f clean half at the fixed positional
offset, pick the single window-level GT offset ``m`` (in +-max_drift_frames)
that best aligns the SUPERVISED band (the DMD chunks) with GT, then source
clean_x / gt_target / gt_ctx / actions from ``+m`` (student content unshifted),
back-shifted ``npb`` so RoPE stays pinned at v14's ``+npb``. Gate + target both
follow the same ``m``. Fixes time-drift (actions too weak/strong) without OOD.

The real method can't be imported on a CPU/login node (model import pulls in
wan.modules.t5's CUDA-at-import), so the runnable tests below mirror the
method's exact offset-search + bounds + back-shift math and lock the expected
behavior; a GPU-guarded test exercises the real method when available.

Run: python -m pytest testing/test_clean_match_offset.py -q
"""
import torch


# ---- Mirror of compute_clean_match_offset's core (geometry + search) --------
def _match_offset(ride_lat, chunk, cf, noisy_start_sdn, npb=3, N=21, ns=3,
                  gt_after_chunks=2, cap=40, overlap=0, new_frames=0,
                  rolling=False, min_improve=0.0, forward=False):
    chunk_lo_raw = cf + noisy_start_sdn
    if rolling:
        sup_frames = new_frames
        gt_after_frames = npb
        n_ctx = N - sup_frames - gt_after_frames
        band_lo = chunk_lo_raw + overlap
        band_off = overlap
    else:
        num_sup = 1 if ns == 1 else ns - 1
        sup_frames = num_sup * npb
        gt_after_frames = (
            gt_after_chunks * npb if gt_after_chunks > 0 else ns * npb - sup_frames
        )
        n_ctx = N - sup_frames - gt_after_frames
        band_lo = chunk_lo_raw
        band_off = 0
    noisy_lo = band_lo - n_ctx
    noisy_hi = band_lo + sup_frames + gt_after_frames
    L = ride_lat.shape[1]
    clen = chunk.shape[1]
    clean_base = npb if forward else -npb
    lo_base = min(noisy_lo + clean_base, noisy_lo, chunk_lo_raw)
    hi_base = max(noisy_hi, noisy_lo + clean_base + N, chunk_lo_raw + clen)
    m_lo = max(-cap, -lo_base)
    m_hi = min(cap, L - hi_base)
    m = 0
    if m_hi >= m_lo and (band_off + sup_frames) <= clen:
        stu = chunk[:, band_off:band_off + sup_frames].float()
        best_m, best_d, d0 = 0, None, None
        for c in range(m_lo, m_hi + 1):
            g = ride_lat[:, band_lo + c:band_lo + c + sup_frames].float()
            d = float((stu - g).abs().mean().item())
            if c == 0:
                d0 = d
            if best_d is None or d < best_d:
                best_m, best_d = c, d
        if d0 is not None and best_d is not None and best_d > d0 * (1.0 - min_improve):
            best_m = 0
        m = best_m
    clean_lo = noisy_lo + clean_base + m     # back-shift (-npb) or forward (+npb)
    return m, clean_lo, noisy_lo, (m_lo, m_hi)


def _ride(L, C=4, H=2, W=2):
    # frame t holds constant value t -> L2(frame a, b) = |a-b| * C*H*W
    return torch.arange(L, dtype=torch.float32).view(
        1, L, 1, 1, 1).expand(1, L, C, H, W).clone()


def test_recovers_positive_shift():
    L = 200
    r = _ride(L)
    cf, nss = 18, 42          # chunk_lo = 60
    clen = 12
    chunk = r[:, 60 + 5:60 + 5 + clen].clone()   # student looks like GT@+5
    m, clean_lo, noisy_lo, _ = _match_offset(r, chunk, cf, nss)
    assert m == 5
    assert clean_lo == noisy_lo - 3 + 5          # back-shift + matched offset


def test_recovers_negative_shift():
    r = _ride(200)
    chunk = r[:, 60 - 4:60 - 4 + 12].clone()
    m, _, _, _ = _match_offset(r, chunk, 18, 42)
    assert m == -4


def test_cap_clamps():
    r = _ride(200)
    chunk = r[:, 60 + 50:60 + 50 + 12].clone()   # planted +50, cap 40
    m, _, _, (_, m_hi) = _match_offset(r, chunk, 18, 42, cap=40)
    assert m == 40 and m_hi == 40                # clamped to the cap


def test_bounds_clamp_near_end_of_ride():
    r = _ride(80)
    chunk = r[:, 60:60 + 12].clone()             # GT@0
    m, _, _, (_, m_hi) = _match_offset(r, chunk, 18, 42, cap=40)
    assert m == 0 and m_hi == 8                  # m_hi = L - max(72,69,72) = 8


def test_zero_offset_is_v14_backshift():
    # m=0 (disabled-equivalent) leaves clean_lo at the pure -npb back-shift.
    r = _ride(200)
    chunk = r[:, 60:60 + 12].clone()
    m, clean_lo, noisy_lo, _ = _match_offset(r, chunk, 18, 42)
    assert m == 0 and clean_lo == noisy_lo - 3


def test_forward_shift_geometry():
    # Forward variant: same matched offset, but clean half placed +npb AHEAD
    # (clean_lo = noisy_lo + npb + m) instead of -npb behind.
    r = _ride(200)
    chunk = r[:, 60 + 5:60 + 5 + 12].clone()      # student looks like GT@+5
    m_b, clean_lo_b, noisy_lo, _ = _match_offset(r, chunk, 18, 42, forward=False)
    m_f, clean_lo_f, _, _ = _match_offset(r, chunk, 18, 42, forward=True)
    assert m_b == 5 and m_f == 5                   # match offset unchanged
    assert clean_lo_b == noisy_lo - 3 + 5          # back-shift
    assert clean_lo_f == noisy_lo + 3 + 5          # forward = +npb
    assert clean_lo_f - clean_lo_b == 6            # exactly 2*npb apart


def test_rolling_recovers_shift():
    # Rolling: supervised band = new frames at chunk[:, overlap:overlap+nf];
    # plant the new frames as GT@(band_lo+6). band_lo = chunk_lo_raw + overlap.
    r = _ride(200)
    cf, nss = 18, 42                      # chunk_lo_raw = 60
    overlap, nf = 6, 6                    # band_lo = 66, sup=6, n_ctx=21-6-3=12
    clen = overlap + nf                   # raw chunk = [overlap ctx | new]
    chunk = torch.empty(1, clen, 4, 2, 2)
    chunk[:, :overlap] = r[:, 60:66]                 # ctx (unmatched here)
    chunk[:, overlap:] = r[:, 66 + 6:66 + 6 + nf]    # new frames = GT@+6
    m, _, _, _ = _match_offset(
        r, chunk, cf, nss, overlap=overlap, new_frames=nf, rolling=True)
    assert m == 6, (m,)


def test_safeguard_falls_back_when_no_real_improvement():
    # A student that matches positional GT about as well as any other window
    # (no clean time-shift) -> with min_improve>0, fall back to m=0.
    r = _ride(200)
    # student = positional GT + uniform constant offset (equally bad at every m)
    chunk = r[:, 60:60 + 12].clone() + 0.5
    m, _, _, _ = _match_offset(r, chunk, 18, 42, min_improve=0.10)
    assert m == 0, (m,)
    # With a genuine +5 shift, the safeguard still accepts it (big improvement).
    chunk2 = r[:, 60 + 5:60 + 5 + 12].clone()
    m2, _, _, _ = _match_offset(r, chunk2, 18, 42, min_improve=0.10)
    assert m2 == 5, (m2,)


# ---- GPU-guarded test of the REAL method (runs on a GPU node / CI) ----------
try:
    from model.dmd_action_forcing import ActionForcingDMD  # noqa: F401
    _REAL_OK = True
except Exception:
    _REAL_OK = False

import pytest


@pytest.mark.skipif(not _REAL_OK, reason="model import needs a GPU (t5 CUDA-at-import)")
def test_real_method_disabled_returns_zero_none():
    m = ActionForcingDMD.__new__(ActionForcingDMD)
    m.dmd_42f_clean_match_enabled = False
    off, mae = ActionForcingDMD.compute_clean_match_offset(
        m, torch.zeros(1, 12, 4, 2, 2), {"current_length": 30, "new_frames": 6, "overlap": 0})
    assert off == 0 and mae is None


if __name__ == "__main__":
    test_recovers_positive_shift()
    test_recovers_negative_shift()
    test_cap_clamps()
    test_bounds_clamp_near_end_of_ride()
    test_zero_offset_is_v14_backshift()
    test_forward_shift_geometry()
    test_rolling_recovers_shift()
    test_safeguard_falls_back_when_no_real_improvement()
    print("matched-clean offset tests passed")
