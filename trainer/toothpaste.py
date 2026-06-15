"""Pure (torch-free, import-light) helpers for the FT_v3 phase-2 rolling
toothpaste depth-curriculum + the same-ride-resampling setup floor.

Kept in their own module so they can be unit-tested without importing the
full trainer (which CUDA-inits at import). See
testing/test_toothpaste_and_defer_disc.py. The trainer's static methods
delegate here.
"""
from typing import List, Optional, Tuple


def toothpaste_grow(
    window: List[float],
    win: int,
    prev_avg: Optional[float],
    tol: float,
) -> Tuple[bool, Optional[float]]:
    """Depth-growth decision. ``window`` = the sliding list of recent CLEAN
    (non-gone) frontier MAEs at the current depth (already capped to
    ``win``). Grow when the window is full AND its mean is
    ``<= prev_avg * (1 + tol)`` — or when ``prev_avg`` is None (the first
    depth, which just establishes the baseline). Returns
    ``(grow, baseline)`` where ``baseline`` is the window mean to store as
    the new prev-depth average when growing (else ``prev_avg`` unchanged).
    """
    if len(window) < int(win):
        return False, prev_avg
    avg = sum(window) / float(len(window))
    if prev_avg is None or avg <= float(prev_avg) * (1.0 + float(tol)):
        return True, avg
    return False, prev_avg


def toothpaste_gone(
    frontier_mae: float,
    gone_base: Optional[float],
    gone_factor: float,
) -> bool:
    """Off-manifold ('gone') decision: True when a baseline exists and the
    frontier MAE exceeds ``gone_base * gone_factor``."""
    if gone_base is None:
        return False
    return float(frontier_mae) > float(gone_base) * float(gone_factor)


def setup_s_local_max(
    ride_len: int,
    cf: int,
    cap: int,
    slack: int,
    anchor: int,
    min_new: int,
    npb: int,
) -> int:
    """Largest motion-aware seed offset ``s`` such that the rolling
    ``roll_cap = min(cap, ride_len-cf-s) - slack`` still clears the
    ``anchor + min_new`` floor for EVERY s in [0, s_local_max] (the
    ``- npb`` covers the actual_cap npb-snap). Reserving the floor in ``s``
    keeps the roll_cap reject from splitting control flow across ranks under
    same-ride resampling (DDP-hang fix)."""
    floor_cap = ride_len - cf - slack - anchor - min_new - npb
    return max(0, min(ride_len // 2, ride_len - cf - cap, floor_cap))
