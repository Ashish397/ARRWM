"""Preprocessing must produce motion vectors the shipped action basis can read.

The tracker's grid size and the basis dimensionality are coupled: the basis is
fitted on 200 = 100 points x (dx, dy). A grid change silently produces motion
files the encoder cannot project, so it is pinned here rather than left to a
module-level constant nobody re-reads.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Paper, Sec. Data Derived Action Space: a fixed 10x10 grid, j = 1..100.
EXPECTED_GRID = 10
EXPECTED_POINTS = EXPECTED_GRID**2
EXPECTED_FLAT_DIM = EXPECTED_POINTS * 2


def test_tracker_grid_matches_the_basis():
    src = (REPO / "preprocessing" / "pre_encode_motion.py").read_text()
    m = re.search(r"^grid_size\s*=\s*(\d+)", src, re.M)
    assert m, "grid_size not found at module scope in pre_encode_motion.py"
    assert int(m.group(1)) == EXPECTED_GRID, (
        f"tracker emits a {m.group(1)}x{m.group(1)} grid but the shipped basis "
        f"expects {EXPECTED_GRID}x{EXPECTED_GRID}; motion files would not project"
    )


def test_basis_dimensionality_matches_the_grid(pca_basis):
    mean, comp = pca_basis
    assert mean.shape == (EXPECTED_FLAT_DIM,)
    assert comp.shape[1] == EXPECTED_FLAT_DIM


def test_encoder_consumes_the_documented_motion_shape(pca_basis):
    """(n_chunks, 100, 3) in -> (n_chunks, 8) out."""
    from utils.zarr_dataset import _encode_motion_pca_raw

    mean, comp = pca_basis
    motion = np.zeros((5, EXPECTED_POINTS, 3), dtype=np.float32)
    assert _encode_motion_pca_raw(motion, mean, comp, n_out=8).shape == (5, 8)


def test_encoder_rejects_a_different_grid(pca_basis):
    """A 20x20 grid must fail loudly rather than project wrongly."""
    from utils.zarr_dataset import _encode_motion_pca_raw

    mean, comp = pca_basis
    wrong = np.zeros((5, 400, 3), dtype=np.float32)
    with pytest.raises(Exception):
        _encode_motion_pca_raw(wrong, mean, comp, n_out=8)
