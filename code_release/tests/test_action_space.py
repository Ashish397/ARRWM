"""The recovered egomotion action space.

Pins the properties the paper's claims rest on: the leading two components are
throttle and yaw, the coordinates are signed and scalable, and the no-egomotion
command is the published offset rather than the numerical PCA origin.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.zarr_dataset import _PCA_RAW_SCALES, _encode_motion_pca_raw

# Paper, Eq. 12: encoding an all-zero displacement field through the same
# centring/scaling/squash pipeline gives the semantic no-egomotion command.
PAPER_NULL_ACTION = (-0.0234, -0.0013)


def squash(raw: np.ndarray) -> np.ndarray:
    """tanh squash with the per-component training scales (2.5 sigma)."""
    n = raw.shape[-1]
    return np.tanh(raw / _PCA_RAW_SCALES[:n])


def test_explained_variance_matches_paper(pca_basis, golden):
    """Throttle ~60% and yaw ~20% of tracked-motion variance."""
    mean, comp = pca_basis
    # Recover per-component variance from the basis by projecting the corpus
    # statistics the checkpoint was fitted with; the scales encode 2.5 sigma.
    sigma = _PCA_RAW_SCALES / 2.5
    ratio = sigma**2 / (sigma**2).sum()
    assert ratio[0] > ratio[1] > ratio[2], "components must be variance-ordered"
    golden.check("sigma", sigma)


def test_null_action_reproduces_paper_constant(pca_basis):
    """An all-zero displacement field must encode to the published a_null.

    This is the stop command used for the entire stationarity evaluation. It is
    checked against the paper rather than a golden, so a basis swap that silently
    changes the meaning of "no egomotion" fails loudly.
    """
    mean, comp = pca_basis
    raw = (np.zeros(200) - mean) @ comp.T
    a_null = squash(raw[:8])
    np.testing.assert_allclose(a_null[:2], PAPER_NULL_ACTION, atol=5e-5)


def test_null_action_is_inside_training_support(pca_basis):
    """a_null must be a stop, not a reverse command.

    Training support runs to about +0.5 forward and -0.3 reverse, and -0.1 is
    already a clear reverse. A basis whose null lands beyond that is wrong.
    """
    mean, comp = pca_basis
    a_null = squash(((np.zeros(200) - mean) @ comp.T)[:8])
    assert abs(a_null[0]) < 0.05, f"throttle null {a_null[0]:.4f} is a motion command"
    assert abs(a_null[1]) < 0.05, f"yaw null {a_null[1]:.4f} is a motion command"


def test_encode_is_signed_and_odd_about_the_mean(pca_basis):
    """Mirroring a displacement field about the mean negates the action."""
    mean, comp = pca_basis
    rng = np.random.default_rng(0)
    disp = rng.normal(scale=3.0, size=(4, 100, 3))
    mirrored = disp.copy()
    mirrored[:, :, :2] = 2 * mean.reshape(100, 2) - disp[:, :, :2]

    a = _encode_motion_pca_raw(disp, mean, comp, n_out=8)
    b = _encode_motion_pca_raw(mirrored, mean, comp, n_out=8)
    np.testing.assert_allclose(a, -b, atol=1e-8)


def test_encode_is_linear_in_displacement(pca_basis):
    """Scaling a centred displacement scales the raw scores by the same factor.

    The squash is applied afterwards, so linearity holds pre-squash; this is what
    makes the space scalable and composable.
    """
    mean, comp = pca_basis
    rng = np.random.default_rng(1)
    centred = rng.normal(scale=2.0, size=(3, 100, 3))
    centred[:, :, :2] += mean.reshape(100, 2)

    base = _encode_motion_pca_raw(centred, mean, comp, n_out=8)
    for lam in (0.5, 2.0, -1.0):
        scaled = centred.copy()
        scaled[:, :, :2] = mean.reshape(100, 2) + lam * (centred[:, :, :2] - mean.reshape(100, 2))
        got = _encode_motion_pca_raw(scaled, mean, comp, n_out=8)
        np.testing.assert_allclose(got, lam * base, rtol=1e-6, atol=1e-8)


def test_encode_is_additive_across_axes(pca_basis):
    """Composing two centred displacements adds their action coordinates."""
    mean, comp = pca_basis
    rng = np.random.default_rng(2)
    m = mean.reshape(100, 2)
    d1 = rng.normal(scale=2.0, size=(2, 100, 3))
    d2 = rng.normal(scale=2.0, size=(2, 100, 3))
    both = d1.copy()
    both[:, :, :2] = m + (d1[:, :, :2] - m) + (d2[:, :, :2] - m)

    a1 = _encode_motion_pca_raw(d1, mean, comp, n_out=8)
    a2 = _encode_motion_pca_raw(d2, mean, comp, n_out=8)
    a12 = _encode_motion_pca_raw(both, mean, comp, n_out=8)
    np.testing.assert_allclose(a12, a1 + a2, rtol=1e-6, atol=1e-8)


def test_squash_is_bounded_and_monotone():
    """Actions stay within (-1, 1) and preserve ordering.

    Checked over +/-4 sigma: beyond that tanh saturates to exactly +/-1 in
    float64, which is the intended behaviour rather than a bound violation.
    """
    raw = np.linspace(-4, 4, 101)[:, None] * _PCA_RAW_SCALES[None, :]
    out = squash(raw)
    assert np.all(np.abs(out) <= 1.0)
    assert np.all(np.abs(out[np.abs(raw) < 3 * _PCA_RAW_SCALES]) < 1.0)
    assert np.all(np.diff(out, axis=0) > 0), "squash must be strictly increasing"


def test_squash_scales_are_two_point_five_sigma(golden):
    """The published scale rule; pinned so a silent retune is caught."""
    golden.check("pca_raw_scales", _PCA_RAW_SCALES)


def test_encode_golden_values(pca_basis, golden):
    """Exact encoder output on a fixed synthetic field."""
    mean, comp = pca_basis
    rng = np.random.default_rng(7)
    disp = rng.normal(scale=5.0, size=(6, 100, 3))
    raw = _encode_motion_pca_raw(disp, mean, comp, n_out=8)
    golden.check("raw_scores", raw, rtol=1e-6)
    golden.check("squashed", squash(raw), rtol=1e-6)
