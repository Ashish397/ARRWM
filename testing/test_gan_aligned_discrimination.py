import importlib.util
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold


_PATH = Path(__file__).resolve().parents[1] / "analysis/gan_tuning/gan_aligned_discrimination.py"
_SPEC = importlib.util.spec_from_file_location("gan_aligned_discrimination", _PATH)
M = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = M
_SPEC.loader.exec_module(M)


def _rows():
    rng = np.random.default_rng(7)
    out = []
    # Six complete ride groups; the fake direction is shared across rides.
    for ride in range(6):
        for step in range(4):
            real = rng.normal(0, 0.3, (2, 5, 6)).astype(np.float32)
            fake = real.copy()
            # Coordinate-specific rather than a uniform per-tap shift: the
            # deployed per-example LayerNorm deliberately removes the latter.
            fake[..., 0] += 1.2
            fake[..., 3] += 0.7
            out.append(M.Row(real, fake, f"ride{ride}", step, ride, "x"))
    return out


def test_evidence_geometry_is_the_live_k_f_budget():
    x = np.arange(2 * 5 * 3).reshape(2, 5, 3)
    assert M.select_evidence(x, "k1f1").shape == (1, 3)
    assert M.select_evidence(x, "k1f2").shape == (2, 3)
    assert M.select_evidence(x, "k2f5").shape == (10, 3)
    assert np.array_equal(M.select_evidence(x, "k1f2"), x[0, [0, 4]])


def test_ride_held_out_linear_probe_finds_shared_direction():
    rows = _rows()
    groups = np.asarray([r.ride for r in rows])
    folds = list(GroupKFold(3).split(np.arange(len(rows)), groups=groups))
    got = M.linear_probe(rows, "k2f5", folds, 1)
    assert got["image_auc"]["min"] > 0.98
    assert got["row_mean_auc"]["min"] > 0.98


def test_online_rpgan_head_reports_fake_positive_auc():
    rows = _rows()
    groups = np.asarray([r.ride for r in rows])
    folds = list(GroupKFold(3).split(np.arange(len(rows)), groups=groups))
    got = M.online_probe(
        rows, [3, 3], "k2f5", folds,
        lr=1e-2, substeps=20, seed=3,
    )
    assert got["last_train_real_minus_fake"]["median"] > 0
    assert got["image_auc"]["median"] > 0.90
    assert got["row_mean_auc"]["median"] > 0.90
