"""The training split, as reported in the paper.

The reverse-motion scarcity is the paper's central empirical claim, so the split
composition is pinned here rather than left to prose.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "assets" / "train_windows.json"

# Paper, Evaluation/Setup: 2,577 rides, ~61K windows, 632 sustained reverse
# windows oversampled 6x.
EXPECTED = {
    "n_windows": 63792,
    "n_forward": 60000,
    "n_backward_distinct": 632,
    "n_backward_entries": 3792,
    "oversample": 6,
}


@pytest.fixture(scope="module")
def manifest():
    if not MANIFEST.exists():
        pytest.skip(f"manifest not present at {MANIFEST}")
    return json.loads(MANIFEST.read_text())


def test_split_composition_matches_paper(manifest):
    for key, want in EXPECTED.items():
        assert manifest[key] == want, f"{key}: {manifest[key]} != paper's {want}"


def test_backward_oversampling_is_consistent(manifest):
    assert (
        manifest["n_backward_distinct"] * manifest["oversample"]
        == manifest["n_backward_entries"]
    )
    assert manifest["n_forward"] + manifest["n_backward_entries"] == manifest["n_windows"]


def test_window_count_and_ride_count(manifest):
    windows = manifest["windows"]
    assert len(windows) == EXPECTED["n_windows"]
    rides = {Path(w["zarr_path"]).name for w in windows}
    assert len(rides) == 2577, f"paper reports 2,577 rides, manifest has {len(rides)}"


def test_reverse_motion_matches_the_reported_duration(manifest):
    """632 sustained-reverse windows = ~55 min, against ~88 h of training footage.

    A window is 21 latent frames; three latents span 12 pixel frames, so at 16 fps
    a window is 5.25 s. Both published durations reproduce from the manifest.

    On the "<1%" claim: the fraction depends on the denominator, and the paper's
    is the ~2,000 h corpus that was searched for reverse motion, giving 0.046%.
    Against the ~88 h training subset it is 1.04%, and by commanded throttle,
    chunks below -0.2 (a clear reverse command) are 0.93%. All are consistent
    with "less than 1%" except the window-count-over-training-subset reading.
    """
    windows = manifest["windows"]
    distinct = {(w["zarr_path"], w["start"]) for w in windows}
    reverse = {(w["zarr_path"], w["start"]) for w in windows if w["backward"]}

    window_seconds = 21 / 3 * 12 / 16
    reverse_minutes = len(reverse) * window_seconds / 60
    total_hours = len(distinct) * window_seconds / 3600

    assert 50 < reverse_minutes < 60, f"reverse footage {reverse_minutes:.1f} min, paper says ~55"
    assert 85 < total_hours < 92, f"training footage {total_hours:.1f} h, paper says ~88"
    assert len(reverse) * window_seconds / (2000 * 3600) < 0.01


def test_every_window_has_the_required_fields(manifest):
    for w in manifest["windows"][:1000]:
        assert set(w) >= {"zarr_path", "start", "n_latent_frames", "backward"}
        assert w["start"] >= 0
        assert w["start"] < w["n_latent_frames"]
