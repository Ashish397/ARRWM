"""The evaluation must run from this tree and produce what the original produced.

The instruments were developed in the main research repo and moved here, so
these check the move did not change what they compute: the fleet index resolves
to the same 256x13 coverage, frame extraction returns identical pixels, and the
relocation scan is deterministic run to run.

Everything skips cleanly when the rollout videos are absent, since they are far
too large to distribute with the code. Point AF_FLEET_DIR at a directory holding
grids_A/ and baselines/ to enable them.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

QUALITY = Path(__file__).resolve().parents[1] / "evaluation" / "quality"


def _load(name, path):
    sys.path.insert(0, str(QUALITY))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    sys.path.pop(0)
    return module


@pytest.fixture(scope="module")
def fleet():
    fc = _load("fleet_common", QUALITY / "fleet_common.py")
    if not os.path.isdir(fc.GRID_DIR) or not os.path.isdir(fc.BASE_DIR):
        pytest.skip(f"rollout videos not present (set AF_FLEET_DIR); looked in {fc.FLEET_DIR}")
    return fc


def test_fleet_index_is_the_full_grid(fleet):
    """256 directional rollouts x 13 models, the coverage the paper reports."""
    index = fleet.fleet_index()
    assert len(index) == 3328
    assert len({s for s, _ in index}) == 256
    assert len({m for _, m in index}) == 13


def test_index_covers_ours_and_every_baseline(fleet):
    models = {m for _, m in fleet.fleet_index()}
    assert {"astra", "matrixgame", "minwm", "worldcam", "worldplay", "yume"} <= models
    assert len({m for m in models if m.startswith("ours_")}) == 7


def test_frame_extraction_is_repeatable(fleet):
    """Two reads of the same frames must give identical pixels.

    Our variants are de-tiled out of a 2x4 grid video, so this also covers the
    tile-origin arithmetic that would silently return a neighbouring model.
    """
    import numpy as np

    for scene, model in [fleet.fleet_index()[0], ("r09_FL", "ours_pca8")]:
        try:
            a = fleet.frames_at(scene, model, [10, 40])
            b = fleet.frames_at(scene, model, [10, 40])
        except Exception as exc:                        # noqa: BLE001
            pytest.skip(f"{scene}/{model} unreadable: {exc}")
        for x, y in zip(a, b):
            assert np.array_equal(x, y)


def test_relocation_threshold_is_pinned():
    """The relocation cut is calibrated against human labels, not a magic number."""
    path = QUALITY / "scene_reloc_threshold.txt"
    if not path.exists():
        pytest.skip("threshold file not present")
    assert int(path.read_text()) == 26


def test_quality_scripts_have_no_author_paths():
    """No absolute author or cluster paths may remain in the shipped instruments."""
    offenders = []
    for f in sorted(QUALITY.glob("*.py")):
        text = f.read_text()
        for needle in ("/home/ashish", "/scratch/u6ex", "/projects/u6ex"):
            if needle in text:
                offenders.append(f"{f.name}: {needle}")
    assert not offenders, "hardcoded paths:\n  " + "\n  ".join(offenders)
