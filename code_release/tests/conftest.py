"""Shared fixtures and the golden-value harness.

Cleanup work is only safe if behaviour is pinned first. These tests record
outputs of the verbatim code as JSON goldens, then assert every later revision
reproduces them. Regenerate deliberately with ``UPDATE_GOLDENS=1 pytest``;
an unexplained regeneration is a silent behaviour change.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
GOLDEN_DIR = Path(__file__).parent / "goldens"
UPDATE = os.environ.get("UPDATE_GOLDENS") == "1"

# The PCA basis and squash scales are the paper's action space. Both are
# load-bearing for every reported number, so they are pinned exactly.
PCA_BASIS = REPO / "preprocessing" / "checkpoints" / "pca_basis.pt"


@pytest.fixture(scope="session")
def pca_basis():
    """(mean, components) of the egomotion action basis.

    Extracted from the original training checkpoint into a standalone artefact
    so nothing in the release depends on the retired ss_vae model; see
    docs/DECISIONS.md.
    """
    if not PCA_BASIS.exists():
        pytest.skip(f"PCA basis not present at {PCA_BASIS}")
    ck = torch.load(PCA_BASIS, map_location="cpu", weights_only=False)
    return (np.asarray(ck["pca_mean"], dtype=np.float64),
            np.asarray(ck["pca_comp"], dtype=np.float64))


@pytest.fixture(autouse=True)
def deterministic():
    """Seed every RNG so module-construction and forward passes are reproducible."""
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(False)  # 3D conv has no deterministic kernel
    yield


class Golden:
    """Compare against recorded values, or record them when UPDATE_GOLDENS=1."""

    def __init__(self, name: str):
        self.path = GOLDEN_DIR / f"{name}.json"
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.dirty = False

    def check(self, key: str, value, rtol: float = 1e-5, atol: float = 1e-7):
        arr = np.asarray(value, dtype=np.float64)
        if key not in self.data:
            if not UPDATE:
                pytest.fail(
                    f"no golden for '{key}' in {self.path.name}; "
                    f"run UPDATE_GOLDENS=1 pytest to record"
                )
            self.data[key] = arr.tolist()
            self.dirty = True
            return
        ref = np.asarray(self.data[key], dtype=np.float64)
        assert arr.shape == ref.shape, f"{key}: shape {arr.shape} != golden {ref.shape}"
        np.testing.assert_allclose(arr, ref, rtol=rtol, atol=atol, err_msg=f"{key} drifted")

    def save(self):
        if self.dirty and UPDATE:
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=1, sort_keys=True) + "\n")


@pytest.fixture
def golden(request):
    g = Golden(request.node.module.__name__.split(".")[-1])
    yield g
    g.save()
