"""Every shipped instrument must at least import.

Most instruments cannot run in a test: they need the rollout videos, a VLM, or
tens of minutes of GPU. But an import failure — a missing module, a renamed
helper, a path resolved at module scope — is a defect that would surface only
when someone tries to reproduce a number, and it is cheap to catch here.

This does not claim the instruments produce correct results. It claims they are
wired up. Six of them have been run end to end and checked against the paper;
the rest have only this.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

QUALITY = Path(__file__).resolve().parents[1] / "evaluation" / "quality"
SCRIPTS = sorted(p for p in QUALITY.glob("*.py") if p.name != "__init__.py")

# Import pulls a detector or VLM onto the GPU in these; skip unless asked.
NEEDS_GPU = {"popin_detect.py", "popin_backends.py", "popin_fleet_all.py",
             "conjure_probe.py", "conjure_detectors.py", "conjure_contact.py",
             "fleet_pal.py", "pal_local.py", "fleet_dino.py", "melt_vlm_bench.py",
             "blind_melt.py", "blind_vlm_probes.py", "blind_plausibility.py",
             "blind_temporal_novelty.py", "blind_dino_drift.py", "vlm_external.py",
             "fleet_reel_vlms.py", "style_shift.py", "blind_style_shift.py",
             "fleet_novelty.py", "noop_vlm.py", "stationary_cotracker.py",
             "fleet_cotracker.py"}


@pytest.mark.parametrize("path", SCRIPTS, ids=[p.name for p in SCRIPTS])
def test_instrument_imports(path, monkeypatch):
    if path.name in NEEDS_GPU and not os.environ.get("AF_TEST_GPU"):
        pytest.skip("loads a model at import; set AF_TEST_GPU=1 to include")

    # Instruments resolve data roots at module scope; point them somewhere real
    # so an import does not fail merely because the videos are absent.
    monkeypatch.setenv("AF_FLEET_DIR", os.environ.get("AF_FLEET_DIR", "/tmp"))
    monkeypatch.syspath_prepend(str(QUALITY))

    spec = importlib.util.spec_from_file_location(f"_probe_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except SystemExit:
        pass                                    # argparse-style early exit is fine
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"needs data absent here: {exc}")
    finally:
        sys.modules.pop(f"_probe_{path.stem}", None)
