"""Collects the end-to-end training gate into the normal test run.

`training_smoke.py` is a CLI (`--save` / `--check`), so pytest's default
discovery never picked it up even though the validation notes advertised it as
an asserted gate. This wrapper runs it in `--check` mode when the machine can
actually train, and skips with a reason when it cannot, rather than silently
contributing nothing.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SMOKE = HERE / "training_smoke.py"
GOLDEN = HERE / "goldens" / "training_smoke.json"


def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def test_training_losses_match_the_recorded_reference():
    if not SMOKE.exists():
        pytest.skip("training_smoke.py not present")
    if not GOLDEN.exists():
        pytest.skip(f"no recorded reference at {GOLDEN}; run training_smoke.py --save")
    if not _has_gpu():
        pytest.skip("needs a GPU: the gate trains real steps")
    if not os.environ.get("ARRWM_RUN_TRAINING_SMOKE"):
        pytest.skip("set ARRWM_RUN_TRAINING_SMOKE=1 to run; the golden records real\n"
                    "losses and is machine-specific, so it only means something on the\n"
                    "hardware it was recorded on")
    cfg = HERE.parent / "configs" / "causal_lora_diffusion_teacher_v14e.yaml"
    proc = subprocess.run(
        [sys.executable, str(SMOKE), "--config", str(cfg),
         "--logdir", str(HERE.parent / "logs" / "smoke_gate"), "--check"],
        capture_output=True, text=True, cwd=HERE.parent)
    assert proc.returncode == 0, (
        f"training smoke gate failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")


def test_the_recorded_reference_is_well_formed():
    """Cheap, GPU-free: the golden must be usable even where the gate cannot run."""
    if not GOLDEN.exists():
        pytest.skip(f"no recorded reference at {GOLDEN}")
    ref = json.loads(GOLDEN.read_text())
    steps = ref["steps"] if isinstance(ref, dict) and "steps" in ref else ref
    assert steps, "reference records no steps"
    for s in steps:
        assert "loss" in s, f"reference entry missing 'loss': {s}"
        assert isinstance(s["loss"], (int, float))
