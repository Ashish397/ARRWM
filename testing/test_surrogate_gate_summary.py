import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / "analysis/gan_tuning/summarize_surrogate_gate_screen.py"
ROTATIONS = ROOT / "analysis/gan_tuning/summarize_surrogate_gate_rotations.py"
SPEC = importlib.util.spec_from_file_location("surrogate_gate_summary", SCREEN)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


def _field(q1, median=0.70, global_cosine=0.99):
    return {
        "cosine": {"q1": q1, "median": median, "n": 27},
        "global_cosine": global_cosine,
    }


def _candidate(
    path, q1_steps, next_q1, *, rotation=0, substeps=12, seeds=3,
    condition="latent",
):
    if not isinstance(next_q1, list):
        next_q1 = [next_q1] * (len(q1_steps) - 1)
    results = []
    for seed_index in range(seeds):
        tracking = []
        seed_offset = seed_index * 0.005
        for index, q1 in enumerate(q1_steps):
            tracking.append({
                "global_step": 100 + index * 10,
                "fit_substeps": substeps,
                "post_current": _field(q1 + seed_offset),
                "post_to_next": (
                    _field(next_q1[index] + seed_offset)
                    if index + 1 < len(q1_steps) else None
                ),
            })
        results.append({"seed": 1000 + seed_index, "tracking": tracking})
    path.write_text(json.dumps({
        "source": "vgg", "rotation": rotation, "condition": condition,
        "teacher_mode": "online", "lr": 1e-3,
        "optimizer": "adam_beta0_0_beta1_0.9", "architecture": "spatial",
        "loss_mode": "mse", "temporal_blocks": 2, "head_init_std": 1e-3,
        "width": 96, "blocks": 6,
        "train_samples": len(q1_steps) * 27,
        "test_samples": len(q1_steps) * 27,
        "substeps_per_global_step": substeps, "results": results,
    }))


def _run_rotations(tmp_path, specs):
    inputs = []
    for directory_index, embedded_rotation in enumerate(specs):
        root = tmp_path / f"r{directory_index}"
        root.mkdir()
        _candidate(
            root / "spatial_mse_lr2e4.json", [0.30, 0.40, 0.45, 0.46],
            [0.30, 0.40, 0.45], rotation=embedded_rotation,
        )
        _candidate(
            root / "winner.json", [0.51, 0.54, 0.55, 0.56],
            [0.52, 0.53, 0.54], rotation=embedded_rotation,
        )
        inputs.append(root)
    output = tmp_path / "summary.md"
    proc = subprocess.run(
        [sys.executable, str(ROTATIONS), "--inputs", *map(str, inputs),
         "--output", str(output), "--expected-count", "2"],
        cwd=ROOT, text=True, capture_output=True,
    )
    return proc, output


def test_gate_uses_worst_seed_q1_and_cumulative_optimizer_updates(tmp_path):
    path = tmp_path / "candidate.json"
    _candidate(path, [0.49, 0.51, 0.52], [0.52, 0.53], substeps=12)
    row = M.load_candidate(path)
    assert row["first_gate"] == 24
    assert row["final_q1"] == 0.52
    assert row["final_global"] == 0.99


def test_cross_rotation_authorizes_only_robust_baseline_win(tmp_path):
    proc, output = _run_rotations(tmp_path, [0, 1, 2])
    assert proc.returncode == 0, proc.stderr
    report = output.read_text()
    assert "| winner | YES | yes | yes |" in report
    assert "| spatial_mse_lr2e4 | NO | no | no |" in report


def test_duplicate_embedded_rotation_fails_closed(tmp_path):
    proc, _ = _run_rotations(tmp_path, [0, 0, 2])
    assert proc.returncode != 0
    assert "distinct embedded rotations" in proc.stderr


def test_missing_seed_fails_closed(tmp_path):
    path = tmp_path / "candidate.json"
    _candidate(path, [0.51, 0.52, 0.53], [0.52, 0.53], seeds=2)
    with pytest.raises(ValueError, match="expected 3 seeds"):
        M.load_candidate(path)


def test_rgb_condition_requires_explicit_provenance_expectation(tmp_path):
    path = tmp_path / "candidate.json"
    _candidate(
        path, [0.51, 0.52, 0.53], [0.52, 0.53], condition="rgbmaxmin",
    )
    with pytest.raises(ValueError, match="expected VGG latent"):
        M.load_candidate(path)
    row = M.load_candidate(path, expected_condition="rgbmaxmin")
    assert row["provenance"]["condition"] == "rgbmaxmin"


def test_teacher_evidence_requires_explicit_provenance_expectation(tmp_path):
    path = tmp_path / "candidate.json"
    _candidate(
        path, [0.51, 0.52, 0.53], [0.52, 0.53],
        condition="vggstats_headgrad",
    )
    with pytest.raises(ValueError, match="expected VGG latent"):
        M.load_candidate(path)
    row = M.load_candidate(path, expected_condition="vggstats_headgrad")
    assert row["provenance"]["condition"] == "vggstats_headgrad"


def test_reordered_tracking_step_fails_closed(tmp_path):
    path = tmp_path / "candidate.json"
    _candidate(path, [0.51, 0.52, 0.53], [0.52, 0.53])
    data = json.loads(path.read_text())
    data["results"][2]["tracking"][1]["global_step"] = 99
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="reordered/duplicated"):
        M.load_candidate(path)


def test_nan_decision_metric_fails_closed(tmp_path):
    path = tmp_path / "candidate.json"
    _candidate(path, [0.51, 0.52, 0.53], [0.52, 0.53])
    data = json.loads(path.read_text())
    data["results"][1]["tracking"][0]["post_current"]["cosine"]["q1"] = float("nan")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="non-finite"):
        M.load_candidate(path)


def test_truncated_per_step_sample_count_fails_closed(tmp_path):
    path = tmp_path / "candidate.json"
    _candidate(path, [0.51, 0.52, 0.53], [0.52, 0.53])
    data = json.loads(path.read_text())
    data["results"][1]["tracking"][0]["post_current"]["cosine"]["n"] = 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="per-sample count"):
        M.load_candidate(path)


def test_transient_current_crossing_is_not_sustained(tmp_path):
    path = tmp_path / "candidate.json"
    _candidate(path, [0.51, 0.52, 0.49], [0.53, 0.53])
    row = M.load_candidate(path)
    assert row["first_gate"] == 12
    assert not row["sustained_current"]


def test_poor_early_next_tracking_cannot_hide_behind_final_transition(tmp_path):
    path = tmp_path / "candidate.json"
    _candidate(path, [0.51, 0.53, 0.54, 0.55], [0.10, 0.53, 0.55])
    row = M.load_candidate(path)
    assert row["next_final_q1"] == 0.55
    assert row["next_postgate_q1"] < 0.50


def test_two_transition_q1_cannot_hide_one_below_gate(tmp_path):
    path = tmp_path / "candidate.json"
    _candidate(path, [0.51, 0.53, 0.55], [0.49, 0.55])
    row = M.load_candidate(path)
    assert row["next_postgate_q1"] > 0.50
    assert row["next_postgate_min"] == 0.49
