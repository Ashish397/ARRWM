import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


_PATH = (
    Path(__file__).resolve().parents[1]
    / "analysis/gan_tuning/surrogate_domain_balance_benchmark.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "surrogate_domain_balance_bench", _PATH,
)
M = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = M
_SPEC.loader.exec_module(M)


def _paired_bank(path: Path):
    rng = np.random.default_rng(7)
    latents = []
    gradients = []
    domains = []
    pair_ids = []
    rides = []
    roles = []
    origins = []
    steps = []
    # Two teacher versions, with two train and two test pairs at each.
    for step in (0, 5):
        for role in ("surrogate_train", "surrogate_test"):
            for row in range(2):
                base = rng.normal(size=(1, 16, 4, 4)).astype(np.float32)
                pair_id = f"step{step}:{role}:row{row}:crop0"
                for domain in ("real", "fake"):
                    latent = base + (0.15 if domain == "real" else -0.15)
                    gradient = np.tanh(latent)
                    if domain == "real":
                        gradient = -0.5 * gradient
                    latents.append(latent)
                    gradients.append(gradient.astype(np.float32))
                    domains.append(domain)
                    pair_ids.append(pair_id)
                    rides.append(f"{role}_ride{row}")
                    roles.append(role)
                    origins.append((0, 0))
                    steps.append(step)
    np.savez(
        path,
        latent=np.asarray(latents, dtype=np.float32),
        gradient_online=np.asarray(gradients, dtype=np.float32),
        domain=np.asarray(domains),
        pair_id=np.asarray(pair_ids),
        ride=np.asarray(rides),
        role=np.asarray(roles),
        origin_yx=np.asarray(origins, dtype=np.int16),
        step=np.asarray(steps, dtype=np.int32),
        teacher_value_online=np.zeros(len(latents), dtype=np.float32),
        teacher_grad_rms_online=np.ones(len(latents), dtype=np.float32),
        metadata=np.asarray(json.dumps({
            "source": "vgg", "rotation": 0,
            "head": "synthetic_head.pt", "head_lr": 1e-3,
        })),
    )


def _fit_args(bank: Path, output: Path):
    return argparse.Namespace(
        targets=bank, output=output, device="cpu", lr=1e-3,
        substeps_per_step=1, fake_compute_substeps=2,
        seeds=1, seed=13, eval_batch_size=4,
        width=8, blocks=1, head_init_std=1e-3,
        architecture="spatial", temporal_blocks=1, loss_mode="cosine",
    )


def test_capture_pair_preserves_paired_rows_and_selected_crop(tmp_path):
    path = tmp_path / "capture.npz"
    real_latent = np.zeros((2, 2, 3, 16, 4, 4), dtype=np.float16)
    fake_latent = np.ones_like(real_latent)
    real = np.zeros((2, 2, 5, 8, 8, 3), dtype=np.uint8)
    fake = np.ones_like(real)
    np.savez(
        path, real_latent=real_latent, fake_latent=fake_latent,
        real=real, fake=fake,
        metadata=np.asarray(json.dumps({"step": 0})),
    )
    meta, pair = M._capture_pair(path, 1)
    assert meta["step"] == 0
    assert pair["real"]["latent"].shape == (2, 3, 16, 4, 4)
    assert pair["fake"]["pixels"].shape == (2, 5, 8, 8, 3)
    assert np.all(pair["real"]["latent"] == 0)
    assert np.all(pair["fake"]["latent"] == 1)


def test_pair_validator_and_explicit_domain_weights(tmp_path):
    bank = tmp_path / "paired.npz"
    _paired_bank(bank)
    with np.load(bank, allow_pickle=False) as zf:
        arrays = {key: np.array(zf[key], copy=True) for key in zf.files}
    M._validate_paired_arrays(arrays)

    losses = {"real": torch.tensor(2.0), "fake": torch.tensor(8.0)}
    assert M._combine_domain_losses(
        losses, {"real": 1.0, "fake": 1.0},
    ).item() == pytest.approx(5.0)
    assert M._combine_domain_losses(
        losses, {"real": 0.0, "fake": 1.0},
    ).item() == pytest.approx(8.0)

    bad = dict(arrays)
    keep = np.arange(len(bad["latent"]) - 1)
    for key, value in list(bad.items()):
        if key != "metadata" and getattr(value, "ndim", 0) > 0:
            bad[key] = value[keep]
    with pytest.raises(RuntimeError, match="expected one real and one fake"):
        M._validate_paired_arrays(bad)


def test_three_arm_comparison_uses_matched_seeds_and_fake_test_only(tmp_path):
    bank = tmp_path / "paired.npz"
    output = tmp_path / "comparison.json"
    _paired_bank(bank)
    M._fit_comparison(_fit_args(bank, output))
    result = json.loads(output.read_text())

    assert set(result["arms"]) == {"fake24", "realfake24", "fake48"}
    assert result["decision_metrics_use_fake_test_only"] is True
    assert result["evaluation_domain"] == "fake"
    assert result["arms"]["fake24"]["domain_weights"] == {
        "real": 0.0, "fake": 1.0,
    }
    assert result["arms"]["realfake24"]["domain_weights"] == {
        "real": 1.0, "fake": 1.0,
    }
    assert result["arms"]["fake48"]["substeps_per_global_step"] == 2

    fingerprints = [
        result["arms"][name]["results"][0]["initial_fingerprint"]
        for name in ("fake24", "realfake24", "fake48")
    ]
    assert fingerprints[0] == fingerprints[1] == fingerprints[2]
    for arm in result["arms"].values():
        for item in arm["results"][0]["tracking"]:
            assert item["evaluation_domain"] == "fake"
            # There are exactly two fake test rows at each teacher version;
            # real test rows must not leak into the decision metric.
            assert item["post_current"]["cosine"]["n"] == 2

    decisions = result["paired_decisions"]["post_to_next"]
    assert decisions["realfake24_minus_fake24"]["delta"]["n"] == 1
    assert decisions["realfake24_minus_fake48"]["delta"]["n"] == 1
    assert (
        decisions["realfake24_minus_fake24"]["metric"]
        == "fake_test_per_sample_cosine_q1"
    )
    assert isinstance(
        result["paired_decisions"]["realfake_wins_this_rotation"], bool,
    )


def test_cross_rotation_summary_fails_mixed_route_closed(tmp_path):
    summary_script = (
        Path(__file__).resolve().parents[1]
        / "analysis/gan_tuning/summarize_surrogate_domain_balance.py"
    )
    inputs = []
    for rotation, wins in enumerate((True, False, True)):
        comparison = {
            "delta": {"q1": 0.01, "n": 9},
        }
        decisions = {
            "primary_metric": "fake_test_per_sample_cosine_q1",
            "realfake_wins_this_rotation": wins,
            "post_current": {
                "realfake24_minus_fake24": comparison,
                "realfake24_minus_fake48": comparison,
            },
            "post_to_next": {
                "realfake24_minus_fake24": comparison,
                "realfake24_minus_fake48": comparison,
            },
        }
        data = {
            "source": "vgg", "rotation": rotation,
            "teacher_head_lr": 1e-3,
            "optimizer": "adam_beta0_0_beta1_0.9", "lr": 1e-3,
            "architecture": "temporal_global", "loss_mode": "cosine",
            "width": 96, "blocks": 6, "temporal_blocks": 2,
            "head_init_std": 1e-3, "seeds": 3,
            "evaluation_domain": "fake", "paired_decisions": decisions,
        }
        path = tmp_path / f"r{rotation}.json"
        path.write_text(json.dumps(data)); inputs.append(path)
    output = tmp_path / "SUMMARY.md"
    proc = subprocess.run(
        [sys.executable, str(summary_script), "--inputs", *map(str, inputs),
         "--selected-candidate", "winner", "--output", str(output)],
        text=True, capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Selected target domain: **fakeonly**" in output.read_text()
