import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


_PATH = (
    Path(__file__).resolve().parents[1]
    / "analysis/gan_tuning/surrogate_feature_field_benchmark.py"
)
_SPEC = importlib.util.spec_from_file_location("surrogate_feature_field_bench", _PATH)
M = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = M
_SPEC.loader.exec_module(M)


def _head_split():
    split = {
        "rank_roles": {
            "teacher": [0, 1],
            "surrogate_train": [2, 3, 4],
            "surrogate_test": [5, 6, 7],
        },
    }
    for role, ranks in split["rank_roles"].items():
        split[role] = [
            f"rank{rank}_era{era}" for rank in ranks for era in range(2)
        ]
    return split


def test_rank_stratified_roles_are_ride_disjoint_and_rotate():
    rows = []
    for rank in range(8):
        for era in range(3):
            rows.append(argparse.Namespace(rank=rank, ride=f"rank{rank}_era{era}"))
    seen = []
    for rotation in range(3):
        split = M._split_rows(rows, rotation)
        sets = [
            set(split[x])
            for x in ("teacher", "surrogate_train", "surrogate_test")
        ]
        assert not sets[0] & sets[1]
        assert not sets[0] & sets[2]
        assert not sets[1] & sets[2]
        seen.append(tuple(split["rank_roles"]["teacher"]))
    assert len(set(seen)) == 3


def test_target_extraction_split_default_preserves_head_split():
    original = _head_split()
    before = json.loads(json.dumps(original))
    effective, policy = M._effective_target_extraction_split(original, False)
    assert original == before
    assert effective == before
    assert policy["include_teacher_rides_in_train"] is False
    assert policy["head_calibration_rank_counts"] == {
        "teacher": 2, "surrogate_train": 3, "surrogate_test": 3,
    }
    assert policy["effective_rank_counts"] == {
        "teacher": 2, "surrogate_train": 3, "surrogate_test": 3,
    }


def test_target_extraction_split_adds_two_teacher_ranks_only_to_train():
    original = _head_split()
    effective, policy = M._effective_target_extraction_split(original, True)
    assert effective["teacher"] == []
    assert effective["rank_roles"]["teacher"] == []
    assert set(effective["surrogate_train"]) == (
        set(original["teacher"]) | set(original["surrogate_train"])
    )
    assert effective["rank_roles"]["surrogate_train"] == [0, 1, 2, 3, 4]
    assert effective["surrogate_test"] == original["surrogate_test"]
    assert (
        effective["rank_roles"]["surrogate_test"]
        == original["rank_roles"]["surrogate_test"]
    )
    assert not set(effective["surrogate_train"]) & set(
        effective["surrogate_test"]
    )
    assert policy["effective_rank_counts"] == {
        "teacher": 0, "surrogate_train": 5, "surrogate_test": 3,
    }
    assert policy["surrogate_test_unchanged"] is True
    assert policy["effective_split_disjoint"] is True


def test_target_extraction_split_rejects_overlap_and_wrong_rank_counts():
    overlapped = _head_split()
    overlapped["surrogate_test"][0] = overlapped["teacher"][0]
    with pytest.raises(RuntimeError, match="ride overlap"):
        M._effective_target_extraction_split(overlapped, True)

    wrong_count = _head_split()
    wrong_count["rank_roles"]["teacher"].pop()
    with pytest.raises(RuntimeError, match="rank count"):
        M._effective_target_extraction_split(wrong_count, True)


def test_teacher_ride_scaling_flag_is_default_off(monkeypatch, tmp_path):
    base = [
        "surrogate_feature_field_benchmark.py", "extract-targets",
        "--capture-dir", str(tmp_path), "--head", str(tmp_path / "head.pt"),
        "--source", "vgg", "--rotation", "0",
        "--output", str(tmp_path / "targets.npz"),
    ]
    monkeypatch.setattr(sys, "argv", base)
    assert M.parse_args().include_teacher_rides_in_train is False
    monkeypatch.setattr(
        sys, "argv", base + ["--include-teacher-rides-in-train"],
    )
    assert M.parse_args().include_teacher_rides_in_train is True


def _expanded_bank_roles_rides_metadata():
    head_split = {
        "rank_roles": {
            "teacher": [0, 1],
            "surrogate_train": [2, 3, 4],
            "surrogate_test": [5, 6, 7],
        },
        "teacher": ["teacher0", "teacher1"],
        "surrogate_train": ["strict0", "strict1", "strict2"],
        "surrogate_test": ["test0", "test1", "test2"],
    }
    effective, policy = M._effective_target_extraction_split(head_split, True)
    train_rides = effective["surrogate_train"]
    test_rides = effective["surrogate_test"]
    rides = np.asarray([
        ride for ride in train_rides for _row in range(3)
    ] + [
        ride for ride in test_rides for _row in range(3)
    ])
    roles = np.asarray(
        ["surrogate_train"] * (3 * len(train_rides))
        + ["surrogate_test"] * (3 * len(test_rides))
    )
    metadata = {
        "head_calibration_split": head_split,
        "split": effective,
        "extraction_role_policy": policy,
    }
    return roles, rides, metadata


def test_expanded_bank_supports_bank_and_strict_train_masks():
    roles, rides, metadata = _expanded_bank_roles_rides_metadata()
    expanded_ids, expanded = M._select_training_ride_ids(
        roles, rides, metadata, "bank",
    )
    strict_ids, strict = M._select_training_ride_ids(
        roles, rides, metadata, "head-split-strict",
    )
    test_ids = np.flatnonzero(roles == "surrogate_test")
    assert len(expanded_ids) == 15
    assert len(strict_ids) == 9
    assert expanded["selected_train_ride_count"] == 5
    assert strict["selected_train_ride_count"] == 3
    assert set(rides[strict_ids]) == {"strict0", "strict1", "strict2"}
    assert len(test_ids) == 9
    assert set(rides[test_ids]) == {"test0", "test1", "test2"}


def test_online_sample_budget_is_exact_deterministic_and_ride_balanced():
    rides = np.asarray([
        ride for ride in ("r0", "r1", "r2", "r3", "r4")
        for _row in range(3)
    ])
    ids = list(range(15))
    selections = [
        M._balanced_online_sample_ids(ids, rides, sample_budget=9, cycle=cycle)
        for cycle in range(5)
    ]
    assert selections == [
        M._balanced_online_sample_ids(ids, rides, sample_budget=9, cycle=cycle)
        for cycle in range(5)
    ]
    aggregate = {ride: 0 for ride in sorted(set(rides))}
    for selected in selections:
        assert len(selected) == len(set(selected)) == 9
        counts = {
            ride: int(np.sum(rides[selected] == ride)) for ride in aggregate
        }
        assert max(counts.values()) - min(counts.values()) <= 1
        for ride, count in counts.items():
            aggregate[ride] += count
    assert set(aggregate.values()) == {9}
    assert M._balanced_online_sample_ids(ids, rides, 0, cycle=99) == ids
    with pytest.raises(RuntimeError, match="exceeds local rows"):
        M._balanced_online_sample_ids(ids, rides, 16, cycle=0)


def _target_bank(path: Path, *, merged: bool = False):
    rng = np.random.default_rng(3)
    n = 16 if merged else 12
    latent = rng.normal(size=(n, 1, 16, 4, 4)).astype(np.float32)
    # Learnable local field with a small online change at the second step.
    g0 = np.tanh(latent).astype(np.float32)
    g1 = g0.copy(); g1[n // 2:] *= -0.5
    if merged:
        roles = np.asarray(
            (["surrogate_train"] * 4 + ["surrogate_test"] * 4) * 2
        )
        crop_index = np.asarray([0, 1, 0, 1] * 4, dtype=np.int16)
        source_unit_id = np.asarray([
            f"step{step}:{role}:unit{unit}"
            for step in range(2)
            for role in ("train", "test")
            for unit in range(2)
            for _crop in range(2)
        ])
        metadata = {
            "source": "vgg", "rotation": 0, "crop_indices": [0, 1],
            "crop_merge": {
                "schema_version": 1,
                "ride_disjoint_roles_verified": True,
                "capture_latents_verified_exact": True,
                "target_row_multiplier": 2.0,
                "role_crop_counts": {
                    "surrogate_train/crop0": 4,
                    "surrogate_train/crop1": 4,
                    "surrogate_test/crop0": 4,
                    "surrogate_test/crop1": 4,
                },
            },
        }
    else:
        roles = np.asarray(
            (["surrogate_train"] * 3 + ["surrogate_test"] * 3) * 2
        )
        crop_index = None
        source_unit_id = None
        metadata = {"source": "vgg", "rotation": 0}
    arrays = dict(
        latent=latent,
        gradient_online=g1,
        gradient_converged=g0,
        rgbmaxmin=np.zeros((n, 1, 6, 4, 4), dtype=np.float16),
        teacher_feature_stats=rng.normal(size=(n, 1, 9)).astype(np.float32),
        teacher_feature_head_grad_online=rng.normal(
            size=(n, 1, 9),
        ).astype(np.float32),
        teacher_feature_head_grad_converged=rng.normal(
            size=(n, 1, 9),
        ).astype(np.float32),
        role=roles,
        ride=np.asarray([f"ride{i}" for i in range(n)]),
        origin_yx=(
            np.stack([crop_index, crop_index * 3], axis=1)
            if merged else np.zeros((n, 2), dtype=np.int16)
        ),
        step=np.asarray([0] * (n // 2) + [5] * (n // 2), dtype=np.int32),
        metadata=np.asarray(json.dumps(metadata)),
    )
    if merged:
        arrays["crop_index"] = crop_index
        arrays["source_unit_id"] = source_unit_id
    np.savez(path, **arrays)


def _args(bank: Path, out: Path, mode: str, condition: str = "latent"):
    return argparse.Namespace(
        targets=bank, output=out, condition=condition, teacher_mode=mode,
        device="cpu", lr=1e-3, checkpoints="1,2", substeps_per_step=2,
        seeds=1, seed=11, batch_size=3, width=8, blocks=1,
    )


def test_frozen_ceiling_and_online_tracking_have_distinct_outputs(tmp_path):
    bank = tmp_path / "bank.npz"
    _target_bank(bank)
    frozen = tmp_path / "frozen.json"
    online = tmp_path / "online.json"
    M._fit_surrogate(_args(bank, frozen, "converged"))
    M._fit_surrogate(_args(bank, online, "online"))
    a = json.loads(frozen.read_text())
    b = json.loads(online.read_text())
    assert "2" in a["aggregate"]
    assert "post_current_global_cosine" in b["aggregate"]
    assert len(b["results"][0]["tracking"]) == 2
    assert b["train_ride_policy"] == "bank"
    assert b["online_samples_per_substep"] == 0


def test_selected_frame_teacher_evidence_maps_five_frames_to_three_latents():
    stats = torch.arange(5 * 2, dtype=torch.float32).reshape(5, 2)
    head_grad = torch.arange(10, 20, dtype=torch.float32).reshape(5, 2)
    grouped_stats, grouped_grad, assignment = (
        M._group_selected_teacher_evidence(
            stats, head_grad,
            selected_frame_indices=[0, 3, 6, 8, 11],
            decoded_frames=12, latent_frames=3,
        )
    )
    assert assignment == [0, 0, 1, 2, 2]
    assert torch.equal(grouped_stats[0], stats[[0, 1]].mean(dim=0))
    assert torch.equal(grouped_stats[1], stats[2])
    assert torch.equal(grouped_stats[2], stats[[3, 4]].mean(dim=0))
    assert torch.equal(grouped_grad[0], head_grad[[0, 1]].sum(dim=0))
    assert torch.equal(grouped_grad[1], head_grad[2])
    assert torch.equal(grouped_grad[2], head_grad[[3, 4]].sum(dim=0))

    try:
        M._group_selected_teacher_evidence(
            stats[:2], head_grad[:2], selected_frame_indices=[0, 3],
            decoded_frames=12, latent_frames=3,
        )
    except ValueError as exc:
        assert "do not cover every latent frame" in str(exc)
    else:
        raise AssertionError("incomplete selected-frame coverage was accepted")


def test_saved_head_stat_gradient_is_exact_and_fit_cli_consumes_it(tmp_path):
    torch.manual_seed(7)
    head = M.ALIGNED.OnlineTapHead([4], hidden=5)
    stats = torch.randn(5, 4)
    got = M._head_mean_stat_gradient(head, stats)
    leaf = stats.detach().clone().requires_grad_(True)
    expected = torch.autograd.grad(head(leaf).mean(), leaf)[0]
    assert torch.equal(got, expected)
    assert not got.requires_grad
    assert all(parameter.grad is None for parameter in head.parameters())

    bank = tmp_path / "teacher_condition_bank.npz"
    out = tmp_path / "teacher_condition.json"
    _target_bank(bank)
    M._fit_surrogate(_args(
        bank, out, "converged", condition="vggstats_headgrad",
    ))
    payload = json.loads(out.read_text())
    assert payload["condition"] == "vggstats_headgrad"
    assert len(payload["results"]) == 1


def test_merged_bank_trains_both_crops_but_filters_held_out_eval(tmp_path):
    bank = tmp_path / "merged_bank.npz"
    out = tmp_path / "crop0_eval.json"
    _target_bank(bank, merged=True)
    args = _args(bank, out, "converged")
    args.eval_crop_index = 0
    M._fit_surrogate(args)
    payload = json.loads(out.read_text())
    assert payload["train_samples"] == 8
    assert payload["test_samples"] == 4
    audit = payload["evaluation_crop_filter"]
    assert audit["applied"] is True
    assert audit["requested_crop_index"] == 0
    assert audit["test_samples_before_filter"] == 8
    assert audit["test_samples_after_filter"] == 4
    assert audit["train_crop_counts"] == {"0": 4, "1": 4}
    assert audit["test_crop_counts_before_filter"] == {"0": 4, "1": 4}
    assert audit["test_crop_counts_after_filter"] == {"0": 4}
    metrics = payload["results"][0]["checkpoints"]["2"]
    assert metrics["train"]["cosine"]["n"] == 8
    assert metrics["test"]["cosine"]["n"] == 4


def test_eval_crop_filter_rejects_unverified_or_unpaired_banks(tmp_path):
    legacy = tmp_path / "legacy.npz"
    _target_bank(legacy)
    args = _args(legacy, tmp_path / "legacy.json", "converged")
    args.eval_crop_index = 0
    with pytest.raises(RuntimeError, match="verified merged-bank provenance"):
        M._fit_surrogate(args)

    merged = tmp_path / "merged.npz"
    _target_bank(merged, merged=True)
    with np.load(merged, allow_pickle=False) as zf:
        arrays = {key: np.array(zf[key], copy=True) for key in zf.files}
    arrays["crop_index"][1] = 0
    with open(merged, "wb") as fh:
        np.savez(fh, **arrays)
    args = _args(merged, tmp_path / "unpaired.json", "converged")
    args.eval_crop_index = 0
    with pytest.raises(RuntimeError, match="paired crop provenance"):
        M._fit_surrogate(args)
