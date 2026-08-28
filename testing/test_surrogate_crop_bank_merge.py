import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


_PATH = (
    Path(__file__).resolve().parents[1]
    / "analysis/gan_tuning/surrogate_crop_bank_merge.py"
)
_SPEC = importlib.util.spec_from_file_location("surrogate_crop_bank_merge", _PATH)
M = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = M
_SPEC.loader.exec_module(M)


def _fixture(tmp_path: Path):
    capture = tmp_path / "capture"
    capture.mkdir(parents=True)
    head = tmp_path / "head.pt"
    head.write_bytes(b"teacher-head")
    split = {
        "rank_roles": {
            "teacher": [0, 1], "surrogate_train": [2, 3, 4],
            "surrogate_test": [5, 6, 7],
        },
        "teacher": ["teacher_ride"],
        "surrogate_train": ["train_ride"],
        "surrogate_test": ["test_ride"],
    }
    records = []
    for rank, ride, role in (
        (2, "train_ride", "surrogate_train"),
        (5, "test_ride", "surrogate_test"),
    ):
        path = capture / f"step000000_rank{rank:02d}.npz"
        latent = np.zeros((2, 2, 1, 16, 4, 4), dtype=np.float16)
        for row in range(2):
            latent[row, 0] = 10 + rank + row
            latent[row, 1] = 30 + rank + row
        meta = {
            "step": 0, "rank": rank, "ride": ride,
            "crop_origins_yx": [[0, 0], [1, 3]],
            "crop_rows": 4, "crop_cols": 4,
        }
        np.savez(
            path, fake_latent=latent,
            metadata=np.asarray(json.dumps(meta)),
        )
        records.append((path, latent, ride, role, rank))

    banks = []
    for crop in (0, 1):
        latent_rows = []
        rides = []
        roles = []
        origins = []
        steps = []
        for _path, latent, ride, role, _rank in records:
            for row in range(2):
                latent_rows.append(latent[row, crop].astype(np.float32))
                rides.append(ride); roles.append(role)
                origins.append((0, 0) if crop == 0 else (1, 3))
                steps.append(0)
        latent_rows = np.asarray(latent_rows, dtype=np.float32)
        bank = tmp_path / f"crop{crop}.npz"
        metadata = {
            "source": "vgg", "rotation": 0, "head": str(head),
            "crop_index": crop, "teacher_modes": ["online", "converged"],
            "split": split, "samples": len(latent_rows),
        }
        np.savez(
            bank,
            latent=latent_rows,
            gradient_online=np.tanh(latent_rows).astype(np.float32),
            gradient_converged=(2 * np.tanh(latent_rows)).astype(np.float32),
            ride=np.asarray(rides), role=np.asarray(roles),
            origin_yx=np.asarray(origins, dtype=np.int16),
            step=np.asarray(steps, dtype=np.int32),
            teacher_value_online=np.zeros(len(latent_rows), dtype=np.float32),
            teacher_value_converged=np.zeros(len(latent_rows), dtype=np.float32),
            teacher_grad_rms_online=np.ones(len(latent_rows), dtype=np.float32),
            teacher_grad_rms_converged=np.ones(len(latent_rows), dtype=np.float32),
            metadata=np.asarray(json.dumps(metadata)),
        )
        banks.append(bank)
    return capture, banks, head


def _args(capture, banks, output):
    return argparse.Namespace(
        capture_dir=capture, crop0=banks[0], crop1=banks[1], output=output,
    )


def test_merge_reconstructs_provenance_and_doubles_views_not_source_units(tmp_path):
    capture, banks, _head = _fixture(tmp_path)
    output = tmp_path / "merged.npz"
    M._merge(_args(capture, banks, output))
    with np.load(output, allow_pickle=False) as zf:
        meta = json.loads(zf["metadata"].item())
        assert zf["latent"].shape[0] == 8
        assert zf["crop_index"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
        assert len(set(zf["source_unit_id"].astype(str))) == 4
        assert set(zf["domain"].astype(str)) == {"fake"}
    merge = meta["crop_merge"]
    assert merge["target_row_multiplier"] == 2.0
    assert merge["distinct_crop_origin_multiplier"] == 2.0
    assert merge["unique_ride_step_row_multiplier"] == 1.0
    assert merge["statistical_independence_multiplier"] is None
    assert merge["ride_disjoint_roles_verified"] is True
    assert merge["geometry"]["exact_duplicate_origins"] == 0


def test_merge_rejects_latent_provenance_mismatch(tmp_path):
    capture, banks, _head = _fixture(tmp_path)
    with np.load(banks[1], allow_pickle=False) as zf:
        arrays = {key: np.array(zf[key], copy=True) for key in zf.files}
    arrays["latent"][0, 0, 0, 0, 0] += 1
    with open(banks[1], "wb") as fh:
        np.savez(fh, **arrays)
    with pytest.raises(RuntimeError, match="not its captured latent"):
        M._merge(_args(capture, banks, tmp_path / "merged.npz"))


def test_merge_rejects_role_leakage_and_duplicate_crop_indices(tmp_path):
    capture, banks, _head = _fixture(tmp_path)
    with np.load(banks[1], allow_pickle=False) as zf:
        arrays = {key: np.array(zf[key], copy=True) for key in zf.files}
    meta = json.loads(arrays["metadata"].item())
    meta["crop_index"] = 0
    arrays["metadata"] = np.asarray(json.dumps(meta))
    with open(banks[1], "wb") as fh:
        np.savez(fh, **arrays)
    with pytest.raises(RuntimeError, match="requires crop_index 0 and 1"):
        M._merge(_args(capture, banks, tmp_path / "merged.npz"))

    capture, banks, _head = _fixture(tmp_path / "leak")
    for bank in banks:
        with np.load(bank, allow_pickle=False) as zf:
            arrays = {key: np.array(zf[key], copy=True) for key in zf.files}
        meta = json.loads(arrays["metadata"].item())
        meta["split"]["surrogate_test"].append("train_ride")
        arrays["metadata"] = np.asarray(json.dumps(meta))
        with open(bank, "wb") as fh:
            np.savez(fh, **arrays)
    with pytest.raises(RuntimeError, match="ride leakage"):
        M._merge(_args(capture, banks, tmp_path / "leak" / "merged.npz"))
