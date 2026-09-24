from pathlib import Path

import pandas as pd
import pytest

from grids.eval.panel32_reuse_relocation_adjudication import reuse


def test_reuses_only_same_abrupt_proposals(tmp_path: Path):
    dense = tmp_path / "dense.csv"
    old = tmp_path / "old.csv"
    out = tmp_path / "out.csv"
    pd.DataFrame([
        {"scene": "s", "model": "m", "time_s": 6.0, "abrupt_cut": 1,
         "cross_inliers": 4},
        {"scene": "x", "model": "m", "time_s": 8.0, "abrupt_cut": 0,
         "cross_inliers": 9},
    ]).to_csv(dense, index=False)
    pd.DataFrame([{
        "scene": "s", "model": "m", "time_s": 6.0,
        "relocation_flag": 0, "adjudication_reason": "continuous",
    }]).to_csv(old, index=False)
    report = reuse(dense, old, out)
    assert report["abrupt_candidates"] == 1
    assert out.read_text() == old.read_text()


def test_rejects_changed_abrupt_keys(tmp_path: Path):
    dense = tmp_path / "dense.csv"
    old = tmp_path / "old.csv"
    out = tmp_path / "out.csv"
    pd.DataFrame([{
        "scene": "new", "model": "m", "time_s": 6.0, "abrupt_cut": 1,
    }]).to_csv(dense, index=False)
    pd.DataFrame([{
        "scene": "old", "model": "m", "time_s": 6.0,
        "relocation_flag": 0, "adjudication_reason": "continuous",
    }]).to_csv(old, index=False)
    with pytest.raises(ValueError, match="proposal keys differ"):
        reuse(dense, old, out)
