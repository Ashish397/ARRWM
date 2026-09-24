from pathlib import Path

import pandas as pd
import pytest

from grids.eval.panel32_reuse_conjuration_adjudication import reuse


def write(path: Path, rows: list[dict[str, str]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_reuses_only_identical_candidate_evidence(tmp_path):
    previous = tmp_path / "previous.csv"
    current = tmp_path / "current.csv"
    write(previous, [{"candidate_id": "a", "scene": "s", "video_sha256": "h",
                      "adjudication": "1", "reason": "confirmed"}])
    write(current, [{"candidate_id": "a", "scene": "s", "video_sha256": "h",
                     "adjudication": "", "reason": ""}])
    report = reuse(previous, current)
    result = pd.read_csv(current, keep_default_na=False, dtype=str)
    assert report["accepted_candidates"] == 1
    assert result.loc[0, "adjudication"] == "1"
    assert result.loc[0, "reason"] == "confirmed"


def test_rejects_changed_immutable_evidence(tmp_path):
    previous = tmp_path / "previous.csv"
    current = tmp_path / "current.csv"
    write(previous, [{"candidate_id": "a", "scene": "s", "video_sha256": "old",
                      "adjudication": "0", "reason": "negative"}])
    write(current, [{"candidate_id": "a", "scene": "s", "video_sha256": "new",
                     "adjudication": "", "reason": ""}])
    with pytest.raises(ValueError, match="immutable candidate evidence differs"):
        reuse(previous, current)
