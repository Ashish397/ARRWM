from pathlib import Path

import pandas as pd
import pytest

from grids.eval.panel32_conjuration_v2_review import (
    ReviewError,
    _validate_record,
    adjudicated_event_table,
    canonical_box,
)


def test_indexed_manifest_series_retains_scene_model_identity(tmp_path: Path) -> None:
    row = pd.Series({"path": str(tmp_path / "video.mp4")}, name=("scene-a", "model-b"))
    with pytest.raises(ReviewError, match="scene-a/model-b: wrong audit schema"):
        _validate_record(tmp_path / "record.json", {}, row, tmp_path)


def test_manifest_row_without_identity_fails_cleanly(tmp_path: Path) -> None:
    row = pd.Series({"path": str(tmp_path / "video.mp4")})
    with pytest.raises(ReviewError, match="manifest row has no .* identity"):
        _validate_record(tmp_path / "record.json", {}, row, tmp_path)


def test_subpixel_detector_box_overshoot_is_clipped() -> None:
    assert canonical_box(
        [741.7, -1.3, 1033.2, 704.2], width=1280, height=704, where="candidate"
    ) == [741.7, 0.0, 1033.2, 704.0]


def test_material_detector_box_overshoot_is_rejected() -> None:
    with pytest.raises(ReviewError, match="invalid/out-of-frame box"):
        canonical_box(
            [741.7, -1.6, 1033.2, 704.0], width=1280, height=704,
            where="candidate",
        )


def test_adjudicated_events_preserve_late_time_and_collapse_collisions() -> None:
    candidates = pd.DataFrame([
        {"scene": "ctx_F", "model": "minwm", "birth_s": 4.0,
         "candidate_id": "early", "adjudication": 1},
        {"scene": "ctx_F", "model": "minwm", "birth_s": 8.5,
         "candidate_id": "late-negative", "adjudication": 0},
        {"scene": "ctx_F", "model": "minwm", "birth_s": 8.5,
         "candidate_id": "late-positive", "adjudication": 1},
    ])
    events = adjudicated_event_table(candidates)
    assert list(events[["scene", "model", "time_s"]].itertuples(
        index=False, name=None)) == [
            ("ctx_F", "minwm", 4.0), ("ctx_F", "minwm", 8.5)
        ]
    assert events.conjuration_flag.tolist() == [1, 1]
    assert events.accepted_candidate_ids.tolist() == [
        '["early"]', '["late-positive"]'
    ]


def test_adjudicated_events_emits_headers_for_empty_audit() -> None:
    events = adjudicated_event_table(pd.DataFrame())
    assert events.empty
    assert events.columns.tolist() == [
        "scene", "model", "time_s", "conjuration_flag",
        "accepted_candidate_ids",
    ]
