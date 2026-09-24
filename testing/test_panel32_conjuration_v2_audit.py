from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd
import pytest


PATH = Path(__file__).resolve().parents[1] / "grids/eval/panel32_conjuration_v2_audit.py"
SPEC = importlib.util.spec_from_file_location("panel32_conjuration_v2_audit", PATH)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


class Detector:
    HI = 0.7
    MIN_AREA = 0.0015
    CXLO, CXHI = 0.15, 0.85
    EDGE = 0.06
    PERSIST_TAIL = 0.75
    PERSIST_FRAC = 0.55

    @staticmethod
    def cen(box):
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    @staticmethod
    def edge_trace(track, width):
        return min(track["bbox"][0] / width, (width - track["bbox"][2]) / width)


def persistent_track(cls="person"):
    return {
        "cls": cls, "bbox": [200, 100, 500, 400], "birth": 64, "last": 124,
        "hits": 61, "peak": 0.97,
    }


def test_person_and_mobile_objects_enter_fail_closed_candidate_queue():
    for cls in ("person", "bicycle", "dog", "car"):
        passed, gates = M.cheap_gate(persistent_track(cls), 125, 832, 480, 29, Detector)
        assert passed, gates


def test_non_salient_or_nonpersistent_tracks_do_not_enter_queue():
    passed, gates = M.cheap_gate(persistent_track("toaster"), 125, 832, 480, 29, Detector)
    assert not passed and not gates["salient_class"]
    track = persistent_track()
    track["last"] = 70
    passed, gates = M.cheap_gate(track, 125, 832, 480, 29, Detector)
    assert not passed and not gates["alive_in_tail"]


def test_negative_legacy_score_is_still_retained_for_human_adjudication():
    class FakeDetector(Detector):
        @staticmethod
        def track(_detections, _width, _height):
            return [persistent_track()]

        @staticmethod
        def features(_video, _detections, track, _n, _ctx):
            return {
                "cls": track["cls"], "birth": track["birth"], "last": track["last"],
                "peak": track["peak"], "box": track["bbox"], "onset": 1.1,
                "zprobe": 0.0, "bncc": 0.72,
            }

        @staticmethod
        def score(_features):
            return -0.78

    # A 29-frame context plus one six-second 16-fps window.
    video = np.zeros((125, 480, 832, 3), dtype=np.uint8)
    rows = M.candidate_rows(
        video, [[] for _ in video], context_frames=29, fps=16,
        windows=(0,), detector=FakeDetector,
    )
    assert rows[0]["candidate_count"] == 1
    candidate = rows[0]["candidates"][0]
    assert candidate["legacy_score"] == -0.78
    assert candidate["legacy_positive"] is False
    assert candidate["human_adjudication"] == "unadjudicated"


def test_default_windows_are_nonoverlapping_and_cover_30_seconds():
    assert M.parse_windows("0,6,12,18,24") == (0, 6, 12, 18, 24)
    assert M.needed_last_index(29, 16, M.DEFAULT_WINDOWS) == 508
    with pytest.raises(M.AuditError, match="must not overlap"):
        M.parse_windows("0,6,9,12")


def test_sharding_is_disjoint_and_complete(tmp_path):
    rows = []
    for index in range(12):
        rows.append({
            "scene": f"scene{index:02d}_F", "model": "minwm", "path": "/x",
            "decoded_frames": 509, "context_frames": 29, "fps": 16,
        })
    pd.DataFrame(rows).to_csv(tmp_path / "video_manifest.csv", index=False)
    shards = [M.load_work(tmp_path / "video_manifest.csv", models=None, scenes=None,
                          shard_index=i, shard_count=4) for i in range(4)]
    keys = [{(row.scene, row.model) for row in shard.itertuples()} for shard in shards]
    assert not any(keys[i] & keys[j] for i in range(4) for j in range(i + 1, 4))
    assert set().union(*keys) == {(row["scene"], row["model"]) for row in rows}


def test_refuses_published_conjuration_output(tmp_path):
    with pytest.raises(M.AuditError, match="protected final output"):
        M.validate_output_location(tmp_path, tmp_path)
    with pytest.raises(M.AuditError, match="protected final output"):
        M.validate_output_location(tmp_path, tmp_path / "conjuration")
    M.validate_output_location(tmp_path, tmp_path / "conjuration_v2_audit")


def test_resume_requires_matching_sha_windows_classes_and_evidence(tmp_path):
    evidence = tmp_path / "evidence.png"
    evidence.write_bytes(b"png")
    target = tmp_path / "record.json"
    record = {
        "schema_version": 1, "audit": "panel32_conjuration_v2",
        "scene": "scene_F", "model": "minwm", "video_sha256": "a" * 64,
        "salient_classes": sorted(M.SALIENT_MOBILE_CLASSES),
        "windows": [{"window_start_s": 0, "candidate_count": 1,
                     "candidates": [{"evidence_path": str(evidence)}]}],
    }
    target.write_text(json.dumps(record))
    assert M.reusable_record(target, scene="scene_F", model="minwm",
                             digest="a" * 64, windows=(0,)) == record
    assert M.reusable_record(target, scene="scene_F", model="minwm",
                             digest="b" * 64, windows=(0,)) is None
