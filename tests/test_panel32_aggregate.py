from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from grids.eval.panel32_aggregate import (
    ACTIONS,
    AggregationError,
    FAMILY_SEAT_COUNT,
    OBSERVATION_COLUMNS,
    ONSET_METRICS,
    ONSET_SEMANTICS,
    WINDOW_METRICS,
    WINDOW_SEMANTICS,
    WINDOW_STARTS,
    aggregate,
    load_aggregate_config,
    validate_inputs,
)


DATASETS = ("FrodoBots", "Ego4D", "Sekai", "SpatialVID")
PREFIX = {
    "FrodoBots": "frodobots",
    "Ego4D": "ego4d",
    "Sekai": "sekai",
    "SpatialVID": "spatialvid",
}
MODELS = (
    ("ours", "Ours", "ours-family", "ours"),
    ("lingbot", "LingBot-World-V2", "lingbot-family", "external"),
    ("dreamx", "DreamX-World", "dreamx-family", "external"),
    ("matrixgame2", "Matrix-Game 2.0", "matrixgame-family", "external"),
    ("minwm", "minWM", "minwm-family", "external"),
    ("yume5b", "YUME-5B", "yume-family", "external"),
)


def make_manifest(path: Path) -> dict:
    contexts = []
    for dataset in DATASETS:
        for index in range(8):
            contexts.append({
                "context_id": f"{PREFIX[dataset]}-c{index:02d}",
                "dataset": dataset,
                "source_id": f"{PREFIX[dataset]}-source-{index:02d}",
                "source_video": f"raw/{PREFIX[dataset]}-{index:02d}.mp4",
                "source_sha256": hashlib.sha256(
                    f"{dataset}-{index}".encode()).hexdigest(),
                "source_uri": f"https://example.invalid/{PREFIX[dataset]}-{index:02d}",
                "license": "synthetic-test-only",
                "outdoor_verified": True,
                "outdoor_verification_note": "synthetic outdoor fixture",
                "boundary": {"timestamp_s": 10.0},
            })
    payload = {
        "schema_version": 1,
        "panel_id": "synthetic-mixed-panel32",
        "selection_protocol": "synthetic balanced test fixture",
        "created_utc": "2026-09-23T00:00:00Z",
        "dataset_counts": {dataset: 8 for dataset in DATASETS},
        "actions": list(ACTIONS),
        "canonical_source": {
            "frame_count": 33,
            "fps": 20,
            "width": 832,
            "height": 480,
            "resize_mode": "center_crop",
            "boundary_frame_index": 32,
            "wan_latent_frames": 9,
        },
        "contexts": contexts,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def make_config(path: Path) -> dict:
    payload = {
        "schema_version": 1,
        "evaluation_id": "synthetic-six-family-eval",
        "actions": list(ACTIONS),
        "window_metrics": list(WINDOW_METRICS),
        "onset_metrics": list(ONSET_METRICS),
        "window_starts_s": list(WINDOW_STARTS),
        "window_duration_s": 6,
        "family_seat_count": FAMILY_SEAT_COUNT,
        "models": [
            {"model_id": model_id, "label": label, "family": family, "role": role}
            for model_id, label, family, role in MODELS
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def make_observations(path: Path, manifest: dict, config: dict) -> list[dict[str, str]]:
    rows = []
    dataset_index = {dataset: index for index, dataset in enumerate(DATASETS)}
    for context in manifest["contexts"]:
        for action in ACTIONS:
            for model in config["models"]:
                identity = f"{context['context_id']}/{action}/{model['model_id']}"
                video_sha = hashlib.sha256(identity.encode()).hexdigest()
                common = {
                    "observation_schema_version": "1",
                    "panel_id": manifest["panel_id"],
                    "evaluation_id": config["evaluation_id"],
                    "context_id": context["context_id"],
                    "dataset": context["dataset"],
                    "action": action,
                    "model_id": model["model_id"],
                    "family_seat_count": "6",
                    "video_sha256": video_sha,
                }
                # Dataset rates alternate 0%, 100%, 0%, 100%, hence macro 50%.
                failed = str(dataset_index[context["dataset"]] % 2)
                for metric in WINDOW_METRICS:
                    for start in WINDOW_STARTS:
                        rows.append({
                            **common,
                            "metric": metric,
                            "semantics": WINDOW_SEMANTICS,
                            "window_start_s": str(start),
                            "window_end_s": str(start + 6),
                            "failed": failed,
                        })
                for metric in ONSET_METRICS:
                    rows.append({
                        **common,
                        "metric": metric,
                        "semantics": ONSET_SEMANTICS,
                        "window_start_s": "0",
                        "window_end_s": "6",
                        "failed": failed,
                    })
    write_observations(path, rows)
    return rows


def write_observations(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OBSERVATION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def complete_inputs(tmp_path: Path):
    manifest_path = tmp_path / "panel.json"
    config_path = tmp_path / "aggregate.json"
    observations_path = tmp_path / "observations.csv"
    manifest = make_manifest(manifest_path)
    config = make_config(config_path)
    rows = make_observations(observations_path, manifest, config)
    return manifest_path, config_path, observations_path, manifest, config, rows


def test_complete_balanced_aggregation_and_semantics(complete_inputs, tmp_path: Path) -> None:
    manifest_path, config_path, observations_path, _, _, _ = complete_inputs
    result = aggregate(manifest_path, config_path, observations_path, tmp_path / "out")
    assert result["observation_rows"] == 32 * 9 * 6 * (4 * 5 + 2)
    assert result["family_seat_count"] == 6
    assert result["dataset_rate_rows"] == 4 * 6 * 27
    assert result["macro_rate_rows"] == 6 * 27

    macro_path = Path(result["outputs"]["macro"])
    with macro_path.open(newline="", encoding="utf-8") as handle:
        macro = list(csv.DictReader(handle))
    assert {row["macro_rate_pct"] for row in macro} == {"50.0"}
    assert {row["aggregation"] for row in macro} == {
        "equal_weight_mean_of_dataset_rates"
    }
    onset = [row for row in macro if row["metric"] in ONSET_METRICS]
    assert len(onset) == 6 * 2
    assert {(row["window_start_s"], row["window_end_s"]) for row in onset} == {
        ("0", "6")
    }
    control = [row for row in macro if row["metric"] == "control"]
    assert {row["population"] for row in control} == {"directional", "noop"}
    hf = [row for row in macro if row["metric"] == "hf"]
    assert {row["family_seat_count"] for row in hf} == {"6"}
    assert {row["reference_panel"] for row in hf} == {
        "six families; one seat per family"
    }
    assert "five" not in macro_path.read_text(encoding="utf-8").lower()


@pytest.mark.parametrize("mutation,match", [
    ("duplicate", "duplicate metric observation"),
    ("missing", "missing 1"),
    ("late_onset", "onset metric"),
    ("five_seats", "old five-seat rows"),
])
def test_observations_fail_closed(complete_inputs, mutation: str, match: str) -> None:
    manifest_path, config_path, observations_path, _, _, rows = complete_inputs
    if mutation == "duplicate":
        rows.append(dict(rows[0]))
    elif mutation == "missing":
        rows.pop()
    elif mutation == "late_onset":
        target = next(row for row in rows if row["metric"] == "relocation")
        target["window_start_s"] = "6"
        target["window_end_s"] = "12"
    elif mutation == "five_seats":
        rows[0]["family_seat_count"] = "5"
    write_observations(observations_path, rows)
    with pytest.raises(AggregationError, match=match):
        validate_inputs(manifest_path, config_path, observations_path)


def test_config_rejects_five_families_and_requires_yume(complete_inputs) -> None:
    _, config_path, _, _, config, _ = complete_inputs
    config["family_seat_count"] = 5
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(AggregationError, match="five-family"):
        load_aggregate_config(config_path)

    config["family_seat_count"] = 6
    config["models"][-1]["model_id"] = "not-yume"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(AggregationError, match="yume5b"):
        load_aggregate_config(config_path)


def test_checked_in_config_has_six_distinct_families_and_yume() -> None:
    path = Path(__file__).parents[1] / "grids/eval/panel32_aggregate_config.json"
    config = load_aggregate_config(path)
    assert len(config.models) == 6
    assert len({model.family for model in config.models}) == 6
    assert "yume5b" in {model.model_id for model in config.models}
