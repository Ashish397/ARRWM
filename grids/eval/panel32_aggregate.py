"""Fail-closed aggregation for the balanced mixed panel32 evaluation.

This module is deliberately separate from the historical FrodoBots
summarizers.  It consumes a normalized, per-video metric table only after the
metric producers have finished, validates the complete Cartesian product, and
emits dataset-stratified and equal-dataset-macro failure rates.

Four metrics (control, style, geometry, and HF) are independent six-second
window decisions at starts 0, 6, 12, 18, and 24 seconds.  Conjuration and
relocation are six-second onset events and occur exactly once per video; they
are never copied across later endpoints.  Control keeps movement commands and
no-op in separate populations.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from .panel32_manifest import (
        ACTIONS,
        DATASET_COUNTS,
        PanelManifest,
        load_panel_manifest,
        sha256_file,
    )
except ImportError:  # Direct execution: python grids/eval/panel32_aggregate.py
    from panel32_manifest import (  # type: ignore
        ACTIONS,
        DATASET_COUNTS,
        PanelManifest,
        load_panel_manifest,
        sha256_file,
    )


CONFIG_SCHEMA_VERSION = 1
OBSERVATION_SCHEMA_VERSION = 1
AGGREGATE_SCHEMA_VERSION = 1
FAMILY_SEAT_COUNT = 6
WINDOW_STARTS = (0, 6, 12, 18, 24)
WINDOW_DURATION_S = 6
WINDOW_METRICS = ("control", "style", "geometry", "hf")
ONSET_METRICS = ("conjuration", "relocation")
ALL_METRICS = WINDOW_METRICS + ONSET_METRICS
WINDOW_SEMANTICS = "independent_six_second_window"
ONSET_SEMANTICS = "six_second_onset_event"
OBSERVATION_COLUMNS = (
    "observation_schema_version",
    "panel_id",
    "evaluation_id",
    "context_id",
    "dataset",
    "action",
    "model_id",
    "metric",
    "semantics",
    "window_start_s",
    "window_end_s",
    "failed",
    "family_seat_count",
    "video_sha256",
)


CONFIG_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ARRWM mixed panel32 aggregation configuration",
    "type": "object",
    "required": [
        "schema_version", "evaluation_id", "actions", "window_metrics",
        "onset_metrics", "window_starts_s", "window_duration_s",
        "family_seat_count", "models",
    ],
    "additionalProperties": False,
    "properties": {
        "schema_version": {"const": CONFIG_SCHEMA_VERSION},
        "evaluation_id": {"type": "string", "minLength": 1},
        "actions": {"const": list(ACTIONS)},
        "window_metrics": {"const": list(WINDOW_METRICS)},
        "onset_metrics": {"const": list(ONSET_METRICS)},
        "window_starts_s": {"const": list(WINDOW_STARTS)},
        "window_duration_s": {"const": WINDOW_DURATION_S},
        "family_seat_count": {"const": FAMILY_SEAT_COUNT},
        "models": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["model_id", "label", "family", "role"],
                "additionalProperties": False,
                "properties": {
                    "model_id": {"type": "string", "minLength": 1},
                    "label": {"type": "string", "minLength": 1},
                    "family": {"type": "string", "minLength": 1},
                    "role": {"enum": ["ours", "external"]},
                },
            },
        },
    },
}


class AggregationError(ValueError):
    """The aggregation inputs are incomplete, duplicated, or inconsistent."""


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    label: str
    family: str
    role: str


@dataclass(frozen=True)
class AggregateConfig:
    path: Path
    sha256: str
    evaluation_id: str
    models: tuple[ModelSpec, ...]
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class Observation:
    panel_id: str
    evaluation_id: str
    context_id: str
    dataset: str
    action: str
    model_id: str
    metric: str
    semantics: str
    window_start_s: int
    window_end_s: int
    failed: int
    family_seat_count: int
    video_sha256: str

    @property
    def key(self) -> tuple[str, str, str, str, int]:
        return (
            self.context_id,
            self.action,
            self.model_id,
            self.metric,
            self.window_start_s,
        )

    @property
    def video_key(self) -> tuple[str, str, str]:
        return self.context_id, self.action, self.model_id


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AggregationError(message)


def _exact_keys(mapping: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(mapping) - allowed
    missing = allowed - set(mapping)
    _require(not missing, f"{where}: missing keys {sorted(missing)}")
    _require(not unknown, f"{where}: unknown keys {sorted(unknown)}")


def _strict_string(value: Any, where: str) -> str:
    _require(isinstance(value, str), f"{where} must be a string")
    value = value.strip()
    _require(bool(value), f"{where} is empty")
    return value


def load_aggregate_config(path: str | Path) -> AggregateConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AggregationError(f"cannot read aggregation config {config_path}: {exc}") from exc
    _require(isinstance(payload, Mapping), "aggregation config root must be an object")
    required = {
        "schema_version", "evaluation_id", "actions", "window_metrics",
        "onset_metrics", "window_starts_s", "window_duration_s",
        "family_seat_count", "models",
    }
    _exact_keys(payload, required, "aggregation config")
    _require(payload["schema_version"] == CONFIG_SCHEMA_VERSION,
             f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    evaluation_id = _strict_string(payload["evaluation_id"], "evaluation_id")
    _require(payload["actions"] == list(ACTIONS),
             f"actions must be exactly {list(ACTIONS)}")
    _require(payload["window_metrics"] == list(WINDOW_METRICS),
             f"window_metrics must be exactly {list(WINDOW_METRICS)}")
    _require(payload["onset_metrics"] == list(ONSET_METRICS),
             f"onset_metrics must be exactly {list(ONSET_METRICS)}")
    _require(payload["window_starts_s"] == list(WINDOW_STARTS),
             f"window_starts_s must be exactly {list(WINDOW_STARTS)}")
    _require(payload["window_duration_s"] == WINDOW_DURATION_S,
             f"window_duration_s must be {WINDOW_DURATION_S}")
    _require(payload["family_seat_count"] == FAMILY_SEAT_COUNT,
             "family_seat_count must be 6; five-family results are incompatible")
    raw_models = payload["models"]
    _require(isinstance(raw_models, list) and len(raw_models) > 0,
             "models must contain at least one row")
    models = []
    for index, row in enumerate(raw_models):
        _require(isinstance(row, Mapping), f"models[{index}] must be an object")
        _exact_keys(row, {"model_id", "label", "family", "role"},
                    f"models[{index}]")
        model = ModelSpec(
            model_id=_strict_string(row["model_id"], f"models[{index}].model_id"),
            label=_strict_string(row["label"], f"models[{index}].label"),
            family=_strict_string(row["family"], f"models[{index}].family"),
            role=_strict_string(row["role"], f"models[{index}].role"),
        )
        _require(model.role in {"ours", "external"},
                 f"{model.model_id}: role must be ours or external")
        models.append(model)
    _require(len({model.model_id for model in models}) == len(models),
             "model_id values must be unique")
    # The main comparison is the one-model-per-family six-seat panel and must
    # include YUME as the fifth external baseline.  ODE and DMD subset configs
    # deliberately contain repeated/partial families, but still record that
    # their HF decisions were computed against the same six frozen seats.
    families = {model.family for model in models}
    if len(models) == FAMILY_SEAT_COUNT and len(families) == FAMILY_SEAT_COUNT:
        _require(sum(model.role == "ours" for model in models) == 1,
                 "six-family comparison must contain exactly one ours row")
        _require(sum(model.role == "external" for model in models) == 5,
                 "six-family comparison must contain exactly five external rows")
        yume = [model for model in models if model.model_id == "yume5b"]
        _require(len(yume) == 1 and yume[0].role == "external",
                 "six-family comparison must include external model_id yume5b")
    return AggregateConfig(
        path=config_path,
        sha256=sha256_file(config_path),
        evaluation_id=evaluation_id,
        models=tuple(models),
        raw=payload,
    )


def _strict_int(value: str, where: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise AggregationError(f"{where} must be an integer, got {value!r}") from exc
    _require(str(parsed) == str(value).strip(),
             f"{where} must use canonical integer syntax, got {value!r}")
    return parsed


def _parse_observation(
    raw: Mapping[str, str], line: int, panel: PanelManifest,
    config: AggregateConfig, context_datasets: Mapping[str, str],
    model_ids: set[str],
) -> Observation:
    where = f"observations line {line}"
    version = _strict_int(raw["observation_schema_version"],
                          f"{where}.observation_schema_version")
    _require(version == OBSERVATION_SCHEMA_VERSION,
             f"{where}: observation_schema_version must be {OBSERVATION_SCHEMA_VERSION}")
    panel_id = _strict_string(raw["panel_id"], f"{where}.panel_id")
    evaluation_id = _strict_string(raw["evaluation_id"], f"{where}.evaluation_id")
    context_id = _strict_string(raw["context_id"], f"{where}.context_id")
    dataset = _strict_string(raw["dataset"], f"{where}.dataset")
    action = _strict_string(raw["action"], f"{where}.action")
    model_id = _strict_string(raw["model_id"], f"{where}.model_id")
    metric = _strict_string(raw["metric"], f"{where}.metric")
    semantics = _strict_string(raw["semantics"], f"{where}.semantics")
    start = _strict_int(raw["window_start_s"], f"{where}.window_start_s")
    end = _strict_int(raw["window_end_s"], f"{where}.window_end_s")
    failed = _strict_int(raw["failed"], f"{where}.failed")
    seats = _strict_int(raw["family_seat_count"], f"{where}.family_seat_count")
    video_sha256 = _strict_string(raw["video_sha256"], f"{where}.video_sha256")

    _require(panel_id == panel.panel_id, f"{where}: panel_id mismatch")
    _require(evaluation_id == config.evaluation_id,
             f"{where}: evaluation_id mismatch")
    _require(context_id in context_datasets, f"{where}: unknown context_id {context_id}")
    _require(dataset == context_datasets.get(context_id),
             f"{where}: dataset does not match panel manifest for {context_id}")
    _require(action in ACTIONS, f"{where}: unknown action {action}")
    _require(model_id in model_ids, f"{where}: unknown model_id {model_id}")
    _require(metric in ALL_METRICS, f"{where}: unknown metric {metric}")
    _require(failed in (0, 1), f"{where}: failed must be 0 or 1")
    _require(seats == FAMILY_SEAT_COUNT,
             f"{where}: family_seat_count must be 6; old five-seat rows are invalid")
    _require(len(video_sha256) == 64
             and all(char in "0123456789abcdef" for char in video_sha256),
             f"{where}: video_sha256 must be 64 lowercase hex characters")
    if metric in WINDOW_METRICS:
        _require(semantics == WINDOW_SEMANTICS,
                 f"{where}: {metric} must use {WINDOW_SEMANTICS}")
        _require(start in WINDOW_STARTS and end == start + WINDOW_DURATION_S,
                 f"{where}: invalid independent window [{start}, {end})")
    else:
        _require(semantics == ONSET_SEMANTICS,
                 f"{where}: {metric} must use {ONSET_SEMANTICS}")
        _require(start == 0 and end == WINDOW_DURATION_S,
                 f"{where}: onset metric {metric} must occur once at [0, 6)")
    return Observation(
        panel_id=panel_id,
        evaluation_id=evaluation_id,
        context_id=context_id,
        dataset=dataset,
        action=action,
        model_id=model_id,
        metric=metric,
        semantics=semantics,
        window_start_s=start,
        window_end_s=end,
        failed=failed,
        family_seat_count=seats,
        video_sha256=video_sha256,
    )


def expected_keys(panel: PanelManifest, config: AggregateConfig) -> set[tuple[str, str, str, str, int]]:
    keys: set[tuple[str, str, str, str, int]] = set()
    for context in panel.contexts:
        for action in ACTIONS:
            for model in config.models:
                for metric in WINDOW_METRICS:
                    for start in WINDOW_STARTS:
                        keys.add((context.context_id, action, model.model_id, metric, start))
                for metric in ONSET_METRICS:
                    keys.add((context.context_id, action, model.model_id, metric, 0))
    return keys


def load_observations(
    path: str | Path, panel: PanelManifest, config: AggregateConfig,
) -> tuple[list[Observation], str]:
    observations_path = Path(path).expanduser().resolve()
    context_datasets = {context.context_id: context.dataset for context in panel.contexts}
    model_ids = {model.model_id for model in config.models}
    try:
        handle = observations_path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise AggregationError(f"cannot read observations {observations_path}: {exc}") from exc
    with handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames is not None, "observations CSV has no header")
        _require(tuple(reader.fieldnames) == OBSERVATION_COLUMNS,
                 "observations CSV columns/order must be exactly "
                 f"{list(OBSERVATION_COLUMNS)}")
        observations = [
            _parse_observation(row, line, panel, config, context_datasets, model_ids)
            for line, row in enumerate(reader, start=2)
        ]

    keys = [row.key for row in observations]
    seen: set[tuple[str, str, str, str, int]] = set()
    duplicate = next((key for key in keys if key in seen or seen.add(key)), None)
    _require(duplicate is None, f"duplicate metric observation: {duplicate}")
    actual = set(keys)
    expected = expected_keys(panel, config)
    missing = expected - actual
    extra = actual - expected
    if missing:
        sample = sorted(missing)[:3]
        raise AggregationError(
            f"missing {len(missing)} of {len(expected)} metric observations; sample={sample}")
    if extra:
        sample = sorted(extra)[:3]
        raise AggregationError(f"unexpected metric observations; sample={sample}")
    _require(len(observations) == len(expected),
             f"expected {len(expected)} observations, found {len(observations)}")

    video_hashes: dict[tuple[str, str, str], str] = {}
    for row in observations:
        previous = video_hashes.setdefault(row.video_key, row.video_sha256)
        _require(previous == row.video_sha256,
                 f"metric rows disagree on video SHA for {row.video_key}")
    expected_videos = len(panel.contexts) * len(ACTIONS) * len(config.models)
    _require(len(video_hashes) == expected_videos,
             f"expected {expected_videos} unique model videos, found {len(video_hashes)}")
    return observations, sha256_file(observations_path)


def _rate_row(
    rows: Iterable[Observation], dataset: str, model: ModelSpec, metric: str,
    semantics: str, start: int, end: int, population: str,
) -> dict[str, Any]:
    values = list(rows)
    scored = len(values)
    flagged = sum(row.failed for row in values)
    _require(scored > 0, "internal error: empty aggregation cell")
    return {
        "dataset": dataset,
        "model_id": model.model_id,
        "model_label": model.label,
        "family": model.family,
        "role": model.role,
        "metric": metric,
        "semantics": semantics,
        "window_start_s": start,
        "window_end_s": end,
        "population": population,
        "scored": scored,
        "flagged": flagged,
        "rate_pct": 100.0 * flagged / scored,
        "family_seat_count": FAMILY_SEAT_COUNT,
        "reference_panel": ("six families; one seat per family"
                            if metric == "hf" else "not applicable"),
    }


def aggregate_by_dataset(
    observations: list[Observation], config: AggregateConfig,
) -> list[dict[str, Any]]:
    index: dict[tuple[str, str, str, int], list[Observation]] = {}
    for row in observations:
        index.setdefault(
            (row.dataset, row.model_id, row.metric, row.window_start_s), []
        ).append(row)
    output = []
    for dataset in DATASET_COUNTS:
        for model in config.models:
            for metric in WINDOW_METRICS:
                for start in WINDOW_STARTS:
                    candidates = index[(dataset, model.model_id, metric, start)]
                    if metric == "control":
                        populations = (
                            ("directional", [row for row in candidates if row.action != "N"], 64),
                            ("noop", [row for row in candidates if row.action == "N"], 8),
                        )
                    else:
                        populations = (("all_actions", candidates, 72),)
                    for population, rows, expected in populations:
                        _require(len(rows) == expected,
                                 f"{dataset}/{model.model_id}/{metric}/{start}/{population}: "
                                 f"expected {expected} rows, found {len(rows)}")
                        output.append(_rate_row(
                            rows, dataset, model, metric, WINDOW_SEMANTICS,
                            start, start + WINDOW_DURATION_S, population,
                        ))
            for metric in ONSET_METRICS:
                candidates = index[(dataset, model.model_id, metric, 0)]
                _require(len(candidates) == 72,
                         f"{dataset}/{model.model_id}/{metric}/onset: "
                         f"expected 72 rows, found {len(candidates)}")
                output.append(_rate_row(
                    candidates, dataset, model, metric, ONSET_SEMANTICS,
                    0, WINDOW_DURATION_S, "all_actions",
                ))
    return output


def macro_aggregate(dataset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dimensions = (
        "model_id", "model_label", "family", "role", "metric", "semantics",
        "window_start_s", "window_end_s", "population", "family_seat_count",
        "reference_panel",
    )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in dataset_rows:
        grouped.setdefault(tuple(row[key] for key in dimensions), []).append(row)
    output = []
    for key, rows in grouped.items():
        datasets = {row["dataset"] for row in rows}
        _require(datasets == set(DATASET_COUNTS),
                 f"macro cell missing datasets: {set(DATASET_COUNTS) - datasets}")
        _require(len(rows) == len(DATASET_COUNTS), "duplicate dataset macro rows")
        rates = [float(row["rate_pct"]) for row in rows]
        scored = sum(int(row["scored"]) for row in rows)
        flagged = sum(int(row["flagged"]) for row in rows)
        result = dict(zip(dimensions, key))
        result.update({
            "dataset_count": len(DATASET_COUNTS),
            "datasets": ",".join(DATASET_COUNTS),
            "aggregation": "equal_weight_mean_of_dataset_rates",
            "scored_total": scored,
            "flagged_total": flagged,
            "macro_rate_pct": sum(rates) / len(rates),
            "pooled_rate_pct": 100.0 * flagged / scored,
        })
        _require(math.isclose(result["macro_rate_pct"], result["pooled_rate_pct"],
                              rel_tol=0.0, abs_tol=1e-12),
                 "balanced panel invariant failed: macro and pooled rates differ")
        output.append(result)
    return output


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _require(bool(rows), f"refusing to write empty CSV {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def validate_inputs(
    manifest_path: str | Path, config_path: str | Path,
    observations_path: str | Path,
) -> tuple[PanelManifest, AggregateConfig, list[Observation], str]:
    panel = load_panel_manifest(manifest_path, verify_sources=False)
    config = load_aggregate_config(config_path)
    observations, observations_sha256 = load_observations(
        observations_path, panel, config)
    return panel, config, observations, observations_sha256


def aggregate(
    manifest_path: str | Path, config_path: str | Path,
    observations_path: str | Path, output_root: str | Path,
) -> dict[str, Any]:
    panel, config, observations, observations_sha256 = validate_inputs(
        manifest_path, config_path, observations_path)
    dataset_rows = aggregate_by_dataset(observations, config)
    macro_rows = macro_aggregate(dataset_rows)
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset_csv = output / "panel32_rates_by_dataset.csv"
    macro_csv = output / "panel32_rates_macro.csv"
    provenance_json = output / "panel32_aggregation_provenance.json"
    _atomic_write_csv(dataset_csv, dataset_rows)
    _atomic_write_csv(macro_csv, macro_rows)
    expected_observations = len(expected_keys(panel, config))
    provenance = {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "status": "pass",
        "panel_id": panel.panel_id,
        "panel_manifest": str(panel.path),
        "panel_manifest_sha256": panel.sha256,
        "evaluation_id": config.evaluation_id,
        "aggregation_config": str(config.path),
        "aggregation_config_sha256": config.sha256,
        "observations": str(Path(observations_path).expanduser().resolve()),
        "observations_sha256": observations_sha256,
        "observation_rows": len(observations),
        "expected_observation_rows": expected_observations,
        "contexts": len(panel.contexts),
        "dataset_counts": DATASET_COUNTS,
        "actions": list(ACTIONS),
        "models": [model.__dict__ for model in config.models],
        "family_seat_count": FAMILY_SEAT_COUNT,
        "family_reference_policy": "six families; one seat per family",
        "window_metrics": list(WINDOW_METRICS),
        "window_starts_s": list(WINDOW_STARTS),
        "onset_metrics": list(ONSET_METRICS),
        "onset_window": [0, WINDOW_DURATION_S],
        "control_populations": {
            "directional": list(ACTIONS[:-1]),
            "noop": ["N"],
        },
        "dataset_rate_rows": len(dataset_rows),
        "macro_rate_rows": len(macro_rows),
        "macro_policy": "equal_weight_mean_of_four_dataset_rates",
        "outputs": {
            "by_dataset": str(dataset_csv),
            "macro": str(macro_csv),
        },
    }
    _atomic_write_json(provenance_json, provenance)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    schema = subparsers.add_parser("schema", help="print config and observation schemas")
    schema.add_argument("--kind", choices=("config", "observations"), required=True)
    for command in ("validate", "aggregate"):
        child = subparsers.add_parser(command)
        child.add_argument("--manifest", required=True, type=Path)
        child.add_argument("--config", required=True, type=Path)
        child.add_argument("--observations", required=True, type=Path)
        if command == "aggregate":
            child.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "schema":
        payload: Any = (CONFIG_JSON_SCHEMA if args.kind == "config" else {
            "format": "CSV",
            "columns_in_required_order": list(OBSERVATION_COLUMNS),
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "row_count_formula": (
                "32 contexts * 9 actions * configured models * "
                "(4*5 window + 2 onset)"
            ),
            "window_semantics": WINDOW_SEMANTICS,
            "onset_semantics": ONSET_SEMANTICS,
            "family_seat_count": FAMILY_SEAT_COUNT,
        })
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if args.command == "validate":
        panel, config, observations, observations_sha256 = validate_inputs(
            args.manifest, args.config, args.observations)
        print(json.dumps({
            "status": "pass",
            "panel_id": panel.panel_id,
            "evaluation_id": config.evaluation_id,
            "observations": len(observations),
            "observations_sha256": observations_sha256,
            "models": [model.model_id for model in config.models],
            "family_seat_count": FAMILY_SEAT_COUNT,
        }, indent=2, sort_keys=True))
        return
    result = aggregate(args.manifest, args.config, args.observations, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
