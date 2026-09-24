"""Convert completed final-v2/v3 measurements to panel32 observations.

The historical evaluators write one table per instrument and include an
overlapping 9--15 second diagnostic window.  The balanced-panel aggregator
instead consumes one normalized row per decision.  This converter is the only
bridge between those formats: it checks the complete Cartesian product,
retains only the five non-overlapping windows, emits conjuration and
relocation once at 0--6 seconds, and refuses to combine measurements whose
video SHA-256 values disagree.

The conversion is model-subset aware.  A completed 15-model evaluation can be
converted independently with the main, ODE, and DMD aggregation configs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable

import pandas as pd

try:
    from .panel32_control_rule import CONTROL_RULE_ID
except ImportError:  # Direct execution from grids/eval.
    from panel32_control_rule import CONTROL_RULE_ID  # type: ignore

try:
    from .panel32_aggregate import (
        FAMILY_SEAT_COUNT,
        OBSERVATION_COLUMNS,
        OBSERVATION_SCHEMA_VERSION,
        ONSET_SEMANTICS,
        WINDOW_SEMANTICS,
        WINDOW_STARTS,
        load_aggregate_config,
    )
    from .panel32_manifest import ACTIONS, load_panel_manifest, sha256_file
except ImportError:  # Direct execution from grids/eval.
    from panel32_aggregate import (  # type: ignore
        FAMILY_SEAT_COUNT,
        OBSERVATION_COLUMNS,
        OBSERVATION_SCHEMA_VERSION,
        ONSET_SEMANTICS,
        WINDOW_SEMANTICS,
        WINDOW_STARTS,
        load_aggregate_config,
    )
    from panel32_manifest import ACTIONS, load_panel_manifest, sha256_file  # type: ignore


CONVERTER_SCHEMA_VERSION = 2
PRODUCER_WINDOW_STARTS = (0, 6, 9, 12, 18, 24)
FINALIZED_V2_WINDOW_STARTS = tuple(WINDOW_STARTS)
HF_ENDPOINTS = tuple(start + 6 for start in WINDOW_STARTS)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REFERENCE_SEATS = {
    "ours": "ours_no_gan",
    "lingbot": "lingbot",
    "dreamx": "dreamx",
    "matrixgame2": "matrixgame2",
    "minwm": "minwm",
    "yume5b": "yume5b",
}


class ConversionError(ValueError):
    """A producer result is missing, duplicated, stale, or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConversionError(message)


def _read_csv(root: Path, name: str, columns: Iterable[str]) -> pd.DataFrame:
    path = root / name
    return _read_csv_path(path, name, columns)


def _read_csv_path(
    path: Path, name: str, columns: Iterable[str],
) -> pd.DataFrame:
    _require(path.is_file(), f"missing required producer table: {path}")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise ConversionError(f"cannot read {path}: {exc}") from exc
    missing = set(columns) - set(frame.columns)
    _require(not missing, f"{name}: missing columns {sorted(missing)}")
    return frame


def _selected(frame: pd.DataFrame, models: set[str], name: str) -> pd.DataFrame:
    _require("model" in frame.columns, f"{name}: missing model column")
    available = set(frame.model.dropna().astype(str))
    missing = models - available
    _require(not missing, f"{name}: missing configured models {sorted(missing)}")
    return frame[frame.model.astype(str).isin(models)].copy()


def _strict_integer_column(frame: pd.DataFrame, column: str, name: str) -> None:
    values = pd.to_numeric(frame[column], errors="coerce")
    _require(values.notna().all(), f"{name}.{column}: contains non-numeric values")
    _require((values == values.round()).all(),
             f"{name}.{column}: contains non-integer values")
    frame[column] = values.astype(int)


def _binary(frame: pd.DataFrame, column: str, name: str) -> pd.Series:
    values = frame[column]
    if values.dtype == bool:
        numeric = values.astype(int)
    else:
        normalized = values.replace({"True": 1, "False": 0, "true": 1, "false": 0})
        numeric = pd.to_numeric(normalized, errors="coerce")
    _require(numeric.notna().all(), f"{name}.{column}: contains missing/non-binary values")
    _require(numeric.isin([0, 1]).all(), f"{name}.{column}: must contain only 0/1")
    return numeric.astype(int)


def _check_grid(
    frame: pd.DataFrame,
    name: str,
    videos: set[tuple[str, str]],
    time_column: str,
    times: Iterable[int],
) -> None:
    _strict_integer_column(frame, time_column, name)
    keys = list(zip(frame.scene.astype(str), frame.model.astype(str), frame[time_column]))
    _require(not pd.Series(keys).duplicated().any(),
             f"{name}: duplicate (scene, model, {time_column}) row")
    expected = {(scene, model, time) for scene, model in videos for time in times}
    actual = set(keys)
    missing = expected - actual
    extra = actual - expected
    _require(not missing,
             f"{name}: missing {len(missing)} rows; sample={sorted(missing)[:3]}")
    _require(not extra,
             f"{name}: unexpected selected-model rows; sample={sorted(extra)[:3]}")


def _check_window_table(
    frame: pd.DataFrame, name: str, videos: set[tuple[str, str]],
) -> None:
    _check_grid(frame, name, videos, "window_start_s", PRODUCER_WINDOW_STARTS)
    _strict_integer_column(frame, "window_end_s", name)
    _require((frame.window_end_s == frame.window_start_s + 6).all(),
             f"{name}: every producer window must be exactly six seconds")


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _merge_hashes(
    canonical: dict[tuple[str, str], str], frame: pd.DataFrame, name: str,
    videos: set[tuple[str, str]],
) -> None:
    _require("video_sha256" in frame.columns, f"{name}: missing video_sha256")
    relevant = frame[["scene", "model", "video_sha256"]].copy()
    relevant["scene"] = relevant.scene.astype(str)
    relevant["model"] = relevant.model.astype(str)
    for key, group in relevant.groupby(["scene", "model"], sort=False):
        values = set(group.video_sha256)
        _require(len(values) == 1, f"{name}: multiple video hashes for {key}")
        digest = next(iter(values))
        _require(_valid_sha(digest), f"{name}: invalid lowercase SHA-256 for {key}")
        previous = canonical.setdefault(key, digest)
        _require(previous == digest,
                 f"{name}: video SHA-256 mismatch for {key}: {digest} != {previous}")
    _require(set(canonical) == videos,
             f"{name}: hash coverage differs from configured video set")


def _manifest_rows(
    root: Path, panel_contexts: dict[str, str], model_ids: set[str],
) -> tuple[pd.DataFrame, set[tuple[str, str]], dict[str, tuple[str, str]]]:
    frame = _read_csv(root, "video_manifest.csv", [
        "scene", "uid", "direction", "model", "family",
    ])
    all_rows = frame.copy()
    _require(not all_rows.duplicated(["scene", "model"]).any(),
             "video_manifest.csv: duplicate (scene, model) row")
    expected_scenes = {
        f"{context_id}_{action}": (context_id, action)
        for context_id in panel_contexts for action in ACTIONS
    }
    expected = {
        (scene, model) for scene in expected_scenes for model in model_ids
    }
    reference_expected = {
        (scene, model) for scene in expected_scenes
        for model in REFERENCE_SEATS.values()
    }
    all_keys = set(zip(all_rows.scene.astype(str), all_rows.model.astype(str)))
    reference_actual = {
        key for key in all_keys if key[1] in REFERENCE_SEATS.values()
    }
    missing_reference = reference_expected - reference_actual
    extra_reference = reference_actual - reference_expected
    _require(not missing_reference,
             "video_manifest.csv: incomplete six-family HF reference panel; "
             f"missing {len(missing_reference)} rows; "
             f"sample={sorted(missing_reference)[:3]}")
    _require(not extra_reference,
             "video_manifest.csv: stale/foreign six-family reference rows; "
             f"sample={sorted(extra_reference)[:3]}")
    for family, model in REFERENCE_SEATS.items():
        observed = set(all_rows.loc[all_rows.model.astype(str) == model, "family"].astype(str))
        _require(observed == {family},
                 f"video_manifest.csv: reference seat {model} must have family {family!r}, "
                 f"found {sorted(observed)}")
    frame = _selected(all_rows, model_ids, "video_manifest.csv")
    actual = set(zip(frame.scene.astype(str), frame.model.astype(str)))
    missing = expected - actual
    extra = actual - expected
    _require(not missing,
             f"video_manifest.csv: missing {len(missing)} videos; sample={sorted(missing)[:3]}")
    _require(not extra,
             f"video_manifest.csv: unexpected selected-model videos; sample={sorted(extra)[:3]}")
    for row in frame.itertuples(index=False):
        context_id, action = expected_scenes[str(row.scene)]
        _require(str(row.uid) == context_id,
                 f"video_manifest.csv: {row.scene} uid is {row.uid!r}, expected {context_id!r}")
        _require(str(row.direction) == action,
                 f"video_manifest.csv: {row.scene} direction is {row.direction!r}, expected {action!r}")
    return frame, expected, expected_scenes


def _relocation_table(
    root: Path, models: set[str], videos: set[tuple[str, str]],
) -> tuple[pd.DataFrame, str, str]:
    panel_path = root / "relocation_panel_rows.csv"
    if panel_path.is_file():
        name = panel_path.name
        frame = _read_csv(root, name, [
            "scene", "model", "horizon_s", "video_sha256",
        ])
        candidates = (
            "relocation_flag_6s_exploratory",
            "relocation_flag_6s",
            "relocation_flag_50",
        )
    else:
        name = "relocation_rows.csv"
        frame = _read_csv(root, name, ["scene", "model", "horizon_s"])
        candidates = ("relocation_flag_50", "relocation_flag_6s")
    flag = next((column for column in candidates if column in frame.columns), None)
    _require(flag is not None, f"{name}: no recognized six-second relocation flag")
    frame = _selected(frame, models, name)
    _strict_integer_column(frame, "horizon_s", name)
    frame = frame[frame.horizon_s == 6].copy()
    _check_grid(frame, name, videos, "horizon_s", (6,))
    frame[flag] = _binary(frame, flag, name)
    return frame, flag, name


def _observation(
    *, panel_id: str, evaluation_id: str, context_id: str, dataset: str,
    action: str, model: str, metric: str, semantics: str, start: int,
    failed: int, digest: str,
) -> dict[str, Any]:
    return {
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "panel_id": panel_id,
        "evaluation_id": evaluation_id,
        "context_id": context_id,
        "dataset": dataset,
        "action": action,
        "model_id": model,
        "metric": metric,
        "semantics": semantics,
        "window_start_s": start,
        "window_end_s": start + 6,
        "failed": int(failed),
        "family_seat_count": FAMILY_SEAT_COUNT,
        "video_sha256": digest,
    }


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _require(bool(rows), f"refusing to write empty observation CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OBSERVATION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def convert(
    manifest_path: str | Path, config_path: str | Path,
    evaluation_root: str | Path, output_path: str | Path,
    finalized_conjuration_v2_windows: str | Path | None = None,
) -> dict[str, Any]:
    panel = load_panel_manifest(manifest_path, verify_sources=False)
    config = load_aggregate_config(config_path)
    root = Path(evaluation_root).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    _require(root.is_dir(), f"evaluation root is not a directory: {root}")
    model_ids = {model.model_id for model in config.models}
    datasets = {context.context_id: context.dataset for context in panel.contexts}
    _, videos, scenes = _manifest_rows(root, datasets, model_ids)

    style = _selected(_read_csv(root, "style_windows.csv", [
        "scene", "model", "window_start_s", "window_end_s",
        "drift_from_real", "video_sha256",
    ]), model_ids, "style_windows.csv")
    control = _selected(_read_csv(root, "control_window_scored.csv", [
        "scene", "model", "window_start_s", "window_end_s",
        "control_failure", "control_rule_id", "video_sha256",
    ]), model_ids, "control_window_scored.csv")
    _require(
        set(control.control_rule_id.astype(str)) == {CONTROL_RULE_ID},
        "control_window_scored.csv: wrong or mixed control rule",
    )
    geometry = _selected(_read_csv(root, "geometry_windows.csv", [
        "scene", "model", "window_start_s", "window_end_s",
        "geometry_flag", "video_sha256",
    ]), model_ids, "geometry_windows.csv")
    if finalized_conjuration_v2_windows is None:
        conjuration_path = root / "conjuration_windows.csv"
        conjuration_name = "conjuration_windows.csv"
        conjuration_starts = PRODUCER_WINDOW_STARTS
        conjuration = _read_csv_path(conjuration_path, conjuration_name, [
            "scene", "model", "window_start_s", "window_end_s",
            "conjuration_flag", "video_sha256",
        ])
        conjuration_is_finalized_v2 = False
    else:
        conjuration_path = Path(
            finalized_conjuration_v2_windows).expanduser().resolve()
        conjuration_name = str(conjuration_path)
        conjuration_starts = FINALIZED_V2_WINDOW_STARTS
        conjuration = _read_csv_path(conjuration_path, conjuration_name, [
            "scene", "model", "window_start_s", "window_end_s",
            "conjuration_flag", "video_sha256", "instrument_profile",
        ])
        profiles = set(conjuration.instrument_profile.dropna().astype(str))
        _require(
            profiles == {"panel32-conjuration-v2-human-adjudicated"},
            f"{conjuration_name}: not a finalized human-adjudicated v2 table; "
            f"profiles={sorted(profiles)}",
        )
        conjuration_is_finalized_v2 = True
    conjuration = _selected(conjuration, model_ids, conjuration_name)
    hf = _selected(_read_csv(root, "hf_five_horizon_panel_rows.csv", [
        "scene", "model", "endpoint_s", "hf_failure",
    ]), model_ids, "hf_five_horizon_panel_rows.csv")
    hf_support = _selected(_read_csv(root, "hf_window_trajectory.csv", [
        "scene", "model", "window_start_s", "window_end_s", "video_sha256",
    ]), model_ids, "hf_window_trajectory.csv")
    relocation, relocation_flag, relocation_name = _relocation_table(
        root, model_ids, videos)

    for name, frame in (
        ("style_windows.csv", style),
        ("control_window_scored.csv", control),
        ("geometry_windows.csv", geometry),
        ("hf_window_trajectory.csv", hf_support),
    ):
        _check_window_table(frame, name, videos)
    _check_grid(
        conjuration, conjuration_name, videos,
        "window_start_s", conjuration_starts,
    )
    _strict_integer_column(conjuration, "window_end_s", conjuration_name)
    _require(
        (conjuration.window_end_s == conjuration.window_start_s + 6).all(),
        f"{conjuration_name}: every producer window must be exactly six seconds",
    )
    _check_grid(hf, "hf_five_horizon_panel_rows.csv", videos,
                "endpoint_s", HF_ENDPOINTS)
    # HF is candidate-relative but its frozen median must come from all six
    # family seats.  Ensure the producer tables contain a complete seat panel
    # even when this conversion reports only ODE models or DMD ablations.
    reference_videos = {
        (scene, model) for scene in scenes for model in REFERENCE_SEATS.values()
    }
    hf_all = _read_csv(root, "hf_five_horizon_panel_rows.csv", [
        "scene", "model", "endpoint_s", "hf_failure",
    ])
    hf_reference = hf_all[hf_all.model.astype(str).isin(REFERENCE_SEATS.values())].copy()
    _check_grid(hf_reference, "hf_five_horizon_panel_rows.csv[reference seats]",
                reference_videos, "endpoint_s", HF_ENDPOINTS)

    style["failed_value"] = (
        pd.to_numeric(style.drift_from_real, errors="coerce") > 0.72
    ).astype(int)
    _require(pd.to_numeric(style.drift_from_real, errors="coerce").notna().all(),
             "style_windows.csv.drift_from_real contains non-numeric values")
    if "style_flag_072" in style.columns:
        # The historical producer declares this legacy flag only for the
        # six-second anchor and leaves later rows blank.  Validate every value
        # it does declare, while deriving all five official decisions from the
        # complete numeric drift column above.
        mask = style.style_flag_072.notna()
        declared = _binary(
            style.loc[mask], "style_flag_072", "style_windows.csv")
        _require((declared == style.loc[mask, "failed_value"]).all(),
                 "style_windows.csv: style_flag_072 disagrees with drift > 0.72")
    control["failed_value"] = _binary(
        control, "control_failure", "control_window_scored.csv")
    geometry["failed_value"] = _binary(
        geometry, "geometry_flag", "geometry_windows.csv")
    conjuration["failed_value"] = _binary(
        conjuration, "conjuration_flag", conjuration_name)
    hf["failed_value"] = _binary(
        hf, "hf_failure", "hf_five_horizon_panel_rows.csv")
    relocation["failed_value"] = _binary(relocation, relocation_flag, relocation_name)

    hashes: dict[tuple[str, str], str] = {}
    for name, frame in (
        ("style_windows.csv", style),
        ("control_window_scored.csv", control),
        ("geometry_windows.csv", geometry),
        ("hf_window_trajectory.csv", hf_support),
    ):
        _merge_hashes(hashes, frame, name, videos)
    _merge_hashes(hashes, conjuration, conjuration_name, videos)
    if "video_sha256" in relocation.columns:
        _merge_hashes(hashes, relocation, relocation_name, videos)

    decisions: dict[tuple[str, str, str, int], int] = {}
    for metric, frame in (
        ("style", style), ("control", control), ("geometry", geometry),
    ):
        for row in frame[frame.window_start_s.isin(WINDOW_STARTS)].itertuples():
            decisions[(str(row.scene), str(row.model), metric,
                       int(row.window_start_s))] = int(row.failed_value)
    for row in hf.itertuples():
        start = int(row.endpoint_s) - 6
        decisions[(str(row.scene), str(row.model), "hf", start)] = int(row.failed_value)
    for row in conjuration[conjuration.window_start_s == 0].itertuples():
        decisions[(str(row.scene), str(row.model), "conjuration", 0)] = int(row.failed_value)
    for row in relocation.itertuples():
        decisions[(str(row.scene), str(row.model), "relocation", 0)] = int(row.failed_value)

    rows: list[dict[str, Any]] = []
    for scene in sorted(scenes):
        context_id, action = scenes[scene]
        for model in (spec.model_id for spec in config.models):
            digest = hashes[(scene, model)]
            for metric in ("control", "style", "geometry", "hf"):
                for start in WINDOW_STARTS:
                    key = (scene, model, metric, start)
                    _require(key in decisions, f"internal conversion gap: {key}")
                    rows.append(_observation(
                        panel_id=panel.panel_id, evaluation_id=config.evaluation_id,
                        context_id=context_id, dataset=datasets[context_id],
                        action=action, model=model, metric=metric,
                        semantics=WINDOW_SEMANTICS, start=start,
                        failed=decisions[key], digest=digest,
                    ))
            for metric in ("conjuration", "relocation"):
                key = (scene, model, metric, 0)
                _require(key in decisions, f"internal conversion gap: {key}")
                rows.append(_observation(
                    panel_id=panel.panel_id, evaluation_id=config.evaluation_id,
                    context_id=context_id, dataset=datasets[context_id],
                    action=action, model=model, metric=metric,
                    semantics=ONSET_SEMANTICS, start=0,
                    failed=decisions[key], digest=digest,
                ))

    expected_rows = len(videos) * (4 * len(WINDOW_STARTS) + 2)
    _require(len(rows) == expected_rows,
             f"expected {expected_rows} normalized rows, built {len(rows)}")
    _atomic_csv(output, rows)
    inputs = [
        "video_manifest.csv", "style_windows.csv", "control_window_scored.csv",
        "geometry_windows.csv",
        "hf_five_horizon_panel_rows.csv", "hf_window_trajectory.csv",
        relocation_name,
    ]
    provenance = {
        "schema_version": CONVERTER_SCHEMA_VERSION,
        "status": "pass",
        "panel_id": panel.panel_id,
        "panel_manifest": str(panel.path),
        "panel_manifest_sha256": panel.sha256,
        "evaluation_id": config.evaluation_id,
        "aggregation_config": str(config.path),
        "aggregation_config_sha256": config.sha256,
        "evaluation_root": str(root),
        "models": [model.model_id for model in config.models],
        "videos": len(videos),
        "normalized_rows": len(rows),
        "window_starts_s": list(WINDOW_STARTS),
        "discarded_overlapping_window_start_s": 9,
        "onset_window_s": [0, 6],
        "family_seat_count": FAMILY_SEAT_COUNT,
        "reference_seats": REFERENCE_SEATS,
        "control_rule_id": CONTROL_RULE_ID,
        "inputs": {name: sha256_file(root / name) for name in inputs},
        "conjuration_windows": {
            "path": str(conjuration_path),
            "sha256": sha256_file(conjuration_path),
            "finalized_v2": conjuration_is_finalized_v2,
            "window_starts_s": list(conjuration_starts),
        },
        "output": str(output),
        "output_sha256": sha256_file(output),
    }
    provenance_path = output.with_suffix(output.suffix + ".provenance.json")
    _atomic_json(provenance_path, provenance)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evaluation-out", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--finalized-conjuration-v2-windows", type=Path,
        help=("finalized human-adjudicated v2 window table; when supplied, "
              "the converter requires and uses it instead of the legacy table"),
    )
    args = parser.parse_args()
    result = convert(
        args.manifest, args.config, args.evaluation_out, args.output,
        args.finalized_conjuration_v2_windows,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
