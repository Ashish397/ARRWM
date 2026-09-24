"""Produce a fail-closed paper-results report from panel32 observations.

The converter outputs are the sole measurement input.  This script does not
read producer caches, rerun metrics, or alter an evaluation directory.  It
validates all three normalized grids with :mod:`panel32_aggregate`, verifies
that observations shared by two reporting groups are identical, and writes
an explicit JSON/Markdown handoff.

Reported rates retain integer numerators and denominators.  Both pooled and
equal-dataset-weighted estimates are included even though they coincide on
the locked, balanced 8+8+8+8 context panel.  McNemar tests are paired by
``(context_id, action)`` and use the exact two-sided binomial test.  Holm
correction is applied separately within each declared outcome family across
all six DMD contrasts and all applicable endpoints.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from .panel32_aggregate import (
        ALL_METRICS,
        ONSET_METRICS,
        WINDOW_DURATION_S,
        WINDOW_METRICS,
        WINDOW_STARTS,
        AggregateConfig,
        AggregationError,
        ModelSpec,
        Observation,
        load_aggregate_config,
        load_observations,
    )
    from .panel32_manifest import (
        ACTIONS,
        DATASET_COUNTS,
        PanelManifest,
        load_panel_manifest,
    )
except ImportError:  # Direct execution from grids/eval.
    from panel32_aggregate import (  # type: ignore
        ALL_METRICS,
        ONSET_METRICS,
        WINDOW_DURATION_S,
        WINDOW_METRICS,
        WINDOW_STARTS,
        AggregateConfig,
        AggregationError,
        ModelSpec,
        Observation,
        load_aggregate_config,
        load_observations,
    )
    from panel32_manifest import (  # type: ignore
        ACTIONS,
        DATASET_COUNTS,
        PanelManifest,
        load_panel_manifest,
    )


REPORT_SCHEMA_VERSION = 3
GROUPS = ("main", "ode", "dmd")
ENDPOINTS = tuple(start + WINDOW_DURATION_S for start in WINDOW_STARTS)
CLEAN_METRICS = (
    "control", "style", "geometry", "hf", "conjuration", "relocation"
)
DEFAULT_DMD_BASELINE = "ours_no_gan"


class ReportError(ValueError):
    """The report inputs are incomplete, inconsistent, or ambiguous."""


@dataclass(frozen=True)
class GroupInput:
    name: str
    config: AggregateConfig
    observations_path: Path
    observations_sha256: str
    observations: tuple[Observation, ...]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fraction_payload(value: Fraction) -> dict[str, Any]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "fraction": f"{value.numerator}/{value.denominator}",
        "rate_pct": float(value * 100),
    }


def _binary_summary(
    samples: Iterable[tuple[str, int]], *, expected_per_dataset: int,
    positive_name: str, where: str,
) -> dict[str, Any]:
    by_dataset: dict[str, list[int]] = {dataset: [] for dataset in DATASET_COUNTS}
    for dataset, value in samples:
        _require(dataset in by_dataset, f"{where}: unknown dataset {dataset!r}")
        _require(value in (0, 1), f"{where}: non-binary value {value!r}")
        by_dataset[dataset].append(value)

    dataset_payload: dict[str, Any] = {}
    fractions: list[Fraction] = []
    pooled_positive = 0
    pooled_scored = 0
    for dataset in DATASET_COUNTS:
        values = by_dataset[dataset]
        _require(
            len(values) == expected_per_dataset,
            f"{where}/{dataset}: expected {expected_per_dataset} paired values, "
            f"found {len(values)}",
        )
        positive = sum(values)
        fraction = Fraction(positive, len(values))
        fractions.append(fraction)
        pooled_positive += positive
        pooled_scored += len(values)
        dataset_payload[dataset] = {
            positive_name: positive,
            "scored": len(values),
            **_fraction_payload(fraction),
        }

    expected_total = expected_per_dataset * len(DATASET_COUNTS)
    _require(
        pooled_scored == expected_total,
        f"{where}: expected denominator {expected_total}, found {pooled_scored}",
    )
    pooled = Fraction(pooled_positive, pooled_scored)
    macro = sum(fractions, Fraction()) / len(fractions)
    # The locked panel is balanced.  A mismatch indicates either a denominator
    # error or a future panel whose estimand must be reconsidered explicitly.
    _require(
        pooled == macro,
        f"{where}: balanced-panel invariant failed: pooled={pooled}, macro={macro}",
    )
    return {
        "positive_name": positive_name,
        "pooled": {
            positive_name: pooled_positive,
            "scored": pooled_scored,
            **_fraction_payload(pooled),
        },
        "equal_dataset_weighted": {
            "dataset_count": len(DATASET_COUNTS),
            **_fraction_payload(macro),
        },
        "by_dataset": dataset_payload,
    }


def _load_group(
    name: str, panel: PanelManifest, config_path: str | Path,
    observations_path: str | Path,
) -> GroupInput:
    config = load_aggregate_config(config_path)
    path = Path(observations_path).expanduser().resolve()
    observations, digest = load_observations(path, panel, config)
    return GroupInput(
        name=name,
        config=config,
        observations_path=path,
        observations_sha256=digest,
        observations=tuple(observations),
    )


def _validate_group_contracts(groups: Mapping[str, GroupInput]) -> None:
    _require(set(groups) == set(GROUPS), f"groups must be exactly {list(GROUPS)}")
    main, ode, dmd = (groups[name] for name in GROUPS)
    _require(
        len(main.config.models) == 6
        and len({model.family for model in main.config.models}) == 6,
        "main report must contain exactly six distinct model families",
    )
    _require(
        sum(model.role == "external" for model in main.config.models) == 5,
        "main report must contain exactly five external models",
    )
    _require(len(ode.config.models) == 3, "ODE report must contain exactly three models")
    _require(len(dmd.config.models) == 5, "DMD report must contain exactly five models")
    _require(
        DEFAULT_DMD_BASELINE in {model.model_id for model in dmd.config.models},
        f"DMD report is missing baseline {DEFAULT_DMD_BASELINE}",
    )

    # The complete model appears in both main and DMD outputs.  More generally,
    # every overlap must be byte-for-byte identical at the normalized decision
    # level; otherwise a paper table could silently mix different evaluations.
    seen: dict[tuple[str, str, str, str, int], tuple[int, str, str]] = {}
    for group_name in GROUPS:
        for row in groups[group_name].observations:
            key = (row.context_id, row.action, row.model_id,
                   row.metric, row.window_start_s)
            value = (row.failed, row.video_sha256, group_name)
            if key in seen:
                old_failed, old_hash, old_group = seen[key]
                _require(
                    (row.failed, row.video_sha256) == (old_failed, old_hash),
                    f"shared observation differs between {old_group} and {group_name}: {key}",
                )
            else:
                seen[key] = value


def _population_rows(
    rows: Sequence[Observation], metric: str,
) -> tuple[tuple[str, list[Observation], int], ...]:
    if metric == "control":
        return (
            ("all_actions", list(rows), 72),
            ("directional", [row for row in rows if row.action != "N"], 64),
            ("noop", [row for row in rows if row.action == "N"], 8),
        )
    return (("all_actions", list(rows), 72),)


def rate_rows(group: GroupInput) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, str, int], list[Observation]] = defaultdict(list)
    for row in group.observations:
        indexed[(row.model_id, row.metric, row.window_start_s)].append(row)
    output: list[dict[str, Any]] = []
    for model in group.config.models:
        for metric in WINDOW_METRICS:
            for start in WINDOW_STARTS:
                candidates = indexed[(model.model_id, metric, start)]
                for population, selected, expected_per_dataset in _population_rows(
                    candidates, metric
                ):
                    summary = _binary_summary(
                        ((row.dataset, row.failed) for row in selected),
                        expected_per_dataset=expected_per_dataset,
                        positive_name="flagged",
                        where=(f"{group.name}/{model.model_id}/{metric}/"
                               f"{start}/{population}"),
                    )
                    output.append({
                        "group": group.name,
                        "model_id": model.model_id,
                        "model_label": model.label,
                        "family": model.family,
                        "role": model.role,
                        "metric": metric,
                        "semantics": "independent_six_second_window",
                        "window_start_s": start,
                        "window_end_s": start + WINDOW_DURATION_S,
                        "endpoint_s": start + WINDOW_DURATION_S,
                        "population": population,
                        **summary,
                    })
        for metric in ONSET_METRICS:
            candidates = indexed[(model.model_id, metric, 0)]
            summary = _binary_summary(
                ((row.dataset, row.failed) for row in candidates),
                expected_per_dataset=72,
                positive_name="flagged",
                where=f"{group.name}/{model.model_id}/{metric}/onset",
            )
            output.append({
                "group": group.name,
                "model_id": model.model_id,
                "model_label": model.label,
                "family": model.family,
                "role": model.role,
                "metric": metric,
                "semantics": "six_second_onset_event",
                "window_start_s": 0,
                "window_end_s": WINDOW_DURATION_S,
                "endpoint_s": WINDOW_DURATION_S,
                "population": "all_actions",
                **summary,
            })
    return output


def _outcome_index(group: GroupInput) -> dict[tuple[str, str, str, str, int], int]:
    return {
        (row.context_id, row.action, row.model_id, row.metric,
         row.window_start_s): row.failed
        for row in group.observations
    }


def _load_adjudicated_events(
    path: str | Path,
    panel: PanelManifest,
    groups: Mapping[str, GroupInput],
    *, event_name: str, flag_column: str,
) -> tuple[dict[str, dict[tuple[str, str], float]], str]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"long-{event_name} adjudication missing: {resolved}")
    contexts = {context.context_id for context in panel.contexts}
    models = {
        model.model_id
        for group in groups.values()
        for model in group.config.models
    }
    events: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
    seen_rows: set[tuple[str, str, float]] = set()
    with resolved.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"scene", "model", "time_s", flag_column}
        _require(
            reader.fieldnames is not None
            and required.issubset(reader.fieldnames),
            f"long-{event_name} adjudication missing columns {sorted(required)}",
        )
        for line_number, row in enumerate(reader, 2):
            try:
                flag = int(row[flag_column])
            except (TypeError, ValueError) as exc:
                raise ReportError(
                    f"long-{event_name} row {line_number}: non-binary "
                    f"{flag_column}={row.get(flag_column)!r}"
                ) from exc
            _require(
                flag in (0, 1),
                f"long-{event_name} row {line_number}: {flag_column} must be 0/1",
            )
            scene = str(row.get("scene", ""))
            _require(
                "_" in scene,
                f"long-{event_name} row {line_number}: malformed scene {scene!r}",
            )
            context_id, action = scene.rsplit("_", 1)
            model_id = str(row.get("model", ""))
            try:
                time_s = float(row["time_s"])
            except (TypeError, ValueError) as exc:
                raise ReportError(
                    f"long-{event_name} row {line_number}: invalid time_s"
                ) from exc
            _require(math.isfinite(time_s),
                     f"long-{event_name} row {line_number}: non-finite time_s")
            _require(context_id in contexts,
                     f"unknown {event_name} context {context_id}")
            _require(action in ACTIONS, f"unknown {event_name} action {action}")
            _require(0.0 <= time_s <= max(ENDPOINTS),
                     f"invalid {event_name} time {time_s}")
            # The adjudication files cover the complete 15-row evaluation
            # fleet, while the paper report deliberately exposes only the
            # models selected by the main/ODE/DMD configs.  Retain the full
            # audit files and ignore rows for unreported ablations here.
            if model_id not in models:
                continue
            row_key = (scene, model_id, time_s)
            _require(
                row_key not in seen_rows,
                f"long-{event_name} adjudication duplicates {row_key}",
            )
            seen_rows.add(row_key)
            if flag != 1:
                continue
            key = (context_id, action)
            previous = events[model_id].get(key)
            events[model_id][key] = time_s if previous is None else min(previous, time_s)
    return dict(events), _sha256(resolved)


def _cumulative_failure_vectors(
    group: GroupInput,
    long_relocation_events: Mapping[str, Mapping[tuple[str, str], float]],
    long_conjuration_events: Mapping[str, Mapping[tuple[str, str], float]],
) -> dict[tuple[str, int], dict[tuple[str, str], int]]:
    index = _outcome_index(group)
    output: dict[tuple[str, int], dict[tuple[str, str], int]] = {}
    for model in group.config.models:
        for endpoint in ENDPOINTS:
            last_start = endpoint - WINDOW_DURATION_S
            vectors: dict[tuple[str, str], int] = {}
            for context_id in {
                row.context_id for row in group.observations
                if row.model_id == model.model_id
            }:
                for action in ACTIONS:
                    failures = [
                        index[(context_id, action, model.model_id,
                               metric, 0)]
                        for metric in ONSET_METRICS
                    ]
                    for metric in WINDOW_METRICS:
                        failures.extend(
                            index[(context_id, action, model.model_id, metric, start)]
                            for start in WINDOW_STARTS if start <= last_start
                        )
                    expected_components = (
                        len(ONSET_METRICS)
                        + len(WINDOW_METRICS) * (endpoint // 6)
                    )
                    _require(
                        len(failures) == expected_components,
                        f"{group.name}/{model.model_id}/{context_id}/{action}/"
                        f"{endpoint}: expected {expected_components} clean components",
                    )
                    later_relocation = (
                        long_relocation_events
                        .get(model.model_id, {})
                        .get((context_id, action), math.inf)
                        <= endpoint
                    )
                    later_conjuration = (
                        long_conjuration_events
                        .get(model.model_id, {})
                        .get((context_id, action), math.inf)
                        <= endpoint
                    )
                    vectors[(context_id, action)] = int(
                        any(failures) or later_relocation or later_conjuration
                    )
            _require(
                len(vectors) == sum(DATASET_COUNTS.values()) * len(ACTIONS),
                f"{group.name}/{model.model_id}/{endpoint}: incomplete cumulative vector",
            )
            output[(model.model_id, endpoint)] = vectors
    return output


def cumulative_clean_rows(
    group: GroupInput,
    panel: PanelManifest,
    long_relocation_events: Mapping[str, Mapping[tuple[str, str], float]],
    long_conjuration_events: Mapping[str, Mapping[tuple[str, str], float]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[tuple[str, str], int]]]:
    context_dataset = {context.context_id: context.dataset for context in panel.contexts}
    failures = _cumulative_failure_vectors(
        group, long_relocation_events, long_conjuration_events)
    output: list[dict[str, Any]] = []
    for model in group.config.models:
        for endpoint in ENDPOINTS:
            vector = failures[(model.model_id, endpoint)]
            summary = _binary_summary(
                ((context_dataset[context_id], 1 - failed)
                 for (context_id, _action), failed in vector.items()),
                expected_per_dataset=72,
                positive_name="clean",
                where=f"{group.name}/{model.model_id}/cumulative-clean/{endpoint}",
            )
            output.append({
                "group": group.name,
                "model_id": model.model_id,
                "model_label": model.label,
                "family": model.family,
                "role": model.role,
                "endpoint_s": endpoint,
                "population": "all_actions",
                "clean_definition": {
                    "metrics": list(CLEAN_METRICS),
                    "window_metrics_accumulated_through_endpoint": True,
                    "conjuration_onset_included": True,
                    "relocation_onset_included": True,
                    "adjudicated_later_conjuration_included": True,
                    "adjudicated_later_relocation_included": True,
                },
                **summary,
            })
    return output, failures


def _mean_fraction(values: Sequence[Fraction]) -> Fraction:
    _require(bool(values), "cannot average an empty set")
    return sum(values, Fraction()) / len(values)


def external_average_rows(
    rows: Sequence[dict[str, Any]], *, value_name: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["group"] != "main" or row["role"] != "external":
            continue
        if value_name == "flagged":
            signature = (
                row["metric"], row["semantics"], row["endpoint_s"],
                row["population"],
            )
        else:
            signature = ("cumulative_clean", row["endpoint_s"], row["population"])
        grouped[signature].append(row)

    output: list[dict[str, Any]] = []
    for signature, candidates in grouped.items():
        _require(len(candidates) == 5,
                 f"main external average {signature}: expected five models")
        pooled_positive = sum(row["pooled"][value_name] for row in candidates)
        pooled_scored = sum(row["pooled"]["scored"] for row in candidates)
        pooled = Fraction(pooled_positive, pooled_scored)
        model_rates = [
            Fraction(row["pooled"][value_name], row["pooled"]["scored"])
            for row in candidates
        ]
        macro_rates = [
            Fraction(
                row["equal_dataset_weighted"]["numerator"],
                row["equal_dataset_weighted"]["denominator"],
            )
            for row in candidates
        ]
        unweighted_models = _mean_fraction(model_rates)
        equal_dataset_and_model = _mean_fraction(macro_rates)
        _require(
            pooled == unweighted_models == equal_dataset_and_model,
            f"main external average {signature}: balanced denominator invariant failed",
        )
        output.append({
            "group": "main",
            "outcome": value_name,
            "signature": list(signature),
            "model_count": 5,
            "model_ids": [row["model_id"] for row in candidates],
            "pooled_across_external_videos": {
                value_name: pooled_positive,
                "scored": pooled_scored,
                **_fraction_payload(pooled),
            },
            "unweighted_mean_of_model_rates": _fraction_payload(unweighted_models),
            "equal_dataset_and_model_weighted": _fraction_payload(
                equal_dataset_and_model),
        })
    return output


def exact_mcnemar_p(baseline_only: int, comparison_only: int) -> Fraction:
    """Return the exact two-sided McNemar p-value as a rational number."""
    _require(baseline_only >= 0 and comparison_only >= 0,
             "McNemar discordant counts must be non-negative")
    discordant = baseline_only + comparison_only
    if discordant == 0:
        return Fraction(1, 1)
    tail_limit = min(baseline_only, comparison_only)
    tail_count = sum(math.comb(discordant, k) for k in range(tail_limit + 1))
    value = Fraction(2 * tail_count, 2 ** discordant)
    return min(Fraction(1, 1), value)


def _mcnemar_row(
    *, baseline: ModelSpec, comparison: ModelSpec, outcome: str,
    metric: str, semantics: str, endpoint: int, holm_family: str,
    baseline_vector: Mapping[tuple[str, str], int],
    comparison_vector: Mapping[tuple[str, str], int],
) -> dict[str, Any]:
    _require(
        set(baseline_vector) == set(comparison_vector),
        f"DMD pairing differs for {comparison.model_id}/{outcome}/{endpoint}",
    )
    keys = sorted(baseline_vector)
    _require(len(keys) == 288,
             f"DMD McNemar denominator must be 288, found {len(keys)}")
    both_pass = baseline_only = comparison_only = both_fail = 0
    for key in keys:
        a = baseline_vector[key]
        b = comparison_vector[key]
        _require(a in (0, 1) and b in (0, 1), "non-binary McNemar outcome")
        if a == 0 and b == 0:
            both_pass += 1
        elif a == 1 and b == 0:
            baseline_only += 1
        elif a == 0 and b == 1:
            comparison_only += 1
        else:
            both_fail += 1
    p_value = exact_mcnemar_p(baseline_only, comparison_only)
    baseline_failed = baseline_only + both_fail
    comparison_failed = comparison_only + both_fail
    return {
        "group": "dmd",
        "baseline_model_id": baseline.model_id,
        "baseline_label": baseline.label,
        "comparison_model_id": comparison.model_id,
        "comparison_label": comparison.label,
        "outcome": outcome,
        "metric": metric,
        "semantics": semantics,
        "endpoint_s": endpoint,
        "population": "all_actions",
        "paired_n": len(keys),
        "both_pass": both_pass,
        "baseline_only_failure": baseline_only,
        "comparison_only_failure": comparison_only,
        "both_fail": both_fail,
        "discordant": baseline_only + comparison_only,
        "baseline_failure": {
            "flagged": baseline_failed, "scored": len(keys),
            **_fraction_payload(Fraction(baseline_failed, len(keys))),
        },
        "comparison_failure": {
            "flagged": comparison_failed, "scored": len(keys),
            **_fraction_payload(Fraction(comparison_failed, len(keys))),
        },
        "difference_comparison_minus_baseline_pct_points": (
            100.0 * (comparison_failed - baseline_failed) / len(keys)
        ),
        "test": "exact_two_sided_mcnemar_binomial",
        "p_value_exact_fraction": (
            f"{p_value.numerator}/{p_value.denominator}"
        ),
        "p_value": float(p_value),
        "holm_family": holm_family,
        "_p_fraction": p_value,
    }


def _apply_holm(rows: list[dict[str, Any]], alpha: Fraction = Fraction(1, 20)) -> None:
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        families[row["holm_family"]].append(row)
    for family, candidates in families.items():
        ordered = sorted(
            candidates,
            key=lambda row: (
                row["_p_fraction"], row["comparison_model_id"], row["endpoint_s"]
            ),
        )
        count = len(ordered)
        running = Fraction(0, 1)
        for rank, row in enumerate(ordered, start=1):
            adjusted = min(Fraction(1, 1), (count - rank + 1) * row["_p_fraction"])
            running = max(running, adjusted)
            row["holm_family_size"] = count
            row["holm_rank"] = rank
            row["p_value_holm_exact_fraction"] = (
                f"{running.numerator}/{running.denominator}"
            )
            row["p_value_holm"] = float(running)
            row["reject_holm_0_05"] = running <= alpha
        _require(all(row["holm_family"] == family for row in ordered),
                 "internal Holm-family error")
    for row in rows:
        row.pop("_p_fraction", None)


def dmd_mcnemar_rows(
    group: GroupInput,
    cumulative_failures: Mapping[
        tuple[str, int], Mapping[tuple[str, str], int]
    ],
    baseline_id: str = DEFAULT_DMD_BASELINE,
) -> list[dict[str, Any]]:
    models = {model.model_id: model for model in group.config.models}
    _require(baseline_id in models, f"DMD baseline {baseline_id!r} is absent")
    baseline = models[baseline_id]
    comparisons = [model for model in group.config.models if model.model_id != baseline_id]
    _require(len(comparisons) == 4, "DMD report requires four baseline--ablation contrasts")
    observations = {
        (row.model_id, row.metric, row.window_start_s, row.context_id, row.action): row.failed
        for row in group.observations
    }

    def vector(model_id: str, metric: str, start: int) -> dict[tuple[str, str], int]:
        result = {
            (context_id, action): value
            for (candidate, candidate_metric, candidate_start,
                 context_id, action), value in observations.items()
            if (candidate, candidate_metric, candidate_start)
            == (model_id, metric, start)
        }
        _require(len(result) == 288,
                 f"DMD vector {model_id}/{metric}/{start} has {len(result)} rows")
        return result

    output: list[dict[str, Any]] = []
    for comparison in comparisons:
        for metric in WINDOW_METRICS:
            for start in WINDOW_STARTS:
                endpoint = start + WINDOW_DURATION_S
                output.append(_mcnemar_row(
                    baseline=baseline, comparison=comparison,
                    outcome="window_failure", metric=metric,
                    semantics="independent_six_second_window",
                    endpoint=endpoint, holm_family=f"window_failure:{metric}",
                    baseline_vector=vector(baseline_id, metric, start),
                    comparison_vector=vector(comparison.model_id, metric, start),
                ))
        for metric in ONSET_METRICS:
            output.append(_mcnemar_row(
                baseline=baseline, comparison=comparison,
                outcome="onset_failure", metric=metric,
                semantics="six_second_onset_event",
                endpoint=WINDOW_DURATION_S,
                holm_family=f"onset_failure:{metric}",
                baseline_vector=vector(baseline_id, metric, 0),
                comparison_vector=vector(comparison.model_id, metric, 0),
            ))
        for endpoint in ENDPOINTS:
            output.append(_mcnemar_row(
                baseline=baseline, comparison=comparison,
                outcome="cumulative_any_failure", metric="joint_clean_complement",
                semantics="cumulative_through_endpoint",
                endpoint=endpoint, holm_family="cumulative_any_failure",
                baseline_vector=cumulative_failures[(baseline_id, endpoint)],
                comparison_vector=cumulative_failures[(comparison.model_id, endpoint)],
            ))
    _apply_holm(output)
    return output


def build_report(
    *, manifest_path: str | Path,
    main_config: str | Path, main_observations: str | Path,
    ode_config: str | Path, ode_observations: str | Path,
    dmd_config: str | Path, dmd_observations: str | Path,
    long_relocation_adjudication: str | Path,
    long_conjuration_adjudication: str | Path,
    dmd_baseline: str = DEFAULT_DMD_BASELINE,
) -> dict[str, Any]:
    try:
        panel = load_panel_manifest(manifest_path, verify_sources=False)
        groups = {
            "main": _load_group("main", panel, main_config, main_observations),
            "ode": _load_group("ode", panel, ode_config, ode_observations),
            "dmd": _load_group("dmd", panel, dmd_config, dmd_observations),
        }
    except (AggregationError, OSError, ValueError) as exc:
        if isinstance(exc, ReportError):
            raise
        raise ReportError(str(exc)) from exc
    _validate_group_contracts(groups)
    long_relocation_events, long_relocation_sha256 = (
        _load_adjudicated_events(
            long_relocation_adjudication, panel, groups,
            event_name="relocation", flag_column="relocation_flag",
        )
    )
    long_conjuration_events, long_conjuration_sha256 = (
        _load_adjudicated_events(
            long_conjuration_adjudication, panel, groups,
            event_name="conjuration", flag_column="conjuration_flag",
        )
    )

    rates: list[dict[str, Any]] = []
    cumulative: list[dict[str, Any]] = []
    cumulative_vectors: dict[
        str, dict[tuple[str, int], dict[tuple[str, str], int]]
    ] = {}
    for name in GROUPS:
        rates.extend(rate_rows(groups[name]))
        clean_rows, vectors = cumulative_clean_rows(
            groups[name], panel, long_relocation_events,
            long_conjuration_events,
        )
        cumulative.extend(clean_rows)
        cumulative_vectors[name] = vectors

    mcnemar = dmd_mcnemar_rows(
        groups["dmd"], cumulative_vectors["dmd"], dmd_baseline)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "pass",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "panel": {
            "panel_id": panel.panel_id,
            "manifest": str(panel.path),
            "manifest_sha256": panel.sha256,
            "contexts": len(panel.contexts),
            "dataset_counts": dict(DATASET_COUNTS),
            "actions": list(ACTIONS),
        },
        "long_relocation_adjudication": {
            "path": str(Path(long_relocation_adjudication).expanduser().resolve()),
            "sha256": long_relocation_sha256,
            "accepted_events": sum(
                len(events) for events in long_relocation_events.values()
            ),
        },
        "long_conjuration_adjudication": {
            "path": str(Path(long_conjuration_adjudication).expanduser().resolve()),
            "sha256": long_conjuration_sha256,
            "accepted_events": sum(
                len(events) for events in long_conjuration_events.values()
            ),
        },
        "denominator_contract": {
            "per_dataset_all_actions": 72,
            "per_dataset_directional": 64,
            "per_dataset_noop": 8,
            "pooled_all_actions": 288,
            "pooled_directional": 256,
            "pooled_noop": 32,
            "datasets": 4,
            "main_external_models": 5,
            "dmd_pairs_per_comparison": 288,
        },
        "groups": {
            name: {
                "evaluation_id": group.config.evaluation_id,
                "config": str(group.config.path),
                "config_sha256": group.config.sha256,
                "observations": str(group.observations_path),
                "observations_sha256": group.observations_sha256,
                "observation_rows": len(group.observations),
                "models": [model.__dict__ for model in group.config.models],
            }
            for name, group in groups.items()
        },
        "rates": rates,
        "onset_tables": [row for row in rates if row["metric"] in ONSET_METRICS],
        "cumulative_clean": cumulative,
        "main_external_rate_averages": external_average_rows(
            rates, value_name="flagged"),
        "main_external_cumulative_clean_averages": external_average_rows(
            cumulative, value_name="clean"),
        "dmd_paired_mcnemar_holm": {
            "baseline_model_id": dmd_baseline,
            "pairing_unit": "context_id,action",
            "test": "exact_two_sided_mcnemar_binomial",
            "holm_policy": (
                "separate family per outcome: each window metric across six "
                "contrasts and five endpoints; each onset metric across six "
                "contrasts; cumulative-any-failure across six contrasts and "
                "five endpoints"
            ),
            "alpha": 0.05,
            "comparisons": mcnemar,
        },
    }


def _rate_cell(row: dict[str, Any], value_name: str) -> str:
    pooled = row["pooled"]
    macro = row["equal_dataset_weighted"]
    return (
        f"{pooled[value_name]}/{pooled['scored']} "
        f"({pooled['rate_pct']:.2f}%; EW {macro['rate_pct']:.2f}%)"
    )


def _markdown_rate_tables(lines: list[str], report: Mapping[str, Any], group: str) -> None:
    models = report["groups"][group]["models"]
    rates = [row for row in report["rates"] if row["group"] == group]
    lines.extend([f"## {group.upper()} rates", ""])
    for metric in WINDOW_METRICS:
        populations = ("all_actions", "directional", "noop") if metric == "control" else ("all_actions",)
        for population in populations:
            lines.extend([
                f"### {metric} — {population}", "",
                "| Model | 6 s | 12 s | 18 s | 24 s | 30 s |",
                "|---|---:|---:|---:|---:|---:|",
            ])
            for model in models:
                candidates = {
                    row["endpoint_s"]: row for row in rates
                    if row["model_id"] == model["model_id"]
                    and row["metric"] == metric
                    and row["population"] == population
                }
                _require(set(candidates) == set(ENDPOINTS),
                         f"Markdown missing {group}/{model['model_id']}/{metric}/{population}")
                cells = [_rate_cell(candidates[endpoint], "flagged") for endpoint in ENDPOINTS]
                lines.append(f"| {model['label']} | " + " | ".join(cells) + " |")
            lines.append("")

    lines.extend([
        "### Six-second onset events", "",
        "| Model | Conjuration | Relocation |",
        "|---|---:|---:|",
    ])
    for model in models:
        cells = {}
        for metric in ONSET_METRICS:
            candidates = [
                row for row in rates
                if row["model_id"] == model["model_id"]
                and row["metric"] == metric
            ]
            _require(len(candidates) == 1,
                     f"Markdown onset missing {group}/{model['model_id']}/{metric}")
            cells[metric] = _rate_cell(candidates[0], "flagged")
        lines.append(
            f"| {model['label']} | {cells['conjuration']} | {cells['relocation']} |"
        )
    lines.append("")

    lines.extend([
        "### Cumulative clean fraction", "",
        ("Control, style, and geometry/HF failures accumulate through the "
         "endpoint; conjuration and relocation persist after onset. All six "
         "axes are included."),
        "",
        "| Model | 6 s | 12 s | 18 s | 24 s | 30 s |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    clean = [row for row in report["cumulative_clean"] if row["group"] == group]
    for model in models:
        candidates = {
            row["endpoint_s"]: row for row in clean
            if row["model_id"] == model["model_id"]
        }
        _require(set(candidates) == set(ENDPOINTS),
                 f"Markdown cumulative rows missing for {group}/{model['model_id']}")
        cells = [_rate_cell(candidates[endpoint], "clean") for endpoint in ENDPOINTS]
        lines.append(f"| {model['label']} | " + " | ".join(cells) + " |")
    lines.append("")


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Panel32 final results", "",
        f"Panel: `{report['panel']['panel_id']}`. Status: **{report['status']}**.",
        "",
        ("Each cell is `numerator/denominator (pooled %; EW equal-dataset %)`; "
         "the two estimates coincide because every dataset contributes eight contexts."),
        "",
    ]
    for group in GROUPS:
        _markdown_rate_tables(lines, report, group)

    lines.extend([
        "## Main external-model averages", "",
        "The averages contain exactly five external models and exclude Ours.", "",
        "| Outcome | Endpoint | Population | Pooled external rate | Mean model rate |",
        "|---|---:|---|---:|---:|",
    ])
    for row in report["main_external_rate_averages"]:
        metric, _semantics, endpoint, population = row["signature"]
        pooled = row["pooled_across_external_videos"]
        mean = row["unweighted_mean_of_model_rates"]
        lines.append(
            f"| {metric} | {endpoint} s | {population} | "
            f"{pooled['flagged']}/{pooled['scored']} ({pooled['rate_pct']:.2f}%) | "
            f"{mean['rate_pct']:.2f}% |"
        )
    for row in report["main_external_cumulative_clean_averages"]:
        _name, endpoint, population = row["signature"]
        pooled = row["pooled_across_external_videos"]
        mean = row["unweighted_mean_of_model_rates"]
        lines.append(
            f"| cumulative clean | {endpoint} s | {population} | "
            f"{pooled['clean']}/{pooled['scored']} ({pooled['rate_pct']:.2f}%) | "
            f"{mean['rate_pct']:.2f}% |"
        )
    lines.append("")

    comparisons = report["dmd_paired_mcnemar_holm"]["comparisons"]
    displayed = [
        row for row in comparisons
        if row["reject_holm_0_05"] or row["p_value"] < 0.05
    ]
    lines.extend([
        "## DMD paired exact McNemar tests", "",
        (f"{len(comparisons)} tests are recorded in JSON; the table shows the "
         f"{len(displayed)} rows with raw p < 0.05 or Holm-adjusted rejection."),
        "",
        "| Ablation | Outcome | Endpoint | Base-only | Ablation-only | Raw p | Holm p | Reject |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in displayed:
        outcome = row["metric"] if row["outcome"] != "cumulative_any_failure" else "cumulative any failure"
        lines.append(
            f"| {row['comparison_label']} | {outcome} | {row['endpoint_s']} s | "
            f"{row['baseline_only_failure']} | {row['comparison_only_failure']} | "
            f"{row['p_value']:.6g} | {row['p_value_holm']:.6g} | "
            f"{'yes' if row['reject_holm_0_05'] else 'no'} |"
        )
    lines.append("")
    return "\n".join(lines)


def _atomic_write(path: Path, content: str) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_report(report: Mapping[str, Any], json_path: Path, markdown_path: Path) -> None:
    _require(json_path.expanduser().resolve() != markdown_path.expanduser().resolve(),
             "JSON and Markdown outputs must be different files")
    _atomic_write(json_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    _atomic_write(markdown_path, render_markdown(report))


def main() -> None:
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--main-config", type=Path,
                        default=directory / "panel32_aggregate_main.json")
    parser.add_argument("--main-observations", type=Path, required=True)
    parser.add_argument("--ode-config", type=Path,
                        default=directory / "panel32_aggregate_ode.json")
    parser.add_argument("--ode-observations", type=Path, required=True)
    parser.add_argument("--dmd-config", type=Path,
                        default=directory / "panel32_aggregate_dmd.json")
    parser.add_argument("--dmd-observations", type=Path, required=True)
    parser.add_argument(
        "--long-relocation-adjudication", type=Path, required=True,
    )
    parser.add_argument(
        "--long-conjuration-adjudication", type=Path, required=True,
    )
    parser.add_argument("--dmd-baseline", default=DEFAULT_DMD_BASELINE)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        manifest_path=args.manifest,
        main_config=args.main_config,
        main_observations=args.main_observations,
        ode_config=args.ode_config,
        ode_observations=args.ode_observations,
        dmd_config=args.dmd_config,
        dmd_observations=args.dmd_observations,
        long_relocation_adjudication=args.long_relocation_adjudication,
        long_conjuration_adjudication=args.long_conjuration_adjudication,
        dmd_baseline=args.dmd_baseline,
    )
    write_report(report, args.output_json, args.output_markdown)
    print(json.dumps({
        "status": "pass",
        "json": str(args.output_json.expanduser().resolve()),
        "markdown": str(args.output_markdown.expanduser().resolve()),
        "rate_rows": len(report["rates"]),
        "cumulative_clean_rows": len(report["cumulative_clean"]),
        "mcnemar_tests": len(report["dmd_paired_mcnemar_holm"]["comparisons"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
