"""Focused tests for the final panel32 paper reporting layer."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grids.eval.panel32_aggregate import (  # noqa: E402
    OBSERVATION_COLUMNS,
    ONSET_METRICS,
    ONSET_SEMANTICS,
    WINDOW_METRICS,
    WINDOW_SEMANTICS,
    WINDOW_STARTS,
    load_aggregate_config,
)
from grids.eval.panel32_manifest import ACTIONS, load_panel_manifest  # noqa: E402
from grids.eval.panel32_paper_report import (  # noqa: E402
    ReportError,
    build_report,
    exact_mcnemar_p,
    render_markdown,
    write_report,
)
from grids.eval.panel32_render_six_axis_overall import summarize_group  # noqa: E402


PANEL = ROOT / "grids/eval/panel32_locked_v1.json"
CONFIGS = {
    name: ROOT / f"grids/eval/panel32_aggregate_{name}.json"
    for name in ("main", "ode", "dmd")
}


def _video_digest(context_id: str, action: str, model_id: str) -> str:
    return hashlib.sha256(
        f"{context_id}/{action}/{model_id}".encode("utf-8")
    ).hexdigest()


def _failed(
    group: str, role: str, model_id: str, pair_index: int,
    action: str, metric: str, start: int,
) -> int:
    # Every main external model fails style for the F command in the first
    # window: 32/288 = 1/9, making the five-model external mean exact.
    if (group == "main" and role == "external" and metric == "style"
            and start == 0 and action == "F"):
        return 1
    # Every model has the same relocation onset pattern.
    if metric == "relocation" and action == "R":
        return 1
    # One DMD ablation has 11 paired one-sided discordances.  Its exact raw
    # p is 1/1024 and Holm p within the 30-test control family is 30/1024.
    if (group == "dmd" and model_id == "ours_no_commit"
            and metric == "control" and start == 0 and pair_index < 11):
        return 1
    return 0


def _write_observations(path: Path, group: str) -> None:
    panel = load_panel_manifest(PANEL, verify_sources=False)
    config = load_aggregate_config(CONFIGS[group])
    rows = []
    pair_index = 0
    for context in panel.contexts:
        for action in ACTIONS:
            for model in config.models:
                digest = _video_digest(context.context_id, action, model.model_id)
                for metric in WINDOW_METRICS:
                    for start in WINDOW_STARTS:
                        rows.append({
                            "observation_schema_version": 1,
                            "panel_id": panel.panel_id,
                            "evaluation_id": config.evaluation_id,
                            "context_id": context.context_id,
                            "dataset": context.dataset,
                            "action": action,
                            "model_id": model.model_id,
                            "metric": metric,
                            "semantics": WINDOW_SEMANTICS,
                            "window_start_s": start,
                            "window_end_s": start + 6,
                            "failed": _failed(
                                group, model.role, model.model_id, pair_index,
                                action, metric, start),
                            "family_seat_count": 6,
                            "video_sha256": digest,
                        })
                for metric in ONSET_METRICS:
                    rows.append({
                        "observation_schema_version": 1,
                        "panel_id": panel.panel_id,
                        "evaluation_id": config.evaluation_id,
                        "context_id": context.context_id,
                        "dataset": context.dataset,
                        "action": action,
                        "model_id": model.model_id,
                        "metric": metric,
                        "semantics": ONSET_SEMANTICS,
                        "window_start_s": 0,
                        "window_end_s": 6,
                        "failed": _failed(
                            group, model.role, model.model_id, pair_index,
                            action, metric, 0),
                        "family_seat_count": 6,
                        "video_sha256": digest,
                    })
            pair_index += 1
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OBSERVATION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: Path) -> dict[str, Path]:
    observations = {}
    for group in CONFIGS:
        path = tmp_path / f"{group}.csv"
        _write_observations(path, group)
        observations[group] = path
    return observations


def _write_sidecar(path: Path, fieldnames: list[str], rows: list[dict] | None = None) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows or [])


def _build(
    observations: dict[str, Path], *,
    relocation_rows: list[dict] | None = None,
    conjuration_rows: list[dict] | None = None,
):
    directory = observations["main"].parent
    relocation = directory / "long_relocation.csv"
    conjuration = directory / "long_conjuration.csv"
    _write_sidecar(
        relocation, ["scene", "model", "time_s", "relocation_flag"],
        relocation_rows,
    )
    _write_sidecar(
        conjuration, ["scene", "model", "time_s", "conjuration_flag"],
        conjuration_rows,
    )
    return build_report(
        manifest_path=PANEL,
        main_config=CONFIGS["main"],
        main_observations=observations["main"],
        ode_config=CONFIGS["ode"],
        ode_observations=observations["ode"],
        dmd_config=CONFIGS["dmd"],
        dmd_observations=observations["dmd"],
        long_relocation_adjudication=relocation,
        long_conjuration_adjudication=conjuration,
    )


def test_report_exact_rates_external_average_clean_and_mcnemar(
    tmp_path: Path,
) -> None:
    report = _build(_fixture(tmp_path))
    assert report["status"] == "pass"
    assert len(report["rates"]) == (6 + 3 + 5) * 32
    assert len(report["cumulative_clean"]) == (6 + 3 + 5) * 5

    external_style = next(
        row for row in report["main_external_rate_averages"]
        if row["signature"] == [
            "style", "independent_six_second_window", 6, "all_actions"
        ]
    )
    pooled = external_style["pooled_across_external_videos"]
    assert (pooled["flagged"], pooled["scored"], pooled["fraction"]) == (
        160, 1440, "1/9")
    assert external_style["equal_dataset_and_model_weighted"]["fraction"] == "1/9"

    ours_clean = next(
        row for row in report["cumulative_clean"]
        if row["group"] == "main"
        and row["model_id"] == "ours_no_gan"
        and row["endpoint_s"] == 30
    )
    assert ours_clean["pooled"]["fraction"] == "8/9"
    external_clean = next(
        row for row in report["cumulative_clean"]
        if row["group"] == "main"
        and row["model_id"] == "lingbot"
        and row["endpoint_s"] == 30
    )
    assert external_clean["pooled"]["fraction"] == "7/9"

    tests = report["dmd_paired_mcnemar_holm"]["comparisons"]
    assert len(tests) == 4 * (4 * 5 + 2 + 5)
    no_commit = next(
        row for row in tests
        if row["comparison_model_id"] == "ours_no_commit"
        and row["outcome"] == "window_failure"
        and row["metric"] == "control"
        and row["endpoint_s"] == 6
    )
    assert no_commit["paired_n"] == 288
    assert no_commit["baseline_only_failure"] == 0
    assert no_commit["comparison_only_failure"] == 11
    assert no_commit["p_value_exact_fraction"] == "1/1024"
    assert no_commit["p_value_holm_exact_fraction"] == "5/256"
    assert no_commit["reject_holm_0_05"]

    markdown = render_markdown(report)
    assert "# Panel32 final results" in markdown
    assert "160/1440 (11.11%)" in markdown
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    write_report(report, json_path, markdown_path)
    assert json.loads(json_path.read_text())["status"] == "pass"
    assert markdown_path.read_text().startswith("# Panel32 final results")


def test_report_rejects_shared_model_decision_mismatch(tmp_path: Path) -> None:
    observations = _fixture(tmp_path)
    rows = list(csv.DictReader(observations["dmd"].open()))
    target = next(
        row for row in rows
        if row["model_id"] == "ours_no_gan"
        and row["metric"] == "style"
        and row["window_start_s"] == "0"
    )
    target["failed"] = "1"
    with observations["dmd"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OBSERVATION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ReportError, match="shared observation differs"):
        _build(observations)


def test_report_rejects_incomplete_normalized_grid(tmp_path: Path) -> None:
    observations = _fixture(tmp_path)
    rows = list(csv.DictReader(observations["ode"].open()))
    with observations["ode"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OBSERVATION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows[1:])
    with pytest.raises(ReportError, match="missing 1 of"):
        _build(observations)


def test_later_conjuration_persists_in_clean_curve_and_shaded_band(
    tmp_path: Path,
) -> None:
    observations = _fixture(tmp_path)
    event = {
        "scene": "frodobots-a20_N", "model": "ours_no_gan",
        "time_s": 7.0, "conjuration_flag": 1,
    }
    report = _build(observations, conjuration_rows=[event])
    clean = {
        row["endpoint_s"]: row["pooled"]["clean"]
        for row in report["cumulative_clean"]
        if row["group"] == "main"
        and row["model_id"] == "ours_no_gan"
    }
    assert clean[6] == 256
    assert clean[12] == 255
    assert clean[30] == 255
    assert report["long_conjuration_adjudication"]["accepted_events"] == 1

    frame = pytest.importorskip("pandas").read_csv(observations["main"])
    summary = summarize_group(
        "main", frame, report["groups"]["main"]["models"], {},
        {"ours_no_gan": [("frodobots-a20", "N", 7.0)]},
    )
    ours = summary[summary.model_id == "ours_no_gan"].set_index(
        "endpoint_s")
    assert ours.loc[6, "conjuration_rate_pct"] == 0.0
    assert ours.loc[12, "conjuration_rate_pct"] == pytest.approx(100 / 288)
    assert ours.loc[30, "conjuration_rate_pct"] == pytest.approx(100 / 288)


def test_report_rejects_invalid_long_conjuration_sidecar(tmp_path: Path) -> None:
    observations = _fixture(tmp_path)
    with pytest.raises(ReportError, match="conjuration_flag must be 0/1"):
        _build(observations, conjuration_rows=[{
            "scene": "frodobots-a20_N", "model": "ours_no_gan",
            "time_s": 7.0, "conjuration_flag": 2,
        }])


def test_exact_mcnemar_known_values() -> None:
    assert exact_mcnemar_p(0, 0) == 1
    assert exact_mcnemar_p(0, 1) == 1
    assert exact_mcnemar_p(0, 11).numerator == 1
    assert exact_mcnemar_p(0, 11).denominator == 1024
    assert exact_mcnemar_p(5, 5) == 1
