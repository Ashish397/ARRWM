#!/usr/bin/env python3
"""Render the audited Panel32 summary figures with all six failure axes.

The four windowed failure rates are marginal rates in the current six-second
window.  Conjuration and relocation are onset events, so their shaded bands
persist after onset.  The green curve is stricter: a rollout remains clean only
if it has never failed any of the six axes through the plotted endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


ENDPOINTS = (6, 12, 18, 24, 30)
WINDOW_METRICS = ("control", "style", "geometry", "hf")
BAND_METRICS = WINDOW_METRICS + ("conjuration", "relocation")
BAND_LABELS = {
    "control": "Control",
    "style": "Style",
    "geometry": "Geometry",
    "hf": "HF",
    "conjuration": "Conjuration",
    "relocation": "Relocation",
}
BAND_COLORS = {
    "control": "#4D4D4D",
    "style": "#7B5AA6",
    "geometry": "#D2556F",
    "hf": "#F28E2B",
    "conjuration": "#76B7B2",
    "relocation": "#4E79A7",
}
GROUP_FILES = {
    "main": "panel32_main_observations.csv",
    "ode": "panel32_ode_observations.csv",
    "dmd": "panel32_dmd_observations.csv",
}
OUTPUT_FILES = {
    "main": "overall_external.png",
    "ode": "overall_ode.png",
    "dmd": "overall_ablations.png",
}

# Keep the no-commit arm in the audited observations and numeric summary, but
# omit it from the compact overall-ablation figure.  ``No CARN`` is the single
# CARN-removal comparison shown there; the dedicated trajectory material can
# still use the no-commit measurements when required.
OVERALL_PLOT_EXCLUSIONS = {
    "dmd": {"ours_no_commit"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--final-dir", type=Path,
        default=Path("analysis/panel32_v1/final"),
    )
    parser.add_argument(
        "--paper-figures", type=Path, default=Path("iclr/Figures"),
    )
    parser.add_argument("--long-relocation-adjudication", type=Path)
    parser.add_argument("--long-conjuration-adjudication", type=Path)
    return parser.parse_args()


def rollout_keys(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(frame["context_id"], frame["action"]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def accepted_long_events(
    path: Path, *, event_name: str, flag_column: str,
    expected_sha256: str, valid_models: set[str],
    valid_rollouts: set[tuple[str, str]],
) -> dict[str, list[tuple[str, str, float]]]:
    if not path.is_file():
        raise RuntimeError(f"missing long-{event_name} adjudication: {path}")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"long-{event_name} adjudication SHA-256 mismatch: "
            f"{actual_sha256} != {expected_sha256}"
        )
    frame = pd.read_csv(path)
    required = {"scene", "model", "time_s", flag_column}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"long-{event_name} adjudication missing columns {sorted(missing)}"
        )
    flags = pd.to_numeric(frame[flag_column], errors="coerce")
    times = pd.to_numeric(frame["time_s"], errors="coerce")
    if flags.isna().any() or not flags.isin([0, 1]).all():
        raise RuntimeError(f"long-{event_name} {flag_column} must contain only 0/1")
    if times.isna().any() or not np.isfinite(times).all() or not times.between(0, 30).all():
        raise RuntimeError(f"long-{event_name} time_s must be finite and within [0,30]")
    frame = frame.assign(**{flag_column: flags.astype(int), "time_s": times})
    output: dict[str, list[tuple[str, str, float]]] = {}
    seen: set[tuple[str, str, float]] = set()
    for row in frame.itertuples(index=False):
        try:
            context_id, action = str(row.scene).rsplit("_", 1)
        except ValueError as exc:
            raise RuntimeError(
                f"long-{event_name} adjudication has malformed scene {row.scene!r}"
            ) from exc
        if (context_id, action) not in valid_rollouts:
            raise RuntimeError(
                f"long-{event_name} adjudication has unknown scene {row.scene!r}"
            )
        # The audit sidecar covers the complete evaluation fleet, including
        # raw ablations intentionally omitted from the paper configurations.
        # Keep that evidence in the sidecar but do not plot unreported rows.
        if str(row.model) not in valid_models:
            continue
        key = (str(row.scene), str(row.model), float(row.time_s))
        if key in seen:
            raise RuntimeError(f"long-{event_name} adjudication duplicates {key}")
        seen.add(key)
        if int(getattr(row, flag_column)) != 1:
            continue
        output.setdefault(str(row.model), []).append(
            (context_id, action, float(row.time_s))
        )
    return output


def summarize_group(
    group: str,
    observations: pd.DataFrame,
    models: list[dict],
    long_relocation_events: dict[str, list[tuple[str, str, float]]],
    long_conjuration_events: dict[str, list[tuple[str, str, float]]],
) -> pd.DataFrame:
    records: list[dict] = []
    expected_models = [model["model_id"] for model in models]
    if list(observations["model_id"].drop_duplicates()) != expected_models:
        raise RuntimeError(f"{group}: observation/model order mismatch")

    for model in models:
        model_id = model["model_id"]
        subset = observations[observations["model_id"] == model_id]
        all_rollouts = rollout_keys(subset)
        if len(all_rollouts) != 288:
            raise RuntimeError(f"{group}/{model_id}: expected 288 rollouts")

        direct_onsets: dict[str, set[tuple[str, str]]] = {}
        for metric in ("conjuration", "relocation"):
            metric_rows = subset[subset["metric"] == metric]
            if len(metric_rows) != 288:
                raise RuntimeError(
                    f"{group}/{model_id}/{metric}: expected 288 onset rows"
                )
            direct_onsets[metric] = rollout_keys(
                metric_rows[metric_rows["failed"] == 1]
            )

        for endpoint in ENDPOINTS:
            marginal: dict[str, float] = {}
            cumulative_bad: set[tuple[str, str]] = set()
            for metric in WINDOW_METRICS:
                metric_rows = subset[
                    (subset["metric"] == metric)
                    & (subset["window_end_s"] == endpoint)
                ]
                if metric == "control":
                    metric_rows = metric_rows[metric_rows["action"] != "N"]
                    expected = 256
                else:
                    expected = 288
                if len(metric_rows) != expected:
                    raise RuntimeError(
                        f"{group}/{model_id}/{metric}/{endpoint}: "
                        f"expected {expected} rows"
                    )
                marginal[metric] = 100.0 * metric_rows["failed"].sum() / expected

                history = subset[
                    (subset["metric"] == metric)
                    & (subset["window_end_s"] <= endpoint)
                    & (subset["failed"] == 1)
                ]
                cumulative_bad |= rollout_keys(history)

            cumulative_bad |= direct_onsets["conjuration"]
            cumulative_bad |= direct_onsets["relocation"]
            conjuration = set(direct_onsets["conjuration"])
            relocation = set(direct_onsets["relocation"])
            for context_id, action, time_s in long_relocation_events.get(model_id, []):
                if time_s <= endpoint:
                    relocation.add((context_id, action))
                    cumulative_bad.add((context_id, action))
            for context_id, action, time_s in long_conjuration_events.get(model_id, []):
                if time_s <= endpoint:
                    conjuration.add((context_id, action))
                    cumulative_bad.add((context_id, action))

            marginal["conjuration"] = 100.0 * len(conjuration) / 288
            marginal["relocation"] = 100.0 * len(relocation) / 288
            clean_count = len(all_rollouts - cumulative_bad)
            records.append({
                "group": group,
                "model_id": model_id,
                "model_label": model["label"],
                "endpoint_s": endpoint,
                **{f"{metric}_rate_pct": marginal[metric]
                   for metric in BAND_METRICS},
                "clean_all_six_count": clean_count,
                "clean_all_six_rate_pct": 100.0 * clean_count / 288,
            })
    return pd.DataFrame.from_records(records)


def layout(group: str, count: int) -> tuple[int, int, tuple[float, float]]:
    if group == "ode":
        return 1, 3, (13.8, 4.6)
    if group == "main":
        return 2, 3, (14.2, 8.0)
    if group == "dmd" and count <= 4:
        return 2, 2, (10.0, 7.6)
    return 2, 4, (16.0, 7.6)


def render_group(group: str, summary: pd.DataFrame, output: Path) -> None:
    model_order = list(summary["model_id"].drop_duplicates())
    nrows, ncols, figsize = layout(group, len(model_order))
    plt.rcParams.update({
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.2,
    })
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, sharex=True, sharey=True, squeeze=False
    )
    axes_flat = axes.ravel()
    x = np.asarray((0,) + ENDPOINTS, dtype=float)

    max_total = 100.0
    for model_id in model_order:
        model_rows = summary[summary["model_id"] == model_id]
        totals = model_rows[[f"{m}_rate_pct" for m in BAND_METRICS]].sum(axis=1)
        max_total = max(max_total, float(totals.max()))
    ymax = max(100, int(math.ceil((max_total + 8) / 50.0) * 50))

    for index, model_id in enumerate(model_order):
        ax = axes_flat[index]
        model_rows = summary[summary["model_id"] == model_id].set_index(
            "endpoint_s"
        ).loc[list(ENDPOINTS)]
        bands = [
            np.r_[0.0, model_rows[f"{metric}_rate_pct"].to_numpy(float)]
            for metric in BAND_METRICS
        ]
        clean = np.r_[100.0, model_rows["clean_all_six_rate_pct"].to_numpy(float)]
        total = np.sum(np.vstack(bands), axis=0)
        ax.stackplot(
            x, bands,
            colors=[BAND_COLORS[m] for m in BAND_METRICS],
            alpha=0.80,
            edgecolor="white",
            linewidth=0.35,
        )
        ax.plot(x, total, color="#111111", marker="o", markersize=2.8,
                linewidth=1.7, zorder=5)
        ax.plot(x, clean, color="#00843D", marker="o", markersize=3.2,
                linewidth=2.3, linestyle="--", zorder=6)
        label = model_rows["model_label"].iloc[0]
        label = label.replace(" 1.3B", "").replace(" 5B", "")
        label = label.replace(" (ODE)", "")
        ax.text(0.03, 0.95, label, transform=ax.transAxes, ha="left", va="top",
                fontweight="bold", fontsize=11.5)
        ax.set_xlim(-0.7, 30.7)
        ax.set_ylim(0, ymax)
        ax.set_xticks(x)
        ax.grid(True, color="#D7D7D7", linewidth=0.6, alpha=0.75)
        if index % ncols == 0:
            ax.set_ylabel("Stacked rates (%)")
        if index // ncols == nrows - 1:
            ax.set_xlabel("Endpoint (s)")

    for ax in axes_flat[len(model_order):]:
        ax.axis("off")

    handles = [
        Patch(facecolor=BAND_COLORS[m], label=BAND_LABELS[m], alpha=0.80)
        for m in BAND_METRICS
    ]
    handles.extend([
        Line2D([0], [0], color="#00843D", marker="o", linestyle="--",
               linewidth=2.3, markersize=3.2,
               label="Clean on all 6 axes through endpoint"),
        Line2D([0], [0], color="#111111", marker="o", linewidth=1.7,
               markersize=2.8, label="Sum of marginal failure rates"),
    ])
    legend_columns = 4 if group != "ode" else 4
    fig.legend(handles=handles, loc="lower center", ncol=legend_columns,
               frameon=False, bbox_to_anchor=(0.5, 0.005))
    bottom = 0.18 if group == "ode" else 0.15
    fig.tight_layout(rect=(0, bottom, 1, 1), h_pad=1.3, w_pad=1.1)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    report_path = args.final_dir / "panel32_final_report.json"
    report = json.loads(report_path.read_text())
    if report.get("status") != "pass":
        raise RuntimeError("refusing to render from a non-passing report")
    observation_dir = args.final_dir / "observations"
    if not observation_dir.is_dir():
        # The remote publisher writes normalized observations beside the
        # report; the local handoff packages them in an observations/
        # subdirectory.  Both layouts carry the same hash-checked files.
        observation_dir = args.final_dir
    observations_by_group = {
        group: pd.read_csv(observation_dir / GROUP_FILES[group])
        for group in ("main", "ode", "dmd")
    }
    valid_models = {
        model["model_id"]
        for group in report["groups"].values()
        for model in group["models"]
    }
    valid_rollouts = set().union(*(
        rollout_keys(frame) for frame in observations_by_group.values()
    ))

    def sidecar_path(argument: Path | None, filename: str, block: str) -> Path:
        if argument is not None:
            return argument.expanduser().resolve()
        local = observation_dir / filename
        if local.is_file():
            return local.resolve()
        return Path(report[block]["path"]).expanduser().resolve()

    relocation_path = sidecar_path(
        args.long_relocation_adjudication,
        "long_relocation_events_adjudicated.csv",
        "long_relocation_adjudication",
    )
    conjuration_path = sidecar_path(
        args.long_conjuration_adjudication,
        "long_conjuration_events_adjudicated.csv",
        "long_conjuration_adjudication",
    )
    long_relocation_events = accepted_long_events(
        relocation_path, event_name="relocation", flag_column="relocation_flag",
        expected_sha256=report["long_relocation_adjudication"]["sha256"],
        valid_models=valid_models, valid_rollouts=valid_rollouts,
    )
    long_conjuration_events = accepted_long_events(
        conjuration_path, event_name="conjuration", flag_column="conjuration_flag",
        expected_sha256=report["long_conjuration_adjudication"]["sha256"],
        valid_models=valid_models, valid_rollouts=valid_rollouts,
    )

    summaries = []
    for group in ("main", "ode", "dmd"):
        observations = observations_by_group[group]
        summary = summarize_group(
            group, observations, report["groups"][group]["models"],
            long_relocation_events, long_conjuration_events,
        )
        summaries.append(summary)
        final_output = args.final_dir / "figures" / OUTPUT_FILES[group]
        plot_summary = summary[
            ~summary["model_id"].isin(OVERALL_PLOT_EXCLUSIONS.get(group, set()))
        ]
        render_group(group, plot_summary, final_output)
        paper_output = args.paper_figures / OUTPUT_FILES[group]
        paper_output.write_bytes(final_output.read_bytes())

    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(args.final_dir / "six_axis_plot_summary.csv", index=False)
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
