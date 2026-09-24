"""Convert completed panel32 evaluator outputs to strict aggregation rows.

The output schema is consumed by :mod:`panel32_aggregate`.  This converter is
fail-closed: every selected model must have the exact 32 x 9 video product,
five independent windows for each time-varying metric, and one six-second row
for each onset metric.  Video hashes must agree across all producers.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from panel32_aggregate import (
    FAMILY_SEAT_COUNT,
    OBSERVATION_COLUMNS,
    OBSERVATION_SCHEMA_VERSION,
    ONSET_SEMANTICS,
    WINDOW_SEMANTICS,
    WINDOW_STARTS,
)
from panel32_manifest import ACTIONS, load_panel_manifest


def read_unique(path: Path, keys: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty or frame.duplicated(keys).any():
        raise AssertionError((path, "empty or duplicate rows", keys))
    return frame


def hf_flags(out: Path, manifest: pd.DataFrame, resolver: dict[str, Any],
             selected_models: list[str]) -> pd.DataFrame:
    """Reproduce the dynamic six-family panel for all five paper windows."""
    windows = read_unique(out / "cpu_windows.csv", ["scene", "model", "window_start_s"])
    windows = windows[windows.window_start_s.isin(WINDOW_STARTS)].copy()
    values = windows[["scene", "model", "window_start_s", "d_blur_from_early_base"]].rename(
        columns={"d_blur_from_early_base": "d_blur"}
    )
    # Preserve the deployed six-second anchor (one native frame later than the
    # fixed-window row for historical producer compatibility).
    anchor = read_unique(out / "cpu_endpoints_scored.csv", ["scene", "model", "horizon_s"])
    anchor = anchor[anchor.horizon_s == 6][["scene", "model", "d_blur"]].copy()
    anchor["window_start_s"] = 0
    values = pd.concat([values[values.window_start_s != 0], anchor], ignore_index=True)
    if values.duplicated(["scene", "model", "window_start_s"]).any():
        raise AssertionError("duplicate HF source rows")

    family = manifest.set_index(["scene", "model"]).family.to_dict()
    seats = {str(k): str(v) for k, v in resolver["family_seats"].items()}
    lookup = values.set_index(["scene", "model", "window_start_s"]).d_blur.to_dict()
    rows = []
    for (scene, start), group in values.groupby(["scene", "window_start_s"], sort=True):
        fixed = {name: lookup[(scene, model, start)] for name, model in seats.items()}
        if not all(np.isfinite(value) for value in fixed.values()):
            raise AssertionError((scene, start, fixed))
        for candidate in group[group.model.isin(selected_models)].itertuples():
            candidate_seats = fixed.copy()
            candidate_seats[family[(scene, candidate.model)]] = float(candidate.d_blur)
            reference = float(np.median(list(candidate_seats.values())))
            rows.append({
                "scene": scene,
                "model": candidate.model,
                "window_start_s": int(start),
                "hf_failure": int(reference - float(candidate.d_blur) > 150),
            })
    result = pd.DataFrame(rows)
    if result.duplicated(["scene", "model", "window_start_s"]).any():
        raise AssertionError("duplicate normalized HF rows")
    return result


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OBSERVATION_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path,
                        help="completed final_v2/v3 evaluation directory")
    parser.add_argument("--manifest", required=True, type=Path,
                        help="locked panel32 source manifest")
    parser.add_argument("--eval-config", required=True, type=Path,
                        help="panel32 fleet resolver configuration")
    parser.add_argument("--aggregate-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    out = args.out.resolve()
    panel = load_panel_manifest(args.manifest.resolve(), verify_sources=False)
    resolver = json.loads(args.eval_config.resolve().read_text())
    aggregate = json.loads(args.aggregate_config.resolve().read_text())
    models = [record["model_id"] for record in aggregate["models"]]
    evaluation_id = str(aggregate["evaluation_id"])
    if aggregate["family_seat_count"] != FAMILY_SEAT_COUNT:
        raise AssertionError("aggregation config is not the six-family panel")
    if not set(models) <= set(resolver["models"]):
        raise AssertionError("aggregation config references unknown resolver models")

    full_manifest = read_unique(out / "video_manifest.csv", ["scene", "model"])
    manifest = full_manifest[full_manifest.model.isin(models)].copy()
    expected_videos = len(panel.contexts) * len(ACTIONS) * len(models)
    if len(manifest) != expected_videos or not manifest.local_video.all():
        raise AssertionError(("incomplete selected manifest", len(manifest), expected_videos))
    identity = manifest.set_index(["scene", "model"])[["uid", "direction"]]
    datasets = {context.context_id: context.dataset for context in panel.contexts}

    cpu = read_unique(out / "cpu_endpoints_scored.csv", ["scene", "model", "horizon_s"])
    hashes = cpu[cpu.horizon_s == 6].set_index(["scene", "model"]).video_sha256
    if len(hashes.loc[identity.index]) != expected_videos:
        raise AssertionError("incomplete canonical video hashes")

    control = read_unique(out / "control_window_scored.csv", ["scene", "model", "window_start_s"])
    style = read_unique(out / "style_windows.csv", ["scene", "model", "window_start_s"])
    geometry = read_unique(out / "geometry_windows.csv", ["scene", "model", "window_start_s"])
    conjuration = read_unique(out / "conjuration_windows.csv", ["scene", "model", "window_start_s"])
    relocation = read_unique(out / "relocation_panel_rows.csv", ["scene", "model", "horizon_s"])
    hf = hf_flags(out, full_manifest, resolver, models)

    sources = {
        "control": (control[control.window_start_s.isin(WINDOW_STARTS)], "control_failure", "window_start_s"),
        "style": (style[style.window_start_s.isin(WINDOW_STARTS)].assign(
            normalized_flag=lambda x: (x.drift_from_real > 0.72).astype(int)),
                  "normalized_flag", "window_start_s"),
        "geometry": (geometry[geometry.window_start_s.isin(WINDOW_STARTS)], "geometry_flag", "window_start_s"),
        "hf": (hf, "hf_failure", "window_start_s"),
        "conjuration": (conjuration[conjuration.window_start_s == 0], "conjuration_flag", "window_start_s"),
        "relocation": (relocation[relocation.horizon_s == 6],
                       "relocation_flag_6s_exploratory", None),
    }
    rows: list[dict[str, Any]] = []
    selected_keys = set(identity.index)
    for metric, (frame, flag, start_column) in sources.items():
        frame = frame[frame.model.isin(models)].copy()
        onset = metric in ("conjuration", "relocation")
        expected = expected_videos if onset else expected_videos * len(WINDOW_STARTS)
        if len(frame) != expected:
            raise AssertionError((metric, len(frame), expected))
        if not set(zip(frame.scene, frame.model)) <= selected_keys:
            raise AssertionError((metric, "unexpected scene/model"))
        for record in frame.itertuples(index=False):
            key = (record.scene, record.model)
            canonical_hash = str(hashes.loc[key])
            producer_hash = getattr(record, "video_sha256", canonical_hash)
            if str(producer_hash) != canonical_hash:
                raise AssertionError((metric, key, "video hash mismatch"))
            uid = str(identity.loc[key, "uid"])
            action = str(identity.loc[key, "direction"])
            start = 0 if onset else int(getattr(record, start_column))
            rows.append({
                "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
                "panel_id": panel.panel_id,
                "evaluation_id": evaluation_id,
                "context_id": uid,
                "dataset": datasets[uid],
                "action": action,
                "model_id": record.model,
                "metric": metric,
                "semantics": ONSET_SEMANTICS if onset else WINDOW_SEMANTICS,
                "window_start_s": start,
                "window_end_s": start + 6,
                "failed": int(getattr(record, flag)),
                "family_seat_count": FAMILY_SEAT_COUNT,
                "video_sha256": canonical_hash,
            })

    expected_rows = expected_videos * (4 * len(WINDOW_STARTS) + 2)
    if len(rows) != expected_rows:
        raise AssertionError(("normalized rows", len(rows), expected_rows))
    rows.sort(key=lambda row: (
        row["context_id"], ACTIONS.index(row["action"]), row["model_id"],
        row["metric"], row["window_start_s"],
    ))
    atomic_csv(args.output.resolve(), rows)
    print(json.dumps({
        "output": str(args.output.resolve()), "models": models,
        "videos": expected_videos, "observations": len(rows),
    }, indent=2))


if __name__ == "__main__":
    main()
