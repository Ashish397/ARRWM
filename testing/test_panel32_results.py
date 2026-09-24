"""Focused tests for panel32 result conversion and aggregation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grids.eval.panel32_aggregate import (
    AggregationError,
    aggregate,
    load_aggregate_config,
    load_observations,
)
from grids.eval.panel32_convert_results import ConversionError, REFERENCE_SEATS, convert
from grids.eval.panel32_control_rule import CONTROL_RULE_ID
from grids.eval.panel32_manifest import ACTIONS, load_panel_manifest
from grids.eval import iclr_long_relocation
from grids.eval import final_v3_quality_gpu
from grids.eval.preflight_action_conventions import shard_model_context_pairs
from grids.eval.panel32_dispatch_eval import (
    TASKS,
    holder_sort_key,
    require_prerequisites,
)
from grids.eval.panel32_validate_generations import black_screen_stats


PANEL = ROOT / "grids/eval/panel32_locked_v1.json"


def _config(path: Path, model: str = "ours_recovery_base") -> Path:
    payload = {
        "schema_version": 1,
        "evaluation_id": "panel32-converter-test",
        "actions": list(ACTIONS),
        "window_metrics": ["control", "style", "geometry", "hf"],
        "onset_metrics": ["conjuration", "relocation"],
        "window_starts_s": [0, 6, 12, 18, 24],
        "window_duration_s": 6,
        "family_seat_count": 6,
        "models": [{
            "model_id": model, "label": "Test model",
            "family": "ours", "role": "ours",
        }],
    }
    path.write_text(json.dumps(payload))
    return path


def _digest(scene: str, model: str) -> str:
    return hashlib.sha256(f"{scene}/{model}".encode()).hexdigest()


def _producer_fixture(root: Path, model: str = "ours_recovery_base") -> None:
    panel = load_panel_manifest(PANEL, verify_sources=False)
    videos = []
    style = []
    control = []
    geometry = []
    conjuration = []
    hf_support = []
    hf = []
    relocation = []
    starts = (0, 6, 9, 12, 18, 24)
    for context in panel.contexts:
        for action_index, action in enumerate(ACTIONS):
            scene = f"{context.context_id}_{action}"
            digest = _digest(scene, model)
            videos.append({
                "scene": scene, "uid": context.context_id,
                "direction": action, "model": model, "family": "ours",
            })
            for family, reference_model in REFERENCE_SEATS.items():
                if reference_model == model:
                    continue
                videos.append({
                    "scene": scene, "uid": context.context_id,
                    "direction": action, "model": reference_model,
                    "family": family,
                })
            for start in starts:
                base = {
                    "scene": scene, "model": model,
                    "window_start_s": start, "window_end_s": start + 6,
                    "video_sha256": digest,
                }
                style_failure = int((action_index + start) % 2 == 0)
                style.append({
                    **base,
                    "drift_from_real": 0.8 if style_failure else 0.2,
                    # Match the real producer: only the six-second anchor
                    # carries the legacy flag; later decisions come from the
                    # complete drift column.
                    "style_flag_072": style_failure if start == 0 else None,
                })
                control.append({
                    **base,
                    "control_failure": int(action_index == 0),
                    "control_rule_id": CONTROL_RULE_ID,
                })
                geometry.append({**base, "geometry_flag": int(start == 24)})
                # Later positives deliberately prove that only the onset row is emitted.
                conjuration.append({
                    **base, "conjuration_flag": int(start in (0, 12)),
                })
                hf_support.append(base)
            for endpoint in (6, 12, 18, 24, 30):
                hf.append({
                    "scene": scene, "model": model, "endpoint_s": endpoint,
                    "hf_B": 200 if endpoint == 30 else 0,
                    "hf_failure": int(endpoint == 30),
                })
                for reference_model in REFERENCE_SEATS.values():
                    if reference_model == model:
                        continue
                    hf.append({
                        "scene": scene, "model": reference_model,
                        "endpoint_s": endpoint, "hf_B": 0, "hf_failure": 0,
                    })
            for horizon in (6, 15, 30):
                relocation.append({
                    "scene": scene, "model": model, "horizon_s": horizon,
                    "relocation_flag_6s_exploratory": int(horizon == 6 and action == "R"),
                    "video_sha256": digest,
                })
    pd.DataFrame(videos).to_csv(root / "video_manifest.csv", index=False)
    pd.DataFrame(style).to_csv(root / "style_windows.csv", index=False)
    pd.DataFrame(control).to_csv(root / "control_window_scored.csv", index=False)
    pd.DataFrame(geometry).to_csv(root / "geometry_windows.csv", index=False)
    pd.DataFrame(conjuration).to_csv(root / "conjuration_windows.csv", index=False)
    pd.DataFrame(hf_support).to_csv(root / "hf_window_trajectory.csv", index=False)
    pd.DataFrame(hf).to_csv(root / "hf_five_horizon_panel_rows.csv", index=False)
    pd.DataFrame(relocation).to_csv(root / "relocation_panel_rows.csv", index=False)


def _finalized_v2_fixture(root: Path) -> Path:
    legacy = pd.read_csv(root / "conjuration_windows.csv")
    finalized = legacy[legacy.window_start_s.isin((0, 6, 12, 18, 24))].copy()
    finalized["instrument_profile"] = (
        "panel32-conjuration-v2-human-adjudicated"
    )
    path = root / "conjuration_v2_windows.csv"
    finalized.to_csv(path, index=False)
    return path


def test_converter_emits_exact_nonoverlap_and_onset_contract(tmp_path: Path) -> None:
    _producer_fixture(tmp_path)
    config_path = _config(tmp_path / "config.json")
    output = tmp_path / "observations.csv"
    result = convert(PANEL, config_path, tmp_path, output)
    assert result["normalized_rows"] == 32 * 9 * (4 * 5 + 2)

    rows = pd.read_csv(output)
    window = rows[rows.metric.isin(["control", "style", "geometry", "hf"])]
    onset = rows[rows.metric.isin(["conjuration", "relocation"])]
    assert set(window.window_start_s) == {0, 6, 12, 18, 24}
    assert 9 not in set(rows.window_start_s)
    assert set(onset.window_start_s) == {0}
    assert set(onset.window_end_s) == {6}
    assert set(rows.dataset) == {"FrodoBots", "Ego4D", "Sekai", "SpatialVID"}
    assert (rows.groupby(["context_id", "action", "model_id"]).video_sha256.nunique() == 1).all()

    panel = load_panel_manifest(PANEL, verify_sources=False)
    config = load_aggregate_config(config_path)
    observations, digest = load_observations(output, panel, config)
    assert len(observations) == len(rows)
    assert len(digest) == 64
    aggregate_result = aggregate(
        PANEL, config_path, output, tmp_path / "aggregated")
    assert aggregate_result["status"] == "pass"
    assert aggregate_result["dataset_rate_rows"] == 4 * 27
    assert aggregate_result["macro_rate_rows"] == 27


def test_converter_rejects_cross_instrument_hash_mismatch(tmp_path: Path) -> None:
    _producer_fixture(tmp_path)
    config_path = _config(tmp_path / "config.json")
    style_path = tmp_path / "style_windows.csv"
    style = pd.read_csv(style_path)
    first_scene = style.loc[0, "scene"]
    style.loc[style.scene == first_scene, "video_sha256"] = "0" * 64
    style.to_csv(style_path, index=False)
    with pytest.raises(ConversionError, match="SHA-256 mismatch"):
        convert(PANEL, config_path, tmp_path, tmp_path / "observations.csv")


def test_converter_rejects_missing_producer_window(tmp_path: Path) -> None:
    _producer_fixture(tmp_path)
    config_path = _config(tmp_path / "config.json")
    path = tmp_path / "geometry_windows.csv"
    frame = pd.read_csv(path).iloc[1:].copy()
    frame.to_csv(path, index=False)
    with pytest.raises(ConversionError, match="missing 1 rows"):
        convert(PANEL, config_path, tmp_path, tmp_path / "observations.csv")


def test_converter_explicit_finalized_v2_controls_anchor_and_provenance(
    tmp_path: Path,
) -> None:
    _producer_fixture(tmp_path)
    config_path = _config(tmp_path / "config.json")
    finalized_path = _finalized_v2_fixture(tmp_path)
    finalized = pd.read_csv(finalized_path)
    target = finalized.index[
        (finalized.scene == "frodobots-a20_F")
        & (finalized.model == "ours_recovery_base")
        & (finalized.window_start_s == 0)
    ][0]
    finalized.loc[target, "conjuration_flag"] = 1
    finalized.to_csv(finalized_path, index=False)

    output = tmp_path / "observations.csv"
    result = convert(
        PANEL, config_path, tmp_path, output,
        finalized_conjuration_v2_windows=finalized_path,
    )
    rows = pd.read_csv(output)
    anchor = rows[
        (rows.context_id == "frodobots-a20")
        & (rows.action == "F")
        & (rows.model_id == "ours_recovery_base")
        & (rows.metric == "conjuration")
    ]
    assert anchor.failed.tolist() == [1]
    assert result["conjuration_windows"]["finalized_v2"] is True
    assert result["conjuration_windows"]["path"] == str(finalized_path.resolve())


def test_converter_rejects_incomplete_or_stale_finalized_v2(tmp_path: Path) -> None:
    _producer_fixture(tmp_path)
    config_path = _config(tmp_path / "config.json")
    finalized_path = _finalized_v2_fixture(tmp_path)
    finalized = pd.read_csv(finalized_path).iloc[1:].copy()
    finalized.to_csv(finalized_path, index=False)
    with pytest.raises(ConversionError, match="missing 1 rows"):
        convert(
            PANEL, config_path, tmp_path, tmp_path / "observations.csv",
            finalized_conjuration_v2_windows=finalized_path,
        )

    _finalized_v2_fixture(tmp_path)
    finalized = pd.read_csv(finalized_path)
    first_scene = finalized.loc[0, "scene"]
    finalized.loc[finalized.scene == first_scene, "video_sha256"] = "0" * 64
    finalized.to_csv(finalized_path, index=False)
    with pytest.raises(ConversionError, match="SHA-256 mismatch"):
        convert(
            PANEL, config_path, tmp_path, tmp_path / "observations.csv",
            finalized_conjuration_v2_windows=finalized_path,
        )


def test_shipped_configs_cover_main_ode_and_dmd_sets() -> None:
    directory = ROOT / "grids/eval"
    main = load_aggregate_config(directory / "panel32_aggregate_main.json")
    ode = load_aggregate_config(directory / "panel32_aggregate_ode.json")
    dmd = load_aggregate_config(directory / "panel32_aggregate_dmd.json")
    assert len(main.models) == 6
    assert len(ode.models) == 3
    assert len(dmd.models) == 5
    assert all(config.raw["family_seat_count"] == 6 for config in (main, ode, dmd))


def test_aggregate_config_still_rejects_non_six_seat_hf_panel(tmp_path: Path) -> None:
    path = _config(tmp_path / "bad.json")
    payload = json.loads(path.read_text())
    payload["family_seat_count"] = 5
    path.write_text(json.dumps(payload))
    with pytest.raises(AggregationError, match="family_seat_count must be 6"):
        load_aggregate_config(path)


def test_long_relocation_cache_invalidates_when_video_changes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    paths = []
    for model in ("model_a", "model_b"):
        path = videos / f"{model}.mp4"
        path.write_bytes(model.encode())
        paths.append(path)
    rows = [
        {"scene": "ctx_F", "model": model, "path": str(path)}
        for model, path in zip(("model_a", "model_b"), paths)
    ]
    calls: list[str] = []

    def fake_score(row, **_kwargs):
        calls.append(row["model"])
        return {"model": row["model"], "endpoint_rows": [], "events": []}

    monkeypatch.setattr(iclr_long_relocation, "score_video", fake_score)
    kwargs = {
        "local_prefix": "", "remote_prefix": "", "sample_hz": 4.0,
        "bank_s": 1.0, "cross_threshold": 20,
        "coherence_threshold": 35, "keypoint_threshold": 100,
    }
    destination = str(tmp_path / "results")
    iclr_long_relocation.process_scene(rows, destination, False, **kwargs)
    assert calls == ["model_a", "model_b"]

    iclr_long_relocation.process_scene(rows, destination, False, **kwargs)
    assert calls == ["model_a", "model_b"]

    # A same-path regenerated rollout must invalidate the cached scene.
    paths[0].write_bytes(b"replacement-video")
    iclr_long_relocation.process_scene(rows, destination, False, **kwargs)
    assert calls == ["model_a", "model_b", "model_a", "model_b"]


def test_preflight_48_shards_cover_full_model_context_product() -> None:
    models = [f"model_{index}" for index in range(15)]
    contexts = [f"context_{index}" for index in range(32)]
    partitions = [
        shard_model_context_pairs(models, contexts, index, 48)
        for index in range(48)
    ]
    flattened = [pair for partition in partitions for pair in partition]
    assert all(len(partition) == 10 for partition in partitions)
    assert len(flattened) == 15 * 32
    assert len(set(flattened)) == len(flattened)


def test_conjuration_cache_requires_complete_window_grid(tmp_path: Path) -> None:
    target = tmp_path / "cache.json"
    digest = "a" * 64
    row = SimpleNamespace(scene="ctx_F", model="model_a")

    def payload(starts):
        return {
            "scene": row.scene, "model": row.model, "metric": "conjuration",
            "video_sha256": digest,
            "rows": [
                {"window_start_s": start, "video_sha256": digest}
                for start in starts
            ],
        }

    target.write_text(json.dumps(payload((0, 6, 9, 12, 18))))
    assert not final_v3_quality_gpu.valid_cache(
        target, row, "conjuration", digest)
    target.write_text(json.dumps(payload((0, 6, 9, 12, 18, 24))))
    assert final_v3_quality_gpu.valid_cache(
        target, row, "conjuration", digest)


def test_dispatcher_enforces_evaluation_phase_order(tmp_path: Path) -> None:
    evaluation = tmp_path / "eval_final"
    (evaluation / "logs").mkdir(parents=True)
    (evaluation / "preflight").mkdir()
    (evaluation / "markers").mkdir()
    require_prerequisites("setup", evaluation, 4)
    with pytest.raises(SystemExit, match="setup is not complete"):
        require_prerequisites("preflight", evaluation, 4)
    (evaluation / "logs/SETUP_COMPLETE").touch()
    with pytest.raises(SystemExit, match="missing 4 shards"):
        require_prerequisites("preflight-finalize", evaluation, 4)
    for shard in range(4):
        (evaluation / f"markers/preflight_shard{shard}.COMPLETE").touch()
    require_prerequisites("preflight-finalize", evaluation, 4)
    with pytest.raises(SystemExit, match="preflight is not complete"):
        require_prerequisites("metrics", evaluation, 4)
    (evaluation / "preflight/ACTION_PREFLIGHT_COMPLETE").touch()
    require_prerequisites("metrics", evaluation, 4)
    with pytest.raises(SystemExit, match="missing 24 metric shards"):
        require_prerequisites("finish", evaluation, 4)
    for task in TASKS:
        for shard in range(4):
            (evaluation / f"markers/{task}_shard{shard}.COMPLETE").touch()
    require_prerequisites("finish", evaluation, 4)


def test_holder_array_ids_sort_without_integer_conversion_failure() -> None:
    ids = ["6823827_47", "6823826", "6823827_3", "6823827_12"]
    assert sorted(ids, key=holder_sort_key) == [
        "6823826", "6823827_3", "6823827_12", "6823827_47",
    ]


def test_generation_black_screen_gate_is_narrow() -> None:
    assert black_screen_stats(np.zeros((32, 32, 3), dtype=np.uint8))["black_screen"]
    dark_texture = np.zeros((32, 32, 3), dtype=np.uint8)
    dark_texture[::2, ::2] = 20
    assert not black_screen_stats(dark_texture)["black_screen"]
