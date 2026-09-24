"""Audit command adapters and measured yaw/throttle orientation.

This is deliberately separate from model-quality scoring.  It rejects a
systematic sign inversion, while recording an inconclusive response as such
rather than pretending a non-responsive model proves an adapter convention.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


YUME_ACTIONS = {
    "F": ("W", "·", 4, 0, 0),
    "FR": ("W", "→", 4, 4, 4),
    "R": ("None", "→", 0, 4, 4),
    "BR": ("S", "→", 4, 4, 4),
    "B": ("S", "·", 4, 0, 0),
    "BL": ("S", "←", 4, 4, 4),
    "L": ("None", "←", 0, 4, 4),
    "FL": ("W", "←", 4, 4, 4),
    "N": ("None", "·", 0, 0, 0),
}


def med(frame: pd.DataFrame, direction: str, component: str, expected: int) -> float:
    values = frame.loc[frame.direction == direction, component]
    assert len(values) == expected, (direction, component, len(values), expected)
    return float(values.median())


def status(delta: float, tolerance: float = 0.02) -> str:
    if delta > tolerance:
        return "correct"
    if delta < -tolerance:
        return "reversed"
    return "inconclusive_low_response"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--require-positive", default="minwm,minwm_ode")
    args = parser.parse_args()
    out = args.out.resolve()
    control = pd.read_csv(out / "control_windows.csv")
    control = control[(control.window_start_s == 0) & (control.direction != "N")]
    manifest = pd.read_csv(out / "video_manifest.csv")
    contexts = int(manifest.uid.nunique())
    expected_per_model = contexts * 9
    expected_directional = contexts * 8
    rows = []
    for model, frame in control.groupby("model"):
        assert len(frame) == expected_directional, (model, len(frame))
        throttle_delta = 0.5 * (
            med(frame, "F", "g0", contexts) + med(frame, "FR", "g0", contexts) + med(frame, "FL", "g0", contexts)
            - med(frame, "B", "g0", contexts) - med(frame, "BR", "g0", contexts) - med(frame, "BL", "g0", contexts)
        ) / 3
        yaw_delta = (
            med(frame, "R", "g1", contexts) + med(frame, "FR", "g1", contexts) + med(frame, "BR", "g1", contexts)
            - med(frame, "L", "g1", contexts) - med(frame, "FL", "g1", contexts) - med(frame, "BL", "g1", contexts)
        ) / 3
        rows.append(dict(model=model, throttle_delta=throttle_delta,
                         throttle_status=status(throttle_delta), yaw_delta=yaw_delta,
                         yaw_status=status(yaw_delta), median_cosine=float(frame.cosine.median()),
                         direction_failure_pct=100 * float(frame.wrong_direction_60.mean())))

    audit = pd.DataFrame(rows).sort_values("model")
    audit.to_csv(out / "action_convention_audit.csv", index=False)
    reversed_rows = audit[(audit.throttle_status == "reversed") |
                          (audit.yaw_status == "reversed")]
    assert reversed_rows.empty, reversed_rows.to_dict("records")

    # Both regenerated minWM fleets use the common command convention in the
    # runner itself.  Verify every path and sidecar directly: no post-hoc file
    # relabelling or left/right swap is permitted in this evaluation.
    for model in ("minwm", "minwm_ode"):
        subset = manifest[manifest.model == model]
        assert len(subset) == expected_per_model
        for row in subset.itertuples():
            disk_direction = "NOOP" if row.direction == "N" else row.direction
            assert str(row.path).endswith(f"_{disk_direction}.mp4"), (
                row.scene, row.path, disk_direction
            )
            sidecar = Path(str(row.path) + ".json")
            assert sidecar.exists(), sidecar
            meta = json.loads(sidecar.read_text())
            assert meta["direction"] == disk_direction, (sidecar, meta["direction"])
            assert meta["yaw_adapter"] == (
                "released_minwm_yaw_negated_to_common_convention"
            ), sidecar
            if row.direction in ("FR", "BR", "BL", "FL"):
                assert meta["camera_pose_update"] == (
                    "released_rotate_then_local_translate"
                ), sidecar
            assert meta["generation_boundary_real_frame"] == 32, sidecar
            if model == "minwm":
                assert meta["seed_start_frame"] == 4, sidecar
                assert meta["seed_frames"] == 29 and meta["seed_latents"] == 8, sidecar
            else:
                assert meta["stage"] == "ode", sidecar
                assert meta["config"] == "causal_ode_camera.yaml", sidecar
                assert meta["seed_start_frame"] == 20, sidecar
                assert meta["seed_frames"] == 13 and meta["seed_latents"] == 4, sidecar

    if "yume5b" in set(manifest.model):
        subset = manifest[manifest.model == "yume5b"]
        assert len(subset) == expected_per_model
        for row in subset.itertuples():
            assert str(row.path).endswith(f"/yume5b_{row.uid}_{row.direction}.mp4"), (
                row.scene, row.path
            )
            sidecar = Path(str(row.path) + ".json")
            assert sidecar.exists(), sidecar
            meta = json.loads(sidecar.read_text())
            assert meta["model"] == "YUME-5B-720P", sidecar
            assert meta["eval_model_key"] == "yume5b", sidecar
            assert meta["action"] == row.direction, (sidecar, meta["action"])
            observed = (
                meta["yume_keyboard"], meta["yume_mouse_yaw"],
                meta["yume_distance"], meta["yume_turn"], meta["yume_rotation"],
            )
            assert observed == YUME_ACTIONS[row.direction], (
                sidecar, observed, YUME_ACTIONS[row.direction]
            )
            assert meta["context_frames"] == 1, sidecar
            # YUME's provenance sidecar uses the panel-wide canonical/source
            # boundary names rather than the minWM runner's legacy
            # ``generation_boundary_real_frame`` key.  Require the canonical
            # panel boundary and, when present, require the legacy alias to
            # agree; ``source_boundary_frame_index`` is an index in the
            # original source and legitimately differs between datasets.
            assert meta["canonical_boundary_frame"] == 32, sidecar
            if "generation_boundary_real_frame" in meta:
                assert meta["generation_boundary_real_frame"] == 32, sidecar
            assert meta["generation_start_video_frame"] == 1, sidecar

    required_names = [x for x in args.require_positive.split(",") if x]
    required = audit[audit.model.isin(required_names)]
    assert len(required) == len(required_names), (required_names, required.model.tolist())
    assert (required.throttle_status == "correct").all(), required.to_dict("records")
    assert (required.yaw_status == "correct").all(), required.to_dict("records")
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
