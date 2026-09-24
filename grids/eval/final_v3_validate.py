"""Final coverage, key, hash, window, and provenance checks for ICLR v3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from .panel32_control_rule import (
        CONTROL_ORB_STATIC_CUTOFF, CONTROL_RULE_ID, DIAGONAL_COMPONENTS,
    )
except ImportError:  # Direct execution from grids/eval.
    from panel32_control_rule import (  # type: ignore
        CONTROL_ORB_STATIC_CUTOFF, CONTROL_RULE_ID, DIAGONAL_COMPONENTS,
    )

WINDOWS = {0, 6, 9, 12, 18, 24}


def check_windows(out, name, hashes, manifest, expected_videos):
    d = pd.read_csv(out / f"{name}_windows.csv")
    assert len(d) == expected_videos * 6, (name, len(d))
    assert not d.duplicated(["scene", "model", "window_start_s"]).any(), name
    assert set(d.window_start_s) == WINDOWS
    assert (d.groupby(["scene", "model"]).size() == 6).all(), name
    q = d[d.window_start_s == 0].set_index(["scene", "model"])
    assert len(q) == expected_videos
    assert (q.video_sha256 == hashes.loc[q.index]).all(), f"{name} video hashes"
    if name == "geometry":
        assert d.p_uncanny.between(0, 1).all()
        for r in d.itertuples():
            indices = json.loads(r.generated_indices)
            assert len(indices) == 16 and min(indices) >= 0
            assert max(indices) < int(manifest.loc[(r.scene, r.model), "decoded_frames"])
            ctx = int(manifest.loc[(r.scene, r.model), "context_frames"])
            expected_real = [32, 32, 32, 32] if ctx == 1 else [20, 24, 28, 32]
            assert json.loads(r.real_indices) == expected_real, (
                r.scene, r.model, r.real_indices, expected_real
            )
            assert r.geometry_flag == int(r.p_uncanny > 0.5)
    elif name == "style":
        assert d.drift_from_real.notna().all()
    elif name == "control":
        assert d.magnitude.notna().all()
        directional = d[d.direction != "N"]
        assert (directional.cosine.notna() | (directional.magnitude < 1e-12)).all()
    elif name == "conjuration":
        assert d.conjuration_flag.isin([0, 1]).all()
        assert d.events.notna().all()
    return len(d)


def main(out):
    m = pd.read_csv(out / "video_manifest.csv")
    assert len(m) > 0 and m.local_video.all()
    assert not m.duplicated(["scene", "model"]).any()
    contexts = int(m.uid.nunique())
    models = sorted(m.model.unique())
    expected_per_model = contexts * 9
    expected_directional = contexts * 8
    expected_noop = contexts
    expected_videos = len(models) * expected_per_model
    assert len(m) == expected_videos
    assert (m.groupby("model").size() == expected_per_model).all()
    assert m.decoded_frames.notna().all()
    assert (m.decoded_frames == m.container_frames).all()
    assert (m.generated_duration_s >= 30).all()
    assert (m.groupby("model").context_frames.nunique() == 1).all()
    expected_ctx = {"lingbot": 1, "dreamx": 1, "yume5b": 1, "matrixgame2": 1,
                    "minwm": 29, "minwm_ode": 13,
                    "ours_kl4rung": 33, "ours_mse4rung": 33}
    for model, ctx in expected_ctx.items():
        if model not in models:
            continue
        assert set(m.loc[m.model == model, "context_frames"]) == {ctx}, (model, ctx)
    e = pd.read_csv(out / "cpu_endpoints_scored.csv")
    w = pd.read_csv(out / "cpu_windows.csv")
    assert len(e) == expected_videos * 3 and len(w) == expected_videos * 6
    assert not e.duplicated(["scene", "model", "horizon_s"]).any()
    assert not w.duplicated(["scene", "model", "window_start_s"]).any()
    assert e.hf_panel_complete.all()
    hashes = e[e.horizon_s == 6].set_index(["scene", "model"]).video_sha256
    assert len(hashes) == expected_videos and hashes.notna().all()
    indexed = m.set_index(["scene", "model"])
    rows = {name: check_windows(out, name, hashes, indexed, expected_videos) for name in
            ("style", "geometry", "conjuration", "control")}
    q = pd.read_csv(out / "quality_horizon_summary.csv")
    assert len(q) == len(models) * 3
    assert (q.control_rule_id == CONTROL_RULE_ID).all()
    for name, expected in (("style", expected_per_model),
                           ("geometry", expected_per_model),
                           ("conjuration", expected_per_model),
                           ("hf", expected_per_model),
                           ("control", expected_directional)):
        assert (q[f"{name}_scored"] == expected).all(), name
    per = pd.read_csv(out / "quality_endpoints_per_video.csv")
    assert len(per) == expected_videos * 3 and not per.duplicated(["scene", "model", "horizon_s"]).any()
    assert per.near_static_450.notna().sum() == len(models) * expected_directional * 3
    for col in ("hf_flag_150_descriptive", "style_flag_072_descriptive",
                "geometry_flag", "conjuration_flag"):
        assert per[col].notna().all(), col
    assert per.control_failure.notna().all()
    assert (per.control_rule_id == CONTROL_RULE_ID).all()
    move = per[per.direction != "N"]
    native_move = (move.wrong_direction_60 == 1) | (move.near_static_450 == 1)
    assert (move.native_control_failure == native_move).all()
    cardinal = move[move.direction.isin(("F", "B", "L", "R"))]
    assert (cardinal.control_failure == cardinal.native_control_failure).all()
    assert (cardinal.diagonal_precondition_failure == 0).all()
    for diagonal, components in DIAGONAL_COMPONENTS.items():
        diagonal_rows = move[move.direction == diagonal]
        for row in diagonal_rows.itertuples(index=False):
            component_rows = move[
                (move.uid == row.uid)
                & (move.model == row.model)
                & (move.horizon_s == row.horizon_s)
                & move.direction.isin(components)
            ]
            assert len(component_rows) == 2
            precondition = int(component_rows.native_control_failure.astype(int).any())
            assert row.diagonal_precondition_failure == precondition
            assert row.control_failure == int(
                bool(row.native_control_failure) or bool(precondition))
    noop = per[per.direction == "N"]
    assert len(noop) == len(models) * expected_noop * 3
    assert noop.noop_motion_010.notna().all()
    assert (noop.control_failure == noop.noop_motion_010).all()
    cw = pd.read_csv(out / "control_window_scored.csv")
    assert len(cw) == expected_videos * 6
    assert (cw.control_rule_id == CONTROL_RULE_ID).all()
    cw_move = cw[cw.direction != "N"]
    expected_cw_native = (
        (cw_move.wrong_direction_60 == 1) | (cw_move.near_static_450 == 1)
    )
    assert (cw_move.native_control_failure == expected_cw_native).all()
    cw_cardinal = cw_move[cw_move.direction.isin(("F", "B", "L", "R"))]
    assert (cw_cardinal.control_failure == cw_cardinal.native_control_failure).all()
    for diagonal, components in DIAGONAL_COMPONENTS.items():
        diagonal_rows = cw_move[cw_move.direction == diagonal]
        for row in diagonal_rows.itertuples(index=False):
            component_rows = cw_move[
                (cw_move.uid == row.uid)
                & (cw_move.model == row.model)
                & (cw_move.window_start_s == row.window_start_s)
                & cw_move.direction.isin(components)
            ]
            assert len(component_rows) == 2
            precondition = int(component_rows.native_control_failure.astype(int).any())
            assert row.diagonal_precondition_failure == precondition
            assert row.control_failure == int(
                bool(row.native_control_failure) or bool(precondition))
    pooled = move.groupby(["model", "horizon_s"]).control_failure.agg(["size", "sum"])
    reported = q.set_index(["model", "horizon_s"])
    assert (pooled["size"] == expected_directional).all()
    assert (reported.loc[pooled.index, "control_flagged"] == pooled["sum"]).all()
    diag = pd.read_csv(out / "quality_horizon_diagnostics.csv")
    assert len(diag) == 4 * expected_videos * 3
    assert (diag.scored_windows == diag.expected_windows).all()
    relocation = pd.read_csv(out / "relocation_panel_rows.csv")
    # Direct relocation is calibrated only at the six-second anchor.  Later
    # relocation is handled by the temporal break detector and adjudication.
    assert len(relocation) == expected_videos
    assert not relocation.duplicated(["scene", "model", "horizon_s"]).any()
    relocation_h6 = relocation[relocation.horizon_s == 6]
    assert len(relocation_h6) == expected_videos
    assert relocation_h6.relocation_flag_6s_exploratory.isin([0, 1]).all()
    relocation_hashes = relocation_h6.set_index(["scene", "model"]).video_sha256
    assert (relocation_hashes == hashes.loc[relocation_hashes.index]).all()
    alignment = json.loads((out / "context_alignment_rows.json").read_text())
    assert alignment["status"] == "pass"
    assert alignment["videos"] == expected_videos
    assert alignment["models"] == len(models)
    assert alignment["common_generation_boundary_real_frame"] == 32
    report = dict(videos=len(m), decoded=len(m), cpu_endpoints=len(e),
                  cpu_windows=len(w), gpu_windows=rows, horizon_summaries=len(q),
                  per_video_endpoints=len(per), diagnostic_rows=len(diag),
                  relocation_rows=len(relocation),
                  context_alignment_rows=alignment["videos"],
                  control_rule_id=CONTROL_RULE_ID,
                  control_orb_cutoff=CONTROL_ORB_STATIC_CUTOFF,
                  checks="passed")
    (out / "validation_final.json").write_text(json.dumps(report, indent=2))
    print(report)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    main(a.out.resolve())
