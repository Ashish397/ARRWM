"""Build the paper evaluation dataset with the validated 29-frame minWM run.

The 2026-09-17 full evaluation remains immutable.  This script copies the
small scored tables used by the paper, replaces only the minWM DMD producer
rows, and recomputes the five-family HF panel because minWM occupies one of
its frozen family seats.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from final_v2_iclr import FAMILY, SEAT


COMMAND = {
    "F": (1.0, 0.0), "FR": (1.0, 1.0), "R": (0.0, 1.0),
    "BR": (-1.0, 1.0), "B": (-1.0, 0.0), "BL": (-1.0, -1.0),
    "L": (0.0, -1.0), "FL": (1.0, -1.0),
}


def correct_minwm_yaw_convention(frame: pd.DataFrame) -> pd.DataFrame:
    """Map released minWM yaw into the paper's left/right convention.

    The generation runner used the opposite camera-yaw sign.  Existing L/R
    files therefore contain R/L motion (and likewise for diagonals).  Flipping
    the lateral readout is exactly equivalent to swapping those paired files
    for control scoring, while leaving no-op and longitudinal motion intact.
    """
    out = frame.copy()
    if "control_failure" in out:
        out["control_failure"] = out["control_failure"].astype(int)
    affected = out.model.astype(str).str.startswith("minwm") & out.direction.isin(COMMAND)
    out.loc[affected, "g1"] = -out.loc[affected, "g1"].astype(float)
    vectors = np.asarray([COMMAND.get(d, (np.nan, np.nan)) for d in out.direction], float)
    motion = out[["g0", "g1"]].to_numpy(float)
    valid = affected.to_numpy()
    cosine = np.sum(vectors[valid] * motion[valid], axis=1) / (
        np.linalg.norm(vectors[valid], axis=1) * np.linalg.norm(motion[valid], axis=1) + 1e-12)
    out.loc[affected, "cosine"] = cosine
    out.loc[affected, "wrong_direction_60"] = (cosine < 0.5).astype(int)
    if "control_failure" in out:
        out.loc[affected, "control_failure"] = (cosine < 0.5).astype(int)
    out["control_convention_correction"] = np.where(
        affected, "minwm_yaw_sign_flipped", "none")
    return out


def flatten_json(directory: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(directory.glob("*.json")):
        rows.extend(json.loads(path.read_text())["rows"])
    return pd.DataFrame(rows)


def aligned(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column not in out:
            out[column] = np.nan
    return out[columns]


def replace_model(base: pd.DataFrame, replacement: pd.DataFrame,
                  model: str = "minwm") -> pd.DataFrame:
    replacement = replacement.copy()
    replacement["model"] = model
    replacement = aligned(replacement, list(base.columns))
    out = pd.concat([base[base.model != model], replacement], ignore_index=True)
    return out


def recompute_endpoint_hf(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    lookup = data.set_index(["scene", "model", "horizon_s"]).d_blur.to_dict()
    refs, scores = [], []
    for r in data.itertuples(index=False):
        seats = {family: lookup[(r.scene, seat, r.horizon_s)]
                 for family, seat in SEAT.items()}
        seats[FAMILY[r.model]] = float(r.d_blur)
        reference = float(np.median(list(seats.values())))
        refs.append(reference)
        scores.append(reference - float(r.d_blur))
    data["hf_reference_median"] = refs
    data["B_v2_iclrfive"] = scores
    data["hf_panel_complete"] = True
    data["hf_panel_flag_150_exploratory"] = (data.B_v2_iclrfive > 150).astype(int)
    return data


def build(base: Path, sensitivity: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(sensitivity / "video_manifest.csv")
    direction = manifest.set_index("scene").direction.to_dict()

    # Control: the paper rule is cosine-only for directional actions and
    # magnitude-only for no-op.  ORB diagnostics are deliberately excluded.
    control_base = correct_minwm_yaw_convention(
        pd.read_csv(base / "control_window_scored.csv"))
    control_new = correct_minwm_yaw_convention(
        pd.read_csv(sensitivity / "control_windows.csv"))
    is_directional = control_new.direction != "N"
    control_new["control_failure"] = np.where(
        is_directional,
        control_new.wrong_direction_60.fillna(0),
        control_new.noop_motion_010.fillna(0)).astype(int)
    control_new["control_rule_id"] = "cosine_only_plus_noop_magnitude"
    control_new["direction_manifest"] = control_new.direction
    control_new["horizon_s"] = control_new.window_end_s
    control = replace_model(control_base, control_new)
    assert len(control) == 15 * 288 * 6
    control.to_csv(dest / "control_window_scored.csv", index=False)

    style_base = pd.read_csv(base / "style_windows.csv")
    style_new = pd.read_csv(sensitivity / "style_windows.csv")
    style = replace_model(style_base, style_new)
    assert len(style) == 15 * 288 * 6
    style.to_csv(dest / "style_windows.csv", index=False)

    geometry_base = pd.read_csv(base / "geometry_windows.csv")
    geometry_new = flatten_json(sensitivity / "geometry")
    geometry = replace_model(geometry_base, geometry_new)
    assert len(geometry) == 15 * 288 * 6
    geometry.to_csv(dest / "geometry_windows.csv", index=False)

    conj_base = pd.read_csv(base / "conjuration_windows.csv")
    conj_new = flatten_json(sensitivity / "conjuration")
    conjuration = replace_model(conj_base, conj_new)
    assert len(conjuration) == 15 * 288 * 6
    conjuration.to_csv(dest / "conjuration_windows.csv", index=False)

    hf_base = pd.read_csv(base / "hf_window_trajectory.csv")
    cpu_windows = pd.read_csv(sensitivity / "cpu_windows.csv")
    hf_new = cpu_windows[["scene", "model", "window_start_s", "window_end_s",
                          "end_blur", "d_blur_from_early_base", "video_sha256"]].copy()
    hf_new["direction"] = hf_new.scene.map(direction)
    hf_new["early_base_blur"] = hf_new.end_blur - hf_new.d_blur_from_early_base
    hf_new["hf_retention_ratio"] = hf_new.end_blur / hf_new.early_base_blur
    hf = replace_model(hf_base, hf_new)
    assert len(hf) == 15 * 288 * 6
    hf.to_csv(dest / "hf_window_trajectory.csv", index=False)

    endpoint_base = pd.read_csv(base / "cpu_endpoints_scored.csv")
    endpoint_new = pd.read_csv(sensitivity / "cpu_endpoints.csv")
    endpoint_new["ctx"] = 29
    endpoint_new["family"] = "minwm"
    endpoint_new["hf_drop_from_early_base"] = endpoint_new.d_blur
    endpoint_new["hf_retention_ratio"] = endpoint_new.end_blur / endpoint_new.base_blur
    endpoint_new["feature_valid"] = True
    endpoint_new["active_6s"] = np.nan
    endpoints = replace_model(endpoint_base, endpoint_new)
    endpoints = recompute_endpoint_hf(endpoints)
    assert len(endpoints) == 15 * 288 * 3
    endpoints.to_csv(dest / "cpu_endpoints_scored.csv", index=False)

    manifest_base = pd.read_csv(base / "video_manifest.csv")
    manifest_new = manifest.copy()
    manifest_new["model"] = "minwm"
    paper_manifest = replace_model(manifest_base, manifest_new)
    assert len(paper_manifest) == 15 * 288
    paper_manifest.to_csv(dest / "video_manifest.csv", index=False)

    provenance = {
        "base_evaluation": str(base.resolve()),
        "replacement_evaluation": str(sensitivity.resolve()),
        "replacement": "minWM DMD: 13 conditioning pixel frames -> 29 pixel frames (two native blocks)",
        "videos_replaced": 288,
        "hf_policy": "Recomputed all candidate panels after replacing the minWM family seat",
        "primary_model_name": "minwm",
    }
    (dest / "DERIVATION.json").write_text(json.dumps(provenance, indent=2) + "\n")
    run_report = (sensitivity / "VALIDATION_REPORT.md").read_text()
    run_report = run_report.replace(
        "- Status: separate sensitivity analysis; the primary 13-frame minWM result was not overwritten.",
        "- Status: adopted as the primary minWM DMD evaluation for the ICLR paper; the archived 13-frame run remains unchanged.")
    run_report = run_report.replace(
        "## Change from the primary 13-frame minWM run",
        "## Change from the previous 13-frame minWM run")
    (dest / "MINWM_SEED29_RUN_VALIDATION.md").write_text(run_report)
    print("built", dest)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--sensitivity", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True)
    a = p.parse_args()
    build(a.base.resolve(), a.sensitivity.resolve(), a.dest.resolve())


if __name__ == "__main__":
    main()
