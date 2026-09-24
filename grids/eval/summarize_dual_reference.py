"""Summarize the primary seed geometry and dual-reference diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEY = ["scene", "model", "window_start_s"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old-out", required=True, type=Path)
    p.add_argument("--dual-out", required=True, type=Path)
    a = p.parse_args()
    style = pd.read_csv(a.dual_out / "style_full.csv")
    geometry = pd.read_csv(a.dual_out / "geometry_full.csv")
    seed_geometry = pd.read_csv(a.old_out / "geometry_windows.csv")
    style = style[style.condition == "baseline"].copy()
    geometry = geometry[geometry.condition == "baseline"].copy()
    assert len(style) == len(geometry) == 21600
    assert not style.duplicated(KEY).any() and not geometry.duplicated(KEY).any()
    seed_geometry = seed_geometry[seed_geometry.window_start_s.isin([0, 6, 12, 18, 24])]
    geometry = geometry.merge(
        seed_geometry[KEY + ["p_uncanny", "geometry_flag"]].rename(columns={
            "p_uncanny": "p_geometry_seed_conditioned_existing",
            "geometry_flag": "geometry_seed_conditioned_flag_existing",
        }), on=KEY, validate="one_to_one")
    geometry["geometry_absolute_flag"] = (geometry.p_geometry_absolute > 0.5).astype(int)
    geometry["geometry_rolling_break_flag"] = (geometry.p_geometry_rolling_break > 0.5).astype(int)
    geometry["geometry_local_or_boundary_diagnostic"] = (
        geometry.geometry_absolute_flag | geometry.geometry_rolling_break_flag
    ).astype(int)
    style["endpoint_s"] = style.window_end_s.astype(int)
    geometry["endpoint_s"] = geometry.window_end_s.astype(int)
    ssum = style.groupby(["model", "endpoint_s"], as_index=False).agg(
        seed_dino_drift=("seed_dino_drift", "mean"),
        rolling_dino_drift=("rolling_dino_drift", "mean"),
        seam_dino_drift=("seam_dino_drift", "mean"),
        within_window_dino_drift=("within_window_dino_drift", "mean"),
        seed_style_failure=("seed_dino_drift", lambda x: float((x > 0.72).mean())),
        scored=("scene", "size"),
    )
    gsum = geometry.groupby(["model", "endpoint_s"], as_index=False).agg(
        seed_conditioned_probability=("p_geometry_seed_conditioned_existing", "mean"),
        absolute_probability=("p_geometry_absolute", "mean"),
        rolling_break_probability=("p_geometry_rolling_break", "mean"),
        seed_conditioned_failure=("geometry_seed_conditioned_flag_existing", "mean"),
        absolute_failure=("geometry_absolute_flag", "mean"),
        rolling_break_failure=("geometry_rolling_break_flag", "mean"),
        local_or_boundary_diagnostic=("geometry_local_or_boundary_diagnostic", "mean"),
        scored=("scene", "size"),
    )
    assert ssum.scored.eq(288).all() and gsum.scored.eq(288).all()
    style.to_csv(a.dual_out / "style_dual_per_video.csv", index=False)
    geometry.to_csv(a.dual_out / "geometry_dual_per_video.csv", index=False)
    ssum.to_csv(a.dual_out / "style_dual_summary.csv", index=False)
    gsum.to_csv(a.dual_out / "geometry_dual_summary.csv", index=False)
    print("validated", len(style), len(geometry), len(ssum), len(gsum))


if __name__ == "__main__":
    main()
