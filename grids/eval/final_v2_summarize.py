"""Build denominator-explicit, provenance-aware summaries from measured rows.

Never fills a missing component from old pooled output. Long-horizon thresholds
are labelled exploratory; no 30 s composite legitimacy rate is produced.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from final_v2_iclr import FAMILY, MODELS, SEAT


def merged_csv(out, stem):
    paths = sorted(out.glob(f"{stem}_shard*.csv"))
    if not paths:
        return pd.read_csv(out / f"{stem}.csv")
    d = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    key = ["scene", "model", "horizon_s" if stem == "cpu_endpoints" else "window_start_s"]
    assert not d.duplicated(key).any(), stem
    d.to_csv(out / f"{stem}.csv", index=False)
    return d


def panel_b(e):
    """Frozen family seats; each candidate occupies its own family seat."""
    lookup = e.set_index(["scene", "model", "horizon_s"]).d_blur.to_dict()
    refs, rows = [], []
    for horizon in (6, 15, 30):
        for scene in sorted(e.scene.unique()):
            base = {family: lookup.get((scene, model, horizon)) for family, model in SEAT.items()}
            for family, val in base.items():
                refs.append(dict(scene=scene, horizon_s=horizon, family=family,
                                 seat_model=SEAT[family], seat_d_blur=val))
            sub = e[(e.scene == scene) & (e.horizon_s == horizon)]
            for r in sub.itertuples():
                seats = base.copy()
                seats[FAMILY[r.model]] = r.d_blur
                valid = all(v is not None and np.isfinite(v) for v in seats.values())
                med = float(np.median(list(seats.values()))) if valid else np.nan
                rows.append(dict(scene=scene, model=r.model, horizon_s=horizon,
                                 hf_reference_median=med, B_v2_iclrfive=med-r.d_blur if valid else np.nan,
                                 hf_panel_complete=valid, hf_panel_members=json.dumps(seats, sort_keys=True)))
    return pd.DataFrame(rows), pd.DataFrame(refs)


def summarize(out):
    m = pd.read_csv(out / "video_manifest.csv")
    contexts = int(m.uid.nunique())
    expected_all = contexts * 9
    expected_directional = contexts * 8
    expected_noop = contexts
    e = merged_csv(out, "cpu_endpoints")
    w = merged_csv(out, "cpu_windows")
    assert e.video_sha256.notna().all() and w.video_sha256.notna().all()
    assert not e.duplicated(["scene", "model", "horizon_s"]).any()
    assert not w.duplicated(["scene", "model", "window_start_s"]).any()
    b, refs = panel_b(e)
    b.to_csv(out / "hf_panel_rows.csv", index=False)
    refs.to_csv(out / "hf_frozen_seats.csv", index=False)
    e = e.merge(b.drop(columns="hf_panel_members"), on=["scene", "model", "horizon_s"], validate="one_to_one")
    e["hf_drop_from_early_base"] = e.base_blur - e.end_blur
    e["hf_retention_ratio"] = np.where(e.base_blur > 0, e.end_blur / e.base_blur, np.nan)
    e["feature_valid"] = True  # selected sources passed the context-feature gate
    e["active_6s"] = np.where((e.horizon_s == 6) & (e.direction != "N"),
                              e.feature_valid & (e.active_static_600.fillna(0) == 0), np.nan)
    e["hf_panel_flag_150_exploratory"] = np.where((e.horizon_s == 6) &
                                                (e.direction != "N") & e.hf_panel_complete,
                                                e.B_v2_iclrfive > 150, np.nan)
    e.to_csv(out / "cpu_endpoints_scored.csv", index=False)
    e[["scene", "model", "direction", "horizon_s", "base_blur", "end_blur", "d_blur",
       "hf_drop_from_early_base", "hf_retention_ratio", "video_sha256"]].to_csv(
           out / "hf_within_video.csv", index=False)
    traj = w[["scene", "model", "direction", "window_start_s", "window_end_s",
              "end_blur", "d_blur_from_early_base", "video_sha256"]].copy()
    traj["early_base_blur"] = traj.end_blur - traj.d_blur_from_early_base
    traj["hf_retention_ratio"] = np.where(traj.early_base_blur > 0,
                                           traj.end_blur / traj.early_base_blur, np.nan)
    traj.to_csv(out / "hf_window_trajectory.csv", index=False)
    summary = []
    for horizon in (6, 15, 30):
        for model in MODELS:
            full = m[m.model == model]
            sub = e[(e.model == model) & (e.horizon_s == horizon)]
            directional = sub[sub.direction != "N"]
            active = directional[directional.active_6s == 1] if horizon == 6 else pd.DataFrame()
            panel = directional[directional.hf_panel_complete]
            summary.append(dict(horizon_s=horizon, model=model,
                                listed=len(full), local_video=int(full.local_video.sum()),
                                available_video=int(full.available_video.sum()) if "available_video" in full else int(full.local_video.sum()),
                                remote_video_verified=int(full.remote_video_verified.sum()) if "remote_video_verified" in full else 0,
                                decoded=int(full.decoded_frames.notna().sum()),
                                cpu_scored=len(sub), directional_attempted=expected_directional,
                                directional_scored=len(directional), noop_attempted=expected_noop,
                                noop_cpu_scored=int((sub.direction == "N").sum()),
                                feature_valid_scored=len(directional),
                                active_scored=len(active) if horizon == 6 else np.nan,
                                control_near_static_500_n=int(directional.control_near_static_500.sum()) if horizon == 6 else np.nan,
                                starting_view_overlap_median=float(directional.starting_view_inliers.median()) if len(directional) else np.nan,
                                d_blur_median=float(directional.d_blur.median()) if len(directional) else np.nan,
                                hf_retention_ratio_median=float(directional.hf_retention_ratio.median()) if len(directional) else np.nan,
                                hf_panel_scored=len(panel),
                                B_median=float(panel.B_v2_iclrfive.median()) if len(panel) else np.nan,
                                hf_over_150_exploratory_n=int((panel.B_v2_iclrfive > 150).sum()) if horizon == 6 else np.nan,
                                hf_threshold_status=(
                                    f"No within-video cutoff; {len(SEAT)}-family "
                                    "B>150 exploratory only"
                                    if horizon == 6 else
                                    "no validated long-horizon cutoff"
                                )))
    pd.DataFrame(summary).to_csv(out / "summary_by_horizon.csv", index=False)
    if (out / "control_windows.csv").exists():
        c = pd.read_csv(out / "control_windows.csv")
        local = w[["scene", "model", "window_start_s", "local_orb_inliers"]]
        c = c.merge(local, on=["scene", "model", "window_start_s"], how="left", validate="one_to_one")
        anchor_orb = e[e.horizon_s == 6][["scene", "model", "starting_view_inliers"]]
        c = c.merge(anchor_orb, on=["scene", "model"], how="left", validate="many_to_one")
        c["control_orb_inliers"] = np.where(c.window_start_s == 0,
                                             c.starting_view_inliers, c.local_orb_inliers)
        c["control_fail_window_exploratory"] = np.where(c.direction != "N",
             (c.wrong_direction_60 == 1) | (c.control_orb_inliers > 500), np.nan)
        c.to_csv(out / "control_window_diagnostics.csv", index=False)
        rows = []
        for (scene, model), g in c[c.direction != "N"].groupby(["scene", "model"]):
            for horizon in (6, 15, 30):
                starts = {6: [0], 15: [0, 6, 9], 30: [0, 6, 12, 18, 24]}[horizon]
                z = g[g.window_start_s.isin(starts)].sort_values("window_start_s")
                bad = z[z.control_fail_window_exploratory == 1]
                rows.append(dict(scene=scene, model=model, horizon_s=horizon,
                                 windows_scored=len(z), endpoint_failure=(float(z.iloc[-1].control_fail_window_exploratory) if len(z)==len(starts) else np.nan),
                                 bad_window_fraction=float(z.control_fail_window_exploratory.mean()) if len(z)==len(starts) else np.nan,
                                 first_failure_time_s=float(bad.window_end_s.min()) if len(bad) else np.nan,
                                 ever_failed_by_h=int(len(bad)>0) if len(z)==len(starts) else np.nan,
                                 status="exploratory; window exposure differs by horizon"))
        pd.DataFrame(rows).to_csv(out / "control_horizon_diagnostics.csv", index=False)
    return len(e), len(w)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    print("scored rows", summarize(a.out.resolve()))
