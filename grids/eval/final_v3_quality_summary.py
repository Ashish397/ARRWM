"""Merge measured ICLR GPU windows and publish denominator-explicit summaries.

Percentages are emitted only when every selected clip for a model and endpoint
has actually been scored. The long
window style and HF cutoffs are descriptive extensions, not validated AAAI
failure rates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .panel32_control_rule import (
        CONTROL_ORB_STATIC_CUTOFF,
        CONTROL_RULE_ID,
        apply_cardinal_precondition_rule,
    )
except ImportError:  # Direct execution from grids/eval.
    from panel32_control_rule import (  # type: ignore
        CONTROL_ORB_STATIC_CUTOFF,
        CONTROL_RULE_ID,
        apply_cardinal_precondition_rule,
    )

WINDOW_TO_H = {0: 6, 9: 15, 24: 30}


def shards(out, stem):
    paths = sorted(out.glob(f"{stem}_shard*.csv"))
    if not paths:
        return pd.DataFrame()
    d = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    assert not d.duplicated(["scene", "model", "window_start_s"]).any(), stem
    d.to_csv(out / f"{stem}.csv", index=False)
    return d


def json_windows(out, kind):
    rows = []
    for p in sorted((out / kind).glob("*.json")):
        obj = json.loads(p.read_text())
        assert len(obj["rows"]) == 6, p
        for row in obj["rows"]:
            assert row["video_sha256"] == obj["video_sha256"], p
            if "events" in row:
                row["events"] = json.dumps(row["events"])
            if "generated_indices" in row:
                row["generated_indices"] = json.dumps(row["generated_indices"])
            if "real_indices" in row:
                row["real_indices"] = json.dumps(row["real_indices"])
            rows.append(row)
    d = pd.DataFrame(rows)
    if len(d):
        assert not d.duplicated(["scene", "model", "window_start_s"]).any(), kind
        if kind == "geometry":
            if "p_uncanny_raw" not in d:
                d["p_uncanny_raw"] = d.p_uncanny
            else:
                d["p_uncanny_raw"] = d.p_uncanny_raw.fillna(d.p_uncanny)
            d["p_uncanny"] = d.p_uncanny_raw.round(4)
            d["geometry_flag"] = (d.p_uncanny > 0.5).astype(int)
        d.to_csv(out / f"{kind}_windows.csv", index=False)
    return d


def count_and_rate(d, model, horizon, flag, *, expected_all,
                   expected_directional, expected_noop,
                   directional=False, noop=False):
    if d.empty:
        return 0, np.nan, np.nan
    q = d[(d.model == model) & (d.horizon_s == horizon)]
    if directional:
        q = q[q.direction != "N"]
        expected = expected_directional
    elif noop:
        q = q[q.direction == "N"]
        expected = expected_noop
    else:
        expected = expected_all
    q = q[q[flag].notna()]
    if len(q) > expected:
        raise AssertionError((model, horizon, flag, len(q), expected))
    n_bad = int(q[flag].astype(int).sum())
    return len(q), n_bad, 100*n_bad/expected if len(q) == expected else np.nan


def main(out):
    manifest = pd.read_csv(out / "video_manifest.csv")
    assert not manifest.duplicated(["scene", "model"]).any()
    contexts = int(manifest.uid.nunique())
    models = sorted(manifest.model.unique())
    expected_all = contexts * 9
    expected_directional = contexts * 8
    expected_noop = contexts
    family_count = int(manifest.family.nunique())
    assert len(manifest) == len(models) * expected_all
    assert (manifest.groupby("model").size() == expected_all).all()
    ids = manifest[["scene", "uid", "model", "direction"]]
    style = shards(out, "style_windows")
    control = shards(out, "control_windows")
    geometry = json_windows(out, "geometry")
    conj = json_windows(out, "conjuration")
    cpu = pd.read_csv(out / "cpu_endpoints_scored.csv") if (out / "cpu_endpoints_scored.csv").exists() else pd.DataFrame()
    data = {}
    full_data = {}
    for name, d in (("style", style), ("control", control), ("geometry", geometry), ("conjuration", conj)):
        if d.empty:
            data[name] = d
            full_data[name] = d
            continue
        d = d.copy()
        d["horizon_s"] = d.window_start_s.map(WINDOW_TO_H)
        d = d.merge(ids, on=["scene", "model"], how="left", validate="many_to_one",
                    suffixes=("", "_manifest"))
        assert d.direction.notna().all()
        if "direction_manifest" in d:
            assert (d.direction == d.direction_manifest).all(), name
        full_data[name] = d
        data[name] = d[d.horizon_s.notna()].copy()
    if not style.empty:
        d = full_data["style"]
        d["style_flag_072_descriptive"] = (d.drift_from_real > 0.72).astype(int)
        data["style"] = d[d.horizon_s.notna()].copy()
    if not control.empty:
        d = full_data["control"]
        d["noop_motion_010"] = np.where(d.direction == "N", d.magnitude >= 0.1, np.nan)
        if (out / "cpu_windows.csv").exists():
            w = pd.read_csv(out / "cpu_windows.csv")
            d = d.merge(w[["scene", "model", "window_start_s", "local_orb_inliers"]],
                        on=["scene", "model", "window_start_s"], validate="one_to_one")
            if not cpu.empty:
                anchor = cpu[cpu.horizon_s == 6][["scene", "model", "starting_view_inliers"]]
                d = d.merge(anchor, on=["scene", "model"], validate="many_to_one")
                d["control_orb_inliers"] = np.where(d.window_start_s == 0,
                    d.starting_view_inliers, d.local_orb_inliers)
            else:
                d["control_orb_inliers"] = d.local_orb_inliers
            d["direction_failure"] = np.where(
                d.direction != "N", d.wrong_direction_60 == 1, np.nan)
            d["near_static_450"] = np.where(
                d.direction != "N",
                d.control_orb_inliers > CONTROL_ORB_STATIC_CUTOFF,
                np.nan,
            )
            d = apply_cardinal_precondition_rule(d, context_column="uid")
            full_data["control"] = d
            data["control"] = d[d.horizon_s.notna()].copy()
            d.to_csv(out / "control_window_scored.csv", index=False)
    if not cpu.empty:
        cpu = cpu.copy()
        cpu["hf_flag_150_descriptive"] = np.where(cpu.hf_panel_complete,
                                                   cpu.B_v2_iclrfive > 150, np.nan)
        base = ["scene", "model", "direction", "horizon_s", "video_sha256",
                "feature_valid", "active_6s", "starting_view_inliers",
                "base_blur", "end_blur", "d_blur",
                "hf_reference_median", "B_v2_iclrfive", "hf_flag_150_descriptive"]
        endpoints = cpu[base].copy()
        take = {
            "style": ["drift_from_real", "local_adjacent_drift", "style_flag_072_descriptive"],
            "geometry": ["p_uncanny_raw", "p_uncanny", "geometry_flag", "real_reference_policy", "real_sha256"],
            "conjuration": ["conjuration_flag", "positive_events", "top_score", "events"],
            "control": ["uid", "cosine", "magnitude", "wrong_direction_60", "control_orb_inliers",
                        "direction_failure", "near_static_450", "native_control_failure",
                        "diagonal_precondition_failure", "failed_cardinal_components",
                        "control_rule_id", "control_failure", "noop_motion_010"],
        }
        for kind, cols in take.items():
            d = data[kind]
            present = [x for x in cols if x in d.columns]
            if not present:
                continue
            add = d[["scene", "model", "horizon_s", "video_sha256"] + present].copy()
            add = add.rename(columns={"video_sha256": f"video_sha256_{kind}"})
            endpoints = endpoints.merge(add, on=["scene", "model", "horizon_s"],
                                        how="left", validate="one_to_one")
            h = f"video_sha256_{kind}"
            assert (endpoints[h].isna() | (endpoints[h] == endpoints.video_sha256)).all(), kind
        assert len(endpoints) == len(manifest) * 3 and not endpoints.duplicated(["scene", "model", "horizon_s"]).any()
        endpoints.to_csv(out / "quality_endpoints_per_video.csv", index=False)
    summary = []
    for model in models:
        for horizon in (6, 15, 30):
            row = dict(model=model, horizon_s=horizon, listed=expected_all,
                       control_attempted=expected_directional, control_rule_id=CONTROL_RULE_ID)
            for kind, flag in (
                ("style", "style_flag_072_descriptive"),
                ("geometry", "geometry_flag"),
                ("conjuration", "conjuration_flag"),
                ("control", "control_failure"),
            ):
                d = data[kind]
                if flag not in d.columns:
                    n, bad, pct = 0, np.nan, np.nan
                else:
                    n, bad, pct = count_and_rate(
                        d, model, horizon, flag,
                        expected_all=expected_all,
                        expected_directional=expected_directional,
                        expected_noop=expected_noop,
                        directional=(kind == "control"))
                row.update({f"{kind}_scored": n, f"{kind}_flagged": bad,
                            f"{kind}_pct_if_complete": pct})
            control_at_h = data["control"]
            for component in ("direction_failure", "near_static_450"):
                n, bad, pct = count_and_rate(
                    control_at_h, model, horizon, component,
                    expected_all=expected_all,
                    expected_directional=expected_directional,
                    expected_noop=expected_noop, directional=True)
                row.update({f"{component}_scored": n,
                            f"{component}_flagged": bad,
                            f"{component}_pct_if_complete": pct})
            if cpu.empty:
                row.update(hf_scored=0, hf_flagged=np.nan, hf_pct_if_complete=np.nan)
            else:
                n, bad, pct = count_and_rate(cpu, model, horizon,
                                              "hf_flag_150_descriptive",
                                              expected_all=expected_all,
                                              expected_directional=expected_directional,
                                              expected_noop=expected_noop)
                row.update(hf_scored=n, hf_flagged=bad, hf_pct_if_complete=pct)
            row["style_threshold_status"] = "AAAI six-second cutoff" if horizon == 6 else "exploratory on later original-context drift"
            row["hf_threshold_status"] = (
                f"ICLR {family_count}-family panel with "
                "one seat per family; calibration transfer unvalidated"
            )
            row["long_window_status"] = "AAAI anchor" if horizon == 6 else "fixed-six-second endpoint extension"
            summary.append(row)
    pd.DataFrame(summary).to_csv(out / "quality_horizon_summary.csv", index=False)
    stationary = []
    for model in models:
        for horizon in (6, 15, 30):
            row = dict(model=model, horizon_s=horizon, noop_attempted=expected_noop,
                       status="AAAI stationary anchor" if horizon == 6 else "fixed-six-second stationary extension")
            for kind, flag in (("style", "style_flag_072_descriptive"),
                               ("geometry", "geometry_flag"),
                               ("conjuration", "conjuration_flag"),
                               ("control", "noop_motion_010")):
                d = data[kind]
                n, bad, pct = (count_and_rate(
                    d, model, horizon, flag,
                    expected_all=expected_all,
                    expected_directional=expected_directional,
                    expected_noop=expected_noop, noop=True)
                               if flag in d.columns else (0, np.nan, np.nan))
                row.update({f"{kind}_scored": n, f"{kind}_flagged": bad,
                            f"{kind}_pct_if_complete": pct})
                n, bad, pct = (count_and_rate(
                           cpu, model, horizon, "hf_flag_150_descriptive",
                           expected_all=expected_all,
                           expected_directional=expected_directional,
                           expected_noop=expected_noop, noop=True) if not cpu.empty
                           else (0, np.nan, np.nan))
            row.update(hf_scored=n, hf_flagged=bad, hf_pct_if_complete=pct)
            stationary.append(row)
    pd.DataFrame(stationary).to_csv(out / "noop_quality_horizon_summary.csv", index=False)
    # Reproduce the AAAI population rule as a separate audit. Every video is
    # still scored; this file only changes the denominator for comparison with
    # the historical six-second quality tables. The no-op table above is an
    # action-level diagnostic; the main control rate uses the eight
    # directional commands and reports no-op separately above.
    if not cpu.empty:
        population = cpu[(cpu.horizon_s == 6) & (cpu.direction != "N")][
            ["scene", "model", "active_6s", "feature_valid", "active_static_600"]].copy()
        assert len(population) == len(models) * expected_directional
        active_rows = []
        for model in models:
            active = population[(population.model == model) & (population.active_6s == 1)]
            row = dict(model=model, directional_attempted=expected_directional,
                       feature_valid=int(population[population.model == model].feature_valid.sum()),
                       active_scored=len(active),
                       static_excluded=int(population[population.model == model].active_static_600.sum()))
            for kind, flag in (("style", "style_flag_072_descriptive"),
                               ("geometry", "geometry_flag"),
                               ("conjuration", "conjuration_flag")):
                d = data[kind]
                if flag not in d.columns:
                    n, bad = 0, np.nan
                else:
                    matched = active[["scene", "model"]].merge(
                        d[(d.model == model) & (d.horizon_s == 6)][
                            ["scene", "model", flag]],
                        on=["scene", "model"], how="left", validate="one_to_one")
                    n = int(matched[flag].notna().sum())
                    bad = int(matched[flag].fillna(0).astype(int).sum())
                row.update({f"{kind}_scored": n, f"{kind}_flagged": bad,
                            f"{kind}_pct_if_complete":
                            100 * bad / len(active) if n == len(active) and n else np.nan})
            matched = active[["scene", "model"]].merge(
                cpu[(cpu.model == model) & (cpu.horizon_s == 6)][
                    ["scene", "model", "hf_flag_150_descriptive"]],
                on=["scene", "model"], how="left", validate="one_to_one")
            n = int(matched.hf_flag_150_descriptive.notna().sum())
            bad = int(matched.hf_flag_150_descriptive.fillna(0).astype(int).sum())
            row.update(hf_scored=n, hf_flagged=bad,
                       hf_pct_if_complete=100 * bad / len(active) if n == len(active) and n else np.nan,
                       population_rule="directional, feature-valid, ORB static inliers <=600")
            active_rows.append(row)
        pd.DataFrame(active_rows).to_csv(out / "aaai_active_h6_quality_summary.csv", index=False)
    diagnostics = []
    starts_for_h = {6: (0,), 15: (0, 6, 9), 30: (0, 6, 12, 18, 24)}
    for kind, flag in (("style", "style_flag_072_descriptive"),
                       ("geometry", "geometry_flag"),
                       ("conjuration", "conjuration_flag"),
                       ("control", "control_failure")):
        d = full_data[kind]
        if flag not in d.columns:
            continue
        for (scene, model), group in d.groupby(["scene", "model"]):
            direction = group.direction.iloc[0]
            used_flag = flag
            for horizon, starts in starts_for_h.items():
                z = group[group.window_start_s.isin(starts)].sort_values("window_start_s")
                complete = len(z) == len(starts) and z[used_flag].notna().all()
                endpoint = z[z.window_start_s == starts[-1]]
                endpoint_flag = (float(endpoint[used_flag].iloc[0])
                                 if len(endpoint) == 1 and endpoint[used_flag].notna().all()
                                 else np.nan)
                bad = z[z[used_flag] == 1]
                if kind == "control" and direction != "N" and complete and horizon > 6:
                    late = z[z.window_start_s > 0]
                    reversed_ = late[(late.cosine < 0) & (late.magnitude >= 0.1)]
                    late_stop = first_late_stop = np.nan
                    late_reversal = int(len(reversed_) > 0)
                    first_reversal = float(reversed_.window_end_s.min()) if len(reversed_) else np.nan
                else:
                    late_stop = late_reversal = first_late_stop = first_reversal = np.nan
                diagnostics.append(dict(scene=scene, model=model, direction=direction,
                                        metric=kind, horizon_s=horizon,
                                        window_starts_s=",".join(map(str, starts)),
                                        expected_windows=len(starts), scored_windows=int(z[used_flag].notna().sum()),
                                        endpoint_failure=endpoint_flag,
                                        bad_window_fraction=float(z[used_flag].mean()) if complete else np.nan,
                                        first_failure_time_s=float(bad.window_end_s.min()) if complete and len(bad) else np.nan,
                                        ever_failed_by_h=int(len(bad) > 0) if complete else np.nan,
                                        late_stop=late_stop, late_reversal=late_reversal,
                                        first_late_stop_time_s=first_late_stop,
                                        first_reversal_time_s=first_reversal,
                                        late_stop_rule="not scored; ORB cutoff removed",
                                        late_reversal_rule="cosine <0 and magnitude >=0.1",
                                        threshold_status=("cardinal-preconditioned wrong-direction-or-near-static rule" if kind == "control" and horizon == 6 else
                                            "AAAI anchor" if horizon == 6 else
                                            "fixed-six-second extension; style cutoff exploratory" if kind == "style" else
                                            "fixed-six-second extension; ever-failed is exposure dependent")))
    diagnostic_df = pd.DataFrame(diagnostics)
    diagnostic_df.to_csv(out / "quality_horizon_diagnostics.csv", index=False)
    summary_df = pd.DataFrame(summary)
    if len(diagnostic_df):
        ctrl = diagnostic_df[(diagnostic_df.metric == "control") &
                             (diagnostic_df.direction != "N")]
        for i, r in summary_df.iterrows():
            z = ctrl[(ctrl.model == r.model) & (ctrl.horizon_s == r.horizon_s)]
            z = z[z.late_reversal.notna()]
            summary_df.at[i, "late_control_scored"] = len(z)
            summary_df.at[i, "late_reversal_n"] = int(z.late_reversal.sum()) if len(z) else np.nan
            summary_df.at[i, "late_reversal_pct_if_complete"] = (
                100 * z.late_reversal.mean() if len(z) == expected_directional else np.nan)
    summary_df.to_csv(out / "quality_horizon_summary.csv", index=False)
    print("summary rows", len(summary), "metric windows", {k: len(v) for k,v in data.items()})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    main(a.out.resolve())
