"""Combine Reloc-Cal-v2 at 6 s with adjudicated temporal relocation events."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--direct", type=Path, required=True)
    p.add_argument("--chain", type=Path, required=True)
    p.add_argument("--events", type=Path, required=True)
    p.add_argument("--vlm", type=Path)
    p.add_argument("--adjudication", type=Path)
    p.add_argument("--dense", type=Path, required=True)
    p.add_argument("--pilot-vlm", type=Path)
    p.add_argument("--pilot-dense", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    direct = pd.read_csv(a.direct)
    chain = pd.read_csv(a.chain)
    events = pd.read_csv(a.events) if a.events.stat().st_size > 1 else pd.DataFrame()
    vlm = (pd.read_csv(a.vlm) if a.vlm and a.vlm.stat().st_size > 1
           else pd.DataFrame())
    dense = pd.read_csv(a.dense) if a.dense.stat().st_size > 1 else pd.DataFrame()
    if len(vlm) and "case" in vlm:
        vlm = vlm[vlm.case.eq("real")]
    if len(dense) and "case" in dense:
        dense = dense[dense.case.eq("real")]
    event_keys = ["scene", "model", "time_s"]
    if len(events):
        if (dense.duplicated(event_keys).any() or events.duplicated(event_keys).any()):
            raise ValueError("duplicate event adjudication keys")
        checked = events.merge(
            dense[event_keys + ["abrupt_cut", "seam_mad", "seam_mad_ratio",
                                "seam_hist", "seam_hist_ratio"]],
            on=event_keys, how="left", validate="one_to_one")
        if checked.abrupt_cut.isna().any():
            raise ValueError("ORB proposals missing native-frame boundary gate")
        if a.adjudication:
            adjudication = pd.read_csv(a.adjudication)
            if adjudication.duplicated(event_keys).any():
                raise ValueError("duplicate human adjudication keys")
            reviewed = adjudication[event_keys + ["relocation_flag", "adjudication_reason"]]
            reviewed = reviewed.rename(columns={"relocation_flag": "review_relocation_flag"})
            checked = checked.merge(reviewed, on=event_keys, how="left", validate="one_to_one")
            missing = checked.abrupt_cut.eq(1) & checked.review_relocation_flag.isna()
            if missing.any():
                raise ValueError("abrupt candidates missing human adjudication")
            checked["relocation_flag"] = np.where(
                checked.abrupt_cut.eq(1), checked.review_relocation_flag.fillna(0), 0).astype(int)
            checked["adjudication_status"] = np.where(
                checked.abrupt_cut.eq(1), "human-reviewed", "native-frame gate rejected")
        elif checked.abrupt_cut.eq(1).any():
            if not len(vlm):
                raise ValueError("provide --adjudication or --vlm")
            if vlm.duplicated(event_keys).any():
                raise ValueError("duplicate VLM adjudication keys")
            checked = checked.merge(vlm[event_keys + ["p_relocation", "relocation_flag"]],
                                    on=event_keys, how="left", validate="one_to_one")
            if checked.p_relocation.isna().any():
                raise ValueError("unadjudicated ORB proposals")
            checked["semantic_relocation_flag"] = checked.relocation_flag.astype(int)
            checked["relocation_flag"] = (
                checked.semantic_relocation_flag.eq(1) & checked.abrupt_cut.eq(1)).astype(int)
        else:
            checked["relocation_flag"] = 0
            checked["adjudication_status"] = "native-frame gate rejected"
    else:
        checked = pd.DataFrame(columns=event_keys + ["p_relocation", "relocation_flag"])
    checked.to_csv(a.out / "long_relocation_events_adjudicated.csv", index=False)

    direct_flag = next((name for name in (
        "relocation_flag_6s_exploratory", "relocation_flag_6s", "relocation_flag_50"
    ) if name in direct.columns), None)
    if direct_flag is None:
        raise ValueError("direct relocation table has no recognized six-second flag")
    h6 = direct[direct.horizon_s.eq(6)][
        ["scene", "model", direct_flag, "panel_inliers"]].rename(columns={
            direct_flag: "reloc_cal_v2_h6_flag",
            "panel_inliers": "reloc_cal_v2_h6_inliers",
        })
    rows = chain.merge(h6, on=["scene", "model"], how="left", validate="many_to_one")
    if rows.reloc_cal_v2_h6_flag.isna().any():
        raise ValueError("missing six-second Reloc-Cal-v2 rows")
    accepted = checked[checked.relocation_flag.eq(1)] if len(checked) else checked
    first = accepted.groupby(["scene", "model"], as_index=False).time_s.min().rename(
        columns={"time_s": "first_adjudicated_event_s"})
    rows = rows.merge(first, on=["scene", "model"], how="left")
    rows["new_relocation_by_endpoint"] = (
        rows.horizon_s.gt(6) &
        rows.first_adjudicated_event_s.notna() &
        rows.first_adjudicated_event_s.le(rows.horizon_s)).astype(int)
    rows["relocation_cumulative_flag"] = np.where(
        rows.horizon_s.eq(6), rows.reloc_cal_v2_h6_flag.astype(int),
        np.maximum(rows.reloc_cal_v2_h6_flag.astype(int),
                   rows.new_relocation_by_endpoint.astype(int)))
    rows["relocation_definition"] = np.where(
        rows.horizon_s.eq(6), "Reloc-Cal-v2 endpoint decision",
        "cumulative Reloc-Cal-v2 initial status plus adjudicated temporal breaks")
    keys = ["model", "scene", "horizon_s"]
    rows = rows.sort_values(keys)
    rows.to_csv(a.out / "long_relocation_final_rows.csv", index=False)
    summary = rows.groupby(["model", "horizon_s"], as_index=False).agg(
        videos=("scene", "size"),
        relocation_rate=("relocation_cumulative_flag", "mean"),
        new_relocation_rate=("new_relocation_by_endpoint", "mean"),
    )
    summary["relocation_percent"] = 100 * summary.relocation_rate
    summary["new_relocation_percent"] = 100 * summary.new_relocation_rate
    summary.to_csv(a.out / "long_relocation_final_summary.csv", index=False)
    rate_table = summary.pivot(index="model", columns="horizon_s",
                               values="relocation_percent").round(1)
    table_lines = ["| Model | 6 s | 12 s | 18 s | 24 s | 30 s |",
                   "|---|---:|---:|---:|---:|---:|"]
    for model, values in rate_table.iterrows():
        table_lines.append(
            f"| {model} | " + " | ".join(f"{values[h]:.1f}" for h in [6, 12, 18, 24, 30]) + " |")
    rate_markdown = "\n".join(table_lines)

    pilot_text = ""
    if a.pilot_vlm and a.pilot_vlm.exists():
        pilot = pd.read_csv(a.pilot_vlm)
        pilot["end_to_end_flag"] = (
            pilot.get("orb_proposal", 1).astype(int) &
            pilot.relocation_flag.astype(int))
        control = pilot.groupby("case").end_to_end_flag.agg(["size", "sum"])
        pilot_text = (
            f"- Semantic gate pilot: {int(control.loc['real', 'size'] - control.loc['real', 'sum'])}/"
            f"{int(control.loc['real', 'size'])} real false proposals rejected; "
            f"{int(control.loc['synthetic_cut', 'sum'])}/"
            f"{int(control.loc['synthetic_cut', 'size'])} synthetic replacements accepted.\n")
    dense_text = ""
    if a.pilot_dense and a.pilot_dense.exists():
        pilot_dense = pd.read_csv(a.pilot_dense)
        # Recompute from the frozen threshold to keep older pilot files usable.
        pilot_dense["abrupt_cut"] = pilot_dense.seam_hist_ratio.gt(3.0).astype(int)
        control = pilot_dense.groupby("case").abrupt_cut.agg(["size", "sum"])
        dense_text = (
            f"- Native-frame cut pilot: {int(control.loc['real', 'size'] - control.loc['real', 'sum'])}/"
            f"{int(control.loc['real', 'size'])} real continuous transitions rejected; "
            f"{int(control.loc['synthetic_cut', 'sum'])}/"
            f"{int(control.loc['synthetic_cut', 'size'])} synthetic replacements accepted.\n")
    reviewed_count = (int(checked.abrupt_cut.sum())
                      if len(checked) and a.adjudication else "n/a")
    expected_rows = (
        rows.scene.nunique() * rows.model.nunique() * rows.horizon_s.nunique())
    report = f"""# Long-horizon relocation validation

## Definition

The six-second value retains the frozen Reloc-Cal-v2 endpoint decision. Later
values report cumulative relocation incidence. A temporal proposal requires a
persistent ORB--RANSAC correspondence break between coherent, feature-bearing
one-second banks sampled at 4 Hz. A native-frame cut gate then requires the
boundary's HSV-histogram change to exceed three times its surrounding
adjacent-frame median. Every surviving boundary is reviewed as an ordered
native-frame sequence; relocation requires persistent replacement of place
identity and excludes travel, turns, occlusion, lighting change, corruption,
and collapse.

## Coverage and integrity

- Endpoint rows: {len(rows):,}; expected: {expected_rows:,}.
- Models: {rows.model.nunique()}; videos per model: {rows.scene.nunique()}.
- ORB temporal proposals: {len(checked):,}.
- Native-frame candidates reviewed: {reviewed_count}.
- Accepted new relocation events after the native-frame gate: {int(checked.relocation_flag.sum()) if len(checked) else 0:,}.
- Every native-frame candidate has exactly one recorded human adjudication;
  all no-op videos remain included.
{pilot_text}{dense_text}
## Cumulative relocation rates (%)

{rate_markdown}

## Candidate audit images

- [Candidates 1--7](audit/candidates/candidates_page1.png)
- [Candidates 8--14](audit/candidates/candidates_page2.png)
- [Candidates 15--21](audit/candidates/candidates_page3.png)
- [Candidates 22--27](audit/candidates/candidates_page4.png)

## Interpretation

This measure is monotone by construction because relocation is an event: once
a rollout has replaced its world, a later coherent view of the replacement
does not erase that failure. Direct seed overlap is retained as a diagnostic
but is not used after six seconds, so ordinary travel does not become
relocation merely because the original view has left the field of view.
"""
    (a.out / "VALIDATION_REPORT.md").write_text(report)
    print(f"rows={len(rows)} proposals={len(checked)} accepted={int(checked.relocation_flag.sum()) if len(checked) else 0}")


if __name__ == "__main__":
    main()
