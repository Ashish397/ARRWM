"""Render the three provisional ICLR horizon tables from measured local rows.

Only CPU measurements with verified local video provenance enter these tables.
The script refuses missing or duplicate model/horizon rows so a partial merge
cannot silently create a paper table.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


GROUPS = (
    ("Main comparison", (
        ("ours_recovery_base", "Ours (recovery base)"),
        ("lingbot", "LingBot-World-V2 1.3B"),
        ("dreamx", "DreamX-World 5B"),
        ("matrixgame2", "Matrix-Game 2.0"),
        ("minwm", "minWM 1.3B (DMD)"),
    )),
    ("ODE-stage models", (
        ("ours_kl4rung", "Ours, local KL (ODE)"),
        ("ours_mse4rung", "Ours, pointwise MSE (ODE)"),
        ("minwm_ode", "minWM 1.3B (ODE)"),
    )),
    ("Internal variants", (
        ("ours_no_commit", "No CARN commit"),
        ("ours_no_aux", "No CARN internalisation"),
        ("ours_no_gan", "No GAN"),
        ("ours_no_carn", "No CARN (v2 recipe)"),
        ("ours_stat_mean_only", "Seq. Norm mean + energy (v2)"),
        ("ours_stat_nonmean_only", "Seq. Norm variance + TV (v2)"),
        ("ours_base_v2_BROKEN", "V2 base (broken; diagnostic)"),
    )),
)


def table(data: pd.DataFrame, horizon: int) -> str:
    ncol = 7 if horizon == 6 else 5
    cols = "lrrrrrr" if horizon == 6 else "lrrrr"
    if horizon == 6:
        header = (r"Model & $n_{\rm dir}$ & $n_{\rm active}$ & Static $>500$"
                  r" & Start ORB & $\widetilde{\Delta S}$ & $\widetilde{S_e/S_b}$ \\")
    else:
        header = (r"Model & $n_{\rm dir}$ & Start ORB"
                  r" & $\widetilde{\Delta S}$ & $\widetilde{S_e/S_b}$ \\")
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{4pt}",
             r"\resizebox{\linewidth}{!}{%", rf"\begin{{tabular}}{{{cols}}}",
             r"\hline", header, r"\hline"]
    for group, members in GROUPS:
        lines.append(rf"\multicolumn{{{ncol}}}{{l}}{{\emph{{{group}}}}} \\")
        for model, label in members:
            r = data.loc[(horizon, model)]
            n = int(r.directional_scored)
            orb = f"{r.starting_view_overlap_median:.1f}"
            delta = f"{r.d_blur_median:.2f}"
            retain = f"{r.hf_retention_ratio_median:.2f}"
            if horizon == 6:
                row = (f"{label} & {n} & {int(r.active_scored)} & "
                       f"{int(r.control_near_static_500_n)} & {orb} & {delta} & {retain} \\\\")
            else:
                row = f"{label} & {n} & {orb} & {delta} & {retain} \\\\"
            lines.append(row)
        lines.append(r"\hline")
    lines += [r"\end{tabular}}"]
    if horizon == 6:
        caption = ("Local ICLR videos at 6 s. Directional rollouts only. "
                   "The active count requires a feature-valid context and at most "
                   "600 starting-view ORB inliers; Static $>500$ counts near-static "
                   "rollouts among all $n_{\\rm dir}$, including those outside the "
                   "active population. Static is only one component of control failure. "
                   "Start ORB, $\\Delta S=S_e-S_b$, and $S_e/S_b$ are medians; "
                   "negative $\\Delta S$ means lower end-window sharpness. "
                   "The 6-s endpoint uses the validated six-second producer index "
                   "on ICLR clips. "
                   "This is a provisional local-video diagnostic table, not a "
                   "final ICLR failure-rate table.")
    else:
        caption = (f"Local ICLR videos at {horizon} s. Directional rollouts only. "
                   "Start ORB is overlap with the first generated view, not a "
                   "relocation failure flag. $\\Delta S=S_e-S_b$ and $S_e/S_b$ "
                   "are medians of within-video sharpness change and retention. "
                   "The endpoint is the last included frame of the stated duration. "
                   "These diagnostics have no validated long-horizon failure cutoff.")
    lines += [rf"\caption{{{caption}}}", rf"\label{{tab:iclr-verified-h{horizon}}}",
              r"\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--paper", type=Path, help="also refresh the marked block in the ICLR template")
    a = p.parse_args()
    data = pd.read_csv(a.summary)
    if data.duplicated(["horizon_s", "model"]).any():
        raise ValueError("duplicate model/horizon rows")
    expected = {(h, m) for h in (6, 15, 30) for _, members in GROUPS for m, _ in members}
    found = set(zip(data.horizon_s, data.model))
    if found != expected:
        raise ValueError(f"missing={expected-found}, unexpected={found-expected}")
    if not (data.cpu_scored == data.local_video).all():
        raise ValueError("local CPU scoring incomplete")
    keyed = data.set_index(["horizon_s", "model"])
    rendered = "% Generated from summary_by_horizon.csv; do not hand-edit.\n" \
               + "\n".join(table(keyed, h) for h in (6, 15, 30))
    a.out.write_text(rendered)
    if a.paper:
        begin = "% BEGIN VERIFIED ICLR HORIZON TABLES"
        end = "% END VERIFIED ICLR HORIZON TABLES"
        paper = a.paper.read_text()
        if paper.count(begin) != 1 or paper.count(end) != 1:
            raise ValueError("paper is missing unique generated-table markers")
        left = paper.index(begin) + len(begin)
        right = paper.index(end, left)
        a.paper.write_text(paper[:left] + "\n" + rendered + paper[right:])
    print(a.out, "three horizon tables")


if __name__ == "__main__":
    main()
