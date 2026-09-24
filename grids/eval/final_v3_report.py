"""Write the final reproducibility report only after the all-video gate passes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from final_v3_tex_tables import ORDER, verify


def rate_table(q, horizon):
    lines = ["| Model | Control % | Style % | Geom. % | Conj. % | HF % |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for label, model in ORDER:
        r = q.loc[(model, horizon)]
        values = [r[f"{k}_pct_if_complete"] for k in
                  ("control", "style", "geometry", "conjuration", "hf")]
        lines.append("| " + label + " | " + " | ".join(f"{x:.1f}" for x in values) + " |")
    return "\n".join(lines)


def main(out):
    q = verify(out)
    valid = json.loads((out / "validation_final.json").read_text())
    assert valid["checks"] == "passed"
    control_audit = json.loads((out / "cross_site/control_comparison_summary.json").read_text())
    active = pd.read_csv(out / "aaai_active_h6_quality_summary.csv")
    assert len(active) == 15
    assert (active[[f"{k}_scored" for k in ("style", "geometry", "conjuration", "hf")]]
            .eq(active.active_scored, axis=0)).all().all()
    parts = ["# Final ICLR 30-second fleet evaluation", "",
             "**Validated:** all 4,320 specified ICLR videos, 15 models × 32 contexts × 9 commands. "
             "Each of the four GPU instruments and both CPU instruments has 25,920 fixed-window "
             "scores; 12,960 endpoint rows cover 6, 15, and 30 seconds. The 2,304 DMD clips on "
             "u6qf were decoded and checked, alongside 2,016 clips on this workstation.", "",
             f"The same-input control audit compared {control_audit['compared_windows']:,} "
             "fixed windows scored on both machines. It found "
             f"{control_audit['flag_changes']['wrong_direction_60']} directional and "
             f"{control_audit['flag_changes']['noop_motion_010']} no-op flag changes. "
             "Final control rates use u6qf consistently across all 15 models.", "",
             "## Six-second AAAI anchor and ICLR adaptation", "",
             "The original final-v2 AAAI flag builder reproduced all 3,376 shipped rollout rows "
             "and its reported numbers. The stationary builder reproduced all 14 model rows. "
             "Stratified direct producer checks reproduced shipped ORB, HF, DINO, Qwen3-VL, "
             "RT-DETR, and teacher readouts where original inputs were available; this is a "
             "sample producer audit, not a complete rerun of every AAAI source video. "
             "See `../out_iclr_final_v2_20260917/aaai_reproduction/`, "
             "`../out_iclr_final_v2_20260917/aaai_stationary_reproduction/`, and "
             "`../out_iclr_final_v2_20260917/aaai_producer_sample/`.", "",
             "The ICLR videos are a separate test set. The original AAAI measurement producers "
             "are used at the six-second anchor, with an ICLR movement-control flag based "
             "only on native-spatial CoTracker/PCA direction cosine below 0.5. The AAAI v2 "
             "rule also used ORB inliers above 500, and a later ICLR sensitivity analysis used "
             "800. Neither ORB cutoff enters the current tables. Near-static movement with "
             "a passing direction can therefore pass; these rates are not directly "
             "comparable to the original AAAI rates. No-op is flagged when action magnitude is "
             "at least 0.1. Style uses DINOv2 drift above 0.72, geometry uses deployed "
             "Qwen3-VL uncanny probability above 0.5, and conjuration uses RT-DETR temporal "
             "conjuration. The historical AAAI active-population audit uses ORB above 600; frozen videos "
             "remain in the control denominator. The 32 no-op flags are pooled with "
             "the 256 movement-command flags for one control rate over 288 clips.", "",
             "For the original AAAI `r00_F` Matrix-Game source, the shipped Qwen uncanny "
             "probability is 0.2695. The local `venv_qwen3` reproduction returned 0.2695; "
             "the initial u6qf environment returned 0.1484, and its package-matched "
             "variant returned 0.1826. Even giving that variant identical predecoded "
             "PNG inputs returned 0.3203. At the user's request, final geometry uses "
             "the faster u6qf environment on every ICLR video. Its threshold flags "
             "have demonstrated environment sensitivity and are not a bitwise "
             "reproduction of shipped AAAI source scores.", "",
             "The shipped AAAI Matrix-Game DINO style score is 0.5489. The local flash "
             "producer reproduced 0.5489, whereas the u6qf producer returned 0.5543. "
             "At the user's request, final style for every ICLR video uses the faster "
             "u6qf producer. Its threshold flags therefore have a measured environment "
             "sensitivity; the local DINO scores remain diagnostic only.", "",
             "On six original AAAI clips, the u6qf RT-DETR producer matched all six "
             "shipped conjuration flags and event counts. Some continuous event details "
             "differed, including up to 0.19 in the top score; the sample does not bound "
             "fleet-wide flag stability. The exact comparisons are in "
             "`cross_site/aaai_conj_remote_six.csv`.", "",
             "All 288 clips per model were scored for each quality metric, including no-op and "
             "near-static clips, as requested for the ICLR tables. `aaai_active_h6_quality_summary.csv` "
             "also reports the historical active-directional denominator for comparison. "
             "It must not be conflated with the all-video rate. All 32 ICLR source contexts "
             "were checked for feature validity; the AAAI wet-lens exclusions were not copied.", "",
             "Models have different conditioning histories: one frame for LingBot, DreamX, and "
             "Matrix-Game; 13 for minWM; 33 for our models. Qwen receives four frames selected "
             "from each candidate's actual real seed span (a repeated first frame for single-image "
             "models), then 16 generated frames per six-second window. This is an ICLR adaptation "
             "of the deployed AAAI prompt and logit scoring. Its reference choice changes some "
             "probabilities and is recorded per video. Relocation was excluded at the user's direction.", "",
             "## Long-horizon extension", "",
             "H15 and H30 use the final six-second window as the endpoint and keep the same sample "
             "density. Fixed windows start at 0, 6, 9, 12, 18, and 24 seconds. The 9–15 window "
             "supplies the H15 endpoint; the nonoverlapping series supplies trajectories. "
             "`quality_horizon_diagnostics.csv` separates endpoint flag, fraction of flagged "
             "windows, first flagged time, and ever flagged. Ever flagged depends on exposure "
             "length. Movement control is direction-only at H6, H15, and H30. "
             "`control_window_scored.csv` preserves the teacher vector, cosine, magnitude, "
             "and raw ORB count as an unthresholded diagnostic; late reversal remains a "
             "separate diagnostic. `conjuration_windows.csv` preserves "
             "event times, boxes, and temporal evidence. `style_windows.csv` preserves original "
             "context drift and local adjacent-window drift without assigning an unvalidated "
             "cutoff to the latter.", "",
             "A 480-frame video at 16 fps contains 30 seconds, with its last generated frame "
             "29.9375 seconds after the first. The exact AAAI six-second endpoint uses "
             "`ctx+round(6×fps)`; later endpoints use the last in-range generated frame. "
             "Matrix-Game's 25-fps clips use native-time indices; directional teacher inputs "
             "retain native spatial resolution. Out-of-range indices are errors, not clamped.", "",
             "HF uses original Laplacian variance windows and a frozen one-seat-per-family "
             "ICLR panel. `hf_within_video.csv` and `hf_window_trajectory.csv` also report raw "
             "sharpness loss and retention without a new cutoff. The inherited B>150 threshold "
             "was calibrated for AAAI's seven-family panel, so all ICLR HF percentages are "
             "descriptive until validated for the five-family panel. H15/H30 style and HF "
             "cutoff rates are exploratory. Starting-view ORB overlap at long horizons measures "
             "overlap after travel; it is not a relocation failure. No combined long-horizon "
             "legitimacy rate is defined.", "",
             "## Complete all-video endpoint tables", ""]
    for h in (6, 15, 30):
        parts += [f"### H{h}", "", rate_table(q, h), ""]
    parts += ["Each displayed rate uses all 288 videos per model: 256 movement commands "
              "and 32 no-op clips. "
              "All percentages are lower-is-better instrument flags; H15/H30 are fixed-window "
              "extensions, not AAAI six-second rates.", "",
              "## Provenance and validation", "",
              "- `video_manifest.csv`: per-video path, SHA256, FPS, dimensions, decoded frames, "
              "context length, and local/HPC storage site.",
              "- `model_provenance.csv` and `dmd_sidecar_manifest.csv`: 15 checkpoint identities "
              "and the 2,304 DMD sidecars. DreamX and Matrix-Game source clips lack checkpoint "
              "sidecars; their model identities come from the ICLR manuscript.",
              "- `instrument_provenance.json`: scoring code, model snapshot commits, local "
              "and u6qf library versions, and matching CoTracker/PCA weight hashes. "
              "`cross_site/style_comparison.csv` compares the same video across environments; "
              "the maximum six-window drift difference was 0.0048 with no 0.72 flag changes "
              "in that sample. `cross_site/geometry_comparison.csv` found up to 0.215 "
              "uncanny-probability difference on the same video, also without a flag change "
              "in that sample. A targeted boundary audit in "
              "`cross_site/geometry_near_boundary_comparison.csv` found three flag changes "
              "in 24 windows on four identical videos. These samples are not a fleet-wide "
              "stability bound. `instrument_provenance.json` records the AAAI source "
              "reproduction: only the local `venv_qwen3` environment matched the shipped "
              "uncanny probability exactly (0.2695). The local flash DINO environment "
              "also reproduces the AAAI style sample exactly (0.5489). Final style "
              "and geometry both use u6qf as requested, with these measured "
              "environment differences disclosed.",
              "- `cross_site/aaai_conj_remote_six.csv` and its summary compare six "
              "original AAAI videos scored by the u6qf RT-DETR producer with shipped "
              "reference measurements; flags and event counts match in all six.",
              "- `decoded_audit_remote.csv`: direct full decode of every u6qf clip. "
              "`seed65_sha_local.txt` and `seed65_sha_remote.txt` match for all 32 "
              "real seed clips. `validation_final.json`: coverage, key, window, hash, "
              "and denominator gates.",
              "- `cross_site/control_no_orb_counterfactual.csv`: the current direction-only "
              "control rates alongside the previous >800 rates, with counts of low-motion "
              "movement clips that pass. `cross_site/control_threshold_sensitivity.csv` "
              "retains the >500, >800, other ORB cutoffs, and the pre-existing AAAI "
              "magnitude alternative as historical sensitivity analyses.",
              "- `quality_endpoints_per_video.csv`, `cpu_windows.csv`, and the four "
              "`*_windows.csv` files: raw per-video and per-window measurements.",
              "- `hf_contact_sheet.png` plus the four `*_contact_sheet.png` review sheets: "
              "flagged clips and hard negatives, with matching `*_sources.csv` hashes.",
              "- `salvage_audit_final.md`: the final decisions for old artifacts, with the "
              "earlier row ledger retained as historical audit evidence. The old substitute "
              "PCA, pooled HF, and earlier ORB flags do not feed these tables.", "",
              "The eight DMD fleets ran on u6qf interactive GPU holder jobs 6650058 and "
              "6650528; all producers are resumable. The seven workstation fleets were "
              "staged and rescored there for control, style, geometry, and conjuration. "
              "No final GPU instrument mixes workstation and u6qf scores. "
              "Per-video JSON and CSV rows include source hashes; "
              "remote and local results were joined only after checking keys and hashes. "
              "The ICLR paper is populated by `final_v3_tex_tables.py` after the same validation gate.", "",
              "## Reproduction commands", "",
              "```bash",
              "OUT=grids/eval/out_iclr_final_v3_full_20260917",
              "# u6qf holder: run remote_work/holder_gpu_full.sh, holder_gpu_resume.sh,",
              "# and holder_local_gpu.sh for style, geometry, control, and conjuration",
              "python grids/eval/final_v3_quality_summary.py --out \"$OUT\"",
              "python grids/eval/final_v2_summarize.py --out \"$OUT\"",
              "python grids/eval/final_v3_validate.py --out \"$OUT\"",
              "python grids/eval/final_v3_review_sheets.py --out \"$OUT\"",
              "python grids/eval/final_v3_tex_tables.py --out \"$OUT\" --write-paper",
              "```", "",
              "The exact GPU commands and environment variables are retained in "
              "`remote_work/holder_gpu_full.sh`, `remote_work/holder_gpu_resume.sh`, "
              "and `remote_work/holder_local_gpu.sh`. Earlier local Qwen and DINO "
              "logs remain diagnostic only. "
              "The ICLR paper tables should replace the provisional local-only tables because "
              "their full-fleet denominators passed `validation_final.json`.", ""]
    (out / "FINAL_EVAL_REPORT.md").write_text("\n".join(parts))
    print(out / "FINAL_EVAL_REPORT.md")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    main(a.out.resolve())
