"""Generate the final ICLR percentage tables after complete all-video validation.

No output is written to the paper unless --write-paper is passed. The command
refuses any missing endpoint or fixed-window score, so partial runs cannot be
mistaken for final tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from .panel32_control_rule import CONTROL_RULE_ID
except ImportError:  # Direct execution from grids/eval.
    from panel32_control_rule import CONTROL_RULE_ID  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "iclr/iclr2027_conference.tex"
ORDER = [
    ("Ours", "ours_recovery_base"),
    ("LingBot-World-V2 1.3B", "lingbot"),
    ("DreamX-World 5B", "dreamx"),
    ("Matrix-Game 2.0", "matrixgame2"),
    ("minWM 1.3B (DMD)", "minwm"),
    ("Ours, local KL (ODE)", "ours_kl4rung"),
    ("Ours, pointwise MSE (ODE)", "ours_mse4rung"),
    ("minWM 1.3B (ODE)", "minwm_ode"),
    ("No CARN commit", "ours_no_commit"),
    ("No CARN internalisation", "ours_no_aux"),
    ("No GAN", "ours_no_gan"),
    ("No CARN", "ours_no_carn"),
    ("Mean + energy only", "ours_stat_mean_only"),
    ("Variance + TV only", "ours_stat_nonmean_only"),
]


def verify(out):
    m = pd.read_csv(out / "video_manifest.csv")
    assert len(m) == 4320 and m.available_video.all()
    assert m.decoded_frames.notna().all() and m[m.storage_site == "u6qf"].remote_video_verified.all()
    e = pd.read_csv(out / "cpu_endpoints_scored.csv")
    assert len(e) == 12960 and e.hf_panel_complete.all()
    q = pd.read_csv(out / "quality_horizon_summary.csv")
    assert len(q) == 45 and not q.duplicated(["model", "horizon_s"]).any()
    assert (q.control_rule_id == CONTROL_RULE_ID).all(), "wrong control rule"
    for metric, expected in (("style", 288), ("geometry", 288), ("conjuration", 288),
                             ("hf", 288), ("control", 256)):
        assert (q[f"{metric}_scored"] == expected).all(), f"incomplete {metric}"
        assert q[f"{metric}_pct_if_complete"].notna().all(), metric
    d = pd.read_csv(out / "quality_horizon_diagnostics.csv")
    assert len(d) == 4 * 4320 * 3
    assert (d.scored_windows == d.expected_windows).all(), "incomplete fixed windows"
    return q.set_index(["model", "horizon_s"])


def table(q, horizon):
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\linewidth}{!}{%",
             r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
             r"Model & Control\,\% & Style\,\% & Geom.\,\% & Conj.\,\% & HF\,\% \\",
             r"\midrule"]
    for i, (label, model) in enumerate(ORDER):
        if i in (5, 8):
            lines.append(r"\midrule")
        r = q.loc[(model, horizon)]
        vals = [r[f"{c}_pct_if_complete"] for c in
                ("control", "style", "geometry", "conjuration", "hf")]
        lines.append(label + " & " + " & ".join(f"{float(x):.1f}" for x in vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}"]
    if horizon == 6:
        caption = ("Failure rates (\\%) in the first six seconds; lower is better. "
                   "Control uses 256 directional videos (32 contexts $\\times$ eight commands); "
                   "the four visual rates use all 288 videos, including no-op. "
                   "The five criteria and their thresholds are defined "
                   "in the evaluation setup. The last six rows are component ablations.")
    else:
        caption = (f"Failure rates (\\%) at the {horizon}-second endpoint; lower is better. "
                   "Control uses the 256 directional videos; the four visual scores use "
                   "all 288 videos. The last six rows are component ablations.")
    lines += [r"\caption{" + caption + "}",
              rf"\label{{tab:iclr-final-h{horizon}}}", r"\end{table}"]
    return "\n".join(lines)


def section(q, control_audit):
    del control_audit  # retained in the machine-readable provenance report
    def pct(model, horizon, metric):
        return float(q.loc[(model, horizon), f"{metric}_pct_if_complete"])

    prose = r"""\subsection{Evaluation setup}

We compare our four-step student with LingBot-World-V2~\citep{gao2026lingbot},
DreamX-World~\citep{dreamx2026world}, Matrix-Game 2.0~\citep{he2025matrixgame2},
and minWM~\citep{zhao2026minwm}. The test set contains 32 held-out real-video
contexts. For each context, every model receives eight directional commands
and a no-op command, giving 288 generated 30-second videos per model. We
evaluate all videos, including stationary continuations, with the same
commands and endpoint times. Every continuation begins immediately after the
same underlying real-video frame (frame 32). Each model retains its native
conditioning span: one frame for LingBot-World-V2, DreamX-World, and
Matrix-Game 2.0; 29 pixel frames for minWM; and 33 pixel frames for our
checkpoints. The geometry judge sees matched real reference times ending at
frame 32.

Following the multi-axis evaluation practice of interactive world-model
papers, we measure action response and distinct visual failure modes
separately. Control is flagged when
the CoTracker--PCA motion readout has direction cosine $<0.5$ under a movement
command or when more than 450 ORB--RANSAC inliers indicate a near-static
continuation. No-op is reported separately and fails at action magnitude
$\geq0.1$. Style is flagged when
DINOv2 drift from the real context exceeds $0.72$. Geometry uses Qwen3-VL
uncanny probability $>0.5$; conjuration uses the RT-DETR temporal event rule.
High-frequency (HF) degradation uses Laplacian-variance loss and $B>150$,
with a fixed family-balanced reference panel. Control uses the 256 directional
videos, the four visual measures use all 288 videos, and no-op motion uses its
32 videos. Direction-only and near-static components are retained separately
in the evaluation artifacts.

We report the initial six-second window and the final six-second windows
ending at 15 and 30 seconds. The sampling density and decision rule for each
measure remain fixed across horizons. ODE variants and component ablations
are grouped separately from the primary external comparison. Per-window
trajectories and first flagged times are provided with the evaluation
artifacts.

\subsection{Quantitative results}

Tables~\ref{tab:iclr-final-h6}--\ref{tab:iclr-final-h30} show the endpoint
failure rates. At six seconds, our model combines low control, style,
geometry, and HF rates. LingBot-World-V2 shows more style and HF failures;
Matrix-Game 2.0 has more geometry failures; minWM is most frequently flagged
for control despite its low geometry failure rate. DreamX-World also retains
low geometry failure. These outcomes show distinct strengths across the five
measures.
"""
    h6 = table(q, 6)
    h15 = table(q, 15)
    h30 = table(q, 30)
    discussion = rf"""
The long-horizon comparison separates appearance stability from scene
structure. At 15 seconds our model has {pct('ours_recovery_base', 15, 'style'):.1f}\%
style failures, versus {pct('lingbot', 15, 'style'):.1f}\% for LingBot-World-V2
and {pct('dreamx', 15, 'style'):.1f}\% for DreamX-World. Its geometry failure
rate rises to {pct('ours_recovery_base', 15, 'geometry'):.1f}\%, however, and
reaches {pct('ours_recovery_base', 30, 'geometry'):.1f}\% at 30 seconds. Thus
style preservation and directional response do not imply persistent
geometry; this is the principal remaining limitation of the 30-second
rollouts.

The ODE comparison isolates the initial transfer objective. Local KL reduces
six-second style failure from {pct('ours_mse4rung', 6, 'style'):.1f}\% with
pointwise MSE to {pct('ours_kl4rung', 6, 'style'):.1f}\%, and geometry failure
from {pct('ours_mse4rung', 6, 'geometry'):.1f}\% to
{pct('ours_kl4rung', 6, 'geometry'):.1f}\%. Among the DMD ablations, removing
the CARN cache commit increases six-second geometry failure from
{pct('ours_recovery_base', 6, 'geometry'):.1f}\% to
{pct('ours_no_commit', 6, 'geometry'):.1f}\%. Restricting Sequence Norm to
either statistic subset also substantially increases geometry failures.
The no-GAN variant has fewer six-second flags than the complete model in
several columns, so the ablations do not establish a uniform gain from every
component at every horizon.

\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{Figures/failure_grid_external_h6.png}}
\caption{{Per-video failure maps at six seconds for the primary comparison.
Rows are the same 32
contexts and columns are the nine commands, including no-op (N). Each tile
is one video; colour marks its first flagged measure in the fixed priority
control $>$ geometry $>$ HF $>$ style $>$ conjuration. Neutral means none
of these five measures flagged the video. Panel headings give the unflagged
count out of 288.}}
\label{{fig:iclr-failure-external}}
\end{{figure}}

Figure~\ref{{fig:iclr-failure-external}} makes the principal failure profiles
visible at the level of individual contexts and commands. In particular,
minWM's control flags cluster across movement commands, while Matrix-Game 2.0
is dominated by geometry flags. Ablations are instead compared across rollout
time using stacked marginal-failure trajectories, avoiding separate
context-by-command grids at every endpoint. The corresponding 30-second
external map is provided in the evaluation artifacts; it is dominated by
geometry flags for our model, consistent with
Table~\ref{{tab:iclr-final-h30}}.
"""
    return prose + "\n\n" + h6 + "\n\n" + h15 + "\n\n" + h30 + "\n\n" + discussion + "\n\n"


def main(out, write_paper):
    q = verify(out)
    control_audit = json.loads((out / "cross_site/control_comparison_summary.json").read_text())
    content = section(q, control_audit)
    (out / "iclr_tables_final.tex").write_text(content)
    if write_paper:
        source = PAPER.read_text()
        begin = source.find(r"\subsection{Evaluation setup}")
        if begin < 0:
            begin = source.find(r"\subsection{Evaluation protocol}")
        if begin < 0:
            begin = source.index(r"\subsection{Protocol and current coverage}")
        end = source.index(r"\subsection{Runtime}", begin)
        PAPER.write_text(source[:begin] + content + source[end:])
        print(PAPER)
    else:
        print(out / "iclr_tables_final.tex")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--write-paper", action="store_true")
    a = p.parse_args()
    main(a.out.resolve(), a.write_paper)
