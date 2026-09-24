"""Plot the four time-varying ICLR dimensions over fixed six-second windows.

The script consumes the completed per-video measurements.  It does not rerun
any model or evaluator.  The primary trajectory uses the non-overlapping
windows ending at 6, 12, 18, 24, and 30 seconds.  Style is shown as a
continuous DINOv2 score because both seed-referenced and rolling scores are
available; the other dimensions use their deployed binary decisions.
Conjuration and relocation are retained in the scored outputs for their
six-second event table, but are not plotted as trajectories.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from final_v2_iclr import FAMILY, SEAT


# Paper figures are routinely reduced to a quarter or half page.  Set the
# typography at source so axes remain legible after that reduction, and leave
# descriptive titles to the LaTeX captions.
plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 11,
})


ENDPOINTS = [6, 12, 18, 24, 30]
STARTS = [0, 6, 12, 18, 24]

EXTERNAL = [
    ("Ours", "ours_no_gan"),
    ("LingBot-World-V2", "lingbot"),
    ("DreamX-World", "dreamx"),
    ("Matrix-Game 2.0", "matrixgame2"),
    ("minWM (DMD)", "minwm"),
    ("YUME-5B", "yume5b"),
]

ODE = [
    ("Ours, local KL (ODE)", "ours_kl4rung"),
    ("Ours, pointwise MSE (ODE)", "ours_mse4rung"),
    ("minWM (ODE)", "minwm_ode"),
]

ABLATIONS = [
    ("Ours", "ours_no_gan"),
    ("No CARN", "ours_no_carn"),
    ("No CARN commit", "ours_no_commit"),
    ("Mean + energy only", "ours_stat_mean_only"),
    ("Variance + TV only", "ours_stat_nonmean_only"),
]

COLORS = {
    "ours_recovery_base": "#009E73",
    "lingbot": "#0072B2",
    "dreamx": "#E69F00",
    "matrixgame2": "#CC79A7",
    "minwm": "#D55E00",
    "yume5b": "#9467BD",
    "ours_kl4rung": "#56B4E9",
    "ours_mse4rung": "#B2182B",
    "minwm_ode": "#8C510A",
    "ours_no_commit": "#5E3C99",
    "ours_no_aux": "#1B9E77",
    "ours_no_gan": "#7570B3",
    "ours_no_carn": "#E7298A",
    "ours_stat_mean_only": "#66A61E",
    "ours_stat_nonmean_only": "#E6AB02",
}

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h"]


GROUPS = [EXTERNAL, ODE, ABLATIONS]
GROUP_TITLES = ["External models vs. ours", "ODE models", "Ablations including ours"]


def _base_axes(ylabel: str):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), sharex=True, sharey=True)
    for ax in axes:
        ax.set_xticks(ENDPOINTS)
        ax.set_xlim(5.3, 30.7)
        ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.8)
        ax.set_xlabel("Rollout endpoint (s)")
    axes[0].set_ylabel(ylabel)
    return fig, axes


def _plot_groups(axes, table: pd.DataFrame, value: str, percent: bool = False):
    for ax, group in zip(axes, GROUPS):
        for i, (label, model) in enumerate(group):
            z = table[table.model == model].set_index("endpoint_s").reindex(ENDPOINTS)
            y = z[value].to_numpy(float) * (100 if percent else 1)
            main = model == "ours_no_gan"
            ax.plot(
                ENDPOINTS, y, label=label, color=COLORS[model],
                marker=MARKERS[i % len(MARKERS)], markersize=5.2,
                linewidth=2.8 if main else 1.55, alpha=1.0 if main else 0.88,
                zorder=5 if main else 2,
            )
        ax.legend(fontsize=10.5, frameon=False, ncol=1, loc="best")


def _save(fig, figures: Path, stem: str, rect=None):
    fig.tight_layout(rect=rect)
    fig.savefig(figures / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _rate_table(frame: pd.DataFrame, flag: str) -> pd.DataFrame:
    d = frame[frame.window_start_s.isin(STARTS)].copy()
    d["endpoint_s"] = d.window_end_s.astype(int)
    return d.groupby(["model", "endpoint_s"], as_index=False)[flag].mean()


def _cumulative_rate_table(frame: pd.DataFrame, flag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert a window event into cumulative incidence for each rollout."""
    d = frame[frame.window_start_s.isin(STARTS)].copy()
    d["endpoint_s"] = d.window_end_s.astype(int)
    d = d.sort_values(["model", "scene", "endpoint_s"])
    d[flag] = d.groupby(["model", "scene"], sort=False)[flag].cummax()
    rates = d.groupby(["model", "endpoint_s"], as_index=False)[flag].mean()
    return rates, d


def _hf_panel_rates(out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(out / "hf_window_trajectory.csv")
    d = d[d.window_start_s.isin(STARTS)].copy()
    lookup = d.set_index(["scene", "model", "window_end_s"]).d_blur_from_early_base.to_dict()
    rows = []
    for endpoint in ENDPOINTS:
        for scene in sorted(d.scene.unique()):
            fixed = {family: lookup[(scene, model, endpoint)] for family, model in SEAT.items()}
            candidates = d[(d.scene == scene) & (d.window_end_s == endpoint)]
            for r in candidates.itertuples():
                seats = fixed.copy()
                seats[FAMILY[r.model]] = r.d_blur_from_early_base
                reference = float(np.median(list(seats.values())))
                b = reference - float(r.d_blur_from_early_base)
                rows.append({"scene": scene, "model": r.model, "endpoint_s": endpoint,
                             "hf_B": b, "hf_failure": int(b > 150)})
    per_video = pd.DataFrame(rows)
    # Preserve the deployed AAAI six-second anchor exactly.  Its endpoint is
    # one native frame later than the fixed-window producer for historical
    # compatibility; later points use the non-overlapping window trajectory.
    anchor = pd.read_csv(out / "cpu_endpoints_scored.csv")
    anchor = anchor[anchor.horizon_s == 6][["scene", "model", "B_v2_iclrfive"]].copy()
    anchor["endpoint_s"] = 6
    anchor["hf_B"] = anchor.B_v2_iclrfive
    anchor["hf_failure"] = (anchor.hf_B > 150).astype(int)
    anchor = anchor[["scene", "model", "endpoint_s", "hf_B", "hf_failure"]]
    per_video = pd.concat([per_video[per_video.endpoint_s != 6], anchor], ignore_index=True)
    rates = per_video.groupby(["model", "endpoint_s"], as_index=False).hf_failure.mean()
    return rates, per_video


def plot_rate(table: pd.DataFrame, value: str, title: str, stem: str, figures: Path):
    fig, axes = _base_axes("Videos flagged (%)")
    _plot_groups(axes, table, value, percent=True)
    for ax in axes:
        ax.set_ylim(-2, 102)
    _save(fig, figures, stem)


def plot_rate_group(table: pd.DataFrame, value: str, title: str, stem: str,
                    figures: Path, group, group_title: str):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for i, (label, model) in enumerate(group):
        z = table[table.model == model].set_index("endpoint_s").reindex(ENDPOINTS)
        main = model == "ours_no_gan"
        ax.plot(ENDPOINTS, z[value].to_numpy(float) * 100, label=label,
                color=COLORS[model], marker=MARKERS[i % len(MARKERS)],
                markersize=5.5, linewidth=2.8 if main else 1.7,
                zorder=5 if main else 2)
    ax.set(xlabel="Rollout endpoint (s)", ylabel="Videos flagged (%)",
           xlim=(5.3, 30.7), ylim=(-2, 102))
    ax.set_xticks(ENDPOINTS)
    ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.legend(fontsize=11, frameon=False, loc="best")
    _save(fig, figures, stem)


def plot_style(style: pd.DataFrame, figures: Path):
    d = style[style.window_start_s.isin(STARTS)].copy()
    d["endpoint_s"] = d.window_end_s.astype(int)
    summary = d.groupby(["model", "endpoint_s"], as_index=False).agg(
        seed_dino_drift=("drift_from_real", "mean"),
        rolling_dino_drift=("local_adjacent_drift", "mean"),
        style_failure=("drift_from_real", lambda x: float((x > 0.72).mean())),
    )
    fig, axes = _base_axes("Mean DINOv2 cosine drift")
    for ax, group in zip(axes, GROUPS):
        for i, (label, model) in enumerate(group):
            z = summary[summary.model == model].set_index("endpoint_s").reindex(ENDPOINTS)
            color = COLORS[model]
            main = model == "ours_no_gan"
            ax.plot(ENDPOINTS, z.seed_dino_drift, label=label, color=color,
                    marker=MARKERS[i % len(MARKERS)], markersize=5.2,
                    linewidth=2.8 if main else 1.55, zorder=5 if main else 2)
        ax.axhline(0.72, color="#333333", linestyle=":", linewidth=1.2,
                   label="Seed-drift threshold (0.72)")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=10, frameon=False, ncol=2, loc="upper center",
                  bbox_to_anchor=(0.5, -0.16), columnspacing=0.9,
                  handlelength=2.2)
    _save(fig, figures, "trajectory_style_dino", rect=(0, 0.25, 1, 1))
    for suffix, group, title in zip(["external", "ode", "ablations"],
                                    GROUPS, GROUP_TITLES):
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        for i, (label, model) in enumerate(group):
            z = summary[summary.model == model].set_index("endpoint_s").reindex(ENDPOINTS)
            color = COLORS[model]
            main = model == "ours_no_gan"
            ax.plot(ENDPOINTS, z.seed_dino_drift, label=label, color=color,
                    marker=MARKERS[i % len(MARKERS)], markersize=5.5,
                    linewidth=2.8 if main else 1.7, zorder=5 if main else 2)
        ax.axhline(0.72, color="#333333", linestyle=":", linewidth=1.2,
                   label="Seed-drift threshold (0.72)")
        ax.set(xlabel="Rollout endpoint (s)", ylabel="Mean DINOv2 cosine drift",
               xlim=(5.3, 30.7), ylim=(-0.02, 1.02))
        ax.set_xticks(ENDPOINTS)
        ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.8)
        ax.legend(fontsize=11, frameon=False, loc="best")
        _save(fig, figures, f"trajectory_style_dino_{suffix}")
    return summary


STACKED_METRICS = [
    ("Control", "control_failure", "#333333"),
    ("Style", "style_failure", "#7954a6"),
    ("Geometry", "geometry_flag", "#cf4a60"),
    ("HF", "hf_failure", "#f28e2b"),
]


def _joint_timevarying_scores(control_raw: pd.DataFrame,
                              style_raw: pd.DataFrame,
                              geometry_raw: pd.DataFrame,
                              hf_per_video: pd.DataFrame,
                              relocation_raw: pd.DataFrame,
                              conjuration_per_video: pd.DataFrame) -> pd.DataFrame:
    """Join per-video decisions so overall plots measure joint outcomes."""
    keys = ["model", "scene", "endpoint_s"]
    c = control_raw[control_raw.window_start_s.isin(STARTS)].copy()
    c["endpoint_s"] = c.window_end_s.astype(int)
    c = c[keys + ["control_failure"]]
    s = style_raw[style_raw.window_start_s.isin(STARTS)].copy()
    s["endpoint_s"] = s.window_end_s.astype(int)
    s["style_failure"] = (s.drift_from_real > 0.72).astype(int)
    s = s[keys + ["style_failure"]]
    g = geometry_raw[geometry_raw.window_start_s.isin(STARTS)].copy()
    g["endpoint_s"] = g.window_end_s.astype(int)
    g = g[keys + ["geometry_flag"]]
    h = hf_per_video[keys + ["hf_failure"]]
    joint = c.merge(s, on=keys, validate="one_to_one")
    joint = joint.merge(g, on=keys, validate="one_to_one")
    joint = joint.merge(h, on=keys, validate="one_to_one")
    flags = ["control_failure", "style_failure", "geometry_flag", "hf_failure"]
    joint["failure_count"] = joint[flags].sum(axis=1).astype(int)
    r = relocation_raw.rename(columns={"horizon_s": "endpoint_s"})
    r = r[keys + ["relocation_flag_50"]]
    q = conjuration_per_video[keys + ["conjuration_flag"]]
    joint = joint.merge(r, on=keys, validate="one_to_one")
    joint = joint.merge(q, on=keys, validate="one_to_one")
    joint["nonreloc_failure_count"] = (
        joint.failure_count + joint.conjuration_flag)
    joint["all_axis_failure_count"] = (
        joint.failure_count + joint.relocation_flag_50 + joint.conjuration_flag)
    return joint


def plot_overall_group(control: pd.DataFrame, style: pd.DataFrame,
                       geometry: pd.DataFrame, hf: pd.DataFrame,
                       joint: pd.DataFrame, figures: Path, suffix: str,
                       group: list[tuple[str, str]], group_title: str):
    """Plot marginal failures plus rollouts clean on every axis so far."""
    tables = {
        "control_failure": control,
        "style_failure": style,
        "geometry_flag": geometry,
        "hf_failure": hf,
    }
    ncols = min(3 if len(group) <= 5 else 4, len(group))
    nrows = int(np.ceil(len(group) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.35 * ncols, 4.0 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    ymax = 100.0
    plot_endpoints = [0] + ENDPOINTS
    for ax, (label, model) in zip(axes.flat, group):
        values = []
        for _, value, _ in STACKED_METRICS:
            z = tables[value]
            z = z[z.model == model].set_index("endpoint_s").reindex(ENDPOINTS)
            values.append(np.r_[0.0, z[value].to_numpy(float) * 100])

        candidate = joint[joint.model == model].sort_values(
            ["scene", "endpoint_s"]).copy()
        candidate["failed_so_far"] = candidate.groupby(
            "scene").nonreloc_failure_count.cummax()
        clean_through = np.r_[100.0, np.array([
            100 * (candidate[candidate.endpoint_s == endpoint].failed_so_far == 0).mean()
            for endpoint in ENDPOINTS
        ])]
        bands = np.vstack(values)
        cumulative = np.cumsum(bands, axis=0)
        colors = [color for _, _, color in STACKED_METRICS]
        ax.stackplot(plot_endpoints, bands, colors=colors, alpha=0.76,
                     edgecolor="white", linewidth=0.8)
        for boundary, color in zip(cumulative, colors):
            ax.plot(plot_endpoints, boundary, color=color, linewidth=1.15)
        ax.plot(plot_endpoints, cumulative[-1], color="#111111", linewidth=2.0,
                marker="o", markersize=3.3)
        ax.plot(plot_endpoints, clean_through, color="#16833a", linewidth=2.7,
                linestyle="--", marker="o", markersize=4.2, zorder=10)
        ymax = max(ymax, float(np.nanmax(cumulative[-1])))
        ax.text(0.03, 0.96, label, transform=ax.transAxes, va="top",
                fontsize=14, fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72})
        ax.set_xticks(plot_endpoints)
        ax.set_xlim(-0.6, 30.7)
        ax.grid(True, color="#d9d9d9", linewidth=0.65, alpha=0.75)
        ax.set_xlabel("Endpoint (s)")
    for ax in list(axes.flat)[len(group):]:
        ax.axis("off")
    ymax = min(500, max(100, int(np.ceil(ymax / 50.0) * 50)))
    for ax in axes.flat[:len(group)]:
        ax.set_ylim(0, ymax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Stacked rates (%)")
    handles = [Patch(facecolor=color, edgecolor="white", label=label)
               for label, _, color in STACKED_METRICS]
    handles.append(plt.Line2D([0], [0], color="#16833a", linewidth=2.7,
                              linestyle="--", marker="o", markersize=4.2,
                              label="Clean on 5 axes through endpoint (excl. relocation)"))
    handles.append(plt.Line2D([0], [0], color="#111111", linewidth=2,
                              marker="o", markersize=3.3,
                              label="Sum of marginal failure rates"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, 0.005))
    _save(fig, figures, f"overall_{suffix}", rect=(0, 0.13, 1, 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--figures", type=Path, required=True)
    a = p.parse_args()
    out, figures = a.out.resolve(), a.figures.resolve()
    figures.mkdir(parents=True, exist_ok=True)

    control_raw = pd.read_csv(out / "control_window_scored.csv")
    # Directional control and no-op use different decision rules.  Keep the
    # paper trajectory on the eight-command directional population, matching
    # final_v3_quality_summary and panel32_aggregate; no-op remains available
    # separately in the normalized aggregation outputs.  The joint per-video
    # clean curve below still applies the appropriate rule to all nine actions.
    control = _rate_table(
        control_raw[control_raw.direction != "N"], "control_failure")
    style_raw = pd.read_csv(out / "style_windows.csv")
    style = plot_style(style_raw, figures)
    geometry_raw = pd.read_csv(out / "geometry_windows.csv")
    geometry = _rate_table(geometry_raw, "geometry_flag")
    long_relocation_path = out / "long_relocation_final_rows.csv"
    if long_relocation_path.exists():
        relocation_raw = pd.read_csv(long_relocation_path).rename(columns={
            "relocation_cumulative_flag": "relocation_flag_50",
            "reloc_cal_v2_h6_inliers": "panel_inliers",
        })
        relocation_raw["threshold_status"] = np.where(
            relocation_raw.horizon_s.eq(6),
            "Reloc-Cal-v2 endpoint decision",
            "cumulative adjudicated temporal relocation incidence")
        relocation_title = "Cumulative scene relocation over rollout time"
        relocation_short_title = "Cumulative scene relocation"
    else:
        panel_path = out / "relocation_panel_rows.csv"
        if panel_path.exists():
            # Relocation is an onset event in this protocol.  Carry the
            # six-second decision forward only to satisfy the per-endpoint
            # join used for the non-relocation plots below; it is excluded
            # from their clean curve and is reported once in the paper table.
            anchor = pd.read_csv(panel_path)
            anchor = anchor[anchor.horizon_s == 6].rename(columns={
                "relocation_flag_6s_exploratory": "relocation_flag_50",
            })
            pieces = []
            for endpoint in ENDPOINTS:
                copy = anchor.copy()
                copy["horizon_s"] = endpoint
                copy["threshold_status"] = (
                    "six-second onset decision carried forward; not a repeated endpoint test"
                )
                pieces.append(copy)
            relocation_raw = pd.concat(pieces, ignore_index=True)
        else:
            relocation_raw = pd.read_csv(out / "relocation_rows.csv")
        relocation_title = "Six-second scene-relocation onset"
        relocation_short_title = "Scene-relocation onset"
    relocation = relocation_raw.groupby(
        ["model", "horizon_s"], as_index=False).relocation_flag_50.mean().rename(
            columns={"horizon_s": "endpoint_s"})
    conjuration_raw = pd.read_csv(out / "conjuration_windows.csv")
    conjuration, conjuration_per_video = _cumulative_rate_table(
        conjuration_raw, "conjuration_flag")
    hf, hf_per_video = _hf_panel_rates(out)

    plot_rate(control, "control_failure", "Control failure over rollout time",
              "trajectory_control", figures)
    plot_rate(geometry, "geometry_flag", "Geometric corruption over rollout time",
              "trajectory_geometry", figures)
    plot_rate(hf, "hf_failure", "High-frequency degradation over rollout time",
              "trajectory_hf", figures)
    rate_specs = [
        (control, "control_failure", "Control failure", "trajectory_control"),
        (geometry, "geometry_flag", "Geometric corruption", "trajectory_geometry"),
        (hf, "hf_failure", "High-frequency degradation", "trajectory_hf"),
    ]
    for table, value, title, stem in rate_specs:
        for suffix, group, group_title in zip(["external", "ode", "ablations"],
                                               GROUPS, GROUP_TITLES):
            plot_rate_group(table, value, title, f"{stem}_{suffix}", figures,
                            group, group_title)

    joint = _joint_timevarying_scores(
        control_raw, style_raw, geometry_raw, hf_per_video,
        relocation_raw, conjuration_per_video)
    joint.to_csv(out / "joint_failure_burden.csv", index=False)
    for suffix, group, group_title in zip(["external", "ode", "ablations"],
                                           GROUPS, GROUP_TITLES):
        plot_overall_group(control, style, geometry, hf, joint, figures,
                           suffix, group, group_title)

    control["metric"] = "control_failure_rate"
    geometry["metric"] = "geometry_failure_rate"
    relocation["metric"] = "relocation_failure_rate"
    conjuration["metric"] = "conjuration_failure_rate"
    hf["metric"] = "hf_failure_rate"
    rates = pd.concat([
        control.rename(columns={"control_failure": "value"}),
        geometry.rename(columns={"geometry_flag": "value"}),
        relocation.rename(columns={"relocation_flag_50": "value"}),
        conjuration.rename(columns={"conjuration_flag": "value"}),
        hf.rename(columns={"hf_failure": "value"}),
    ], ignore_index=True)
    rates[["model", "endpoint_s", "metric", "value"]].to_csv(
        out / "metric_trajectory_rates.csv", index=False)
    style.to_csv(out / "style_dino_trajectory_summary.csv", index=False)
    hf_per_video.to_csv(out / "hf_five_horizon_panel_rows.csv", index=False)
    conjuration_per_video.to_csv(out / "conjuration_cumulative_windows.csv", index=False)

    # One row per rollout and endpoint for the paper's failure maps.  Style
    # and geometry retain the real conditioning span as their reference;
    # conjuration is cumulative after its first detected event.
    keys = ["model", "scene", "endpoint_s"]
    c = control_raw[control_raw.window_start_s.isin(STARTS)].copy()
    c["endpoint_s"] = c.window_end_s.astype(int)
    c = c[keys + ["direction", "control_failure", "control_rule_id"]]
    s = style_raw[style_raw.window_start_s.isin(STARTS)].copy()
    s["endpoint_s"] = s.window_end_s.astype(int)
    s["style_flag_072_descriptive"] = (s.drift_from_real > 0.72).astype(int)
    s = s[keys + ["style_flag_072_descriptive"]]
    g = geometry_raw[geometry_raw.window_start_s.isin(STARTS)].copy()
    g["endpoint_s"] = g.window_end_s.astype(int)
    g = g[keys + ["geometry_flag"]]
    r = relocation_raw.rename(columns={"horizon_s": "endpoint_s"})
    r = r[keys + ["relocation_flag_50", "panel_inliers",
                  "threshold_status"]]
    q = conjuration_per_video[keys + ["conjuration_flag"]]
    h = hf_per_video[keys + ["hf_failure"]].rename(
        columns={"hf_failure": "hf_flag_150_descriptive"})
    scored = c.merge(s, on=keys, validate="one_to_one")
    scored = scored.merge(g, on=keys, validate="one_to_one")
    scored = scored.merge(r, on=keys, validate="one_to_one")
    scored = scored.merge(q, on=keys, validate="one_to_one")
    scored = scored.merge(h, on=keys, validate="one_to_one")
    scored = scored.rename(columns={"endpoint_s": "horizon_s"})
    expected = control_raw[["scene", "model"]].drop_duplicates().shape[0] * 5
    assert len(scored) == expected
    assert not scored.isna().any().any()
    scored.to_csv(out / "paper_five_window_scores.csv", index=False)
    print("wrote", figures)


if __name__ == "__main__":
    main()
