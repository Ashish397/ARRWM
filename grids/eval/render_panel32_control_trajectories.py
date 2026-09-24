#!/usr/bin/env python3
"""Render control trajectories from the audited six-axis summary."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ENDPOINTS = [6, 12, 18, 24, 30]
GROUPS = {
    "external": [
        ("ours_no_gan", "Ours"),
        ("lingbot", "LingBot-World-V2"),
        ("dreamx", "DreamX-World"),
        ("matrixgame2", "Matrix-Game 2.0"),
        ("minwm", "minWM (DMD)"),
        ("yume5b", "YUME-5B"),
    ],
    "ode": [
        ("ours_kl4rung", "Ours, local KL (ODE)"),
        ("ours_mse4rung", "Ours, pointwise MSE (ODE)"),
        ("minwm_ode", "minWM (ODE)"),
    ],
    "ablations": [
        ("ours_no_gan", "Ours"),
        ("ours_no_carn", "No CARN"),
        ("ours_no_commit", "No CARN commit"),
        ("ours_stat_mean_only", "Mean + energy only"),
        ("ours_stat_nonmean_only", "Variance + TV only"),
    ],
}
COLORS = ["#009E73", "#0072B2", "#E69F00", "#CC79A7", "#D55E00", "#9467BD"]
MARKERS = ["o", "s", "^", "D", "v", "P"]


def draw(ax, table: pd.DataFrame, group: str) -> None:
    subset = table[table["group"].eq({
        "external": "main", "ode": "ode", "ablations": "dmd"
    }[group])]
    for index, (model_id, label) in enumerate(GROUPS[group]):
        rows = subset[subset["model_id"].eq(model_id)].set_index("endpoint_s").reindex(ENDPOINTS)
        if rows["control_rate_pct"].isna().any():
            raise RuntimeError(f"missing control trajectory for {group}/{model_id}")
        ax.plot(
            ENDPOINTS, rows["control_rate_pct"], label=label,
            color=COLORS[index], marker=MARKERS[index], markersize=5.5,
            linewidth=2.8 if model_id == "ours_no_gan" else 1.8,
            zorder=5 if model_id == "ours_no_gan" else 2,
        )
    ax.set_xlim(5.3, 30.7)
    ax.set_ylim(-2, 102)
    ax.set_xticks(ENDPOINTS)
    ax.set_xlabel("Rollout endpoint (s)")
    ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.legend(frameon=False, fontsize=10, loc="best")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--paper-figures", type=Path, required=True)
    args = parser.parse_args()
    table = pd.read_csv(args.summary)
    args.figures.mkdir(parents=True, exist_ok=True)
    args.paper_figures.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), sharex=True, sharey=True)
    for ax, group in zip(axes, ("external", "ode", "ablations")):
        draw(ax, table, group)
    axes[0].set_ylabel("Directional-control failure (%)")
    fig.tight_layout()
    combined = args.figures / "trajectory_control.png"
    fig.savefig(combined, dpi=220, bbox_inches="tight")
    plt.close(fig)

    for group in ("external", "ode", "ablations"):
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        draw(ax, table, group)
        ax.set_ylabel("Directional-control failure (%)")
        fig.tight_layout()
        output = args.figures / f"trajectory_control_{group}.png"
        fig.savefig(output, dpi=220, bbox_inches="tight")
        plt.close(fig)

    for source in args.figures.glob("trajectory_control*.png"):
        (args.paper_figures / source.name).write_bytes(source.read_bytes())


if __name__ == "__main__":
    main()
