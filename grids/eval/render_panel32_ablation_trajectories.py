#!/usr/bin/env python3
"""Render the four DMD ablation trajectory panels from the final summary."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ORDER = [
    "ours_no_gan",
    "ours_no_carn",
    "ours_no_commit",
    "ours_stat_mean_only",
    "ours_stat_nonmean_only",
]

LABELS = {
    "ours_no_gan": "Ours",
    "ours_no_carn": "No CARN",
    "ours_no_commit": "No CARN commit",
    "ours_stat_mean_only": "Mean + energy only",
    "ours_stat_nonmean_only": "Variance + TV only",
}

COLORS = {
    "ours_no_gan": "#009E73",
    "ours_no_carn": "#E7298A",
    "ours_no_commit": "#5E3C99",
    "ours_stat_mean_only": "#4D9221",
    "ours_stat_nonmean_only": "#E69F00",
}

MARKERS = {
    "ours_no_gan": "o",
    "ours_no_carn": "s",
    "ours_no_commit": "^",
    "ours_stat_mean_only": "P",
    "ours_stat_nonmean_only": "X",
}

METRICS = {
    "control_rate_pct": ("Directional-control failure (%)", "trajectory_control_ablations.png"),
    "style_rate_pct": ("Style failure (%)", "trajectory_style_dino_ablations.png"),
    "geometry_rate_pct": ("Geometry failure (%)", "trajectory_geometry_ablations.png"),
    "hf_rate_pct": ("High-frequency failure (%)", "trajectory_hf_ablations.png"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()

    table = pd.read_csv(args.summary)
    table = table[table["group"].eq("dmd")]
    args.figures.mkdir(parents=True, exist_ok=True)

    for metric, (ylabel, filename) in METRICS.items():
        fig, ax = plt.subplots(figsize=(8.6, 6.0), dpi=200)
        for model in ORDER:
            rows = table[table["model_id"].eq(model)].sort_values("endpoint_s")
            if rows.empty:
                raise ValueError(f"missing DMD summary rows for {model}")
            ax.plot(
                rows["endpoint_s"],
                rows[metric],
                color=COLORS[model],
                marker=MARKERS[model],
                linewidth=2.5 if model == "ours_no_gan" else 2.0,
                markersize=6.5,
                label=LABELS[model],
            )
        ax.set_xlim(5.3, 30.7)
        ax.set_ylim(-2, 102)
        ax.set_xticks([6, 12, 18, 24, 30])
        ax.set_xlabel("Rollout endpoint (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.35)
        ax.legend(loc="best", frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(args.figures / filename, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
