"""Publication plots for dual-reference style and geometry trajectories."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from iclr_metric_trajectories import (ABLATIONS, COLORS, ENDPOINTS, EXTERNAL,
                                      GROUP_TITLES, MARKERS, ODE)

GROUPS = [("external", EXTERNAL, GROUP_TITLES[0]),
          ("ode", ODE, GROUP_TITLES[1]),
          ("ablations", ABLATIONS, GROUP_TITLES[2])]


def line_panel(ax, d, group, column, title, ylabel, percent=False):
    for i, (label, model) in enumerate(group):
        z = d[d.model == model].set_index("endpoint_s").reindex(ENDPOINTS)
        y = z[column] * (100 if percent else 1)
        main = model == "ours_recovery_base"
        ax.plot(ENDPOINTS, y, label=label, color=COLORS[model],
                marker=MARKERS[i % len(MARKERS)], markersize=5.3,
                linewidth=2.8 if main else 1.65, zorder=5 if main else 2)
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel("Rollout endpoint (s)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(ENDPOINTS)
    ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.8)


def style_plot(d, figures: Path, suffix, group, group_title):
    fig, axes = plt.subplots(1, 3, figsize=(14.3, 4.1), sharex=True, sharey=True)
    specs = [("seed_dino_drift", "Real-seed drift"),
             ("rolling_dino_drift", "One-second rolling drift"),
             ("seam_dino_drift", "Boundary seam drift")]
    for ax, (col, title) in zip(axes, specs):
        line_panel(ax, d, group, col, title, "Mean DINOv2 cosine drift")
        ax.set_ylim(-0.02, 1.02)
    axes[0].legend(fontsize=7.8, frameon=False, loc="best")
    fig.suptitle(f"Dual-reference style trajectory: {group_title}", fontsize=12)
    fig.tight_layout()
    fig.savefig(figures / f"trajectory_style_dual_{suffix}.pdf", bbox_inches="tight")
    fig.savefig(figures / f"trajectory_style_dual_{suffix}.png", dpi=220,
                bbox_inches="tight")
    plt.close(fig)


def geometry_plot(d, figures: Path, suffix, group, group_title):
    fig, axes = plt.subplots(1, 3, figsize=(14.3, 4.1), sharex=True, sharey=True)
    specs = [("seed_conditioned_failure", "Primary: seed-conditioned geometry"),
             ("absolute_failure", "Diagnostic: within-window plausibility"),
             ("rolling_break_failure", "Diagnostic: boundary continuity")]
    for ax, (col, title) in zip(axes, specs):
        line_panel(ax, d, group, col, title, "Videos flagged (%)", percent=True)
        ax.set_ylim(-2, 102)
    axes[0].legend(fontsize=7.8, frameon=False, loc="best")
    fig.suptitle(f"Geometry trajectory and diagnostics: {group_title}", fontsize=12)
    fig.tight_layout()
    fig.savefig(figures / f"trajectory_geometry_dual_{suffix}.pdf", bbox_inches="tight")
    fig.savefig(figures / f"trajectory_geometry_dual_{suffix}.png", dpi=220,
                bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dual-out", required=True, type=Path)
    p.add_argument("--figures", required=True, type=Path)
    a = p.parse_args()
    style = pd.read_csv(a.dual_out / "style_dual_summary.csv")
    geometry = pd.read_csv(a.dual_out / "geometry_dual_summary.csv")
    a.figures.mkdir(parents=True, exist_ok=True)
    for suffix, group, title in GROUPS:
        style_plot(style, a.figures, suffix, group, title)
        geometry_plot(geometry, a.figures, suffix, group, title)
    print(a.figures)


if __name__ == "__main__":
    main()
