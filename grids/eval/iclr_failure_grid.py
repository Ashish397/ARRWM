"""Render AAAI-style per-rollout failure maps from scored ICLR endpoints.

Each tile is one of the 32 contexts crossed with nine actions. The maps use
only previously scored endpoint flags; they do not read or evaluate videos.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ENDPOINTS = ROOT / "grids/eval/out_iclr_final_v3_seed29_primary_20260920/paper_five_window_scores.csv"
FIGURES = ROOT / "iclr/Figures"
ACTIONS = ["F", "FL", "L", "BL", "B", "BR", "R", "FR", "N"]
FLAGS = [
    ("control_failure", "Control", "#181818"),
    ("geometry_flag", "Geometry", "#cf4a60"),
    ("hf_flag_150_descriptive", "HF", "#f28e2b"),
    ("style_flag_072_descriptive", "Style", "#7954a6"),
]
COLORS = ["#f4f4f4"] + [entry[2] for entry in FLAGS]
MAIN = [
    ("Ours", "ours_no_gan"),
    ("LingBot-World-V2", "lingbot"),
    ("DreamX-World", "dreamx"),
    ("Matrix-Game 2.0", "matrixgame2"),
    ("minWM", "minwm"),
]
ODE = [
    ("Local KL (ours)", "ours_kl4rung"),
    ("Pointwise MSE (ours)", "ours_mse4rung"),
    ("minWM (ODE)", "minwm_ode"),
]
ABLATIONS = [
    ("Ours", "ours_no_gan"),
    ("No CARN", "ours_no_carn"),
    ("No CARN commit", "ours_no_commit"),
    ("Mean + energy only", "ours_stat_mean_only"),
    ("Variance + TV only", "ours_stat_nonmean_only"),
]


def render_temporal(data: pd.DataFrame, models: list[tuple[str, str]],
                    stem: str, dest: Path) -> None:
    """One panel per model; each video cell contains four post-anchor columns.

    Concurrent failures divide an endpoint cell into horizontal colour bands,
    so no priority rule hides a second failure dimension.
    """
    endpoints = [12, 18, 24, 30]
    contexts = sorted(data.scene.str.rsplit("_", n=1).str[0].unique())
    context_index = {name: i for i, name in enumerate(contexts)}
    action_index = {name: i for i, name in enumerate(ACTIONS)}
    rgb = {column: np.asarray(matplotlib.colors.to_rgb(color))
           for column, _, color in FLAGS}
    cell = 10
    ncols = min(5 if len(models) == 5 else 4, len(models))
    nrows = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.0 * ncols, 3.1 * nrows + 0.75),
                             squeeze=False)
    for ax, (label_text, model) in zip(axes.flat, models):
        rows = data[(data.model == model) & data.horizon_s.isin(endpoints)]
        assert len(rows) == 288 * 4, (model, len(rows))
        canvas = np.ones((32 * cell, 9 * 4 * cell, 3), dtype=float)
        for row in rows.itertuples(index=False):
            context = row.scene.rsplit("_", 1)[0]
            action = row.direction
            endpoint_i = endpoints.index(int(row.horizon_s))
            active = [column for column, _, _ in FLAGS
                      if bool(getattr(row, column))]
            if not active:
                continue
            y0 = context_index[context] * cell
            x0 = (action_index[action] * 4 + endpoint_i) * cell
            edges = np.linspace(y0, y0 + cell, len(active) + 1).round().astype(int)
            for i, column in enumerate(active):
                canvas[edges[i]:edges[i + 1], x0:x0 + cell] = rgb[column]
        ax.imshow(canvas, interpolation="nearest", aspect="auto")
        ax.set_title(label_text, fontsize=10)
        ax.set_xticks([(i * 4 + 2.0) * cell - 0.5 for i in range(9)], ACTIONS,
                      fontsize=7)
        ax.set_yticks([(i + 0.5) * cell - 0.5 for i in (0, 15, 31)],
                      ["1", "16", "32"], fontsize=7)
        ax.tick_params(length=0)
        for i in range(1, 9):
            ax.axvline(i * 4 * cell - 0.5, color="#777777", linewidth=0.8)
        for i in range(1, 36):
            if i % 4:
                ax.axvline(i * cell - 0.5, color="white", linewidth=0.12,
                           alpha=0.7)
        for edge in ax.spines.values():
            edge.set_linewidth(0.5)
            edge.set_color("#8c8c8c")
    for ax in list(axes.flat)[len(models):]:
        ax.axis("off")
    handles = [Patch(facecolor="#f4f4f4", edgecolor="#999999", label="Unflagged")]
    handles += [Patch(facecolor=color, label=name) for _, name, color in FLAGS]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.005), fontsize=8)
    fig.text(0.5, 0.105,
             "Command; four post-anchor subcolumns are 12, 18, 24, 30 s",
             ha="center", fontsize=9)
    fig.text(0.012, 0.52, "Context", rotation=90, va="center", fontsize=9)
    fig.subplots_adjust(left=0.05, right=0.995, top=0.93, bottom=0.22,
                        wspace=0.20, hspace=0.28)
    dest.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
def render(data: pd.DataFrame, horizon: int, models: list[tuple[str, str]],
           stem: str, dest: Path) -> pd.DataFrame:
    selected = data[data.horizon_s == horizon]
    contexts = sorted(selected.scene.str.rsplit("_", n=1).str[0].unique())
    assert len(contexts) == 32, len(contexts)
    context_index = {name: i for i, name in enumerate(contexts)}
    action_index = {name: i for i, name in enumerate(ACTIONS)}
    ncols = min(5, len(models))
    nrows = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.2, 5.0 * nrows), squeeze=False)
    cmap = ListedColormap(COLORS)
    records = []
    for ax, (label, model) in zip(axes.flat, models):
        rows = selected[selected.model == model]
        assert len(rows) == 288 and not rows.scene.duplicated().any(), (model, horizon)
        assert (rows.control_rule_id == "cosine_only_plus_noop_magnitude").all()
        assert rows[[col for col, _, _ in FLAGS]].notna().all().all()
        grid = np.full((32, 9), -1, dtype=np.int8)
        counts = np.zeros(len(COLORS), dtype=int)
        for row in rows.itertuples(index=False):
            context = row.scene.rsplit("_", 1)[0]
            action = row.direction
            assert action in action_index and row.scene.endswith("_" + action)
            category = next((i for i, (col, _, _) in enumerate(FLAGS, start=1)
                             if bool(getattr(row, col))), 0)
            r, c = context_index[context], action_index[action]
            assert grid[r, c] == -1, (model, horizon, context, action)
            grid[r, c] = category
            counts[category] += 1
        assert (grid >= 0).all() and counts.sum() == 288
        ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=len(COLORS) - 0.5,
                  interpolation="nearest", aspect="auto")
        ax.set_title(f"{label}\n{counts[0]}/288 unflagged", fontsize=11, pad=6)
        ax.set_xticks(range(9), ACTIONS, fontsize=8)
        ax.set_yticks([0, 15, 31], ["1", "16", "32"], fontsize=8)
        ax.tick_params(length=0)
        for edge in ax.spines.values():
            edge.set_linewidth(0.5)
            edge.set_color("#8c8c8c")
        records.append({"model": model, "horizon_s": horizon, "videos": 288,
                        "unflagged": int(counts[0]),
                        **{name.lower(): int(counts[i]) for i, (_, name, _) in enumerate(FLAGS, start=1)}})
    for ax in list(axes.flat)[len(models):]:
        ax.axis("off")
    handles = [Patch(facecolor=COLORS[0], edgecolor="#999999", label="Unflagged")]
    handles += [Patch(facecolor=color, label=name) for _, name, color in FLAGS]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 0.012), fontsize=9)
    fig.text(0.013, 0.52, "Context", rotation=90, va="center", fontsize=10)
    fig.text(0.5, 0.07, "Command (N = no-op)", ha="center", fontsize=9)
    fig.subplots_adjust(left=0.065, right=0.995, top=0.90, bottom=0.13,
                        wspace=0.26, hspace=0.31)
    dest.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest / f"{stem}.png", dpi=300)
    plt.close(fig)
    result = pd.DataFrame(records)
    result.to_csv(dest / f"{stem}_counts.csv", index=False)
    return result


def main(path: Path, dest: Path):
    d = pd.read_csv(path)
    assert len(d) == 15 * 288 * 5
    assert set(d.horizon_s.unique()) == {6, 12, 18, 24, 30}
    for horizon, models, stem in (
        (6, MAIN, "failure_grid_external_h6"),
        (30, MAIN, "failure_grid_external_h30"),
    ):
        result = render(d, horizon, models, stem, dest)
        print(stem, result[["model", "unflagged"]].to_dict("records"))
    for models, stem in (
        (MAIN, "failure_grid_temporal_external"),
        (ODE, "failure_grid_temporal_ode"),
        (ABLATIONS, "failure_grid_temporal_ablations"),
    ):
        render_temporal(d, models, stem, dest)
        print(stem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoints", type=Path, default=ENDPOINTS)
    parser.add_argument("--figures", type=Path, default=FIGURES)
    args = parser.parse_args()
    main(args.endpoints, args.figures)
