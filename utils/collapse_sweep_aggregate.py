#!/usr/bin/env python3
"""Aggregate Phase-A collapse-sweep outputs into a single tidy table +
quick diagnostic plots. Read by analysts after Phase A completes; not
called from Phase A itself.

Inputs:
  --sweep_dir <path>     Path to the OUTDIR Phase A wrote (containing
                         rank_0/, rank_1/, ..., rides_full.json).

Outputs (under <sweep_dir>/aggregated/):
  metrics_all.csv        All rides x all rolling steps in one CSV with
                         a (zarr, offset, tag) prefix on every row.
  metrics_all.parquet    Same as above (preferred for downstream
                         analysis -- pyarrow if installed, else skipped).
  per_metric_hist.png    One histogram per metric across the whole
                         sweep population (helps decide z-score
                         normalisation: heavy-tailed metrics might
                         need log-space).
  per_ride_traces/<tag>__<zarr>__off<offset>.png
                         One PNG per ride showing the full trajectory
                         of the most useful metrics overlayed on a
                         shared time axis. Hand-eyeball these against
                         each ride's rollout.mp4 to label the visible
                         collapse step.

This is just a convenience -- the metrics CSVs Phase A writes are the
ground truth.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s | %(message)s",
)
log = logging.getLogger("collapse_sweep_aggregate")


METRICS_TO_PLOT = [
    "commit_rms",
    "commit_peak",
    "commit_to_commit_l2",
    "commit_to_gt_l2",
    "real_residual_to_student",
    "real_residual_to_gt",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep_dir", type=str, required=True)
    p.add_argument(
        "--no_plots", action="store_true",
        help="Skip matplotlib plots (parquet/CSV only).",
    )
    args = p.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        raise SystemExit(f"--sweep_dir {sweep_dir} does not exist.")

    out_dir = sweep_dir / "aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Walk rank dirs, concatenate metrics CSVs --------------------
    rows: List[pd.DataFrame] = []
    n_csvs = 0
    for rank_dir in sorted(sweep_dir.glob("rank_*")):
        if not rank_dir.is_dir():
            continue
        for ride_dir in sorted(rank_dir.iterdir()):
            if not ride_dir.is_dir():
                continue
            csv_path = ride_dir / "metrics.csv"
            meta_path = ride_dir / "meta.json"
            if not csv_path.exists():
                continue
            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:  # noqa: BLE001
                log.warning("skipping %s: %s", csv_path, exc)
                continue
            ride_meta = {}
            if meta_path.exists():
                try:
                    ride_meta = json.loads(meta_path.read_text())
                except Exception:  # noqa: BLE001
                    pass
            df["zarr"] = ride_meta.get("zarr", ride_dir.name.split("__")[1] if "__" in ride_dir.name else "?")
            df["offset"] = int(ride_meta.get("offset", 0))
            df["tag"] = ride_meta.get("tag", ride_dir.name.split("__")[0] if "__" in ride_dir.name else "?")
            df["rank"] = int(rank_dir.name.split("_")[-1])
            df["ride_dir"] = str(ride_dir)
            df["wall_seconds_total"] = float(ride_meta.get("wall_seconds", float("nan")))
            rows.append(df)
            n_csvs += 1

    if not rows:
        raise SystemExit(f"No metrics.csv found under {sweep_dir}.")
    log.info("Loaded %d per-ride CSVs.", n_csvs)

    metrics_all = pd.concat(rows, ignore_index=True, sort=False)
    log.info("metrics_all: %d rows x %d cols", len(metrics_all),
             len(metrics_all.columns))

    csv_path = out_dir / "metrics_all.csv"
    metrics_all.to_csv(csv_path, index=False)
    log.info("Wrote %s", csv_path)

    try:
        parquet_path = out_dir / "metrics_all.parquet"
        metrics_all.to_parquet(parquet_path, index=False)
        log.info("Wrote %s", parquet_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("Parquet write skipped (%s).", exc)

    if args.no_plots:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        log.warning("Matplotlib not available (%s); skipping plots.", exc)
        return

    # ---- Per-metric histograms over whole population ----------------
    avail_metrics = [m for m in METRICS_TO_PLOT if m in metrics_all.columns]
    n = len(avail_metrics)
    cols = 3
    rows_p = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_p, cols, figsize=(5 * cols, 3.5 * rows_p))
    axes = axes.flatten() if n > 1 else [axes]
    for i, m in enumerate(avail_metrics):
        ax = axes[i]
        data = metrics_all[m].replace([float("inf"), -float("inf")], float("nan")).dropna()
        if data.empty:
            ax.set_title(f"{m} (no data)")
            continue
        ax.hist(data, bins=80, alpha=0.85)
        ax.set_yscale("log")
        ax.set_title(m)
        ax.grid(True, alpha=0.3)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.tight_layout()
    hist_path = out_dir / "per_metric_hist.png"
    fig.savefig(hist_path, dpi=130)
    plt.close(fig)
    log.info("Wrote %s", hist_path)

    # ---- Per-ride trajectory traces --------------------------------
    traces_dir = out_dir / "per_ride_traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    grouped = metrics_all.groupby(["tag", "zarr", "offset"], dropna=False)
    for (tag, zarr, offset), df in grouped:
        df = df.sort_values("step_idx")
        fig, axes = plt.subplots(
            len(avail_metrics), 1,
            figsize=(10, 2.0 * len(avail_metrics)),
            sharex=True,
        )
        if len(avail_metrics) == 1:
            axes = [axes]
        for ax, m in zip(axes, avail_metrics):
            data = df[m].replace([float("inf"), -float("inf")], float("nan"))
            ax.plot(df["step_idx"], data, lw=1.2)
            ax.set_ylabel(m, fontsize=9)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("rolling step")
        fig.suptitle(f"{tag} | {zarr} | offset={offset}", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        ride_stem = f"{tag}__{Path(str(zarr)).stem}__off{offset}"
        out_png = traces_dir / f"{ride_stem}.png"
        fig.savefig(out_png, dpi=110)
        plt.close(fig)
    log.info("Wrote %d per-ride traces under %s", len(grouped), traces_dir)


if __name__ == "__main__":
    main()
