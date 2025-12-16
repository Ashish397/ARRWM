#!/usr/bin/env python3
"""
Aggregate `input_actions` CSV files and render an XY heatmap of their distribution.

Typical usage:

    python utils/visualise_action_heatmap.py /path/to/output_rides --output heatmap.png

The script searches recursively for files named like `input_actions_*.csv`, extracts
action pairs (preferring columns such as `linear`/`angular` and otherwise taking the
last two numeric values per row), and builds a 2D histogram visualised as a heatmap.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


class PartialLoadInterrupt(KeyboardInterrupt):
    """
    KeyboardInterrupt carrying partial data accumulated before the interrupt.
    """

    def __init__(
        self,
        coords: Sequence[tuple[float, float]],
        processed_files: int,
        total_hint: int | None,
    ):
        super().__init__("KeyboardInterrupt during input action aggregation.")
        self.coords = tuple(coords)
        self.processed_files = processed_files
        self.total_hint = total_hint


DEFAULT_PATTERN = "input_actions_*.csv"


def _coerce_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _looks_like_header(row: Sequence[str]) -> bool:
    non_empty = [cell.strip() for cell in row if cell.strip()]
    if not non_empty:
        return False
    convertible = sum(1 for cell in non_empty if _coerce_float(cell) is not None)
    return convertible < len(non_empty)


_X_KEYWORDS: tuple[tuple[str, ...], ...] = (
    ("linear",),
    ("linear_velocity",),
    ("lin_vel",),
    ("velocity_linear",),
    ("action_linear",),
    ("action_lin",),
    ("forward",),
    ("speed",),
    ("vx",),
    ("vel_x",),
    ("v_x",),
    ("throttle",),
)
_Y_KEYWORDS: tuple[tuple[str, ...], ...] = (
    ("angular",),
    ("angular_velocity",),
    ("ang_vel",),
    ("velocity_angular",),
    ("action_angular",),
    ("action_ang",),
    ("turn",),
    ("yaw",),
    ("omega",),
    ("wz",),
    ("vel_w",),
    ("steer",),
    ("steering",),
)


def _infer_action_columns(header: Sequence[str]) -> tuple[int, int] | None:
    normalized = [cell.strip().lower().replace("-", "_") for cell in header]
    if not normalized:
        return None

    def _locate(keywords: Sequence[str]) -> int | None:
        for keyword in keywords:
            keyword_norm = keyword.strip().lower().replace("-", "_")
            for idx, name in enumerate(normalized):
                if not name:
                    continue
                if name == keyword_norm:
                    return idx
        return None

    for x_keywords in _X_KEYWORDS:
        x_idx = _locate(x_keywords)
        if x_idx is None:
            continue
        for y_keywords in _Y_KEYWORDS:
            y_idx = _locate(y_keywords)
            if y_idx is None or y_idx == x_idx:
                continue
            return x_idx, y_idx
    return None


def _iter_file_pairs(csv_path: Path) -> Iterable[tuple[float, float]]:
    """
    Yield (x, y) pairs from a CSV file. The function is deliberately tolerant:
    it skips empty rows and headers, prefers well-known action columns when
    they are present, and otherwise falls back to the last two numeric values
    found in each row (useful when CSVs prefix actions with frame counters).
    """
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)

        # Detect delimiter heuristically when possible, fall back to comma.
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.get_dialect("excel")

        reader = csv.reader(handle, dialect=dialect)
        header_checked = False
        column_indices: tuple[int, int] | None = None

        for row in reader:
            if not row:
                continue
            stripped = [cell.strip() for cell in row]
            if not header_checked:
                header_checked = True
                if _looks_like_header(stripped):
                    column_indices = _infer_action_columns(stripped)
                    continue
            numeric_values: list[tuple[int, float]] = []
            for idx, cell in enumerate(stripped):
                if not cell:
                    continue
                maybe_val = _coerce_float(cell)
                if maybe_val is None:
                    continue
                numeric_values.append((idx, maybe_val))
            if len(numeric_values) < 2:
                continue
            if column_indices is not None:
                idx_x, idx_y = column_indices
                value_map = {idx: value for idx, value in numeric_values}
                x_val = value_map.get(idx_x)
                y_val = value_map.get(idx_y)
                if x_val is not None and y_val is not None:
                    yield x_val, y_val
                    continue
            x_val = numeric_values[-2][1]
            y_val = numeric_values[-1][1]
            yield x_val, y_val


def _collect_csv_files(
    inputs: Sequence[Path],
    pattern: str,
    *,
    max_files: int | None = None,
    file_seed: int | None = None,
) -> list[Path]:
    discovered: list[Path] = []
    seen: set[Path] = set()
    for raw in inputs:
        path = raw.expanduser()
        if not path.exists():
            print(f"[Warning] Skipping missing path: {path}", file=sys.stderr)
            continue
        if path.is_file():
            candidate = path.resolve()
            if candidate not in seen:
                discovered.append(candidate)
                seen.add(candidate)
            continue
        if path.is_dir():
            for csv_path in sorted(path.rglob(pattern)):
                resolved = csv_path.resolve()
                if resolved not in seen:
                    discovered.append(resolved)
                    seen.add(resolved)
            continue
        print(f"[Warning] Skipping unsupported path type: {path}", file=sys.stderr)
    if not discovered:
        return discovered
    if file_seed is not None:
        rng = np.random.default_rng(file_seed)
        rng.shuffle(discovered)
    if max_files is not None and max_files > 0:
        discovered = discovered[:max_files]
    return discovered


def _log_debug(message: str, *, force: bool = False) -> None:
    if _log_debug.enabled or force:
        print(f"[Debug] {message}", file=sys.stderr)


_log_debug.enabled = False  # type: ignore[attr-defined]


def _describe_actions(label: str, actions: np.ndarray, *, force: bool = False) -> None:
    if actions.size == 0:
        _log_debug(f"{label}: no action pairs available.", force=force)
        return
    x_vals = actions[:, 0]
    y_vals = actions[:, 1]
    summary = (
        f"{label}: count={actions.shape[0]:,}, "
        f"x[min={x_vals.min():.4f}, max={x_vals.max():.4f}, mean={x_vals.mean():.4f}, std={x_vals.std():.4f}], "
        f"y[min={y_vals.min():.4f}, max={y_vals.max():.4f}, mean={y_vals.mean():.4f}, std={y_vals.std():.4f}]"
    )
    _log_debug(summary, force=force)


def _load_actions(
    csv_files: Sequence[Path],
    progress_interval: int = 100,
    total_hint: int | None = None,
    *,
    debug: bool = False,
) -> np.ndarray:
    _log_debug.enabled = debug  # type: ignore[attr-defined]
    coords: list[tuple[float, float]] = []
    processed = 0
    try:
        for idx, csv_file in enumerate(csv_files, start=1):
            file_pairs = list(_iter_file_pairs(csv_file))
            if not file_pairs:
                print(f"[Info] No XY pairs found in {csv_file}", file=sys.stderr)
                continue
            if debug:
                file_array = np.asarray(file_pairs, dtype=np.float32)
                _describe_actions(f"File {csv_file}", file_array)
            coords.extend(file_pairs)
            processed = idx
            if progress_interval > 0 and (idx % progress_interval == 0 or idx == len(csv_files)):
                total_suffix = f"/{total_hint}" if total_hint is not None else ""
                print(
                    f"[Info] Processed {idx}{total_suffix} CSV files ({len(coords):,} action pairs collected so far).",
                    file=sys.stderr,
                )
    except KeyboardInterrupt as exc:
        total_suffix = f"/{total_hint}" if total_hint is not None else ""
        print(
            f"[Warning] Interrupted after {processed}{total_suffix} files; using collected actions so far.",
            file=sys.stderr,
        )
        raise PartialLoadInterrupt(coords, processed, total_hint) from exc
    if not coords:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(coords, dtype=np.float32)


def _compute_unique_actions(
    actions: np.ndarray,
    decimals: int = 6,
) -> list[tuple[float, float, int]]:
    """
    Compute unique action pairs and their occurrence counts.
    
    Returns a list of (x, y, count) tuples sorted by count (descending).
    Actions are rounded to `decimals` decimal places for uniqueness comparison.
    """
    if actions.size == 0:
        return []
    
    # Round actions to specified decimal places for comparison
    rounded = np.round(actions, decimals=decimals)
    
    # Use a dictionary to count occurrences
    counts: dict[tuple[float, float], int] = {}
    for x, y in rounded:
        key = (float(x), float(y))
        counts[key] = counts.get(key, 0) + 1
    
    # Sort by count descending, then by x, then by y
    result = [(x, y, count) for (x, y), count in counts.items()]
    result.sort(key=lambda item: (-item[2], item[0], item[1]))
    
    return result


def _output_unique_actions(
    unique_actions: list[tuple[float, float, int]],
    output_path: Path | None,
    top_n: int | None,
    total_count: int,
) -> None:
    """
    Output unique action pairs with their counts and percentages.
    """
    if top_n is not None and top_n > 0:
        display_actions = unique_actions[:top_n]
        truncated = len(unique_actions) > top_n
    else:
        display_actions = unique_actions
        truncated = False
    
    lines: list[str] = []
    lines.append(f"# Unique action pairs: {len(unique_actions):,}")
    lines.append(f"# Total action samples: {total_count:,}")
    lines.append("# Sorted by frequency (descending)")
    lines.append("")
    lines.append("action_x,action_y,count,percentage")
    
    for x, y, count in display_actions:
        pct = (count / total_count) * 100 if total_count > 0 else 0.0
        lines.append(f"{x},{y},{count},{pct:.4f}")
    
    if truncated:
        lines.append(f"# ... and {len(unique_actions) - top_n:,} more unique actions")
    
    content = "\n".join(lines) + "\n"
    
    if output_path is not None:
        output_path = output_path.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        print(f"[Info] Saved unique actions to {output_path}", file=sys.stderr)
    else:
        # Print to stdout
        print(content)


def _print_unique_actions_summary(
    unique_actions: list[tuple[float, float, int]],
    total_count: int,
    top_n: int = 20,
) -> None:
    """
    Print a summary of unique actions to stderr.
    """
    print(f"\n[Unique Actions Summary]", file=sys.stderr)
    print(f"  Total unique action pairs: {len(unique_actions):,}", file=sys.stderr)
    print(f"  Total action samples: {total_count:,}", file=sys.stderr)
    
    if not unique_actions:
        return
    
    # Coverage statistics
    cumulative = 0
    coverage_50 = coverage_90 = coverage_99 = None
    for i, (_, _, count) in enumerate(unique_actions):
        cumulative += count
        pct = (cumulative / total_count) * 100
        if coverage_50 is None and pct >= 50:
            coverage_50 = i + 1
        if coverage_90 is None and pct >= 90:
            coverage_90 = i + 1
        if coverage_99 is None and pct >= 99:
            coverage_99 = i + 1
            break
    
    print(f"  Actions covering 50% of samples: {coverage_50 or 'N/A'}", file=sys.stderr)
    print(f"  Actions covering 90% of samples: {coverage_90 or 'N/A'}", file=sys.stderr)
    print(f"  Actions covering 99% of samples: {coverage_99 or 'N/A'}", file=sys.stderr)
    
    print(f"\n  Top {min(top_n, len(unique_actions))} most frequent actions:", file=sys.stderr)
    print(f"  {'Rank':<6} {'Action X':<12} {'Action Y':<12} {'Count':<10} {'%':>8}", file=sys.stderr)
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*10} {'-'*8}", file=sys.stderr)
    
    for i, (x, y, count) in enumerate(unique_actions[:top_n], start=1):
        pct = (count / total_count) * 100 if total_count > 0 else 0.0
        print(f"  {i:<6} {x:<12.6f} {y:<12.6f} {count:<10,} {pct:>7.3f}%", file=sys.stderr)
    
    if len(unique_actions) > top_n:
        print(f"  ... and {len(unique_actions) - top_n:,} more unique actions", file=sys.stderr)
    print("", file=sys.stderr)


def _build_histogram(
    actions: np.ndarray,
    bins_x: int,
    bins_y: int,
    x_range: tuple[float, float] | None,
    y_range: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ranges = None
    working = actions
    if x_range or y_range:
        # Populate missing range bounds from the observed data to avoid numpy complaints.
        if x_range is None:
            x_range = (float(actions[:, 0].min()), float(actions[:, 0].max()))
        if y_range is None:
            y_range = (float(actions[:, 1].min()), float(actions[:, 1].max()))
        # To honour the documentation ("Clamp the X axis ...") we clip the values
        # instead of silently discarding out-of-range samples via numpy.histogram2d.
        working = actions.copy()
        if x_range is not None:
            lower, upper = x_range
            if lower >= upper:
                raise ValueError("Invalid X range: minimum must be smaller than maximum.")
            np.clip(working[:, 0], lower, upper, out=working[:, 0])
        if y_range is not None:
            lower, upper = y_range
            if lower >= upper:
                raise ValueError("Invalid Y range: minimum must be smaller than maximum.")
            np.clip(working[:, 1], lower, upper, out=working[:, 1])
        ranges = [x_range, y_range]
    _describe_actions("Pre-histogram (possibly clipped)", working, force=True)
    histogram, x_edges, y_edges = np.histogram2d(
        working[:, 0],
        working[:, 1],
        bins=[bins_x, bins_y],
        range=ranges,
    )
    _log_debug(
        f"Histogram summary: total_count={int(histogram.sum())}, "
        f"nonzero_bins={int(np.count_nonzero(histogram))}/{histogram.size}"
    , force=True)
    return histogram, x_edges, y_edges


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an XY heatmap from input_actions CSV files."
    )
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="Directories or CSV files to scan for input action data.",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help=f"Glob pattern (relative) used to discover CSV files inside directories. Default: {DEFAULT_PATTERN}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path for the rendered heatmap image. When omitted the figure is shown interactively.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=200,
        help="Number of histogram bins to use for both axes (default: 200).",
    )
    parser.add_argument(
        "--bins-x",
        type=int,
        help="Override the number of bins on the X axis.",
    )
    parser.add_argument(
        "--bins-y",
        type=int,
        help="Override the number of bins on the Y axis.",
    )
    parser.add_argument(
        "--x-range",
        type=float,
        nargs=2,
        metavar=("X_MIN", "X_MAX"),
        default=(-1.0, 1.0),
        help="Clamp the X axis to a fixed range before building the histogram (default: -1 1).",
    )
    parser.add_argument(
        "--y-range",
        type=float,
        nargs=2,
        metavar=("Y_MIN", "Y_MAX"),
        default=(-1.0, 1.0),
        help="Clamp the Y axis to a fixed range before building the histogram (default: -1 1).",
    )
    parser.add_argument(
        "--cmap",
        default="magma",
        help="Matplotlib colour map to apply to the heatmap (default: magma).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Dots per inch for the saved figure (default: 200).",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(6.0, 6.0),
        help="Figure size in inches (default: 6 6).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Randomly subsample this many action pairs before building the heatmap.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="Random seed used when --sample-size is provided.",
    )
    parser.add_argument(
        "--auto-range",
        action="store_true",
        help="Derive axis ranges from the data instead of forcing [-1, 1].",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Limit processing to at most this many CSV files.",
    )
    parser.add_argument(
        "--file-seed",
        type=int,
        help="Random seed used to shuffle files before applying --max-files.",
    )
    parser.add_argument(
        "--title",
        default="Input Action Distribution",
        help="Title for the plot (default: Input Action Distribution).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Force rendering the figure interactively even when --output is given.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging about parsed data and histogram construction.",
    )
    parser.add_argument(
        "--log-counts",
        action="store_true",
        help="Display the heatmap using a logarithmic colour scale (helps when a few bins dominate).",
    )
    parser.add_argument(
        "--clip-percentile",
        type=float,
        default=99.5,
        help="Clip the colour scale at this upper percentile of non-zero bin counts (set to 100 to disable clipping). Default: 99.5.",
    )
    parser.add_argument(
        "--swap-axes",
        action="store_true",
        help="Exchange X and Y before plotting (useful when actions are stored as Y,X).",
    )
    parser.add_argument(
        "--unique-actions",
        action="store_true",
        help="Output a list of unique action pairs with their occurrence counts.",
    )
    parser.add_argument(
        "--unique-output",
        type=Path,
        help="Path to save unique action counts as CSV (default: print to stdout).",
    )
    parser.add_argument(
        "--unique-top-n",
        type=int,
        default=None,
        help="Only show top N most frequent unique actions (default: show all).",
    )
    parser.add_argument(
        "--unique-decimals",
        type=int,
        default=6,
        help="Number of decimal places to round actions for uniqueness comparison (default: 6).",
    )
    args = parser.parse_args(argv)
    if args.clip_percentile is not None and not (0 < args.clip_percentile <= 100):
        parser.error("--clip-percentile must be within (0, 100].")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    csv_files = _collect_csv_files(
        args.inputs,
        args.pattern,
        max_files=args.max_files,
        file_seed=args.file_seed,
    )
    if not csv_files:
        print("[Error] No matching CSV files were found.", file=sys.stderr)
        return 1

    total_files = len(csv_files)
    print(f"[Info] Found {total_files} CSV files. Loading actions...", file=sys.stderr)

    actions: np.ndarray | None = None
    interrupted = False
    try:
        actions = _load_actions(csv_files, total_hint=total_files, debug=args.debug)
    except PartialLoadInterrupt as exc:
        interrupted = True
        actions = (
            np.asarray(exc.coords, dtype=np.float32)
            if exc.coords
            else np.empty((0, 2), dtype=np.float32)
        )
        total_suffix = f"/{exc.total_hint}" if exc.total_hint is not None else ""
        print(
            f"[Warning] Proceeding with partial data from {exc.processed_files}{total_suffix} files.",
            file=sys.stderr,
        )
    except KeyboardInterrupt:
        interrupted = True
        actions = np.empty((0, 2), dtype=np.float32)
        print("[Warning] Interrupted before any actions were collected.", file=sys.stderr)
    finally:
        if actions is None:
            actions = np.empty((0, 2), dtype=np.float32)

    _describe_actions("Loaded actions (raw)", actions, force=True)
    if actions.size > 0:
        finite_mask = np.isfinite(actions).all(axis=1)
        if not np.all(finite_mask):
            removed = int(np.count_nonzero(~finite_mask))
            print(
                f"[Warning] Dropped {removed:,} action pairs containing NaN or infinite values.",
                file=sys.stderr,
            )
            actions = actions[finite_mask]
            _describe_actions("Loaded actions (finite)", actions, force=True)

    if actions.size == 0:
        print("[Error] No XY action pairs were extracted.", file=sys.stderr)
        return 2

    bins_x = args.bins_x or args.bins
    bins_y = args.bins_y or args.bins

    original_count = actions.shape[0]
    if args.sample_size is not None and args.sample_size > 0 and original_count > args.sample_size:
        rng = np.random.default_rng(args.sample_seed)
        indices = rng.choice(original_count, size=args.sample_size, replace=False)
        actions = actions[indices]
        print(
            f"[Info] Applied action subsample to {actions.shape[0]:,} pairs "
            f"(from {original_count:,} total pairs prior to sampling).",
            file=sys.stderr,
        )
    elif interrupted:
        print(f"[Info] Using {original_count:,} action pairs collected before interrupt.", file=sys.stderr)

    if args.swap_axes:
        actions = actions[:, [1, 0]]
        print("[Info] Swapped X and Y action columns per --swap-axes.", file=sys.stderr)
        _describe_actions("Post-swap actions", actions, force=True)

    # Compute and output unique actions if requested
    if args.unique_actions or args.unique_output:
        unique_actions = _compute_unique_actions(actions, decimals=args.unique_decimals)
        total_count = int(actions.shape[0])
        
        # Always print summary to stderr
        _print_unique_actions_summary(unique_actions, total_count)
        
        # Output full list to file or stdout
        _output_unique_actions(
            unique_actions,
            output_path=args.unique_output,
            top_n=args.unique_top_n,
            total_count=total_count,
        )

    if args.auto_range:
        x_range = None
        y_range = None
    else:
        x_range = tuple(args.x_range)
        y_range = tuple(args.y_range)
        if actions.size > 0:
            if x_range is not None:
                x_low, x_high = x_range
                outside_x = int(np.count_nonzero((actions[:, 0] < x_low) | (actions[:, 0] > x_high)))
                ratio_x = outside_x / actions.shape[0]
                _log_debug(
                    f"X range clamp stats: bounds=({x_low}, {x_high}), outside={outside_x:,}/{actions.shape[0]:,} ({ratio_x:.2%})",
                    force=True,
                )
                if ratio_x >= 0.9:
                    x_range = (float(actions[:, 0].min()), float(actions[:, 0].max()))
                    print(
                        "[Warning] Over 90% of X actions lie outside the requested range. "
                        "Expanding range to cover observed data.",
                        file=sys.stderr,
                    )
                    _log_debug(f"Adjusted X range to {x_range}", force=True)
            if y_range is not None:
                y_low, y_high = y_range
                outside_y = int(np.count_nonzero((actions[:, 1] < y_low) | (actions[:, 1] > y_high)))
                ratio_y = outside_y / actions.shape[0]
                _log_debug(
                    f"Y range clamp stats: bounds=({y_low}, {y_high}), outside={outside_y:,}/{actions.shape[0]:,} ({ratio_y:.2%})",
                    force=True,
                )
                if ratio_y >= 0.9:
                    y_range = (float(actions[:, 1].min()), float(actions[:, 1].max()))
                    print(
                        "[Warning] Over 90% of Y actions lie outside the requested range. "
                        "Expanding range to cover observed data.",
                        file=sys.stderr,
                    )
                    _log_debug(f"Adjusted Y range to {y_range}", force=True)

    histogram, x_edges, y_edges = _build_histogram(actions, bins_x, bins_y, x_range, y_range)
    if not np.count_nonzero(histogram):
        print(
            "[Warning] Histogram contains only zeros after binning. "
            "Verify the input ranges and CSV parsing logic.",
            file=sys.stderr,
        )

    hist_to_display = histogram
    clip_annotation: str | None = None
    if args.clip_percentile is not None and args.clip_percentile < 100:
        positive_bins = histogram[histogram > 0]
        if positive_bins.size:
            clip_value = np.percentile(positive_bins, args.clip_percentile)
            if clip_value > 0:
                hist_to_display = histogram.copy()
                np.clip(hist_to_display, 0, clip_value, out=hist_to_display)
                clip_annotation = f"<=p{args.clip_percentile:g}≈{clip_value:,.0f}"
                _log_debug(
                    f"Applied percentile clipping at p{args.clip_percentile:g}, threshold≈{clip_value:.2f}",
                    force=True,
                )

    norm = None
    if args.log_counts:
        positive_bins = hist_to_display[hist_to_display > 0]
        if positive_bins.size >= 2:
            norm = colors.LogNorm(
                vmin=float(positive_bins.min()), vmax=float(positive_bins.max())
            )
        elif positive_bins.size == 1:
            norm = colors.LogNorm(
                vmin=float(positive_bins.min()),
                vmax=float(max(positive_bins.min() * 10, positive_bins.min())),
            )

    fig, ax = plt.subplots(figsize=args.figsize, constrained_layout=True)
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    mesh = ax.imshow(
        hist_to_display.T,
        origin="lower",
        extent=extent,
        cmap=args.cmap,
        norm=norm,
        aspect="equal",
    )
    cbar = fig.colorbar(mesh, ax=ax)
    label = "Count"
    if args.log_counts:
        label += " (log scale)"
    if clip_annotation:
        label += f" {clip_annotation}"
    cbar.set_label(label)
    ax.set_xlabel("Action X")
    ax.set_ylabel("Action Y")
    ax.set_title(args.title)

    total_actions = int(actions.shape[0])
    ax.text(
        0.02,
        0.98,
        f"Total actions: {total_actions:,}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5),
    )

    if args.output:
        output_path = args.output.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=args.dpi)
        print(f"[Info] Saved heatmap to {output_path}", file=sys.stderr)
    if args.show or not args.output:
        plt.show()
    else:
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
