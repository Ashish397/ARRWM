import argparse
import csv
import math
import sys
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple, Union


DEFAULT_INPUT_ROOT = Path("/projects/u5as/frodobots")
DEFAULT_OUTPUT_ROOT = Path("/projects/u5as/frodobots_actions")
DEFAULT_MAX_FRAMES = 6001
DEFAULT_MAX_ROWS = 501


@dataclass(frozen=True)
class ControlSample:
    timestamp: float
    linear: float
    angular: float


@dataclass(frozen=True)
class CameraSample:
    frame_id: int
    timestamp: float


@dataclass(frozen=True)
class Action:
    linear: float
    angular: float
    
    def is_nan(self) -> bool:
        return math.isnan(self.linear) or math.isnan(self.angular)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate input action files from front camera timestamps and control data."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root directory containing frodobots control/timestamp data.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory where actions files will be written.",
    )
    parser.add_argument(
        "--frames-per-block",
        type=int,
        default=12,
        help="Number of real frames per block (default 12 = 3 latent frames × 4 real frames per latent).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help="Maximum number of frames from each front camera file to process.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help="Maximum number of output action rows per file.",
    )
    return parser.parse_args(argv)


def discover_rides(input_root: Path) -> Iterator[Path]:
    for output_dir in sorted(
        d for d in input_root.iterdir() if d.is_dir() and d.name.startswith("output_rides_")
    ):
        for ride_dir in sorted(d for d in output_dir.iterdir() if d.is_dir()):
            yield ride_dir


def collect_pairs(ride_dir: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for front_path in sorted(ride_dir.glob("front_camera_timestamps_*.csv")):
        control_name = front_path.name.replace("front_camera_timestamps_", "control_data_")
        control_path = ride_dir / control_name
        if control_path.is_file():
            pairs.append((front_path, control_path))
    return pairs


def read_camera_samples(path: Path, max_frames: int) -> List[CameraSample]:
    samples: List[CameraSample] = []
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            if not row:
                continue
            frame_id = int(row[0])
            timestamp = float(row[1])
            samples.append(CameraSample(frame_id=frame_id, timestamp=timestamp))
            if len(samples) >= max_frames:
                break
    return samples


def read_control_samples(path: Path) -> List[ControlSample]:
    samples: List[ControlSample] = []
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            if not row or row[0] == "linear":
                continue
            linear = float(row[0])
            angular = float(row[1])
            timestamp = float(row[-1])
            samples.append(ControlSample(timestamp=timestamp, linear=linear, angular=angular))
    samples.sort(key=lambda sample: sample.timestamp)
    return samples


def select_control_window(
    control_timestamps: Sequence[float],
    controls: Sequence[ControlSample],
    window_start: float,
    window_end: float,
) -> Sequence[ControlSample]:
    if not controls:
        return ()
    left_pos = bisect_left(control_timestamps, window_start)
    right_pos = bisect_right(control_timestamps, window_end)
    if left_pos < right_pos:
        return controls[left_pos:right_pos]
    return ()


def compute_actions(
    camera_samples: Sequence[CameraSample],
    control_samples: Sequence[ControlSample],
    max_actions: int,
    frames_per_block: int = 12,
) -> Tuple[List[Action], int, float, int]:
    control_timestamps = [sample.timestamp for sample in control_samples]
    last_control_ts = control_timestamps[-1] if control_timestamps else 0.0
    first_camera_ts = camera_samples[0].timestamp if camera_samples else 0.0
    actions: List[Action] = []
    nan_count = 0
    
    # Block 0: Independent first frame - action is always (0, 0)
    actions.append(Action(linear=0.0, angular=0.0))
    end_frame_id = 0
    end_timestamp = first_camera_ts
    
    # Process remaining frames in blocks starting from frame 1
    frame_idx = 1
    while frame_idx < len(camera_samples) and len(actions) < max_actions:
        chunk_end = min(frame_idx + frames_per_block, len(camera_samples))
        chunk = camera_samples[frame_idx:chunk_end]
        if not chunk:
            break
        
        start_ts = chunk[0].timestamp
        end_ts = chunk[-1].timestamp
        window_start = start_ts - 0.1
        window_end = end_ts
        
        window = select_control_window(control_timestamps, control_samples, window_start, window_end)
        
        if not window:
            # Check if control data has ended (no more control data available)
            if end_ts > last_control_ts:
                # Control data has ended, terminate
                break
            else:
                # Gap in control data or control hasn't started yet - use NaN
                actions.append(Action(linear=math.nan, angular=math.nan))
                nan_count += 1
        else:
            weight = len(window)
            linear_avg = sum(sample.linear for sample in window) / weight
            angular_avg = sum(sample.angular for sample in window) / weight
            actions.append(Action(linear=linear_avg, angular=angular_avg))
        
        end_frame_id = chunk[-1].frame_id
        end_timestamp = chunk[-1].timestamp
        frame_idx = chunk_end
    
    return (actions, end_frame_id, end_timestamp, nan_count)


def write_actions(output_path: Path, actions: Sequence[Action], max_rows: int) -> None:
    import math
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_to_write = list(actions[:max_rows])
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_id", "linear", "angular"])
        for block_idx, action in enumerate(rows_to_write):
            if math.isnan(action.linear) or math.isnan(action.angular):
                writer.writerow([block_idx, "NaN", "NaN"])
            else:
                writer.writerow([block_idx, f"{action.linear:.6f}", f"{action.angular:.6f}"])


def write_start_frame(
    output_path: Path, 
    end_frame_id: int,
    end_timestamp: float,
    nan_count: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_id", "timestamp"])
        writer.writerow([end_frame_id, f"{end_timestamp:.6f}"])
        writer.writerow([nan_count, "nan_blocks"])


def process_pair(
    front_path: Path,
    control_path: Path,
    input_root: Path,
    output_root: Path,
    max_frames: int,
    max_rows: int,
    frames_per_block: int = 12,
) -> Tuple[str, int]:
    """Process a pair and return (status, num_actions). Status: 'error', 'zero', 'few', 'incomplete', 'full'"""
    camera_samples = read_camera_samples(front_path, max_frames)
    control_samples = read_control_samples(control_path)
    if not camera_samples or not control_samples:
        print(f"[skip] Missing data for {front_path}", file=sys.stderr)
        return ("error", 0)
    
    actions, end_frame_id, end_timestamp, nan_count = compute_actions(
        camera_samples, 
        control_samples, 
        max_actions=max_rows, 
        frames_per_block=frames_per_block,
    )
    if not actions:
        print(f"[skip] No actions computed for {front_path}", file=sys.stderr)
        return ("zero", 0)

    num_actions = len(actions)
    
    relative = front_path.relative_to(input_root)
    parts = relative.parts
    output_relative = Path("train") / Path(*parts[:-1])
    
    input_actions_path = output_root / output_relative / relative.name.replace(
        "front_camera_timestamps_", "input_actions_"
    )
    start_frame_path = output_root / output_relative / relative.name.replace(
        "front_camera_timestamps_", "start_frame_"
    )

    write_actions(input_actions_path, actions, max_rows)
    write_start_frame(start_frame_path, end_frame_id, end_timestamp, nan_count)
    print(f"[ok] Wrote {num_actions} rows -> {input_actions_path}, start_frame -> {start_frame_path}")
    
    # Determine status
    if num_actions >= max_rows:
        status = "full"
    elif num_actions == 0:
        status = "zero"
    elif num_actions < 10:
        status = "few"
    else:
        status = "incomplete"
    
    return (status, num_actions)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    rides = list(discover_rides(args.input_root))
    
    if not rides:
        print("[warn] No ride directories found.", file=sys.stderr)
        return 0

    # Statistics tracking
    stats = {
        "full": 0,        # 501 rows
        "incomplete": 0,  # < 501 but >= 10
        "few": 0,         # < 10 but > 0
        "zero": 0,        # 0 rows
        "error": 0,       # errors that stopped processing
    }
    
    for ride_dir in rides:
        pairs = collect_pairs(ride_dir)
        if not pairs:
            print(f"[warn] No matching CSV pairs in {ride_dir}", file=sys.stderr)
            continue
        for front_path, control_path in pairs:
            status, num_actions = process_pair(
                front_path=front_path,
                control_path=control_path,
                input_root=args.input_root,
                output_root=args.output_root,
                max_frames=args.max_frames,
                max_rows=args.max_rows,
                frames_per_block=args.frames_per_block,
            )
            stats[status] += 1
    
    # Print and save summary statistics
    total = sum(stats.values())
    summary_lines = [
        "="*60,
        "SUMMARY STATISTICS",
        "="*60,
        f"Total videos processed: {total}",
    ]
    
    if total > 0:
        summary_lines.extend([
            f"  Full ({args.max_rows} rows): {stats['full']} ({100*stats['full']/total:.1f}%)",
            f"  Incomplete (< {args.max_rows} but >= 10 rows): {stats['incomplete']} ({100*stats['incomplete']/total:.1f}%)",
            f"  Few (< 10 rows): {stats['few']} ({100*stats['few']/total:.1f}%)",
            f"  Zero rows: {stats['zero']} ({100*stats['zero']/total:.1f}%)",
            f"  Errors: {stats['error']} ({100*stats['error']/total:.1f}%)",
        ])
    else:
        summary_lines.extend([
            f"  Full ({args.max_rows} rows): 0",
            "  Incomplete: 0",
            "  Few: 0",
            "  Zero: 0",
            "  Errors: 0",
        ])
    
    summary_lines.append("="*60)
    summary_text = "\n".join(summary_lines)
    
    # Print to console
    print("\n" + summary_text)
    
    # Save to file
    summary_path = args.output_root / "summary_statistics.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        f.write(summary_text + "\n")
    print(f"\nSummary saved to: {summary_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
