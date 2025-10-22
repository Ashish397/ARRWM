import argparse
import csv
import sys
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple


DEFAULT_INPUT_ROOT = Path("/projects/u5as/frodobots")
DEFAULT_OUTPUT_ROOT = Path("/projects/u5as/frodobots_actions")
DEFAULT_MAX_FRAMES = 6000
DEFAULT_MAX_ROWS = 1500


@dataclass(frozen=True)
class ControlSample:
    timestamp: float
    linear: float
    angular: float
    rpm_1: float
    rpm_2: float
    rpm_3: float
    rpm_4: float


@dataclass(frozen=True)
class CameraSample:
    frame_id: int
    timestamp: float


@dataclass(frozen=True)
class AggregatedAction:
    frame_id: int
    linear: float
    angular: float
    rpm_1: float
    rpm_2: float
    rpm_3: float
    rpm_4: float


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate averaged input/output action files from front camera timestamps and control data."
        )
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
        "--chunk-size",
        type=int,
        default=4,
        help="Number of frames per chunk when aggregating actions.",
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
        help="Maximum number of output action rows per file (default 1500).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Process only the first ride in train/output_rides_0 for quick validation.",
    )
    return parser.parse_args(argv)


def discover_rides(input_root: Path, test_mode: bool) -> Iterator[Path]:
    """Yield ride directories that contain the required CSV files."""
    if test_mode:
        candidate_root = input_root / "train" / "output_rides_0"
        if not candidate_root.is_dir():
            raise FileNotFoundError(
                f"Expected directory {candidate_root} when running in test mode."
            )
        ride_dirs = sorted(d for d in candidate_root.iterdir() if d.is_dir())
        if ride_dirs:
            yield ride_dirs[0]
        return

    for split_dir in sorted(
        d for d in input_root.iterdir() if d.is_dir()
    ):
        for output_dir in sorted(
            d for d in split_dir.iterdir() if d.is_dir() and d.name.startswith("output_rides_")
        ):
            for ride_dir in sorted(d for d in output_dir.iterdir() if d.is_dir()):
                yield ride_dir


def collect_pairs(ride_dir: Path) -> List[Tuple[Path, Path]]:
    """Return matched (front camera, control) CSV pairs for the ride."""
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
        _ = next(reader, None)
        for row in reader:
            if not row:
                continue
            try:
                frame_id = int(row[0])
                timestamp = float(row[1])
            except (ValueError, IndexError):
                continue
            samples.append(CameraSample(frame_id=frame_id, timestamp=timestamp))
            if len(samples) >= max_frames:
                break
    return samples


def read_control_samples(path: Path) -> List[ControlSample]:
    samples: List[ControlSample] = []
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        _ = next(reader, None)
        for row in reader:
            if not row or row[0] == "linear":
                continue
            try:
                linear = float(row[0])
                angular = float(row[1])
                rpm_1 = float(row[2])
                rpm_2 = float(row[3])
                rpm_3 = float(row[4])
                rpm_4 = float(row[5])
                timestamp = float(row[-1])
            except (ValueError, IndexError):
                continue
            samples.append(
                ControlSample(
                    timestamp=timestamp,
                    linear=linear,
                    angular=angular,
                    rpm_1=rpm_1,
                    rpm_2=rpm_2,
                    rpm_3=rpm_3,
                    rpm_4=rpm_4,
                )
            )
    samples.sort(key=lambda sample: sample.timestamp)
    return samples


def select_control_window(
    control_timestamps: Sequence[float],
    controls: Sequence[ControlSample],
    start_ts: float,
    end_ts: float,
) -> Sequence[ControlSample]:
    if not controls:
        return ()

    left_idx = bisect_left(control_timestamps, start_ts) - 1
    if left_idx < 0:
        left_idx = 0

    right_pos = bisect_right(control_timestamps, end_ts)
    if right_pos >= len(controls):
        right_idx = len(controls) - 1
    else:
        right_idx = right_pos

    if right_idx < left_idx:
        right_idx = left_idx

    return controls[left_idx : right_idx + 1]


def compute_actions(
    camera_samples: Sequence[CameraSample],
    control_samples: Sequence[ControlSample],
    chunk_size: int,
    max_actions: int,
) -> List[AggregatedAction]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    limited_frames = len(camera_samples) - (len(camera_samples) % chunk_size)
    control_timestamps = [sample.timestamp for sample in control_samples]
    actions: List[AggregatedAction] = []
    for chunk_start in range(0, limited_frames, chunk_size):
        chunk = camera_samples[chunk_start : chunk_start + chunk_size]
        if not chunk:
            continue
        start_ts = chunk[0].timestamp
        end_ts = chunk[-1].timestamp
        window = select_control_window(control_timestamps, control_samples, start_ts, end_ts)

        if not window:
            continue

        weight = len(window)
        linear_avg = sum(sample.linear for sample in window) / weight
        angular_avg = sum(sample.angular for sample in window) / weight
        rpm_1_avg = sum(sample.rpm_1 for sample in window) / weight
        rpm_2_avg = sum(sample.rpm_2 for sample in window) / weight
        rpm_3_avg = sum(sample.rpm_3 for sample in window) / weight
        rpm_4_avg = sum(sample.rpm_4 for sample in window) / weight
        actions.append(
            AggregatedAction(
                frame_id=chunk[0].frame_id,
                linear=linear_avg,
                angular=angular_avg,
                rpm_1=rpm_1_avg,
                rpm_2=rpm_2_avg,
                rpm_3=rpm_3_avg,
                rpm_4=rpm_4_avg,
            )
        )
        if len(actions) >= max_actions:
            break
    return actions


def write_input_actions(
    output_path: Path,
    actions: Sequence[AggregatedAction],
    max_rows: int,
    chunk_size: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_to_write = list(actions[:max_rows])
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_id", "linear", "angular"])
        for action in rows_to_write:
            writer.writerow(
                [
                    int(action.frame_id/chunk_size),
                    f"{action.linear:.6f}",
                    f"{action.angular:.6f}",
                ]
            )


def write_output_actions(
    output_path: Path,
    actions: Sequence[AggregatedAction],
    max_rows: int,
    chunk_size: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_to_write = list(actions[:max_rows])
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_id", "rpm_1", "rpm_2", "rpm_3", "rpm_4"])
        for action in rows_to_write:
            writer.writerow(
                [
                    int(action.frame_id/chunk_size),
                    f"{action.rpm_1:.6f}",
                    f"{action.rpm_2:.6f}",
                    f"{action.rpm_3:.6f}",
                    f"{action.rpm_4:.6f}",
                ]
            )


def process_pair(
    front_path: Path,
    control_path: Path,
    input_root: Path,
    output_root: Path,
    chunk_size: int,
    max_frames: int,
    max_rows: int,
) -> None:
    camera_samples = read_camera_samples(front_path, max_frames)
    control_samples = read_control_samples(control_path)
    if not camera_samples:
        print(f"[skip] No camera samples in {front_path}", file=sys.stderr)
        return
    if not control_samples:
        print(f"[skip] No control samples in {control_path}", file=sys.stderr)
        return

    actions = compute_actions(
        camera_samples,
        control_samples,
        chunk_size,
        max_actions=max_rows,
    )
    if not actions:
        print(f"[skip] No actions computed for {front_path}", file=sys.stderr)
        return

    try:
        relative = front_path.relative_to(input_root)
    except ValueError:
        relative = front_path.resolve().relative_to(input_root.resolve())

    input_actions_path = output_root / relative.parent / relative.name.replace(
        "front_camera_timestamps_",
        "input_actions_",
    )
    output_actions_path = output_root / relative.parent / relative.name.replace(
        "front_camera_timestamps_",
        "output_actions_",
    )

    if len(actions) < max_rows:
        print(
            f"[warn] Only {len(actions)} actions available for {front_path}; expected {max_rows}.",
            file=sys.stderr,
        )

    write_input_actions(input_actions_path, actions, max_rows, chunk_size)
    write_output_actions(output_actions_path, actions, max_rows, chunk_size)
    written_rows = min(len(actions), max_rows)
    print(
        f"[ok] Wrote {written_rows} rows -> {input_actions_path} and {output_actions_path}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    try:
        rides = list(discover_rides(args.input_root, args.test))
    except FileNotFoundError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    if not rides:
        print("[warn] No ride directories found.")
        return 0

    for ride_dir in rides:
        pairs = collect_pairs(ride_dir)
        if not pairs:
            print(f"[warn] No matching CSV pairs in {ride_dir}", file=sys.stderr)
            continue
        for front_path, control_path in pairs:
            process_pair(
                front_path=front_path,
                control_path=control_path,
                input_root=args.input_root,
                output_root=args.output_root,
                chunk_size=args.chunk_size,
                max_frames=args.max_frames,
                max_rows=args.max_rows,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
