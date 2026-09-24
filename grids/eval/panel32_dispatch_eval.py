#!/usr/bin/env python3
"""Queue missing panel32 evaluation batches on idle four-GPU holders."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


TASKS = ("geometry", "conjuration", "control", "style", "cpu", "longreloc")


def holder_sort_key(job_id: str) -> tuple[int, int]:
    """Sort both ordinary Slurm IDs and array-element IDs numerically."""
    fields = job_id.split("_", 1)
    return int(fields[0]), int(fields[1]) if len(fields) == 2 else -1


def remaining_seconds(value: str) -> int:
    """Parse Slurm's ``%L`` value for longest-remaining-first dispatch."""
    try:
        day_text, clock = value.split("-", 1) if "-" in value else ("0", value)
        fields = [int(field) for field in clock.split(":")]
        if len(fields) == 3:
            hours, minutes, seconds = fields
        elif len(fields) == 2:
            hours, minutes, seconds = 0, *fields
        else:
            return 0
        return int(day_text) * 86400 + hours * 3600 + minutes * 60 + seconds
    except (TypeError, ValueError):
        return 0


def running_holders() -> list[str]:
    result = subprocess.run(
        ["squeue", "-h", "-u", subprocess.check_output(["id", "-un"], text=True).strip(),
         "-o", "%A|%T|%j|%L"],
        check=True, capture_output=True, text=True,
    )
    holders: dict[str, int] = {}
    for line in result.stdout.splitlines():
        job, state, name, remaining = line.split("|", 3)
        if state == "RUNNING" and name == "panel32-hold":
            holders[job] = remaining_seconds(remaining)
    # Long GPU producers should land on allocations with the most time left;
    # the job ID provides a deterministic tie-break only.
    return sorted(holders, key=lambda job: (-holders[job], holder_sort_key(job)))


def inflight(holder_logs: Path) -> set[str]:
    phases: set[str] = set()
    for pattern in (".holder_cmd_*.sh", ".holder_running_*.sh"):
        for path in holder_logs.glob(pattern):
            try:
                phases.update(re.findall(r"eval:[a-z-]+(?::\d+)?", path.read_text()))
            except OSError:
                pass
    return phases


def idle_holders(holder_logs: Path) -> list[str]:
    answer = []
    for job in running_holders():
        if not (holder_logs / f".holder_cmd_{job}.sh").exists() and not (
                holder_logs / f".holder_running_{job}.sh").exists():
            answer.append(job)
    return answer


def complete(markers: Path, phase: str, shards: int) -> bool:
    if phase.startswith("eval:preflight:"):
        batch = int(phase.rsplit(":", 1)[1])
        task = "preflight"
    else:
        _, task, batch_text = phase.split(":")
        batch = int(batch_text)
    return all((markers / f"{task}_shard{batch * 4 + lane}.COMPLETE").is_file()
               for lane in range(4))


def desired(mode: str, markers: Path, shards: int) -> list[str]:
    batches = shards // 4
    if mode == "preflight":
        phases = [f"eval:preflight:{batch}" for batch in range(batches)]
    elif mode == "metrics":
        # Put the long GPU producers first; CPU scans fill holders as they free.
        phases = [f"eval:{task}:{batch}" for task in TASKS for batch in range(batches)]
    else:
        phases = [f"eval:{mode}"]
    if mode in {"preflight", "metrics"}:
        phases = [phase for phase in phases if not complete(markers, phase, shards)]
    return phases


def require_prerequisites(mode: str, evaluation: Path, shards: int) -> None:
    """Reject out-of-order dispatch instead of burning holders on failures."""
    setup = evaluation / "logs/SETUP_COMPLETE"
    preflight = evaluation / "preflight/ACTION_PREFLIGHT_COMPLETE"
    markers = evaluation / "markers"
    if mode != "setup" and not setup.is_file():
        raise SystemExit("panel32 evaluation setup is not complete")
    if mode == "preflight-finalize":
        missing = [
            shard for shard in range(shards)
            if not (markers / f"preflight_shard{shard}.COMPLETE").is_file()
        ]
        if missing:
            raise SystemExit(
                f"cannot finalize preflight; missing {len(missing)} shards: {missing[:8]}")
    if mode in {"metrics", "finish"} and not preflight.is_file():
        raise SystemExit("action-convention preflight is not complete")
    if mode == "finish":
        missing = [
            f"{task}:{shard}"
            for task in TASKS for shard in range(shards)
            if not (markers / f"{task}_shard{shard}.COMPLETE").is_file()
        ]
        if missing:
            raise SystemExit(
                f"cannot finish; missing {len(missing)} metric shards: {missing[:8]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("setup", "preflight", "preflight-finalize",
                                         "metrics", "finish"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=48)
    args = parser.parse_args()
    if args.shards < 4 or args.shards % 4:
        raise SystemExit("--shards must be a positive multiple of four")

    evaluation = args.root / "panel32_stage/eval_final"
    holder_logs = args.root / "panel32_stage/logs/holders"
    markers = evaluation / "markers"
    require_prerequisites(args.mode, evaluation, args.shards)
    active = inflight(holder_logs)
    pending = [phase for phase in desired(args.mode, markers, args.shards)
               if phase not in active]
    idle = idle_holders(holder_logs)
    if args.mode in {"setup", "preflight-finalize", "finish"}:
        pending = pending[:1]
        idle = idle[:1]
    queue = args.code / "grids/eval/panel32_queue_holder.py"
    sequence = args.code / "sbatch/u6qf/panel32_eval_sequence.sh"
    assigned = []
    for job, phase in zip(idle, pending):
        subprocess.run([
            str(args.root / "miniforge3/envs/arrwm/bin/python"), str(queue),
            job, phase, "--root", str(holder_logs), "--sequence", str(sequence),
        ], check=True)
        assigned.append((job, phase))
    print({"mode": args.mode, "assigned": assigned, "idle_seen": len(idle),
           "pending_after_assignment": len(pending) - len(assigned),
           "inflight": sorted(active)})


if __name__ == "__main__":
    main()
