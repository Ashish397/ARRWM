#!/usr/bin/env python3
"""Atomically queue one task sequence for a running panel32 holder."""

from __future__ import annotations

import argparse
import os
import shlex
import tempfile
from pathlib import Path


KINDS = {"external", "yume", "ours", "ode", "smoke", "eval"}


def validate_task(task: str) -> None:
    fields = task.split(":")
    if not fields or fields[0] not in KINDS:
        raise ValueError(f"invalid panel32 task: {task!r}")
    if fields[0] == "eval":
        if len(fields) == 2 and fields[1] in {
                "setup", "matrix-strict", "preflight", "preflight-finalize", "finish"}:
            return
        if (len(fields) == 3 and
                fields[1] in {"preflight", "cpu", "style", "control", "geometry",
                              "conjuration", "conjuration-v2",
                              "conjuration-audit-v2", "conjuration-audit-kept-v2",
                              "longreloc"} and
                fields[2].isdigit()):
            return
        raise ValueError(f"invalid panel32 evaluation task: {task!r}")
    if fields[0] in {"yume", "smoke"} and len(fields) == 2:
        return
    if fields[0] not in {"yume", "smoke"} and len(fields) == 3:
        return
    raise ValueError(f"invalid panel32 task fields: {task!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_id")
    parser.add_argument("tasks", nargs="+")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--prefix", default=".holder",
                        choices=(".holder", ".frodo8_holder"))
    args = parser.parse_args()
    for task in args.tasks:
        validate_task(task)
    args.root.mkdir(parents=True, exist_ok=True)
    command = args.root / f"{args.prefix}_cmd_{args.job_id}.sh"
    running = args.root / f"{args.prefix}_running_{args.job_id}.sh"
    if command.exists() or running.exists():
        raise SystemExit(f"holder {args.job_id} already has queued/running work")
    body = "#!/bin/bash\nset -euo pipefail\nexec bash {} {}\n".format(
        shlex.quote(str(args.sequence)),
        " ".join(shlex.quote(task) for task in args.tasks),
    )
    fd, temporary = tempfile.mkstemp(prefix=f"{args.prefix}_cmd_{args.job_id}.", dir=args.root)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o755)
        # Publish without replacement.  Multiple dispatcher invocations may
        # observe the same holder as idle; an atomic hard-link makes exactly
        # one queue operation win instead of silently overwriting a task that
        # another dispatcher just installed.
        try:
            os.link(temporary, command)
        except FileExistsError as exc:
            raise SystemExit(
                f"holder {args.job_id} received concurrent queued work") from exc
        os.unlink(temporary)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"queued holder={args.job_id} tasks={','.join(args.tasks)} command={command}")


if __name__ == "__main__":
    main()
