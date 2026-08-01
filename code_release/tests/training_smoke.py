#!/usr/bin/env python3
"""End-to-end behaviour gate: run N training steps and compare the losses.

The unit tests pin components; this pins the whole pipeline — data loading,
action encoding, the LoRA-adapted DiT, the critic, and the optimiser step —
against recorded per-step losses. Run it before a batch of edits and after.

    python tests/training_smoke.py --config configs/<cfg>.yaml \
        --logdir logs/smoke --save     # record the reference
    python tests/training_smoke.py --config configs/<cfg>.yaml \
        --logdir logs/smoke --check    # assert nothing moved

The reference is machine-specific: it records real losses, so a golden
recorded on other hardware will not match. Record it where you check it.

The config is deliberately reduced (9 frames, LoRA rank 32, small critic) so it
fits one 32 GB card; the paper's own config needs 4x GH200 and OOMs here. That
is fine for this purpose: the gate is *differential*, comparing the same reduced
config before and after a change, not reproducing training.

Requires the environment described in docs/RUNNING.md (Wan2.1 weights, an
encoded-zarr root, motion and caption roots). Skips with a clear message if
those are absent.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REFERENCE = Path(__file__).parent / "goldens" / "training_smoke.json"
STEP_RE = re.compile(r"step (\d+) \| loss ([0-9.]+) \| flow ([0-9.]+)")

# Measured on an RTX 5090 (torch 2.8, CUDA 12.8): two runs of identical code and
# environment give a bit-identical step 1 but differ by ~5e-4 from step 2 on.
# Step 1's loss is a forward pass from the seeded initial state, so it is exact.
# Later steps inherit the first backward, and the backward is not deterministic
# (flex-attention backward and 3D-conv atomics), so seeding alone cannot fix it.
#
# Step 1 is therefore asserted exactly — the sharpest available signal — and
# later steps within a tolerance an order of magnitude above the observed
# run-to-run spread, so real behaviour changes still show up while kernel
# nondeterminism does not.
FIRST_STEP_TOLERANCE = 0.0
LATER_STEP_TOLERANCE = 5e-3


def run(config: Path, logdir: Path) -> list[dict]:
    env = {
        **os.environ,
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29799",
        "RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1",
        "ARRWM_ACTION_ENCODER": "pca_raw",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HUB_OFFLINE": "1",
    }
    proc = subprocess.run(
        [sys.executable, "train.py", "--config_path", str(config),
         "--logdir", str(logdir), "--disable-wandb", "--no_visualize", "--no_save"],
        cwd=REPO, env=env, capture_output=True, text=True,
    )
    steps = [
        {"step": int(s), "loss": float(l), "flow": float(f)}
        for s, l, f in STEP_RE.findall(proc.stdout + proc.stderr)
    ]
    if not steps:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-25:])
        raise SystemExit(f"no training steps completed (exit {proc.returncode}):\n{tail}")
    return steps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True, help="reduced smoke config")
    ap.add_argument("--logdir", type=Path, required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--save", action="store_true", help="record the reference")
    mode.add_argument("--check", action="store_true", help="compare against it")
    args = ap.parse_args()

    steps = run(args.config, args.logdir)

    if args.save:
        REFERENCE.parent.mkdir(parents=True, exist_ok=True)
        REFERENCE.write_text(json.dumps({"steps": steps}, indent=1) + "\n")
        print(f"recorded {len(steps)} steps -> {REFERENCE}")
        for s in steps:
            print(f"  step {s['step']}: loss {s['loss']:.6f}")
        return 0

    if not REFERENCE.exists():
        raise SystemExit(f"no reference at {REFERENCE}; run --save first")
    want = json.loads(REFERENCE.read_text())["steps"]

    if len(steps) != len(want):
        raise SystemExit(f"step count changed: {len(steps)} vs reference {len(want)}")

    drifted = 0
    for i, (a, b) in enumerate(zip(steps, want)):
        tol = FIRST_STEP_TOLERANCE if i == 0 else LATER_STEP_TOLERANCE
        delta = abs(a["loss"] - b["loss"])
        bad = delta > tol or a["step"] != b["step"]
        drifted += bad
        print(f"  step {a['step']}: {a['loss']:.6f} vs {b['loss']:.6f}  "
              f"|d|={delta:.2e} tol={tol:.0e}  [{'DRIFT' if bad else 'ok'}]")
    if drifted:
        raise SystemExit(f"{drifted} step(s) drifted — the change altered training")
    print(f"all {len(steps)} steps within tolerance of the reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
