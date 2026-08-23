"""Noise-schedule helpers for the action-forcing random-timestep ODE stage.

``gen_lmdb.py`` snapshots the teacher's trajectory at ODE step indices
``SNAPSHOT_STEPS = [0, 18, 36, 40, 44, 46, -1]`` using a
``FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)`` with
``num_inference_steps=48``. The snapshot indexed ``-1`` is the teacher's
fully denoised x0 output; ``0`` is the pure-noise starting latent.

The random-timestep recipe draws from a user-configurable subset of those
snapshots via ``random_steps`` (default ``[0, 36, 44, 46]``). A config can
provide its own ``snapshot_steps`` vocabulary for checkpoints recorded on a
different solver grid, such as v14e's 20-step KL trajectory. The
``-1`` / teacher_x0 entry is intentionally NOT in the default pool: at
timestep 0 the wrapper returns ``pred_x0 == xt`` and the resulting frame
trains a zero-loss identity example that dilutes the MSE. Opt in by
adding ``-1`` to ``random_steps`` explicitly if you want it back.
"""

from __future__ import annotations
from typing import List, Sequence

import torch


# Full 7-entry snapshot pool. Each entry is the ODE step index the
# snapshot latent was captured at; ``-1`` is aliased as "after step N".
import os as _os
# ARRWM 14e pilot: env overrides for the chained 20-step LMDBs
# (AF_SNAPSHOT_STEPS="0,15,18,19,-1" AF_EVAL_STEPS=20); defaults
# unchanged for the v14/v14d 48-step datasets.
SNAPSHOT_STEPS: List[int] = (
    [int(x) for x in _os.environ["AF_SNAPSHOT_STEPS"].split(",")]
    if "AF_SNAPSHOT_STEPS" in _os.environ else [0, 18, 36, 40, 44, 46, -1])

# Default: use ALL 7 stored snapshots.
DEFAULT_USABLE_INDICES: List[int] = [0, 1, 2, 3, 4, 5, 6]

# Default random-mode denoising pool in ODE-step-value vocabulary.
# 4 entries spanning the noisy portion of the ODE arc (``0`` = pure noise,
# ``46`` = near-clean). ``-1`` (teacher_x0 / timestep 0) is deliberately
# omitted because at timestep 0 the flow-matching wrapper returns
# ``pred_x0 == xt`` and the loss on that chunk is zero by construction;
# keeping it in the pool wastes ~1/N of the batch on identity frames.
DEFAULT_RANDOM_STEPS: List[int] = [0, 36, 44, 46]

EVAL_STEPS: int = int(_os.environ.get("AF_EVAL_STEPS", "48"))
TIMESTEP_SHIFT: float = 5.0

# Number of chunks / block in the 21-frame window.
NUM_CHUNKS: int = 7
NUM_FRAME_PER_BLOCK: int = 3


def step_value_to_snap_idx(v: int) -> int:
    """Map an ODE step value (``-1``/``0``/``18``/...) to its index in ``SNAPSHOT_STEPS``.

    Raises ``ValueError`` if ``v`` is not a recorded snapshot step.
    """
    try:
        return SNAPSHOT_STEPS.index(int(v))
    except ValueError as e:
        raise ValueError(
            f"step value {v} is not one of SNAPSHOT_STEPS={SNAPSHOT_STEPS}"
        ) from e


def resolve_denoising_step_list(
    usable_stored_indices: Sequence[int] | None = None,
    num_inference_steps: int = EVAL_STEPS,
    shift: float = TIMESTEP_SHIFT,
    snapshot_steps: Sequence[int] | None = None,
) -> torch.Tensor:
    """Return the fp32 teacher-scheduler timestep for each kept snapshot.

    Indexing follows the order in ``usable_stored_indices`` (which indexes
    into ``snapshot_steps``, or ``SNAPSHOT_STEPS`` when omitted). For the
    default full pool ``[0..6]`` the return is:

        denoising_step_list[0] = sched.timesteps[0]   (high, ~999)
        denoising_step_list[1] = sched.timesteps[18]
        denoising_step_list[2] = sched.timesteps[36]
        denoising_step_list[3] = sched.timesteps[40]
        denoising_step_list[4] = sched.timesteps[44]
        denoising_step_list[5] = sched.timesteps[46]
        denoising_step_list[6] = 0.0                  (teacher_x0)
    """
    from utils.scheduler import FlowMatchScheduler

    if usable_stored_indices is None:
        usable_stored_indices = DEFAULT_USABLE_INDICES
    usable_stored_indices = list(usable_stored_indices)
    snapshot_steps = list(SNAPSHOT_STEPS if snapshot_steps is None else snapshot_steps)

    sched = FlowMatchScheduler(shift=shift, sigma_min=0.0, extra_one_step=True)
    sched.set_timesteps(num_inference_steps=num_inference_steps, denoising_strength=1.0)
    ts = sched.timesteps.detach().to(torch.float32).cpu()
    N = int(ts.shape[0])
    assert N == num_inference_steps, (
        f"FlowMatchScheduler returned {N} timesteps; expected {num_inference_steps}"
    )

    out = []
    for idx in usable_stored_indices:
        if not (0 <= idx < len(snapshot_steps)):
            raise IndexError(
                f"usable_stored_indices contains {idx}; must be in "
                f"[0, {len(snapshot_steps)})"
            )
        s = snapshot_steps[idx]
        if s == -1:
            out.append(torch.tensor(0.0, dtype=torch.float32))
        elif s < 0:
            raise ValueError(
                f"snapshot step {s} is invalid; only -1 may represent "
                "the clean endpoint"
            )
        elif s >= N:
            raise ValueError(
                f"snapshot step {s} is outside the {N}-step scheduler; "
                "set config.snapshot_steps to the vocabulary used to build "
                "this ODE checkpoint"
            )
        else:
            out.append(ts[int(s)])
    return torch.stack(out, dim=0)
