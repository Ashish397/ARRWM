"""Shared numerical contract for CARN-corrected recurrent commits.

The recurrent pipeline and every loss that claims to score the committed
state must use these helpers.  Keeping the schedule, absolute CARN level and
relative-displacement limiter here prevents the GAN from silently training on
an approximation of a different state transition.
"""
from __future__ import annotations

from typing import Tuple

import torch


def resolved_commit_alpha(
    *, step: int, target_alpha: float, start_step: int, ramp_steps: int,
) -> float:
    """Resolve the scalar commit dose at an absolute optimizer step."""
    step = int(step)
    start_step = int(start_step)
    ramp_steps = int(ramp_steps)
    target_alpha = float(target_alpha)
    if step < start_step:
        return 0.0
    if ramp_steps <= 0:
        return target_alpha
    fraction = min(
        1.0,
        max(0.0, float(step - start_step) / float(ramp_steps)),
    )
    return target_alpha * fraction


def absolute_carn_level(
    *, frame_start: int, frames_per_block: int, num_seed_chunks: int,
    max_level: int,
) -> int:
    """CARN level used by the real commit at one block-aligned position."""
    frame_start = int(frame_start)
    frames_per_block = int(frames_per_block)
    if frames_per_block <= 0 or frame_start % frames_per_block != 0:
        raise ValueError(
            "absolute CARN level requires a block-aligned frame_start; "
            f"got frame_start={frame_start}, frames_per_block="
            f"{frames_per_block}."
        )
    return min(
        max(
            0,
            (frame_start // frames_per_block) - (int(num_seed_chunks) - 1),
        ),
        int(max_level),
    )


def blend_carn_commit(
    raw: torch.Tensor,
    corrected: torch.Tensor,
    *,
    alpha: float,
    max_relative_shift: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Blend a CARN correction with an optional per-sample trust region.

    ``max_relative_shift`` constrains the *applied* displacement

        ||commit - raw||_2 / ||raw||_2

    independently for every leading batch row.  A value <= 0 disables the
    limiter.  The returned ``effective_alpha`` has one scalar per batch row;
    it is useful for sparse telemetry and for proving that GAN and recurrent
    memory used the same realized dose.
    """
    if raw.shape != corrected.shape:
        raise ValueError(
            "CARN commit blend requires identical raw/corrected shapes; "
            f"got raw={tuple(raw.shape)} corrected={tuple(corrected.shape)}."
        )
    alpha = float(alpha)
    max_relative_shift = float(max_relative_shift)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"commit alpha must be in [0, 1]; got {alpha}.")
    if max_relative_shift < 0.0:
        raise ValueError(
            "max_relative_shift must be >= 0; got "
            f"{max_relative_shift}."
        )

    batch = int(raw.shape[0])
    if alpha == 0.0 or raw.numel() == 0:
        effective = torch.zeros(
            batch, device=raw.device, dtype=torch.float32,
        )
        return raw, effective
    if alpha == 1.0 and max_relative_shift <= 0.0:
        # Preserve the historical full-dose path bit-for-bit. Reconstructing
        # ``corrected`` as raw + (corrected - raw) can round by one ULP.
        effective = torch.ones(
            batch, device=raw.device, dtype=torch.float32,
        )
        return corrected, effective

    delta = corrected - raw
    effective = torch.full(
        (batch,), alpha, device=raw.device, dtype=torch.float32,
    )
    if max_relative_shift > 0.0:
        raw_norm = raw.detach().float().flatten(1).norm(dim=1)
        delta_norm = delta.detach().float().flatten(1).norm(dim=1)
        cap = max_relative_shift * raw_norm / delta_norm.clamp_min(1.0e-12)
        effective = torch.minimum(effective, cap.clamp(min=0.0, max=1.0))

    view = effective.to(dtype=delta.dtype).view(
        batch, *([1] * (delta.dim() - 1))
    )
    return raw + view * delta, effective


def straight_through_value(
    raw_graph: torch.Tensor, committed_value: torch.Tensor,
) -> torch.Tensor:
    """Return committed values with an identity gradient to ``raw_graph``."""
    if raw_graph.shape != committed_value.shape:
        raise ValueError(
            "straight-through committed value requires identical shapes; "
            f"got raw={tuple(raw_graph.shape)} committed="
            f"{tuple(committed_value.shape)}."
        )
    # Put the algebraic zero on the graph term, then add it to the desired
    # value. ``raw + (committed - raw)`` can round by one ULP and would make
    # the claimed forward equality false even though its gradient is right.
    return committed_value.detach() + (raw_graph - raw_graph.detach())
