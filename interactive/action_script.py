"""Deterministic scripted action sequence for the bench harness.

Exported as plain data so the bench, the UI's demo mode and any future
regression test drive the engine through exactly the same trajectory.

Convention (from engine_api.Action): throttle = pca_0, steer = pca_1, both
normalized to [-1, 1]. One Action is latched for one whole horizon
(cfg.horizon_chunks chunks), so the sequence length == number of horizons.
"""

from __future__ import annotations

from typing import List, Tuple

try:
    from interactive.engine_api import Action
except ImportError:  # running with interactive/ itself on sys.path
    from engine_api import Action  # type: ignore

# name, throttle, steer  — 8 horizons, one per entry.
_SPEC: Tuple[Tuple[str, float, float], ...] = (
    ("straight-idle",      0.0,  0.0),
    ("straight-throttle",  1.0,  0.0),
    ("left-throttle",      0.7, -0.5),
    ("right-throttle",     0.7,  0.5),
    ("hard-left",          0.3, -1.0),
    ("hard-right",         0.3,  1.0),
    ("reverse-brake",     -1.0,  0.0),
    ("straight-throttle2", 1.0,  0.0),
)

SCRIPTED_SEQUENCE: List[Action] = [
    Action(throttle=t, steer=s) for _, t, s in _SPEC
]

SCRIPTED_NAMES: List[str] = [n for n, _, _ in _SPEC]

NUM_HORIZONS: int = len(SCRIPTED_SEQUENCE)

assert NUM_HORIZONS == 8, "the bench script is defined as 8 horizons"


def describe() -> str:
    """One-line-per-horizon human readable dump of the script."""
    return "\n".join(
        f"{i}: {n:<18} throttle={a.throttle:+.2f} steer={a.steer:+.2f}"
        for i, (n, a) in enumerate(zip(SCRIPTED_NAMES, SCRIPTED_SEQUENCE))
    )


if __name__ == "__main__":
    print(describe())
