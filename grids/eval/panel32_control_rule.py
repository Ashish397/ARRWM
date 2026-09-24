"""Auditable control-decision rules for the nine-command Panel32 fleet.

The motion reader produces an independent native decision for each rollout.
Diagonal commands additionally depend on their two cardinal constituents: a
rollout commanded ``FR`` cannot be credited with control unless the matched
``F`` and ``R`` rollouts for the same context, model, and time window also
pass.  No-op retains its separate motion-magnitude rule.
"""

from __future__ import annotations

from collections.abc import Hashable

import numpy as np
import pandas as pd


ACTIONS = ("F", "FR", "R", "BR", "B", "BL", "L", "FL", "N")
CARDINAL_ACTIONS = ("F", "B", "L", "R")
DIAGONAL_COMPONENTS = {
    "FR": ("F", "R"),
    "FL": ("F", "L"),
    "BR": ("B", "R"),
    "BL": ("B", "L"),
}
CONTROL_ORB_STATIC_CUTOFF = 450
CONTROL_RULE_ID = "cardinal_preconditioned_direction_or_static_450_v1"

# Final vector-based paper rule.  The direction test is applied to every
# directional rollout independently.  Motion strength uses a 0.15 cutoff for
# cardinal commands and the per-axis-equivalent 0.15 / sqrt(2) cutoff for
# diagonal commands.  No-op fails at or above 0.15 magnitude.
VECTOR_CONTROL_ANGLE_DEGREES = 45.0
VECTOR_CONTROL_COSINE_CUTOFF = float(np.cos(np.deg2rad(VECTOR_CONTROL_ANGLE_DEGREES)))
VECTOR_CONTROL_CARDINAL_MAGNITUDE_CUTOFF = 0.15
VECTOR_CONTROL_DIAGONAL_MAGNITUDE_CUTOFF = float(
    VECTOR_CONTROL_CARDINAL_MAGNITUDE_CUTOFF / np.sqrt(2.0)
)
VECTOR_CONTROL_NOOP_MAGNITUDE_CUTOFF = 0.15
VECTOR_CONTROL_RULE_ID = "vector_cos45_mag015_diag015_over_sqrt2_v1"


def _binary(series: pd.Series, name: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any() or not values.isin([0, 1]).all():
        raise ValueError(f"{name} must be complete and binary")
    return values.astype(int)


def apply_cardinal_precondition_rule(
    frame: pd.DataFrame,
    *,
    context_column: str = "uid",
) -> pd.DataFrame:
    """Return native and cardinal-preconditioned control decisions.

    Required input columns are ``context_column``, ``model``,
    ``window_start_s``, ``direction``, ``direction_failure``,
    ``near_static_450``, and ``noop_motion_010``.  Every
    (context, model, window) group must contain the complete nine-action set.
    """

    required = {
        context_column,
        "model",
        "window_start_s",
        "direction",
        "direction_failure",
        "near_static_450",
        "noop_motion_010",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"control rule missing columns: {sorted(missing)}")

    result = frame.copy()
    keys = [context_column, "model", "window_start_s"]
    expected = set(ACTIONS)
    for key, group in result.groupby(keys, sort=False, dropna=False):
        actions = list(group.direction.astype(str))
        if len(actions) != len(ACTIONS) or set(actions) != expected:
            raise ValueError(
                f"incomplete/duplicate action set for {key}: {sorted(actions)}"
            )

    directional = result.direction.ne("N")
    native = pd.Series(np.nan, index=result.index, dtype=float)
    native.loc[directional] = (
        _binary(
            result.loc[directional, "direction_failure"],
            "direction_failure",
        ).eq(1)
        | _binary(
            result.loc[directional, "near_static_450"],
            "near_static_450",
        ).eq(1)
    ).astype(int)
    result["native_control_failure"] = native

    noop = result.direction.eq("N")
    noop_decisions = _binary(
        result.loc[noop, "noop_motion_010"], "noop_motion_010"
    )
    result["diagonal_precondition_failure"] = np.where(
        result.direction.isin(DIAGONAL_COMPONENTS), 0.0,
        np.where(directional, 0.0, np.nan),
    )
    result["failed_cardinal_components"] = ""
    result["control_failure"] = 0
    result.loc[directional, "control_failure"] = native.loc[directional].astype(int)
    result.loc[noop, "control_failure"] = noop_decisions.to_numpy(dtype=int)

    cardinal = result[result.direction.isin(CARDINAL_ACTIONS)]
    lookup: dict[tuple[Hashable, Hashable, Hashable, str], int] = {
        (getattr(row, context_column), row.model, row.window_start_s, row.direction):
        int(row.native_control_failure)
        for row in cardinal.itertuples(index=False)
    }
    for diagonal, components in DIAGONAL_COMPONENTS.items():
        mask = result.direction.eq(diagonal)
        for index, row in result.loc[mask].iterrows():
            failed = [
                component for component in components
                if lookup[(
                    row[context_column], row["model"],
                    row["window_start_s"], component,
                )] == 1
            ]
            precondition_failure = int(bool(failed))
            result.at[index, "diagonal_precondition_failure"] = precondition_failure
            result.at[index, "failed_cardinal_components"] = "+".join(failed)
            result.at[index, "control_failure"] = int(
                bool(result.at[index, "native_control_failure"])
                or precondition_failure
            )

    result["control_failure"] = result.control_failure.astype(int)
    result["control_rule_id"] = CONTROL_RULE_ID
    return result


def apply_vector_control_rule(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the final independent vector-based control rule.

    Required columns are ``direction``, ``cosine``, and ``magnitude``.  A
    directional rollout fails if its realized vector is more than 45 degrees
    from the commanded vector or is too weak.  Cardinal commands use a 0.15
    magnitude cutoff and diagonal commands use ``0.15 / sqrt(2)``.  No-op
    fails when magnitude is at least 0.15.
    """

    required = {"direction", "cosine", "magnitude"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"vector control rule missing columns: {sorted(missing)}")

    result = frame.copy()
    directions = result["direction"].astype(str)
    if not directions.isin(ACTIONS).all():
        unknown = sorted(set(directions) - set(ACTIONS))
        raise ValueError(f"unknown control actions: {unknown}")

    magnitude = pd.to_numeric(result["magnitude"], errors="coerce")
    cosine = pd.to_numeric(result["cosine"], errors="coerce")
    if magnitude.isna().any() or not np.isfinite(magnitude).all():
        raise ValueError("magnitude must be complete and finite")
    directional = directions.ne("N")
    if cosine.loc[directional].isna().any() or not np.isfinite(
        cosine.loc[directional]
    ).all():
        raise ValueError("directional cosine must be complete and finite")

    diagonal = directions.isin(DIAGONAL_COMPONENTS)
    cardinal = directions.isin(CARDINAL_ACTIONS)
    result["angle_failure"] = 0
    result.loc[directional, "angle_failure"] = (
        cosine.loc[directional] < VECTOR_CONTROL_COSINE_CUTOFF
    ).astype(int)
    result["weak_magnitude_failure"] = 0
    result.loc[cardinal, "weak_magnitude_failure"] = (
        magnitude.loc[cardinal] < VECTOR_CONTROL_CARDINAL_MAGNITUDE_CUTOFF
    ).astype(int)
    result.loc[diagonal, "weak_magnitude_failure"] = (
        magnitude.loc[diagonal] < VECTOR_CONTROL_DIAGONAL_MAGNITUDE_CUTOFF
    ).astype(int)
    result["noop_magnitude_failure"] = 0
    noop = directions.eq("N")
    result.loc[noop, "noop_magnitude_failure"] = (
        magnitude.loc[noop] >= VECTOR_CONTROL_NOOP_MAGNITUDE_CUTOFF
    ).astype(int)
    result["control_failure"] = 0
    result.loc[directional, "control_failure"] = (
        result.loc[directional, "angle_failure"].eq(1)
        | result.loc[directional, "weak_magnitude_failure"].eq(1)
    ).astype(int)
    result.loc[noop, "control_failure"] = result.loc[
        noop, "noop_magnitude_failure"
    ].astype(int)
    result["control_failure"] = result["control_failure"].astype(int)
    result["control_rule_id"] = VECTOR_CONTROL_RULE_ID
    return result
