from __future__ import annotations

import pandas as pd
import pytest

from grids.eval.panel32_control_rule import (
    ACTIONS,
    CONTROL_RULE_ID,
    apply_cardinal_precondition_rule,
)


def fixture() -> pd.DataFrame:
    rows = []
    for action in ACTIONS:
        rows.append({
            "uid": "context-1",
            "scene": f"context-1_{action}",
            "model": "model",
            "window_start_s": 0,
            "direction": action,
            "direction_failure": 0 if action != "N" else float("nan"),
            "near_static_450": 0 if action != "N" else float("nan"),
            "noop_motion_010": 0 if action == "N" else float("nan"),
        })
    return pd.DataFrame(rows)


def test_diagonal_requires_both_cardinal_components_and_native_pass() -> None:
    frame = fixture()
    frame.loc[frame.direction == "F", "direction_failure"] = 1
    frame.loc[frame.direction == "L", "near_static_450"] = 1
    frame.loc[frame.direction == "BR", "direction_failure"] = 1

    scored = apply_cardinal_precondition_rule(frame).set_index("direction")

    # F propagates to both forward diagonals; L also independently propagates
    # to FL.  The diagonal's own motion test remains a necessary condition.
    assert scored.loc["F", "control_failure"] == 1
    assert scored.loc["FR", "control_failure"] == 1
    assert scored.loc["FR", "failed_cardinal_components"] == "F"
    assert scored.loc["FL", "control_failure"] == 1
    assert scored.loc["FL", "failed_cardinal_components"] == "F+L"
    assert scored.loc["BR", "control_failure"] == 1
    assert scored.loc["BR", "diagonal_precondition_failure"] == 0
    assert scored.loc["BL", "control_failure"] == 1
    assert scored.loc["BL", "failed_cardinal_components"] == "L"
    assert scored.loc["R", "control_failure"] == 0
    assert scored.loc["B", "control_failure"] == 0
    assert scored.loc["N", "control_failure"] == 0
    assert set(scored.control_rule_id) == {CONTROL_RULE_ID}


def test_noop_retains_separate_motion_rule() -> None:
    frame = fixture()
    frame.loc[frame.direction == "N", "noop_motion_010"] = 1
    scored = apply_cardinal_precondition_rule(frame).set_index("direction")
    assert scored.loc["N", "control_failure"] == 1
    assert pd.isna(scored.loc["N", "native_control_failure"])
    assert pd.isna(scored.loc["N", "diagonal_precondition_failure"])


def test_rule_fails_closed_on_incomplete_action_set() -> None:
    frame = fixture()[lambda d: d.direction != "R"]
    with pytest.raises(ValueError, match="incomplete/duplicate action set"):
        apply_cardinal_precondition_rule(frame)
