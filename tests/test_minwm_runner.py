from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "third_party/minWM/minwm_runner.py"
VENDOR_ROOT = Path(os.environ.get("MINWM_VENDOR_ROOT", ROOT / "third_party/minWM"))
VENDOR = VENDOR_ROOT / "Wan21/wan_utils/camera_trajectory.py"


def _runner_make_viewmats():
    tree = ast.parse(RUNNER.read_text())
    selected = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "make_viewmats"
    ]
    namespace = {
        "np": np,
        "STEP_T": 0.08,
        "STEP_R": np.radians(3.0),
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(RUNNER), "exec"), namespace)
    return namespace["make_viewmats"]


def _vendor_generate():
    tree = ast.parse(VENDOR.read_text())
    names = {"_rot_x", "_rot_y", "_generate_c2w_trajectory"}
    selected = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {"np": np}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(VENDOR), "exec"), namespace)
    return namespace["_generate_c2w_trajectory"]


def _expected(direction: str, n_total: int, n_seed: int) -> np.ndarray:
    yaw = 0.0
    if direction in ("L", "FL", "BL"):
        yaw = -np.radians(3.0)
    elif direction in ("R", "FR", "BR"):
        yaw = np.radians(3.0)
    forward = 0.0
    if direction in ("F", "FL", "FR"):
        forward = 0.08
    elif direction in ("B", "BL", "BR"):
        forward = -0.08
    move = {}
    if yaw:
        move["yaw"] = yaw
    if forward:
        move["forward"] = forward
    poses = _vendor_generate()([move.copy() for _ in range(n_total - n_seed)])
    viewmats = np.stack([np.linalg.inv(pose) for pose in poses]).astype(np.float32)
    static_prefix = np.repeat(np.eye(4, dtype=np.float32)[None], n_seed - 1, axis=0)
    return np.concatenate([static_prefix, viewmats])


@pytest.mark.parametrize("direction", ("F", "FR", "R", "BR", "B", "BL", "L", "FL", "N"))
@pytest.mark.parametrize("n_seed", (4, 8))
def test_viewmats_match_released_minwm_pose_updates(direction: str, n_seed: int) -> None:
    n_total = n_seed + 12
    actual = _runner_make_viewmats()(direction, n_total, n_seed)
    expected = _expected(direction, n_total, n_seed)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-7)
    np.testing.assert_array_equal(
        actual[:n_seed], np.repeat(np.eye(4, dtype=np.float32)[None], n_seed, axis=0)
    )


def test_left_right_are_adapted_at_input_not_relabelled() -> None:
    make_viewmats = _runner_make_viewmats()
    left = np.linalg.inv(make_viewmats("L", 6, 4))
    right = np.linalg.inv(make_viewmats("R", 6, 4))
    assert left[4, 0, 2] < 0.0
    assert right[4, 0, 2] > 0.0
    text = RUNNER.read_text()
    assert '"direction": d' in text
    assert '"yaw_adapter": "released_minwm_yaw_negated_to_common_convention"' in text
    assert '"camera_pose_update": "released_rotate_then_local_translate"' in text


def test_rejects_unknown_action() -> None:
    with pytest.raises(ValueError, match="unsupported common action"):
        _runner_make_viewmats()("UNKNOWN", 12, 4)
