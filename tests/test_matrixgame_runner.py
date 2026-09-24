"""Contract checks for the Matrix-Game 2.0 panel32 adapter."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "code_release/baselines/matrixgame_runner.py"
VENDOR = Path(os.environ.get(
    "MATRIXGAME_VENDOR_ROOT",
    ROOT / "third_party/Matrix-Game/Matrix-Game-2",
)).resolve()


def _runner_make_actions():
    """Load only the small pure tensor helper, without importing GPU code."""
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "make_actions"
    )
    namespace = {"torch": torch, "KDIM": 4, "CAM": 0.1}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(RUNNER), "exec"), namespace)
    return namespace["make_actions"]


def test_released_universal_action_contract() -> None:
    config = json.loads(
        (VENDOR / "configs/distilled_model/universal/config.json").read_text()
    )
    action = config["action_config"]
    assert action["keyboard_dim_in"] == 4
    assert action["mouse_dim_in"] == 2
    assert action["enable_keyboard"] is True
    assert action["enable_mouse"] is True
    assert config["local_attn_size"] == 6

    make_actions = _runner_make_actions()
    expected = {
        "F": ([1, 0, 0, 0], [0, 0]),
        "FR": ([1, 0, 0, 0], [0, 0.1]),
        "R": ([0, 0, 0, 0], [0, 0.1]),
        "BR": ([0, 1, 0, 0], [0, 0.1]),
        "B": ([0, 1, 0, 0], [0, 0]),
        "BL": ([0, 1, 0, 0], [0, -0.1]),
        "L": ([0, 0, 0, 0], [0, -0.1]),
        "FL": ([1, 0, 0, 0], [0, -0.1]),
        "NOOP": ([0, 0, 0, 0], [0, 0]),
    }
    for command, (keyboard, mouse) in expected.items():
        actual_keyboard, actual_mouse = make_actions(command, 17)
        torch.testing.assert_close(
            actual_keyboard, torch.tensor(keyboard, dtype=torch.float32).repeat(17, 1)
        )
        torch.testing.assert_close(
            actual_mouse, torch.tensor(mouse, dtype=torch.float32).repeat(17, 1)
        )


def test_held_action_equals_repeated_streaming_blocks() -> None:
    """Mirror the exact slice arithmetic in vendor ``cond_current``.

    The interactive entrypoint replaces nine pixel frames for its first
    three-latent-frame block and twelve thereafter.  Repeating one command at
    every prompt must produce the same tensor supplied by the batch adapter.
    """
    latent_frames = 189
    latent_frames_per_block = 3
    pixel_frames = 1 + 4 * (latent_frames - 1)
    make_actions = _runner_make_actions()

    for command in ("F", "FR", "R", "BR", "B", "BL", "L", "FL", "NOOP"):
        batch_keyboard, batch_mouse = make_actions(command, pixel_frames)
        streaming_keyboard = torch.zeros_like(batch_keyboard)
        streaming_mouse = torch.zeros_like(batch_mouse)
        current_keyboard = batch_keyboard[0]
        current_mouse = batch_mouse[0]
        for current_start in range(0, latent_frames, latent_frames_per_block):
            last_frame_count = (
                1 + 4 * (latent_frames_per_block - 1)
                if current_start == 0 else 4 * latent_frames_per_block
            )
            final_frame = 1 + 4 * (
                current_start + latent_frames_per_block - 1
            )
            start = final_frame - last_frame_count
            streaming_keyboard[start:final_frame] = current_keyboard
            streaming_mouse[start:final_frame] = current_mouse

        torch.testing.assert_close(streaming_keyboard, batch_keyboard)
        torch.testing.assert_close(streaming_mouse, batch_mouse)


def test_runner_uses_vendor_pipeline_and_native_geometry() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "from pipeline import CausalInferencePipeline" in source
    assert "pipeline.inference(" in source
    assert "v2.Resize(size=(352, 640), antialias=True)" in source
    assert '"tile_size": [44, 80]' in source
    assert '"tile_stride": [23, 38]' in source
    assert "[1, 16, NUM_LAT, 44, 80]" in source
    assert "fps=25" in source
