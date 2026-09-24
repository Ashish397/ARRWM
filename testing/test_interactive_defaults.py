import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interactive.engine_api import (  # noqa: E402
    Action,
    EngineConfig,
    NUM_FRAME_PER_BLOCK,
    PHYSICAL_NULL_STEER,
    PHYSICAL_NULL_THROTTLE,
    SEED_PREFILL_CHUNKS,
)
from interactive.play import SettingsPanel  # noqa: E402


def test_interactive_defaults_are_sequential_four_step_and_seven_seed_chunks():
    cfg = EngineConfig()
    assert cfg.block_chunks == 1
    assert cfg.denoising_steps == 4
    assert cfg.seed_prefill_chunks == 7
    assert SEED_PREFILL_CHUNKS == 7
    assert SEED_PREFILL_CHUNKS * NUM_FRAME_PER_BLOCK == 21


def test_engine_action_default_is_the_paper_physical_null():
    action = Action()
    assert action.throttle == PHYSICAL_NULL_THROTTLE == -0.023394260555505753
    assert action.steer == PHYSICAL_NULL_STEER == -0.0013313499512150884


def test_settings_expose_no_rolling_mode():
    panel = SettingsPanel(EngineConfig())
    assert "block_mode" not in panel.items
    assert not panel.select("block_mode")
    assert panel.select("seed_length_s")
