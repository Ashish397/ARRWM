import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interactive.engine_api import (  # noqa: E402
    EngineConfig,
    PIXEL_FRAMES_PER_CHUNK,
)
from interactive.mock_engine import MockEngine  # noqa: E402
from interactive.play import (  # noqa: E402
    DEFAULT_NOOP_STEER,
    DEFAULT_NOOP_THROTTLE,
    EngineWorker,
    FrameRing,
    PlayerApp,
    SettingsPanel,
    seed_chunks_from_seconds,
    seed_seconds_from_chunks,
)


def test_hud_reports_generated_content_seconds_not_raw_frame_counters():
    cfg = EngineConfig()
    app = PlayerApp(cfg, lambda c: MockEngine(c), headless=True)
    app.generated_frames_shown = 32
    app.frames_shown = 40
    app.display_ticks = 80
    hud = "\n".join(app.hud_lines())
    assert "generated 2.0s" in hud
    assert "frames 40/80" not in hud
    assert " fr)" not in hud


def test_seed_length_is_adjustable_in_tab_menu_in_seconds():
    panel = SettingsPanel(EngineConfig())
    assert panel.select("seed_length_s")
    field, seconds = panel.adjust(-1)
    assert field == "seed_length_s"
    assert seconds == seed_seconds_from_chunks(6) == 4.3125
    assert panel.cfg.seed_prefill_chunks == 6
    assert panel.pending_seed_reset
    assert any("4.3125s (6 chunks)" in line for line in panel.lines())
    assert panel.lines()[-2] == "  [ENTER] apply -> reset ride"


def test_app_applies_seed_length_with_reset_not_model_rebuild():
    cfg = EngineConfig()
    app = PlayerApp(cfg, lambda c: MockEngine(c), headless=True)
    app.panel.select("seed_length_s")
    app.panel.adjust(-1)
    live = []
    resets = []
    rebuilds = []
    app.worker.request_live_settings = lambda **kw: live.append(kw)
    app.worker.request_reset = lambda: resets.append(True)
    app.worker.request_rebuild = lambda new_cfg: rebuilds.append(new_cfg)

    app.apply_rebuild()

    assert app.cfg.seed_prefill_chunks == 6
    assert live == [{"seed_prefill_chunks": 6}]
    assert resets == [True]
    assert not rebuilds


def test_seed_seconds_round_and_clamp_to_supported_context():
    assert seed_chunks_from_seconds(0.5625) == 1
    assert seed_chunks_from_seconds(2.0625) == 3
    assert seed_chunks_from_seconds(5.0625) == 7
    assert seed_chunks_from_seconds(99) == 7


def test_mock_engine_obeys_selected_seed_length():
    cfg = EngineConfig(seed_prefill_chunks=2)
    seed = MockEngine(cfg).reset()
    assert len(seed) == 2 * PIXEL_FRAMES_PER_CHUNK


def test_latched_startup_action_preempts_stale_seed_playback():
    cfg = EngineConfig()
    ring = FrameRing()
    worker = EngineWorker(cfg, lambda c: MockEngine(c), ring)
    seed = np.zeros((7 * PIXEL_FRAMES_PER_CHUNK, 2, 2, 3), dtype=np.uint8)

    worker.set_action(0.5, 0.0)
    worker._push_seed(seed)
    assert len(ring) == 1
    _, generated = ring.pop_tagged()
    assert not generated

    worker.set_action(DEFAULT_NOOP_THROTTLE, DEFAULT_NOOP_STEER)
    worker._push_seed(seed)
    assert len(ring) == len(seed)


def test_numeric_zero_is_not_mistaken_for_the_physical_noop():
    cfg = EngineConfig()
    ring = FrameRing()
    worker = EngineWorker(cfg, lambda c: MockEngine(c), ring)
    seed = np.zeros((7 * PIXEL_FRAMES_PER_CHUNK, 2, 2, 3), dtype=np.uint8)

    worker.set_action(0.0, 0.0)
    worker._push_seed(seed)
    assert len(ring) == 1
