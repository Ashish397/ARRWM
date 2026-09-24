#!/usr/bin/env python
"""Interactive pygame player for the ARRWM world model (WS-C).

Runs the engine on a worker thread that pushes decoded frames into a ring
buffer; the main loop presents from that buffer.

Presentation model (two rates, deliberately separate)
-----------------------------------------------------
* **Display rate** — the window ALWAYS ticks and blits at ``cfg.playback_fps``
  (16 Hz by default), no matter how slow the engine is. The refresh rate is a
  fixed promise, not something the pacing servo is allowed to lower.
* **Content rate** — how fast we walk forward through the frame ring. In
  ``--playback auto`` this is servoed to the measured sustained generation
  throughput, so we never outrun the generator.

A fractional accumulator bridges them: each display tick advances the ring by
``content_fps / display_fps`` frames, so when generation is slower the current
frame is simply re-blitted (a held frame). 1 step (>=16 fps generation) is
identical to a plain 16 fps player; 4 steps is honest slow motion at a
rock-steady 16 Hz; 12 steps is ultra slow motion, still at 16 Hz.

``--playback <fps>`` is the opt-out: content and display are both pinned to
that number (the pre-adaptive behaviour), so ``--playback 8`` really does give
an 8 Hz window.

Usage
-----
    python interactive/play.py --mock
    python interactive/play.py --ckpt <path.pt> --seed_zarr ~/20240224003808.zarr

    # headless smoke test (no display, scripted actions, exits 0 on success)
    SDL_VIDEODRIVER=dummy python interactive/play.py --mock --smoke

Running over VS Code Remote-SSH (no local display)
--------------------------------------------------
pygame cannot open a window on the remote host, so use web mode:

    python interactive/play.py --web --ckpt <ckpt> [--port 8765]

It prints ``Open http://localhost:8765 in your browser``. VS Code forwards the
port automatically; if it does not, open the PORTS tab and forward it by hand.
The browser page is the same player -- same controls, HUD, presets, settings,
seed picker (P) and recording (V) -- driven by the same ControlScheme and the
same 16 Hz presentation model. See interactive/web_play.py.

Seeding from your own video
---------------------------
Drop any mp4/mov/mkv/webm/avi/m4v into ``interactive/user_videos/`` (created
for you), launch, and pick the ``[VID]`` entry in the seed picker -- or point
straight at a file with ``--seed_video <path>``.

Your clip is put into TRAINING SPACE by reproducing the zarr-encoder's
transform (audited: utils/pre_encode_local.py VideoLoader.stream_blocks
line 139, ``-vf scale=832:480,showinfo``, no crop, no fps filter):

    20 fps  ->  maximal crop to the camera's 16:9 aspect (bottom band)
            ->  [optional barrel warp, off by default]
            ->  anamorphic squash to 832x480  ->  /255*2-1  ->  Wan encode

20 fps because the rides were encoded at their NATIVE rate and every ride
zarr says ``fps: 20.0`` -- 16 fps is only our render rate. The squash is a
whole-frame anamorphic resize with NO aspect preservation, exactly as
training did; the crop stage exists only to put a phone frame into the
camera's aspect first, taking the LARGEST region at that aspect (full height
for a wider-than-16:9 source, full width for a taller one) -- a sliver off
landscape, a heavy band off portrait. When height is trimmed the BOTTOM band
is kept by default (``--seed_video_crop {bottom,center,top}``): the training
camera is low-mounted at ground level, so the road ahead matches it far
better than the horizon or sky. Width trims stay centred. The barrel warp (``--seed_video_fisheye``) mimics the
robot's wide-angle lens for rectilinear phone footage; it is OFF by default
because it costs field of view, and is opt-in at 0.15-0.22.

The first 81 frames become the 7-chunk (21-latent) seed context that fills
the whole trained attention span, and you drive from there. Shorter clips
seed at reduced depth and say so. See ``interactive/user_videos/README.md``.

Controls
--------
    W / Up          throttle +0.5             S / Down   throttle -0.3
    A / Left        steer left (-1.0)         D / Right  steer right (+1.0)
    SPACE           physical no-op (-0.02339426, -0.00133135), absolute
    (digital+latch by default: a press latches instantly and the action
     persists after release. Two presses within COMBO_MERGE_WINDOW_S on
     different axes MERGE into the diagonal, so tapping W then A gives
         the calibrated diagonal without having to hold both. --controls analog restores the
     original ramp/decay integrator, which does not merge.)
    R               reset ride (re-prefill from seed)
    V               toggle mp4 recording of what is on screen
    TAB             settings panel (Up/Down select, Left/Right adjust,
                    Enter applies a pending rebuild, TAB/ESC closes)
    Q / ESC         quit

Only `interactive/play.py` and `interactive/mock_engine.py` are owned by this
workstream; the engine is consumed strictly through `interactive.engine_api`.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from collections import deque
from dataclasses import replace
from typing import Callable, Deque, List, Optional, Tuple

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from interactive.engine_api import (  # noqa: E402
    PIXEL_FRAMES_PER_CHUNK,
    PIXEL_H,
    PIXEL_W,
    PLAYBACK_FPS,
    PHYSICAL_NULL_STEER,
    PHYSICAL_NULL_THROTTLE,
    SEED_FIRST_PIXEL_FRAMES,
    SEED_PREFILL_CHUNKS,
    Action,
    EngineBase,
    EngineConfig,
    HorizonStats,
)

PRECISION_CHOICES = ["fp32", "bf16", "fp8_wo", "fp8_dyn", "fp4_wo"]
COMPILE_CHOICES = ["off", "max-autotune-no-cudagraphs", "reduce-overhead",
                   "max-autotune"]
LADDER_CHOICES = ["interp", "trained", "grid"]
#: Fully-joint research block widths. Production stays sequential (1).
BLOCK_CHOICES = [1, 2, 4]

# ---------------------------------------------------------------------------
# Quality presets
# ---------------------------------------------------------------------------
#: Named speed/quality points, measured on an RTX 5090 at bf16 / compile=off /
#: auto playback (see interactive/BENCH_REPORT.md). Presets vary ONLY these two
#: fields; precision, compile and pacing are deliberately left alone so a
#: preset switch never clobbers something the user set on purpose.
PRESETS = {
    "speed":   {"denoising_steps": 1, "decoder": "taew2_1"},
    "balance": {"denoising_steps": 2, "decoder": "taew2_1"},
    "quality": {"denoising_steps": 4, "decoder": "taew2_1"},
}
PRESET_ORDER = ["speed", "balance", "quality"]
PRESET_BLURB = {
    "speed":   "1 step  ~16 fps  real-time",
    "balance": "2 steps ~11 fps  smooth slow-mo",
    "quality": "4 steps  ~7 fps  best fidelity",
}
#: Shown when the settings do not match any preset.
PRESET_CUSTOM = "custom"


def active_preset(cfg: EngineConfig) -> str:
    """Which preset the current settings correspond to, or 'custom'."""
    for name in PRESET_ORDER:
        p = PRESETS[name]
        if (int(cfg.denoising_steps) == p["denoising_steps"]
                and str(getattr(cfg, "decoder", "")) == p["decoder"]):
            return name
    return PRESET_CUSTOM

RAMP_RATE = 2.0     # units/second toward the held direction
DECAY_RATE = 3.0    # units/second back toward 0 on release


def seed_seconds_from_chunks(chunks: int) -> float:
    """Visible seed duration under the streaming decoder contract."""
    n = int(np.clip(chunks, 1, SEED_PREFILL_CHUNKS))
    frames = SEED_FIRST_PIXEL_FRAMES + (n - 1) * PIXEL_FRAMES_PER_CHUNK
    return frames / PLAYBACK_FPS


def seed_chunks_from_seconds(seconds: float) -> int:
    """Resolve seconds to the nearest selectable visible seed duration."""
    value = float(seconds)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("seed duration must be a positive number of seconds")
    choices = range(1, SEED_PREFILL_CHUNKS + 1)
    return min(choices, key=lambda n: abs(seed_seconds_from_chunks(n) - value))


# ==========================================================================
# Engine construction
# ==========================================================================
def make_engine_factory(use_mock: bool) -> Callable[[EngineConfig], EngineBase]:
    """Return a callable that builds an engine from an EngineConfig."""
    if use_mock:
        from interactive.mock_engine import MockEngine
        return lambda cfg: MockEngine(cfg)

    try:
        from interactive import engine as engine_mod  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on WS-A landing
        raise SystemExit(
            "interactive/engine.py could not be imported "
            f"({type(exc).__name__}: {exc}).\n"
            "The real engine (WS-A) may not have landed yet. "
            "Run with --mock to drive the procedural MockEngine instead:\n"
            "    python interactive/play.py --mock"
        )

    ctor = None
    for name in ("WorldModelEngine", "Engine", "InteractiveEngine",
                 "build_engine", "create_engine", "make_engine"):
        cand = getattr(engine_mod, name, None)
        if cand is not None:
            ctor = cand
            break
    if ctor is None:
        raise SystemExit(
            "interactive/engine.py imported but exposes no known entry point "
            "(looked for WorldModelEngine / Engine / build_engine / "
            "create_engine). Use --mock, or ask WS-A for the constructor name."
        )
    return lambda cfg: ctor(cfg)


# ==========================================================================
# Frame ring buffer
# ==========================================================================
class FrameRing:
    """Thread-safe FIFO of individual RGB frames."""

    #: Seconds of history used to estimate sustained generation throughput.
    #: A sliding window, NOT a per-push EMA: the engine yields the previous
    #: chunk each iteration and then flushes the last one immediately, so
    #: every horizon ends with two pushes microseconds apart. An EMA gets
    #: yanked upward by that spike (measuring ~70 fps for a ~7 fps engine);
    #: total-frames / total-time over a window is immune to it.
    THROUGHPUT_WINDOW_S = 12.0

    def __init__(self):
        self._dq: Deque[np.ndarray] = deque()
        self._generated: Deque[bool] = deque()
        self._lock = threading.Lock()
        self.total_pushed = 0
        self.total_popped = 0
        # Sustained production rate in pixel-frames per wall second, measured
        # at the ring so it includes decode and device->host transfer, not
        # just the denoise. None until two blocks have landed.
        self._pushes: Deque[Tuple[float, int]] = deque()   # (arrival, n_frames)
        self._last_push_t: Optional[float] = None
        self.discontinuities = 0

    def push_block(self, frames: np.ndarray, *, generated: bool = True) -> None:
        now = time.perf_counter()
        with self._lock:
            n = 0
            for f in frames:
                self._dq.append(f)
                self._generated.append(bool(generated))
                self.total_pushed += 1
                n += 1
            # Seed frames are real context, not generated throughput. Keeping
            # them out also prevents a shorter Tab-selected seed from moving
            # the generation-rate estimate.
            if n and generated:
                self._pushes.append((now, n))
                cutoff = now - self.THROUGHPUT_WINDOW_S
                while len(self._pushes) > 2 and self._pushes[0][0] < cutoff:
                    self._pushes.popleft()
                self._last_push_t = now

    @property
    def throughput_fps(self) -> Optional[float]:
        """Sustained pixel-frames per wall second, or None until measurable.

        Frames from the FIRST push in the window are excluded: they arrived
        before the window opened, so counting them against the window's span
        would overstate the rate.
        """
        with self._lock:
            if len(self._pushes) < 2:
                return None
            t0 = self._pushes[0][0]
            t1 = self._pushes[-1][0]
            span = t1 - t0
            if span <= 1e-6:
                return None
            frames = sum(n for _, n in list(self._pushes)[1:])
            return frames / span

    def note_gap(self) -> None:
        """Drop the measurement window across a deliberate pause.

        Otherwise the idle span is averaged in as though nothing were being
        produced, and the estimated throughput collapses.
        """
        with self._lock:
            self._last_push_t = None
            self._pushes.clear()

    def pop(self) -> Optional[np.ndarray]:
        frame, _ = self.pop_tagged()
        return frame

    def pop_tagged(self) -> Tuple[Optional[np.ndarray], bool]:
        """Pop one frame and whether it is generated (rather than seed)."""
        with self._lock:
            if not self._dq:
                return None, False
            self.total_popped += 1
            return self._dq.popleft(), self._generated.popleft()

    def clear(self) -> None:
        with self._lock:
            self._dq.clear()
            self._generated.clear()
            self._last_push_t = None
            self._pushes.clear()
            # A reset/reseed empties the ring on purpose. Consumers use this
            # to tell an intentional discontinuity from a starvation stall.
            self.discontinuities += 1

    def __len__(self) -> int:
        with self._lock:
            return len(self._dq)


# ==========================================================================
# Engine worker thread
# ==========================================================================
class EngineWorker(threading.Thread):
    """Owns the engine; generates horizons under the latest latched action."""

    def __init__(self, cfg: EngineConfig,
                 factory: Callable[[EngineConfig], EngineBase],
                 ring: FrameRing):
        super().__init__(name="engine-worker", daemon=True)
        self.cfg = replace(cfg)
        self.factory = factory
        self.ring = ring

        self._lock = threading.Lock()
        self._action = Action()
        self._stop_evt = threading.Event()
        self._reset_req = threading.Event()
        self._rebuild_cfg: Optional[EngineConfig] = None
        self._live_req: Optional[dict] = None
        self._seed_req: Optional[str] = None
        # Hard pause held by the UI while a modal (the seed picker) is up, so
        # its thumbnail decoder never shares the GPU with a live horizon.
        self._ui_pause = threading.Event()
        self._ui_idle = threading.Event()

        self.engine: Optional[EngineBase] = None
        self.state = "starting"
        self.warmup_msg = ""
        self.warmup_seconds = 0.0
        self.error: Optional[str] = None
        self.last_stats: Optional[HorizonStats] = None
        self.horizons_done = 0

    # -- control surface (called from the UI thread) ----------------------
    def set_action(self, throttle: float, steer: float) -> None:
        with self._lock:
            self._action = Action(float(throttle), float(steer))

    def latched_action(self) -> Action:
        with self._lock:
            return Action(self._action.throttle, self._action.steer)

    def request_reset(self) -> None:
        self._reset_req.set()

    def request_seed(self, zarr_path: str) -> None:
        """Restart the ride from a new seed zarr (no engine rebuild)."""
        with self._lock:
            self._seed_req = str(zarr_path)

    def pause_for_ui(self, timeout: float = 30.0) -> bool:
        """Block the worker off the GPU and wait until it is actually idle.

        Returns True if the worker parked in time.  The picker decodes
        thumbnails on the default stream, so it must not overlap a horizon.
        """
        self._ui_pause.set()
        return self._ui_idle.wait(timeout)

    def resume_from_ui(self) -> None:
        # The modal held the worker off the GPU; without this the idle span
        # reads as one huge inter-chunk gap and craters the throughput EMA.
        self.ring.note_gap()
        self._ui_pause.clear()
        self._ui_idle.clear()

    @property
    def ride_id(self) -> str:
        eng = self.engine
        rid = getattr(eng, "ride_id", None) if eng is not None else None
        if rid:
            return str(rid)
        seed = getattr(self.cfg, "seed_zarr", "") or ""
        return os.path.splitext(os.path.basename(seed.rstrip("/")))[0] or "(noise)"

    def request_rebuild(self, cfg: EngineConfig) -> None:
        with self._lock:
            self._rebuild_cfg = replace(cfg)

    def request_live_settings(self, **kw) -> None:
        # Coalesce, don't clobber: several panel edits can land between two
        # worker iterations and all of them must reach the engine.
        with self._lock:
            merged = dict(self._live_req or {})
            merged.update(kw)
            self._live_req = merged

    def request_stop(self) -> None:
        self._stop_evt.set()

    # -- internals --------------------------------------------------------
    def _max_frames(self) -> int:
        return max(1, int(self.cfg.max_buffered_horizons)
                   * max(1, int(self.cfg.horizon_chunks))
                   * PIXEL_FRAMES_PER_CHUNK)

    def _push_seed(self, seed) -> None:
        """Publish real seed frames without hiding an already-latched action.

        A control can arrive while reset() is still decoding the seed.  In
        that race the old code published the whole seed *after* the UI had
        cleared its queue, so the action looked ignored for several seconds.
        Keep the seed in the model's KV context, but show only its final frame
        when a non-neutral action is already waiting; generated frames under
        that action are then the next content the player can advance to.
        """
        if seed is None or not len(seed):
            return
        frames = np.asarray(seed)
        action = self.latched_action()
        is_noop = (np.isclose(action.throttle, PHYSICAL_NULL_THROTTLE,
                              rtol=0.0, atol=1e-9)
                   and np.isclose(action.steer, PHYSICAL_NULL_STEER,
                                  rtol=0.0, atol=1e-9))
        if (bool(self.cfg.extra.get("drop_stale_on_action", True))
                and not is_noop):
            frames = frames[-1:]
        self.ring.push_block(frames, generated=False)

    def _interrupted(self) -> bool:
        if (self._stop_evt.is_set() or self._reset_req.is_set()
                or self._ui_pause.is_set()):
            return True
        with self._lock:
            return self._rebuild_cfg is not None or self._seed_req is not None

    def _teardown_engine(self, *, reset_state: bool = True) -> None:
        """Release the current engine COMPLETELY before another is built.

        ``reset_state`` drops the stats and buffered frames belonging to the
        engine being discarded -- correct for a REBUILD, where carrying them
        forward would report the dead engine's numbers as the new one's. On
        shutdown it is False: the engine still has to be freed, but the run's
        final stats are a record the caller (the smoke test, the HUD) reads
        after the loop exits.

        A rebuild used to construct the second engine while the first was
        still resident: two full models on the card at once, which OOMs
        during the new engine's reset(). Closing is not enough -- every
        Python reference has to go and the CUDA caching allocator has to be
        told to release the freed blocks, or the memory stays reserved.
        """
        had_engine = self.engine is not None
        if had_engine:
            try:
                self.engine.close()
            except Exception as exc:
                print(f"[play] engine.close() failed during rebuild: {exc}")
        self.engine = None
        if reset_state:
            # Stats can hold references into engine-owned tensors, and the
            # ring holds frames decoded by the engine being discarded.
            self.last_stats = None
            self.ring.clear()

        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                # synchronize first: blocks still referenced by in-flight work
                # cannot be returned to the allocator.
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                free, total = torch.cuda.mem_get_info()
                if had_engine:
                    print(f"[play] rebuild: freed old engine, "
                          f"{free / 2**30:.1f} GiB free of {total / 2**30:.1f} GiB")
        except Exception as exc:
            print(f"[play] could not report free memory: {exc}")

    def _drain_live_settings(self) -> None:
        """Hand any queued live settings to the engine, right now.

        Called from several points in the run loop (top of loop, every chunk
        boundary, and inside the backpressure waits) because the worker often
        spends nearly all of its wall time blocked inside one horizon.
        """
        if self.engine is None:
            return
        with self._lock:
            live = self._live_req
            self._live_req = None
        if not live:
            return
        # Only pass settings this engine actually accepts. The panel gains
        # options over time and older engines (the mock, anything
        # third-party) would otherwise die on an unexpected keyword.
        try:
            import inspect
            sig = inspect.signature(self.engine.apply_live_settings)
            accepted = set(sig.parameters)
            if not any(p.kind is inspect.Parameter.VAR_KEYWORD
                       for p in sig.parameters.values()):
                dropped = [k for k in live if k not in accepted]
                if dropped:
                    log_once = getattr(self, "_warned_live", set())
                    for k in dropped:
                        if k not in log_once:
                            print(f"[play] engine ignores live setting {k!r}")
                            log_once.add(k)
                    self._warned_live = log_once
                live = {k: v for k, v in live.items() if k in accepted}
        except (TypeError, ValueError):
            pass
        if live:
            self.engine.apply_live_settings(**live)
        for k, v in live.items():
            if v is not None and hasattr(self.cfg, k):
                setattr(self.cfg, k, v)

    def _build(self) -> None:
        self.state = "rebuilding"
        self._teardown_engine()
        self.engine = self.factory(self.cfg)
        # Let the engine re-read the action mailbox at every chunk boundary
        # instead of once per horizon. Engines that predate this simply
        # don't expose the hook.
        setter = getattr(self.engine, "set_action_provider", None)
        if setter is not None:
            setter(self.latched_action)

        # Compile warmup BEFORE the engine goes live. Otherwise inductor
        # compiles on first hit of each shape during real play, freezing the
        # first horizons for minutes with no frames -- which reads as "it
        # isn't generating while rendering".
        warm = getattr(self.engine, "warmup", None)
        if warm is not None and str(self.cfg.compile_mode or "off") != "off":
            self.state = "warmup"

            def _progress(phase: str, done: int, total: int) -> None:
                if phase == "seed":
                    self.warmup_msg = "compiling kernels... (one-time, may take minutes)"
                elif phase == "done":
                    self.warmup_msg = ""
                else:
                    self.warmup_msg = (f"compiling kernels... chunk {done}/{total} "
                                       f"(one-time, may take minutes)")

            try:
                secs = warm(progress=_progress)
                self.warmup_seconds = float(secs)
                print(f"[play] compile warmup finished in {secs:.1f}s")
            except Exception as exc:
                import traceback
                traceback.print_exc()
                print(f"[play] compile warmup failed ({exc}); continuing")
            finally:
                self.warmup_msg = ""

        self.ring.clear()
        seed = self.engine.reset()
        self._push_seed(seed)

    def run(self) -> None:  # noqa: C901 - a control loop, kept flat on purpose
        try:
            self._build()
            while not self._stop_evt.is_set():
                # --- UI modal holds the GPU (seed picker) ----------------
                if self._ui_pause.is_set():
                    self.state = "picker"
                    self._ui_idle.set()
                    time.sleep(0.01)
                    continue
                self._ui_idle.clear()

                # --- pending seed swap (new ride, no rebuild) ------------
                with self._lock:
                    seed_path = self._seed_req
                    self._seed_req = None
                if seed_path is not None:
                    self.state = "reseeding"
                    self.ring.clear()
                    self.cfg.seed_zarr = seed_path
                    reseed = getattr(self.engine, "reset_with_seed", None)
                    if reseed is not None:
                        seed = reseed(seed_path)
                    else:   # mock engines: same effect via cfg + reset()
                        self.engine.cfg.seed_zarr = seed_path
                        seed = self.engine.reset()
                    self._push_seed(seed)
                    self.horizons_done = 0
                    continue

                # --- pending rebuild (precision / compile changed) --------
                with self._lock:
                    new_cfg = self._rebuild_cfg
                    self._rebuild_cfg = None
                if new_cfg is not None:
                    self.cfg = new_cfg
                    self._build()
                    self.horizons_done = 0
                    continue

                # --- pending reset ---------------------------------------
                if self._reset_req.is_set():
                    self._reset_req.clear()
                    self.state = "resetting"
                    self.ring.clear()
                    # Seed-length edits are queued as live settings and then
                    # paired with this reset. Apply them before loading seed
                    # latents, even though reset handling precedes the normal
                    # live-settings drain in this loop.
                    self._drain_live_settings()
                    seed = self.engine.reset()
                    self._push_seed(seed)
                    self.horizons_done = 0
                    continue

                # --- pending live settings -------------------------------
                self._drain_live_settings()

                # --- backpressure ----------------------------------------
                if len(self.ring) >= self._max_frames():
                    self.state = "paused"
                    time.sleep(0.005)
                    self._drain_live_settings()
                    continue

                # --- one latched horizon ---------------------------------
                action = self.latched_action()
                self.state = "generating"
                for chunk in self.engine.generate_horizon(action):
                    self.ring.push_block(np.asarray(chunk), generated=True)
                    # Live settings must be drained at every chunk boundary,
                    # not just at the top of this loop. Under backpressure the
                    # worker can sit inside a single generate_horizon() for
                    # many seconds, and a preset hotkey / panel edit made in
                    # that window would otherwise never reach the engine at
                    # all. apply_live_settings() is defined to take effect
                    # from the NEXT horizon, so calling it mid-horizon is safe
                    # and simply makes the pickup deterministic.
                    self._drain_live_settings()
                    if self._interrupted():
                        break
                    while (len(self.ring) >= self._max_frames()
                           and not self._interrupted()):
                        self.state = "paused"
                        time.sleep(0.005)
                        self._drain_live_settings()
                    self.state = "generating"
                stats = self.engine.last_stats
                if stats is not None:
                    self.last_stats = stats
                self.horizons_done += 1
        except Exception as exc:  # pragma: no cover - surfaced in the HUD
            import traceback
            self.error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            self.state = "error"
        finally:
            state = self.state
            self._teardown_engine(reset_state=False)
            self.state = state if state == "error" else "stopped"


# ==========================================================================
# Recording (same pattern as utils/play_world_model.py FrameSink)
# ==========================================================================
class FrameSink:
    """Collects RGB frames and writes an mp4 on demand (ffmpeg via stdin pipe)."""

    def __init__(self, out_path: str, fps: float):
        self.out_path = out_path
        self.fps = fps
        self.frames: List[np.ndarray] = []

    def add(self, frame: np.ndarray) -> None:
        self.frames.append(np.ascontiguousarray(frame))

    def write(self) -> Optional[str]:
        if not self.frames:
            return None
        import subprocess
        arr = np.stack(self.frames, 0)
        h, w = arr.shape[1], arr.shape[2]
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", str(self.fps), "-i", "pipe:0",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", self.out_path,
        ]
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            _, err = proc.communicate(input=arr.tobytes(), timeout=300)
            if proc.returncode != 0:
                print(f"[rec] ffmpeg failed: {err.decode(errors='replace')[:400]}")
                return None
            print(f"[rec] wrote {len(self.frames)} frames -> {self.out_path}")
            return self.out_path
        except FileNotFoundError:
            try:
                import imageio
                alt = self.out_path.replace(".mp4", ".gif")
                imageio.mimsave(alt, self.frames, fps=self.fps)
                print(f"[rec] ffmpeg missing; wrote GIF -> {alt}")
                return alt
            except Exception as exc:
                print(f"[rec] no ffmpeg and imageio fallback failed: {exc}")
                return None


# ==========================================================================
# Control scheme
# ==========================================================================
#: Logical control names -> their axis. Endpoint values are calibrated below.
CONTROL_AXES = {
    "w": "throttle",
    "s": "throttle",
    "a": "steer",
    "d": "steer",
}
# The 50-command response surface found conservative, separable endpoints for
# all cardinal and diagonal controls. Throttle -1 folds into forward; ±1 steer
# works but introduces more unintended longitudinal motion than ±0.5.
DEFAULT_FORWARD_THROTTLE = 0.5
DEFAULT_REVERSE_THROTTLE = -0.3
DEFAULT_LEFT_STEER = -1.0
DEFAULT_RIGHT_STEER = 1.0
# Physical stillness in the paper's normalized PCA coordinates.  Numeric zero
# is a nearby command, but it is not the inverse-PCA no-op.
DEFAULT_NOOP_THROTTLE = PHYSICAL_NULL_THROTTLE
DEFAULT_NOOP_STEER = PHYSICAL_NULL_STEER
#: Stop / no-op. SPACE is unbound elsewhere in this file, so it is the pick.
STOP_CONTROL = "stop"
STOP_KEY_NAME = "space"
#: The other control on the same axis. Holding both is contradictory.
OPPOSITE_CONTROL = {"w": "s", "s": "w", "a": "d", "d": "a"}
#: Deterministic scan order when several controls go down in the same tick.
CONTROL_ORDER = ("w", "s", "d", "a")

#: Combo-merge window, seconds.
#:
#: Two-axis inputs are usually pressed *sequentially*, not simultaneously: the
#: user taps W and then A a moment later, often releasing W in between. Without
#: a merge window the A press is a brand-new input on its own and the two taps
#: read as two separate one-axis actions instead of the diagonal the user
#: meant. So the FIRST valid press latches immediately (no added latency) and
#: opens this window; a press on the OTHER axis inside it upgrades the latch to
#: the two-axis combo.
COMBO_MERGE_WINDOW_S = 0.5


class ControlScheme:
    """Two independent axes of behaviour: mode x latching.

    mode='digital'  a held key means its configured endpoint immediately.
    mode='analog'   the original ramp/decay integrator.

    latch=True      the action persists after every key is released; only a
                    new valid input or the stop key changes it.
    latch=False     releasing everything returns to physical no-op (instantly in
                    digital, via the existing decay in analog).

    Rules that hold in BOTH modes:
      * the stop key wins over anything else held with it, and latches the
        physical no-op;
      * a contradictory same-axis combo (W+S, or A+D) is INVALID and is
        ignored -- that axis simply keeps whatever it had, so the user sees
        "nothing happens" rather than a lurch.

    DIGITAL SEMANTICS, precisely (see COMBO_MERGE_WINDOW_S):

      1. Latest valid input wins PER AXIS; the other axis keeps its latched
         value. Pressing A does not zero the throttle -- it sets steer and
         leaves throttle where it was. (It used to zero it: a press rewrote
         both axes from the held set, so tapping W then A gave (0, -1) and the
         throttle the user had just asked for was silently thrown away.)
      2. The first valid press latches IMMEDIATELY and opens a
         COMBO_MERGE_WINDOW_S merge window.
      3. A press on the OTHER axis inside that window MERGES: both actions are
         (re-)asserted together, so tap-W then tap-A 0.3 s later latches
         the configured forward-left endpoints even though the keys never
         overlapped.
      4. Only the first TWO unique actions in a window count. Once two are
         taken the merge is final, and every further press is swallowed until
         the window elapses; the next press after that opens a fresh window.
      5. A second press on an axis already taken in the window is ignored --
         the first one keeps the axis (tap-W then tap-S inside the window
         keeps W). It does not consume a slot, so W, S, A still merges to the
         configured forward-left endpoints.
      6. Holding a control while its opposite goes down is invalid in the
         original sense and is ignored outright (neither counts, no slot is
         consumed), so a simultaneous W+S press changes nothing.
      7. SPACE is instant and absolute: it restores physical no-op on both
         axes and closes any open window, so nothing merges across a stop.

    Merging applies with latch on AND off; the window re-asserts the first
    action even when releasing the key had already restored physical no-op.
    """

    MODES = ("digital", "analog")

    def __init__(self, mode: str = "digital", latch: bool = True,
                 forward_throttle: float = DEFAULT_FORWARD_THROTTLE,
                 reverse_throttle: float = DEFAULT_REVERSE_THROTTLE,
                 left_steer: float = DEFAULT_LEFT_STEER,
                 right_steer: float = DEFAULT_RIGHT_STEER):
        self.mode = mode if mode in self.MODES else "digital"
        self.latch = bool(latch)
        self.forward_throttle = float(forward_throttle)
        self.reverse_throttle = float(reverse_throttle)
        self.left_steer = float(left_steer)
        self.right_steer = float(right_steer)
        if not (0.0 < self.forward_throttle <= 1.0):
            raise ValueError("forward_throttle must be in (0, 1]")
        if not (-1.0 <= self.reverse_throttle < 0.0):
            raise ValueError("reverse_throttle must be in [-1, 0)")
        if not (-1.0 <= self.left_steer < 0.0):
            raise ValueError("left_steer must be in [-1, 0)")
        if not (0.0 < self.right_steer <= 1.0):
            raise ValueError("right_steer must be in (0, 1]")
        self._endpoints = {
            "w": self.forward_throttle,
            "s": self.reverse_throttle,
            "a": self.left_steer,
            "d": self.right_steer,
        }
        self.noop_throttle = DEFAULT_NOOP_THROTTLE
        self.noop_steer = DEFAULT_NOOP_STEER
        self.throttle = self.noop_throttle
        self.steer = self.noop_steer
        self._prev_held: frozenset = frozenset()
        # Clock accumulated from the caller's dt rather than read from
        # perf_counter, so the merge window is exactly as testable as the rest
        # of the scheme (no sleeping in the smoke test).
        self._t = 0.0
        self._win_at: Optional[float] = None    # when the merge window opened
        self._win_taken: List[str] = []         # unique controls taken in it

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _axis_dir(held, pos: str, neg: str):
        """+1 / -1 / 0, or None when both are held (invalid, ignore)."""
        p, n = pos in held, neg in held
        if p and n:
            return None
        return (1.0 if p else (-1.0 if n else 0.0))

    def describe(self) -> str:
        letters = ""
        if self.throttle > self.noop_throttle:
            letters += "W"
        elif self.throttle < self.noop_throttle:
            letters += "S"
        if self.steer > self.noop_steer:
            letters += "D"
        elif self.steer < self.noop_steer:
            letters += "A"
        return letters or "--"

    def _close_window(self) -> None:
        self._win_at = None
        self._win_taken = []

    # -- update -----------------------------------------------------------
    def update(self, dt: float, held) -> None:
        held = frozenset(held)
        self._t += max(float(dt), 0.0)
        if self.mode == "digital":
            self._update_digital(held)
        else:
            self._update_analog(dt, held)
        self._prev_held = held

    def _update_digital(self, held: frozenset) -> None:
        # Digital only acts on CHANGES of the held set: holding a key steady
        # must not keep re-latching, and neither should a repeat event.
        if held == self._prev_held:
            return

        if STOP_CONTROL in held:
            self.throttle = self.noop_throttle
            self.steer = self.noop_steer
            self._close_window()        # nothing merges across a stop
            return

        # Expire a stale window lazily -- it only matters at the next press.
        if (self._win_at is not None
                and self._t - self._win_at > COMBO_MERGE_WINDOW_S):
            self._close_window()

        pressed = held - self._prev_held
        took = False
        for ctl in CONTROL_ORDER:
            if ctl not in pressed:
                continue
            if OPPOSITE_CONTROL[ctl] in held:
                continue                # contradictory same-axis: ignore both
            if self._win_at is None:
                self._win_at = self._t          # first press opens the window
            elif len(self._win_taken) >= 2:
                continue                # window full: swallow until it lapses
            if ctl in self._win_taken:
                continue                # duplicate
            axis = CONTROL_AXES[ctl]
            if any(CONTROL_AXES[t] == axis for t in self._win_taken):
                continue                # axis already taken: first one wins
            self._win_taken.append(ctl)
            took = True

        if took:
            # Re-assert EVERY action taken in this window, not just the new
            # one. That is what makes a merge work after the first key was
            # released: with latch off the release had neutralized that axis, and
            # this puts it back.
            for ctl in self._win_taken:
                setattr(self, CONTROL_AXES[ctl], self._endpoints[ctl])

        if not self.latch:
            # Release semantics: an axis with nothing held and nothing owned
            # by the live merge window returns to physical no-op.
            owned = {CONTROL_AXES[t] for t in self._win_taken} if took else set()
            if not (held & {"w", "s"}) and "throttle" not in owned:
                self.throttle = self.noop_throttle
            if not (held & {"a", "d"}) and "steer" not in owned:
                self.steer = self.noop_steer

    def _update_analog(self, dt: float, held: frozenset) -> None:
        if STOP_CONTROL in held:
            self.throttle = self.noop_throttle
            self.steer = self.noop_steer
            return
        thr = self._axis_dir(held, "w", "s")
        ste = self._axis_dir(held, "d", "a")

        def step(v: float, d, lo: float, hi: float, neutral: float) -> float:
            if d is None:
                return v                       # invalid combo: hold this axis
            if d != 0.0:
                v += RAMP_RATE * dt * d
            elif not self.latch:
                if v > neutral:
                    v = max(neutral, v - DECAY_RATE * dt)
                elif v < neutral:
                    v = min(neutral, v + DECAY_RATE * dt)
            return float(np.clip(v, lo, hi))

        self.throttle = step(
            self.throttle, thr, self.reverse_throttle,
            self.forward_throttle, self.noop_throttle)
        self.steer = step(self.steer, ste, self.left_steer,
                          self.right_steer, self.noop_steer)


# ==========================================================================
# Settings panel model (renderer-agnostic)
# ==========================================================================
class SettingsPanel:
    """Keyboard-driven settings list. Owns no rendering."""

    LIVE = {"denoising_steps", "horizon_chunks", "decode_half_res", "playback_fps",
            "max_buffered_horizons", "ladder_mode", "playback_mode"}
    RESTART = {"precision", "compile_mode", "block_chunks"}
    #: UI-only settings stored in cfg.extra rather than as EngineConfig fields.
    CONTROL_ENDPOINT_DEFAULTS = {
        "forward_throttle": DEFAULT_FORWARD_THROTTLE,
        "reverse_throttle": DEFAULT_REVERSE_THROTTLE,
        "left_steer": DEFAULT_LEFT_STEER,
        "right_steer": DEFAULT_RIGHT_STEER,
    }
    UI_ONLY = {"controls", "latch", *CONTROL_ENDPOINT_DEFAULTS}
    #: Live settings that live in cfg.extra rather than as EngineConfig fields.
    EXTRA_LIVE = {"ladder_mode", "playback_mode"}
    RESET = {"seed_length_s"}

    def __init__(self, cfg: EngineConfig):
        self.cfg = replace(cfg)
        self.visible = False
        self.sel = 0
        self.pending_rebuild = False
        self.pending_seed_reset = False
        self.items = ["preset", "controls", "latch",
                      "forward_throttle", "reverse_throttle",
                      "left_steer", "right_steer",
                      "denoising_steps", "ladder_mode",
                      "seed_length_s", "horizon_chunks", "max_buffered_horizons",
                      "decode_half_res", "playback_mode", "playback_fps",
                      "precision", "compile_mode", "block_chunks"]

    _EXTRA_DEFAULTS = {"ladder_mode": "interp", "playback_mode": "auto"}

    def _get(self, key: str):
        if key == "preset":
            # Derived, never stored: hand-tuning any field it covers makes
            # this read back as 'custom' on its own.
            name = active_preset(self.cfg)
            return (f"{name}  ({PRESET_BLURB[name]})" if name in PRESET_BLURB
                    else name)
        if key == "seed_length_s":
            return seed_seconds_from_chunks(self.cfg.seed_prefill_chunks)
        if key in self.UI_ONLY:
            if key == "controls":
                default = "digital"
            elif key == "latch":
                default = True
            else:
                default = self.CONTROL_ENDPOINT_DEFAULTS[key]
            return self.cfg.extra.get(key, default)
        if key in self.EXTRA_LIVE:
            return self.cfg.extra.get(key, self._EXTRA_DEFAULTS.get(key))
        return getattr(self.cfg, key)

    def toggle(self) -> None:
        self.visible = not self.visible

    def move(self, d: int) -> None:
        self.sel = (self.sel + d) % len(self.items)

    def select(self, name: str) -> bool:
        """Select a field by NAME. Scripts use this so that inserting a new
        setting cannot silently repoint their keystrokes at another field."""
        if name in self.items:
            self.sel = self.items.index(name)
            return True
        return False

    def adjust(self, d: int) -> Tuple[str, object]:
        """Adjust the selected field; returns (field, new_value)."""
        key = self.items[self.sel]
        if key == "preset":
            cur = active_preset(self.cfg)
            i = PRESET_ORDER.index(cur) if cur in PRESET_ORDER else -1
            name = PRESET_ORDER[(i + d) % len(PRESET_ORDER)]
            for f, v in PRESETS[name].items():
                setattr(self.cfg, f, v)
            return key, name
        if key == "controls":
            modes = list(ControlScheme.MODES)
            i = modes.index(self._get("controls")) if self._get("controls") in modes else 0
            self.cfg.extra["controls"] = modes[(i + d) % len(modes)]
        elif key == "latch":
            self.cfg.extra["latch"] = not bool(self._get("latch"))
        elif key in self.CONTROL_ENDPOINT_DEFAULTS:
            value = float(self._get(key)) + d * 0.05
            if key in ("forward_throttle", "right_steer"):
                value = float(np.clip(value, 0.05, 1.0))
            else:
                value = float(np.clip(value, -1.0, -0.05))
            self.cfg.extra[key] = round(value, 2)
        elif key == "max_buffered_horizons":
            self.cfg.max_buffered_horizons = int(
                np.clip(self.cfg.max_buffered_horizons + d, 1, 4))
        elif key == "denoising_steps":
            self.cfg.denoising_steps = int(np.clip(
                self.cfg.denoising_steps + d, 1, 12))
        elif key == "seed_length_s":
            self.cfg.seed_prefill_chunks = int(np.clip(
                self.cfg.seed_prefill_chunks + d, 1, SEED_PREFILL_CHUNKS))
            self.pending_seed_reset = True
        elif key == "playback_mode":
            self.cfg.extra["playback_mode"] = (
                "manual" if self._get("playback_mode") == "auto" else "auto")
        elif key == "ladder_mode":
            i = (LADDER_CHOICES.index(self._get("ladder_mode"))
                 if self._get("ladder_mode") in LADDER_CHOICES else 0)
            self.cfg.extra["ladder_mode"] = LADDER_CHOICES[(i + d) % len(LADDER_CHOICES)]
        elif key == "horizon_chunks":
            self.cfg.horizon_chunks = int(np.clip(self.cfg.horizon_chunks + d, 1, 12))
        elif key == "decode_half_res":
            self.cfg.decode_half_res = not self.cfg.decode_half_res
        elif key == "playback_fps":
            self.cfg.playback_fps = float(np.clip(self.cfg.playback_fps + d * 2.0, 2.0, 60.0))
        elif key == "precision":
            i = PRECISION_CHOICES.index(self.cfg.precision) if self.cfg.precision in PRECISION_CHOICES else 1
            self.cfg.precision = PRECISION_CHOICES[(i + d) % len(PRECISION_CHOICES)]
            self.pending_rebuild = True
        elif key == "block_chunks":
            opts = BLOCK_CHOICES
            i = opts.index(int(self.cfg.block_chunks)) if int(self.cfg.block_chunks) in opts else 0
            self.cfg.block_chunks = opts[(i + d) % len(opts)]
            self.pending_rebuild = True
        elif key == "compile_mode":
            i = COMPILE_CHOICES.index(self.cfg.compile_mode) if self.cfg.compile_mode in COMPILE_CHOICES else 0
            self.cfg.compile_mode = COMPILE_CHOICES[(i + d) % len(COMPILE_CHOICES)]
            self.pending_rebuild = True
        return key, self._get(key)

    def lines(self) -> List[str]:
        out = []
        for i, key in enumerate(self.items):
            val = self._get(key)
            if key == "seed_length_s":
                val = f"{val:g}s ({self.cfg.seed_prefill_chunks} chunks)"
            elif isinstance(val, float):
                val = f"{val:g}"
            if key in self.RESTART:
                tag = "  (restart required)"
            elif key in self.RESET:
                tag = "  (reset required)"
            else:
                tag = ""
            cur = ">" if i == self.sel else " "
            out.append(f"{cur} {key:<16} {val}{tag}")
        if self.pending_rebuild:
            out.append("  [ENTER] apply -> rebuild engine")
        elif self.pending_seed_reset:
            out.append("  [ENTER] apply -> reset ride")
        out.append("  [TAB/ESC] close")
        return out


# ==========================================================================
# Application
# ==========================================================================
class PlayerApp:
    def __init__(self, cfg: EngineConfig,
                 factory: Callable[[EngineConfig], EngineBase],
                 window_scale: float = 1.0,
                 headless: bool = False,
                 record_dir: str = "interactive/recordings"):
        self.cfg = replace(cfg)
        self.factory = factory
        self.window_scale = window_scale
        self.headless = headless
        self.record_dir = record_dir

        self.ring = FrameRing()
        self.worker = EngineWorker(self.cfg, factory, self.ring)
        self.panel = SettingsPanel(self.cfg)

        self.controls = ControlScheme(
            mode=str(cfg.extra.get("controls", "digital")),
            latch=bool(cfg.extra.get("latch", True)),
            forward_throttle=float(cfg.extra.get(
                "forward_throttle", DEFAULT_FORWARD_THROTTLE)),
            reverse_throttle=float(cfg.extra.get(
                "reverse_throttle", DEFAULT_REVERSE_THROTTLE)),
            left_steer=float(cfg.extra.get(
                "left_steer", DEFAULT_LEFT_STEER)),
            right_steer=float(cfg.extra.get(
                "right_steer", DEFAULT_RIGHT_STEER)),
        )
        self.throttle = DEFAULT_NOOP_THROTTLE
        self.steer = DEFAULT_NOOP_STEER
        self.last_frame: Optional[np.ndarray] = None
        self.frames_shown = 0        # CONTENT frames advanced through
        self.generated_frames_shown = 0  # excludes real seed frames
        self.display_ticks = 0       # blits / presentations (held frames too)
        self.actual_fps = 0.0        # measured DISPLAY rate
        self.sink: Optional[FrameSink] = None
        self.quit = False
        self.seed_root: Optional[str] = cfg.extra.get("seed_root")
        self.video_dir: Optional[str] = cfg.extra.get("video_dir")
        self.playback_mode = str(cfg.extra.get("playback_mode", "auto"))
        # Servoed content-advance rate (fps) and its fractional accumulator.
        self._content_fps: Optional[float] = None
        self._content_accum = 0.0
        self._content_starved_ticks = 0
        self.display_fps_measured = 0.0
        self.content_fps_measured = 0.0
        self._rate_hist: Deque[Tuple[float, int]] = deque()
        self._max_frame_gap = 0.0        # worst gap between CONTENT advances
        self._max_tick_gap = 0.0         # worst DISPLAY tick gap, steady state
        self._max_tick_gap_busy = 0.0    # worst DISPLAY tick gap during a build
        self._last_advance_at: Optional[float] = None
        self._last_tick_at: Optional[float] = None
        self._seen_disc = 0
        # Responsiveness instrumentation (measured, not assumed).
        self.action_changed_at = 0.0
        self.reset_requested_at = 0.0
        self.notice: str = ""
        self.notice_until = 0.0
        self._loop_ms: List[float] = []

        # --- alternative front-end hooks (web mode) ----------------------
        # web_play.py drives the SAME PlayerApp headlessly: it supplies the
        # held-key set each poll and receives every presented frame. Nothing
        # else about pacing, controls or the engine changes, so the browser
        # and the pygame window are the same player with a different surface.
        self.input_provider: Optional[Callable[[], set]] = None
        self.frame_hook: Optional[Callable[[Optional[np.ndarray], bool], None]] = None

        self.pg = None
        self.screen = None
        self.font = None
        self.small_font = None

    # -- input ------------------------------------------------------------
    def apply_controls(self, dt: float, held) -> None:
        """Feed the held control set through the scheme and latch the result."""
        before = (self.controls.throttle, self.controls.steer)
        self.controls.update(dt, held)
        self.throttle = self.controls.throttle
        self.steer = self.controls.steer
        if (self.throttle, self.steer) != before:
            self.action_changed_at = time.perf_counter()
            if bool(self.cfg.extra.get("drop_stale_on_action", True)):
                # Everything still queued was generated under the OLD action;
                # showing it delays the response by the whole buffer. Drop it
                # so the next visible frame is the first one that could react.
                self.ring.clear()
        self.worker.set_action(self.throttle, self.steer)

    def ramp(self, dt: float, up: bool, down: bool, left: bool, right: bool) -> None:
        """Back-compat shim for scripted callers: booleans -> held set."""
        held = set()
        if up:
            held.add("w")
        if down:
            held.add("s")
        if left:
            held.add("a")
        if right:
            held.add("d")
        self.apply_controls(dt, held)

    # -- HUD text ---------------------------------------------------------
    def hud_lines(self) -> List[str]:
        st = self.worker.last_stats
        state = self.worker.state
        if self.worker.error:
            state = f"error: {self.worker.error}"
        # Buffer depth in seconds at the CONTENT rate: that is how long the
        # queued frames will take to play out, and so what "how far ahead am
        # I" really means.
        content = self._content_fps if self._content_fps else self.display_fps()
        ahead = len(self.ring) / max(content, 1e-3)
        gen = self.ring.throughput_fps
        gen_s = f"{gen:4.1f}" if gen is not None else "  --"
        mode_s = "auto" if self.playback_mode == "auto" else "fixed"
        l1 = (f"[{self.controls.describe():>2}] thr {self.throttle:+.3f} "
              f"str {self.steer:+.3f}   ahead {ahead:4.1f}s"
              # Display is the MEASURED refresh rate, not the promise, so a
              # genuine presentation problem is visible rather than papered
              # over. Content is the servoed advance rate.
              f"   display {self.display_fps_measured or self.display_fps():4.1f} fps"
              f" | content {content:4.1f} fps"
              f" ({mode_s})   gen {gen_s} fps   state {state}")
        if st is None:
            l2 = "no horizon stats yet"
        else:
            l2 = (f"H{st.horizon_index}  gen {st.gen_fps:6.1f} fps   "
                  f"first-frame {st.first_frame_latency_s * 1000:6.0f} ms   "
                  f"vram {st.peak_vram_gb:4.1f} GB   "
                  f"steps {st.denoising_steps}   prec {st.precision}")
        block_chunks = int(getattr(self.cfg, "block_chunks", 1))
        l3 = (f"[{active_preset(self.cfg)}]  ride {self.worker.ride_id}"
              f"  {self.controls.mode}/{'latch' if self.controls.latch else 'no-latch'}"
              f"  horizons {self.worker.horizons_done}"
              f"  generated {self.generated_frames_shown / PLAYBACK_FPS:.1f}s"
              f"  chunks/h {self.cfg.horizon_chunks}"
              + (f"  JOINT x{block_chunks}"
                 if block_chunks > 1 else "")
              + ("   REC" if self.sink is not None else "")
              + "   WASD move  [SPACE] stop  [1/2/3] preset  [P] seed  [R] reset"
                "  [TAB] settings  [V] rec  [Q] quit")
        return [l1, l2, l3]

    # -- pygame plumbing --------------------------------------------------
    def _init_display(self) -> None:
        import pygame
        self.pg = pygame
        pygame.init()
        pygame.display.set_caption("ARRWM interactive world model")
        w = int(PIXEL_W * self.window_scale)
        h = int(PIXEL_H * self.window_scale)
        self.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
        pygame.font.init()
        self.font = pygame.font.SysFont("monospace", 15)
        self.small_font = pygame.font.SysFont("monospace", 14)

    def _blit(self, frame: Optional[np.ndarray]) -> None:
        pg = self.pg
        self.screen.fill((10, 10, 14))
        sw, sh = self.screen.get_size()
        if frame is not None:
            surf = pg.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            surf = pg.transform.smoothscale(surf, (sw, sh))
            self.screen.blit(surf, (0, 0))

        lines = self.hud_lines()
        strip_h = 6 + 18 * len(lines)
        strip = pg.Surface((sw, strip_h), pg.SRCALPHA)
        strip.fill((0, 0, 0, 165))
        self.screen.blit(strip, (0, sh - strip_h))
        for i, txt in enumerate(lines):
            self.screen.blit(self.font.render(txt, True, (235, 235, 235)),
                             (8, sh - strip_h + 3 + 18 * i))

        if self.worker.state in ("rebuilding", "warmup"):
            msg = (self.worker.warmup_msg or "compiling kernels ..."
                   if self.worker.state == "warmup" else "rebuilding engine ...")
            txt = self.font.render(msg, True, (255, 210, 90))
            box = pg.Surface((sw, 46), pg.SRCALPHA)
            box.fill((0, 0, 0, 200))
            self.screen.blit(box, (0, sh // 2 - 23))
            self.screen.blit(txt, (sw // 2 - txt.get_width() // 2, sh // 2 - 8))

        # Transient acknowledgement (reset / reseed / setting change). Drawn
        # from UI state so it appears on the NEXT frame, without waiting for
        # the worker to notice anything.
        if self.notice and time.perf_counter() < self.notice_until:
            txt = self.font.render(self.notice, True, (255, 225, 120))
            box = pg.Surface((txt.get_width() + 24, 34), pg.SRCALPHA)
            box.fill((0, 0, 0, 205))
            self.screen.blit(box, (sw // 2 - box.get_width() // 2, 24))
            self.screen.blit(txt, (sw // 2 - txt.get_width() // 2, 33))

        if self.panel.visible:
            plines = self.panel.lines()
            ph = 12 + 20 * len(plines)
            pw = min(sw - 40, 460)
            box = pg.Surface((pw, ph), pg.SRCALPHA)
            box.fill((12, 12, 20, 225))
            self.screen.blit(box, (20, 20))
            for i, txt in enumerate(plines):
                colr = (255, 230, 120) if txt.startswith(">") else (215, 215, 215)
                self.screen.blit(self.small_font.render(txt, True, colr),
                                 (30, 26 + 20 * i))
        pg.display.flip()

    def _handle_key(self, key_name: str) -> None:
        """Discrete key presses (both real and scripted)."""
        if key_name in ("q", "escape") and not self.panel.visible:
            self.quit = True
        elif key_name == "escape" and self.panel.visible:
            self.panel.visible = False
        elif key_name == "tab":
            self.panel.toggle()
        elif key_name == "r":
            # Acknowledge on the UI thread IMMEDIATELY. The worker can only
            # notice the request at its next chunk boundary and then spends
            # seconds re-prefilling; without this the screen keeps showing
            # stale video and R feels dead.
            self.reset_requested_at = time.perf_counter()
            self.ring.clear()
            self.generated_frames_shown = 0
            self._notify("resetting...", 6.0)
            self.worker.request_reset()
        elif key_name in ("1", "2", "3"):
            # Digits are free in the main loop: the seed picker's digit filter
            # runs in its own modal loop and never reaches this handler.
            self.apply_preset(PRESET_ORDER[int(key_name) - 1])
        elif key_name == "p":
            self.open_seed_picker()
        elif key_name == "v":
            self.toggle_record()
        elif self.panel.visible and key_name in ("up", "down", "left", "right", "return"):
            if key_name == "up":
                self.panel.move(-1)
            elif key_name == "down":
                self.panel.move(+1)
            elif key_name in ("left", "right"):
                field, val = self.panel.adjust(+1 if key_name == "right" else -1)
                self._apply_panel_field(field, val)
            elif key_name == "return":
                self.apply_rebuild()

    def apply_preset(self, name: str) -> None:
        """Switch to a named quality preset.

        ``denoising_steps`` applies live; ``decoder`` needs a rebuild. All
        three presets use taew2_1, so a preset switch is normally live-only --
        a rebuild happens only when coming from a different decoder (e.g. a
        session launched with --decoder wan).
        """
        if name not in PRESETS:
            return
        spec = PRESETS[name]
        needs_rebuild = str(getattr(self.cfg, "decoder", "")) != spec["decoder"]
        target_steps = int(spec["denoising_steps"])
        for f, v in spec.items():
            setattr(self.cfg, f, v)
        self.cfg.denoising_steps = target_steps
        self.panel.cfg.denoising_steps = target_steps
        self.panel.cfg.decoder = spec["decoder"]
        if needs_rebuild:
            self._notify(f"preset {name} - rebuilding for {spec['decoder']}", 6.0)
            self.ring.clear()
            self.worker.request_rebuild(self.cfg)
        else:
            self.worker.request_live_settings(
                denoising_steps=target_steps)
            self._notify(f"preset: {name}  ({PRESET_BLURB[name]})", 3.0)

    def _apply_panel_field(self, field: str, val) -> None:
        if field == "preset":
            self.apply_preset(val)
            return
        if field in SettingsPanel.UI_ONLY:
            self.cfg.extra[field] = val
            if field == "controls":
                self.controls.mode = str(val)
            elif field == "latch":
                self.controls.latch = bool(val)
            else:
                setattr(self.controls, field, float(val))
                ctl = {
                    "forward_throttle": "w", "reverse_throttle": "s",
                    "left_steer": "a", "right_steer": "d",
                }[field]
                self.controls._endpoints[ctl] = float(val)
            self._notify(f"{field} = {val}")
            return
        if field in SettingsPanel.LIVE:
            if field in SettingsPanel.EXTRA_LIVE:
                self.cfg.extra[field] = val
            else:
                setattr(self.cfg, field, val)
            if field in ("playback_fps", "playback_mode"):
                # Pacing is entirely a display concern; the engine has no say.
                if field == "playback_mode":
                    self.playback_mode = str(val)
                    self._notify(f"playback = {val}")
                return
            self.worker.request_live_settings(**{field: val})

    def apply_rebuild(self) -> None:
        if not (self.panel.pending_rebuild or self.panel.pending_seed_reset):
            return
        self.cfg = replace(self.panel.cfg)
        self.generated_frames_shown = 0
        self.ring.clear()
        if self.panel.pending_rebuild:
            self.panel.pending_rebuild = False
            self.panel.pending_seed_reset = False
            self._notify("applying settings - rebuilding engine", 6.0)
            self.worker.request_rebuild(self.cfg)
        else:
            self.panel.pending_seed_reset = False
            chunks = int(self.cfg.seed_prefill_chunks)
            self.worker.request_live_settings(seed_prefill_chunks=chunks)
            self.worker.request_reset()
            self._notify(
                f"seed length = {seed_seconds_from_chunks(chunks):g}s - resetting",
                6.0)

    # -- presentation: fixed display rate, adaptive content rate -----------
    #
    # DISPLAY and CONTENT are two different rates and this is the whole
    # design:
    #
    #   * DISPLAY rate  = cfg.playback_fps (16 Hz by default). The render loop
    #     ALWAYS ticks and blits at this rate, whatever the engine is doing.
    #     The window is therefore never seen to run at 7 fps.
    #   * CONTENT rate  = how fast we walk forward through the frame ring. In
    #     auto mode this is servoed to the measured sustained generation
    #     throughput (the logic below, unchanged in spirit from the original
    #     adaptive pacing), so we never outrun the generator.
    #
    # A fractional accumulator bridges the two: each display tick adds
    # content_fps / display_fps to it, and a whole unit pops one frame. When
    # content is slower than display the accumulator spends most ticks below
    # 1 and the current frame is simply re-blitted (a held frame) -- the
    # result is honest slow motion at a rock-steady refresh, instead of a
    # stuttering low-rate window.
    #
    #: Anti-wedge floor for the content rate only. This is NOT a "playback
    #: never goes below this" guarantee any more (the old MIN_AUTO_FPS was):
    #: content is allowed to crawl, because the display no longer crawls with
    #: it. It exists purely so the servo cannot latch at exactly zero and take
    #: forever to slew back up once frames start arriving again.
    MIN_CONTENT_FPS = 0.5
    #: Content-rate slew limit, fps per second. Keeps changes imperceptible.
    FPS_SLEW = 4.0
    #: Ring depth (frames) under which we ease the content rate down.
    #: Half a chunk: frames arrive in 12-frame bursts, so reacting only at
    #: 4 frames leaves too little runway to bridge the gap to the next burst.
    STARVE_FRAMES = 6
    #: Advance a few percent SLOWER than we generate. Production is bursty (a
    #: whole chunk lands at once, then nothing); matching the mean rate
    #: exactly means running dry at the end of every burst. The deficit
    #: accumulates a small cushion instead, at a cost far below perception.
    AUTO_FPS_SAFETY = 0.92
    #: Seconds of history for the measured display/content rate readouts.
    RATE_WINDOW_S = 3.0
    #: Worker states during which the worker thread holds the GIL for long
    #: stretches (model construction, gc, cuda sync), so the render loop
    #: cannot be expected to keep its tick. Reported separately, not ignored.
    BUSY_STATES = frozenset({"starting", "rebuilding", "warmup", "picker"})

    def display_fps(self) -> float:
        """The rate the window refreshes at. Never adaptive.

        In auto mode cfg.playback_fps is the *cap* on content and doubles as
        the display rate (16 Hz by default). In manual mode (--playback <n>)
        content and display are both pinned to that number, which is exactly
        the pre-adaptive behaviour -- so `--playback 8` really does give an
        8 Hz window. That is the documented opt-out; auto is the mode that
        holds 16 Hz.
        """
        return max(float(self.cfg.playback_fps), 1e-3)

    def target_content_fps(self) -> float:
        """Where the CONTENT-advance rate wants to be, before slew limiting.

        Paced to the measured sustained generation throughput, so consuming
        chunk N takes about as long as producing chunk N+1. Capped at the
        display rate: content is never shown faster than the window refreshes.
        """
        cap = self.display_fps()
        if self.playback_mode != "auto":
            return cap
        thr = self.ring.throughput_fps
        if thr is None:
            return cap                     # no measurement yet
        target = min(max(thr * self.AUTO_FPS_SAFETY, self.MIN_CONTENT_FPS), cap)
        # Servo on buffer depth: starving -> slow down a little more so the
        # next chunk lands before we run dry; comfortable -> drift back up so
        # queued video (and therefore latency) does not accumulate.
        buf = len(self.ring)
        if buf < self.STARVE_FRAMES:
            target = max(self.MIN_CONTENT_FPS, target * 0.85)
        elif buf > PIXEL_FRAMES_PER_CHUNK:
            target = min(cap, target * 1.10)
        return target

    def update_content_fps(self, dt: float) -> float:
        """Slew-limited move toward target_content_fps(); returns the rate."""
        target = self.target_content_fps()
        if self._content_fps is None:
            self._content_fps = target
        else:
            step = self.FPS_SLEW * max(dt, 0.0)
            delta = float(np.clip(target - self._content_fps, -step, step))
            self._content_fps += delta
        return max(self._content_fps, 1e-3)

    def _rate_sample(self, advanced: int, now: float) -> None:
        """Record one display tick for the measured display/content readouts."""
        self._rate_hist.append((now, int(advanced)))
        cutoff = now - self.RATE_WINDOW_S
        while len(self._rate_hist) > 2 and self._rate_hist[0][0] < cutoff:
            self._rate_hist.popleft()
        span = self._rate_hist[-1][0] - self._rate_hist[0][0]
        if span > 1e-6:
            ticks = len(self._rate_hist) - 1
            self.display_fps_measured = ticks / span
            self.content_fps_measured = (
                sum(n for _, n in list(self._rate_hist)[1:]) / span)
        self.actual_fps = float(self.display_fps_measured)

    def present_tick(self, now: float, dt: float) -> None:
        """One display tick: advance content by a fraction, then blit.

        Runs at exactly `display_fps()`. The ring is advanced by
        content_fps/display_fps frames' worth per tick via a fractional
        accumulator, so a slow generator produces held (repeated) frames
        rather than a slow window.
        """
        content = self.update_content_fps(dt)
        self._content_accum += content / self.display_fps()
        advanced = 0
        while self._content_accum >= 1.0:
            frame, generated = self.ring.pop_tagged()
            if frame is None:
                # Starved: nothing to advance to. Drop the accumulated debt
                # instead of carrying it -- otherwise the ring's next burst is
                # fast-forwarded through to "catch up", which is the stutter
                # this design exists to remove.
                self._content_starved_ticks += 1
                self._content_accum = 0.0
                break
            self.last_frame = frame
            self.frames_shown += 1
            if generated:
                self.generated_frames_shown += 1
            self._content_accum -= 1.0
            advanced += 1

        if advanced:
            if self._last_advance_at is not None:
                gap = now - self._last_advance_at
                disc = self.ring.discontinuities
                if self.frames_shown > 2 and disc == self._seen_disc:
                    self._max_frame_gap = max(self._max_frame_gap, gap)
                self._seen_disc = disc
            self._last_advance_at = now

        # Recording captures every DISPLAYED tick, not every new content
        # frame: the mp4 is written at the display rate and contains the same
        # held/repeated frames the user actually saw, so the recording plays
        # back at exactly the pace of the live session.
        if self.sink is not None and self.last_frame is not None:
            self.sink.add(self.last_frame)

        if self._last_tick_at is not None:
            gap = now - self._last_tick_at
            # An engine build/rebuild runs gc.collect(), torch.cuda.synchronize()
            # and model construction on the worker thread, all of which hold
            # the GIL and therefore stall this loop for a few hundred ms. That
            # is a real hiccup and is measured — just kept in its own bucket,
            # because it says nothing about whether steady-state pacing holds
            # 16 Hz.
            if self.worker.state in self.BUSY_STATES:
                self._max_tick_gap_busy = max(self._max_tick_gap_busy, gap)
            else:
                self._max_tick_gap = max(self._max_tick_gap, gap)
        self._last_tick_at = now
        self.display_ticks += 1
        self._rate_sample(advanced, now)

        if self.frame_hook is not None:
            # `advanced` is 0 when the ring was held/starved, so a web front
            # end can send ONLY on content change and let the browser hold the
            # last frame -- far less bandwidth than re-sending held frames.
            self.frame_hook(self.last_frame, bool(advanced))

        if not self.headless:
            self._blit(self.last_frame)

    def _notify(self, text: str, seconds: float = 2.0) -> None:
        """Show a short overlay message starting on the very next drawn frame."""
        self.notice = text
        self.notice_until = time.perf_counter() + seconds

    # -- seed picker ------------------------------------------------------
    def open_seed_picker(self) -> None:
        """Pause the engine, browse rides, and reseed with the chosen one."""
        if self.headless or self.screen is None:
            print("[play] seed picker needs a display")
            return
        from interactive.seed_picker import (
            SeedPicker, ThumbnailCache, list_rides, list_videos)
        self._notify("opening seed picker...", 3.0)
        self._blit(self.last_frame)
        # The picker decodes thumbnails on the default stream; the engine must
        # be off the GPU first or the two fight over it.
        if not self.worker.pause_for_ui(timeout=30.0):
            print("[play] worker did not park in time; opening picker anyway")
        cache = ThumbnailCache()
        try:
            # User videos first: they are the short, hand-picked list, and
            # burying them behind 2,418 rides would make the feature invisible.
            rides = list_videos(self.video_dir) + list_rides(self.seed_root)
            picker = SeedPicker(self.screen, rides, cache=cache,
                                font=self.font, small_font=self.small_font)
            chosen = picker.run()
        except Exception as exc:
            print(f"[play] seed picker failed: {exc}")
            chosen = None
        finally:
            cache.close()
            self.worker.resume_from_ui()
        if chosen:
            self.ring.clear()
            self.generated_frames_shown = 0
            self._notify(f"reseeding -> {os.path.splitext(os.path.basename(chosen))[0]}", 6.0)
            self.worker.request_seed(chosen)

    def toggle_record(self) -> None:
        if self.sink is None:
            os.makedirs(self.record_dir, exist_ok=True)
            path = os.path.join(self.record_dir,
                                f"session_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
            # Recorded at the DISPLAY rate: present_tick() appends one frame
            # per displayed tick (repeating held frames when content is
            # slower), so the mp4 is a faithful capture of what was on screen
            # rather than a sped-up reel of the distinct generated frames.
            self.sink = FrameSink(path, self.display_fps())
            print(f"[rec] recording -> {path}")
        else:
            self.sink.write()
            self.sink = None

    # -- main loop --------------------------------------------------------
    def run(self, max_seconds: Optional[float] = None,
            script: Optional[List[Tuple[float, str]]] = None) -> int:
        """Run the app. `script` is a list of (t_seconds, key_name) events."""
        if not self.headless:
            self._init_display()
        self.worker.start()

        script = sorted(script or [], key=lambda x: x[0])
        scripted = bool(script)
        s_idx = 0
        self._scripted_held: set = set()
        t0 = time.perf_counter()
        next_tick_at = t0
        last_poll_t = t0
        last_tick_t = t0

        while not self.quit:
            now = time.perf_counter()
            elapsed = now - t0
            if max_seconds is not None and elapsed >= max_seconds:
                break

            # Input is polled faster than the display ticks, so the control
            # integrator gets its own dt.
            dt = max(now - last_poll_t, 1e-4)
            last_poll_t = now

            # --- scripted events -------------------------------------
            while s_idx < len(script) and script[s_idx][0] <= elapsed:
                ev = script[s_idx][1]
                s_idx += 1
                print(f"[script] t={elapsed:5.2f}s  {ev}")
                if ev.startswith("action:"):
                    _, thr, ste = ev.split(":")
                    self.throttle = float(np.clip(float(thr), -1, 1))
                    self.steer = float(np.clip(float(ste), -1, 1))
                    self.worker.set_action(self.throttle, self.steer)
                elif ev.startswith("sel:"):
                    name = ev.split(":", 1)[1].strip()
                    if not self.panel.select(name):
                        print(f"[script] unknown panel field {name!r}")
                elif ev.startswith("hold:"):
                    # "hold:w", "hold:w,a", "hold:stop", "hold:none"
                    spec = ev.split(":", 1)[1].strip()
                    self._scripted_held = (
                        set() if spec in ("", "none")
                        else {t.strip() for t in spec.split(",") if t.strip()})
                else:
                    self._handle_key(ev)

            # --- scripted held-key set (digital semantics under test) ---
            if scripted:
                self.apply_controls(dt, self._scripted_held)

            # --- real input ------------------------------------------
            if not self.headless and not scripted:
                pg = self.pg
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        self.quit = True
                    elif event.type == pg.KEYDOWN:
                        self._handle_key(pg.key.name(event.key))
                keys = pg.key.get_pressed()
                held = set()
                if not self.panel.visible:
                    if keys[pg.K_w] or keys[pg.K_UP]:
                        held.add("w")
                    if keys[pg.K_s] or keys[pg.K_DOWN]:
                        held.add("s")
                    if keys[pg.K_a] or keys[pg.K_LEFT]:
                        held.add("a")
                    if keys[pg.K_d] or keys[pg.K_RIGHT]:
                        held.add("d")
                    if keys[pg.K_SPACE]:
                        held.add(STOP_CONTROL)
                self.apply_controls(dt, held)

            # --- external front end (web mode) -----------------------
            # Same ControlScheme, same dt cadence as the pygame branch above;
            # only the source of the held set differs.
            if self.headless and not scripted and self.input_provider is not None:
                self.apply_controls(dt, self.input_provider())

            # --- present: ALWAYS tick at the display rate ---------------
            # The display rate is fixed (16 Hz by default). It does not react
            # to generation at all; only the content-advance rate inside
            # present_tick() does. A starved ring re-blits the held frame.
            period = 1.0 / self.display_fps()
            if now >= next_tick_at:
                self.present_tick(now, max(now - last_tick_t, 1e-4))
                last_tick_t = now
                next_tick_at += period
                if now - next_tick_at > 0.5:   # fell far behind; resync
                    next_tick_at = now + period

            # Yield to the engine worker; never spin the GIL flat out. The
            # poll cap keeps key input responsive between display ticks.
            slack = next_tick_at - time.perf_counter()
            time.sleep(max(0.0005, min(slack, 0.006)))

            if self.worker.state == "error":
                print(f"[play] engine worker failed: {self.worker.error}")
                break

        # --- shutdown -------------------------------------------------
        if self.sink is not None:
            self.sink.write()
            self.sink = None
        self.worker.request_stop()
        self.worker.join(timeout=20.0)
        if not self.headless and self.pg is not None:
            self.pg.quit()
        return 1 if self.worker.error else 0


# ==========================================================================
# Smoke test
# ==========================================================================
def _test_presets() -> List[str]:
    """Preset resolution and cycling, with no engine involved."""
    bad: List[str] = []

    def check(label, got, want):
        if got != want:
            bad.append(f"{label}: got {got!r}, want {want!r}")

    cfg = EngineConfig(ckpt_path="", seed_zarr="")
    # Each preset must resolve back to its own name.
    for name, spec in PRESETS.items():
        for f, v in spec.items():
            setattr(cfg, f, v)
        check(f"resolve {name}", active_preset(cfg), name)
    # Hand-tuning a covered field falls out of the preset.
    cfg.denoising_steps = 3
    check("hand-tuned -> custom", active_preset(cfg), PRESET_CUSTOM)
    # A field the presets do NOT cover must not affect resolution.
    for f, v in PRESETS["speed"].items():
        setattr(cfg, f, v)
    cfg.horizon_chunks = 12
    check("uncovered field stays speed", active_preset(cfg), "speed")

    # Panel cycling walks the order and wraps.
    panel = SettingsPanel(cfg)
    if not panel.select("preset"):
        bad.append("panel has no 'preset' entry")
    else:
        if panel.items[0] != "preset":
            bad.append(f"preset is not the top entry (got {panel.items[0]!r})")
        seen = []
        for _ in range(len(PRESET_ORDER) + 1):
            _, name = panel.adjust(+1)
            seen.append(name)
        start = PRESET_ORDER.index("speed")
        want = [PRESET_ORDER[(start + i + 1) % len(PRESET_ORDER)]
                for i in range(len(PRESET_ORDER) + 1)]
        check("panel cycles + wraps", seen, want)
        # Cycling must actually write the settings through.
        check("cycle applied steps", panel.cfg.denoising_steps,
              PRESETS[seen[-1]]["denoising_steps"])

    if not bad:
        print(f"[smoke] presets: OK ({', '.join(PRESET_ORDER)}; "
              "custom on hand-tune; panel cycles)")
    return bad


def _test_combo_merge() -> List[str]:
    """The COMBO_MERGE_WINDOW_S sequential-tap merge, asserted directly.

    Time is driven by the dt handed to update(), so these are exact, not
    wall-clock races.
    """
    bad: List[str] = []
    W = COMBO_MERGE_WINDOW_S

    def check(label, got, want):
        if got != want:
            bad.append(f"{label}: got {got}, want {want}")

    def tap(c, ctl, then: float) -> None:
        """Press `ctl`, release it, and let `then` seconds pass in total."""
        c.update(0.016, {ctl})
        c.update(0.016, set())
        c.update(max(then - 0.032, 0.0), set())

    def scheme(latch=True):
        return ControlScheme("digital", latch=latch)

    # 1. tap W, tap A 0.3 s later -> merged diagonal, though the keys never
    #    overlapped. This is the case that used to discard W.
    c = scheme()
    tap(c, "w", 0.30)
    check("merge W then A @0.3s", (c.throttle, c.steer),
          (DEFAULT_FORWARD_THROTTLE, DEFAULT_NOOP_STEER))  # latched
    c.update(0.016, {"a"})
    check("merge W+A @0.3s", (c.throttle, c.steer),
          (DEFAULT_FORWARD_THROTTLE, DEFAULT_LEFT_STEER))

    # 2. tap W, tap S 0.3 s later -> same axis inside the window: the SECOND
    #    is ignored and the first keeps the axis.
    c = scheme()
    tap(c, "w", 0.30)
    c.update(0.016, {"s"})
    check("same-axis inside window ignored", (c.throttle, c.steer),
          (DEFAULT_FORWARD_THROTTLE, DEFAULT_NOOP_STEER))
    #    ... and it did not consume a slot: A still merges.
    c.update(0.016, set())
    c.update(0.10, {"a"})
    check("rejected same-axis kept its slot", (c.throttle, c.steer),
          (DEFAULT_FORWARD_THROTTLE, DEFAULT_LEFT_STEER))

    # 3. tap W, tap S 0.6 s later -> the window has lapsed, so S is a fresh
    #    valid input and wins its axis outright.
    c = scheme()
    tap(c, "w", W + 0.1)
    c.update(0.016, {"s"})
    check("new window: S wins the throttle axis", (c.throttle, c.steer),
          (DEFAULT_REVERSE_THROTTLE, DEFAULT_NOOP_STEER))

    # 4. tap W, tap A 0.6 s later -> also the calibrated diagonal, but for the
    #    OTHER reason:
    #    A is a fresh input on the steer axis and W's latch simply persists.
    #    ("latest valid input wins per axis, latch persists otherwise")
    c = scheme()
    tap(c, "w", W + 0.1)
    c.update(0.016, {"a"})
    check("new window: A takes steer, W's latch persists",
          (c.throttle, c.steer),
          (DEFAULT_FORWARD_THROTTLE, DEFAULT_LEFT_STEER))

    # 5. three presses inside one window -> only the first two uniques.
    c = scheme()
    tap(c, "w", 0.15)
    c.update(0.016, {"a"})
    c.update(0.016, set())
    c.update(0.10, {"d"})           # third press, still inside the window
    check("third press in window ignored", (c.throttle, c.steer),
          (DEFAULT_FORWARD_THROTTLE, DEFAULT_LEFT_STEER))
    #    ... but a press after the window lapses is honoured again.
    c.update(0.016, set())
    c.update(W + 0.1, set())
    c.update(0.016, {"d"})
    check("press after the window is honoured", (c.throttle, c.steer),
          (DEFAULT_FORWARD_THROTTLE, DEFAULT_RIGHT_STEER))

    # 6. merging works with latch OFF too: release neutralized throttle, and
    #    the merge puts it back.
    c = scheme(latch=False)
    tap(c, "w", 0.30)
    check("nolatch release neutralizes", (c.throttle, c.steer),
          (DEFAULT_NOOP_THROTTLE, DEFAULT_NOOP_STEER))
    c.update(0.016, {"a"})
    check("nolatch merge restores W", (c.throttle, c.steer),
          (DEFAULT_FORWARD_THROTTLE, DEFAULT_LEFT_STEER))

    # 7. simultaneous holds are unaffected by any of this.
    c = scheme()
    c.update(0.016, {"w", "a"})
    check("simultaneous W+A", (c.throttle, c.steer),
          (DEFAULT_FORWARD_THROTTLE, DEFAULT_LEFT_STEER))

    if not bad:
        print(f"[smoke] combo-merge: OK ({W:g}s window; sequential taps merge, "
              "same-axis second ignored, 3rd press ignored, latch on+off)")
    return bad


def _test_control_scheme() -> List[str]:
    """Assert the control semantics directly, with no engine in the way."""
    bad: List[str] = []

    def check(label, got, want):
        if got != want:
            bad.append(f"{label}: got {got}, want {want}")

    def clear(c) -> None:
        """SPACE, then let the merge window lapse: a clean slate for the next
        assertion. Without this every press in this function would land inside
        the previous one's merge window."""
        c.update(0.016, {STOP_CONTROL})
        c.update(0.016, set())
        c.update(COMBO_MERGE_WINDOW_S + 0.1, set())

    # --- digital + latch on (the default) --------------------------------
    c = ControlScheme("digital", latch=True)
    c.update(0.016, {"w"})
    check("digital W", (c.throttle, c.steer),
          (DEFAULT_FORWARD_THROTTLE, DEFAULT_NOOP_STEER))
    c.update(0.016, set())
    check("digital release latches", (c.throttle, c.steer),
          (DEFAULT_FORWARD_THROTTLE, DEFAULT_NOOP_STEER))
    clear(c)
    c.update(0.016, {"a"})
    check("digital A is -steer (left)", (c.throttle, c.steer),
          (DEFAULT_NOOP_THROTTLE, DEFAULT_LEFT_STEER))
    clear(c)
    c.update(0.016, {"d"})
    check("digital D is +steer (right)", (c.throttle, c.steer),
          (DEFAULT_NOOP_THROTTLE, DEFAULT_RIGHT_STEER))
    clear(c)
    c.update(0.016, {"w", "a"})
    check("digital W+A combo", (c.throttle, c.steer),
          (DEFAULT_FORWARD_THROTTLE, DEFAULT_LEFT_STEER))
    clear(c)
    c.update(0.016, {"s", "d"})
    check("digital S+D combo", (c.throttle, c.steer),
          (DEFAULT_REVERSE_THROTTLE, DEFAULT_RIGHT_STEER))
    # invalid same-axis combo -> unchanged (neither key counts)
    clear(c)
    c.update(0.016, {"s", "d"})
    before = (c.throttle, c.steer)
    clear2 = COMBO_MERGE_WINDOW_S + 0.1
    c.update(clear2, {"w", "s"})
    check("digital W+S invalid", (c.throttle, c.steer), before)
    c.update(clear2, {"a", "d"})
    check("digital A+D invalid", (c.throttle, c.steer), before)
    # holding one control while its opposite goes down is equally invalid
    clear(c)
    c.update(0.016, {"w"})
    c.update(0.016, {"w", "s"})
    check("digital W held + S pressed invalid", (c.throttle, c.steer),
          (DEFAULT_FORWARD_THROTTLE, DEFAULT_NOOP_STEER))
    # stop wins over anything held with it
    c.update(0.016, {"w", "a", STOP_CONTROL})
    check("digital stop wins", (c.throttle, c.steer),
          (DEFAULT_NOOP_THROTTLE, DEFAULT_NOOP_STEER))
    # ... and closes the window, so the next press cannot merge across it
    c.update(0.016, set())
    c.update(0.10, {"a"})
    check("digital nothing merges across stop", (c.throttle, c.steer),
          (DEFAULT_NOOP_THROTTLE, DEFAULT_LEFT_STEER))

    # --- digital + latch off ---------------------------------------------
    c = ControlScheme("digital", latch=False)
    c.update(0.016, {"w"})
    check("digital/nolatch W", (c.throttle, c.steer),
          (DEFAULT_FORWARD_THROTTLE, DEFAULT_NOOP_STEER))
    c.update(COMBO_MERGE_WINDOW_S + 0.1, set())
    check("digital/nolatch release -> no-op", (c.throttle, c.steer),
          (DEFAULT_NOOP_THROTTLE, DEFAULT_NOOP_STEER))

    # --- analog + latch off (the original behaviour) ---------------------
    c = ControlScheme("analog", latch=False)
    c.update(0.5, {"w"})
    if not (0.0 < c.throttle <= DEFAULT_FORWARD_THROTTLE + 1e-9):
        bad.append(f"analog ramps gradually: got {c.throttle}")
    ramped = c.throttle
    c.update(0.5, set())
    if not (c.throttle < ramped):
        bad.append(f"analog decays on release: {ramped} -> {c.throttle}")
    c.update(2.0, {"s"})
    check("analog reverse respects supported endpoint", c.throttle,
          DEFAULT_REVERSE_THROTTLE)

    # Explicit compatibility control for reproducing the old UI input.
    c = ControlScheme("digital", latch=True, reverse_throttle=-1.0)
    c.update(0.016, {"s"})
    check("custom reverse endpoint", c.throttle, -1.0)

    # --- analog + latch on -> holds instead of decaying -------------------
    c = ControlScheme("analog", latch=True)
    c.update(0.5, {"w"})
    held = c.throttle
    c.update(0.5, set())
    check("analog/latch holds on release", c.throttle, held)

    if not bad:
        print("[smoke] controls: all semantics OK "
              "(digital instant/latch/combos/invalid/stop, analog ramp+decay)")
    return bad


def _test_slow_content(cfg: EngineConfig, seconds: float = 7.0) -> List[str]:
    """The headline property: a SLOW engine must not slow the window down.

    Runs the mock with a per-chunk latency far above real time, then asserts
    from measured wall-clock timestamps that the display kept ticking at
    cfg.playback_fps while content advanced at roughly the mock's throughput.
    """
    from interactive.mock_engine import MockEngine
    bad: List[str] = []

    cfg = replace(cfg)
    cfg.extra = dict(cfg.extra)
    cfg.extra["playback_mode"] = "auto"
    cfg.denoising_steps = 4
    cfg.horizon_chunks = 6
    # 12 pixel frames per 1.5 s -> ~8 fps of content against a 16 Hz display.
    cfg.extra["mock_latency_s"] = 1.5

    app = PlayerApp(cfg, lambda c: MockEngine(c), window_scale=1.0,
                    headless=False, record_dir="interactive/recordings")
    rc = app.run(max_seconds=seconds, script=[(0.2, "hold:w")])

    disp_want = app.display_fps()
    gen = app.ring.throughput_fps
    print(f"[smoke] slow-content: display_measured={app.display_fps_measured:.2f} "
          f"ticks={app.display_ticks} content_measured={app.content_fps_measured:.2f} "
          f"content_servo={app._content_fps} gen={gen} "
          f"max_tick_gap={app._max_tick_gap * 1000:.0f} ms "
          f"(during rebuild {app._max_tick_gap_busy * 1000:.0f} ms) "
          f"starved_ticks={app._content_starved_ticks} "
          f"buffer_end={len(app.ring)}")

    if rc != 0 or app.worker.error:
        bad.append(f"slow-content run failed (rc={rc}, err={app.worker.error})")
    # 1. The display held its rate despite a ~8 fps engine.
    if app.display_fps_measured < disp_want * 0.9:
        bad.append(f"display fell to {app.display_fps_measured:.2f} fps with a "
                   f"slow engine (want ~{disp_want:.0f})")
    if app._max_tick_gap > 2.5 / disp_want:
        bad.append(f"display tick gap {app._max_tick_gap * 1000:.0f} ms with a "
                   "slow engine")
    # 2. Content really was slower -- i.e. held frames were being blitted,
    #    which is the whole point (otherwise this run proves nothing).
    if app.content_fps_measured > disp_want * 0.8:
        bad.append(f"content {app.content_fps_measured:.2f} fps was not slower "
                   f"than the display; the slow path was not exercised")
    # 3. Content converged toward what the mock can actually produce.
    if gen is None:
        bad.append("slow-content: throughput never measured")
    elif not (gen * 0.4 <= app.content_fps_measured <= gen * 1.35 + 1.0):
        bad.append(f"content {app.content_fps_measured:.2f} fps did not track "
                   f"generation {gen:.2f} fps")
    # 4. The ring never ran dry for long: holding is fine, starving is not.
    if app._content_starved_ticks > app.display_ticks * 0.4:
        bad.append(f"ring starved on {app._content_starved_ticks} of "
                   f"{app.display_ticks} ticks")

    if not bad:
        print(f"[smoke] slow-content: OK (display held "
              f"{app.display_fps_measured:.1f} fps while content ran "
              f"{app.content_fps_measured:.1f} fps)")
    return bad


def _test_manual_playback(cfg: EngineConfig, seconds: float = 3.0) -> List[str]:
    """`--playback <fps>` pins content AND display to that rate (no servo)."""
    from interactive.mock_engine import MockEngine
    bad: List[str] = []

    want = 10.0
    cfg = replace(cfg)
    cfg.extra = dict(cfg.extra)
    cfg.extra["playback_mode"] = "manual"
    cfg.extra["mock_latency_s"] = 0.05      # engine far faster than playback
    cfg.playback_fps = want
    cfg.denoising_steps = 1

    app = PlayerApp(cfg, lambda c: MockEngine(c), window_scale=1.0,
                    headless=False, record_dir="interactive/recordings")
    rc = app.run(max_seconds=seconds, script=[(0.2, "hold:w")])
    print(f"[smoke] manual: want={want} display_measured={app.display_fps_measured:.2f} "
          f"content_measured={app.content_fps_measured:.2f} "
          f"content_servo={app._content_fps} ticks={app.display_ticks}")
    if rc != 0 or app.worker.error:
        bad.append(f"manual run failed (rc={rc}, err={app.worker.error})")
    if not (want * 0.85 <= app.display_fps_measured <= want * 1.15):
        bad.append(f"manual display {app.display_fps_measured:.2f} != {want}")
    if not (want * 0.85 <= app.content_fps_measured <= want * 1.15):
        bad.append(f"manual content {app.content_fps_measured:.2f} != {want} "
                   "(manual must not servo)")
    if not bad:
        print(f"[smoke] manual: OK (content == display == {want:g} fps)")
    return bad


def _test_video_seed_math() -> List[str]:
    """Video->seed frame arithmetic and start-offset parsing (no ffmpeg, no GPU).

    Guards the two things most likely to silently rot: the Wan temporal
    packing (1 + 4*(L-1)) used to decide how many pixel frames a seed of N
    chunks needs, and the deliberately overloaded --seed_video_start spec.
    """
    from interactive.seed_picker import (
        CAMERA_AR, DEFAULT_CROP_BAND, DEFAULT_FISHEYE, SEED_ENCODE_FPS,
        _seed_filter_chain, crop_dims, is_video, parse_start_spec,
        seed_chunks_for_frames, seed_pixel_frames,
    )
    bad: List[str] = []

    def check(label, got, want):
        if got != want:
            bad.append(f"{label}: got {got!r}, want {want!r}")

    # 7 chunks = 21 latents = 1 + 4*20 = 81 pixel frames, the full attn span.
    check("7 chunks -> pixel frames", seed_pixel_frames(7), 81)
    check("1 chunk  -> pixel frames", seed_pixel_frames(1), 9)
    for n in (1, 2, 3, 4, 7):
        check(f"roundtrip {n}", seed_chunks_for_frames(seed_pixel_frames(n)), n)
    # A short clip yields whole chunks only, and never more than it can fill.
    check("50 frames -> 4 chunks", seed_chunks_for_frames(50), 4)
    check("8 frames -> 0 chunks", seed_chunks_for_frames(8), 0)

    # The encode rate is the RIDES' native 20 fps, not our 16 fps render rate.
    check("seed encode fps", SEED_ENCODE_FPS, 20.0)

    # Start spec: bare int = frames, float / 's' = seconds, 'f' = frames.
    check("start 0", parse_start_spec(0), 0.0)
    check("start 40 (frames)", parse_start_spec(40), 40.0 / SEED_ENCODE_FPS)
    check("start '40' (frames)", parse_start_spec("40"), 40.0 / SEED_ENCODE_FPS)
    check("start 3.0 (seconds)", parse_start_spec(3.0), 3.0)
    check("start '3s'", parse_start_spec("3s"), 3.0)
    check("start '90f'", parse_start_spec("90f"), 90.0 / SEED_ENCODE_FPS)
    # The two spellings of the same instant must agree (shared cache entry).
    check("40 frames == 2.0 s", parse_start_spec(40), parse_start_spec("2.0s"))

    # Filter chain: the training transform, in the right order.
    check("fisheye is OFF by default", DEFAULT_FISHEYE, 0.0)
    chain = _seed_filter_chain()                      # the DEFAULT chain
    check("default chain has no warp", "lenscorrection" in chain, False)
    order = [chain.index(k) for k in ("fps=", "crop=", "scale=")]
    if order != sorted(order):
        bad.append(f"filter stages out of order: {chain}")
    check("squash is the training resize", "scale=832:480" in chain, True)
    if "min(iw,ih*1.777778)" not in chain:
        bad.append(f"crop is not to the camera AR: {chain}")
    # Opting in must insert the warp BETWEEN the crop and the squash.
    warped = _seed_filter_chain(fisheye=0.18)
    if not (warped.index("crop=") < warped.index("lenscorrection")
            < warped.index("scale=")):
        bad.append(f"opt-in warp is not between crop and squash: {warped}")
    # The vertical band defaults to the BOTTOM: the training camera is
    # low-mounted, so ground-ahead beats horizon/sky for portrait sources.
    check("default crop band", DEFAULT_CROP_BAND, "bottom")
    if ":y='ih-out_h'" not in _seed_filter_chain():
        bad.append(f"default chain does not take the bottom band: {chain}")

    # The band is VERTICAL-ONLY: every mode must keep the same x-offset and
    # differ only in y, so an ultrawide source is never cropped off-centre.
    def _xy(c):
        ch = _seed_filter_chain(crop=c)
        return (ch.split(":x=")[1].split(":y=")[0], ch.split(":y=")[1].split(",")[0])
    xs = {_xy(c)[0] for c in ("center", "top", "bottom")}
    ys = {_xy(c)[1] for c in ("center", "top", "bottom")}
    check("horizontal offset is band-independent", len(xs), 1)
    check("horizontal offset is centred", xs.pop(), "'(iw-out_w)/2'")
    check("each band has its own y", len(ys), 3)
    check("camera AR is 16:9", round(CAMERA_AR, 4), 1.7778)

    # The crop must be MAXIMAL: one dimension always survives in full, and
    # the result always carries the camera's aspect.
    for iw, ih in [(1920, 1080), (1080, 1920), (640, 360), (832, 480),
                   (854, 640), (2560, 1080), (1440, 1080), (3840, 2160)]:
        w, h = crop_dims(iw, ih)
        if not (w >= iw - 1 or h >= ih - 1):
            bad.append(f"crop {iw}x{ih} -> {w}x{h} keeps neither dimension whole")
        if w > iw or h > ih:
            bad.append(f"crop {iw}x{ih} -> {w}x{h} exceeds the source")
        if abs((w / max(h, 1)) - CAMERA_AR) > 0.01:
            bad.append(f"crop {iw}x{ih} -> {w}x{h} is not at the camera AR")
    check("ultrawide keeps full height", crop_dims(2560, 1080), (1920, 1080))
    check("portrait keeps full width", crop_dims(1080, 1920), (1080, 607))
    check("already 16:9 is untouched", crop_dims(1920, 1080), (1920, 1080))

    check("mp4 is video", is_video("/x/y.MP4"), True)
    check("webm is video", is_video("a.webm"), True)
    check("zarr is not video", is_video("/rides/20240101.zarr"), False)

    if not bad:
        print("[smoke] video-seed math: OK (7 chunks = 81 frames; short clips "
              "round down to whole chunks; start spec frames/seconds; "
              "20 fps encode rate; maximal crop, bottom band, vertical-only; "
              "no warp by default)")
    return bad


def run_smoke(cfg: EngineConfig, seconds: float = 5.0) -> int:
    """Scripted, display-free run against MockEngine. Returns a process code."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    ctl_problems = (_test_control_scheme() + _test_combo_merge()
                    + _test_presets() + _test_video_seed_math())

    # Pin the fields the panel script edits to an explicit starting value.
    # They used to ride on the argparse defaults, so changing a default (e.g.
    # denoising_steps 4 -> 1) silently broke the assertions -- at 1 the "left"
    # keypress clamps and the value never moves.
    cfg = replace(cfg)
    SMOKE_START_STEPS, SMOKE_START_CHUNKS = 4, 6
    cfg.denoising_steps = SMOKE_START_STEPS
    cfg.horizon_chunks = SMOKE_START_CHUNKS
    for p in ctl_problems:
        print(f"[smoke] FAIL (controls): {p}")

    headless = False
    try:
        import pygame  # noqa: F401
    except Exception as exc:
        print(f"[smoke] pygame unavailable ({exc}); using headless fallback "
              "renderer (threading/buffer/HUD-text logic still exercised).")
        headless = True

    from interactive.mock_engine import MockEngine
    app = PlayerApp(cfg, lambda c: MockEngine(c), window_scale=1.0,
                    headless=headless,
                    record_dir="interactive/recordings")
    # Panel fields are addressed BY NAME (sel:<field>), so inserting a new
    # setting cannot silently repoint these keystrokes at a different field.
    script = [
        (0.20, "hold:w"),               # digital: instant full throttle
        # settings panel: live edits ...
        (0.60, "tab"),
        (0.65, "sel:denoising_steps"), (0.70, "left"),   # 4 -> 3   (live)
        (0.75, "sel:horizon_chunks"),  (0.80, "left"),   # 6 -> 5   (live)
        # ... then a restart-required edit applied with ENTER (rebuild path).
        (0.90, "sel:precision"), (0.95, "right"),  # bf16 -> fp8_wo (pending)
        (1.05, "return"),               # apply -> engine rebuild
        (1.20, "tab"),
        (1.30, "hold:w,a"),             # combo -> (+1, +1)
        (2.00, "r"),                    # reset ride
        (2.60, "hold:none"),            # released, latching -> action persists
        (3.60, "hold:stop"),            # SPACE -> physical no-op
        (4.00, "hold:d"),               # -> steer -1
        # preset hotkeys: quality -> balance -> speed (all share taew2_1, so
        # these are live-only, no rebuild)
        (4.20, "3"), (4.40, "2"), (4.60, "1"),
    ]
    rc = app.run(max_seconds=seconds, script=script)

    shown = app.frames_shown
    pushed = app.ring.total_pushed
    print(f"[smoke] frames pushed={pushed} content_shown={shown} "
          f"display_ticks={app.display_ticks} "
          f"horizons={app.worker.horizons_done} "
          f"buffer_end={len(app.ring)} state={app.worker.state} "
          f"actual_fps={app.actual_fps:.1f} headless={headless}")
    st = app.worker.last_stats
    if st is not None:
        print(f"[smoke] last stats: gen_fps={st.gen_fps:.1f} "
              f"first_frame={st.first_frame_latency_s * 1000:.0f} ms "
              f"steps={st.denoising_steps} prec={st.precision} "
              f"vram={st.peak_vram_gb:.1f} GB")
    for line in app.hud_lines():
        print(f"[smoke] hud | {line}")
    wc = app.worker.cfg
    print(f"[smoke] engine cfg after panel edits: denoising_steps="
          f"{wc.denoising_steps} horizon_chunks={wc.horizon_chunks} "
          f"decode_half_res={wc.decode_half_res} precision={wc.precision}")

    # --- presentation: fixed display rate, adaptive content rate -----------
    gen_thr = app.ring.throughput_fps
    disp_want = app.display_fps()
    print(f"[smoke] pacing: mode={app.playback_mode} "
          f"display_want={disp_want:.1f} display_measured={app.display_fps_measured:.2f} "
          f"ticks={app.display_ticks} content_servo={app._content_fps} "
          f"content_measured={app.content_fps_measured:.2f} measured_gen={gen_thr} "
          f"max_tick_gap={app._max_tick_gap * 1000:.0f} ms "
          f"(during rebuild {app._max_tick_gap_busy * 1000:.0f} ms) "
          f"max_content_gap={app._max_frame_gap * 1000:.0f} ms "
          f"starved_ticks={app._content_starved_ticks}")

    problems = list(ctl_problems)
    # The display rate is a hard promise now: it ticks at cfg.playback_fps
    # regardless of what the engine is doing.
    if app.display_ticks < int(seconds * disp_want * 0.85):
        problems.append(f"display ticked {app.display_ticks} times in {seconds}s; "
                        f"expected ~{int(seconds * disp_want)} at {disp_want:.0f} Hz")
    if not (disp_want * 0.85 <= app.display_fps_measured <= disp_want * 1.15):
        problems.append(f"measured display rate {app.display_fps_measured:.2f} fps "
                        f"is not ~{disp_want:.0f} fps")
    if app._max_tick_gap > 2.5 / disp_want:
        problems.append(f"display tick gap {app._max_tick_gap * 1000:.0f} ms "
                        f"(> 2.5 display periods)")
    if app.playback_mode == "auto":
        if gen_thr is None:
            problems.append("throughput was never measured at the ring")
        elif app._content_fps is None:
            problems.append("content fps never resolved")
        else:
            # Content never runs faster than the display, and never faster
            # than the generator can sustain.
            if app._content_fps > disp_want + 1e-6:
                problems.append(f"content {app._content_fps:.2f} fps exceeds "
                                f"display {disp_want:.2f} fps")
            if gen_thr < disp_want * 0.9:
                # Generation is the bottleneck: content should have converged
                # near it rather than staying pinned at the display rate.
                if app._content_fps > gen_thr * 1.6 + 1e-6:
                    problems.append(
                        f"content {app._content_fps:.2f} fps did not converge "
                        f"toward generation {gen_thr:.2f} fps")
    # The final scripted state is hold:d -> right, neutral throttle.
    if (app.throttle, app.steer) != (DEFAULT_NOOP_THROTTLE,
                                     DEFAULT_RIGHT_STEER):
        problems.append(f"scripted digital controls did not reach the app "
                        f"(got {app.throttle:+.2f},{app.steer:+.2f}; want "
                        f"{DEFAULT_NOOP_THROTTLE:+.2f},"
                        f"{DEFAULT_RIGHT_STEER:+.2f})")
    if app.worker.error:
        problems.append(f"worker error: {app.worker.error}")
    if rc != 0:
        problems.append(f"app returned {rc}")
    if pushed < 100:
        problems.append(f"too few frames produced ({pushed})")
    # In auto mode CONTENT deliberately paces to generation, so the expected
    # distinct-frame count is bounded by throughput, not by the display rate.
    exp_fps = disp_want
    if app.playback_mode == "auto" and gen_thr is not None:
        exp_fps = min(exp_fps, max(gen_thr, PlayerApp.MIN_CONTENT_FPS))
    # This five-second script deliberately clears stale queued content on
    # action changes, reset and rebuild.  Distinct frames can consequently
    # land just below half the theoretical uninterrupted count while all
    # presentation and generation checks pass.
    min_content_fraction = 0.45
    if shown < int(seconds * exp_fps * min_content_fraction):
        problems.append(f"too few content frames advanced ({shown}; expected "
                        f">= {int(seconds * exp_fps * min_content_fraction)} "
                        f"at {exp_fps:.1f} fps)")
    if app.worker.horizons_done < 1:
        problems.append("no horizon completed")
    if st is None:
        problems.append("HorizonStats never populated")
    # horizon_chunks proves the panel's live edits reached the engine. Steps
    # is NOT checked here: the preset hotkeys later in the script deliberately
    # overwrite it, and that is asserted separately below.
    if wc.horizon_chunks != SMOKE_START_CHUNKS - 1:
        problems.append("live settings did not reach the engine "
                        f"(chunks={wc.horizon_chunks}, want {SMOKE_START_CHUNKS - 1})")
    # The script ends on preset hotkey "1" (speed), which applies live.
    want_steps = PRESETS["speed"]["denoising_steps"]
    if wc.denoising_steps != want_steps:
        problems.append(f"preset hotkey did not reach the engine "
                        f"(steps={wc.denoising_steps}, want {want_steps})")
    if active_preset(app.cfg) != "speed":
        problems.append(f"app preset is {active_preset(app.cfg)!r}, want 'speed'")
    if wc.precision != "fp8_wo":
        problems.append(f"rebuild did not apply precision (got {wc.precision})")
    if st is not None and st.precision != "fp8_wo":
        problems.append("no horizon completed after the rebuild "
                        f"(last stats precision={st.precision})")

    # Second phase: a deliberately slow engine must NOT slow the window.
    problems += _test_slow_content(cfg)
    # Third phase: the manual (fixed-rate) opt-out still behaves as before.
    problems += _test_manual_playback(cfg)

    if problems:
        for p in problems:
            print(f"[smoke] FAIL: {p}")
        return 1
    print("[smoke] PASS")
    return 0


# ==========================================================================
# CLI
# ==========================================================================
def build_cfg(a: argparse.Namespace) -> EngineConfig:
    cfg = EngineConfig(
        ckpt_path=a.ckpt or "",
        config_path=a.config,
        wan_model_path=a.wan_model_path,
        # A video seed flows through the same field: everything downstream
        # dispatches on the path's suffix.
        seed_zarr=(a.seed_video or a.seed_zarr or ""),
        # Production-eval convention: RAW `generator` weights. This used to
        # read `not a.no_ema`, which with a store_true --no_ema meant every
        # plain launch resolved use_ema=True -- silently loading the EMA
        # overlay and disqualifying the lean checkpoint.
        use_ema=bool(a.use_ema),
        precision=a.precision,
        compile_mode=a.compile_mode,
        denoising_steps=a.denoising_steps,
        kv_cache_chunks=a.kv_cache_chunks,
        seed_prefill_chunks=seed_chunks_from_seconds(a.seed_seconds),
        horizon_chunks=a.horizon_chunks,
        block_chunks=a.block_chunks,
        decode_half_res=a.decode_half_res,
        playback_fps=a.playback_fps,
        max_buffered_horizons=a.max_buffered_horizons,
        device=a.device,
        seed=a.seed,
    )
    cfg.extra["mock_latency_s"] = a.mock_latency
    cfg.extra["controls"] = a.controls
    cfg.extra["latch"] = (a.latch == "on")
    # Constructor validation keeps CLI and in-app controller contracts equal.
    ControlScheme(
        forward_throttle=a.forward_throttle,
        reverse_throttle=a.reverse_throttle,
        left_steer=a.left_steer,
        right_steer=a.right_steer,
    )
    cfg.extra["forward_throttle"] = float(a.forward_throttle)
    cfg.extra["reverse_throttle"] = float(a.reverse_throttle)
    cfg.extra["left_steer"] = float(a.left_steer)
    cfg.extra["right_steer"] = float(a.right_steer)
    cfg.extra["action_latch"] = a.action_latch
    cfg.extra["drop_stale_on_action"] = (a.drop_stale == "on")
    cfg.extra["seed_root"] = a.seed_root
    cfg.extra["ladder_mode"] = a.ladder_mode
    cfg.extra["seed_video_start"] = a.seed_video_start
    cfg.extra["seed_video_start_s"] = a.seed_video_start_s
    cfg.extra["seed_video_crop"] = a.seed_video_crop
    if a.seed_video_fisheye is not None:
        cfg.extra["seed_video_fisheye"] = a.seed_video_fisheye
    cfg.extra["video_dir"] = a.video_dir
    cfg.extra["seed_frame_index"] = int(a.seed_frame_index)
    if a.prompt:
        cfg.extra["prompt"] = a.prompt
    if a.caption_root:
        cfg.extra["caption_root"] = a.caption_root
    if a.motion_root:
        cfg.extra["motion_root"] = a.motion_root
    if a.seed_actions:
        cfg.extra["seed_actions"] = a.seed_actions
    # Applied last so it overrides the individual flags it covers.
    if getattr(a, "preset", None):
        for f, v in PRESETS[a.preset].items():
            setattr(cfg, f, v)
        print(f"[play] preset {a.preset}: {PRESET_BLURB[a.preset]}")
    # --playback auto | <fps>. A number means fixed pacing at that rate for
    # BOTH content and display, so "--playback 12" behaves exactly as before
    # (a 12 Hz window). Auto keeps the window at --playback_fps and adapts
    # only the content-advance rate.
    pb = str(a.playback).strip().lower()
    if pb in ("auto", ""):
        cfg.extra["playback_mode"] = "auto"
    else:
        try:
            cfg.playback_fps = float(pb)
            cfg.extra["playback_mode"] = "manual"
        except ValueError:
            print(f"[play] bad --playback {a.playback!r}; using auto.")
            cfg.extra["playback_mode"] = "auto"
    return cfg


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Interactive world-model player")
    p.add_argument("--mock", action="store_true",
                   help="drive the GPU-free procedural MockEngine")
    p.add_argument("--smoke", action="store_true",
                   help="headless scripted 5 s self-test against MockEngine")
    p.add_argument("--smoke_seconds", type=float, default=5.0)

    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--config", type=str,
                   default="configs/action_forcing_phase3_dmd.yaml")
    p.add_argument("--wan_model_path", type=str,
                   default="/home/ashish/Wan2.1/Wan2.1-T2V-1.3B/")
    p.add_argument("--seed_zarr", type=str, default=None,
                   help="starting ride; omit to open the seed picker at startup")
    p.add_argument("--seed_video", type=str, default=None,
                   help="seed from a local video instead of a ride zarr: its "
                        "first 81 frames are centre-cropped to 832x480 and Wan-"
                        "encoded to 21 latents. Out of distribution — expect drift.")
    p.add_argument("--seed_video_start", type=str, default="0",
                   help="offset into --seed_video. A bare INTEGER means "
                        "FRAMES at the 20 fps encode rate (e.g. 40 = 2 s); "
                        "a bare float or an "
                        "'s' suffix means SECONDS (e.g. 2.5 or '2.5s'); an "
                        "'f' suffix forces frames ('90f'). 0 is 0 either way.")
    p.add_argument("--seed_video_start_s", type=float, default=None,
                   help="offset into --seed_video in SECONDS, unambiguously. "
                        "Overrides --seed_video_start when given.")
    p.add_argument("--seed_video_fisheye", type=float, default=None,
                   help="barrel-distortion pre-warp applied to your video "
                        "before the training squash, to mimic the robot's "
                        "fisheye lens. OFF by default (0.0); try 0.15-0.22 "
                        "to opt in. Real rides do show pronounced barrel, but "
                        "the warp costs field of view a phone cannot spare.")
    p.add_argument("--seed_video_crop", choices=["center", "top", "bottom"],
                   default="bottom",
                   help="which VERTICAL band of your frame survives the crop "
                        "to the camera's 16:9 aspect. Default 'bottom': the "
                        "training camera is low-mounted at ground level, so "
                        "the bottom of a phone frame (road ahead) matches it "
                        "far better than the middle or the sky. A no-op for "
                        "16:9 landscape; decisive for PORTRAIT. Horizontal "
                        "trims (ultrawide) always stay centred.")
    p.add_argument("--video_dir", type=str, default=None,
                   help="folder of your own videos to list in the seed picker. "
                        "Default: interactive/user_videos/ (created for you -- "
                        "the documented drop-folder) plus ~/Videos.")
    p.add_argument("--seed_frame_index", type=int, default=0,
                   help="latent-frame offset within --seed_zarr")
    p.add_argument("--seed-seconds", type=float,
                   default=seed_seconds_from_chunks(SEED_PREFILL_CHUNKS),
                   help="visible real-seed duration; rounded to streaming "
                        "decoder chunks (0.75 s increments after the first) "
                        "and adjustable from the TAB menu")
    p.add_argument("--seed_root", type=str, default=None,
                   help="directory of ride zarrs for the seed picker "
                        "(default: frodobots_encoded, falling back to smoke_zarr)")
    p.add_argument("--prompt", type=str, default=None,
                   help="explicit text prompt; overrides the seed ride's exact "
                        "training caption")
    p.add_argument("--caption_root", type=str, default=None,
                   help="caption tree override; default auto-detects the local "
                        "frodobots training captions")
    p.add_argument("--motion_root", type=str, default=None,
                   help="motion tree override; default auto-detects the local "
                        "frodobots motion conditioning")
    p.add_argument("--seed_actions", type=str, default=None,
                   help="optional pre-squashed seed action tensor override")
    p.add_argument("--use_ema", action="store_true",
                   help="overlay the generator_ema shadow on top of the raw "
                        "weights. Off by default (production-eval convention) "
                        "and incompatible with the lean checkpoint.")
    p.add_argument("--no_ema", action="store_true",
                   help="deprecated no-op: raw weights are already the default")

    # -- controls ---------------------------------------------------------
    p.add_argument("--controls", choices=["digital", "analog"], default="digital",
                   help="digital: a key means full deflection instantly, the "
                        f"latest input wins per axis, and two presses within "
                        f"{COMBO_MERGE_WINDOW_S:g}s on different axes merge "
                        "into the diagonal. analog: the original ramp/decay "
                        "integrator (no merging).")
    p.add_argument("--forward-throttle", type=float,
                   default=DEFAULT_FORWARD_THROTTLE,
                   help="calibrated positive W/up endpoint")
    p.add_argument("--reverse-throttle", type=float,
                   default=DEFAULT_REVERSE_THROTTLE,
                   help="negative endpoint for S/down and analog reverse; "
                        "-0.3 is the measured training-support boundary "
                        "(-1.0 reproduces the old folded/forward response)")
    p.add_argument("--left-steer", type=float, default=DEFAULT_LEFT_STEER,
                   help="calibrated negative A/left endpoint")
    p.add_argument("--right-steer", type=float, default=DEFAULT_RIGHT_STEER,
                   help="calibrated positive D/right endpoint")
    p.add_argument("--latch", choices=["on", "off"], default="on",
                   help="on: the action persists after key release until a new "
                        "valid input or SPACE. off: release returns to the "
                        "paper-derived physical no-op.")
    p.add_argument("--action_latch", choices=["chunk", "horizon"], default="chunk",
                   help="how often the engine re-reads the action. 'horizon' is "
                        "the old behaviour and can ignore input for seconds.")
    p.add_argument("--drop_stale", choices=["on", "off"], default="on",
                   help="on an action change, drop already-buffered frames that "
                        "were generated under the previous action.")

    p.add_argument("--precision", type=str, default="bf16", choices=PRECISION_CHOICES)
    p.add_argument("--compile_mode", type=str, default="off", choices=COMPILE_CHOICES,
                   help="max-autotune-no-cudagraphs autotunes kernels without "
                        "graph capture (the capturing modes are incompatible "
                        "with the persistent KV cache)")
    p.add_argument("--ladder_mode", type=str, default="interp", choices=LADDER_CHOICES,
                   help="interp: keep the 4 trained rungs and subdivide the gaps "
                        "(default, supports >4 steps in-distribution). "
                        "trained: top-N prefix, sigma grid past 4. grid: sigma grid.")
    p.add_argument("--preset", type=str, default=None, choices=PRESET_ORDER,
                   help="quality preset: speed (1 step, ~16 fps real-time), "
                        "balance (2 steps, ~11 fps), quality (4 steps, ~7 fps). "
                        "Applied after the individual flags, so it wins over "
                        "--denoising_steps/--decoder. Switch live with 1/2/3.")
    p.add_argument("--denoising_steps", type=int,
                   default=EngineConfig.denoising_steps,
                   help="default 4 traverses every trained timestep rung")
    p.add_argument("--kv_cache_chunks", type=int, default=7)
    p.add_argument("--horizon_chunks", type=int, default=6)
    p.add_argument("--block_chunks", type=int, default=EngineConfig.block_chunks,
                   choices=BLOCK_CHOICES,
                   help="fully-joint research block width; production default 1")
    p.add_argument("--decode_half_res", action="store_true")
    p.add_argument("--playback_fps", type=float, default=16.0,
                   help="the DISPLAY refresh rate (the window always ticks at "
                        "this rate) and, in auto mode, the cap on the content-"
                        "advance rate")
    p.add_argument("--playback", type=str, default="auto",
                   help="'auto' (default): the window refreshes at "
                        "--playback_fps while CONTENT advances at the measured "
                        "generation throughput (held frames when generation is "
                        "slower), so motion never stalls and the fps never "
                        "drops. Or a number: fixed content == display rate, "
                        "the pre-adaptive behaviour.")
    # Default 1, not 2: every buffered horizon is video the user must watch
    # before a new action can possibly show up. At 6 chunks / 16 fps one
    # horizon is already ~4.5 s of queued footage.
    p.add_argument("--max_buffered_horizons", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--web", action="store_true",
                   help="serve the player in a BROWSER instead of a pygame "
                        "window. Required over VS Code Remote-SSH, where "
                        "pygame has no display. Prints a localhost URL that "
                        "VS Code forwards automatically.")
    p.add_argument("--port", type=int, default=8765,
                   help="port for --web (bound to 127.0.0.1 only)")
    p.add_argument("--web_quality", type=int, default=80,
                   help="JPEG quality for the --web frame stream (1-100)")
    p.add_argument("--window_scale", type=float, default=1.0)
    p.add_argument("--record_dir", type=str, default="interactive/recordings")
    p.add_argument("--mock_latency", type=float, default=0.30,
                   help="simulated per-chunk latency for MockEngine (seconds)")
    a = p.parse_args(argv)

    cfg = build_cfg(a)

    if a.smoke:
        return run_smoke(cfg, seconds=a.smoke_seconds)

    if not a.mock and not a.ckpt:
        print("No --ckpt given. Either pass --ckpt/--seed_zarr for the real "
              "engine, or run with --mock.")
        return 2

    factory = make_engine_factory(use_mock=a.mock)

    # Web mode needs no pygame at all: the browser is the surface, and the
    # startup picker is an HTML overlay (press P) rather than a pygame modal.
    headless = bool(a.web)
    try:
        if a.web:
            raise ImportError("web mode")
        import pygame  # noqa: F401
    except Exception as exc:
        if not a.web:
            print(f"pygame is not available in this interpreter ({exc}).")
            print("Falling back to the headless renderer: frames are generated "
                  "and drained but nothing is displayed and keys are not read.")
            print("For a real window use an interpreter that has pygame "
                  "installed (do NOT install into the `flash` env) -- or run "
                  "with --web and drive it from your browser.")
        headless = True

    # No starting ride given: choose one before the engine is built, so the
    # first thing built is already seeded correctly.
    if not cfg.seed_zarr and not a.mock:
        if a.seed == -1 or str(a.seed_zarr or "").lower() == "random":
            from interactive.seed_picker import pick_random
            ride = pick_random(a.seed_root)
            cfg.seed_zarr = ride.path
            print(f"[play] random seed ride: {ride.ride_id}")
        elif headless:
            from interactive.seed_picker import pick_random
            ride = pick_random(a.seed_root)
            cfg.seed_zarr = ride.path
            if a.web:
                print(f"[play] no --seed_zarr: starting on {ride.ride_id}. "
                      "Press P in the browser to pick another ride or one of "
                      "your own videos.")
            else:
                print(f"[play] headless, no --seed_zarr: picked {ride.ride_id}")
        else:
            # Build the engine WHILE the user browses. The build needs no seed
            # (the seed is only read in reset()), so the two are independent
            # and the ~17 s build overlaps thinking time instead of following
            # it. The build owns the GPU; the picker serves disk-cached
            # thumbnails and defers fresh decodes until it finishes.
            chosen, prebuilt = _startup_seed_picker(
                a.seed_root, a.window_scale, cfg=cfg, factory=factory,
                video_dir=a.video_dir)
            if not chosen:
                print("[play] no ride chosen; nothing to do.")
                if prebuilt is not None:
                    try:
                        prebuilt.close()
                    except Exception:
                        pass
                return 0
            cfg.seed_zarr = chosen
            if prebuilt is not None:
                # Hand the already-built engine to the worker instead of
                # building a second one.
                prebuilt.cfg.seed_zarr = chosen
                factory = _prebuilt_factory(prebuilt)
                # Drop main()'s reference NOW. It would otherwise stay alive
                # for the whole of app.run(), pinning the first engine on the
                # GPU so a later rebuild could never free it.
                del prebuilt

    if a.web:
        from interactive.web_play import run_web
        return run_web(cfg, factory, port=a.port, record_dir=a.record_dir,
                       quality=a.web_quality)

    app = PlayerApp(cfg, factory, window_scale=a.window_scale,
                    headless=headless, record_dir=a.record_dir)
    return app.run()


def _prebuilt_factory(engine):
    """Return a factory that hands over ``engine`` once, then builds fresh.

    A later rebuild (precision/compile change) must not resurrect the
    pre-built engine, so the reference is dropped after the first call.
    """
    box = {"engine": engine}

    def factory(cfg):
        eng = box.pop("engine", None)
        if eng is not None:
            eng.cfg.seed_zarr = cfg.seed_zarr
            return eng
        from interactive.engine import WorldModelEngine
        return WorldModelEngine(cfg)

    return factory


def _startup_seed_picker(seed_root: Optional[str], window_scale: float,
                         cfg: Optional[EngineConfig] = None,
                         factory=None,
                         video_dir: Optional[str] = None) -> Tuple[Optional[str], object]:
    """Pick a starting ride, optionally building the engine in parallel.

    Returns ``(chosen_path, prebuilt_engine_or_None)``.
    """
    import pygame as pg
    from interactive.seed_picker import (
        SeedPicker, ThumbnailCache, list_rides, list_videos)
    pg.init()
    pg.display.set_caption("ARRWM - choose a starting ride")
    screen = pg.display.set_mode((int(1180 * window_scale), int(760 * window_scale)),
                                 pg.RESIZABLE)
    pg.font.init()

    gate = threading.Event()
    built: dict = {}

    def _build() -> None:
        try:
            t0 = time.perf_counter()
            # Build with NO seed: reset() is what reads one, and the ride is
            # applied afterwards via reset_with_seed.
            bcfg = replace(cfg)
            bcfg.seed_zarr = ""
            built["engine"] = factory(bcfg)
            built["seconds"] = time.perf_counter() - t0
            print(f"[play] engine pre-built in {built['seconds']:.1f}s "
                  f"while you were choosing")
        except Exception as exc:      # never let this kill the picker
            import traceback
            traceback.print_exc()
            built["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            gate.set()

    thread = None
    if cfg is not None and factory is not None:
        thread = threading.Thread(target=_build, name="engine-prebuild",
                                  daemon=True)
        thread.start()
    else:
        gate.set()

    cache = ThumbnailCache(gpu_gate=gate)
    try:
        entries = list_videos(video_dir) + list_rides(seed_root)
        picker = SeedPicker(screen, entries, cache=cache,
                            font=pg.font.SysFont("monospace", 20),
                            small_font=pg.font.SysFont("monospace", 15))
        chosen = picker.run()
    finally:
        cache.close()

    if thread is not None:
        if not gate.is_set():
            print("[play] waiting for the engine build to finish...")
            thread.join()
        else:
            thread.join(timeout=1.0)
    if built.get("error"):
        print(f"[play] pre-build failed ({built['error']}); building normally.")
    return chosen, built.get("engine")


if __name__ == "__main__":
    raise SystemExit(main())
