"""Measure the action-to-screen latency chain in the interactive player.

Three delays stack up between a keypress and a visibly different frame:

  1. PICKUP   how long before the engine even consults the new action.
              action_latch='horizon' consults it once per horizon;
              'chunk' consults it at every chunk boundary.
  2. GENERATE the chunk conditioned on the new action has to be produced.
  3. BUFFER   every frame already queued in the ring must be played first,
              at playback_fps, before the new one is on screen.

This harness measures (1) and (3) directly against the MockEngine -- whose
per-chunk latency is configurable, so the arithmetic matches the real engine
without occupying the GPU -- and reports the total budget.

    python -m interactive.latency_probe --chunk_latency 1.7 --horizon_chunks 6
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from interactive.engine_api import Action, EngineConfig, PIXEL_FRAMES_PER_CHUNK  # noqa: E402
from interactive.mock_engine import MockEngine  # noqa: E402
from interactive.play import EngineWorker, FrameRing  # noqa: E402


def measure(latch: str, chunk_latency: float, horizon_chunks: int,
            max_buffered: int, playback_fps: float,
            settle_s: float = 6.0, ckpt: str = "", seed_zarr: str = "",
            denoising_steps: int = 4) -> dict:
    """Measure one configuration.

    With ``ckpt`` set this drives the REAL engine (the only one that
    implements ``set_action_provider``, so it is the only one that can show
    chunk-level pickup); otherwise it uses MockEngine for the buffer maths.
    """
    real = bool(ckpt)
    cfg = EngineConfig(
        ckpt_path=ckpt, seed_zarr=seed_zarr, horizon_chunks=horizon_chunks,
        max_buffered_horizons=max_buffered, playback_fps=playback_fps,
        denoising_steps=denoising_steps,
    )
    cfg.extra["mock_latency_s"] = chunk_latency
    cfg.extra["action_latch"] = latch

    ring = FrameRing()
    if real:
        from interactive.engine import WorldModelEngine
        worker = EngineWorker(cfg, lambda c: WorldModelEngine(c), ring)
        settle_s = max(settle_s, 180.0)
    else:
        worker = EngineWorker(cfg, lambda c: MockEngine(c), ring)

    # Timestamp every mailbox read. Wrapping the worker's own accessor
    # catches BOTH paths fairly: the horizon-start read in EngineWorker.run
    # and the per-chunk read the engine makes through the provider.
    picked: List[tuple] = []
    orig_latched = worker.latched_action

    def timed_latched() -> Action:
        a = orig_latched()
        picked.append((time.perf_counter(), a.throttle, a.steer))
        return a

    worker.latched_action = timed_latched  # type: ignore[assignment]

    # Drain the ring at playback_fps, exactly as the UI does. Without this the
    # worker fills the buffer, parks under backpressure and stops reading the
    # action mailbox at all -- which looks like "infinite pickup latency" but
    # is just a missing consumer.
    stop_drain = threading.Event()
    drained = {"n": 0}

    def drain() -> None:
        period = 1.0 / max(playback_fps, 1e-3)
        nxt = time.perf_counter()
        while not stop_drain.is_set():
            now = time.perf_counter()
            if now >= nxt:
                if ring.pop() is not None:
                    drained["n"] += 1
                nxt += period
                if now - nxt > 0.5:
                    nxt = now + period
            time.sleep(0.002)

    drainer = threading.Thread(target=drain, name="drain", daemon=True)

    worker.start()
    drainer.start()
    t_end = time.perf_counter() + settle_s
    while time.perf_counter() < t_end and worker.horizons_done < 1:
        time.sleep(0.02)

    # Let it settle into steady state, then change the action.
    time.sleep(min(3.0, chunk_latency * 2))
    buffered_frames = len(ring)
    t_change = time.perf_counter()
    worker.set_action(1.0, 1.0)

    # Wait for the engine to consult the NEW action.
    pickup = float("nan")
    deadline = time.perf_counter() + horizon_chunks * chunk_latency * 3 + 10
    while time.perf_counter() < deadline:
        hit = [t for (t, thr, ste) in picked if t >= t_change and thr == 1.0]
        if hit:
            pickup = hit[0] - t_change
            break
        time.sleep(0.01)

    stop_drain.set()
    drainer.join(timeout=5)
    worker.request_stop()
    worker.join(timeout=60)

    buffer_s = buffered_frames / max(playback_fps, 1e-3)
    gen_s = chunk_latency
    return {
        "latch": latch,
        "pickup_s": pickup,
        "buffered_frames": buffered_frames,
        "buffer_s": buffer_s,
        "generate_s": gen_s,
        "total_s": (0.0 if pickup != pickup else pickup) + gen_s + buffer_s,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunk_latency", type=float, default=1.7,
                    help="seconds per chunk (real engine at 4 steps ~1.7 s)")
    ap.add_argument("--horizon_chunks", type=int, default=6)
    ap.add_argument("--playback_fps", type=float, default=16.0)
    ap.add_argument("--ckpt", default="", help="drive the REAL engine")
    ap.add_argument("--seed_zarr", default=str(Path.home() / "20240224003808.zarr"))
    ap.add_argument("--denoising_steps", type=int, default=4)
    a = ap.parse_args(argv)

    print(f"[lat] chunk_latency={a.chunk_latency}s horizon={a.horizon_chunks} chunks "
          f"({a.chunk_latency * a.horizon_chunks:.1f}s of generation per horizon)")
    print(f"[lat] one chunk = {PIXEL_FRAMES_PER_CHUNK} frames = "
          f"{PIXEL_FRAMES_PER_CHUNK / a.playback_fps:.2f}s of video\n")

    kw = dict(ckpt=a.ckpt, seed_zarr=a.seed_zarr,
              denoising_steps=a.denoising_steps)
    rows = []
    # BEFORE: horizon latching, 2 buffered horizons (the old defaults).
    rows.append(measure("horizon", a.chunk_latency, a.horizon_chunks,
                        max_buffered=2, playback_fps=a.playback_fps, **kw))
    # AFTER: chunk latching, 1 buffered horizon (the new defaults).
    rows.append(measure("chunk", a.chunk_latency, a.horizon_chunks,
                        max_buffered=1, playback_fps=a.playback_fps, **kw))

    print(f"{'config':>24}  {'pickup':>8}  {'generate':>9}  {'buffer':>8}  {'TOTAL':>8}")
    for r in rows:
        label = ("BEFORE horizon-latch/buf2" if r["latch"] == "horizon"
                 else "AFTER  chunk-latch/buf1")
        print(f"{label:>24}  {r['pickup_s']:7.2f}s  {r['generate_s']:8.2f}s  "
              f"{r['buffer_s']:7.2f}s  {r['total_s']:7.2f}s")
    if len(rows) == 2 and rows[0]["total_s"] > 0:
        print(f"\n[lat] action-to-screen improved "
              f"{rows[0]['total_s']:.2f}s -> {rows[1]['total_s']:.2f}s "
              f"({rows[0]['total_s'] / max(rows[1]['total_s'], 1e-6):.1f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
