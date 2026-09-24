"""GPU-free stand-in engine for the interactive world-model player.

`MockEngine` implements the full `EngineBase` contract from
`interactive/engine_api.py` with a cheap procedural renderer, so `play.py`
(threading, ring buffer, backpressure, HUD, settings panel, recording) can be
developed and smoke-tested with no checkpoint, no CUDA and no Wan weights.

The rendering deliberately *responds to the action*:

  * `steer`    -> horizontal shift of the scrolling road/gradient
  * `throttle` -> scroll speed (and a warm tint at high throttle)

Each frame carries a small bitmap-font overlay with the horizon index, the
chunk index within the horizon and the latched action, so a viewer can tell at
a glance that new content is arriving and that actions are being latched.

Per-chunk latency is simulated (default 300 ms, override with
``cfg.extra["mock_latency_s"]``) so the UI's buffering/backpressure behaviour
matches something realistic.
"""

from __future__ import annotations

import time
from typing import Dict, Iterator, List, Optional

import numpy as np

from interactive.engine_api import (
    PIXEL_FRAMES_PER_CHUNK,
    PIXEL_H,
    PIXEL_W,
    SEED_PREFILL_CHUNKS,
    Action,
    EngineBase,
    EngineConfig,
    HorizonStats,
)

# --------------------------------------------------------------------------
# Tiny 3x5 bitmap font (digits + the handful of glyphs the overlay needs).
# --------------------------------------------------------------------------
_FONT: Dict[str, List[str]] = {
    "0": ["111", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "111"],
    "2": ["111", "001", "111", "100", "111"],
    "3": ["111", "001", "111", "001", "111"],
    "4": ["101", "101", "111", "001", "001"],
    "5": ["111", "100", "111", "001", "111"],
    "6": ["111", "100", "111", "101", "111"],
    "7": ["111", "001", "010", "010", "010"],
    "8": ["111", "101", "111", "101", "111"],
    "9": ["111", "101", "111", "001", "111"],
    "H": ["101", "101", "111", "101", "101"],
    "C": ["111", "100", "100", "100", "111"],
    "T": ["111", "010", "010", "010", "010"],
    "S": ["111", "100", "111", "001", "111"],
    "E": ["111", "100", "111", "100", "111"],
    "D": ["110", "101", "101", "101", "110"],
    "M": ["101", "111", "111", "101", "101"],
    "K": ["101", "110", "100", "110", "101"],
    "/": ["001", "001", "010", "100", "100"],
    ".": ["000", "000", "000", "000", "010"],
    "-": ["000", "000", "111", "000", "000"],
    "+": ["000", "010", "111", "010", "000"],
    ":": ["000", "010", "000", "010", "000"],
    " ": ["000", "000", "000", "000", "000"],
}


def _draw_text(img: np.ndarray, text: str, x: int, y: int, scale: int = 3,
               color=(255, 255, 255)) -> None:
    """Blit `text` into an [H, W, 3] uint8 image at (x, y), in place."""
    h, w = img.shape[:2]
    col = np.asarray(color, dtype=np.uint8)
    cx = x
    for ch in text.upper():
        glyph = _FONT.get(ch)
        if glyph is None:
            cx += 4 * scale
            continue
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit != "1":
                    continue
                px0, py0 = cx + gx * scale, y + gy * scale
                px1, py1 = min(px0 + scale, w), min(py0 + scale, h)
                if px0 >= w or py0 >= h or px1 <= 0 or py1 <= 0:
                    continue
                img[max(py0, 0):py1, max(px0, 0):px1] = col
        cx += 4 * scale


class MockEngine(EngineBase):
    """Procedural, CPU-only implementation of :class:`EngineBase`."""

    def __init__(self, cfg: EngineConfig):
        super().__init__(cfg)
        self.latency_s = float(cfg.extra.get("mock_latency_s", 0.30))
        self._rng = np.random.default_rng(cfg.seed)
        self._stats: Optional[HorizonStats] = None
        self._horizon_index = 0
        self._phase_x = 0.0
        self._phase_y = 0.0
        self._closed = False
        # Coarse compute grid; upsampled 4x to the pixel grid (cheap).
        self._ds = 4
        self._build_grid()

    # -- geometry ---------------------------------------------------------
    def _out_hw(self):
        if self.cfg.decode_half_res:
            return PIXEL_H // 2, PIXEL_W // 2
        return PIXEL_H, PIXEL_W

    def _build_grid(self):
        h, w = self._out_hw()
        gh, gw = max(h // self._ds, 1), max(w // self._ds, 1)
        self._gh, self._gw = gh, gw
        yy, xx = np.meshgrid(np.arange(gh, dtype=np.float32),
                             np.arange(gw, dtype=np.float32), indexing="ij")
        self._xx, self._yy = xx, yy
        self._grid_for = (h, w)

    def _ensure_grid(self):
        if self._grid_for != self._out_hw():
            self._build_grid()

    # -- rendering --------------------------------------------------------
    def _render_frame(self, action: Action, label: str) -> np.ndarray:
        self._ensure_grid()
        h, w = self._out_hw()
        gh, gw = self._gh, self._gw
        xx, yy = self._xx, self._yy

        # Advance the scroll: throttle -> speed, steer -> lateral drift.
        speed = 0.6 + 1.8 * float(action.throttle)
        self._phase_y += speed * 0.55
        self._phase_x += float(action.steer) * 2.4

        u = xx * 0.085 - self._phase_x * 0.12
        v = yy * 0.10 + self._phase_y * 0.10

        horizon = gh * 0.42
        ground = (yy > horizon).astype(np.float32)

        # Perspective-ish road: lane bands converging at the horizon.
        depth = np.maximum(yy - horizon, 1e-3) / max(gh - horizon, 1e-3)
        lane = np.sin((xx - gw * 0.5 - float(action.steer) * gw * 0.35)
                      / (depth * gw * 0.55 + 1e-3) * 3.0
                      + self._phase_y * 0.35 * speed)

        base = 0.5 + 0.5 * np.sin(u) * np.cos(v * 0.7) + 0.25 * np.sin(u * 2.3 + v * 1.7)
        sky = 0.35 + 0.45 * np.sin(u * 0.5 + 0.7) * 0.5 + 0.35 * (1.0 - yy / gh)

        r = ground * (0.35 * base + 0.30 * (lane > 0.55)) + (1 - ground) * (sky * 0.55)
        g = ground * (0.30 * base + 0.35 * (lane > 0.55)) + (1 - ground) * (sky * 0.75)
        b = ground * (0.25 * base + 0.20 * (lane > 0.55)) + (1 - ground) * (sky * 1.00)

        # Throttle tint: warm at high throttle, cool when braking.
        thr = float(np.clip(action.throttle, -1.0, 1.0))
        r = r + 0.18 * max(thr, 0.0)
        b = b + 0.18 * max(-thr, 0.0)

        small = np.stack([r, g, b], axis=-1)
        small = np.clip(small, 0.0, 1.0) * 255.0
        img = small.astype(np.uint8)
        img = np.repeat(np.repeat(img, self._ds, axis=0), self._ds, axis=1)
        if img.shape[0] != h or img.shape[1] != w:
            img = np.ascontiguousarray(img[:h, :w])
            if img.shape[0] < h or img.shape[1] < w:
                pad = np.zeros((h, w, 3), dtype=np.uint8)
                pad[:img.shape[0], :img.shape[1]] = img
                img = pad
        else:
            img = np.ascontiguousarray(img)

        scale = 2 if self.cfg.decode_half_res else 3
        _draw_text(img, label, 8, 8, scale=scale, color=(255, 255, 60))
        return img

    def _render_chunk(self, action: Action, label_prefix: str) -> np.ndarray:
        frames = [self._render_frame(action, f"{label_prefix}-{i:02d}")
                  for i in range(PIXEL_FRAMES_PER_CHUNK)]
        return np.stack(frames, axis=0)

    # -- EngineBase -------------------------------------------------------
    def reset(self) -> np.ndarray:
        self._horizon_index = 0
        self._phase_x = 0.0
        self._phase_y = 0.0
        self._stats = None
        idle = Action()
        n_seed = int(np.clip(
            getattr(self.cfg, "seed_prefill_chunks", SEED_PREFILL_CHUNKS),
            1, SEED_PREFILL_CHUNKS))
        chunks = [self._render_chunk(idle, f"SEED{c}") for c in range(n_seed)]
        return np.concatenate(chunks, axis=0)

    def generate_horizon(self, action: Action) -> Iterator[np.ndarray]:
        t0 = time.perf_counter()
        n_chunks = max(int(self.cfg.horizon_chunks), 1)
        per_chunk_ms: List[float] = []
        first_frame_latency = 0.0
        gen_seconds = 0.0
        decode_seconds = 0.0
        idx = self._horizon_index

        for c in range(n_chunks):
            c0 = time.perf_counter()
            # Simulated compute: latency scales with the denoising ladder depth.
            budget = self.latency_s * (0.25 + 0.75 * self.cfg.denoising_steps / 4.0)
            if self.cfg.decode_half_res:
                budget *= 0.65
            frames = self._render_chunk(action, f"H{idx}C{c}")
            spent = time.perf_counter() - c0
            if budget > spent:
                time.sleep(budget - spent)
            dt = time.perf_counter() - c0
            per_chunk_ms.append(dt * 1000.0)
            gen_seconds += dt * 0.72
            decode_seconds += dt * 0.28
            if c == 0:
                first_frame_latency = time.perf_counter() - t0
            yield frames

        wall = max(time.perf_counter() - t0, 1e-6)
        total_frames = n_chunks * PIXEL_FRAMES_PER_CHUNK
        self._stats = HorizonStats(
            horizon_index=idx,
            action=Action(action.throttle, action.steer),
            denoising_steps=int(self.cfg.denoising_steps),
            precision=str(self.cfg.precision),
            gen_seconds=gen_seconds,
            decode_seconds=decode_seconds,
            first_frame_latency_s=first_frame_latency,
            gen_fps=total_frames / max(gen_seconds, 1e-6),
            end_to_end_fps=total_frames / wall,
            # Synthetic but plausible, so the HUD's formatting is exercised.
            peak_vram_gb=6.0 + 0.45 * self.cfg.denoising_steps
            + (0.0 if self.cfg.decode_half_res else 1.2),
            per_chunk_gen_ms=per_chunk_ms,
        )
        self._horizon_index += 1

    @property
    def last_stats(self) -> Optional[HorizonStats]:
        return self._stats

    def apply_live_settings(self, *, denoising_steps: Optional[int] = None,
                            horizon_chunks: Optional[int] = None,
                            decode_half_res: Optional[bool] = None,
                            seed_prefill_chunks: Optional[int] = None) -> None:
        if denoising_steps is not None:
            self.cfg.denoising_steps = int(np.clip(denoising_steps, 1, 12))
        if horizon_chunks is not None:
            self.cfg.horizon_chunks = max(int(horizon_chunks), 1)
        if decode_half_res is not None:
            self.cfg.decode_half_res = bool(decode_half_res)
            self._build_grid()
        if seed_prefill_chunks is not None:
            self.cfg.seed_prefill_chunks = int(np.clip(
                seed_prefill_chunks, 1, SEED_PREFILL_CHUNKS))

    def close(self) -> None:
        self._closed = True
