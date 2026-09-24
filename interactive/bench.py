#!/usr/bin/env python
"""Benchmark + quality harness for the interactive world model (WS-D).

One invocation = one configuration (or, with --matrix, a sequential sweep of
configurations, rebuilding the engine per config).

Each run:
  * builds an engine (interactive.engine.WorldModelEngine, or
    interactive.mock_engine.MockEngine with --mock),
  * resets from the same seed with a fixed torch seed,
  * drives the 8-horizon deterministic SCRIPTED_SEQUENCE,
  * collects HorizonStats per horizon,
  * writes interactive/bench_out/<slug>/clip.mp4 + stats.json,
  * appends one row to interactive/bench_results.csv,
  * compares frames against the reference clip (PSNR always, LPIPS when the
    `lpips` package is importable — never installed by this script).

Examples
--------
  python interactive/bench.py --mock --make-ref
  python interactive/bench.py --mock --matrix
  python interactive/bench.py --ckpt ckpt.pt --seed_zarr ride.zarr \
      --precision bf16 --steps 4 --compile off
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from interactive.engine_api import (  # noqa: E402
    Action,
    EngineConfig,
    HorizonStats,
    PIXEL_FRAMES_PER_CHUNK,
    PLAYBACK_FPS,
)
from interactive.action_script import (  # noqa: E402
    SCRIPTED_NAMES,
    SCRIPTED_SEQUENCE,
)

BENCH_DIR = os.path.join(_HERE, "bench_out")
CSV_PATH = os.path.join(_HERE, "bench_results.csv")
REPORT_PATH = os.path.join(_HERE, "BENCH_REPORT.md")
REF_NPZ = os.path.join(BENCH_DIR, "reference_frames.npz")
REF_META = os.path.join(BENCH_DIR, "reference_meta.json")

CSV_COLUMNS = [
    "timestamp",
    "ckpt_path",
    "ckpt_sha256_first16",
    "precision",
    "denoising_steps",
    "compile_mode",
    "decoder",
    "kv_cache_chunks",
    "horizon_chunks",
    "mean_gen_fps",
    "mean_end_to_end_fps",
    "mean_first_frame_latency_s",
    "p95_chunk_gen_ms",
    "peak_vram_gb",
    "psnr_vs_ref",
    "lpips_vs_ref",
    "clip_path",
    "notes",
]

# Every 4th frame is scored — quality metrics here detect collapse/artefacts,
# not bit-exactness, so a quarter of the frames is plenty and keeps LPIPS cheap.
QUALITY_FRAME_STRIDE = 4


# --------------------------------------------------------------------------
# private fallback mock (only used if interactive/mock_engine.py is unavailable)
# --------------------------------------------------------------------------
class _FallbackMockEngine:
    """Minimal EngineBase-shaped stand-in so the harness is testable alone.

    Deliberately private and only reachable when `interactive.mock_engine` is
    unimportable. It renders a cheap deterministic moving-gradient scene that
    responds to throttle/steer, so PSNR against a reference is meaningful.
    """

    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self._stats: Optional[HorizonStats] = None
        self._h = 0
        self._t = 0.0
        self._x = 0.0
        self._H, self._W = 120, 208

    def _frame(self) -> np.ndarray:
        yy, xx = np.mgrid[0 : self._H, 0 : self._W].astype(np.float32)
        u = (xx / self._W + self._x) % 1.0
        v = (yy / self._H + 0.15 * self._t) % 1.0
        r = 0.5 + 0.5 * np.sin(6.283 * (u + 0.1 * self._t))
        g = 0.5 + 0.5 * np.sin(6.283 * (v * 2.0))
        b = 0.5 + 0.5 * np.sin(6.283 * (u + v))
        return (np.stack([r, g, b], -1) * 255.0).astype(np.uint8)

    def reset(self) -> np.ndarray:
        self._h = 0
        self._t = 0.0
        self._x = 0.0
        self._stats = None
        return np.stack([self._frame() for _ in range(PIXEL_FRAMES_PER_CHUNK)])

    def generate_horizon(self, action: Action):
        t0 = time.perf_counter()
        per_chunk_ms: List[float] = []
        first_frame_at = None
        # A cheaper ladder is faster, mirroring the real speed/steps tradeoff.
        work = 0.004 * max(1, int(self.cfg.denoising_steps))
        for _ in range(self.cfg.horizon_chunks):
            c0 = time.perf_counter()
            frames = []
            for _f in range(PIXEL_FRAMES_PER_CHUNK):
                self._t += 1.0 / PLAYBACK_FPS
                self._x += 0.02 * action.throttle + 0.01 * action.steer
                frames.append(self._frame())
            # Fewer denoising steps -> visibly noisier frames (quality signal).
            noise = max(0, 4 - int(self.cfg.denoising_steps)) * 6
            out = np.stack(frames)
            if noise:
                rng = np.random.default_rng(1234 + self._h)
                out = np.clip(
                    out.astype(np.int16) + rng.integers(-noise, noise + 1, out.shape),
                    0, 255,
                ).astype(np.uint8)
            time.sleep(work)
            per_chunk_ms.append((time.perf_counter() - c0) * 1e3)
            if first_frame_at is None:
                first_frame_at = time.perf_counter()
            yield out
        wall = time.perf_counter() - t0
        n_pix = self.cfg.horizon_chunks * PIXEL_FRAMES_PER_CHUNK
        self._stats = HorizonStats(
            horizon_index=self._h,
            action=action,
            denoising_steps=self.cfg.denoising_steps,
            precision=self.cfg.precision,
            gen_seconds=wall * 0.8,
            decode_seconds=wall * 0.2,
            first_frame_latency_s=(first_frame_at or time.perf_counter()) - t0,
            gen_fps=n_pix / max(wall * 0.8, 1e-9),
            end_to_end_fps=n_pix / max(wall, 1e-9),
            peak_vram_gb=0.0,
            per_chunk_gen_ms=per_chunk_ms,
        )
        self._h += 1

    @property
    def last_stats(self) -> Optional[HorizonStats]:
        return self._stats

    def apply_live_settings(self, **kw) -> None:
        for k, v in kw.items():
            if v is not None:
                setattr(self.cfg, k, v)

    def close(self) -> None:
        pass


_USED_FALLBACK_MOCK = False


def build_engine(cfg: EngineConfig, mock: bool):
    """Instantiate the engine for one configuration."""
    global _USED_FALLBACK_MOCK
    if mock:
        try:
            from interactive.mock_engine import MockEngine  # type: ignore
            return MockEngine(cfg)
        except Exception as exc:  # pragma: no cover - depends on WS-C landing
            _USED_FALLBACK_MOCK = True
            print(f"[bench] interactive.mock_engine unavailable ({exc}); "
                  f"using bench.py's private fallback mock")
            return _FallbackMockEngine(cfg)
    from interactive.engine import WorldModelEngine  # type: ignore
    return WorldModelEngine(cfg)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def slugify(cfg: EngineConfig, mock: bool) -> str:
    tag = "mock" if mock else os.path.splitext(os.path.basename(cfg.ckpt_path or "nockpt"))[0]
    # The decoder is appended only when it is not the reference "wan", so
    # every pre-existing slug (and the reference clip's directory) is stable.
    dec = str(getattr(cfg, "decoder", "wan") or "wan").lower()
    dec_bit = "" if dec == "wan" else f"_d{dec}"
    bc = int(getattr(cfg, "block_chunks", 1))
    block_bit = "" if bc == 1 else f"_b{bc}joint"
    return (f"{tag}_{cfg.precision}_s{cfg.denoising_steps}"
            f"_c{cfg.compile_mode}_kv{cfg.kv_cache_chunks}_h{cfg.horizon_chunks}"
            f"{dec_bit}{block_bit}")


def ckpt_sha256(ckpt_path: str) -> str:
    """SHA-256 of the checkpoint, cached under bench_out (never beside the ckpt)."""
    if not ckpt_path or not os.path.isfile(ckpt_path):
        return ""
    os.makedirs(BENCH_DIR, exist_ok=True)
    key = hashlib.sha256(os.path.abspath(ckpt_path).encode()).hexdigest()[:16]
    cache = os.path.join(BENCH_DIR, f"{os.path.basename(ckpt_path)}.{key}.sha256")
    st = os.stat(ckpt_path)
    if os.path.isfile(cache):
        try:
            with open(cache) as fh:
                rec = json.load(fh)
            if rec.get("size") == st.st_size and rec.get("mtime") == st.st_mtime:
                return rec["sha256"]
        except Exception:
            pass
    h = hashlib.sha256()
    with open(ckpt_path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    digest = h.hexdigest()
    with open(cache, "w") as fh:
        json.dump({"sha256": digest, "size": st.st_size, "mtime": st.st_mtime,
                   "path": os.path.abspath(ckpt_path)}, fh)
    return digest


def p95(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), 95))


def write_mp4(frames: np.ndarray, path: str, fps: float = PLAYBACK_FPS) -> bool:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        import imageio.v2 as imageio
        with imageio.get_writer(path, fps=fps, macro_block_size=1,
                                codec="libx264", quality=8) as w:
            for f in frames:
                w.append_data(f)
        return True
    except Exception as exc:
        print(f"[bench] mp4 write failed ({exc}); trying cv2")
    try:
        import cv2
        h, w = frames.shape[1:3]
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in frames:
            vw.write(f[:, :, ::-1])
        vw.release()
        return True
    except Exception as exc:
        print(f"[bench] cv2 mp4 write failed too ({exc})")
        return False


# --------------------------------------------------------------------------
# quality metrics
# --------------------------------------------------------------------------
def _match_shape(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Resize `a` to b's HxW if needed (bench compares across configs)."""
    if a.shape[1:3] == b.shape[1:3]:
        return a
    try:
        import cv2
        h, w = b.shape[1:3]
        return np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in a])
    except Exception:
        return a


def psnr_vs_ref(frames: np.ndarray, ref: np.ndarray) -> float:
    n = min(len(frames), len(ref))
    if n == 0:
        return float("nan")
    idx = range(0, n, QUALITY_FRAME_STRIDE)
    a = _match_shape(frames[:n][list(idx)], ref[:n][list(idx)])
    b = ref[:n][list(idx)]
    if a.shape != b.shape:
        return float("nan")
    vals = []
    for x, y in zip(a.astype(np.float64), b.astype(np.float64)):
        mse = float(np.mean((x - y) ** 2))
        vals.append(100.0 if mse <= 1e-12 else 10.0 * math.log10(255.0 ** 2 / mse))
    return float(np.mean(vals))


def lpips_vs_ref(frames: np.ndarray, ref: np.ndarray) -> tuple[float, str]:
    """Mean LPIPS(alex). Returns (value, note). Never installs anything."""
    try:
        import lpips as lpips_pkg
        import torch
    except Exception:
        return float("nan"), "lpips-unavailable"
    n = min(len(frames), len(ref))
    if n == 0:
        return float("nan"), "no-frames"
    idx = list(range(0, n, QUALITY_FRAME_STRIDE))
    a = _match_shape(frames[:n][idx], ref[:n][idx])
    b = ref[:n][idx]
    if a.shape != b.shape:
        return float("nan"), "shape-mismatch"
    try:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        net = lpips_pkg.LPIPS(net="alex").to(dev).eval()
        vals = []
        with torch.no_grad():
            for i in range(0, len(a), 8):
                ta = torch.from_numpy(a[i:i + 8]).permute(0, 3, 1, 2).float().to(dev)
                tb = torch.from_numpy(b[i:i + 8]).permute(0, 3, 1, 2).float().to(dev)
                ta = ta / 127.5 - 1.0
                tb = tb / 127.5 - 1.0
                vals.append(net(ta, tb).flatten().cpu().numpy())
        del net
        return float(np.concatenate(vals).mean()), ""
    except Exception as exc:
        return float("nan"), f"lpips-failed:{type(exc).__name__}"


# --------------------------------------------------------------------------
# single run
# --------------------------------------------------------------------------
def run_one(cfg: EngineConfig, *, mock: bool, make_ref: bool,
            torch_seed: int, notes: str = "") -> Dict[str, object]:
    try:
        import torch
        torch.manual_seed(torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch_seed)
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        torch = None  # type: ignore
    np.random.seed(torch_seed)

    slug = slugify(cfg, mock)
    out_dir = os.path.join(BENCH_DIR, slug)
    os.makedirs(out_dir, exist_ok=True)

    engine = build_engine(cfg, mock)
    all_frames: List[np.ndarray] = []
    stats_list: List[HorizonStats] = []
    note_bits = [notes] if notes else []
    if _USED_FALLBACK_MOCK:
        note_bits.append("fallback-mock")

    try:
        seed_frames = engine.reset()
        if seed_frames is not None and len(seed_frames):
            all_frames.append(np.asarray(seed_frames, dtype=np.uint8))
        for i, action in enumerate(SCRIPTED_SEQUENCE):
            for chunk in engine.generate_horizon(action):
                all_frames.append(np.asarray(chunk, dtype=np.uint8))
            st = engine.last_stats
            if st is None:
                st = HorizonStats(horizon_index=i, action=action,
                                  denoising_steps=cfg.denoising_steps,
                                  precision=cfg.precision)
                note_bits.append(f"h{i}-no-stats")
            stats_list.append(st)
            print(f"[bench] {slug} h{i} ({SCRIPTED_NAMES[i]}): "
                  f"gen_fps={st.gen_fps:.1f} e2e_fps={st.end_to_end_fps:.1f} "
                  f"ffl={st.first_frame_latency_s * 1e3:.0f}ms")
    finally:
        try:
            engine.close()
        except Exception:
            pass

    frames = np.concatenate(all_frames, axis=0) if all_frames else np.zeros((0, 8, 8, 3), np.uint8)

    clip_path = os.path.join(out_dir, "clip.mp4")
    if not write_mp4(frames, clip_path):
        clip_path = ""
        note_bits.append("mp4-write-failed")

    # ---- quality vs reference
    psnr = float("nan")
    lp = float("nan")
    if make_ref:
        os.makedirs(BENCH_DIR, exist_ok=True)
        np.savez_compressed(REF_NPZ, frames=frames)
        with open(REF_META, "w") as fh:
            json.dump({"slug": slug, "precision": cfg.precision,
                       "denoising_steps": cfg.denoising_steps,
                       "compile_mode": cfg.compile_mode,
                       "kv_cache_chunks": cfg.kv_cache_chunks,
                       "horizon_chunks": cfg.horizon_chunks,
                       "mock": mock, "n_frames": int(len(frames)),
                       "shape": list(frames.shape[1:]),
                       "torch_seed": torch_seed,
                       "created": datetime.now(timezone.utc).isoformat()}, fh, indent=2)
        note_bits.append("reference")
        print(f"[bench] wrote reference ({len(frames)} frames) -> {REF_NPZ}")
    elif os.path.isfile(REF_NPZ):
        ref = np.load(REF_NPZ)["frames"]
        psnr = psnr_vs_ref(frames, ref)
        lp, lnote = lpips_vs_ref(frames, ref)
        if lnote:
            note_bits.append(lnote)
    else:
        note_bits.append("no-reference")

    peak_vram = max([s.peak_vram_gb for s in stats_list] or [0.0])
    if torch is not None:
        try:
            if torch.cuda.is_available():
                peak_vram = max(peak_vram, torch.cuda.max_memory_allocated() / 1e9)
        except Exception:
            pass

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ckpt_path": cfg.ckpt_path,
        "ckpt_sha256_first16": ckpt_sha256(cfg.ckpt_path)[:16],
        "precision": cfg.precision,
        "denoising_steps": cfg.denoising_steps,
        "compile_mode": cfg.compile_mode,
        "decoder": getattr(cfg, "decoder", "wan"),
        "kv_cache_chunks": cfg.kv_cache_chunks,
        "horizon_chunks": cfg.horizon_chunks,
        "mean_gen_fps": round(float(np.mean([s.gen_fps for s in stats_list])), 3),
        "mean_end_to_end_fps": round(float(np.mean([s.end_to_end_fps for s in stats_list])), 3),
        "mean_first_frame_latency_s": round(
            float(np.mean([s.first_frame_latency_s for s in stats_list])), 4),
        "p95_chunk_gen_ms": round(
            p95([ms for s in stats_list for ms in s.per_chunk_gen_ms]), 2),
        "peak_vram_gb": round(float(peak_vram), 3),
        "psnr_vs_ref": "" if math.isnan(psnr) else round(psnr, 3),
        "lpips_vs_ref": "" if math.isnan(lp) else round(lp, 5),
        "clip_path": clip_path,
        "notes": ";".join([n for n in note_bits if n]),
    }

    with open(os.path.join(out_dir, "stats.json"), "w") as fh:
        json.dump({
            "config": dataclasses.asdict(cfg),
            "mock": mock,
            "torch_seed": torch_seed,
            "script": SCRIPTED_NAMES,
            "row": row,
            "horizons": [dataclasses.asdict(s) for s in stats_list],
        }, fh, indent=2, default=str)

    append_csv(row)
    return row


def append_csv(row: Dict[str, object]) -> None:
    exists = os.path.isfile(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def _f(v, default=float("nan")) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def write_report(rows: Optional[List[Dict[str, str]]] = None) -> str:
    if rows is None:
        if not os.path.isfile(CSV_PATH):
            return ""
        with open(CSV_PATH) as fh:
            rows = list(csv.DictReader(fh))
    if not rows:
        return ""

    # Partition before analysing.  Pooling these together produces nonsense:
    # mock rows are a CPU stand-in running ~9x faster than the real model (and
    # self-compare at PSNR 100), and failed rows carry no numbers at all, which
    # turns every mean into NaN.  Only real runs that produced numbers are
    # summarised; the rest are listed separately for the record.
    all_rows = rows
    failed = [r for r in all_rows if not (r.get("mean_end_to_end_fps") or "").strip()]
    scored_rows = [r for r in all_rows if r not in failed]
    mock = [r for r in scored_rows if not (r.get("ckpt_sha256_first16") or "").strip()]
    rows = [r for r in scored_rows if r not in mock]
    if not rows:  # nothing real yet — fall back so the report is still useful
        rows, mock = scored_rows, []

    srt = sorted(rows, key=lambda r: -_f(r.get("mean_end_to_end_fps"), -1e9))
    lines: List[str] = []
    lines.append("# Interactive world model — bench report")
    lines.append("")
    lines.append(f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
                 f"from `interactive/bench_results.csv` "
                 f"({len(all_rows)} rows: {len(rows)} real, {len(mock)} mock, "
                 f"{len(failed)} failed).")
    lines.append("")
    lines.append("Tables and commentary below cover **real-checkpoint runs only**. "
                 "Mock-engine rows and failed configs are listed at the end.")
    lines.append("")
    if os.path.isfile(REF_META):
        with open(REF_META) as fh:
            meta = json.load(fh)
        lines.append(f"Reference clip: `{meta.get('slug')}` "
                     f"({meta.get('precision')}, {meta.get('denoising_steps')} steps, "
                     f"compile={meta.get('compile_mode')}, {meta.get('n_frames')} frames). "
                     f"PSNR/LPIPS below are against it, every "
                     f"{QUALITY_FRAME_STRIDE}th frame.")
        lines.append("")

    # ---- headline: steps vs speed/quality (the primary research question)
    lines.append("## Denoising steps vs speed and quality")
    lines.append("")
    lines.append("| precision | steps | decoder | compile | e2e fps | gen fps | first-frame s "
                 "| p95 chunk ms | PSNR vs ref | LPIPS vs ref |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r.get("precision", ""),
                                         -_f(r.get("denoising_steps"), 0),
                                         r.get("decoder", ""),
                                         r.get("compile_mode", ""))):
        lines.append(
            f"| {r.get('precision','')} | {r.get('denoising_steps','')} "
            f"| {r.get('decoder','') or 'wan'} "
            f"| {r.get('compile_mode','')} | {_f(r.get('mean_end_to_end_fps')):.2f} "
            f"| {_f(r.get('mean_gen_fps')):.2f} "
            f"| {_f(r.get('mean_first_frame_latency_s')):.3f} "
            f"| {_f(r.get('p95_chunk_gen_ms')):.1f} "
            f"| {r.get('psnr_vs_ref','') or '—'} | {r.get('lpips_vs_ref','') or '—'} |")
    lines.append("")

    # ---- full table sorted by e2e fps
    lines.append("## All runs (sorted by end-to-end fps)")
    lines.append("")
    hdr = ["timestamp", "precision", "denoising_steps", "compile_mode", "decoder",
           "kv_cache_chunks", "horizon_chunks", "mean_gen_fps",
           "mean_end_to_end_fps", "mean_first_frame_latency_s",
           "p95_chunk_gen_ms", "peak_vram_gb", "psnr_vs_ref", "lpips_vs_ref",
           "ckpt_sha256_first16", "notes"]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "---|" * len(hdr))
    for r in srt:
        lines.append("| " + " | ".join(str(r.get(k, "") or "—") for k in hdr) + " |")
    lines.append("")

    # ---- commentary
    lines.append("## Commentary (auto-generated)")
    lines.append("")
    fastest = srt[0]
    lines.append(f"- Fastest overall: **{fastest.get('precision')} / "
                 f"{fastest.get('denoising_steps')} steps / decoder="
                 f"{fastest.get('decoder') or 'wan'} / compile="
                 f"{fastest.get('compile_mode')}** at "
                 f"{_f(fastest.get('mean_end_to_end_fps')):.2f} end-to-end fps "
                 f"({_f(fastest.get('mean_first_frame_latency_s')):.3f}s to first frame).")

    # Real-time target: playback is 16 fps, so end-to-end >= 16 means the
    # engine generates at least as fast as the video plays.
    rt = [r for r in srt if _f(r.get("mean_end_to_end_fps"), 0) >= 16.0]
    if rt:
        lines.append(f"- **Real-time capable (>= 16 e2e fps): {len(rt)} config(s).** "
                     "Fastest: "
                     + ", ".join(f"{r.get('denoising_steps')} steps/"
                                 f"{r.get('decoder') or 'wan'} "
                                 f"({_f(r.get('mean_end_to_end_fps')):.2f} fps)"
                                 for r in rt[:3]))
    else:
        lines.append("- **No config reaches the 16 fps real-time target.** Best is "
                     f"{_f(srt[0].get('mean_end_to_end_fps')):.2f} e2e fps "
                     f"({srt[0].get('denoising_steps')} steps, decoder="
                     f"{srt[0].get('decoder') or 'wan'}).")

    scored = [r for r in srt if r.get("psnr_vs_ref") not in (None, "")]
    if scored:
        tiers = [("high (PSNR >= 30 dB)", 30.0, float("inf")),
                 ("medium (25-30 dB)", 25.0, 30.0),
                 ("low (< 25 dB)", -float("inf"), 25.0)]
        for name, lo, hi in tiers:
            cand = [r for r in scored if lo <= _f(r.get("psnr_vs_ref")) < hi]
            if not cand:
                continue
            best = cand[0]  # already sorted by fps
            lines.append(f"- Best fps at quality tier {name}: "
                         f"{best.get('precision')} / {best.get('denoising_steps')} steps "
                         f"/ decoder={best.get('decoder') or 'wan'} "
                         f"/ compile={best.get('compile_mode')} — "
                         f"{_f(best.get('mean_end_to_end_fps')):.2f} fps, "
                         f"PSNR {best.get('psnr_vs_ref')} dB"
                         + (f", LPIPS {best.get('lpips_vs_ref')}"
                            if best.get("lpips_vs_ref") else ""))
    else:
        lines.append("- No PSNR values yet: run once with `--make-ref` first.")

    # steps tradeoff, within the dominant precision
    by_steps: Dict[int, List[Dict[str, str]]] = {}
    for r in rows:
        try:
            by_steps.setdefault(int(_f(r.get("denoising_steps"), 0)), []).append(r)
        except Exception:
            pass
    if len(by_steps) > 1:
        ks = sorted(by_steps)
        base = max(ks)
        base_fps = float(np.mean([_f(r.get("mean_end_to_end_fps")) for r in by_steps[base]]))
        for k in ks:
            fps = float(np.mean([_f(r.get("mean_end_to_end_fps")) for r in by_steps[k]]))
            ps = [_f(r.get("psnr_vs_ref")) for r in by_steps[k]
                  if r.get("psnr_vs_ref") not in (None, "")]
            pstr = f", mean PSNR {np.mean(ps):.2f} dB" if ps else ""
            speed = fps / base_fps if base_fps else float("nan")
            lines.append(f"- {k} steps: mean {fps:.2f} e2e fps "
                         f"({speed:.2f}x vs {base} steps){pstr}")

    if any("lpips-unavailable" in (r.get("notes") or "") for r in rows):
        lines.append("- LPIPS is NaN/blank for some rows: the `lpips` package is not "
                     "importable in this env, and the harness never installs into `flash`.")
    if any("fallback-mock" in (r.get("notes") or "") for r in rows):
        lines.append("- Some rows used bench.py's private fallback mock engine "
                     "(`interactive/mock_engine.py` was unavailable); timings are "
                     "synthetic and say nothing about the real model.")
    lines.append("")

    # ---- excluded rows, kept for the record
    if failed:
        lines.append("## Failed configs (excluded from the tables above)")
        lines.append("")
        for r in failed:
            lines.append(f"- **{r.get('precision','')} / {r.get('denoising_steps','')} steps "
                         f"/ compile={r.get('compile_mode','')}** "
                         f"({r.get('timestamp','')}): {r.get('notes','') or 'no reason recorded'}")
        lines.append("")
    if mock:
        lines.append("## Mock-engine rows (not the real model)")
        lines.append("")
        lines.append("The mock engine is a CPU/synthetic stand-in used to exercise the harness. "
                     "Its timings are roughly an order of magnitude faster than the real model "
                     "and its PSNR is a self-comparison, so it is never mixed into the numbers "
                     "above.")
        lines.append("")
        lines.append("| precision | steps | compile | e2e fps | gen fps | first-frame s |")
        lines.append("|---|---|---|---|---|---|")
        for r in sorted(mock, key=lambda r: -_f(r.get("mean_end_to_end_fps"), -1e9)):
            lines.append(f"| {r.get('precision','')} | {r.get('denoising_steps','')} "
                         f"| {r.get('compile_mode','')} "
                         f"| {_f(r.get('mean_end_to_end_fps')):.2f} "
                         f"| {_f(r.get('mean_gen_fps')):.2f} "
                         f"| {_f(r.get('mean_first_frame_latency_s')):.3f} |")
        lines.append("")

    text = "\n".join(lines)
    with open(REPORT_PATH, "w") as fh:
        fh.write(text)
    print(f"[bench] wrote {REPORT_PATH}")
    return text


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="", help="checkpoint path")
    p.add_argument("--seed_zarr", default="", help="seed ride zarr")
    p.add_argument("--config", default=EngineConfig.config_path)
    p.add_argument("--wan_model_path", default=EngineConfig.wan_model_path)
    p.add_argument("--precision", default="bf16")
    p.add_argument("--steps", type=int, default=4, help="denoising steps")
    p.add_argument("--compile", dest="compile_mode", default="off")
    p.add_argument("--kv_cache_chunks", type=int, default=7)
    p.add_argument("--horizon_chunks", type=int, default=EngineConfig.horizon_chunks)
    p.add_argument("--block_chunks", type=int, default=1, choices=[1, 2, 4])
    p.add_argument("--seed", type=int, default=0, help="engine/torch seed")
    p.add_argument("--mock", action="store_true", help="use the mock engine")
    p.add_argument("--make-ref", dest="make_ref", action="store_true",
                   help="save this run's frames as the quality reference")
    p.add_argument("--matrix", action="store_true", help="run a sweep sequentially")
    p.add_argument("--precisions", type=_csv_list, default=["bf16"],
                   help="matrix: comma list, e.g. bf16,fp8_wo,fp4_wo")
    p.add_argument("--steps-list", dest="steps_list", type=_csv_list,
                   default=["4", "3", "2", "1"], help="matrix: comma list of step counts")
    p.add_argument("--compiles", type=_csv_list, default=["off"],
                   help="matrix: comma list, e.g. off,reduce-overhead")
    p.add_argument("--decoder", default="wan",
                   help="latent->pixel decoder: wan (reference) | lightvaew2_1 "
                        "| taew2_1 | lighttaew2_1 (see interactive/decoders.py)")
    p.add_argument("--decoders", type=_csv_list, default=None,
                   help="matrix: comma list of decoders (defaults to --decoder)")
    p.add_argument("--notes", default="")
    p.add_argument("--report-only", dest="report_only", action="store_true",
                   help="regenerate BENCH_REPORT.md from the existing CSV and exit")
    return p


def cfg_from_args(a: argparse.Namespace, *, precision=None, steps=None,
                  compile_mode=None, decoder=None) -> EngineConfig:
    cfg = EngineConfig(
        ckpt_path=a.ckpt,
        config_path=a.config,
        wan_model_path=a.wan_model_path,
        seed_zarr=a.seed_zarr,
        precision=precision or a.precision,
        compile_mode=compile_mode or a.compile_mode,
        denoising_steps=int(steps if steps is not None else a.steps),
        kv_cache_chunks=a.kv_cache_chunks,
        horizon_chunks=a.horizon_chunks,
        block_chunks=a.block_chunks,
        seed=a.seed,
        decoder=decoder or a.decoder,
    )
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    os.makedirs(BENCH_DIR, exist_ok=True)

    if a.report_only:
        write_report()
        return 0

    if a.matrix:
        rows: List[Dict[str, object]] = []
        decs = a.decoders or [a.decoder]
        combos = [(p, int(s), c, d) for p in a.precisions
                  for s in a.steps_list for c in a.compiles for d in decs]
        print(f"[bench] matrix: {len(combos)} configurations")
        for i, (prec, steps, comp, dec) in enumerate(combos):
            print(f"\n[bench] === {i + 1}/{len(combos)}: "
                  f"{prec} / {steps} steps / compile={comp} / decoder={dec} ===")
            cfg = cfg_from_args(a, precision=prec, steps=steps, compile_mode=comp,
                                decoder=dec)
            try:
                rows.append(run_one(cfg, mock=a.mock, make_ref=False,
                                    torch_seed=a.seed, notes=a.notes))
            except Exception as exc:
                print(f"[bench] config FAILED: {type(exc).__name__}: {exc}")
                import traceback
                traceback.print_exc()
        write_report()
        print(f"\n[bench] matrix done: {len(rows)}/{len(combos)} configs succeeded")
        return 0 if len(rows) == len(combos) else 1

    cfg = cfg_from_args(a)
    run_one(cfg, mock=a.mock, make_ref=a.make_ref, torch_seed=a.seed, notes=a.notes)
    write_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
