#!/usr/bin/env python3
"""Standalone quality/speed bench for the latent->pixel decoder zoo (WS-F).

No DiT, no checkpoint, no T5: this reads N consecutive GT latent frames
straight out of the seed zarr and pushes them through each decoder with the
engine's streaming contract (one ``reset()``, then one ``decode_chunk()`` per
3-latent-frame chunk).  The Wan VAE's output is the reference; every
alternative is scored against it.

Run (lpips lives in the flash-q clone, not flash):

    conda run -n flash-q python -m interactive.decoder_bench \\
        --zarr ~/20240224003808.zarr --chunks 8 \\
        --wan_model_path /home/ashish/Wan2.1/

Outputs under interactive/decoder_bench_out/:
    <name>.mp4          decoded clip per decoder
    compare_f<k>.png    one frame from every decoder, stacked + labelled
    results.csv         machine-readable table
    results.md          the markdown table pasted into DECODER_NOTES.md
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interactive.decoders import DECODER_NAMES, build_decoder  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "decoder_bench_out"
LATENT_FRAMES_PER_CHUNK = 3
PIXEL_FRAMES_PER_CHUNK = 12
PLAYBACK_FPS = 16.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def gpu_busy_mb() -> int:
    """Largest per-process GPU memory (MiB) among running compute apps."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"], text=True, timeout=30)
    except Exception as exc:                       # driver gone / no nvidia-smi
        raise SystemExit(f"ABORT: cannot query the GPU ({exc}).")
    biggest = 0
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[1].isdigit():
            biggest = max(biggest, int(parts[1]))
    return biggest


def wait_for_free_gpu(limit_mb: int, max_wait_s: int, poll_s: int) -> None:
    t0 = time.time()
    while True:
        busy = gpu_busy_mb()
        if busy <= limit_mb:
            print(f"[gpu] free (largest compute process {busy} MiB <= {limit_mb} MiB)")
            return
        waited = time.time() - t0
        if waited > max_wait_s:
            raise SystemExit(
                f"ABORT: GPU still busy ({busy} MiB) after {waited/60:.0f} min.")
        print(f"[gpu] busy ({busy} MiB); waiting {poll_s}s "
              f"(elapsed {waited/60:.1f} min)", flush=True)
        time.sleep(poll_s)


def load_latents(zarr_path: str, n_frames: int, offset: int) -> torch.Tensor:
    import zarr as zarr_lib
    g = zarr_lib.open_group(os.path.expanduser(zarr_path), mode="r")
    lat = g["latents"][offset:offset + n_frames]         # [T, C, H, W] fp16
    t = torch.from_numpy(np.asarray(lat).astype(np.float32))
    if t.shape[0] < n_frames:
        raise SystemExit(f"zarr has only {t.shape[0]} frames from offset {offset}")
    return t.unsqueeze(0)                                # [1, T, C, H, W]


def write_mp4(path: Path, frames: np.ndarray, fps: float) -> None:
    """frames: uint8 [T, H, W, 3]."""
    h, w = frames.shape[1:3]
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", "16",
           "-pix_fmt", "yuv420p", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    p.stdin.write(frames.tobytes())
    p.stdin.close()
    if p.wait() != 0:
        print(f"[warn] ffmpeg failed for {path}")


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """uint8 [T, H, W, 3] vs uint8, per-clip PSNR on the 0-255 scale."""
    d = a.astype(np.float64) - b.astype(np.float64)
    mse = float((d * d).mean())
    return float("inf") if mse == 0 else 10.0 * float(np.log10(255.0 ** 2 / mse))


def lpips_score(model, a: np.ndarray, b: np.ndarray, device, batch: int = 8) -> float:
    """Mean LPIPS over frames.  a/b uint8 [T, H, W, 3]."""
    vals = []
    for i in range(0, a.shape[0], batch):
        ta = torch.from_numpy(a[i:i + batch]).to(device).permute(0, 3, 1, 2).float()
        tb = torch.from_numpy(b[i:i + batch]).to(device).permute(0, 3, 1, 2).float()
        ta = ta.div_(127.5).sub_(1.0)
        tb = tb.div_(127.5).sub_(1.0)
        with torch.no_grad():
            vals.append(model(ta, tb).flatten().float().cpu())
    return float(torch.cat(vals).mean())


def label_strip(width: int, text: str, height: int = 26) -> np.ndarray:
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (width, height), (16, 16, 16))
    ImageDraw.Draw(img).text((8, 6), text, fill=(240, 240, 240))
    return np.asarray(img)


# ---------------------------------------------------------------------------
# per-decoder run
# ---------------------------------------------------------------------------
def run_decoder(name: str, latents: torch.Tensor, args, dtype) -> dict:
    device = torch.device(args.device)
    n_chunks = latents.shape[1] // LATENT_FRAMES_PER_CHUNK

    t_build = time.time()
    dec = build_decoder(name, device=device, dtype=dtype,
                        wan_model_path=args.wan_model_path)
    torch.cuda.synchronize(device)
    build_s = time.time() - t_build

    # --- warmup: a full pass (allocator warm, cudnn algos picked, caches sized).
    # A whole pass, not a repeated first chunk, so the causal state we measure
    # later starts from a genuine reset rather than mid-stream.
    dec.reset()
    for c in range(n_chunks):
        dec.decode_chunk(latents[:, c * LATENT_FRAMES_PER_CHUNK:
                                    (c + 1) * LATENT_FRAMES_PER_CHUNK])
    torch.cuda.synchronize(device)

    per_chunk_ms: list[list[float]] = [[] for _ in range(n_chunks)]
    frames_cpu = None
    torch.cuda.reset_peak_memory_stats(device)
    base_mem = torch.cuda.memory_allocated(device)

    for rep in range(args.repeats):
        dec.reset()
        out = []
        for c in range(n_chunks):
            chunk = latents[:, c * LATENT_FRAMES_PER_CHUNK:
                               (c + 1) * LATENT_FRAMES_PER_CHUNK]
            ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
            ev0.record()
            px = dec.decode_chunk(chunk)
            ev1.record()
            ev1.synchronize()
            per_chunk_ms[c].append(ev0.elapsed_time(ev1))
            if rep == 0:
                out.append(px.cpu().numpy())
        if rep == 0:
            frames_cpu = np.concatenate(out, axis=0)

    peak_gb = torch.cuda.max_memory_allocated(device) / 2 ** 30
    weights_gb = base_mem / 2 ** 30

    # chunk 0 is the ragged one (9 frames, cold caches); the steady-state
    # number is what an engine at 12 frames/chunk actually pays.
    steady = [t for c in range(1, n_chunks) for t in per_chunk_ms[c]]
    all_ms = [t for c in range(n_chunks) for t in per_chunk_ms[c]]
    med = statistics.median(steady) if steady else statistics.median(all_ms)

    res = {
        "decoder": name,
        "params_M": dec.parameter_count() / 1e6,
        "build_s": build_s,
        "ms_per_chunk_median": med,
        "ms_per_chunk_p90": float(np.percentile(steady or all_ms, 90)),
        "ms_chunk0": statistics.median(per_chunk_ms[0]),
        "fps_equiv": PIXEL_FRAMES_PER_CHUNK * 1000.0 / med,
        "realtime_x": (PIXEL_FRAMES_PER_CHUNK * 1000.0 / med) / PLAYBACK_FPS,
        "s_per_6chunk_horizon": 6 * med / 1000.0,
        "peak_vram_gb": peak_gb,
        "weights_vram_gb": weights_gb,
        "n_frames": int(frames_cpu.shape[0]),
        "per_chunk_ms_median": [statistics.median(v) for v in per_chunk_ms],
    }
    del dec
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    return res, frames_cpu


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", default=str(Path.home() / "20240224003808.zarr"))
    ap.add_argument("--offset", type=int, default=0, help="first latent frame")
    ap.add_argument("--chunks", type=int, default=8, help="chunks of 3 latent frames")
    ap.add_argument("--repeats", type=int, default=3, help="timed passes per decoder")
    ap.add_argument("--decoders", default=",".join(DECODER_NAMES))
    ap.add_argument("--wan_model_path", default="/home/ashish/Wan2.1/")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--compare_frame", type=int, default=40)
    ap.add_argument("--no_lpips", action="store_true")
    ap.add_argument("--gpu_free_mb", type=int, default=2048,
                    help="max per-process GPU MiB tolerated before starting")
    ap.add_argument("--gpu_wait_min", type=int, default=120)
    ap.add_argument("--gpu_poll_s", type=int, default=300)
    ap.add_argument("--skip_gpu_wait", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("ABORT: no CUDA device.")
    if not args.skip_gpu_wait:
        wait_for_free_gpu(args.gpu_free_mb, args.gpu_wait_min * 60, args.gpu_poll_s)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]
    device = torch.device(args.device)
    names = [n.strip() for n in args.decoders.split(",") if n.strip()]
    if names[0] != "wan":
        names = ["wan"] + [n for n in names if n != "wan"]

    n_lat = args.chunks * LATENT_FRAMES_PER_CHUNK
    latents = load_latents(args.zarr, n_lat, args.offset).to(device, dtype)
    print(f"[bench] latents {tuple(latents.shape)} dtype={args.dtype} "
          f"from {args.zarr}[{args.offset}:{args.offset + n_lat}]")

    results, clips = [], {}
    for name in names:
        print(f"\n[bench] === {name} ===", flush=True)
        try:
            res, frames = run_decoder(name, latents, args, dtype)
        except torch.cuda.OutOfMemoryError as exc:
            print(f"[bench] {name}: OOM ({exc}); skipped.")
            torch.cuda.empty_cache()
            continue
        except Exception as exc:                       # noqa: BLE001
            print(f"[bench] {name}: FAILED -> {type(exc).__name__}: {exc}")
            torch.cuda.empty_cache()
            continue
        results.append(res)
        clips[name] = frames
        print(f"[bench] {name}: {res['ms_per_chunk_median']:.1f} ms/chunk "
              f"({res['fps_equiv']:.0f} fps-equiv, {res['realtime_x']:.1f}x realtime), "
              f"peak {res['peak_vram_gb']:.2f} GB, {res['n_frames']} frames")
        write_mp4(out_dir / f"{name}.mp4", frames, PLAYBACK_FPS)

    if "wan" not in clips:
        raise SystemExit("ABORT: the Wan reference decoder did not run; no baseline.")
    ref = clips["wan"]

    lp = None
    if not args.no_lpips:
        try:
            import lpips as lpips_lib
            lp = lpips_lib.LPIPS(net="alex").to(device).eval()
            print("[bench] LPIPS(alex) loaded")
        except Exception as exc:                       # noqa: BLE001
            print(f"[bench] LPIPS unavailable ({exc}); PSNR only. "
                  "Run under `conda run -n flash-q` to get LPIPS.")

    for res in results:
        cur = clips[res["decoder"]]
        n = min(len(ref), len(cur))
        res["psnr_db"] = float("inf") if res["decoder"] == "wan" else psnr(ref[:n], cur[:n])
        res["lpips"] = 0.0 if res["decoder"] == "wan" else (
            lpips_score(lp, ref[:n], cur[:n], device) if lp is not None else float("nan"))
        res["speedup_vs_wan"] = (
            results[0]["ms_per_chunk_median"] / res["ms_per_chunk_median"])

    # --- side-by-side PNG ---------------------------------------------------
    k = min(args.compare_frame, len(ref) - 1)
    tiles = []
    for res in results:
        nm = res["decoder"]
        tiles.append(label_strip(
            ref.shape[2],
            f"{nm}   {res['ms_per_chunk_median']:.0f} ms/chunk   " +
            ("reference" if nm == "wan" else
             f"PSNR {res['psnr_db']:.2f} dB   LPIPS {res['lpips']:.4f}")))
        tiles.append(clips[nm][k])
    from PIL import Image
    Image.fromarray(np.concatenate(tiles, axis=0)).save(
        out_dir / f"compare_f{k}.png")
    print(f"[bench] wrote {out_dir / f'compare_f{k}.png'}")

    # --- tables -------------------------------------------------------------
    cols = ["decoder", "params_M", "ms_per_chunk_median", "ms_per_chunk_p90",
            "ms_chunk0", "fps_equiv", "realtime_x", "s_per_6chunk_horizon",
            "speedup_vs_wan", "peak_vram_gb", "weights_vram_gb",
            "psnr_db", "lpips", "build_s"]
    import csv
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)

    hdr = ("| decoder | dec params | ms/chunk (med) | ms/chunk (p90) | fps-equiv | "
           "x realtime | 6-chunk horizon | speedup | peak VRAM | PSNR vs Wan | LPIPS vs Wan |")
    sep = "|" + "---|" * 11
    lines = [hdr, sep]
    for r in results:
        is_ref = r["decoder"] == "wan"
        q = "reference" if is_ref else "%.2f dB" % r["psnr_db"]
        l = "reference" if is_ref else "%.4f" % r["lpips"]
        lines.append(
            f"| `{r['decoder']}` | {r['params_M']:.1f} M | "
            f"{r['ms_per_chunk_median']:.1f} | {r['ms_per_chunk_p90']:.1f} | "
            f"{r['fps_equiv']:.0f} | {r['realtime_x']:.1f}x | "
            f"{r['s_per_6chunk_horizon']:.2f} s | {r['speedup_vs_wan']:.2f}x | "
            f"{r['peak_vram_gb']:.2f} GB | {q} | {l} |")
    table = "\n".join(lines)
    (out_dir / "results.md").write_text(table + "\n")
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print("\n" + table)
    print(f"\n[bench] artefacts in {out_dir}")


if __name__ == "__main__":
    main()
