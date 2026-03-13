#!/usr/bin/env python3
"""Pre-encode FrodoBots 2K videos with Wan 2.1 VAE; output per-episode Zarr stores."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


def expand_path(p: str) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(p))).resolve()


def require_ffmpeg() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg and ffprobe must be on PATH.")


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

@dataclass
class RideScheduleEntry:
    ride_ts: str
    path_2k: str
    ride_dir_2k: str
    zarr_start_row: int
    zarr_end_row: int


def load_ride_schedule(
    matches_csv: Path,
    markers_csv: Path,
    r2k_base: Optional[Path] = None,
) -> List[RideScheduleEntry]:
    markers: Dict[str, Tuple[int, int]] = {}
    with open(markers_csv, newline="") as f:
        for row in csv.DictReader(f):
            ts = row.get("ride_ts", "")
            if ts:
                markers[ts] = (int(row["start_row"]), int(row["end_row"]))

    schedule: List[RideScheduleEntry] = []
    with open(matches_csv, newline="") as f:
        for row in csv.DictReader(f):
            ts = row.get("ts", "")
            if ts not in markers:
                continue
            path_2k = row.get("path_2k", "")
            ride_dir_2k = row.get("ride_dir_2k", "")
            if r2k_base and path_2k and not os.path.isabs(path_2k):
                path_2k = str((r2k_base / path_2k).resolve())
            if r2k_base and ride_dir_2k and not os.path.isabs(ride_dir_2k):
                ride_dir_2k = str((r2k_base / ride_dir_2k).resolve())
            start_row, end_row = markers[ts]
            schedule.append(RideScheduleEntry(
                ride_ts=ts, path_2k=path_2k, ride_dir_2k=ride_dir_2k,
                zarr_start_row=start_row, zarr_end_row=end_row,
            ))
    return schedule


# ---------------------------------------------------------------------------
# Video loading
# ---------------------------------------------------------------------------

@dataclass
class VideoStreamInfo:
    fps: float
    width: int
    height: int


class VideoLoader:
    def __init__(self, scale: Tuple[int, int] = (832, 480)):
        self.scale = scale

    def _probe_input(self, video_path: str) -> VideoStreamInfo:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate", "-of", "json", video_path],
            capture_output=True, check=True, text=True,
        )
        data = json.loads(out.stdout)
        s = data.get("streams", [{}])[0]
        w, h = int(s.get("width", 0)), int(s.get("height", 0))
        fps = None
        rf = s.get("r_frame_rate", "")
        if "/" in rf:
            try:
                n, d = map(int, rf.split("/"))
                if d: fps = n / d
            except (ValueError, ZeroDivisionError):
                pass
        return VideoStreamInfo(fps=fps, width=w, height=h)

    def stream_blocks(
        self,
        video_path: str,
        block_size: int = 600,
        start_seconds: Optional[float] = None,
        duration_seconds: Optional[float] = None,
        max_frames: Optional[int] = None,
        frame_timeout: float = 60.0,
    ) -> Tuple[VideoStreamInfo, Generator[Tuple[torch.Tensor, np.ndarray], None, None]]:
        """Yields (frame_block_uint8, pts_timestamps). Block 0 has block_size+1 frames; later blocks overlap by 1 (keep 2 after block0, then 1)."""
        info = self._probe_input(video_path)
        out_w, out_h = self.scale
        yield_size = block_size + 1
        bpf = out_w * out_h * 3

        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "info", "-copyts", "-vsync", "0", "-i", video_path]
        if start_seconds is not None:
            cmd += ["-ss", str(float(start_seconds))]
        if duration_seconds is not None:
            cmd += ["-t", str(float(duration_seconds))]
        cmd += ["-an", "-vf", f"scale={out_w}:{out_h},showinfo", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]

        pts_re = re.compile(r"pts_time:([0-9]+\.[0-9]+|[0-9]+)")
        _SENTINEL = b"__EOF__"

        def gen() -> Generator[Tuple[torch.Tensor, np.ndarray], None, None]:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=bpf * 8)
            if proc.stdout is None or proc.stderr is None:
                raise RuntimeError("Failed to open ffmpeg pipes.")

            pts_q: queue.Queue[float] = queue.Queue(maxsize=10000)
            frame_q: queue.Queue[Optional[bytes]] = queue.Queue(maxsize=64)
            stderr_exc: List[Exception] = []

            def read_pts():
                try:
                    for line in iter(proc.stderr.readline, b""):
                        m = pts_re.search(line.decode("utf-8", "ignore"))
                        if m:
                            pts_q.put(float(m.group(1)))
                except Exception as e:
                    stderr_exc.append(e)

            def read_frames():
                try:
                    while True:
                        raw = proc.stdout.read(bpf)
                        if not raw or len(raw) < bpf:
                            frame_q.put(None)  # EOF
                            break
                        frame_q.put(raw)
                except Exception:
                    frame_q.put(None)

            threading.Thread(target=read_pts, daemon=True).start()
            threading.Thread(target=read_frames, daemon=True).start()

            frame_buf: List[bytes] = []
            ts_buf: List[float] = []
            idx = 0
            block_idx = 0

            def next_frame() -> Optional[bytes]:
                try:
                    return frame_q.get(timeout=frame_timeout)
                except queue.Empty:
                    raise RuntimeError(f"ffmpeg frame timeout ({frame_timeout}s); hung on {video_path}")

            def get_pts() -> float:
                try:
                    return pts_q.get(timeout=frame_timeout)
                except queue.Empty:
                    raise RuntimeError(f"PTS timeout ({frame_timeout}s); ffmpeg ret={proc.poll()}")

            try:
                while True:
                    if max_frames is not None and idx >= max_frames:
                        break
                    raw = next_frame()
                    if raw is None:
                        break
                    frame_buf.append(raw)
                    ts_buf.append(get_pts())
                    idx += 1

                    if len(frame_buf) == yield_size:
                        arr = np.frombuffer(b"".join(frame_buf), dtype=np.uint8).reshape(yield_size, out_h, out_w, 3)
                        block = torch.from_numpy(arr.copy()).permute(0, 3, 1, 2).contiguous()
                        yield block, np.asarray(ts_buf, dtype=np.float64)
                        keep = 2 if block_idx == 0 else 1
                        frame_buf = frame_buf[-keep:]
                        ts_buf = ts_buf[-keep:]
                        block_idx += 1

                if len(frame_buf) > 1:
                    n = len(frame_buf)
                    arr = np.frombuffer(b"".join(frame_buf), dtype=np.uint8).reshape(n, out_h, out_w, 3)
                    yield torch.from_numpy(arr.copy()).permute(0, 3, 1, 2).contiguous(), np.asarray(ts_buf, dtype=np.float64)

                ret = proc.wait(timeout=10.0)
                if ret not in (0, None, -9):
                    raise RuntimeError(f"ffmpeg exit {ret}")
                if stderr_exc:
                    raise RuntimeError(str(stderr_exc[0]))
            finally:
                with contextlib.suppress(Exception):
                    proc.kill()

        return info, gen()


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------

def load_vae(device: str, dtype: torch.dtype, vae_path: Optional[str] = None):
    try:
        from utils.wan_wrapper import WanVAEWrapper  # type: ignore
    except Exception as e:
        raise RuntimeError(f"WanVAEWrapper import failed: {e}")
    kwargs = {}
    if vae_path:
        p = Path(vae_path)
        kwargs["model_root"] = str(p.parent if p.suffix == ".pth" else p)
    vae = WanVAEWrapper(**kwargs) if kwargs else WanVAEWrapper()
    return vae.to(device=device, dtype=dtype).eval()


class VAEEncoder:
    def __init__(self, device: str = "cuda", model_dtype: torch.dtype = torch.float16, vae_path: Optional[str] = None):
        self.device = device
        self.model_dtype = model_dtype
        self.vae = load_vae(device, model_dtype, vae_path)
        torch.backends.cudnn.benchmark = True

    @torch.no_grad()
    def encode_block(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: [t,C,H,W] uint8 or float [0,1]. Returns latents [1,T,C,H,W]."""
        # Transfer uint8 to GPU first (4x less bandwidth), convert there
        if frames.dtype == torch.uint8:
            x = frames.to(device=self.device, dtype=self.model_dtype).div_(255.0)
        else:
            x = frames.to(device=self.device, dtype=self.model_dtype)
        x = x * 2.0 - 1.0
        # [t,C,H,W] -> [1,C,t,H,W]
        x = x.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()

        # Inline encode: avoid wrapper's .float() round-trip and O(n^2) torch.cat
        scale = [self.vae.mean.to(device=self.device, dtype=self.model_dtype),
                 1.0 / self.vae.std.to(device=self.device, dtype=self.model_dtype)]
        latent = self._fast_encode(self.vae.model, x.squeeze(0), scale)
        # [C,T,H,W] -> [1,T,C,H,W]
        return latent.unsqueeze(0).permute(0, 2, 1, 3, 4)

    @torch.no_grad()
    def _fast_encode(self, model, x: torch.Tensor, scale) -> torch.Tensor:
        """Encode [C,t,H,W] -> [C,T_lat,H',W']. Pre-allocates output to avoid O(n^2) cat."""
        x = x.unsqueeze(0)  # [1,C,t,H,W]
        model.clear_cache()
        t = x.shape[2]
        n_iter = 1 + (t - 1) // 4
        chunks = []
        for i in range(n_iter):
            model._enc_conv_idx = [0]
            if i == 0:
                chunks.append(model.encoder(x[:, :, :1, :, :], feat_cache=model._enc_feat_map, feat_idx=model._enc_conv_idx))
            else:
                chunks.append(model.encoder(x[:, :, 1 + 4*(i-1):1 + 4*i, :, :], feat_cache=model._enc_feat_map, feat_idx=model._enc_conv_idx))
        out = torch.cat(chunks, dim=2)
        mu = model.conv1(out).chunk(2, dim=1)[0]
        mu = (mu - scale[0].view(1, -1, 1, 1, 1)) * scale[1].view(1, -1, 1, 1, 1)
        model.clear_cache()
        return mu.squeeze(0)  # [C,T_lat,H',W']


def latents_to_time_major_numpy(latents: torch.Tensor, out_dtype: np.dtype) -> np.ndarray:
    """Wrapper returns [B, T, C, H, W]. Strip batch -> [T, C, H, W]."""
    t = latents
    if t.ndim == 5:
        t = t[0] if t.shape[0] == 1 else t.reshape(-1, *t.shape[2:])
    # t is now [T, C, H, W] — already time-major from wan_wrapper
    arr = t.detach().cpu().numpy()
    return arr if arr.dtype == out_dtype else arr.astype(out_dtype, copy=False)


# ---------------------------------------------------------------------------
# Zarr
# ---------------------------------------------------------------------------

def _zarr_blosc():
    import zarr  # type: ignore
    from numcodecs import Blosc  # type: ignore
    return zarr, Blosc


def make_blosc(Blosc, cname: str = "zstd", clevel: int = 1, shuffle: str = "bitshuffle"):
    sh = {"none": Blosc.NOSHUFFLE, "shuffle": Blosc.SHUFFLE, "bitshuffle": Blosc.BITSHUFFLE}
    if shuffle not in sh:
        raise ValueError(f"compressor_shuffle must be one of {list(sh)}")
    return Blosc(cname=cname, clevel=clevel, shuffle=sh[shuffle])


def zarr_has_latents(path: Path, name: str = "latents") -> bool:
    if not path.exists():
        return False
    try:
        z, _ = _zarr_blosc()
        g = z.open_group(str(path), mode="r")
        return name in g and g[name].shape[0] > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def parse_dtype(name: str) -> np.dtype:
    n = name.lower()
    if n in ("float16", "fp16", "f16"):
        return np.float16
    if n in ("float32", "fp32", "f32"):
        return np.float32
    if n in ("bfloat16", "bf16"):
        return np.dtype("bfloat16")
    raise ValueError("dtype must be float16 or float32")


def process_ride(
    entry: RideScheduleEntry,
    action_start_sec: float,
    action_end_sec: float,
    encoder: VAEEncoder,
    video_loader: VideoLoader,
    out_root: Path,
    overwrite: bool,
    encode_stride: int,
    zarr_latent_chunk: int,
    latent_out_dtype: np.dtype,
    compressor_cfg: dict,
) -> None:
    episode_id = entry.ride_ts
    out_zarr = out_root / f"{episode_id}.zarr"
    if not overwrite and zarr_has_latents(out_zarr):
        print(f"[SKIP] {episode_id}: already encoded at {out_zarr}")
        return
    video_path = entry.path_2k
    if not video_path or not os.path.exists(video_path):
        print(f"[SKIP] {episode_id}: video not found")
        return
    duration_sec = action_end_sec - action_start_sec
    if duration_sec <= 0:
        print(f"[SKIP] {episode_id}: bad duration")
        return

    if overwrite and out_zarr.exists():
        shutil.rmtree(out_zarr, ignore_errors=True)

    zarr, Blosc = _zarr_blosc()
    comp = make_blosc(Blosc, **compressor_cfg)
    g = zarr.open_group(str(out_zarr), mode="w")
    block_size = encode_stride + 1

    for k, v in [
        ("episode_id", episode_id), ("ride_ts", entry.ride_ts), ("source_video_2k", video_path),
        ("ride_dir_2k", entry.ride_dir_2k), ("zarr_start_row", entry.zarr_start_row), ("zarr_end_row", entry.zarr_end_row),
        ("action_start_sec", action_start_sec), ("action_end_sec", action_end_sec),
        ("encode_stride", encode_stride), ("video_block_size", block_size), ("zarr_latent_chunk", zarr_latent_chunk),
        ("expected_latents_per_full_block", 1 + encode_stride // 4), ("latent_dtype", str(latent_out_dtype)),
    ]:
        g.attrs[k] = v

    info, blocks = video_loader.stream_blocks(
        video_path, block_size=encode_stride, start_seconds=action_start_sec, duration_seconds=duration_sec,
    )
    g.attrs["fps"] = float(info.fps)
    print(f"[RIDE] {episode_id} {action_start_sec:.1f}s–{action_end_sec:.1f}s ({duration_sec:.1f}s) {info.width}x{info.height}")

    lat_arr = None
    ts_arr = None
    blk_i = 0
    ride_t0 = time.monotonic()
    for frames_block, ts_block in blocks:
        t_decode = time.monotonic() - (t_write_end if blk_i > 0 else ride_t0)
        print(f"  block {blk_i}: {frames_block.shape[0]}f {frames_block.dtype} "
              f"pts=[{ts_block[0]:.3f}..{ts_block[-1]:.3f}] decode={t_decode:.1f}s", flush=True)

        t0 = time.monotonic()
        lat = encoder.encode_block(frames_block)
        lat_np = latents_to_time_major_numpy(lat, latent_out_dtype)
        t_enc = time.monotonic() - t0

        t_lat, t_ts = lat_np.shape[0], ts_block.shape[0]
        if blk_i == 0:
            print(f"  latent shape={lat_np.shape}, dtype={lat_np.dtype}", flush=True)
        if lat_arr is None:
            lat_arr = g.create_dataset(
                "latents", shape=(0, *lat_np.shape[1:]), chunks=(zarr_latent_chunk, *lat_np.shape[1:]),
                dtype=latent_out_dtype, compressor=comp, overwrite=True,
            )
            ts_arr = g.create_dataset(
                "timestamps", shape=(0,), chunks=(block_size,), dtype=np.float64, compressor=comp, overwrite=True,
            )

        t0 = time.monotonic()
        lat_arr.resize(lat_arr.shape[0] + t_lat, *lat_arr.shape[1:])
        lat_arr[-t_lat:] = lat_np
        ts_arr.resize(ts_arr.shape[0] + t_ts)
        ts_arr[-t_ts:] = ts_block
        t_write = time.monotonic() - t0
        t_write_end = time.monotonic()

        print(f"  block {blk_i}: +{t_lat} lat (total {lat_arr.shape[0]}) "
              f"enc={t_enc:.1f}s write={t_write:.1f}s", flush=True)
        blk_i += 1

    elapsed = time.monotonic() - ride_t0
    if lat_arr is not None:
        print(f"[DONE] {episode_id}: {lat_arr.shape[0]} latents ({lat_arr.dtype}), "
              f"{ts_arr.shape[0]} ts, {blk_i} blocks in {elapsed:.1f}s -> {out_zarr}")


def main() -> None:
    # Ensure project root is on path so "utils" package can be imported
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    p = argparse.ArgumentParser(description="Pre-encode FrodoBots 2K with Wan VAE -> per-ride Zarr.")
    p.add_argument("--matches_csv", default="overlap_checks/matches.csv", help="2K↔7K matches CSV")
    p.add_argument("--markers_csv", default="overlap_checks/actual_episode_markers.csv", help="Zarr ride boundaries")
    p.add_argument("--zarr_root", default="/home/ashish/fbots7k/extracted/frodobots_dataset", help="7K zarr root")
    p.add_argument("--r2k_base", default="/home/ashish/frodobots", help="Base for relative 2K paths")
    p.add_argument("--output_root", default="/home/ashish/frodobots/frodobots_encoded", help="Output dir; writes <root>/<ride_ts>.zarr/")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing zarr")
    p.add_argument("--vae_path", default="/home/ashish/Wan2.1/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--model_dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--latent_dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--scale", default="832x480", help="WxH")
    p.add_argument("--encode_stride", type=int, default=600, help="Block stride; block has encode_stride+1 frames")
    p.add_argument("--zarr_latent_chunk", type=int, default=32)
    p.add_argument("--compressor_cname", default="zstd")
    p.add_argument("--compressor_level", type=int, default=1)
    p.add_argument("--compressor_shuffle", default="bitshuffle", choices=["none", "shuffle", "bitshuffle"])
    args = p.parse_args()

    if "x" not in args.scale:
        raise ValueError("--scale must be WxH")
    scale_w, scale_h = map(int, args.scale.split("x"))

    matches_csv = expand_path(args.matches_csv)
    markers_csv = expand_path(args.markers_csv)
    out_root = expand_path(args.output_root)
    safe_mkdir(out_root)
    r2k_base = expand_path(args.r2k_base)

    schedule = load_ride_schedule(matches_csv, markers_csv, r2k_base=r2k_base)
    if not schedule:
        print("No rides to process.")
        return
    print(f"Schedule: {len(schedule)} rides")

    from utils.frodobots_zarr_loader import load_full_dataset  # type: ignore
    zarr_root = args.zarr_root or os.environ.get("FRODOBOTS_DATASET_ROOT", str(Path.home() / "fbots7k" / "extracted" / "frodobots_dataset"))
    ds = load_full_dataset(zarr_root)
    ts_zarr = ds.zarr["observation.images.front.timestamp"]

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("WARNING: using CPU")
    require_ffmpeg()

    encoder = VAEEncoder(device=device, model_dtype=torch.float16 if args.model_dtype == "float16" else torch.float32, vae_path=args.vae_path)
    video_loader = VideoLoader(scale=(scale_w, scale_h))
    comp_cfg = {"cname": args.compressor_cname, "clevel": args.compressor_level, "shuffle": args.compressor_shuffle}

    done = errors = 0
    for entry in tqdm(schedule, desc="Rides"):
        try:
            action_start_sec = float(ts_zarr[entry.zarr_start_row])
            action_end_sec = float(ts_zarr[entry.zarr_end_row - 1])
        except Exception as e:
            print(f"[ERROR] {entry.ride_ts}: {e}")
            errors += 1
            continue
        try:
            process_ride(
                entry, action_start_sec, action_end_sec, encoder, video_loader, out_root,
                args.overwrite, args.encode_stride, args.zarr_latent_chunk,
                parse_dtype(args.latent_dtype), comp_cfg,
            )
            done += 1
        except Exception as e:
            print(f"[ERROR] {entry.ride_ts}: {e}")
            errors += 1
    print(f"Done: {done}, Errors: {errors}")


if __name__ == "__main__":
    main()
