"""ZarrSequentialDataset: sequential video latent dataset from frodobots zarr files.

Each zarr encodes a single ride with latents (T, 16, 60, 104), timestamps, and attrs
containing ride_dir_2k, action_start_sec, fps, etc.

Motion is loaded from motion.npy, aligned with the 0.8s delay, encoded through the
ss_vae to 8D latents, tanh-squashed per dimension, then subsampled to latent frame rate.

The dataset builds a flat list of (zarr_path, window_start) tuples ordered sequentially
within each ride, so DistributedSampler(shuffle=False) naturally iterates rides in order.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import zarr as zarr_lib
from torch.utils.data import Dataset

from action_query.ss_vae_model import load_ss_vae

# ---------------------------------------------------------------------------
# Hardcoded constants (matching test_zarr_chunks.py)
# ---------------------------------------------------------------------------

# The ride_dir_2k paths inside zarr attrs are absolute paths on the original
# machine; we strip this prefix to get the relative path used for motion/caption lookup.
_DATA_ROOT = Path("/home/ashish/frodobots/frodobots_data")

# Latent temporal compression: 1 latent frame corresponds to 4 video frames.
_LATENT_TO_VIDEO = 4

# Motion window size in video frames (12-frame CoTracker windows).
_MOTION_WINDOW_FRAMES = 12

# ss_vae encoding batch size.
_ENCODE_BATCH = 128

# Per-dimension tanh squash scales: tanh(z / scale) -> asymptotic ±1.
# z2 = turn/sides (raw range ≈ ±25), z7 = fwd/back (raw range ≈ ±10).
_ZACTION_SCALES = torch.full((8,), 25.0, dtype=torch.float32)
_ZACTION_SCALES[2] = 25.0
_ZACTION_SCALES[7] = 10.0


def _format_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.2f} GiB"


def _read_proc_status_memory() -> Tuple[Optional[int], Optional[int]]:
    """Return current RSS and high-water mark from /proc/self/status in bytes."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError:
        return None, None

    vmrss_bytes: Optional[int] = None
    vmhwm_bytes: Optional[int] = None

    for line in lines:
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2:
                vmrss_bytes = int(parts[1]) * 1024
        elif line.startswith("VmHWM:"):
            parts = line.split()
            if len(parts) >= 2:
                vmhwm_bytes = int(parts[1]) * 1024

    return vmrss_bytes, vmhwm_bytes


def _build_memory_log_message() -> str:
    parts = []

    rss_bytes, hwm_bytes = _read_proc_status_memory()
    if rss_bytes is not None:
        parts.append(f"RSS={_format_gib(rss_bytes)}")
    if hwm_bytes is not None:
        parts.append(f"PeakRSS={_format_gib(hwm_bytes)}")

    if torch.cuda.is_available():
        try:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            used_bytes = total_bytes - free_bytes
            parts.append(
                "GPU="
                f"{_format_gib(used_bytes)} used / "
                f"{_format_gib(free_bytes)} free / "
                f"{_format_gib(total_bytes)} total"
            )
        except Exception:
            pass

    if not parts:
        return "memory stats unavailable"
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Motion helpers (ported from utils/test_zarr_chunks.py)
# ---------------------------------------------------------------------------

def _load_aligned_motion_for_zarr(
    attrs: dict,
    n_video_frames: int,
    motion_root: Path,
) -> np.ndarray:
    """Upsample motion.npy to per-video-frame resolution.

    Preserves the hardcoded 0.8s action delay and 12-frame motion window alignment
    from test_zarr_chunks.py lines 85-135.

    Returns float32 array of shape (n_video_frames, 100, 3).
    """
    ride_dir_2k = attrs.get("ride_dir_2k", "")
    if not ride_dir_2k:
        raise RuntimeError("ride_dir_2k not present in zarr attrs")

    ride_path = Path(ride_dir_2k)
    try:
        rel = ride_path.relative_to(_DATA_ROOT)
    except ValueError:
        raise RuntimeError(
            f"ride_dir_2k={ride_dir_2k} is not under {_DATA_ROOT}"
        )

    motion_path = motion_root / rel / "motion.npy"
    if not motion_path.exists():
        raise FileNotFoundError(f"motion.npy not found at {motion_path}")

    motion = np.load(motion_path)  # [M, N, 3]
    if motion.ndim != 3 or motion.shape[2] != 3:
        raise RuntimeError(f"Unexpected motion shape {motion.shape}")

    action_start_sec = float(attrs.get("action_start_sec", 0.0))
    fps = float(attrs.get("fps", 20.0))

    # 0.8s delay preserved from test_zarr_chunks.py line 117
    offset_frames_prelim = int((action_start_sec - 0.8) * fps)
    offset_windows = offset_frames_prelim // _MOTION_WINDOW_FRAMES
    partial_first = _MOTION_WINDOW_FRAMES - (offset_frames_prelim % _MOTION_WINDOW_FRAMES)
    n_rest_windows = (n_video_frames - partial_first + _MOTION_WINDOW_FRAMES - 1) // _MOTION_WINDOW_FRAMES

    first_motion = np.repeat(
        motion[offset_windows: offset_windows + 1], partial_first, axis=0
    )
    rest_motion = np.repeat(
        motion[offset_windows + 1: offset_windows + 1 + n_rest_windows],
        _MOTION_WINDOW_FRAMES,
        axis=0,
    )
    per_frame = np.concatenate([first_motion, rest_motion], axis=0)[:n_video_frames]

    # Pad if motion data is shorter than required
    if per_frame.shape[0] < n_video_frames:
        reps = n_video_frames - per_frame.shape[0]
        per_frame = np.concatenate(
            [per_frame, np.repeat(per_frame[-1:], reps, axis=0)], axis=0
        )

    return per_frame.astype(np.float32)  # (n_video_frames, 100, 3)


def _encode_motion_ss_vae(
    motion_per_frame: np.ndarray,
    model,
    scale: float,
    device: str,
    batch_size: int = _ENCODE_BATCH,
) -> np.ndarray:
    """Encode motion (n, 100, 3) to latent mu (n, latent_ch) via ss_vae encoder.

    Ported from test_zarr_chunks.py lines 180-197.
    dx/dy are used; visibility is ignored.
    """
    n = motion_per_frame.shape[0]
    # Reshape (n, 100, 3) -> (n, 10, 10, 2) taking only dx, dy
    xy = motion_per_frame[:, :, :2].reshape(n, 10, 10, 2)
    x = torch.from_numpy(xy).permute(0, 3, 1, 2).float() / scale  # (n, 2, 10, 10)

    zs = []
    model.eval()
    with torch.no_grad():
        for s in range(0, n, batch_size):
            mu, _ = model.encoder(x[s: s + batch_size].to(device))
            zs.append(mu.squeeze(-1).squeeze(-1).cpu().numpy())  # (B, latent_ch)
    return np.concatenate(zs, axis=0)  # (n, latent_ch)


def _tanh_squash(z_raw: torch.Tensor) -> torch.Tensor:
    """Apply per-dimension tanh squash.  Output is in (-1, 1)."""
    scales = _ZACTION_SCALES.to(z_raw.device)
    return torch.tanh(z_raw / scales)


# ---------------------------------------------------------------------------
# Caption helpers
# ---------------------------------------------------------------------------

def _find_encoded_caption(caption_root: Path, rel_ride_path: Path) -> Optional[Path]:
    """Find the *_encoded.json caption file for a ride.

    caption_root layout: {caption_root}/{output_rides_X}/{ride_name}/*_encoded.json
    rel_ride_path: output_rides_X/ride_ID_TS  (relative from data_root)
    """
    ride_caption_dir = caption_root / rel_ride_path
    if not ride_caption_dir.exists():
        return None
    candidates = sorted(ride_caption_dir.glob("*_encoded.json"))
    return candidates[0] if candidates else None


def _load_prompt_embeds(encoded_json: Path) -> torch.Tensor:
    """Load pre-encoded caption embeddings from JSON.  Returns (seq_len, dim) float32."""
    with encoded_json.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    embedding = payload.get("caption_encoded")
    if embedding is None:
        raise ValueError(f"'caption_encoded' missing in {encoded_json}")
    t = torch.tensor(embedding, dtype=torch.float32)
    if t.ndim != 2:
        raise ValueError(f"Expected 2D embedding, got {t.shape} from {encoded_json}")
    return t


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

_MANIFEST_VERSION = 2


def _index_single_zarr(
    zpath: Path,
    caption_root: Path,
) -> Tuple[torch.Tensor, dict, int]:
    """Read metadata and captions for one zarr — no motion encoding."""
    g = zarr_lib.open_group(str(zpath), mode="r")
    attrs = dict(g.attrs)
    n_latent_frames = g["latents"].shape[0]

    ride_dir_2k = attrs.get("ride_dir_2k", "")
    if not ride_dir_2k:
        raise RuntimeError("ride_dir_2k missing from zarr attrs")
    ride_path = Path(ride_dir_2k)
    try:
        ride_path.relative_to(_DATA_ROOT)
    except ValueError:
        raise RuntimeError(f"ride_dir_2k={ride_dir_2k} not under {_DATA_ROOT}")

    caption_file = _find_encoded_caption(caption_root, ride_path.relative_to(_DATA_ROOT))
    if caption_file is None:
        raise FileNotFoundError(f"No encoded caption for {ride_path}")
    prompt_embeds = _load_prompt_embeds(caption_file)

    return prompt_embeds, attrs, n_latent_frames


def build_ride_manifest(
    encoded_root: str,
    caption_root: str,
    min_ride_frames: int = 21,
    cache_path: Optional[str] = None,
) -> List[dict]:
    """Scan all zarr rides and return a manifest list, with optional disk caching.

    Each entry is ``{"zarr_path": str, "prompt_embeds": Tensor, "attrs": dict,
    "n_latent_frames": int}``.  The list is sorted by zarr filename.

    If *cache_path* points to a valid manifest whose *encoded_root* and zarr
    file count match the current directory, it is loaded directly (typically
    <0.5 s vs. minutes for a fresh scan).
    """
    enc = Path(encoded_root)
    zarr_paths = sorted(enc.glob("*.zarr"))
    n_zarr = len(zarr_paths)

    if cache_path:
        try:
            cached = torch.load(cache_path, map_location="cpu")
            if (
                cached.get("version") == _MANIFEST_VERSION
                and cached.get("encoded_root") == str(enc)
                and cached.get("n_zarr_files") == n_zarr
            ):
                rides = cached["rides"]
                logging.info(
                    "Loaded ride manifest from cache (%d rides, %d zarr files): %s",
                    len(rides), n_zarr, cache_path,
                )
                return rides
            logging.info("Manifest cache stale (root/count mismatch), rebuilding.")
        except Exception:
            logging.info("No usable manifest cache at %s, building from scratch.", cache_path)

    cap_root = Path(caption_root)
    logging.info("Scanning %d zarr files in %s (ride-level)", n_zarr, enc)
    t0 = time.perf_counter()
    rides: List[dict] = []
    skipped = 0
    for zpath in zarr_paths:
        try:
            prompt_embeds, zarr_attrs, n_lat = _index_single_zarr(zpath, cap_root)
        except Exception as exc:
            logging.warning("Skipping %s: %s", zpath.name, exc)
            skipped += 1
            continue
        if n_lat < min_ride_frames:
            skipped += 1
            continue
        rides.append({
            "zarr_path": str(zpath),
            "prompt_embeds": prompt_embeds,
            "attrs": zarr_attrs,
            "n_latent_frames": n_lat,
        })
    elapsed = time.perf_counter() - t0
    logging.info(
        "Ride manifest: %d rides indexed (%d skipped) in %.1fs (%.2fs/ride)",
        len(rides), skipped, elapsed, elapsed / max(len(rides), 1),
    )
    if not rides:
        raise RuntimeError("No valid rides found. Check encoded_root, caption_root, motion_root.")

    if cache_path:
        try:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "version": _MANIFEST_VERSION,
                "encoded_root": str(enc),
                "n_zarr_files": n_zarr,
                "rides": rides,
            }, cache_path)
            logging.info("Saved ride manifest cache to %s", cache_path)
        except Exception as exc:
            logging.warning("Failed to save manifest cache: %s", exc)

    return rides


class ZarrRideDataset(Dataset):
    """Ride-level zarr dataset for streaming training.

    One sample per ride (not per window).  Returns ride metadata and
    full-ride z_actions so the trainer can slice arbitrary chunks during
    the streaming loop.  Latents are loaded lazily via ``load_latent_chunk``.

    Motion encoding through the ss_vae is deferred until a ride is first
    accessed (``__getitem__``), so dataset construction only validates
    metadata and is fast regardless of how many rides exist.

    Construct either via ``__init__`` (full scan) or via the faster
    ``from_manifest`` class method (pre-built ride list, no scan).
    """

    def __init__(
        self,
        encoded_root: str,
        caption_root: str,
        motion_root: str,
        ss_vae_checkpoint: str,
        min_ride_frames: int = 21,
        device: str = "cpu",
        ss_vae_device: Optional[str] = None,
        start_zarr_index: int = 0,
        max_rides: Optional[int] = None,
    ):
        self.encoded_root = Path(encoded_root)
        self.caption_root = Path(caption_root)
        self.motion_root = Path(motion_root)
        self.min_ride_frames = min_ride_frames
        self.start_zarr_index = start_zarr_index
        self.max_rides = max_rides

        ss_dev = ss_vae_device or device
        logging.info("Loading ss_vae from %s on %s", ss_vae_checkpoint, ss_dev)
        ss_vae_model, ss_scale = load_ss_vae(ss_vae_checkpoint, ss_dev)
        self._ss_vae = ss_vae_model
        self._ss_scale = float(ss_scale)
        self._ss_dev = ss_dev

        self._rides: List[Tuple[Path, torch.Tensor, dict, int]] = []
        self._attrs_by_path: dict = {}
        self._build_index()

    @classmethod
    def from_manifest(
        cls,
        rides_data: List[dict],
        motion_root: str,
        ss_vae_checkpoint: str,
        device: str = "cpu",
        ss_vae_device: Optional[str] = None,
        _share_ss_vae: Optional["ZarrRideDataset"] = None,
    ) -> "ZarrRideDataset":
        """Construct from a pre-built manifest (no zarr scan needed).

        Pass ``_share_ss_vae`` to reuse another dataset's ss_vae model
        instead of loading a second copy.
        """
        obj = object.__new__(cls)
        obj.motion_root = Path(motion_root)
        obj.encoded_root = Path(rides_data[0]["zarr_path"]).parent if rides_data else Path(".")
        obj.caption_root = Path(".")
        obj.min_ride_frames = 0
        obj.start_zarr_index = 0
        obj.max_rides = None

        if _share_ss_vae is not None:
            obj._ss_vae = _share_ss_vae._ss_vae
            obj._ss_scale = _share_ss_vae._ss_scale
            obj._ss_dev = _share_ss_vae._ss_dev
        else:
            ss_dev = ss_vae_device or device
            logging.info("Loading ss_vae from %s on %s", ss_vae_checkpoint, ss_dev)
            ss_vae_model, ss_scale = load_ss_vae(ss_vae_checkpoint, ss_dev)
            obj._ss_vae = ss_vae_model
            obj._ss_scale = float(ss_scale)
            obj._ss_dev = ss_dev

        obj._rides = []
        obj._attrs_by_path = {}
        for r in rides_data:
            zpath = Path(r["zarr_path"])
            obj._rides.append((zpath, r["prompt_embeds"], r["attrs"], r["n_latent_frames"]))
            obj._attrs_by_path[r["zarr_path"]] = r["attrs"]

        logging.info("ZarrRideDataset.from_manifest: %d rides loaded (no scan)", len(obj._rides))
        return obj

    def _build_index(self) -> None:
        zarr_paths = sorted(self.encoded_root.glob("*.zarr"))
        if zarr_paths and self.start_zarr_index:
            start = self.start_zarr_index % len(zarr_paths)
            zarr_paths = zarr_paths[start:] + zarr_paths[:start]
        logging.info("Scanning %d zarr files in %s (ride-level)", len(zarr_paths), self.encoded_root)

        t_index_start = time.perf_counter()
        skipped = 0
        for zpath in zarr_paths:
            if self.max_rides is not None and len(self._rides) >= self.max_rides:
                logging.info("Reached ride cap (%d); stopping index build.", self.max_rides)
                break
            try:
                prompt_embeds, zarr_attrs, n_latent_frames = _index_single_zarr(zpath, self.caption_root)
            except Exception as exc:
                logging.warning("Skipping %s: %s", zpath.name, exc)
                skipped += 1
                continue

            if n_latent_frames < self.min_ride_frames:
                skipped += 1
                continue

            self._rides.append((zpath, prompt_embeds, zarr_attrs, n_latent_frames))
            self._attrs_by_path[str(zpath)] = zarr_attrs

        elapsed = time.perf_counter() - t_index_start
        logging.info(
            "ZarrRideDataset: %d rides indexed (%d skipped) in %.1fs (%.2fs/ride)",
            len(self._rides), skipped, elapsed,
            elapsed / max(len(self._rides), 1),
        )
        if not self._rides:
            raise RuntimeError("No valid rides found. Check encoded_root, caption_root, motion_root.")

    def encode_z_actions_window(
        self,
        zarr_path: str,
        n_latent_frames: int,
        latent_start: int,
        latent_end: int,
    ) -> torch.Tensor:
        """Encode motion only for the latent frames in ``[latent_start, latent_end)``.

        Loads the full motion.npy (cheap, ~3 ms), then subsamples to the
        video-frame indices that correspond to the requested latent window
        before pushing through the ss_vae.  For a 21-frame window this
        encodes ~21 frames instead of the full ride (often 10 000+).

        Returns ``[latent_end - latent_start, z_dim]`` float32 tensor.
        """
        zarr_attrs = self._attrs_by_path[zarr_path]
        n_video_frames = 1 + _LATENT_TO_VIDEO * (n_latent_frames - 1)

        t0 = time.perf_counter()
        motion_all = _load_aligned_motion_for_zarr(
            zarr_attrs, n_video_frames, self.motion_root,
        )
        t_loaded = time.perf_counter()

        n_out = latent_end - latent_start
        if motion_all.shape[0] == 0:
            logging.warning(
                "  z_actions [%d:%d]: empty motion for %s, returning zeros",
                latent_start, latent_end, zarr_path,
            )
            return torch.zeros(n_out, 8, dtype=torch.float32)

        vid_indices = np.arange(latent_start, latent_end) * _LATENT_TO_VIDEO
        vid_indices = np.clip(vid_indices, 0, max(motion_all.shape[0] - 1, 0))
        motion_window = motion_all[vid_indices]

        z_window = _encode_motion_ss_vae(
            motion_window, self._ss_vae, self._ss_scale, self._ss_dev,
        )
        t_encoded = time.perf_counter()

        z_tensor = torch.from_numpy(z_window)
        z_squashed = _tanh_squash(z_tensor)

        logging.info(
            "  z_actions [%d:%d]: encode %d frames | "
            "motion_load=%.3fs  ss_vae=%.3fs  total=%.3fs",
            latent_start, latent_end, len(vid_indices),
            t_loaded - t0, t_encoded - t_loaded, time.perf_counter() - t0,
        )
        return z_squashed

    def __len__(self) -> int:
        return len(self._rides)

    def __getitem__(self, idx: int) -> dict:
        zpath, prompt_embeds, _attrs, n_frames = self._rides[idx]
        return {
            "zarr_path": str(zpath),
            "prompt_embeds": prompt_embeds,
            "n_latent_frames": n_frames,
        }

    @staticmethod
    def load_latent_chunk(zarr_path: str, start: int, end: int) -> torch.Tensor:
        """Lazy-load a latent slice ``[start:end]`` from a zarr ride file."""
        g = zarr_lib.open_group(zarr_path, mode="r")
        lat_np = g["latents"][start:end]
        return torch.from_numpy(lat_np.astype(np.float32))


class ZarrSequentialDataset(Dataset):
    """Sequential video latent dataset backed by frodobots zarr files.

    Each item is a sliding window of `window_size` consecutive latent frames
    from a single ride, returned in ride order.

    Args:
        encoded_root: directory containing `*.zarr` ride files.
        caption_root: directory tree with `*_encoded.json` caption files.
        motion_root: directory tree with `motion.npy` motion files.
        ss_vae_checkpoint: path to the ss_vae_8free.pt checkpoint.
        window_size: number of latent frames per sample (default 21).
        window_stride: step between successive windows within a ride (default 1).
        device: device for ss_vae inference (default "cuda" if available).
        ss_vae_device: override device specifically for ss_vae encoding at init.
    """

    def __init__(
        self,
        encoded_root: str,
        caption_root: str,
        motion_root: str,
        ss_vae_checkpoint: str,
        window_size: int = 21,
        window_stride: int = 1,
        device: str = "cpu",
        ss_vae_device: Optional[str] = None,
        log_every_n_validated: Optional[int] = None,
        start_zarr_index: int = 0,
        max_samples: Optional[int] = None,
        context_frames: int = 3,
    ):
        self.encoded_root = Path(encoded_root)
        self.caption_root = Path(caption_root)
        self.motion_root = Path(motion_root)
        self.window_size = window_size
        self.window_stride = window_stride
        self.context_frames = context_frames
        self.log_every_n_validated = log_every_n_validated
        self.start_zarr_index = start_zarr_index
        self.max_samples = max_samples

        ss_dev = ss_vae_device or device
        logging.info("Loading ss_vae from %s on %s", ss_vae_checkpoint, ss_dev)
        ss_vae_model, ss_scale = load_ss_vae(ss_vae_checkpoint, ss_dev)
        self._ss_vae = ss_vae_model
        self._ss_scale = float(ss_scale)
        self._ss_dev = ss_dev

        # Build index: list of (zarr_path, prompt_embeds_tensor, z_actions_tensor, window_start)
        self._samples: List[Tuple[Path, torch.Tensor, torch.Tensor, int]] = []
        self._build_index()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        zarr_paths = sorted(self.encoded_root.glob("*.zarr"))
        if zarr_paths and self.start_zarr_index:
            start = self.start_zarr_index % len(zarr_paths)
            zarr_paths = zarr_paths[start:] + zarr_paths[:start]
        logging.info("Scanning %d zarr files in %s", len(zarr_paths), self.encoded_root)

        skipped = 0
        total_windows = 0
        validated = 0

        for zpath in zarr_paths:
            try:
                prompt_embeds, z_actions_latent, n_latent_frames = self._process_zarr(zpath)
            except Exception as exc:
                logging.warning("Skipping %s: %s", zpath.name, exc)
                skipped += 1
                continue

            # Slide window over latent frames (account for context prepended to each window)
            max_start = n_latent_frames - (self.window_size + self.context_frames)
            if max_start < 0:
                skipped += 1
                continue

            windows = list(range(0, max_start + 1, self.window_stride))
            if self.max_samples is not None:
                remaining = self.max_samples - len(self._samples)
                if remaining <= 0:
                    logging.info(
                        "Reached requested sample cap (%d); stopping index build early.",
                        self.max_samples,
                    )
                    return
                windows = windows[:remaining]

            for start in windows:
                self._samples.append((zpath, prompt_embeds, z_actions_latent, start))

            total_windows += len(windows)
            validated += 1

            if self.log_every_n_validated and validated % self.log_every_n_validated == 0:
                logging.info(
                    "Validated %d rides so far (%d skipped, %d windows) | %s",
                    validated,
                    skipped,
                    total_windows,
                    _build_memory_log_message(),
                )

            if self.max_samples is not None and len(self._samples) >= self.max_samples:
                logging.info(
                    "Reached requested sample cap (%d); stopping index build early.",
                    self.max_samples,
                )
                return

        logging.info(
            "ZarrSequentialDataset: %d windows from %d rides (%d skipped)",
            total_windows,
            len(zarr_paths) - skipped,
            skipped,
        )
        if not self._samples:
            raise RuntimeError(
                "No valid samples found. Check encoded_root, caption_root, motion_root."
            )

    def _process_zarr(
        self, zpath: Path
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Load and preprocess a single zarr ride.

        Returns:
            prompt_embeds: (seq_len, dim) float32
            z_actions_latent: (n_latent_frames, 8) float32, tanh-squashed
            n_latent_frames: number of latent frames in this ride
        """
        g = zarr_lib.open_group(str(zpath), mode="r")
        attrs = dict(g.attrs)
        lat_ds = g["latents"]
        n_latent_frames = lat_ds.shape[0]  # (T, 16, 60, 104)

        # Derive relative ride path for caption/motion lookup
        ride_dir_2k = attrs.get("ride_dir_2k", "")
        if not ride_dir_2k:
            raise RuntimeError("ride_dir_2k missing from zarr attrs")
        ride_path = Path(ride_dir_2k)
        try:
            rel = ride_path.relative_to(_DATA_ROOT)
        except ValueError:
            raise RuntimeError(f"ride_dir_2k={ride_dir_2k} not under {_DATA_ROOT}")

        # Caption
        caption_file = _find_encoded_caption(self.caption_root, rel)
        if caption_file is None:
            raise FileNotFoundError(f"No encoded caption for {rel}")
        prompt_embeds = _load_prompt_embeds(caption_file)

        # Video frame count corresponding to all latent frames:
        # n_video = 1 + 4*(n_lat - 1)  (from test_zarr_chunks.py lines 393-398)
        n_video_frames = 1 + _LATENT_TO_VIDEO * (n_latent_frames - 1)

        # Motion -> per-video-frame -> encode -> per-latent-frame
        motion_per_video = _load_aligned_motion_for_zarr(
            attrs, n_video_frames, self.motion_root
        )  # (n_video_frames, 100, 3)

        z_per_video = _encode_motion_ss_vae(
            motion_per_video, self._ss_vae, self._ss_scale, self._ss_dev
        )  # (n_video_frames, 8) float32

        # Subsample to latent frame rate: take frame indices 0, 4, 8, ...
        latent_indices = np.arange(n_latent_frames) * _LATENT_TO_VIDEO
        latent_indices = np.clip(latent_indices, 0, n_video_frames - 1)
        z_per_latent = z_per_video[latent_indices]  # (n_latent_frames, 8)

        # Tanh squash
        z_tensor = torch.from_numpy(z_per_latent)  # (n_latent_frames, 8)
        z_squashed = _tanh_squash(z_tensor)

        return prompt_embeds, z_squashed, n_latent_frames

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        zpath, prompt_embeds, z_actions_latent, start = self._samples[idx]

        end = start + self.window_size + self.context_frames

        # Load latents lazily from zarr (window + leading context)
        g = zarr_lib.open_group(str(zpath), mode="r")
        lat_np = g["latents"][start:end]  # (window_size+context_frames, 16, 60, 104)
        latents = torch.from_numpy(lat_np.astype(np.float32))

        z_window = z_actions_latent[start:end]  # (window_size+context_frames, 8)

        return {
            "real_latents": latents,          # (window_size+context_frames, 16, 60, 104)
            "prompt_embeds": prompt_embeds,   # (seq_len, dim)
            "z_actions": z_window,            # (window_size+context_frames, 8)
        }
