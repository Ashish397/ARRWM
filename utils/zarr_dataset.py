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


def _extract_ride_rel(ride_dir_2k: str) -> Path:
    """Extract the relative ride path (output_rides_X/ride_Y) from any absolute prefix."""
    parts = Path(ride_dir_2k).parts
    for i, part in enumerate(parts):
        if part.startswith("output_rides_"):
            return Path(*parts[i:])
    try:
        return Path(ride_dir_2k).relative_to(_DATA_ROOT)
    except ValueError:
        raise RuntimeError(
            f"ride_dir_2k={ride_dir_2k}: cannot find output_rides_*/ride_* "
            f"and is not under {_DATA_ROOT}"
        )

# Latent temporal compression: 1 latent frame corresponds to 4 video frames
# in the Wan VAE — except for the SPECIAL HEAD LATENT (zarr index 0) which
# encodes a single video frame. We drop that head latent in the loader so
# every reported latent index maps to a clean 4-video-frame window. After
# the drop: dataset latent index k <-> Wan zarr latent index k+1 <-> video
# frames [4k+1, 4k+5) (in the original / pre-encode-motion coordinate
# system; see ``_load_aligned_motion_for_zarr`` for the alignment with
# motion.npy's "first video frame dropped" convention).
_LATENT_TO_VIDEO = 4

# Motion window size in video frames (CoTracker emits one motion entry per
# 12-frame disjoint window; see utils/pre_encode_motion.py:212-213
# (``output_chunk_size=12``)). With ``_LATENT_TO_VIDEO=4``, that's exactly
# ``_LATENTS_PER_MOTION_CHUNK = 3`` post-drop dataset latents per motion
# entry — i.e. one motion entry encodes one 3-latent training chunk.
_MOTION_WINDOW_FRAMES = 12
_LATENTS_PER_MOTION_CHUNK = _MOTION_WINDOW_FRAMES // _LATENT_TO_VIDEO  # = 3

# Number of leading Wan zarr latents to drop in the loader so dataset
# latent index 0 corresponds to a clean 4-video-frame window aligned with
# motion entry 0. v14's pre_encode_motion drops video frame 0 (line 153);
# ARRWM's pre_encode_local does NOT, so we drop the head latent here in
# the loader to recover v14's discipline without re-encoding rides.
_LATENT_HEAD_DROP = 1

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

def _resolve_motion_path(attrs: dict, motion_root: Path) -> Path:
    ride_dir_2k = attrs.get("ride_dir_2k", "")
    if not ride_dir_2k:
        raise RuntimeError("ride_dir_2k not present in zarr attrs")
    rel = _extract_ride_rel(ride_dir_2k)
    motion_path = motion_root / rel / "motion.npy"
    if not motion_path.exists():
        raise FileNotFoundError(f"motion.npy not found at {motion_path}")
    return motion_path


def _peek_motion_chunks(attrs: dict, motion_root: Path) -> int:
    """Return motion.npy's chunk count (= shape[0]) without loading the body.
    Used at manifest-build time to cap n_latent_frames at min(zarr-1,
    n_motion_chunks * 3) so the trainer never sees stale-padded motion.
    """
    motion_path = _resolve_motion_path(attrs, motion_root)
    motion = np.load(motion_path, mmap_mode="r")
    if motion.ndim != 3 or motion.shape[2] != 3:
        raise RuntimeError(f"Unexpected motion shape {motion.shape} at {motion_path}")
    return int(motion.shape[0])


def _load_aligned_motion_for_zarr(
    attrs: dict,
    n_latent_frames: int,
    motion_root: Path,
) -> np.ndarray:
    """Return per-DATASET-LATENT motion mapped from chunk-grain motion.npy.

    v14 discipline (see utils/pre_encode_motion.py:153 for the dropped
    first video frame, and the spec at the top of this module for the
    head-latent drop): each motion entry covers exactly
    ``_LATENTS_PER_MOTION_CHUNK`` (= 3) consecutive dataset latents, so
    dataset latent index ``k`` reads from ``motion[k // 3]``. No
    upsampling, no ``action_start_sec - 0.8`` offset, no stale-padding —
    if the caller asks for more latents than the motion file supports
    we raise: the caller must cap ``n_latent_frames`` at
    ``min(zarr_latents - _LATENT_HEAD_DROP, n_motion_chunks * 3)`` (the
    manifest indexer does this; ``_load_ride_tensors`` inherits the cap
    via ``meta["n_latent_frames"]``).

    Returns float32 array of shape ``(n_latent_frames, 100, 3)``. Within
    each chunk the 3 entries are identical (= the chunk's motion.npy
    row); across chunks they advance with the ride.
    """
    motion_path = _resolve_motion_path(attrs, motion_root)
    motion = np.load(motion_path)  # [n_motion_chunks, N, 3]
    if motion.ndim != 3 or motion.shape[2] != 3:
        raise RuntimeError(f"Unexpected motion shape {motion.shape}")

    n_motion_chunks = int(motion.shape[0])
    n_avail_latents = n_motion_chunks * _LATENTS_PER_MOTION_CHUNK
    if n_latent_frames > n_avail_latents:
        raise ValueError(
            f"motion file has only {n_motion_chunks} chunks "
            f"(= {n_avail_latents} usable latents) but caller requested "
            f"{n_latent_frames}; cap n_latent_frames at min(zarr_latents "
            f"- {_LATENT_HEAD_DROP}, n_motion_chunks * "
            f"{_LATENTS_PER_MOTION_CHUNK}) before calling."
        )

    chunk_indices = np.arange(n_latent_frames) // _LATENTS_PER_MOTION_CHUNK
    return motion[chunk_indices].astype(np.float32)


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

_MANIFEST_VERSION = 3  # bumped: head-latent drop + motion-chunk cap


def _index_single_zarr(
    zpath: Path,
    caption_root: Path,
    motion_root: Optional[Path] = None,
) -> Tuple[torch.Tensor, dict, int]:
    """Read metadata and captions for one zarr — no motion encoding.

    Returns ``n_latent_frames`` as the EFFECTIVE post-drop, motion-capped
    count: ``min(zarr_latents - _LATENT_HEAD_DROP, n_motion_chunks *
    _LATENTS_PER_MOTION_CHUNK)``. All downstream slicing in this module
    + ``_load_ride_tensors`` inherits this cap, so:
      * the head latent (Wan VAE's special 1-frame leading latent) is
        never visible to the trainer;
      * stale-padded motion past ``n_motion_chunks * 3`` latents is
        excluded so cmd_actions always reflect real motion.
    """
    g = zarr_lib.open_group(str(zpath), mode="r")
    attrs = dict(g.attrs)
    zarr_n_latents = int(g["latents"].shape[0])

    ride_dir_2k = attrs.get("ride_dir_2k", "")
    if not ride_dir_2k:
        raise RuntimeError("ride_dir_2k missing from zarr attrs")
    rel = _extract_ride_rel(ride_dir_2k)

    caption_file = _find_encoded_caption(caption_root, rel)
    if caption_file is None:
        raise FileNotFoundError(f"No encoded caption for {rel}")
    prompt_embeds = _load_prompt_embeds(caption_file)

    n_after_head_drop = max(0, zarr_n_latents - _LATENT_HEAD_DROP)
    if motion_root is None:
        # Caller didn't pass motion_root — return uncapped (legacy
        # callers e.g. test scripts). Trainer paths always pass it.
        n_latent_frames = n_after_head_drop
    else:
        try:
            n_motion_chunks = _peek_motion_chunks(attrs, motion_root)
        except FileNotFoundError:
            raise
        motion_capped = n_motion_chunks * _LATENTS_PER_MOTION_CHUNK
        n_latent_frames = min(n_after_head_drop, motion_capped)

    return prompt_embeds, attrs, n_latent_frames


def build_ride_manifest(
    encoded_root: str,
    caption_root: str,
    min_ride_frames: int = 21,
    cache_path: Optional[str] = None,
    motion_root: Optional[str] = None,
) -> List[dict]:
    """Scan all zarr rides and return a manifest list, with optional disk caching.

    Each entry is ``{"zarr_path": str, "prompt_embeds": Tensor, "attrs": dict,
    "n_latent_frames": int}``.  The list is sorted by zarr filename.
    ``n_latent_frames`` is the EFFECTIVE post-head-drop, motion-capped
    count (see ``_index_single_zarr``).

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
    mot_root = Path(motion_root) if motion_root is not None else None
    logging.info("Scanning %d zarr files in %s (ride-level)", n_zarr, enc)
    t0 = time.perf_counter()
    rides: List[dict] = []
    skipped = 0
    for zpath in zarr_paths:
        try:
            prompt_embeds, zarr_attrs, n_lat = _index_single_zarr(
                zpath, cap_root, mot_root,
            )
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
        sort_by_length: Optional[str] = None,
    ):
        self.encoded_root = Path(encoded_root)
        self.caption_root = Path(caption_root)
        self.motion_root = Path(motion_root)
        self.min_ride_frames = min_ride_frames
        self.start_zarr_index = start_zarr_index
        self.max_rides = max_rides
        self.sort_by_length = sort_by_length

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
        # Smoke-test affordance: skip the per-zarr scan and load a pre-built manifest.
        _manifest_pickle = os.environ.get("ARRWM_MANIFEST_PICKLE")
        if _manifest_pickle:
            try:
                _cached = torch.load(_manifest_pickle, map_location="cpu", weights_only=False)
                _rides = _cached.get("rides") if isinstance(_cached, dict) else None
                if _rides:
                    if self.max_rides is not None:
                        _rides = _rides[: int(self.max_rides)]
                    for r in _rides:
                        zp = Path(r["zarr_path"])
                        self._rides.append((zp, r["prompt_embeds"], r["attrs"], int(r["n_latent_frames"])))
                        self._attrs_by_path[str(zp)] = r["attrs"]
                    logging.info(
                        "[ZarrRideDataset] ARRWM_MANIFEST_PICKLE=%s -> loaded %d rides (no scan)",
                        _manifest_pickle, len(self._rides),
                    )
                    if self.sort_by_length in ("asc", "desc"):
                        self._rides.sort(key=lambda r: r[3], reverse=(self.sort_by_length == "desc"))
                    return
            except Exception as exc:
                logging.warning("[ZarrRideDataset] manifest pickle load failed (%s); falling back to scan", exc)
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
                prompt_embeds, zarr_attrs, n_latent_frames = _index_single_zarr(
                    zpath, self.caption_root, self.motion_root,
                )
            except Exception as exc:
                logging.warning("Skipping %s: %s", zpath.name, exc)
                skipped += 1
                continue

            if n_latent_frames < self.min_ride_frames:
                skipped += 1
                continue

            self._rides.append((zpath, prompt_embeds, zarr_attrs, n_latent_frames))
            self._attrs_by_path[str(zpath)] = zarr_attrs
            resolved = str(zpath.resolve())
            if resolved != str(zpath):
                self._attrs_by_path[resolved] = zarr_attrs

        elapsed = time.perf_counter() - t_index_start
        logging.info(
            "ZarrRideDataset: %d rides indexed (%d skipped) in %.1fs (%.2fs/ride)",
            len(self._rides), skipped, elapsed,
            elapsed / max(len(self._rides), 1),
        )
        if not self._rides:
            raise RuntimeError("No valid rides found. Check encoded_root, caption_root, motion_root.")

        # Optional ride ordering by latent frame count. Driven by
        # ``self.sort_by_length`` (set by the trainer via __init__
        # kwarg or post-construction attribute):
        #   "asc"  → shortest → longest (good for curriculum warm-up;
        #            small rides exercise every stage quickly and expose
        #            bugs on easy data before we spend hours on long ones)
        #   "desc" → longest → shortest
        #   None / "none" → leave in glob-sorted path order (default)
        sort_mode = getattr(self, "sort_by_length", None)
        if sort_mode in ("asc", "desc"):
            reverse = (sort_mode == "desc")
            self._rides.sort(key=lambda r: r[3], reverse=reverse)
            lengths = [r[3] for r in self._rides]
            logging.info(
                "ZarrRideDataset: sorted %d rides by length (%s); "
                "range [%d, %d], median=%d",
                len(self._rides), sort_mode,
                min(lengths), max(lengths),
                sorted(lengths)[len(lengths) // 2],
            )

    def encode_z_actions_window(
        self,
        zarr_path: str,
        n_latent_frames: int,
        latent_start: int,
        latent_end: int,
    ) -> torch.Tensor:
        """Encode motion for the latent frames in ``[latent_start, latent_end)``.

        v14-aligned (chunk-grain): each ``_LATENTS_PER_MOTION_CHUNK`` (= 3)
        consecutive dataset latents are encoded from the SAME motion entry,
        so the returned z stream is constant within each chunk. We
        encode at chunk-grain (one ss_vae forward per unique chunk in
        the window, NOT per latent), then ``np.repeat`` to per-latent
        for the trainer's per-frame action stream contract. The output
        values are identical to the per-latent path; only the
        encoding cost is reduced (3x fewer ss_vae forwards).

        ``n_latent_frames`` is the EFFECTIVE post-head-drop, motion-capped
        ride length (= ``meta["n_latent_frames"]`` from the manifest);
        the underlying motion file always has at least
        ``n_latent_frames / _LATENTS_PER_MOTION_CHUNK`` chunks, so no
        stale-padding can sneak in.

        Returns ``[latent_end - latent_start, z_dim]`` float32 tensor.
        """
        zarr_attrs = self._attrs_by_path[zarr_path]

        t0 = time.perf_counter()
        # Per-latent motion view (chunk-grain values, repeated within
        # each chunk). Cap to n_latent_frames so we never read past
        # the motion file's coverage.
        motion_per_latent = _load_aligned_motion_for_zarr(
            zarr_attrs, n_latent_frames, self.motion_root,
        )
        t_loaded = time.perf_counter()

        # Encode at chunk-grain to avoid 3x redundant ss_vae forwards.
        # Window touches chunks [chunk_lo, chunk_hi) — one motion row
        # per chunk; the per-latent stream then repeats each chunk's z
        # ``_LATENTS_PER_MOTION_CHUNK`` times.
        chunk_lo = latent_start // _LATENTS_PER_MOTION_CHUNK
        chunk_hi = (latent_end + _LATENTS_PER_MOTION_CHUNK - 1) // _LATENTS_PER_MOTION_CHUNK
        chunk_motion = motion_per_latent[
            chunk_lo * _LATENTS_PER_MOTION_CHUNK : chunk_hi * _LATENTS_PER_MOTION_CHUNK
            : _LATENTS_PER_MOTION_CHUNK
        ]  # one motion entry per chunk in [chunk_lo, chunk_hi)

        z_chunks = _encode_motion_ss_vae(
            chunk_motion, self._ss_vae, self._ss_scale, self._ss_dev,
        )  # [n_chunks_window, 8]
        t_encoded = time.perf_counter()

        # Per-latent broadcast of chunk-grain z, then slice to the
        # requested latent window (relative to chunk_lo's absolute
        # latent start = chunk_lo * _LATENTS_PER_MOTION_CHUNK).
        z_per_latent_full = np.repeat(z_chunks, _LATENTS_PER_MOTION_CHUNK, axis=0)
        rel_start = latent_start - chunk_lo * _LATENTS_PER_MOTION_CHUNK
        rel_end = rel_start + (latent_end - latent_start)
        z_window = z_per_latent_full[rel_start:rel_end]

        z_tensor = torch.from_numpy(z_window)
        z_squashed = _tanh_squash(z_tensor)

        logging.info(
            "  z_actions [%d:%d]: encode %d chunks (%d latents) | "
            "motion_load=%.3fs  ss_vae=%.3fs  total=%.3fs",
            latent_start, latent_end, chunk_hi - chunk_lo,
            latent_end - latent_start,
            t_loaded - t0, t_encoded - t_loaded, time.perf_counter() - t0,
        )
        return z_squashed

    def load_motion_magnitudes(
        self,
        zarr_path: str,
        n_latent_frames: int,
    ) -> np.ndarray:
        """Per-latent-frame motion magnitude for a ride.

        v14-aligned (chunk-grain): each motion entry encodes the mean
        per-frame |dx,dy| over its 12-video-frame window (= 3 latent
        frames). We compute one magnitude per chunk and broadcast to
        per-latent so all 3 latents in a chunk share the same
        magnitude. Used by the action-forcing trainer's offset picker
        to bias rollout starts toward windows with non-trivial motion.

        Returns ``(n_latent_frames,)`` float32 — units are the raw
        motion scale, NOT normalised. Empirically across the dataset
        well-moving windows have mean magnitude > 0.5; sub-0.05 is
        effectively parked.
        """
        zarr_attrs = self._attrs_by_path[zarr_path]
        motion_per_latent = _load_aligned_motion_for_zarr(
            zarr_attrs, n_latent_frames, self.motion_root,
        )  # (n_latent_frames, 100, 3) — chunk-grain values, repeated within chunk
        return np.linalg.norm(
            motion_per_latent[:, :, :2], axis=-1,
        ).mean(axis=-1).astype(np.float32)

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
        """Lazy-load a latent slice ``[start:end]`` from a zarr ride file.

        Indices are POST-head-drop dataset latents (not raw zarr
        indices). Internally we shift by ``_LATENT_HEAD_DROP`` so
        dataset latent index 0 reads from zarr index 1 — skipping the
        Wan VAE's special 1-frame head latent so that motion.npy's
        "first video frame dropped" convention aligns with our latent
        indexing 1:1. See module-level _LATENT_HEAD_DROP comment.
        """
        g = zarr_lib.open_group(zarr_path, mode="r")
        lat_np = g["latents"][start + _LATENT_HEAD_DROP : end + _LATENT_HEAD_DROP]
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
        zarr_n_latents = int(lat_ds.shape[0])  # raw (T, 16, 60, 104)

        # Derive relative ride path for caption/motion lookup
        ride_dir_2k = attrs.get("ride_dir_2k", "")
        if not ride_dir_2k:
            raise RuntimeError("ride_dir_2k missing from zarr attrs")
        rel = _extract_ride_rel(ride_dir_2k)

        # Caption
        caption_file = _find_encoded_caption(self.caption_root, rel)
        if caption_file is None:
            raise FileNotFoundError(f"No encoded caption for {rel}")
        prompt_embeds = _load_prompt_embeds(caption_file)

        # v14-aligned dataset latent count: drop the head latent + cap
        # at min(zarr - 1, n_motion_chunks * 3). See _index_single_zarr.
        n_after_head_drop = max(0, zarr_n_latents - _LATENT_HEAD_DROP)
        n_motion_chunks = _peek_motion_chunks(attrs, self.motion_root)
        motion_capped = n_motion_chunks * _LATENTS_PER_MOTION_CHUNK
        n_latent_frames = min(n_after_head_drop, motion_capped)

        # Per-latent motion view (chunk-grain values, repeated within
        # each chunk). Encode at chunk-grain to avoid 3x redundant
        # ss_vae forwards, then repeat back to per-latent.
        motion_per_latent = _load_aligned_motion_for_zarr(
            attrs, n_latent_frames, self.motion_root,
        )
        chunk_motion = motion_per_latent[::_LATENTS_PER_MOTION_CHUNK]
        z_chunks = _encode_motion_ss_vae(
            chunk_motion, self._ss_vae, self._ss_scale, self._ss_dev,
        )  # (n_chunks, 8)
        z_per_latent = np.repeat(
            z_chunks, _LATENTS_PER_MOTION_CHUNK, axis=0,
        )[:n_latent_frames]

        z_tensor = torch.from_numpy(z_per_latent)
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
