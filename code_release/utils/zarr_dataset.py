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
_DATA_ROOT = Path(os.path.join(os.environ.get("DATA_ROOT", ""), "frodobots_data"))


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

# Head-latent drop knob. v14's ``pre_encode_motion`` drops video frame
# 0 (line 153) but ARRWM's ``pre_encode_local`` does NOT, which leaves
# a 1-video-frame phase offset between motion entry 0 and Wan latent 0.
# The mismatch is small (1/12 of a chunk window) and empirically does
# not materially affect training, so we leave this disabled (= 0). All
# ``+ _LATENT_HEAD_DROP`` shifts are no-ops at this value; flip to 1
# if a future analysis decides the head latent ever needs to be hidden
# from the training stream.
_LATENT_HEAD_DROP = 0

# Action-vs-visual causal lag in seconds. Joystick command at time T
# becomes visible in the camera ~``_ACTION_CAUSAL_LAG_SEC`` seconds later.
# See utils/test_zarr_chunks.py:117 for v14's matching convention.
_ACTION_CAUSAL_LAG_SEC = 0.8

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
    The usable count after the v14 alignment offset is
    ``max(0, n_motion_chunks - _motion_chunk_offset(attrs))`` — see
    ``_motion_capped_latents``.
    """
    motion_path = _resolve_motion_path(attrs, motion_root)
    motion = np.load(motion_path, mmap_mode="r")
    if motion.ndim != 3 or motion.shape[2] != 3:
        raise RuntimeError(f"Unexpected motion shape {motion.shape} at {motion_path}")
    return int(motion.shape[0])


def _motion_chunk_offset(attrs: dict) -> int:
    """v14-aligned chunk offset between motion.npy and dataset latents.

    motion.npy is encoded from source-video frame 0 (pre_encode_motion.py
    drops only the first frame), but latents are encoded starting at
    ``action_start_sec`` of the source video, with a fixed causal lag of
    ``_ACTION_CAUSAL_LAG_SEC`` seconds between joystick command and
    visible response. v14's ``load_aligned_motion``
    (utils/test_zarr_chunks.py:117) shifts motion by
    ``(action_start_sec - 0.8) * fps`` source-video frames; chunk-grain
    rounds to whole motion windows. Result is non-negative (clamped at 0).

    NOTE: this round+clamp convention is the DELIBERATE, visually-verified
    alignment from commit 06e14c4 ("fixed the timing bug and verified
    visually") and is what the entire v14 family trains on. Do NOT change it
    (e.g. to floor / negative offsets) without re-verifying visually and
    re-baselining every run — the lags and magic numbers here must match v14.
    """
    action_start_sec = float(attrs.get("action_start_sec", 0.0))
    fps = float(attrs.get("fps", 20.0))
    offset_frames = (action_start_sec - _ACTION_CAUSAL_LAG_SEC) * fps
    chunk_offset = int(round(offset_frames / _MOTION_WINDOW_FRAMES))
    return max(0, chunk_offset)


def _motion_capped_latents(attrs: dict, motion_root: Path) -> int:
    """How many dataset latents are spanned by the usable motion window
    after applying ``_motion_chunk_offset``. Returns 0 if the offset
    consumes more than the available motion."""
    n_motion_chunks = _peek_motion_chunks(attrs, motion_root)
    chunk_offset = _motion_chunk_offset(attrs)
    usable_chunks = max(0, n_motion_chunks - chunk_offset)
    return usable_chunks * _LATENTS_PER_MOTION_CHUNK


def _load_aligned_motion_for_zarr(
    attrs: dict,
    n_latent_frames: int,
    motion_root: Path,
) -> np.ndarray:
    """Return per-DATASET-LATENT motion mapped from chunk-grain motion.npy.

    v14 discipline (see utils/pre_encode_motion.py:153 for the dropped
    first video frame, the head-latent spec at the top of this module,
    and ``_motion_chunk_offset`` for the
    ``(action_start_sec - 0.8) * fps`` shift between motion.npy and the
    encoded latents). After applying the per-ride chunk offset, each
    motion entry covers exactly ``_LATENTS_PER_MOTION_CHUNK`` (= 3)
    consecutive dataset latents: dataset latent index ``k`` reads from
    ``motion[chunk_offset + k // 3]``. No upsampling, no stale-padding —
    if the caller asks for more latents than the motion file supports
    we raise: the caller must cap ``n_latent_frames`` at
    ``_motion_capped_latents(attrs, motion_root)`` (the manifest indexer
    does this; ``_load_ride_tensors`` inherits the cap via
    ``meta["n_latent_frames"]``).

    Returns float32 array of shape ``(n_latent_frames, 100, 3)``. Within
    each chunk the 3 entries are identical (= the chunk's motion.npy
    row); across chunks they advance with the ride.
    """
    motion_path = _resolve_motion_path(attrs, motion_root)
    motion = np.load(motion_path)  # [n_motion_chunks, N, 3]
    if motion.ndim != 3 or motion.shape[2] != 3:
        raise RuntimeError(f"Unexpected motion shape {motion.shape}")

    n_motion_chunks = int(motion.shape[0])
    chunk_offset = _motion_chunk_offset(attrs)
    usable_chunks = max(0, n_motion_chunks - chunk_offset)
    n_avail_latents = usable_chunks * _LATENTS_PER_MOTION_CHUNK
    if n_latent_frames > n_avail_latents:
        raise ValueError(
            f"motion file has {n_motion_chunks} chunks; after chunk_offset="
            f"{chunk_offset} only {usable_chunks} usable (= {n_avail_latents} "
            f"latents) but caller requested {n_latent_frames}. Cap "
            f"n_latent_frames at _motion_capped_latents(attrs, motion_root) "
            f"before calling."
        )

    chunk_indices = chunk_offset + np.arange(n_latent_frames) // _LATENTS_PER_MOTION_CHUNK
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


# --- PCA action-vector encoder (drop-in alternative to the SS-VAE) ----------
# The SS-VAE's z2/z7 are each ~0.99 a SINGLE PCA component (z2<-PC1, z7<-PC0);
# the VAE was literally PCA-regularised. These affine maps put the PCA coords
# back into the VAE's z-space so the SAME tanh scales (25/10) apply and a
# VAE-trained model sees a near-identical conditioning signal (squashed-action
# MAE ~1.5-2%). Constants fit on 80k real motion windows.
_PCA_SLOT_COMP = {2: 1, 7: 0}                                  # action slot -> PCA component
_PCA_SLOT_AFFINE = {2: (0.34343, 0.3876), 7: (0.19748, 0.5047)}  # z_slot = a*PC + b


def _encode_motion_pca(motion_per_frame, pca_mean, pca_comp, latent_ch):
    """Encode motion (n,100,3) -> (n, latent_ch) via PCA + affine into VAE z-space.
    Drop-in for _encode_motion_ss_vae: only the action slots (2,7) are populated;
    the model selects action_dims=[2,7] downstream. dx/dy used; visibility ignored."""
    n = motion_per_frame.shape[0]
    flat = motion_per_frame[:, :, :2].reshape(n, 200).astype(np.float64)
    P = (flat - pca_mean) @ pca_comp.T                         # (n, n_pca)
    out = np.zeros((n, latent_ch), dtype=np.float32)
    for slot, comp in _PCA_SLOT_COMP.items():
        a, b = _PCA_SLOT_AFFINE[slot]
        out[:, slot] = (a * P[:, comp] + b).astype(np.float32)
    return out


# Raw-PCA (14e): NO affine, NO slot-remap. Dim i = PCA component i (pca_0=throttle,
# pca_1=steer, ...). tanh(P/scale) squash, scale = 2.5*std per component (saturation
# ~1%, squashed std ~0.3). The model conditions on dims [0,1]; the critic supervises
# the top-N. Pre-squash raw values returned here; squash applied by the caller.
_PCA_RAW_SCALES = np.array([93.7, 57.7, 22.5, 21.2, 18.1, 14.5, 12.6, 10.8], dtype=np.float32)


def _encode_motion_pca_raw(motion_per_frame, pca_mean, pca_comp, n_out=8):
    """Encode motion (n,100,3) -> (n, n_out) RAW top-n_out PCA projections (pre-squash)."""
    n = motion_per_frame.shape[0]
    flat = motion_per_frame[:, :, :2].reshape(n, 200).astype(np.float64)
    P = (flat - pca_mean) @ pca_comp.T                         # (n, n_pca)
    return P[:, :n_out].astype(np.float32)


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

_MANIFEST_VERSION = 4  # bumped: v14 (action_start_sec - 0.8) chunk offset


def _index_single_zarr(
    zpath: Path,
    caption_root: Path,
    motion_root: Optional[Path] = None,
) -> Tuple[torch.Tensor, dict, int]:
    """Read metadata and captions for one zarr — no motion encoding.

    Returns ``n_latent_frames`` as the EFFECTIVE post-drop, motion-capped
    count: ``min(zarr_latents - _LATENT_HEAD_DROP,
    _motion_capped_latents(attrs, motion_root))``. All downstream slicing
    in this module + ``_load_ride_tensors`` inherits this cap, so:
      * the head latent (Wan VAE's special 1-frame leading latent) is
        never visible to the trainer;
      * the v14 chunk offset (``_motion_chunk_offset``) is consumed at
        the start of motion.npy, and stale-padded motion past
        ``(n_motion_chunks - chunk_offset) * 3`` latents is excluded so
        cmd_actions always reflect real, properly-aligned motion.
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
            motion_capped = _motion_capped_latents(attrs, motion_root)
        except FileNotFoundError:
            raise
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
        cache_path: Optional[str] = None,
    ):
        self.encoded_root = Path(encoded_root)
        self.caption_root = Path(caption_root)
        self.motion_root = Path(motion_root)
        self.min_ride_frames = min_ride_frames
        self.start_zarr_index = start_zarr_index
        self.max_rides = max_rides
        self.sort_by_length = sort_by_length
        # Optional on-disk cache for the ride scan. ``_build_index`` reads
        # from it on entry (skipping the 30-min zarr scan) and writes
        # back to it after a fresh scan. The version + encoded_root are
        # validated on load — mismatches force a re-scan.
        self.cache_path = cache_path

        ss_dev = ss_vae_device or device
        logging.info("Loading ss_vae from %s on %s", ss_vae_checkpoint, ss_dev)
        ss_vae_model, ss_scale = load_ss_vae(ss_vae_checkpoint, ss_dev)
        self._ss_vae = ss_vae_model
        self._ss_scale = float(ss_scale)
        self._ss_dev = ss_dev

        # Action-vector encoder: "ss_vae" (default; PCA-regularised VAE) or "pca"
        # (pure-PCA drop-in, affine-mapped into VAE z-space). PCA basis lives in
        # the ss_vae checkpoint, so both modes load the same file.
        self._action_encoder = os.environ.get("ARRWM_ACTION_ENCODER", "ss_vae").lower()
        _ss_ck = torch.load(ss_vae_checkpoint, map_location="cpu", weights_only=False)
        self._pca_mean = np.asarray(_ss_ck["pca_mean"], dtype=np.float64)
        self._pca_comp = np.asarray(_ss_ck["pca_comp"], dtype=np.float64)
        self._latent_ch = int(_ss_ck["hparams"]["LATENT_CH"])
        if getattr(self, "_action_encoder", "ss_vae") == "pca":
            logging.info("ACTION ENCODER = PCA drop-in (slots %s <- PCA comps %s, affine->VAE z-space)",
                         list(_PCA_SLOT_COMP), [_PCA_SLOT_COMP[s] for s in _PCA_SLOT_COMP])

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

        # Action-vector encoder (see __init__): "ss_vae" default or "pca" drop-in.
        # from_manifest bypasses __init__, so set it here too.
        obj._action_encoder = os.environ.get("ARRWM_ACTION_ENCODER", "ss_vae").lower()
        if _share_ss_vae is not None and hasattr(_share_ss_vae, "_pca_mean"):
            obj._pca_mean = _share_ss_vae._pca_mean
            obj._pca_comp = _share_ss_vae._pca_comp
            obj._latent_ch = _share_ss_vae._latent_ch
        else:
            _ss_ck = torch.load(ss_vae_checkpoint, map_location="cpu", weights_only=False)
            obj._pca_mean = np.asarray(_ss_ck["pca_mean"], dtype=np.float64)
            obj._pca_comp = np.asarray(_ss_ck["pca_comp"], dtype=np.float64)
            obj._latent_ch = int(_ss_ck["hparams"]["LATENT_CH"])
        if obj._action_encoder == "pca":
            logging.info("ACTION ENCODER = PCA drop-in (from_manifest): slots %s <- PCA %s",
                         list(_PCA_SLOT_COMP), [_PCA_SLOT_COMP[s] for s in _PCA_SLOT_COMP])

        obj._rides = []
        obj._attrs_by_path = {}
        # Optional per-entry forced window start (curated motion-y / backward
        # pool). -1 = no force (batcher picks its usual random offset). Parallel
        # to _rides so the existing tuple layout is untouched.
        obj._forced_offsets = []
        for r in rides_data:
            zpath = Path(r["zarr_path"])
            obj._rides.append((zpath, r["prompt_embeds"], r["attrs"], r["n_latent_frames"]))
            obj._attrs_by_path[r["zarr_path"]] = r["attrs"]
            obj._forced_offsets.append(int(r.get("forced_offset", -1)))

        logging.info("ZarrRideDataset.from_manifest: %d rides loaded (no scan)", len(obj._rides))
        return obj

    def _build_index(self) -> None:
        # Trainer-supplied cache (constructor kwarg). When provided AND
        # the file exists with a matching version + encoded_root, this
        # short-circuits the 30-min zarr scan. After a fresh scan the
        # cache is written back so the next run starts instantly.
        # Mirrors the shape of ``build_ride_manifest``'s on-disk cache
        # (rides, version, encoded_root keys).
        if self.cache_path and os.path.exists(self.cache_path):
            try:
                _cached = torch.load(
                    self.cache_path, map_location="cpu", weights_only=False,
                )
                _ver = (
                    _cached.get("version") if isinstance(_cached, dict) else None
                )
                _enc = (
                    _cached.get("encoded_root") if isinstance(_cached, dict) else None
                )
                _rides = (
                    _cached.get("rides") if isinstance(_cached, dict) else None
                )
                if (
                    _ver == _MANIFEST_VERSION
                    and _enc == str(self.encoded_root)
                    and _rides
                ):
                    if self.max_rides is not None:
                        _rides = _rides[: int(self.max_rides)]
                    for r in _rides:
                        zp = Path(r["zarr_path"])
                        self._rides.append((zp, r["prompt_embeds"], r["attrs"], int(r["n_latent_frames"])))
                        self._attrs_by_path[str(zp)] = r["attrs"]
                    logging.info(
                        "[ZarrRideDataset] cache_path=%s -> loaded %d rides (no scan)",
                        self.cache_path, len(self._rides),
                    )
                    if self.sort_by_length in ("asc", "desc"):
                        self._rides.sort(key=lambda r: r[3], reverse=(self.sort_by_length == "desc"))
                    return
                else:
                    logging.info(
                        "[ZarrRideDataset] cache at %s rejected (version=%s "
                        "expected=%d, encoded_root=%s expected=%s) — rescanning.",
                        self.cache_path, _ver, _MANIFEST_VERSION,
                        _enc, str(self.encoded_root),
                    )
            except Exception as exc:
                logging.warning(
                    "[ZarrRideDataset] cache load failed (%s); rescanning.", exc,
                )

        # Smoke-test affordance: skip the per-zarr scan and load a pre-built manifest.
        _manifest_pickle = os.environ.get("ARRWM_MANIFEST_PICKLE")
        if _manifest_pickle:
            try:
                _cached = torch.load(_manifest_pickle, map_location="cpu", weights_only=False)
                # Mirror the version guard ``build_ride_manifest`` does
                # for ``cache_path`` — pre-v3 pickles have uncapped
                # ``n_latent_frames`` and would crash inside
                # ``encode_z_actions_window`` on first call. Reject
                # silently and fall through to a fresh scan.
                _cached_version = (
                    _cached.get("version") if isinstance(_cached, dict) else None
                )
                if _cached_version != _MANIFEST_VERSION:
                    logging.warning(
                        "[ZarrRideDataset] ARRWM_MANIFEST_PICKLE=%s has "
                        "version=%s but current=%d — ignoring cache, "
                        "rebuilding from scratch.",
                        _manifest_pickle, _cached_version, _MANIFEST_VERSION,
                    )
                    _cached = None
                _rides = (
                    _cached.get("rides")
                    if isinstance(_cached, dict) else None
                )
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

        # Persist the freshly-scanned manifest if the caller supplied
        # ``cache_path``. RANK-0 ONLY: 32 ranks racing torch.save on the
        # same Lustre path would corrupt the pickle. Other ranks just
        # log + continue (they paid the scan cost in parallel anyway,
        # which is fine since the scan is read-only on the encoded
        # zarrs). Future runs (same encoded_root, same version) short-
        # circuit the scan via the load path above.
        _is_rank0 = True
        try:
            import torch.distributed as _dist
            if _dist.is_available() and _dist.is_initialized():
                _is_rank0 = _dist.get_rank() == 0
        except Exception:
            pass
        if self.cache_path and _is_rank0:
            try:
                # Atomic write via tempfile + rename so a partial write
                # (process killed mid-save) doesn't poison the cache.
                _tmp = f"{self.cache_path}.tmp.{os.getpid()}"
                Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
                _payload = {
                    "version": _MANIFEST_VERSION,
                    "encoded_root": str(self.encoded_root),
                    "rides": [
                        {
                            "zarr_path": str(zp),
                            "prompt_embeds": pe,
                            "attrs": at,
                            "n_latent_frames": nl,
                        }
                        for (zp, pe, at, nl) in self._rides
                    ],
                }
                torch.save(_payload, _tmp)
                os.replace(_tmp, self.cache_path)
                logging.info(
                    "[ZarrRideDataset rank0] manifest cache written to %s "
                    "(%d rides; reuse on next run skips the %.1fs scan)",
                    self.cache_path, len(self._rides), elapsed,
                )
            except Exception as exc:
                logging.warning(
                    "[ZarrRideDataset] manifest cache write failed (%s); "
                    "next run will re-scan.", exc,
                )

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

        _enc = getattr(self, "_action_encoder", "ss_vae")
        if _enc == "pca":
            z_chunks = _encode_motion_pca(
                chunk_motion, self._pca_mean, self._pca_comp, self._latent_ch,
            )  # [n_chunks_window, latent_ch] (affine into VAE z-space)
        elif _enc == "pca_raw":
            z_chunks = _encode_motion_pca_raw(
                chunk_motion, self._pca_mean, self._pca_comp, self._latent_ch,
            )  # [n_chunks_window, latent_ch] RAW top-N PCA (pre-squash)
        else:
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
        if getattr(self, "_action_encoder", "ss_vae") == "pca_raw":
            scales = torch.from_numpy(_PCA_RAW_SCALES[:z_tensor.shape[-1]]).to(z_tensor.dtype)
            z_squashed = torch.tanh(z_tensor / scales)             # per-component squash
        else:
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
        forced = self._forced_offsets[idx] if getattr(self, "_forced_offsets", None) else -1
        return {
            "zarr_path": str(zpath),
            "prompt_embeds": prompt_embeds,
            "n_latent_frames": n_frames,
            "forced_offset": int(forced),
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

    # ------------------------------------------------------------------
    # Dual-view (Study 1 / AOO): rear camera as a simultaneous second view
    # ------------------------------------------------------------------
    @staticmethod
    def load_reverse_latent_chunk(
        reverse_zarr_path: str,
        rear_indices: "np.ndarray",
    ) -> torch.Tensor:
        """Gather rear latents at WALL-CLOCK-aligned rear indices.

        ``rear_indices`` is one rear dataset-latent index per forward latent in
        the window (from ``<ride>.align.npy``, see utils/build_rear_alignment.py).
        We fail LOUD on any -1 (unaligned) entry — callers must constrain windows
        to the aligned span so this never fires; a silent clamp would misalign
        the two views. One contiguous read [min..max] then index in numpy
        (rear indices are monotonic with possible VFR repeats).
        """
        assert _LATENT_HEAD_DROP == 0, (
            "dual-view rear gather assumes _LATENT_HEAD_DROP==0 (current default); "
            "update gather offsets if it changes."
        )
        idx = np.asarray(rear_indices).astype(np.int64)
        if (idx < 0).any():
            raise ValueError(
                "reverse gather hit unaligned (-1) frames; window not constrained "
                "to the aligned span."
            )
        g = zarr_lib.open_group(reverse_zarr_path, mode="r")
        n_rear = int(g["latents"].shape[0])
        idx = np.clip(idx, 0, n_rear - 1)
        rmin, rmax = int(idx.min()), int(idx.max())
        block = g["latents"][rmin : rmax + 1]
        out = block[idx - rmin]
        return torch.from_numpy(out.astype(np.float32))

    @staticmethod
    def load_alignment_map(reverse_root: str, ride_ts: str) -> "np.ndarray":
        """Load ``<reverse_root>/<ride_ts>.align.npy`` (int32, len = T_forward)."""
        return np.load(str(Path(reverse_root) / f"{ride_ts}.align.npy"))

    @staticmethod
    def aligned_span(amap: "np.ndarray") -> Tuple[int, int]:
        """Largest contiguous run of aligned (>=0) forward latents -> (start, end).

        The map is built from monotonic wall-clock so the aligned region is a
        single contiguous interval flanked by -1; we still scan for the longest
        run to be robust. Returns (0, 0) if nothing is aligned.
        """
        valid = np.asarray(amap) >= 0
        best_s = best_e = 0
        s = None
        for i, v in enumerate(valid):
            if v and s is None:
                s = i
            elif not v and s is not None:
                if i - s > best_e - best_s:
                    best_s, best_e = s, i
                s = None
        if s is not None and len(valid) - s > best_e - best_s:
            best_s, best_e = s, len(valid)
        return best_s, best_e


def flip_reverse_z(z: torch.Tensor, dims: Tuple[int, ...] = (2, 7)) -> torch.Tensor:
    """Sign-flip the egomotion dims for the rear/reverse view.

    The rear camera under forward translation sees the world recede (z7 throttle
    flips) and yaw pans oppositely (z2 steering flips). The reverse view reuses
    the forward z with these dims negated (NOT a re-extraction). Operates on the
    raw 8-D ss_vae z BEFORE any action_dims slicing, so it's correct regardless
    of action_dims ordering. Returns a new tensor; input is not mutated.
    """
    z = z.clone()
    for d in dims:
        z[..., d] = -z[..., d]
    return z


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

        # Action-vector encoder: "ss_vae" (default) or "pca" drop-in. See the
        # other constructor for the rationale; PCA basis is in the ss_vae ckpt.
        self._action_encoder = os.environ.get("ARRWM_ACTION_ENCODER", "ss_vae").lower()
        _ss_ck = torch.load(ss_vae_checkpoint, map_location="cpu", weights_only=False)
        self._pca_mean = np.asarray(_ss_ck["pca_mean"], dtype=np.float64)
        self._pca_comp = np.asarray(_ss_ck["pca_comp"], dtype=np.float64)
        self._latent_ch = int(_ss_ck["hparams"]["LATENT_CH"])
        if getattr(self, "_action_encoder", "ss_vae") == "pca":
            logging.info("ACTION ENCODER = PCA drop-in (slots %s <- PCA comps %s, affine->VAE z-space)",
                         list(_PCA_SLOT_COMP), [_PCA_SLOT_COMP[s] for s in _PCA_SLOT_COMP])

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
        # at min(zarr - 1, _motion_capped_latents). The motion cap
        # accounts for the per-ride (action_start_sec - 0.8) chunk
        # offset between motion.npy and the encoded latents.
        n_after_head_drop = max(0, zarr_n_latents - _LATENT_HEAD_DROP)
        motion_capped = _motion_capped_latents(attrs, self.motion_root)
        n_latent_frames = min(n_after_head_drop, motion_capped)

        # Per-latent motion view (chunk-grain values, repeated within
        # each chunk). Encode at chunk-grain to avoid 3x redundant
        # ss_vae forwards, then repeat back to per-latent.
        motion_per_latent = _load_aligned_motion_for_zarr(
            attrs, n_latent_frames, self.motion_root,
        )
        chunk_motion = motion_per_latent[::_LATENTS_PER_MOTION_CHUNK]
        if getattr(self, "_action_encoder", "ss_vae") == "pca":
            z_chunks = _encode_motion_pca(
                chunk_motion, self._pca_mean, self._pca_comp, self._latent_ch,
            )  # (n_chunks, latent_ch)
        else:
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

        # Load latents lazily from zarr (window + leading context).
        # ``z_actions_latent`` and ``self._samples``'s ``start`` are
        # POST-head-drop dataset latents; shift the zarr read by
        # ``_LATENT_HEAD_DROP`` so frame i of the returned tensor
        # corresponds to the same time position as ``z_actions_latent[i]``
        # (= chunk i // 3's z, mapped to the post-drop video segment).
        g = zarr_lib.open_group(str(zpath), mode="r")
        lat_np = g["latents"][start + _LATENT_HEAD_DROP : end + _LATENT_HEAD_DROP]
        latents = torch.from_numpy(lat_np.astype(np.float32))

        z_window = z_actions_latent[start:end]  # (window_size+context_frames, 8)

        return {
            "real_latents": latents,          # (window_size+context_frames, 16, 60, 104)
            "prompt_embeds": prompt_embeds,   # (seq_len, dim)
            "z_actions": z_window,            # (window_size+context_frames, 8)
        }
