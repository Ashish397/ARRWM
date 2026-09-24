"""Visual starting-seed picker for the interactive world model.

A ride is a zarr holding ``latents`` [T, 16, 60, 104].  A ride is only usable
as a seed if it has at least ``MIN_SEED_FRAMES`` (= SEED_PREFILL_CHUNKS *
NUM_FRAME_PER_BLOCK = 21) latent frames, because ``engine.reset()`` prefills
seven ground-truth chunks before it hands control to the model.

The picker also lists the user's OWN VIDEOS (tagged ``[VID]``) from
``interactive/user_videos/`` and ``~/Videos``.  A video is put into training
space by ``read_video_frames`` / ``load_seed_from_video``, which reproduce the
zarr encoder's transform: 20 fps (the rides' NATIVE encode rate) -> crop to
the camera's aspect -> optional mild barrel warp -> anamorphic squash to
832x480 -> /255*2-1.  See ``_seed_filter_chain`` and ``SEED_ENCODE_FPS``.

Two entry points:

    pick_random(seed_root)              -> RideInfo   (headless, for --seed random)
    SeedPicker(...).run()               -> str | None (pygame grid browser)

Thumbnails
----------
One frame per ride: the first chunk (3 latent frames) is decoded with a
STANDALONE ``taew2_1`` decoder and frame 0 is kept at half resolution.  TAEHV
is ~10M params and ~24 ms/chunk, so a 12-cell page costs well under half a
second cold, and nothing after that -- results are cached as JPEGs under
``interactive/seed_thumbs/<ride_id>.jpg``.

This decoder instance is deliberately SEPARATE from the engine's.  The engine
pins every VAE op to one CUDA stream for the life of a session (see the
_vae_stream invariant in engine.py); borrowing its decoder from the UI thread
would violate that.  The picker instead runs on the default stream, and
callers are expected to have the engine worker paused while it does.
"""

from __future__ import annotations

import logging
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np

from interactive.engine_api import (
    NUM_FRAME_PER_BLOCK,
    PLAYBACK_FPS,
    SEED_PREFILL_CHUNKS,
)

log = logging.getLogger("seed_picker")

#: A serving seed fills the complete seven-chunk attention span.
MIN_SEED_FRAMES = SEED_PREFILL_CHUNKS * NUM_FRAME_PER_BLOCK   # 21

DEFAULT_SEED_ROOT = "/home/ashish/frodobots/frodobots_encoded"
FALLBACK_SEED_ROOT = str(Path(__file__).resolve().parent.parent / "smoke_zarr")
THUMB_DIR = Path(__file__).resolve().parent / "seed_thumbs"

GRID_COLS, GRID_ROWS = 4, 3
PAGE_SIZE = GRID_COLS * GRID_ROWS


# ---------------------------------------------------------------------------
# Ride discovery
# ---------------------------------------------------------------------------
@dataclass
class RideInfo:
    path: str
    ride_id: str
    n_latents: int = -1          # -1 = not probed yet

    @property
    def is_video(self) -> bool:
        return is_video(self.path)

    @property
    def usable(self) -> bool:
        """Probed and long enough. Unprobed rides report False -- probe first.

        A ride zarr must supply the full seven-chunk span. A user VIDEO only
        needs one whole chunk: short clips are seeded at reduced depth on
        purpose (the engine logs "video supports k/7 seed chunks") rather than
        being rejected outright.
        """
        if self.is_video:
            return self.n_latents >= NUM_FRAME_PER_BLOCK
        return self.n_latents >= MIN_SEED_FRAMES

    @property
    def probed(self) -> bool:
        return self.n_latents >= 0


def resolve_seed_root(seed_root: Optional[str] = None) -> str:
    """Pick the rides directory, falling back to the small smoke set."""
    for cand in (seed_root, DEFAULT_SEED_ROOT, FALLBACK_SEED_ROOT):
        if cand and Path(cand).is_dir() and list(Path(cand).glob("*.zarr")):
            return str(cand)
    raise SystemExit(
        f"no rides found: tried {seed_root!r}, {DEFAULT_SEED_ROOT!r}, "
        f"{FALLBACK_SEED_ROOT!r}")


def list_rides(seed_root: Optional[str] = None) -> List[RideInfo]:
    """Every ride zarr under ``seed_root``, sorted by id (ids are timestamps)."""
    root = resolve_seed_root(seed_root)
    rides = [RideInfo(path=str(p), ride_id=p.stem)
             for p in sorted(Path(root).glob("*.zarr"))]
    log.info("seed root %s -> %d rides", root, len(rides))
    return rides


def probe_ride(ride: RideInfo) -> RideInfo:
    """Fill in ``n_latents`` cheaply: zarr metadata, or ffprobe for a video."""
    if ride.probed:
        return ride
    if is_video(ride.path):
        # For a video, "n_latents" is what the clip WOULD yield after the
        # 20 fps resample, so `usable` and the picker's length rules work
        # unchanged for both entry kinds.
        info = probe_video(ride.path)
        ride.n_latents = int(info["chunks"]) * NUM_FRAME_PER_BLOCK
        if info["error"]:
            log.warning("probe failed for video %s: %s", ride.ride_id, info["error"])
        return ride
    try:
        import zarr as zarr_lib
        # NB: the installed zarr rejects zarr.open(path, "r") positionally;
        # open_group(path, mode=...) is what utils/play_world_model.py uses.
        g = zarr_lib.open_group(ride.path, mode="r")
        ride.n_latents = int(g["latents"].shape[0])
    except Exception as exc:
        log.warning("probe failed for %s: %s", ride.ride_id, exc)
        ride.n_latents = 0
    return ride


def filter_rides(rides: Sequence[RideInfo], query: str) -> List[RideInfo]:
    """Substring filter on the ride id (ids are digit timestamps)."""
    q = (query or "").strip()
    if not q:
        return list(rides)
    # Case-insensitive: ride ids are digit timestamps, but video entries are
    # user filenames and typing exact case would be a nuisance.
    ql = q.lower()
    return [r for r in rides if ql in r.ride_id.lower()]


def pick_random(seed_root: Optional[str] = None,
                rng: Optional[random.Random] = None,
                max_tries: int = 40) -> RideInfo:
    """A random USABLE ride. Headless; used for ``--seed random``."""
    rides = list_rides(seed_root)
    if not rides:
        raise SystemExit("no rides to pick from")
    rnd = rng or random
    pool = list(rides)
    rnd.shuffle(pool)
    for ride in pool[:max_tries]:
        if probe_ride(ride).usable:
            return ride
    raise SystemExit(
        f"no ride with >= {MIN_SEED_FRAMES} latent frames in {max_tries} tries")


# ---------------------------------------------------------------------------
# Seeding from an arbitrary local video
# ---------------------------------------------------------------------------
VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")
VIDEO_SEED_DIR = Path(__file__).resolve().parent / "video_seeds"
#: Documented drop-folder: put your own clips here and they show up in the
#: picker tagged [VID].  Created on demand by ``list_videos``.
USER_VIDEO_DIR = Path(__file__).resolve().parent / "user_videos"
#: Searched in order when no --video_dir is given.
DEFAULT_VIDEO_DIRS = (USER_VIDEO_DIR, Path.home() / "Videos")

#: The rate the TRAINING LATENTS were encoded at -- NOT the playback rate.
#:
#: Audited from the zarr-creation pipeline.  Definitive references:
#:   * ``utils/pre_encode_local.py`` VideoLoader.stream_blocks line 139:
#:     ``-vf scale=832:480,showinfo`` with ``-vsync 0``, rawvideo rgb24 --
#:     a plain whole-frame squash, NO crop and NO ``fps=`` filter / ``-r``;
#:   * ``utils/pre_encode_direct.py`` wraps it (``--scale 832x480``);
#:   * ``utils/pre_encode.py`` line 89 (older HLS path) uses the same
#:     ``scale=832:480``;
#:   * values: ``VAEEncoder.encode_block`` does ``div_(255)`` then ``*2-1``
#:     -- which this module already matched, so nothing changed there.
#: Frames were therefore encoded at the rides' NATIVE rate, and every ride
#: zarr records ``attrs['fps'] = 20.0``.  One latent spans 4 source frames =
#: 0.20 s of real time.
#:
#: PLAYBACK_FPS (16) is only the rate we RENDER at, which is why a rollout
#: plays back slightly slow-motion; it is not the rate the model's temporal
#: prior was learned at.  Seeding a user clip at 16 fps packed 0.25 s of real
#: motion into each latent -- 25 % more than training -- so user footage is
#: resampled to 20 fps.
SEED_ENCODE_FPS = 20.0
#: Back-compat alias; older callers imported SEED_FPS.
SEED_FPS = SEED_ENCODE_FPS
PIXEL_W, PIXEL_H = 832, 480

#: Aspect ratio of the ROBOT CAMERA's native frame, before the zarr pipeline
#: squashed it to 832x480.
#:
#: ASSUMPTION, NOT VERIFIED LOCALLY -- flagged deliberately.  The ride zarrs
#: name ``ride_dir_2k`` / ``source_video_2k`` under
#: ``/home/ashish/frodobots/frodobots_data/``, which DOES NOT EXIST on this
#: machine (checked), and the encoder records only ``fps`` in the zarr attrs
#: -- no width/height anywhere.  A FrodoBots 2K cam is 16:9-ish, so 16/9 is
#: the working assumption; probe a real raw ride and correct this constant if
#: one ever lands locally.
#:
#: Note 832/480 = 1.7333 is NOT 16:9 (1.7778), so the training transform was
#: a mild ANAMORPHIC SQUEEZE of ~2.5 % horizontally -- reproduced below.
#: (``utils/pre_encode_text.py`` STYLE_SENTENCE describes the camera as
#: "~100 degrees HFOV and 70 degrees VFOV ... mild fisheye distortion, soft
#: corners, faint vignette, 480p".  A 100:70 FOV ratio is consistent with a
#: wide 16:9-ish frame once fisheye projection is accounted for, but it is
#: not a precise AR measurement, so it corroborates rather than replaces the
#: assumption.)
CAMERA_AR = 16.0 / 9.0

#: Default barrel-distortion strength for user footage (0 disables).
#:
#: Target, from ``utils/pre_encode_text.py`` STYLE_SENTENCE: the robot camera
#: is a "low-mounted wide-angle dash/action camera (around 100 degrees HFOV
#: and 70 degrees VFOV) ... mild fisheye distortion, soft corners".  A typical
#: phone is ~70-80 degrees HFOV and near-rectilinear, so we add barrel to
#: match the GEOMETRY (how straight lines bow toward the edges).
#:
#: DEFAULT OFF (user decision).  The warp is opt-in via
#: ``--seed_video_fisheye <s>``; at 0.0 the lenscorrection stage is omitted
#: from the filter chain entirely rather than added as a k=0 no-op, so the
#: default path is a pure crop+squash with no resampling loss.
#:
#: Why it exists, and why it is off: real decoded ride frames show pronounced
#: barrel (curved horizons, bowing kerbs -- see
#: interactive/video_seed_out/tuning/fisheye_sweep2.png), so a warp moves a
#: rectilinear phone clip's GEOMETRY toward training.  But negative-k1
#: sampling pulls the frame inward, costing field of view that phone footage
#: cannot spare, and a warp cannot manufacture the ~20-30 degrees the phone
#: never captured -- so it buys a partial geometry match at a real framing
#: cost.  Off by default; try 0.15-0.22 if a clip looks too "flat".
#: Mapped to ffmpeg lenscorrection as k1 = -s, k2 = -s/3.
DEFAULT_FISHEYE = 0.0

#: Which VERTICAL band survives when the crop-to-camera-AR has to trim height
#: (portrait / any source taller than 16:9).  BOTTOM by default: the training
#: camera is a LOW-MOUNTED dash cam sitting near ground level, so the bottom
#: of a hand-held phone frame -- road and ground running ahead -- is far
#: closer to the robot's view than the middle (horizon) or the top (sky).
#:
#: VERTICAL AXIS ONLY.  When the crop instead trims WIDTH (an ultrawide
#: source), the region stays horizontally CENTRED regardless of this setting:
#: the ffmpeg x-offset is always ``(iw-out_w)/2`` and only the y-offset
#: changes.  There is deliberately no left/right band selector -- a robot
#: drives toward the centre of its frame, so off-centre horizontal cropping
#: would misrepresent the heading.
CROP_MODES = ("center", "top", "bottom")
DEFAULT_CROP_BAND = "bottom"


def crop_dims(iw: int, ih: int, ar: float = CAMERA_AR) -> Tuple[int, int]:
    """The MAXIMAL centred crop of ``iw x ih`` at aspect ``ar``.

    A pure-Python mirror of the ffmpeg expression in ``_seed_filter_chain``,
    so the maximality property is testable without invoking ffmpeg.

    Maximal means one dimension is always kept in FULL:
      * input WIDER than ar  -> full HEIGHT, width trimmed  (2560x1080 ->
        1920x1080)
      * input TALLER than ar -> full WIDTH, height trimmed  (1080x1920 ->
        1080x607); WHICH height strip survives is the caller's ``crop`` band
      * input at ar          -> the whole frame untouched
    There is no inset or margin: the crop is anchored by
    ``x=(iw-out_w)/2`` and the y expression for the chosen band.
    """
    w = min(int(iw), int(ih * ar))
    h = min(int(ih), int(iw / ar))
    return int(w), int(h)


def _seed_filter_chain(fisheye: float = DEFAULT_FISHEYE,
                       crop: str = DEFAULT_CROP_BAND,
                       fps: float = SEED_ENCODE_FPS) -> str:
    """The ffmpeg filter chain that turns ANY clip into training-space frames.

    Four stages, in this order, and the order is the point:

      1. ``fps``   resample to the rate the training latents were encoded at
                   (20), so one latent covers the same real time as training.
      2. ``crop``  MAXIMAL crop to the CAMERA's aspect ratio (16:9 assumed)
                   -- one dimension is always kept in full, see ``crop_dims``.
                   Horizontally always CENTRED; vertically the ``crop`` band
                   decides which strip survives, defaulting to the BOTTOM
                   because the robot camera is low-mounted (see
                   DEFAULT_CROP_BAND).
                   The zarr pipeline never cropped -- it squashed whole frames
                   straight from the camera -- so to reproduce that transform
                   faithfully we must first put the user's frame INTO the
                   camera's aspect.  For landscape phone video (16:9 or 20:9)
                   this is a sliver; for PORTRAIT it is a heavy but unavoidable
                   crop, and ``crop=`` picks which band survives.
      3. ``lenscorrection``  OPTIONAL, OFF BY DEFAULT barrel warp, in the
                   CAMERA's geometry -- i.e. before the anamorphic squeeze,
                   because a real lens distorts before any pixel-aspect
                   scaling.  Negative k1 samples inward, so no black corners.
      4. ``scale=832:480``  the training squash itself, verbatim from
                   ``utils/pre_encode.py:89``: a plain anamorphic resize with
                   NO aspect preservation.
    """
    if crop not in CROP_MODES:
        crop = DEFAULT_CROP_BAND
    y = {"center": "(ih-out_h)/2", "top": "0", "bottom": "ih-out_h"}[crop]
    ar = CAMERA_AR
    parts = [f"fps={fps:g}"]
    # Largest centred region with the camera's aspect that fits the input.
    parts.append(
        f"crop=w='min(iw,ih*{ar:.6f})':h='min(ih,iw/{ar:.6f})'"
        f":x='(iw-out_w)/2':y='{y}'")
    s = float(fisheye or 0.0)
    if s > 0:
        # k1 negative = ADD barrel. Every output pixel samples from inside the
        # source, so the frame stays full-bleed (no fill needed).
        parts.append(f"lenscorrection=k1={-s:.4f}:k2={-s / 3.0:.4f}:i=bilinear")
    parts.append(f"scale={PIXEL_W}:{PIXEL_H}")
    return ",".join(parts)


def is_video(path: str) -> bool:
    """True for any path with a container suffix we can decode."""
    return Path(str(path)).suffix.lower() in VIDEO_SUFFIXES


def seed_pixel_frames(n_chunks: int) -> int:
    """Pixel frames needed for ``n_chunks`` seed chunks.

    The Wan temporal packing is 1 + 4*(L-1): the "special first" latent
    encodes a single frame and every later latent encodes four.  For the
    default seven chunks that is 21 latents = 1 + 4*20 = 81 pixel frames.
    """
    n_lat = max(int(n_chunks), 1) * NUM_FRAME_PER_BLOCK
    return 1 + 4 * (n_lat - 1)


def seed_chunks_for_frames(n_pixel_frames: int) -> int:
    """How many WHOLE seed chunks ``n_pixel_frames`` pixel frames support."""
    if n_pixel_frames < 1:
        return 0
    n_lat = 1 + (int(n_pixel_frames) - 1) // 4          # inverse of 1+4*(L-1)
    return int(n_lat // NUM_FRAME_PER_BLOCK)


#: 81 pixel frames is exactly 21 Wan latent frames (the seven-chunk default).
SEED_PIXEL_FRAMES = seed_pixel_frames(SEED_PREFILL_CHUNKS)      # 81


def parse_start_spec(spec, fps: float = SEED_FPS) -> float:
    """Start offset -> SECONDS.

    Accepted forms (the ambiguity is deliberate and documented):
      * ``12``      bare INTEGER  -> 12 frames at the resampled rate
      * ``12.5``    bare FLOAT    -> 12.5 seconds
      * ``"3s"``    's' suffix    -> 3 seconds
      * ``"90f"``   'f' suffix    -> 90 frames
    ``--seed_video_start_s`` always means seconds and does not come through
    here.  0 is 0 either way, which is why the default is unaffected.
    """
    if spec is None:
        return 0.0
    if isinstance(spec, (int, np.integer)) and not isinstance(spec, bool):
        return float(spec) / float(fps)                 # integer = frames
    if isinstance(spec, float):
        return float(spec)                              # float = seconds
    txt = str(spec).strip().lower()
    if not txt:
        return 0.0
    if txt.endswith("s"):
        return float(txt[:-1] or 0.0)
    if txt.endswith("f"):
        return float(txt[:-1] or 0.0) / float(fps)
    if re.fullmatch(r"[+-]?\d+", txt):                  # integer text = frames
        return float(int(txt)) / float(fps)
    return float(txt)                                   # anything else = seconds


def _ffmpeg_exe() -> str:
    """System ffmpeg, else the binary imageio-ffmpeg ships, else give up.

    System ffmpeg already decodes every container in VIDEO_SUFFIXES; the
    imageio-ffmpeg fallback exists so a machine without a system ffmpeg still
    works without anything being installed here.
    """
    from shutil import which
    exe = which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise SystemExit(
            "no ffmpeg available: install ffmpeg, or `pip install "
            f"imageio-ffmpeg` in this environment ({exc})")


def probe_video(path: str) -> dict:
    """Container metadata via ffprobe: size, fps, duration, usable chunks."""
    import json as _json
    import subprocess
    from shutil import which
    info = {"path": str(path), "width": 0, "height": 0, "fps": 0.0,
            "duration": 0.0, "frames_at_encode_fps": 0, "chunks": 0, "error": ""}
    exe = which("ffprobe")
    if exe is None:
        info["error"] = "ffprobe not found"
        return info
    try:
        out = subprocess.run(
            [exe, "-v", "error", "-select_streams", "v:0", "-show_streams",
             "-show_format", "-of", "json", str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        blob = _json.loads(out.stdout.decode(errors="replace") or "{}")
        st = (blob.get("streams") or [{}])[0]
        info["width"] = int(st.get("width") or 0)
        info["height"] = int(st.get("height") or 0)
        num, _, den = str(st.get("avg_frame_rate") or "0/1").partition("/")
        den = float(den or 1) or 1.0
        info["fps"] = float(num or 0) / den
        dur = st.get("duration") or (blob.get("format") or {}).get("duration") or 0
        info["duration"] = float(dur or 0.0)
        # What matters downstream is how many frames survive the 20 fps
        # resample, not how many the file holds.
        info["frames_at_encode_fps"] = int(info["duration"] * SEED_ENCODE_FPS)
        info["chunks"] = min(seed_chunks_for_frames(info["frames_at_encode_fps"]),
                             SEED_PREFILL_CHUNKS)
    except Exception as exc:
        info["error"] = str(exc)
    return info


def read_video_frames(path: str, n: int = SEED_PIXEL_FRAMES,
                      start_s: float = 0.0, fps: float = SEED_ENCODE_FPS,
                      fisheye: float = DEFAULT_FISHEYE,
                      crop: str = DEFAULT_CROP_BAND) -> np.ndarray:
    """Up to ``n`` frames in TRAINING SPACE: uint8 [k, 480, 832, 3], k <= n.

    A SHORT read is not an error here; the caller decides how many whole
    chunks that supports.  See ``_seed_filter_chain`` for the transform and
    why each stage is where it is.
    """
    import subprocess
    vf = _seed_filter_chain(fisheye=fisheye, crop=crop, fps=fps)
    cmd = [_ffmpeg_exe(), "-v", "error"]
    if start_s > 0:
        # Before -i: ffmpeg seeks and then decodes to the exact point, which
        # is both fast and accurate on every container we accept.
        cmd += ["-ss", f"{float(start_s):.6f}"]
    cmd += ["-i", str(path), "-vf", vf, "-frames:v", str(int(n)),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: "
                           f"{proc.stderr.decode(errors='replace')[:300]}")
    frame_bytes = PIXEL_W * PIXEL_H * 3
    got = len(proc.stdout) // frame_bytes
    if got < 1:
        raise SystemExit(f"{path}: no decodable frames at start={start_s:.2f}s")
    # np.frombuffer gives a READ-ONLY view of the pipe bytes; copy so callers
    # (torch.from_numpy) get a writable array instead of an undefined-behaviour
    # warning.
    arr = np.frombuffer(proc.stdout[:got * frame_bytes], dtype=np.uint8).copy()
    return arr.reshape(got, PIXEL_H, PIXEL_W, 3)


def load_seed_from_video(path: str, wan_model_path: str = "/home/ashish/Wan2.1/",
                         device: str = "cuda", start=0,
                         cache: bool = True, vae=None,
                         seed_chunks: int = SEED_PREFILL_CHUNKS,
                         start_s: Optional[float] = None,
                         fisheye: float = DEFAULT_FISHEYE,
                         crop: str = DEFAULT_CROP_BAND):
    """Encode a local video's opening into seed latents ``[1, 3*k, 16, 60, 104]``.

    ``k`` is the number of whole seed chunks the clip actually supports, which
    is ``seed_chunks`` for anything long enough and fewer for a short clip
    (logged loudly).  The result is in the SAME normalised Wan latent space
    the zarr rides use, so the engine consumes it without knowing the origin.

    Encodes are cached under ``interactive/video_seeds/`` keyed by path, mtime,
    size, start offset, fps AND the requested chunk count -- asking for seven
    chunks must never be served a three-chunk cache entry.
    """
    import hashlib
    import torch

    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"seed video not found: {path}")
    want_chunks = max(int(seed_chunks), 1)
    off_s = float(start_s) if start_s is not None else parse_start_spec(start)
    need_px = seed_pixel_frames(want_chunks)

    st = p.stat()
    # Every knob that changes the PIXELS must be in the key, or a re-tuned
    # fisheye / a different crop band would be served a stale encode.
    key = hashlib.sha256(
        f"{p.resolve()}|{st.st_mtime_ns}|{st.st_size}|{off_s:.6f}|"
        f"{SEED_ENCODE_FPS:g}|{want_chunks}|{need_px}|"
        f"fe{float(fisheye or 0.0):.4f}|cr{crop}|ar{CAMERA_AR:.6f}"
        .encode()).hexdigest()[:32]
    cpath = VIDEO_SEED_DIR / f"{key}.pt"
    if cache and cpath.is_file():
        try:
            lat = torch.load(cpath, map_location="cpu", weights_only=False)
            log.info("Video seed from cache (%s) -> %s (%d chunks)",
                     cpath.name, tuple(lat.shape), lat.shape[1] // NUM_FRAME_PER_BLOCK)
            return lat
        except Exception as exc:
            log.warning("video seed cache %s unreadable (%s); re-encoding.",
                        cpath.name, exc)

    frames = read_video_frames(str(p), need_px, start_s=off_s,
                               fps=SEED_ENCODE_FPS, fisheye=fisheye, crop=crop)
    got_chunks = seed_chunks_for_frames(len(frames))
    if got_chunks < 1:
        raise SystemExit(
            f"{p.name}: only {len(frames)} frames at {SEED_ENCODE_FPS:g} fps from "
            f"start={off_s:.2f}s -- need at least "
            f"{seed_pixel_frames(1)} for a single seed chunk.")
    if got_chunks < want_chunks:
        log.warning("video supports %d/%d seed chunks (%d frames at %g fps "
                    "from %.2fs; %d needed for %d chunks). Seeding with %d.",
                    got_chunks, want_chunks, len(frames), SEED_ENCODE_FPS, off_s,
                    need_px, want_chunks, got_chunks)
    use_px = seed_pixel_frames(got_chunks)
    frames = frames[:use_px]

    log.warning(
        "SEEDING FROM AN ARBITRARY VIDEO (%s). The model was trained on "
        "frodobots first-person sidewalk rides; footage with different optics, "
        "motion or content is OUT OF DISTRIBUTION and the rollout may drift or "
        "collapse. Expect worse behaviour than a real ride seed.", p.name)

    own_vae = vae is None
    if own_vae:
        import utils.wan_wrapper as ww
        from utils.wan_wrapper import WanVAEWrapper
        wp = wan_model_path if wan_model_path.endswith("/") else wan_model_path + "/"
        ww._default_wan_model_path = wp
        vae = WanVAEWrapper().to(device, torch.bfloat16).eval()

    # [n,H,W,3] uint8 -> [1,3,n,H,W] in [-1, 1], the VAE's input convention.
    x = torch.from_numpy(np.ascontiguousarray(frames)).to(device)
    x = x.permute(3, 0, 1, 2).unsqueeze(0).to(torch.bfloat16)
    x = x / 127.5 - 1.0
    with torch.no_grad():
        lat = vae.encode_to_latent(x).float().cpu()
    if own_vae:
        del vae
        torch.cuda.empty_cache()

    need_lat = got_chunks * NUM_FRAME_PER_BLOCK
    if lat.shape[1] < need_lat:
        raise SystemExit(f"video seed produced {lat.shape[1]} latent frames, "
                         f"need {need_lat}")
    lat = lat[:, :need_lat]
    if cache:
        try:
            VIDEO_SEED_DIR.mkdir(parents=True, exist_ok=True)
            tmp = cpath.with_suffix(".pt.tmp")
            torch.save(lat, tmp)
            tmp.replace(cpath)
            log.info("Video seed cached -> %s", cpath)
        except Exception as exc:
            log.warning("could not cache video seed: %s", exc)
    log.info("Video seed %s: %d px frames @%g fps from %.2fs "
             "(crop=%s->%.4f AR, fisheye=%.3f, squash to %dx%d) -> %s (%d chunks)",
             p.name, len(frames), SEED_ENCODE_FPS, off_s, crop, CAMERA_AR,
             float(fisheye or 0.0), PIXEL_W, PIXEL_H, tuple(lat.shape), got_chunks)
    return lat


def list_videos(video_dir: Optional[str] = None) -> List[RideInfo]:
    """User videos as pickable entries, newest first.

    Searched: ``--video_dir`` if given, else interactive/user_videos/ and
    ~/Videos.  The drop-folder is created (empty) so its existence documents
    itself the first time the picker runs.
    """
    roots: List[Path] = []
    if video_dir:
        roots.append(Path(video_dir).expanduser())
    else:
        try:
            USER_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        roots.extend(DEFAULT_VIDEO_DIRS)
    out: List[RideInfo] = []
    seen = set()
    for root in roots:
        if not root.is_dir():
            continue
        for f in sorted(root.iterdir(), key=lambda q: -q.stat().st_mtime
                        if q.is_file() else 0):
            if not f.is_file() or f.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            rp = str(f.resolve())
            if rp in seen:
                continue
            seen.add(rp)
            # n_latents is filled by probe_ride() lazily, as for rides.
            out.append(RideInfo(path=str(f), ride_id=f.name, n_latents=-1))
    log.info("video dirs %s -> %d videos",
             [str(r) for r in roots], len(out))
    return out


# ---------------------------------------------------------------------------
# Thumbnails
# ---------------------------------------------------------------------------
class ThumbnailCache:
    """Lazily decodes and caches one JPEG per ride.

    The decoder is built on first real use, so listing/filtering rides costs
    no GPU. Call ``close()`` to drop it when the picker exits.
    """

    def __init__(self, thumb_dir: Path = THUMB_DIR, device: str = "cuda",
                 decoder_name: str = "taew2_1", gpu_gate=None):
        self.thumb_dir = Path(thumb_dir)
        self.thumb_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.decoder_name = decoder_name
        self._dec = None
        # When the engine is being built concurrently it owns the GPU; this
        # gate stays clear until it is done. Disk-cached thumbnails are served
        # throughout -- only fresh decodes wait.
        self.gpu_gate = gpu_gate
        self.deferred: set = set()

    @property
    def gpu_ready(self) -> bool:
        return self.gpu_gate is None or self.gpu_gate.is_set()

    # -- decoder lifecycle ------------------------------------------------
    def _decoder(self):
        if self._dec is None:
            import torch
            from interactive.decoders import build_decoder
            dev = self.device if torch.cuda.is_available() else "cpu"
            self._dec = build_decoder(self.decoder_name, device=dev,
                                      dtype=torch.bfloat16)
            log.info("thumbnail decoder: %s on %s", self.decoder_name, dev)
        return self._dec

    def close(self) -> None:
        if self._dec is not None:
            self._dec = None
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    # -- public -----------------------------------------------------------
    def path_for(self, ride_id: str) -> Path:
        # Video ids are bare filenames from arbitrary folders, so they are
        # hashed to keep the thumb cache collision-free and path-safe.
        if any(ch in ride_id for ch in "/\\ .") or len(ride_id) > 64:
            import hashlib
            h = hashlib.sha256(ride_id.encode()).hexdigest()[:24]
            return self.thumb_dir / f"vid_{h}.jpg"
        return self.thumb_dir / f"{ride_id}.jpg"

    def has(self, ride_id: str) -> bool:
        return self.path_for(ride_id).is_file()

    def get(self, ride: RideInfo) -> Optional[Path]:
        """Return the JPEG path for ``ride``, decoding it if needed.

        Returns None when the ride is too short or decoding fails; the caller
        renders those cells as unpickable.
        """
        out = self.path_for(ride.ride_id)
        if out.is_file():
            self.deferred.discard(ride.ride_id)
            return out
        if not probe_ride(ride).usable:
            return None
        if not self.gpu_ready and not ride.is_video:
            # Engine build owns the GPU; come back for this one later.
            self.deferred.add(ride.ride_id)
            return None
        self.deferred.discard(ride.ride_id)
        try:
            img = self._render(ride)
        except Exception as exc:
            log.warning("thumb failed for %s: %s", ride.ride_id, exc)
            return None
        try:
            from PIL import Image
            Image.fromarray(img).save(out, quality=88)
        except Exception as exc:
            log.warning("thumb save failed for %s: %s", ride.ride_id, exc)
            return None
        return out

    def _render(self, ride: RideInfo) -> np.ndarray:
        """Chunk 0 of a ride, or frame 0 of a video -> uint8 RGB [H, W, 3]."""
        if ride.is_video:
            # No VAE round-trip needed (and no GPU): the first cropped frame
            # IS what the model will be seeded with.
            # SAME transform the engine will apply, so the thumbnail is
            # literally what the model gets seeded with -- squash included.
            fr = read_video_frames(ride.path, n=1, start_s=0.0,
                                   fps=SEED_ENCODE_FPS)
            return np.ascontiguousarray(fr[0, ::2, ::2])
        import torch
        from utils.play_world_model import load_seed_from_zarr

        dec = self._decoder()
        lat = load_seed_from_zarr(ride.path, NUM_FRAME_PER_BLOCK)
        lat = lat[:, :NUM_FRAME_PER_BLOCK]
        dev = next(iter([getattr(dec, "device", "cpu")]))
        # reset() before every ride: TAEHV carries temporal state between
        # chunks, and these chunks come from unrelated videos.
        dec.reset()
        frames = dec.decode_chunk(lat.to(device=dev, dtype=torch.bfloat16),
                                  half_res=True)
        return frames[0].detach().cpu().numpy()

    def ensure_page(self, rides: Sequence[RideInfo],
                    progress: Optional[Callable[[int, int], None]] = None
                    ) -> List[Optional[Path]]:
        """Decode every missing thumbnail for one page."""
        out: List[Optional[Path]] = []
        for i, ride in enumerate(rides):
            out.append(self.get(ride))
            if progress is not None:
                progress(i + 1, len(rides))
        return out


# ---------------------------------------------------------------------------
# pygame browser
# ---------------------------------------------------------------------------
class SeedPicker:
    """Paginated grid of ride thumbnails.

    Keys: arrows move, PgUp/PgDn page, digits filter, BACKSPACE edits the
    filter, ENTER selects, ESC cancels.
    """

    BG = (14, 14, 20)
    FG = (225, 225, 225)
    DIM = (120, 120, 130)
    SEL = (255, 205, 90)
    BAD = (170, 70, 70)
    VID = (110, 200, 255)      # user-video accent

    def __init__(self, screen, rides: Sequence[RideInfo],
                 cache: Optional[ThumbnailCache] = None,
                 font=None, small_font=None):
        import pygame as pg
        self.pg = pg
        self.screen = screen
        self.all_rides = list(rides)
        self.cache = cache or ThumbnailCache()
        self.font = font or pg.font.SysFont("monospace", 20)
        self.small_font = small_font or pg.font.SysFont("monospace", 15)

        self.query = ""
        self.rides = list(self.all_rides)
        self.page = 0
        self.cursor = 0
        self._surf_cache: dict = {}

    # -- state ------------------------------------------------------------
    @property
    def n_pages(self) -> int:
        return max(1, (len(self.rides) + PAGE_SIZE - 1) // PAGE_SIZE)

    def page_rides(self) -> List[RideInfo]:
        start = self.page * PAGE_SIZE
        return self.rides[start:start + PAGE_SIZE]

    def _reflow(self) -> None:
        self.rides = filter_rides(self.all_rides, self.query)
        self.page = 0
        self.cursor = 0

    def _move(self, dcol: int, drow: int) -> None:
        page = self.page_rides()
        if not page:
            return
        col = self.cursor % GRID_COLS + dcol
        row = self.cursor // GRID_COLS + drow
        if col < 0:
            col = GRID_COLS - 1
            row -= 1
        elif col >= GRID_COLS:
            col = 0
            row += 1
        if row < 0:
            if self.page > 0:
                self.page -= 1
                row = GRID_ROWS - 1
            else:
                row = 0
        elif row >= GRID_ROWS:
            if self.page + 1 < self.n_pages:
                self.page += 1
                row = 0
            else:
                row = GRID_ROWS - 1
        self.cursor = min(row * GRID_COLS + col, max(0, len(self.page_rides()) - 1))

    def _turn_page(self, delta: int) -> None:
        self.page = max(0, min(self.n_pages - 1, self.page + delta))
        self.cursor = 0

    def selected(self) -> Optional[RideInfo]:
        page = self.page_rides()
        return page[self.cursor] if 0 <= self.cursor < len(page) else None

    # -- rendering --------------------------------------------------------
    def _thumb_surface(self, ride: RideInfo):
        # A deferred ride has no thumbnail YET (the GPU was busy), so do not
        # let the negative result stick in the surface cache.
        if ride.ride_id in self.cache.deferred and self.cache.gpu_ready:
            self._surf_cache.pop(ride.ride_id, None)
        if ride.ride_id in self._surf_cache:
            return self._surf_cache[ride.ride_id]
        path = self.cache.get(ride)
        surf = None
        if path is not None:
            try:
                surf = self.pg.image.load(str(path)).convert()
            except Exception as exc:
                log.warning("load thumb %s: %s", path, exc)
        self._surf_cache[ride.ride_id] = surf
        return surf

    def draw(self, status: str = "") -> None:
        pg = self.pg
        self.screen.fill(self.BG)
        sw, sh = self.screen.get_size()

        n_vid = sum(1 for r in self.all_rides if r.is_video)
        head = (f"SELECT STARTING SEED   {len(self.rides)}/{len(self.all_rides)} "
                f"entries" + (f" ({n_vid} [VID])" if n_vid else "")
                + f"   page {self.page + 1}/{self.n_pages}")
        self.screen.blit(self.font.render(head, True, self.FG), (20, 14))
        filt = f"filter: {self.query}_" if self.query else "filter: (type digits)"
        self.screen.blit(self.small_font.render(filt, True, self.SEL), (20, 40))

        top, pad = 64, 12
        foot = 46
        cw = (sw - 2 * pad - (GRID_COLS - 1) * pad) // GRID_COLS
        ch = (sh - top - foot - (GRID_ROWS - 1) * pad) // GRID_ROWS
        page = self.page_rides()

        for i, ride in enumerate(page):
            col, row = i % GRID_COLS, i // GRID_COLS
            x = pad + col * (cw + pad)
            y = top + row * (ch + pad)
            sel = (i == self.cursor)
            surf = self._thumb_surface(ride)
            if surf is not None:
                img = pg.transform.smoothscale(surf, (cw, ch - 20))
                self.screen.blit(img, (x, y))
            else:
                if ride.ride_id in self.cache.deferred:
                    pg.draw.rect(self.screen, (26, 26, 40), (x, y, cw, ch - 20))
                    msg, colr = "loading (engine building)...", self.DIM
                else:
                    pg.draw.rect(self.screen, (40, 28, 28), (x, y, cw, ch - 20))
                    msg, colr = ("too short" if ride.probed else "unreadable"), self.BAD
                self.screen.blit(
                    self.small_font.render(msg, True, colr), (x + 8, y + 8))
            colr = self.SEL if sel else (self.FG if surf is not None else self.DIM)
            label = ride.ride_id
            if ride.is_video:
                # Distinct tag + accent so a user clip is never mistaken for a
                # real ride: they behave differently (OOD, no GT actions).
                label = f"[VID] {label}"
                if not sel:
                    colr = self.VID
                pg.draw.rect(self.screen, self.VID, (x, y, cw, ch - 20), 2)
                chunks = (ride.n_latents // NUM_FRAME_PER_BLOCK) if ride.probed else 0
                if ride.probed and 0 < chunks < SEED_PREFILL_CHUNKS:
                    tagl = self.small_font.render(
                        f"{chunks}/{SEED_PREFILL_CHUNKS} chunks", True, self.BAD)
                    self.screen.blit(tagl, (x + 6, y + 6))
            if len(label) > 30:
                label = label[:14] + ".." + label[-14:]
            self.screen.blit(
                self.small_font.render(label, True, colr), (x + 2, y + ch - 18))
            if sel:
                pg.draw.rect(self.screen, self.SEL, (x - 2, y - 2, cw + 4, ch + 4), 2)

        tip = ("arrows move   PgUp/PgDn page   type to filter   BACKSPACE del   "
               "ENTER select   ESC cancel      [VID] = your own video "
               "(drop clips in interactive/user_videos/)")
        self.screen.blit(self.small_font.render(tip, True, self.DIM), (20, sh - 38))
        if status:
            self.screen.blit(self.small_font.render(status, True, self.SEL),
                             (20, sh - 20))
        pg.display.flip()

    # -- loop -------------------------------------------------------------
    def run(self) -> Optional[str]:
        """Modal loop. Returns the chosen zarr path, or None if cancelled."""
        pg = self.pg
        self.draw("decoding thumbnails...")
        while True:
            for ev in pg.event.get():
                if ev.type == pg.QUIT:
                    return None
                if ev.type != pg.KEYDOWN:
                    continue
                k = ev.key
                if k == pg.K_ESCAPE:
                    return None
                if k == pg.K_RETURN:
                    ride = self.selected()
                    if ride is not None and probe_ride(ride).usable:
                        return ride.path
                    self.draw("that ride is too short to seed from")
                    continue
                if k == pg.K_BACKSPACE:
                    self.query = self.query[:-1]
                    self._reflow()
                elif k == pg.K_PAGEUP:
                    self._turn_page(-1)
                elif k == pg.K_PAGEDOWN:
                    self._turn_page(+1)
                elif k == pg.K_LEFT:
                    self._move(-1, 0)
                elif k == pg.K_RIGHT:
                    self._move(+1, 0)
                elif k == pg.K_UP:
                    self._move(0, -1)
                elif k == pg.K_DOWN:
                    self._move(0, +1)
                elif ev.unicode and ev.unicode.isdigit():
                    self.query += ev.unicode
                    self._reflow()
                else:
                    continue
                self.draw()
            self.draw()
            pg.time.wait(16)


def pick_interactive(screen, seed_root: Optional[str] = None,
                     cache: Optional[ThumbnailCache] = None,
                     font=None, small_font=None) -> Optional[str]:
    """Convenience wrapper: list rides, run the picker, return a zarr path."""
    picker = SeedPicker(screen, list_rides(seed_root), cache=cache,
                        font=font, small_font=small_font)
    return picker.run()


# ---------------------------------------------------------------------------
# Headless smoke
# ---------------------------------------------------------------------------
def smoke(seed_root: Optional[str] = None, n_thumbs: int = 12) -> int:
    """Exercise listing, filtering, probing, thumbnails and pick_random."""
    problems: List[str] = []

    rides = list_rides(seed_root)
    print(f"[seedsmoke] rides: {len(rides)} under {resolve_seed_root(seed_root)}")
    if len(rides) < 1:
        print("[seedsmoke] FAIL: no rides")
        return 1
    print(f"[seedsmoke] first ids: {[r.ride_id for r in rides[:4]]}")

    # filter
    q = rides[0].ride_id[:6]
    hit = filter_rides(rides, q)
    print(f"[seedsmoke] filter {q!r} -> {len(hit)} rides")
    if not hit or any(q not in r.ride_id for r in hit):
        problems.append("filter returned non-matching rides")
    if len(filter_rides(rides, "")) != len(rides):
        problems.append("empty filter should return everything")
    if filter_rides(rides, "zzzznope"):
        problems.append("nonsense filter should return nothing")

    # probe + thumbnails
    cache = ThumbnailCache()
    made, skipped = 0, 0
    t0 = time.time()
    for ride in rides:
        if made >= n_thumbs:
            break
        path = cache.get(ride)
        if path is None:
            skipped += 1
            continue
        made += 1
    dt = time.time() - t0
    cache.close()
    print(f"[seedsmoke] thumbnails: {made} made/cached, {skipped} unpickable, "
          f"{dt:.1f}s -> {THUMB_DIR}")
    if made < n_thumbs:
        problems.append(f"only {made} thumbnails (< {n_thumbs})")
    on_disk = len(list(THUMB_DIR.glob('*.jpg')))
    print(f"[seedsmoke] cache now holds {on_disk} jpgs")

    # pick_random
    r = pick_random(seed_root)
    print(f"[seedsmoke] pick_random -> {r.ride_id} ({r.n_latents} latents)")
    if not r.usable:
        problems.append("pick_random returned an unusable ride")
    if not Path(r.path).exists():
        problems.append("pick_random path does not exist")

    for p in problems:
        print(f"[seedsmoke] FAIL: {p}")
    print("[seedsmoke] PASS" if not problems else "[seedsmoke] FAILED")
    return 1 if problems else 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="seed picker headless smoke")
    ap.add_argument("--seed_root", default=None)
    ap.add_argument("--n_thumbs", type=int, default=12)
    ap.add_argument("--smoke", action="store_true")
    _a = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [seed_picker] %(levelname)s | %(message)s")
    raise SystemExit(smoke(_a.seed_root, _a.n_thumbs))
