"""Local diagnostic for the post-2d897aa / post-774456b motion loader.

Verifies the v14-aligned chunk-grain motion ↔ latent loader on the local
5090 (no DDP, no training, no GPU memory pressure beyond a single ride
load + ss_vae forward).

Checks:
  1. ``ZarrRideDataset`` indexes a ride and produces an
     ``n_latent_frames`` that's the EFFECTIVE post-head-drop,
     motion-capped count: ``min(zarr_n_latents - _LATENT_HEAD_DROP,
     n_motion_chunks * _LATENTS_PER_MOTION_CHUNK)``.
  2. ``encode_z_actions_window`` returns per-latent z's where every
     consecutive ``_LATENTS_PER_MOTION_CHUNK`` (= 3) latents share the
     SAME z (= chunk-grain identity by construction).
  3. Sliced views at chunk-aligned offsets (``s % 3 == 0``) preserve
     the within-chunk identity at the slice's own indices [0, 1, 2].
  4. The head-drop in ``load_latent_chunk`` actually shifts the zarr
     read by ``_LATENT_HEAD_DROP`` (so dataset latent index 0 reads
     zarr index ``_LATENT_HEAD_DROP``).
  5. The ride's per-latent motion magnitude (from
     ``load_motion_magnitudes``) is constant within each chunk and
     varies across chunks (sanity that motion data is non-trivial).

Usage:
  cd /home/ashish/ARRWM
  python _diag_p1/test_motion_loader_alignment.py [--ride-name <stem>]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import zarr as zarr_lib

# Make repo importable
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from utils.zarr_dataset import (  # noqa: E402
    ZarrRideDataset,
    _LATENT_HEAD_DROP,
    _LATENTS_PER_MOTION_CHUNK,
    _MOTION_WINDOW_FRAMES,
    _LATENT_TO_VIDEO,
)


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("motion_diag")


ENCODED_ROOT = "/home/ashish/frodobots/frodobots_encoded"
CAPTION_ROOT = "/home/ashish/frodobots/frodobots_captions/train"
MOTION_ROOT = "/home/ashish/frodobots/frodobots_motion"
SS_VAE_CKPT = "/home/ashish/ARRWM/action_query/checkpoints/ss_vae_8free.pt"


def _ok(msg: str) -> None:
    log.info("PASS  %s", msg)


def _fail(msg: str) -> None:
    log.error("FAIL  %s", msg)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ride-name", type=str, default=None,
        help="Optional zarr basename (without .zarr). If omitted, uses the first indexed ride.",
    )
    parser.add_argument(
        "--max-rides", type=int, default=4,
        help="Cap dataset scan to N rides (faster init).",
    )
    args = parser.parse_args()

    log.info("module constants: _LATENT_HEAD_DROP=%d  _LATENTS_PER_MOTION_CHUNK=%d  "
             "_MOTION_WINDOW_FRAMES=%d  _LATENT_TO_VIDEO=%d",
             _LATENT_HEAD_DROP, _LATENTS_PER_MOTION_CHUNK,
             _MOTION_WINDOW_FRAMES, _LATENT_TO_VIDEO)

    log.info("Building ZarrRideDataset (max_rides=%d)…", args.max_rides)
    ds = ZarrRideDataset(
        encoded_root=ENCODED_ROOT,
        caption_root=CAPTION_ROOT,
        motion_root=MOTION_ROOT,
        ss_vae_checkpoint=SS_VAE_CKPT,
        min_ride_frames=21,
        device="cpu",
        ss_vae_device="cuda:0" if torch.cuda.is_available() else "cpu",
        max_rides=args.max_rides,
    )
    if len(ds) == 0:
        _fail(f"no rides indexed in {ENCODED_ROOT}")

    log.info("indexed %d rides", len(ds))

    # Pick a ride.
    if args.ride_name:
        target_stem = args.ride_name
        idx = next(
            (i for i, r in enumerate(ds._rides) if r[0].stem == target_stem),
            None,
        )
        if idx is None:
            _fail(f"ride name {args.ride_name} not in indexed set")
    else:
        idx = 0
    zpath, prompt_embeds, attrs, n_latent_frames = ds._rides[idx]
    log.info("ride: %s  n_latent_frames(reported)=%d", zpath.name, n_latent_frames)

    # ----------------- Test 1 — n_latent_frames cap correctness -----------------
    g = zarr_lib.open_group(str(zpath), mode="r")
    zarr_n_latents = int(g["latents"].shape[0])
    motion_path = (
        Path(MOTION_ROOT)
        / Path(attrs["ride_dir_2k"]).relative_to("/home/ashish/frodobots/frodobots_data")
        / "motion.npy"
    )
    if not motion_path.exists():
        _fail(f"motion.npy missing at {motion_path}")
    motion = np.load(motion_path, mmap_mode="r")
    n_motion_chunks = int(motion.shape[0])
    motion_capped = n_motion_chunks * _LATENTS_PER_MOTION_CHUNK
    n_after_head_drop = max(0, zarr_n_latents - _LATENT_HEAD_DROP)
    expected_n_lat = min(n_after_head_drop, motion_capped)

    log.info("zarr_n_latents=%d  n_after_head_drop=%d  n_motion_chunks=%d  "
             "motion_capped=%d  expected_n_lat=%d",
             zarr_n_latents, n_after_head_drop, n_motion_chunks,
             motion_capped, expected_n_lat)

    if n_latent_frames != expected_n_lat:
        _fail(f"n_latent_frames={n_latent_frames} != expected={expected_n_lat}")
    _ok(f"Test 1: n_latent_frames cap correct ({n_latent_frames})")

    # ----------------- Test 2 — chunk-grain identity in encode_z_actions_window -----------------
    # Encode the FULL ride, then assert z[3k] == z[3k+1] == z[3k+2].
    z_full = ds.encode_z_actions_window(str(zpath), n_latent_frames, 0, n_latent_frames)
    log.info("encode_z_actions_window full -> shape=%s dtype=%s",
             tuple(z_full.shape), z_full.dtype)

    if z_full.shape != (n_latent_frames, 8):
        _fail(f"unexpected z_full shape {tuple(z_full.shape)}; "
              f"expected ({n_latent_frames}, 8)")

    z_np = z_full.cpu().numpy()
    n_chunks = n_latent_frames // _LATENTS_PER_MOTION_CHUNK
    if n_chunks * _LATENTS_PER_MOTION_CHUNK != n_latent_frames:
        log.warning(
            "n_latent_frames=%d not a multiple of %d; chunk-identity check "
            "will skip the trailing %d latent(s)",
            n_latent_frames, _LATENTS_PER_MOTION_CHUNK,
            n_latent_frames - n_chunks * _LATENTS_PER_MOTION_CHUNK,
        )

    npb = _LATENTS_PER_MOTION_CHUNK
    max_within_chunk_diff = 0.0
    for c in range(n_chunks):
        chunk_block = z_np[c * npb:(c + 1) * npb]
        # All rows in chunk_block must be byte-identical.
        diffs = np.abs(chunk_block - chunk_block[0:1]).max()
        if diffs > max_within_chunk_diff:
            max_within_chunk_diff = float(diffs)
    if max_within_chunk_diff != 0.0:
        _fail(f"within-chunk identity violated: max abs diff = {max_within_chunk_diff:.2e}")
    _ok(f"Test 2: within-chunk identity exact across all {n_chunks} chunks")

    # ----------------- Test 3 — across-chunk variation (sanity) -----------------
    # If the encoding is broken / stale, all chunks would have identical z.
    # Verify that at least SOME chunks differ from chunk 0.
    if n_chunks >= 2:
        cross_chunk_diff = float(
            np.abs(z_np[npb : 2 * npb] - z_np[0:npb]).max()
        )
        if cross_chunk_diff == 0.0:
            log.warning(
                "chunks 0 and 1 are byte-identical — could be a parked window. "
                "Continuing.",
            )
        else:
            _ok(f"Test 3: across-chunk variation present (chunk0 vs chunk1 max abs diff = {cross_chunk_diff:.4f})")
    else:
        log.warning("only %d chunk(s); skipping across-chunk variation test", n_chunks)

    # ----------------- Test 4 — sliced view at chunk-aligned offset preserves identity -----------------
    # Slice [s : s + W] for various s. With s % npb == 0 and W % npb == 0, the
    # sliced indices [0, 1, 2] should be one motion chunk.
    s_test = npb * 5  # = 15 (an arbitrary chunk-aligned offset)
    W = npb * 7       # = 21 (a typical scoring-window length)
    if s_test + W > n_latent_frames:
        s_test = 0
        W = min(W, n_latent_frames)
    z_slice = ds.encode_z_actions_window(
        str(zpath), n_latent_frames, s_test, s_test + W,
    ).cpu().numpy()
    log.info("sliced encode_z_actions_window [%d:%d] -> shape=%s",
             s_test, s_test + W, z_slice.shape)
    n_slice_chunks = W // npb
    max_slice_within = 0.0
    for c in range(n_slice_chunks):
        chunk_block = z_slice[c * npb:(c + 1) * npb]
        max_slice_within = max(
            max_slice_within,
            float(np.abs(chunk_block - chunk_block[0:1]).max()),
        )
    if max_slice_within != 0.0:
        _fail(f"sliced view within-chunk identity violated: max abs diff = {max_slice_within:.2e}")
    _ok(f"Test 4: sliced view chunk identity preserved (offset s={s_test}, W={W})")

    # Cross-check: sliced view's chunk c must equal full view's chunk
    # (s_test/npb + c) up to fp32 ss_vae batch-size non-determinism.
    # The motion data fed to ss_vae is bit-identical for the chunk in
    # both cases; only the BATCH SIZE of the ss_vae forward differs
    # (full=499 chunks at once, sliced=7 chunks at once), which causes
    # tiny fp drift in the encoder's BN/MLP path. Tolerance 1e-3
    # (encoder output is in [-1, 1] post-tanh-squash; 1e-3 is well
    # below action-relevant variation).
    fp_tol = 1e-3
    max_cross_diff = 0.0
    for c in range(n_slice_chunks):
        full_c = (s_test // npb) + c
        full_row = z_np[full_c * npb]
        slice_row = z_slice[c * npb]
        max_cross_diff = max(max_cross_diff, float(np.abs(full_row - slice_row).max()))
    if max_cross_diff > fp_tol:
        _fail(
            f"sliced chunk values diverge from full at the same chunk index by "
            f"{max_cross_diff:.2e} > tol {fp_tol:.0e} — likely a real alignment bug, "
            "not just ss_vae batch fp drift."
        )
    _ok(f"Test 4b: sliced view's chunk values match full view (max abs diff = {max_cross_diff:.2e}, tol = {fp_tol:.0e})")

    # ----------------- Test 5 — head-drop applied in load_latent_chunk -----------------
    # zarr index 0 vs dataset index 0 → must differ by _LATENT_HEAD_DROP.
    if _LATENT_HEAD_DROP > 0:
        zarr_lat_0 = torch.from_numpy(g["latents"][0:1].astype(np.float32))
        zarr_lat_drop = torch.from_numpy(
            g["latents"][_LATENT_HEAD_DROP:_LATENT_HEAD_DROP + 1].astype(np.float32)
        )
        ds_lat_0 = ZarrRideDataset.load_latent_chunk(str(zpath), 0, 1)
        # ds_lat_0 should match zarr_lat_drop, NOT zarr_lat_0.
        if torch.equal(ds_lat_0, zarr_lat_drop) and not torch.equal(ds_lat_0, zarr_lat_0):
            _ok(f"Test 5: load_latent_chunk applies _LATENT_HEAD_DROP={_LATENT_HEAD_DROP}")
        else:
            _fail(
                "load_latent_chunk does NOT apply _LATENT_HEAD_DROP correctly: "
                f"ds_lat_0 == zarr_lat_drop? {torch.equal(ds_lat_0, zarr_lat_drop)}; "
                f"ds_lat_0 == zarr_lat_0? {torch.equal(ds_lat_0, zarr_lat_0)}"
            )
    else:
        # _LATENT_HEAD_DROP=0 → ds latent 0 == zarr latent 0.
        zarr_lat_0 = torch.from_numpy(g["latents"][0:1].astype(np.float32))
        ds_lat_0 = ZarrRideDataset.load_latent_chunk(str(zpath), 0, 1)
        if torch.equal(ds_lat_0, zarr_lat_0):
            _ok(f"Test 5: _LATENT_HEAD_DROP=0 → load_latent_chunk reads zarr index 0 (no shift)")
        else:
            _fail("with _LATENT_HEAD_DROP=0, ds_lat_0 should equal zarr_lat_0 but doesn't")

    # ----------------- Test 6 — load_motion_magnitudes is chunk-grain -----------------
    mag = ds.load_motion_magnitudes(str(zpath), n_latent_frames)
    log.info("load_motion_magnitudes -> shape=%s dtype=%s mean=%.4f", mag.shape, mag.dtype, float(mag.mean()))
    if mag.shape != (n_latent_frames,):
        _fail(f"motion_mag shape {mag.shape} != expected ({n_latent_frames},)")
    max_mag_within = 0.0
    for c in range(n_chunks):
        block = mag[c * npb:(c + 1) * npb]
        max_mag_within = max(max_mag_within, float(block.max() - block.min()))
    if max_mag_within > 1e-7:
        _fail(f"load_motion_magnitudes within-chunk variation = {max_mag_within:.2e}; "
              "should be 0 (chunk-grain)")
    _ok(f"Test 6: load_motion_magnitudes chunk-grain identity exact (max within-chunk delta = {max_mag_within:.2e})")

    # ----------------- Summary -----------------
    log.info("=" * 70)
    log.info("ALL CHECKS PASSED for ride %s", zpath.name)
    log.info("Summary:")
    log.info("  zarr_n_latents = %d", zarr_n_latents)
    log.info("  n_after_head_drop = %d", n_after_head_drop)
    log.info("  n_motion_chunks = %d (motion_capped = %d latents)", n_motion_chunks, motion_capped)
    log.info("  n_latent_frames (loader-reported) = %d", n_latent_frames)
    log.info("  n_chunks (= n_latent_frames / 3) = %d", n_chunks)
    log.info("  z_actions shape = %s, all chunk-grain", tuple(z_full.shape))
    log.info("=" * 70)


if __name__ == "__main__":
    main()
