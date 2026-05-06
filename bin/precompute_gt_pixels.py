"""Precompute GT pixel videos by decoding ride latents through the WAN VAE.

Stores the result as a uint8 zarr per ride at ``<cache_root>/<ride_name>.zarr``.
Re-uses the dummy-leading-frame trick that the trainer's
``_decode_no_grad`` uses, so the per-frame pixel content is byte-for-byte
identical to what the GAN would have decoded live.

Run via ``sbatch sbatch/precompute_gt_pixels.sbatch [n_rides]`` to GPU-decode
the first N rides (default 10). Smoke for the cache hit is at v20's
``smoke_action_forcing_phase1_gan_v20_action_aux.sbatch``.

Layout of the output zarr (per ride):
    cache_root/<ride_name>/
        pixels (uint8, [F_pix, 3, H_pix, W_pix])
        attrs:
            n_latent_frames (int)
            head_drop (int)            # dataset _LATENT_HEAD_DROP
            wan_pixel_resolution (tuple)
            decoded_at_resolution     # raw decode shape

The trainer reads pixel slice [chunk_lo*4 : chunk_hi*4] (= 4× temporal
expansion) for each chunk_size-frame latent window.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import zarr as zarr_lib

# Repo path setup so this can be run from anywhere.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _setup_logger() -> logging.Logger:
    log = logging.getLogger("precompute_gt_pixels")
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        log.addHandler(h)
    return log


def _decode_ride(vae, latents: torch.Tensor) -> torch.Tensor:
    """Run WAN VAE decode with the dummy-leading-frame trick. Mirrors
    the trainer's ``_decode_no_grad`` path. Returns
    ``[F_pix, 3, H_pix, W_pix]`` in [-1, 1] (fp32).
    """
    # latents shape: [F_lat, 16, 60, 104]; expected by vae.decode_to_pixel
    # in [B, F, C, H, W] form.
    assert latents.dim() == 4, f"expected [F, C, H, W], got {latents.shape}"
    lat = latents.unsqueeze(0)  # [1, F_lat, 16, 60, 104]
    dummy = lat[:, 0:1]
    lat_pad = torch.cat([dummy, lat], dim=1)
    pix = vae.decode_to_pixel(lat_pad, use_cache=False)
    # Strip the dummy frame's pixel output (the 1-frame "first" decode
    # output) — same trick as the trainer.
    pix = pix[:, 1:, ...]
    pix = pix.squeeze(0)  # [F_pix, 3, H, W]
    return pix


def _save_pixels_uint8(out_zarr: Path, pixels: torch.Tensor, n_lat: int) -> None:
    """Quantize fp32 [-1, 1] pixels to uint8 [0, 255] and write zarr.

    Lossy quantization but SAM2 is robust to it (it was trained on real
    images). The dequantize in the loader does the inverse.
    """
    if pixels.min() < -1.01 or pixels.max() > 1.01:
        raise RuntimeError(
            f"pixels out of [-1, 1] range: min={pixels.min().item():.4f} "
            f"max={pixels.max().item():.4f}"
        )
    pix_np = pixels.detach().to(torch.float32).clamp_(-1.0, 1.0).cpu().numpy()
    quantized = ((pix_np + 1.0) * 127.5).round().clip(0, 255).astype(np.uint8)
    F_pix, C, H, W = quantized.shape
    g = zarr_lib.open_group(str(out_zarr), mode="w")
    arr = g.create_dataset(
        "pixels",
        shape=quantized.shape,
        dtype=quantized.dtype,
        chunks=(min(8, F_pix), C, H, W),
    )
    arr[...] = quantized
    g.attrs["n_latent_frames"] = int(n_lat)
    g.attrs["pixel_shape"] = list(quantized.shape)
    g.attrs["scale"] = 127.5
    g.attrs["offset"] = -1.0  # decoded value = (uint8 / 127.5) + offset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=str, required=True,
                   help="Path to .ride_manifest.pt (e.g. v6 manifest)")
    p.add_argument("--out", type=str, required=True,
                   help="Output cache root dir")
    p.add_argument("--n_rides", type=int, default=10,
                   help="Decode first N rides only (default 10)")
    p.add_argument("--start", type=int, default=0,
                   help="Start ride index")
    p.add_argument("--decode_chunk", type=int, default=21,
                   help="Latent frames per VAE decode call (memory bound)")
    args = p.parse_args()

    log = _setup_logger()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # Lazy imports so the script can be parsed without the full repo env.
    from utils.zarr_dataset import ZarrRideDataset, _LATENT_HEAD_DROP
    from utils.wan_wrapper import WanVAEWrapper

    # Load the manifest pickle directly — ZarrRideDataset.from_manifest
    # would also try to load ss_vae which we don't need for decode-only.
    log.info("Loading manifest from %s", args.manifest)
    rides_data = torch.load(args.manifest, weights_only=False)
    if isinstance(rides_data, dict) and "rides" in rides_data:
        rides_data = rides_data["rides"]
    elif not isinstance(rides_data, list):
        raise RuntimeError(
            f"unexpected manifest format: {type(rides_data)}"
        )
    log.info("Manifest has %d rides", len(rides_data))
    end_idx = min(args.start + args.n_rides, len(rides_data))
    log.info("Will decode rides [%d, %d)", args.start, end_idx)

    def _ride_iter():
        for r in rides_data[args.start:end_idx]:
            yield {
                "zarr_path": r["zarr_path"],
                "n_latent_frames": int(r["n_latent_frames"]),
            }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Loading WAN VAE on %s", device)
    vae = WanVAEWrapper().to(device=device, dtype=torch.float32)
    vae.eval()

    t0 = time.time()
    items = list(_ride_iter())
    for offset, item in enumerate(items):
        idx = args.start + offset
        zpath = item["zarr_path"]
        n_lat = int(item["n_latent_frames"])
        ride_name = Path(zpath).stem
        out_zarr = out_root / f"{ride_name}.zarr"
        if out_zarr.exists():
            log.info("[%d/%d] %s already exists — skipping", idx, end_idx,
                     ride_name)
            continue
        log.info("[%d/%d] decoding %s (n_lat=%d)", idx, end_idx, ride_name,
                 n_lat)
        # Stream latents in decode_chunk-sized blocks; concatenate.
        all_pixels = []
        with torch.no_grad():
            for s in range(0, n_lat, args.decode_chunk):
                e = min(s + args.decode_chunk, n_lat)
                lat = ZarrRideDataset.load_latent_chunk(zpath, s, e)
                lat = lat.to(device=device, dtype=torch.float32)
                pix = _decode_ride(vae, lat)  # [F_pix, 3, H, W]
                all_pixels.append(pix.cpu())
                del lat, pix
                torch.cuda.empty_cache()
        full_pixels = torch.cat(all_pixels, dim=0)
        del all_pixels
        log.info("[%d/%d] %s decoded → pixels=%s, writing zarr…",
                 idx, end_idx, ride_name, tuple(full_pixels.shape))
        _save_pixels_uint8(out_zarr, full_pixels, n_lat)
        del full_pixels
        torch.cuda.empty_cache()
        log.info("[%d/%d] %s done (elapsed=%.1fs)",
                 idx, end_idx, ride_name, time.time() - t0)
    log.info("All rides processed (total elapsed=%.1fs)", time.time() - t0)


if __name__ == "__main__":
    main()
