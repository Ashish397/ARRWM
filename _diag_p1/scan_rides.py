"""Scan a few rides and report scene-shift MAE per 3-frame offset to
pick a high-motion clip for the diagnostic test."""
from __future__ import annotations
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from utils.zarr_dataset import ZarrRideDataset, _index_single_zarr

ENCODED_ROOT = Path("/home/ashish/frodobots/frodobots_encoded")
CAPTION_ROOT = Path("/home/ashish/frodobots/frodobots_captions/train")
N = 21
CF = 3
SAMPLE_FRAMES = 121  # need s+cf+N up to 121 → s up to 97

zarrs = sorted(ENCODED_ROOT.glob("*.zarr"))[:30]
print(f"scanning {len(zarrs)} zarrs ...")
results = []
for zp in zarrs:
    try:
        _, attrs, n_lat = _index_single_zarr(zp, CAPTION_ROOT)
    except Exception as e:
        continue
    if n_lat < SAMPLE_FRAMES:
        continue
    # Load N+CF latents from positions [0:24] (s=0) and [97:121] (s=97)
    # and report scene-shift MAE between clean_context and noisy_window
    # at each offset.
    # load_latent_chunk(start, end) uses end as ABSOLUTE-EXCLUSIVE index.
    lat_top = ZarrRideDataset.load_latent_chunk(str(zp), 0, CF + N).float()
    lat_deep = ZarrRideDataset.load_latent_chunk(str(zp), 97, 97 + CF + N).float()
    mae_top = (lat_top[:N] - lat_top[CF:CF + N]).abs().mean().item()
    mae_deep = (lat_deep[:N] - lat_deep[CF:CF + N]).abs().mean().item()
    # Also probe a few mid offsets to find the highest-motion window in the ride.
    best_offset, best_mae = 0, mae_top
    for s in [50, 100, 200, 400, 800, 1600] if n_lat >= 1600 + CF + N else [50, 100, 200]:
        if s + CF + N > n_lat:
            break
        lat_s = ZarrRideDataset.load_latent_chunk(str(zp), s, s + CF + N).float()
        m = (lat_s[:N] - lat_s[CF:CF + N]).abs().mean().item()
        if m > best_mae:
            best_offset, best_mae = s, m
    results.append((zp.name, n_lat, mae_top, mae_deep, best_offset, best_mae))

results.sort(key=lambda r: r[5], reverse=True)
print(f"{'ride':45s}  {'n_frames':>9s}  {'mae@s=0':>8s}  {'mae@s=97':>9s}  {'best_off':>8s}  {'best_mae':>8s}")
for name, n_lat, mae_top, mae_deep, off, m in results[:15]:
    print(f"{name:45s}  {n_lat:>9d}  {mae_top:>8.4f}  {mae_deep:>9.4f}  {off:>8d}  {m:>8.4f}")
