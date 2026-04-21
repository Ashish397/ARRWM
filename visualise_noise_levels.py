#!/usr/bin/env python3
"""Visualise what different noise augmentation levels look like on real latents.

Decodes a sample ride's latents at noise levels 0, 100, 200, 300 and saves
side-by-side comparison videos to vis/ folder.
"""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, ".")

from pathlib import Path
from utils.scheduler import FlowMatchScheduler


def decode_latents_to_frames(vae_wrapper, latents, device="cuda"):
    """Decode [1, T, C, H, W] latents to uint8 numpy [T_video, H, W, 3]."""
    with torch.no_grad():
        latents = latents.to(device=device, dtype=torch.bfloat16)
        # decode_to_pixel: [B, T, C, H, W] -> [B, T_video, C, H, W] in [-1, 1]
        frames = vae_wrapper.decode_to_pixel(latents)
        frames = frames[0]  # [T_video, C, H, W]
        frames = frames.permute(0, 2, 3, 1)  # [T_video, H, W, C]
        frames = ((frames.float().clamp(-1, 1) + 1) / 2 * 255).byte().cpu().numpy()
    return frames


def save_video(frames, path, fps=16):
    """Save numpy frames [T, H, W, 3] as mp4."""
    import cv2
    h, w = frames.shape[1], frames.shape[2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def main():
    import zarr as zarr_lib

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vis_dir = Path("vis")
    vis_dir.mkdir(exist_ok=True)

    # Load VAE via WanVAEWrapper
    from utils.wan_wrapper import WanVAEWrapper
    print("Loading VAE...")
    vae_wrapper = WanVAEWrapper()
    vae = vae_wrapper.to(device=device, dtype=torch.bfloat16).eval()

    # Load a sample ride
    weu_dir = "/projects/u6ex/fbots/frodobots_encoded_weu"
    zarr_files = sorted([f for f in os.listdir(weu_dir) if f.endswith(".zarr")])
    sample_zarr = os.path.join(weu_dir, zarr_files[50])  # pick a ride
    print(f"Loading latents from {zarr_files[50]}...")
    g = zarr_lib.open_group(sample_zarr, mode="r")
    # Take 21 frames (one training window)
    latents_np = g["latents"][:21]
    latents = torch.from_numpy(latents_np).unsqueeze(0).float()  # [1, 21, 16, 60, 104]
    print(f"Latent shape: {latents.shape}")

    # Set up scheduler
    scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(num_inference_steps=1000, denoising_strength=1.0)
    n_steps = len(scheduler.timesteps)

    noise_levels = [0, 10, 20, 30, 40, 50]
    noise = torch.randn_like(latents)

    for level in noise_levels:
        print(f"\nNoise level {level}:")
        if level == 0:
            noisy = latents
            actual_timestep = 0.0
        else:
            # Sample from LOW-noise end (fixed version)
            idx = n_steps - level  # e.g., 1000 - 200 = index 800
            timestep = scheduler.timesteps[idx]
            actual_timestep = timestep.item()
            t = torch.full((latents.shape[0] * latents.shape[1],), timestep, device="cpu")
            noisy = scheduler.add_noise(
                latents.flatten(0, 1), noise.flatten(0, 1), t,
            ).view_as(latents)

        print(f"  Timestep value: {actual_timestep:.1f}")
        print(f"  Decoding...")
        frames = decode_latents_to_frames(vae, noisy, device)
        out_path = vis_dir / f"noise_{level}.mp4"
        save_video(frames, out_path)
        print(f"  Saved {out_path} ({frames.shape[0]} frames)")

    print(f"\nDone! Videos saved to {vis_dir}/")


if __name__ == "__main__":
    main()
