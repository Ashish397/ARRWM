#!/usr/bin/env python3
"""
Simple script to:
1. Load encoded latents from frodobots_encoded
2. Get first 45 frames
3. Decode them using VAE
4. Run through cotracker offline to create motion file
"""

from pathlib import Path
import sys
import os
import numpy as np
import torch

# Disable PyTorch compilation to avoid multiprocessing issues
os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')
torch._dynamo.config.suppress_errors = True

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.wan_wrapper import WanVAEWrapper
from cotracker.predictor import CoTrackerPredictor

device = 'cuda'
grid_size = 10
output_chunk_size = 12  # frames per output window
compute_T = 48  # frames per compute chunk

def load_encoded_latents(latent_dir):
    """Load all encoded_video_*.pt files from directory and concatenate them."""
    latent_dir = Path(latent_dir)
    latent_files = sorted(latent_dir.glob("encoded_video_*.pt"))
    
    if not latent_files:
        raise ValueError(f"No encoded_video_*.pt files found in {latent_dir}")
    
    print(f"Found {len(latent_files)} latent files")
    
    latents_list = []
    for latent_file in latent_files:
        tensor = torch.load(latent_file, map_location="cpu")
        # Handle different tensor shapes
        if isinstance(tensor, (list, tuple)):
            tensor = tensor[0]
        if tensor.dim() == 5:  # [B, T, C, H, W]
            tensor = tensor[0]  # Take first batch
        elif tensor.dim() == 4:  # [T, C, H, W]
            pass
        else:
            raise ValueError(f"Unexpected latent shape: {tensor.shape}")
        latents_list.append(tensor)
    
    # Concatenate all latents along time dimension
    latents = torch.cat(latents_list, dim=0)  # [T_total, C, H, W]
    print(f"Total latents shape: {latents.shape}")
    return latents

def decode_latents_to_pixels(vae, latents, device, decode_chunk_size=128):
    """
    Decode latents to pixels.
    latents: [T, C, H, W] where T is total frames, C=16
    Returns: [1, T, 3, H, W] in [0, 255] range (float)
    """
    vae = vae.to(device).eval()
    T, C, H, W = latents.shape
    
    print(f"Decoding {T} latent frames...")
    all_pixels = []
    num_chunks = (T + decode_chunk_size - 1) // decode_chunk_size
    
    with torch.no_grad():
        for chunk_idx, decode_start in enumerate(range(0, T, decode_chunk_size)):
            if chunk_idx % 10 == 0 or chunk_idx == num_chunks - 1:
                print(f"  Decoding chunk {chunk_idx + 1}/{num_chunks}...")
            decode_end = min(decode_start + decode_chunk_size, T)
            latent_chunk = latents[decode_start:decode_end, :, :, :].unsqueeze(0).to(device).float()  # [1, chunk, C, H, W]
            
            # Decode chunk - VAE returns in [-1, 1] range
            pixels_chunk = vae.decode_to_pixel(latent_chunk)  # [1, chunk, 3, H, W] in [-1, 1]
            
            # Convert to [0, 255] range
            pixels_chunk = (255 * 0.5 * (pixels_chunk + 1.0)).clamp(0, 255).float()
            
            all_pixels.append(pixels_chunk.cpu())
            
            # Clear VAE cache
            if hasattr(vae.model, 'clear_cache'):
                vae.model.clear_cache()
    
    # Concatenate all chunks: [1, T, 3, H, W]
    pixels = torch.cat(all_pixels, dim=1).to(device)
    print(f"Decoding complete. Pixels shape: {pixels.shape}")
    return pixels

def compute_motion_from_pixels(cotracker, pixels, grid_size=10, output_chunk_size=12, compute_T=48, device='cuda'):
    """
    Compute motion from pixels using CoTracker offline.
    pixels: [1, T, 3, H, W] in [0, 255] range (float)
    Returns: [N_windows, 100, 3] motion tensor
    """
    cotracker = cotracker.to(device).eval()
    N = grid_size ** 2
    n_out_full = compute_T // output_chunk_size
    assert compute_T % output_chunk_size == 0
    
    outs_gpu = []
    T_total = pixels.shape[1]
    num_chunks = (T_total + compute_T - 1) // compute_T
    
    print(f"Processing {T_total} frames in {num_chunks} chunks of {compute_T} frames...")
    
    # Process in chunks of compute_T frames
    with torch.inference_mode():
        dropped_first = False
        for chunk_idx, chunk_start in enumerate(range(0, T_total, compute_T)):
            if chunk_idx % 5 == 0 or chunk_idx == num_chunks - 1:
                print(f"  Processing chunk {chunk_idx + 1}/{num_chunks}...")
            chunk_end = min(chunk_start + compute_T, T_total)
            frames_chunk = pixels[:, chunk_start:chunk_end, :, :, :]  # [1, T_chunk, 3, H, W]
            
            # Drop first frame overall
            if not dropped_first:
                if frames_chunk.shape[1] == 0:
                    continue
                frames_chunk = frames_chunk[:, 1:, :, :, :]
                dropped_first = True
                if frames_chunk.shape[1] == 0:
                    continue
            
            # Keep only complete 12-frame windows
            n_out = frames_chunk.shape[1] // output_chunk_size
            if n_out == 0:
                continue
            
            used_frames = n_out * output_chunk_size
            frames_chunk = frames_chunk[:, :used_frames, :, :, :].clone()
            
            # Run CoTracker
            with torch.amp.autocast(device_type='cuda', enabled=True):
                pred_tracks, pred_visibility = cotracker(frames_chunk, grid_size=grid_size)  # [B,T,N,2], [B,T,N] or [B,T,N,1]
            
            B = pred_tracks.shape[0]
            T_chunk = pred_tracks.shape[1]
            
            # Group into disjoint 12-frame windows
            tracks_w = pred_tracks.reshape(B, n_out, output_chunk_size, N, 2)  # [B,n_out,12,N,2]
            # Handle visibility shape
            if pred_visibility.dim() == 3:  # [B,T,N]
                vis_w = pred_visibility.reshape(B, n_out, output_chunk_size, N).unsqueeze(-1)  # [B,n_out,12,N,1]
            else:  # [B,T,N,1]
                vis_w = pred_visibility.reshape(B, n_out, output_chunk_size, N, 1)  # [B,n_out,12,N,1]
            
            # Compute motion: mean within-window deltas (11 deltas for 12 frames)
            d_w = tracks_w[:, :, 1:] - tracks_w[:, :, :-1]  # [B,n_out,11,N,2]
            motion_out = d_w.mean(dim=2)  # [B,n_out,N,2]
            
            # Mean visibility over 12 frames
            vis_out = vis_w.to(dtype=motion_out.dtype).mean(dim=2)  # [B,n_out,N,1]
            
            out = torch.cat([motion_out, vis_out], dim=-1).squeeze(0)  # [n_out,N,3]
            outs_gpu.append(out)
    
    if not outs_gpu:
        print("ERROR: No motion windows computed!")
        return None
    
    motion_gpu = torch.cat(outs_gpu, dim=0)  # [total_out, N, 3]
    print(f"Motion computation complete. Final motion shape: {motion_gpu.shape}")
    return motion_gpu.cpu()

def main():
    # Path from terminal selection
    latent_dir = Path("/projects/u5dk/as1748/frodobots_encoded/train/output_rides_0/ride_16446_20240115041752")
    output_dir = Path("/home/u5dk/as1748.u5dk/ARRWM/motion_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Loading encoded latents...")
    print("="*80)
    
    # Load encoded latents
    latents = load_encoded_latents(latent_dir)
    
    # Get first 45 frames
    latents = latents[:45]
    print(f"Using first 45 frames. Latents shape: {latents.shape}")
    
    # Initialize VAE
    print("\nInitializing VAE...")
    vae = WanVAEWrapper().to(device).eval()
    print("VAE initialized")
    
    # Decode latents to pixels
    print("\nDecoding latents to pixels...")
    pixels = decode_latents_to_pixels(vae, latents, device)
    
    # Initialize CoTracker
    print("\nInitializing CoTracker...")
    cotracker_checkpoint = Path("/scratch/u5dk/as1748.u5dk/torch_hub/scaled_offline.pth")
    cotracker = CoTrackerPredictor(
        checkpoint=str(cotracker_checkpoint),
        offline=True,
        window_len=60,
    ).to(device).eval()
    print("CoTracker initialized")
    
    # Compute motion from pixels
    print("\nComputing motion from pixels using CoTracker...")
    motion = compute_motion_from_pixels(
        cotracker, pixels, grid_size, output_chunk_size, compute_T, device
    )
    
    if motion is None:
        print("ERROR: Failed to compute motion")
        return
    
    # Save motion file
    motion_file = output_dir / "motion.npy"
    np.save(motion_file, motion.numpy())
    print(f"\nMotion file saved to: {motion_file}")
    print(f"Motion shape: {motion.shape}")

if __name__ == "__main__":
    main()
