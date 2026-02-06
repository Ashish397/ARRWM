#!/usr/bin/env python3
"""
Simple script to:
1. Load motion from motion_local.npy
2. Load encoded latents from frodobots_encoded
3. Load Motion2Action model checkpoint (xi 8244)
4. Estimate actions from latents and motion
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

from latent_actions.train_motion_2_action import Motion2ActionModel

device = 'cuda'

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

def _load_checkpoint_with_storage_fallback(checkpoint_path, map_location="cpu"):
    """Load checkpoint with fallback for different storage backends."""
    try:
        return torch.load(checkpoint_path, map_location=map_location)
    except Exception as e:
        print(f"Warning: Failed to load checkpoint with default method: {e}")
        # Try with weights_only=False for older checkpoints
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)

def main():
    # Paths
    motion_file = Path("/home/u5dk/as1748.u5dk/ARRWM/motion_outputs/motion_local.npy")
    latent_dir = Path("/projects/u5dk/as1748/frodobots_encoded/train/output_rides_0/ride_16446_20240115041752")
    checkpoint_path = Path("/scratch/u5dk/as1748.u5dk/frodobots_lam/checkpoints/checkpoint_xi_noise0.05_step8244.pth")
    output_dir = Path("/home/u5dk/as1748.u5dk/ARRWM/motion_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Loading motion and latents...")
    print("="*80)
    
    # Load motion
    print(f"\nLoading motion from {motion_file}...")
    if not motion_file.exists():
        raise FileNotFoundError(f"Motion file not found: {motion_file}")
    motion = np.load(motion_file, mmap_mode='r')
    motion = torch.from_numpy(motion).float()  # [N_windows, 100, 3]
    print(f"Motion shape: {motion.shape}")
    
    # Load encoded latents
    print(f"\nLoading encoded latents from {latent_dir}...")
    latents = load_encoded_latents(latent_dir)
    
    # Get first 45 frames (to match test_motion_from_scratch.py)
    latents = latents[:45]
    print(f"Using first 45 frames. Latents shape: {latents.shape}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = _load_checkpoint_with_storage_fallback(str(checkpoint_path), map_location="cpu")
    checkpoint_config = checkpoint.get('config', {})
    film_hidden_dim = checkpoint_config.get('film_hidden_dim', 1024)
    head_mode = checkpoint_config.get('head_mode', 'distribution')
    
    print(f"Checkpoint configuration:")
    print(f"  film_hidden_dim: {film_hidden_dim}")
    print(f"  head_mode: {head_mode}")
    
    # Initialize Motion2Action model
    print("\nInitializing Motion2Action model...")
    motion2action_model = Motion2ActionModel(
        latent_channels=16,
        film_hidden_dim=film_hidden_dim,
        motion_grid_size=10,
        film_gamma_scale=0.5,
        head_out_t=2,
        head_out_h=2,
        head_out_w=4,
        action_minmax=(-1.0, 1.0),
        head_mode=head_mode,
        log_std_bounds=(-5.0, 2.0),
        dist_eps=1e-6,
        return_dist=(head_mode == "distribution"),
    )
    
    print("Loading model state dict...")
    motion2action_model.load_state_dict(checkpoint['model_state_dict'])
    print("Moving model to device...")
    motion2action_model = motion2action_model.to(device=device, dtype=torch.float32).eval()
    for param in motion2action_model.parameters():
        param.requires_grad_(False)
    print("Motion2Action model initialized")
    
    # Prepare latents for motion2action model
    # Motion2action expects 21 frames (7 segments * 3 frames per segment)
    # Latents shape: [T, C, H, W] where T=45, C=16, H=60, W=104
    print(f"\nPreparing latents for motion2action model...")
    T_latents, C_latents, H_latents, W_latents = latents.shape
    print(f"  Original latents shape: [{T_latents}, {C_latents}, {H_latents}, {W_latents}]")
    
    # Process all 45 frames in batches of 21 frames
    # Batch 1: frames 0-20 (21 frames)
    # Batch 2: frames 21-41 (21 frames)  
    # Batch 3: frames 42-44 (3 frames) -> pad to 21
    frames_per_batch = 21
    all_estimated_actions = []
    all_log_std = []
    
    motion_normalized = motion.to(device=device, dtype=torch.float32) / 10.0
    
    num_batches = (T_latents + frames_per_batch - 1) // frames_per_batch
    print(f"  Processing {T_latents} frames in {num_batches} batches of {frames_per_batch} frames...")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * frames_per_batch
        end_idx = min(start_idx + frames_per_batch, T_latents)
        latents_batch = latents[start_idx:end_idx, :, :, :]
        
        # Pad if needed to get exactly 21 frames
        if latents_batch.shape[0] < frames_per_batch:
            frames_needed = frames_per_batch - latents_batch.shape[0]
            last_frame = latents_batch[-1:, :, :, :]
            padding = last_frame.repeat(frames_needed, 1, 1, 1)
            latents_batch = torch.cat([latents_batch, padding], dim=0)
        
        print(f"  Batch {batch_idx + 1}/{num_batches}: frames {start_idx}-{end_idx-1} (shape: {latents_batch.shape})")
        
        # Reshape to [7, 3, 16, 60, 104] then permute to [7, 16, 3, 60, 104] as expected by motion2action
        latents_for_model = latents_batch.reshape(7, 3, 16, 60, 104).permute(0, 2, 1, 3, 4)  # [7, 16, 3, 60, 104]
        latents_for_model = latents_for_model.to(device=device, dtype=torch.float32)
        
        # Prepare motion for this batch - use corresponding motion windows
        motion_start_idx = batch_idx * 7
        motion_end_idx = min(motion_start_idx + 7, motion_normalized.shape[0])
        motion_batch = motion_normalized[motion_start_idx:motion_end_idx, :, :]
        
        # Pad motion if needed to get exactly 7 windows
        if motion_batch.shape[0] < 7:
            windows_needed = 7 - motion_batch.shape[0]
            last_window = motion_batch[-1:, :, :] if motion_batch.shape[0] > 0 else motion_normalized[-1:, :, :]
            padding = last_window.repeat(windows_needed, 1, 1)
            motion_for_model = torch.cat([motion_batch, padding], dim=0)
        else:
            motion_for_model = motion_batch[:7, :, :]
        
        print(f"    Using motion windows {motion_start_idx}-{motion_end_idx-1} (shape: {motion_for_model.shape})")
        
        # Run motion2action model
        with torch.no_grad():
            model_output = motion2action_model(latents_for_model, motion_for_model)
        
        # Extract predicted actions
        if isinstance(model_output, tuple) and len(model_output) == 3:
            # return_dist=True: (dist, mean_action, log_std)
            _, estimated_actions, log_std = model_output
        elif isinstance(model_output, tuple) and len(model_output) == 2:
            # return_dist=False: (mean_action, log_std)
            estimated_actions, log_std = model_output
        else:
            estimated_actions = model_output
            log_std = None
        
        all_estimated_actions.append(estimated_actions.cpu())
        if log_std is not None:
            all_log_std.append(log_std.cpu())
    
    # Concatenate all batches
    estimated_actions = torch.cat(all_estimated_actions, dim=0)
    if all_log_std:
        log_std = torch.cat(all_log_std, dim=0)
    else:
        log_std = None
    
    print(f"\nProcessed all {T_latents} frames")
    print(f"Estimated actions shape: {estimated_actions.shape}")
    print(f"Estimated actions range: [{estimated_actions.min().item():.4f}, {estimated_actions.max().item():.4f}]")
    print(f"Estimated actions mean: {estimated_actions.mean().item():.4f}, std: {estimated_actions.std().item():.4f}")
    
    # Save estimated actions
    estimated_actions_file = output_dir / "estimated_actions.npy"
    np.save(estimated_actions_file, estimated_actions.numpy())
    print(f"\nSaved estimated actions to: {estimated_actions_file}")
    
    if log_std is not None:
        log_std_file = output_dir / "estimated_actions_log_std.npy"
        np.save(log_std_file, log_std.numpy())
        print(f"Saved estimated actions log_std to: {log_std_file}")
    
    print("="*80)
    print("Done!")

if __name__ == "__main__":
    main()
