#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform, AffineTransform

from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import argparse
import re

# Seed
random.seed(42)

#################################
# Configuration
#################################

device = 'cuda'
motion_base = Path("/projects/u5dk/as1748/frodobots_motion")
actions_base = Path("/projects/u5dk/as1748/frodobots_actions/train")
data_base = Path("/projects/u5dk/as1748/frodobots_encoded/train")

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--noise_level", type=float, default=0.02, help="Noise level used during training")
parser.add_argument("--name", type=str, required=True, help="Name of the checkpoint to load")
parser.add_argument("--head_mode", type=str, default="distribution", choices=["regression", "distribution"],
                    help="Head mode: regression or distribution (should match training)")
parser.add_argument("--find_method", type=str, default="latest", choices=["highest", "latest"],
                    help="Method to find checkpoint: 'highest' (by number in filename) or 'latest' (by modification time)")
parser.add_argument("--step", type=int, default=None,
                    help="Specific checkpoint step to load (e.g., 4000, 4333). If provided, overrides find_method")
parser.add_argument("--zero_motion", action="store_true", help="Use zero motion")
parser.add_argument("--offset_weight", type=float, default=0.0, help="Offset weight for data augmentation")
args = parser.parse_args()

# Testing
batch_size = 512
num_workers = 0
noise_level = args.noise_level
checkpoint_name = args.name
checkpoint_step = args.step
zero_motion = args.zero_motion
offset_weight = args.offset_weight
# Model config
test_head_mode = args.head_mode
find_method = args.find_method
motion_norm = True
film_hidden_dim = 256

action_minmax = (-1.0, 1.0)  # Shared range for both linear and angular actions

# Test output_rides (hardcoded to 22, 23)
test_output_rides_list = [20, 21, 22, 23]

# Directories
log_dir = Path("logs")
checkpoint_dir = Path("/scratch/u5dk/as1748.u5dk/frodobots_lam/checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(exist_ok=True)
predictions_dir = Path("/home/u5dk/as1748.u5dk/ARRWM/test_predictions")
predictions_dir.mkdir(parents=True, exist_ok=True)

#################################
# Model
#################################

class ResConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, stride=(1,1,1), groups=16):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.gn1 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(min(groups, out_ch), out_ch)

        self.skip = None
        if stride != (1,1,1) or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride),
                nn.GroupNorm(min(groups, out_ch), out_ch),
            )

    def forward(self, x):
        identity = x if self.skip is None else self.skip(x)
        x = self.act(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        return self.act(x + identity)

class Motion2ActionModel(nn.Module):
    """
    FiLM conditioning from a 10x10 CoTracker grid.

    Changes vs your version:
    - Removed motion normalization from the model (do it in the dataloader).
    - Replaced the huge flatten->LayerNorm->Linear head with a strided-conv reduction head + small MLP.
    - Removed expensive LayerNorm(feat_dim). Uses GroupNorm in conv blocks instead.
    - Avoids unnecessary clones in forward (no in-place normalization anymore).
    - Keeps your manual out_t/out_h/out_w assumption out of the head by using adaptive pooling to a fixed size.
      (If you truly never want any pooling, tell me; this is *only* at the very end to stabilize head size.)
    """
    def __init__(
        self,
        latent_channels: int,
        film_hidden_dim: int = 256,
        motion_grid_size: int = 10,
        film_gamma_scale: float = 0.5,
        head_hidden: int = 512,
        head_out_t: int = 2,
        head_out_h: int = 2,
        head_out_w: int = 4,
        action_minmax: tuple[float, float] = (-1.0, 1.0),
        head_mode: str = "distribution",
        log_std_bounds: tuple[float, float] = (-5.0, 2.0),
        dist_eps: float = 1e-6,
        return_dist: bool = True,
    ):
        super().__init__()
        self.motion_grid_size = motion_grid_size
        self.film_gamma_scale = film_gamma_scale

        self.action_low, self.action_high = action_minmax
        assert self.action_high > self.action_low

        self.head_out_t = head_out_t
        self.head_out_h = head_out_h
        self.head_out_w = head_out_w
        
        self.head_mode = head_mode
        self.log_std_bounds = log_std_bounds
        self.dist_eps = dist_eps
        self.return_dist = return_dist

        # ---- Motion encoder (2D on 10x10 grid) ----
        self.motion_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(16, 128),
            nn.SiLU(inplace=True),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(16, 128),
            nn.SiLU(inplace=True),
        )

        # Flatten 10x10 features -> FiLM params (gamma,beta) for latent channels
        motion_flat_dim = 128 * motion_grid_size * motion_grid_size
        # Shared motion encoder features
        self.motion_encoder_features = nn.Sequential(
            nn.Flatten(1),  # [B, motion_flat_dim]
            nn.Linear(motion_flat_dim, film_hidden_dim),
            nn.SiLU(inplace=True),
        )
        # Separate FiLM projections for input latents and each block
        # Input latents: latent_channels
        self.motion_proj_input = nn.Linear(film_hidden_dim, 2 * latent_channels)
        # Block1: 256 channels
        self.motion_proj_block1 = nn.Linear(film_hidden_dim, 2 * 256)
        # Block2: 512 channels
        self.motion_proj_block2 = nn.Linear(film_hidden_dim, 2 * 512)
        # Block3: 512 channels
        self.motion_proj_block3 = nn.Linear(film_hidden_dim, 2 * 512)

        # ---- Latent backbone (3D conv) ----
        self.input_proj = nn.Sequential(
            nn.Conv3d(latent_channels, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.GroupNorm(16, 128),
            nn.SiLU(inplace=True),
        )

        self.block1 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.GroupNorm(16, 256),
            nn.SiLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.GroupNorm(32, 512),
            nn.SiLU(inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.GroupNorm(32, 512),
            nn.SiLU(inplace=True),
        )

        # ---- Strided-conv head (reduces volume cheaply) ----
        # A couple more strided convs to collapse spatial/temporal footprint before the MLP.
        # If T is already 1 after block1, stride on T won't do anything harmful with padding.
        self.head_conv = nn.Sequential(
            ResConv3d(512, 256, stride=(1,2,2), groups=16),
            ResConv3d(256, 128, stride=(1,2,2), groups=8),
        )
        
        feat_dim = 128 * self.head_out_t * self.head_out_h * self.head_out_w

        # Head output size: 2 for regression (mean only), 4 for distribution (mean + log_std)
        head_output_dim = 4 if head_mode == "distribution" else 2
        self.head_mlp = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(feat_dim, head_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(head_hidden, head_output_dim),
        )

    def _scale_from_tanh(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map x in (-1,1) to [action_low, action_high] via affine transform.
        If action_minmax is (-1,1), this is identity.
        """
        low, high = self.action_low, self.action_high
        mid = (high + low) / 2.0
        half = (high - low) / 2.0
        return x * half + mid

    def _atanh_scale(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map x from [action_low, action_high] to unbounded space via atanh.
        First normalize to (-1,1), then apply atanh.
        Used for converting targets to unbounded space for training.
        """
        low, high = self.action_low, self.action_high
        mid = (high + low) / 2.0
        half = (high - low) / 2.0
        # Normalize to (-1,1)
        x_norm = (x - mid) / half
        # Clamp to avoid atanh(±1) = ±inf
        x_norm = torch.clamp(x_norm, -1.0 + 1e-7, 1.0 - 1e-7)
        # Apply atanh to get unbounded values
        return torch.atanh(x_norm)

    def forward(self, latents: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        """
        latents: [B, C, T, H, W]
        motion:  [B, (G*G), 3] where G = motion_grid_size (default 10)
        """
        B, C, T, H, W = latents.shape
        expected_pts = self.motion_grid_size * self.motion_grid_size
        if motion.shape[1] != expected_pts or motion.shape[-1] != 3:
            raise RuntimeError(
                f"Expected motion shape [B,{expected_pts},3], got {tuple(motion.shape)}"
            )

        # Motion grid: [B,3,G,G]
        motion_grid = motion.view(B, self.motion_grid_size, self.motion_grid_size, 3).permute(0, 3, 1, 2)

        # Shared motion features
        motion_feat = self.motion_encoder(motion_grid)       # [B,128,G,G]
        motion_encoded = self.motion_encoder_features(motion_feat)  # [B, film_hidden_dim]

        # FiLM params for input latents
        film_params_input = self.motion_proj_input(motion_encoded)  # [B, 2*C]
        gamma_input, beta_input = film_params_input[:, :C], film_params_input[:, C:]  # [B,C], [B,C]
        gamma_input = self.film_gamma_scale * torch.tanh(gamma_input)
        gamma_input = gamma_input.view(B, C, 1, 1, 1)
        beta_input = beta_input.view(B, C, 1, 1, 1)
        latents = latents * (1.0 + gamma_input) + beta_input

        # backbone
        x = self.input_proj(latents)
        
        # Apply FiLM to block1
        film_params_block1 = self.motion_proj_block1(motion_encoded)  # [B, 2*256]
        gamma_block1, beta_block1 = film_params_block1[:, :256], film_params_block1[:, 256:]  # [B,256], [B,256]
        gamma_block1 = self.film_gamma_scale * torch.tanh(gamma_block1)
        gamma_block1 = gamma_block1.view(B, 256, 1, 1, 1)
        beta_block1 = beta_block1.view(B, 256, 1, 1, 1)
        x = self.block1(x)
        x = x * (1.0 + gamma_block1) + beta_block1
        
        # Apply FiLM to block2
        film_params_block2 = self.motion_proj_block2(motion_encoded)  # [B, 2*512]
        gamma_block2, beta_block2 = film_params_block2[:, :512], film_params_block2[:, 512:]  # [B,512], [B,512]
        gamma_block2 = self.film_gamma_scale * torch.tanh(gamma_block2)
        gamma_block2 = gamma_block2.view(B, 512, 1, 1, 1)
        beta_block2 = beta_block2.view(B, 512, 1, 1, 1)
        x = self.block2(x)
        x = x * (1.0 + gamma_block2) + beta_block2
        
        # Apply FiLM to block3
        film_params_block3 = self.motion_proj_block3(motion_encoded)  # [B, 2*512]
        gamma_block3, beta_block3 = film_params_block3[:, :512], film_params_block3[:, 512:]  # [B,512], [B,512]
        gamma_block3 = self.film_gamma_scale * torch.tanh(gamma_block3)
        gamma_block3 = gamma_block3.view(B, 512, 1, 1, 1)
        beta_block3 = beta_block3.view(B, 512, 1, 1, 1)
        x = self.block3(x)
        x = x * (1.0 + gamma_block3) + beta_block3

        # head
        x = self.head_conv(x)
        # Optional: catch mismatches early (recommended)
        if (x.shape[2] != self.head_out_t) or (x.shape[3] != self.head_out_h) or (x.shape[4] != self.head_out_w):
            raise RuntimeError(
                f"Head feature size mismatch: expected [*,128,{self.head_out_t},{self.head_out_h},{self.head_out_w}], "
                f"got {tuple(x.shape)}"
            )

        out = self.head_mlp(x)
        
        if self.head_mode == "distribution":
            # Split into mean and log_std (2 values each)
            mean = out[:, :2]
            log_std = out[:, 2:]
            
            # Clamp log_std to bounds
            log_std = torch.clamp(log_std, self.log_std_bounds[0], self.log_std_bounds[1])
            std = torch.exp(log_std) + self.dist_eps
            
            # Create base Normal distribution in unbounded space
            base_dist = Normal(mean, std)
            
            # Squash to (-1,1) per-dim using tanh
            transforms = [TanhTransform(cache_size=1)]

            # Optionally scale from (-1,1) to [action_low, action_high]
            low, high = self.action_low, self.action_high
            if not (abs(low + 1.0) < 1e-9 and abs(high - 1.0) < 1e-9):
                mid = (high + low) / 2.0
                half = (high - low) / 2.0
                transforms.append(AffineTransform(loc=mid, scale=half))
            dist = TransformedDistribution(base_dist, transforms)
            
            # Compute mean_action explicitly: scale(tanh(mean)) - this is what was used before
            mean_action = self._scale_from_tanh(torch.tanh(mean))
            
            if self.return_dist:
                # Return tuple: (dist, mean_action, log_std) - matches original API
                return dist, mean_action, log_std
            else:
                # Return mean action and log_std
                return mean_action, log_std
        else:
            # Regression mode: Return unbounded values u ∈ R^2
            # Loss will be computed in unbounded space (after converting targets with atanh)
            # At inference, caller should apply: tanh(u) then _scale_from_tanh to get [low,high]
            return out

#################################
# Dataset
#################################

class LatentsMotionActionsIterable(IterableDataset):
    def __init__(self, actions_base, motion_base, data_base, batch_size, noise_level, output_rides_list, offset_weight=0.0):
        super().__init__()
        self.actions_base = actions_base
        self.motion_base = motion_base
        self.data_base = data_base
        self.batch_size = int(batch_size)
        self.noise_level = noise_level
        self.output_rides_list = output_rides_list
        self.zero_motion = zero_motion
        self.offset_weight = offset_weight
    def __iter__(self):
        batch_latents = None
        batch_motion = None
        batch_actions = None
        cur = 0
        def _emit():
            nonlocal batch_latents, batch_motion, batch_actions, cur
            #We want to ensure that motion is normalised roughly by dividing by 10
            batch_motion = batch_motion / 10.0
            if self.zero_motion:
                batch_motion = torch.zeros_like(batch_motion)
            out = {
                "latents": batch_latents + torch.randn_like(batch_latents) * self.noise_level,
                "motion": batch_motion + torch.randn_like(batch_motion) * self.noise_level,
                "actions": batch_actions,
            }
            batch_latents = None
            batch_motion = None
            batch_actions = None
            cur = 0
            return out
        # Only iterate through specified output_rides directories
        for output_rides_num in self.output_rides_list:
            output_rides_name = f"output_rides_{output_rides_num}"
            output_rides_dir = self.actions_base / output_rides_name
            if not output_rides_dir.is_dir():
                continue
            actions_dir = self.actions_base / output_rides_name
            motion_dir = self.motion_base / output_rides_name
            encoded_dir = self.data_base / output_rides_name
            if not motion_dir.exists() or not encoded_dir.exists():
                continue
            ride_dirs = sorted(actions_dir.glob("ride_*"))
            print(f"[Dataset] Processing {len(ride_dirs)} rides in {output_rides_name}")
            for ride_idx, ride_actions_dir in enumerate(ride_dirs):
                if not ride_actions_dir.is_dir():
                    continue
                ride_name = ride_actions_dir.name
                motion_file = motion_dir / ride_name / "motion.npy"
                if not motion_file.exists():
                    continue
                ride_encoded_dir = encoded_dir / ride_name
                latent_files = sorted(ride_encoded_dir.glob("encoded_video_*.pt"))
                if not latent_files:
                    continue
                action_files = sorted(ride_actions_dir.glob("input_actions_*.csv"))
                if not action_files:
                    continue
                # Load per-ride sources
                if ride_idx == 0:
                    print(f"[Dataset] Loading first ride: {ride_name} (this may take a moment...)")
                actions_arr = pd.read_csv(action_files[0]).to_numpy()[...,1:]
                if ride_idx == 0:
                    print(f"[Dataset] Loaded actions CSV: {actions_arr.shape}")
                motion_arr = np.load(motion_file, mmap_mode="r")  # [N,100,3]
                if ride_idx == 0:
                    print(f"[Dataset] Loaded motion array: {motion_arr.shape}")
                if ride_idx == 0:
                    print(f"[Dataset] Loading latent file: {latent_files[0]} (this may be slow...)")
                latents = torch.load(latent_files[0], map_location="cpu")[0]
                if ride_idx == 0:
                    print(f"[Dataset] Loaded latents: {latents.shape}")
                min_length = min(len(actions_arr), len(motion_arr))
                if len(latents) < (int(min_length*3)+2):
                    continue
                actions_arr = actions_arr[:min_length]
                motion_arr = motion_arr[:min_length]
                if self.offset_weight > 0.0:
                    latents = latents[1:int(min_length*3)+2]
                    latents = latents[1:] + (latents[1:] - latents[:-1]) * self.offset_weight
                else:
                    latents = latents[1:int(min_length*3)+1]
                # Reshape to [B, T, C, H, W] then permute to [B, C, T, H, W] for Conv3d
                latents = latents.view(latents.shape[0] // 3, 3, latents.shape[1], latents.shape[2], latents.shape[3])
                latents = latents.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] -> [B, C, T, H, W]
                # Check for NaN along all dimensions except the first (group dimension)
                # This creates a 1D boolean array indicating which groups have any NaN
                actions_arr_nan = np.isnan(actions_arr).any(axis=1)
                actions_arr = torch.from_numpy(actions_arr[~actions_arr_nan]).float()
                motion_arr = torch.from_numpy(motion_arr[~actions_arr_nan]).float()
                latents = latents[~actions_arr_nan]
                # Stream into fixed-size batches with leftovers handled
                offset = 0
                while offset < len(actions_arr):
                    space = self.batch_size - cur
                    take = min(space, len(actions_arr) - offset)
                    lat_chunk = latents[offset : offset + take]
                    mot_chunk = motion_arr[offset : offset + take]
                    act_chunk = actions_arr[offset : offset + take]
                    if batch_latents is None:
                        batch_latents = lat_chunk
                        batch_motion = mot_chunk
                        batch_actions = act_chunk
                    else:
                        batch_latents = torch.cat([batch_latents, lat_chunk], dim=0)
                        batch_motion = torch.cat([batch_motion, mot_chunk], dim=0)
                        batch_actions = torch.cat([batch_actions, act_chunk], dim=0)
                    cur += take
                    offset += take
                    if cur == self.batch_size:
                        yield _emit()
        if cur > 0:
            yield _emit()

#################################
# Test Function
#################################

def test():
    print("=" * 80)
    print("Motion2Action Test Script - Generate Predictions")
    print("=" * 80)
    
    # Use module-level variables (avoid scoping issues)
    test_noise_level = noise_level
    # Use module-level test_head_mode, but allow it to be overridden by checkpoint
    current_head_mode = test_head_mode
    
    # Find the checkpoint matching name and noise_level
    if checkpoint_step is not None:
        # If specific step is provided, look for exact match
        pattern = f"checkpoint_{checkpoint_name}_noise{test_noise_level}_step{checkpoint_step}.pth"
        checkpoint_path = checkpoint_dir / pattern
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                f"Looking for step {checkpoint_step} for {checkpoint_name}"
            )
        print(f"Found checkpoint (specific step {checkpoint_step}): {checkpoint_path.name}")
        actual_step = checkpoint_step
    else:
        # Use find_method to select checkpoint
        pattern = f"checkpoint_{checkpoint_name}_noise{test_noise_level}_*.pth"
        checkpoints = list(checkpoint_dir.glob(pattern))
        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoint found in {checkpoint_dir}. "
                f"Looking for pattern: {pattern}"
            )
        
        # Select checkpoint based on find_method
        if find_method == "highest":
            # Extract number from filename: checkpoint_{name}_noise{level}_step{number}.pth
            def extract_number(path):
                # Match the number after step
                match = re.search(r'step(\d+)\.pth$', path.name)
                if match:
                    return int(match.group(1))
                return -1
            
            checkpoint_path = max(checkpoints, key=extract_number)
            actual_step = extract_number(checkpoint_path)
            print(f"Found checkpoint (highest number): {checkpoint_path.name}")
        else:  # latest
            # Use the most recent checkpoint (by modification time)
            checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
            # Extract step from filename
            step_match = re.search(r'step(\d+)\.pth$', checkpoint_path.name)
            actual_step = int(step_match.group(1)) if step_match else None
            print(f"Found checkpoint (latest by modification time): {checkpoint_path.name}")
    
    # Load checkpoint first to get configuration
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract configuration from checkpoint (with defaults for backward compatibility)
    config = checkpoint.get('config', {})
    motion_norm = config.get('motion_norm', True)
    film_hidden_dim = config.get('film_hidden_dim', 256)
    # Use checkpoint's head_mode if available, otherwise use command-line argument
    checkpoint_head_mode = config.get('head_mode', current_head_mode)
    if checkpoint_head_mode != current_head_mode:
        print(f"Warning: Checkpoint head_mode ({checkpoint_head_mode}) differs from command-line ({current_head_mode})")
        print(f"Using checkpoint head_mode: {checkpoint_head_mode}")
        current_head_mode = checkpoint_head_mode
    
    print(f"Checkpoint configuration:")
    print(f"  name: {checkpoint_name}")
    print(f"  motion_norm: {motion_norm}")
    print(f"  film_hidden_dim: {film_hidden_dim}")
    print(f"  head_mode: {current_head_mode}")
    
    # Initialize model
    print("\nInitializing model...")
    model = Motion2ActionModel(
        latent_channels=16,
        film_hidden_dim=film_hidden_dim,
        motion_grid_size=10,
        film_gamma_scale=0.5,
        head_out_t=2,
        head_out_h=2,
        head_out_w=4,
        action_minmax=action_minmax,
        head_mode=current_head_mode,
        log_std_bounds=(-5.0, 2.0),
        dist_eps=1e-6,
        return_dist=(current_head_mode == "distribution"),
    ).to(device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}, step {checkpoint.get('step', 'unknown')}")
    
    # Create test dataset
    print("\nCreating test dataset...")
    dataset = LatentsMotionActionsIterable(
        actions_base=actions_base,
        motion_base=motion_base,
        data_base=data_base,
        batch_size=batch_size,
        noise_level=0.02,  # No noise for testing
        output_rides_list=test_output_rides_list,
        offset_weight=offset_weight,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=None,  # Dataset yields batches directly
        num_workers=num_workers,
        pin_memory=True,
    )
    
    # Collect all original (ground truth from input_actions) and predicted actions
    print(f"\nRunning inference on test set...")
    print(f"[Inference] Starting DataLoader iteration (this may take a moment to load first batch)...")
    original_actions = []
    predicted_actions = []
    log_std_values = []
    std_values = []
    samples_processed = 0
    batch_count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):            
            
            latents = batch["latents"].to(device, non_blocking=True)   # [B, C, T, H, W]
            motion = batch["motion"].to(device, non_blocking=True)     # [B, 100, 3]
            actions = batch["actions"].to(device, non_blocking=True)   # [B, 2]
            
            # Run inference
            if current_head_mode == "distribution":
                # Distribution mode: model returns (mean_action, log_std) or (dist, mean_action, log_std)
                model_output = model(latents, motion)
                if isinstance(model_output, tuple) and len(model_output) == 3:
                    # return_dist=True: (dist, mean_action, log_std)
                    _, mean_action, log_std = model_output
                    pred_bounded = mean_action
                else:
                    # return_dist=False: (mean_action, log_std)
                    mean_action, log_std = model_output
                    pred_bounded = mean_action
                pred_actions = pred_bounded.cpu().numpy()
                log_std_np = log_std.cpu().numpy()
                # Calculate std from log_std: std = exp(log_std) + dist_eps
                std_np = np.exp(log_std_np) + model.dist_eps
            else:
                # Regression mode: Model outputs unbounded values u ∈ R^2
                pred_unbounded = model(latents, motion)  # [B, 2] - unbounded values
                # Convert to bounded space: tanh then scale to [action_low, action_high]
                pred_bounded = torch.tanh(pred_unbounded)  # (-1,1)
                pred_bounded = model._scale_from_tanh(pred_bounded)  # [low,high]
                pred_actions = pred_bounded.cpu().numpy()
                # In regression mode, log_std and std are not available
                log_std_np = np.full_like(pred_actions, np.nan)
                std_np = np.full_like(pred_actions, np.nan)
            
            # Convert to numpy
            actions_np = actions.cpu().numpy()
            
            # Store results
            batch_size_actual = len(actions_np)
            for i in range(batch_size_actual):
                original_actions.append((actions_np[i, 0], actions_np[i, 1]))
                predicted_actions.append((pred_actions[i, 0], pred_actions[i, 1]))
                log_std_values.append((log_std_np[i, 0], log_std_np[i, 1]))
                std_values.append((std_np[i, 0], std_np[i, 1]))
                samples_processed += 1
            
            batch_count += 1
    
    print(f"\nProcessed {len(original_actions)} samples from {batch_count} test batches")
    
    # Save predictions to CSV
    print(f"\nSaving predictions to CSV...")
    if actual_step is not None:
        csv_filename = f"predictions_{checkpoint_name}_noise{test_noise_level}_step{actual_step}.csv"
    else:
        csv_filename = f"predictions_{checkpoint_name}_noise{test_noise_level}.csv"
    csv_path = predictions_dir / csv_filename
    
    # Create DataFrame with original and predicted actions, log_std, and std
    df = pd.DataFrame({
        'original_linear': [a[0] for a in original_actions],
        'original_angular': [a[1] for a in original_actions],
        'predicted_linear': [a[0] for a in predicted_actions],
        'predicted_angular': [a[1] for a in predicted_actions],
        'log_std_linear': [a[0] for a in log_std_values],
        'log_std_angular': [a[1] for a in log_std_values],
        'std_linear': [a[0] for a in std_values],
        'std_angular': [a[1] for a in std_values],
    })
    
    df.to_csv(csv_path, index=False)
    print(f"Saved predictions to: {csv_path.absolute()}")
    print(f"Total samples: {len(df)}")
    print("=" * 80)

if __name__ == "__main__":
    test()
