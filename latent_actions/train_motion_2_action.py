#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform, AffineTransform

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
from datetime import datetime
import argparse

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
parser.add_argument("--name", type=str, required=True, help="Name for checkpoint files")
parser.add_argument("--head_mode", type=str, default="distribution", choices=["regression", "distribution"],
                    help="Head mode: regression or distribution")
parser.add_argument("--weighted_loss", action="store_true", default=False, help="Use weighted loss")
parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for distribution loss (only used in distribution mode)")
parser.add_argument("--zero_motion", action="store_true", default=False, help="Zeros motion to ablate motion conditioning")
parser.add_argument("--noise_level", type=float, default=0.02, help="Noise level for data augmentation")
args = parser.parse_args()

# Training
batch_size = args.batch_size
learning_rate = 5e-4
num_epochs = 50
num_workers = 0
noise_level = args.noise_level
checkpoint_name = args.name
head_mode = args.head_mode
early_stopping_patience = 50
zero_motion = args.zero_motion

# Parse output_rides list (convert to integers)
# output_rides_list = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]
output_rides_list = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
test_output_rides_list = [21, 22, 23] #[18]

# Model config
motion_norm = True
film_hidden_dim = 1024

action_minmax = (-1.0, 1.0)  # Shared range for both linear and angular actions

# Distribution head config (only used when head_mode="distribution")
aux_huber_weight = 0.1
std_reg_weight = 0.01
target_eps = 1e-6  # Clamp targets away from ±1 to avoid tanh boundary issues
temperature = args.temperature  # Temperature scaling for NLL loss

# Directories
log_dir = Path("logs")
checkpoint_dir = Path("/scratch/u5dk/as1748.u5dk/frodobots_lam/checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(exist_ok=True)

#################################
# Global Variables
#################################

GLOBAL_WEIGHTS = {
    "0": 0.5, # stationary
    "1": 0.5, # strong forward
    "2": 1.0, # med forward
    "3": 2.0, # strong reverse
    "4": 3.0, # med reverse
    "5": 4.0, # strong right
    "6": 4.0, # med right
    "7": 2.0, # strong left
    "8": 2.0, # med left
}

# How much of the batch imbalance signal to apply per batch (small!)
BIAS_STRENGTH = 0.10   # 0.05–0.20 is typical
# How fast GLOBAL_WEIGHTS evolves over time (very small!)
EMA_ALPHA = 0.005      # 0.001–0.01 is typical
W_MIN, W_MAX = 0.5, 5.0

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
        return_dist: bool = False,
    ):
        super().__init__()
        self.motion_grid_size = motion_grid_size
        self.film_gamma_scale = film_gamma_scale
        self.head_mode = head_mode
        self.log_std_bounds = log_std_bounds
        self.dist_eps = dist_eps
        self.return_dist = return_dist

        self.action_low, self.action_high = action_minmax
        assert self.action_high > self.action_low

        self.head_out_t = head_out_t
        self.head_out_h = head_out_h
        self.head_out_w = head_out_w

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
        head_out_size = 4 if head_mode == "distribution" else 2
        self.head_mlp = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(feat_dim, head_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(head_hidden, head_out_size),
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
                # Return mean action only
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
    def __init__(self, actions_base, motion_base, data_base, batch_size, noise_level, output_rides_list, zero_motion):
        super().__init__()
        self.actions_base = actions_base
        self.motion_base = motion_base
        self.data_base = data_base
        self.batch_size = int(batch_size)
        self.noise_level = noise_level
        self.output_rides_list = output_rides_list
        self.zero_motion = zero_motion
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
            for ride_actions_dir in sorted(actions_dir.glob("ride_*")):
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
                actions_arr = pd.read_csv(action_files[0]).to_numpy()[...,1:]
                motion_arr = np.load(motion_file, mmap_mode="r")  # [N,100,3]
                latents = torch.load(latent_files[0], map_location="cpu")[0]
                min_length = min(len(actions_arr), len(motion_arr))
                if len(latents) < int(min_length*3):
                    continue
                actions_arr = actions_arr[:min_length]
                motion_arr = motion_arr[:min_length]
                latents = latents[:int(min_length*3)]
                # Reshape to [B, T, C, H, W] then permute to [B, C, T, H, W] for Conv3d
                latents = latents.view(latents.shape[0] // 3, 3, latents.shape[1], latents.shape[2], latents.shape[3])
                latents = latents.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] -> [B, C, T, H, W]
                # Check for NaN/inf in all data sources
                # Actions: check for NaN in any column
                actions_arr_nan = np.isnan(actions_arr).any(axis=1) | np.isinf(actions_arr).any(axis=1)
                # Motion: check for NaN/inf in any of the 100x3 values
                motion_arr_nan = np.isnan(motion_arr).any(axis=(1, 2)) | np.isinf(motion_arr).any(axis=(1, 2))
                # Latents: check for NaN/inf in any dimension
                latents_nan = torch.isnan(latents).any(dim=(1, 2, 3, 4)) | torch.isinf(latents).any(dim=(1, 2, 3, 4))
                # Combine all NaN masks - exclude samples with NaN/inf in ANY source
                valid_mask = ~(actions_arr_nan | motion_arr_nan | latents_nan.cpu().numpy())
                if valid_mask.sum() == 0:
                    # Skip this ride if no valid samples remain
                    continue
                actions_arr = torch.from_numpy(actions_arr[valid_mask]).float()
                motion_arr = torch.from_numpy(motion_arr[valid_mask]).float()
                latents = latents[valid_mask]
                #Only keep 10% of these linear dominant samples
                # Check linear dominance for each of the 3 items, then reduce to per-group
                ld = (torch.abs(actions_arr[:, 0]) > torch.abs(actions_arr[:, 1])) & (actions_arr[:, 0] > 0.1)
                noop = (torch.abs(actions_arr[:, 0]) < 0.1 ) & (torch.abs(actions_arr[:, 1]) < 0.1)
                r = torch.rand(actions_arr.shape[0], device=actions_arr.device)
                keep = (~ld & ~noop) | (ld & (r < 0.1)) | (noop & (r < 0.04))
                actions_arr = actions_arr[keep]
                motion_arr  = motion_arr[keep]
                latents     = latents[keep]
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
# Training
#################################

def train():
    print("=" * 60)
    print(f"Training {head_mode} model")
    print("=" * 60)

    # Disable mixed precision - use float32 for stability
    use_amp = False
    scaler = None

    # ---- Model ----
    model = Motion2ActionModel(
        latent_channels=16,
        film_hidden_dim=film_hidden_dim,
        motion_grid_size=10,
        film_gamma_scale=0.5,
        head_out_t=2,
        head_out_h=2,
        head_out_w=4,
        action_minmax=action_minmax,
        head_mode=head_mode,
        log_std_bounds=(-5.0, 2.0),
        dist_eps=1e-6,
        return_dist=(head_mode == "distribution"),
    ).to(device).float()  # Explicitly use float32

    # ---- Dataset & Loader ----
    dataset = LatentsMotionActionsIterable(
        actions_base=actions_base,
        motion_base=motion_base,
        data_base=data_base,
        batch_size=batch_size,
        noise_level=noise_level,
        output_rides_list=output_rides_list,
        zero_motion=zero_motion,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ---- Test Dataset & Loader ----
    test_dataset = LatentsMotionActionsIterable(
        actions_base=actions_base,
        motion_base=motion_base,
        data_base=data_base,
        batch_size=batch_size,
        noise_level=0.0,
        output_rides_list=test_output_rides_list,
        zero_motion=zero_motion,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ---- Optimizer ----
    # AdamW is a strong default for inverse dynamics / behavior models
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # ---- Losses ----
    huber_loss = nn.HuberLoss(reduction="mean", delta=1.0)
    huber_loss_none = nn.HuberLoss(reduction="none", delta=1.0)
    grad_clip_norm = 1.0
    aux_huber_weight = 0.1
    std_reg_weight = 1e-4

    # ---- TensorBoard ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir / f"tensorboard_{timestamp}")

    def save_checkpoint(epoch, global_step, batch_idx):
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "step": global_step,
            "batch_idx": batch_idx,
            "config": {
                "name": checkpoint_name,
                "motion_norm": motion_norm,
                "film_hidden_dim": film_hidden_dim,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "noise_level": noise_level,
                "head_mode": head_mode,
                "aux_huber_weight": aux_huber_weight if head_mode == "distribution" else None,
                "std_reg_weight": std_reg_weight if head_mode == "distribution" else None,
                "target_eps": target_eps if head_mode == "distribution" else None,
                "temperature": temperature if head_mode == "distribution" else None,
            },
        }
        # Save with name and noise level in filename
        filename = f"checkpoint_{checkpoint_name}_noise{noise_level}_step{global_step}.pth"
        checkpoint_path = checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
    
    # Helper for safe clamp of targets for TanhTransform
    action_low, action_high = action_minmax
    # keep away from the boundaries; tanh-squash log_prob can blow up at exactly ±1
    target_eps = 1e-5 * (action_high - action_low)

    def action_binner(actions: torch.Tensor):
        """
        actions: [B,2] in [-1,1]
        returns:
        bins: [B] long in {1..9}
        weights: [B] float (from GLOBAL_WEIGHTS, with a small batch-based bias)
        """
        global GLOBAL_WEIGHTS

        B = actions.shape[0]
        device = actions.device

        bins = torch.zeros(B, device=device, dtype=torch.long)

        lin = actions[:, 0]
        ang = actions[:, 1]

        # ensure ties go somewhere so nothing is left unbinned
        linear_dom  = torch.abs(lin) > torch.abs(ang)
        angular_dom = ~linear_dom

        strong_forward = lin > 0.6
        med_forward    = (lin >= 0.1) & (lin <= 0.6)

        strong_reverse = lin < -0.6
        med_reverse    = (lin <= -0.1) & (lin >= -0.6)

        strong_right = ang > 0.6
        med_right    = (ang >= 0.1) & (ang <= 0.6)

        strong_left  = ang < -0.6
        med_left     = (ang <= -0.1) & (ang >= -0.6)

        stationary = (torch.abs(lin) < 0.1) & (torch.abs(ang) < 0.1)

        bins[stationary] = 1
        bins[strong_forward & linear_dom] = 2
        bins[med_forward    & linear_dom] = 3
        bins[strong_reverse & linear_dom] = 4
        bins[med_reverse    & linear_dom] = 5
        bins[strong_right   & angular_dom] = 6
        bins[med_right      & angular_dom] = 7
        bins[strong_left    & angular_dom] = 8
        bins[med_left       & angular_dom] = 9

        assert (bins == 0).sum().item() == 0, "Some actions did not fall into any bin."

        # bins 1..9 -> idx 0..8 (matches your GLOBAL_WEIGHTS keys)
        idx = bins - 1  # [B] in 0..8

        # ---- 1) base global prior weights (from dict) ----
        # keep this on CPU, then move to GPU once
        base_w_cpu = torch.tensor([GLOBAL_WEIGHTS[str(i)] for i in range(9)], dtype=torch.float32)  # [9]
        base_w = base_w_cpu.to(device)  # [9]

        # ---- 2) batch inverse-frequency bias (mean-normalized) ----
        counts = torch.bincount(idx, minlength=9).float()  # [9]
        # inverse freq factor; mean-normalize so the average factor is ~1
        inv = (counts.mean() / counts.clamp_min(1.0))      # [9]
        inv = inv / inv.mean().clamp_min(1e-8)             # [9]

        # blend in gently; inv^BIAS_STRENGTH is a nice smooth bias
        bias = inv.pow(BIAS_STRENGTH)                      # [9] ~1 with mild tilting

        # per-sample weights
        weights = (base_w * bias)[idx]                     # [B]
        weights = weights.clamp(W_MIN, W_MAX)

        # ---- 3) slowly update GLOBAL_WEIGHTS (EMA toward base_w * bias) ----
        # update happens in FP32 on CPU for stability
        with torch.no_grad():
            target_w_cpu = (base_w_cpu * bias.detach().cpu()).clamp(W_MIN, W_MAX)  # [9]
            updated = (1.0 - EMA_ALPHA) * base_w_cpu + EMA_ALPHA * target_w_cpu
            updated = updated.clamp(W_MIN, W_MAX)

            for i in range(9):
                GLOBAL_WEIGHTS[str(i)] = float(updated[i].item())

        return weights

    def compute_loss(latents, motion, actions, compute_metrics=False, bin_weight=False):
        """Compute loss for a batch. Returns loss and optional metrics dict."""
        # Use float32 - no autocast
        if head_mode == "distribution":
            # Distribution mode: model returns (dist, mean_action, log_std) tuple
            dist, mean_action, log_std = model(latents, motion)
            actions_safe = actions.clamp(action_low + target_eps, action_high - target_eps)
            if bin_weight:
                weights = action_binner(actions_safe)
                nll = (-dist.log_prob(actions_safe).sum(dim=-1) / temperature * weights).sum() / weights.sum()
                aux = (huber_loss_none(mean_action, actions).sum(dim=-1) * weights).sum() / weights.sum()
            else:
                nll = (-dist.log_prob(actions_safe).sum(dim=-1) / temperature).mean()
                aux = huber_loss_none(mean_action, actions).sum(dim=-1).mean()
            std_reg = torch.exp(log_std).mean()
            loss = nll + aux_huber_weight * aux + std_reg_weight * std_reg
            metrics = {}
            if compute_metrics:
                # Use mean_action for metrics (explicit bounded proxy, not dist.mode)
                pred_bounded = mean_action
                
                metrics["loss_linear"] = huber_loss(pred_bounded[:, 0], actions[:, 0]).item()
                metrics["loss_angular"] = huber_loss(pred_bounded[:, 1], actions[:, 1]).item()
                metrics["nll"] = nll.item()
                metrics["aux"] = aux.item()
                metrics["std_reg"] = std_reg.item()
                with torch.no_grad():
                    std = torch.exp(log_std)
                    metrics["std_linear_mean"] = std[:, 0].mean().item()
                    metrics["std_angular_mean"] = std[:, 1].mean().item()
        else:
            # Regression mode: Model outputs unbounded values u ∈ R^2
            pred_unbounded = model(latents, motion)  # [B,2]
            
            # Convert targets to unbounded space using atanh
            actions_unbounded = model._atanh_scale(actions)  # [B,2]
            
            # Compute loss in unbounded space (no gradient saturation!)
            loss = huber_loss_none(pred_unbounded, actions_unbounded).sum(dim=-1).mean()
            loss += 0.1 * huber_loss_none(pred_unbounded[:,0] - pred_unbounded[:,1], actions_unbounded[:,0] - actions_unbounded[:,1]).sum(dim=-1).mean()
            loss += 0.1 * huber_loss_none(pred_unbounded[:,0] + pred_unbounded[:,1], actions_unbounded[:,0] + actions_unbounded[:,1]).sum(dim=-1).mean()

            metrics = {}
            if compute_metrics:
                # For metrics, convert predictions back to bounded space for comparison
                pred_bounded = torch.tanh(pred_unbounded)  # (-1,1)
                pred_bounded = model._scale_from_tanh(pred_bounded)  # [low,high]
                
                metrics["loss_linear"] = huber_loss(pred_bounded[:, 0], actions[:, 0]).item()
                metrics["loss_angular"] = huber_loss(pred_bounded[:, 1], actions[:, 1]).item()
        return loss, metrics

    global_step = 0
    batch_idx = 0
    
    # Early stopping variables
    best_test_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        # Get a fresh iterator for this epoch (starts from beginning of files)
        data_iter = iter(dataloader)
        
        # ========== TRAINING PHASE ==========
        model.train()
        train_losses = []
        pbar = tqdm(
            desc=f"Epoch {epoch+1}/{num_epochs} [Train]",
            unit="batch",
            ncols=120,
            dynamic_ncols=True
        )

        while True:
            try:
                batch = next(data_iter)
            except StopIteration:
                break

            latents = batch["latents"].to(device, non_blocking=True).float()   # [B, C, T, H, W] - float32
            motion  = batch["motion"].to(device, non_blocking=True).float()    # [B, 100, 3] - float32
            actions = batch["actions"].to(device, non_blocking=True).float()   # [B, 2] - float32

            optimizer.zero_grad(set_to_none=True)

            loss, metrics = compute_loss(latents, motion, actions, compute_metrics=True, bin_weight=args.weighted_loss)

            # backward (no scaler - using float32)
            loss.backward()

            # grad clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            optimizer.step()

            train_losses.append(loss.item())

            # logging
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            if metrics:
                if "loss_linear" in metrics:
                    writer.add_scalar("train/loss_linear", metrics["loss_linear"], global_step)
                if "loss_angular" in metrics:
                    writer.add_scalar("train/loss_angular", metrics["loss_angular"], global_step)
                if "nll" in metrics:
                    writer.add_scalar("train/nll", metrics["nll"], global_step)
                if "aux" in metrics:
                    writer.add_scalar("train/aux", metrics["aux"], global_step)
                if "std_reg" in metrics:
                    writer.add_scalar("train/std_reg", metrics["std_reg"], global_step)
            pbar_postfix = {
                "batch": batch_idx,
                "step": global_step,
                "loss": f"{loss.item():.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
            }
            if metrics:
                if "loss_linear" in metrics:
                    pbar_postfix["lin"] = f"{metrics['loss_linear']:.4f}"
                if "loss_angular" in metrics:
                    pbar_postfix["ang"] = f"{metrics['loss_angular']:.4f}"
                if "nll" in metrics:
                    pbar_postfix["nll"] = f"{metrics['nll']:.4f}"
                if "aux" in metrics:
                    pbar_postfix["aux"] = f"{metrics['aux']:.4f}"
                if "std_reg" in metrics:
                    pbar_postfix["std_reg"] = f"{metrics['std_reg']:.4f}"
            pbar.set_postfix(pbar_postfix)
            pbar.update(1)

            global_step += 1
            batch_idx += 1

            if global_step % 1000 == 0:
                save_checkpoint(epoch, global_step, batch_idx)

        # ========== EVALUATION PHASE ==========
        model.eval()
        test_losses = []
        test_metrics_agg = {}

        with torch.no_grad():
            test_data_iter = iter(test_dataloader)
            eval_pbar = tqdm(desc=f"Epoch {epoch+1}/{num_epochs} [Eval]", leave=False)
            while True:
                try:
                    batch = next(test_data_iter)
                except StopIteration:
                    break

                latents = batch["latents"].to(device, non_blocking=True)   # [B, C, T, H, W]
                motion  = batch["motion"].to(device, non_blocking=True)    # [B, 100, 3]
                actions = batch["actions"].to(device, non_blocking=True)   # [B, 2]

                loss, metrics = compute_loss(latents, motion, actions, compute_metrics=True, bin_weight=args.weighted_loss)
                test_losses.append(loss.item())

                # Aggregate metrics for logging
                for key, value in metrics.items():
                    if key not in test_metrics_agg:
                        test_metrics_agg[key] = []
                    test_metrics_agg[key].append(value)
                eval_pbar.update(1)

        # Compute average test loss and metrics
        avg_test_loss = np.mean(test_losses) if test_losses else float('inf')
        writer.add_scalar("test/loss", avg_test_loss, epoch)

        for key, values in test_metrics_agg.items():
            if values:
                writer.add_scalar(f"test/{key}", np.mean(values), epoch)

        metrics_str = ", ".join([f"{k}: {np.mean(v):.4f}" for k, v in test_metrics_agg.items() if v])
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {np.mean(train_losses):.4f}, Test Loss: {avg_test_loss:.4f}, {metrics_str}")

        # Early stopping check
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            patience_counter = 0
            # Save best checkpoint
            save_checkpoint(epoch, global_step, batch_idx)
            print(f"New best test loss: {best_test_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}")

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs. Best test loss: {best_test_loss:.4f}")
            break

    writer.close()

if __name__ == "__main__":
    train()
