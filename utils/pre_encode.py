#!/usr/bin/env python3
"""
Pre-encode FrodoBots-2K Dataset using Wan 2.1 VAE
Encodes all videos in the frodobots dataset with strided encoding.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Tuple
import subprocess
import tempfile
import cv2
import numpy as np
import torch
from tqdm import tqdm
import json
import imageio_ffmpeg

# Add Self-Forcing to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from wan.modules.vae import WanVAE

# Get ffmpeg executable path
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoLoader:
    """Handles loading videos from HLS streams"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary directory: {self.temp_dir}")
    
    def __del__(self):
        """Clean up temporary directory"""
        import shutil
        try:
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory: {e}")
    
    def find_video_playlists(self, ride_dir: Path) -> List[Path]:
        """Find front camera video m3u8 playlists in a ride directory (ignoring rear videos and audio)"""
        recordings_dir = ride_dir / "recordings"
        if not recordings_dir.exists():
            return []
        
        # Find FRONT camera video playlists only (uid_s_1000 = front camera, not audio/rear)
        playlists = list(recordings_dir.glob("*uid_s_1000*video*.m3u8"))
        return playlists
    
    def load_video_frames(self, video_path: Path, target_resolution: Tuple[int, int] = (832, 464), max_frames: int = 6000) -> torch.Tensor:
        """
        Load video and return as tensor [T, C, H, W] in range [0, 1]
        
        Args:
            video_path: Path to video file (.m3u8 or .mp4)
            target_resolution: (width, height) to resize to
            max_frames: Maximum number of frames to load (default 6000 = ~5 minutes at 20fps)
        
        Returns:
            Tensor of shape [T, C, H, W] with values in [0, 1]
        """
        try:
            # Convert to temporary MP4 using ffmpeg
            temp_mp4 = os.path.join(self.temp_dir, f"temp_{os.getpid()}.mp4")
            
            width, height = target_resolution
            
            # Use ffmpeg binary from imageio-ffmpeg
            cmd = [
                FFMPEG_EXE, '-y', '-i', str(video_path),
                '-vf', f'scale={width}:{height}',
                '-c:v', 'libx264', '-preset', 'ultrafast',
                '-pix_fmt', 'yuv420p',
                temp_mp4
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                raise RuntimeError(f"Failed to convert video: {result.stderr}")
            
            # Load frames using OpenCV
            cap = cv2.VideoCapture(temp_mp4)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video file: {temp_mp4}")
            
            frames = []
            frame_count = 0
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB and normalize to [0,1]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
                frames.append(frame_tensor)
                frame_count += 1
            
            cap.release()
            
            # Clean up temp file
            if os.path.exists(temp_mp4):
                os.remove(temp_mp4)
            
            if not frames:
                raise ValueError("No frames extracted from video")
            
            # Stack frames: [T, H, W, C] -> [T, C, H, W]
            video_tensor = torch.stack(frames).permute(0, 3, 1, 2)
            
            return video_tensor
            
        except Exception as e:
            logger.error(f"Error loading video {video_path}: {e}")
            raise


class VAEEncoder:
    """Handles VAE encoding with strided windows"""
    
    def __init__(self, vae_path: str, device: str = "cuda", dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        
        # Load VAE model using WanVAE
        logger.info(f"Loading VAE from {vae_path}")
        
        # Find the actual VAE checkpoint file
        vae_pth_file = None
        if os.path.exists(vae_path) and os.path.isdir(vae_path):
            # Look for Wan2.1_VAE.pth in the directory
            potential_paths = [
                os.path.join(vae_path, "Wan2.1_VAE.pth"),
                os.path.join(vae_path, "vae.pth"),
            ]
            for p in potential_paths:
                if os.path.exists(p):
                    vae_pth_file = p
                    break
        elif os.path.exists(vae_path) and vae_path.endswith('.pth'):
            vae_pth_file = vae_path
        
        if vae_pth_file is None:
            raise FileNotFoundError(f"Could not find VAE checkpoint in {vae_path}")
        
        logger.info(f"Loading VAE checkpoint: {vae_pth_file}")
        self.vae = WanVAE(
            z_dim=16,
            vae_pth=vae_pth_file,
            dtype=dtype,
            device=device
        )
        logger.info("VAE loaded successfully")
    
    def encode_video_strided(self, video_tensor: torch.Tensor, stride: int = 2, window: int = 4) -> torch.Tensor:
        """
        Encode video with strided windows using WanVAE.
        
        Args:
            video_tensor: [T, C, H, W] tensor in range [0, 1]
            stride: Stride between windows (default 2)
            window: Window size (default 4)
        
        Returns:
            Encoded latents of shape [T_encoded, C_latent, H_latent, W_latent]
        """
        T, C, H, W = video_tensor.shape
        
        # Calculate number of encoded frames
        num_encoded = (T - window) // stride + 1
        
        if num_encoded <= 0:
            logger.warning(f"Video too short ({T} frames), skipping")
            return None
        
        # Prepare all windows and convert to the correct dtype and device
        windows = []
        for i in range(num_encoded):
            start_idx = i * stride
            end_idx = start_idx + window
            
            # Extract window: [window, C, H, W] -> [C, window, H, W]
            # Convert to the VAE's dtype (float16) AND move to CUDA device
            window_frames = video_tensor[start_idx:end_idx].permute(1, 0, 2, 3).to(device=self.device, dtype=self.dtype)
            windows.append(window_frames)
        
        # Encode all windows using WanVAE
        # WanVAE.encode expects list of [C, T, H, W] tensors
        # and returns list of encoded tensors [C_latent, T_latent, H_latent, W_latent]
        with torch.no_grad():
            encoded_list = self.vae.encode(windows)
        
        # Process encoded frames
        # Each encoded tensor may have temporal dimension, take middle or first frame
        encoded_frames = []
        for encoded in encoded_list:
            # encoded shape: [C_latent, T_latent, H_latent, W_latent]
            if encoded.shape[1] > 1:
                # Take middle frame from temporal dimension
                mid_frame = encoded.shape[1] // 2
                latent_frame = encoded[:, mid_frame, :, :]  # [C_latent, H_latent, W_latent]
            else:
                latent_frame = encoded[:, 0, :, :]  # [C_latent, H_latent, W_latent]
            
            encoded_frames.append(latent_frame)
        
        # Stack all encoded frames: [T_encoded, C_latent, H_latent, W_latent]
        encoded_video = torch.stack(encoded_frames, dim=0)
        
        return encoded_video


def find_all_rides(dataset_path: Path) -> List[Path]:
    """Find all ride directories in the dataset"""
    rides = []
    
    # Iterate through output_rides_* directories
    for output_dir in sorted(dataset_path.glob("output_rides_*")):
        if not output_dir.is_dir():
            continue
        
        # Find ride directories inside
        inner_output_dir = output_dir / output_dir.name
        if inner_output_dir.exists():
            for ride_dir in sorted(inner_output_dir.glob("ride_*")):
                if ride_dir.is_dir():
                    rides.append(ride_dir)
    
    return rides


def setup_slurm():
    """Setup SLURM environment - no distributed PyTorch needed"""
    if 'SLURM_PROCID' in os.environ:
        # Running under SLURM with multiple tasks
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        
        # With --gpu-bind=single:1, each task sees only its assigned GPU as device 0
        # So we always use cuda:0 or just 'cuda'
        logger.info(f"SLURM setup: rank={rank}, world_size={world_size}")
        logger.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        
        return rank, world_size
    else:
        # Single GPU mode
        return 0, 1


def process_videos(rank: int, world_size: int, args):
    """Main processing function for each GPU"""
    
    logger.info(f"[Rank {rank}/{world_size}] Starting processing")
    
    # Setup paths
    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create a progress tracking directory
    progress_dir = output_path / ".progress"
    progress_dir.mkdir(exist_ok=True)
    
    # Find all rides
    all_rides = find_all_rides(dataset_path)
    logger.info(f"[Rank {rank}] Found {len(all_rides)} total rides")
    
    # Distribute rides across GPUs
    rides_per_gpu = all_rides[rank::world_size]
    logger.info(f"[Rank {rank}] Processing {len(rides_per_gpu)} rides")
    
    # Initialize VAE - use 'cuda' since each task sees only its assigned GPU as device 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = VAEEncoder(args.vae_path, device=device, dtype=torch.float16)
    
    # Initialize video loader
    video_loader = VideoLoader()
    
    # Process each ride
    for ride_dir in tqdm(rides_per_gpu, desc=f"GPU {rank}", position=rank):
        try:
            ride_name = ride_dir.name
            
            # Check if already processed
            output_ride_dir = output_path / ride_dir.parent.parent.name / ride_name
            progress_file = progress_dir / f"{ride_name}.done"
            lock_dir = progress_dir / f"{ride_name}.lock"
            
            if progress_file.exists():
                logger.info(f"[Rank {rank}] Skipping {ride_name} (already processed)")
                continue
            
            # Atomic lock to prevent duplicate processing across jobs
            try:
                lock_dir.mkdir(parents=False, exist_ok=False)
            except FileExistsError:
                logger.info(f"[Rank {rank}] Skipping {ride_name} (locked by another worker)")
                continue
            
            # Find video playlists in this ride
            playlists = video_loader.find_video_playlists(ride_dir)
            
            if not playlists:
                logger.warning(f"[Rank {rank}] No video playlists found in {ride_name}")
                continue
            
            # Process each camera view (usually just one front camera)
            for playlist_idx, playlist in enumerate(playlists):
                try:
                    # Load video
                    video_tensor = video_loader.load_video_frames(
                        playlist,
                        target_resolution=(832, 464)
                    )
                    
                    logger.info(f"[Rank {rank}] Loaded {ride_name} video {playlist_idx}: {video_tensor.shape}")
                    
                    # Encode with stride
                    encoded = encoder.encode_video_strided(
                        video_tensor,
                        stride=args.stride,
                        window=args.window
                    )
                    
                    if encoded is None:
                        logger.warning(f"[Rank {rank}] Failed to encode {ride_name} video {playlist_idx}")
                        continue
                    
                    logger.info(f"[Rank {rank}] Encoded {ride_name} video {playlist_idx}: {encoded.shape}")
                    
                    # Save encoded video
                    output_ride_dir.mkdir(parents=True, exist_ok=True)
                    output_file = output_ride_dir / f"encoded_video_{playlist_idx}.pt"
                    torch.save(encoded, output_file)
                    
                    logger.info(f"[Rank {rank}] Saved encoded video to {output_file}")
                    
                except Exception as e:
                    logger.error(f"[Rank {rank}] Error processing {ride_name} video {playlist_idx}: {e}")
                    continue
            
            # Mark as complete
            progress_file.touch()
            
            # Remove lock
            try:
                lock_dir.rmdir()
            except Exception:
                pass
            
        except Exception as e:
            logger.error(f"[Rank {rank}] Error processing ride {ride_dir.name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Remove lock on error
            try:
                lock_dir.rmdir()
            except Exception:
                pass
            continue
    
    logger.info(f"[Rank {rank}] Finished processing")


def main():
    parser = argparse.ArgumentParser(description="Pre-encode FrodoBots dataset with Wan VAE")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/projects/u5as/frodobots/",
        help="Path to FrodoBots dataset"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/projects/u5as/frodobots_encoded/",
        help="Path to save encoded videos"
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default="/scratch/u5as/as1748.u5as/frodobots/wan_models/Wan2.1-T2V-1.3B",
        help="Path to VAE checkpoint or HuggingFace model ID"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Stride between encoding windows (default: 2)"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=4,
        help="Window size for encoding (default: 4)"
    )
    
    args = parser.parse_args()
    
    # Setup SLURM environment
    rank, world_size = setup_slurm()
    
    # Process videos
    process_videos(rank, world_size, args)


if __name__ == "__main__":
    main()

