#!/usr/bin/env python3
"""
Pre-encode FrodoBots-2K Dataset using Wan 2.1 VAE
"""

import os
import sys
import torch
import numpy as np
import cv2
from pathlib import Path
import tempfile
import subprocess
from typing import Tuple, List
import logging
import ffmpeg
import argparse
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoLoader:
    """Handles loading videos from HLS streams and extracting frames"""
    
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
    
    def find_actual_video_path(self, json_path: str) -> str:
        """Convert JSON path to actual video files in dataset"""
        # First try the JSON path as-is (might be direct MP4)
        if os.path.exists(json_path):
            return json_path
            
        # Extract ride info from JSON path
        path_parts = json_path.split('/')
        ride_info = None
        for part in path_parts:
            if part.startswith('ride_'):
                ride_info = part
                break
        
        if not ride_info:
            raise ValueError(f"Could not extract ride info from {json_path}")
        
        # Try to find combined MP4 in recordings folder
        base_path = Path("/projects/u5as/frodobots/train")
        for output_dir in base_path.glob("output_rides_*"):
            ride_dir = output_dir / ride_info
            if ride_dir.exists():
                recordings_dir = ride_dir / "recordings"
                if recordings_dir.exists():
                    # Look for combined MP4 file
                    mp4_files = list(recordings_dir.glob("*combined_audio_video.mp4"))
                    if mp4_files:
                        return str(mp4_files[0])
                    
                    # Look for front camera video playlist as fallback
                    m3u8_files = list(recordings_dir.glob("*uid_s_1000*video*.m3u8"))
                    if m3u8_files:
                        return str(m3u8_files[0])
        
        raise FileNotFoundError(f"Could not find video files for {json_path}")
    
    def load_video_frames(self, video_path: str, duration_seconds: float = 60.0) -> torch.Tensor:
        """Load first N seconds of video and return as tensor [T, C, H, W]"""
        try:
            # Convert HLS to temporary MP4 using ffmpeg
            temp_mp4 = os.path.join(self.temp_dir, f"temp_{os.getpid()}.mp4")
            
            cmd = [
                'ffmpeg', '-y', '-i', video_path, 
                '-t', str(duration_seconds),
                '-vf', 'scale=832:480',
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
            max_frames = 6000  # Limit frames to 200 for consistency
            
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
            
            logger.info(f"Loaded video with shape: {video_tensor.shape}")
            return video_tensor
            
        except Exception as e:
            logger.error(f"Error loading video {video_path}: {e}")
            raise

    def find_video_playlists(self, ride_dir: Path) -> List[Path]:
        """Find video playlists in a ride directory"""
        recordings_dir = ride_dir / "recordings"
        if not recordings_dir.exists():
            return []
        
        # Look for front camera video playlist
        m3u8_files = list(recordings_dir.glob("*uid_s_1000*video*.m3u8"))
        return m3u8_files


def load_longlive_vae(device='cuda', dtype=torch.float16):
    """Load the Wan2.1 VAE model using LongLive's WanVAEWrapper"""
    print("Loading Wan2.1 VAE model using LongLive's WanVAEWrapper...")
    
    # Import the LongLive WanVAEWrapper
    import sys
    sys.path.append('/home/u5as/as1748.u5as/frodobots/LongLive')
    from utils.wan_wrapper import WanVAEWrapper
    
    # Create the VAE wrapper exactly as LongLive does
    vae_wrapper = WanVAEWrapper()
    
    # Move to device and convert to specified dtype
    vae_wrapper = vae_wrapper.to(device, dtype=dtype)
    vae_wrapper.eval()
    
    print(f"Wan2.1 VAE model loaded successfully using LongLive's WanVAEWrapper")
    return vae_wrapper


class VAEEncoder:
    """VAE Encoder using LongLive's WanVAEWrapper"""
    
    def __init__(self, vae_path: str, device: str = 'cuda', dtype: torch.dtype = torch.float16):
        self.device = device
        self.dtype = dtype
        self.vae = load_longlive_vae(device, dtype=dtype)
    
    def encode_video(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """Encode video frames to latents using LongLive VAE"""
        logger.info("Encoding video frames to latents...")
        
        # Convert from [0,1] to [-1,1] range for VAE
        video_tensor = (video_tensor * 2.0) - 1.0
        
        video_tensor = video_tensor.half()
        # Move frames to device
        video_tensor = video_tensor.to(device=self.device, dtype=self.dtype)
        
        # LongLive VAE expects [batch_size, num_channels, num_frames, height, width]
        # Current shape: [T, C, H, W] -> [1, C, T, H, W]
        video_tensor = video_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, C, T, H, W]
        
        logger.info(f"Input video shape for VAE: {video_tensor.shape}")
        
        with torch.no_grad():
            # Use LongLive's encode_to_latent method
            latents = self.vae.encode_to_latent(video_tensor)
        
        logger.info(f"Encoded latents shape: {latents.shape}")
        return latents





def find_all_rides(dataset_path: Path) -> List[Path]:
    """Find all ride directories in the dataset"""
    rides = []
    
    # Iterate through output_rides_* directories
    for output_dir in sorted(dataset_path.glob("output_rides_*")):
        if not output_dir.is_dir():
            continue
        
        # Find ride directories directly inside output_rides_* directories
        for ride_dir in sorted(output_dir.glob("ride_*")):
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
            output_ride_dir = output_path / ride_dir.parent.name / ride_name
            
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
                        str(playlist),
                        duration_seconds=300.0
                    )
                    
                    logger.info(f"[Rank {rank}] Loaded {ride_name} video {playlist_idx}: {video_tensor.shape}")
                    
                    encoded = encoder.encode_video(
                        video_tensor,
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
        default="/projects/u5as/frodobots/train",
        help="Path to FrodoBots dataset"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/projects/u5as/frodobots_encoded/train",
        help="Path to save encoded videos"
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default="/scratch/u5as/as1748.u5as/frodobots/wan_models/Wan2.1-T2V-1.3B",
        help="Path to VAE checkpoint or HuggingFace model ID"
    )
    
    args = parser.parse_args()
    
    # Setup SLURM environment
    rank, world_size = setup_slurm()
    
    # Process videos
    process_videos(rank, world_size, args)


if __name__ == "__main__":
    main()

