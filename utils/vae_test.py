#!/usr/bin/env python3
"""
Test script to encode and decode a video using Wan VAE
Finds the first video in output_rides_20, saves it as MP4,
encodes it using Wan VAE, decodes it, and saves the decoded video.
"""

import os
import sys
import torch
import numpy as np
import cv2
from pathlib import Path
import tempfile
import subprocess
import logging
import shutil

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.wan_wrapper import WanVAEWrapper


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_longlive_vae(device='cuda', dtype=torch.float16, longlive_path=None):
    """Load the Wan2.1 VAE model using LongLive's WanVAEWrapper"""
    print("Loading Wan2.1 VAE model using LongLive's WanVAEWrapper...")
        
    # Create the VAE wrapper exactly as LongLive does
    vae_wrapper = WanVAEWrapper()
    
    # Move to device and convert to specified dtype
    vae_wrapper = vae_wrapper.to(device, dtype=dtype)
    vae_wrapper.eval()
    
    print(f"Wan2.1 VAE model loaded successfully using LongLive's WanVAEWrapper")
    return vae_wrapper



def find_first_video(base_path):
    """Find the first video file in output_rides_20"""
    base_path = Path(base_path)
    
    # Try output_rides_20
    train_path = base_path / "output_rides_20"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Could not find output_rides_20 directory at {base_path}")
    
    logger.info(f"Searching for videos in: {train_path}")
    
    # Find first ride directory
    ride_dirs = sorted(train_path.glob("ride_*"))
    if not ride_dirs:
        raise FileNotFoundError(f"No ride directories found in {train_path}")
    
    ride_dir = ride_dirs[0]
    logger.info(f"Found first ride: {ride_dir.name}")
    
    # Find video file in recordings directory
    recordings_dir = ride_dir / "recordings"
    if not recordings_dir.exists():
        raise FileNotFoundError(f"Recordings directory not found in {ride_dir}")
    
    # Look for video files (prefer uid_s_1000, then uid_s_1001)
    for pattern in ["*uid_s_1000*video*.m3u8", "*uid_s_1001*video*.m3u8"]:
        matches = list(recordings_dir.glob(pattern))
        if matches:
            video_path = matches[0]
            logger.info(f"Found video: {video_path}")
            return video_path
    
    raise FileNotFoundError(f"No video files found in {recordings_dir}")


def convert_to_mp4(video_path, output_mp4, duration_seconds=60.0, size=(832, 480)):
    """Convert HLS video to MP4"""
    logger.info(f"Converting {video_path} to {output_mp4}")
    
    cmd = [
        'ffmpeg', '-y', '-i', str(video_path),
        '-t', str(duration_seconds),
        '-vf', f'scale={size[0]}:{size[1]}',
        '-c:v', 'libx264', '-preset', 'medium',
        '-pix_fmt', 'yuv420p',
        str(output_mp4)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"FFmpeg error: {result.stderr}")
        raise RuntimeError(f"Failed to convert video: {result.stderr}")
    
    logger.info(f"Saved MP4 to: {output_mp4}")


def load_video_frames(video_path, max_frames=6000):
    """Load video frames and return as tensor [T, C, H, W]"""
    logger.info(f"Loading frames from {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")
    
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
    
    if not frames:
        raise ValueError("No frames extracted from video")
    
    # Stack frames: [T, H, W, C] -> [T, C, H, W]
    video_tensor = torch.stack(frames).permute(0, 3, 1, 2)
    
    logger.info(f"Loaded video with shape: {video_tensor.shape}")
    return video_tensor


def encode_video(vae, video_tensor, device='cuda', dtype=torch.float16):
    """Encode video frames to latents using Wan VAE"""
    logger.info("Encoding video frames to latents...")
    
    # Convert from [0,1] to [-1,1] range for VAE
    video_tensor = (video_tensor * 2.0) - 1.0
    
    # Move frames to device
    video_tensor = video_tensor.to(device=device, dtype=dtype)
    
    # LongLive VAE expects [batch_size, num_channels, num_frames, height, width]
    # Current shape: [T, C, H, W] -> [1, C, T, H, W]
    video_tensor = video_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, C, T, H, W]
    
    logger.info(f"Input video shape for VAE: {video_tensor.shape}")
    
    with torch.no_grad():
        # Use LongLive's encode_to_latent method
        latents = vae.encode_to_latent(video_tensor)
    
    logger.info(f"Encoded latents shape: {latents.shape}")
    return latents


def decode_latents(vae, latents, device='cuda', dtype=torch.float16):
    """Decode latents back to video frames using Wan VAE"""
    logger.info("Decoding latents to video frames...")
    
    logger.info(f"Input latents shape: {latents.shape}")
    
    # Convert latents to correct device and dtype to match VAE model
    latents = latents.to(device=device, dtype=dtype)
    
    with torch.no_grad():
        # Use LongLive's decode_to_pixel method
        decoded = vae.decode_to_pixel(latents)
    
    logger.info(f"Decoded video shape: {decoded.shape}")
    
    # decode_to_pixel returns [batch_size, num_frames, num_channels, height, width]
    # which is [1, T, C, H, W], convert to [T, C, H, W]
    decoded = decoded.squeeze(0)  # [T, C, H, W]
    
    # Convert from [-1,1] to [0,1] range
    decoded = (decoded + 1.0) / 2.0
    
    # Clamp to valid range
    decoded = torch.clamp(decoded, 0.0, 1.0)
    
    return decoded


def save_video_from_tensor(video_tensor, output_path, fps=20):
    """Save video tensor to MP4 file using ffmpeg for better compatibility"""
    logger.info(f"Saving video to {output_path}")
    
    # Convert from [T, C, H, W] to [T, H, W, C] and to numpy uint8
    video_np = video_tensor.permute(0, 2, 3, 1).cpu().numpy()
    video_np = (video_np * 255.0).astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    video_np = video_np[..., ::-1]
    
    T, H, W, C = video_np.shape
    logger.info(f"Video shape: {T} frames, {H}x{W}, {C} channels")
    
    # Use ffmpeg for more reliable video encoding
    # Save frames to temporary directory first
    temp_dir = tempfile.mkdtemp()
    try:
        # Save individual frames
        frame_files = []
        for i, frame in enumerate(video_np):
            frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
            cv2.imwrite(frame_path, frame)
            frame_files.append(frame_path)
        
        logger.info(f"Saved {len(frame_files)} frames to temporary directory")
        
        # Use ffmpeg to create video from frames
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', os.path.join(temp_dir, 'frame_%06d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'medium',
            '-crf', '23',
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            raise RuntimeError(f"Failed to create video: {result.stderr}")
        
        logger.info(f"Saved video to: {output_path}")
        
    finally:
        # Clean up temporary frames
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory: {e}")


def main():
    # Setup paths
    video_path = Path("/home/u5dk/as1748.u5dk/ARRWM/videos/original_video.mp4")
    decoded_mp4 = video_path.parent / "decoded_video.mp4"
    
    # Load video frames
    logger.info(f"Loading video from: {video_path}")
    video_tensor = load_video_frames(video_path, max_frames=6000)
    
    # Setup device and VAE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    vae = load_longlive_vae(device=device, dtype=torch.float16)
    
    # Encode video
    latents = encode_video(vae, video_tensor, device=device, dtype=torch.float16)
    
    # Decode video
    decoded_tensor = decode_latents(vae, latents, device=device, dtype=torch.float16)
    
    # Save decoded video
    save_video_from_tensor(decoded_tensor, decoded_mp4, fps=20)
    
    logger.info("=" * 60)
    logger.info("Test complete!")
    logger.info(f"Decoded video saved to: {decoded_mp4}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

