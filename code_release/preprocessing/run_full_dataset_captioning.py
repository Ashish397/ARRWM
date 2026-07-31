#!/usr/bin/env python3
"""
Full Dataset Video Captioning Runner

This script processes all videos in the FrodoBots-2K dataset with the 
balanced rover captioning system. It handles multiple output_rides directories
and manages processing across the entire dataset.
"""

import os
import subprocess
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_output_rides_dirs(base_path: str = "./FrodoBots-2K/data") -> list:
    """Find all output_rides directories"""
    output_dirs = []
    for item in os.listdir(base_path):
        if item.startswith('output_rides_') and os.path.isdir(os.path.join(base_path, item)):
            output_dirs.append(item)
    return sorted(output_dirs)

def estimate_processing_time(output_dir: str) -> int:
    """Estimate processing time for an output_rides directory"""
    base_path = f"./FrodoBots-2K/data/{output_dir}"
    total_videos = 0
    
    for ride_folder in os.listdir(base_path):
        if ride_folder.startswith('ride_'):
            recordings_path = os.path.join(base_path, ride_folder, 'recordings')
            if os.path.exists(recordings_path):
                videos = [f for f in os.listdir(recordings_path) if f.endswith('.ts')]
                total_videos += len(videos)
    
    # Estimate ~30 seconds per video (based on observed performance)
    estimated_seconds = total_videos * 30
    return estimated_seconds, total_videos

def run_captioning_for_output_dir(output_dir: str, max_videos_per_ride: int = None):
    """Run captioning for a specific output_rides directory"""
    
    estimated_time, total_videos = estimate_processing_time(output_dir)
    hours = estimated_time // 3600
    minutes = (estimated_time % 3600) // 60
    
    logger.info(f"Processing {output_dir} with {total_videos} videos")
    logger.info(f"Estimated time: {hours}h {minutes}m")
    
    # Build command
    cmd = [
        "python", "caption_video_final.py",
        "--output_rides_dir", output_dir
    ]
    
    if max_videos_per_ride:
        cmd.extend(["--max_videos_per_ride", str(max_videos_per_ride)])
    
    # Run the command
    log_file = f"processing_{output_dir}.log"
    logger.info(f"Starting processing, logging to {log_file}")
    
    start_time = time.time()
    
    with open(log_file, 'w') as f:
        process = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    
    end_time = time.time()
    actual_time = end_time - start_time
    actual_hours = int(actual_time // 3600)
    actual_minutes = int((actual_time % 3600) // 60)
    
    logger.info(f"Completed {output_dir} in {actual_hours}h {actual_minutes}m")
    
    return process.returncode == 0

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process full FrodoBots-2K dataset for video captioning')
    parser.add_argument('--output_rides_dir', help='Specific output_rides directory to process')
    parser.add_argument('--max_videos_per_ride', type=int, help='Max videos per ride (for testing)')
    parser.add_argument('--start_from', help='Start from specific output_rides directory')
    
    args = parser.parse_args()
    
    # Activate environment
    if 'frodo_env' not in os.environ.get('VIRTUAL_ENV', ''):
        logger.info("Activating frodo_env...")
        activate_script = os.path.join(os.getcwd(), 'frodo_env', 'bin', 'activate')
        if not os.path.exists(activate_script):
            logger.error("frodo_env not found!")
            return
    
    # Find output directories
    output_dirs = find_output_rides_dirs()
    logger.info(f"Found {len(output_dirs)} output_rides directories: {output_dirs}")
    
    if args.output_rides_dir:
        if args.output_rides_dir in output_dirs:
            output_dirs = [args.output_rides_dir]
        else:
            logger.error(f"Directory {args.output_rides_dir} not found!")
            return
    
    if args.start_from:
        try:
            start_idx = output_dirs.index(args.start_from)
            output_dirs = output_dirs[start_idx:]
            logger.info(f"Starting from {args.start_from}, processing {len(output_dirs)} directories")
        except ValueError:
            logger.error(f"Start directory {args.start_from} not found!")
            return
    
    # Process each directory
    successful = 0
    total_time = 0
    
    for i, output_dir in enumerate(output_dirs):
        logger.info(f"\n=== Processing directory {i+1}/{len(output_dirs)}: {output_dir} ===")
        
        dir_start_time = time.time()
        success = run_captioning_for_output_dir(output_dir, args.max_videos_per_ride)
        dir_time = time.time() - dir_start_time
        total_time += dir_time
        
        if success:
            successful += 1
            logger.info(f"✓ Successfully completed {output_dir}")
        else:
            logger.error(f"✗ Failed to process {output_dir}")
        
        # Show progress
        remaining = len(output_dirs) - (i + 1)
        if remaining > 0:
            avg_time = total_time / (i + 1)
            estimated_remaining = remaining * avg_time
            est_hours = int(estimated_remaining // 3600)
            est_minutes = int((estimated_remaining % 3600) // 60)
            logger.info(f"Progress: {i+1}/{len(output_dirs)}, estimated {est_hours}h {est_minutes}m remaining")
    
    # Final summary
    total_hours = int(total_time // 3600)
    total_minutes = int((total_time % 3600) // 60)
    
    logger.info(f"\n=== FULL DATASET PROCESSING COMPLETE ===")
    logger.info(f"Successfully processed: {successful}/{len(output_dirs)} directories")
    logger.info(f"Total time: {total_hours}h {total_minutes}m")
    logger.info(f"Results saved in captions/ directory")

if __name__ == "__main__":
    main()