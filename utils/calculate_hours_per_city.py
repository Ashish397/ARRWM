#!/usr/bin/env python3
"""
Script to calculate total video hours per city from frodobots dataset.

Scans through all videos in /projects/u5dk/as1748/frodobots_data and
/projects/u5dk/as1748/frodobots_captions, extracts city information from GPS data,
and calculates total video duration per city.

Usage:
    python utils/calculate_hours_per_city.py
    python utils/calculate_hours_per_city.py --data-dir /projects/u5dk/as1748/frodobots_data
"""

import argparse
import json
import logging
import subprocess
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from geopy.distance import geodesic

# Suppress geopy warnings about latitude normalization for invalid coordinates
warnings.filterwarnings('ignore', category=UserWarning, module='geopy')

# Add parent directory to path to import from utils
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# City database from ride_level_caption.py
DATASET_CITIES = {
    'Adelaide': {'lat': -34.9285, 'lon': 138.6007, 'timezone': 'Australia/Adelaide'},
    'Auckland': {'lat': -36.8485, 'lon': 174.7633, 'timezone': 'Pacific/Auckland'},
    'Blekinge': {'lat': 56.1612, 'lon': 15.5869, 'timezone': 'Europe/Stockholm'},
    'Brisbane': {'lat': -27.4705, 'lon': 153.0260, 'timezone': 'Australia/Brisbane'},
    'Brighton': {'lat': 50.8225, 'lon': -0.1372, 'timezone': 'Europe/London'},
    'Fort Pierce': {'lat': 27.4467, 'lon': -80.3256, 'timezone': 'America/New_York'},
    'Hawaii': {'lat': 21.3099, 'lon': -157.8581, 'timezone': 'Pacific/Honolulu'},
    'Shenzhen': {'lat': 22.5431, 'lon': 114.0579, 'timezone': 'Asia/Shanghai'},
    'Vienna': {'lat': 48.2082, 'lon': 16.3738, 'timezone': 'Europe/Vienna'},
    'Lagos': {'lat': 6.5244, 'lon': 3.3792, 'timezone': 'Africa/Lagos'},
    'Madrid': {'lat': 40.4168, 'lon': -3.7038, 'timezone': 'Europe/Madrid'},
    'Manila': {'lat': 14.5995, 'lon': 120.9842, 'timezone': 'Asia/Manila'},
    'Melbourne': {'lat': -37.8136, 'lon': 144.9631, 'timezone': 'Australia/Melbourne'},
    'Muncie': {'lat': 40.1934, 'lon': -85.3863, 'timezone': 'America/Indiana/Indianapolis'},
    'Nairobi': {'lat': -1.2921, 'lon': 36.8219, 'timezone': 'Africa/Nairobi'},
    'New Hampshire': {'lat': 43.4525, 'lon': -71.5639, 'timezone': 'America/New_York'},
    'Newland': {'lat': 35.6079, 'lon': -81.9287, 'timezone': 'America/New_York'},
    'Oudewater': {'lat': 52.0257, 'lon': 4.8665, 'timezone': 'Europe/Amsterdam'},
    'Perth': {'lat': -31.9505, 'lon': 115.8605, 'timezone': 'Australia/Perth'},
    'Peterborough': {'lat': 52.5695, 'lon': -0.2405, 'timezone': 'Europe/London'},
    'Porto': {'lat': 41.1579, 'lon': -8.6291, 'timezone': 'Europe/Lisbon'},
    'Rome': {'lat': 41.9028, 'lon': 12.4964, 'timezone': 'Europe/Rome'},
    'Santiago': {'lat': -33.4489, 'lon': -70.6693, 'timezone': 'America/Santiago'},
    'Stockholm': {'lat': 59.3293, 'lon': 18.0686, 'timezone': 'Europe/Stockholm'},
    'Sydney': {'lat': -33.8688, 'lon': 151.2093, 'timezone': 'Australia/Sydney'},
    'Taipei': {'lat': 25.0330, 'lon': 121.5654, 'timezone': 'Asia/Taipei'},
    'Tokyo': {'lat': 35.6762, 'lon': 139.6503, 'timezone': 'Asia/Tokyo'},
    'Vancouver': {'lat': 49.2827, 'lon': -123.1207, 'timezone': 'America/Vancouver'},
    'Wellington': {'lat': -41.2865, 'lon': 174.7762, 'timezone': 'Pacific/Auckland'},
    'Wuhan': {'lat': 30.5928, 'lon': 114.3055, 'timezone': 'Asia/Shanghai'},
}


def get_city_from_coordinates(lat: float, lon: float) -> str:
    """Determine city from GPS coordinates. Always returns the closest city, never 'Unknown'."""
    min_distance = float('inf')
    closest_city = 'Unknown'

    for city, info in DATASET_CITIES.items():
        distance = geodesic((lat, lon), (info['lat'], info['lon'])).kilometers
        if distance < min_distance:
            min_distance = distance
            closest_city = city

    # Always return the closest city, regardless of distance
    return closest_city


def get_ride_metadata(ride_dir: Path) -> Dict[str, any]:
    """Extract city and datetime information from ride GPS data."""
    ride_id = ride_dir.name.split('_')[1] if '_' in ride_dir.name else ride_dir.name
    gps_file = ride_dir / f'gps_data_{ride_id}.csv'

    metadata = {
        'city': 'Unknown',
        'latitude': None,
        'longitude': None
    }

    if not gps_file.exists():
        logger.debug(f"GPS file not found: {gps_file}")
        return metadata

    try:
        # Read second line of CSV (skip header, read first data row)
        with open(gps_file, 'r') as f:
            lines = f.readlines()
            if len(lines) >= 2:
                # Parse second line (first data row after header)
                parts = lines[1].strip().split(',')
                lat = float(parts[0])
                lon = float(parts[1])

                # Validate coordinate ranges before using geopy
                # Valid latitude: -90 to 90, Valid longitude: -180 to 180
                if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                    logger.debug(f"Invalid GPS coordinates for {ride_dir.name}: lat={lat}, lon={lon}")
                    return metadata

                city = get_city_from_coordinates(lat, lon)

                metadata.update({
                    'city': city,
                    'latitude': lat,
                    'longitude': lon
                })
    except (ValueError, IndexError) as e:
        logger.debug(f"Could not parse GPS metadata from {gps_file}: {e}")
    except Exception as e:
        logger.debug(f"Could not read GPS metadata from {gps_file}: {e}")

    return metadata


def find_video_files(ride_dir: Path) -> List[Path]:
    """Find video files in a ride directory."""
    recordings_dir = ride_dir / 'recordings'
    if not recordings_dir.exists():
        return []

    video_files = []
    
    # Look for combined MP4 file first (most reliable)
    mp4_files = list(recordings_dir.glob("*combined_audio_video.mp4"))
    if mp4_files:
        video_files.extend(mp4_files)
    
    # Look for front camera video playlist as fallback
    m3u8_files = list(recordings_dir.glob("*uid_s_1000*video*.m3u8"))
    if m3u8_files:
        video_files.extend(m3u8_files)
    
    # If no playlists found, look for TS segments (front camera only)
    if not video_files:
        ts_files = list(recordings_dir.glob("*uid_s_1000*video*.ts"))
        if ts_files:
            # Sort by filename to get segments in order
            video_files.extend(sorted(ts_files))
    
    return video_files


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using OpenCV (works with TS, MP4, M3U8)."""
    try:
        # Try OpenCV first (works with TS segments and MP4)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.debug(f"Could not open video: {video_path}")
            return 0.0
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if fps > 0 and frame_count > 0:
            duration = frame_count / fps
            return duration
        
        # If OpenCV fails, try ffprobe as fallback (for M3U8 playlists)
        try:
            cmd = [
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                duration = float(result.stdout.strip())
                if duration > 0:
                    return duration
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass
        
        return 0.0
    except Exception as e:
        logger.debug(f"Error getting duration for {video_path}: {e}")
        return 0.0


def process_ride_directory(ride_dir: Path) -> Tuple[str, float]:
    """Process a single ride directory and return (city, duration_seconds)."""
    # Get city from GPS data
    metadata = get_ride_metadata(ride_dir)
    city = metadata['city']
    
    # Only warn if we actually tried to process GPS data but couldn't determine city
    # (i.e., GPS file exists but city is still Unknown - this shouldn't happen with valid coords)
    if city == 'Unknown' and metadata.get('latitude') is not None:
        logger.debug(f"Could not determine city for {ride_dir.name} (coordinates: {metadata.get('latitude')}, {metadata.get('longitude')})")
    
    # Find video files
    video_files = find_video_files(ride_dir)
    
    if not video_files:
        logger.debug(f"No video files found in {ride_dir.name}")
        return city, 0.0
    
    # Calculate total duration
    total_duration = 0.0
    
    # Prefer M3U8 playlist if available (contains full video)
    m3u8_files = [f for f in video_files if f.suffix == '.m3u8']
    mp4_files = [f for f in video_files if f.suffix == '.mp4']
    
    if m3u8_files:
        # Use M3U8 playlist (front camera)
        main_video = m3u8_files[0]
        duration = get_video_duration(main_video)
        total_duration = duration
    elif mp4_files:
        # Use MP4 file
        main_video = mp4_files[0]
        duration = get_video_duration(main_video)
        total_duration = duration
    else:
        # Sum durations of all TS segments (front camera only)
        ts_files = [f for f in video_files if f.suffix == '.ts' and 'uid_s_1000' in f.name]
        if ts_files:
            for video_file in sorted(ts_files):
                duration = get_video_duration(video_file)
                total_duration += duration
        else:
            # Fallback: try all video files
            for video_file in video_files:
                duration = get_video_duration(video_file)
                total_duration += duration
    
    return city, total_duration


def process_directory(data_dir: Path, caption_dir: Optional[Path] = None) -> Dict[str, float]:
    """Process all ride directories and return hours per city."""
    city_hours = defaultdict(float)
    total_rides = 0
    processed_rides = 0
    skipped_rides = 0
    
    # Process data directory
    logger.info(f"Scanning {data_dir} for ride directories...")
    for output_dir in sorted(data_dir.glob("output_rides_*")):
        if not output_dir.is_dir():
            continue
        
        logger.info(f"Processing {output_dir.name}...")
        ride_dirs = sorted(output_dir.glob("ride_*"))
        total_rides += len(ride_dirs)
        
        for ride_dir in ride_dirs:
            if not ride_dir.is_dir():
                continue
            
            try:
                city, duration_seconds = process_ride_directory(ride_dir)
                if duration_seconds > 0:
                    hours = duration_seconds / 3600.0
                    city_hours[city] += hours
                    processed_rides += 1
                else:
                    skipped_rides += 1
                    logger.debug(f"Skipped {ride_dir.name} (no valid video)")
            except Exception as e:
                logger.warning(f"Error processing {ride_dir.name}: {e}")
                skipped_rides += 1
    
    # Process caption directory if provided (might have additional metadata)
    if caption_dir and caption_dir.exists():
        logger.info(f"Scanning {caption_dir} for additional data...")
        # Caption directories might have different structure, but we'll check
        for split_dir in ['train', 'test']:
            split_path = caption_dir / split_dir
            if split_path.exists():
                # Look for ride directories in caption structure
                for ride_dir in sorted(split_path.glob("ride_*")):
                    if not ride_dir.is_dir():
                        continue
                    
                    # Check if we already processed this ride
                    # (we might have already counted it from data_dir)
                    # For now, we'll skip caption dir processing to avoid double counting
                    # But we could use it to verify city information
                    pass
    
    logger.info(f"Processed {processed_rides} rides, skipped {skipped_rides} rides out of {total_rides} total")
    
    return dict(city_hours)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate total video hours per city from frodobots dataset"
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('/projects/u5dk/as1748/frodobots_data'),
        help='Path to frodobots_data directory'
    )
    parser.add_argument(
        '--caption-dir',
        type=Path,
        default=Path('/projects/u5dk/as1748/frodobots_captions'),
        help='Path to frodobots_captions directory (optional, for verification)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output JSON file to save results (optional)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not args.data_dir.exists():
        logger.error(f"Data directory does not exist: {args.data_dir}")
        sys.exit(1)
    
    # Process all directories
    city_hours = process_directory(args.data_dir, args.caption_dir)
    
    # Print results
    print("\n" + "=" * 60)
    print("VIDEO HOURS PER CITY")
    print("=" * 60)
    
    if not city_hours:
        print("No data found!")
        return
    
    # Sort by hours (descending)
    sorted_cities = sorted(city_hours.items(), key=lambda x: x[1], reverse=True)
    
    total_hours = sum(city_hours.values())
    
    for city, hours in sorted_cities:
        percentage = (hours / total_hours * 100) if total_hours > 0 else 0
        print(f"{city:20s}: {hours:8.2f} hours ({percentage:5.2f}%)")
    
    print("=" * 60)
    print(f"{'TOTAL':20s}: {total_hours:8.2f} hours")
    print("=" * 60)
    
    # Save to JSON if requested
    if args.output:
        output_data = {
            'city_hours': city_hours,
            'total_hours': total_hours,
            'total_cities': len(city_hours)
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()

