#!/usr/bin/env python3
"""
Ride-level scene-change captioning with Video Captioning Models Testbench

- Two-phase processing:
  Phase 1: Precompute and save video frame chunks (1fps, 32-second chunks)
  Phase 2: Load frame chunks and generate captions using modern vision models
- Processes each `ride_*` folder once (ignoring separate .ts/.m3u8/audio files)
- Stitches video segments logically by iterating `.ts` video chunks in order
- Uses InternVL3 model for captioning
- First chunk describes the scene, environment, weather, and spatial layout
- Subsequent chunks describe changes, events, and environmental transitions
- High-level, human-like descriptions avoiding technical details
- Avoids mentioning camera movement or ego robot movement
- Writes model-specific JSON files per ride with timestamped scene captions

Usage examples:
    python ride_level_caption.py --ride_dir ./FrodoBots-2K/data/output_rides_0/ride_17788_20240202090154 --phase precompute
    python ride_level_caption.py --ride_dir ./FrodoBots-2K/data/output_rides_0/ride_17788_20240202090154 --phase caption --model_name OpenGVLab/InternVL3-8B
    python ride_level_caption.py --output_rides_dir ./FrodoBots-2K/data/output_rides_0 --phase both --model_name OpenGVLab/InternVL3-8B
    python ride_level_caption.py --rides_folder_dir /path/to/folder/containing/rides --phase both --model_name OpenGVLab/InternVL3-8B
    python ride_level_caption.py --ride_dir ./FrodoBots-2K/data/output_rides_0/ride_17788_20240202090154 --phase both --model_name OpenGVLab/InternVL3-8B --include_rear
    python ride_level_caption.py --ride_dir ./FrodoBots-2K/data/output_rides_0/ride_17788_20240202090154 --phase both --model_name OpenGVLab/InternVL3-8B --full_video
"""

import os
import json
import argparse
import logging
import pickle
import warnings
import gc
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import cv2
import torch
import time
import pytz
from datetime import datetime, timezone
from geopy.distance import geodesic
from PIL import Image, ImageOps

# Set up logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load config and set HF_HOME BEFORE importing transformers
def _set_hf_cache_from_config():
    """Load config and set HF cache directory before any HuggingFace imports."""
    script_dir = Path(__file__).parent
    config_path = script_dir / 'config_paths.yaml'
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            if config and config.get('hf_cache'):
                os.environ['HF_HOME'] = config['hf_cache']
                logger.info(f"Set HuggingFace cache directory to: {config['hf_cache']}")
        except Exception as e:
            logger.warning(f"Could not load config for HF cache: {e}")

# Call this before transformers import
_set_hf_cache_from_config()

# Suppress specific warnings
warnings.filterwarnings('ignore', message='.*use_fast.*')
warnings.filterwarnings('ignore', message='.*image processor class does not have a fast version.*')

# Video captioning model imports (now happens AFTER HF_HOME is set)
try:
    import transformers  # type: ignore
    VIDEO_CAPTIONING_AVAILABLE = True
except ImportError:
    VIDEO_CAPTIONING_AVAILABLE = False
    logging.warning("Video captioning models not available. Please install transformers library.")

DEFAULT_CHUNK_DURATION = 32.0
DEFAULT_MAX_FRAMES = 32  # Maximum number of frame timestamps to sample from chunks


def load_config_paths(config_file: str = 'config_paths.yaml') -> Dict[str, str]:
    """
    Load path configurations from YAML file.
    
    Args:
        config_file: Path to the YAML config file (default: 'config_paths.yaml' in script directory)
    
    Returns:
        Dictionary with path configurations
    """
    # Default paths if config file doesn't exist
    default_config = {
        'dataset_root': None,
        'data_dir': None,
        'captions_dir': './captions',
        'analysis_dir': './analysis',
        'hf_cache': None,  # No default, use HuggingFace's default if not specified
    }
    
    # Try to find config file in script directory
    script_dir = Path(__file__).parent
    config_path = script_dir / config_file
    
    if not config_path.exists():
        logger.warning(f"Config file not found at {config_path}, using default paths")
        return default_config
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        
        # Ensure required keys exist, use defaults if missing
        for key in default_config:
            if key not in config:
                config[key] = default_config[key]
                if key != 'hf_cache':  # Don't warn about missing hf_cache
                    logger.warning(f"Missing '{key}' in config, using default: {default_config[key]}")
        
        return config
    except Exception as e:
        logger.error(f"Error loading config file {config_path}: {e}")
        logger.warning("Using default paths")
        return default_config

# City database from analyse_dataset.py
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

def get_local_time_from_timestamp(timestamp_ms: int, lat: float, lon: float) -> dict:
    """Convert Unix timestamp to local datetime information based on GPS coordinates.

    Returns:
        dict with 'time' (HH:MM), 'date' (YYYY-MM-DD), 'day' (Monday, Tuesday, etc.),
        'date_readable' (e.g., 'Monday, January 15, 2024')
    """
    city = get_city_from_coordinates(lat, lon)

    if city == 'Unknown' or city not in DATASET_CITIES:
        # Fallback to UTC
        dt_utc = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
        return {
            'time': dt_utc.strftime('%H:%M'),
            'date': dt_utc.strftime('%Y-%m-%d'),
            'day': dt_utc.strftime('%A'),
            'date_readable': dt_utc.strftime('%A, %B %d, %Y')
        }

    # Get timezone for the city
    tz_str = DATASET_CITIES[city]['timezone']
    tz = pytz.timezone(tz_str)

    # Convert timestamp to local time
    dt_utc = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
    dt_local = dt_utc.astimezone(tz)

    return {
        'time': dt_local.strftime('%H:%M'),
        'date': dt_local.strftime('%Y-%m-%d'),
        'day': dt_local.strftime('%A'),
        'date_readable': dt_local.strftime('%A, %B %d, %Y')
    }

def get_ride_metadata(ride_dir: Path) -> Dict[str, any]:
    """Extract city and datetime information from ride GPS data."""
    ride_id = ride_dir.name.split('_')[1]
    gps_file = ride_dir / f'gps_data_{ride_id}.csv'

    metadata = {
        'city': 'Unknown',
        'time': 'Unknown',
        'date': 'Unknown',
        'day': 'Unknown',
        'date_readable': 'Unknown',
        'latitude': None,
        'longitude': None
    }

    if not gps_file.exists():
        logger.warning(f"GPS file not found: {gps_file}")
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
                timestamp = int(parts[2])

                city = get_city_from_coordinates(lat, lon)
                datetime_info = get_local_time_from_timestamp(timestamp, lat, lon)

                metadata.update({
                    'city': city,
                    'time': datetime_info['time'],
                    'date': datetime_info['date'],
                    'day': datetime_info['day'],
                    'date_readable': datetime_info['date_readable'],
                    'latitude': lat,
                    'longitude': lon
                })
                logger.info(f"GPS metadata extracted: {city} on {datetime_info['date_readable']} at {datetime_info['time']}, coords=({lat:.6f}, {lon:.6f})")
    except Exception as e:
        logger.warning(f"Could not read GPS metadata from {gps_file}: {e}")
        import traceback
        logger.warning(traceback.format_exc())

    return metadata


def list_ride_dirs(base_path: str) -> List[Path]:
    rides: List[Path] = []
    base = Path(base_path)
    if not base.exists():
        return rides
    for item in base.iterdir():
        if item.is_dir() and item.name.startswith('output_rides_'):
            for ride in item.iterdir():
                if ride.is_dir() and ride.name.startswith('ride_'):
                    rides.append(ride)
    return sorted(rides)


def list_video_segments(ride_dir: Path, include_rear: bool = False) -> Tuple[List[Path], List[Path]]:
    """
    List video segments for both front and rear cameras.

    Args:
        ride_dir: Path to the ride directory
        include_rear: If True, include rear camera segments (default: False, front only)

    Returns:
        Tuple of (front_camera_videos, rear_camera_videos)
        - uid 1000 = front camera
        - uid 1001 = rear camera (fisheye)
    """
    recordings = ride_dir / 'recordings'
    if not recordings.exists():
        return [], []

    front_segs: List[Path] = []
    rear_segs: List[Path] = []

    for p in recordings.iterdir():
        if p.suffix.lower() == '.ts' and 'video' in p.name.lower():
            if 'uid_s_1000' in p.name:
                front_segs.append(p)
            elif 'uid_s_1001' in p.name and include_rear:
                rear_segs.append(p)

    # Sort by filename timestamp naturally
    return sorted(front_segs), sorted(rear_segs)


def get_frame_chunks_path(ride_dir: Path) -> Path:
    """Get path for storing precomputed frame chunks"""
    return ride_dir / 'frame_chunks_32s.pkl'


def preprocess_front_frame(frame: np.ndarray) -> np.ndarray:
    """
    Convert front camera frame from BGR to RGB without resizing or cropping.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def preprocess_rear_frame(frame: np.ndarray) -> np.ndarray:
    """
    Convert rear camera frame from BGR to RGB without resizing or cropping.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def extract_frames_for_30s_chunks(front_ts_files: List[Path], rear_ts_files: List[Path], chunk_duration: float = DEFAULT_CHUNK_DURATION) -> List[Tuple[List[Dict], float]]:
    """
    Extract frames from both cameras and organize into chunks.

    Args:
        front_ts_files: List of front camera video segments
        rear_ts_files: List of rear camera video segments
        chunk_duration: Duration of each chunk in seconds

    Returns:
        List of (chunk_frames, start_time) tuples where chunk_frames is a list of dicts
        Each dict contains {'front': front_frame, 'rear': rear_frame, 'timestamp': time}
    """
    all_frames = []  # List of {'front': frame, 'rear': frame, 'timestamp': time}
    cumulative_time = 0.0

    # Process both cameras together, assuming they're aligned
    max_segments = max(len(front_ts_files), len(rear_ts_files))

    for seg_idx in range(max_segments):
        front_path = front_ts_files[seg_idx] if seg_idx < len(front_ts_files) else None
        rear_path = rear_ts_files[seg_idx] if seg_idx < len(rear_ts_files) else None

        # Open both cameras
        front_cap = cv2.VideoCapture(str(front_path)) if front_path else None
        rear_cap = cv2.VideoCapture(str(rear_path)) if rear_path else None

        if front_cap and not front_cap.isOpened():
            logger.warning(f"Could not open front segment: {front_path.name}")
            front_cap = None

        if rear_cap and not rear_cap.isOpened():
            logger.warning(f"Could not open rear segment: {rear_path.name}")
            rear_cap = None

        if not front_cap and not rear_cap:
            continue

        # Get FPS and frame count from front camera (primary)
        fps = front_cap.get(cv2.CAP_PROP_FPS) if front_cap else rear_cap.get(cv2.CAP_PROP_FPS)
        fps = fps or 0.0
        frame_count = int(front_cap.get(cv2.CAP_PROP_FRAME_COUNT)) if front_cap else int(rear_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = (frame_count / fps) if fps > 0 else 0.0

        if fps <= 0 or duration <= 0:
            if front_cap:
                front_cap.release()
            if rear_cap:
                rear_cap.release()
            cumulative_time += max(duration, 1.0)
            continue

        # Sample at 1fps for efficiency
        step = max(1, int(round(fps)))
        frame_idx = 0

        while frame_idx < frame_count:
            frames_dict = {'front': None, 'rear': None, 'timestamp': 0.0}

            # Read front camera
            if front_cap:
                front_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = front_cap.read()
                if ok and frame is not None:
                    frame_rgb = preprocess_front_frame(frame)
                    frames_dict['front'] = frame_rgb

            # Read rear camera
            if rear_cap:
                rear_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = rear_cap.read()
                if ok and frame is not None:
                    frame_rgb = preprocess_rear_frame(frame)
                    frames_dict['rear'] = frame_rgb

            local_time = frame_idx / fps
            global_time = cumulative_time + local_time
            frames_dict['timestamp'] = global_time

            # Only add if we have at least one valid frame
            if frames_dict['front'] is not None or frames_dict['rear'] is not None:
                all_frames.append(frames_dict)

            frame_idx += step

        if front_cap:
            front_cap.release()
        if rear_cap:
            rear_cap.release()

        cumulative_time += duration
        logger.info(f"Extracted frames from segment {seg_idx+1}/{max_segments}")

    # Organize into chunks based on chunk_duration
    chunks = []
    current_chunk_start = 0

    for i, frame_dict in enumerate(all_frames):
        timestamp = frame_dict['timestamp']
        # Check if we need to start a new chunk
        if i == 0 or (timestamp - all_frames[current_chunk_start]['timestamp']) >= chunk_duration:
            # Save previous chunk if it exists
            if i > 0:
                chunk_frames = all_frames[current_chunk_start:i]
                chunk_start_time = all_frames[current_chunk_start]['timestamp']
                chunks.append((chunk_frames, chunk_start_time))

            # Start new chunk
            current_chunk_start = i

    # Add the last chunk
    if current_chunk_start < len(all_frames):
        chunk_frames = all_frames[current_chunk_start:]
        chunk_start_time = all_frames[current_chunk_start]['timestamp']
        chunks.append((chunk_frames, chunk_start_time))

    logger.info(f"Created {len(chunks)} chunks of ~{chunk_duration} seconds each")
    return chunks


def save_frame_chunks(chunks: List[Tuple[List[np.ndarray], float]], chunks_path: Path):
    """Save precomputed frame chunks to disk"""
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks, f)
    logger.info(f"Saved {len(chunks)} frame chunks to {chunks_path}")


def load_frame_chunks(chunks_path: Path) -> Optional[List[Tuple[List[np.ndarray], float]]]:
    """Load precomputed frame chunks from disk"""
    if not chunks_path.exists():
        return None
        
    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)
    logger.info(f"Loaded {len(chunks)} frame chunks from {chunks_path}")
    return chunks


class VideoCaptionerTestbench:
    def __init__(self, model_name: str = "OpenGVLab/InternVL3-8B"):
        if not VIDEO_CAPTIONING_AVAILABLE:
            raise ImportError("Video captioning models not available. Please install transformers library.")

        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading video captioning model: {model_name}")

        model_name_lower = model_name.lower()

        if "internvl" in model_name_lower:
            from transformers import AutoTokenizer, AutoModel, AutoImageProcessor

            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            self.image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).eval()
            self.model = self.model.to(self.device)
        else:
            raise ValueError(f"Unsupported model: {model_name}. Only InternVL3 models are supported.")

        logger.info(f"Model {model_name} loaded successfully")

    def caption_from_frames(
        self,
        frames: List[Dict],
        is_first_chunk: bool = False,
        metadata: Dict = None,
        previous_caption: Optional[str] = None,
    ) -> str:
        """
        Generate caption from video frames using the loaded model.

        Args:
            frames: List of dicts with 'front' and 'rear' frame arrays
            is_first_chunk: Whether this is the first chunk (scene description) or subsequent (change description)
            metadata: Optional dict with location and datetime keys (city, date, day, time, date_readable)
            previous_caption: Caption text generated for the previous chunk (for change detection context)

        Returns:
            Generated caption text
        """
        if not frames:
            return ""

        # Sample up to 32 frames (1 FPS across ~32 seconds)
        num_samples = min(DEFAULT_MAX_FRAMES, len(frames))
        if num_samples == 1:
            frame_indices = [0]
        else:
            # Evenly distribute samples across the chunk
            frame_indices = [int(i * (len(frames) - 1) / (num_samples - 1)) for i in range(num_samples)]
        
        selected_frame_dicts = [frames[i] for i in frame_indices]

        # Build interleaved front+rear frames (front frame 1, rear frame 1, front frame 2, rear frame 2, ...)
        interleaved_frames: List[Image.Image] = []
        front_present = False
        rear_present = False

        for frame_dict in selected_frame_dicts:
            front_arr = frame_dict.get('front')
            rear_arr = frame_dict.get('rear')

            if front_arr is not None:
                front_present = True
                front_img = Image.fromarray(front_arr)
                interleaved_frames.append(front_img)

            if rear_arr is not None:
                rear_present = True
                rear_img = Image.fromarray(rear_arr)
                interleaved_frames.append(rear_img)

        if not interleaved_frames:
            logger.warning("No camera frames available to caption.")
            return "No frames available"

        # Prepare prompts based on chunk type
        # Build location/date/time context explicitly
        location_datetime = ""
        if metadata and metadata.get('city') != 'Unknown' and metadata.get('date_readable') != 'Unknown':
            location_datetime = f"This video is recorded in {metadata['city']} on {metadata['date_readable']} at {metadata['time']}."

        sequence_note = ""
        if front_present and rear_present:
            sequence_note = ("Frames are interleaved from two synchronized cameras (front view and rear fisheye view), "
                             "sampled at 1 frame per second over roughly thirty-two seconds. "
                             "The sequence alternates: front frame, rear frame, next front frame, next rear frame, etc.")
        elif front_present:
            sequence_note = ("Frames are sampled at 1 frame per second over roughly thirty-two seconds "
                             "from the front camera view only.")
        elif rear_present:
            sequence_note = ("Frames are sampled at 1 frame per second over roughly thirty-two seconds "
                             "from the rear fisheye camera only.")

        context_parts = [segment for segment in [location_datetime.strip(), sequence_note.strip()] if segment]
        context_prefix = (" ".join(context_parts) + " ") if context_parts else ""

        prior_context = ""
        if not is_first_chunk and previous_caption:
            prior_summary = previous_caption.strip()
            if prior_summary:
                prior_context = f"Previously, the environment was summarized as: \"{prior_summary}\". "

        if is_first_chunk:
            prompt = (
                f"{context_prefix}You are analysing a 30-second scene from an ego robot's forward view. Produce a detailed, objective description sufficient to reconstruct the environment for localization and scene understanding, paint a picture with your words. "
                "Guidelines: Be literal and precise; describe only what is visible and do not speculate about causes or intentions. Do NOT mention video, frames, timestamps, stitching, or camera movement. Cover everything visible once without dwelling on minor items—group similar small elements and summarize with counts or small ranges. Use metric units; left/right/center are relative to the ego view; do not use cardinal directions. Use spatial bands—near (<=5 m), mid (5–20 m), far (>20 m)—and provide approximate dimensions when apparent (e.g., widths/heights in meters or floors). Do not assume any specific urban typology; describe what is actually present (streets, plazas, trails, parks, waterfronts, courtyards, etc.). "
                "In your free flowing description, include, when present, the overall context and atmosphere, making particular note of place type, lighting, weather, time of day, and season; the circulation and layout of movement spaces including but not limited to paths, open areas, and any vehicular or cycle facilities or traffic objects if present. Also describe the built environment and any street furniture such as facades, entrances, boundaries, overhead elements, temporary structures, walls, fences and any other items present. Make particular note of materials, colours and textures of the surfaces and terrain as this helps to recreate the scene; make note of the natural elements present in the scene such as trees, animals, plants, shrubbery, and anything else relevant. When multiple small elements are present, describe them with concise counts or ranges. Describe the humans in scene and the activities they are engaged in, with reference to their numbers, groupings, posture, movement, carried or leashed items, and anything else relevant. Make note of any clear changes across the 30 second scene without mentioning any ego camera movement. Also make note of any limits to visibility including occlusions, glare, shadows or weather. "
                "Output free flowing prose noting the details described above. Keep sentences short and declarative. Ensure comprehensive coverage while remaining concise within each section and avoid fixating on individual features for too long."
            )
        else:
            prompt = (
                f"{context_prefix}You are analysing a follow-on 30-second segment of the *same* scene. Your task is to report only objective, visible changes since the previously described segment.\n"
                f"{prior_context}\n"
                "\n"
                "Guidelines:\n"
                "• Integrate information from all available views into one unified description. Do NOT mention video, frames, timestamps, stitching, or cameras.\n"
                "• Report *deltas only*: new appearances, disappearances, state changes, changes in motion/speed/direction, counts increasing or decreasing, items entering or leaving occlusion.\n"
                "• Ignore differences explainable solely by ego/viewpoint motion, crop/zoom, exposure or white-balance shifts, compression artifacts, or minor parallax.\n"
                "• Use metric units; left/right/center are relative to the ego view; do not use cardinal directions. Use spatial bands: near (<=5 m), mid (5–20 m), far (>20 m). Provide approximate dimensions if clearly visible.\n"
                "• Be concise: do not restate the baseline; group similar small changes and summarize with counts or small ranges. Mark uncertain details as unknown/unclear/not visible; avoid speculation.\n"
                "• Focus on changes relevant to localization and scene understanding.\n"
                "\n"
                "Describe changes in the following order (2–3 compact sentences per section; skip a section if no change):\n"
                "1) Context & Atmosphere: lighting/weather/visibility that newly changes (e.g., cloud cover, shadows, glare, precipitation).\n"
                "2) Circulation & Layout: paths/trails/plazas/steps/ramps/roads—segments newly revealed or now blocked; temporary barriers or works appearing/disappearing; junctions or lines-of-movement becoming clear.\n"
                "3) Built & Natural Elements: building facades/entrances, walls/fences/railings, canopies or temporary structures; vegetation/canopy density, water or terrain features—note any elements now visible or no longer visible; newly noticeable materials/colours/textures.\n"
                "4) Wayfinding & Objects: signage/maps/notices (quote *new* legible text/symbols and script/language if obvious), lights turning on/off, furniture/art/bins/bollards/bike racks or other objects appearing/disappearing or moving.\n"
                "5) Participants & Activities: changes in counts, groupings, posture/activity, and movement directions; agents entering or exiting (people, cyclists, scooters, prams, pets, service carts, vehicles); carried or leashed items newly present or gone.\n"
                "6) Event/State Changes: signals/displays, storefront shutters/awnings, fountains/sprinklers, construction operations starting/stopping, other mechanism or access-state changes.\n"
                "7) Occlusions & Sightlines: elements newly blocked or revealed; reflections, glare, mist, or spray that newly affect visibility.\n"
                "\n"
                "If no clear changes are visible, output exactly: No clear changes.\n"
                "DO NOT repeat any information already provided in the prompt or prior summary. Output multi-section prose following the numbered order above, using short declarative clauses separated by commas or semicolons; no bullet lists or headings."
            )

        logger.info(f"Processing with model: {self.model_name}, prompt: {prompt[:80]}...")

        response = ""
        try:
            if not interleaved_frames:
                logger.warning("No images available for InternVL3 captioning")
                return "No frames available"

            # For interleaved frames, we can use more frames since they're not stacked
            # Each timestamp contributes 2 frames (front + rear), so adjust max accordingly
            max_frames = DEFAULT_MAX_FRAMES * 2  # Allow up to 16 frames (8 front + 8 rear)
            frames_for_model = interleaved_frames[:max_frames]
            logger.info(
                "InternVL3 frames: using %d of %d available (interleaved front/rear, 1 FPS)",
                len(frames_for_model),
                len(interleaved_frames)
            )

            # Use InternVL3's official image processor for dynamic resolution tiling
            # Build prompt with numbered image references
            image_refs = '\n'.join([f'Frame-{i+1}: <image>' for i in range(len(frames_for_model))])
            full_prompt = f"{image_refs}\n{prompt}"

            # Process images using the official image processor (handles dynamic tiling and aspect ratio)
            processed = self.image_processor(images=frames_for_model, return_tensors='pt')
            pixel_values = processed['pixel_values']

            # Convert to the same dtype as the model
            model_dtype = next(self.model.parameters()).dtype
            pixel_values = pixel_values.to(dtype=model_dtype, device=self.device)

            # Extract num_patches_list from processor output
            num_patches_list = processed.get('num_patches_list', None)

            generation_config = {
                'max_new_tokens': 100000,  # Increased limit for longer captions
                'do_sample': False,
            }

            with torch.no_grad():
                response = self.model.chat(
                    tokenizer=self.tokenizer,
                    pixel_values=pixel_values,
                    question=full_prompt,
                    generation_config=generation_config,
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=False
                )

            logger.info(f"InternVL3 response: {response[:200]}...")
        except Exception as e:
            import traceback
            logger.error(f"InternVL3 error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

        response = response.strip()

        return response if response else "No description generated"

    def caption_chunk(self, frames: List[np.ndarray], chunk_index: int = 0, metadata: Dict = None, previous_caption: Optional[str] = None) -> str:
        """
        Caption a chunk of video using the frames directly.

        Args:
            frames: List of video frames as numpy arrays (RGB)
            chunk_index: Index of the chunk (0 for first chunk)
            metadata: Optional dict with location and datetime keys (city, date, day, time, date_readable)
            previous_caption: Caption text generated for the immediately preceding chunk

        Returns:
            Generated caption
        """
        is_first_chunk = (chunk_index == 0)

        start_t = time.time()
        caption = self.caption_from_frames(frames, is_first_chunk, metadata, previous_caption)
        elapsed = time.time() - start_t

        logger.info(f"Chunk {chunk_index} captioned in {elapsed:.2f}s ({len(frames)} frames) using {self.model_name}")
        return caption


def ensure_output_dir(base_caption_dir: Path, ride_dir: Path) -> Path:
    # base_caption_dir / output_rides_X / ride_Y
    output_rides = ride_dir.parent.name
    out_dir = base_caption_dir / output_rides / ride_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def precompute_ride_frames(ride_dir: Path, chunk_duration: float = DEFAULT_CHUNK_DURATION, include_rear: bool = False) -> bool:
    """
    Phase 1: Precompute and save frame chunks for a ride.

    Args:
        ride_dir: Path to the ride directory
        chunk_duration: Duration of each chunk in seconds
        include_rear: If True, include rear camera processing (default: False, front only)

    Returns:
        True if successful, False otherwise
    """
    chunks_path = get_frame_chunks_path(ride_dir)

    if chunks_path.exists():
        logger.info(f"Frame chunks already exist for {ride_dir.name}: {chunks_path}")
        return True

    front_ts_files, rear_ts_files = list_video_segments(ride_dir, include_rear=include_rear)
    if not front_ts_files and not rear_ts_files:
        logger.warning(f"No video segments found in {ride_dir}")
        return False

    logger.info(f"Precomputing frame chunks for {ride_dir.name}: {len(front_ts_files)} front, {len(rear_ts_files)} rear segments")

    # Extract frames organized into chunks from both cameras
    chunks = extract_frames_for_30s_chunks(front_ts_files, rear_ts_files, chunk_duration)
    if not chunks:
        logger.warning(f"No frames extracted from {ride_dir}")
        return False

    # Save frame chunks
    save_frame_chunks(chunks, chunks_path)
    return True


def process_ride_with_frames(ride_dir: Path, captioner: VideoCaptionerTestbench, prompt: str = None, full_video: bool = False) -> Dict:
    """
    Phase 2: Process ride using precomputed frame chunks.

    Args:
        ride_dir: Path to the ride directory
        captioner: VideoCaptioner instance
        prompt: Base prompt for captioning (unused, kept for compatibility)
        full_video: If True, process all chunks; if False, process only the first chunk

    Returns:
        Dictionary containing ride analysis results
    """
    # Get ride metadata (city and time)
    metadata = get_ride_metadata(ride_dir)
    logger.info(f"Ride metadata: {metadata['city']} on {metadata['date_readable']} at {metadata['time']}")

    chunks_path = get_frame_chunks_path(ride_dir)
    frame_chunks = load_frame_chunks(chunks_path)

    if frame_chunks is None:
        logger.error(f"No frame chunks found for {ride_dir.name}. Run precompute phase first.")
        return {}

    # Determine how many chunks to process
    chunks_to_process = len(frame_chunks) if full_video else 1
    logger.info(f"Processing {ride_dir.name} with {chunks_to_process} frame chunks (full_video={full_video})")
    chunk_results: List[Dict] = []

    prev_caption: Optional[str] = None

    for idx, (frame_chunk, start_time) in enumerate(frame_chunks[:chunks_to_process]):
        logger.info(f"Processing chunk {idx+1}/{len(frame_chunks)} starting at {start_time:.2f}s")

        if not frame_chunk:
            logger.info("Empty frame chunk; skipping")
            continue

        # Use the new testbench captioning method with metadata
        caption = captioner.caption_chunk(frame_chunk, idx, metadata, previous_caption=prev_caption)
        if caption and caption != "No description generated":
            prev_caption = caption

        # Estimate end time based on chunk duration
        estimated_duration = DEFAULT_CHUNK_DURATION
        end_time = start_time + estimated_duration

        chunk_results.append({
            "chunk_index": idx,
            "timestamp_range": f"[{start_time:.0f}-{end_time:.0f}]",
            "start_seconds": float(round(start_time, 3)),
            "end_seconds": float(round(end_time, 3)),
            "chunk_caption": caption,
            "frames_processed": len(frame_chunk),
            "chunk_type": "scene_description" if idx == 0 else "change_detection"
        })

    combined_text = "\n".join([c.get("chunk_caption", "") for c in chunk_results if c.get("chunk_caption")])

    # Clear frame chunks from memory
    del frame_chunks
    gc.collect()

    # Clear GPU cache to free memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Build GPS coordinates dict
    gps_coords = None
    if metadata.get('latitude') is not None and metadata.get('longitude') is not None:
        gps_coords = {
            "latitude": metadata['latitude'],
            "longitude": metadata['longitude']
        }

    return {
        "ride_id": ride_dir.name.replace('ride_', ''),
        "analysis_type": "ride_level_video_captioner_testbench",
        "metadata": {
            "location": metadata.get('city', 'Unknown'),
            "local_date": metadata.get('date', 'Unknown'),
            "local_day": metadata.get('day', 'Unknown'),
            "local_time": metadata.get('time', 'Unknown'),
            "local_datetime_readable": metadata.get('date_readable', 'Unknown'),
            "gps": gps_coords,
            "total_chunks": len(chunk_results),
            "processing_method": "chunk_video_captioning",
            "chunk_duration_seconds": DEFAULT_CHUNK_DURATION,
            "sampling_fps": 1.0,
            "model_name": captioner.model_name,
        },
        "chunks": chunk_results,
        "combined_analysis": combined_text,
    }


def main():
    # Load config paths
    config = load_config_paths()
    
    parser = argparse.ArgumentParser(description='Ride-level captioning with video captioning models (two-phase processing)')
    parser.add_argument('--data_root', default=config.get('data_dir', './FrodoBots-2K/data'), 
                        help='Dataset root containing output_rides_* (default from config_paths.yaml)')
    parser.add_argument('--captions_dir', default=config.get('captions_dir', './captions'),
                        help='Directory to save caption outputs (default from config_paths.yaml)')
    parser.add_argument('--output_rides_dir', default=None, help='Specific output_rides_* directory to process')
    parser.add_argument('--ride_dir', default=None, help='Single specific ride directory to process')
    parser.add_argument('--rides_folder_dir', default=None, help='Directory containing multiple ride_* folders to process')
    parser.add_argument('--phase', choices=['precompute', 'caption', 'both'], default='both', 
                        help='Processing phase: precompute frame chunks, caption from chunks, or both')
    parser.add_argument('--model_name', default='OpenGVLab/InternVL3-8B',
                        help='Video captioning model name from HuggingFace (InternVL3 models only)')
    parser.add_argument('--prompt', default='Describe the scene for navigation and context.', help='Base prompt (legacy, not used in new implementation)')
    parser.add_argument('--chunk_duration', type=float, default=DEFAULT_CHUNK_DURATION, help='Duration of each chunk in seconds')
    parser.add_argument('--reverse', action='store_true', help='Process rides in reverse order (last to first)')
    parser.add_argument('--include_rear', action='store_true', help='Include rear camera (default: front camera only)')
    parser.add_argument('--full_video', action='store_true', help='Process all video chunks (default: only first chunk)')

    args = parser.parse_args()
    
    # Log the paths being used
    logger.info(f"Using data_root: {args.data_root}")
    logger.info(f"Using captions_dir: {args.captions_dir}")
    if args.include_rear:
        logger.info("Including rear camera in processing")
    else:
        logger.info("Front camera only mode (default): rear camera will be skipped")

    # Initialize captioner only if needed for caption phase
    captioner = None
    if args.phase in ['caption', 'both']:
        captioner = VideoCaptionerTestbench(args.model_name)

    # Resolve ride directories to process
    rides: List[Path] = []
    if args.ride_dir:
        ride_path = Path(args.ride_dir)
        if not ride_path.exists():
            logger.error(f"Ride directory does not exist: {ride_path}")
            return
        rides = [ride_path]
        logger.info(f"Processing single ride directory: {ride_path}")
    elif args.rides_folder_dir:
        rides_folder = Path(args.rides_folder_dir)
        if not rides_folder.exists():
            logger.error(f"Rides folder directory does not exist: {rides_folder}")
            return
        
        # Find all ride_* subdirectories
        potential_rides = [d for d in rides_folder.iterdir() if d.is_dir() and d.name.startswith('ride_')]
        
        if not potential_rides:
            logger.error(f"No ride_* directories found in {rides_folder}")
            return
        
        rides = sorted(potential_rides)
        logger.info(f"Found {len(rides)} ride directories in {rides_folder}")
    elif args.output_rides_dir:
        base = Path(args.output_rides_dir)
        if base.exists() and base.name.startswith('output_rides_'):
            for ride in base.iterdir():
                if ride.is_dir() and ride.name.startswith('ride_'):
                    rides.append(ride)
    else:
        rides = list_ride_dirs(args.data_root)

    if not rides:
        logger.error("No ride directories found to process")
        return

    # Apply reverse order if requested
    if args.reverse:
        rides = list(reversed(rides))
        logger.info(f"Processing rides in reverse order (last to first)")

    base_caption_dir = Path(args.captions_dir)

    # Process each ride sequentially: save frames → generate captions → move to next
    if args.phase in ['precompute', 'caption', 'both']:
        logger.info(f"Processing {len(rides)} rides sequentially (save frames → caption → next)")
        
        for idx, ride_dir in enumerate(rides, 1):
            logger.info(f"\n=== Processing ride {idx}/{len(rides)}: {ride_dir.name} ===")
            
            # Check if caption output already exists
            ride_id = ride_dir.name.replace('ride_', '')
            model_suffix = args.model_name.replace('/', '_').replace('-', '_') if args.phase in ['caption', 'both'] else ''
            out_dir = ensure_output_dir(base_caption_dir, ride_dir)
            out_path = out_dir / f"ride_{ride_id}_video_captions_{model_suffix}.json" if model_suffix else None
            
            if args.phase in ['caption', 'both'] and out_path and out_path.exists():
                logger.info(f"Skipping {ride_dir.name}; output already exists at {out_path}")
                continue
            
            # Step 1: Precompute frame chunks for this ride
            if args.phase in ['precompute', 'both']:
                logger.info(f"Step 1/2: Precomputing frame chunks for {ride_dir.name}")
                try:
                    success = precompute_ride_frames(ride_dir, args.chunk_duration, include_rear=args.include_rear)
                    if not success:
                        logger.error(f"Failed to precompute frame chunks for {ride_dir.name}; skipping caption phase")
                        continue
                except Exception as e:
                    logger.error(f"Error precomputing frame chunks for {ride_dir}: {e}")
                    continue
            
            # Step 2: Generate captions for this ride
            if args.phase in ['caption', 'both']:
                logger.info(f"Step 2/2: Generating captions for {ride_dir.name}")
                try:
                    result = process_ride_with_frames(ride_dir, captioner, args.prompt, args.full_video)
                    if not result:
                        logger.warning(f"No result for {ride_dir}")
                        continue
                    
                    with open(out_path, 'w') as f:
                        json.dump(result, f, indent=2)
                    logger.info(f"Saved video captions: {out_path}")
                    
                except Exception as e:
                    logger.error(f"Failed generating captions for {ride_dir}: {e}")
                    continue
            
            logger.info(f"=== Completed {ride_dir.name} ({idx}/{len(rides)}) ===")


if __name__ == '__main__':
    main()
