"""Build (and cache) the weunz ride manifest for v14b.

weunz = ${DATA_ROOT}/frodobots_encoded_weunz (2639 rides) — the larger
in-domain set (vs the 858-ride weu used by v14). The cached manifest is reused
by BOTH the window scorer and v14b training (the trainer loads this exact
cache_path on first run).
"""
import os
import sys, os
sys.path.insert(0, '${AF_ROOT}')
os.chdir('${AF_ROOT}')
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
import torch
from utils.zarr_dataset import build_ride_manifest

ENC = os.path.join(os.environ.get("DATA_ROOT", ""), "frodobots_encoded_weunz")
CAP = os.path.join(os.environ.get("DATA_ROOT", ""), "frodobots_captions/train")
MOT = os.path.join(os.environ.get("DATA_ROOT", ""), "frodobots_motion")
LOGDIR = os.path.join(os.environ.get("AF_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs/v14b_weunz")
CACHE = os.path.join(LOGDIR, ".ride_manifest.pt")
os.makedirs(LOGDIR, exist_ok=True)

rides = build_ride_manifest(
    encoded_root=ENC, caption_root=CAP,
    min_ride_frames=24,                 # streaming_chunk_size(21)+context_frames(3)
    cache_path=CACHE, motion_root=MOT,
)
print(f"weunz manifest: {len(rides)} rides -> {CACHE}", flush=True)
