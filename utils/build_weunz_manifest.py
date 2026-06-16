"""Build (and cache) the weunz ride manifest for v14b.

weunz = /projects/u6ex/fbots/frodobots_encoded_weunz (2639 rides) — the larger
in-domain set (vs the 858-ride weu used by v14). The cached manifest is reused
by BOTH the window scorer and v14b training (the trainer loads this exact
cache_path on first run).
"""
import sys, os
sys.path.insert(0, '/lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM')
os.chdir('/lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM')
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
import torch
from utils.zarr_dataset import build_ride_manifest

ENC = "/projects/u6ex/fbots/frodobots_encoded_weunz"
CAP = "/projects/u6ex/fbots/frodobots_captions/train"
MOT = "/projects/u6ex/fbots/frodobots_motion"
LOGDIR = "/scratch/u6ex/as1748.u6ex/ARRWM/logs/v14b_weunz"
CACHE = os.path.join(LOGDIR, ".ride_manifest.pt")
os.makedirs(LOGDIR, exist_ok=True)

rides = build_ride_manifest(
    encoded_root=ENC, caption_root=CAP,
    min_ride_frames=24,                 # streaming_chunk_size(21)+context_frames(3)
    cache_path=CACHE, motion_root=MOT,
)
print(f"weunz manifest: {len(rides)} rides -> {CACHE}", flush=True)
