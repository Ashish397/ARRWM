#!/usr/bin/env python3
"""Decode each variant in a saved ``latents.pt`` to its own mp4.

Loads the file written by ``save_latents_three_variants.py`` (keys
``gt``, ``ar_refresh``, ``ar_refresh_once``, ``append_baseline``,
each ``[1, T, C, H, W]``) and writes one mp4 per variant.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_AF_ROOT = _REPO_ROOT / "action-forcing"
if str(_AF_ROOT) not in sys.path:
    sys.path.insert(0, str(_AF_ROOT))

from utils.eval_causal_AR_chain import ODEARRefreshPipeline
from utils.eval_chain import frames_to_mp4

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--latents_pt", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--student_ckpt", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--variants", nargs="+",
                   default=["gt", "ar_refresh", "ar_refresh_once", "append_baseline"])
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    pipe = ODEARRefreshPipeline(device, dtype=dtype)
    pipe.build(args.config, use_action_tokens=True)
    pipe.load_checkpoint(args.student_ckpt)

    blob = torch.load(args.latents_pt, map_location="cpu", weights_only=False)
    for v in args.variants:
        if v not in blob:
            log.warning("variant %s not present, skipping", v); continue
        lat = blob[v].to(device=device, dtype=dtype)
        t0 = time.time()
        video_np = pipe.decode_latents(lat)
        path = out / f"{v}.mp4"
        frames_to_mp4(video_np, str(path), fps=args.fps)
        log.info("%s -> %s (frames=%d, %.1fs)", v, path, video_np.shape[0], time.time() - t0)


if __name__ == "__main__":
    main()
