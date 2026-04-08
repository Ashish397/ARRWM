#!/usr/bin/env python3
"""Causal rolling-context eval: v12 only, 4 GPUs × 6 videos with chained clean_x.

Video 1: ``clean_x`` = 21 ride latents (same as ``eval_chain``).

Video 2: first chunk from zarr + first 6 chunks of video-1 latents (drop last chunk).

Video k (3≤k≤6): chunk 0 = zarr clean chunk; for j=1..k-2, chunk j from generation j
(0-based chunk index j); remaining chunks from generation k-1 starting at **chunk index k-1**
(aligned with the diagonal), through the end of that generation.

So by video 6 (the last run), chunk 0 is zarr; chunks 1..4 come from gen 1..4 at indices 1..4;
chunks 5..6 come from gen 5 at chunk indices 5 and 6.

Each GPU loads **v12** and a **distinct** manifest ride (first ``world_size`` eligible
distinct ``zarr_path`` entries). No cross-GPU latent exchange.

**Actions:** Both **clean** and **noisy** conditioning use ``(z2, z7) = (0.5, 0.5)`` on every
frame (every chunk is the same). Diffusion noise seeds depend on video index only, not rank.

Usage:
    torchrun --nproc_per_node=4 utils/eval_causal_chain.py --output_dir eval/eval_causal_chain_out
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse pipeline and utilities from eval_chain (same model / VAE / critic wiring).
from utils.eval_chain import (
    ChainPipeline,
    NUM_ACTION_CHUNKS,
    NUM_FRAME_PER_BLOCK,
    NUM_FRAMES,
    RAW_ACTION_DIM,
    STREAM_LATENT_SPAN,
    EVAL_LATENT_START_OFFSET,
    VIDEO_NOISE_BASE,
    VIDEO_NOISE_SEED_STRIDE,
    frame_actions_to_chunk_actions,
    annotate_video,
    CRITIC_ACTION_DIMS,
    _unwrap_manifest_rides,
    _count_latent_frames,
    frames_to_mp4,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

NUM_CAUSAL_VIDEOS = 6

# Fixed (z2, z7) for both clean and noisy branches; each 3-frame chunk is identical.
FIXED_ACTION_Z2 = 0.5
FIXED_ACTION_Z7 = 0.5

V12_ASSIGNMENT: Dict = {
    "ckpt": "logs/z_critic_v12_probe_fixes/causal_lora_step0003250.pt",
    "label": "v12_probe_fixes",
    "has_critic": True,
    "has_adaln": True,
    "has_action_tokens": True,
    "critic_base_ch": 128,
    "critic_res_blocks": 4,
}


def pick_distinct_eligible_rides(
    rides: list, world: int, need_latents: int,
) -> List[Tuple[int, dict, str, int]]:
    """Return ``world`` entries ``(manifest_idx, ride_dict, zpath, n_lat)`` with unique zpaths."""
    eligible: List[Tuple[int, dict, str, int]] = []
    seen = set()
    for idx, cand in enumerate(rides):
        zp = str(cand["zarr_path"])
        if zp in seen:
            continue
        n_lat = _count_latent_frames(zp)
        if n_lat < need_latents:
            continue
        eligible.append((idx, cand, zp, n_lat))
        seen.add(zp)
        if len(eligible) >= world:
            break
    if len(eligible) < world:
        raise RuntimeError(
            f"Need {world} distinct rides with ≥{need_latents} latents; found {len(eligible)}."
        )
    return eligible


def load_one_ride_tensors(
    slot: Tuple[int, dict, str, int],
    latent_start_offset: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Return seed [3,C,H,W], clean_21 [1,21,C,H,W], prompt [1,S,D], meta."""
    manifest_idx, ride, zpath, _ = slot
    import zarr as zarr_lib

    g = zarr_lib.open_group(zpath, mode="r")
    lat_np = g["latents"][latent_start_offset : latent_start_offset + NUM_FRAMES]
    lat = torch.from_numpy(lat_np.astype(np.float32)).cpu()
    prompt_embeds = ride["prompt_embeds"].cpu()

    seed = lat[:NUM_FRAME_PER_BLOCK].clone().to(device)
    clean_21 = lat[:NUM_FRAMES].unsqueeze(0).to(device)
    pe = prompt_embeds.unsqueeze(0).to(device)
    meta = {"manifest_idx": manifest_idx, "zarr_path": zpath}
    return seed, clean_21, pe, meta


def build_causal_clean_x(
    video_1based: int,
    zarr_clean_21: torch.Tensor,
    prev_gens: List[torch.Tensor],
) -> torch.Tensor:
    """``prev_gens[i]`` = latents from video ``i+1``; shape [1,21,C,H,W]."""
    if video_1based == 1:
        return zarr_clean_21
    if video_1based == 2:
        z0 = zarr_clean_21[:, :NUM_FRAME_PER_BLOCK]
        g1 = prev_gens[0]
        rest = g1[:, : NUM_FRAMES - NUM_FRAME_PER_BLOCK]
        return torch.cat([z0, rest], dim=1)
    v = video_1based
    parts = [zarr_clean_21[:, :NUM_FRAME_PER_BLOCK]]
    for j in range(1, v - 1):
        g = prev_gens[j - 1]
        lo = j * NUM_FRAME_PER_BLOCK
        hi = lo + NUM_FRAME_PER_BLOCK
        parts.append(g[:, lo:hi])
    n_tail_chunks = NUM_ACTION_CHUNKS - 1 - (v - 2)
    tail_lo = (v - 1) * NUM_FRAME_PER_BLOCK
    tail_hi = tail_lo + n_tail_chunks * NUM_FRAME_PER_BLOCK
    tail = prev_gens[v - 2][:, tail_lo:tail_hi]
    parts.append(tail)
    out = torch.cat(parts, dim=1)
    assert out.shape[1] == NUM_FRAMES, (out.shape[1], NUM_FRAMES, v)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="eval/eval_causal_chain_out")
    parser.add_argument("--config", type=str, default="configs/causal_lora_diffusion_teacher.yaml")
    parser.add_argument("--manifest", type=str,
                        default="logs/z_critic_v10_state_tokens/.ride_manifest.pt")
    parser.add_argument("--latent_start_offset", type=int, default=EVAL_LATENT_START_OFFSET)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    need = args.latent_start_offset + STREAM_LATENT_SPAN
    manifest = torch.load(args.manifest, map_location="cpu", weights_only=False)
    rides, src = _unwrap_manifest_rides(manifest)
    if not rides:
        raise RuntimeError(f"No rides in manifest {args.manifest}")
    eligible = pick_distinct_eligible_rides(rides, world, need)
    if rank >= len(eligible):
        raise RuntimeError(f"LOCAL_RANK {rank} >= {len(eligible)} eligible rides")
    slot = eligible[rank]

    log.info(
        "Rank %d/%d: manifest source=%s, ride idx=%d, zarr=%s",
        rank, world, src, slot[0], Path(slot[2]).name,
    )

    seed_lat, zarr_clean_21, prompt_embeds, ride_meta = load_one_ride_tensors(
        slot, args.latent_start_offset, device,
    )

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    meta_path = out_root / f"rank{rank}_ride.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(ride_meta, fh, indent=2)

    label = V12_ASSIGNMENT["label"]
    out_dir = out_root / f"rank{rank}_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = ChainPipeline(device)
    pipe.build(args.config, use_action_tokens=V12_ASSIGNMENT.get("has_action_tokens", True))
    pipe.load_checkpoint(V12_ASSIGNMENT["ckpt"], V12_ASSIGNMENT)

    pe_dtype = pipe.dtype
    prompt_embeds = prompt_embeds.to(dtype=pe_dtype)

    fixed_frame_actions = torch.empty(1, NUM_FRAMES, RAW_ACTION_DIM, device=device, dtype=pe_dtype)
    fixed_frame_actions[..., 0] = FIXED_ACTION_Z2
    fixed_frame_actions[..., 1] = FIXED_ACTION_Z7
    chunk_dev_all = frame_actions_to_chunk_actions(fixed_frame_actions)

    prev_gens: List[torch.Tensor] = []

    for vid in range(1, NUM_CAUSAL_VIDEOS + 1):
        chunk_dev = chunk_dev_all

        clean_x = build_causal_clean_x(vid, zarr_clean_21, prev_gens)

        cond = pipe.build_conditional(
            prompt_embeds, fixed_frame_actions, fixed_frame_actions,
            V12_ASSIGNMENT.get("has_adaln", False),
            V12_ASSIGNMENT.get("has_action_tokens", False),
        )

        noise_seed = args.seed + VIDEO_NOISE_BASE + vid * VIDEO_NOISE_SEED_STRIDE
        torch.manual_seed(noise_seed)
        torch.cuda.manual_seed(noise_seed)

        t0 = time.time()
        gen_latents = pipe.generate(cond, clean_x)
        log.info(
            "[%s] rank=%d video=%d/%d clean_x built | gen %.1fs | noise_seed=%s",
            label, rank, vid, NUM_CAUSAL_VIDEOS, time.time() - t0, noise_seed,
        )
        prev_gens.append(gen_latents.detach())

        context_latents_3 = seed_lat.unsqueeze(0)
        context_np = pipe.decode_latents(context_latents_3)
        video_np = pipe.decode_latents(gen_latents)
        video_cat = np.concatenate([context_np, video_np], axis=0)

        motion, teacher_z_8d = pipe.compute_teacher_visuals(gen_latents)
        n_c = teacher_z_8d.shape[1]
        teacher_z2z7 = teacher_z_8d[:, :, CRITIC_ACTION_DIMS]
        critic_z2z7 = None
        if pipe.action_critic is not None:
            cp = pipe.run_critic(gen_latents, chunk_dev)
            if cp is not None:
                critic_z2z7 = cp[:, :, CRITIC_ACTION_DIMS]
        target_z = chunk_dev[:, :n_c].contiguous()
        title = f"{label} r{rank} v{vid} causal_clean"
        ann = annotate_video(video_np, teacher_z2z7, critic_z2z7, target_z, motion, title)
        ann_cat = np.concatenate([context_np, ann], axis=0)

        frames_to_mp4(ann_cat, str(out_dir / f"video_{vid:02d}_annotated.mp4"))
        frames_to_mp4(video_cat, str(out_dir / f"video_{vid:02d}_raw.mp4"))
        torch.cuda.empty_cache()

    log.info("[%s] rank=%d done → %s", label, rank, out_dir)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
