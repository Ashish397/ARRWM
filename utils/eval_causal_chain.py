#!/usr/bin/env python3
"""Causal rolling-context eval with per-rank ride / offset / action configuration.

Produces ``NUM_CAUSAL_VIDEOS`` chained 21-frame videos per rank. By default
``NUM_CAUSAL_VIDEOS = NUM_ACTION_CHUNKS = 7`` so the rollout commits one
generated chunk per video and produces a full causally-generated 24-frame
video (3 seed + 7*3 chained frames) at the end.

**Temporal layout (training-aligned):**

Training uses ``clean_x`` = ride latent frames ``[0..20]`` and the noisy
target = frames ``[3..23]`` (shifted +3). So clean chunk ``c`` is at
temporal frames ``[3c, 3c+1, 3c+2]`` and noisy chunk ``c`` is at
``[3c+3, 3c+4, 3c+5]`` — i.e. clean chunk ``c`` is the 3 frames
**immediately preceding** noisy chunk ``c``. With ``context_shift=1``,
noisy chunk ``c`` attends to clean chunks ``[0..c]``, i.e. all 3(c+1) past
frames, and never its own temporal position.

**Causal rollout (strict per-chunk swap-in):**

Gen ``i+1``'s 0-based chunk ``i`` represents temporal frames
``[off+3(i+1), off+3(i+1)+2]``, which matches clean-slot chunk ``i+1`` at
temporal frames ``[off+3(i+1), off+3(i+1)+2]``. So the correct causal swap
is **gen_j chunk (j-1) → clean_x slot j** (one-chunk-earlier on the gen
side). At video ``v`` (1-based):

- slot 0: ``zarr`` chunk 0 (initial state, anchor).
- slot j for ``1 ≤ j ≤ v-1``: ``gen_j`` chunk ``(j-1)`` (committed).
- slot j for ``v ≤ j ≤ 6``: ``gen_{v-1}`` chunk ``(j-1)`` (latest generated
  carry-forward, not ground-truth).

So from ``v >= 2`` onward, **all non-anchor slots are generated** (no clean
future placeholders), which enforces strict causal rollout and allows error
compounding. By ``v = NUM_ACTION_CHUNKS = 7``, all six non-anchor slots are
filled with committed gen chunks, and ``gen_7`` contributes the final chunk
``g_7[6]`` at temporal frames ``[off+21, off+23]``.

The **displayable causal rollout** is::

    [zarr chunk 0] ++ [g_1[0], g_2[1], g_3[2], g_4[3], g_5[4], g_6[5], g_7[6]]

= 8 chunks = 24 frames, covering ``[off..off+23]``.

This script supports two driving modes:

1. **Legacy v12 fixed-action mode** (default): each rank picks a distinct ride
   from ``--manifest``, uses fixed ``(z2, z7) = (0.5, 0.5)`` on every frame
   for both clean and noisy branches, and runs the ``V12_ASSIGNMENT``
   checkpoint. Matches the original ``torchrun --nproc_per_node=4`` call.

2. **Per-rank config mode** (new): passing ``--per_rank_zarrs``,
   ``--per_rank_offsets`` and ``--per_rank_modes`` plus ``--assignment_index``
   runs each rank on a distinct ride/window with **dataset-driven z-actions**
   (via the ss_vae). ``mode=counterfactual`` negates the noisy-branch actions
   (clean stays untouched). OOD rides are loaded via the disk fallback in
   ``utils.eval_chain._load_ride_entry_from_disk`` when not in the manifest.

Usage:
    # Legacy:
    torchrun --nproc_per_node=4 utils/eval_causal_chain.py \\
        --output_dir eval/eval_causal_chain_out

    # Per-rank (v14) — intended to be launched as 4 parallel single-GPU
    # processes (WORLD_SIZE=1, one rank each) so per-rank config matches
    # CUDA_VISIBLE_DEVICES:
    #   CUDA_VISIBLE_DEVICES=$GPU WORLD_SIZE=1 LOCAL_RANK=0 python utils/eval_causal_chain.py \\
    #       --assignment_index 5 \\
    #       --config configs/causal_lora_diffusion_teacher_v14.yaml \\
    #       --rank_zarr 20240216101235.zarr --rank_offset 100 --rank_mode dataset \\
    #       --encoded_root /projects/u6ex/fbots/frodobots_encoded \\
    #       --caption_root /projects/u6ex/fbots/frodobots_captions/train \\
    #       --output_dir eval/eval_causal_chain_v14_<ts>/gpu0_...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.eval_chain import (
    ChainPipeline,
    MODEL_ASSIGNMENTS,
    NUM_ACTION_CHUNKS,
    NUM_FRAME_PER_BLOCK,
    NUM_FRAMES,
    RAW_ACTION_DIM,
    STREAM_LATENT_SPAN,
    EVAL_LATENT_START_OFFSET,
    VIDEO_NOISE_BASE,
    VIDEO_NOISE_SEED_STRIDE,
    DEFAULT_CAPTION_ROOT,
    DEFAULT_ENCODED_ROOT,
    frame_actions_to_chunk_actions,
    annotate_video,
    CRITIC_ACTION_DIMS,
    _unwrap_manifest_rides,
    _count_latent_frames,
    _build_ts_to_ride_dir,
    _load_ride_entry_from_disk,
    frames_to_mp4,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

NUM_CAUSAL_VIDEOS = NUM_ACTION_CHUNKS  # 7: one video per non-anchor slot to fully commit the rollout.

# Used only in legacy (v12 fixed-action) mode.
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


# ---------------------------------------------------------------------------
# Legacy (v12) ride picker — kept for the fixed-action mode.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# New per-rank loader: disk fallback + dataset-driven clean z-actions.
# ---------------------------------------------------------------------------

def _find_ride_in_manifest(
    rides: list, zarr_basename: str,
) -> Optional[Tuple[int, dict, str, int]]:
    """Search manifest for a ride matching zarr_basename; return slot or None."""
    for idx, cand in enumerate(rides):
        zp = str(cand["zarr_path"])
        if Path(zp).name == zarr_basename:
            try:
                n_lat = _count_latent_frames(zp)
            except Exception:
                continue
            return (idx, cand, zp, n_lat)
    return None


def load_per_rank_ride(
    zarr_basename: str,
    latent_start_offset: int,
    manifest_path: Optional[str],
    encoded_root: str,
    caption_root: str,
    motion_root: str,
    ss_vae_checkpoint: str,
    action_dims: List[int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Return ``(seed, clean_21, prompt_embeds, clean_frame_actions, meta)`` for one rank.

    Tries manifest first (if provided and exists), then falls back to building the
    ride entry directly from ``encoded_root`` + ``caption_root`` via the same path
    ``utils.eval_chain`` uses for OOD rides.

    ``clean_frame_actions`` has shape ``[1, NUM_FRAMES, len(action_dims)]`` and is the
    ss_vae-encoded (tanh-squashed) dataset z-actions sliced to ``action_dims``
    (e.g. ``[2, 7]`` → z2, z7).
    """
    import zarr as zarr_lib
    from utils.zarr_dataset import ZarrRideDataset

    need = latent_start_offset + STREAM_LATENT_SPAN
    ride_dict: Optional[dict] = None
    slot_info: Optional[Tuple[int, str, int]] = None

    if manifest_path and Path(manifest_path).exists():
        log.info("[rank] Searching manifest %s for %s", manifest_path, zarr_basename)
        try:
            manifest = torch.load(manifest_path, map_location="cpu", weights_only=False)
            rides, src = _unwrap_manifest_rides(manifest)
            hit = _find_ride_in_manifest(rides, zarr_basename)
            if hit is not None:
                m_idx, cand, zp, n_lat = hit
                log.info(
                    "[rank] Manifest hit (src=%s, idx=%d): %s (%d latents)",
                    src, m_idx, Path(zp).name, n_lat,
                )
                if n_lat < need:
                    raise RuntimeError(
                        f"Manifest ride {zarr_basename} has {n_lat} latents "
                        f"< required {need}."
                    )
                g = zarr_lib.open_group(zp, mode="r")
                attrs = cand.get("attrs") or dict(g.attrs)
                ride_dict = {
                    "zarr_path": zp,
                    "prompt_embeds": cand["prompt_embeds"].cpu(),
                    "attrs": attrs,
                    "n_latent_frames": n_lat,
                }
                slot_info = (m_idx, zp, n_lat)
            del manifest, rides
        except Exception as exc:
            log.warning("[rank] Manifest lookup failed for %s: %s", zarr_basename, exc)

    if ride_dict is None:
        log.info(
            "[rank] Disk-fallback: building ride entry for %s from encoded_root=%s",
            zarr_basename, encoded_root,
        )
        ts_map = _build_ts_to_ride_dir(Path(caption_root))
        ride_dict = _load_ride_entry_from_disk(
            zarr_basename, Path(encoded_root), Path(caption_root), ts_map,
        )
        n_lat = int(ride_dict["n_latent_frames"])
        if n_lat < need:
            raise RuntimeError(
                f"Disk ride {zarr_basename} has {n_lat} latents < required {need}."
            )
        slot_info = (-1, ride_dict["zarr_path"], n_lat)

    m_idx, zpath, n_lat = slot_info  # type: ignore[assignment]
    assert ride_dict is not None

    z_ds = ZarrRideDataset.from_manifest(
        rides_data=[ride_dict],
        motion_root=motion_root,
        ss_vae_checkpoint=ss_vae_checkpoint,
        device="cpu",
        ss_vae_device="cpu",
    )
    # Cap n_lat at the motion-available latents: rides whose motion file is
    # shorter than the full ride would otherwise overrun encode_z_actions_window
    # (the normal dataset path caps this; per-rank mode did not).
    try:
        from utils.zarr_dataset import _motion_capped_latents
        _attrs = dict(zarr_lib.open_group(zpath, mode="r").attrs)
        _cap = _motion_capped_latents(_attrs, Path(motion_root))
        if _cap > 0:
            n_lat = min(int(n_lat), int(_cap))
    except Exception as _e:  # pragma: no cover - defensive
        log.warning("motion-cap check failed for %s: %s", zpath, _e)
    z_win = z_ds.encode_z_actions_window(
        zpath, n_lat,
        latent_start_offset,
        latent_start_offset + STREAM_LATENT_SPAN,
    )
    clean_fa = z_win[:NUM_FRAMES, action_dims].unsqueeze(0).float()

    g = zarr_lib.open_group(zpath, mode="r")
    lat_np = g["latents"][latent_start_offset : latent_start_offset + NUM_FRAMES]
    assert lat_np.shape[0] == NUM_FRAMES, lat_np.shape
    lat = torch.from_numpy(lat_np.astype(np.float32)).cpu()
    pe = ride_dict["prompt_embeds"].cpu()

    seed = lat[:NUM_FRAME_PER_BLOCK].clone().to(device)
    clean_21 = lat[:NUM_FRAMES].unsqueeze(0).to(device)
    prompt_embeds = pe.unsqueeze(0).to(device)
    clean_frame_actions = clean_fa.to(device)

    meta = {
        "manifest_idx": m_idx,
        "zarr_path": zpath,
        "n_latent_frames": n_lat,
        "latent_start_offset": latent_start_offset,
    }
    log.info(
        "[rank] Loaded %s slice [%d:%d); clean_frame_actions shape=%s",
        Path(zpath).name, latent_start_offset, latent_start_offset + NUM_FRAMES,
        tuple(clean_frame_actions.shape),
    )
    return seed, clean_21, prompt_embeds, clean_frame_actions, meta


# ---------------------------------------------------------------------------
# Core chained clean_x builder (unchanged — this is the heart of the causal
# rolling-context scheme).
# ---------------------------------------------------------------------------

def build_causal_clean_x(
    video_1based: int,
    zarr_clean_21: torch.Tensor,
    prev_gens: List[torch.Tensor],
) -> torch.Tensor:
    """Per-chunk strict causal swap-in.

    ``prev_gens[i]`` is the latents from video ``i+1``; shape ``[1, 21, C, H, W]``,
    representing temporal frames ``[off+3, off+23]``.

    At video ``v`` (1-based) the returned ``clean_x`` has 7 chunks of 3 latent
    frames. Slot indexing matches clean-side temporal positions
    (slot ``j`` covers temporal frames ``[off+3j, off+3j+2]``):

    * slot 0:             zarr chunk 0                 (initial state)
    * slot j (1..v-1):    ``gen_j`` chunk ``(j-1)``    (committed gen, temporally aligned)
    * slot j (v..6):      ``gen_{v-1}`` chunk ``(j-1)`` (latest generated carry-forward)

    At ``v = 1`` the whole ``clean_x`` is zarr. For ``v >= 2``, all non-anchor
    slots are generated content (either committed from earlier videos or the
    latest generation's carry-forward tail). This avoids leaking clean future
    chunks into the causal rollout.
    """
    v = video_1based
    if v == 1:
        return zarr_clean_21
    out_chunks: List[torch.Tensor] = [zarr_clean_21[:, :NUM_FRAME_PER_BLOCK]]
    for j in range(1, NUM_ACTION_CHUNKS):
        if j <= v - 1:
            # Temporal alignment: clean slot j covers the same frames as gen_j chunk (j-1).
            g = prev_gens[j - 1]
            lo = (j - 1) * NUM_FRAME_PER_BLOCK
            hi = lo + NUM_FRAME_PER_BLOCK
            out_chunks.append(g[:, lo:hi])
        else:
            # Strict-causal carry-forward tail: take future slots from the latest
            # available generation (gen_{v-1}), aligned as slot j <- chunk (j-1).
            g_tail = prev_gens[v - 2]
            lo = (j - 1) * NUM_FRAME_PER_BLOCK
            hi = lo + NUM_FRAME_PER_BLOCK
            out_chunks.append(g_tail[:, lo:hi])
    out = torch.cat(out_chunks, dim=1)
    assert out.shape[1] == NUM_FRAMES, (out.shape[1], NUM_FRAMES, v)
    return out


def build_rollout_latents(
    video_1based: int,
    current_gen: torch.Tensor,
    prev_gens: List[torch.Tensor],
) -> torch.Tensor:
    """Return the displayable rollout-so-far latents for video ``v``.

    Display semantics match the causal commitment process the user expects:

    * output chunk j < v-1: use the previously committed chunk from ``gen_{j+1}``
    * output chunk j >= v-1: use the current video's fresh generation

    This means ``video_06`` displays:
        seed ++ [g1[0], g2[1], g3[2], g4[3], g5[4], g6[5], g6[6]]

    rather than a full re-generation ``g6[0..6]``. The latter is still useful
    for debugging, but it is *not* the causal rollout-so-far.
    """
    out_chunks: List[torch.Tensor] = []
    for j in range(NUM_ACTION_CHUNKS):
        lo = j * NUM_FRAME_PER_BLOCK
        hi = lo + NUM_FRAME_PER_BLOCK
        if j < video_1based - 1:
            src = prev_gens[j]
        else:
            src = current_gen
        out_chunks.append(src[:, lo:hi])
    out = torch.cat(out_chunks, dim=1)
    assert out.shape[1] == NUM_FRAMES, (out.shape[1], NUM_FRAMES, video_1based)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="eval/eval_causal_chain_out")
    parser.add_argument("--config", type=str, default="configs/causal_lora_diffusion_teacher.yaml")
    parser.add_argument("--manifest", type=str,
                        default="logs/z_critic_v10_state_tokens/.ride_manifest.pt")
    parser.add_argument("--latent_start_offset", type=int, default=EVAL_LATENT_START_OFFSET,
                        help="Legacy mode only — single offset shared across ranks.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_causal_videos", type=int, default=NUM_CAUSAL_VIDEOS,
                        help="Number of chained videos per rank (default 7).")

    # ---- Per-rank (v14) mode ----
    parser.add_argument("--assignment_index", type=int, default=None,
                        help="If set, use MODEL_ASSIGNMENTS[idx] instead of V12_ASSIGNMENT. "
                             "Required for per-rank mode.")
    parser.add_argument("--rank_zarr", type=str, default=None,
                        help="Zarr basename for THIS rank (single-process-per-GPU style). "
                             "Triggers per-rank mode.")
    parser.add_argument("--rank_offset", type=int, default=None,
                        help="latent_start_offset for THIS rank. Required with --rank_zarr.")
    parser.add_argument("--rank_mode", choices=["dataset", "counterfactual"], default=None,
                        help="Action mode for THIS rank. 'dataset' = noisy = clean = "
                             "ss_vae-encoded dataset z-actions. 'counterfactual' = noisy = "
                             "-dataset, clean = dataset.")
    parser.add_argument("--rank_tag", type=str, default=None,
                        help="Optional short tag to embed in output filenames "
                             "(e.g. 'madrid_dataset').")
    parser.add_argument("--encoded_root", type=str, default=DEFAULT_ENCODED_ROOT,
                        help="Fallback encoded-zarr root for rides not in manifest.")
    parser.add_argument("--caption_root", type=str, default=DEFAULT_CAPTION_ROOT,
                        help="Caption root (for prompt embeds + ts→ride_dir map).")
    args = parser.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # Decide which mode we're in.
    per_rank_mode = args.rank_zarr is not None
    if per_rank_mode:
        if args.rank_offset is None or args.rank_mode is None:
            raise SystemExit("--rank_zarr requires --rank_offset and --rank_mode")
        if args.assignment_index is None:
            raise SystemExit("--rank_zarr requires --assignment_index (pick entry in MODEL_ASSIGNMENTS)")
        if world != 1:
            log.warning(
                "rank_zarr mode is designed for single-process-per-GPU launches "
                "(WORLD_SIZE=1). world=%d may produce unexpected results.", world,
            )
        assignment = MODEL_ASSIGNMENTS[args.assignment_index]
    else:
        assignment = V12_ASSIGNMENT
    label = assignment["label"]

    log.info(
        "Rank %d/%d: mode=%s, assignment=%s (%s)",
        rank, world, "per_rank" if per_rank_mode else "legacy_fixed",
        label, assignment["ckpt"],
    )

    # Load OmegaConf for motion_root / ss_vae_checkpoint / action_dims (per-rank mode only).
    from omegaconf import OmegaConf
    _cfg = OmegaConf.load(args.config)
    motion_root = str(_cfg.get("motion_root", "") or "")
    ss_vae_ckpt = str(_cfg.get("ss_vae_checkpoint", "action_query/checkpoints/ss_vae_8free.pt"))
    action_dims = list(_cfg.get("action_dims", [2, 7]))

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- Ride loading ----
    if per_rank_mode:
        if not motion_root:
            raise SystemExit("--config must define motion_root (required for per-rank mode)")
        seed_lat, zarr_clean_21, prompt_embeds, clean_fa, ride_meta = load_per_rank_ride(
            zarr_basename=args.rank_zarr,
            latent_start_offset=args.rank_offset,
            manifest_path=args.manifest,
            encoded_root=args.encoded_root,
            caption_root=args.caption_root,
            motion_root=motion_root,
            ss_vae_checkpoint=ss_vae_ckpt,
            action_dims=action_dims,
            device=device,
        )
        ride_meta["rank_mode"] = args.rank_mode
        ride_meta["rank_tag"] = args.rank_tag or ""
    else:
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
        clean_fa = None  # built from fixed values later
        ride_meta["latent_start_offset"] = args.latent_start_offset

    meta_path = out_root / f"rank{rank}_ride.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(ride_meta, fh, indent=2, default=str)

    # ---- Per-rank output subdir naming ----
    if per_rank_mode and args.rank_tag:
        out_dir = out_root / f"rank{rank}_{label}_{args.rank_tag}"
    else:
        out_dir = out_root / f"rank{rank}_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = ChainPipeline(device)
    pipe.build(args.config, use_action_tokens=assignment.get("has_action_tokens", True))
    pipe.load_checkpoint(assignment["ckpt"], assignment)

    pe_dtype = pipe.dtype
    prompt_embeds = prompt_embeds.to(dtype=pe_dtype)

    # ---- Build clean / noisy frame actions ----
    if per_rank_mode:
        clean_fa = clean_fa.to(dtype=pe_dtype)  # [1, NUM_FRAMES, len(action_dims)]
        if args.rank_mode == "dataset":
            noisy_fa = clean_fa.clone()
            cond_tag = "dataset"
        elif args.rank_mode == "counterfactual":
            noisy_fa = -clean_fa
            cond_tag = "counterfactual"
        else:  # defensive
            raise SystemExit(f"Unknown rank_mode {args.rank_mode!r}")
        log.info(
            "[%s] rank=%d mode=%s  clean_fa shape=%s  (z2 range [%.3f,%.3f], z7 [%.3f,%.3f])",
            label, rank, cond_tag, tuple(clean_fa.shape),
            float(clean_fa[..., 0].min()), float(clean_fa[..., 0].max()),
            float(clean_fa[..., 1].min()), float(clean_fa[..., 1].max()),
        )
    else:
        fixed = torch.empty(1, NUM_FRAMES, RAW_ACTION_DIM, device=device, dtype=pe_dtype)
        fixed[..., 0] = FIXED_ACTION_Z2
        fixed[..., 1] = FIXED_ACTION_Z7
        clean_fa = fixed
        noisy_fa = fixed
        cond_tag = "fixed"

    chunk_dev_all = frame_actions_to_chunk_actions(noisy_fa)

    prev_gens: List[torch.Tensor] = []
    prev_video_raws: List[np.ndarray] = []

    total_videos = 0
    for vid in range(1, args.num_causal_videos + 1):
        chunk_dev = chunk_dev_all

        clean_x = build_causal_clean_x(vid, zarr_clean_21, prev_gens)

        cond = pipe.build_conditional(
            prompt_embeds, noisy_fa, clean_fa,
            assignment.get("has_adaln", False),
            assignment.get("has_action_tokens", False),
        )

        noise_seed = args.seed + VIDEO_NOISE_BASE + vid * VIDEO_NOISE_SEED_STRIDE
        torch.manual_seed(noise_seed)
        torch.cuda.manual_seed(noise_seed)

        t0 = time.time()
        gen_latents = pipe.generate(cond, clean_x)
        log.info(
            "[%s] rank=%d video=%d/%d cond=%s clean_x built | gen %.1fs | noise_seed=%s",
            label, rank, vid, args.num_causal_videos, cond_tag,
            time.time() - t0, noise_seed,
        )
        gen_latents = gen_latents.detach()

        context_latents_3 = seed_lat.unsqueeze(0)
        context_np = pipe.decode_latents(context_latents_3)
        video_np = pipe.decode_latents(gen_latents)

        motion, teacher_z_8d = pipe.compute_teacher_visuals(gen_latents)
        n_c = teacher_z_8d.shape[1]
        teacher_z2z7 = teacher_z_8d[:, :, CRITIC_ACTION_DIMS]
        critic_z2z7 = None
        if pipe.action_critic is not None:
            cp = pipe.run_critic(gen_latents, chunk_dev)
            if cp is not None:
                critic_z2z7 = cp[:, :, CRITIC_ACTION_DIMS]
        target_z = chunk_dev[:, :n_c].contiguous()

        title_bits = [label, f"r{rank}", f"v{vid}", cond_tag]
        if per_rank_mode and args.rank_tag:
            title_bits.append(args.rank_tag)
        title = " ".join(title_bits)
        ann = annotate_video(video_np, teacher_z2z7, critic_z2z7, target_z, motion, title)

        # -----------------------------------------------------------------
        # Build the causal rollout-so-far for display:
        #   seed ++ [committed chunks from earlier videos] ++ [current suffix]
        #
        # This is what users expect ``video_0v`` to mean. Saving ``g_v[0..6]``
        # directly causes apparent "resets" because already-committed chunks are
        # re-generated instead of frozen from earlier videos.
        # -----------------------------------------------------------------
        rollout_latents = build_rollout_latents(vid, gen_latents, prev_gens)
        rollout_motion, rollout_teacher_z_8d = pipe.compute_teacher_visuals(rollout_latents)
        rollout_teacher_z2z7 = rollout_teacher_z_8d[:, :, CRITIC_ACTION_DIMS]
        rollout_critic_z2z7 = None
        if pipe.action_critic is not None:
            rollout_cp = pipe.run_critic(rollout_latents, chunk_dev)
            if rollout_cp is not None:
                rollout_critic_z2z7 = rollout_cp[:, :, CRITIC_ACTION_DIMS]
        rollout_target_z = chunk_dev[:, : rollout_teacher_z_8d.shape[1]].contiguous()

        chunk_px = video_np.shape[0] // NUM_ACTION_CHUNKS
        assert video_np.shape[0] == chunk_px * NUM_ACTION_CHUNKS, (
            video_np.shape[0], chunk_px, NUM_ACTION_CHUNKS,
        )
        assert context_np.shape[0] == chunk_px, (context_np.shape[0], chunk_px)
        rollout_parts = [context_np]
        for j in range(NUM_ACTION_CHUNKS):
            lo = j * chunk_px
            hi = lo + chunk_px
            if j < vid - 1:
                src = prev_video_raws[j]
            else:
                src = video_np
            rollout_parts.append(src[lo:hi])
        rollout_video_cat = np.concatenate(rollout_parts, axis=0)

        rollout_title_bits = [label, f"r{rank}", f"v{vid}", "rollout", cond_tag]
        if per_rank_mode and args.rank_tag:
            rollout_title_bits.append(args.rank_tag)
        rollout_title = " ".join(rollout_title_bits)
        rollout_ann = annotate_video(
            rollout_video_cat[chunk_px:],
            rollout_teacher_z2z7,
            rollout_critic_z2z7,
            rollout_target_z,
            rollout_motion,
            rollout_title,
        )
        rollout_ann_cat = np.concatenate([context_np, rollout_ann], axis=0)

        if per_rank_mode:
            zarr_ts = Path(ride_meta["zarr_path"]).stem
            pieces = [f"video_{vid:02d}", zarr_ts, cond_tag]
            if args.rank_tag:
                pieces.append(args.rank_tag)
            fname_stem = "_".join(pieces)
        else:
            fname_stem = f"video_{vid:02d}"

        # Main outputs: rollout-so-far semantics.
        frames_to_mp4(rollout_ann_cat, str(out_dir / f"{fname_stem}_annotated.mp4"))
        frames_to_mp4(rollout_video_cat, str(out_dir / f"{fname_stem}_raw.mp4"))
        # Debug outputs: raw generation from this pass only.
        frames_to_mp4(np.concatenate([context_np, ann], axis=0), str(out_dir / f"{fname_stem}_pass_annotated.mp4"))
        frames_to_mp4(np.concatenate([context_np, video_np], axis=0), str(out_dir / f"{fname_stem}_pass_raw.mp4"))

        prev_gens.append(gen_latents)
        prev_video_raws.append(video_np)
        total_videos += 1
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------------
    # Final causal rollout: seed ++ [g1[0], g2[1], ..., g7[6]].
    #
    # Raw pixels are stitched from per-pass decodes to avoid VAE boundary
    # contamination when neighboring chunks come from different generations.
    # ---------------------------------------------------------------------
    if len(prev_gens) >= NUM_ACTION_CHUNKS:
        committed_chunks: List[torch.Tensor] = []
        for i in range(NUM_ACTION_CHUNKS):
            g = prev_gens[i]  # gen_{i+1}
            lo = i * NUM_FRAME_PER_BLOCK
            hi = lo + NUM_FRAME_PER_BLOCK
            committed_chunks.append(g[:, lo:hi])
        rollout_gen = torch.cat(committed_chunks, dim=1)
        assert rollout_gen.shape[1] == NUM_FRAMES, rollout_gen.shape

        # --- latent chunk-boundary diagnostic (LATENT_BOUNDARY_DIAG=1) --------
        # Decode-vs-training discriminator: measure the per-adjacent-frame RMS
        # delta in LATENT space on the committed rollout. Frame index i is a
        # chunk boundary iff i % NUM_FRAME_PER_BLOCK == 0 (i>0): frames i-1, i
        # come from different committed gens. If loo_f3 shows boundary>>interior
        # but v14 doesn't, the discontinuity is in the model's committed latents
        # (a real AR chunk-consistency regression) — NOT a decode-stitch seam.
        if os.environ.get("LATENT_BOUNDARY_DIAG") == "1":
            with torch.no_grad():
                rg = rollout_gen[0].float()  # [T, C, H, W]
                d = (rg[1:] - rg[:-1]).flatten(1).pow(2).mean(1).sqrt()  # [T-1] RMS per gap
                bnd, intr = [], []
                for i in range(1, rg.shape[0]):
                    (bnd if i % NUM_FRAME_PER_BLOCK == 0 else intr).append(float(d[i - 1]))
                b = float(np.mean(bnd)) if bnd else 0.0
                it = float(np.mean(intr)) if intr else 0.0
                log.info(
                    "[LATENT_BOUNDARY_DIAG] %s rank=%d boundary_rms=%.5f interior_rms=%.5f ratio=%.3f gaps=%s",
                    label, rank, b, it, (b / it if it else float("nan")),
                    [round(float(x), 4) for x in d.tolist()],
                )

        context_np = pipe.decode_latents(seed_lat.unsqueeze(0))
        if not prev_video_raws:
            raise RuntimeError("Expected cached decoded videos for final rollout.")
        chunk_px = prev_video_raws[0].shape[0] // NUM_ACTION_CHUNKS
        if os.environ.get("SMOOTH_DECODE") == "1":
            # Decode the committed latent rollout in ONE pass so the VAE temporal
            # conv spans the whole sequence -> no per-chunk decode seams. (The
            # default per-pass-stitched path below takes chunk i's pixels from a
            # different generation's decode, leaving a seam every chunk.)
            rollout_cat = np.concatenate([context_np, pipe.decode_latents(rollout_gen)], axis=0)
        else:
            rollout_parts = [context_np]
            for i in range(NUM_ACTION_CHUNKS):
                lo = i * chunk_px
                hi = lo + chunk_px
                rollout_parts.append(prev_video_raws[i][lo:hi])
            rollout_cat = np.concatenate(rollout_parts, axis=0)

        motion_r, teacher_z_8d_r = pipe.compute_teacher_visuals(rollout_gen)
        n_c_r = teacher_z_8d_r.shape[1]
        teacher_z2z7_r = teacher_z_8d_r[:, :, CRITIC_ACTION_DIMS]
        critic_z2z7_r = None
        if pipe.action_critic is not None:
            cp_r = pipe.run_critic(rollout_gen, chunk_dev_all)
            if cp_r is not None:
                critic_z2z7_r = cp_r[:, :, CRITIC_ACTION_DIMS]
        target_z_r = chunk_dev_all[:, :n_c_r].contiguous()

        title_bits_r = [label, f"r{rank}", "rollout", cond_tag]
        if per_rank_mode and args.rank_tag:
            title_bits_r.append(args.rank_tag)
        title_r = " ".join(title_bits_r)
        ann_r = annotate_video(
            rollout_cat[chunk_px:], teacher_z2z7_r, critic_z2z7_r, target_z_r, motion_r, title_r,
        )
        ann_cat_r = np.concatenate([context_np, ann_r], axis=0)

        if per_rank_mode:
            zarr_ts = Path(ride_meta["zarr_path"]).stem
            pieces = ["final_causal_rollout", zarr_ts, cond_tag]
            if args.rank_tag:
                pieces.append(args.rank_tag)
            rollout_stem = "_".join(pieces)
        else:
            rollout_stem = "final_causal_rollout"
        frames_to_mp4(ann_cat_r, str(out_dir / f"{rollout_stem}_annotated.mp4"))
        frames_to_mp4(rollout_cat, str(out_dir / f"{rollout_stem}_raw.mp4"))
        # Ground-truth video of the same 24-frame span (seed + real continuation),
        # decoded with the same pipeline so it aligns frame-for-frame with the
        # generated rollout: context_np is the real seed; the continuation is the
        # real latents [off+block .. off+block+NUM_FRAMES).
        try:
            import zarr as zarr_lib
            _g_gt = zarr_lib.open_group(ride_meta["zarr_path"], mode="r")
            _o2 = int(ride_meta["latent_start_offset"]) + NUM_FRAME_PER_BLOCK
            _gt_np = _g_gt["latents"][_o2:_o2 + NUM_FRAMES]
            _gt_lat = torch.from_numpy(_gt_np.astype(np.float32)).unsqueeze(0).to(seed_lat.device)
            _gt_rest = pipe.decode_latents(_gt_lat)
            _gt_cat = np.concatenate([context_np, _gt_rest], axis=0)
            frames_to_mp4(_gt_cat, str(out_dir / f"{rollout_stem}_gt.mp4"))
            log.info("[%s] rank=%d GT rollout written: %s_gt.mp4", label, rank, rollout_stem)
        except Exception as _e:  # pragma: no cover - defensive
            log.warning("GT rollout write failed: %s", _e)
        log.info(
            "[%s] rank=%d final rollout written: %s (8 chunks = 24 frames, %s)",
            label, rank, rollout_stem, cond_tag,
        )
    else:
        log.warning(
            "[%s] rank=%d only %d gens (need %d) — skipping final causal rollout.",
            label, rank, len(prev_gens), NUM_ACTION_CHUNKS,
        )

    log.info("[%s] rank=%d done: %d chained videos → %s", label, rank, total_videos, out_dir)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
