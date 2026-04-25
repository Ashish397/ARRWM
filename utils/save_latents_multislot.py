#!/usr/bin/env python3
"""AR_refresh staircase rollout with N parallel future slots.

Staircase scheduling: each slot lives at a different rung of the
denoising ladder. Slot 0 is closest to clean (committed first); slot
N-1 is freshly noised. Each rolling step does P forward passes
advancing every live slot by P rungs; slot 0 reaches the bottom of
the ladder and commits one chunk. After the commit, slots shift up
(slot k+1 → slot k) and a fresh-noise slot is appended at the back.

Constraint: NS × P = total_denoise (= 4 for the ODE student).
  * NS=2  P=2  ladder=[1000, 683, 367, 50]  →  steady state slot
    indices [2, 0] (slot 0 at t≈367, slot 1 at t=1000).
  * NS=4  P=1  ladder=[1000, 683, 367, 50]  →  steady state slot
    indices [3, 2, 1, 0] (slot 0 at t=50, slot 3 at t=1000).

Slot 0 carries the real chunk action (the action of the chunk it's
about to commit). Slots 1..N-1 carry decayed echoes of slot 0's action
(no future-action peeking — matches deployment semantics):
  * NS=2 → ``[1.0, 0.5]``
  * NS=4 → ``[1.0, 0.75, 0.5, 0.25]``

Two attention-mask modes between slots:
  * ``causal``         — slot k attends to ctx + slots 0..k. Default
    block-causal mask (each slot is its own 3-frame block).
  * ``bidirectional``  — all N slots attend to each other AND all ctx.
    Built explicitly via ``create_block_mask`` with all slot tokens
    sharing the same ``ends`` value.

Outputs per config::
    <output_dir>/<config_name>/
      latents.pt       # {gt: ..., ar_refresh_multislot: ...}
      ar_refresh.mp4   # decoded student
      manifest.json
    <output_dir>/gt.mp4   # decoded GT (shared)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.nn.attention.flex_attention import create_block_mask

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_AF_ROOT = _REPO_ROOT / "action-forcing"
if str(_AF_ROOT) not in sys.path:
    sys.path.insert(0, str(_AF_ROOT))

from utils.eval_causal_AR import (
    load_per_rank_ride_ar,
    FRAME_SPATIAL_TOKENS,
    BASE_CHUNK_FRAMES,
)
from utils.eval_causal_AR_chain import ODEARRefreshPipeline
from utils.eval_chain import frames_to_mp4

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def build_multislot_mask(
    device: torch.device,
    n_ctx_chunks: int,
    num_slots: int,
    frame_seqlen: int,
    npb: int = 3,
):
    """Block-causal mask among ctx chunks; all current slots glued into
    one mega-block that fully attends to itself + all ctx."""
    ctx_frames = n_ctx_chunks * npb
    slot_frames = num_slots * npb
    total_frames = ctx_frames + slot_frames
    total_length = total_frames * frame_seqlen
    padded_length = math.ceil(total_length / 128) * 128 - total_length

    ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
    for c in range(n_ctx_chunks):
        s = c * npb * frame_seqlen
        e = s + npb * frame_seqlen
        ends[s:e] = e
    if slot_frames > 0:
        s = ctx_frames * frame_seqlen
        e = total_length
        ends[s:e] = e

    def attention_mask(b, h, q_idx, kv_idx):
        return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)

    return create_block_mask(
        attention_mask, B=None, H=None,
        Q_LEN=total_length + padded_length,
        KV_LEN=total_length + padded_length,
        _compile=True, device=device,
    )


@torch.no_grad()
def run_ar_refresh_staircase(
    pipe, *,
    prompt_embeds_dev, noisy_fa_full, initial_latents_dev,
    num_gen_chunks, fifo_size, num_frame_per_block, device, dtype,
    num_slots: int, action_filter: List[float], mask_mode: str,
) -> torch.Tensor:
    assert mask_mode in ("causal", "bidirectional")
    assert len(action_filter) == num_slots
    assert num_slots in (1, 2, 4), f"num_slots must be 1/2/4, got {num_slots}"

    base_dit = pipe.wrapper.model
    if hasattr(base_dit, "get_base_model"):
        try: base_dit = base_dit.get_base_model()
        except Exception: pass

    B = 1
    NS = num_slots
    total_denoise = 4
    P = total_denoise // NS  # passes per rolling step

    seed_frames = int(initial_latents_dev.shape[1])
    seed_chunks = seed_frames // num_frame_per_block
    max_window_blocks = fifo_size + NS
    max_window_frames = max_window_blocks * num_frame_per_block
    action_tokens_per_frame = int(getattr(base_dit, "action_tokens_per_frame", 1))
    frame_seq_length = FRAME_SPATIAL_TOKENS + action_tokens_per_frame
    pipe.wrapper.seq_len = max(int(pipe.wrapper.seq_len), max_window_frames * frame_seq_length)
    base_dit.num_frame_per_block = num_frame_per_block

    prev_local_attn_size = getattr(base_dit, "local_attn_size", -1)
    base_dit.local_attn_size = -1
    for _, module in base_dit.named_modules():
        if hasattr(module, "local_attn_size"):
            try: module.local_attn_size = -1
            except Exception: pass

    scheduler = pipe.scheduler
    scheduler.sigmas = scheduler.sigmas.to(device)
    # Descending ladder, length total_denoise.
    ladder = torch.linspace(1000.0, 50.0, total_denoise).tolist()

    C = int(initial_latents_dev.shape[2])
    H = int(initial_latents_dev.shape[3])
    W = int(initial_latents_dev.shape[4])

    # FIFO of committed chunks (oldest first). Seeded with the GT seed.
    fifo: List[torch.Tensor] = [initial_latents_dev]
    next_commit_chunk = seed_chunks  # global chunk index of the next commit
    # live_slots: list of (latent, ladder_idx). Slot 0 is first to commit.
    live_slots: List[Tuple[torch.Tensor, int]] = []
    generated: List[torch.Tensor] = []
    cached_window_blocks = -1
    cached_n_ctx_blocks = -1

    log.info("staircase rollout: NS=%d P=%d ladder=%s filter=%s mask=%s",
             NS, P, [round(t, 1) for t in ladder], action_filter, mask_mode)

    def _build_fa_window(n_ctx_blocks: int, n_slots_now: int) -> torch.Tensor:
        cur_global_lo = next_commit_chunk * num_frame_per_block
        ctx_global_lo = cur_global_lo - n_ctx_blocks * num_frame_per_block
        ctx_fa = (
            noisy_fa_full[:, ctx_global_lo:cur_global_lo].contiguous()
            if n_ctx_blocks > 0 else None
        )
        slot0_a = noisy_fa_full[:, cur_global_lo:cur_global_lo + num_frame_per_block]
        slot_pieces = [action_filter[s] * slot0_a for s in range(n_slots_now)]
        slot_fa = torch.cat(slot_pieces, dim=1) if slot_pieces else None
        if ctx_fa is not None and slot_fa is not None:
            return torch.cat([ctx_fa, slot_fa], dim=1).contiguous()
        if ctx_fa is not None:
            return ctx_fa
        return slot_fa.contiguous()

    def _ensure_block_mask(n_ctx_blocks: int, window_blocks: int):
        nonlocal cached_window_blocks, cached_n_ctx_blocks
        if window_blocks == cached_window_blocks and n_ctx_blocks == cached_n_ctx_blocks:
            return
        if mask_mode == "bidirectional":
            base_dit.block_mask = build_multislot_mask(
                device=device,
                n_ctx_chunks=n_ctx_blocks,
                num_slots=window_blocks - n_ctx_blocks,
                frame_seqlen=frame_seq_length,
                npb=num_frame_per_block,
            )
        else:
            base_dit.block_mask = None  # default block-causal rebuilds
        cached_window_blocks = window_blocks
        cached_n_ctx_blocks = n_ctx_blocks
        log.info("[staircase] mask rebuilt: window=%d blocks (%d ctx + %d slots)  mode=%s",
                 window_blocks, n_ctx_blocks, window_blocks - n_ctx_blocks, mask_mode)

    def _do_forward():
        """One forward pass; advance each slot by 1 ladder step; commit if slot 0 reaches bottom."""
        nonlocal live_slots, next_commit_chunk
        n_ctx_blocks = min(len(fifo), fifo_size)
        n_slots_now = len(live_slots)
        window_blocks = n_ctx_blocks + n_slots_now
        _ensure_block_mask(n_ctx_blocks, window_blocks)

        ctx_chunks = fifo[-n_ctx_blocks:] if n_ctx_blocks > 0 else []
        ctx_cat = torch.cat(ctx_chunks, dim=1) if ctx_chunks else None
        slot_cat = torch.cat([s[0] for s in live_slots], dim=1) if live_slots else None
        if ctx_cat is not None and slot_cat is not None:
            x_full = torch.cat([ctx_cat, slot_cat], dim=1)
        elif ctx_cat is not None:
            x_full = ctx_cat
        else:
            x_full = slot_cat

        fa_window = _build_fa_window(n_ctx_blocks, n_slots_now)

        # Per-frame timestep: ctx at 0, each slot at its ladder rung.
        t_ctx_vec = torch.zeros(
            [B, n_ctx_blocks * num_frame_per_block], device=device, dtype=torch.float32,
        )
        t_slot_pieces = [
            torch.full(
                [B, num_frame_per_block], float(ladder[idx]),
                device=device, dtype=torch.float32,
            )
            for _, idx in live_slots
        ]
        t_slot_vec = torch.cat(t_slot_pieces, dim=1) if t_slot_pieces else None
        if t_slot_vec is not None and t_ctx_vec.numel() > 0:
            tt = torch.cat([t_ctx_vec, t_slot_vec], dim=1)
        elif t_slot_vec is not None:
            tt = t_slot_vec
        else:
            tt = t_ctx_vec

        cond = pipe._build_action_cond_chunk(
            prompt_embeds_dev, fa_window,
            num_frames=window_blocks * num_frame_per_block,
        )

        with torch.amp.autocast("cuda", dtype=dtype):
            out = pipe.wrapper(
                noisy_image_or_video=x_full,
                conditional_dict=cond,
                timestep=tt,
                clean_x=None, aug_t=None,
            )
        pred_x0_window = out[1]

        slot_offset = n_ctx_blocks * num_frame_per_block
        new_live: List[Tuple[torch.Tensor, int]] = []
        for s_idx in range(n_slots_now):
            lat, idx = live_slots[s_idx]
            new_idx = idx + 1
            f0 = slot_offset + s_idx * num_frame_per_block
            f1 = f0 + num_frame_per_block
            pred_x0 = pred_x0_window[:, f0:f1]
            if new_idx >= total_denoise:
                # Slot 0 committing.
                if s_idx != 0:
                    raise RuntimeError(
                        f"non-slot-0 reached total_denoise (s_idx={s_idx}, new_idx={new_idx})"
                    )
                generated.append(pred_x0.detach().to(torch.float32))
                if len(fifo) >= fifo_size:
                    fifo.pop(0)
                fifo.append(pred_x0.detach().to(dtype))
                next_commit_chunk += 1
                log.info("[staircase] committed chunk %d/%d (cumulative)",
                         len(generated), num_gen_chunks)
            else:
                next_t = ladder[new_idx]
                flat = pred_x0.flatten(0, 1).float()
                flat_noise = torch.randn_like(flat)
                flat_t = torch.full(
                    (flat.shape[0],), next_t, device=device, dtype=torch.float32,
                )
                new_lat = (
                    scheduler.add_noise(flat, flat_noise, flat_t)
                    .view(B, num_frame_per_block, C, H, W)
                    .to(dtype)
                )
                new_live.append((new_lat, new_idx))
        live_slots = new_live

    def _add_fresh_slot():
        nonlocal live_slots
        noise = torch.randn(
            [B, num_frame_per_block, C, H, W], dtype=torch.float32, device=device,
        )
        live_slots.append((noise.to(dtype), 0))

    try:
        # Each rolling step: add at most ONE fresh slot (so warmup
        # staggers ladder indices: step 1 starts with [0], step 2 with
        # [P, 0], etc.) Then run P forward passes — slot 0 commits on
        # the last pass once the ladder fills up.
        while len(generated) < num_gen_chunks:
            if len(live_slots) < NS:
                _add_fresh_slot()
            for _ in range(P):
                _do_forward()
                if len(generated) >= num_gen_chunks:
                    break
    finally:
        base_dit.local_attn_size = prev_local_attn_size
        for _, module in base_dit.named_modules():
            if hasattr(module, "local_attn_size"):
                try: module.local_attn_size = prev_local_attn_size
                except Exception: pass
        base_dit.block_mask = None

    if len(generated) < num_gen_chunks:
        log.warning("only generated %d of %d chunks", len(generated), num_gen_chunks)

    seed_real = initial_latents_dev[:, :seed_frames].to(torch.float32)
    return torch.cat([seed_real] + generated[:num_gen_chunks], dim=1)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--student_ckpt", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--rank_zarr", required=True)
    p.add_argument("--rank_offset", type=int, default=0)
    p.add_argument("--encoded_root", required=True)
    p.add_argument("--caption_root", required=True)
    p.add_argument("--motion_root", required=True)
    p.add_argument("--ss_vae_checkpoint", required=True)
    p.add_argument("--ar_initial_chunks", type=int, default=1)
    p.add_argument("--ar_gen_chunks", type=int, default=30)
    p.add_argument("--fifo_size", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--video_fps", type=int, default=20)
    p.add_argument("--configs", nargs="+",
                   default=["2_causal", "2_bidirectional", "4_causal", "4_bidirectional"],
                   help="Names like NS_mask, e.g. 2_causal, 4_bidirectional.")
    p.add_argument("--filter_2", type=float, nargs=2, default=None,
                   help="Override 2-slot action filter (length 2).")
    p.add_argument("--filter_4", type=float, nargs=4, default=None,
                   help="Override 4-slot action filter (length 4).")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    out_root = Path(args.output_dir); out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    torch.manual_seed(int(args.seed)); torch.cuda.manual_seed_all(int(args.seed))
    pipe = ODEARRefreshPipeline(device, dtype=dtype)
    pipe.build(args.config, use_action_tokens=True)
    pipe.load_checkpoint(args.student_ckpt)
    pipe.set_denoising_steps(4)  # staircase always uses 4-rung ladder

    npb = BASE_CHUNK_FRAMES
    seed_frames = args.ar_initial_chunks * npb
    total_frames = seed_frames + args.ar_gen_chunks * npb

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)
    action_dims = list(cfg.get("action_dims", [2, 7]))

    initial_latents, prompt_embeds, noisy_fa_full, _ = load_per_rank_ride_ar(
        zarr_basename=args.rank_zarr, latent_start_offset=int(args.rank_offset),
        total_frames=total_frames, manifest_path=None,
        encoded_root=args.encoded_root, caption_root=args.caption_root,
        motion_root=args.motion_root, ss_vae_checkpoint=args.ss_vae_checkpoint,
        action_dims=action_dims, device=device,
    )
    initial_latents_dev = initial_latents[:, :seed_frames].to(device=device, dtype=dtype)
    prompt_embeds_dev = prompt_embeds.to(device=device, dtype=dtype)
    noisy_fa_full_dev = noisy_fa_full.to(device=device, dtype=dtype)
    gt_latents = initial_latents[:, :total_frames].to("cpu", dtype=torch.float32).clone()

    FILTERS = {
        2: list(args.filter_2) if args.filter_2 is not None else [1.0, 0.5],
        4: list(args.filter_4) if args.filter_4 is not None else [1.0, 0.75, 0.5, 0.25],
    }

    gt_mp4_written = False
    summary_rows = []
    for cfg_name in args.configs:
        try:
            ns_str, mask_mode = cfg_name.split("_", 1)
            ns = int(ns_str)
        except Exception:
            log.error("config name %s invalid; expected NS_mask", cfg_name); continue
        if ns not in FILTERS:
            log.error("config %s: ns=%d not in %s, skipping", cfg_name, ns, list(FILTERS)); continue
        if mask_mode not in ("causal", "bidirectional"):
            log.error("config %s: mask=%s invalid", cfg_name, mask_mode); continue

        cfg_out = out_root / cfg_name; cfg_out.mkdir(parents=True, exist_ok=True)
        log.info("\n========== config %s (NS=%d filter=%s mask=%s) ==========",
                 cfg_name, ns, FILTERS[ns], mask_mode)

        noise_seed = int(args.seed) + 1_000_003
        torch.manual_seed(noise_seed); torch.cuda.manual_seed(noise_seed)

        try:
            t0 = time.time()
            lat = run_ar_refresh_staircase(
                pipe,
                prompt_embeds_dev=prompt_embeds_dev,
                noisy_fa_full=noisy_fa_full_dev,
                initial_latents_dev=initial_latents_dev,
                num_gen_chunks=int(args.ar_gen_chunks),
                fifo_size=int(args.fifo_size),
                num_frame_per_block=npb,
                device=device, dtype=dtype,
                num_slots=ns, action_filter=FILTERS[ns], mask_mode=mask_mode,
            )
            wall = time.time() - t0
            lat_cpu = lat.to("cpu", dtype=torch.float32).clone()
            log.info("[%s] latents=%s wall=%.1fs", cfg_name, tuple(lat_cpu.shape), wall)

            torch.save({
                "meta": {
                    "config": cfg_name, "num_slots": ns,
                    "action_filter": FILTERS[ns], "mask_mode": mask_mode,
                    "ar_gen_chunks": args.ar_gen_chunks, "fifo_size": args.fifo_size,
                    "seed": args.seed,
                    "rank_zarr": args.rank_zarr, "rank_offset": args.rank_offset,
                    "shape": list(lat_cpu.shape), "wall_s": wall,
                    "scheduler": "staircase",
                    "passes_per_step": 4 // ns,
                    "ladder": [round(float(x), 2) for x in
                               torch.linspace(1000.0, 50.0, 4).tolist()],
                },
                "gt": gt_latents,
                "ar_refresh_multislot": lat_cpu,
            }, cfg_out / "latents.pt")

            lat_dev = lat_cpu.to(device=device, dtype=dtype)
            video_np = pipe.decode_latents(lat_dev)
            frames_to_mp4(video_np, str(cfg_out / "ar_refresh.mp4"), fps=args.video_fps)
            log.info("[%s] wrote ar_refresh.mp4", cfg_name)
            if not gt_mp4_written:
                gt_dev = gt_latents.to(device=device, dtype=dtype)
                gt_video = pipe.decode_latents(gt_dev)
                frames_to_mp4(gt_video, str(out_root / "gt.mp4"), fps=args.video_fps)
                gt_mp4_written = True
                log.info("wrote shared gt.mp4")
            del lat, lat_cpu, lat_dev
            torch.cuda.empty_cache()
            with (cfg_out / "manifest.json").open("w") as fh:
                json.dump({
                    "config": cfg_name, "num_slots": ns,
                    "action_filter": FILTERS[ns], "mask_mode": mask_mode,
                    "wall_s": wall, "status": "ok",
                }, fh, indent=2)
            summary_rows.append({"config": cfg_name, "status": "ok", "wall_s": wall})
        except Exception as e:
            tb = traceback.format_exc()
            log.error("[%s] CRASHED: %s", cfg_name, e)
            log.error(tb)
            with (cfg_out / "manifest.json").open("w") as fh:
                json.dump({"config": cfg_name, "status": "error",
                           "error": str(e), "traceback": tb}, fh, indent=2)
            summary_rows.append({"config": cfg_name, "status": "error", "error": str(e)})
            torch.cuda.empty_cache()

    with (out_root / "summary.json").open("w") as fh:
        json.dump({"configs": summary_rows}, fh, indent=2)
    log.info("=== Summary ===")
    for r in summary_rows:
        log.info("  %s", r)


if __name__ == "__main__":
    main()
