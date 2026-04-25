#!/usr/bin/env python3
"""AR_refresh with a write-once K/V cache for ctx chunks.

Algorithm:
  * AR_refresh joint-window forward as usual (4 denoise passes per
    rolling step on the entire window).
  * A per-chunk, per-layer K/V cache is maintained.
  * On every forward, for token positions belonging to a ctx chunk:
      - If the chunk's K/V is already in the cache → replay (override
        ``self.k``/``self.v`` output at those positions with cached
        values).
      - If not in the cache → compute fresh and **store** for future
        use.
  * Current-chunk positions: always compute fresh K/V (never read from
    cache, never overwritten).
  * Cache entries are never updated after being written. Eviction
    follows the FIFO so once a chunk falls out of the window its
    entry is removed.

Difference from the earlier ``appendlike`` script: there is no
extra commit-refresh forward. The cache is populated entirely from
AR_refresh's natural forward passes — specifically, the very first
forward in which the chunk appears as ctx (which is the next step's
pass 0 after that chunk committed). After that the entry is frozen.

This is the closest thing to append_baseline's "K/V committed once,
never refreshed" semantics, expressed inside the AR_refresh path
where ctx and current chunks share the same joint-window forward.

Outputs latents.pt + ar_refresh_writeonce.mp4 + gt.mp4.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

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


class _State:
    """chunk_position_map  — list of (chunk_id, token_lo, token_hi) for the current forward.
    current_chunk_id     — the live chunk; never read from or write into the cache.
    cache[id][layer]     — {"k": tensor, "v": tensor} captured the first time `id` appears as ctx.
    """
    def __init__(self):
        self.chunk_position_map: List[Tuple[int, int, int]] = []
        self.current_chunk_id: Optional[int] = None
        self.cache: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}

    def evict(self, chunk_id: int):
        self.cache.pop(chunk_id, None)


class WriteOnceWrapper(nn.Module):
    def __init__(self, inner: nn.Linear, state: _State, layer_idx: int, kind: str):
        super().__init__()
        self.inner = inner
        self.state = state
        self.layer_idx = layer_idx
        self.kind = kind  # "k" or "v"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.inner(x)
        if not self.state.chunk_position_map:
            return out

        # Lazily clone if we need to override at least one position.
        cloned = False

        for chunk_id, lo, hi in self.state.chunk_position_map:
            # Current chunk: always fresh, no cache touch.
            if chunk_id == self.state.current_chunk_id:
                continue
            chunk_layer = self.state.cache.setdefault(chunk_id, {}).setdefault(self.layer_idx, {})
            cached = chunk_layer.get(self.kind)
            if cached is not None and cached.shape == out[:, lo:hi, :].shape:
                if not cloned:
                    out = out.clone(); cloned = True
                out[:, lo:hi, :] = cached.to(out.dtype)
            else:
                # First time this chunk's K/V is being computed → store it.
                chunk_layer[self.kind] = out[:, lo:hi, :].detach().clone()
        return out


def install_writeonce_hooks(base_dit, state: _State):
    layer_idx = 0
    wrappers: List[WriteOnceWrapper] = []
    for name, module in base_dit.named_modules():
        if hasattr(module, "self_attn"):
            sa = module.self_attn
            if hasattr(sa, "k") and isinstance(sa.k, nn.Linear):
                wk = WriteOnceWrapper(sa.k, state, layer_idx, "k"); sa.k = wk; wrappers.append(wk)
            if hasattr(sa, "v") and isinstance(sa.v, nn.Linear):
                wv = WriteOnceWrapper(sa.v, state, layer_idx, "v"); sa.v = wv; wrappers.append(wv)
            layer_idx += 1
    log.info("writeonce hooks installed on %d projections (k+v) over %d layers",
             len(wrappers), layer_idx)
    return wrappers


def restore_hooks(base_dit, wrappers: List[WriteOnceWrapper]):
    for name, module in base_dit.named_modules():
        if hasattr(module, "self_attn"):
            sa = module.self_attn
            if hasattr(sa, "k") and isinstance(sa.k, WriteOnceWrapper):
                sa.k = sa.k.inner
            if hasattr(sa, "v") and isinstance(sa.v, WriteOnceWrapper):
                sa.v = sa.v.inner


@torch.no_grad()
def run_ar_refresh_writeonce(
    pipe, *,
    prompt_embeds_dev, noisy_fa_full, initial_latents_dev,
    num_gen_chunks, fifo_size, num_frame_per_block, device, dtype,
) -> torch.Tensor:
    base_dit = pipe.wrapper.model
    if hasattr(base_dit, "get_base_model"):
        try: base_dit = base_dit.get_base_model()
        except Exception: pass

    B = 1
    seed_frames = int(initial_latents_dev.shape[1])
    seed_chunks = seed_frames // num_frame_per_block
    max_window_blocks = fifo_size + 1
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
    ts = pipe.denoising_step_list

    state = _State()
    wrappers = install_writeonce_hooks(base_dit, state)

    C = int(initial_latents_dev.shape[2])
    H = int(initial_latents_dev.shape[3])
    W = int(initial_latents_dev.shape[4])

    tokens_per_frame = frame_seq_length
    tokens_per_chunk = num_frame_per_block * tokens_per_frame

    fifo: List[Tuple[int, torch.Tensor]] = [(0, initial_latents_dev)]
    next_chunk_id = 1
    current_window_blocks = -1
    generated: List[torch.Tensor] = []

    log.info("AR_refresh writeonce-cache rollout: seed=%d  gen=%d  fifo=%d",
             seed_frames, num_gen_chunks, fifo_size)

    def _build_position_map(ctx_chunks: List[Tuple[int, torch.Tensor]],
                            current_chunk_id: int) -> List[Tuple[int, int, int]]:
        out = []
        pos = 0
        for cid, _ in ctx_chunks:
            out.append((cid, pos, pos + tokens_per_chunk))
            pos += tokens_per_chunk
        out.append((current_chunk_id, pos, pos + tokens_per_chunk))
        return out

    try:
        for chunk_idx in range(int(num_gen_chunks)):
            n_ctx_blocks = min(len(fifo), fifo_size)
            window_blocks = n_ctx_blocks + 1
            window_frames = window_blocks * num_frame_per_block

            cur_global_lo = (seed_chunks + chunk_idx) * num_frame_per_block
            cur_global_hi = cur_global_lo + num_frame_per_block
            ctx_global_lo = cur_global_lo - n_ctx_blocks * num_frame_per_block

            if window_blocks != current_window_blocks:
                base_dit.block_mask = None
                current_window_blocks = window_blocks

            ctx_entries = fifo[-n_ctx_blocks:]
            ctx_cat = torch.cat([t for _, t in ctx_entries], dim=1)
            fa_window = noisy_fa_full[:, ctx_global_lo:cur_global_hi].contiguous()
            t_ctx_vec = torch.zeros(
                [B, n_ctx_blocks * num_frame_per_block], device=device, dtype=torch.float32,
            )

            current_chunk_id = next_chunk_id
            next_chunk_id += 1
            pos_map = _build_position_map(ctx_entries, current_chunk_id)

            current_noise = torch.randn(
                [B, num_frame_per_block, C, H, W], dtype=torch.float32, device=device,
            )
            x_cur = current_noise.to(dtype)
            pred_x0_window: Optional[torch.Tensor] = None

            for d_idx in range(int(ts.shape[0])):
                state.chunk_position_map = pos_map
                state.current_chunk_id = current_chunk_id

                t_val = float(ts[d_idx].item())
                t_cur_vec = torch.full(
                    [B, num_frame_per_block], t_val, device=device, dtype=torch.float32,
                )
                tt = torch.cat([t_ctx_vec, t_cur_vec], dim=1)
                x_full = torch.cat([ctx_cat, x_cur], dim=1)
                cond = pipe._build_action_cond_chunk(
                    prompt_embeds_dev, fa_window, num_frames=window_frames,
                )
                with torch.amp.autocast("cuda", dtype=dtype):
                    out = pipe.wrapper(
                        noisy_image_or_video=x_full,
                        conditional_dict=cond,
                        timestep=tt,
                        clean_x=None, aug_t=None,
                    )
                pred_x0_window = out[1]
                if d_idx < int(ts.shape[0]) - 1:
                    next_t = float(ts[d_idx + 1].item())
                    cur_pred_x0 = pred_x0_window[:, n_ctx_blocks * num_frame_per_block:]
                    flat = cur_pred_x0.flatten(0, 1).float()
                    flat_noise = torch.randn_like(flat)
                    flat_t = torch.full(
                        (flat.shape[0],), next_t, device=device, dtype=torch.float32,
                    )
                    x_cur = (
                        scheduler.add_noise(flat, flat_noise, flat_t)
                        .view(B, num_frame_per_block, C, H, W)
                        .to(dtype)
                    )

            assert pred_x0_window is not None
            cur_pred = pred_x0_window[:, n_ctx_blocks * num_frame_per_block:]
            generated.append(cur_pred.detach().to(torch.float32))

            if len(fifo) >= fifo_size:
                evicted_id, _ = fifo.pop(0)
                state.evict(evicted_id)
            fifo.append((current_chunk_id, cur_pred.to(dtype)))

            log.info("[writeonce] chunk %d/%d committed (id=%d) | cached chunks=%s",
                     chunk_idx + 1, num_gen_chunks, current_chunk_id,
                     sorted(state.cache.keys()))
    finally:
        restore_hooks(base_dit, wrappers)
        base_dit.local_attn_size = prev_local_attn_size
        for _, module in base_dit.named_modules():
            if hasattr(module, "local_attn_size"):
                try: module.local_attn_size = prev_local_attn_size
                except Exception: pass
        base_dit.block_mask = None

    seed_real = initial_latents_dev[:, :seed_frames].to(torch.float32)
    return torch.cat([seed_real] + generated, dim=1)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
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
    p.add_argument("--denoising_steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--video_fps", type=int, default=20)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    torch.manual_seed(int(args.seed)); torch.cuda.manual_seed_all(int(args.seed))
    pipe = ODEARRefreshPipeline(device, dtype=dtype)
    pipe.build(args.config, use_action_tokens=True)
    pipe.load_checkpoint(args.student_ckpt)
    pipe.set_denoising_steps(int(args.denoising_steps))

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

    noise_seed = int(args.seed) + 1_000_003
    torch.manual_seed(noise_seed); torch.cuda.manual_seed(noise_seed)

    t0 = time.time()
    lat = run_ar_refresh_writeonce(
        pipe,
        prompt_embeds_dev=prompt_embeds_dev,
        noisy_fa_full=noisy_fa_full_dev,
        initial_latents_dev=initial_latents_dev,
        num_gen_chunks=int(args.ar_gen_chunks),
        fifo_size=int(args.fifo_size),
        num_frame_per_block=npb,
        device=device, dtype=dtype,
    )
    wall = time.time() - t0
    lat_cpu = lat.to("cpu", dtype=torch.float32).clone()
    log.info("ar_refresh_writeonce latents=%s wall=%.1fs", tuple(lat_cpu.shape), wall)

    torch.save({
        "meta": {
            "student_ckpt": args.student_ckpt,
            "rank_zarr": args.rank_zarr, "rank_offset": args.rank_offset,
            "seed": args.seed, "ar_gen_chunks": args.ar_gen_chunks,
            "fifo_size": args.fifo_size, "denoising_steps": args.denoising_steps,
            "shape": list(lat_cpu.shape), "wall_s": wall,
            "variant": "ar_refresh_writeonce",
            "note": "AR_refresh + write-once K/V cache for ctx chunks; entries captured on first appearance and never updated; current chunk K/V always fresh; no extra commit-refresh forward",
        },
        "gt": gt_latents,
        "ar_refresh_writeonce": lat_cpu,
    }, out / "latents.pt")
    log.info("saved %s", out / "latents.pt")

    lat_dev = lat_cpu.to(device=device, dtype=dtype)
    video_np = pipe.decode_latents(lat_dev)
    frames_to_mp4(video_np, str(out / "ar_refresh_writeonce.mp4"), fps=args.video_fps)
    log.info("wrote %s", out / "ar_refresh_writeonce.mp4")

    gt_dev = gt_latents.to(device=device, dtype=dtype)
    gt_video = pipe.decode_latents(gt_dev)
    frames_to_mp4(gt_video, str(out / "gt.mp4"), fps=args.video_fps)
    log.info("wrote %s", out / "gt.mp4")

    with (out / "manifest.json").open("w") as fh:
        json.dump({"variant": "ar_refresh_writeonce", "wall_s": wall, "status": "ok"}, fh, indent=2)


if __name__ == "__main__":
    main()
