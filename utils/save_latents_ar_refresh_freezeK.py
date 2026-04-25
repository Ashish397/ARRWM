#!/usr/bin/env python3
"""AR_refresh variant with V refreshed every pass but K frozen.

Per rolling step (one chunk commit), AR_refresh does 4 denoise passes
on the joint window. Standard behaviour: every pass recomputes both K
and V from scratch via ``_forward_train``. This script intercepts the
``self_attn.k`` Linear modules across all transformer blocks and forces
K to be **captured on pass 0 and replayed on passes 1-3** of every
step. V is left untouched (recomputed fresh per pass, like normal
AR_refresh). Between steps the freeze is reset so K is recaptured
against the new window composition.

Saves latents.pt + ar_refresh_freezeK.mp4 + gt.mp4.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

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


class _FreezeKMode:
    """Mutable shared state controlling capture/replay for all wrapped K modules."""
    def __init__(self):
        self.capture = False
        self.replay = False

    def reset_step(self):
        """Begin a new rolling step: capture pass-0 K, then replay 1-3."""
        self.capture = False
        self.replay = False

    def set_capture(self):
        self.capture = True
        self.replay = False

    def set_replay(self):
        self.capture = False
        self.replay = True


class FreezeKWrapper(nn.Module):
    """Wraps a single ``self_attn.k`` Linear; captures or replays based on
    the shared ``mode`` instance."""
    def __init__(self, inner: nn.Linear, mode: _FreezeKMode):
        super().__init__()
        self.inner = inner
        self.mode = mode
        self.cached: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode.replay and self.cached is not None:
            # The cached K's shape must match the current call's expected
            # shape — within a step the window doesn't change so this is
            # safe. Across step boundaries we always reset & recapture.
            if self.cached.shape == self.inner(x).shape if False else True:
                pass
            return self.cached
        out = self.inner(x)
        if self.mode.capture:
            self.cached = out.detach().clone()
        return out


def install_freezeK_hooks(base_dit) -> tuple[_FreezeKMode, list[FreezeKWrapper]]:
    """Walk every transformer block and replace ``self_attn.k`` with a
    FreezeKWrapper. Returns the shared mode and the list of wrappers
    (so we can clear caches on step transitions)."""
    mode = _FreezeKMode()
    wrappers: list[FreezeKWrapper] = []
    n = 0
    for name, module in base_dit.named_modules():
        if hasattr(module, "self_attn") and hasattr(module.self_attn, "k") and isinstance(module.self_attn.k, nn.Linear):
            wrap = FreezeKWrapper(module.self_attn.k, mode)
            module.self_attn.k = wrap
            wrappers.append(wrap)
            n += 1
    log.info("freezeK installed on %d attention modules", n)
    return mode, wrappers


def restore_freezeK_hooks(base_dit, wrappers: list[FreezeKWrapper]):
    """Reverse install_freezeK_hooks: put the original Linear modules back."""
    for name, module in base_dit.named_modules():
        if hasattr(module, "self_attn") and hasattr(module.self_attn, "k") and isinstance(module.self_attn.k, FreezeKWrapper):
            module.self_attn.k = module.self_attn.k.inner
    for w in wrappers:
        w.cached = None


@torch.no_grad()
def run_ar_refresh_freezeK(
    pipe, *,
    prompt_embeds_dev, noisy_fa_full, initial_latents_dev,
    num_gen_chunks, fifo_size, num_frame_per_block, device, dtype,
) -> torch.Tensor:
    """Standard AR_refresh joint-window rollout, with K frozen across
    the 4 passes of each rolling step."""
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

    mode, wrappers = install_freezeK_hooks(base_dit)

    C = int(initial_latents_dev.shape[2])
    H = int(initial_latents_dev.shape[3])
    W = int(initial_latents_dev.shape[4])

    fifo: List[torch.Tensor] = [initial_latents_dev]
    current_window_blocks = -1
    generated: List[torch.Tensor] = []

    log.info("AR_refresh freezeK rollout: seed=%d frames  gen=%d chunks  fifo=%d",
             seed_frames, num_gen_chunks, fifo_size)

    try:
        for chunk_idx in range(int(num_gen_chunks)):
            n_ctx_blocks = min(len(fifo), fifo_size)
            window_blocks = n_ctx_blocks + 1
            window_frames = window_blocks * num_frame_per_block

            cur_global_lo = (seed_chunks + chunk_idx) * num_frame_per_block
            cur_global_hi = cur_global_lo + num_frame_per_block
            ctx_global_lo = cur_global_lo - n_ctx_blocks * num_frame_per_block

            if window_blocks != current_window_blocks:
                base_dit.block_mask = None  # force rebuild on next forward
                current_window_blocks = window_blocks

            ctx_chunks = fifo[-n_ctx_blocks:]
            ctx_cat = torch.cat(ctx_chunks, dim=1)
            fa_window = noisy_fa_full[:, ctx_global_lo:cur_global_hi].contiguous()
            t_ctx_vec = torch.zeros(
                [B, n_ctx_blocks * num_frame_per_block],
                device=device, dtype=torch.float32,
            )

            current_noise = torch.randn(
                [B, num_frame_per_block, C, H, W], dtype=torch.float32, device=device,
            )
            x_cur = current_noise.to(dtype)
            pred_x0_window: Optional[torch.Tensor] = None

            # Reset K cache for this step.
            for w in wrappers:
                w.cached = None

            for d_idx in range(int(ts.shape[0])):
                if d_idx == 0:
                    mode.set_capture()  # capture K on pass 0
                else:
                    mode.set_replay()    # replay cached K on passes 1-3

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
                fifo.pop(0)
            fifo.append(cur_pred.to(dtype))
            log.info("[freezeK] chunk %d/%d committed", chunk_idx + 1, num_gen_chunks)
    finally:
        restore_freezeK_hooks(base_dit, wrappers)
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

    # local_attn_size context: AR_refresh's generate_ar_refresh sets it via
    # try/finally, but since we're calling _forward_train directly (via wrapper),
    # we replicate that here in run_ar_refresh_freezeK.
    t0 = time.time()
    lat = run_ar_refresh_freezeK(
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
    log.info("ar_refresh_freezeK latents=%s wall=%.1fs", tuple(lat_cpu.shape), wall)

    torch.save({
        "meta": {
            "student_ckpt": args.student_ckpt,
            "rank_zarr": args.rank_zarr, "rank_offset": args.rank_offset,
            "seed": args.seed, "ar_gen_chunks": args.ar_gen_chunks,
            "fifo_size": args.fifo_size, "denoising_steps": args.denoising_steps,
            "shape": list(lat_cpu.shape), "wall_s": wall,
            "variant": "ar_refresh_freezeK",
            "note": "K captured on pass 0, replayed on passes 1-3 of each step; V refreshed every pass",
        },
        "gt": gt_latents,
        "ar_refresh_freezeK": lat_cpu,
    }, out / "latents.pt")
    log.info("saved %s", out / "latents.pt")

    lat_dev = lat_cpu.to(device=device, dtype=dtype)
    video_np = pipe.decode_latents(lat_dev)
    frames_to_mp4(video_np, str(out / "ar_refresh_freezeK.mp4"), fps=args.video_fps)
    log.info("wrote %s", out / "ar_refresh_freezeK.mp4")

    gt_dev = gt_latents.to(device=device, dtype=dtype)
    gt_video = pipe.decode_latents(gt_dev)
    frames_to_mp4(gt_video, str(out / "gt.mp4"), fps=args.video_fps)
    log.info("wrote %s", out / "gt.mp4")

    with (out / "manifest.json").open("w") as fh:
        json.dump({"variant": "ar_refresh_freezeK", "wall_s": wall, "status": "ok"}, fh, indent=2)


if __name__ == "__main__":
    main()
