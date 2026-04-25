#!/usr/bin/env python3
"""Run AR_refresh and capture per-layer K/V tensors at every chunk so we
can visualise how the student's recomputed context evolves.

Hooks every ``CausalWanSelfAttention`` block's ``norm_k`` and ``v`` (the
post-projection, pre-RoPE K and V) and grabs the tensors on the LAST
denoise pass of each chunk (``t == ts[-1]``). At the end of the rollout
writes two artefacts into the output directory:

  1. ``kv_capture.pt`` — a torch-serialised dict::

         {
           "meta": {
             "ts": [...],
             "num_frame_per_block": 3,
             "num_blocks": 30,
             "head_dim": 128,
             "num_heads": 12,
             "chunk_shapes": [(window_frames, ...), ...],
           },
           "chunks": [
             {
               "chunk_idx": int,
               "window_blocks": int,
               "n_ctx_blocks": int,
               "cur_global_frame_lo": int,
               "K": [B, F_tot, n_heads, head_dim]  fp16 CPU,
               "V": [B, F_tot, n_heads, head_dim]  fp16 CPU,
               # only layer 0 by default to keep size reasonable; see
               # ``--capture_layers``.
             },
             ...
           ],
         }

  2. ``kv_summary.png`` — a 2-panel heatmap: y-axis = captured layer,
     x-axis = frame index within the (n_ctx+1)*3 window, cell = mean
     absolute K (left) / V (right) for that chunk's final denoise.
     Multiple chunks stacked as sub-plot rows.

Usage::

    conda activate flash
    CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 LOCAL_RANK=0 \\
        python utils/analyse_AR_refresh_kv.py \\
            --config configs/action_ode_distill_local.yaml \\
            --student_ckpt /home/ashish/action_ode_step0001000.pt \\
            --rank_zarr 20240408152948.zarr \\
            --rank_offset 0 \\
            --encoded_root /home/ashish/frodobots/frodobots_encoded \\
            --caption_root /home/ashish/frodobots/frodobots_captions/train \\
            --motion_root /home/ashish/frodobots/frodobots_motion \\
            --ss_vae_checkpoint action_query/checkpoints/ss_vae_8free.pt \\
            --ar_initial_chunks 1 --ar_gen_chunks 7 --fifo_size 3 \\
            --denoising_steps 4 --seed 42 \\
            --output_dir /home/ashish/ARRWM/eval/kv_capture_$(date +%Y%m%d_%H%M%S)

``--capture_layers``: comma-separated transformer-block indices to
capture (default ``0,14,29`` — first, middle, last). ``all`` captures
every layer (large file).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_AF_ROOT = _REPO_ROOT / "action-forcing"
if str(_AF_ROOT) not in sys.path:
    sys.path.insert(0, str(_AF_ROOT))

from utils.eval_causal_AR import load_per_rank_ride_ar
from utils.eval_causal_AR_chain import ODEARRefreshPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# K/V capture — monkey-patch CausalWanSelfAttention.forward to stash the
# post-projection, post-norm, pre-RoPE q/k/v on the module instance so
# the outer loop can read them without refactoring the attention code.
# ---------------------------------------------------------------------------


def _install_kv_capture(base_dit, capture_layer_indices: List[int]) -> List[torch.nn.Module]:
    """Find every CausalWanAttentionBlock's self_attn; attach a
    capture-enabled flag; wrap ``qkv_fn``'s outputs via a forward hook on
    the ``v`` Linear (which is called last inside ``qkv_fn``) + the ``k``
    Linear (called second). Post-hook stashes the Linear outputs, and
    after each forward we pull them off and reshape.

    Simpler path: register forward hooks on the ``k`` and ``v`` Linear
    submodules. They return the pre-norm, pre-reshape projections; we
    store them and the outer loop can reshape to [B, F, n_heads,
    head_dim] given the DiT's num_heads.

    Returns the list of wrapped self_attn modules (indexed by
    transformer-block index) so the caller knows which block each
    captured tensor belongs to.
    """
    attn_blocks: List[torch.nn.Module] = []
    for _, module in base_dit.named_modules():
        if module.__class__.__name__ == "CausalWanSelfAttention":
            attn_blocks.append(module)

    # Tag which blocks we care about.
    for i, blk in enumerate(attn_blocks):
        blk._capture_enabled = i in capture_layer_indices
        blk._captured_k = None
        blk._captured_v = None

    def _make_k_hook(blk):
        def hook(module, inp, out):
            if getattr(blk, "_capture_enabled", False):
                blk._captured_k = out.detach()
        return hook

    def _make_v_hook(blk):
        def hook(module, inp, out):
            if getattr(blk, "_capture_enabled", False):
                blk._captured_v = out.detach()
        return hook

    for blk in attn_blocks:
        blk.k.register_forward_hook(_make_k_hook(blk))
        blk.v.register_forward_hook(_make_v_hook(blk))

    return attn_blocks


# ---------------------------------------------------------------------------
# Orchestration — build pipeline, patch the AR-refresh loop to snapshot
# K/V after the FINAL denoise pass of each chunk.
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--student_ckpt", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--rank_zarr", type=str, required=True)
    p.add_argument("--rank_offset", type=int, default=0)
    p.add_argument("--encoded_root", type=str, required=True)
    p.add_argument("--caption_root", type=str, required=True)
    p.add_argument("--motion_root", type=str, required=True)
    p.add_argument("--ss_vae_checkpoint", type=str, required=True)
    p.add_argument("--ar_initial_chunks", type=int, default=1)
    p.add_argument("--ar_gen_chunks", type=int, default=7)
    p.add_argument("--fifo_size", type=int, default=3)
    p.add_argument("--denoising_steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--capture_layers", type=str, default="0,14,29",
                   help="Comma-separated block indices or 'all'.")
    p.add_argument("--capture_dtype", type=str, default="float16",
                   choices=["float16", "bfloat16", "float32"],
                   help="Dtype for saved K/V tensors (fp16 keeps the .pt "
                        "reasonable).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    capture_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.capture_dtype]

    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))

    # --- Build the AR_refresh pipeline (same as eval_causal_AR_chain.py). ---
    pipe = ODEARRefreshPipeline(device, dtype=dtype)
    pipe.build(args.config, use_action_tokens=True)
    pipe.load_checkpoint(args.student_ckpt)
    pipe.set_denoising_steps(int(args.denoising_steps))

    # --- Ride loader. ---
    num_frame_per_block = 3
    seed_frames = args.ar_initial_chunks * num_frame_per_block
    total_frames = seed_frames + args.ar_gen_chunks * num_frame_per_block
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)
    action_dims = list(cfg.get("action_dims", [2, 7]))

    initial_latents, prompt_embeds, noisy_fa_full, meta = load_per_rank_ride_ar(
        zarr_basename=args.rank_zarr,
        latent_start_offset=int(args.rank_offset),
        total_frames=total_frames,
        manifest_path=None,
        encoded_root=args.encoded_root,
        caption_root=args.caption_root,
        motion_root=args.motion_root,
        ss_vae_checkpoint=args.ss_vae_checkpoint,
        action_dims=action_dims,
        device=device,
    )
    # Slice seed_frames worth of prefill for AR_refresh's seed. The chain
    # pipeline expects exactly 1 chunk of seed by default.
    initial_latents = initial_latents[:, :seed_frames].to(dtype=dtype)
    prompt_embeds = prompt_embeds.to(dtype=dtype)
    noisy_fa_full = noisy_fa_full.to(dtype=dtype)

    # --- Resolve capture layers. ---
    base_dit = pipe.wrapper.model
    if hasattr(base_dit, "get_base_model"):
        try:
            base_dit = base_dit.get_base_model()
        except Exception:
            pass
    total_blocks = len(base_dit.blocks)
    if args.capture_layers.strip().lower() == "all":
        capture_layer_indices = list(range(total_blocks))
    else:
        capture_layer_indices = [int(x) for x in args.capture_layers.split(",") if x.strip()]
        capture_layer_indices = [i for i in capture_layer_indices if 0 <= i < total_blocks]
    log.info("Capturing K/V at layers %s (of %d total)", capture_layer_indices, total_blocks)

    attn_blocks = _install_kv_capture(base_dit, capture_layer_indices)

    # --- Monkey-patch the chain pipeline's _run_ar_refresh to snapshot
    #     the layer-level captures after the FINAL denoise pass of each
    #     chunk. We keep the original logic intact and just wrap the
    #     method to wire the capture glue in. ---
    import types
    orig_run = pipe._run_ar_refresh
    captures: List[Dict[str, Any]] = []

    def _run_with_capture(
        self,
        *,
        prompt_embeds,
        noisy_fa_full,
        initial_latents,
        num_gen_chunks,
        fifo_size,
        context_noise_timestep,
        base_dit,
        num_frame_per_block,
        frame_seq_length,
        max_window_blocks,
    ):
        # Minimal copy of the original loop so we can observe state
        # AROUND the wrapper calls. We call the original helper for the
        # one-time setup then rebuild the loop.
        B = 1
        scheduler = self.scheduler
        scheduler.sigmas = scheduler.sigmas.to(self.device)
        ts = self.denoising_step_list

        seed_frames_ = int(initial_latents.shape[1])
        if seed_frames_ != num_frame_per_block:
            raise SystemExit(
                f"AR_refresh: expected exactly 1 GT chunk for the seed; "
                f"got {seed_frames_} frames. Use --ar_initial_chunks 1."
            )

        C = int(initial_latents.shape[2])
        H = int(initial_latents.shape[3])
        W = int(initial_latents.shape[4])

        noisy_fa_full_dev = noisy_fa_full.to(device=self.device, dtype=self.dtype)
        prompt_embeds_dev = prompt_embeds.to(device=self.device, dtype=self.dtype)
        initial_latents_dev = initial_latents.to(device=self.device, dtype=self.dtype)

        fifo: List[torch.Tensor] = [initial_latents_dev]
        current_window_blocks = -1
        generated: List[torch.Tensor] = []
        seed_chunks = seed_frames_ // num_frame_per_block

        for chunk_idx in range(num_gen_chunks):
            n_ctx_blocks = min(len(fifo), fifo_size)
            window_blocks = n_ctx_blocks + 1
            window_frames = window_blocks * num_frame_per_block

            cur_global_frame_lo = (seed_chunks + chunk_idx) * num_frame_per_block
            cur_global_frame_hi = cur_global_frame_lo + num_frame_per_block
            ctx_global_frame_lo = cur_global_frame_lo - n_ctx_blocks * num_frame_per_block

            if window_blocks != current_window_blocks:
                base_dit.block_mask = None
                current_window_blocks = window_blocks

            ctx_chunks = fifo[-n_ctx_blocks:]
            ctx_cat = torch.cat(ctx_chunks, dim=1)

            fa_window = noisy_fa_full_dev[:, ctx_global_frame_lo:cur_global_frame_hi].contiguous()
            t_ctx_vec = torch.full(
                [B, n_ctx_blocks * num_frame_per_block],
                float(context_noise_timestep),
                device=self.device, dtype=torch.float32,
            )

            current_noise = torch.randn(
                [B, num_frame_per_block, C, H, W],
                dtype=torch.float32, device=self.device,
            )
            x_cur = current_noise.to(self.dtype)

            pred_x0_window: Optional[torch.Tensor] = None
            for d_idx in range(int(ts.shape[0])):
                t_val = float(ts[d_idx].item())
                t_cur_vec = torch.full(
                    [B, num_frame_per_block], t_val,
                    device=self.device, dtype=torch.float32,
                )
                tt = torch.cat([t_ctx_vec, t_cur_vec], dim=1)
                x_full = torch.cat([ctx_cat, x_cur], dim=1)
                cond = self._build_action_cond_chunk(
                    prompt_embeds_dev, fa_window, num_frames=window_frames,
                )
                with torch.amp.autocast("cuda", dtype=self.dtype):
                    out = self.wrapper(
                        noisy_image_or_video=x_full,
                        conditional_dict=cond,
                        timestep=tt,
                        clean_x=None,
                        aug_t=None,
                    )
                pred_x0_window = out[1]
                if d_idx < int(ts.shape[0]) - 1:
                    next_t = float(ts[d_idx + 1].item())
                    cur_pred_x0 = pred_x0_window[:, n_ctx_blocks * num_frame_per_block:]
                    flat = cur_pred_x0.flatten(0, 1).float()
                    flat_noise = torch.randn_like(flat)
                    flat_t = torch.full(
                        (flat.shape[0],), next_t,
                        device=self.device, dtype=torch.float32,
                    )
                    x_cur = (
                        scheduler.add_noise(flat, flat_noise, flat_t)
                        .view(B, num_frame_per_block, C, H, W)
                        .to(self.dtype)
                    )

            # --- AFTER FINAL DENOISE PASS: snapshot K/V per captured layer ---
            layer_captures: Dict[int, Dict[str, torch.Tensor]] = {}
            for layer_idx, blk in enumerate(attn_blocks):
                if not getattr(blk, "_capture_enabled", False):
                    continue
                k = blk._captured_k
                v = blk._captured_v
                if k is None or v is None:
                    continue
                # Reshape Linear outputs [B, F*spatial_per_frame, dim]
                # → [B, F, spatial_per_frame, n_heads, head_dim] is
                # expensive; instead we just keep as [B, seq, dim] and
                # the viz script slices/aggregates.
                layer_captures[layer_idx] = {
                    "k": k.to(device="cpu", dtype=capture_dtype).clone(),
                    "v": v.to(device="cpu", dtype=capture_dtype).clone(),
                }
                # Clear to save GPU memory (hook will refill next forward).
                blk._captured_k = None
                blk._captured_v = None

            assert pred_x0_window is not None
            cur_pred = pred_x0_window[:, n_ctx_blocks * num_frame_per_block:]
            generated.append(cur_pred.detach().to(torch.float32))

            captures.append({
                "chunk_idx": int(chunk_idx),
                "window_blocks": int(window_blocks),
                "n_ctx_blocks": int(n_ctx_blocks),
                "cur_global_frame_lo": int(cur_global_frame_lo),
                "window_frames": int(window_frames),
                "layers": layer_captures,
            })
            log.info(
                "[AR_refresh/capture] chunk %d/%d committed | "
                "window=%d blocks | fifo_len=%d | captured %d layer(s) K/V",
                chunk_idx + 1, num_gen_chunks, window_blocks,
                len(fifo), len(layer_captures),
            )

            if len(fifo) >= fifo_size:
                fifo.pop(0)
            fifo.append(cur_pred.to(self.dtype))

        # Assemble output latents (same as original).
        seed_real = initial_latents_dev[:, :seed_frames_].to(torch.float32)
        full = torch.cat([seed_real] + generated, dim=1)
        return full

    pipe._run_ar_refresh = types.MethodType(_run_with_capture, pipe)

    # --- Actually run. ---
    t0 = time.time()
    full_latents = pipe.generate_ar_refresh(
        prompt_embeds=prompt_embeds,
        noisy_fa_full=noisy_fa_full,
        initial_latents=initial_latents,
        num_gen_chunks=int(args.ar_gen_chunks),
        fifo_size=int(args.fifo_size),
        context_noise_timestep=0.0,
    )
    t1 = time.time()
    log.info("Rollout + capture done in %.1fs. Captured %d chunks.", t1 - t0, len(captures))

    # --- Save the capture. ---
    out_pt = out_dir / "kv_capture.pt"
    payload = {
        "meta": {
            "ts": pipe.denoising_step_list.detach().cpu().tolist(),
            "num_frame_per_block": num_frame_per_block,
            "num_blocks": total_blocks,
            "capture_layers": capture_layer_indices,
            "fifo_size": int(args.fifo_size),
            "ar_gen_chunks": int(args.ar_gen_chunks),
            "ar_initial_chunks": int(args.ar_initial_chunks),
            "seed": int(args.seed),
            "rank_zarr": args.rank_zarr,
            "rank_offset": int(args.rank_offset),
            "student_ckpt": args.student_ckpt,
            "capture_dtype": str(capture_dtype),
        },
        "chunks": captures,
    }
    torch.save(payload, out_pt)
    log.info("Saved K/V capture -> %s (%.1f MB)",
             out_pt, out_pt.stat().st_size / (1024 * 1024))

    # --- Simple visualisation: per-layer, per-frame mean-absolute K/V for
    #     each chunk. Save one figure per chunk showing K on the left,
    #     V on the right; rows = captured layers, x = frame-in-window. ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available — skipping summary plot.")
        plt = None

    if plt is not None:
        npb = num_frame_per_block
        # Reshape each captured K,V from [B, seq, dim] to per-frame mean.
        # For the attention blocks, seq = frame_seqlen * num_frames where
        # frame_seqlen = 1560 + action_tokens_per_frame (= 1561).
        fig, axes = plt.subplots(
            nrows=len(captures), ncols=2,
            figsize=(10, max(3, 1.6 * len(captures))),
            sharex=False, sharey=True, squeeze=False,
        )
        for row, chunk in enumerate(captures):
            for col, key in enumerate(("k", "v")):
                ax = axes[row][col]
                n_ctx = chunk["n_ctx_blocks"]
                wf = chunk["window_frames"]
                # Compute per-frame magnitude per captured layer.
                layer_rows = []
                layer_labels = []
                for layer_idx in sorted(chunk["layers"].keys()):
                    t = chunk["layers"][layer_idx][key]  # [B, seq, dim]
                    seq = t.shape[1]
                    frame_seqlen = seq // wf
                    # Reshape [B, wf, frame_seqlen, dim] → mean across
                    # spatial+action tokens and across dim.
                    per_frame = t.float().view(
                        t.shape[0], wf, frame_seqlen, t.shape[2],
                    ).abs().mean(dim=(2, 3)).squeeze(0).numpy()  # [wf]
                    layer_rows.append(per_frame)
                    layer_labels.append(f"L{layer_idx}")
                grid = np.stack(layer_rows, axis=0) if layer_rows else np.zeros((1, wf))
                im = ax.imshow(grid, aspect="auto", cmap="viridis")
                ax.set_title(
                    f"chunk {chunk['chunk_idx']} ({chunk['window_blocks']} blocks) — {key.upper()}"
                )
                ax.set_xlabel("frame in window")
                ax.set_ylabel("layer")
                ax.set_yticks(range(len(layer_labels)))
                ax.set_yticklabels(layer_labels)
                # Mark the boundary between context and current chunk.
                ax.axvline(x=n_ctx * npb - 0.5, color="red", linewidth=0.8, linestyle="--")
                fig.colorbar(im, ax=ax, fraction=0.04)
        fig.suptitle(
            f"AR_refresh K/V per-layer per-frame |abs| — "
            f"ride {Path(args.rank_zarr).stem}, seed {args.seed}",
            y=1.02,
        )
        fig.tight_layout()
        out_png = out_dir / "kv_summary.png"
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved summary plot -> %s", out_png)

    # --- Also write a small JSON manifest for easy browsing. ---
    manifest = {
        "pt": str(out_pt),
        "png": str(out_dir / "kv_summary.png"),
        "meta": payload["meta"],
        "n_chunks": len(captures),
    }
    with (out_dir / "manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
