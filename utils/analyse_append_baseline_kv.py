#!/usr/bin/env python3
"""Append-baseline K/V capture (apples-to-apples with
``analyse_AR_refresh_kv.py``).

The append baseline uses ``eval_causal_AR.generate_ar`` with
``cache_refresh="append"``. Each chunk runs 4 denoise forwards on the
current 3-frame chunk (with cached context) plus one t=0 cache-refresh
on the clean ``pred_x0``. We capture the pre-RoPE K/V of every forward
via hooks on each attention block's ``k`` and ``v`` ``nn.Linear``
modules, then assemble a per-chunk "window" that matches the AR_refresh
view:

  window = [last ``cache_chunks`` committed K/V (t=0, clean)] + [current chunk K/V at t=50 (noisy last denoise)]

Output layout mirrors the AR_refresh script:

  - ``kv_capture.pt`` with the same schema.
  - ``kv_summary.png`` with per-chunk per-layer |K|/|V| heatmaps.
  - ``manifest.json`` for quick browsing.

Usage::

    python utils/analyse_append_baseline_kv.py \\
        --config configs/action_ode_distill_local.yaml \\
        --student_ckpt /home/ashish/action_ode_step0001000.pt \\
        --rank_zarr 20240408152948.zarr --rank_offset 0 \\
        --encoded_root /home/ashish/frodobots/frodobots_encoded \\
        --caption_root /home/ashish/frodobots/frodobots_captions/train \\
        --motion_root /home/ashish/frodobots/frodobots_motion \\
        --ss_vae_checkpoint action_query/checkpoints/ss_vae_8free.pt \\
        --ar_initial_chunks 1 --ar_gen_chunks 7 --cache_chunks 3 \\
        --denoising_steps 4 --seed 42 \\
        --output_dir /home/ashish/ARRWM/eval/kv_capture_append_<ts>
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

from utils.eval_causal_AR import (
    ODEChainPipeline,
    load_per_rank_ride_ar,
    _initialize_kv_cache,
    _initialize_crossattn_cache,
    _set_attention_window,
    FRAME_SPATIAL_TOKENS,
    BASE_CHUNK_FRAMES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


def _install_kv_capture(base_dit, capture_layer_indices: List[int]) -> List[torch.nn.Module]:
    """Same scheme as analyse_AR_refresh_kv: forward hooks on
    ``self_attn.k`` and ``self_attn.v`` Linear modules stash their
    outputs on the parent block so the outer loop can read them after
    each forward."""
    attn_blocks: List[torch.nn.Module] = []
    for _, module in base_dit.named_modules():
        if module.__class__.__name__ == "CausalWanSelfAttention":
            attn_blocks.append(module)

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


def _snapshot(attn_blocks, capture_dtype) -> Dict[int, Dict[str, torch.Tensor]]:
    """Read current ``_captured_k``/``_captured_v`` off each block and
    move to CPU in the requested dtype. Returns a dict
    {layer_idx: {"k": ..., "v": ...}} for blocks that have capture
    enabled and whose captures aren't None."""
    out: Dict[int, Dict[str, torch.Tensor]] = {}
    for layer_idx, blk in enumerate(attn_blocks):
        if not getattr(blk, "_capture_enabled", False):
            continue
        k = blk._captured_k
        v = blk._captured_v
        if k is None or v is None:
            continue
        out[layer_idx] = {
            "k": k.to(device="cpu", dtype=capture_dtype).clone(),
            "v": v.to(device="cpu", dtype=capture_dtype).clone(),
        }
        blk._captured_k = None
        blk._captured_v = None
    return out


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
    p.add_argument("--cache_chunks", type=int, default=3,
                   help="KV cache depth (committed chunks). Match AR_refresh's "
                        "fifo_size for apples-to-apples heatmaps.")
    p.add_argument("--denoising_steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--capture_layers", type=str, default="0,14,29")
    p.add_argument("--capture_dtype", type=str, default="float16",
                   choices=["float16", "bfloat16", "float32"])
    return p.parse_args()


@torch.no_grad()
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

    # --- Build the AR (non-refresh) pipeline. Same as eval_causal_AR. ---
    pipe = ODEChainPipeline(device, dtype=dtype)
    pipe.build(args.config, use_action_tokens=True)
    pipe.load_checkpoint(args.student_ckpt)
    pipe.set_denoising_steps(int(args.denoising_steps))

    num_frame_per_block = BASE_CHUNK_FRAMES
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
    initial_latents_dev = initial_latents[:, :seed_frames].to(device=device, dtype=dtype)
    prompt_embeds_dev = prompt_embeds.to(device=device, dtype=dtype)
    noisy_fa_full = noisy_fa_full.to(device=device, dtype=dtype)

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

    # --- Configure the DiT for streaming (mirrors generate_ar setup). ---
    B = 1
    chunks_per_step = 1
    cache_chunks = int(args.cache_chunks)
    base_dit.num_frame_per_block = num_frame_per_block
    base_dit.block_mask = None
    action_tokens_per_frame = int(getattr(base_dit, "action_tokens_per_frame", 1))
    frame_seq_length = FRAME_SPATIAL_TOKENS + action_tokens_per_frame
    local_attn_size_frames = (cache_chunks + chunks_per_step) * BASE_CHUNK_FRAMES
    kv_cache_tokens = local_attn_size_frames * frame_seq_length
    required_chunk_tokens = num_frame_per_block * frame_seq_length
    pipe.wrapper.seq_len = max(int(pipe.wrapper.seq_len), required_chunk_tokens)
    _set_attention_window(
        base_dit, local_attn_size_frames=local_attn_size_frames, max_tokens=kv_cache_tokens,
    )

    num_transformer_blocks = len(base_dit.blocks)
    C = int(initial_latents_dev.shape[2])
    H = int(initial_latents_dev.shape[3])
    W = int(initial_latents_dev.shape[4])

    kv_cache = _initialize_kv_cache(
        num_transformer_blocks=num_transformer_blocks,
        batch_size=B,
        kv_cache_size_tokens=kv_cache_tokens,
        dtype=dtype,
        device=device,
    )
    crossattn_cache = _initialize_crossattn_cache(
        num_transformer_blocks=num_transformer_blocks,
        batch_size=B,
        dtype=dtype,
        device=device,
    )
    scheduler = pipe.scheduler
    scheduler.sigmas = scheduler.sigmas.to(device)
    ts = pipe.denoising_step_list

    noise_seed = int(args.seed) + 1_000_003
    torch.manual_seed(noise_seed)
    torch.cuda.manual_seed(noise_seed)

    # Seed prefill: push the first block of initial_latents into the cache at t=0.
    current_start_frame = 0
    refresh_t_block = torch.full([B, num_frame_per_block], 0.0, device=device, dtype=torch.float32)

    # Keep a FIFO of the LAST `cache_chunks` committed chunks' K/V captures,
    # keyed by captured layer index. Each entry is {"k": [B, seq, dim], "v": ...}
    # where seq = 3 frames × frame_seq_length.
    committed_fifo: List[Dict[int, Dict[str, torch.Tensor]]] = []

    # --- Seed prefill forward (1 real chunk). ---
    seed_lo, seed_hi = 0, num_frame_per_block
    seed_lat_block = initial_latents_dev[:, seed_lo:seed_hi]
    seed_fa_block = noisy_fa_full[:, seed_lo:seed_hi]
    seed_cond = pipe._build_action_cond_chunk(
        prompt_embeds_dev, seed_fa_block, num_frames=num_frame_per_block,
    )
    with torch.amp.autocast("cuda", dtype=dtype):
        pipe.wrapper(
            noisy_image_or_video=seed_lat_block,
            conditional_dict=seed_cond,
            timestep=refresh_t_block,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start_frame * frame_seq_length,
        )
    committed_fifo.append(_snapshot(attn_blocks, capture_dtype))
    current_start_frame += num_frame_per_block
    log.info("[append/capture] seed prefill committed (1 chunk) — fifo_len=%d", len(committed_fifo))

    # --- AR generation loop. ---
    num_steps_needed = int(args.ar_gen_chunks)
    captures: List[Dict[str, Any]] = []
    for step_idx in range(num_steps_needed):
        frame_lo = current_start_frame
        frame_hi = frame_lo + num_frame_per_block
        block_fa = noisy_fa_full[:, frame_lo:frame_hi]
        cond = pipe._build_action_cond_chunk(
            prompt_embeds_dev, block_fa, num_frames=num_frame_per_block,
        )

        noise = torch.randn(
            [B, num_frame_per_block, C, H, W],
            dtype=torch.float32, device=device,
        )
        x = noise.to(dtype)

        pred_x0: Optional[torch.Tensor] = None
        final_pass_capture: Dict[int, Dict[str, torch.Tensor]] = {}
        for d_idx in range(int(ts.shape[0])):
            t_val = float(ts[d_idx].item())
            tt = torch.full(
                [B, num_frame_per_block], t_val,
                device=device, dtype=torch.float32,
            )
            with torch.amp.autocast("cuda", dtype=dtype):
                out = pipe.wrapper(
                    noisy_image_or_video=x,
                    conditional_dict=cond,
                    timestep=tt,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start_frame * frame_seq_length,
                )
            pred_x0 = out[1]
            if d_idx == int(ts.shape[0]) - 1:
                final_pass_capture = _snapshot(attn_blocks, capture_dtype)
            if d_idx < int(ts.shape[0]) - 1:
                next_t = float(ts[d_idx + 1].item())
                flat = pred_x0.flatten(0, 1).float()
                flat_noise = torch.randn_like(flat)
                flat_t = torch.full(
                    (flat.shape[0],), next_t,
                    device=device, dtype=torch.float32,
                )
                x = (
                    scheduler.add_noise(flat, flat_noise, flat_t)
                    .view(B, num_frame_per_block, C, H, W)
                    .to(dtype)
                )
        assert pred_x0 is not None

        # --- Commit cache-refresh: t=0 forward on clean pred_x0. The
        #     hooks overwrite the captures with the CLEAN K/V; snapshot
        #     that as this chunk's committed FIFO entry. ---
        with torch.amp.autocast("cuda", dtype=dtype):
            pipe.wrapper(
                noisy_image_or_video=pred_x0,
                conditional_dict=cond,
                timestep=refresh_t_block,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start_frame * frame_seq_length,
            )
        clean_commit_capture = _snapshot(attn_blocks, capture_dtype)

        # Assemble the "window" for this chunk: the last `cache_chunks`
        # committed K/Vs (pre-RoPE, at t=0) followed by the current
        # chunk's final-denoise K/V (pre-RoPE, at t=ts[-1]=50).
        ctx_entries = committed_fifo[-cache_chunks:] if cache_chunks > 0 else []
        n_ctx_blocks = len(ctx_entries)
        window_blocks = n_ctx_blocks + 1
        window_frames = window_blocks * num_frame_per_block

        window_per_layer: Dict[int, Dict[str, torch.Tensor]] = {}
        for layer_idx in capture_layer_indices:
            ctx_ks = [e[layer_idx]["k"] for e in ctx_entries if layer_idx in e]
            ctx_vs = [e[layer_idx]["v"] for e in ctx_entries if layer_idx in e]
            cur = final_pass_capture.get(layer_idx)
            if cur is None:
                continue
            k_cat = torch.cat(ctx_ks + [cur["k"]], dim=1) if ctx_ks else cur["k"]
            v_cat = torch.cat(ctx_vs + [cur["v"]], dim=1) if ctx_vs else cur["v"]
            window_per_layer[layer_idx] = {"k": k_cat, "v": v_cat}

        captures.append({
            "chunk_idx": int(step_idx),
            "window_blocks": int(window_blocks),
            "n_ctx_blocks": int(n_ctx_blocks),
            "cur_global_frame_lo": int(frame_lo),
            "window_frames": int(window_frames),
            "layers": window_per_layer,
        })

        committed_fifo.append(clean_commit_capture)
        # Keep FIFO bounded so the context snapshot mirrors the cache's
        # eviction behaviour.
        if len(committed_fifo) > cache_chunks + 1:  # seed + last N commits
            committed_fifo.pop(0)

        current_start_frame += num_frame_per_block
        log.info(
            "[append/capture] step %d/%d committed | window=%d blocks "
            "(%d ctx + 1 cur) | cache chunks holding=%d | captured %d layer(s)",
            step_idx + 1, num_steps_needed, window_blocks, n_ctx_blocks,
            len(committed_fifo), len(window_per_layer),
        )

    # --- Save. ---
    payload = {
        "meta": {
            "ts": ts.detach().cpu().tolist(),
            "num_frame_per_block": num_frame_per_block,
            "num_blocks": total_blocks,
            "capture_layers": capture_layer_indices,
            "cache_chunks": cache_chunks,
            "ar_gen_chunks": int(args.ar_gen_chunks),
            "ar_initial_chunks": int(args.ar_initial_chunks),
            "seed": int(args.seed),
            "rank_zarr": args.rank_zarr,
            "rank_offset": int(args.rank_offset),
            "student_ckpt": args.student_ckpt,
            "capture_dtype": str(capture_dtype),
            "variant": "append_baseline",
        },
        "chunks": captures,
    }
    out_pt = out_dir / "kv_capture.pt"
    torch.save(payload, out_pt)
    log.info("Saved K/V capture -> %s (%.1f MB)",
             out_pt, out_pt.stat().st_size / (1024 * 1024))

    # --- Summary plot (identical style to AR_refresh). ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available — skipping summary plot.")
        plt = None

    if plt is not None:
        npb = num_frame_per_block
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
                layer_rows = []
                layer_labels = []
                for layer_idx in sorted(chunk["layers"].keys()):
                    t = chunk["layers"][layer_idx][key]  # [B, seq, dim]
                    seq = t.shape[1]
                    if wf == 0 or seq % wf != 0:
                        continue
                    frame_seqlen = seq // wf
                    per_frame = t.float().view(
                        t.shape[0], wf, frame_seqlen, t.shape[2],
                    ).abs().mean(dim=(2, 3)).squeeze(0).numpy()
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
                ax.axvline(x=n_ctx * npb - 0.5, color="red", linewidth=0.8, linestyle="--")
                fig.colorbar(im, ax=ax, fraction=0.04)
        fig.suptitle(
            f"Append-baseline K/V per-layer per-frame |abs| — "
            f"ride {Path(args.rank_zarr).stem}, seed {args.seed}, cache_chunks={cache_chunks}",
            y=1.02,
        )
        fig.tight_layout()
        out_png = out_dir / "kv_summary.png"
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved summary plot -> %s", out_png)

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
