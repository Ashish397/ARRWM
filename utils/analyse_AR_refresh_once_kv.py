#!/usr/bin/env python3
"""AR_refresh variant that recaches context ONCE per chunk (not per
denoise pass), with K/V capture for comparison against
``analyse_AR_refresh_kv.py`` (per-pass recompute) and
``analyse_append_baseline_kv.py`` (classic append KV cache).

At each chunk:

  1. Reset the KV cache. Forward the full FIFO (all context latents
     concatenated) through ``_forward_inference`` with ``timestep=0`` in
     **one** call. This populates the KV cache with context K/V — the
     RoPE/cache positions are 0, npb, 2*npb, ... matching AR_refresh's
     context layout. Unlike ``eval_causal_AR.generate_ar``'s
     ``cache_refresh="full_fifo"`` which runs one small forward per
     FIFO entry, this is a single joint forward so the context chunks
     "see each other" the way AR_refresh's ``_forward_train`` has them
     see each other in its full-window forward.
  2. Run the 4 ODE denoise passes on the **current chunk only** (3
     frames), reusing the cached context. This is the recache saving
     — the context is NOT rebuilt between passes.
  3. After the final denoise pass, capture K/V (same hook scheme as
     the other two analyse scripts) and commit pred_x0 to the FIFO.

Output layout mirrors ``analyse_AR_refresh_kv.py`` and
``analyse_append_baseline_kv.py`` (same ``kv_capture.pt`` +
``kv_summary.png`` + ``manifest.json``) so the three heatmaps line up
pixel-for-pixel for comparison.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from utils.eval_chain import frames_to_mp4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


def _reset_kv_cache(kv_cache) -> None:
    """Zero every block's K/V buffer and reset the end-index counters so
    the next forward starts from a clean slate."""
    for blk in kv_cache:
        blk["k"].zero_()
        blk["v"].zero_()
        blk["global_end_index"].zero_()
        blk["local_end_index"].zero_()


def _install_kv_capture(base_dit, capture_layer_indices: List[int]) -> List[torch.nn.Module]:
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
    p.add_argument("--fifo_size", type=int, default=3,
                   help="Max context chunks retained in the FIFO.")
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

    B = 1
    fifo_size = int(args.fifo_size)
    chunks_per_step = 1
    max_window_blocks = fifo_size + 1
    base_dit.num_frame_per_block = num_frame_per_block
    base_dit.block_mask = None
    action_tokens_per_frame = int(getattr(base_dit, "action_tokens_per_frame", 1))
    frame_seq_length = FRAME_SPATIAL_TOKENS + action_tokens_per_frame
    local_attn_size_frames = max_window_blocks * BASE_CHUNK_FRAMES
    kv_cache_tokens = local_attn_size_frames * frame_seq_length
    required_chunk_tokens = max_window_blocks * num_frame_per_block * frame_seq_length
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

    # FIFO of raw clean latents (same as AR_refresh's fifo) + their ride
    # frame-lo for action lookup.
    seed_chunks = seed_frames // num_frame_per_block
    fifo_lat: List[torch.Tensor] = []
    fifo_frame_lo: List[int] = []
    for c in range(seed_chunks):
        lo = c * num_frame_per_block
        hi = lo + num_frame_per_block
        fifo_lat.append(initial_latents_dev[:, lo:hi])
        fifo_frame_lo.append(lo)

    captures: List[Dict[str, Any]] = []
    generated: List[torch.Tensor] = []   # per-chunk pred_x0 for the final mp4
    t0 = time.time()
    for chunk_idx in range(int(args.ar_gen_chunks)):
        n_ctx_blocks = min(len(fifo_lat), fifo_size)
        window_blocks = n_ctx_blocks + 1
        window_frames = window_blocks * num_frame_per_block
        cur_global_frame_lo = (seed_chunks + chunk_idx) * num_frame_per_block
        cur_global_frame_hi = cur_global_frame_lo + num_frame_per_block

        # --- ONCE-PER-STEP RECACHE: one joint forward over the full FIFO
        #     at t=0 to populate the context K/V in the KV cache. ---
        _reset_kv_cache(kv_cache)
        ctx_entries = fifo_lat[-n_ctx_blocks:] if n_ctx_blocks > 0 else []
        ctx_frame_los = fifo_frame_lo[-n_ctx_blocks:] if n_ctx_blocks > 0 else []
        current_start_frame = 0
        if ctx_entries:
            ctx_cat = torch.cat(ctx_entries, dim=1)  # [B, n_ctx*npb, C, H, W]
            ctx_fa = torch.cat(
                [noisy_fa_full[:, fl:fl + num_frame_per_block] for fl in ctx_frame_los],
                dim=1,
            )
            ctx_cond = pipe._build_action_cond_chunk(
                prompt_embeds_dev, ctx_fa, num_frames=n_ctx_blocks * num_frame_per_block,
            )
            ctx_t = torch.full(
                [B, n_ctx_blocks * num_frame_per_block], 0.0,
                device=device, dtype=torch.float32,
            )
            with torch.amp.autocast("cuda", dtype=dtype):
                pipe.wrapper(
                    noisy_image_or_video=ctx_cat,
                    conditional_dict=ctx_cond,
                    timestep=ctx_t,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start_frame * frame_seq_length,
                )
            current_start_frame += n_ctx_blocks * num_frame_per_block

        # After the context forward the hooks hold the LAST context
        # block's K/V (they get overwritten each forward). For the
        # once-per-step run that's the "refreshed" context; snapshot
        # here so the visualisation can see what went into the cache.
        # But to build the window heatmap we need per-block context
        # captures too, which aren't directly available (only last
        # block's). Workaround: split the context rebuild into the
        # single joint forward above PLUS per-block single forwards
        # JUST to capture their K/V one at a time — those captures
        # are used only for visualisation, and we discard the cache
        # state they produce by issuing a rebuild immediately after.
        # That way the once-per-step semantics is preserved for the
        # denoise passes (they run against the JOINT-forward cache),
        # and we still get per-block context K/V for the heatmap.
        ctx_captures_per_block: List[Dict[int, Dict[str, torch.Tensor]]] = []
        if ctx_entries:
            # Temporarily back up the cache state we want to preserve.
            saved_k = [blk["k"].clone() for blk in kv_cache]
            saved_v = [blk["v"].clone() for blk in kv_cache]
            saved_ge = [blk["global_end_index"].clone() for blk in kv_cache]
            saved_le = [blk["local_end_index"].clone() for blk in kv_cache]
            # Run per-block forwards just for capture (discard cache
            # updates afterwards).
            _reset_kv_cache(kv_cache)
            per_block_cstart = 0
            for f_lat, f_lo in zip(ctx_entries, ctx_frame_los):
                f_fa = noisy_fa_full[:, f_lo:f_lo + num_frame_per_block]
                f_cond = pipe._build_action_cond_chunk(
                    prompt_embeds_dev, f_fa, num_frames=num_frame_per_block,
                )
                f_t = torch.full(
                    [B, num_frame_per_block], 0.0,
                    device=device, dtype=torch.float32,
                )
                with torch.amp.autocast("cuda", dtype=dtype):
                    pipe.wrapper(
                        noisy_image_or_video=f_lat,
                        conditional_dict=f_cond,
                        timestep=f_t,
                        kv_cache=kv_cache,
                        crossattn_cache=crossattn_cache,
                        current_start=per_block_cstart * frame_seq_length,
                    )
                ctx_captures_per_block.append(_snapshot(attn_blocks, capture_dtype))
                per_block_cstart += num_frame_per_block
            # Restore the joint-forward cache state for the denoise passes.
            for blk, k, v, ge, le in zip(kv_cache, saved_k, saved_v, saved_ge, saved_le):
                blk["k"].copy_(k)
                blk["v"].copy_(v)
                blk["global_end_index"].copy_(ge)
                blk["local_end_index"].copy_(le)

        # --- 4 denoise passes on the CURRENT chunk only, reusing cache. ---
        block_fa = noisy_fa_full[:, cur_global_frame_lo:cur_global_frame_hi]
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

        # Assemble the window capture: per-block context captures
        # (captured on separate forwards, for viz) + current chunk.
        window_per_layer: Dict[int, Dict[str, torch.Tensor]] = {}
        for layer_idx in capture_layer_indices:
            ctx_ks = [e[layer_idx]["k"] for e in ctx_captures_per_block if layer_idx in e]
            ctx_vs = [e[layer_idx]["v"] for e in ctx_captures_per_block if layer_idx in e]
            cur = final_pass_capture.get(layer_idx)
            if cur is None:
                continue
            k_cat = torch.cat(ctx_ks + [cur["k"]], dim=1) if ctx_ks else cur["k"]
            v_cat = torch.cat(ctx_vs + [cur["v"]], dim=1) if ctx_vs else cur["v"]
            window_per_layer[layer_idx] = {"k": k_cat, "v": v_cat}

        captures.append({
            "chunk_idx": int(chunk_idx),
            "window_blocks": int(window_blocks),
            "n_ctx_blocks": int(n_ctx_blocks),
            "cur_global_frame_lo": int(cur_global_frame_lo),
            "window_frames": int(window_frames),
            "layers": window_per_layer,
        })
        log.info(
            "[AR_refresh_once/capture] chunk %d/%d committed | window=%d blocks | "
            "captured %d layer(s)",
            chunk_idx + 1, int(args.ar_gen_chunks), window_blocks, len(window_per_layer),
        )

        # Commit pred_x0 to FIFO. Evict oldest when above fifo_size.
        if len(fifo_lat) >= fifo_size:
            fifo_lat.pop(0)
            fifo_frame_lo.pop(0)
        fifo_lat.append(pred_x0.detach().to(dtype))
        fifo_frame_lo.append(cur_global_frame_lo)
        generated.append(pred_x0.detach().to(torch.float32))

    log.info("Rollout + capture done in %.1fs. Captured %d chunks.",
             time.time() - t0, len(captures))

    payload = {
        "meta": {
            "ts": ts.detach().cpu().tolist(),
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
            "variant": "ar_refresh_once_per_step",
        },
        "chunks": captures,
    }
    out_pt = out_dir / "kv_capture.pt"
    torch.save(payload, out_pt)
    log.info("Saved K/V capture -> %s (%.1f MB)",
             out_pt, out_pt.stat().st_size / (1024 * 1024))

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
                    t = chunk["layers"][layer_idx][key]
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
            f"AR_refresh_ONCE K/V per-layer per-frame |abs| — "
            f"ride {Path(args.rank_zarr).stem}, seed {args.seed}, fifo={fifo_size}",
            y=1.02,
        )
        fig.tight_layout()
        out_png = out_dir / "kv_summary.png"
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved summary plot -> %s", out_png)

    # --- Decode + save mp4 (seed + AR-generated chunks). ---
    out_mp4 = out_dir / "rollout_raw.mp4"
    try:
        seed_real = initial_latents_dev[:, :seed_frames].to(torch.float32)
        full_latents = torch.cat([seed_real] + generated, dim=1)
        log.info(
            "Decoding %d latent frames -> mp4 ...", int(full_latents.shape[1]),
        )
        video_np = pipe.decode_latents(full_latents)
        frames_to_mp4(video_np, str(out_mp4), fps=20)
        log.info("Saved mp4 -> %s", out_mp4)
    except Exception as exc:  # noqa: BLE001
        log.warning("mp4 write failed: %s", exc)
        out_mp4 = None

    manifest = {
        "pt": str(out_pt),
        "png": str(out_dir / "kv_summary.png"),
        "mp4": str(out_mp4) if out_mp4 is not None else None,
        "meta": payload["meta"],
        "n_chunks": len(captures),
    }
    with (out_dir / "manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
