#!/usr/bin/env python3
"""AR_refresh (per-pass, the good variant) with attention-mask overlay.

For each generated chunk, we compute per-query-position → per-KV-chunk
attention weights (softmax(QK^T/sqrt(d)), post-norm, post-RoPE,
block-causal-masked, aggregated over heads and captured layers), then
composite an additive colour overlay on the decoded pixel video:

  - chunk 0 (oldest KV in window) → RED
  - chunk 1                       → GREEN
  - chunk 2                       → BLUE
  - chunk 3 (self, if included)   → YELLOW = R+G

Colours add, so a patch that attends to both chunk 0 and chunk 2 shows
magenta. Each chunk's overlay intensity reflects how strongly the
current chunk's query at that spatial location attended to that KV
chunk's keys.

Writes two mp4s + a .pt with the raw attention tensors:

  - ``rollout_raw.mp4``        — pixel video, no overlay.
  - ``rollout_attention.mp4``  — pixel video with additive overlay.
  - ``attention.pt``           — per-chunk per-layer attention maps.

Usage::

    python utils/visualise_AR_refresh_attention.py \\
        --config configs/action_ode_distill_local.yaml \\
        --student_ckpt /home/ashish/action_ode_step0001000.pt \\
        --rank_zarr 20240408152948.zarr --rank_offset 0 \\
        --encoded_root /home/ashish/frodobots/frodobots_encoded \\
        --caption_root /home/ashish/frodobots/frodobots_captions/train \\
        --motion_root /home/ashish/frodobots/frodobots_motion \\
        --ss_vae_checkpoint action_query/checkpoints/ss_vae_8free.pt \\
        --ar_initial_chunks 1 --ar_gen_chunks 7 --fifo_size 3 \\
        --denoising_steps 4 --seed 42 \\
        --output_dir /home/ashish/ARRWM/eval/attention_overlay_<ts>
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_AF_ROOT = _REPO_ROOT / "action-forcing"
if str(_AF_ROOT) not in sys.path:
    sys.path.insert(0, str(_AF_ROOT))

from utils.eval_causal_AR import load_per_rank_ride_ar, BASE_CHUNK_FRAMES
from utils.eval_causal_AR_chain import ODEARRefreshPipeline
from utils.eval_chain import frames_to_mp4
from wan.modules.model import rope_apply
from wan.modules.causal_model import _separate_action_tokens, _merge_action_tokens

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hook scheme — capture Q, K raw (Linear outputs, pre-norm, pre-RoPE) on
# each captured attention block's q/k nn.Linear modules. We run the
# model's own ``norm_q``/``norm_k`` and ``rope_apply`` externally after
# each forward to compute attention weights.
# ---------------------------------------------------------------------------


def _install_qk_capture(base_dit, capture_layer_indices: List[int]) -> List[torch.nn.Module]:
    attn_blocks: List[torch.nn.Module] = []
    for _, module in base_dit.named_modules():
        if module.__class__.__name__ == "CausalWanSelfAttention":
            attn_blocks.append(module)
    for i, blk in enumerate(attn_blocks):
        blk._capture_enabled = i in capture_layer_indices
        blk._captured_q = None
        blk._captured_k = None

    def _make_q_hook(blk):
        def hook(module, inp, out):
            if getattr(blk, "_capture_enabled", False):
                blk._captured_q = out.detach()
        return hook

    def _make_k_hook(blk):
        def hook(module, inp, out):
            if getattr(blk, "_capture_enabled", False):
                blk._captured_k = out.detach()
        return hook

    for blk in attn_blocks:
        blk.q.register_forward_hook(_make_q_hook(blk))
        blk.k.register_forward_hook(_make_k_hook(blk))
    return attn_blocks


# ---------------------------------------------------------------------------
# Attention-weights computation (post-norm, post-RoPE, block-causal).
# ---------------------------------------------------------------------------


def _compute_attention_per_chunk(
    blk,
    q_raw: torch.Tensor,           # [B, seq_total, dim] — output of self.q(x)
    k_raw: torch.Tensor,           # [B, seq_total, dim] — output of self.k(x)
    *,
    num_frames: int,               # F in grid_sizes
    spatial_h: int,                # H in grid_sizes
    spatial_w: int,                # W in grid_sizes
    action_per_frame: int,
    freqs: torch.Tensor,
    n_ctx_blocks: int,
    num_frame_per_block: int,
) -> torch.Tensor:
    """Run norm_q, norm_k, RoPE, and scaled-dot-product attention with a
    block-causal mask on the captured Q/K. Returns per-current-frame,
    per-spatial-patch, per-KV-chunk attention weights.

    Returns: Tensor [cur_frames, spatial_h, spatial_w, n_kv_blocks]
    — attention from each query patch in the current chunk (last
    ``num_frame_per_block`` frames of the window) to each KV chunk
    (``n_kv_blocks = n_ctx_blocks + 1``, including self).
    """
    device = q_raw.device
    # Move to fp32 for numerical stability of softmax.
    q_raw = q_raw.float()
    k_raw = k_raw.float()
    B, seq_total, dim = q_raw.shape
    n_heads = blk.num_heads
    head_dim = blk.head_dim
    frame_seq = spatial_h * spatial_w + action_per_frame
    assert seq_total == num_frames * frame_seq, (seq_total, num_frames, frame_seq)

    # Apply RMS norm (matches qkv_fn's ``self.norm_q(self.q(x))``).
    q = blk.norm_q(q_raw).view(B, seq_total, n_heads, head_dim)
    k = blk.norm_k(k_raw).view(B, seq_total, n_heads, head_dim)

    grid_sizes = torch.tensor(
        [[num_frames, spatial_h, spatial_w]], dtype=torch.long, device=device,
    )

    if action_per_frame > 0:
        q_sp, q_act = _separate_action_tokens(q, grid_sizes, action_per_frame)
        k_sp, k_act = _separate_action_tokens(k, grid_sizes, action_per_frame)
        q_sp_roped = rope_apply(q_sp, grid_sizes, freqs.float(), temporal_offset=0)
        k_sp_roped = rope_apply(k_sp, grid_sizes, freqs.float(), temporal_offset=0)
        q_roped = _merge_action_tokens(q_sp_roped, q_act, grid_sizes, action_per_frame)
        k_roped = _merge_action_tokens(k_sp_roped, k_act, grid_sizes, action_per_frame)
    else:
        q_roped = rope_apply(q, grid_sizes, freqs.float(), temporal_offset=0)
        k_roped = rope_apply(k, grid_sizes, freqs.float(), temporal_offset=0)

    # Transpose to [B, n_heads, seq_total, head_dim] for attention.
    q_h = q_roped.transpose(1, 2).contiguous()
    k_h = k_roped.transpose(1, 2).contiguous()

    # Scaled dot product. [B, n_heads, seq_q, seq_k].
    scale = 1.0 / math.sqrt(head_dim)
    scores = (q_h @ k_h.transpose(-1, -2)) * scale

    # Build block-causal mask at the CHUNK level (chunk i attends to
    # chunks 0..i). Mask shape: [seq_q, seq_k], expanded on head/batch.
    nb = n_ctx_blocks + 1
    # frame → block index
    frame_to_block = torch.arange(num_frames, device=device) // num_frame_per_block
    # token → frame index
    token_frame = torch.arange(seq_total, device=device) // frame_seq
    token_block = frame_to_block[token_frame]
    # block-causal: q_block >= k_block
    causal = token_block.unsqueeze(0) >= token_block.unsqueeze(1)  # [seq_q, seq_k] True where visible
    # False => -inf
    neg_inf = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(~causal, neg_inf)

    attn = F.softmax(scores, dim=-1)  # [B, n_heads, seq_q, seq_k]

    # Average over heads.
    attn_mean = attn.mean(dim=1)  # [B, seq_q, seq_k]

    # We only care about queries in the CURRENT chunk (last num_frame_per_block frames).
    current_start_q = (num_frames - num_frame_per_block) * frame_seq
    attn_cur = attn_mean[:, current_start_q:, :]  # [B, cur_seq_q, seq_k]

    # Collapse action-token rows in q: for each frame in the current
    # chunk, we only visualise spatial rows. Slice out action tokens.
    # cur_seq_q spans [num_frame_per_block * frame_seq] tokens.
    cur_seq_q = num_frame_per_block * frame_seq
    assert attn_cur.shape[1] == cur_seq_q, (attn_cur.shape, cur_seq_q)

    # Reshape q side: [B, cur_frames, frame_seq, seq_k]
    attn_cur_f = attn_cur.view(B, num_frame_per_block, frame_seq, seq_total)

    # Slice out action-token query rows: keep the first spatial_h*spatial_w.
    spatial_per_frame = spatial_h * spatial_w
    attn_cur_sp = attn_cur_f[:, :, :spatial_per_frame, :]  # [B, cur_f, sp, seq_k]

    # Collapse k side: for each KV block, sum attention over all its
    # tokens (both spatial and action across the 3 frames of that
    # chunk).
    # Reshape k-dim to [num_frames, frame_seq]: token_block tells us
    # which block each token belongs to.
    # Simpler: sum over the 3 * frame_seq tokens per block.
    per_block_attn = []
    for b in range(nb):
        # mask of tokens in block b along seq_k
        b_mask = (token_block == b)
        # attn_cur_sp[..., b_mask].sum(-1) → [B, cur_f, sp]
        per_block_attn.append(attn_cur_sp[..., b_mask].sum(dim=-1))
    attn_per_block = torch.stack(per_block_attn, dim=-1)  # [B, cur_f, sp, nb]

    # Reshape sp -> (H, W).
    attn_hw = attn_per_block.view(B, num_frame_per_block, spatial_h, spatial_w, nb)
    # Drop B.
    return attn_hw[0].detach().cpu()  # [cur_f, H, W, nb]


# ---------------------------------------------------------------------------
# Overlay rendering.
# ---------------------------------------------------------------------------


# Additive RGB colours for up to 6 KV chunks. If more, they cycle.
# Chosen for visibility under additive compositing: R, G, B, then
# secondary mixes.
_KV_COLOURS = np.array([
    [1.0, 0.0, 0.0],  # red    — chunk 0 (oldest)
    [0.0, 1.0, 0.0],  # green  — chunk 1
    [0.0, 0.0, 1.0],  # blue   — chunk 2
    [1.0, 1.0, 0.0],  # yellow — chunk 3 (self in fifo_size=3 steady)
    [1.0, 0.0, 1.0],  # magenta
    [0.0, 1.0, 1.0],  # cyan
], dtype=np.float32)


def _render_overlay_for_chunk(
    pixel_frames_chunk: np.ndarray,      # [T_pix, H_pix, W_pix, 3] uint8 — current chunk's pixels
    attn_per_block: np.ndarray,          # [cur_f, H_lat, W_lat, nb] float32
    *,
    alpha: float = 0.6,
    saturate: float = 1.0,
) -> np.ndarray:
    """Composite additive coloured attention overlays on the current
    chunk's pixel frames.

    ``attn_per_block`` is at latent spatial resolution (30×52 per frame).
    Each latent frame maps to ``T_pix / cur_f`` pixel frames (VAE
    temporal upscale). We nearest-upsample both dims to pixel resolution.

    ``alpha`` = overlay opacity; 0.0 = no overlay, 1.0 = overlay fully
    replaces pixel RGB. ``saturate`` scales the raw attention weights
    before colour multiplication (1.0 = use raw probabilities, which
    rarely exceed ~0.3 per chunk; bump higher to make colours visible).
    """
    T_pix, H_pix, W_pix, _ = pixel_frames_chunk.shape
    cur_f, H_lat, W_lat, nb = attn_per_block.shape

    # Upsample spatial: latent grid → pixel grid (bilinear).
    tens = torch.from_numpy(attn_per_block).permute(0, 3, 1, 2).float()  # [cur_f, nb, H_lat, W_lat]
    tens_pix = F.interpolate(tens, size=(H_pix, W_pix), mode="bilinear", align_corners=False)
    tens_pix = tens_pix.permute(0, 2, 3, 1).numpy()  # [cur_f, H_pix, W_pix, nb]

    # Temporal nearest: map cur_f latent frames onto T_pix pixel frames.
    t_idx = np.linspace(0, cur_f - 1e-6, num=T_pix).astype(np.int32)
    tens_time = tens_pix[t_idx]  # [T_pix, H_pix, W_pix, nb]

    # Colour composition: per-pixel RGB = sum_b attention_b * colour_b.
    overlay_rgb = np.zeros((T_pix, H_pix, W_pix, 3), dtype=np.float32)
    for b in range(nb):
        c = _KV_COLOURS[b % len(_KV_COLOURS)]
        overlay_rgb += tens_time[:, :, :, b:b + 1] * c[None, None, None, :] * float(saturate)
    overlay_rgb = np.clip(overlay_rgb, 0.0, 1.0)

    # Additive blend with alpha.
    base = pixel_frames_chunk.astype(np.float32) / 255.0
    out = base * (1.0 - alpha) + overlay_rgb * alpha
    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def _colour_legend_strip(nb: int, height: int = 28, width: int = 832) -> np.ndarray:
    """Build a small legend strip showing which colour corresponds to
    which KV chunk. Returns uint8 HxWx3."""
    strip = np.zeros((height, width, 3), dtype=np.float32)
    cell_w = max(1, width // nb)
    for b in range(nb):
        c = _KV_COLOURS[b % len(_KV_COLOURS)]
        x0 = b * cell_w
        x1 = width if b == nb - 1 else (b + 1) * cell_w
        strip[:, x0:x1] = c[None, None, :] * 255.0
    return np.clip(strip, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Orchestration.
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
                   help="Which transformer blocks to aggregate attention over.")
    p.add_argument("--alpha", type=float, default=0.6,
                   help="Overlay opacity (0=no overlay, 1=fully replace pixel RGB).")
    p.add_argument("--saturate", type=float, default=4.0,
                   help="Multiplier on raw attention before colour "
                        "multiply (attention probs rarely exceed ~0.3 "
                        "per KV-chunk, so bumping makes the colours "
                        "visible without over-saturating).")
    p.add_argument("--fps", type=int, default=20)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))

    # --- Build AR_refresh (per-pass) pipeline. ---
    pipe = ODEARRefreshPipeline(device, dtype=dtype)
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
    capture_layer_indices = [int(x) for x in args.capture_layers.split(",") if x.strip()]
    capture_layer_indices = [i for i in capture_layer_indices if 0 <= i < total_blocks]
    log.info("Capturing attention at layers %s (of %d total)", capture_layer_indices, total_blocks)

    attn_blocks = _install_qk_capture(base_dit, capture_layer_indices)
    action_per_frame = int(getattr(base_dit, "action_tokens_per_frame", 1))
    freqs = base_dit.freqs

    # --- Replicate AR_refresh's per-pass forward loop inline so we can
    #     capture Q/K at the FINAL denoise pass of each chunk. Mirrors
    #     ``ODEARRefreshPipeline._run_ar_refresh``. ---
    B = 1
    fifo_size = int(args.fifo_size)
    chunks_per_step = 1
    max_window_blocks = fifo_size + 1
    base_dit.num_frame_per_block = num_frame_per_block

    # Temporarily set local_attn_size to -1 for global attention (matches
    # ODEARRefreshPipeline.generate_ar_refresh).
    prev_local_attn_size = getattr(base_dit, "local_attn_size", -1)
    prev_max_attn_size = getattr(base_dit, "max_attention_size", None)
    base_dit.local_attn_size = -1
    for _, module in base_dit.named_modules():
        if hasattr(module, "local_attn_size"):
            try:
                module.local_attn_size = -1
            except Exception:
                pass

    scheduler = pipe.scheduler
    scheduler.sigmas = scheduler.sigmas.to(device)
    ts = pipe.denoising_step_list

    noise_seed = int(args.seed) + 1_000_003
    torch.manual_seed(noise_seed)
    torch.cuda.manual_seed(noise_seed)

    C = int(initial_latents_dev.shape[2])
    H = int(initial_latents_dev.shape[3])
    W = int(initial_latents_dev.shape[4])

    seed_chunks = seed_frames // num_frame_per_block
    fifo: List[torch.Tensor] = [initial_latents_dev]
    current_window_blocks = -1

    generated: List[torch.Tensor] = []
    attention_per_chunk: List[Dict[str, Any]] = []

    try:
        t0 = time.time()
        for chunk_idx in range(int(args.ar_gen_chunks)):
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

            fa_window = noisy_fa_full[:, ctx_global_frame_lo:cur_global_frame_hi].contiguous()
            t_ctx_vec = torch.full(
                [B, n_ctx_blocks * num_frame_per_block], 0.0,
                device=device, dtype=torch.float32,
            )

            current_noise = torch.randn(
                [B, num_frame_per_block, C, H, W],
                dtype=torch.float32, device=device,
            )
            x_cur = current_noise.to(dtype)

            pred_x0_window: Optional[torch.Tensor] = None
            for d_idx in range(int(ts.shape[0])):
                t_val = float(ts[d_idx].item())
                t_cur_vec = torch.full(
                    [B, num_frame_per_block], t_val,
                    device=device, dtype=torch.float32,
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
                        device=device, dtype=torch.float32,
                    )
                    x_cur = (
                        scheduler.add_noise(flat, flat_noise, flat_t)
                        .view(B, num_frame_per_block, C, H, W)
                        .to(dtype)
                    )
            assert pred_x0_window is not None

            # --- Compute per-layer attention at the FINAL denoise pass. ---
            attn_layers: List[np.ndarray] = []
            for layer_idx, blk in enumerate(attn_blocks):
                if not getattr(blk, "_capture_enabled", False):
                    continue
                q_raw = blk._captured_q
                k_raw = blk._captured_k
                if q_raw is None or k_raw is None:
                    continue
                attn_cur = _compute_attention_per_chunk(
                    blk,
                    q_raw, k_raw,
                    num_frames=window_frames,
                    spatial_h=30, spatial_w=52,
                    action_per_frame=action_per_frame,
                    freqs=freqs,
                    n_ctx_blocks=n_ctx_blocks,
                    num_frame_per_block=num_frame_per_block,
                )  # [cur_f, H_lat, W_lat, nb]
                attn_layers.append(attn_cur.numpy())
                blk._captured_q = None
                blk._captured_k = None
            if attn_layers:
                attn_mean = np.stack(attn_layers, axis=0).mean(axis=0)  # [cur_f, H, W, nb]
            else:
                attn_mean = np.zeros(
                    (num_frame_per_block, 30, 52, window_blocks), dtype=np.float32,
                )
            attention_per_chunk.append({
                "chunk_idx": int(chunk_idx),
                "window_blocks": int(window_blocks),
                "n_ctx_blocks": int(n_ctx_blocks),
                "attn": attn_mean,  # float32
            })

            cur_pred = pred_x0_window[:, n_ctx_blocks * num_frame_per_block:]
            generated.append(cur_pred.detach().to(torch.float32))

            log.info(
                "[attn_overlay] chunk %d/%d committed | window=%d blocks | "
                "attn shape %s | min/max %.4f / %.4f",
                chunk_idx + 1, int(args.ar_gen_chunks), window_blocks,
                tuple(attn_mean.shape), float(attn_mean.min()), float(attn_mean.max()),
            )

            if len(fifo) >= fifo_size:
                fifo.pop(0)
            fifo.append(cur_pred.to(dtype))

        log.info("Rollout done in %.1fs.", time.time() - t0)
    finally:
        base_dit.local_attn_size = prev_local_attn_size
        if prev_max_attn_size is not None:
            base_dit.max_attention_size = prev_max_attn_size
        for _, module in base_dit.named_modules():
            if hasattr(module, "local_attn_size"):
                try:
                    module.local_attn_size = prev_local_attn_size
                except Exception:
                    pass
            if hasattr(module, "max_attention_size") and prev_max_attn_size is not None:
                try:
                    module.max_attention_size = prev_max_attn_size
                except Exception:
                    pass
        base_dit.block_mask = None

    # --- Decode rollout to pixels. ---
    seed_real = initial_latents_dev[:, :seed_frames].to(torch.float32)
    full_latents = torch.cat([seed_real] + generated, dim=1)
    log.info("Decoding %d latent frames -> pixels ...", int(full_latents.shape[1]))
    pixel_video = pipe.decode_latents(full_latents)  # [T_pix, H_pix, W_pix, 3] uint8
    log.info("Pixel video shape: %s", pixel_video.shape)

    # Write raw mp4 (no overlay).
    raw_mp4 = out_dir / "rollout_raw.mp4"
    frames_to_mp4(pixel_video, str(raw_mp4), fps=float(args.fps))
    log.info("Wrote raw mp4 -> %s", raw_mp4)

    # --- Build overlay mp4. Per-chunk overlay only on the pixel frames
    #     corresponding to the current chunk's latent frames (not on the
    #     seed/preceding chunks). ---
    T_pix = pixel_video.shape[0]
    total_latent_frames = int(full_latents.shape[1])
    # VAE decoder outputs 1 pixel frame per latent at the start, then 4
    # pixel frames per subsequent latent (the "prepend dummy, drop first
    # pixel frame" convention). For our purposes the ratio is
    # T_pix / total_latent_frames — compute per-chunk pixel range by
    # proportion. The seed occupies the first `seed_frames` latent
    # frames; each gen chunk occupies the next npb latent frames.
    def _pixel_range_for_latents(lat_lo: int, lat_hi: int) -> Tuple[int, int]:
        p_lo = int(round(lat_lo / total_latent_frames * T_pix))
        p_hi = int(round(lat_hi / total_latent_frames * T_pix))
        return max(0, p_lo), min(T_pix, p_hi)

    overlay_video = pixel_video.copy()
    for entry in attention_per_chunk:
        chunk_idx = int(entry["chunk_idx"])
        attn = entry["attn"]  # [cur_f, H_lat, W_lat, nb]
        # Latent range for this chunk's current frames (global).
        lat_lo = (seed_chunks + chunk_idx) * num_frame_per_block
        lat_hi = lat_lo + num_frame_per_block
        p_lo, p_hi = _pixel_range_for_latents(lat_lo, lat_hi)
        if p_hi <= p_lo:
            continue
        chunk_pixels = pixel_video[p_lo:p_hi]
        overlaid = _render_overlay_for_chunk(
            chunk_pixels, attn, alpha=float(args.alpha), saturate=float(args.saturate),
        )
        overlay_video[p_lo:p_hi] = overlaid

    # Add legend strip at the top.
    max_nb = max(entry["attn"].shape[-1] for entry in attention_per_chunk)
    legend = _colour_legend_strip(max_nb, height=28, width=overlay_video.shape[2])
    legend_tile = np.broadcast_to(
        legend[None, ...], (overlay_video.shape[0],) + legend.shape,
    )
    overlay_with_legend = np.concatenate([legend_tile, overlay_video], axis=1)

    overlay_mp4 = out_dir / "rollout_attention.mp4"
    frames_to_mp4(overlay_with_legend, str(overlay_mp4), fps=float(args.fps))
    log.info("Wrote overlay mp4 -> %s", overlay_mp4)

    # --- Save raw attention tensors for offline re-analysis. ---
    attn_pt = out_dir / "attention.pt"
    torch.save({
        "meta": {
            "ts": ts.detach().cpu().tolist(),
            "num_frame_per_block": num_frame_per_block,
            "capture_layers": capture_layer_indices,
            "fifo_size": int(args.fifo_size),
            "ar_gen_chunks": int(args.ar_gen_chunks),
            "ar_initial_chunks": int(args.ar_initial_chunks),
            "seed": int(args.seed),
            "rank_zarr": args.rank_zarr,
            "kv_colours": _KV_COLOURS.tolist(),
            "alpha": float(args.alpha),
            "saturate": float(args.saturate),
        },
        "chunks": attention_per_chunk,
    }, attn_pt)
    log.info("Saved attention -> %s", attn_pt)

    with (out_dir / "manifest.json").open("w") as fh:
        json.dump({
            "raw_mp4": str(raw_mp4),
            "overlay_mp4": str(overlay_mp4),
            "attention_pt": str(attn_pt),
            "kv_colours": _KV_COLOURS.tolist(),
            "alpha": float(args.alpha),
            "saturate": float(args.saturate),
        }, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
