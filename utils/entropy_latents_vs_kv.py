#!/usr/bin/env python3
"""Compare the information content of the raw latents vs the append
baseline's stored K and V tensors.

Three entropy measures per tensor:

  1. **Shannon entropy (256-bin histogram)**. Discretises values into
     256 equal-width bins over the per-tensor min/max range, computes
     ``H = -Σ p log₂ p`` in bits per scalar. Simple empirical measure
     that makes no distributional assumption.

  2. **Gaussian entropy**. Assumes the tensor's values are i.i.d.
     Gaussian; returns ``H = 0.5 · log₂(2πe σ²)`` bits per scalar,
     using the tensor's own std. A quick sanity check — for well-
     behaved activations this is within ~0.1 bits of the histogram
     entropy; a big gap indicates heavy-tailed / sparse structure.

  3. **Effective rank (stable rank)** of the flattened matrix
     [rows × cols] = ``||X||_F² / ||X||_2²``. Upper-bounded by
     ``min(rows, cols)``. Tells us how many effective dimensions the
     tensor spans — a high-rank tensor packs independent info across
     more axes; a low-rank one lives on a narrow subspace.

  4. **Total info** = per-scalar entropy × number of scalars. Useful
     because K/V has ~10× more scalars than the latent at deeper
     layers, so the "info per scalar" and "total info" can diverge.

Per-chunk + per-layer breakdown, one row per chunk, one column per
variant (latent / K_layerN / V_layerN).

Output:

    <output_dir>/
      entropy_table.json      — numeric per-chunk per-variant stats
      entropy_bars.png        — bar chart (total bits) per variant
      entropy_per_layer.png   — curves of K/V entropy across layers per chunk
      effective_rank.png      — rank curves

Usage::

    python utils/entropy_latents_vs_kv.py \\
        --latents /home/ashish/ARRWM/eval/latents_three_variants_<ts>/latents.pt \\
        --append_kv /home/ashish/ARRWM/eval/kv_capture_append_<ts>/kv_capture.pt \\
        --output_dir /home/ashish/ARRWM/eval/entropy_<ts>
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


def shannon_entropy_histogram(x: torch.Tensor, n_bins: int = 256) -> float:
    """Shannon entropy in bits/scalar using an n_bins equi-width
    histogram over the per-tensor min/max range."""
    arr = x.detach().float().flatten().cpu().numpy()
    if arr.size == 0:
        return 0.0
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(arr, bins=bins)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts.astype(np.float64) / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def gaussian_entropy(x: torch.Tensor) -> float:
    """Per-scalar Gaussian entropy (bits). ``H = 0.5·log₂(2πe σ²)``."""
    arr = x.detach().float().flatten()
    std = arr.std().item()
    if std < 1e-12:
        return 0.0
    return 0.5 * math.log2(2.0 * math.pi * math.e * std * std)


def stable_rank(x: torch.Tensor, max_rows: int = 8192) -> float:
    """Stable rank ``||X||_F² / ||X||_2²`` of the flattened matrix
    ``[rows, cols]`` where rows = samples, cols = feature dim. For a
    tensor ``[B, T, C, H, W]`` we reshape to ``[B*T, C*H*W]``; for a
    KV tensor ``[B, seq, dim]`` we reshape to ``[B*seq, dim]``.

    For very large matrices we randomly subsample rows (seeded) to
    keep the spectral norm computation tractable.
    """
    x = x.detach().float()
    if x.ndim == 5:
        B, T, C, H, W = x.shape
        mat = x.reshape(B * T, C * H * W)
    elif x.ndim == 3:
        mat = x.reshape(-1, x.shape[-1])
    else:
        mat = x.reshape(x.shape[0], -1)

    rows, cols = mat.shape
    if rows > max_rows:
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(rows, generator=g)[:max_rows]
        mat = mat[idx]
        rows = max_rows
    fro_sq = float((mat * mat).sum().item())
    # Largest singular value (spectral norm) via power iteration on a
    # CPU float32 copy.
    try:
        s = torch.linalg.svdvals(mat)
        top_sq = float((s[0] ** 2).item())
    except Exception as e:  # noqa: BLE001
        log.warning("svdvals failed (%s); using .norm() fallback", e)
        top_sq = float((mat.norm(dim=1).max() ** 2).item())
    if top_sq < 1e-12:
        return 0.0
    return fro_sq / top_sq


def describe_tensor(x: torch.Tensor) -> Dict[str, Any]:
    arr = x.detach().float().flatten()
    return {
        "numel": int(arr.numel()),
        "shape": list(x.shape),
        "mean": float(arr.mean().item()),
        "std":  float(arr.std().item()),
        "min":  float(arr.min().item()),
        "max":  float(arr.max().item()),
        "H_hist_bits_per_scalar": shannon_entropy_histogram(x),
        "H_gauss_bits_per_scalar": gaussian_entropy(x),
        "stable_rank": stable_rank(x),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--latents", type=str, required=True,
                   help="Path to latents.pt from save_latents_three_variants.py")
    p.add_argument("--append_kv", type=str, required=True,
                   help="Path to kv_capture.pt from analyse_append_baseline_kv.py")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--n_bins", type=int, default=256)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading %s", args.latents)
    latents = torch.load(args.latents, map_location="cpu", weights_only=False)
    log.info("Loading %s", args.append_kv)
    kv = torch.load(args.append_kv, map_location="cpu", weights_only=False)

    meta = latents["meta"]
    npb = int(meta["num_frame_per_block"])
    seed_frames = int(meta["seed_frames"])
    gen_chunks = int(meta["ar_gen_chunks"])
    capture_layers = list(kv["meta"]["capture_layers"])

    # --- Per-chunk analysis ---
    #   latent_chunk:  slice [1, npb, C, H, W] from the append_baseline latents
    #   K_chunk, V_chunk: for each captured layer, slice the CURRENT
    #     chunk's frames (last npb) from the window tensor saved by
    #     the append capture.
    ap_lat = latents["append_baseline"]  # [1, T, C, H, W]

    per_chunk: Dict[int, Dict[str, Any]] = {}
    for chunk_entry in kv["chunks"]:
        cidx = int(chunk_entry["chunk_idx"])
        wf = int(chunk_entry["window_frames"])
        # Append latent for this chunk's frames.
        lat_lo = seed_frames + cidx * npb
        lat_hi = lat_lo + npb
        latent_chunk = ap_lat[:, lat_lo:lat_hi].contiguous()
        row: Dict[str, Any] = {"latent": describe_tensor(latent_chunk)}
        for L in capture_layers:
            if L not in chunk_entry["layers"]:
                continue
            k_full = chunk_entry["layers"][L]["k"]  # [1, seq, dim]
            v_full = chunk_entry["layers"][L]["v"]
            # Slice LAST npb frames' worth of K/V (= current chunk's).
            seq = k_full.shape[1]
            frame_seqlen = seq // wf
            cur_slice_lo = (wf - 1) * frame_seqlen  # last block's start is at (wf-npb)*frame_seqlen
            # Correction: the CURRENT CHUNK is the last block of npb
            # frames, starting at (wf - npb) * frame_seqlen.
            cur_slice_lo = (wf - npb) * frame_seqlen
            cur_slice_hi = wf * frame_seqlen
            k_cur = k_full[:, cur_slice_lo:cur_slice_hi].contiguous()
            v_cur = v_full[:, cur_slice_lo:cur_slice_hi].contiguous()
            row[f"K_L{L}"] = describe_tensor(k_cur)
            row[f"V_L{L}"] = describe_tensor(v_cur)
            # ALSO capture stats on the full window K/V (includes
            # context) so we can reason about info present across the
            # cache the query attends against.
            row[f"K_L{L}_window"] = describe_tensor(k_full)
            row[f"V_L{L}_window"] = describe_tensor(v_full)
        per_chunk[cidx] = row
        log.info("chunk %d: latent H_hist=%.3f numel=%d | K_L%d H_hist=%.3f numel=%d",
                 cidx,
                 row["latent"]["H_hist_bits_per_scalar"], row["latent"]["numel"],
                 capture_layers[0],
                 row[f"K_L{capture_layers[0]}"]["H_hist_bits_per_scalar"],
                 row[f"K_L{capture_layers[0]}"]["numel"])

    # --- JSON dump ---
    json_path = out_dir / "entropy_table.json"
    with json_path.open("w") as fh:
        json.dump({
            "meta": {
                "latents_pt": args.latents,
                "append_kv_pt": args.append_kv,
                "capture_layers": capture_layers,
                "ar_gen_chunks": gen_chunks,
                "num_frame_per_block": npb,
                "ride": meta.get("rank_zarr", "?"),
            },
            "per_chunk": per_chunk,
        }, fh, indent=2, default=str)
    log.info("Wrote %s", json_path)

    # --- Plots ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib missing — skipping plots.")
        return

    chunks = sorted(per_chunk.keys())
    # Plot 1: Total bits per tensor per chunk (current chunk only).
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, key in zip(axes, ("H_hist_bits_per_scalar", "H_gauss_bits_per_scalar")):
        lat_bits = [per_chunk[c]["latent"][key] * per_chunk[c]["latent"]["numel"] for c in chunks]
        ax.plot(chunks, lat_bits, marker="o", label="latent (current chunk)", lw=2.2)
        for L in capture_layers:
            k_bits = [per_chunk[c].get(f"K_L{L}", {"H_hist_bits_per_scalar": 0, "H_gauss_bits_per_scalar": 0, "numel": 0})[key] *
                      per_chunk[c].get(f"K_L{L}", {"numel": 0})["numel"] for c in chunks]
            v_bits = [per_chunk[c].get(f"V_L{L}", {"H_hist_bits_per_scalar": 0, "H_gauss_bits_per_scalar": 0, "numel": 0})[key] *
                      per_chunk[c].get(f"V_L{L}", {"numel": 0})["numel"] for c in chunks]
            ax.plot(chunks, k_bits, marker="s", linestyle="--", label=f"K L{L} (current chunk)")
            ax.plot(chunks, v_bits, marker="x", linestyle=":",  label=f"V L{L} (current chunk)")
        ax.set_yscale("log")
        ax.set_xlabel("gen chunk idx")
        ax.set_ylabel("total info (bits)")
        ax.set_title(key.replace("_", " "))
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Total information per tensor (bits = bits/scalar × numel)", y=1.02)
    fig.tight_layout()
    p1 = out_dir / "entropy_bars.png"
    fig.savefig(p1, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", p1)

    # Plot 2: Per-scalar entropy across layers per chunk.
    fig, ax = plt.subplots(figsize=(9, 5))
    lat_h = [per_chunk[c]["latent"]["H_hist_bits_per_scalar"] for c in chunks]
    ax.plot(chunks, lat_h, marker="o", color="black", lw=2.2, label="latent")
    colours = plt.cm.viridis(np.linspace(0.2, 0.9, len(capture_layers)))
    for colour, L in zip(colours, capture_layers):
        k_h = [per_chunk[c].get(f"K_L{L}", {"H_hist_bits_per_scalar": 0})["H_hist_bits_per_scalar"] for c in chunks]
        v_h = [per_chunk[c].get(f"V_L{L}", {"H_hist_bits_per_scalar": 0})["H_hist_bits_per_scalar"] for c in chunks]
        ax.plot(chunks, k_h, marker="s", linestyle="--", color=colour, label=f"K L{L}")
        ax.plot(chunks, v_h, marker="x", linestyle=":",  color=colour, label=f"V L{L}")
    ax.set_xlabel("gen chunk idx")
    ax.set_ylabel("H_hist (bits / scalar)")
    ax.set_title("Per-scalar Shannon entropy (256-bin histogram)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    p2 = out_dir / "entropy_per_layer.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", p2)

    # Plot 3: Stable rank.
    fig, ax = plt.subplots(figsize=(9, 5))
    lat_rk = [per_chunk[c]["latent"]["stable_rank"] for c in chunks]
    ax.plot(chunks, lat_rk, marker="o", color="black", lw=2.2, label="latent")
    for colour, L in zip(colours, capture_layers):
        k_rk = [per_chunk[c].get(f"K_L{L}", {"stable_rank": 0})["stable_rank"] for c in chunks]
        v_rk = [per_chunk[c].get(f"V_L{L}", {"stable_rank": 0})["stable_rank"] for c in chunks]
        ax.plot(chunks, k_rk, marker="s", linestyle="--", color=colour, label=f"K L{L}")
        ax.plot(chunks, v_rk, marker="x", linestyle=":",  color=colour, label=f"V L{L}")
    ax.set_xlabel("gen chunk idx")
    ax.set_ylabel("stable rank (||F||² / ||2||²)")
    ax.set_title("Effective-rank / stable-rank of flattened tensor")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    p3 = out_dir / "effective_rank.png"
    fig.tight_layout()
    fig.savefig(p3, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", p3)

    # --- Quick text summary to stdout ---
    log.info("---- SUMMARY (chunk 3, middle of rollout) ----")
    if 3 in per_chunk:
        row = per_chunk[3]
        for name, stats in row.items():
            total_bits = stats["H_hist_bits_per_scalar"] * stats["numel"]
            log.info(
                "  %-16s  numel=%10d  H_hist=%.3f b/scalar  H_gauss=%.3f b/scalar  "
                "rank=%6.1f  total=%.2e bits",
                name, stats["numel"],
                stats["H_hist_bits_per_scalar"], stats["H_gauss_bits_per_scalar"],
                stats["stable_rank"], total_bits,
            )


if __name__ == "__main__":
    main()
