#!/usr/bin/env python3
"""Per-chunk Shannon entropy of latent values vs time.

For each chunk c (3 frames × 16 channels × 60 × 104 = 299 520 values),
build a 256-bin histogram between [-q, q] (with q a high-percentile
clip) and compute H = -Σ p_i log2 p_i  in bits/scalar.

Collapse intuitions:
  * frozen / blur collapse  → values concentrate near zero  → entropy ↓
  * saturation collapse     → values broaden (high variance) → entropy ≈
    log of the histogram bin count (caps out)
  * healthy student         → entropy tracks GT's

Reads ride subdirs under each ``--root`` (latents.pt expected).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def shannon_entropy(chunk: torch.Tensor, n_bins: int = 256, clip_q: float = 0.999) -> float:
    """Histogram-based Shannon entropy (bits / scalar)."""
    x = chunk.float().flatten()
    q = float(torch.quantile(x.abs(), clip_q).item())
    q = max(q, 1e-6)
    edges = torch.linspace(-q, q, n_bins + 1)
    hist = torch.histc(x.clamp(-q, q), bins=n_bins, min=-q, max=q)
    p = hist / hist.sum().clamp(min=1)
    p = p[p > 0]
    return float(-(p * torch.log2(p)).sum().item())


def gaussian_entropy(chunk: torch.Tensor) -> float:
    """h = 0.5 log2(2πe σ²)  bits / scalar — closed-form Gaussian entropy."""
    x = chunk.float().flatten()
    var = float(x.var(unbiased=False).item())
    return 0.5 * math.log2(2 * math.pi * math.e * max(var, 1e-12))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--num_frame_per_block", type=int, default=3)
    p.add_argument("--video_fps", type=int, default=20)
    p.add_argument("--vae_temporal_upsample", type=int, default=4)
    p.add_argument("--n_bins", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()
    secs_per_chunk = args.num_frame_per_block * args.vae_temporal_upsample / args.video_fps

    all_h_stu: dict[str, list[float]] = {}
    all_h_gt:  dict[str, list[float]] = {}
    all_g_stu: dict[str, list[float]] = {}
    all_g_gt:  dict[str, list[float]] = {}

    for root_s in args.roots:
        root = Path(root_s)
        ride_dirs = sorted([d for d in root.iterdir() if d.is_dir() and (d / "latents.pt").exists()])
        for rd in ride_dirs:
            blob = torch.load(rd / "latents.pt", map_location="cpu", weights_only=False)
            npb = args.num_frame_per_block
            stu, gt = blob["ar_refresh"], blob["gt"]
            n_chunks = stu.shape[1] // npb
            h_stu = [shannon_entropy(stu[0, c*npb:(c+1)*npb], args.n_bins) for c in range(n_chunks)]
            h_gt  = [shannon_entropy(gt[0,  c*npb:(c+1)*npb], args.n_bins) for c in range(n_chunks)]
            g_stu = [gaussian_entropy(stu[0, c*npb:(c+1)*npb]) for c in range(n_chunks)]
            g_gt  = [gaussian_entropy(gt[0,  c*npb:(c+1)*npb]) for c in range(n_chunks)]
            times = [c * secs_per_chunk for c in range(n_chunks)]

            key = f"{root.name}/{rd.name}"
            all_h_stu[key] = h_stu; all_h_gt[key] = h_gt
            all_g_stu[key] = g_stu; all_g_gt[key] = g_gt
            print(f"[{rd.name}]  H_stu seed/end={h_stu[0]:.2f}/{h_stu[-1]:.2f}  "
                  f"H_gt seed/end={h_gt[0]:.2f}/{h_gt[-1]:.2f}  "
                  f"|  GaussH_stu seed/end={g_stu[0]:.2f}/{g_stu[-1]:.2f}  "
                  f"GaussH_gt seed/end={g_gt[0]:.2f}/{g_gt[-1]:.2f}")

            # Per-ride plot.
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
            ax1.plot(times, h_stu, "-o", color="C0", lw=2, ms=3, label="ar_refresh Shannon H")
            ax1.plot(times, h_gt,  "-o", color="k",  lw=2, ms=3, label="GT Shannon H")
            ax1.set_ylabel("Shannon entropy (bits / scalar)")
            ax1.set_title(f"{rd.name} — per-chunk entropy")
            ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
            ax2.plot(times, g_stu, "-o", color="C0", lw=2, ms=3, label="ar_refresh Gaussian H")
            ax2.plot(times, g_gt,  "-o", color="k",  lw=2, ms=3, label="GT Gaussian H")
            ax2.set_xlabel("video time (s)"); ax2.set_ylabel("Gaussian entropy (bits / scalar)")
            ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(rd / "entropy_per_chunk.png", dpi=130); plt.close(fig)

        # Combined per-root.
        keys = [k for k in all_h_stu.keys() if k.startswith(root.name + "/")]
        if not keys: continue
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        cmap = plt.get_cmap("tab10")
        for i, key in enumerate(keys):
            c = cmap(i % 10)
            ride = key.split("/", 1)[1]
            ts = [j * secs_per_chunk for j in range(len(all_h_stu[key]))]
            axes[0].plot(ts, all_h_stu[key], "-",  color=c, lw=2, label=f"{ride} student")
            axes[0].plot(ts, all_h_gt[key],  "--", color=c, lw=1.2, alpha=0.7)
            axes[1].plot(ts, all_g_stu[key], "-",  color=c, lw=2, label=f"{ride} student")
            axes[1].plot(ts, all_g_gt[key],  "--", color=c, lw=1.2, alpha=0.7)
        axes[0].set_ylabel("Shannon entropy (bits / scalar)")
        axes[0].set_title(f"{root.name} — per-chunk Shannon entropy (solid=student, dashed=GT)")
        axes[0].legend(fontsize=7, ncol=2); axes[0].grid(alpha=0.3)
        axes[1].set_xlabel("video time (s)"); axes[1].set_ylabel("Gaussian entropy")
        axes[1].set_title("Per-chunk Gaussian entropy (= 0.5·log₂(2πe σ²); equivalent to log·RMS)")
        axes[1].legend(fontsize=7, ncol=2); axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(root / "combined_entropy.png", dpi=130); plt.close(fig)
        print(f"wrote {root/'combined_entropy.png'}")

    # Cross-root combined plot.
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    cmap = plt.get_cmap("tab20")
    keys = list(all_h_stu.keys())
    for i, key in enumerate(keys):
        c = cmap(i % 20)
        ride_label = key.split("/", 1)[1]
        ts = [j * secs_per_chunk for j in range(len(all_h_stu[key]))]
        axes[0].plot(ts, all_h_stu[key], "-",  color=c, lw=1.6, label=ride_label)
        axes[0].plot(ts, all_h_gt[key],  "--", color=c, lw=1.0, alpha=0.6)
        axes[1].plot(ts, all_g_stu[key], "-",  color=c, lw=1.6, label=ride_label)
        axes[1].plot(ts, all_g_gt[key],  "--", color=c, lw=1.0, alpha=0.6)
    axes[0].set_ylabel("Shannon H (bits/scalar)")
    axes[0].set_title("All 8 rides — Shannon entropy (solid=student, dashed=GT)")
    axes[1].set_xlabel("video time (s)"); axes[1].set_ylabel("Gaussian H (bits/scalar)")
    axes[1].set_title("All 8 rides — Gaussian entropy")
    for ax in axes:
        ax.legend(fontsize=6, ncol=2, loc="best"); ax.grid(alpha=0.3)
    fig.tight_layout()
    out_combined = Path(args.roots[0]).parent / "combined_entropy_all.png"
    fig.savefig(out_combined, dpi=130); plt.close(fig)
    print(f"wrote {out_combined}")

    summary = {
        "n_bins": args.n_bins,
        "per_ride": {
            key: {
                "shannon_stu": all_h_stu[key], "shannon_gt": all_h_gt[key],
                "gaussian_stu": all_g_stu[key], "gaussian_gt": all_g_gt[key],
            } for key in keys
        },
    }
    with (Path(args.roots[0]).parent / "entropy_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
