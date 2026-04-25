#!/usr/bin/env python3
"""Per-chunk stable rank for collapse detection.

Loads the saved ``latents.pt`` from ``save_latents_three_variants.py``
(keys: ``gt``, ``ar_refresh``, ``ar_refresh_once``, ``append_baseline``,
each [1, T_total, C, H, W]) and for every variant + GT computes the
stable rank ``(||X||_F / ||X||_2)^2`` of each 3-frame chunk's
spatial-fold matrix ``X = chunk.reshape(T*C, H*W)``.

Stable rank is a soft, differentiable analogue of rank: equals the
number of "effective" non-zero singular values. Low stable rank means
the chunk's spatial patterns have collapsed onto a low-dim subspace
(blur, repeating tiles, frozen frames).

Outputs:
    <output_dir>/stable_rank_per_chunk.png      — abs value per chunk
    <output_dir>/stable_rank_ratio_per_chunk.png — ratio vs chunk 0
    <output_dir>/stable_rank.json                — numbers
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def chunk_stable_rank(chunk: torch.Tensor) -> float:
    """chunk: [T, C, H, W] (no batch) — return scalar stable rank."""
    T, C, H, W = chunk.shape
    X = chunk.reshape(T * C, H * W).float()
    fro_sq = (X * X).sum().item()
    spec = torch.linalg.matrix_norm(X, ord=2).item()
    return fro_sq / (spec * spec + 1e-12)


def chunk_hf_fraction(chunk: torch.Tensor, hf_cutoff: float = 0.5) -> float:
    """High-frequency spectral fraction. ``chunk``: [T, C, H, W].

    Compute the per-frame, per-channel 2D spatial FFT magnitude, build a
    radial distance map, and return the fraction of total power that
    sits at radii >= ``hf_cutoff * r_max``. Lower values indicate
    spatial blur / collapse.
    """
    T, C, H, W = chunk.shape
    X = chunk.float()
    fft = torch.fft.fftshift(torch.fft.fft2(X, dim=(-2, -1)), dim=(-2, -1))
    mag = fft.abs() ** 2  # power
    fy = torch.fft.fftshift(torch.fft.fftfreq(H)).abs()
    fx = torch.fft.fftshift(torch.fft.fftfreq(W)).abs()
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    r = (yy ** 2 + xx ** 2).sqrt()
    r_max = r.max().item()
    mask_hf = (r >= hf_cutoff * r_max).float()
    total = mag.sum().item()
    hf = (mag * mask_hf).sum().item()
    return hf / (total + 1e-12)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--latents_pt", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--num_frame_per_block", type=int, default=3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    blob = torch.load(args.latents_pt, map_location="cpu", weights_only=False)
    npb = int(args.num_frame_per_block)

    variants = ["gt", "ar_refresh", "ar_refresh_once", "append_baseline"]
    colors = {
        "gt":               "#000000",
        "ar_refresh":       "#1f77b4",
        "ar_refresh_once":  "#d62728",
        "append_baseline":  "#2ca02c",
    }

    ranks: dict[str, list[float]] = {}
    hfs:   dict[str, list[float]] = {}
    for v in variants:
        if v not in blob:
            print(f"[skip] {v} not in latents.pt"); continue
        lat = blob[v]
        assert lat.dim() == 5 and lat.shape[0] == 1, lat.shape
        T = lat.shape[1]
        n_chunks = T // npb
        chunk_ranks: list[float] = []
        chunk_hfs:   list[float] = []
        for c in range(n_chunks):
            chunk = lat[0, c * npb : (c + 1) * npb]
            chunk_ranks.append(chunk_stable_rank(chunk))
            chunk_hfs.append(chunk_hf_fraction(chunk))
        ranks[v] = chunk_ranks
        hfs[v]   = chunk_hfs
        print(f"{v:20s}  rank={[round(r, 2) for r in chunk_ranks]}")
        print(f"{'':20s}  hf  ={[round(h, 4) for h in chunk_hfs]}")

    # --- plot 1: absolute stable rank per chunk ---
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for v, rs in ranks.items():
        xs = list(range(len(rs)))
        ax.plot(xs, rs, "-o", color=colors.get(v, "#888"), label=v, lw=2, ms=5)
    ax.axvline(0.5, color="grey", ls="--", lw=0.8)
    ax.text(0.0, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 1, " seed",
            color="grey", fontsize=9, va="top")
    ax.set_xlabel("chunk index (0 = GT seed, 1..7 = generated)")
    ax.set_ylabel("stable rank  (||X||_F / ||X||_2)^2")
    ax.set_title("per-chunk stable rank — collapse detector candidate")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p1 = out / "stable_rank_per_chunk.png"
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    print(f"wrote {p1}")

    # --- plot 2: ratio vs chunk 0 ---
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for v, rs in ranks.items():
        r0 = rs[0]
        ratios = [r / r0 for r in rs]
        xs = list(range(len(rs)))
        ax.plot(xs, ratios, "-o", color=colors.get(v, "#888"), label=v, lw=2, ms=5)
    ax.axhline(1.0, color="grey", ls="-", lw=0.8)
    ax.axhline(0.6, color="red",  ls="--", lw=1.0, alpha=0.6,
               label="proposed abort threshold (0.6×)")
    ax.set_xlabel("chunk index (0 = GT seed)")
    ax.set_ylabel("stable rank ratio vs chunk 0")
    ax.set_title("auto-calibrated collapse signal — ratio against the rollout's seed chunk")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p2 = out / "stable_rank_ratio_per_chunk.png"
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    print(f"wrote {p2}")

    # --- plot 3: HF fraction per chunk ---
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for v, hs in hfs.items():
        xs = list(range(len(hs)))
        ax.plot(xs, hs, "-o", color=colors.get(v, "#888"), label=v, lw=2, ms=5)
    ax.set_xlabel("chunk index (0 = GT seed, 1..7 = generated)")
    ax.set_ylabel("HF spectral power fraction (radii >= 0.5 r_max)")
    ax.set_title("per-chunk high-frequency content — collapse drains HF")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p3 = out / "hf_fraction_per_chunk.png"
    fig.savefig(p3, dpi=130)
    plt.close(fig)
    print(f"wrote {p3}")

    # --- plot 4: gen-vs-GT ratio for both metrics. DMD has GT at training
    # time (rollout overlays an existing ride), so we can compute ratios.
    if "gt" in ranks:
        gt_r = ranks["gt"]
        gt_h = hfs["gt"]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
        for v in variants:
            if v == "gt" or v not in ranks: continue
            xs = list(range(len(ranks[v])))
            r_ratio = [ranks[v][i] / gt_r[i] for i in xs]
            h_ratio = [hfs[v][i]   / gt_h[i] for i in xs]
            axes[0].plot(xs, r_ratio, "-o", color=colors.get(v, "#888"), label=v, lw=2, ms=5)
            axes[1].plot(xs, h_ratio, "-o", color=colors.get(v, "#888"), label=v, lw=2, ms=5)
        for ax, title, ylab in zip(
            axes,
            ["stable rank ratio gen / GT", "HF fraction ratio gen / GT"],
            ["rank_gen / rank_GT", "hf_gen / hf_GT"],
        ):
            ax.axhline(1.0, color="grey", ls="-", lw=0.8)
            ax.set_xlabel("chunk index"); ax.set_ylabel(ylab); ax.set_title(title)
            ax.legend(loc="best", fontsize=9); ax.grid(alpha=0.3)
        fig.suptitle("DMD-style auto-calibrated collapse signals (gen / GT per chunk)")
        fig.tight_layout()
        p4 = out / "gen_vs_gt_ratio.png"
        fig.savefig(p4, dpi=130)
        plt.close(fig)
        print(f"wrote {p4}")

    with (out / "stable_rank.json").open("w") as fh:
        json.dump({
            "stable_rank_per_chunk": ranks,
            "hf_fraction_per_chunk": hfs,
            "ratio_vs_chunk0": {v: [r / rs[0] for r in rs] for v, rs in ranks.items()},
        }, fh, indent=2)
    print(f"wrote {out / 'stable_rank.json'}")


if __name__ == "__main__":
    main()
