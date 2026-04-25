#!/usr/bin/env python3
"""Spatial-FFT spectrum vs time for AR_refresh rollouts.

For every ride subdir under ``<root>`` that has a ``latents.pt``:
  1. Compute per-chunk 2D spatial FFT (averaged across channels and the
     3 frames in a chunk), radially bin the magnitude spectrum.
  2. Produce:
       - ``spectrum_time.png``  — heatmap [chunk × radial-bin] log-power,
          two panels (GT vs student), plus a total-power-vs-time curve
          below.
       - ``radial_samples.png`` — line plot of radial power spectrum at
          seed / mid / final chunks, GT vs student side-by-side.
  3. Writes a combined figure ``combined_power_vs_time.png`` showing
     every ride's total / HF / LF power trajectory overlaid.

Quantities:
  * total     = sum of |FFT|**2 over all spatial frequencies (≈ chunk
    variance; close to Parseval companion of RMS²)
  * HF_frac   = fraction of total power sitting at radii >= 0.5 r_max
  * spectral_centroid = Σ(r · P(r)) / Σ P(r)  — higher = sharper texture
  * spectral_entropy  = Shannon entropy of the normalised radial
    distribution — low values mean power concentrated in a few bands
    (collapse / repeating pattern).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True,
                   help="Directory containing per-ride subdirs with latents.pt.")
    p.add_argument("--num_frame_per_block", type=int, default=3)
    p.add_argument("--video_fps", type=int, default=20)
    p.add_argument("--vae_temporal_upsample", type=int, default=4)
    p.add_argument("--hf_cutoff", type=float, default=0.5,
                   help="Fraction of r_max above which counts as HF (for hf_frac).")
    p.add_argument("--n_radial_bins", type=int, default=32)
    return p.parse_args()


def radial_power(chunk: torch.Tensor, n_bins: int) -> tuple[np.ndarray, torch.Tensor]:
    """chunk: [T, C, H, W] → radial power spectrum, averaged over T*C.

    Returns
    -------
    r_centers : np.ndarray[n_bins]
        Radial bin centres (normalised to [0, 1]).
    power     : torch.Tensor[n_bins]
        Mean power in each bin.
    """
    T, C, H, W = chunk.shape
    X = chunk.float()
    fft = torch.fft.fftshift(torch.fft.fft2(X, dim=(-2, -1)), dim=(-2, -1))
    mag_sq = fft.abs() ** 2  # [T, C, H, W]
    # Build radial distance map.
    fy = torch.fft.fftshift(torch.fft.fftfreq(H)).abs()
    fx = torch.fft.fftshift(torch.fft.fftfreq(W)).abs()
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    r = (yy ** 2 + xx ** 2).sqrt()
    r_max = r.max().item()
    rn = (r / r_max).cpu().numpy().ravel()
    flat = mag_sq.mean(dim=(0, 1)).cpu().numpy().ravel()  # average over T, C

    edges = np.linspace(0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(np.searchsorted(edges[1:-1], rn, side="right"), 0, n_bins - 1)
    power = np.zeros(n_bins, dtype=np.float64)
    count = np.zeros(n_bins, dtype=np.int64)
    np.add.at(power, bin_idx, flat)
    np.add.at(count, bin_idx, 1)
    power = power / np.maximum(count, 1)
    return centers, torch.from_numpy(power)


def summarise_chunk(chunk: torch.Tensor, n_bins: int, hf_cutoff: float):
    centers, power = radial_power(chunk, n_bins)
    p = power.numpy()
    total = float(p.sum())
    # HF fraction on the radial bins.
    hf_mask = centers >= hf_cutoff
    hf = float(p[hf_mask].sum())
    hf_frac = hf / (total + 1e-12)
    # Spectral centroid & entropy.
    pnorm = p / (total + 1e-12)
    centroid = float((centers * pnorm).sum())
    with np.errstate(divide="ignore"):
        ent = float(-np.sum(pnorm * np.log(pnorm + 1e-12)))
    return {
        "centers": centers, "power": p,
        "total": total, "hf_frac": hf_frac,
        "centroid": centroid, "entropy": ent,
    }


def main():
    args = parse_args()
    root = Path(args.root)
    secs_per_chunk = args.num_frame_per_block * args.vae_temporal_upsample / args.video_fps
    ride_dirs = sorted([d for d in root.iterdir() if d.is_dir() and (d / "latents.pt").exists()])
    if not ride_dirs:
        raise SystemExit(f"no ride subdirs with latents.pt under {root}")

    all_totals:   dict[str, dict] = {}   # {ride: {"gt": [..], "stu": [..]}}
    all_hf:       dict[str, dict] = {}
    all_centroid: dict[str, dict] = {}
    all_entropy:  dict[str, dict] = {}

    for rd in ride_dirs:
        blob = torch.load(rd / "latents.pt", map_location="cpu", weights_only=False)
        if "ar_refresh" not in blob or "gt" not in blob:
            print(f"skip {rd.name}: missing ar_refresh/gt"); continue
        npb = args.num_frame_per_block
        stu = blob["ar_refresh"]; gt = blob["gt"]
        n_chunks = stu.shape[1] // npb
        times = [c * secs_per_chunk for c in range(n_chunks)]

        stats_stu = [summarise_chunk(stu[0, c*npb:(c+1)*npb], args.n_radial_bins, args.hf_cutoff)
                     for c in range(n_chunks)]
        stats_gt  = [summarise_chunk(gt[0, c*npb:(c+1)*npb],  args.n_radial_bins, args.hf_cutoff)
                     for c in range(n_chunks)]

        # Build heatmap matrices [n_chunks, n_bins].
        spec_stu = np.stack([s["power"] for s in stats_stu], axis=0)  # rows=chunks
        spec_gt  = np.stack([s["power"] for s in stats_gt],  axis=0)
        centers = stats_stu[0]["centers"]

        # Per-ride heatmap + total-power plot.
        fig = plt.figure(figsize=(12, 7.5))
        gs = fig.add_gridspec(3, 2, height_ratios=[3, 3, 2], hspace=0.42, wspace=0.18)
        ax_gt  = fig.add_subplot(gs[0, :])
        ax_stu = fig.add_subplot(gs[1, :], sharex=ax_gt)
        ax_tot = fig.add_subplot(gs[2, :], sharex=ax_gt)

        # Log-power heatmaps — row axis = time, col axis = spatial frequency.
        def show_spec(ax, spec, title):
            spec_log = np.log10(spec.T + 1e-12)  # [bins, chunks], imshow rows top→bottom
            vmin, vmax = np.percentile(spec_log, [1, 99])
            im = ax.imshow(spec_log, aspect="auto", origin="lower",
                           extent=[times[0], times[-1], 0, 1],
                           vmin=vmin, vmax=vmax, cmap="viridis")
            ax.set_ylabel("spatial freq  (r / r_max)")
            ax.set_title(title)
            return im
        im1 = show_spec(ax_gt,  spec_gt,  f"{rd.name} — GT  log10(radial power)")
        im2 = show_spec(ax_stu, spec_stu, f"{rd.name} — AR_refresh  log10(radial power)")
        fig.colorbar(im1, ax=ax_gt,  pad=0.01)
        fig.colorbar(im2, ax=ax_stu, pad=0.01)

        tot_stu = np.array([s["total"] for s in stats_stu])
        tot_gt  = np.array([s["total"] for s in stats_gt])
        hf_stu  = np.array([s["hf_frac"] for s in stats_stu])
        hf_gt   = np.array([s["hf_frac"] for s in stats_gt])
        ax_tot.plot(times, np.log10(tot_gt + 1e-12),  "-o", color="k", lw=2, ms=3,
                    label="GT log10(total power)")
        ax_tot.plot(times, np.log10(tot_stu + 1e-12), "-o", color="C0", lw=2, ms=3,
                    label="student log10(total power)")
        ax_tot2 = ax_tot.twinx()
        ax_tot2.plot(times, hf_gt,  "--", color="k",  lw=1.2, alpha=0.7,
                     label="GT HF fraction (r ≥ 0.5)")
        ax_tot2.plot(times, hf_stu, "--", color="C0", lw=1.2, alpha=0.9,
                     label="student HF fraction (r ≥ 0.5)")
        ax_tot.set_xlabel("video time (s)")
        ax_tot.set_ylabel("log10(total power)")
        ax_tot2.set_ylabel("HF fraction")
        ax_tot.grid(alpha=0.3)
        h1, l1 = ax_tot.get_legend_handles_labels()
        h2, l2 = ax_tot2.get_legend_handles_labels()
        ax_tot.legend(h1 + h2, l1 + l2, fontsize=7, loc="best", ncol=2)
        fig.savefig(rd / "spectrum_time.png", dpi=130)
        plt.close(fig)
        print(f"wrote {rd/'spectrum_time.png'}")

        # Radial-spectrum samples at seed / mid / final chunk.
        mid = n_chunks // 2; last = n_chunks - 1
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for label, idx, ls in [("seed (c=0)", 0, "-"), (f"mid (c={mid})", mid, "--"),
                                (f"final (c={last})", last, ":")]:
            ax.plot(centers, stats_gt[idx]["power"],  color="k",  ls=ls, lw=1.5, label=f"GT {label}")
            ax.plot(centers, stats_stu[idx]["power"], color="C0", ls=ls, lw=1.5, label=f"student {label}")
        ax.set_xlabel("r / r_max"); ax.set_ylabel("radial power (linear)")
        ax.set_yscale("log")
        ax.set_title(f"{rd.name} — radial power spectra: seed vs mid vs final")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(rd / "radial_samples.png", dpi=130); plt.close(fig)

        # Stash for combined plots.
        all_totals[rd.name]   = {"stu": tot_stu.tolist(), "gt": tot_gt.tolist(), "t": times}
        all_hf[rd.name]       = {"stu": hf_stu.tolist(),  "gt": hf_gt.tolist()}
        all_centroid[rd.name] = {"stu": [s["centroid"] for s in stats_stu],
                                 "gt":  [s["centroid"] for s in stats_gt]}
        all_entropy[rd.name]  = {"stu": [s["entropy"]  for s in stats_stu],
                                 "gt":  [s["entropy"]  for s in stats_gt]}

    # Combined plots.
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    cmap = plt.get_cmap("tab10")
    axes_flat = axes.flatten()
    panels = [
        (axes[0, 0], all_totals,   lambda x: np.log10(np.array(x) + 1e-12), "log10(total spectral power)"),
        (axes[0, 1], all_hf,       lambda x: np.array(x),                   "HF fraction (r ≥ 0.5)"),
        (axes[1, 0], all_centroid, lambda x: np.array(x),                   "spectral centroid"),
        (axes[1, 1], all_entropy,  lambda x: np.array(x),                   "spectral entropy (bits)"),
    ]
    for i, (ax, data, xform, ylabel) in enumerate(panels):
        for j, (ride, d) in enumerate(data.items()):
            c = cmap(j % 10)
            ts = all_totals[ride]["t"]
            ax.plot(ts, xform(d["stu"]), "-",  color=c, lw=2,   label=f"{ride} student" if i == 0 else None)
            ax.plot(ts, xform(d["gt"]),  "--", color=c, lw=1.2, alpha=0.7,
                    label=f"{ride} GT" if i == 0 else None)
        ax.set_ylabel(ylabel); ax.grid(alpha=0.3)
    axes[1, 0].set_xlabel("video time (s)"); axes[1, 1].set_xlabel("video time (s)")
    axes[0, 0].legend(fontsize=6, ncol=2, loc="best")
    fig.suptitle("Spectral summaries vs time — every ride overlaid (bold=student, dashed=GT)")
    fig.tight_layout()
    fig.savefig(root / "combined_power_vs_time.png", dpi=130); plt.close(fig)
    print(f"wrote {root/'combined_power_vs_time.png'}")

    with (root / "spectral_summary.json").open("w") as fh:
        json.dump({
            "totals": all_totals, "hf_frac": all_hf,
            "centroid": all_centroid, "entropy": all_entropy,
            "hf_cutoff": float(args.hf_cutoff),
        }, fh, indent=2)
    print(f"wrote {root/'spectral_summary.json'}")


if __name__ == "__main__":
    main()
