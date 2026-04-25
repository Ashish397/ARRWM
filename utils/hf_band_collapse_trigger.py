#!/usr/bin/env python3
"""HF-band (r/r_max ∈ [0.4, 1.0]) power trigger for collapse detection.

User observation: on per-ride radial-power plots, the student's
spectrum systematically falls below GT's, and the gap is strongest
past r/r_max ≈ 0.4. Integrate power over that band per chunk →
one scalar; track it over time.

Two trigger variants:

  abs   — ``HF_stu / HF_gt < T_abs``           (needs GT; ideal when GT
          is available at DMD training time)
  rel   — ``med_w(HF_stu_c / HF_stu_seed) < T_rel``  (GT-free; uses the
          rollout's own seed chunk as baseline)

For every ride under ``--root``, produces a per-ride ``hf_band_trigger.png``
and a combined ``combined_hf_band.png``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def hf_band_power(chunk: torch.Tensor, r_lo: float = 0.4, r_hi: float = 1.0) -> float:
    """Integrate 2D-FFT magnitude² over radii in [r_lo, r_hi] · r_max.
    Averaged across T, C of the chunk."""
    T, C, H, W = chunk.shape
    X = chunk.float()
    fft = torch.fft.fftshift(torch.fft.fft2(X, dim=(-2, -1)), dim=(-2, -1))
    mag_sq = fft.abs() ** 2
    fy = torch.fft.fftshift(torch.fft.fftfreq(H)).abs()
    fx = torch.fft.fftshift(torch.fft.fftfreq(W)).abs()
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    r = (yy ** 2 + xx ** 2).sqrt()
    r_max = r.max().item()
    rn = r / r_max
    mask = ((rn >= r_lo) & (rn <= r_hi)).to(mag_sq.dtype)
    # broadcast mask across T, C.
    band = (mag_sq * mask).mean(dim=(0, 1))  # [H, W]
    # Integrate over spatial freq — use mean over masked pixels to make
    # the magnitude invariant to how many bins fall in the band.
    px_in_band = mask.sum().item()
    return float(band.sum().item() / max(px_in_band, 1.0))


def rolling_median(vals, window=3):
    out = []
    for i in range(len(vals)):
        lo = max(0, i - window + 1)
        out.append(float(np.median(vals[lo:i + 1])))
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--r_lo", type=float, default=0.4)
    p.add_argument("--r_hi", type=float, default=1.0)
    p.add_argument("--window", type=int, default=3)
    p.add_argument("--abs_threshold", type=float, default=0.5,
                   help="Fire when med_w(HF_stu / HF_gt) < this.")
    p.add_argument("--rel_threshold", type=float, default=0.5,
                   help="Fire when med_w(HF_stu / HF_stu_seed) < this.")
    p.add_argument("--num_frame_per_block", type=int, default=3)
    p.add_argument("--video_fps", type=int, default=20)
    p.add_argument("--vae_temporal_upsample", type=int, default=4)
    return p.parse_args()


def process_root(root: Path, args):
    secs_per_chunk = args.num_frame_per_block * args.vae_temporal_upsample / args.video_fps
    ride_dirs = sorted([d for d in root.iterdir() if d.is_dir() and (d / "latents.pt").exists()])
    all_hf_stu: dict[str, list[float]] = {}
    all_hf_gt:  dict[str, list[float]] = {}
    fires: dict[str, dict] = {}

    for rd in ride_dirs:
        blob = torch.load(rd / "latents.pt", map_location="cpu", weights_only=False)
        npb = args.num_frame_per_block
        stu, gt = blob["ar_refresh"], blob["gt"]
        n_chunks = stu.shape[1] // npb
        times = [c * secs_per_chunk for c in range(n_chunks)]

        hf_stu = [hf_band_power(stu[0, c*npb:(c+1)*npb], args.r_lo, args.r_hi) for c in range(n_chunks)]
        hf_gt  = [hf_band_power(gt[0,  c*npb:(c+1)*npb], args.r_lo, args.r_hi) for c in range(n_chunks)]
        all_hf_stu[rd.name] = hf_stu
        all_hf_gt[rd.name]  = hf_gt

        # Absolute (GT-referenced) trigger: med_w(HF_stu / HF_gt).
        ratio_abs = [hf_stu[c] / max(hf_gt[c], 1e-12) for c in range(n_chunks)]
        med_abs   = rolling_median(ratio_abs, args.window)
        # Relative (GT-free) trigger: med_w(HF_stu / HF_stu_seed).
        seed = max(hf_stu[0], 1e-12)
        ratio_rel = [hf_stu[c] / seed for c in range(n_chunks)]
        med_rel   = rolling_median(ratio_rel, args.window)

        fire_abs = next((c for c in range(args.window, n_chunks) if med_abs[c] < args.abs_threshold), None)
        fire_rel = next((c for c in range(args.window, n_chunks) if med_rel[c] < args.rel_threshold), None)
        fires[rd.name] = {
            "abs_fire_chunk": fire_abs,
            "abs_fire_time_s": fire_abs * secs_per_chunk if fire_abs is not None else None,
            "rel_fire_chunk": fire_rel,
            "rel_fire_time_s": fire_rel * secs_per_chunk if fire_rel is not None else None,
            "hf_stu_seed": float(seed),
            "hf_stu_end": float(hf_stu[-1]),
            "hf_gt_end": float(hf_gt[-1]),
            "hf_stu_over_gt_end": float(hf_stu[-1] / max(hf_gt[-1], 1e-12)),
        }
        print(f"[{root.name}/{rd.name}] seed_HF={seed:.4f}  "
              f"ABS fire @{fire_abs} ({fire_abs*secs_per_chunk:.1f}s)" if fire_abs is not None
              else f"[{root.name}/{rd.name}] seed_HF={seed:.4f}  ABS never")

        # Per-ride plot: raw HF power, ratio_abs, ratio_rel (3 panels).
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        ax = axes[0]
        ax.plot(times, hf_stu, "-o", color="C0", lw=2, ms=3, label="ar_refresh HF-band")
        ax.plot(times, hf_gt,  "-o", color="k",  lw=2, ms=3, label="GT HF-band")
        ax.set_yscale("log"); ax.set_ylabel(f"HF-band power (r/r_max∈[{args.r_lo},{args.r_hi}])")
        ax.set_title(f"{rd.name}  |  seed_HF_stu={seed:.4f}  seed_HF_gt={hf_gt[0]:.4f}")
        ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")

        ax = axes[1]
        ax.plot(times, ratio_abs, "-o", color="C0", lw=1.2, ms=3, alpha=0.5, label="HF_stu / HF_gt")
        ax.plot(times, med_abs,   "-",  color="C0", lw=2, label=f"median{args.window}")
        ax.axhline(args.abs_threshold, color="red", ls="--", lw=1.1, label=f"abs trigger @ {args.abs_threshold}")
        ax.axhline(1.0, color="grey", lw=0.6)
        if fire_abs is not None:
            ax.axvline(fire_abs * secs_per_chunk, color="red", ls=":", lw=1.3,
                       label=f"fires @ {fire_abs*secs_per_chunk:.1f}s")
        ax.set_ylabel("HF_stu / HF_gt")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax = axes[2]
        ax.plot(times, ratio_rel, "-o", color="C0", lw=1.2, ms=3, alpha=0.5, label="HF_stu / HF_stu_seed")
        ax.plot(times, med_rel,   "-",  color="C0", lw=2, label=f"median{args.window}")
        ax.axhline(args.rel_threshold, color="orange", ls="--", lw=1.1, label=f"rel trigger @ {args.rel_threshold}")
        ax.axhline(1.0, color="grey", lw=0.6)
        if fire_rel is not None:
            ax.axvline(fire_rel * secs_per_chunk, color="orange", ls=":", lw=1.3,
                       label=f"fires @ {fire_rel*secs_per_chunk:.1f}s")
        ax.set_xlabel("video time (s)"); ax.set_ylabel("HF_stu / HF_stu_seed")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(rd / "hf_band_trigger.png", dpi=130); plt.close(fig)

    # Combined overlay.
    n_max = max(len(v) for v in all_hf_stu.values())
    times = [c * secs_per_chunk for c in range(n_max)]
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    cmap = plt.get_cmap("tab10")

    for i, (tag, hf) in enumerate(all_hf_stu.items()):
        c = cmap(i % 10)
        axes[0].plot(times[:len(hf)], hf,                "-",  color=c, lw=2,   label=f"{tag} student")
        axes[0].plot(times[:len(hf)], all_hf_gt[tag],    "--", color=c, lw=1.2, alpha=0.7)
        ratio_abs = [hf[j] / max(all_hf_gt[tag][j], 1e-12) for j in range(len(hf))]
        seed = max(hf[0], 1e-12)
        ratio_rel = [hf[j] / seed for j in range(len(hf))]
        med_abs = rolling_median(ratio_abs, args.window)
        med_rel = rolling_median(ratio_rel, args.window)
        axes[1].plot(times[:len(hf)], med_abs, "-", color=c, lw=2, label=f"{tag}")
        axes[2].plot(times[:len(hf)], med_rel, "-", color=c, lw=2, label=f"{tag}")
        if fires[tag]["abs_fire_chunk"] is not None:
            axes[1].axvline(fires[tag]["abs_fire_time_s"], color=c, ls=":", lw=1.2, alpha=0.7)
        if fires[tag]["rel_fire_chunk"] is not None:
            axes[2].axvline(fires[tag]["rel_fire_time_s"], color=c, ls=":", lw=1.2, alpha=0.7)

    axes[0].set_yscale("log")
    axes[0].set_ylabel(f"HF-band power r∈[{args.r_lo},{args.r_hi}]")
    axes[0].set_title("Raw HF-band power — solid=student, dashed=GT")
    axes[0].legend(fontsize=7, ncol=2, loc="best"); axes[0].grid(alpha=0.3, which="both")

    axes[1].axhline(args.abs_threshold, color="red", ls="--", lw=1.1, label=f"threshold @ {args.abs_threshold}")
    axes[1].axhline(1.0, color="grey", lw=0.6)
    axes[1].set_ylabel("HF_stu / HF_gt (median w)")
    axes[1].set_title("GT-referenced trigger (needs GT at DMD training time)")
    axes[1].legend(fontsize=7, ncol=2, loc="best"); axes[1].grid(alpha=0.3)

    axes[2].axhline(args.rel_threshold, color="orange", ls="--", lw=1.1, label=f"threshold @ {args.rel_threshold}")
    axes[2].axhline(1.0, color="grey", lw=0.6)
    axes[2].set_xlabel("video time (s)")
    axes[2].set_ylabel("HF_stu / HF_stu_seed (median w)")
    axes[2].set_title("GT-free seed-relative trigger")
    axes[2].legend(fontsize=7, ncol=2, loc="best"); axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(root / "combined_hf_band.png", dpi=130); plt.close(fig)
    print(f"wrote {root/'combined_hf_band.png'}")

    with (root / "hf_band_summary.json").open("w") as fh:
        json.dump({
            "fires": fires,
            "r_lo": args.r_lo, "r_hi": args.r_hi,
            "abs_threshold": args.abs_threshold, "rel_threshold": args.rel_threshold,
            "window": args.window,
            "secs_per_chunk": secs_per_chunk,
            "hf_stu": all_hf_stu, "hf_gt": all_hf_gt,
        }, fh, indent=2)
    return fires


def main():
    args = parse_args()
    all_fires = {}
    for r in args.roots:
        root = Path(r)
        fires = process_root(root, args)
        all_fires[root.name] = fires
    print("\n=== Cross-root summary ===")
    for root_name, fires in all_fires.items():
        print(f"  [{root_name}]")
        for ride, f in fires.items():
            print(f"    {ride}  ABS@{f['abs_fire_time_s']}s  REL@{f['rel_fire_time_s']}s  "
                  f"HF_stu/HF_gt(end)={f['hf_stu_over_gt_end']:.3f}")


if __name__ == "__main__":
    main()
