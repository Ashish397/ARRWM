#!/usr/bin/env python3
"""Re-generate RMS trigger plots from existing latents.pt files.

Walks ``<root>/<ride>/latents.pt`` for every ride subdir, recomputes
RMS / seed-relative-RMS / rolling medians, and writes per-ride
``rms_trigger.png`` plus a combined ``combined_rms.png`` with both
absolute and seed-relative panels.

Use this after extending the analysis without re-running rollouts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def rolling_median(vals, window=3):
    out = []
    for i in range(len(vals)):
        lo = max(0, i - window + 1)
        out.append(float(np.median(vals[lo:i + 1])))
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--abs_threshold", type=float, default=0.70)
    p.add_argument("--rel_threshold", type=float, default=0.85)
    p.add_argument("--window", type=int, default=3)
    p.add_argument("--video_fps", type=int, default=20)
    p.add_argument("--vae_temporal_upsample", type=int, default=4)
    p.add_argument("--num_frame_per_block", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    secs_per_chunk = args.num_frame_per_block * args.vae_temporal_upsample / args.video_fps

    ride_dirs = sorted([d for d in root.iterdir() if d.is_dir() and (d / "latents.pt").exists()])
    if not ride_dirs:
        raise SystemExit(f"no ride subdirs with latents.pt under {root}")

    all_rms, all_rms_gt, all_fires = {}, {}, {}

    for rd in ride_dirs:
        blob = torch.load(rd / "latents.pt", map_location="cpu", weights_only=False)
        # Some older payloads don't carry rms lists — recompute from latents.
        if "rms" in blob and "rms_gt" in blob:
            rms, rms_gt = list(blob["rms"]), list(blob["rms_gt"])
        else:
            npb = args.num_frame_per_block
            ar = blob["ar_refresh"]; gt = blob["gt"]
            n = ar.shape[1] // npb
            rms    = [ar[0, c*npb:(c+1)*npb].float().pow(2).mean().sqrt().item() for c in range(n)]
            rms_gt = [gt[0, c*npb:(c+1)*npb].float().pow(2).mean().sqrt().item() for c in range(n)]
        n_chunks = len(rms)
        med    = rolling_median(rms,    args.window)
        med_gt = rolling_median(rms_gt, args.window)
        seed = rms[0] if rms[0] > 1e-6 else 1e-6
        seed_gt = rms_gt[0] if rms_gt[0] > 1e-6 else 1e-6
        seed_ratio    = [r / seed    for r in rms]
        seed_ratio_gt = [r / seed_gt for r in rms_gt]
        med_rel    = rolling_median(seed_ratio,    args.window)
        med_rel_gt = rolling_median(seed_ratio_gt, args.window)

        fire_abs     = next((c for c in range(args.window, n_chunks) if med[c]        < args.abs_threshold), None)
        fire_abs_gt  = next((c for c in range(args.window, n_chunks) if med_gt[c]     < args.abs_threshold), None)
        fire_rel     = next((c for c in range(args.window, n_chunks) if med_rel[c]    < args.rel_threshold), None)
        fire_rel_gt  = next((c for c in range(args.window, n_chunks) if med_rel_gt[c] < args.rel_threshold), None)

        all_rms[rd.name] = rms
        all_rms_gt[rd.name] = rms_gt
        all_fires[rd.name] = {
            "seed_rms": float(seed),
            "abs_fire_chunk": fire_abs,
            "abs_fire_time_s": fire_abs * secs_per_chunk if fire_abs is not None else None,
            "abs_gt_fire_chunk": fire_abs_gt,
            "rel_fire_chunk": fire_rel,
            "rel_fire_time_s": fire_rel * secs_per_chunk if fire_rel is not None else None,
            "rel_gt_fire_chunk": fire_rel_gt,
        }

        # Per-ride plot.
        times = [c * secs_per_chunk for c in range(n_chunks)]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(times, rms,    "-o", color="#1f77b4", label="ar_refresh RMS", lw=1.3, ms=3, alpha=0.5)
        ax1.plot(times, med,    "-",  color="#1f77b4", label=f"ar_refresh median{args.window}", lw=2)
        ax1.plot(times, rms_gt, "-o", color="#000000", label="GT RMS",       lw=1.3, ms=3, alpha=0.5)
        ax1.plot(times, med_gt, "-",  color="#000000", label=f"GT median{args.window}", lw=2)
        ax1.axhline(args.abs_threshold, color="red", ls="--", lw=1.1,
                    label=f"abs trigger @ {args.abs_threshold}")
        if fire_abs is not None:
            ax1.axvline(fire_abs * secs_per_chunk, color="red", ls=":", lw=1.3,
                        label=f"abs fires @ {fire_abs * secs_per_chunk:.1f}s")
        ax1.set_ylabel("chunk RMS (absolute)")
        ax1.set_title(f"Ride {rd.name} | seed_RMS={seed:.3f}")
        ax1.legend(fontsize=7, loc="best"); ax1.grid(alpha=0.3)

        ax2.plot(times, seed_ratio,    "-o", color="#1f77b4", label="ar_refresh RMS/seed", lw=1.3, ms=3, alpha=0.5)
        ax2.plot(times, med_rel,       "-",  color="#1f77b4", label=f"ar_refresh median{args.window}", lw=2)
        ax2.plot(times, seed_ratio_gt, "-o", color="#000000", label="GT RMS/seed", lw=1.3, ms=3, alpha=0.5)
        ax2.plot(times, med_rel_gt,    "-",  color="#000000", label=f"GT median{args.window}", lw=2)
        ax2.axhline(args.rel_threshold, color="orange", ls="--", lw=1.1,
                    label=f"rel trigger @ {args.rel_threshold}")
        ax2.axhline(1.0, color="grey", ls="-", lw=0.6)
        if fire_rel is not None:
            ax2.axvline(fire_rel * secs_per_chunk, color="orange", ls=":", lw=1.3,
                        label=f"rel fires @ {fire_rel * secs_per_chunk:.1f}s")
        ax2.set_xlabel("video time (s)"); ax2.set_ylabel("RMS / RMS_seed")
        ax2.legend(fontsize=7, loc="best"); ax2.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(rd / "rms_trigger.png", dpi=130); plt.close(fig)
        print(f"wrote {rd/'rms_trigger.png'} — seed_RMS={seed:.3f} "
              f"| ABS@{fire_abs} REL@{fire_rel}")

    # Combined overlay plot.
    n_chunks_plot = max(len(v) for v in all_rms.values())
    times = [c * secs_per_chunk for c in range(n_chunks_plot)]
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    cmap = plt.get_cmap("tab10")
    for i, tag in enumerate(all_rms.keys()):
        c = cmap(i % 10)
        rms = all_rms[tag]; rms_gt = all_rms_gt[tag]
        seed = rms[0] if rms[0] > 1e-6 else 1e-6
        seed_gt = rms_gt[0] if rms_gt[0] > 1e-6 else 1e-6
        rel    = [r / seed    for r in rms]
        rel_gt = [r / seed_gt for r in rms_gt]
        axes[0].plot(times[:len(rms)], rms,    "-",  color=c, lw=2,   label=f"{tag} student")
        axes[0].plot(times[:len(rms)], rms_gt, "--", color=c, lw=1.2, alpha=0.7, label=f"{tag} GT")
        axes[1].plot(times[:len(rms)], rel,    "-",  color=c, lw=2,   label=f"{tag} student")
        axes[1].plot(times[:len(rms)], rel_gt, "--", color=c, lw=1.2, alpha=0.7, label=f"{tag} GT")
        if all_fires[tag]["rel_fire_chunk"] is not None:
            axes[1].axvline(all_fires[tag]["rel_fire_time_s"], color=c, ls=":", lw=1.2, alpha=0.8)
    axes[0].axhline(args.abs_threshold, color="red", ls="--", lw=1.1,
                    label=f"abs trigger @ {args.abs_threshold}")
    axes[0].set_ylabel("chunk RMS (absolute)")
    axes[0].set_title("Absolute RMS — seed magnitude varies per ride → threshold unreliable")
    axes[0].legend(fontsize=6, loc="best", ncol=2); axes[0].grid(alpha=0.3)
    axes[1].axhline(args.rel_threshold, color="orange", ls="--", lw=1.1,
                    label=f"rel trigger @ {args.rel_threshold}")
    axes[1].axhline(1.0, color="grey", lw=0.6)
    axes[1].set_xlabel("video time (s)"); axes[1].set_ylabel("RMS / RMS_seed")
    axes[1].set_title("Seed-relative — auto-calibrated to each ride's own seed chunk")
    axes[1].legend(fontsize=6, loc="best", ncol=2); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(root / "combined_rms.png", dpi=130); plt.close(fig)
    print(f"wrote {root/'combined_rms.png'}")

    with (root / "summary.json").open("w") as fh:
        json.dump({
            "fires": all_fires,
            "threshold_abs": float(args.abs_threshold),
            "threshold_rel": float(args.rel_threshold),
            "window": int(args.window),
            "secs_per_chunk": secs_per_chunk,
        }, fh, indent=2)
    print(f"wrote {root/'summary.json'}")


if __name__ == "__main__":
    main()
