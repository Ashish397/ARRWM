#!/usr/bin/env python3
"""Cumulative MAE of student (ar_refresh) vs GT latents, per chunk.

Simple signal: for each chunk c, MAE_c = mean(|stu_c - gt_c|).
Cumulative across chunks: cum_c = sum_{k=0..c} MAE_k.

A healthy rollout keeps MAE small and bounded; a collapsing one
accumulates error quickly. The cumulative curve bends upward at the
moment of collapse — a one-threshold abort rule ("cum > T") or
slope-based rule ("ΔMAE > T_slope") is trivial to apply.

Walks ride subdirs under each ``--root``, reuses ``latents.pt``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--num_frame_per_block", type=int, default=3)
    p.add_argument("--video_fps", type=int, default=20)
    p.add_argument("--vae_temporal_upsample", type=int, default=4)
    p.add_argument("--mae_threshold", type=float, default=1.5,
                   help="Per-chunk MAE abort threshold.")
    p.add_argument("--cum_threshold", type=float, default=8.0,
                   help="Cumulative MAE abort threshold.")
    return p.parse_args()


def main():
    args = parse_args()
    secs_per_chunk = args.num_frame_per_block * args.vae_temporal_upsample / args.video_fps

    all_mae: dict[str, dict] = {}
    all_cum: dict[str, dict] = {}

    for root_s in args.roots:
        root = Path(root_s)
        ride_dirs = sorted([d for d in root.iterdir() if d.is_dir() and (d / "latents.pt").exists()])
        for rd in ride_dirs:
            blob = torch.load(rd / "latents.pt", map_location="cpu", weights_only=False)
            npb = args.num_frame_per_block
            stu, gt = blob["ar_refresh"], blob["gt"]
            n_chunks = stu.shape[1] // npb
            mae = []
            for c in range(n_chunks):
                a = stu[0, c*npb:(c+1)*npb].float()
                b = gt[0,  c*npb:(c+1)*npb].float()
                mae.append((a - b).abs().mean().item())
            cum = np.cumsum(mae).tolist()
            times = [c * secs_per_chunk for c in range(n_chunks)]

            first_mae_fire = next((c for c in range(n_chunks) if mae[c] > args.mae_threshold), None)
            first_cum_fire = next((c for c in range(n_chunks) if cum[c] > args.cum_threshold), None)

            key = f"{root.name}/{rd.name}"
            all_mae[key] = {"mae": mae, "t": times, "fire_chunk": first_mae_fire}
            all_cum[key] = {"cum": cum, "t": times, "fire_chunk": first_cum_fire}

            # Per-ride plot.
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
            ax1.plot(times, mae, "-o", color="C0", lw=2, ms=3)
            ax1.axhline(args.mae_threshold, color="red", ls="--", lw=1.1,
                        label=f"per-chunk MAE threshold @ {args.mae_threshold}")
            if first_mae_fire is not None:
                ax1.axvline(first_mae_fire * secs_per_chunk, color="red", ls=":", lw=1.3,
                            label=f"fires @ {first_mae_fire * secs_per_chunk:.1f}s")
            ax1.set_ylabel("MAE_c = mean(|stu - gt|)")
            ax1.set_title(f"{rd.name} — per-chunk MAE vs GT")
            ax1.legend(fontsize=7); ax1.grid(alpha=0.3)

            ax2.plot(times, cum, "-o", color="C0", lw=2, ms=3)
            ax2.axhline(args.cum_threshold, color="orange", ls="--", lw=1.1,
                        label=f"cumulative MAE threshold @ {args.cum_threshold}")
            if first_cum_fire is not None:
                ax2.axvline(first_cum_fire * secs_per_chunk, color="orange", ls=":", lw=1.3,
                            label=f"fires @ {first_cum_fire * secs_per_chunk:.1f}s")
            ax2.set_xlabel("video time (s)"); ax2.set_ylabel("cumulative MAE")
            ax2.legend(fontsize=7); ax2.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(rd / "cumulative_mae.png", dpi=130); plt.close(fig)
            print(f"[{rd.name}]  MAE_end={mae[-1]:.3f}  cum_end={cum[-1]:.1f}  "
                  f"mae_fire@{first_mae_fire}  cum_fire@{first_cum_fire}")

        # Combined per-root.
        keys = [k for k in all_cum.keys() if k.startswith(root.name + "/")]
        if not keys: continue
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        cmap = plt.get_cmap("tab10")
        for i, key in enumerate(keys):
            c = cmap(i % 10)
            ride = key.split("/", 1)[1]
            ts = all_mae[key]["t"]
            axes[0].plot(ts, all_mae[key]["mae"], "-o", color=c, lw=2, ms=3, label=ride)
            axes[1].plot(ts, all_cum[key]["cum"], "-o", color=c, lw=2, ms=3, label=ride)
            if all_mae[key]["fire_chunk"] is not None:
                axes[0].axvline(all_mae[key]["fire_chunk"] * secs_per_chunk, color=c, ls=":", lw=1.0, alpha=0.6)
            if all_cum[key]["fire_chunk"] is not None:
                axes[1].axvline(all_cum[key]["fire_chunk"] * secs_per_chunk, color=c, ls=":", lw=1.0, alpha=0.6)
        axes[0].axhline(args.mae_threshold, color="red", ls="--", lw=1.1,
                        label=f"per-chunk threshold @ {args.mae_threshold}")
        axes[1].axhline(args.cum_threshold, color="orange", ls="--", lw=1.1,
                        label=f"cumulative threshold @ {args.cum_threshold}")
        axes[0].set_ylabel("MAE_c"); axes[0].set_title(f"{root.name} — per-chunk MAE")
        axes[1].set_xlabel("video time (s)"); axes[1].set_ylabel("cumulative MAE")
        axes[1].set_title(f"{root.name} — cumulative MAE (simple abort signal)")
        for ax in axes:
            ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(root / "combined_cumulative_mae.png", dpi=130); plt.close(fig)
        print(f"wrote {root/'combined_cumulative_mae.png'}")

    # Cross-root combined plot.
    all_keys = list(all_cum.keys())
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    cmap = plt.get_cmap("tab20")
    for i, key in enumerate(all_keys):
        c = cmap(i % 20)
        ts = all_mae[key]["t"]
        ride_label = key.split("/", 1)[1]
        axes[0].plot(ts, all_mae[key]["mae"], "-",  color=c, lw=1.6, label=ride_label)
        axes[1].plot(ts, all_cum[key]["cum"], "-",  color=c, lw=1.6, label=ride_label)
    axes[0].axhline(args.mae_threshold, color="red", ls="--", lw=1.0, label=f"threshold @ {args.mae_threshold}")
    axes[1].axhline(args.cum_threshold, color="orange", ls="--", lw=1.0, label=f"threshold @ {args.cum_threshold}")
    axes[0].set_ylabel("per-chunk MAE"); axes[0].set_title("All 8 rides — per-chunk MAE vs GT")
    axes[1].set_xlabel("video time (s)"); axes[1].set_ylabel("cumulative MAE")
    axes[1].set_title("All 8 rides — cumulative MAE vs GT")
    for ax in axes:
        ax.legend(fontsize=6, ncol=2, loc="best"); ax.grid(alpha=0.3)
    fig.tight_layout()
    out_combined = Path(args.roots[0]).parent / "combined_cumulative_mae_all.png"
    fig.savefig(out_combined, dpi=130); plt.close(fig)
    print(f"wrote {out_combined}")

    summary = {
        "mae_threshold": args.mae_threshold,
        "cum_threshold": args.cum_threshold,
        "per_ride": {
            key: {
                "mae": all_mae[key]["mae"],
                "cum": all_cum[key]["cum"],
                "mae_fire_chunk": all_mae[key]["fire_chunk"],
                "cum_fire_chunk": all_cum[key]["fire_chunk"],
                "mae_fire_s": (all_mae[key]["fire_chunk"] * secs_per_chunk)
                              if all_mae[key]["fire_chunk"] is not None else None,
                "cum_fire_s": (all_cum[key]["fire_chunk"] * secs_per_chunk)
                              if all_cum[key]["fire_chunk"] is not None else None,
            } for key in all_keys
        }
    }
    with (Path(args.roots[0]).parent / "cumulative_mae_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
