#!/usr/bin/env python3
"""Plot per-chunk RMS rate-of-change to mark the collapse event.

The absolute RMS of all variants drifts gently, but the *first
difference* ``RMS_c - RMS_{c-1}`` shows a sharp simultaneous drop at
the moment of visible collapse — far larger than GT's natural
chunk-to-chunk variation. Threshold the ratio ``|ΔRMS_c| /
EMA(|ΔRMS_GT|)`` against e.g. ``3.0`` and you have a one-scalar,
near-zero-cost collapse trigger that auto-calibrates to the rollout's
own scene-motion magnitude.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--latents_pt", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_frame_per_block", type=int, default=3)
    p.add_argument("--video_fps", type=int, default=20)
    p.add_argument("--vae_temporal_upsample", type=int, default=4)
    return p.parse_args()


def chunk_rms(chunk: torch.Tensor) -> float:
    return chunk.float().pow(2).mean().sqrt().item()


def main():
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    blob = torch.load(args.latents_pt, map_location="cpu", weights_only=False)
    npb = int(args.num_frame_per_block)
    secs_per_chunk = npb * args.vae_temporal_upsample / args.video_fps  # 0.6 by default

    variants = ["gt", "ar_refresh", "ar_refresh_once", "append_baseline"]
    colors = {
        "gt":               "#000000",
        "ar_refresh":       "#1f77b4",
        "ar_refresh_once":  "#d62728",
        "append_baseline":  "#2ca02c",
    }

    rms: dict[str, list[float]] = {}
    for v in variants:
        if v not in blob: continue
        lat = blob[v]
        T = lat.shape[1]; n_chunks = T // npb
        rms[v] = [chunk_rms(lat[0, c * npb : (c + 1) * npb]) for c in range(n_chunks)]

    drms: dict[str, list[float]] = {v: [0.0] + [vals[c] - vals[c-1] for c in range(1, len(vals))]
                                    for v, vals in rms.items()}
    n_chunks = len(rms["gt"])
    times = [c * secs_per_chunk for c in range(n_chunks)]

    # GT rolling baseline: take |ΔRMS_GT| EMA, span ~5 chunks.
    span = 5
    abs_dgt = np.abs(np.array(drms["gt"]))
    ema = np.zeros_like(abs_dgt)
    alpha = 2.0 / (span + 1)
    ema[0] = abs_dgt[0]
    for i in range(1, len(abs_dgt)):
        ema[i] = alpha * abs_dgt[i] + (1 - alpha) * ema[i-1]
    # Floor to avoid divide-by-zero.
    ema = np.maximum(ema, 1e-3)

    # Variant RMS / seed-RMS ratio — auto-calibrated, doesn't depend on GT.
    seed_ratio: dict[str, list[float]] = {v: [r / vals[0] for r in vals] for v, vals in rms.items()}
    # 3-chunk rolling median for sustained-drop detection.
    def rolling_med(arr, w=3):
        out = []
        for i in range(len(arr)):
            lo = max(0, i - w + 1)
            out.append(float(np.median(arr[lo:i+1])))
        return out
    seed_ratio_med: dict[str, list[float]] = {v: rolling_med(rs) for v, rs in seed_ratio.items()}

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    user_collapse = 10.0  # seconds

    # Panel 1: absolute RMS.
    for v, vals in rms.items():
        axes[0].plot(times, vals, "-o", color=colors.get(v, "#888"), label=v, lw=2, ms=4)
    axes[0].axvline(user_collapse, color="orange", ls="--", lw=1.2, alpha=0.7,
                    label="user-marked collapse (~10s)")
    axes[0].set_ylabel("M1: chunk RMS")
    axes[0].set_title("Absolute RMS — drifts continuously, no clean event boundary")
    axes[0].legend(loc="best", fontsize=8); axes[0].grid(alpha=0.3)

    # Panel 2: ΔRMS, the collapse-event signal.
    for v, vals in drms.items():
        axes[1].plot(times, vals, "-o", color=colors.get(v, "#888"), label=v, lw=2, ms=4)
    axes[1].axhline(0, color="grey", lw=0.6)
    axes[1].axhline(-0.05, color="red", ls="--", lw=1.0, alpha=0.6,
                    label="abort threshold ΔRMS < -0.05")
    axes[1].axvline(user_collapse, color="orange", ls="--", lw=1.2, alpha=0.7)
    axes[1].set_ylabel("ΔRMS = RMS_c - RMS_{c-1}")
    axes[1].set_title("First-difference of RMS — synchronized -0.07 to -0.15 drop at ~10.8s for ALL three variants")
    axes[1].legend(loc="best", fontsize=8); axes[1].grid(alpha=0.3)

    # Panel 3: 3-chunk median seed-relative RMS — robust to single-chunk transients.
    for v, vals in seed_ratio_med.items():
        axes[2].plot(times, vals, "-o", color=colors.get(v, "#888"), label=v, lw=2, ms=4)
    axes[2].axhline(1.0, color="grey", ls="-", lw=0.6)
    axes[2].axhline(0.85, color="red", ls="--", lw=1.0, alpha=0.6,
                    label="abort threshold (sustained < 0.85× seed)")
    axes[2].axvline(user_collapse, color="orange", ls="--", lw=1.2, alpha=0.7)
    axes[2].set_ylabel("3-chunk median(RMS_c / RMS_seed)")
    axes[2].set_xlabel("video time (s)")
    axes[2].set_title("Sustained deflation — robust to single-chunk transients; clean separation from GT")
    axes[2].legend(loc="best", fontsize=8); axes[2].grid(alpha=0.3)

    fig.tight_layout()
    p1 = out / "rms_collapse_event.png"
    fig.savefig(p1, dpi=130); plt.close(fig)
    print(f"wrote {p1}")

    # Print the trigger crossings for two candidate triggers.
    print()
    print("Trigger A: ΔRMS < -0.05  (sharp single-chunk deflation)")
    for v, vals in drms.items():
        if v == "gt": continue
        firsts = [(c, vals[c]) for c in range(1, len(vals)) if vals[c] < -0.05]
        if firsts:
            c, dv = firsts[0]
            print(f"  {v:20s}  fires at chunk {c:>2d}  (t={c*secs_per_chunk:.1f}s)  ΔRMS={dv:+.3f}")
        else:
            print(f"  {v:20s}  never fires")

    print()
    print("Trigger B: 3-chunk median(RMS/seed) < 0.85  (sustained deflation)")
    for v, vals in seed_ratio_med.items():
        if v == "gt": continue
        firsts = [(c, vals[c]) for c in range(2, len(vals)) if vals[c] < 0.85]
        if firsts:
            c, mv = firsts[0]
            print(f"  {v:20s}  fires at chunk {c:>2d}  (t={c*secs_per_chunk:.1f}s)  med={mv:.3f}")
        else:
            print(f"  {v:20s}  never fires")
    # Also print GT's value at the same trigger to confirm GT doesn't fire.
    print()
    print("GT values at user-marked collapse (~10s = chunk 16-18):")
    for c in [16, 17, 18, 19, 20]:
        print(f"  chunk {c:>2d} (t={c*secs_per_chunk:.1f}s):  RMS={rms['gt'][c]:.3f}  ΔRMS={drms['gt'][c]:+.3f}  med(RMS/seed)={seed_ratio_med['gt'][c]:.3f}")

    with (out / "rms_collapse_event.json").open("w") as fh:
        json.dump({
            "rms": rms, "drms": drms,
            "ema_abs_dgt": ema.tolist(),
            "secs_per_chunk": secs_per_chunk,
        }, fh, indent=2)


if __name__ == "__main__":
    main()
