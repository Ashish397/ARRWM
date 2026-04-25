#!/usr/bin/env python3
"""Cheap collapse-detection signals on saved latents.

Three of the four signals proposed (M1, M2, M3) — M4 needs a teacher
forward and is excluded here.

For each variant in ``latents.pt`` and for each 3-frame chunk:

  M1: RMS = sqrt(mean(x**2))
      Catches:
        - frozen frame  (RMS deflates toward 0)
        - noise explosion (RMS inflates)
      Use seed-chunk RMS as the per-rollout EMA baseline; trigger when
      ``|RMS_c - RMS_seed| / RMS_seed > 0.5`` (tunable).

  M2: peak = max(|x|)
      Hard cap for catastrophic explosion. Trigger when peak > some
      absolute (e.g. 1e2) — calibrate from GT distribution.

  M3: ||commit_c - commit_{c-1}||_F  (inter-commit L2 delta)
      Catches frozen frame (delta -> 0) and thrash (delta -> huge).
      Both directions matter.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def chunk_rms(chunk: torch.Tensor) -> float:
    return chunk.float().pow(2).mean().sqrt().item()


def chunk_peak(chunk: torch.Tensor) -> float:
    return chunk.float().abs().amax().item()


def chunk_delta_l2(prev: torch.Tensor, cur: torch.Tensor) -> float:
    return (cur.float() - prev.float()).pow(2).sum().sqrt().item()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--latents_pt", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_frame_per_block", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    blob = torch.load(args.latents_pt, map_location="cpu", weights_only=False)
    npb = int(args.num_frame_per_block)

    variants = ["gt", "ar_refresh", "ar_refresh_once", "append_baseline"]
    colors = {
        "gt":               "#000000",
        "ar_refresh":       "#1f77b4",
        "ar_refresh_once":  "#d62728",
        "append_baseline":  "#2ca02c",
    }

    rms_d:   dict[str, list[float]] = {}
    peak_d:  dict[str, list[float]] = {}
    delta_d: dict[str, list[float]] = {}

    for v in variants:
        if v not in blob: continue
        lat = blob[v]
        T = lat.shape[1]
        n_chunks = T // npb
        rms_l, peak_l, delta_l = [], [], []
        prev = None
        for c in range(n_chunks):
            chunk = lat[0, c * npb : (c + 1) * npb]
            rms_l.append(chunk_rms(chunk))
            peak_l.append(chunk_peak(chunk))
            delta_l.append(0.0 if prev is None else chunk_delta_l2(prev, chunk))
            prev = chunk
        rms_d[v]   = rms_l
        peak_d[v]  = peak_l
        delta_d[v] = delta_l
        print(f"{v:20s}")
        print(f"  RMS    = {[round(x,4) for x in rms_l]}")
        print(f"  peak   = {[round(x,2) for x in peak_l]}")
        print(f"  Δ_L2   = {[round(x,2) for x in delta_l]}")

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    for v, vals in rms_d.items():
        axes[0].plot(range(len(vals)), vals, "-o", color=colors.get(v, "#888"), label=v, lw=2, ms=4)
    if "gt" in rms_d:
        # show ±50% of seed RMS as the suggested abort band, using the
        # generated-variant scale (use ar_refresh's seed which equals GT's seed).
        rs = rms_d.get("ar_refresh", rms_d["gt"])[0]
        axes[0].axhline(rs * 1.5, color="grey", ls="--", lw=0.8, alpha=0.6,
                        label="seed × 1.5  (inflate cap)")
        axes[0].axhline(rs * 0.5, color="grey", ls=":",  lw=0.8, alpha=0.6,
                        label="seed × 0.5  (deflate cap)")
    axes[0].set_ylabel("M1: chunk RMS")
    axes[0].set_title("M1 — commit-chunk latent RMS  (frozen ↘  /  noise-explode ↗)")
    axes[0].legend(loc="best", fontsize=8); axes[0].grid(alpha=0.3)

    for v, vals in peak_d.items():
        axes[1].plot(range(len(vals)), vals, "-o", color=colors.get(v, "#888"), label=v, lw=2, ms=4)
    axes[1].set_ylabel("M2: chunk peak |x|")
    axes[1].set_title("M2 — commit-chunk peak |pred_x0|  (NaN-adjacent guard)")
    axes[1].legend(loc="best", fontsize=8); axes[1].grid(alpha=0.3)

    for v, vals in delta_d.items():
        # skip the 0 at index 0 to keep autoscale honest
        axes[2].plot(range(1, len(vals)), vals[1:], "-o",
                     color=colors.get(v, "#888"), label=v, lw=2, ms=4)
    axes[2].set_ylabel("M3: ||commit_c - commit_{c-1}||_2")
    axes[2].set_xlabel("chunk index")
    axes[2].set_title("M3 — inter-commit L2 delta  (frozen ↘ 0  /  thrash ↗)")
    axes[2].legend(loc="best", fontsize=8); axes[2].grid(alpha=0.3)

    fig.tight_layout()
    p1 = out / "cheap_signals_per_chunk.png"
    fig.savefig(p1, dpi=130); plt.close(fig)
    print(f"wrote {p1}")

    # Combined: all signals normalised to their own seed-chunk value, log
    # scale, so they're plotted on the same axes. Easy at-a-glance fail
    # detection — anything that diverges from y=1.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, (name, d) in zip(axes, [("M1 RMS", rms_d), ("M2 peak", peak_d), ("M3 Δ_L2", delta_d)]):
        for v, vals in d.items():
            base = vals[0] if name != "M3 Δ_L2" else (vals[1] if len(vals) > 1 else 1.0)
            if base <= 0: continue
            xs = list(range(len(vals)))
            if name == "M3 Δ_L2":
                xs = xs[1:]; vals = vals[1:]
            ax.plot(xs, [val / base for val in vals], "-o",
                    color=colors.get(v, "#888"), label=v, lw=2, ms=4)
        ax.axhline(1.0, color="grey", ls="-", lw=0.8)
        ax.axhline(1.5, color="red", ls="--", lw=1.0, alpha=0.5, label="abort hi (1.5×)")
        ax.axhline(0.5, color="red", ls=":",  lw=1.0, alpha=0.5, label="abort lo (0.5×)")
        ax.set_yscale("log")
        ax.set_xlabel("chunk index"); ax.set_title(name)
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="best", fontsize=7)
    axes[0].set_ylabel("ratio vs seed (log)")
    fig.suptitle("Cheap signals normalised to seed chunk — exit band [0.5, 1.5]")
    fig.tight_layout()
    p2 = out / "cheap_signals_seed_ratio.png"
    fig.savefig(p2, dpi=130); plt.close(fig)
    print(f"wrote {p2}")

    payload = {"M1_rms": rms_d, "M2_peak": peak_d, "M3_l2_delta": delta_d}
    with (out / "cheap_signals.json").open("w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {out / 'cheap_signals.json'}")


if __name__ == "__main__":
    main()
