#!/usr/bin/env python3
"""FFT comparison of the three inference variants' latents vs GT.

Loads the ``latents.pt`` produced by ``save_latents_three_variants.py``
(with keys ``gt``, ``ar_refresh``, ``ar_refresh_once``,
``append_baseline``) and runs:

  1. **Per-chunk 2D spatial FFT**. Average magnitude across the 16
     channels and the 3 latent frames per chunk → one 2D spectrum per
     (variant, chunk). Spectra are ``fftshift``-ed so DC is at centre.

  2. **Radially-averaged 1D spatial spectrum**. Azimuthally averages
     the 2D magnitude into a 1D curve of power vs spatial frequency.
     Lets us plot variants on the same axes to see excess/missing
     energy at each spatial scale.

  3. **Temporal FFT along the 24-frame axis**. Average across channels
     and spatial positions → one 1D temporal spectrum per variant,
     showing whether chunks differ in their temporal-frequency
     content (expect AR_refresh_once's stagger to show up as
     excess low-frequency temporal energy, and append's constant
     offset as a broadband bias that's close-to-flat across time).

  4. **2D spectral-difference heatmaps vs GT**. Plots
     ``|FFT(variant) - FFT(gt)|`` magnitude averaged over chunks and
     channels — shows *where* in spatial-frequency space each variant
     diverges from GT.

  5. **Per-chunk MAD in FFT domain**. For each chunk, compute the
     mean-abs-diff of the variant's spatial magnitude spectrum vs GT's
     and plot against chunk index — the time-invariant signature
     (append) should plateau; the growing-drift signature
     (AR_refresh_once) should climb.

Outputs:

    <output_dir>/
      fft_radial_avg.png       — 1D radial spatial spectrum, averaged over chunks
      fft_radial_per_chunk.png — 1D radial spatial spectrum, one line per chunk
      fft_temporal.png         — 1D temporal spectrum, one line per variant
      fft_diff_heatmap.png     — 2D magnitude diff vs GT (per variant)
      fft_mad_per_chunk.png    — MAD(FFT(variant, c), FFT(gt, c)) per chunk
      fft_data.pt              — raw FFT tensors for offline re-analysis

Usage::

    python utils/fft_compare_latents.py \\
        --latents /home/ashish/ARRWM/eval/latents_three_variants_<ts>/latents.pt \\
        --output_dir /home/ashish/ARRWM/eval/fft_compare_<ts>
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


VARIANTS = ["gt", "ar_refresh", "ar_refresh_once", "append_baseline"]
PLOT_STYLE = {
    "gt":              dict(color="black", linestyle="-",  lw=2.0, label="GT"),
    "ar_refresh":      dict(color="tab:green",  linestyle="-",  lw=1.5, label="AR_refresh"),
    "ar_refresh_once": dict(color="tab:red",    linestyle="--", lw=1.5, label="AR_refresh_once"),
    "append_baseline": dict(color="tab:blue",   linestyle=":",  lw=1.8, label="append_baseline"),
}


def _spatial_fft_magnitude(
    latents: torch.Tensor,
) -> torch.Tensor:
    """2D spatial FFT magnitude.

    Input  : [1, T, C, H, W] real
    Output : [T, C, H, W] real, fftshifted so DC is at centre, magnitude.
    """
    x = latents[0]  # [T, C, H, W]
    spec = torch.fft.fft2(x, dim=(-2, -1))
    spec = torch.fft.fftshift(spec, dim=(-2, -1))
    return spec.abs()


def _radial_average(spec_2d: torch.Tensor) -> torch.Tensor:
    """Azimuthal average of a shifted 2D spectrum.

    Input  : [..., H, W] magnitude (DC at centre)
    Output : [..., R]    with R = min(H, W)//2 + 1 radial bins

    Each radial bin r is the mean magnitude over all (h, w) with
    integer-rounded radius r from the centre.
    """
    *lead, H, W = spec_2d.shape
    cy, cx = H // 2, W // 2
    yy, xx = torch.meshgrid(
        torch.arange(H), torch.arange(W), indexing="ij",
    )
    r = torch.sqrt((yy - cy).float() ** 2 + (xx - cx).float() ** 2).round().to(torch.int64)
    r_max = int(r.max().item())
    # Flatten spectrum to [N, H*W] then bincount-weighted mean per radius.
    spec_flat = spec_2d.reshape(-1, H * W)
    r_flat = r.reshape(-1)
    out = torch.zeros(spec_flat.shape[0], r_max + 1, dtype=spec_flat.dtype)
    counts = torch.bincount(r_flat, minlength=r_max + 1).to(spec_flat.dtype)
    for i in range(spec_flat.shape[0]):
        s = torch.bincount(r_flat, weights=spec_flat[i], minlength=r_max + 1)
        out[i] = s / counts.clamp(min=1)
    out = out.view(*lead, r_max + 1)
    return out


def _temporal_fft_magnitude(latents: torch.Tensor) -> torch.Tensor:
    """1D FFT along the temporal (T) axis.

    Input  : [1, T, C, H, W] real
    Output : [T, C, H, W] magnitude, fftshifted.
    """
    x = latents[0]  # [T, C, H, W]
    spec = torch.fft.fft(x, dim=0)
    spec = torch.fft.fftshift(spec, dim=0)
    return spec.abs()


def _load(pt_path: Path) -> Dict[str, Any]:
    log.info("Loading %s (%.1f MB)", pt_path, pt_path.stat().st_size / (1024 * 1024))
    d = torch.load(pt_path, map_location="cpu", weights_only=False)
    for v in VARIANTS:
        if v not in d:
            raise SystemExit(f"latents.pt missing key '{v}'; have {list(d.keys())}")
    return d


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--latents", type=str, required=True,
                   help="Path to latents.pt from save_latents_three_variants.py")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--channel_mean", action="store_true",
                   help="Average spectra across 16 latent channels (default True — cleaner plots).")
    p.add_argument("--no-channel_mean", dest="channel_mean", action="store_false")
    p.set_defaults(channel_mean=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load(Path(args.latents))
    meta = data["meta"]
    npb = int(meta["num_frame_per_block"])
    seed_frames = int(meta["seed_frames"])
    total_frames = int(meta["total_frames"])
    gen_chunks = int(meta["ar_gen_chunks"])
    ride = meta.get("rank_zarr", "?")
    log.info("Ride %s | total_frames=%d | seed=%d gen=%d npb=%d",
             ride, total_frames, seed_frames, gen_chunks, npb)

    # --- 1. Per-chunk 2D spatial FFT magnitude ---
    # per_variant_2d: {variant: [gen_chunks, C, H, W]}   (channel-mean optional)
    per_variant_2d: Dict[str, torch.Tensor] = {}
    for v in VARIANTS:
        lat = data[v]  # [1, T, C, H, W]
        mag = _spatial_fft_magnitude(lat)  # [T, C, H, W]
        # Only keep the GEN portion (skip seed frames) and slice into chunks.
        mag_gen = mag[seed_frames:]  # [gen_frames, C, H, W]
        # Reshape to chunks.
        if mag_gen.shape[0] != gen_chunks * npb:
            log.warning("Unexpected shape: gen mag has %d frames, expected %d",
                        mag_gen.shape[0], gen_chunks * npb)
        chunks = mag_gen.view(gen_chunks, npb, *mag_gen.shape[1:]).mean(dim=1)  # [gen_chunks, C, H, W]
        if args.channel_mean:
            chunks = chunks.mean(dim=1, keepdim=True)  # [gen_chunks, 1, H, W]
        per_variant_2d[v] = chunks

    # --- 2. Radial average of 2D spectra ---
    # per_variant_radial: {variant: [gen_chunks, C, R]}
    per_variant_radial: Dict[str, torch.Tensor] = {}
    for v, spec_2d in per_variant_2d.items():
        per_variant_radial[v] = _radial_average(spec_2d)

    # --- 3. Temporal FFT along T=24 dim ---
    # per_variant_temporal: {variant: [T, C, H, W]} magnitude of 1D FFT along T
    per_variant_temporal: Dict[str, torch.Tensor] = {}
    for v in VARIANTS:
        lat = data[v]
        t_mag = _temporal_fft_magnitude(lat)  # [T, C, H, W]
        # Average across spatial and channel dims → [T] 1D temporal spectrum.
        t_1d = t_mag.mean(dim=(1, 2, 3))
        per_variant_temporal[v] = t_1d

    # --- Plots ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib missing — skipping plots.")
        return

    # Plot 1: Radial spectrum, averaged over chunks.
    fig, ax = plt.subplots(figsize=(8, 5))
    for v in VARIANTS:
        rad = per_variant_radial[v]  # [gen_chunks, C, R]
        curve = rad.mean(dim=(0, 1)).numpy()  # [R]
        ax.plot(curve, **PLOT_STYLE[v])
    ax.set_xlabel("spatial radial frequency bin")
    ax.set_ylabel("magnitude (log scale)")
    ax.set_yscale("log")
    ax.set_title(f"Radial spatial spectrum — averaged over {gen_chunks} generated chunks ({ride})")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    p1 = out_dir / "fft_radial_avg.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", p1)

    # Plot 2: Radial spectrum per chunk, 2x2 grid (one panel per variant).
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    for i, v in enumerate(VARIANTS):
        ax = axes_flat[i]
        rad = per_variant_radial[v].mean(dim=1).numpy()  # [gen_chunks, R]
        n = rad.shape[0]
        for c in range(n):
            colour = plt.cm.viridis(c / max(n - 1, 1))
            ax.plot(rad[c], color=colour, lw=1.2, label=f"chunk {c}")
        ax.set_yscale("log")
        ax.set_title(v)
        ax.grid(True, which="both", alpha=0.3)
        if i // 2 == 1:
            ax.set_xlabel("radial freq bin")
        if i % 2 == 0:
            ax.set_ylabel("magnitude")
        if i == 1:
            ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.suptitle(f"Radial spatial spectrum per generated chunk ({ride})", y=1.01)
    fig.tight_layout()
    p2 = out_dir / "fft_radial_per_chunk.png"
    fig.savefig(p2, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", p2)

    # Plot 3: Temporal FFT per variant (one line per variant).
    fig, ax = plt.subplots(figsize=(8, 5))
    T = total_frames
    # Frequency bin index (fftshifted): -T/2 .. T/2-1
    freqs = np.fft.fftshift(np.fft.fftfreq(T))  # cycles / sample
    for v in VARIANTS:
        curve = per_variant_temporal[v].numpy()
        ax.plot(freqs, curve, **PLOT_STYLE[v])
    ax.set_xlabel("temporal frequency (cycles / latent-frame)")
    ax.set_ylabel("magnitude (log scale)")
    ax.set_yscale("log")
    ax.set_title(f"Temporal spectrum of {T}-frame latent sequence ({ride})")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    p3 = out_dir / "fft_temporal.png"
    fig.tight_layout()
    fig.savefig(p3, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", p3)

    # Plot 4: Spectral diff heatmap vs GT (2D, averaged over chunks+channels).
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    gt_2d = per_variant_2d["gt"].mean(dim=(0, 1)).numpy()  # [H, W]
    for i, v in enumerate([x for x in VARIANTS if x != "gt"]):
        ax = axes[i]
        spec = per_variant_2d[v].mean(dim=(0, 1)).numpy()  # [H, W]
        diff = np.abs(spec - gt_2d)
        im = ax.imshow(np.log1p(diff), aspect="auto", cmap="magma")
        ax.set_title(f"log(1 + |FFT({v}) - FFT(gt)|)")
        ax.set_xlabel("kx bin")
        if i == 0:
            ax.set_ylabel("ky bin")
        # DC marker.
        cy, cx = diff.shape[0] // 2, diff.shape[1] // 2
        ax.plot(cx, cy, marker="+", color="cyan", markersize=10)
        fig.colorbar(im, ax=ax, fraction=0.045)
    fig.suptitle(f"2D spectral-magnitude diff vs GT — chunk-averaged ({ride})", y=1.02)
    fig.tight_layout()
    p4 = out_dir / "fft_diff_heatmap.png"
    fig.savefig(p4, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", p4)

    # Plot 5: Per-chunk MAD in FFT magnitude domain.
    fig, ax = plt.subplots(figsize=(8, 5))
    gt_rad = per_variant_radial["gt"]  # [gen_chunks, C, R]
    for v in VARIANTS:
        if v == "gt":
            continue
        rad = per_variant_radial[v]
        per_chunk = (rad - gt_rad).abs().mean(dim=(1, 2)).numpy()  # [gen_chunks]
        ax.plot(range(len(per_chunk)), per_chunk, marker="o", **PLOT_STYLE[v])
    ax.set_xlabel("generated chunk index")
    ax.set_ylabel("mean |Δ radial spectrum| vs GT")
    ax.set_title("Per-chunk drift in radial spatial spectrum vs GT")
    ax.legend()
    ax.grid(True, alpha=0.3)
    p5 = out_dir / "fft_mad_per_chunk.png"
    fig.tight_layout()
    fig.savefig(p5, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", p5)

    # --- Save raw tensors for downstream re-analysis. ---
    fft_pt = out_dir / "fft_data.pt"
    torch.save({
        "meta": meta,
        "per_variant_2d": {k: v for k, v in per_variant_2d.items()},
        "per_variant_radial": {k: v for k, v in per_variant_radial.items()},
        "per_variant_temporal": {k: v for k, v in per_variant_temporal.items()},
    }, fft_pt)
    log.info("Saved FFT tensors -> %s (%.1f MB)",
             fft_pt, fft_pt.stat().st_size / (1024 * 1024))

    # --- Quick numeric summary dumped to JSON + logged. ---
    summary: Dict[str, Any] = {"ride": ride, "per_chunk_radial_mad_vs_gt": {}}
    for v in VARIANTS:
        if v == "gt":
            continue
        rad = per_variant_radial[v]
        per_chunk = (rad - gt_rad).abs().mean(dim=(1, 2)).tolist()
        summary["per_chunk_radial_mad_vs_gt"][v] = [round(x, 5) for x in per_chunk]
    # Temporal-spectrum MAD vs GT (1 scalar per variant).
    gt_t = per_variant_temporal["gt"].numpy()
    summary["temporal_spectrum_mad_vs_gt"] = {}
    for v in VARIANTS:
        if v == "gt":
            continue
        t_mag = per_variant_temporal[v].numpy()
        summary["temporal_spectrum_mad_vs_gt"][v] = float(np.mean(np.abs(t_mag - gt_t)))
    log.info("Summary: %s", json.dumps(summary, indent=2))
    with (out_dir / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
