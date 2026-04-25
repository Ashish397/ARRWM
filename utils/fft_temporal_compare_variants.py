#!/usr/bin/env python3
"""1D temporal FFT spectrum across arbitrary variants for the |0.3| peak.

Pass any number of ``--variant <label>=<path/to/latents.pt>:<key>``
entries. Plots all on one axis. Highlights the |0.3| frequency band
(target where append_baseline shows its periodic-stagger peak).

Example::

    python utils/fft_temporal_compare_variants.py \\
        --output_dir eval/fft_compare_$(date +%Y%m%d_%H%M%S) \\
        --variant gt=eval/latents_three_variants_30c_*/latents.pt:gt \\
        --variant ar_refresh=eval/latents_three_variants_30c_*/latents.pt:ar_refresh \\
        --variant append=eval/latents_three_variants_30c_*/latents.pt:append_baseline \\
        --variant appendmatch=eval/ar_refresh_appendmatch_*/latents.pt:ar_refresh_appendmatch
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)


def temporal_fft_magnitude_1d(lat: torch.Tensor) -> np.ndarray:
    """lat: [1, T, C, H, W] — return 1D temporal spectrum [T] (averaged across C, H, W)."""
    assert lat.dim() == 5 and lat.shape[0] == 1
    x = lat[0].float()  # [T, C, H, W]
    fft = torch.fft.fft(x, dim=0)  # FFT along T
    mag = fft.abs()  # [T, C, H, W]
    mag_1d = mag.mean(dim=(1, 2, 3))  # [T]
    return torch.fft.fftshift(mag_1d).numpy()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", action="append", required=True,
                   help="<label>=<path>:<key>")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--peak_freq", type=float, default=0.333,
                   help="Target frequency to annotate (default 0.333 = chunk-period).")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("tab10")

    spectra: dict[str, np.ndarray] = {}
    for v in args.variant:
        try:
            label, rest = v.split("=", 1)
            path_str, key = rest.rsplit(":", 1)
        except Exception:
            log.error("bad --variant %s; expected label=path:key", v); continue
        # Glob if path has *.
        paths = sorted(Path().glob(path_str)) if "*" in path_str else [Path(path_str)]
        if not paths:
            log.error("no match for %s", path_str); continue
        path = paths[0]
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if key not in blob:
            log.error("key %s not in %s; available=%s", key, path, list(blob.keys())); continue
        lat = blob[key]
        spec = temporal_fft_magnitude_1d(lat)
        spectra[label] = spec
        log.info("%-30s  T=%d  spec[max-1]=%.3f  spec[peak_freq]=%.3f",
                 label, lat.shape[1], spec[1] if len(spec) > 1 else 0.0, 0.0)

    # Frequencies (cycles per latent-frame). For T frames, frequencies = fftshift(fftfreq(T)).
    T_any = next(iter(spectra.values())).shape[0]
    freqs = np.fft.fftshift(np.fft.fftfreq(T_any))

    # Plot.
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {
        "gt":             "#000000",
        "ar_refresh":     "#2ca02c",
        "append_baseline":"#1f77b4",
        "append":         "#1f77b4",
        "ar_refresh_once":"#d62728",
        "ar_refresh_appendlike":         "#9467bd",
        "ar_refresh_appendmatch":        "#ff7f0e",
        "ar_refresh_postrope_freezeK":   "#8c564b",
        "ar_refresh_writeonce":          "#e377c2",
        "ar_refresh_freezeK":            "#bcbd22",
        "ar_refresh_freezeQK":           "#17becf",
    }
    for i, (label, spec) in enumerate(spectra.items()):
        c = colors.get(label, cmap(i % 10))
        ls = "-" if label != "gt" else "-"
        lw = 2.5 if label == "gt" else 1.6
        ax.plot(freqs, spec, ls=ls, lw=lw, color=c, label=label)
    ax.set_yscale("log")
    ax.axvline(args.peak_freq, color="grey", ls=":", lw=0.8, alpha=0.7, label=f"target |{args.peak_freq:.2f}|")
    ax.axvline(-args.peak_freq, color="grey", ls=":", lw=0.8, alpha=0.7)
    ax.set_xlabel("temporal frequency (cycles / latent-frame)")
    ax.set_ylabel("magnitude (log)")
    ax.set_title(f"Temporal spectrum, T={T_any} latent frames; chunk period = 1/0.33 ≈ 3 frames")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    p1 = out / "fft_temporal_compare.png"
    fig.savefig(p1, dpi=130); plt.close(fig)
    log.info("wrote %s", p1)

    # Quantify peak height at +-0.333 (chunk period). Relative to GT same band.
    def near(arr, x):
        idx = np.argmin(np.abs(freqs - x))
        return float(arr[idx])
    summary = {
        "freqs_t_axis_T": int(T_any),
        "peak_freq_target": float(args.peak_freq),
        "peak_pos": {label: near(s, args.peak_freq) for label, s in spectra.items()},
        "peak_neg": {label: near(s, -args.peak_freq) for label, s in spectra.items()},
    }
    if "gt" in spectra:
        summary["peak_pos_ratio_vs_gt"] = {
            label: near(s, args.peak_freq) / max(near(spectra["gt"], args.peak_freq), 1e-9)
            for label, s in spectra.items()
        }
    with (out / "fft_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    log.info("peak ratios vs GT @ |%.2f|:", args.peak_freq)
    if "gt" in spectra:
        for label, r in summary["peak_pos_ratio_vs_gt"].items():
            log.info("  %-30s  +%.2f → %.3f×",
                     label, args.peak_freq, r)


if __name__ == "__main__":
    main()
