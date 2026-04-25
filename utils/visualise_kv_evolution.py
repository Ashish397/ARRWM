#!/usr/bin/env python3
"""Visualise the ACTUAL K and V tensor values across chunks (not just
scalar |Δ|).

For each persisting latent (seed, gen_0, gen_1, gen_2 — i.e. those
that appear in the window across multiple chunks), at each captured
layer, we plot a 2D heatmap with rows = chunk index (in the order the
latent re-appears) and columns = feature dimension. Each cell is the
mean value of that latent's K (or V) across the spatial tokens of its
3-frame slice in that chunk.

This makes the "frozen vs morphing" story directly visible:

  - **append baseline**: every row of the heatmap should be identical
    (the cache stored the K/V once and reused it).
  - **AR_refresh**: rows differ, with deeper layers showing more
    visible morphing because deeper K/V depends on cumulative
    attention over a context that changes per chunk.
  - **AR_refresh_once**: similar to AR_refresh in that it does
    recompute, but with one rebuild per chunk (vs per pass).

We render two heatmaps per (latent × layer × K|V × variant):

  - **raw values**: zero-centred diverging colormap, common scale per
    figure to make magnitudes comparable across variants and chunks.
  - **delta from first appearance**: same layout, cell = current
    chunk's value − chunk-0 value. For append this is identically
    zero everywhere.

Output structure::

    <output_dir>/
      latent_seed_K.png
      latent_seed_V.png
      latent_gen_0_K.png
      latent_gen_0_V.png
      ...
      manifest.json

Usage::

    python utils/visualise_kv_evolution.py \\
        --ar_refresh /home/ashish/ARRWM/eval/kv_capture_<ts>/kv_capture.pt \\
        --append /home/ashish/ARRWM/eval/kv_capture_append_<ts>/kv_capture.pt \\
        --ar_refresh_once /home/ashish/ARRWM/eval/kv_capture_AR_refresh_once_<ts>/kv_capture.pt \\
        --output_dir /home/ashish/ARRWM/eval/kv_evolution_<ts>
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


def _load(pt_path: Path) -> Dict[str, Any]:
    log.info("Loading %s (%.1f MB)", pt_path, pt_path.stat().st_size / (1024 * 1024))
    return torch.load(pt_path, map_location="cpu", weights_only=False)


def _identify_latent_track(cap: Dict[str, Any], fifo_size: int, seed_chunks: int) -> List[Dict[str, Any]]:
    chunks = cap["chunks"]
    npb = cap["meta"]["num_frame_per_block"]
    n_latents = fifo_size + 1  # seed + first fifo_size gen chunks
    out: List[Dict[str, Any]] = []
    for e in range(n_latents):
        label = "seed" if e == 0 else f"gen_{e - 1}"
        apps: List[Tuple[int, int]] = []
        for c_entry in chunks:
            c_idx = int(c_entry["chunk_idx"])
            n_ctx = int(c_entry["n_ctx_blocks"])
            n_entries_at_start = min(fifo_size, seed_chunks + c_idx)
            evicted = max(0, seed_chunks + c_idx - fifo_size)
            pos = e - evicted
            if pos < 0 or pos >= n_entries_at_start:
                continue
            frame_lo = pos * npb
            apps.append((c_idx, frame_lo))
        if len(apps) >= 1:
            out.append({"label": label, "entry_idx": e, "appearances": apps})
    return out


def _per_chunk_dim_vector(
    cap: Dict[str, Any], record: Dict[str, Any], layer_idx: int, key: str, npb: int,
) -> Optional[np.ndarray]:
    """For each appearance of a latent, return its [dim] vector (mean
    over the npb spatial-token slices). Returns [n_appearances, dim].
    Returns None if the layer wasn't captured for any appearance."""
    apps = record["appearances"]
    by_chunk = {int(c["chunk_idx"]): c for c in cap["chunks"]}
    rows: List[np.ndarray] = []
    for c_idx, frame_lo in apps:
        c = by_chunk.get(c_idx)
        if c is None or layer_idx not in c["layers"]:
            return None
        wf = int(c["window_frames"])
        full = c["layers"][layer_idx][key]  # [1, seq, dim]
        seq = full.shape[1]
        frame_seqlen = seq // wf
        # Slice the npb frames starting at frame_lo.
        seq_lo = frame_lo * frame_seqlen
        seq_hi = seq_lo + npb * frame_seqlen
        block = full[:, seq_lo:seq_hi, :].float().squeeze(0)  # [npb*frame_seqlen, dim]
        # Mean over tokens within this latent's npb frames.
        vec = block.mean(dim=0).numpy()  # [dim]
        rows.append(vec)
    if not rows:
        return None
    return np.stack(rows, axis=0)  # [n_appearances, dim]


def _heatmap(ax, data: np.ndarray, title: str, vmin: float, vmax: float, cmap: str = "RdBu_r",
             apps: Optional[List[Tuple[int, int]]] = None) -> Any:
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("dim", fontsize=8)
    ax.set_ylabel("chunk", fontsize=8)
    if apps is not None:
        ax.set_yticks(range(len(apps)))
        ax.set_yticklabels([f"c{c_idx}" for c_idx, _ in apps], fontsize=7)
    return im


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ar_refresh", type=str, required=True)
    p.add_argument("--append", type=str, required=True)
    p.add_argument("--ar_refresh_once", type=str, default=None)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--max_dim", type=int, default=512,
                   help="Subsample dim axis to this many for plotting "
                        "(default 512 of 1536). Set 0 to disable.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap_ar = _load(Path(args.ar_refresh))
    cap_ap = _load(Path(args.append))
    cap_once = _load(Path(args.ar_refresh_once)) if args.ar_refresh_once else None

    npb = cap_ar["meta"]["num_frame_per_block"]
    fifo_size = min(int(cap_ar["meta"]["fifo_size"]),
                    int(cap_ap["meta"].get("cache_chunks", cap_ar["meta"]["fifo_size"])))
    seed_chunks = int(cap_ar["meta"]["ar_initial_chunks"])
    capture_layers = list(cap_ar["meta"]["capture_layers"])

    records_ar = _identify_latent_track(cap_ar, fifo_size, seed_chunks)
    records_ap = _identify_latent_track(cap_ap, fifo_size, seed_chunks)
    records_once = _identify_latent_track(cap_once, fifo_size, seed_chunks) if cap_once else []
    rec_ar_by_label = {r["label"]: r for r in records_ar}
    rec_ap_by_label = {r["label"]: r for r in records_ap}
    rec_once_by_label = {r["label"]: r for r in records_once}

    log.info("fifo_size=%d  seed_chunks=%d  layers=%s  tracked=%s",
             fifo_size, seed_chunks, capture_layers,
             [r["label"] for r in records_ar])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.error("matplotlib missing — cannot produce plots.")
        return

    # Figure layout: per latent × per (K|V), one figure with rows =
    # capture_layers, cols = (raw AR_refresh, raw append, [raw once,]
    # delta AR_refresh, delta append, [delta once]).
    variants_present = ["AR_refresh", "append"]
    cap_by_variant: Dict[str, Dict[str, Any]] = {"AR_refresh": cap_ar, "append": cap_ap}
    rec_by_variant: Dict[str, Dict[str, Any]] = {"AR_refresh": rec_ar_by_label, "append": rec_ap_by_label}
    if cap_once is not None:
        variants_present.append("AR_refresh_once")
        cap_by_variant["AR_refresh_once"] = cap_once
        rec_by_variant["AR_refresh_once"] = rec_once_by_label

    n_var = len(variants_present)
    written: List[str] = []

    for label in [r["label"] for r in records_ar]:
        for kv_key, kv_name in [("k", "K"), ("v", "V")]:
            n_layers = len(capture_layers)
            fig, axes = plt.subplots(
                nrows=n_layers, ncols=2 * n_var,
                figsize=(3.2 * 2 * n_var, 2.4 * n_layers + 0.6),
                squeeze=False,
            )
            # Per-figure global vmin/vmax for raw values.
            all_raw: List[np.ndarray] = []
            all_delta: List[np.ndarray] = []
            cells: Dict[Tuple[int, int, str], np.ndarray] = {}
            for li, L in enumerate(capture_layers):
                for vi, v in enumerate(variants_present):
                    rec = rec_by_variant[v].get(label)
                    if rec is None:
                        continue
                    data = _per_chunk_dim_vector(cap_by_variant[v], rec, L, kv_key, npb)
                    if data is None:
                        continue
                    if args.max_dim and data.shape[1] > args.max_dim:
                        data = data[:, :args.max_dim]
                    cells[(li, vi, "raw")] = data
                    delta = data - data[0:1]
                    cells[(li, vi, "delta")] = delta
                    all_raw.append(data)
                    all_delta.append(delta)
            if not cells:
                plt.close(fig)
                continue
            raw_concat = np.concatenate(all_raw, axis=0)
            raw_abs = np.percentile(np.abs(raw_concat), 99)
            delta_concat = np.concatenate(all_delta, axis=0)
            delta_abs = np.percentile(np.abs(delta_concat), 99)
            if delta_abs < 1e-6:
                delta_abs = max(1e-6, np.abs(delta_concat).max())

            for li, L in enumerate(capture_layers):
                for vi, v in enumerate(variants_present):
                    rec = rec_by_variant[v].get(label)
                    apps = rec["appearances"] if rec else None
                    raw = cells.get((li, vi, "raw"))
                    delta = cells.get((li, vi, "delta"))
                    ax_raw = axes[li][vi]
                    ax_delta = axes[li][n_var + vi]
                    if raw is not None:
                        im_raw = _heatmap(
                            ax_raw, raw,
                            f"L{L} {v} {kv_name} raw",
                            -raw_abs, raw_abs, cmap="RdBu_r", apps=apps,
                        )
                        if vi == n_var - 1:
                            fig.colorbar(im_raw, ax=ax_raw, fraction=0.045)
                        im_delta = _heatmap(
                            ax_delta, delta,
                            f"L{L} {v} {kv_name} Δ from c{apps[0][0]}",
                            -delta_abs, delta_abs, cmap="RdBu_r", apps=apps,
                        )
                        if vi == n_var - 1:
                            fig.colorbar(im_delta, ax=ax_delta, fraction=0.045)
                    else:
                        ax_raw.set_title(f"L{L} {v} {kv_name} (n/a)", fontsize=9)
                        ax_delta.set_title(f"L{L} {v} {kv_name} Δ (n/a)", fontsize=9)
                        ax_raw.axis("off")
                        ax_delta.axis("off")

            fig.suptitle(
                f"{label} — {kv_name} tensor evolution across chunks (mean over npb spatial tokens, "
                f"99th-pct symmetric scale, ride {cap_ar['meta'].get('rank_zarr', '?')})",
                y=1.005, fontsize=11,
            )
            fig.tight_layout()
            out_png = out_dir / f"latent_{label}_{kv_name}.png"
            fig.savefig(out_png, dpi=110, bbox_inches="tight")
            plt.close(fig)
            written.append(str(out_png))
            log.info("Wrote %s", out_png)

    with (out_dir / "manifest.json").open("w") as fh:
        json.dump({
            "written": written,
            "fifo_size": fifo_size,
            "seed_chunks": seed_chunks,
            "capture_layers": capture_layers,
            "variants": variants_present,
            "max_dim": int(args.max_dim),
        }, fh, indent=2)


if __name__ == "__main__":
    main()
