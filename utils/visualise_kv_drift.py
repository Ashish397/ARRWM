#!/usr/bin/env python3
"""Compare how K and V tensors DRIFT across chunks between AR_refresh
(context K/V is recomputed every chunk) and append baseline (context
K/V is computed once at commit and cached forever).

Consumes the ``kv_capture.pt`` files produced by
``analyse_AR_refresh_kv.py`` and ``analyse_append_baseline_kv.py`` (and
optionally ``analyse_AR_refresh_once_kv.py``). For each *persisting*
window-frame position — i.e. a latent that sits in the window across
multiple consecutive chunks — we extract its K and V at each captured
layer and compute the delta from its chunk-0 value. In append baseline
the delta is (structurally) zero because the K/V was written to the
cache once at commit and reused; in AR_refresh the delta grows because
the model recomputes the same latent's K/V against a changing context.

Outputs:

  - ``drift_per_latent.png``: one row per latent (by entry-into-FIFO
    order — ``seed``, then gen chunk 0, gen chunk 1, …), two columns
    (K and V). Inside each cell: lines for AR_refresh (solid) and
    append (dashed), x = "chunk index at which we sample this latent's
    K/V", y = mean absolute delta vs the first chunk where this latent
    appears.

  - ``drift_per_layer.png``: one row per captured layer, two columns
    (K and V). Curves collapse the above over latents to show the
    layer-wise drift trend.

  - ``drift.json``: the numeric data for both plots.

Usage::

    python utils/visualise_kv_drift.py \\
        --ar_refresh /home/ashish/ARRWM/eval/kv_capture_<ts>/kv_capture.pt \\
        --append     /home/ashish/ARRWM/eval/kv_capture_append_<ts>/kv_capture.pt \\
        --output_dir /home/ashish/ARRWM/eval/kv_drift_<ts>
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


def _load_capture(pt_path: Path) -> Dict[str, Any]:
    log.info("Loading %s (%.1f MB)", pt_path, pt_path.stat().st_size / (1024 * 1024))
    return torch.load(pt_path, map_location="cpu", weights_only=False)


def _per_frame_slices(
    K_or_V: torch.Tensor, window_frames: int,
) -> List[torch.Tensor]:
    """Split a [B, seq_total, dim] K/V capture into per-frame slices
    [B, frame_seqlen, dim]. Returns a list of length ``window_frames``."""
    seq = K_or_V.shape[1]
    assert seq % window_frames == 0, (seq, window_frames)
    fl = seq // window_frames
    return [K_or_V[:, i * fl:(i + 1) * fl, :] for i in range(window_frames)]


def _identify_latent_track(
    cap: Dict[str, Any], fifo_size: int, seed_chunks: int,
) -> List[Dict[str, Any]]:
    """For every *unique* latent that ever sits in the window, return a
    list of (chunk_idx, window_frame_index) pairs describing where that
    latent appears across chunks.

    The FIFO evicts oldest when it saturates, so a latent that entered
    at commit time ``C`` sits at window-frame-position
    ``max(0, entries_so_far - fifo_size) → entries_so_far - 1`` — it
    drifts leftwards as new chunks push in, then gets evicted.

    Concretely for ``seed_chunks=1``, ``fifo_size=3``, ``ar_gen_chunks=7``:

      - seed:  chunks 0, 1, 2 → window frames 0, 0, 0 (always leftmost until evicted)
      - cur_0: chunks 1, 2, 3 → window frames 3-5, 3-5, 0-2 (shifts left)
      - cur_1: chunks 2, 3, 4 → window frames 6-8, 3-5, 0-2
      - cur_2: chunks 3, 4, 5 → window frames 6-8, 3-5, 0-2
      - cur_3: chunks 4, 5, 6 → ...
      - ...

    Note: for this visualiser we only track the seed + first ``fifo_size``
    generated chunks (the ones that appear in ≥ 2 chunks' windows, so
    there's a drift to measure). Returns records shaped:

        [
          {"label": "seed", "appearances": [(chunk_idx, frame_lo), ...]},
          {"label": "gen_0", "appearances": [(chunk_idx, frame_lo), ...]},
          ...
        ]

    where ``frame_lo`` is the starting window-frame index for that
    latent's 3-frame block in that chunk.
    """
    chunks = cap["chunks"]
    npb = cap["meta"]["num_frame_per_block"]

    # A latent "entered the FIFO" at commit time. seed enters at commit 0.
    # Generated chunk k enters at commit k+seed_chunks. Each latent stays
    # in the FIFO for up to `fifo_size` commits (then evicted).
    records: List[Dict[str, Any]] = []

    # Seed latent: enters at entry_idx = 0. First appears in the window
    # of chunk 0 at frame_lo = 0 (when n_ctx_blocks = 1 in chunk 0).
    # Continues to appear as long as it's in the FIFO — which is for
    # fifo_size commits total starting from its entry. For seed that's
    # chunks 0, 1, ..., fifo_size-1.
    # But `n_ctx_blocks` changes during warmup; before eviction, seed's
    # frame_lo = 0 in chunks 0, 1, 2 (at n_ctx=1, 2, 3 respectively).
    # In chunk 3 the seed gets evicted.
    # Rather than reconstruct all this arithmetically, we scan the
    # captured chunks and match by window position.

    # For each unique latent by entry-order, track appearances.
    # Entry order = FIFO entry order = [seed, gen_0, gen_1, gen_2, ...].
    # Latent at entry index e is at window-frame-position
    # ``pos(e, c) = e - max(0, (c + seed_chunks) - fifo_size)``
    # where c is the generated chunk index, provided that
    # ``0 <= pos(e, c) <= n_ctx_blocks(c) - 1`` (i.e. it's still in FIFO).

    n_latents_to_track = fifo_size + 1  # seed + first fifo_size gen chunks
    for e in range(n_latents_to_track):
        label = "seed" if e == 0 else f"gen_{e - 1}"
        apps: List[Tuple[int, int]] = []
        for c_entry in chunks:
            c_idx = int(c_entry["chunk_idx"])
            n_ctx = int(c_entry["n_ctx_blocks"])
            # entries in the FIFO at the start of chunk c_idx = seed + gen_0..gen_{c_idx-1}
            # but capped at fifo_size.
            n_entries_at_start = min(fifo_size, seed_chunks + c_idx)
            # eviction count
            evicted = max(0, seed_chunks + c_idx - fifo_size)
            # position of latent e in the FIFO (0=oldest)
            pos = e - evicted
            if pos < 0 or pos >= n_entries_at_start:
                continue  # this latent has been evicted or not yet entered
            frame_lo = pos * npb
            apps.append((c_idx, frame_lo))
        if len(apps) >= 2:
            records.append({"label": label, "entry_idx": e, "appearances": apps})
    return records


def _delta_K_V(
    cap: Dict[str, Any],
    layer_idx: int,
    record: Dict[str, Any],
    npb: int,
) -> Dict[str, Any]:
    """For a single latent's track, compute the per-appearance delta of
    its K and V at ``layer_idx`` compared to its FIRST appearance.

    Returns {"chunks": [c_idx, ...], "k_delta": [...], "v_delta": [...]}
    where each delta is a scalar (mean absolute difference across seq
    and dim)."""
    apps = record["appearances"]
    chunks = cap["chunks"]
    # Build a lookup by chunk_idx.
    by_chunk = {int(c["chunk_idx"]): c for c in chunks}

    ref_k = ref_v = None
    chunk_ids: List[int] = []
    k_deltas: List[float] = []
    v_deltas: List[float] = []

    for c_idx, frame_lo in apps:
        c = by_chunk.get(c_idx)
        if c is None:
            continue
        if layer_idx not in c["layers"]:
            continue
        wf = int(c["window_frames"])
        k_full = c["layers"][layer_idx]["k"]  # [1, seq, dim]
        v_full = c["layers"][layer_idx]["v"]
        # Slice frames [frame_lo, frame_lo+npb)
        slices_k = _per_frame_slices(k_full, wf)
        slices_v = _per_frame_slices(v_full, wf)
        k_block = torch.cat(slices_k[frame_lo:frame_lo + npb], dim=1).float()
        v_block = torch.cat(slices_v[frame_lo:frame_lo + npb], dim=1).float()

        if ref_k is None:
            ref_k = k_block
            ref_v = v_block
            chunk_ids.append(c_idx)
            k_deltas.append(0.0)
            v_deltas.append(0.0)
            continue
        # Mean absolute difference.
        k_d = (k_block - ref_k).abs().mean().item()
        v_d = (v_block - ref_v).abs().mean().item()
        chunk_ids.append(c_idx)
        k_deltas.append(k_d)
        v_deltas.append(v_d)

    return {"chunks": chunk_ids, "k_delta": k_deltas, "v_delta": v_deltas}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ar_refresh", type=str, required=True,
                   help="Path to analyse_AR_refresh_kv.py's kv_capture.pt")
    p.add_argument("--append", type=str, required=True,
                   help="Path to analyse_append_baseline_kv.py's kv_capture.pt")
    p.add_argument("--ar_refresh_once", type=str, default=None,
                   help="(Optional) Path to analyse_AR_refresh_once_kv.py's kv_capture.pt for a 3-way comparison.")
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap_ar = _load_capture(Path(args.ar_refresh))
    cap_ap = _load_capture(Path(args.append))
    cap_once = _load_capture(Path(args.ar_refresh_once)) if args.ar_refresh_once else None

    npb = cap_ar["meta"]["num_frame_per_block"]
    assert npb == cap_ap["meta"]["num_frame_per_block"]
    # FIFO size for AR_refresh is ``meta.fifo_size``; for append it's ``meta.cache_chunks``.
    fifo_ar = int(cap_ar["meta"]["fifo_size"])
    fifo_ap = int(cap_ap["meta"].get("cache_chunks", fifo_ar))
    if fifo_ar != fifo_ap:
        log.warning(
            "fifo_size mismatch: AR_refresh=%d, append=%d — using min for matching.",
            fifo_ar, fifo_ap,
        )
    fifo_size = min(fifo_ar, fifo_ap)
    seed_chunks = int(cap_ar["meta"]["ar_initial_chunks"])

    records_ar = _identify_latent_track(cap_ar, fifo_size, seed_chunks)
    records_ap = _identify_latent_track(cap_ap, fifo_size, seed_chunks)
    # Map label → record for both.
    rec_ar_by_label = {r["label"]: r for r in records_ar}
    rec_ap_by_label = {r["label"]: r for r in records_ap}
    records_once_by_label = {}
    if cap_once is not None:
        records_once = _identify_latent_track(cap_once, fifo_size, seed_chunks)
        records_once_by_label = {r["label"]: r for r in records_once}

    capture_layers = list(cap_ar["meta"]["capture_layers"])
    log.info("fifo_size=%d  seed_chunks=%d  capture_layers=%s  tracked_latents=%s",
             fifo_size, seed_chunks, capture_layers,
             [r["label"] for r in records_ar])

    # --- Compute deltas. ---
    results = {
        "meta": {
            "fifo_size": fifo_size,
            "seed_chunks": seed_chunks,
            "capture_layers": capture_layers,
            "ar_refresh_pt": str(args.ar_refresh),
            "append_pt": str(args.append),
            "ar_refresh_once_pt": str(args.ar_refresh_once) if cap_once else None,
        },
        "per_latent": {},   # {label: {layer: {"ar": {...}, "append": {...}, "once": {...}}}}
    }

    for label in sorted(set(rec_ar_by_label) | set(rec_ap_by_label)):
        rec_a = rec_ar_by_label.get(label)
        rec_p = rec_ap_by_label.get(label)
        rec_o = records_once_by_label.get(label) if cap_once else None
        per_layer: Dict[int, Dict[str, Any]] = {}
        for L in capture_layers:
            row: Dict[str, Any] = {}
            if rec_a is not None:
                row["ar_refresh"] = _delta_K_V(cap_ar, L, rec_a, npb)
            if rec_p is not None:
                row["append"] = _delta_K_V(cap_ap, L, rec_p, npb)
            if rec_o is not None:
                row["ar_refresh_once"] = _delta_K_V(cap_once, L, rec_o, npb)
            per_layer[L] = row
        results["per_latent"][label] = per_layer

    # --- Plots. ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available — skipping plots.")
        plt = None

    if plt is not None:
        # Plot 1: per latent, one row per latent, two columns K and V.
        labels = sorted(results["per_latent"].keys(),
                        key=lambda s: (-1 if s == "seed" else int(s.split("_")[-1])))
        n_labels = len(labels)
        fig1, ax1 = plt.subplots(
            nrows=n_labels, ncols=2,
            figsize=(11, max(3, 2.2 * n_labels)),
            squeeze=False,
        )
        for row, label in enumerate(labels):
            for col, key in enumerate(("k_delta", "v_delta")):
                ax = ax1[row][col]
                for L in capture_layers:
                    info = results["per_latent"][label].get(L, {})
                    if "ar_refresh" in info:
                        d = info["ar_refresh"]
                        ax.plot(d["chunks"], d[key],
                                marker="o", linestyle="-", label=f"AR_refresh L{L}")
                    if "append" in info:
                        d = info["append"]
                        ax.plot(d["chunks"], d[key],
                                marker="x", linestyle="--", label=f"append L{L}")
                    if "ar_refresh_once" in info:
                        d = info["ar_refresh_once"]
                        ax.plot(d["chunks"], d[key],
                                marker="^", linestyle=":", label=f"AR_once L{L}")
                ax.set_title(f"{label} — |Δ{key.split('_')[0].upper()}| vs first appearance")
                ax.set_xlabel("chunk idx")
                ax.set_ylabel("mean |Δ|")
                ax.grid(True, alpha=0.3)
                if row == 0 and col == 1:
                    ax.legend(fontsize=7, loc="upper left",
                              bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
        fig1.suptitle("Per-latent K/V drift: AR_refresh (solid) vs append (dashed)", y=1.01)
        fig1.tight_layout()
        p1 = out_dir / "drift_per_latent.png"
        fig1.savefig(p1, dpi=120, bbox_inches="tight")
        plt.close(fig1)
        log.info("Wrote %s", p1)

        # Plot 2: per layer, aggregated over latents.
        n_layers = len(capture_layers)
        fig2, ax2 = plt.subplots(
            nrows=n_layers, ncols=2,
            figsize=(11, max(3, 2.2 * n_layers)),
            squeeze=False,
        )
        for row, L in enumerate(capture_layers):
            for col, key in enumerate(("k_delta", "v_delta")):
                ax = ax2[row][col]
                for label in labels:
                    info = results["per_latent"][label].get(L, {})
                    if "ar_refresh" in info:
                        d = info["ar_refresh"]
                        ax.plot(d["chunks"], d[key],
                                marker="o", linestyle="-",
                                label=f"{label} AR_refresh", alpha=0.85)
                    if "append" in info:
                        d = info["append"]
                        ax.plot(d["chunks"], d[key],
                                marker="x", linestyle="--",
                                label=f"{label} append", alpha=0.85)
                    if "ar_refresh_once" in info:
                        d = info["ar_refresh_once"]
                        ax.plot(d["chunks"], d[key],
                                marker="^", linestyle=":",
                                label=f"{label} AR_once", alpha=0.85)
                ax.set_title(f"Layer {L} — |Δ{key.split('_')[0].upper()}|")
                ax.set_xlabel("chunk idx")
                ax.set_ylabel("mean |Δ|")
                ax.grid(True, alpha=0.3)
                if row == 0 and col == 1:
                    ax.legend(fontsize=7, loc="upper left",
                              bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
        fig2.suptitle("Per-layer K/V drift — one curve per persisting latent", y=1.01)
        fig2.tight_layout()
        p2 = out_dir / "drift_per_layer.png"
        fig2.savefig(p2, dpi=120, bbox_inches="tight")
        plt.close(fig2)
        log.info("Wrote %s", p2)

    # --- JSON dump of numbers. ---
    # Strip tensors / numpy arrays; keep only scalars.
    json_safe = json.loads(json.dumps(results, default=lambda o: float(o)
                                      if hasattr(o, "__float__") else str(o)))
    with (out_dir / "drift.json").open("w") as fh:
        json.dump(json_safe, fh, indent=2)
    log.info("Wrote %s", out_dir / "drift.json")


if __name__ == "__main__":
    main()
