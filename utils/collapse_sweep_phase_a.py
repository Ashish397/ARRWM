#!/usr/bin/env python3
"""Phase A of the rollout-collapse calibration sweep.

Drives ``RollingStaircaseTrainingPipeline`` on a set of held-out rides
under ``inference_mode``, capturing per-rolling-step diagnostic metrics
plus an annotated mp4 per ride. Designed to be paired with a post-hoc
labelling + ROC analysis pass that decides which metric (and threshold)
best correlates with visible rollout collapse.

Output layout (one process per GPU; ``rank`` slices the rides list):

  ``<output_dir>/rank_<RANK>/<ride_dir>/``
    metrics.csv            -- one row per rolling step (incl. the
                              transition commit at step_idx=0)
    meta.json              -- ride metadata + ckpt + config + sweep
                              args snapshot
    rollout.mp4            -- decoded slot-0 commits stacked over GT
    (optional) summary log lives in ``<output_dir>/rank_<RANK>/summary.json``

Per-step metrics
================

The schema below is what every row of ``metrics.csv`` carries. Bonus
columns (``real_residual_*`` / ``real_target_t``) are NaN unless
``--enable_real_score`` is on. Every signal here is computed under
``inference_mode`` from tensors the pipeline already emits inside
``RollingStepOutput`` -- no extra generator forward beyond what the
rolling step already does. Real-score forwards (when enabled) are an
extra teacher pass per rolling step.

  step_idx                int     0-based rolling-step counter; step 0
                                  is the transition commit (kv_anchor
                                  before the first steady-state step),
                                  steps >=1 are steady-state commits.
  global_frame_start      int     Absolute ride latent index where
                                  this slot-0 chunk lives.
  chunk_idx               int     ``global_frame_start // npb``.
  phase                   str     "transition" / "steady" / ...
  slot{i}_t               int     Ladder rung the i-th slot was at on
                                  this rolling step (NaN at step 0,
                                  whose only payload is the
                                  transition's slot-0 anchor).
  slot{i}_rms             float   ``pred_x0_slot_i.flatten(-3).std(-1)
                                  .mean()`` (RMS over channels per
                                  pixel, then averaged over pixels and
                                  frames). M1 raw signal. NaN at step 0.
  slot{i}_peak            float   ``pred_x0_slot_i.abs().amax()``. M2
                                  raw signal. NaN at step 0.
  commit_rms              float   = slot0_rms (chunk being committed).
                                  Populated at every step including
                                  step 0 (uses kv_anchor at step 0).
  commit_peak             float   = slot0_peak. Same.
  commit_to_commit_l2     float   ``||commit_t - commit_{t-1}||_2 /
                                  sqrt(numel)`` (RMS-normalised chunk-
                                  to-chunk delta). NaN at step 0.
  commit_to_gt_l2         float   Same, but commit vs GT at the same
                                  ride position. Lets us separate
                                  "drift away from GT" from
                                  "convergence to a different fixed
                                  point that just freezes".
  gt_chunk_rms            float   ``gt_at_same_pos.flatten(-3).std(-1)
                                  .mean()`` -- baseline reference for
                                  M1 z-score normalisation.
  real_residual_to_student float  M4. ``||pred_real - pred_x0_slot0|| /
                                  sqrt(numel)``: how much the v14
                                  teacher disagrees with the student's
                                  own clean estimate. Big = student
                                  output is OOD as far as the teacher
                                  is concerned. NaN if real-score off.
  real_residual_to_gt     float   ``||pred_real - gt_at_same_pos|| /
                                  sqrt(numel)``: how far the teacher's
                                  own denoising lands from GT. Tracks
                                  how hard the chunk's *position* is
                                  for the teacher independent of
                                  student behaviour.
  real_target_t           float   Timestep the live target was
                                  noisified to before the real-score
                                  forward (default 625, mid of the
                                  trained pool ``[1000, 625, 500,
                                  312.5]``). Comparable across steps
                                  when fixed.
  step_wall_seconds       float   Per-step elapsed time. Cheap sanity
                                  check that nothing's slowing down
                                  pathologically (e.g. CPU-pinned
                                  tensor operations).

Distribution
============

Single-node, 4 GPUs, one Python process per GPU. We do not use
``torch.distributed.init_process_group`` -- there are no collective ops
in this script. Each process reads ``RANK`` / ``WORLD_SIZE`` from env
vars (or falls back to ``--rank`` / ``--world_size``) and processes
the slice ``rides[rank::world_size]`` of the rides list. The wrapper
sbatch (``sbatch/run_collapse_sweep_phase_a.sbatch``) backgrounds 4
``python`` invocations with ``CUDA_VISIBLE_DEVICES=$i`` + ``RANK=$i``.

Usage
=====

::

  # auto-discover 32 rides from the manifest, run on 4 GPUs
  for r in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$r RANK=$r WORLD_SIZE=4 \\
      python utils/collapse_sweep_phase_a.py \\
        --config configs/longlive_phase1_rolling_staircase.yaml \\
        --student_ckpt logs/.../student.pt \\
        --auto_n 32 --auto_min_frames 300 \\
        --output_dir eval/collapse_sweep_$(date +%Y%m%d_%H%M%S) &
  done
  wait

  # explicit rides list with manual offsets
  python utils/collapse_sweep_phase_a.py \\
      --config ... --student_ckpt ... \\
      --rides_json /path/to/rides.json \\
      --output_dir eval/collapse_sweep_<stamp>

  # turn on M4 (real-score residual; +~1 teacher forward per step)
  python utils/collapse_sweep_phase_a.py ... --enable_real_score \\
      --v14_teacher_checkpoint logs/v14_balanced_weunz/causal_lora_step0006600.pt
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_AF_ROOT = _REPO_ROOT / "action-forcing"
if str(_AF_ROOT) not in sys.path:
    sys.path.insert(0, str(_AF_ROOT))

# Reuse the rolling-staircase eval scaffolding so the student / pipeline
# / teacher / decode paths stay bit-identical to the existing
# inference-mode runner.
from utils.eval_rolling_staircase import (  # noqa: E402
    _annotate_and_write_mp4,
    _load_omega,
    _noise_block,
    _teacher_score,
    build_student_and_pipeline,
    build_v14_teacher,
    load_ride_for_rollout,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s | %(message)s",
)
log = logging.getLogger("collapse_sweep_phase_a")


# ---------------------------------------------------------------------------
# Rides discovery
# ---------------------------------------------------------------------------


def _scan_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    """Return ``[{zarr_basename, n_latent_frames}, ...]`` from a manifest.

    The manifest format the rest of the repo writes is a torch-pickled
    dict with a ``"rides"`` list, or a bare list. Each entry has at
    least ``zarr_path`` and ``n_latent_frames`` (sometimes embedded
    inside an ``attrs`` dict).
    """
    raw = torch.load(manifest_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "rides" in raw:
        rides = raw["rides"]
    else:
        rides = raw
    out: List[Dict[str, Any]] = []
    for entry in rides:
        if not isinstance(entry, dict):
            continue
        zpath = entry.get("zarr_path")
        n_lat = entry.get("n_latent_frames")
        if n_lat is None:
            attrs = entry.get("attrs") or {}
            n_lat = attrs.get("n_latent_frames")
        if zpath is None or n_lat is None:
            continue
        try:
            n_lat_i = int(n_lat)
        except Exception:  # noqa: BLE001
            continue
        out.append({
            "zarr_basename": Path(str(zpath)).name,
            "n_latent_frames": n_lat_i,
        })
    return out


def _auto_discover_rides(
    *,
    manifest_path: Optional[str],
    encoded_root: Optional[str],
    n: int,
    min_frames: int,
) -> List[Dict[str, Any]]:
    """Pick ``n`` rides with at least ``min_frames`` latent frames.

    Strategy:
      1. If ``manifest_path`` exists, scan it.
      2. Otherwise, glob ``encoded_root/**/*.zarr`` and probe each.
         (Slower; fine for one-shot discovery.)
    Sorted by descending frame count so we get the longest available
    rides first -- collapse only manifests after enough rolling steps.
    """
    rides: List[Dict[str, Any]] = []
    if manifest_path and Path(manifest_path).exists():
        log.info("Auto-discover: scanning manifest %s", manifest_path)
        rides = _scan_manifest(manifest_path)
    else:
        if not encoded_root:
            raise SystemExit(
                "Auto-discover requires either --manifest or --encoded_root."
            )
        import zarr as zarr_lib  # noqa: F401  (defer import)
        log.info(
            "Auto-discover: globbing %s/**/*.zarr (no manifest -- this can be slow)",
            encoded_root,
        )
        for p in Path(encoded_root).rglob("*.zarr"):
            try:
                grp = __import__("zarr").open_group(str(p), mode="r")
                n_lat = int(grp["latents"].shape[0])
            except Exception:  # noqa: BLE001
                continue
            rides.append({
                "zarr_basename": p.name,
                "n_latent_frames": n_lat,
            })

    rides = [r for r in rides if r["n_latent_frames"] >= int(min_frames)]
    rides.sort(key=lambda r: -int(r["n_latent_frames"]))
    chosen = rides[: int(n)]
    if not chosen:
        raise SystemExit(
            f"Auto-discover found 0 rides with n_latent_frames >= {min_frames}. "
            f"Lower --auto_min_frames or supply --rides_json."
        )
    log.info(
        "Auto-discover: picked %d rides (longest=%d frames, shortest=%d frames)",
        len(chosen), chosen[0]["n_latent_frames"], chosen[-1]["n_latent_frames"],
    )
    out: List[Dict[str, Any]] = []
    for r in chosen:
        stem = Path(r["zarr_basename"]).stem
        out.append({
            "zarr": r["zarr_basename"],
            "offset": 0,
            "tag": f"auto_{stem}",
            "n_latent_frames": int(r["n_latent_frames"]),
        })
    return out


def _load_rides_json(path: str) -> List[Dict[str, Any]]:
    """Read a JSON list of ride descriptors. Each entry must have at
    least ``zarr``; ``offset`` defaults to 0; ``tag`` defaults to the
    zarr stem."""
    rides_raw = json.loads(Path(path).read_text())
    if not isinstance(rides_raw, list):
        raise SystemExit(
            f"--rides_json must be a JSON list; got {type(rides_raw).__name__}."
        )
    out: List[Dict[str, Any]] = []
    for i, e in enumerate(rides_raw):
        if not isinstance(e, dict) or "zarr" not in e:
            raise SystemExit(
                f"--rides_json entry {i} missing required 'zarr' key: {e!r}"
            )
        zarr = str(e["zarr"])
        if not zarr.endswith(".zarr"):
            zarr = f"{zarr}.zarr"
        out.append({
            "zarr": zarr,
            "offset": int(e.get("offset", 0)),
            "tag": str(e.get("tag", Path(zarr).stem)),
            "n_latent_frames": int(e.get("n_latent_frames", -1)),
        })
    return out


# ---------------------------------------------------------------------------
# Per-step metric extraction
# ---------------------------------------------------------------------------


def _chunk_rms(t: torch.Tensor) -> float:
    """``t`` shape ``[B, F, C, H, W]`` -> RMS std over channels per pixel,
    averaged over pixels + frames + batch. Equivalent to "how spread
    are the values around their per-pixel mean" -- a stand-in for
    "is this latent close to the trained latent distribution".

    Implementation note: ``flatten(-3)`` collapses (C, H, W) -> a
    single feature vector per (B, F) pair, then ``std(-1)`` is the
    std of that vector (so the C channels mix into the variance --
    fine, they are jointly distributed in the trained data). That's
    what M1 buys: a single scalar per chunk that tracks both
    deflation (frozen frame -> std collapses to ~0) and inflation
    (noise explosion -> std climbs).
    """
    f = t.detach().float().flatten(-3)  # [B, F, C*H*W]
    return float(f.std(dim=-1).mean().item())


def _chunk_peak(t: torch.Tensor) -> float:
    return float(t.detach().float().abs().amax().item())


def _rms_norm_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    """``||a - b||_2 / sqrt(numel)`` cast to float. Both tensors must
    broadcast to the same shape."""
    diff = (a.detach().float() - b.detach().float()).flatten()
    if diff.numel() == 0:
        return float("nan")
    return float(diff.norm().item() / math.sqrt(diff.numel()))


def _build_metric_row_at_step(
    *,
    step_idx: int,
    step_records: List[Any],
    prev_commit_cpu: Optional[torch.Tensor],
    gt_latents: torch.Tensor,         # [1, T, C, H, W] on device
    npb: int,
    num_slots: int,
    real_score_state: Optional[Dict[str, Any]],
    step_wall_seconds: float,
) -> Tuple[Dict[str, Any], torch.Tensor, Optional[torch.Tensor]]:
    """Compute the per-step metric row for one rolling step.

    Returns ``(row, current_commit_cpu, gt_chunk_cpu_or_none)``. The
    returned commit is what M3 uses on the NEXT call (``prev_commit``
    of the next step). ``gt_chunk_cpu_or_none`` is included in the
    row's record stream so the caller can later assemble a
    side-by-side mp4 without having to slice ``gt_latents`` again.
    """
    # Locate the slot-0 record. The first record of the step also
    # carries the batched-S_n payload (pred_x0_all_slots etc.) for the
    # whole live window.
    first = step_records[0] if step_records else None
    slot0 = None
    for r in step_records:
        if int(getattr(r, "slot_idx", -1)) == 0:
            slot0 = r
            break
    if slot0 is None:
        # Should never happen -- pipeline always emits slot 0 -- but
        # we'd rather log a NaN row than crash a 100-step rollout.
        nan_row = {"step_idx": int(step_idx), "phase": "missing_slot0"}
        return nan_row, prev_commit_cpu if prev_commit_cpu is not None else torch.zeros(0), None

    # Prefer the FINAL-pass committed output (what actually advances
    # the KV cache). For passes_per_step==1 this equals slot0.pred_x0.
    pred_committed = getattr(slot0, "pred_x0_committed", None)
    if pred_committed is None:
        pred_committed = slot0.pred_x0
    pred_committed_dev = pred_committed.detach()

    gfs = int(getattr(slot0, "global_frame_start", -1))
    chunk_idx = gfs // npb if gfs >= 0 else -1
    phase = str(getattr(slot0, "phase", "steady"))

    # GT at the same ride position (if it exists -- end-of-ride padding
    # path matches the pipeline's defensive fallback).
    gt_chunk_dev: Optional[torch.Tensor] = None
    if gfs >= 0 and gfs + npb <= int(gt_latents.shape[1]):
        gt_chunk_dev = gt_latents[:, gfs:gfs + npb].detach()

    row: Dict[str, Any] = {
        "step_idx": int(step_idx),
        "global_frame_start": int(gfs),
        "chunk_idx": int(chunk_idx),
        "phase": phase,
        "step_wall_seconds": float(step_wall_seconds),
    }

    # ----- Per-slot M1/M2 from pred_x0_all_slots if present, else slot0
    # -------------------------------------------------------------
    pred_all = getattr(first, "pred_x0_all_slots", None) if first is not None else None
    per_slot_t = (
        getattr(first, "per_slot_timesteps_list", None)
        if first is not None else None
    )
    for s in range(num_slots):
        if pred_all is not None and pred_all.shape[1] >= (s + 1) * npb:
            slot_chunk = pred_all[:, s * npb : (s + 1) * npb].detach()
            row[f"slot{s}_rms"] = _chunk_rms(slot_chunk)
            row[f"slot{s}_peak"] = _chunk_peak(slot_chunk)
        elif s == 0:
            row[f"slot{s}_rms"] = _chunk_rms(pred_committed_dev)
            row[f"slot{s}_peak"] = _chunk_peak(pred_committed_dev)
        else:
            row[f"slot{s}_rms"] = float("nan")
            row[f"slot{s}_peak"] = float("nan")
        if per_slot_t is not None and s < len(per_slot_t):
            row[f"slot{s}_t"] = int(per_slot_t[s])
        else:
            row[f"slot{s}_t"] = (
                int(getattr(slot0, "slot_timestep", -1)) if s == 0 else -1
            )

    # ----- Commit-stream M1/M2/M3 + commit-vs-GT ------------------
    row["commit_rms"] = row["slot0_rms"]
    row["commit_peak"] = row["slot0_peak"]
    if prev_commit_cpu is not None and tuple(prev_commit_cpu.shape) == tuple(
        pred_committed_dev.shape
    ):
        row["commit_to_commit_l2"] = _rms_norm_l2(
            pred_committed_dev.cpu(), prev_commit_cpu,
        )
    else:
        row["commit_to_commit_l2"] = float("nan")
    if gt_chunk_dev is not None:
        row["commit_to_gt_l2"] = _rms_norm_l2(pred_committed_dev, gt_chunk_dev)
        row["gt_chunk_rms"] = _chunk_rms(gt_chunk_dev)
    else:
        row["commit_to_gt_l2"] = float("nan")
        row["gt_chunk_rms"] = float("nan")

    # ----- Optional real-score residual (M4) ----------------------
    row["real_residual_to_student"] = float("nan")
    row["real_residual_to_gt"] = float("nan")
    row["real_target_t"] = float("nan")

    if real_score_state is not None:
        try:
            res = _real_score_residual_slot0(
                first=first, slot0=slot0, gt_chunk_dev=gt_chunk_dev,
                pred_committed_dev=pred_committed_dev,
                npb=npb, **real_score_state,
            )
            row["real_residual_to_student"] = res["residual_to_student"]
            row["real_residual_to_gt"] = res["residual_to_gt"]
            row["real_target_t"] = float(res["target_t"])
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "real-score residual failed at step %d: %s", step_idx, exc,
            )

    pred_committed_cpu = pred_committed_dev.to(
        device="cpu", dtype=torch.float32,
    ).clone()
    gt_chunk_cpu = (
        gt_chunk_dev.to(device="cpu", dtype=torch.float32).clone()
        if gt_chunk_dev is not None else None
    )
    return row, pred_committed_cpu, gt_chunk_cpu


def _real_score_residual_slot0(
    *,
    first: Any,
    slot0: Any,
    gt_chunk_dev: Optional[torch.Tensor],
    pred_committed_dev: torch.Tensor,
    npb: int,
    teacher: Any,
    action_projection: Any,
    action_token_projection: Any,
    scheduler: Any,
    prompt_embeds: torch.Tensor,
    real_score_num_gt_chunks: int,
    action_decay_per_slot_0: float,
    target_t: int,
) -> Dict[str, float]:
    """Run the v14 teacher on slot 0's S_n^real window (no_grad) and
    return ``{residual_to_student, residual_to_gt, target_t}``.

    Window layout (matches the trainer's
    ``_dmd_loss_batched_sn`` slot-0 case):

        ctx = [GT_{-real_k-1}, ..., GT_{-2}, kv_anchor]   (real_k+1 chunks)
        live = pred_x0_all_slots[:, 0:npb]                (slot 0)
        live_noisy = scheduler.add_noise(live, randn, target_t)

    Then real_score(live_noisy + ctx, t = [0...0, target_t...]) yields
    pred_x0 for the whole window; we compare its target slice against
    ``pred_committed_dev`` (student's clean output, M4 to-student) and
    against ``gt_chunk_dev`` (GT, M4 to-gt).
    """
    if first is None:
        raise RuntimeError("First record missing batched-S_n payload.")
    gt_ctx = getattr(first, "gt_context_chunks", None)
    gt_actions_ctx = getattr(first, "gt_context_action_frames", None)
    kv_anchor = getattr(first, "kv_anchor_chunk", None)
    if gt_ctx is None or gt_actions_ctx is None or kv_anchor is None:
        raise RuntimeError("Batched-S_n payload incomplete on first record.")
    # gt_context_chunks: (real_k + 3) * npb frames, indices
    #   [-real_k-1, ..., +1] relative to slot 0 ride pos.
    # Slot 0 consumes positions [-real_k-1, ..., -2] = first real_k chunks.
    real_k = int(real_score_num_gt_chunks)
    expected_len = (real_k + 3) * npb
    if int(gt_ctx.shape[1]) != expected_len:
        raise RuntimeError(
            f"gt_context_chunks has {int(gt_ctx.shape[1])} frames; "
            f"expected (real_k+3)*npb = {expected_len}."
        )
    gt_chunks_slot0 = gt_ctx[:, : real_k * npb]
    gt_actions_slot0 = gt_actions_ctx[:, : real_k * npb]
    # Anchor action sits at GT-window index real_k (= position -1
    # relative to slot 0). Apply slot-0's action decay (matches the
    # trainer's S_0^real assembly).
    anchor_action = gt_actions_ctx[:, real_k * npb : (real_k + 1) * npb] * float(
        action_decay_per_slot_0
    )

    pred_all = getattr(first, "pred_x0_all_slots")
    live_clean = pred_all[:, :npb].detach()
    target_action = getattr(first, "per_slot_action_frames")[:, :npb].detach()

    # Noisify live target. We don't reseed -- caller's torch.manual_seed
    # already governs noise realisations.
    live_noisy = _noise_block(scheduler, live_clean, int(target_t))

    ctx_chunks = [gt_chunks_slot0[:, k * npb : (k + 1) * npb]
                  for k in range(real_k)] + [kv_anchor.detach()]
    ctx_actions = [gt_actions_slot0[:, k * npb : (k + 1) * npb]
                   for k in range(real_k)] + [anchor_action.detach()]

    pred_target = _teacher_score(
        teacher=teacher,
        action_projection=action_projection,
        action_token_projection=action_token_projection,
        ctx_clean_chunks=ctx_chunks,
        ctx_clean_actions=ctx_actions,
        noisy_target_chunk=live_noisy,
        target_action=target_action,
        target_timestep=int(target_t),
        prompt_embeds=prompt_embeds,
    )

    res_to_student = _rms_norm_l2(pred_target, pred_committed_dev)
    if gt_chunk_dev is not None:
        res_to_gt = _rms_norm_l2(pred_target, gt_chunk_dev)
    else:
        res_to_gt = float("nan")
    return {
        "residual_to_student": float(res_to_student),
        "residual_to_gt": float(res_to_gt),
        "target_t": float(target_t),
    }


# ---------------------------------------------------------------------------
# Per-ride driver
# ---------------------------------------------------------------------------


CSV_FIELDS_FIXED = [
    "step_idx", "global_frame_start", "chunk_idx", "phase",
    "commit_rms", "commit_peak",
    "commit_to_commit_l2", "commit_to_gt_l2", "gt_chunk_rms",
    "real_residual_to_student", "real_residual_to_gt", "real_target_t",
    "step_wall_seconds",
]


def _csv_fieldnames(num_slots: int) -> List[str]:
    slot_fields: List[str] = []
    for s in range(num_slots):
        slot_fields.extend([f"slot{s}_t", f"slot{s}_rms", f"slot{s}_peak"])
    return CSV_FIELDS_FIXED + slot_fields


def run_ride_sweep(
    *,
    ride: Dict[str, Any],
    pipeline: Any,
    vae: Any,
    ode_model: Any,
    teacher: Optional[Any],
    args: argparse.Namespace,
    phase1_cfg: Any,
    device: torch.device,
    dtype: torch.dtype,
    out_dir: Path,
    embedded_step: int,
) -> Dict[str, Any]:
    """Roll a single ride and emit metrics.csv + meta.json + rollout.mp4.

    Returns a dict summarising what happened (n steps, errors, paths).
    """
    zarr = ride["zarr"]
    offset = int(ride["offset"])
    tag = ride["tag"]
    ride_dir = out_dir / f"{tag}__{Path(zarr).stem}__off{offset}"
    ride_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = ride_dir / "metrics.csv"
    meta_path = ride_dir / "meta.json"
    mp4_path = ride_dir / "rollout.mp4"

    log.info("[ride %s|off=%d|tag=%s] start", zarr, offset, tag)
    t_ride_start = time.time()

    # Drop any caches left over from the previous ride so memory doesn't
    # leak across iterations. ``rollout_ride`` reallocates on entry.
    pipeline.kv_cache1 = None
    pipeline.crossattn_cache = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load ride.
    action_dims = list(getattr(phase1_cfg, "action_dims", [2, 7]))
    npb = int(getattr(phase1_cfg, "num_frame_per_block", 3))
    prime_kv_frames = int(getattr(phase1_cfg, "prime_kv_frames", 9))

    gt_latents, prompt_embeds, gt_actions, ride_meta = load_ride_for_rollout(
        zarr_basename=zarr,
        latent_start_offset=offset,
        total_frames=(int(args.max_ride_frames)
                      if args.max_ride_frames is not None else None),
        manifest_path=args.manifest,
        encoded_root=args.encoded_root,
        caption_root=args.caption_root,
        motion_root=args.motion_root,
        ss_vae_checkpoint=args.ss_vae_checkpoint,
        action_dims=action_dims,
        device=device,
    )
    gt_latents = gt_latents.to(dtype=dtype)
    gt_actions = gt_actions.to(dtype=dtype)
    prompt_embeds = prompt_embeds.to(dtype=dtype)

    log.info(
        "[ride %s] latents=%s actions=%s prompt=%s",
        zarr, tuple(gt_latents.shape), tuple(gt_actions.shape),
        tuple(prompt_embeds.shape),
    )

    # Optional real-score state passed through into the metric builder.
    real_score_state: Optional[Dict[str, Any]] = None
    if teacher is not None:
        action_decay_table = list(
            getattr(phase1_cfg, "action_decay_per_slot", [1.0, 0.75, 0.5, 0.25])
        )
        real_score_state = {
            "teacher": teacher,
            "action_projection": ode_model.action_projection,
            "action_token_projection": ode_model.action_token_projection,
            "scheduler": ode_model.scheduler,
            "prompt_embeds": prompt_embeds,
            "real_score_num_gt_chunks": int(
                getattr(phase1_cfg, "real_score_num_gt_chunks", 2)
            ),
            "action_decay_per_slot_0": float(action_decay_table[0]),
            "target_t": int(args.real_target_t),
        }

    num_slots = int(pipeline.num_live_slots)
    fieldnames = _csv_fieldnames(num_slots)

    rows: List[Dict[str, Any]] = []
    commits: List[Dict[str, Any]] = []
    transition_captured = False
    prev_commit_cpu: Optional[torch.Tensor] = None
    err: Optional[str] = None

    try:
        with torch.inference_mode():
            it = pipeline.rollout_ride(
                gt_latents=gt_latents,
                gt_actions=gt_actions,
                prompt_embeds=prompt_embeds,
                max_rolling_steps=(
                    int(args.max_rolling_steps)
                    if args.max_rolling_steps is not None else None
                ),
            )
            for step_idx, step_records in enumerate(it):
                t_step = time.time()
                if not step_records:
                    continue

                # Capture transition commit ONCE on the very first
                # step's first record (kv_anchor before steady-state).
                if not transition_captured:
                    first = step_records[0]
                    kv_anchor = getattr(first, "kv_anchor_chunk", None)
                    if torch.is_tensor(kv_anchor):
                        anchor_cpu = kv_anchor.detach().to(
                            device="cpu", dtype=torch.float32,
                        ).clone()
                        # GT-alignment position of the transition
                        # anchor. The transition commits warmup-pass-3
                        # slot-0 (action a_3 in the sliding-action
                        # convention), so its GT-aligned position is
                        # ``n_primed + 3*npb``. NW = NS*P is fixed at
                        # 4 (ODE-distill invariant; enforced inside
                        # the pipeline ctor) so the offset is always
                        # ``(NW-1)*npb = 3*npb`` regardless of the
                        # NS/P split. Note: this is INDEPENDENT of the
                        # cache's logical slot for the anchor, which
                        # the transition path commits at logical
                        # position ``n_primed`` to keep the cache
                        # contiguous.
                        nw = (
                            int(pipeline.num_live_slots)
                            * int(getattr(pipeline, "passes_per_step", 1))
                        )
                        gfs_anchor = int(prime_kv_frames) + (nw - 1) * npb
                        gt_anchor_dev: Optional[torch.Tensor] = None
                        if gfs_anchor >= 0 and gfs_anchor + npb <= int(
                            gt_latents.shape[1]
                        ):
                            gt_anchor_dev = gt_latents[
                                :, gfs_anchor : gfs_anchor + npb
                            ].detach()
                        gt_anchor_cpu = (
                            gt_anchor_dev.to(
                                device="cpu", dtype=torch.float32,
                            ).clone()
                            if gt_anchor_dev is not None else None
                        )
                        anchor_row = {
                            "step_idx": 0,
                            "global_frame_start": int(gfs_anchor),
                            "chunk_idx": (
                                gfs_anchor // npb if gfs_anchor >= 0 else -1
                            ),
                            "phase": "transition",
                            "commit_rms": _chunk_rms(kv_anchor),
                            "commit_peak": _chunk_peak(kv_anchor),
                            "commit_to_commit_l2": float("nan"),
                            "commit_to_gt_l2": (
                                _rms_norm_l2(kv_anchor, gt_anchor_dev)
                                if gt_anchor_dev is not None else float("nan")
                            ),
                            "gt_chunk_rms": (
                                _chunk_rms(gt_anchor_dev)
                                if gt_anchor_dev is not None else float("nan")
                            ),
                            "real_residual_to_student": float("nan"),
                            "real_residual_to_gt": float("nan"),
                            "real_target_t": float("nan"),
                            "step_wall_seconds": 0.0,
                        }
                        for s in range(num_slots):
                            anchor_row[f"slot{s}_t"] = -1
                            anchor_row[f"slot{s}_rms"] = float("nan")
                            anchor_row[f"slot{s}_peak"] = float("nan")
                        rows.append(anchor_row)
                        commits.append({
                            "latent": anchor_cpu,
                            "gt_latent": gt_anchor_cpu,
                            "action": None,
                            "chunk_index": (
                                gfs_anchor // npb if gfs_anchor >= 0 else -1
                            ),
                            "global_frame_start": int(gfs_anchor),
                            "slot_timestep": 0,
                            "phase": "transition",
                            "rolling_steps_done": 0,
                        })
                        prev_commit_cpu = anchor_cpu
                    transition_captured = True

                step_wall = time.time() - t_step
                row, commit_cpu, gt_chunk_cpu = _build_metric_row_at_step(
                    step_idx=step_idx + 1,  # +1 so transition lives at idx 0
                    step_records=step_records,
                    prev_commit_cpu=prev_commit_cpu,
                    gt_latents=gt_latents,
                    npb=npb,
                    num_slots=num_slots,
                    real_score_state=real_score_state,
                    step_wall_seconds=step_wall,
                )
                rows.append(row)

                # Stash the commit (for mp4) along with the per-frame
                # action and chunk index. We pull the action from the
                # slot-0 record's ``action_frame`` if present.
                slot0_rec = next(
                    (r for r in step_records
                     if int(getattr(r, "slot_idx", -1)) == 0),
                    None,
                )
                action_frame = (
                    getattr(slot0_rec, "action_frame", None)
                    if slot0_rec is not None else None
                )
                if torch.is_tensor(action_frame) and action_frame.numel() > 0:
                    action_vec = action_frame[0].detach().float().mean(
                        dim=0
                    ).cpu().clone()
                else:
                    action_vec = None
                commits.append({
                    "latent": commit_cpu,
                    "gt_latent": gt_chunk_cpu,
                    "action": action_vec,
                    "chunk_index": int(row.get("chunk_idx", -1)),
                    "global_frame_start": int(row.get("global_frame_start", -1)),
                    "slot_timestep": int(row.get("slot0_t", -1)),
                    "phase": str(row.get("phase", "steady")),
                    "rolling_steps_done": step_idx + 1,
                })

                prev_commit_cpu = commit_cpu

                # Periodic progress log.
                if (step_idx + 1) % 50 == 0:
                    log.info(
                        "[ride %s] step %d | rms=%.3f peak=%.3f c2c=%.3f c2gt=%.3f",
                        zarr, step_idx + 1, row.get("commit_rms", float("nan")),
                        row.get("commit_peak", float("nan")),
                        row.get("commit_to_commit_l2", float("nan")),
                        row.get("commit_to_gt_l2", float("nan")),
                    )
    except Exception:  # noqa: BLE001
        err = traceback.format_exc()
        log.error("[ride %s] rollout failed: %s", zarr, err)

    # ----- Write metrics.csv ------------------------------------------
    with metrics_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, float("nan")) for k in fieldnames})
    log.info("[ride %s] metrics.csv: %d rows -> %s", zarr, len(rows), metrics_path)

    # ----- Write mp4 (best-effort; failure is non-fatal) -------------
    mp4_stats: Optional[Dict[str, Any]] = None
    if args.decode_mp4 and len(commits) > 0 and err is None:
        try:
            prime = gt_latents[:, :prime_kv_frames].detach().clone().to(
                device="cpu", dtype=torch.float32,
            )
            mp4_stats = _annotate_and_write_mp4(
                prime_latents=prime,
                commits=commits,
                vae=vae,
                device=device,
                fps=int(args.fps),
                out_path=mp4_path,
                ride_tag=tag,
                ride_basename=Path(zarr).stem,
                num_frame_per_block=npb,
                embedded_step=int(embedded_step),
            )
        except Exception:  # noqa: BLE001
            log.warning("[ride %s] mp4 decode failed: %s", zarr,
                        traceback.format_exc())
            mp4_stats = {"error": "decode_failed"}

    # ----- Write meta.json -------------------------------------------
    summary = {
        "zarr": zarr,
        "offset": offset,
        "tag": tag,
        "ride_meta": ride_meta,
        "n_rolling_steps_captured": len(rows),
        "embedded_step": int(embedded_step),
        "real_score_enabled": real_score_state is not None,
        "real_target_t": int(args.real_target_t),
        "wall_seconds": time.time() - t_ride_start,
        "mp4_path": str(mp4_path) if mp4_stats is not None else None,
        "mp4_stats": mp4_stats,
        "error": err,
        "config": {
            "config": str(args.config),
            "ode_config": str(args.ode_config),
            "student_ckpt": str(args.student_ckpt),
            "real_score_num_gt_chunks": int(
                getattr(phase1_cfg, "real_score_num_gt_chunks", 2)
            ),
            "num_live_slots": int(pipeline.num_live_slots),
            "passes_per_step": int(getattr(pipeline, "passes_per_step", 1)),
            "num_frame_per_block": int(npb),
            "prime_kv_frames": int(prime_kv_frames),
            "max_rolling_steps": (
                int(args.max_rolling_steps)
                if args.max_rolling_steps is not None else None
            ),
            "max_ride_frames": (
                int(args.max_ride_frames)
                if args.max_ride_frames is not None else None
            ),
        },
    }
    meta_path.write_text(json.dumps(summary, indent=2, default=str))

    log.info(
        "[ride %s] DONE | %d steps | wall=%.1fs | mp4=%s",
        zarr, len(rows), summary["wall_seconds"],
        bool(mp4_stats and "error" not in (mp4_stats or {})),
    )
    return summary


# ---------------------------------------------------------------------------
# Argument parsing + main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config", type=str,
        default="configs/longlive_phase1_rolling_staircase.yaml",
    )
    p.add_argument(
        "--ode_config", type=str,
        default="configs/action_ode_distill.yaml",
    )
    p.add_argument(
        "--student_ckpt", type=str, required=True,
        help="Phase-1-compatible student snapshot (default at ODE init: "
             "logs/action_ode_distill_E/action_ode_step0001000.pt).",
    )
    p.add_argument(
        "--output_dir", type=str, required=True,
        help="Sweep output root. Per-rank subdirs land under "
             "<output_dir>/rank_<RANK>/.",
    )

    # ---- Rides selection ----
    p.add_argument(
        "--rides_json", type=str, default=None,
        help="Path to a JSON list of {zarr, offset, tag} entries. "
             "If absent we auto-discover via --auto_n / "
             "--auto_min_frames using --manifest (preferred) or "
             "--encoded_root (slow).",
    )
    p.add_argument(
        "--auto_n", type=int, default=32,
        help="Number of rides to auto-pick when --rides_json is None.",
    )
    p.add_argument(
        "--auto_min_frames", type=int, default=300,
        help="Minimum n_latent_frames per auto-picked ride (~100 "
             "rolling steps at npb=3).",
    )

    # ---- Distribution ----
    p.add_argument(
        "--rank", type=int, default=int(os.environ.get("RANK", 0)),
        help="This process's rank (0-indexed). Defaults to env RANK.",
    )
    p.add_argument(
        "--world_size", type=int,
        default=int(os.environ.get("WORLD_SIZE", 1)),
        help="Total number of processes. Defaults to env WORLD_SIZE.",
    )

    # ---- Rollout config ----
    p.add_argument(
        "--max_ride_frames", type=int, default=None,
        help="Optional cap on per-ride latent frames. Default = roll "
             "to natural ride end.",
    )
    p.add_argument(
        "--max_rolling_steps", type=int, default=None,
        help="Optional safety cap on rolling-step count. Default = no cap.",
    )

    # ---- Real-score (M4) ----
    p.add_argument(
        "--enable_real_score", action="store_true",
        help="Compute M4 (real-score residual) at every rolling step. "
             "Requires --v14_teacher_checkpoint.",
    )
    p.add_argument(
        "--v14_teacher_checkpoint", type=str,
        default=("/scratch/u6ex/as1748.u6ex/ARRWM/logs/v14_balanced_weunz/"
                 "causal_lora_step0006600.pt"),
    )
    p.add_argument("--v14_lora_rank", type=int, default=256)
    p.add_argument("--v14_lora_alpha", type=float, default=256.0)
    p.add_argument("--v14_lora_dropout", type=float, default=0.0)
    p.add_argument(
        "--real_target_t", type=int, default=625,
        help="Timestep used to noisify slot-0's live target before "
             "the real-score forward. 625 = mid of the trained pool "
             "[1000, 625, 500, 312.5]; comparable across steps when "
             "fixed.",
    )

    # ---- Decode / mp4 ----
    p.add_argument(
        "--decode_mp4", action="store_true", default=True,
        help="Decode commits + GT into a side-by-side mp4 per ride "
             "(default ON).",
    )
    p.add_argument(
        "--no_decode_mp4", dest="decode_mp4", action="store_false",
        help="Skip the mp4 decode (saves ~30s per ride).",
    )
    p.add_argument("--fps", type=int, default=8)

    # ---- Data paths ----
    p.add_argument("--manifest", type=str, default=None)
    p.add_argument(
        "--encoded_root", type=str,
        default="/projects/u6ex/fbots/frodobots_encoded_weu",
    )
    p.add_argument(
        "--caption_root", type=str,
        default="/projects/u6ex/fbots/frodobots_captions/train",
    )
    p.add_argument(
        "--motion_root", type=str,
        default="/projects/u6ex/fbots/frodobots_motion",
    )
    p.add_argument(
        "--ss_vae_checkpoint", type=str,
        default="action_query/checkpoints/ss_vae_8free.pt",
    )

    # ---- Misc ----
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    p.add_argument(
        "--disable_renoise", action="store_true",
        help="Mirrors eval_rolling_staircase: skip the staircase's "
             "carryover-renoise step.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    rank = int(args.rank)
    world_size = max(int(args.world_size), 1)
    if rank < 0 or rank >= world_size:
        raise SystemExit(
            f"Invalid rank/world_size: rank={rank} world={world_size}."
        )

    # Device.
    gpu_id = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    log.info(
        "rank=%d/%d device=%s dtype=%s | config=%s student=%s",
        rank, world_size, device, dtype, args.config, args.student_ckpt,
    )

    torch.manual_seed(int(args.seed) + rank * 1009)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed) + rank * 1009)

    # ---- Build student + pipeline + VAE -----------------------------
    ode_model, pipeline, vae = build_student_and_pipeline(
        phase1_cfg_path=args.config,
        ode_cfg_path=args.ode_config,
        student_ckpt_path=args.student_ckpt,
        device=device,
        dtype=dtype,
        disable_renoise=bool(args.disable_renoise),
    )
    for m in (ode_model,):
        for p_ in m.parameters():
            p_.requires_grad_(False)
        m.eval()

    # ---- Optional v14 teacher ---------------------------------------
    teacher: Optional[Any] = None
    if args.enable_real_score:
        if not Path(args.v14_teacher_checkpoint).exists():
            raise SystemExit(
                f"--enable_real_score requires --v14_teacher_checkpoint "
                f"to exist; given {args.v14_teacher_checkpoint}"
            )
        # Real-score window for slot 0 at training time: real_k GT
        # chunks + 1 prev (kv_anchor) + 1 live = (real_k + 2) * npb
        # frames. We size the teacher to that.
        phase1_cfg = _load_omega(args.config)
        npb = int(getattr(phase1_cfg, "num_frame_per_block", 3))
        real_k = int(getattr(phase1_cfg, "real_score_num_gt_chunks", 2))
        teacher_window_chunks = real_k + 2
        teacher_num_context_frames = teacher_window_chunks * npb
        teacher = build_v14_teacher(
            v14_ckpt_path=args.v14_teacher_checkpoint,
            v14_lora_rank=int(args.v14_lora_rank),
            v14_lora_alpha=float(args.v14_lora_alpha),
            v14_lora_dropout=float(args.v14_lora_dropout),
            num_context_frames=teacher_num_context_frames,
            action_per_frame=1,
            device=device,
            dtype=dtype,
        )
        log.info(
            "real-score teacher loaded: window=%d chunks (%d frames) target_t=%d",
            teacher_window_chunks, teacher_num_context_frames,
            int(args.real_target_t),
        )

    phase1_cfg = _load_omega(args.config)

    # ---- Rides list -------------------------------------------------
    if args.rides_json:
        rides_full = _load_rides_json(args.rides_json)
        log.info("Loaded %d rides from %s", len(rides_full), args.rides_json)
    else:
        rides_full = _auto_discover_rides(
            manifest_path=args.manifest,
            encoded_root=args.encoded_root,
            n=int(args.auto_n),
            min_frames=int(args.auto_min_frames),
        )

    my_rides = rides_full[rank::world_size]
    log.info(
        "rank=%d processing %d/%d rides", rank, len(my_rides), len(rides_full),
    )

    out_root = Path(args.output_dir)
    rank_dir = out_root / f"rank_{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)

    # Persist the global rides list once on rank 0 so the aggregator
    # has a stable manifest of what was supposed to run.
    if rank == 0:
        (out_root / "rides_full.json").write_text(
            json.dumps(rides_full, indent=2, default=str)
        )

    # Recover embedded step from ckpt name (cosmetic; for mp4 corner tag).
    embedded_step = -1
    mm = re.search(r"step(\d+)", Path(args.student_ckpt).name)
    if mm:
        try:
            embedded_step = int(mm.group(1))
        except ValueError:
            embedded_step = -1

    # ---- Per-ride loop ---------------------------------------------
    rank_summaries: List[Dict[str, Any]] = []
    t_rank_start = time.time()
    for i, ride in enumerate(my_rides):
        log.info("rank=%d ride %d/%d", rank, i + 1, len(my_rides))
        try:
            summary = run_ride_sweep(
                ride=ride,
                pipeline=pipeline,
                vae=vae,
                ode_model=ode_model,
                teacher=teacher,
                args=args,
                phase1_cfg=phase1_cfg,
                device=device,
                dtype=dtype,
                out_dir=rank_dir,
                embedded_step=embedded_step,
            )
        except Exception:  # noqa: BLE001
            err = traceback.format_exc()
            log.error("rank=%d ride %s failed top-level: %s",
                      rank, ride.get("zarr", "?"), err)
            summary = {"zarr": ride.get("zarr"), "error": err}
        rank_summaries.append(summary)

        # Aggressive cleanup between rides.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Rank-level summary ----------------------------------------
    rank_summary = {
        "rank": rank,
        "world_size": world_size,
        "n_rides": len(my_rides),
        "wall_seconds": time.time() - t_rank_start,
        "rides": rank_summaries,
    }
    (rank_dir / "summary.json").write_text(
        json.dumps(rank_summary, indent=2, default=str)
    )
    log.info(
        "rank=%d DONE | %d rides | wall=%.1fs | summary=%s",
        rank, len(my_rides), rank_summary["wall_seconds"],
        rank_dir / "summary.json",
    )


if __name__ == "__main__":
    main()
