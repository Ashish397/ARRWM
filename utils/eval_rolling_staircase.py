#!/usr/bin/env python3
"""Rolling-staircase inference for the ODE-distilled student.

What this script does is exactly what Phase-1 training does, with the
loss + teacher + fake-score stack removed:

  1. Build the ODE student (``ODERegression``, which also merges the
     teacher's LoRA into the DiT and builds ``action_projection`` /
     ``action_token_projection`` / ``state_probe`` / ``action_critic``).
  2. Overlay a Phase-1-compatible student snapshot (``generator`` +
     ``action_projection`` + ``action_token_projection`` keys). If the
     caller passes the 1000-step ODE init checkpoint (the default
     when ``--student_ckpt`` points at
     ``action_ode_distill_E/action_ode_step0001000.pt``), the only
     overlay is the ODE init itself and the student runs exactly as it
     would at the very first iteration of Phase-1 training.
  3. Wrap the student's generator with ``RollingStaircaseTrainingPipeline``
     using the same topology knobs as the Phase-1 config
     (``num_live_slots``, ``passes_per_step``, ``denoising_step_list``,
     ``action_decay_per_slot``, ``local_attn_size``, etc.).
  4. For one ride per rank, run ``pipeline.rollout_ride`` under
     ``torch.inference_mode()`` and capture the slot-0 committed chunk
     yielded by each rolling step.
  5. Decode the committed chunks through the Wan VAE, annotate each
     frame (ride label + chunk index + commanded action bars) in the
     same style ``eval_chain.annotate_video`` uses, and write an mp4
     per rank via ``frames_to_mp4`` (ffmpeg).

Launch model: single-GPU-per-process, one zarr per process, so you can
spread 4 GPUs across 4 distinct rides in a single sbatch. Mirrors
``utils/eval_causal_AR.py``'s layout:

  CUDA_VISIBLE_DEVICES=$GPU WORLD_SIZE=1 LOCAL_RANK=0 \\
      python utils/eval_rolling_staircase.py \\
          --config configs/longlive_phase1_rolling_staircase.yaml \\
          --student_ckpt logs/action_ode_distill_E/action_ode_step0001000.pt \\
          --rank_zarr 20240216101235.zarr \\
          --rank_offset 100 \\
          --rank_tag madrid_held \\
          --output_dir eval/eval_rolling_staircase_<ts>/gpu0_madrid

The ODE config referenced by ``--ode_config`` (defaults to
``configs/action_ode_distill.yaml``) is used ONLY to build the
``ODERegression`` shell. All rolling-staircase knobs are read from the
Phase-1 config ``--config`` (so the two live in exactly the same
representation that Phase-1 training reads).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# af_model + streaming helpers live under action-forcing/.
_AF_ROOT = _REPO_ROOT / "action-forcing"
if str(_AF_ROOT) not in sys.path:
    sys.path.insert(0, str(_AF_ROOT))

from utils.eval_chain import (
    frames_to_mp4,
)
from utils.eval_causal_AR import load_per_rank_ride_ar

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Student + pipeline construction
# ---------------------------------------------------------------------------


def _load_omega(path: str):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(path)
    OmegaConf.set_struct(cfg, False)
    return cfg


def build_student_and_pipeline(
    phase1_cfg_path: str,
    ode_cfg_path: str,
    student_ckpt_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Any, Any, Any]:
    """Build and return ``(ode_model, pipeline, vae)``.

    ``ode_model`` is a live ``ODERegression`` with the student overlay
    applied. ``pipeline`` is a ``RollingStaircaseTrainingPipeline``
    wrapping the student's generator. ``vae`` is a fresh ``WanVAEWrapper``
    on ``device`` (kept separate from ``ode_model`` because
    ``ODERegression`` does not expose it directly — we build our own
    copy for the decode path).
    """
    phase1_cfg = _load_omega(phase1_cfg_path)
    ode_cfg = _load_omega(ode_cfg_path)

    # NOTE: we deliberately do NOT flip ``use_motion_pipeline`` off in
    # the config — ODERegression.__init__ raises on that path. Instead
    # we simply never call ``ensure_motion_pipeline()``; the
    # VAE/CoTracker/SS-VAE stack is only exercised when that method is
    # invoked, and it's gated by ``self._motion_pipeline_ready`` flag.
    # We're not training or computing action-teacher targets here so
    # the pipeline is unused.
    from af_model.ode_regression import ODERegression
    log.info("Building ODERegression (merges teacher LoRA into DiT)...")
    ode_model = ODERegression(ode_cfg, device=device).eval()

    # Overlay the student checkpoint (full-rank DiT + action heads).
    log.info("Overlaying student snapshot: %s", student_ckpt_path)
    raw = torch.load(student_ckpt_path, map_location="cpu", weights_only=False)
    if "generator" not in raw:
        raise RuntimeError(
            f"Student checkpoint {student_ckpt_path} missing 'generator' key; "
            "expected a full-rank ODE-distill snapshot (train.py writes "
            "'generator' / 'action_projection' / 'action_token_projection')."
        )
    missing, unexpected = ode_model.generator.model.load_state_dict(
        raw["generator"], strict=False,
    )
    if missing:
        log.warning("generator: %d missing keys (first 5: %s)",
                    len(missing), list(missing)[:5])
    if unexpected:
        log.warning("generator: %d unexpected keys (first 5: %s)",
                    len(unexpected), list(unexpected)[:5])
    if ode_model.action_projection is not None and "action_projection" in raw:
        ode_model.action_projection.load_state_dict(raw["action_projection"])
        log.info("Loaded action_projection.")
    if (
        ode_model.action_token_projection is not None
        and "action_token_projection" in raw
    ):
        ode_model.action_token_projection.load_state_dict(
            raw["action_token_projection"]
        )
        log.info("Loaded action_token_projection.")
    embedded_step = int(raw.get("step", -1))
    del raw
    log.info(
        "Student overlay OK (embedded step=%s).",
        embedded_step if embedded_step >= 0 else "?",
    )

    # Cast to the eval dtype. The VAE stays in fp32 (its convs have fp32
    # biases and overflow silently in bf16 — matches ``BaseModel`` behavior
    # in the main trainer).
    ode_model.generator.model.to(device=device, dtype=dtype)
    if ode_model.action_projection is not None:
        ode_model.action_projection.to(device=device, dtype=dtype)
    if ode_model.action_token_projection is not None:
        ode_model.action_token_projection.to(device=device, dtype=dtype)
    # state_probe / action_critic are NOT exercised here; skip their
    # dtype casts to keep memory down.

    # Build the rolling-staircase pipeline from the same knobs Phase-1
    # training reads. We route through the Phase-1 YAML so the two are
    # guaranteed to match.
    from pipeline.rolling_staircase_training import RollingStaircaseTrainingPipeline

    denoising_step_list = list(
        getattr(phase1_cfg, "denoising_step_list", [1000, 750, 500, 250])
    )
    num_live_slots = int(getattr(phase1_cfg, "num_live_slots", 4))
    passes_per_step = int(getattr(phase1_cfg, "passes_per_step", 1))
    default_decay = (
        [1.0, 0.75, 0.5, 0.25] if num_live_slots == 4 else [1.0, 0.5]
    )
    action_decay_slot = tuple(
        getattr(phase1_cfg, "action_decay_per_slot", default_decay)
    )

    pipeline = RollingStaircaseTrainingPipeline(
        denoising_step_list=denoising_step_list,
        scheduler=ode_model.scheduler,
        generator=ode_model.generator,
        num_frame_per_block=int(getattr(phase1_cfg, "num_frame_per_block", 3)),
        num_slots=num_live_slots,
        passes_per_step=passes_per_step,
        action_decay_per_slot=action_decay_slot,
        prime_kv_frames=int(getattr(phase1_cfg, "prime_kv_frames", 9)),
        kv_frames_total=int(getattr(phase1_cfg, "kv_frames_total", 21)),
        kv_committed_max_frames=int(
            getattr(phase1_cfg, "kv_committed_max_frames", 9)
        ),
        local_attn_size=int(getattr(phase1_cfg, "local_attn_size", 21)),
        action_projection=ode_model.action_projection,
        action_token_projection=ode_model.action_token_projection,
        real_score_num_gt_chunks=int(
            getattr(phase1_cfg, "real_score_num_gt_chunks", 2)
        ),
    )
    log.info(
        "Pipeline: num_live_slots=%d passes_per_step=%d denoising_step_list=%s "
        "action_decay_per_slot=%s kv_committed_max_frames=%d local_attn_size=%d",
        num_live_slots, passes_per_step, denoising_step_list,
        list(action_decay_slot),
        int(getattr(phase1_cfg, "kv_committed_max_frames", 9)),
        int(getattr(phase1_cfg, "local_attn_size", 21)),
    )

    # Separate VAE copy for decode (ODERegression doesn't expose one).
    from utils.wan_wrapper import WanVAEWrapper
    log.info("Loading Wan VAE (fp32) for decode...")
    vae = WanVAEWrapper()
    vae.to(device=device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    return ode_model, pipeline, vae


# ---------------------------------------------------------------------------
# Rollout + capture
# ---------------------------------------------------------------------------


def run_rollout_and_capture(
    *,
    pipeline: Any,
    gt_latents: torch.Tensor,     # [1, T, C, H, W]
    gt_actions: torch.Tensor,     # [1, T, action_dim]
    prompt_embeds: torch.Tensor,  # [1, L, C_txt]
    num_frame_per_block: int,
    prime_kv_frames: int,
    max_rolling_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the rolling-staircase rollout in inference mode and collect
    the slot-0 committed latents for visualization.

    Returns a dict with:
      - ``prime_latents``:    [1, prime_kv_frames, C, H, W] — the clean
                              GT latents pushed into the KV cache by
                              ``warmup_prime_kv_cache`` (same frames the
                              student's "past" sees at the start of the
                              rollout). Drawn at the front of the mp4
                              with a ``prime`` label.
      - ``commits``:          list of dict entries, each containing:
                                * ``latent``: [1, npb, C, H, W] fp32 CPU
                                * ``action``: [action_dim] fp32 CPU
                                  (mean-pooled over the 3 frames)
                                * ``chunk_index``: int
                                * ``global_frame_start``: int
                                * ``slot_timestep``: int
                                * ``phase``: str ("transition"/"steady")
                                * ``rolling_steps_done``: int
      - ``stats``: dict of per-step wall-clock and per-ride counters.
    """
    device = gt_latents.device
    dtype = gt_latents.dtype
    npb = int(num_frame_per_block)

    # Seed the prime view from the front of the ride — same slice the
    # pipeline's ``warmup_prime_kv_cache`` will commit internally. We
    # snapshot it HERE so the mp4 can show a ``prime`` header before the
    # generated commits even though the pipeline doesn't yield records
    # for the prime forward.
    prime = gt_latents[:, :prime_kv_frames].detach().clone().to(
        device="cpu", dtype=torch.float32,
    )

    commits: List[Dict[str, Any]] = []
    transition_captured = False
    t0 = time.time()
    last_log = t0

    with torch.inference_mode():
        it = pipeline.rollout_ride(
            gt_latents=gt_latents,
            gt_actions=gt_actions,
            prompt_embeds=prompt_embeds,
            max_rolling_steps=max_rolling_steps,
        )
        for step_idx, step_records in enumerate(it):
            if not step_records:
                continue
            # First yielded step exposes the transition-commit anchor
            # (kv_anchor_chunk) on its first record — this is the clean
            # slot-0 output produced by the warmup+transition path,
            # committed into the cache BEFORE the first steady-state
            # rolling step. Push it once so it appears between prime
            # and the first steady commit.
            if not transition_captured:
                first = step_records[0]
                kv_anchor = getattr(first, "kv_anchor_chunk", None)
                if torch.is_tensor(kv_anchor):
                    commits.append({
                        "latent": kv_anchor.detach().to(
                            device="cpu", dtype=torch.float32,
                        ).clone(),
                        "action": None,
                        "chunk_index": -1,
                        "global_frame_start": -1,
                        "slot_timestep": 0,
                        "phase": "transition",
                        "rolling_steps_done": 0,
                    })
                transition_captured = True

            # Find the slot-0 record for this step (one per rolling step
            # regardless of how many slots the pipeline was configured
            # with — slot 0 is always the cleanest and is always the
            # chunk being committed).
            slot0 = None
            for r in step_records:
                if int(getattr(r, "slot_idx", -1)) == 0:
                    slot0 = r
                    break
            if slot0 is None:
                continue

            pred = slot0.pred_x0.detach().to(
                device="cpu", dtype=torch.float32,
            ).clone()
            action_frame = getattr(slot0, "action_frame", None)
            if torch.is_tensor(action_frame) and action_frame.numel() > 0:
                action_vec = (
                    action_frame[0].detach().float().mean(dim=0).cpu().clone()
                )
            else:
                action_vec = None

            gfs = int(getattr(slot0, "global_frame_start", -1))
            commits.append({
                "latent": pred,
                "action": action_vec,
                "chunk_index": gfs // npb if gfs >= 0 else -1,
                "global_frame_start": gfs,
                "slot_timestep": int(getattr(slot0, "slot_timestep", -1)),
                "phase": str(getattr(slot0, "phase", "steady")),
                "rolling_steps_done": step_idx + 1,
            })

            # Lightweight progress log every 15 s.
            now = time.time()
            if now - last_log >= 15.0:
                log.info(
                    "rollout: %d rolling steps captured | wall=%.1fs | last chunk=%d",
                    len(commits), now - t0, commits[-1]["chunk_index"],
                )
                last_log = now

    t1 = time.time()
    stats = {
        "n_commits": len(commits),
        "wall_seconds": t1 - t0,
        "rolling_steps_per_sec": (
            (len(commits) - (1 if transition_captured else 0))
            / max(t1 - t0, 1e-6)
        ),
    }
    log.info(
        "rollout done: %d commits (transition=%s) | %.1fs | %.3f steps/s",
        len(commits), transition_captured, stats["wall_seconds"],
        stats["rolling_steps_per_sec"],
    )
    return {
        "prime_latents": prime,
        "commits": commits,
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# Decode + annotate + write mp4
# ---------------------------------------------------------------------------


def _pixels_to_uint8_hwc(pixels: torch.Tensor) -> np.ndarray:
    """Wan VAE output → ``[T, H, W, 3] uint8`` in [0, 255]. Same shape
    coercion the Phase-1 vis uses; see ``utils.multislot_vis``."""
    if pixels.dim() == 5:
        pixels = pixels[0]
    if pixels.dim() != 4:
        raise RuntimeError(f"unexpected VAE output shape: {pixels.shape}")
    if pixels.shape[0] == 3 and pixels.shape[1] != 3:
        pixels = pixels.permute(1, 0, 2, 3)
    pixels = pixels.permute(0, 2, 3, 1).contiguous()
    if pixels.min().item() < -0.1:
        arr = ((pixels.clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
    else:
        arr = (pixels.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    arr_np = arr.numpy()
    if arr_np.shape[-1] == 1:
        arr_np = np.repeat(arr_np, 3, axis=-1)
    return arr_np


def _decode_latents(vae: Any, latents: torch.Tensor) -> torch.Tensor:
    """Decode ``[1, T, C, H, W]`` → ``[1, T_pix, 3, H_px, W_px]`` fp32.

    Uses the v14 / eval_chain convention: prepend one dummy latent frame
    so temporal upsampling emits a contiguous clip, then drop the first
    decoded frame so pixels line up 1:1 with input latents. VAE runs
    in fp32 with autocast disabled — the Wan VAE's convs have fp32
    biases and overflow silently in bf16."""
    device = next(vae.parameters()).device if hasattr(vae, "parameters") else latents.device
    lat = latents.to(device=device, dtype=torch.float32)
    dummy = lat[:, 0:1]
    with torch.no_grad():
        if lat.is_cuda:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                px = vae.decode_to_pixel(torch.cat([dummy, lat], dim=1))
        else:
            px = vae.decode_to_pixel(torch.cat([dummy, lat], dim=1))
    if px.dim() == 5 and px.shape[1] > 1:
        px = px[:, 1:, ...]
    return px.detach().float().cpu()


def _annotate_and_write_mp4(
    *,
    prime_latents: torch.Tensor,          # [1, prime_frames, C, H, W] fp32 CPU
    commits: List[Dict[str, Any]],
    vae: Any,
    device: torch.device,
    fps: int,
    out_path: Path,
    ride_tag: str,
    ride_basename: str,
    num_frame_per_block: int,
    embedded_step: int = -1,
) -> Dict[str, Any]:
    """Decode prime + commits to pixels, annotate each frame, write mp4.

    Returns a stats dict with timings + frame counts.
    """
    import cv2
    t_start = time.time()

    # ------------------------------------------------------------------
    # 1) Concatenate latents in display order:
    #      prime (9 frames) + transition commit (if present) + commits
    # ------------------------------------------------------------------
    pieces: List[torch.Tensor] = []
    piece_labels: List[Dict[str, Any]] = []

    npb = int(num_frame_per_block)
    # Prime chunks: we show the prime_latents as ``prime_frames // npb``
    # "prime" chunks with chunk indices 0..n-1 so the mp4's chronology
    # is clean.
    prime_frames = int(prime_latents.shape[1])
    prime_chunks = prime_frames // npb
    for c in range(prime_chunks):
        lo = c * npb
        hi = lo + npb
        pieces.append(prime_latents[:, lo:hi])
        piece_labels.append({
            "label_top": f"{ride_basename} | chunk {c} (prime)",
            "label_bot": "t=0 | GT-primed",
            "action": None,
        })

    for e in commits:
        pieces.append(e["latent"])
        if e["phase"] == "transition":
            top = f"{ride_basename} | chunk {prime_chunks} (xition)"
            bot = "transition commit | t=0"
        else:
            gfs = int(e["global_frame_start"])
            chunk_idx = gfs // npb if gfs >= 0 else -1
            top = f"{ride_basename} | chunk {chunk_idx}"
            bot = (
                f"step {e['rolling_steps_done']} | "
                f"t_in={e['slot_timestep']} | steady"
            )
        piece_labels.append({
            "label_top": top,
            "label_bot": bot,
            "action": e.get("action"),
        })

    lat_all = torch.cat(pieces, dim=1)
    log.info("decode: %d total latents (%d prime + %d commits) on %s",
             int(lat_all.shape[1]), prime_frames, len(commits), device)

    # ------------------------------------------------------------------
    # 2) Decode through VAE (fp32).
    # ------------------------------------------------------------------
    t_before_decode = time.time()
    # Chunk the decode to keep peak memory bounded on longer rollouts.
    # We decode up to ``max_decode_frames`` latent frames at a time,
    # re-apply the ``prepend-dummy-drop-first-pixel`` convention per
    # chunk, and concatenate pixel tensors. Longer rollouts (200+
    # commits = 600+ latent frames = ~1 min video @8fps) would otherwise
    # exceed the VAE's peak memory envelope on an 80 GB H100.
    max_decode_frames = 180  # ~8 GB peak at 16x60x104 latents.
    if lat_all.shape[1] <= max_decode_frames:
        pixels = _decode_latents(vae, lat_all.to(device=device))
    else:
        pieces_px: List[torch.Tensor] = []
        n = int(lat_all.shape[1])
        for lo in range(0, n, max_decode_frames):
            hi = min(lo + max_decode_frames, n)
            chunk = lat_all[:, lo:hi].to(device=device)
            pieces_px.append(_decode_latents(vae, chunk))
        pixels = torch.cat(pieces_px, dim=1)
    t_after_decode = time.time()

    arr = _pixels_to_uint8_hwc(pixels)
    t_after_cast = time.time()

    # ------------------------------------------------------------------
    # 3) Annotate (cv2 — v14 style, see ``eval_chain.annotate_video``).
    # ------------------------------------------------------------------
    n_pixel_frames = int(arr.shape[0])
    # Distribute labels across pixel frames proportionally to latent span.
    # Each ``pieces[i]`` has ``lat_pieces_frames[i]`` latent frames; after
    # VAE upsampling the same proportion maps to pixel frames.
    lat_pieces_frames = [int(p.shape[1]) for p in pieces]
    total_lat = sum(lat_pieces_frames)
    # Compute pixel-frame boundaries so rounding errors don't accumulate.
    bounds: List[int] = [0]
    acc = 0
    for lf in lat_pieces_frames:
        acc += lf
        bounds.append(int(round(acc / max(total_lat, 1) * n_pixel_frames)))

    out = np.empty_like(arr)
    for pi in range(len(pieces)):
        fa = bounds[pi]
        fb = bounds[pi + 1]
        if fb <= fa:
            continue
        lbl = piece_labels[pi]
        for fi in range(fa, fb):
            f = arr[fi].copy()
            _draw_header(f, lbl["label_top"], lbl["label_bot"])
            act = lbl.get("action")
            if act is not None:
                _draw_action_panel(f, act)
            # Bottom-right tag: run-level metadata so multi-mp4 grids
            # from the same sbatch are self-describing.
            _draw_tag(f, ride_tag, embedded_step)
            out[fi] = f
    t_after_ann = time.time()

    # ------------------------------------------------------------------
    # 4) Encode via ffmpeg.
    # ------------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_to_mp4(out, str(out_path), fps=float(fps))
    t_done = time.time()

    log.info(
        "write: %s | %d pixel frames | decode=%.2fs cast=%.2fs annotate=%.2fs "
        "encode=%.2fs total=%.2fs",
        out_path, n_pixel_frames,
        t_after_decode - t_before_decode,
        t_after_cast - t_after_decode,
        t_after_ann - t_after_cast,
        t_done - t_after_ann,
        t_done - t_start,
    )

    return {
        "out_path": str(out_path),
        "n_pixel_frames": n_pixel_frames,
        "seconds_decode": t_after_decode - t_before_decode,
        "seconds_annotate": t_after_ann - t_after_cast,
        "seconds_encode": t_done - t_after_ann,
        "seconds_total": t_done - t_start,
    }


def _draw_header(f: np.ndarray, title: str, subtitle: str) -> None:
    import cv2
    h, w = f.shape[:2]
    hdr_h = 42
    hdr_w = min(w, 360)
    f[:hdr_h, :hdr_w] = (
        f[:hdr_h, :hdr_w].astype(np.float32) * 0.22
    ).astype(np.uint8)
    cv2.putText(
        f, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
        (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        f, subtitle, (6, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
        (200, 220, 255), 1, cv2.LINE_AA,
    )


def _draw_action_panel(f: np.ndarray, action: torch.Tensor) -> None:
    import cv2
    vals = [float(v) for v in action.detach().float().cpu().numpy().tolist()]
    n_dims = len(vals)
    if n_dims == 0:
        return
    h, w = f.shape[:2]
    pw = 180
    px0 = max(0, w - pw)
    panel_h = min(h - 4, 18 + 22 * n_dims)
    f[:panel_h, px0:] = (
        f[:panel_h, px0:].astype(np.float32) * 0.22
    ).astype(np.uint8)
    cv2.putText(
        f, "action", (px0 + 6, 14),
        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 120), 1, cv2.LINE_AA,
    )
    labels = ["z2", "z7"] + [f"d{i}" for i in range(2, n_dims)]
    clip = 1.0
    cx = px0 + 100
    for i, v in enumerate(vals):
        y = 34 + 22 * i
        if y + 10 > h:
            break
        bl = int(abs(max(-clip, min(clip, v))) / clip * 60)
        cv2.line(f, (cx, y - 8), (cx, y + 8), (120, 120, 120), 1)
        col = (80, 220, 80) if v >= 0 else (80, 80, 220)
        if v >= 0:
            cv2.rectangle(f, (cx, y - 6), (cx + bl, y + 6), col, -1)
        else:
            cv2.rectangle(f, (cx - bl, y - 6), (cx, y + 6), col, -1)
        cv2.putText(
            f, f"{labels[i]} {v:+.2f}", (px0 + 6, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.34, col, 1, cv2.LINE_AA,
        )


def _draw_tag(f: np.ndarray, tag: str, embedded_step: int) -> None:
    import cv2
    h, w = f.shape[:2]
    txt = f"{tag} | step={embedded_step}" if embedded_step >= 0 else tag
    # Tiny tag in the bottom-left so you can tell runs apart at a glance.
    tag_h = 20
    tag_w = min(w, 260)
    f[h - tag_h:, :tag_w] = (
        f[h - tag_h:, :tag_w].astype(np.float32) * 0.28
    ).astype(np.uint8)
    cv2.putText(
        f, txt, (6, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
        (255, 240, 200), 1, cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Ride loader (delegates to eval_causal_AR.load_per_rank_ride_ar)
# ---------------------------------------------------------------------------


def load_ride_for_rollout(
    *,
    zarr_basename: str,
    latent_start_offset: int,
    total_frames: Optional[int],
    manifest_path: Optional[str],
    encoded_root: str,
    caption_root: str,
    motion_root: str,
    ss_vae_checkpoint: str,
    action_dims: List[int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Load ``(gt_latents, gt_actions, prompt_embeds, meta)`` for rollout.

    Uses ``eval_causal_AR.load_per_rank_ride_ar`` to handle manifest vs
    disk-fallback; if ``total_frames`` is None we auto-probe the
    ride's latent count via the disk path and use ``n_lat -
    latent_start_offset`` (i.e. roll to natural end, as Phase-1 training
    does).
    """
    # Phase-1 rolls to natural end-of-ride. To respect that, we need the
    # caller to tell us the ride length, or we probe it. Easiest is to
    # call ``load_per_rank_ride_ar`` with a large-but-safe upper bound
    # the first time to get the metadata, then re-call with the actual
    # cap if needed. In practice we know the ride length from the
    # manifest / disk attrs; do a quick probe:
    import zarr as zarr_lib
    from utils.eval_chain import _count_latent_frames, _load_ride_entry_from_disk, _build_ts_to_ride_dir
    from utils.eval_causal_chain import _find_ride_in_manifest

    zpath: Optional[str] = None
    n_lat: Optional[int] = None
    if manifest_path and Path(manifest_path).exists():
        try:
            manifest = torch.load(
                manifest_path, map_location="cpu", weights_only=False,
            )
            if isinstance(manifest, dict) and "rides" in manifest:
                rides = manifest["rides"]
            else:
                rides = manifest
            hit = _find_ride_in_manifest(rides, zarr_basename)
            if hit is not None:
                _, _, zpath, n_lat = hit
            del manifest
        except Exception as e:  # noqa: BLE001
            log.warning("manifest probe failed: %s", e)
    if zpath is None or n_lat is None:
        # Disk fallback to get the actual ride length.
        ts_map = _build_ts_to_ride_dir(Path(caption_root))
        ride_dict = _load_ride_entry_from_disk(
            zarr_basename, Path(encoded_root), Path(caption_root), ts_map,
        )
        zpath = str(ride_dict["zarr_path"])
        n_lat = int(ride_dict["n_latent_frames"])

    if total_frames is None:
        total_frames = max(0, int(n_lat) - int(latent_start_offset))
    else:
        total_frames = min(int(total_frames), int(n_lat) - int(latent_start_offset))
    if total_frames < 30:
        raise SystemExit(
            f"Ride {zarr_basename}: only {total_frames} latents available "
            f"after offset {latent_start_offset} (need >= 30 for warmup)."
        )

    log.info(
        "ride %s: n_lat=%d, start=%d, rolling for %d latent frames",
        zarr_basename, n_lat, latent_start_offset, total_frames,
    )

    initial_latents, prompt_embeds, noisy_fa_full, meta = load_per_rank_ride_ar(
        zarr_basename=zarr_basename,
        latent_start_offset=latent_start_offset,
        total_frames=total_frames,
        manifest_path=manifest_path,
        encoded_root=encoded_root,
        caption_root=caption_root,
        motion_root=motion_root,
        ss_vae_checkpoint=ss_vae_checkpoint,
        action_dims=action_dims,
        device=device,
    )
    # ``initial_latents`` is the whole slice [1, total_frames, C, H, W]
    # — exactly the ``gt_latents`` the pipeline needs.
    return initial_latents, prompt_embeds, noisy_fa_full, meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config", type=str,
        default="configs/longlive_phase1_rolling_staircase.yaml",
        help="Phase-1 config; pipeline topology (ladder / decay / kv / "
             "attn) is read from here.",
    )
    p.add_argument(
        "--ode_config", type=str,
        default="configs/action_ode_distill.yaml",
        help="ODE-distill config for ODERegression's shell (teacher "
             "merge, action head dims, scheduler). Not used for "
             "pipeline topology — that comes from --config.",
    )
    p.add_argument(
        "--student_ckpt", type=str, required=True,
        help="Path to the ODE-distilled student snapshot (default "
             "Phase-1 init: "
             "logs/action_ode_distill_E/action_ode_step0001000.pt).",
    )
    p.add_argument(
        "--output_dir", type=str, required=True,
        help="Where to write the per-rank mp4 and stats.json.",
    )
    p.add_argument(
        "--ride_tag", type=str, required=True,
        help="Human-readable tag drawn into the bottom-left of every "
             "frame (e.g. 'madrid_held', 'brighton_turny').",
    )
    p.add_argument(
        "--rank_zarr", type=str, required=True,
        help="Basename of the zarr file to roll out on this rank "
             "(e.g. '20240216101235.zarr').",
    )
    p.add_argument(
        "--rank_offset", type=int, default=0,
        help="Latent frame index to start the rollout at (default 0 "
             "= very first frame of the ride).",
    )
    p.add_argument(
        "--max_ride_frames", type=int, default=None,
        help="Optional cap on the number of latent frames to roll out "
             "(per-rank, after --rank_offset). Default = natural end "
             "of ride, like Phase-1 training.",
    )
    p.add_argument(
        "--max_rolling_steps", type=int, default=None,
        help="Optional cap on the number of rolling steps to run after "
             "warmup. Default = no cap.",
    )
    p.add_argument(
        "--manifest", type=str, default=None,
        help="Optional path to a ride manifest (.ride_manifest.pt). "
             "Used first; disk fallback otherwise.",
    )
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
    p.add_argument(
        "--fps", type=int, default=8,
        help="Output mp4 fps.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Seed for the pipeline's noise samples inside rolling_step.",
    )
    p.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

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
        "Device=%s | dtype=%s | config=%s | ode_config=%s | student=%s",
        device, dtype, args.config, args.ode_config, args.student_ckpt,
    )

    # Seeds. We only set CPU + CUDA seeds; pipeline RNG will pull from
    # this pool inside its add_noise / randn calls.
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    # ------------------------------------------------------------------
    # Build student + pipeline + VAE.
    # ------------------------------------------------------------------
    ode_model, pipeline, vae = build_student_and_pipeline(
        phase1_cfg_path=args.config,
        ode_cfg_path=args.ode_config,
        student_ckpt_path=args.student_ckpt,
        device=device,
        dtype=dtype,
    )

    # Freeze everything; this script is pure inference.
    for m in (ode_model,):
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()

    # ------------------------------------------------------------------
    # Load a ride.
    # ------------------------------------------------------------------
    phase1_cfg = _load_omega(args.config)
    action_dims = list(getattr(phase1_cfg, "action_dims", [2, 7]))
    num_frame_per_block = int(getattr(phase1_cfg, "num_frame_per_block", 3))
    prime_kv_frames = int(getattr(phase1_cfg, "prime_kv_frames", 9))

    gt_latents, prompt_embeds, gt_actions, meta = load_ride_for_rollout(
        zarr_basename=args.rank_zarr,
        latent_start_offset=int(args.rank_offset),
        total_frames=args.max_ride_frames,
        manifest_path=args.manifest,
        encoded_root=args.encoded_root,
        caption_root=args.caption_root,
        motion_root=args.motion_root,
        ss_vae_checkpoint=args.ss_vae_checkpoint,
        action_dims=action_dims,
        device=device,
    )
    # Cast to pipeline dtype.
    gt_latents = gt_latents.to(dtype=dtype)
    gt_actions = gt_actions.to(dtype=dtype)
    prompt_embeds = prompt_embeds.to(dtype=dtype)

    log.info(
        "ride %s: latents=%s actions=%s prompt=%s",
        args.rank_zarr, tuple(gt_latents.shape),
        tuple(gt_actions.shape), tuple(prompt_embeds.shape),
    )

    # ------------------------------------------------------------------
    # Roll out.
    # ------------------------------------------------------------------
    capture = run_rollout_and_capture(
        pipeline=pipeline,
        gt_latents=gt_latents,
        gt_actions=gt_actions,
        prompt_embeds=prompt_embeds,
        num_frame_per_block=num_frame_per_block,
        prime_kv_frames=prime_kv_frames,
        max_rolling_steps=(
            int(args.max_rolling_steps)
            if args.max_rolling_steps is not None else None
        ),
    )
    if not capture["commits"]:
        log.error("No rolling-step records captured — ride may be too short.")
        return

    # ------------------------------------------------------------------
    # Write the mp4.
    # ------------------------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / f"{Path(args.rank_zarr).stem}_{args.ride_tag}.mp4"

    # Recover the embedded step from the ckpt name if possible (for the
    # corner tag — cheap cosmetic).
    embedded_step = -1
    m = Path(args.student_ckpt).name
    import re
    mm = re.search(r"step(\d+)", m)
    if mm:
        try:
            embedded_step = int(mm.group(1))
        except ValueError:
            embedded_step = -1

    write_stats = _annotate_and_write_mp4(
        prime_latents=capture["prime_latents"],
        commits=capture["commits"],
        vae=vae,
        device=device,
        fps=int(args.fps),
        out_path=out_mp4,
        ride_tag=args.ride_tag,
        ride_basename=Path(args.rank_zarr).stem,
        num_frame_per_block=num_frame_per_block,
        embedded_step=embedded_step,
    )

    # Stats JSON — handy for comparing runs / plotting.
    stats = {
        "student_ckpt": args.student_ckpt,
        "embedded_step": embedded_step,
        "rank_zarr": args.rank_zarr,
        "rank_offset": int(args.rank_offset),
        "ride_tag": args.ride_tag,
        "n_commits": capture["stats"]["n_commits"],
        "wall_seconds_rollout": capture["stats"]["wall_seconds"],
        "rolling_steps_per_sec": capture["stats"]["rolling_steps_per_sec"],
        "out_path": write_stats["out_path"],
        "n_pixel_frames": write_stats["n_pixel_frames"],
        "seconds_decode": write_stats["seconds_decode"],
        "seconds_annotate": write_stats["seconds_annotate"],
        "seconds_encode": write_stats["seconds_encode"],
        "ride_meta": meta,
    }
    stats_path = out_dir / f"{Path(args.rank_zarr).stem}_{args.ride_tag}_stats.json"
    with stats_path.open("w") as fh:
        json.dump(stats, fh, indent=2, default=str)
    log.info("Wrote %s (mp4 + stats).", out_mp4)


if __name__ == "__main__":
    main()
