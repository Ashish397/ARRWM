#!/usr/bin/env python3
"""Tests for the eval pipeline: dtype correctness and optimizer offload/restore.

Covers the recurring issues in ``_maybe_eval`` → ``_generate_eval`` →
``_decode_latents`` → ``_compute_teacher_visuals`` → ``_annotate_action_video``.

Tests:
  1. ``bf16_to_numpy``         — Verifies the .detach().float().cpu().numpy()
     pattern that is required for bfloat16 tensors.
  2. ``chunk_actions_bf16``    — Verifies mean-pool chunking on bf16 tensors.
  3. ``eval_metrics_dtypes``   — Verifies F.mse_loss().item() with mixed dtypes.
  4. ``annotation_dtypes``     — Verifies _annotate_action_video handles the
     exact dtype mix from the eval pipeline (bf16 targets, float32
     teacher/critic, float16 motion, requires_grad critic outputs).
  5. ``optimizer_offload``     — Verifies optimizer-state CPU offload/restore
     cycle.  Requires GPU.

Usage:
    python testing/test_eval_pipeline.py                    # all tests
    python testing/test_eval_pipeline.py --mode cpu          # cpu-only tests
    python testing/test_eval_pipeline.py --mode gpu          # gpu-only tests
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


N_CHUNKS = 7
Z_DIM = 2
N_GRID = 100          # 10x10
N_PIXEL_FRAMES = 84   # n_chunks * output_chunk_size (7 * 12)
FRAME_H = 200
FRAME_W = 300


# ===================================================================
# Inlined functions under test (avoids importing the full trainer
# module which requires CUDA at import time via utils/memory.py)
# ===================================================================

def _chunk_actions(per_frame: torch.Tensor, chunk_frames: int) -> torch.Tensor:
    """Mean-pool ``[B, F, D]`` into ``[B, n_chunks, D]`` with exact chunk size."""
    B, F_len, D = per_frame.shape
    n_chunks = F_len // chunk_frames
    trimmed = per_frame[:, :n_chunks * chunk_frames]
    return trimmed.reshape(B, n_chunks, chunk_frames, D).mean(dim=2)


def _draw_triplet_action_overlay(
    frame: np.ndarray,
    teacher_z: np.ndarray,
    critic_z: np.ndarray,
    target_z: np.ndarray,
    teacher_reward: float,
    critic_reward: float,
    title: str,
    frame_idx: int,
    seg_idx: int,
    clip: float = 1.0,
) -> np.ndarray:
    h, w = frame.shape[:2]
    out = frame.copy()
    panel_w = 220
    panel_x0 = w - panel_w
    out[:, panel_x0:] = (out[:, panel_x0:].astype(np.float32) * 0.22).astype(np.uint8)

    def draw_bar(y_mid: int, value: float, color: tuple, label: str):
        cx = panel_x0 + 120
        half = 80
        bar_h = 7
        v = float(max(-clip, min(clip, value)))
        bar_len = int(abs(v) / clip * half)
        cv2.line(out, (cx, y_mid - 10), (cx, y_mid + 10), (120, 120, 120), 1)
        if v >= 0:
            cv2.rectangle(out, (cx, y_mid - bar_h), (cx + bar_len, y_mid + bar_h), color, -1)
        else:
            cv2.rectangle(out, (cx - bar_len, y_mid - bar_h), (cx, y_mid + bar_h), color, -1)
        cv2.putText(
            out, f"{label} {value:+.3f}", (panel_x0 + 8, y_mid + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA,
        )

    cv2.putText(out, title, (panel_x0 + 8, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"frame={frame_idx} seg={seg_idx}", (panel_x0 + 8, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(out, "teacher(g) critic(o) target(b)", (panel_x0 + 8, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (180, 180, 180), 1, cv2.LINE_AA)

    rows = [
        ("z2", float(teacher_z[0]), float(critic_z[0]), float(target_z[0]), 82),
        ("z7", float(teacher_z[1]), float(critic_z[1]), float(target_z[1]), 140),
    ]
    for z_name, teacher_val, critic_val, target_val, y0 in rows:
        cv2.putText(out, z_name, (panel_x0 + 8, y0 - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 120), 1, cv2.LINE_AA)
        draw_bar(y0, teacher_val, (80, 220, 80), "t")
        draw_bar(y0 + 16, critic_val, (0, 170, 255), "c")
        draw_bar(y0 + 32, target_val, (80, 80, 220), "y")

    cv2.putText(out, f"teacher_r {teacher_reward:+.4f}", (panel_x0 + 8, h - 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (100, 255, 100), 1, cv2.LINE_AA)
    cv2.putText(out, f"critic_r  {critic_reward:+.4f}", (panel_x0 + 8, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (100, 200, 255), 1, cv2.LINE_AA)
    return out


def _annotate_action_video(
    video_np: np.ndarray,
    est_motion: torch.Tensor,
    teacher_z: torch.Tensor,
    teacher_reward: torch.Tensor,
    critic_z: torch.Tensor,
    critic_reward: torch.Tensor,
    target_action_z: torch.Tensor,
    title: str,
    grid_size: int = 10,
    output_chunk_size: int = 12,
) -> np.ndarray:
    """Mirrors the trainer's _annotate_action_video exactly."""
    teacher_np = teacher_z[0].detach().cpu().numpy()
    teacher_reward_np = teacher_reward[0].squeeze(-1).detach().cpu().numpy()
    critic_np = critic_z[0].detach().cpu().numpy()
    critic_reward_np = critic_reward[0].squeeze(-1).detach().cpu().numpy()
    n_seg = teacher_z.shape[1]
    target_seg_np = target_action_z[0, :n_seg].detach().float().cpu().numpy()
    motion_np = est_motion[0].detach().cpu().numpy()

    n_frames = video_np.shape[0]
    annotated = []
    panel_w = 220
    for frame_idx in range(n_frames):
        f = video_np[frame_idx].copy()
        seg_idx = min(frame_idx // output_chunk_size, teacher_np.shape[0] - 1)
        h, w = f.shape[:2]

        if seg_idx < motion_np.shape[0]:
            mvecs = motion_np[seg_idx]
            drawable_w = max(1, w - panel_w)
            for gy in range(grid_size):
                for gx in range(grid_size):
                    idx = gy * grid_size + gx
                    dx, dy, vis_val = mvecs[idx]
                    if vis_val < 0.2:
                        continue
                    cx_pt = int((gx + 0.5) * drawable_w / grid_size)
                    cy_pt = int((gy + 0.5) * h / grid_size)
                    ex_pt = int(cx_pt - dx * 3)
                    ey_pt = int(cy_pt - dy * 3)
                    color = (0, 255, 0) if vis_val >= 0.5 else (0, 200, 255)
                    cv2.arrowedLine(f, (cx_pt, cy_pt), (ex_pt, ey_pt), color, 1, tipLength=0.3)

        f = _draw_triplet_action_overlay(
            f, teacher_np[seg_idx], critic_np[seg_idx], target_seg_np[seg_idx],
            float(teacher_reward_np[seg_idx]), float(critic_reward_np[seg_idx]),
            title=title, frame_idx=frame_idx, seg_idx=seg_idx,
        )
        annotated.append(f)
    return np.stack(annotated)


# ===================================================================
# Tensor factories
# ===================================================================

def _make_eval_tensors(device="cpu", requires_grad=False):
    """Create tensors matching the exact dtypes produced by the eval pipeline."""
    B = 1
    return {
        "video_np": np.random.randint(0, 255, (N_PIXEL_FRAMES, FRAME_H, FRAME_W, 3), dtype=np.uint8),
        "motion": torch.randn(B, N_CHUNKS, N_GRID, 3, device=device, dtype=torch.float16),
        "teacher_z": torch.randn(B, N_CHUNKS, Z_DIM, device=device, dtype=torch.float32),
        "teacher_reward": torch.randn(B, N_CHUNKS, 1, device=device, dtype=torch.float32),
        "critic_z": torch.randn(B, N_CHUNKS, Z_DIM, device=device, dtype=torch.float32,
                                 requires_grad=requires_grad),
        "critic_reward": torch.randn(B, N_CHUNKS, 1, device=device, dtype=torch.float32,
                                      requires_grad=requires_grad),
        "target_chunk": torch.randn(B, N_CHUNKS, Z_DIM, device=device, dtype=torch.bfloat16),
    }


# ===================================================================
# Test 1: bf16 → numpy conversion
# ===================================================================

def test_bf16_to_numpy() -> bool:
    """Verify the .detach().float().cpu().numpy() pattern for bf16 tensors."""
    t = torch.randn(N_CHUNKS, Z_DIM, dtype=torch.bfloat16)

    try:
        _ = t.numpy()
        logging.error("BUG: bf16 .numpy() should have raised but didn't")
        return False
    except (TypeError, RuntimeError):
        pass

    try:
        arr = t.detach().float().cpu().numpy()
    except Exception as exc:
        logging.error(".detach().float().cpu().numpy() raised: %s", exc, exc_info=True)
        return False

    if arr.dtype != np.float32:
        logging.error("Expected float32, got %s", arr.dtype)
        return False

    logging.info("  bf16 → float32 → numpy: shape=%s dtype=%s", arr.shape, arr.dtype)
    return True


# ===================================================================
# Test 2: _chunk_actions on bfloat16
# ===================================================================

def test_chunk_actions_bf16() -> bool:
    """Verify _chunk_actions preserves bf16 without error."""
    per_frame = torch.randn(1, 21, 2, dtype=torch.bfloat16)

    try:
        chunked = _chunk_actions(per_frame, chunk_frames=3)
    except Exception as exc:
        logging.error("_chunk_actions raised: %s", exc, exc_info=True)
        return False

    if chunked.dtype != torch.bfloat16:
        logging.error("Expected bfloat16 output, got %s", chunked.dtype)
        return False
    if chunked.shape != (1, 7, 2):
        logging.error("Expected shape (1, 7, 2), got %s", tuple(chunked.shape))
        return False

    logging.info("  chunked: %s  dtype: %s", tuple(chunked.shape), chunked.dtype)
    return True


# ===================================================================
# Test 3: eval metric dtype handling
# ===================================================================

def test_eval_metrics_dtypes() -> bool:
    """Verify F.mse_loss + .item() works with mixed bf16/float32 inputs."""
    t = _make_eval_tensors(device="cpu")

    try:
        critic_z_mse = F.mse_loss(
            t["critic_z"].float(), t["teacher_z"].float()
        ).item()
        gen_z_mse = F.mse_loss(
            t["critic_z"].float(), t["target_chunk"].float()
        ).item()
    except Exception as exc:
        logging.error("Metric computation raised: %s", exc, exc_info=True)
        return False

    if not (isinstance(critic_z_mse, float) and isinstance(gen_z_mse, float)):
        logging.error("Expected float, got %s / %s", type(critic_z_mse), type(gen_z_mse))
        return False

    logging.info("  critic_z_mse=%.6f  gen_z_mse=%.6f", critic_z_mse, gen_z_mse)
    return True


# ===================================================================
# Test 4: _annotate_action_video dtype + grad handling
# ===================================================================

def test_annotation_dtypes() -> bool:
    """Verify _annotate_action_video handles mixed dtypes and requires_grad."""
    t = _make_eval_tensors(device="cpu", requires_grad=True)

    try:
        annotated = _annotate_action_video(
            t["video_np"],
            t["motion"],
            t["teacher_z"],
            t["teacher_reward"],
            t["critic_z"],
            t["critic_reward"],
            t["target_chunk"],
            title="dtype+grad test",
        )
    except TypeError as exc:
        if "BFloat16" in str(exc):
            logging.error("bf16→numpy not handled: %s", exc)
        elif "requires grad" in str(exc).lower():
            logging.error("requires_grad not handled: %s", exc)
        else:
            logging.error("Unexpected TypeError: %s", exc)
        return False
    except RuntimeError as exc:
        logging.error("RuntimeError: %s", exc)
        return False

    if not isinstance(annotated, np.ndarray):
        logging.error("Expected np.ndarray, got %s", type(annotated))
        return False
    if annotated.dtype != np.uint8:
        logging.error("Expected uint8, got %s", annotated.dtype)
        return False
    if annotated.shape[0] != N_PIXEL_FRAMES:
        logging.error("Expected %d frames, got %d", N_PIXEL_FRAMES, annotated.shape[0])
        return False

    logging.info("  Output: %s  dtype=%s  (bf16 targets + requires_grad critics OK)",
                 annotated.shape, annotated.dtype)
    return True


# ===================================================================
# Test 5: optimizer state offload / restore
# ===================================================================

def test_optimizer_offload(device: torch.device) -> bool:
    """Verify Adam optimizer states can be offloaded to CPU and restored."""
    model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 16)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randn(2, 64, device=device)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    def _gpu_states(opt):
        """Yield (key, tensor) for optimizer state tensors that live on GPU."""
        for state in opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and v.is_cuda:
                    yield state, k, v

    n_gpu = sum(1 for _ in _gpu_states(optimizer))
    if n_gpu == 0:
        logging.error("Pre-offload: no optimizer states on GPU")
        return False
    logging.info("  %d optimizer state tensors on GPU before offload", n_gpu)

    # Offload (mirrors _offload_training_state)
    model.zero_grad(set_to_none=True)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cpu()
    torch.cuda.empty_cache()

    n_gpu_after = sum(1 for _ in _gpu_states(optimizer))
    if n_gpu_after != 0:
        logging.error("Post-offload: %d states still on GPU", n_gpu_after)
        return False
    logging.info("  Offloaded %d state groups to CPU", len(optimizer.state))

    # Restore (mirrors _restore_training_state)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    n_gpu_restored = sum(1 for _ in _gpu_states(optimizer))
    if n_gpu_restored < n_gpu:
        logging.error("Post-restore: only %d GPU states, expected at least %d", n_gpu_restored, n_gpu)
        return False
    logging.info("  Restored %d state tensors to GPU (was %d before offload)", n_gpu_restored, n_gpu)

    optimizer.zero_grad(set_to_none=True)
    x = torch.randn(2, 64, device=device)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    if not torch.isfinite(loss):
        logging.error("Loss not finite after offload/restore")
        return False

    logging.info("  Post-restore training step OK (loss=%.4f)", loss.item())
    return True


# ===================================================================
# Main
# ===================================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Eval pipeline dtype & memory tests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["cpu", "gpu", "all"], default="all")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    results = {}

    if args.mode in ("cpu", "all"):
        logging.info("=" * 60)
        logging.info("CPU TESTS")
        logging.info("=" * 60)

        for name, fn in [
            ("bf16_to_numpy", test_bf16_to_numpy),
            ("chunk_actions_bf16", test_chunk_actions_bf16),
            ("eval_metrics_dtypes", test_eval_metrics_dtypes),
            ("annotation_dtypes", test_annotation_dtypes),
        ]:
            logging.info("[TEST] %s", name)
            results[name] = fn()

    if args.mode in ("gpu", "all"):
        if not torch.cuda.is_available():
            logging.warning("No GPU available, skipping GPU tests.")
        else:
            logging.info("=" * 60)
            logging.info("GPU TESTS")
            logging.info("=" * 60)
            device = torch.device(args.device)

            logging.info("[TEST] optimizer_offload")
            results["optimizer_offload"] = test_optimizer_offload(device)

    logging.info("=" * 60)
    logging.info("RESULTS")
    logging.info("=" * 60)
    passed = 0
    failed = 0
    for name, ok in results.items():
        tag = "PASS" if ok else "FAIL"
        logging.info("  [%s] %s", tag, name)
        if ok:
            passed += 1
        else:
            failed += 1

    logging.info("=" * 60)
    logging.info("%d passed, %d failed, %d total", passed, failed, passed + failed)
    logging.info("=" * 60)

    if failed > 0:
        sys.exit(1)
    logging.info("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
