"""Rank-0 slot-0 visualization for the multi-slot rolling-staircase trainer.

Maintains a FIFO buffer of the latest committed slot-0 ``pred_x0`` latents
from ride-slot 0 on rank 0, then every ``wallclock_seconds`` decodes them,
annotates each frame (ride label + chunk index + commanded action bars)
with OpenCV, encodes the result via an ``ffmpeg`` subprocess, and uploads
as a ``wandb.Video``.

Alignment with v14 / ODE-distill eval style. The earlier implementation
rolled its own PIL text overlay + imageio mp4 writer, which (a) produced
noticeably uglier frames than the ``eval_chain``/``eval_causal_AR``
pipeline and (b) held the GIL for 8–9 minutes on occasional flushes,
tripping NCCL's 10-minute watchdog on DDP all-reduces while rank 0 was
busy. This module now reuses
``utils.eval_chain.frames_to_mp4`` (ffmpeg), cv2-based annotation, and
logs the wall-clock timing of VAE decode + encode + wandb upload on every
flush so a long flush is visible in the logs rather than silently
manifesting as an NCCL crash.

Preserves the original API (``observe`` / ``maybe_flush`` / ``reset_timer``
/ ``buffer_len``) so the trainer call sites don't need to change.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


class MultislotVisRecorder:
    """Holds rank-0's vis buffer and knows how to flush it.

    Trainer-side usage is unchanged:

        rec = MultislotVisRecorder(
            enabled=(self.is_main_process),
            out_dir=self.log_dir / "vis",
            wallclock_seconds=1800,
            num_chunks=10,
            fps=8,
            num_frame_per_block=3,
        )
        rec.observe(records, slot_state)
        rec.maybe_flush(vae=..., device=..., dtype=..., step=...)

    When ``enabled=False`` (non-main rank) all methods are no-ops.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        out_dir: Path,
        wallclock_seconds: float,
        num_chunks: int,
        fps: int,
        num_frame_per_block: int,
        log_to_wandb: bool = True,
        save_local: bool = False,
        emit_on_first_step: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.out_dir = Path(out_dir)
        self.wallclock_seconds = float(wallclock_seconds)
        self.num_chunks = int(num_chunks)
        self.fps = int(fps)
        self.num_frame_per_block = int(num_frame_per_block)
        self.log_to_wandb = bool(log_to_wandb)
        self.save_local = bool(save_local)
        self.emit_on_first_step = bool(emit_on_first_step)

        self._buffer: List[Dict[str, Any]] = []
        self._init_flush_time()

        self._wandb = None
        if self.enabled and self.log_to_wandb:
            try:
                import wandb  # type: ignore
                self._wandb = wandb
            except Exception as e:  # noqa: BLE001
                logging.warning(
                    "vis: wandb import failed, disabling wandb logging: %s", e,
                )
                self.log_to_wandb = False

        if self.enabled and self.save_local:
            self.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def observe(self, records: List[Any], slot_state: Any) -> None:
        """Append the slot-0 record from this rolling step to the buffer.

        ``records`` is a list of ``RollingStepOutput`` produced by
        ``PerSlotRideBatcher.step_slot``. We only keep the ``slot_idx==0``
        entry (the cleanest slot, whose ``pred_x0`` is about to be
        committed).
        """
        if not self.enabled or not records:
            return
        slot0 = None
        for rec in records:
            if int(getattr(rec, "slot_idx", -1)) == 0:
                slot0 = rec
                break
        if slot0 is None:
            return

        pred = slot0.pred_x0
        if not torch.is_tensor(pred):
            return

        global_frame_start = int(getattr(slot0, "global_frame_start", -1))
        chunk_index = (
            global_frame_start // self.num_frame_per_block
            if global_frame_start >= 0
            else -1
        )
        n_latent_frames = int(getattr(slot_state, "n_latent_frames", 0) or 0)
        total_chunks = (
            n_latent_frames // self.num_frame_per_block
            if n_latent_frames > 0
            else 0
        )
        ride_id = str(getattr(slot_state, "ride_id", "") or "")

        # Commanded action for this committed chunk — used for the bar
        # overlay, same convention as ``utils.eval_chain.annotate_video``.
        # ``action_frame`` is [B, npb, action_dim] with decay already
        # applied (slot-0 decay = 1.0, so this equals the raw commanded
        # action). Mean-pool over the 3 frames to get one value per
        # latent block — matches v14's chunk-level action display.
        action_frame = getattr(slot0, "action_frame", None)
        if torch.is_tensor(action_frame) and action_frame.numel() > 0:
            # [B, npb, D] -> [D] (batch 0, mean across npb frames)
            action_chunk = (
                action_frame[0].detach().float().mean(dim=0).cpu()
            )
        else:
            action_chunk = None

        slot_timestep = int(getattr(slot0, "slot_timestep", -1))
        phase = str(getattr(slot0, "phase", "?"))

        entry: Dict[str, Any] = {
            "pred_x0": pred.detach().clone().to(torch.float32).cpu(),
            "chunk_index": int(chunk_index),
            "global_frame_start": int(global_frame_start),
            "ride_id": ride_id,
            "total_chunks": int(total_chunks),
            "rolling_steps_done": int(getattr(slot_state, "rolling_steps_done", 0)),
            "slot_timestep": int(slot_timestep),
            "phase": phase,
            "action_chunk": action_chunk,  # None or [action_dim] float32 CPU.
        }
        self._buffer.append(entry)
        if len(self._buffer) > self.num_chunks:
            self._buffer = self._buffer[-self.num_chunks:]

    # ------------------------------------------------------------------
    # Flushing
    # ------------------------------------------------------------------
    def maybe_flush(
        self,
        *,
        vae: Any,
        device: torch.device,
        dtype: torch.dtype,
        step: int,
    ) -> bool:
        """Flush the buffer to an mp4 if the wall-clock interval has elapsed.

        Returns True if a flush occurred.
        """
        if not self.enabled:
            return False
        if not self._buffer:
            return False
        if (time.time() - self._last_flush_time) < self.wallclock_seconds:
            return False
        try:
            self._flush_impl(vae=vae, device=device, dtype=dtype, step=step)
        except Exception as e:  # noqa: BLE001
            logging.warning("multislot vis flush failed: %s", e, exc_info=True)
        self._last_flush_time = time.time()
        return True

    def _flush_impl(
        self,
        *,
        vae: Any,
        device: torch.device,
        dtype: torch.dtype,
        step: int,
    ) -> None:
        if vae is None:
            logging.info("vis: VAE unavailable, skipping mp4 flush")
            return

        t_start = time.time()

        # ------------------------------------------------------------------
        # 1) Concatenate latents, decode through the VAE in fp32.
        # ------------------------------------------------------------------
        latents = torch.cat([e["pred_x0"] for e in self._buffer], dim=1)
        latents_dev = latents.to(device=device, dtype=torch.float32)

        # v14 / eval_chain convention: prepend the first latent frame as
        # a "dummy" so the Wan VAE's temporal upsampling produces a
        # contiguous pixel clip, then drop the first decoded frame so
        # pixels are 1:1 with input latents. See
        # ``utils.eval_chain.ChainPipeline.decode_latents``.
        dummy = latents_dev[:, 0:1]
        with torch.no_grad():
            if latents_dev.is_cuda:
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    pixels = vae.decode_to_pixel(
                        torch.cat([dummy, latents_dev], dim=1)
                    )
            else:
                pixels = vae.decode_to_pixel(
                    torch.cat([dummy, latents_dev], dim=1)
                )
        if pixels.dim() == 5 and pixels.shape[1] > 1:
            # Drop the dummy-aligned first pixel frame.
            pixels = pixels[:, 1:, ...]
        pixels = pixels.detach().float().cpu()
        t_decoded = time.time()

        # ------------------------------------------------------------------
        # 2) Coerce to [T, H, W, 3] uint8, build per-frame labels, annotate.
        # ------------------------------------------------------------------
        arr = self._pixels_to_uint8_hwc(pixels)
        annotated = self._annotate_frames(arr)
        t_annotated = time.time()

        # ------------------------------------------------------------------
        # 3) Encode as mp4 via ffmpeg + upload to wandb / save locally.
        # ------------------------------------------------------------------
        wrote_path: Optional[Path] = None
        tmp_path: Optional[Path] = None
        try:
            if self.log_to_wandb and self._wandb is not None:
                tf = tempfile.NamedTemporaryFile(
                    suffix=".mp4", prefix=f"vis_step_{step:07d}_", delete=False,
                )
                tf.close()
                tmp_path = Path(tf.name)
                if self._write_mp4(tmp_path, annotated, self.fps):
                    wrote_path = tmp_path
                    try:
                        self._wandb.log(
                            {"vis/slot0": self._wandb.Video(
                                str(tmp_path), fps=int(self.fps), format="mp4",
                            )},
                            step=int(step),
                        )
                    except Exception as e:  # noqa: BLE001
                        logging.warning("vis: wandb.log(Video) failed: %s", e)

            if self.save_local and (wrote_path is not None or not self.log_to_wandb):
                out_path = self.out_dir / f"step_{step:07d}_slot0.mp4"
                if wrote_path is not None:
                    import shutil
                    try:
                        shutil.copyfile(str(wrote_path), str(out_path))
                    except OSError as e:
                        logging.warning(
                            "vis: copy to local %s failed (%s); re-encoding",
                            out_path, e,
                        )
                        self._write_mp4(out_path, annotated, self.fps)
                else:
                    self._write_mp4(out_path, annotated, self.fps)
        finally:
            if tmp_path is not None and tmp_path.exists():
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        t_done = time.time()
        logging.info(
            "vis: flush step=%d chunks=%d frames=%d | "
            "decode=%.2fs annotate=%.2fs encode+upload=%.2fs total=%.2fs",
            step, len(self._buffer), annotated.shape[0],
            t_decoded - t_start,
            t_annotated - t_decoded,
            t_done - t_annotated,
            t_done - t_start,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _pixels_to_uint8_hwc(self, pixels: torch.Tensor) -> np.ndarray:
        """Coerce VAE output into [T, H, W, 3] uint8 numpy in [0, 255].

        Wan VAE emits [B, T, C, H, W] in [-1, 1]. Be defensive about
        range and channel order; matches ``eval_chain.ChainPipeline.
        decode_latents``.
        """
        if pixels.dim() == 5:
            pixels = pixels[0]  # [T, C, H, W]
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

    def _annotate_frames(self, arr: np.ndarray) -> np.ndarray:
        """Draw a v14-style overlay on every frame.

        Each input chunk covers ``pix_per_chunk`` pixel frames after VAE
        upsampling; we draw the same label across those frames. The
        overlay is a two-line header in the top-left and, when the
        chunk carries a commanded action, a stack of horizontal action
        bars in the top-right — matching ``eval_chain.annotate_video``.

        Falls back to returning ``arr`` unchanged if cv2 is unavailable.
        """
        try:
            import cv2  # type: ignore
        except ImportError:
            logging.warning("vis: cv2 not available, skipping text overlay")
            return arr

        n_frames = arr.shape[0]
        if n_frames <= 0 or not self._buffer:
            return arr

        pix_per_chunk = max(1, n_frames // len(self._buffer))

        # Build per-frame metadata.
        per_frame_entry: List[Dict[str, Any]] = []
        for e in self._buffer:
            per_frame_entry.extend([e] * pix_per_chunk)
        if len(per_frame_entry) < n_frames:
            pad = per_frame_entry[-1] if per_frame_entry else self._buffer[-1]
            per_frame_entry.extend([pad] * (n_frames - len(per_frame_entry)))
        per_frame_entry = per_frame_entry[:n_frames]

        out = np.empty_like(arr)
        for i in range(n_frames):
            f = arr[i].copy()
            h, w = f.shape[:2]
            e = per_frame_entry[i]

            # ---- Header block (top-left, dimmed background). -----------
            # Title: short ride id + chunk/total. Subtitle: step + t_in.
            ride_id = e.get("ride_id", "") or "?"
            short_ride = Path(ride_id).name[:28] if ride_id != "?" else "?"
            chunk_idx = int(e["chunk_index"])
            total = int(e.get("total_chunks", 0))
            steps_done = int(e.get("rolling_steps_done", 0))
            phase = str(e.get("phase", "?"))
            slot_t = int(e.get("slot_timestep", -1))

            title = (
                f"{short_ride} | chunk {chunk_idx}/{total}"
                if total > 0 else f"{short_ride} | chunk {chunk_idx}"
            )
            subtitle = f"{phase} | step {steps_done} | t_in={slot_t}"

            hdr_h = 42
            hdr_w = min(w, 360)
            # Dim the header region for text readability (no external
            # font deps — cv2's default sans is fine at this scale).
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

            # ---- Action bars (top-right, if commanded action available).
            # Matches ``eval_chain._draw_bar`` visual convention: one
            # labelled horizontal bar per action dim, centered at 0.
            action = e.get("action_chunk", None)
            if action is not None:
                self._draw_action_panel(f, action)

            out[i] = f

        return out

    @staticmethod
    def _draw_action_panel(frame: np.ndarray, action: torch.Tensor) -> None:
        """Draw a labelled panel of horizontal action bars (top-right).

        Kept visually close to ``eval_chain.annotate_video`` so train-time
        vis and eval-time vis read the same. Works in-place on ``frame``.
        """
        try:
            import cv2  # type: ignore
        except ImportError:
            return

        vals = [float(v) for v in action.detach().float().cpu().numpy().tolist()]
        n_dims = len(vals)
        if n_dims == 0:
            return

        h, w = frame.shape[:2]
        pw = 180
        px0 = max(0, w - pw)
        panel_h = 18 + 22 * n_dims
        panel_h = min(panel_h, h - 4)

        # Dim the panel background.
        frame[:panel_h, px0:] = (
            frame[:panel_h, px0:].astype(np.float32) * 0.22
        ).astype(np.uint8)
        cv2.putText(
            frame, "action", (px0 + 6, 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 120), 1, cv2.LINE_AA,
        )

        # Same dim naming as v14: dim 0 → z2 (forward/back),
        # dim 1 → z7 (turn). Fall back to numeric labels past 2.
        labels = ["z2", "z7"] + [f"d{i}" for i in range(2, n_dims)]
        clip = 1.0
        cx = px0 + 100
        for i, v in enumerate(vals):
            y = 34 + 22 * i
            if y + 10 > h:
                break
            bl = int(abs(max(-clip, min(clip, v))) / clip * 60)
            cv2.line(frame, (cx, y - 8), (cx, y + 8), (120, 120, 120), 1)
            col = (80, 220, 80) if v >= 0 else (80, 80, 220)
            if v >= 0:
                cv2.rectangle(
                    frame, (cx, y - 6), (cx + bl, y + 6), col, -1,
                )
            else:
                cv2.rectangle(
                    frame, (cx - bl, y - 6), (cx, y + 6), col, -1,
                )
            cv2.putText(
                frame, f"{labels[i]} {v:+.2f}", (px0 + 6, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, col, 1, cv2.LINE_AA,
            )

    def _write_mp4(self, out_path: Path, arr: np.ndarray, fps: int) -> bool:
        """Encode ``arr`` as an mp4 at ``out_path`` using ffmpeg.

        Uses ``utils.eval_chain.frames_to_mp4`` (ffmpeg subprocess,
        H.264 libx264 CRF 18, yuv420p) — the same encoder used by the
        standalone eval scripts, which is consistently faster and
        produces smaller files than the ``imageio``/``torchvision.io``
        fallbacks the previous implementation used.

        Falls back to imageio → torchvision in that order if ffmpeg is
        missing; returns True if any encoder succeeded.
        """
        try:
            from utils.eval_chain import frames_to_mp4
            frames_to_mp4(arr, str(out_path), fps=float(fps))
            if out_path.exists() and out_path.stat().st_size > 0:
                return True
        except Exception as e:  # noqa: BLE001
            logging.debug("vis: ffmpeg frames_to_mp4 failed: %s", e)

        try:
            import imageio  # type: ignore
            imageio.mimwrite(str(out_path), arr, fps=int(fps), quality=7)
            return True
        except Exception as e:  # noqa: BLE001
            logging.debug("vis: imageio mp4 write failed: %s", e)

        try:
            from torchvision.io import write_video  # type: ignore
            video_tensor = torch.from_numpy(arr).contiguous()
            write_video(str(out_path), video_tensor, fps=int(fps))
            return True
        except Exception as e:  # noqa: BLE001
            logging.debug("vis: torchvision write_video failed: %s", e)

        return False

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def buffer_len(self) -> int:
        return len(self._buffer)

    def reset_timer(self) -> None:
        """Reset the wall-clock timer (optionally priming an
        immediate-first-flush window when
        ``emit_on_first_step=True``)."""
        self._init_flush_time()

    def _init_flush_time(self) -> None:
        now = time.time()
        if self.emit_on_first_step:
            self._last_flush_time = now - self.wallclock_seconds - 1.0
        else:
            self._last_flush_time = now
