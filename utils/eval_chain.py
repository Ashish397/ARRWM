#!/usr/bin/env python3
"""Independent video evaluation: 4 GPUs × 4 segments (one checkpoint per GPU).

**Noisy actions:** Fixed schedules shared on all ranks.  ``z2`` (forward/back) is
constant per segment: ``0.5``, ``-0.5``, ``0.25``, ``-1.0``.  ``z7`` (turn) runs in
seven 3-frame blocks: ``+0.25, -0.25, +0.5, -0.5, +0.75, -0.75``, then ``0`` (straight).

**Denoising noise:** Per-segment CUDA seeds (shared across GPUs) make each segment a
different rollout while keeping cross-model comparison fair for that segment.

**Training-aligned window:** ``clean_x`` is 21 consecutive ride latents; clean-side
actions come from motion (ss_vae) on ``[offset, offset+24)``.

MP4 prepends the first **3** ride latents of **that segment’s** zarr window.

Each segment uses a **different** manifest ride (distinct ``zarr_path``), scanned from
``--test_ride_idx`` in manifest order, skipping rides shorter than ``offset+24`` latents.

A frozen **v12** action critic is loaded on every rank for comparable critic bars.

Usage (via torchrun on 4 GPUs):
    torchrun --nproc_per_node=4 utils/eval_chain.py --output_dir eval/eval_chain_out

Single-GPU smoke (``WORLD_SIZE=1``), e.g. assignment slot 3::

    CUDA_VISIBLE_DEVICES=0 python utils/eval_chain.py --assignment_index 3 --num_segments 1 ...
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_FRAME_PER_BLOCK = 3
CONTEXT_FRAMES = 3
NUM_FRAMES = 21
EVAL_STEPS = 48
# If set (list of timesteps high->low, e.g. [1000, 625, 312.5, 178.6]), generate()
# uses this explicit few-step schedule instead of the shift=5.0 auto schedule
# (which at low step counts bunches all evals at high t and never reaches clean).
EVAL_DENOISING_STEP_LIST = None
VIDEO_NOISE_BASE = 1_000_003
# Training uses window_size + context_frames latents per step; need offset+24.
STREAM_LATENT_SPAN = CONTEXT_FRAMES + NUM_FRAMES
EVAL_LATENT_START_OFFSET = 100
MODEL_NAME = "Wan2.1-T2V-1.3B"
RAW_ACTION_DIM = 2
CRITIC_ACTION_DIMS = [2, 7]
CRITIC_Z_OUT = 8

# Default fallback roots for building ride entries on-the-fly (e.g. OOD zarrs
# that aren't in a training manifest). See _load_ride_entry_from_disk.
DEFAULT_CAPTION_ROOT = "/projects/u6ex/fbots/frodobots_captions/train"
DEFAULT_ENCODED_ROOT = "/projects/u6ex/fbots/frodobots_encoded"

# Same critic for all models → apples-to-apples critic readout (8-D head, bc=128).
EVAL_ACTION_CRITIC_CKPT = "logs/z_critic_v12_probe_fixes/causal_lora_step0003250.pt"
EVAL_CRITIC_BASE_CH = 128
EVAL_CRITIC_RES_BLOCKS = 4

_SHARED_V12_ACTION_CRITIC_SD: Optional[Dict[str, torch.Tensor]] = None

# ---------------------------------------------------------------------------
# Per-GPU model assignments  (rank → checkpoint info)
# ---------------------------------------------------------------------------
MODEL_ASSIGNMENTS = [
    {
        "ckpt": "logs/z_critic_v12_probe_fixes/causal_lora_step0003250.pt",
        "label": "v12_probe_fixes",
        "has_critic": True,
        "has_adaln": True,
        "has_action_tokens": True,
        "critic_base_ch": 128,
        "critic_res_blocks": 4,
    },
    {
        "ckpt": "logs/z_critic_v8_aux_critic/causal_lora_step0002900.pt",
        "label": "v8_aux_critic",
        "has_critic": True,
        "has_adaln": True,
        "has_action_tokens": True,
        "critic_base_ch": 128,
        "critic_res_blocks": 4,
    },
    {
        "ckpt": "logs/reward_critic_v5b/causal_lora_step0001400.pt",
        "label": "v5_reward_critic",
        "has_critic": True,
        "has_adaln": True,
        "has_action_tokens": True,
        "critic_base_ch": 128,
        "critic_res_blocks": 4,
    },
    {
        "ckpt": "logs/z_critic_v10_state_tokens/causal_lora_step0001650.pt",
        "label": "v10_state_tokens",
        "has_critic": True,
        "has_adaln": True,
        "has_action_tokens": True,
        "critic_base_ch": 128,
        "critic_res_blocks": 4,
    },
    {
        "ckpt": "logs/v13_balanced_weunz/causal_lora_step0004550.pt",
        "label": "v13_balanced_weunz",
        "has_critic": True,
        "has_adaln": True,
        "has_action_tokens": True,
        "critic_base_ch": 128,
        "critic_res_blocks": 4,
    },
    {
        "ckpt": "logs/v14_balanced_weunz/causal_lora_step0006600.pt",
        "label": "v14_balanced_weunz",
        "has_critic": True,
        "has_adaln": True,
        "has_action_tokens": True,
        "critic_base_ch": 128,
        "critic_res_blocks": 4,
    },
    {  # index 6: v14 with state probe ablated (loo_f3); injection unchanged
        "ckpt": "logs/v14_loo_f3/causal_lora_step0003700.pt",
        "label": "v14_loo_f3",
        "has_critic": True,
        "has_adaln": True,
        "has_action_tokens": True,
        "critic_base_ch": 128,
        "critic_res_blocks": 4,
    },
    {  # index 7: v14 with action tokens ablated (loo_tokens); AdaLN-only injection
        "ckpt": "logs/v14_loo_tokens/causal_lora_step0004000.pt",
        "label": "v14_loo_tokens",
        "has_critic": True,
        "has_adaln": True,
        "has_action_tokens": False,   # mode=adaln -> no action-token projection
        "critic_base_ch": 128,
        "critic_res_blocks": 4,
    },
    {  # index 8: v14 with AdaLN ablated (loo_adaln); action-tokens-only injection
        "ckpt": "logs/v14_loo_adaln/causal_lora_step0004000.pt",
        "label": "v14_loo_adaln",
        "has_critic": True,
        "has_adaln": False,           # mode=action_tokens -> no AdaLN projection
        "has_action_tokens": True,
        "critic_base_ch": 128,
        "critic_res_blocks": 4,
    },
    {  # index 9: v14 with critic guidance loss ablated (loo_f2); injection unchanged
        "ckpt": "logs/v14_loo_f2/causal_lora_step0004000.pt",
        "label": "v14_loo_f2",
        "has_critic": True,
        "has_adaln": True,
        "has_action_tokens": True,
        "critic_base_ch": 128,
        "critic_res_blocks": 4,
    },
    {  # index 10: critic8 — probe OFF, action critic on ALL 8 dims (6 @0.5x guid)
        "ckpt": "logs/v14_loo_critic8_noprobe/causal_lora_step0002150.pt",
        "label": "v14_loo_critic8_noprobe",
        "has_critic": True,
        "has_adaln": True,
        "has_action_tokens": True,
        "critic_base_ch": 128,
        "critic_res_blocks": 4,
    },
    {  # index 11: probe2dim — FULL v14 with state probe target reduced to z2/z7
        "ckpt": "logs/v14_loo_probe2dim/causal_lora_step0002150.pt",
        "label": "v14_loo_probe2dim",
        "has_critic": True,
        "has_adaln": True,
        "has_action_tokens": True,
        "critic_base_ch": 128,
        "critic_res_blocks": 4,
    },
    {  # index 12: v14b — critic8 mechanism (probe OFF, 8-dim critic @0.35 guid) + curated high-motion+backward weunz data
        "ckpt": "logs/v14b_critic8_curated/causal_lora_step0002000.pt",
        "label": "v14b_critic8_curated",
        "has_critic": True,
        "has_adaln": True,
        "has_action_tokens": True,
        "critic_base_ch": 128,
        "critic_res_blocks": 4,
    },
]

# Per-segment denoising noise: same across all GPUs for a given segment index.
VIDEO_NOISE_SEED_STRIDE = 7919
NUM_EVAL_SEGMENTS = 4
NUM_ACTION_CHUNKS = NUM_FRAMES // NUM_FRAME_PER_BLOCK

# Noisy-side z2 (forward / back) constant within each segment; z7 (turn) in 7×3-frame blocks.
# Turn pattern: +0.25 right, -0.25 left, +0.5 right, -0.5 left, +0.75 right, -0.75 left, then straight.
NOISY_Z2_BY_SEGMENT = np.array([0.5, -0.5, 0.25, -1.0], dtype=np.float32)
NOISY_Z7_CHUNKS = np.array(
    [0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 0.0], dtype=np.float32
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_segment_noisy_frame_actions(segment_index: int) -> np.ndarray:
    """[NUM_FRAMES, 2] with z2 fixed per segment and z7 stepping each 3-frame block."""
    if not (0 <= segment_index < len(NOISY_Z2_BY_SEGMENT)):
        raise ValueError(f"segment_index must be 0..{len(NOISY_Z2_BY_SEGMENT) - 1}, got {segment_index}")
    if NOISY_Z7_CHUNKS.shape[0] != NUM_ACTION_CHUNKS:
        raise RuntimeError(
            f"NOISY_Z7_CHUNKS length {NOISY_Z7_CHUNKS.shape[0]} != NUM_ACTION_CHUNKS {NUM_ACTION_CHUNKS}"
        )
    out = np.zeros((NUM_FRAMES, 2), dtype=np.float32)
    z2 = float(NOISY_Z2_BY_SEGMENT[segment_index])
    for c in range(NUM_ACTION_CHUNKS):
        lo = c * NUM_FRAME_PER_BLOCK
        hi = lo + NUM_FRAME_PER_BLOCK
        out[lo:hi, 0] = z2
        out[lo:hi, 1] = NOISY_Z7_CHUNKS[c]
    return out


def build_all_segment_noisy_frame_actions(
    base_seed: int, num_segments: int = NUM_EVAL_SEGMENTS,
) -> List[np.ndarray]:
    """Deterministic schedules; ``base_seed`` ignored (CLI still passes it for video noise seeds)."""
    _ = base_seed
    return [build_segment_noisy_frame_actions(i) for i in range(num_segments)]


def frame_actions_to_chunk_actions(frame_actions: torch.Tensor) -> torch.Tensor:
    """Reduce any chunk-aligned frame-action sequence to chunk actions."""
    b, f, d = frame_actions.shape
    if d != RAW_ACTION_DIM:
        raise ValueError(
            f"expected action dim {RAW_ACTION_DIM}, got {d}"
        )
    if f % NUM_FRAME_PER_BLOCK != 0:
        raise ValueError(
            f"frame count {f} is not divisible by chunk size "
            f"{NUM_FRAME_PER_BLOCK}"
        )
    num_chunks = f // NUM_FRAME_PER_BLOCK
    x = frame_actions.reshape(b, num_chunks, NUM_FRAME_PER_BLOCK, d)
    return x.mean(dim=2)


def frames_to_mp4(frames: np.ndarray, path: str, fps: float = 5.0):
    h, w = frames.shape[1], frames.shape[2]
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    _, err = proc.communicate(input=frames.tobytes(), timeout=120)
    if proc.returncode != 0:
        log.warning("ffmpeg failed for %s: %s", path, err.decode(errors="replace"))


def _tanh_squash(z: torch.Tensor) -> torch.Tensor:
    return torch.tanh(z * 0.5)


def _collect_lora_target_modules(model: nn.Module) -> list:
    targets = set()
    for mod_name, mod in model.named_modules():
        if mod.__class__.__name__ in {"WanAttentionBlock", "CausalWanAttentionBlock"}:
            for full_name, sub in mod.named_modules(prefix=mod_name):
                if isinstance(sub, nn.Linear):
                    targets.add(full_name)
    return sorted(targets)


# ---------------------------------------------------------------------------
# Annotation (teacher + critic bars)
# ---------------------------------------------------------------------------

def _draw_bar(out, px0, ym, val, col, lbl, clip=1.0):
    cx = px0 + 120
    v = float(max(-clip, min(clip, val)))
    bl = int(abs(v) / clip * 80)
    cv2.line(out, (cx, ym - 10), (cx, ym + 10), (120, 120, 120), 1)
    if v >= 0:
        cv2.rectangle(out, (cx, ym - 7), (cx + bl, ym + 7), col, -1)
    else:
        cv2.rectangle(out, (cx - bl, ym - 7), (cx, ym + 7), col, -1)
    cv2.putText(out, f"{lbl} {val:+.3f}", (px0 + 8, ym + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)


def annotate_video(video_np, teacher_z2z7, critic_z2z7, target_z2z7,
                   motion, title, grid_size=10, output_chunk_size=12):
    t_np = teacher_z2z7[0].cpu().float().numpy()
    c_np = critic_z2z7[0].cpu().float().numpy() if critic_z2z7 is not None else None
    tgt_np = target_z2z7[0].cpu().float().numpy()
    m_np = motion[0].cpu().float().numpy() if motion is not None else None
    n_seg = t_np.shape[0]
    pw = 220
    frames = []
    for fi in range(video_np.shape[0]):
        si = min(fi // output_chunk_size, n_seg - 1)
        f = video_np[fi].copy()
        h, w = f.shape[:2]
        if m_np is not None and si < m_np.shape[0]:
            dw = max(1, w - pw)
            for gy in range(grid_size):
                for gx in range(grid_size):
                    idx = gy * grid_size + gx
                    if idx >= m_np[si].shape[0]:
                        break
                    dx, dy, vis = m_np[si][idx]
                    if vis < 0.2:
                        continue
                    cx_p = int((gx + 0.5) * dw / grid_size)
                    cy_p = int((gy + 0.5) * h / grid_size)
                    col = (0, 255, 0) if vis >= 0.5 else (0, 200, 255)
                    cv2.arrowedLine(f, (cx_p, cy_p),
                                    (int(cx_p - dx * 3), int(cy_p - dy * 3)),
                                    col, 1, tipLength=0.3)
        px0 = w - pw
        f[:, px0:] = (f[:, px0:].astype(np.float32) * 0.22).astype(np.uint8)
        cv2.putText(f, title, (px0 + 8, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(f, f"frame={fi} seg={si}", (px0 + 8, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 200), 1, cv2.LINE_AA)
        has_c = c_np is not None
        legend = "teacher(g) critic(c) target(b)" if has_c else "teacher(g) target(b)"
        cv2.putText(f, legend, (px0 + 8, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180, 180, 180), 1, cv2.LINE_AA)
        for zn, tv, cv_v, gv, y0 in [
            ("z2", t_np[si, 0], c_np[si, 0] if has_c else None, tgt_np[si, 0], 78),
            ("z7", t_np[si, 1], c_np[si, 1] if has_c else None, tgt_np[si, 1], 138),
        ]:
            cv2.putText(f, zn, (px0 + 8, y0 - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 120), 1, cv2.LINE_AA)
            y = y0
            _draw_bar(f, px0, y, tv, (80, 220, 80), "t"); y += 18
            if cv_v is not None:
                _draw_bar(f, px0, y, cv_v, (100, 200, 255), "c"); y += 18
            _draw_bar(f, px0, y, gv, (80, 80, 220), "y")
        frames.append(f)
    return np.stack(frames)


# ---------------------------------------------------------------------------
# Pipeline (one per GPU)
# ---------------------------------------------------------------------------

class ChainPipeline:
    def __init__(self, device, dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.wrapper = None
        self.vae = None
        self.cotracker = None
        self.ss_vae = None
        self.ss_vae_scale = 1.0
        self.action_projection = None
        self.action_token_projection = None
        self.action_critic = None

    def build(
        self,
        config_path="configs/causal_lora_diffusion_teacher.yaml",
        *,
        use_action_tokens: bool = True,
    ):
        from omegaconf import OmegaConf
        from utils.wan_wrapper import WanDiffusionWrapper, WanVAEWrapper
        from model.action_model_patch import apply_action_patches
        from model.action_modulation import ActionModulationProjection, ActionTokenProjection

        cfg = OmegaConf.load(config_path)
        model_kwargs = dict(OmegaConf.to_container(cfg.get("model_kwargs", {})))

        log.info("[%s] Building Wan base model...", self.device)
        self.wrapper = WanDiffusionWrapper(
            model_name=cfg.get("model_name", MODEL_NAME),
            is_causal=True, **model_kwargs,
        )
        self.wrapper.model.num_frame_per_block = NUM_FRAME_PER_BLOCK
        self.wrapper.model.context_shift = CONTEXT_FRAMES // NUM_FRAME_PER_BLOCK
        if cfg.get("gradient_checkpointing", True):
            self.wrapper.model.enable_gradient_checkpointing()

        apply_action_patches(self.wrapper)

        model_dim = getattr(self.wrapper.model, "dim", 2048)
        act = cfg.get("action_activation", "silu")
        self.action_projection = ActionModulationProjection(
            action_dim=RAW_ACTION_DIM, activation=act, hidden_dim=model_dim, zero_init=True)
        self.action_token_projection = ActionTokenProjection(
            action_dim=RAW_ACTION_DIM, activation=act, hidden_dim=model_dim, zero_init=True)
        # v2 (and similar) checkpoints have no in-sequence action tokens; RoPE must match seq length.
        if use_action_tokens:
            self.wrapper.model.action_tokens_per_frame = 1
            self.wrapper.adjust_seq_len_for_action_tokens(num_frames=NUM_FRAMES, action_per_frame=1)
        else:
            self.wrapper.model.action_tokens_per_frame = 0
            log.info("[%s] Action tokens disabled (seq_len=%s)", self.device, self.wrapper.seq_len)

        import peft
        from peft import LoraConfig
        adapter_cfg = OmegaConf.to_container(cfg.get("adapter", {}))
        rank = int(adapter_cfg.get("rank", 256))
        alpha = float(adapter_cfg.get("alpha", rank))
        targets = _collect_lora_target_modules(self.wrapper.model) or ["q", "k", "v", "o"]
        self.wrapper.model = peft.get_peft_model(self.wrapper.model, LoraConfig(
            r=rank, lora_alpha=alpha, lora_dropout=0.0,
            target_modules=targets, bias="none",
        ))
        inner = self.wrapper.model.get_base_model() if hasattr(
            self.wrapper.model, "get_base_model") else self.wrapper.model
        inner.action_tokens_per_frame = 1 if use_action_tokens else 0

        self.wrapper.to(self.device).eval()
        self.action_projection.to(self.device).eval()
        self.action_token_projection.to(self.device).eval()

        log.info("[%s] Loading frozen VAE, CoTracker, ss_vae...", self.device)
        self.vae = WanVAEWrapper(); self.vae.to(self.device).eval()
        self.cotracker = torch.hub.load(
            "facebookresearch/co-tracker", "cotracker2", skip_validation=True,
        ).to(self.device).eval()
        ss_ckpt = cfg.get("ss_vae_checkpoint", "action_query/checkpoints/ss_vae_8free.pt")
        from action_query.ss_vae_model import load_ss_vae
        self.ss_vae, self.ss_vae_scale = load_ss_vae(ss_ckpt, device=str(self.device))
        log.info("[%s] Pipeline ready.", self.device)

    def load_checkpoint(self, ckpt_path, vinfo):
        from peft import set_peft_model_state_dict, get_peft_model_state_dict
        log.info("[%s] Loading %s", self.device, ckpt_path)
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ck = {k: v for k, v in raw.items() if k not in ("optimizer", "critic_optimizer")}
        del raw

        try:
            set_peft_model_state_dict(self.wrapper.model, ck["lora"])
        except Exception:
            cur = get_peft_model_state_dict(self.wrapper.model)
            n = sum(1 for k in cur if k in ck["lora"] and cur[k].shape == ck["lora"][k].shape)
            for k in cur:
                if k in ck["lora"] and cur[k].shape == ck["lora"][k].shape:
                    cur[k] = ck["lora"][k]
            set_peft_model_state_dict(self.wrapper.model, cur)
            log.info("[%s]   Cross-loaded %d LoRA keys", self.device, n)

        if "action_projection" in ck and vinfo.get("has_adaln"):
            self.action_projection.load_state_dict(ck["action_projection"])
        if "action_token_projection" in ck and vinfo.get("has_action_tokens"):
            self.action_token_projection.load_state_dict(ck["action_token_projection"])

        self.action_critic = None
        self._attach_shared_v12_eval_critic()

        self.wrapper.eval()
        return ck.get("step", "?")

    def _attach_shared_v12_eval_critic(self) -> None:
        """Load v12 critic weights once per process (CPU cache), build module on this GPU."""
        global _SHARED_V12_ACTION_CRITIC_SD
        if _SHARED_V12_ACTION_CRITIC_SD is None:
            if not os.path.exists(EVAL_ACTION_CRITIC_CKPT):
                # z_critic_v12 was removed in the 2026-08-02 checkpoint
                # decimation; the shared critic readout is optional
                # (flow probes never consume it) — disable gracefully.
                log.warning("[%s] Shared eval critic ckpt missing (%s); "
                            "critic bars disabled.", self.device,
                            EVAL_ACTION_CRITIC_CKPT)
                return
            log.info("[%s] Loading shared eval critic from %s", self.device, EVAL_ACTION_CRITIC_CKPT)
            blob = torch.load(
                EVAL_ACTION_CRITIC_CKPT, map_location="cpu", weights_only=False,
            )
            if "action_critic" not in blob:
                log.warning("[%s] No action_critic in %s; critic bars disabled.", self.device, EVAL_ACTION_CRITIC_CKPT)
                return
            _SHARED_V12_ACTION_CRITIC_SD = blob["action_critic"]
            del blob
        from model.action_critic import ActionCritic
        c = ActionCritic(
            latent_channels=16,
            chunk_frames=NUM_FRAME_PER_BLOCK,
            action_dim=len(CRITIC_ACTION_DIMS),
            z_out_dim=CRITIC_Z_OUT,
            base_channels=EVAL_CRITIC_BASE_CH,
            num_res_blocks=EVAL_CRITIC_RES_BLOCKS,
        )
        inc = c.load_state_dict(_SHARED_V12_ACTION_CRITIC_SD, strict=False)
        if inc.missing_keys or inc.unexpected_keys:
            log.info("[%s]   eval critic load missing=%s unexpected=%s", self.device, inc.missing_keys, inc.unexpected_keys)
        c.to(self.device).eval()
        self.action_critic = c
        log.info("[%s] Shared v12 eval critic attached.", self.device)

    def build_conditional(self, prompt_embeds, z_noisy_frame, z_clean_frame, use_adaln, use_tokens):
        cond = {"prompt_embeds": prompt_embeds}
        if use_adaln:
            cond["_action_modulation"] = self.action_projection(z_noisy_frame, num_frames=NUM_FRAMES)
            cond["_action_modulation_clean"] = self.action_projection(z_clean_frame, num_frames=NUM_FRAMES)
        if use_tokens:
            cond["_action_tokens"] = self.action_token_projection(z_noisy_frame)
            cond["_action_tokens_clean"] = self.action_token_projection(z_clean_frame)
        return cond

    @torch.no_grad()
    def generate(self, conditional, clean_x):
        from utils.scheduler import FlowMatchScheduler
        sched = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
        if EVAL_DENOISING_STEP_LIST is not None:
            # Explicit few-step schedule. sigma = t/1000 (the scheduler's own
            # invariant, line 133); step() auto-finishes the last step to sigma=0.
            ts = torch.tensor(list(EVAL_DENOISING_STEP_LIST), dtype=torch.float32)
            sched.timesteps = ts
            sched.sigmas = ts / float(sched.num_train_timesteps)
        else:
            sched.set_timesteps(num_inference_steps=EVAL_STEPS, denoising_strength=1.0)
        sched.sigmas = sched.sigmas.to(self.device)

        B = clean_x.shape[0]
        C, H, W = clean_x.shape[2], clean_x.shape[3], clean_x.shape[4]
        lat = torch.randn([B, NUM_FRAMES, C, H, W], dtype=torch.float32, device=self.device)

        for t in sched.timesteps:
            ts = t * torch.ones([B, NUM_FRAMES], device=self.device, dtype=torch.float32)
            with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                out = self.wrapper(lat, conditional, ts, clean_x=clean_x, aug_t=None)
                flow = out[0]
            lat = sched.step(
                flow.flatten(0, 1), ts.flatten(0, 1), lat.flatten(0, 1),
            ).unflatten(dim=0, sizes=flow.shape[:2])
        return lat

    @torch.no_grad()
    def decode_latents(self, latents):
        # Wan VAE expects an extra leading latent; we decode [1:,] so pixels are 1:1 with input frames (no temporal tiling).
        dummy = latents[:, 0:1]
        lat_wd = torch.cat([dummy, latents], dim=1)
        px = self.vae.decode_to_pixel(lat_wd.float())[:, 1:, ...]
        vid = (0.5 * (px.float() + 1.0)).clamp(0, 1)
        vid_np = (vid[0].cpu().numpy() * 255).astype(np.uint8)
        if vid_np.shape[-1] != 3:
            vid_np = vid_np.transpose(0, 2, 3, 1)
        return vid_np

    @torch.no_grad()
    def compute_teacher_visuals(self, latents):
        B, F_len, C, H, W = latents.shape
        n_chunks = F_len // NUM_FRAME_PER_BLOCK
        gs = 10; ocs = 12; cT = 48; N = gs ** 2
        dummy = latents[:, 0:1]
        px = self.vae.decode_to_pixel(torch.cat([dummy, latents], dim=1).float())[:, 1:, ...]
        video = (255.0 * 0.5 * (px + 1.0)).clamp(0, 255).float()
        all_m, all_z = [], []
        for b in range(B):
            vid = video[b].unsqueeze(0)
            mws = []
            for cs in range(0, vid.shape[1], cT):
                ch = vid[:, cs:min(cs + cT, vid.shape[1])]
                no = ch.shape[1] // ocs
                if no == 0: continue
                ch = ch[:, :no * ocs].clone()
                with torch.amp.autocast(device_type="cuda", enabled=True):
                    tr, vi = self.cotracker(ch, grid_size=gs)
                tw = tr.reshape(1, no, ocs, N, 2)
                vw = vi.reshape(1, no, ocs, N, 1) if vi.dim() != 3 else vi.reshape(1, no, ocs, N).unsqueeze(-1)
                dw = tw[:, :, 1:] - tw[:, :, :-1]
                mws.append(torch.cat([dw.mean(2), vw.to(dtype=dw.dtype).mean(2)], dim=-1).squeeze(0))
            if not mws:
                all_m.append(torch.zeros(n_chunks, N, 3, device=self.device))
                all_z.append(torch.zeros(n_chunks, 8, device=self.device))
                continue
            em = torch.cat(mws, 0)
            xy = em[:, :, :2].reshape(em.shape[0], 10, 10, 2)
            mu, _ = self.ss_vae.encoder(xy.permute(0, 3, 1, 2).float() / self.ss_vae_scale)
            zf = _tanh_squash(mu.squeeze(-1).squeeze(-1))
            rn = zf.shape[0]
            if rn >= n_chunks:
                all_m.append(em[:n_chunks]); all_z.append(zf[:n_chunks])
            else:
                all_m.append(F.pad(em, (0, 0, 0, 0, 0, n_chunks - rn)))
                all_z.append(F.pad(zf, (0, 0, 0, n_chunks - rn)))
        return torch.stack(all_m), torch.stack(all_z)

    @torch.no_grad()
    def run_critic(self, latents, chunk_z2z7):
        if self.action_critic is None:
            return None
        nc = latents.shape[1] // NUM_FRAME_PER_BLOCK
        try:
            return self.action_critic(
                latents, torch.zeros(1, nc, device=self.device),
                chunk_z2z7[:, :nc].to(self.device),
            )[:, :nc]
        except Exception:
            # e.g. critic8: an 8-dim action critic vs the 2-dim z2/z7 fed here.
            # The critic readout is an optional annotation overlay; skip it
            # rather than aborting the whole rollout video.
            return None


# ---------------------------------------------------------------------------
# Load 3 seed frames from a test ride (using cached manifest)
# ---------------------------------------------------------------------------

def _unwrap_manifest_rides(manifest):
    """Return a flat ride list from either a plain list or a cached split dict."""
    if isinstance(manifest, (list, tuple)):
        return list(manifest), "root"

    if isinstance(manifest, dict):
        preferred_keys = (
            "eval", "eval_rides", "test", "test_rides",
            "val", "validation", "rides",
            "all_rides", "items", "manifest",
        )
        for key in preferred_keys:
            value = manifest.get(key)
            if isinstance(value, (list, tuple)):
                return list(value), key

        values = list(manifest.values())
        if values and all(isinstance(v, dict) and "zarr_path" in v for v in values):
            return values, "values"

    raise TypeError(
        f"Unsupported manifest structure: {type(manifest).__name__}"
    )


def _count_latent_frames(zpath: str) -> int:
    import zarr as zarr_lib
    g = zarr_lib.open_group(zpath, mode="r")
    return int(g["latents"].shape[0])


# ---------------------------------------------------------------------------
# Disk-based ride-entry fallback (for rides absent from a training manifest,
# e.g. OOD cities the v14 model wasn't trained on).
# ---------------------------------------------------------------------------

_TS_TO_RIDE_DIR_CACHE: Dict[str, Dict[str, Path]] = {}


def _build_ts_to_ride_dir(caption_root: Path) -> Dict[str, Path]:
    """Map ride_ts -> Path(caption_root/output_rides_N/ride_<id>_<ts>).

    Cached per caption_root for cheap re-use across ranks in-process.
    """
    key = str(caption_root.resolve())
    cached = _TS_TO_RIDE_DIR_CACHE.get(key)
    if cached is not None:
        return cached
    mapping: Dict[str, Path] = {}
    if not caption_root.exists():
        _TS_TO_RIDE_DIR_CACHE[key] = mapping
        return mapping
    for out_dir in caption_root.iterdir():
        if not out_dir.is_dir():
            continue
        for ride_dir in out_dir.iterdir():
            if not ride_dir.is_dir():
                continue
            ts = ride_dir.name.split("_")[-1]
            mapping.setdefault(ts, ride_dir)
    _TS_TO_RIDE_DIR_CACHE[key] = mapping
    return mapping


def _load_prompt_embeds_from_encoded_json(ride_dir: Path) -> torch.Tensor:
    """Load T5 prompt embeds (shape [512, 4096]) from *_encoded.json."""
    p = ride_dir / f"{ride_dir.name}_video_captions_OpenGVLab_InternVL3_8B_encoded.json"
    if not p.exists():
        raise FileNotFoundError(
            f"Encoded caption JSON not found for ride {ride_dir.name} at {p}"
        )
    with open(p, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    arr = data.get("caption_encoded")
    if arr is None:
        raise RuntimeError(f"'caption_encoded' key missing in {p}")
    return torch.tensor(arr, dtype=torch.float32)


def _load_ride_entry_from_disk(
    zarr_basename: str,
    encoded_root: Path,
    caption_root: Path,
    ts_to_ride_dir: Dict[str, Path],
) -> dict:
    """Build a manifest-compatible ride dict from disk for OOD / missing zarrs.

    Returns the same keys ZarrRideDataset.from_manifest expects:
    {zarr_path, prompt_embeds, attrs, n_latent_frames}.
    """
    import zarr as zarr_lib

    zpath = encoded_root / zarr_basename
    if not zpath.exists():
        raise FileNotFoundError(f"Encoded zarr not found: {zpath}")
    ts = zpath.name.removesuffix(".zarr")
    ride_dir = ts_to_ride_dir.get(ts)
    if ride_dir is None:
        raise FileNotFoundError(
            f"No caption ride dir found for ts={ts} under caption_root."
        )
    pe = _load_prompt_embeds_from_encoded_json(ride_dir)
    g = zarr_lib.open_group(str(zpath), mode="r")
    attrs = dict(g.attrs)
    n_lat = int(g["latents"].shape[0])
    return {
        "zarr_path": str(zpath),
        "prompt_embeds": pe,
        "attrs": attrs,
        "n_latent_frames": n_lat,
    }


def load_seed_frames(
    device,
    manifest_path,
    num_segments: int,
    test_ride_idx=0,
    latent_start_offset: int = EVAL_LATENT_START_OFFSET,
    motion_root: str = "",
    ss_vae_checkpoint: str = "action_query/checkpoints/ss_vae_8free.pt",
    action_dims: Optional[List[int]] = None,
    eval_zarr_list: Optional[List[str]] = None,
    fallback_encoded_root: Optional[str] = None,
    fallback_caption_root: Optional[str] = None,
    per_rank_load: bool = False,
):
    """Load one distinct zarr ride per segment (clean latents, caption, motion z).

    If ``eval_zarr_list`` is provided, those zarrs are used in the given order
    (matched by basename or full path against manifest entries). Otherwise,
    the manifest is scanned from ``test_ride_idx`` (wrapping), keeping rides
    with ≥ ``latent_start_offset + STREAM_LATENT_SPAN`` latents, skipping
    duplicate ``zarr_path`` strings so each segment uses a different file
    when possible.
    """
    if action_dims is None:
        action_dims = [2, 7]
    if num_segments < 1:
        raise ValueError("num_segments must be >= 1")
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    payload = [None]

    # In per-rank mode every rank loads its own rides; otherwise only rank 0 loads
    # and broadcasts so all ranks share the same rides.
    do_load = True if per_rank_load else (rank == 0)

    if do_load:
        log.info("Loading manifest from %s ...", manifest_path)
        manifest = torch.load(manifest_path, map_location="cpu", weights_only=False)
        rides, source = _unwrap_manifest_rides(manifest)
        if not rides:
            raise RuntimeError(f"No rides found in manifest: {manifest_path}")
        need = latent_start_offset + STREAM_LATENT_SPAN
        log.info(
            "Manifest loaded: %d rides from '%s' (need ≥%d latents; picking %d distinct zarrs)",
            len(rides), source, need, num_segments,
        )

        n_r = len(rides)
        chosen: List[Tuple[int, dict, str, int]] = []

        if eval_zarr_list:
            # Explicit zarr list: match by basename (fall back to full-path match).
            by_base: Dict[str, Tuple[int, dict]] = {}
            by_full: Dict[str, Tuple[int, dict]] = {}
            for i, r in enumerate(rides):
                zp = str(r["zarr_path"])
                by_full[zp] = (i, r)
                by_base.setdefault(Path(zp).name, (i, r))

            # Prepare disk-fallback lookup once (only if fallback roots were provided).
            ts_to_ride_dir: Dict[str, Path] = {}
            enc_root_path: Optional[Path] = None
            cap_root_path: Optional[Path] = None
            if fallback_encoded_root and fallback_caption_root:
                enc_root_path = Path(fallback_encoded_root)
                cap_root_path = Path(fallback_caption_root)
                ts_to_ride_dir = _build_ts_to_ride_dir(cap_root_path)
                log.info(
                    "Disk-fallback enabled: encoded_root=%s caption_root=%s "
                    "(indexed %d ride_dirs)",
                    enc_root_path, cap_root_path, len(ts_to_ride_dir),
                )

            n_synth = -1
            for req in eval_zarr_list:
                req_str = str(req).strip()
                key = req_str if req_str in by_full else Path(req_str).name
                entry = by_full.get(req_str) or by_base.get(key)
                if entry is not None:
                    midx, ride = entry
                    zp = str(ride["zarr_path"])
                    n_lat = _count_latent_frames(zp)
                    if n_lat < need:
                        raise RuntimeError(
                            f"Ride {Path(zp).name} has {n_lat} latents < required {need}; "
                            f"increase --latent_start_offset tolerance or pick another zarr."
                        )
                    chosen.append((midx, ride, zp, n_lat))
                    continue

                # Not in manifest → try disk fallback.
                if enc_root_path is None:
                    raise RuntimeError(
                        f"Requested eval zarr {req_str!r} not found in manifest "
                        f"({manifest_path}) and no fallback roots provided. "
                        "Pass --encoded_root / --caption_root to enable disk fallback."
                    )
                basename = Path(req_str).name
                try:
                    ride = _load_ride_entry_from_disk(
                        basename, enc_root_path, cap_root_path, ts_to_ride_dir,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Disk fallback failed for {basename!r}: {exc}"
                    ) from exc
                if ride["n_latent_frames"] < need:
                    raise RuntimeError(
                        f"Ride {basename} has {ride['n_latent_frames']} latents < required {need}."
                    )
                # Use a synthetic manifest_idx so we can still log it.
                chosen.append((n_synth, ride, ride["zarr_path"], ride["n_latent_frames"]))
                log.info(
                    "  [disk-fallback] loaded %s (%d latents) from %s",
                    basename, ride["n_latent_frames"], enc_root_path,
                )
                n_synth -= 1

            log.info(
                "Using explicit eval_zarr_list (%d rides): %s",
                len(chosen), [Path(p).name for _, _, p, _ in chosen],
            )
        else:
            start_i = min(max(0, test_ride_idx), n_r - 1)
            order = list(range(start_i, n_r)) + list(range(0, start_i))

            used_paths = set()
            for idx in order:
                if len(chosen) >= num_segments:
                    break
                cand = rides[idx]
                zp = str(cand["zarr_path"])
                n_lat = _count_latent_frames(zp)
                if n_lat < need:
                    log.info(
                        "Skipping ride idx=%d (%s): %d latents < required %d",
                        idx, Path(zp).name, n_lat, need,
                    )
                    continue
                if zp in used_paths:
                    log.info(
                        "Skipping ride idx=%d (%s): duplicate zarr_path (already used)",
                        idx, Path(zp).name,
                    )
                    continue
                chosen.append((idx, cand, zp, n_lat))
                used_paths.add(zp)

            if len(chosen) < num_segments:
                raise RuntimeError(
                    f"Need {num_segments} distinct eligible rides (≥{need} latents each); "
                    f"found only {len(chosen)}. test_ride_idx={start_i}, manifest has {n_r} rides."
                )

        from utils.zarr_dataset import ZarrRideDataset
        import zarr as zarr_lib

        rides_for_ds = []
        for _mi, ride, zpath, n_lat in chosen:
            g = zarr_lib.open_group(zpath, mode="r")
            attrs = ride.get("attrs")
            if attrs is None:
                attrs = dict(g.attrs)
            pe = ride["prompt_embeds"].cpu()
            rides_for_ds.append({
                "zarr_path": zpath,
                "prompt_embeds": pe,
                "attrs": attrs,
                "n_latent_frames": n_lat,
            })

        z_ds = ZarrRideDataset.from_manifest(
            rides_data=rides_for_ds,
            motion_root=motion_root,
            ss_vae_checkpoint=ss_vae_checkpoint,
            device="cpu",
            ss_vae_device="cpu",
        )

        segments_out = []
        ride_meta_out = []
        for manifest_idx, ride, zpath, n_lat in chosen:
            g = zarr_lib.open_group(zpath, mode="r")
            lat_np = g["latents"][latent_start_offset : latent_start_offset + NUM_FRAMES]
            assert lat_np.shape[0] == NUM_FRAMES
            lat = torch.from_numpy(lat_np.astype(np.float32)).cpu()
            pe = ride["prompt_embeds"].cpu()

            z_win = z_ds.encode_z_actions_window(
                zpath, n_lat,
                latent_start_offset,
                latent_start_offset + STREAM_LATENT_SPAN,
            )
            z_clean = z_win[:NUM_FRAMES, action_dims].unsqueeze(0).float()

            log.info(
                "Segment clean ride manifest_idx=%d (%s): %d latents, slice [%d:%d)",
                manifest_idx, Path(zpath).name, n_lat,
                latent_start_offset, latent_start_offset + NUM_FRAMES,
            )

            segments_out.append({
                "seed": lat[:CONTEXT_FRAMES].clone(),
                "clean_21": lat[:NUM_FRAMES].clone(),
                "prompt_embeds": pe.clone(),
                "clean_frame_actions": z_clean.clone(),
            })
            ride_meta_out.append({
                "manifest_idx": manifest_idx,
                "zarr_path": zpath,
            })

        payload[0] = {"segments": segments_out, "ride_meta": ride_meta_out}
        del manifest, rides

    if dist.is_available() and dist.is_initialized() and not per_rank_load:
        dist.broadcast_object_list(payload, src=0)

    data = payload[0]
    segs = data["segments"]
    ride_meta = data["ride_meta"]
    per_seg_seed = [s["seed"].to(device) for s in segs]
    per_seg_clean_x = [s["clean_21"].unsqueeze(0).to(device) for s in segs]
    per_seg_prompt = [s["prompt_embeds"].unsqueeze(0).to(device) for s in segs]
    per_seg_clean_actions = [s["clean_frame_actions"].to(device) for s in segs]
    return per_seg_seed, per_seg_clean_x, per_seg_prompt, per_seg_clean_actions, ride_meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="eval/eval_chain_out")
    parser.add_argument("--config", type=str, default="configs/causal_lora_diffusion_teacher.yaml")
    parser.add_argument("--manifest", type=str,
                        default="logs/z_critic_v10_state_tokens/.ride_manifest.pt")
    parser.add_argument("--test_ride_idx", type=int, default=0,
                        help="First manifest index to scan for segment-1 zarr; each segment uses the next distinct eligible ride")
    parser.add_argument(
        "--latent_start_offset", type=int, default=EVAL_LATENT_START_OFFSET,
        help="Start index into ride latents for clean_x (needs offset+24 latents for motion z window)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--assignment_index",
        type=int,
        default=None,
        help="Single-GPU smoke test only: run MODEL_ASSIGNMENTS[i] (e.g. 3=v10). "
        "Requires WORLD_SIZE=1 (plain python, not torchrun --nproc_per_node>1).",
    )
    parser.add_argument(
        "--num_segments",
        type=int,
        default=None,
        help=f"If set, how many rollout segments to run (default {NUM_EVAL_SEGMENTS}).",
    )
    parser.add_argument(
        "--eval_zarrs",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Explicit list of zarr basenames (e.g. 20240216101235.zarr) or full paths "
            "to use as eval rides, in order. If given, overrides --test_ride_idx / "
            "manifest scanning. num_segments defaults to len(eval_zarrs)."
        ),
    )
    parser.add_argument(
        "--noisy_mode",
        type=str,
        choices=["fixed", "dataset", "dataset_plus_cf"],
        default="fixed",
        help=(
            "How to drive the noisy-side action conditioning. "
            "'fixed' (default) uses the hardcoded z2/z7 schedule per segment. "
            "'dataset' uses the ride's own clean z-actions (same as clean side). "
            "'dataset_plus_cf' produces TWO videos per segment: normal (noisy=dataset) "
            "and counterfactual (noisy=-dataset). Both share the same denoising seed."
        ),
    )
    parser.add_argument(
        "--per_rank_zarrs",
        type=str,
        nargs="+",
        default=None,
        help=(
            "List of per-rank comma-separated zarr lists. Item i is used by rank i. "
            "E.g. --per_rank_zarrs 'madrid1.zarr,rome1.zarr,wellington1.zarr' "
            "'madrid1.zarr,rome1.zarr,brighton1.zarr'. Overrides --eval_zarrs / --test_ride_idx. "
            "Output goes to output_dir/<label>/rank<i>/."
        ),
    )
    parser.add_argument(
        "--encoded_root",
        type=str,
        default=DEFAULT_ENCODED_ROOT,
        help=(
            "Fallback encoded zarr root for rides NOT in the manifest (e.g. OOD cities). "
            "Defaults to the big shared dataset."
        ),
    )
    parser.add_argument(
        "--caption_root",
        type=str,
        default=DEFAULT_CAPTION_ROOT,
        help=(
            "Fallback caption root for loading prompt-embeds from *_encoded.json "
            "when a ride is not in the manifest."
        ),
    )
    parser.add_argument(
        "--force_same_assignment",
        action="store_true",
        help=(
            "Force ALL ranks to use MODEL_ASSIGNMENTS[--assignment_index] (same checkpoint). "
            "Typical with --per_rank_zarrs for running one model across different rides "
            "per GPU via torchrun."
        ),
    )
    args = parser.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)

    if args.assignment_index is not None:
        if not (0 <= args.assignment_index < len(MODEL_ASSIGNMENTS)):
            raise SystemExit(
                f"--assignment_index must be in [0, {len(MODEL_ASSIGNMENTS) - 1}]"
            )
        if world != 1 and not args.force_same_assignment:
            raise SystemExit(
                "--assignment_index with multi-rank requires --force_same_assignment "
                "(all ranks run the same checkpoint)."
            )
        eff_rank = args.assignment_index
    elif args.force_same_assignment:
        raise SystemExit("--force_same_assignment requires --assignment_index to be set.")
    else:
        eff_rank = rank

    # Resolve per-rank zarr list if --per_rank_zarrs was given.
    rank_zarrs: Optional[List[str]] = None
    if args.per_rank_zarrs is not None:
        if args.eval_zarrs is not None:
            raise SystemExit("Use either --per_rank_zarrs or --eval_zarrs, not both.")
        if len(args.per_rank_zarrs) < world:
            raise SystemExit(
                f"--per_rank_zarrs has {len(args.per_rank_zarrs)} groups but world_size={world}."
            )
        grp = args.per_rank_zarrs[rank]
        rank_zarrs = [z.strip() for z in grp.split(",") if z.strip()]
        if not rank_zarrs:
            raise SystemExit(f"Rank {rank} per_rank_zarrs group is empty: {grp!r}")
        log.info("Rank %d: per_rank_zarrs group = %s", rank, rank_zarrs)

    effective_zarr_list = rank_zarrs if rank_zarrs is not None else args.eval_zarrs

    if effective_zarr_list:
        default_n = len(effective_zarr_list)
    else:
        default_n = NUM_EVAL_SEGMENTS
    n_seg = args.num_segments if args.num_segments is not None else default_n
    if n_seg < 1:
        raise SystemExit("--num_segments must be >= 1")
    if args.noisy_mode == "fixed" and n_seg > NUM_EVAL_SEGMENTS:
        raise SystemExit(
            f"--num_segments must be <= {NUM_EVAL_SEGMENTS} when --noisy_mode=fixed "
            f"(the hardcoded z2/z7 schedule only has {NUM_EVAL_SEGMENTS} segments)."
        )
    if effective_zarr_list and n_seg != len(effective_zarr_list):
        raise SystemExit(
            f"--num_segments ({n_seg}) must match len(effective zarr list) ({len(effective_zarr_list)})."
        )

    if args.noisy_mode == "fixed":
        segment_actions_np = build_all_segment_noisy_frame_actions(args.seed, NUM_EVAL_SEGMENTS)
    else:
        segment_actions_np = None  # driven from per-segment dataset z-actions instead

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if rank == 0 and args.noisy_mode == "fixed":
        meta_path = Path(args.output_dir) / "segment_noisy_frame_actions.json"
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "cli_seed": args.seed,
                    "noisy_z2_by_segment": NOISY_Z2_BY_SEGMENT.tolist(),
                    "noisy_z7_chunks": NOISY_Z7_CHUNKS.tolist(),
                    "segments": [a.tolist() for a in segment_actions_np],
                },
                fh,
                indent=2,
            )
        log.info("Wrote shared per-segment noisy actions (all GPUs): %s", meta_path)
    elif rank == 0:
        log.info(
            "noisy_mode=%s: per-segment noisy actions will be derived from dataset z-actions.",
            args.noisy_mode,
        )

    if eff_rank >= len(MODEL_ASSIGNMENTS):
        log.info("Rank %d has no model assignment, exiting.", eff_rank)
        return

    assignment = MODEL_ASSIGNMENTS[eff_rank]
    label = assignment["label"]
    log.info("Rank %d (assignment %d) → %s  (%s)", rank, eff_rank, label, assignment["ckpt"])

    # When ranks run different rides (per_rank_zarrs OR force_same_assignment with world>1),
    # split outputs into per-rank subdirs to avoid cross-rank file collisions.
    needs_rank_subdir = (rank_zarrs is not None) or (args.force_same_assignment and world > 1)
    if needs_rank_subdir:
        out_dir = Path(args.output_dir) / label / f"rank{rank}"
    else:
        out_dir = Path(args.output_dir) / label
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = ChainPipeline(device)
    pipe.build(args.config, use_action_tokens=assignment.get("has_action_tokens", True))
    step = pipe.load_checkpoint(assignment["ckpt"], assignment)

    from omegaconf import OmegaConf
    _cfg = OmegaConf.load(args.config)
    motion_root = str(_cfg.get("motion_root", "") or "")
    if not motion_root:
        raise ValueError("config motion_root is required for clean-side motion encoding")
    ss_vae_ckpt = str(_cfg.get("ss_vae_checkpoint", "action_query/checkpoints/ss_vae_8free.pt"))
    action_dims = list(_cfg.get("action_dims", [2, 7]))

    (
        per_seg_seed,
        per_seg_clean_x,
        per_seg_prompt,
        per_seg_clean_actions,
        ride_meta,
    ) = load_seed_frames(
        device, args.manifest, n_seg,
        test_ride_idx=args.test_ride_idx,
        latent_start_offset=args.latent_start_offset,
        motion_root=motion_root,
        ss_vae_checkpoint=ss_vae_ckpt,
        action_dims=action_dims,
        eval_zarr_list=effective_zarr_list,
        fallback_encoded_root=args.encoded_root,
        fallback_caption_root=args.caption_root,
        per_rank_load=(rank_zarrs is not None),
    )
    # Write per-segment clean-ride manifest. Keep metadata co-located with the
    # video output: if each rank writes to its own subdir (per_rank_zarrs OR
    # force_same_assignment + world>1), each rank also writes its own JSON
    # next to its videos. Otherwise rank 0 writes a single shared file under
    # output_dir/.
    if needs_rank_subdir:
        rides_path = out_dir / "segment_clean_rides.json"
        write_meta = True
    else:
        rides_path = Path(args.output_dir) / "segment_clean_rides.json"
        write_meta = rank == 0
    if write_meta:
        with open(rides_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "latent_start_offset": args.latent_start_offset,
                    "segments": ride_meta,
                },
                fh,
                indent=2,
            )
        log.info("Wrote per-segment clean zarr paths: %s", rides_path)

    total_videos = 0
    for seg_idx in range(n_seg):
        seg_label_base = f"{label} seg{seg_idx + 1}"
        log.info("[%s] === Segment %d/%d ===", label, seg_idx + 1, n_seg)

        prompt_embeds_seg = per_seg_prompt[seg_idx].to(dtype=pipe.dtype)
        clean_frame_actions_seg = per_seg_clean_actions[seg_idx].to(dtype=pipe.dtype)
        context_latents_3 = per_seg_seed[seg_idx].unsqueeze(0)
        clean_x = per_seg_clean_x[seg_idx]

        zarr_name = Path(ride_meta[seg_idx]["zarr_path"]).stem if ride_meta else ""

        # Build list of (cond_tag, noisy_frame_actions_tensor) to run for this segment.
        conditions: List[Tuple[str, torch.Tensor]] = []
        if args.noisy_mode == "fixed":
            a = segment_actions_np[seg_idx]
            log.info(
                "[%s]   Fixed noisy cmd z2=%.2f (const) z7 blocks %s",
                label, float(NOISY_Z2_BY_SEGMENT[seg_idx]), NOISY_Z7_CHUNKS.tolist(),
            )
            noisy_fa = torch.from_numpy(a).unsqueeze(0).to(device=device, dtype=pipe.dtype)
            conditions.append(("fixed", noisy_fa))
        else:
            dataset_fa = clean_frame_actions_seg.clone()
            if args.noisy_mode == "dataset":
                conditions.append(("normal", dataset_fa))
                log.info("[%s]   Noisy = dataset z-actions", label)
            elif args.noisy_mode == "dataset_plus_cf":
                conditions.append(("normal", dataset_fa))
                conditions.append(("counterfactual", -dataset_fa))
                log.info(
                    "[%s]   Running BOTH normal (noisy=dataset) and counterfactual (noisy=-dataset)",
                    label,
                )

        for cond_tag, noisy_frame_actions in conditions:
            chunk_actions_dev = frame_actions_to_chunk_actions(noisy_frame_actions)

            cond = pipe.build_conditional(
                prompt_embeds_seg, noisy_frame_actions, clean_frame_actions_seg,
                assignment.get("has_adaln", False),
                assignment.get("has_action_tokens", False),
            )

            # Same denoising seed across sibling conditions (e.g. normal vs cf) of the
            # same segment → clean A/B comparison at the noise-init level.
            noise_seed = args.seed + VIDEO_NOISE_BASE + seg_idx * VIDEO_NOISE_SEED_STRIDE
            torch.manual_seed(noise_seed)
            torch.cuda.manual_seed(noise_seed)

            t0 = time.time()
            gen_latents = pipe.generate(cond, clean_x)
            log.info(
                "[%s]   [%s] Generated in %.1fs (noise_seed=%s)",
                label, cond_tag, time.time() - t0, noise_seed,
            )

            context_np = pipe.decode_latents(context_latents_3)
            video_np = pipe.decode_latents(gen_latents)
            video_with_context_np = np.concatenate([context_np, video_np], axis=0)

            motion, teacher_z_8d = pipe.compute_teacher_visuals(gen_latents)
            n_c = teacher_z_8d.shape[1]
            teacher_z2z7 = teacher_z_8d[:, :, CRITIC_ACTION_DIMS]

            critic_z2z7 = None
            if pipe.action_critic is not None:
                cp = pipe.run_critic(gen_latents, chunk_actions_dev)
                if cp is not None:
                    critic_z2z7 = cp[:, :, CRITIC_ACTION_DIMS]

            target_z = chunk_actions_dev[:, :n_c].contiguous()

            seg_label = f"{seg_label_base} {cond_tag}"
            if args.noisy_mode == "fixed":
                fname_stem = f"seg{seg_idx + 1}"
            else:
                pieces = [f"seg{seg_idx + 1}"]
                if zarr_name:
                    pieces.append(zarr_name)
                pieces.append(cond_tag)
                fname_stem = "_".join(pieces)

            ann = annotate_video(video_np, teacher_z2z7, critic_z2z7, target_z, motion, seg_label)
            ann_with_context = np.concatenate([context_np, ann], axis=0)
            frames_to_mp4(ann_with_context, str(out_dir / f"{fname_stem}_annotated.mp4"))
            frames_to_mp4(video_with_context_np, str(out_dir / f"{fname_stem}_raw.mp4"))
            total_videos += 1

            torch.cuda.empty_cache()

    log.info("[%s] Done! Saved %d video(s) across %d segment(s) to %s",
             label, total_videos, n_seg, out_dir)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
