#!/usr/bin/env python3
"""Offline evaluation: generate a video from each checkpoint with user-specified
z2/z7 action pairs, then overlay teacher + critic annotation bars.

Usage (single GPU):

    python utils/evaluate_models.py \\
        --actions "0.3 0.0, 0.5 0.1, -0.2 0.3, 0.0 -0.5, 0.6 0.0, -0.1 0.2, 0.4 -0.1" \\
        --versions v7 v8 v11 v12 \\
        --output_dir eval/eval_output

    # Evaluate all registered versions:
    python utils/evaluate_models.py --actions "..." --all --output_dir eval/eval_output

    # Or pass explicit .pt paths:
    python utils/evaluate_models.py --actions "..." \\
        --checkpoints logs/z_critic_v12_probe_fixes/causal_lora_step0003250.pt \\
        --output_dir eval/eval_output

Seven "action pairs" → 7 chunks × 3 latent frames = 21-frame generation window.
The script loads the Wan base model once, then hot-swaps LoRA / critic / projection
weights per checkpoint.  Overlays show teacher (CoTracker → ss_vae) z2/z7, commanded target z2/z7, and
critic z2/z7 from a **shared frozen v12** action critic so scores are comparable
across checkpoints.
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
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (match training config)
# ---------------------------------------------------------------------------
NUM_FRAME_PER_BLOCK = 3
CONTEXT_FRAMES = 3
NUM_FRAMES = 21
EVAL_STEPS = 48
MODEL_NAME = "Wan2.1-T2V-1.3B"
RAW_ACTION_DIM = 2
ACTION_DIMS = [2, 7]
CRITIC_ACTION_DIMS = [2, 7]
CRITIC_Z_OUT = 8

EVAL_ACTION_CRITIC_CKPT = "logs/z_critic_v12_probe_fixes/causal_lora_step0003250.pt"
EVAL_CRITIC_BASE_CH = 128
EVAL_CRITIC_RES_BLOCKS = 4

_SHARED_V12_ACTION_CRITIC_SD: Optional[Dict[str, torch.Tensor]] = None

# ---------------------------------------------------------------------------
# Version registry: each entry describes what modules that run contains
# ---------------------------------------------------------------------------
CHECKPOINT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "v1": {
        "logdir": "logs/causal_lora_teacher",
        "label": "v1 (baseline causal teacher)",
        "has_critic": False,
        "has_adaln": False,
        "has_action_tokens": False,
        "critic_base_ch": 64,
        "critic_res_blocks": 3,
    },
    "v2": {
        "logdir": "logs/causal_lora_teacher_v2",
        "label": "v2 (16-GPU, upgraded critic)",
        "has_critic": False,
        "has_adaln": False,
        "has_action_tokens": False,
        "critic_base_ch": 128,
        "critic_res_blocks": 4,
    },
    "v7": {
        "logdir": "logs/z_critic_v7_aux_token",
        "label": "v7 (aux z-token, action injection)",
        "has_critic": True,
        "has_adaln": True,
        "has_action_tokens": True,
        "critic_base_ch": 128,
        "critic_res_blocks": 4,
    },
    "v8": {
        "logdir": "logs/z_critic_v8_aux_critic",
        "label": "v8 (multi-step critic)",
        "has_critic": True,
        "has_adaln": True,
        "has_action_tokens": True,
        "critic_base_ch": 128,
        "critic_res_blocks": 4,
    },
    "v9": {
        "logdir": "logs/z_critic_v9_state_tokens",
        "label": "v9 (state tokens)",
        "has_critic": True,
        "has_adaln": True,
        "has_action_tokens": True,
        "critic_base_ch": 128,
        "critic_res_blocks": 4,
    },
    "v11": {
        "logdir": "logs/z_critic_v11_cross_attn_probes",
        "label": "v11 (cross-attn probes)",
        "has_critic": True,
        "has_adaln": True,
        "has_action_tokens": True,
        "critic_base_ch": 128,
        "critic_res_blocks": 4,
    },
    "v12": {
        "logdir": "logs/z_critic_v12_probe_fixes",
        "label": "v12 (probe fixes)",
        "has_critic": True,
        "has_adaln": True,
        "has_action_tokens": True,
        "critic_base_ch": 128,
        "critic_res_blocks": 4,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_action_pairs(raw: str) -> torch.Tensor:
    """Parse "z2 z7, z2 z7, ..." into ``[1, 7, 2]`` tensor."""
    pairs = [p.strip() for p in raw.split(",")]
    if len(pairs) != 7:
        raise ValueError(f"Need exactly 7 action pairs (one per chunk), got {len(pairs)}")
    out = []
    for p in pairs:
        vals = p.split()
        if len(vals) != 2:
            raise ValueError(f"Each pair must be 'z2 z7', got: '{p}'")
        out.append([float(vals[0]), float(vals[1])])
    return torch.tensor(out, dtype=torch.float32).unsqueeze(0)


def expand_chunks_to_frames(chunks: torch.Tensor) -> torch.Tensor:
    """[1, 7, D] → [1, 21, D]"""
    return chunks.repeat_interleave(NUM_FRAME_PER_BLOCK, dim=1)


def latest_checkpoint(logdir: str) -> Optional[Path]:
    d = Path(logdir)
    if not d.exists():
        return None
    pts = sorted(d.glob("causal_lora_step*.pt"))
    return pts[-1] if pts else None


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
    """Replicate the trainer's target-module collection: every Linear inside
    WanAttentionBlock / CausalWanAttentionBlock."""
    target_modules = set()
    for module_name, module in model.named_modules():
        if module.__class__.__name__ in {"WanAttentionBlock", "CausalWanAttentionBlock"}:
            for full_name, submodule in module.named_modules(prefix=module_name):
                if isinstance(submodule, nn.Linear):
                    target_modules.add(full_name)
    return sorted(target_modules)


# ---------------------------------------------------------------------------
# Annotation overlay (matches _draw_z_action_overlay / _annotate_action_video)
# ---------------------------------------------------------------------------

def _draw_bar(out, panel_x0, y_mid, value, color, label, clip=1.0):
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
    cv2.putText(out, f"{label} {value:+.3f}", (panel_x0 + 8, y_mid + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)


def annotate_video(video_np, teacher_z2z7, critic_z2z7, target_z2z7, motion,
                   title, grid_size=10, output_chunk_size=12):
    """Overlay motion arrows + teacher/critic/target z2/z7 bars onto video frames."""
    teacher_np = teacher_z2z7[0].cpu().float().numpy()
    critic_np = critic_z2z7[0].cpu().float().numpy() if critic_z2z7 is not None else None
    target_np = target_z2z7[0].cpu().float().numpy()
    motion_np = motion[0].cpu().float().numpy() if motion is not None else None
    n_seg = teacher_np.shape[0]
    panel_w = 220

    frames_out = []
    for fi in range(video_np.shape[0]):
        si = min(fi // output_chunk_size, n_seg - 1)
        f = video_np[fi].copy()
        h, w = f.shape[:2]

        if motion_np is not None and si < motion_np.shape[0]:
            mvecs = motion_np[si]
            drawable_w = max(1, w - panel_w)
            for gy in range(grid_size):
                for gx in range(grid_size):
                    idx = gy * grid_size + gx
                    if idx >= mvecs.shape[0]:
                        break
                    dx, dy, vis_val = mvecs[idx]
                    if vis_val < 0.2:
                        continue
                    cx_pt = int((gx + 0.5) * drawable_w / grid_size)
                    cy_pt = int((gy + 0.5) * h / grid_size)
                    ex_pt = int(cx_pt - dx * 3)
                    ey_pt = int(cy_pt - dy * 3)
                    color = (0, 255, 0) if vis_val >= 0.5 else (0, 200, 255)
                    cv2.arrowedLine(f, (cx_pt, cy_pt), (ex_pt, ey_pt), color, 1, tipLength=0.3)

        panel_x0 = w - panel_w
        f[:, panel_x0:] = (f[:, panel_x0:].astype(np.float32) * 0.22).astype(np.uint8)

        cv2.putText(f, title, (panel_x0 + 8, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(f, f"frame={fi} seg={si}", (panel_x0 + 8, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 200), 1, cv2.LINE_AA)

        has_critic = critic_np is not None
        if has_critic:
            cv2.putText(f, "teacher(g) critic(c) target(b)", (panel_x0 + 8, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180, 180, 180), 1, cv2.LINE_AA)
        else:
            cv2.putText(f, "teacher(g) target(b)", (panel_x0 + 8, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180, 180, 180), 1, cv2.LINE_AA)

        step = 18
        z2_y0, z7_y0 = 78, 138

        for z_name, t_val, c_val, tgt_val, y0 in [
            ("z2", teacher_np[si, 0], (critic_np[si, 0] if has_critic else None), target_np[si, 0], z2_y0),
            ("z7", teacher_np[si, 1], (critic_np[si, 1] if has_critic else None), target_np[si, 1], z7_y0),
        ]:
            cv2.putText(f, z_name, (panel_x0 + 8, y0 - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 120), 1, cv2.LINE_AA)
            y = y0
            _draw_bar(f, panel_x0, y, t_val, (80, 220, 80), "t")
            y += step
            if c_val is not None:
                _draw_bar(f, panel_x0, y, c_val, (100, 200, 255), "c")
                y += step
            _draw_bar(f, panel_x0, y, tgt_val, (80, 80, 220), "y")

        frames_out.append(f)
    return np.stack(frames_out)


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------

class EvalPipeline:
    """Builds the model scaffold once, then hot-swaps weights per checkpoint."""

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.bfloat16):
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
        self._text_encoder = None

    def build(self, config_path: str = "configs/causal_lora_diffusion_teacher.yaml"):
        from omegaconf import OmegaConf
        from utils.wan_wrapper import WanDiffusionWrapper, WanVAEWrapper
        from model.action_model_patch import apply_action_patches
        from model.action_modulation import ActionModulationProjection, ActionTokenProjection

        cfg = OmegaConf.load(config_path)

        log.info("Building Wan base model (%s)...", MODEL_NAME)
        model_kwargs = dict(OmegaConf.to_container(cfg.get("model_kwargs", {})))
        self.wrapper = WanDiffusionWrapper(
            model_name=cfg.get("model_name", MODEL_NAME),
            is_causal=True,
            **model_kwargs,
        )
        self.wrapper.model.num_frame_per_block = NUM_FRAME_PER_BLOCK
        self.wrapper.model.context_shift = CONTEXT_FRAMES // NUM_FRAME_PER_BLOCK

        if cfg.get("gradient_checkpointing", True):
            self.wrapper.model.enable_gradient_checkpointing()

        apply_action_patches(self.wrapper)

        model_dim = getattr(self.wrapper.model, "dim", 2048)
        activation = cfg.get("action_activation", "silu")

        self.action_projection = ActionModulationProjection(
            action_dim=RAW_ACTION_DIM, activation=activation,
            hidden_dim=model_dim, zero_init=True,
        )
        self.action_token_projection = ActionTokenProjection(
            action_dim=RAW_ACTION_DIM, activation=activation,
            hidden_dim=model_dim, zero_init=True,
        )
        self.wrapper.model.action_tokens_per_frame = 1
        self.wrapper.adjust_seq_len_for_action_tokens(
            num_frames=NUM_FRAMES, action_per_frame=1,
        )

        # LoRA (same target collection as the trainer)
        import peft
        from peft import LoraConfig
        adapter_cfg = OmegaConf.to_container(cfg.get("adapter", {}))
        rank = int(adapter_cfg.get("rank", 256))
        alpha = float(adapter_cfg.get("alpha", rank))
        dropout = float(adapter_cfg.get("dropout", 0.0))
        target_modules = _collect_lora_target_modules(self.wrapper.model)
        if not target_modules:
            target_modules = ["q", "k", "v", "o"]
        lora_config = LoraConfig(
            r=rank, lora_alpha=alpha, lora_dropout=dropout,
            target_modules=target_modules, bias="none",
        )
        self.wrapper.model = peft.get_peft_model(self.wrapper.model, lora_config)
        log.info("LoRA applied: rank=%d, alpha=%.1f, %d target modules", rank, alpha, len(target_modules))

        self.wrapper.to(self.device).eval()
        self.action_projection.to(self.device).eval()
        self.action_token_projection.to(self.device).eval()

        # Frozen evaluator modules
        log.info("Loading frozen VAE...")
        self.vae = WanVAEWrapper()
        self.vae.to(self.device).eval()

        log.info("Loading frozen CoTracker...")
        self.cotracker = torch.hub.load(
            "facebookresearch/co-tracker", "cotracker2", skip_validation=True,
        )
        self.cotracker.to(self.device).eval()

        log.info("Loading frozen ss_vae...")
        ss_vae_ckpt = cfg.get("ss_vae_checkpoint", "action_query/checkpoints/ss_vae_8free.pt")
        from action_query.ss_vae_model import load_ss_vae
        self.ss_vae, self.ss_vae_scale = load_ss_vae(ss_vae_ckpt, device=str(self.device))

        log.info("Pipeline ready.")

    def load_context_latents(self, encoded_root: str, caption_root: str,
                             ride_index: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load context latents + prompt embeds from a zarr ride."""
        from utils.zarr_dataset import build_ride_manifest, ZarrRideDataset
        import zarr as zarr_lib

        manifest_cache = Path(encoded_root) / ".ride_manifest_eval.pt"
        rides = build_ride_manifest(encoded_root, caption_root,
                                    min_ride_frames=NUM_FRAMES + CONTEXT_FRAMES,
                                    cache_path=str(manifest_cache))
        if not rides:
            raise RuntimeError(f"No valid rides in {encoded_root}")

        ride = rides[min(ride_index, len(rides) - 1)]
        zpath = ride["zarr_path"]
        prompt_embeds = ride["prompt_embeds"]

        window = NUM_FRAMES + CONTEXT_FRAMES
        latents = ZarrRideDataset.load_latent_chunk(zpath, 0, window)
        context = latents[:NUM_FRAMES].unsqueeze(0).to(self.device)
        pe = prompt_embeds.unsqueeze(0).to(self.device, dtype=self.dtype)
        log.info("Loaded context from %s (%d frames)", Path(zpath).name, window)
        return context, pe

    def load_checkpoint(self, ckpt_path: str, vinfo: dict) -> dict:
        """Swap LoRA + optional critic/projection weights from a checkpoint."""
        from peft import set_peft_model_state_dict
        log.info("Loading: %s", ckpt_path)
        ck_raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Drop optimizer/critic_optimizer state to save memory (multi-GB each)
        ck = {k: v for k, v in ck_raw.items() if k not in ("optimizer", "critic_optimizer")}
        del ck_raw
        step = ck.get("step", "?")
        keys = sorted(ck.keys())
        log.info("  step=%s  keys=%s", step, keys)

        # LoRA
        try:
            set_peft_model_state_dict(self.wrapper.model, ck["lora"])
        except Exception as exc:
            log.warning("LoRA load issue (attempting cross-load): %s", exc)
            from peft import get_peft_model_state_dict
            current = get_peft_model_state_dict(self.wrapper.model)
            matched = 0
            for k in current:
                if k in ck["lora"] and current[k].shape == ck["lora"][k].shape:
                    current[k] = ck["lora"][k]
                    matched += 1
            set_peft_model_state_dict(self.wrapper.model, current)
            log.info("  Cross-loaded %d/%d LoRA keys", matched, len(current))

        # Action projections
        if "action_projection" in ck and vinfo.get("has_adaln"):
            self.action_projection.load_state_dict(ck["action_projection"])
        if "action_token_projection" in ck and vinfo.get("has_action_tokens"):
            self.action_token_projection.load_state_dict(ck["action_token_projection"])

        self.action_critic = None
        self._attach_shared_v12_eval_critic()

        self.wrapper.eval()
        return {"step": step, "keys": keys}

    def _attach_shared_v12_eval_critic(self) -> None:
        global _SHARED_V12_ACTION_CRITIC_SD
        if _SHARED_V12_ACTION_CRITIC_SD is None:
            log.info("Loading shared eval critic from %s", EVAL_ACTION_CRITIC_CKPT)
            blob = torch.load(
                EVAL_ACTION_CRITIC_CKPT, map_location="cpu", weights_only=False,
            )
            if "action_critic" not in blob:
                log.warning("No action_critic in %s; critic metrics/bars disabled.", EVAL_ACTION_CRITIC_CKPT)
                return
            _SHARED_V12_ACTION_CRITIC_SD = blob["action_critic"]
            del blob
        from model.action_critic import ActionCritic
        critic = ActionCritic(
            latent_channels=16,
            chunk_frames=NUM_FRAME_PER_BLOCK,
            action_dim=len(CRITIC_ACTION_DIMS),
            z_out_dim=CRITIC_Z_OUT,
            base_channels=EVAL_CRITIC_BASE_CH,
            num_res_blocks=EVAL_CRITIC_RES_BLOCKS,
        )
        inc = critic.load_state_dict(_SHARED_V12_ACTION_CRITIC_SD, strict=False)
        if inc.missing_keys or inc.unexpected_keys:
            log.warning("  eval critic load missing=%s unexpected=%s", inc.missing_keys, inc.unexpected_keys)
        critic.to(self.device).eval()
        self.action_critic = critic
        log.info("Shared v12 eval critic attached.")

    def build_conditional(self, prompt_embeds, z_frame_actions, use_adaln, use_tokens):
        """Build the conditional dict matching how the trainer does it."""
        conditional = {"prompt_embeds": prompt_embeds}
        if use_adaln:
            conditional["_action_modulation"] = self.action_projection(
                z_frame_actions, num_frames=NUM_FRAMES)
            conditional["_action_modulation_clean"] = self.action_projection(
                z_frame_actions, num_frames=NUM_FRAMES)
        if use_tokens:
            conditional["_action_tokens"] = self.action_token_projection(z_frame_actions)
            conditional["_action_tokens_clean"] = self.action_token_projection(z_frame_actions)
        return conditional

    @torch.no_grad()
    def generate(self, conditional, context_latents):
        from utils.scheduler import FlowMatchScheduler
        scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
        scheduler.set_timesteps(num_inference_steps=EVAL_STEPS, denoising_strength=1.0)
        scheduler.sigmas = scheduler.sigmas.to(self.device)

        B = context_latents.shape[0]
        C, H, W = context_latents.shape[2], context_latents.shape[3], context_latents.shape[4]
        latents = torch.randn([B, NUM_FRAMES, C, H, W], dtype=torch.float32, device=self.device)

        for t in scheduler.timesteps:
            ts = t * torch.ones([B, NUM_FRAMES], device=self.device, dtype=torch.float32)
            with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                model_out = self.wrapper(
                    latents, conditional, ts,
                    clean_x=context_latents, aug_t=None,
                )
                flow_pred = model_out[0]
            latents = scheduler.step(
                flow_pred.flatten(0, 1), ts.flatten(0, 1), latents.flatten(0, 1),
            ).unflatten(dim=0, sizes=flow_pred.shape[:2])
        return latents

    @torch.no_grad()
    def decode_latents(self, latents):
        dummy = latents[:, 0:1]
        lat_wd = torch.cat([dummy, latents], dim=1)
        pixels = self.vae.decode_to_pixel(lat_wd.float())[:, 1:, ...]
        video = (0.5 * (pixels.float() + 1.0)).clamp(0, 1)
        vid_np = (video[0].cpu().numpy() * 255).astype(np.uint8)
        if vid_np.shape[-1] != 3:
            vid_np = vid_np.transpose(0, 2, 3, 1)
        return vid_np

    @torch.no_grad()
    def compute_teacher_visuals(self, latents):
        """CoTracker → ss_vae → teacher z (8D) + motion vectors."""
        B, F_len, C, H, W = latents.shape
        n_chunks = F_len // NUM_FRAME_PER_BLOCK
        grid_size = 10
        output_chunk_size = 12
        compute_T = 48
        N = grid_size ** 2

        dummy = latents[:, 0:1]
        latents_wd = torch.cat([dummy, latents], dim=1)
        pixels = self.vae.decode_to_pixel(latents_wd.float())[:, 1:, ...]
        video = (255.0 * 0.5 * (pixels + 1.0)).clamp(0, 255).float()

        all_motion, all_z = [], []
        for b in range(B):
            vid = video[b].unsqueeze(0)
            T_total = vid.shape[1]
            mws = []
            for cs in range(0, T_total, compute_T):
                ce = min(cs + compute_T, T_total)
                ch = vid[:, cs:ce]
                n_out = ch.shape[1] // output_chunk_size
                if n_out == 0:
                    continue
                used = n_out * output_chunk_size
                ch = ch[:, :used].clone()
                with torch.amp.autocast(device_type="cuda", enabled=True):
                    tracks, vis = self.cotracker(ch, grid_size=grid_size)
                tw = tracks.reshape(1, n_out, output_chunk_size, N, 2)
                if vis.dim() == 3:
                    vw = vis.reshape(1, n_out, output_chunk_size, N).unsqueeze(-1)
                else:
                    vw = vis.reshape(1, n_out, output_chunk_size, N, 1)
                dw = tw[:, :, 1:] - tw[:, :, :-1]
                mo = dw.mean(dim=2)
                vo = vw.to(dtype=mo.dtype).mean(dim=2)
                mw = torch.cat([mo, vo], dim=-1).squeeze(0)
                mws.append(mw)

            if not mws:
                all_motion.append(torch.zeros(n_chunks, N, 3, device=self.device))
                all_z.append(torch.zeros(n_chunks, 8, device=self.device))
                continue

            est_motion = torch.cat(mws, dim=0)
            raw_n = est_motion.shape[0]
            xy = est_motion[:, :, :2].reshape(raw_n, 10, 10, 2)
            x_in = xy.permute(0, 3, 1, 2).float() / self.ss_vae_scale
            mu, _ = self.ss_vae.encoder(x_in.to(self.device))
            z_full = mu.squeeze(-1).squeeze(-1)
            z_full = _tanh_squash(z_full)

            if raw_n >= n_chunks:
                all_motion.append(est_motion[:n_chunks])
                all_z.append(z_full[:n_chunks])
            else:
                all_motion.append(F.pad(est_motion, (0, 0, 0, 0, 0, n_chunks - raw_n)))
                all_z.append(F.pad(z_full, (0, 0, 0, n_chunks - raw_n)))

        return torch.stack(all_motion), torch.stack(all_z)

    @torch.no_grad()
    def run_critic(self, latents, chunk_actions_z2z7):
        if self.action_critic is None:
            return None
        n_chunks = latents.shape[1] // NUM_FRAME_PER_BLOCK
        eval_t = torch.zeros(1, n_chunks, device=self.device)
        actions = chunk_actions_z2z7[:, :n_chunks].to(self.device)
        return self.action_critic(latents, eval_t, actions)[:, :n_chunks]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate causal LoRA teacher checkpoints")
    parser.add_argument("--actions", type=str, required=True,
                        help='7 comma-separated z2/z7 pairs, e.g. "0.3 0.0, 0.5 0.1, ..."')
    parser.add_argument("--checkpoints", type=str, nargs="*", default=None,
                        help="Explicit .pt checkpoint paths")
    parser.add_argument("--versions", type=str, nargs="*", default=None,
                        help="Version names to evaluate (e.g. v7 v8 v12)")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all registered versions")
    parser.add_argument("--output_dir", type=str, default="eval/eval_output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="configs/causal_lora_diffusion_teacher.yaml")
    parser.add_argument("--ride_index", type=int, default=0,
                        help="Which ride from the manifest to use as context (default: 0)")
    parser.add_argument("--encoded_root", type=str, default=None,
                        help="Override encoded_root from config")
    parser.add_argument("--caption_root", type=str, default=None,
                        help="Override caption_root from config")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    chunk_actions = parse_action_pairs(args.actions)
    log.info("Chunk actions (7 × z2,z7): %s", chunk_actions[0].tolist())

    # Resolve checkpoint list
    eval_list: List[Tuple[str, dict, str]] = []
    if args.checkpoints:
        for cp in args.checkpoints:
            matched = False
            for vname, vinfo in CHECKPOINT_REGISTRY.items():
                if vinfo["logdir"] in cp:
                    eval_list.append((cp, vinfo, vinfo["label"]))
                    matched = True
                    break
            if not matched:
                eval_list.append((cp, {
                    "has_critic": True, "has_adaln": True, "has_action_tokens": True,
                    "critic_base_ch": 128, "critic_res_blocks": 4,
                    "label": Path(cp).stem,
                }, Path(cp).stem))
    else:
        versions = list(CHECKPOINT_REGISTRY.keys()) if args.all else (args.versions or [])
        if not versions:
            parser.error("Specify --checkpoints, --versions, or --all")
        for vname in versions:
            if vname not in CHECKPOINT_REGISTRY:
                log.warning("Unknown version '%s', skipping", vname)
                continue
            vinfo = CHECKPOINT_REGISTRY[vname]
            ckpt = latest_checkpoint(vinfo["logdir"])
            if ckpt is None:
                log.warning("No checkpoint for %s in %s", vname, vinfo["logdir"])
                continue
            eval_list.append((str(ckpt), vinfo, vinfo["label"]))

    if not eval_list:
        log.error("No checkpoints to evaluate!")
        sys.exit(1)

    log.info("Will evaluate %d checkpoint(s)", len(eval_list))
    for i, (cp, _, lbl) in enumerate(eval_list):
        log.info("  [%d] %s — %s", i, lbl, cp)

    pipeline = EvalPipeline(device)
    pipeline.build(args.config)

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)
    encoded_root = args.encoded_root or cfg.get("encoded_root", "/projects/u6ej/fbots/frodobots_encoded")
    caption_root = args.caption_root or cfg.get("caption_root", "/projects/u6ej/fbots/frodobots_captions/train")

    log.info("Loading context latents from dataset...")
    context_latents, prompt_embeds = pipeline.load_context_latents(
        encoded_root, caption_root, ride_index=args.ride_index)

    frame_actions = expand_chunks_to_frames(chunk_actions)
    z_frame = frame_actions.to(device, dtype=pipeline.dtype)

    noise_state = torch.get_rng_state()
    cuda_noise_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

    results_summary = []

    for ckpt_path, vinfo, label in eval_list:
        log.info("=" * 70)
        log.info("Evaluating: %s", label)
        meta = pipeline.load_checkpoint(ckpt_path, vinfo)

        use_adaln = vinfo.get("has_adaln", False)
        use_tokens = vinfo.get("has_action_tokens", False)
        conditional = pipeline.build_conditional(prompt_embeds, z_frame, use_adaln, use_tokens)

        # Deterministic noise
        torch.set_rng_state(noise_state)
        if cuda_noise_state is not None:
            torch.cuda.set_rng_state(cuda_noise_state)

        log.info("  Generating (%d ODE steps)...", EVAL_STEPS)
        t0 = time.time()
        latents = pipeline.generate(conditional, context_latents)
        gen_time = time.time() - t0
        log.info("  Generation took %.1fs", gen_time)

        log.info("  Decoding to pixels...")
        video_np = pipeline.decode_latents(latents)

        log.info("  Computing teacher visuals (CoTracker → ss_vae)...")
        motion, teacher_z_8d = pipeline.compute_teacher_visuals(latents)
        n_chunks = teacher_z_8d.shape[1]
        teacher_z2z7 = teacher_z_8d[:, :, CRITIC_ACTION_DIMS]

        critic_z2z7 = None
        if pipeline.action_critic is not None:
            log.info("  Running action critic...")
            critic_pred = pipeline.run_critic(latents, chunk_actions.to(device))
            if critic_pred is not None:
                critic_z2z7 = critic_pred[:, :, CRITIC_ACTION_DIMS]

        target_z = chunk_actions[:, :n_chunks].to(device)

        log.info("  Annotating video...")
        annotated = annotate_video(
            video_np, teacher_z2z7, critic_z2z7, target_z, motion, label,
        )

        safe = label.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
        out_ann = os.path.join(args.output_dir, f"{safe}_step{meta['step']}_annotated.mp4")
        out_raw = os.path.join(args.output_dir, f"{safe}_step{meta['step']}_raw.mp4")
        frames_to_mp4(annotated, out_ann)
        frames_to_mp4(video_np, out_raw)
        log.info("  Saved: %s", out_ann)
        log.info("  Saved: %s", out_raw)

        # Summary metrics
        teacher_z2_mean = teacher_z2z7[0, :, 0].mean().item()
        teacher_z7_mean = teacher_z2z7[0, :, 1].mean().item()
        entry = {
            "version": label, "step": meta["step"],
            "teacher_z2_mean": f"{teacher_z2_mean:+.4f}",
            "teacher_z7_mean": f"{teacher_z7_mean:+.4f}",
            "gen_time_s": f"{gen_time:.1f}",
        }
        if critic_z2z7 is not None:
            c_z2_mse = F.mse_loss(critic_z2z7[:, :, 0], teacher_z2z7[:, :, 0]).item()
            c_z7_mse = F.mse_loss(critic_z2z7[:, :, 1], teacher_z2z7[:, :, 1]).item()
            entry["critic_z2_mse"] = f"{c_z2_mse:.6f}"
            entry["critic_z7_mse"] = f"{c_z7_mse:.6f}"
        results_summary.append(entry)

        torch.cuda.empty_cache()

    log.info("=" * 70)
    log.info("SUMMARY")
    log.info("-" * 70)
    for r in results_summary:
        log.info("  %s", r)

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    log.info("Summary written to %s", summary_path)
    log.info("All done. Results in: %s", args.output_dir)


if __name__ == "__main__":
    main()
