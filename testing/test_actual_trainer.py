#!/usr/bin/env python3
"""End-to-end training pipeline test with pretrained LoRA cross-loading.

Loads a pretrained bidirectional LoRA into the causal streaming model,
generates a BEFORE video via 48-step ODE denoising, trains 3 steps with
action conditioning on non-overlapping windows (stride=21), generates an
AFTER video, and saves a side-by-side MP4 comparison + loss curve PNG.

Usage:
    python testing/test_actual_trainer.py --device cuda:0
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import cv2

# ===================================================================
# Helpers
# ===================================================================

def _save_mp4(frames: np.ndarray, path: Path, fps: float) -> int:
    h, w = frames.shape[1], frames.shape[2]
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    proc.stdin.write(frames.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode == 0:
        logging.info("Saved %s", path)
    else:
        logging.error("ffmpeg failed for %s", path)
    return proc.returncode


def _chunk_actions(per_frame: torch.Tensor, chunk_frames: int) -> torch.Tensor:
    """Mean-pool ``[B, F, D]`` into ``[B, n_chunks, D]`` with exact chunk size."""
    B, F_len, D = per_frame.shape
    n_chunks = F_len // chunk_frames
    trimmed = per_frame[:, :n_chunks * chunk_frames]
    return trimmed.reshape(B, n_chunks, chunk_frames, D).mean(dim=2)


def _reduce_to_segments(per_frame: torch.Tensor, n_seg: int) -> torch.Tensor:
    """Mean-pool ``[F, D]`` into ``[n_seg, D]``."""
    F_len = per_frame.shape[0]
    seg_size = max(1, F_len // n_seg)
    segs = []
    for i in range(n_seg):
        s = i * seg_size
        e = min(s + seg_size, F_len)
        segs.append(per_frame[s:e].mean(dim=0))
    return torch.stack(segs)


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
    """Draw teacher/critic/target action bars for z2 and z7."""
    h, w = frame.shape[:2]
    out = frame.copy()
    panel_w = 220
    panel_x0 = w - panel_w
    out[:, panel_x0:] = (out[:, panel_x0:].astype(np.float32) * 0.22).astype(np.uint8)

    def draw_bar(y_mid: int, value: float, color: tuple[int, int, int], label: str):
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

    cv2.putText(
        out, title, (panel_x0 + 8, 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        out, f"frame={frame_idx} seg={seg_idx}", (panel_x0 + 8, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 200), 1, cv2.LINE_AA,
    )
    cv2.putText(
        out, "teacher(g) critic(o) target(b)", (panel_x0 + 8, 48),
        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (180, 180, 180), 1, cv2.LINE_AA,
    )

    rows = [
        ("z2", float(teacher_z[0]), float(critic_z[0]), float(target_z[0]), 82),
        ("z7", float(teacher_z[1]), float(critic_z[1]), float(target_z[1]), 140),
    ]
    for z_name, teacher_val, critic_val, target_val, y0 in rows:
        cv2.putText(
            out, z_name, (panel_x0 + 8, y0 - 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 120), 1, cv2.LINE_AA,
        )
        draw_bar(y0, teacher_val, (80, 220, 80), "t")
        draw_bar(y0 + 16, critic_val, (0, 170, 255), "c")
        draw_bar(y0 + 32, target_val, (80, 80, 220), "y")

    cv2.putText(
        out, f"teacher_r {teacher_reward:+.4f}", (panel_x0 + 8, h - 34),
        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (100, 255, 100), 1, cv2.LINE_AA,
    )
    cv2.putText(
        out, f"critic_r  {critic_reward:+.4f}", (panel_x0 + 8, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (100, 200, 255), 1, cv2.LINE_AA,
    )
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
    """Overlay evaluator motion + teacher/critic/target action summaries.

    ``target_action_z`` should already be chunked to ``[B, n_chunks, D]``.
    """
    teacher_np = teacher_z[0].cpu().numpy()
    teacher_reward_np = teacher_reward[0].squeeze(-1).cpu().numpy()
    critic_np = critic_z[0].cpu().numpy()
    critic_reward_np = critic_reward[0].squeeze(-1).cpu().numpy()
    n_seg = teacher_z.shape[1]
    target_seg_np = target_action_z[0, :n_seg].cpu().numpy()
    motion_np = est_motion[0].cpu().numpy()

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
                    cx = int((gx + 0.5) * drawable_w / grid_size)
                    cy = int((gy + 0.5) * h / grid_size)
                    ex = int(cx - dx * 3)
                    ey = int(cy - dy * 3)
                    color = (0, 255, 0) if vis_val >= 0.5 else (0, 200, 255)
                    cv2.arrowedLine(f, (cx, cy), (ex, ey), color, 1, tipLength=0.3)

        f = _draw_triplet_action_overlay(
            f,
            teacher_np[seg_idx],
            critic_np[seg_idx],
            target_seg_np[seg_idx],
            float(teacher_reward_np[seg_idx]),
            float(critic_reward_np[seg_idx]),
            title=title,
            frame_idx=frame_idx,
            seg_idx=seg_idx,
        )
        annotated.append(f)
    return np.stack(annotated)


@torch.no_grad()
def _compute_teacher_targets(
    pred_x0, target_action_z,
    scheduler, frozen_vae, cotracker, ss_vae, ss_scale,
    device, dtype, noise_t=25, critic_dims=None, chunk_frames=3,
):
    """Produce n_chunks teacher targets aligned to the chunkwise critic."""
    from utils.zarr_dataset import _tanh_squash

    if critic_dims is None:
        critic_dims = [2, 7]

    B, F, C, H, W = pred_x0.shape
    n_chunks = F // chunk_frames

    dummy = pred_x0[:, 0:1]
    lat_wd = torch.cat([dummy, pred_x0], dim=1)

    noise = torch.randn_like(lat_wd.flatten(0, 1))
    t_f = torch.full((noise.shape[0],), noise_t, dtype=torch.long, device=device)
    noisy = scheduler.add_noise(lat_wd.flatten(0, 1), noise, t_f).unflatten(0, (B, F + 1))

    pixels = frozen_vae.decode_to_pixel(noisy.to(dtype=dtype))[:, 1:, ...]
    video = (255.0 * 0.5 * (pixels.float() + 1.0)).clamp(0, 255)

    grid_size = 10
    output_chunk_size = 12
    N = grid_size ** 2

    all_z = []
    all_r = []
    for b in range(B):
        vid = video[b].unsqueeze(0)
        T_total = vid.shape[1]
        mws = []
        for cs in range(0, T_total, 48):
            ce = min(cs + 48, T_total)
            ch = vid[:, cs:ce]
            n_out = ch.shape[1] // output_chunk_size
            if n_out == 0:
                continue
            used = n_out * output_chunk_size
            ch = ch[:, :used].clone()
            with torch.amp.autocast(device_type="cuda", enabled=True):
                tracks, vis = cotracker(ch, grid_size=grid_size)
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
            all_z.append(torch.zeros(n_chunks, len(critic_dims), device=device))
            all_r.append(torch.zeros(n_chunks, 1, device=device))
            continue

        em = torch.cat(mws, dim=0)
        raw_n = em.shape[0]
        xy = em[:, :, :2].reshape(raw_n, 10, 10, 2)
        x_in = xy.permute(0, 3, 1, 2).float() / ss_scale
        mu, _ = ss_vae.encoder(x_in.to(device))
        z8 = _tanh_squash(mu.squeeze(-1).squeeze(-1))
        gz = z8[:, critic_dims]

        gz_chunked = _reduce_to_segments(gz, n_chunks)
        target_chunked = _reduce_to_segments(target_action_z[b], n_chunks)
        reward = -((gz_chunked - target_chunked) ** 2).mean(dim=-1, keepdim=True)
        all_z.append(gz_chunked)
        all_r.append(reward)

    tz = torch.stack(all_z)
    tr = torch.stack(all_r)
    return tz.detach(), tr.detach()


@torch.no_grad()
def _compute_teacher_visuals(
    pred_x0, target_action_z,
    scheduler, frozen_vae, cotracker, ss_vae, ss_scale,
    device, dtype, noise_t=25, critic_dims=None, chunk_frames=3,
):
    """Return evaluator motion, z2/z7, and reward aligned to n_chunks."""
    from utils.zarr_dataset import _tanh_squash

    if critic_dims is None:
        critic_dims = [2, 7]

    B, F, _, _, _ = pred_x0.shape
    n_chunks = F // chunk_frames

    dummy = pred_x0[:, 0:1]
    lat_wd = torch.cat([dummy, pred_x0], dim=1)

    noise = torch.randn_like(lat_wd.flatten(0, 1))
    t_f = torch.full((noise.shape[0],), noise_t, dtype=torch.long, device=device)
    noisy = scheduler.add_noise(lat_wd.flatten(0, 1), noise, t_f).unflatten(0, (B, F + 1))

    pixels = frozen_vae.decode_to_pixel(noisy.to(dtype=dtype))[:, 1:, ...]
    video = (255.0 * 0.5 * (pixels.float() + 1.0)).clamp(0, 255)

    grid_size = 10
    output_chunk_size = 12
    N = grid_size ** 2
    all_motion = []
    all_z = []
    all_r = []

    for b in range(B):
        vid = video[b].unsqueeze(0)
        T_total = vid.shape[1]
        mws = []
        for cs in range(0, T_total, 48):
            ce = min(cs + 48, T_total)
            ch = vid[:, cs:ce]
            n_out = ch.shape[1] // output_chunk_size
            if n_out == 0:
                continue
            used = n_out * output_chunk_size
            ch = ch[:, :used].clone()
            with torch.amp.autocast(device_type="cuda", enabled=True):
                tracks, vis = cotracker(ch, grid_size=grid_size)
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
            all_motion.append(torch.zeros(n_chunks, N, 3, device=device))
            all_z.append(torch.zeros(n_chunks, len(critic_dims), device=device))
            all_r.append(torch.zeros(n_chunks, 1, device=device))
            continue

        est_motion = torch.cat(mws, dim=0)
        raw_n = est_motion.shape[0]
        xy = est_motion[:, :, :2].reshape(raw_n, 10, 10, 2)
        x_in = xy.permute(0, 3, 1, 2).float() / ss_scale
        mu, _ = ss_vae.encoder(x_in.to(device))
        z8 = _tanh_squash(mu.squeeze(-1).squeeze(-1))
        gz = z8[:, critic_dims]

        gz_chunked = _reduce_to_segments(gz, n_chunks)
        motion_chunked = _reduce_to_segments(
            est_motion.reshape(raw_n, -1), n_chunks,
        ).reshape(n_chunks, N, 3)
        target_chunked = _reduce_to_segments(target_action_z[b], n_chunks)
        reward = -((gz_chunked - target_chunked) ** 2).mean(dim=-1, keepdim=True)

        all_motion.append(motion_chunked)
        all_z.append(gz_chunked)
        all_r.append(reward)

    motion = torch.stack(all_motion)
    tz = torch.stack(all_z)
    tr = torch.stack(all_r)
    return motion.detach(), tz.detach(), tr.detach()


# ===================================================================
# Generation (48-step ODE denoising)
# ===================================================================

@torch.no_grad()
def _generate(
    wrapper, conditional, context_latents,
    num_frames, device, dtype,
    inference_steps=48,
):
    """Multi-step ODE denoising for a non-distilled causal model."""
    from utils.scheduler import FlowMatchScheduler

    scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(num_inference_steps=inference_steps, denoising_strength=1.0)
    scheduler.sigmas = scheduler.sigmas.to(device)

    B = context_latents.shape[0]
    C, H, W = context_latents.shape[2], context_latents.shape[3], context_latents.shape[4]
    latents = torch.randn(
        [B, num_frames, C, H, W], dtype=torch.float32, device=device,
    )

    for t in scheduler.timesteps:
        timestep = t * torch.ones([B, num_frames], device=device, dtype=torch.float32)

        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            flow_pred, pred_x0 = wrapper(
                latents, conditional, timestep,
                clean_x=context_latents, aug_t=None,
            )

        latents = scheduler.step(
            flow_pred.flatten(0, 1),
            timestep.flatten(0, 1),
            latents.flatten(0, 1),
        ).unflatten(dim=0, sizes=flow_pred.shape[:2])

    return latents


@torch.no_grad()
def _decode_latents(frozen_vae, latents, dtype):
    """Decode latents to pixel-space uint8 numpy [T, H, W, 3]."""
    dummy = latents[:, 0:1]
    lat_wd = torch.cat([dummy, latents], dim=1)
    pixels = frozen_vae.decode_to_pixel(lat_wd.to(dtype=dtype))[:, 1:, ...]
    video = (0.5 * (pixels.float() + 1.0)).clamp(0, 1)
    vid_np = (video[0].cpu().numpy() * 255).astype(np.uint8)
    if vid_np.shape[-1] != 3:
        vid_np = vid_np.transpose(0, 2, 3, 1)
    return vid_np


# ===================================================================
# Main
# ===================================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="End-to-end training test with pretrained LoRA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--lora_ckpt",
        default="../frodobots/pretrained_weights/retrain_no_gan_256/logs/diffusion_lora_step0018600.pt",
    )
    parser.add_argument("--encoded_root", default="/projects/u6ej/fbots/frodobots_encoded")
    parser.add_argument("--caption_root", default="/projects/u6ej/fbots/frodobots_captions/train")
    parser.add_argument("--motion_root", default="/projects/u6ej/fbots/frodobots_motion")
    parser.add_argument("--ss_vae_checkpoint", default="action_query/checkpoints/ss_vae_8free.pt")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--inference_steps", type=int, default=48)
    parser.add_argument("--output_dir", default="testing/outputs")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        logging.error("This test requires a GPU.")
        sys.exit(1)

    device = torch.device(args.device)
    dtype = torch.bfloat16
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cf = 3
    num_frame_per_block = 3
    window_size = 21
    window_total = window_size + cf  # 24 frames per window
    num_frames = window_size          # 21 noisy/clean frames
    num_steps = args.steps
    total_frames_needed = window_total + (num_steps - 1) * window_size
    context_shift = cf // num_frame_per_block  # 1

    logging.info("Config: cf=%d, window_size=%d, window_total=%d, num_frames=%d, "
                 "steps=%d, total_frames=%d, context_shift=%d",
                 cf, window_size, window_total, num_frames, num_steps,
                 total_frames_needed, context_shift)

    # ==================================================================
    # 1. Load data
    # ==================================================================
    logging.info("Loading dataset ...")
    from utils.zarr_dataset import ZarrRideDataset

    dataset = ZarrRideDataset(
        encoded_root=args.encoded_root,
        caption_root=args.caption_root,
        motion_root=args.motion_root,
        ss_vae_checkpoint=args.ss_vae_checkpoint,
        min_ride_frames=total_frames_needed,
        device="cpu",
        max_rides=4,
    )

    ride = dataset[0]
    zarr_path = ride["zarr_path"]
    n_lat = int(ride["n_latent_frames"])
    actual_frames = min(total_frames_needed, n_lat)

    chunk = ZarrRideDataset.load_latent_chunk(zarr_path, 0, actual_frames)
    all_latents = chunk.unsqueeze(0).to(device=device, dtype=torch.float32)
    prompt_embeds = ride["prompt_embeds"].unsqueeze(0).to(device=device, dtype=dtype)

    all_z_actions = dataset.encode_z_actions_window(
        zarr_path, n_lat, 0, actual_frames,
    ).unsqueeze(0).to(device=device, dtype=dtype)

    logging.info("Data: all_latents=%s  all_z_actions=%s  (%d steps, stride=%d)",
                 tuple(all_latents.shape), tuple(all_z_actions.shape),
                 num_steps, window_size)

    # ==================================================================
    # 2. Build model
    # ==================================================================
    logging.info("Building CausalWanModel + action patches + LoRA ...")
    import peft
    from peft import LoraConfig, set_peft_model_state_dict

    from utils.wan_wrapper import WanDiffusionWrapper, WanVAEWrapper
    from model.action_model_patch import apply_action_patches
    from model.action_modulation import ActionModulationProjection, ActionTokenProjection
    from model.action_critic import ActionCritic
    from action_query.ss_vae_model import load_ss_vae

    wrapper = WanDiffusionWrapper(
        model_name="Wan2.1-T2V-1.3B",
        is_causal=True,
        timestep_shift=5.0,
        local_attn_size=-1,
        sink_size=0,
    )
    wrapper.enable_gradient_checkpointing()
    wrapper.model.num_frame_per_block = num_frame_per_block
    wrapper.model.context_shift = context_shift

    apply_action_patches(wrapper)

    model_dim = getattr(wrapper.model, "dim", 2048)
    action_dim = 2
    action_dims = [2, 7]

    action_projection = ActionModulationProjection(
        action_dim=action_dim, activation="silu",
        hidden_dim=model_dim, num_frames=1, zero_init=True,
    ).to(device)

    action_token_projection = ActionTokenProjection(
        action_dim=action_dim, activation="silu",
        hidden_dim=model_dim, zero_init=True,
    ).to(device)
    wrapper.model.action_tokens_per_frame = 1
    wrapper.adjust_seq_len_for_action_tokens(num_frames=num_frames, action_per_frame=1)

    # LoRA -- match rank/alpha from causal_lora_diffusion_teacher.yaml
    target_modules = set()
    for mn, mod in wrapper.model.named_modules():
        if mod.__class__.__name__ in {"WanAttentionBlock", "CausalWanAttentionBlock"}:
            for fn, sub in mod.named_modules(prefix=mn):
                if isinstance(sub, torch.nn.Linear):
                    target_modules.add(fn)
    lora_config = LoraConfig(
        r=256, lora_alpha=256, lora_dropout=0.0,
        target_modules=sorted(target_modules),
        init_lora_weights="gaussian",
    )
    wrapper.model = peft.get_peft_model(wrapper.model, lora_config)
    causal_model = wrapper.model.base_model.model  # underlying CausalWanModel

    # Cross-load pretrained bidirectional LoRA
    lora_ckpt_path = Path(args.lora_ckpt)
    if lora_ckpt_path.exists():
        logging.info("Cross-loading LoRA from %s", lora_ckpt_path)
        ckpt = torch.load(str(lora_ckpt_path), map_location="cpu")
        lora_state = ckpt.get("lora") or ckpt.get("generator_lora")
        if lora_state is None:
            logging.warning("Checkpoint has no 'lora' key. Keys: %s. Starting LoRA from scratch.",
                            list(ckpt.keys()))
        else:
            set_peft_model_state_dict(wrapper.model, lora_state)
            logging.info("Loaded %d LoRA tensors from pretrained checkpoint", len(lora_state))
    else:
        logging.warning("LoRA checkpoint not found at %s, starting from scratch", lora_ckpt_path)

    wrapper.to(device=device, dtype=dtype)
    wrapper.train()

    # Scheduler for training
    train_scheduler = wrapper.get_scheduler()
    train_scheduler.set_timesteps(1000, training=True)

    # ==================================================================
    # 3. Build action critic + frozen evaluators
    # ==================================================================
    logging.info("Building ActionCritic + frozen evaluators ...")
    critic = ActionCritic(latent_channels=16, z_dim=2, base_channels=64, num_res_blocks=3, chunk_frames=num_frame_per_block)
    critic.to(device=device, dtype=dtype)
    critic.train()

    frozen_vae = WanVAEWrapper().to(device=device, dtype=dtype).eval()
    frozen_vae.requires_grad_(False)

    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    cotracker.eval()
    for p in cotracker.parameters():
        p.requires_grad_(False)

    ss_vae, ss_scale = load_ss_vae(args.ss_vae_checkpoint, device=str(device))
    ss_vae.eval()
    ss_vae.requires_grad_(False)

    # ==================================================================
    # 4. Build optimizer
    # ==================================================================
    params = [p for p in wrapper.parameters() if p.requires_grad]
    params += list(action_projection.parameters())
    params += list(action_token_projection.parameters())
    params += list(critic.parameters())
    optimizer = torch.optim.AdamW(params, lr=5e-5, weight_decay=0.01)

    n_trainable = sum(p.numel() for p in params)
    logging.info("Trainable params: %d  (%d tensors)", n_trainable, len(params))

    # ==================================================================
    # 5. Generate BEFORE video
    # ==================================================================
    logging.info("=" * 60)
    logging.info("GENERATING BEFORE VIDEO (%d ODE steps)", args.inference_steps)
    logging.info("=" * 60)

    wrapper.eval()
    causal_model.block_mask = None  # force mask rebuild for eval

    gen_context = all_latents[:, :num_frames]
    gen_z_actions = all_z_actions[:, :window_total]
    gen_z_sliced = gen_z_actions[..., action_dims]
    gen_z_noisy = gen_z_sliced[:, cf:]
    gen_z_clean = gen_z_sliced[:, :num_frames]
    gen_target_action_z = gen_z_actions[:, cf:][..., action_dims][:, :num_frames]

    gen_action_mod = action_projection(gen_z_noisy, num_frames=num_frames)
    gen_action_mod_clean = action_projection(gen_z_clean, num_frames=num_frames)
    gen_action_tok = action_token_projection(gen_z_noisy)
    gen_action_tok_clean = action_token_projection(gen_z_clean)

    gen_conditional = {
        "prompt_embeds": prompt_embeds,
        "_action_modulation": gen_action_mod,
        "_action_modulation_clean": gen_action_mod_clean,
        "_action_tokens": gen_action_tok,
        "_action_tokens_clean": gen_action_tok_clean,
    }

    t0 = time.perf_counter()
    before_latents = _generate(
        wrapper, gen_conditional, gen_context,
        num_frames, device, dtype,
        inference_steps=args.inference_steps,
    )
    t_before = time.perf_counter() - t0
    logging.info("BEFORE generation: %.1fs", t_before)

    before_vid = _decode_latents(frozen_vae, before_latents, dtype)
    logging.info("BEFORE video shape: %s", before_vid.shape)

    logging.info("Computing BEFORE evaluator/critic overlays ...")
    before_motion, before_teacher_z, before_teacher_r = _compute_teacher_visuals(
        before_latents, gen_target_action_z,
        train_scheduler, frozen_vae, cotracker, ss_vae, ss_scale,
        device, dtype, noise_t=25, critic_dims=action_dims, chunk_frames=num_frame_per_block,
    )
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=dtype):
        before_critic_z, before_critic_r = critic(before_latents)
    n_chunks_before = before_teacher_z.shape[1]
    before_critic_z = before_critic_z[:, :n_chunks_before].float().detach()
    before_critic_r = before_critic_r[:, :n_chunks_before].float().detach()
    before_target_chunk = _chunk_actions(gen_target_action_z.float(), num_frame_per_block)[:, :n_chunks_before]
    before_vid_annot = _annotate_action_video(
        before_vid,
        before_motion,
        before_teacher_z,
        before_teacher_r,
        before_critic_z,
        before_critic_r,
        before_target_chunk,
        title="BEFORE teacher@t25 / critic / target",
    )

    # ==================================================================
    # 6. Train
    # ==================================================================
    logging.info("=" * 60)
    logging.info("TRAINING %d STEPS (stride=%d)", num_steps, window_size)
    logging.info("=" * 60)

    wrapper.train()
    causal_model.block_mask = None  # force mask rebuild for training

    timesteps = train_scheduler.timesteps.to(device)
    bsz = 1

    loss_history = {
        "flow": [], "critic_z": [], "critic_r": [], "critic_total": [],
        "gen_z": [], "gen_r": [], "gen_total": [], "total": [],
    }

    for step_i in range(1, num_steps + 1):
        t_step_start = time.perf_counter()
        win_start = (step_i - 1) * window_size
        win_end = win_start + window_total
        logging.info("-" * 60)
        logging.info("STEP %d / %d  (frames %d-%d)", step_i, num_steps, win_start, win_end)
        logging.info("-" * 60)

        full_latents = all_latents[:, win_start:win_end]
        z_actions_full = all_z_actions[:, win_start:win_end]
        context_latents = full_latents[:, :num_frames]
        target_latents = full_latents[:, cf:]

        z_actions_sliced = z_actions_full[..., action_dims]
        z_noisy = z_actions_sliced[:, cf:]
        z_clean = z_actions_sliced[:, :num_frames]
        target_action_z = z_actions_full[:, cf:][..., action_dims][:, :num_frames]

        optimizer.zero_grad(set_to_none=True)

        action_modulation = action_projection(z_noisy, num_frames=num_frames)
        action_modulation_clean = action_projection(z_clean, num_frames=num_frames)
        action_tokens = action_token_projection(z_noisy)
        action_tokens_clean = action_token_projection(z_clean)

        conditional = {
            "prompt_embeds": prompt_embeds,
            "_action_modulation": action_modulation,
            "_action_modulation_clean": action_modulation_clean,
            "_action_tokens": action_tokens,
            "_action_tokens_clean": action_tokens_clean,
        }

        t_idx = torch.randint(0, len(timesteps), (bsz, num_frames), device=device)
        t_idx = t_idx.reshape(bsz, -1, num_frame_per_block)
        t_idx[:, :, 1:] = t_idx[:, :, 0:1]
        t_idx = t_idx.reshape(bsz, num_frames)
        ts = timesteps[t_idx]

        noise = torch.randn_like(target_latents)
        noisy_latents = train_scheduler.add_noise(
            target_latents.flatten(0, 1), noise.flatten(0, 1), ts.flatten(0, 1),
        ).view_as(target_latents)
        training_target = train_scheduler.training_target(
            target_latents.flatten(0, 1), noise.flatten(0, 1), ts.flatten(0, 1),
        ).view_as(target_latents)

        # Forward
        t0 = time.perf_counter()
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            flow_pred, pred_x0 = wrapper(
                noisy_latents, conditional, ts,
                clean_x=context_latents, aug_t=None,
            )
            flow_loss_raw = F.mse_loss(flow_pred.float(), training_target.float(), reduction="none")
            flow_loss_raw = flow_loss_raw.mean(dim=(2, 3, 4))
            weights = train_scheduler.training_weight(ts.flatten(0, 1)).view(bsz, num_frames)
            flow_loss = (flow_loss_raw * weights).mean()
        t_fwd = time.perf_counter() - t0

        # Teacher targets
        t0 = time.perf_counter()
        teacher_z, teacher_reward = _compute_teacher_targets(
            pred_x0.detach(), target_action_z,
            train_scheduler, frozen_vae, cotracker, ss_vae, ss_scale,
            device, dtype, noise_t=25, critic_dims=action_dims, chunk_frames=num_frame_per_block,
        )
        t_teacher = time.perf_counter() - t0

        # Critic + generator losses (chunkwise -- no temporal pooling needed)
        n_chunks = teacher_z.shape[1]
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            pred_z_c, pred_r_c = critic(pred_x0.detach())
            pred_z_c = pred_z_c[:, :n_chunks].float()
            pred_r_c = pred_r_c[:, :n_chunks].float()

            critic_z_loss = F.mse_loss(pred_z_c, teacher_z.float())
            critic_r_loss = F.mse_loss(pred_r_c, teacher_reward.float())
            critic_loss = critic_z_loss + 0.1 * critic_r_loss

            critic.requires_grad_(False)
            gen_z, gen_r = critic(pred_x0)
            gen_z = gen_z[:, :n_chunks].float()
            gen_r = gen_r[:, :n_chunks].float()
            target_chunk = _chunk_actions(target_action_z, num_frame_per_block)[:, :n_chunks].float()

            gen_z_loss = F.mse_loss(gen_z, target_chunk)
            gen_reward_loss = -gen_r.mean()
            gen_action_loss = 1.0 * gen_z_loss + 0.1 * gen_reward_loss
            critic.requires_grad_(True)

            total_loss = flow_loss + critic_loss + gen_action_loss

        # Backward
        t0 = time.perf_counter()
        total_loss.backward()
        t_bwd = time.perf_counter() - t0

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        t_step = time.perf_counter() - t_step_start
        logging.info("  timing: fwd=%.2fs  teacher=%.2fs  bwd=%.2fs  total=%.2fs",
                     t_fwd, t_teacher, t_bwd, t_step)
        logging.info("  flow_loss       = %.6f", flow_loss.item())
        logging.info("  critic_z_loss   = %.6f   critic_r_loss = %.6f   critic_total = %.6f",
                     critic_z_loss.item(), critic_r_loss.item(), critic_loss.item())
        logging.info("  gen_z_loss      = %.6f   gen_reward    = %.6f   gen_total    = %.6f",
                     gen_z_loss.item(), gen_reward_loss.item(), gen_action_loss.item())
        logging.info("  TOTAL           = %.6f", total_loss.item())

        loss_history["flow"].append(flow_loss.item())
        loss_history["critic_z"].append(critic_z_loss.item())
        loss_history["critic_r"].append(critic_r_loss.item())
        loss_history["critic_total"].append(critic_loss.item())
        loss_history["gen_z"].append(gen_z_loss.item())
        loss_history["gen_r"].append(gen_reward_loss.item())
        loss_history["gen_total"].append(gen_action_loss.item())
        loss_history["total"].append(total_loss.item())

    # ==================================================================
    # 7. Generate AFTER video
    # ==================================================================
    logging.info("=" * 60)
    logging.info("GENERATING AFTER VIDEO (%d ODE steps)", args.inference_steps)
    logging.info("=" * 60)

    wrapper.eval()
    causal_model.block_mask = None

    gen_action_mod = action_projection(gen_z_noisy, num_frames=num_frames)
    gen_action_mod_clean = action_projection(gen_z_clean, num_frames=num_frames)
    gen_action_tok = action_token_projection(gen_z_noisy)
    gen_action_tok_clean = action_token_projection(gen_z_clean)

    gen_conditional = {
        "prompt_embeds": prompt_embeds,
        "_action_modulation": gen_action_mod,
        "_action_modulation_clean": gen_action_mod_clean,
        "_action_tokens": gen_action_tok,
        "_action_tokens_clean": gen_action_tok_clean,
    }

    t0 = time.perf_counter()
    after_latents = _generate(
        wrapper, gen_conditional, gen_context,
        num_frames, device, dtype,
        inference_steps=args.inference_steps,
    )
    t_after = time.perf_counter() - t0
    logging.info("AFTER generation: %.1fs", t_after)

    after_vid = _decode_latents(frozen_vae, after_latents, dtype)
    logging.info("AFTER video shape: %s", after_vid.shape)

    logging.info("Computing AFTER evaluator/critic overlays ...")
    after_motion, after_teacher_z, after_teacher_r = _compute_teacher_visuals(
        after_latents, gen_target_action_z,
        train_scheduler, frozen_vae, cotracker, ss_vae, ss_scale,
        device, dtype, noise_t=25, critic_dims=action_dims, chunk_frames=num_frame_per_block,
    )
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=dtype):
        after_critic_z, after_critic_r = critic(after_latents)
    n_chunks_after = after_teacher_z.shape[1]
    after_critic_z = after_critic_z[:, :n_chunks_after].float().detach()
    after_critic_r = after_critic_r[:, :n_chunks_after].float().detach()
    after_target_chunk = _chunk_actions(gen_target_action_z.float(), num_frame_per_block)[:, :n_chunks_after]
    after_vid_annot = _annotate_action_video(
        after_vid,
        after_motion,
        after_teacher_z,
        after_teacher_r,
        after_critic_z,
        after_critic_r,
        after_target_chunk,
        title="AFTER teacher@t25 / critic / target",
    )

    # ==================================================================
    # 8. Save side-by-side MP4
    # ==================================================================
    logging.info("Saving comparison video ...")
    n_frames = min(before_vid_annot.shape[0], after_vid_annot.shape[0])
    bv = before_vid_annot[:n_frames]
    av = after_vid_annot[:n_frames]

    if bv.shape[1:3] != av.shape[1:3]:
        h, w = bv.shape[1], bv.shape[2]
        av_resized = np.stack([cv2.resize(f, (w, h)) for f in av])
        av = av_resized

    h, w = bv.shape[1], bv.shape[2]
    label_h = 40
    side_by_side = np.zeros((n_frames, h + label_h, w * 2, 3), dtype=np.uint8)
    for i in range(n_frames):
        canvas = np.zeros((h + label_h, w * 2, 3), dtype=np.uint8)
        canvas[label_h:, :w] = bv[i]
        canvas[label_h:, w:] = av[i]
        cv2.putText(canvas, "BEFORE", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "AFTER", (w + 10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        side_by_side[i] = canvas

    mp4_path = out_dir / "test_actual_trainer_comparison.mp4"
    _save_mp4(side_by_side, mp4_path, fps=8.0)

    # ==================================================================
    # 9. Save loss curve
    # ==================================================================
    logging.info("Saving loss curve ...")
    steps_x = list(range(1, num_steps + 1))

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    for key in ["flow", "critic_total", "gen_total", "total"]:
        axes[0].plot(steps_x, loss_history[key], marker="o", label=key)
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Top-level losses")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for key in ["critic_z", "critic_r", "gen_z", "gen_r"]:
        axes[1].plot(steps_x, loss_history[key], marker="o", label=key)
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Loss / reward term")
    axes[1].set_title("Action-related components")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    png_path = out_dir / "test_actual_trainer_losses.png"
    fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved %s", png_path)

    logging.info("=" * 60)
    logging.info("DONE. Outputs in %s", out_dir)
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
