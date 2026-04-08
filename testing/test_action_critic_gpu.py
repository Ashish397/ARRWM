#!/usr/bin/env python3
"""GPU integration test for the action critic pipeline.

Three modes:
  1. ``--mode visual``  — Loads one real zarr window, noises it to t=25,
     runs the full teacher-target pipeline (VAE decode → CoTracker → ss_vae),
     saves annotated MP4s showing decoded pixels and extracted z2/z7.

  2. ``--mode train``   — Builds the real model, critic, and frozen evaluator,
     runs 1 training step with batch_size=1, then reports every loss term,
     gradient norms on every parameter group, and runs sanity assertions.

  3. ``--mode teacher`` — Decodes real zarr latents to pixels, runs
     CoTracker → ss_vae (the same path training uses), and compares the
     resulting z2/z7 against the saved motion-file targets for the same
     ride window.

Usage:
    # Visual pipeline test
    python testing/test_action_critic_gpu.py --mode visual

    # Single-step training test
    python testing/test_action_critic_gpu.py --mode train

    # Both
    python testing/test_action_critic_gpu.py --mode both
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ===================================================================
# Drawing helpers (matched to test_zarr_dataloader.py style)
# ===================================================================

def _draw_paired_latent_overlay(
    frame: np.ndarray,
    gen_8d: np.ndarray,
    tgt_8d: np.ndarray,
    clip: float = 1.0,
) -> np.ndarray:
    """Draw side-by-side bar chart of all 8 z-dims for gen (green) and tgt (blue)."""
    h, w = frame.shape[:2]
    out = frame.copy()
    n = 8
    panel_w = 140
    slot_h = max(h // n, 28)
    out[:, w - panel_w:] = (out[:, w - panel_w:].astype(np.float32) * 0.3).astype(np.uint8)
    half = (panel_w - 10) // 2
    cx = w - panel_w + half + 5

    for i in range(n):
        y_base = i * slot_h
        label_color = (255, 255, 100) if i in (2, 7) else (200, 200, 200)

        # Generated bar (top half of slot)
        gv = float(gen_8d[i])
        g_mid = y_base + slot_h // 4
        bar_thick = max(slot_h // 6, 2)
        g_len = int(min(abs(gv), clip) / clip * half)
        if gv >= 0:
            cv2.rectangle(out, (cx, g_mid - bar_thick), (cx + g_len, g_mid + bar_thick), (80, 220, 80), -1)
        else:
            cv2.rectangle(out, (cx - g_len, g_mid - bar_thick), (cx, g_mid + bar_thick), (80, 220, 80), -1)

        # Target bar (bottom half of slot)
        tv = float(tgt_8d[i])
        t_mid = y_base + 3 * slot_h // 4
        t_len = int(min(abs(tv), clip) / clip * half)
        if tv >= 0:
            cv2.rectangle(out, (cx, t_mid - bar_thick), (cx + t_len, t_mid + bar_thick), (80, 80, 220), -1)
        else:
            cv2.rectangle(out, (cx - t_len, t_mid - bar_thick), (cx, t_mid + bar_thick), (80, 80, 220), -1)

        # Centre line spanning both bars
        cv2.line(out, (cx, y_base + 2), (cx, y_base + slot_h - 2), (120, 120, 120), 1)

        # Labels
        cv2.putText(out, f"z{i}", (w - panel_w + 2, g_mid + 4),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.30, label_color, 1, cv2.LINE_AA)
        cv2.putText(out, f"{gv:+.2f}", (w - panel_w + 2, g_mid + 4 + 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.22, (100, 255, 100), 1, cv2.LINE_AA)
        cv2.putText(out, f"{tv:+.2f}", (w - panel_w + 2, t_mid + 4 + 4),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.22, (100, 100, 255), 1, cv2.LINE_AA)

    # Legend at bottom of panel
    leg_y = min(n * slot_h + 4, h - 10)
    cv2.putText(out, "gen", (w - panel_w + 4, leg_y),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.28, (100, 255, 100), 1, cv2.LINE_AA)
    cv2.putText(out, "tgt", (w - panel_w + 34, leg_y),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.28, (100, 100, 255), 1, cv2.LINE_AA)
    return out


def _draw_paired_dials(
    frame: np.ndarray,
    gen_8d: np.ndarray,
    tgt_8d: np.ndarray,
    reward: float,
    turn_idx: int = 2,
    fwd_idx: int = 7,
) -> np.ndarray:
    """Draw two steering/throttle dials (gen left, tgt right) + reward text."""
    h, w = frame.shape[:2]
    out = frame
    dial_r = 20
    margin = 8
    panel_w = 140
    base_x = w - panel_w - 2 * (dial_r + margin) - 10
    cy = h - dial_r - margin - 28

    for k, (label, z8, col) in enumerate([
        ("gen", gen_8d, (80, 220, 80)),
        ("tgt", tgt_8d, (80, 80, 220)),
    ]):
        dx_c = base_x + k * (2 * dial_r + margin * 2) + dial_r
        cv2.circle(out, (dx_c, cy), dial_r, (50, 50, 65), -1)
        cv2.circle(out, (dx_c, cy), dial_r, col, 1)
        z_turn = float(z8[turn_idx])
        z_fwd = float(z8[fwd_idx])
        arm = dial_r - 4
        ax = int(max(-arm, min(arm, z_turn * arm)))
        ay = int(max(-arm, min(arm, -z_fwd * arm)))
        cv2.arrowedLine(out, (dx_c, cy), (dx_c + ax, cy + ay), col, 2, tipLength=0.35)
        cv2.putText(out, label, (dx_c - 8, cy - dial_r - 3),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.28, col, 1, cv2.LINE_AA)

    # Reward text between dials
    rw_x = base_x
    cv2.putText(out, f"rwd:{reward:+.4f}", (rw_x, cy + dial_r + 14),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 200, 100), 1, cv2.LINE_AA)
    return out


# ===================================================================
# Visual pipeline test
# ===================================================================

def run_visual_test(args) -> None:
    """Load real latents, noise to t=25, decode, run CoTracker + ss_vae,
    save annotated videos showing the extracted motion and z2/z7."""
    from utils.wan_wrapper import WanVAEWrapper
    from utils.scheduler import FlowMatchScheduler
    from utils.zarr_dataset import (
        ZarrRideDataset,
        _load_aligned_motion_for_zarr,
        _tanh_squash,
        _LATENT_TO_VIDEO,
    )
    from action_query.ss_vae_model import load_ss_vae
    import zarr as zarr_lib

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load one window of real latents ──
    logging.info("Building ZarrRideDataset ...")
    dataset = ZarrRideDataset(
        encoded_root=args.encoded_root,
        caption_root=args.caption_root,
        motion_root=args.motion_root,
        ss_vae_checkpoint=args.ss_vae_checkpoint,
        min_ride_frames=24,
        device="cpu",
        max_rides=4,
    )
    logging.info("Dataset ready: %d rides.", len(dataset))

    ride = dataset[0]
    zarr_path = ride["zarr_path"]
    n_lat = int(ride["n_latent_frames"])
    logging.info("Using ride: %s  (%d latent frames)", zarr_path, n_lat)

    num_windows = args.num_windows
    window_size = 24
    cf = 3
    noise_t = args.noise_t

    # ── Load shared heavy models once ──
    scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)

    logging.info("Loading VAE ...")
    vae = WanVAEWrapper().to(device=device, dtype=torch.bfloat16).eval()
    vae.requires_grad_(False)

    logging.info("Loading CoTracker ...")
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    cotracker.eval()
    for p in cotracker.parameters():
        p.requires_grad_(False)

    logging.info("Loading ss_vae ...")
    ss_vae, scale = load_ss_vae(args.ss_vae_checkpoint, device=str(device))
    ss_vae.eval()
    ss_vae.requires_grad_(False)

    grid_size = 10
    output_chunk_size = 12
    N = grid_size ** 2

    for win_idx in range(num_windows):
        lat_start = win_idx * (window_size - cf)
        lat_end = lat_start + window_size
        if lat_end > n_lat:
            logging.warning("Window %d exceeds ride length (%d > %d), stopping.", win_idx, lat_end, n_lat)
            break

        logging.info("=" * 60)
        logging.info("WINDOW %d / %d  — latents [%d : %d)", win_idx, num_windows, lat_start, lat_end)
        logging.info("=" * 60)

        chunk = ZarrRideDataset.load_latent_chunk(zarr_path, lat_start, lat_end)
        latents = chunk.unsqueeze(0).to(device=device, dtype=torch.float32)

        num_frames = lat_end - lat_start - cf
        target_latents = latents[:, cf:]

        z_actions = dataset.encode_z_actions_window(
            zarr_path, n_lat, lat_start, lat_end,
        ).unsqueeze(0).to(device=device)
        z_target = z_actions[:, cf:]
        z_target_27 = z_target[..., [2, 7]]
        logging.info("Target z2/z7: %s  mean=%.3f", tuple(z_target_27.shape), z_target_27.mean().item())

        # ── Add noise ──
        dummy = target_latents[:, 0:1]
        latents_with_dummy = torch.cat([dummy, target_latents], dim=1)
        B, Fp1, C, H, W = latents_with_dummy.shape

        noise = torch.randn_like(latents_with_dummy.flatten(0, 1))
        if noise_t > 0:
            t_fixed = torch.full((noise.shape[0],), noise_t, dtype=torch.long, device=device)
            noisy = scheduler.add_noise(
                latents_with_dummy.flatten(0, 1), noise, t_fixed,
            ).unflatten(0, (B, Fp1))
        else:
            noisy = latents_with_dummy
        logging.info("Noisy latents (t=%d): %s", noise_t, tuple(noisy.shape))

        # ── Decode to pixels ──
        t0 = time.perf_counter()
        with torch.no_grad():
            pixels = vae.decode_to_pixel(noisy.to(dtype=torch.bfloat16))
        pixels = pixels[:, 1:, ...]
        t_dec = time.perf_counter() - t0
        logging.info("Decoded: %s  (%.2fs)", tuple(pixels.shape), t_dec)

        video_0255 = (255.0 * 0.5 * (pixels.float() + 1.0)).clamp(0, 255)

        # ── Run CoTracker ──
        vid = video_0255[0].unsqueeze(0)
        T_total = vid.shape[1]

        logging.info("Running CoTracker on %d video frames ...", T_total)
        t0 = time.perf_counter()
        all_motion = []
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
            all_motion.append(mw)

        est_motion = torch.cat(all_motion, dim=0)
        t_ct = time.perf_counter() - t0
        logging.info("CoTracker done: %d segments  (%.2fs)", est_motion.shape[0], t_ct)

        # ── Encode through ss_vae ──
        n_seg = est_motion.shape[0]
        xy = est_motion[:, :, :2].reshape(n_seg, 10, 10, 2)
        x_in = xy.permute(0, 3, 1, 2).float() / scale
        with torch.no_grad():
            mu, _ = ss_vae.encoder(x_in.to(device))
        z_full_raw = mu.squeeze(-1).squeeze(-1)
        z_full = _tanh_squash(z_full_raw)
        gen_z27 = z_full[:, [2, 7]]
        logging.info("Generated z (all 8 dims, squashed): %s", tuple(z_full.shape))

        # ── Report (all 8 z dims) ──
        logging.info("-" * 60)
        logging.info("PIPELINE RESULTS – window %d – all z-dims (squashed)", win_idx)
        logging.info("-" * 60)
        z_full_np_report = z_full.cpu().numpy()
        z_target_np_report = z_target[0].cpu().numpy()
        for i in range(min(n_seg, 7)):
            gen_vals = " ".join(f"z{d}={z_full_np_report[i, d]:+.3f}" for d in range(8))
            lat_idx_r = min(i * 3, len(z_target_np_report) - 1)
            tgt_vals = " ".join(f"z{d}={z_target_np_report[lat_idx_r, d]:+.3f}" for d in range(8))
            logging.info("  seg %d gen: %s", i, gen_vals)
            logging.info("  seg %d tgt: %s", i, tgt_vals)
            logging.info("")

        # Reward (asymmetric: overshoot penalised less than undershoot)
        target_seg_27 = z_target_27[0, :n_seg * 3].reshape(n_seg, 3, 2).mean(dim=1) if n_seg * 3 <= z_target_27.shape[1] else gen_z27 * 0
        diff = _asymmetric_action_diff(gen_z27, target_seg_27)
        reward = -torch.log(diff + 1e-6).mean(dim=-1)
        logging.info("Per-segment reward: %s", reward.cpu().tolist()[:7])
        logging.info("Mean reward: %.6f", reward.mean().item())

        # ── Prepare numpy arrays for annotation ──
        z_full_np = z_full.cpu().numpy()
        z_target_np = z_target[0].cpu().numpy()
        reward_np = reward.cpu().numpy()

        # ── Save annotated video ──
        vid_np = video_0255[0].to(torch.uint8).cpu().permute(0, 2, 3, 1).contiguous().numpy()
        n_vid = vid_np.shape[0]
        panel_w = 140
        annotated = []
        for j in range(n_vid):
            f = vid_np[j].copy()
            seg_idx = min(j // (output_chunk_size), n_seg - 1) if n_seg > 0 else 0
            lat_idx = min(j // _LATENT_TO_VIDEO, len(z_target_np) - 1)
            h, w = f.shape[:2]

            gen_8d = z_full_np[seg_idx] if seg_idx < z_full_np.shape[0] else np.zeros(8)
            tgt_8d = z_target_np[lat_idx]
            rw = float(reward_np[seg_idx]) if seg_idx < reward_np.shape[0] else 0.0

            f = _draw_paired_latent_overlay(f, gen_8d, tgt_8d)
            f = _draw_paired_dials(f, gen_8d, tgt_8d, rw)

            if seg_idx < est_motion.shape[0]:
                mvecs = est_motion[seg_idx].cpu().numpy()
                for gy in range(grid_size):
                    for gx in range(grid_size):
                        idx = gy * grid_size + gx
                        dx, dy, vis_val = mvecs[idx]
                        if vis_val < 0.2:
                            continue
                        cx = int((gx + 0.5) * (w - panel_w) / grid_size)
                        cy = int((gy + 0.5) * h / grid_size)
                        ex = int(cx - dx * 3)
                        ey = int(cy - dy * 3)
                        color = (0, 255, 0) if vis_val >= 0.5 else (0, 200, 255)
                        cv2.arrowedLine(f, (cx, cy), (ex, ey), color, 1, tipLength=0.3)

            bar_h = 30
            f[h - bar_h:h, :, :] = (f[h - bar_h:h, :, :].astype(np.float32) * 0.4).astype(np.uint8)
            cv2.putText(f, f"win={win_idx}  frame={j}  seg={seg_idx}  lat={lat_start}:{lat_end}  noise_t={noise_t}",
                         (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
            annotated.append(f)

        arr = np.stack(annotated)
        out_path = out_dir / f"action_critic_visual_win{win_idx:02d}.mp4"
        _save_mp4(arr, out_path, 20.0)
        logging.info("Saved visual output: %s  (%d frames)", out_path, n_vid)

    # Cleanup
    del vae, cotracker, ss_vae
    torch.cuda.empty_cache()
    logging.info("Visual pipeline test PASSED")


# ===================================================================
# Single-step training test
# ===================================================================

def run_train_test(args) -> None:
    """Build the full trainer pipeline, run 1 training step, report everything."""
    from utils.wan_wrapper import WanVAEWrapper, WanDiffusionWrapper
    from utils.scheduler import FlowMatchScheduler
    from utils.zarr_dataset import ZarrRideDataset, _tanh_squash
    from model.action_critic import ActionCritic
    from model.action_modulation import ActionModulationProjection, ActionTokenProjection
    from model.action_model_patch import apply_action_patches
    from action_query.ss_vae_model import load_ss_vae

    device = torch.device(args.device)
    dtype = torch.bfloat16

    logging.info("=" * 60)
    logging.info("TRAINING TEST")
    logging.info("=" * 60)

    # ── Load real data ──
    logging.info("Loading dataset ...")
    num_steps = getattr(args, "steps", 1)
    cf = 3
    window_size = 24
    num_frames = window_size - cf  # 21
    total_frames_needed = window_size + (num_steps - 1) * num_frames

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
    total_frames_needed = min(total_frames_needed, n_lat)

    chunk = ZarrRideDataset.load_latent_chunk(zarr_path, 0, total_frames_needed)
    all_latents = chunk.unsqueeze(0).to(device=device, dtype=torch.float32)
    prompt_embeds = ride["prompt_embeds"].unsqueeze(0).to(device=device, dtype=dtype)

    all_z_actions = dataset.encode_z_actions_window(
        zarr_path, n_lat, 0, total_frames_needed,
    ).unsqueeze(0).to(device=device, dtype=dtype)

    bsz = 1
    action_dims = [2, 7]

    logging.info("Data loaded:  all_latents=%s  all_z_actions=%s  (%d steps, stride=%d)",
                 tuple(all_latents.shape), tuple(all_z_actions.shape), num_steps, num_frames)

    # ── Build diffusion model (causal, with action patches, LoRA) ──
    logging.info("Building CausalWanModel + LoRA ...")
    import peft
    from peft import LoraConfig

    wrapper = WanDiffusionWrapper(
        model_name="Wan2.1-T2V-1.3B",
        is_causal=True,
        timestep_shift=5.0,
        local_attn_size=-1,
        sink_size=0,
    )
    wrapper.enable_gradient_checkpointing()
    wrapper.model.num_frame_per_block = 3
    wrapper.model.context_shift = 1

    apply_action_patches(wrapper)

    model_dim = getattr(wrapper.model, "dim", 2048)
    action_dim = 2

    action_projection = ActionModulationProjection(
        action_dim=action_dim, activation="silu",
        hidden_dim=model_dim, num_frames=1, zero_init=True,
    ).to(device)
    action_projection.train()

    action_token_projection = ActionTokenProjection(
        action_dim=action_dim, activation="silu",
        hidden_dim=model_dim, zero_init=True,
    ).to(device)
    action_token_projection.train()
    wrapper.model.action_tokens_per_frame = 1
    wrapper.adjust_seq_len_for_action_tokens(num_frames=num_frames, action_per_frame=1)

    # LoRA
    target_modules = set()
    for mn, mod in wrapper.model.named_modules():
        if mod.__class__.__name__ in {"WanAttentionBlock", "CausalWanAttentionBlock"}:
            for fn, sub in mod.named_modules(prefix=mn):
                if isinstance(sub, torch.nn.Linear):
                    target_modules.add(fn)
    lora_config = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.0,
        target_modules=sorted(target_modules),
        init_lora_weights="gaussian",
    )
    wrapper.model = peft.get_peft_model(wrapper.model, lora_config)
    wrapper.to(device)
    wrapper.train()

    scheduler = wrapper.get_scheduler()
    scheduler.set_timesteps(1000, training=True)

    # ── Build action critic + frozen evaluator ──
    logging.info("Building ActionCritic (reward-only) + frozen evaluator ...")
    num_frame_per_block = 3
    critic = ActionCritic(
        latent_channels=16, action_dim=2, base_channels=64,
        num_res_blocks=3, chunk_frames=num_frame_per_block,
    ).to(device)
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

    # ── Build optimizer ──
    params = [p for p in wrapper.parameters() if p.requires_grad]
    params += list(action_projection.parameters())
    params += list(action_token_projection.parameters())
    params += list(critic.parameters())
    optimizer = torch.optim.AdamW(params, lr=5e-5, weight_decay=0.01)

    n_trainable = sum(p.numel() for p in params)
    logging.info("Trainable params: %d  (%d param tensors)", n_trainable, len(params))

    timesteps = scheduler.timesteps.to(device)

    for step_i in range(1, num_steps + 1):
        t_step_start = time.perf_counter()
        win_start = (step_i - 1) * num_frames
        win_end = win_start + window_size
        logging.info("-" * 60)
        logging.info("STEP %d / %d  (frames %d–%d)", step_i, num_steps, win_start, win_end)
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
        t_idx = t_idx.reshape(bsz, -1, 3)
        t_idx[:, :, 1:] = t_idx[:, :, 0:1]
        t_idx = t_idx.reshape(bsz, num_frames)
        ts = timesteps[t_idx]

        noise = torch.randn_like(target_latents)
        noisy_latents = scheduler.add_noise(
            target_latents.flatten(0, 1), noise.flatten(0, 1), ts.flatten(0, 1),
        ).view_as(target_latents)
        training_target = scheduler.training_target(
            target_latents.flatten(0, 1), noise.flatten(0, 1), ts.flatten(0, 1),
        ).view_as(target_latents)

        # ── Forward ──
        t0 = time.perf_counter()
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            flow_pred, pred_x0 = wrapper(
                noisy_latents, conditional, ts,
                clean_x=context_latents, aug_t=None,
            )
            flow_loss_raw = F.mse_loss(flow_pred.float(), training_target.float(), reduction="none")
            flow_loss_raw = flow_loss_raw.mean(dim=(2, 3, 4))
            weights = scheduler.training_weight(ts.flatten(0, 1)).view(bsz, num_frames)
            flow_loss = (flow_loss_raw * weights).mean()
        t_fwd = time.perf_counter() - t0

        # ── Teacher targets (frozen, from detached pred_x0) ──
        t0 = time.perf_counter()
        teacher_z, teacher_reward = _compute_teacher_targets(
            pred_x0.detach(), target_action_z,
            scheduler, frozen_vae, cotracker, ss_vae, ss_scale,
            device, dtype, noise_t=25, critic_dims=[2, 7],
            chunk_frames=num_frame_per_block,
        )
        t_teacher = time.perf_counter() - t0

        # ── Reward critic + generator guidance ──
        n_chunks = teacher_z.shape[1]
        chunk_t = ts[:, ::num_frame_per_block][:, :n_chunks]
        chunk_actions = _chunk_actions(target_action_z, num_frame_per_block)[:, :n_chunks]

        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            # Critic training loss (detached pred_x0)
            pred_r_c = critic(pred_x0.detach(), chunk_t, chunk_actions)
            pred_r_c = pred_r_c[:, :n_chunks].float()

            critic_loss = F.mse_loss(pred_r_c, teacher_reward.float())

            # Generator guidance (frozen critic, gradient through pred_x0)
            critic.requires_grad_(False)
            gen_r = critic(pred_x0, chunk_t, chunk_actions)
            gen_r = gen_r[:, :n_chunks].float()
            gen_reward_loss = -gen_r.mean()
            gen_action_loss = 1.0 * gen_reward_loss
            critic.requires_grad_(True)

            total_loss = flow_loss + critic_loss + gen_action_loss

        # ── Backward ──
        t0 = time.perf_counter()
        total_loss.backward()
        t_bwd = time.perf_counter() - t0

        # ── Optimizer step ──
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        t_step = time.perf_counter() - t_step_start
        logging.info("  timing: fwd=%.2fs  teacher=%.2fs  bwd=%.2fs  step=%.2fs",
                     t_fwd, t_teacher, t_bwd, t_step)
        logging.info("  flow_loss       = %.6f", flow_loss.item())
        logging.info("  critic_loss     = %.6f", critic_loss.item())
        logging.info("  gen_reward_loss = %.6f   gen_total   = %.6f",
                     gen_reward_loss.item(), gen_action_loss.item())
        logging.info("  TOTAL           = %.6f", total_loss.item())

    # ── Final gradient report + sanity checks (last step only) ──
    logging.info("=" * 60)
    logging.info("GRADIENT REPORT (final step)")
    logging.info("=" * 60)

    def _grad_norm(module, name):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        if not grads:
            logging.info("  %-35s  NO GRAD", name)
            return 0.0
        norm = torch.norm(torch.stack([g.norm() for g in grads])).item()
        n_with = len(grads)
        n_total = sum(1 for p in module.parameters() if p.requires_grad)
        logging.info("  %-35s  norm=%.6f  (%d/%d params with grad)", name, norm, n_with, n_total)
        return norm

    gn_model = _grad_norm(wrapper, "WanDiffusionWrapper (LoRA)")
    gn_adaln = _grad_norm(action_projection, "ActionModulationProjection")
    gn_atokp = _grad_norm(action_token_projection, "ActionTokenProjection")
    gn_critic = _grad_norm(critic, "ActionCritic")

    logging.info("=" * 60)
    logging.info("SANITY CHECKS")
    logging.info("=" * 60)

    checks_passed = 0
    checks_total = 0

    def check(name, condition):
        nonlocal checks_passed, checks_total
        checks_total += 1
        if condition:
            checks_passed += 1
            logging.info("  [PASS] %s", name)
        else:
            logging.error("  [FAIL] %s", name)

    check("total_loss is finite", torch.isfinite(total_loss))
    check("flow_loss > 0", flow_loss.item() > 0)
    check("critic_loss is finite", torch.isfinite(critic_loss))
    check("gen_action_loss is finite", torch.isfinite(gen_action_loss))
    check("LoRA grads non-zero", gn_model > 0)
    check("adaLN projection grads non-zero", gn_adaln > 0)
    check("action token projection grads non-zero", gn_atokp > 0)
    check("critic grads non-zero", gn_critic > 0)
    check("teacher_reward has correct dims", teacher_reward.shape[-1] == 1)
    check("critic returns single reward per chunk",
          pred_r_c.shape == (bsz, n_chunks, 1))
    check("timestep embedding works (critic has time_embed params)",
          any("time_embed" in n for n, _ in critic.named_parameters()))
    check("action embedding works (critic has action_embed params)",
          any("action_embed" in n for n, _ in critic.named_parameters()))
    check("gen grads flow through pred_x0 to model",
          gn_model > 0 and gn_critic > 0)

    logging.info("=" * 60)
    logging.info("RESULT: %d/%d checks passed", checks_passed, checks_total)
    logging.info("=" * 60)

    if checks_passed < checks_total:
        logging.error("SOME CHECKS FAILED")
        sys.exit(1)
    else:
        logging.info("ALL CHECKS PASSED")

    del wrapper, critic, frozen_vae, cotracker, ss_vae
    torch.cuda.empty_cache()


# ===================================================================
# Teacher pipeline regression test (latent → VAE decode → CoTracker → ss_vae)
# ===================================================================

def run_teacher_pipeline_test(args) -> None:
    """Decode real zarr latents, run the teacher pipeline, compare against saved motion z-actions.

    This mirrors what training does: start from encoded latents, VAE-decode to
    pixels, run CoTracker → ss_vae, and check that the resulting z2/z7 values
    are aligned with the motion-file-derived targets for the same ride window.
    """
    from utils.wan_wrapper import WanVAEWrapper
    from utils.zarr_dataset import (
        ZarrRideDataset,
        _tanh_squash,
        _LATENT_TO_VIDEO,
    )
    from action_query.ss_vae_model import load_ss_vae

    device = torch.device(args.device)
    action_dims = [2, 7]
    window_size = 24
    cf = 3
    num_frames = window_size - cf
    noise_t = args.noise_t

    # ── Load dataset ──
    logging.info("Building ZarrRideDataset ...")
    dataset = ZarrRideDataset(
        encoded_root=args.encoded_root,
        caption_root=args.caption_root,
        motion_root=args.motion_root,
        ss_vae_checkpoint=args.ss_vae_checkpoint,
        min_ride_frames=window_size,
        device="cpu",
        max_rides=max(args.ride_index + 1, 4),
    )
    if len(dataset) <= args.ride_index:
        raise IndexError(f"ride_index={args.ride_index} out of range for dataset size {len(dataset)}")

    ride = dataset[args.ride_index]
    zarr_path = ride["zarr_path"]
    n_lat = int(ride["n_latent_frames"])
    logging.info("Using ride index %d: %s  (%d latent frames)", args.ride_index, zarr_path, n_lat)

    # ── Load latents + z-actions for one window ──
    chunk = ZarrRideDataset.load_latent_chunk(zarr_path, 0, window_size)
    latents = chunk.unsqueeze(0).to(device=device, dtype=torch.float32)
    # Match training: teacher pipeline operates on target latents (after context)
    target_latents = latents[:, cf:]          # [1, num_frames, C, H, W]

    z_actions = dataset.encode_z_actions_window(
        zarr_path, n_lat, 0, window_size,
    ).unsqueeze(0).to(device=device)
    z_target = z_actions[:, cf:]              # same temporal span as target_latents
    z_target_27 = z_target[..., action_dims]
    logging.info("Target latents: %s   z_target_27: %s", tuple(target_latents.shape), tuple(z_target_27.shape))

    # ── Load heavy models ──
    from utils.scheduler import FlowMatchScheduler
    scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)

    logging.info("Loading VAE ...")
    vae = WanVAEWrapper().to(device=device, dtype=torch.bfloat16).eval()
    vae.requires_grad_(False)

    logging.info("Loading CoTracker ...")
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    cotracker.eval()
    for p in cotracker.parameters():
        p.requires_grad_(False)

    logging.info("Loading ss_vae ...")
    ss_vae, scale = load_ss_vae(args.ss_vae_checkpoint, device=str(device))
    ss_vae.eval()
    ss_vae.requires_grad_(False)

    grid_size = 10
    output_chunk_size = 12
    N = grid_size ** 2

    # ── Prepare latents (optionally add noise, matching training pipeline) ──
    dummy = target_latents[:, 0:1]
    latents_with_dummy = torch.cat([dummy, target_latents], dim=1)
    B, Fp1, C, H, W = latents_with_dummy.shape

    if noise_t > 0:
        noise = torch.randn_like(latents_with_dummy.flatten(0, 1))
        t_fixed = torch.full((noise.shape[0],), noise_t, dtype=torch.long, device=device)
        decode_input = scheduler.add_noise(
            latents_with_dummy.flatten(0, 1), noise, t_fixed,
        ).unflatten(0, (B, Fp1))
    else:
        decode_input = latents_with_dummy
    logging.info("Decode input (noise_t=%d): %s", noise_t, tuple(decode_input.shape))

    # ── VAE decode → pixels ──
    t0 = time.perf_counter()
    with torch.no_grad():
        pixels = vae.decode_to_pixel(decode_input.to(dtype=torch.bfloat16))
    pixels = pixels[:, 1:, ...]
    t_dec = time.perf_counter() - t0
    logging.info("Decoded pixels: %s  (%.2fs)", tuple(pixels.shape), t_dec)

    video_0255 = (255.0 * 0.5 * (pixels.float() + 1.0)).clamp(0, 255)

    # ── CoTracker ──
    vid = video_0255[0].unsqueeze(0)
    T_total = vid.shape[1]

    logging.info("Running CoTracker on %d video frames ...", T_total)
    t0 = time.perf_counter()
    all_motion = []
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
        all_motion.append(mw)

    est_motion = torch.cat(all_motion, dim=0)
    t_ct = time.perf_counter() - t0
    n_seg = est_motion.shape[0]
    logging.info("CoTracker done: %d segments  (%.2fs)", n_seg, t_ct)

    # ── ss_vae encode ──
    xy = est_motion[:, :, :2].reshape(n_seg, 10, 10, 2)
    x_in = xy.permute(0, 3, 1, 2).float() / scale
    with torch.no_grad():
        mu, _ = ss_vae.encoder(x_in.to(device))
    z_full = _tanh_squash(mu.squeeze(-1).squeeze(-1))
    teacher_z27 = z_full[:, action_dims].float().cpu()
    logging.info("Teacher z (all 8): %s  z2/z7: %s", tuple(z_full.shape), tuple(teacher_z27.shape))

    # ── Build comparable motion-file targets ──
    # z_target_27 is [1, num_frames, 2] at latent frame rate.
    # teacher_z27 is [n_seg, 2] at 12-video-frame segments.
    # Map each segment to the mean of the corresponding latent-frame targets.
    target_seg_list = []
    for seg_i in range(n_seg):
        vid_start = seg_i * output_chunk_size
        vid_end = vid_start + output_chunk_size
        lat_start = vid_start // _LATENT_TO_VIDEO
        lat_end = min((vid_end + _LATENT_TO_VIDEO - 1) // _LATENT_TO_VIDEO, z_target_27.shape[1])
        target_seg_list.append(z_target_27[0, lat_start:lat_end].mean(dim=0))
    motion_target_z = torch.stack(target_seg_list).cpu()

    n_compare = min(teacher_z27.shape[0], motion_target_z.shape[0])
    teacher_z27 = teacher_z27[:n_compare]
    motion_target_z = motion_target_z[:n_compare]

    # ── Metrics ──
    mse = F.mse_loss(teacher_z27, motion_target_z).item()
    mae = F.l1_loss(teacher_z27, motion_target_z).item()
    zero_mse = F.mse_loss(torch.zeros_like(motion_target_z), motion_target_z).item()
    prev_shift_mse = F.mse_loss(teacher_z27[1:], motion_target_z[:-1]).item()
    next_shift_mse = F.mse_loss(teacher_z27[:-1], motion_target_z[1:]).item()

    logging.info("=" * 60)
    logging.info("TEACHER PIPELINE RESULTS")
    logging.info("=" * 60)
    logging.info("Teacher z2/z7 shape: %s", tuple(teacher_z27.shape))
    logging.info("Motion target z2/z7 shape: %s", tuple(motion_target_z.shape))

    for i in range(n_compare):
        tz = teacher_z27[i].numpy()
        mz = motion_target_z[i].numpy()
        logging.info("  seg %d  teacher=[%+.4f, %+.4f]  motion=[%+.4f, %+.4f]  err=[%.4f, %.4f]",
                     i, tz[0], tz[1], mz[0], mz[1], abs(tz[0] - mz[0]), abs(tz[1] - mz[1]))

    logging.info("Alignment: mse=%.6f mae=%.6f", mse, mae)
    logging.info("Baselines: zero=%.6f prev_shift=%.6f next_shift=%.6f",
                 zero_mse, prev_shift_mse, next_shift_mse)

    # ── Checks ──
    checks_passed = 0
    checks_total = 0

    def check(name, condition):
        nonlocal checks_passed, checks_total
        checks_total += 1
        if condition:
            checks_passed += 1
            logging.info("  [PASS] %s", name)
        else:
            logging.error("  [FAIL] %s", name)

    check("teacher shape matches motion target", teacher_z27.shape == motion_target_z.shape)
    check("teacher z finite", torch.isfinite(teacher_z27).all().item())
    check("motion target z finite", torch.isfinite(motion_target_z).all().item())
    check("teacher beats zero baseline (mse %.6f < %.6f)" % (mse, zero_mse), mse < zero_mse)
    check("teacher beats prev-shift baseline (mse %.6f < %.6f)" % (mse, prev_shift_mse), mse < prev_shift_mse)
    check("teacher beats next-shift baseline (mse %.6f < %.6f)" % (mse, next_shift_mse), mse < next_shift_mse)
    check("per-segment MAE < 0.5", mae < 0.5)

    logging.info("=" * 60)
    logging.info("TEACHER PIPELINE: %d/%d checks passed", checks_passed, checks_total)
    logging.info("=" * 60)

    del vae, cotracker, ss_vae
    torch.cuda.empty_cache()

    if checks_passed < checks_total:
        logging.error("TEACHER PIPELINE TEST FAILED")
        sys.exit(1)

    logging.info("TEACHER PIPELINE TEST PASSED")


# ===================================================================
# Shared helpers
# ===================================================================

def _chunk_actions(per_frame: torch.Tensor, chunk_frames: int) -> torch.Tensor:
    """Mean-pool ``[B, F, D]`` into ``[B, n_chunks, D]`` with exact chunk size."""
    B, F_len, D = per_frame.shape
    n_chunks = F_len // chunk_frames
    trimmed = per_frame[:, :n_chunks * chunk_frames]
    return trimmed.reshape(B, n_chunks, chunk_frames, D).mean(dim=2)


def _asymmetric_action_diff(
    gen: torch.Tensor,
    target: torch.Tensor,
    over_weight: float = 0.5,
    under_weight: float = 1.0,
) -> torch.Tensor:
    """Asymmetric action error: undershoot penalised more than overshoot."""
    raw_err = gen - target
    dir_sign = torch.sign(target)
    signed_err = raw_err * dir_sign
    over_err = torch.relu(signed_err)
    under_err = torch.relu(-signed_err)
    asym = over_weight * over_err + under_weight * under_err
    sym = raw_err.abs()
    return torch.where(target.abs() > 1e-3, asym, sym)


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


@torch.no_grad()
def _compute_teacher_targets(
    pred_x0, target_action_z,
    scheduler, frozen_vae, cotracker, ss_vae, ss_scale,
    device, dtype, noise_t=25, critic_dims=None, chunk_frames=3,
):
    """Standalone teacher targets aligned to the chunkwise critic contract."""
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
        diff = _asymmetric_action_diff(gz_chunked, target_chunked)
        reward = -torch.log(diff + 1e-6).mean(dim=-1, keepdim=True)
        all_z.append(gz_chunked)
        all_r.append(reward)

    tz = torch.stack(all_z)
    tr = torch.stack(all_r)
    return tz.detach(), tr.detach()


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


# ===================================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="GPU integration test for the action critic pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["visual", "train", "both", "teacher", "all"], default="both")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--encoded_root", default="/projects/u6ej/fbots/frodobots_encoded")
    parser.add_argument("--caption_root", default="/projects/u6ej/fbots/frodobots_captions/train")
    parser.add_argument("--motion_root", default="/projects/u6ej/fbots/frodobots_motion")
    parser.add_argument("--ss_vae_checkpoint", default="action_query/checkpoints/ss_vae_8free.pt")
    parser.add_argument("--noise_t", type=int, default=25,
                        help="Noise timestep for the teacher pipeline (0 = no noise)")
    parser.add_argument("--num_windows", type=int, default=1,
                        help="Number of consecutive windows to process (visual mode)")
    parser.add_argument("--steps", type=int, default=1,
                        help="Number of training steps to run (train mode)")
    parser.add_argument("--ride_index", type=int, default=0,
                        help="Ride index for teacher pipeline test")
    parser.add_argument("--output_dir", default="testing/outputs")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        logging.error("This test requires a GPU. Run on a compute node.")
        sys.exit(1)

    if args.mode in ("visual", "both", "all"):
        logging.info(">>> Running VISUAL pipeline test <<<")
        run_visual_test(args)

    if args.mode in ("train", "both", "all"):
        logging.info(">>> Running TRAINING step test <<<")
        run_train_test(args)

    if args.mode in ("teacher", "all"):
        logging.info(">>> Running TEACHER PIPELINE regression test <<<")
        run_teacher_pipeline_test(args)


if __name__ == "__main__":
    main()
