#!/usr/bin/env python3
"""Motion Habitat: probe frozen action pipeline across diffusion timesteps.

For each video, at each of 50 denoising steps, run the frozen action pipeline
(VAE decode -> CoTracker -> ss_vae) on pred_x0 and record the raw teacher_z
values for each of the 7 three-frame blocks.

Produces 14 charts per video (7 blocks x {z2, z7}):
  - Bar chart: raw teacher_z value at each denoising step
  - Horizontal lines: GT clean, GT+t=25, and dataset z_actions

Usage:
    python testing/motion_habitat.py \\
        --config configs/causal_lora_diffusion_teacher.yaml \\
        --checkpoint logs/causal_lora_teacher/causal_lora_step0003150.pt \\
        --num_videos 16 --num_steps 50 --device cuda:0
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import peft
from peft import LoraConfig, set_peft_model_state_dict
from omegaconf import OmegaConf

from utils.wan_wrapper import WanDiffusionWrapper, WanVAEWrapper
from utils.scheduler import FlowMatchScheduler
from utils.zarr_dataset import (
    ZarrRideDataset,
    build_ride_manifest,
    _tanh_squash,
    _ZACTION_SCALES,
)
from model.action_model_patch import apply_action_patches
from model.action_modulation import ActionModulationProjection, ActionTokenProjection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

NUM_BLOCKS = 7
BLOCK_SIZE = 3
DIM_NAMES = ["z2", "z7"]


# ---------------------------------------------------------------------------
# Frozen action pipeline
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_frozen_action_pipeline(
    pred_x0: torch.Tensor,
    frozen_vae: WanVAEWrapper,
    frozen_cotracker,
    frozen_ss_vae,
    ss_vae_scale: float,
    action_critic_dims: list[int],
    scheduler: FlowMatchScheduler | None = None,
    noise_timestep: int = 0,
    device: torch.device | str = "cuda",
) -> list[torch.Tensor]:
    """Run VAE decode -> CoTracker -> ss_vae on pred_x0.

    Returns list (one per batch item) of [n_chunks, z_dim] tensors with raw
    tanh-squashed z values.
    """
    B, F_frames, C, H, W = pred_x0.shape

    dummy = pred_x0[:, 0:1]
    latents_with_dummy = torch.cat([dummy, pred_x0], dim=1)

    if noise_timestep > 0 and scheduler is not None:
        noise = torch.randn_like(latents_with_dummy.flatten(0, 1))
        t_fixed = torch.full(
            (noise.shape[0],), noise_timestep,
            dtype=torch.long, device=pred_x0.device,
        )
        latents_with_dummy = scheduler.add_noise(
            latents_with_dummy.flatten(0, 1), noise, t_fixed,
        ).unflatten(0, (B, F_frames + 1))

    pixels = frozen_vae.decode_to_pixel(latents_with_dummy.float())[:, 1:, ...]
    video = (255.0 * 0.5 * (pixels + 1.0)).clamp(0, 255).float()

    grid_size = 10
    output_chunk_size = 12
    compute_T = 48

    all_teacher_z: list[torch.Tensor] = []

    for b in range(B):
        vid = video[b]
        T_total = vid.shape[0]
        vid_5d = vid.unsqueeze(0)

        motion_windows: list[torch.Tensor] = []
        for chunk_start in range(0, T_total, compute_T):
            chunk_end = min(chunk_start + compute_T, T_total)
            chunk = vid_5d[:, chunk_start:chunk_end]
            n_out = chunk.shape[1] // output_chunk_size
            if n_out == 0:
                continue
            used = n_out * output_chunk_size
            chunk = chunk[:, :used].clone()

            with torch.amp.autocast(device_type="cuda", enabled=True):
                pred_tracks, pred_vis = frozen_cotracker(chunk, grid_size=grid_size)

            N = grid_size ** 2
            tracks_w = pred_tracks.reshape(1, n_out, output_chunk_size, N, 2)
            if pred_vis.dim() == 3:
                vis_w = pred_vis.reshape(1, n_out, output_chunk_size, N).unsqueeze(-1)
            else:
                vis_w = pred_vis.reshape(1, n_out, output_chunk_size, N, 1)

            d_w = tracks_w[:, :, 1:] - tracks_w[:, :, :-1]
            motion_out = d_w.mean(dim=2)
            vis_out = vis_w.to(dtype=motion_out.dtype).mean(dim=2)
            mw = torch.cat([motion_out, vis_out], dim=-1).squeeze(0)
            motion_windows.append(mw)

        if not motion_windows:
            z_dim = len(action_critic_dims)
            all_teacher_z.append(torch.zeros(1, z_dim, device=pred_x0.device))
            continue

        est_motion = torch.cat(motion_windows, dim=0)
        raw_n = est_motion.shape[0]
        xy = est_motion[:, :, :2].reshape(raw_n, 10, 10, 2)
        x_in = xy.permute(0, 3, 1, 2).float() / ss_vae_scale
        mu, _ = frozen_ss_vae.encoder(x_in.to(device))
        z_full_8d = mu.squeeze(-1).squeeze(-1)
        z_full_8d = _tanh_squash(z_full_8d)
        gen_z = z_full_8d[:, action_critic_dims]
        all_teacher_z.append(gen_z)

    return all_teacher_z


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def collect_target_modules(model) -> list[str]:
    target_modules = set()
    for module_name, module in model.named_modules():
        if module.__class__.__name__ in {"WanAttentionBlock", "CausalWanAttentionBlock"}:
            for full_name, submodule in module.named_modules(prefix=module_name):
                if isinstance(submodule, torch.nn.Linear):
                    target_modules.add(full_name)
    return sorted(target_modules)


def build_model(config, checkpoint_path: str, device: torch.device):
    model_kwargs = dict(getattr(config, "model_kwargs", {}))
    wrapper = WanDiffusionWrapper(
        model_name=getattr(config, "model_name", "Wan2.1-T2V-1.3B"),
        is_causal=True,
        **model_kwargs,
    )

    num_train_timestep = int(getattr(config, "num_train_timestep", 1000))
    train_scheduler = wrapper.get_scheduler()
    train_scheduler.set_timesteps(num_inference_steps=num_train_timestep, denoising_strength=1.0)

    num_frame_per_block = int(getattr(config, "num_frame_per_block", 3))
    context_frames = int(getattr(config, "context_frames", 3))
    wrapper.model.num_frame_per_block = num_frame_per_block
    wrapper.model.context_shift = context_frames // num_frame_per_block

    model_variant = getattr(config, "model_variant", None)
    action_projection = None
    action_token_projection = None
    num_train_frames = int(getattr(config, "num_training_frames", 21))

    if model_variant == "action-injection":
        apply_action_patches(wrapper)
        model_dim = getattr(wrapper.model, "dim", 2048)
        action_dim = int(getattr(config, "raw_action_dim", 8))
        enable_adaln_zero = bool(getattr(config, "enable_adaln_zero", True))
        activation = getattr(config, "action_activation", "silu")
        mode = getattr(config, "action_conditioning_mode", "both")

        if mode in ("adaln", "both"):
            action_projection = ActionModulationProjection(
                action_dim=action_dim, activation=activation,
                hidden_dim=model_dim, num_frames=1, zero_init=enable_adaln_zero,
            )
        if mode in ("action_tokens", "both"):
            action_token_projection = ActionTokenProjection(
                action_dim=action_dim, activation=activation,
                hidden_dim=model_dim, zero_init=enable_adaln_zero,
            )
            wrapper.model.action_tokens_per_frame = 1
            wrapper.adjust_seq_len_for_action_tokens(
                num_frames=num_train_frames, action_per_frame=1,
            )

    lora_cfg = getattr(config, "adapter", None)
    if lora_cfg is None:
        raise ValueError("LoRA adapter configuration required")

    rank = lora_cfg.get("rank", 128) if isinstance(lora_cfg, dict) else getattr(lora_cfg, "rank", 128)
    alpha = lora_cfg.get("alpha", rank) if isinstance(lora_cfg, dict) else getattr(lora_cfg, "alpha", rank)
    dropout = lora_cfg.get("dropout", 0.0) if isinstance(lora_cfg, dict) else getattr(lora_cfg, "dropout", 0.0)

    target_modules = collect_target_modules(wrapper.model)
    if not target_modules:
        target_modules = ["q", "k", "v", "o"]

    lora_config = LoraConfig(
        r=int(rank), lora_alpha=float(alpha), lora_dropout=float(dropout),
        target_modules=target_modules, bias="none",
    )
    wrapper.model = peft.get_peft_model(wrapper.model, lora_config)
    logging.info("LoRA applied: rank=%s alpha=%s", rank, alpha)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    set_peft_model_state_dict(wrapper.model, checkpoint["lora"])
    logging.info("Loaded LoRA weights from %s (step %d)", checkpoint_path, checkpoint.get("step", -1))

    if action_projection is not None and "action_projection" in checkpoint:
        action_projection.load_state_dict(checkpoint["action_projection"])
        logging.info("Restored action modulation projection")
    if action_token_projection is not None and "action_token_projection" in checkpoint:
        action_token_projection.load_state_dict(checkpoint["action_token_projection"])
        logging.info("Restored action token projection")

    wrapper.to(device)
    wrapper.eval()
    if action_projection is not None:
        action_projection.to(device).eval()
    if action_token_projection is not None:
        action_token_projection.to(device).eval()

    return wrapper, train_scheduler, action_projection, action_token_projection


def build_frozen_evaluators(device: torch.device):
    frozen_vae = WanVAEWrapper()
    frozen_vae.eval().requires_grad_(False).to(device)

    frozen_cotracker = torch.hub.load(
        "facebookresearch/co-tracker", "cotracker3_offline",
    ).to(device)
    frozen_cotracker.eval()
    for p in frozen_cotracker.parameters():
        p.requires_grad_(False)

    from action_query.ss_vae_model import load_ss_vae
    ss_vae, scale = load_ss_vae("action_query/checkpoints/ss_vae_8free.pt", device=str(device))
    ss_vae.eval().requires_grad_(False)

    return frozen_vae, frozen_cotracker, ss_vae, scale


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_videos(config, num_videos: int, device: torch.device,
                video_offset: int = 0, manifest_cache: str | None = None):
    motion_root = config.motion_root
    ss_vae_ckpt = getattr(config, "ss_vae_checkpoint", "action_query/checkpoints/ss_vae_8free.pt")

    context_frames = int(getattr(config, "context_frames", 3))
    streaming_chunk_size = int(getattr(config, "streaming_chunk_size", 21))
    min_ride_frames = streaming_chunk_size + context_frames

    test_start_index = int(getattr(config, "test_start_index", 0))
    test_num_rides = int(getattr(config, "test_num_rides", 20))

    if manifest_cache and Path(manifest_cache).exists():
        cached = torch.load(manifest_cache, map_location="cpu")
        all_rides = cached if isinstance(cached, list) else cached.get("rides", cached)
        logging.info("Loaded manifest from cache: %d rides (%s)", len(all_rides), manifest_cache)
    else:
        all_rides = build_ride_manifest(
            encoded_root=config.encoded_root,
            caption_root=config.caption_root,
            min_ride_frames=min_ride_frames,
        )

    train_start = test_start_index + test_num_rides
    train_rides = all_rides[train_start:] + all_rides[:test_start_index]
    train_rides = train_rides[video_offset : video_offset + num_videos]

    dataset = ZarrRideDataset.from_manifest(
        rides_data=train_rides,
        motion_root=motion_root,
        ss_vae_checkpoint=ss_vae_ckpt,
    )

    action_dims = list(getattr(config, "action_dims", [2, 7]))
    videos = []
    for i in range(min(num_videos, len(dataset))):
        ride = dataset[i]
        zarr_path = ride["zarr_path"]
        prompt_embeds = ride["prompt_embeds"]
        n_latent_frames = ride["n_latent_frames"]

        full_latents = ZarrRideDataset.load_latent_chunk(zarr_path, 0, min_ride_frames)
        context_latents = full_latents[:streaming_chunk_size]
        target_latents = full_latents[context_frames:]

        z_actions = dataset.encode_z_actions_window(
            zarr_path, n_latent_frames, context_frames, min_ride_frames,
        )
        z_actions_sliced = z_actions[..., action_dims]

        videos.append({
            "context_latents": context_latents,
            "target_latents": target_latents,
            "prompt_embeds": prompt_embeds,
            "z_actions_sliced": z_actions_sliced,
            "z_actions_full": z_actions,
            "name": Path(zarr_path).stem,
        })
        logging.info("Loaded video %d/%d: %s (%d latent frames)",
                      i + 1, num_videos, Path(zarr_path).stem, n_latent_frames)

    return videos, action_dims


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_conditional(prompt_embeds, z_noisy, z_clean, num_frames,
                      action_projection, action_token_projection):
    conditional = {"prompt_embeds": prompt_embeds}
    if action_projection is not None:
        conditional["_action_modulation"] = action_projection(z_noisy, num_frames=num_frames)
        conditional["_action_modulation_clean"] = action_projection(z_clean, num_frames=num_frames)
    if action_token_projection is not None:
        conditional["_action_tokens"] = action_token_projection(z_noisy)
        conditional["_action_tokens_clean"] = action_token_projection(z_clean)
    return conditional


def chunk_z_to_blocks(z_per_frame: torch.Tensor, num_blocks: int = NUM_BLOCKS,
                      block_size: int = BLOCK_SIZE) -> torch.Tensor:
    """Mean-pool per-frame z [F, D] into per-block z [num_blocks, D]."""
    trimmed = z_per_frame[: num_blocks * block_size]
    return trimmed.reshape(num_blocks, block_size, -1).mean(dim=1)


# ---------------------------------------------------------------------------
# Per-video chart generation
# ---------------------------------------------------------------------------

def generate_video_charts(
    vid_name: str,
    vid_dir: Path,
    steps: list[int],
    step_z: np.ndarray,
    gt_clean_z: np.ndarray,
    gt_noisy_z: np.ndarray,
    dataset_z: np.ndarray,
):
    """Generate 14 charts for one video (7 blocks x 2 dims).

    Args:
        step_z:     [num_steps, num_blocks, 2] — raw teacher_z per step.
        gt_clean_z: [num_blocks, 2] — teacher_z from clean GT latents.
        gt_noisy_z: [num_blocks, 2] — teacher_z from GT latents + t=25 noise.
        dataset_z:  [num_blocks, 2] — ground truth z_actions from the dataset.
    """
    vid_dir.mkdir(parents=True, exist_ok=True)

    for block_idx in range(gt_clean_z.shape[0]):
        for dim_idx, dim_name in enumerate(DIM_NAMES):
            bar_vals = step_z[:, block_idx, dim_idx]
            gt_clean_val = gt_clean_z[block_idx, dim_idx]
            gt_noisy_val = gt_noisy_z[block_idx, dim_idx]
            dataset_val = dataset_z[block_idx, dim_idx]

            fig, ax = plt.subplots(figsize=(16, 5))

            ax.bar(steps, bar_vals, color="#4C72B0", alpha=0.85, width=0.8,
                   label=f"pred_x0 teacher {dim_name}")

            ax.axhline(y=dataset_val, color="#C44E52", linewidth=2, linestyle="-",
                        label=f"Dataset {dim_name} = {dataset_val:.4f}")
            ax.axhline(y=gt_clean_val, color="#DD8452", linewidth=2, linestyle="--",
                        label=f"GT clean {dim_name} = {gt_clean_val:.4f}")
            ax.axhline(y=gt_noisy_val, color="#55A868", linewidth=2, linestyle=":",
                        label=f"GT+t=25 {dim_name} = {gt_noisy_val:.4f}")

            ax.set_xlabel("Denoising step", fontsize=12)
            ax.set_ylabel(f"Teacher {dim_name} value", fontsize=12)
            ax.set_title(f"{vid_name} — block {block_idx} (frames {block_idx*3}-{block_idx*3+2}) — {dim_name}",
                         fontsize=13)
            ax.legend(fontsize=10, loc="best")
            ax.set_xticks(steps[::5])
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()

            out_path = vid_dir / f"block{block_idx}_{dim_name}.png"
            fig.savefig(str(out_path), dpi=120)
            plt.close(fig)

    logging.info("  Saved 14 charts to %s", vid_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Motion Habitat: probe action pipeline across timesteps")
    parser.add_argument("--config", type=str, default="configs/causal_lora_diffusion_teacher.yaml")
    parser.add_argument("--checkpoint", type=str,
                        default="logs/causal_lora_teacher/causal_lora_step0003150.pt")
    parser.add_argument("--num_videos", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="testing/motion_habitat_out")
    parser.add_argument("--video_offset", type=int, default=0)
    parser.add_argument("--manifest_cache", type=str,
                        default="logs/causal_lora_teacher/.ride_manifest.pt")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    config = OmegaConf.load(args.config)

    context_frames = int(getattr(config, "context_frames", 3))
    streaming_chunk_size = int(getattr(config, "streaming_chunk_size", 21))
    action_critic_dims = list(getattr(config, "action_critic_dims", [2, 7]))
    action_critic_noise_timestep = int(getattr(config, "action_critic_noise_timestep", 25))

    # ---- Load model ----
    logging.info("Building diffusion model ...")
    wrapper, train_scheduler, action_projection, action_token_projection = build_model(
        config, args.checkpoint, device,
    )

    # ---- Load frozen evaluators ----
    logging.info("Building frozen evaluator modules ...")
    frozen_vae, frozen_cotracker, frozen_ss_vae, ss_vae_scale = build_frozen_evaluators(device)

    # ---- Load data ----
    logging.info("Loading %d training videos (offset=%d) ...", args.num_videos, args.video_offset)
    videos, action_dims = load_videos(
        config, args.num_videos, device,
        video_offset=args.video_offset, manifest_cache=args.manifest_cache,
    )

    # ---- Inference scheduler ----
    infer_scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    infer_scheduler.set_timesteps(num_inference_steps=args.num_steps, denoising_strength=1.0)
    infer_scheduler.sigmas = infer_scheduler.sigmas.to(device)

    num_steps = len(infer_scheduler.timesteps)
    steps_list = list(range(num_steps))
    logging.info("Denoising with %d ODE steps", num_steps)

    pipeline_kwargs = dict(
        frozen_vae=frozen_vae,
        frozen_cotracker=frozen_cotracker,
        frozen_ss_vae=frozen_ss_vae,
        ss_vae_scale=ss_vae_scale,
        action_critic_dims=action_critic_dims,
        device=device,
    )

    z_dim = len(action_critic_dims)

    for vid_idx, vid in enumerate(videos):
        vid_name = vid["name"]
        global_vid_idx = args.video_offset + vid_idx
        logging.info("=" * 60)
        logging.info("Video %d (global %d): %s", vid_idx + 1, global_vid_idx, vid_name)
        logging.info("=" * 60)

        context_latents = vid["context_latents"].unsqueeze(0).to(device)
        target_latents = vid["target_latents"].unsqueeze(0).to(device)
        prompt_embeds = vid["prompt_embeds"].unsqueeze(0).to(device)
        z_actions_sliced = vid["z_actions_sliced"].unsqueeze(0).to(device)
        z_actions_full = vid["z_actions_full"].unsqueeze(0).to(device)

        # Dataset ground truth z per block: mean-pool [21, 2] -> [7, 2]
        dataset_z_blocks = chunk_z_to_blocks(z_actions_sliced[0]).cpu().numpy()

        # ---- Build action conditioning ----
        with torch.no_grad():
            z_noisy = z_actions_sliced
            z_clean = z_actions_full[..., action_dims][:, :streaming_chunk_size]
            conditional = build_conditional(
                prompt_embeds, z_noisy, z_clean, streaming_chunk_size,
                action_projection, action_token_projection,
            )

        # ---- GT baselines ----
        logging.info("  Computing GT baselines ...")

        gt_clean_list = run_frozen_action_pipeline(
            target_latents, scheduler=None, noise_timestep=0, **pipeline_kwargs,
        )
        gt_noisy_list = run_frozen_action_pipeline(
            target_latents, scheduler=train_scheduler,
            noise_timestep=action_critic_noise_timestep, **pipeline_kwargs,
        )

        gt_clean_z = gt_clean_list[0].cpu().numpy()
        gt_noisy_z = gt_noisy_list[0].cpu().numpy()
        n_blocks = min(gt_clean_z.shape[0], NUM_BLOCKS)

        gt_clean_z = gt_clean_z[:n_blocks]
        gt_noisy_z = gt_noisy_z[:n_blocks]
        dataset_z_blocks = dataset_z_blocks[:n_blocks]

        logging.info("  GT clean  z: %s", gt_clean_z.tolist())
        logging.info("  GT+t=25   z: %s", gt_noisy_z.tolist())
        logging.info("  Dataset   z: %s", dataset_z_blocks.tolist())

        # ---- Denoising loop: collect raw z per step ----
        step_z = np.zeros((num_steps, n_blocks, z_dim), dtype=np.float64)

        B = 1
        C, H, W = target_latents.shape[2], target_latents.shape[3], target_latents.shape[4]
        num_frames = streaming_chunk_size
        latents = torch.randn([B, num_frames, C, H, W], dtype=torch.float32, device=device)

        with torch.no_grad():
            for step_idx, t in enumerate(infer_scheduler.timesteps):
                t0 = time.perf_counter()
                timestep = t * torch.ones([B, num_frames], device=device, dtype=torch.float32)

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    flow_pred, pred_x0 = wrapper(
                        latents, conditional, timestep,
                        clean_x=context_latents, aug_t=None,
                    )

                sz_list = run_frozen_action_pipeline(
                    pred_x0.float(), scheduler=None, noise_timestep=0, **pipeline_kwargs,
                )
                sz = sz_list[0].cpu().numpy()
                step_z[step_idx, :min(sz.shape[0], n_blocks)] = sz[:n_blocks]

                latents = infer_scheduler.step(
                    flow_pred.flatten(0, 1), timestep.flatten(0, 1), latents.flatten(0, 1),
                ).unflatten(dim=0, sizes=flow_pred.shape[:2])

                elapsed = time.perf_counter() - t0
                logging.info(
                    "  step %2d/%d | t=%7.1f | z=%s | %.1fs",
                    step_idx, num_steps, t.item(),
                    np.array2string(sz[:n_blocks], precision=3, separator=", "),
                    elapsed,
                )

        # ---- Save raw values alongside charts ----
        vid_dir = output_dir / f"video_{global_vid_idx:02d}_{vid_name}"
        vid_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(vid_dir / "raw_values.npz"),
            step_z=step_z,
            gt_clean_z=gt_clean_z,
            gt_noisy_z=gt_noisy_z,
            dataset_z=dataset_z_blocks,
            steps=np.array(steps_list),
            timesteps=np.array([infer_scheduler.timesteps[i].item() for i in range(num_steps)]),
        )

        # ---- Generate 14 charts ----
        generate_video_charts(
            vid_name, vid_dir, steps_list,
            step_z, gt_clean_z, gt_noisy_z, dataset_z_blocks,
        )

        torch.cuda.empty_cache()

    logging.info("=" * 60)
    logging.info("All done. Charts in %s", output_dir)
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
