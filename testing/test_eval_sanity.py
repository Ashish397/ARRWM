#!/usr/bin/env python3
"""Sanity check: run v10/v11/v12 eval on the EXACT same rides v12 trained on.

Uses the original frodobots_encoded zarrs (not WEU additions).
Includes ground truth, context, action overlays.
"""

import sys
import os
import subprocess
import tempfile
import numpy as np
import torch

sys.path.insert(0, ".")

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# The original 2880 zarrs v12 trained on start from 20240131.
# First 20 (after validation) are the eval rides.
# We hardcode a few representative ones including the ones
# that would have been used at various training steps.
EVAL_RIDE_INDICES = [0, 5, 10, 15]  # ride_idx = step % 20


def frames_to_mp4(frames, path, fps=5.0):
    h, w = frames.shape[1], frames.shape[2]
    with tempfile.NamedTemporaryFile(suffix=".rgb", delete=False) as tmp:
        tmp.write(frames.tobytes())
        tmp_path = tmp.name
    subprocess.run([
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", str(fps),
        "-i", tmp_path, "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", str(path),
    ], capture_output=True)
    os.unlink(tmp_path)


def main():
    from omegaconf import OmegaConf
    from trainer.causal_diffusion_teacher_train import (
        CausalLoRADiffusionTrainer,
        _annotate_action_video,
        _chunk_actions,
    )
    from utils.zarr_dataset import ZarrRideDataset, build_ride_manifest
    from torch.nn.parallel import DistributedDataParallel as DDP
    from peft import set_peft_model_state_dict

    device = "cuda:0"

    # Build config pointing to original data
    config = OmegaConf.load("configs/causal_lora_diffusion_teacher.yaml")
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    config.no_save = True
    config.no_visualize = True
    config.disable_wandb = True
    config.auto_resume = False
    config.use_one_logger = False
    config.max_iters = 0
    config.encoded_root = "/projects/u6ex/fbots/frodobots_encoded"
    config.caption_root = "/projects/u6ex/fbots/frodobots_captions/train"
    config.motion_root = "/projects/u6ex/fbots/frodobots_motion"
    config.pretrained_lora_ckpt = "/scratch/u6ex/as1748.u6ex/ARRWM/logs/z_critic_v12_probe_fixes/causal_lora_step0003250.pt"
    config.logdir = "/tmp/sanity_eval"

    os.environ["HF_HOME"] = "/scratch/u6ex/as1748.u6ex/frodobots/hf_cache"

    # Build manifest matching v12's original data
    # v12 had 2880 zarrs starting from 20240131. Filter out our new additions.
    logging.info("Building manifest from original data...")
    all_rides = build_ride_manifest(
        encoded_root=config.encoded_root,
        caption_root=config.caption_root,
        min_ride_frames=24,
        motion_root=config.motion_root,
    )
    # Filter to only original zarrs (timestamp >= 20240131)
    original_rides = [r for r in all_rides if os.path.basename(r["zarr_path"]) >= "20240131"]
    logging.info("Original rides: %d (filtered from %d)", len(original_rides), len(all_rides))

    # Eval rides = first 20
    eval_rides = original_rides[:20]
    logging.info("Eval rides:")
    for i, r in enumerate(eval_rides):
        logging.info("  %d: %s", i, os.path.basename(r["zarr_path"]))

    # Build trainer
    trainer = CausalLoRADiffusionTrainer(config)
    logging.info("Trainer ready")

    # Replace eval dataset with the correct rides
    ss_vae_ckpt = "action_query/checkpoints/ss_vae_8free.pt"
    trainer.eval_dataset = ZarrRideDataset.from_manifest(
        rides_data=eval_rides,
        motion_root=config.motion_root,
        ss_vae_checkpoint=ss_vae_ckpt,
        _share_ss_vae=trainer.dataset,
    )
    logging.info("Eval dataset replaced: %d rides", len(trainer.eval_dataset))

    checkpoints = [
        ("/scratch/u6ex/as1748.u6ex/ARRWM/logs/z_critic_v10_state_tokens/causal_lora_step0001650.pt", "v10_step1650"),
        ("/scratch/u6ex/as1748.u6ex/ARRWM/logs/z_critic_v11_cross_attn_probes/causal_lora_step0000850.pt", "v11_step850"),
        ("/scratch/u6ex/as1748.u6ex/ARRWM/logs/z_critic_v12_probe_fixes/causal_lora_step0003250.pt", "v12_step3250"),
    ]

    wrapper = trainer.model.module if isinstance(trainer.model, DDP) else trainer.model

    for ckpt_path, label in checkpoints:
        if not os.path.exists(ckpt_path):
            logging.warning("Not found: %s", ckpt_path)
            continue

        logging.info("=== %s ===", label)
        out_dir = f"vis/sanity/{label}"
        os.makedirs(out_dir, exist_ok=True)

        # Load checkpoint weights
        ckpt = torch.load(ckpt_path, map_location="cpu")
        set_peft_model_state_dict(wrapper.model, ckpt["lora"])
        logging.info("Loaded LoRA (%d keys)", len(ckpt["lora"]))

        if trainer.action_critic is not None and "action_critic" in ckpt:
            critic_mod = trainer.action_critic.module if isinstance(trainer.action_critic, DDP) else trainer.action_critic
            critic_mod.load_state_dict(ckpt["action_critic"], strict=False)

        trainer._offload_training_state()
        wrapper.eval()
        causal_model = wrapper.model
        if hasattr(causal_model, "base_model"):
            causal_model = causal_model.base_model.model
        saved_mask = getattr(causal_model, "block_mask", None)
        causal_model.block_mask = None

        critic_mod = None
        if trainer.action_critic is not None:
            critic_mod = trainer.action_critic.module if isinstance(trainer.action_critic, DDP) else trainer.action_critic
            critic_mod.eval()

        num_frames = trainer.streaming_chunk_size
        cf = trainer.context_frames
        window_total = num_frames + cf

        for ride_idx in EVAL_RIDE_INDICES:
            if ride_idx >= len(trainer.eval_dataset):
                continue
            ride = trainer.eval_dataset[ride_idx]
            zarr_path = ride["zarr_path"]
            ts = os.path.basename(zarr_path).replace(".zarr", "")
            n_lat = ride["n_latent_frames"]
            if n_lat < window_total:
                continue

            logging.info("[%s] Ride %d: %s", label, ride_idx, ts)

            prompt_embeds = ride["prompt_embeds"].unsqueeze(0).to(device, dtype=trainer.dtype)
            full_latents = ZarrRideDataset.load_latent_chunk(zarr_path, 0, window_total)
            full_latents = full_latents.unsqueeze(0).to(device, dtype=torch.float32)

            z_actions = trainer.eval_dataset.encode_z_actions_window(
                zarr_path, n_lat, 0, window_total,
            ).unsqueeze(0).to(device, dtype=trainer.dtype)

            context_latents = full_latents[:, :num_frames]
            gt_target = full_latents[:, cf:]
            z_sliced = z_actions if trainer.action_dims is None else z_actions[..., trainer.action_dims]
            z_noisy = z_sliced[:, cf:]
            z_clean = z_sliced[:, :num_frames]
            target_action_z = z_actions[:, cf:][..., trainer.action_critic_dims][:, :num_frames]
            target_chunk = _chunk_actions(target_action_z, trainer.num_frame_per_block)

            conditional = trainer._build_conditional(prompt_embeds, z_noisy, z_clean, num_frames)

            prefix = f"ride_{ride_idx}_{ts}"

            with torch.no_grad():
                gen_latents, _ = trainer._generate_eval(wrapper, conditional, context_latents, num_frames)
                gen_vid = trainer._decode_latents(gen_latents)
                gt_vid = trainer._decode_latents(gt_target)
                context_vid = trainer._decode_latents(context_latents)

            frames_to_mp4(context_vid, os.path.join(out_dir, f"{prefix}_context.mp4"))
            frames_to_mp4(gt_vid, os.path.join(out_dir, f"{prefix}_ground_truth.mp4"))

            if critic_mod is not None:
                with torch.no_grad():
                    gen_motion, gen_tz = trainer._compute_teacher_visuals(gen_latents)
                    n_c = gen_tz.shape[1]
                    eval_t = torch.zeros(1, n_c, device=device)
                    gen_cz = critic_mod(gen_latents, eval_t, target_chunk[:, :n_c])[:, :n_c]
                gen_ann = _annotate_action_video(
                    gen_vid, gen_motion,
                    gen_tz[:, :, trainer.action_critic_dims],
                    gen_cz[:, :, trainer.action_critic_dims],
                    target_chunk[:, :n_c],
                    title=f"{label} gen ride {ride_idx}",
                )
                frames_to_mp4(gen_ann, os.path.join(out_dir, f"{prefix}_generated_annotated.mp4"))

                with torch.no_grad():
                    gt_motion, gt_tz = trainer._compute_teacher_visuals(gt_target)
                    n_c = gt_tz.shape[1]
                    gt_cz = critic_mod(gt_target, eval_t[:, :n_c], target_chunk[:, :n_c])[:, :n_c]
                gt_ann = _annotate_action_video(
                    gt_vid, gt_motion,
                    gt_tz[:, :, trainer.action_critic_dims],
                    gt_cz[:, :, trainer.action_critic_dims],
                    target_chunk[:, :n_c],
                    title=f"{label} GT ride {ride_idx}",
                )
                frames_to_mp4(gt_ann, os.path.join(out_dir, f"{prefix}_gt_annotated.mp4"))
            else:
                frames_to_mp4(gen_vid, os.path.join(out_dir, f"{prefix}_generated.mp4"))

            logging.info("[%s] Done ride %d", label, ride_idx)

        causal_model.block_mask = saved_mask
        trainer._restore_training_state()
        torch.cuda.empty_cache()

    logging.info("All done!")


if __name__ == "__main__":
    main()
