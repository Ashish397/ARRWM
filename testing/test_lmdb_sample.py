#!/usr/bin/env python3
"""Visualise sample .pt ODE trajectory files — one per city.

Uses ChainPipeline from eval_chain for decoding and action overlays.

When --lmdb_dir points to counterfactual data, loads the SAME rides/windows
as the normal LMDB and applies the counterfactual transform to the target
overlay so the bars show what was actually commanded.

Usage:
    python testing/test_lmdb_sample.py
    python testing/test_lmdb_sample.py --lmdb_dir /projects/u6ex/fbots/frodobots_lmdb_counterfac --output_dir vis/lmdb_sample_counterfac
"""

import sys
import os
import subprocess
import argparse
import numpy as np
import torch
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

NORMAL_LMDB = "/projects/u6ex/fbots/frodobots_lmdb"


def frames_to_mp4(frames, path, fps=5.0):
    h, w = frames.shape[1], frames.shape[2]
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.communicate(input=frames.tobytes(), timeout=120)


def main():
    from utils.eval_chain import (
        ChainPipeline, annotate_video, NUM_FRAMES, CONTEXT_FRAMES,
        NUM_FRAME_PER_BLOCK, CRITIC_ACTION_DIMS,
    )
    from utils.zarr_dataset import ZarrRideDataset
    from omegaconf import OmegaConf
    import zarr as zarr_lib

    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb_dir", default=NORMAL_LMDB)
    parser.add_argument("--output_dir", default="vis/lmdb_sample")
    parser.add_argument(
        "--normal_lmdb", default=NORMAL_LMDB,
        help="Directory used to pick one ride per city (defaults to v12 root; "
             "for v14 pass /projects/u6ex/fbots/frodobots_lmdb/v14).",
    )
    args = parser.parse_args()

    is_counterfactual = "counterfac" in args.lmdb_dir
    normal_lmdb = args.normal_lmdb

    device = torch.device("cuda:0")
    os.environ["HF_HOME"] = "/scratch/u6ex/as1748.u6ex/frodobots/hf_cache"

    config_path = "configs/causal_lora_diffusion_teacher.yaml"
    ckpt_path = "logs/z_critic_v12_probe_fixes/causal_lora_step0003250.pt"

    cfg = OmegaConf.load(config_path)
    motion_root = str(cfg.get("motion_root", ""))
    if "u6ej" in motion_root:
        motion_root = motion_root.replace("u6ej", "u6ex")
    ss_vae_ckpt = str(cfg.get("ss_vae_checkpoint", "action_query/checkpoints/ss_vae_8free.pt"))
    action_dims = list(cfg.get("action_dims", [2, 7]))

    STREAM_LATENT_SPAN = CONTEXT_FRAMES + NUM_FRAMES

    log.info("Building pipeline...")
    pipe = ChainPipeline(device)
    pipe.build(config_path, use_action_tokens=True)
    pipe.load_checkpoint(ckpt_path, {
        "has_critic": True, "has_adaln": True, "has_action_tokens": True,
    })
    log.info("Pipeline ready")

    # Always pick samples from the NORMAL lmdb first (to get the same rides/windows)
    normal_pts = sorted([f for f in os.listdir(normal_lmdb) if f.endswith(".pt")])
    city_filenames = {}
    for pf in normal_pts:
        d = torch.load(os.path.join(normal_lmdb, pf), map_location="cpu")
        city = d.get("city") or "unknown"
        if city not in city_filenames:
            city_filenames[city] = pf
        if len(city_filenames) >= 3:
            break

    log.info("Selected (from normal LMDB): %s", city_filenames)

    # Now load the actual .pt files from the target lmdb_dir (same filenames)
    city_samples = {}
    for city, pf in city_filenames.items():
        pt_path = os.path.join(args.lmdb_dir, pf)
        if not os.path.exists(pt_path):
            log.warning("File %s not found in %s, skipping %s", pf, args.lmdb_dir, city)
            continue
        d = torch.load(pt_path, map_location="cpu")
        city_samples[city] = (pf, d)

    log.info("Loaded from %s: %s", args.lmdb_dir, {c: pf for c, (pf, _) in city_samples.items()})

    # Build ZarrRideDataset for motion encoding
    rides_for_ds = []
    for city, (pf, data) in city_samples.items():
        zp = data["zarr_path"]
        if "u6ej" in zp:
            zp = zp.replace("/projects/u6ej/fbots/frodobots_encoded",
                            "/projects/u6ex/fbots/frodobots_encoded")
        g = zarr_lib.open_group(zp, mode="r")
        rides_for_ds.append({
            "zarr_path": zp,
            "prompt_embeds": torch.zeros(1, 512, 4096),
            "attrs": dict(g.attrs),
            "n_latent_frames": g["latents"].shape[0],
        })

    z_ds = ZarrRideDataset.from_manifest(
        rides_data=rides_for_ds,
        motion_root=motion_root,
        ss_vae_checkpoint=ss_vae_ckpt,
        device="cpu",
        ss_vae_device=str(device),
    )

    def frame_to_chunk(fa):
        b, f, d = fa.shape
        return fa.reshape(b, f // NUM_FRAME_PER_BLOCK, NUM_FRAME_PER_BLOCK, d).mean(dim=2)

    for city, (pt_name, data) in city_samples.items():
        log.info("=== %s: %s ===", city, pt_name)
        out_dir = f"{args.output_dir}/{city}"
        os.makedirs(out_dir, exist_ok=True)

        zarr_path = data["zarr_path"]
        if "u6ej" in zarr_path:
            zarr_path = zarr_path.replace("/projects/u6ej/fbots/frodobots_encoded",
                                          "/projects/u6ex/fbots/frodobots_encoded")
        offset = data["window_offset"]
        n_lat = data["n_latent_frames"]

        # Load GT latents
        g = zarr_lib.open_group(zarr_path, mode="r")
        gt_full_np = g["latents"][offset:offset + STREAM_LATENT_SPAN]
        gt_context = torch.from_numpy(gt_full_np[:NUM_FRAMES].astype(np.float32)).unsqueeze(0).to(device)
        gt_target = torch.from_numpy(gt_full_np[CONTEXT_FRAMES:].astype(np.float32)).unsqueeze(0).to(device)

        # Load actions from ride
        z_win = z_ds.encode_z_actions_window(zarr_path, n_lat, offset, offset + STREAM_LATENT_SPAN)
        z_noisy = z_win[CONTEXT_FRAMES:, action_dims].unsqueeze(0).to(device, dtype=pipe.dtype)

        # Apply counterfactual transform if needed (so overlay shows actual commanded actions).
        # Prefer the saved ``z_noisy_cf`` when present (v14+ writes it); otherwise fall
        # back to the legacy v12 rule (``z7 = 1 - z7``, ``z2 = -z2``).
        if is_counterfactual:
            saved_cf = data.get("z_noisy_cf", None)
            if saved_cf is not None:
                z_noisy_cf = saved_cf.unsqueeze(0).to(device, dtype=pipe.dtype)
                log.info("Using saved z_noisy_cf (cf_rule=%r)",
                         data.get("cf_rule", "unspecified"))
            else:
                z_noisy_cf = z_noisy.clone()
                z_noisy_cf[..., 1] = 1.0 - z_noisy[..., 1]
                z_noisy_cf[..., 0] = -z_noisy[..., 0]
                log.info("Using legacy v12 cf rule (z7=1-z7, z2=-z2)")
            target_chunk = frame_to_chunk(z_noisy_cf)
            log.info(
                "Counterfactual target z2 range: %.3f to %.3f, z7 range: %.3f to %.3f",
                z_noisy_cf[..., 0].min().item(), z_noisy_cf[..., 0].max().item(),
                z_noisy_cf[..., 1].min().item(), z_noisy_cf[..., 1].max().item(),
            )
        else:
            target_chunk = frame_to_chunk(z_noisy)

        # Decode context
        log.info("Decoding context...")
        context_vid = pipe.decode_latents(gt_context)
        frames_to_mp4(context_vid, os.path.join(out_dir, "context_frames.mp4"))

        # Decode and annotate GT (always uses original actions for GT overlay)
        log.info("Decoding ground truth...")
        gt_vid = pipe.decode_latents(gt_target)
        frames_to_mp4(gt_vid, os.path.join(out_dir, "ground_truth.mp4"))

        gt_target_chunk = frame_to_chunk(z_noisy)  # GT always uses original actions
        with torch.no_grad():
            gt_motion, gt_tz = pipe.compute_teacher_visuals(gt_target)
            n_c = gt_tz.shape[1]
            gt_cz = pipe.run_critic(gt_target, gt_target_chunk[:, :n_c])
            gt_ann = annotate_video(
                gt_vid, gt_tz[:, :, CRITIC_ACTION_DIMS],
                gt_cz[:, :, CRITIC_ACTION_DIMS] if gt_cz is not None else None,
                gt_target_chunk[:, :n_c], gt_motion, f"GT {city}",
            )
        frames_to_mp4(gt_ann, os.path.join(out_dir, "ground_truth_annotated.mp4"))

        # Decode each trajectory snapshot
        trajectory = data["trajectory"]
        step_indices = data["step_indices"]

        for i, step_idx in enumerate(step_indices):
            log.info("Step %d...", step_idx)
            snap = trajectory[i].unsqueeze(0).to(device, dtype=torch.float32)
            snap_vid = pipe.decode_latents(snap)

            if step_idx > 0 and pipe.action_critic is not None:
                with torch.no_grad():
                    motion, tz = pipe.compute_teacher_visuals(snap)
                    n_c = tz.shape[1]
                    cz = pipe.run_critic(snap, target_chunk[:, :n_c])
                ann = annotate_video(
                    snap_vid, tz[:, :, CRITIC_ACTION_DIMS],
                    cz[:, :, CRITIC_ACTION_DIMS] if cz is not None else None,
                    target_chunk[:, :n_c], motion,
                    f"{city} step {step_idx}/{step_indices[-1]}",
                )
            else:
                ann = snap_vid

            frames_to_mp4(ann, os.path.join(out_dir, f"trajectory_step_{step_idx:02d}.mp4"))

        log.info("Done with %s", city)

    log.info("All done!")


if __name__ == "__main__":
    main()
