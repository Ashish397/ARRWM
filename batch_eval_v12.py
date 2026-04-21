#!/usr/bin/env python3
"""Batch eval: run v12's full-ride ODE generation on Madrid/Rome/Stockholm daytime rides.

Uses ChainPipeline from eval_chain (proven working) for model setup.

For each 21-frame window:
  1. Saves ODE trajectory snapshots at steps [0, 18, 36, 40, 44, 46, -1] as .pt
     to /projects/u6ex/fbots/frodobots_lmdb/ for later LMDB creation + distillation.
  2. Saves final pred_x0 as zarr to /projects/u6ex/fbots/frodobots_noise/
     matching frodobots_encoded format, for noise analysis.

4 independent GPU processes, each handling a shard. Randomly shuffled for city coverage.
"""

import sys
import os
import logging
import json
import glob
import random
import numpy as np
import torch
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

NOISE_ROOT = "/projects/u6ex/fbots/frodobots_noise"
LMDB_ROOT = "/projects/u6ex/fbots/frodobots_lmdb"

SNAPSHOT_STEPS = [0, 18, 36, 40, 44, 46, -1]

# Import constants from eval_chain
from utils.eval_chain import (
    NUM_FRAME_PER_BLOCK, CONTEXT_FRAMES, NUM_FRAMES, EVAL_STEPS,
    CRITIC_ACTION_DIMS, ChainPipeline, _unwrap_manifest_rides,
)

STREAM_LATENT_SPAN = CONTEXT_FRAMES + NUM_FRAMES  # 24
LATENT_START_OFFSET = 0  # we iterate through full rides


def find_daytime_city_rides(caption_root, cities=("Madrid", "Rome", "Stockholm")):
    ts_to_city = {}
    for fpath in glob.glob(os.path.join(caption_root, "output_rides_*/ride_*/ride_*_video_captions_OpenGVLab_InternVL3_8B.json")):
        try:
            with open(fpath) as f:
                d = json.load(f)
            loc = d.get("metadata", {}).get("location", "")
            if loc not in cities:
                continue
            local_time = d.get("metadata", {}).get("local_time", "")
            if local_time:
                hour = int(local_time.split(":")[0])
                if hour < 6 or hour >= 20:
                    continue
            chunks = d.get("chunks", [])
            is_dark = False
            for chunk in chunks[:2]:
                caption = chunk.get("caption", "").lower()
                if any(w in caption for w in ["night", "dark", "nighttime", "evening darkness"]):
                    is_dark = True
                    break
            if is_dark:
                continue
            ride_name = fpath.split("/")[-2]
            ts = ride_name.split("_")[-1]
            ts_to_city[ts] = loc
        except Exception:
            pass
    return ts_to_city


def save_noise_zarr(out_path, latents_list, source_attrs, city):
    import zarr
    from numcodecs import Blosc
    all_latents = np.concatenate(latents_list, axis=0)
    comp = Blosc(cname="zstd", clevel=1, shuffle=Blosc.BITSHUFFLE)
    g = zarr.open_group(str(out_path), mode="w")
    g.create_dataset(
        "latents", data=all_latents,
        chunks=(32, *all_latents.shape[1:]),
        dtype=np.float16, compressor=comp, overwrite=True,
    )
    for k, v in source_attrs.items():
        g.attrs[k] = v
    g.attrs["generated_by"] = "v12_batch_eval_full_ride"
    g.attrs["city"] = city
    g.attrs["generated_latent_frames"] = int(all_latents.shape[0])


def generate_with_trajectory(pipe, conditional, clean_x):
    """Run ODE eval and return final latents + trajectory snapshots.

    Based on ChainPipeline.generate() but captures intermediate states.

    Returns:
        final_latents: [1, NUM_FRAMES, C, H, W]
        trajectory: [len(SNAPSHOT_STEPS), NUM_FRAMES, C, H, W] float16
        step_indices: list of ints
    """
    from utils.scheduler import FlowMatchScheduler

    sched = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    sched.set_timesteps(num_inference_steps=EVAL_STEPS, denoising_strength=1.0)
    sched.sigmas = sched.sigmas.to(pipe.device)

    B = clean_x.shape[0]
    C, H, W = clean_x.shape[2], clean_x.shape[3], clean_x.shape[4]
    lat = torch.randn([B, NUM_FRAMES, C, H, W], dtype=torch.float32, device=pipe.device)

    n_steps = len(sched.timesteps)
    snapshot_indices = set()
    for s in SNAPSHOT_STEPS:
        if s == -1:
            snapshot_indices.add(n_steps)
        else:
            snapshot_indices.add(s)

    snapshots = []

    if 0 in snapshot_indices:
        snapshots.append((0, lat[0].cpu().half()))

    for step_idx, t in enumerate(sched.timesteps):
        ts = t * torch.ones([B, NUM_FRAMES], device=pipe.device, dtype=torch.float32)
        with torch.amp.autocast(device_type="cuda", dtype=pipe.dtype):
            out = pipe.wrapper(lat, conditional, ts, clean_x=clean_x, aug_t=None)
            flow = out[0]
        lat = sched.step(
            flow.flatten(0, 1), ts.flatten(0, 1), lat.flatten(0, 1),
        ).unflatten(dim=0, sizes=flow.shape[:2])

        ode_step = step_idx + 1
        if ode_step in snapshot_indices:
            snapshots.append((ode_step, lat[0].cpu().half()))

    if n_steps in snapshot_indices and (not snapshots or snapshots[-1][0] != n_steps):
        snapshots.append((n_steps, lat[0].cpu().half()))

    snapshots.sort(key=lambda x: x[0])
    trajectory = torch.stack([s[1] for s in snapshots], dim=0)
    step_indices = [s[0] for s in snapshots]

    return lat, trajectory, step_indices


def main():
    from omegaconf import OmegaConf
    from utils.zarr_dataset import ZarrRideDataset
    import zarr as zarr_lib

    gpu_rank = int(os.environ.get("GPU_RANK", 0))
    num_gpus = int(os.environ.get("NUM_GPUS", 4))
    device = torch.device("cuda:0")

    config_path = "configs/causal_lora_diffusion_teacher.yaml"
    manifest_path = "logs/z_critic_v10_state_tokens/.ride_manifest.pt"
    ckpt_path = "logs/z_critic_v12_probe_fixes/causal_lora_step0003250.pt"

    cfg = OmegaConf.load(config_path)
    motion_root = str(cfg.get("motion_root", ""))
    if "u6ej" in motion_root:
        motion_root = motion_root.replace("u6ej", "u6ex")
    ss_vae_ckpt = str(cfg.get("ss_vae_checkpoint", "action_query/checkpoints/ss_vae_8free.pt"))
    action_dims = list(cfg.get("action_dims", [2, 7]))

    os.environ["HF_HOME"] = "/scratch/u6ex/as1748.u6ex/frodobots/hf_cache"

    # Build pipeline (same as eval_chain)
    log.info("GPU %d: Building pipeline...", gpu_rank)
    pipe = ChainPipeline(device)
    pipe.build(config_path, use_action_tokens=True)
    step = pipe.load_checkpoint(ckpt_path, {
        "has_critic": True, "has_adaln": True, "has_action_tokens": True,
    })
    log.info("GPU %d: Pipeline ready (checkpoint step %s)", gpu_rank, step)

    # Load manifest and find city rides
    log.info("GPU %d: Loading manifest...", gpu_rank)
    manifest = torch.load(manifest_path, map_location="cpu", weights_only=False)
    rides_all, _ = _unwrap_manifest_rides(manifest)

    # Fix u6ej paths
    for r in rides_all:
        zp = r["zarr_path"]
        if "u6ej" in zp:
            r["zarr_path"] = zp.replace("/projects/u6ej/fbots/frodobots_encoded",
                                        "/projects/u6ex/fbots/frodobots_encoded")

    caption_root = "/projects/u6ex/fbots/frodobots_captions/train"
    ts_to_city = find_daytime_city_rides(caption_root)

    # Build ride-to-manifest mapping
    eval_rides = []
    for i, r in enumerate(rides_all):
        ts = os.path.basename(r["zarr_path"]).replace(".zarr", "")
        if ts in ts_to_city:
            eval_rides.append((i, ts, ts_to_city[ts], r))
    del manifest

    random.seed(42)
    random.shuffle(eval_rides)

    my_rides = eval_rides[gpu_rank::num_gpus]
    log.info("GPU %d: %d city rides total, shard %d/%d (%d rides)",
             gpu_rank, len(eval_rides), gpu_rank, num_gpus, len(my_rides))

    # Build a ZarrRideDataset for motion encoding
    rides_for_ds = []
    for _, ts, city, r in my_rides:
        zp = r["zarr_path"]
        g = zarr_lib.open_group(zp, mode="r")
        n_lat = g["latents"].shape[0]
        rides_for_ds.append({
            "zarr_path": zp,
            "prompt_embeds": r["prompt_embeds"],
            "attrs": r.get("attrs", dict(g.attrs)),
            "n_latent_frames": n_lat,
        })

    z_ds = ZarrRideDataset.from_manifest(
        rides_data=rides_for_ds,
        motion_root=motion_root,
        ss_vae_checkpoint=ss_vae_ckpt,
        device="cpu",
        ss_vae_device=str(device),
    )
    log.info("GPU %d: ZarrRideDataset ready (%d rides)", gpu_rank, len(z_ds))

    os.makedirs(NOISE_ROOT, exist_ok=True)
    os.makedirs(LMDB_ROOT, exist_ok=True)

    done_rides = 0
    done_windows = 0
    skipped = 0

    for ride_i, (manifest_idx, ts, city, ride) in enumerate(my_rides):
        noise_zarr = os.path.join(NOISE_ROOT, f"{ts}.zarr")
        if os.path.exists(noise_zarr):
            skipped += 1
            continue

        try:
            zarr_path = ride["zarr_path"]
            g = zarr_lib.open_group(zarr_path, mode="r")
            n_lat = g["latents"].shape[0]
            source_attrs = dict(g.attrs)

            if n_lat < STREAM_LATENT_SPAN:
                skipped += 1
                continue

            prompt_embeds = ride["prompt_embeds"].unsqueeze(0).to(device, dtype=pipe.dtype)

            pred_x0_chunks = []
            offset = 0
            window_idx = 0

            while offset + STREAM_LATENT_SPAN <= n_lat:
                # Load latents
                lat_np = g["latents"][offset:offset + NUM_FRAMES]
                clean_x = torch.from_numpy(lat_np.astype(np.float32)).unsqueeze(0).to(device)

                # Load actions via ZarrRideDataset
                z_win = z_ds.encode_z_actions_window(
                    zarr_path, n_lat, offset, offset + STREAM_LATENT_SPAN,
                )
                z_clean = z_win[:NUM_FRAMES, action_dims].unsqueeze(0).to(device, dtype=pipe.dtype)
                z_noisy = z_win[CONTEXT_FRAMES:, action_dims].unsqueeze(0).to(device, dtype=pipe.dtype)

                # Build conditional (same as eval_chain)
                cond = pipe.build_conditional(
                    prompt_embeds, z_noisy, z_clean,
                    use_adaln=True, use_tokens=True,
                )

                with torch.no_grad():
                    gen_latents, trajectory, step_indices = generate_with_trajectory(
                        pipe, cond, clean_x,
                    )

                # Save pred_x0 for noise zarr
                pred_x0_chunks.append(gen_latents[0].cpu().numpy().astype(np.float16))

                # Save ODE trajectory as .pt for LMDB creation
                pt_data = {
                    "trajectory": trajectory,
                    "step_indices": step_indices,
                    "zarr_path": zarr_path,
                    "ride_ts": ts,
                    "city": city,
                    "window_offset": offset,
                    "window_idx": window_idx,
                    "n_latent_frames": n_lat,
                }
                pt_path = os.path.join(LMDB_ROOT, f"{ts}_w{window_idx:04d}.pt")
                torch.save(pt_data, pt_path)

                done_windows += 1
                offset += NUM_FRAMES
                window_idx += 1

            if pred_x0_chunks:
                save_noise_zarr(noise_zarr, pred_x0_chunks, source_attrs, city)
                done_rides += 1

                if done_rides % 5 == 0:
                    log.info("GPU %d: %d rides (%d windows), %d skipped (latest: %s/%s)",
                             gpu_rank, done_rides, done_windows, skipped, city, ts)

        except Exception as exc:
            log.warning("GPU %d: failed %s/%s: %s", gpu_rank, city, ts, exc)
            import traceback
            traceback.print_exc()
            skipped += 1

    log.info("GPU %d finished: %d rides (%d windows), %d skipped",
             gpu_rank, done_rides, done_windows, skipped)


if __name__ == "__main__":
    main()
