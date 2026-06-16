#!/usr/bin/env python3
"""gen_lmdb.py — Generate ODE-trajectory .pt files for 1500 random WEU windows.

For each sampled 24-latent window (3 context + 21 target) we run TWO generations:
  1. Clean   — real ride z-actions (z2, z7) on the noisy side.
  2. Counterfactual — z-actions transformed as:
        cf_z2 = -z2
        cf_z7 = -z7             if z7 <  -0.2
        cf_z7 = 1 - |z7|        if z7 >= -0.2

Outputs (paired filenames, different roots):
  /projects/u6ex/fbots/frodobots_lmdb/v14/{ts}_o{offset:05d}.pt             (clean)
  /projects/u6ex/fbots/frodobots_lmdb_counterfac/v14/{ts}_o{offset:05d}.pt  (cf)

Total target: 1500 × 2 = 3000 files.

Model  : v14-balanced-weunz (LoRA rank 256), checkpoint step 6600.
Dataset: WEU (western-Europe) manifest at logs/v14_balanced_weunz/.ride_manifest.pt.
Mode   : fully latent — no VAE decode, no CoTracker.

Launch (4 GPUs on one node):
    cd /scratch/u6ex/as1748.u6ex/ARRWM
    for R in 0 1 2 3; do
      GPU_RANK=$R NUM_GPUS=4 CUDA_VISIBLE_DEVICES=$R \
        python gen_lmdb.py > logs/gen_lmdb_r$R.out 2>&1 &
    done
    wait
"""

from __future__ import annotations

import os
import sys
import time
import random
import logging
import zlib
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("gen_lmdb")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG_PATH   = "configs/causal_lora_diffusion_teacher_v14.yaml"
MANIFEST_PATH = "logs/v14_balanced_weunz/.ride_manifest.pt"
CKPT_PATH     = "logs/v14_balanced_weunz/causal_lora_step0006600.pt"

LMDB_ROOT    = "/projects/u6ex/fbots/frodobots_lmdb/v14"
LMDB_CF_ROOT = "/projects/u6ex/fbots/frodobots_lmdb_counterfac/v14"

NUM_WINDOWS    = 1500
SNAPSHOT_STEPS = [0, 18, 36, 40, 44, 46, -1]
SAMPLE_SEED    = 20260419
CF_Z7_THRESH   = -0.2

from utils.eval_chain import (  # noqa: E402
    NUM_FRAME_PER_BLOCK,
    CONTEXT_FRAMES,
    NUM_FRAMES,
    EVAL_STEPS,
    ChainPipeline,
    _unwrap_manifest_rides,
)

STREAM_LATENT_SPAN = CONTEXT_FRAMES + NUM_FRAMES  # 24


# ---------------------------------------------------------------------------
# ODE generation that captures trajectory snapshots
# ---------------------------------------------------------------------------

def generate_with_trajectory(pipe, conditional, clean_x, *, noise_seed: int | None = None):
    """Run ODE denoising; return (final_latents, trajectory_snapshots, step_indices).

    Snapshots are taken at ODE steps in ``SNAPSHOT_STEPS`` (``-1`` = final step).
    If ``noise_seed`` is given, the initial latent is drawn from a seeded CUDA
    generator so that paired (clean / counterfactual) calls at the same window
    share identical initial noise — any downstream difference is then purely
    due to the action edit.
    """
    from utils.scheduler import FlowMatchScheduler

    sched = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    sched.set_timesteps(num_inference_steps=EVAL_STEPS, denoising_strength=1.0)
    sched.sigmas = sched.sigmas.to(pipe.device)

    B = clean_x.shape[0]
    C, H, W = clean_x.shape[2], clean_x.shape[3], clean_x.shape[4]
    lat_shape = [B, NUM_FRAMES, C, H, W]
    if noise_seed is not None:
        gen = torch.Generator(device=pipe.device)
        gen.manual_seed(int(noise_seed) & 0x7FFFFFFF)
        lat = torch.randn(
            lat_shape, dtype=torch.float32, device=pipe.device, generator=gen,
        )
    else:
        lat = torch.randn(
            lat_shape, dtype=torch.float32, device=pipe.device,
        )

    n_steps = len(sched.timesteps)
    snapshot_indices = set()
    for s in SNAPSHOT_STEPS:
        snapshot_indices.add(n_steps if s == -1 else int(s))

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

    snapshots.sort(key=lambda s: s[0])
    trajectory = torch.stack([s[1] for s in snapshots], dim=0)
    step_indices = [s[0] for s in snapshots]
    return lat, trajectory, step_indices


# ---------------------------------------------------------------------------
# Counterfactual transform
# ---------------------------------------------------------------------------

def window_noise_seed(ts: str, offset: int) -> int:
    """Stable seed per window, identical across runs and across processes.

    ``hash(str)`` is salted by PYTHONHASHSEED per-process, so we use crc32
    over a canonical ``ts:offset`` key instead. XOR with ``SAMPLE_SEED`` so
    re-running with a different global seed gives a fresh noise assignment.
    """
    key = f"{ts}:{offset}".encode()
    return (zlib.crc32(key) ^ SAMPLE_SEED) & 0x7FFFFFFF


def apply_counterfactual(z_noisy: torch.Tensor) -> torch.Tensor:
    """z_noisy: [..., 2] with (z2, z7) on the last axis.

    Element-wise transform:
        cf_z2 = -z2
        cf_z7 = -z7            if z7 <  CF_Z7_THRESH
        cf_z7 = 1 - |z7|       if z7 >= CF_Z7_THRESH
    """
    cf = z_noisy.clone()
    z2 = z_noisy[..., 0]
    z7 = z_noisy[..., 1]
    cf[..., 0] = -z2
    lower_mask = z7 < CF_Z7_THRESH
    cf[..., 1] = torch.where(lower_mask, -z7, 1.0 - z7.abs())
    return cf


# ---------------------------------------------------------------------------
# Sampling: 1500 unique (ride_idx, offset) pairs
# ---------------------------------------------------------------------------

def _fix_u6ej_path(zp: str) -> str:
    if "u6ej" in zp:
        return zp.replace(
            "/projects/u6ej/fbots/frodobots_encoded",
            "/projects/u6ex/fbots/frodobots_encoded",
        )
    return zp


def sample_windows(rides_all, num_windows, seed):
    """Pick ``num_windows`` unique (ride_idx, offset) pairs.

    Windows step by ``NUM_FRAMES`` within each ride (non-overlapping).  Offsets
    are drawn uniformly from the pool of all valid windows across all rides.
    """
    import zarr as zarr_lib

    pool = []
    for i, r in enumerate(rides_all):
        zp = _fix_u6ej_path(r["zarr_path"])
        try:
            g = zarr_lib.open_group(zp, mode="r")
            n_lat = int(g["latents"].shape[0])
        except Exception as exc:
            log.warning("skip %s: %s", zp, exc)
            continue
        if n_lat < STREAM_LATENT_SPAN:
            continue
        n_win = (n_lat - STREAM_LATENT_SPAN) // NUM_FRAMES + 1
        for w in range(n_win):
            pool.append((i, w * NUM_FRAMES, n_lat))

    log.info(
        "Candidate pool: %d non-overlapping windows across %d rides",
        len(pool), len(rides_all),
    )
    if len(pool) < num_windows:
        log.warning(
            "Pool (%d) smaller than requested (%d); using all.",
            len(pool), num_windows,
        )
        num_windows = len(pool)

    rng = random.Random(seed)
    return rng.sample(pool, num_windows)


def sample_windows_curated(rides_all, pool_json, num_windows):
    """Pick windows from a curated high-motion + backward pool JSON.

    ``pool_json`` is paper_assets/v14b_train_windows.json (built by
    utils/build_curated_pool.py): the most-motion-y windows + backward
    oversampler. We dedup (the pool oversamples backward), then take ALL
    backward windows (guarantee reverse coverage) + the top forward windows
    by score, up to ``num_windows``. Returns ``[(ride_idx, start, n_lat), ...]``
    matched to ``rides_all`` by zarr basename.
    """
    import json
    with open(pool_json) as f:
        wm = json.load(f)
    windows = wm["windows"] if isinstance(wm, dict) else wm
    # ride basename -> index in rides_all
    idx_by_base = {Path(_fix_u6ej_path(r["zarr_path"])).name: i for i, r in enumerate(rides_all)}
    # dedup by (basename, start), keep best score; track backward
    best = {}
    for w in windows:
        base = Path(w["zarr_path"]).name
        if base not in idx_by_base:
            continue
        key = (base, int(w["start"]))
        if key not in best or w["score"] > best[key]["score"]:
            best[key] = w
    uniq = list(best.values())
    bwd = [w for w in uniq if w.get("backward")]
    fwd = [w for w in uniq if not w.get("backward")]
    fwd.sort(key=lambda w: -w["score"])
    chosen = bwd + fwd[: max(0, num_windows - len(bwd))]
    log.info(
        "Curated pool %s: %d unique windows -> chose %d (%d backward + %d forward)",
        pool_json, len(uniq), len(chosen), len(bwd), len(chosen) - len(bwd),
    )
    out = []
    for w in chosen:
        i = idx_by_base[Path(w["zarr_path"]).name]
        out.append((i, int(w["start"]), int(w.get("n_latent_frames", 0))))
    return out


def _parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="Generate ODE-trajectory LMDB pairs")
    ap.add_argument("--config", default=CONFIG_PATH)
    ap.add_argument("--manifest", default=MANIFEST_PATH)
    ap.add_argument("--ckpt", default=CKPT_PATH)
    ap.add_argument("--lmdb_root", default=LMDB_ROOT)
    ap.add_argument("--lmdb_cf_root", default=LMDB_CF_ROOT)
    ap.add_argument("--num_windows", type=int, default=NUM_WINDOWS)
    ap.add_argument("--curated_pool", default=None,
                    help="if set, sample from this curated high-motion+backward pool JSON "
                         "instead of uniform-random windows")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    global CONFIG_PATH, MANIFEST_PATH, CKPT_PATH, LMDB_ROOT, LMDB_CF_ROOT, NUM_WINDOWS
    CONFIG_PATH = args.config
    MANIFEST_PATH = args.manifest
    CKPT_PATH = args.ckpt
    LMDB_ROOT = args.lmdb_root
    LMDB_CF_ROOT = args.lmdb_cf_root
    NUM_WINDOWS = args.num_windows

    from omegaconf import OmegaConf
    from utils.zarr_dataset import ZarrRideDataset
    import zarr as zarr_lib

    gpu_rank = int(os.environ.get("GPU_RANK", 0))
    num_gpus = int(os.environ.get("NUM_GPUS", 4))
    device = torch.device("cuda:0")

    cfg = OmegaConf.load(CONFIG_PATH)
    motion_root = str(cfg.get("motion_root", "")).replace("u6ej", "u6ex")
    ss_vae_ckpt = str(cfg.get(
        "ss_vae_checkpoint", "action_query/checkpoints/ss_vae_8free.pt",
    ))
    action_dims = list(cfg.get("action_dims", [2, 7]))

    os.environ.setdefault(
        "HF_HOME", "/scratch/u6ex/as1748.u6ex/frodobots/hf_cache",
    )

    os.makedirs(LMDB_ROOT,    exist_ok=True)
    os.makedirs(LMDB_CF_ROOT, exist_ok=True)

    log.info(
        "GPU %d/%d: loading manifest %s",
        gpu_rank, num_gpus, MANIFEST_PATH,
    )
    manifest = torch.load(MANIFEST_PATH, map_location="cpu", weights_only=False)
    rides_all, src_key = _unwrap_manifest_rides(manifest)
    for r in rides_all:
        r["zarr_path"] = _fix_u6ej_path(r["zarr_path"])
    log.info(
        "GPU %d: %d rides (manifest key=%s)",
        gpu_rank, len(rides_all), src_key,
    )

    if args.curated_pool:
        samples = sample_windows_curated(rides_all, args.curated_pool, NUM_WINDOWS)
        log.info("GPU %d: sampled %d curated high-motion+backward windows", gpu_rank, len(samples))
    else:
        samples = sample_windows(rides_all, NUM_WINDOWS, SAMPLE_SEED)
        log.info("GPU %d: sampled %d windows (seed=%d)", gpu_rank, len(samples), SAMPLE_SEED)

    my_samples = samples[gpu_rank::num_gpus]
    log.info(
        "GPU %d: %d windows in this shard (rank stride %d)",
        gpu_rank, len(my_samples), num_gpus,
    )
    del manifest

    # -----------------------------------------------------------------------
    # Build pipeline
    # -----------------------------------------------------------------------
    log.info("GPU %d: building v14 pipeline...", gpu_rank)
    pipe = ChainPipeline(device)
    pipe.build(CONFIG_PATH, use_action_tokens=True)
    step = pipe.load_checkpoint(CKPT_PATH, {
        "has_critic": True,
        "has_adaln": True,
        "has_action_tokens": True,
    })
    log.info("GPU %d: pipeline ready (checkpoint step %s)", gpu_rank, step)

    # -----------------------------------------------------------------------
    # ZarrRideDataset for z-action encoding (only over the rides this rank touches)
    # -----------------------------------------------------------------------
    unique_ride_idxs = sorted({ri for (ri, _off, _n) in my_samples})
    rides_for_ds = []
    for ri in unique_ride_idxs:
        r = rides_all[ri]
        zp = r["zarr_path"]
        g = zarr_lib.open_group(zp, mode="r")
        n_lat = int(g["latents"].shape[0])
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

    # -----------------------------------------------------------------------
    # Per-ride lazy cache (zarr group, prompt_embeds on GPU)
    # -----------------------------------------------------------------------
    ride_cache: dict = {}

    def get_ride(ri):
        info = ride_cache.get(ri)
        if info is not None:
            return info
        r = rides_all[ri]
        zp = r["zarr_path"]
        g = zarr_lib.open_group(zp, mode="r")
        attrs = r.get("attrs", dict(g.attrs))
        city = ""
        for key in ("location", "city", "ride_city"):
            if key in attrs:
                city = attrs[key]
                break
        info = {
            "zarr_path": zp,
            "zarr_group": g,
            "ts": os.path.basename(zp).replace(".zarr", ""),
            "prompt_embeds": r["prompt_embeds"].unsqueeze(0).to(
                device, dtype=pipe.dtype,
            ),
            "n_lat": int(g["latents"].shape[0]),
            "city": str(city),
            "attrs": attrs,
        }
        ride_cache[ri] = info
        return info

    # -----------------------------------------------------------------------
    # Generation loop
    # -----------------------------------------------------------------------
    done = 0
    skipped = 0
    failed = 0
    t_start = time.time()

    for idx, (ri, offset, _n_lat) in enumerate(my_samples):
        info = get_ride(ri)
        ts = info["ts"]
        fname = f"{ts}_o{offset:05d}.pt"
        clean_path = os.path.join(LMDB_ROOT,    fname)
        cf_path    = os.path.join(LMDB_CF_ROOT, fname)

        if os.path.exists(clean_path) and os.path.exists(cf_path):
            skipped += 1
            continue

        try:
            g = info["zarr_group"]
            lat_np = g["latents"][offset:offset + NUM_FRAMES]
            clean_x = torch.from_numpy(lat_np.astype(np.float32)).unsqueeze(0).to(device)

            z_win = z_ds.encode_z_actions_window(
                info["zarr_path"], info["n_lat"],
                offset, offset + STREAM_LATENT_SPAN,
            )
            z_clean = z_win[:NUM_FRAMES, action_dims].unsqueeze(0).to(
                device, dtype=pipe.dtype,
            )
            z_noisy = z_win[CONTEXT_FRAMES:, action_dims].unsqueeze(0).to(
                device, dtype=pipe.dtype,
            )

            # Shared initial-noise seed — clean and CF draw identical init latents
            # so the only source of trajectory divergence is the action edit.
            noise_seed = window_noise_seed(ts, offset)

            # ---- Clean pass ----
            if not os.path.exists(clean_path):
                cond = pipe.build_conditional(
                    info["prompt_embeds"], z_noisy, z_clean,
                    use_adaln=True, use_tokens=True,
                )
                with torch.no_grad():
                    _, traj_c, steps_c = generate_with_trajectory(
                        pipe, cond, clean_x, noise_seed=noise_seed,
                    )
                torch.save(
                    {
                        "trajectory":       traj_c,
                        "step_indices":     steps_c,
                        "zarr_path":        info["zarr_path"],
                        "ride_ts":          ts,
                        "city":             info["city"],
                        "window_offset":    offset,
                        "n_latent_frames":  info["n_lat"],
                        "counterfactual":   False,
                        "model_label":      "v14_balanced_weunz",
                        "checkpoint_step":  int(step) if isinstance(step, (int, float)) else 6600,
                        "noise_seed":       int(noise_seed),
                        "z_clean":          z_clean[0].detach().cpu().float(),
                        "z_noisy":          z_noisy[0].detach().cpu().float(),
                    },
                    clean_path,
                )

            # ---- Counterfactual pass (shares noise_seed with clean pass) ----
            if not os.path.exists(cf_path):
                z_cf = apply_counterfactual(z_noisy)
                cond_cf = pipe.build_conditional(
                    info["prompt_embeds"], z_cf, z_clean,
                    use_adaln=True, use_tokens=True,
                )
                with torch.no_grad():
                    _, traj_cf, steps_cf = generate_with_trajectory(
                        pipe, cond_cf, clean_x, noise_seed=noise_seed,
                    )
                torch.save(
                    {
                        "trajectory":       traj_cf,
                        "step_indices":     steps_cf,
                        "zarr_path":        info["zarr_path"],
                        "ride_ts":          ts,
                        "city":             info["city"],
                        "window_offset":    offset,
                        "n_latent_frames":  info["n_lat"],
                        "counterfactual":   True,
                        "cf_rule":          "z2=-z2; z7=-z7 if z7<-0.2 else 1-|z7|",
                        "model_label":      "v14_balanced_weunz",
                        "checkpoint_step":  int(step) if isinstance(step, (int, float)) else 6600,
                        "noise_seed":       int(noise_seed),
                        "z_clean":          z_clean[0].detach().cpu().float(),
                        "z_noisy":          z_noisy[0].detach().cpu().float(),
                        "z_noisy_cf":       z_cf[0].detach().cpu().float(),
                    },
                    cf_path,
                )

            done += 1
            if done % 10 == 0 or done == 1:
                elapsed = time.time() - t_start
                rate = done / max(elapsed, 1e-6)
                remaining_n = len(my_samples) - done - skipped - failed
                eta_min = remaining_n / max(rate, 1e-6) / 60.0
                log.info(
                    "GPU %d: %d/%d  done=%d skip=%d fail=%d  %.2f win/s  ETA %.1f min  last=%s@%d",
                    gpu_rank, idx + 1, len(my_samples),
                    done, skipped, failed, rate, eta_min, ts, offset,
                )

        except Exception as exc:
            log.warning("GPU %d: FAILED %s @%d: %s", gpu_rank, ts, offset, exc)
            import traceback
            traceback.print_exc()
            failed += 1

    elapsed = time.time() - t_start
    log.info(
        "GPU %d FINISHED: done=%d skipped=%d failed=%d  (%.1f min)",
        gpu_rank, done, skipped, failed, elapsed / 60.0,
    )


if __name__ == "__main__":
    main()
