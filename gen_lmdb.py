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
CLEAN_ONLY     = False   # --clean_only: skip the counterfactual pass entirely

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
    # Window ranking score: older curated pools carry an explicit "score";
    # the v14d pool (paper_assets/v14d_train_windows.json) ranks by "motion"
    # only. Fall back motion -> 0.0 so either schema works (the v14d pool has
    # no "score" key and would otherwise KeyError here).
    def _w_score(w):
        return float(w.get("score", w.get("motion", 0.0)))
    # dedup by (basename, start), keep best score; track backward
    best = {}
    for w in windows:
        base = Path(w["zarr_path"]).name
        if base not in idx_by_base:
            continue
        key = (base, int(w["start"]))
        if key not in best or _w_score(w) > _w_score(best[key]):
            best[key] = w
    uniq = list(best.values())
    bwd = [w for w in uniq if w.get("backward")]
    fwd = [w for w in uniq if not w.get("backward")]
    fwd.sort(key=lambda w: -_w_score(w))
    # All backward first (guarantee reverse coverage), then top forward by
    # score, then a FINAL hard cap at num_windows — without the cap a small
    # num_windows (< #backward, e.g. a 5-window smoke) would return ALL
    # backward and blow the budget.
    chosen = (bwd + fwd[: max(0, num_windows - len(bwd))])[: num_windows]
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
    ap.add_argument("--clean_only", action="store_true",
                    help="generate ONLY the clean pass (no counterfactual twin). "
                         "Spends the whole budget on distinct clean windows.")
    ap.add_argument("--analyze_actions", action="store_true",
                    help="don't generate; just encode z-actions for the selected "
                         "windows and print the fwd/bwd/left/right/stationary "
                         "breakdown, then exit (skips the teacher pipe load).")
    ap.add_argument("--analyze_dump", default=None,
                    help="with --analyze_actions: write per-window class rows to "
                         "<path>.<gpu_rank>.json (for building a balanced pool).")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    global CONFIG_PATH, MANIFEST_PATH, CKPT_PATH, LMDB_ROOT, LMDB_CF_ROOT, NUM_WINDOWS, CLEAN_ONLY
    CONFIG_PATH = args.config
    MANIFEST_PATH = args.manifest
    CKPT_PATH = args.ckpt
    LMDB_ROOT = args.lmdb_root
    LMDB_CF_ROOT = args.lmdb_cf_root
    NUM_WINDOWS = args.num_windows
    CLEAN_ONLY = bool(args.clean_only)
    ANALYZE_ACTIONS = bool(args.analyze_actions)
    ANALYZE_DUMP = args.analyze_dump

    from omegaconf import OmegaConf
    from utils.zarr_dataset import ZarrRideDataset, _motion_capped_latents
    import zarr as zarr_lib

    gpu_rank = int(os.environ.get("GPU_RANK", 0))
    num_gpus = int(os.environ.get("NUM_GPUS", 4))
    # Auto-detect: analyze-only runs (no teacher pipe) work fine on CPU,
    # which avoids the GPU queue. Generation requires CUDA.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    pipe = None
    step = None
    if not ANALYZE_ACTIONS:
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
    # Action-distribution analysis (no generation). Encodes z-actions for the
    # selected windows and reports the fwd/bwd/left/right/stationary split.
    # z[:,0]=z2 (steer), z[:,1]=z7 (throttle). Per-frame classification:
    #   stationary: |z7|<0.1 and |z2|<0.1
    #   forward:    |z7|>|z2| and z7> 0.1     backward: |z7|>|z2| and z7<-0.1
    #   left:       |z2|>=|z7| and z2> 0.1    right:    |z2|>=|z7| and z2<-0.1
    # -----------------------------------------------------------------------
    if ANALYZE_ACTIONS:
        import collections
        n_lat_by_ri = {
            ri: rides_for_ds[k]["n_latent_frames"]
            for k, ri in enumerate(unique_ride_idxs)
        }
        attrs_by_ri = {
            ri: rides_for_ds[k]["attrs"]
            for k, ri in enumerate(unique_ride_idxs)
        }
        frame_ct = collections.Counter()
        win_ct = collections.Counter()
        zstat = collections.Counter()   # raw z2/z7 sign distribution (diagnostic)
        dump_rows = []
        n_win = 0
        n_beyond = 0
        for j, (ri, offset, _n) in enumerate(my_samples):
            zp = rides_all[ri]["zarr_path"]
            # Cap n_latent_frames at the MOTION-file length: encode_z_actions_window
            # raises if asked for more latents than the motion supports. Most
            # windows sit within the (often-shorter) motion; skip only those
            # whose span runs past it.
            n_cap = _motion_capped_latents(attrs_by_ri[ri], Path(motion_root))
            if offset + STREAM_LATENT_SPAN > n_cap:
                n_beyond += 1
                continue
            try:
                z_win = z_ds.encode_z_actions_window(
                    zp, min(n_lat_by_ri[ri], n_cap), offset, offset + STREAM_LATENT_SPAN,
                )
            except Exception as exc:
                log.warning("analyze: skip %s@%d: %s", zp, offset, exc)
                continue
            z = z_win[:NUM_FRAMES, action_dims].float()  # [F,2]
            z2, z7 = z[:, 0], z[:, 1]
            a2, a7 = z2.abs(), z7.abs()
            # Raw sign diagnostic (independent of throttle-vs-steer dominance):
            zstat["steer_left(z2>0.1)"] += int((z2 > 0.1).sum().item())
            zstat["steer_right(z2<-0.1)"] += int((z2 < -0.1).sum().item())
            zstat["steer_none(|z2|<0.1)"] += int((a2 < 0.1).sum().item())
            zstat["thr_fwd(z7>0.1)"] += int((z7 > 0.1).sum().item())
            zstat["thr_bwd(z7<-0.1)"] += int((z7 < -0.1).sum().item())
            zstat["thr_none(|z7|<0.1)"] += int((a7 < 0.1).sum().item())
            zstat["TOTAL_frames"] += int(z2.numel())
            # Turn = significant steering relative to throttle (|z2| > 0.5|z7|
            # and |z2|>0.1), taking PRIORITY over fwd/bwd so a moving turn
            # counts as a turn (not "forward"). Was |z2|>|z7| which dropped
            # nearly all moving turns into 'forward'.
            stat = (a7 < 0.1) & (a2 < 0.1)
            turn = (~stat) & (a2 > 0.5 * a7) & (a2 > 0.1)
            left = turn & (z2 > 0)
            right = turn & (z2 < 0)
            fwd = (~stat) & (~turn) & (z7 > 0.1)
            bwd = (~stat) & (~turn) & (z7 < -0.1)
            other = ~(stat | left | right | fwd | bwd)
            cls = {"forward": fwd, "backward": bwd, "left": left,
                   "right": right, "stationary": stat, "other": other}
            for name, mask in cls.items():
                frame_ct[name] += int(mask.sum().item())
            # per-window dominant class (by frame count)
            per_cls = {name: int(mask.sum().item()) for name, mask in cls.items()}
            dom = max(per_cls.items(), key=lambda kv: kv[1])[0]
            win_ct[dom] += 1
            n_win += 1
            if ANALYZE_DUMP:
                dump_rows.append({
                    "zarr_path": zp, "start": int(offset),
                    "n_latent_frames": int(n_lat_by_ri[ri]),
                    "dom": dom, "counts": per_cls,
                })
            if (j + 1) % 200 == 0:
                log.info("analyze: %d/%d windows", j + 1, len(my_samples))
        order = ["forward", "backward", "left", "right", "stationary", "other"]
        tot_f = sum(frame_ct.values()) or 1
        tot_w = sum(win_ct.values()) or 1
        print("\n===== ACTION DISTRIBUTION (%d windows, %d frames; "
              "%d skipped: span past motion) ====="
              % (n_win, tot_f, n_beyond), flush=True)
        print(f"{'class':<12}{'per-frame %':>14}{'(count)':>12}"
              f"{'per-window %':>16}{'(count)':>12}", flush=True)
        for name in order:
            print(f"{name:<12}{100.0*frame_ct[name]/tot_f:>13.2f}%"
                  f"{frame_ct[name]:>12}{100.0*win_ct[name]/tot_w:>15.2f}%"
                  f"{win_ct[name]:>12}", flush=True)
        print("=" * 66, flush=True)
        zt = zstat.get("TOTAL_frames", 0) or 1
        print("\n--- RAW z-sign distribution (per-frame, independent of "
              "throttle-vs-steer dominance) ---", flush=True)
        for k in ["steer_left(z2>0.1)", "steer_right(z2<-0.1)", "steer_none(|z2|<0.1)",
                  "thr_fwd(z7>0.1)", "thr_bwd(z7<-0.1)", "thr_none(|z7|<0.1)"]:
            print(f"  {k:<24}{100.0*zstat[k]/zt:>8.2f}%  ({zstat[k]})", flush=True)
        print(f"  steer L/R ratio: {zstat['steer_left(z2>0.1)']/max(1,zstat['steer_right(z2<-0.1)']):.2f}",
              flush=True)
        if ANALYZE_DUMP:
            import json as _json
            dpath = f"{ANALYZE_DUMP}.{gpu_rank}.json"
            with open(dpath, "w") as _f:
                _json.dump(dump_rows, _f)
            log.info("GPU %d: wrote %d class rows -> %s",
                     gpu_rank, len(dump_rows), dpath)
        return

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

        if os.path.exists(clean_path) and (CLEAN_ONLY or os.path.exists(cf_path)):
            skipped += 1
            continue

        try:
            # Cap n_latent_frames at the motion-file length (encode_z_actions_window
            # raises if asked for more latents than the motion supports). Skip the
            # window only if its span runs past the usable motion.
            n_cap = _motion_capped_latents(info["attrs"], Path(motion_root))
            if offset + STREAM_LATENT_SPAN > n_cap:
                skipped += 1
                continue

            g = info["zarr_group"]
            lat_np = g["latents"][offset:offset + NUM_FRAMES]
            clean_x = torch.from_numpy(lat_np.astype(np.float32)).unsqueeze(0).to(device)

            z_win = z_ds.encode_z_actions_window(
                info["zarr_path"], min(info["n_lat"], n_cap),
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
            if not CLEAN_ONLY and not os.path.exists(cf_path):
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
