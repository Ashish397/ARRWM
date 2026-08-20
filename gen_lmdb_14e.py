"""DRAFT (review before submission): 14e ODE-distill dataset generator.

Differences vs gen_lmdb.py (the v14/v14d generator), per the ODE-step audit:

  1. CHAINED self-committed generation: stream_causal_chain (block-by-block
     KV-cache rollout, clean commit at t=0) — the teacher's REAL sampler —
     instead of the joint all-frames-same-t teacher-forced denoise. Chunks
     c>0 therefore condition on the teacher's own committed context: the
     AR-compounding exposure the current LMDB completely lacks.
  2. GT + FLIP pair per context (apply_counterfactual), same noise seed —
     counterfactual coverage on the weak backward/mirror axis at half the
     cost of the old 8-dir plan.

  ACTION CONVENTION (canonical, follows 14e — legacy z2/z7 names retired):
     action vector column 0 = THROTTLE (forward/backward), column 1 = STEER.
     Empirically calibrated on the 14e teacher (col-0 command -> pure forward
     motion; col-1 command -> pure turn). ``action_dims`` still selects the
     same two components from the ss_vae vector; only the naming changes.
  3. 5 snapshots per chunk at the PINNED rung grid [1000, 625, 312.5, 178.6]
     + final x0 (SNAP_REC below; grid must match training random_steps AND
     the eval denoising_step_list — one list everywhere).
  4. Inline QA: committed-chunk mean/std per chunk written to a jsonl so
     dataset drift is measured at generation time (flow_degradation finding:
     teacher chain drifts; we must know what the students are distilling).

Layout: OUT_ROOT/{ride_ts}_o{offset:05d}_{gt|flip}.pt with
  trajectory [n_chunks, 5, 3, C, H, W] fp16, z [tot_f, 2], seed frames
  (offset), noise_seed, committed stats.

Env: GL_CONFIG (teacher config), GL_CKPT, GL_POOL (curated pool json),
GL_OUT, GL_NUM (contexts for this shard), GL_SHARD / GL_NSHARDS,
GL_CHUNKS (def 6), GL_STEPS (def 48).

VALIDATE BEFORE SUBMIT:
  - snapshot index<->rung mapping (SNAP_REC uses step_recorder si semantics:
    si=-1 initial noise = rung t=1000; post-step si s = state at
    sigmas[s+1]; confirm against af_utils/schedule.resolve_denoising_step_list)
  - flip z handling for the seed frames (zeros like flow_record? we carry
    the GT z on seed frames, flipped only on generated frames)
  - storage: ~36 MB per context pair at 6 chunks.
"""
import os, json, time
os.environ.setdefault("WORLD_SIZE", "1"); os.environ.setdefault("RANK", "0"); os.environ.setdefault("LOCAL_RANK", "0")
import numpy as np
import torch
from omegaconf import OmegaConf

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
CONFIG = os.environ.get("GL_CONFIG", f"{ARR}/configs/causal_lora_diffusion_teacher_v14e.yaml")
CKPT = os.environ.get("GL_CKPT", f"{ARR}/logs/v14e_pca8_raw/causal_lora_step0005000.pt")
POOL = os.environ.get("GL_POOL", f"{ARR}/paper_assets/v14d_train_windows_balanced.json")
OUT = os.environ.get("GL_OUT", "/projects/u6ex/fbots/frodobots_lmdb/v14e")
NUM = int(os.environ.get("GL_NUM", "100"))
SHARD = int(os.environ.get("GL_SHARD", "0"))
NSHARDS = int(os.environ.get("GL_NSHARDS", "1"))
GEN_CHUNKS = int(os.environ.get("GL_CHUNKS", "6"))
SEED_CHUNKS = int(os.environ.get("GL_SEED_CHUNKS", "3"))  # real-context chunks
STEPS = int(os.environ.get("GL_STEPS", "20"))    # validated vs 48 (flow_step_fidelity)
assert STEPS == 20, "SNAP_REC indices are only valid for the 20-step schedule"

# recorder si -> stored rung state; si=-1 is the initial noise (t=1000);
# post-step record si sits at the step's target sigma -> state at schedule
# index si+1. 14e PINNED RUNG GRID (20-step shift-5 schedule, validated by
# flow_step_fidelity: block-0 endpoint delta 14% of inter-action sep):
#   t = [1000, 625, 357, 208] + final x0
#   = schedule idx [0, 15, 18, 19] -> recorder si [-1, 14, 17, 18], final 19.
# Same grid MUST be used in training random_steps and eval denoising_step_list.
SNAP_REC = [-1, 14, 17, 18, 19]
NFB = 3

# Action-multiplicity variant sets (pilot campaign). Compass entries use a
# constant (throttle, steer) on generated frames, |z| = 0.5 like flow_record.
_M = 0.5; _D = _M / (2 ** 0.5)
COMPASS = {"cF": (_M, 0.0), "cFR": (_D, _D), "cR": (0.0, _M), "cBR": (-_D, _D),
           "cB": (-_M, 0.0), "cBL": (-_D, -_D), "cL": (0.0, -_M), "cFL": (_D, -_D),
           "cN": (0.0, 0.0)}
_VSETS = {"gt": ["gt"], "flip2": ["gt", "flip"],
          "dir4": ["gt", "flip", "cL", "cR"],
          "dir8": ["c" + d for d in ("F", "FR", "R", "BR", "B", "BL", "L", "FL")],
          "dir8n": ["c" + d for d in ("F", "FR", "R", "BR", "B", "BL", "L", "FL", "N")]}
VARIANTS = _VSETS[os.environ.get("GL_VARIANT", "flip2")]


def main():
    from trainer.causal_diffusion_teacher_train import CausalLoRADiffusionTrainer
    from utils.causal_chain_rollout import stream_causal_chain
    from utils.zarr_dataset import ZarrRideDataset
    from gen_lmdb import apply_counterfactual, window_noise_seed as _wns
    # GL_NOISE_VARIANT=k (k>=1): generate an ADDITIONAL noise realization of
    # every chain — same context, same actions, different seed — stored under
    # a suffixed ride id ({ts}ns{k}) so the chunked dataset ingests it as an
    # independent chain. Multi-noise supervision makes noise->future routing
    # learnable (single-noise data makes mean-prediction optimal).
    _NV = int(os.environ.get("GL_NOISE_VARIANT", "0"))
    def window_noise_seed(ts_id, off):
        return _wns(ts_id, off) + 90001 * _NV

    os.makedirs(OUT, exist_ok=True)
    cfg = OmegaConf.merge(OmegaConf.load(f"{ARR}/configs/default_config.yaml"),
                          OmegaConf.load(CONFIG))
    os.environ["ARRWM_ACTION_ENCODER"] = str(cfg.get("teacher_action_encoder", "pca_raw"))
    # per-shard scratch (4 shards sharing one dir would collide on the
    # ~22GB manifest write, cf. jobs 5736570/5736740); pre-seed the ride
    # manifest via symlink so the trainer never rebuilds/rescans it.
    _scratch = f"{OUT}/.gen_scratch_s{SHARD}"
    os.makedirs(_scratch, exist_ok=True)
    _cache = f"{ARR}/analysis/eval_final/flow_viz/.rec_scratch/.ride_manifest.pt"
    if not os.path.exists(f"{_scratch}/.ride_manifest.pt") and os.path.exists(_cache):
        os.symlink(_cache, f"{_scratch}/.ride_manifest.pt")
    cfg.logdir = _scratch; cfg.auto_resume = False; cfg.control_test = False
    cfg.save_checkpoints = False; cfg.stop_at_step = 0
    trainer = CausalLoRADiffusionTrainer(cfg)
    trainer.config.resume_from = CKPT; trainer.start_step = 0
    trainer._maybe_resume()
    trainer._offload_training_state()
    from torch.nn.parallel import DistributedDataParallel as DDP
    wrapper = trainer.model.module if isinstance(trainer.model, DDP) else trainer.model
    wrapper.eval()
    cm = wrapper.model
    if hasattr(cm, "base_model"):
        cm = cm.base_model.model
    cm.block_mask = None
    device, dtype = trainer.device, trainer.dtype

    pool = json.load(open(POOL))
    if isinstance(pool, dict):                 # balanced-pool wrapper
        pool = pool["windows"]
    windows = pool[SHARD::NSHARDS][:NUM]
    print(f"[gen14e] shard {SHARD}/{NSHARDS}: {len(windows)} contexts, "
          f"{GEN_CHUNKS} chunks, snap {SNAP_REC}", flush=True)

    manifest = torch.load(
        f"{ARR}/analysis/eval_final/flow_viz/.rec_scratch/.ride_manifest.pt",
        map_location="cpu")
    rides = manifest["rides"] if isinstance(manifest, dict) else manifest
    pe_by_zarr = {r["zarr_path"]: r["prompt_embeds"] for r in rides}
    # z-action dataset: mirrors gen_lmdb.py:380-415 (from_manifest with
    # motion_root + ss_vae ckpt); restricted to the pool's rides.
    pool_zarrs = {w["zarr_path"] for w in windows}
    rides_for_ds, n_lat_by_zarr = [], {}
    for r in rides:
        if r["zarr_path"] not in pool_zarrs:
            continue
        if "n_latent_frames" not in r:
            raise SystemExit(f"manifest ride missing n_latent_frames: {r['zarr_path']}")
        n_lat = int(r["n_latent_frames"])
        rides_for_ds.append({"zarr_path": r["zarr_path"],
                             "prompt_embeds": r["prompt_embeds"],
                             "attrs": r.get("attrs", {}),
                             "n_latent_frames": n_lat})
        n_lat_by_zarr[r["zarr_path"]] = n_lat
    z_ds = ZarrRideDataset.from_manifest(
        rides_data=rides_for_ds,
        motion_root=str(cfg.get("motion_root", "/projects/u6ex/fbots/frodobots_motion")),
        ss_vae_checkpoint=str(cfg.ss_vae_checkpoint),
        device="cpu", ss_vae_device=str(device))
    action_dims = list(cfg.get("action_dims", [2, 7]))   # -> (throttle, steer) order, 14e convention

    seed_f = NFB * SEED_CHUNKS
    tot_f = seed_f + NFB * GEN_CHUNKS
    qa = open(f"{OUT}/qa_shard{SHARD}.jsonl", "a")
    n_skipped = 0
    for w in windows:
        zp, off = w["zarr_path"], int(w.get("offset", w.get("start")))
        ts_id = os.path.basename(zp).replace(".zarr", "")
        if int(os.environ.get("GL_NOISE_VARIANT", "0")):
            ts_id = f"{ts_id}ns{os.environ['GL_NOISE_VARIANT']}"
        cap = min(n_lat_by_zarr.get(zp, 0), int(w.get("n_latent_frames", 1 << 30)))
        if zp not in pe_by_zarr or off + tot_f > cap:
            n_skipped += 1
            print(f"[gen14e] SKIP {ts_id}_o{off:05d} (span {off + tot_f} > cap {cap} "
                  f"or missing prompt)", flush=True)
            continue
        seedlat = ZarrRideDataset.load_latent_chunk(zp, off, off + seed_f) \
            .unsqueeze(0).to(device, torch.float32)
        pe = pe_by_zarr[zp].unsqueeze(0).to(device, dtype)
        # GT per-frame z for the full span (seed + generated), via the same
        # instance loader gen_lmdb.py uses: build z_ds once (see below) and
        # call encode_z_actions_window(zp, n_latents, start, end); slice the
        # config's action_dims to (z2, z7). Motion-length capping as in
        # gen_lmdb._motion_capped_latents — skip windows whose span exceeds it.
        z_win = z_ds.encode_z_actions_window(zp, n_lat_by_zarr[zp], off, off + tot_f)
        z_gt = z_win[:, action_dims].float().to(device, dtype).view(1, tot_f, -1)[..., :2]

        for variant in VARIANTS:
            dst = f"{OUT}/{ts_id}_o{off:05d}_{variant}.pt"
            if os.path.exists(dst):
                continue
            z = z_gt.clone()
            if variant == "flip":                 # flip generated frames only
                z[:, seed_f:] = apply_counterfactual(z[:, seed_f:])
            elif variant in COMPASS:              # constant compass command
                thr, ste = COMPASS[variant]
                z[:, seed_f:, 0] = thr
                z[:, seed_f:, 1] = ste
            rec = {b: {} for b in range(GEN_CHUNKS)}
            def recorder(b, si, lat, _rec=rec):
                if b in _rec and si in SNAP_REC:
                    _rec[b][si] = lat.detach().float().to(torch.float16).cpu()
            t0 = time.time()
            stream_causal_chain(wrapper, trainer.action_projection,
                                trainer.action_token_projection,
                                pe, seedlat, z, gen_chunks=GEN_CHUNKS,
                                eval_steps=STEPS, dtype=dtype, device=device,
                                nfb=NFB, seed_base=window_noise_seed(ts_id, off),
                                step_recorder=recorder)
            traj = torch.stack([torch.stack([rec[b][s][0] for s in SNAP_REC])
                                for b in range(GEN_CHUNKS)])   # [C_n, 5, 3, C, H, W]
            committed = traj[:, -1]
            stats = {"mean": [float(committed[c].float().mean()) for c in range(GEN_CHUNKS)],
                     "std": [float(committed[c].float().std()) for c in range(GEN_CHUNKS)]}
            _tmp = f"{dst}.tmp{os.getpid()}"   # pid suffix: concurrent jobs never share a tmp
            torch.save({"trajectory": traj, "z": z[0].cpu(),
                        "zarr_path": zp, "window_offset": off, "variant": variant,
                        "noise_seed": window_noise_seed(ts_id, off),
                        "snap_rec": SNAP_REC, "gen_chunks": GEN_CHUNKS,
                        "seed_chunks": SEED_CHUNKS,
                        "committed_stats": stats}, _tmp)
            os.rename(_tmp, dst)      # atomic: no truncated files on kill
            qa.write(json.dumps({"w": f"{ts_id}_o{off}", "v": variant, **stats}) + "\n")
            qa.flush()
            print(f"[gen14e] {ts_id}_o{off:05d}_{variant} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    qa.close()
    print("[gen14e] shard done", flush=True)


if __name__ == "__main__":
    main()
