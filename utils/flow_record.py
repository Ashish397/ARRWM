"""Record denoising trajectories for ONE run (config+ckpt) — no plotting.

Same protocol as utils/flow_viz.py (8 compass dirs x FR_NSEEDS seeds, same
context window + same per-seed noise, block-0 x_t recorded at the initial
noise and all FR_STEPS denoising steps) but parameterized per run and saved
float16 so every v14e ablation can be recorded with one code path:

  flow_viz/trajs_{FR_RUN}_w{FR_WINDOW}.npz   keys {dir}_{seed} -> [1+S, D]

seed_base matches flow_viz.py (1234 + seed*7919) and the latent shape is
identical across ablations, so the initial noise draws are IDENTICAL across
runs -> matched (seed, step) points are directly comparable across models.

Env: FR_RUN, FR_CONFIG, FR_CKPT, FR_WINDOW (def 8), FR_NSEEDS (def 4),
FR_STEPS (def 48), FR_OUT, FR_BLOCKS (csv block idx, def "0"; block b>0 keys
are saved as "b{b}_{dir}_{seed}", block 0 keeps the bare "{dir}_{seed}" keys).
"""
import os, json
os.environ.setdefault("WORLD_SIZE", "1"); os.environ.setdefault("RANK", "0"); os.environ.setdefault("LOCAL_RANK", "0")
import numpy as np
import torch
from omegaconf import OmegaConf

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
RUN = os.environ["FR_RUN"]
CONFIG = os.environ["FR_CONFIG"]
CKPT = os.environ["FR_CKPT"]
WINDOW = int(os.environ.get("FR_WINDOW", "8"))
NSEEDS = int(os.environ.get("FR_NSEEDS", "4"))
STEPS = int(os.environ.get("FR_STEPS", "48"))
OUT = os.environ.get("FR_OUT", f"{ARR}/analysis/eval_final/flow_viz")

M = 0.5; Dv = M / (2 ** 0.5)
DIRS = {"F": (M, 0.0), "FR": (Dv, Dv), "R": (0.0, M), "BR": (-Dv, Dv),
        "B": (-M, 0.0), "BL": (-Dv, -Dv), "L": (0.0, -M), "FL": (Dv, -Dv)}
if os.environ.get("FR_NOOP"):          # stationary/no-op branch
    DIRS = {"N": (0.0, 0.0)}
   # (throttle, steer)


def main():
    os.makedirs(OUT, exist_ok=True)
    dst = f"{OUT}/trajs_{RUN}_w{WINDOW}.npz"
    if os.path.exists(dst):
        print(f"[rec] {dst} exists, skipping", flush=True)
        return
    from trainer.causal_diffusion_teacher_train import CausalLoRADiffusionTrainer
    from utils.causal_chain_rollout import stream_causal_chain
    from utils.zarr_dataset import ZarrRideDataset

    cfg = OmegaConf.merge(OmegaConf.load(f"{ARR}/configs/default_config.yaml"),
                          OmegaConf.load(CONFIG))
    os.environ["ARRWM_ACTION_ENCODER"] = str(cfg.get("teacher_action_encoder", "pca_raw"))
    # per-run scratch: two concurrent flow_record jobs sharing one logdir
    # collide on the ~22GB .ride_manifest_shared.pt broadcast write (observed
    # hang, jobs 5736570/5736740)
    cfg.logdir = f"{OUT}/.rec_scratch_{RUN}"; cfg.auto_resume = False; cfg.control_test = False
    cfg.save_checkpoints = False; cfg.stop_at_step = 0
    trainer = CausalLoRADiffusionTrainer(cfg)
    trainer.config.resume_from = CKPT; trainer.start_step = 0
    trainer._maybe_resume()

    nfb = trainer.num_frame_per_block
    device, dtype = trainer.device, trainer.dtype
    blocks_env = [int(x) for x in os.environ.get("FR_BLOCKS", "0").split(",")]
    gen_chunks = max(2, max(blocks_env) + 1)        # enough blocks to cover FR_BLOCKS
    tot_f = nfb * (1 + gen_chunks)

    trainer._offload_training_state()
    from torch.nn.parallel import DistributedDataParallel as DDP
    wrapper = trainer.model.module if isinstance(trainer.model, DDP) else trainer.model
    wrapper.eval()
    cm = wrapper.model
    if hasattr(cm, "base_model"):
        cm = cm.base_model.model
    cm.block_mask = None

    windows = json.load(open(f"{ARR}/analysis/eval_final/phaseA_windows.json"))
    w = windows[WINDOW]
    zp, off = w["zarr_path"], int(w["offset"])
    seed = ZarrRideDataset.load_latent_chunk(zp, off, off + nfb).unsqueeze(0).to(device, torch.float32)
    manifest = torch.load(f"{ARR}/analysis/eval_final/manifest_unseen.pt", map_location="cpu")
    pe = {r["zarr_path"]: r["prompt_embeds"] for r in manifest}[zp].unsqueeze(0).to(device, dtype)

    blocks = [int(x) for x in os.environ.get("FR_BLOCKS", "0").split(",")]
    trajs = {}
    for dname, (thr, ste) in DIRS.items():
        for sd in range(NSEEDS):
            rec = {b: [] for b in blocks}
            def recorder(b, si, lat, _rec=rec):
                if b in _rec:
                    _rec[b].append(lat.detach().float().flatten().cpu().numpy().astype(np.float16))
            z_cond = torch.zeros(1, tot_f, 2, device=device, dtype=dtype)
            z_cond[:, nfb:, 0] = thr; z_cond[:, nfb:, 1] = ste
            stream_causal_chain(wrapper, trainer.action_projection, trainer.action_token_projection,
                                pe, seed, z_cond, gen_chunks=gen_chunks, eval_steps=STEPS,
                                dtype=dtype, device=device, nfb=nfb,
                                seed_base=1234 + sd * 7919, step_recorder=recorder)
            for b in blocks:
                key = f"{dname}_{sd}" if b == 0 else f"b{b}_{dname}_{sd}"
                trajs[key] = np.stack(rec[b])
            print(f"[rec] {RUN} {dname} seed{sd}: {trajs[f'{dname}_{sd}'].shape} x{len(blocks)}blk", flush=True)

    np.savez_compressed(dst, **trajs)
    print(f"[rec] saved {dst}", flush=True)


if __name__ == "__main__":
    main()
