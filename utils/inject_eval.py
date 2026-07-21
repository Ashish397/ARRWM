"""Final eval driver: seed from UNSEEN non-moving windows, inject actions, generate.

Two phases (env IE_PHASE):
  A = action-injection: for each seed window, roll out 8 compass directions at a
      fixed moderate magnitude (~0.5 z). Tests true action response with NO GT/flip
      confound and NO seed-motion confound. branch = direction name (F,FR,R,...).
  B = static render:     zero action -> the model should hold the (moving-actor) scene
      still. branch = "static". Real continuation kept for MAE/LPIPS in chunk_metrics.

Reuses the trainer's loaded model + projections + _decode_latents + _compute_teacher_visuals
and utils.causal_chain_rollout.stream_causal_chain. Writes control_test-format
videos (stepXXXXX_rRR_{branch}_raw.mp4) + JSONL so utils/chunk_metrics.py consumes it.

Sharding: IE_SHARD / IE_NSHARDS split the (window,branch) work list across GPUs (one
process per GPU). rank in the JSONL/filename = window index; each shard writes its own
metrics_r*.jsonl. Env:
  IE_CONFIG, IE_CKPT, IE_PHASE(A|B), IE_WINDOWS(json), IE_MANIFEST(pt),
  IE_OUT(logdir), IE_SHARD, IE_NSHARDS, IE_STEP(default 5000)
"""
import os, json, glob
os.environ.setdefault("WORLD_SIZE", "1"); os.environ.setdefault("RANK", "0"); os.environ.setdefault("LOCAL_RANK", "0")
import numpy as np, torch, imageio
from omegaconf import OmegaConf
from trainer.causal_diffusion_teacher_train import CausalLoRADiffusionTrainer
from utils.causal_chain_rollout import stream_causal_chain
from utils.zarr_dataset import ZarrRideDataset

CONFIG = os.environ["IE_CONFIG"]
CKPT = os.environ["IE_CKPT"]
PHASE = os.environ.get("IE_PHASE", "A").upper()
WINDOWS = os.environ["IE_WINDOWS"]
MANIFEST = os.environ["IE_MANIFEST"]
OUT = os.environ["IE_OUT"]
SHARD = int(os.environ.get("IE_SHARD", "0"))
NSHARDS = int(os.environ.get("IE_NSHARDS", "1"))
STEP = int(os.environ.get("IE_STEP", "5000"))

M = 0.5; D = M / (2 ** 0.5)
DIRS = {"F": (M, 0.0), "FR": (D, D), "R": (0.0, M), "BR": (-D, D),
        "B": (-M, 0.0), "BL": (-D, -D), "L": (0.0, -M), "FL": (D, -D)}   # (throttle, steer)
# Phase S: forward-throttle magnitude sweep from stationary seeds. Positive
# counterfactual for the response-gain graph (rides are ~all forward, so FLIP
# only covers negative throttle); 0.6-0.8 probe extrapolation past |a|=0.5.
SWEEP = {f"F{int(round(m * 100)):03d}": (m, 0.0)
         for m in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}
# Phase G: the rest of the regular 0.1 dose grid (phase S already covers
# +0.1..+0.8 throttle): backward throttle and both steer signs, pure-axis.
GRID = {}
for _m in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    _t = f"{int(round(_m * 100)):03d}"
    GRID[f"B{_t}"] = (-_m, 0.0)
    GRID[f"SR{_t}"] = (0.0, _m)
    GRID[f"SL{_t}"] = (0.0, -_m)
# Phase FL: flip-eval on mined windows (utils/mine_left_windows.py) -- the
# real per-chunk z commands sign-flipped after the seed, exactly the training
# control-test FLIP protocol but on windows chosen to carry moderate-left
# steer (fills the sparse +0.35..+0.8 flipped-steer band).


def main():
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(CONFIG))
    os.environ["ARRWM_ACTION_ENCODER"] = str(cfg.get("teacher_action_encoder", "pca_raw"))
    cfg.logdir = os.path.abspath(OUT); cfg.auto_resume = False; cfg.control_test = False
    cfg.save_checkpoints = False; cfg.stop_at_step = 0
    cfg.skip_train_dataloader = True   # eval driver: no training data needed
    out_dir = os.path.join(OUT, "control_test"); os.makedirs(out_dir, exist_ok=True)

    trainer = CausalLoRADiffusionTrainer(cfg)
    trainer.config.resume_from = CKPT; trainer.start_step = 0
    trainer._maybe_resume()

    nfb = trainer.num_frame_per_block
    gen_chunks = int(getattr(cfg, "control_gen_chunks", 8))
    eval_steps = int(os.environ.get("IE_STEPS", getattr(cfg, "eval_inference_steps", 48)))
    tot_f = nfb * (1 + gen_chunks)
    ego = list(trainer.action_dims) if trainer.action_dims is not None else [0, 1]
    z2_idx, z7_idx = ego[0], ego[1]
    device, dtype = trainer.device, trainer.dtype

    trainer._offload_training_state()
    from torch.nn.parallel import DistributedDataParallel as DDP
    wrapper = trainer.model.module if isinstance(trainer.model, DDP) else trainer.model
    wrapper.eval()
    cm = wrapper.model
    if hasattr(cm, "base_model"):
        cm = cm.base_model.model
    cm.block_mask = None

    windows = json.load(open(WINDOWS))
    manifest = torch.load(MANIFEST, map_location="cpu")
    emb_by_path = {r["zarr_path"]: r["prompt_embeds"] for r in manifest}

    if PHASE == "A":
        branches = list(DIRS.keys())
    elif PHASE == "S":
        branches = list(SWEEP.keys())
    elif PHASE == "G":
        branches = list(GRID.keys())
    elif PHASE == "FL":
        branches = ["FL"]
    else:
        branches = ["static"]

    enc_ds = None
    if PHASE == "FL":   # dataset over the mined rides for real-command encoding
        want = {w["zarr_path"] for w in windows}
        picked = [r for r in manifest if r["zarr_path"] in want]
        enc_ds = ZarrRideDataset.from_manifest(
            rides_data=picked, motion_root=cfg.motion_root,
            ss_vae_checkpoint="action_query/checkpoints/ss_vae_8free.pt")
        nlat_by_path = {r["zarr_path"]: int(r["n_latent_frames"]) for r in picked}
    only_br = os.environ.get("IE_BRANCHES", "").strip()
    if only_br:                    # restrict branches (e.g. steps ablation)
        branches = [b for b in branches if b in set(only_br.split(","))]
    work = [(wi, w, br) for wi, w in enumerate(windows) for br in branches]
    only = os.environ.get("IE_ONLY", "").strip()
    if only:                       # regenerate specific window slots only
        keep = {int(x) for x in only.replace(":", ",").split(",")}
        work = [t for t in work if t[0] in keep]
    mine = work[SHARD::NSHARDS]
    print(f"[inject] phase {PHASE} shard {SHARD}/{NSHARDS}: {len(mine)}/{len(work)} gens "
          f"(step {STEP}, tot_f {tot_f}, ego {ego})", flush=True)

    def chunk_pool(pf):
        F_ = pf.shape[1]; nc = F_ // nfb
        return pf[:, :nc * nfb].reshape(1, nc, nfb, pf.shape[2]).mean(2)

    def _corr(a, b):
        a = a - a.mean(); b = b - b.mean()
        return float((a @ b / (a.norm() * b.norm()).clamp_min(1e-8)).item())

    mfile = os.path.join(out_dir, f"metrics_r{SHARD:03d}.jsonl")
    fh = open(mfile, "a")
    for wi, w, br in mine:
        if os.path.exists(os.path.join(out_dir, f"step{STEP:05d}_r{wi:02d}_{br}_raw.mp4")):
            continue                     # resume: video (and its jsonl line) already written
        zp, off = w["zarr_path"], int(w["offset"])
        seed = ZarrRideDataset.load_latent_chunk(zp, off, off + nfb).unsqueeze(0).to(device, torch.float32)
        pe = emb_by_path[zp].unsqueeze(0).to(device, dtype)
        z_cond = torch.zeros(1, tot_f, 2, device=device, dtype=dtype)
        if PHASE in ("A", "S", "G"):
            thr, ste = {"A": DIRS, "S": SWEEP, "G": GRID}[PHASE][br]
            z_cond[:, nfb:, 0] = thr; z_cond[:, nfb:, 1] = ste
        elif PHASE == "FL":   # real commands, sign-flipped after the seed
            z8 = enc_ds.encode_z_actions_window(zp, nlat_by_path[zp], off, off + tot_f)
            z_cond = z8.unsqueeze(0)[..., ego].to(device, dtype)
            z_cond[:, nfb:, :] *= -1.0
        # PHASE B: z_cond stays all-zero (hold still)
        try:
            full = stream_causal_chain(wrapper, trainer.action_projection, trainer.action_token_projection,
                                       pe, seed, z_cond, gen_chunks=gen_chunks, eval_steps=eval_steps,
                                       dtype=dtype, device=device, nfb=nfb)
            vid = trainer._decode_latents(full)                       # [F,H,W,3] uint8
            _, tz8 = trainer._compute_teacher_visuals(full)           # [1,nc,8]
        except Exception as e:
            print(f"[inject] w{wi} {br} FAILED: {e}", flush=True); continue
        tag = f"step{STEP:05d}_r{wi:02d}_{br}"
        imageio.mimwrite(os.path.join(out_dir, f"{tag}_raw.mp4"),
                         list(vid), fps=16, quality=8, macro_block_size=1)
        nc = tz8.shape[1]
        t2d = tz8[:, :, [z2_idx, z7_idx]]
        cmd = chunk_pool(z_cond)[:, :nc]
        t_gen = t2d[0, 1:].float(); c_gen = cmd[0, 1:].float()
        rec = {br: {"tz2": [round(x, 4) for x in t_gen[:, 0].tolist()],
                    "tz7": [round(x, 4) for x in t_gen[:, 1].tolist()],
                    "cz2": [round(x, 4) for x in c_gen[:, 0].tolist()],
                    "cz7": [round(x, 4) for x in c_gen[:, 1].tolist()],
                    "corr_z2": _corr(t_gen[:, 0], c_gen[:, 0]),
                    "corr_z7": _corr(t_gen[:, 1], c_gen[:, 1])}}
        fh.write(json.dumps({"step": STEP, "rank": wi, "ride": os.path.basename(zp),
                             "offset": off, "phase": PHASE, **rec}) + "\n")
        fh.flush()
    fh.close()
    print(f"[inject] shard {SHARD} DONE -> {mfile}", flush=True)


if __name__ == "__main__":
    main()
