"""Flow-space visualization: how ACTIONS bend the denoising trajectory.

Rectified-flow-blog-style diagram (alechelbling.com/blog/rectified-flow) for our
world model: SAME context latents (one seed window) + SAME initial noise per
seed draw, denoised once per action direction (8 compass commands, |z|=0.5)
x FV_NSEEDS noise draws. Records x_t at every denoising step of the FIRST
generated block via the step_recorder hook in stream_causal_chain, projects
all trajectories into one shared 2D PCA basis, and plots:
  (1) 8 direction-colored trajectory bundles fanning out from the shared
      noise start points (o = noise, * = final committed latent),
  (2) mean inter-action latent distance vs denoising step ("when the action
      bends the flow"), matched-seed pairs only.

Mirrors utils/inject_eval.py's trainer loading + rollout call exactly.
Env: FV_CONFIG, FV_CKPT (default pca8@5000), FV_WINDOW (phase-A idx, def 8),
FV_NSEEDS (def 4), FV_STEPS (def 48), FV_OUT dir.
"""
import os, json
os.environ.setdefault("WORLD_SIZE", "1"); os.environ.setdefault("RANK", "0"); os.environ.setdefault("LOCAL_RANK", "0")
import numpy as np
import torch
from omegaconf import OmegaConf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
CONFIG = os.environ.get("FV_CONFIG", f"{ARR}/configs/causal_lora_diffusion_teacher_v14e.yaml")
CKPT = os.environ.get("FV_CKPT", f"{ARR}/logs/v14e_pca8_raw/causal_lora_step0005000.pt")
WINDOW = int(os.environ.get("FV_WINDOW", "8"))
NSEEDS = int(os.environ.get("FV_NSEEDS", "4"))
STEPS = int(os.environ.get("FV_STEPS", "48"))
OUT = os.environ.get("FV_OUT", f"{ARR}/analysis/eval_final/flow_viz")

M = 0.5; Dv = M / (2 ** 0.5)
DIRS = {"F": (M, 0.0), "FR": (Dv, Dv), "R": (0.0, M), "BR": (-Dv, Dv),
        "B": (-M, 0.0), "BL": (-Dv, -Dv), "L": (0.0, -M), "FL": (Dv, -Dv)}   # (throttle, steer)
COLORS = {"F": "#1f77b4", "FR": "#17becf", "R": "#2ca02c", "BR": "#bcbd22",
          "B": "#d62728", "BL": "#e377c2", "L": "#9467bd", "FL": "#8c564b"}


def main():
    os.makedirs(OUT, exist_ok=True)
    from trainer.causal_diffusion_teacher_train import CausalLoRADiffusionTrainer
    from utils.causal_chain_rollout import stream_causal_chain
    from utils.zarr_dataset import ZarrRideDataset

    cfg = OmegaConf.merge(OmegaConf.load(f"{ARR}/configs/default_config.yaml"),
                          OmegaConf.load(CONFIG))
    os.environ["ARRWM_ACTION_ENCODER"] = str(cfg.get("teacher_action_encoder", "pca_raw"))
    cfg.logdir = OUT; cfg.auto_resume = False; cfg.control_test = False
    cfg.save_checkpoints = False; cfg.stop_at_step = 0
    trainer = CausalLoRADiffusionTrainer(cfg)
    trainer.config.resume_from = CKPT; trainer.start_step = 0
    trainer._maybe_resume()

    nfb = trainer.num_frame_per_block
    device, dtype = trainer.device, trainer.dtype
    gen_chunks = 2                                  # block 0 is what we record
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

    trajs = {}   # (dir, seed) -> [1 + STEPS, D] flattened block-0 latents
    for dname, (thr, ste) in DIRS.items():
        for sd in range(NSEEDS):
            rec = []
            def recorder(b, si, lat, _rec=rec):
                if b == 0:
                    _rec.append(lat.detach().float().flatten().cpu().numpy())
            z_cond = torch.zeros(1, tot_f, 2, device=device, dtype=dtype)
            z_cond[:, nfb:, 0] = thr; z_cond[:, nfb:, 1] = ste
            stream_causal_chain(wrapper, trainer.action_projection, trainer.action_token_projection,
                                pe, seed, z_cond, gen_chunks=gen_chunks, eval_steps=STEPS,
                                dtype=dtype, device=device, nfb=nfb,
                                seed_base=1234 + sd * 7919, step_recorder=recorder)
            trajs[(dname, sd)] = np.stack(rec)
            print(f"[flow] {dname} seed{sd}: traj {trajs[(dname, sd)].shape}", flush=True)

    np.savez_compressed(f"{OUT}/trajectories_w{WINDOW}.npz",
                        **{f"{d}_{s}": v for (d, s), v in trajs.items()})

    # shared 2D PCA basis over every recorded point of every trajectory
    allpts = np.concatenate(list(trajs.values()), 0)
    mu = allpts.mean(0)
    U, S, V = torch.pca_lowrank(torch.from_numpy(allpts - mu), q=2, niter=6)
    P = V[:, :2].numpy()
    evr = (S[:2] ** 2 / torch.from_numpy(allpts - mu).pow(2).sum()).numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 8),
                                   gridspec_kw={"width_ratios": [1.6, 1]})
    for (dname, sd), tr in trajs.items():
        p2 = (tr - mu) @ P
        ax1.plot(p2[:, 0], p2[:, 1], color=COLORS[dname], alpha=0.8, lw=1.6,
                 label=dname if sd == 0 else None)
        ax1.scatter(p2[0, 0], p2[0, 1], color="black", s=24, zorder=5, marker="o")
        ax1.scatter(p2[-1, 0], p2[-1, 1], color=COLORS[dname], s=70, zorder=5,
                    marker="*", edgecolor="black", linewidth=0.5)
    ax1.set_title(f"Denoising trajectories under 8 actions — window r{WINDOW:02d}, "
                  f"{NSEEDS} noise seeds\nsame context + same noise per seed; "
                  f"o = initial noise, * = final latent "
                  f"(PC1 {evr[0]*100:.0f}%, PC2 {evr[1]*100:.0f}% var)")
    ax1.legend(ncol=4, fontsize=9)
    ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2")

    dnames = list(DIRS)
    npts = min(tr.shape[0] for tr in trajs.values())
    sep = []
    for si in range(npts):
        ds = []
        for sd in range(NSEEDS):
            pts = np.stack([trajs[(d, sd)][si] for d in dnames])
            diffs = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
            ds.append(diffs[np.triu_indices(len(dnames), 1)].mean())
        sep.append(float(np.mean(ds)))
    ax2.plot(range(npts), sep, lw=2, color="#333")
    ax2.set_xlabel(f"denoising step (0 = initial noise, {npts-1} = final)")
    ax2.set_ylabel("mean pairwise latent distance between actions")
    ax2.set_title("When the action bends the flow")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/flow_trajectories_w{WINDOW}.png", dpi=130)
    print(f"[flow] saved {OUT}/flow_trajectories_w{WINDOW}.png  sep0={sep[0]:.1f} sepEnd={sep[-1]:.1f}",
          flush=True)


if __name__ == "__main__":
    main()
