"""3D 'grove of trees' over DIFFUSION time: many (context, noise) starting points,
8 actions each, z-axis = denoising step.

Extends utils/flow_viz.py from one window to FV_WINDOWS contexts, one distinct
noise draw per context (1 noise sample per context frames, as requested), 8
compass actions on that SAME noise. Records block-0 x_t at all steps, then:

  grove_3d.png          raw joint PCA + z=step: each context/noise is a tree
                        rooted at its own noise point, branches colored by action
  grove_overlay_3d.png  action-residual overlay: per (tree, step) subtract the
                        across-action mean -> every tree collapses to the origin
                        and only the action-driven deviation remains. If actions
                        are a GLOBAL operator on flow space, same colors align
                        across trees; if context-local, they scatter.
  grove_rot.gif         rotating view of the raw grove
  grove_cosine.png      8x8 heatmap: mean cross-tree cosine between final-step
                        action residuals (diag >> offdiag = consistent operator)

Env: FV_CONFIG, FV_CKPT, FV_WINDOWS (csv idx, def 0,4,8,12,16,20,24,28),
FV_STEPS (48), FV_OUT.
"""
import os, json
os.environ.setdefault("WORLD_SIZE", "1"); os.environ.setdefault("RANK", "0"); os.environ.setdefault("LOCAL_RANK", "0")
import numpy as np
import torch
from omegaconf import OmegaConf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
CONFIG = os.environ.get("FV_CONFIG", f"{ARR}/configs/causal_lora_diffusion_teacher_v14e.yaml")
CKPT = os.environ.get("FV_CKPT", f"{ARR}/logs/v14e_pca8_raw/causal_lora_step0005000.pt")
WINDOWS = [int(x) for x in os.environ.get("FV_WINDOWS", "0,4,8,12,16,20,24,28").split(",")]
STEPS = int(os.environ.get("FV_STEPS", "48"))
OUT = os.environ.get("FV_OUT", f"{ARR}/analysis/eval_final/flow_viz")

M = 0.5; Dv = M / (2 ** 0.5)
DIRS = {"F": (M, 0.0), "FR": (Dv, Dv), "R": (0.0, M), "BR": (-Dv, Dv),
        "B": (-M, 0.0), "BL": (-Dv, -Dv), "L": (0.0, -M), "FL": (Dv, -Dv)}   # (throttle, steer)
COLORS = {"F": "#1f77b4", "FR": "#17becf", "R": "#2ca02c", "BR": "#bcbd22",
          "B": "#d62728", "BL": "#e377c2", "L": "#9467bd", "FL": "#8c564b"}
DNAMES = list(DIRS)


def rollout_all():
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
    gen_chunks = 2
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
    manifest = torch.load(f"{ARR}/analysis/eval_final/manifest_unseen.pt", map_location="cpu")
    pe_by_zarr = {r["zarr_path"]: r["prompt_embeds"] for r in manifest}

    trajs = {}   # (widx, dir) -> [1 + STEPS, D] float16
    for widx in WINDOWS:
        w = windows[widx]
        zp, off = w["zarr_path"], int(w["offset"])
        seed = ZarrRideDataset.load_latent_chunk(zp, off, off + nfb).unsqueeze(0).to(device, torch.float32)
        pe = pe_by_zarr[zp].unsqueeze(0).to(device, dtype)
        for dname, (thr, ste) in DIRS.items():
            rec = []
            def recorder(b, si, lat, _rec=rec):
                if b == 0:
                    _rec.append(lat.detach().float().flatten().cpu().numpy().astype(np.float16))
            z_cond = torch.zeros(1, tot_f, 2, device=device, dtype=dtype)
            z_cond[:, nfb:, 0] = thr; z_cond[:, nfb:, 1] = ste
            stream_causal_chain(wrapper, trainer.action_projection, trainer.action_token_projection,
                                pe, seed, z_cond, gen_chunks=gen_chunks, eval_steps=STEPS,
                                dtype=dtype, device=device, nfb=nfb,
                                seed_base=1234 + widx * 7919,       # 1 distinct noise per context
                                step_recorder=recorder)
            trajs[(widx, dname)] = np.stack(rec)
            print(f"[grove] w{widx:02d} {dname}: {trajs[(widx, dname)].shape}", flush=True)
    np.savez_compressed(f"{OUT}/grove_trajs.npz",
                        **{f"w{w}_{d}": v for (w, d), v in trajs.items()})
    return trajs


def plot_all(trajs):
    T = next(iter(trajs.values())).shape[0]
    zs = np.arange(T)

    # ---------- raw grove: joint PCA over every point of every tree ----------
    allpts = np.concatenate([v.astype(np.float32) for v in trajs.values()], 0)
    mu = allpts.mean(0)
    U, S, V = torch.pca_lowrank(torch.from_numpy(allpts - mu), q=2, niter=6)
    P = V[:, :2].numpy()
    del allpts

    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection="3d")
    for (widx, d), tr in trajs.items():
        p2 = (tr.astype(np.float32) - mu) @ P
        ax.plot(p2[:, 0], p2[:, 1], zs, color=COLORS[d], alpha=0.8, lw=1.4,
                label=d if widx == WINDOWS[0] else None)
        ax.scatter(p2[0, 0], p2[0, 1], 0, color="black", s=25, zorder=5)
        ax.scatter(p2[-1, 0], p2[-1, 1], T - 1, color=COLORS[d], s=55, marker="*",
                   edgecolor="black", linewidth=0.4, zorder=5)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("denoising step")
    ax.set_title(f"Grove over diffusion time — {len(WINDOWS)} contexts x 1 noise x 8 actions\n"
                 f"(o = noise root at z=0, * = final latent at z={T-1})")
    ax.legend(ncol=4, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/grove_3d.png", dpi=120)

    def spin(i):
        ax.view_init(elev=22, azim=i * 4)
        return []
    anim = FuncAnimation(fig, spin, frames=90, blit=False)
    anim.save(f"{OUT}/grove_rot.gif", writer=PillowWriter(fps=12), dpi=70)
    plt.close(fig)
    print(f"[grove] saved grove_3d.png + grove_rot.gif", flush=True)

    # ---------- residual overlay: subtract per-tree across-action mean ----------
    res = {}
    for widx in WINDOWS:
        tr = np.stack([trajs[(widx, d)].astype(np.float32) for d in DNAMES])   # [8,T,D]
        tr -= tr.mean(0, keepdims=True)
        for i, d in enumerate(DNAMES):
            res[(widx, d)] = tr[i]
    allr = np.concatenate(list(res.values()), 0)
    Ur, Sr, Vr = torch.pca_lowrank(torch.from_numpy(allr), q=2, niter=6)
    Pr = Vr[:, :2].numpy()
    del allr

    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection="3d")
    for (widx, d), r in res.items():
        p2 = r @ Pr
        ax.plot(p2[:, 0], p2[:, 1], zs, color=COLORS[d], alpha=0.75, lw=1.4,
                label=d if widx == WINDOWS[0] else None)
        ax.scatter(p2[-1, 0], p2[-1, 1], T - 1, color=COLORS[d], s=55, marker="*",
                   edgecolor="black", linewidth=0.4, zorder=5)
    ax.set_xlabel("res PC1"); ax.set_ylabel("res PC2"); ax.set_zlabel("denoising step")
    ax.set_title("Action-residual overlay — all trees collapsed to a shared origin\n"
                 "same color aligned across trees = action acts as a global operator")
    ax.legend(ncol=4, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/grove_overlay_3d.png", dpi=120)

    def spin2(i):
        ax.view_init(elev=22, azim=i * 4)
        return []
    anim = FuncAnimation(fig, spin2, frames=90, blit=False)
    anim.save(f"{OUT}/grove_overlay_rot.gif", writer=PillowWriter(fps=12), dpi=70)
    plt.close(fig)
    print(f"[grove] saved grove_overlay_3d.png + grove_overlay_rot.gif", flush=True)

    # ---------- hypothesis test: cross-tree cosine of final action residuals ----------
    fin = {k: r[-1] for k, r in res.items()}
    nrm = {k: v / (np.linalg.norm(v) + 1e-8) for k, v in fin.items()}
    C = np.zeros((8, 8)); n = np.zeros((8, 8))
    for a, da in enumerate(DNAMES):
        for b, db in enumerate(DNAMES):
            for w1 in WINDOWS:
                for w2 in WINDOWS:
                    if w1 >= w2:
                        continue
                    C[a, b] += float(nrm[(w1, da)] @ nrm[(w2, db)]); n[a, b] += 1
    C /= np.maximum(n, 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(8), DNAMES); ax.set_yticks(range(8), DNAMES)
    for a in range(8):
        for b in range(8):
            ax.text(b, a, f"{C[a, b]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Cross-tree cosine of final action residuals\n"
                 "(diag >> offdiag = same action bends different contexts the same way)")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(f"{OUT}/grove_cosine.png", dpi=130)
    plt.close(fig)
    diag = float(np.trace(C) / 8)
    off = float((C.sum() - np.trace(C)) / 56)
    print(f"[grove] same-action cross-tree cosine {diag:.3f} | cross-action {off:.3f}", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    f = f"{OUT}/grove_trajs.npz"
    if os.path.exists(f):
        z = np.load(f)
        trajs = {}
        for k in z.files:
            wpart, d = k.rsplit("_", 1)
            trajs[(int(wpart[1:]), d)] = z[k]
        print(f"[grove] reusing {f} ({len(trajs)} trajs)", flush=True)
    else:
        trajs = rollout_all()
    plot_all(trajs)
