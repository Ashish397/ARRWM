"""When actions bend the flow — cross-FLOW-MODEL comparison (window r08).

Only genuine flow-ODE samplers qualify: ours (48-step FlowMatch, shift=5),
astra (50-step FlowMatch + camera CFG), worldcam (64-step diffusion-forcing
ladder, stage=8 micro-steps per AR step). Few-step DMD samplers
(minwm/matrixgame) re-noise between steps -> no flow path; 4-step Euler
distillates (yume/worldplay) excluded as too coarse.

Per model: mean pairwise latent distance between the 8 compass-action
trajectories (matched noise: ours + astra pin the seed, worldcam seed=0)
at every recorded state, against the ACTUAL diffusion timestep of that
state. Primary panel is normalized (sep / final sep) because guidance
regimes differ (ours guidance-free, astra camera-CFG, worldcam cfg=4);
raw curves shown too since all three share the Wan2.1 VAE latent space.

Inputs: flow_viz/trajs_{pca8,16node}_w8.npz (utils/flow_record.py),
flow_viz/flow_astra/r08_{D}/chunk0.npz, flow_viz/flow_worldcam/r08_{D}/steps.npz.
Writes flow_viz/flow_divergence_models.png. Env: FDM_OUT.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
FV = f"{ARR}/analysis/eval_final/flow_viz"
OUT = os.environ.get("FDM_OUT", FV)
DIRS = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]
IU = np.triu_indices(len(DIRS), 1)
MC = {"pca8": "#1f77b4", "16node": "#17becf", "astra": "#d62728", "worldcam": "#2ca02c"}


def pairsep(states):
    """states: [8, T, ...] -> mean pairwise distance per T (flattened dims).

    Converts one timestep slice at a time (4GB login-node cgroup)."""
    x = states.reshape(states.shape[0], states.shape[1], -1)
    out = np.empty(x.shape[1])
    for t in range(x.shape[1]):
        xt = x[:, t].astype(np.float32)
        dd = np.linalg.norm(xt[:, None] - xt[None, :], axis=-1)
        out[t] = dd[IU].mean()
    return out


def ours_curve(run, nseeds=4, steps=48, shift=5.0):
    z = np.load(f"{FV}/trajs_{run}_w8.npz")
    sep = None
    for sd in range(nseeds):
        st = np.stack([z[f"{d}_{sd}"] for d in DIRS])          # [8, 49, D]
        s = pairsep(st)
        sep = s if sep is None else sep + s
        del st
    sep /= nseeds
    sig = np.linspace(1.0, 0.0, steps + 1)[:-1]
    sig = shift * sig / (1 + (shift - 1) * sig)
    ts = np.concatenate([[1.0], sig[1:], [0.0]]) * 1000.0
    return ts, sep


def astra_curve():
    per_dir = []
    ts = None
    for d in DIRS:
        f = f"{FV}/flow_astra/r08_{d}/chunk0.npz"
        if not os.path.exists(f):
            print(f"[fdm] MISSING {f}")
            return None
        z = np.load(f)
        n = len([k for k in z.files if k.startswith("s")])
        per_dir.append(np.stack([z[f"s{j}"][0] for j in range(n)]))   # [51, C, F, H, W]
        if ts is None:
            # state 0 = pure noise (t=1000); state j>=1 sits at step j-1's
            # TARGET sigma = the next step's consumed timestep; final -> 0
            tcons = z["ts"]
            ts = np.concatenate([[1000.0], tcons[1:], [0.0]])
    return ts, pairsep(np.stack(per_dir))


def worldcam_curve():
    """Track ONE generated frame through its noise->clean ladder, all dirs.

    steps.npz: meta=[condition_num, generated_num, stage, num_ar_steps],
    ik=[(i,k)] with k=-1 the pre-denoise window state, t{j}= per-position
    timesteps, x{j}= full window latents [1, C, P, H, W]. The window slides
    one frame per AR step; the frame exiting at AR step i sits at position
    condition_num + (i_exit - i) during AR step i. We pick the LAST frame
    whose full lifetime is recorded and read its latent + timestep at every
    (i, k) of its life, per direction; separation at matched (i, k).
    """
    recs = {}
    for d in DIRS:
        f = f"{FV}/flow_worldcam/r08_{d}/steps.npz"
        if not os.path.exists(f):
            print(f"[fdm] MISSING {f}")
            return None
        recs[d] = np.load(f)
    z0 = recs[DIRS[0]]
    cond, gen, stage, nar = [int(v) for v in z0["meta"]]
    ik = z0["ik"]
    P = z0["x0"].shape[2]                                     # window positions
    life = P - cond                                           # frames in gen region
    i_exit = nar - 1                                          # last fully recorded exit
    i_birth = i_exit - life + 1
    if i_birth < 0:
        i_birth, i_exit = 0, life - 1

    # records for AR steps i_birth..i_exit, position of tracked frame at i:
    # pos(i) = cond + (i_exit - i). t labels: k=-1 pre-state sits at its rung
    # start; a post-update state (i,k) sits at the NEXT micro-rung (recorded
    # as t of record j+1); k=7 states duplicate the next AR step's k=-1
    # pre-state, so keep only the very last one (fully denoised, t=0).
    traj, ts = {d: [] for d in DIRS}, []
    for j, (i, k) in enumerate(ik):
        if not (i_birth <= i <= i_exit):
            continue
        pos = cond + (i_exit - i)
        if pos >= P:
            continue
        if k == -1:
            t_here = float(z0[f"t{j}"][pos])
        elif k < stage - 1:
            t_here = float(z0[f"t{j + 1}"][pos])
        elif i == i_exit:                                     # final clean exit
            t_here = 0.0
        else:
            continue                                          # dup of next k=-1
        ts.append(t_here)
        for d in DIRS:
            traj[d].append(recs[d][f"x{j}"][0, :, pos])
    order = np.argsort(-np.asarray(ts), kind="stable")        # noise -> clean
    ts = np.asarray(ts)[order]
    states = np.stack([np.stack(traj[d])[order] for d in DIRS])   # [8, T, C, H, W]
    return ts, pairsep(states)


def main():
    curves = {}
    for run in ("pca8", "16node"):
        curves[run] = ours_curve(run)
        print(f"[fdm] {run}: {curves[run][1].shape[0]} states, "
              f"sep end {curves[run][1][-1]:.1f}", flush=True)
    for name, fn in (("astra", astra_curve), ("worldcam", worldcam_curve)):
        c = fn()
        if c is not None:
            curves[name] = c
            print(f"[fdm] {name}: {c[1].shape[0]} states, sep end {c[1][-1]:.1f}", flush=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2))
    for ax in (ax1, ax2):
        ax.set_xlim(1010, -10)
        ax.set_xlabel("diffusion timestep t (1000 = pure noise)")
        ax.grid(alpha=0.3)
    # ours: 3 latent frames, astra: 8, worldcam tracked frame: 1 (x 16x60x104)
    dims = {"pca8": 299520, "16node": 299520, "astra": 798720, "worldcam": 99840}
    for name, (ts, sep) in curves.items():
        ax1.plot(ts, sep / max(sep[-1], 1e-8), lw=2.0, color=MC[name],
                 marker="o" if len(ts) < 25 else None, ms=3.5, label=name)
        rms = sep / np.sqrt(dims[name])
        ax2.semilogy(ts, np.maximum(rms, 1e-5), lw=2.0, color=MC[name],
                     marker="o" if len(ts) < 25 else None, ms=3.5, label=name)
    ax1.set_ylabel("fraction of final inter-action separation")
    ax1.set_title("Normalized: when the action bends each model's flow")
    ax2.set_ylabel("per-element RMS inter-action distance (log)")
    ax2.set_title("Magnitude, dim-corrected (all share the Wan2.1 VAE latent space)")
    ax1.legend(); ax2.legend()
    fig.suptitle("Action divergence vs diffusion timestep — flow-ODE models only, window r08\n"
                 "ours guidance-free 48-step | astra camera-CFG 50-step | worldcam cfg=4 "
                 "diffusion-forcing ladder (tracked frame)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(f"{OUT}/flow_divergence_models.png", dpi=130)
    print(f"[fdm] saved {OUT}/flow_divergence_models.png", flush=True)


if __name__ == "__main__":
    main()
