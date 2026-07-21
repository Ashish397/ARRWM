"""How the FLOW itself differs across v14e ablations, aggregated over actions.

Reads flow_viz/trajs_{run}_w{W}.npz (utils/flow_record.py; falls back to
trajectories_w{W}.npz for pca8) and averages each model's 32 recorded
denoising trajectories (8 dirs x 4 seeds) into ONE mean flow path [1+S, D].
Because every run records the same context window with IDENTICAL noise draws
(same seed_base and latent shape), the mean paths all start at the same
point and live in the same latent space -> a joint PCA basis and pointwise
cross-model distances are both meaningful. Panels:

  1) joint 2-PC plot of the mean flow paths (o = shared noise start,
     * = final latent), step ticks every 8 steps; all 32 individual
     trajectories per model overlaid faint in the same plane
  2) departure from the pca8 reference vs diffusion timestep:
     ||mean_m(t) - mean_pca8(t)|| (log y)
  3) flow speed ||mean_m(t+1) - mean_m(t)|| vs timestep

Prints the cross-model endpoint distance matrix. Writes
flow_viz/flow_compare_w{W}.png. Env: FC_RUNS colon list, FC_WINDOW (def 8),
FC_OUT.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
FV = f"{ARR}/analysis/eval_final/flow_viz"                  # recordings live here
OUT = os.environ.get("FC_OUT", FV)
W = int(os.environ.get("FC_WINDOW", "8"))
RUNS = os.environ.get("FC_RUNS", "pca8:pca4:pca2:16node:4node:noatok:noadaln").split(":")
REF = RUNS[0]
STEPS = 48
SHIFT = 5.0
MCOLORS = plt.get_cmap("tab10").colors


def record_timesteps():
    sig = np.linspace(1.0, 0.0, STEPS + 1)[:-1]
    sig = SHIFT * sig / (1 + (SHIFT - 1) * sig)
    return np.concatenate([[1.0], sig[1:], [0.0]]) * 1000.0


def run_file(run):
    f = f"{FV}/trajs_{run}_w{W}.npz"
    if not os.path.exists(f) and run == "pca8":
        f = f"{FV}/trajectories_w{W}.npz"          # original float32 recording
    return f if os.path.exists(f) else None


def mean_flow(run):
    f = run_file(run)
    if f is None:
        print(f"[cmp] MISSING trajs for {run}", flush=True)
        return None
    z = np.load(f)
    acc, n = None, 0
    for k in z.files:                               # one [1+S, D] array at a time
        v = z[k].astype(np.float32)
        acc = v if acc is None else acc + v
        n += 1
    print(f"[cmp] {run}: {n} trajs from {os.path.basename(f)}", flush=True)
    return acc / n


def proj_trajs(run, mu, P):
    """Every individual trajectory of a run projected into the joint plane."""
    z = np.load(run_file(run))
    return [(z[k].astype(np.float32) - mu) @ P for k in z.files]


def main():
    ts = record_timesteps()
    flows = {}
    for run in RUNS:
        m = mean_flow(run)
        if m is not None:
            flows[run] = m
    runs = list(flows)
    T = flows[runs[0]].shape[0]

    allp = np.concatenate([flows[r] for r in runs], 0)
    mu = allp.mean(0)
    allp -= mu
    # tiny row count (len(runs) * T) -> exact SVD via the Gram matrix
    G = allp @ allp.T
    lam, u = np.linalg.eigh(G)
    P = allp.T @ (u[:, -2:][:, ::-1] / np.sqrt(np.maximum(lam[-2:][::-1], 1e-6)))
    evr = lam[-2:][::-1] / np.trace(G)
    del allp

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 6.5),
                                        gridspec_kw={"width_ratios": [1.3, 1, 1]})
    for i, r in enumerate(runs):
        c = MCOLORS[i % 10]
        for q in proj_trajs(r, mu, P):              # 32 faint individual trajs
            ax1.plot(q[:, 0], q[:, 1], color=c, alpha=0.18, lw=0.5, zorder=1)
        p2 = (flows[r] - mu) @ P
        ax1.plot(p2[:, 0], p2[:, 1], color=c, lw=2.2, label=r, zorder=3)
        ax1.scatter(p2[::8, 0], p2[::8, 1], color=c, s=14, zorder=4)
        ax1.scatter(p2[0, 0], p2[0, 1], color="black", s=40, zorder=5)
        ax1.scatter(p2[-1, 0], p2[-1, 1], color=c, s=110, marker="*",
                    edgecolor="black", linewidth=0.6, zorder=5)
    ax1.set_title(f"Mean flow path per model — window r{W:02d}, avg over 8 dirs x 4 seeds\n"
                  f"same noise across models; o = shared start, * = final "
                  f"(PC1 {evr[0]*100:.0f}%, PC2 {evr[1]*100:.0f}%)")
    ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2")
    ax1.legend(fontsize=9)

    for i, r in enumerate(runs):
        if r == REF:
            continue
        d = np.linalg.norm(flows[r] - flows[REF], axis=1)
        ax2.semilogy(ts, np.maximum(d, 1e-3), lw=1.8, color=MCOLORS[i % 10], label=r)
    ax2.set_xlim(1010, -10)
    ax2.set_xlabel("diffusion timestep t (1000 = pure noise)")
    ax2.set_ylabel(f"|| mean_m(t) - mean_{REF}(t) ||  (log)")
    ax2.set_title(f"Departure from the {REF} flow")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=9)

    for i, r in enumerate(runs):
        sp = np.linalg.norm(np.diff(flows[r], axis=0), axis=1)
        ax3.plot((ts[:-1] + ts[1:]) / 2, sp, lw=1.8, color=MCOLORS[i % 10], label=r)
    ax3.set_xlim(1010, -10)
    ax3.set_xlabel("diffusion timestep t")
    ax3.set_ylabel("flow speed || mean(t+1) - mean(t) ||")
    ax3.set_title("Where along the schedule each model moves")
    ax3.grid(alpha=0.3); ax3.legend(fontsize=9)

    fig.suptitle("Flow maps aggregated across directions and noise seeds — v14e ablations",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{OUT}/flow_compare_w{W}.png", dpi=130)
    print(f"[cmp] saved {OUT}/flow_compare_w{W}.png", flush=True)

    E = np.stack([flows[r][-1] for r in runs])
    D = np.linalg.norm(E[:, None] - E[None], axis=-1)
    print("[cmp] endpoint distance matrix:")
    print("      " + "  ".join(f"{r:>8s}" for r in runs))
    for i, r in enumerate(runs):
        print(f"{r:>8s} " + "  ".join(f"{D[i, j]:8.1f}" for j in range(len(runs))))


if __name__ == "__main__":
    main()
