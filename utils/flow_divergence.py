"""WHEN do actions bend the flow: divergence vs diffusion timestep, one model.

Reads flow_viz/trajectories_w{W}.npz (utils/flow_viz.py: 8 dirs x FV_NSEEDS
seeds, block-0 x_t recorded at the initial noise + all 48 denoising steps,
same context + same noise per seed). Computes, per recorded step, the mean
pairwise latent distance between the 8 action trajectories (matched seeds),
plus the throttle-only (F-B) and steer-only (L-R) pairs, and maps every
recorded index to its actual diffusion timestep via the FlowMatchScheduler
formula used in stream_causal_chain (shift=5.0, extra_one_step, 48 steps):
record 0 = initial noise at t=1000; record i>=1 = x after step i-1, i.e. at
the step's TARGET sigma. Panels:

  1) separation vs diffusion timestep t (log y), all-pairs + F-B + L-R
  2) fraction of final separation vs t, with 10/50/90% crossing markers
  3) per-step separation increment vs t (where the bending is fastest)

Writes flow_viz/flow_divergence_w{W}.png + prints crossing timesteps.
Env: FV_WINDOW (def 8), FV_NSEEDS (def 4), FV_OUT.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
OUT = os.environ.get("FV_OUT", f"{ARR}/analysis/eval_final/flow_viz")
W = int(os.environ.get("FV_WINDOW", "8"))
NS = int(os.environ.get("FV_NSEEDS", "4"))
DIRS = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]
STEPS = 48
SHIFT = 5.0


def record_timesteps():
    """Diffusion timestep of each recorded point (0 = initial noise)."""
    sig = np.linspace(1.0, 0.0, STEPS + 1)[:-1]          # extra_one_step
    sig = SHIFT * sig / (1 + (SHIFT - 1) * sig)          # shifted sigmas, len 48
    # record i>=1 sits at the step's target sigma (sigmas[i], final step -> 0)
    rec_sig = np.concatenate([[1.0], sig[1:], [0.0]])    # len 49
    return rec_sig * 1000.0


def main():
    z = np.load(f"{OUT}/trajectories_w{W}.npz")
    T = z["F_0"].shape[0]
    ts = record_timesteps()
    assert T == STEPS + 1, f"expected 49 records, got {T}"

    iu = np.triu_indices(len(DIRS), 1)
    i_thr = (DIRS.index("F"), DIRS.index("B"))
    i_ste = (DIRS.index("L"), DIRS.index("R"))
    sep_all = np.zeros(T); sep_thr = np.zeros(T); sep_ste = np.zeros(T)
    for sd in range(NS):                                  # one seed at a time (4GB cap)
        tr = np.stack([z[f"{d}_{sd}"] for d in DIRS])     # [8, T, D] float32
        for si in range(T):
            dd = np.linalg.norm(tr[:, None, si] - tr[None, :, si], axis=-1)
            sep_all[si] += dd[iu].mean() / NS
            sep_thr[si] += dd[i_thr] / NS
            sep_ste[si] += dd[i_ste] / NS
        del tr
        print(f"[div] seed {sd} done", flush=True)

    frac = sep_all / sep_all[-1]
    cross = {}
    for q in (0.10, 0.50, 0.90):
        i = int(np.argmax(frac >= q))
        cross[q] = (i, ts[i])
    dsep = np.diff(sep_all)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 6))
    for ax in (ax1, ax2, ax3):
        ax.set_xlim(1010, -10)                            # denoise left -> right
        ax.set_xlabel("diffusion timestep t (1000 = pure noise)")
        ax.grid(alpha=0.3)

    ax1.semilogy(ts, np.maximum(sep_all, 1e-3), lw=2.4, color="#333", label="all 28 pairs (mean)")
    ax1.semilogy(ts, np.maximum(sep_thr, 1e-3), lw=1.8, color="#1f77b4", ls="--", label="throttle only (F vs B)")
    ax1.semilogy(ts, np.maximum(sep_ste, 1e-3), lw=1.8, color="#2ca02c", ls="--", label="steer only (L vs R)")
    ax1.set_ylabel("inter-action latent distance (log)")
    ax1.set_title(f"Action separation vs diffusion timestep — pca8, window r{W:02d}, {NS} seeds")
    ax1.legend(fontsize=10)

    ax2.plot(ts, frac, lw=2.4, color="#333")
    for q, (i, t) in cross.items():
        ax2.axvline(t, color="#d62728", lw=1, ls=":")
        ax2.annotate(f"{int(q*100)}% @ t={t:.0f}\n(step {i})", (t, q),
                     textcoords="offset points", xytext=(8, -4), fontsize=9, color="#d62728")
        ax2.scatter([t], [frac[i]], color="#d62728", s=30, zorder=5)
    ax2.set_ylabel("fraction of final separation")
    ax2.set_title("Cumulative divergence (share of final inter-action distance)")

    mid_t = (ts[:-1] + ts[1:]) / 2
    ax3.bar(mid_t, dsep, width=np.abs(np.diff(ts)) * 0.85, color="#4c72b0")
    pk = int(np.argmax(dsep))
    ax3.annotate(f"peak: step {pk}->{pk+1}\nt={mid_t[pk]:.0f}", (mid_t[pk], dsep[pk]),
                 textcoords="offset points", xytext=(10, -2), fontsize=10, color="#d62728")
    ax3.set_ylabel("separation gained per step")
    ax3.set_title("Where the action bends the flow fastest")

    fig.suptitle("When actions affect the flow — divergence timing over the 48-step "
                 "denoise (FlowMatchScheduler shift=5: half the steps sit above t=830)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{OUT}/flow_divergence_w{W}.png", dpi=130)
    print(f"[div] saved {OUT}/flow_divergence_w{W}.png", flush=True)
    for q, (i, t) in cross.items():
        print(f"[div] {int(q*100)}% of final separation reached at step {i} (t={t:.0f})", flush=True)
    print(f"[div] fastest bending at step {pk}->{pk+1} (t~{mid_t[pk]:.0f}), "
          f"sep end={sep_all[-1]:.1f}", flush=True)


if __name__ == "__main__":
    main()
