"""World-time flow comparison INCLUDING external models (no denoising access).

Comparators expose no diffusion states, so this compares flow through latent
space over VIDEO time using the existing world_lats_{run}.npz caches (same 32
phase-A scenes for every model, same Wan VAE). Per model: average the 8
action videos of each window, rebase at the window's first latent frame
(displacement from the shared real-seed start), then average the 32 windows
-> one mean world-flow path [Fl, D]. Panels:

  1) mean displacement paths in a joint 2-PC plane — only runs sharing
     D=99840 (matrixgame 56320 / yume 225280 are dimension-incompatible)
  2) distance from the pca8 path at matched normalized video time (same-D runs)
  3) cumulative normalized arc length vs normalized time — dimension-free,
     ALL runs: how front-loaded each model's latent motion is

Writes flow_viz/world_flow_compare.png. Env: WFC_RUNS colon list
(def pca8_8node:16node:minwm:worldcam:worldplay:astra:matrixgame:yume),
WFC_OUT.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
FV = f"{ARR}/analysis/eval_final/flow_viz"
OUT = os.environ.get("WFC_OUT", FV)
RUNS = os.environ.get("WFC_RUNS",
                      "pca8_8node:16node:minwm:worldcam:worldplay:astra:matrixgame:yume").split(":")
REF = RUNS[0]
DNAMES = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]
MCOLORS = plt.get_cmap("tab10").colors
GRID = np.linspace(0, 1, 50)


def mean_path(run):
    """Across-window mean of dir-averaged, start-rebased latent paths."""
    z = np.load(f"{FV}/world_lats_{run}.npz")
    keys = {}
    for k in z.files:
        wpart, d = k.rsplit("_", 1)
        keys.setdefault(int(wpart[1:]), {})[d] = k
    wins = sorted(w for w, ds in keys.items() if all(d in ds for d in DNAMES))
    Fl = min(z[keys[w][d]].shape[0] for w in wins for d in DNAMES)
    acc = None
    for w in wins:                                   # one window at a time (4GB cap)
        tr = np.stack([z[keys[w][d]][:Fl].astype(np.float32) for d in DNAMES]).mean(0)
        tr -= tr[0:1]                                # displacement from seed start
        acc = tr if acc is None else acc + tr
    m = acc / len(wins)
    print(f"[wfc] {run}: {len(wins)} windows x {Fl} frames, D={m.shape[1]}", flush=True)
    return m


def main():
    paths = {r: mean_path(r) for r in RUNS}
    Dref = paths[REF].shape[1]
    joint = [r for r in RUNS if paths[r].shape[1] == Dref]
    skipped = [r for r in RUNS if r not in joint]
    if skipped:
        print(f"[wfc] excluded from joint plane (latent dim mismatch): {skipped}", flush=True)

    allp = np.concatenate([paths[r] for r in joint], 0)
    mu = allp.mean(0)
    allp -= mu
    G = allp @ allp.T
    lam, u = np.linalg.eigh(G)
    P = allp.T @ (u[:, -2:][:, ::-1] / np.sqrt(np.maximum(lam[-2:][::-1], 1e-6)))
    evr = lam[-2:][::-1] / np.trace(G)
    del allp

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 6.5),
                                        gridspec_kw={"width_ratios": [1.3, 1, 1]})
    ci = {r: MCOLORS[i % 10] for i, r in enumerate(RUNS)}

    for r in joint:
        p2 = (paths[r] - mu) @ P
        ax1.plot(p2[:, 0], p2[:, 1], color=ci[r], lw=2.0, label=r)
        ax1.scatter(p2[0, 0], p2[0, 1], color="black", s=40, zorder=5)
        ax1.scatter(p2[-1, 0], p2[-1, 1], color=ci[r], s=110, marker="*",
                    edgecolor="black", linewidth=0.6, zorder=5)
    ax1.set_title(f"Mean world-flow path (displacement from seed frame), joint plane\n"
                  f"32 scenes x 8 actions averaged; o = seed start, * = final frame "
                  f"(PC1 {evr[0]*100:.0f}%, PC2 {evr[1]*100:.0f}%)"
                  + (f"\nexcluded (latent-dim mismatch): {', '.join(skipped)}" if skipped else ""))
    ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2")
    ax1.legend(fontsize=9)

    def resample(m):
        t = np.linspace(0, 1, m.shape[0])
        return np.stack([np.interp(GRID, t, m[:, j]) for j in range(2)], 1)

    # matched-normalized-time distance needs full D; interp in full D per grid point
    ref = paths[REF]
    tref = np.linspace(0, 1, ref.shape[0])
    for r in joint:
        if r == REF:
            continue
        m = paths[r]
        tm = np.linspace(0, 1, m.shape[0])
        d = []
        for g in GRID:
            a = ref[np.argmin(np.abs(tref - g))]
            b = m[np.argmin(np.abs(tm - g))]
            d.append(np.linalg.norm(a - b))
        ax2.plot(GRID, d, lw=1.8, color=ci[r], label=r)
    ax2.set_xlabel("normalized video time")
    ax2.set_ylabel(f"|| path_m - path_{REF} ||")
    ax2.set_title(f"Departure from the {REF} world-flow path")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=9)

    for r in RUNS:
        m = paths[r]
        sp = np.linalg.norm(np.diff(m, axis=0), axis=1)
        arc = np.concatenate([[0], np.cumsum(sp)])
        arc /= max(arc[-1], 1e-8)
        ax3.plot(np.linspace(0, 1, m.shape[0]), arc, lw=1.8, color=ci[r],
                 ls="--" if r in skipped else "-", label=r + (" (dim-mismatch)" if r in skipped else ""))
    ax3.plot([0, 1], [0, 1], color="gray", lw=0.8, ls=":")
    ax3.set_xlabel("normalized video time")
    ax3.set_ylabel("cumulative fraction of latent path length")
    ax3.set_title("How front-loaded each model's latent motion is (all models)")
    ax3.grid(alpha=0.3); ax3.legend(fontsize=9)

    fig.suptitle("World-time flow comparison — ours (pca8, 16node) vs external models, "
                 "from the same 32 eval scenes (VAE re-encoded videos)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(f"{OUT}/world_flow_compare.png", dpi=130)
    print(f"[wfc] saved {OUT}/world_flow_compare.png", flush=True)

    E = {r: paths[r][-1] for r in joint}
    print("[wfc] endpoint distance matrix (joint-plane runs):")
    print("      " + "  ".join(f"{r:>10s}" for r in joint))
    for a in joint:
        print(f"{a:>10s} " + "  ".join(f"{np.linalg.norm(E[a] - E[b]):10.1f}" for b in joint))


if __name__ == "__main__":
    main()
