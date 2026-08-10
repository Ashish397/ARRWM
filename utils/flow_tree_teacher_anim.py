"""Animate the dense teacher's flow tree through its 48 denoising steps.

Frame s = the same tree diagram as flow_tree_serve.py, but built from the
teacher's INTERMEDIATE state x_s at step s (per block, per action; block
trajectories drawn cumulatively origin->block0..block5), projected on the
fixed teacher-committed PCA plane. Watching s advance shows WHEN in the
schedule the action-tree structure emerges. Frames nearest the 4-rung
ladder's t-values (1000, 625, 357.1, 208.3) are flagged red — that is all
the 4-rung sampler ever sees.

Output: flow_viz/flow_tree_teacher_anim.gif (and .mp4). CPU, streams keys.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

FV = "/scratch/u6ex/as1748.u6ex/ARRWM/analysis/eval_final/flow_viz"
DIRS = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]
DCOL = {"F": "#d62728", "FR": "#ff7f0e", "R": "#bcbd22", "BR": "#2ca02c",
        "B": "#17becf", "BL": "#1f77b4", "L": "#9467bd", "FL": "#e377c2"}
NBLK, NSEEDS = 6, 4
RUNG_T = [1000.0, 625.0, 357.142857, 208.333333]


def main():
    # PCA plane fitted on 20-step committed blocks (same basis as the
    # static figures, so shapes/scales match across all tree plots).
    z20 = np.load(f"{FV}/trajs_14e8s20_w8.npz")
    allT = []
    for d in DIRS:
        for s in range(NSEEDS):
            for b in range(NBLK):
                k = f"{d}_{s}" if b == 0 else f"b{b}_{d}_{s}"
                if k in z20.files:
                    allT.append(z20[k][-1].astype(np.float32))
    allT = np.stack(allT)
    m0 = allT.mean(0)
    R = allT - m0
    G = R @ R.T
    w_, v_ = np.linalg.eigh(G)
    pcs = (R.T @ v_[:, -2:]) / np.sqrt(np.maximum(w_[-2:], 1e-9))

    z48 = np.load(f"{FV}/trajs_14e8_w8.npz")
    nrow = None
    proj = {}                                # (d, s, b) -> [nrow, 2]
    for d in DIRS:
        for s in range(NSEEDS):
            for b in range(NBLK):
                k = f"{d}_{s}" if b == 0 else f"b{b}_{d}_{s}"
                if k not in z48.files:
                    continue
                arr = z48[k].astype(np.float32)
                if nrow is None:
                    nrow = arr.shape[0]
                proj[(d, s, b)] = (arr - m0[None]) @ pcs
                del arr
    print(f"[anim] projected {len(proj)} block-trajectories, {nrow} rows each")

    nst = nrow - 1
    ts = [1000.0 * 5 * u / (1 + 4 * u) for u in (1 - i / nst for i in range(nst))] + [0.0]
    rung_rows = [int(np.argmin([abs(t - rt) for t in ts])) for rt in RUNG_T]

    frames = []
    for s_i in range(nrow):
        fig, ax = plt.subplots(figsize=(7, 7))
        ends = []
        for d in DIRS:
            per_seed = []
            for sd in range(NSEEDS):
                tr = np.stack([proj[(d, sd, b)][s_i] for b in range(NBLK)
                               if (d, sd, b) in proj])
                ax.plot(np.r_[0, tr[:, 0]], np.r_[0, tr[:, 1]],
                        color=DCOL[d], alpha=0.25, lw=1.0)
                ends.append(tr[-1]); per_seed.append(tr)
            mtr = np.mean(per_seed, axis=0)
            ax.plot(np.r_[0, mtr[:, 0]], np.r_[0, mtr[:, 1]],
                    color=DCOL[d], lw=2.2, label=d)
            ax.plot(mtr[-1, 0], mtr[-1, 1], "*", color=DCOL[d], ms=12)
        E = np.stack(ends)
        cov = np.cov((E - E.mean(0)).T)
        ev, evec = np.linalg.eigh(cov)
        th = np.linspace(0, 2 * np.pi, 100)
        ell = (evec @ np.diag(2 * np.sqrt(np.maximum(ev, 0))) @
               np.stack([np.cos(th), np.sin(th)])).T + E.mean(0)
        ax.plot(ell[:, 0], ell[:, 1], "k--", lw=1.2, alpha=0.7)
        disp = float(np.sqrt(np.trace(cov)))
        is_rung = s_i in rung_rows
        tag = f"  <== ~4-RUNG ladder input t={RUNG_T[rung_rows.index(s_i)]:.0f}" if is_rung else ""
        ax.set_title(f"teacher 48-step | step {s_i}/{nst}  t={ts[s_i]:.0f}  "
                     f"dispersal={disp:.0f}{tag}",
                     color=("crimson" if is_rung else "black"))
        for sp in ax.spines.values():
            sp.set_edgecolor("crimson" if is_rung else "black")
            sp.set_linewidth(3 if is_rung else 1)
        ax.set_xlim(-900, 900); ax.set_ylim(-900, 900); ax.set_aspect("equal")
        ax.axhline(0, color="k", lw=0.3); ax.axvline(0, color="k", lw=0.3)
        ax.legend(fontsize=7, ncol=2, loc="lower left")
        fig.tight_layout()
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(img)
        plt.close(fig)
    # hold the rung frames + final a bit longer
    seq = []
    for i, f in enumerate(frames):
        seq.append(f)
        if i in rung_rows or i == len(frames) - 1:
            seq.extend([f] * 2)
    imageio.mimsave(f"{FV}/flow_tree_teacher_anim.gif", seq, fps=5, loop=0)
    imageio.mimsave(f"{FV}/flow_tree_teacher_anim.mp4", seq, fps=5, quality=8)
    print(f"[anim] saved {FV}/flow_tree_teacher_anim.gif + .mp4 ({len(frames)} steps)")


if __name__ == "__main__":
    main()
