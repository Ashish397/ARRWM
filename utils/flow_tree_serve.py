"""Flow-tree (fan) panels for the serve-variant models — window r08.

One panel per model: the 8 action branches (committed-chunk trajectories,
chunks 0..5, averaged over seeds; faint = per-seed) projected on a SHARED
2-PC plane fitted on the dense teacher's committed blocks, so length =
real latent distance in a common basis. Dashed ellipse = 2-sigma of the
final-chunk cloud (all dirs/seeds): its size vs the teacher's shows
dispersal retention; branch length shows motion/contraction.

Serve transforms (lock/hyb/chroma) are re-applied to the recorded
pre-commit latents to obtain the true EMITTED latents (deterministic).

Env: FT_OUT (default flow_viz/flow_tree_serve.png)
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
FV = f"{ARR}/analysis/eval_final/flow_viz"
DIRS = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]
DCOL = {"F": "#d62728", "FR": "#ff7f0e", "R": "#bcbd22", "BR": "#2ca02c",
        "B": "#17becf", "BL": "#1f77b4", "L": "#9467bd", "FL": "#e377c2"}
NFB, C = 3, 16


def seed_stats():
    sys.path.insert(0, ARR)
    from utils.zarr_dataset import ZarrRideDataset
    w = json.load(open(f"{ARR}/analysis/eval_final/phaseA_windows.json"))[8]
    s = np.asarray(ZarrRideDataset.load_latent_chunk(
        w["zarr_path"], int(w["offset"]), int(w["offset"]) + NFB), dtype=np.float32)
    mu = s.reshape(NFB, C, -1).mean(axis=(0, 2))
    sd = s.reshape(NFB, C, -1).std(axis=(0, 2))
    return mu, sd


SEED_MU, SEED_SD = None, None
CHROMA = None


def xform(x, mode):
    """x [NFB, C, H, W] pre-commit -> emitted latent under serve mode."""
    global CHROMA
    if mode == "none":
        return x
    mu = x.reshape(NFB, C, -1).mean(axis=(0, 2))
    sd = x.reshape(NFB, C, -1).std(axis=(0, 2))
    xm = x - mu[None, :, None, None]
    if mode == "lock":
        return (xm / (sd[None, :, None, None] + 1e-6) * SEED_SD[None, :, None, None]
                + SEED_MU[None, :, None, None])
    if mode == "hyb":
        kmu = min(1.0 / 0.894, 1.2)
        return (xm / (sd[None, :, None, None] + 1e-6) * SEED_SD[None, :, None, None]
                + (mu * kmu)[None, :, None, None])
    if mode == "inv":
        ksd = min(1.0 / 0.9565, 1.1)
        kmu = min(1.0 / 0.894, 1.2)
        return xm * ksd + (mu * kmu)[None, :, None, None]
    if mode == "chroma":
        if CHROMA is None:
            cz = np.load(f"{FV}/chroma_subspace.npz")
            Mc = cz["Mc"]; CHROMA = (Mc, np.linalg.pinv(Mc))
        Mc, Mp = CHROMA
        dmu = Mp @ (Mc @ (SEED_MU - mu))
        return (xm / (sd[None, :, None, None] + 1e-6) * SEED_SD[None, :, None, None]
                + (mu + dmu)[None, :, None, None])
    raise ValueError(mode)


def committed(run, mode, nseeds=2):
    out = {}
    for d in DIRS:
        for sd_i in range(nseeds):
            p = f"{FV}/flow_{run}/r08_{d}_s{sd_i}/steps.npz"
            if not os.path.exists(p):
                continue
            z = np.load(p); sdt = z["sdt"]
            last = {}
            for i, (c, r, t) in enumerate(sdt):
                if t > 0 and int(r) >= 0:
                    last[int(c)] = i
            tr = []
            for c in sorted(last):
                x = z[f"x{last[c]}"].astype(np.float32).reshape(NFB, C, 60, 104)
                tr.append(xform(x, mode).ravel())
            out[(d, sd_i)] = np.stack(tr)
    return out


def teacher_committed(nblocks=6, nseeds=4):
    z = np.load(f"{FV}/trajs_14e8s20_w8.npz")
    out = {}
    for d in DIRS:
        for sd_i in range(nseeds):
            tr = []
            for b in range(nblocks):
                k = f"{d}_{sd_i}" if b == 0 else f"b{b}_{d}_{sd_i}"
                if k in z.files:
                    tr.append(z[k][-1].astype(np.float32))
            if tr:
                out[(d, sd_i)] = np.stack(tr)
    return out


def main():
    global SEED_MU, SEED_SD
    SEED_MU, SEED_SD = seed_stats()
    if os.environ.get("FT_SET", "B") == "A":
        # Grid-A lineup: the serve-sampler series on the nr checkpoint.
        panels = [
            ("teacher 20-step", "T", None),
            ("teacher-init 4-rung", "pilot_gt0", "none"),
            ("baseline (restart)", "pilot3_flip2nr", "none"),
            ("det (transport)", "pilot3_flip2nr_det", "none"),
            ("inv (bias inversion)", "pilot3_flip2nr_inv", "inv"),
            ("det+inv", "pilot3_flip2nr_detinv", "inv"),
            ("hybrid (std-pin)", "pilot3_flip2nr_hyb", "hyb"),
            ("full LOCK", "pilot3_flip2nr_lock", "lock"),
        ]
        default_out = f"{FV}/flow_tree_gridA.png"
    else:
        panels = [
            ("teacher 20-step", "T", None),
            ("teacher-init 4-rung", "pilot_gt0", "none"),
            ("nr (restart)", "pilot3_flip2nr", "none"),
            ("nr + full LOCK", "pilot3_flip2nr_lock", "lock"),
            ("nr + chroma", "pilot4_nr_chroma", "chroma"),
            ("nr + hybrid", "pilot4_nr_hyb", "hyb"),
            ("nr+critic + chroma", "pilot4_nrcg_chroma", "chroma"),
            ("nr+critic + hybrid", "pilot4_nrcg_hyb", "hyb"),
        ]
        default_out = f"{FV}/flow_tree_serve.png"
    tea = teacher_committed()
    allT = np.stack([v for tr in tea.values() for v in tr])
    m0 = allT.mean(0)
    R = allT - m0
    G = R @ R.T
    w_, v_ = np.linalg.eigh(G)
    pcs = (R.T @ v_[:, -2:]) / np.sqrt(np.maximum(w_[-2:], 1e-9))  # [D,2]

    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    for ax, (title, run, mode) in zip(axes.ravel(), panels):
        data = tea if run == "T" else committed(run, mode)
        if not data:
            ax.set_title(f"{title} (missing)"); continue
        ends = []
        for d in DIRS:
            trs = [((tr - m0) @ pcs) for (dd, s), tr in data.items() if dd == d]
            if not trs:
                continue
            for t2 in trs:
                ax.plot(np.r_[0, t2[:, 0]], np.r_[0, t2[:, 1]],
                        color=DCOL[d], alpha=0.25, lw=1.0)
                ends.append(t2[-1])
            mtr = np.mean(trs, axis=0)
            ax.plot(np.r_[0, mtr[:, 0]], np.r_[0, mtr[:, 1]],
                    color=DCOL[d], lw=2.2, label=d)
            ax.plot(mtr[-1, 0], mtr[-1, 1], "*", color=DCOL[d], ms=12)
        E = np.stack(ends)
        mu2 = E.mean(0); cov = np.cov((E - mu2).T)
        ev, evec = np.linalg.eigh(cov)
        th = np.linspace(0, 2 * np.pi, 100)
        ell = (evec @ np.diag(2 * np.sqrt(np.maximum(ev, 0))) @
               np.stack([np.cos(th), np.sin(th)])).T + mu2
        ax.plot(ell[:, 0], ell[:, 1], "k--", lw=1.2, alpha=0.7)
        disp = float(np.sqrt(np.trace(cov)))
        ax.set_title(f"{title}\nfinal-cloud 2$\\sigma$ dispersal={disp:.0f}")
        ax.set_aspect("equal"); ax.axhline(0, color="k", lw=0.3); ax.axvline(0, color="k", lw=0.3)
    axes[0, 0].legend(fontsize=8, ncol=2)
    for ax in axes.ravel():
        ax.set_xlim(-900, 900); ax.set_ylim(-900, 900)
    out = os.environ.get("FT_OUT", default_out)
    fig.suptitle("Flow trees: committed-chunk trajectories in the shared teacher PCA plane "
                 "(branch length = motion; ellipse = final dispersal)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"[ft] saved {out}")


if __name__ == "__main__":
    main()
