"""AR-timeline flow-tree animations (2D + 3D), window r08 — one renderer.

Panels/animations: teacher 48-step, teacher-init 4-rung, lock serve
(committed vertices lock-transformed). Styling: GT real-video line GREY
(zorder-top, first in legend), branches muted, grey X = noise start of
the current chunk (seed-shared across dirs). GT/TF overlays optional.

Env: TL_WHICH csv of {teacher,4rung,lock} (default all), TL_3D=1 adds 3D
versions, TL_TF_RUN name of a flow_{run} TF recording to overlay (adds
dashed dark-green teacher-forced line when present).
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
FV = f"{ARR}/analysis/eval_final/flow_viz"
DIRS = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]
DCOL = {"F": "#d62728", "FR": "#ff7f0e", "R": "#bcbd22", "BR": "#2ca02c",
        "B": "#17becf", "BL": "#1f77b4", "L": "#9467bd", "FL": "#e377c2",
        "N": "#8c564b"}
NBLK, NFB, C = 6, 3, 16
GT_COL = "0.55"
NOISE_COL = "0.2"
TF_COL = "#006400"


def fit_basis(k3=False):
    z20 = np.load(f"{FV}/trajs_14e8s20_w8.npz")
    allT = []
    for d in DIRS:
        for s in range(4):
            for b in range(NBLK):
                k = f"{d}_{s}" if b == 0 else f"b{b}_{d}_{s}"
                if k in z20.files:
                    allT.append(z20[k][-1].astype(np.float32))
    allT = np.stack(allT); m0 = allT.mean(0); R = allT - m0
    G = R @ R.T
    w_, v_ = np.linalg.eigh(G)
    n = 3 if k3 else 2
    return m0, (R.T @ v_[:, -n:]) / np.sqrt(np.maximum(w_[-n:], 1e-9))


def gt_latents():
    sys.path.insert(0, ARR)
    from utils.zarr_dataset import ZarrRideDataset
    w = json.load(open(f"{ARR}/analysis/eval_final/phaseA_windows.json"))[8]
    return np.asarray(ZarrRideDataset.load_latent_chunk(
        w["zarr_path"], int(w["offset"]), int(w["offset"]) + NFB * 7), dtype=np.float32)


def stage_of(r, t):
    if r == -1:
        return 0
    if t < 0:
        for i, tv in enumerate([625.0, 357.14, 208.33]):
            if abs(abs(t) - tv) < 1:
                return i + 1
    if t > 0 and r == 3:
        return 4
    return None


def load_run(run, m0, pcs, nseeds=2, lockx=None):
    traj = {}
    for d in DIRS + ["N"]:
        for sd in range(nseeds):
            p = f"{FV}/flow_{run}/r08_{d}_s{sd}/steps.npz"
            if not os.path.exists(p):
                continue
            z = np.load(p); sdt = z["sdt"]
            per = {}
            for i, (c, r, t) in enumerate(sdt):
                si = stage_of(int(r), float(t))
                if si is None:
                    continue
                x = z[f"x{i}"].astype(np.float32).reshape(NFB, C, 60, 104)
                if si == 4 and lockx is not None:
                    x = lockx(x)
                per.setdefault(int(c), {})[si] = (x.ravel() - m0) @ pcs
            traj[(d, sd)] = per
    return traj


def load_tf(run, m0, pcs):
    pts = []
    for c in range(1, NBLK + 1):
        p = f"{FV}/flow_{run}/r08_TF_c{c}/steps.npz"
        if not os.path.exists(p):
            return None
        z = np.load(p); sdt = z["sdt"]
        last = None
        for i, (cc, r, t) in enumerate(sdt):
            if t > 0 and int(r) == 3:
                last = i
        pts.append((z[f"x{last}"].astype(np.float32).ravel() - m0) @ pcs)
    return np.stack(pts)


def render(name, stages, gt_line, out, three_d=False, tf_line=None, seed_pt=None):
    frames = []; az = 0.0
    for (title, per_dir, noise_pts) in stages:
        _o = seed_pt if seed_pt is not None else np.zeros(3)
        if three_d:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection="3d")
            P = lambda a: (np.r_[_o[0], a[:, 0]], np.r_[_o[1], a[:, 1]], np.r_[_o[2], a[:, 2]])
        else:
            fig, ax = plt.subplots(figsize=(7, 7))
            P = lambda a: (np.r_[_o[0], a[:, 0]], np.r_[_o[1], a[:, 1]])
        h_gt, = ax.plot(*P(gt_line), ":", color=GT_COL, lw=2.8, zorder=10, alpha=0.2,
                        label="GT (real video)",
                        marker="s", ms=7, mfc=GT_COL, mec="0.25")
        handles = [h_gt]
        if tf_line is not None:
            h_tf, = ax.plot(*P(tf_line), "--", color=TF_COL, lw=2.6, zorder=10,
                            label="teacher-forced (GT ctx)")
            handles.append(h_tf)
        if seed_pt is not None:
            if three_d:
                h_s = ax.scatter([_o[0]], [_o[1]], [_o[2]], marker="D", s=90,
                                 color="#444444", zorder=12, label="SEED chunk (real)")
            else:
                h_s = ax.scatter([_o[0]], [_o[1]], marker="D", s=110,
                                 color="#444444", zorder=12, label="SEED chunk (real)")
            handles.append(h_s)
        if noise_pts is not None and len(noise_pts):
            npm = np.mean(noise_pts, axis=0)
            if three_d:
                h_n = ax.scatter([npm[0]], [npm[1]], [npm[2]], marker="x", s=150,
                                 color=NOISE_COL, linewidths=3, label="noise start")
            else:
                h_n = ax.scatter([npm[0]], [npm[1]], marker="x", s=170,
                                 color=NOISE_COL, linewidths=3, zorder=9,
                                 label="noise start")
            handles.append(h_n)
        for d in [dd for dd in DIRS + ['N'] if per_dir.get(dd)]:
            trs = per_dir.get(d, [])
            for tr in trs:
                ax.plot(*P(tr), color=DCOL[d], alpha=0.22, lw=0.9)
            mtr = np.mean(trs, axis=0)
            h, = ax.plot(*P(mtr), color=DCOL[d], lw=2.0, alpha=0.85, label=d)
            handles.append(h)
            if three_d:
                ax.scatter(mtr[-1:, 0], mtr[-1:, 1], mtr[-1:, 2],
                           facecolors="none", edgecolors=DCOL[d], s=80,
                           linewidths=1.6, alpha=0.95)
            else:
                ax.plot(mtr[:-1, 0], mtr[:-1, 1], "o", color=DCOL[d], ms=3.5, alpha=0.85)
                ax.plot(mtr[-1, 0], mtr[-1, 1], "o", mfc="none", mec=DCOL[d],
                        ms=10, mew=1.6, alpha=0.95)
        ax.set_title(title, fontsize=9.5)
        lim = 800 if three_d else 900
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        if three_d:
            ax.set_zlim(-lim, lim); ax.view_init(elev=22, azim=az); az += 2.0
        else:
            ax.set_aspect("equal")
            ax.axhline(0, color="k", lw=0.3); ax.axvline(0, color="k", lw=0.3)
        ax.legend(handles=handles, fontsize=6.5, ncol=2,
                  loc=("upper left" if three_d else "lower left"))
        fig.tight_layout(); fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(out, frames, fps=5, quality=8)
    print(f"[tl] saved {out} ({len(frames)} frames)")


def main():
    which = os.environ.get("TL_WHICH", "teacher,4rung,lock,badmts").split(",")
    three_d = bool(os.environ.get("TL_3D"))
    for k3 in ([False, True] if three_d else [False]):
        m0, pcs = fit_basis(k3)
        gt = gt_latents()
        gt_line = np.stack([((gt[NFB*c:NFB*(c+1)].ravel() - m0) @ pcs)
                            for c in range(1, 7)])
        seed_pt = (gt[:NFB].ravel() - m0) @ pcs   # TRUE seed-chunk projection
        seed = gt[:NFB]
        smu = seed.reshape(NFB, C, -1).mean(axis=(0, 2))
        ssd = seed.reshape(NFB, C, -1).std(axis=(0, 2))

        def lockx(x, smu=smu, ssd=ssd):
            mu = x.reshape(NFB, C, -1).mean(axis=(0, 2))
            sd = x.reshape(NFB, C, -1).std(axis=(0, 2))
            return ((x - mu[None, :, None, None]) / (sd[None, :, None, None] + 1e-6)
                    * ssd[None, :, None, None] + smu[None, :, None, None])

        tf_line = None
        tf_run = os.environ.get("TL_TF_RUN", "tf_gt0")
        tf_line = load_tf(tf_run, m0, pcs)
        sfx = "_3d" if k3 else ""
        LBL = ["fresh noise t=1000", "re-noised @625", "re-noised @357",
               "re-noised @208", "COMMITTED"]

        if "teacher" in which:
            z48 = np.load(f"{FV}/trajs_14e8_w8.npz")
            proj = {}
            for d in DIRS:
                for s in range(4):
                    for b in range(NBLK):
                        k = f"{d}_{s}" if b == 0 else f"b{b}_{d}_{s}"
                        if k in z48.files:
                            proj[(d, s, b)] = (z48[k].astype(np.float32) - m0[None]) @ pcs
            _np_path = f"{FV}/trajs_14e8noop_w8.npz"
            if os.path.exists(_np_path):
                zn = np.load(_np_path)
                for s in range(4):
                    for b in range(NBLK):
                        k = f"N_{s}" if b == 0 else f"b{b}_N_{s}"
                        if k in zn.files:
                            proj[("N", s, b)] = (zn[k].astype(np.float32) - m0[None]) @ pcs
            nrow = next(iter(proj.values())).shape[0]; nst = nrow - 1
            ts = [1000.0*5*u/(1+4*u) for u in (1 - i/nst for i in range(nst))] + [0.0]
            stages = []
            for b_cur in range(NBLK):
                noise = [proj[(d, sd, b_cur)][0] for d in DIRS for sd in range(4)]
                for s_i in list(range(0, nrow, 3)) + [nrow - 1]:
                    per = {d: [np.stack([proj[(d, sd, b)][-1] for b in range(b_cur)]
                                        + [proj[(d, sd, b_cur)][s_i]])
                               for sd in range(4) if (d, sd, b_cur) in proj]
                           for d in DIRS + ["N"]}
                    per = {d: v for d, v in per.items() if v}
                    stages.append((f"TEACHER 48-step AR TIMELINE | committed: {b_cur} | "
                                   f"block {b_cur}: step {s_i}/{nst} (t={ts[s_i]:.0f})",
                                   per, noise))
                stages.extend([stages[-1]] * 4)
            render("teacher", stages, gt_line,
                   f"{FV}/flow_tree_teacher_AR_timeline{sfx}.mp4", k3, tf_line, seed_pt)

        for label, run, lx in [("4rung", "pilot_gt0", None),
                               ("lock", "pilot3_flip2nr_lock", lockx),
                               ("badmts", "pilot3_flip2mts", None),
                               ("nrmom2", "pilot3_flip2nrmom2", None),
                               ("v1std", "pilot2_flip2", None),
                               ("c10mse", "pilot7_c10mse_s0002800", None),
                               ("c10kl", "pilot7_c10kl_s0002800", None),
                               ("c10mserep", "pilot7_c10mserep_s0002800", None),
                               ("c10klrep", "pilot7_c10klrep_s0002800", None),
                               ("c10mserep2", "pilot7_c10mserep2_s0002092", None),
                               ("alldir8", "pilot4_alldir82", None),
                               ("alldir8kl", "pilot4_alldir8kl2", None),
                               # grid-parity set (2026-08-08): every method
                               # that appears as a panel in GRID3..GRID7
                               ("nr", "pilot3_flip2nr", None),
                               ("alldir8n", "pilot4_alldir8n2", None),
                               ("msecg8n", "pilot4_msecg8n2", None),
                               ("kl8n", "pilot4_kl8n2", None),
                               ("emd1", "pilot4_emd12", None),
                               ("emd2", "pilot4_emd22", None),
                               ("emdc", "pilot4_emdc2", None),
                               ("emdz", "pilot4_emdz2", None),
                               ("a8n_rm10", "pilot5_a8n_rm10", None),
                               ("a8n_rmauto2", "pilot5_a8n_rmauto2", None),
                               ("nr_rm05", "pilot3_nr_emdremap05", None),
                               ("nr_rm10", "pilot3_nr_emdremap10", None),
                               # serve-sampler series (GRID) + critic/chroma (GRID2).
                               # No lock transform: these recordings already
                               # contain the SERVED states.
                               ("det", "pilot3_flip2nr_det", None),
                               ("inv", "pilot3_flip2nr_inv", None),
                               ("detinv", "pilot3_flip2nr_detinv", None),
                               ("hyb", "pilot3_flip2nr_hyb", None),
                               ("nr_chroma", "pilot4_nr_chroma", None),
                               ("nr_hyb", "pilot4_nr_hyb", None),
                               ("nrcg_chroma", "pilot4_nrcg_chroma", None),
                               ("nrcg_hyb", "pilot4_nrcg_hyb", None)] + [
                               # TL_CUSTOM="label=flow_run_name,..." appends
                               # arbitrary recordings (e.g. the per-checkpoint
                               # v100_* probes) without touching this table.
                               (kv.split("=", 1)[0], kv.split("=", 1)[1], None)
                               for kv in os.environ.get("TL_CUSTOM", "").split(",")
                               if "=" in kv]:
            if label not in which:
                continue
            traj = load_run(run, m0, pcs, 2, lx)
            _noop_run = {"4rung": "pilot_gt0noop",
                         "alldir8n": "pilot5_a8n_noop",
                         "a8n_rmauto2": "pilot5_a8n_noop_auto"}.get(label)
            if _noop_run and os.path.isdir(f"{FV}/flow_{_noop_run}"):
                for kk, vv in load_run(_noop_run, m0, pcs, 2, None).items():
                    traj[("N", kk[1])] = vv
            stages = []
            for b_cur in range(NBLK):
                noise = [traj[(d, sd)][b_cur][0] for d in DIRS for sd in range(2)
                         if (d, sd) in traj and 0 in traj[(d, sd)].get(b_cur, {})]
                for si in range(5):
                    per = {d: [np.stack([traj[(d, sd)][b][4] for b in range(b_cur)]
                                        + [traj[(d, sd)][b_cur][si]])
                               for sd in range(2) if (d, sd) in traj
                               and b_cur in traj[(d, sd)]] for d in DIRS + ["N"]}
                    per = {d: v for d, v in per.items() if v}
                    hdr = {"4rung": "4-RUNG", "lock": "LOCK serve",
                           "badmts": "BAD DISTILL (mts arm)",
                           "nrmom2": "NRMOM2 (moment loss)",
                           "v1std": "STANDARD ODE (v1 MSE)",
                           "c10mse": "C10MSE (curriculum, MSE)",
                           "c10kl": "C10KL (curriculum, KL)",
                           "c10mserep": "C10MSEREP (MSE + repulsor)",
                           "c10klrep": "C10KLREP (KL + repulsor)",
                           "c10mserep2": "C10MSEREP2 (v2: backward slot)",
                           "alldir8": "ALLDIR8 (8-dir global batch, MSE)",
                           "alldir8kl": "ALLDIR8-KL (8-dir global batch, KL)",
                           "nr": "NR (next-rung targets)",
                           "alldir8n": "ALLDIR8N (global batch MSE, 8dir+noop)",
                           "msecg8n": "MSE + CRITIC 0.75",
                           "kl8n": "KL (normal pairs)",
                           "emd1": "EMD1 (every-rung 1/d2)",
                           "emd2": "EMD2 (every-rung 1/dd)",
                           "emdc": "EMDC (commit-clock)",
                           "emdz": "EMDZ (zero-init transport head)",
                           "a8n_rm10": "ALLDIR8N + EMD remap lam=1.0",
                           "a8n_rmauto2": "ALLDIR8N + EMD remap ADAPTIVE",
                           "nr_rm05": "NR + EMD remap lam=0.5",
                           "nr_rm10": "NR + EMD remap lam=1.0",
                           "det": "DET (transport serve)",
                           "inv": "INV (bias inversion serve)",
                           "detinv": "DET+INV serve",
                           "hyb": "HYBRID (std-pin serve)",
                           "nr_chroma": "NR + CHROMA-lock",
                           "nr_hyb": "NR + hybrid",
                           "nrcg_chroma": "NR+CRITIC + chroma",
                           "nrcg_hyb": "NR+CRITIC + hybrid"}.get(
                               label, label.upper())
                    stages.append((f"{hdr} AR TIMELINE | committed: {b_cur} | "
                                   f"chunk {b_cur}: {LBL[si]}", per, noise))
                stages.extend([stages[-1]] * 4)
            render(label, stages, gt_line,
                   f"{FV}/flow_tree_{label}_AR_timeline{sfx}.mp4", k3, tf_line, seed_pt)


if __name__ == "__main__":
    main()
