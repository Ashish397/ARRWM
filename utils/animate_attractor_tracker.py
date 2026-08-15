"""Animate the AttractorTracker estimating on real recordings.

One frame per sync step: the LIVE per-direction fixed-point estimates (hollow
stars) wander while the estimator accumulates evidence; every freeze_k steps
the FROZEN actuation reference (filled stars, size/alpha = gate) snaps to the
current estimate — the target-network cadence the trainer will run. Faint
lines = the recorded rollout trajectories (chunks 0-5); a fading trail
follows each frozen star so the tracking motion is visible.

Env: AT_RUN (pilot_gt0), AT_PASSES (60), AT_FPS (6), AT_OUT (mp4 path).
CPU-only. Reuses the loaders from utils/test_attractor_tracker.py.
"""
import importlib.util
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
FV = f"{ARR}/analysis/eval_final/flow_viz"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


T = _load("t_at", f"{ARR}/utils/test_attractor_tracker.py")
AT = _load("attractor_tracker",
           f"{ARR}/action-forcing/af_model/attractor_tracker.py")

RUN = os.environ.get("AT_RUN", "pilot_gt0")
PASSES = int(os.environ.get("AT_PASSES", "60"))
FPS = int(os.environ.get("AT_FPS", "6"))
OUT = os.environ.get("AT_OUT", f"{FV}/attractor_tracker_anim_{RUN}.mp4")
DIRS, DCOL, NFB, N_CH = T.DIRS, T.DCOL, T.NFB, T.N_CH


def live_estimate(tr):
    """Solve the CURRENT (un-frozen) fixed point per (bin, mu-dim)."""
    n = tr.ema_sn.clamp(min=1e-6)
    var_x = (tr.ema_sxx / n - (tr.ema_sx / n).pow(2)).clamp(min=1e-8)
    cov = tr.ema_sxy / n - (tr.ema_sx / n) * (tr.ema_sy / n)
    alpha = cov / var_x
    beta_i = (tr.ema_sy - alpha * tr.ema_sx) / n
    a = beta_i / (1.0 - alpha).clamp(min=1e-3)
    return a[:, :N_CH]


def main():
    student = T.load_student(RUN)
    teacher = T.load_teacher()
    m0, pcs = T.fit_basis()
    tr = AT.AttractorTracker(warmup=0, freeze_k=5, ema_beta=0.99,
                             min_eff_n=8.0)
    # static background: trajectories + teacher endpoints
    bg = []
    for d in DIRS:
        for chunks in student.get(d, []):
            bg.append((DCOL[d],
                       np.stack([T.proj_latent(c, m0, pcs) for c in chunks])))
    frames, trails = [], {d: [] for d in DIRS}
    for step in range(PASSES):
        for di, d in enumerate(DIRS):
            if d not in student or d not in teacher:
                continue
            for si, chunks in enumerate(student[d]):
                chain = torch.from_numpy(chunks.reshape(1, -1, T.C, T.H, T.W))
                tgt = torch.from_numpy(teacher[d][si % len(teacher[d])][None])
                n = min(chain.shape[1] // NFB, tgt.shape[1])
                tr.observe(chain[:, :n * NFB], tgt[:, :n], di)
        froze_before = int(tr.n_freezes[0])
        tr.sync(step)
        froze = int(tr.n_freezes[0]) > froze_before

        fig, ax = plt.subplots(figsize=(8, 8))
        for col, pts in bg:
            ax.plot(pts[:, 0], pts[:, 1], "-", color=col, lw=1.0, alpha=0.25)
            ax.plot(pts[-1, 0], pts[-1, 1], "o", color=col, ms=3, alpha=0.4)
        live = live_estimate(tr)
        for di, d in enumerate(DIRS):
            if d not in student:
                continue
            # hollow stars: per-direction LIVE estimates (the stratified votes)
            lp = T.mu_point(live[di].numpy(), m0, pcs)
            ax.plot(*lp, marker="*", ms=11, mfc="none", mec=DCOL[d],
                    mew=1.3, zorder=5)
        # THE single global frozen attractor (direction-invariant actuation ref)
        gg = tr.frozen_gate_glob[:N_CH]
        if float(gg.sum()) > 0:
            fp = T.mu_point(tr.frozen_a_glob[:N_CH].numpy(), m0, pcs)
            trails.setdefault("GLOB", []).append(fp)
            tl = np.stack(trails["GLOB"])
            ax.plot(tl[:, 0], tl[:, 1], "-", color="goldenrod", lw=1.2,
                    alpha=0.7, zorder=6)
            ax.plot(*fp, marker="*", color="gold", markeredgecolor="k",
                    mew=2.0, ms=16 + 16 * float(gg.mean()), zorder=7,
                    alpha=0.5 + 0.5 * float(gg.mean()))
        gs = float(gg.sum())
        sup = float((tr.frozen_gate[:, :N_CH] > 0).float().sum(0).mean())
        ax.set_title(
            f"{RUN} — AttractorTracker (GLOBAL), sync step {step + 1}/{PASSES}"
            f"{'   [FREEZE]' if froze else ''}\n"
            f"hollow stars = per-direction live estimates (votes); gold star = "
            f"frozen GLOBAL attractor\n(size/alpha = gate); "
            f"gate_sum={gs:.1f}  support={sup:.1f} bins")
        ax.set_xlim(-90, 260); ax.set_ylim(-120, 260)
        fig.tight_layout(); fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(OUT, frames, fps=FPS, quality=8)
    print(f"[anim] wrote {OUT} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
