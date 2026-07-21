"""What does ODE distillation lose? Teacher-vs-student flow comparison, r08.

Inputs (all same Wan VAE latent space, same window r08, same pinned noise):
  trajs_pca8_w8.npz / trajs_14d_w8.npz       dense 48-step teacher ODE paths
                                             (8 dirs x 4 seeds, block 0)
  flow_{student}/r08_{d}_s{sd}/steps.npz     student 4-step re-noise sampler:
      sdt rows (chunk, rung, t): rung=-1 initial noise (t=1000);
      t>0 pred_x0 at that rung; t<0 re-noised input at |t|.

Because the student's chunk-0 noise is generator-identical to the teacher's
block-0 noise, ||student state - teacher state|| at matched t is a PATHWISE
distance along the same noise-conditioned trajectory. Panels:

  1) departure from the 14d teacher ODE path vs t: 14e (dense, reference for
     "a different good model") and each student's x_t at its 4 rungs
     (mean over 8 dirs x 4 seeds)
  2) inter-action separation vs t: dense curves for 14d/14e; students' points
     use pred_x0 separation at each rung (the state that matters for control)
  3) endpoint bars: ||final x0 - teacher final x_0|| and final inter-action
     separation, per model (fraction of teacher's)

Writes flow_viz/flow_distill_compare.png + prints the table.
Env: FDC_STUDENTS colon list (def odeF:odeF_teacherCD:odeF_noCD), FDC_OUT.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
FV = f"{ARR}/analysis/eval_final/flow_viz"
OUT = os.environ.get("FDC_OUT", FV)
STUDENTS = os.environ.get("FDC_STUDENTS", "odeF:odeF_teacherCD:odeF_noCD").split(":")
DIRS = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]
NSEEDS = 4
IU = np.triu_indices(len(DIRS), 1)
STEPS = 48; SHIFT = 5.0
COLORS = {"14d": "#1f77b4", "pca8": "#7f7f7f", "odeF": "#d62728",
          "odeF_teacherCD": "#2ca02c", "odeF_noCD": "#9467bd"}


def teacher_ts():
    sig = np.linspace(1.0, 0.0, STEPS + 1)[:-1]
    sig = SHIFT * sig / (1 + (SHIFT - 1) * sig)
    return np.concatenate([[1.0], sig[1:], [0.0]]) * 1000.0


class TeacherLazy:
    """Lazy per-key access to a dense teacher recording (4GB cgroup)."""

    def __init__(self, run):
        self.z = np.load(f"{FV}/trajs_{run}_w8.npz")

    def get(self, d, sd):
        return self.z[f"{d}_{sd}"]                  # [49, D] float16, fresh copy


def load_student(run):
    """{(d, sd): {'t_x': [(t, x)], 't_x0': [(t, x0)], 'final': x0}} chunk 0."""
    out = {}
    for d in DIRS:
        for sd in range(NSEEDS):
            f = f"{FV}/flow_{run}/r08_{d}_s{sd}/steps.npz"
            if not os.path.exists(f):
                print(f"[fdc] MISSING {f}")
                return None
            z = np.load(f)
            sdt = z["sdt"]
            t_x, t_x0 = [], []
            for j, (chunk, rung, t) in enumerate(sdt):
                if int(chunk) != 0:
                    continue
                x = z[f"x{j}"].reshape(-1)
                if int(rung) == -1:
                    t_x.append((1000.0, x))            # initial noise
                elif t > 0:
                    t_x0.append((float(t), x))         # pred_x0 at rung t
                else:
                    t_x.append((-float(t), x))         # re-noised input at |t|
            out[(d, sd)] = {"t_x": t_x, "t_x0": t_x0, "final": t_x0[-1][1]}
    return out


def main():
    ts48 = teacher_ts()
    t14d = TeacherLazy("14d")
    t14e = TeacherLazy("pca8")
    students = {s: load_student(s) for s in STUDENTS}
    students = {s: v for s, v in students.items() if v is not None}

    def sep_at(states):                                # [8, D] -> mean pair dist
        x = np.stack(states).astype(np.float32)
        dd = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
        return float(dd[IU].mean())

    # dense teacher curves — one seed's 8 trajectories in memory at a time
    def dense_sep(tr):
        out = np.zeros(len(ts48))
        for sd in range(NSEEDS):
            tr8 = [tr.get(d, sd) for d in DIRS]
            for i in range(len(ts48)):
                out[i] += sep_at([t[i] for t in tr8]) / NSEEDS
            del tr8
        return out
    sep14d, sep14e = dense_sep(t14d), dense_sep(t14e)

    # 14e departure from 14d (pathwise, matched noise), dense
    dep14e = np.zeros(len(ts48))
    for d in DIRS:
        for sd in range(NSEEDS):
            a = t14d.get(d, sd).astype(np.float32)
            b = t14e.get(d, sd).astype(np.float32)
            dep14e += np.linalg.norm(a - b, axis=1) / (len(DIRS) * NSEEDS)
            del a, b

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 6.2),
                                        gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for ax in (ax1, ax2):
        ax.set_xlim(1010, -10)
        ax.set_xlabel("diffusion timestep t (1000 = pure noise)")
        ax.grid(alpha=0.3)

    ax1.axhline(dep14e[-1], lw=1.6, color=COLORS["pca8"], ls="--",
                label=f"14e final vs 14d final ({dep14e[-1]:.0f})")
    ax2.plot(ts48, sep14d, lw=2.2, color=COLORS["14d"], label="14d teacher (dense ODE)")
    ax2.plot(ts48, sep14e, lw=1.6, color=COLORS["pca8"], ls="--", label="14e (dense ODE)")

    table = {}
    for s, data in students.items():
        # panel 1: ||pred_x0(rung) - teacher final x0|| — noise-free
        # convergence of the student's estimate to the teacher ODE endpoint
        # (raw x_t comparisons at mid rungs are re-noise-variance dominated).
        rung_ts = sorted({t for (t, _) in data[(DIRS[0], 0)]["t_x0"]}, reverse=True)
        d_acc = np.zeros(len(rung_ts))
        for d in DIRS:
            for sd in range(NSEEDS):
                tfin = t14d.get(d, sd)[-1].astype(np.float32)
                for ri, t in enumerate(rung_ts):
                    x0 = [v for (tt, v) in data[(d, sd)]["t_x0"] if tt == t][0]
                    d_acc[ri] += float(np.linalg.norm(x0.astype(np.float32) - tfin))
                del tfin
        d_pts = d_acc / (len(DIRS) * NSEEDS)
        accf = float(d_pts[-1])                     # final pred_x0 IS the output
        ax1.plot(rung_ts, d_pts, "-o", lw=1.8, ms=6,
                 color=COLORS.get(s, "k"), label=f"{s} pred_x0 (4-step)")

        # panel 2: pred_x0 inter-action separation at each rung
        x0_ts = sorted({t for (t, _) in data[(DIRS[0], 0)]["t_x0"]}, reverse=True)
        s_ts, s_sep = [], []
        for t in x0_ts:
            acc = 0.0
            for sd in range(NSEEDS):
                acc += sep_at([[v for (tt, v) in data[(d, sd)]["t_x0"] if tt == t][0]
                               for d in DIRS]) / NSEEDS
            s_ts.append(t); s_sep.append(acc)
        ax2.plot(s_ts, s_sep, "-o", lw=1.8, ms=6, color=COLORS.get(s, "k"),
                 label=f"{s} pred_x0 sep")
        table[s] = (float(accf), s_sep[-1])

    ax1.set_ylabel("|| pred_x0(rung) - 14d ODE final x0 ||  (matched noise)")
    ax1.set_title("Convergence of the student estimate to the teacher ODE endpoint")
    ax1.legend(fontsize=9)
    ax2.set_ylabel("inter-action latent distance")
    ax2.set_title("Action separation: dense teachers vs student pred_x0 rungs")
    ax2.legend(fontsize=9)

    names = ["14d", "pca8"] + list(table)
    endsep = [sep14d[-1], sep14e[-1]] + [v[1] for v in table.values()]
    ax3.bar(range(len(names)), endsep,
            color=[COLORS.get(n, "k") for n in names])
    ax3.axhline(sep14d[-1], color="#1f77b4", lw=1, ls=":")
    ax3.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=9)
    ax3.set_ylabel("final inter-action separation")
    ax3.set_title("Endpoint action separation\n(distill loss of grounding)")
    ax3.grid(alpha=0.3, axis="y")

    fig.suptitle("ODE distillation flow diagnosis — window r08, 8 dirs x 4 seeds, "
                 "identical initial noise across all models", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(f"{OUT}/flow_distill_compare.png", dpi=130)
    print(f"[fdc] saved {OUT}/flow_distill_compare.png", flush=True)

    print(f"[fdc] teacher 14d: end action-sep {sep14d[-1]:.1f} | 14e: {sep14e[-1]:.1f} "
          f"| pathwise 14e-vs-14d end {dep14e[-1]:.1f}")
    for s, (dx0, ssep) in table.items():
        print(f"[fdc] {s}: final-x0 dist to teacher {dx0:.1f} "
              f"({dx0 / max(dep14e[-1], 1e-6):.2f}x the 14e-14d gap) | "
              f"end action-sep {ssep:.1f} ({ssep / sep14d[-1] * 100:.0f}% of teacher)")


if __name__ == "__main__":
    main()
