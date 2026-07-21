"""WHY do the well-trained DMD students still diverge? Window r08 diagnosis.

Students (their real 4-step re-noise sampler, chunks 0+1, pinned noise):
  afall_freal_cd  -> teacher 14d2 (v14d_critic8_natural@2150, dense blk 0+1)
  statwave_Freal  -> teacher 14   (v14_balanced_weunz@6600,  dense blk 0+1)

Angles:
  1) rung convergence per chunk: ||pred_x0(rung) - teacher ODE final|| for
     chunk 0 (real seed context) vs chunk 1 (own generated context). Chunk-1
     teacher reference conditions on the TEACHER's committed block 0, so the
     c1-c0 gap bundles per-step error + context mismatch = compounding.
  2) action separation at each rung vs the teacher's dense curve, per chunk.
  3) per-direction final-x0 distance to teacher (is divergence isotropic or
     concentrated in specific actions, e.g. backward?).
  4) endpoint assignment: is student dir d nearest teacher dir d (correct
     action mode, just offset) or another dir (wrong mode)?

Writes flow_viz/flow_diverge_dmd3.png + prints tables. Env: FDD_OUT.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
FV = f"{ARR}/analysis/eval_final/flow_viz"
OUT = os.environ.get("FDD_OUT", FV)
DIRS = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]
NSEEDS = 4
IU = np.triu_indices(len(DIRS), 1)
STEPS = 48; SHIFT = 5.0
PAIRS = dict(kv.split("=") for kv in os.environ.get(
    "FDD_PAIRS", "afall_freal_cd=14d2:statwave_Freal=14").split(":"))
FIGNAME = os.environ.get("FDD_FIGNAME", "flow_diverge_dmd3")
_SCPOOL = ["#d62728", "#9467bd", "#ff7f0e", "#8c564b"]
SC = {s: _SCPOOL[i % 4] for i, s in enumerate(PAIRS)}
TC = {"14d2": "#1f77b4", "14": "#2ca02c"}


def teacher_ts():
    sig = np.linspace(1.0, 0.0, STEPS + 1)[:-1]
    sig = SHIFT * sig / (1 + (SHIFT - 1) * sig)
    return np.concatenate([[1.0], sig[1:], [0.0]]) * 1000.0


class TeacherLazy:
    def __init__(self, run):
        self.z = np.load(f"{FV}/trajs_{run}_w8.npz")

    def get(self, d, sd, blk=0):
        return self.z[f"{d}_{sd}" if blk == 0 else f"b{blk}_{d}_{sd}"]


def load_student(run):
    """{(d, sd): {chunk: {'x0': [(t, vec)], 'final': vec}}}"""
    out = {}
    for d in DIRS:
        for sd in range(NSEEDS):
            f = f"{FV}/flow_{run}/r08_{d}_s{sd}/steps.npz"
            if not os.path.exists(f):
                print(f"[fdd] MISSING {f}")
                return None
            z = np.load(f)
            per = {0: [], 1: []}
            for j, (chunk, rung, t) in enumerate(z["sdt"]):
                # rung -1 is the initial NOISE record (also at t=+1000) —
                # exclude it or it shadows the rung-0 pred_x0
                if t > 0 and int(rung) >= 0 and int(chunk) in per:
                    per[int(chunk)].append((float(t), z[f"x{j}"].reshape(-1)))
            out[(d, sd)] = {c: {"x0": v, "final": v[-1][1]} for c, v in per.items()}
    return out


def sep_at(vecs):
    x = np.stack(vecs).astype(np.float32)
    dd = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
    return float(dd[IU].mean())


def main():
    ts48 = teacher_ts()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19.5, 6.2))
    for ax in (ax1, ax2):
        ax.set_xlim(1010, -10)
        ax.set_xlabel("diffusion timestep t")
        ax.grid(alpha=0.3)

    barsW, barsL, barsC = [], [], []
    for s, trun in PAIRS.items():
        data = load_student(s)
        if data is None:
            continue
        T = TeacherLazy(trun)
        rung_ts = sorted({t for (t, _) in data[(DIRS[0], 0)][0]["x0"]}, reverse=True)

        # dense teacher separation, blocks 0 and 1 (sep accrues over world
        # time — block-1 contexts already differ between actions)
        tseps = {}
        for blk, ls in ((0, "-"), (1, ":")):
            tsep = np.zeros(len(ts48))
            for sd in range(NSEEDS):
                tr8 = [T.get(d, sd, blk=blk) for d in DIRS]
                for i in range(len(ts48)):
                    tsep[i] += sep_at([t[i] for t in tr8]) / NSEEDS
                del tr8
            tseps[blk] = tsep
            ax2.plot(ts48, tsep, ls, lw=1.6, color=TC[trun],
                     label=f"{trun} teacher blk{blk} (dense)")
        tsep = tseps[0]

        for chunk, ls in ((0, "-"), (1, "--")):
            conv = np.zeros(len(rung_ts))
            ssep = np.zeros(len(rung_ts))
            perdir = np.zeros(len(DIRS))
            for di, d in enumerate(DIRS):
                for sd in range(NSEEDS):
                    tfin = T.get(d, sd, blk=chunk)[-1].astype(np.float32)
                    x0s = data[(d, sd)][chunk]["x0"]
                    for ri, t in enumerate(rung_ts):
                        x0 = [v for (tt, v) in x0s if tt == t][0].astype(np.float32)
                        conv[ri] += np.linalg.norm(x0 - tfin)
                    perdir[di] += np.linalg.norm(
                        data[(d, sd)][chunk]["final"].astype(np.float32) - tfin) / NSEEDS
                    del tfin
            conv /= len(DIRS) * NSEEDS
            for ri, t in enumerate(rung_ts):
                for sd in range(NSEEDS):
                    ssep[ri] += sep_at([[v for (tt, v) in data[(d, sd)][chunk]["x0"]
                                         if tt == t][0] for d in DIRS]) / NSEEDS
            ax1.plot(rung_ts, conv, ls, marker="o", ms=5, lw=1.8, color=SC[s],
                     label=f"{s} chunk{chunk}")
            ax2.plot(rung_ts, ssep, ls, marker="o", ms=5, lw=1.8, color=SC[s],
                     label=f"{s} c{chunk} pred_x0 sep")
            barsW.append(f"{s[:6]}·c{chunk}"); barsL.append(perdir.copy()); barsC.append(SC[s])
            print(f"[fdd] {s} chunk{chunk}: conv {conv[0]:.0f}->{conv[-1]:.0f} | "
                  f"sep {ssep[0]:.0f}->{ssep[-1]:.0f} "
                  f"(teacher blk{chunk} end {tseps[chunk][-1]:.0f})", flush=True)

        # endpoint assignment (chunk 0): student dir d vs nearest teacher dir
        hits = 0
        conf = np.zeros((8, 8), int)
        for sd in range(NSEEDS):
            tf = {d2: T.get(d2, sd)[-1].astype(np.float32) for d2 in DIRS}
            for di, d in enumerate(DIRS):
                sf = data[(d, sd)][0]["final"].astype(np.float32)
                dists = [np.linalg.norm(sf - tf[d2]) for d2 in DIRS]
                j = int(np.argmin(dists))
                conf[di, j] += 1
                hits += int(j == di)
            del tf
        print(f"[fdd] {s}: endpoint->teacher-action assignment "
              f"{hits}/{8 * NSEEDS} correct ({hits / (8 * NSEEDS) * 100:.0f}%)", flush=True)
        print("      rows=student dir, cols=nearest teacher dir "
              + " ".join(f"{d:>3s}" for d in DIRS))
        for di, d in enumerate(DIRS):
            print(f"      {d:>3s} " + " ".join(f"{conf[di, j]:3d}" for j in range(8)))

    ax1.set_ylabel("|| pred_x0(rung) - teacher ODE final ||")
    ax1.set_title("Convergence to the teacher endpoint: real seed (c0) vs own context (c1)")
    ax1.legend(fontsize=9)
    ax2.set_ylabel("inter-action latent distance")
    ax2.set_title("Action separation: teacher dense vs student rungs")
    ax2.legend(fontsize=8)

    x = np.arange(len(DIRS))
    for i, (w, l, c) in enumerate(zip(barsW, barsL, barsC)):
        ax3.bar(x + (i - len(barsW) / 2) * 0.2 + 0.1, l, width=0.2,
                color=c, alpha=1.0 - 0.45 * (i % 2), label=w)
    ax3.set_xticks(x, DIRS)
    ax3.set_ylabel("final-x0 distance to teacher")
    ax3.set_title("Per-direction divergence (dashed pairs = chunk 1)")
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3, axis="y")

    fig.suptitle("Why the trained DMD students still diverge — window r08, 8 dirs x 4 seeds, "
                 "pinned noise; c1 teacher ref conditions on teacher context (gap = compounding)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(f"{OUT}/{FIGNAME}.png", dpi=130)
    print(f"[fdd] saved {OUT}/{FIGNAME}.png", flush=True)


if __name__ == "__main__":
    main()
