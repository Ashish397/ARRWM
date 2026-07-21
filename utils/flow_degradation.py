"""AR degradation in the flow recordings: committed-chunk stats vs chunk index.

Per model, per committed chunk c (the last-rung pred_x0 of chunk c in the
student steps.npz recordings; row 48 / b{k} row 48 in the dense-teacher
trajs npz), averaged over dirs x seeds:

  M1  global mean / std of the committed latent
  M2  per-channel std profile distance to the GT chunk's profile
  M3  high-freq energy mean(|dW|+|dH|), as ratio to the GT chunk
  M4  temporal step RMS(x_c - x_{c-1})/sqrt(D)
  M6  ladder contraction std(last-rung x0)/std(rung-0 x0)   (students only)

GT anchor: the window's own zarr latents, chunk c = frames off+3(c+1)..+3.
Degradation signature = monotone exit of M1/M2/M3 from the GT band; "student
degrades sooner than teacher" = larger per-chunk drift rate than the dense
teacher's blocks over the shared range.

Writes flow_viz/flow_degradation.png + .csv. Env:
FD_STUDENTS colon list of flow_{name}/ recording dirs (def the two _vid),
FD_TEACHER (def 14d8 if present else 14d2), FD_OUT.
"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
FV = f"{ARR}/analysis/eval_final/flow_viz"
OUT = os.environ.get("FD_OUT", FV)
STUDENTS = os.environ.get("FD_STUDENTS",
                          "afall_freal_cd_vid:statwave_Freal_vid").split(":")
_tdef = ("14d8" if os.path.exists(f"{FV}/trajs_14d8_w8.npz") else "14d2")
if os.path.exists(f"{FV}/trajs_14e8_w8.npz"):
    _tdef += ":14e8"
TEACHERS = os.environ.get("FD_TEACHERS", os.environ.get("FD_TEACHER", _tdef)).split(":")
DIRS = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]
SHAPE = (3, 16, 60, 104)
MC = plt.get_cmap("tab10").colors


def stats(x):                                   # x [3,16,60,104] float32
    ch_std = x.std(axis=(0, 2, 3))
    grad = np.abs(np.diff(x, axis=3)).mean() + np.abs(np.diff(x, axis=2)).mean()
    return float(x.mean()), float(x.std()), ch_std, float(grad)


def gt_chunks(n=8):
    from utils.zarr_dataset import ZarrRideDataset
    w = json.load(open(f"{ARR}/analysis/eval_final/phaseA_windows.json"))[8]
    zp, off = w["zarr_path"], int(w["offset"])
    out = []
    for c in range(n):
        lat = ZarrRideDataset.load_latent_chunk(zp, off + 3 * (c + 1), off + 3 * (c + 2))
        out.append(stats(np.asarray(lat, dtype=np.float32)))
    return out


def student_series(name):
    """-> {c: dict(mean,std,chdist,grad,ladder)} averaged over dir/seed files."""
    acc = {}
    files = sorted(glob.glob(f"{FV}/flow_{name}/r08_*_s*/steps.npz"))
    for f in files:
        z = np.load(f)
        sdt = z["sdt"]
        by_chunk = {}
        for j, (c, r, t) in enumerate(sdt):
            if t > 0 and int(r) >= 0:                       # pred_x0 records only
                by_chunk.setdefault(int(c), []).append((int(r), j))
        prev = None
        for c in sorted(by_chunk):
            rs = sorted(by_chunk[c])
            first = z[f"x{rs[0][1]}"][0].astype(np.float32)
            last = z[f"x{rs[-1][1]}"][0].astype(np.float32)
            m, s, chs, g = stats(last)
            d = acc.setdefault(c, {"mean": [], "std": [], "chstd": [], "grad": [],
                                   "ladder": [], "step": []})
            d["mean"].append(m); d["std"].append(s); d["chstd"].append(chs)
            d["grad"].append(g)
            d["ladder"].append(float(last.std() / max(first.std(), 1e-6)))
            if prev is not None:
                d["step"].append(float(np.sqrt(((last - prev) ** 2).mean())))
            prev = last
    return acc


def teacher_series(run):
    z = np.load(f"{FV}/trajs_{run}_w8.npz")
    blocks = sorted({0} | {int(k.split("_")[0][1:]) for k in z.files if k.startswith("b")})
    acc = {}
    for b in blocks:
        pref = "" if b == 0 else f"b{b}_"
        for d in DIRS:
            for sd in range(4):
                k = f"{pref}{d}_{sd}"
                if k not in z.files:
                    continue
                x = z[k][-1].astype(np.float32).reshape(SHAPE)
                m, s, chs, g = stats(x)
                dd = acc.setdefault(b, {"mean": [], "std": [], "chstd": [], "grad": []})
                dd["mean"].append(m); dd["std"].append(s)
                dd["chstd"].append(chs); dd["grad"].append(g)
    return acc


def main():
    gt = gt_chunks()
    gtm = [g[0] for g in gt]; gts = [g[1] for g in gt]; gtg = [g[3] for g in gt]

    series = {s: student_series(s) for s in STUDENTS}
    tsers = {t: teacher_series(t) for t in TEACHERS}

    fig, axes = plt.subplots(1, 5, figsize=(24, 5.2))
    axm, axs, axg, axc, axl = axes
    cs = np.arange(8)
    for ax, gtv, ttl in ((axm, gtm, "M1 global mean"), (axs, gts, "M1 global std")):
        ax.fill_between(cs, np.min(gtv) - 0.005, np.max(gtv) + 0.005,
                        color="gray", alpha=0.18, label="GT band")
        ax.plot(cs, gtv, color="gray", lw=1.2, ls=":", label="GT chunk")
        ax.set_title(ttl)
    axg.plot(cs, np.ones(8), color="gray", lw=1.2, ls=":", label="GT (=1)")
    axg.set_title("M3 HF energy / GT chunk")
    axc.set_title("M2 per-channel std dist to GT")
    axl.set_title("M6 ladder contraction std(last)/std(rung0)")

    rows = []
    for i, (name, ser) in enumerate(series.items()):
        c_idx = sorted(ser)
        mean = [np.mean(ser[c]["mean"]) for c in c_idx]
        std = [np.mean(ser[c]["std"]) for c in c_idx]
        grad = [np.mean(ser[c]["grad"]) / gtg[c] for c in c_idx]
        chd = [np.mean([np.linalg.norm(v - gt[c][2]) for v in ser[c]["chstd"]])
               for c in c_idx]
        lad = [np.mean(ser[c]["ladder"]) for c in c_idx]
        col = MC[i % 10]
        axm.plot(c_idx, mean, "-o", ms=4, color=col, label=name)
        axs.plot(c_idx, std, "-o", ms=4, color=col, label=name)
        axg.plot(c_idx, grad, "-o", ms=4, color=col, label=name)
        axc.plot(c_idx, chd, "-o", ms=4, color=col, label=name)
        axl.plot(c_idx, lad, "-o", ms=4, color=col, label=name)
        for c in c_idx:
            rows.append((name, c, mean[c], std[c], grad[c], chd[c], lad[c]))
        print(f"[deg] {name}: mean {mean[0]:+.3f}->{mean[-1]:+.3f} | "
              f"std {std[0]:.3f}->{std[-1]:.3f} | hf/gt {grad[0]:.2f}->{grad[-1]:.2f} | "
              f"chdist {chd[0]:.3f}->{chd[-1]:.3f}", flush=True)

    tcolors = {0: ("black", "-s"), 1: ("#555555", "--D")}
    for ti, (tname, tser) in enumerate(tsers.items()):
        tb = sorted(tser)
        tmean = [np.mean(tser[b]["mean"]) for b in tb]
        tstd = [np.mean(tser[b]["std"]) for b in tb]
        tgrad = [np.mean(tser[b]["grad"]) / gtg[b] for b in tb]
        tchd = [np.mean([np.linalg.norm(v - gt[b][2]) for v in tser[b]["chstd"]]) for b in tb]
        col, fmt = tcolors.get(ti, ("#999999", ":o"))
        for ax, vals in ((axm, tmean), (axs, tstd), (axg, tgrad), (axc, tchd)):
            ax.plot(tb, vals, fmt, ms=6, color=col, label=f"teacher {tname} (48-step)")
        for b in tb:
            rows.append((f"teacher_{tname}", b, tmean[b], tstd[b], tgrad[b], tchd[b], np.nan))
        print(f"[deg] teacher {tname}: blocks {tb}, mean {tmean[0]:+.3f}->{tmean[-1]:+.3f}, "
              f"std {tstd[0]:.3f}->{tstd[-1]:.3f}, chdist {tchd[0]:.3f}->{tchd[-1]:.3f}", flush=True)

    for ax in axes:
        ax.set_xlabel("committed chunk index")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("AR degradation in flow space — committed-chunk latent statistics, window r08 "
                 "(students: 8-chunk own-context rollouts; teacher: dense 48-step; gray = GT)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(f"{OUT}/flow_degradation.png", dpi=130)
    with open(f"{OUT}/flow_degradation.csv", "w") as fh:
        fh.write("model,chunk,mean,std,hf_over_gt,chstd_dist,ladder\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")
    print(f"[deg] saved {OUT}/flow_degradation.png + .csv", flush=True)


if __name__ == "__main__":
    main()
