"""Which statistic contracts under few-step sampling: MEAN, VARIANCE, or both?

Per-chunk ledger of (a) per-channel-mean vector norm ||mu|| (16-dim, the
'mean' — its norm shrinking = pull toward zero/prior; its direction moving
= hue drift) and (b) overall centred std (the 'variance'), for:

  GT           real encoded latents (window r08)
  teacher48    dense 48-step committed blocks       <- '40 steps'
  teacher20    dense 20-step committed blocks
  path@rungs   teacher DENSE path states read at the 4 rung times
               ('doing 40 steps but looking at 4 of them')
  gt0-4rung    teacher WEIGHTS run as 4 one-jump predictions (student
               sampler) — committed chunks AND per-rung pred_x0

If mean-norm and std both track step count -> both moments contract in the
one-jump predictions; path@rungs should show NO extra contraction beyond
the dense end state (it IS the dense path).
CPU-only, streams npz. Output printed + stat_mean_ledger.csv.
"""
import json, os
import numpy as np

FV = "/scratch/u6ex/as1748.u6ex/ARRWM/analysis/eval_final/flow_viz"
ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
DIRS = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]
NFB, C = 3, 16
RUNG_T = [1000.0, 625.0, 357.142857, 208.333333]


def mom(x):
    """x [*, C, H, W] flattened frames -> (||mu||, centred std)."""
    x = x.reshape(-1, C, x.shape[-2] * x.shape[-1]).astype(np.float64)
    mu = x.mean(axis=(0, 2))                     # [16]
    sd = (x - mu[None, :, None]).std()
    return float(np.linalg.norm(mu)), float(sd)


def sched_t(nsteps):
    return [1000.0 * 5 * u / (1 + 4 * u) for u in (1 - i / nsteps for i in range(nsteps))]


def main():
    import sys
    sys.path.insert(0, ARR)
    from utils.zarr_dataset import ZarrRideDataset
    rows = []

    w = json.load(open(f"{ARR}/analysis/eval_final/phaseA_windows.json"))[8]
    gt = np.asarray(ZarrRideDataset.load_latent_chunk(
        w["zarr_path"], int(w["offset"]), int(w["offset"]) + NFB * 9), dtype=np.float32)
    print(f"{'series':14s} {'chunk':>5s} {'||mu||':>7s} {'std':>6s}")
    for c in range(1, 8):
        m, s = mom(gt[NFB * c:NFB * (c + 1)])
        rows.append(("GT", c, m, s))
    m0 = np.mean([r[2] for r in rows]); s0 = np.mean([r[3] for r in rows])
    print(f"{'GT':14s}   avg {m0:7.3f} {s0:6.3f}")

    for tag, nst in [("14e8", 48), ("14e8s20", 20)]:
        z = np.load(f"{FV}/trajs_{tag}_w8.npz")
        per = {}
        # committed blocks
        for d in DIRS:
            for sd_i in range(4):
                for b in range(8):
                    k = f"{d}_{sd_i}" if b == 0 else f"b{b}_{d}_{sd_i}"
                    if k in z.files:
                        arr = z[k][-1].astype(np.float32).reshape(NFB, C, 60, 104)
                        per.setdefault(b, []).append(mom(arr))
        for b in sorted(per):
            ms = np.mean([v[0] for v in per[b]]); ss = np.mean([v[1] for v in per[b]])
            rows.append((f"teacher{nst}", b, ms, ss))
        line = " ".join(f"{np.mean([v[0] for v in per[b]]):.3f}/{np.mean([v[1] for v in per[b]]):.3f}" for b in sorted(per))
        print(f"teacher{nst:<2d} committed (||mu||/std by block): {line}")
        # dense-path states read at the rung times, block 0 only
        ts = sched_t(nst)
        idx = [int(np.argmin([abs(t - rt) for t in ts])) for rt in RUNG_T]
        pr = {r: [] for r in range(4)}
        for d in DIRS:
            for sd_i in range(4):
                k = f"{d}_{sd_i}"
                if k in z.files:
                    arr = z[k].astype(np.float32)
                    for r, i in enumerate(idx):
                        pr[r].append(mom(arr[i].reshape(NFB, C, 60, 104)))
        line = " ".join(f"t{int(RUNG_T[r])}:{np.mean([v[0] for v in pr[r]]):.3f}/{np.mean([v[1] for v in pr[r]]):.3f}" for r in range(4))
        print(f"teacher{nst:<2d} PATH@rungs blk0 (mixed x_t states): {line}")

    # student sampler on teacher weights: committed + per-rung pred_x0
    perc, perr = {}, {r: [] for r in range(4)}
    for d in DIRS:
        for sd_i in range(2):
            p = f"{FV}/flow_pilot_gt0/r08_{d}_s{sd_i}/steps.npz"
            if not os.path.exists(p):
                continue
            z = np.load(p)
            sdt = z["sdt"]
            for i, (c, r, t) in enumerate(sdt):
                if t > 0 and int(r) >= 0:
                    x = z[f"x{i}"].astype(np.float32).reshape(NFB, C, 60, 104)
                    perr[int(r)].append(mom(x))
                    if int(r) == 3:
                        perc.setdefault(int(c), []).append(mom(x))
    line = " ".join(f"r{r}:{np.mean([v[0] for v in perr[r]]):.3f}/{np.mean([v[1] for v in perr[r]]):.3f}" for r in range(4))
    print(f"gt0-4rung pred_x0 by RUNG (all chunks):  {line}")
    for c in sorted(perc):
        ms = np.mean([v[0] for v in perc[c]]); ss = np.mean([v[1] for v in perc[c]])
        rows.append(("gt0-4rung", c, ms, ss))
    line = " ".join(f"{np.mean([v[0] for v in perc[c]]):.3f}/{np.mean([v[1] for v in perc[c]]):.3f}" for c in sorted(perc))
    print(f"gt0-4rung committed by chunk:            {line}")

    with open(f"{FV}/stat_mean_ledger.csv", "w") as f:
        f.write("series,chunk,mu_norm,std\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")
    print(f"[ml] saved {FV}/stat_mean_ledger.csv")


if __name__ == "__main__":
    main()
