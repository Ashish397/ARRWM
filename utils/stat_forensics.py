"""Root-cause forensics for AR core-statistic degradation (window r08).

User hypothesis (2026-07-28): autoregressive generation contracts core
stats (std/mean/min/max) and compounds through self-context; locking the
stats helps. This tool decides WHY, from existing recordings only:

  T1 DOSE-RESPONSE (same weights, step count varies): 14e8 (48-step dense),
     14e8s20 (20-step), pilot_gt0 (4-rung ladder, exact teacher init).
     If drift grows as steps shrink -> few-step conditional-mean bias
     (law of total variance), not training.
  T2 PER-RUNG LEDGER: where in the 4-rung ladder variance dies (rung-0
     mean collapse? no recovery? renoise leak?) + cross-seed diversity.
  T3 SIGMA AUDIT: regress each recorded renoised state on its pred_x0:
     slope ~= (1-sigma), residual std ~= sigma. Compare vs nominal
     sigma = |t|/1000 of the shift-5 grid. Mismatch = bookkeeping bug.
  T4 COMPOUNDING LAW: per-chunk committed std s_c: chunk-0 deficit (clean
     GT context => pure sampler effect) vs per-chunk decay factor
     (feedback effect). Geometric fit.
  T5 AFFINE SUFFICIENCY: per-channel affine-correct student committed
     chunks to the paired teacher block (same pinned noise/dir/seed);
     fraction of per-channel quantile (W2) gap explained by affine alone.
     High fraction => stat-locking is the *correct* correction class.

CPU-only; streams npz keys (4GB login cgroup safe). Env: SF_RUNS colon
list of student runs (default pilot_gt0:pilot2_flip2), SF_NSEEDS (2).
Outputs: printed report + stat_forensics.csv + stat_forensics.png.
"""
import json, os
import numpy as np

FV = "/scratch/u6ex/as1748.u6ex/ARRWM/analysis/eval_final/flow_viz"
ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
RUNS = os.environ.get("SF_RUNS", "pilot_gt0:pilot2_flip2").split(":")
NSEEDS = int(os.environ.get("SF_NSEEDS", "2"))
DIRS = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]
NFB, C = 3, 16
RUNGS_T = [1000.0, 625.0, 357.142857, 208.333333]


def stats(a):
    a = a.astype(np.float64)
    return dict(mean=a.mean(), std=a.std(), amin=a.min(), amax=a.max())


def load_student(run, d, sd):
    """-> dict: committed[c], pred[c][r], ren[c][r], noise[c] (float32)."""
    p = f"{FV}/flow_{run}/r08_{d}_s{sd}/steps.npz"
    if not os.path.exists(p):
        return None
    z = np.load(p)
    sdt = z["sdt"]
    out = {"pred": {}, "ren": {}, "noise": {}, "committed": {}}
    for i, (c, r, t) in enumerate(sdt):
        c, r, t = int(c), int(r), float(t)
        x = z[f"x{i}"].astype(np.float32).ravel()
        if r == -1:
            out["noise"][c] = x
        elif t > 0:
            out["pred"].setdefault(c, {})[r] = x
        else:
            out["ren"].setdefault(c, {})[r] = (abs(t), x)
    for c, rp in out["pred"].items():
        out["committed"][c] = rp[max(rp)]
    return out


def teacher_blocks(tag, d, sd, want_rows=False):
    """Stream teacher npz: yield (block, final_row or full)."""
    z = np.load(f"{FV}/trajs_{tag}_w8.npz")
    for b in range(8):
        k = f"{d}_{sd}" if b == 0 else f"b{b}_{d}_{sd}"
        if k not in z.files:
            continue
        arr = z[k]
        yield b, (arr.astype(np.float32) if want_rows
                  else arr[-1].astype(np.float32))


def gt_latents():
    import sys
    sys.path.insert(0, ARR)
    from utils.zarr_dataset import ZarrRideDataset
    w = json.load(open(f"{ARR}/analysis/eval_final/phaseA_windows.json"))[8]
    lat = ZarrRideDataset.load_latent_chunk(
        w["zarr_path"], int(w["offset"]), int(w["offset"]) + NFB * 9)
    return np.asarray(lat, dtype=np.float32)      # [27, C, H, W]


def main():
    rows = []
    rep = []

    def log(s):
        rep.append(s); print(s, flush=True)

    gt = gt_latents()
    gt_chunks = [gt[NFB * i:NFB * (i + 1)] for i in range(9)]
    gt_std = np.mean([c.std() for c in gt_chunks[1:9]])
    log(f"[T0] GT anchor: per-chunk std {gt_std:.4f}, "
        f"mean {np.mean([c.mean() for c in gt_chunks]):.4f}")

    # ---- T1 dose-response ----
    log("\n[T1] DOSE-RESPONSE (same weights; committed-block std by chunk)")
    t1 = {}
    for tag, nst in [("14e8", 48), ("14e8s20", 20)]:
        per_b = {b: [] for b in range(8)}
        for d in DIRS:
            for sd in range(4):
                for b, x in teacher_blocks(tag, d, sd):
                    per_b[b].append(x.std())
        s = [float(np.mean(per_b[b])) for b in range(8) if per_b[b]]
        t1[tag] = s
        log(f"  {tag} ({nst} steps): " + " ".join(f"{v:.3f}" for v in s)
            + f"  | last/first {s[-1]/s[0]:.3f} | c0/GT {s[0]/gt_std:.3f}")
    for run in RUNS:
        per_c = {}
        for d in DIRS:
            for sd in range(NSEEDS):
                st = load_student(run, d, sd)
                if st is None:
                    continue
                for c, x in st["committed"].items():
                    per_c.setdefault(c, []).append(x.std())
        if not per_c:
            continue
        cs = sorted(per_c)
        s = [float(np.mean(per_c[c])) for c in cs]
        t1[run] = s
        log(f"  {run} (4 rungs): " + " ".join(f"{v:.3f}" for v in s)
            + f"  | last/first {s[-1]/s[0]:.3f} | c0/GT {s[0]/gt_std:.3f}")

    # ---- T2 per-rung ledger + diversity ----
    log("\n[T2] PER-RUNG LEDGER (avg over chunks/dirs/seeds)")
    for run in RUNS:
        acc = {("pred", r): [] for r in range(4)}
        acc.update({("ren", r): [] for r in range(0, 3)})
        div = {r: [] for r in range(4)}
        data = {}
        for d in DIRS:
            for sd in range(NSEEDS):
                st = load_student(run, d, sd)
                if st:
                    data[(d, sd)] = st
        for (d, sd), st in data.items():
            for c, rp in st["pred"].items():
                for r, x in rp.items():
                    acc[("pred", r)].append(x.std())
            for c, rr in st["ren"].items():
                for r, (t, x) in rr.items():
                    acc[("ren", r)].append(x.std())
        # cross-seed diversity of pred_x0 at each rung (same dir, chunk)
        for d in DIRS:
            if (d, 0) in data and (d, 1) in data:
                for c in data[(d, 0)]["pred"]:
                    for r in data[(d, 0)]["pred"][c]:
                        a = data[(d, 0)]["pred"][c][r]
                        b = data[(d, 1)]["pred"][c].get(r)
                        if b is not None:
                            div[r].append(np.linalg.norm(a - b) /
                                          (np.linalg.norm(a) + 1e-9))
        log(f"  {run}:")
        for r in range(4):
            p = np.mean(acc[("pred", r)]) if acc[("pred", r)] else float("nan")
            n = np.mean(acc[("ren", r)]) if r < 3 and acc[("ren", r)] else float("nan")
            dv = np.mean(div[r]) if div[r] else float("nan")
            log(f"    rung {r} (t={RUNGS_T[r]:7.1f}): std(pred_x0)={p:.4f}"
                + (f"  std(renoise->t{RUNGS_T[r+1]:.0f})={n:.4f}" if r < 3 else " " * 26)
                + f"  xseed-div={dv:.3f}")
            rows.append((run, "rung", r, p, n if r < 3 else np.nan, dv))

    # ---- T3 sigma audit ----
    log("\n[T3] SIGMA AUDIT (renoised = (1-sigma)*pred + sigma*eps ?)")
    for run in RUNS[:1]:   # gt0 suffices: serve path identical across runs
        sl_by_t, rs_by_t = {}, {}
        for d in DIRS:
            for sd in range(NSEEDS):
                st = load_student(run, d, sd)
                if not st:
                    continue
                for c in st["ren"]:
                    for r, (t_abs, xr) in st["ren"][c].items():
                        xp = st["pred"][c][r]      # renoise row shares its rung's pred
                        vp = xp.var()
                        slope = float(np.dot(xr - xr.mean(), xp - xp.mean())
                                      / (len(xp) * vp))
                        resid = float(np.std(xr - slope * xp))
                        sl_by_t.setdefault(round(t_abs, 1), []).append(slope)
                        rs_by_t.setdefault(round(t_abs, 1), []).append(resid)
        for t in sorted(sl_by_t, reverse=True):
            sig = t / 1000.0
            sl, rs = np.mean(sl_by_t[t]), np.mean(rs_by_t[t])
            log(f"  t={t:7.1f}: slope {sl:.4f} (expect 1-sigma={1-sig:.4f})"
                f"   resid-std {rs:.4f} (expect sigma*1={sig:.4f})")
            rows.append((run, "sigma", t, sl, 1 - sig, rs))

    # ---- T4 compounding ----
    log("\n[T4] COMPOUNDING LAW (log-linear fit of committed std)")
    for run, s in t1.items():
        s = np.array(s, dtype=np.float64)
        c = np.arange(len(s))
        r_fit = float(np.exp(np.polyfit(c, np.log(s), 1)[0]))
        log(f"  {run}: c0/GT deficit {s[0]/gt_std:.3f}, per-chunk factor "
            f"{r_fit:.4f} -> {'feedback-compounding' if r_fit < 0.99 else 'flat'}"
            f" (pure-sampler-once would predict factor ~1.0)")
        rows.append((run, "compound", -1, s[0] / gt_std, r_fit, np.nan))

    # ---- T5 affine sufficiency (vs paired teacher 20-step blocks) ----
    log("\n[T5] AFFINE SUFFICIENCY (per-channel W2 gap explained, vs 14e8s20)")
    for run in RUNS:
        expl = []
        for d in DIRS:
            for sd in range(NSEEDS):
                st = load_student(run, d, sd)
                if not st:
                    continue
                tb = {b: x for b, x in teacher_blocks("14e8s20", d, sd)}
                for c, x in st["committed"].items():
                    if c not in tb:
                        continue
                    xs = x.reshape(NFB, C, -1)
                    xt = tb[c].reshape(NFB, C, -1)
                    for ch in range(C):
                        a = np.sort(xs[:, ch].ravel())
                        b = np.sort(xt[:, ch].ravel())
                        w2_before = np.mean((a - b) ** 2)
                        a2 = (a - a.mean()) / (a.std() + 1e-9) * b.std() + b.mean()
                        w2_after = np.mean((np.sort(a2) - b) ** 2)
                        if w2_before > 1e-12:
                            expl.append(1.0 - w2_after / w2_before)
        if expl:
            log(f"  {run}: affine explains {100*np.mean(expl):.1f}% of the "
                f"per-channel quantile gap (median {100*np.median(expl):.1f}%)")
            rows.append((run, "affine", -1, np.mean(expl), np.median(expl), np.nan))

    with open(f"{FV}/stat_forensics.csv", "w") as f:
        f.write("run,test,key,v1,v2,v3\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")
    with open(f"{FV}/STAT_FORENSICS_REPORT.txt", "w") as f:
        f.write("\n".join(rep))
    print(f"\n[sf] saved {FV}/stat_forensics.csv + STAT_FORENSICS_REPORT.txt")


if __name__ == "__main__":
    main()
