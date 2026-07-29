"""Quantify core-statistic drift over chained AR rollouts — window r08.

User observation (2026-07-27): pilot ODE students visibly shift colour and
contrast over the rollout. This measures it in BOTH domains:

  LATENT: per committed chunk, channel-agnostic mean & std of the [3,C,H,W]
    latent block. Teacher reference = trajs_14e8_w8.npz committed blocks
    (keys '{d}_{s}' block0, 'b{k}_{d}_{s}' blocks 1..7). Student = the final
    pred_x0 of each chunk in flow_{run}/r08_{d}_s{s}/steps.npz (sdt row with
    rung==max_rung, t>0 per chunk). Seed anchor = real zarr context stats.
  PIXEL: per video chunk (12 frames at 4x temporal upsample), RGB channel
    means + luma std from .motion_check/{run}/r08_{d}_s{sd}.mp4.

Reported per run: chunk trajectory of (lat_mean, lat_std, R, G, B, luma_std)
averaged over dirs/seeds, plus drift = last-chunk minus first-gen-chunk.
Env: SD_RUNS colon list of student runs (default pilot sets), SD_NSEEDS.
Output: stat_drift.csv + printed table.
"""
import glob, os, sys
import numpy as np

FV = "/scratch/u6ex/as1748.u6ex/ARRWM/analysis/eval_final/flow_viz"
RUNS = os.environ.get(
    "SD_RUNS",
    "pilot_gt0:pilot2_gt:pilot2_flip2:pilot2_dir4:pilot2_dir8:pilot2_mixed").split(":")
NSEEDS = int(os.environ.get("SD_NSEEDS", "2"))
DIRS = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]


def latent_chunks_student(run, d, sd):
    """Committed (final-rung pred_x0) latent per chunk from steps.npz."""
    p = f"{FV}/flow_{run}/r08_{d}_s{sd}/steps.npz"
    if not os.path.exists(p):
        return None
    z = np.load(p)
    sdt = z["sdt"]                       # rows: (chunk, rung, t)
    out = []
    for c in sorted(set(int(r[0]) for r in sdt)):
        rows = [i for i, r in enumerate(sdt)
                if int(r[0]) == c and r[2] > 0]
        if not rows:
            continue
        i = rows[-1]                     # last positive-t row = final pred_x0
        out.append(z[f"x{i}"].astype(np.float32))
    return out


def latent_chunks_teacher(d, sd):
    z = np.load(f"{FV}/trajs_14e8_w8.npz")
    out = []
    for b in range(8):
        k = f"{d}_{sd}" if b == 0 else f"b{b}_{d}_{sd}"
        if k in z.files:
            out.append(z[k].astype(np.float32))
    return out


def pixel_chunks(run, d, sd):
    p = f"{FV}/.motion_check/{run}/r08_{d}_s{sd}.mp4"
    if not os.path.exists(p):
        return None
    import imageio.v2 as imageio
    rd = imageio.get_reader(p)
    frames = np.stack([f for f in rd], 0).astype(np.float32) / 255.0
    rd.close()
    n = frames.shape[0] // 12
    out = []
    for c in range(n):
        blk = frames[c * 12:(c + 1) * 12]
        luma = blk @ np.array([0.299, 0.587, 0.114], np.float32)
        out.append((blk[..., 0].mean(), blk[..., 1].mean(),
                    blk[..., 2].mean(), luma.std()))
    return out


def agg(run, teacher=False):
    lat, pix = {}, {}
    for d in DIRS:
        for sd in range(NSEEDS):
            lc = latent_chunks_teacher(d, sd) if teacher else \
                latent_chunks_student(run, d, sd)
            if lc:
                for c, x in enumerate(lc):
                    lat.setdefault(c, []).append((x.mean(), x.std()))
            if not teacher:
                pc = pixel_chunks(run, d, sd)
                if pc:
                    for c, v in enumerate(pc):
                        pix.setdefault(c, []).append(v)
    return lat, pix


def main():
    rows = []
    print(f"{'run':14s} {'ch':>2s} {'latMU':>7s} {'latSD':>6s} "
          f"{'R':>6s} {'G':>6s} {'B':>6s} {'lumSD':>6s}")
    for run in ["TEACHER_14e8"] + RUNS:
        teacher = run == "TEACHER_14e8"
        lat, pix = agg(run, teacher=teacher)
        for c in sorted(lat):
            lm = np.mean([v[0] for v in lat[c]])
            ls = np.mean([v[1] for v in lat[c]])
            if c in pix:
                r, g, b, us = (np.mean([v[i] for v in pix[c]])
                               for i in range(4))
            else:
                r = g = b = us = float("nan")
            rows.append((run, c, lm, ls, r, g, b, us))
            print(f"{run:14s} {c:2d} {lm:7.4f} {ls:6.3f} "
                  f"{r:6.3f} {g:6.3f} {b:6.3f} {us:6.3f}")
        if len(lat) >= 2:
            cs = sorted(lat)
            d_sd = (np.mean([v[1] for v in lat[cs[-1]]]) /
                    max(np.mean([v[1] for v in lat[cs[0]]]), 1e-9))
            print(f"{run:14s}  -> latent std last/first ratio {d_sd:.3f}")
    import csv
    with open(f"{FV}/stat_drift.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "chunk", "lat_mean", "lat_std", "R", "G", "B", "luma_std"])
        w.writerows(rows)
    print(f"[sd] saved {FV}/stat_drift.csv")


if __name__ == "__main__":
    main()
