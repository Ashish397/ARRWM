"""Student-vs-TEACHER video match score (the campaign's target metric).

For every commanded direction, compare a student rollout video against the
dense 48-step teacher video of the SAME window/seed/direction and report

    mse   = mean (I_s - I_t)^2           on [0,1] RGB
    match = 1 - mse / var(I_t)           (explained-variance / R^2 form)

`match` is the number to drive to 0.95. It is scale-aware: predicting the
teacher's mean frame everywhere scores 0, and a perfect match scores 1, so
"0.95" means 95% of the teacher video's pixel variance is reproduced.

Alignment: both videos start from the same real seed context; the student
emits 6 generated chunks. We skip SKIP leading frames (seed + transient,
same convention as motion_check) and compare the common prefix after that.
Raw PSNR is reported alongside because MSE alone is hard to read.

Env: TM_RUNS colon list of run names under flow_viz/.motion_check,
TM_TEACHER (dir of step05000_r08_{DIR}_raw.mp4), TM_SKIP, TM_OUT csv,
TM_DIRS (default all 8).
"""
import os
import numpy as np
import imageio.v2 as imageio
from PIL import Image

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
FV = f"{ARR}/analysis/eval_final/flow_viz"
MC = f"{FV}/.motion_check"
TEACH = os.environ.get(
    "TM_TEACHER", f"{ARR}/logs/eval_final/A/pca8_8node/control_test")
RUNS = [r for r in os.environ.get("TM_RUNS", "").split(":") if r]
REF_RUN = os.environ.get("TM_REF_RUN", "")
DIRS = os.environ.get("TM_DIRS", "F,FR,R,BR,B,BL,L,FL").split(",")
SKIP = int(os.environ.get("TM_SKIP", "14"))
# Horizon cap (frames scored AFTER skip). The window is otherwise
# min(len(student), len(teacher)), i.e. set by whichever recording is
# shorter -- so two runs recorded at different chunk counts are scored over
# DIFFERENT rollout horizons and their `match` values are not comparable
# (measured: c10mserep2 94f vs alldir8n2 70f). Match falls with horizon
# because AR contraction accumulates, so the longer-recorded run is
# penalised for being recorded longer. Set TM_MAXF to the shortest run in
# the comparison to score like-for-like. 0 = unlimited (previous behaviour).
MAXF = int(os.environ.get("TM_MAXF", "0"))
OUT = os.environ.get("TM_OUT", f"{FV}/teacher_match.csv")
FFMPEG = dict(input_params=["-threads", "1"], output_params=["-threads", "1"])


def read(path, hw=None):
    r = imageio.get_reader(path, format="ffmpeg", **FFMPEG)
    out = []
    while True:
        try:
            f = r.get_next_data()
        except Exception:
            break
        if hw is not None and f.shape[:2] != hw:
            f = np.asarray(Image.fromarray(f).resize((hw[1], hw[0]),
                                                     Image.BILINEAR))
        out.append(f.astype(np.float32) / 255.0)
    r.close()
    return np.stack(out) if out else None


def score(student_path, teacher_path):
    t = read(teacher_path)
    if t is None:
        return None
    s = read(student_path, hw=t.shape[1:3])
    if s is None:
        return None
    n = min(len(s), len(t))
    if n <= SKIP + 1:
        return None
    if MAXF > 0:
        n = min(n, SKIP + MAXF)
    s, t = s[SKIP:n], t[SKIP:n]
    mse = float(np.mean((s - t) ** 2))
    var = float(np.var(t))
    return mse, 1.0 - mse / max(var, 1e-8), 10 * np.log10(1.0 / max(mse, 1e-12)), n - SKIP


def main():
    if not RUNS:
        raise SystemExit("set TM_RUNS=run1:run2:...")
    rows = []
    print(f"{'run':30} {'match':>7} {'mse':>9} {'psnr':>7}  (mean over dirs/seeds)")
    for run in RUNS:
        per = []
        for d in DIRS:
            for sd in (0, 1):
                # TM_REF_RUN: a SEED-MATCHED reference recording under
                # .motion_check (same window, same FR seed stream), so the
                # student and reference share their noise. Without it we
                # compare against a differently-seeded teacher video,
                # whose ceiling is only ~0.85 (measured: a model vs
                # ITSELF at another seed) — 0.95 is unreachable there.
                if REF_RUN:
                    tp = f"{MC}/{REF_RUN}/r08_{d}_s{sd}.mp4"
                else:
                    tp = f"{TEACH}/step05000_r08_{d}_raw.mp4"
                if not os.path.exists(tp):
                    continue
                sp = f"{MC}/{run}/r08_{d}_s{sd}.mp4"
                if not os.path.exists(sp):
                    continue
                r = score(sp, tp)
                if r is None:
                    continue
                mse, match, psnr, nf = r
                per.append((mse, match, psnr))
                rows.append((run, d, sd, mse, match, psnr, nf))
        if not per:
            print(f"{run:30}   (no comparable videos)")
            continue
        a = np.array(per)
        print(f"{run:30} {a[:,1].mean():7.4f} {a[:,0].mean():9.5f} "
              f"{a[:,2].mean():7.2f}   n={len(per)}")
    with open(OUT, "w") as fh:
        fh.write("run,dir,seed,mse,match,psnr,frames\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")
    print(f"\n[tm] wrote {OUT}  (target: match >= 0.95)")


if __name__ == "__main__":
    main()
