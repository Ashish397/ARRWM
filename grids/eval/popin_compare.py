"""Compare object-detection backbones for the pop-in detector.

The temporal logic is backbone-agnostic, so the choice of detector is an empirical question.
This runs the identical pipeline with three architecturally distinct backbones:

  rtdetr     transformer / set prediction (NMS-free)
  frcnn      two-stage CNN (region proposals + ROI heads)
  retinanet  one-stage CNN (dense anchors + focal loss)

over two sets:

  minwm32    the 32 stationary_evaluation clips, which have ground truth -> precision/recall
  blind100   the blind evaluation set, which has no pop-in labels -> agreement + cost, and the
             annotation material needed to *get* labels

Outputs (out/):
  popin_backend_minwm32.csv     per-clip scores on the labelled set
  popin_backend_blind100.csv    per-clip scores on the blind set
  popin_backend_summary.txt     accuracy, agreement matrix, throughput

Usage: python popin_compare.py [--sets minwm32,blind100] [--backends rtdetr,frcnn,retinanet]
"""
import os, sys, time, json, glob
import numpy as np, imageio.v3 as iio, pandas as pd

import popin_detect as P
import popin_backends as B

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "out")
GT_MINWM = {2, 6, 8, 11, 15, 21, 23, 28, 29, 31}


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def minwm32_clips():
    out = []
    for p in sorted(glob.glob("/home/ashish/stationary_evaluation/minwm_r*.mp4")):
        r = int(os.path.basename(p).split("_r")[1].split(".")[0])
        out.append(dict(uid=f"minwm_r{r:02d}", path=p, ctx=13, gt=int(r in GT_MINWM)))
    return out


def blind100_clips():
    import blind100_common as BC
    return [dict(uid=r["blind_id"], path=r["path"], ctx=r["ctx"], gt=None,
                 model=r["model"], scene=r["scene"], direction=r["direction"]) for r in BC.refs()]


def run_set(clips, backends, setname):
    """Decode each clip once, score it with every backend."""
    built = {b: B.build(b) for b in backends}
    rows, timing = [], {b: [0.0, 0] for b in backends}
    for i, c in enumerate(clips, 1):
        try:
            vid = iio.imread(c["path"], plugin="pyav")
            fps = float(iio.immeta(c["path"], plugin="pyav").get("fps", 16) or 16)
        except Exception as e:
            print(f"[{setname} {i:3d}/{len(clips)}] {c['uid']:<14} SKIP ({e})", flush=True)
            continue
        P.set_fps(fps)
        row = {k: v for k, v in c.items() if k != "path"}
        row.update(n=len(vid), fps=fps, w=vid.shape[2], h=vid.shape[1])
        msg = []
        for b in backends:
            dense, crop = built[b]
            P.set_detector(crop)
            t0 = time.time()
            rec = {"n": len(vid), "w": vid.shape[2], "h": vid.shape[1], "dets": dense(vid)}
            ev = P.analyse(vid, rec, c["ctx"])
            timing[b][0] += time.time() - t0
            timing[b][1] += len(vid)
            hits = [f for f in ev if f["score"] > 0]
            top = ev[0] if ev else None
            row[f"{b}_flag"] = int(bool(hits))
            row[f"{b}_score"] = top["score"] if top else None
            row[f"{b}_cls"] = top["cls"] if top else None
            row[f"{b}_birth"] = top["birth"] if top else None
            row[f"{b}_branch"] = top.get("branch") if top else None
            row[f"{b}_box"] = json.dumps([round(v, 1) for v in top["box"]]) if top else None
            msg.append(f"{b}={'HIT' if hits else '-  '}" + (f"{top['score']:+.2f}" if top else "     "))
        rows.append(row)
        print(f"[{setname} {i:3d}/{len(clips)}] {c['uid']:<14} n={len(vid):4d} fps={fps:4.0f}  " + "  ".join(msg), flush=True)
    return pd.DataFrame(rows), timing


def accuracy(df, backends):
    lines = ["ACCURACY on minwm32 (ground truth: 10 of 32 clips contain a conjured object)", ""]
    lines.append(f"{'backend':<12}{'TP':>4}{'FP':>4}{'FN':>4}{'prec':>7}{'rec':>7}{'F1':>7}   flagged")
    for b in backends:
        fl = set(df.loc[df[f"{b}_flag"] == 1, "uid"])
        gt = set(df.loc[df["gt"] == 1, "uid"])
        tp, fp, fn = len(fl & gt), len(fl - gt), len(gt - fl)
        pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
        f1 = 2 * pr * rc / max(1e-9, pr + rc)
        short = sorted(int(u.split("_r")[1]) for u in fl)
        lines.append(f"{b:<12}{tp:>4}{fp:>4}{fn:>4}{pr:>7.2f}{rc:>7.2f}{f1:>7.2f}   {short}")
    return "\n".join(lines)


def agreement(df, backends, setname):
    lines = [f"", f"AGREEMENT on {setname} ({len(df)} clips) -- pairwise Jaccard / both-flag / either-flag", ""]
    for i, a in enumerate(backends):
        for b in backends[i + 1:]:
            A = set(df.loc[df[f"{a}_flag"] == 1, "uid"]); Bs = set(df.loc[df[f"{b}_flag"] == 1, "uid"])
            inter, union = len(A & Bs), len(A | Bs)
            lines.append(f"  {a:<10} vs {b:<10} J={inter/max(1,union):.2f}  both={inter:3d}  either={union:3d}  "
                         f"{a}-only={len(A-Bs):3d}  {b}-only={len(Bs-A):3d}")
    lines.append("")
    lines.append(f"{'backend':<12}{'flagged':>8}{'rate':>8}")
    for b in backends:
        k = int(df[f"{b}_flag"].sum())
        lines.append(f"{b:<12}{k:>8}{k/max(1,len(df)):>8.2f}")
    return "\n".join(lines)


def main():
    sets = arg("--sets", "minwm32,blind100").split(",")
    backends = arg("--backends", ",".join(B.NAMES)).split(",")
    os.makedirs(OUTD, exist_ok=True)
    report = []

    if "minwm32" in sets:
        df, tm = run_set(minwm32_clips(), backends, "minwm32")
        df.to_csv(os.path.join(OUTD, "popin_backend_minwm32.csv"), index=False)
        report.append(accuracy(df, backends))
        report.append(agreement(df, backends, "minwm32"))

    if "blind100" in sets:
        df2, tm2 = run_set(blind100_clips(), backends, "blind100")
        df2.to_csv(os.path.join(OUTD, "popin_backend_blind100.csv"), index=False)
        report.append(agreement(df2, backends, "blind100"))
        report.append("\nTHROUGHPUT on blind100 (detect + full temporal analysis)\n")
        report.append(f"{'backend':<12}{'frames':>9}{'seconds':>10}{'fps':>8}")
        for b in backends:
            s, n = tm2[b]
            report.append(f"{b:<12}{n:>9}{s:>10.1f}{n/max(1e-9,s):>8.1f}")

    txt = "\n".join(report)
    open(os.path.join(OUTD, "popin_backend_summary.txt"), "w").write(txt + "\n")
    print("\n" + txt)


if __name__ == "__main__":
    main()
