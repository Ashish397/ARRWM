"""Run the deployed conjuration detector over every clip in stationary_evaluation/.

448 clips = 14 systems x 32 scenes, mixed frame rate, resolution and context length. The `real`
system is real footage and acts as a control: a detector that flags conjurations should return
almost nothing on it.

Outputs (out/popin_stationary/):
  popin_stationary.csv        one row per clip: flag, score, class, birth frame, evidence
  summary.txt                 per-system flag counts and rates
  reel_<system>_<scene>.png   filmstrip for every flagged clip
  popin_stationary_<sys>.png  all of a system's flagged clips stacked

Usage: python popin_stationary.py [--backend rtdetr] [--dir DIR]
"""
import os, sys, json, time
import numpy as np, cv2, imageio.v3 as iio, pandas as pd

import popin_detect as P
import popin_backends as B

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = "/home/ashish/stationary_evaluation"
OUTD = os.path.join(HERE, "out", "popin_stationary")

# real context frames per system (fleet convention; ablations use the frame-12 boundary).
# `real` is genuine footage with no generated span -- scored from the same boundary as the
# ablations so the control is measured over a comparable window.
CTX = {"minwm": 13, "astra": 4, "matrixgame": 1, "worldcam": 65, "worldplay": 1, "yume": 1,
       "16node": 12, "4node": 12, "pca8": 12, "pca4": 12, "pca2": 12,
       "noatok": 12, "noadaln": 12, "real": 12}


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def reel(vid, fps, f, title, path):
    b, n = f["birth"], len(vid)
    offs = [-1.0, -0.5, -0.15, 0.0, 0.5, 1.5, 3.0]
    x0, y0, x1, y1 = [int(v) for v in f["box"]]
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    s = int(max(x1 - x0, y1 - y0) * 1.8)
    H, W = vid.shape[1:3]
    TW, TH = 300, 172
    full, zoom = [], []
    for o in offs:
        k = min(n - 1, max(0, b + int(round(o * fps))))
        fr = vid[k].copy()
        cv2.rectangle(fr, (x0, y0), (x1, y1), (60, 225, 60), 3)
        fr = cv2.resize(fr, (TW, TH))
        cv2.putText(fr, f"{o:+.1f}s", (4, 15), 0, 0.45, (255, 255, 0), 1)
        full.append(fr)
        cr = vid[k][max(0, cy - s):min(H, cy + s), max(0, cx - s):min(W, cx + s)]
        cr = cv2.resize(cr, (TH, TH), interpolation=cv2.INTER_NEAREST) if cr.size else np.zeros((TH, TH, 3), np.uint8)
        zoom.append(cv2.copyMakeBorder(cr, 0, 0, (TW - TH) // 2, TW - TH - (TW - TH) // 2,
                                       cv2.BORDER_CONSTANT, value=(18, 18, 18)))
    body = np.concatenate([np.concatenate(full, 1), np.concatenate(zoom, 1)], 0)
    lab = np.full((28, body.shape[1], 3), 16, np.uint8)
    cv2.putText(lab, title, (5, 20), 0, 0.48, (255, 255, 255), 1)
    img = np.concatenate([lab, body], 0)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return img


def main():
    os.makedirs(OUTD, exist_ok=True)
    vdir = arg("--dir", DIR)
    backend = arg("--backend", "rtdetr")
    dense, crop = B.build(backend)
    P.set_detector(crop)

    # MANIFEST.csv covers only the seven ablations, so enumerate the directory instead and
    # take system/scene from the filename -- otherwise half the clips are silently skipped.
    import glob
    files = []
    for p in sorted(glob.glob(os.path.join(vdir, "*.mp4"))):
        fn = os.path.basename(p)
        stem = os.path.splitext(fn)[0]
        sysname, _, sc = stem.rpartition("_r")
        if not sysname or not sc.isdigit():
            print(f"skipping unparsable name {fn}")
            continue
        files.append((sysname, int(sc), fn))
    man = files
    rows, hits, t0 = [], {}, time.time()
    for i, (sysname, scene, fn) in enumerate(man, 1):
        path = os.path.join(vdir, fn)
        if not os.path.exists(path):
            print(f"[{i:3d}/{len(man)}] {fn:<22} MISSING", flush=True)
            continue
        vid = iio.imread(path, plugin="pyav")
        fps = float(iio.immeta(path, plugin="pyav").get("fps", 16) or 16)
        ctx = CTX.get(sysname, 12)
        P.set_fps(fps)
        ev = P.analyse(vid, {"n": len(vid), "w": vid.shape[2], "h": vid.shape[1], "dets": dense(vid)}, ctx)
        pos = [f for f in ev if f["score"] > 0]
        top = ev[0] if ev else None
        rows.append(dict(file=fn, system=sysname, scene=scene, n=len(vid), fps=fps, ctx=ctx,
                         flag=int(bool(pos)), n_events=len(pos),
                         score=top["score"] if top else None,
                         cls=top["cls"] if top else None,
                         birth=top["birth"] if top else None,
                         birth_s=round(top["birth"] / fps, 2) if top else None,
                         branch=top.get("branch") if top else None,
                         onset=top["onset"] if top else None,
                         zprobe=top["zprobe"] if top else None,
                         bncc=top["bncc"] if top else None,
                         box=json.dumps([round(v, 1) for v in top["box"]]) if top else None))
        if pos:
            f = pos[0]
            title = (f"{sysname} r{scene:02d}   score={f['score']:+.2f}  [{f['cls']}]  "
                     f"birth={f['birth']/fps:.1f}s   onset={f['onset']} zprobe={f['zprobe']} bncc={f['bncc']}")
            p = os.path.join(OUTD, f"reel_{sysname}_r{scene:02d}.png")
            reel(vid, fps, f, title, p)
            hits.setdefault(sysname, []).append(p)
        pd.DataFrame(rows).to_csv(os.path.join(OUTD, "popin_stationary.csv"), index=False)
        mark = f"HIT {top['score']:+.2f}" if pos else ("-   " + (f"{top['score']:+.2f}" if top else "     "))
        print(f"[{i:3d}/{len(man)}] {fn:<22} n={len(vid):4d} fps={fps:4.0f} {mark}   {time.time()-t0:.0f}s", flush=True)

    d = pd.DataFrame(rows)
    lines = [f"conjuration detector ({backend}) on {len(d)} clips in {vdir}", ""]
    lines.append(f"{'system':<12}{'clips':>6}{'flagged':>9}{'rate':>7}   scenes")
    for s, g in d.groupby("system"):
        fl = sorted(g.loc[g["flag"] == 1, "scene"].tolist())
        lines.append(f"{s:<12}{len(g):>6}{int(g['flag'].sum()):>9}{g['flag'].mean():>7.2f}   {fl}")
    lines.append("")
    lines.append(f"TOTAL flagged: {int(d['flag'].sum())} / {len(d)}  ({d['flag'].mean():.3f})")
    txt = "\n".join(lines)
    open(os.path.join(OUTD, "summary.txt"), "w").write(txt + "\n")
    print("\n" + txt)

    for s, ps in hits.items():
        ims = [cv2.imread(p) for p in ps]
        w = max(i.shape[1] for i in ims)
        ims = [cv2.copyMakeBorder(i, 0, 6, 0, w - i.shape[1], cv2.BORDER_CONSTANT, value=(18, 18, 18)) for i in ims]
        cv2.imwrite(os.path.join(OUTD, f"popin_stationary_{s}.png"), np.concatenate(ims, 0))
    print(f"\nwrote {OUTD}")


if __name__ == "__main__":
    main()
