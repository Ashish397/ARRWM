"""Scan random fleet clips with all three detection backbones until each has N positives.

Sampling is shared: every clip is decoded once and scored by all three backbones, so the
comparison is paired (same clips, same temporal logic, only the detector differs). Scanning
continues until the *slowest* backbone reaches its quota, so each ends up with >= N positives
and the extra clips are simply additional shared negatives.

Covers the full 256x13 fleet via fleet_common: the six external baselines plus the seven
ours_* ablations, which are de-tiled from the 3328x960 grid videos frame by frame.

Outputs (out/popin_fleet/):
  popin_fleet_scan.csv        one row per scanned clip, per-backend flag/score/class/birth
  reel_<backend>_NN_<clip>.png  filmstrip for every positive
  popin_fleet_<backend>.png   all that backend's positives stacked

Usage:
  python popin_fleet_scan.py [--n 20] [--seed 0] [--max 1200]
"""
import os, sys, json, random, time
import numpy as np, cv2, imageio, imageio.v3 as iio, pandas as pd

import popin_detect as P
import popin_backends as B
import fleet_common as fc

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "out", "popin_fleet")
os.makedirs(OUTD, exist_ok=True)


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def load(scene, model):
    """Decode a whole fleet clip. ours_* tiles are cropped out of the grid as it streams, so
    the full 3328x960 grid is never held in memory."""
    path = fc._path(scene, model)
    rd = imageio.get_reader(path)
    fps = rd.get_meta_data().get("fps", 16) or 16
    if model.startswith("ours_"):
        x, y = fc.POS[model[len("ours_"):]]
        frames = [np.asarray(f)[y + 32:y + 480, x:x + 832] for f in rd.iter_data()]
    else:
        frames = [np.asarray(f) for f in rd.iter_data()]
    rd.close()
    return np.stack(frames), float(fps)


def reel(vid, fps, f, title, path):
    b, n = f["birth"], len(vid)
    offs = [-1.2, -0.7, -0.25, 0.0, 0.35, 0.9, 1.8, 3.0]
    ks = [min(n - 1, max(0, b + int(round(o * fps)))) for o in offs]
    x0, y0, x1, y1 = [int(v) for v in f["box"]]
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    s = int(max(x1 - x0, y1 - y0) * 1.8)
    TW, TH = 280, 160
    H, W = vid.shape[1:3]
    full, zoom = [], []
    for k, o in zip(ks, offs):
        fr = vid[k].copy()
        cv2.rectangle(fr, (x0, y0), (x1, y1), (255, 45, 45), 2)
        fr = cv2.resize(fr, (TW, TH))
        cv2.putText(fr, f"{o:+.2f}s", (4, 14), 0, 0.4, (0, 255, 0), 1)
        full.append(fr)
        cr = vid[k][max(0, cy - s):min(H, cy + s), max(0, cx - s):min(W, cx + s)]
        cr = cv2.resize(cr, (TH, TH), interpolation=cv2.INTER_NEAREST) if cr.size else np.zeros((TH, TH, 3), np.uint8)
        zoom.append(cv2.copyMakeBorder(cr, 0, 0, (TW - TH) // 2, TW - TH - (TW - TH) // 2,
                                       cv2.BORDER_CONSTANT, value=(18, 18, 18)))
    body = np.concatenate([np.concatenate(full, 1), np.concatenate(zoom, 1)], 0)
    lab = np.full((28, body.shape[1], 3), 18, np.uint8)
    cv2.putText(lab, title, (5, 20), 0, 0.48, (255, 255, 255), 1)
    img = np.concatenate([lab, body], 0)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return img


def main():
    want = int(arg("--n", 20))
    seed = int(arg("--seed", 0))
    cap = int(arg("--max", 1200))
    backends = arg("--backends", ",".join(B.NAMES)).split(",")

    idx = fc.fleet_index()
    random.Random(seed).shuffle(idx)
    built = {b: B.build(b) for b in backends}
    got = {b: [] for b in backends}
    rows, scanned, t0 = [], 0, time.time()
    csv_path = os.path.join(OUTD, "popin_fleet_scan.csv")

    print(f"fleet={len(idx)} clips; scanning until every backend has {want} positives (cap {cap})\n", flush=True)
    for scene, model in idx:
        if all(len(got[b]) >= want for b in backends) or scanned >= cap:
            break
        scanned += 1
        uid = f"{model}_{scene}"
        try:
            vid, fps = load(scene, model)
        except Exception as e:
            print(f"[{scanned:4d}] {uid:<24} SKIP ({e})", flush=True)
            continue
        ctx = fc.ctx_of(model)
        P.set_fps(fps)
        row = dict(uid=uid, model=model, scene=scene, n=len(vid), fps=fps, ctx=ctx)
        msg = []
        for b in backends:
            dense, crop = built[b]
            P.set_detector(crop)
            ev = P.analyse(vid, {"n": len(vid), "w": vid.shape[2], "h": vid.shape[1], "dets": dense(vid)}, ctx)
            hits = [f for f in ev if f["score"] > 0]
            top = ev[0] if ev else None
            row[f"{b}_flag"] = int(bool(hits))
            row[f"{b}_score"] = top["score"] if top else None
            row[f"{b}_cls"] = top["cls"] if top else None
            row[f"{b}_birth"] = top["birth"] if top else None
            row[f"{b}_branch"] = top.get("branch") if top else None
            row[f"{b}_box"] = json.dumps([round(v, 1) for v in top["box"]]) if top else None
            if hits and len(got[b]) < want:
                k = len(got[b]) + 1
                f = hits[0]
                title = (f"{b} #{k}  {uid}  |  {f['cls']}  birth=f{f['birth']}  score={f['score']:+.2f} "
                         f"branch={f['branch']}  onset={f['onset']} zprobe={f['zprobe']} bncc={f['bncc']}")
                p = os.path.join(OUTD, f"reel_{b}_{k:02d}_{uid}.png")
                reel(vid, fps, f, title, p)
                got[b].append(dict(uid=uid, model=model, png=p, **{kk: f[kk] for kk in
                                   ("cls", "birth", "score", "branch", "onset", "zprobe", "bncc")}))
            msg.append(f"{b[:4]}={'HIT' if hits else '-  '}" + (f"{top['score']:+.2f}" if top else "     "))
        rows.append(row)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        tally = " ".join(f"{b[:4]}:{len(got[b])}" for b in backends)
        print(f"[{scanned:4d}] {uid:<24} n={len(vid):4d} " + "  ".join(msg) +
              f"   | {tally}  {time.time()-t0:.0f}s", flush=True)

    for b in backends:
        json.dump(got[b], open(os.path.join(OUTD, f"positives_{b}.json"), "w"), indent=1)
        if got[b]:
            ims = [cv2.imread(g["png"]) for g in got[b]]
            w = max(i.shape[1] for i in ims)
            ims = [cv2.copyMakeBorder(i, 0, 6, 0, w - i.shape[1], cv2.BORDER_CONSTANT, value=(18, 18, 18)) for i in ims]
            cv2.imwrite(os.path.join(OUTD, f"popin_fleet_{b}.png"), np.concatenate(ims, 0))
    print(f"\nscanned {scanned} clips in {time.time()-t0:.0f}s")
    for b in backends:
        print(f"  {b:<11} {len(got[b])} positives  (rate {len(got[b])/max(1,scanned):.3f})")
    print(f"csv: {csv_path}")


if __name__ == "__main__":
    main()
