"""Scan random baseline clips with the pop-in detector until N objects trip it, and build a
picture reel for each hit so the flagged objects can be eyed directly.

The baselines differ in frame rate (16-30), resolution and context length, so each clip is fed
to the detector with its own fps (temporal windows rescale) and its own context boundary.

Usage:
  python popin_scan.py [--n 5] [--seed 0] [--max 80] [--models astra,minwm,...]
"""
import os, sys, glob, json, random
import numpy as np, cv2, torch, imageio.v3 as iio
from transformers import AutoModelForObjectDetection, AutoImageProcessor

import popin_detect as P

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.path.dirname(HERE), "baselines")
OUTD = os.path.join(HERE, "out", "popin_reels")
os.makedirs(OUTD, exist_ok=True)

# context frames per model (matches fleet_common.EXT_CTX); worldplay is not laid out as
# A_<model>/<model>_<scene>.mp4 so it is out of scope here.
CTX = {"astra": 4, "matrixgame": 1, "minwm": 13, "worldcam": 65, "yume": 1}
LOW = 0.10          # cache threshold: the tracker and the prior-evidence probes pick their own
BATCH = 8


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def build_detector():
    proc = AutoImageProcessor.from_pretrained("PekingU/rtdetr_r50vd", use_fast=True)
    mdl = AutoModelForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd").eval().cuda()
    id2label = mdl.config.id2label

    @torch.no_grad()
    def dense(vid):
        H, W = vid.shape[1:3]
        out = []
        for i in range(0, len(vid), BATCH):
            fr = list(vid[i:i + BATCH])
            inp = proc(images=fr, return_tensors="pt").to("cuda")
            res = proc.post_process_object_detection(mdl(**inp), target_sizes=[(H, W)] * len(fr), threshold=LOW)
            for r in res:
                out.append([{"box": [float(x) for x in b], "cls": id2label[int(l)], "score": float(s)}
                            for b, l, s in zip(r["boxes"].cpu().numpy(), r["labels"].cpu().numpy(),
                                               r["scores"].cpu().numpy())])
        return out
    return dense


def reel(vid, fps, f, title, path):
    """A two-row strip: full frames across the birth, and zoomed crops of the same moments.
    Times are chosen relative to birth in seconds so reels are comparable across frame rates."""
    b, n = f["birth"], len(vid)
    offs = [-1.2, -0.7, -0.25, 0.0, 0.35, 0.9, 1.8, 3.0]
    ks = [min(n - 1, max(0, b + int(round(o * fps)))) for o in offs]
    x0, y0, x1, y1 = [int(v) for v in f["box"]]
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    s = int(max(x1 - x0, y1 - y0) * 1.8)
    TW, TH = 300, 172
    full, zoom = [], []
    for k, o in zip(ks, offs):
        fr = vid[k].copy()
        cv2.rectangle(fr, (x0, y0), (x1, y1), (255, 45, 45), 2)
        fr = cv2.resize(fr, (TW, TH))
        cv2.putText(fr, f"{o:+.2f}s f{k}", (4, 15), 0, 0.42, (0, 255, 0), 1)
        full.append(fr)
        H, W = vid.shape[1:3]
        zx0, zy0 = max(0, cx - s), max(0, cy - s)
        cr = vid[k][zy0:min(H, cy + s), zx0:min(W, cx + s)]
        cr = cv2.resize(cr, (TH, TH), interpolation=cv2.INTER_NEAREST) if cr.size else np.zeros((TH, TH, 3), np.uint8)
        zoom.append(cv2.copyMakeBorder(cr, 0, 0, (TW - TH) // 2, TW - TH - (TW - TH) // 2,
                                       cv2.BORDER_CONSTANT, value=(18, 18, 18)))
    body = np.concatenate([np.concatenate(full, 1), np.concatenate(zoom, 1)], 0)
    lab = np.full((30, body.shape[1], 3), 18, np.uint8)
    cv2.putText(lab, title, (5, 21), 0, 0.52, (255, 255, 255), 1)
    img = np.concatenate([lab, body], 0)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return img


def main():
    want = int(arg("--n", 5))
    seed = int(arg("--seed", 0))
    cap = int(arg("--max", 80))
    models = arg("--models", ",".join(CTX)).split(",")

    paths = []
    for m in models:
        paths += [(m, p) for p in sorted(glob.glob(os.path.join(BASE, f"A_{m}", f"{m}_*.mp4")))]
    random.Random(seed).shuffle(paths)
    print(f"{len(paths)} clips across {models}; scanning at most {cap} for {want} hits\n")

    dense = build_detector()
    hits, scanned, imgs = [], 0, []
    for model, path in paths:
        if len(hits) >= want or scanned >= cap:
            break
        scanned += 1
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            vid = iio.imread(path, plugin="pyav")
            fps = iio.immeta(path, plugin="pyav").get("fps", 16) or 16
        except Exception as e:
            print(f"[{scanned:3d}] {name:<28} skipped ({e})")
            continue
        P.set_fps(fps)
        rec = {"n": len(vid), "w": vid.shape[2], "h": vid.shape[1], "dets": dense(vid)}
        ev = [f for f in P.analyse(vid, rec, CTX[model]) if f["score"] > 0]
        top = max((f["score"] for f in ev), default=None)
        print(f"[{scanned:3d}] {name:<28} n={len(vid):4d} fps={fps:.0f} "
              f"{'HIT ' + str(len(ev)) if ev else 'clean'}" + (f"  best={top:+.2f}" if ev else ""))
        for f in ev:
            if len(hits) >= want:
                break
            idx = len(hits) + 1
            title = (f"#{idx}  {name}  |  {f['cls']}  birth=f{f['birth']} ({f['birth']/fps:.1f}s)  "
                     f"score={f['score']:+.2f} branch={f['branch']}  "
                     f"onset={f['onset']} zprobe={f['zprobe']} bncc={f['bncc']}")
            out = os.path.join(OUTD, f"reel_{idx:02d}_{name}.png")
            imgs.append(reel(vid, fps, f, title, out))
            hits.append({"clip": name, "model": model, "fps": fps, "path": path, **f})
            print(f"        -> reel {out}")

    json.dump(hits, open(os.path.join(OUTD, "hits.json"), "w"), indent=1)
    if imgs:
        w = max(i.shape[1] for i in imgs)
        imgs = [cv2.copyMakeBorder(i, 0, 8, 0, w - i.shape[1], cv2.BORDER_CONSTANT, value=(18, 18, 18)) for i in imgs]
        allp = os.path.join(OUTD, "popin_reels_all.png")
        cv2.imwrite(allp, cv2.cvtColor(np.concatenate(imgs, 0), cv2.COLOR_RGB2BGR))
        print(f"\nwrote {allp}")
    print(f"\n{len(hits)} hits from {scanned} clips scanned")
    for h in hits:
        print(f"  {h['clip']:<28} {h['cls']:<8} score={h['score']:+.2f} branch={h['branch']}")


if __name__ == "__main__":
    main()
