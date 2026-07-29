"""Run the good conjuration probe (EXACT conjure_probe.py logic: Faster R-CNN R50-FPN v2,
track-birth + central birth coordinate + spot-was-empty occupancy check, 0.15s sampling) over
shuffled fleet videos until it has found 20 conjured objects. Draws each birth box on its frame
and writes a 20-object review montage. Does not stop before 20."""
import os
import numpy as np, torch, cv2
import fleet_common as fc
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

DEV = "cuda"
HERE = os.path.dirname(os.path.abspath(__file__)); OUTD = os.path.join(HERE, "out")
OUT = os.path.join(OUTD, "conjure_find20.png")
KEEP = {1: "person", 2: "bike", 3: "car", 4: "moto", 6: "bus", 8: "truck", 27: "backpack", 31: "handbag", 33: "suitcase"}
SCORE = 0.55; CX = (0.20, 0.80); EDGE = 0.12; MIN_AREA = 0.008
STEP_S = 0.15; MATCH_CDIST = 0.14; LOOKBACK = 8; OCC_R = 0.13
SHARP_MIN = 150.0          # quality gate: box Laplacian variance (crisp object, not a melt-blob)
TARGET = 20; SEED = 11
OUT = os.path.join(HERE, "out", "conjure_find20_sharp.png")


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1]); x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, x1-x0), max(0, y1-y0); inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


def main():
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn_v2(weights=weights).eval().to(DEV); tf = weights.transforms()

    @torch.no_grad()
    def detect(frame):
        x = tf(torch.from_numpy(frame).permute(2, 0, 1)).to(DEV); o = model([x])[0]
        return [(b, int(l)) for b, l, s in zip(o["boxes"].cpu().numpy(), o["labels"].cpu().numpy(),
                o["scores"].cpu().numpy()) if s >= SCORE and int(l) in KEEP]

    def events_for(scene, mdl):
        n, fps = fc.meta(scene, mdl); ctx = fc.ctx_of(mdl); end = min(n-1, ctx+int(round(6*fps)))
        step = max(1, int(round(STEP_S*fps))); idx = list(range(max(0, ctx-1), end+1, step))
        frames = fc.frames_at(scene, mdl, idx); H, W = frames[0].shape[:2]; diag = (W**2+H**2)**0.5
        tracks = []; hist = []
        for k, fr in enumerate(frames):
            ds = detect(fr); used = [False]*len(ds)
            hist.append([((b[0]+b[2])/2, (b[1]+b[3])/2) for b, l in ds])
            for t in tracks:
                if k - t["last"] > 2:
                    continue
                tcx, tcy = (t["box"][0]+t["box"][2])/2, (t["box"][1]+t["box"][3])/2; best, bi = 0.0, -1
                for di, (b, l) in enumerate(ds):
                    if used[di] or l != t["cls"]:
                        continue
                    bcx, bcy = (b[0]+b[2])/2, (b[1]+b[3])/2; cd = ((tcx-bcx)**2+(tcy-bcy)**2)**0.5/diag
                    sco = max(iou(t["box"], b)/0.3, (MATCH_CDIST-cd)/MATCH_CDIST)
                    if sco > 1.0 and sco > best:
                        best, bi = sco, di
                if bi >= 0:
                    used[bi] = True; t["box"] = ds[bi][0]; t["last"] = k
            for di, (b, l) in enumerate(ds):
                if used[di]:
                    continue
                cx = (b[0]+b[2])/2/W; area = (b[2]-b[0])*(b[3]-b[1])/(W*H)
                central = (CX[0] < cx < CX[1] and b[0] > EDGE*W and b[2] < (1-EDGE)*W and area >= MIN_AREA)
                tracks.append({"box": b, "cls": l, "birth_k": k, "last": k, "cen": ((b[0]+b[2])/2, (b[1]+b[3])/2), "central": central, "bb": b})

        def empty(t):
            cx0, cy0 = t["cen"]; kb = t["birth_k"]
            for kk in range(max(0, kb-LOOKBACK), kb):
                for (hx, hy) in hist[kk]:
                    if ((hx-cx0)**2+(hy-cy0)**2)**0.5/diag < OCC_R:
                        return False
            return True
        out = []
        for t in tracks:
            if t["birth_k"] > 0 and t["central"] and t["last"]-t["birth_k"] >= 1 and empty(t):
                fr = frames[t["birth_k"]]; b = t["bb"].astype(int)
                g = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY).astype(np.float64)
                patch = g[max(0, b[1]):b[3], max(0, b[0]):b[2]]
                sharp = cv2.Laplacian(patch, cv2.CV_64F).var() if patch.size > 20 else 0.0
                if sharp < SHARP_MIN:                       # quality gate: reject low-def melt-blobs
                    continue
                out.append(dict(scene=scene, mdl=mdl, cls=t["cls"], img=fr.copy(), box=b, sharp=sharp))
        return out

    order = list(fc.fleet_index()); np.random.RandomState(SEED).shuffle(order)
    events = []
    for vn, (scene, mdl) in enumerate(order):
        if len(events) >= TARGET:
            break
        try:
            events.extend(events_for(scene, mdl))
        except Exception as e:
            print("skip", mdl, scene, str(e)[:40]); continue
        print(f"[{vn+1} videos] found={len(events)}", flush=True)
    events = events[:TARGET]
    print(f"DONE: {len(events)} conjured objects", flush=True)
    cells = []
    for e in events:
        fr = e["img"].copy(); b = e["box"]
        cv2.rectangle(fr, (b[0], b[1]), (b[2], b[3]), (60, 60, 255), 3)
        im = cv2.cvtColor(cv2.resize(fr, (320, int(320*fr.shape[0]/fr.shape[1]))), cv2.COLOR_RGB2BGR)
        h = np.full((18, im.shape[1], 3), 25, np.uint8)
        cv2.putText(h, f"{e['mdl']}_{e['scene']} {KEEP[e['cls']]} s{e['sharp']:.0f}", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1)
        cells.append(np.concatenate([h, im], 0))
    ch = max(c.shape[0] for c in cells); cw = max(c.shape[1] for c in cells)
    cells = [cv2.copyMakeBorder(c, 0, ch-c.shape[0], 0, cw-c.shape[1], cv2.BORDER_CONSTANT, value=(25, 25, 25)) for c in cells]
    cols = 5
    rows = []
    for i in range(0, len(cells), cols):
        row = cells[i:i+cols] + [np.full((ch, cw, 3), 25, np.uint8)]*(cols-len(cells[i:i+cols]))
        rows.append(np.concatenate(row, 1))
    cv2.imwrite(OUT, np.concatenate(rows, 0)); print("wrote", OUT)


if __name__ == "__main__":
    main()
