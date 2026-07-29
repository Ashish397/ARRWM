"""1-fps contact sheets for the same 90 random fleet videos (seed 3), with every conjuration
event the three detectors flagged drawn ON the frames, colour-coded by detector:
  Faster R-CNN = red, RetinaNet = green, FCOS = cyan.
A box is drawn on the 1-sec frame nearest each birth; the row tag lists which detectors fired.
Unmarked rows = nothing flagged (scan these for misses)."""
import os
import numpy as np, torch, cv2
import fleet_common as fc
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,
    retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights,
    fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights)

DEV = "cuda"
HERE = os.path.dirname(os.path.abspath(__file__)); OUTD = os.path.join(HERE, "out")
KEEP = {1: "person", 2: "bike", 3: "car", 4: "moto", 6: "bus", 8: "truck", 27: "backpack", 31: "handbag", 33: "suitcase"}
SCORE = 0.55; CX = (0.20, 0.80); EDGE = 0.12; MIN_AREA = 0.008
STEP_S = 0.15; MATCH_CDIST = 0.14; LOOKBACK = 8; OCC_R = 0.13
SEED = 3; NVID = 90; PER_SHEET = 18; PANW = 200; MAXCOL = 8
COL = {"frcnn": (60, 60, 255), "retinanet": (60, 220, 60), "fcos": (255, 220, 60)}  # BGR


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1]); x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, x1-x0), max(0, y1-y0); inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


def load_detectors():
    specs = [("frcnn", fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1),
             ("retinanet", retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1),
             ("fcos", fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights.COCO_V1)]
    out = {}
    for name, ctor, w in specs:
        model = ctor(weights=w).eval().to(DEV); tf = w.transforms()

        @torch.no_grad()
        def detect(frame, _m=model, _tf=tf):
            x = _tf(torch.from_numpy(frame).permute(2, 0, 1)).to(DEV); o = _m([x])[0]
            return [(b, int(l)) for b, l, s in zip(o["boxes"].cpu().numpy(), o["labels"].cpu().numpy(),
                    o["scores"].cpu().numpy()) if s >= SCORE and int(l) in KEEP]
        out[name] = detect
    return out


def events_for(detect, frames, W, H, fine_idx):
    diag = (W**2 + H**2) ** 0.5; tracks = []; hist = []
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
    ev = []
    for t in tracks:
        if t["birth_k"] > 0 and t["central"] and t["last"]-t["birth_k"] >= 1 and empty(t):
            ev.append(dict(abs=fine_idx[t["birth_k"]], box=t["bb"], cls=t["cls"]))
    return ev


def main():
    dets = load_detectors()
    order = list(fc.fleet_index()); np.random.RandomState(SEED).shuffle(order); order = order[:NVID]
    rows = []
    for vn, (scene, mdl) in enumerate(order):
        try:
            n, fps = fc.meta(scene, mdl); ctx = fc.ctx_of(mdl); end = min(n-1, ctx+int(round(6*fps)))
            fstep = max(1, int(round(STEP_S*fps))); fine_idx = list(range(max(0, ctx-1), end+1, fstep))
            fine = fc.frames_at(scene, mdl, fine_idx); H, W = fine[0].shape[:2]
            evs = {name: events_for(det, fine, W, H, fine_idx) for name, det in dets.items()}
            # contact frames (1/sec)
            cidx = ([max(0, ctx-1)] + list(range(ctx, end+1, max(1, int(round(fps))))))[:MAXCOL]
            cf = fc.frames_at(scene, mdl, cidx); sc = PANW/W
            cells = []
            for j, f in enumerate(cf):
                im = cv2.cvtColor(cv2.resize(f, (PANW, int(PANW*f.shape[0]/f.shape[1]))), cv2.COLOR_RGB2BGR)
                # draw any birth whose abs frame is closest to this contact frame
                for name, elist in evs.items():
                    for e in elist:
                        near = min(cidx, key=lambda ci: abs(ci-e["abs"]))
                        if near == cidx[j]:
                            b = (e["box"]*sc).astype(int)
                            cv2.rectangle(im, (b[0], b[1]), (b[2], b[3]), COL[name], 2)
                lb = "real" if j == 0 else f"{j}s"
                h = np.full((15, im.shape[1], 3), 28, np.uint8)
                cv2.putText(h, lb, (3, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
                cells.append(np.concatenate([h, im], 0))
            ch = cells[0].shape[0]
            while len(cells) < MAXCOL:
                cells.append(np.full((ch, PANW, 3), 28, np.uint8))
            body = np.concatenate(cells, 1)
            fired = [name for name in dets if evs[name]]
            tag = np.full((body.shape[0], 150, 3), (40, 40, 40) if fired else (18, 18, 18), np.uint8)
            cv2.putText(tag, mdl, (5, body.shape[0]//2-16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 220, 255), 1)
            cv2.putText(tag, scene, (5, body.shape[0]//2+2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 220, 255), 1)
            for i, name in enumerate(fired):
                cv2.putText(tag, name, (5, body.shape[0]//2+20+i*15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL[name], 1)
            rows.append(np.concatenate([tag, body], 1))
            print(f"[{vn+1}/90] {mdl}_{scene} fired={fired}", flush=True)
        except Exception as e:
            print("skip", mdl, scene, str(e)[:40])
    W = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, W-r.shape[1], cv2.BORDER_CONSTANT, value=(18, 18, 18)) for r in rows]
    for s in range(0, len(rows), PER_SHEET):
        stacked = []
        for r in rows[s:s+PER_SHEET]:
            stacked += [r, np.full((2, W, 3), 60, np.uint8)]
        p = os.path.join(OUTD, f"conjure_marked_{s//PER_SHEET+1}.png")
        cv2.imwrite(p, np.concatenate(stacked, 0)); print("wrote", p)


if __name__ == "__main__":
    main()
