"""Validated conjuration probe on the FLEET, run until 5 hits.
Probe (validated 9/10 on the labeled minwm set):
  RT-DETR conf>=0.60 -> track-birth of an object into a spot empty of the SAME class recently
  -> central band (0.33<cx<0.67), crisp (max-area>=0.010)
  -> AND (its class is NEW to the scene [absent from the real-context frames] OR it grows >=2x).
The real-context gate rejects real objects that were present/approaching from the start.
Draws each hit at its largest frame. Writes out/conjure_find5_fleet.png."""
import os
import numpy as np, torch, cv2
import fleet_common as fc
from transformers import AutoModelForObjectDetection, AutoImageProcessor

DEV = "cuda"; MODEL_ID = "PekingU/rtdetr_r50vd"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "conjure_find5_fleet.png")
KEEPNAMES = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "backpack", "handbag", "suitcase"}
SCORE = 0.60; RC_THR = 0.45; CXLO, CXHI = 0.33, 0.67; EDGE = 0.12; AMAX_MIN = 0.010; AMAX_MAX = 0.15; GROW_MIN = 2.0
STEP_S = 0.15; MATCH_CDIST = 0.14; LOOKBACK = 8; OCC_R = 0.13
TARGET = 5; SEED = 11


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1]); x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, x1-x0), max(0, y1-y0); inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


def main():
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForObjectDetection.from_pretrained(MODEL_ID).eval().to(DEV)
    id2label = model.config.id2label

    @torch.no_grad()
    def detect(frame, thr):
        H, W = frame.shape[:2]
        inp = proc(images=frame, return_tensors="pt").to(DEV); out = model(**inp)
        res = proc.post_process_object_detection(out, target_sizes=[(H, W)], threshold=thr)[0]
        return [(b, id2label[int(l)]) for b, l, s in zip(res["boxes"].cpu().numpy(), res["labels"].cpu().numpy(),
                res["scores"].cpu().numpy()) if id2label[int(l)] in KEEPNAMES]

    def events_for(scene, mdl):
        n, fps = fc.meta(scene, mdl); ctx = fc.ctx_of(mdl); end = min(n-1, ctx+int(round(6*fps)))
        # real-context classes (objects really in the scene before generation)
        rc = set()
        for i in range(0, ctx, max(1, ctx // 6)):
            for b, l in detect(fc.frames_at(scene, mdl, [i])[0], RC_THR):
                rc.add(l)
        step = max(1, int(round(STEP_S*fps))); idx = list(range(max(0, ctx-1), end+1, step))
        frames = fc.frames_at(scene, mdl, idx); H, W = frames[0].shape[:2]; diag = (W**2+H**2)**0.5
        tracks = []; hist = []
        for k, fr in enumerate(frames):
            ds = detect(fr, SCORE); used = [False]*len(ds)
            hist.append([((b[0]+b[2])/2, (b[1]+b[3])/2, l) for b, l in ds])
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
                    a = (ds[bi][0][2]-ds[bi][0][0])*(ds[bi][0][3]-ds[bi][0][1])/(W*H)
                    if a > t["amax"]:
                        t["amax"] = a; t["amax_box"] = ds[bi][0]; t["amax_k"] = idx[k]
            for di, (b, l) in enumerate(ds):
                if used[di]:
                    continue
                a = (b[2]-b[0])*(b[3]-b[1])/(W*H)
                tracks.append({"box": b, "cls": l, "birth_k": k, "last": k, "cen": ((b[0]+b[2])/2, (b[1]+b[3])/2),
                               "a0": a, "amax": a, "amax_box": b, "amax_k": idx[k]})

        def empty(t):
            cx0, cy0 = t["cen"]; kb = t["birth_k"]
            for kk in range(max(0, kb-LOOKBACK), kb):
                for (hx, hy, hl) in hist[kk]:
                    if hl == t["cls"] and ((hx-cx0)**2+(hy-cy0)**2)**0.5/diag < OCC_R:
                        return False
            return True
        out = []
        for t in tracks:
            cx = t["cen"][0]/W; grow = t["amax"]/max(t["a0"], 1e-5)
            if (t["birth_k"] > 0 and t["last"]-t["birth_k"] >= 1 and empty(t) and CXLO < cx < CXHI
                    and AMAX_MIN <= t["amax"] <= AMAX_MAX and (t["cls"] not in rc or grow >= GROW_MIN)):
                out.append(dict(scene=scene, mdl=mdl, cls=t["cls"], newcls=(t["cls"] not in rc), grow=grow,
                                img=fc.frames_at(scene, mdl, [t["amax_k"]])[0], box=t["amax_box"].astype(int)))
        return out

    order = list(fc.fleet_index()); np.random.RandomState(SEED).shuffle(order)
    events = []
    for vn, (scene, mdl) in enumerate(order):
        if len(events) >= TARGET:
            break
        try:
            events.extend(events_for(scene, mdl))
        except Exception as e:
            print("skip", mdl, scene, str(e)[:50]); continue
        print(f"[{vn+1} videos] found={len(events)}", flush=True)
    events = events[:TARGET]
    print(f"DONE: {len(events)}", flush=True)
    cells = []
    for e in events:
        fr = e["img"].copy(); b = e["box"]
        cv2.rectangle(fr, (b[0], b[1]), (b[2], b[3]), (60, 60, 255), 3)
        im = cv2.cvtColor(cv2.resize(fr, (360, int(360*fr.shape[0]/fr.shape[1]))), cv2.COLOR_RGB2BGR)
        h = np.full((18, im.shape[1], 3), 25, np.uint8)
        why = "new-class" if e["newcls"] else f"grow{e['grow']:.1f}x"
        cv2.putText(h, f"{e['mdl']}_{e['scene']} {e['cls']} ({why})", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1)
        cells.append(np.concatenate([h, im], 0))
    ch = max(c.shape[0] for c in cells); cw = max(c.shape[1] for c in cells)
    cells = [cv2.copyMakeBorder(c, 0, ch-c.shape[0], 0, cw-c.shape[1], cv2.BORDER_CONSTANT, value=(25, 25, 25)) for c in cells]
    cv2.imwrite(OUT, np.concatenate(cells, 0)); print("wrote", OUT)


if __name__ == "__main__":
    main()
