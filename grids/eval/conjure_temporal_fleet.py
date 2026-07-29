"""Temporal conjuration probe: a SUDDEN pop of a crisp central object.
For each sampled frame k, a conjuration is a high-confidence RT-DETR object (>=HI) in the central
band that (a) has NO precursor - no low-threshold (>=LO) detection of the same class near that
spot in the previous PREV frames (it did not ease in from small/afar), and (b) shows a large
motion-compensated pixel JUMP there between k-1 and k (a bunch of pixels suddenly changed).
Gradually approaching real objects have a small-detection precursor -> rejected. Runs the fleet
until 5 pops; draws each as a before->after pair so the sudden appearance is visible."""
import os
import numpy as np, torch, cv2
import fleet_common as fc
from transformers import AutoModelForObjectDetection, AutoImageProcessor

DEV = "cuda"; MODEL_ID = "PekingU/rtdetr_r50vd"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "conjure_temporal_fleet.png")
KEEPNAMES = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "backpack", "handbag", "suitcase"}
HI = 0.75; LO = 0.30; CXLO, CXHI = 0.30, 0.70; EDGE = 0.12
AREA_LO, AREA_HI = 0.006, 0.20            # crisp object, not a tiny speck nor a giant occlusion
PREV = 3                                   # frames to look back for a precursor
PRECUR_R = 0.16                            # precursor search radius (frac of diag)
SUDDEN_MIN = 2.0                           # box pixel-jump must exceed this x the whole-frame jump
STEP_S = 0.15; TARGET = 5; SEED = 11


def main():
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForObjectDetection.from_pretrained(MODEL_ID).eval().to(DEV)
    id2label = model.config.id2label

    @torch.no_grad()
    def detect(frame, thr):
        H, W = frame.shape[:2]
        inp = proc(images=frame, return_tensors="pt").to(DEV); out = model(**inp)
        res = proc.post_process_object_detection(out, target_sizes=[(H, W)], threshold=thr)[0]
        return [(b, id2label[int(l)], float(s)) for b, l, s in zip(res["boxes"].cpu().numpy(),
                res["labels"].cpu().numpy(), res["scores"].cpu().numpy()) if id2label[int(l)] in KEEPNAMES]

    def events_for(scene, mdl):
        n, fps = fc.meta(scene, mdl); ctx = fc.ctx_of(mdl); end = min(n-1, ctx+int(round(6*fps)))
        step = max(1, int(round(STEP_S*fps))); idx = list(range(max(0, ctx-1), end+1, step))
        frames = fc.frames_at(scene, mdl, idx); H, W = frames[0].shape[:2]; diag = (W**2+H**2)**0.5
        gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32) for f in frames]
        lo = [detect(f, LO) for f in frames]
        hi = [(b, l) for b, l, s in lo[0] if s >= HI]  # placeholder not used
        out = []
        for k in range(PREV, len(frames)):
            for b, l, s in lo[k]:
                if s < HI:
                    continue
                cx = (b[0]+b[2])/2/W; area = (b[2]-b[0])*(b[3]-b[1])/(W*H)
                if not (CXLO < cx < CXHI and b[0] > EDGE*W and b[2] < (1-EDGE)*W and AREA_LO <= area <= AREA_HI):
                    continue
                ocx, ocy = (b[0]+b[2])/2, (b[1]+b[3])/2
                # precursor? any LO same-class detection near this spot in the previous PREV frames
                prec = False
                for kk in range(k-PREV, k):
                    for pb, pl, ps in lo[kk]:
                        if pl == l and (((pb[0]+pb[2])/2-ocx)**2 + ((pb[1]+pb[3])/2-ocy)**2)**0.5/diag < PRECUR_R:
                            prec = True; break
                    if prec:
                        break
                if prec:
                    continue
                # sudden pixel jump in the box vs whole frame
                bi = b.astype(int)
                dbox = np.abs(gray[k][max(0, bi[1]):bi[3], max(0, bi[0]):bi[2]] - gray[k-1][max(0, bi[1]):bi[3], max(0, bi[0]):bi[2]])
                dfull = np.abs(gray[k] - gray[k-1])
                sudden = dbox.mean()/max(dfull.mean(), 1e-3) if dbox.size else 0.0
                if sudden < SUDDEN_MIN:
                    continue
                out.append(dict(scene=scene, mdl=mdl, cls=l, k=k, idx=idx, box=bi, prevframe=frames[k-1], nowframe=frames[k], sudden=sudden))
                break   # one pop per video is enough
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
    rows = []
    for e in events:
        pair = []
        for tag, fr, box in [("before", e["prevframe"], None), ("AFTER: pop", e["nowframe"], e["box"])]:
            im = fr.copy()
            if box is not None:
                cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), (60, 60, 255), 3)
            im = cv2.cvtColor(cv2.resize(im, (360, int(360*im.shape[0]/im.shape[1]))), cv2.COLOR_RGB2BGR)
            h = np.full((18, im.shape[1], 3), 25, np.uint8)
            cv2.putText(h, tag, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
            pair.append(np.concatenate([h, im], 0))
        body = np.concatenate(pair, 1)
        lab = np.full((28, body.shape[1], 3), 18, np.uint8)
        cv2.putText(lab, f"{e['mdl']}_{e['scene']}  {e['cls']}  sudden={e['sudden']:.1f}", (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 220, 255), 1)
        rows.append(np.concatenate([lab, body], 0))
    W = max(r.shape[1] for r in rows); rows = [cv2.copyMakeBorder(r, 0, 0, 0, W-r.shape[1], cv2.BORDER_CONSTANT, value=(18, 18, 18)) for r in rows]
    cv2.imwrite(OUT, np.concatenate(rows, 0)); print("wrote", OUT)


if __name__ == "__main__":
    main()
