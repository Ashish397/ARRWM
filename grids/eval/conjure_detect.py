"""Prototype: conjuration = a real object (COCO vehicle/person) that is BORN in the
central portion of the frame, not touching any edge, and persists. This removes the
edge-reveal confound (real objects enter from edges as the camera moves) and ignores
warps/haze (not detected as objects at all).
Usage: python conjure_detect.py   (runs on a fixed probe set, writes viz PNGs)"""
import os, sys
import numpy as np, torch, cv2
import fleet_common as fc
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

DEV = "cuda"
HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "out", "conjure_proto"); os.makedirs(OUTD, exist_ok=True)
# COCO ids: 1 person, 2 bicycle, 3 car, 4 motorcycle, 6 bus, 8 truck
KEEP = {1: "person", 2: "bike", 3: "car", 4: "moto", 6: "bus", 8: "truck"}
SCORE = 0.55
# central-birth gates
CX = (0.20, 0.80); CY = (0.10, 0.95)     # box CENTER must be here (bottom allowed: foreground)
EDGE = 0.04                               # box must not touch LEFT/RIGHT edge (side reveal)
MIN_AREA = 0.010                          # birth box >=1% of frame (not a distant speck)
PERSIST = 2                    # sampled frames (0.3s apart) a central object must survive
IOU_MATCH = 0.3
STEP_S = 0.30                             # sample every 0.30s (dense enough to track)

PROBE = [("r08_F", "minwm"), ("r03_BR", "yume"), ("r05_BL", "ours_pca4"),
         ("r00_FR", "worldcam"), ("r16_L", "ours_4node"), ("r03_FL", "minwm"),
         ("r09_F", "minwm")]


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1]); x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, x1 - x0), max(0, y1 - y0); inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn_v2(weights=weights).eval().to(DEV)
    tf = weights.transforms()

    @torch.no_grad()
    def detect(frame):
        x = tf(torch.from_numpy(frame).permute(2, 0, 1)).to(DEV)
        o = model([x])[0]
        out = []
        for b, l, s in zip(o["boxes"].cpu().numpy(), o["labels"].cpu().numpy(), o["scores"].cpu().numpy()):
            if s >= SCORE and int(l) in KEEP:
                out.append((b, int(l), float(s)))
        return out

    for scene, mdl in PROBE:
        n, fps = fc.meta(scene, mdl); ctx = fc.ctx_of(mdl)
        end = min(n - 1, ctx + int(round(6.0 * fps)))
        step = max(1, int(round(STEP_S * fps)))
        idx = list(range(ctx, end + 1, step))
        frames = fc.frames_at(scene, mdl, idx)
        H, W = frames[0].shape[:2]
        tracks = []   # each: {box, cls, birth_k, age, birth_central}
        events = []
        for k, fr in enumerate(frames):
            dets = detect(fr)
            used = [False] * len(dets)
            diag = (W ** 2 + H ** 2) ** 0.5
            def cen(bx):
                return ((bx[0]+bx[2])/2, (bx[1]+bx[3])/2)
            for t in tracks:
                if k - t["last"] > 2:            # track went stale; don't revive across big gaps
                    continue
                best, bi = 0.0, -1
                tcx, tcy = cen(t["box"])
                for di, (b, l, s) in enumerate(dets):
                    if used[di] or l != t["cls"]:
                        continue
                    bcx, bcy = cen(b)
                    cdist = ((tcx-bcx)**2 + (tcy-bcy)**2) ** 0.5 / diag
                    score = max(iou(t["box"], b) / IOU_MATCH, (0.14 - cdist) / 0.14)  # IoU>0.3 OR centroid<14%
                    if score > 1.0 and score > best:
                        best, bi = score, di
                if bi >= 0:
                    used[bi] = True; t["box"] = dets[bi][0]; t["age"] += 1; t["last"] = k
            for di, (b, l, s) in enumerate(dets):
                if used[di]:
                    continue
                cx = (b[0]+b[2])/2/W; cy = (b[1]+b[3])/2/H
                area = (b[2]-b[0])*(b[3]-b[1])/(W*H)
                inside_lr = (b[0] > EDGE*W and b[2] < (1-EDGE)*W)   # did not enter from a side edge
                central = (CX[0] < cx < CX[1]) and (CY[0] < cy < CY[1]) and inside_lr and area >= MIN_AREA
                tracks.append({"box": b, "cls": l, "birth_k": k, "age": 1, "last": k,
                               "central": central, "birth_box": b})
        # a conjuration event = born-central track that persisted
        for t in tracks:
            if t["central"] and t["age"] >= PERSIST and t["birth_k"] > 0:
                events.append(t)
        # viz: birth frames of events
        tag = f"{mdl}_{scene}"
        print(f"{tag:22s} tracks={len(tracks):3d}  central_births={sum(x['central'] for x in tracks):2d}  "
              f"CONJURE_EVENTS={len(events)}  -> {'FLAG' if events else 'clean'}")
        vis = frames[0].copy()
        panels = []
        for t in events[:4]:
            fr = frames[t["birth_k"]].copy()
            b = t["birth_box"].astype(int)
            cv2.rectangle(fr, (b[0], b[1]), (b[2], b[3]), (255, 60, 60), 3)
            cv2.putText(fr, f"{KEEP[t['cls']]} born k={t['birth_k']}", (b[0], max(20, b[1]-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 60, 60), 2)
            panels.append(cv2.cvtColor(cv2.resize(fr, (360, int(360*fr.shape[0]/fr.shape[1]))), cv2.COLOR_RGB2BGR))
        if not panels:  # clean: just show a mid frame
            fr = frames[len(frames)//2]
            panels.append(cv2.cvtColor(cv2.resize(fr, (360, int(360*fr.shape[0]/fr.shape[1]))), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(OUTD, f"{tag}.png"), np.concatenate(panels, 1))
    print("viz ->", OUTD)


if __name__ == "__main__":
    main()
