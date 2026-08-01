"""Conjuration probe = object detector + new-object birth + birth COORDINATE.

Per rollout, sample ~every 0.3s (frame 0 is the last real context frame, to seed objects that
were really there). Detect COCO objects (person/vehicle + portable parcels) each frame and track
them by IoU/centroid. A track is BORN the frame it first appears. A birth is CONJURATION when its
coordinate is in the CENTRAL band and it did not touch a left/right edge (i.e. it did not slide in
from the side) - the object materialised in the middle of the scene. Objects present in the real
context (born at frame 0) and objects that enter from a side edge never count. Warps are ignored
(not detected as objects). Rollout score = number of central-birth events (>=1 => FLAG)."""
import os
import numpy as np, torch, cv2
import fleet_common as fc
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

DEV = "cuda"
HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "out", "conjure_probe"); os.makedirs(OUTD, exist_ok=True)
KEEP = {1: "person", 2: "bike", 3: "car", 4: "moto", 6: "bus", 8: "truck",
        27: "backpack", 31: "handbag", 33: "suitcase"}
SCORE = 0.55
CX = (0.20, 0.80)          # birth center-x in the central band
EDGE = 0.12                # birth box must stay out of the outer 12% each side (not near an edge)
MIN_AREA = 0.008           # birth box >= this fraction of frame
STEP_S = float(os.environ.get("STEP_S","0.15"))
MATCH_CDIST = 0.14         # centroid match radius (fraction of diagonal)
LOOKBACK = 8               # frames of occupancy memory before a birth (~1.2s at 0.15s)
OCC_R = 0.13               # a birth spot counts as "occupied" if a detection was this close before

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
        return [(b, int(l)) for b, l, s in zip(o["boxes"].cpu().numpy(), o["labels"].cpu().numpy(),
                o["scores"].cpu().numpy()) if s >= SCORE and int(l) in KEEP]

    for scene, mdl in PROBE:
        n, fps = fc.meta(scene, mdl); ctx = fc.ctx_of(mdl)
        end = min(n - 1, ctx + int(round(6.0 * fps)))
        step = max(1, int(round(STEP_S * fps)))
        idx = list(range(max(0, ctx - 1), end + 1, step))
        frames = fc.frames_at(scene, mdl, idx)
        H, W = frames[0].shape[:2]; diag = (W**2 + H**2) ** 0.5
        tracks = []
        hist = []           # per-frame list of detection centers (px) - occupancy memory
        for k, fr in enumerate(frames):
            dets = detect(fr); used = [False] * len(dets)
            hist.append([((b[0]+b[2])/2, (b[1]+b[3])/2) for b, l in dets])
            for t in tracks:
                if k - t["last"] > 2:
                    continue
                tcx, tcy = (t["box"][0]+t["box"][2])/2, (t["box"][1]+t["box"][3])/2
                best, bi = 0.0, -1
                for di, (b, l) in enumerate(dets):
                    if used[di] or l != t["cls"]:
                        continue
                    bcx, bcy = (b[0]+b[2])/2, (b[1]+b[3])/2
                    cdist = ((tcx-bcx)**2 + (tcy-bcy)**2) ** 0.5 / diag
                    sco = max(iou(t["box"], b) / 0.3, (MATCH_CDIST - cdist) / MATCH_CDIST)
                    if sco > 1.0 and sco > best:
                        best, bi = sco, di
                if bi >= 0:
                    used[bi] = True; t["box"] = dets[bi][0]; t["last"] = k
            for di, (b, l) in enumerate(dets):
                if used[di]:
                    continue
                cx = (b[0]+b[2])/2/W; area = (b[2]-b[0])*(b[3]-b[1])/(W*H)
                central = (CX[0] < cx < CX[1] and b[0] > EDGE*W and b[2] < (1-EDGE)*W and area >= MIN_AREA)
                tracks.append({"box": b, "cls": l, "birth_k": k, "last": k, "cen": ((b[0]+b[2])/2, (b[1]+b[3])/2),
                               "birth_central": central, "birth_box": b})

        def spot_was_empty(t):
            """True if no detection was near the birth location in the LOOKBACK frames before birth
            (a fresh spawn into empty space), vs a spot already occupied by a pre-existing/crowd object."""
            cx0, cy0 = t["cen"]; kb = t["birth_k"]
            for kk in range(max(0, kb - LOOKBACK), kb):
                for (hx, hy) in hist[kk]:
                    if ((hx - cx0)**2 + (hy - cy0)**2) ** 0.5 / diag < OCC_R:
                        return False
            return True

        events = [t for t in tracks if t["birth_k"] > 0 and t["birth_central"]
                  and t["last"] - t["birth_k"] >= 1 and spot_was_empty(t)]
        tag = f"{mdl}_{scene}"
        print(f"{tag:20s} events={len(events)}  -> {'FLAG' if events else 'clean'}"
              + ("  " + "; ".join(f"{KEEP[e['cls']]}@k{e['birth_k']}" for e in events[:4]) if events else ""))
        panels = []
        for e in events[:4]:
            fr = frames[e["birth_k"]].copy(); b = e["birth_box"].astype(int)
            cv2.rectangle(fr, (b[0], b[1]), (b[2], b[3]), (255, 60, 60), 3)
            cv2.putText(fr, f"{KEEP[e['cls']]} born k{e['birth_k']}", (b[0], max(20, b[1]-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 60, 60), 2)
            panels.append(cv2.cvtColor(cv2.resize(fr, (360, int(360*fr.shape[0]/fr.shape[1]))), cv2.COLOR_RGB2BGR))
        if not panels:
            fr = frames[len(frames)//2]
            panels.append(cv2.cvtColor(cv2.resize(fr, (360, int(360*fr.shape[0]/fr.shape[1]))), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(OUTD, f"{tag}.png"), np.concatenate(panels, 1))
    print("viz ->", OUTD)


if __name__ == "__main__":
    main()
