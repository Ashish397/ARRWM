"""Three-detector bake-off for the conjuration probe. Same random fleet videos for all three
detectors (Faster R-CNN two-stage / RetinaNet one-stage / FCOS anchor-free, all COCO). Process
random rollouts until EACH detector has accumulated >=20 conjuration events (central new-object
births into previously-empty space), then write one review montage per detector so we can compare
which detector localises conjured objects best. CPU-light detectors on GPU."""
import os
import numpy as np, torch, cv2
import fleet_common as fc
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,
    retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights,
    fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights)

DEV = "cuda"
HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "out"); os.makedirs(OUTD, exist_ok=True)
KEEP = {1: "person", 2: "bike", 3: "car", 4: "moto", 6: "bus", 8: "truck",
        27: "backpack", 31: "handbag", 33: "suitcase"}
SCORE = 0.55
CX = (0.20, 0.80); EDGE = 0.12; MIN_AREA = 0.008
STEP_S = 0.15; MATCH_CDIST = 0.14; LOOKBACK = 8; OCC_R = 0.13
TARGET = 20
MAX_VIDEOS = 90
SEED = 3


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1]); x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, x1 - x0), max(0, y1 - y0); inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_detectors():
    specs = [
        ("frcnn", fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1),
        ("retinanet", retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1),
        ("fcos", fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights.COCO_V1)]
    dets = {}
    for name, ctor, weights in specs:
        model = ctor(weights=weights).eval().to(DEV)
        tf = weights.transforms()

        @torch.no_grad()
        def detect(frame, _m=model, _tf=tf):
            x = _tf(torch.from_numpy(frame).permute(2, 0, 1)).to(DEV)
            o = _m([x])[0]
            return [(b, int(l)) for b, l, s in zip(o["boxes"].cpu().numpy(), o["labels"].cpu().numpy(),
                    o["scores"].cpu().numpy()) if s >= SCORE and int(l) in KEEP]
        dets[name] = detect
    return dets


def events_for(detect, scene, mdl):
    n, fps = fc.meta(scene, mdl); ctx = fc.ctx_of(mdl)
    end = min(n - 1, ctx + int(round(6.0 * fps)))
    step = max(1, int(round(STEP_S * fps)))
    idx = list(range(max(0, ctx - 1), end + 1, step))
    frames = fc.frames_at(scene, mdl, idx)
    H, W = frames[0].shape[:2]; diag = (W**2 + H**2) ** 0.5
    tracks = []; hist = []
    for k, fr in enumerate(frames):
        ds = detect(fr); used = [False] * len(ds)
        hist.append([((b[0]+b[2])/2, (b[1]+b[3])/2) for b, l in ds])
        for t in tracks:
            if k - t["last"] > 2:
                continue
            tcx, tcy = (t["box"][0]+t["box"][2])/2, (t["box"][1]+t["box"][3])/2
            best, bi = 0.0, -1
            for di, (b, l) in enumerate(ds):
                if used[di] or l != t["cls"]:
                    continue
                bcx, bcy = (b[0]+b[2])/2, (b[1]+b[3])/2
                cd = ((tcx-bcx)**2 + (tcy-bcy)**2) ** 0.5 / diag
                sco = max(iou(t["box"], b) / 0.3, (MATCH_CDIST - cd) / MATCH_CDIST)
                if sco > 1.0 and sco > best:
                    best, bi = sco, di
            if bi >= 0:
                used[bi] = True; t["box"] = ds[bi][0]; t["last"] = k
        for di, (b, l) in enumerate(ds):
            if used[di]:
                continue
            cx = (b[0]+b[2])/2/W; area = (b[2]-b[0])*(b[3]-b[1])/(W*H)
            central = (CX[0] < cx < CX[1] and b[0] > EDGE*W and b[2] < (1-EDGE)*W and area >= MIN_AREA)
            tracks.append({"box": b, "cls": l, "birth_k": k, "last": k,
                           "cen": ((b[0]+b[2])/2, (b[1]+b[3])/2), "central": central, "bb": b})

    def empty(t):
        cx0, cy0 = t["cen"]; kb = t["birth_k"]
        for kk in range(max(0, kb - LOOKBACK), kb):
            for (hx, hy) in hist[kk]:
                if ((hx - cx0)**2 + (hy - cy0)**2) ** 0.5 / diag < OCC_R:
                    return False
        return True

    out = []
    for t in tracks:
        if t["birth_k"] > 0 and t["central"] and t["last"] - t["birth_k"] >= 1 and empty(t):
            out.append(dict(scene=scene, mdl=mdl, cls=t["cls"], img=frames[t["birth_k"]].copy(), box=t["bb"].astype(int)))
    return out


def montage(events, path):
    cells = []
    for e in events[:TARGET]:
        fr = e["img"].copy(); b = e["box"]
        cv2.rectangle(fr, (b[0], b[1]), (b[2], b[3]), (255, 60, 60), 3)
        im = cv2.cvtColor(cv2.resize(fr, (300, int(300*fr.shape[0]/fr.shape[1]))), cv2.COLOR_RGB2BGR)
        h = np.full((18, im.shape[1], 3), 25, np.uint8)
        cv2.putText(h, f"{e['mdl']}_{e['scene']} {KEEP[e['cls']]}", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1)
        cells.append(np.concatenate([h, im], 0))
    if not cells:
        return
    ch = max(c.shape[0] for c in cells); cw = max(c.shape[1] for c in cells)
    cells = [cv2.copyMakeBorder(c, 0, ch-c.shape[0], 0, cw-c.shape[1], cv2.BORDER_CONSTANT, value=(25, 25, 25)) for c in cells]
    cols = 5
    rows = [np.concatenate(cells[i:i+cols] + [np.full((ch, cw, 3), 25, np.uint8)]*((cols-len(cells[i:i+cols]))), 1)
            for i in range(0, len(cells), cols)]
    cv2.imwrite(path, np.concatenate(rows, 0)); print("wrote", path, f"({len(cells)} events)")


def main():
    dets = load_detectors()
    idx = fc.fleet_index()
    rng = np.random.RandomState(SEED); order = list(idx); rng.shuffle(order)
    acc = {name: [] for name in dets}
    used_videos = []
    for vi, (scene, mdl) in enumerate(order):
        if vi >= MAX_VIDEOS or all(len(acc[n]) >= TARGET for n in dets):
            break
        try:
            for name, detect in dets.items():
                acc[name].extend(events_for(detect, scene, mdl))
        except Exception as e:
            print("skip", mdl, scene, str(e)[:50]); continue
        used_videos.append(f"{mdl}_{scene}")
        if (vi + 1) % 5 == 0:
            print(f"[{vi+1} videos] " + "  ".join(f"{n}={len(acc[n])}" for n in dets), flush=True)
    print("counts:", {n: len(acc[n]) for n in dets}, "over", len(used_videos), "videos")
    for name in dets:
        montage(acc[name], os.path.join(OUTD, f"conjure_review_{name}.png"))


if __name__ == "__main__":
    main()
