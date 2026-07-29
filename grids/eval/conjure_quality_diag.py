"""Diagnostic: for the videos behind the current 20 finds, report each conjuration birth's
DETECTION CONFIDENCE and PATCH SHARPNESS (Laplacian variance of the box, and its ratio to the
whole-frame Laplacian variance). Real conjured objects (crisp cars/rovers) should show high
confidence + high sharpness; melt-blob false positives should be low on both. Prints a table so
we can pick the quality gate."""
import os
import numpy as np, torch, cv2
import fleet_common as fc
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

DEV = "cuda"
KEEP = {1: "person", 2: "bike", 3: "car", 4: "moto", 6: "bus", 8: "truck", 27: "backpack", 31: "handbag", 33: "suitcase"}
SCORE = 0.55; CX = (0.20, 0.80); EDGE = 0.12; MIN_AREA = 0.008
STEP_S = 0.15; MATCH_CDIST = 0.14; LOOKBACK = 8; OCC_R = 0.13

VIDEOS = [("r15_FL", "worldcam"), ("r21_FR", "ours_noatok"), ("r06_FR", "ours_pca8"), ("r01_R", "worldplay"),
          ("r14_F", "yume"), ("r02_FR", "worldcam"), ("r03_FR", "ours_noadaln"), ("r21_L", "ours_pca8"),
          ("r20_R", "yume"), ("r04_L", "ours_16node"), ("r17_L", "yume"), ("r21_FL", "worldplay"),
          ("r26_F", "ours_pca4"), ("r07_R", "minwm"), ("r01_FR", "yume")]


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1]); x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, x1-x0), max(0, y1-y0); inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


def main():
    w = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn_v2(weights=w).eval().to(DEV); tf = w.transforms()

    @torch.no_grad()
    def detect(frame):
        x = tf(torch.from_numpy(frame).permute(2, 0, 1)).to(DEV); o = model([x])[0]
        return [(b, int(l), float(s)) for b, l, s in zip(o["boxes"].cpu().numpy(), o["labels"].cpu().numpy(),
                o["scores"].cpu().numpy()) if s >= SCORE and int(l) in KEEP]

    @torch.no_grad()
    def reconfirm(frame, box):
        """Crop the box (padded), upsample, re-detect. Real objects survive isolation; melt does not.
        Returns (best same-or-any KEEP conf on the crop, its area fraction of the crop)."""
        H, W = frame.shape[:2]
        bw, bh = box[2]-box[0], box[3]-box[1]
        px, py = int(0.3*bw), int(0.3*bh)
        x0, y0 = max(0, box[0]-px), max(0, box[1]-py); x1, y1 = min(W, box[2]+px), min(H, box[3]+py)
        crop = frame[y0:y1, x0:x1]
        if crop.shape[0] < 8 or crop.shape[1] < 8:
            return 0.0, 0.0
        scale = 512 / max(crop.shape[0], crop.shape[1])
        crop = cv2.resize(crop, (int(crop.shape[1]*scale), int(crop.shape[0]*scale)))
        x = tf(torch.from_numpy(crop).permute(2, 0, 1)).to(DEV); o = model([x])[0]
        ch, cw = crop.shape[:2]; best, frac = 0.0, 0.0
        for b, l, s in zip(o["boxes"].cpu().numpy(), o["labels"].cpu().numpy(), o["scores"].cpu().numpy()):
            if int(l) in KEEP and s > best:
                best = float(s); frac = (b[2]-b[0])*(b[3]-b[1])/(cw*ch)
        return best, frac

    print(f"{'video':24s} {'cls':9s} {'conf':>5s} {'sharp':>7s} {'recon':>6s} {'rfrac':>6s}")
    for scene, mdl in VIDEOS:
        n, fps = fc.meta(scene, mdl); ctx = fc.ctx_of(mdl); end = min(n-1, ctx+int(round(6*fps)))
        step = max(1, int(round(STEP_S*fps))); idx = list(range(max(0, ctx-1), end+1, step))
        frames = fc.frames_at(scene, mdl, idx); H, W = frames[0].shape[:2]; diag = (W**2+H**2)**0.5
        tracks = []; hist = []
        for k, fr in enumerate(frames):
            ds = detect(fr); used = [False]*len(ds)
            hist.append([((b[0]+b[2])/2, (b[1]+b[3])/2) for b, l, s in ds])
            for t in tracks:
                if k - t["last"] > 2:
                    continue
                tcx, tcy = (t["box"][0]+t["box"][2])/2, (t["box"][1]+t["box"][3])/2; best, bi = 0.0, -1
                for di, (b, l, s) in enumerate(ds):
                    if used[di] or l != t["cls"]:
                        continue
                    bcx, bcy = (b[0]+b[2])/2, (b[1]+b[3])/2; cd = ((tcx-bcx)**2+(tcy-bcy)**2)**0.5/diag
                    sco = max(iou(t["box"], b)/0.3, (MATCH_CDIST-cd)/MATCH_CDIST)
                    if sco > 1.0 and sco > best:
                        best, bi = sco, di
                if bi >= 0:
                    used[bi] = True; t["box"] = ds[bi][0]; t["last"] = k
            for di, (b, l, s) in enumerate(ds):
                if used[di]:
                    continue
                cx = (b[0]+b[2])/2/W; area = (b[2]-b[0])*(b[3]-b[1])/(W*H)
                central = (CX[0] < cx < CX[1] and b[0] > EDGE*W and b[2] < (1-EDGE)*W and area >= MIN_AREA)
                tracks.append({"box": b, "cls": l, "birth_k": k, "last": k, "conf": s,
                               "cen": ((b[0]+b[2])/2, (b[1]+b[3])/2), "central": central})

        def empty(t):
            cx0, cy0 = t["cen"]; kb = t["birth_k"]
            for kk in range(max(0, kb-LOOKBACK), kb):
                for (hx, hy) in hist[kk]:
                    if ((hx-cx0)**2+(hy-cy0)**2)**0.5/diag < OCC_R:
                        return False
            return True
        for t in tracks:
            if t["birth_k"] > 0 and t["central"] and t["last"]-t["birth_k"] >= 1 and empty(t):
                b = t["box"].astype(int); fr = frames[t["birth_k"]]
                g = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY).astype(np.float64)
                patch = g[max(0, b[1]):b[3], max(0, b[0]):b[2]]
                sharp = cv2.Laplacian(patch, cv2.CV_64F).var() if patch.size > 20 else 0.0
                rc, rf = reconfirm(fr, b)
                cxf = (b[0]+b[2])/2/W; x0f, x1f = b[0]/W, b[2]/W
                print(f"{mdl+'_'+scene:24s} {KEEP[t['cls']]:9s} {t['conf']:5.2f} {sharp:7.1f} {rc:6.2f} {rf:6.2f}"
                      f"   cx={cxf:.2f} x0={x0f:.2f} x1={x1f:.2f}")


if __name__ == "__main__":
    main()
