"""What distinguishes the two TRUE conjurations (minwm_r06_F, minwm_r11_BR) from the 18
real-but-revealed objects? Re-run RT-DETR on the videos behind the 20 finds and, for each
central new-object birth, measure candidate features:
  conf     - RT-DETR confidence at birth
  cx       - horizontal location (0..1)
  sudden   - local pixel change at birth (box region abs-diff birth vs birth-1) / whole-frame diff
             (>1 = the object region changed much more than the moving background = popped in)
  precur   - strongest LOW-threshold (0.25) same-class detection near the birth spot in the 3
             frames BEFORE birth (0 = no precursor at all = truly sudden; >0 = it was easing in)
  grow     - area a few frames after birth / area at birth (>1 = approaching/growing object)
  persist  - frames the track survived
Rows for the two good ones are marked <<< GOOD."""
import os
import numpy as np, torch, cv2
import fleet_common as fc
from transformers import AutoModelForObjectDetection, AutoImageProcessor

DEV = "cuda"; MODEL_ID = "PekingU/rtdetr_r50vd"
KEEPNAMES = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "backpack", "handbag", "suitcase"}
HI = 0.80; LO = 0.25; CX = (0.20, 0.80); EDGE = 0.12; MIN_AREA = 0.008
STEP_S = 0.15; MATCH_CDIST = 0.14; LOOKBACK = 8; OCC_R = 0.13
GOOD = {("r06_F", "minwm"), ("r11_BR", "minwm")}
VIDEOS = [("r04_B", "worldplay"), ("r02_BR", "ours_pca8"), ("r17_L", "yume"), ("r07_R", "minwm"),
          ("r22_R", "astra"), ("r01_FR", "yume"), ("r02_FR", "yume"), ("r23_B", "minwm"),
          ("r22_FL", "ours_noatok"), ("r03_BL", "ours_pca8"), ("r22_B", "matrixgame"), ("r30_FL", "minwm"),
          ("r25_R", "worldcam"), ("r30_R", "worldplay"), ("r11_BR", "minwm"), ("r22_BL", "astra"),
          ("r06_F", "minwm"), ("r22_F", "minwm")]


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
        return [(b, id2label[int(l)], float(s)) for b, l, s in zip(res["boxes"].cpu().numpy(),
                res["labels"].cpu().numpy(), res["scores"].cpu().numpy()) if id2label[int(l)] in KEEPNAMES]

    print(f"{'video':22s}{'cls':9s}{'conf':>5s}{'cx':>5s}{'sudden':>7s}{'precur':>7s}{'grow':>6s}{'pers':>5s}")
    for scene, mdl in VIDEOS:
        n, fps = fc.meta(scene, mdl); ctx = fc.ctx_of(mdl); end = min(n-1, ctx+int(round(6*fps)))
        step = max(1, int(round(STEP_S*fps))); idx = list(range(max(0, ctx-1), end+1, step))
        frames = fc.frames_at(scene, mdl, idx); H, W = frames[0].shape[:2]; diag = (W**2+H**2)**0.5
        hi_d = [detect(f, HI) for f in frames]           # birth detector (0.8)
        lo_d = [detect(f, LO) for f in frames]           # precursor detector (0.25)
        grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32) for f in frames]
        tracks = []; hist = []
        for k in range(len(frames)):
            ds = [(b, l) for b, l, s in hi_d[k]]; used = [False]*len(ds)
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
                    used[bi] = True; t["box"] = ds[bi][0]; t["last"] = k; t["area_seq"].append((k, (ds[bi][0][2]-ds[bi][0][0])*(ds[bi][0][3]-ds[bi][0][1])/(W*H)))
            for di, (b, l) in enumerate(ds):
                if used[di]:
                    continue
                cx = (b[0]+b[2])/2/W; area = (b[2]-b[0])*(b[3]-b[1])/(W*H)
                central = (CX[0] < cx < CX[1] and b[0] > EDGE*W and b[2] < (1-EDGE)*W and area >= MIN_AREA)
                tracks.append({"box": b, "cls": l, "birth_k": k, "last": k, "cen": ((b[0]+b[2])/2, (b[1]+b[3])/2),
                               "central": central, "bb": b, "area_seq": [(k, area)]})

        def empty(t):
            cx0, cy0 = t["cen"]; kb = t["birth_k"]
            for kk in range(max(0, kb-LOOKBACK), kb):
                for (hx, hy) in hist[kk]:
                    if ((hx-cx0)**2+(hy-cy0)**2)**0.5/diag < OCC_R:
                        return False
            return True

        for t in tracks:
            if not (t["birth_k"] > 0 and t["central"] and t["last"]-t["birth_k"] >= 1 and empty(t)):
                continue
            kb = t["birth_k"]; b = t["bb"].astype(int); cx0, cy0 = t["cen"]
            # sudden: box-region abs diff (birth vs prev) / whole-frame abs diff
            dbox = np.abs(grays[kb][max(0, b[1]):b[3], max(0, b[0]):b[2]] - grays[kb-1][max(0, b[1]):b[3], max(0, b[0]):b[2]])
            dfull = np.abs(grays[kb] - grays[kb-1])
            sudden = (dbox.mean() / max(dfull.mean(), 1e-3)) if dbox.size else 0.0
            # precursor: strongest LOW-thresh same-class detection near birth loc in 3 frames before
            precur = 0.0
            for kk in range(max(0, kb-3), kb):
                for bb, ll, ss in lo_d[kk]:
                    if ll == t["cls"]:
                        pcx, pcy = (bb[0]+bb[2])/2, (bb[1]+bb[3])/2
                        if ((pcx-cx0)**2+(pcy-cy0)**2)**0.5/diag < 0.18:
                            precur = max(precur, ss)
            a0 = t["area_seq"][0][1]; alater = t["area_seq"][min(2, len(t["area_seq"])-1)][1]
            grow = alater / max(a0, 1e-4)
            mark = "  <<< GOOD" if (scene, mdl) in GOOD else ""
            conf = max((s for bb, ll, s in hi_d[kb] if ll == t["cls"]), default=0.0)
            print(f"{mdl+'_'+scene:22s}{t['cls']:9s}{conf:5.2f}{cx0/W:5.2f}{sudden:7.2f}{precur:7.2f}{grow:6.2f}{t['last']-t['birth_k']:5d}{mark}")


if __name__ == "__main__":
    main()
