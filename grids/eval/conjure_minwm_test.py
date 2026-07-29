"""Run the RT-DETR conjuration probe on the 32 standalone minwm_rXX.mp4 in
/home/ashish/stationary_evaluation and compare the flagged set to ground truth.
GT conjured = {2,6,8,11,15,21,23,28,29,31}. Tunable thresholds via env."""
import os, sys
import numpy as np, torch, cv2, imageio.v3 as iio
from transformers import AutoModelForObjectDetection, AutoImageProcessor

DEV = "cuda"; MODEL_ID = "PekingU/rtdetr_r50vd"
DIR = "/home/ashish/stationary_evaluation"
CTX = 13; FPS = 16.0
KEEPNAMES = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "backpack", "handbag", "suitcase"}
SCORE = float(os.environ.get("SCORE", "0.80"))
CXLO, CXHI = [float(x) for x in os.environ.get("CX", "0.20,0.80").split(",")]
EDGE = float(os.environ.get("EDGE", "0.12"))
MIN_AREA = float(os.environ.get("MIN_AREA", "0.008"))
STEP_S = float(os.environ.get("STEP_S", "0.15"))
LOOKBACK = int(os.environ.get("LOOKBACK", "8")); OCC_R = float(os.environ.get("OCC_R", "0.13"))
MATCH_CDIST = 0.14
GT = {2, 6, 8, 11, 15, 21, 23, 28, 29, 31}
SAVE = "--save" in sys.argv


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
    def detect(frame, thr=SCORE):
        H, W = frame.shape[:2]
        inp = proc(images=frame, return_tensors="pt").to(DEV); out = model(**inp)
        res = proc.post_process_object_detection(out, target_sizes=[(H, W)], threshold=thr)[0]
        return [(b, id2label[int(l)]) for b, l, s in zip(res["boxes"].cpu().numpy(), res["labels"].cpu().numpy(),
                res["scores"].cpu().numpy()) if id2label[int(l)] in KEEPNAMES]

    RC_THR = 0.45   # lower threshold for scanning the real context (catch even faint present objects)

    def rc_classes(vid):
        """Classes present in the REAL context frames (0..CTX-1), i.e. objects really in the scene."""
        cls = set()
        for i in range(0, CTX, 2):
            for b, l in detect(vid[i], RC_THR):
                cls.add(l)
        return cls

    def events_for(vid):
        end = min(len(vid)-1, CTX + int(round(6.0*FPS)))
        step = max(1, int(round(STEP_S*FPS))); idx = list(range(max(0, CTX-1), end+1, step))
        frames = [vid[i] for i in idx]; H, W = frames[0].shape[:2]; diag = (W**2+H**2)**0.5
        tracks = []; hist = []
        for k, fr in enumerate(frames):
            ds = detect(fr); used = [False]*len(ds)
            hist.append([((b[0]+b[2])/2, (b[1]+b[3])/2, l) for b, l in ds])   # store class too
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
                area = (t["box"][2]-t["box"][0])*(t["box"][3]-t["box"][1])/(W*H)
                if bi >= 0:
                    t["amax"] = max(t["amax"], area)
            for di, (b, l) in enumerate(ds):
                if used[di]:
                    continue
                cx = (b[0]+b[2])/2/W; cy = (b[1]+b[3])/2/H; area = (b[2]-b[0])*(b[3]-b[1])/(W*H)
                central = (CXLO < cx < CXHI and b[0] > EDGE*W and b[2] < (1-EDGE)*W and area >= MIN_AREA)
                tracks.append({"box": b, "cls": l, "birth_k": k, "last": k, "cen": ((b[0]+b[2])/2, (b[1]+b[3])/2),
                               "central": central, "bb": b, "img_k": idx[k], "a0": area, "amax": area, "cy0": cy})

        def empty(t):
            cx0, cy0 = t["cen"]; kb = t["birth_k"]
            for kk in range(max(0, kb-LOOKBACK), kb):
                for (hx, hy, hl) in hist[kk]:
                    if hl == t["cls"] and ((hx-cx0)**2+(hy-cy0)**2)**0.5/diag < OCC_R:  # same-class only
                        return False
            return True
        out = []
        for t in tracks:
            cx = t["cen"][0]/W
            if t["birth_k"] > 0 and t["last"]-t["birth_k"] >= 1 and empty(t) and 0.25 < cx < 0.75 and t["a0"] >= 0.0008:
                out.append(dict(cls=t["cls"], cx=round(cx, 2), cy=round(t["cy0"], 2), a0=t["a0"], amax=t["amax"],
                                grow=t["amax"]/max(t["a0"], 1e-5), pers=t["last"]-t["birth_k"], img=vid[t["img_k"]], box=t["bb"]))
        return out

    print(f"{'vid':4s}{'GT':4s} {'cls':8s}{'cx':>5s}{'amax':>7s}{'new?':>6s}   real-context-classes")
    allcand = {}
    for r in range(32):
        vid = iio.imread(f"{DIR}/minwm_r{r:02d}.mp4", plugin="pyav")
        rc = rc_classes(vid)
        ev = events_for(vid)
        allcand[r] = (ev, rc)
        g = "GT+" if r in GT else "  -"
        if not ev:
            print(f"r{r:02d} {g}  (no candidate)                          rc={sorted(rc)}")
        for e in ev:
            new = e["cls"] not in rc
            print(f"r{r:02d} {g}  {e['cls']:8s}{e['cx']:5.2f}{e['amax']:7.3f}{('NEW' if new else 'in-ctx'):>6s}   rc={sorted(rc)}")
    # final rule: central object whose CLASS is new to the scene (absent from the real context)
    def flag(ev, rc):
        # central crisp object that is EITHER a class new to the scene OR grows strongly (approaches
        # from nothing) - real objects present at the start are same-class AND barely grow.
        return any(0.33 < e["cx"] < 0.67 and e["amax"] >= 0.010 and (e["cls"] not in rc or e["grow"] >= 2.0) for e in ev)
    flagged = {r for r, (ev, rc) in allcand.items() if flag(ev, rc)}
    tp = flagged & GT; fp = flagged - GT; fn = GT - flagged
    print(f"\nflagged={sorted(flagged)}\nGT     ={sorted(GT)}")
    print(f"TP={sorted(tp)} ({len(tp)})  FP={sorted(fp)}  FN={sorted(fn)}")
    print("EXACT MATCH" if flagged == GT else f"NOT exact: {len(fp)} FP, {len(fn)} FN")
    if SAVE and viz:
        W = max(v.shape[1] for v in viz); viz = [cv2.copyMakeBorder(v, 0, 0, 0, W-v.shape[1], cv2.BORDER_CONSTANT, value=(25, 25, 25)) for v in viz]
        rows = [np.concatenate(viz[i:i+5] + [np.full((viz[0].shape[0], W, 3), 25, np.uint8)]*(5-len(viz[i:i+5])), 1) for i in range(0, len(viz), 5)]
        cv2.imwrite("out/conjure_minwm_test.png", np.concatenate(rows, 0)); print("wrote out/conjure_minwm_test.png")


if __name__ == "__main__":
    main()
