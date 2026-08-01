"""Pop-in detector: finds objects that were not in the scene and then materialise in it.

Every decision here is temporal. No frame is ever judged on its own -- the question asked of
each object is always "what was here before?", and the answer comes from comparing frames.

Stage 1 (popin_cache.py) runs an object detector densely over every frame. This stage links
those detections into tracks and interrogates each track. The core problem is that "a new
object appeared" is not the same as "an object popped into existence": in driving footage new
objects appear constantly and legitimately. They drive in from the side of the frame, they
approach from the distance until they cross the detector's resolution limit, and they emerge
from behind occluders. All three are ruled out explicitly.

CANDIDATE GATES (cheap, from the tracks alone)
  born during generation   birth frame >= the context boundary, so the object is not part of
                           the real footage that was handed to the model.
  not an edge entry        the first appearance is well inside the frame and the track does not
                           trace back to a left/right border in its first half second. Things
                           that drive in from the side are new but did not pop in.
  persistent and crisp     reaches high detector confidence and is still alive at the end of the
                           clip. Conjured objects stay; detector flicker and haze do not.

Two branches then decide, because pop-in has two distinct signatures in practice.

BRANCH A -- MATERIALISATION (an object appears on ground that was empty)
  onset      the local pixel change at the birth box, divided by the change over the whole frame
             in the same frame pair. The denominator is the point: ego-motion makes every pixel
             change, so only a *relative* spike means something arrived here.
  zoom probe the decisive test. Crop where the object is about to be, in the frames before it
             exists, upscale, and re-run the detector. This asks "if I look closely, is it
             already there?" and it is what keeps distant real cars from being flagged: they
             look like nothing at native resolution but resolve into a confident vehicle when
             zoomed. A conjuration's ground stays empty under any magnification.
  back-match the object's own patch, matched into the pre-birth frames at the box predicted by
             extrapolating its own trajectory backwards. Catches partial occlusion reveals,
             where the object was physically present but only half visible.

BRANCH B -- CRYSTALLISATION (an ambiguous blob is upgraded into a definite object)
  A blob can already be present and still count as a pop-in if the model turns it into a
  fully-formed vehicle. The trap is that real distant objects also go from ambiguous to
  confident -- but they do so *because they got closer*. So branch B fires only when the
  confidence gain is not explained by approach: the object was ambiguous before (low prior
  confidence, not merely small), it is confidently identified after, and it barely grew across
  the transition. An object that becomes recognisable without getting any nearer is anomalous.

The detector's own low-confidence tail is deliberately not used as evidence of presence: it
emits persistent 0.3-0.5 "car" boxes on empty background clutter, which would veto true
pop-ins. Presence is established by the zoom probe and by pixel matching instead.

Every criterion is a normalised margin and a track's score is the weakest of them, so output is
rankable rather than a bare boolean. A clip scores as the max over its tracks.

Usage:
  python popin_cache.py                  # stage 1, once per video set
  python popin_detect.py                 # score every clip, print the ranking
  python popin_detect.py --save          # also write annotated evidence panels
  python popin_detect.py --tag X --dir D --ctx N     # another video set
"""
import os, sys, json
import numpy as np, cv2, imageio.v3 as iio

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "out")

# Classes tracked for context. Everything here can serve as prior evidence that something was
# already present at a location, whether or not it is the kind of thing that gets conjured.
KEEP = {"car", "truck", "bus", "motorcycle", "bicycle", "person",
        "suitcase", "backpack", "handbag", "bench", "chair", "refrigerator", "tv", "microwave"}

# Classes that can be reported as conjured: rigid man-made objects. Deliberately excludes
# people -- pedestrians routinely step out from behind bushes, poles and parked cars, which is
# an occlusion reveal rather than a pop-in, and the phenomenon being detected is vehicles and
# cargo. Widen this set if the target domain conjures other things.
CONJURABLE = {"car", "truck", "bus", "motorcycle", "suitcase", "refrigerator", "bench"}

# ---- tracker ---------------------------------------------------------------------------
TRK = 0.30          # score to belong to a track (low, so a track survives brief dips)
HI = 0.70           # score a track must reach to count as a real, crisp object
MATCH_IOU = 0.25
MATCH_CD = 0.10     # centroid distance as a fraction of the image diagonal
# (GAP and the pixel-evidence windows are defined with the fps rescaling below.)

# ---- candidate gates -------------------------------------------------------------------
CTX = 13            # frames 0..CTX-1 are real footage; generation starts at frame CTX
EDGE = 0.06         # birth box must clear the left/right border by this fraction of width
CXLO, CXHI = 0.15, 0.85
MIN_AREA = 0.0015   # birth box at least this fraction of the frame (not a horizon speck)
PERSIST_TAIL = 0.75 # track must still be alive this far through the clip
PERSIST_FRAC = 0.55 # ...and present in this fraction of the frames since birth

# ---- branch A: materialisation ---------------------------------------------------------
ONSET_T = 5.0       # local/global pixel-change ratio at birth
ZPROBE_T = 0.50     # confidence of a zoomed re-detection before birth, above which it was there
BNCC_T = 0.65       # back-matched correlation, above which the object has a visual history

# ---- branch B: crystallisation ---------------------------------------------------------
# Disabled. Branch B fires when an already-present blob is upgraded into a confident object,
# which is a real phenomenon but is not separable from an occlusion reveal with the evidence
# available here. Measured fleet-wide it produced 9 of the 11 flags on source scene r30 -- an
# event the real-footage control shows is a false positive -- and its median score (0.15) is
# less than half branch A's (0.36). Set True to restore it; the code below is unchanged.
USE_BRANCH_B = False
SPRE_T = 0.50       # prior confidence below which the blob counts as semantically ambiguous
PEAK_B = 0.85       # confidence the object must reach afterwards
GROW_B = 1.60       # size ratio across the transition, above which approach explains it
ONSET_B = 3.0       # a milder onset requirement than branch A

# ---- pixel evidence windows ------------------------------------------------------------
# All frame counts below are quoted at 16 fps and rescaled by set_fps() for other sources, so
# every window means a fixed duration rather than a fixed number of frames. Without this the
# detector would look back twice as far in time on a 30 fps clip as on a 16 fps one.
BASE_FPS = 16.0
BACK_W = 14         # frames of history examined (~0.9 s)
BACK_MIN = 3        # skip the frames straddling birth, where the object is half formed
SETTLE = 3          # frames after birth at which the object's own template is taken
GROW_W = 12         # half-window for the branch-B size comparison
EDGE_W = 8          # frames over which an edge entry must reveal itself
ONSET_W = 3         # frames over which the birth transition is measured
GAP = 3             # frames a track may go unmatched before it is closed

_S = 1.0            # current fps rescale factor


def set_fps(fps):
    """Rescale every temporal window so the detector means the same thing at any frame rate."""
    global _S
    _S = float(fps) / BASE_FPS


def W_(frames):
    """A window quoted at 16 fps, in frames at the current frame rate."""
    return max(1, int(round(frames * _S)))

# Validation only -- the detector never reads this. minwm_rXX clips known to be conjured.
GT = {2, 6, 8, 11, 15, 21, 23, 28, 29, 31}

_MODEL = None


def set_detector(crop_fn):
    """Inject the backend the zoom probe should re-detect with. Must match the backbone that
    produced the cached detections, or prior-evidence confidences are not comparable."""
    global _MODEL
    _MODEL = crop_fn


def _detector():
    """RT-DETR by default, loaded lazily -- only the zoom probe needs it."""
    global _MODEL
    if _MODEL is None:
        import torch
        from transformers import AutoModelForObjectDetection, AutoImageProcessor
        proc = AutoImageProcessor.from_pretrained("PekingU/rtdetr_r50vd", use_fast=True)
        mdl = AutoModelForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd").eval().cuda()

        @torch.no_grad()
        def run(img, thr=0.10):
            inp = proc(images=img, return_tensors="pt").to("cuda")
            out = mdl(**inp)
            r = proc.post_process_object_detection(out, target_sizes=[img.shape[:2]], threshold=thr)[0]
            return [(b, mdl.config.id2label[int(l)], float(s)) for b, l, s in
                    zip(r["boxes"].cpu().numpy(), r["labels"].cpu().numpy(), r["scores"].cpu().numpy())]
        _MODEL = run
    return _MODEL


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def cen(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def clipbox(box, W, H):
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    return max(0, x0), max(0, y0), min(W, x1), min(H, y1)


def track(dets, W, H):
    """Link per-frame detections into tracks. Linking at full frame rate (not subsampled) is
    what keeps a fast-growing near object attached to its own history."""
    diag = (W * W + H * H) ** 0.5
    tracks, live = [], []
    for k, frame in enumerate(dets):
        ds = [d for d in frame if d["cls"] in KEEP and d["score"] >= TRK]
        used = [False] * len(ds)
        for t in live:
            tc = cen(t["box"])
            best, bi = 0.0, -1
            for i, d in enumerate(ds):
                if used[i] or d["cls"] != t["cls"]:
                    continue
                dc = cen(d["box"])
                cd = ((tc[0] - dc[0]) ** 2 + (tc[1] - dc[1]) ** 2) ** 0.5 / diag
                s = max(iou(t["box"], d["box"]) / MATCH_IOU, (MATCH_CD - cd) / MATCH_CD)
                if s > 1.0 and s > best:
                    best, bi = s, i
            if bi >= 0:
                used[bi] = True
                t["box"], t["last"] = ds[bi]["box"], k
                t["peak"] = max(t["peak"], ds[bi]["score"])
                t["hits"] += 1
                t["seen"].append((k, ds[bi]["score"], ds[bi]["box"]))
        for i, d in enumerate(ds):
            if used[i]:
                continue
            live.append({"cls": d["cls"], "box": d["box"], "birth": k, "last": k, "hits": 1,
                         "peak": d["score"], "bbox": d["box"], "seen": [(k, d["score"], d["box"])]})
        tracks.extend([t for t in live if k - t["last"] > W_(GAP)])
        live = [t for t in live if k - t["last"] <= W_(GAP)]
    tracks.extend(live)
    return tracks


def back_box(t, nfit=12):
    """Extrapolate the track's own box backwards in time by a linear fit to its first frames.
    Used to look for the object's history where it would actually have been, rather than where
    it ended up -- an approaching object is both smaller and elsewhere a second earlier."""
    s = t["seen"][:nfit]
    ks = np.array([k for k, _, _ in s], float)
    B = np.array([b for _, _, b in s], float)
    if len(ks) < 4:
        return lambda k: [float(v) for v in t["bbox"]]
    co = [np.polyfit(ks, B[:, j], 1) for j in range(4)]
    return lambda k: [float(np.polyval(co[j], k)) for j in range(4)]


def edge_trace(t, W):
    """Closest the track ever comes to the left/right border in its first half second.
    Something that drove in from the side is still near a border just after birth."""
    m = 1.0
    for k, s, b in t["seen"]:
        if k > t["birth"] + W_(EDGE_W):
            break
        m = min(m, b[0] / W, (W - b[2]) / W)
    return m


def candidate(t, n, W, H, ctx=CTX):
    """Cheap detector-only gates. Deliberately loose -- the pixel evidence makes the call."""
    b = t["bbox"]
    area = (b[2] - b[0]) * (b[3] - b[1]) / (W * H)
    return (t["cls"] in CONJURABLE and t["birth"] >= ctx and t["peak"] >= HI
            and area >= MIN_AREA and CXLO < cen(b)[0] / W < CXHI and edge_trace(t, W) > EDGE
            and (t["last"] + 1) / n >= PERSIST_TAIL
            and t["hits"] / max(1, n - t["birth"]) >= PERSIST_FRAC)


def onset(vid, t):
    """Local pixel change at the birth box over the birth transition, divided by the same
    quantity over the whole frame. Normalising by the global change is what makes this a
    statement about the object rather than about the camera."""
    b0 = t["birth"]
    if b0 < 1:
        return 0.0
    H, W = vid.shape[1:3]
    x0, y0, x1, y1 = clipbox(t["bbox"], W, H)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return 0.0
    best = 0.0
    for k in range(b0, min(b0 + W_(ONSET_W), len(vid))):
        d = np.abs(cv2.cvtColor(vid[k], cv2.COLOR_RGB2GRAY).astype(np.float32)
                   - cv2.cvtColor(vid[k - 1], cv2.COLOR_RGB2GRAY).astype(np.float32))
        best = max(best, float(d[y0:y1, x0:x1].mean()) / (float(d.mean()) + 1e-3))
    return best


def zoom_probe(vid, t, gbox):
    """Magnify where the object will be, before it exists, and re-run the detector there.

    This is the test that separates a conjuration from a real object approaching from far
    away. At native resolution a distant car is a handful of pixels and goes undetected, which
    looks exactly like absence; magnified, it resolves into a confident vehicle. Empty ground
    stays empty at any magnification. Returns (best confidence, (frame, class, score))."""
    det = _detector()
    H, W = vid.shape[1:3]
    b = t["birth"]
    best, where = 0.0, None
    for p in range(max(0, b - W_(BACK_W)), max(1, b - W_(BACK_MIN) + 1)):
        x0, y0, x1, y1 = gbox(p)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        s = max(x1 - x0, y1 - y0) * 2.5
        cx0, cy0, cx1, cy1 = clipbox([cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2], W, H)
        if cx1 - cx0 < 12 or cy1 - cy0 < 12:
            continue
        crop = vid[p][cy0:cy1, cx0:cx1]
        f = max(1.0, 320 / max(crop.shape[0], crop.shape[1]))
        crop = cv2.resize(crop, (int(crop.shape[1] * f), int(crop.shape[0] * f)), interpolation=cv2.INTER_CUBIC)
        pb = [(x0 - cx0) * f, (y0 - cy0) * f, (x1 - cx0) * f, (y1 - cy0) * f]
        for bb, l, sc in det(crop):
            if l in KEEP and sc > best and iou(bb, pb) > 0.25:
                best, where = sc, (p, l, round(sc, 2))
    return best, where


def back_match(vid, t, gbox):
    """Correlate the object's own settled patch into the frames before its birth, at the box
    its trajectory extrapolates to. Matching at the predicted scale and place -- rather than
    sweeping freely -- is what stops smooth road from correlating with anything."""
    H, W = vid.shape[1:3]
    kk = min(t["birth"] + W_(SETTLE), t["last"])
    tb = next((b for k, s, b in t["seen"] if k == kk), t["bbox"])
    x0, y0, x1, y1 = clipbox(tb, W, H)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0.0
    tpl0 = cv2.cvtColor(vid[kk], cv2.COLOR_RGB2GRAY)[y0:y1, x0:x1]
    best = 0.0
    for p in range(max(0, t["birth"] - W_(BACK_W)), max(1, t["birth"] - W_(BACK_MIN) + 1)):
        px0, py0, px1, py1 = clipbox(gbox(p), W, H)
        pw, ph = px1 - px0, py1 - py0
        if pw < 8 or ph < 8:
            continue
        sx0, sy0, sx1, sy1 = clipbox([px0 - pw * 0.3, py0 - ph * 0.3, px1 + pw * 0.3, py1 + ph * 0.3], W, H)
        prev = cv2.cvtColor(vid[p], cv2.COLOR_RGB2GRAY)[sy0:sy1, sx0:sx1]
        if prev.shape[0] < ph or prev.shape[1] < pw:
            continue
        best = max(best, float(cv2.matchTemplate(prev, cv2.resize(tpl0, (pw, ph)), cv2.TM_CCOEFF_NORMED).max()))
    return best


def blob_at(dets_frame, box):
    """Best detection of any class overlapping a box: (height, confidence). This is how we ask
    whether an ambiguous *something* occupied the spot, independently of what it was called."""
    best, h = 0.0, None
    for d in dets_frame:
        if iou(d["box"], box) > 0.3 and d["score"] > best:
            best, h = d["score"], d["box"][3] - d["box"][1]
    return h, best


def features(vid, dets, t, n, ctx):
    W, H = vid.shape[2], vid.shape[1]
    g = back_box(t)
    b = t["bbox"]
    a0, a1 = max(0, t["birth"] - W_(GROW_W)), min(n - 1, t["birth"] + W_(GROW_W))
    hpre, spre = blob_at(dets[a0], g(a0))
    hpost, _ = blob_at(dets[a1], g(a1))
    zp, zwhere = zoom_probe(vid, t, g)
    return {
        "cls": t["cls"], "birth": t["birth"], "last": t["last"], "peak": round(t["peak"], 2),
        "cx": round(cen(b)[0] / W, 2), "cy": round(cen(b)[1] / H, 2),
        "area": round((b[2] - b[0]) * (b[3] - b[1]) / (W * H), 4),
        "edge": round(edge_trace(t, W), 3),
        "tail": round((t["last"] + 1) / n, 2), "frac": round(t["hits"] / max(1, n - t["birth"]), 2),
        "onset": round(onset(vid, t), 2), "zprobe": round(zp, 2), "zwhere": zwhere,
        "bncc": round(back_match(vid, t, g), 2),
        "spre": round(spre, 2), "hpre": hpre,
        "grow": round(hpost / hpre, 2) if (hpre and hpost) else None,
        "box": [round(v, 1) for v in b],
    }


def score(f):
    """Two branches; a track's score is the better of them. >0 means it popped in, and the
    magnitude is the weakest satisfied margin."""
    a = min([(f["onset"] - ONSET_T) / ONSET_T,
             (ZPROBE_T - f["zprobe"]) / ZPROBE_T,
             (BNCC_T - f["bncc"]) / BNCC_T])
    b = -1.0
    if USE_BRANCH_B and f["hpre"] and f["grow"] is not None:   # branch B needs a prior blob
        b = min([(SPRE_T - f["spre"]) / SPRE_T,
                 (f["peak"] - PEAK_B) / 0.15,
                 (GROW_B - f["grow"]) / GROW_B,
                 (f["onset"] - ONSET_B) / ONSET_B])
    f["branch"] = "A" if a >= b else "B"
    return max(a, b)


def analyse(vid, rec, ctx=CTX):
    n, W, H = rec["n"], rec["w"], rec["h"]
    out = []
    for t in track(rec["dets"], W, H):
        if not candidate(t, n, W, H, ctx):
            continue
        f = features(vid, rec["dets"], t, n, ctx)
        f["score"] = round(score(f), 2)
        out.append(f)
    out.sort(key=lambda f: -f["score"])
    return out


def panel(vid, name, f, outp):
    b = f["birth"]
    x0, y0, x1, y1 = [int(v) for v in f["box"]]
    ims = []
    for k in [max(0, b - 12), max(0, b - 6), max(0, b - 2), b, min(len(vid) - 1, b + 4),
              min(len(vid) - 1, b + 12), min(len(vid) - 1, b + 28)]:
        fr = vid[k].copy()
        cv2.rectangle(fr, (x0, y0), (x1, y1), (255, 40, 40), 2)
        fr = cv2.resize(fr, (300, 173))
        cv2.putText(fr, str(k), (4, 16), 0, 0.5, (0, 255, 0), 1)
        ims.append(fr)
    row = np.concatenate(ims, 1)
    lab = np.full((26, row.shape[1], 3), 20, np.uint8)
    cv2.putText(lab, f"{name} {f['cls']} birth={b} score={f['score']:+.2f} branch={f['branch']} "
                     f"onset={f['onset']} zprobe={f['zprobe']} bncc={f['bncc']}", (4, 19), 0, 0.45, (255, 255, 255), 1)
    cv2.imwrite(outp, cv2.cvtColor(np.concatenate([lab, row], 0), cv2.COLOR_RGB2BGR))


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main():
    tag = arg("--tag", "minwm")
    vdir = arg("--dir", "/home/ashish/stationary_evaluation")
    ctx = int(arg("--ctx", CTX))
    cache = json.load(open(os.path.join(OUTD, f"popin_dets_{tag}.json")))
    rows, flagged, panels = {}, set(), []

    print(f"{'clip':<13}{'GT':>4} {'score':>6} {'br':>3}  {'cls':<8}{'birth':>6}{'onset':>7}{'zprb':>6}{'bncc':>6}{'spre':>6}{'grow':>6}")
    for name, rec in sorted(cache.items()):
        vid = iio.imread(os.path.join(vdir, name + ".mp4"), plugin="pyav")
        ev = analyse(vid, rec, ctx)
        rows[name] = ev
        r = int(name.split("_r")[1]) if "_r" in name else -1
        g = "GT+" if r in GT else " -"
        top = ev[0] if ev else None
        if top and top["score"] > 0:
            flagged.add(r)
        if top:
            print(f"{name:<13}{g:>4} {top['score']:+6.2f} {top['branch']:>3}  {top['cls']:<8}{top['birth']:>6}"
                  f"{top['onset']:>7.2f}{top['zprobe']:>6.2f}{top['bncc']:>6.2f}{top['spre']:>6.2f}"
                  f"{(top['grow'] if top['grow'] is not None else float('nan')):>6.2f}")
            if "--save" in sys.argv and top["score"] > 0:
                p = os.path.join(OUTD, f"popin_{name}.png")
                panel(vid, name, top, p)
                panels.append(p)
        else:
            print(f"{name:<13}{g:>4} {'--':>6}      (no persistent newborn interior object)")

    json.dump(rows, open(os.path.join(OUTD, f"popin_tracks_{tag}.json"), "w"), indent=1)
    print(f"\nflagged={sorted(flagged)}")
    if GT and tag == "minwm":
        tp, fp, fn = flagged & GT, flagged - GT, GT - flagged
        print(f"GT     ={sorted(GT)}")
        print(f"TP={sorted(tp)} ({len(tp)}/{len(GT)})  FP={sorted(fp)}  FN={sorted(fn)}")
        print(f"precision={len(tp)/max(1,len(flagged)):.2f}  recall={len(tp)/len(GT):.2f}")
    if panels:
        ims = [cv2.imread(p) for p in panels]
        w = max(i.shape[1] for i in ims)
        ims = [cv2.copyMakeBorder(i, 0, 0, 0, w - i.shape[1], cv2.BORDER_CONSTANT, value=(20, 20, 20)) for i in ims]
        cv2.imwrite(os.path.join(OUTD, f"popin_evidence_{tag}.png"), np.concatenate(ims, 0))
        print(f"wrote out/popin_evidence_{tag}.png")


if __name__ == "__main__":
    main()
