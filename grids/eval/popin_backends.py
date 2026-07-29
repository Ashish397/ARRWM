"""Pluggable object-detection backends for the pop-in detector.

The temporal logic in popin_detect.py is backbone-agnostic: it only needs, per frame, a list of
{box, cls, score}. This module exposes three architecturally distinct detectors so the choice
of backbone can be measured rather than assumed:

  rtdetr     PekingU/rtdetr_r50vd     transformer / set prediction, NMS-free
  frcnn      torchvision R50-FPN v2   two-stage CNN, region proposals + ROI heads
  retinanet  torchvision R50-FPN v2   one-stage CNN, dense anchors + focal loss

All three are COCO-trained with the same category names, so KEEP/CONJURABLE transfer unchanged.
Each backend returns the same two callables:
  dense(vid)        -> per-frame detection lists for a whole clip (batched)
  crop(img, thr)    -> [(box, label, score)] for one image, used by the zoom probe
"""
import numpy as np, torch

DEV = "cuda"
LOW = 0.10          # cache threshold; the tracker and probes apply their own, higher ones
BATCH = 8

NAMES = ("rtdetr", "frcnn", "retinanet")


def _rtdetr():
    from transformers import AutoModelForObjectDetection, AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained("PekingU/rtdetr_r50vd", use_fast=True)
    mdl = AutoModelForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd").eval().to(DEV)
    id2label = mdl.config.id2label

    @torch.no_grad()
    def _run(frames, thr):
        H, W = frames[0].shape[:2]
        inp = proc(images=list(frames), return_tensors="pt").to(DEV)
        res = proc.post_process_object_detection(mdl(**inp), target_sizes=[(H, W)] * len(frames), threshold=thr)
        return [[(b, id2label[int(l)], float(s)) for b, l, s in
                 zip(r["boxes"].cpu().numpy(), r["labels"].cpu().numpy(), r["scores"].cpu().numpy())] for r in res]
    return _run


def _torchvision(kind):
    import torchvision
    if kind == "frcnn":
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
        w = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        mdl = fasterrcnn_resnet50_fpn_v2(weights=w, box_score_thresh=LOW)
    else:
        from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
        w = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
        mdl = retinanet_resnet50_fpn_v2(weights=w, score_thresh=LOW)
    mdl = mdl.eval().to(DEV)
    cats = w.meta["categories"]

    @torch.no_grad()
    def _run(frames, thr):
        xs = [torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1).float().div(255).to(DEV) for f in frames]
        out = []
        for o in mdl(xs):
            out.append([(b, cats[int(l)], float(s)) for b, l, s in
                        zip(o["boxes"].cpu().numpy(), o["labels"].cpu().numpy(), o["scores"].cpu().numpy())
                        if s >= thr])
        return out
    return _run


def build(name):
    """Return (dense, crop) for a backend name."""
    run = _rtdetr() if name == "rtdetr" else _torchvision(name)

    def dense(vid):
        out = []
        for i in range(0, len(vid), BATCH):
            for per in run(list(vid[i:i + BATCH]), LOW):
                out.append([{"box": [float(x) for x in b], "cls": c, "score": float(s)} for b, c, s in per])
        return out

    def crop(img, thr=LOW):
        return run([img], thr)[0]

    return dense, crop
