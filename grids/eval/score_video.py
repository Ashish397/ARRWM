"""Production scorer: reference-free quality + style-shift scores for rollout videos.

Usage:
  ./venv/bin/python score_video.py vid1.mp4 vid2.mp4 ...          # one group (same scenario)
  ./venv/bin/python score_video.py --groups "sceneA:a1.mp4,a2.mp4" "sceneB:b1.mp4,..."

Videos passed together in a group are assumed to share the scenario (same start
context / actions), which lets grid-relative features cancel content effects.
A single video can be scored alone (rel features fall back to 0 = population median).

Outputs per video:
  quality_score   ridge composite calibrated on the human GRID LAYOUT.txt evals
                  (LOGO-CV Spearman ~0.70 vs human quality ranks; scale ~0-10)
  style_shift     z-composite of MUSIQ-drift + VGG-Gram + MS-SWD start-vs-end
                  (AUC 0.89 vs human style-shift flags; >0.8 = likely shift)
  plus all raw component metrics.
"""
import argparse, json, os, sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "MS-SWD"))
DEV = "cuda"
W = 16  # start/end window length in frames

# population stats of the 42 calibration tiles, for z-scoring style components
STYLE_POP = {
    "ss_musiq_drift": None,  # filled from style_stats.json at load
}


def read_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames in {path}")
    return np.stack(frames)


def to01(frames):
    return torch.from_numpy(frames).permute(0, 3, 1, 2).float().div(255.0)


class Models:
    def __init__(self):
        import pyiqa, timm, torchvision
        self.musiq = pyiqa.create_metric("musiq-spaq", device=DEV)
        self.niqe = pyiqa.create_metric("niqe", device=DEV)
        self.unique = pyiqa.create_metric("unique", device=DEV)
        self.liqe = pyiqa.create_metric("liqe", device=DEV)
        self.dino = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224).to(DEV).eval()
        from style_shift import VGGStyle, gram_distance
        self.vgg = VGGStyle().to(DEV)
        self.gram_distance = gram_distance
        from style_shift2 import load_csd
        self.csd = load_csd()
        from MS_SWD import MS_SWD
        self.msswd = MS_SWD(num_scale=5, num_proj=128).to(DEV)
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        self.raft = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(DEV).eval()

    @torch.no_grad()
    def embed(self, frames, model, mean, std, batch=8):
        x = to01(frames)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        out = []
        for i in range(0, len(x), batch):
            out.append(F.normalize(model(x[i : i + batch].to(DEV)).float(), dim=-1).cpu())
        return torch.cat(out)


INET = (torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
CLIPN = (torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1), torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))


@torch.no_grad()
def extract(models, path):
    from run_flow_battery import warping_error
    frames = read_frames(path)
    start, end = frames[:W], frames[-W:]
    m = {}

    # style components
    mu_s = float(models.musiq(to01(start).to(DEV)).mean())
    mu_e = float(models.musiq(to01(end).to(DEV)).mean())
    m["ss_musiq_drift"] = mu_s - mu_e
    def grams(fr):
        x = to01(fr).to(DEV)
        outs = [models.vgg(x[i : i + 8]) for i in range(0, len(x), 8)]
        return [torch.cat([o[l] for o in outs]) for l in range(len(outs[0]))]
    m["ss_gram_dist"] = models.gram_distance(grams(start), grams(end))
    m["ss_msswd"] = float(models.msswd(to01(start[::2]).to(DEV), to01(end[::2]).to(DEV)).mean())

    # quality components
    e_dino = models.embed(frames[:: max(1, len(frames) // 32)], models.dino, *INET)
    adj = (e_dino[:-1] * e_dino[1:]).sum(-1)
    first = (e_dino[0] * e_dino[1:]).sum(-1)
    m["dino_consistency"] = float((adj.mean() + first.mean()) / 2)
    c_s = models.embed(start, models.csd, *CLIPN).mean(0)
    c_e = models.embed(end, models.csd, *CLIPN).mean(0)
    m["ss_csd_drift"] = float(1 - F.cosine_similarity(c_s, c_e, dim=0))
    # worst-region CSD drift spread over a 4x2 patch grid (localized mangling)
    T, H, Wd, _ = frames.shape
    ph, pw = H // 2, Wd // 4
    region_drift = []
    for gy in range(2):
        for gx in range(4):
            reg = frames[:, gy*ph:(gy+1)*ph, gx*pw:(gx+1)*pw]
            r_s = models.embed(reg[:W], models.csd, *CLIPN).mean(0)
            r_e = models.embed(reg[-W:], models.csd, *CLIPN).mean(0)
            region_drift.append(float(1 - F.cosine_similarity(r_s, r_e, dim=0)))
    m["p_csd_drift_spread"] = float(np.max(region_drift) - np.median(region_drift))
    gray = frames.astype(np.float32).mean(-1)
    m["ss_d_dark_frac"] = float((gray[-W:] < 30).mean() - (gray[:W] < 30).mean())
    idx = np.linspace(0, len(frames) - 1, 16).round().astype(int)
    m["niqe"] = float(models.niqe(to01(frames[idx]).to(DEV)).mean())
    m["unique"] = float(models.unique(to01(frames[idx]).to(DEV)).mean())
    m["liqe"] = float(models.liqe(to01(frames[idx]).to(DEV)).mean())
    m["s_dark_channel"] = float(np.mean([
        cv2.erode(frames[i].min(-1), np.ones((15, 15), np.uint8)).mean() for i in idx
    ]))
    m["warping_error"], _ = warping_error(frames, models.raft)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*")
    ap.add_argument("--groups", nargs="*", help='each "name:path1,path2,..."')
    ap.add_argument("--out", default=None, help="optional csv output path")
    args = ap.parse_args()

    groups = {}
    if args.groups:
        for g in args.groups:
            name, paths = g.split(":", 1)
            groups[name] = paths.split(",")
    if args.videos:
        groups["default"] = args.videos
    if not groups:
        sys.exit("no videos given")

    qm = json.load(open(os.path.join(HERE, "quality_model.json")))
    sp = json.load(open(os.path.join(HERE, "style_stats.json")))
    models = Models()

    results = []
    for gname, paths in groups.items():
        feats = {p: extract(models, p) for p in paths}
        raw = {k: np.array([feats[p][k] for p in paths]) for k in next(iter(feats.values()))}
        med = {k: np.median(v) for k, v in raw.items()}
        for p in paths:
            f = feats[p]
            rel = {f"rel__{k}": (f[k] - med[k]) if len(paths) > 1 else 0.0 for k in f}
            x = np.array([rel.get(name, f.get(name.replace("rel__", ""), 0.0)) for name in qm["features"]])
            xz = (x - np.array(qm["mu"])) / np.array(qm["sd"])
            quality = float(np.dot(np.append(xz, 1.0), qm["weights"]))
            zsum = sum((f[k] - sp[k]["mean"]) / sp[k]["std"] for k in ("ss_musiq_drift", "ss_gram_dist", "ss_msswd"))
            row = {"group": gname, "video": os.path.basename(p),
                   "quality_score": round(quality, 2), "style_shift_z": round(zsum, 2),
                   "style_shift_flag": bool(zsum > sp["_threshold"]), **{k: round(v, 5) for k, v in f.items()}}
            results.append(row)
            print(row, flush=True)

    if args.out:
        import pandas as pd
        pd.DataFrame(results).to_csv(args.out, index=False)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
