"""Worst-region (patch-level) metrics: catch localized structural mangling that
frame-mean metrics average away ("big messy structure on the right").

Frame is split into a 4x2 grid of patches (208x224 each). For each patch region:
per-region NIQE / MUSIQ / CSD-drift / gram-drift; report the WORST region value
and the spread (worst - median across regions).

Writes results_patch.csv.
"""
import glob, os, sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

DEV = "cuda"
HERE = os.path.dirname(os.path.abspath(__file__))
TILE_DIR = os.path.join(HERE, "tiles")
OUT_CSV = os.path.join(HERE, "results_patch.csv")
sys.path.insert(0, HERE)
W = 16
GX, GY = 4, 2  # patch grid


def read_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


def to01(f):
    return torch.from_numpy(f).permute(0, 3, 1, 2).float().div(255.0)


def patches(frames):
    """T,H,W,3 -> list of (T,h,w,3) region stacks, row-major."""
    T, H, Wd, _ = frames.shape
    ph, pw = H // GY, Wd // GX
    return [frames[:, gy*ph:(gy+1)*ph, gx*pw:(gx+1)*pw] for gy in range(GY) for gx in range(GX)]


def main():
    import pyiqa, timm
    musiq = pyiqa.create_metric("musiq-spaq", device=DEV)
    niqe = pyiqa.create_metric("niqe", device=DEV)
    from style_shift import VGGStyle, gram_distance
    vgg = VGGStyle().to(DEV)
    from style_shift2 import load_csd
    csd = load_csd()
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

    @torch.no_grad()
    def csd_embed(fr):
        x = to01(fr)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = ((x - clip_mean) / clip_std).to(DEV)
        out = [F.normalize(csd(x[i:i+8]).float(), dim=-1).cpu() for i in range(0, len(x), 8)]
        return torch.cat(out).mean(0)

    @torch.no_grad()
    def grams(fr):
        x = to01(fr).to(DEV)
        outs = [vgg(x[i:i+8]) for i in range(0, len(x), 8)]
        return [torch.cat([o[l] for o in outs]) for l in range(len(outs[0]))]

    rows = []
    for fp in sorted(glob.glob(os.path.join(TILE_DIR, "*.mp4"))):
        name = os.path.basename(fp)[: -len(".mp4")]
        grid, variant = name.split("__")
        frames = read_frames(fp)
        idx = np.linspace(0, len(frames) - 1, 8).round().astype(int)
        regs = patches(frames)
        vals = {"p_niqe": [], "p_musiq_end": [], "p_csd_drift": [], "p_gram_drift": []}
        with torch.no_grad():
            for reg in regs:
                vals["p_niqe"].append(float(niqe(to01(reg[idx]).to(DEV)).mean()))
                vals["p_musiq_end"].append(float(musiq(to01(reg[-W::4]).to(DEV)).mean()))
                vals["p_csd_drift"].append(float(1 - F.cosine_similarity(csd_embed(reg[:W]), csd_embed(reg[-W:]), dim=0)))
                vals["p_gram_drift"].append(gram_distance(grams(reg[:W:2]), grams(reg[-W::2])))
        m = {}
        for k, v in vals.items():
            v = np.array(v)
            hi_bad = k in ("p_niqe", "p_csd_drift", "p_gram_drift")  # higher = worse
            worst = v.max() if hi_bad else v.min()
            m[f"{k}_worst"] = float(worst)
            m[f"{k}_spread"] = float((v.max() - np.median(v)) if hi_bad else (np.median(v) - v.min()))
        for k, v in m.items():
            rows.append({"grid": grid, "variant": variant, "metric": k, "value": v})
        print(f"{name}: niqe_worst={m['p_niqe_worst']:.2f} csd_worst={m['p_csd_drift_worst']:.3f} gram_spread={m['p_gram_drift_spread']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    if os.path.exists(OUT_CSV):
        old = pd.read_csv(OUT_CSV)
        old = old[~old.metric.isin(df.metric.unique())]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV} ({len(df)} rows)")


if __name__ == "__main__":
    main()
