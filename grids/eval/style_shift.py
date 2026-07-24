"""Start-vs-end style shift detector for rollout videos (reference-free).

For each tile video, compare a start window vs an end window with:
  - VGG16 Gram-matrix style distance (content-insensitive style drift)
  - CLIP embedding cosine distance (semantic/style drift, content-confounded)
  - Lab color statistics shift (mean/std per channel)
  - Haze (dark channel prior), saturation, contrast, darkness drift

Outputs eval/results_style.csv with one row per (grid, variant, metric).
Because all variants of a grid share scene+actions, downstream analysis should
also use per-grid median-relative values to cancel content-driven drift.
"""
import argparse, glob, os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import torchvision

DEV = "cuda"
TILE_DIR = os.path.join(os.path.dirname(__file__), "tiles")
OUT_CSV = os.path.join(os.path.dirname(__file__), "results_style.csv")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


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


class VGGStyle(torch.nn.Module):
    """VGG16 features at relu1_2, relu2_2, relu3_3, relu4_3 -> Gram matrices."""

    LAYERS = [3, 8, 15, 22]

    def __init__(self):
        super().__init__()
        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg.features[: self.LAYERS[-1] + 1].eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):  # x: B,3,H,W in [0,1]
        x = (x - IMAGENET_MEAN.to(x)) / IMAGENET_STD.to(x)
        grams = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.LAYERS:
                b, c, h, w = x.shape
                f = x.reshape(b, c, h * w)
                g = torch.bmm(f, f.transpose(1, 2)) / (c * h * w)
                grams.append(g)
        return grams  # list of B,C,C


def gram_distance(grams_a, grams_b):
    """Mean Frobenius distance between per-window mean Grams, normalized per layer."""
    d = 0.0
    for ga, gb in zip(grams_a, grams_b):
        ma, mb = ga.mean(0), gb.mean(0)
        denom = (ma.norm() + mb.norm()) / 2 + 1e-8
        d += float((ma - mb).norm() / denom)
    return d / len(grams_a)


def window_stats(frames):
    """Color/haze stats for a window of uint8 RGB frames."""
    f32 = frames.astype(np.float32)
    lab = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2LAB) for f in frames]).astype(np.float32)
    hsv = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2HSV) for f in frames]).astype(np.float32)
    gray = f32.mean(-1)
    dark_ch = []
    for f in frames:
        dc = cv2.erode(f.min(-1), np.ones((15, 15), np.uint8))
        dark_ch.append(dc.mean())
    return {
        "lab_mean": lab.reshape(-1, 3).mean(0),          # L,a,b means
        "lab_std": lab.reshape(-1, 3).std(0),
        "saturation": hsv[..., 1].mean(),
        "contrast": gray.std(),
        "dark_frac": float((gray < 30).mean()),           # fraction near-black
        "haze_dark_channel": float(np.mean(dark_ch)),     # high = hazy veil
        "laplacian": float(np.mean([cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() for f in frames])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--out", default=OUT_CSV)
    args = ap.parse_args()

    vggm = VGGStyle().to(DEV)
    import open_clip
    clip_model, _, clip_pre = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    clip_model = clip_model.to(DEV).eval()
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

    def clip_embed(frames):
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div(255.0)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = ((x - clip_mean) / clip_std).to(DEV)
        with torch.no_grad():
            e = clip_model.encode_image(x)
        return F.normalize(e, dim=-1).mean(0)

    def vgg_grams(frames):
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div(255.0).to(DEV)
        with torch.no_grad():
            out = [vggm(x[i : i + 8]) for i in range(0, len(x), 8)]
        return [torch.cat([o[l] for o in out]) for l in range(len(out[0]))]

    rows = []
    for fp in sorted(glob.glob(os.path.join(TILE_DIR, "*.mp4"))):
        name = os.path.basename(fp)[: -len(".mp4")]
        grid, variant = name.split("__")
        frames = read_frames(fp)
        W = args.window
        start, end = frames[:W], frames[-W:]

        m = {}
        m["ss_gram_dist"] = gram_distance(vgg_grams(start), vgg_grams(end))
        e_s, e_e = clip_embed(start), clip_embed(end)
        m["ss_clip_dist"] = float(1 - (e_s @ e_e))

        st_s, st_e = window_stats(start), window_stats(end)
        m["ss_dL"] = float(st_e["lab_mean"][0] - st_s["lab_mean"][0])          # brightness shift
        m["ss_dab"] = float(np.linalg.norm(st_e["lab_mean"][1:] - st_s["lab_mean"][1:]))  # chroma shift
        m["ss_d_contrast"] = float(st_e["contrast"] - st_s["contrast"])
        m["ss_d_saturation"] = float(st_e["saturation"] - st_s["saturation"])
        m["ss_d_dark_frac"] = float(st_e["dark_frac"] - st_s["dark_frac"])      # + = got blacker
        m["ss_d_haze"] = float(st_e["haze_dark_channel"] - st_s["haze_dark_channel"])  # + = got hazier
        m["ss_d_sharpness"] = float(st_e["laplacian"] - st_s["laplacian"])      # - = got blurrier
        # composite magnitude of photometric drift
        m["ss_photometric"] = abs(m["ss_dL"]) / 10 + m["ss_dab"] / 5 + abs(m["ss_d_saturation"]) / 20 + abs(m["ss_d_contrast"]) / 10

        for k, v in m.items():
            rows.append({"grid": grid, "variant": variant, "metric": k, "value": v})
        print(f"{name}: gram={m['ss_gram_dist']:.4f} clip={m['ss_clip_dist']:.4f} dL={m['ss_dL']:+.1f} haze={m['ss_d_haze']:+.1f} dark={m['ss_d_dark_frac']:+.3f}", flush=True)

    df = pd.DataFrame(rows)
    if os.path.exists(args.out):
        old = pd.read_csv(args.out)
        old = old[~old.metric.isin(df.metric.unique())]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
