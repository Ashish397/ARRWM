"""Extended style-shift battery (literature-backed):

- ss_musiq_drift: Rolling-Forcing DeltaDriftQuality = MUSIQ(first window) - MUSIQ(last window)
  (published on 832x480@16fps AR rollouts; CausVid=2.18, Self-Forcing=1.66, RollingForcing=0.01)
- ss_dino_drift: DINOv2 embedding cosine distance start vs end (texture/blur sensitive)
- ss_csd_drift: CSD (contrastive style descriptor, content-invariant) cosine distance start vs end
- ss_msswd: MS-SWD multiscale sliced Wasserstein color distance start vs end (human-validated)
- ss_hf_ratio_drift: change in high/low-frequency radial spectrum energy ratio (AR drift kills HF first)

Appends to results_style.csv.
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
OUT_CSV = os.path.join(HERE, "results_style.csv")
sys.path.insert(0, os.path.join(HERE, "MS-SWD"))


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


def to01(frames):
    return torch.from_numpy(frames).permute(0, 3, 1, 2).float().div(255.0)


def hf_ratio(frames):
    """High-frequency / low-frequency radial power ratio, mean over frames."""
    ratios = []
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        spec = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
        H, W = g.shape
        yy, xx = np.mgrid[0:H, 0:W]
        r = np.sqrt((yy - H / 2) ** 2 + (xx - W / 2) ** 2) / (min(H, W) / 2)
        lo = spec[(r > 0.02) & (r < 0.2)].sum()
        hi = spec[(r >= 0.4) & (r < 1.0)].sum()
        ratios.append(hi / (lo + 1e-12))
    return float(np.mean(ratios))


def load_csd():
    import clip as clip_pkg
    from huggingface_hub import hf_hub_download
    model, _ = clip_pkg.load("ViT-L/14", device="cpu")
    visual = model.visual.float()
    p = hf_hub_download("tomg-group-umd/CSD-ViT-L", "pytorch_model.bin")
    sd = torch.load(p, map_location="cpu", weights_only=False)["model_state_dict"]
    bb = {k[len("module.backbone."):]: v for k, v in sd.items() if k.startswith("module.backbone.")}
    missing, unexpected = visual.load_state_dict(bb, strict=False)
    style_proj = sd["module.last_layer_style"].float()
    # replace CLIP's proj with style projection
    visual.proj = torch.nn.Parameter(style_proj)
    print(f"CSD loaded (missing={len(missing)} unexpected={len(unexpected)})")
    return visual.to(DEV).eval()


@torch.no_grad()
def embed_generic(frames, model, size, mean, std, batch=8):
    x = to01(frames)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    x = (x - mean) / std
    out = []
    for i in range(0, len(x), batch):
        out.append(F.normalize(model(x[i : i + batch].to(DEV)).float(), dim=-1).cpu())
    return torch.cat(out).mean(0)


def main():
    import pyiqa
    musiq = pyiqa.create_metric("musiq-spaq", device=DEV)  # VBench imaging-quality flavor
    import timm
    dino = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224).to(DEV).eval()
    csd = load_csd()
    from MS_SWD import MS_SWD
    msswd = MS_SWD(num_scale=5, num_proj=128).to(DEV)

    inet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    inet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

    W = 16
    rows = []
    for fp in sorted(glob.glob(os.path.join(TILE_DIR, "*.mp4"))):
        name = os.path.basename(fp)[: -len(".mp4")]
        grid, variant = name.split("__")
        frames = read_frames(fp)
        start, end = frames[:W], frames[-W:]

        with torch.no_grad():
            mu_s = float(musiq(to01(start).to(DEV)).mean())
            mu_e = float(musiq(to01(end).to(DEV)).mean())
        m = {"ss_musiq_start": mu_s, "ss_musiq_end": mu_e,
             "ss_musiq_drift": mu_s - mu_e, "ss_musiq_drift_abs": abs(mu_s - mu_e)}

        e_s = embed_generic(start, dino, 224, inet_mean, inet_std)
        e_e = embed_generic(end, dino, 224, inet_mean, inet_std)
        m["ss_dino_drift"] = float(1 - F.cosine_similarity(e_s, e_e, dim=0))

        c_s = embed_generic(start, csd, 224, clip_mean, clip_std)
        c_e = embed_generic(end, csd, 224, clip_mean, clip_std)
        m["ss_csd_drift"] = float(1 - F.cosine_similarity(c_s, c_e, dim=0))

        with torch.no_grad():
            xs = to01(start[::2]).to(DEV)
            xe = to01(end[::2]).to(DEV)
            m["ss_msswd"] = float(msswd(xs, xe).mean())

        m["ss_hf_ratio_drift"] = hf_ratio(start[::4]) - hf_ratio(end[::4])

        for k, v in m.items():
            rows.append({"grid": grid, "variant": variant, "metric": k, "value": v})
        print(f"{name}: musiq_drift={m['ss_musiq_drift']:+.2f} dino={m['ss_dino_drift']:.4f} csd={m['ss_csd_drift']:.4f} msswd={m['ss_msswd']:.3f} hf={m['ss_hf_ratio_drift']:+.5f}", flush=True)

    df = pd.DataFrame(rows)
    if os.path.exists(OUT_CSV):
        old = pd.read_csv(OUT_CSV)
        old = old[~old.metric.isin(df.metric.unique())]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
