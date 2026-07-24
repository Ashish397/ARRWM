"""Temporal metrics battery: RAFT warping error (EvalCrafter-style), CLIP-Temp,
DINOv2 frame consistency. Reference-free, per tile video.

Writes results_flow.csv (grid, variant, metric, value).
"""
import glob, os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import torchvision
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

DEV = "cuda"
TILE_DIR = os.path.join(os.path.dirname(__file__), "tiles")
OUT_CSV = os.path.join(os.path.dirname(__file__), "results_flow.csv")


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


def flow_warp(img, flow):
    """Backward-warp img (B,C,H,W) using flow (B,2,H,W) given in pixels: out(x) = img(x + flow(x))."""
    B, C, H, W = img.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=img.device), torch.arange(W, device=img.device), indexing="ij")
    grid = torch.stack([xx, yy], 0).float()[None] + flow
    gx = grid[:, 0] / (W - 1) * 2 - 1
    gy = grid[:, 1] / (H - 1) * 2 - 1
    return F.grid_sample(img, torch.stack([gx, gy], -1), align_corners=True, padding_mode="border")


@torch.no_grad()
def warping_error(frames, raft, batch=4, iters=12, stride=2):
    """Mean occlusion-masked photometric error between RAFT-warped frame t+stride and frame t."""
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div(255.0)
    xn = x * 2 - 1  # raft expects [-1,1]
    errs, flow_mags = [], []
    pairs = [(i, i + stride) for i in range(0, len(x) - stride, stride)]
    for b0 in range(0, len(pairs), batch):
        chunk = pairs[b0 : b0 + batch]
        i1 = torch.stack([xn[i] for i, _ in chunk]).to(DEV)
        i2 = torch.stack([xn[j] for _, j in chunk]).to(DEV)
        fw = raft(i1, i2, num_flow_updates=iters)[-1]  # flow mapping img1 pixels -> img2
        bw = raft(i2, i1, num_flow_updates=iters)[-1]
        # occlusion: forward-backward consistency
        bw_at_fw = flow_warp(bw, fw)
        fb = (fw + bw_at_fw).norm(dim=1)  # B,H,W
        mag = fw.norm(dim=1)
        occ = fb > (0.05 * (mag + fb) + 1.0)  # occluded / inconsistent
        img1 = torch.stack([x[i] for i, _ in chunk]).to(DEV)
        img2 = torch.stack([x[j] for _, j in chunk]).to(DEV)
        warped = flow_warp(img2, fw)  # img2 warped into frame t coords
        err = (warped - img1).abs().mean(1)  # B,H,W
        valid = ~occ
        errs.append(((err * valid).sum((1, 2)) / valid.sum((1, 2)).clamp(min=1)).cpu())
        flow_mags.append(mag.mean((1, 2)).cpu())
    return float(torch.cat(errs).mean()), float(torch.cat(flow_mags).mean())


@torch.no_grad()
def clip_temp(frames, clip_model, stride=1):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div(255.0)
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    x = ((x - mean) / std)
    embs = []
    for i in range(0, len(x), 16):
        embs.append(F.normalize(clip_model.encode_image(x[i : i + 16].to(DEV)), dim=-1).cpu())
    e = torch.cat(embs)
    adj = (e[:-stride] * e[stride:]).sum(-1)
    return float(adj.mean()), float((e[0] * e[1:]).sum(-1).mean())


@torch.no_grad()
def dino_consistency(frames, dino, stride=1):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div(255.0)
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    x = (x - mean) / std
    embs = []
    for i in range(0, len(x), 16):
        embs.append(F.normalize(dino(x[i : i + 16].to(DEV)), dim=-1).cpu())
    e = torch.cat(embs)
    adj = (e[:-stride] * e[stride:]).sum(-1)
    first = (e[0] * e[1:]).sum(-1)
    return float((adj.mean() + first.mean()) / 2)


def main():
    raft = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(DEV).eval()
    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    clip_model = clip_model.to(DEV).eval()
    import timm
    dino = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224).to(DEV).eval()

    rows = []
    for fp in sorted(glob.glob(os.path.join(TILE_DIR, "*.mp4"))):
        name = os.path.basename(fp)[: -len(".mp4")]
        grid, variant = name.split("__")
        frames = read_frames(fp)
        werr, fmag = warping_error(frames, raft)
        ct_adj, ct_first = clip_temp(frames, clip_model)
        dc = dino_consistency(frames, dino)
        m = {"warping_error": werr, "flow_mag": fmag, "clip_temp": ct_adj,
             "clip_first_sim": ct_first, "dino_consistency": dc}
        for k, v in m.items():
            rows.append({"grid": grid, "variant": variant, "metric": k, "value": v})
        print(f"{name}: warp={werr:.5f} flow={fmag:.2f} clipT={ct_adj:.4f} dino={dc:.4f}", flush=True)

    df = pd.DataFrame(rows)
    if os.path.exists(OUT_CSV):
        old = pd.read_csv(OUT_CSV)
        old = old[~old.metric.isin(df.metric.unique())]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
