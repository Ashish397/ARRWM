"""ADOPTED mangle composite on an arbitrary eval video set: mean rank of
  nov_dino_max  (DINOv2 patch novelty vs blur/brightness-augmented seed bank, time-max)
  mae_max       (ViT-MAE masked-recon error, p90 over tiles, time-max)
  dists_max     (SD-VAE recon scored with DISTS, time-max)
Env: MC_RANK (window, default 1), MC_BRANCH (default R), MC_PHASE (default A).
Prints per-model raw scores + ranks + composite; saves analysis/eval_final/mangle_composite_r{RANK}_{BRANCH}.csv
"""
import os
import numpy as np, pandas as pd, imageio, torch
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RUNS = os.environ.get("MC_RUNS", "pca8_8node,pca4,pca2,16node,4node,noatok").replace(":", ",").split(",")
RANK = int(os.environ.get("MC_RANK", "1"))
BR = os.environ.get("MC_BRANCH", "R")
PH = os.environ.get("MC_PHASE", "A")
SEED_IDX = [0, 3, 6, 9]
GEN_IDX = list(range(12, 106, 6))
PATCH, STRIDE = 320, 160

import clip as _c  # noqa (env sanity)
DINO = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(DEV).eval()
DMEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
DSTD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
from diffusers import AutoencoderKL
AE = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEV).eval().requires_grad_(False)
import pyiqa
DISTS = pyiqa.create_metric("dists", device=DEV)
from transformers import ViTMAEForPreTraining
MAE = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base").to(DEV).eval()
MAE.config.mask_ratio = 0.75


def unfold(t, p=PATCH, s=STRIDE):
    return F.unfold(t, p, stride=s).transpose(1, 2).reshape(-1, 3, p, p)


def dino_feats(ps):
    v = F.interpolate(ps, size=224, mode="bilinear", align_corners=False)
    with torch.no_grad():
        e = DINO((v - DMEAN) / DSTD)
    return (e / e.norm(dim=-1, keepdim=True)).float()


def aug_views(x):
    outs = [x]
    for k in (9, 21, 41):
        w = torch.ones(3, 1, k, k, device=DEV) / (k * k)
        outs.append(F.conv2d(x, w, padding=k // 2, groups=3))
    outs.append((x * 0.7).clamp(0, 1)); outs.append((x * 1.3).clamp(0, 1))
    return outs


def mae_err(t):
    tiles = F.interpolate(unfold(t, 320, 256), size=224, mode="bilinear", align_corners=False)
    tiles = (tiles - DMEAN) / DSTD
    errs = []
    with torch.no_grad():
        for _ in range(3):
            out = MAE(pixel_values=tiles)
            pp = MAE.patchify(MAE.unpatchify(out.logits)); tp = MAE.patchify(tiles)
            e = ((pp - tp) ** 2).mean(-1)
            e = (e * out.mask).sum(1) / out.mask.sum(1)
            errs.append(e.cpu().numpy())
    return float(np.percentile(np.mean(errs, 0), 90))


def dists_ae(t):
    with torch.no_grad():
        lat = AE.encode(t * 2 - 1).latent_dist.mode()
        rec = ((AE.decode(lat).sample + 1) / 2).clamp(0, 1)
    return float(DISTS(t, rec).item())


def main():
    rows = []
    for run in RUNS:
        if run == "minwm":
            p = f"logs/eval_final/A_minwm/minwm_r{RANK:02d}_{BR}.mp4"
        else:
            p = f"logs/eval_final/{PH}/{run}/control_test/step05000_r{RANK:02d}_{BR}_raw.mp4"
        r = imageio.get_reader(p)
        n = r.count_frames()
        if n == float("inf") or n is None:
            n = 108
            try:
                while True:
                    r.get_data(int(n)); n += 1
            except Exception:
                pass
        gen_idx = [i for i in GEN_IDX if i < n - 1] or [int(n) - 2]
        frames = {i: np.asarray(r.get_data(i)) for i in SEED_IDX + gen_idx}
        r.close()
        t = lambda i: torch.tensor(frames[i]).permute(2, 0, 1).unsqueeze(0).float().to(DEV) / 255.
        bank = torch.cat([dino_feats(unfold(v)) for i in SEED_IDX for v in aug_views(t(i))], 0)
        nov, mae_v, dst = [], [], []
        for i in gen_idx:
            ps = unfold(t(i))
            d = (1 - dino_feats(ps) @ bank.T).min(1).values.cpu().numpy()
            nov.append(float(np.percentile(d, 95)))
            if i % 12 == 0:
                mae_v.append(mae_err(t(i))); dst.append(dists_ae(t(i)))
        rows.append(dict(run=run, nov_dino_max=round(max(nov), 4),
                         mae_max=round(max(mae_v), 4), dists_max=round(max(dst), 4)))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    for c in ("nov_dino_max", "mae_max", "dists_max"):
        df[f"rk_{c}"] = df[c].rank()
    df["composite"] = df[[f"rk_{c}" for c in ("nov_dino_max", "mae_max", "dists_max")]].mean(1).round(2)
    df = df.sort_values("composite")
    df.to_csv(f"analysis/eval_final/mangle_composite_r{RANK:02d}_{BR}.csv", index=False)
    print(f"\n=== COMPOSITE prediction for r{RANK:02d}_{BR} (low = clean, high = mangled) ===")
    print(df[["run", "nov_dino_max", "mae_max", "dists_max", "composite"]].to_string(index=False))


if __name__ == "__main__":
    main()
