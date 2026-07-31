"""Stage-2 GPU metrics per video (one pass, one GPU):

  - CSD contrastive-style-descriptor drift: BASE->END (composite feature) and
    CTX->END (style-shift instrument b), plus worst-region spread over a
    4x2 patch grid (max patch drift - median patch drift).
  - DINOv2 frame consistency over the generated span.
  - VGG-Gram matrix distance CTX vs END (style instrument a component).
  - MS-SWD multi-scale sliced Wasserstein color distance CTX vs END.
  - pyiqa NIQE / UNIQUE / LIQE / MUSIQ on BASE and END windows (+drift).
  - RAFT warping error over the generated span.

CSD loading order: TB2_CSD_CKPT (learn2phoenix/CSD checkpoint on a CLIP
ViT-L/14 backbone) -> plain open_clip ViT-L/14 image embeddings as a
fallback (column csd_backend records which). Thresholds are calibrated per
fleet for either backend; population statistics do not transfer between them.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fleet

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_GRID = (4, 2)          # (cols, rows) worst-region grid
N_CONSIST = 16               # frames for DINO consistency / RAFT
IQA_FRAMES = 4               # frames per window for pyiqa


# ---------------------------------------------------------------- style (CSD)
class StyleEmbedder:
    """CSD if a checkpoint is available, else CLIP ViT-L/14 image tower."""

    def __init__(self):
        import open_clip
        self.backend = "clip_vitl14_fallback"
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai", cache_dir=os.environ.get("HF_HOME"))
        self.visual = model.visual.to(DEV).eval()
        self.style_head = None
        ckpt = os.environ.get("TB2_CSD_CKPT", "")
        if ckpt and os.path.exists(ckpt):
            # tomg-group-umd/CSD-ViT-L pytorch_model.bin layout:
            # module.backbone.* (CLIP ViT-L/14 visual, proj removed) +
            # module.last_layer_style (1024->768 style projection applied to
            # the PRE-projection pooled feature)
            # weights_only=False: ckpt carries numpy scalars (trusted HF repo)
            sd = torch.load(ckpt, map_location="cpu", weights_only=False)
            sd = sd.get("model_state_dict", sd)
            sd = {k[len("module."):] if k.startswith("module.") else k: v
                  for k, v in sd.items()}
            bb = {k[len("backbone."):]: v for k, v in sd.items()
                  if k.startswith("backbone.")}
            head_w = next((v for k, v in sd.items() if "last_layer_style" in k),
                          None)
            if bb and head_w is not None:
                missing, unexpected = self.visual.load_state_dict(bb, strict=False)
                self.visual.proj = None  # style head replaces the projection
                feat_dim = head_w.shape[0] if head_w.dim() == 2 else None
                self.style_head = (head_w if head_w.dim() == 2 and
                                   feat_dim == 1024 else head_w.T).to(DEV).float()
                self.backend = "csd"
                print(f"CSD loaded: {len(bb)} backbone keys "
                      f"({len(missing)} missing, {len(unexpected)} unexpected), "
                      f"style head {tuple(self.style_head.shape)}")
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=DEV)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=DEV)

    @torch.no_grad()
    def embed(self, imgs_uint8):
        """[N,H,W,3] uint8 -> [N,D] L2-normalized style embeddings."""
        x = torch.from_numpy(imgs_uint8).to(DEV).permute(0, 3, 1, 2).float() / 255.
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.mean[:, None, None]) / self.std[:, None, None]
        feats = []
        for i in range(0, len(x), 32):
            f = self.visual(x[i:i + 32]).float()
            if self.style_head is not None:
                f = f @ self.style_head
            feats.append(F.normalize(f, dim=-1))
        return torch.cat(feats)


def emb_drift(embedder, frames_a, frames_b):
    """1 - cos(mean emb A, mean emb B), both windows of uint8 frames."""
    ea = embedder.embed(frames_a).mean(0)
    eb = embedder.embed(frames_b).mean(0)
    return float(1 - F.cosine_similarity(ea, eb, dim=0))


def patch_grid_crops(frames):
    """[N,H,W,3] -> list of (4x2 grid) patch stacks, each [N,h,w,3]."""
    n_c, n_r = PATCH_GRID
    H, W = frames.shape[1:3]
    hs, ws = H // n_r, W // n_c
    return [frames[:, r * hs:(r + 1) * hs, c * ws:(c + 1) * ws]
            for r in range(n_r) for c in range(n_c)]


# -------------------------------------------------------------------- DINOv2
class Dino:
    def __init__(self):
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(DEV).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=DEV)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=DEV)

    @torch.no_grad()
    def cls(self, imgs_uint8):
        x = torch.from_numpy(imgs_uint8).to(DEV).permute(0, 3, 1, 2).float() / 255.
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.mean[:, None, None]) / self.std[:, None, None]
        return F.normalize(self.model(x).float(), dim=-1)


# ------------------------------------------------------------------ VGG Gram
class VggGram:
    LAYERS = (3, 8, 15, 22)  # relu1_2, relu2_2, relu3_3, relu4_3

    def __init__(self):
        from torchvision.models import vgg16, VGG16_Weights
        self.net = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(DEV).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=DEV)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=DEV)

    @torch.no_grad()
    def grams(self, imgs_uint8):
        x = torch.from_numpy(imgs_uint8).to(DEV).permute(0, 3, 1, 2).float() / 255.
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.mean[:, None, None]) / self.std[:, None, None]
        out, feats = [], x
        for i, layer in enumerate(self.net):
            feats = layer(feats)
            if i in self.LAYERS:
                b, c, h, w = feats.shape
                f = feats.reshape(b, c, h * w)
                out.append((f @ f.transpose(1, 2)) / (c * h * w))
            if i >= max(self.LAYERS):
                break
        return out  # list of [B,C,C] grams, one per layer

    def distance(self, imgs_a, imgs_b):
        ga, gb = self.grams(imgs_a), self.grams(imgs_b)
        d = 0.0
        for a, b in zip(ga, gb):
            d += float((a.mean(0) - b.mean(0)).pow(2).mean().sqrt())
        return d / len(ga)


# -------------------------------------------------------------------- MS-SWD
def ms_swd(imgs_a, imgs_b, n_proj=128, scales=3, seed=0):
    """Multi-scale sliced Wasserstein distance between the color
    distributions of two frame sets (uint8 RGB). Training-free."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    ta = torch.from_numpy(imgs_a).to(DEV).permute(0, 3, 1, 2).float() / 255.
    tb = torch.from_numpy(imgs_b).to(DEV).permute(0, 3, 1, 2).float() / 255.
    total = 0.0
    for s in range(scales):
        pa = ta.permute(0, 2, 3, 1).reshape(-1, 3)
        pb = tb.permute(0, 2, 3, 1).reshape(-1, 3)
        n = min(len(pa), len(pb), 200_000)
        ia = torch.randperm(len(pa), generator=g, device=DEV)[:n]
        ib = torch.randperm(len(pb), generator=g, device=DEV)[:n]
        pa, pb = pa[ia], pb[ib]
        dirs = F.normalize(torch.randn(3, n_proj, generator=g, device=DEV), dim=0)
        qa, _ = torch.sort((pa @ dirs), dim=0)
        qb, _ = torch.sort((pb @ dirs), dim=0)
        total += float((qa - qb).abs().mean())
        if s < scales - 1:
            ta = F.avg_pool2d(ta, 2)
            tb = F.avg_pool2d(tb, 2)
    return total / scales


# ---------------------------------------------------------------------- RAFT
class Raft:
    def __init__(self):
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        self.model = raft_small(weights=Raft_Small_Weights.DEFAULT).to(DEV).eval()

    @torch.no_grad()
    def warp_error_pairs(self, frames_a, frames_b):
        """Mean photometric error warping each b (frame t+1) back to its
        paired a (frame t). Pairs are adjacent native frames."""
        def prep(f):
            x = torch.from_numpy(f).to(DEV).permute(0, 3, 1, 2).float()
            x = x / 127.5 - 1.0
            H = (x.shape[2] // 8) * 8
            W = (x.shape[3] // 8) * 8
            return x[:, :, :H, :W]
        xa, xb = prep(frames_a), prep(frames_b)
        H, W = xa.shape[2], xa.shape[3]
        errs = []
        for i in range(len(xa)):
            a, b = xa[i:i + 1], xb[i:i + 1]
            flow = self.model(a, b)[-1]
            yy, xx = torch.meshgrid(torch.arange(H, device=DEV),
                                    torch.arange(W, device=DEV), indexing="ij")
            gx = (xx + flow[0, 0]) / (W - 1) * 2 - 1
            gy = (yy + flow[0, 1]) / (H - 1) * 2 - 1
            warped = F.grid_sample(b, torch.stack([gx, gy], -1)[None],
                                   align_corners=True, padding_mode="border")
            errs.append(float((warped - a).abs().mean()))
        return float(np.mean(errs))


# ----------------------------------------------------------------------- IQA
class Iqa:
    NAMES = ("niqe", "musiq", "unique", "liqe")

    def __init__(self):
        import pyiqa
        self.metrics = {n: pyiqa.create_metric(n, device=DEV) for n in self.NAMES}

    @torch.no_grad()
    def score(self, imgs_uint8):
        x = torch.from_numpy(imgs_uint8).to(DEV).permute(0, 3, 1, 2).float() / 255.
        return {n: float(m(x).mean()) for n, m in self.metrics.items()}


# ---------------------------------------------------------------------- main
def video_features(ref, models):
    frames, times, native_fps = fleet.load_video(ref.path)
    ctx, base, end = fleet.windows(times)
    f_ctx, f_base, f_end = frames[ctx], frames[base], frames[end]
    # true adjacent native-frame pairs for temporal metrics (fps-dependent
    # by construction — scorecard.py checks the confound before interpreting)
    adj = fleet.gen_adjacent_pairs(times, N_CONSIST)
    f_adj_a = frames[[i for i, _ in adj]]
    f_adj_b = frames[[j for _, j in adj]]

    style, dino, vgg, raft, iqa = (models[k] for k in
                                   ("style", "dino", "vgg", "raft", "iqa"))
    row = dict(model=ref.model, scene=ref.scene, direction=ref.direction,
               vid=ref.vid, native_fps=native_fps, csd_backend=style.backend)

    # style embeddings
    row["csd_drift_base_end"] = emb_drift(style, f_base, f_end)
    row["csd_drift_ctx_end"] = emb_drift(style, f_ctx, f_end)
    patch_drifts = [emb_drift(style, pa, pb) for pa, pb in
                    zip(patch_grid_crops(f_base), patch_grid_crops(f_end))]
    row["csd_patch_worst"] = float(np.max(patch_drifts))
    row["csd_patch_spread"] = float(np.max(patch_drifts) - np.median(patch_drifts))

    # DINO consistency: mean adjacent-native-frame cosine over the generated span
    ca, cb = dino.cls(f_adj_a), dino.cls(f_adj_b)
    row["dino_consistency"] = float((ca * cb).sum(-1).mean())
    row["dino_drift_base_end"] = float(
        1 - F.cosine_similarity(dino.cls(f_base).mean(0),
                                dino.cls(f_end).mean(0), dim=0))

    # style instrument components (first second = real CTX vs last second)
    row["gram_ctx_end"] = vgg.distance(f_ctx, f_end)
    row["msswd_ctx_end"] = ms_swd(f_ctx, f_end)

    # IQA on base + end
    sub = lambda w: w[np.linspace(0, len(w) - 1, min(IQA_FRAMES, len(w))).astype(int)]
    for wname, wf in (("base", sub(f_base)), ("end", sub(f_end)), ("ctx", sub(f_ctx))):
        for k, v in iqa.score(wf).items():
            row[f"{k}_{wname}"] = v
    for k in Iqa.NAMES:
        # positive drift = quality got worse (niqe is lower-better, rest higher-better)
        sign = -1.0 if k == "niqe" else 1.0
        row[f"{k}_drift"] = sign * (row[f"{k}_base"] - row[f"{k}_end"])
    row["musiq_ctx_end_drift"] = row["musiq_ctx"] - row["musiq_end"]

    # RAFT warping error over adjacent native-frame pairs
    row["raft_warp_err"] = raft.warp_error_pairs(f_adj_a, f_adj_b)
    return row


def main():
    out = os.environ.get("TB2_GPU_OUT",
                         os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "out", "gpu_features.csv"))
    shard = int(os.environ.get("TB2_SHARD", "0"))
    nshard = int(os.environ.get("TB2_NSHARD", "1"))
    if nshard > 1:
        out = out.replace(".csv", f".shard{shard}.csv")

    refs = fleet.refs_from_env()
    refs = [r for i, r in enumerate(refs) if i % nshard == shard]
    models = dict(style=StyleEmbedder(), dino=Dino(), vgg=VggGram(),
                  raft=Raft(), iqa=Iqa())
    print(f"{len(refs)} videos, csd_backend={models['style'].backend}", flush=True)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    for i, ref in enumerate(refs):
        try:
            rows.append(video_features(ref, models))
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {ref.vid}: {e}", flush=True)
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(refs)}", flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
