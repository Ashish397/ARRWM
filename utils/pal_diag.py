"""Dump PAL4VST masks for the 3 failing GT cases: pca8-B (should be clean),
noadaln-BL (clean), noatok-BL (mangled), real_20240205 (real outlier)."""
import os, numpy as np, imageio, torch
import torch.nn.functional as F
from PIL import Image

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
DEV = "cuda"
TS = f"{ARR}/third_party/PAL4VST/deployment/pal4vst/swin-large_upernet_unified_512x512/end2end.pt"
model = torch.jit.load(TS).to(DEV).eval()
MEAN = torch.tensor([123.675, 116.28, 103.53], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([58.395, 57.12, 57.375], device=DEV).view(1, 3, 1, 1)

CASES = [
    ("pca8_B", f"{ARR}/logs/eval_final/A/pca8_8node/control_test/step05000_r08_B_raw.mp4", [30, 55, 70, 85, 100]),
    ("pca8_BL", f"{ARR}/logs/eval_final/A/pca8_8node/control_test/step05000_r08_BL_raw.mp4", [30, 55, 70, 85, 100]),
]

@torch.no_grad()
def mask_full(img):
    t = torch.from_numpy(img).permute(2, 0, 1)[None].to(DEV).float()
    H0, W0 = t.shape[-2:]
    m = np.zeros((H0, W0), dtype=np.float32)
    for j, sl in enumerate([slice(0, W0 // 2), slice(W0 // 2, W0)]):
        x = F.interpolate(t[..., :, sl], size=(512, 512), mode="bilinear", align_corners=False)
        x = (x - MEAN) / STD
        out = model(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        mk = out.argmax(1) if (out.dim() == 4 and out.shape[1] > 1) else (out.squeeze(1) > 0.5).long()
        mk = F.interpolate(mk[None].float(), size=(H0, W0 // 2), mode="nearest")[0, 0].cpu().numpy()
        m[:, sl] = mk
    return m

rows = []
for name, path, idx in CASES:
    r = imageio.get_reader(path)
    fr = [np.asarray(f) for f in r]; r.close()
    panels = []
    for i in idx:
        i = min(i, len(fr) - 1)
        f = fr[i].copy()
        m = mask_full(fr[i])
        f[m > 0.5] = (0.5 * f[m > 0.5] + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
        panels.append(f)
        print(f"{name} f{i}: artifact_frac={m.mean():.4f}", flush=True)
    rows.append(np.concatenate(panels, axis=1))
grid = np.concatenate(rows, axis=0)
g = Image.fromarray(grid); g = g.resize((g.width // 2, g.height // 2))
g.save(f"{ARR}/analysis/eval_final/_diag/pal_masks.png")
print("saved pal_masks.png (rows: pca8_B, noadaln_BL, noatok_BL, real0205)")
