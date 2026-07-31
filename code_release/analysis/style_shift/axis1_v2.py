"""Axis-1 v2: global style-flip / quality-collapse detection.

Usage: axis1_v2.py <dir> <fps>

Features per frame (416-wide working res):
  A. CLIP ViT-B/32 embedding (reused from <dir>/clip_embs.npy if present).
  B. Global Gram/AdaIN style stats: VGG16 relu1_2/2_2/3_3 channel mean+std,
     + color moments (prompt-free style representation).
  C. Quality features: Laplacian blur, RMS contrast, saturation,
     colorfulness, gray entropy (collapse channel, e.g. vista sepia mush).

Detection:
  - ruptures KernelCPD(rbf) on z-scored [CLIP | style] -> segment boundaries
    (penalty swept, reported at several sensitivities).
  - Quality collapse: robust slope + sustained-drop test per quality feature.
CSD embeddings are a planned GPU upgrade (ViT-L too slow on this CPU cap).
"""
import os
import sys
import cv2
import numpy as np
import torch
import ruptures as rpt

DIRN = sys.argv[1]
FPS = float(sys.argv[2])
frames = np.load(f"{DIRN}/frames.npy", mmap_mode="r")
T, H0, W0, _ = frames.shape
W = 416
H = int(round(H0 * W / W0 / 8)) * 8
small = np.stack([cv2.resize(np.asarray(f), (W, H), interpolation=cv2.INTER_AREA) for f in frames])
del frames
print(f"{DIRN}: {T} frames @{FPS}fps, working res {W}x{H}")

# ---- A: CLIP ----
if os.path.exists(f"{DIRN}/clip_embs.npy"):
    clip = np.load(f"{DIRN}/clip_embs.npy").astype(np.float32)
else:
    import open_clip
    from PIL import Image
    model, _, prep = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    model.eval()
    es = []
    with torch.no_grad():
        for i in range(0, T, 16):
            b = torch.stack([prep(Image.fromarray(f)) for f in small[i:i + 16]])
            es.append(torch.nn.functional.normalize(model.encode_image(b), dim=-1))
    clip = torch.cat(es).numpy().astype(np.float32)
    np.save(f"{DIRN}/clip_embs.npy", clip)
    del model

# ---- B: global style stats ----
from torchvision.models import vgg16, VGG16_Weights
torch.set_num_threads(16)
net = vgg16(weights=VGG16_Weights.DEFAULT).features[:16].eval()
TAPS = {3: 64, 8: 128, 15: 256}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def gstats(batch):
    x = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0
    out, cur = [], (x - MEAN) / STD
    with torch.no_grad():
        for i, layer in enumerate(net):
            cur = layer(cur)
            if i in TAPS:
                out += [cur.mean((2, 3)), cur.std((2, 3))]
    out += [x.mean((2, 3)), x.std((2, 3))]
    return torch.cat(out, 1).numpy()

style = np.concatenate([gstats(small[i:i + 4]) for i in range(0, T, 4)]).astype(np.float32)
del net

# ---- C: quality features ----
def quality(f):
    g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
    blur = cv2.Laplacian(g, cv2.CV_32F).var()
    contrast = g.std()
    x = f.astype(np.float32)
    sat = (x.max(-1) - x.min(-1)).mean()
    rg, yb = x[..., 0] - x[..., 1], 0.5 * (x[..., 0] + x[..., 1]) - x[..., 2]
    colorful = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    hist = np.histogram(g, 64, (0, 255))[0] / g.size
    ent = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
    return np.array([blur, contrast, sat, colorful, ent], np.float32)

qual = np.stack([quality(f) for f in small])
QNAMES = ["blur_var", "contrast", "saturation", "colorfulness", "entropy"]

# ---- change-point detection ----
def z(m):
    return (m - m.mean(0)) / (m.std(0) + 1e-6)

feat = np.concatenate([z(clip) / np.sqrt(clip.shape[1]), z(style) / np.sqrt(style.shape[1])], 1)
min_size = max(4, int(0.75 * FPS))
algo = rpt.KernelCPD(kernel="rbf", min_size=min_size).fit(feat)
print("\nKernelCPD segment boundaries (s) by penalty:")
results = {}
for pen in (2, 5, 10, 20):
    bks = algo.predict(pen=pen)[:-1]
    results[pen] = bks
    print(f"  pen={pen:>2}: " + (", ".join(f"{b / FPS:.2f}" for b in bks) if bks else "none"))

# ---- quality collapse: sustained deviation from the first-second baseline ----
k = max(3, int(FPS / 2) | 1)
collapse = {}
qs = (qual - qual[:int(FPS)].mean(0)) / (qual[:int(FPS)].std(0) + 1e-6)
for j, n in enumerate(QNAMES):
    s = np.convolve(qs[:, j], np.ones(k) / k, "same")
    hit = None
    need = int(1.0 * FPS)
    for t in range(int(FPS), T - need):
        if (np.abs(s[t:t + need]) > 4).all():
            hit = t
            break
    collapse[n] = hit
    if hit is not None:
        print(f"quality collapse [{n}]: onset {hit / FPS:.2f}s (dir {'-' if s[hit] < 0 else '+'})")

np.savez(f"{DIRN}/axis1_v2.npz", style=style, qual=qual,
         bks_pen5=np.array(results[5]), bks_pen10=np.array(results[10]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ts = np.arange(T) / FPS
fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
# style-feature novelty (for context): distance to running median segment mean
sfeat = z(style)
w = max(4, int(1.0 * FPS))
nov = np.zeros(T)
for t in range(w, T - w):
    a, b = sfeat[t - w:t].mean(0), sfeat[t:t + w].mean(0)
    nov[t] = np.linalg.norm(a - b) / np.sqrt(style.shape[1])
ax[0].plot(ts, nov, color="purple", label="Gram/AdaIN novelty (1s)")
for b in results[5]:
    ax[0].axvline(b / FPS, color="r", ls="--", alpha=0.7)
for b in results[10]:
    ax[0].axvline(b / FPS, color="darkred", ls="-", alpha=0.9, lw=2)
ax[0].set_ylabel("style novelty")
ax[0].legend(fontsize=8)
ax[0].set_title(f"{DIRN}: Axis-1 v2 (red dashed pen=5, solid pen=10 changepoints)")
for j, n in enumerate(QNAMES):
    ax[1].plot(ts, qs[:, j], label=n, lw=1)
ax[1].axhline(4, color="gray", ls=":"); ax[1].axhline(-4, color="gray", ls=":")
ax[1].set_ylabel("quality (z vs first 1s)")
ax[1].legend(fontsize=7, ncol=5)
d0 = 1 - clip @ (clip[:int(FPS)].mean(0) / np.linalg.norm(clip[:int(FPS)].mean(0)))
ax[2].plot(ts, d0, color="g")
ax[2].set_ylabel("CLIP dist to context")
ax[2].set_xlabel("time (s)")
fig.tight_layout()
fig.savefig(f"{DIRN}/axis1_v2.png", dpi=110)
print(f"saved {DIRN}/axis1_v2.png")
