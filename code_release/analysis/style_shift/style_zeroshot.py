"""Pass 2: CLIP zero-shot STYLE classification per frame + long-window novelty.

P(style class | frame) via CLIP text anchors -> directly answers "when does it
stop looking photoreal and lock onto the blocky game style", independent of
scene content. Also: 1.5s-window novelty for major transitions and greedy
segment merge for a piecewise-constant style timeline.
"""
import os
import sys
import numpy as np
import torch
import open_clip
from PIL import Image

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("AF_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "analysis/style_shift")
FPS = float(sys.argv[2]) if len(sys.argv) > 2 else 25
frames = np.load(f"{OUT}/frames.npy")
T = len(frames)
TITLE = OUT.rstrip("/").split("/")[-1]

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k")
tok = open_clip.get_tokenizer("ViT-B-32")
model.eval()

STYLES = {
    "photoreal": "a real photograph from a car dashcam, a real street scene",
    "degraded": "a blurry distorted corrupted AI-generated image with melted warped objects",
    "game_render": "a 3D rendered video game environment with clean flat textures",
    "blocky_voxel": "a screenshot from a blocky voxel video game like Minecraft with large cube blocks",
}
with torch.no_grad():
    tfeat = model.encode_text(tok(list(STYLES.values())))
    tfeat = torch.nn.functional.normalize(tfeat, dim=-1)

embs = []
with torch.no_grad():
    for i in range(0, T, 32):
        b = torch.stack([preprocess(Image.fromarray(f)) for f in frames[i:i + 32]])
        embs.append(torch.nn.functional.normalize(model.encode_image(b), dim=-1))
        print(f"  clip {i + len(b)}/{T}", flush=True)
embs = torch.cat(embs)
np.save(f"{OUT}/clip_embs.npy", embs.numpy())

probs = (100.0 * embs @ tfeat.T).softmax(-1).numpy()  # (T, 4)

# smooth 0.5 s
k = np.ones(max(3, int(FPS / 2) | 1)) / max(3, int(FPS / 2) | 1)
probs_s = np.stack([np.convolve(probs[:, j], k, mode="same") for j in range(probs.shape[1])], 1)

names = list(STYLES)
dom = probs_s.argmax(1)
print("\nDominant-style timeline (merged runs >0.4s):")
segs, start = [], 0
for t in range(1, T + 1):
    if t == T or dom[t] != dom[start]:
        if (t - start) >= 0.4 * FPS:
            segs.append((start, t, names[dom[start]]))
        start = t
for a, b, n in segs:
    print(f"  {a/FPS:6.2f}-{b/FPS:6.2f} s : {n}")

# long-window novelty (1.5 s halves) on raw embeddings
W = max(4, int(1.5 * FPS))
e = embs.numpy()
nov = np.zeros(T)
for t in range(W, T - W):
    a = e[t - W:t].mean(0); b = e[t:t + W].mean(0)
    nov[t] = 1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
peaks = [t for t in range(W, T - W)
         if nov[t] == nov[max(0, t - W):t + W].max() and nov[t] > 0.5 * nov.max()]
print("\nMajor transitions (1.5s-window CLIP novelty):")
for t in peaks:
    print(f"  t = {t/FPS:.2f} s   nov={nov[t]:.4f}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ts = np.arange(T) / FPS
fig, ax = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
for j, n in enumerate(names):
    ax[0].plot(ts, probs_s[:, j], label=n, lw=1.6)
ax[0].set_ylabel("CLIP zero-shot P(style)")
ax[0].legend(loc="center right", fontsize=8)
ax[0].set_title(f"{TITLE}: per-frame style classification")
ax[1].plot(ts, nov, color="purple")
for t in peaks:
    ax[1].axvline(t / FPS, color="r", ls="--", alpha=0.6)
    ax[1].text(t / FPS, nov.max() * 0.92, f"{t/FPS:.1f}s", color="r",
               ha="center", fontsize=8)
ax[1].set_ylabel("CLIP novelty (w=1.5s)")
ax[1].set_xlabel("time (s)")
fig.tight_layout()
fig.savefig(f"{OUT}/style_timeline.png", dpi=110)
print(f"saved {OUT}/style_timeline.png")
