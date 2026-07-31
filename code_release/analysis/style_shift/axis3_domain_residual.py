"""Axis-3: semantic/domain drift under constant photorealism.

Usage: axis3_domain_residual.py <dir> <fps> <seed_mp4>

The eval rollouts are scripted F/R commands (not the ride's true trajectory),
so no frame-aligned GT continuation exists. But commanded forward/turn cannot
change the DOMAIN (city, architecture, object population), so we use:

  A. CLIP domain residual: distance of each generated frame to the ride
     prototype (mean CLIP embedding of the real seed clip), calibrated by the
     seed frames' own residual spread. Sustained excess = domain reversion.
     Caveat: whole-frame CLIP still carries layout, so slow growth is partly
     ego-motion; the signal is the LEVEL it saturates at and sustained excess.
  B. Object-population trajectory: YOLOv8n vehicle/person counts per frame vs
     the seed-clip counts; step-ups flag spawn events (yume's car at ~1s).

Outputs: axis3_scores.csv, axis3_curve.png, printed onsets.
"""
import sys
import numpy as np
import torch

DIRN = sys.argv[1]
FPS = float(sys.argv[2])
SEED_MP4 = sys.argv[3]

frames = np.load(f"{DIRN}/frames.npy", mmap_mode="r")
T = len(frames)

# seed clip frames
import av
c = av.open(SEED_MP4)
seed_frames = np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])
print(f"{DIRN}: {T} gen frames, {len(seed_frames)} seed frames")

# ---- A: CLIP domain residual ----
import open_clip
from PIL import Image
model, _, prep = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
model.eval()
torch.set_num_threads(16)

def embed(arr):
    es = []
    with torch.no_grad():
        for i in range(0, len(arr), 16):
            b = torch.stack([prep(Image.fromarray(np.asarray(f))) for f in arr[i:i + 16]])
            es.append(torch.nn.functional.normalize(model.encode_image(b), dim=-1))
    return torch.cat(es).numpy().astype(np.float32)

import os
gen_emb = (np.load(f"{DIRN}/clip_embs.npy").astype(np.float32)
           if os.path.exists(f"{DIRN}/clip_embs.npy") else embed(frames))
seed_emb = embed(seed_frames)
del model

proto = seed_emb.mean(0)
proto /= np.linalg.norm(proto)
seed_res = 1 - seed_emb @ proto
thr = seed_res.mean() + 3 * seed_res.std()
res = 1 - gen_emb @ proto
k = max(3, int(FPS / 2) | 1)
res_s = np.convolve(res, np.ones(k) / k, "same")
onset = None
need = int(1.0 * FPS)
for t in range(0, T - need):
    if (res_s[t:t + need] > thr).all():
        onset = t
        break
print(f"domain residual: seed thr {thr:.4f}; onset "
      + (f"{onset / FPS:.2f}s" if onset is not None else "none")
      + f"; final level {res_s[-k:].mean():.4f} ({res_s[-k:].mean() / max(thr, 1e-6):.1f}x thr)")

# ---- B: object population ----
from ultralytics import YOLO
yolo = YOLO("yolov8n.pt")
VEH = {2, 3, 5, 7}   # car, motorcycle, bus, truck
PER = {0}

def counts(arr):
    cv_, cp_ = np.zeros(len(arr)), np.zeros(len(arr))
    for i in range(0, len(arr), 16):
        batch = [np.asarray(f)[..., ::-1] for f in arr[i:i + 16]]  # BGR for yolo
        for j, r in enumerate(yolo(batch, verbose=False, conf=0.3)):
            cls = r.boxes.cls.cpu().numpy().astype(int)
            cv_[i + j] = np.isin(cls, list(VEH)).sum()
            cp_[i + j] = np.isin(cls, list(PER)).sum()
        print(f"  yolo {min(i + 16, len(arr))}/{len(arr)}", flush=True)
    return cv_, cp_

gen_veh, gen_per = counts(frames)
seed_veh, seed_per = counts(seed_frames)
veh_s = np.convolve(gen_veh, np.ones(k) / k, "same")
base = seed_veh.mean()
print(f"vehicles: seed mean {base:.2f}; gen start {veh_s[:int(FPS)].mean():.2f}; "
      f"gen end {veh_s[-int(FPS):].mean():.2f}; max {veh_s.max():.2f}")
spawn = None
for t in range(T - need):
    if (veh_s[t:t + need] > base + 0.9).all():
        spawn = t
        break
print("vehicle-spawn onset: " + (f"{spawn / FPS:.2f}s" if spawn is not None else "none"))

ts = np.arange(T) / FPS
np.savetxt(f"{DIRN}/axis3_scores.csv",
           np.column_stack([ts, res, gen_veh, gen_per]),
           delimiter=",", header="t_sec,domain_residual,vehicles,persons", comments="")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
ax[0].plot(ts, res, alpha=0.4)
ax[0].plot(ts, res_s, lw=2, color="C0")
ax[0].axhline(thr, color="gray", ls=":", label="seed-clip 3sd")
if onset is not None:
    ax[0].axvline(onset / FPS, color="r", ls="--")
    ax[0].text(onset / FPS, res_s.max(), f" domain onset {onset / FPS:.2f}s", color="r", fontsize=9)
ax[0].set_ylabel("CLIP dist to ride prototype")
ax[0].legend(fontsize=8)
ax[0].set_title(f"{DIRN}: Axis-3 domain drift")
ax[1].plot(ts, gen_veh, alpha=0.35, color="C1")
ax[1].plot(ts, veh_s, lw=2, color="C1", label="vehicles (smoothed)")
ax[1].plot(ts, np.convolve(gen_per, np.ones(k) / k, "same"), lw=1.2, color="C2", label="persons")
ax[1].axhline(base, color="gray", ls=":", label="seed vehicle mean")
if spawn is not None:
    ax[1].axvline(spawn / FPS, color="r", ls="--")
    ax[1].text(spawn / FPS, veh_s.max(), f" spawn {spawn / FPS:.2f}s", color="r", fontsize=9)
ax[1].set_ylabel("object count")
ax[1].set_xlabel("time (s)")
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{DIRN}/axis3_curve.png", dpi=110)
print(f"saved {DIRN}/axis3_curve.png")
