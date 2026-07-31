"""Axis-2: localized style reversion in NEWLY GENERATED content.

Usage: axis2_local_style.py <dir> <fps> <seed_frames> [<mp4>]
  <dir>         subdir with frames.npy (or give <mp4> to decode first)
  <seed_frames> number of leading frames that are real (teacher-forced seed)

Pipeline (all half-res, CPU):
  1. DIS optical flow (fwd+bwd) with forward-backward consistency check.
  2. Propagate a "real-traceable" mask from the seed frames: a pixel is REAL
     if flow traces it back to the seed without an fb-consistency break.
     Everything else is GENERATED content.
  3. Per-tile style stats (VGG16 relu1_2/2_2/3_3 channel mean+std [AdaIN
     stats] + RGB/saturation moments) on a 16px grid.
  4. Relative score per GENERATED tile = cosine distance to the 25th-pct
     nearest REAL tile in the SAME frame (cancels exposure/blur/compression).
     When <MIN_REAL real tiles remain (e.g. after a long turn), fall back to
     the seed-frame tile bank (flagged in output).
  5. Outputs: axis2_scores.csv, axis2_curve.png, keyframe heatmaps,
     printed onset/peak (onset = sustained > null_mean + 3*null_std).
"""
import os
import sys
import cv2
import numpy as np
import torch

DIRN = sys.argv[1]
FPS = float(sys.argv[2])
SEED_N = int(sys.argv[3])
TILE = 16          # tile size at half-res
FB_TOL = 1.5       # px, forward-backward tolerance (+ 5% of flow mag)
MIN_REAL = 8       # min same-frame real tiles before seed-bank fallback
FRAC = 0.6         # tile purity threshold

if len(sys.argv) > 4:
    os.system(f"python decode_video.py {sys.argv[4]} {DIRN}")
frames = np.load(f"{DIRN}/frames.npy", mmap_mode="r")
T, H0, W0, _ = frames.shape
W = 416                                    # fixed working width -> comparable scores
H = int(round(H0 * W / W0 / TILE)) * TILE
small = np.stack([cv2.resize(np.asarray(f), (W, H), interpolation=cv2.INTER_AREA) for f in frames])
del frames
gray = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in small])
print(f"{T} frames, working res {W}x{H}, seed {SEED_N} frames")

# ---- 1+2: flow + real-mask propagation (rolling, memory-capped node) ----
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
real_bin = np.zeros((T, H, W), bool)
real_bin[:SEED_N] = True
prev = np.ones((H, W), np.float32)
for t in range(max(1, SEED_N), T):
    bw = dis.calc(gray[t], gray[t - 1], None)        # t -> t-1
    fw = dis.calc(gray[t - 1], gray[t], None)        # t-1 -> t
    sx, sy = gx + bw[..., 0], gy + bw[..., 1]        # src coords in t-1
    fw_at_src = cv2.remap(fw, sx, sy, cv2.INTER_LINEAR, borderValue=1e6)
    fb_err = np.linalg.norm(bw + fw_at_src, axis=-1)
    mag = np.linalg.norm(bw, axis=-1) + np.linalg.norm(fw_at_src, axis=-1)
    ok = (fb_err < FB_TOL + 0.05 * mag) & (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
    cur = np.where(ok, cv2.remap(prev, sx, sy, cv2.INTER_LINEAR, borderValue=0), 0.0)
    real_bin[t] = cur > 0.5
    prev = cur
del gray
# erode so tile stats don't straddle the boundary
real_er = np.stack([cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) for m in real_bin])
gen_er = np.stack([cv2.erode((~m).astype(np.uint8), np.ones((5, 5), np.uint8)) for m in real_bin])
print("real-traceable fraction: t0 %.2f  mid %.2f  end %.2f" %
      (real_bin[min(SEED_N, T - 1)].mean(), real_bin[T // 2].mean(), real_bin[-1].mean()))

# ---- 3: per-tile style stats ----
from torchvision.models import vgg16, VGG16_Weights
torch.set_num_threads(32)
net = vgg16(weights=VGG16_Weights.DEFAULT).features[:16].eval()
TAPS = {3: 64, 8: 128, 15: 256}                       # relu1_2, relu2_2, relu3_3
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
ty, tx = H // TILE, W // TILE

def tile_stats_batch(batch):  # (B,H,W,3) uint8 -> (B, ty, tx, D)
    x = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0
    feats, cur = [], (x - MEAN) / STD
    with torch.no_grad():
        for i, layer in enumerate(net):
            cur = layer(cur)
            if i in TAPS:
                feats.append(cur)
    out = []
    for f in feats:                                   # (B,C,h,w)
        s = f.shape[-2] * f.shape[-1] // (H * W)      # spatial ratio never int; use adaptive
        p = torch.nn.functional.adaptive_avg_pool2d(f, (ty, tx))
        p2 = torch.nn.functional.adaptive_avg_pool2d(f ** 2, (ty, tx))
        std = (p2 - p ** 2).clamp_min(0).sqrt()
        out += [p, std]
    xs = x
    for m in (xs, xs.max(1, keepdim=True).values - xs.min(1, keepdim=True).values):  # rgb + sat proxy
        p = torch.nn.functional.adaptive_avg_pool2d(m, (ty, tx))
        p2 = torch.nn.functional.adaptive_avg_pool2d(m ** 2, (ty, tx))
        out += [p, (p2 - p ** 2).clamp_min(0).sqrt()]
    return torch.cat(out, 1).permute(0, 2, 3, 1).numpy()

stats = []
B = 2
for i in range(0, T, B):
    stats.append(tile_stats_batch(small[i:i + B]).astype(np.float16))
    print(f"  vgg {min(i + B, T)}/{T}", flush=True)
stats = np.concatenate(stats)                          # (T, ty, tx, D) fp16
del net
flat = stats.reshape(-1, stats.shape[-1])
mu = flat.mean(0, dtype=np.float32)
sd = flat.std(0, dtype=np.float32) + 1e-6
for i in range(0, T, 32):                              # normalize in chunks, stay fp16
    c = (stats[i:i + 32].astype(np.float32) - mu) / sd
    c /= np.linalg.norm(c, axis=-1, keepdims=True) + 1e-9
    stats[i:i + 32] = c.astype(np.float16)
del flat

# tile-level masks
def tile_frac(m):  # (T,H,W) -> (T,ty,tx)
    return m.reshape(T, ty, TILE, tx, TILE).mean((2, 4))
real_t = tile_frac(real_er.astype(np.float32)) > FRAC
gen_t = tile_frac(gen_er.astype(np.float32)) > FRAC

# seed-frame tile bank (fallback reference)
bank = stats[:SEED_N].reshape(-1, stats.shape[-1]) if SEED_N > 0 else stats[:1].reshape(-1, stats.shape[-1])

# ---- 4: relative style score ----
# NN distance of each generated tile to the real tiles, self-calibrated by the
# NN distances real tiles show among THEMSELVES (excess above real-self p90).
score_map = np.zeros((T, ty, tx), np.float32)
frame_score, frame_top, used_fallback = np.zeros(T), np.zeros(T), np.zeros(T, bool)
for t in range(T):
    gsel = gen_t[t]
    if not gsel.any():
        continue
    g = stats[t][gsel].astype(np.float32)
    if real_t[t].sum() >= MIN_REAL:
        ref = stats[t][real_t[t]].astype(np.float32)
    else:
        ref = bank.astype(np.float32)
        used_fallback[t] = True
    g_min = (1 - g @ ref.T).min(1)                     # NN dist gen -> real
    d_rr = 1 - ref @ ref.T
    np.fill_diagonal(d_rr, 2.0)
    r_ref = np.quantile(d_rr.min(1), 0.9)              # real-self NN level
    excess = np.clip(g_min - r_ref, 0, None)
    score_map[t][gsel] = excess
    frame_score[t] = excess.mean()
    frame_top[t] = np.quantile(excess, 0.75) if gsel.sum() >= 6 else excess.mean()

# ---- 5: onset/peak + outputs ----
k = max(3, int(FPS / 2) | 1)
sm = np.convolve(frame_top, np.ones(k) / k, mode="same")
null_end = min(T, SEED_N + int(1.0 * FPS))
null = sm[SEED_N:null_end]
thr = null.mean() + 3 * null.std() if len(null) > 2 else sm.mean()
above = sm > thr
onset = None
need = int(0.5 * FPS)
for t in range(SEED_N, T - need):
    if above[t:t + need].all():
        onset = t
        break
peak = int(np.argmax(sm))
print(f"\nnull thr {thr:.4f}; onset: "
      + (f"{onset / FPS:.2f}s" if onset is not None else "none")
      + f"; peak {sm[peak]:.4f} @ {peak / FPS:.2f}s; fallback frames: {used_fallback.sum()}")

ts = np.arange(T) / FPS
np.savetxt(f"{DIRN}/axis2_scores.csv",
           np.column_stack([ts, frame_score, frame_top, used_fallback]),
           delimiter=",", header="t_sec,gen_tile_mean,gen_tile_p75,seed_bank_fallback", comments="")
np.save(f"{DIRN}/axis2_score_map.npy", score_map)
np.save(f"{DIRN}/axis2_real_mask.npy", real_bin)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(ts, frame_top, alpha=0.4, label="p75 gen-tile score")
ax.plot(ts, sm, lw=2, label="smoothed")
ax.axhline(thr, color="gray", ls=":", label="null+3sd")
if onset is not None:
    ax.axvline(onset / FPS, color="r", ls="--")
    ax.text(onset / FPS, sm.max(), f" onset {onset / FPS:.2f}s", color="r")
fb = np.where(used_fallback)[0]
if len(fb):
    ax.axvspan(fb[0] / FPS, fb[-1] / FPS, color="orange", alpha=0.08, label="seed-bank fallback")
ax.set_xlabel("time (s)"); ax.set_ylabel("local style anomaly")
ax.set_title(f"{DIRN}: Axis-2 new-content style anomaly")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{DIRN}/axis2_curve.png", dpi=110)

# keyframe heatmaps: 6 frames spread over video
from PIL import Image
keys = np.linspace(SEED_N, T - 1, 6).astype(int)
tiles_img = []
vmax = max(np.quantile(score_map[score_map > 0], 0.98) if (score_map > 0).any() else 1e-3, 1e-3)
for t in keys:
    base = small[t].astype(np.float32)
    heat = cv2.resize(score_map[t], (W, H), interpolation=cv2.INTER_NEAREST) / vmax
    heat = np.clip(heat, 0, 1)[..., None] * np.array([255.0, 0, 0])
    over = np.clip(base * 0.6 + heat * 0.7, 0, 255).astype(np.uint8)
    edge = cv2.Canny(real_bin[t].astype(np.uint8) * 255, 50, 150)
    over[edge > 0] = (0, 255, 0)
    im = Image.fromarray(over)
    from PIL import ImageDraw
    d = ImageDraw.Draw(im); d.rectangle([0, 0, 70, 18], fill=(0, 0, 0))
    d.text((3, 2), f"{t / FPS:.1f}s", fill=(255, 255, 0))
    tiles_img.append(im)
sh = Image.new("RGB", (3 * W, 2 * H))
for i, im in enumerate(tiles_img):
    sh.paste(im, ((i % 3) * W, (i // 3) * H))
sh.save(f"{DIRN}/axis2_heatmaps.png")

# real-mask audit overlays (green tint = real-traceable) on the same keyframes
tiles_img = []
for t in keys:
    over = small[t].astype(np.float32)
    over[..., 1] = np.clip(over[..., 1] + real_bin[t] * 80, 0, 255)
    im = Image.fromarray(over.astype(np.uint8))
    d = ImageDraw.Draw(im); d.rectangle([0, 0, 70, 18], fill=(0, 0, 0))
    d.text((3, 2), f"{t / FPS:.1f}s", fill=(255, 255, 0))
    tiles_img.append(im)
sh = Image.new("RGB", (3 * W, 2 * H))
for i, im in enumerate(tiles_img):
    sh.paste(im, ((i % 3) * W, (i // 3) * H))
sh.save(f"{DIRN}/axis2_realmask.png")
print(f"saved {DIRN}/axis2_curve.png, axis2_heatmaps.png, axis2_realmask.png")
