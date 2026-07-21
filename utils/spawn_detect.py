"""Spatially-resolved SPAWN detector: objects materializing in the frame
INTERIOR (not entering through the frame boundary), phase-A videos.

Replaces the center-crop novelty for spawn detection — the crop missed
off-center spawns (minwm r02_FR taxi appears left-of-center) and a scalar
level can't separate "materialized from nothing" from "legitimately
approached". Method, per video (first SD_MAX_SEC seconds):

  1. DINOv2-S PATCH tokens on 448x448 frames -> 32x32 novelty grid per
     sampled frame (stride 3): per-patch min cos distance to a bank of ALL
     patch tokens of the seed frames (0,3,6,9) + blur/brightness augs.
  2. Adaptive threshold tau per video: leave-one-frame-out novelty of the
     seed frames' own patches -> tau = p99.9 + 0.05 margin (pans/quality
     drift stay below tau; genuinely new objects exceed it).
  3. Binary mask -> connected blobs (scipy label), area >= 4 patches.
  4. SPAWN blob = (a) does NOT touch the outer patch ring (boundary entries
     excluded), (b) no overlap with the previous sample's novelty mask
     dilated by 2 patches (it appeared fresh, didn't move/grow in).
  5. Per video: n_spawns, max spawn blob area (patches), first spawn time.

Saves per-model npz of the full novelty-mask stacks for replotting/QC.
Env: SD_RUNS colon list, SD_OUT csv, SD_NPZ_DIR, SD_MAX_SEC (def 6),
     SD_ONLY comma list like "r02_FR,r06_F" to restrict (smoke).
"""
import os
import numpy as np
import pandas as pd
import imageio
import torch
import torch.nn.functional as F
from scipy.ndimage import label as cc_label, binary_dilation

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
RUNS = os.environ.get("SD_RUNS", "minwm").split(":")
OUT = os.environ.get("SD_OUT", f"{ARR}/analysis/eval_final/spawn_detect.csv")
NPZ_DIR = os.environ.get("SD_NPZ_DIR", f"{ARR}/analysis/eval_final/spawn_masks")
MAX_SEC = float(os.environ.get("SD_MAX_SEC", "6.0"))
ONLY = set(os.environ["SD_ONLY"].split(",")) if os.environ.get("SD_ONLY") else None
DIRS = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]
DEV = "cuda"
RES = 448                 # -> 32x32 patch grid
GRID = RES // 14
MIN_AREA = 4              # patches
TAU_MARGIN = 0.05
STRIDE = 3
_MWSW = {"L": "R", "R": "L", "FL": "FR", "FR": "FL", "BL": "BR", "BR": "BL"}
os.makedirs(NPZ_DIR, exist_ok=True)


def vid_path(run, wi, d):
    if run == "minwm":
        return f"{ARR}/logs/eval_final/A_minwm/minwm_r{wi:02d}_{_MWSW.get(d, d)}.mp4"
    for m in ("matrixgame", "worldcam", "yume", "worldplay", "astra"):
        if run == m:
            return f"{ARR}/logs/eval_final/A_{m}/{m}_r{wi:02d}_{d}.mp4"
    return f"{ARR}/logs/eval_final/A/{run}/control_test/step05000_r{wi:02d}_{d}_raw.mp4"


dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(DEV).eval()
DM = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
DS = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)


@torch.no_grad()
def patch_feats(imgs):
    """imgs [B,3,H,W] 0-1 -> [B, GRID*GRID, 384] L2-normalized patch tokens."""
    v = F.interpolate(imgs, size=RES, mode="bilinear", align_corners=False)
    t = dino.get_intermediate_layers((v - DM) / DS, n=1)[0]      # [B, N, C]
    return t / t.norm(dim=-1, keepdim=True)


def aug(x):
    outs = [x]
    for k in (9, 21, 41):
        w = torch.ones(3, 1, k, k, device=DEV) / (k * k)
        outs.append(F.conv2d(x, w, padding=k // 2, groups=3))
    outs += [(x * 0.7).clamp(0, 1), (x * 1.3).clamp(0, 1)]
    return outs


@torch.no_grad()
def novelty_to_bank(feats, bank):
    """feats [N,C], bank [M,C] -> [N] min cos distance, chunked."""
    best = torch.full((feats.shape[0],), 2.0, device=DEV)
    for j in range(0, bank.shape[0], 8192):
        d = 1 - feats @ bank[j:j + 8192].T
        best = torch.minimum(best, d.min(1).values)
    return best


@torch.no_grad()
def analyze(frames):
    t = lambda f: torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1)[None].to(DEV).float() / 255.0
    seed_idx = [0, 3, 6, 9]
    seed_feats = []                                   # per seed frame: [augs*N, C]
    for i in seed_idx:
        vs = torch.cat(aug(t(frames[i])), 0)
        seed_feats.append(patch_feats(vs).reshape(-1, 384))
    bank = torch.cat(seed_feats, 0)

    # adaptive tau: seed frames' own patches vs the OTHER seed frames' bank
    loo = []
    for si in range(len(seed_idx)):
        others = torch.cat([seed_feats[j] for j in range(len(seed_idx)) if j != si], 0)
        own = patch_feats(t(frames[seed_idx[si]]))[0]
        loo.append(novelty_to_bank(own, others).cpu().numpy())
    tau = float(np.percentile(np.concatenate(loo), 99.9)) + TAU_MARGIN

    masks, idxs = [], []
    for i in range(12, len(frames), STRIDE):
        nov = novelty_to_bank(patch_feats(t(frames[i]))[0], bank)
        masks.append((nov.reshape(GRID, GRID) > tau).cpu().numpy())
        idxs.append(i)

    n_spawns, max_area, first_t = 0, 0, -1.0
    prev_dil = np.zeros((GRID, GRID), bool)
    ring = np.zeros((GRID, GRID), bool)
    ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
    events = []
    for k, m in enumerate(masks):
        lab, nb = cc_label(m)
        for b in range(1, nb + 1):
            blob = lab == b
            if blob.sum() < MIN_AREA:
                continue
            if (blob & ring).any():                   # touches boundary -> entered
                continue
            if (blob & prev_dil).any():               # moved/grew from existing novelty
                continue
            n_spawns += 1
            max_area = max(max_area, int(blob.sum()))
            events.append((idxs[k], int(blob.sum())))
            if first_t < 0:
                first_t = idxs[k]
        prev_dil = binary_dilation(m, iterations=2)
    return tau, np.stack(masks) if masks else np.zeros((0, GRID, GRID), bool), \
        np.array(idxs), n_spawns, max_area, first_t, events


def main():
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(zip(prev.run, prev.window, prev["dir"]))
    f = open(OUT, "a" if done else "w")
    if not done:
        f.write("run,window,dir,n_spawns,max_spawn_area,first_spawn_frame,tau\n")
    for run in RUNS:
        npz_path = f"{NPZ_DIR}/spawn_masks_{run}.npz"
        trajs = dict(np.load(npz_path)) if os.path.exists(npz_path) else {}
        for wi in range(32):
            for d in DIRS:
                key = f"r{wi:02d}_{d}"
                if ONLY and key not in ONLY:
                    continue
                if (run, f"r{wi:02d}", d) in done and key in trajs:
                    continue
                path = vid_path(run, wi, d)
                if not os.path.exists(path):
                    continue
                try:
                    r = imageio.get_reader(path)
                    fps = float(r.get_meta_data().get("fps", 16.0))
                    max_f = max(int(fps * MAX_SEC), 24)
                    frames = []
                    for fi, x in enumerate(r):
                        if fi >= max_f:
                            break
                        frames.append(np.asarray(x))
                    r.close()
                    tau, masks, idxs, n_sp, ma, ft, ev = analyze(frames)
                    trajs[key] = np.packbits(masks, axis=None)
                    trajs[key + "_shape"] = np.array(masks.shape)
                    trajs[key + "_idx"] = idxs
                    if (run, f"r{wi:02d}", d) not in done:
                        f.write(f"{run},r{wi:02d},{d},{n_sp},{ma},{ft:.0f},{tau:.3f}\n")
                        f.flush()
                    if ev:
                        print(f"[spawn] {run} {key}: {n_sp} spawns {ev} tau={tau:.3f}", flush=True)
                except Exception as e:
                    print(f"[spawn] FAIL {run} r{wi:02d} {d}: {e}", flush=True)
        np.savez_compressed(npz_path, **trajs)
        print(f"[spawn] {run} done -> {npz_path}", flush=True)
    f.close()


if __name__ == "__main__":
    main()
