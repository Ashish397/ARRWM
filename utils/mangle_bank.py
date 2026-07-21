"""Real-bank patch-kNN mangle metric (AnomalyDINO/PatchCore recipe).

Bank = DINOv2 ViT-L patch tokens from the 32 REAL reference clips (multi-layer,
r=3 neighborhood-pooled, L2-normalized). Query frame score = mean cosine distance
of the top-1% most-anomalous patches to their bank nearest neighbor. Video score
= mean/max over generated frames (seed frames skipped).

Validation targets (user ground truth):
  r08_B : 16node worst BY FAR; pca8 hardly bad
  r08_BL: pca8 best, pca2 2nd, noadaln very good; pca4/16node/4node/noatok VERY mangled
  real refs (leave-one-clip-out): clean floor

Env: MB_WINDOWS "r08_B:r08_BL:r01_R", MB_RUNS colon list, MB_STRIDE (bank frame
stride, def 4), MB_QSTRIDE (query stride, def 8), MB_LAYERS "11:17" (0-based),
MB_TAIL (def 0.01), MB_OUT csv. Outputs per (window, run): tail-mean and max.
"""
import os, glob
import numpy as np
import pandas as pd
import imageio
import torch
import torch.nn.functional as F

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
DEV = "cuda"
WINDOWS = os.environ.get("MB_WINDOWS", "r08_B:r08_BL:r01_R").split(":")
RUNS = os.environ.get("MB_RUNS", "pca8_8node:pca4:pca2:16node:4node:noatok:noadaln").split(":")
STRIDE = int(os.environ.get("MB_STRIDE", "4"))
QSTRIDE = int(os.environ.get("MB_QSTRIDE", "8"))
LAYERS = [int(x) for x in os.environ.get("MB_LAYERS", "11:17").split(":")]
TAIL = float(os.environ.get("MB_TAIL", "0.01"))
OUT = os.environ.get("MB_OUT", f"{ARR}/analysis/eval_final/mangle_bank.csv")
H, W = 476, 826          # multiples of 14 close to 480x832
GH, GW = H // 14, W // 14

model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14").to(DEV).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)


def frames(path, stride, start=0):
    r = imageio.get_reader(path)
    out = [np.asarray(f) for i, f in enumerate(r) if i >= start and (i - start) % stride == 0]
    r.close()
    return out


@torch.no_grad()
def feats(imgs):
    """(N,H0,W0,3) uint8 -> (N, GH*GW, D) pooled multi-layer patch features."""
    all_f = []
    for i in range(0, len(imgs), 8):
        batch = torch.stack([torch.from_numpy(x).permute(2, 0, 1) for x in imgs[i:i + 8]])
        batch = batch.to(DEV).float() / 255.0
        batch = F.interpolate(batch, size=(H, W), mode="bilinear", align_corners=False)
        batch = (batch - MEAN) / STD
        layers = model.get_intermediate_layers(batch.to(torch.float32), n=LAYERS, reshape=True)
        # each: (B, D, GH, GW); r=3 neighborhood pooling then concat layers
        if os.environ.get("MB_POOL", "1") == "1":
            pooled = [F.avg_pool2d(l, 3, stride=1, padding=1) for l in layers]
        else:
            pooled = list(layers)
        f = torch.cat(pooled, dim=1)                     # (B, D*, GH, GW)
        f = f.flatten(2).transpose(1, 2)                 # (B, GH*GW, D*)
        f = F.normalize(f, dim=-1)
        all_f.append(f.half().cpu())
    return torch.cat(all_f)


def knn_dist(q, bank_gpu, chunk=8192):
    """q: (P, D) cpu half; bank_gpu: (M, D) gpu half -> (P,) min cosine distance."""
    mins = []
    for i in range(0, q.shape[0], chunk):
        sim = q[i:i + chunk].to(DEV) @ bank_gpu.T        # (p, M)
        mins.append((1.0 - sim.max(dim=1).values).float().cpu())
    return torch.cat(mins)


def video_score(path, bank_gpu, start):
    fr = frames(path, QSTRIDE, start=start)
    if not fr:
        return None
    fs = feats(fr)                                       # (N, P, D)
    per_frame = []
    k = max(1, int(round(fs.shape[1] * TAIL)))
    for n in range(fs.shape[0]):
        d = knn_dist(fs[n], bank_gpu)
        per_frame.append(float(d.topk(k).values.mean()))
    return float(np.mean(per_frame)), float(np.max(per_frame)), len(per_frame)


def main():
    refs = sorted(glob.glob(f"{ARR}/analysis/eval_final/real_refs/*.mp4"))
    print(f"[bank] {len(refs)} real ref clips", flush=True)
    per_clip = []
    for p in refs:
        per_clip.append(feats(frames(p, STRIDE)))
        print(f"[bank] {os.path.basename(p)}: {per_clip[-1].shape}", flush=True)
    full = torch.cat([c.reshape(-1, c.shape[-1]) for c in per_clip])
    print(f"[bank] total tokens: {full.shape}", flush=True)

    rows = []
    bank_gpu = full.to(DEV)
    for w in WINDOWS:
        for run in RUNS:
            path = f"{ARR}/logs/eval_final/A/{run}/control_test/step05000_{w}_raw.mp4"
            if not os.path.exists(path):
                print(f"[skip] {path}"); continue
            mean_s, max_s, n = video_score(path, bank_gpu, start=13)
            rows.append(dict(window=w, run=run, tail_mean=round(mean_s, 4),
                             tail_max=round(max_s, 4), n_frames=n))
            print(f"[score] {w} {run}: mean={mean_s:.4f} max={max_s:.4f}", flush=True)
        # minwm reference (77-frame, label-swapped disk names handled upstream)
        mw = {"r08_B": f"{ARR}/logs/eval_final/A_minwm/minwm_r08_B.mp4",
              "r08_BL": f"{ARR}/logs/eval_final/A_minwm/minwm_r08_BR.mp4",
              "r01_R": f"{ARR}/logs/eval_final/A_minwm/minwm_r01_L.mp4"}.get(w)
        if mw and os.path.exists(mw):
            mean_s, max_s, n = video_score(mw, bank_gpu, start=13)
            rows.append(dict(window=w, run="minwm", tail_mean=round(mean_s, 4),
                             tail_max=round(max_s, 4), n_frames=n))
            print(f"[score] {w} minwm: mean={mean_s:.4f} max={max_s:.4f}", flush=True)
    del bank_gpu; torch.cuda.empty_cache()

    # real refs: leave-one-clip-out
    for i, p in enumerate(refs[:8]):
        others = torch.cat([c.reshape(-1, c.shape[-1]) for j, c in enumerate(per_clip) if j != i])
        og = others.to(DEV)
        mean_s, max_s, n = video_score(p, og, start=0)
        del og; torch.cuda.empty_cache()
        rows.append(dict(window="REAL", run=os.path.basename(p)[:24], tail_mean=round(mean_s, 4),
                         tail_max=round(max_s, 4), n_frames=n))
        print(f"[score] REAL {os.path.basename(p)}: mean={mean_s:.4f} max={max_s:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))
    print("\n=== per-window ranking (tail_mean desc = most mangled first) ===")
    for w in WINDOWS:
        sub = df[df.window == w].sort_values("tail_mean", ascending=False)
        print(w, ":", " ".join(f"{r.run}:{r.tail_mean:.3f}" for r in sub.itertuples()))


if __name__ == "__main__":
    main()
