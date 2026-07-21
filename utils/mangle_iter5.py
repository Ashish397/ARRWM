"""Mangle iteration 5 (augmented seed bank)

Bank = seed patches + gaussian-blurred (k=9,21,41) + brightness±30% variants.
Blur smudges (4node) now match the blurred vocabulary -> novelty ~0; structural
melt (pca4 canopy) matches nothing -> stays novel. Otherwise same as iter 4 (r08 BL): STRUCTURED novelty + temporal warp-residual.

Iter3 learning: CLIP seed-novelty time-max finally caught pca4's melted canopy
(highest score) and pca8 as cleanest — but 4node's soft blur smudges also fired.
Distinction: structural melt has EDGES (crisp impossible geometry); blur doesn't.

  A) struct-novelty : per 320px patch, novelty = min cosine distance of patch
     feature (CLIP + DINOv2 variants) to the video's own seed-frame patch bank,
     WEIGHTED by relative gradient energy (Laplacian var / frame median, capped);
     blur patches -> weight ~0, melt keeps full weight. Frame score p95, video
     score = max over rollout.
  B) warp-residual  : consecutive sampled frames, SIFT similarity fit, warp, then
     1 - SSIM on gradient-magnitude maps per patch in the valid overlap; p90 per
     step, video score = max over steps. Melt = structure that rigid motion can't
     explain; blur = low-gradient -> low residual.

Ground truth BL: CLEAN={pca8_8node,4node}, MANGLED={pca4,pca2,16node,noatok}.
Saves analysis/eval_final/mangle_iter5_r08.csv
"""
import os
import numpy as np, pandas as pd, imageio, torch, cv2
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RUNS = ["pca8_8node", "pca4", "pca2", "16node", "4node", "noatok"]
CLEAN = {"pca8_8node", "4node"}
SEED_IDX = [0, 3, 6, 9]
GEN_IDX = list(range(12, 106, 6))
PATCH, STRIDE = 320, 160

import clip as clip_pkg
CM, _ = clip_pkg.load("ViT-B/32", device=DEV)
CMEAN = torch.tensor([0.4815, 0.4578, 0.4082], device=DEV).view(1, 3, 1, 1)
CSTD = torch.tensor([0.2686, 0.2613, 0.2758], device=DEV).view(1, 3, 1, 1)
try:
    DINO = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(DEV).eval()
    DMEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
    DSTD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
    print("[dino] loaded dinov2_vits14")
except Exception as e:
    DINO = None
    print(f"[dino] unavailable ({str(e)[:80]}) -> CLIP only")


def unfold(t):
    return F.unfold(t, PATCH, stride=STRIDE).transpose(1, 2).reshape(-1, 3, PATCH, PATCH)


def feats(ps, model):
    if model == "clip":
        v = F.interpolate(ps, size=224, mode="bilinear", align_corners=False)
        with torch.no_grad():
            e = CM.encode_image((v - CMEAN) / CSTD)
    else:
        v = F.interpolate(ps, size=224, mode="bilinear", align_corners=False)
        with torch.no_grad():
            e = DINO((v - DMEAN) / DSTD)
    return (e / e.norm(dim=-1, keepdim=True)).float()


def grad_weight(ps):
    """Relative gradient energy per patch (blur -> ~0)."""
    g = ps.mean(1, keepdim=True)
    lap = F.conv2d(g, torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], device=DEV, dtype=torch.float32), padding=1)
    lv = lap.var(dim=(1, 2, 3))
    return (lv / (lv.median() + 1e-8)).clamp(0, 1.5).cpu().numpy()


def warp_residual(prev_g, cur_g, det, bf):
    k0, d0 = det.detectAndCompute(prev_g, None); kt, dt = det.detectAndCompute(cur_g, None)
    if d0 is None or dt is None:
        return None
    good = [a for a, b in bf.knnMatch(d0, dt, k=2) if a.distance < 0.75 * b.distance]
    if len(good) < 8:
        return None
    src = np.float32([k0[x.queryIdx].pt for x in good]).reshape(-1, 1, 2)
    dst = np.float32([kt[x.trainIdx].pt for x in good]).reshape(-1, 1, 2)
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3)
    if M is None:
        return None
    H, W = cur_g.shape
    warped = cv2.warpAffine(prev_g, M, (W, H))
    valid = cv2.warpAffine(np.ones_like(prev_g), M, (W, H)) > 0
    def gmag(x):
        gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, 3); gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, 3)
        return np.sqrt(gx ** 2 + gy ** 2)
    a, b = gmag(warped), gmag(cur_g)
    mu_a, mu_b = cv2.blur(a, (15, 15)), cv2.blur(b, (15, 15))
    va = cv2.blur(a * a, (15, 15)) - mu_a ** 2; vb = cv2.blur(b * b, (15, 15)) - mu_b ** 2
    cab = cv2.blur(a * b, (15, 15)) - mu_a * mu_b
    C = (255 * 0.03) ** 2
    ssim = ((2 * mu_a * mu_b + C) * (2 * cab + C)) / ((mu_a ** 2 + mu_b ** 2 + C) * (va + vb + C) + 1e-8)
    res = (1 - ssim) * valid
    # patch-p90 over 160px cells inside valid area
    cells = []
    for y in range(0, H - 160, 80):
        for x in range(0, W - 160, 80):
            v = valid[y:y + 160, x:x + 160]
            if v.mean() > 0.9:
                cells.append(res[y:y + 160, x:x + 160].mean())
    return float(np.percentile(cells, 90)) if cells else None


def main():
    det = cv2.SIFT_create(); bf = cv2.BFMatcher()
    models = ["clip"] + (["dino"] if DINO is not None else [])
    rows = []
    for run in RUNS:
        p = f"logs/eval_final/A/{run}/control_test/step05000_r08_{os.environ.get('MI_BRANCH','BL')}_raw.mp4"
        r = imageio.get_reader(p)
        frames = {i: np.asarray(r.get_data(i)) for i in SEED_IDX + GEN_IDX}
        r.close()
        t = lambda i: torch.tensor(frames[i]).permute(2, 0, 1).unsqueeze(0).float().to(DEV) / 255.
        row = dict(run=run)

        def aug_views(x):
            outs = [x]
            for k in (9, 21, 41):
                w = torch.ones(3, 1, k, k, device=DEV) / (k * k)
                outs.append(F.conv2d(x, w, padding=k // 2, groups=3))
            outs.append((x * 0.7).clamp(0, 1))
            outs.append((x * 1.3).clamp(0, 1))
            return outs

        for m in models:
            bank = torch.cat([feats(unfold(v), m) for i in SEED_IDX for v in aug_views(t(i))], 0)
            scores = []
            for i in GEN_IDX:
                ps = unfold(t(i))
                e = feats(ps, m)
                d = (1 - e @ bank.T).min(1).values.cpu().numpy()
                w = grad_weight(ps)
                scores.append((float(np.percentile(d, 95)), float(np.percentile(d * w, 95))))
            row[f"nov_{m}_max"] = round(max(s[0] for s in scores), 4)
            row[f"nov_{m}_mean"] = round(float(np.mean([s[0] for s in scores])), 4)
            row[f"gnov_{m}_max"] = round(max(s[1] for s in scores), 4)
            row[f"gnov_{m}_mean"] = round(float(np.mean([s[1] for s in scores])), 4)
        wr = []
        gs = {i: cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY) for i in GEN_IDX}
        gi = GEN_IDX
        for a, b in zip(gi[:-1], gi[1:]):
            v = warp_residual(gs[a], gs[b], det, bf)
            if v is not None:
                wr.append(v)
        row["warp_max"] = round(max(wr), 4) if wr else np.nan
        row["warp_mean"] = round(float(np.mean(wr)), 4) if wr else np.nan
        rows.append(row)
        print(row, flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"analysis/eval_final/mangle_iter5_r08_{os.environ.get('MI_BRANCH','BL')}.csv", index=False)

    print("\n=== VALIDATION (BL): mangled {pca4,pca2,16node,noatok} vs clean {pca8_8node,4node} ===")
    for col in [c for c in df.columns if c != "run"]:
        o = df.sort_values(col)
        order = list(o.run)
        hi = set(order[-4:]) == set(RUNS) - CLEAN
        lo = set(order[:4]) == set(RUNS) - CLEAN
        v = "PASS (mangled high)" if hi else ("PASS (mangled low)" if lo else "fail")
        print(f"  {col:15}: " + " ".join(f"{r}:{x}" for r, x in zip(o.run, o[col])) + f" -> {v}")


if __name__ == "__main__":
    main()
