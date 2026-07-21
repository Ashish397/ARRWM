"""Mangle iter 6: (a) DIAGNOSE pca8's high B score, (b) new AE / learned-loss metrics.

(a) For each model on r08 B: per-frame nov_dino p95 trajectory; save the argmax
    frame with the top-3 most-novel patches boxed -> analysis/eval_final/_diag/
    nov_argmax_{run}_B.png. Answers whether pca8's 0.400 is a real artifact moment
    or a false positive.

(b) New candidates (patch-level, time-max over rollout, both branches BL+B):
    1. sd_resid : SDEdit-style natural-image-prior residual. Add noise at t~0.3 to
       the frame, one-step denoise with SD2.1-base UNet (prompt ""), decode,
       LPIPS-spatial residual vs original -> patch p90. Melted geometry is off the
       natural manifold -> the prior "corrects" it -> high residual.
    2. mae_err  : ViT-MAE-base masked reconstruction error. 4 random 75% masks,
       mean per-patch MSE where masked (normalized by patch var) on 224 tiles ->
       p90 over tiles. Impossible geometry is unpredictable from context.
    3. dists_ae : AEROBLADE with DISTS instead of LPIPS2 (SD-VAE recon).
    4. liqe_mix / arniqa / ilniqe : newer learned NR-IQA, frame-level, time-max
       of (worst) score.

Validation: BL clean={pca8_8node,4node}; B: 16node must be worst AND pca8 must be
mid/low (user: 'hardly bad'), noatok lowest-ish.
"""
import os
import numpy as np, pandas as pd, imageio, torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RUNS = ["pca8_8node", "pca4", "pca2", "16node", "4node", "noatok"]
CLEAN_BL = {"pca8_8node", "4node"}
SEED_IDX = [0, 3, 6, 9]
GEN_IDX = list(range(12, 106, 6))
PATCH, STRIDE = 320, 160
OUTD = "analysis/eval_final"
os.makedirs(f"{OUTD}/_diag", exist_ok=True)

# ---------- shared ----------
def unfold(t, p=PATCH, s=STRIDE):
    return F.unfold(t, p, stride=s).transpose(1, 2).reshape(-1, 3, p, p)

def patch_grid_coords(H, W, p=PATCH, s=STRIDE):
    ys = list(range(0, H - p + 1, s)); xs = list(range(0, W - p + 1, s))
    return [(y, x) for y in ys for x in xs], len(xs)

DINO = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(DEV).eval()
DMEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
DSTD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)

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

def load_frames(run, br):
    r = imageio.get_reader(f"logs/eval_final/A/{run}/control_test/step05000_r08_{br}_raw.mp4")
    fr = {i: np.asarray(r.get_data(i)) for i in SEED_IDX + GEN_IDX}
    r.close(); return fr

def t_of(fr, i):
    return torch.tensor(fr[i]).permute(2, 0, 1).unsqueeze(0).float().to(DEV) / 255.

# ---------- (a) diagnose nov_dino argmax ----------
def diagnose(br="B"):
    print(f"=== (a) nov_dino diagnosis on {br}: per-frame p95 + argmax frame w/ top patches ===")
    for run in RUNS:
        fr = load_frames(run, br)
        bank = torch.cat([dino_feats(unfold(v)) for i in SEED_IDX for v in aug_views(t_of(fr, i))], 0)
        traj, best = [], (-1, None, None)
        for i in GEN_IDX:
            ps = unfold(t_of(fr, i))
            d = (1 - dino_feats(ps) @ bank.T).min(1).values.cpu().numpy()
            p95 = float(np.percentile(d, 95))
            traj.append((i, round(p95, 3)))
            if p95 > best[0]:
                best = (p95, i, d)
        p95m, fi, d = best
        H, W = fr[fi].shape[:2]
        coords, _ = patch_grid_coords(H, W)
        im = Image.fromarray(fr[fi]); dr = ImageDraw.Draw(im)
        for idx in np.argsort(d)[-3:]:
            y, x = coords[idx]
            dr.rectangle([x, y, x + PATCH, y + PATCH], outline=(255, 255, 0), width=4)
            dr.text((x + 6, y + 6), f"{d[idx]:.2f}", fill=(255, 255, 0))
        dr.text((6, 4), f"{run} {br} f{fi} p95={p95m:.3f}", fill=(255, 0, 0))
        im.save(f"{OUTD}/_diag/nov_argmax_{run}_{br}.png")
        print(f"  {run:11} argmax f{fi} p95={p95m:.3f} | traj={traj}")

# ---------- (b) new metrics ----------
def build_sd():
    from diffusers import StableDiffusionImg2ImgPipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16, safety_checker=None,
        requires_safety_checker=False).to(DEV)
    pipe.set_progress_bar_config(disable=True)
    return pipe

def main():
    diagnose("B")

    import pyiqa, lpips as lpips_pkg
    LPs = lpips_pkg.LPIPS(net="vgg", spatial=True).to(DEV).eval()
    DISTS = pyiqa.create_metric("dists", device=DEV)
    NR = {}
    for name in ("liqe_mix", "arniqa", "ilniqe"):
        try:
            NR[name] = pyiqa.create_metric(name, device=DEV)
            print(f"[nr] {name} loaded")
        except Exception as e:
            print(f"[nr] {name} SKIP {str(e)[:60]}")
    from diffusers import AutoencoderKL
    ae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEV).eval().requires_grad_(False)
    try:
        sd = build_sd(); print("[sd] img2img pipe loaded")
    except Exception as e:
        sd = None; print(f"[sd] SKIP {str(e)[:80]}")
    try:
        from transformers import ViTMAEForPreTraining, AutoImageProcessor
        mae = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base").to(DEV).eval()
        mae.config.mask_ratio = 0.75
        print("[mae] vit-mae-base loaded")
    except Exception as e:
        mae = None; print(f"[mae] SKIP {str(e)[:80]}")

    def sd_resid(t):
        if sd is None:
            return np.nan
        img = F.interpolate(t, size=(512, 768), mode="bilinear", align_corners=False)
        with torch.no_grad():
            out = sd(prompt="", image=img.half(), strength=0.25, num_inference_steps=8,
                     guidance_scale=1.0, output_type="pt").images
            rec = F.interpolate(out.float(), size=t.shape[-2:], mode="bilinear", align_corners=False)
            m = LPs(t * 2 - 1, rec.clamp(0, 1) * 2 - 1)
        pm = F.avg_pool2d(m, 160, 80).flatten().cpu().numpy()
        return float(np.percentile(pm, 90))

    def mae_err(t):
        if mae is None:
            return np.nan
        tiles = F.interpolate(unfold(t, 320, 256), size=224, mode="bilinear", align_corners=False)
        tiles = (tiles - DMEAN) / DSTD
        errs = []
        with torch.no_grad():
            for _ in range(3):
                out = mae(pixel_values=tiles)
                pred = mae.unpatchify(out.logits)
                tgt = tiles
                mask = out.mask.unsqueeze(-1)                       # [B, L, 1]
                pp = mae.patchify(pred); tp = mae.patchify(tgt)
                e = ((pp - tp) ** 2).mean(-1)                       # [B, L]
                e = (e * out.mask).sum(1) / out.mask.sum(1)         # masked-only mean per tile
                errs.append(e.cpu().numpy())
        e = np.mean(errs, 0)
        return float(np.percentile(e, 90))

    def dists_ae(t):
        with torch.no_grad():
            lat = ae.encode(t * 2 - 1).latent_dist.mode()
            rec = ((ae.decode(lat).sample + 1) / 2).clamp(0, 1)
        return float(DISTS(t, rec).item())

    rows = []
    for br in ("BL", "B"):
        for run in RUNS:
            fr = load_frames(run, br)
            vals = {k: [] for k in ("sd", "mae", "dists")}
            nrv = {k: [] for k in NR}
            for i in GEN_IDX[::2]:                                  # every 12th frame (8 frames)
                t = t_of(fr, i)
                vals["sd"].append(sd_resid(t)); vals["mae"].append(mae_err(t)); vals["dists"].append(dists_ae(t))
                for k, m in NR.items():
                    try:
                        nrv[k].append(float(m(t).item()))
                    except Exception:
                        pass
            row = dict(branch=br, run=run,
                       sd_max=round(np.nanmax(vals["sd"]), 4), sd_mean=round(float(np.nanmean(vals["sd"])), 4),
                       mae_max=round(np.nanmax(vals["mae"]), 4),
                       dists_max=round(np.nanmax(vals["dists"]), 4))
            for k in NR:
                if nrv[k]:
                    lower_better = k == "ilniqe"
                    row[f"{k}_worst"] = round(max(nrv[k]) if lower_better else min(nrv[k]), 3)
            rows.append(row); print(row, flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"{OUTD}/mangle_iter6_r08.csv", index=False)

    print("\n=== VALIDATION ===")
    for br, note in [("BL", "mangled {pca4,pca2,16node,noatok} vs clean {pca8,4node}"),
                     ("B", "16node worst AND pca8 mid/low AND noatok low")]:
        s = df[df.branch == br]
        print(f" [{br}] {note}")
        for col in [c for c in s.columns if c not in ("branch", "run")]:
            o = s.sort_values(col)
            order = list(o.run)
            if br == "BL":
                ok = set(order[-4:]) == set(RUNS) - CLEAN_BL or set(order[:4]) == set(RUNS) - CLEAN_BL
            else:
                ok = (order[-1] == "16node" and order.index("pca8_8node") <= 3)
            print(f"   {col:12}: " + " ".join(f"{r}:{x}" for r, x in zip(o.run, o[col])) + ("  -> PASS" if ok else "  -> fail"))


if __name__ == "__main__":
    main()
