"""Patch-LOCAL mangle detectors on r08 (iteration 2 after global AEROBLADE failed).

Mangle is local (melted blobs), so global means dilute it. Three patch-level
candidates, each aggregated over the worst patches of the LAST frames:

  1) AERO-patch : spatial LPIPS(vgg) map between frame and its SD1.5-VAE recon;
                  per-patch means; report p10 (easiest recon = most 'generated')
                  and p90 (hardest) — direction determined empirically.
  2) NIQE-patch : NIQE per 160x160 patch; p90 = worst-patch naturalness.
  3) CLIP-mangle: CLIP ViT-B/32 patch embedding scored against a prompt pair
                  ('a distorted warped melted image' vs 'a sharp photograph of
                  a street'); p90 of the mangled-probability over patches.

Ground truth (r08 BL, by eye): CLEAN={pca8_8node,4node}; MANGLED={pca4,pca2,16node,noatok}.
A metric PASSES if it ranks all 4 mangled above both clean (or vice versa, consistently).
Saves analysis/eval_final/patch_mangle_r08.csv
"""
import os
import numpy as np, pandas as pd, imageio, torch
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RUNS = ["pca8_8node", "pca4", "pca2", "16node", "4node", "noatok"]
CLEAN = {"pca8_8node", "4node"}
BRANCHES = ["BL", "B"]
FIRST = [0, 3, 6, 9]
LAST = [96, 99, 102, 105]
PATCH, STRIDE = 160, 80


def load(run, br, idxs):
    r = imageio.get_reader(f"logs/eval_final/A/{run}/control_test/step05000_r08_{br}_raw.mp4")
    out = [torch.tensor(np.asarray(r.get_data(i))).permute(2, 0, 1).unsqueeze(0).float().to(DEV) / 255. for i in idxs]
    r.close(); return out


def patches(t):
    return F.unfold(t, PATCH, stride=STRIDE).transpose(1, 2).reshape(-1, 3, PATCH, PATCH)


from diffusers import AutoencoderKL
ae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEV).eval().requires_grad_(False)
import lpips as lpips_pkg
LP = lpips_pkg.LPIPS(net="vgg", spatial=True).to(DEV).eval()
import pyiqa
NIQE = pyiqa.create_metric("niqe", device=DEV)
import clip as clip_pkg
CM, CPre = clip_pkg.load("ViT-B/32", device=DEV)
TXT = clip_pkg.tokenize(["a distorted warped melted image", "a sharp photograph of a street"]).to(DEV)
with torch.no_grad():
    TF = CM.encode_text(TXT); TF = TF / TF.norm(dim=-1, keepdim=True)


def aero_patch(t):
    with torch.no_grad():
        lat = ae.encode(t * 2 - 1).latent_dist.mode()
        rec = ((ae.decode(lat).sample + 1) / 2).clamp(0, 1)
        m = LP(t * 2 - 1, rec * 2 - 1)                     # [1,1,H,W] spatial LPIPS map
    pm = F.avg_pool2d(m, PATCH, STRIDE).flatten().cpu().numpy()
    return float(np.percentile(pm, 10)), float(np.percentile(pm, 90))


def niqe_patch(t):
    ps = patches(t)
    vals = []
    for i in range(len(ps)):
        try:
            vals.append(float(NIQE(ps[i:i + 1]).item()))
        except Exception:
            pass
    return float(np.percentile(vals, 90)) if vals else np.nan


def clip_patch(t):
    ps = patches(t)
    ps = F.interpolate(ps, size=224, mode="bilinear", align_corners=False)
    mean = torch.tensor([0.4815, 0.4578, 0.4082], device=DEV).view(1, 3, 1, 1)
    std = torch.tensor([0.2686, 0.2613, 0.2758], device=DEV).view(1, 3, 1, 1)
    with torch.no_grad():
        e = CM.encode_image((ps - mean) / std); e = e / e.norm(dim=-1, keepdim=True)
        logits = 100. * e.float() @ TF.float().T
        p_mangle = logits.softmax(-1)[:, 0].cpu().numpy()
    return float(np.percentile(p_mangle, 90))


def main():
    rows = []
    for br in BRANCHES:
        for run in RUNS:
            first, last = load(run, br, FIRST), load(run, br, LAST)
            def agg(frames):
                a10, a90, nq, cl = [], [], [], []
                for t in frames:
                    x, y = aero_patch(t); a10.append(x); a90.append(y)
                    nq.append(niqe_patch(t)); cl.append(clip_patch(t))
                return (np.mean(a10), np.mean(a90), np.nanmean(nq), np.mean(cl))
            f = agg(first); l = agg(last)
            rows.append(dict(branch=br, run=run,
                             aeroP10_last=round(l[0], 4), aeroP90_last=round(l[1], 4),
                             niqeP90_last=round(l[2], 2), clipP90_last=round(l[3], 3),
                             aeroP10_d=round(l[0] - f[0], 4), niqeP90_d=round(l[2] - f[2], 2),
                             clipP90_d=round(l[3] - f[3], 3)))
            print(f"[{br}] {run:11} aeroP10={l[0]:.4f} aeroP90={l[1]:.4f} niqeP90={l[2]:5.2f} clipP90={l[3]:.3f}", flush=True)
    df = pd.DataFrame(rows); df.to_csv("analysis/eval_final/patch_mangle_r08.csv", index=False)

    print("\n=== VALIDATION (BL): does metric separate mangled {pca4,pca2,16node,noatok} from clean {pca8_8node,4node}? ===")
    s = df[df.branch == "BL"]
    for col in ["aeroP10_last", "aeroP90_last", "niqeP90_last", "clipP90_last", "aeroP10_d", "niqeP90_d", "clipP90_d"]:
        o = s.sort_values(col)
        order = list(o.run)
        top4, bot2 = set(order[-4:]), set(order[:2])
        hi_mangled = top4 == (set(RUNS) - CLEAN)
        lo_mangled = bot2 == CLEAN and False  # placeholder for readability
        o2 = s.sort_values(col, ascending=False)
        lo4 = set(list(o2.run)[-4:])
        pass_hi = top4 == set(RUNS) - CLEAN     # mangled highest
        pass_lo = set(order[:4]) == set(RUNS) - CLEAN   # mangled lowest
        verdict = "PASS (mangled high)" if pass_hi else ("PASS (mangled low)" if pass_lo else "fail")
        print(f"  {col:13}: " + " ".join(f"{r}:{v}" for r, v in zip(o.run, o[col])) + f"  -> {verdict}")


if __name__ == "__main__":
    main()
