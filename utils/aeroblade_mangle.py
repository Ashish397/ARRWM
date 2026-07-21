"""AEROBLADE-style mangle detector on r08 (Part 3).

AEROBLADE (Ricker et al. 2024): reconstruct an image through pretrained latent-
diffusion autoencoders and measure perceptual (LPIPS layer-2) reconstruction error.
Generated/mangled content reconstructs EASIER (lower error) than natural content.

We run it on r08 BL (heavy mangle: by-eye ONLY pca8_8node + 4node are clean) and B.
Per model: AE-recon error on first frames (real seed, 0/3/6/9) vs last frames
(generated, 96/99/102/105), min over {SD1.5-VAE, SD2.1-VAE}, plus NIQE secondary.

Validation target on BL last-frames: {pca4, pca2, 16node, noatok} score LOWER
(more generated/mangled) than {pca8_8node, 4node}.
Saves analysis/eval_final/aeroblade_r08.csv
"""
import os
import numpy as np, pandas as pd, imageio, torch
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RUNS = ["pca8_8node", "pca4", "pca2", "16node", "4node", "noatok"]
BRANCHES = ["BL", "B"]
FIRST = [0, 3, 6, 9]
LAST = [96, 99, 102, 105]
CLEAN_EXPECT = {"pca8_8node", "4node"}

from diffusers import AutoencoderKL
AES = {}
for name, repo, sub in [("sd15", "stabilityai/sd-vae-ft-mse", None),
                        ("sd21", "stabilityai/stable-diffusion-2-1", "vae")]:
    try:
        ae = (AutoencoderKL.from_pretrained(repo, subfolder=sub) if sub
              else AutoencoderKL.from_pretrained(repo)).to(DEV).eval().requires_grad_(False)
        AES[name] = ae
        print(f"[ae] loaded {name}")
    except Exception as e:
        print(f"[ae] {name} FAILED: {str(e)[:120]}")
assert AES, "no autoencoder loaded"

# LPIPS layer-2 (AEROBLADE default); fall back to full LPIPS if lpips pkg missing
try:
    import lpips as lpips_pkg
    LP = lpips_pkg.LPIPS(net="vgg").to(DEV).eval()
    def lpips2(a, b):
        with torch.no_grad():
            _, per = LP(a * 2 - 1, b * 2 - 1, retPerLayer=True)
        return float(per[1].mean().item())
    print("[lpips] vgg layer-2 (AEROBLADE default)")
except Exception as e:
    import pyiqa
    _m = pyiqa.create_metric("lpips", device=DEV)
    def lpips2(a, b):
        return float(_m(a, b).item())
    print(f"[lpips] pkg missing ({str(e)[:60]}) -> pyiqa full LPIPS fallback")

import pyiqa
NIQE = pyiqa.create_metric("niqe", device=DEV)


def load(run, br, idxs):
    r = imageio.get_reader(f"logs/eval_final/A/{run}/control_test/step05000_r08_{br}_raw.mp4")
    out = []
    for i in idxs:
        f = np.asarray(r.get_data(i))
        t = torch.tensor(f).permute(2, 0, 1).unsqueeze(0).float().to(DEV) / 255.
        out.append(F.interpolate(t, size=(512, 512), mode="bilinear", align_corners=False))
    r.close(); return out


def aeroblade(t):
    errs = []
    for ae in AES.values():
        with torch.no_grad():
            lat = ae.encode(t * 2 - 1).latent_dist.mode()
            rec = (ae.decode(lat).sample + 1) / 2
        errs.append(lpips2(t.clamp(0, 1), rec.clamp(0, 1)))
    return min(errs)


def main():
    rows = []
    for br in BRANCHES:
        for run in RUNS:
            try:
                first, last = load(run, br, FIRST), load(run, br, LAST)
            except Exception as e:
                print(f"skip {run} {br}: {e}"); continue
            a0 = float(np.mean([aeroblade(t) for t in first]))
            aT = float(np.mean([aeroblade(t) for t in last]))
            n0 = float(np.mean([NIQE(t).item() for t in first]))
            nT = float(np.mean([NIQE(t).item() for t in last]))
            rows.append(dict(branch=br, run=run, aero_first=round(a0, 4), aero_last=round(aT, 4),
                             aero_delta=round(aT - a0, 4), niqe_first=round(n0, 2),
                             niqe_last=round(nT, 2)))
            print(f"[{br}] {run:11} aero first={a0:.4f} last={aT:.4f} d={aT-a0:+.4f} | niqe {n0:.2f}->{nT:.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("analysis/eval_final/aeroblade_r08.csv", index=False)

    print("\n=== VALIDATION on BL last frames (lower aero = more generated/mangled) ===")
    s = df[df.branch == "BL"].sort_values("aero_last")
    print(s[["run", "aero_first", "aero_last", "aero_delta", "niqe_last"]].to_string(index=False))
    order = list(s.run)
    mangled_rank = [r for r in order if r not in CLEAN_EXPECT]
    ok_last = all(order.index(m) < order.index(c) for m in mangled_rank[:4] for c in CLEAN_EXPECT if c in order) \
        if len(order) == 6 else False
    print(f"\n absolute-last separates clean {sorted(CLEAN_EXPECT)} on top? {'YES' if ok_last else 'NO'}")
    s2 = df[df.branch == "BL"].sort_values("aero_delta")
    print("\n by DELTA (most-negative drop = got more 'generated' over rollout):")
    print(s2[["run", "aero_delta"]].to_string(index=False))


if __name__ == "__main__":
    main()
