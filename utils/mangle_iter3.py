"""Mangle detector iteration 3 (r08 BL): semantic-novelty + contextual CLIP + time-max.

Learned from iter1/2: mangle = structural surrealism (melted canopy in pca4), often
smooth/bright -> statistically invisible to NIQE/BRISQUE/LPIPS-recon; small patches
lose context (melt patch ~ sky). New candidates, all per-frame image metrics scanned
over the WHOLE rollout (score = max over time = worst moment):

  A) seed-novelty : CLIP ViT-B/32 features of 320px patches (stride 160); for each
     generated-frame patch, cosine distance to the NEAREST patch of the SAME video's
     first 4 frames (the seed vocabulary). Frame score = p95 over patches.
     Structural melt is unlike any seed patch; newly-revealed normal content still
     matches the street vocabulary.
  B) ctx-clip     : contextual prompt ensemble on 320px patches + top-half + full
     frame: P("melted/distorted/surreal") via paired prompts, frame score = max.
  C) aero-time    : aeroP10 (easiest-recon patch decile) scanned over all frames,
     score = min over time (peak 'generatedness' moment).

Ground truth BL: CLEAN={pca8_8node,4node}, MANGLED={pca4,pca2,16node,noatok}.
Saves analysis/eval_final/mangle_iter3_r08.csv
"""
import os
import numpy as np, pandas as pd, imageio, torch
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RUNS = ["pca8_8node", "pca4", "pca2", "16node", "4node", "noatok"]
CLEAN = {"pca8_8node", "4node"}
SEED_IDX = [0, 3, 6, 9]
GEN_IDX = list(range(12, 106, 6))
PATCH, STRIDE = 320, 160

import clip as clip_pkg
CM, _ = clip_pkg.load("ViT-B/32", device=DEV)
MEAN = torch.tensor([0.4815, 0.4578, 0.4082], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.2686, 0.2613, 0.2758], device=DEV).view(1, 3, 1, 1)
POS = ["a melted distorted building", "a surreal warped structure", "a glitched deformed object",
       "an impossible dripping shape", "corrupted image artifacts"]
NEG = ["a photograph of a street", "a building facade", "a paved road", "trees and sky",
       "a normal outdoor scene"]
TXT = clip_pkg.tokenize(POS + NEG).to(DEV)
with torch.no_grad():
    TF = CM.encode_text(TXT); TF = TF / TF.norm(dim=-1, keepdim=True)

from diffusers import AutoencoderKL
ae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEV).eval().requires_grad_(False)
import lpips as lpips_pkg
LP = lpips_pkg.LPIPS(net="vgg", spatial=True).to(DEV).eval()


def clip_feats(t):
    """320px patches + top half + full frame -> normalized CLIP features [N,512]."""
    ps = F.unfold(t, PATCH, stride=STRIDE).transpose(1, 2).reshape(-1, 3, PATCH, PATCH)
    tops = t[:, :, : t.shape[2] // 2, :]
    views = [F.interpolate(x, size=224, mode="bilinear", align_corners=False)
             for x in (ps, tops, t)]
    v = torch.cat(views, 0)
    with torch.no_grad():
        e = CM.encode_image((v - MEAN) / STD)
    return e / e.norm(dim=-1, keepdim=True)


def frame_scores(t, seed_bank):
    e = clip_feats(t).float()
    # A) novelty vs seed vocabulary (patches only, exclude the 2 global views)
    pe = e[:-2]
    d = 1 - (pe @ seed_bank.T)                    # cosine distance to every seed patch
    novelty = float(np.percentile(d.min(1).values.cpu().numpy(), 95))
    # B) contextual mangle probability (all views)
    logits = 100. * e @ TF.float().T
    p = logits.softmax(-1)
    p_mangle = p[:, :len(POS)].sum(-1)
    ctx = float(p_mangle.max().item())
    # C) aero patch p10
    with torch.no_grad():
        lat = ae.encode(t * 2 - 1).latent_dist.mode()
        rec = ((ae.decode(lat).sample + 1) / 2).clamp(0, 1)
        m = LP(t * 2 - 1, rec * 2 - 1)
    pm = F.avg_pool2d(m, 160, 80).flatten().cpu().numpy()
    aero10 = float(np.percentile(pm, 10))
    return novelty, ctx, aero10


def main():
    rows = []
    for run in RUNS:
        p = f"logs/eval_final/A/{run}/control_test/step05000_r08_BL_raw.mp4"
        r = imageio.get_reader(p)
        def grab(i):
            return torch.tensor(np.asarray(r.get_data(i))).permute(2, 0, 1).unsqueeze(0).float().to(DEV) / 255.
        seed_bank = torch.cat([clip_feats(grab(i))[:-2] for i in SEED_IDX], 0).float()
        nov, ctx, aer = [], [], []
        for i in GEN_IDX:
            try:
                a, b, c = frame_scores(grab(i), seed_bank)
            except Exception:
                continue
            nov.append(a); ctx.append(b); aer.append(c)
        r.close()
        rows.append(dict(run=run, novelty_max=round(max(nov), 4), novelty_mean=round(float(np.mean(nov)), 4),
                         ctx_max=round(max(ctx), 4), ctx_mean=round(float(np.mean(ctx)), 4),
                         aero10_min=round(min(aer), 4)))
        print(f"{run:11} novelty max={max(nov):.4f} mean={np.mean(nov):.4f} | ctx max={max(ctx):.4f} "
              f"mean={np.mean(ctx):.4f} | aero10 min={min(aer):.4f}", flush=True)
    df = pd.DataFrame(rows); df.to_csv("analysis/eval_final/mangle_iter3_r08.csv", index=False)

    print("\n=== VALIDATION (BL): mangled {pca4,pca2,16node,noatok} must separate from clean {pca8_8node,4node} ===")
    for col in ["novelty_max", "novelty_mean", "ctx_max", "ctx_mean", "aero10_min"]:
        o = df.sort_values(col)
        order = list(o.run)
        hi = set(order[-4:]) == set(RUNS) - CLEAN
        lo = set(order[:4]) == set(RUNS) - CLEAN
        v = "PASS (mangled high)" if hi else ("PASS (mangled low)" if lo else "fail")
        print(f"  {col:12}: " + " ".join(f"{r}:{x}" for r, x in zip(o.run, o[col])) + f" -> {v}")


if __name__ == "__main__":
    main()
