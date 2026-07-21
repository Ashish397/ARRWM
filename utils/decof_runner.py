"""DeCoF (Generated-Video Detection via Frame Consistency) on our eval videos.

Faithful inference per the DeCoF repo (wuwuwuyue/DeCoF) with the re-released
AIGVDBench weights (third_party/DeCoF/DeCoF.tar = torch ckpt of the temporal-ViT
head; backbone = frozen CLIP ViT-L/14):
  clip = 8 frames evenly from a 32-frame span, center square crop, CLIP preprocess
  score = softmax(head(CLIP_feats))[:,1] = P(fake)

Per video we score TWO clips: FIRST 32 frames (seed + earliest gen) vs LAST 32
frames (fully generated, where mangle lives) -> (first, last, delta).
Videos: r01_R (blind test window) + r08 B and BL (known ground truth anchors).
Saves analysis/eval_final/decof_r08_r01.csv
"""
import os, sys
import numpy as np, pandas as pd, imageio, torch
import torch.nn.functional as F

sys.path.insert(0, "third_party/DeCoF/src")
from vit import ViT                      # DeCoF temporal head (matches ckpt keys)
import clip as clip_pkg

DEV = "cuda"
RUNS = ["pca8_8node", "pca4", "pca2", "16node", "4node", "noatok"]
TARGETS = [(1, "R"), (8, "B"), (8, "BL")]

CM, _ = clip_pkg.load("ViT-L/14", device=DEV)
CM.eval()
HEAD = ViT().to(DEV).eval()
ck = torch.load("third_party/DeCoF/DeCoF.tar", map_location="cpu", weights_only=False)
missing, unexpected = HEAD.load_state_dict(ck["model"], strict=True), None
print(f"[decof] head loaded (epoch {ck.get('epoch')})")
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=DEV).view(1, 3, 1, 1)


def clip_score(frames):
    """frames: list of 8 HWC uint8 -> P(fake)."""
    t = torch.stack([torch.tensor(f).permute(2, 0, 1) for f in frames]).float().to(DEV) / 255.
    H, W = t.shape[-2:]
    s = min(H, W)                                    # center square crop
    t = t[..., (H - s) // 2:(H + s) // 2, (W - s) // 2:(W + s) // 2]
    t = F.interpolate(t, size=224, mode="bicubic", align_corners=False)
    t = (t - MEAN) / STD
    with torch.no_grad():
        feats = CM.encode_image(t).float().unsqueeze(0)      # [1,8,768]
        logits = HEAD(feats)
        p = logits.softmax(-1)[0, 1].item()
    return float(p)


def main():
    rows = []
    for rank, br in TARGETS:
        for run in RUNS:
            p = f"logs/eval_final/A/{run}/control_test/step05000_r{rank:02d}_{br}_raw.mp4"
            r = imageio.get_reader(p)
            frames = [np.asarray(f) for f in r]
            r.close()
            n = len(frames)
            first_idx = np.linspace(0, min(31, n - 1), 8).astype(int)
            last_idx = np.linspace(max(0, n - 32), n - 1, 8).astype(int)
            pf = clip_score([frames[i] for i in first_idx])
            pl = clip_score([frames[i] for i in last_idx])
            rows.append(dict(window=f"r{rank:02d}_{br}", run=run,
                             pfake_first=round(pf, 4), pfake_last=round(pl, 4),
                             delta=round(pl - pf, 4)))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("analysis/eval_final/decof_r08_r01.csv", index=False)
    print("\n=== per-window ranking by pfake_last (and delta) ===")
    for w in df.window.unique():
        s = df[df.window == w].sort_values("pfake_last")
        print(f" [{w}] " + " ".join(f"{r}:{x:.3f}(d{d:+.2f})" for r, x, d in zip(s.run, s.pfake_last, s.delta)))


if __name__ == "__main__":
    main()
