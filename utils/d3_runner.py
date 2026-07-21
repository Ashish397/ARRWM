"""D3 (ICCV'25, training-free) on our eval videos — faithful to Zig-HS/D3 defaults:
XCLIP-16 encoder, frames resized 224x224 (no crop), ImageNet norm, 16 frames,
loss=l2: score = std of second-order differences of consecutive-frame feature
distances (higher = more 'AI-generated' per the paper).

Per video: FIRST 16 frames vs LAST 16 frames (+ optional real reference clips in
analysis/eval_final/real_refs/*.mp4 as calibration anchors).
Videos: r01_R (blind window) + r08 B / BL anchors, all 6 models.
Saves analysis/eval_final/d3_scores.csv
"""
import os, glob
import numpy as np, pandas as pd, imageio, torch
import torch.nn.functional as F
from transformers import XCLIPVisionModel

DEV = "cuda"
RUNS = os.environ.get("D3_RUNS", "pca8_8node,pca4,pca2,16node,4node,noatok").replace(":", ",").split(",")
TARGETS = [(1, "R"), (8, "B"), (8, "BL")]
ENC = XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch16").to(DEV).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)



_MWSW = {"L": "R", "R": "L", "FL": "FR", "FR": "FL", "BL": "BR", "BR": "BL"}
def _vid_path(run, rank, br, arr="."):
    if run == "minwm":   # disk labels are yaw-sign-flipped; swap to reach TRUE direction
        return f"{arr}/logs/eval_final/A_minwm/minwm_r{rank:02d}_{_MWSW.get(br, br)}.mp4"
    return f"{arr}/logs/eval_final/A/{run}/control_test/step05000_r{rank:02d}_{br}_raw.mp4"

def d3_score(frames):
    """frames: list of 16 HWC uint8 -> (dis2nd_avg, dis2nd_std). std = D3 fake score."""
    t = torch.stack([torch.tensor(f).permute(2, 0, 1) for f in frames]).float().to(DEV) / 255.
    t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
    t = (t - MEAN) / STD
    with torch.no_grad():
        out = ENC(t, output_hidden_states=True).pooler_output      # [16, D]
    v1, v2 = out[:-1], out[1:]
    d1 = torch.norm(v1 - v2, p=2, dim=-1)                          # first-order
    d2 = d1[1:] - d1[:-1]                                          # second-order
    return float(d2.mean().item()), float(d2.std().item())


def clips(path):
    r = imageio.get_reader(path)
    fr = [np.asarray(f) for f in r]
    r.close()
    n = len(fr)
    first = [fr[i] for i in np.linspace(0, min(15, n - 1), 16).astype(int)]
    last = [fr[i] for i in np.linspace(max(0, n - 16), n - 1, 16).astype(int)]
    return first, last


def main():
    rows = []
    for rank, br in TARGETS:
        for run in RUNS:
            p = _vid_path(run, rank, br)
            first, last = clips(p)
            _, sf = d3_score(first)
            _, sl = d3_score(last)
            rows.append(dict(window=f"r{rank:02d}_{br}", run=run,
                             d3_first=round(sf, 4), d3_last=round(sl, 4), delta=round(sl - sf, 4)))
            print(rows[-1], flush=True)
    # calibration on REAL reference clips if present
    for p in sorted(glob.glob("analysis/eval_final/real_refs/*.mp4"))[:16]:
        first, last = clips(p)
        _, sf = d3_score(first)
        _, sl = d3_score(last)
        rows.append(dict(window="REAL_ref", run=os.path.basename(p)[:20],
                         d3_first=round(sf, 4), d3_last=round(sl, 4), delta=round(sl - sf, 4)))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows); df.to_csv("analysis/eval_final/d3_scores.csv", index=False)
    print("\n=== ranking by d3_last per window (higher = more AI/mangled per D3) ===")
    for w in df.window.unique():
        s = df[df.window == w].sort_values("d3_last")
        print(f" [{w}] " + " ".join(f"{r}:{x:.3f}" for r, x in zip(s.run, s.d3_last)))


if __name__ == "__main__":
    main()
