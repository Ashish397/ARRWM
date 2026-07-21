"""DINO EDGE novelty for phase-A videos — the periphery counterpart of
suite_scoreA's center_nov, computed with the IDENTICAL recipe so the two are
directly comparable:

  bank   = DINOv2-S embeddings of seed frames (0,3,6,9) + blur/brightness augs
           (whole-frame 320px patches, stride 160 — same bank as center_nov)
  frame  = every 6th frame from index 12
  center = middle [H/4:3H/4, W/3:2W/3] crop  -> min cos dist to bank (existing)
  EDGE   = four periphery strips (left/right W/6 full-height, top/bottom H/6
           full-width) -> per strip min cos dist to bank; frame score = max
           strip; video score = max over frames  (same max-over-time as center)

High edge novelty is EXPECTED for any moving camera (new content enters at the
periphery); center novelty flags content materializing mid-view. The contrast
is the point.

Env: EN_RUNS colon list, EN_OUT csv (append, resume-safe).
Writes: run,window,dir,edge_nov,edge_nov_meant (mean over frames of max strip).
"""
import os
import numpy as np
import pandas as pd
import imageio
import torch
import torch.nn.functional as F

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
RUNS = os.environ.get("EN_RUNS", "pca8_8node").split(":")
OUT = os.environ.get("EN_OUT", f"{ARR}/analysis/eval_final/edge_novelty.csv")
DIRS = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]
DEV = "cuda"
_MWSW = {"L": "R", "R": "L", "FL": "FR", "FR": "FL", "BL": "BR", "BR": "BL"}


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
PATCH, STRIDE = 320, 160


def unfold(t):
    return F.unfold(t, PATCH, stride=STRIDE).transpose(1, 2).reshape(-1, 3, PATCH, PATCH)


@torch.no_grad()
def dfeat(ps):
    v = F.interpolate(ps, size=224, mode="bilinear", align_corners=False)
    e = dino((v - DM) / DS)
    return (e / e.norm(dim=-1, keepdim=True)).float()


def aug(x):
    outs = [x]
    for k in (9, 21, 41):
        w = torch.ones(3, 1, k, k, device=DEV) / (k * k)
        outs.append(F.conv2d(x, w, padding=k // 2, groups=3))
    outs += [(x * 0.7).clamp(0, 1), (x * 1.3).clamp(0, 1)]
    return outs


def edge_strips(fr):
    H, W = fr.shape[:2]
    return [fr[:, :W // 6], fr[:, 5 * W // 6:], fr[:H // 6, :], fr[5 * H // 6:, :]]


@torch.no_grad()
def edge_scores(frames):
    t = lambda f: torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1)[None].to(DEV).float() / 255.0
    bank = torch.cat([dfeat(unfold(v)) for i in (0, 3, 6, 9) for v in aug(t(frames[i]))], 0)
    per_frame = []
    for i in range(12, len(frames) - 1, 6):
        strips = torch.cat([F.interpolate(t(s), size=224, mode="bilinear", align_corners=False)
                            for s in edge_strips(frames[i])], 0)
        e = dfeat(strips)                                   # [4, D]
        d = (1 - e @ bank.T).min(1).values                  # min dist to bank per strip
        per_frame.append(float(d.max().item()))             # most novel edge region
    if not per_frame:
        return float("nan"), float("nan")
    return float(max(per_frame)), float(np.mean(per_frame))


def main():
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(zip(prev.run, prev.window, prev["dir"]))
    f = open(OUT, "a" if done else "w")
    if not done:
        f.write("run,window,dir,edge_nov,edge_nov_meant\n")
    for run in RUNS:
        for wi in range(32):
            for d in DIRS:
                if (run, f"r{wi:02d}", d) in done:
                    continue
                path = vid_path(run, wi, d)
                if not os.path.exists(path):
                    print(f"[edge] MISSING {path}", flush=True)
                    continue
                try:
                    r = imageio.get_reader(path)
                    frames = [np.asarray(x) for x in r]
                    r.close()
                    mx, mn = edge_scores(frames)
                    f.write(f"{run},r{wi:02d},{d},{mx:.4f},{mn:.4f}\n")
                    f.flush()
                except Exception as e:
                    print(f"[edge] FAIL {run} r{wi:02d} {d}: {e}", flush=True)
        print(f"[edge] {run} done", flush=True)
    f.close()


if __name__ == "__main__":
    main()
