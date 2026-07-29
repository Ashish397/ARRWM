"""PAL4VST artifact-segmentation geometry judge on the 84 labelled videos (LOCAL).

Adapts utils/pal4vst_judge.py (cluster) to the local tiles. Per generated frame:
832x480 -> drop top 15% haze band -> two 512x512 L/R tiles -> artifact-pixel fraction.
Video score = median/mean/MAX over sampled generated frames (ensemble uses pal4vst_max).

Boundary-clean: samples from frame 13 (stride 8), entirely inside the generated region
under the corrected boundary (generation begins frame 12), so no 9-11 leakage.

Env: PAL_TS (TorchScript path, default ~/end2end.pt), PAL_STRIDE (8), PAL_TOPCUT (0.15),
     PAL_OUT (default out/pal4vst_local.csv).
"""
import os
import numpy as np, imageio, torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda"
TILE_DIRS = [os.path.join(HERE, "tiles_new"), os.path.join(HERE, "tiles")]
LABELS = os.path.join(HERE, "human_tiers.csv")
TS = os.environ.get("PAL_TS", os.path.expanduser("~/end2end.pt"))
STRIDE = int(os.environ.get("PAL_STRIDE", "8"))
START = int(os.environ.get("PAL_START", "13"))
TOPCUT = float(os.environ.get("PAL_TOPCUT", "0.15"))
OUT = os.environ.get("PAL_OUT", os.path.join(HERE, "out", "pal4vst_local.csv"))

model = torch.jit.load(TS).to(DEV).eval()
MEAN = torch.tensor([123.675, 116.28, 103.53], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([58.395, 57.12, 57.375], device=DEV).view(1, 3, 1, 1)


def vid_path(scene, mdl):
    for d in TILE_DIRS:
        p = os.path.join(d, f"{scene}__{mdl}.mp4")
        if os.path.exists(p):
            return p
    return None


def gen_frames(path):
    r = imageio.get_reader(path)
    out = [np.asarray(f) for i, f in enumerate(r) if i >= START and (i - START) % STRIDE == 0]
    r.close()
    return out


@torch.no_grad()
def frame_score(img):
    t = torch.from_numpy(img).permute(2, 0, 1)[None].to(DEV).float()
    tc = int(t.shape[-2] * TOPCUT)
    t = t[..., tc:, :]
    W0 = t.shape[-1]
    fracs = []
    for tile in (t[..., :, :W0 // 2], t[..., :, W0 // 2:]):
        x = F.interpolate(tile, size=(512, 512), mode="bilinear", align_corners=False)
        x = (x - MEAN) / STD
        out = model(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        mask = out.argmax(1) if (out.dim() == 4 and out.shape[1] > 1) else (out.squeeze(1) > 0.5).long()
        fracs.append(float(mask.float().mean().item()))
    return float(np.mean(fracs))


def main():
    import pandas as pd
    lab = pd.read_csv(LABELS)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print(f"[pal] TS={TS} start={START} stride={STRIDE} topcut={TOPCUT}", flush=True)
    rows = []
    for _, r in lab.iterrows():
        vid = vid_path(r.scene, r.model)
        if vid is None:
            print(f"[pal] MISSING {r.scene}__{r.model}", flush=True); continue
        fr = gen_frames(vid)
        s = [frame_score(f) for f in fr]
        med, mean, mx = float(np.median(s)), float(np.mean(s)), float(np.max(s))
        rows.append(dict(scene=r.scene, model=r.model, tier=r.tier, note=r.note,
                         pal_median=med, pal_mean=mean, pal_max=mx))
        print(f"[pal] {r.scene:8s} {r.model:8s} tier={r.tier:<4} n={len(s)} "
              f"med={med:.4f} mean={mean:.4f} max={mx:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"[pal] wrote {OUT} ({len(df)} videos)", flush=True)


if __name__ == "__main__":
    main()
