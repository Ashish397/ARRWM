"""PAL4VST (ICCV23) artifact-fraction mangle judge — TorchScript, no mmseg.

Model: swin-large+upernet unified artifact segmenter (512x512 in, binary mask
out). Per frame: run on two 512x512 crops (left/right halves of the 832x480
frame upscaled), score = artifact-pixel fraction. Video score = mean over
sampled generated frames + max.

Validation targets (user ground truth):
  r08_B : 16node worst BY FAR; pca8 hardly bad
  r08_BL: pca8 best, pca2 2nd, noadaln very good; pca4/16node/4node/noatok VERY mangled
  real refs: clean floor

Env: PV_WINDOWS, PV_RUNS, PV_QSTRIDE (def 8), PV_OUT.
"""
import os, glob
import numpy as np
import pandas as pd
import imageio
import torch
import torch.nn.functional as F

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
DEV = "cuda"
TS = f"{ARR}/third_party/PAL4VST/deployment/pal4vst/swin-large_upernet_unified_512x512/end2end.pt"
WINDOWS = os.environ.get("PV_WINDOWS", "r08_B:r08_BL:r01_R").split(":")
RUNS = os.environ.get("PV_RUNS", "pca8_8node:pca4:pca2:16node:4node:noatok:noadaln").split(":")
QSTRIDE = int(os.environ.get("PV_QSTRIDE", "8"))
OUT = os.environ.get("PV_OUT", f"{ARR}/analysis/eval_final/pal4vst.csv")

model = torch.jit.load(TS).to(DEV).eval()
MEAN = torch.tensor([123.675, 116.28, 103.53], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([58.395, 57.12, 57.375], device=DEV).view(1, 3, 1, 1)


def frames(path, stride, start=0):
    r = imageio.get_reader(path)
    out = [np.asarray(f) for i, f in enumerate(r) if i >= start and (i - start) % stride == 0]
    r.close()
    return out


@torch.no_grad()
def frame_score(img):
    """832x480 -> two 512x512 tiles (left/right, resized from 416x480) -> artifact fraction."""
    t = torch.from_numpy(img).permute(2, 0, 1)[None].to(DEV).float()
    topcut = int(t.shape[-2] * float(os.environ.get("PV_TOPCUT", "0.15")))
    t = t[..., topcut:, :]              # drop top haze band
    H0, W0 = t.shape[-2:]
    tiles = [t[..., :, :W0 // 2], t[..., :, W0 // 2:]]
    fracs = []
    for tile in tiles:
        x = F.interpolate(tile, size=(512, 512), mode="bilinear", align_corners=False)
        x = (x - MEAN) / STD
        out = model(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        if out.dim() == 4 and out.shape[1] > 1:
            mask = out.argmax(1)
        else:
            mask = (out.squeeze(1) > 0.5).long()
        fracs.append(float(mask.float().mean().item()))
    return float(np.mean(fracs))


def video_score(path, start):
    fr = frames(path, QSTRIDE, start=start)
    if not fr:
        return None
    s = [frame_score(f) for f in fr]
    return float(np.median(s)), float(np.mean(s)), len(s)


def main():
    rows = []
    # PV_VIDS mode: "label=path:label=path" scored directly (plus real floor)
    if os.environ.get("PV_VIDS"):
        for item in os.environ["PV_VIDS"].split(":"):
            label, path = item.split("=", 1)
            w, run = label.split("|", 1)
            m, mx, n = video_score(path, start=13)
            rows.append(dict(window=w, run=run, art_med=round(m, 4), art_mean=round(mx, 4)))
            print(f"[pal] {w} {run}: med={m:.4f} mean={mx:.4f}", flush=True)
        import pandas as _pd
        _pd.DataFrame(rows).to_csv(OUT, index=False)
        print("done (PV_VIDS mode)")
        return
    for w in WINDOWS:
        for run in RUNS:
            p = f"{ARR}/logs/eval_final/A/{run}/control_test/step05000_{w}_raw.mp4"
            if not os.path.exists(p):
                continue
            m, mx, n = video_score(p, start=13)
            rows.append(dict(window=w, run=run, art_med=round(m, 4), art_mean=round(mx, 4)))
            print(f"[pal] {w} {run}: mean={m:.4f} max={mx:.4f}", flush=True)
        mw = {"r08_B": f"{ARR}/logs/eval_final/A_minwm/minwm_r08_B.mp4",
              "r08_BL": f"{ARR}/logs/eval_final/A_minwm/minwm_r08_BR.mp4",
              "r01_R": f"{ARR}/logs/eval_final/A_minwm/minwm_r01_L.mp4"}.get(w)
        if mw and os.path.exists(mw):
            m, mx, n = video_score(mw, start=13)
            rows.append(dict(window=w, run="minwm", art_med=round(m, 4), art_mean=round(mx, 4)))
            print(f"[pal] {w} minwm: mean={m:.4f} max={mx:.4f}", flush=True)
    scored = 0
    for p in sorted(glob.glob(f"{ARR}/analysis/eval_final/real_refs/*.mp4")):
        if scored >= 8:
            break
        r = imageio.get_reader(p); f0 = np.asarray(r.get_data(5)); r.close()
        if float(f0.mean()) < 60:      # night clip: eval floor is daytime-only
            print(f"[pal] skip night ref {os.path.basename(p)} (luma {f0.mean():.0f})", flush=True)
            continue
        scored += 1
        m, mx, n = video_score(p, start=0)
        rows.append(dict(window="REAL", run=os.path.basename(p)[:24], art_med=round(m, 4), art_mean=round(mx, 4)))
        print(f"[pal] REAL {os.path.basename(p)}: mean={m:.4f} max={mx:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print("\n=== per-window ranking (art_mean desc) ===")
    for w in WINDOWS:
        sub = df[df.window == w].sort_values("art_med", ascending=False)
        print(w, ":", " ".join(f"{r.run}:{r.art_med:.3f}" for r in sub.itertuples()))


if __name__ == "__main__":
    main()
