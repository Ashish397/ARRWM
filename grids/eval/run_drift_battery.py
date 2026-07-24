"""Windowed drift features + harsher warp variants.

For each IQA metric in a cheap subset: score on start window (first 16) and end
window (last 16) -> <m>_w_start, <m>_w_end, <m>_w_drift (start-end).
Warping error variants at stride 1: mean, p95 (local worst-case melt), flow-normalized.
End-window Laplacian sharpness and its ratio to start.

Writes results_drift.csv.
"""
import glob, os
import cv2
import numpy as np
import torch
import pandas as pd

DEV = "cuda"
HERE = os.path.dirname(os.path.abspath(__file__))
TILE_DIR = os.path.join(HERE, "tiles")
OUT_CSV = os.path.join(HERE, "results_drift.csv")
W = 16
IQA_METRICS = ["niqe", "unique", "brisque", "clipiqa+", "paq2piq", "arniqa", "hyperiqa", "topiq_nr"]


def read_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


def to01(f):
    return torch.from_numpy(f).permute(0, 3, 1, 2).float().div(255.0)


def flow_warp(img, flow):
    import torch.nn.functional as F
    B, C, H, Wd = img.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=img.device), torch.arange(Wd, device=img.device), indexing="ij")
    grid = torch.stack([xx, yy], 0).float()[None] + flow
    gx = grid[:, 0] / (Wd - 1) * 2 - 1
    gy = grid[:, 1] / (H - 1) * 2 - 1
    return F.grid_sample(img, torch.stack([gx, gy], -1), align_corners=True, padding_mode="border")


@torch.no_grad()
def warp_variants(frames, raft, batch=6, iters=12):
    x = to01(frames)
    xn = x * 2 - 1
    errs_mean, errs_p95, errs_per_flow = [], [], []
    pairs = [(i, i + 1) for i in range(len(x) - 1)]
    for b0 in range(0, len(pairs), batch):
        chunk = pairs[b0 : b0 + batch]
        i1 = torch.stack([xn[i] for i, _ in chunk]).to(DEV)
        i2 = torch.stack([xn[j] for _, j in chunk]).to(DEV)
        fw = raft(i1, i2, num_flow_updates=iters)[-1]
        bw = raft(i2, i1, num_flow_updates=iters)[-1]
        bw_at_fw = flow_warp(bw, fw)
        fb = (fw + bw_at_fw).norm(dim=1)
        mag = fw.norm(dim=1)
        occ = fb > (0.05 * (mag + fb) + 1.0)
        img1 = torch.stack([x[i] for i, _ in chunk]).to(DEV)
        img2 = torch.stack([x[j] for _, j in chunk]).to(DEV)
        err = (flow_warp(img2, fw) - img1).abs().mean(1)
        valid = ~occ
        for b in range(len(chunk)):
            e = err[b][valid[b]]
            if e.numel() < 100:
                continue
            errs_mean.append(float(e.mean()))
            errs_p95.append(float(torch.quantile(e, 0.95)))
            errs_per_flow.append(float(e.mean() / (mag[b][valid[b]].mean() + 0.5)))
    return {
        "warp1_mean": float(np.mean(errs_mean)),
        "warp1_p95": float(np.mean(errs_p95)),
        "warp1_p95_max": float(np.max(errs_p95)),        # worst adjacent pair in the video
        "warp1_per_flow": float(np.mean(errs_per_flow)),
        "warp1_end_mean": float(np.mean(errs_mean[-W:])),  # melt in the late rollout
        "warp1_end_p95": float(np.mean(errs_p95[-W:])),
    }


def sharpness(frames):
    return float(np.mean([cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() for f in frames]))


def main():
    import pyiqa
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    raft = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(DEV).eval()

    files = sorted(glob.glob(os.path.join(TILE_DIR, "*.mp4")))
    vids = {}
    rows = []
    for fp in files:
        name = os.path.basename(fp)[: -len(".mp4")]
        grid, variant = name.split("__")
        frames = read_frames(fp)
        vids[(grid, variant)] = frames
        wv = warp_variants(frames, raft)
        sh_s, sh_e = sharpness(frames[:W:2]), sharpness(frames[-W::2])
        wv["sharp_w_end"] = sh_e
        wv["sharp_w_ratio"] = sh_e / (sh_s + 1e-6)
        for k, v in wv.items():
            rows.append({"grid": grid, "variant": variant, "metric": k, "value": v})
        print(f"{name}: warp1_p95={wv['warp1_p95']:.4f} end_p95={wv['warp1_end_p95']:.4f} sharp_ratio={wv['sharp_w_ratio']:.3f}", flush=True)
    del raft
    torch.cuda.empty_cache()

    for mname in IQA_METRICS:
        try:
            model = pyiqa.create_metric(mname, device=DEV)
        except Exception as e:
            print(f"SKIP {mname}: {e}", flush=True)
            continue
        for (grid, variant), frames in vids.items():
            with torch.no_grad():
                vals = {}
                for wname, win in [("start", frames[:W]), ("end", frames[-W:])]:
                    batch_scores = []
                    b = to01(win).to(DEV)
                    for i in range(0, len(b), 4):
                        batch_scores.append(model(b[i : i + 4]).detach().float().cpu().flatten())
                    vals[wname] = float(torch.cat(batch_scores).mean())
            key = mname.replace("+", "p")
            for k2, v2 in [(f"{key}_w_start", vals["start"]), (f"{key}_w_end", vals["end"]),
                           (f"{key}_w_drift", vals["start"] - vals["end"])]:
                rows.append({"grid": grid, "variant": variant, "metric": k2, "value": v2})
        print(f"done {mname}", flush=True)
        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    if os.path.exists(OUT_CSV):
        old = pd.read_csv(OUT_CSV)
        old = old[~old.metric.isin(df.metric.unique())]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV} ({len(df)} rows)")


if __name__ == "__main__":
    main()
