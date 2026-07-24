"""Run a battery of no-reference IQA metrics + temporal metrics on variant tile videos.

Outputs eval/results_iqa.csv with one row per (grid, variant, metric).
"""
import argparse, glob, json, os, sys
import cv2
import numpy as np
import torch
import pandas as pd

DEV = "cuda"
TILE_DIR = os.path.join(os.path.dirname(__file__), "tiles")
OUT_CSV = os.path.join(os.path.dirname(__file__), "results_iqa.csv")


def read_frames(path, max_frames=None):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)  # T,H,W,3 uint8


def to_tensor(frames):
    # T,H,W,3 uint8 -> T,3,H,W float in [0,1]
    return torch.from_numpy(frames).permute(0, 3, 1, 2).float().div(255.0)


def sample_idx(n, k):
    return np.linspace(0, n - 1, k).round().astype(int)


def temporal_metrics(frames):
    """Cheap temporal artifact stats on full frame sequence (uint8 RGB)."""
    f = frames.astype(np.float32) / 255.0
    gray = f.mean(-1)  # T,H,W
    d1 = np.abs(np.diff(gray, axis=0))            # adjacent frame diff
    # shimmer: high-frequency temporal flicker = 2nd temporal derivative magnitude
    d2 = np.abs(np.diff(gray, 2, axis=0))
    # blockwise flicker: mean over 16x16 blocks then temporal std
    T, H, W = gray.shape
    bs = 16
    blocks = gray[:, : H // bs * bs, : W // bs * bs].reshape(T, H // bs, bs, W // bs, bs).mean((2, 4))
    return {
        "t_framediff": float(d1.mean()),
        "t_shimmer": float(d2.mean()),
        "t_block_flicker": float(np.std(np.diff(blocks, axis=0), axis=0).mean()),
    }


def sharpness_metrics(frames):
    """Laplacian variance (blur proxy) and local contrast, averaged over sampled frames."""
    vals_lap, vals_contrast, vals_sat, vals_dark = [], [], [], []
    for i in sample_idx(len(frames), 16):
        g = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        vals_lap.append(cv2.Laplacian(g, cv2.CV_64F).var())
        vals_contrast.append(g.std())
        hsv = cv2.cvtColor(frames[i], cv2.COLOR_RGB2HSV)
        vals_sat.append(hsv[..., 1].mean())
        # dark channel prior (haze proxy): min over channels then local min-filter
        dc = frames[i].min(-1)
        dc = cv2.erode(dc, np.ones((15, 15), np.uint8))
        vals_dark.append(dc.mean())
    return {
        "s_laplacian_var": float(np.mean(vals_lap)),
        "s_contrast": float(np.mean(vals_contrast)),
        "s_saturation": float(np.mean(vals_sat)),
        "s_dark_channel": float(np.mean(vals_dark)),  # high = hazy/bright veil
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="musiq,clipiqa,brisque,niqe,maniqa,topiq_nr,liqe,arniqa,hyperiqa,dbcnn,paq2piq,nima,cnniqa,tres,unique,clipiqa+,qualiclip+")
    ap.add_argument("--frames", type=int, default=16, help="frames sampled per video for IQA")
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--skip-temporal", action="store_true")
    args = ap.parse_args()

    import pyiqa

    files = sorted(glob.glob(os.path.join(TILE_DIR, "*.mp4")))
    if not files:
        sys.exit("no tiles found")

    metric_names = [m for m in args.metrics.split(",") if m]
    rows = []

    # cache decoded videos
    vids = {}
    for fp in files:
        name = os.path.basename(fp)[: -len(".mp4")]
        grid, variant = name.split("__")
        vids[(grid, variant)] = read_frames(fp)
        if not args.skip_temporal:
            tm = temporal_metrics(vids[(grid, variant)])
            tm.update(sharpness_metrics(vids[(grid, variant)]))
            for k, v in tm.items():
                rows.append({"grid": grid, "variant": variant, "metric": k, "value": v})
        print(f"decoded {name}: {vids[(grid, variant)].shape}", flush=True)

    for mname in metric_names:
        try:
            model = pyiqa.create_metric(mname, device=DEV)
        except Exception as e:
            print(f"SKIP {mname}: {e}", flush=True)
            continue
        for (grid, variant), frames in vids.items():
            idx = sample_idx(len(frames), args.frames)
            batch = to_tensor(frames[idx]).to(DEV)
            scores = []
            with torch.no_grad():
                for i in range(0, len(batch), 4):
                    s = model(batch[i : i + 4])
                    scores.append(s.detach().float().cpu().flatten())
            val = float(torch.cat(scores).mean())
            rows.append({"grid": grid, "variant": variant, "metric": mname, "value": val})
            print(f"{mname} {grid} {variant}: {val:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    # merge with any existing results (overwrite same metric keys)
    if os.path.exists(args.out):
        old = pd.read_csv(args.out)
        old = old[~old.metric.isin(df.metric.unique())]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
