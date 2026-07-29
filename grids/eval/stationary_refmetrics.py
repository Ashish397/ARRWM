"""Reference metrics on the stationary set vs the real_rNN ground-truth future.

Per rollout (temporally aligned to the real clip's generated segment):
  LPIPS, SSIM, PSNR  (mean over aligned generated frames)
Per model (distribution level):
  FVD  (cd-fvd, model's generated clips vs real's clips)

Run in the refmetrics venv:  ~/refmetrics_venv/bin/python stationary_refmetrics.py
Writes out/stationary_refmetrics.csv (per-rollout) and prints per-model FVD.
"""
import os, glob
import numpy as np, torch, cv2, imageio, pandas as pd

DIR = "/home/ashish/stationary_evaluation"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "stationary_refmetrics.csv")
CTX = {"astra": 4, "matrixgame": 1, "minwm": 13, "worldcam": 65, "worldplay": 1, "yume": 1}
OURS_REAL_CTX = 12
NF, SIZE = 24, (256, 144)   # aligned frames, common size
DEV = "cuda"


def ctx_of(m):
    return CTX.get(m, OURS_REAL_CTX)


def gen_frames(path, ctx, n_take=NF):
    r = imageio.get_reader(path); n = r.count_frames()
    idx = np.linspace(min(ctx, n - 2), n - 1, n_take).round().astype(int)
    fr = [cv2.resize(np.asarray(r.get_data(int(i))), SIZE) for i in idx]
    r.close()
    return np.stack(fr)   # NF,H,W,3 uint8 RGB


def main():
    import lpips
    from skimage.metrics import structural_similarity as ssim
    from cdfvd import fvd
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    lp = lpips.LPIPS(net="alex").to(DEV).eval()

    scenes = sorted({os.path.basename(f).split("_r")[1][:-4] for f in glob.glob(f"{DIR}/real_r*.mp4")})
    models = sorted({os.path.basename(f).rsplit("_r", 1)[0] for f in glob.glob(f"{DIR}/*.mp4")} - {"real"})

    # cache real frames per scene
    real = {}
    for s in scenes:
        p = f"{DIR}/real_r{s}.mp4"
        if os.path.exists(p):
            real[s] = gen_frames(p, OURS_REAL_CTX)

    rows = []
    for m in models:
        for s in scenes:
            p = f"{DIR}/{m}_r{s}.mp4"
            if not os.path.exists(p) or s not in real:
                continue
            g = gen_frames(p, ctx_of(m)); rr = real[s]
            gt = torch.from_numpy(g).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(DEV)
            rt = torch.from_numpy(rr).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(DEV)
            with torch.no_grad():
                lpv = float(lp(gt, rt).mean().item())
            ss = np.mean([ssim(cv2.cvtColor(g[i], cv2.COLOR_RGB2GRAY), cv2.cvtColor(rr[i], cv2.COLOR_RGB2GRAY))
                          for i in range(NF)])
            ps = np.mean([cv2.PSNR(g[i], rr[i]) for i in range(NF)])
            rows.append(dict(model=m, scene="r" + s, lpips=round(lpv, 4),
                             ssim=round(float(ss), 4), psnr=round(float(ps), 2)))
        print(f"[ref] {m} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(df.groupby("model")[["lpips", "ssim", "psnr"]].mean().round(3).to_string())

    # ---- FVD per model (cd-fvd): model clips vs real clips ----
    print("\n=== FVD (vs real) ===", flush=True)
    ev = fvd.cdfvd("i3d", device=DEV)
    def stack(m):
        vids = []
        for s in scenes:
            p = f"{DIR}/{m}_r{s}.mp4" if m != "real" else f"{DIR}/real_r{s}.mp4"
            if os.path.exists(p):
                vids.append(gen_frames(p, ctx_of(m)))
        return np.stack(vids)   # N,NF,H,W,3
    lv = lambda arr: ev.load_videos(arr, data_type="video_numpy", resolution=128, sequence_length=16)
    ev.compute_real_stats(lv(stack("real")))
    fvd_rows = []
    for m in models:
        ev.empty_fake_stats()
        ev.compute_fake_stats(lv(stack(m)))
        v = float(ev.compute_fvd_from_stats())
        fvd_rows.append((m, v)); print(f"  {m:12s} FVD={v:.1f}", flush=True)
    pd.DataFrame(fvd_rows, columns=["model", "fvd"]).to_csv(os.path.join(HERE, "out", "stationary_fvd.csv"), index=False)


if __name__ == "__main__":
    main()
