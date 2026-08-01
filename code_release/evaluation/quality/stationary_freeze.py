"""Freeze baseline for the stationary set: repeat the last real CONTEXT frame across
the generated horizon, and score it against the real future with the same reference
metrics. If freeze scores WELL (low LPIPS / FVD), the reference metrics reward
inactivity and cannot substitute for a signs-of-life measure.
Run with ~/refmetrics_venv/bin/python."""
import os
import os, glob
import numpy as np, cv2, imageio, pandas as pd

DIR = os.environ.get("AF_STATIONARY_DIR",
               os.path.expanduser("~/stationary_evaluation"))
HERE = os.path.dirname(os.path.abspath(__file__))
NF, SIZE, CTX = 24, (256, 144), 12
DEV = "cuda"


def real_gen(path):
    r = imageio.get_reader(path); n = r.count_frames()
    idx = np.linspace(CTX, n - 1, NF).round().astype(int)
    fr = [cv2.resize(np.asarray(r.get_data(int(i))), SIZE) for i in idx]
    r.close(); return np.stack(fr)


def real_ctx_last(path):
    r = imageio.get_reader(path); f = cv2.resize(np.asarray(r.get_data(CTX - 1)), SIZE); r.close(); return f


def main():
    import lpips, torch
    from skimage.metrics import structural_similarity as ssim
    lp = lpips.LPIPS(net="alex").to(DEV).eval()
    scenes = sorted({os.path.basename(f).split("_r")[1][:-4] for f in glob.glob(f"{DIR}/real_r*.mp4")})
    rows = []
    for s in scenes:
        p = f"{DIR}/real_r{s}.mp4"
        rr = real_gen(p)                       # real future NF,H,W,3
        frz = np.repeat(real_ctx_last(p)[None], NF, 0)   # frozen last-context frame
        gt = torch.from_numpy(frz).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(DEV)
        rt = torch.from_numpy(rr).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(DEV)
        with torch.no_grad():
            lpv = float(lp(gt, rt).mean())
        ss = np.mean([ssim(cv2.cvtColor(frz[i], cv2.COLOR_RGB2GRAY), cv2.cvtColor(rr[i], cv2.COLOR_RGB2GRAY)) for i in range(NF)])
        ps = np.mean([cv2.PSNR(frz[i], rr[i]) for i in range(NF)])
        rows.append(dict(scene="r" + s, lpips=round(lpv, 4), ssim=round(float(ss), 4), psnr=round(float(ps), 2)))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(HERE, "out", "stationary_freeze_ref.csv"), index=False)
    print("FREEZE baseline vs real (mean):")
    print(df[["lpips", "ssim", "psnr"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
