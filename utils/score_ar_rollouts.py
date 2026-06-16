"""Score AR-rollout quality per config, from the rendered rollout mp4s.

For each (config, clip) under eval/ar_ablation/, load the generated rollout
(_raw.mp4) and its ground truth (_gt.mp4), and compute, over the rollout:
  * paired fidelity vs GT: PSNR, SSIM, LPIPS  (per frame)
  * intrinsic quality: MUSIQ                  (per frame)
  * DRIFT: the per-frame-position slope of each (how fast quality decays as the
    autoregressive rollout proceeds) -- the number that captures "ablations
    wobble over AR steps but v14 holds".
Aggregates per config and writes paper_assets/ar_rollout_scores.json.

This is a post-process on the rendered videos (no generation) -- needs a GPU
only for LPIPS/MUSIQ.
"""
import os, sys, json, glob
sys.path.insert(0, '/lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM')
os.chdir('/lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM')
import numpy as np
import cv2
import torch
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
import lpips, pyiqa
lpips_fn = lpips.LPIPS(net='alex').to(DEV).eval()
musiq = pyiqa.create_metric('musiq', device=DEV)

OUT = os.environ.get('AR_OUT', 'eval/ar_ablation')
CONFIGS = ['v14', 'loo_adaln', 'loo_tokens', 'loo_f3', 'loo_f2']


def read_mp4(path):
    cap = cv2.VideoCapture(path); frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames) if frames else None


def to_t(arr):  # [T,H,W,3] uint8 -> [T,3,H,W] in [-1,1] (lpips) / [0,1] (musiq)
    return torch.from_numpy(arr).float().permute(0, 3, 1, 2).to(DEV)


def slope(y):
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y, 1)[0]) if len(y) > 1 else 0.0


def score_pair(gen, gt):
    T = min(len(gen), len(gt)); gen, gt = gen[:T], gt[:T]
    psnr_pf = np.array([sk_psnr(gt[i], gen[i], data_range=255) for i in range(T)])
    ssim_pf = np.array([sk_ssim(gt[i], gen[i], channel_axis=2, data_range=255) for i in range(T)])
    with torch.no_grad():
        lp_pf = lpips_fn(to_t(gen) / 127.5 - 1, to_t(gt) / 127.5 - 1).reshape(-1).cpu().numpy()
        mq_pf = musiq(to_t(gen) / 255.0).reshape(-1).cpu().numpy()
    return {
        'psnr': float(psnr_pf.mean()), 'ssim': float(ssim_pf.mean()),
        'lpips': float(lp_pf.mean()), 'musiq': float(mq_pf.mean()),
        'psnr_drift': slope(psnr_pf), 'lpips_drift': slope(lp_pf), 'musiq_drift': slope(mq_pf),
    }


agg = {c: [] for c in CONFIGS}
for c in CONFIGS:
    for d in sorted(glob.glob(f'{OUT}/{c}_c*')):
        if not os.path.isdir(d):
            continue
        raw = glob.glob(f'{d}/rank0_*/final_causal_rollout_*_raw.mp4')
        gt = glob.glob(f'{d}/rank0_*/final_causal_rollout_*_gt.mp4')
        if not raw or not gt:
            continue
        g = read_mp4(raw[0]); t = read_mp4(gt[0])
        if g is None or t is None:
            continue
        try:
            agg[c].append(score_pair(g, t))
        except Exception as e:
            print('skip', d, e)

summary = {}
for c in CONFIGS:
    rs = agg[c]
    if not rs:
        summary[c] = None; continue
    summary[c] = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
    summary[c]['n'] = len(rs)

json.dump({'summary': summary}, open('paper_assets/ar_rollout_scores.json', 'w'), indent=1)
print(f"\n{'config':<12}{'n':>3}{'PSNR':>7}{'SSIM':>7}{'LPIPS':>7}{'MUSIQ':>7}{'psnrDrift':>10}{'lpipsDrift':>11}{'musiqDrift':>11}")
for c in CONFIGS:
    s = summary[c]
    if not s:
        print(f"{c:<12}  -- no rollouts --"); continue
    print(f"{c:<12}{s['n']:>3}{s['psnr']:>7.2f}{s['ssim']:>7.3f}{s['lpips']:>7.3f}{s['musiq']:>7.1f}"
          f"{s['psnr_drift']:>+10.3f}{s['lpips_drift']:>+11.4f}{s['musiq_drift']:>+11.3f}")
print("\nDrift = per-frame slope over the AR rollout. psnr_drift<0 / lpips_drift>0 / musiq_drift<0 = quality DECAYING over AR steps.")
