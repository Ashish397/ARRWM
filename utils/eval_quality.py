"""Visual-quality eval: paired fidelity (PSNR/SSIM/LPIPS) + FVD, in one pass.

For each held-out ride we slide a window across the ride; at each window we
regenerate the continuation under the TRUE command (teacher-forced clean
context + real action), decode to RGB, and:
  * compare it frame-by-frame to the ground-truth video -> PSNR / SSIM / LPIPS
    (paired prediction-fidelity metrics, sensitive for ablation comparison);
  * extract Kinetics video features (torchvision r3d_18) of the generated and
    the real clip -> a Frechet Video Distance (FVD) between the two sets.

CAVEAT on FVD: our held-out set is small, so the feature covariance is
rank-deficient; we add shrinkage and report FVD as INDICATIVE (relative ordering
across ablations), not comparable to other papers' I3D-FVD absolute values.

Run single-GPU:  python utils/eval_quality.py --config <cfg> --logdir <dir> --out <json> --windows 12
"""
import argparse, json, os, sys, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import scipy.linalg
from omegaconf import OmegaConf
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

from trainer.causal_diffusion_teacher_train import CausalLoRADiffusionTrainer
from utils.zarr_dataset import ZarrRideDataset
from utils.eval_action_swap import generate_with_command

_KIN_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1)
_KIN_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--logdir", default="")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--windows", type=int, default=12, help="windows per ride")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(args.config))
    if args.logdir:
        config.logdir = args.logdir
    if args.checkpoint:
        config.resume_from = args.checkpoint
    config.disable_wandb = True; config.no_save = True
    config.no_visualize = True; config.auto_resume = True; config.use_one_logger = False

    trainer = CausalLoRADiffusionTrainer(config)
    wrapper = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    wrapper.eval()
    dev = trainer.device

    import lpips
    lpips_fn = lpips.LPIPS(net="alex").to(dev).eval()

    import torchvision
    vnet = torchvision.models.video.r3d_18(weights="KINETICS400_V1")
    vnet.fc = torch.nn.Identity()
    vnet = vnet.to(dev).eval()
    kin_mean, kin_std = _KIN_MEAN.to(dev), _KIN_STD.to(dev)

    # VBench-aligned intrinsic quality models (match the Self-Forcing/LongLive
    # lineage): MUSIQ = imaging quality, NIMA = aesthetic quality, CLIP feature
    # consistency = subject/background consistency, temporal flicker = motion
    # smoothness. Computed on the generated clip alone (no ground truth needed).
    import pyiqa, open_clip
    musiq_m = pyiqa.create_metric("musiq", device=dev)
    nima_m = pyiqa.create_metric("nima", device=dev)
    clip_model = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")[0].to(dev).eval()
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(dev)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(dev)

    @torch.no_grad()
    def intrinsic_quality(arr):  # [T,H,W,3] uint8 -> dict
        T = arr.shape[0]
        idx = np.linspace(0, T - 1, min(T, 8)).round().astype(int)
        fr = (torch.from_numpy(arr[idx]).float().permute(0, 3, 1, 2) / 255.0).to(dev)  # [n,3,H,W] in [0,1]
        musiq = float(musiq_m(fr).mean().item())
        nima = float(nima_m(fr).mean().item())
        c = F.interpolate(fr, size=(224, 224), mode="bilinear", align_corners=False)
        feats = clip_model.encode_image((c - clip_mean) / clip_std).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        cons = float((feats[1:] * feats[:-1]).sum(-1).mean().item()) if feats.shape[0] > 1 else 1.0
        g = torch.from_numpy(arr).float() / 255.0
        flick = float((g[1:] - g[:-1]).abs().mean().item())
        return {"musiq": musiq, "nima": nima, "clip_consistency": cons, "temporal_flicker": flick}

    @torch.no_grad()
    def drift_metrics(gen, gt):
        # Short-rollout drift: how fidelity / intrinsic quality degrade across
        # frame position, plus statistical (brightness/contrast) drift -- the
        # over-saturation / error-accumulation the AR-video lineage flags.
        T = min(len(gen), len(gt)); gen2, gt2 = gen[:T], gt[:T]
        pos = np.arange(T, dtype=np.float64)

        def slope(x, y):
            x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
            return float(np.polyfit(x, y, 1)[0]) if len(x) > 1 else 0.0

        psnr_pf = np.array([sk_psnr(gt2[i], gen2[i], data_range=255) for i in range(T)])
        lp_pf = lpips_fn(to_lpips(gen2), to_lpips(gt2)).reshape(-1).cpu().numpy()
        idx = np.linspace(0, T - 1, min(T, 8)).round().astype(int)
        fr = (torch.from_numpy(gen2[idx]).float().permute(0, 3, 1, 2) / 255.0).to(dev)
        mq_pf = musiq_m(fr).reshape(-1).cpu().numpy()
        gf = gen2.astype(np.float32) / 255.0
        bright = gf.reshape(T, -1).mean(1); contr = gf.reshape(T, -1).std(1)
        return {
            "psnr_slope": slope(pos, psnr_pf),          # dB/frame (neg = fidelity drift)
            "lpips_slope": slope(pos, lp_pf),           # /frame (pos = error accumulation)
            "musiq_slope": slope(idx, mq_pf),           # /frame (neg = quality drift)
            "brightness_drift": float(abs(bright[-1] - bright[0])),
            "contrast_drift": float(abs(contr[-1] - contr[0])),
        }

    num_frames = trainer.streaming_chunk_size
    cf = trainer.context_frames
    window_total = num_frames + cf

    def to_lpips(arr):
        return (torch.from_numpy(arr).float().permute(0, 3, 1, 2) / 127.5 - 1.0).to(dev)

    def frame_metrics(gen, gt):
        T = min(gen.shape[0], gt.shape[0]); gen, gt = gen[:T], gt[:T]
        psnr = float(np.mean([sk_psnr(gt[i], gen[i], data_range=255) for i in range(T)]))
        ssim = float(np.mean([sk_ssim(gt[i], gen[i], channel_axis=2, data_range=255) for i in range(T)]))
        with torch.no_grad():
            lp = float(lpips_fn(to_lpips(gen), to_lpips(gt)).mean().item())
        return psnr, ssim, lp

    @torch.no_grad()
    def vfeat(arr):  # [T,H,W,3] uint8 -> [512] Kinetics feature
        T = arr.shape[0]
        idx = np.linspace(0, T - 1, 16).round().astype(int)
        frames = torch.from_numpy(arr[idx]).float() / 255.0       # [16,H,W,3]
        frames = frames.permute(0, 3, 1, 2)                       # [16,3,H,W]
        frames = F.interpolate(frames, size=(112, 112), mode="bilinear", align_corners=False)
        clip = frames.permute(1, 0, 2, 3).unsqueeze(0).to(dev)    # [1,3,16,112,112]
        clip = (clip - kin_mean) / kin_std
        return vnet(clip).squeeze(0).float().cpu().numpy()

    records, real_feats, gen_feats = [], [], []
    for ride_idx in range(len(trainer.eval_dataset)):
        try:
            ride = trainer.eval_dataset[ride_idx]
            zarr_path = ride["zarr_path"]; name = os.path.basename(zarr_path)
            n_lat = ride["n_latent_frames"]
            if n_lat < window_total:
                continue
            prompt = ride["prompt_embeds"].unsqueeze(0).to(dev, dtype=trainer.dtype)
            max_start = n_lat - window_total
            starts = np.unique(np.linspace(0, max_start, args.windows).round().astype(int)) if max_start > 0 else np.array([0])
            for start in starts:
                start = int(start)
                full = ZarrRideDataset.load_latent_chunk(zarr_path, start, start + window_total).unsqueeze(0).to(dev, dtype=torch.float32)
                z_actions = trainer.eval_dataset.encode_z_actions_window(zarr_path, n_lat, start, start + window_total).unsqueeze(0).to(dev, dtype=trainer.dtype)
                ctx = full[:, :num_frames]
                z_sliced = z_actions[..., trainer.action_dims] if trainer.action_dims is not None else z_actions
                z_clean = z_sliced[:, :num_frames]; z_true = z_sliced[:, cf:]
                # candidate GT latent offsets 0..cf (the window has cf extra
                # frames), pick the one that best aligns with the generation.
                gt_cands = {off: trainer._decode_latents(full[:, off:off + num_frames])
                            for off in range(cf + 1)}
                gen = trainer._decode_latents(generate_with_command(trainer, wrapper, prompt, ctx, z_clean, z_true, num_frames, 0))
                best = None
                for off, gt in gt_cands.items():
                    p, s, l = frame_metrics(gen, gt)
                    if best is None or p > best[1]:
                        best = (off, p, s, l, gt)
                off, psnr, ssim, lp, gt_best = best
                rec = {"ride": name, "start": start, "align_off": off, "psnr": psnr, "ssim": ssim, "lpips": lp}
                rec.update(intrinsic_quality(gen))
                rec.update(drift_metrics(gen, gt_best))
                records.append(rec)
                real_feats.append(vfeat(gt_best)); gen_feats.append(vfeat(gen))
                logging.info("ride=%s start=%d off=%d | PSNR=%.2f SSIM=%.3f LPIPS=%.3f", name, start, off, psnr, ssim, lp)
        except Exception as e:
            logging.exception("ride %d failed: %s", ride_idx, e)

    def frechet(fr, fg):
        fr, fg = np.asarray(fr), np.asarray(fg)
        mu1, mu2 = fr.mean(0), fg.mean(0)
        d = fr.shape[1]
        eps = 1e-6
        s1 = np.cov(fr, rowvar=False) + eps * np.eye(d)
        s2 = np.cov(fg, rowvar=False) + eps * np.eye(d)
        cm = scipy.linalg.sqrtm(s1 @ s2)
        if np.iscomplexobj(cm):
            cm = cm.real
        return float((mu1 - mu2) @ (mu1 - mu2) + np.trace(s1 + s2 - 2 * cm))

    def mean(k):
        v = [r[k] for r in records]; return float(np.mean(v)) if v else float("nan")
    fvd = frechet(real_feats, gen_feats) if len(gen_feats) >= 4 else float("nan")
    summary = {"n_clips": len(records),
               # prediction fidelity (paired vs ground truth)
               "PSNR": mean("psnr"), "SSIM": mean("ssim"), "LPIPS": mean("lpips"),
               # VBench-aligned intrinsic quality (generated clip only)
               "MUSIQ_imaging": mean("musiq"), "NIMA_aesthetic": mean("nima"),
               "CLIP_consistency": mean("clip_consistency"), "temporal_flicker": mean("temporal_flicker"),
               # short-rollout drift (degradation across frame position)
               "psnr_slope": mean("psnr_slope"), "lpips_slope": mean("lpips_slope"),
               "musiq_slope": mean("musiq_slope"), "brightness_drift": mean("brightness_drift"),
               "contrast_drift": mean("contrast_drift"),
               # distributional
               "FVD_r3d18_indicative": fvd}
    out = {"config": args.config, "checkpoint_logdir": str(trainer.logdir), "records": records, "summary": summary}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n=== VISUAL QUALITY ({args.config})  (n={summary['n_clips']} clips) ===")
    print(f"  fidelity:   PSNR={summary['PSNR']:.2f}  SSIM={summary['SSIM']:.3f}  LPIPS={summary['LPIPS']:.3f}")
    print(f"  VBench-like: MUSIQ={summary['MUSIQ_imaging']:.2f}  NIMA={summary['NIMA_aesthetic']:.3f}  "
          f"CLIPcons={summary['CLIP_consistency']:.3f}  flicker={summary['temporal_flicker']:.4f}")
    print(f"  drift:      PSNRslope={summary['psnr_slope']:.3f}  LPIPSslope={summary['lpips_slope']:.4f}  "
          f"MUSIQslope={summary['musiq_slope']:.3f}  brightΔ={summary['brightness_drift']:.4f}  contrastΔ={summary['contrast_drift']:.4f}")
    print(f"  FVD(r3d18,indicative)={fvd:.1f}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
