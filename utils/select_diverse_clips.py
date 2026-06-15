"""Re-select N visually-DIVERSE high-motion clips for the AR-rollout demos.

Problem: selecting purely by turning+motion score clusters on similar
park/greenery scenes. Fix: among the top high-motion candidates, embed one
decoded frame of each with CLIP and do farthest-point sampling so the chosen
clips are maximally different in appearance (street / plaza / park / lighting /
colour), while all remaining high-motion.

Writes paper_assets/diverse_clips.json (the N selected windows).
"""
import sys, os, json, argparse
sys.path.insert(0, '/lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM')
os.chdir('/lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM')
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/causal_lora_diffusion_teacher_v14.yaml")
ap.add_argument("--logdir", default="logs/v14_balanced_weunz")
ap.add_argument("--candidates", default="paper_assets/interesting_windows.json")
ap.add_argument("--topk", type=int, default=120, help="top high-motion candidates to consider")
ap.add_argument("--n", type=int, default=20, help="diverse clips to pick")
ap.add_argument("--out", default="paper_assets/diverse_clips.json")
args = ap.parse_args()

from trainer.causal_diffusion_teacher_train import CausalLoRADiffusionTrainer
from utils.zarr_dataset import ZarrRideDataset

cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(args.config))
cfg.logdir = args.logdir
cfg.disable_wandb = True; cfg.no_save = True; cfg.no_visualize = True
cfg.auto_resume = True; cfg.use_one_logger = False
tr = CausalLoRADiffusionTrainer(cfg)
dev = tr.device

import open_clip
clip_model = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")[0].to(dev).eval()
clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(dev)
clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(dev)

cands = json.load(open(args.candidates))["windows"]
cands = sorted(cands, key=lambda w: -w["score"])[:args.topk]
print(f"embedding {len(cands)} high-motion candidates...")

feats, kept = [], []
with torch.no_grad():
    for w in cands:
        try:
            zp, s = w["zarr_path"], int(w["start"])
            mid = s + 9
            lat = ZarrRideDataset.load_latent_chunk(zp, mid, mid + 3).unsqueeze(0).to(dev, dtype=torch.float32)
            frames = tr._decode_latents(lat)            # [T,H,W,3] uint8
            fr = frames[len(frames) // 2]
            t = torch.from_numpy(fr).float().permute(2, 0, 1).unsqueeze(0).to(dev) / 255.0
            t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
            e = clip_model.encode_image((t - clip_mean) / clip_std).float()
            e = e / e.norm(dim=-1, keepdim=True)
            feats.append(e.squeeze(0).cpu().numpy()); kept.append(w)
        except Exception as ex:
            continue
feats = np.asarray(feats)
print(f"embedded {len(kept)} candidates; running farthest-point sampling for {args.n}...")

# farthest-point sampling in CLIP space, seeded by the highest-motion candidate
sel = [0]
dist = 1.0 - feats @ feats[0]
for _ in range(min(args.n, len(kept)) - 1):
    nxt = int(np.argmax(dist))
    sel.append(nxt)
    dist = np.minimum(dist, 1.0 - feats @ feats[nxt])
chosen = [kept[i] for i in sel]

with open(args.out, "w") as f:
    json.dump({"n": len(chosen), "windows": chosen}, f, indent=1)
print(f"\nwrote {args.out} ({len(chosen)} diverse clips):")
for w in chosen:
    print(f"  {os.path.basename(w['zarr_path'])} s{w['start']}  turn={w['turn']} motion={w['motion']}")
