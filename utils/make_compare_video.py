"""Render generated rollouts (true command) from one checkpoint to MP4, on a
FIXED set of eval windows with a FIXED seed -- so two runs (e.g. full vs
ablated state-probe) are directly comparable frame-for-frame.

Run once per model into its own out dir; stitch side-by-side afterwards.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import cv2
from omegaconf import OmegaConf

from trainer.causal_diffusion_teacher_train import CausalLoRADiffusionTrainer
from utils.zarr_dataset import ZarrRideDataset
from utils.eval_action_swap import generate_with_command


def write_mp4(frames, path, fps=10):  # frames [T,H,W,3] uint8 RGB
    h, w = frames.shape[1:3]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--logdir", default="")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--windows_per_ride", type=int, default=2)
    ap.add_argument("--save_gt", action="store_true", help="also write the ground-truth clip")
    args = ap.parse_args()

    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(args.config))
    if args.logdir:
        cfg.logdir = args.logdir
    if args.checkpoint:
        cfg.resume_from = args.checkpoint
    cfg.disable_wandb = True; cfg.no_save = True; cfg.no_visualize = True
    cfg.auto_resume = True; cfg.use_one_logger = False
    os.makedirs(args.out_dir, exist_ok=True)

    tr = CausalLoRADiffusionTrainer(cfg)
    wrapper = tr.model.module if hasattr(tr.model, "module") else tr.model
    wrapper.eval()
    nf = tr.streaming_chunk_size; cf = tr.context_frames; wt = nf + cf

    for ride_idx in range(len(tr.eval_dataset)):
        ride = tr.eval_dataset[ride_idx]
        zp = ride["zarr_path"]; name = os.path.splitext(os.path.basename(zp))[0]
        n_lat = ride["n_latent_frames"]
        if n_lat < wt:
            continue
        prompt = ride["prompt_embeds"].unsqueeze(0).to(tr.device, dtype=tr.dtype)
        max_start = n_lat - wt
        starts = np.unique(np.linspace(0, max_start, args.windows_per_ride).round().astype(int)) if max_start > 0 else [0]
        for start in starts:
            start = int(start)
            full = ZarrRideDataset.load_latent_chunk(zp, start, start + wt).unsqueeze(0).to(tr.device, dtype=torch.float32)
            z = tr.eval_dataset.encode_z_actions_window(zp, n_lat, start, start + wt).unsqueeze(0).to(tr.device, dtype=tr.dtype)
            ctx = full[:, :nf]
            zs = z[..., tr.action_dims] if tr.action_dims is not None else z
            gen = tr._decode_latents(generate_with_command(tr, wrapper, prompt, ctx, zs[:, :nf], zs[:, cf:], nf, args.seed))
            tag = f"{name}_s{start}"
            write_mp4(gen, os.path.join(args.out_dir, f"{tag}_gen.mp4"))
            if args.save_gt:
                write_mp4(tr._decode_latents(full[:, :nf]), os.path.join(args.out_dir, f"{tag}_gt.mp4"))
            print(f"wrote {tag}_gen.mp4 ({gen.shape[0]} frames)")


if __name__ == "__main__":
    main()
