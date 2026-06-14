"""Independent action-controllability eval via counterfactual command swaps.

Loads a trained causal world-model checkpoint and, for each held-out eval ride,
regenerates the same clip under several *counterfactual* action commands:

    true     - the real commanded egomotion (z2,z7)
    zero     - no commanded motion
    flip     - sign-reversed command (-z)
    shuffle  - command with its time order permuted (right marginals, wrong path)

For every generation we read the egomotion ACTUALLY rendered into the frames
using ONLY frozen, externally-pretrained modules (CoTracker optical flow +
frozen ss_vae) -- i.e. the same `_compute_teacher_visuals` pipeline the trainer
already uses, which is *not* trained by the world-model run.  This gives a
controllability metric with no circularity (the critic / state-probe, which are
trained inside the run, are never used as the judge here).

Decisive signal:
  * the rendered egomotion should track whatever command is FED (high corr to the
    fed command across all conditions), and
  * under true it should match the true command, while under flip/zero/shuffle it
    should deviate from the true command -- demonstrating the command *causes*
    the rendered egomotion.

Run single-GPU:  python utils/eval_action_swap.py --config configs/...v14.yaml
"""
import argparse
import json
import os
import logging
from pathlib import Path

import sys
# Ensure the repo root is importable regardless of how the script is launched.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from omegaconf import OmegaConf

# Reuse the trainer + its module-level helpers so the generation path is identical.
from trainer.causal_diffusion_teacher_train import (
    CausalLoRADiffusionTrainer,
    _chunk_actions,
    _safe_corr,
)
from utils.zarr_dataset import ZarrRideDataset


def _mse(a, b):
    return torch.mean((a.float() - b.float()) ** 2).item()


@torch.no_grad()
def generate_with_command(trainer, wrapper, prompt_embeds, context_latents,
                          z_clean, z_noisy_cond, num_frames, seed):
    """Run the trainer's eval generation with a (possibly counterfactual) command.

    Mirrors the block_mask handling in the trainer eval loop so the path matches.
    """
    causal_model = wrapper.model
    if hasattr(causal_model, "base_model"):
        causal_model = causal_model.base_model.model
    saved_mask = getattr(causal_model, "block_mask", None)
    causal_model.block_mask = None
    try:
        conditional = trainer._build_conditional(
            prompt_embeds, z_noisy_cond, z_clean, num_frames,
        )
        # Same initial noise across conditions within a seed -> isolates the
        # causal effect of the command on the rendered egomotion.
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        gen_latents, _ = trainer._generate_eval(
            wrapper, conditional, context_latents, num_frames,
        )
    finally:
        causal_model.block_mask = saved_mask
    return gen_latents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--logdir", default="", help="override checkpoint logdir")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="/scratch/u6ex/as1748.u6ex/ARRWM/paper_assets/action_swap_results.json")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    config = OmegaConf.load(args.config)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    if args.logdir:
        config.logdir = args.logdir
    # eval-only flags
    config.disable_wandb = True
    config.no_save = True
    config.no_visualize = True
    config.auto_resume = True
    config.use_one_logger = False

    logging.info("Instantiating trainer (this builds the model, frozen CoTracker+ss_vae, "
                 "eval dataset, and resumes the checkpoint)...")
    trainer = CausalLoRADiffusionTrainer(config)

    if trainer.eval_dataset is None:
        raise RuntimeError("No eval dataset built -- check eval_ride_zarrs / manifest.")
    if trainer.action_critic is None:
        raise RuntimeError("Frozen teacher pipeline not built (action_critic disabled?).")

    wrapper = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    wrapper.eval()

    nfpb = trainer.num_frame_per_block
    num_frames = trainer.streaming_chunk_size
    cf = trainer.context_frames
    window_total = num_frames + cf
    acrit = trainer.action_critic_dims

    n_rides = len(trainer.eval_dataset)
    ride_names = []
    records = []

    CONDITIONS = ["true", "zero", "flip", "shuffle"]
    # fixed time permutation for 'shuffle' (reproducible, non-identity)
    perm = None

    for ride_idx in range(n_rides):
        try:
            ride = trainer.eval_dataset[ride_idx]
            zarr_path = ride["zarr_path"]
            name = Path(zarr_path).name
            ride_names.append(name)
            prompt_embeds = ride["prompt_embeds"].unsqueeze(0).to(trainer.device, dtype=trainer.dtype)
            n_lat = ride["n_latent_frames"]
            if n_lat < window_total:
                logging.warning("ride %s too short (%d<%d), skip", name, n_lat, window_total)
                continue

            full_latents = ZarrRideDataset.load_latent_chunk(zarr_path, 0, window_total)
            full_latents = full_latents.unsqueeze(0).to(trainer.device, dtype=torch.float32)
            z_actions = trainer.eval_dataset.encode_z_actions_window(
                zarr_path, n_lat, 0, window_total,
            ).unsqueeze(0).to(trainer.device, dtype=trainer.dtype)

            context_latents = full_latents[:, :num_frames]
            z_sliced = z_actions[..., trainer.action_dims] if trainer.action_dims is not None else z_actions
            z_clean = z_sliced[:, :num_frames]
            z_noisy_true = z_sliced[:, cf:]                      # [1, T, 2] fed command
            # true command target for the generated frames (z2,z7), chunked
            target_action_z = z_actions[:, cf:][..., acrit][:, :num_frames]
            target_chunk = _chunk_actions(target_action_z, nfpb)  # [1, n_chunks, 2]
            n_chunks = target_chunk.shape[1]

            if perm is None:
                T = z_noisy_true.shape[1]
                g = torch.Generator(device="cpu").manual_seed(1234)
                perm = torch.randperm(T, generator=g).to(trainer.device)

            cond_tensors = {
                "true": z_noisy_true,
                "zero": torch.zeros_like(z_noisy_true),
                "flip": -z_noisy_true,
                "shuffle": z_noisy_true[:, perm, :],
            }

            for cond in CONDITIONS:
                z_fed = cond_tensors[cond]
                fed_chunk = _chunk_actions(z_fed[..., :len(acrit)] if z_fed.shape[-1] >= len(acrit) else z_fed, nfpb)
                fed_chunk = fed_chunk[:, :n_chunks]
                for seed in range(args.seeds):
                    gen_latents = generate_with_command(
                        trainer, wrapper, prompt_embeds, context_latents,
                        z_clean, z_fed, num_frames, seed,
                    )
                    _, teacher_z8 = trainer._compute_teacher_visuals(gen_latents)
                    teacher_z27 = teacher_z8[:, :n_chunks][..., acrit]   # [1, n_chunks, 2]
                    rec = {
                        "ride": name, "ride_idx": ride_idx, "condition": cond, "seed": seed,
                        "mse_rendered_vs_true": _mse(teacher_z27, target_chunk[:, :n_chunks]),
                        "corr_rendered_vs_true": _safe_corr(teacher_z27, target_chunk[:, :n_chunks]),
                        "mse_rendered_vs_fed": _mse(teacher_z27, fed_chunk),
                        "corr_rendered_vs_fed": _safe_corr(teacher_z27, fed_chunk),
                        "rendered_z2_mean": teacher_z27[..., 0].mean().item(),
                        "rendered_z7_mean": teacher_z27[..., 1].mean().item(),
                        "true_z2_mean": target_chunk[..., 0].mean().item(),
                        "true_z7_mean": target_chunk[..., 1].mean().item(),
                        # Raw per-chunk arrays [n_chunks, 2] = (z2,z7) so we can
                        # stratify controllability by command DIRECTION offline
                        # (e.g. forward vs reverse / sign of z7) without rerunning.
                        "rendered_z27": teacher_z27[0].float().cpu().tolist(),
                        "true_cmd_z27": target_chunk[0, :n_chunks].float().cpu().tolist(),
                        "fed_cmd_z27": fed_chunk[0].float().cpu().tolist(),
                    }
                    records.append(rec)
                    logging.info("ride=%s cond=%-7s seed=%d | corr(rendered,fed)=%.3f corr(rendered,true)=%.3f mse_true=%.4f",
                                 name, cond, seed, rec["corr_rendered_vs_fed"],
                                 rec["corr_rendered_vs_true"], rec["mse_rendered_vs_true"])
        except Exception as e:
            logging.exception("ride_idx=%d failed: %s", ride_idx, e)

    # ---- aggregate ----
    def agg(cond, key):
        vals = [r[key] for r in records if r["condition"] == cond]
        return (sum(vals) / len(vals)) if vals else float("nan")

    summary = {}
    for cond in CONDITIONS:
        summary[cond] = {
            "n": sum(1 for r in records if r["condition"] == cond),
            "mean_corr_rendered_vs_fed": agg(cond, "corr_rendered_vs_fed"),
            "mean_corr_rendered_vs_true": agg(cond, "corr_rendered_vs_true"),
            "mean_mse_rendered_vs_true": agg(cond, "mse_rendered_vs_true"),
            "mean_mse_rendered_vs_fed": agg(cond, "mse_rendered_vs_fed"),
        }

    out = {
        "checkpoint_logdir": str(trainer.logdir),
        "config": args.config,
        "rides": ride_names,
        "conditions": CONDITIONS,
        "seeds": args.seeds,
        "records": records,
        "summary": summary,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("\n================ ACTION-SWAP SUMMARY (frozen-pipeline judge) ================")
    print(f"{'condition':<9} {'n':>3} {'corr(rend,FED)':>15} {'corr(rend,TRUE)':>16} {'mse(rend,TRUE)':>15}")
    for cond in CONDITIONS:
        s = summary[cond]
        print(f"{cond:<9} {s['n']:>3} {s['mean_corr_rendered_vs_fed']:>15.3f} "
              f"{s['mean_corr_rendered_vs_true']:>16.3f} {s['mean_mse_rendered_vs_true']:>15.4f}")
    print("\nInterpretation: high corr(rend,FED) across ALL conditions => rendered egomotion")
    print("tracks whatever is commanded (controllability). corr(rend,TRUE) should be high for")
    print("'true' and drop / go negative for flip/zero/shuffle => the command CAUSES the motion.")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
