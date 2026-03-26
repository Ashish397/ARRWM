# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
import argparse
import os
from omegaconf import OmegaConf
import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="", help="Path to the directory to save logs")
    parser.add_argument("--wandb-save-dir", type=str, default="", help="Path to the directory to save wandb logs")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--no-auto-resume", action="store_true", help="Disable auto resume from latest checkpoint in logdir")
    parser.add_argument("--no-one-logger", action="store_true", help="Disable One Logger (enabled by default)")
    parser.add_argument("--experiment-name", type=str, default="", help="Name for the experiment in wandb")

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    config.logdir = args.logdir
    config.wandb_save_dir = args.wandb_save_dir
    config.disable_wandb = args.disable_wandb
    config.auto_resume = not args.no_auto_resume
    config.use_one_logger = not args.no_one_logger

    if config.trainer == "score_distillation":
        from trainer.distillation import Trainer as ScoreDistillationTrainer
        trainer = ScoreDistillationTrainer(config)
    elif config.trainer == "lora_diffusion":
        from trainer.diffusion_train import LoRADiffusionTrainer
        trainer = LoRADiffusionTrainer(config)
    elif config.trainer == "causal_lora_diffusion":
        from trainer.causal_diffusion_teacher_train import CausalLoRADiffusionTrainer
        trainer = CausalLoRADiffusionTrainer(config)
    else:
        raise ValueError(f"Unknown trainer: {config.trainer}")
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
