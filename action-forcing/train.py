"""Action-Forcing ODE distillation entrypoint.

Usage (single node, 4 GPUs)::

    torchrun --nproc_per_node=4 action-forcing/train.py \\
        --config configs/action_ode_distill.yaml

Any field in the yaml can be overridden on the CLI via ``key=value`` pairs
(OmegaConf dotlist syntax), e.g.::

    torchrun ... train.py --config configs/action_ode_distill.yaml \\
        lr=5e-5 critic_updates_per_step=1 lambda_cf=0.5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make ``action-forcing/`` importable as top-level so ``af_model.*`` /
# ``af_utils.*`` / ``af_trainer.*`` resolve without needing a package name
# that contains a hyphen.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
# Workspace root (so ``utils.*`` / ``model.*`` / ``wan.*`` resolve).
_WORKSPACE = _THIS_DIR.parent
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Action-Forcing ODE distillation")
    p.add_argument("--config", type=str, required=True, help="Path to yaml config.")
    p.add_argument(
        "overrides", nargs="*",
        help="Optional OmegaConf dotlist overrides (key=value).",
    )
    return p.parse_args()


def _setup_logging(rank: int) -> None:
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format=f"[%(asctime)s][rank={rank}][%(levelname)s][%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    args = _parse_args()

    import os
    rank = int(os.environ.get("RANK", 0))
    _setup_logging(rank)

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    OmegaConf.set_struct(cfg, False)

    if "config_name" not in cfg:
        cfg.config_name = Path(args.config).stem

    from af_trainer import Trainer
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
