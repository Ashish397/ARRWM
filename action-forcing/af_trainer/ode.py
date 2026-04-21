"""Action-Forcing ODE distillation trainer (full-rank DiT, packed B=2).

Runs the student from ``af_model.ODERegression`` on the paired
``PairedTrajectoryDataset``. The teacher's rank-256 LoRA has been folded
into the base DiT at load time (``merge_and_unload``), so from step 1
onward the trainer updates the *full* DiT state dict — there is no
PEFT wrapper at train time.

Each training step does a **packed paired forward**: the clean and CF
branches of the same pair are stacked along the batch dim and pushed
through a single DiT call of size ``2 * batch_size``. The model's
``generator_loss`` slices the output back into per-branch tensors and
returns ``L_total = L_clean + lambda_cf * L_cf``; the trainer does one
``loss.backward()`` per step. The action critic runs its own self-
contained mini-loop (``critic_updates_per_step`` updates) over the
detached per-branch ``pred_x0`` + ``teacher_z_8d`` outputs.

Distributed strategy: **DDP** on the generator (now a plain
``CausalWanModel``-wrapped ``WanDiffusionWrapper``) and DDP on the
action critic. Action projections (``action_projection``,
``action_token_projection``) are left unwrapped and manually gradient-
synced before the generator optimiser step.

Checkpoint layout:
  ``{"generator": <full DiT state_dict>,
     "action_projection": …, "action_token_projection": …,
     "state_probe": …, "action_critic": …,
     "optimizer": …, "critic_optimizer": …,
     "step": int, "config_name": str}``
Files are saved as ``action_ode_step{:07d}.pt`` — renamed from the
LoRA-era ``causal_lora_step*.pt`` pattern to avoid confusion with
teacher / archived LoRA artefacts sitting in adjacent log dirs.
Old LoRA-era checkpoints are NOT migration-loaded; point
``resume_from`` at a fresh ``action_ode_step*.pt`` file instead.
"""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    import wandb
except ImportError:
    wandb = None

from af_model.ode_regression import ODERegression
from af_utils.dataset import PairedTrajectoryDataset, cycle


log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Distributed helpers
# ----------------------------------------------------------------------

def _is_main(rank: int) -> bool:
    return rank == 0


def _init_distributed() -> tuple:
    """Initialise (rank, world_size, local_rank, is_dist).

    Accepts standard torchrun env vars; falls back to single-process.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        return rank, world_size, local_rank, True
    return 0, 1, 0, False


def _dist_mean(t: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        out = t.detach().clone()
        dist.all_reduce(out, op=dist.ReduceOp.AVG)
        return out
    return t.detach()


# ----------------------------------------------------------------------
# Batch movement
# ----------------------------------------------------------------------

def _to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_device(x, device) for x in obj]
    return obj


# ----------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------


class Trainer:
    """Action-Forcing ODE distillation trainer (DDP + paired dual forward)."""

    def __init__(self, config: Any) -> None:
        self.config = config

        # ------------------------------------------------------------------
        # Distributed + device
        # ------------------------------------------------------------------
        self.rank, self.world_size, self.local_rank, self.is_distributed = _init_distributed()
        self.device = torch.device(
            f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        )
        self.is_main = _is_main(self.rank)

        # ------------------------------------------------------------------
        # Seed + housekeeping
        # ------------------------------------------------------------------
        seed = int(getattr(config, "seed", 0)) + self.rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.config_name = str(getattr(config, "config_name", "action_ode_distill"))
        self.logdir = Path(getattr(config, "logdir", f"logs/{self.config_name}"))
        if self.is_main:
            self.logdir.mkdir(parents=True, exist_ok=True)

        self.mixed_precision = bool(getattr(config, "mixed_precision", True))
        self.dtype = torch.bfloat16 if self.mixed_precision else torch.float32

        # ------------------------------------------------------------------
        # Training hyperparams
        # ------------------------------------------------------------------
        self.total_steps = int(getattr(config, "total_steps", 20000))
        self.batch_size = int(getattr(config, "batch_size", 1))
        # ``grad_accum`` was previously a silently-broken knob — it only
        # scaled the loss by 1/N without an actual accumulation loop, so
        # values > 1 just under-gradiented the optimiser. We now refuse
        # anything other than 1 so nobody silently retrains with the
        # broken behaviour. If you need a larger effective batch, use
        # ``batch_size`` or add more DDP ranks (both of those are real).
        _grad_accum_cfg = int(getattr(config, "grad_accum", 1))
        if _grad_accum_cfg != 1:
            raise ValueError(
                f"config.grad_accum={_grad_accum_cfg} is not supported. "
                "The action-forcing trainer does not implement gradient "
                "accumulation; the previous ``loss / grad_accum`` hack "
                "only scaled the loss and under-gradiented without any "
                "accumulation loop. Use a larger batch_size or more DDP "
                "ranks instead, and leave grad_accum=1."
            )
        self.grad_clip = float(getattr(config, "grad_clip", 1.0))
        self.lambda_cf = float(getattr(config, "lambda_cf", 1.0))

        self.warmup_steps = int(getattr(config, "warmup_steps", 0))
        self.log_interval = int(getattr(config, "log_interval", 25))
        self.save_interval = int(getattr(config, "save_interval", 500))
        self.keep_last_ckpts = int(getattr(config, "keep_last_ckpts", 3))
        # Rolling-latest cadence. Writes a ``latest_{step:07d}.pt`` alongside
        # the permanent ``action_ode_step*.pt`` snapshots and atomically
        # rotates it by deleting older ``latest_*.pt`` after a successful
        # write. ``rolling_interval <= 0`` disables the feature.
        self.rolling_interval = int(getattr(config, "rolling_interval", 0))
        self.critic_updates_per_step = int(getattr(config, "critic_updates_per_step", 2))

        # Eval knobs (distilled, few-step rolling clean+CF generator).
        self.eval_interval = int(getattr(config, "eval_interval", 200))
        self.eval_num_samples = int(getattr(config, "eval_num_samples", 4))
        self.eval_fps = int(getattr(config, "eval_fps", 5))

        # ------------------------------------------------------------------
        # Model
        # ------------------------------------------------------------------
        self._log("Building ODERegression model ...")
        self.model = ODERegression(config, device=self.device)

        # DDP-wrap the generator and the action critic separately; leave
        # action projections unwrapped (sync gradients manually on step).
        if self.is_distributed:
            # find_unused_parameters=False: after LoRA merge_and_unload()
            # every DiT parameter is trainable and participates in every
            # forward pass, including the state-probe branch. Enabling
            # the extra graph traversal costs ~5-10% per step and would
            # silently absorb any future bug that leaves a parameter
            # genuinely unused (e.g. a disabled conditioning path). We
            # want such bugs to surface as loud DDP errors.
            self.model.generator = DDP(
                self.model.generator,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
            if self.model.action_critic is not None:
                self.model.action_critic = DDP(
                    self.model.action_critic,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    broadcast_buffers=False,
                    find_unused_parameters=False,
                )

        # Lazy-build motion pipeline after distributed rendezvous so the
        # rank-0 cotracker download populates hub cache first.
        if self.model.use_motion_pipeline:
            self.model.ensure_motion_pipeline(distributed=self.is_distributed)

        # ------------------------------------------------------------------
        # Optimisers
        # ------------------------------------------------------------------
        self.optimizer, self.critic_optimizer = self._build_optimizers()

        # AMP scaler (bf16 doesn't use a scaler; keep a no-op shim for API
        # parity with the teacher trainer).
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)

        # ------------------------------------------------------------------
        # Dataset + sampler
        # ------------------------------------------------------------------
        self._log("Building PairedTrajectoryDataset ...")
        self.dataset = PairedTrajectoryDataset(
            clean_root=str(getattr(config, "clean_root")),
            cf_root=str(getattr(config, "cf_root")),
            caption_root=str(getattr(config, "caption_root")),
            max_pair=int(getattr(config, "max_pair", 0)) or None,
            require_cf=bool(getattr(config, "require_cf", True)),
            allow_cf_fallback=bool(getattr(config, "allow_cf_fallback", False)),
            max_consecutive_same_fail=int(
                getattr(config, "dataset_max_consecutive_same_fail", 5)
            ),
            max_consecutive_skips_total=int(
                getattr(config, "dataset_max_consecutive_skips_total", 64)
            ),
        )
        sampler = (
            DistributedSampler(self.dataset, shuffle=True, drop_last=True)
            if self.is_distributed else None
        )
        self.sampler = sampler
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=int(getattr(config, "num_workers", 2)),
            pin_memory=True,
            drop_last=True,
            persistent_workers=int(getattr(config, "num_workers", 2)) > 0,
        )
        self.data_iter = cycle(self.loader)

        # ------------------------------------------------------------------
        # Resume
        # ------------------------------------------------------------------
        self.global_step = 0
        self._try_resume()

        # ------------------------------------------------------------------
        # W&B
        # ------------------------------------------------------------------
        self._wandb_run = None
        if self.is_main and not bool(getattr(config, "disable_wandb", False)) and wandb is not None:
            self._init_wandb()

        self._log(
            f"Trainer ready | rank={self.rank}/{self.world_size} "
            f"device={self.device} "
            f"steps={self.total_steps} start_step={self.global_step} "
            f"random_steps={self.model.random_steps} "
            f"critic_num_chunks={self.model.critic_num_chunks} "
            f"lambda_cf={self.lambda_cf}"
        )
        if self.is_main and self._wandb_run is not None:
            try:
                # Each rank processes ``batch_size`` pairs per step; each
                # pair is packed into 2 DiT slots (clean + CF) → the DiT
                # sees ``2 * batch_size * world_size`` sample-views per
                # optimiser step.
                self._wandb_run.summary["effective_batch"] = (
                    2 * self.batch_size * self.world_size
                )
                self._wandb_run.summary["world_size"] = self.world_size
                self._wandb_run.summary["ckpt_path"] = str(getattr(config, "generator_ckpt", ""))
                self._wandb_run.summary["random_steps"] = list(self.model.random_steps)
                self._wandb_run.summary["critic_num_chunks"] = self.model.critic_num_chunks
                self._wandb_run.summary["lambda_cf"] = self.lambda_cf
                # Step-metric registration so eval/* aligns on its own axis.
                wandb.define_metric("train/*", step_metric="step")
                wandb.define_metric("eval/step")
                wandb.define_metric("eval/*", step_metric="eval/step")
            except Exception as exc:
                log.warning("wandb summary registration failed: %s", exc)

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.is_main:
            log.info(msg)

    def _gen_base(self):
        m = self.model.generator
        return m.module if isinstance(m, DDP) else m

    def _critic_base(self):
        if self.model.action_critic is None:
            return None
        m = self.model.action_critic
        return m.module if isinstance(m, DDP) else m

    # ------------------------------------------------------------------
    # Optimisers
    # ------------------------------------------------------------------

    def _build_optimizers(self):
        cfg = self.config
        lr = float(getattr(cfg, "lr", 1e-4))
        beta1 = float(getattr(cfg, "beta1", 0.9))
        beta2 = float(getattr(cfg, "beta2", 0.999))
        weight_decay = float(getattr(cfg, "weight_decay", 0.01))
        critic_lr = float(getattr(cfg, "critic_lr", 3e-4))

        gen_params = [p for p in self._gen_base().parameters() if p.requires_grad]
        if self.model.action_projection is not None:
            gen_params.extend(p for p in self.model.action_projection.parameters() if p.requires_grad)
        if self.model.action_token_projection is not None:
            gen_params.extend(p for p in self.model.action_token_projection.parameters() if p.requires_grad)
        if not gen_params:
            raise RuntimeError("No trainable generator-group parameters.")

        optimizer = torch.optim.AdamW(
            gen_params, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay,
        )
        self._log(f"Generator optimiser: AdamW lr={lr:.2e} params={len(gen_params)}")

        critic_optimizer = None
        cm = self._critic_base()
        if cm is not None:
            critic_params = [p for p in cm.parameters() if p.requires_grad]
            critic_optimizer = torch.optim.AdamW(
                critic_params, lr=critic_lr, betas=(beta1, beta2), weight_decay=weight_decay,
            )
            self._log(f"Critic optimiser: AdamW lr={critic_lr:.2e} params={len(critic_params)}")

        return optimizer, critic_optimizer

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------

    def _init_wandb(self) -> None:
        cfg = self.config
        project = getattr(cfg, "wandb_project", None)
        if not project or wandb is None:
            return
        key = getattr(cfg, "wandb_key", None)
        if key:
            try:
                wandb.login(key=key)
            except Exception as exc:
                log.warning("wandb.login failed: %s", exc)
        # OmegaConf's DictConfig doesn't expose ``__dict__`` as the real
        # config contents, so ``vars(cfg)`` previously silently returned
        # only internal flags and nothing actually got logged. Use the
        # official OmegaConf container conversion instead, falling back
        # to ``vars`` for plain namespaces (tests / scripts).
        cfg_dict: Optional[Dict[str, Any]] = None
        try:
            from omegaconf import DictConfig, OmegaConf
            if isinstance(cfg, DictConfig):
                cfg_dict = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
        except Exception:
            cfg_dict = None
        if cfg_dict is None and hasattr(cfg, "__dict__"):
            cfg_dict = {
                k: v for k, v in vars(cfg).items()
                if isinstance(v, (int, float, str, bool, list, tuple))
            }
        self._wandb_run = wandb.init(
            project=project,
            name=getattr(cfg, "wandb_name", self.config_name),
            entity=getattr(cfg, "wandb_entity", None),
            dir=str(self.logdir),
            config=cfg_dict,
        )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    _CKPT_GLOB = "action_ode_step*.pt"

    def _checkpoint_path(self, step: int) -> Path:
        return self.logdir / f"action_ode_step{step:07d}.pt"

    def _try_resume(self) -> None:
        """Resume from a prior ``action_ode_step*.pt`` in ``self.logdir``.

        The format is the full-rank one written by ``_save_checkpoint``
        below. LoRA-era ``causal_lora_step*.pt`` files from the previous
        action-forcing iteration are *not* migration-loaded — they've
        been archived to ``logs/archive/`` for baseline comparison only;
        point ``resume_from`` at a new action_ode_step*.pt if you want
        to resume in-run.

        Auto-resume also considers rolling ``latest_*.pt`` snapshots
        (written every ``rolling_interval`` steps). When both are
        present, the one with the larger embedded ``step`` is preferred
        so we don't silently roll *back* to an older permanent snapshot
        just because the glob order put it last.
        """

        def _step_of(path: Path) -> int:
            # Both naming patterns encode the step as a 7-digit suffix.
            return int(path.stem.split("step")[-1].split("_")[-1])

        permanent = sorted(self.logdir.glob(self._CKPT_GLOB))
        rolling = sorted(self.logdir.glob("latest_*.pt"))
        candidates = permanent + rolling
        resume_from = getattr(self.config, "resume_from", None)
        target: Optional[Path] = None
        if resume_from:
            cand = Path(resume_from)
            if cand.exists():
                target = cand
        elif candidates:
            target = max(candidates, key=_step_of)
        if target is None:
            return
        self._log(f"Resuming from {target}")
        ck = torch.load(target, map_location="cpu", weights_only=False)

        if "generator" not in ck:
            raise RuntimeError(
                f"Resume checkpoint {target} is missing the 'generator' key. "
                "This trainer saves full DiT state_dicts under 'generator' and "
                "does not support loading LoRA-era ({'lora': ...}) checkpoints. "
                "Re-initialise from the teacher ckpt via config.generator_ckpt "
                "or point resume_from at a newer action_ode_step*.pt file."
            )

        # Refuse to silently load a checkpoint that was trained under a
        # different config. A drift here would cascade into misaligned
        # optimiser states, mismatched random_steps / probe dims, and
        # hours of "why is this regressing?" debugging. Users who
        # genuinely want to cross-load can set
        # ``config.allow_config_name_drift=True`` to bypass, but the
        # default is loud failure.
        ck_cfg_name = ck.get("config_name")
        if ck_cfg_name is not None and str(ck_cfg_name) != self.config_name:
            allow_drift = bool(getattr(self.config, "allow_config_name_drift", False))
            msg = (
                f"Resume checkpoint config_name={ck_cfg_name!r} does not "
                f"match current config_name={self.config_name!r}. "
                "Cross-config resumes are refused by default because the "
                "optimiser states, random_steps pool, action-critic / "
                "state-probe shapes, and LR schedule are all config-"
                "specific. Set allow_config_name_drift=True in the "
                "current config to override."
            )
            if not allow_drift:
                raise RuntimeError(msg)
            log.warning(msg + " [allow_config_name_drift=True, proceeding]")

        self._gen_base().model.load_state_dict(ck["generator"], strict=True)

        # Every action-forcing head/projection is required — the student
        # REFUSES to run without them (see ODERegression.__init__ asserts),
        # so a resume that's missing one indicates the checkpoint was
        # written by an older/different trainer and is fundamentally
        # incompatible. Fail loud rather than silently re-initialise
        # with random weights and destroy the warm start.
        if self.model.action_projection is not None:
            if "action_projection" not in ck:
                raise RuntimeError(
                    f"Resume checkpoint {target} is missing 'action_projection'; "
                    "refusing to resume with randomly-initialised action "
                    "modulation weights."
                )
            self.model.action_projection.load_state_dict(ck["action_projection"])
        if self.model.action_token_projection is not None:
            if "action_token_projection" not in ck:
                raise RuntimeError(
                    f"Resume checkpoint {target} is missing "
                    "'action_token_projection'; refusing to resume with "
                    "randomly-initialised action-token weights."
                )
            self.model.action_token_projection.load_state_dict(ck["action_token_projection"])

        base = self._gen_base()
        if hasattr(base, "_state_probe") and base._state_probe is not None:
            if "state_probe" not in ck:
                raise RuntimeError(
                    f"Resume checkpoint {target} is missing 'state_probe'; "
                    "the student enforces state_probe_mode=True and refuses "
                    "to resume without probe weights."
                )
            missing, unexpected = base._state_probe.load_state_dict(
                ck["state_probe"], strict=False,
            )
            if missing or unexpected:
                raise RuntimeError(
                    f"state_probe load from {target} reported "
                    f"{len(missing)} missing / {len(unexpected)} unexpected "
                    f"keys: missing[:5]={list(missing)[:5]} "
                    f"unexpected[:5]={list(unexpected)[:5]}. The probe "
                    "architecture changed relative to the checkpoint."
                )
        cm = self._critic_base()
        if cm is not None:
            if "action_critic" not in ck:
                raise RuntimeError(
                    f"Resume checkpoint {target} is missing 'action_critic'; "
                    "the student enforces action_critic_enabled=True and "
                    "refuses to resume without critic weights."
                )
            missing, unexpected = cm.load_state_dict(
                ck["action_critic"], strict=False,
            )
            if missing or unexpected:
                raise RuntimeError(
                    f"action_critic load from {target} reported "
                    f"{len(missing)} missing / {len(unexpected)} unexpected "
                    f"keys: missing[:5]={list(missing)[:5]} "
                    f"unexpected[:5]={list(unexpected)[:5]}. The critic "
                    "architecture changed relative to the checkpoint."
                )
        if "optimizer" in ck:
            # Optimizer state mismatches are fatal on resume: a silently
            # dropped optimiser state produces a fresh-Adam moment
            # accumulation which destroys the warm start's momentum.
            try:
                self.optimizer.load_state_dict(ck["optimizer"])
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load generator optimizer state from "
                    f"{target}: {exc!r}. Refusing to resume with a fresh "
                    "optimiser — that would destroy the warm-start's "
                    "Adam moment buffers and silently regress training."
                ) from exc
        if self.critic_optimizer is not None and "critic_optimizer" in ck:
            try:
                self.critic_optimizer.load_state_dict(ck["critic_optimizer"])
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load critic optimizer state from "
                    f"{target}: {exc!r}. Refusing to resume with a fresh "
                    "critic optimiser."
                ) from exc
        self.global_step = int(ck.get("step", 0))

    def _build_checkpoint_state(self, step: int) -> Dict[str, Any]:
        """Assemble the full training-state payload for a checkpoint.

        Shared between the permanent ``action_ode_step*.pt`` saves and
        the rolling ``latest_*.pt`` saves so both are byte-identical
        resumable snapshots (just written at different cadences).
        """
        base = self._gen_base()
        state: Dict[str, Any] = {
            "step": step,
            # Full DiT state (LoRA has been folded at load time; this is
            # the full-rank merged model).
            "generator": base.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config_name": self.config_name,
        }
        cm = self._critic_base()
        if cm is not None:
            state["action_critic"] = cm.state_dict()
        if self.critic_optimizer is not None:
            state["critic_optimizer"] = self.critic_optimizer.state_dict()
        if self.model.action_projection is not None:
            state["action_projection"] = self.model.action_projection.state_dict()
        if self.model.action_token_projection is not None:
            state["action_token_projection"] = self.model.action_token_projection.state_dict()
        if hasattr(base, "_state_probe") and base._state_probe is not None:
            state["state_probe"] = base._state_probe.state_dict()
        return state

    def _save_checkpoint(self, step: int) -> None:
        if not self.is_main:
            return
        path = self._checkpoint_path(step)
        state = self._build_checkpoint_state(step)
        torch.save(state, path)
        log.info("Saved checkpoint to %s", path)
        self._cleanup_old_ckpts()

    def _save_rolling_checkpoint(self, step: int) -> None:
        """Write the rolling ``latest_{step:07d}.pt`` snapshot.

        Rotation is done *after* a successful write so a torch.save crash
        mid-step (rare, but possible on disk-full) can't leave us with
        zero rolling snapshots. Older ``latest_*.pt`` files are only
        removed once the new one has been fsynced to disk.
        """
        if not self.is_main:
            return
        new_path = self.logdir / f"latest_{step:07d}.pt"
        state = self._build_checkpoint_state(step)
        torch.save(state, new_path)
        log.info("Saved rolling checkpoint to %s", new_path)
        for old in self.logdir.glob("latest_*.pt"):
            if old.resolve() == new_path.resolve():
                continue
            try:
                old.unlink()
                log.info("Removed stale rolling checkpoint %s", old.name)
            except OSError as exc:
                log.warning("Failed to remove stale rolling %s: %s", old, exc)

    def _cleanup_old_ckpts(self) -> None:
        if not self.is_main:
            return
        ckpts = sorted(self.logdir.glob(self._CKPT_GLOB))
        if len(ckpts) <= self.keep_last_ckpts:
            return
        for old in ckpts[: -self.keep_last_ckpts]:
            try:
                old.unlink()
                log.info("Removed old checkpoint %s", old.name)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Gradient sync / optimiser step
    # ------------------------------------------------------------------

    def _sync_projection_grads(self) -> None:
        """All-reduce grads on projections not wrapped in DDP."""
        if not self.is_distributed:
            return
        for mod in (self.model.action_projection, self.model.action_token_projection):
            if mod is None:
                continue
            for p in mod.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

    def _generator_optim_step(self) -> float:
        self._sync_projection_grads()

        all_gen_params: list = []
        all_gen_params.extend(self._gen_base().parameters())
        if self.model.action_projection is not None:
            all_gen_params.extend(self.model.action_projection.parameters())
        if self.model.action_token_projection is not None:
            all_gen_params.extend(self.model.action_token_projection.parameters())
        grad_norm = 0.0
        if self.grad_clip and self.grad_clip > 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(all_gen_params, self.grad_clip))

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return grad_norm

    # ------------------------------------------------------------------
    # Critic update loop
    # ------------------------------------------------------------------

    def _critic_update_loop(
        self,
        pred_x0_detached: torch.Tensor,    # [B, F, C, H, W]
        teacher_z_8d: Optional[torch.Tensor],  # [B, n_chunks, 8]
        chunk_t: torch.Tensor,             # [B, n_chunks]
        chunk_actions: torch.Tensor,       # [B, n_chunks, 2]
        chunk_mask: torch.Tensor,          # [B, n_chunks] bool — cleanest-K chunks
    ) -> Dict[str, float]:
        """Run ``critic_updates_per_step`` self-contained optimiser steps.

        The critic is *only* trained on chunks whose input noise level is
        among the ``critic_num_chunks`` cleanest pool entries
        (``chunk_mask``). This prevents corruption of the critic's
        reader by high-noise ``pred_x0`` samples — the same protection
        the teacher uses.

        Weighted-MSE on the 8-D teacher z target, scaled by
        ``action_critic_z_loss_weight``, with its own zero_grad/backward/
        clip/step per iteration.  Forwards through the DDP-wrapped critic
        so gradients are all-reduced across ranks.
        """
        critic_ddp = self.model.action_critic
        cm = self._critic_base()
        if (
            critic_ddp is None
            or teacher_z_8d is None
            or self.critic_optimizer is None
            or not chunk_mask.any()
        ):
            return {}

        n_chunks = chunk_t.shape[1]
        tgt = teacher_z_8d[:, :n_chunks]
        action_critic_dims = self.model.action_critic_dims
        loss_weight = float(self.model.action_critic_z_loss_weight)
        mask_f = chunk_mask[:, :n_chunks].to(pred_x0_detached.dtype)  # [B, n_chunks]

        last_loss: float = 0.0
        last_z2_mse: float = 0.0
        last_z7_mse: float = 0.0
        for _ in range(self.critic_updates_per_step):
            self.critic_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda", dtype=self.dtype, enabled=self.mixed_precision,
            ):
                pred_z = critic_ddp(pred_x0_detached, chunk_t, chunk_actions)
                pred_z = pred_z[:, :n_chunks]

                w = torch.ones(
                    pred_z.shape[-1], device=pred_z.device, dtype=pred_z.dtype,
                )
                for d in action_critic_dims:
                    w[d] = 2.0
                per_chunk = (w * (pred_z - tgt) ** 2).mean(dim=-1)       # [B, n_chunks]
                denom = mask_f.sum().clamp_min(1.0)
                critic_z_loss = (per_chunk * mask_f).sum() / denom
                critic_loss = loss_weight * critic_z_loss

                # Per-dim diagnostics on the action-relevant critic dims.
                diag_dims = action_critic_dims
                diag = {}
                for d in diag_dims:
                    sq = ((pred_z[..., d] - tgt[..., d]) ** 2) * mask_f
                    diag[d] = float((sq.sum() / denom).detach())
            critic_loss.backward()

            if self.grad_clip and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(cm.parameters(), self.grad_clip)
            self.critic_optimizer.step()
            last_loss = float(critic_z_loss.detach())
            if len(diag_dims) >= 1:
                last_z2_mse = diag.get(diag_dims[0], 0.0)
            if len(diag_dims) >= 2:
                last_z7_mse = diag.get(diag_dims[1], 0.0)

        return {
            "train/critic_z_loss": last_loss,
            "train/critic_mask_frac": float(mask_f.mean().detach()),
            "train/critic_z2_mse": last_z2_mse,
            "train/critic_z7_mse": last_z7_mse,
        }

    # ------------------------------------------------------------------
    # One training step (packed B=2 paired forward)
    # ------------------------------------------------------------------

    def _train_step(self) -> Dict[str, float]:
        """Single optimiser step on one data pair (clean + CF packed).

        The clean and CF branches are stacked along the batch dim inside
        ``model.generator_loss`` into a single DiT forward of size
        ``2 * batch_size`` (see ``af_model.ODERegression.generator_loss``).
        One ``backward()``, one optimiser step — DDP's
        ``find_unused_parameters=True`` traverse runs once instead of
        twice. OOM at the current ``batch_size`` raises loudly rather
        than falling back to a two-pass path.

        Critic-training inner loop (``_critic_update_loop``) is
        unchanged: it fires once per branch on the detached per-slot
        ``pred_x0`` + ``teacher_z_8d`` the packed forward returned.
        """
        batch = next(self.data_iter)
        batch = _to_device(batch, self.device)
        self.optimizer.zero_grad(set_to_none=True)

        # ---- Packed B=2 forward ----
        with torch.amp.autocast(
            "cuda", dtype=self.dtype, enabled=self.mixed_precision,
        ):
            loss, logs = self.model.generator_loss(
                trajectory_clean=batch["trajectory_clean"],
                trajectory_cf=batch["trajectory_cf"],
                prompt_embeds=batch["prompt_embeds"],
                z_clean=batch["z_clean"],
                z_noisy=batch["z_noisy"],
                z_noisy_cf=batch["z_noisy_cf"],
                clean_x_gt=batch["clean_x_gt"],
                step=self.global_step,
            )
        loss.backward()

        # ---- Generator optimiser step ----
        grad_norm = self._generator_optim_step()

        # ---- Critic inner loop: once per branch (clean, cf) ----
        critic_logs: Dict[str, float] = {}
        for suffix in ("clean", "cf"):
            sub = self._critic_update_loop(
                pred_x0_detached=logs[f"pred_x0_detached_{suffix}"],
                teacher_z_8d=logs.get(f"teacher_z_8d_{suffix}"),
                chunk_t=logs[f"chunk_t_{suffix}"],
                chunk_actions=logs[f"chunk_actions_{suffix}"],
                chunk_mask=logs[f"critic_chunk_mask_{suffix}"],
            )
            # Accumulate: each branch contributes 50% to the averaged
            # critic metric row (matches the old two-pass trainer's
            # 0.5-weighted accumulation for both branches).
            for k, v in sub.items():
                critic_logs[k] = critic_logs.get(k, 0.0) + 0.5 * v

        # ---- Aggregate metrics ----
        loss_clean_f = float(logs["loss_clean"])
        loss_cf_f = float(logs["loss_cf"])
        out: Dict[str, float] = {
            "loss/clean": loss_clean_f,
            "loss/cf": loss_cf_f,
            "loss/total": loss_clean_f + self.lambda_cf * loss_cf_f,
            "loss/ode_clean": float(logs["ode_loss_clean"]),
            "loss/ode_cf":    float(logs["ode_loss_cf"]),
            # Teacher-prefix renames (state probe / critic guidance).
            "train/state_z_loss_clean":    float(logs["state_loss_clean"]),
            "train/state_z_loss_cf":       float(logs["state_loss_cf"]),
            "train/state_guidance_loss_clean":  float(logs.get("state_guidance_loss_clean", 0.0)),
            "train/state_guidance_loss_cf":     float(logs.get("state_guidance_loss_cf", 0.0)),
            "train/state_guidance_scale":       float(logs.get("state_guidance_scale_clean", 0.0)),
            "train/gen_action_loss_clean": float(logs["critic_guidance_loss_clean"]),
            "train/gen_action_loss_cf":    float(logs["critic_guidance_loss_cf"]),
            "train/z_guidance_scale":      float(logs["guidance_scale"]),
            "train/grad_norm":              grad_norm,
            "train/lr":                     self.optimizer.param_groups[0]["lr"],
            "train/flow_pred_norm":         float(logs.get("flow_pred_norm", 0.0)),
        }
        out.update(critic_logs)

        # Per-chunk ODE MSE diagnostic on clean branch (packed path
        # reports this for slot-0 only; dashboard parity with the old
        # trainer).
        per_chunk_clean = logs.get("per_chunk_ode_mse")
        if isinstance(per_chunk_clean, torch.Tensor):
            for i in range(per_chunk_clean.shape[0]):
                out[f"train/per_chunk_ode_mse_c{i}"] = float(per_chunk_clean[i].item())

        # Teacher-z per-dim mean diagnostics (motion-pipeline signal sanity).
        def _tz_mean(tz: Any, dim: int) -> float:
            if not isinstance(tz, torch.Tensor) or tz.shape[-1] <= dim:
                return 0.0
            return float(tz[..., dim].mean().item())

        tz_clean = logs.get("teacher_z_8d_clean")
        tz_cf = logs.get("teacher_z_8d_cf")
        action_dims = list(self.model.action_critic_dims)
        if len(action_dims) >= 1:
            d = action_dims[0]
            out["train/teacher_z2_mean_clean"] = _tz_mean(tz_clean, d)
            out["train/teacher_z2_mean_cf"]    = _tz_mean(tz_cf, d)
        if len(action_dims) >= 2:
            d = action_dims[1]
            out["train/teacher_z7_mean_clean"] = _tz_mean(tz_clean, d)
            out["train/teacher_z7_mean_cf"]    = _tz_mean(tz_cf, d)

        # CF-vs-clean divergence on the student's pred_x0 (last chunk only —
        # whole-frame MSE would average out the most-denoised signal).
        px_clean = logs.get("pred_x0_detached_clean")
        px_cf = logs.get("pred_x0_detached_cf")
        if isinstance(px_clean, torch.Tensor) and isinstance(px_cf, torch.Tensor):
            nb = self.model.num_frame_per_block
            last_clean = px_clean[:, -nb:]
            last_cf = px_cf[:, -nb:]
            out["train/pred_x0_cf_vs_clean_mse"] = float(
                F.mse_loss(last_clean.float(), last_cf.float()).item()
            )

        # Reduce means across ranks so rank 0's wandb log is representative.
        if self.is_distributed:
            reduced: Dict[str, float] = {}
            for k, v in out.items():
                if k == "train/lr":
                    reduced[k] = v
                    continue
                t = torch.tensor([float(v)], device=self.device)
                reduced[k] = float(_dist_mean(t).item())
            out = reduced
        return out

    # ------------------------------------------------------------------
    # In-loop distilled evaluation (rank-0 only)
    # ------------------------------------------------------------------

    def _maybe_eval(self) -> None:
        """Run distilled eval at ``eval_interval``; log video + action diagnostics to wandb.

        Exactly the same few-step student inference validated by
        ``action-forcing/bin/smoke_eval.py``. Rank-0 only; other ranks
        idle at the outer barrier so DDP stays coherent.
        """
        if self.eval_interval <= 0:
            return
        # Step 0 is handled by the startup smoke (``_startup_smoke``) so
        # the first-eval-ever crash path runs *before* any training is
        # done. Past step 0 we gate on the usual modulo cadence.
        if self.global_step == 0:
            return
        if self.global_step % self.eval_interval != 0:
            return
        self._run_eval_with_barriers(tag="eval")

    def _run_eval_with_barriers(self, *, tag: str) -> None:
        """Shared DDP-safe eval dispatch used by both ``_maybe_eval``
        (interval-gated) and ``_startup_smoke`` (one-shot pre-training).

        Rank-0 does the work; other ranks idle at the two barriers so
        DDP stays coherent. Wrapped in the consecutive-failure guard so
        a dead wandb / cotracker loop still crashes the run loudly.
        """
        if self.is_distributed:
            dist.barrier()
        if not self.is_main or self._wandb_run is None:
            if self.is_distributed:
                dist.barrier()
            return

        base = self._gen_base()
        base.eval()
        try:
            self._run_eval_once()
            # Succeeded — reset the consecutive-failure counter.
            self._eval_consecutive_failures = 0
        except Exception as exc:
            # Allow one transient failure (cotracker flake, wandb socket
            # blip, etc.) but raise loudly on a second consecutive miss.
            # Eval is our only visibility channel on distilled quality
            # + action signal; silently skipping it for hours would mask
            # real training-loop regressions.
            self._eval_consecutive_failures = getattr(
                self, "_eval_consecutive_failures", 0
            ) + 1
            log.warning(
                "[%s step=%d] failed (%d consecutive): %s",
                tag, self.global_step, self._eval_consecutive_failures, exc,
                exc_info=True,
            )
            if self._eval_consecutive_failures >= 2:
                raise RuntimeError(
                    f"Eval failed {self._eval_consecutive_failures} consecutive "
                    f"times — aborting training rather than silently continuing "
                    f"without wandb video / action diagnostics. Last error: "
                    f"{exc!r}"
                ) from exc
        finally:
            base.train()
            if self.is_distributed:
                dist.barrier()

    def _startup_smoke(self) -> None:
        """Dry-run every periodic side-effect *once* before training.

        The design principle: anything that would otherwise first fire
        after hours of training (eval video upload, checkpoint save,
        critic diagnostics) should execute on step 0, on the freshly-
        loaded model, so any crash is discovered in minutes rather than
        after a full wall-clock day. Any failure here is fatal —
        training should not start if a periodic hook is broken.

        Runs:
          * ``_save_checkpoint(global_step)``: exercises state-dict
            assembly, torch.save, disk write, and the old-checkpoint
            rotation logic.
          * ``_run_eval_with_barriers(tag='smoke_eval')``: exercises
            ``generate_eval`` (clean + CF), VAE decode, motion pipeline
            (cotracker + ss_vae), critic forward, video encode
            (moviepy/ffmpeg), and wandb upload.

        Idempotent across resumes: if ``global_step`` is non-zero
        (auto-resumed), the smoke still runs at that step — the
        checkpoint overwrites the resume source with a byte-identical
        copy, and the eval just produces an extra datapoint at the
        resume step. Skipping it on resume would defeat the point
        (a broken hook after a code change wouldn't be caught).
        """
        self._log(f"Startup smoke: saving checkpoint + running eval at step={self.global_step} ...")
        if self.is_distributed:
            dist.barrier()
        self._save_checkpoint(self.global_step)
        if self.is_distributed:
            dist.barrier()
        self._run_eval_with_barriers(tag="smoke_eval")
        self._log("Startup smoke: OK.")

    def _run_eval_once(self) -> None:
        """Actual eval body. See ``_maybe_eval`` for orchestration.

        RNG hygiene: the full eval (generate_eval + cotracker + critic
        forwards) runs under a save/restore guard so rank-0's deterministic
        seeding during eval cannot silently diverge its RNG from the other
        ranks' training RNG state. Without this guard each eval would
        leave rank-0's RNG advanced relative to the rest of the world.
        """
        idx = (self.global_step // max(self.eval_interval, 1)) % max(self.eval_num_samples, 1)
        pair = self.dataset[int(idx) % len(self.dataset)]

        device = self.device

        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(device) if torch.cuda.is_available() else None
        try:
            self._run_eval_once_inner(pair, device)
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, device)

    def _run_eval_once_inner(self, pair: Dict[str, Any], device: torch.device) -> None:
        def _add_batch(t: torch.Tensor) -> torch.Tensor:
            return t.unsqueeze(0).to(device)

        prompt_embeds = _add_batch(pair["prompt_embeds"])
        z_clean = _add_batch(pair["z_clean"])
        z_noisy = _add_batch(pair["z_noisy"])
        z_noisy_cf = _add_batch(pair["z_noisy_cf"])
        clean_x = _add_batch(pair["clean_x_gt"])
        B, F_ = clean_x.shape[:2]

        cond_clean = self.model._build_conditional(
            prompt_embeds.to(self.dtype), z_noisy.to(self.dtype),
            z_clean.to(self.dtype), num_frames=F_,
        )
        cond_cf = self.model._build_conditional(
            prompt_embeds.to(self.dtype), z_noisy_cf.to(self.dtype),
            z_clean.to(self.dtype), num_frames=F_,
        )

        # Deterministic paired noise: clean and CF must share the full
        # noise draw so the only diff between them is the action edit.
        # The outer ``_run_eval_once`` save/restore guard ensures this
        # seed does not leak into training RNG on this rank.
        torch.manual_seed(self.global_step)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.global_step)
        noise_shape = (B, F_, clean_x.shape[2], clean_x.shape[3], clean_x.shape[4])
        noise = torch.randn(noise_shape, dtype=torch.float32, device=device)

        gen_clean = self.model.generate_eval(cond_clean, clean_x, noise)
        # Re-seed so the ``torch.randn_like`` calls inside ``generate_eval``
        # produce identical re-noising steps for CF.
        torch.manual_seed(self.global_step)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.global_step)
        gen_cf = self.model.generate_eval(cond_cf, clean_x, noise)

        # Decode + side-by-side video
        frozen_vae = self.model._frozen_vae
        def _decode_np(latents: torch.Tensor) -> 'np.ndarray':
            import numpy as np
            dummy = latents[:, 0:1]
            lat_wd = torch.cat([dummy, latents], dim=1)
            px = frozen_vae.decode_to_pixel(lat_wd.float())[:, 1:, ...]
            vid = (0.5 * (px.float() + 1.0)).clamp(0, 1)
            vid_np = (vid[0].cpu().numpy() * 255).astype(np.uint8)
            if vid_np.shape[-1] != 3:
                vid_np = vid_np.transpose(0, 2, 3, 1)
            return vid_np

        import numpy as np
        vid_clean = _decode_np(gen_clean)                   # [F, H, W, 3]
        vid_cf = _decode_np(gen_cf)
        vid_sbs = np.concatenate([vid_clean, vid_cf], axis=2)

        # Motion-pipeline and critic diagnostics on the generated videos.
        # Batched clean+CF call — one CoTracker + ss_vae pass over
        # ``[2, F, C, H, W]`` rather than two sequential B=1 calls.
        tz_both = self.model._compute_action_teacher_targets(
            torch.cat([gen_clean, gen_cf], dim=0)
        )                                                                   # [2, n_c, 8]
        tz_clean = tz_both[:1]
        tz_cf = tz_both[1:]
        n_c = self.model.n_chunks
        d0 = int(self.model.action_critic_dims[0])
        d1 = int(self.model.action_critic_dims[1])

        critic = self.model.action_critic
        critic_base = critic.module if isinstance(critic, DDP) else critic

        # Critic conditioning must match the commanded action for the
        # noisy window being scored (not the context-window ``z_clean``).
        chunk_actions_clean = (
            z_noisy.reshape(B, n_c, self.model.num_frame_per_block, -1).mean(dim=2)
        )
        chunk_actions_cf = (
            z_noisy_cf.reshape(B, n_c, self.model.num_frame_per_block, -1).mean(dim=2)
        )
        chunk_t = torch.zeros(B, n_c, device=device, dtype=torch.float32)

        # Eval-only critic forwards — wrap in ``torch.no_grad`` so we
        # don't spin up autograd buffers (the critic is ~100M+ params;
        # forward-without-no_grad left large intermediate activations
        # around for the duration of the eval block for no reason).
        with torch.no_grad():
            pz_clean = critic_base(gen_clean, chunk_t, chunk_actions_clean)[:, :n_c]
            pz_cf = critic_base(gen_cf, chunk_t, chunk_actions_cf)[:, :n_c]

        # Commanded-z correlations (action dims only).
        def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
            af = a.float().reshape(-1)
            bf = b.float().reshape(-1)
            af = af - af.mean()
            bf = bf - bf.mean()
            denom = (af.std(unbiased=False) * bf.std(unbiased=False) + 1e-8)
            return float(((af * bf).mean() / denom).item())

        eval_metrics = {
            "eval/step": self.global_step,
            # CF sanity
            "eval/pred_x0_cf_vs_clean_mse": float(F.mse_loss(gen_clean, gen_cf).item()),
            "eval/teacher_z_cf_vs_clean_mse": float(F.mse_loss(tz_clean, tz_cf).item()),
            # Teacher-z per-dim means
            "eval/teacher_z2_mean_clean": float(tz_clean[..., d0].mean().item()),
            "eval/teacher_z2_mean_cf": float(tz_cf[..., d0].mean().item()),
            "eval/teacher_z7_mean_clean": float(tz_clean[..., d1].mean().item()),
            "eval/teacher_z7_mean_cf": float(tz_cf[..., d1].mean().item()),
            # Critic vs teacher
            "eval/critic_z2_mse_clean": float(F.mse_loss(pz_clean[..., d0], tz_clean[..., d0]).item()),
            "eval/critic_z2_mse_cf": float(F.mse_loss(pz_cf[..., d0], tz_cf[..., d0]).item()),
            "eval/critic_z7_mse_clean": float(F.mse_loss(pz_clean[..., d1], tz_clean[..., d1]).item()),
            "eval/critic_z7_mse_cf": float(F.mse_loss(pz_cf[..., d1], tz_cf[..., d1]).item()),
            "eval/critic_z_mse_clean": float(F.mse_loss(pz_clean, tz_clean).item()),
            "eval/critic_z_mse_cf": float(F.mse_loss(pz_cf, tz_cf).item()),
            # Correlations: commanded action vs motion-pipeline reading of generated output
            "eval/corr_state_teacher_z27_clean": _corr(
                tz_clean[..., [d0, d1]], chunk_actions_clean,
            ),
            "eval/corr_state_teacher_z27_cf": _corr(
                tz_cf[..., [d0, d1]], chunk_actions_cf,
            ),
            # Direct command-vs-teacher MSE on action dims
            "eval/mse_teacher_cmd_z27_clean": float(
                F.mse_loss(tz_clean[..., [d0, d1]], chunk_actions_clean).item()
            ),
            "eval/mse_teacher_cmd_z27_cf": float(
                F.mse_loss(tz_cf[..., [d0, d1]], chunk_actions_cf).item()
            ),
        }

        # Wandb video upload is the main visibility channel for the
        # distilled eval. Each video is encoded independently and in a
        # try/except so a single missing codec / ffmpeg flake drops only
        # that video key rather than killing the whole eval payload —
        # the numeric ``eval/*`` metrics (critic MSE, correlations, CF
        # sanity) don't need moviepy and should always upload. The
        # outer consecutive-failure guard (below, on the full payload
        # upload) still escalates if wandb itself is dead.
        payload = dict(eval_metrics)
        _video_specs = (
            ("eval/video_clean", vid_clean),
            ("eval/video_cf", vid_cf),
            ("eval/video_sidebyside", vid_sbs),
        )
        for key, frames in _video_specs:
            try:
                payload[key] = wandb.Video(
                    frames.transpose(0, 3, 1, 2),
                    fps=self.eval_fps, format="mp4",
                )
            except Exception as vexc:
                log.warning(
                    "[eval step=%d] skipping %s (video encode failed: %r); "
                    "numeric eval metrics will still upload.",
                    self.global_step, key, vexc,
                )
        try:
            self._wandb_run.log(payload, step=self.global_step)
            self._wandb_eval_log_failures = 0
        except Exception as exc:
            self._wandb_eval_log_failures = getattr(
                self, "_wandb_eval_log_failures", 0
            ) + 1
            log.warning(
                "wandb eval log failed (%d consecutive): %s",
                self._wandb_eval_log_failures, exc,
            )
            if self._wandb_eval_log_failures >= 2:
                raise RuntimeError(
                    f"wandb eval log failed {self._wandb_eval_log_failures} "
                    f"consecutive times; refusing to silently train with no "
                    f"video dashboard. Last error: {exc!r}"
                ) from exc

        log.info(
            "[eval step=%d] pair=%s pred_x0_cf_vs_clean=%.4f tz_cf_vs_clean=%.4f "
            "tz2 clean/cf=%.3f/%.3f tz7 clean/cf=%.3f/%.3f",
            self.global_step, pair["meta"]["filename"],
            eval_metrics["eval/pred_x0_cf_vs_clean_mse"],
            eval_metrics["eval/teacher_z_cf_vs_clean_mse"],
            eval_metrics["eval/teacher_z2_mean_clean"],
            eval_metrics["eval/teacher_z2_mean_cf"],
            eval_metrics["eval/teacher_z7_mean_clean"],
            eval_metrics["eval/teacher_z7_mean_cf"],
        )

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    def train(self) -> None:
        # Run every periodic hook once before training so a broken
        # eval/save path crashes in minutes rather than after hours.
        self._startup_smoke()

        start = time.time()
        last_log = start
        while self.global_step < self.total_steps:
            if self.sampler is not None:
                self.sampler.set_epoch(self.global_step)

            metrics = self._train_step()
            self.global_step += 1

            if self.is_main and (self.global_step % self.log_interval == 0 or self.global_step == 1):
                now = time.time()
                dt = now - last_log
                last_log = now
                total_dt = now - start
                log.info(
                    "step=%d  clean=%.4f cf=%.4f ode_c=%.4f ode_cf=%.4f "
                    "sp_c=%.4f cg_c=%.4f gscale=%.3f gn=%.3f  dt=%.2fs total=%.1fs",
                    self.global_step,
                    metrics["loss/clean"], metrics["loss/cf"],
                    metrics["loss/ode_clean"], metrics["loss/ode_cf"],
                    metrics["train/state_z_loss_clean"],
                    metrics["train/gen_action_loss_clean"],
                    metrics["train/z_guidance_scale"], metrics["train/grad_norm"],
                    dt, total_dt,
                )
                if self._wandb_run is not None:
                    try:
                        self._wandb_run.log(
                            {**metrics, "step": self.global_step},
                            step=self.global_step,
                        )
                    except Exception as exc:
                        log.warning("wandb.log failed: %s", exc)

            # Distilled eval (rank-0 only; all other ranks barrier inside).
            self._maybe_eval()

            if self.global_step % self.save_interval == 0:
                if self.is_distributed:
                    dist.barrier()
                self._save_checkpoint(self.global_step)
                if self.is_distributed:
                    dist.barrier()

            if (
                self.rolling_interval > 0
                and self.global_step % self.rolling_interval == 0
                # Skip the rolling write on a step that also triggered the
                # permanent snapshot above — the on-disk state is identical
                # and the permanent file already survives, so a rolling
                # save would duplicate IO and briefly double disk use.
                and self.global_step % self.save_interval != 0
            ):
                if self.is_distributed:
                    dist.barrier()
                self._save_rolling_checkpoint(self.global_step)
                if self.is_distributed:
                    dist.barrier()

        # Final checkpoint
        if self.is_distributed:
            dist.barrier()
        self._save_checkpoint(self.global_step)
        if self._wandb_run is not None:
            try:
                self._wandb_run.finish()
            except Exception:
                pass
