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
            # Generous timeout: rank 0 runs a long in-loop eval (2x generate +
            # VAE decode + cotracker + the Madrid AR rollout + per-frame IQA +
            # wandb upload) while ranks 1..N wait at the post-eval barrier. The
            # NCCL/c10d default (~10 min) can expire during a slow first eval
            # and abort the group; 60 min is comfortably above worst-case.
            from datetime import timedelta
            dist.init_process_group(
                backend="nccl", init_method="env://",
                timeout=timedelta(minutes=60),
            )
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

        # Pre-warm pyiqa MUSIQ/NIQE on rank 0 OUTSIDE any barrier. These weights
        # download lazily on first use; if that first use is the in-loop Madrid
        # eval (under the post-eval barrier), the download can push rank-0 past
        # the NCCL barrier timeout while the other ranks wait. Warm it here.
        if self.is_main and bool(getattr(config, "madrid_chain_eval_enabled", False)):
            try:
                self._ensure_iqa_metrics()
            except Exception as _iqa_exc:  # noqa: BLE001 — optional diagnostic
                log.warning("pyiqa pre-warm failed (non-fatal): %r", _iqa_exc)

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
        if bool(getattr(config, "chunked_lmdb", False)):
            # ARRWM 14e pilot: chained chunk-level LMDBs (gen_lmdb_14e.py).
            # Set AF_SNAPSHOT_STEPS/AF_EVAL_STEPS + random_steps to the
            # pinned 20-step grid in the same config.
            from af_utils.chunked_ode_dataset import ChunkedODEDataset
            from af_utils.dataset import _load_prompt_embeds, _build_ts_to_caption_json
            from utils.zarr_dataset import ZarrRideDataset as _ZRD
            self._log("Building ChunkedODEDataset ...")
            _cap = _build_ts_to_caption_json(Path(str(getattr(config, "caption_root"))))
            _pcache = {}
            def _prompt(ts):
                if ts not in _pcache:
                    if len(_pcache) > 256:            # bound worker RSS
                        _pcache.clear()
                    _pcache[ts] = _load_prompt_embeds(_cap[ts])
                return _pcache[ts]
            self.dataset = ChunkedODEDataset(
                root=str(getattr(config, "clean_root")),
                zarr_loader=_ZRD.load_latent_chunk,
                prompt_loader=_prompt,
                clean_only=bool(getattr(config, "clean_only", False)),
            )
        else:
            self._log("Building PairedTrajectoryDataset ...")
            self.dataset = PairedTrajectoryDataset(
                clean_root=str(getattr(config, "clean_root")),
                cf_root=str(getattr(config, "cf_root")),
                caption_root=str(getattr(config, "caption_root")),
                max_pair=int(getattr(config, "max_pair", 0)) or None,
                require_cf=bool(getattr(config, "require_cf", True)),
                allow_cf_fallback=bool(getattr(config, "allow_cf_fallback", False)),
                clean_only=bool(getattr(config, "clean_only", False)),
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
        if bool(getattr(config, "alldir_batches", False)):
            # All-directions batches: dataset emitted each (context, chunk)'s
            # cf-variants consecutively; batch = one whole group, so every
            # optimizer step sees ALL directions of one context (packed
            # forward handles B_pair = group size natively).
            import random as _rnd
            _rank = dist.get_rank() if self.is_distributed else 0
            _world = dist.get_world_size() if self.is_distributed else 1

            class _GroupedBatches:
                # Same seeded group order on every rank; each rank takes its
                # contiguous slice of the group -> the GLOBAL batch is one
                # whole group (e.g. 8 directions over 4 ranks x 2).
                def __init__(self, sizes, rank, world):
                    self.rank, self.world = rank, world
                    self.groups = []
                    i = 0
                    for g in sizes:
                        self.groups.append(list(range(i, i + g)))
                        i += g
                    self.epoch = 0
                def _k(self):
                    # world > group size: concatenate K whole groups per
                    # global batch (32 ranks / groups of 8 -> 4 contexts x
                    # 8 dirs per optimizer step). K=1 == old behavior.
                    g = len(self.groups[0]) if self.groups else 1
                    return max(1, self.world // max(g, 1))
                def __iter__(self):
                    order = list(self.groups)
                    _rnd.Random(1234 + self.epoch).shuffle(order)
                    self.epoch += 1
                    k = self._k()
                    for i in range(0, len(order) - k + 1, k):
                        grp = [j for g in order[i:i + k] for j in g]
                        per = max(len(grp) // self.world, 1)
                        lo = self.rank * per
                        yield grp[lo:lo + per] if lo < len(grp) else grp[:per]
                def __len__(self):
                    return max(len(self.groups) // self._k(), 1)

            class _CurriculumBatches(_GroupedBatches):
                """Hard-direction curriculum (user design, 2026-08-09).

                Epoch 0 trains ALL directions of every (context, chunk)
                group. From epoch 1 on, each group contributes only 2-4
                directions: the WORST ones by recorded per-sample error
                ("bad"), plus an equal number of well-learned ones for
                anti-forgetting, chosen as the OPPOSITE compass direction
                where possible (maximum contrast inside the fan batch,
                which is what makes the global-batch mechanism work) and
                at random otherwise.
                    2 bad -> 4 total | 1 bad -> 2 total | 0 bad -> 2 random
                Selected groups are packed so every global batch holds
                exactly `world` samples of WHOLE groups, preserving the
                fan-contrast structure within an optimizer step.
                """
                OPP = {"cF": "cB", "cB": "cF", "cL": "cR", "cR": "cL",
                       "cFR": "cBL", "cBL": "cFR", "cFL": "cBR",
                       "cBR": "cFL"}
                # MEASURED 2026-08-10: pre-curriculum alldir8n HAD a reverse
                # response (BR -0.44, BL -0.54 fwd); after ~1 epoch of
                # curriculum it was GONE (BR +0.25, B +0.26 — forward motion
                # for a reverse command) while forward was untouched
                # (F +0.767 -> +0.765). Reverse is the rarest, hardest mode,
                # so a mean-seeking loss drags it toward the data's dominant
                # forward motion. Reserve a slot so a BACKWARD direction is
                # always in the selected set (user-directed).
                BACK = ("cB", "cBR", "cBL")

                def __init__(self, sizes, rank, world, dirs, groups_of,
                             n_bad_max=2, seed=1234):
                    super().__init__(sizes, rank, world)
                    self.dirs = dirs
                    self.n_bad_max = int(n_bad_max)
                    self.seed = seed
                    self.err = None            # [N] mean per-sample error
                    self.epoch_i = 0

                def set_errors(self, err):
                    self.err = err

                def _force_back(self, grp, picks):
                    """Guarantee a backward direction in the selection."""
                    if any(self.dirs[i] in self.BACK for i in picks):
                        return picks
                    cand = [i for i in grp if self.dirs[i] in self.BACK
                            and i not in picks]
                    if not cand:
                        return picks
                    worst = max(cand, key=lambda i: (float(self.err[i])
                                                     if self.err[i] >= 0 else 1e9))
                    if len(picks) >= 4:        # swap out the easiest pick
                        drop = min(picks, key=lambda i: (float(self.err[i])
                                                         if self.err[i] >= 0 else -1.0))
                        picks = [p for p in picks if p != drop]
                    return picks + [worst]

                def _select(self, grp, rng):
                    mode = getattr(self, "sel_mode", "hard")
                    if mode == "all9":
                        return list(grp)       # alternative arm: every direction
                    if self.err is None:       # epoch 0: everything
                        return list(grp)
                    if mode == "actsplit":
                        # Rank by the per-direction ROLLING-MEAN action residual
                        # (the reference), worst first. 3 random from the bottom
                        # half + 2 random from the top half.
                        ref = getattr(self, "act_ref", None)
                        def _score(i):
                            d = self.dirs[i]
                            if ref is not None and d in ref and ref[d] >= 0:
                                return float(ref[d])
                            return float(self.err[i]) if self.err[i] >= 0 else 1e9
                        ordered = sorted(grp, key=_score, reverse=True)  # worst first
                        h = max(1, len(ordered) // 2)
                        bottom, top = ordered[:h], ordered[h:]
                        picks = list(rng.sample(bottom, min(3, len(bottom))))
                        picks += list(rng.sample(top, min(2, len(top))))
                        return self._force_back(grp, picks) if picks else list(grp)
                    scored = sorted(grp, key=lambda i: -float(self.err[i]))
                    seen = [i for i in grp if self.err[i] >= 0]
                    if not seen:
                        return self._force_back(
                            grp, list(rng.sample(grp, min(2, len(grp)))))
                    # ABSOLUTE threshold, not the group median. The median
                    # always marks half the group "bad" by construction, so
                    # every group selected exactly 4 and the 1-bad/0-bad
                    # cases never fired. The error is rung-normalised with
                    # global mean 1.0, so "worse than the dataset average"
                    # is a meaningful absolute bar and the count can vary.
                    med = float(getattr(self, "bad_thresh", 1.0))
                    bad = [i for i in scored if float(self.err[i]) > med][:self.n_bad_max]
                    if not bad:
                        return self._force_back(
                            grp, list(rng.sample(grp, min(2, len(grp)))))
                    # Anti-forgetting partners must be directions we have
                    # actually OBSERVED to be well-learned; an unseen
                    # sample (err == -1) is unknown, not good.
                    good_pool = [i for i in grp
                                 if i not in bad and self.err[i] >= 0]
                    if not good_pool:
                        good_pool = [i for i in grp if i not in bad]
                    picks = list(bad)
                    for b in bad:
                        want = self.OPP.get(self.dirs[b])
                        opp = [i for i in good_pool
                               if self.dirs[i] == want and i not in picks]
                        if opp:
                            picks.append(opp[0])
                        else:
                            rem = [i for i in good_pool if i not in picks]
                            if rem:
                                picks.append(rng.choice(rem))
                    return self._force_back(grp, picks)

                def _batches(self):
                    rng = _rnd.Random(self.seed + self.epoch_i)
                    order = list(self.groups)
                    rng.shuffle(order)
                    sel = [self._select(g, rng) for g in order]
                    sel = [s for s in sel if s]
                    # BIN-PACK to exactly `world`. A naive "reset cur on
                    # overflow" loop DISCARDS the partial bin: harmless
                    # when every group is the same size, but the selection
                    # rules emit a MIX of 2s and 4s, and mixing knocks the
                    # accumulator off alignment so it overflows constantly
                    # (simulated: 5% size-2 groups -> ~25% of samples
                    # silently dropped). Bucket by size and always take a
                    # group that FITS, so only one final partial bin is
                    # ever left over.
                    from collections import defaultdict as _dd
                    buckets = _dd(list)
                    for s in sel:
                        buckets[len(s)].append(s)
                    sizes_desc = sorted(buckets, reverse=True)
                    out, cur = [], []
                    while True:
                        room = self.world - len(cur)
                        pick = next((z for z in sizes_desc
                                     if z <= room and buckets[z]), None)
                        if pick is None:
                            # No WHOLE group fits the remaining room. Breaking
                            # here discards everything -- and when every group
                            # is the same size that does not divide `world`
                            # (actsplit emits exactly 5; world=32) it discards
                            # ALL of them, returning ZERO batches. `cycle()`
                            # then spins forever on an empty loader: no error,
                            # no NCCL timeout, all ranks silently wedged.
                            # Fill the bin with a PARTIAL group instead. Only
                            # `edist` cares about whole fans, and it masks by
                            # group_id, so a split group is still correct.
                            src = next((z for z in sizes_desc if buckets[z]), None)
                            if src is None:
                                break                  # genuinely exhausted
                            g = buckets[src].pop()
                            cur.extend(g[:room])
                            rest = g[room:]
                            if rest:
                                buckets[len(rest)].append(rest)
                                sizes_desc = sorted(buckets, reverse=True)
                            if len(cur) == self.world:
                                out.append(cur); cur = []
                            continue
                        cur.extend(buckets[pick].pop())
                        if len(cur) == self.world:
                            out.append(cur); cur = []
                    left = len(cur) + sum(len(g) for z in buckets
                                          for g in buckets[z])
                    if left and self.rank == 0:
                        log.info("[curriculum] epoch %d: %d batches, %d "
                                 "samples left over (not a full batch)",
                                 self.epoch_i, len(out), left)
                    return out

                def __iter__(self):
                    bs = self._batches()
                    self.epoch_i += 1
                    self._last_len = len(bs)
                    for gb in bs:
                        per = max(len(gb) // self.world, 1)
                        lo = self.rank * per
                        yield gb[lo:lo + per]

                def __len__(self):
                    # NOTE: _batches() must not be called here after the
                    # first epoch — epoch_i has already advanced, so it
                    # would report the NEXT epoch's length. Cache instead,
                    # and never let a legitimate 0 fall through to the
                    # recompute (an empty loader makes the epoch boundary
                    # fire every step and cycle() spin forever).
                    n = getattr(self, "_last_len", None)
                    if n is None:
                        n = len(self._batches())
                        self._last_len = n
                    return n

            _curric = bool(getattr(config, "ode_curriculum", False))
            _sampler_obj = (
                _CurriculumBatches(
                    self.dataset.group_sizes, _rank, _world,
                    self.dataset.sample_dir, self.dataset.sample_group,
                    n_bad_max=int(getattr(config, "ode_curriculum_bad_max", 2)))
                if _curric else
                _GroupedBatches(self.dataset.group_sizes, _rank, _world))
            self.curriculum = _sampler_obj if _curric else None
            if self.curriculum is not None:
                self.curriculum.sel_mode = str(
                    getattr(config, "ode_curriculum_mode", "hard"))
            self.loader = DataLoader(
                self.dataset,
                batch_sampler=_sampler_obj,
                num_workers=int(getattr(config, "num_workers", 2)),
                pin_memory=True,
                persistent_workers=int(getattr(config, "num_workers", 2)) > 0,
            )
        else:
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
        # ---- Hard-direction curriculum state -----------------------------
        # err_sum/err_cnt are indexed by DATASET sample id and are summed
        # across ranks at each epoch boundary, so every rank selects the
        # same directions for the next epoch (the sampler must stay
        # identical on all ranks or the global batch desyncs).
        self._curric_epochs_done = 0
        self._curric_steps_in_epoch = 0
        if bool(getattr(config, "ode_curriculum", False)) and \
                getattr(self, "curriculum", None) is None:
            raise RuntimeError(
                "ode_curriculum=true requires alldir_batches=true — the "
                "curriculum selects DIRECTIONS within a (context, chunk) "
                "group, which only exists in the grouped sampler. Refusing "
                "to run with the curriculum silently disabled.")
        if getattr(self, "curriculum", None) is not None:
            n = len(self.dataset)
            # Per-EPOCH accumulators (reset every boundary) + a persistent
            # EMA. The accumulators must never be all_reduced in place:
            # doing so leaves the global sum in each rank's local tensor,
            # so the next epoch re-reduces its own history and the mean
            # freezes at the epoch-0 value forever.
            self._err_ep_sum = torch.zeros(n, device=self.device)
            self._err_ep_cnt = torch.zeros(n, device=self.device)
            self._err_ep_rung = torch.full((n,), -1.0, device=self.device)
            self._err_ema = torch.full((n,), -1.0, device=self.device)
            self._rung_vals = [float(v) for v in
                               self.model.denoising_step_list.tolist()]
            _nr = len(self._rung_vals)
            self._rung_sum = torch.zeros(_nr, device=self.device)
            self._rung_cnt = torch.zeros(_nr, device=self.device)
            # Per-DIRECTION rolling mean of the action residual (the reference
            # the weighting is relative to). Index order follows _ACT_DIRS.
            self._ACT_DIRS = ["cF", "cFR", "cR", "cBR", "cB", "cBL", "cL", "cFL", "cN"]
            self._act_res_ema = torch.full((len(self._ACT_DIRS),), -1.0,
                                           device=self.device)
            self._act_res_beta = float(getattr(config, "ode_act_res_beta", 0.9))
            self.max_epochs = int(getattr(config, "ode_curriculum_epochs", 10))
            self._log(f"Curriculum ON: {n} samples, epoch 0 = all directions, "
                      f"then 2-4 per group; rung-normalised error; "
                      f"stopping after {self.max_epochs} epochs")
        else:
            self.max_epochs = 0

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
        # VRFM prior/posterior/z-projection live OUTSIDE DDP, like the action
        # projections: they must be in the optimizer, manually all_reduced in
        # _sync_projection_grads, and clipped with the rest.
        for _vm in (getattr(self.model, "vrfm", None),
                    getattr(self.model, "z_modulation", None)):
            if _vm is not None:
                gen_params.extend(p for p in _vm.parameters() if p.requires_grad)
        for _hn in ("emd_head_scale", "emd_head_shift"):
            _hp = getattr(self.model, _hn, None)
            if _hp is not None and _hp.requires_grad:
                # The wrapper module is never .to(device)'d wholesale —
                # move the head params to the compute device BEFORE the
                # optimizer captures them, else their grads live on CPU
                # and the NCCL all-reduce crashes (job 5934528).
                _hp.data = _hp.data.to(self.device)
                gen_params.append(_hp)
        if not gen_params:
            raise RuntimeError("No trainable generator-group parameters.")

        # The EMD transport head is a 16-dim ZERO-INIT gate, not a DiT
        # weight: the fine-tuning lr (2e-6) caps its total movement at
        # ~1e-3 over a short run and weight decay actively pulls it back
        # to its zero init. Measured on jobs 5941221/5946996: after 500
        # steps |scale| <= 6e-4, i.e. the head never left zero and both
        # head arms were effectively no-ops. Give it its own group:
        # much larger lr, NO weight decay.
        head_params = [p for p in (
            getattr(self.model, "emd_head_scale", None),
            getattr(self.model, "emd_head_shift", None)) if p is not None]
        if head_params:
            hp_ids = {id(p) for p in head_params}
            gen_params = [p for p in gen_params if id(p) not in hp_ids]
            head_lr = float(getattr(cfg, "ode_emdhead_lr", 1e-2))
            optimizer = torch.optim.AdamW(
                [{"params": gen_params, "lr": lr, "weight_decay": weight_decay},
                 {"params": head_params, "lr": head_lr, "weight_decay": 0.0}],
                betas=(beta1, beta2),
            )
            self._log(f"Generator optimiser: AdamW lr={lr:.2e} "
                      f"params={len(gen_params)} + EMD-head group "
                      f"lr={head_lr:.2e} wd=0 params={len(head_params)}")
            return optimizer, self._build_critic_optimizer(critic_lr, beta1,
                                                           beta2, weight_decay)
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

    def _build_critic_optimizer(self, critic_lr, beta1, beta2, weight_decay):
        cm = self._critic_base()
        if cm is None:
            return None
        critic_params = [p for p in cm.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            critic_params, lr=critic_lr, betas=(beta1, beta2),
            weight_decay=weight_decay,
        )
        self._log(f"Critic optimiser: AdamW lr={critic_lr:.2e} "
                  f"params={len(critic_params)}")
        return opt

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
        if "act_res_ema" in ck and hasattr(self, "_act_res_ema"):
            self._act_res_ema = ck["act_res_ema"].clone()
        for _vn in ("vrfm", "z_modulation"):
            _vm = getattr(self.model, _vn, None)
            if _vm is not None:
                if _vn not in ck:
                    raise RuntimeError(
                        f"Resume checkpoint {target} has no '{_vn}' but "
                        f"ode_vrfm is on -- resuming would silently reset the "
                        f"prior/posterior and invalidate the run.")
                _vm.load_state_dict(ck[_vn])
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
        _cs = ck.get("curriculum")
        if _cs is not None and getattr(self, "curriculum", None) is not None:
            self._err_ema = _cs["err_ema"].to(self.device)
            self._curric_epochs_done = int(_cs["epochs_done"])
            self._curric_steps_in_epoch = int(_cs.get("steps_in_epoch", 0))
            self.curriculum.epoch_i = int(_cs.get("sampler_epoch", 0))
            self._rung_sum = _cs["rung_sum"].to(self.device)
            self._rung_cnt = _cs["rung_cnt"].to(self.device)
            if self._curric_epochs_done > 0:
                self.curriculum.set_errors(self._err_ema.cpu().tolist())
            self._log(f"Curriculum resumed at epoch {self._curric_epochs_done}")
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
            "config_name": self.config_name,
        }
        # ckpt_skip_optimizer=true drops optimizer states (~12GB of a
        # 17.9GB checkpoint). Weight-only saves (~6GB) are for probe-only
        # runs that never resume (e.g. short pilots) — during the
        # 2026-07-27 Lustre instability every 17.9GB save wedged while
        # every ~6GB save succeeded, so this is also a reliability lever.
        if not bool(getattr(self.config, "ckpt_skip_optimizer", False)):
            state["optimizer"] = self.optimizer.state_dict()
        cm = self._critic_base()
        if cm is not None:
            state["action_critic"] = cm.state_dict()
        if (self.critic_optimizer is not None
                and not bool(getattr(self.config, "ckpt_skip_optimizer", False))):
            state["critic_optimizer"] = self.critic_optimizer.state_dict()
        if self.model.action_projection is not None:
            state["action_projection"] = self.model.action_projection.state_dict()
        if hasattr(self, "_act_res_ema"):
            # Not saving this made every requeue restart actsplit at the -1
            # sentinel, i.e. plain error ranking for a whole epoch.
            state["act_res_ema"] = self._act_res_ema.detach().cpu()
        if self.model.action_token_projection is not None:
            state["action_token_projection"] = self.model.action_token_projection.state_dict()
        for _vn in ("vrfm", "z_modulation"):
            _vm = getattr(self.model, _vn, None)
            if _vm is not None:
                # The conditional PRIOR is needed at inference (the probe draws
                # z ~ p(.|x0,xt,t,a)); without it in the checkpoint a VRFM run
                # could not be sampled at all.
                state[_vn] = _vm.state_dict()
        if hasattr(base, "_state_probe") and base._state_probe is not None:
            state["state_probe"] = base._state_probe.state_dict()
        if getattr(self, "curriculum", None) is not None:
            # Without this a requeue restarts the curriculum at epoch 0
            # with no error history while global_step carries on, so
            # "10 epochs" would not be resume-stable.
            state["curriculum"] = {
                "err_ema": self._err_ema.detach().cpu(),
                "epochs_done": self._curric_epochs_done,
                "steps_in_epoch": self._curric_steps_in_epoch,
                "sampler_epoch": getattr(self.curriculum, "epoch_i", 0),
                "rung_sum": self._rung_sum.detach().cpu(),
                "rung_cnt": self._rung_cnt.detach().cpu(),
            }
        if getattr(self.model, "emd_head_scale", None) is not None:
            state["emd_head"] = {
                "scale": self.model.emd_head_scale.detach().cpu(),
                "shift": self.model.emd_head_shift.detach().cpu()}
        return state

    def _save_checkpoint(self, step: int) -> None:
        if not self.is_main:
            return
        path = self._checkpoint_path(step)
        state = self._build_checkpoint_state(step)
        # Atomic write: a wedged/killed torch.save must never leave a
        # truncated file under the final name (a Lustre write stall
        # corrupted a resume source in place on 2026-07-27). The .tmp
        # name does not match _CKPT_GLOB, so rotation never sees it.
        tmp = path.with_name(path.name + ".tmp")
        if bool(getattr(self.config, "ckpt_local_stage", False)):
            # Stage on node-local disk first: during the 2026-07-27
            # Lustre client instability, torch.save's write pattern
            # wedged repeatedly on Lustre while plain sequential copies
            # (dd/cp-style) succeeded — so serialize locally, then
            # stream the finished file out.
            import shutil
            local_dir = os.environ.get("TMPDIR", "/tmp")
            # Unique per process: co-scheduled jobs on one node staged
            # to the SAME /tmp filename and one unlink raced another's
            # copy (job 5915952). Suffix with job id + pid.
            _uid = f"{os.environ.get('SLURM_JOB_ID', 'x')}.{os.getpid()}"
            local_tmp = os.path.join(local_dir, f"{path.name}.{_uid}.tmp")
            torch.save(state, local_tmp)
            try:
                shutil.copyfile(local_tmp, str(tmp))
            finally:
                try:
                    os.unlink(local_tmp)
                except OSError:
                    pass
        else:
            torch.save(state, tmp)
        os.replace(tmp, path)
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
        for mod in (self.model.action_projection, self.model.action_token_projection,
                    getattr(self.model, "vrfm", None),
                    getattr(self.model, "z_modulation", None)):
            if mod is None:
                continue
            for p in mod.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        # EMD transport head: the 0.0*touch in the loss guarantees these
        # grads exist on every rank whenever the head is enabled, so the
        # collective can never desync across ranks.
        for _hn in ("emd_head_scale", "emd_head_shift"):
            _hp = getattr(self.model, _hn, None)
            if _hp is not None and _hp.grad is not None:
                dist.all_reduce(_hp.grad, op=dist.ReduceOp.AVG)

    def _generator_optim_step(self) -> float:
        self._sync_projection_grads()

        all_gen_params: list = []
        all_gen_params.extend(self._gen_base().parameters())
        if self.model.action_projection is not None:
            all_gen_params.extend(self.model.action_projection.parameters())
        if self.model.action_token_projection is not None:
            all_gen_params.extend(self.model.action_token_projection.parameters())
        for _vm in (getattr(self.model, "vrfm", None),
                    getattr(self.model, "z_modulation", None)):
            if _vm is not None:
                all_gen_params.extend(_vm.parameters())
        # The EMD transport head is clipped SEPARATELY: its objective can
        # spike (measured on job 5941221: head grad-norm 13-29 vs flow
        # ~0.2), and a joint clip_grad_norm_ would scale the flow map's
        # own gradients down by up to 3x on those steps — coupling the
        # two through the clipper and defeating the stop-gradient that
        # is supposed to keep the flow map untouched.
        head_params: list = []
        for _hn in ("emd_head_scale", "emd_head_shift"):
            _hp = getattr(self.model, _hn, None)
            if _hp is not None:
                head_params.append(_hp)
        grad_norm = 0.0
        if self.grad_clip and self.grad_clip > 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(all_gen_params, self.grad_clip))
            if head_params:
                torch.nn.utils.clip_grad_norm_(head_params, self.grad_clip)

        # Non-finite guard (pre-launch review): clip_grad_norm_ SCALES by the
        # total norm, so a single NaN/inf gradient anywhere poisons every
        # parameter on this step and nothing downstream detects it — on a 24h
        # unattended run that destroys all subsequent training. Skip the step
        # (grads are still zeroed) and log; byte-identical when healthy.
        # clip_grad_norm_ is deterministic given the grads, and the grads are
        # identical on all ranks after the DDP reduction + manual projection
        # sync above, so this branch is rank-uniform — no divergence risk.
        import math as _math
        if self.grad_clip and self.grad_clip > 0 and not _math.isfinite(grad_norm):
            log.warning(f"[optim] non-finite grad_norm={grad_norm}; SKIPPING "
                        f"optimizer step at global_step={self.global_step}")
            self.optimizer.zero_grad(set_to_none=True)
            return grad_norm

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
        # NOTE: do NOT short-circuit on ``not chunk_mask.any()``. The critic is
        # a SEPARATELY DDP-wrapped module whose forward+backward launches an
        # all-reduce collective. ``chunk_mask`` is drawn from a per-rank random
        # pool_idx, so an empty mask on ONE rank (while others are non-empty)
        # would make that rank skip the collective -> 32-rank deadlock. The
        # masked-mean below uses ``denom.clamp_min(1.0)``, so an all-empty mask
        # yields a well-defined 0 loss / 0 grad — safe to run unconditionally.
        # Only the config-level (rank-symmetric) None checks may early-return.
        if (
            critic_ddp is None
            or teacher_z_8d is None
            or self.critic_optimizer is None
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

        # ---- KV-cache AR ROLLOUT stage (teacher-aligned) ----
        # The rollout takes different batch keys (real seed + full action
        # stream + all committed chunks) and drives the cache path itself,
        # so it bypasses the packed teacher-forced forward entirely.
        if getattr(self.model, "ode_rollout", False):
            # cache_enabled=False: this OUTER autocast region spans the
            # rollout's no-grad seed/commit forwards, its grad forwards AND
            # its per-rung backwards. With the cast cache on, the no-grad
            # forwards populate the cache and the grad forwards reuse those
            # casts (no grad_fn -> weight gradients silently severed; and the
            # checkpoint recompute diverges -> the CheckpointError that killed
            # jobs 5989253/5998622/6004035). The DiT forward itself autocasts
            # in ode_rollout.fwd (also cache_enabled=False); this region only
            # needs to cover the projection/cond math, where caching buys
            # nothing measurable.
            with torch.amp.autocast(
                "cuda", dtype=self.dtype, enabled=self.mixed_precision,
                cache_enabled=False,
            ):
                loss, logs = self.model.rollout_loss(batch, step=self.global_step)
            # The rollout backwards each rung DURING the rollout (gradient
            # checkpointing recomputes against the LIVE kv_cache, so a deferred
            # backward recomputes against the wrong cache state). The returned
            # scalar is detached and for logging only.
            if not bool(logs.get("backwarded", False)):
                loss.backward()
            if getattr(self, "curriculum", None) is not None:
                _e = logs.get("per_sample_err_cf")
                _si = batch.get("sample_idx")
                _rg = logs.get("per_sample_rung_cf")
                if _e is not None and _si is not None:
                    dev = self._err_ep_sum.device
                    _si = _si.to(dev).view(-1)
                    _e = _e.detach().float().view(-1).to(dev)
                    assert _si.numel() == _e.numel(), (
                        f"rollout curriculum: {_si.numel()} ids vs "
                        f"{_e.numel()} errors")
                    self._err_ep_sum.index_add_(0, _si, _e)
                    self._err_ep_cnt.index_add_(0, _si, torch.ones_like(_e))
                    # MUST also fill the rung stats: the boundary divides by
                    # rung_mean, which clamps to 1e-8 when they are all zero and
                    # inflates the curriculum EMA by 1e8.
                    # rolling mean of the per-DIRECTION action residual
                    _ar = logs.get("act_residual")
                    _ad = batch.get("meta", {}).get("dir_idx")
                    if _ar is not None and _ad is not None:
                        _i = int(torch.as_tensor(_ad).reshape(-1)[0].item())
                        if 0 <= _i < self._act_res_ema.numel():
                            _v = float(_ar)
                            _b = self._act_res_beta
                            _prev = float(self._act_res_ema[_i])
                            self._act_res_ema[_i] = (
                                _v if _prev < 0 else _b * _prev + (1 - _b) * _v)
                    if _rg is not None:
                        _rg = _rg.detach().float().view(-1).to(dev)
                        if _rg.numel() == 1 and _e.numel() > 1:
                            _rg = _rg.expand_as(_e)
                        self._err_ep_rung.index_copy_(0, _si, _rg)
                        rv = torch.tensor(self._rung_vals, device=dev)
                        ridx = (_rg.view(-1, 1) - rv.view(1, -1)).abs().argmin(dim=1)
                        self._rung_sum.index_add_(0, ridx, _e)
                        self._rung_cnt.index_add_(0, ridx, torch.ones_like(_e))
            grad_norm = self._generator_optim_step()
            _lt = float(loss.detach().item())
            _lc = float(logs.get("ode_loss_clean", loss).detach().item())
            _lf = float(logs.get("ode_loss_cf", loss).detach().item())
            if self.is_distributed:
                # Match the packed path: log WORLD means, or roll arms are not
                # comparable with every other arm.
                _t = torch.tensor([_lt, _lc, _lf], device=self.device)
                dist.all_reduce(_t, op=dist.ReduceOp.AVG)
                _lt, _lc, _lf = (float(x) for x in _t.tolist())
            return {
                "loss/total": _lt,
                "loss/clean": _lc,
                "loss/cf": _lf,
                "loss/ode_clean": _lc,
                "loss/ode_cf": _lf,
                "train/lr": self.optimizer.param_groups[0]["lr"],
                "train/state_z_loss_clean": 0.0,
                "train/gen_action_loss_clean": 0.0,
                "train/z_guidance_scale": 0.0,
                "train/grad_norm": grad_norm,
                # VRFM diagnostics. Without these the run is undiagnosable:
                # nothing distinguishes "z is informative" from "z collapsed"
                # from "q is leaking the target". Watch sigma_q -- pinned at
                # 1.0 while z_absmean grows means z is noise, not information.
                **{f"vrfm/{_k}": float(_v) for _k, _v in logs.items()
                   if _k.startswith("vrfm_")},
                # Attractor-tracker diagnostics (review H2): without these the
                # rollar smoke cannot demonstrate whether the gates opened and
                # the actuation path ran at all.
                **{f"attr/{_k}": float(_v) for _k, _v in logs.items()
                   if _k.startswith("attr_")},
                # Ratio of the KL term to the base loss: the single number that
                # says whether beta is doing anything at all (see H1).
                **({"vrfm/kl_over_base": float(logs.get("vrfm_kl_cf", 0.0))
                    * float(getattr(self.model, "ode_vrfm_beta", 0.0))
                    / max(abs(_lf), 1e-9)} if logs.get("vrfm_kl_cf") else {}),
            }

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
                clean_x_gt_cf=batch.get("clean_x_gt_cf"),
                step=self.global_step,
            )
        loss.backward()

        # ---- Curriculum: attribute this step's error to its samples ----
        if getattr(self, "curriculum", None) is not None:
            _e = logs.get("per_sample_err_cf")
            _si = batch.get("sample_idx")
            _rg = logs.get("per_sample_rung_cf")
            if _e is not None and _si is not None:
                dev = self._err_ep_sum.device
                _si = _si.to(dev).view(-1)
                _e = _e.detach().float().view(-1).to(dev)
                assert _si.numel() == _e.numel(), (
                    f"curriculum: {_si.numel()} sample ids vs {_e.numel()} "
                    "errors — silent truncation would mis-attribute errors")
                self._err_ep_sum.index_add_(0, _si, _e)
                self._err_ep_cnt.index_add_(0, _si,
                                            torch.ones_like(_e))
                if _rg is not None:
                    _rg = _rg.detach().float().view(-1).to(dev)
                    # remember the rung each sample was scored at, and
                    # build the per-rung reference used to normalise it
                    self._err_ep_rung.index_copy_(0, _si, _rg)
                    rv = torch.tensor(self._rung_vals, device=dev)
                    ridx = (_rg.view(-1, 1) - rv.view(1, -1)).abs().argmin(dim=1)
                    self._rung_sum.index_add_(0, ridx, _e)
                    self._rung_cnt.index_add_(0, ridx, torch.ones_like(_e))

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

        # Online dual-CD losses (Causal-Forcing++ teacher-CD + student self-CD).
        # These keep tightening even when the ODE-regression MSE plateaus early,
        # so they're the clearest training-time progress signal. Computed in the
        # model's generator_loss; surface them here.
        for _k in ("cd_teacher_loss_raw", "cd_student_loss_raw",
                   "cd_term_weighted", "cd_weight_effective"):
            if _k in logs:
                out[f"train/{_k}"] = float(logs[_k])

        # Latent high-frequency energy of the predicted x0 (clean branch): a
        # cheap, no-decode sharpness proxy. ODE-MSE is low-freq-dominated and
        # flattens fast; this keeps rising as the student learns fine detail.
        _px = logs.get("pred_x0_detached_clean")
        if isinstance(_px, torch.Tensor) and _px.dim() == 5:
            _f = _px.float()
            _dh = (_f[..., 1:, :] - _f[..., :-1, :]).abs().mean()
            _dw = (_f[..., :, 1:] - _f[..., :, :-1]).abs().mean()
            out["train/pred_x0_hf_energy"] = float((_dh + _dw).item())

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

    # ------------------------------------------------------------------
    # Optional: in-loop causal-chain (AR rollout) eval on the Madrid ride
    # ------------------------------------------------------------------

    def _ensure_iqa_metrics(self):
        """Lazily build + cache the pyiqa MUSIQ/NIQE metrics on ``self``.

        Built once on the eval device and reused across evals. Returns the
        ``(musiq, niqe)`` tuple, or ``(None, None)`` if pyiqa is unavailable
        (in which case the chain eval still logs the video, just no IQA).
        """
        if getattr(self, "_iqa_metrics", None) is not None:
            return self._iqa_metrics
        musiq = niqe = None
        try:
            import pyiqa  # local import: not a hard dependency for import-time
            musiq = pyiqa.create_metric("musiq", device=self.device)
            niqe = pyiqa.create_metric("niqe", device=self.device)
        except Exception as exc:
            log.warning("[madrid_chain] pyiqa metrics unavailable (%r); "
                        "skipping IQA, will still log the rollout video.", exc)
        self._iqa_metrics = (musiq, niqe)
        return self._iqa_metrics

    @torch.no_grad()
    def _madrid_ar_rollout(
        self,
        prompt_embeds: torch.Tensor,    # [1, L, D]
        noisy_fa_full: torch.Tensor,    # [1, total_frames, A]
        initial_latents: torch.Tensor,  # [1, npb, C, H, W]  (1 GT chunk seed)
        num_gen_chunks: int,
        fifo_size: int = 3,
    ) -> torch.Tensor:
        """Minimal per-pass AR-refresh rollout on the LIVE student model.

        Mirrors ``utils/eval_causal_AR_chain.py``'s ``per_pass`` recipe but
        runs against the in-memory ``self.model.generator`` (DDP-wrapped
        student DiT) — NO checkpoint is reloaded. Every denoise step runs
        the train-path forward (``clean_x=None``, ``kv_cache=None``) over a
        ``(N+1)*npb``-frame window so context K/V are recomputed from raw
        latents each pass (chain-style freshness). Returns
        ``[1, npb + num_gen_chunks*npb, C, H, W]`` (GT seed at the front).
        """
        model = self.model
        npb = int(model.num_frame_per_block)
        device = self.device
        dtype = self.dtype

        # The generator is DDP-wrapped (see __init__), but this eval is
        # RANK-0 ONLY: calling the DDP forward here would launch a collective
        # with no peers and hang, and ``DDP.model`` doesn't exist. Unwrap to
        # the raw generator module for both attribute access and forward.
        gen = self._gen_base()
        base_dit = gen.model
        if hasattr(base_dit, "get_base_model"):
            base_dit = base_dit.get_base_model()

        ts = model.denoising_step_list.detach().to(device=device, dtype=torch.float32)
        ts, _ = torch.sort(ts, descending=True)
        scheduler = model.scheduler
        scheduler.sigmas = scheduler.sigmas.to(device)

        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        noisy_fa_full = noisy_fa_full.to(device=device, dtype=dtype)
        seed = initial_latents.to(device=device, dtype=dtype)
        B = 1
        C, H, W = int(seed.shape[2]), int(seed.shape[3]), int(seed.shape[4])

        fifo = [seed]                       # list of [1, npb, C, H, W]
        generated = []
        prev_window_blocks = -1

        for chunk_idx in range(num_gen_chunks):
            n_ctx = min(len(fifo), fifo_size)
            window_blocks = n_ctx + 1
            window_frames = window_blocks * npb

            cur_lo = (1 + chunk_idx) * npb              # 1 seed chunk in front
            cur_hi = cur_lo + npb
            ctx_lo = cur_lo - n_ctx * npb

            # Force block_mask rebuild when the window size changes (warmup
            # while the FIFO fills). The DiT keys its cached mask only on
            # frame_seqlen, not frame count, so we must invalidate manually.
            if window_blocks != prev_window_blocks:
                base_dit.block_mask = None
                prev_window_blocks = window_blocks

            ctx_cat = torch.cat(fifo[-n_ctx:], dim=1)  # [1, n_ctx*npb, C, H, W]
            fa_window = noisy_fa_full[:, ctx_lo:cur_hi].contiguous()

            # z-conditioning over the full window; the "noisy" branch keys
            # are what the plain (clean_x=None) train path consumes.
            cond = {"prompt_embeds": prompt_embeds}
            if model.use_adaln and model.action_projection is not None:
                cond["_action_modulation"] = model.action_projection(
                    fa_window, num_frames=window_frames,
                )
            if model.use_action_tokens and model.action_token_projection is not None:
                cond["_action_tokens"] = model.action_token_projection(fa_window)

            t_ctx = torch.zeros(B, n_ctx * npb, device=device, dtype=torch.float32)
            x_cur = torch.randn(B, npb, C, H, W, device=device, dtype=torch.float32).to(dtype)

            pred_window = None
            for d_idx in range(int(ts.shape[0])):
                t_val = float(ts[d_idx].item())
                t_cur = torch.full((B, npb), t_val, device=device, dtype=torch.float32)
                tt = torch.cat([t_ctx, t_cur], dim=1)
                x_full = torch.cat([ctx_cat, x_cur], dim=1)
                with torch.amp.autocast("cuda", dtype=dtype):
                    out = gen(
                        noisy_image_or_video=x_full,
                        conditional_dict=cond,
                        timestep=tt,
                        clean_x=None,
                        aug_t=None,
                    )
                pred_window = out[1]
                if d_idx < int(ts.shape[0]) - 1:
                    next_t = float(ts[d_idx + 1].item())
                    cur_pred = pred_window[:, n_ctx * npb:]
                    flat = cur_pred.flatten(0, 1).float()
                    flat_noise = torch.randn_like(flat)
                    flat_t = torch.full((flat.shape[0],), next_t,
                                        device=device, dtype=torch.float32)
                    x_cur = (
                        scheduler.add_noise(flat, flat_noise, flat_t)
                        .view(B, npb, C, H, W).to(dtype)
                    )

            cur_pred = pred_window[:, n_ctx * npb:].detach()
            generated.append(cur_pred.float())
            if len(fifo) >= fifo_size:
                fifo.pop(0)
            fifo.append(cur_pred.to(dtype))

        # Invalidate the mask we built — the training forward owns it.
        base_dit.block_mask = None
        full = torch.cat([seed.float()] + generated, dim=1)
        return full

    def _run_madrid_chain_eval(self) -> None:
        """Roll the live student AR-refresh on the Madrid ride and log a
        rollout video + IQA metrics to the same wandb run/step.

        SAFETY: wrapped in its own try/except — logs a warning and returns
        on ANY error. It must NEVER raise (the trainer aborts after 2
        consecutive eval failures, and this addition must not contribute).
        """
        try:
            import numpy as np

            cfg = self.config
            npb = int(self.model.num_frame_per_block)
            num_gen_chunks = int(getattr(cfg, "madrid_chain_gen_chunks", 7))
            fifo_size = int(getattr(cfg, "madrid_chain_fifo_size", 3))
            offset = int(getattr(cfg, "madrid_chain_offset", 0))
            zarr_name = str(getattr(
                cfg, "madrid_chain_zarr", "20240216101235.zarr"))
            encoded_root = str(getattr(
                cfg, "madrid_chain_encoded_root",
                "/projects/u6ex/fbots/frodobots_encoded_weunz"))
            caption_root = str(getattr(
                cfg, "caption_root",
                "/projects/u6ex/fbots/frodobots_captions/train"))
            motion_root = str(getattr(
                cfg, "madrid_chain_motion_root",
                "/projects/u6ex/fbots/frodobots_motion"))
            ss_vae_ckpt = str(getattr(
                cfg, "ss_vae_checkpoint",
                "action_query/checkpoints/ss_vae_8free.pt"))
            action_dims = list(getattr(cfg, "action_dims", [2, 7]))
            fps = int(getattr(cfg, "madrid_chain_fps", 12))

            total_frames = (1 + num_gen_chunks) * npb

            # Load 1 GT seed chunk + the real Madrid z-action stream. This
            # path builds its own CPU ss_vae and reads latents directly from
            # the zarr — NO student checkpoint is reloaded.
            import sys
            from pathlib import Path as _Path
            _repo = _Path(__file__).resolve().parents[2]
            for p in (str(_repo), str(_repo / "utils")):
                if p not in sys.path:
                    sys.path.insert(0, p)
            from utils.eval_causal_AR import load_per_rank_ride_ar

            (
                initial_latents_full,  # [1, total_frames, C, H, W]
                prompt_embeds,         # [1, L, D]
                noisy_fa_full,         # [1, total_frames, A]
                _ride_meta,
            ) = load_per_rank_ride_ar(
                zarr_basename=zarr_name,
                latent_start_offset=offset,
                total_frames=total_frames,
                manifest_path="",  # force disk fallback; no manifest dependency
                encoded_root=encoded_root,
                caption_root=caption_root,
                motion_root=motion_root,
                ss_vae_checkpoint=ss_vae_ckpt,
                action_dims=action_dims,
                device=self.device,
            )

            seed = initial_latents_full[:, :npb]
            # Deterministic rollout noise; the outer _run_eval_once guard
            # restores training RNG afterwards.
            torch.manual_seed(self.global_step + 777)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.global_step + 777)

            # Eval-only: the 7-chunk rollout + VAE decode must NOT build an
            # autograd graph (it would waste ~GBs and risk OOM on the full
            # 1.3B run). Everything here is inference; outputs are detached.
            with torch.no_grad():
                full_latents = self._madrid_ar_rollout(
                    prompt_embeds=prompt_embeds,
                    noisy_fa_full=noisy_fa_full,
                    initial_latents=seed,
                    num_gen_chunks=num_gen_chunks,
                    fifo_size=fifo_size,
                )
                gen_latents = full_latents[:, npb:]  # drop the GT seed for scoring

                # Decode to pixels (reuse the eval decode convention: prepend a
                # dummy frame, drop it after decode; output [F, H, W, 3] uint8).
                frozen_vae = self.model._frozen_vae
                dummy = gen_latents[:, 0:1]
                lat_wd = torch.cat([dummy, gen_latents], dim=1)
                px = frozen_vae.decode_to_pixel(lat_wd.float())[:, 1:, ...]
                vid01 = (0.5 * (px.float() + 1.0)).clamp(0, 1)  # [1, F, 3, H, W]
                vid01 = vid01[0]                                 # [F, 3, H, W]
            vid_np = (vid01.cpu().numpy() * 255).astype(np.uint8)
            if vid_np.shape[1] != 3:
                vid_np = vid_np.transpose(0, 3, 1, 2)        # ensure [F,3,H,W]

            payload = {"eval/step": self.global_step}

            # IQA on the decoded rollout frames ([F, 3, H, W] in [0,1]).
            musiq, niqe = self._ensure_iqa_metrics()
            frames01 = vid01.to(self.device)  # [F, 3, H, W]
            n_f = int(frames01.shape[0])
            if musiq is not None and n_f > 0:
                try:
                    ms = []
                    for i in range(n_f):
                        ms.append(float(musiq(frames01[i:i + 1]).item()))
                    ms_arr = np.asarray(ms, dtype=np.float64)
                    third = max(1, n_f // 3)
                    early = float(ms_arr[:third].mean())
                    late = float(ms_arr[-third:].mean())
                    payload["eval/madrid_chain_musiq"] = float(ms_arr.mean())
                    payload["eval/madrid_chain_musiq_late_minus_early"] = late - early
                except Exception as iqa_exc:
                    log.warning("[madrid_chain] MUSIQ failed: %r", iqa_exc)
            if niqe is not None and n_f > 0:
                try:
                    ns = []
                    for i in range(n_f):
                        ns.append(float(niqe(frames01[i:i + 1]).item()))
                    payload["eval/madrid_chain_niqe"] = float(
                        np.asarray(ns, dtype=np.float64).mean())
                except Exception as iqa_exc:
                    log.warning("[madrid_chain] NIQE failed: %r", iqa_exc)

            # Rollout video.
            try:
                payload["eval/madrid_causal_chain"] = wandb.Video(
                    vid_np, fps=fps, format="mp4",
                )
            except Exception as vexc:
                log.warning("[madrid_chain] video encode failed: %r", vexc)

            if self._wandb_run is not None:
                self._wandb_run.log(payload, step=self.global_step)
            log.info(
                "[madrid_chain step=%d] gen_chunks=%d frames=%d musiq=%s "
                "niqe=%s late-early=%s",
                self.global_step, num_gen_chunks, n_f,
                payload.get("eval/madrid_chain_musiq"),
                payload.get("eval/madrid_chain_niqe"),
                payload.get("eval/madrid_chain_musiq_late_minus_early"),
            )
        except Exception as exc:
            # NEVER propagate — this is an optional diagnostic.
            log.warning(
                "[madrid_chain step=%d] skipped (error: %r)",
                getattr(self, "global_step", -1), exc,
            )
            return

    def _run_eval_once(self) -> None:
        """Actual eval body. See ``_maybe_eval`` for orchestration.

        RNG hygiene: the full eval (generate_eval + cotracker + critic
        forwards) runs under a save/restore guard so rank-0's deterministic
        seeding during eval cannot silently diverge its RNG from the other
        ranks' training RNG state. Without this guard each eval would
        leave rank-0's RNG advanced relative to the rest of the world.
        """
        if getattr(self.model, "ode_rollout", False):
            # The rollout dataset item has no z_clean/z_noisy/clean_x_gt, so the
            # teacher-forced eval below cannot run on it. Skipping is REQUIRED:
            # two consecutive eval failures raise, and _maybe_eval runs before
            # the save_interval block, so the run would die without ever writing
            # the final checkpoint.
            log.info("[eval] skipped: ode_rollout batches are not "
                     "teacher-forced pairs (no z_clean/clean_x_gt)")
            return
        idx = (self.global_step // max(self.eval_interval, 1)) % max(self.eval_num_samples, 1)
        pair = self.dataset[int(idx) % len(self.dataset)]

        device = self.device

        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(device) if torch.cuda.is_available() else None
        try:
            self._run_eval_once_inner(pair, device)
            # Optional in-loop causal-chain (AR-rollout) eval on the Madrid
            # ride. Flag-gated and fully self-contained: it has its OWN
            # try/except inside ``_run_madrid_chain_eval`` and NEVER raises,
            # so a failure here can't trip the trainer's consecutive-eval-
            # failure abort. Runs inside this RNG save/restore guard so its
            # deterministic seeding cannot leak into rank-0's training RNG.
            if getattr(self.config, "madrid_chain_eval_enabled", False):
                self._run_madrid_chain_eval()
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

            # ---- Curriculum epoch boundary --------------------------------
            if getattr(self, "curriculum", None) is not None:
                self._curric_steps_in_epoch += 1
                if self._curric_steps_in_epoch >= len(self.loader):
                    # Reduce COPIES: an in-place all_reduce would leave the
                    # global sum in every rank's accumulator and the next
                    # epoch would re-reduce its own history (weights *=
                    # world_size each epoch), freezing the mean at epoch 0.
                    s = self._err_ep_sum.clone()
                    c = self._err_ep_cnt.clone()
                    rg = self._err_ep_rung.clone()
                    rs = self._rung_sum.clone()
                    rc = self._rung_cnt.clone()
                    if self.is_distributed:
                        dist.all_reduce(s, op=dist.ReduceOp.SUM)
                        dist.all_reduce(c, op=dist.ReduceOp.SUM)
                        dist.all_reduce(rg, op=dist.ReduceOp.MAX)
                        dist.all_reduce(rs, op=dist.ReduceOp.SUM)
                        dist.all_reduce(rc, op=dist.ReduceOp.SUM)
                    obs = c > 0
                    raw = s / c.clamp(min=1)
                    # RUNG NORMALISATION: divide by the mean error at the
                    # rung the sample was scored at, so "hard direction"
                    # means hard relative to its own noise level rather
                    # than "drew the noisiest timestep".
                    rung_mean = (rs / rc.clamp(min=1)).clamp(min=1e-8)
                    rv = torch.tensor(self._rung_vals, device=raw.device)
                    ridx = (rg.view(-1, 1) - rv.view(1, -1)).abs().argmin(dim=1)
                    norm = raw / rung_mean[ridx]
                    epoch_err = torch.where(obs, norm,
                                            torch.full_like(norm, -1.0))
                    # EMA so a single unlucky draw cannot condemn a
                    # direction for the whole run; unobserved keep history.
                    prev = self._err_ema
                    self._err_ema = torch.where(
                        obs, torch.where(prev < 0, epoch_err,
                                         0.5 * prev + 0.5 * epoch_err), prev)
                    self.curriculum.set_errors(
                        self._err_ema.detach().cpu().tolist())
                    if hasattr(self, "_act_res_ema"):
                        # _act_res_ema is updated only from the direction THIS
                        # rank happened to draw, so it differs per rank. The
                        # batch sampler must be identical on every rank -- an
                        # actsplit ranked on divergent references makes ranks
                        # select different groups, desyncing the global batch
                        # and eventually hanging in NCCL when their epoch
                        # boundaries land on different steps. Average across
                        # ranks over the entries that were actually observed
                        # (-1 is the never-seen sentinel).
                        _e = self._act_res_ema.detach().clone().to(self.device)
                        _m = (_e >= 0).float()
                        _v = torch.where(_m > 0, _e, torch.zeros_like(_e))
                        if self.is_distributed:
                            dist.all_reduce(_v, op=dist.ReduceOp.SUM)
                            dist.all_reduce(_m, op=dist.ReduceOp.SUM)
                        _e = torch.where(_m > 0, _v / _m.clamp(min=1.0),
                                         torch.full_like(_v, -1.0))
                        self._act_res_ema = _e.cpu()
                        self.curriculum.act_ref = {
                            d: float(v) for d, v in
                            zip(self._ACT_DIRS, _e.tolist())}
                    self._curric_epochs_done += 1
                    seen = int(obs.sum().item())
                    if self.is_main:
                        _m = epoch_err[obs]
                        log.info(
                            "[curriculum] epoch %d done at step %d | seen "
                            "%d/%d | rung-norm err mean %.4f min %.4f max "
                            "%.4f | rung means %s",
                            self._curric_epochs_done, self.global_step, seen,
                            epoch_err.numel(),
                            float(_m.mean()) if _m.numel() else -1,
                            float(_m.min()) if _m.numel() else -1,
                            float(_m.max()) if _m.numel() else -1,
                            [round(float(x), 5) for x in rung_mean.tolist()])
                    # reset per-epoch accumulators (EMA carries history)
                    self._err_ep_sum.zero_(); self._err_ep_cnt.zero_()
                    self._err_ep_rung.fill_(-1.0)
                    # Reset the rung reference too: as a LIFETIME mean it makes
                    # the normalised error drift monotonically downward, `bad`
                    # empties against the absolute threshold, and the curriculum
                    # silently becomes random sampling.
                    self._rung_sum.zero_(); self._rung_cnt.zero_()
                    if self.is_main and hasattr(self, "_act_res_ema"):
                        _seen = self._act_res_ema >= 0
                        if bool(_seen.any()):
                            _m = float(self._act_res_ema[_seen].mean())
                            log.info("[actres] rolling mean per direction "
                                     "(avg %.4f): %s", _m,
                                     {d: round(float(v), 4) for d, v in
                                      zip(self._ACT_DIRS,
                                          self._act_res_ema.tolist()) if v >= 0})
                    self._curric_steps_in_epoch = 0
                    if self._curric_epochs_done >= self.max_epochs:
                        if self.is_main:
                            log.info("[curriculum] reached %d epochs — stopping",
                                     self.max_epochs)
                        self._save_checkpoint(self.global_step)
                        break

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
