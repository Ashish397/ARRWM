"""Phase-1 rolling-staircase DMD trainer for action-aware video world models.

One ride per DDP rank per step. Per-ride flow:

  1) Load ride metadata (zarr_path, prompt_embeds, n_latent_frames) from
     `ZarrRideDataset`.
  2) Lazily load the full-ride latents + z-actions via the dataset's
     `load_latent_chunk` and `encode_z_actions_window` helpers.
  3) Run the `RollingStaircaseTrainingPipeline`:
       - Prime the KV cache with 9 clean GT frames (t=0 forwards).
       - Warmup: lockstep-denoise 4 pure-noise blocks all the way to clean.
       - Transition: commit slot 0 to KV, re-noise sliced clean slots 1-3 to
         [250, 500, 750] and sample fresh noise for new slot 3.
       - Steady-state rolling: one action per step, grad on slot 0 + one
         random aux slot in {1,2,3}, yielded as RollingStepOutput records.
  4) For each emitted record, compute DMD + GAN + optional MSE and accumulate
     gradients into the generator; apply `optimizer.step()` after each ride
     (respecting `max_grad_norm`).
  5) Checkpoint + W&B log + periodic visualization.

This trainer is deliberately self-contained (does NOT inherit from the large
`CausalLoRADiffusionTrainer`): the phase-1 surface is much smaller and the
ride-level loop differs substantially.
"""
from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torch.amp.autocast_mode import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from omegaconf import OmegaConf

# wandb is optional (disabled on non-main ranks or via config).
try:
    import wandb  # type: ignore
    _HAS_WANDB = True
except Exception:  # pragma: no cover
    wandb = None  # type: ignore
    _HAS_WANDB = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.dmd2b2blam_staircase import DMD2B2BLAM_Staircase
from pipeline.rolling_staircase_training import (
    RollingStaircaseTrainingPipeline,
    RollingStepOutput,
)
from utils.multislot_vis import MultislotVisRecorder
from utils.ride_slot_batcher import PerSlotRideBatcher
from utils.zarr_dataset import ZarrRideDataset
from utils.distributed import barrier


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _ride_metadata_collate(batch: List[dict]) -> List[dict]:
    """Keep ride metadata as a Python list; latents are loaded in-ride."""
    return batch


def _load_ride_tensors(
    dataset: ZarrRideDataset,
    meta: dict,
    device: torch.device,
    dtype: torch.dtype,
    *,
    action_dims: Optional[List[int]] = None,
    max_frames: Optional[int] = None,
) -> Optional[Dict[str, torch.Tensor]]:
    """Load (latents, z_actions, prompt_embeds) for one ride."""
    zarr_path = meta["zarr_path"]
    n_latent_frames = int(meta["n_latent_frames"])
    if max_frames is not None:
        n_latent_frames = min(n_latent_frames, int(max_frames))
    if n_latent_frames <= 0:
        return None

    try:
        latents = ZarrRideDataset.load_latent_chunk(zarr_path, 0, n_latent_frames)
    except Exception as e:
        logging.warning("load_latent_chunk failed for %s: %s", zarr_path, e)
        return None

    try:
        resolved = zarr_path
        if resolved not in dataset._attrs_by_path:  # pylint: disable=protected-access
            resolved = str(Path(zarr_path).resolve())
        z_actions = dataset.encode_z_actions_window(resolved, n_latent_frames, 0, n_latent_frames)
    except Exception as e:
        logging.warning("encode_z_actions_window failed for %s: %s", zarr_path, e)
        return None

    if action_dims is not None:
        z_actions = z_actions[..., action_dims]

    prompt_embeds = meta["prompt_embeds"]
    if not isinstance(prompt_embeds, torch.Tensor):
        prompt_embeds = torch.tensor(prompt_embeds)
    if prompt_embeds.dim() == 2:
        prompt_embeds = prompt_embeds.unsqueeze(0)

    return {
        "latents": latents.unsqueeze(0).to(device=device, dtype=dtype),            # [1, T, C, H, W]
        "z_actions": z_actions.unsqueeze(0).to(device=device, dtype=dtype),        # [1, T, action_dim]
        "prompt_embeds": prompt_embeds.to(device=device, dtype=dtype),             # [1, L, C_txt]
        "zarr_path": zarr_path,
        "n_latent_frames": n_latent_frames,
    }


# ---------------------------------------------------------------------------
# Main trainer class
# ---------------------------------------------------------------------------
class RollingStaircaseDMDTrainer:
    """Phase-1 rolling-staircase DMD trainer."""

    def __init__(self, config_path_or_obj):
        """Construct the trainer from either a YAML path or a pre-loaded
        OmegaConf DictConfig.

        Accepting the in-memory object avoids the race-condition that
        bit us on first launch: when ``main()`` rewrote a merged tmp
        YAML on every rank concurrently, a late rank could ``load``
        another rank's half-written truncation and quietly drop keys
        like ``denoising_loss_type``. Passing the DictConfig directly
        from ``main()`` sidesteps disk entirely.
        """
        if isinstance(config_path_or_obj, (str, Path)):
            self.config = OmegaConf.load(str(config_path_or_obj))
            self.config_path = str(config_path_or_obj)
        else:
            # DictConfig / ListConfig / dict — treat as already loaded.
            self.config = config_path_or_obj
            self.config_path = "<in-memory>"

        # DDP setup.
        self._setup_distributed()
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)

        self.is_main_process = self.rank == 0
        self._configure_logging()

        # Mixed precision dtype.
        self.use_mixed_precision = bool(getattr(self.config, "mixed_precision", True))
        self.autocast_dtype = torch.bfloat16 if self.use_mixed_precision else torch.float32
        self.dtype = torch.bfloat16 if self.use_mixed_precision else torch.float32

        if self.is_main_process:
            logging.info("Rolling-Staircase DMD: world_size=%d rank=%d local_rank=%d",
                         self.world_size, self.rank, self.local_rank)
            logging.info("Config:\n%s", OmegaConf.to_yaml(self.config))

        # Build model, pipeline, dataset, optimizer.
        self._build_model()
        self._build_pipeline()
        self._build_dataset()
        self._build_optimizer()
        self._build_action_teacher()
        self._init_wandb()
        self._init_vis_recorder()

        # Training state.
        self.step = 0
        self._maybe_resume()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _setup_distributed(self) -> None:
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            # Bump the NCCL watchdog / barrier timeout from PyTorch's
            # default 10 min to 60 min. Rationale: rank 0 periodically
            # blocks for several minutes during the visualization flush
            # (VAE decode + mp4 encode + wandb upload), while every other
            # rank waits at an all-reduce on the very next rolling step.
            # At the default 10 min this tripped the watchdog on jobs
            # 4249138 / 4251723 / 4251724 when the flush ran long.
            # Configurable via `nccl_timeout_minutes` (default 60).
            from datetime import timedelta
            timeout_min = int(
                getattr(self.config, "nccl_timeout_minutes", 60)
            )
            dist.init_process_group(
                backend="nccl",
                timeout=timedelta(minutes=timeout_min),
            )
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = int(os.environ.get("LOCAL_RANK", self.rank % torch.cuda.device_count()))
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0

    def _configure_logging(self) -> None:
        level = logging.INFO if self.is_main_process else logging.WARNING
        logging.basicConfig(
            level=level,
            format=f"%(asctime)s [rank {self.rank}] %(levelname)s %(message)s",
        )

    def _build_model(self) -> None:
        # Alias a few config fields expected by the DMD model's `__init__`
        # (derived from BaseModel). The base model reads `args` via attribute
        # access, and OmegaConf's DictConfig supports that pattern.
        args = self.config
        if not hasattr(args, "text_pre_encoded"):
            OmegaConf.update(args, "text_pre_encoded", True, merge=True)
        if not hasattr(args, "mixed_precision"):
            OmegaConf.update(args, "mixed_precision", self.use_mixed_precision, merge=True)

        model = DMD2B2BLAM_Staircase(args=args, device=self.device)
        # Ensure the modules and projection are on device+dtype.
        model.generator.model.to(device=self.device, dtype=self.dtype)
        model.fake_score.model.to(device=self.device, dtype=self.dtype)
        model.real_score.model.to(device=self.device, dtype=self.dtype)
        if model.action_projection is not None:
            model.action_projection.to(device=self.device, dtype=self.dtype)
        # Auxiliary probe / critic live on the generator wrapper (not on
        # ``generator.model``), so the .to(dtype) above does not touch them.
        # Their LayerNorm / Linear weights stay at the default dtype (fp32)
        # and crash with "expected scalar type BFloat16 but found Float"
        # when called inside the bf16 autocast context during training.
        # Cast explicitly here to match the DiT's dtype.
        if getattr(model, "state_probe", None) is not None:
            model.state_probe.to(device=self.device, dtype=self.dtype)
        if getattr(model, "action_critic", None) is not None:
            model.action_critic.to(device=self.device, dtype=self.dtype)
        # The VAE is used only by the action teacher and the visualizer —
        # neither path does a manual .to(). Keep it in fp32 (it's loaded
        # that way and has fp32 conv biases; casting to bf16 breaks the
        # upstream VAE decoder), but ensure it's on the correct GPU.
        if getattr(model, "vae", None) is not None:
            try:
                model.vae.to(device=self.device)
            except AttributeError:
                # Some VAE wrappers expose ``.model`` instead of being nn.Modules.
                inner_vae = getattr(model.vae, "model", None)
                if inner_vae is not None:
                    inner_vae.to(device=self.device)
        self.model = model

        # Wrap generator's inner DiT with DDP for data-parallel training.
        # `debug_find_unused_parameters` is a dev-only flag; when True, DDP
        # walks the graph each backward and errors on rank-asymmetric
        # parameter touches — useful for catching conditional branches
        # that silently break the `no_sync` accumulation pattern. Flip
        # back to False for production throughput once clean.
        debug_fup = bool(getattr(self.config, "debug_find_unused_parameters", False))
        self.generator_ddp: Optional[DDP] = None
        self.fake_score_ddp: Optional[DDP] = None
        if self.world_size > 1:
            self.generator_ddp = DDP(
                model.generator.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=debug_fup,
                broadcast_buffers=False,
            )
            if debug_fup and self.is_main_process:
                logging.warning(
                    "DDP find_unused_parameters=True (debug mode). "
                    "Expect ~5-15%% slower backwards; disable for prod."
                )
            # Swap: the pipeline needs to call model.generator(...) (the wrapper),
            # which in turn calls `model.generator.model` — so we redirect that
            # attribute to the DDP-wrapped module. Since the wrapper uses
            # `self.model(...)` internally, we need to point the wrapper's
            # `.model` attribute to the DDP wrapper. WanDiffusionWrapper
            # treats `self.model` as the inner module directly.
            model.generator.model = self.generator_ddp  # type: ignore

            # Fake-score DDP: only wrap if fake-score updates are enabled
            # for this remit. When frozen we save the allreduce cost by
            # skipping DDP entirely (params have requires_grad=False and
            # no grads would be produced).
            if bool(getattr(self.config, "fake_score_updates_enabled", False)):
                # Fake-score forward touches DIFFERENT parameters depending
                # on which slot (and whether CFG is on) — we enable
                # find_unused_parameters to be safe here, gated on the
                # same debug flag. With shared forward (Option A) all
                # params are in a single forward → no unused-param issue
                # in practice; this is belt-and-braces.
                self.fake_score_ddp = DDP(
                    model.fake_score.model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=debug_fup,
                    broadcast_buffers=False,
                )
                model.fake_score.model = self.fake_score_ddp  # type: ignore

    def _build_pipeline(self) -> None:
        cfg = self.config
        denoising_step_list = list(getattr(cfg, "denoising_step_list", [1000, 750, 500, 250]))
        num_frame_per_block = int(getattr(cfg, "num_frame_per_block", 3))
        local_attn_size = int(getattr(cfg, "local_attn_size", 21))
        kv_frames_total = int(getattr(cfg, "kv_frames_total", 21))
        kv_committed_max_frames = int(getattr(cfg, "kv_committed_max_frames", 9))
        prime_kv_frames = int(getattr(cfg, "prime_kv_frames", 9))

        # Live-window topology. Defaults preserve the historical 4x1
        # behavior (4 slots, 1 pass per rolling step). Set
        # `num_live_slots: 2` + `passes_per_step: 2` in the config for
        # the 2x2 ladder ([1.0, 0.5] decay + [500, 1000] ladder).
        num_live_slots = int(getattr(cfg, "num_live_slots", 4))
        passes_per_step = int(getattr(cfg, "passes_per_step", 1))
        default_decay = (
            [1.0, 0.75, 0.5, 0.25] if num_live_slots == 4 else [1.0, 0.5]
        )
        action_decay_slot = tuple(
            getattr(cfg, "action_decay_per_slot", default_decay)
        )

        self.pipeline = RollingStaircaseTrainingPipeline(
            denoising_step_list=denoising_step_list,
            scheduler=self.model.scheduler,
            generator=self.model.generator,
            num_frame_per_block=num_frame_per_block,
            num_slots=num_live_slots,
            passes_per_step=passes_per_step,
            action_decay_per_slot=action_decay_slot,
            prime_kv_frames=prime_kv_frames,
            kv_frames_total=kv_frames_total,
            kv_committed_max_frames=kv_committed_max_frames,
            local_attn_size=local_attn_size,
            action_projection=self.model.action_projection,
            action_token_projection=self.model.action_token_projection,
            real_score_num_gt_chunks=int(getattr(cfg, "real_score_num_gt_chunks", 2)),
        )
        # The DMD model's slot-loss helper needs the pipeline for optional
        # ride-aware conditioning (not currently used). We store a reference.
        setattr(self.model, "_staircase_pipeline", self.pipeline)

    def _build_dataset(self) -> None:
        cfg = self.config
        encoded_root = str(cfg.encoded_root)
        caption_root = str(cfg.caption_root)
        motion_root = str(cfg.motion_root)
        ss_vae_checkpoint = str(cfg.ss_vae_checkpoint)
        min_ride_frames = int(getattr(cfg, "min_ride_frames", 64))
        max_rides = getattr(cfg, "max_rides", None)

        # Optional ride-length ordering. When ``sort_rides_by_length`` is
        # "asc" (shortest→longest) or "desc", the dataset sorts its
        # internal ride list after indexing and the DistributedSampler
        # runs with shuffle=False so the sort order is preserved
        # on-the-wire. With "asc", rank r consumes rides at indices
        # [r, r+W, r+2W, ...] of the sorted list — strictly increasing
        # length, interleaved across ranks. With "none" (default) rides
        # are in glob-sorted path order and the sampler shuffles every
        # epoch, matching v14/ODE behavior.
        sort_mode = getattr(cfg, "sort_rides_by_length", None)
        if sort_mode in ("none", "", None):
            sort_mode = None
        elif sort_mode not in ("asc", "desc"):
            raise ValueError(
                f"sort_rides_by_length must be one of "
                f"{{None, 'none', 'asc', 'desc'}}; got {sort_mode!r}"
            )

        self.dataset = ZarrRideDataset(
            encoded_root=encoded_root,
            caption_root=caption_root,
            motion_root=motion_root,
            ss_vae_checkpoint=ss_vae_checkpoint,
            min_ride_frames=min_ride_frames,
            device="cpu",
            max_rides=int(max_rides) if max_rides is not None else None,
            sort_by_length=sort_mode,
        )

        shuffle_rides = (sort_mode is None)
        self.sampler = DistributedSampler(
            self.dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle_rides,
            drop_last=True,
        ) if self.world_size > 1 else None

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            sampler=self.sampler,
            shuffle=(self.sampler is None and shuffle_rides),
            num_workers=int(getattr(cfg, "num_workers", 2)),
            pin_memory=False,
            collate_fn=_ride_metadata_collate,
        )

        # Optional action-dims projection (for z_action subset).
        action_dims = getattr(cfg, "action_dims", None)
        self.action_dims: Optional[List[int]] = list(action_dims) if action_dims is not None else None
        self.max_ride_frames = getattr(cfg, "max_ride_frames", None)

    def _build_optimizer(self) -> None:
        cfg = self.config
        lr = float(getattr(cfg, "lr", 1e-5))
        betas = tuple(getattr(cfg, "betas", [0.9, 0.999]))
        eps = float(getattr(cfg, "eps", 1e-8))
        wd = float(getattr(cfg, "weight_decay", 0.0))

        # Generator-only: inner DiT + action_projection (if trainable).
        trainable_params: List[torch.nn.Parameter] = []
        trainable_params.extend(
            p for p in self.model.generator.model.parameters() if p.requires_grad
        )
        if self.model.action_projection is not None:
            ap_trainable = bool(getattr(cfg, "train_action_projection", False))
            for p in self.model.action_projection.parameters():
                p.requires_grad_(ap_trainable)
                if ap_trainable:
                    trainable_params.append(p)

        # Stream-B action-token projection. Paired with `action_projection`
        # in both training distribution and DDP handling (both live on
        # `BaseModel`, outside the generator DDP wrapper, and need manual
        # grad all-reduce — see `_all_reduce_extra_trainable_grads`).
        # Defaults to trainable for the same reason `action_projection`
        # does: the ODE student's two action streams are *paired* during
        # distillation, so training one while freezing the other would
        # introduce a calibration mismatch between the streams as the DiT
        # drifts under DMD.
        if self.model.action_token_projection is not None:
            atp_trainable = bool(
                getattr(cfg, "train_action_token_projection", False)
            )
            for p in self.model.action_token_projection.parameters():
                p.requires_grad_(atp_trainable)
                if atp_trainable:
                    trainable_params.append(p)

        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=lr, betas=betas, eps=eps, weight_decay=wd
        )
        self.max_grad_norm = float(getattr(cfg, "max_grad_norm", 1.0))

        # ------------------------------------------------------------------
        # Fake-score optimizer (only when fake-score updates are enabled).
        # Separate AdamW over fake_score.model parameters; stepped at the
        # end of every iter alongside the generator optimizer (1:1 cadence
        # per `dfake_gen_update_ratio=1` default in the config).
        # ------------------------------------------------------------------
        self.fake_optimizer: Optional[torch.optim.Optimizer] = None
        self.fake_score_updates_enabled = bool(
            getattr(cfg, "fake_score_updates_enabled", False)
        )
        self.dfake_gen_update_ratio = int(
            getattr(cfg, "dfake_gen_update_ratio", 1)
        )
        if self.fake_score_updates_enabled:
            fake_lr = float(getattr(cfg, "fake_lr", lr))
            fake_betas = tuple(getattr(cfg, "fake_betas", betas))
            fake_eps = float(getattr(cfg, "fake_eps", eps))
            fake_wd = float(getattr(cfg, "fake_weight_decay", wd))
            fake_params = [
                p for p in self.model.fake_score.model.parameters() if p.requires_grad
            ]
            if not fake_params:
                raise RuntimeError(
                    "fake_score_updates_enabled=True but fake_score has no "
                    "trainable parameters. Did DMD2B2BLAM_Staircase.__init__ "
                    "accidentally freeze them?"
                )
            self.fake_optimizer = torch.optim.AdamW(
                fake_params,
                lr=fake_lr,
                betas=fake_betas,
                eps=fake_eps,
                weight_decay=fake_wd,
            )
            self.fake_max_grad_norm = float(
                getattr(cfg, "fake_max_grad_norm", self.max_grad_norm)
            )

    # ------------------------------------------------------------------
    # Action teacher (CoTracker + ss_vae) for state_probe / action_critic
    # supervision. Lazy-loaded when ``action_teacher_mode != "off"``.
    # ------------------------------------------------------------------
    def _build_action_teacher(self) -> None:
        """Optionally load the frozen motion-pipeline action teacher.

        Controlled by ``action_teacher_mode`` (single source of truth):

          - ``action_teacher_mode="off"``:
              teacher not built; ``_frozen_cotracker`` / ``_frozen_ss_vae``
              remain ``None``. The DMD model auto-disables aux heads
              (see ``_build_action_aux_heads``).
          - ``action_teacher_mode="slot0"``: teacher IS built and runs
              on the full live window (CoTracker needs the ~12-frame
              context regardless); the DMD model pins aux supervision
              to slot 0 only as a hard override.
          - ``action_teacher_mode="all"``: teacher built and runs on the
              full live window; all NS slots produce teacher-z vectors
              and aux supervision follows ``aux_loss_slot_policy``
              (slot0 / match_dmd / random).

        Implementation note: the compute cost of CoTracker + VAE decode
        cannot be restricted to a single slot (CoTracker needs the full
        window for tracking quality, and VAE decodes the whole batch at
        once). So ``slot0`` vs ``all`` differ only in which teacher-z
        vectors are CONSUMED by the aux losses — the teacher's compute
        cost is identical. The ~1 ms ss_vae encode per slot is not
        worth specialising.

        Loads CoTracker-3 (offline) and the ss_vae encoder from disk
        and attaches them to ``self`` as ``_frozen_cotracker``,
        ``_frozen_ss_vae``, and ``_frozen_ss_vae_scale``. Both modules
        are put into eval mode with ``requires_grad=False``.

        The ``WanVAEWrapper`` on ``self.model`` (``self.model.vae``) is
        re-used as the latent→pixel decoder, so we don't load a second
        VAE copy.

        Cost (per rolling step, 12 latent frames → 45 pixel frames):
          - VAE decode: ~30 ms (bf16)
          - CoTracker forward: ~150 ms (fp16 autocast, grid=10)
          - ss_vae encoder: <1 ms
          Total ~200 ms/rolling step. Over a 200-step ride this is
          ~40 s of extra wall-clock. Enable only when the teacher-z
          supervision is worth that cost on the target GPU.
        """
        self._frozen_cotracker = None
        self._frozen_ss_vae = None
        self._frozen_ss_vae_scale = 1.0
        from utils.action_teacher import resolve_action_teacher_mode
        self.action_teacher_mode = resolve_action_teacher_mode(
            getattr(self.config, "action_teacher_mode", None),
            source_label="action_teacher_mode",
        )
        # Internal derived bool (NOT a config knob any more); downstream
        # code paths use this to cheaply gate teacher forwards.
        self.action_teacher_enabled = self.action_teacher_mode != "off"
        if not self.action_teacher_enabled:
            if self.is_main_process:
                logging.info(
                    "Action teacher disabled (action_teacher_mode=off); "
                    "aux losses auto-disabled inside the DMD model."
                )
            return
        if self.is_main_process:
            logging.info(
                "Action teacher mode=%s (teacher built; aux consumes %s).",
                self.action_teacher_mode,
                "slot 0 only" if self.action_teacher_mode == "slot0"
                else "per aux_loss_slot_policy",
            )

        if self.is_main_process:
            logging.info("Building frozen action teacher (CoTracker + ss_vae)...")

        # Local-checkpoint path (v14-style): when
        # ``cotracker_source_dir`` + ``cotracker_checkpoint_path`` are set,
        # import ``CoTrackerPredictor`` from the given repo dir and load
        # the ``.pth`` weights directly — no download, no torch.hub
        # network call. Otherwise fall back to ``torch.hub.load(...)``
        # which uses the torch-hub cache under ``TORCH_HOME`` (default
        # ``~/.cache/torch``).
        cot_src = getattr(self.config, "cotracker_source_dir", None)
        cot_ckpt = getattr(self.config, "cotracker_checkpoint_path", None)
        use_local = bool(
            cot_src and cot_ckpt
            and os.path.isdir(str(cot_src))
            and os.path.isfile(str(cot_ckpt))
        )
        if use_local:
            self._frozen_cotracker = self._load_cotracker_from_local(
                source_dir=str(cot_src), checkpoint_path=str(cot_ckpt),
            )
        else:
            if cot_src or cot_ckpt:
                if self.is_main_process:
                    logging.warning(
                        "cotracker_source_dir / cotracker_checkpoint_path set "
                        "but one or both paths don't exist (src=%s ckpt=%s); "
                        "falling back to torch.hub.load.",
                        cot_src, cot_ckpt,
                    )
            # Rank 0 downloads first (populates hub cache), then other
            # ranks load from cache to avoid download races on multi-node
            # runs. With the torch-hub cache already populated (as it is
            # on this cluster at ~/.cache/torch/hub), neither of these
            # calls hits the network.
            if self.is_main_process:
                self._frozen_cotracker = torch.hub.load(
                    "facebookresearch/co-tracker", "cotracker3_offline",
                ).to(self.device)
            if self.world_size > 1:
                barrier()
            if not self.is_main_process:
                self._frozen_cotracker = torch.hub.load(
                    "facebookresearch/co-tracker", "cotracker3_offline",
                ).to(self.device)
            if self.world_size > 1:
                barrier()
        self._frozen_cotracker.eval()
        for p in self._frozen_cotracker.parameters():
            p.requires_grad_(False)

        from action_query.ss_vae_model import load_ss_vae
        ss_vae_ckpt = str(getattr(
            self.config, "ss_vae_checkpoint",
            "action_query/checkpoints/ss_vae_8free.pt",
        ))
        ss_vae, scale = load_ss_vae(ss_vae_ckpt, device=str(self.device))
        ss_vae.eval()
        ss_vae.requires_grad_(False)
        self._frozen_ss_vae = ss_vae
        self._frozen_ss_vae_scale = float(scale)

        if self.is_main_process:
            logging.info(
                "Frozen action teacher ready (CoTracker + ss_vae, "
                "ss_vae_scale=%.3f)", self._frozen_ss_vae_scale,
            )

    def _load_cotracker_from_local(
        self, *, source_dir: str, checkpoint_path: str,
    ) -> torch.nn.Module:
        """Load ``CoTrackerPredictor`` from a locally-cached repo + weights.

        Equivalent to ``torch.hub.load(repo_or_dir=source_dir,
        source='local', model='cotracker3_offline', pretrained=False)``
        plus a manual ``load_state_dict`` of the ``scaled_offline.pth``
        file, but does it ourselves so we (a) never hit the network and
        (b) have a single call site to blame if either piece moves.

        ``source_dir`` must contain the cloned ``facebookresearch/co-tracker``
        repo (with ``cotracker/predictor.py`` inside) — e.g.
        ``~/.cache/torch/hub/facebookresearch_co-tracker_main`` or a
        staged copy under ``/scratch``.

        ``checkpoint_path`` must point at ``scaled_offline.pth`` (the
        v14-era cotracker3_offline weights).
        """
        if self.is_main_process:
            logging.info(
                "Loading CoTracker from local source_dir=%s checkpoint=%s",
                source_dir, checkpoint_path,
            )
        # Inject the repo into sys.path so `cotracker.predictor` is
        # importable. Don't permanently prepend — append-only (we only
        # need it during this call).
        if source_dir not in sys.path:
            sys.path.append(source_dir)
        try:
            from cotracker.predictor import CoTrackerPredictor  # type: ignore
        except ImportError as e:
            raise ImportError(
                f"Failed to import cotracker.predictor from {source_dir}. "
                f"Make sure the directory contains the cloned co-tracker "
                f"repo (with a `cotracker/` package inside). Error: {e}"
            )
        # Matches hubconf.py's cotracker3_offline defaults: window_len=60,
        # v2=False. The predictor loads checkpoint inside __init__ when a
        # path is given.
        predictor = CoTrackerPredictor(
            checkpoint=checkpoint_path, window_len=60, v2=False,
        )
        predictor = predictor.to(self.device)
        return predictor

    @torch.no_grad()
    def _compute_teacher_z_per_slot(
        self,
        pred_x0_all_slots: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Run the frozen motion teacher on the 4-slot pred_x0 window.

        Args:
          pred_x0_all_slots: ``[B, NS*npb, C, H, W]`` (graph-carrying or
            detached; this method detaches internally).

        Returns:
          ``[B, NS, 8]`` teacher z vectors (one per slot) or ``None``
          when the teacher is disabled / unavailable.

        Implementation: VAE-decodes the full live window ([B, 12, C, H, W]
        → [B, 12, 3, H_pix, W_pix] pixels in [-1, 1]), runs CoTracker-3
        offline with ``grid_size=10`` to get per-frame 2D tracks + visibility,
        computes per-slot mean track displacements + visibility over each
        3-frame block (``output_chunk_size = num_frame_per_block = npb``),
        and encodes the resulting motion vectors through ss_vae's encoder
        + tanh-squash. The result is 1 ss_vae ``z`` vector per slot per
        batch element, matching the shape the aux losses expect.

        Numerical care:
          - Runs under ``torch.no_grad()`` (teacher is frozen).
          - Uses fp32 for the VAE decode to match ``_frozen_vae`` in the
            parent trainer; pred_x0 is cast to float.
          - CoTracker is wrapped in ``autocast(cuda, enabled=True)`` so
            it uses fp16 on A100/H100 regardless of the training dtype.

        Returns ``None`` on any failure (e.g. cotracker's offline model
        refuses very-short sequences on some builds); callers fall back
        to the commanded-action branch automatically because the aux
        losses treat ``None`` that way.
        """
        if self._frozen_cotracker is None or self._frozen_ss_vae is None:
            return None
        if pred_x0_all_slots is None:
            return None
        vae = getattr(self.model, "vae", None)
        if vae is None:
            return None

        npb = int(getattr(self.config, "num_frame_per_block", 3))
        x0 = pred_x0_all_slots.detach()
        if x0.dim() != 5:
            return None
        B, F_total = x0.shape[0], x0.shape[1]
        if F_total % npb != 0:
            return None
        n_slots = F_total // npb

        try:
            # VAE wants [B, T, C, H, W] in fp32. Disable autocast: the
            # outer training forward runs in bf16, which would otherwise
            # cast convs' inputs back to bf16 and clash with the VAE's
            # fp32 bias tensors ("Input type BFloat16 and bias type Float
            # should be the same").
            with torch.amp.autocast(device_type="cuda", enabled=False):
                pixels = vae.decode_to_pixel(x0.float())  # [B, T_pix, 3, H, W]
        except Exception as e:
            logging.warning("action teacher: VAE decode failed: %s", e)
            return None
        # Pixels are in [-1, 1]; CoTracker wants [0, 255] float, [B, T, 3, H, W].
        video = (255.0 * 0.5 * (pixels + 1.0)).clamp(0, 255).float()

        grid_size = 10
        N = grid_size ** 2

        # We want 1 motion vector PER SLOT (== npb pixel frames per slot).
        # Decode gives T_pix = K * F_lat for some VAE upsample ratio K. We
        # pool motion vectors across each slot's pixel frames.
        T_pix = video.shape[1]
        if T_pix < 2:
            return None
        # Frames-per-slot at the pixel resolution. Typical VAE: K=4 → 45
        # pixel frames for 12 latent frames (first 3 from the first latent,
        # then 4 per subsequent latent). We approximate with a uniform
        # split T_pix // n_slots; remainder at the tail is discarded.
        per_slot_pix = T_pix // n_slots
        if per_slot_pix < 2:
            return None
        used_pix = per_slot_pix * n_slots
        vid = video[:, :used_pix]  # [B, used, 3, H, W]

        try:
            with torch.amp.autocast(device_type="cuda", enabled=True):
                pred_tracks, pred_vis = self._frozen_cotracker(
                    vid, grid_size=grid_size,
                )
        except Exception as e:
            logging.warning("action teacher: CoTracker forward failed: %s", e)
            return None
        # pred_tracks: [B, used_pix, N, 2], pred_vis: [B, used_pix, N] or [..., 1]
        # Displacement per adjacent pixel-frame pair; pool across each slot.
        d = pred_tracks[:, 1:] - pred_tracks[:, :-1]            # [B, used-1, N, 2]
        if pred_vis.dim() == 3:
            vis = pred_vis.unsqueeze(-1)                        # [B, used, N, 1]
        else:
            vis = pred_vis                                      # [B, used, N, 1]
        # Align visibility with displacement time-axis (drop the first frame).
        vis = vis[:, 1:]                                        # [B, used-1, N, 1]
        # Slot-wise pooling: split the used-1 time axis into `n_slots`
        # equal-size chunks (last chunk is 1 shorter because used-1 is
        # used-1 per slot-1 with 1 fewer). Handle by reshape with padding
        # of the last sample where necessary.
        eff_len = used_pix - 1
        base = eff_len // n_slots
        if base < 1:
            return None
        # Trim to base * n_slots so we can reshape cleanly.
        d = d[:, : base * n_slots]
        vis = vis[:, : base * n_slots]
        d = d.reshape(B, n_slots, base, N, 2).mean(dim=2)       # [B, NS, N, 2]
        vis = vis.reshape(B, n_slots, base, N, 1).to(d.dtype).mean(dim=2)

        motion = torch.cat([d, vis], dim=-1)                    # [B, NS, N, 3]
        # ss_vae encoder expects 2-channel [N=B_eff, 2, 10, 10] input.
        xy = motion[..., :2]                                    # [B, NS, N, 2]
        xy = xy.reshape(B * n_slots, grid_size, grid_size, 2)
        x_in = xy.permute(0, 3, 1, 2).float() / self._frozen_ss_vae_scale
        try:
            mu, _ = self._frozen_ss_vae.encoder(x_in.to(self.device))
        except Exception as e:
            logging.warning("action teacher: ss_vae encoder failed: %s", e)
            return None
        # mu: [B*NS, 8, 1, 1] → [B, NS, 8]
        z = mu.squeeze(-1).squeeze(-1)
        from utils.zarr_dataset import _tanh_squash
        z = _tanh_squash(z)
        z = z.reshape(B, n_slots, -1)
        return z.detach().contiguous()

    def _init_wandb(self) -> None:
        self.wandb_enabled = (
            self.is_main_process
            and _HAS_WANDB
            and not bool(getattr(self.config, "disable_wandb", False))
        )
        if not self.wandb_enabled:
            return
        run_name = str(getattr(self.config, "run_name", "phase1_rolling_staircase"))
        project = str(getattr(self.config, "wandb_project", "longlive-phase1"))
        wandb.init(  # type: ignore
            project=project,
            name=run_name,
            config=OmegaConf.to_container(self.config, resolve=True),
            dir=str(getattr(self.config, "wandb_dir", "./wandb")),
        )

    # ------------------------------------------------------------------
    # DDP grad-uniformity check (catches rank-asymmetric parameter
    # touches under the no_sync accumulation pattern)
    # ------------------------------------------------------------------
    def _assert_grad_uniformity(self) -> None:
        """All-reduce MIN/MAX of the per-rank count of trainable params
        with non-None .grad. Raises if they differ.

        This catches the failure mode where:
          - some rank's forward touches parameter P (accumulates .grad)
          - another rank's forward does NOT touch P (.grad stays None)
          - the final-backward allreduce fires only on params touched in
            the FINAL forward, so P's local-accumulated grads never get
            averaged across ranks -> silent divergence.

        With `find_unused_parameters=False` DDP normally tolerates this
        (one reduction bucket per backward), so we add an explicit check.
        Cost: one int-allreduce per optimizer step.
        """
        if self.world_size <= 1 or not dist.is_initialized():
            return
        if not bool(getattr(self.config, "assert_grad_uniformity", True)):
            return
        # Check both generator and fake_score optimizers (if either exists).
        for optim, name in (
            (self.optimizer, "generator"),
            (getattr(self, "fake_optimizer", None), "fake_score"),
        ):
            if optim is None:
                continue
            trainable = [
                p for g in optim.param_groups for p in g["params"] if p.requires_grad
            ]
            local_count = sum(1 for p in trainable if p.grad is not None)
            t = torch.tensor([local_count], device=self.device, dtype=torch.long)
            t_min = t.clone()
            t_max = t.clone()
            dist.all_reduce(t_min, op=dist.ReduceOp.MIN)
            dist.all_reduce(t_max, op=dist.ReduceOp.MAX)
            lo = int(t_min.item())
            hi = int(t_max.item())
            if lo != hi:
                raise RuntimeError(
                    f"[rank {self.rank}] DDP grad-uniformity check ({name}) "
                    f"failed at step={self.step}: "
                    f"local_grad_param_count={local_count}, "
                    f"world_min={lo}, world_max={hi}. Some rank backprop'd a "
                    f"parameter others did not, which silently corrupts the "
                    f"no_sync gradient-accumulation pattern. Re-run with "
                    f"debug_find_unused_parameters=true to locate the offender."
                )

    # ------------------------------------------------------------------
    # Manual gradient sync for params OUTSIDE the generator-DDP wrapper
    # ------------------------------------------------------------------
    def _all_reduce_extra_trainable_grads(self) -> None:
        """Average grads across ranks for trainable params that live OUTSIDE
        the DDP-wrapped generator DiT.

        `action_projection` is part of the BaseModel but not part of
        `self.generator.model`, so DDP's autograd hook doesn't sync its
        gradients automatically. With `no_sync()` wrapping the per-slot
        backwards we also can't rely on DDP's final-backward all-reduce
        (which only covers the DDP-registered parameters). Without this
        step, action_projection drifts per-rank and training corrupts
        silently.

        Called AFTER the final backward (so all grads are populated) and
        BEFORE `_assert_grad_uniformity` (so the check sees the synced
        state on every rank).
        """
        if self.world_size <= 1 or not dist.is_initialized():
            return
        extras: List[torch.nn.Parameter] = []
        ap = getattr(self.model, "action_projection", None)
        if ap is not None:
            extras.extend(p for p in ap.parameters() if p.requires_grad)
        atp = getattr(self.model, "action_token_projection", None)
        if atp is not None:
            extras.extend(p for p in atp.parameters() if p.requires_grad)
        if not extras:
            return
        scale = 1.0 / float(self.world_size)
        for p in extras:
            if p.grad is None:
                # If grad is None on some ranks and populated on others,
                # `_assert_grad_uniformity` will catch it and raise with a
                # more informative message. Don't mask the divergence.
                continue
            # Average (not sum): matches DDP's default bucket reduction.
            dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
            p.grad.data.mul_(scale)

    # ------------------------------------------------------------------
    # Multi-slot visualization recorder
    # ------------------------------------------------------------------
    def _init_vis_recorder(self) -> None:
        wallclock = float(getattr(self.config, "vis_wallclock_seconds", 900.0))
        num_chunks = int(getattr(self.config, "vis_num_chunks", 10))
        fps = int(getattr(self.config, "vis_fps", 8))
        npb = int(getattr(self.config, "num_frame_per_block", 3))
        # Wandb is the primary sink (user request); local disk is opt-in.
        # When wandb is disabled globally via `disable_wandb`, fall back
        # to local disk so the clips aren't lost.
        wandb_on = (
            not bool(getattr(self.config, "disable_wandb", False))
            and bool(getattr(self.config, "vis_to_wandb", True))
        )
        save_local = bool(getattr(
            self.config, "vis_save_local", not wandb_on,
        ))
        emit_on_first_step = bool(getattr(
            self.config, "vis_emit_on_first_step", False,
        ))
        self.vis_recorder = MultislotVisRecorder(
            enabled=(self.is_main_process and wallclock > 0),
            out_dir=self.log_dir / "vis",
            wallclock_seconds=wallclock,
            num_chunks=num_chunks,
            fps=fps,
            num_frame_per_block=npb,
            log_to_wandb=wandb_on,
            save_local=save_local,
            emit_on_first_step=emit_on_first_step,
        )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    @property
    def log_dir(self) -> Path:
        base = Path(str(getattr(self.config, "log_dir", "./logs/phase1_rolling_staircase")))
        run_name = str(getattr(self.config, "run_name", "phase1_rolling_staircase"))
        out = base / run_name
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _checkpoint_path(self, step: int) -> Path:
        return self.log_dir / f"phase1_step{step:07d}.pt"

    def _save_checkpoint(self) -> None:
        if not self.is_main_process:
            return
        path = self._checkpoint_path(self.step)
        state = {
            "step": self.step,
            "generator": (self.generator_ddp.module if self.generator_ddp is not None else self.model.generator.model).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config_name": os.path.basename(self.config_path),
        }
        if self.model.action_projection is not None:
            state["action_projection"] = self.model.action_projection.state_dict()
        torch.save(state, path)
        logging.info("Saved checkpoint: %s", path)

    def _maybe_resume(self) -> None:
        if not bool(getattr(self.config, "auto_resume", False)):
            return
        ckpts = sorted(self.log_dir.glob("phase1_step*.pt"))
        if not ckpts:
            return
        path = ckpts[-1]
        if self.is_main_process:
            logging.info("Auto-resuming from %s", path)
        state = torch.load(path, map_location="cpu")
        inner = self.generator_ddp.module if self.generator_ddp is not None else self.model.generator.model
        missing, unexpected = inner.load_state_dict(state["generator"], strict=False)
        if self.is_main_process:
            logging.info("resume: generator missing=%d unexpected=%d", len(missing), len(unexpected))
        if "action_projection" in state and self.model.action_projection is not None:
            self.model.action_projection.load_state_dict(state["action_projection"], strict=False)
        try:
            self.optimizer.load_state_dict(state["optimizer"])
        except Exception as e:
            logging.warning("optimizer state load failed: %s", e)
        self.step = int(state.get("step", 0))

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(self) -> None:
        cfg = self.config
        max_steps = int(getattr(cfg, "max_steps", 10000))
        ckpt_interval = int(getattr(cfg, "checkpoint_interval", 500))
        log_interval = int(getattr(cfg, "log_interval", 10))
        vis_interval = int(getattr(cfg, "vis_interval", 0))
        max_rolling_steps = getattr(cfg, "max_rolling_steps", None)
        max_rolling_steps = int(max_rolling_steps) if max_rolling_steps is not None else None
        aux_loss_weight = float(getattr(cfg, "aux_loss_weight", 1.0))

        # Multi-slot knobs. When both are at their defaults we fall back to
        # the original one-ride-per-iter code path so existing configs keep
        # working bit-identically.
        slots_per_rank = int(getattr(cfg, "slots_per_rank", 1))
        rolling_steps_per_iter_cfg = getattr(cfg, "rolling_steps_per_iter", None)
        rolling_steps_per_iter = (
            int(rolling_steps_per_iter_cfg) if rolling_steps_per_iter_cfg is not None else None
        )
        use_multislot = (slots_per_rank > 1) or (rolling_steps_per_iter is not None)

        if self.is_main_process:
            logging.info(
                "Starting training: max_steps=%d ckpt_interval=%d log_interval=%d",
                max_steps, ckpt_interval, log_interval,
            )
            if use_multislot:
                logging.info(
                    "Multi-slot mode: slots_per_rank=%d rolling_steps_per_iter=%s",
                    slots_per_rank, rolling_steps_per_iter,
                )
            else:
                logging.info("Single-ride-per-iter mode (legacy path).")

        # First vis flush fires one interval AFTER train start, not immediately.
        self.vis_recorder.reset_timer()

        if use_multislot:
            self._train_multislot(
                max_steps=max_steps,
                ckpt_interval=ckpt_interval,
                log_interval=log_interval,
                vis_interval=vis_interval,
                aux_loss_weight=aux_loss_weight,
                slots_per_rank=slots_per_rank,
                rolling_steps_per_iter=int(rolling_steps_per_iter or 16),
            )
        else:
            self._train_single_ride(
                max_steps=max_steps,
                ckpt_interval=ckpt_interval,
                log_interval=log_interval,
                vis_interval=vis_interval,
                max_rolling_steps=max_rolling_steps,
                aux_loss_weight=aux_loss_weight,
            )

        self._save_checkpoint()
        if self.wandb_enabled:
            wandb.finish()  # type: ignore

    # ------------------------------------------------------------------
    # Single-ride-per-iter path (legacy; slots_per_rank=1, no M cap)
    # ------------------------------------------------------------------
    def _train_single_ride(
        self,
        *,
        max_steps: int,
        ckpt_interval: int,
        log_interval: int,
        vis_interval: int,
        max_rolling_steps: Optional[int],
        aux_loss_weight: float,
    ) -> None:
        epoch = 0
        while self.step < max_steps:
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            for batch in self.dataloader:
                if self.step >= max_steps:
                    break
                t0 = time.time()

                meta_list = batch  # list of dicts (batch_size=1).
                if not meta_list:
                    continue
                meta = meta_list[0]

                ride = _load_ride_tensors(
                    self.dataset,
                    meta,
                    device=self.device,
                    dtype=self.dtype,
                    action_dims=self.action_dims,
                    max_frames=self.max_ride_frames,
                )
                if ride is None:
                    continue

                ride_stats = self._train_one_ride(
                    ride=ride,
                    max_rolling_steps=max_rolling_steps,
                    aux_loss_weight=aux_loss_weight,
                )

                self.step += 1
                ride_time = time.time() - t0

                if self.is_main_process and (self.step % log_interval == 0):
                    logging.info(
                        "step=%d ride=%s n_rec=%d total_loss=%.6f ride_time=%.2fs",
                        self.step,
                        os.path.basename(ride["zarr_path"]),
                        ride_stats["num_records"],
                        ride_stats["total_loss"],
                        ride_time,
                    )
                if self.wandb_enabled and (self.step % log_interval == 0):
                    payload = dict(ride_stats)
                    payload["time/ride_seconds"] = ride_time
                    payload["ride/n_latent_frames"] = ride["n_latent_frames"]
                    wandb.log(payload, step=self.step)  # type: ignore

                if self.step > 0 and (self.step % ckpt_interval == 0):
                    self._save_checkpoint()
                    if dist.is_initialized():
                        barrier()

                if vis_interval > 0 and self.step > 0 and (self.step % vis_interval == 0):
                    self._maybe_visualize(ride)

            epoch += 1

    # ------------------------------------------------------------------
    # Multi-slot path (per-rank B slots, M rolling steps per iter,
    # cross-rank lockstep guaranteed by prepare_for_iter)
    # ------------------------------------------------------------------
    def _make_ride_loader(self):
        """Return a zero-arg callable that yields fresh ride tensors on
        demand, cycling the DistributedSampler across epochs indefinitely.

        The batcher calls this to refill a slot whose ride has exhausted.
        Rank-locality is enforced by the existing DistributedSampler —
        each rank sees only its own shard of rides.
        """

        def _gen():
            epoch = 0
            while True:
                if self.sampler is not None:
                    self.sampler.set_epoch(epoch)
                produced_any = False
                for batch in self.dataloader:
                    meta_list = batch
                    if not meta_list:
                        continue
                    meta = meta_list[0]
                    ride = _load_ride_tensors(
                        self.dataset,
                        meta,
                        device=self.device,
                        dtype=self.dtype,
                        action_dims=self.action_dims,
                        max_frames=self.max_ride_frames,
                    )
                    if ride is None:
                        continue
                    produced_any = True
                    yield ride
                if not produced_any:
                    # No rides in an entire epoch — treat as exhaustion
                    # so the outer loop can exit instead of spinning.
                    return
                epoch += 1

        gen_iter = _gen()

        def _loader():
            try:
                return next(gen_iter)
            except StopIteration:
                return None

        return _loader

    def _train_multislot(
        self,
        *,
        max_steps: int,
        ckpt_interval: int,
        log_interval: int,
        vis_interval: int,
        aux_loss_weight: float,
        slots_per_rank: int,
        rolling_steps_per_iter: int,
    ) -> None:
        ride_loader = self._make_ride_loader()
        batcher = PerSlotRideBatcher(
            pipeline=self.pipeline,
            slots_per_rank=slots_per_rank,
            device=self.device,
            dtype=self.dtype,
            ride_loader=ride_loader,
        )

        while self.step < max_steps:
            t0 = time.time()

            # Lockstep refill: every slot gets >= M steps before we roll,
            # so every rank will do exactly B*M rolling-step forwards.
            ready = batcher.prepare_for_iter(
                required_rolling_steps=rolling_steps_per_iter
            )
            if ready < slots_per_rank:
                if self.is_main_process:
                    logging.info(
                        "Dataset exhausted (ready=%d/%d slots). Stopping.",
                        ready, slots_per_rank,
                    )
                break

            iter_stats = self._train_one_iter_multislot(
                batcher=batcher,
                rolling_steps_per_iter=rolling_steps_per_iter,
                aux_loss_weight=aux_loss_weight,
            )

            self.step += 1
            iter_time = time.time() - t0

            if self.is_main_process and (self.step % log_interval == 0):
                # GPU memory footprint snapshot. ``max_memory_allocated``
                # tracks peak allocator usage since last reset (we don't
                # reset — so this grows monotonically through a session
                # and tells us the high-water mark we need to budget for).
                # ``memory_reserved`` is what CUDA holds, which is what
                # nvidia-smi reports.
                if torch.cuda.is_available():
                    mem_alloc_gib = torch.cuda.memory_allocated() / (1024 ** 3)
                    mem_reserved_gib = torch.cuda.memory_reserved() / (1024 ** 3)
                    mem_peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    mem_str = (
                        f"mem(alloc={mem_alloc_gib:.2f}GiB "
                        f"reserved={mem_reserved_gib:.2f}GiB "
                        f"peak={mem_peak_gib:.2f}GiB)"
                    )
                else:
                    mem_str = "mem(cuda-off)"
                logging.info(
                    "step=%d n_rec=%d total_loss=%.6f iter_time=%.2fs %s | %s",
                    self.step,
                    iter_stats["num_records"],
                    iter_stats["total_loss"],
                    iter_time,
                    mem_str,
                    batcher.summary(),
                )
            if self.wandb_enabled and (self.step % log_interval == 0):
                payload = dict(iter_stats)
                payload["time/iter_seconds"] = iter_time
                payload["batcher/slots_per_rank"] = float(slots_per_rank)
                payload["batcher/rolling_steps_per_iter"] = float(rolling_steps_per_iter)
                wandb.log(payload, step=self.step)  # type: ignore

            if self.step > 0 and (self.step % ckpt_interval == 0):
                self._save_checkpoint()
                if dist.is_initialized():
                    barrier()

            # Wall-clock-triggered slot-0 visualization. Records were
            # buffered inside `_train_one_iter_multislot` on rank 0 for
            # ride_slot_idx=0; this call decodes + writes an mp4 if the
            # configured interval has elapsed. No-op on non-main ranks.
            flushed = self.vis_recorder.maybe_flush(
                vae=getattr(self.model, "vae", None),
                device=self.device,
                dtype=self.dtype,
                step=self.step,
            )
            if flushed and self.is_main_process:
                logging.info(
                    "vis: flushed %d chunks at step=%d",
                    self.vis_recorder.buffer_len(), self.step,
                )

            # Legacy step-count vis (single-ride path); ignored here.
            _ = vis_interval

    # ------------------------------------------------------------------
    # One multi-slot training iter
    # ------------------------------------------------------------------
    def _train_one_iter_multislot(
        self,
        *,
        batcher: PerSlotRideBatcher,
        rolling_steps_per_iter: int,
        aux_loss_weight: float,
    ) -> Dict[str, float]:
        """Run `rolling_steps_per_iter` rolling-step forwards on every slot.

        Backward pattern: one-step look-ahead `no_sync()`. All backwards
        except the last are wrapped in `generator_ddp.no_sync()`, so DDP
        accumulates gradients locally across (slots_per_rank *
        rolling_steps_per_iter) backwards and all-reduces exactly once
        per iter on the final backward.

        Lockstep guarantee: `prepare_for_iter` has already ensured every
        slot on every rank has >= rolling_steps_per_iter steps remaining,
        so every rank runs the same number of backwards — DDP pairs up.
        """
        self.optimizer.zero_grad(set_to_none=True)
        if self.fake_optimizer is not None:
            self.fake_optimizer.zero_grad(set_to_none=True)

        total_loss_accum = 0.0
        fake_loss_accum = 0.0
        num_records = 0
        num_fake_backwards = 0
        slot_loss_sum: Dict[int, float] = {}
        slot_loss_count: Dict[int, int] = {}
        slot_ride_contributions: Dict[int, int] = {}

        def _gen_no_sync():
            if self.generator_ddp is not None:
                return self.generator_ddp.no_sync()
            return contextlib.nullcontext()

        def _fake_no_sync():
            if self.fake_score_ddp is not None:
                return self.fake_score_ddp.no_sync()
            return contextlib.nullcontext()

        def _process_pending(pending_records, pending_ride_slot_idx, is_final):
            """Backward one slot's records under the right sync contexts
            for BOTH the generator and fake_score (each has its own DDP
            wrapper + no_sync lookahead)."""
            teacher_z = None
            if self.action_teacher_enabled and pending_records:
                first = pending_records[0]
                p = getattr(first, "pred_x0_all_slots", None)
                if p is not None:
                    teacher_z = self._compute_teacher_z_per_slot(p)
            step_loss, step_log, fake_loss = self.model.generator_loss_on_slots(
                slot_outputs=list(pending_records),
                aux_loss_weight=aux_loss_weight,
                teacher_z_per_slot=teacher_z,
            )
            gen_ctx = contextlib.nullcontext() if is_final else _gen_no_sync()
            fake_ctx = contextlib.nullcontext() if is_final else _fake_no_sync()
            if step_loss.requires_grad:
                with gen_ctx:
                    # retain_graph when we still need the graph for fake
                    # backward (fake_loss shares the autocast / step but
                    # its backward is through a DIFFERENT graph — no
                    # shared intermediates — so retain_graph is NOT
                    # needed; the graphs are independent). Keep plain.
                    step_loss.backward()
            fake_val = 0.0
            if fake_loss is not None and fake_loss.requires_grad:
                with fake_ctx:
                    fake_loss.backward()
                fake_val = float(fake_loss.detach().item())
            return (
                float(step_loss.detach().item()),
                len(pending_records),
                step_log,
                fake_val,
            )

        def _record_bookkeeping(pending_records, sv, n_rec, slog, fake_val, pending_ride_slot_idx):
            nonlocal total_loss_accum, num_records, fake_loss_accum, num_fake_backwards
            total_loss_accum += sv
            num_records += n_rec
            if fake_val != 0.0:
                fake_loss_accum += fake_val
                num_fake_backwards += 1
            slot_ride_contributions[pending_ride_slot_idx] = (
                slot_ride_contributions.get(pending_ride_slot_idx, 0) + 1
            )
            for rec in pending_records:
                slot_idx = int(rec.slot_idx)
                mean_key = f"loss/slot{slot_idx}_mean"
                mean_val_tensor = slog.get(mean_key)
                if mean_val_tensor is not None and torch.is_tensor(mean_val_tensor):
                    rec_val = float(mean_val_tensor.item())
                else:
                    rec_val = sv / max(n_rec, 1)
                slot_loss_sum[slot_idx] = slot_loss_sum.get(slot_idx, 0.0) + rec_val
                slot_loss_count[slot_idx] = slot_loss_count.get(slot_idx, 0) + 1

        with autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=self.use_mixed_precision):
            # We generate (records, ride_slot_idx) tuples and use a
            # one-step look-ahead to backward everything except the last
            # under `no_sync`. The last backward runs under nullcontext
            # and fires the single all-reduce for the iter.
            pending_records = None
            pending_ride_slot_idx = -1

            for step_idx in range(rolling_steps_per_iter):
                for ride_slot_idx in range(batcher.slots_per_rank):
                    records = batcher.step_slot(ride_slot_idx)
                    if records is None or len(records) == 0:
                        continue
                    # Visualization: capture rank-0 ride_slot_0's slot-0
                    # pred_x0 into the rolling buffer. `observe` no-ops
                    # on non-main ranks or when vis is disabled.
                    if ride_slot_idx == 0:
                        self.vis_recorder.observe(
                            records, batcher.slots[ride_slot_idx]
                        )
                    if pending_records is not None:
                        sv, n, slog, fv = _process_pending(
                            pending_records, pending_ride_slot_idx, is_final=False
                        )
                        _record_bookkeeping(
                            pending_records, sv, n, slog, fv, pending_ride_slot_idx
                        )
                        del slog
                    pending_records = records
                    pending_ride_slot_idx = ride_slot_idx

            # Final backward (allreduce here).
            if pending_records is not None:
                sv, n, slog, fv = _process_pending(
                    pending_records, pending_ride_slot_idx, is_final=True
                )
                _record_bookkeeping(
                    pending_records, sv, n, slog, fv, pending_ride_slot_idx
                )

        # Sync grads for trainable params outside the DDP-wrapped DiT
        # (action_projection). Must run BEFORE the uniformity check.
        self._all_reduce_extra_trainable_grads()
        # Grad-uniformity check BEFORE clip + step (raises on mismatch).
        self._assert_grad_uniformity()

        # Clip + optimizer step (generator).
        total_grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.optimizer.param_groups[0]["params"] if p.grad is not None],
            max_norm=self.max_grad_norm,
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        # Clip + optimizer step (fake_score).
        fake_grad_norm_val = 0.0
        if self.fake_optimizer is not None:
            fake_params_with_grad = [
                p
                for p in self.fake_optimizer.param_groups[0]["params"]
                if p.grad is not None
            ]
            if fake_params_with_grad:
                fake_grad_norm = torch.nn.utils.clip_grad_norm_(
                    fake_params_with_grad, max_norm=self.fake_max_grad_norm
                )
                fake_grad_norm_val = (
                    float(fake_grad_norm.item())
                    if torch.is_tensor(fake_grad_norm)
                    else float(fake_grad_norm)
                )
                self.fake_optimizer.step()
            self.fake_optimizer.zero_grad(set_to_none=True)

        stats: Dict[str, float] = {
            "total_loss": total_loss_accum / max(num_records, 1),
            "total_loss_sum": total_loss_accum,
            "num_records": float(num_records),
            "grad_norm": float(total_grad_norm.item())
                if torch.is_tensor(total_grad_norm)
                else float(total_grad_norm),
        }
        if self.fake_optimizer is not None:
            stats["fake/loss"] = fake_loss_accum / max(num_fake_backwards, 1)
            stats["fake/loss_sum"] = fake_loss_accum
            stats["fake/num_backwards"] = float(num_fake_backwards)
            stats["fake/grad_norm"] = fake_grad_norm_val
        for slot_idx, loss_sum in slot_loss_sum.items():
            cnt = max(slot_loss_count.get(slot_idx, 1), 1)
            stats[f"loss/slot{slot_idx}_mean"] = loss_sum / cnt
            stats[f"loss/slot{slot_idx}_count"] = float(cnt)
        for ride_slot_idx, contrib in slot_ride_contributions.items():
            stats[f"batcher/ride_slot{ride_slot_idx}_steps"] = float(contrib)
        return stats

    # ------------------------------------------------------------------
    # One ride (legacy single-ride path)
    # ------------------------------------------------------------------
    def _train_one_ride(
        self,
        ride: Dict[str, torch.Tensor],
        *,
        max_rolling_steps: Optional[int],
        aux_loss_weight: float,
    ) -> Dict[str, float]:
        """Run warmup + steady-state rolling on a single ride. Backprops per
        record (or per step) to bound memory. Returns aggregated ride stats."""
        self.optimizer.zero_grad(set_to_none=True)
        if self.fake_optimizer is not None:
            self.fake_optimizer.zero_grad(set_to_none=True)

        total_loss_accum = 0.0
        fake_loss_accum = 0.0
        num_records = 0
        num_fake_backwards = 0
        slot_loss_sum: Dict[int, float] = {}
        slot_loss_count: Dict[int, int] = {}

        # Drive the pipeline's ride-rollout generator. Each yielded record is
        # a grad-enabled slot output. We compute its loss, backward it, and
        # discard the graph (records do NOT span steps because the pipeline
        # detaches the next-step live-window state).
        with autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=self.use_mixed_precision):
            rollout_iter = self.pipeline.rollout_ride(
                gt_latents=ride["latents"],
                gt_actions=ride["z_actions"],
                prompt_embeds=ride["prompt_embeds"],
                max_rolling_steps=max_rolling_steps,
            )

            # Pull step-level lists of records. Each list is one rolling
            # step's grad slots (slot 0 + aux, sharing a single grad graph).
            # Summed-loss backward per list so we don't backprop through the
            # same graph twice.
            #
            # DDP OPTIMIZATION — one-step look-ahead with `no_sync()`:
            #   optimizer.step() fires once per ride (after the whole rollout),
            #   but there are O(ride_length / num_frame_per_block) backwards
            #   in between (~100s per ride). By default each backward triggers
            #   a full all-reduce across ranks, which is ~99% wasted work:
            #   with `no_sync()`, intermediate backwards accumulate gradients
            #   LOCALLY into `.grad` and only the FINAL backward triggers a
            #   single all-reduce that syncs the full accumulated gradient.
            #   This is mathematically identical to the per-step-all-reduce
            #   version (DDP averages across ranks; accumulation is a sum
            #   across steps; `avg(sum) == sum(avg)`), up to float-order.
            #   If we're not using DDP the context is a no-op.
            #
            # The one-step look-ahead is needed because `rollout_iter` is a
            # generator and the last step isn't known until StopIteration.
            def _gen_no_sync():
                if self.generator_ddp is not None:
                    return self.generator_ddp.no_sync()
                return contextlib.nullcontext()

            def _fake_no_sync():
                if self.fake_score_ddp is not None:
                    return self.fake_score_ddp.no_sync()
                return contextlib.nullcontext()

            def _process_step(step_records, is_last):
                """Backward one step (gen + fake); returns
                (step_loss_val, n_records, step_log, fake_loss_val)
                for bookkeeping."""
                teacher_z = None
                if self.action_teacher_enabled and step_records:
                    first = step_records[0]
                    p = getattr(first, "pred_x0_all_slots", None)
                    if p is not None:
                        teacher_z = self._compute_teacher_z_per_slot(p)
                step_loss, step_log, fake_loss = self.model.generator_loss_on_slots(
                    slot_outputs=list(step_records),
                    aux_loss_weight=aux_loss_weight,
                    teacher_z_per_slot=teacher_z,
                )
                gen_ctx = contextlib.nullcontext() if is_last else _gen_no_sync()
                fake_ctx = contextlib.nullcontext() if is_last else _fake_no_sync()
                if step_loss.requires_grad:
                    with gen_ctx:
                        step_loss.backward()
                fake_val = 0.0
                if fake_loss is not None and fake_loss.requires_grad:
                    with fake_ctx:
                        fake_loss.backward()
                    fake_val = float(fake_loss.detach().item())
                sv = float(step_loss.detach().item())
                n = len(step_records)
                return sv, n, step_log, fake_val

            def _record_step_bookkeeping(step_records, step_loss_val, n_step_records, step_log, fake_val):
                nonlocal total_loss_accum, num_records, fake_loss_accum, num_fake_backwards
                total_loss_accum += step_loss_val
                num_records += n_step_records
                if fake_val != 0.0:
                    fake_loss_accum += fake_val
                    num_fake_backwards += 1
                for rec in step_records:
                    slot_idx = int(rec.slot_idx)
                    mean_key = f"loss/slot{slot_idx}_mean"
                    mean_val_tensor = step_log.get(mean_key)
                    if mean_val_tensor is not None and torch.is_tensor(mean_val_tensor):
                        rec_val = float(mean_val_tensor.item())
                    else:
                        rec_val = step_loss_val / max(n_step_records, 1)
                    slot_loss_sum[slot_idx] = slot_loss_sum.get(slot_idx, 0.0) + rec_val
                    slot_loss_count[slot_idx] = slot_loss_count.get(slot_idx, 0) + 1

            # Prefetch one record-list so we can tell "more coming" vs "last".
            iterator = iter(rollout_iter)
            pending = None
            while True:
                try:
                    candidate = next(iterator)
                except StopIteration:
                    candidate = None
                if candidate is not None and len(candidate) == 0:
                    continue
                if pending is None:
                    pending = candidate
                    if pending is None:
                        break  # ride produced no grad-enabled records
                    continue
                is_last = candidate is None
                sv, n_rec, slog, fv = _process_step(pending, is_last=is_last)
                _record_step_bookkeeping(pending, sv, n_rec, slog, fv)
                del slog
                pending = candidate
                if is_last:
                    break

        # Sync grads for trainable params outside the DDP-wrapped DiT
        # (action_projection). Must run BEFORE the uniformity check so
        # the check sees a consistent world state.
        self._all_reduce_extra_trainable_grads()
        # Grad-uniformity check BEFORE clip + step (raises on mismatch).
        self._assert_grad_uniformity()

        # Gradient clip + optimizer step at the END of the ride (generator).
        total_grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.optimizer.param_groups[0]["params"] if p.grad is not None],
            max_norm=self.max_grad_norm,
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        # Gradient clip + optimizer step (fake_score).
        fake_grad_norm_val = 0.0
        if self.fake_optimizer is not None:
            fake_params_with_grad = [
                p
                for p in self.fake_optimizer.param_groups[0]["params"]
                if p.grad is not None
            ]
            if fake_params_with_grad:
                fake_grad_norm = torch.nn.utils.clip_grad_norm_(
                    fake_params_with_grad, max_norm=self.fake_max_grad_norm
                )
                fake_grad_norm_val = (
                    float(fake_grad_norm.item())
                    if torch.is_tensor(fake_grad_norm)
                    else float(fake_grad_norm)
                )
                self.fake_optimizer.step()
            self.fake_optimizer.zero_grad(set_to_none=True)

        # Assemble stats.
        stats: Dict[str, float] = {
            "total_loss": total_loss_accum / max(num_records, 1),
            "total_loss_sum": total_loss_accum,
            "num_records": float(num_records),
            "grad_norm": float(total_grad_norm.item()) if torch.is_tensor(total_grad_norm) else float(total_grad_norm),
        }
        if self.fake_optimizer is not None:
            stats["fake/loss"] = fake_loss_accum / max(num_fake_backwards, 1)
            stats["fake/loss_sum"] = fake_loss_accum
            stats["fake/num_backwards"] = float(num_fake_backwards)
            stats["fake/grad_norm"] = fake_grad_norm_val
        for slot_idx, loss_sum in slot_loss_sum.items():
            cnt = max(slot_loss_count.get(slot_idx, 1), 1)
            stats[f"loss/slot{slot_idx}_mean"] = loss_sum / cnt
            stats[f"loss/slot{slot_idx}_count"] = float(cnt)
        return stats

    # ------------------------------------------------------------------
    # Visualization (simple: dump slot 0 pred_x0 pixels for the first few
    # rolling steps to an MP4). Off by default via vis_interval=0.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _maybe_visualize(self, ride: Dict[str, torch.Tensor]) -> None:
        if not self.is_main_process:
            return
        try:
            frames: List[torch.Tensor] = []
            rollout_iter = self.pipeline.rollout_ride(
                gt_latents=ride["latents"],
                gt_actions=ride["z_actions"],
                prompt_embeds=ride["prompt_embeds"],
                max_rolling_steps=int(getattr(self.config, "vis_max_steps", 32)),
            )
            for step_records in rollout_iter:
                for out in step_records:
                    if out.slot_idx != 0:
                        continue
                    frames.append(out.pred_x0.detach().float().cpu())
            if not frames:
                return
            latents = torch.cat(frames, dim=1)  # [1, steps*npb, C, H, W]
            # Decode to pixel using the generator's VAE wrapper (if available).
            vae = getattr(self.model, "vae", None)
            if vae is None:
                return
            pixels = vae.decode_to_pixel(latents.to(self.device, dtype=self.dtype))
            # Save latents for debugging (writing a full MP4 encoder here is
            # overkill for a placeholder; downstream tooling decodes).
            out_dir = self.log_dir / "vis"
            out_dir.mkdir(exist_ok=True)
            path = out_dir / f"step_{self.step:07d}_pixels.pt"
            torch.save(pixels.detach().cpu(), path)
            logging.info("Saved visualization: %s", path)
        except Exception as e:
            logging.warning("visualization failed: %s", e)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Phase-1 rolling-staircase DMD trainer")
    parser.add_argument("--config", type=str, required=True, help="Path to OmegaConf yaml")
    parser.add_argument("--override", type=str, nargs="*", default=[],
                        help="Optional OmegaConf dotlist overrides, e.g. lr=1e-5 max_steps=1000")
    args = parser.parse_args()

    # Merge overrides in-memory on each rank independently. Earlier we
    # wrote the merged config to a shared tmp YAML and reloaded by
    # path, which raced across ranks — a later rank could load another
    # rank's half-written file and quietly drop keys like
    # ``denoising_loss_type``, crashing deep in model build. Keeping the
    # merge in-memory is race-free: every rank produces the same
    # DictConfig object from the same inputs and never touches disk.
    if args.override:
        base = OmegaConf.load(args.config)
        override = OmegaConf.from_dotlist(list(args.override))
        cfg = OmegaConf.merge(base, override)
    else:
        cfg = OmegaConf.load(args.config)

    trainer = RollingStaircaseDMDTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
