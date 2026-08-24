"""Phase-1 Action-Forcing DMD trainer (NO STAIRCASE).

Subclasses ``RollingStaircaseDMDTrainer`` to reuse the heavy
infrastructure (DDP setup, ZarrRideDataset, action teacher plumbing,
W&B init, EMA, checkpoint save/resume, vis recorder), but overrides
the model/pipeline construction and the training loop to a CF-style
recipe:

  Per training iter (CF parity):
    - Pull a ``num_training_frames``-long ride (default 21 frames)
      from the dataset.
    - Build per-frame action conditioning (Stream A modulation + Stream
      B tokens) from the ride's GT actions. NO decay, NO attenuation —
      every frame gets its actual GT action, exactly mirroring the
      ODE student's training distribution.
    - Run ``model.generator_loss(...)`` (CF-parity DMD on the student's
      pred, no GT context, real/fake symmetric on noisy student).
    - ``generator_loss.backward()`` -> generator optimizer step + EMA.
    - Pull another ride.
    - Run ``model.critic_loss(...)`` (CF-parity fake_score denoise loss
      on the student's pred). ``critic_loss.backward()`` -> fake_score
      optimizer step.

The dataset is constrained to rides with at least
``num_training_frames`` latent frames; we slice the leading
``num_training_frames`` frames + corresponding actions for every ride.
"""
from __future__ import annotations

import ast
import atexit
import gc
import logging
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import wandb  # type: ignore
    _HAS_WANDB = True
except Exception:
    wandb = None  # type: ignore
    _HAS_WANDB = False

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from trainer.causal_rolling_staircase_train import (
    RollingStaircaseDMDTrainer,
    _finalize_ride_to_gpu,
    _load_ride_tensors,
    _load_ride_tensors_cpu_part,
)
from model.dmd_action_forcing import ActionForcingDMD
from model.r3gan import (
    rpgan_d_loss,
    rpgan_g_loss,
    rpgan_d_loss_allpairs,
    rpgan_g_loss_allpairs,
    r1_penalty,
)
from pipeline.action_forcing_training import ActionForcingTrainingPipeline
from utils.infinity_rope import install as _install_infinity_rope


def _frames_to_mp4_bytes(frames: np.ndarray, fps: float = 5.0) -> Optional[bytes]:
    """Encode uint8ndarray ``[T, H, W, 3]`` to mp4 bytes via ffmpeg.

    Mirrors the helper in ``trainer/causal_diffusion_teacher_train.py``.
    Returns ``None`` on any subprocess / encode failure so the caller
    can skip the wandb upload silently. Best-effort: never raises.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        return None
    h, w = int(frames.shape[1]), int(frames.shape[2])
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-f", "mp4",
        "-movflags", "frag_keyframe+empty_moov",
        "pipe:1",
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out_bytes, _ = proc.communicate(
            input=frames.tobytes(), timeout=120,
        )
        if proc.returncode == 0 and len(out_bytes) > 0:
            return out_bytes
    except Exception:
        pass
    return None


def _chunk_actions(per_frame: torch.Tensor, chunk_frames: int) -> torch.Tensor:
    """Mean-pool ``[B, F, D]`` into ``[B, n_chunks, D]`` (drop tail).

    Mirrors ``trainer.causal_diffusion_teacher_train._chunk_actions``;
    duplicated rather than imported to keep the trainer's dependency
    surface tight (``causal_diffusion_teacher_train.py`` pulls in
    motion-pipeline imports we don't want at module load time when
    aux losses are disabled).
    """
    B, F_len, D = per_frame.shape
    n_chunks = F_len // chunk_frames
    trimmed = per_frame[:, : n_chunks * chunk_frames]
    return trimmed.reshape(B, n_chunks, chunk_frames, D).mean(dim=2)


def grad_at(
    loss: torch.Tensor,
    params,
    *,
    retain_graph: bool = True,
) -> Optional[torch.Tensor]:
    """Gradient of ``loss`` at ``params``, FLATTENED — or ``None``.

    WP-PIXGAN T3-C.  Shared by the LADD path (WP-14B) and the pixel path,
    so it lives at MODULE level in this file: the LADD branch can call it
    with no import at all, and nothing pixel-specific is dragged in.

    Args:
        loss: any graph-bearing scalar (or non-scalar — ``grad_outputs``
            is not supplied, so a non-scalar raises exactly as
            ``torch.autograd.grad`` would).
        params: a SINGLE ``torch.Tensor`` or an iterable of tensors.  The
            single-tensor form is the one the A7 probe uses (``_p_last``,
            the last ``requires_grad`` generator parameter) and it is what
            keeps the probe affordable — one parameter, not the whole net.
        retain_graph: forwarded to ``torch.autograd.grad``.  Default True
            because every caller so far measures SEVERAL losses that share
            one graph.

    Returns:
        A detached 1-D ``float32`` tensor: the per-param gradients
        flattened and concatenated in the order given.  ``None`` when the
        gradient does not exist.

    WHY A VECTOR AND NOT A RATIO (this is load-bearing, do not "simplify"):

      * The readout that separates "small but ALIGNED" from "small and
        actively FIGHTING DMD" is the cosine
        ``dot(g_a, g_b) / (|g_a| |g_b|)``, which needs the vectors.  A
        helper returning ratios cannot produce it.
      * Vectors let ONE backward serve every ratio AND every cosine.  Two
        ratio-returning calls would each re-backprop the shared DMD term;
        on the LADD path that second pass runs through a checkpointed
        teacher recompute costing ~11 s and ~37 GiB per fire.  Callers
        therefore take one vector per loss and combine them locally.

    WHY ``None`` AND NEVER ``0.0`` (docs/WP_PIXGAN.md §16/§19/§21):
        ``0.0`` is a FORGEABLE ZERO — indistinguishable from the genuine
        reading "this term contributes nothing", which is a live disputed
        conclusion in this campaign.  ``None`` is not a number and cannot
        be mistaken for one; the call site logs its own distinct error
        key.  ``allow_unused=True`` is kept so an unused param yields
        ``None`` rather than an exception, and if ANY requested param is
        unused the whole call returns ``None`` — a partially-filled vector
        would not be layout-comparable with the other loss's vector, and
        zero-filling the hole would re-import the forged zero.
    """
    if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
        return None
    if isinstance(params, torch.Tensor):
        plist = [params]
    else:
        plist = [p for p in params]
    if not plist:
        return None
    grads = torch.autograd.grad(
        loss, plist, retain_graph=retain_graph, allow_unused=True,
    )
    if any(g is None for g in grads):
        return None
    return torch.cat([g.detach().flatten().float() for g in grads])


def ladd_unweighted_ratio(
    *,
    gg: Optional[torch.Tensor],
    gr: Optional[torch.Tensor],
    mode_count: Optional[int],
    total_weight: Optional[float],
    stat_value: Optional[float],
    pix_folded: bool,
) -> Dict[str, float]:
    """WP-14B: the LADD term's gradient ratio/norm AT WEIGHT 1.0.

    Pure function — no trainer, no ``self`` — so the guard logic is
    unit-testable without a CUDA-touching import. All five inputs are
    values the caller has ALREADY computed; this function only decides
    whether dividing ``gg`` by ``total_weight`` is a valid way to recover
    the unweighted norm, and does the division.

    DIVISION, NOT RAW-THREADING: ``_ladd_run_pair_mode`` (the function
    that actually produces the LADD term) is ~2900 lines with delicate
    R1-fire-rate cadence semantics documented in its own header comment.
    Threading a second raw tensor out of both its return sites, through
    ``_compute_ladd_losses``'s per-mode summation, was judged higher-risk
    than exploiting linearity: when ``gg`` truly equals
    ``total_weight * (the LADD term's own gradient)`` — which is exactly
    what the five guards below establish — then
    ``‖raw grad‖ = ‖gg‖ / total_weight`` is EXACT, not approximate, and
    needs no extra backward (``gg`` is already computed by the caller for
    the existing A7 telemetry).

    Tradeoff, stated plainly: unlike a raw-threaded term (e.g. the pixel
    path's ``pix_raw``), this CANNOT answer a true weight-free probe —
    ``total_weight == 0`` leaves ``gg`` with no graph at all, hence no
    caller-supplied value for it. It only works retroactively on a run
    with weight > 0, which is what the campaign settled on after the
    weight-probe identity was verified on wandb history ("a division over
    the first ~50 steps of any arm now suffices") — this closes exactly
    that gap and no more.

    The cosine between ``gg`` and ``gr`` is NOT recomputed here: cosine is
    scale-invariant for a positive multiplier, so the existing
    ``train/gan_dmd_grad_cos`` (computed by the caller from the same two
    vectors) already IS the unweighted cosine.

    Args:
        gg: gradient of the (possibly weighted, possibly multi-term) GAN
            loss at the probe parameter — the caller's existing ``_gg``.
        gr: gradient of the non-GAN generator loss at the same
            parameter — the caller's existing ``_gr`` (the A7 denominator).
        mode_count: how many LADD pair-modes were enabled this call.
            Anything other than 1 means ``gg`` is a SUM of differently-
            weighted per-mode terms, so no single scalar recovers an
            unweighted value for any one of them.
        total_weight: the LIVE resolved multiplier — ``per-mode weight
            (e.g. ladd_gt_transition_weight) x gen_gan_weight`` — stashed
            by the caller at the moment ``_ladd_run_pair_mode`` computed
            it. Never re-derived from cfg (the class of bug this exists
            to avoid: two echoes reporting config while runtime differed).
        stat_value: the LIVE ``gen_gan_stat_value`` from the same call.
            Must be exactly 0.0 — nonzero means the stat-head sideband is
            additively folded into the LADD term, breaking the pure-
            scalar-multiple precondition the division depends on.
        pix_folded: whether the pixel G-term has been added into the same
            ``gen_gan_loss`` this call's ``gg`` was differentiated from.
            If so ``gg`` is LADD+pixel mixed and dividing it by the
            LADD-only ``total_weight`` would not isolate the LADD term.

    Returns:
        A dict of ``train/ladd_gan_grad_*`` keys. On any guard failure,
        ``train/ladd_gan_grad_unweighted_unavailable=1.0`` plus exactly
        one ``..._reason_*`` key — never a forgeable ``0.0`` standing in
        for "the term contributes nothing" (docs/WP_PIXGAN.md §16/§19/§21;
        the same discipline ``grad_at`` above documents at length).
    """
    out: Dict[str, float] = {}
    if pix_folded:
        out["train/ladd_gan_grad_unweighted_unavailable"] = 1.0
        out["train/ladd_gan_grad_unweighted_reason_pix_folded"] = 1.0
        return out
    if mode_count != 1:
        out["train/ladd_gan_grad_unweighted_unavailable"] = 1.0
        out["train/ladd_gan_grad_unweighted_reason_multimode"] = float(
            mode_count or 0)
        return out
    if stat_value is None or stat_value != 0.0:
        out["train/ladd_gan_grad_unweighted_unavailable"] = 1.0
        out["train/ladd_gan_grad_unweighted_reason_stat_active"] = 1.0
        return out
    if total_weight is None or total_weight <= 0.0:
        out["train/ladd_gan_grad_unweighted_unavailable"] = 1.0
        out["train/ladd_gan_grad_unweighted_reason_weight_unavailable"] = 1.0
        return out
    if gg is None or gr is None:
        out["train/ladd_gan_grad_unweighted_unavailable"] = 1.0
        out["train/ladd_gan_grad_unweighted_reason_no_grad"] = 1.0
        return out
    ng_raw = float(gg.norm()) / total_weight
    out["train/ladd_gan_grad_norm_unweighted"] = ng_raw
    nr = float(gr.norm())
    if nr > 0.0:
        out["train/ladd_gan_grad_ratio_unweighted"] = ng_raw / nr
    else:
        out["train/ladd_gan_grad_unweighted_unavailable"] = 1.0
        out["train/ladd_gan_grad_unweighted_reason_dmd_denom_zero"] = 1.0
    return out


class ActionForcingDMDTrainer(RollingStaircaseDMDTrainer):
    """CF-style DMD2 trainer for the Action-Forcing (no-staircase) Phase-1 recipe.

    Auxiliary losses (action critic + frozen-teacher z guidance) are
    OFF by default. Enable with ``action_critic_aux_enabled: true``
    AND ``action_teacher_mode: all`` (or ``slot0``) in the YAML.
    When enabled the trainer:

      - un-freezes ``self.model.action_critic`` (the model
        instantiates it frozen for ODE checkpoint compatibility),
        wraps it in DDP, and builds a separate AdamW for it;
      - inherits the parent staircase trainer's
        ``_build_action_teacher`` so ``_frozen_cotracker`` /
        ``_frozen_ss_vae`` are loaded; and
      - on every generator step computes teacher z via
        ``_compute_teacher_z_per_slot`` (parent), runs
        ``critic_updates_per_step`` independent critic optim steps,
        then adds the gen-side z-guidance loss to the DMD loss
        before the generator backward.

    State-probe aux losses are NOT wired here on purpose — Phase-1
    Action-Forcing's contract is "DMD + action critic only". The
    state probe head is still instantiated (for ODE checkpoint
    parity) and remains frozen.

    R3GAN (RpGAN + R1) is OPT-IN via ``gan_enabled: true``. When
    enabled the trainer:

      - builds ``model.r3gan.R3GANDiscriminator3D`` on the latent
        video shape and DDP-wraps it (separate from the ``fake_score``
        / ``action_critic`` DDPs);
      - builds a dedicated AdamW (``gan_lr``, ``gan_betas``,
        ``gan_weight_decay``);
      - on every generator step computes:
            real = ride["latents"][:, gen_window]   (GT latent video)
            fake = aux["pred_image"]                (student rollout
                                                     last num_training_
                                                     frames)
        and runs:
            * D-update (RpGAN-D + R1 on real; backward into D only —
              fake is detached so no flow into G);
            * G's RpGAN term added to the generator loss (graph flows
              through ``pred_image`` -> generator);
      - persists / resumes ``r3gan_discriminator`` +
        ``r3gan_optimizer`` state in the checkpoint file.

    See ``model/r3gan.py`` for the loss / discriminator code and
    ``configs/action_forcing_phase1_aux_gan.yaml`` for the
    intended hyper-parameters.

    SC-DMD (Salt paper, arXiv 2604.03118v1) is OPT-IN via
    ``sc_dmd_enabled: true``. When enabled the trainer adds the
    semigroup defect regularizer
        L_SC = E[||Ψ_θ^{ts→te}(x_ts) - Ψ_θ^{tm→te}(Ψ_θ^{ts→tm}(x_ts))||²]
    to the generator loss. This penalizes the gap between the model's
    direct one-step denoising (``t_s → t_e``) and the same denoising
    composed through an intermediate rung ``t_m``, addressing DMD's
    compositionality deficit in multi-step inference.

    Implementation lives entirely on the model
    (``ActionForcingDMD.sc_dmd_loss`` — single chunk, fresh KV cache,
    2 extra DiT forwards on a 3-frame chunk per gen step ≈ 1-2%
    extra wallclock). The trainer threads weight + warmup config and
    adds the weighted SC loss to the unified generator backward
    alongside DMD / aux / GAN. SC works with or without aux/GAN
    enabled (independent loss).
    """

    # ------------------------------------------------------------------
    # Model & pipeline construction (replace staircase model + pipeline).
    # ------------------------------------------------------------------
    def _build_model(self) -> None:
        args = self.config
        if not hasattr(args, "text_pre_encoded"):
            OmegaConf.update(args, "text_pre_encoded", True, merge=True)
        if not hasattr(args, "mixed_precision"):
            OmegaConf.update(args, "mixed_precision", self.use_mixed_precision, merge=True)

        model = ActionForcingDMD(args=args, device=self.device)
        # Move sub-modules to the target device + dtype (mirroring the
        # staircase trainer's setup).
        model.generator.model.to(device=self.device, dtype=self.dtype)
        model.fake_score.model.to(device=self.device, dtype=self.dtype)
        model.real_score.model.to(device=self.device, dtype=self.dtype)
        # Gradient checkpointing — recompute transformer-block
        # activations during backward to free forward-time memory.
        # WAN ships with native checkpoint support
        # (``wan/modules/model.py:767-772``); each block routes through
        # ``torch.utils.checkpoint.checkpoint`` when
        # ``self.gradient_checkpointing`` is True. Saves ~80% of the
        # forward-activation memory on each grad-active gen/real/fake
        # forward at the cost of ~33% extra compute (one extra forward
        # per block during backward).
        # Defaults: gen ON when distilled-critic mode is on (the
        # dual_grad path stacks per-block last-rung grad-active
        # forwards on top of the rolling rollout, OOMing without
        # checkpointing); real_score / fake_score OFF by default
        # (their forwards are detached at the gen-grad path).
        # Configurable per knob below.
        gen_grad_ckpt = bool(
            getattr(args, "gen_gradient_checkpointing", False)
        )
        rs_grad_ckpt = bool(
            getattr(args, "real_score_gradient_checkpointing", False)
        )
        fs_grad_ckpt = bool(
            getattr(args, "fake_score_gradient_checkpointing", False)
        )
        if gen_grad_ckpt:
            model.generator.model.gradient_checkpointing = True
        if rs_grad_ckpt:
            model.real_score.model.gradient_checkpointing = True
        if fs_grad_ckpt:
            model.fake_score.model.gradient_checkpointing = True
        if self.is_main_process:
            logging.info(
                "[ActionForcing] gradient_checkpointing: gen=%s "
                "real_score=%s fake_score=%s",
                gen_grad_ckpt, rs_grad_ckpt, fs_grad_ckpt,
            )
        if model.action_projection is not None:
            model.action_projection.to(device=self.device, dtype=self.dtype)
        if model.action_token_projection is not None:
            model.action_token_projection.to(device=self.device, dtype=self.dtype)
        if getattr(model, "state_probe", None) is not None:
            model.state_probe.to(device=self.device, dtype=self.dtype)
        if getattr(model, "action_critic", None) is not None:
            model.action_critic.to(device=self.device, dtype=self.dtype)
        if getattr(model, "vae", None) is not None:
            try:
                model.vae.to(device=self.device)
            except AttributeError:
                inner_vae = getattr(model.vae, "model", None)
                if inner_vae is not None:
                    inner_vae.to(device=self.device)
        self.model = model

        # ------------------------------------------------------------------
        # torch.compile (optional, config-gated). Wraps the inner DiT
        # modules BEFORE DDP wrap so the compiled graph is what DDP all-
        # reduces over. Real_score is frozen but still benefits from
        # compile on its inference forward (3 forwards per gen step).
        #
        # Knobs:
        #   ``compile_dit`` (bool, default False): turn it on.
        #   ``compile_dit_mode`` (str, default "default"): forwarded to
        #       ``torch.compile(mode=...)``. ``"default"`` is safe with
        #       in-place KV cache mutation; ``"reduce-overhead"`` enables
        #       CUDA graphs which conflict with our cache writes — avoid.
        #   ``compile_dit_dynamic`` (bool|None, default None = auto):
        #       ``True`` = single shape-polymorphic graph (faster compile,
        #       slightly slower runtime); ``False`` = recompile per shape
        #       (slower compile, fastest runtime). ``None`` lets dynamo
        #       decide. Our forward shapes vary (rolling 3-frame chunks
        #       vs scoring 21-frame chunks vs local_attn_size_schedule
        #       transitions), so the dynamo cache holds multiple graphs.
        #   ``compile_dynamo_cache_size`` (int, default 64): bump
        #       dynamo's cache_size_limit so shape-polymorphic recompiles
        #       don't fall back to eager.
        #
        # First-step wallclock spikes during compilation (~minutes for
        # a 1.3B DiT). Steady-state speedup is typically 1.5-3x on the
        # forward path. If compile fails (dynamic-shape edge case, DDP
        # interaction), we log + fall back to eager — training continues.
        # ------------------------------------------------------------------
        compile_dit = bool(getattr(self.config, "compile_dit", False))
        if compile_dit:
            compile_mode = str(getattr(self.config, "compile_dit_mode", "default"))
            compile_dynamic = getattr(self.config, "compile_dit_dynamic", None)
            cache_size_limit = int(getattr(self.config, "compile_dynamo_cache_size", 64))
            try:
                import torch._dynamo as _dynamo
                _dynamo.config.cache_size_limit = max(
                    int(_dynamo.config.cache_size_limit), cache_size_limit,
                )
            except Exception as _exc:
                logging.warning(
                    "[ActionForcing] torch._dynamo cache_size_limit bump failed: %s",
                    _exc,
                )
            if self.is_main_process:
                logging.info(
                    "[ActionForcing] torch.compile DiT modules (mode=%s, "
                    "dynamic=%s, dynamo_cache=%d). First-step wallclock will "
                    "spike during compilation; steady-state should be ~1.5-3x "
                    "faster on forward.",
                    compile_mode, compile_dynamic, cache_size_limit,
                )
            try:
                model.generator.model = torch.compile(
                    model.generator.model,
                    mode=compile_mode, dynamic=compile_dynamic,
                )
                model.fake_score.model = torch.compile(
                    model.fake_score.model,
                    mode=compile_mode, dynamic=compile_dynamic,
                )
                model.real_score.model = torch.compile(
                    model.real_score.model,
                    mode=compile_mode, dynamic=compile_dynamic,
                )
            except Exception as _exc:
                logging.warning(
                    "[ActionForcing] torch.compile failed (%s) — falling "
                    "back to eager.", _exc,
                )

        debug_fup = bool(getattr(self.config, "debug_find_unused_parameters", False))
        self.generator_ddp: Optional[DDP] = None
        self.fake_score_ddp: Optional[DDP] = None
        _phase_lora_no_ddp = bool(
            getattr(self.config, "student_phase_lora_enabled", False)
        )
        if self.world_size > 1:
            if _phase_lora_no_ddp:
                # Phase LoRA: the student is deliberately NOT DDP-
                # wrapped. Every DDP reducer configuration was tried
                # and failed — the per-iter-varying autograd graph
                # (random exit rung selects a different lora branch
                # each iter) and the ckpt-recompute forwards (which
                # call DDP _pre_forward MID-backward) together break
                # all of the reducer's modes:
                #   * find_unused=True: deferred bucket rebuild fires
                #     mid-recompute at step 21 (= gan_disc_start_step
                #     + 1, when rung_flash's first real grad completes
                #     the rebuild precondition) -> INTERNAL ASSERT
                #     (j5147634/5/6, j5151298+).
                #   * find_unused=False + ghost: ghost is a shallow
                #     loss node, all lora grads arrive at backward
                #     START, the rebuild precondition completes mid-
                #     backward at iter 1 -> "Expected to have finished
                #     reduction" (j5173579).
                #   * + static_graph: per-iter graph-structure
                #     variation (different lora branch per exit rung)
                #     violates the static-graph contract -> "training
                #     graph has changed in this iteration" at iter 2
                #     (j5176005/11).
                # With freeze_base=true the ONLY trainable student
                # params are the lora matrices (~tens of MB), so
                # manual gradient sync is cheap and removes the entire
                # reducer state machine from the problem. Mirrors the
                # existing ``action_projection`` manual-sync pattern
                # (see ``_all_reduce_extra_trainable_grads``, which
                # also syncs the lora grads in this mode).
                #
                # DDP's construction-time param broadcast is replaced
                # by an explicit one-time broadcast: peft initialises
                # lora_A from per-rank RNG, so ranks MUST be aligned
                # before the first forward.
                if dist.is_initialized():
                    _n_bcast = 0
                    for _n, _p in model.generator.model.named_parameters():
                        if "lora_" in _n:
                            dist.broadcast(_p.data, src=0)
                            _n_bcast += 1
                    if self.is_main_process:
                        logging.info(
                            "[ActionForcing] phase LoRA no-DDP mode: "
                            "broadcast %d lora params from rank 0; "
                            "student grads sync manually in "
                            "_all_reduce_extra_trainable_grads.",
                            _n_bcast,
                        )
            else:
                gen_fup = debug_fup
                self.generator_ddp = DDP(
                    model.generator.model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=gen_fup,
                    broadcast_buffers=False,
                )
                model.generator.model = self.generator_ddp  # type: ignore

            if bool(getattr(self.config, "fake_score_updates_enabled", True)):
                # v21: fake_alt_head_enabled adds head_alt params that
                # see no gradient on iters where compute_alt_head=False
                # (e.g. the DMD gen-step's fake_score call). DDP with
                # find_unused_parameters=False would hang on this. Flip
                # the flag to True when alt head is active so the
                # reducer skips alt-head params that weren't touched.
                fake_score_fup = debug_fup or bool(
                    getattr(self.config, "fake_alt_head_enabled", False)
                )
                self.fake_score_ddp = DDP(
                    model.fake_score.model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=fake_score_fup,
                    broadcast_buffers=False,
                )
                model.fake_score.model = self.fake_score_ddp  # type: ignore

            # Online real_teacher: peft adapter is alive on
            # model.real_score.model. Wrap with DDP using
            # find_unused_parameters=True (peft only touches LoRA-target
            # modules — non-LoRA layers see no gradient per step).
            self.real_score_ddp: Optional[DDP] = None
            if bool(getattr(self.config, "real_teacher_train_online", False)):
                self.real_score_ddp = DDP(
                    model.real_score.model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=True,
                    broadcast_buffers=False,
                )
                model.real_score.model = self.real_score_ddp  # type: ignore

            # v27B: ForwardNoiser DDP wrap. The noiser is a separate
            # small ConvNet trained on rollout1→rollout2 chunk pairs
            # with CARN-step conditioning.
            # find_unused_parameters=True (defensive): the critic-step
            # loss path has multiple early-return guards (no rollout2
            # stash, no aligned pairs, chunk alignment edge cases). In
            # principle these are rank-symmetric, but with =False any
            # rank-divergent return path would AllReduce-hang. =True
            # makes DDP traverse the autograd graph to find used params
            # per iter — small perf cost on a 30M-param noiser, zero
            # hang risk.
            self.forward_noiser_ddp: Optional[DDP] = None
            if (
                bool(getattr(self.config, "forward_noiser_enabled", False))
                and getattr(model, "forward_noiser", None) is not None
            ):
                # Match the rest of the model's dtype (bf16) so 3D conv
                # bias matches input dtype. Without this, forward_noiser
                # stays in fp32 (default nn.Module dtype) while inputs
                # arrive as bf16 → RuntimeError on Conv3d bias.
                # Derive the dtype from fake_score's parameters so we
                # automatically follow whatever precision the run uses.
                noiser_dtype = torch.float32
                fs_model = getattr(model.fake_score, "model", None)
                if fs_model is not None:
                    fs_param = next(fs_model.parameters(), None)
                    if fs_param is not None:
                        noiser_dtype = fs_param.dtype
                model.forward_noiser = model.forward_noiser.to(
                    device=self.device, dtype=noiser_dtype,
                )
                self.forward_noiser_ddp = DDP(
                    model.forward_noiser,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=True,
                    broadcast_buffers=False,
                )
                model.forward_noiser = self.forward_noiser_ddp  # type: ignore

            # CARN-cycle: DDP-wrap the REVERSE noiser G exactly like F.
            # find_unused_parameters=True for the same reason — the cycle
            # training has bail paths (no rollout2, misaligned pairs) that the
            # dual zero-anchor covers, but =True is the zero-hang-risk default.
            self.reverse_noiser_ddp: Optional[DDP] = None
            if (
                bool(getattr(self.config, "forward_noiser_cycle_enabled", False))
                and getattr(model, "reverse_noiser", None) is not None
            ):
                # MUST stay in lockstep with F's dtype derivation above: G's
                # output feeds F's grad-on ``fn_out`` graph in L_cyc, so a
                # dtype/device divergence would throw inside G(...) on one rank
                # and hang its peers in the collective. Keep these two blocks
                # identical if either is edited.
                noiser_dtype = torch.float32
                fs_model = getattr(model.fake_score, "model", None)
                if fs_model is not None:
                    fs_param = next(fs_model.parameters(), None)
                    if fs_param is not None:
                        noiser_dtype = fs_param.dtype
                model.reverse_noiser = model.reverse_noiser.to(
                    device=self.device, dtype=noiser_dtype,
                )
                self.reverse_noiser_ddp = DDP(
                    model.reverse_noiser,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=True,
                    broadcast_buffers=False,
                )
                model.reverse_noiser = self.reverse_noiser_ddp  # type: ignore
        else:
            self.real_score_ddp = None
            self.forward_noiser_ddp = None
            self.reverse_noiser_ddp = None

        # ------------------------------------------------------------------
        # Auxiliary action critic (CF-parity ActionCritic, frozen teacher
        # supervised). The model's ``_build_action_aux_heads_compat``
        # instantiated the critic in ``eval()`` + ``requires_grad_(False)``
        # mode purely so the ODE checkpoint's critic state_dict can be
        # loaded without losing keys. When aux losses are TURNED ON via
        # config we un-freeze it here, set train(), and DDP-wrap it. The
        # critic's optimizer is built in ``_build_optimizer``.
        #
        # The "aux losses active" flag requires BOTH ``action_critic_
        # aux_enabled=true`` AND ``action_teacher_mode != "off"``. The
        # base ``configs/action_forcing_phase1.yaml`` has the head
        # enabled (for ODE checkpoint compat) but the teacher off, so
        # the aux loss path stays cold there — no DDP-wrap, no critic
        # optimizer, no behavioural change vs the pre-aux trainer.
        # The aux variant ``configs/action_forcing_phase1_aux.yaml``
        # enables both and gets the full critic + teacher path.
        # ------------------------------------------------------------------
        ac_aux_enabled_cfg = bool(
            getattr(self.config, "action_critic_aux_enabled", False)
        )
        teacher_mode_cfg = str(
            getattr(self.config, "action_teacher_mode", "off") or "off"
        ).strip().lower()
        self.action_critic_aux_enabled = ac_aux_enabled_cfg
        self.action_critic_loss_active = (
            ac_aux_enabled_cfg and teacher_mode_cfg != "off"
        )
        self.action_critic_ddp: Optional[DDP] = None
        if self.action_critic_loss_active:
            ac = getattr(model, "action_critic", None)
            if ac is None:
                raise RuntimeError(
                    "action_critic_aux_enabled=true but "
                    "model.action_critic is None — set "
                    "action_critic_aux_enabled=true in the YAML so "
                    "ActionForcingDMD instantiates the head, or set "
                    "action_critic_aux_enabled=false to disable the "
                    "aux loss path."
                )
            ac.requires_grad_(True)
            ac.train()
            if self.world_size > 1:
                self.action_critic_ddp = DDP(
                    ac,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=False,
                    broadcast_buffers=False,
                )
                # Keep ``model.action_critic`` pointing at the DDP
                # wrapper so all call sites (loss path, optim, EMA,
                # checkpoint save) see the same handle. The wrapped
                # ``.module`` still exposes the underlying ActionCritic
                # for state_dict save.
                model.action_critic = self.action_critic_ddp  # type: ignore

        # ------------------------------------------------------------------
        # State-probe (LoRA-side z-supervision). Mirror of the action_
        # critic DDP-wrap+optim plumbing, fired only when the LoRA-side
        # state_probe loss is requested (= state_probe_aux_enabled AND
        # action_teacher_mode != off AND state_probe_z_loss_weight > 0).
        # The probe was instantiated frozen by ``_build_action_aux_heads_
        # compat`` for ckpt-load compat; flip to trainable here when the
        # loss is wired in. Heads share weights between gen-side
        # (attached to generator wrapper, currently no loss) and
        # LoRA-side (attached to real_score wrapper by
        # ``_attach_state_probe_to_real_score``). Single optimizer.
        # ------------------------------------------------------------------
        self.state_probe_ddp: Optional[DDP] = None
        self.state_probe_optimizer: Optional[torch.optim.Optimizer] = None
        self.state_probe_z_loss_weight = float(
            getattr(self.config, "state_probe_z_loss_weight", 0.0)
        )
        self.state_probe_aux_active = (
            self.action_critic_loss_active
            and bool(getattr(self.config, "state_probe_aux_enabled", False))
            and self.state_probe_z_loss_weight > 0.0
            and getattr(model, "state_probe", None) is not None
        )
        self.state_probe_max_grad_norm = float(
            getattr(self.config, "state_probe_max_grad_norm", 1.0)
        )
        if self.state_probe_aux_active:
            sp = model.state_probe
            sp.requires_grad_(True)
            sp.train()
            if self.world_size > 1:
                self.state_probe_ddp = DDP(
                    sp,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=False,
                    broadcast_buffers=False,
                )
                model.state_probe = self.state_probe_ddp  # type: ignore
                # Keep the shared wrapper-side references in sync so the
                # real_score wrapper's forward calls the DDP-wrapped
                # module (matched all_reduce across ranks). Also point
                # generator._state_probe at the DDP wrap so any gen-side
                # forward that does fire the probe (rare under streaming
                # geometry) routes through DDP rather than the raw
                # module, avoiding a desync.
                if hasattr(model, "real_score"):
                    setattr(model.real_score, "_state_probe", model.state_probe)
                if hasattr(model, "generator"):
                    setattr(model.generator, "_state_probe", model.state_probe)
                # The wrapper's probe-firing gate reads
                # ``self._state_probe.num_frame_per_block`` directly
                # (utils/wan_wrapper.py:624) without unwrapping DDP.
                # DDP doesn't proxy arbitrary attrs, so the lookup
                # returns 0 and the gate fails — state_preds never
                # gets returned, the LoRA-side state_probe loss never
                # fires, no params get gradient. Mirror the attribute
                # onto the DDP wrap explicitly. Same goes for the
                # other init-time attrs the wrapper reads from the
                # probe in older code paths (defensive).
                self.state_probe_ddp.num_frame_per_block = (
                    sp.num_frame_per_block
                )
            sp_lr = float(getattr(self.config, "state_probe_lr", 1.0e-04))
            sp_betas = tuple(
                getattr(self.config, "state_probe_betas", [0.9, 0.999])
            )
            sp_eps = float(getattr(self.config, "state_probe_eps", 1.0e-08))
            sp_wd = float(
                getattr(self.config, "state_probe_weight_decay", 0.0)
            )
            sp_params = [p for p in sp.parameters() if p.requires_grad]
            self.state_probe_optimizer = torch.optim.AdamW(
                sp_params,
                lr=sp_lr, betas=sp_betas, eps=sp_eps, weight_decay=sp_wd,
            )
            if self.is_main_process:
                logging.info(
                    "[ActionForcing] state_probe optimizer built: "
                    "AdamW lr=%.2e wd=%.4f weight=%.3f params=%.2fM",
                    sp_lr, sp_wd, self.state_probe_z_loss_weight,
                    sum(p.numel() for p in sp_params) / 1e6,
                )

        # LoRA-side action_critic z-guidance weight (mirror of the
        # gen-side ``generator_action_z_guidance_weight`` knob, applied
        # to the LoRA's denoised x0 instead of the student's pred). Set
        # to 0.0 to disable. Reads from ``self.config`` so it can be
        # overridden per-sbatch.
        self.lora_action_critic_z_guidance_weight = float(
            getattr(
                self.config,
                "lora_action_critic_z_guidance_weight",
                0.0,
            )
        )

        # ------------------------------------------------------------------
        # R3GAN — RpGAN + R1 discriminator (opt-in via ``gan_enabled``).
        # Built in fp32 to keep the gradient-penalty (second-order)
        # arithmetic numerically clean. The discriminator operates on
        # the same latent video tensor the student predicts; both real
        # (GT) and fake (``pred_image``) come from the trainer's
        # ride / aux dict, so the disc never has to know about the
        # RoPE-cache / scheduler state of the generator.
        # ------------------------------------------------------------------
        self.gan_enabled = bool(getattr(self.config, "gan_enabled", False))
        self.r3gan_disc: Optional[torch.nn.Module] = None
        self.r3gan_disc_ddp: Optional[DDP] = None
        # ``gan_backbone`` selects the discriminator architecture. Only
        # "ladd_teacher_feat" is supported: a LADD adjacent-chunk
        # discriminator on WAN teacher intermediate features.
        self.gan_backbone = str(
            getattr(self.config, "gan_backbone", "ladd_teacher_feat")
        )
        if self.gan_backbone != "ladd_teacher_feat":
            raise ValueError(
                "gan_backbone must be 'ladd_teacher_feat'; got "
                f"{self.gan_backbone!r}."
            )
        if self.gan_enabled:
            if self.gan_backbone == "ladd_teacher_feat":
                # ===== LADD adjacent-chunk discriminator (v28) =====
                # Frozen WAN-teacher feature taps + trainable CCM/CSM
                # + 1D SpectralConv heads. See model/ladd_disc.py for
                # the architecture and the design rationale (LADD,
                # Projected GAN, ASD-style adjacent-chunk loss).
                from model.ladd_disc import build_ladd_disc
                # Tap indices: per-teacher-size defaults if config left
                # empty. 1.3B has 30 transformer blocks; 14B has 40.
                ladd_blocks = list(
                    getattr(self.config, "ladd_feature_blocks", []) or []
                )
                # Did the config name the taps, or are we about to
                # auto-derive them? The 14B backbone branch below refuses
                # to run on auto-derived taps (the 40-block default is
                # [8, 16, 24, 32, 39] -> a full 28 GB load).
                _ladd_blocks_explicit = bool(ladd_blocks)
                # Pull the underlying WanModel's transformer blocks from
                # real_score. Walk all submodules (handles arbitrary
                # wrap depth: DDP / WanDiffusionWrapper / v14 LoRA /
                # alt-head plumbing). Pick the first nn.ModuleList of
                # WAN transformer blocks we find.
                _real_score = self.model.real_score
                _blocks_iter = None
                _wm_for_blocks = _real_score
                for _candidate in [_real_score] + list(
                    _real_score.modules()
                ):
                    for _attr in ("transformer_blocks", "blocks"):
                        _b = getattr(_candidate, _attr, None)
                        if (
                            isinstance(_b, torch.nn.ModuleList)
                            and len(_b) >= 10  # WAN has 30+ blocks
                        ):
                            _blocks_iter = _b
                            _wm_for_blocks = _candidate
                            break
                    if _blocks_iter is not None:
                        break
                if _blocks_iter is None:
                    raise AttributeError(
                        "LADD: could not find a transformer-block "
                        "ModuleList anywhere under real_score. real_score "
                        f"type={type(_real_score).__name__}. Inspect the "
                        "wrap chain and either patch the search loop or "
                        "specify ``ladd_feature_blocks`` explicitly."
                    )
                _n_blocks = len(_blocks_iter)
                if not ladd_blocks:
                    # Span the depth: ~5 taps evenly distributed.
                    if _n_blocks <= 8:
                        ladd_blocks = list(range(_n_blocks))
                    else:
                        ladd_blocks = [
                            max(0, int(_n_blocks * f / 5))
                            for f in (1, 2, 3, 4, 4.95)
                        ]
                        # Dedup + sort; trim to <= 5.
                        ladd_blocks = sorted(set(min(b, _n_blocks - 1) for b in ladd_blocks))[:5]
                # Infer teacher hidden dim from a tap block. WAN
                # transformer block has .hidden_size or we read it
                # from any Linear layer in the block.
                _tap_block = _blocks_iter[ladd_blocks[0]]
                _dim_teacher = None
                for _name in ("hidden_size", "dim", "inner_dim"):
                    if hasattr(_tap_block, _name):
                        _dim_teacher = int(getattr(_tap_block, _name))
                        break
                if _dim_teacher is None:
                    # Fallback: scan block's parameters for a square
                    # weight (linear self-projection often has it).
                    for _p in _tap_block.parameters():
                        if _p.dim() == 2 and _p.shape[0] == _p.shape[1]:
                            _dim_teacher = int(_p.shape[0])
                            break
                if _dim_teacher is None:
                    raise RuntimeError(
                        "LADD: could not infer dim_teacher from a tap "
                        f"block at index {ladd_blocks[0]}. Specify "
                        "ladd_feature_blocks explicitly."
                    )
                # Prompt-conditioning dim from real_score's text embed.
                _prompt_embed_dim = 0
                if bool(getattr(self.config, "ladd_use_prompt_cond", True)):
                    _prompt_embed_dim = int(
                        getattr(self.config, "text_embed_dim", 4096)
                    )
                _wavelet_hf_enabled = bool(
                    getattr(self.config, "ladd_wavelet_hf_enabled", False)
                )
                _wavelet_in_channels = int(
                    getattr(self.config, "gan_disc_in_channels", 16)
                )
                # WAN tokenisation params for the 2D head reshape.
                # patch_size: read off the WAN model attribute if
                # present (default (1, 2, 2) for Wan-1.3B/14B);
                # action_tokens_per_frame: needed to strip per-frame
                # action tokens before the spatial reshape.
                _ps_attr = getattr(_wm_for_blocks, "patch_size", None)
                if _ps_attr is None:
                    _ps_attr = (1, 2, 2)
                _patch_size = tuple(int(p) for p in _ps_attr)
                _a_per_f = 0
                for _cand in [_real_score] + list(_real_score.modules()):
                    if hasattr(_cand, "action_tokens_per_frame"):
                        _v = int(getattr(_cand, "action_tokens_per_frame", 0))
                        if _v > 0:
                            _a_per_f = _v
                            break
                # ===== WP-14B: independent frozen 14B projection backbone =====
                # ``ladd_disc_backbone_model_name`` (default "" = OFF,
                # byte-identical legacy behaviour) swaps the network the
                # disc PROJECTS ONTO from ``real_score`` — the same v14e
                # teacher the student is distilled from, hence largely
                # redundant with the DMD term (GAN_REDESIGN Point 6) — to a
                # stock Wan2.1-T2V-14B PREFIX: blocks 0..max(taps) only,
                # frozen, bf16 (~6.8 GB for taps [0, 2, 4, 8]).
                # ``real_score`` / ``real_name`` are NOT touched; DMD keeps
                # its teacher untouched.
                _ladd_backbone = None
                _ladd_bk_name = str(
                    getattr(self.config, "ladd_disc_backbone_model_name", "")
                    or ""
                )
                if _ladd_bk_name:
                    from model.wan14b_prefix import load_wan14b_prefix
                    if not _ladd_blocks_explicit:
                        raise ValueError(
                            "ladd_disc_backbone_model_name is set but "
                            "``ladd_feature_blocks`` is empty. The 14B "
                            "backbone requires EXPLICIT shallow taps (e.g. "
                            "[0, 2, 4, 8]); the auto-default for a 40-block "
                            "teacher is [8, 16, 24, 32, 39], which would "
                            "load the full 28 GB model."
                        )
                    _bk_dtype_name = str(getattr(
                        self.config, "ladd_disc_backbone_dtype", "bfloat16"
                    ))
                    _bk_dtype = getattr(torch, _bk_dtype_name, None)
                    if not isinstance(_bk_dtype, torch.dtype):
                        raise ValueError(
                            "ladd_disc_backbone_dtype must name a torch "
                            f"dtype; got {_bk_dtype_name!r}."
                        )
                    _bk_max_tap = max(int(b) for b in ladd_blocks)
                    _ladd_backbone = load_wan14b_prefix(
                        _ladd_bk_name,
                        max_block=_bk_max_tap,
                        device=self.device,
                        dtype=_bk_dtype,
                        load_head=bool(getattr(
                            self.config,
                            "ladd_disc_backbone_load_head",
                            True,
                        )),
                    )
                    _bk_layers = len(_ladd_backbone.blocks)
                    # Hard assert: max(taps) < num_layers_loaded.
                    assert _bk_max_tap < _bk_layers, (
                        f"LADD 14B backbone: deepest tap {_bk_max_tap} >= "
                        f"{_bk_layers} blocks loaded."
                    )
                    # Use the BACKBONE's own tokenisation params, never
                    # real_score's. ``a_per_f`` is hard 0: stock T2V has no
                    # action tokens, and harvesting
                    # ``action_tokens_per_frame=1`` off real_score would
                    # silently mis-slice the head reshape.
                    _dim_teacher = int(_ladd_backbone.dim)
                    _patch_size = tuple(
                        int(p) for p in _ladd_backbone.patch_size
                    )
                    _a_per_f = 0
                    if self.is_main_process:
                        # print+flush, NOT logging.info: this run's
                        # ``logging.info`` calls are silently suppressed
                        # (0 hits across two full smoke logs tonight,
                        # WARNING/ERROR unaffected -- root cause unknown).
                        # ``[ROLL]``/``[LADD-DIAG]``/``[42F-DRIFT]`` below
                        # already use this pattern and are proven to
                        # survive; this diagnostic exists specifically to
                        # settle "did the resolved value reach the
                        # process", so it must not be one more line that
                        # silently vanishes.
                        print(
                            "[ActionForcing] LADD disc backbone OVERRIDE: "
                            "%s blocks=%d/%d taps=%s dim_teacher=%d "
                            "patch=%s a_per_f=0 dtype=%s (real_score "
                            "untouched)" % (
                                _ladd_bk_name, _bk_layers,
                                int(getattr(
                                    _ladd_backbone,
                                    "_wan14b_prefix_ckpt_num_layers",
                                    _bk_layers,
                                )),
                                ladd_blocks, _dim_teacher, _patch_size,
                                _bk_dtype_name,
                            ),
                            file=sys.stderr, flush=True,
                        )
                disc = build_ladd_disc(
                    real_score=_real_score,
                    backbone=_ladd_backbone,
                    block_indices=ladd_blocks,
                    dim_teacher=_dim_teacher,
                    dim_proj=int(
                        getattr(self.config, "ladd_proj_dim", 256)
                    ),
                    use_csm=bool(
                        getattr(self.config, "ladd_use_csm", True)
                    ),
                    use_lateral_proj=bool(
                        getattr(self.config, "ladd_use_lateral_proj", False)
                    ),
                    head_kernel_size=int(
                        getattr(self.config, "ladd_disc_head_kernel", 3)
                    ),
                    cmap_dim=int(
                        getattr(self.config, "ladd_cmap_dim", 64)
                    ) if _prompt_embed_dim > 0 else 0,
                    prompt_embed_dim=_prompt_embed_dim,
                    wavelet_hf_enabled=_wavelet_hf_enabled,
                    wavelet_hf_in_channels=_wavelet_in_channels,
                    wavelet_hf_drop_ll=bool(
                        getattr(
                            self.config,
                            "ladd_wavelet_hf_drop_ll",
                            False,
                        )
                    ),
                    wavelet_hf_drop_hh=bool(
                        getattr(
                            self.config,
                            "ladd_wavelet_hf_drop_hh",
                            False,
                        )
                    ),
                    wavelet_hf_adapter_init_gain=float(
                        getattr(
                            self.config,
                            "ladd_wavelet_hf_adapter_init_gain",
                            0.1,
                        )
                    ),
                    wavelet_hf_ll_weight=float(
                        getattr(
                            self.config,
                            "ladd_wavelet_hf_ll_weight",
                            0.15,
                        )
                    ),
                    patch_size=_patch_size,
                    action_tokens_per_frame=_a_per_f,
                    # Parallel stat-head sideband: distribution-match
                    # std statistics adversarially. Opt-in; default
                    # OFF so existing runs are unaffected.
                    stat_head_enabled=bool(
                        getattr(
                            self.config,
                            "ladd_stat_head_enabled",
                            False,
                        )
                    ),
                    stat_head_frames_per_window=int(
                        getattr(
                            self.config,
                            "ladd_stat_head_frames_per_window",
                            int(getattr(self.config, "num_frame_per_block", 3)),
                        )
                    ),
                    stat_head_pool_size=int(
                        getattr(
                            self.config,
                            "ladd_stat_head_pool_size",
                            4,
                        )
                    ),
                    stat_head_hidden_dim=int(
                        getattr(
                            self.config,
                            "ladd_stat_head_hidden_dim",
                            256,
                        )
                    ),
                    scalar_output=bool(getattr(
                        self.config, "ladd_scalar_output", False,
                    )),
                    freeze_projector_mixing=bool(getattr(
                        self.config, "ladd_freeze_projector_mixing", False,
                    )),
                )
                # wavelet_hf_augment: ADD the wavelet HF view to the raw latent
                # (preserve BOTH modalities) instead of REPLACING it. Set
                # post-build so the disc.forward reads it via getattr; default
                # off => replace (legacy / byte-identical).
                disc.wavelet_hf_augment = bool(getattr(
                    self.config, "ladd_wavelet_hf_augment",
                    getattr(self.model, "ladd_wavelet_hf_augment", False)))
                # All-fp32 for R1 stability.
                disc.to(device=self.device, dtype=torch.float32)
                disc.train()
                self.r3gan_disc = disc
                self.ladd_block_indices = ladd_blocks
                if self.world_size > 1:
                    # Trainable params = CCM + CSM + heads + cmapper.
                    # Teacher params are not in disc.parameters() (the
                    # projector is a plain attribute, not a submodule).
                    self.r3gan_disc_ddp = DDP(
                        disc,
                        device_ids=[self.local_rank],
                        output_device=self.local_rank,
                        find_unused_parameters=False,
                        broadcast_buffers=False,
                    )
                if self.is_main_process:
                    n_total = sum(p.numel() for p in disc.parameters())
                    n_train = sum(
                        p.numel() for p in disc.parameters()
                        if p.requires_grad
                    )
                    # print+flush, NOT logging.info: see the note at the
                    # OVERRIDE line above -- this run's logging.info calls
                    # are silently suppressed (0 hits across two full smoke
                    # logs, WARNING/ERROR unaffected), so the two
                    # diagnostics below (pre-existing + the resolved-value
                    # echo) are converted to the pattern already proven to
                    # survive in this file.
                    print(
                        "[ActionForcing] LADD discriminator built: "
                        "blocks=%s dim_teacher=%d dim_proj=%d "
                        "use_csm=%s cmap_dim=%d wavelet_hf=%s scalar=%s "
                        "freeze_mixing=%s "
                        "params_total=%.2fM params_trainable=%.2fM "
                        "(DDP=%s)" % (
                            ladd_blocks, _dim_teacher,
                            int(getattr(self.config, "ladd_proj_dim", 256)),
                            bool(getattr(self.config, "ladd_use_csm", True)),
                            int(getattr(self.config, "ladd_cmap_dim", 64))
                            if _prompt_embed_dim > 0 else 0,
                            bool(_wavelet_hf_enabled),
                            bool(getattr(
                                self.config, "ladd_scalar_output", False)),
                            bool(getattr(
                                self.config,
                                "ladd_freeze_projector_mixing", False,
                            )),
                            n_total / 1e6, n_train / 1e6,
                            self.r3gan_disc_ddp is not None,
                        ),
                        file=sys.stderr, flush=True,
                    )
                    # WP-14B resolved-value echo, promised after tonight's
                    # 14B-smoke OOM investigation: the launch script showed
                    # ``ladd_gen_guidance_micro_batch_groups=16`` and the
                    # smoke still OOM'd, and the reason took a manual trace
                    # of the holder script + $DEXTRA merge order to find --
                    # a launch line looking right is not proof it reached
                    # this process. These three EXACT getattr chains are
                    # copy-pasted from their consumer sites (not
                    # re-derived, so this cannot itself become one of the
                    # two-echoes-disagreeing bugs PIXGAN's review found
                    # elsewhere): a divergence between what a launch script
                    # intended and what ``self.config``/``self.model``
                    # actually hold is now visible at build time, before
                    # any step runs, rather than only inferable from an
                    # OOM stack trace afterward.
                    print(
                        "[ActionForcing] LADD resolved knobs (build-time, "
                        "read from the SAME objects the per-step code "
                        "reads): ladd_gen_guidance_micro_batch_groups=%s "
                        "ladd_disc_micro_batch_groups=%s "
                        "ladd_gt_transition_match_max_real=%s" % (
                            getattr(
                                self.config,
                                "ladd_gen_guidance_micro_batch_groups",
                                getattr(
                                    self.model,
                                    "ladd_gen_guidance_micro_batch_groups",
                                    1)),
                            getattr(
                                self.config, "ladd_disc_micro_batch_groups",
                                getattr(self.model,
                                        "ladd_disc_micro_batch_groups", 1)),
                            getattr(self.model,
                                    "ladd_gt_transition_match_max_real", 12),
                        ),
                        file=sys.stderr, flush=True,
                    )


        # ------------------------------------------------------------------
        # WP-PIXGAN (B1 / T3) — pixel-space texture PatchGAN critic.
        # Spec: docs/TEXTURE_GAN_DESIGN.md §4 / §5.2, docs/WP_PIXGAN.md §1.
        #
        # Gate: ``gan_pixel_texture_enabled`` (default False). This is NOT a
        # new ``gan_backbone`` value — ``gan_backbone`` stays
        # "ladd_teacher_feat" and in the pixel arm the transition GAN is OFF
        # (``gan_enabled: false``). The two critics are INDEPENDENT and may be
        # independently on/off; nothing in this block reads ``gan_enabled``.
        #
        # Built in fp32 for the same reason as the LADD critic: the §5.3 R1
        # finite-difference estimator is arithmetically delicate.
        #
        # The architecture is frozen by §4 (3->64->128->256->1, k4/k4/k4/k3,
        # spectral norm everywhere, GroupNorm(8) on the two middle blocks) and
        # deliberately exposes no config knobs — see model/pixel_texture_disc.py.
        # ------------------------------------------------------------------
        self.gan_pixel_texture_enabled = bool(
            getattr(self.config, "gan_pixel_texture_enabled", False)
        )
        # FROZEN CONTRACT 1 (docs/WP_PIXGAN.md §1): the trainer attribute MUST
        # be named exactly ``pixel_texture_disc`` — with that name
        # ``model/disc_holdout_probe.py`` (``disc_holdout_probe_critic_attr``
        # defaults to "pixel_texture_disc") needs zero edits. Do not rename.
        self.pixel_texture_disc: Optional[torch.nn.Module] = None
        self.pixel_texture_disc_ddp: Optional[DDP] = None
        if self.gan_pixel_texture_enabled:
            # CONFIG-RESOLUTION TIME. Every pix_* invariant (the A24 band
            # pin, the degenerate-count checks, a frozen real pool, an
            # uncalibrated pix_r1_gamma) is checked HERE, at construction,
            # before a single step runs -- not inside whichever consumer
            # happens to reach its own copy of the check first. An arm that
            # is misconfigured should not reach gan_disc_start_step to find
            # out. Return value discarded: this call is the CHECK; the
            # consumers resolve again when they need the numbers.
            self._pix_resolve_cfg()
            from model.pixel_texture_disc import PixelTextureDisc
            pix_disc = PixelTextureDisc()
            # Same device/dtype placement idiom as the LADD critic above.
            pix_disc.to(device=self.device, dtype=torch.float32)
            pix_disc.train()
            self.pixel_texture_disc = pix_disc
            if self.world_size > 1:
                # ``find_unused_parameters=False``: every parameter
                # participates in the graph on a D update. NOTE the R1
                # subsample branch — ``conv4.bias`` receives an exactly-zero
                # (but materialised, not ``None``) gradient from R1 alone,
                # which DDP accepts; see model/pixel_texture_disc.py's header.
                self.pixel_texture_disc_ddp = DDP(
                    pix_disc,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=False,
                    broadcast_buffers=False,
                )
            if self.is_main_process:
                _pix_total = sum(p.numel() for p in pix_disc.parameters())
                _pix_train = sum(
                    p.numel() for p in pix_disc.parameters()
                    if p.requires_grad
                )
                # WARNING, not info: a WIRING claim (was D actually
                # DDP-wrapped? did the arm come up at all?) plus the
                # param counts §12 calibration reads. ``logging.info`` in
                # this trainer is invisible whenever a root handler is
                # installed before basicConfig runs -- see
                # ``_pix_emit_actionable``. The stderr leg that helper
                # also carries cannot be added here: this block is exec'd
                # by testing/test_pixgan_trainer_wiring.py against a
                # namespace holding a recording ``logging`` stub and no
                # ``sys``, and that file is not this package's to change.
                # WARNING alone already clears the failure mode.
                logging.warning(
                    "[ActionForcing] Pixel-texture discriminator built: "
                    "params_total=%.2fM params_trainable=%.2fM "
                    "(DDP=%s, gan_enabled=%s)",
                    _pix_total / 1e6, _pix_train / 1e6,
                    self.pixel_texture_disc_ddp is not None,
                    self.gan_enabled,
                )

        # ------------------------------------------------------------------
        # WP-SURROGATE (B3) — latent surrogate critic. BUILD STEP ONLY this
        # window; see docs/WP_SURROGATE.md for the full mechanism. Sits
        # immediately after the pixel critic above because that critic IS
        # this one's teacher (``pixel_texture_disc o decode``) — matching
        # the teacher-before-student ordering the mechanism uses everywhere
        # else.
        #
        # Gate: ``surrogate_critic_enabled`` (default False), independent
        # of ``gan_pixel_texture_enabled`` AT BUILD TIME — the critic can
        # be constructed with no teacher connected. A real distillation
        # step needs the pixel critic, but nothing here enforces that
        # dependency at BUILD time. The per-step wiring is landed
        # (2026-08-24, MAIN under researcher order,
        # docs/TASK_SURROGATE_CONSUMPTION.md): consumption branch in
        # ``_compute_pixel_texture_g_loss``, distillation in
        # ``_maybe_run_surrogate_distillation`` (called right after the
        # pixel D-updates), FAIL-LOUD save/resume (save block in
        # causal_rolling_staircase_train.py, restore in ``_maybe_resume``
        # here). ``_maybe_run_surrogate_distillation`` enforces the
        # teacher dependency at RUN time (missing pixel disc -> regime
        # flag, missing distiller while enabled -> raise).
        #
        # ``build_from_config`` (model/latent_texture_critic.py) is the
        # ONLY place these config keys are read — the fix for the seam
        # class this campaign kept finding (a knob read off two different
        # objects, or read correctly but never observably reaching its
        # consumer). This block is therefore exactly the four-line call the
        # seam tests were written against, plus a resolved-value echo
        # (docs/WP_PIXGAN.md's rule: log what was ACTUALLY built off the
        # live objects, not what the config requested).
        # ------------------------------------------------------------------
        from model.latent_texture_critic import build_from_config
        (
            self.latent_texture_critic,
            self.latent_critic_optimizer,
            self.latent_texture_distiller,
        ) = build_from_config(self.config, device=self.device)
        # Derived, never independently read off config — so this flag can
        # never drift from what was actually built (the seam class above).
        self.surrogate_critic_enabled = self.latent_texture_critic is not None
        if self.surrogate_critic_enabled and self.is_main_process:
            logging.info(
                "[ActionForcing] Latent surrogate critic built: "
                "params=%.2fM d_model=%d num_blocks=%d "
                "pix_teacher_refresh_every=%d grad_loss_normalize=%s "
                "grad_loss_weight=%.4g value_loss_weight=%.4g "
                "cache_capacity=%d teacher_use_checkpoint=%s",
                self.latent_texture_critic.num_params / 1e6,
                self.latent_texture_critic.d_model,
                self.latent_texture_critic.num_blocks,
                self.latent_texture_distiller.pix_teacher_refresh_every,
                self.latent_texture_distiller.grad_loss_normalize,
                self.latent_texture_distiller.grad_loss_weight,
                self.latent_texture_distiller.value_loss_weight,
                self.latent_texture_distiller.cache.capacity,
                self.latent_texture_distiller.teacher_use_checkpoint,
            )
            if (not getattr(self, "gan_pixel_texture_enabled", False)
                    and str(getattr(
                        self.config, "surrogate_teacher_backbone", "pixel",
                    )) == "pixel"
                    and self.is_main_process):
                # Found on the 2026-08-24 consumption-wiring review: BOTH
                # consumption sites are structurally gated on the pixel
                # critic being present (``_compute_pixel_texture_g_loss``
                # returns before ever reaching the surrogate branch when
                # ``gan_pixel_texture_enabled`` is False;
                # ``_maybe_run_surrogate_distillation`` early-returns on
                # ``pixel_texture_disc is None``, which is exactly this
                # case). So this exact configuration builds a full critic
                # + optimizer + distiller that will NEVER train and NEVER
                # be consumed for the whole run -- indistinguishable, from
                # the INFO line above, from a working surrogate. A WARNING
                # here (not a raise -- validating gate combinations is out
                # of this window's authorized scope, docs/
                # TASK_SURROGATE_CONSUMPTION.md §5) closes the "looks
                # active, is inert" gap the campaign kept finding
                # elsewhere, without changing what any existing config does.
                logging.warning(
                    "[ActionForcing] surrogate_critic_enabled=true but "
                    "gan_pixel_texture_enabled=false: the surrogate critic "
                    "just built will NEVER train (no teacher) and NEVER be "
                    "consumed by the generator (the G-term branch is "
                    "unreachable without the pixel critic) for this entire "
                    "run. This is not an error, but it is very likely not "
                    "what was intended -- set gan_pixel_texture_enabled=true "
                    "as well, or surrogate_critic_enabled=false.",
                )

        # ------------------------------------------------------------------
        # WP-SURROGATE (B3) — TEACHER BACKBONE selection (researcher
        # directive, 2026-08-24 afternoon): this branch's surrogate teacher
        # must be a PRETRAINED pixel discriminator, not the from-scratch
        # PatchGAN. ``surrogate_teacher_backbone: sam2`` restores the
        # 835b1df teacher identity — frozen SAM2 Hiera-B+ encoder +
        # trainable ADM 2D heads (model/r3gan_sam2.py, restored verbatim).
        # Default stays "pixel" so every existing config is byte-identical;
        # the from-scratch teacher remains available to B1's own arm.
        # ------------------------------------------------------------------
        self.surrogate_teacher_backbone = str(
            getattr(self.config, "surrogate_teacher_backbone", "pixel")
        )
        self.sam2_teacher_disc = None
        self.sam2_teacher_optimizer = None
        if (self.surrogate_critic_enabled
                and self.surrogate_teacher_backbone in ("sam2", "dinov2", "convnext")):
            _bk = self.surrogate_teacher_backbone
            if _bk != "sam2":
                # dinov2 / convnext -- pretrained ALTERNATIVES so the
                # teacher basis is measured, not assumed. Same ADM heads
                # as sam2 (pretrained_pixel_disc reuses _R3GANDiscHeads
                # verbatim), so an arm-to-arm delta is attributable to the
                # FEATURE BASIS alone. res=None -> that backbone's own
                # default, each clearing the head's 8x8 minimum.
                from model.pretrained_pixel_disc import PretrainedPixelDisc
                _res = getattr(self.config, f"surrogate_{_bk}_resolution", None)
                _var = getattr(self.config, f"surrogate_{_bk}_variant", None)
                _lay = getattr(self.config, f"surrogate_{_bk}_layers", None)
                _s2 = PretrainedPixelDisc(
                    _bk,
                    variant=(str(_var) if _var else None),
                    layers=(list(_lay) if _lay else None),
                    image_resolution=(int(_res) if _res else None),
                    device=self.device, dtype=torch.float32,
                    pad_to_square=bool(getattr(
                        self.config, f"surrogate_{_bk}_pad_to_square", True)),
                    frame_pool=str(getattr(
                        self.config, f"surrogate_{_bk}_frame_pool", "mean")),
                )
            else:
                from model.r3gan_sam2 import R3GANDiscriminatorSAM2Pixel
                _s2_ckpt = getattr(
                    self.config, "surrogate_sam2_checkpoint_path", None,
                )
                _s2_cfg = getattr(
                    self.config, "surrogate_sam2_config_path", None,
                )
                if not _s2_ckpt or not _s2_cfg:
                    raise ValueError(
                        "surrogate_teacher_backbone=sam2 requires "
                        "surrogate_sam2_checkpoint_path and "
                        "surrogate_sam2_config_path."
                    )
                _s2 = R3GANDiscriminatorSAM2Pixel(
                    sam2_checkpoint_path=str(_s2_ckpt),
                    sam2_config_path=str(_s2_cfg),
                    image_resolution=int(getattr(
                        self.config, "surrogate_sam2_resolution", 512)),
                    device=self.device,
                    dtype=torch.float32,
                    preserve_aspect=bool(getattr(
                        self.config, "surrogate_sam2_preserve_aspect", True)),
                    pad_to_square=bool(getattr(
                        self.config, "surrogate_sam2_pad_to_square", False)),
                    frame_pool=str(getattr(
                        self.config, "surrogate_sam2_frame_pool", "mean")),
                )
            _s2.train()  # heads -> train; encoder pinned eval by class
            self.sam2_teacher_disc = _s2
            self.sam2_teacher_optimizer = torch.optim.Adam(
                _s2.heads_module.parameters(),
                lr=float(getattr(
                    self.config, f"surrogate_{_bk}_lr",
                    getattr(self.config, "surrogate_sam2_lr", 2e-4))),
                betas=(0.0, 0.9),
            )
            # NOTE deliberately NO DDP wrap in this window: the SAM2 heads
            # D-update runs identically on every rank (crops come from the
            # rank-synced private generator), and the smoke asserts
            # mechanics, not multi-rank optimality. Heads-only DDP (the
            # ancestor's gan_sam2_distilled_critic topology) is the
            # follow-up when this graduates from smoke to arm.
            if self.is_main_process:
                _s2_total = sum(p.numel() for p in _s2.parameters())
                _s2_train = sum(
                    p.numel() for p in _s2.heads_module.parameters()
                    if p.requires_grad
                )
                logging.info(
                    "[ActionForcing] PRETRAINED surrogate teacher built: "
                    "backbone=%s params_total=%.1fM heads_trainable=%.2fM "
                    "resolution=%d lr=%.2g (encoder frozen)",
                    self.surrogate_teacher_backbone, _s2_total / 1e6, _s2_train / 1e6,
                    _s2.image_resolution,
                    self.sam2_teacher_optimizer.param_groups[0]["lr"],
                )

        # ------------------------------------------------------------------
        # SC-DMD (Salt) — semigroup defect regularizer. Default OFF.
        # No new modules / optimizers needed; the SC pass shares the
        # existing generator's parameters via the same DDP-wrapped
        # ``model.generator``. We only thread a config flag, weight,
        # and warmup into the gen-loss compute path. Triplet sampling
        # and the two extra DiT forwards live on
        # ``ActionForcingDMD.sc_dmd_loss``; this trainer just calls it
        # and adds the weighted loss to the unified generator backward.
        # ------------------------------------------------------------------
        self.sc_dmd_enabled = bool(
            getattr(self.config, "sc_dmd_enabled", False)
        )
        self.sc_dmd_loss_weight = float(
            getattr(self.config, "sc_dmd_loss_weight", 0.05)
        )
        self.sc_dmd_warmup_steps = int(
            getattr(self.config, "sc_dmd_warmup_steps", 0)
        )
        if self.sc_dmd_enabled and self.is_main_process:
            logging.info(
                "[ActionForcing] SC-DMD (Salt) ENABLED: weight=%.4f "
                "warmup_steps=%d (loss multiplier ramps linearly from "
                "0 to ``weight`` over ``warmup_steps`` steps; SC pass "
                "costs ~2 extra DiT forwards on a ``num_frame_per_block``-"
                "frame chunk per gen step).",
                self.sc_dmd_loss_weight,
                self.sc_dmd_warmup_steps,
            )

        # ------------------------------------------------------------------
        # Online causal-CD loss (Causal-Forcing++, arXiv 2605.15141). The
        # consistency math lives in ``ActionForcingDMD.cd_loss`` (single
        # chunk, teacher-forced clean_x=GT, resident-GPU EMA target); this
        # trainer just calls it once per ride and folds the weighted loss
        # into the unified generator backward (mirrors SC-DMD). Default off.
        # ------------------------------------------------------------------
        self.cd_loss_enabled = bool(
            getattr(self.config, "cd_loss_enabled", False)
        )
        self.cd_loss_weight = float(
            getattr(self.config, "cd_loss_weight", 1.0)
        )
        self.cd_loss_warmup_steps = int(
            getattr(self.config, "cd_loss_warmup_steps", 0)
        )
        if self.cd_loss_enabled and self.is_main_process:
            logging.info(
                "[ActionForcing] causal-CD loss ENABLED: weight=%.4f "
                "warmup_steps=%d (CD pass costs 3 extra DiT forwards "
                "[frozen-teacher ODE step + student + EMA-student] on a "
                "``num_frame_per_block``-frame teacher-forced chunk per "
                "gen step; resident-GPU EMA student adds ~2.6GB).",
                self.cd_loss_weight,
                self.cd_loss_warmup_steps,
            )

        # ------------------------------------------------------------------
        # dmd_context (hardcoded "self") — single source of truth
        # is the model. ``self.model.dmd_context_clean_frames`` is the
        # KV-cache seed prefill size (= 9 by default, configurable via
        # the ``dmd_context_clean_frames`` knob the model reads from
        # ``args``). The trainer just reads it back to size its ride
        # slices. The clean/noisy shift used inside DMD scoring is fixed
        # at ``num_frame_per_block`` (= 1 chunk) and lives entirely in
        # the model — the trainer never sees it.
        # ------------------------------------------------------------------

        # Streaming-mode flag (LongLive-style persistent KV cache +
        # rolling-sequence training). When True, ``_fwdbwd_one_step``
        # uses ``model.setup_sequence`` / ``generate_next_chunk`` /
        # ``compute_*_loss_streaming`` instead of the legacy single-
        # iter ``generator_loss`` / ``critic_loss`` flow.
        # FUNDAMENTAL: defaults to True. Set ``streaming_mode: false``
        # in the YAML to fall back to the legacy single-batch path.
        self.streaming_mode: bool = bool(
            getattr(self.config, "streaming_mode", True)
        )
        self.streaming_max_length: int = int(
            getattr(self.config, "streaming_max_length", 57)
        )
        # Phase-B streaming-mode aux wiring is in
        # ``_fwdbwd_streaming_step`` (action-critic z-guidance, R3GAN,
        # SC-DMD all mirror the legacy gen-with-aux block; sliced to
        # the per-iter chunk's ride window via the streaming state).

        self.sample_interval = int(
            getattr(self.config, "sample_interval", 0) or 0
        )
        self.sample_fps = int(getattr(self.config, "sample_fps", 5) or 5)
        self.sample_max_frames = int(
            getattr(self.config, "sample_max_frames", 0) or 0
        )
        # pred_image_7_chunk eval video: seed the student with 7 GT
        # context chunks, roll 7 more, log the full 14-chunk rollout.
        # Default ON; disable via ``sample_7chunk_enabled=false`` if the
        # all-ranks inference rollout ever misbehaves on the cluster.
        self.sample_7chunk_enabled = bool(
            getattr(self.config, "sample_7chunk_enabled", True)
        )
        self._sample_7chunk_ride: Optional[Dict[str, torch.Tensor]] = None
        # Optional explicit step list (in addition to the periodic interval).
        # Semantics: each entry fires on the FIRST eligible step at or
        # after that value. Necessary because dfake_gen_update_ratio>1
        # makes even-indexed steps critic-only (no rollout to emit), so
        # asking for an emission at e.g. step 10 with ratio=2 has to be
        # interpreted as "at the next gen-step", which is step 11.
        _sas = getattr(self.config, "sample_at_steps", None)
        self._sample_at_steps_pending: list = (
            sorted({int(s) for s in _sas}) if _sas else []
        )
        # When true, ALSO save mp4 to <log_dir>/samples/step_<n>.mp4 — this
        # works even when wandb is disabled, so smoke runs can inspect early
        # rollouts.
        self.vis_save_local = bool(
            getattr(self.config, "vis_save_local", False)
        )
        self._pending_video_latents: Optional[torch.Tensor] = None
        self._video_logger_warned_no_vae: bool = False
        # Diagnostic stash populated by the DMD model when a sample
        # video is due. Keys: ``pred_real``, ``pred_fake``,
        # ``clean_x_fake``, ``clean_x_real`` (latent tensors), plus
        # scalar metadata (``dmd_timestep``, ``clean_x_aug_t``,
        # ``dmd_context``). Decoded alongside ``_pending_video_latents``
        # at the video-logger callsite.
        self._pending_dmd_eval_latents: Optional[Dict[str, Any]] = None
        # Sticky "due" bits — set when the cadence boundary is crossed
        # on a step that can't actually emit (critic-only iter under
        # dfake_gen_update_ratio>1). The next iter that CAN emit
        # (gen-iter with a fresh chunk / generator_log_dict) consumes
        # the bit. Without these, log_interval / sample_interval values
        # whose parity collides with dfake_gen_update_ratio silently
        # never log — what bit us at the start of phase_1_b2b.
        self._wandb_log_due: bool = False
        self._video_sample_due_bit: bool = False
        if self.sample_interval > 0 and self.is_main_process:
            logging.info(
                "[ActionForcing] Video sampling ENABLED: every %d "
                "generator steps, decode pred_image -> mp4 -> wandb "
                "(fps=%d, max_frames=%s)",
                self.sample_interval,
                self.sample_fps,
                ("all" if self.sample_max_frames <= 0
                 else str(self.sample_max_frames)),
            )
        if self.is_main_process:
            cf = int(getattr(self.model, "dmd_context_clean_frames", 0))
            N = int(getattr(self.config, "num_training_frames", 21))
            logging.info(
                "[ActionForcing] dmd_context: KV-cache seed prefill "
                "= %d frames (= dmd_context_clean_frames). Per iter, "
                "trainer picks random offset s ~ U[0, ride_len/2] "
                "(broadcast from rank 0 in DDP), then slices "
                "seed=ride[s:s+%d] and rollout=ride[s+%d:s+%d]; "
                "scorers see clean/noisy shift of one chunk "
                "(= num_frame_per_block).",
                cf, cf, cf, cf + N,
            )

    def _build_pipeline(self) -> None:
        cfg = self.config
        denoising_step_list = list(getattr(
            cfg, "denoising_step_list", [1000, 750, 500, 250],
        ))
        num_frame_per_block = int(getattr(cfg, "num_frame_per_block", 3))
        chunks_per_rolling_step = int(getattr(cfg, "chunks_per_rolling_step", 1))
        same_step_across_blocks = bool(getattr(cfg, "same_step_across_blocks", True))
        last_step_only = bool(getattr(cfg, "last_step_only", False))
        num_max_frames = int(getattr(cfg, "num_training_frames", 21))
        # CF-style long rollout: ``rollout_frames`` (default = unset =
        # ``num_training_frames``) controls the per-iter rollout
        # length. When > num_training_frames, the pipeline rolls that
        # many frames but only the LAST ``num_training_frames`` get
        # gradient — matches CF's
        # ``start_gradient_frame_index = num_output_frames - 21``.
        rollout_frames_raw = getattr(cfg, "rollout_frames", None)
        rollout_frames = (
            int(rollout_frames_raw) if rollout_frames_raw is not None
            else num_max_frames
        )
        if rollout_frames < num_max_frames:
            raise ValueError(
                f"cfg.rollout_frames ({rollout_frames}) must be >= "
                f"num_training_frames ({num_max_frames}). The gradient "
                f"window is the last num_training_frames frames of the "
                f"rollout, so the rollout must be at least that long."
            )
        if rollout_frames % num_frame_per_block != 0:
            raise ValueError(
                f"cfg.rollout_frames ({rollout_frames}) must be divisible "
                f"by num_frame_per_block ({num_frame_per_block})."
            )
        context_noise = int(getattr(cfg, "context_noise", 0))
        # Optional non-uniform exit-flag sampling for phase-LoRA (or
        # plain DMD) training. List of K non-negative floats reweights
        # the random exit-rung selection; None / empty preserves uniform
        # behavior. Use to concentrate training on middle-t rungs
        # (min-SNR-γ-style) where score-matching has the best signal-
        # to-gradient-variance trade-off — extremes (rung 0 = highest
        # noise, rung K-1 = lowest noise) get less weight.
        exit_flag_weights_raw = getattr(cfg, "exit_flag_weights", None)
        exit_flag_weights: Optional[List[float]] = None
        if exit_flag_weights_raw is not None:
            if isinstance(exit_flag_weights_raw, str):
                # OmegaConf override "[0.1,0.4,0.4,0.1]" may arrive as
                # a string when the launcher does not parse the list
                # literal. Defensive parse handles both forms.
                try:
                    exit_flag_weights = list(
                        ast.literal_eval(exit_flag_weights_raw)
                    )
                except Exception:
                    exit_flag_weights = None
            else:
                exit_flag_weights = [float(v) for v in exit_flag_weights_raw]
            if exit_flag_weights is not None and (
                len(exit_flag_weights) != len(denoising_step_list)
            ):
                raise ValueError(
                    f"cfg.exit_flag_weights has length "
                    f"{len(exit_flag_weights)} but denoising_step_list "
                    f"has length {len(denoising_step_list)}"
                )

        self.pipeline = ActionForcingTrainingPipeline(
            denoising_step_list=denoising_step_list,
            scheduler=self.model.scheduler,
            generator=self.model.generator,
            num_frame_per_block=num_frame_per_block,
            chunks_per_rolling_step=chunks_per_rolling_step,
            same_step_across_blocks=same_step_across_blocks,
            last_step_only=last_step_only,
            num_max_frames=num_max_frames,
            rollout_frames=rollout_frames,
            context_noise=context_noise,
            exit_flag_weights=exit_flag_weights,
            # 14e alignment: the DMD streaming path prefills
            # ``dmd_context_clean_frames`` of context into the KV cache
            # BEFORE the first generated chunk. The pipeline needs that
            # count to size the buffer so the cache never evicts, and
            # ``seed_prefill_mode`` selects what gets written there.
            # Both default to the legacy behaviour (0 frames declared,
            # "estimate" prefill) so every other config is unchanged.
            seed_prefill_frames=int(
                getattr(cfg, "dmd_context_clean_frames", 0) or 0),
            seed_prefill_mode=str(
                getattr(cfg, "seed_prefill_mode", "estimate")),
            kv_cache_seed_headroom=bool(
                getattr(cfg, "kv_cache_seed_headroom", False)),
        )
        # The ActionForcingDMD model needs the pipeline reference for backward
        # simulation inside generator_loss / critic_loss.
        self.model.inference_pipeline = self.pipeline

        # ------------------------------------------------------------------
        # WP-PIXGAN (B1 / T3-B) — CONFIG -> PIPELINE SEAM for the A23 gate.
        #
        # ``pipeline/action_forcing_training.py`` reads this gate as a PLAIN
        # ATTRIBUTE ON ITSELF inside
        # ``ActionForcingTrainingPipeline.generate_chunk_with_cache``:
        # ``getattr(self, "pix_finish_grad_enabled", False)``. That is that
        # file's convention and not ours to change. This trainer reads the
        # knob off ``self.config``. Nothing connected the two, so
        # ``pix_finish_grad_enabled=true`` on the launch line set the
        # trainer's view True while the pipeline's stayed False: the grad
        # buffer was never built, ``finish_denoised_chunk_grad`` published
        # None, and the ladder-fake path could not run at all. Found by a
        # live 60-step smoke; the override guard's "merged silently, NO
        # effect" report was a TRUE POSITIVE.
        #
        # THE RECEIVER IS ``self.pipeline``, NOT ``self.model``. Both exist
        # and both are plausible: ``self.model`` is the ActionForcingDMD
        # module and ``self.pipeline`` is the ActionForcingTrainingPipeline,
        # and the trainer even cross-links them
        # (``self.model.inference_pipeline = self.pipeline`` above). Setting
        # the flag on ``self.model`` compiles, runs, logs, and does nothing
        # whatsoever — a silent no-op indistinguishable from the original
        # bug. ``test_flag_lands_on_the_object_whose_class_reads_it`` pins
        # the receiver by PARSING which class performs the read, so this
        # cannot drift back.
        #
        # Set UNCONDITIONALLY (both True and False), never conditionally:
        # behaviourally byte-identical when off, because the pipeline's own
        # read defaults to False when the attribute is absent; but an
        # explicit False is DISTINGUISHABLE from "nobody ever set it", which
        # is the difference between a configured-off arm and this exact
        # plumbing bug recurring silently.
        # ------------------------------------------------------------------
        self.pipeline.pix_finish_grad_enabled = bool(
            getattr(self.config, "pix_finish_grad_enabled", False)
        )
        if self.is_main_process:
            # WARNING + stderr, not info. This line is the ONLY runtime
            # evidence that the A23 gate reached the object whose class
            # reads it; the bug it exists to expose (set on self.model
            # instead of self.pipeline) is a silent no-op that every
            # value-level check still passes. A proof-of-connection that
            # is itself only conditionally connected proves nothing --
            # ``logging.info`` here is invisible whenever a root handler
            # is installed before basicConfig runs. See
            # ``_pix_emit_actionable`` for the mechanism; emitted inline
            # rather than through it because this block is exec'd as a
            # SEAM fixture against a bare namespace with no bound methods.
            _a23_msg = (
                "[ActionForcing] A23 pipeline gate propagated to %s: "
                "pix_finish_grad_enabled=%s" % (
                    type(self.pipeline).__name__,
                    self.pipeline.pix_finish_grad_enabled,
                )
            )
            logging.warning(_a23_msg)
            print(_a23_msg, file=sys.stderr, flush=True)

        # ------------------------------------------------------------------
        # exit_exclude_last_rung (MAIN 14:1x, RESEARCHER-APPROVED for the KV
        # recompute-order probe): excludes the last denoising rung from the
        # streaming exit draw so the A23 grad path always has a finish rung
        # to attach to. Default False = byte-identical (the pipeline call
        # site's previous hardcoded value). This CHANGES THE DMD EXIT
        # DISTRIBUTION when on -- probe/diagnostic use only, never a
        # default. Same set-unconditionally + WARNING-echo discipline as
        # the A23 gate above (same silent-no-op plumbing hazard).
        # ------------------------------------------------------------------
        self.pipeline.exit_exclude_last_rung = bool(
            getattr(self.config, "exit_exclude_last_rung", False)
        )
        if self.is_main_process and self.pipeline.exit_exclude_last_rung:
            _eelr_msg = (
                "[ActionForcing] exit_exclude_last_rung=True propagated to "
                "%s: streaming exit rungs sampled from [0, K-1) -- DMD exit "
                "distribution CHANGED (KV-probe regime)."
                % type(self.pipeline).__name__
            )
            logging.warning(_eelr_msg)
            print(_eelr_msg, file=sys.stderr, flush=True)

        if self.is_main_process:
            logging.info(
                "[ActionForcing] Pipeline: num_frame_per_block=%d "
                "chunks_per_rolling_step=%d denoising_step_list=%s "
                "context_noise=%d num_max_frames=%d rollout_frames=%d "
                "(gradient window = last %d frames; warmup = %d frames)",
                num_frame_per_block, chunks_per_rolling_step,
                denoising_step_list, context_noise, num_max_frames,
                rollout_frames, num_max_frames,
                rollout_frames - num_max_frames,
            )

        # ------------------------------------------------------------------
        # Infinity-RoPE (Block-Relativistic RoPE) on the KV-cache path.
        # ``utils.infinity_rope.install()`` monkey-patches
        # ``CausalWanSelfAttention.forward`` to store un-roped K in the
        # cache and rotate Q / cached-K with bounded window-relative
        # indices at attention time. Default ON so that the Action-
        # Forcing student is trained with the SAME RoPE convention
        # that ``utils/eval_causal_AR.py`` runs with at AR rollout
        # time (its ``infinity_rope`` flag also defaults true). Within
        # an Action-Forcing training iter the cache never rolls
        # (num_training_frames ==
        # local_attn_size, sink_size=0), so the patch is mathematically
        # equivalent to the original RoPE path during training; the
        # point is purely train/inference parity. Set
        # ``infinity_rope=false`` in the YAML / via ``--override`` to
        # restore the original RoPE training path for A/B studies.
        infinity_rope_enabled = bool(getattr(cfg, "infinity_rope", True))
        if infinity_rope_enabled:
            _install_infinity_rope(self._inner_dit_for_rope())
            if self.is_main_process:
                logging.info(
                    "[ActionForcing] Infinity-RoPE patch installed on "
                    "CausalWanSelfAttention.forward (train/inference "
                    "parity with utils/eval_causal_AR.py defaults)."
                )
        else:
            # PREREQUISITE for the original (absolute ``causal_rope_apply``)
            # cached path when per-frame action tokens are live.
            # ``causal_model.py:306-326`` ropes the cached branch as a
            # contiguous (f, h, w) grid UNLESS ``cached_rope_action_aware``
            # is set on the attention module; with
            # ``action_tokens_per_frame=1`` the interleaved action token
            # shifts every frame's spatial grid by one token — a per-frame
            # column shear, i.e. a WORSE defect than the RoPE convention
            # mismatch we are removing. Every other cache-driven consumer
            # of these weights sets it (``utils/eval_causal_AR.py:648-654``,
            # ``utils/causal_chain_rollout.py:76-82``,
            # ``action-forcing/af_model/ode_rollout.py:228-229``); the DMD
            # trainer never did, because the infinity-RoPE patch (default
            # ON) bypasses that code entirely.
            #
            # Scope: the flag is read ONLY inside the ``kv_cache is not
            # None`` branch, so it is inert on the teacher-forced scorer
            # forwards (which pass no kv_cache). Setting it on the scorers
            # too therefore changes nothing for the TF heads and fixes the
            # optional AR head (``dmd_ar_head_weight``), which drives the
            # same weights through a real KV cache.
            #
            # Predicate is ``hasattr`` (as in ode_rollout.py:228 /
            # causal_chain_rollout.py:76), NOT ``value > 0``: the DiT only
            # PROPAGATES its ``action_tokens_per_frame`` down to
            # ``block.self_attn`` inside ``_forward_inference`` /
            # ``_forward_train`` (causal_model.py:1276-1278, 1579), so at
            # construction time every attention module still reads 0.
            # Setting the flag where apf is genuinely 0 is a no-op — the
            # consuming branch is ``_apf > 0 and cached_rope_action_aware``.
            _rope_targets = [self._inner_dit_for_rope()]
            for _name in ("real_score", "fake_score"):
                _sm = getattr(self.model, _name, None)
                if _sm is not None:
                    _rope_targets.append(_sm)
            _cra_set = 0
            for _root in _rope_targets:
                if _root is None:
                    continue
                for _m in _root.modules():
                    if hasattr(_m, "action_tokens_per_frame"):
                        _m.cached_rope_action_aware = True
                        _cra_set += 1
            if self.is_main_process:
                logging.info(
                    "[ActionForcing] Infinity-RoPE patch DISABLED "
                    "(infinity_rope=false). Training will run the "
                    "original absolute-RoPE cached path; "
                    "cached_rope_action_aware=True set on %d module(s) so "
                    "the interleaved per-frame action tokens are not roped "
                    "as spatial ones. AR-eval defaults to "
                    "infinity_rope=true so a train/inference mismatch "
                    "is expected on long-horizon rollouts.",
                    _cra_set,
                )

        # ------------------------------------------------------------------
        # Step-scheduled local_attn_size (KV cache window). Lets training
        # start cheap (small attention window = fast iters, modest cache)
        # then grow context as the student's predictions stabilise and
        # long-horizon coherence becomes the bottleneck. Format:
        #
        #   local_attn_size_schedule: [[step_threshold, frames], ...]
        #
        # At step S, the active window is the size for the LARGEST
        # threshold ≤ S. Omit the key (or set null) to disable the
        # schedule and use the static ``model_kwargs.local_attn_size``
        # for the whole run. The schedule first waypoint should match
        # ``model_kwargs.local_attn_size`` (the model's construction-
        # time value) so step-0 attention behaviour is unchanged.
        sched_raw = getattr(cfg, "local_attn_size_schedule", None)
        self._attn_size_schedule: List[Tuple[int, int]] = []
        if sched_raw:
            try:
                pairs = [(int(s), int(n)) for s, n in sched_raw]
                pairs.sort(key=lambda p: p[0])
                self._attn_size_schedule = pairs
            except Exception as e:
                if self.is_main_process:
                    logging.warning(
                        "[ActionForcing] Bad local_attn_size_schedule %r: %s "
                        "— ignoring schedule.", sched_raw, e,
                    )
        # Last applied frames count, set to None until first apply so the
        # first sequence open at step 0 always seeds with the right size.
        self._current_attn_frames: Optional[int] = None
        if self._attn_size_schedule and self.is_main_process:
            logging.info(
                "[ActionForcing] local_attn_size_schedule active: %s",
                self._attn_size_schedule,
            )

    # ------------------------------------------------------------------
    # Optimizer (parent builds gen + fake_score; we add the critic).
    # ------------------------------------------------------------------
    def _build_optimizer(self) -> None:
        super()._build_optimizer()
        cfg = self.config
        # Aux-loss hyperparameters. Defaults match the ODE distill
        # teacher trainer (``causal_diffusion_teacher_train.py``):
        # critic_lr=3e-4, weight=0.25, gen z-guidance=0.125 (warmup
        # 500), 2 critic updates per gen step, dims=[2,7].
        critic_dims_cfg = getattr(cfg, "action_critic_dims", None)
        self.action_critic_dims: List[int] = (
            list(critic_dims_cfg) if critic_dims_cfg is not None else [2, 7]
        )
        self.action_critic_z_loss_weight = float(
            getattr(cfg, "action_critic_z_loss_weight", 0.25)
        )
        self.generator_action_z_guidance_weight = float(
            getattr(cfg, "generator_action_z_guidance_weight", 0.0)
        )
        self.z_guidance_warmup_steps = int(
            getattr(cfg, "z_guidance_warmup_steps", 500)
        )
        self.warmup_steps_for_guidance = int(getattr(cfg, "warmup_steps", 0))
        self.critic_updates_per_step = int(
            getattr(cfg, "critic_updates_per_step", 1)
        )
        self.critic_optimizer: Optional[torch.optim.Optimizer] = None
        if self.action_critic_loss_active and self.model.action_critic is not None:
            critic_lr = float(getattr(cfg, "critic_lr", 3e-4))
            critic_betas = tuple(
                getattr(cfg, "critic_betas", getattr(cfg, "betas", [0.9, 0.999]))
            )
            critic_eps = float(getattr(cfg, "critic_eps", getattr(cfg, "eps", 1e-8)))
            critic_wd = float(
                getattr(cfg, "critic_weight_decay", getattr(cfg, "weight_decay", 0.0))
            )
            critic_params = [
                p for p in self.model.action_critic.parameters() if p.requires_grad
            ]
            if not critic_params:
                raise RuntimeError(
                    "action_critic has no trainable parameters even after "
                    "un-freeze; check ActionCritic.__init__ + the trainer's "
                    "requires_grad_(True) call in _build_model."
                )
            self.critic_optimizer = torch.optim.AdamW(
                critic_params,
                lr=critic_lr,
                betas=critic_betas,
                eps=critic_eps,
                weight_decay=critic_wd,
            )
            self.critic_max_grad_norm = float(
                getattr(cfg, "critic_max_grad_norm", self.max_grad_norm)
            )
            if self.is_main_process:
                n_params = sum(p.numel() for p in critic_params)
                logging.info(
                    "[ActionForcing] Action critic optimizer built: "
                    "AdamW lr=%.2e wd=%.4f params=%d (z_loss_weight=%.4f, "
                    "gen_z_guidance_weight=%.4f, warmup_steps=%d, "
                    "z_guidance_warmup_steps=%d, critic_updates_per_step=%d, "
                    "dims=%s)",
                    critic_lr, critic_wd, n_params,
                    self.action_critic_z_loss_weight,
                    self.generator_action_z_guidance_weight,
                    self.warmup_steps_for_guidance,
                    self.z_guidance_warmup_steps,
                    self.critic_updates_per_step,
                    self.action_critic_dims,
                )

        # ------------------------------------------------------------------
        # R3GAN discriminator optimizer + hyperparameters. Defaults
        # follow the StyleGAN/R3GAN convention: AdamW with
        # ``betas=(0.0, 0.9)`` (high-momentum first-order beta is
        # *unsafe* for GAN D — keep first-moment beta near 0 so D
        # doesn't lock into pre-G's-update statistics).
        # ------------------------------------------------------------------
        self.gan_loss_weight = float(getattr(cfg, "gan_loss_weight", 0.05))
        self.gan_r1_gamma = float(getattr(cfg, "gan_r1_gamma", 1.0))
        # GAN-vs-DMD rebalancing under the MAE gate. When the MAE gate cuts the
        # DMD weight (teacher unreliable, esp. at high t), the GAN otherwise runs
        # at full strength and can drag the student off the teacher manifold onto
        # the (possibly overtrained) discriminator's. When enabled, the gen-side
        # GAN weight follows the gate DOWN, but gentler (``gate_w**beta``, beta<1
        # => sqrt), never below a floor, and kept >= ``min_ratio`` x the gated
        # DMD weight so the GAN still carries when DMD is bad. Default off =>
        # byte-identical.
        self.gan_gate_couple_enabled = bool(
            getattr(cfg, "gan_gate_couple_enabled", False))
        self.gan_gate_couple_beta = float(
            getattr(cfg, "gan_gate_couple_beta", 0.5))
        self.gan_gate_couple_floor_frac = float(
            getattr(cfg, "gan_gate_couple_floor_frac", 0.3))
        self.gan_gate_couple_min_ratio = float(
            getattr(cfg, "gan_gate_couple_min_ratio", 2.0))
        # Scale sanity for the coupling: the ``>= min_ratio x gated-DMD`` floor
        # is only satisfiable while gan_full >= min_ratio*dmd_eff. With gan_full
        # = gan_loss_weight*ladd_disc_loss_weight and dmd_eff = dmd_base*gate_w,
        # the floor saturates (-> GAN pinned at base, no reduction) for all
        # gate_w >= gan_full/(min_ratio*dmd_base). If that threshold is tiny
        # (gan_full << dmd_base, e.g. the default gan_loss_weight=0.05 vs
        # dmd_loss_weight=1.0) the coupling is effectively inert. Surface it once
        # so it's never a SILENT no-op (per code review).
        if self.gan_gate_couple_enabled and self.is_main_process:
            _gan_full = self.gan_loss_weight * float(
                getattr(cfg, "ladd_disc_loss_weight", 1.0))
            _dmd_base = float(getattr(cfg, "dmd_loss_weight", 1.0))
            _denom = self.gan_gate_couple_min_ratio * max(_dmd_base, 1e-9)
            _gate_thresh = _gan_full / _denom if _denom > 0 else 0.0
            logging.info(
                "[GAN-GATE-COUPLE] enabled (beta=%.2f floor_frac=%.2f "
                "min_ratio=%.2f): gan_full=%.4f dmd_base=%.4f -> GAN reduces "
                "only when gate_w < %.3f (floor saturates above that). %s",
                self.gan_gate_couple_beta, self.gan_gate_couple_floor_frac,
                self.gan_gate_couple_min_ratio, _gan_full, _dmd_base, _gate_thresh,
                ("WARNING: threshold < 0.1 -> coupling is effectively INERT; "
                 "raise gan_loss_weight or lower gan_gate_couple_min_ratio."
                 if _gate_thresh < 0.1 else ""),
            )
        # ``gan_warmup_steps``: ramp the *generator-side* RpGAN-G loss
        # weight from 0 → ``gan_loss_weight`` over this many steps so
        # the gen doesn't see noise from a freshly-initialised D.
        # The D-side update runs from step 0 (it has to learn before
        # it can give G a signal). Default 500 (== z_guidance_warmup_steps).
        self.gan_warmup_steps = int(getattr(cfg, "gan_warmup_steps", 500))
        # ``gan_warmup_shape``: shape of the gen-side GAN weight ramp
        # over ``gan_warmup_steps``. ``"linear"`` (default, legacy
        # behaviour): t/T constant derivative — first non-zero step
        # delivers (1/T) × plateau weight. ``"quadratic"``: (t/T)² —
        # zero derivative at gate-open; first non-zero step delivers
        # (1/T²) × plateau, two orders of magnitude softer than
        # linear at small t. ``"cosine"``: ½(1-cos(πt/T)) — S-curve
        # with zero derivative at both ends. Recommended "quadratic"
        # to suppress fish-scale artifacts at GAN onset.
        self.gan_warmup_shape = str(
            getattr(cfg, "gan_warmup_shape", "linear")
        ).lower()
        if self.gan_warmup_shape not in ("linear", "quadratic", "cosine"):
            raise ValueError(
                "gan_warmup_shape must be one of 'linear' | 'quadratic' "
                f"| 'cosine'; got {self.gan_warmup_shape!r}."
            )
        self.gan_updates_per_step = int(
            getattr(cfg, "gan_updates_per_step", 1)
        )
        self.gan_max_grad_norm = float(
            getattr(cfg, "gan_max_grad_norm", self.max_grad_norm)
        )
        # ------------------------------------------------------------------
        # A4 ``gan_grad_target_norm`` was REMOVED 2026-08-23 (researcher
        # decision, GAN_REDESIGN.md A4/A15): the cap-only gen-side GAN
        # gradient rescale was unusable as built (no nan_to_num guard, tau
        # measured post-``gan_loss_weight``/warmup AND diluted by zero-grad
        # frames, and one extra teacher-class backward per CHUNK with no
        # cadence knob). The method, both call sites, the knob parse and all
        # five ``gan_grad_cap_*`` log keys are gone. A correct gradient
        # governor can be rebuilt if the need is demonstrated. NOT related to
        # ``dmd_grad_target_norm``, which is a different, live, DMD-side knob.
        # The A7 grad telemetry (``gan_grad_telemetry_every`` ->
        # ``train/gan_grad_norm`` / ``gan_dmd_grad_ratio`` /
        # ``gan_dmd_grad_cos``) is UNCHANGED and now reports the UNCAPPED
        # gen-side GAN gradient (previously it read post-cap).
        # ------------------------------------------------------------------
        # A5: no-grad decoder tripwire. Every ``texture_tripwire_every``
        # OUTER steps, decode a small slice of the generator's latents under
        # ``torch.no_grad()`` and run the anisotropic texture battery
        # (analysis/texture_stats.py) on the decoded pixels, logging the
        # result. Purpose: catch a critic gaming a latent-space objective
        # into decoder artefacts DURING training instead of discovering it in
        # a 60 s eval two hours later. No backward, no graph, no gradient —
        # cannot change training behaviour by construction. 0 (default) = off.
        _ttw = getattr(cfg, "texture_tripwire_every", 0)
        self.texture_tripwire_every = int(_ttw or 0)
        if self.texture_tripwire_every < 0:
            raise ValueError(
                f"texture_tripwire_every must be >= 0 (0 = off); got "
                f"{self.texture_tripwire_every}"
            )
        # Number of LATENT frames decoded per tripwire fire (the WAN VAE
        # expands 4x temporally, so 2 latent frames ~ 8 pixel frames).
        self.texture_tripwire_frames = int(
            getattr(cfg, "texture_tripwire_frames", 2) or 2
        )
        # HF band cut as a fraction of Nyquist (texture_stats convention).
        self.texture_tripwire_hf_cut = float(
            getattr(cfg, "texture_tripwire_hf_cut", 0.25)
        )
        # When true, ALSO decode the co-located GT latents and log the
        # sample/reference ratio per battery term (``battery_delta``). Costs
        # a second no-grad decode; off by default so the tripwire stays cheap.
        self.texture_tripwire_gt_ref = bool(
            getattr(cfg, "texture_tripwire_gt_ref", False)
        )
        # ------------------------------------------------------------------
        # A6: real-sample diversity telemetry. Pure logging, no behaviour
        # change: per discriminator batch, count the unique source rides, the
        # unique source windows and the repeat rate over the real slots the
        # D actually consumes. Makes "is the real pool actually diverse"
        # checkable — it currently is not (docs flag it [UNVERIFIED]).
        # A6 completion (2026-08-23): covers BOTH real-draw paths now — the
        # matched draw (inside ``_match_select``) and the POSITIONAL draw,
        # which the R3 directive ``ladd_gt_transition_match=false`` routes
        # every step through and which used to emit ZERO keys silently.
        # Positional rows carry ``gan_real_div_positional=1.0`` so the two
        # estimators can never be confused, and the telemetry dict is reset
        # at the top of ``_ladd_run_pair_mode`` so a skipped D-update can no
        # longer re-emit the previous step's counts as if they were fresh.
        # False (default) = off, byte-identical.
        self.gan_real_diversity_log = bool(
            getattr(cfg, "gan_real_diversity_log", False)
        )
        # ------------------------------------------------------------------
        # A22: held-out discriminator generalisation probe. Default OFF
        # (``disc_holdout_probe_every=0``) => byte-identical. Only the hot
        # gate is cached on ``self``; every other knob is read lazily off
        # ``self.config`` inside ``model/disc_holdout_probe.py`` (see
        # ``DEFAULTS`` there), so this block never has to grow.
        # Asks "is D MEMORISING its real training crops?" — distinct from the
        # A14 positive control ("has D learned to recognise obvious
        # corruption?") and from the fixed-route eval.
        # ------------------------------------------------------------------
        _dhp_every = int(getattr(cfg, "disc_holdout_probe_every", 0) or 0)
        if _dhp_every < 0:
            raise ValueError(
                f"disc_holdout_probe_every must be >= 0 (0 = off); got "
                f"{_dhp_every}"
            )
        self.disc_holdout_probe_every = _dhp_every
        self.r3gan_optimizer: Optional[torch.optim.Optimizer] = None
        if self.gan_enabled and self.r3gan_disc is not None:
            disc_lr = float(getattr(cfg, "gan_lr", 2e-4))
            disc_betas = tuple(
                getattr(cfg, "gan_betas", [0.0, 0.9])
            )
            disc_eps = float(getattr(cfg, "gan_eps", 1e-8))
            disc_wd = float(getattr(cfg, "gan_weight_decay", 0.0))
            disc_params = [
                p for p in self.r3gan_disc.parameters() if p.requires_grad
            ]
            if not disc_params:
                raise RuntimeError(
                    "r3gan_disc has no trainable parameters; check the "
                    "constructor."
                )
            self.r3gan_optimizer = torch.optim.AdamW(
                disc_params,
                lr=disc_lr,
                betas=disc_betas,
                eps=disc_eps,
                weight_decay=disc_wd,
            )
            if self.is_main_process:
                n_params = sum(p.numel() for p in disc_params)
                logging.info(
                    "[ActionForcing] R3GAN optimizer built: AdamW "
                    "lr=%.2e betas=%s wd=%.4f params=%.2fM "
                    "(loss_weight=%.4f, R1_gamma=%.3f, "
                    "warmup_steps=%d, gan_updates_per_step=%d)",
                    disc_lr, disc_betas, disc_wd, n_params / 1e6,
                    self.gan_loss_weight, self.gan_r1_gamma,
                    self.gan_warmup_steps,
                    self.gan_updates_per_step,
                )

        # ------------------------------------------------------------------
        # WP-PIXGAN §5.2 — the pixel critic's own optimizer.
        # Adam (NOT AdamW: the LADD path above uses AdamW, the pixel spec
        # says plain Adam — follow the spec), betas (0.0, 0.9),
        # lr=``pix_gan_lr`` (1e-5). Gated on ``gan_pixel_texture_enabled``
        # ONLY — independent of ``gan_enabled``.
        # ------------------------------------------------------------------
        self.pix_optimizer: Optional[torch.optim.Optimizer] = None
        if (self.gan_pixel_texture_enabled
                and self.pixel_texture_disc is not None):
            from model.pixel_texture_disc import PIX_GAN_BETAS, PIX_GAN_LR
            pix_lr = float(getattr(cfg, "pix_gan_lr", PIX_GAN_LR))
            pix_betas = tuple(
                float(b) for b in getattr(cfg, "pix_gan_betas", PIX_GAN_BETAS)
            )
            pix_params = [
                p for p in self.pixel_texture_disc.parameters()
                if p.requires_grad
            ]
            if not pix_params:
                raise RuntimeError(
                    "pixel_texture_disc has no trainable parameters; check "
                    "the constructor."
                )
            self.pix_optimizer = torch.optim.Adam(
                pix_params,
                lr=pix_lr,
                betas=pix_betas,
            )
            if self.is_main_process:
                _pix_n = sum(p.numel() for p in pix_params)
                # WARNING, not info: ``pix_gan_lr`` / ``pix_gan_betas`` at
                # their ONE resolution point. This is the number
                # ``_pix_optimizer_hyper`` later re-reads off param_groups
                # for the rule-4 echo, and it is a direct §12 calibration
                # input -- exactly the class that must not be decided by
                # import order. Same no-stderr-leg reason as the disc
                # build above.
                logging.warning(
                    "[ActionForcing] Pixel-texture optimizer built: Adam "
                    "lr=%.2e betas=%s params=%.2fM",
                    pix_lr, pix_betas, _pix_n / 1e6,
                )

        # GAN warmup / disc-start schedule (used by the LADD path).
        # ``gan_critic_warmup_steps``: defers the gen-side GAN gradient
        # until this outer step; D trains from step 0 (or from
        # ``gan_disc_start_step`` if non-zero). ``gan_disc_start_step``:
        # defers the entire D-side pass until this step.
        self.gan_critic_warmup_steps = int(
            getattr(cfg, "gan_critic_warmup_steps", 500)
        )
        self.gan_disc_start_step = int(
            getattr(cfg, "gan_disc_start_step", 0)
        )
        # ---------- Pixel-space perceptual losses (v13) ----------
        # LPIPS (Zhang et al., CVPR 2018) — VGG-based perceptual
        # distance between gen pixels and GT pixels. Direct anti-blur
        # signal that doesn't go through the GAN distillation chain.
        # Operates on a small frame subset per iter (memory budget).
        # ``lpips_loss_weight=0`` disables. Lazy-loaded on first use.
        self.lpips_loss_weight = float(
            getattr(cfg, "lpips_loss_weight", 0.0)
        )
        self.lpips_n_frames = int(
            getattr(cfg, "lpips_n_frames", 2)
        )
        # Pixel-space L1/MSE (the "(3)" term). Anti-streaming-drift
        # signal — direct pixel supervision that DMD doesn't provide.
        # Low weight (0.01-0.1) is the standard.
        self.pixel_recon_loss_weight = float(
            getattr(cfg, "pixel_recon_loss_weight", 0.0)
        )
        # ``mse | l1`` — L1 is more outlier-robust, MSE is smoother.
        self.pixel_recon_loss_type = str(
            getattr(cfg, "pixel_recon_loss_type", "l1")
        )
        # Optional random crop for LPIPS / pixel-recon — keeps memory
        # bounded when full-resolution decode (480×832) is too large.
        # 0 disables cropping (use full decoded resolution).
        self.lpips_crop_size = int(
            getattr(cfg, "lpips_crop_size", 0)
        )

        # ------------------------------------------------------------------
        # Online real_teacher (v14-LoRA flow training vs GT).
        # ------------------------------------------------------------------
        # Builds an AdamW over the LoRA params kept alive on
        # ``model.real_score.model`` by ``_load_real_score_with_v14_lora``
        # (which skipped ``merge_and_unload`` because the YAML flag
        # ``real_teacher_train_online`` was True). Stays None when the
        # flag is False.
        self.real_teacher_optimizer: Optional[torch.optim.Optimizer] = None
        self.real_teacher_train_online = bool(
            getattr(cfg, "real_teacher_train_online", False)
        )
        if self.real_teacher_train_online:
            rt_params = list(
                getattr(self.model, "_real_teacher_trainable_params", []) or []
            )
            if not rt_params:
                # Fallback: discover by `requires_grad`. Should never
                # trigger if the model-side load went well, but a
                # missing stash here means the LoRA wrap broke silently
                # — fail loud.
                rt_params = [
                    p for p in self.model.real_score.model.parameters()
                    if p.requires_grad
                ]
            if not rt_params:
                raise RuntimeError(
                    "real_teacher_train_online=True but real_score has "
                    "no trainable params; check "
                    "_load_real_score_with_v14_lora's online path."
                )
            rt_lr = float(getattr(cfg, "real_teacher_lr", 5.0e-05))
            rt_betas = tuple(getattr(cfg, "real_teacher_betas", [0.9, 0.999]))
            rt_eps = float(getattr(cfg, "real_teacher_eps", 1.0e-08))
            rt_wd = float(getattr(cfg, "real_teacher_weight_decay", 0.01))
            self.real_teacher_optimizer = torch.optim.AdamW(
                rt_params,
                lr=rt_lr,
                betas=rt_betas,
                eps=rt_eps,
                weight_decay=rt_wd,
            )
            self.real_teacher_max_grad_norm = float(
                getattr(cfg, "real_teacher_max_grad_norm", 1.0)
            )
            self.real_teacher_warmup_steps = int(
                getattr(cfg, "real_teacher_warmup_steps", 200)
            )
            self._real_teacher_base_lr = rt_lr
            # ``real_teacher_lr_decay_end_step`` (0 = off): linearly decay the
            # real-teacher LR from base at step 0 to 0 at this step, then FREEZE
            # the teacher entirely (no optimizer step, no anchor, no EMA) for the
            # rest of the run. The 'froz' experiment: train the teacher hard
            # early under the low-noise stabilisers, then stop, to test whether
            # late teacher training helps or hurts. Set real_teacher_warmup_steps
            # =0 alongside it so the LR truly starts at base ("from its starting
            # value"). 0 => byte-identical to the standard warmup-then-hold path.
            self.real_teacher_lr_decay_end_step = int(
                getattr(cfg, "real_teacher_lr_decay_end_step", 0)
            )
            if self.real_teacher_lr_decay_end_step < 0:
                raise ValueError(
                    "real_teacher_lr_decay_end_step must be >= 0, got "
                    f"{self.real_teacher_lr_decay_end_step}"
                )
            # Light v14 anchor: after each online-teacher LoRA update, pull the
            # LoRA params a small fraction toward 0 (= toward the merged-v14
            # base). WEIGHT-space (not output-space): v14 is causal with a
            # different clean_x contract, so matching its OUTPUTS would fight
            # the bidir adaptation; shrinking the delta only keeps the teacher
            # NEAR v14 without forcing its behaviour. lr-independent, so it bites
            # even at small rt_lr (unlike AdamW weight_decay). 0.0 => off
            # (byte-identical). e.g. 0.003 ~= 0.3% pull-to-v14 per update.
            self.real_teacher_anchor_lambda = float(
                getattr(cfg, "real_teacher_anchor_lambda", 0.0)
            )
            if self.real_teacher_anchor_lambda < 0.0:
                raise ValueError(
                    "real_teacher_anchor_lambda must be >= 0, got "
                    f"{self.real_teacher_anchor_lambda}"
                )
            if self.is_main_process:
                n_params = sum(p.numel() for p in rt_params)
                logging.info(
                    "[ActionForcing] real_teacher optimizer built: "
                    "AdamW lr=%.2e betas=%s wd=%.4f params=%.2fM "
                    "(warmup_steps=%d, max_grad_norm=%.2f)",
                    rt_lr, rt_betas, rt_wd, n_params / 1e6,
                    self.real_teacher_warmup_steps,
                    self.real_teacher_max_grad_norm,
                )
        else:
            self.real_teacher_max_grad_norm = 1.0
            self.real_teacher_warmup_steps = 0
            self._real_teacher_base_lr = 0.0
            self.real_teacher_lr_decay_end_step = 0

        # ``fake_score_ema_weight`` (0.0 = off, current behavior; e.g.
        # 0.95 = engage). After each generator optimizer.step(), the
        # fake_score's params get EMA-pulled toward the generator's
        # newly-updated params:
        #     ψ ← w · ψ + (1 - w) · θ
        # This lets the fake_score track p_gen with much fewer dedicated
        # diffusion-loss updates, so dfake_gen_update_ratio can be
        # dropped from 5 to 1 or 2 with no quality loss (paper §3.3).
        self.fake_score_ema_weight = float(
            getattr(cfg, "fake_score_ema_weight", 0.0)
        )
        if not (0.0 <= self.fake_score_ema_weight < 1.0):
            raise ValueError(
                f"fake_score_ema_weight must be in [0, 1); got "
                f"{self.fake_score_ema_weight!r}."
            )
        if self.is_main_process and self.fake_score_ema_weight > 0.0:
            logging.info(
                "[ActionForcing] fake_score_ema_weight=%.4f",
                self.fake_score_ema_weight,
            )

    def _maybe_ema_fake_score_from_generator(self) -> None:
        """Flash-DMD §3.3 EMA: after each generator optimizer.step(),
        pull the fake_score's params toward the generator's params
        with weight ``self.fake_score_ema_weight``. No-op when the
        weight is 0.

        Iterates by ``named_parameters()`` and matches by name +
        shape. On the first call we audit the match coverage and log
        a one-shot summary so silent param-structure drift (rename,
        shape mismatch, accidental architectural divergence) is
        loud — without that audit a fake_score param could quietly
        decouple from the generator forever.
        """
        if self.fake_score_ema_weight <= 0.0:
            return
        # Both modules may be DDP-wrapped; unwrap to the inner module.
        gen_inner = (
            self.generator_ddp.module
            if getattr(self, "generator_ddp", None) is not None
            else self.model.generator.model
        )
        fake_inner = (
            self.fake_score_ddp.module
            if getattr(self, "fake_score_ddp", None) is not None
            else self.model.fake_score.model
        )
        gen_params = dict(gen_inner.named_parameters())
        w = self.fake_score_ema_weight

        # One-shot startup audit. Counts matched / skipped on the
        # first invocation only; logs the summary so operators see
        # whether 100% of fake_score params are EMA-tracked. If the
        # match rate is < 100%, the SKIPPED ones drift untouched and
        # that's almost always wrong (architectural drift, not
        # intentional).
        if not getattr(self, "_fake_ema_audit_done", False):
            matched = 0
            skipped = 0
            skipped_examples = []
            for name, p_fake in fake_inner.named_parameters():
                p_gen = gen_params.get(name)
                if p_gen is None or p_gen.shape != p_fake.shape:
                    skipped += 1
                    if len(skipped_examples) < 5:
                        reason = (
                            "missing-in-generator" if p_gen is None
                            else f"shape-mismatch ({tuple(p_gen.shape)} vs {tuple(p_fake.shape)})"
                        )
                        skipped_examples.append(f"{name}: {reason}")
                else:
                    matched += 1
            total = matched + skipped
            if self.is_main_process:
                logging.info(
                    "[ActionForcing] fake_score EMA audit: matched=%d "
                    "skipped=%d total=%d (%.1f%% coverage). EMA "
                    "weight=%.4f.",
                    matched, skipped, total,
                    100.0 * matched / max(1, total),
                    w,
                )
                if skipped > 0:
                    logging.warning(
                        "[ActionForcing] fake_score EMA: %d params NOT "
                        "EMA-tracked. First %d examples: %s",
                        skipped, len(skipped_examples), skipped_examples,
                    )
            self._fake_ema_audit_done = True

        with torch.no_grad():
            for name, p_fake in fake_inner.named_parameters():
                p_gen = gen_params.get(name)
                if p_gen is None or p_gen.shape != p_fake.shape:
                    continue
                p_fake.data.mul_(w).add_(p_gen.data, alpha=1.0 - w)

    def _inner_dit_for_rope(self):
        """Return the bare DiT module under the (possibly DDP / LoRA)
        generator so ``infinity_rope.install()`` can clear any
        per-attention-module rotated-prefix state on its
        ``CausalWanSelfAttention`` children. The class-level monkey
        patch is global, so this argument is only used for state
        clearing — passing the wrapper is fine since
        ``_clear_module_state`` walks ``module.modules()``.
        """
        gen = self.model.generator.model
        return gen

    # ------------------------------------------------------------------
    # Training loop (CF-parity).
    # ------------------------------------------------------------------
    def train(self) -> None:
        cfg = self.config
        # Memory audit: pre-training snapshot. Captures static param +
        # buffer footprint AFTER all heads / DDP wraps / optimizers
        # are built but BEFORE any step's activation memory has hit
        # the allocator. Call again after step 1 + step 5 below
        # (gated by ``memory_audit_enabled`` so prod runs aren't
        # spammed). Default off; smoke turns on.
        if bool(getattr(cfg, "memory_audit_enabled", False)):
            self._dump_memory_audit("pre_train")
            self._dump_module_inventory()
        max_steps = int(getattr(cfg, "max_steps", 10000))
        ckpt_interval = int(getattr(cfg, "checkpoint_interval", 500))
        log_interval = int(getattr(cfg, "log_interval", 10))
        dfake_gen_update_ratio = int(getattr(cfg, "dfake_gen_update_ratio", 1))
        num_training_frames = int(getattr(cfg, "num_training_frames", 21))
        num_frame_per_block = int(getattr(cfg, "num_frame_per_block", 3))
        rollout_frames_raw = getattr(cfg, "rollout_frames", None)
        rollout_frames = (
            int(rollout_frames_raw) if rollout_frames_raw is not None
            else num_training_frames
        )
        # MAE-extension code path was removed; rollouts no longer grow
        # beyond ``rollout_frames``. ``max_total_rollout_frames`` stays
        # as a name for downstream slicing but equals ``rollout_frames``.
        max_rolls_per_ride = int(
            getattr(cfg, "max_rolls_per_ride", 0)
        )
        max_total_rollout_frames = rollout_frames

        # dmd_context shifts the student's rollout window forward by
        # ``cf`` frames in the ride (= ride frames [cf, cf +
        # max_total_rollout_frames)) so the leading ``cf`` ride frames
        # become the KV-cache seed prefill. ``cf`` is the model's
        # single source of truth (= ``self.model.dmd_context_clean_frames``,
        # default 9 = KV-cache size).
        cf_dmdctx = int(getattr(self.model, "dmd_context_clean_frames", 0))
        # CF-parity #6: periodic memory hygiene.
        # ``empty_cache_interval``: how often to release PyTorch's caching
        # allocator pool back to the driver (CF: every 20 steps).
        # ``gc_interval``: how often to run a full Python ``gc.collect``
        # (CF: every ``gc_interval``, default 100). Both reduce peak
        # memory and fragmentation on long runs without affecting losses.
        empty_cache_interval = int(getattr(cfg, "empty_cache_interval", 20))
        gc_interval = int(getattr(cfg, "gc_interval", 100))

        if self.is_main_process:
            logging.info(
                "[ActionForcing] Starting CF-style DMD training: max_steps=%d "
                "dfake_gen_update_ratio=%d num_training_frames=%d "
                "rollout_frames=%d (gradient on last %d frames; "
                "warmup_frames=%d) max_rolls_per_ride=%d "
                "(=> max_total_rollout_frames=%d when ride is long enough "
                "and MAE stays under threshold)",
                max_steps, dfake_gen_update_ratio, num_training_frames,
                rollout_frames, num_training_frames,
                rollout_frames - num_training_frames,
                max_rolls_per_ride, max_total_rollout_frames,
            )
            logging.info(
                "[ActionForcing] memory hygiene: empty_cache every %d steps, "
                "gc.collect every %d steps",
                empty_cache_interval, gc_interval,
            )
            logging.info(
                "[ActionForcing] EMA: weight=%.4f start_step=%d (lazy-init at "
                "start_step; EMA shadow not created until then)",
                float(getattr(cfg, "ema_weight", 0.0) or 0.0),
                int(getattr(cfg, "ema_start_step", 0) or 0),
            )

        previous_time: Optional[float] = None
        self._epoch = 0
        self._ride_iter = self._fresh_ride_iter(self._epoch)

        # Apply the schedule's step-0 value before the first sequence
        # opens (no-op when no schedule configured).
        self._apply_attn_size_if_changed()

        while self.step < max_steps:
            # CF-parity #5: put the whole DMD module in eval mode at the
            # start of each iter to suppress any latent randomness
            # (Dropout / BN running stats). Our DiT has no Dropout/BN
            # so this is a no-op in math, but we match CF's
            # ``self.model.eval()`` at ``Causal-Forcing/trainer/
            # distillation.py:230`` to keep the code path identical.
            self.model.eval()
            # Cheap per-iter check for an attn-size schedule transition.
            # Only does work on the iter that crosses a threshold (rest
            # are dict-lookup + int-compare). On a transition this also
            # closes the open streaming sequence so the next sequence
            # re-allocates the KV cache at the new size.
            self._apply_attn_size_if_changed()
            # Decide whether this iter trains the generator or the critic.
            train_generator = (self.step % dfake_gen_update_ratio == 0)

            if train_generator:
                generator_log_dict = self._fwdbwd_one_step(
                    train_generator=True,
                    rollout_frames=rollout_frames,
                    max_total_rollout_frames=max_total_rollout_frames,
                    cf_dmdctx=cf_dmdctx,
                )

            # Always run the critic step (CF parity).
            critic_log_dict = self._fwdbwd_one_step(
                train_generator=False,
                rollout_frames=rollout_frames,
                max_total_rollout_frames=max_total_rollout_frames,
                cf_dmdctx=cf_dmdctx,
            )

            # Step optimizers.
            if train_generator:
                self._all_reduce_extra_trainable_grads()
                gen_grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.optimizer.param_groups[0]["params"]
                     if p.grad is not None],
                    max_norm=self.max_grad_norm,
                )
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                # Phase LoRA: step the K-1 remaining per-rung
                # optimizers (rung_0 IS self.optimizer above; the
                # rest each own only their adapter's LoRA params).
                # Each adapter is clipped INDEPENDENTLY at the same
                # max_grad_norm as rung_0: per-adapter granularity is
                # the right unit (a spike on one rung — most likely
                # rung_flash, which alone absorbs the full
                # gan_loss_weight adversarial gradient — must not
                # shrink another rung's legitimate update, which a
                # single global clip over all lora params would do).
                for _opt in getattr(self, "phase_lora_optimizers", [])[1:]:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in _opt.param_groups[0]["params"]
                         if p.grad is not None],
                        max_norm=getattr(
                            self, "phase_lora_max_grad_norm",
                            self.max_grad_norm,
                        ),
                    )
                    _opt.step()
                    _opt.zero_grad(set_to_none=True)
                self._maybe_update_generator_ema()
            else:
                gen_grad_norm = torch.tensor(0.0, device=self.device)

            fake_grad_norm_val = 0.0
            if self.fake_optimizer is not None:
                fake_params_with_grad = [
                    p for p in self.fake_optimizer.param_groups[0]["params"]
                    if p.grad is not None
                ]
                if fake_params_with_grad:
                    fgn = torch.nn.utils.clip_grad_norm_(
                        fake_params_with_grad,
                        max_norm=self.fake_max_grad_norm,
                    )
                    fake_grad_norm_val = (
                        float(fgn.item()) if torch.is_tensor(fgn) else float(fgn)
                    )
                    self.fake_optimizer.step()
                self.fake_optimizer.zero_grad(set_to_none=True)

            # v27B forward-noiser optimizer step.
            forward_noiser_grad_norm_val = 0.0
            if getattr(self, "forward_noiser_optimizer", None) is not None:
                fn_params_with_grad = [
                    p for p in
                    self.forward_noiser_optimizer.param_groups[0]["params"]
                    if p.grad is not None
                ]
                if fn_params_with_grad:
                    fngn = torch.nn.utils.clip_grad_norm_(
                        fn_params_with_grad,
                        max_norm=self.forward_noiser_max_grad_norm,
                    )
                    forward_noiser_grad_norm_val = (
                        float(fngn.item()) if torch.is_tensor(fngn)
                        else float(fngn)
                    )
                    self.forward_noiser_optimizer.step()
                self.forward_noiser_optimizer.zero_grad(set_to_none=True)
                # Surface grad norm to wandb via out dict if available.
                if isinstance(generator_log_dict, dict):
                    generator_log_dict["forward_noiser_grad_norm"] = (
                        forward_noiser_grad_norm_val
                    )

            # CARN-cycle: reverse noiser G optimizer step (mirror F).
            reverse_noiser_grad_norm_val = 0.0
            if getattr(self, "reverse_noiser_optimizer", None) is not None:
                rn_params_with_grad = [
                    p for p in
                    self.reverse_noiser_optimizer.param_groups[0]["params"]
                    if p.grad is not None
                ]
                if rn_params_with_grad:
                    rngn = torch.nn.utils.clip_grad_norm_(
                        rn_params_with_grad,
                        max_norm=self.reverse_noiser_max_grad_norm,
                    )
                    reverse_noiser_grad_norm_val = (
                        float(rngn.item()) if torch.is_tensor(rngn)
                        else float(rngn)
                    )
                    self.reverse_noiser_optimizer.step()
                self.reverse_noiser_optimizer.zero_grad(set_to_none=True)
                if isinstance(generator_log_dict, dict):
                    generator_log_dict["reverse_noiser_grad_norm"] = (
                        reverse_noiser_grad_norm_val
                    )

            # ----- State-probe optimizer step (LoRA-side aux loss) -----
            # Mirrors the fake/real_teacher pattern: clip → step → zero.
            # Probe gets gradient ONLY when the LoRA aux pass populated
            # state_preds AND the gen-step backward fired (= gen iter).
            # On critic-only iters the probe params have no grad, so
            # the params_with_grad check elides the step naturally.
            state_probe_grad_norm_val = 0.0
            if self.state_probe_optimizer is not None:
                sp_params_with_grad = [
                    p for p in self.state_probe_optimizer.param_groups[0]["params"]
                    if p.grad is not None
                ]
                if sp_params_with_grad:
                    sgn = torch.nn.utils.clip_grad_norm_(
                        sp_params_with_grad,
                        max_norm=self.state_probe_max_grad_norm,
                    )
                    state_probe_grad_norm_val = (
                        float(sgn.item()) if torch.is_tensor(sgn) else float(sgn)
                    )
                    self.state_probe_optimizer.step()
                self.state_probe_optimizer.zero_grad(set_to_none=True)

            # Flash-DMD §3.3: EMA fake_score toward generator AFTER
            # the fake_optimizer step. Order matters: applying Adam's
            # accumulated momentum/variance to fake's pre-step params
            # is the intended math; the EMA pull then becomes the
            # FINAL mutation of fake's params for the iter, with no
            # stale-momentum interaction. No-op when
            # ``fake_score_ema_weight == 0`` (default).
            if (
                train_generator
                and self.fake_score_ema_weight > 0.0
            ):
                self._maybe_ema_fake_score_from_generator()

            # ----- Online real_teacher: clip + step + zero -----
            real_teacher_grad_norm_val = 0.0
            # Hard gate: skip the LoRA optimizer step entirely until
            # ``aux_teacher_start_step``. The aux pass is also gated
            # in ``compute_generator_loss_streaming`` (so no aux grad
            # accumulates pre-start), but defensively zero any stray
            # gradient here too in case some other path populates it.
            _aux_start_step = int(
                getattr(self.config, "aux_teacher_start_step", 0)
            )
            _aux_step_open = self.step >= _aux_start_step
            if (
                self.real_teacher_optimizer is not None
                and _aux_step_open
            ):
                rt_params_with_grad = [
                    p for p in self.real_teacher_optimizer.param_groups[0]["params"]
                    if p.grad is not None
                ]
                if rt_params_with_grad:
                    # LR schedule: linear warmup over real_teacher_warmup_steps,
                    # then (if real_teacher_lr_decay_end_step>0) a linear decay
                    # to 0 by that step, after which the teacher is FROZEN -- no
                    # optimizer step, no anchor, no EMA -- a truly fixed target
                    # (the 'froz' experiment). decay_end=0 => original behaviour.
                    _rt_decay = self.real_teacher_lr_decay_end_step
                    _rt_frozen = (_rt_decay > 0 and self.step >= _rt_decay)
                    if _rt_frozen:
                        for pg in self.real_teacher_optimizer.param_groups:
                            pg["lr"] = 0.0
                        # (grad_norm left at its prior value; no update logged)
                    else:
                        if _rt_decay > 0:
                            # base * warmup_factor * linear-decay_factor
                            _wf = 1.0
                            if (
                                self.real_teacher_warmup_steps > 0
                                and self.step < self.real_teacher_warmup_steps
                            ):
                                _wf = (self.step + 1) / self.real_teacher_warmup_steps
                            _df = max(0.0, (_rt_decay - self.step) / float(_rt_decay))
                            for pg in self.real_teacher_optimizer.param_groups:
                                pg["lr"] = self._real_teacher_base_lr * _wf * _df
                        elif (
                            self.real_teacher_warmup_steps > 0
                            and self.step < self.real_teacher_warmup_steps
                        ):
                            warm_factor = (
                                (self.step + 1) / self.real_teacher_warmup_steps
                            )
                            for pg in self.real_teacher_optimizer.param_groups:
                                pg["lr"] = self._real_teacher_base_lr * warm_factor
                        elif self.real_teacher_warmup_steps > 0:
                            # Restore base LR once warmup completes (no-op
                            # after first post-warmup step but harmless).
                            for pg in self.real_teacher_optimizer.param_groups:
                                pg["lr"] = self._real_teacher_base_lr
                        rtgn = torch.nn.utils.clip_grad_norm_(
                            rt_params_with_grad,
                            max_norm=self.real_teacher_max_grad_norm,
                        )
                        real_teacher_grad_norm_val = (
                            float(rtgn.item()) if torch.is_tensor(rtgn) else float(rtgn)
                        )
                        # NaN/inf guard: clip_grad_norm_ does NOT sanitize a
                        # non-finite grad (it returns a non-finite norm). Stepping
                        # would poison the AdamW moments AND the EMA shadow
                        # irrecoverably (the teacher target is dead for the rest
                        # of the run). Skip step+anchor+EMA on a non-finite norm;
                        # grads are zeroed below. Finite path is byte-identical.
                        if torch.isfinite(torch.as_tensor(real_teacher_grad_norm_val)):
                            self.real_teacher_optimizer.step()
                            # Light v14 anchor: shrink the LoRA delta toward 0 (= v14
                            # base) by a small fraction each update. Fires AFTER
                            # optim.step so it acts on the post-step weights. Default
                            # lambda 0 => skipped (byte-identical). Not run once frozen
                            # (above) so the frozen teacher is genuinely fixed.
                            if self.real_teacher_anchor_lambda > 0.0:
                                _keep = 1.0 - self.real_teacher_anchor_lambda
                                with torch.no_grad():
                                    for _p in self.real_teacher_optimizer.param_groups[0]["params"]:
                                        _p.mul_(_keep)
                            # Target-network EMA pull on the LoRA adapter
                            # (no-op when real_score_ema_weight == 0). Fires
                            # AFTER optim.step on every LoRA update so the
                            # EMA tracks the post-step weights, including
                            # the just-clipped gradient's effect.
                            self.model.ema_update_real_score_lora()
                        elif self.is_main_process:
                            logging.warning(
                                "[ActionForcing] real_teacher grad norm non-finite "
                                "(%.3e) at step %d -> skipped teacher step+anchor+EMA",
                                real_teacher_grad_norm_val, self.step,
                            )
                self.real_teacher_optimizer.zero_grad(set_to_none=True)
            elif self.real_teacher_optimizer is not None:
                # Below start_step (or aux gate closed for any other
                # reason): zero any stray gradient so the next iter
                # starts clean. No optimizer step taken.
                self.real_teacher_optimizer.zero_grad(set_to_none=True)

            self.step += 1

            # Memory audit at steps 1 and 5 (steady-state activation
            # memory after the warm-up forward + grad path is fully
            # established). Gated by ``memory_audit_enabled`` so prod
            # logs don't get spammed.
            if (
                bool(getattr(cfg, "memory_audit_enabled", False))
                and self.step in (1, 5)
            ):
                self._dump_memory_audit(f"after_step_{self.step}")
            # Per-step boundary breakdown: dump every step so we can see
            # which named boundary's ``delta_alloc`` is non-zero across
            # steps (= the leak's residence). Stays gated by
            # ``memory_audit_enabled``. Smoke turns this on.
            if bool(getattr(cfg, "memory_audit_enabled", False)):
                self._dump_step_mem_breakdown(f"step_{self.step}")
                if (
                    torch.cuda.is_available()
                    and getattr(self, "is_main_process", True)
                ):
                    _alloc_gb = (
                        torch.cuda.memory_allocated() / (1024 ** 3)
                    )
                    _reserved_gb = (
                        torch.cuda.memory_reserved() / (1024 ** 3)
                    )
                    logging.info(
                        "[mem-end step=%d] alloc=%6.2f GB  reserved=%6.2f GB",
                        self.step, _alloc_gb, _reserved_gb,
                    )

            # CUDA allocation-history snapshot (definitive per-callsite/per-
            # tensor attribution of the transient peak). ``cuda_mem_history_
            # dump_step=N`` starts recording at step N-2 (so a couple of
            # GAN-active disc transients land in the ring buffer) and dumps a
            # pickle at step N. Analyse offline: the device_traces timeline
            # shows the peak-segment allocations + their Python stacks even
            # after they're freed. rank-0 only; off by default (N=0).
            _snap_step = int(getattr(cfg, "cuda_mem_history_dump_step", 0))
            if (
                _snap_step > 0
                and torch.cuda.is_available()
                and getattr(self, "is_main_process", True)
            ):
                if self.step == max(1, _snap_step - 2):
                    try:
                        torch.cuda.memory._record_memory_history(
                            max_entries=300000)
                        logging.info(
                            "[mem-snapshot] history recording ON at step %d",
                            self.step)
                    except Exception as _e:
                        logging.warning("[mem-snapshot] record failed: %s", _e)
                if self.step == _snap_step:
                    try:
                        _p = (
                            f"logs/smoke/cuda_mem_snapshot_step{self.step}"
                            f"_rank0.pickle")
                        torch.cuda.memory._dump_snapshot(_p)
                        torch.cuda.memory._record_memory_history(enabled=None)
                        logging.info("[mem-snapshot] dumped %s", _p)
                    except Exception as _e:
                        logging.warning("[mem-snapshot] dump failed: %s", _e)

            # Mark wandb log as DUE if the cadence boundary just
            # crossed. With ``dfake_gen_update_ratio>1`` the boundary
            # often lands on a critic-only iter (no fresh gen data),
            # so we just stash a sticky bit and emit on the next iter
            # that has both gen and critic data — meeting the spec
            # "if it is not the right step then it needs to happen in
            # the subsequent step." Without this, e.g. log_interval=10
            # with dfake=2 silently skips every gen log because
            # boundary post-step parity always lands on a critic iter.
            if self.step % log_interval == 0:
                self._wandb_log_due = True

            # Mark video sample as DUE on the same cadence-boundary
            # principle. The ``_video_sample_due`` callsite (inside
            # the gen rollout) already runs only on gen iters, so the
            # bit is consumed there; we just need to set it whenever
            # the boundary crossed but no chunk got stashed this iter
            # (== ``_pending_video_latents is None`` at this point,
            # since stash happens during the gen-step BEFORE step++).
            if (
                self.sample_interval > 0
                and (self.step % self.sample_interval) == 0
                and self._pending_video_latents is None
            ):
                self._video_sample_due_bit = True

            # Logging — emit only when we have BOTH fresh gen and
            # critic data this iter (i.e. ``train_generator`` was True
            # and the gen step actually produced a dict). On a
            # critic-only iter we leave ``_wandb_log_due`` set so the
            # next gen iter emits the deferred log.
            gen_ready = (train_generator and generator_log_dict is not None)
            critic_ready = (critic_log_dict is not None)
            if (
                self.is_main_process
                and self._wandb_log_due
                and gen_ready
                and critic_ready
            ):
                msg_parts = [f"step={self.step}/{max_steps}"]
                msg_parts.append(
                    f"gen_loss={generator_log_dict.get('generator_loss', 0.0):.4f}"
                )
                if "dmdtrain_gradient_norm" in generator_log_dict:
                    msg_parts.append(
                        f"dmd_grad={generator_log_dict['dmdtrain_gradient_norm']:.4f}"
                    )
                # Streaming mode: the standalone critic iter is a no-op
                # (its dict has no critic_loss) — the REAL diffusion loss
                # comes from the gtfix path merged into the gen-step
                # dict. Without this fallback the line printed a
                # misleading 0.0000 for weeks.
                _cl_disp = critic_log_dict.get(
                    "critic_loss",
                    generator_log_dict.get("critic_loss", 0.0),
                )
                if not _cl_disp:
                    _cl_disp = generator_log_dict.get("critic_loss", 0.0)
                msg_parts.append(f"critic_loss={float(_cl_disp):.4f}")
                msg_parts.append(
                    f"gen_grad_norm={float(gen_grad_norm.item()) if torch.is_tensor(gen_grad_norm) else float(gen_grad_norm):.4f}"
                )
                msg_parts.append(f"fake_grad_norm={fake_grad_norm_val:.4f}")
                # GAN diagnostics: surface the 3 most-watched signals
                # so terminal trace alone is enough to spot disc/critic
                # health (full set on wandb under train/*).
                #   d_real   = disc score on real (should rise > 0)
                #   d_fake   = disc score on fake-detached (D-update)
                #   c_corr   = critic↔disc value Pearson (should → 1)
                _d_real = generator_log_dict.get("train/r3gan_d_real")
                if _d_real is not None:
                    msg_parts.append(f"d_real={float(_d_real):+.3f}")
                _d_fake = generator_log_dict.get("train/r3gan_d_fake_detached")
                if _d_fake is not None:
                    msg_parts.append(f"d_fake={float(_d_fake):+.3f}")
                _c_corr = generator_log_dict.get("train/critic_disc_corr")
                if _c_corr is not None:
                    msg_parts.append(f"c_corr={float(_c_corr):+.3f}")
                # WP-SURROGATE: the two keys that make c_corr DECIDABLE
                # from the terminal trace alone. Added because the
                # 2026-08-24 backbone comparison produced a clean sign
                # flip (sam2 +0.44 vs dinov2 -0.52 at matched step 51)
                # that could NOT be diagnosed live: c_corr alone cannot
                # distinguish "the critic is anti-fitting a good teacher"
                # from "the teacher's values are noise, so any
                # correlation with them is arbitrary". The first is a
                # surrogate bug; the second is a teacher that is not
                # separating real from fake -- opposite fixes.
                #   t_dloss = the PRETRAINED teacher's own NS-logistic
                #             D-loss (2*ln2 = 1.386 at init; FLAT means
                #             the teacher never learned, which makes
                #             c_corr meaningless rather than alarming)
                #   c_vloss = the critic's value-distillation MSE (if
                #             this FALLS while c_corr goes negative, the
                #             two are inconsistent and something is
                #             genuinely wrong)
                _t_dloss = generator_log_dict.get(
                    "train/surrogate_sam2_d_loss")
                if _t_dloss is not None:
                    msg_parts.append(f"t_dloss={float(_t_dloss):.4f}")
                _c_vloss = generator_log_dict.get(
                    "train/critic_value_loss")
                if _c_vloss is not None:
                    msg_parts.append(f"c_vloss={float(_c_vloss):.4g}")
                # MANIFOLD GATE measurement (docs/DMD_MANIFOLD_GATE.md).
                # These two ARE the experiment: dmd_err is DMD's own
                # error |x0 - pred_real| (teacher competence here), and
                # align = cos(-grad, GT - x0) is whether the DMD update
                # actually points at the manifold. align going NEGATIVE
                # as dmd_err grows is the thesis; the crossover is the
                # gate threshold. Both on the step line so the crossover
                # is readable from the terminal trace alone.
                # Read off the MODEL, not the log dict: dmd_log_dict is
                # wandb-only and never reaches stderr, so a dict lookup
                # here silently finds nothing forever. getattr on the
                # model is the path already proven by
                # _last_dmd_mae_gate_weight below.
                for _a, _lbl, _fmt in (
                    ("_last_dmd_align", "align", "+.3f"),
                    ("_last_dmd_err", "dmd_err", ".4f"),
                    ("_last_dmd_gt_dist", "gt_dist", ".4f"),
                    ("_last_dmd_err_gate_weight", "eg_w", ".3f"),
                    ("_last_dmd_align_manifold", "align_mf", "+.3f"),
                    # FINGERPRINT gate (docs/DMD_FINGERPRINT_PROBE.md) —
                    # the signal that SUPERSEDES eg_w. fp_m is the
                    # calibrated manifold-localisation score; fp_w* are
                    # the per-frame weights actually applied to the DMD
                    # term; fp_share is the mean weight as a fraction of
                    # the unattenuated 1.0, i.e. how much of DMD survives
                    # the gate. All start None, so an unavailable one is
                    # an ABSENT key, never a forgeable 0.0 (which would
                    # read as "DMD fully gated off").
                    ("_last_dmd_fp_m", "fp_m", ".3f"),
                    ("_last_dmd_fp_gate_w_mean", "fp_w", ".3f"),
                    ("_last_dmd_fp_gate_w_min", "fp_wmin", ".3f"),
                    ("_last_dmd_fp_gate_w_max", "fp_wmax", ".3f"),
                    ("_last_dmd_fp_gate_share", "fp_share", ".3f"),
                ):
                    _v = getattr(self.model, _a, None)
                    if _v is not None:
                        msg_parts.append(f"{_lbl}={float(_v):{_fmt}}")
                _t_dreal = generator_log_dict.get(
                    "train/surrogate_sam2_d_real")
                _t_dfake = generator_log_dict.get(
                    "train/surrogate_sam2_d_fake")
                if _t_dreal is not None and _t_dfake is not None:
                    # Separation, not the raw scores: this is the single
                    # number that says whether the teacher can tell real
                    # from fake at all. ~0 => teacher is blind.
                    msg_parts.append(
                        f"t_sep={float(_t_dreal) - float(_t_dfake):+.3f}")
                # MAE diagnostics: surface baseline_last_chunk_mae +
                # baseline_avg_rollout_mae to stdout so we can detect
                # student collapse from the iter trace alone (the
                # primary "is the student degrading" signal).
                _mae_last = generator_log_dict.get("baseline_last_chunk_mae")
                if _mae_last is not None:
                    msg_parts.append(f"mae_last={float(_mae_last):.4f}")
                _mae_avg = generator_log_dict.get("baseline_avg_rollout_mae")
                if _mae_avg is not None:
                    msg_parts.append(f"mae_avg={float(_mae_avg):.4f}")
                # Peak GPU memory this step (resets the high-water
                # mark each iter). Useful for OOM-margin diagnostics.
                try:
                    if torch.cuda.is_available():
                        peak_gb = (
                            torch.cuda.max_memory_allocated() / (1024 ** 3)
                        )
                        msg_parts.append(f"peak_gb={peak_gb:.2f}")
                        torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass
                logging.info("[ActionForcing] " + " ".join(msg_parts))
                if (
                    _HAS_WANDB
                    and getattr(self, "wandb_enabled", False)
                ):
                    log_payload: Dict[str, Any] = {}
                    # Skip private/internal keys (e.g. ``_aux_teacher_tensors``
                    # — a dict of bf16 graph tensors stashed for the LoRA
                    # action/probe losses, NOT a metric). Leaving it in
                    # poisons the payload: wandb can't serialize the nested
                    # bf16 and drops the WHOLE step, so every gen/critic curve
                    # silently vanishes. Also drop multi-element tensors here.
                    for k, v in generator_log_dict.items():
                        if str(k).startswith("_"):
                            continue
                        if torch.is_tensor(v):
                            if v.numel() != 1:
                                continue
                            v = float(v.item())
                        log_payload[f"gen/{k}"] = v
                    for k, v in critic_log_dict.items():
                        if str(k).startswith("_"):
                            continue
                        if torch.is_tensor(v):
                            if v.numel() != 1:
                                continue
                            v = float(v.item())
                        log_payload[f"critic/{k}"] = v
                    log_payload["gen/grad_norm"] = (
                        float(gen_grad_norm.item())
                        if torch.is_tensor(gen_grad_norm)
                        else float(gen_grad_norm)
                    )
                    log_payload["critic/grad_norm"] = fake_grad_norm_val
                    # Online real_teacher (LoRA) gets the same treatment
                    # as the gen/critic optimizers: log the post-clip
                    # grad_norm and the warmup-aware effective LR. Both
                    # are always-defined locals (init'd to 0.0 / read
                    # off param_groups[0]["lr"] above) so the keys
                    # appear every step regardless of whether the
                    # optimizer fired this iter.
                    log_payload["critic/real_teacher_grad_norm"] = real_teacher_grad_norm_val
                    if self.real_teacher_optimizer is not None:
                        log_payload["critic/real_teacher_lr"] = float(
                            self.real_teacher_optimizer.param_groups[0]["lr"]
                        )
                    # EMA-LoRA drift diagnostic (None when
                    # real_score_ema_weight==0 or before first LoRA
                    # optim step). Picks up whatever the most recent
                    # EMA-update computed.
                    _ema_rel_l2 = getattr(
                        self.model, "_real_score_ema_rel_l2", None,
                    )
                    if _ema_rel_l2 is not None:
                        log_payload["critic/real_score_ema_rel_l2"] = float(_ema_rel_l2)
                    log_payload["critic/state_probe_grad_norm"] = (
                        state_probe_grad_norm_val
                    )
                    if self.state_probe_optimizer is not None:
                        log_payload["critic/state_probe_lr"] = float(
                            self.state_probe_optimizer.param_groups[0]["lr"]
                        )
                    if previous_time is not None:
                        log_payload["per_iter_time"] = time.time() - previous_time
                    try:
                        # Final guard: wandb chokes on bf16/fp16 tensors
                        # ("Got unsupported ScalarType BFloat16") and the
                        # exception drops the WHOLE payload for the step.
                        # Coerce scalar tensors to float and DROP anything
                        # that isn't a plain number/bool (containers, strings,
                        # multi-element tensors) so one bad value can never
                        # nuke the curves again.
                        clean_payload: Dict[str, Any] = {}
                        for _k, _v in log_payload.items():
                            if torch.is_tensor(_v):
                                if _v.numel() == 1:
                                    clean_payload[_k] = float(_v.item())
                                continue
                            if isinstance(_v, (int, float, bool)):
                                clean_payload[_k] = _v
                        wandb.log(clean_payload, step=self.step)
                    except Exception as e:
                        logging.warning("wandb.log failed: %s", e)
                previous_time = time.time()
                self._wandb_log_due = False

            # ---- flag-gated debug INPUT-DUMP grid video ----
            # (debug_dump_scorer_inputs_every) Decodes the DMD scorer +
            # GAN disc input stashes written earlier this step. No-op
            # when the stashes are empty; can never raise.
            if self.is_main_process:
                try:
                    self._maybe_dump_debug_inputs()
                except Exception as _dbg_exc:
                    logging.warning(
                        "[DBG-INPUTS] dump failed at step=%d: %s",
                        int(self.step), _dbg_exc)

            if (
                self.is_main_process
                and self._pending_video_latents is not None
            ):
                self._log_pred_image_video(
                    self._pending_video_latents, int(self.step),
                )
                self._pending_video_latents = None

                # Rolling: also emit the FULL ride rollout accumulated
                # across rolls (the 21f pred_image window only shows the
                # latest slice once rides run deep). cap_frames=False —
                # the whole rollout is the point.
                _acc = getattr(self, "_rollout_video_acc", None)
                if _acc is not None and int(_acc.shape[1]) > 21:
                    try:
                        self._log_pred_image_video(
                            _acc.to(device=self.device, dtype=self.dtype),
                            int(self.step),
                            name="pred_image_rollout",
                            caption_suffix=(
                                f"full ride rollout "
                                f"({int(_acc.shape[1])} latent frames)"
                            ),
                            cap_frames=False,
                        )
                    except Exception as _exc:
                        logging.warning(
                            "[ActionForcing] pred_image_rollout decode "
                            "failed (len=%d): %s",
                            int(_acc.shape[1]), _exc,
                        )

                # DMD scorer + clean_x diagnostic videos. ``pred_real``
                # / ``pred_fake`` are the scorers' denoised x0 estimates
                # on the SAME noisy student input — divergence between
                # them is what DMD is asking the student to close;
                # ``clean_x_fake`` / ``clean_x_real`` are what each
                # scorer was conditioned on. Same VAE decode + ffmpeg
                # encode path as ``pred_image``; best-effort, never
                # raises.
                eval_latents = getattr(self, "_pending_dmd_eval_latents", None)
                if isinstance(eval_latents, dict) and eval_latents:
                    _t = eval_latents.get("dmd_timestep")
                    _aug = eval_latents.get("clean_x_aug_t")
                    _ctx = eval_latents.get("dmd_context")
                    _suffix_real = (
                        f"t={_t} aug_t={_aug} ctx={_ctx}"
                        if _t is not None
                        else ""
                    )
                    # ``pred_real`` is the DMD-path teacher's x0 (frozen
                    # merged-v14 in dual-teacher mode; the LoRA in
                    # single-teacher mode). ``pred_real_lora`` is the
                    # aux-pass LoRA teacher's x0 — only present when
                    # dual-teacher is on (frozen pass + aux pass both
                    # populate the stash). Operators get a side-by-
                    # side comparison: the frozen oracle (DMD's
                    # gradient source) vs the moving-target online
                    # LoRA (aux pass's training target).
                    _aux_t = eval_latents.get("aux_teacher_timestep")
                    _aux_was_gt = eval_latents.get("aux_teacher_input_was_gt")
                    _suffix_aux = (
                        f"t={_aux_t} input={'GT' if _aux_was_gt else 'student'}"
                        if _aux_t is not None
                        else ""
                    )
                    for _key, _name, _cap in (
                        ("pred_real", "pred_real", _suffix_real or ""),
                        ("pred_real_lora", "pred_real_lora", _suffix_aux or ""),
                        ("pred_fake", "pred_fake", _suffix_real or ""),
                        ("clean_x_fake", "clean_x_fake", "fake_score conditioning"),
                        ("clean_x_real", "clean_x_real", f"real_score conditioning ({_ctx})"),
                        # v27B: aux-teacher TF context + noisy_input source
                        ("clean_x_aux", "clean_x_aux",
                         "aux_teacher TF context (post forward-noiser/alt-head override)"),
                        ("aux_noise_base", "aux_noise_base",
                         "aux_teacher noise_base (= causal_AR_GT when fn/alt active, else gt_target)"),
                    ):
                        _t_lat = eval_latents.get(_key)
                        if _t_lat is None or not torch.is_tensor(_t_lat):
                            continue
                        # Overlay enabled on ``clean_x_real`` only. The
                        # consolidated ``_draw_action_overlay`` draws
                        # both the action z-value bars (from
                        # ``clean_z_actions``) and the per-frame
                        # zarr_lat / motion-chunk text (from
                        # ``index_overlay``) in a single pass; both
                        # are gated on action_overlay being non-None,
                        # so passing the clean_z tensor turns the
                        # whole overlay on.
                        _clean_z_overlay = eval_latents.get(
                            "clean_z_actions"
                        )
                        _overlay = (
                            _clean_z_overlay
                            if _key == "clean_x_real"
                            and torch.is_tensor(_clean_z_overlay)
                            else None
                        )
                        # Per-frame zarr-latent + motion.npy chunk
                        # annotations on clean_x_real ONLY (other views
                        # are pred_*/clean_x_fake which don't trace back
                        # to a specific ride window). The model stashes
                        # the absolute zarr-latent index of clean_x_real's
                        # first frame plus the ride's motion-chunk offset
                        # in ``compute_generator_loss_streaming``.
                        _index_overlay = None
                        if _key == "clean_x_real":
                            _zarr_lo = eval_latents.get("clean_x_real_zarr_lat_lo")
                            _mco = eval_latents.get("clean_x_real_motion_chunk_offset")
                            _ovl_npb = eval_latents.get("clean_x_real_npb")
                            _zarr_name = eval_latents.get(
                                "clean_x_real_zarr_name", ""
                            )
                            if (
                                _zarr_lo is not None
                                and _mco is not None
                                and _ovl_npb is not None
                            ):
                                _index_overlay = (
                                    int(_zarr_lo),
                                    int(_mco),
                                    int(_ovl_npb),
                                    str(_zarr_name),
                                )
                        try:
                            self._log_pred_image_video(
                                _t_lat.to(torch.float32),
                                int(self.step),
                                name=_name,
                                caption_suffix=_cap,
                                action_overlay=_overlay,
                                index_overlay=_index_overlay,
                            )
                        except Exception as _exc:
                            logging.warning(
                                "[ActionForcing] DMD eval video '%s' "
                                "failed at step=%d: %s",
                                _name, int(self.step), _exc,
                            )
                self._pending_dmd_eval_latents = None

            # Checkpoint.
            if self.step > 0 and self.step % ckpt_interval == 0:
                if self.is_main_process:
                    self._save_checkpoint()
                if self.world_size > 1:
                    dist.barrier()

            # CF-parity #6: periodic memory hygiene. Release PyTorch's
            # caching allocator pool back to the driver every
            # ``empty_cache_interval`` steps (CF: 20) and run a full
            # Python ``gc.collect`` every ``gc_interval`` steps (CF: 100).
            # Both are cheap; the empty_cache call returns memory to
            # the driver but PyTorch will re-grow the pool on the next
            # alloc.
            if (
                empty_cache_interval > 0
                and self.step % empty_cache_interval == 0
            ):
                torch.cuda.empty_cache()
            if gc_interval > 0 and self.step % gc_interval == 0:
                gc.collect()
                if self.is_main_process:
                    logging.debug("[ActionForcing] gc.collect at step=%d", self.step)

        if self.is_main_process:
            logging.info("[ActionForcing] Training complete (step=%d).", self.step)
            self._save_checkpoint()

    # ------------------------------------------------------------------
    # Checkpoint save/resume — extend the parent to persist auxiliary
    # modules it does not own. The parent already saves the generator,
    # fake scorer, EMA, CARN, and GAN discriminator/optimizer. Re-appending
    # GAN state here used to force a redundant read and rewrite of the full
    # multi-GB checkpoint at every save.
    # ------------------------------------------------------------------
    def _save_checkpoint(self) -> None:
        eval_path_template = getattr(
            self.config, "eval_checkpoint_path", None,
        )
        if self.is_main_process and eval_path_template:
            eval_path = Path(
                str(eval_path_template).format(step=int(self.step))
            ).expanduser()
            eval_path.parent.mkdir(parents=True, exist_ok=True)
            gen_module = (
                self.generator_ddp.module
                if self.generator_ddp is not None
                else self.model.generator.model
            )
            eval_state = {
                "step": self.step,
                "generator": gen_module.state_dict(),
                "config_name": os.path.basename(self.config_path),
            }
            if self.model.action_projection is not None:
                eval_state["action_projection"] = (
                    self.model.action_projection.state_dict()
                )
            if getattr(self.model, "action_token_projection", None) is not None:
                eval_state["action_token_projection"] = (
                    self.model.action_token_projection.state_dict()
                )
            if (
                bool(getattr(self.config, "eval_checkpoint_include_ema", False))
                and self.generator_ema is not None
            ):
                eval_state["generator_ema"] = self.generator_ema.state_dict()
            torch.save(eval_state, eval_path)
            logging.info(
                "[ActionForcing] Saved minimal evaluation checkpoint: %s",
                eval_path,
            )

        if not bool(getattr(self.config, "save_full_checkpoint", True)):
            if self.is_main_process:
                logging.info(
                    "[ActionForcing] save_full_checkpoint=false; skipped full "
                    "resume-state checkpoint."
                )
            return

        super()._save_checkpoint()
        if not self.is_main_process:
            return
        # Fast-path for the common DMD/GAN recipe: all of its state was
        # already written by the parent.
        if not (
            self.action_critic_loss_active
            or self.real_teacher_train_online
            or self.state_probe_aux_active
        ):
            return
        path = self._checkpoint_path(self.step)
        if not os.path.exists(path):
            return
        try:
            state = torch.load(path, map_location="cpu")
        except Exception as exc:
            logging.warning(
                "[ActionForcing] Could not re-open checkpoint for "
                "auxiliary append: %s. Skipping.", exc,
            )
            return
        appended = []
        if self.action_critic_loss_active:
            ac = self.model.action_critic
            ac_module = ac.module if isinstance(ac, DDP) else ac
            if ac_module is not None:
                state["action_critic"] = ac_module.state_dict()
                appended.append("action_critic")
            if self.critic_optimizer is not None:
                state["critic_optimizer"] = self.critic_optimizer.state_dict()
                appended.append("critic_optimizer")
        if self.state_probe_aux_active:
            sp = self.model.state_probe
            sp_module = sp.module if isinstance(sp, DDP) else sp
            if sp_module is not None:
                state["state_probe"] = sp_module.state_dict()
                appended.append("state_probe")
            if self.state_probe_optimizer is not None:
                state["state_probe_optimizer"] = (
                    self.state_probe_optimizer.state_dict()
                )
                appended.append("state_probe_optimizer")
        if self.real_teacher_train_online:
            # FAIL-LOUD on save: silently dropping the LoRA state from
            # the checkpoint pairs with the resume path's silent
            # restart-from-warm-init to erase hours of online teacher
            # training without operator notice. Better to crash the
            # save and require investigation than to leave a
            # checkpoint that quietly discards progress on the next
            # resume. (The action critic retains fail-soft behavior because
            # it is bootstrapped from a fixed checkpoint; only the
            # online-trained LoRA has this asymmetric risk profile.)
            try:
                from peft import get_peft_model_state_dict
                rs_inner = self.model.real_score.model
                if isinstance(rs_inner, DDP):
                    rs_inner = rs_inner.module
                state["real_teacher_lora"] = get_peft_model_state_dict(rs_inner)
                appended.append("real_teacher_lora")
            except Exception as exc:
                raise RuntimeError(
                    "[ActionForcing] real_teacher LoRA save failed: "
                    f"{exc!r}. Refusing to silently drop trained LoRA "
                    "state from the checkpoint — the next auto-resume "
                    "would rewind the teacher to v14 warm-start without "
                    "any operator-visible signal. Investigate the save "
                    "failure (most likely a peft version mismatch or a "
                    "DDP-wrap-shape issue on real_score.model) and rerun."
                ) from exc
            if self.real_teacher_optimizer is not None:
                state["real_teacher_optimizer"] = (
                    self.real_teacher_optimizer.state_dict()
                )
                appended.append("real_teacher_optimizer")
        if not appended:
            return
        torch.save(state, path)
        logging.info(
            "[ActionForcing] Appended state to %s: %s",
            path, ", ".join(appended),
        )

    def _maybe_resume(self) -> None:
        super()._maybe_resume()
        if not (
            self.action_critic_loss_active
            or self.gan_enabled
            # WP-PIXGAN: the pixel critic is gated on
            # ``gan_pixel_texture_enabled``, NOT on ``gan_enabled`` (the two
            # critics are independent). Without this clause the pixel restore
            # below would never run and the critic would silently re-init on
            # every resume — the exact trap this chunk exists to close.
            or getattr(self, "gan_pixel_texture_enabled", False)
            # WP-SURROGATE: same trap, same fix, added on review (B3's own
            # consumption wiring never trains the critic without the pixel
            # disc present — see ``_maybe_run_surrogate_distillation``'s
            # graceful no-teacher skip — so ``gan_pixel_texture_enabled``
            # above already covers every configuration that has real state
            # to lose. Listed explicitly anyway: that safety currently
            # depends on an invariant enforced nowhere else, and defense in
            # depth here costs one line.
            or getattr(self, "surrogate_critic_enabled", False)
            or self.real_teacher_train_online
            or self.state_probe_aux_active
        ):
            return
        if not bool(getattr(self.config, "auto_resume", False)):
            return
        ckpts = sorted(self.log_dir.glob("phase1_step*.pt"))
        if not ckpts:
            return
        path = ckpts[-1]
        try:
            state = torch.load(path, map_location="cpu")
        except Exception as exc:
            if self.is_main_process:
                logging.warning(
                    "[ActionForcing] Could not load checkpoint for "
                    "aux/GAN resume: %s", exc,
                )
            return
        if self.action_critic_loss_active:
            ac = self.model.action_critic
            ac_module = ac.module if isinstance(ac, DDP) else ac
            if ac_module is not None and "action_critic" in state:
                ac_missing, ac_unexpected = ac_module.load_state_dict(
                    state["action_critic"], strict=False,
                )
                if self.is_main_process:
                    logging.info(
                        "resume: action_critic missing=%d unexpected=%d",
                        len(ac_missing), len(ac_unexpected),
                    )
            if self.critic_optimizer is not None and "critic_optimizer" in state:
                try:
                    self.critic_optimizer.load_state_dict(state["critic_optimizer"])
                    if self.is_main_process:
                        logging.info("resume: critic_optimizer state restored")
                except Exception as exc:
                    if self.is_main_process:
                        logging.warning(
                            "resume: critic_optimizer load failed: %s. "
                            "Starting critic optim from fresh state.", exc,
                        )
        if self.state_probe_aux_active:
            sp = self.model.state_probe
            sp_module = sp.module if isinstance(sp, DDP) else sp
            if sp_module is not None and "state_probe" in state:
                sp_missing, sp_unexpected = sp_module.load_state_dict(
                    state["state_probe"], strict=False,
                )
                if self.is_main_process:
                    logging.info(
                        "resume: state_probe missing=%d unexpected=%d",
                        len(sp_missing), len(sp_unexpected),
                    )
            if (
                self.state_probe_optimizer is not None
                and "state_probe_optimizer" in state
            ):
                try:
                    self.state_probe_optimizer.load_state_dict(
                        state["state_probe_optimizer"]
                    )
                    if self.is_main_process:
                        logging.info("resume: state_probe_optimizer state restored")
                except Exception as exc:
                    if self.is_main_process:
                        logging.warning(
                            "resume: state_probe_optimizer load failed: %s. "
                            "Starting state_probe optim from fresh state.", exc,
                        )
        if self.gan_enabled and self.r3gan_disc is not None:
            disc_module = (
                self.r3gan_disc_ddp.module if self.r3gan_disc_ddp is not None
                else self.r3gan_disc
            )
            if "r3gan_discriminator" in state:
                # Shape-tolerant load (2026-08-20): strict=False does NOT
                # forgive SIZE mismatches (e.g. a ckpt trained with a
                # different ladd_cmap_dim / wavelet band count). Drop
                # shape-mismatched keys so the matching 95% of the disc still
                # warm-starts and only the divergent heads re-init fresh.
                _dsd = state["r3gan_discriminator"]
                _own = disc_module.state_dict()
                _drop = [k for k, v in _dsd.items()
                         if k in _own and _own[k].shape != v.shape]
                for k in _drop:
                    _dsd.pop(k)
                d_missing, d_unexpected = disc_module.load_state_dict(
                    _dsd, strict=False,
                )
                if self.is_main_process:
                    logging.info(
                        "resume: r3gan_discriminator missing=%d unexpected=%d "
                        "shape_dropped=%d %s",
                        len(d_missing), len(d_unexpected), len(_drop),
                        _drop[:4],
                    )
            if self.r3gan_optimizer is not None and "r3gan_optimizer" in state:
                try:
                    self.r3gan_optimizer.load_state_dict(state["r3gan_optimizer"])
                    if self.is_main_process:
                        logging.info("resume: r3gan_optimizer state restored")
                except Exception as exc:
                    if self.is_main_process:
                        logging.warning(
                            "resume: r3gan_optimizer load failed: %s. "
                            "Starting r3gan optim from fresh state.", exc,
                        )
        # ------------------------------------------------------------------
        # WP-PIXGAN — pixel-texture critic + its Adam. FAIL LOUD.
        #
        # Written by ``trainer/causal_rolling_staircase_train.py`` (grep
        # ``pixel_texture_disc`` in the checkpoint save block); keys
        # "pixel_texture_disc" / "pix_optimizer" on both sides.
        #
        # DO NOT "fix" this back to the tolerant LADD behaviour twenty lines
        # above. That path deliberately drops shape-mismatched CCM keys and
        # only logs, because a LADD disc whose cmap_dim changed should still
        # warm-start its matching 95%. The pixel critic has a FROZEN §4
        # architecture — there is no legitimate shape drift — and a critic
        # that silently re-initialises mid-run restarts its warmup and
        # corrupts the adversarial signal *invisibly*: D collapses back to
        # ln 2, the G-term goes to noise, and nothing in the logs says so.
        # A missing key or an unclean load is therefore a HARD ERROR.
        #
        # If you are deliberately turning the pixel arm on from a checkpoint
        # that predates it, start a fresh log_dir (or auto_resume=false) —
        # do not weaken this check.
        # ------------------------------------------------------------------
        if (getattr(self, "gan_pixel_texture_enabled", False)
                and getattr(self, "pixel_texture_disc", None) is not None):
            pix_module = (
                self.pixel_texture_disc_ddp.module
                if getattr(self, "pixel_texture_disc_ddp", None) is not None
                else self.pixel_texture_disc
            )
            if "pixel_texture_disc" not in state:
                raise RuntimeError(
                    "resume: gan_pixel_texture_enabled=true but checkpoint "
                    f"{path} has no 'pixel_texture_disc' key. Continuing "
                    "would silently re-initialise the pixel critic, "
                    "restarting its warmup and corrupting the adversarial "
                    "signal with no visible symptom. Either resume from a "
                    "checkpoint written with the pixel arm on, or start a "
                    "fresh run (auto_resume=false / new log_dir)."
                )
            try:
                pix_missing, pix_unexpected = pix_module.load_state_dict(
                    state["pixel_texture_disc"], strict=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    "resume: 'pixel_texture_disc' state_dict from "
                    f"{path} did not load cleanly: {exc}. The §4 pixel-critic "
                    "architecture is frozen, so this means the checkpoint "
                    "does not belong to this critic. Refusing to continue "
                    "with a partially-initialised critic."
                ) from exc
            if pix_missing or pix_unexpected:
                # strict=True already raises on these; belt-and-braces so a
                # future torch that softens strict= cannot reopen the trap.
                raise RuntimeError(
                    "resume: 'pixel_texture_disc' loaded with "
                    f"missing={list(pix_missing)} "
                    f"unexpected={list(pix_unexpected)} from {path}. "
                    "A partially-restored critic is a silently corrupted "
                    "adversarial signal; refusing to continue."
                )
            if self.is_main_process:
                logging.info(
                    "resume: pixel_texture_disc restored (strict) from %s",
                    path,
                )
            if getattr(self, "pix_optimizer", None) is not None:
                if "pix_optimizer" not in state:
                    raise RuntimeError(
                        "resume: gan_pixel_texture_enabled=true but "
                        f"checkpoint {path} has no 'pix_optimizer' key. "
                        "Adam betas (0.0, 0.9) carry the critic's second "
                        "moment; dropping it silently changes the effective "
                        "critic learning rate mid-run. Refusing to continue."
                    )
                try:
                    self.pix_optimizer.load_state_dict(state["pix_optimizer"])
                except Exception as exc:
                    raise RuntimeError(
                        "resume: 'pix_optimizer' state from "
                        f"{path} did not load: {exc}. Unlike the r3gan "
                        "optimizer above, this is NOT recoverable by "
                        "restarting from fresh optimizer state — that would "
                        "silently reset the critic's Adam moments mid-run."
                    ) from exc
                if self.is_main_process:
                    logging.info("resume: pix_optimizer state restored")
        # ------------------------------------------------------------------
        # WP-SURROGATE (B3) — latent surrogate critic + its optimizer.
        # FAIL LOUD, same rationale as the pixel critic above (spec
        # WP_SURROGATE.md §4.4): a silently re-initialised surrogate hands
        # the generator a near-zero gradient that reads as "GAN term
        # present and quiet" on every dashboard. Missing key while enabled
        # is a HARD ERROR; resume from a pre-surrogate checkpoint means a
        # fresh log_dir / auto_resume=false, not a weakened check.
        # ------------------------------------------------------------------
        if (bool(getattr(self, "surrogate_critic_enabled", False))
                and getattr(self, "latent_texture_critic", None) is not None):
            _sur_mod = (
                self.latent_texture_critic.module
                if hasattr(self.latent_texture_critic, "module")
                else self.latent_texture_critic
            )
            if "latent_texture_critic" not in state:
                raise RuntimeError(
                    "resume: surrogate_critic_enabled=true but checkpoint "
                    f"{path} has no 'latent_texture_critic' key. Continuing "
                    "would silently re-initialise the surrogate critic — the "
                    "generator would consume a zero-init critic whose "
                    "gradient is noise, with no visible symptom. Resume from "
                    "a checkpoint written with the surrogate on, or start a "
                    "fresh run (auto_resume=false / new log_dir)."
                )
            try:
                sur_missing, sur_unexpected = _sur_mod.load_state_dict(
                    state["latent_texture_critic"], strict=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    "resume: 'latent_texture_critic' state_dict from "
                    f"{path} did not load cleanly: {exc}. Refusing to "
                    "continue with a partially-initialised surrogate critic."
                ) from exc
            if sur_missing or sur_unexpected:
                raise RuntimeError(
                    "resume: 'latent_texture_critic' loaded with "
                    f"missing={list(sur_missing)} "
                    f"unexpected={list(sur_unexpected)} from {path}. "
                    "Refusing to continue."
                )
            if self.is_main_process:
                logging.info(
                    "resume: latent_texture_critic restored (strict) from %s",
                    path,
                )
            if getattr(self, "latent_critic_optimizer", None) is not None:
                if "latent_critic_optimizer" not in state:
                    raise RuntimeError(
                        "resume: surrogate_critic_enabled=true but "
                        f"checkpoint {path} has no 'latent_critic_optimizer' "
                        "key. Dropping the Adam moments silently changes the "
                        "effective critic learning rate mid-run. Refusing to "
                        "continue."
                    )
                try:
                    self.latent_critic_optimizer.load_state_dict(
                        state["latent_critic_optimizer"],
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "resume: 'latent_critic_optimizer' state from "
                        f"{path} did not load: {exc}. Refusing to restart "
                        "from fresh optimizer state mid-run."
                    ) from exc
                if self.is_main_process:
                    logging.info(
                        "resume: latent_critic_optimizer state restored",
                    )
        # ForwardNoiser (CARN) + its optimizer.
        _fn = getattr(self.model, "forward_noiser", None)
        if _fn is not None and "forward_noiser" in state:
            _fn_mod = _fn.module if hasattr(_fn, "module") else _fn
            fn_missing, fn_unexpected = _fn_mod.load_state_dict(
                state["forward_noiser"], strict=False,
            )
            if self.is_main_process:
                logging.info(
                    "resume: forward_noiser missing=%d unexpected=%d",
                    len(fn_missing), len(fn_unexpected),
                )
            _fn_opt = getattr(self, "forward_noiser_optimizer", None)
            if _fn_opt is not None and "forward_noiser_optimizer" in state:
                try:
                    _fn_opt.load_state_dict(state["forward_noiser_optimizer"])
                    if self.is_main_process:
                        logging.info("resume: forward_noiser_optimizer restored")
                except Exception as exc:
                    if self.is_main_process:
                        logging.warning(
                            "resume: forward_noiser_optimizer load failed: %s. "
                            "Starting FN optim from fresh state.", exc,
                        )
        # CARN-cycle: restore the reverse noiser G + its optimizer.
        _rn = getattr(self.model, "reverse_noiser", None)
        if _rn is not None and "reverse_noiser" in state:
            _rn_mod = _rn.module if hasattr(_rn, "module") else _rn
            rn_missing, rn_unexpected = _rn_mod.load_state_dict(
                state["reverse_noiser"], strict=False,
            )
            if self.is_main_process:
                logging.info(
                    "resume: reverse_noiser missing=%d unexpected=%d",
                    len(rn_missing), len(rn_unexpected),
                )
            _rn_opt = getattr(self, "reverse_noiser_optimizer", None)
            if _rn_opt is not None and "reverse_noiser_optimizer" in state:
                try:
                    _rn_opt.load_state_dict(state["reverse_noiser_optimizer"])
                    if self.is_main_process:
                        logging.info("resume: reverse_noiser_optimizer restored")
                except Exception as exc:
                    if self.is_main_process:
                        logging.warning(
                            "resume: reverse_noiser_optimizer load failed: %s. "
                            "Starting reverse-noiser optim from fresh state.",
                            exc,
                        )
        if self.real_teacher_train_online:
            # FAIL-LOUD on resume: silently rolling the LoRA back to
            # v14 warm-start because the load errored — or because the
            # checkpoint just doesn't contain it — would erase every
            # step of online teacher training while the student
            # KEPT its optimizer state. That asymmetric rewind is
            # almost never what the operator wants. To force a fresh
            # teacher start, delete the checkpoint or set
            # auto_resume=False (which short-circuits this whole
            # block above).
            if "real_teacher_lora" not in state:
                raise RuntimeError(
                    "[ActionForcing] real_teacher_train_online=True but "
                    f"the resume checkpoint at {path} does not contain "
                    "'real_teacher_lora' state. This means either the "
                    "previous run wasn't training the teacher (expected: "
                    "the checkpoint predates this feature), or a save "
                    "failed silently in an earlier step. To proceed from "
                    "v14 warm-start instead of resuming, delete the "
                    "checkpoint or set auto_resume=False."
                )
            try:
                from peft import set_peft_model_state_dict
                rs_inner = self.model.real_score.model
                if isinstance(rs_inner, DDP):
                    rs_inner = rs_inner.module
                set_peft_model_state_dict(rs_inner, state["real_teacher_lora"])
                if self.is_main_process:
                    logging.info("resume: real_teacher LoRA state restored")
            except Exception as exc:
                raise RuntimeError(
                    "[ActionForcing] real_teacher LoRA load failed: "
                    f"{exc!r}. Refusing to silently restart from v14 "
                    "warm-start — would discard every step of online "
                    "teacher training. Investigate the checkpoint "
                    "(likely a peft / config mismatch with the "
                    "current LoRA setup) and rerun."
                ) from exc
            if self.real_teacher_optimizer is not None:
                if "real_teacher_optimizer" not in state:
                    raise RuntimeError(
                        "[ActionForcing] real_teacher_optimizer is built "
                        "but resume checkpoint lacks 'real_teacher_optimizer' "
                        "state. Same reasoning as the LoRA-state check above: "
                        "silently re-warming-up Adam moments + the LR "
                        "schedule on a checkpoint past the warmup phase "
                        "is almost never wanted."
                    )
                try:
                    self.real_teacher_optimizer.load_state_dict(
                        state["real_teacher_optimizer"]
                    )
                    if self.is_main_process:
                        logging.info(
                            "resume: real_teacher_optimizer state restored"
                        )
                except Exception as exc:
                    raise RuntimeError(
                        "[ActionForcing] real_teacher_optimizer load "
                        f"failed: {exc!r}. Refusing to silently start "
                        "from fresh optimizer state."
                    ) from exc

    # ------------------------------------------------------------------
    # Auxiliary action-critic losses (ported from
    # ``trainer.causal_diffusion_teacher_train._compute_action_critic_losses``,
    # adapted for the action-forcing rollout shape: ``pred_x0`` is the
    # student's last ``num_training_frames``-frame baseline window
    # rather than a per-frame flow-matching pred, so we use a single
    # broadcast timestep — the rollout's rung-exit timestep — instead
    # of the per-frame random timestep that the teacher trainer has).
    # ------------------------------------------------------------------
    def _weighted_z_mse(self, pred_z: torch.Tensor, target_z: torch.Tensor) -> torch.Tensor:
        w = torch.ones(pred_z.shape[-1], device=pred_z.device, dtype=pred_z.dtype)
        for dim_idx in self.action_critic_dims:
            w[dim_idx] = 2.0
        return (w * (pred_z - target_z) ** 2).mean()

    def _slice_actions_for_critic(self, actions: torch.Tensor) -> torch.Tensor:
        """Slice the action stream's last dim down to ``action_critic_dims``.

        The data loader (``_load_ride_tensors``) pre-slices ``z_actions``
        by ``self.action_dims`` before storing them on the ride dict, so
        ``actions[..., -1]`` may already be the critic-target slice.
        We therefore have three valid configurations to handle:

          1. ``action_dims is None`` AND ``actions.shape[-1] == 8``
             (or whatever the SS-VAE z-dim is): no pre-slicing happened,
             slice with ``action_critic_dims`` directly.
          2. ``action_dims is not None`` AND ``actions.shape[-1] ==
             len(action_dims)``: pre-slicing happened. Map
             ``action_critic_dims`` (absolute indices into the original
             8-dim z space) into RELATIVE indices within the
             ``action_dims``-sliced stream, then slice. If
             ``action_dims == action_critic_dims`` this is just
             ``[0..len(action_critic_dims)-1]``, i.e. a no-op slice.
          3. Anything else: fall back to the absolute-index slice and
             let the index op surface the bug as an out-of-bounds error.
        """
        last = int(actions.shape[-1])
        crit_dims = list(self.action_critic_dims)
        if self.action_dims is None:
            return actions[..., crit_dims]
        if last == len(self.action_dims):
            try:
                rel = [int(self.action_dims.index(int(d))) for d in crit_dims]
            except ValueError as exc:
                raise RuntimeError(
                    f"action_critic_dims={crit_dims} contains an index "
                    f"not present in action_dims={list(self.action_dims)}. "
                    "The action critic z-target stream is pre-sliced by "
                    "action_dims in the data loader; action_critic_dims "
                    "must be a subset of action_dims."
                ) from exc
            return actions[..., rel]
        return actions[..., crit_dims]

    def _compute_action_critic_losses(
        self,
        pred_x0: torch.Tensor,
        target_action_z: torch.Tensor,
        chunk_t: torch.Tensor,
        current_step: int,
    ):
        """Train the action critic and return generator z-guidance loss.

        Args:
            pred_x0: ``[B, F, C, H, W]`` student rollout pred (graph-
                carrying; we detach internally for the critic-update
                branch and use the live tensor for the gen-side
                guidance branch).
            target_action_z: ``[B, F, K]`` per-frame commanded action
                (already sliced to ``self.action_critic_dims``).
            chunk_t: ``[B, n_chunks]`` per-chunk diffusion timestep
                (broadcast from the rollout's rung-exit timestep).
            current_step: training step (for guidance warmup ramp).

        Returns:
            ``(generator_action_loss, logs, teacher_z_8d)`` —
            ``generator_action_loss`` is graph-carrying and should
            be added to the DMD loss before backward; ``logs`` is a
            flat ``str -> float`` dict for wandb.
        """
        # DDP-wrap-aware handles. ``critic_for_update`` goes through
        # the DDP wrapper so backward triggers all-reduce on critic
        # gradients — required for distributed correctness on the
        # critic optim. ``critic_for_guidance`` is the UNWRAPPED
        # ActionCritic; we call it directly in the gen-side branch
        # so DDP isn't involved (we set ``requires_grad_(False)``
        # there anyway, and the gradient we care about is on the
        # generator's pred_x0, which flows through the outer
        # generator-DDP backward).
        critic_for_update = self.model.action_critic
        critic_for_guidance = (
            self.model.action_critic.module
            if isinstance(self.model.action_critic, DDP)
            else self.model.action_critic
        )
        chunk_frames = int(self.config.num_frame_per_block)
        B = pred_x0.shape[0]
        n_chunks = pred_x0.shape[1] // chunk_frames

        chunk_actions = _chunk_actions(target_action_z, chunk_frames)[:, :n_chunks]
        # If the student rollout went off-window relative to the
        # action stream we fed the loader, we slice both to the
        # shorter min — defensive, the trainer already aligns them.
        if chunk_actions.shape[1] < n_chunks:
            n_chunks = chunk_actions.shape[1]

        zero = torch.tensor(0.0, device=pred_x0.device, dtype=pred_x0.dtype)

        teacher_z_8d = self._compute_teacher_z_per_slot(pred_x0.detach())
        # Cross-rank coordination: if ANY rank's teacher failed
        # (cotracker / VAE / ss_vae error on this rank's pred_x0), ALL
        # ranks must skip aux training together. Otherwise the failing
        # ranks early-return without firing the action_critic_ddp
        # forward (a DDP-wrapped collective) while the succeeding ranks
        # do — NCCL collective op count mismatches across ranks → hang.
        # This is the action-critic analog of the slide-loop's
        # any_crossed MAX-reduce.
        local_unavailable = 1 if teacher_z_8d is None else 0
        if dist.is_initialized() and dist.get_world_size() > 1:
            flag_t = torch.tensor(
                [local_unavailable], device=pred_x0.device, dtype=torch.long,
            )
            dist.all_reduce(flag_t, op=dist.ReduceOp.MAX)
            any_unavailable = bool(int(flag_t.item()))
        else:
            any_unavailable = bool(local_unavailable)
        if any_unavailable:
            # Skip aux loss on every rank this iter; teacher_unavailable
            # log fires only on the rank(s) that actually failed.
            logs = {
                "train/critic_z_loss": 0.0,
                "train/critic_loss": 0.0,
                "train/critic_z2_mse": 0.0,
                "train/critic_z7_mse": 0.0,
                "train/gen_z_loss": 0.0,
                "train/gen_action_loss": 0.0,
                "train/teacher_z2_mean": 0.0,
                "train/teacher_z7_mean": 0.0,
                "train/z_guidance_scale": 0.0,
                "train/teacher_unavailable": float(local_unavailable),
            }
            return zero, logs, None
        teacher_z_8d = teacher_z_8d[:, :n_chunks]
        chunk_actions = chunk_actions[:, :n_chunks]
        chunk_t = chunk_t[:, :n_chunks]

        pred_x0_detached = pred_x0.detach()

        # The action critic is built in fp32 (its internal Conv3d /
        # Linear blocks default to fp32 weights, see
        # ``ActionForcingDMD._build_aux_heads``). The DMD generator
        # produces ``pred_x0`` in bf16, so a direct call into the
        # critic raises ``RuntimeError: mat1 and mat2 must have the
        # same dtype``. The v14 trainer hides this by wrapping the
        # critic call in ``autocast(bf16)``; we do the same for both
        # the critic-update and the gen-side z-guidance path so the
        # critic transparently picks up the same bf16 mat-mul kernels
        # as the rest of training. Grad flow:
        #   * critic-update path: backward stays on the critic's
        #     fp32 params (autocast unscales + casts back); optim
        #     step happens in fp32.
        #   * gen-side path: grad flows through ``pred_x0`` (bf16)
        #     into the generator's DDP backward; the critic's
        #     params are frozen for this branch.
        # --- Multi-step critic training (independent optim loop) ---
        critic_z_loss = zero
        critic_loss_k = zero
        pred_z = None
        for _k in range(max(1, self.critic_updates_per_step)):
            self.critic_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=True,
            ):
                pred_z = critic_for_update(
                    pred_x0_detached, chunk_t, chunk_actions,
                )
                pred_z = pred_z[:, :n_chunks]
                critic_z_loss = self._weighted_z_mse(
                    pred_z, teacher_z_8d.to(pred_z.dtype),
                )
                critic_loss_k = (
                    self.action_critic_z_loss_weight * critic_z_loss
                )
            critic_loss_k.backward()
            if self.critic_max_grad_norm is not None and self.critic_max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    critic_for_guidance.parameters(), self.critic_max_grad_norm,
                )
            self.critic_optimizer.step()

        # --- Generator z-guidance (frozen critic, grad through pred) ---
        warmup_start = self.warmup_steps_for_guidance
        if current_step < warmup_start:
            guidance_scale = 0.0
        elif self.z_guidance_warmup_steps > 0:
            ramp = min(
                1.0,
                (current_step - warmup_start) / max(1, self.z_guidance_warmup_steps),
            )
            guidance_scale = ramp * self.generator_action_z_guidance_weight
        else:
            guidance_scale = self.generator_action_z_guidance_weight

        if guidance_scale > 0:
            critic_for_guidance.requires_grad_(False)
            try:
                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=True,
                ):
                    gen_pred_z = critic_for_guidance(
                        pred_x0, chunk_t, chunk_actions,
                    )
                    gen_pred_z = gen_pred_z[:, :n_chunks]
                    gen_z2z7 = gen_pred_z[:, :, self.action_critic_dims]
                    target_z2z7 = chunk_actions
                    gen_z_loss = F.mse_loss(
                        gen_z2z7, target_z2z7.to(gen_z2z7.dtype),
                    )
                    generator_action_loss = guidance_scale * gen_z_loss
            finally:
                critic_for_guidance.requires_grad_(True)
        else:
            gen_z_loss = zero
            generator_action_loss = zero

        with torch.no_grad():
            z2_idx, z7_idx = (
                self.action_critic_dims[0],
                self.action_critic_dims[1] if len(self.action_critic_dims) > 1
                else self.action_critic_dims[0],
            )
            z2_mse = (
                F.mse_loss(pred_z[:, :, z2_idx].float(), teacher_z_8d[:, :, z2_idx].float()).item()
                if pred_z is not None else 0.0
            )
            z7_mse = (
                F.mse_loss(pred_z[:, :, z7_idx].float(), teacher_z_8d[:, :, z7_idx].float()).item()
                if pred_z is not None else 0.0
            )
        logs = {
            "train/critic_z_loss": float(critic_z_loss.detach().item()) if torch.is_tensor(critic_z_loss) else 0.0,
            "train/critic_loss": float(critic_loss_k.detach().item()) if torch.is_tensor(critic_loss_k) else 0.0,
            "train/critic_z2_mse": z2_mse,
            "train/critic_z7_mse": z7_mse,
            "train/gen_z_loss": float(gen_z_loss.detach().item()) if torch.is_tensor(gen_z_loss) else 0.0,
            "train/gen_action_loss": float(generator_action_loss.detach().item()) if torch.is_tensor(generator_action_loss) else 0.0,
            "train/teacher_z2_mean": float(teacher_z_8d[:, :, z2_idx].mean().item()),
            "train/teacher_z7_mean": float(teacher_z_8d[:, :, z7_idx].mean().item()),
            "train/z_guidance_scale": float(guidance_scale),
            "train/teacher_unavailable": 0.0,
        }
        return generator_action_loss, logs, teacher_z_8d

    # ------------------------------------------------------------------
    # LoRA-side (real_score) action losses: action_critic z-guidance on
    # the LoRA's denoised x0 + state_probe MSE on the per-chunk z the
    # probe reads from real_score's internal taps. Mirror of the
    # gen-side z-guidance formulation; SHARES the action_critic +
    # state_probe heads (Option B). Critic is treated as frozen during
    # this branch (its gradient already came from the gen-side inner
    # loop in ``_compute_action_critic_losses``); state_probe is
    # treated as trainable here (the probe gets its only gradient from
    # this loss, since gen-side currently has no probe loss).
    # ------------------------------------------------------------------
    def _compute_aux_teacher_disc_losses(
        self,
        *,
        lora_x0: Optional[torch.Tensor],
        gt_target: Optional[torch.Tensor],
        chunk_lo: int,
        current_step: int,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        """Borrow the GAN disc's signal as a regulariser on the LoRA
        aux teacher's x0 estimate.

        Two paths, each gated by its own weight knob on the model:

          * ``aux_teacher_disc_adv_weight`` > 0 — RpGAN gen-loss style.
            Concat ``[gt_target, lora_x0]`` along batch, forward the
            LADD disc once, split the logits, apply
            ``rpgan_g_loss(d_real.detach(), d_fake)`` to push lora_x0
            toward "real" from the disc's perspective. Disc params
            are detached for this forward (we use the disc as a critic;
            it's still trained on the main gen-side path).

          * ``aux_teacher_disc_feat_weight`` > 0 — feature matching.
            Run the disc's projector (only) on both lora_x0 (grad-on)
            and gt_target (no_grad), L2 between per-block teacher
            features. LPIPS-style signal — no adversarial framing.

        Both default to 0 (off). When both > 0, both signals are
        summed. Gated on top of the aux teacher's own start gate by
        ``aux_teacher_disc_warmup_steps`` so the disc has had enough
        D-updates to be a useful critic before we read its score.

        The disc forward operates on a SINGLE per-block (npb-frame)
        chunk slice (``lora_x0[:, :npb]``) — keeps the forward at the
        disc's training distribution (it trains on per-chunk slices,
        not the full aux teacher window). One slice per iter is enough
        to give the LoRA a useful gradient.

        ``chunk_lo`` is the absolute ride-window frame index of
        ``lora_x0[:, 0]``, used to slice the matching action tokens.
        The aux teacher computes ``gt_target = ride_window[:, chunk_lo:
        chunk_lo + chunk_size]`` so ``lora_x0[:, :npb]`` corresponds
        to ride frames ``[chunk_lo, chunk_lo + npb)``. Caller MUST
        pass the chunk_lo it computed for this iter — reading the
        ``streaming_state["last_chunk_lo_in_ride_window"]`` stash
        would race with the main GAN path (set after this helper
        runs, so we'd see the PREVIOUS iter's value → action tokens
        misaligned with lora_x0's actual ride position).

        Returns ``(total_loss, logs)`` where ``total_loss`` is a
        graph-bearing scalar (gradient flows to LoRA params via
        ``lora_x0``), or ``None`` when the path is fully gated off.
        """
        # DDP-safety note: all early-return paths below are derivable
        # from globally-consistent state — config knobs (weights,
        # warmup), aux-teacher's own DDP-synced skip (which makes
        # lora_x0 / gt_target None on ALL ranks together when the
        # ride is too short), or deterministic shape/state derived
        # from those. Ride-actions length is tied to ride-window
        # length (same loader output), so its shape check inherits
        # the aux-teacher's sync. So every rank reaches the disc
        # forward together, or none do — no explicit all_reduce
        # needed here. If any of these conditions ever DOES diverge
        # per-rank in a future code change, the LoRA's DDP allreduce
        # at backward will detect the unused-param mismatch.
        adv_w = float(getattr(self.model, "aux_teacher_disc_adv_weight", 0.0))
        feat_w = float(
            getattr(self.model, "aux_teacher_disc_feat_weight", 0.0)
        )
        if adv_w == 0.0 and feat_w == 0.0:
            return None, {}
        if (
            not getattr(self, "gan_enabled", False)
            or self.r3gan_disc is None
        ):
            return None, {}
        if lora_x0 is None or gt_target is None:
            return None, {}
        warmup = int(
            getattr(self.model, "aux_teacher_disc_warmup_steps", 100)
        )
        if current_step < warmup:
            return None, {"train/aux_disc_skipped_warmup": 1.0}

        device = lora_x0.device
        disc = self.r3gan_disc
        npb = int(getattr(self.model, "num_frame_per_block", 3))
        if lora_x0.shape[1] < npb:
            return None, {"train/aux_disc_skipped_too_short": 1.0}

        # First npb frames — deterministic across ranks (DDP-safe).
        lora_chunk = lora_x0[:, :npb].to(torch.float32)
        gt_chunk = gt_target[:, :npb].to(torch.float32).detach()
        B = lora_chunk.shape[0]

        # Disc at t=0 (wavelet_hf path keeps clean inputs; mirrors
        # what ``_ladd_run_pair_mode`` does when wavelet_on).
        t_disc = torch.zeros((B, npb), dtype=torch.long, device=device)

        # ----- Prompt embeddings + pooled prompt -----
        s_state = getattr(self.model, "streaming_state", None) or {}
        prompt_embeds = (
            s_state.get("prompt_embeds")
            if isinstance(s_state, dict) else None
        )
        if prompt_embeds is None:
            return None, {"train/aux_disc_skipped_no_prompt": 1.0}
        # Batch broadcast. In practice the aux teacher and the ride
        # share the same B, so the shapes match. If they ever don't
        # (e.g. multi-prompt ride), require a clean integer ratio so
        # we don't silently produce a misaligned slice.
        if prompt_embeds.shape[0] != B:
            if B % max(1, prompt_embeds.shape[0]) != 0:
                return None, {"train/aux_disc_skipped_prompt_shape": 1.0}
            reps = B // prompt_embeds.shape[0]
            prompt_embeds = prompt_embeds.repeat_interleave(reps, dim=0)
        pooled_prompt = (
            prompt_embeds.float().mean(dim=1)
            if getattr(disc, "cmap_dim", 0) > 0 else None
        )

        # ----- Per-chunk action conditioning -----
        # ``chunk_lo`` is passed by the caller — it's the absolute
        # ride frame index of ``lora_x0[:, 0]``. Slicing the action
        # window at ``[chunk_lo : chunk_lo + npb)`` matches the
        # frames lora_x0[:, :npb] cover; the disc's action-aware
        # WAN forward then sees actions consistent with the latents.
        a_per_f = 0
        _rs = self.model.real_score
        for _cand in [_rs] + list(_rs.modules()):
            if hasattr(_cand, "action_tokens_per_frame"):
                a_per_f = int(getattr(_cand, "action_tokens_per_frame", 0))
                if a_per_f > 0:
                    break
        cond_extra: Optional[Dict[str, torch.Tensor]] = None
        if a_per_f > 0:
            ride_actions = (
                s_state.get("ride_actions_window")
                if isinstance(s_state, dict) else None
            )
            atp = getattr(self.model, "action_token_projection", None)
            ap = getattr(self.model, "action_projection", None)
            if (
                ride_actions is None
                or atp is None
            ):
                return None, {"train/aux_disc_skipped_no_actions": 1.0}
            lo, hi = int(chunk_lo), int(chunk_lo) + npb
            if lo < 0 or ride_actions.shape[1] < hi:
                return None, {"train/aux_disc_skipped_no_actions": 1.0}
            acts = ride_actions[:, lo:hi].to(
                device=device, dtype=prompt_embeds.dtype,
            )
            with torch.no_grad():
                act_tokens = atp(acts).detach()
                act_mod = (
                    ap(acts, num_frames=npb).detach()
                    if ap is not None else None
                )
            cond_extra = {"_action_tokens": act_tokens}
            if act_mod is not None:
                cond_extra["_action_modulation"] = act_mod

        logs: Dict[str, float] = {}
        total_loss: Optional[torch.Tensor] = None

        # Disc in eval (no spectral_norm buffer mutation), params
        # detached (critic mode — not trained on this signal).
        disc_was_training = disc.training
        disc.eval()
        disc.requires_grad_(False)
        # CRITICAL: the LADD disc's WanFeatureProjector holds
        # ``self.real_score`` as a plain Python attribute, not as an
        # nn.Module submodule (model/ladd_disc.py:99). So
        # ``disc.requires_grad_(False)`` above does NOT touch the
        # LoRA params inside real_score. Without freezing them
        # explicitly here, the disc/projector forward on ``lora_chunk``
        # would open a feedback path:
        #   disc_loss → projector's real_score(lora_chunk) → LoRA params
        # which is anti-correlated with the wanted signal: instead of
        # making ``lora_chunk`` closer to GT, it would tune the LoRA
        # so the FEATURE EXTRACTOR maps any input to GT-like features.
        # For the FM path this is especially toxic (no adversarial
        # counterbalance). Freezing the LoRA params for the duration
        # of the disc forward kills the feedback loop while
        # preserving the wanted path (disc_loss → lora_chunk → aux
        # teacher's autograd graph → LoRA params).
        # Memory bonus: the projector's activation checkpoint no
        # longer needs to save LoRA-bearing layer activations for a
        # backward that will never reach those params.
        _rs = self.model.real_score
        _lora_params_to_restore = [
            p for p in _rs.parameters() if p.requires_grad
        ]
        for _p in _lora_params_to_restore:
            _p.requires_grad_(False)
        try:
            if adv_w > 0.0:
                from model.r3gan import rpgan_g_loss
                combined = torch.cat([gt_chunk, lora_chunk], dim=0)
                _t = torch.cat([t_disc, t_disc], dim=0)
                _pe = torch.cat([prompt_embeds, prompt_embeds], dim=0)
                _pp = (
                    torch.cat([pooled_prompt, pooled_prompt], dim=0)
                    if pooled_prompt is not None else None
                )
                _ce = None
                if cond_extra is not None:
                    _ce = {
                        k: torch.cat([v, v], dim=0)
                        for k, v in cond_extra.items()
                    }
                combined_logits = disc(
                    x_noisy=combined,
                    timestep=_t,
                    prompt_embeds=_pe,
                    pooled_prompt=_pp,
                    conditional_extra=_ce,
                )
                d_real = combined_logits[:B]
                d_fake = combined_logits[B:]
                K_stat = int(getattr(disc, "stat_logit_count", 0))
                W_stat = float(getattr(
                    self.model, "ladd_stat_head_loss_weight", 1.0,
                ))
                if K_stat > 0 and W_stat > 0.0:
                    g_visual = rpgan_g_loss(
                        d_real[:, :-K_stat].detach(),
                        d_fake[:, :-K_stat],
                    )
                    g_stat = rpgan_g_loss(
                        d_real[:, -K_stat:].detach(),
                        d_fake[:, -K_stat:],
                    )
                    g_rp = g_visual + W_stat * g_stat
                else:
                    g_rp = rpgan_g_loss(d_real.detach(), d_fake)
                adv_loss = adv_w * g_rp.to(lora_x0.dtype)
                total_loss = (
                    adv_loss if total_loss is None
                    else total_loss + adv_loss
                )
                logs["train/aux_disc_adv_raw"] = float(
                    g_rp.detach().item()
                )
                logs["train/aux_disc_adv_weighted"] = float(
                    adv_loss.detach().item()
                )

            if feat_w > 0.0:
                projector = disc.projector
                # IMPORTANT: ``LADDDiscriminator.forward`` applies
                # ``self.wavelet_hf`` to x_noisy BEFORE the projector
                # call (model/ladd_disc.py:810-811). The projector was
                # trained to receive wavelet-HF-transformed inputs.
                # Mirror that pre-stage manually here — feeding raw
                # latents to the projector would be OOD relative to
                # the disc's training distribution.
                wavelet_hf = getattr(disc, "wavelet_hf", None)
                if wavelet_hf is not None:
                    lora_for_proj = wavelet_hf(lora_chunk)
                    with torch.no_grad():
                        gt_for_proj = wavelet_hf(gt_chunk)
                else:
                    lora_for_proj = lora_chunk
                    gt_for_proj = gt_chunk
                feats_fake = projector(
                    x_noisy=lora_for_proj,
                    timestep=t_disc,
                    prompt_embeds=prompt_embeds,
                    conditional_extra=cond_extra,
                )
                with torch.no_grad():
                    feats_real = projector(
                        x_noisy=gt_for_proj,
                        timestep=t_disc,
                        prompt_embeds=prompt_embeds,
                        conditional_extra=cond_extra,
                    )
                feat_l2 = None
                n_blocks = 0
                for idx in feats_fake:
                    diff_sq = (
                        feats_fake[idx]
                        - feats_real[idx].detach()
                    ).pow(2).mean()
                    feat_l2 = (
                        diff_sq if feat_l2 is None
                        else feat_l2 + diff_sq
                    )
                    n_blocks += 1
                if feat_l2 is not None and n_blocks > 0:
                    feat_l2 = feat_l2 / float(n_blocks)
                    feat_loss = feat_w * feat_l2.to(lora_x0.dtype)
                    total_loss = (
                        feat_loss if total_loss is None
                        else total_loss + feat_loss
                    )
                    logs["train/aux_disc_feat_raw"] = float(
                        feat_l2.detach().item()
                    )
                    logs["train/aux_disc_feat_weighted"] = float(
                        feat_loss.detach().item()
                    )
        finally:
            disc.requires_grad_(True)
            if disc_was_training:
                disc.train()
            # Restore the LoRA params' requires_grad state. The aux
            # teacher's autograd graph (recorded BEFORE this disc
            # forward) is independent of this restore — its backward
            # path through real_score still works because the
            # recorded ops captured grad-state at forward time.
            for _p in _lora_params_to_restore:
                _p.requires_grad_(True)

        if total_loss is not None:
            logs["train/aux_disc_total"] = float(total_loss.detach().item())
        return total_loss, logs

    def _compute_lora_action_losses(
        self,
        lora_x0: Optional[torch.Tensor],
        lora_state_preds: Optional[torch.Tensor],
        target_action_z: torch.Tensor,
        teacher_z_8d: torch.Tensor,
        chunk_t: torch.Tensor,
        current_step: int,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Dict[str, float],
    ]:
        """Returns ``(lora_action_loss, lora_state_probe_loss, logs)``.

        Each loss tensor is graph-bearing (grad flows back into LoRA
        params via the upstream forwards). ``None`` is returned for
        a loss when its required input is missing or its weight is 0.
        """
        logs: Dict[str, float] = {}
        zero = (
            torch.tensor(0.0, device=target_action_z.device,
                         dtype=target_action_z.dtype)
        )

        # Frozen-critic z-guidance on the LoRA's x0. Same warmup / ramp
        # contract as the gen-side z-guidance — pulls gen-side and
        # LoRA-side together so they activate at the same step. Read the
        # ramp config (``warmup_steps_for_guidance`` /
        # ``z_guidance_warmup_steps``) from existing trainer attrs.
        action_loss: Optional[torch.Tensor] = None
        if (
            lora_x0 is not None
            and self.action_critic_loss_active
            and self.lora_action_critic_z_guidance_weight > 0.0
        ):
            critic_for_guidance = (
                self.model.action_critic.module
                if isinstance(self.model.action_critic, DDP)
                else self.model.action_critic
            )
            warmup_start = self.warmup_steps_for_guidance
            if current_step < warmup_start:
                guidance_scale = 0.0
            elif self.z_guidance_warmup_steps > 0:
                ramp = min(
                    1.0,
                    (current_step - warmup_start)
                    / max(1, self.z_guidance_warmup_steps),
                )
                guidance_scale = ramp * self.lora_action_critic_z_guidance_weight
            else:
                guidance_scale = self.lora_action_critic_z_guidance_weight
            if guidance_scale > 0.0:
                critic_for_guidance.requires_grad_(False)
                try:
                    with torch.amp.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                        enabled=True,
                    ):
                        chunk_frames = int(self.config.num_frame_per_block)
                        n_chunks = lora_x0.shape[1] // chunk_frames
                        # Critic expects chunk-aligned actions
                        # ``[B, n_chunks, K]`` (mean over fpb frames per
                        # chunk), NOT the per-frame ``[B, F, K]`` stream.
                        # Mirror the gen-side prep in
                        # ``_compute_action_critic_losses``.
                        chunk_actions = _chunk_actions(
                            target_action_z, chunk_frames,
                        )[:, :n_chunks]
                        lora_pred_z = critic_for_guidance(
                            lora_x0, chunk_t, chunk_actions,
                        )
                        lora_pred_z = lora_pred_z[:, :n_chunks]
                        lora_z2z7 = lora_pred_z[:, :, self.action_critic_dims]
                        lora_z_mse = F.mse_loss(
                            lora_z2z7, chunk_actions.to(lora_z2z7.dtype),
                        )
                        action_loss = guidance_scale * lora_z_mse
                finally:
                    critic_for_guidance.requires_grad_(True)
                logs["train/lora_critic_z_loss"] = float(lora_z_mse.detach().item())
                logs["train/lora_action_loss"] = float(action_loss.detach().item())
                logs["train/lora_z_guidance_scale"] = float(guidance_scale)

        # State-probe MSE supervision on the LoRA-side per-chunk z. Target
        # is the SAME ``teacher_z_8d`` the gen-side critic trained
        # against (= cotracker+ss_vae extracted from the student's pred).
        # Loss flows to LoRA params via real_score's internal taps and
        # to state_probe params directly. ``state_probe`` is configured
        # as TRAINABLE in this branch (built via DDP at trainer init).
        # Diagnostic logging: surface whether the wrapper actually
        # produced state_preds this iter so a missing state_probe loss
        # is debuggable from wandb alone.
        logs["train/lora_state_preds_present"] = (
            1.0 if lora_state_preds is not None else 0.0
        )
        state_probe_loss: Optional[torch.Tensor] = None
        if (
            lora_state_preds is not None
            and self.state_probe_z_loss_weight > 0.0
        ):
            n_chunks = lora_state_preds.shape[1]
            tz = teacher_z_8d[:, :n_chunks].to(lora_state_preds.dtype)
            sp_mse = F.mse_loss(lora_state_preds, tz)
            state_probe_loss = self.state_probe_z_loss_weight * sp_mse
            logs["train/lora_state_probe_z_loss"] = float(sp_mse.detach().item())
            logs["train/lora_state_probe_loss"] = float(
                state_probe_loss.detach().item()
            )
            logs["train/state_probe_z_loss_weight"] = float(
                self.state_probe_z_loss_weight
            )

        return action_loss, state_probe_loss, logs

    # ------------------------------------------------------------------
    # R3GAN — RpGAN + R1.
    # Trains a separate 3D-conv discriminator on
    #   real = ride["latents"][:, gen_window_start:gen_window_end]
    #   fake = aux["pred_image"]
    # Returns the gen-side RpGAN-G term (graph-carrying through
    # ``pred_image`` -> generator) for the caller to add to the DMD
    # loss before the generator backward. The D-side update +
    # backward + optim step happens inside this method.
    #
    # DDP correctness:
    #   * D-update: uses ``disc_for_update`` (DDP-wrapped). Backward
    #     fires on D's own graph; ``fake_detached``/``real_detached``
    #     are not part of G's graph so no grad escapes.
    #   * G-update: uses ``disc_for_guidance`` (UNWRAPPED + frozen).
    #     We ``requires_grad_(False)`` D's params, so the only path
    #     for grad is through ``fake = pred_image`` -> generator's
    #     own DDP backward, which is fired in the outer caller.
    # ------------------------------------------------------------------
    def _sample_critic_grad_frame_indices(
        self, F: int, k: int, device: torch.device,
    ) -> List[int]:
        """Pick ``k`` distinct frame indices in [0, F) for the
        gradient-distillation subset, broadcast from rank 0 so all
        DDP ranks pick the same subset (otherwise the per-rank
        gradient targets would differ and the all-reduce on the
        critic backward would average inconsistent gradients).
        """
        k = max(1, min(int(k), int(F)))
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            idx = torch.randperm(F, device=device)[:k].sort().values
        else:
            idx = torch.empty(k, dtype=torch.long, device=device)
        if dist.is_initialized():
            dist.broadcast(idx, src=0)
        return idx.tolist()

    def _get_lpips_model(self) -> Optional["torch.nn.Module"]:
        """Lazy-build (and cache) the LPIPS-VGG perceptual distance
        model. Returns None when ``lpips_loss_weight == 0``.

        VGG16 backbone is downloaded once into ``~/.cache/torch/hub`` —
        all ranks already have it after the first run. Model is
        ``eval()`` + ``requires_grad_(False)`` so its parameters are
        not trained — only its forward gradient flows back into the
        gen via the decoded pixels.
        """
        if self.lpips_loss_weight <= 0:
            return None
        cached = getattr(self, "_lpips_model", None)
        if cached is not None:
            return cached
        try:
            import lpips as _lpips
        except ImportError as e:
            raise RuntimeError(
                "lpips_loss_weight>0 requires the 'lpips' python "
                f"package: pip install lpips. Original error: {e}"
            )
        model = _lpips.LPIPS(net="vgg", verbose=False)
        model = model.to(device=self.device, dtype=torch.float32)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._lpips_model = model
        if self.is_main_process:
            n_params = sum(p.numel() for p in model.parameters())
            logging.info(
                "[ActionForcing] LPIPS(vgg) lazy-built: params=%.2fM "
                "(weight=%.3f, n_frames=%d, crop_size=%d)",
                n_params / 1e6, self.lpips_loss_weight,
                self.lpips_n_frames, self.lpips_crop_size,
            )
        return model

    def _vae_decode_grad(
        self,
        latent: torch.Tensor,
        use_checkpoint: bool = True,
    ) -> torch.Tensor:
        """Graph-on VAE decode using the dummy-leading-frame trick.

        ``latent``: ``[B, F_lat, C, H_lat, W_lat]`` (with grad).
        Returns: ``[B, F_pix, 3, H_pix, W_pix]`` in ``[-1, 1]``.

        The WAN VAE single-shot decode produces a "special first
        latent" output (1 pixel frame) and 4× temporal expansion for
        the rest. We prepend a dummy frame and slice ``[:, 1:]`` so
        the dummy absorbs the special-first behavior and our actual
        latents get the full 4× expansion. Caller controls grad
        context — used here for graph-on decode that flows gradient
        back to the gen.

        ``use_checkpoint=True`` wraps the VAE forward in
        ``torch.utils.checkpoint`` so activations are recomputed
        during backward instead of cached. ~30% extra compute but
        ~5-10 GB activation savings — required to fit graph-on decode
        alongside the gen rollout's persistent activations.
        """
        vae = getattr(self.model, "vae", None)
        if vae is None:
            raise RuntimeError(
                "_vae_decode_grad requires self.model.vae."
            )
        # Self-seeding decode (``seed_first``): clears the temporal cache (no
        # cross-clip ghost in frame 0) AND seeds it with the clip's own first
        # latent so frame 0 has a faithful predecessor (no init-frame
        # brightness anomaly). This finally makes the code match the
        # "prepend a dummy frame and slice" geometry described above, and is
        # self-contained so checkpoint recompute is deterministic.
        if use_checkpoint:
            from torch.utils.checkpoint import checkpoint as _ckpt

            def _decode(z):
                return vae.decode_to_pixel(z, seed_first=True)

            pix = _ckpt(_decode, latent, use_reentrant=False)
        else:
            pix = vae.decode_to_pixel(latent, seed_first=True)
        return pix

    def _compute_pixel_perceptual_losses(
        self,
        pred_image: torch.Tensor,
        gt_latents_window: torch.Tensor,
        current_step: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """LPIPS + pixel reconstruction losses on a graph-on subset.

        Decodes a small (``lpips_n_frames``) random subset of the gen
        prediction through the VAE WITH GRAD, decodes the matching
        GT latents WITHOUT GRAD (target), optionally crops to a
        smaller resolution, and computes:

          * LPIPS-VGG distance per frame (anti-blur, perceptual)
          * Pixel L1 / MSE per frame (anti-drift, direct supervision)

        Returns ``(perceptual_loss, logs)`` where ``perceptual_loss``
        is graph-attached scalar (to be added to generator_loss);
        ``logs`` is a flat ``str -> float`` dict.

        Gated by ``self.lpips_loss_weight > 0`` AND/OR
        ``self.pixel_recon_loss_weight > 0``. Returns zero scalar +
        empty logs when both are zero.
        """
        device = pred_image.device
        zero = torch.zeros((), device=device, dtype=torch.float32)
        lpips_active = self.lpips_loss_weight > 0
        recon_active = self.pixel_recon_loss_weight > 0
        if not (lpips_active or recon_active):
            return zero, {}

        # Pick a small random frame subset (DDP-synced).
        F_lat = pred_image.shape[1]
        n_pick = max(1, min(int(self.lpips_n_frames), int(F_lat)))
        frame_idx = self._sample_critic_grad_frame_indices(
            F_lat, n_pick, device,
        )

        # Slice latents to the subset and decode.
        pred_lat_sub = pred_image[:, frame_idx].to(torch.float32)
        gt_lat_sub = (
            gt_latents_window[:, frame_idx].detach().to(torch.float32)
        )
        # Graph-on for gen, no_grad for GT.
        pred_pix = self._vae_decode_grad(pred_lat_sub).to(torch.float32)
        with torch.no_grad():
            gt_pix = self._vae_decode_grad(gt_lat_sub).to(torch.float32)

        # ``pred_pix`` / ``gt_pix`` are ``[B, F_pix, 3, H, W]`` in
        # ``[-1, 1]``. The WAN VAE decode produces 4× temporal
        # expansion; both sides have the same F_pix so they're
        # frame-aligned for per-pixel loss computation.
        B, F_pix, C, H, W = pred_pix.shape

        # Optional random spatial crop. Single crop per iter (DDP-
        # synced) so all ranks compute the loss on the same region.
        crop = int(self.lpips_crop_size)
        if 0 < crop < min(H, W):
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                h_off = torch.randint(0, H - crop + 1, (1,), device=device)
                w_off = torch.randint(0, W - crop + 1, (1,), device=device)
                offsets = torch.cat([h_off, w_off]).long()
            else:
                offsets = torch.empty(2, dtype=torch.long, device=device)
            if dist.is_initialized():
                dist.broadcast(offsets, src=0)
            h_o = int(offsets[0].item())
            w_o = int(offsets[1].item())
            pred_pix = pred_pix[..., h_o:h_o + crop, w_o:w_o + crop]
            gt_pix = gt_pix[..., h_o:h_o + crop, w_o:w_o + crop]

        # Flatten frames into batch for both losses (frame-independent).
        pred_flat = pred_pix.reshape(B * F_pix, C, pred_pix.shape[-2], pred_pix.shape[-1])
        gt_flat = gt_pix.reshape(B * F_pix, C, gt_pix.shape[-2], gt_pix.shape[-1])

        loss = zero
        logs: Dict[str, float] = {}
        if lpips_active:
            lpips_model = self._get_lpips_model()
            # LPIPS expects ``[N, 3, H, W]`` in ``[-1, 1]`` — already
            # in that range from the WAN VAE clamp.
            lpips_dist = lpips_model(pred_flat, gt_flat)
            lpips_loss = lpips_dist.mean()
            loss = loss + self.lpips_loss_weight * lpips_loss
            logs["train/lpips_loss_raw"] = float(lpips_loss.detach().item())
            logs["train/lpips_loss_weighted"] = float(
                (self.lpips_loss_weight * lpips_loss).detach().item()
            )
        if recon_active:
            if self.pixel_recon_loss_type == "mse":
                recon_loss = ((pred_flat - gt_flat) ** 2).mean()
            else:  # default l1
                recon_loss = (pred_flat - gt_flat).abs().mean()
            loss = loss + self.pixel_recon_loss_weight * recon_loss
            logs["train/pixel_recon_loss_raw"] = float(
                recon_loss.detach().item()
            )
            logs["train/pixel_recon_loss_weighted"] = float(
                (self.pixel_recon_loss_weight * recon_loss).detach().item()
            )
        logs["train/lpips_n_frames"] = float(n_pick)
        return loss.to(pred_image.dtype), logs

    def _vae_decode_nograd(self, latent: torch.Tensor) -> torch.Tensor:
        """No-grad VAE decode — the cheap twin of ``_vae_decode_grad``.

        Same self-seeding (``seed_first=True``) decode geometry, but the
        input is detached and the whole forward runs under ``no_grad`` (no
        checkpointing needed: nothing is recomputed because there is no
        backward). Returns ``[B, F_pix, 3, H, W]`` in ``[-1, 1]``.

        Used by the A5 texture tripwire. Deliberately separate from
        ``_vae_decode_grad`` so no call site can accidentally attach a graph
        to the tripwire.
        """
        vae = getattr(self.model, "vae", None)
        if vae is None:
            raise RuntimeError("_vae_decode_nograd requires self.model.vae.")
        with torch.no_grad():
            return vae.decode_to_pixel(
                latent.detach().to(torch.float32), seed_first=True,
            )

    def _maybe_texture_tripwire(
        self,
        pred_latents: torch.Tensor,
        logs: Dict[str, float],
        gt_latents: Optional[torch.Tensor] = None,
    ) -> None:
        """A5 — no-grad decoder tripwire (default OFF).

        Every ``texture_tripwire_every`` OUTER training steps, decode a
        small slice of the generator's latents under ``no_grad`` and run the
        anisotropic texture battery (``analysis/texture_stats.py``) on the
        decoded pixels. Logged under ``train/tripwire_*``.

        Why decoded pixels and not latents: the A/B/C diagnostic located the
        failure in the student latent but it is RENDERED as banding — a
        critic gaming a latent-space objective can look fine in the latent
        and pathological once decoded. This is the in-training early warning
        for exactly that, instead of finding it in a 60 s eval two hours
        later.

        Contract: no gradient, no backward, no state mutation on the model,
        no collective. ``_streaming_train_one_chunk`` runs several times per
        outer step, so a ``_tripwire_last_step`` latch keeps it to one fire
        per step. Any exception is swallowed into a counter — a diagnostic
        must never take training down.
        """
        every = int(getattr(self, "texture_tripwire_every", 0) or 0)
        if every <= 0 or pred_latents is None:
            return
        step = int(getattr(self, "step", 0))
        if step % every != 0:
            return
        if getattr(self, "_tripwire_last_step", None) == step:
            return
        self._tripwire_last_step = step
        try:
            from analysis.texture_stats import texture_battery, battery_delta

            n_lat = int(pred_latents.shape[1])
            n_pick = max(1, min(int(self.texture_tripwire_frames), n_lat))
            # Deterministic slice (the LAST n_pick latent frames): the
            # tripwire is a trend readout across steps, so a fixed rule
            # beats a random draw whose variance would swamp the signal.
            lat = pred_latents[:1, n_lat - n_pick:].detach()
            hf_cut = float(getattr(self, "texture_tripwire_hf_cut", 0.25))
            # Defragment before the decode: the decoder wants a large
            # contiguous workspace and the gen rollout's live activations
            # leave only fragmented gaps (same reason the graph-on
            # perceptual decode does this). Only runs on tripwire steps.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pix = self._vae_decode_nograd(lat)
            with torch.no_grad():
                sample = texture_battery(
                    ((pix.float() + 1.0) * 0.5).clamp(0.0, 1.0),
                    hf_cut=hf_cut,
                )
            for k, v in sample.items():
                logs[f"train/tripwire_{k}"] = float(v)
            logs["train/tripwire_fired"] = 1.0
            logs["train/tripwire_n_lat_frames"] = float(n_pick)
            if bool(getattr(self, "texture_tripwire_gt_ref", False)) and (
                gt_latents is not None
            ):
                gt_lat = gt_latents[:1, -n_pick:].detach()
                gt_pix = self._vae_decode_nograd(gt_lat)
                with torch.no_grad():
                    ref = texture_battery(
                        ((gt_pix.float() + 1.0) * 0.5).clamp(0.0, 1.0),
                        hf_cut=hf_cut,
                    )
                for k, v in ref.items():
                    logs[f"train/tripwire_gt_{k}"] = float(v)
                for k, v in battery_delta(sample, ref).items():
                    logs[f"train/tripwire_ratio_{k}"] = float(v)
            del pix
        except Exception as _e:  # pragma: no cover - diagnostic only
            logs["train/tripwire_err"] = 1.0
            if getattr(self, "is_main_process", True) and getattr(
                self, "_tripwire_err_logged", 0,
            ) < 3:
                self._tripwire_err_logged = getattr(
                    self, "_tripwire_err_logged", 0) + 1
                logging.warning(
                    "[TEXTURE-TRIPWIRE] failed at step %d: %s "
                    "(diagnostic only; training unaffected)", step, _e,
                )

    # ==================================================================
    # WP-PIXGAN (B1 / T3-B) -- real+fake supply and the D-update loop.
    #
    # Spec: docs/TEXTURE_GAN_DESIGN.md §2/§3/§5.2/§5.3, docs/WP_PIXGAN.md
    # §1 (contracts), §2 (fixed decisions), §9 (the §6 memory-budget
    # CORRECTION -- decode cost scales with pix_crops_per_step ONLY,
    # because model/disc_holdout_probe.py::_decode_crops decodes the whole
    # latent crop and only THEN draws the pixel-frame subset).
    #
    # Everything in this block is gated on ``gan_pixel_texture_enabled``
    # (built by T3-A, default False). When the gate is off nothing here
    # runs, nothing decodes, no log key is emitted and -- critically --
    # no RNG is consumed: every draw below comes from a PRIVATE
    # ``torch.Generator`` created inside the gated branch, never from the
    # global stream.
    #
    # NOT in this chunk (T3-C owns them): the generator-side G-term into
    # ``gen_gan_loss``, the full §7 telemetry surface, the ``pix_*``
    # config block.
    # ==================================================================

    # A20/A24/A21 invariants, restated as code-visible constants so a
    # config typo cannot silently relax them.
    PIX_BAND_COUNT_PINNED = 3          # A24 -- COARSE thirds, permanent
    PIX_A21_SUPPORT_FLOOR = 4096       # A21 -- distinct source frames
    #: A21 -- the STEADY-STATE pool's nominal ceiling
    #: (``pix_real_pool_windows x pix_lat_frames_per_crop``) measured in
    #: units of :data:`PIX_A21_SUPPORT_FLOOR`.  A pool that lands ON the
    #: floor has no margin at all: windows overlap inside a ride (a window
    #: at ``s`` contributes ``{s, s+1}``, so a window at ``s+1`` adds only
    #: one new frame), which is why 2,048 x 2 measured 4,069..4,085 rather
    #: than 4,096 across seeds.  Below this ratio the pool alone cannot
    #: carry the claim and it rests ENTIRELY on continuous refresh, which
    #: is a warning, not a failure -- refresh is unbounded, so the floor is
    #: still reachable.  The FAILURE is the measured one below.
    PIX_A21_POOL_MARGIN_MIN = 1.25
    #: A21 -- D-updates allowed to bring the pool from cold to its cap
    #: before ``pix_a21_support_ok`` stops being a warm-up reading and
    #: becomes a hard claim.  Default for ``pix_real_pool_warm_updates``.
    PIX_REAL_POOL_WARM_UPDATES = 64
    #: A20/A21 -- how many D-updates of drawn source frames the rolling
    #: reuse gauge remembers.  Default for ``pix_real_reuse_horizon``.
    PIX_REAL_REUSE_HORIZON = 32
    #: A20 -- ``pix_real_repeat_frac`` alarm.  The gauge measures GENUINE
    #: source-frame overlap among one D-update's reals, so a small nonzero
    #: reading is the expected signature of independent sampling and must
    #: NOT be fatal (forcing it to exactly 0 is forcing without-replacement
    #: sampling, i.e. the opposite of independence).  What is fatal is a
    #: STRUCTURAL break: if frame-within-crop expansion ever leaked to the
    #: real side, every crop's ``k_frames`` images would share one source
    #: frame and the fraction would jump to ``(k-1)/k >= 0.5``.  A
    #: coincidence cannot reach half the batch.
    PIX_A20_REPEAT_ALARM = 0.5
    #: §5.3 -- the shipped ``PIX_R1_GAMMA``.  Measured INERT on BOTH
    #: architectures (R1/d_loss ~ 5e-6 norm-free, ~6e-4 normed; see
    #: ``model/pixel_texture_disc.py``'s INERT-GAMMA WARNING), so a config
    #: carrying it is indistinguishable from a config that never
    #: calibrated.  ``_pix_resolve_r1_gamma`` refuses it.
    PIX_R1_GAMMA_INERT = 1.0

    # ------------------------------------------------------------------
    # CONFIG RESOLUTION -- the single place every pix_* knob becomes a
    # number, and the ONLY place the invariants are checked.
    # ------------------------------------------------------------------
    def _pix_resolve_r1_gamma(self) -> float:
        """§5.3 -- resolve ``pix_r1_gamma``, LOUDLY.

        Symmetry with ``pix_gan_weight``.  That knob is routed through
        ``model.pixel_texture_disc.resolve_gan_weight(..., strict=True)``
        so an uncalibrated weight RAISES, and the reason is that a
        withdrawn default is worse than no default: it launches, it looks
        configured, and it is wrong.  ``pix_r1_gamma`` was in exactly the
        same position with none of the protection -- its module default
        ``PIX_R1_GAMMA = 1.0`` is MEASURED INERT on both architectures
        (R1/d_loss ~= 3.7e-06..7.6e-06 norm-free, 4.5e-04..9.6e-04 normed,
        against ``d_loss = 2 ln 2``; the gamma that would put R1 at 0.1 %
        of ``d_loss`` is 131..273 on the shipped build).  So an arm could
        launch with R1 switched OFF while ``train/pix_r1_rate`` read a
        confident 1.00: the penalty fires on every D-update and
        contributes nothing.

        Both "absent" and "still carrying the inert shipped default" are
        therefore refused.  Two values pass:

          * any calibrated positive gamma (calibrate from a measured
            ``train/pix_r1_grad_sq_mean``, §5.3);
          * an explicit ``0.0`` -- R1 deliberately OFF, the exact analogue
            of the ``pix_gan_weight=0.0`` weight-free probe.  It is a
            choice, it is visible in the resolved-config echo, and
            ``train/pix_r1_disabled`` marks it in the trace.

        There is no acknowledgement flag for 1.0.  An arm that genuinely
        wants no R1 asks for 0.0 and says so.
        """
        raw = getattr(self.config, "pix_r1_gamma", None)
        if raw is None:
            raise ValueError(
                "pix_r1_gamma is not calibrated (None). §5.3's gamma is "
                "tied to the crop it was measured at and was NEVER "
                "calibrated on the shipped norm-free critic; there is no "
                "value to inherit. Calibrate it from a measured "
                "train/pix_r1_grad_sq_mean, or pass an explicit 0.0 to run "
                "with R1 deliberately OFF."
            )
        try:
            g = float(raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"pix_r1_gamma={raw!r} is not a number."
            )
        if not math.isfinite(g):
            raise ValueError(
                f"pix_r1_gamma={g!r} is not finite."
            )
        if g < 0.0:
            raise ValueError(
                f"pix_r1_gamma={g} is negative: R1 is a PENALTY, and a "
                "negative gamma rewards a sharp critic."
            )
        if g == float(self.PIX_R1_GAMMA_INERT):
            raise ValueError(
                "pix_r1_gamma=%r is the INERT shipped default and is "
                "REFUSED. It is not a weak setting, it is an absent one: "
                "at this gamma R1/d_loss measures 3.7e-06..7.6e-06 on the "
                "shipped norm-free critic (4.5e-04..9.6e-04 with the "
                "GroupNorm), i.e. below 0.001 %% of d_loss, so the arm "
                "would train with R1 OFF while train/pix_r1_rate reads "
                "1.00. The gamma that puts R1 at 0.1 %% of d_loss is "
                "131..273. Calibrate from a measured "
                "train/pix_r1_grad_sq_mean, or pass an explicit 0.0 to "
                "mean it." % (self.PIX_R1_GAMMA_INERT,)
            )
        return g

    def _pix_resolve_cfg(self) -> Dict[str, Any]:
        """Resolve + VALIDATE the whole ``pix_*`` block, once, up front.

        Every invariant that a config typo could relax lives HERE, not in
        one of the consumers.  The A24 band pin used to sit inside the
        D-loop, AFTER two early returns, while
        :meth:`_compute_pixel_texture_g_loss` read ``pix_band_count`` with
        no pin at all and ran EARLIER in the step.  A pin enforced by one
        of two consumers, late, is a pin with a hole in it; a pin enforced
        at resolution time is enforced for everyone who resolves, which is
        now both consumers and the constructor.

        Raises (never warns) on: a tightened A24 band count, degenerate
        crop/frame counts, ``pix_reals_per_fake < 1``, a frozen real pool
        (``pix_real_pool_refresh < 1`` -- A21 requires CONTINUOUS refresh,
        so 0 is not "cheap", it is the memorisation hazard by
        construction), and an uncalibrated ``pix_r1_gamma``.

        WARNS once on a real pool whose steady-state ceiling does not
        clear :data:`PIX_A21_SUPPORT_FLOOR` by
        :data:`PIX_A21_POOL_MARGIN_MIN`.  Not fatal, and deliberately so:
        since refresh keeps admitting after the pool reaches its cap
        (FIFO eviction), cumulative distinct-source support is UNBOUNDED
        in the pool size and the floor stays reachable from any pool.  The
        margin is about how much of the claim the pool itself carries.
        The claim that DOES fail hard is the measured one, asserted in the
        D-loop once warm-up is over.
        """
        from model.pixel_texture_disc import (
            PIX_BAND_COUNT, PIX_CROP_LAT, PIX_CROPS_PER_STEP,
            PIX_DECODE_BORDER_TRIM, PIX_FRAMES_PER_CROP, PIX_LOSS_FORM,
            PIX_R1_SIGMA, PIX_REALS_PER_FAKE,
        )

        cfg = self.config
        crop_lat = tuple(getattr(cfg, "pix_crop_lat", PIX_CROP_LAT))
        n_bands = int(getattr(cfg, "pix_band_count", PIX_BAND_COUNT))
        # A24 is PINNED. pix_band_count is registered so an override is not
        # misreported, not so it can be tightened: every approach toward
        # exact-y matching (more bands, per-row bands, bands that scale with
        # row count, +-tolerance matching) is forbidden and permanent.
        if n_bands != self.PIX_BAND_COUNT_PINNED:
            raise ValueError(
                f"pix_band_count={n_bands} rejected. A24 pins COARSE thirds "
                f"({self.PIX_BAND_COUNT_PINNED}); raising the band count is "
                "the first step toward exact-y matching, which is forbidden "
                "and permanent (docs/WP_PIXGAN.md §2)."
            )
        n_crops = int(getattr(cfg, "pix_crops_per_step", PIX_CROPS_PER_STEP))
        k_frames = int(getattr(cfg, "pix_frames_per_crop", PIX_FRAMES_PER_CROP))
        if n_crops < 1 or k_frames < 1:
            raise ValueError(
                f"pix_crops_per_step={n_crops} / pix_frames_per_crop="
                f"{k_frames} must both be >= 1. A degenerate count would "
                "produce empty crop/origin lists and turn the §7 means into "
                "forged zeros."
            )
        reals_per_fake = int(
            getattr(cfg, "pix_reals_per_fake", PIX_REALS_PER_FAKE)
        )
        if reals_per_fake < 1:
            raise ValueError(
                "pix_reals_per_fake must be >= 1: A20 is an invariant (one "
                "independently sourced real per fake), not a budget."
            )
        lat_frames = max(1, int(getattr(cfg, "pix_lat_frames_per_crop", 2)))
        pool_windows = int(getattr(cfg, "pix_real_pool_windows", 2048))
        if pool_windows < 1:
            raise ValueError(
                f"pix_real_pool_windows={pool_windows} must be >= 1."
            )
        refresh = int(getattr(cfg, "pix_real_pool_refresh", 8))
        if refresh < 1:
            raise ValueError(
                f"pix_real_pool_refresh={refresh} rejected. A21 requires the "
                "real pool to be CONTINUOUSLY REFRESHED -- 0 freezes it into "
                "exactly the mini-dataset D can memorise, which is the "
                "hazard A21 exists to prevent, not a cheaper setting."
            )
        warm_updates = int(getattr(
            cfg, "pix_real_pool_warm_updates", self.PIX_REAL_POOL_WARM_UPDATES,
        ))
        if warm_updates < 1:
            raise ValueError(
                f"pix_real_pool_warm_updates={warm_updates} must be >= 1."
            )
        reuse_horizon = int(getattr(
            cfg, "pix_real_reuse_horizon", self.PIX_REAL_REUSE_HORIZON,
        ))
        if reuse_horizon < 1:
            raise ValueError(
                f"pix_real_reuse_horizon={reuse_horizon} must be >= 1."
            )
        r1_num = getattr(cfg, "pix_r1_num_samples", None)
        r1_num = None if r1_num is None else int(r1_num)
        r1_gamma = self._pix_resolve_r1_gamma()
        # Knobs that used to be read straight off ``cfg`` at their consumer
        # AND again, independently, in the resolved-config echo. That is the
        # second-resolution-point defect wearing a different hat: the echo
        # is the artefact a reader TRUSTS to settle what ran, so an echo
        # sourced from anywhere other than the resolution point is the one
        # place a drift is guaranteed not to be noticed.
        #
        # Not routed here, deliberately (each has a consumer that must not
        # reach a validating resolver):
        #   pix_finish_grad_enabled -- read in _build_pipeline, OUTSIDE the
        #     pixel gate; resolving there would raise for every non-pixgan
        #     run. The D-loop reads it ONCE into a local instead.
        #   pix_gan_weight -- its authoritative resolution is
        #     model.pixel_texture_disc.resolve_gan_weight(strict=True) in
        #     _pix_gen_weight; a second, laxer copy here is the exact defect
        #     this block exists to prevent.
        #   pix_poscontrol_* -- _pix_positive_control must do NO work when
        #     off (§8.1: not a tensor, not a decode, not an RNG draw), and a
        #     whole-block resolve above its gate is work.
        r1_every_n = int(getattr(cfg, "pix_r1_every_n", 1))
        seed = int(getattr(cfg, "pix_seed", 20260823))
        rank_synced = bool(
            getattr(cfg, "pix_real_draw_rank_synced", False)
        )
        # §5.2 / RESEARCHER EXPOSURE DIRECTIVE (2026-08-24) --
        # ``pix_gan_updates_per_step`` is FOLLOW-BY-DEFAULT: null/absent
        # means "follow ``gan_updates_per_step``" (5 in the ganfix arms,
        # matching the LADD disc's cadence), and a pinned integer still
        # wins if set. The old shape -- a getattr whose DEFAULT was the
        # follow -- meant the config's pinned 1 silently capped the pixel
        # critic at 1/5th of the LADD cadence while reading as if it
        # followed. Resolved HERE (one read site); the derived value AND
        # its source are echoed (``pix_gan_updates_per_step_derived`` /
        # ``..._followed``) so the trace settles what ran.
        _updates_pinned = getattr(cfg, "pix_gan_updates_per_step", None)
        if _updates_pinned is None:
            n_updates = int(getattr(self, "gan_updates_per_step", 1))
            updates_followed = True
        else:
            n_updates = int(_updates_pinned)
            updates_followed = False
        if n_updates < 0:
            raise ValueError(
                f"pix_gan_updates_per_step={n_updates} must be >= 0 "
                "(0 = no D-updates this step; negative is meaningless)."
            )

        pool_ceiling = int(pool_windows) * int(lat_frames)
        pool_margin = float(pool_ceiling) / float(self.PIX_A21_SUPPORT_FLOOR)
        if (pool_margin < float(self.PIX_A21_POOL_MARGIN_MIN)
                and not getattr(self, "_pix_pool_margin_warned", False)):
            self._pix_pool_margin_warned = True
            logging.warning(
                "[pixgan] A21: pix_real_pool_windows=%d x "
                "pix_lat_frames_per_crop=%d = %d nominal source frames, "
                "only %.2fx the %d floor (want >= %.2fx). Windows OVERLAP "
                "inside a ride, so the realised ceiling is below the "
                "nominal one; the A21 claim then rests entirely on "
                "continuous refresh (%d window(s)/D-update).",
                pool_windows, lat_frames, pool_ceiling, pool_margin,
                int(self.PIX_A21_SUPPORT_FLOOR),
                float(self.PIX_A21_POOL_MARGIN_MIN), refresh,
            )

        return {
            "crop_lat": crop_lat,
            "crop_rows": int(crop_lat[0]),
            "crop_cols": int(crop_lat[1]),
            "n_crops": n_crops,
            "k_frames": k_frames,
            "n_bands": n_bands,
            "reals_per_fake": reals_per_fake,
            "lat_frames": lat_frames,
            "border": int(
                getattr(cfg, "pix_decode_border_trim", PIX_DECODE_BORDER_TRIM)
            ),
            "loss_form": str(getattr(cfg, "pix_loss_form", PIX_LOSS_FORM)),
            "decode_batch": int(getattr(cfg, "pix_decode_batch", 4)),
            "r1_gamma": r1_gamma,
            "r1_sigma": float(getattr(cfg, "pix_r1_sigma", PIX_R1_SIGMA)),
            "r1_num": r1_num,
            "pool_windows": pool_windows,
            "pool_refresh": refresh,
            "pool_warm_updates": warm_updates,
            "reuse_horizon": reuse_horizon,
            "pool_ceiling_frames": pool_ceiling,
            "pool_margin": pool_margin,
            "r1_every_n": r1_every_n,
            "seed": seed,
            "real_draw_rank_synced": rank_synced,
            "n_updates": int(n_updates),
            "n_updates_followed": bool(updates_followed),
        }

    def _pix_sync_generator(
        self, current_step: int, update_idx: int, *, salt: int = 0,
    ) -> torch.Generator:
        """A rank-consistent CPU ``torch.Generator`` for one D-update.

        DDP contract (mirrors ``_sample_critic_grad_frame_indices``): the
        SEED is drawn on rank 0 and ``dist.broadcast``-ed, so every draw
        taken from the returned generator is bit-identical on every rank.
        That is strictly stronger than broadcasting each index tensor and
        costs one 1-element collective per D-update instead of one per
        draw.  The band plan, the crop offsets, the pixel-frame subset and
        the R1 subsample all come from this generator.

        It is a PRIVATE generator: it never touches the global RNG stream,
        which is what makes the gate-off path byte-identical (no global
        draw is added, and no draw is re-ordered).

        ``salt`` separates independent draw streams inside one update.
        """
        base = int(self._pix_resolve_cfg()["seed"])
        val = (
            base * 1000003
            + int(current_step) * 9176
            + int(update_idx) * 31
            + int(salt) * 7919
        ) % (2 ** 31 - 1)
        dev = self.device if torch.cuda.is_available() else torch.device("cpu")
        if dist.is_initialized():
            rank = dist.get_rank()
            t = (
                torch.tensor([val], dtype=torch.long, device=dev)
                if rank == 0
                else torch.empty(1, dtype=torch.long, device=dev)
            )
            dist.broadcast(t, src=0)
            val = int(t.item())
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(val))
        return gen

    def _pix_rank_generator(self, current_step: int, update_idx: int) -> torch.Generator:
        """Per-RANK generator for the real-supply *data* draws.

        Deliberately NOT rank-synced.  Which ride/window each rank loads is
        a data-sharding decision, exactly like the training batch: if every
        rank loaded the identical reals the effective real support would be
        divided by ``world_size`` for no benefit.  Nothing computed from
        this generator enters a quantity that is all-reduced across ranks
        -- the D gradient is averaged over per-rank *samples*, which is the
        whole point of data-parallel training.  Draws that DO feed a shared
        quantity (band plan, crop offsets, R1 subsample/eps) come from
        :meth:`_pix_sync_generator` instead.

        Set ``pix_real_draw_rank_synced: true`` to force the rank-synced
        stream here as well (diagnostic only -- it shrinks real support).
        """
        rc = self._pix_resolve_cfg()
        if bool(rc["real_draw_rank_synced"]):
            return self._pix_sync_generator(current_step, update_idx, salt=5)
        rank = dist.get_rank() if dist.is_initialized() else 0
        base = int(rc["seed"])
        val = (
            base * 7919
            + int(current_step) * 104729
            + int(update_idx) * 1301
            + (rank + 1) * 15485863
        ) % (2 ** 31 - 1)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(val))
        return gen

    # ------------------------------------------------------------------
    # §3.6 / A24 -- the crop helper.  COMMITTED CROSS-PACKAGE CONTRACT.
    # ------------------------------------------------------------------
    def _pix_take_crops_with_origins(
        self,
        source: torch.Tensor,
        *,
        n_crops: int,
        crop_rows: int,
        crop_cols: int,
        n_bands: int,
        gen: torch.Generator,
        bands: Optional[Sequence[int]] = None,
        offs_y: Optional[Sequence[int]] = None,
    ) -> Tuple[torch.Tensor, List[int], List[int], List[int]]:
        """Draw ``n_crops`` latent crops AND RETURN THEIR ORIGINS.

        ``source``: ``[N, F, C, H, W]`` latents.  Pure slicing -- the
        autograd graph of ``source`` is preserved, so this is the helper
        both the grad-on fake path and the ``no_grad`` D-loop use.  What
        separates grad from no-grad is the *decode* helper you call next
        (:meth:`_pix_decode_crops_grad` vs
        :meth:`_pix_decode_crops_nograd`), which are two DIFFERENTLY NAMED
        functions on purpose (docs/WP_PIXGAN.md §4 commitment 1): a
        ``no_grad`` helper with a ``grad=True`` flag would let a caller
        land on the wrong one by defaulting.

        Returns ``(crops, ys, xs, bands)``:
          * ``crops``: ``[n_crops, F, C, crop_rows, crop_cols]``
          * ``ys``:    top-row origin of each crop, in **LATENT rows**
          * ``xs``:    left-col origin of each crop, in **LATENT cols**
          * ``bands``: A24 band index of each crop, in ``[0, n_bands)``

        WHY the origins are returned (docs/WP_PIXGAN.md §4 commitment 2):
        ``model/disc_holdout_probe.py::take_crops`` draws ``x`` internally
        and throws it away, and that file is not WP-PIXGAN's to edit.
        WP-SURROGATE (B3) consumes ``(y0, x0)`` in latent rows/cols for its
        positional embedding, so this signature is a recorded commitment --
        it is stable and must not be reordered.

        ``ys`` still comes from ``disc_holdout_probe.band_plan``, so A24's
        coarse banding is bit-for-bit the probe's: the admissible offset
        range is split into ``n_bands`` equal bins and the band index is
        drawn uniformly.  ``xs`` is drawn uniformly and independently --
        horizontal position is not stratified (dashcam content is not
        horizontally structured the way sky/building/road is vertically).

        ``bands``/``offs_y`` may be supplied to REPLAY a fake's plan onto
        the real side; that is how the A24 real/fake band pairing is
        enforced (every paired real is drawn from the same band, and the
        mismatch counter is asserted 0).
        """
        from model.disc_holdout_probe import band_plan

        if source.dim() != 5:
            raise ValueError(
                "_pix_take_crops_with_origins expects [N,F,C,H,W]; got "
                f"{tuple(source.shape)}"
            )
        n, _f, _c, h, w = source.shape
        cr = min(int(crop_rows), int(h))
        cc = min(int(crop_cols), int(w))
        if bands is None or offs_y is None:
            b_list, y_list = band_plan(
                int(n_crops), int(h), cr, int(n_bands), gen,
            )
        else:
            b_list = [int(b) for b in bands]
            y_list = [int(y) for y in offs_y]
            if len(b_list) != int(n_crops) or len(y_list) != int(n_crops):
                raise ValueError(
                    "_pix_take_crops_with_origins: replayed bands/offs_y must "
                    f"have length n_crops={int(n_crops)}"
                )
        x_hi = max(1, int(w) - cc + 1)
        picks: List[torch.Tensor] = []
        ys: List[int] = []
        xs: List[int] = []
        for y in y_list:
            i = int(torch.randint(0, int(n), (1,), generator=gen).item())
            x0 = int(torch.randint(0, x_hi, (1,), generator=gen).item())
            y0 = max(0, min(int(y), int(h) - cr))
            picks.append(source[i, :, :, y0:y0 + cr, x0:x0 + cc])
            ys.append(int(y0))
            xs.append(int(x0))
        return torch.stack(picks, dim=0), ys, xs, [int(b) for b in b_list]

    # ------------------------------------------------------------------
    # §3.2 -- latent-crop-then-decode.  TWO helpers, two names.
    # ------------------------------------------------------------------
    def _pix_vae_dtype(self) -> torch.dtype:
        """The dtype the WAN VAE's own weights are in -- DERIVED, not assumed.

        The WAN VAE is loaded in fp32 while the streaming latents this
        trainer carries are bf16, and ``F.conv3d`` refuses the mix:

            RuntimeError: Input type (c10::BFloat16) and bias type (float)
                          should be the same

        That is the exact crash that killed run 6113341 at step ~16 in
        ``wan/modules/vae.py`` (``x = self.conv2(z)``), and it is the SAME
        hazard ``configs/action_forcing_phase3_dmd.yaml`` documents on
        ``boundary_vae_roundtrip: false`` -- disabled there rather than
        fixed.  The other decode call sites never tripped it because they
        cast the decode's OUTPUT (``_vae_decode_grad(x).to(float32)``),
        not its INPUT, and their paths are gated off in this config; the
        pixel critic is the first consumer to hand the decoder a bf16
        latent with grad attached.

        Derived from the VAE's first parameter (falling back to its first
        floating-point buffer) so it stays correct if the VAE is ever
        loaded in another precision -- a hard-coded ``float32`` would
        silently become the same class of bug in reverse.  ``float32`` is
        the last-resort fallback: it is what the shipped WAN VAE uses and
        what ``_vae_decode_nograd`` already forces unconditionally, so a
        paramless stub keeps today's behaviour exactly.

        WP-PIXGAN owns this helper and BOTH decode helpers call it, which
        is what makes the real and the fake side dtype-identical by
        construction (§5.2: the two sides are compared by a texture critic,
        so any precision asymmetry between them is a real/fake tell that
        has nothing to do with texture).
        """
        vae = getattr(getattr(self, "model", None), "vae", None)
        if vae is not None and hasattr(vae, "parameters"):
            for p in vae.parameters():
                return p.dtype
        if vae is not None and hasattr(vae, "buffers"):
            for b in vae.buffers():
                if b.is_floating_point():
                    return b.dtype
        return torch.float32

    def _pix_decode_crops_grad(
        self,
        crops: torch.Tensor,
        *,
        border: int,
        n_frames: int,
        decode_batch: int,
        gen: torch.Generator,
    ) -> torch.Tensor:
        """GRAPH-ON decode of latent crops -> ``[n * k, 3, H, W]`` pixels.

        The fake-side decode, and the ONLY grad-on one.  Uses
        ``_vae_decode_grad`` (checkpointed VAE forward) so gradient flows
        back through the decoder to the latent crop and thence to the
        generator.  ``model/disc_holdout_probe.py::_decode_crops`` is
        ``no_grad`` and is REAL-side only (§5.2 mandates no_grad there).

        ``torch.cuda.empty_cache()`` FIRST, before the decode: the decoder
        wants a large contiguous workspace and the gen rollout's live
        activations leave only fragmented gaps.  Without it the allocation
        fails at step 2+ even with sufficient free memory -- the same
        precedent as ``_maybe_texture_tripwire`` and the LPIPS/perceptual
        decode call site.

        Frame selection happens strictly AFTER decode (docs/WP_PIXGAN.md
        §9), matching ``_decode_crops`` exactly, so real and fake see the
        identical geometry and the identical selection rule.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        vdt = self._pix_vae_dtype()
        outs: List[torch.Tensor] = []
        step = max(1, int(decode_batch))
        for i in range(0, int(crops.shape[0]), step):
            # Cast to the VAE's OWN dtype before the decode (the run-6113341
            # bf16-latent/fp32-conv crash).  ``.to`` is differentiable, so the
            # graph back to the latent crop -- and thence to the generator --
            # survives; that is asserted by
            # ``test_grad_decode_flows_gradient_through_the_dtype_cast``.
            # Cast PER SUB-BATCH, not on the whole crop set, so the widened
            # copy is bounded by ``decode_batch`` rather than ``n_crops``.
            sub = crops[i:i + step].to(vdt)
            pix = self._vae_decode_grad(sub)           # [b, F_pix, 3, H, W]
            b = int(border)
            if b > 0 and pix.shape[-2] > 2 * b and pix.shape[-1] > 2 * b:
                pix = pix[..., b:-b, b:-b]
            f_pix = int(pix.shape[1])
            k = max(1, min(int(n_frames), f_pix))
            sel = torch.randperm(f_pix, generator=gen)[:k].tolist()
            outs.append(
                pix[:, sel].reshape(-1, *pix.shape[2:]).to(torch.float32)
            )
        return torch.cat(outs, dim=0)

    def _pix_decode_crops_nograd(
        self,
        crops: torch.Tensor,
        *,
        border: int,
        n_frames: int,
        decode_batch: int,
        gen: torch.Generator,
    ) -> torch.Tensor:
        """NO-GRAD decode -> ``[n * k, 3, H, W]`` on ``self.device``.

        Thin wrapper over ``model/disc_holdout_probe.py::_decode_crops`` so
        the D-loop and the A22 probe are the same code path byte for byte
        (that helper returns CPU tensors; we move them back to the training
        device).  §5.2: the D-update's decodes are ``no_grad`` with
        DETACHED inputs on BOTH sides -- D is trained on pixels, not
        through the generator.
        """
        from model.disc_holdout_probe import _decode_crops

        # IDENTICAL dtype treatment to the grad twin: same derivation
        # (``_pix_vae_dtype``), same cast, applied before the decode on both
        # sides.  This helper serves BOTH the fake side (bf16 rollout
        # latents) and the real side (fp32 pool draws) of the D-update, so
        # without it the critic would see one dtype for the fake and another
        # for the real -- a tell unrelated to texture.
        px = _decode_crops(
            self,
            crops.detach().to(self._pix_vae_dtype()),
            border=int(border),
            n_frames=int(n_frames),
            decode_batch=int(decode_batch),
            gen=gen,
        )
        return px.to(self.device, torch.float32)

    # ------------------------------------------------------------------
    # §3.1 / A23 -- the FAKE, mask-selected.
    # ------------------------------------------------------------------
    def _pix_select_fake_latents(
        self, info: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Resolve the critic's fake latents from the rollout ``info``.

        Two sources, and A23 IS OVERTURNED (docs/WP_PIXGAN.md §6.0 -- the
        d=0.109 verdict was a single-seed read on ``hv_anisotropy``, a
        voter the campaign's own noise protocol flags as noise-dominated;
        seeds 42/43/44 give .1093/.0113/.0128 and the >=3-seed protocol
        returns EQUIVALENT).  The fake source is therefore an EXPERIMENTAL
        VARIABLE and both paths must work:

          ``pix_finish_grad_enabled=false``  (DEFAULT)
              -> ``info["flash_dmd_gan_x0"]``, the flash t=60 x0.
          ``pix_finish_grad_enabled=true``
              -> ``info["finish_denoised_chunk_grad"]``, the
                 inference-parity ladder endpoint published by T1.

        THE MASK IS MANDATORY on the grad path.  That buffer is partially
        detached BY DESIGN: 14.3 % of frames carry no graph on a 7-block
        rollout (the trailing block of multi-block calls mirrors
        ``flash_grad_active``), plus ~1/K more from blocks that exit at the
        last rung and have no post-exit rung to attach.  Buffer-level
        ``requires_grad`` reads True regardless, so averaging the G-loss
        over the whole chunk would silently dilute the adversarial signal
        by that fraction with nothing saying so.  Absent or all-False mask
        => FAIL LOUD, never a silent fall-through to the whole chunk.

        Returns ``(latents [B, L, C, H, W], logs)`` where ``L`` is a
        CONTIGUOUS run of ``pix_lat_frames_per_crop`` mask-valid latent
        frames.  Contiguity is not cosmetic: the VAE decode is temporal, so
        a gappy frame set would decode differently from the reals' always
        contiguous windows and hand the critic a real/fake tell that has
        nothing to do with texture.
        """
        use_grad_src = bool(
            getattr(self.config, "pix_finish_grad_enabled", False)
        )
        logs: Dict[str, float] = {
            "train/pix_fake_source_ladder": 1.0 if use_grad_src else 0.0,
        }
        # ONE RESOLUTION POINT (see _pix_pool_fill): the resolver already
        # applies the same max(1, .) floor to this key, so re-defaulting it
        # here bought nothing and could only ever disagree.
        want_l = int(self._pix_resolve_cfg()["lat_frames"])

        if not use_grad_src:
            z = info.get("flash_dmd_gan_x0")
            if z is None:
                z = info.get("flash_dmd_gan_x0_raw")
            if z is None:
                raise RuntimeError(
                    "gan_pixel_texture_enabled=true with "
                    "pix_finish_grad_enabled=false requires "
                    "info['flash_dmd_gan_x0'] (the flash t=60 x0). It is "
                    "absent -- enable the flash-DMD path or set "
                    "pix_finish_grad_enabled=true to use the ladder "
                    "endpoint. Refusing to substitute another tensor."
                )
            n_lat = int(z.shape[1])
            l = min(want_l, n_lat)
            start = n_lat - l
            # FORGEABLE-ZERO RULE: pix_finish_grad_frames /
            # _frames_total / _frac are §6.2 measurements OF THE A23 GRAD
            # BUFFER. On the flash path that buffer is never consulted, so
            # emitting them here would report a dilution measurement for a
            # computation that did not run (and "0 no-rung frames" is the
            # HEALTHY reading, which is exactly the forgery that matters).
            # They are OMITTED; ``pix_fake_source_ladder`` = 0.0 is the
            # regime flag that says why they are absent.
            logs["train/pix_fake_lat_frames"] = float(l)
            return z[:, start:start + l], logs

        z = info.get("finish_denoised_chunk_grad")
        if z is None:
            raise RuntimeError(
                "pix_finish_grad_enabled=true but "
                "info['finish_denoised_chunk_grad'] is absent. T1 publishes "
                "it only when its own gate is on; check the pipeline flag. "
                "Refusing to fall back to flash_dmd_gan_x0 silently -- the "
                "fake source is the experimental variable (WP_PIXGAN §6.0) "
                "and must never be swapped behind the arm's back."
            )
        mask = info.get("finish_denoised_chunk_grad_mask")
        if mask is None:
            raise RuntimeError(
                "info['finish_denoised_chunk_grad'] is present but "
                "info['finish_denoised_chunk_grad_mask'] is MISSING. The "
                "grad buffer is partially detached by design (14.3 % of "
                "frames carry no graph on a 7-block rollout, WP_PIXGAN "
                "§6.2); consuming it unmasked silently dilutes the "
                "adversarial signal. Refusing to proceed."
            )
        m = torch.as_tensor(mask).reshape(-1).to(torch.bool)
        n_lat = int(z.shape[1])
        if int(m.numel()) != n_lat:
            raise RuntimeError(
                "finish_denoised_chunk_grad_mask has "
                f"{int(m.numel())} entries but the grad buffer has {n_lat} "
                "latent frames; the mask does not describe this buffer."
            )
        n_valid = int(m.sum().item())
        logs["train/pix_finish_grad_frames"] = float(n_valid)
        logs["train/pix_finish_grad_frames_total"] = float(n_lat)
        logs["train/pix_finish_grad_frac"] = (
            float(n_valid) / float(max(1, n_lat))
        )
        if n_valid == 0:
            raise RuntimeError(
                "finish_denoised_chunk_grad_mask is ALL-FALSE: not one "
                "latent frame in this chunk carries a graph, so the pixel "
                "critic would train on a detached fake and the G-term would "
                "be identically zero. Failing loud (WP_PIXGAN §6.2)."
            )
        # Pick a contiguous mask-valid run. Preference order:
        #   1. the LATEST run long enough to supply ``want_l`` frames --
        #      the ladder endpoint's NEWEST frames are the ones inference
        #      actually commits, so a stale early run is the worse sample
        #      even when it is longer;
        #   2. otherwise the LONGEST run, and take what it has.
        # Then the last ``want_l`` frames of the chosen run.
        runs: List[Tuple[int, int]] = []
        lo = None
        for i in range(n_lat + 1):
            on = i < n_lat and bool(m[i].item())
            if on and lo is None:
                lo = i
            elif not on and lo is not None:
                runs.append((lo, i - lo))
                lo = None
        long_enough = [r for r in runs if r[1] >= want_l]
        if long_enough:
            best_lo, best_len = long_enough[-1]
        else:
            best_lo, best_len = max(runs, key=lambda r: r[1])
        l = min(want_l, best_len)
        start = best_lo + best_len - l
        logs["train/pix_fake_lat_frames"] = float(l)
        logs["train/pix_fake_lat_run"] = float(best_len)
        return z[:, start:start + l], logs

    # ------------------------------------------------------------------
    # §3.4 / A20 / A21 / A22 / A24 -- the REAL supply.
    # ------------------------------------------------------------------
    def _pix_train_ride_paths(self, *, step: Optional[int] = None) -> List[Any]:
        """A22 -- the real supply's ride list, PROVED disjoint from the
        reserved rides.

        Sourced from the post-holdout-filter training ride list (the list
        the ``DistributedSampler`` indexes, i.e. the only thing training
        ever sees) and then re-checked against
        ``disc_holdout_probe._holdout_names`` on every rebuild. A holdout
        list leaked into training once before (GAN_REDESIGN A22), so this
        is a hard failure, not a warning: a leaked reserved ride makes the
        held-out generalisation measurement meaningless while still looking
        healthy.
        """
        from model.disc_holdout_probe import (
            _holdout_names, _train_ride_paths, ride_key,
        )

        paths = list(_train_ride_paths(self))
        if not paths:
            raise RuntimeError(
                "pixel-texture GAN: the real supply needs "
                "trainer.dataset._rides, which is empty/absent. A21 "
                "requires reals drawn from the full training dataset; "
                "refusing to fall back to the current ride window."
            )
        held = set(_holdout_names(self))
        leaked = sorted({ride_key(p) for p in paths} & held)
        self._pix_holdout_checked = 1.0
        self._pix_holdout_n = float(len(held))
        self._pix_holdout_leak = float(len(leaked))
        # FRESHNESS. ``_pix_holdout_leak`` is a SAFETY CLAIM, and a safety
        # claim that is re-published without being re-computed is defeated
        # by staleness rather than by omission -- the same failure reached
        # by a quieter route. This method is the only writer, so it stamps
        # the step it ran at and the D-loop refuses to publish a value that
        # was not recomputed for the step it is logging.
        if step is not None:
            self._pix_holdout_checked_step = int(step)
        if leaked:
            raise RuntimeError(
                "A22 VIOLATION: %d reserved ride(s) are present in the "
                "pixel critic's real supply: %s. Reserved rides must be "
                "provably absent from training reals."
                % (len(leaked), leaked[:8])
            )
        return paths

    def _pix_latent_len(self, path: Any) -> int:
        cache = getattr(self, "_pix_lat_len_cache", None)
        if cache is None:
            cache = {}
            self._pix_lat_len_cache = cache
        key = str(path)
        if key not in cache:
            from model.disc_holdout_probe import _latent_len
            try:
                cache[key] = int(_latent_len(path))
            except Exception:
                # Do NOT cache the failure: a transient zarr hiccup would
                # otherwise blacklist a ride for the rest of the run.
                return 0
        return int(cache[key])

    def _pix_pool_fill(
        self,
        n_new: int,
        *,
        gen: torch.Generator,
        crop_rows: int,
        crop_cols: int,
        n_bands: int,
        lat_frames: int,
        force_band: Optional[int] = None,
        step: Optional[int] = None,
    ) -> int:
        """Admit ``n_new`` fresh real windows into the A21 support pool.

        A21 prefers NO cache -- fresh GT latent frames from the full
        training dataset per D update, via
        ``ZarrRideDataset.load_latent_chunk`` (the primitive
        ``disc_holdout_probe.build_latent_pool`` itself uses). At
        ``pix_crops_per_step=4``/``pix_frames_per_crop=3`` A20 needs 12
        independently sourced reals per D-update and
        ``pix_gan_updates_per_step=5``, i.e. 60 zarr window loads per
        training step, which puts the loader squarely on the critical path
        (WP_PIXGAN §3). So a cache is permitted -- but only under A21's
        terms, all of which are enforced here:

          * >= ``PIX_A21_SUPPORT_FLOOR`` (4,096) DISTINCT source frames,
            8-16k preferred (``pix_real_pool_windows`` x
            ``pix_lat_frames_per_crop``);
          * CONTINUOUSLY REFRESHED -- ``pix_real_pool_refresh`` fresh
            windows admitted per D-update FOREVER, including after the
            pool has reached ``pix_real_pool_windows``, at which point the
            FIFO branch below starts evicting the oldest window per
            admission. Admission is the ONLY thing that grows A21 support,
            so stopping at the cap would freeze support at the cap's
            ceiling -- ``pix_real_pool_windows x pix_lat_frames_per_crop``,
            which at the shipped 2,048 x 2 was 4,096 EXACTLY the floor, and
            below it in practice because windows overlap inside a ride.
            Support would then never pass the floor, the pool would be the
            frozen mini-dataset A21 forbids, and the A22 holdout check
            (reached only from here) would go stale. One bug, three
            symptoms; the cure is that the cap bounds MEMORY, never
            refresh;
          * CROSS-RIDE -- the ride is redrawn per window;
          * UNIFORM WITHIN THE BAND -- never similarity- or
            nearest-matched (retrieval-matched reals are scope-frozen).

        Each entry is cropped AT LOAD to one A24 band, so the pool is
        band-stratified and the paired draw in :meth:`_pix_draw_reals`
        can always honour "same band as its fake" exactly.

        Storage is fp16 on CPU: 2,048 windows x 2 latent frames x 16 ch x
        24 x 32 = ~100 MB, versus ~1.6 GB if full 60x104 frames were kept.
        """
        from utils.zarr_dataset import ZarrRideDataset
        from model.disc_holdout_probe import band_plan, ride_key

        paths = self._pix_train_ride_paths(step=step)
        pool = getattr(self, "_pix_real_pool", None)
        if pool is None:
            pool = []
            self._pix_real_pool = pool
        support = getattr(self, "_pix_real_support", None)
        if support is None:
            support = set()
            self._pix_real_support = support
        rides_seen = getattr(self, "_pix_real_rides", None)
        if rides_seen is None:
            rides_seen = set()
            self._pix_real_rides = rides_seen

        lf = max(1, int(lat_frames))
        admitted = 0
        attempts = 0
        max_attempts = max(8, int(n_new) * 8)
        while admitted < int(n_new) and attempts < max_attempts:
            attempts += 1
            pi = int(torch.randint(0, len(paths), (1,), generator=gen).item())
            p = paths[pi]
            n_lat = self._pix_latent_len(p)
            if n_lat < lf:
                continue
            hi = n_lat - lf
            start = (
                0 if hi <= 0
                else int(torch.randint(0, hi + 1, (1,), generator=gen).item())
            )
            try:
                lat = ZarrRideDataset.load_latent_chunk(
                    str(p), start, start + lf,
                )
            except Exception as exc:                      # pragma: no cover
                logging.warning(
                    "[pixgan] load_latent_chunk %s failed: %r", p, exc,
                )
                continue
            lat = torch.as_tensor(lat)
            if int(lat.shape[0]) != lf:
                continue
            h, w = int(lat.shape[-2]), int(lat.shape[-1])
            cr = min(int(crop_rows), h)
            cc = min(int(crop_cols), w)
            if force_band is None:
                b_list, y_list = band_plan(1, h, cr, int(n_bands), gen)
                band, y0 = int(b_list[0]), int(y_list[0])
            else:
                band = int(force_band)
                _b, y_list = band_plan(1, h, cr, int(n_bands), gen)
                # Re-draw the offset inside the FORCED band, using the same
                # bin arithmetic band_plan uses (A24 thirds), so a forced
                # band is geometrically indistinguishable from a drawn one.
                max_off = max(0, h - cr)
                nb = max(1, int(n_bands))
                lo = int(round(band * max_off / nb))
                hib = max(int(round((band + 1) * max_off / nb)), lo)
                y0 = (
                    lo if hib <= lo
                    else int(torch.randint(
                        lo, hib + 1, (1,), generator=gen).item())
                )
            x_hi = max(1, w - cc + 1)
            x0 = int(torch.randint(0, x_hi, (1,), generator=gen).item())
            y0 = max(0, min(int(y0), h - cr))
            crop = lat[:, :, y0:y0 + cr, x0:x0 + cc].to(torch.float16)
            rk = ride_key(p)
            pool.append({
                "lat": crop,
                "band": int(band),
                "y0": int(y0),
                "x0": int(x0),
                "ride": rk,
                "start": int(start),
                # The window's SOURCE FRAME span. A20's "independent" is
                # about source frames, not pool indices, and two windows
                # whose spans intersect are not two independent reals -- so
                # the span has to be recorded at admission, not guessed at
                # draw time from a per-call ``lat_frames``.
                "nf": int(lf),
                "uid": (rk, int(start), int(y0), int(x0)),
            })
            rides_seen.add(rk)
            # A21 support is a MEASURED claim: count distinct (ride, latent
            # frame index) identities actually admitted, never infer it from
            # the loader's nominal size.
            if len(support) < 400000:
                for j in range(lf):
                    support.add((rk, int(start) + j))
            admitted += 1
            # §7 A21 -- MONOTONE cache-refresh counter (never a per-step
            # gauge: the per-step gauge is exactly what aliased to a
            # permanent 0 and produced the "R2 never fired" false alarm).
            self._pix_pool_refresh_total = int(
                getattr(self, "_pix_pool_refresh_total", 0)
            ) + 1

        # ONE RESOLUTION POINT. This used to read pix_real_pool_windows
        # with its OWN getattr default (2048) while _pix_resolve_cfg --
        # the single resolve-and-validate point for the whole pix_* block,
        # called from the constructor so a misconfigured arm fails early
        # -- resolved the same key independently. Two resolution points
        # that can disagree, with nothing enforcing agreement, is exactly
        # the A21 blocker shape: with the key present both said 4096, but
        # an override typo or a config predating the block would silently
        # halve A21's ceiling here to 2048 while the VALIDATED value, the
        # A21 margin warning and the resolved-config echo all still said
        # otherwise. Resolve through the resolver; never re-default.
        cap = max(int(n_new), int(self._pix_resolve_cfg()["pool_windows"]))
        if len(pool) > cap:                       # FIFO -> continuous refresh
            del pool[: len(pool) - cap]
        return admitted

    def _pix_draw_reals(
        self,
        bands: Sequence[int],
        *,
        gen: torch.Generator,
        crop_rows: int,
        crop_cols: int,
        n_bands: int,
        lat_frames: int,
        reuse_horizon: Optional[int] = None,
        step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[int], Dict[str, float]]:
        """A20 + A24 -- one independently sourced real per fake, same band.

        ``bands`` is the fake side's per-IMAGE band list (length =
        ``pix_crops_per_step * pix_frames_per_crop *
        pix_reals_per_fake``). For every entry we draw a pool window whose
        band EQUALS that fake's band; if the band is short of stock we load
        fresh windows forced into that band rather than relaxing the
        pairing. The returned ``real_bands`` are then compared elementwise
        by the caller and the mismatch counter must be 0.

        "Independent" is the strong reading (docs/WP_PIXGAN.md §2, A20):
        the SOURCE FRAME is drawn independently. Three frames from one
        decoded crop, or from one ride window, are NOT three independent
        reals -- so this returns one image per entry, each from its own
        (ride, start, y0, x0) window. Frame-within-crop expansion is a
        FAKE-SIDE device only and must never leak here.

        ``pix_real_repeat_frac`` MEASURES that, and it is measured at the
        level A20 is about. It used to be computed as
        ``len(set(uids)) < len(uids)`` over indices drawn WITHOUT
        REPLACEMENT (the ``used`` set below), which is 0 by construction:
        two distinct pool indices can only share a uid if the identical
        window was admitted twice, so the gauge could not move and the
        guard reading it could not fire. It now counts reals whose SOURCE
        FRAME SPAN -- ``{(ride, start+j)}`` over the window's ``nf``
        frames, recorded at admission -- intersects that of an earlier
        real in the SAME D-update. Overlapping spans are the real hazard:
        a window at ``s`` and a window at ``s+1`` decode largely the same
        GT, and if frame-within-crop expansion ever leaked to the real
        side every crop's ``k_frames`` images would share ONE span.

        Because the gauge now measures something real it is no longer
        required to be exactly 0. Independent sampling from a finite
        support produces occasional coincidences -- demanding zero is
        demanding WITHOUT-replacement sampling, i.e. the opposite of
        independence -- so the caller's guard trips at
        ``PIX_A20_REPEAT_ALARM`` (half the batch), which coincidence
        cannot reach and a structural leak reaches immediately.

        ``pix_real_recent_reuse_frac`` is the same measurement across
        TIME: the share of this update's reals whose span was already
        drawn within the last ``reuse_horizon`` D-updates. That is the
        memorisation-facing number -- a pool that has stopped refreshing
        keeps re-serving the same spans and this rises, while a pool under
        continuous refresh holds it near the sampling-coincidence rate.

        Exact-y matching, per-row bands, bands that scale with row count
        and +-tolerance matching are FORBIDDEN AND PERMANENT (A24). This
        function only ever compares the coarse band INDEX.
        """
        n_req = len(bands)
        pool = getattr(self, "_pix_real_pool", None) or []
        by_band: Dict[int, List[int]] = {}
        for i, e in enumerate(pool):
            by_band.setdefault(int(e["band"]), []).append(i)

        need: Dict[int, int] = {}
        for b in bands:
            need[int(b)] = need.get(int(b), 0) + 1
        for b, k in need.items():
            have = len(by_band.get(int(b), ()))
            if have < k:
                self._pix_pool_fill(
                    k - have, gen=gen, crop_rows=crop_rows,
                    crop_cols=crop_cols, n_bands=n_bands,
                    lat_frames=lat_frames, force_band=int(b), step=step,
                )
        pool = getattr(self, "_pix_real_pool", None) or []
        by_band = {}
        for i, e in enumerate(pool):
            by_band.setdefault(int(e["band"]), []).append(i)

        used: set = set()
        chosen: List[int] = []
        for b in bands:
            cand = [i for i in by_band.get(int(b), ()) if i not in used]
            if not cand:
                raise RuntimeError(
                    "pixel-texture GAN: no un-used real window left in band "
                    f"{int(b)} (pool={len(pool)}). A20 forbids reusing a real "
                    "inside one D-update; raise pix_real_pool_windows."
                )
            j = int(torch.randint(0, len(cand), (1,), generator=gen).item())
            idx = int(cand[j])
            used.add(idx)
            chosen.append(idx)

        lat = torch.stack(
            [pool[i]["lat"].to(torch.float32) for i in chosen], dim=0,
        ).to(self.device)
        real_bands = [int(pool[i]["band"]) for i in chosen]
        uids = [pool[i]["uid"] for i in chosen]
        n_distinct = len(set(uids))
        n_windows = len({(u[0], u[1]) for u in uids})
        n_rides = len({u[0] for u in uids})

        # --- A20, measured on SOURCE FRAMES (see the docstring) ---------
        frame_sets: List[set] = [
            {
                (pool[i]["ride"], int(pool[i]["start"]) + j)
                for j in range(max(1, int(pool[i].get("nf", 1))))
            }
            for i in chosen
        ]
        seen_frames: set = set()
        n_overlap = 0
        for fs in frame_sets:
            if fs & seen_frames:
                n_overlap += 1
            seen_frames |= fs
        repeat_frac = float(n_overlap) / float(max(1, n_req))

        # --- the same thing across TIME: rolling source-frame reuse -----
        horizon = int(
            self.PIX_REAL_REUSE_HORIZON if reuse_horizon is None
            else reuse_horizon
        )
        horizon = max(1, horizon)
        ring = getattr(self, "_pix_recent_frames", None)
        if ring is None:
            ring = []
            self._pix_recent_frames = ring
        prior: set = set()
        for older in ring:
            prior |= older
        n_reuse = sum(1 for fs in frame_sets if fs & prior)
        reuse_frac = float(n_reuse) / float(max(1, n_req))
        ring.append(set(seen_frames))
        if len(ring) > horizon:
            del ring[: len(ring) - horizon]

        logs = {
            # A20 -- genuine source-frame overlap inside this D-update.
            "train/pix_real_repeat_frac": repeat_frac,
            # A20/A21 -- genuine source-frame reuse over the last
            # ``horizon`` D-updates. Rises when the pool stops refreshing.
            "train/pix_real_recent_reuse_frac": reuse_frac,
            # How many D-updates that reuse figure looks back over. Emitted
            # so an early, near-empty ring is not misread as a clean
            # history (it is a SHORT one).
            "train/pix_real_reuse_span_updates": float(len(ring)),
            "train/pix_real_reuse_span_frames": float(len(prior)),
            # STRUCTURALLY PINNED, kept only as a build assertion: the draw
            # below is without replacement over pool INDICES, so distinct
            # uids are guaranteed and this equals the request size unless
            # the draw itself is broken. It is NOT A20 evidence -- that is
            # ``pix_real_repeat_frac`` above.
            "train/pix_real_distinct_sources": float(n_distinct),
            # Distinct SOURCE FRAMES behind this update's reals: the union
            # of the drawn spans, so ``n_req * nf`` when nothing overlaps
            # and less when it does.
            "train/pix_real_unique_frames": float(len(seen_frames)),
            "train/pix_real_unique_windows": float(n_windows),
            "train/pix_real_unique_rides": float(n_rides),
            "train/pix_real_pool_windows": float(len(pool)),
            # §7 A21 "if a cache is used at all": the pool IS the cache.
            "train/pix_real_cache_size": float(len(pool)),
            "train/pix_real_cache_refresh_total": float(
                getattr(self, "_pix_pool_refresh_total", 0)
            ),
            "train/pix_real_support_frames": float(
                len(getattr(self, "_pix_real_support", ()) or ())
            ),
            "train/pix_real_support_rides": float(
                len(getattr(self, "_pix_real_rides", ()) or ())
            ),
        }
        return lat, real_bands, logs

    # ------------------------------------------------------------------
    # §5.2 / §5.3 -- the D-update loop.
    # ------------------------------------------------------------------
    def _pix_disc_module(self) -> torch.nn.Module:
        """DDP wrapper if there is one, raw module otherwise."""
        d = getattr(self, "pixel_texture_disc_ddp", None)
        return d if d is not None else self.pixel_texture_disc

    def _pix_counters(self):
        c = getattr(self, "_pix_counters_obj", None)
        if c is None:
            from model.pixel_texture_disc import PixCounters
            # pix_r1_every_n = 1: R1 fires on EVERY D-update, target
            # pix_r1_rate == 1.00. Cost is bounded by pix_r1_num_samples
            # subsampling, NEVER by raising every_n -- that reintroduces the
            # silent-cadence bug class (A10/A16, TEXTURE_GAN_DESIGN §5.4).
            c = PixCounters(
                r1_every_n=int(self._pix_resolve_cfg()["r1_every_n"])
            )
            self._pix_counters_obj = c
        return c

    def _pix_optimizer_hyper(self) -> Dict[str, Any]:
        """The pixel critic's LIVE optimizer hyper-parameters, for the echo.

        Standing rule 4: a launch line's REQUESTED value and the EFFECTIVE
        value can differ, and only the effective one settles anything. The
        rule exists because two sessions independently reached a wrong
        conclusion about a micro-batch knob whose effective value was
        assembled across two files.

        ``pix_gan_lr`` / ``pix_gan_betas`` are the place in the
        resolved-config echo where that gap can open with NO config
        involved at all. ``_build_optimizer`` is what turned the config
        into the ``param_groups`` entry that actually steps D; everything
        after construction -- a resume restoring an optimizer state dict, a
        scheduler, a hand-edited param group -- moves the live value while
        the config sits unchanged. An echo sourced from ``cfg`` then
        reports a number nothing uses, WHILE LOOKING AUTHORITATIVE: rule 4
        defeated by the exact class rule 4 was written to catch.

        So these are read off the optimizer, and two situations are refused
        a plausible-looking number:

          * no optimizer / no param groups / a non-finite or missing entry
            -> the value is OMITTED and ``<name>_unavailable`` is emitted.
            ``PIX_GAN_LR`` is the constructor's default, not an
            observation; printing it when nothing was built is a forged
            reading of the kind §21 keeps producing.
          * param groups that DISAGREE -> the single value is omitted,
            ``<name>_param_group_disagreement`` + ``<name>_min`` /
            ``<name>_max`` are emitted, and a WARNING is logged. Reporting
            ``param_groups[0]`` would hide the split behind a number that
            is true of exactly one group; the split IS the finding.

        Freshly computed at every emission -- never cached, so the flags
        describe the optimizer as it is when the echo is written.
        """
        opt = getattr(self, "pix_optimizer", None)
        groups = list(getattr(opt, "param_groups", None) or ())
        out: Dict[str, Any] = {
            "pix_gan_optimizer_param_groups": int(len(groups)),
        }

        def _emit(name: str, pick) -> None:
            vals: List[float] = []
            for g in groups:
                try:
                    v = float(pick(g))
                except (KeyError, IndexError, TypeError, ValueError):
                    out[f"{name}_unavailable"] = True
                    return
                if not math.isfinite(v):
                    out[f"{name}_unavailable"] = True
                    return
                vals.append(v)
            if not vals:
                out[f"{name}_unavailable"] = True
                return
            uniq = sorted(set(vals))
            if len(uniq) > 1:
                out[f"{name}_param_group_disagreement"] = True
                out[f"{name}_min"] = uniq[0]
                out[f"{name}_max"] = uniq[-1]
                logging.warning(
                    "[pixgan] %s DIFFERS across the pixel optimizer's %d "
                    "param groups (%s). No single value is echoed; "
                    "train/pix_cfg_%s_min / _max carry the split.",
                    name, len(groups),
                    ", ".join(repr(v) for v in uniq), name,
                )
                return
            out[name] = uniq[0]

        _emit("pix_gan_lr", lambda g: g["lr"])
        _emit("pix_gan_beta1", lambda g: g["betas"][0])
        _emit("pix_gan_beta2", lambda g: g["betas"][1])
        return out

    def _pix_emit_actionable(self, msg: str) -> None:
        """Emit an ACTIONABLE ``pix_*`` line on a path that SURVIVES.

        The rule-4 instrument was itself an instance of the defect class it
        was built to catch: computed correctly, wired to a channel that
        does not reach the run log.

        ``trainer/causal_rolling_staircase_train.py`` configures logging
        with::

            level = logging.INFO if self.is_main_process else logging.WARNING
            logging.basicConfig(level=level, format=...)

        with NO ``force=True``. ``logging.basicConfig`` is a documented
        SILENT NO-OP when the root logger already has handlers, so if any
        import along the way (a library, the launcher, torch/DDP) installs
        a root handler first, the level is never applied, root stays at
        Python's default WARNING, and every ``logging.info`` in the trainer
        vanishes. Two full smoke logs tonight had 0 INFO hits with
        WARNING/ERROR unaffected -- see the ``[ROLL]`` / ``[LADD-DIAG]``
        notes at the LADD disc build, which already carry this pattern.

        Whether it bites depends on IMPORT ORDER. An instrument whose
        visibility is decided by import order is not an instrument. So this
        does not gamble on the root level:

          * ``logging.warning`` -- survives a WARNING-level root, and keeps
            the line in whatever handler/formatter/file the run configured.
          * ``print(..., file=sys.stderr, flush=True)`` -- survives even a
            root with NO usable handler at all, which the logging leg
            cannot. Same belt-and-braces pair as
            ``pipeline/action_forcing_training.py:701-702``.

        NOT fixed by adding ``force=True`` upstream: that file is not this
        package's, other packages depend on its behaviour, and forcing
        would clobber a root handler someone installed deliberately.

        Main-process gated HERE as well as at every call site, so a future
        caller cannot turn this into per-rank stderr spam. Callers own
        once-per-run latching (the echo's ``_pix_cfg_echoed``). ``msg`` is
        pre-rendered and passed with NO logging args, so a ``%`` inside a
        resolved value cannot raise in the formatter.
        """
        if not getattr(self, "is_main_process", True):
            return
        logging.warning(msg)
        print(msg, file=sys.stderr, flush=True)

    def _pix_echo_resolved_config(
        self,
        logs: Dict[str, float],
        *,
        resolved: Dict[str, Any],
    ) -> None:
        """ONE-TIME echo of the RESOLVED value of every ``pix_*`` knob.

        Not a nicety. An hour-old incident in this campaign: two sessions
        independently concluded a 14B smoke's
        ``ladd_gen_guidance_micro_batch_groups`` had fallen back to its
        default because the launch script's DEXTRA did not name it. Both
        were wrong -- the script launches through a HOLDER whose own fixed
        arg list passes the knob one level up, and the resolved value was 4
        all along. Every launcher-level check either session made was
        accurate and the conclusion still came out wrong, because the
        effective value is assembled across two files. **Only the resolved
        value settles it.**

        B1 has the identical exposure: ``pix_decode_batch`` controls the
        grad-decode micro-batch size (the expensive, G-side memory knob) and
        was previously visible only through ``train/pix_decodes``, a count.

        DERIVED, NOT REQUESTED. Where a value is clamped by a tensor
        dimension the DERIVED value is echoed -- ``crop_rows`` after
        ``min(pix_crop_lat[0], H_lat)``, the border after ``_decode_crops``
        declines to trim a too-small crop, the latent-frame count after the
        mask-valid run bounds it, the pixel-frame count after
        ``min(pix_frames_per_crop, F_pix)``. The gap between requested and
        derived is the entire point, so both are emitted where they can
        differ.

        ``pix_band_count`` is echoed explicitly every run (§7): a silently
        tightened banding must be visible in the TRACE, not only in a diff.

        Main process only, once per run (``_pix_cfg_echoed`` latch), and
        entirely inside the ``gan_pixel_texture_enabled`` gate, so
        byte-identical-off is unaffected. ``pix_r1_num_samples=None`` is
        echoed as **-1**, a sentinel OUTSIDE the knob's meaningful range
        (>= 1), never as 0 -- 0 would read as a real setting.

        EMITTED VIA ``_pix_emit_actionable``, never ``logging.info``: see
        that helper for why an INFO-only echo is invisible in production
        for reasons that come down to import order.
        """
        if getattr(self, "_pix_cfg_echoed", False):
            return
        self._pix_cfg_echoed = True
        for k, v in resolved.items():
            if isinstance(v, bool):
                logs[f"train/pix_cfg_{k}"] = 1.0 if v else 0.0
            elif isinstance(v, (int, float)):
                logs[f"train/pix_cfg_{k}"] = float(v)
        if not getattr(self, "is_main_process", True):
            return
        self._pix_emit_actionable(
            "[ActionForcing] WP-PIXGAN resolved config (DERIVED values, "
            "not launch-line requests): "
            + " ".join(f"{k}={resolved[k]}" for k in sorted(resolved))
        )

    # ------------------------------------------------------------------
    # §5.5 / §12 -- the GENERATOR-side pixel term.
    # ------------------------------------------------------------------
    def _pix_emit_grad_ratio(
        self,
        out: Dict[str, float],
        *,
        pix_raw: torch.Tensor,
        dmd_vec: Optional[torch.Tensor],
        param: torch.Tensor,
    ) -> None:
        """Publish the UNWEIGHTED pixel grad ratio (§12 step 1).

        ``pix_raw`` is the PRE-WEIGHT ``g_loss``, so this is the ratio the
        pixel term WOULD have at weight 1.0. That is the number the §12
        calibration needs and the one a weight-free probe cannot get any
        other way: at ``pix_gan_weight=0`` the applied term contributes no
        gradient at all, so every weighted readout is exactly 0.0 and
        settles nothing. The ratio is linear in the weight, so
        ``pix_gan_weight = 0.10 / r`` lands §5.5's 5-20 % band directly.

        ``dmd_vec`` is PASSED IN, never recomputed: one backward of the
        shared DMD term serves the A7 ratio, the A7 cosine, this ratio and
        this cosine. Recomputing it would re-run a checkpointed teacher
        recompute (~11 s / ~37 GiB on the LADD path) for a vector already
        in hand -- which is exactly why ``grad_at`` returns VECTORS and not
        ratios.

        DENOMINATOR RULE, and it is the point of the whole helper: when the
        denominator is zero the ratio is OMITTED and a distinct reason key
        is emitted instead. The neighbouring A7 line does
        ``_ng / _nr if _nr > 0 else 0.0``, and that 0.0 cannot be told
        apart from "the GAN term contributes nothing" -- opposite
        conclusions, identical logged number, and the second one is a live
        disputed claim in this campaign. The A7 line's behaviour is
        UNCHANGED (recorded runs depend on it); this one does not copy it.
        """
        pg = grad_at(pix_raw, param, retain_graph=True)
        if pg is None:
            # NOT 0.0 -- a forged zero (docs/WP_PIXGAN.md §21 instance 3,
            # where a placeholder 0.0 was even pinned by a test).
            out["train/pix_gan_grad_unavailable"] = 1.0
            return
        n_pg = float(pg.norm())
        out["train/pix_gan_grad_norm_unweighted"] = n_pg
        if dmd_vec is None:
            out["train/pix_gan_grad_denom_unavailable"] = 1.0
            return
        n_dmd = float(dmd_vec.norm())
        if n_dmd <= 0.0:
            out["train/pix_gan_grad_denom_zero"] = 1.0
            out["train/pix_gan_grad_cos_undefined"] = 1.0
            return
        out["train/pix_gan_grad_ratio"] = n_pg / n_dmd
        if n_pg > 0.0:
            out["train/pix_gan_grad_cos"] = float(
                torch.dot(pg, dmd_vec) / (n_pg * n_dmd)
            )
        else:
            out["train/pix_gan_grad_cos_undefined"] = 1.0

    def _pix_standalone_grad_telemetry(
        self,
        generator_loss: torch.Tensor,
        pix_raw: torch.Tensor,
        out: Dict[str, float],
        *,
        current_step: int,
    ) -> None:
        """A7-equivalent probe for the arm that runs with the LADD GAN OFF.

        Same cadence knob (``gan_grad_telemetry_every``), same anchor (the
        last ``requires_grad`` generator parameter), same helper for the
        ratio -- so the pixel arm's calibration read is identical whether
        or not the transition GAN happens to be enabled, instead of being
        available on one path only. try/except with the A7 error key so
        telemetry can never take a run down.
        """
        tel = int(getattr(self.config, "gan_grad_telemetry_every", 25) or 0)
        if tel <= 0 or int(current_step) % tel != 0:
            return
        try:
            p_last = None
            for _pn, _pp in self.model.generator.named_parameters():
                if _pp.requires_grad:
                    p_last = _pp
            if p_last is None:
                return
            dmd_vec = grad_at(generator_loss, p_last, retain_graph=True)
            if dmd_vec is not None:
                out["train/dmd_grad_norm_shared"] = float(dmd_vec.norm())
            self._pix_emit_grad_ratio(
                out, pix_raw=pix_raw, dmd_vec=dmd_vec, param=p_last,
            )
        except Exception:
            out["train/gan_grad_telemetry_err"] = 1.0

    def _pix_positive_control(
        self,
        real_lat: torch.Tensor,
        student_px: torch.Tensor,
        disc: torch.nn.Module,
        *,
        current_step: int,
        update_idx: int,
        border: int,
        k_frames: int,
        decode_batch: int,
    ) -> Dict[str, float]:
        """§8.1 positive control -- makes ``pix_d_loss ~ ln 2`` DECIDABLE.

        ``d_loss ~ ln 2`` reads identically for "critic undertrained" and
        "design wrong", and that ambiguity has cost this campaign
        repeatedly. Every ``pix_poscontrol_every`` steps this scores, under
        ``no_grad`` and off the training graph, GT vs a C1-corrupted GT
        (the control) and GT vs the student (the arm), and publishes the
        §8.1 decision-table inputs plus ``pix_poscontrol_rank_ok``.

        The clean and corrupted decodes use two generators seeded
        IDENTICALLY (salt 23), so the post-decode frame selection is the
        same on both sides and the pair differs only by the C1
        perturbation -- the whole readout is a paired comparison and an
        unmatched frame subset would silently confound it.

        DEFAULT OFF (``pix_poscontrol_every = 0``) and, when off, not a
        single tensor, decode or RNG draw happens here. On off-cadence
        steps NOTHING is emitted rather than a stale or zeroed row.
        """
        every = int(getattr(self.config, "pix_poscontrol_every", 0) or 0)
        if every <= 0 or int(update_idx) != 0:
            return {}
        if int(current_step) % every != 0:
            return {}
        from model.pixel_texture_disc import (
            c1_structured_hf, positive_control_readout,
        )

        amp = float(getattr(self.config, "pix_poscontrol_amplitude", 0.5))
        n_boot = int(getattr(self.config, "pix_poscontrol_boot", 1000))
        try:
            with torch.no_grad():
                g_clean = self._pix_sync_generator(
                    int(current_step), int(update_idx), salt=23)
                g_corr = self._pix_sync_generator(
                    int(current_step), int(update_idx), salt=23)
                g_c1 = self._pix_sync_generator(
                    int(current_step), int(update_idx), salt=29)
                corr_lat = c1_structured_hf(real_lat, amp, generator=g_c1)
                clean_px = self._pix_decode_crops_nograd(
                    real_lat, border=border, n_frames=k_frames,
                    decode_batch=decode_batch, gen=g_clean,
                )
                corr_px = self._pix_decode_crops_nograd(
                    corr_lat, border=border, n_frames=k_frames,
                    decode_batch=decode_batch, gen=g_corr,
                )
                readout = positive_control_readout(
                    disc, clean_px, corr_px, student_px=student_px,
                    n_boot=n_boot, generator=g_c1,
                )
            out = {f"train/{k}": float(v) for k, v in readout.items()}
            out["train/pix_poscontrol_amplitude"] = amp
            return out
        except Exception:
            # A diagnostic must never take a training run down, and it must
            # not fake a reading either: the error key is DISTINCT from any
            # separation value (A7 precedent).
            return {"train/pix_poscontrol_err": 1.0}

    def _pix_r1_grad_share(
        self,
        d_term: torch.Tensor,
        r1_term: torch.Tensor,
        *,
        current_step: int,
    ) -> Dict[str, float]:
        """R1's SHARE of the D-side gradient -- not just its magnitude.

        §7 requires ``pix_r1_grad_sq``, pre-γ and mean-reduced. That is a
        MAGNITUDE, and R1 is the one term in this build where too-weak and
        too-strong have OPPOSITE failure modes -- an unregularised D that
        runs away, versus a D pinned so hard it cannot separate real from
        fake at all. **No value of a magnitude distinguishes those two**:
        the same ``pix_r1_grad_sq`` reads high whether γ is doing useful
        work or crushing the critic, because the number says nothing about
        what it is being compared against. A SHARE does:

            share = ||∇_θ R1|| / (||∇_θ R1|| + ||∇_θ d_loss||)

        -> ~0 means the penalty is decorative (unregularised regime);
        -> ~1 means the penalty owns the D step (pinned regime);
        and the accompanying cosine says whether R1 pulls WITH or AGAINST
        the separation direction. This is docs/WP_PIXGAN.md §21's named
        pattern (#1 and #2 are both "value, not share") applied before it
        can bite a third time.

        Measured POST-γ deliberately: the question is what the penalty AS
        APPLIED does to the step. ``pix_r1_grad_sq`` stays pre-γ per §7, so
        both readings are available and neither is inferable from the other.

        Cadenced with the A7 probe (``gan_grad_telemetry_every``) because
        it costs two extra backwards through the critic; on off-cadence
        steps NOTHING is emitted -- a missing key honestly reads "not
        measured" (§16/§19), where a 0.0 would read "decorative penalty".
        Anchored on one parameter (the critic's last ``requires_grad``
        param), exactly like the A7 probe, which is what keeps it cheap.
        """
        tel = int(getattr(self.config, "gan_grad_telemetry_every", 25) or 0)
        if tel <= 0 or int(current_step) % tel != 0:
            return {}
        try:
            p_last = None
            for _p in self.pixel_texture_disc.parameters():
                if _p.requires_grad:
                    p_last = _p
            if p_last is None:
                return {"train/pix_r1_grad_share_unavailable": 1.0}
            g_r1 = grad_at(r1_term, p_last, retain_graph=True)
            g_d = grad_at(d_term, p_last, retain_graph=True)
            if g_r1 is None or g_d is None:
                return {"train/pix_r1_grad_share_unavailable": 1.0}
            n_r1 = float(g_r1.norm())
            n_d = float(g_d.norm())
            logs = {
                "train/pix_r1_grad_norm": n_r1,
                "train/pix_d_grad_norm": n_d,
            }
            den = n_r1 + n_d
            if den > 0.0:
                logs["train/pix_r1_grad_share"] = n_r1 / den
            else:
                # OMITTED, not 0.0: 0.0 is the "decorative penalty"
                # reading and would be indistinguishable from "both
                # gradients vanished".
                logs["train/pix_r1_grad_share_denom_zero"] = 1.0
            if n_r1 > 0.0 and n_d > 0.0:
                logs["train/pix_r1_d_grad_cos"] = float(
                    torch.dot(g_r1, g_d) / (n_r1 * n_d)
                )
            return logs
        except Exception:
            # Telemetry can never take a training run down (A7 precedent).
            return {"train/pix_r1_grad_share_err": 1.0}

    def _pix_band_histogram(
        self,
        bands: Sequence[int],
        n_bands: int,
        *,
        prefix: str = "pix_",
    ) -> Dict[str, float]:
        """§7 crop-band histogram -- fraction of crops in each A24 band.

        Emitted as a COMPLETE set or not at all, so a ``0.0`` here is the
        honest reading "no crop landed in this band" rather than a
        placeholder for a computation that did not run: the set's presence
        is itself the regime flag (§16/§19).
        """
        n = max(1, len(list(bands)))
        counts = [0] * max(1, int(n_bands))
        for b in bands:
            i = int(b)
            if 0 <= i < len(counts):
                counts[i] += 1
        out = {
            f"train/{prefix}band_frac_{i}": float(c) / float(n)
            for i, c in enumerate(counts)
        }
        out[f"train/{prefix}band_count"] = float(int(n_bands))
        return out

    def _pix_g_snapshot_disc(self) -> torch.nn.Module:
        """The critic COPY the G-term scores against, and why one is needed.

        MEASURED, not assumed: the G-term forward happens in the gen-loss
        block, T3-B's D-loop calls ``pix_optimizer.step()`` later in the
        SAME iteration, and ``generator_loss.backward()`` runs later still.
        Adam's step is in-place on the critic weights, so backward through
        a graph built on the live module raises

            "one of the variables needed for gradient computation has been
            modified by an inplace operation ... expected version 1"

        (reproduced on CPU before this was written; it fires even with the
        critic's ``requires_grad`` set False, because the version check is
        on the SAVED tensor, not on its grad-ness).  The LADD path dodges
        the same hazard by deferring its disc update
        (``ladd_defer_disc_update``); the pixel path cannot, because the
        D-loop's position is T3-B's and the A22 probe reads after it.

        So the G-term scores a snapshot refreshed from the live critic at
        the top of every G-term call.  Three consequences, all wanted:
          * the later in-place optimizer step cannot corrupt this graph;
          * NO gradient ever lands on the real critic's ``.grad`` (the
            snapshot's params are ``requires_grad=False``), so the G
            backward cannot pollute the next D-update;
          * G is trained against D as of the start of the step -- ordinary
            alternating-GAN semantics.

        ``eval()`` because the only mode-dependent piece in
        ``PixelTextureDisc`` is spectral norm's power iteration (GroupNorm
        and LeakyReLU are mode-independent, and there is no dropout or
        batchnorm): eval reuses the live critic's current ``u``/``v``, so
        the snapshot forward is deterministic and free of side effects.
        """
        import copy

        disc = self.pixel_texture_disc
        snap = getattr(self, "_pix_g_disc_snapshot", None)
        if snap is None:
            snap = copy.deepcopy(disc)
            for _p in snap.parameters():
                _p.requires_grad_(False)
            self._pix_g_disc_snapshot = snap
        with torch.no_grad():
            snap.load_state_dict(disc.state_dict())
        snap.eval()
        return snap

    def _pix_gen_weight(self, current_step: int) -> float:
        """Resolved gen-side pixel weight for this step (0 before D starts).

        ``pix_gan_weight`` has NO DEFAULT (TEXTURE_GAN_DESIGN §5.5 withdrew
        0.03; docs/WP_PIXGAN.md §24 forbids inheriting the LADD arms'
        0.65-0.80 / 0.054 -- different domain, reduction, loss form and
        critic).  It is resolved through
        ``model.pixel_texture_disc.resolve_gan_weight(v, strict=True)``, so
        its ABSENCE RAISES.  An explicit ``0.0`` is legal and is exactly
        the §12 weight-free probe: no gradient is contributed, while the
        UNWEIGHTED ratio telemetry still reads the term's would-be share.

        The ramp mirrors the LADD gen-side one (``gan_warmup_steps`` with
        ``gan_warmup_shape``), anchored at ``gan_disc_start_step`` -- the
        same step T3-B's D-loop starts at, so G never scores an untrained
        critic.
        """
        from model.pixel_texture_disc import resolve_gan_weight

        base = resolve_gan_weight(
            getattr(self.config, "pix_gan_weight", None), strict=True,
        )
        start = int(getattr(self, "gan_disc_start_step", 0))
        if int(current_step) < start:
            return 0.0
        warm = int(getattr(self, "gan_warmup_steps", 0) or 0)
        if warm > 0 and int(current_step) < start + warm:
            ramp = self._gan_warmup_shape_apply(
                (int(current_step) - start) / max(1, warm)
            )
            return float(base) * float(ramp)
        return float(base)

    def _maybe_run_surrogate_distillation(
        self,
        info: Dict[str, Any],
        out: Dict[str, float],
        *,
        current_step: int,
    ) -> None:
        """WP-SURROGATE (B3) distillation step -- wired by MAIN under
        researcher order (docs/TASK_SURROGATE_CONSUMPTION.md; spec =
        WP_SURROGATE.md §4.2/§4.3).

        Runs immediately AFTER the pixel D-updates so the critic fits the
        JUST-UPDATED disc (spec §4.3). Gate-off is byte-identical: no RNG
        draw, no log key, no tensor touched -- every draw below comes from
        a private generator created inside the gate.

        Real crops are drawn READ-ONLY from ``_pix_real_pool`` -- option
        (a) of WP-PIXGAN's 2026-08-24 review: NEVER via
        ``_pix_draw_reals``, whose ``_pix_recent_frames`` ring is A20
        state owned by the D-loop; a second caller would make
        ``pix_real_reuse_frac`` describe draws the D-loop never made
        (spuriously tripping, or masking, an A20 alarm). No admission, no
        eviction, no counters here: A20/A21 telemetry keeps describing
        exactly the D-loop.
        """
        if not bool(getattr(self, "surrogate_critic_enabled", False)):
            return
        distiller = getattr(self, "latent_texture_distiller", None)
        critic = getattr(self, "latent_texture_critic", None)
        if distiller is None or critic is None:
            raise RuntimeError(
                "surrogate_critic_enabled is True but the distiller/critic "
                "attributes are missing -- the build step did not run (or a "
                "resume dropped them). Failing loud: a silently skipped "
                "distillation leaves the generator consuming a never-"
                "trained critic, which reads as 'GAN term present and "
                "quiet' on every dashboard."
            )
        # Teacher selection. NOTE this block was silently LOST once (an
        # edit script asserted out before writing, and only its second
        # half was re-applied): `backbone` went undefined, the guard below
        # fell through to the pixel disc, and in surrogate-only mode that
        # is None -- so two full 60-step "SAM2 surrogate" smokes ran with
        # the distillation never executing and every log looking healthy.
        # The regime flag was doing its job; nobody read it. Hence the
        # explicit assertion under it.
        backbone = str(getattr(self, "surrogate_teacher_backbone", "pixel"))
        teacher_disc = (
            getattr(self, "sam2_teacher_disc", None)
            if backbone in ("sam2", "dinov2", "convnext")
            else getattr(self, "pixel_texture_disc", None)
        )
        if teacher_disc is None:
            # No teacher: nothing to distill from. Regime flag, never
            # silence (forgeable-zero rule).
            out["train/surrogate_distill_no_teacher"] = 1.0
            out["train/surrogate_distill_ran"] = 0.0
            return
        if int(current_step) < int(getattr(self, "gan_disc_start_step", 0)):
            out["train/surrogate_distill_warmup_skipped"] = 1.0
            return

        rc = self._pix_resolve_cfg()
        n_crops = rc["n_crops"]
        crop_rows, crop_cols = rc["crop_rows"], rc["crop_cols"]
        n_bands = rc["n_bands"]
        border = rc["border"]

        # Distinct salt: D-loop uses salt 0 (default), the G-term salt 11;
        # the distiller's crops must be neither.
        gen = self._pix_sync_generator(int(current_step), 0, salt=13)

        # Fake crops: mask-selected BEFORE the crop draw (the
        # zero-gradient trap -- a crop from a detached frame yields a
        # teacher gradient of exactly zero, and the Sobolev term would
        # faithfully learn "no gradient here" for ordinary texture; under
        # the normalized loss an all-zero target is numerically loud, not
        # ignorable). Then DETACHED: the distiller owns its own graph.
        fake_lat, _fake_logs = self._pix_select_fake_latents(info)
        crops_f, ys_f, _xs_f, _bands_f = self._pix_take_crops_with_origins(
            fake_lat.detach(),
            n_crops=n_crops, crop_rows=crop_rows, crop_cols=crop_cols,
            n_bands=n_bands, gen=gen,
        )

        # Real crops: read-only pool draw (see docstring) -- EXCEPT in
        # surrogate-only SAM2 mode, where the pixel D-loop (the pool's
        # only producer) never runs and the pool would stay empty forever:
        # the SAM2 heads would train against no reals, the distiller would
        # fit a fake-only value field, and every metric would look healthy
        # -- the exact looks-active-is-inert shape this campaign keeps
        # finding. So THIS mode admits its own windows via the same
        # ``_pix_pool_fill`` primitive the D-loop uses (A21 semantics
        # preserved: cross-ride, band-stratified, FIFO-refreshed), sized
        # to 2x the per-step draw with one fresh admission per step.
        if backbone in ("sam2", "dinov2", "convnext"):
            _pool_now = len(getattr(self, "_pix_real_pool", None) or ())
            _want = max(1, 2 * int(n_crops) - _pool_now)
            self._pix_pool_fill(
                _want, gen=gen, crop_rows=crop_rows, crop_cols=crop_cols,
                n_bands=n_bands, lat_frames=int(rc["lat_frames"]),
                step=int(current_step),
            )
        pool = getattr(self, "_pix_real_pool", None) or []
        z_real = None
        origin_real: Optional[List[int]] = None
        if pool:
            n_draw = min(int(n_crops), len(pool))
            perm = torch.randperm(
                len(pool), generator=gen, device=gen.device,
            )
            entries = [pool[int(i)] for i in perm[:n_draw].tolist()]
            z_real = torch.stack([
                e["lat"].to(crops_f.device, crops_f.dtype)
                for e in entries
            ]).detach()
            # LatentTextureCritic.forward's ``latent_origin`` is ONE
            # (y0, x0) pair applied to the WHOLE batch in a single call
            # (model/latent_texture_critic.py:403-410) -- it is not a
            # per-crop list. Found on review: a first draft passed the
            # raw per-crop y-list here, which ``step()`` -> ``compute_
            # teacher_targets`` reads as ``(origin[0], origin[1])``
            # (:693) -- i.e. crop 0's y as y0 and crop 1's y MISREAD as
            # x0, with every other crop's origin silently dropped. Fixed
            # to a single representative: mean y across the batch (x is
            # 0 by construction -- A24 does not band-match horizontal
            # position, TEXTURE_GAN_DESIGN.md §3.6, so there is no
            # meaningful x to average). This is the batch-shared-origin
            # approximation the module's own docs call "a positional
            # prior, not an index" -- correct as far as it goes, but a
            # coarser one than per-crop origins would give; see
            # WP_SURROGATE.md §4.3 for the open item to do this properly
            # if it turns out to matter.
            origin_real = [
                int(round(sum(int(e["y0"]) for e in entries) / len(entries))),
                0,
            ]
        else:
            out["train/surrogate_distill_no_reals"] = 1.0

        # ---- SAM2 backbone: heads D-update BEFORE distillation --------
        # The pretrained-teacher branch (researcher directive 2026-08-24).
        # The frozen Hiera encoder needs no update; the ADM heads DO -- a
        # surrogate distilled from never-trained heads learns noise. One
        # (or ``surrogate_sam2_updates_per_step``) NS-logistic update on
        # this step's real/fake crops, decoded under no_grad (D inputs are
        # detached by definition; only head params receive grad). This is
        # deliberately simpler than the ancestor's full RpGAN+R1+R2 --
        # mechanics first; the penalty stack is a follow-up once the smoke
        # proves the path.
        if backbone in ("sam2", "dinov2", "convnext") and z_real is not None:
            _s2_opt = getattr(self, "sam2_teacher_optimizer", None)
            if _s2_opt is not None:
                with torch.no_grad():
                    _rp = self._vae_decode_grad(
                        z_real.to(self._pix_vae_dtype())).float()
                    _fp = self._vae_decode_grad(
                        crops_f.to(self._pix_vae_dtype())).float()
                    if border > 0:
                        _rp = _rp[..., border:-border, border:-border]
                        _fp = _fp[..., border:-border, border:-border]
                _n_upd = max(1, int(getattr(
                    self.config, "surrogate_sam2_updates_per_step", 1)))
                for _k in range(_n_upd):
                    _dr = teacher_disc(_rp)
                    _df = teacher_disc(_fp)
                    _d_loss = (
                        torch.nn.functional.softplus(-_dr).mean()
                        + torch.nn.functional.softplus(_df).mean()
                    )
                    _s2_opt.zero_grad(set_to_none=True)
                    _d_loss.backward()
                    _s2_opt.step()
                out["train/surrogate_sam2_d_loss"] = float(_d_loss.detach())
                out["train/surrogate_sam2_d_real"] = float(
                    _dr.detach().mean())
                out["train/surrogate_sam2_d_fake"] = float(
                    _df.detach().mean())
                self._sam2_d_updates_total = (
                    int(getattr(self, "_sam2_d_updates_total", 0)) + _n_upd
                )
                out["train/surrogate_sam2_d_updates_total"] = float(
                    self._sam2_d_updates_total)

        def _teacher(z_crop: torch.Tensor) -> torch.Tensor:
            # Spec §4.2 closure: graph-on checkpointed decode + border trim
            # + the JUST-UPDATED teacher. ``empty_cache`` first -- the
            # step-2 allocator failure on graph-on decodes is measured-real.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # DTYPE (MAIN 13:1x, PIXGAN-authorized swap): fp32 VAE convs
            # reject bf16 latents (the step-16 pixgan200 crash). This
            # closure calls _vae_decode_grad DIRECTLY -- not the fixed
            # _pix_decode_crops_* helpers -- so it takes the same derived
            # cast. .to() is differentiable; real and fake crops share
            # this one closure, so both sides get identical treatment.
            # float32 into the disc mirrors the helpers' output contract.
            px = self._vae_decode_grad(z_crop.to(self._pix_vae_dtype()))
            if border > 0:
                px = px[..., border:-border, border:-border]
            if backbone in ("sam2", "dinov2", "convnext"):
                # Pretrained disc consumes [N, F, 3, H, W] whole and returns [N]
                # per-sample logits (frame-pooled inside);
                # reduce_patch_logits accepts [N] unchanged. Gradient flows
                # through the frozen encoder to px to z_crop -- encoder
                # params get no grad, the input path does (the ancestor's
                # documented G-path memory shape, now paid only every
                # pix_teacher_refresh_every steps, under the distiller's
                # own checkpointing).
                return teacher_disc(px.to(torch.float32))
            n, f = int(px.shape[0]), int(px.shape[1])
            logits = self.pixel_texture_disc(
                px.flatten(0, 1).to(torch.float32))
            return logits.reshape(n, f, *logits.shape[1:])

        logs = distiller.step(
            z_real=z_real,
            z_fake=crops_f,
            teacher_value_fn=_teacher,
            current_step=int(current_step),
            optimizer=getattr(self, "latent_critic_optimizer", None),
            origin_real=origin_real,
            # Same batch-shared-origin fix as ``origin_real`` above.
            origin_fake=[int(round(sum(ys_f) / len(ys_f))), 0] if ys_f else None,
        )
        out.update(logs)
        out["train/surrogate_distill_ran"] = 1.0
        out["train/surrogate_teacher_is_pretrained"] = (
            1.0 if backbone in ("sam2", "dinov2", "convnext") else 0.0
        )
        # Periodic direct-vs-surrogate gradient audit -- the honest readout
        # for the whole package (docs/WP_SURROGATE.md §2c): cosine between
        # the gradient the generator WOULD get from the true teacher and
        # the one the surrogate actually serves, on this step's fake crops.
        # Found inert on 2026-08-24: ``surrogate_grad_check_every`` was
        # threaded into the distiller by build_from_config but NO trainer
        # call site ever fired the audit -- the same knob-present-but-
        # unconsumed shape as pix_finish_grad_enabled. This is the call
        # site. Costs one extra teacher fwd+grad, hence the cadence gate.
        if distiller.should_grad_check(int(current_step)):
            try:
                chk = distiller.surrogate_grad_check(
                    crops_f,
                    _teacher,
                    origin=(
                        [int(round(sum(ys_f) / len(ys_f))), 0]
                        if ys_f else None
                    ),
                    current_step=int(current_step),
                )
                out.update(chk)
            except Exception as exc:
                # The audit is telemetry: it must never kill a step. But
                # never silently either (forgeable-zero rule).
                out["train/surrogate_check_err"] = 1.0
                if self.is_main_process:
                    logging.warning(
                        "[ActionForcing] surrogate_grad_check failed: %s",
                        exc,
                    )

    def _surrogate_g_snapshot_critic(self) -> torch.nn.Module:
        """The latent-critic COPY the surrogate G-term scores against.

        EXACTLY B1's ``_pix_g_snapshot_disc`` hazard, on the surrogate
        path, and it is an autograd CORRECTNESS bug rather than the
        staleness question §4.3 of docs/WP_SURROGATE.md analysed:

          * the G-term forward runs in the gen-loss block;
          * ``_maybe_run_surrogate_distillation`` calls
            ``latent_critic_optimizer.step()`` LATER in the same iteration;
          * ``generator_loss.backward()`` runs later still.

        Adam's step is in-place on the critic weights, so the backward
        through a graph built on the LIVE critic raises

            "one of the variables needed for gradient computation has been
             modified by an inplace operation: [512, 1] ... expected
             version 3"

        -- observed on all 8 ranks at step 21, the first distillation
        step, in both the dinov2 and sam2 arms. Note this fires even
        though ``generator_surrogate_loss`` sets ``requires_grad_(False)``
        on the critic: the version counter is checked on the SAVED tensor,
        not on its grad-ness (B1 documented the identical trap; I had read
        that docstring and still had to rediscover it from a stack trace).

        Snapshot semantics, all three wanted and all three matching B1:
          * the later in-place optimizer step cannot corrupt this graph;
          * no gradient ever lands on the real critic's ``.grad``, so the
            G backward cannot pollute the next distillation step;
          * G is trained against the critic as of the START of the step --
            ordinary alternating-GAN semantics, and the same one-step
            offset §4.3 already accepted and logs as
            ``surrogate_staleness_steps``.
        """
        import copy

        critic = self.latent_texture_critic
        snap = getattr(self, "_surrogate_g_critic_snapshot", None)
        if snap is None:
            snap = copy.deepcopy(critic)
            self._surrogate_g_critic_snapshot = snap
        with torch.no_grad():
            snap.load_state_dict(critic.state_dict())
        # Re-frozen EVERY call, not just at creation:
        # ``generator_surrogate_loss`` restores ``requires_grad_(True)`` in
        # its finally-block (correct for the live critic it was written
        # for), which would otherwise leave the snapshot accumulating a
        # ``.grad`` it has no optimizer to consume -- a slow leak.
        snap.requires_grad_(False)
        snap.eval()
        return snap

    def _compute_pixel_texture_g_loss(
        self,
        info: Dict[str, Any],
        *,
        current_step: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, float]]:
        """WP-PIXGAN T3-C -- the pixel G-term, added into ``gen_gan_loss``.

        Returns ``(weighted, raw, logs)``:
          * ``weighted``: ``w * g_loss`` to add into ``gen_gan_loss``, or
            ``None`` when ``w == 0`` (nothing is added -- that is the §12
            weight-free probe regime, and adding ``0.0 * g`` would only
            buy a zero-gradient backward through the decoder);
          * ``raw``: the UNWEIGHTED ``g_loss`` tensor, still graph-bearing.
            The A7 block differentiates THIS to publish the ratio the
            pixel term WOULD have at weight 1.0 (§12 step 1); without it a
            weight-free probe reads exactly 0.0 and settles nothing;
          * ``logs``: §7 keys, emitted only on the path that computed them.

        Same geometry as the D-loop by construction (same crop helper,
        same band plan, same frame count, same border trim), so the
        critic sees fake pixels drawn exactly as it was trained -- but a
        SEPARATE private generator stream (salt 11) so the G crops are not
        the D crops, and a GRAPH-ON decode (``_pix_decode_crops_grad``),
        which is the whole point of the term.
        """
        _sur_on = bool(getattr(self, "surrogate_critic_enabled", False))
        if not (bool(getattr(self, "gan_pixel_texture_enabled", False))
                or _sur_on):
            return None, None, {}
        if getattr(self, "pixel_texture_disc", None) is None and not _sur_on:
            return None, None, {}
        # A24/A21/§5.3 resolution happens HERE, above the warm-up return.
        # This consumer runs EARLIER in the step than the D-loop, so a pin
        # that only the D-loop enforced was a pin this path had already run
        # past. Resolution is the checkpoint; both consumers pass through it
        # and neither owns it.
        rc = self._pix_resolve_cfg()
        logs: Dict[str, float] = {}
        start = int(getattr(self, "gan_disc_start_step", 0))
        if int(current_step) < start:
            # Regime flag, not a value: ``pix_g_loss`` is OMITTED here
            # rather than zero-filled (docs/WP_PIXGAN.md §16 forgeable
            # zeros -- 0.0 is a meaningful g_loss).
            logs["train/pix_g_warmup_skipped"] = 1.0
            return None, None, logs

        from model.pixel_texture_disc import (
            g_loss, patch_logit_spatial_variance,
        )

        crop_lat = rc["crop_lat"]
        n_crops = rc["n_crops"]
        k_frames = rc["k_frames"]
        n_bands = rc["n_bands"]
        border = rc["border"]
        loss_form = rc["loss_form"]
        decode_batch = rc["decode_batch"]

        weight = self._pix_gen_weight(int(current_step))

        fake_lat, fake_logs = self._pix_select_fake_latents(info)
        logs.update(fake_logs)
        if not fake_lat.requires_grad:
            # The term would be a constant: no gradient to the generator,
            # so it is neither added nor measured. Distinct reason key --
            # never a 0.0 ratio (docs/WP_PIXGAN.md §21 instance 3).
            logs["train/pix_g_fake_no_grad"] = 1.0
            return None, None, logs

        # ------------------------------------------------------------------
        # WP-SURROGATE (B3) consumption branch -- MAIN, researcher-ordered
        # (docs/TASK_SURROGATE_CONSUMPTION.md). When the surrogate critic is
        # enabled, the generator's texture gradient comes from the LATENT
        # surrogate instead of the decode->disc direct path. The two are
        # ALTERNATIVES, never summed (WP_SURROGATE.md §4.3): this branch
        # returns before the decode path. Contract preserved: ``raw`` is the
        # UNWEIGHTED surrogate loss (the A7 probe differentiates it
        # unchanged); ``weighted`` is ``w*raw`` or None at w==0; the direct
        # path's decode/disc log keys are OMITTED here -- they would report
        # a computation that did not run (forgeable-zero rule). Ordering
        # note, recorded deviation from §4.3 same-step order: this consumer
        # runs EARLIER in the step than the D-loop (see docstring above), so
        # the critic consumed here was distilled at the END of the PREVIOUS
        # step -- one step stale, symmetric with the direct path's
        # pre-update disc snapshot, keeping the alternatives comparable.
        if bool(getattr(self, "surrogate_critic_enabled", False)):
            from model.latent_texture_critic import generator_surrogate_loss

            critic = (
                self._surrogate_g_snapshot_critic()
                if self.latent_texture_critic is not None else None
            )
            if critic is None:
                # Gate says enabled but the build produced nothing: this is
                # the exact silent-skew class; fail loud, never fall back to
                # the direct path behind the arm's back.
                raise RuntimeError(
                    "surrogate_critic_enabled is True but "
                    "latent_texture_critic is None -- the build step did "
                    "not run or was resumed without its keys. Refusing to "
                    "fall back to the direct pixel path silently."
                )
            raw_sur, sur_logs = generator_surrogate_loss(
                critic, fake_lat, weight=1.0,
            )
            logs.update(sur_logs)
            # The module computed at weight=1.0 (that call IS the raw /
            # probe form); re-state the APPLIED weight so the echoes
            # describe what actually entered the loss, not the probe call
            # (echoes-read-runtime rule).
            logs["train/surrogate_g_weight"] = float(weight)
            logs["train/surrogate_g_weighted"] = (
                float(weight) * float(logs.get("train/surrogate_g_main", 0.0))
            )
            logs["train/surrogate_consumed"] = 1.0
            logs["train/surrogate_staleness_steps"] = 1.0
            logs["train/pix_g_weight"] = float(weight)
            weighted_sur = (
                (raw_sur * float(weight)) if weight != 0.0 else None
            )
            return weighted_sur, raw_sur, logs

        gen = self._pix_sync_generator(int(current_step), 0, salt=11)
        crops, ys, xs, bands = self._pix_take_crops_with_origins(
            fake_lat,
            n_crops=n_crops,
            crop_rows=int(crop_lat[0]),
            crop_cols=int(crop_lat[1]),
            n_bands=n_bands,
            gen=gen,
        )
        fake_px = self._pix_decode_crops_grad(
            crops, border=border, n_frames=k_frames,
            decode_batch=decode_batch, gen=gen,
        )
        snap = self._pix_g_snapshot_disc()
        fake_logits = snap(fake_px)
        raw = g_loss(fake_logits, loss_form=loss_form)

        logs["train/pix_g_loss"] = float(raw.detach())
        logs["train/pix_g_weight"] = float(weight)
        logs["train/pix_g_images"] = float(int(fake_px.shape[0]))
        logs["train/pix_g_fake_logit_mean"] = float(
            fake_logits.detach().mean()
        )
        logs["train/pix_g_patch_logit_spatial_var"] = float(
            patch_logit_spatial_variance(fake_logits.detach())
        )
        logs.update(self._pix_band_histogram(bands, n_bands, prefix="pix_g_"))
        weighted = (raw * float(weight)) if weight != 0.0 else None
        return weighted, raw, logs

    def _maybe_run_pixel_texture_d_updates(
        self,
        info: Dict[str, Any],
        out: Dict[str, float],
        *,
        current_step: int,
        chunk_depth: Optional[int] = None,
    ) -> None:
        """WP-PIXGAN T3-B -- run this step's pixel-critic D updates.

        Gated on ``gan_pixel_texture_enabled`` (T3-A). When off this
        returns before touching anything: no decode, no dataset read, no
        RNG draw (every draw below comes from a private generator created
        inside the gate) and no log key. That is the byte-identical-off
        guarantee, and ``testing/test_pixgan_trainer_supply.py`` asserts it
        against the global RNG state.

        Cadence: ``pix_gan_updates_per_step`` FOLLOWS ``gan_updates_per_step``
        (5 in the ganfix arms, 1 by config default) and the D pass is
        deferred until ``gan_disc_start_step`` exactly like the LADD path.
        ``gan_warmup_steps`` is the GENERATOR-side ramp and does not gate D
        -- T3-C consumes it for the G-term.

        Each update, per §5.2:
          1. resolve the fake latents (mask-selected on the grad path);
          2. draw ``pix_crops_per_step`` latent crops + their origins;
          3. DETACH and decode both sides under ``no_grad`` (D is trained
             on pixels, not through the generator);
          4. one independently sourced real per fake, SAME A24 band;
          5. ``d_loss`` + ``r1_penalty`` (every update), step
             ``pix_optimizer``.

        EXPOSURE (researcher directive, 2026-08-24): the critic needs A
        LOT of exposure to fakes and equivalent reals --
        ``pix_crops_per_step=8`` x ``pix_frames_per_crop=3`` x
        (``pix_gan_updates_per_step`` following ``gan_updates_per_step`` =
        5 in the ganfix arms) = 120 fakes + 120 band-matched independent
        reals per training step, every fake decoded from THIS step's
        chunk. Exposure is COUNTED (``pix_fake_images_total`` /
        ``pix_real_images_total``, decode-window count, wall-time delta),
        never assumed: the real-side decodes are no_grad but land on the
        critical path (§6), so the cost is measured here too.

        NO FAKE-SIDE MEMORY -- deliberate and load-bearing (researcher,
        2026-08-24): the axes along which texture collapse happens are
        always changing; it is the POSITIVE samples that stay in the same
        style. The negatives will always be changing in style -- D needs
        the CURRENT degenerate collapse, not all historic ones, and
        long-term memory on the fake side makes the discriminator's job
        HARDER. Cache the stationary class, never the moving one: real
        style is stationary, so the large continuously-refreshed real pool
        (A21) stays; the fake distribution is non-stationary by direction,
        so every fake is drawn LIVE. A SimGAN-style fake replay ring was
        designed and WITHDRAWN on exactly this reasoning -- do not re-add
        one.

        ``chunk_depth`` is the 0-based AR chunk index of this step's
        trained chunk within the current ride's rollout, threaded from
        ``_streaming_train_one_chunk`` where the chunk identity is known.
        It feeds the coarse fake-depth histogram, so "the D sees drifted
        fakes" is a number, not a hope. ``None`` = depth unmeasured: the
        histogram keys are OMITTED, never zero-filled.
        """
        if not bool(getattr(self, "gan_pixel_texture_enabled", False)):
            return
        if getattr(self, "pixel_texture_disc", None) is None:
            return
        if getattr(self, "pix_optimizer", None) is None:
            return
        # Re-assert the CONFIG->PIPELINE seam (see the propagation at
        # ``self.model = model``). Idempotent bool setattr; inside the gate so
        # the off path is untouched. A flag that survives construction but not
        # a pipeline rebuild is the same bug with a longer fuse.
        # ONE read per function: the resolved-config echo below reports
        # THIS local rather than taking a second getattr of its own. The
        # knob is not resolver-owned because _build_pipeline reads it
        # outside the pixel gate, where the validating resolver must never
        # be reached -- but two reads inside ONE function was pure drift
        # surface with nothing bought.
        _finish_grad = bool(
            getattr(self.config, "pix_finish_grad_enabled", False)
        )
        _pipe = getattr(self, "pipeline", None)
        if _pipe is not None:
            _pipe.pix_finish_grad_enabled = _finish_grad
        start = int(getattr(self, "gan_disc_start_step", 0))
        if int(current_step) < start:
            out["train/pix_d_updates"] = 0.0
            out["train/pix_d_warmup_skipped"] = 1.0
            return

        from model.pixel_texture_disc import (
            d_loss, patch_logit_spatial_variance, r1_penalty,
        )

        # §6 -- the real-side decodes are no_grad but the loader lands on
        # the CRITICAL PATH, and at 8 crops x 5 updates the cost must be
        # MEASURED, not assumed: decode-window count and the D-block's
        # wall-time delta are accumulated here and logged once per step.
        _t_pix0 = time.monotonic()
        _decode_windows_step = 0

        cfg = self.config
        # ONE resolution point (see _pix_resolve_cfg). The A24 band pin used
        # to live HERE, after the two early returns above, while the G-side
        # consumer read pix_band_count with no pin at all and ran earlier in
        # the step; it is now enforced wherever the config is resolved.
        rc = self._pix_resolve_cfg()
        crop_lat = rc["crop_lat"]
        crop_rows, crop_cols = rc["crop_rows"], rc["crop_cols"]
        n_crops = rc["n_crops"]
        k_frames = rc["k_frames"]
        n_bands = rc["n_bands"]
        reals_per_fake = rc["reals_per_fake"]
        border = rc["border"]
        loss_form = rc["loss_form"]
        r1_gamma = rc["r1_gamma"]
        r1_sigma = rc["r1_sigma"]
        r1_num = rc["r1_num"]
        decode_batch = rc["decode_batch"]
        lat_frames = rc["lat_frames"]
        n_updates = rc["n_updates"]

        counters = self._pix_counters()
        disc = self._pix_disc_module()
        raw_disc = self.pixel_texture_disc
        band_mismatch = int(getattr(self, "_pix_band_mismatch", 0))
        logs: Dict[str, float] = {}
        n_done = 0

        for u in range(max(0, n_updates)):
            gen = self._pix_sync_generator(current_step, u)
            rgen = self._pix_rank_generator(current_step, u)

            # A21 CONTINUOUS REFRESH -- forever, NOT "until the pool is
            # full".
            #
            # The previous rule was ``min(pool_target - pool_now, ...)``,
            # which reaches 0 the moment the pool reaches its cap and stays
            # there. Since admission is the only thing that grows A21
            # support, that froze support at the pool's ceiling
            # (windows x lat_frames = 4,096 at the shipped 2,048 x 2 --
            # EXACTLY PIX_A21_SUPPORT_FLOOR, and measurably below it once
            # windows overlap inside a ride), so pix_a21_support_ok could
            # never be 1; it froze the FIFO branch in _pix_pool_fill into
            # dead code, making the pool the memorisable mini-dataset A21
            # forbids; and it stopped calling _pix_train_ride_paths, so the
            # A22 holdout claim went stale and was re-published unchanged.
            #
            # So: ``_refresh`` windows EVERY update regardless of pool size,
            # plus a warm-up rate while the pool is still cold. The cap
            # bounds MEMORY (FIFO eviction in _pix_pool_fill); it must never
            # bound refresh.
            _pool_now = len(getattr(self, "_pix_real_pool", None) or ())
            _pool_target = rc["pool_windows"]
            _refresh = rc["pool_refresh"]
            # Warm-up rate: reach the cap in pix_real_pool_warm_updates
            # D-updates instead of pool_target/refresh of them (2,048/8 =
            # 256 updates, i.e. a 300-step arm would spend 85 % of its life
            # below the floor even with refresh fixed).
            _warm_rate = max(
                _refresh,
                -(-_pool_target // max(1, rc["pool_warm_updates"])),
            )
            # Enough stock to serve one D-update's paired draw even on the
            # very first update; _pix_draw_reals force-fills short bands
            # anyway, this just avoids doing it one window at a time.
            _warm_min = n_crops * k_frames * reals_per_fake * 2
            if _pool_now >= _pool_target:
                _grow = _refresh
            else:
                _grow = min(
                    _warm_rate, max(_refresh, _pool_target - _pool_now),
                )
            if _pool_now < _warm_min:
                _grow = max(_grow, min(_warm_min - _pool_now, _pool_target))
            if _grow > 0:
                self._pix_pool_fill(
                    _grow, gen=rgen, crop_rows=crop_rows,
                    crop_cols=crop_cols, n_bands=n_bands,
                    lat_frames=lat_frames, step=int(current_step),
                )

            # LIVE fakes ONLY -- researcher (2026-08-24), verbatim in
            # substance: "The axes along which texture collapse happens
            # are always changing. It is the POSITIVE samples that stay in
            # the same style -- that is what we want to get to. The
            # negatives will always be changing in style; we need to know
            # the CURRENT degenerate texture collapse, not all historic
            # ones. Long-term memory on the fake side makes the
            # discriminator's job HARDER." A fake replay ring was designed
            # and WITHDRAWN on that reasoning: cache the stationary class
            # (the reals -- the A21 pool above), never the moving one. Do
            # not re-add a fake-side history buffer.
            fake_lat, fake_logs = self._pix_select_fake_latents(info)
            logs.update(fake_logs)
            # §5.2: D sees DETACHED pixels on both sides.
            crops, ys, xs, bands = self._pix_take_crops_with_origins(
                fake_lat.detach(),
                n_crops=n_crops, crop_rows=crop_rows, crop_cols=crop_cols,
                n_bands=n_bands, gen=gen,
            )
            fake_px = self._pix_decode_crops_nograd(
                crops, border=border, n_frames=k_frames,
                decode_batch=decode_batch, gen=gen,
            )
            n_fake = int(fake_px.shape[0])
            if not getattr(self, "_pix_cfg_echoed", False):
                # Derived-vs-requested is read off the ACTUAL tensors: the
                # crop helper clamps rows/cols to the latent grid, and
                # _decode_crops declines to trim a crop smaller than 2*border.
                _dh, _dw = int(fake_px.shape[-2]), int(fake_px.shape[-1])
                self._pix_echo_resolved_config(logs, resolved={
                    "gan_pixel_texture_enabled": bool(
                        self.gan_pixel_texture_enabled),
                    "pix_finish_grad_enabled": bool(_finish_grad),
                    "pix_crop_lat_rows_requested": int(crop_lat[0]),
                    "pix_crop_lat_cols_requested": int(crop_lat[1]),
                    "pix_crop_lat_rows_derived": int(crops.shape[-2]),
                    "pix_crop_lat_cols_derived": int(crops.shape[-1]),
                    "pix_crops_per_step": int(n_crops),
                    "pix_frames_per_crop_requested": int(k_frames),
                    "pix_frames_per_crop_derived": int(
                        n_fake // max(1, int(crops.shape[0]))),
                    "pix_lat_frames_per_crop_requested": int(lat_frames),
                    "pix_lat_frames_per_crop_derived": int(
                        fake_lat.shape[1]),
                    "pix_band_count": int(n_bands),
                    "pix_reals_per_fake": int(reals_per_fake),
                    "pix_decode_batch": int(decode_batch),
                    "pix_decode_border_trim_requested": int(border),
                    "pix_decode_border_trim_derived": int(
                        (int(crops.shape[-2]) * 8 - _dh) // 2),
                    "pix_loss_form": loss_form,
                    "pix_loss_form_is_hinge": bool(loss_form == "hinge"),
                    # RESOLVED through _pix_resolve_r1_gamma: neither
                    # absent nor the inert shipped 1.0 can reach here.
                    "pix_r1_gamma": float(r1_gamma),
                    # An explicit 0.0 is legal and means R1 deliberately
                    # OFF -- marked, never inferred from the gamma float.
                    "pix_r1_disabled": bool(float(r1_gamma) == 0.0),
                    "pix_r1_sigma": float(r1_sigma),
                    # WHAT RAN, not what was asked: PixCounters applies
                    # its own max(1, .) floor, so a config 0 executes as
                    # cadence 1 and a cfg-sourced echo would report 0.
                    "pix_r1_every_n": int(counters.r1_every_n),
                    # -1 = "all reals", a sentinel OUTSIDE the >=1 range.
                    "pix_r1_num_samples": (-1 if r1_num is None
                                           else int(r1_num)),
                    # OFF THE OPTIMIZER, never off cfg. _build_optimizer
                    # is what built the number that actually steps D; see
                    # _pix_optimizer_hyper for why a cfg-sourced lr is the
                    # rule-4 defect and what it emits instead of a
                    # plausible default.
                    **self._pix_optimizer_hyper(),
                    "pix_gan_updates_per_step_derived": int(n_updates),
                    # TRUE = the value above was FOLLOWED from
                    # ``gan_updates_per_step`` (config null/absent);
                    # FALSE = a pinned ``pix_gan_updates_per_step`` won.
                    "pix_gan_updates_per_step_followed": bool(
                        rc["n_updates_followed"]),
                    "pix_real_pool_windows": int(rc["pool_windows"]),
                    "pix_real_pool_refresh": int(rc["pool_refresh"]),
                    "pix_real_pool_warm_updates": int(
                        rc["pool_warm_updates"]),
                    # DERIVED: windows admitted per D-update while the pool
                    # is still cold. pix_real_pool_refresh is the STEADY
                    # rate and does not describe warm-up at all.
                    "pix_real_pool_warm_rate_derived": int(_warm_rate),
                    "pix_real_reuse_horizon": int(rc["reuse_horizon"]),
                    # DERIVED: the steady-state pool's nominal source-frame
                    # ceiling and its margin over PIX_A21_SUPPORT_FLOOR.
                    # 2048 x 2 = 4096 = 1.00x is the configuration that
                    # could never pass.
                    "pix_a21_support_floor": int(
                        self.PIX_A21_SUPPORT_FLOOR),
                    "pix_a21_pool_ceiling_frames_derived": int(
                        rc["pool_ceiling_frames"]),
                    "pix_a21_pool_margin_derived": float(rc["pool_margin"]),
                    "pix_real_draw_rank_synced": bool(
                        rc["real_draw_rank_synced"]),
                    # ---- T3-C additions. DERIVED values, not requests.
                    #
                    # ``pix_gan_weight`` is echoed as its CALIBRATED-ness
                    # first: the resolved number is only echoed when one
                    # exists, because ``None`` has no honest float and 0.0
                    # is a legal calibrated weight (the §12 weight-free
                    # probe), so a 0.0 here would be unreadable. The loud
                    # failure for an uncalibrated weight lives in
                    # ``_pix_gen_weight`` (resolve_gan_weight strict=True),
                    # which runs in the gen block BEFORE this echo.
                    "pix_gan_weight_calibrated": bool(
                        getattr(cfg, "pix_gan_weight", None) is not None),
                    # The gen-side ramp is DERIVED from two knobs that are
                    # not named ``pix_*`` at all -- which is precisely the
                    # "effective value assembled elsewhere and never
                    # logged" trap of §21.
                    "pix_g_start_step_derived": int(
                        getattr(self, "gan_disc_start_step", 0)),
                    "pix_g_warmup_steps_derived": int(
                        getattr(self, "gan_warmup_steps", 0) or 0),
                    "pix_g_warmup_shape_is_linear": bool(
                        str(getattr(self, "gan_warmup_shape", "linear"))
                        == "linear"),
                    "pix_grad_telemetry_every_derived": int(
                        getattr(cfg, "gan_grad_telemetry_every", 25) or 0),
                    "pix_poscontrol_every": int(
                        getattr(cfg, "pix_poscontrol_every", 0) or 0),
                    "pix_poscontrol_amplitude": float(
                        getattr(cfg, "pix_poscontrol_amplitude", 0.5)),
                    "pix_poscontrol_boot": int(
                        getattr(cfg, "pix_poscontrol_boot", 1000)),
                    "pix_seed": int(rc["seed"]),
                })
            # A24 pairing: the band of every fake IMAGE (a crop contributes
            # k_frames images, all from the same crop and so the same band).
            per_image_bands: List[int] = []
            for b in bands:
                per_image_bands.extend([int(b)] * max(1, k_frames))
            per_image_bands = per_image_bands[:n_fake]
            real_bands_req = per_image_bands * reals_per_fake

            real_lat, real_bands, real_logs = self._pix_draw_reals(
                real_bands_req, gen=rgen, crop_rows=crop_rows,
                crop_cols=crop_cols, n_bands=n_bands, lat_frames=lat_frames,
                reuse_horizon=rc["reuse_horizon"], step=int(current_step),
            )
            # One image per real window -- n_frames=1. Frame-within-crop
            # expansion is a FAKE-side device (A20) and must not leak here.
            real_px = self._pix_decode_crops_nograd(
                real_lat, border=border, n_frames=1,
                decode_batch=decode_batch, gen=gen,
            )
            logs.update(real_logs)
            mism = sum(
                1 for a, b in zip(real_bands_req, real_bands) if int(a) != int(b)
            )
            band_mismatch += int(mism)
            if mism:
                raise RuntimeError(
                    f"A24 VIOLATION: {mism}/{len(real_bands_req)} paired reals "
                    "came from a different band than their fake."
                )
            if int(real_px.shape[0]) != n_fake * reals_per_fake:
                raise RuntimeError(
                    "A20 VIOLATION: %d reals for %d fakes (reals_per_fake=%d)."
                    % (int(real_px.shape[0]), n_fake, reals_per_fake)
                )
            # A20 -- pix_real_repeat_frac now MEASURES source-frame overlap
            # (see _pix_draw_reals), so it can move, and the threshold is
            # set where only a STRUCTURAL break can reach: coincidental
            # overlap between independently drawn windows cannot put half a
            # batch on shared source frames, while frame-within-crop
            # expansion leaking to the real side would put (k-1)/k of it
            # there immediately. Demanding exactly 0 would demand
            # without-replacement sampling, which is not what "independent"
            # means -- and was in any case unreachable: the old gauge
            # compared pool INDICES drawn without replacement against
            # themselves and was pinned at 0.
            _rep = float(real_logs["train/pix_real_repeat_frac"])
            if _rep > float(self.PIX_A20_REPEAT_ALARM):
                raise RuntimeError(
                    "A20 VIOLATION: pix_real_repeat_frac=%.4f exceeds the "
                    "%.2f alarm. %d of this D-update's %d reals share a "
                    "SOURCE FRAME span with an earlier real, which is far "
                    "past any sampling coincidence: the reals are not "
                    "independently sourced. Frame-within-crop expansion is "
                    "a FAKE-side device only and must never leak to the "
                    "real side."
                    % (_rep, float(self.PIX_A20_REPEAT_ALARM),
                       int(round(_rep * len(real_bands_req))),
                       len(real_bands_req))
                )

            real_px = real_px.detach()
            fake_px = fake_px.detach()

            self.pix_optimizer.zero_grad(set_to_none=True)
            real_logits = disc(real_px)
            fake_logits = disc(fake_px)
            d_out = d_loss(real_logits, fake_logits, loss_form=loss_form)

            # R1 on EVERY D-update (pix_r1_every_n = 1). ``indices`` and
            # ``eps`` are passed EXPLICITLY and drawn from the rank-0
            # broadcast-seeded generator, so the perturbation is identical
            # on every rank and the all-reduce averages a consistent
            # gradient (the failure mode _sample_critic_grad_frame_indices
            # exists to prevent).
            n_real = int(real_px.shape[0])
            idx = None
            if r1_num is not None and int(r1_num) < n_real:
                idx = torch.randperm(n_real, generator=gen)[: int(r1_num)]
                idx = idx.to(self.device, torch.long)
                if dist.is_initialized():
                    dist.broadcast(idx, src=0)
            n_pert = n_real if idx is None else int(idx.numel())
            eps = torch.randn(
                (n_pert, *real_px.shape[1:]), generator=gen,
                dtype=torch.float32,
            ).to(self.device, real_px.dtype)
            if dist.is_initialized():
                dist.broadcast(eps, src=0)
            fired = counters.should_fire_r1()
            r1_out = r1_penalty(
                disc, real_px, gamma=r1_gamma, sigma=r1_sigma,
                num_samples=None, indices=idx, eps=eps,
            )
            # T3-C -- R1's SHARE of the D-side gradient (see the helper's
            # docstring: a magnitude cannot separate "penalty decorative"
            # from "penalty pinning the critic"; a share can). Runs BEFORE
            # the backward so the graph is still intact, and it uses
            # ``retain_graph=True`` so ``total.backward()`` below is
            # unaffected.
            logs.update(self._pix_r1_grad_share(
                d_out["d_loss"], r1_out["r1"], current_step=current_step,
            ))
            total = d_out["d_loss"] + r1_out["r1"]
            total.backward()
            self.pix_optimizer.step()
            # gsq is passed, not just the fired flag: a SINGLE draw of the
            # finite-difference gsq is 126% relative sd at the spec crop
            # (300 draws spanned 2.3e-09..9.0e-03), so 5.3's gamma
            # calibration and 5.4's row-one read cannot use the per-step
            # ``train/pix_r1_grad_sq`` below. The estimator is unbiased, so
            # the running mean PixCounters keeps is the usable statistic --
            # and it stays empty unless the value is handed over here.
            counters.note_d_update(
                r1_fired=True, gsq=float(r1_out["gsq"]),
            )
            n_done += 1

            logs["train/pix_d_loss"] = float(d_out["d_loss"].detach())
            logs["train/pix_d_real_mean"] = float(d_out["d_real_mean"])
            logs["train/pix_d_fake_mean"] = float(d_out["d_fake_mean"])
            logs["train/pix_r1_grad_sq"] = float(r1_out["gsq"])
            logs["train/pix_r1_n_used"] = float(r1_out["n_used"])
            # P is read off the TENSOR (660 post-trim, not the doc's 768).
            logs["train/pix_patch_count"] = float(r1_out["P"])
            logs["train/pix_fake_images"] = float(n_fake)
            logs["train/pix_real_images"] = float(int(real_px.shape[0]))
            logs["train/pix_crop_y0_mean"] = float(sum(ys)) / max(1, len(ys))
            logs["train/pix_crop_x0_mean"] = float(sum(xs)) / max(1, len(xs))
            logs["train/pix_decodes"] = float(
                n_crops + int(real_lat.shape[0])
            )
            _decode_windows_step += int(n_crops) + int(real_lat.shape[0])
            # RESEARCHER EXPOSURE DIRECTIVE (2026-08-24) -- monotone
            # exposure totals, kept on ``self`` so they survive across
            # steps and the trace carries "how much has D actually seen"
            # as a number rather than an assumption.
            self._pix_fake_images_total = (
                int(getattr(self, "_pix_fake_images_total", 0))
                + int(n_fake)
            )
            self._pix_real_images_total = (
                int(getattr(self, "_pix_real_images_total", 0))
                + int(real_px.shape[0])
            )
            # Coarse fake-depth histogram (0-1 / 2-3 / 4+ AR chunks deep):
            # cumulative image counts per bin, so "the D sees drifted
            # fakes" is a number. OMITTED when no depth tag was threaded
            # in (never zero-filled -- forgeable-zero rule below).
            if chunk_depth is not None:
                _d = int(chunk_depth)
                _bin = "0_1" if _d <= 1 else ("2_3" if _d <= 3 else "4p")
                _hist = getattr(self, "_pix_fake_depth_hist", None)
                if _hist is None:
                    _hist = {"0_1": 0, "2_3": 0, "4p": 0}
                    self._pix_fake_depth_hist = _hist
                _hist[_bin] += int(n_fake)
                logs["train/pix_fake_depth"] = float(_d)
                for _bn, _bc in _hist.items():
                    logs[
                        "train/pix_fake_depth_images_" + _bn
                    ] = float(_bc)
            # §7 -- patch-logit spatial variance: is the critic using
            # LOCALITY, or has it collapsed to a global constant scored per
            # patch? Both sides, because a critic that varies over real
            # crops and is flat over fakes is a different diagnosis from one
            # that is flat on both.
            logs["train/pix_patch_logit_spatial_var_real"] = float(
                patch_logit_spatial_variance(real_logits.detach())
            )
            logs["train/pix_patch_logit_spatial_var_fake"] = float(
                patch_logit_spatial_variance(fake_logits.detach())
            )
            # §7 -- crop-band histogram + the mandated per-run echo of
            # ``pix_band_count`` itself, so a silently tightened A24 banding
            # shows up in the trace and not only in a diff.
            logs.update(self._pix_band_histogram(bands, n_bands))
            # §7 / §8.1 -- the positive control (default OFF).
            logs.update(self._pix_positive_control(
                real_lat, fake_px, disc,
                current_step=int(current_step), update_idx=int(u),
                border=border, k_frames=k_frames, decode_batch=decode_batch,
            ))
            del real_px, fake_px, real_logits, fake_logits, crops, real_lat

        # ------------------------------------------------------------------
        # FORGEABLE-ZERO RULE (three instances found across three packages):
        # a diagnostic must never be given a placeholder value inside its own
        # meaningful range. Several keys here have 0.0 as their REQUIRED
        # HEALTHY VALUE (pix_real_repeat_frac, pix_band_mismatch,
        # pix_holdout_leak) or as a SPECIFIC ALARM (pix_r1_rate < 0.99 is a
        # build bug; pix_a21_support_ok). Zero-filling any of them on a path
        # that did not run certifies a perfect result for a computation that
        # never happened -- strictly worse than reporting nothing.
        #
        # So: everything below is emitted ONLY when the loop actually ran.
        # ``pix_d_updates`` is a COUNT, not a health reading, and doubles as
        # the regime flag -- 0 there means "no D-update happened", which is
        # what makes the absent keys interpretable rather than mysterious.
        # ------------------------------------------------------------------
        self._pix_band_mismatch = int(band_mismatch)
        logs["train/pix_d_updates"] = float(n_done)
        if n_done > 0:
            # Exposure totals + measured cost (researcher directive) --
            # emitted only when the loop actually ran (forgeable-zero
            # rule: a 0.0 wall-time on a skipped path would certify a
            # free D-block that never executed).
            logs["train/pix_fake_images_total"] = float(
                getattr(self, "_pix_fake_images_total", 0)
            )
            logs["train/pix_real_images_total"] = float(
                getattr(self, "_pix_real_images_total", 0)
            )
            logs["train/pix_decode_windows_step"] = float(
                _decode_windows_step
            )
            logs["train/pix_d_block_wall_s"] = float(
                time.monotonic() - _t_pix0
            )
            logs["train/pix_band_mismatch"] = float(band_mismatch)
            logs["train/pix_d_updates_total"] = float(counters.dupdate_total)
            logs["train/pix_r1_fires_total"] = float(counters.r1_fired_total)
            # Guarded against the 0/0 derivation: PixCounters.r1_rate is
            # fired/max(1, updates), which reads 0.0 -- a FALSE build-bug
            # alarm -- before the first update.
            if int(counters.dupdate_total) > 0:
                logs["train/pix_r1_rate"] = float(counters.r1_rate)
            # The USABLE gsq statistic (see note_d_update above). Emitted
            # only once observed -- a 0.0 running mean would read as "R1
            # measures no gradient on the reals", a real alarm, rather than
            # "nobody has handed us a value yet" (forgeable-zero rule).
            _gsq_mean = counters.r1_grad_sq_mean
            if _gsq_mean is not None:
                logs["train/pix_r1_grad_sq_mean"] = float(_gsq_mean)
                logs["train/pix_r1_grad_sq_n"] = float(
                    counters.r1_grad_sq_samples
                )
            # ----------------------------------------------------------
            # SAFETY CLAIMS -- a DIFFERENT rule from the diagnostics above.
            #
            # A diagnostic answers "how is it going" and MAY be silent: a
            # missing key honestly reads as "not measured". A safety claim
            # answers "is this run valid", and for those silence and
            # "clean" are too easily confused -- a reader (or a script)
            # that sees no ``pix_holdout_leak`` concludes there was no
            # leak, which is the very failure being guarded against
            # arrived at by another route.
            #
            # So on any path where the D-loop actually ran these are
            # MANDATORY, and if the underlying check could not run we
            # RAISE naming what could not be verified -- never omit, never
            # default. This matches ``model/disc_holdout_probe.py``, which
            # already fails CLOSED with explicit INVALID_NO_HOLDOUT_ROOT /
            # INVALID_NO_TRAIN_RIDES codes rather than a quiet pass; it is
            # deliberately not a second convention.
            # ----------------------------------------------------------
            _checked = float(getattr(self, "_pix_holdout_checked", 0.0))
            if _checked < 1.0:
                raise RuntimeError(
                    "A22 UNVERIFIABLE: %d pixel-critic D-update(s) ran but "
                    "the holdout disjointness check never did, so "
                    "train/pix_holdout_leak cannot be asserted for this "
                    "step. A holdout list leaked into training once before; "
                    "an unverified run must fail, not log nothing."
                    % int(n_done)
                )
            # ...and it must not be STALE. ``_pix_holdout_checked`` is a
            # sticky attribute: once the pool stopped refreshing, the
            # ``< 1.0`` gate above kept passing on a flag set hundreds of
            # steps earlier and this block re-published a leak value that
            # had not been recomputed since, every step, as if fresh. A
            # safety claim defeated by staleness is defeated just as
            # thoroughly as one that was omitted, so the stamp has to match
            # THIS step. Refresh guarantees it does (A21 admits every
            # D-update, and admission runs the A22 check); this asserts it
            # instead of assuming it.
            _checked_at = getattr(self, "_pix_holdout_checked_step", None)
            if _checked_at is None or int(_checked_at) != int(current_step):
                raise RuntimeError(
                    "A22 STALE: %d pixel-critic D-update(s) ran at step %d "
                    "but the holdout disjointness check last ran at step "
                    "%s. train/pix_holdout_leak would be a re-publication "
                    "of an older step's result, which reads exactly like a "
                    "fresh all-clear. A21's continuous refresh re-runs the "
                    "check on every D-update; if it did not, the real pool "
                    "has stopped refreshing."
                    % (int(n_done), int(current_step),
                       "never" if _checked_at is None else int(_checked_at))
                )
            logs["train/pix_holdout_checked"] = _checked
            logs["train/pix_holdout_checked_step"] = float(int(_checked_at))
            logs["train/pix_holdout_rides"] = float(
                getattr(self, "_pix_holdout_n", 0.0)
            )
            logs["train/pix_holdout_leak"] = float(
                getattr(self, "_pix_holdout_leak", 0.0)
            )
            # A21 is a MEASURED claim -- never assumed from the loader. A
            # forged or absent 1.0 would certify the 4,096 distinct-source-
            # frame floor without measuring it.
            _support = getattr(self, "_pix_real_support", None)
            if _support is None:
                raise RuntimeError(
                    "A21 UNVERIFIABLE: %d pixel-critic D-update(s) ran but "
                    "the real-support identity set was never built, so the "
                    "%d distinct-source-frame floor cannot be asserted. A21 "
                    "is a measured claim; refusing to certify it silently."
                    % (int(n_done), int(self.PIX_A21_SUPPORT_FLOOR))
                )
            _n_sup = int(len(_support))
            _warm_n = int(rc["pool_warm_updates"])
            _warming = int(counters.dupdate_total) <= _warm_n
            logs["train/pix_real_support_frames"] = float(_n_sup)
            logs["train/pix_a21_support_ok"] = float(
                1.0 if _n_sup >= self.PIX_A21_SUPPORT_FLOOR else 0.0
            )
            # Support is CUMULATIVE and starts at zero, so a 0 here is
            # honest while the pool is still filling -- and the WARM-UP
            # REGIME KEY is what stops that honest 0 being read as a
            # violation. It is also what stops the regime lasting forever:
            # once warm-up is over an unmet floor is a hard failure, not a
            # logged 0.0. That is the difference between the fixed code and
            # the old one, where 0.0 was emitted on every step of the run
            # and the arm quietly certified its own A21 violation for its
            # whole life.
            if _warming:
                logs["train/pix_a21_warming"] = 1.0
                logs["train/pix_a21_warm_updates_left"] = float(
                    max(0, _warm_n - int(counters.dupdate_total))
                )
            elif _n_sup < self.PIX_A21_SUPPORT_FLOOR:
                raise RuntimeError(
                    "A21 VIOLATION: after %d D-update(s) (warm-up = %d) the "
                    "pixel critic's real supply has covered only %d distinct "
                    "source frames, below the %d floor. Pool=%d/%d windows, "
                    "%d admission(s) total, refresh=%d/update. The floor is "
                    "not a target to log progress against: reals drawn from "
                    "a support this small are a mini-dataset the critic can "
                    "memorise, which is what A21 exists to prevent."
                    % (int(counters.dupdate_total), _warm_n, _n_sup,
                       int(self.PIX_A21_SUPPORT_FLOOR),
                       len(getattr(self, "_pix_real_pool", None) or ()),
                       int(rc["pool_windows"]),
                       int(getattr(self, "_pix_pool_refresh_total", 0)),
                       int(rc["pool_refresh"]))
                )
            # A21's other half, echoed every step it is claimed: how much of
            # the support the STEADY-STATE pool carries on its own. A pool
            # sitting at 1.00x the floor carries none of it.
            logs["train/pix_a21_pool_ceiling_frames"] = float(
                rc["pool_ceiling_frames"]
            )
            logs["train/pix_a21_pool_margin"] = float(rc["pool_margin"])
        out.update(logs)

    def _gan_warmup_shape_apply(self, t_normalized: float) -> float:
        """Apply ``self.gan_warmup_shape`` to a normalised ramp position
        ``t ∈ [0, 1]``. Returns a value in ``[0, 1]`` that the caller
        multiplies by ``gan_loss_weight`` to get the effective gen-side
        weight. See ``__init__`` for shape definitions.
        """
        t = max(0.0, min(1.0, float(t_normalized)))
        if self.gan_warmup_shape == "quadratic":
            return t * t
        if self.gan_warmup_shape == "cosine":
            return 0.5 * (1.0 - math.cos(math.pi * t))
        return t  # "linear"

    def _couple_gan_weight_to_gate(self, gen_gan_weight: float) -> float:
        """Rebalance the gen-side GAN weight against the MAE-gated DMD weight.

        When the gate cuts DMD (gate_w < 1), pull the GAN weight down too but
        gentler (``gate_w**beta``, beta<1 => sqrt), never below a floor, and keep
        it >= ``min_ratio`` x the gated DMD weight so the GAN still carries the
        student when the teacher (DMD) is unreliable, without running free.
        ``min(floor, gan_base)`` ensures we never force GAN ABOVE its base at
        full DMD. No-op when disabled / no gating. See ``__init__`` comment.
        """
        if not self.gan_gate_couple_enabled or gen_gan_weight <= 0.0:
            return gen_gan_weight
        gate_w = float(getattr(self.model, "_last_dmd_mae_gate_weight", 1.0))
        if gate_w >= 1.0:
            return gen_gan_weight  # DMD not gated -> leave GAN at full
        gan_base = float(gen_gan_weight)
        dmd_base = float(getattr(self.model, "dmd_loss_weight", 1.0))
        dmd_eff = dmd_base * gate_w
        coupled = gan_base * (gate_w ** self.gan_gate_couple_beta)
        floor = max(self.gan_gate_couple_min_ratio * dmd_eff,
                    self.gan_gate_couple_floor_frac * gan_base)
        floor = min(floor, gan_base)
        return float(min(max(coupled, floor), gan_base))

    # ==================================================================
    # LADD discriminator (v28)
    # ==================================================================
    # Supports two pair construction modes, independently toggleable:
    #   * "gt_vs_fake" (LADD canonical): real = GT chunks at the same
    #     temporal positions as the gen output. Standard adversarial
    # ==================================================================
    # ForwardNoiser (CARN) — teacher_feat training
    # ==================================================================
    # Train the FN to map rollout1 chunks -> rollout2 chunks (+1 CARN
    # drift step) by DISTRIBUTION-matching them in the FROZEN teacher's
    # feature space (the LADD WanFeatureProjector backbone, NOT its
    # trained heads). The teacher is a fixed, comprehensive critic, so the
    # FN can't exploit an adversary's blind spots. SEPARATE backward
    # (FN grads only; the train() loop steps the FN optimizer) — fully
    # decoupled from fake_score. The FN is step-UNCONDITIONED (carn_step=0
    # always): it learns one generic "+1 shift" transform.
    def _fn_action_blind_cond(self, n_rows: int, n_frames: int,
                              device, dtype):
        """Build action-blind (zeroed) conditioning of the shape the
        action-aware WAN teacher requires, so the projector forward runs.
        Returns None when the teacher has no action tokens."""
        m = self.model
        a_per_f = 0
        _rs = getattr(m, "real_score", None)
        if _rs is None:
            return None
        for _cand in [_rs] + list(_rs.modules()):
            if hasattr(_cand, "action_tokens_per_frame"):
                a_per_f = int(getattr(_cand, "action_tokens_per_frame", 0))
                if a_per_f > 0:
                    break
        if a_per_f <= 0:
            return None
        atp = getattr(m, "action_token_projection", None)
        ap = getattr(m, "action_projection", None)
        s = getattr(m, "streaming_state", None)
        ride_act = s.get("ride_actions_window") if isinstance(s, dict) else None
        a_dim = int(ride_act.shape[-1]) if ride_act is not None else int(
            getattr(m, "raw_action_dim", 2)
        )
        acts_zero = torch.zeros(
            (n_rows, n_frames, a_dim), device=device, dtype=dtype,
        )
        cond: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            if atp is not None:
                cond["_action_tokens"] = atp(acts_zero).detach()
            if ap is not None:
                cond["_action_modulation"] = ap(
                    acts_zero, num_frames=n_frames,
                ).detach()
        return cond if cond else None

    def _fn_teacher_feat_loss(self, fn_out, r2_target, projector):
        """Sliced-Wasserstein per-token distribution match between
        FN(rollout1) [grad-on] and rollout2 [detached] in the frozen
        teacher's feature space. Returns a scalar loss graph-attached
        through ``fn_out`` only.

        NOT a moment (mean+std) match: GT<->student moments barely differ
        and are pinned by the stat anchor, so a moment match is inert (v8d
        fn_tf_loss ~1e-6). The full sliced-Wasserstein marginal captures
        the texture/cartoon distribution shift the FN must learn."""
        m = self.model
        feat_t = int(getattr(m, "forward_noiser_feat_t", 0))
        # Match the disc's input pipeline: when the disc applies a wavelet-HF
        # pre-stage, the projector was conditioned on wavelet'd CLEAN latents
        # (disc forces t=0 when wavelet_on). Replicate both here so the FN is
        # matched in the SAME feature space its output is consumed in
        # (the carn_former feeds FN(GT) through this same wavelet'd disc).
        wavelet = getattr(self.r3gan_disc, "wavelet_hf", None)
        if wavelet is not None:
            feat_t = 0
        n_rows, n_frames = int(fn_out.shape[0]), int(fn_out.shape[1])
        device = fn_out.device
        s = m.streaming_state
        pe = s.get("prompt_embeds")
        if pe is None:
            return fn_out.sum() * 0.0
        # repeat (not repeat_interleave): rows are chunk-major (torch.cat of
        # per-chunk slices) and there is a single global prompt, so a tiled
        # repeat aligns per-row. (For B>1 with per-sample prompts this would
        # need revisiting — not the case in these configs.)
        reps = max(1, n_rows // int(pe.shape[0]))
        pe_eff = pe.repeat(reps, 1, 1) if int(pe.shape[0]) != n_rows else pe
        t = torch.full(
            (n_rows, n_frames), feat_t, dtype=torch.long, device=device,
        )

        # The wavelet-HF adapter is a DISC-owned trainable module; the FN
        # backward must NOT update it (only FN params). Freeze its params
        # for the duration of this forward (the transform stays
        # differentiable w.r.t. the INPUT, so the FN gradient still flows
        # through it). Restored in ``finally``.
        _wav_params = list(wavelet.parameters()) if wavelet is not None else []
        _wav_req = [p.requires_grad for p in _wav_params]

        def _prep(x):
            xn = _noise(x)
            if wavelet is not None:
                xn = wavelet(xn)
            return xn

        def _noise(x):
            if feat_t <= 0:
                return x
            sched = getattr(m, "scheduler", None)
            if sched is None or not hasattr(sched, "add_noise"):
                return x
            eps = torch.randn_like(x)
            xf, ef = x.flatten(0, 1), eps.flatten(0, 1)
            tpf = torch.full(
                (xf.shape[0],), feat_t, dtype=torch.long, device=device,
            )
            return sched.add_noise(xf, ef, tpf).unflatten(0, x.shape[:2])

        cond_extra = self._fn_action_blind_cond(
            n_rows, n_frames, device, pe_eff.dtype,
        )
        feat_space = str(getattr(m, "forward_noiser_feat_space", "teacher"))
        n_proj = int(getattr(m, "forward_noiser_sw_n_proj", 64))
        max_tok = int(getattr(m, "forward_noiser_sw_max_tokens", 4096))

        def _latent_feats():
            # RAW-LATENT (teacher-INDEPENDENT): channel = feature axis, every
            # (B,F,H,W) position is a token. ``_noise`` respects feat_t (0=raw);
            # the disc wavelet is teacher-owned and intentionally skipped.
            # Optional per-channel whiten (real side's stats) so high-variance
            # channels / low-level latent noise don't dominate the SW.
            def _lat_tok(x):
                xn = _noise(x)
                return xn.permute(0, 1, 3, 4, 2).reshape(-1, int(xn.shape[2]))
            _ff = _lat_tok(fn_out)
            _fr = _lat_tok(r2_target).detach()
            if bool(getattr(m, "forward_noiser_latent_normalize", True)):
                _mu = _fr.mean(dim=0, keepdim=True)
                _sd = _fr.std(dim=0, keepdim=True).clamp_min(1e-6)
                _ff = (_ff - _mu) / _sd
                _fr = (_fr - _mu) / _sd
            return {0: _ff}, {0: _fr}

        def _teacher_feats():
            # Frozen teacher/disc projector feature space (legacy).
            for _p in _wav_params:
                _p.requires_grad_(False)
            try:
                ff_ = projector(
                    x_noisy=_prep(fn_out), timestep=t, prompt_embeds=pe_eff,
                    conditional_extra=cond_extra,
                )
                with torch.no_grad():
                    fr_ = projector(
                        x_noisy=_prep(r2_target), timestep=t,
                        prompt_embeds=pe_eff, conditional_extra=cond_extra,
                    )
            finally:
                for _p, _r in zip(_wav_params, _wav_req):
                    _p.requires_grad_(_r)
            return ff_, fr_

        def _sw(feats_fake, feats_real):
            # Sliced-Wasserstein per-token: last dim = channel, every
            # (row, token) = a sample; project onto random unit directions and
            # L2 the SORTED 1D marginals (= 1D Wasserstein-2 per direction,
            # averaged). Captures the full distribution shape (all moments).
            idxs = list(feats_fake.keys())
            loss = fn_out.new_zeros(())
            for idx in idxs:
                ff = feats_fake[idx].float().reshape(-1, int(feats_fake[idx].shape[-1]))
                fr = feats_real[idx].float().reshape(
                    -1, int(feats_real[idx].shape[-1])).detach()
                D = ff.shape[-1]
                M = min(int(ff.shape[0]), int(fr.shape[0]))
                if M == 0:
                    continue
                if max_tok > 0 and M > max_tok:
                    sel_f = torch.randperm(int(ff.shape[0]), device=ff.device)[:max_tok]
                    sel_r = torch.randperm(int(fr.shape[0]), device=fr.device)[:max_tok]
                    ff = ff[sel_f]
                    fr = fr[sel_r]
                elif int(ff.shape[0]) != int(fr.shape[0]):
                    ff = ff[:M]
                    fr = fr[:M]
                dirs = torch.randn(D, n_proj, device=ff.device, dtype=ff.dtype)
                dirs = dirs / dirs.norm(dim=0, keepdim=True).clamp_min(1e-8)
                pf_s, _ = torch.sort(ff @ dirs, dim=0)
                pr_s, _ = torch.sort(fr @ dirs, dim=0)
                loss = loss + (pf_s - pr_s).pow(2).mean()
            return loss / max(1, len(idxs))

        if feat_space == "latent":
            return _sw(*_latent_feats())
        if feat_space == "teacher":
            return _sw(*_teacher_feats())
        # v2-A "combined": keep teacher semantics AND add raw-latent
        # sensitivity (remove the teacher-feature-kernel blind spot) via a
        # weighted sum, rather than replacing the teacher metric entirely.
        w_t = float(getattr(m, "forward_noiser_sw_teacher_weight", 1.0))
        w_l = float(getattr(m, "forward_noiser_sw_latent_weight", 1.0))
        return w_t * _sw(*_teacher_feats()) + w_l * _sw(*_latent_feats())

    def _fn_recon_loss(self, a, b):
        """Paired reconstruction norm for the CARN-cycle (L_rev / L_cyc).
        L1 by default (the CycleGAN convention), L2 when configured."""
        if str(getattr(self.model, "cycle_recon_loss_type", "l1")) == "l2":
            return (a - b).pow(2).mean()
        return (a - b).abs().mean()

    def _reverse_noiser_zero_anchor(self) -> None:
        """DDP balance for the CARN-cycle: fire G's reducer with a zero
        backward so the reverse-noiser DDP bucket all-reduce happens on EVERY
        rank, even when this rank bailed before the real cycle backward.
        Mirrors the forward-noiser ``_anchor`` discipline. On the real path G
        is forwarded once (L_rev) inside the summed backward; on a bail path
        this stand-alone zero-backward gives G's reducer the same single
        firing -> bucket counts stay matched across ranks. No-op when the
        cycle is off (G is None)."""
        m = self.model
        G = getattr(m, "reverse_noiser", None)
        if not (
            bool(getattr(m, "forward_noiser_cycle_enabled", False))
            and G is not None
        ):
            return
        g_inner = G.module if hasattr(G, "module") else G
        p0 = next(G.parameters())
        C = int(getattr(g_inner, "latent_channels", 16))
        x0 = torch.zeros(
            (1, int(getattr(m, "num_frame_per_block", 3)), C, 8, 8),
            device=p0.device, dtype=p0.dtype,
        )
        cz = torch.zeros((1,), dtype=torch.long, device=x0.device)
        (G(x0, cz, residual=True).sum() * 0.0).backward()

    def _fn_cycle_terms(self, fn_out, fn_in, carn, proj):
        """CARN-cycle reverse + cycle losses for one (fn_in -> fn_out=F(fn_in))
        batch, where ``carn`` is the per-row rung label ℓ (the SHARED upper-
        endpoint level used to condition BOTH F and G) and ``fn_in`` is
        detached (FN inputs always are). Returns ``(L_rev, L_cyc)`` or
        ``(None, None)`` when the cycle is off.

          L_rev = recon( G(sg(F(x1)); ℓ), sg(x1) )   -> trains θ_G only
                  (recover rollout1 from F's OWN output distribution; the
                   input to G is DETACHED so this term never drags F).
          L_cyc = recon( G(F(x1); ℓ),    sg(x1) )    -> trains θ_F
                  (forward pushed to be invertible by G; G's PARAMETERS are
                   frozen for this forward when cycle_freeze_g_in_cycle so G
                   acts as a fixed invertibility critic — the transform stays
                   differentiable w.r.t. its INPUT so grad still reaches θ_F.
                   Same freeze-params-but-keep-input-grad pattern as the
                   wavelet handling in _fn_teacher_feat_loss.)
        Identity collapse (F=G=I) is blocked elsewhere by L_fwd (the SW
        forward match): identity fails it because rollout1≠rollout2 in the
        teacher's feature marginals.

        RUNG LABEL ℓ vs STEP GAP g (why one scalar suffices for G): in the
        chain-levels geometry ``cond=k`` maps a level-(k-2) input to level k
        (GT->1=1, GT->2=2, 1->3=3, 2->4=4 ...), so the per-pair gap g is a
        DETERMINISTIC function of ℓ (g=1 at ℓ=1, g=2 at ℓ>=2) — ℓ alone pins
        the (source,dest) pair, so G's inverse target is well-posed per ℓ even
        though g is not separately encoded. We condition G on ℓ = the UPPER
        endpoint = G's own INPUT level (F(x1) sits at level ℓ), which is the
        natural "tell the denoiser its input level" conditioning. At the
        frontier (carn=0 for both F and G) the pair is step-unconditioned, so
        the cycle is internally consistent there too.
        """
        m = self.model
        if not (
            bool(getattr(m, "forward_noiser_cycle_enabled", False))
            and getattr(m, "reverse_noiser", None) is not None
        ):
            return None, None
        G = m.reverse_noiser                       # DDP-wrapped reverse net
        G_inner = G.module if hasattr(G, "module") else G
        fn_in_det = fn_in.detach()
        # ---- L_rev: the ONLY DDP forward of G this step ----
        # G recovers x1 from F's OWN (detached) output -> trains+syncs θ_G.
        # CRITICAL DDP INVARIANT: G is forwarded through its DDP wrapper
        # EXACTLY ONCE per step. Forwarding a DDP module twice before a
        # single backward corrupts the reducer (only the last forward's
        # prepare_for_backward survives), so the L_cyc term below must NOT
        # go through the DDP wrapper.
        g_rev = G(fn_out.detach(), carn, residual=True)
        L_rev = self._fn_recon_loss(g_rev, fn_in_det)
        if float(getattr(m, "cycle_rev_feat_weight", 0.0)) > 0.0:
            L_rev = L_rev + float(m.cycle_rev_feat_weight) * (
                self._fn_teacher_feat_loss(g_rev, fn_in_det, proj)
            )
        # ---- L_cyc: grad flows F -> G -> loss; G is a FROZEN critic ----
        # Run through the INNER module (bypasses DDP, so no second DDP
        # forward) with G's params frozen, so the gradient reaches θ_F via
        # ``fn_out`` only and NOT θ_G (and never bypasses the reducer with an
        # un-synced θ_G contribution). The transform stays differentiable
        # w.r.t. its input — same freeze-params/keep-input-grad trick as the
        # wavelet handling in _fn_teacher_feat_loss.
        freeze = bool(getattr(m, "cycle_freeze_g_in_cycle", True))
        if not freeze:
            # CycleGAN-style (cycle trains BOTH nets) would need a SECOND
            # synced DDP forward of G -> the multi-forward reducer hazard.
            # Not supported under DDP in this implementation; the recommended
            # and default mode is freeze=True (G as a fixed invertibility
            # critic). Fail loud rather than silently de-sync θ_G.
            raise NotImplementedError(
                "cycle_freeze_g_in_cycle=False (CycleGAN-style cycle training "
                "of G) is not supported under DDP: it would require a second "
                "synced DDP forward of the reverse noiser. Use the default "
                "freeze=True (G trained only by L_rev; L_cyc updates θ_F)."
            )
        g_params = list(G_inner.parameters())
        saved_req = [p.requires_grad for p in g_params]
        for _p in g_params:
            _p.requires_grad_(False)
        try:
            g_cyc = G_inner(fn_out, carn, residual=True)
        finally:
            for _p, _r in zip(g_params, saved_req):
                _p.requires_grad_(_r)
        L_cyc = self._fn_recon_loss(g_cyc, fn_in_det)
        return L_rev, L_cyc

    def _fn_forward_backward_with_cycle(
        self, fn_in, fn_tg, carn, proj, base_logs,
    ) -> dict:
        """Run F's forward + the (optional) CARN-cycle reverse/cycle terms,
        sum into ONE backward, and return merged logs. When the cycle is OFF
        this is byte-identical to the legacy ``fn_out=F(...); loss=L_fwd;
        loss.backward()`` (no weight on L_fwd — matching the pre-cycle code).
        """
        m = self.model
        fn_out = m.forward_noiser(fn_in, carn, residual=True)  # grad-on
        L_fwd = self._fn_teacher_feat_loss(fn_out, fn_tg, proj)
        # v2-B: self-rollout paired drift loss — ground F as the student's own
        # causal-drift map (F(z_l) ~= z_{l+1}). fn_tg is the student's NEXT
        # rollout state (detached) — system identification of the student's
        # drift, NOT GT supervision (no value anchoring). Reuses fn_out, so no
        # extra F forward (DDP-safe). 0-weight (default) => byte-identical.
        _w_pair = float(getattr(m, "forward_noiser_pair_loss_weight", 0.0))
        L_pair = None
        if _w_pair > 0.0:
            L_pair = (fn_out - fn_tg.detach()).abs().mean()
            L_fwd = L_fwd + _w_pair * L_pair
        L_rev, L_cyc = self._fn_cycle_terms(fn_out, fn_in, carn, proj)
        logs = dict(base_logs)
        logs["train/fn_fwd_loss"] = float(L_fwd.detach().item())
        if L_pair is not None:
            logs["train/fn_pair_loss"] = float(L_pair.detach().item())
        if L_rev is None:
            total = L_fwd
        else:
            w_rev = float(getattr(m, "cycle_rev_loss_weight", 1.0))
            w_cyc = float(getattr(m, "cycle_consistency_loss_weight", 0.5))
            # F-only warmup: hold the cycle term at 0 until F's SW match is
            # established (so F isn't pulled toward inverting a random map).
            step_now = int(getattr(self, "step", 0))
            warm = int(getattr(m, "cycle_warmup_steps", 0))
            w_cyc_eff = 0.0 if step_now < warm else w_cyc
            # NOTE: L_fwd keeps an implicit weight of 1.0 (the pre-cycle code
            # never scaled it) so it stays the dominant anti-collapse anchor
            # (1.0 >= w_cyc default 0.5).
            total = L_fwd + w_rev * L_rev + w_cyc_eff * L_cyc
            logs["train/fn_rev_loss"] = float(L_rev.detach().item())
            logs["train/fn_cyc_loss"] = float(L_cyc.detach().item())
            logs["train/fn_cyc_weight_eff"] = float(w_cyc_eff)
            # Movement diagnostics: ‖F(x1)-x1‖ vs ‖x2-x1‖ (identity-collapse
            # detector — the former must stay > 0 and track the latter).
            with torch.no_grad():
                logs["train/fn_fwd_move"] = float(
                    (fn_out.detach() - fn_in.detach()).abs().mean().item()
                )
                logs["train/fn_pair_gap"] = float(
                    (fn_tg.detach() - fn_in.detach()).abs().mean().item()
                )
        total.backward()
        # Keep the legacy primary metric for dashboards.
        logs["train/fn_tf_loss"] = float(L_fwd.detach().item())
        return logs

    def _train_fn_frontier_pair(self) -> dict:
        """FN frontier training (phase-2 rolling): consume the
        (rollout1', rollout2') chunk pair generated at the ride's
        frontier (``model.generate_fn_frontier_pair``) and backprop the
        teacher-feat SW loss into the FN. Called ONLY on reset steps
        (rank-lockstep), immediately before ``reset_streaming_state``.
        Mirrors ``_train_forward_noiser_tf``'s DDP discipline: every
        rank runs exactly one FN forward+backward (zero-anchor on any
        bail) so the FN DDP reducer stays matched."""
        m = self.model
        if (getattr(self, "is_main_process", True)
                and getattr(self, "_fn_frontier_entry_dbg", 0) < 5):
            self._fn_frontier_entry_dbg = getattr(
                self, "_fn_frontier_entry_dbg", 0) + 1
            import sys as _sys
            print(
                f"[FN-FRONTIER] invoked at ride depth "
                f"{self._chunks_in_current_ride}",
                file=_sys.stderr, flush=True,
            )

        def _anchor(reason: str) -> dict:
            if (getattr(self, "is_main_process", True)
                    and getattr(self, "_fn_frontier_anchor_dbg", 0) < 5):
                self._fn_frontier_anchor_dbg = getattr(
                    self, "_fn_frontier_anchor_dbg", 0) + 1
                import sys as _sys
                print(
                    f"[FN-FRONTIER] skipped: {reason}",
                    file=_sys.stderr, flush=True,
                )
            fn = m.forward_noiser
            fn_inner = fn.module if hasattr(fn, "module") else fn
            p0 = next(fn.parameters())
            C = int(getattr(fn_inner, "latent_channels", 16))
            x0 = torch.zeros(
                (1, int(getattr(m, "num_frame_per_block", 3)), C, 8, 8),
                device=p0.device, dtype=p0.dtype,
            )
            cz = torch.zeros((1,), dtype=torch.long, device=x0.device)
            (fn(x0, cz, residual=True).sum() * 0.0).backward()
            self._reverse_noiser_zero_anchor()
            return {
                "train/fn_frontier_skipped": 1.0,
            }

        proj = (
            getattr(self.r3gan_disc, "projector", None)
            if self.r3gan_disc is not None else None
        )
        if proj is None:
            return _anchor("no_projector")
        pair = m.generate_fn_frontier_pair()
        if pair is None:
            return _anchor("no_pair_geometry")
        r1c, r2c = pair
        p0 = next(m.forward_noiser.parameters())
        r1c = r1c.to(device=p0.device, dtype=p0.dtype)
        r2c = r2c.to(device=p0.device, dtype=p0.dtype)
        cs = torch.zeros(
            (r1c.shape[0],), dtype=torch.long, device=r1c.device,
        )
        # Forward (+ optional CARN-cycle reverse/cycle) in ONE backward. The
        # frontier conditions on cs (=0, the existing frontier convention) for
        # both F and G.
        cyc_logs = self._fn_forward_backward_with_cycle(
            r1c, r2c, cs, proj, {"train/fn_frontier_pairs": 1.0},
        )
        _floss = float(cyc_logs.get("train/fn_fwd_loss", 0.0))
        if (getattr(self, "is_main_process", True)
                and getattr(self, "_fn_frontier_dbg", 0) < 3):
            self._fn_frontier_dbg = getattr(self, "_fn_frontier_dbg", 0) + 1
            import sys as _sys
            print(
                f"[FN-FRONTIER] trained on frontier pair at ride depth "
                f"{self._chunks_in_current_ride} (fwd_loss="
                f"{_floss:.5f}"
                + (f" rev={cyc_logs.get('train/fn_rev_loss'):.5f}"
                   f" cyc={cyc_logs.get('train/fn_cyc_loss'):.5f}"
                   if 'train/fn_rev_loss' in cyc_logs else "")
                + ")",
                file=_sys.stderr, flush=True,
            )
        out = {"train/fn_frontier_loss": _floss}
        out.update(cyc_logs)
        return out

    def _train_forward_noiser_tf(self, rollout1_chunk, info) -> dict:
        """teacher_feat FN training step (separate FN backward). Builds the
        flattened (rollout1_chunk, rollout2_chunk) pairs, runs FN on the
        rollout1 chunks (step-unconditioned), and backprops the teacher-
        feature distribution-match into the FN. DDP-balanced: every rank
        runs at least one FN forward+backward (zero-anchor on bail)."""
        m = self.model
        if not (
            getattr(m, "forward_noiser_enabled", False)
            and getattr(m, "forward_noiser_loss_mode", "mse") == "teacher_feat"
            and getattr(m, "forward_noiser", None) is not None
        ):
            return {}
        npb = int(m.num_frame_per_block)
        proj = (
            getattr(self.r3gan_disc, "projector", None)
            if self.r3gan_disc is not None else None
        )

        def _anchor(reason: str) -> dict:
            # Keep FN DDP participation identical across ranks: ALWAYS run a
            # zero FN forward+backward so its grad-bucket all-reduce fires on
            # every rank, regardless of why this rank bailed. If a rank ran
            # the real backward while another ran nothing, the FN DDP reducer
            # would hang — so the anchor is unconditional. The FN is fully
            # convolutional, so a minimal synthetic input still produces a
            # gradient on every parameter (bucket shapes are param-shaped and
            # input-size-independent; batch size need not match across ranks).
            fn = m.forward_noiser
            fn_inner = fn.module if hasattr(fn, "module") else fn
            p0 = next(fn.parameters())
            if (rollout1_chunk is not None
                    and int(rollout1_chunk.shape[1]) >= npb):
                x0 = rollout1_chunk[:, :npb].detach()
            else:
                # No usable rollout1 chunk — synthesize a minimal zero input.
                C = int(getattr(fn_inner, "latent_channels", 16))
                B = int(rollout1_chunk.shape[0]) if rollout1_chunk is not None else 1
                x0 = torch.zeros(
                    (B, npb, C, 8, 8), device=p0.device, dtype=p0.dtype,
                )
            cz = torch.zeros(
                (x0.shape[0],), dtype=torch.long, device=x0.device,
            )
            z = fn(x0, cz, residual=True)
            (z.sum() * 0.0).backward()
            self._reverse_noiser_zero_anchor()
            return {"train/fn_tf_loss": 0.0, "train/fn_tf_pairs": 0.0,
                    "train/fn_tf_skipped": 1.0}

        if proj is None:
            return _anchor("no_projector")
        s = getattr(m, "streaming_state", None)
        if not isinstance(s, dict):
            return _anchor("no_state")
        # NOTE: rollout2 (the prebuilt +1-drift rollout) is fetched LATER,
        # just before the carn_recurse=True path that actually consumes it.
        # The carn_recurse=False branch below builds its pairs from the
        # ride window + flash chunk and never touches rollout2, so requiring
        # it here would needlessly gate FN training off whenever the
        # prebuild is skipped (the carn_recurse=False memory optimisation —
        # see the _need_rollout2 gate in dmd_action_forcing).
        # rollout1 input: prefer the t=60 flash chunk (cleaner) like the
        # MSE path; both are detached (FN input is never grad-on upstream).
        # Use the RAW (pre-de-drift) flash slab — the FN/cycle must learn the
        # raw student->rollout2 map, NOT G's de-drifted output (that corrupts
        # the training pairs / creates a feedback loop since G inverts F). The
        # _raw key is the pre-de-drift slab; when de-drift/apply_to_flash is OFF
        # it IS flash_dmd_gan_x0 (byte-identical fallback).
        flash = info.get("flash_dmd_gan_x0_raw")
        if flash is None:
            flash = info.get("flash_dmd_gan_x0")
        r1 = flash.detach() if flash is not None else rollout1_chunk
        if r1 is None or int(r1.shape[1]) % npb != 0:
            return _anchor("bad_r1")
        n_chunks = int(r1.shape[1]) // npb
        abs_new_start = int(info.get("abs_frame_start", 0))
        overlap = int(info.get("overlap", 0))
        chunk_abs_start = abs_new_start - overlap

        # ``forward_noiser_reverse`` (h1): flip the mapping the FN learns.
        #   forward (default): FN(clean/rollout1) -> drifted/rollout2  (ADD a
        #                      drift step; a learned CARN that reproduces the
        #                      rollout's per-step explode/harmonise modes).
        #   reverse  (=true):  FN(drifted/rollout2) -> clean/rollout1  (REMOVE
        #                      a drift step; a learned de-CARN denoiser).
        #                      Applied to GT it suppresses those modes from
        #                      the real distribution (matched-pool gt_both).
        _reverse = bool(getattr(m, "forward_noiser_reverse", False))

        # ===== carn_recurse=False: CUMULATIVE single-call FN training =====
        # forward: FN(clean GT chunk, carn_step=L) -> student-drifted chunk @L,
        # in ONE conditioned call (no rollout2 / no +1 composition). The
        # student's rollout1 chunk at abs position p IS the level-L(p) drift of
        # the clean GT at p; we pair them and condition on L(p).
        # reverse:  FN(student-drifted chunk @L, carn_step=L) -> clean GT — a
        # de-CARN denoiser conditioned on the input's drift level. Application
        # (_apply_forward_noiser_to_gt / matched-pool gt_both) passes a real
        # level (forward_noiser_apply_gt_level, default 1) so the FN removes
        # one drift-level's worth of modes. Level-0 chunks carry no drift, so
        # they are skipped (the FN stays identity at step 0) either direction.
        # chain_levels (default) SUPERSEDES this branch: training is always the
        # conditioned rollout1->rollout2 path below (model-based; no clean->drift
        # GT relation), with the per-pair cond = rollout2 output level.
        _chain_levels = bool(getattr(m, "forward_noiser_chain_levels", True))
        if not bool(getattr(m, "carn_recurse", True)) and not _chain_levels:
            ride_win = s.get("ride_latents_window")
            if ride_win is None:
                return _anchor("no_ride_window_cumulative")
            num_seed = int(m.dmd_context_clean_frames // npb)
            ride_T = int(ride_win.shape[1])
            Bc = int(r1.shape[0])
            cum_inputs, cum_targets, carn_levels = [], [], []
            for c in range(n_chunks):
                f0, f1 = c * npb, c * npb + npb
                a0, a1 = chunk_abs_start + f0, chunk_abs_start + f1
                if a0 < 0 or a1 > ride_T:
                    continue
                lvl = max(0, (a0 // npb) - (num_seed - 1))
                if lvl <= 0:
                    continue  # level-0: no drift to learn (FN identity).
                gt_chunk = ride_win[:, a0:a1].to(
                    dtype=r1.dtype, device=r1.device).detach()
                drift_chunk = r1[:, f0:f1].detach()
                if _reverse:
                    cum_inputs.append(drift_chunk)   # drifted @L  -> input
                    cum_targets.append(gt_chunk)     # clean GT    -> target
                else:
                    cum_inputs.append(gt_chunk)
                    cum_targets.append(drift_chunk)
                carn_levels.append(int(lvl))
            if not cum_inputs:
                return _anchor("no_pairs_cumulative")
            fn_in = torch.cat(cum_inputs, dim=0)          # [N*B, npb, C, H, W]
            cum_tg = torch.cat(cum_targets, dim=0)
            carn = torch.cat([
                torch.full((Bc,), lvl, dtype=torch.long, device=fn_in.device)
                for lvl in carn_levels
            ])
            return self._fn_forward_backward_with_cycle(
                fn_in, cum_tg, carn, proj,
                {
                    "train/fn_tf_pairs": float(len(cum_inputs)),
                    "train/fn_tf_skipped": 0.0,
                    "train/fn_tf_cumulative": 1.0,
                    "train/fn_tf_reverse": 1.0 if _reverse else 0.0,
                },
            )

        # carn_recurse=True (+1 composition) path: this is the ONLY consumer
        # of the prebuilt rollout2. Fetch + require it here (the False branch
        # above already returned), so the prebuild can be skipped entirely
        # when carn_recurse=False without disabling FN training.
        r2 = s.get("rollout2_x0")
        r2_abs = s.get("rollout2_abs_frame_start")
        if r2 is None or r2_abs is None:
            return _anchor("no_rollout2")
        r2_abs = int(r2_abs)
        r2_total = int(r2.shape[1])

        fn_inputs, fn_targets = [], []
        _pair_abs: list = []
        for c in range(n_chunks):
            f0, f1 = c * npb, c * npb + npb
            a0, a1 = chunk_abs_start + f0, chunk_abs_start + f1
            if a0 < r2_abs or a1 > r2_abs + r2_total:
                continue
            s0 = a0 - r2_abs
            r1_chunk = r1[:, f0:f1].detach()
            r2_chunk = r2[:, s0:s0 + npb].to(
                dtype=r1.dtype, device=r1.device).detach()
            _pair_abs.append(int(a0))
            if _reverse:
                fn_inputs.append(r2_chunk)   # input  = rollout2 (drifted)
                fn_targets.append(r1_chunk)  # target = rollout1 (cleaner)
            else:
                fn_inputs.append(r1_chunk)
                fn_targets.append(r2_chunk)
        if not fn_inputs:
            return _anchor("no_pairs")
        # Probe: where the SETUP-WINDOW FN pairs live (abs ride frames).
        # Expected under rolling: only each ride's FIRST window yields
        # pairs (rollout2 covers the setup span) — the frontier pairs
        # ([FN-PAIR]) are the ones that advance with the generator.
        if (getattr(self, "is_main_process", True)
                and getattr(self, "_fn_tf_pos_dbg", 0) < 6):
            self._fn_tf_pos_dbg = getattr(self, "_fn_tf_pos_dbg", 0) + 1
            import sys as _sys
            print(
                f"[FN-TF] setup-window pairs n={len(_pair_abs)} "
                f"r1_abs={_pair_abs} r2_span=[{r2_abs},{r2_abs + r2_total})",
                file=_sys.stderr, flush=True,
            )
        fn_in = torch.cat(fn_inputs, dim=0)       # [N*B, npb, C, H, W]
        fn_tg = torch.cat(fn_targets, dim=0)
        # Conditioning. chain_levels (default): condition each pair on its
        # ROLLOUT2 OUTPUT LEVEL L2 = (a0 - r2_abs)//npb + 1 (slot 0 of r2 is
        # the first generated chunk = level 1), so cond=k learns the increment
        # that lands at level k (GT->1=1, GT->2=2, 1->3=3, 2->4=4, ...). Each
        # pair contributes Bc consecutive rows, so repeat its cond Bc times.
        # Legacy: step-UNCONDITIONED, carn_step=0 (generic +/-1 shift).
        Bc = int(r1.shape[0])
        if _chain_levels:
            _conds = [max(0, (int(_a0) - r2_abs) // npb + 1)
                      for _a0 in _pair_abs]
            carn = torch.cat([
                torch.full((Bc,), int(_c), dtype=torch.long,
                           device=fn_in.device)
                for _c in _conds
            ])
            if (getattr(self, "is_main_process", True)
                    and getattr(self, "_fn_chain_train_dbg", 0) < 4):
                self._fn_chain_train_dbg = getattr(
                    self, "_fn_chain_train_dbg", 0) + 1
                import sys as _sys
                print(
                    f"[FN-CHAIN-TRAIN] conditioned rollout1->rollout2: "
                    f"n_pairs={len(_pair_abs)} r2_abs={r2_abs} "
                    f"conds(L2)={_conds}",
                    file=_sys.stderr, flush=True,
                )
        else:
            carn = torch.zeros(
                (fn_in.shape[0],), dtype=torch.long, device=fn_in.device,
            )
        return self._fn_forward_backward_with_cycle(
            fn_in, fn_tg, carn, proj,
            {
                "train/fn_tf_pairs": float(len(fn_inputs)),
                "train/fn_tf_skipped": 0.0,
                "train/fn_tf_reverse": 1.0 if _reverse else 0.0,
                "train/fn_tf_chain_levels": 1.0 if _chain_levels else 0.0,
            },
        )

    #     "student vs teacher's clean data".
    #   * "adjacent_chunks" (ASD-style): real = chunk_i, fake =
    #     chunk_{i+1} from the same gen rollout. Pushes
    #     chunk_{i+1}'s distribution toward chunk_i's.
    # When both flags are on, two D updates run per iter and both
    # gen-side losses sum into the generator's total.
    def _compute_ladd_losses(
        self,
        pred_image: torch.Tensor,
        gt_latents_window: torch.Tensor,
        current_step: int,
        flash_dmd_gan_x0: Optional[torch.Tensor] = None,
    ) -> tuple:
        device = pred_image.device
        zero = torch.zeros((), device=device, dtype=torch.float32)
        if not self.gan_enabled or self.r3gan_disc is None:
            return zero, {}

        # Source latent for the FAKE side (gradient-bearing for the gen).
        if flash_dmd_gan_x0 is not None:
            src_grad = flash_dmd_gan_x0.to(torch.float32)
        else:
            src_grad = pred_image.to(torch.float32)
        src_detached = src_grad.detach()
        # GT latent for the REAL side in gt_vs_fake mode. Always detached.
        gt_detached = gt_latents_window.detach().to(torch.float32)

        # Mode toggles. Defaults preserve backward compat with v28:
        # adjacent on by default, gt_vs_fake off by default.
        gt_vs_fake_enabled = bool(
            getattr(self.model, "ladd_gt_vs_fake_enabled", False)
        )
        adj_enabled = bool(
            getattr(self.model, "ladd_adjacent_chunks_enabled", True)
        )
        # gt_transition (v28G): action-conditioned GT-supervised adjacent-
        # chunk transition matching. Concatenates (chunk_n, chunk_{n+1})
        # along the F dim, feeds GT on the real side and student on the
        # fake side. The WAN teacher projector already conditions on per-
        # frame action_modulation, so the disc sees the action context
        # naturally — no extra action-projector needed in the disc.
        # Strictly more informative than adjacent_chunks (which is
        # student-vs-student, action-blind); typically run with
        # adjacent_chunks disabled.
        gt_xn_enabled = bool(
            getattr(self.model, "ladd_gt_transition_enabled", False)
        )
        if getattr(self, "_ladd_diag_n", 0) < 4:
            self._ladd_diag_n = getattr(self, "_ladd_diag_n", 0) + 1
            import sys as _sys
            print(
                f"[LADD-DIAG] _compute_ladd_losses REACHED: gt_vs_fake={gt_vs_fake_enabled} "
                f"adj={adj_enabled} gt_xn={gt_xn_enabled} "
                f"(model.ladd_gt_transition_enabled="
                f"{getattr(self.model, 'ladd_gt_transition_enabled', 'MISSING')})",
                file=_sys.stderr, flush=True,
            )
        if not (gt_vs_fake_enabled or adj_enabled or gt_xn_enabled):
            return zero, {}

        gen_loss_total = zero
        logs: dict = {}
        # WP-14B unweighted-ratio call site: reset every call so a call
        # that never reaches a weighted ``generator_gan_loss`` (all modes
        # warmup-gated, or a d_only-only call) cannot leave the A7 caller
        # reading a stale weight from a PREVIOUS step. Set to a real float
        # only inside ``_ladd_run_pair_mode``'s weighted-assignment branch.
        self._ladd_last_gen_gan_weight = None
        self._ladd_last_gen_gan_weight_mode = None
        self._ladd_last_gen_gan_stat_value = None
        self._ladd_last_total_weight = None

        # Two-phase orchestration when MORE THAN ONE pair-mode is
        # enabled (gt_vs_fake + adjacent_chunks): we must run ALL
        # D-updates before ANY gen-side forward, otherwise mode-2's
        # D-update mutates ``spectral_norm._u`` between mode-1's
        # gen-side forward and the outer ``generator_loss.backward()``
        # → version-counter mismatch on saved tensors. The
        # ``_ladd_run_pair_mode`` ``phase`` parameter implements the
        # split: "d_only" runs only the D-update, "g_only" runs only
        # the gen-side, "both" (default) runs both in sequence. The
        # G-side ALSO runs in disc.eval() mode (set inside
        # ``_ladd_run_pair_mode``) so spectral_norm doesn't mutate
        # the running buffers during its forward.
        enabled_modes: List[Tuple[str, torch.Tensor, str]] = []
        if gt_vs_fake_enabled:
            enabled_modes.append(("gt_vs_fake", gt_detached, "_gt"))
        if adj_enabled:
            enabled_modes.append(("adjacent_chunks", src_detached, "_adj"))
        if gt_xn_enabled:
            # Real source is GT for gt_transition (the whole point).
            enabled_modes.append(("gt_transition", gt_detached, "_gtxn"))

        # WP-14B unweighted-ratio call site: the division below is exact
        # only when exactly ONE pair-mode is active (otherwise
        # ``gen_gan_loss`` is a SUM of differently-weighted per-mode
        # terms and no single scalar recovers an unweighted value). The
        # A7 caller reads this to decide whether to compute the ratio at
        # all, rather than silently mis-scoping a multi-mode combination
        # as if it were one term.
        self._ladd_last_mode_count = len(enabled_modes)

        two_phase = len(enabled_modes) > 1

        # In two-phase (multi-mode) runs the D-update happens in the
        # ``d_only`` call below and the ``g_only`` call leaves ``last_d_*``
        # at their 0.0 init — so without capturing these, the merged logs
        # report d_real/d_fake/d_loss/R1 as 0 for every mode (the bug that
        # made smoke B's gtxn gap read 0.000). Stash the d_only logs and
        # overlay the D-side metrics after the g_only merge below.
        _d_only_logs: dict = {}
        if two_phase:
            # Phase 1: all D-updates (each backward + step immediately).
            for mode_name, real_src_t, _suffix in enabled_modes:
                _, _dlog = self._ladd_run_pair_mode(
                    real_src=real_src_t,
                    fake_src_grad=src_grad,
                    fake_src_detached=src_detached,
                    pred_image_dtype=pred_image.dtype,
                    pair_mode=mode_name,
                    current_step=current_step,
                    phase="d_only",
                )
                _d_only_logs[mode_name] = _dlog or {}

        # Phase 2: per-mode gen-side forwards (or single combined
        # call when only one mode is enabled).
        for mode_name, real_src_t, suffix in enabled_modes:
            g_loss, partial_logs = self._ladd_run_pair_mode(
                real_src=real_src_t,
                fake_src_grad=src_grad,
                fake_src_detached=src_detached,
                pred_image_dtype=pred_image.dtype,
                pair_mode=mode_name,
                current_step=current_step,
                phase="g_only" if two_phase else "both",
            )
            if mode_name == "gt_vs_fake":
                w = float(getattr(self.model, "ladd_gt_vs_fake_weight", 1.0))
            elif mode_name == "gt_transition":
                w = float(
                    getattr(self.model, "ladd_gt_transition_weight", 1.0)
                )
            else:
                w = float(getattr(self.model, "ladd_adjacent_chunks_weight", 1.0))
            # WP-14B unweighted-ratio call site: fold this mode's
            # per-mode weight into the stash ``_ladd_run_pair_mode`` just
            # left for us, giving the A7 caller the TOTAL multiplier
            # (``w * gen_gan_weight``) applied to this mode's raw g_rp on
            # its path into ``gen_loss_total``. Guarded on the mode
            # actually matching -- a mismatch means the stash is stale
            # (e.g. this mode's call never reached the weighted branch,
            # so ``_ladd_last_gen_gan_weight_mode`` still names whichever
            # mode set it last) and must not be read as this mode's value.
            if self._ladd_last_gen_gan_weight_mode == mode_name:
                self._ladd_last_total_weight = (
                    w * self._ladd_last_gen_gan_weight)
            gen_loss_total = gen_loss_total + w * g_loss
            for k, v in partial_logs.items():
                logs[k + suffix] = v
            # Overlay D-side metrics from the d_only run — the g_only call
            # above left them at their 0.0 init in two-phase mode. Single-
            # mode runs use phase="both" (no _d_only_logs entry) and are
            # untouched.
            if two_phase and mode_name in _d_only_logs:
                _dl = _d_only_logs[mode_name]
                for _dk in (
                    "train/r3gan_disc_skipped", "train/r3gan_d_loss",
                    "train/r3gan_d_real", "train/r3gan_d_fake_detached",
                    "train/r3gan_d_loss_stat",
                    "train/r3gan_r1", "train/r3gan_r1_grad_sq",
                    "train/r3gan_r1_fired",
                    # A16-D1: the per-block R1 fire count and the per-mode
                    # penalty counters are produced by the D-update, which in
                    # two-phase runs happens ONLY in the ``d_only`` call.
                    "train/r3gan_r1_block_fires",
                    "train/r3gan_r1_mode_updates_total",
                    "train/r3gan_r1_mode_penalties_total",
                    "train/r3gan_r1_mode_fire_rate",
                    "train/ladd_match_n_real", "train/ladd_n_real",
                ):
                    if _dk in _dl:
                        logs[_dk + suffix] = _dl[_dk]
                # ---- A6/D2 (2026-08-23): real-diversity keys are written by
                # the matched ``_match_select`` draw, which in a two-phase
                # (multi-mode) run happens in the ``d_only`` call. The
                # whitelist above enumerated only ``r3gan_*`` /
                # ``ladd_*_n_real``, so every ``gan_real_div_*`` /
                # ``gan_real_*`` key produced by the D-update was DROPPED, and
                # the ``g_only`` call emitted none of its own during critic
                # warmup / zero gen weight. Net effect with
                # ``gan_real_diversity_log=true`` and two modes: ZERO A6 keys,
                # indistinguishable from the flag being off. Overlay by PREFIX
                # so future A6 keys are covered automatically. When
                # ``gan_real_diversity_log`` is off no such key exists and this
                # loop is a no-op.
                for _dk, _dv in _dl.items():
                    if _dk.startswith("train/gan_real_"):
                        # The D-update's draw is authoritative: it is what the
                        # discriminator actually trained against this step.
                        logs[_dk + suffix] = _dv

        # Top-level (non-suffixed) R1 traces so the wandb plot is
        # findable as ``train/r3gan_r1*`` without the ``_gt`` / ``_adj``
        # suffix dance. Aggregate via mean across pair modes — both
        # modes share the same disc weights so their R1 estimates
        # should be comparable. The fired flag is OR'd (firing on EITHER
        # mode is enough to count this iter as "R1 fired"). ``grad_sq``
        # uses nan-aware mean so a non-firing mode doesn't pull the
        # combined value to ~half.
        import math as _math
        r1_keys = [k for k in logs if "r3gan_r1_grad_sq_" in k]
        if r1_keys:
            valid_vals = [
                logs[k] for k in r1_keys
                if isinstance(logs[k], float) and not _math.isnan(logs[k])
            ]
            if valid_vals:
                logs["train/r3gan_r1_grad_sq"] = sum(valid_vals) / len(valid_vals)
            else:
                logs["train/r3gan_r1_grad_sq"] = float("nan")
        _rt = getattr(self, "_ladd_ring_telemetry", None)
        if _rt:
            for _k, _v in _rt.items():
                logs[f"train/ladd_{_k}"] = float(_v)
        # A16-D1: EXACT key match. The old ``"r3gan_r1_fired_" in k``
        # substring test also matched ``train/r3gan_r1_fired_total_<mode>`` --
        # a MONOTONE COUNTER -- so the top-level 0/1 gauge reported the
        # cumulative fire count (hundreds) instead of "did R1 fire this step".
        fired_keys = [
            k for k in (
                "train/r3gan_r1_fired_gt",
                "train/r3gan_r1_fired_adj",
                "train/r3gan_r1_fired_gtxn",
            ) if k in logs
        ]
        if fired_keys:
            logs["train/r3gan_r1_fired"] = max(logs[k] for k in fired_keys)
        # A16-D1: per-mode R1 fire rates aggregated to the top level. The OR'd
        # ``r3gan_r1_fired`` above cannot show failure (b) (one mode starved
        # while another fires); the MIN of the per-mode lifetime fire rates
        # can -- it reads 0.0 for as long as ANY enabled mode has never had a
        # single R1 applied.
        _mode_rate_keys = [
            k for k in (
                "train/r3gan_r1_mode_fire_rate_gt",
                "train/r3gan_r1_mode_fire_rate_adj",
                "train/r3gan_r1_mode_fire_rate_gtxn",
            ) if k in logs
        ]
        if _mode_rate_keys:
            logs["train/r3gan_r1_fire_rate_min_mode"] = min(
                logs[k] for k in _mode_rate_keys)
            logs["train/r3gan_r1_modes_with_penalty"] = float(sum(
                1 for k in _mode_rate_keys if logs[k] > 0.0))
        # Top-level r3gan_r1 = mean of the γ-weighted per-mode losses.
        r1_loss_keys = [
            k for k in logs
            if k in (
                "train/r3gan_r1_gt",
                "train/r3gan_r1_adj",
                "train/r3gan_r1_gtxn",
            )
        ]
        if r1_loss_keys:
            logs["train/r3gan_r1"] = sum(logs[k] for k in r1_loss_keys) / len(r1_loss_keys)
        return gen_loss_total, logs

    # ------------------------------------------------------------------
    # A16-D1 (2026-08-23): R1 cadence semantics, stated ONCE, here.
    # ------------------------------------------------------------------
    # ``ladd_r1_every_n_steps`` (N) is a cadence over TRAINING STEPS, not
    # over D-update iterations and not over pair modes. The contract:
    #
    #   * On a step where a given pair mode's R1 debt is due, R1 fires on
    #     EVERY one of that mode's ``gan_updates_per_step`` (S) D-updates.
    #     Lazy-R1 amortises COST over steps (StyleGAN2 lazy regularisation);
    #     it was never meant to thin the penalty WITHIN a step. N=1 must mean
    #     "R1 on every D-update", which is what the pre-unification positional
    #     modulo already did (S of S).
    #   * The debt latch is keyed by ``pair_mode``. Each enabled mode
    #     (gt_vs_fake / adjacent_chunks / gt_transition) trains the shared
    #     disc against its OWN real distribution, so each needs its own
    #     gradient penalty; one mode firing must never spend another mode's
    #     debt. With M modes enabled, a due step applies M*S R1 penalties.
    #   * The latch is NOT keyed by path (matched vs positional). Within one
    #     step a mode runs EITHER the matched branch OR the positional branch
    #     (the matched branch returns before the positional loop), and across
    #     steps a ``[LADD-MATCH-FALLBACK]`` must not reset the cadence: it is
    #     the same D head either way. Keying by path would let a fallback
    #     step fire R1 immediately after a matched fire, and would restore
    #     exactly the double-fire A16 set out to remove.
    #   * The latch is written ONCE PER D-UPDATE BLOCK (i.e. once per
    #     ``(step, pair_mode)``), BEFORE the ``for _it in range(S)`` loop --
    #     never inside it. Writing it inside the loop is what made the first
    #     iteration spend the debt and starve iterations 2..S.
    #
    # Debt (``step - last_fire >= N``) rather than ``step % N``: the GAN only
    # runs on generator iters, so an exact modulo can be missed forever when
    # N and ``dfake_gen_update_ratio`` are not commensurate.
    #
    # FIRE-RATE TABLE -- R1 applications per DUE step (S = gan_updates_per_step,
    # M = number of enabled pair modes):
    #
    #   path        unified=False (legacy)      unified=True (before D1)   unified=True (after D1)
    #   positional  M*S  (step % N, per iter)   1  (global latch, 1st iter)  M*S
    #   matched     1    (global latch)         1                            M*S
    #   mixed       M*S + 1 (can double-fire)   1                            M*S
    #
    # ``ladd_r1_unified_cadence`` stays DEFAULT FALSE, and when false every
    # expression below is the legacy one, evaluated verbatim.
    # ------------------------------------------------------------------
    def _ladd_r1_block_due(
        self, pair_mode: str, current_step: int, every_n: int
    ) -> bool:
        """Per-``pair_mode`` R1 debt latch for ONE D-update block.

        Call EXACTLY ONCE per ``(current_step, pair_mode)`` D-update block,
        before the ``gan_updates_per_step`` loop; the returned decision then
        applies to every iteration of that loop. Returns True and spends the
        debt, or False and leaves it standing.
        """
        store = getattr(self, "_ladd_last_r1_step_by_mode", None)
        if store is None:
            store = {}
            self._ladd_last_r1_step_by_mode = store
        key = str(pair_mode)
        every_n = max(1, int(every_n))
        last = int(store.get(key, -10 ** 9))
        due = (int(current_step) - last) >= every_n
        if due:
            store[key] = int(current_step)
            # Keep the legacy single-slot latch in sync so any reader of
            # ``_ladd_last_r1_step`` (telemetry, resume logic) still sees the
            # most recent firing.
            self._ladd_last_r1_step = int(current_step)
        return due

    def _ladd_count_penalty_fires(
        self, do_r1: bool, pair_mode: Optional[str] = None
    ) -> None:
        """Monotone counters of R1 firings and of D-updates.

        The per-step ``r3gan_r1_fired`` gauge is sampled by wandb every N
        steps, and N aliases with BOTH the penalty cadence and the
        generator-update cadence (``dfake_gen_update_ratio``), so a logged
        0 is indistinguishable from "R1 never fires at all". These counters
        are monotone, so a single logged sample settles it:
        ``r1_fired_total == 0`` late in a run means R1 genuinely never fired.

        A16-D1: also counts PER PAIR MODE. The global counters cannot show
        the (b) failure (one mode's R1 permanently zero while the OR'd
        top-level gauge reads 1); the per-mode pair can, and
        ``_ladd_penalty_fire_logs(pair_mode)`` surfaces it.
        Pure telemetry -- nothing here touches the loss.
        """
        self._ladd_dupdate_total = getattr(self, "_ladd_dupdate_total", 0) + 1
        if do_r1:
            self._ladd_r1_fire_total = getattr(self, "_ladd_r1_fire_total", 0) + 1
        if pair_mode is None:
            return
        _du = getattr(self, "_ladd_dupdate_by_mode", None)
        if _du is None:
            _du = {}
            self._ladd_dupdate_by_mode = _du
        _fi = getattr(self, "_ladd_r1_fire_by_mode", None)
        if _fi is None:
            _fi = {}
            self._ladd_r1_fire_by_mode = _fi
        key = str(pair_mode)
        _du[key] = _du.get(key, 0) + 1
        _fi[key] = _fi.get(key, 0) + (1 if do_r1 else 0)

    def _ladd_penalty_fire_logs(self, pair_mode: Optional[str] = None) -> dict:
        n = getattr(self, "_ladd_dupdate_total", 0)
        r1 = getattr(self, "_ladd_r1_fire_total", 0)
        out = {
            "train/r3gan_disc_updates_total": float(n),
            "train/r3gan_r1_fired_total": float(r1),
            "train/r3gan_r1_fire_rate": float(r1) / n if n else 0.0,
        }
        if pair_mode is None:
            return out
        # A16-D1 per-mode telemetry. ``_compute_ladd_losses`` appends the
        # mode suffix to every returned key, so these read as e.g.
        # ``train/r3gan_r1_mode_fire_rate_gtxn``. Deliberately NOT named
        # ``*r3gan_r1_fired_*``: the top-level aggregation matches on that
        # substring and would otherwise pick up a monotone counter.
        _du = getattr(self, "_ladd_dupdate_by_mode", None) or {}
        _fi = getattr(self, "_ladd_r1_fire_by_mode", None) or {}
        key = str(pair_mode)
        nm = int(_du.get(key, 0))
        rm = int(_fi.get(key, 0))
        out["train/r3gan_r1_mode_updates_total"] = float(nm)
        out["train/r3gan_r1_mode_penalties_total"] = float(rm)
        out["train/r3gan_r1_mode_fire_rate"] = (
            float(rm) / nm if nm else 0.0
        )
        return out

    # ------------------------------------------------------------------
    # A6-D2 (2026-08-23): real-diversity telemetry must survive the
    # two-phase (multi-pair-mode) split.
    # ------------------------------------------------------------------
    # ``_match_select`` -- the only writer -- runs in the ``d_only`` call
    # (the D-update) and, only once the critic warmup is over AND the gen GAN
    # weight is non-zero, in the ``g_only`` call. ``_ladd_run_pair_mode``
    # reset the attribute to None at entry of BOTH calls, and the two-phase
    # overlay whitelist in ``_compute_ladd_losses`` copied only ``r3gan_*`` /
    # ``ladd_*_n_real`` keys back from the ``d_only`` logs. Result with
    # ``gan_real_diversity_log=true`` and >1 pair mode: the d_only draw's keys
    # were dropped by the whitelist and the g_only call had nothing to emit --
    # ZERO A6 keys, indistinguishable from the flag being off.
    #
    # Fix, two halves: the overlay is now prefix-based (see
    # ``_compute_ladd_losses``), and the draw is stashed PER STEP PER
    # PAIR MODE here, so a call that does not itself draw (g_only during
    # warmup) can still report this step's D-update draw. Cross-step
    # staleness -- the bug the original A6 reset removed -- is still
    # impossible: the store is discarded the moment ``current_step`` changes.
    def _ladd_stash_real_div(
        self, pair_mode: str, current_step: int, payload: dict
    ) -> None:
        self._ladd_real_div_telemetry = payload
        if getattr(self, "_ladd_real_div_step", None) != int(current_step):
            self._ladd_real_div_step = int(current_step)
            self._ladd_real_div_by_mode = {}
        store = getattr(self, "_ladd_real_div_by_mode", None)
        if store is None:
            store = {}
            self._ladd_real_div_by_mode = store
        store[str(pair_mode)] = payload

    def _ladd_real_div_logs(self, pair_mode: str, current_step: int) -> dict:
        """``train/gan_real_*`` keys for this (step, pair_mode), or ``{}``."""
        cur = getattr(self, "_ladd_real_div_telemetry", None)
        if not cur and (
                getattr(self, "_ladd_real_div_step", None) == int(current_step)):
            cur = (getattr(self, "_ladd_real_div_by_mode", None)
                   or {}).get(str(pair_mode))
        return {
            f"train/{_dk}": float(_dv) for _dk, _dv in (cur or {}).items()
        }

    def _ladd_disc_update_microbatched(
        self, *, _it, _rn, _fk, _fseg, _fseg_act, real_m, real_m_rat,
        real_m_ram, group_flat, _do_r1, n_pairs, B, Kk, _m_fwd,
        real_m_rpe=None,
        disc_for_update, _disc_no_sync, rpgan_d_loss, _K_stat, _W_stat,
        _r1_sigma, _r1_gamma, _r1_num_samples,
        current_step, _micro_groups,
    ):
        """Memory-fix micro-batched D-update (ON path; default-OFF leaves
        the verbatim single-batch code untouched).

        Splits the ``n_fake`` fakes into ``_micro_groups`` roughly-equal
        groups and does ONE disc forward+backward per group with gradient
        accumulation, so the transient backward peak (gradient-checkpointed
        teacher disc-backbone recompute over the whole batch) is cut by the
        group factor. The optimizer steps ONCE after all groups, exactly as
        the single-batch path.

        MATH FAITHFULNESS:
          * RpGAN (``_m_rp``): each fake is scored ONLY against its own Kk
            matched reals (block-diagonal), so the loss is a MEAN over
            ``n_fake*Kk`` independent rows laid out FAKE-MAJOR. A fake group
            owns a contiguous block of fake rows -> a contiguous block of
            ``Kk``-row chunks of ``group_flat``. Each group's partial loss is
            ``softplus(rg-fg).SUM()/N_total`` (NOT /N_group); summing the
            per-group backwards == the full ``.mean().backward()`` bit-for-
            bit (up to fp summation order). The optional K_stat/W_stat
            channel-split term is the same row-mean on a disjoint token slice
            and is split identically.
          * FD-R1 (real-side penalty): a mean over the (subsampled) unique
            reals. R1 reals are chosen GLOBALLY (deterministic subsample,
            seeded by current_step), then each is forwarded in the FIRST
            group whose fakes reference it (so it is perturbed-forwarded once
            only); partials are ``sum-of-sq / M`` and sum to the full R1.
            ``_r1_num_samples<=0`` => all unique reals (current behaviour).

        DDP: with multiple ``.backward()`` per step, DDP all-reduces on each
        backward unless wrapped in ``no_sync()``. We wrap groups
        ``0..G-2`` in ``_disc_no_sync()`` and let the LAST group's backward
        do the (single) all-reduce — the standard grad-accumulation pattern.

        Returns a dict of the SAME logging scalars the inline path writes
        (aggregated: d_loss/d_loss_stat are the full-batch means, d_real/
        d_fake are full-batch logit means, r1 is the full penalty).
        """
        device = _rn.device
        n_fake = n_pairs * B
        # group_flat is fake-major: row r in [0, n_fake*Kk) belongs to fake
        # r // Kk. Split fakes into _micro_groups contiguous, ~equal groups.
        G = max(1, min(int(_micro_groups), n_fake))
        bounds = [(g * n_fake) // G for g in range(G + 1)]  # fake-index cuts
        N_total = n_fake * Kk  # total RpGAN rows
        n_uniq = int(real_m.shape[0])

        # ----- R1 global real subsample (deterministic, DDP-consistent) -----
        r1_rows = None        # LongTensor of unique-real indices to penalise
        M_r1 = 0
        if _do_r1:
            if _r1_num_samples is not None and int(_r1_num_samples) > 0 \
                    and int(_r1_num_samples) < n_uniq:
                gseed = int(current_step) * 1000003 + 7 + int(_it) * 9176
                g_cpu = torch.Generator(device="cpu").manual_seed(gseed)
                perm = torch.randperm(n_uniq, generator=g_cpu)
                r1_rows = perm[:int(_r1_num_samples)].sort().values.to(device)
            else:
                r1_rows = torch.arange(n_uniq, device=device)
            M_r1 = int(r1_rows.shape[0])
        # Map: which group "owns" the perturbed-forward of each R1 real (the
        # FIRST group whose fake block references it). Reals never referenced
        # by any group's fakes (cannot happen for picked reals) are skipped.
        r1_owner = {}
        if _do_r1 and M_r1 > 0:
            r1_set = set(int(x) for x in r1_rows.tolist())
            gf_cpu = group_flat.to("cpu")
            for g in range(G):
                lo, hi = bounds[g], bounds[g + 1]
                if hi <= lo:
                    continue
                rows = gf_cpu[lo * Kk: hi * Kk].tolist()
                for u in rows:
                    if u in r1_set and u not in r1_owner:
                        r1_owner[u] = g

        # R1 real-side eps (one draw, indexed per group below).
        eps_r_full = None
        if _do_r1 and M_r1 > 0:
            eps_r_full = _r1_sigma * torch.randn_like(_rn)

        # Accumulators for logging (full-batch equivalents).
        sum_d_r = torch.zeros((), device=device, dtype=torch.float64)
        cnt_d_r = 0
        sum_d_f = torch.zeros((), device=device, dtype=torch.float64)
        cnt_d_f = 0
        d_loss_acc = 0.0
        d_loss_stat_acc = 0.0
        r1_acc = 0.0
        r1_gsq_acc = 0.0

        self._mem_step_snapshot(
            f"disc_it{_it}_micro_pre_groups_G{G}_r1{int(_do_r1)}")
        for g in range(G):
            lo, hi = bounds[g], bounds[g + 1]
            if hi <= lo:
                continue
            n_fake_g = hi - lo
            # Group fakes (and their actions).
            fk_g = _fk[lo:hi]
            fa0_g = (_fseg_act[0][lo:hi]
                     if _fseg_act[0] is not None else None)
            fa1_g = (_fseg_act[1][lo:hi]
                     if _fseg_act[1] is not None else None)
            # Group's slice of group_flat (global unique-real indices), then
            # re-index into a GROUP-LOCAL unique-real set.
            gflat_g = group_flat[lo * Kk: hi * Kk]            # [n_fake_g*Kk]
            local_uniq, gflat_local = torch.unique(
                gflat_g, sorted=True, return_inverse=True)
            ru_g = _rn.index_select(0, local_uniq)
            rat_g = (real_m_rat.index_select(0, local_uniq)
                     if real_m_rat is not None else None)
            ram_g = (real_m_ram.index_select(0, local_uniq)
                     if real_m_ram is not None else None)
            # Phase B fix 1: per-row prompt embeds for the ring reals
            # (None => _m_fwd's legacy current-prompt broadcast).
            rpe_g = (real_m_rpe.index_select(0, local_uniq)
                     if real_m_rpe is not None else None)

            # R1 reals OWNED by this group (perturbed-forwarded here).
            r1_local = None
            if _do_r1 and M_r1 > 0:
                owned = [u for u in local_uniq.tolist()
                         if r1_owner.get(int(u)) == g]
                if owned:
                    owned_t = torch.tensor(
                        sorted(owned), device=device, dtype=torch.long)
                    # position of each owned global-uniq id within local set
                    pos = torch.searchsorted(local_uniq, owned_t)
                    r1_local = (owned_t, pos)

            _segs = [(ru_g, rat_g, ram_g, rpe_g), (fk_g, fa0_g, fa1_g)]
            if r1_local is not None:
                owned_t, pos = r1_local
                rn_pert = ru_g.index_select(0, pos) \
                    + eps_r_full.index_select(0, owned_t)
                # The R1 perturbed segment covers only the OWNED subset of
                # this group's unique reals (``pos`` rows), so its per-row
                # action tokens/modulation must be sliced to ``pos`` too —
                # otherwise the disc forward gets full-group modulation rows
                # against a ``pos``-row latent batch (shape mismatch).
                rat_pert = (rat_g.index_select(0, pos)
                            if rat_g is not None else None)
                ram_pert = (ram_g.index_select(0, pos)
                            if ram_g is not None else None)
                rpe_pert = (rpe_g.index_select(0, pos)
                            if rpe_g is not None else None)
                _segs.append((rn_pert, rat_pert, ram_pert, rpe_pert))

            _outs = _m_fwd(disc_for_update, _segs)
            d_r_g, d_f_g = _outs[0], _outs[1]
            _nxt = 2
            # ---- block-diagonal RpGAN partial (scaled by 1/N_total) ----
            rg = d_r_g.index_select(0, gflat_local)          # [n_fake_g*Kk,t]
            fg = d_f_g.repeat_interleave(Kk, dim=0)
            if _K_stat > 0 and _W_stat > 0.0:
                rv, rs = rg[:, :-_K_stat], rg[:, -_K_stat:]
                fv, fs = fg[:, :-_K_stat], fg[:, -_K_stat:]
                # softplus(rv-fv).sum()/N_total per token col, then /tokcols
                # equals the contribution to mfn(rv,fv).mean()*(N_g/N)*...
                # Match _m_rp EXACTLY: lv = softplus(rg-fg).mean() over
                # ALL elements (rows*tokcols). So partial = SUM/total_elems.
                ltot_v = float(rv.numel()) / float(n_fake_g) * n_fake  # =N_total*vtok
                ltot_s = float(rs.numel()) / float(n_fake_g) * n_fake
                # NB: rpgan_d_loss(real, fake) = softplus(fake - real).mean()
                # (the SIGN is fake-minus-real); we replicate it as a SUM
                # over all elements scaled by 1/total_elems so per-group
                # backwards SUM to the full ``.mean()``.
                lv_p = torch.nn.functional.softplus(fv - rv).sum() / ltot_v
                ls_p = torch.nn.functional.softplus(fs - rs).sum() / ltot_s
                d_rp_g = lv_p + _W_stat * ls_p
                d_loss_stat_acc += float(ls_p.detach().item())
            else:
                ltot = float(rg.numel()) / float(n_fake_g) * n_fake
                d_rp_g = torch.nn.functional.softplus(fg - rg).sum() / ltot
            d_loss_acc += float(d_rp_g.detach().item())

            # ---- R1 partial (sum-of-sq over owned reals / M_r1) ----
            if r1_local is not None:
                owned_t, pos = r1_local
                d_r_pert_g = _outs[_nxt]
                _nxt += 1
                d_r_owned = d_r_g.index_select(0, pos)
                # Token-normalize (ladd_r1_normalize_tokens, default off): see
                # the inline R1 site — divide the FD by the token count so
                # grad_sq estimates ‖∇ MEAN_i D_i‖², not ‖∇ Σ_i D_i‖².
                _r1_tok = (
                    float(d_r_owned.shape[1]) if bool(getattr(
                        self.config, "ladd_r1_normalize_tokens",
                        getattr(self.model, "ladd_r1_normalize_tokens", False)))
                    else 1.0)
                gsq_terms = ((d_r_pert_g.sum(dim=1) - d_r_owned.sum(dim=1))
                             / (_r1_sigma * _r1_tok)).pow(2)
                gsq_part = gsq_terms.sum() / float(M_r1)
                r1_g = 0.5 * _r1_gamma * gsq_part
                r1_acc += float(r1_g.detach().item())
                r1_gsq_acc += float(gsq_part.detach().item())
            else:
                r1_g = d_r_g.sum() * 0.0

            # ---- accumulate logit-mean stats (full-batch d_real/d_fake) ----
            sum_d_r += d_r_g.detach().double().sum()
            cnt_d_r += int(d_r_g.numel())
            sum_d_f += d_f_g.detach().double().sum()
            cnt_d_f += int(d_f_g.numel())

            loss_g = d_rp_g + r1_g
            is_last = (g == G - 1)
            if is_last:
                loss_g.backward()
            else:
                with _disc_no_sync():
                    loss_g.backward()
        self._mem_step_snapshot(f"disc_it{_it}_micro_post_groups")

        if self.gan_max_grad_norm and self.gan_max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.r3gan_disc.parameters()
                 if p.grad is not None],
                self.gan_max_grad_norm)
        self.r3gan_optimizer.step()

        return {
            "d_loss": d_loss_acc,
            "d_loss_stat": d_loss_stat_acc,
            "d_real": float((sum_d_r / max(1, cnt_d_r)).item()),
            "d_fake": float((sum_d_f / max(1, cnt_d_f)).item()),
            "r1": r1_acc,
            "r1_grad_sq": r1_gsq_acc if (_do_r1 and M_r1 > 0) else 0.0,
            "r1_fired": 1.0 if (_do_r1 and M_r1 > 0) else 0.0,
        }

    def _ladd_run_pair_mode(
        self,
        real_src: torch.Tensor,
        fake_src_grad: torch.Tensor,
        fake_src_detached: torch.Tensor,
        pred_image_dtype: torch.dtype,
        pair_mode: str,
        current_step: int,
        phase: str = "both",
    ) -> tuple:
        """Single D-update + gen-side loss for one pair-construction mode.

        Args:
            real_src: detached tensor that supplies the REAL chunks.
                For ``gt_vs_fake`` this is the GT latent window; for
                ``adjacent_chunks`` it is the (detached) source latent.
            fake_src_grad: gradient-bearing source for the FAKE chunks
                on the gen-side forward.
            fake_src_detached: detached source for the FAKE chunks on
                the D-side forward.
            pair_mode: ``"gt_vs_fake"`` or ``"adjacent_chunks"``.
            current_step: training step (for warmup ramp + RNG seed).

        Returns ``(gen_gan_loss, logs)``. ``gen_gan_loss`` is graph-
        attached through ``fake_src_grad``.
        """
        from model.r3gan import rpgan_d_loss, rpgan_g_loss, r1_penalty
        from model.ladd_disc import latent_diff_augment

        # ---- A6 (2026-08-23): reset the real-diversity telemetry ---------
        # ``_ladd_real_div_telemetry`` is written inside the matched
        # ``_match_select`` closure and read when the logs dict is built. It
        # was NEVER cleared, so any step that did NOT run a matched real draw
        # (D-update skipped during ``gan_disc_start_step`` warmup, gen-side
        # gated off, a positional/fallback step, a mode that returned early)
        # re-emitted the PREVIOUS step's counts as if they were fresh. Clear
        # it at entry so a missing draw shows up as absent keys (or the
        # explicit positional marker below) instead of stale ones. Pure
        # telemetry: with ``gan_real_diversity_log`` off (default) nothing
        # ever writes the attribute, so this is a no-op.
        #
        # D2 (2026-08-23): the clear is still cross-STEP correct, but a
        # WITHIN-step fallback now exists -- ``_ladd_stash_real_div`` keeps
        # this step's draw per pair_mode, so the two-phase ``g_only`` call
        # (which does not draw during critic warmup) reports the ``d_only``
        # D-update's draw instead of nothing. See ``_ladd_real_div_logs``.
        self._ladd_real_div_telemetry = None

        device = fake_src_grad.device
        zero = torch.zeros((), device=device, dtype=torch.float32)
        disc_for_update = (
            self.r3gan_disc_ddp
            if self.r3gan_disc_ddp is not None
            else self.r3gan_disc
        )
        disc_for_guidance = self.r3gan_disc

        npb = int(getattr(self.model, "num_frame_per_block", 3))
        F_total = int(fake_src_grad.shape[1])
        if F_total < npb:
            return zero, {"train/ladd_n_pairs": 0.0}
        n_chunks = F_total // npb

        # Per-mode pair index lists. Each pair = (real_idx, fake_idx)
        # where the indices are chunk positions in real_src / fake_src.
        # ``chunks_per_pair`` controls whether the disc sees one chunk
        # (gt_vs_fake / adjacent_chunks) or a pair-of-chunks concatenated
        # along F (gt_transition).
        chunks_per_pair = 1
        if pair_mode == "gt_vs_fake":
            all_pairs = [(i, i) for i in range(n_chunks)]
        elif pair_mode == "adjacent_chunks":
            if n_chunks < 2:
                return zero, {"train/ladd_n_pairs": 0.0}
            all_pairs = [(i, i + 1) for i in range(n_chunks - 1)]
        elif pair_mode == "gt_transition":
            # Action-conditioned GT-supervised adjacent-chunk matching.
            # Each "pair" is the chunk-pair (i, i+1); the disc input is
            # concat(chunk_i, chunk_{i+1}) along the F dim, doubling the
            # frames-per-sample from npb to 2*npb. Real side uses GT,
            # fake side uses student. The pair indices store (i, i+1)
            # so the action-token slicing below can use the same lo_r /
            # lo_f layout (and we cover both chunks per pair).
            if n_chunks < 2:
                return zero, {"train/ladd_n_pairs": 0.0}
            all_pairs = [(i, i + 1) for i in range(n_chunks - 1)]
            chunks_per_pair = 2
        else:
            raise ValueError(
                f"_ladd_run_pair_mode: unknown pair_mode={pair_mode!r}."
            )

        # All-pairs (style, not position) RpGAN: compare EVERY real to
        # EVERY fake in the loss (B_real x B_fake relativistic terms)
        # rather than position-matched. The disc forward is unchanged
        # (same B reals + B fakes); only the loss reduction becomes an
        # outer product, so the extra signal is ~free in memory. Enabled
        # per-mode: gt_vs_fake (chunk vs chunk) via
        # ``ladd_gt_vs_fake_all_pairs``, gt_transition (transition-pair vs
        # transition-pair) via ``ladd_gt_transition_all_pairs``. Each
        # logit stays bound to its own chunk/transition's action through
        # the forward, so the outer product only recombines correctly
        # self-conditioned scalars. adjacent_chunks keeps positional
        # pairing. ``_d_loss_fn`` / ``_g_loss_fn`` are used at every
        # RpGAN reduction site below.
        _all_pairs_mode = (
            (
                pair_mode == "gt_vs_fake"
                and bool(getattr(
                    self.model, "ladd_gt_vs_fake_all_pairs", False))
            )
            or (
                pair_mode == "gt_transition"
                and bool(getattr(
                    self.model, "ladd_gt_transition_all_pairs", False))
            )
        )
        _d_loss_fn = rpgan_d_loss_allpairs if _all_pairs_mode else rpgan_d_loss
        _g_loss_fn = rpgan_g_loss_allpairs if _all_pairs_mode else rpgan_g_loss

        # Mismatched all-pairs (N_real != N_fake) + resample-per-update.
        # N_real GT chunks (sampled fresh each D-update from the full ride
        # window) vs the N_fake student chunks. Only meaningful with
        # all-pairs (the loss is a non-square outer product). Handled by a
        # dedicated self-contained branch below (the matched path is left
        # untouched). 0 = off.
        _n_real = 0
        if _all_pairs_mode and pair_mode == "gt_vs_fake":
            _n_real = int(getattr(self.model, "ladd_gt_vs_fake_n_real", 0))
        elif _all_pairs_mode and pair_mode == "gt_transition":
            _n_real = int(getattr(self.model, "ladd_gt_transition_n_real", 0))
        _mismatched = _n_real > 0

        # Optional pair budget. ``ladd_pair_selection`` decides which pairs
        # when capped: "first" = the first N rolled chunks (deterministic,
        # lowest drift); "random" = random subset seeded by the step (same
        # on all ranks, varies each step). all_pairs is ordered by chunk
        # index, so all_pairs[:N] == the first N rolled chunks.
        ladd_pairs_per_step = int(
            getattr(self.model, "ladd_pairs_per_step", 0)
        )
        pair_selection = str(
            getattr(self.model, "ladd_pair_selection", "random")
        ).lower()
        if (
            ladd_pairs_per_step > 0
            and ladd_pairs_per_step < len(all_pairs)
        ):
            if pair_selection == "first":
                pairs = all_pairs[:ladd_pairs_per_step]
            else:
                g = torch.Generator(device="cpu").manual_seed(int(current_step))
                idx = torch.randperm(len(all_pairs), generator=g)[
                    :ladd_pairs_per_step
                ].tolist()
                pairs = [all_pairs[i] for i in sorted(idx)]
        else:
            pairs = all_pairs
        n_pairs = len(pairs)

        # Wide-real (all-pairs gt_vs_fake only): draw the REAL GT chunks
        # from the FULL loaded ride window (seed + rollout + post-window)
        # rather than the position-matched scored slice, for more varied
        # GT in the style comparison. ``real_positions`` (one absolute
        # chunk index per pair) + ``real_src_eff`` drive BOTH the real-
        # latent slice and the real-action slice below; both index the
        # same-coordinate ride windows (ride_latents_window /
        # ride_actions_window share the ride offset s) at the same
        # absolute frame, so latent and action stay co-located. Counts
        # stay == n_pairs (== fake count), so the batched forward and the
        # R1 finite-diff split are unaffected. Non-wide path reproduces
        # the original behaviour exactly (real_src_eff=real_src,
        # real_positions=[i for (i,_) in pairs]).
        _wide_real = (
            _all_pairs_mode
            and pair_mode == "gt_vs_fake"
            and chunks_per_pair == 1
            and bool(getattr(self.model, "ladd_gt_vs_fake_wide_real", False))
        )
        real_src_eff = real_src
        real_positions = [i for (i, _) in pairs]
        if _wide_real:
            _ss = getattr(self.model, "streaming_state", None)
            _wide_win = (
                _ss.get("ride_latents_window")
                if isinstance(_ss, dict) else None
            )
            if _wide_win is not None and int(_wide_win.shape[1]) >= npb:
                real_src_eff = _wide_win.detach().to(real_src.dtype)
                _w_chunks = int(_wide_win.shape[1] // npb)
                # Step-seeded sample (same positions across ranks is fine —
                # each rank holds a different ride, so GT content still
                # differs; the COUNT is fixed = n_pairs so disc-forward
                # shapes match across ranks → DDP-safe). Cycle if the
                # window holds fewer chunks than n_pairs.
                _g = torch.Generator(device="cpu").manual_seed(
                    int(current_step) * 131 + 17
                )
                _perm = torch.randperm(_w_chunks, generator=_g).tolist()
                real_positions = [
                    _perm[k % _w_chunks] for k in range(n_pairs)
                ]
            else:
                _wide_real = False  # window unavailable → fall back

        def _slice(t, i):
            return t[:, i * npb:(i + 1) * npb]

        def _slice_pair(t, i, j):
            # F-concat both indices: [B, 2*npb, C, H, W].
            return torch.cat([_slice(t, i), _slice(t, j)], dim=1)

        # Magnitude-equalize the two members of a gt_transition pair
        # (former [:, :npb], latter [:, npb:]) to their common average
        # MEAN MAGNITUDE (mean of |x| — the energy/brightness proxy, NOT
        # the signed mean which is direction+magnitude and can cancel to
        # ~0). Equalizing a magnitude requires SCALING (not shifting):
        # scale the lower-magnitude chunk UP and the higher one DOWN by
        # the matching factor so both reach the pair's average magnitude,
        # preserving the pair's overall magnitude. Removes the inter-chunk
        # magnitude (brightness) drift across the transition from the
        # disc's view — counters the transition GAN's progressive
        # brightening. Differentiable, so on the grad-on fake side it also
        # zeroes the gen-side gradient pushing the inter-member magnitude.
        # cpp==2: the transition flag (unchanged). cpp==1 (gt_vs_fake): its
        # OWN flag, default "" = off = byte-identical. The m1/m1m2 branch of
        # _mean_equalize_pair reduces over [F,H,W] keeping C, so it
        # generalises to a single chunk cleanly -- unlike mean_equalize, whose
        # former/latter split is NaN at cpp==1. Applied to the disc input AND
        # (via _mean_eq) to cand_disc, so match queries and candidates are
        # normalised identically -- which is the point: it removes the raw
        # brightness cue a grey student otherwise uses to retrieve an equally
        # grey GT chunk and escape being penalised.
        _mag_mode = (
            str(getattr(self.model, "ladd_gt_transition_mag_norm", "")).lower()
            if chunks_per_pair == 2
            else str(getattr(self.model, "ladd_gt_vs_fake_mag_norm", "")).lower()
        )
        _mean_eq = (_mag_mode in ("m1", "m1m2")) or (
            chunks_per_pair == 2
            and bool(getattr(
                self.model, "ladd_gt_transition_mean_equalize", False))
        )

        def _mean_equalize_pair(pair):
            eps = 1e-6
            if _mag_mode in ("m1", "m1m2"):
                # Per-pair, per-channel normalization (reduce over F,H,W;
                # keep C). Each pair uses its OWN stats -> per-video
                # distributions handled correctly. m1: divide each channel
                # by its RMS (unit energy, keeps contrast). m1m2:
                # standardize each channel (unit energy + contrast).
                if _mag_mode == "m1":
                    rms = pair.pow(2).mean(
                        dim=[1, 3, 4], keepdim=True).add(eps).sqrt()
                    return pair / rms
                mu = pair.mean(dim=[1, 3, 4], keepdim=True)
                sd = pair.std(
                    dim=[1, 3, 4], unbiased=False, keepdim=True).add(eps)
                return (pair - mu) / sd
            # mean_equalize: scale the two members (former/latter) to the
            # pair's common mean-magnitude (mean of |x|); preserves the
            # pair's overall magnitude, removes only the inner drift.
            fmr = pair[:, :npb]
            ltr = pair[:, npb:]
            a_f = fmr.abs().mean(dim=[1, 2, 3, 4], keepdim=True)
            a_l = ltr.abs().mean(dim=[1, 2, 3, 4], keepdim=True)
            a = 0.5 * (a_f + a_l)
            return torch.cat(
                [fmr * (a / (a_f + eps)), ltr * (a / (a_l + eps))], dim=1)

        # CARN-former (ladd_gt_transition_carn_former): degrade the FORMER
        # chunk of each GT transition pair through the trained forward
        # noiser, leaving the latter clean -> the disc sees REAL as a
        # "CARN-degraded -> clean" (self-correcting) transition. Applied
        # under no_grad on the detached real side (does NOT train the FN
        # here; the FN trains via the rollout2 critic loss). Identity until
        # the FN has trained (zero-init), so the effect ramps in.
        # Robust knob read: prefer self.model, fall back to self.config
        # (the override lives on the config object; self.model may not
        # surface every arg).
        _carn_knob = bool(
            getattr(self.model, "ladd_gt_transition_carn_former", None)
            or getattr(self.config, "ladd_gt_transition_carn_former", False)
        )
        # carn_former applies the learned ForwardNoiser (FN) to the GT
        # former chunk: the FN has learned the rollout1->rollout2 "+1
        # cartoon shift" (sliced-Wasserstein teacher-feature match), so
        # FN(GT_former) = "GT with one AR step of texture drift". The real
        # anchor becomes [cartoon-shifted GT former -> clean GT latter] = a
        # self-correcting (de-cartoon) transition the student must match.
        # Latent-space (no wavelet); requires the FN to be present.
        _carn_former_on = (
            pair_mode == "gt_transition"
            and _carn_knob
            and getattr(self.model, "forward_noiser", None) is not None
        )
        if _carn_former_on and getattr(self, "_carn_former_dbg", 0) < 2:
            self._carn_former_dbg = getattr(self, "_carn_former_dbg", 0) + 1
            import sys as _sys
            print(
                "[CARN-FORMER] ON: gt_transition GT former chunk pushed +1 "
                "cartoon step by the learned ForwardNoiser.",
                file=_sys.stderr, flush=True,
            )

        def _carn_former(x):
            n_steps = int(getattr(self.model, "ladd_gt_transition_carn_steps", 1))
            # FIX B (random real-former CARN level): draw the level per PAIR
            # uniformly in [0, max_level] (0 = clean former). The disc then
            # sees real formers at every degradation level (incl. clean), so
            # it can't pull the student toward a single fixed CARN level —
            # it must key on the transition (clean latter | any former).
            if bool(getattr(
                self.model, "ladd_gt_transition_carn_random_level", False)):
                _maxlvl = int(getattr(
                    self.model, "ladd_gt_transition_carn_max_level", n_steps))
                n_steps = int(torch.randint(0, max(1, _maxlvl) + 1, (1,)).item())
                if n_steps <= 0:
                    return x.detach()  # level 0 -> clean GT former, no CARN
            # Step-unconditioned FN: every apply passes carn_step=0 (the FN
            # learned one generic "+1 shift"). Otherwise pass the iteration
            # index as the CARN level (legacy).
            _uncond = bool(getattr(
                self.model, "forward_noiser_step_unconditioned", False))
            _recurse = bool(getattr(self.model, "carn_recurse", True))
            x0 = x  # original former (pre-carn) for moment restoration
            with torch.no_grad():
                if not _recurse:
                    # SINGLE-CALL mode (consistent with the aux CARN): one
                    # conditioned call from the clean former, carn_step =
                    # the target level (= n_steps, default 1). No recursion.
                    cs = torch.full(
                        (x.shape[0],), int(max(1, n_steps)),
                        dtype=torch.long, device=x.device,
                    )
                    x = self.model.forward_noiser(x0, cs, residual=True)
                else:
                    for _s in range(max(1, n_steps)):
                        cs = torch.full(
                            (x.shape[0],), 0 if _uncond else _s,
                            dtype=torch.long, device=x.device,
                        )
                        x = self.model.forward_noiser(x, cs, residual=True)
                # MOMENT-PRESERVING carn (texture only, NO stats — see the
                # "CARN = texture, not stats" directive). The FN trains on
                # un-normalized rollout1->rollout2, where rollout2 runs hot
                # (one more drift step), so its output drifts BRIGHTER. If
                # that reaches the GT former, ``_mean_equalize_pair`` below
                # rescales BOTH members to their common magnitude a=0.5*(a_f
                # +a_l) -> a hot former drags the clean latter UP and lifts
                # the whole REAL pair above the fake pair -> the disc learns
                # "brighter=real" -> student brightens -> rollout2 brighter
                # -> FN brighter -> WHITE COLLAPSE. Restore the original
                # former's (a) per-channel DC mean and (b) per-sample mean-
                # magnitude (the exact stat _mean_equalize_pair keys on), so
                # carn changes only HF/texture STRUCTURE and the equalize
                # sees an unchanged former. Scalar rescale preserves the
                # texture pattern.
                eps = 1e-6
                mc_in = x0.mean(dim=[1, 3, 4], keepdim=True)
                mc_out = x.mean(dim=[1, 3, 4], keepdim=True)
                x = x - mc_out + mc_in
                a_in = x0.abs().mean(dim=[1, 2, 3, 4], keepdim=True)
                a_out = x.abs().mean(dim=[1, 2, 3, 4], keepdim=True)
                x = x * (a_in / (a_out + eps))
            return x.detach()

        def _slice_pair_carn(t, i, j):
            former = _carn_former(_slice(t, i))
            return torch.cat([former, _slice(t, j)], dim=1)

        if chunks_per_pair == 2:
            # gt_transition: real and fake BOTH cover the chunk-pair
            # (i, i+1). Real uses GT (real_src=gt_detached), fake uses
            # student (fake_src_*). Same i,j indices on both sides — the
            # disc compares "GT's chunk pair" vs "student's chunk pair".
            _real_pair_fn = _slice_pair_carn if _carn_former_on else _slice_pair
            real_chunks_det = torch.cat(
                [_real_pair_fn(real_src, i, j) for (i, j) in pairs], dim=0,
            )
            fake_chunks_det = torch.cat(
                [_slice_pair(fake_src_detached, i, j) for (i, j) in pairs],
                dim=0,
            )
            _gen_detach_former = bool(getattr(
                self.model, "ladd_gt_transition_gen_detach_former", False))

            def _slice_pair_gen(t, i, j):
                # FIX A: on the gen-side fake pair, optionally DETACH the
                # former so the GAN gradient flows only to the LATTER (the
                # transition target) — the student learns "given my drifted
                # former, make the next chunk clean" without being pushed to
                # degrade its own former toward the real-pair's CARN level.
                former = _slice(t, i)
                if _gen_detach_former:
                    former = former.detach()
                return torch.cat([former, _slice(t, j)], dim=1)

            fake_chunks_grad_tensor = torch.cat(
                [_slice_pair_gen(fake_src_grad, i, j) for (i, j) in pairs],
                dim=0,
            )
            if _mean_eq and _mag_mode in ("m1", "m1m2"):
                # Per-channel magnitude mode (opt-in): unchanged, per-tensor.
                real_chunks_det = _mean_equalize_pair(real_chunks_det)
                fake_chunks_det = _mean_equalize_pair(fake_chunks_det)
                fake_chunks_grad_tensor = _mean_equalize_pair(
                    fake_chunks_grad_tensor)
            elif _mean_eq:
                # CROSS-equalize the OVERALL level across the real and fake
                # pools so the disc gets NO absolute-brightness cue (kills
                # the white-collapse feedback). a* is the common per-row
                # target level, built from DETACHED magnitudes so the grad-on
                # fake gets a constant target. Two variants
                # (ladd_gt_transition_xeq_preserve_delta):
                #   * flatten (False, default): scale each of the 4 members
                #     INDEPENDENTLY to a* -> within-pair former<->latter
                #     brightness delta removed too; disc fully brightness-
                #     blind.
                #   * preserve_delta (True): scale each PAIR by ONE shared
                #     factor a*/a_pair -> real/fake absolute level equalized
                #     but the within-pair brightness TRANSITION ratio is
                #     preserved, so the disc still sees (and pushes the
                #     student to match GT's) transition brightness. No
                #     collapse because absolute level is still pinned.
                # Either way the rescale is a scalar per member/pair, so
                # texture STRUCTURE is preserved. real-row-i and fake-row-i
                # are the same chunk-pair position (GT vs student).
                _eps = 1e-6
                _preserve_delta = bool(
                    getattr(self.model,
                            "ladd_gt_transition_xeq_preserve_delta", None)
                    or getattr(self.config,
                               "ladd_gt_transition_xeq_preserve_delta", False)
                )
                _per_channel = bool(
                    getattr(self.model,
                            "ladd_gt_transition_xeq_per_channel", None)
                    or getattr(self.config,
                               "ladd_gt_transition_xeq_per_channel", False)
                )
                # Reduce over [F,H,W] keeping C (per-channel) or over
                # [F,C,H,W] (one scalar). keepdim=True so a_star / the
                # factors broadcast either way.
                _red = [1, 3, 4] if _per_channel else [1, 2, 3, 4]

                def _mags(t):
                    return (
                        t[:, :npb].abs().mean(dim=_red, keepdim=True),
                        t[:, npb:].abs().mean(dim=_red, keepdim=True),
                    )

                _da_rf, _da_rl = _mags(real_chunks_det.detach())
                _da_ff, _da_fl = _mags(fake_chunks_det.detach())
                if real_chunks_det.shape[0] == fake_chunks_det.shape[0]:
                    a_star = 0.25 * (_da_rf + _da_rl + _da_ff + _da_fl)  # per row
                else:
                    # Defensive (row-count mismatch): one shared target
                    # (per-channel if enabled, else scalar) so real and fake
                    # still share a level. mean over the stacked rows, keep C.
                    a_star = torch.cat(
                        [_da_rf, _da_rl, _da_ff, _da_fl], dim=0,
                    ).mean(dim=0, keepdim=True)

                if _preserve_delta:
                    def _xeq(t):
                        af, al = _mags(t)  # in-graph for the grad tensor
                        a_pair = 0.5 * (af + al)            # this pair's level
                        factor = a_star / (a_pair + _eps)   # ONE factor, both members
                        return t * factor                   # ratio (transition) kept
                else:
                    def _xeq(t):
                        af, al = _mags(t)  # in-graph for the grad tensor
                        return torch.cat(
                            [t[:, :npb] * (a_star / (af + _eps)),
                             t[:, npb:] * (a_star / (al + _eps))], dim=1,
                        )

                real_chunks_det = _xeq(real_chunks_det)
                fake_chunks_det = _xeq(fake_chunks_det)
                fake_chunks_grad_tensor = _xeq(fake_chunks_grad_tensor)

                # STD equalization (same cross-eq mechanism, on the 2nd
                # moment). Applied AFTER the mean-eq, CENTERED: scale each
                # member's deviations-from-mean to the common std s* and
                # re-add the mean, so it sets spread/contrast WITHOUT
                # disturbing the level. Honours the same per-channel /
                # preserve-delta options. With both mean+std eq on, the disc
                # input is per-channel first+second-moment-free -> the GAN
                # keys on pure texture, never brightness OR contrast.
                _std_eq = bool(
                    getattr(self.model,
                            "ladd_gt_transition_std_equalize", None)
                    or getattr(self.config,
                               "ladd_gt_transition_std_equalize", False)
                )
                if _std_eq:
                    def _stds(t):
                        return (
                            t[:, :npb].std(dim=_red, unbiased=False,
                                           keepdim=True),
                            t[:, npb:].std(dim=_red, unbiased=False,
                                           keepdim=True),
                        )

                    def _means(t):
                        return (
                            t[:, :npb].mean(dim=_red, keepdim=True),
                            t[:, npb:].mean(dim=_red, keepdim=True),
                        )

                    _ds_rf, _ds_rl = _stds(real_chunks_det.detach())
                    _ds_ff, _ds_fl = _stds(fake_chunks_det.detach())
                    if real_chunks_det.shape[0] == fake_chunks_det.shape[0]:
                        s_star = 0.25 * (_ds_rf + _ds_rl + _ds_ff + _ds_fl)
                    else:
                        s_star = torch.cat(
                            [_ds_rf, _ds_rl, _ds_ff, _ds_fl], dim=0,
                        ).mean(dim=0, keepdim=True)

                    if _preserve_delta:
                        def _xeq_std(t):
                            sf, sl = _stds(t)
                            mf, ml = _means(t)
                            s_pair = 0.5 * (sf + sl)
                            fac = s_star / (s_pair + _eps)  # ONE factor, both
                            return torch.cat(
                                [(t[:, :npb] - mf) * fac + mf,
                                 (t[:, npb:] - ml) * fac + ml], dim=1,
                            )
                    else:
                        def _xeq_std(t):
                            sf, sl = _stds(t)
                            mf, ml = _means(t)
                            return torch.cat(
                                [(t[:, :npb] - mf) * (s_star / (sf + _eps)) + mf,
                                 (t[:, npb:] - ml) * (s_star / (sl + _eps)) + ml],
                                dim=1,
                            )

                    real_chunks_det = _xeq_std(real_chunks_det)
                    fake_chunks_det = _xeq_std(fake_chunks_det)
                    fake_chunks_grad_tensor = _xeq_std(fake_chunks_grad_tensor)
        else:
            real_chunks_det = torch.cat(
                [_slice(real_src_eff, real_positions[k])
                 for k in range(n_pairs)], dim=0,
            )
            fake_chunks_det = torch.cat(
                [_slice(fake_src_detached, j) for (_, j) in pairs], dim=0,
            )
            fake_chunks_grad_tensor = torch.cat(
                [_slice(fake_src_grad, j) for (_, j) in pairs], dim=0,
            )

        # ----- Disc timestep -----
        # WAN with action-aware forward expects per-frame timestep
        # ``[B, F]`` so the time_projection output matches the per-frame
        # ``_action_modulation``'s ``[B*F, ...]`` reshape. Without F in
        # the timestep shape, time_projection outputs ``[B, hidden]``
        # while action_modulation is ``[B*F, hidden]`` and the
        # ``am_flat != e0`` shape check raises. Match the main DMD
        # path's convention.
        flash_on = bool(getattr(self.model, "flash_dmd_enabled", False))
        flash_t = int(getattr(self.model, "flash_dmd_gan_t", 60))
        # When the wavelet-HF stage is enabled, force the disc to
        # operate on CLEAN latents (disc_t_int=0). The wavelet step
        # decomposes its input into LL+LH+HL+HH sub-bands; if we noise
        # the input first, the diffusion noise leaks broadband into
        # the HF sub-bands and the disc ends up partly discriminating
        # noise rather than student-vs-GT HF structure (the WGSR
        # frequency-band restriction we're trying to enforce becomes
        # noise-band restriction). Match the original WGSR setup by
        # keeping the wavelet input clean.
        wavelet_on = bool(
            getattr(self.r3gan_disc, "wavelet_hf_enabled", False)
        )
        # Force clean disc input when requested (e.g. wavelet off but the
        # disc is meant to learn brightness/contrast, which t=flash_t noise
        # would swamp). Does NOT touch flash_dmd_gan_t (shared with main DMD).
        _disc_force_clean = bool(
            getattr(self.model, "ladd_disc_force_clean", None)
            or getattr(self.config, "ladd_disc_force_clean", False)
        )
        if wavelet_on or _disc_force_clean:
            disc_t_int = 0
        else:
            disc_t_int = flash_t if flash_on else 0
        # ---- ladd_real_match_fake_t (Phase B fix 3, default OFF) ----
        # BEFORE (audited 2026-08-22): real and fake are ALREADY noised
        # symmetrically — both halves pass through the same
        # ``_m_noise``/``_add_disc_noise`` at ``disc_t_int`` and the disc
        # timestep arg is the same for every row. The real discrepancy is
        # BETWEEN ARMS: wavelet-ON (or ladd_disc_force_clean) forces
        # disc_t_int=0, so the disc compares raw GT x0 vs the student's
        # t=flash_t x0 prediction with a t=0 conditioning; wavelet-OFF
        # arms ran disc_t_int=flash_t (both halves re-noised at t=60,
        # disc conditioned on t=60). The two knobs were never varied
        # independently (GAN_FORENSICS.md B.1).
        # AFTER (flag ON): disc_t_int is pinned to flash_t regardless of
        # the wavelet/force-clean shortcut, so BOTH halves are re-noised
        # via the same scheduler.add_noise pathway at the fake slab's
        # generation timestep and the disc conditions on that same t for
        # both halves — the diffusion-GAN symmetric construction, and it
        # decouples "wavelet on/off" from "disc input noise level".
        # No-op when flash_dmd is off (no defined fake t) or when the
        # flag is off (byte-identical default).
        if flash_on and bool(getattr(
                self.config, "ladd_real_match_fake_t",
                getattr(self.model, "ladd_real_match_fake_t", False))):
            disc_t_int = flash_t
        # Diffusion-GAN corruption sampler. The fixed ``flash_t`` path leaves
        # D solving one narrow low-noise classification problem; opt-in
        # sampling exposes the transition critic to the diffusion interval.
        # The sampled scalar is still shared by real and fake, including the
        # exact scheduler.add_noise path below.
        # Wavelet-HF has strict precedence: its sub-bands must see clean
        # latents. Sampling after the clean-wavelet branch used to silently
        # undo that invariant and turn HF discrimination into broadband-noise
        # discrimination.
        if (
            not wavelet_on
            and bool(getattr(
                self.config, "ladd_disc_sample_t",
                getattr(self.model, "ladd_disc_sample_t", False),
            ))
        ):
            _dt_lo = int(getattr(self.config, "ladd_disc_t_min", 20))
            _dt_hi = int(getattr(self.config, "ladd_disc_t_max", 980))
            _dt_shift = float(getattr(
                self.config, "ladd_disc_timestep_shift", 5.0))
            if not (0 <= _dt_lo <= _dt_hi <= 1000):
                raise ValueError(
                    "ladd_disc_t_min/max must satisfy 0 <= min <= max <= "
                    f"1000; got [{_dt_lo}, {_dt_hi}]."
                )
            _dt = torch.randint(
                _dt_lo, _dt_hi + 1, (1,), device=device,
            ).float()
            if _dt_shift != 1.0:
                _dt_norm = _dt / 1000.0
                _dt = 1000.0 * _dt_shift * _dt_norm / (
                    1.0 + (_dt_shift - 1.0) * _dt_norm
                )
            disc_t_int = int(
                _dt.round().clamp(_dt_lo, _dt_hi).item()
            )
        bsz_eff = real_chunks_det.shape[0]
        # Per-frame timestep matches the disc input's F dim — which is
        # ``npb`` for single-chunk modes and ``2 * npb`` for gt_transition.
        t_frames = chunks_per_pair * npb
        t_disc = torch.full(
            (bsz_eff, t_frames), disc_t_int, dtype=torch.long, device=device,
        )

        def _add_disc_noise(x):
            if disc_t_int <= 0:
                return x
            scheduler = getattr(self.model, "scheduler", None)
            if scheduler is None or not hasattr(scheduler, "add_noise"):
                raise RuntimeError(
                    "_add_disc_noise: disc_t_int > 0 but the model has "
                    "no scheduler with ``add_noise`` available. Either "
                    "set ``self.model.scheduler`` to a usable scheduler "
                    "or set disc_t_int=0 (e.g. ``flash_dmd_enabled="
                    "False``) so the disc operates on clean inputs. "
                    "Silent fallback removed — this used to return ``x`` "
                    "unchanged and mask a misconfigured model."
                )
            eps = torch.randn_like(x)
            x_flat = x.flatten(0, 1)
            eps_flat = eps.flatten(0, 1)
            # t_disc is now [B, F]; flatten matches x_flat's [B*F] axis.
            t_per_frame = t_disc.flatten(0, 1)
            noisy_flat = scheduler.add_noise(x_flat, eps_flat, t_per_frame)
            return noisy_flat.unflatten(0, x.shape[:2])

        # ``_match`` (the matched gt_transition branch) re-noises + re-
        # augments inside its own D-loop (_m_noise / per-update
        # latent_diff_augment) and NEVER reads these whole-pair pre-noised
        # tensors — only the positional / mismatched branches (gt_vs_fake,
        # adjacent, all_pairs) consume real/fake_chunks_det_noisy. Skip the
        # noise + diff-aug-clone work when matching is active. ``_match`` is
        # defined further below and bound to this (``_match = _match_active``)
        # so the two cannot diverge.
        # ``ladd_gt_vs_fake_match`` (default False) enables the SAME
        # nearest-match real retrieval for the SINGLE-CHUNK gt_vs_fake mode:
        # each fake chunk is scored against its K nearest-by-L1 GT CHUNKS
        # drawn from the ride pool, instead of the positional GT chunk that
        # produced the dead disc (d_real ~= d_fake, d_loss == log2). The
        # TRANSITION structure stays exclusive to gt_transition: gt_vs_fake
        # keeps ``chunks_per_pair == 1`` here and in every candidate slice
        # below (the matched path is parameterised by ``chunks_per_pair``,
        # never by the mode name). Read from the model with a config
        # fallback (same pattern as the other late-added LADD knobs).
        _match_active = (
            (
                pair_mode == "gt_transition"
                and chunks_per_pair == 2
                and bool(getattr(self.model, "ladd_gt_transition_match", False))
            )
            or (
                pair_mode == "gt_vs_fake"
                and chunks_per_pair == 1
                and bool(
                    getattr(self.model, "ladd_gt_vs_fake_match", None)
                    or getattr(self.config, "ladd_gt_vs_fake_match", False)
                )
            )
        )
        # DiffAugment — same per-sample randomness on real and fake.
        # Different seeds per mode so the augmentations decorrelate.
        # (Defined unconditionally: the matched D-loop reads diff_aug_policy.)
        diff_aug_policy = str(
            getattr(self.model, "ladd_diff_aug_policy", "")
        )
        seed_offset = 0 if pair_mode == "gt_vs_fake" else 7919
        real_chunks_det_noisy = fake_chunks_det_noisy = None
        if not _match_active:
            real_chunks_det_noisy = _add_disc_noise(real_chunks_det)
            fake_chunks_det_noisy = _add_disc_noise(fake_chunks_det)
            if diff_aug_policy:
                real_chunks_det_noisy, fake_chunks_det_noisy = (
                    latent_diff_augment(
                        real_chunks_det_noisy,
                        fake_chunks_det_noisy,
                        policy=diff_aug_policy,
                        seed=int(current_step) * 31 + seed_offset,
                    )
                )

        # ----- Prompt embedding (broadcast across pairs) -----
        prompt_embeds = None
        pooled_prompt = None
        s_state = getattr(self.model, "streaming_state", None)
        if isinstance(s_state, dict):
            prompt_embeds = s_state.get("prompt_embeds")
        if prompt_embeds is None:
            raise RuntimeError(
                "_ladd_run_pair_mode: prompt_embeds not in "
                "streaming_state. setup_sequence must have run before "
                "the gen step."
            )
        if prompt_embeds.shape[0] != bsz_eff:
            reps = bsz_eff // prompt_embeds.shape[0]
            prompt_embeds_eff = prompt_embeds.repeat_interleave(reps, dim=0)
        else:
            prompt_embeds_eff = prompt_embeds
        if disc_for_guidance.cmap_dim > 0:
            pooled_prompt = prompt_embeds_eff.float().mean(dim=1)

        # ----- Action tokens + modulation (action-aware WAN configs) -----
        # When the WAN model has action_tokens_per_frame > 0, its
        # forward refuses to run without action_tokens, AND the
        # transformer blocks use a per-frame ``_action_modulation``
        # whose frame count MUST match the input's frame count (else
        # "size of tensor a (X) must match (Y)" in the per-block FiLM
        # multiply). Build BOTH streams from per-chunk slices of
        # streaming_state[ride_actions_window].
        real_action_tokens = None
        fake_action_tokens = None
        real_action_modulation = None
        fake_action_modulation = None
        a_per_f = 0
        # Look up the WAN model under real_score to read a_per_f.
        _rs = self.model.real_score
        for _cand in [_rs] + list(_rs.modules()):
            if hasattr(_cand, "action_tokens_per_frame"):
                a_per_f = int(getattr(_cand, "action_tokens_per_frame", 0))
                if a_per_f > 0:
                    break
        ride_actions = (
            s_state.get("ride_actions_window")
            if isinstance(s_state, dict) else None
        )
        atp = getattr(self.model, "action_token_projection", None)
        ap = getattr(self.model, "action_projection", None)
        if a_per_f > 0 and ride_actions is not None and atp is not None:
            cf_state = int(s_state.get("cf", 0))
            # Build per-pair action slices for real and fake sides.
            real_acts = []
            fake_acts = []
            for k, (i_real, j_fake) in enumerate(pairs):
                # ride_actions: [B, cf+rollout, A]. Chunk i covers frames
                # [cf + i*npb, cf + (i+1)*npb) within the rollout. For
                # gt_vs_fake, real and fake use the same chunk index
                # (i_real == j_fake). For adjacent_chunks they differ
                # by one (j_fake = i_real + 1).
                # Wide-real: the real chunk k was sliced from absolute
                # window position real_positions[k] (NOT cf_state-relative);
                # its action is co-located at the same absolute frame in
                # ride_actions_window (latents/actions share the ride
                # offset s), so use real_positions[k]*npb. Only fires for
                # all-pairs gt_vs_fake (chunks_per_pair==1).
                if _wide_real:
                    lo_r = real_positions[k] * npb
                else:
                    lo_r = cf_state + i_real * npb
                lo_f = cf_state + j_fake * npb
                if chunks_per_pair == 2:
                    # gt_transition: both sides cover the (i, i+1)
                    # chunk-pair. Concat the two consecutive action
                    # slices so the WAN teacher's per-frame action
                    # modulation has a slice for every one of the
                    # 2*npb input frames. Real and fake actions are
                    # identical (same time positions, same actions —
                    # only the latents differ).
                    real_acts.append(torch.cat([
                        ride_actions[:, lo_r:lo_r + npb],
                        ride_actions[:, lo_f:lo_f + npb],
                    ], dim=1))
                    fake_acts.append(torch.cat([
                        ride_actions[:, lo_r:lo_r + npb],
                        ride_actions[:, lo_f:lo_f + npb],
                    ], dim=1))
                else:
                    real_acts.append(ride_actions[:, lo_r:lo_r + npb])
                    fake_acts.append(ride_actions[:, lo_f:lo_f + npb])
            real_acts_t = torch.cat(real_acts, dim=0).to(
                device=device, dtype=prompt_embeds_eff.dtype,
            )
            fake_acts_t = torch.cat(fake_acts, dim=0).to(
                device=device, dtype=prompt_embeds_eff.dtype,
            )
            # No_grad: action projections are conditioning-only; we
            # never backward through them on the disc path, so the
            # autograd graph they'd build is dead weight (~1-2 GB).
            with torch.no_grad():
                real_action_tokens = atp(real_acts_t).detach()
                fake_action_tokens = atp(fake_acts_t).detach()
                if ap is not None:
                    real_action_modulation = ap(
                        real_acts_t, num_frames=real_acts_t.shape[1],
                    ).detach()
                    fake_action_modulation = ap(
                        fake_acts_t, num_frames=fake_acts_t.shape[1],
                    ).detach()

        # v28G_2: action-blind variant of gt_transition. Zeroes out the
        # action tokens and modulation BEFORE they hit the disc forward,
        # while preserving tensor shapes so the WAN teacher (which
        # requires action_tokens when ``a_per_f > 0``) still runs. The
        # disc then learns "GT chunk-pair transitions look like this
        # distribution" *independent of which action drove the
        # transition* — useful when the action signal is noisy or when
        # you want to ablate the action-conditioning contribution.
        # Only applies to gt_transition; gt_vs_fake / adjacent_chunks
        # keep their full action conditioning.
        if (
            pair_mode == "gt_transition"
            and bool(getattr(
                self.model, "ladd_gt_transition_action_blind", False,
            ))
        ):
            if real_action_tokens is not None:
                real_action_tokens = torch.zeros_like(real_action_tokens)
            if fake_action_tokens is not None:
                fake_action_tokens = torch.zeros_like(fake_action_tokens)
            if real_action_modulation is not None:
                real_action_modulation = torch.zeros_like(
                    real_action_modulation
                )
            if fake_action_modulation is not None:
                fake_action_modulation = torch.zeros_like(
                    fake_action_modulation
                )

        self._mem_step_snapshot(f"ladd_{pair_mode}_a_cond_built")

        # ----- D-update -----
        # ``phase`` allows the caller to split the D-update from the
        # gen-side forward across multiple pair modes (two-phase
        # orchestration). With "d_only" we run only the D-step + step
        # and skip the gen-side; with "g_only" we run only the
        # gen-side forward; with "both" (default) we do D then G in a
        # single call (correct for single-mode runs).
        disc_skipped = (
            current_step < int(getattr(self, "gan_disc_start_step", 0))
        )
        skip_d = (phase == "g_only")
        skip_g = (phase == "d_only")
        n_disc_updates = (
            0 if (disc_skipped or skip_d) else int(self.gan_updates_per_step)
        )
        last_d_loss = 0.0
        last_d_real = 0.0
        last_d_fake = 0.0
        last_r1 = 0.0
        # Stat-head diagnostic accumulators — non-zero only when
        # ``stat_head_enabled``. Separate D-side stat loss + per-sample
        # real / fake logit means so the user can see the stat side's
        # behaviour independently of the per-token visual side. 0 when
        # stat head is off OR ``ladd_stat_head_loss_weight`` is 0.
        last_d_loss_stat = 0.0
        last_d_real_stat = 0.0
        last_d_fake_stat = 0.0
        # Diagnostic R1 stats (independent of γ + lazy schedule):
        #   * ``last_r1_grad_sq``: raw ‖∇_x Σ_i D_i‖² estimate from the
        #     finite-difference or autograd path. This is the quantity
        #     γ multiplies — log it separately so the gradient norm
        #     itself is visible regardless of how small γ is. NaN on
        #     iters where R1 didn't fire (lazy schedule skip or disc
        #     warmup) so wandb plots gaps rather than misleading 0s.
        #   * ``last_r1_fired``: 1.0 if R1 was actually computed this
        #     iter, 0.0 otherwise. Lets the user filter the trace to
        #     only the iters where R1 has a real value.
        last_r1_grad_sq = float("nan")
        last_r1_fired = 0.0
        # A16-D1: R1 applications made by THIS (step, pair_mode) D-update
        # block. NaN when ``ladd_r1_unified_cadence`` is off (the legacy
        # per-iteration latch/modulo has no block-level answer), so the trace
        # plots a gap instead of a misleading 0.
        last_r1_block_fires = float("nan")

        # ===== Content-MATCHED wide GT (gt_transition; all_pairs=false) =====
        # For each fake transition, score it ONLY against its K GT
        # transitions from the FULL loaded ride that are closest by MAE on
        # the MEAN-EQUALIZED rep (texture, not brightness — so a drifted
        # fake can't cherry-pick a drifted GT). Block-diagonal RpGAN (each
        # fake vs its own K matched GT), NOT all_pairs — gives multiple
        # RELEVANT real views per fake without the content-confounded noise
        # of all_pairs. Position is never used (the rollout drifts in time).
        # Efficient: each UNIQUE matched GT is forwarded ONCE; the per-fake-K
        # grouping is done at the LOGIT level (index_select + repeat), so
        # neither fakes nor reals are re-forwarded per K. Self-contained
        # (own D-loop + gen-side + R1). Assumes per-rank B==1 (single ride),
        # like the mismatched branch's prompt tiling. Mutually exclusive
        # with all_pairs / mismatched n_real.
        _match = _match_active  # bound to the precompute-guard condition above
        # Mutually exclusive with the all_pairs matrix and the mismatched
        # n_real sampler — the match branch is physically first and returns,
        # so a co-set config would SILENTLY ignore those. Fail loud instead.
        if _match and (_all_pairs_mode or _mismatched):
            _mflag = ("ladd_gt_vs_fake_match" if pair_mode == "gt_vs_fake"
                      else "ladd_gt_transition_match")
            _pfx = ("ladd_gt_vs_fake" if pair_mode == "gt_vs_fake"
                    else "ladd_gt_transition")
            raise RuntimeError(
                f"{_mflag} is mutually exclusive with "
                f"{_pfx}_all_pairs / {_pfx}_n_real>0; "
                "set all_pairs=false and n_real=0 for the matched mode. "
                "(The matched branch returns before the all-pairs / "
                "mismatched code, so a co-set config would SILENTLY ignore "
                "them.)"
            )
        _match_lat = None
        if _match:
            _ss_m = getattr(self.model, "streaming_state", None)
            _match_lat = (
                _ss_m.get("gt_match_latents")
                if isinstance(_ss_m, dict) else None
            )
        # Minimum usable pool = ONE candidate = ``chunks_per_pair``
        # consecutive chunks: 2*npb frames for a gt_transition pair, npb
        # frames for a gt_vs_fake single chunk. (Was hard-coded 2*npb.)
        _match_min_frames = chunks_per_pair * npb
        if (
            _match
            and _match_lat is not None
            and int(_match_lat.shape[1]) >= _match_min_frames
        ):
            K = max(1, int(getattr(self.model, "ladd_gt_transition_match_k", 3)))
            _need_pp = pooled_prompt is not None
            _r1_gamma = float(getattr(self.model, "ladd_r1_gamma", 1.0))
            _r1_every_n = max(1, int(
                getattr(self.model, "ladd_r1_every_n_steps", 1)))
            _r1_sigma = float(getattr(self.model, "ladd_r1_sigma", 0.01))
            _K_stat = int(getattr(self.r3gan_disc, "stat_logit_count", 0))
            _W_stat = float(getattr(
                self.model, "ladd_stat_head_loss_weight", 1.0))
            # ``ladd_gt_transition_action_blind`` zeroes the conditioning.
            # The FAKE-side zeroing above is gated on
            # ``pair_mode == "gt_transition"``, so gating the matched REAL
            # side the same way keeps the two sides symmetric — otherwise a
            # gt_vs_fake matched run with that (transition-only) flag set
            # would feed action-BLIND reals against action-FUL fakes, i.e.
            # hand the disc a free giveaway feature. No-op with the flag off
            # (default), and unchanged for gt_transition.
            _action_blind = (
                pair_mode == "gt_transition"
                and bool(getattr(
                    self.model, "ladd_gt_transition_action_blind", False))
            )
            B = int(prompt_embeds.shape[0])
            pool_lat = _match_lat.detach()                       # [B, RL, C,H,W]
            pool_act_full = (
                _ss_m.get("gt_match_actions")
                if isinstance(_ss_m, dict) else None
            )
            pool_chunks = int(pool_lat.shape[1] // npb)
            # A candidate starts at chunk ``u`` and spans chunks
            # [u, u + chunks_per_pair), so the last legal start is
            # ``pool_chunks - chunks_per_pair`` and the count is
            # ``pool_chunks - chunks_per_pair + 1``:
            #   chunks_per_pair == 2 (gt_transition): pool_chunks - 1  (u, u+1)
            #   chunks_per_pair == 1 (gt_vs_fake):    pool_chunks      (u)
            # The guard above proves pool_chunks >= chunks_per_pair, so the
            # max(1, ...) clamp never actually fires (kept as belt-and-braces
            # — it must NEVER manufacture an out-of-range u).
            n_cand = max(1, pool_chunks - chunks_per_pair + 1)
            Kk = min(K, n_cand)
            # Hard cap on distinct reals forwarded per D-update (memory): the
            # combined disc forward is 2*n_uniq + n_fake rows, so this bounds
            # it independent of the (possibly whole-ride) match pool. 0 = no
            # cap. Default keeps it near 5b's n_real footprint.
            _cap = int(getattr(self.model, "ladd_gt_transition_match_max_real", 12))
            _fdt = fake_chunks_det.dtype

            def _pool_pair_lat(b, u):
                if chunks_per_pair == 1:
                    # gt_vs_fake: a candidate is ONE chunk -> [1, npb, ...].
                    # u in [0, pool_chunks) => the slice end (u+1)*npb is
                    # <= pool_chunks*npb <= pool_lat.shape[1].
                    return pool_lat[b:b + 1, u * npb:(u + 1) * npb]
                return torch.cat([
                    pool_lat[b:b + 1, u * npb:(u + 1) * npb],
                    pool_lat[b:b + 1, (u + 1) * npb:(u + 2) * npb],
                ], dim=1)                                        # [1, 2*npb, ...]

            # Lazy cand_disc (memory, flag-gated default OFF): rebuild a
            # single equalized candidate pair on demand instead of holding
            # the whole-window cand_disc list pinned across the D-loop +
            # gen-side (+ the deferred backward when ladd_defer_disc_update).
            # _mean_equalize_pair reduces over dim=[1..4] (per-pair), so
            # _cand_pair(b,u) is byte-identical to cand_disc[b][u]. When lazy,
            # cand_disc is NEVER materialized: the top-M matcher streams it in
            # chunks (below) and _match_select rebuilds only the selected
            # pairs — bounding both the persistent hold and the matcher's
            # [n_cand, D] float transient.
            _lazy_cand = bool(getattr(
                self.config, "ladd_lazy_cand_disc",
                getattr(self.model, "ladd_lazy_cand_disc", False)))

            def _cand_pair(b, u):
                cl = _pool_pair_lat(b, u)
                if _mean_eq:
                    cl = _mean_equalize_pair(cl)
                return cl.to(_fdt)[0]

            # Candidate transitions per batch elem, mean-equalized (the disc
            # input AND the match key — match on the same rep the disc sees).
            # When lazy, skip materializing the whole-window cand_disc — the
            # top-M matcher streams it below and _match_select rebuilds the
            # selected pairs on demand.
            if not _lazy_cand:
                cand_disc = []      # over b: [n_cand, 2*npb, C,H,W] (equalized)
                for b in range(B):
                    cl = torch.cat(
                        [_pool_pair_lat(b, u) for u in range(n_cand)], dim=0)
                    if _mean_eq:
                        cl = _mean_equalize_pair(cl)
                    cand_disc.append(cl.to(_fdt))
            else:
                cand_disc = None

            # Fake queries: fake_chunks_det is [n_pairs*B, 2*npb, ...]
            # pair-major batch-minor, already mean-equalized. View [n_pairs,B].
            fq = fake_chunks_det.detach().view(
                n_pairs, B, *fake_chunks_det.shape[1:])

            # Per-(pair, batch) TOP-M nearest candidate indices by MAE
            # (no_grad; the fake is fixed for the step so this is computed
            # ONCE). ``M`` is the RELEVANT pool each D-update then samples K
            # from — this is how we keep 5b's "resample fresh GT every
            # D-update" decorrelation (the disc never sees the same K reals
            # twice across the D-loop) WITHOUT widening to the irrelevant
            # all-ride noise: only each fake's M nearest are ever eligible.
            # M defaults to 2*K; M==K reduces to deterministic top-K (no
            # resample). Override via ``ladd_gt_transition_match_pool``.
            _M = int(getattr(self.model, "ladd_gt_transition_match_pool", 0))
            if _M <= 0:
                _M = 2 * Kk
            _M = max(Kk, min(_M, n_cand))
            # Candidate-chunk size for the streaming (lazy) matcher; bounds
            # the [chunk, D] float transient. Ignored on the dense path.
            _cand_chunk = max(1, int(getattr(
                self.config, "ladd_lazy_cand_chunk",
                getattr(self.model, "ladd_lazy_cand_chunk", 64))))
            top_idx = [[None] * B for _ in range(n_pairs)]
            with torch.no_grad():
                for b in range(B):
                    fb = fq[:, b].float().flatten(1)             # [n_pairs, D]
                    _Dn = float(fb.shape[1])
                    # L1 mean distance via cdist (p=1 = sum|.|) / D — avoids
                    # materializing the [n_pairs, n_cand, D] difference tensor.
                    if cand_disc is not None:
                        ce = cand_disc[b].float().flatten(1)     # [n_cand, D]
                        mae = torch.cdist(fb, ce, p=1) / _Dn
                    else:
                        # Lazy: stream candidate rows in chunks via _cand_pair
                        # and concatenate the cdist columns. ``mae`` is
                        # [n_pairs, n_cand] — identical to the dense path; only
                        # the per-chunk [chunk, D] float buffer is ever held.
                        _cols = []
                        for _u0 in range(0, n_cand, _cand_chunk):
                            _u1 = min(_u0 + _cand_chunk, n_cand)
                            _ce_c = torch.stack([
                                _cand_pair(b, u).float().flatten()
                                for u in range(_u0, _u1)
                            ], dim=0)                            # [chunk, D]
                            _cols.append(torch.cdist(fb, _ce_c, p=1) / _Dn)
                        mae = torch.cat(_cols, dim=1)            # [n_pairs, n_cand]
                    idx = torch.topk(
                        mae, _M, dim=1, largest=False).indices.tolist()
                    for p in range(n_pairs):
                        top_idx[p][b] = idx[p]

            def _match_select(salt, carn=False):
                # Sample Kk of each fake's M nearest GT — FRESH per call
                # (seeded step+salt) — then DEDUP: forward each unique GT
                # ONCE and recover the per-fake-K block-diagonal at the logit
                # level via ``group_flat`` (gather). Dedup is essential for
                # memory: the rolled fakes are temporally adjacent so their
                # nearest-GT sets overlap heavily, keeping the disc-forward
                # row count near 5b's n_real instead of n_pairs*Kk. Count
                # varies per draw/rank, which DDP tolerates (fwd/bwd-pass
                # count is global-step driven, grads are per-param) — exactly
                # as 5b's rank-variable n_pairs already does.
                g = torch.Generator(device="cpu").manual_seed(
                    int(current_step) * 977 + int(salt) * 131 + 17)
                # ---- Phase B fix 1: cross-ride replay ring split ----
                # When the ring buffer is active (gt_transition only — ring
                # entries are 2*npb transition slabs), each fake's Kk real
                # slots are split: the first ``_Ksel`` come from the ride-
                # local matched pool (existing machinery, untouched), the
                # remaining ``Kh = Kk // 2`` are sampled UNIFORMLY from the
                # cross-ride ring (appended to ``ru`` below, after the
                # CARN/FN real-transform blocks). Ring off => _Ksel == Kk,
                # byte-identical.
                _ring = getattr(self, "_ladd_real_ring", None)
                _ring_on = (
                    chunks_per_pair == 2
                    and Kk >= 2
                    and _ring is not None and len(_ring) > 0
                    and int(getattr(
                        self.config, "ladd_real_pool_cross_ride",
                        getattr(self.model,
                                "ladd_real_pool_cross_ride", 0)) or 0) > 0
                )
                Kh = (Kk // 2) if _ring_on else 0
                _Ksel = Kk - Kh
                # B9 bank telemetry: without these numbers "cross-ride bank"
                # is a config string, not a verified mechanism.
                try:
                    self._ladd_ring_telemetry = {
                        "ring_size": int(len(_ring)) if _ring is not None else 0,
                        "ring_on": 1.0 if _ring_on else 0.0,
                        "ring_slots_per_fake": float(Kh),
                        "matched_slots_per_fake": float(_Ksel),
                    }
                except Exception:
                    pass
                # With half the K-slots served by the ring, halve the
                # matched-unique cap too so the TOTAL distinct reals
                # forwarded stays inside the existing memory envelope.
                _cap_m = (max(1, _cap // 2)
                          if (_ring_on and _cap > 0) else _cap)
                sel = [[None] * B for _ in range(n_pairs)]
                for p in range(n_pairs):
                    for b in range(B):
                        pool = top_idx[p][b]
                        if _M > _Ksel:
                            perm = torch.randperm(_M, generator=g).tolist()
                            sel[p][b] = [pool[i] for i in perm[:_Ksel]]
                        else:
                            sel[p][b] = list(pool[:_Ksel])
                # Build the UNIQUE real set + group_map, with a hard CAP on
                # the number of distinct reals forwarded (``_cap``) so the
                # wide match pool (n_cand can be ~the whole ride) is decoupled
                # from the disc-forward / R1 memory (combined forward is
                # 2*n_uniq + n_fake rows). When the cap is hit, a fake's new
                # pick is remapped to one of ITS OWN already-included nearer
                # picks (still its own relevant GT, just shared) — the
                # block-diagonal stays valid, each fake keeps Kk reals.
                key_to_row = {}
                rows_bu = []
                gm = torch.empty(n_pairs * B, Kk, dtype=torch.long)
                for p in range(n_pairs):
                    for b in range(B):
                        fr = p * B + b
                        for k in range(_Ksel):
                            u = sel[p][b][k]
                            if (b, u) in key_to_row:
                                gm[fr, k] = key_to_row[(b, u)]
                            elif _cap_m > 0 and len(rows_bu) >= _cap_m:
                                alt = next((uu for uu in sel[p][b]
                                            if (b, uu) in key_to_row), None)
                                if alt is None:
                                    alt = next((uu for (bb, uu) in rows_bu
                                                if bb == b), None)
                                gm[fr, k] = (key_to_row[(b, alt)] if alt is not None
                                             else 0)
                            else:
                                key_to_row[(b, u)] = len(rows_bu)
                                rows_bu.append((b, u))
                                gm[fr, k] = key_to_row[(b, u)]
                if Kh > 0:
                    # Temporarily point the ring slots at each fake's slot-0
                    # matched row so every intermediate consumer that scans
                    # ``gm`` BEFORE the ring rows are appended (the CARN
                    # match-pool / FN drift-cap loops below index
                    # row-stat lists sized to the matched rows) stays in
                    # range. Overwritten with real ring row ids after those
                    # blocks. Duplicating slot 0 only makes their per-row
                    # drift caps conservatively tighter, never looser.
                    gm[:, _Ksel:] = gm[:, 0:1].repeat(1, Kh)
                real_rows = [
                    (_cand_pair(b, u) if cand_disc is None
                     else cand_disc[b][u])
                    for (b, u) in rows_bu
                ]
                acts_list = []
                if a_per_f > 0 and pool_act_full is not None and atp is not None:
                    for (b, u) in rows_bu:
                        if chunks_per_pair == 1:
                            # gt_vs_fake: ONE chunk's co-located actions, so
                            # the per-frame action modulation has exactly npb
                            # rows == the disc input's F dim (t_frames).
                            acts_list.append(
                                pool_act_full[b:b + 1, u * npb:(u + 1) * npb])
                        else:
                            acts_list.append(torch.cat([
                                pool_act_full[b:b + 1, u * npb:(u + 1) * npb],
                                pool_act_full[
                                    b:b + 1, (u + 1) * npb:(u + 2) * npb],
                            ], dim=1))
                ru = torch.stack(real_rows, dim=0).to(
                    device=device, dtype=_fdt)                   # [n_uniq, ...]
                # e11: CARN the matched real FORMERS (Req 1 + Req 2). Only on
                # the D-update side (carn=True); the gen-side keeps clean
                # formers (Req 2), so the student is pulled toward clean GT.
                # FORMER/LATTER semantics (ru[:, :npb] vs ru[:, npb:]) only
                # exist for a 2-chunk transition. Under chunks_per_pair == 1
                # ``ru[:, npb:]`` is EMPTY, so this whole block is skipped
                # (gt_vs_fake has no former to degrade).
                if carn and chunks_per_pair == 2 and bool(getattr(
                        self.model, "ladd_gt_transition_carn_match_pool", False)):
                    n_uniq = ru.shape[0]
                    # Per-row cap = (min served-fake drift) - 1. A row's
                    # served fakes come from gm; fake pair p's former drift
                    # = pairs[p][0] (rollout chunk index). The cap keeps the
                    # real former noised LESS than every student former it is
                    # compared against (Req 1).
                    BIG = 1 << 30
                    row_mindrift = [BIG] * n_uniq
                    for _p in range(n_pairs):
                        _fd = int(pairs[_p][0])
                        for _b in range(B):
                            _fr = _p * B + _b
                            for _k in range(Kk):
                                _r = int(gm[_fr, _k].item())
                                if _fd < row_mindrift[_r]:
                                    row_mindrift[_r] = _fd
                    # Fixed level (>0): degrade every eligible real former by
                    # exactly this many CARN steps (still drift-capped by
                    # Req 1). 0 (default) => random level in [1, cap].
                    _fixed_lvl = int(getattr(
                        self.model,
                        "ladd_gt_transition_carn_match_pool_level", 0))
                    _levels = []
                    for _r in range(n_uniq):
                        _lvlcap = (row_mindrift[_r] - 1
                                   if row_mindrift[_r] < BIG else 0)
                        if _lvlcap < 1:
                            _levels.append(0)
                        elif _fixed_lvl > 0:
                            _levels.append(min(_fixed_lvl, _lvlcap))
                        else:
                            _levels.append(
                                int(torch.randint(
                                    1, _lvlcap + 1, (1,)).item()))
                    _maxlvl = max(_levels) if _levels else 0
                    if (getattr(self, "is_main_process", True)
                            and getattr(self, "_carn_match_pool_dbg", 0) < 3):
                        self._carn_match_pool_dbg = getattr(
                            self, "_carn_match_pool_dbg", 0) + 1
                        import sys as _sys
                        print(
                            "[CARN-MATCH-POOL] ACTIVE: n_uniq=%d "
                            "row_mindrift=%s levels=%s maxlvl=%d "
                            "(real formers get recursive CARN < served-fake "
                            "drift; gen-side stays clean)" % (
                                n_uniq, row_mindrift[:8], _levels[:8],
                                _maxlvl,
                            ),
                            file=_sys.stderr, flush=True,
                        )
                    if _maxlvl > 0:
                        _x0 = ru[:, :npb].clone()
                        _cur = _x0.clone()
                        with torch.no_grad():
                            for _kk in range(_maxlvl):
                                _idx = [r for r in range(n_uniq)
                                        if _levels[r] > _kk]
                                if not _idx:
                                    continue
                                _ii = torch.tensor(_idx, device=_cur.device)
                                _sub = _cur.index_select(0, _ii)
                                _cs = torch.zeros(
                                    (_sub.shape[0],), dtype=torch.long,
                                    device=_sub.device)
                                _sub = self.model.forward_noiser(
                                    _sub, _cs, residual=True)
                                _cur = _cur.index_copy(0, _ii, _sub)
                            # Moment-preserving (texture-only) restore.
                            _e = 1e-6
                            _mci = _x0.mean(dim=[1, 3, 4], keepdim=True)
                            _mco = _cur.mean(dim=[1, 3, 4], keepdim=True)
                            _cur = _cur - _mco + _mci
                            _ai = _x0.abs().mean(dim=[1, 2, 3, 4], keepdim=True)
                            _ao = _cur.abs().mean(dim=[1, 2, 3, 4], keepdim=True)
                            _cur = _cur * (_ai / (_ao + _e))
                        ru = torch.cat([_cur.detach(), ru[:, npb:]], dim=1)
                # FN application to the matched real pairs (both naive: the
                # SAME transformed pair feeds the D-update and the gen-side
                # real view — no Req-1 drift cap, no Req-2 gen-side shield):
                #   * apply_gt_both (h1): ONE FN call per half (former AND
                #     latter). With a REVERSE-trained FN this de-CARNs the
                #     real target (suppresses the rollout's explode/
                #     harmonise modes from the real distribution).
                #   * apply_gt_former (h3): ONE FN call on the FORMER only.
                #     With a FORWARD-trained FN the real anchor becomes
                #     [1-step-drifted GT former -> clean GT latter], the
                #     self-correcting transition.
                # Moment-preserving (texture-only) restore per transformed
                # half so only texture STRUCTURE changes (no brightness/
                # colour shift).
                _gt_both = bool(getattr(
                    self.model, "forward_noiser_apply_gt_both", False))
                _gt_former = bool(getattr(
                    self.model, "forward_noiser_apply_gt_former", False))
                if ((_gt_both or _gt_former)
                        and chunks_per_pair == 2
                        and getattr(self.model, "forward_noiser", None)
                        is not None):
                  # chain_levels drives the FORMER-only forward scheme; any
                  # gt_both (reverse de-CARN) config falls to legacy below.
                  _chain_app = bool(getattr(
                      self.model, "forward_noiser_chain_levels", True)
                  ) and _gt_former and not _gt_both
                  if _chain_app:
                    # ===== CHAINED level-conditioned former (default) =====
                    # Deterministic per-row former target level, realized by
                    # COMPOSING the conditioned increments the FN was trained
                    # on (cond = level reached: GT->1=1, GT->2=2, 1->3=3,
                    # 2->4=4, ...). L = the real row's LATTER drift level
                    # (chunk u+1). weak (default): target=(L-1)//2 (0,0,1,1,
                    # 2,2,...); strong: max(0,L-2). Capped at served-fake
                    # former drift - 1 (Req-1: never toward MORE noise). A
                    # single FN call for target<=2; recursion only PAST 2
                    # (e.g. level 4 = FN(FN(GT,cond2),cond4)). FORMER only
                    # (forward FN; the latter is the clean target).
                    _mode = str(getattr(
                        self.model, "forward_noiser_former_mode", "weak"))
                    _seed_r1 = int(self.model.dmd_context_clean_frames
                                   // self.model.num_frame_per_block)
                    _nuq = int(ru.shape[0])
                    _BIGc = 1 << 30
                    _mind = [_BIGc] * _nuq
                    for _p in range(n_pairs):
                        _fd = int(pairs[_p][0])
                        for _b in range(B):
                            _fr = _p * B + _b
                            for _k in range(Kk):
                                _r = int(gm[_fr, _k].item())
                                if _fd < _mind[_r]:
                                    _mind[_r] = _fd
                    _tgt = []
                    for _r in range(_nuq):
                        _u = int(rows_bu[_r][1])
                        _L = max(0, (_u + 1) - (_seed_r1 - 1))
                        _t = (max(0, _L - 2) if _mode == "strong"
                              else max(0, (_L - 1) // 2))
                        # Req-1 cap: _mind = min matched-fake former chunk INDEX
                        # i. The student former's true drift LEVEL is always
                        # >= i+1 (drift = i+1+frontier_offset, offset>=0), so
                        # capping the target at _mind guarantees target < the
                        # student former drift in EVERY regime (exact for
                        # stationary; conservative under rolling). NOTE: _mind
                        # was previously misused as a drift level (cap=_mind-1)
                        # which is off-by-one (too clean) — fixed here. The name
                        # _req1cap avoids the _match_select closure var _cap.
                        _req1cap = _mind[_r] if _mind[_r] < _BIGc else 0
                        _tgt.append(int(max(0, min(_t, _req1cap))))
                    _tgt_t = torch.tensor(
                        _tgt, dtype=torch.long, device=ru.device)
                    _max_t = int(_tgt_t.max().item()) if _nuq else 0
                    with torch.no_grad():
                        _x0 = ru[:, 0:npb]
                        _cur = _x0.clone()
                        # Apply cond=c to rows whose chain includes c (target
                        # >= c and SAME parity) — increasing c feeds each row
                        # its conditioned increments in order at the right
                        # input level (clean -> start -> start+2 -> ... -> t).
                        for _c in range(1, _max_t + 1):
                            _sel = ((_tgt_t >= _c)
                                    & ((_tgt_t % 2) == (_c % 2))).nonzero(
                                        as_tuple=False).flatten()
                            if _sel.numel() == 0:
                                continue
                            _sub = _cur.index_select(0, _sel)
                            _cs = torch.full(
                                (_sub.shape[0],), int(_c),
                                dtype=torch.long, device=_sub.device)
                            _sub = self.model.forward_noiser(
                                _sub, _cs, residual=True)
                            _cur = _cur.index_copy(0, _sel, _sub)
                        # Moment-preserving (texture-only) restore vs original.
                        _e = 1e-6
                        _mci = _x0.mean(dim=[1, 3, 4], keepdim=True)
                        _mco = _cur.mean(dim=[1, 3, 4], keepdim=True)
                        _cur = _cur - _mco + _mci
                        _ai = _x0.abs().mean(
                            dim=[1, 2, 3, 4], keepdim=True)
                        _ao = _cur.abs().mean(
                            dim=[1, 2, 3, 4], keepdim=True)
                        _cur = _cur * (_ai / (_ao + _e))
                        # target-0 rows keep the ORIGINAL clean former.
                        _keep = (_tgt_t > 0).view(-1, 1, 1, 1, 1)
                        _cur = torch.where(_keep, _cur, _x0)
                        ru = torch.cat([_cur.detach(), ru[:, npb:]], dim=1)
                    if (getattr(self, "is_main_process", True)
                            and getattr(self, "_fn_chain_app_dbg", 0) < 3):
                        self._fn_chain_app_dbg = getattr(
                            self, "_fn_chain_app_dbg", 0) + 1
                        import sys as _sys
                        print(
                            "[FN-CHAIN-APP] mode=%s n=%d former targets=%s "
                            "(cap=served-fake drift-1; single<=2, recurse>2)"
                            % (_mode, _nuq, _tgt[:12]),
                            file=_sys.stderr, flush=True,
                        )
                  else:
                    # Conditioning level for the application. A step-
                    # UNCONDITIONED FN trained only at carn_step=0, so apply
                    # at 0. A step-CONDITIONED FN (give it the numbers)
                    # trained at the input's drift level L; applying at
                    # forward_noiser_apply_gt_level (default 1) adds/removes
                    # one drift-level's worth depending on the FN direction.
                    _step_uncond = bool(getattr(
                        self.model,
                        "forward_noiser_step_unconditioned", False))
                    _apply_lvl = 0 if _step_uncond else int(getattr(
                        self.model, "forward_noiser_apply_gt_level", 1))
                    # h4: per-row random level in [apply_lvl, lvl_max]
                    # (step-conditioned only). 0/<=apply_lvl = fixed level.
                    _lvl_max = 0 if _step_uncond else int(getattr(
                        self.model, "forward_noiser_apply_gt_level_max", 0))
                    # h5: drift-capped per-row level — rand[1, mindrift-1]
                    # where mindrift = min drift step of the student
                    # formers this real row is served against (via gm).
                    # Rows served only by drift<=1 fakes keep a CLEAN
                    # former (level 0 -> FN output discarded below).
                    _drift_cap = (not _step_uncond) and bool(getattr(
                        self.model,
                        "forward_noiser_apply_gt_drift_cap", False))
                    _fn_offsets = (0, npb) if _gt_both else (0,)
                    with torch.no_grad():
                        if _drift_cap:
                            _BIGc = 1 << 30
                            _mind = [_BIGc] * int(ru.shape[0])
                            for _p in range(n_pairs):
                                _fd = int(pairs[_p][0])
                                for _b in range(B):
                                    _fr = _p * B + _b
                                    for _k in range(Kk):
                                        _r = int(gm[_fr, _k].item())
                                        if _fd < _mind[_r]:
                                            _mind[_r] = _fd
                            _lvls = []
                            for _r in range(int(ru.shape[0])):
                                _cp = (_mind[_r] - 1
                                       if _mind[_r] < _BIGc else 0)
                                _lvls.append(
                                    int(torch.randint(
                                        1, _cp + 1, (1,)).item())
                                    if _cp >= 1 else 0)
                            _cs0 = torch.tensor(
                                _lvls, dtype=torch.long, device=ru.device)
                        elif _lvl_max > _apply_lvl:
                            _cs0 = torch.randint(
                                _apply_lvl, _lvl_max + 1, (ru.shape[0],),
                                dtype=torch.long, device=ru.device)
                        else:
                            _cs0 = torch.full(
                                (ru.shape[0],), _apply_lvl,
                                dtype=torch.long, device=ru.device)
                        _halves = []
                        for _h0 in (0, npb):
                            _x0 = ru[:, _h0:_h0 + npb]
                            if _h0 not in _fn_offsets:
                                _halves.append(_x0)
                                continue
                            _cur = self.model.forward_noiser(
                                _x0, _cs0, residual=True)
                            _e = 1e-6
                            _mci = _x0.mean(dim=[1, 3, 4], keepdim=True)
                            _mco = _cur.mean(dim=[1, 3, 4], keepdim=True)
                            _cur = _cur - _mco + _mci
                            _ai = _x0.abs().mean(
                                dim=[1, 2, 3, 4], keepdim=True)
                            _ao = _cur.abs().mean(
                                dim=[1, 2, 3, 4], keepdim=True)
                            _cur = _cur * (_ai / (_ao + _e))
                            # Level-0 rows (drift-cap mode: served fakes at
                            # drift <= 1) keep the ORIGINAL clean half — the
                            # FN was never trained at level 0, so its output
                            # there is undefined; discard it.
                            _keep = (_cs0 > 0).view(-1, 1, 1, 1, 1)
                            _cur = torch.where(_keep, _cur, _x0)
                            _halves.append(_cur)
                        ru = torch.cat(_halves, dim=1).detach()
                    if (getattr(self, "is_main_process", True)
                            and getattr(self, "_fn_gt_both_dbg", 0) < 2):
                        self._fn_gt_both_dbg = getattr(
                            self, "_fn_gt_both_dbg", 0) + 1
                        import sys as _sys
                        if _drift_cap:
                            _lvl_desc = (
                                "drift-capped rand[1,mindrift-1] "
                                "(levels=%s)" % (
                                    _cs0.tolist()[:8],))
                        elif _lvl_max > _apply_lvl:
                            _lvl_desc = "rand[%d,%d]" % (
                                _apply_lvl, _lvl_max)
                        else:
                            _lvl_desc = str(_apply_lvl)
                        print(
                            "[FN-GT-%s] ACTIVE: FN applied to %s of %d "
                            "matched real pairs at carn_step=%s "
                            "(step_uncond=%s, naive both-sides)" % (
                                "BOTH" if _gt_both else "FORMER",
                                "BOTH GT latents" if _gt_both
                                else "the FORMER GT latent",
                                ru.shape[0], _lvl_desc, _step_uncond),
                            file=_sys.stderr, flush=True,
                        )
                # ---- Phase B fix 1: append the cross-ride ring reals ----
                # AFTER the CARN/FN real-transform blocks above (those only
                # ever see/transform the ride-local matched rows; ring
                # reals stay clean GT transitions — with the FN/CARN
                # real-transform flags OFF, as in every Phase B config,
                # the two populations are treated identically). Each ring
                # row carries its OWN ride's actions and prompt embeds.
                rpe = None
                self._ladd_ring_last_n = 0.0
                _div_picks = []          # A6: ring entries used this draw
                if Kh > 0:
                    _ring_list = list(_ring)
                    _n_mu = int(ru.shape[0])
                    # Unique ring rows this update: at least Kh, at most
                    # the remaining distinct-real budget under ``_cap``
                    # (so total rows stay inside the pre-ring envelope).
                    _budget = (_cap - _n_mu) if _cap > 0 else 2 * Kh
                    _R = min(len(_ring_list), max(Kh, max(1, _budget)))
                    _rperm = torch.randperm(
                        len(_ring_list), generator=g).tolist()
                    _picks = [_ring_list[i] for i in _rperm[:_R]]
                    _div_picks = _picks  # A6 (reference, not a copy)
                    _ring_lat = []
                    for _ent in _picks:
                        _rl = _ent["lat"].to(
                            device=device, dtype=_fdt).unsqueeze(0)
                        if _mean_eq:
                            # Same per-pair self-normalization the matched
                            # candidates get in _cand_pair.
                            _rl = _mean_equalize_pair(_rl)
                        _ring_lat.append(_rl[0])
                    ru = torch.cat(
                        [ru, torch.stack(_ring_lat, dim=0)], dim=0)
                    if (a_per_f > 0 and pool_act_full is not None
                            and atp is not None):
                        for _ent in _picks:
                            acts_list.append(_ent["act"].to(
                                device=pool_act_full.device,
                                dtype=pool_act_full.dtype))
                    # Overwrite the placeholder ring slots with real ring
                    # row ids (uniform per (fake, slot), seeded generator).
                    for _fr in range(n_pairs * B):
                        for _kk2 in range(Kh):
                            _ridx = int(torch.randint(
                                0, _R, (1,), generator=g).item())
                            gm[_fr, _Ksel + _kk2] = _n_mu + _ridx
                    # Per-row prompt embeds: matched rows use the current
                    # ride's prompt, ring rows their stored one. Zero-pad
                    # to a common seq_len (caption embeds are variable-
                    # length; WAN cross-attn treats zero rows as padding).
                    _pe_parts = (
                        [prompt_embeds[0:1]] * _n_mu
                        + [_ent["pe"][0:1].to(
                            device=device, dtype=prompt_embeds.dtype)
                           for _ent in _picks])
                    _Lmax = max(int(p.shape[1]) for p in _pe_parts)
                    _pe_parts = [
                        (p if int(p.shape[1]) == _Lmax else
                         torch.nn.functional.pad(
                             p, (0, 0, 0, _Lmax - int(p.shape[1]))))
                        for p in _pe_parts
                    ]
                    rpe = torch.cat(_pe_parts, dim=0)
                    self._ladd_ring_last_n = float(_R)
                gflat = gm.reshape(-1).to(device)                # [n_pairs*B*Kk]
                rat = ram = None
                if acts_list:
                    _ra = torch.cat(acts_list, dim=0).to(
                        device=device, dtype=prompt_embeds_eff.dtype)
                    with torch.no_grad():
                        rat = atp(_ra).detach()
                        if ap is not None:
                            ram = ap(_ra, num_frames=_ra.shape[1]).detach()
                if _action_blind:
                    if rat is not None:
                        rat = torch.zeros_like(rat)
                    if ram is not None:
                        ram = torch.zeros_like(ram)
                # ---- A6: real-sample diversity telemetry (default OFF) ----
                # Pure logging. Answers "is the real pool actually diverse"
                # for the batch the D is about to consume, which the docs
                # currently flag [UNVERIFIED]. Counted over the real SLOTS
                # the D really sees (``gm``, one row id per (fake, k) slot),
                # not over the candidate pool — a wide pool that always
                # resolves to the same 3 nearest GT chunks is not diverse.
                #   * matched rows are ride-local  -> key (cur_ride, b, u)
                #   * cross-ride ring rows carry their own provenance tag
                # ``repeat_rate`` = 1 - unique_windows / slots: 0.0 = every
                # slot a different source window, ->1.0 = the same handful of
                # reals shown over and over.
                if bool(getattr(self, "gan_real_diversity_log", False)):
                    try:
                        _cur_ride = ""
                        _ss_d = getattr(self.model, "streaming_state", None)
                        if isinstance(_ss_d, dict):
                            _cur_ride = str(_ss_d.get("zarr_path", "") or "")
                        _row_src = [
                            (_cur_ride, int(_b), int(_u))
                            for (_b, _u) in rows_bu
                        ]
                        _n_matched_rows = len(_row_src)
                        _n_unknown = 0
                        for _i_pk, _pk in enumerate(_div_picks):
                            _sk = _pk.get("src") if isinstance(_pk, dict) \
                                else None
                            if _sk is None:
                                # Entry pushed before the flag was on: keep
                                # it a DISTINCT window (so the window count
                                # stays right) but an unknown ride, and count
                                # it so the number is never quietly wrong.
                                _n_unknown += 1
                                _row_src.append(
                                    ("<unknown-ride>", -1, -1 - _i_pk))
                            else:
                                _row_src.append(
                                    (str(_sk[0]), -2, int(_sk[1])))
                        _gm_flat_l = gm.reshape(-1).tolist()
                        _slots = len(_gm_flat_l)
                        _used = [int(_r) for _r in _gm_flat_l
                                 if 0 <= int(_r) < len(_row_src)]
                        _src_used = [_row_src[_r] for _r in _used]
                        _uniq_win = len(set(_src_used))
                        _uniq_rides = len({_t[0] for _t in _src_used})
                        _n_ring_slots = sum(
                            1 for _r in _used if _r >= _n_matched_rows)
                        self._ladd_stash_real_div(pair_mode, current_step, {
                            "gan_real_slots_total": float(_slots),
                            "gan_real_unique_rows": float(len(set(_used))),
                            "gan_real_unique_rides": float(_uniq_rides),
                            "gan_real_unique_windows": float(_uniq_win),
                            "gan_real_repeat_rate": (
                                1.0 - (_uniq_win / float(_slots))
                                if _slots > 0 else 0.0),
                            "gan_real_ring_slot_frac": (
                                _n_ring_slots / float(_slots)
                                if _slots > 0 else 0.0),
                            "gan_real_src_untagged": float(_n_unknown),
                        })
                    except Exception:
                        self._ladd_stash_real_div(pair_mode, current_step, {
                            "gan_real_div_err": 1.0,
                        })
                return ru, rat, ram, rpe, gflat

            def _m_noise(x):
                if disc_t_int <= 0:
                    return x
                scheduler = getattr(self.model, "scheduler", None)
                if scheduler is None or not hasattr(scheduler, "add_noise"):
                    raise RuntimeError(
                        "_m_noise: disc_t_int>0 but no scheduler.add_noise")
                eps = torch.randn_like(x)
                xf, ef = x.flatten(0, 1), eps.flatten(0, 1)
                tpf = torch.full(
                    (xf.shape[0],), disc_t_int, dtype=torch.long, device=x.device)
                return scheduler.add_noise(xf, ef, tpf).unflatten(0, x.shape[:2])

            def _m_fwd(disc, segs):
                # Each seg is (latents, action_tokens, action_modulation)
                # or — Phase B fix 1 — a 4-tuple with per-ROW prompt embeds
                # ``[rows, L, D]`` as the 4th element (used by the cross-
                # ride replay ring, whose real rows carry their OWN ride's
                # prompt). 3-tuples are normalized to (…, None); the all-
                # None fast path below is byte-identical to the legacy
                # broadcast, so existing configs are unchanged.
                segs = [tuple(s) + (None,) * (4 - len(s)) for s in segs]
                xs = [s[0] for s in segs]
                counts = [int(s[0].shape[0]) for s in segs]
                x = torch.cat(xs, dim=0)
                n_rows = x.shape[0]
                t = torch.full(
                    (n_rows, t_frames), disc_t_int, dtype=torch.long, device=device)
                if all(s[3] is None for s in segs):
                    reps = max(1, n_rows // int(prompt_embeds.shape[0]))
                    pe = prompt_embeds.repeat(reps, 1, 1)
                    pp = (
                        prompt_embeds.float().mean(dim=1).repeat(reps, 1)
                        if _need_pp else None)
                else:
                    # Per-row prompts on at least one seg: build pe row-
                    # aligned with x. Rows without an explicit prompt use
                    # the current ride's (tiled). Different rides' caption
                    # embeds can have different seq_len L — zero-pad to the
                    # longest (WAN cross-attn treats zero rows as inert
                    # padding, matching the encoder's own pad convention).
                    _pe_rows = []
                    for s, c in zip(segs, counts):
                        if s[3] is not None:
                            _pe_rows.append(s[3].to(
                                device=device, dtype=prompt_embeds.dtype))
                        else:
                            _pe_rows.append(prompt_embeds.repeat(c, 1, 1))
                    # pooled prompt per row over each prompt's OWN (unpadded)
                    # seq_len — padding zeros must not dilute the mean.
                    pp = (
                        torch.cat([p.float().mean(dim=1) for p in _pe_rows],
                                  dim=0)
                        if _need_pp else None)
                    _Lmax = max(int(p.shape[1]) for p in _pe_rows)
                    _pe_rows = [
                        (p if int(p.shape[1]) == _Lmax else
                         torch.nn.functional.pad(
                             p, (0, 0, 0, _Lmax - int(p.shape[1]))))
                        for p in _pe_rows
                    ]
                    pe = torch.cat(_pe_rows, dim=0)
                ce = None
                if any(s[1] is not None for s in segs):
                    ce = {"_action_tokens": torch.cat(
                        [s[1] for s in segs], dim=0)}
                    if all(s[2] is not None for s in segs):
                        ce["_action_modulation"] = torch.cat(
                            [s[2] for s in segs], dim=0)
                # Micro-batch the disc forward (EVAL-mode only = the gen-side
                # guidance path) to cut the checkpointed teacher-recompute
                # peak that OOMs the gen backward. Splitting the row dim and
                # concatenating logits is per-row independent -> byte-IDENTICAL
                # logits + gradients (ZERO performance change); only the
                # per-recompute activation peak drops to one micro-batch.
                # EVAL-ONLY: a train-mode disc forward must stay a SINGLE
                # forward (chunking would run spectral_norm's power-iteration
                # G times and change _sigma -> a real result change). Flag
                # ladd_gen_guidance_micro_batch_groups (default 1 = single
                # forward, byte-identical for every existing config).
                _gmg = int(getattr(
                    self.config, "ladd_gen_guidance_micro_batch_groups",
                    getattr(self.model,
                            "ladd_gen_guidance_micro_batch_groups", 1)))
                _Gm = max(1, min(_gmg, n_rows))
                if _Gm <= 1 or disc.training:
                    logits = disc(
                        x_noisy=x, timestep=t, prompt_embeds=pe,
                        pooled_prompt=pp, conditional_extra=ce)
                else:
                    _mb = [(g * n_rows) // _Gm for g in range(_Gm + 1)]
                    _lps = []
                    for _gi in range(_Gm):
                        _lo, _hi = _mb[_gi], _mb[_gi + 1]
                        if _hi <= _lo:
                            continue
                        _cep = ({k: v[_lo:_hi] for k, v in ce.items()}
                                if ce is not None else None)
                        _lps.append(disc(
                            x_noisy=x[_lo:_hi], timestep=t[_lo:_hi],
                            prompt_embeds=pe[_lo:_hi],
                            pooled_prompt=(pp[_lo:_hi]
                                           if pp is not None else None),
                            conditional_extra=_cep))
                    logits = torch.cat(_lps, dim=0)
                out, o = [], 0
                for c in counts:
                    out.append(logits[o:o + c])
                    o += c
                return out

            def _m_rp(d_real_uniq, d_fake, detach_real, group_flat):
                # Block-diagonal RpGAN. ``group_flat`` gathers each unique
                # real's logit into the (pair, batch, k)-major layout; the
                # fakes are repeat_interleave'd to match (fake row (p,b)
                # repeated Kk times lines up with its Kk matched reals). Each
                # fake is scored ONLY against its own K matched GT (NOT
                # all_pairs): standard MATCHED RpGAN on the aligned rows.
                rr = d_real_uniq.detach() if detach_real else d_real_uniq
                rg = rr.index_select(0, group_flat)              # [Nf*Kk, tok]
                fg = d_fake.repeat_interleave(Kk, dim=0)         # [Nf*Kk, tok]
                mfn = rpgan_g_loss if detach_real else rpgan_d_loss
                if _K_stat > 0 and _W_stat > 0.0:
                    rv, rs = rg[:, :-_K_stat], rg[:, -_K_stat:]
                    fv, fs = fg[:, :-_K_stat], fg[:, -_K_stat:]
                    lv = mfn(rv, fv)
                    ls = mfn(rs, fs)
                    return lv + _W_stat * ls, float(ls.detach().item())
                return mfn(rg, fg), 0.0

            disc_skipped = (
                current_step < int(getattr(self, "gan_disc_start_step", 0)))
            last_d_loss = last_d_real = last_d_fake = 0.0
            last_r1 = last_r1_grad_sq = last_r1_fired = 0.0
            last_d_loss_stat = last_d_real_stat = last_d_fake_stat = 0.0
            last_n_real = 0.0
            # A16-D1 telemetry: how many R1 penalties this (step, pair_mode)
            # D-update BLOCK actually applies. NaN when the unified cadence is
            # off (the legacy per-iteration latch has no block-level answer),
            # so the trace plots a gap rather than a misleading 0.
            last_r1_block_fires = float("nan")
            _fseg_act = (fake_action_tokens, fake_action_modulation)

            # ---- D-updates (resample each fake's K matched GT FRESH per
            # update — salt=_it — to preserve 5b's GT decorrelation).
            # Wrapped in a closure so it can be DEFERRED past the outer
            # ``generator_loss.backward()`` when ``ladd_defer_disc_update``
            # is set. The D-update reads ONLY detached inputs (``_fk`` /
            # detached actions / the detached GT match pool), so its result
            # is independent of WHEN it runs relative to the gen backward —
            # but running it AFTER lets the gen graph (~88 GB) free first, so
            # the disc backward no longer stacks on it (the FT_v3 rolling
            # OOM). Flag OFF (default) -> ``_run_disc_updates()`` is called
            # inline here, BYTE-IDENTICAL to the prior behaviour. When
            # deferred, the gen-side term below sees the PRE-update disc
            # (standard G-then-D ordering). ``nonlocal`` keeps the inline
            # path's logging-var writes intact.
            # ---- Memory-fix knobs (default-OFF => byte-identical) ----
            #   ladd_disc_micro_batch_groups (int, default 1): split the
            #     n_fake fakes into this many groups and do one disc
            #     forward+backward per group with grad accumulation. 1 = the
            #     existing single-batch path (verbatim).
            #   ladd_r1_num_samples (int, default 0): when >0, compute FD-R1
            #     on at most this many UNIQUE reals (deterministic subsample
            #     seeded by current_step); 0 = all reals (current). A17
            #     (2026-08-23): honoured on BOTH matched paths now — it used
            #     to apply only when ladd_disc_micro_batch_groups > 1 and was
            #     silently ignored on the inline path. NOT read by the
            #     POSITIONAL D-loop, which has no unique-real structure (its
            #     reals are the ride's own positional chunks).
            _micro_groups = int(getattr(
                self.config, "ladd_disc_micro_batch_groups",
                getattr(self.model, "ladd_disc_micro_batch_groups", 1)))
            _r1_num_samples = int(getattr(
                self.config, "ladd_r1_num_samples",
                getattr(self.model, "ladd_r1_num_samples", 0)))

            def _disc_no_sync():
                # DDP fires an all-reduce on EVERY .backward() unless wrapped
                # in no_sync(). ``disc_for_update`` is the DDP wrapper when
                # DDP is active (r3gan_disc_ddp), else the plain module (no
                # no_sync attr) -> contextlib.nullcontext is the no-op.
                _m = disc_for_update
                if hasattr(_m, "no_sync"):
                    return _m.no_sync()
                import contextlib
                return contextlib.nullcontext()

            def _run_disc_updates():
                nonlocal last_d_loss, last_d_real, last_d_fake, last_r1
                nonlocal last_r1_grad_sq, last_r1_fired, last_d_loss_stat
                nonlocal last_n_real
                nonlocal last_r1_block_fires
                # ---- A16-D1: R1 cadence decided ONCE for this D-update
                # block (see ``_ladd_r1_block_due`` for the full contract +
                # fire-rate table). ``ladd_r1_unified_cadence`` default False
                # => ``_r1_block_due`` stays None and the legacy per-iteration
                # latch below is evaluated verbatim (byte-identical).
                _r1_unified_m = bool(getattr(
                    self.config, "ladd_r1_unified_cadence",
                    getattr(self.model, "ladd_r1_unified_cadence", False)))
                _r1_block_due = None
                if _r1_unified_m and n_disc_updates > 0 and (
                        self.r3gan_optimizer is not None):
                    _r1_block_due = self._ladd_r1_block_due(
                        pair_mode, current_step, _r1_every_n)
                    last_r1_block_fires = float(
                        n_disc_updates if _r1_block_due else 0)
                if n_disc_updates > 0 and self.r3gan_optimizer is not None:
                    for _it in range(n_disc_updates):
                        self.r3gan_optimizer.zero_grad(set_to_none=True)
                        # D-update: POST-CARN matched real formers (Req 2).
                        (real_m, real_m_rat, real_m_ram, real_m_rpe,
                         group_flat) = _match_select(_it, carn=True)
                        last_n_real = float(real_m.shape[0])
                        _rn = _m_noise(real_m)
                        if diff_aug_policy:
                            _rn, _ = latent_diff_augment(
                                _rn, _rn, policy=diff_aug_policy,
                                seed=int(current_step) * 31
                                + seed_offset + _it * 101)
                        _rn = _rn.detach()
                        _fk = _m_noise(fake_chunks_det).detach()
                        # ---- flag-gated INPUT DUMP (matched-real branch) --
                        # Stash what the disc D-update actually scores this
                        # step: the matched (+ CARN/noise/diffaug) real
                        # transition slabs and the fake transition slabs,
                        # PRE-wavelet (the wavelet transform lives inside
                        # the disc forward). Main-rank only; try/except so
                        # diagnostics can never crash training.
                        if _it == 0 and getattr(self, "is_main_process", False):
                            try:
                                _dbg_every = int(getattr(
                                    self.config,
                                    "debug_dump_scorer_inputs_every",
                                    getattr(self.model,
                                            "debug_dump_scorer_inputs_every",
                                            0)) or 0)
                                # DEBT-based cadence (mirrors the R1
                                # fix): the GAN only runs on gen iters,
                                # so ``step % every`` can be missed
                                # forever when the residues never align.
                                _dbg_last = int(getattr(
                                    self, "_dbg_gan_last_fire", -10 ** 9))
                                if (_dbg_every > 0
                                        and int(current_step) - _dbg_last
                                        >= _dbg_every):
                                    self._dbg_gan_last_fire = int(
                                        current_step)
                                    self._dbg_gan_dump = {
                                        "real": _rn.detach().to(
                                            device="cpu",
                                            dtype=torch.float32),
                                        "fake": _fk.detach().to(
                                            device="cpu",
                                            dtype=torch.float32),
                                        "step": int(current_step),
                                        "mode": f"{pair_mode}/match",
                                    }
                            except Exception:
                                pass
                        _fseg = (_fk, _fseg_act[0], _fseg_act[1])
                        # R1 (real-side) gradient penalty on its own lazy
                        # cadence. The perturbed real segment is appended to
                        # the SINGLE batched disc forward only on the steps
                        # it fires.
                        # DEBT-BASED cadence (2026-08-19). The old
                        # ``current_step % _r1_every_n == 0`` required EXACT
                        # alignment, but the GAN only runs on generator iters
                        # (step % dfake_gen_update_ratio == 0), so whenever
                        # _r1_every_n and the gen ratio are not commensurate
                        # the modulo can be missed forever. Fire instead when
                        # at least _r1_every_n steps have elapsed since the
                        # last ACTUAL firing -- slightly delayed at times, but
                        # it can never be skipped indefinitely.
                        # A16-D1: with ``ladd_r1_unified_cadence`` the whole
                        # block shares ONE decision (per pair_mode), so all
                        # ``n_disc_updates`` iterations fire on a due step
                        # instead of only the first. Default OFF -> the legacy
                        # per-iteration global latch below runs verbatim.
                        if _r1_block_due is None:
                            _last_r1_at = int(getattr(self, "_ladd_last_r1_step", -10**9))
                            _do_r1 = (current_step - _last_r1_at >= _r1_every_n)
                            if _do_r1:
                                self._ladd_last_r1_step = int(current_step)
                        else:
                            _do_r1 = bool(_r1_block_due)
                        self._ladd_count_penalty_fires(_do_r1, pair_mode)
                        if _micro_groups > 1:
                            _log = self._ladd_disc_update_microbatched(
                                _it=_it, _rn=_rn, _fk=_fk, _fseg=_fseg,
                                _fseg_act=_fseg_act, real_m=real_m,
                                real_m_rat=real_m_rat, real_m_ram=real_m_ram,
                                real_m_rpe=real_m_rpe,
                                group_flat=group_flat, _do_r1=_do_r1,
                                n_pairs=n_pairs, B=B, Kk=Kk,
                                _m_fwd=_m_fwd, disc_for_update=disc_for_update,
                                _disc_no_sync=_disc_no_sync,
                                rpgan_d_loss=rpgan_d_loss, _K_stat=_K_stat,
                                _W_stat=_W_stat, _r1_sigma=_r1_sigma,
                                _r1_gamma=_r1_gamma,
                                _r1_num_samples=_r1_num_samples,
                                current_step=current_step,
                                _micro_groups=_micro_groups)
                            last_d_loss = _log["d_loss"]
                            last_d_real = _log["d_real"]
                            last_d_fake = _log["d_fake"]
                            last_r1 = max(last_r1, _log["r1"])
                            last_r1_grad_sq = max(last_r1_grad_sq, _log["r1_grad_sq"])
                            last_r1_fired = max(last_r1_fired, _log["r1_fired"])
                            last_d_loss_stat = _log["d_loss_stat"]
                            continue
                        _segs = [
                            (_rn, real_m_rat, real_m_ram, real_m_rpe),
                            _fseg]
                        # ---- A17 (2026-08-23): honour ladd_r1_num_samples
                        # on the INLINE matched path too. It used to be read
                        # only by ``_ladd_disc_update_microbatched``, so with
                        # ``ladd_disc_micro_batch_groups=1`` (the raw_t0
                        # default) a recipe asking for 6 R1 reals silently
                        # perturbed and forwarded ALL of them — the exact
                        # cost/memory the knob exists to bound. Subsample the
                        # UNIQUE reals with the SAME deterministic,
                        # DDP-consistent seeding the micro-batched path uses
                        # (identical rows for identical step/_it), perturb
                        # only those, and take the FD against the matching
                        # baseline rows. The estimator stays unbiased: the
                        # full penalty is a mean of per-real squared FDs, and
                        # this is the mean over a uniform random subset of
                        # exactly those terms. ``ladd_r1_num_samples <= 0``
                        # (default) or >= n_uniq => ``_r1_rows is None`` and
                        # every expression below is the verbatim old code
                        # (byte-identical, same RNG draw).
                        _r1_rows = None
                        if _do_r1:
                            _n_uniq_i = int(_rn.shape[0])
                            if 0 < _r1_num_samples < _n_uniq_i:
                                _gseed = (int(current_step) * 1000003 + 7
                                          + int(_it) * 9176)
                                _gcpu = torch.Generator(
                                    device="cpu").manual_seed(_gseed)
                                _r1_rows = torch.randperm(
                                    _n_uniq_i, generator=_gcpu
                                )[:_r1_num_samples].sort().values.to(
                                    _rn.device)
                                if not getattr(
                                        self, "_ladd_r1_inline_sub_warned",
                                        False):
                                    self._ladd_r1_inline_sub_warned = True
                                    import sys as _sys_r1
                                    print(
                                        "[LADD-R1] inline matched path: "
                                        f"ladd_r1_num_samples="
                                        f"{_r1_num_samples} < n_unique_reals"
                                        f"={_n_uniq_i} -> R1 perturbs a "
                                        "deterministic subsample (A17). Was "
                                        "silently ignored on this path "
                                        "before 2026-08-23.",
                                        file=_sys_r1, flush=True)
                        if _do_r1:
                            if _r1_rows is None:
                                _rn_r1 = _rn
                                _rat_r1, _ram_r1, _rpe_r1 = (
                                    real_m_rat, real_m_ram, real_m_rpe)
                            else:
                                _rn_r1 = _rn.index_select(0, _r1_rows)
                                _sel = (lambda _t: None if _t is None
                                        else _t.index_select(0, _r1_rows))
                                _rat_r1 = _sel(real_m_rat)
                                _ram_r1 = _sel(real_m_ram)
                                _rpe_r1 = _sel(real_m_rpe)
                            _eps_r = _r1_sigma * torch.randn_like(_rn_r1)
                            _segs.append(
                                (_rn_r1 + _eps_r, _rat_r1, _ram_r1,
                                 _rpe_r1))
                        self._mem_step_snapshot(
                            f"disc_it{_it}_pre_fwd_nseg{len(_segs)}"
                            f"_r1{int(_do_r1)}")
                        _outs = _m_fwd(disc_for_update, _segs)
                        self._mem_step_snapshot(f"disc_it{_it}_post_fwd")
                        d_r, d_f = _outs[0], _outs[1]
                        _nxt = 2
                        if _do_r1:
                            d_r_pert = _outs[_nxt]
                            _nxt += 1
                            # ladd_r1_normalize_tokens (default off): divide the
                            # summed-logit FD by the token count so grad_sq is
                            # ‖∇ MEAN_i D_i‖² (token-count-independent) instead of
                            # ‖∇ Σ_i D_i‖² (which scales with ~T² over ~47k
                            # tokens, forcing γ to a tiny un-portable value).
                            _r1_tok = (
                                float(d_r.shape[1]) if bool(getattr(
                                    self.config, "ladd_r1_normalize_tokens",
                                    getattr(self.model,
                                            "ladd_r1_normalize_tokens", False)))
                                else 1.0)
                            # A17: baseline rows must line up with the
                            # perturbed segment. ``_r1_rows is None`` (the
                            # default) => ``_d_r_base is d_r``, verbatim.
                            _d_r_base = (
                                d_r if _r1_rows is None
                                else d_r.index_select(0, _r1_rows))
                            _gsq = (
                                ((d_r_pert.sum(dim=1) - _d_r_base.sum(dim=1))
                                 / (_r1_sigma * _r1_tok)).pow(2).mean())
                            r1 = 0.5 * _r1_gamma * _gsq
                            last_r1_grad_sq = float(_gsq.detach().item())
                            last_r1_fired = 1.0
                        else:
                            r1 = d_r.sum() * 0.0
                        d_rp, _dstat = _m_rp(d_r, d_f, False, group_flat)
                        self._mem_step_snapshot(f"disc_it{_it}_pre_bwd")
                        (d_rp + r1).backward()
                        self._mem_step_snapshot(f"disc_it{_it}_post_bwd")
                        if self.gan_max_grad_norm and self.gan_max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                [p for p in self.r3gan_disc.parameters()
                                 if p.grad is not None],
                                self.gan_max_grad_norm)
                        self.r3gan_optimizer.step()
                        last_d_loss = float(d_rp.detach().item())
                        last_d_real = float(d_r.detach().mean().item())
                        last_d_fake = float(d_f.detach().mean().item())
                        last_r1 = float(r1.detach().item())
                        last_d_loss_stat = _dstat
            if bool(getattr(self.config, "ladd_defer_disc_update", False)):
                self._ladd_pending_disc = _run_disc_updates
            else:
                _run_disc_updates()

            # Persist the last-fired R1 grad_sq for the wandb trace. Lazy R1
            # fires every ladd_r1_every_n_steps, but wandb logs on its own
            # (coprime) interval, so fire-steps were never sampled and the
            # r3gan_r1_grad_sq trace read 0 even though R1 was firing. Hold the
            # most-recent fired value (per pair_mode) so its magnitude is
            # visible at every log step. LOGGING ONLY — the applied penalty
            # (last_r1 / the .backward()) is unchanged; this only overrides the
            # diagnostic last_r1_grad_sq on non-fire steps.
            if not hasattr(self, "_r1_gsq_hold"):
                self._r1_gsq_hold = {}
            if last_r1_fired > 0.0:
                self._r1_gsq_hold[pair_mode] = last_r1_grad_sq
            elif pair_mode in self._r1_gsq_hold:
                last_r1_grad_sq = self._r1_gsq_hold[pair_mode]

            # ---- Gen-side ----
            critic_warmup_done = (
                current_step >= int(getattr(self, "gan_critic_warmup_steps", 0)))
            if (
                self.gan_warmup_steps > 0
                and current_step < (
                    self.gan_critic_warmup_steps + self.gan_warmup_steps)
                and current_step >= self.gan_critic_warmup_steps
            ):
                _ramp_in = current_step - self.gan_critic_warmup_steps
                gen_gan_weight = self._gan_warmup_shape_apply(
                    _ramp_in / max(1, self.gan_warmup_steps)
                ) * self.gan_loss_weight
            elif current_step >= (
                self.gan_critic_warmup_steps + self.gan_warmup_steps
            ):
                gen_gan_weight = self.gan_loss_weight
            else:
                gen_gan_weight = 0.0
            gen_gan_weight = gen_gan_weight * float(
                getattr(self.model, "ladd_disc_loss_weight", 1.0))
            gen_gan_weight = self._couple_gan_weight_to_gate(gen_gan_weight)

            generator_gan_loss = zero
            gen_gan_main_value = 0.0
            gen_gan_stat_value = 0.0
            if critic_warmup_done and gen_gan_weight > 0 and not skip_g:
                disc_for_guidance.requires_grad_(False)
                _disc_was_training = disc_for_guidance.training
                disc_for_guidance.eval()
                try:
                    (real_m, real_m_rat, real_m_ram, real_m_rpe,
                     group_flat) = _match_select(7919)
                    _rn = _m_noise(real_m).detach()
                    _fg = _m_noise(fake_chunks_grad_tensor)
                    if diff_aug_policy:
                        _rn, _ = latent_diff_augment(
                            _rn, _rn, policy=diff_aug_policy,
                            seed=int(current_step) * 31 + seed_offset + 99)
                        _, _fg = latent_diff_augment(
                            _fg, _fg, policy=diff_aug_policy,
                            seed=int(current_step) * 31 + seed_offset + 199)
                    d_rg, d_fg = _m_fwd(disc_for_guidance, [
                        (_rn, real_m_rat, real_m_ram, real_m_rpe),
                        (_fg, _fseg_act[0], _fseg_act[1]),
                    ])
                    g_rp, gen_gan_stat_value = _m_rp(d_rg, d_fg, True, group_flat)
                    generator_gan_loss = (
                        gen_gan_weight * g_rp.to(pred_image_dtype))
                    gen_gan_main_value = float(g_rp.detach().item())
                    # WP-14B unweighted-ratio call site: stash the LIVE
                    # resolved multiplier for THIS pair_mode's g_rp, not a
                    # value re-derived from cfg. Written only here, inside
                    # the branch that actually produced a weighted
                    # ``generator_gan_loss`` -- a d_only-phase call, a
                    # warmup-gated call, or a zero-weight call never
                    # reaches this line, so the stash cannot claim a
                    # multiplier for a loss that was never weighted.
                    # ``gen_gan_stat_value`` accompanies it unconditionally
                    # (0.0 when the stat sideband is off), letting the A7
                    # caller verify the sideband contributed nothing before
                    # treating ``generator_gan_loss`` as a pure scalar
                    # multiple of ``g_rp`` (the precondition the division
                    # below depends on).
                    self._ladd_last_gen_gan_weight = float(gen_gan_weight)
                    self._ladd_last_gen_gan_weight_mode = pair_mode
                    self._ladd_last_gen_gan_stat_value = float(
                        gen_gan_stat_value)
                finally:
                    disc_for_guidance.requires_grad_(True)
                    if _disc_was_training:
                        disc_for_guidance.train()

            logs = {
                "train/r3gan_disc_skipped": 1.0 if disc_skipped else 0.0,
                "train/r3gan_d_loss": last_d_loss,
                "train/r3gan_d_real": last_d_real,
                "train/r3gan_d_fake_detached": last_d_fake,
                "train/r3gan_r1": last_r1,
                "train/r3gan_r1_grad_sq": last_r1_grad_sq,
                "train/r3gan_r1_fired": last_r1_fired,
                "train/r3gan_r1_gamma": float(_r1_gamma),
                # A16-D1: applications of R1 in THIS (step, pair_mode) block.
                # Under the unified cadence a due step reads
                # ``gan_updates_per_step``; the pre-D1 bug read 1.
                # CAVEAT: with ``ladd_defer_disc_update`` the D-update runs
                # AFTER this logs dict is built, so this key (like every other
                # D-side key on that path) stays at its NaN init. The
                # ``r3gan_r1_mode_*`` counters below live on ``self`` and are
                # therefore defer-correct -- read those to audit the rate.
                "train/r3gan_r1_block_fires": last_r1_block_fires,
                **self._ladd_penalty_fire_logs(pair_mode),
                "train/r3gan_g_loss_raw": gen_gan_main_value,
                "train/r3gan_g_loss_weighted": gen_gan_weight * gen_gan_main_value,
                "train/r3gan_g_weight": float(gen_gan_weight),
                "train/critic_warmup_done": 1.0 if critic_warmup_done else 0.0,
                "train/ladd_n_pairs": float(n_pairs),
                # PROOF-OF-MATCHING telemetry. These keys exist ONLY on the
                # matched branch, and _compute_ladd_losses suffixes every
                # per-mode key ("_gt" for gt_vs_fake, "_gtxn" for
                # gt_transition), so seeing ``train/ladd_match_k_gt`` in a
                # run is proof the single-chunk retrieval path executed.
                "train/ladd_match_k": float(Kk),
                "train/ladd_match_pool_m": float(_M),
                "train/ladd_match_n_real": last_n_real,
                "train/ladd_match_active": 1.0,
                "train/ladd_match_n_cand": float(n_cand),
                "train/ladd_match_chunks_per_pair": float(chunks_per_pair),
                # Phase B fix 3 verification: the disc timestep actually in
                # effect on the matched branch (0 = clean, flash_t = matched
                # re-noise). Mirrors the positional branch's key.
                "train/ladd_disc_t": float(disc_t_int),
                # Phase B fix 1 verification: unique cross-ride ring reals
                # in the last _match_select draw + current ring occupancy.
                "train/ladd_ring_n": float(
                    getattr(self, "_ladd_ring_last_n", 0.0)),
                "train/ladd_ring_size": float(
                    len(getattr(self, "_ladd_real_ring", None) or [])),
                # A6 real-sample diversity telemetry (empty dict => no keys
                # emitted when gan_real_diversity_log is off). D2: falls back
                # to THIS STEP's stashed per-pair_mode draw when this
                # particular call did not itself draw (two-phase ``g_only``
                # during critic warmup / zero gen weight).
                **self._ladd_real_div_logs(pair_mode, current_step),
                "train/r3gan_d_loss_stat": last_d_loss_stat,
                "train/r3gan_g_loss_raw_stat": gen_gan_stat_value,
                "train/r3gan_stat_loss_weight": float(
                    getattr(self.model, "ladd_stat_head_loss_weight", 1.0)),
            }
            return generator_gan_loss, logs

        # Degenerate-match fallback: if _match was True but the match pool was
        # absent/too small, the matched branch above fell through and the
        # precompute guard left the noisy tensors None — the positional D-loop
        # below would None-deref. Build them now so behaviour matches the
        # pre-guard graceful fallback (only hit on an empty-match ride; the
        # normal matched path returns above and never reaches here).
        if _match_active and real_chunks_det_noisy is None:
            # LOUD (rate-limited) warning: matching was REQUESTED but the
            # pool was absent or shorter than one candidate, so this step
            # silently used positional pairing instead. Worth shouting
            # about: it changes what the disc learns AND (with
            # ladd_disc_micro_batch_groups > 1) it changes this rank's
            # backward COUNT vs a rank that matched, which is a DDP hang
            # risk. Behaviour is unchanged (still the graceful fallback);
            # only the visibility is new.
            _mf_n = getattr(self, "_ladd_match_fallback_n", 0) + 1
            self._ladd_match_fallback_n = _mf_n
            if _mf_n <= 5 or _mf_n % 100 == 0:
                import sys as _sys
                _pl_f = (0 if _match_lat is None
                         else int(_match_lat.shape[1]))
                print(
                    f"[LADD-MATCH-FALLBACK] mode={pair_mode} step="
                    f"{int(current_step)} n={_mf_n}: match requested but "
                    f"pool frames={_pl_f} < required "
                    f"{chunks_per_pair * npb} (chunks_per_pair="
                    f"{chunks_per_pair}, npb={npb}) -> POSITIONAL pairing "
                    "this step. Check that setup_sequence published "
                    "streaming_state['gt_match_latents'].",
                    file=_sys.stderr, flush=True,
                )
            real_chunks_det_noisy = _add_disc_noise(real_chunks_det)
            fake_chunks_det_noisy = _add_disc_noise(fake_chunks_det)
            if diff_aug_policy:
                real_chunks_det_noisy, fake_chunks_det_noisy = (
                    latent_diff_augment(
                        real_chunks_det_noisy,
                        fake_chunks_det_noisy,
                        policy=diff_aug_policy,
                        seed=int(current_step) * 31 + seed_offset,
                    )
                )

        # ---- A16-D1: unified R1 cadence decided ONCE for this positional
        # D-update block, BEFORE the ``gan_updates_per_step`` loop, and keyed
        # by ``pair_mode``. See ``_ladd_r1_block_due`` for the contract and
        # the fire-rate table. Default OFF => ``_r1_block_due_pos`` is None
        # and the legacy ``current_step % _r1_every_n`` expression inside the
        # loop is evaluated verbatim (byte-identical when off).
        _r1_unified_pos = bool(getattr(
            self.config, "ladd_r1_unified_cadence",
            getattr(self.model, "ladd_r1_unified_cadence", False)))
        _r1_every_n_pos = max(1, int(
            getattr(self.model, "ladd_r1_every_n_steps", 1)))
        _r1_block_due_pos = None
        if (_r1_unified_pos and n_disc_updates > 0
                and self.r3gan_optimizer is not None):
            _r1_block_due_pos = self._ladd_r1_block_due(
                pair_mode, current_step, _r1_every_n_pos)
            last_r1_block_fires = float(
                n_disc_updates if _r1_block_due_pos else 0)
        if n_disc_updates > 0 and self.r3gan_optimizer is not None:
            for _ in range(n_disc_updates):
                self.r3gan_optimizer.zero_grad(set_to_none=True)
                # ----- Batched real+fake forward (v28A OOM mitigation) -----
                # Stack real + fake into ONE disc forward; halves the
                # teacher-class forwards (2 → 1) for the D-update.
                # R1 penalty stays correct: ``d_real_logits.sum()`` only
                # depends on the real rows, so the gradient w.r.t. fake
                # rows is identically zero.
                B_pair_d = real_chunks_det_noisy.shape[0]
                # ---- flag-gated INPUT DUMP (positional branch) ----
                # Same contract as the matched-real stash above: what the
                # disc D-update scores, pre-wavelet. Main-rank only,
                # try/except-wrapped (diagnostics never crash training).
                if getattr(self, "is_main_process", False):
                    try:
                        _dbg_every = int(getattr(
                            self.config,
                            "debug_dump_scorer_inputs_every",
                            getattr(self.model,
                                    "debug_dump_scorer_inputs_every",
                                    0)) or 0)
                        # DEBT-based cadence — see the matched-real stash.
                        _dbg_last = int(getattr(
                            self, "_dbg_gan_last_fire", -10 ** 9))
                        if (_dbg_every > 0
                                and int(current_step) - _dbg_last
                                >= _dbg_every):
                            self._dbg_gan_last_fire = int(current_step)
                            self._dbg_gan_dump = {
                                "real": real_chunks_det_noisy.detach().to(
                                    device="cpu", dtype=torch.float32),
                                "fake": fake_chunks_det_noisy.detach().to(
                                    device="cpu", dtype=torch.float32),
                                "step": int(current_step),
                                "mode": f"{pair_mode}/positional",
                            }
                    except Exception:
                        pass
                combined_in = torch.cat(
                    [real_chunks_det_noisy.detach(),
                     fake_chunks_det_noisy.detach()],
                    dim=0,
                ).requires_grad_(True)
                _disc_t_d = torch.cat([t_disc, t_disc], dim=0)
                _pe_d = torch.cat([prompt_embeds_eff, prompt_embeds_eff], dim=0)
                _pp_d = (
                    torch.cat([pooled_prompt, pooled_prompt], dim=0)
                    if pooled_prompt is not None else None
                )
                _combined_cond_extra = None
                if real_action_tokens is not None:
                    _combined_cond_extra = {
                        "_action_tokens": torch.cat(
                            [real_action_tokens, fake_action_tokens], dim=0,
                        ),
                    }
                    if real_action_modulation is not None:
                        _combined_cond_extra["_action_modulation"] = torch.cat(
                            [real_action_modulation, fake_action_modulation],
                            dim=0,
                        )
                # R1 regularization. Two estimator modes
                # (``ladd_r1_mode`` on the model):
                #   "fd" (default): finite-difference stochastic
                #         estimator. ``r1_grad = (D(x+sigma*eps) -
                #         D(x)) / sigma``; ``r1 = mean(r1_grad**2)``.
                #         ONE extra disc forward, no second-order
                #         autograd graph. Cheap. Reference impl:
                #         Causal-Forcing/model/gan.py:258-271.
                #   "autograd": exact ``torch.autograd.grad(...,
                #         create_graph=True)``. Holds the second-order
                #         graph; defeats disc gradient checkpointing;
                #         ~2× the memory of "fd". Legacy.
                # Lazy R1 (``ladd_r1_every_n_steps`` > 1) skips R1 on
                # off-iters regardless of mode.
                _r1_mode = str(getattr(self.model, "ladd_r1_mode", "fd"))
                _r1_every_n = max(1, int(
                    getattr(self.model, "ladd_r1_every_n_steps", 1)
                ))
                # ---- A16 (2026-08-23) + D1 (2026-08-23): cadence -----------
                # The MATCHED D-update uses a DEBT LATCH; this positional loop
                # used a bare ``current_step % _r1_every_n``, with no latch and
                # no shared state. On any run that mixes the paths
                # (``ladd_gt_transition_match`` on + a ``[LADD-MATCH-FALLBACK]``
                # step, or several pair modes per step) R1 could DOUBLE-FIRE at
                # one step, or fire on a step the matched latch had already
                # "spent".
                #
                # A16 unified the two behind a SINGLE GLOBAL latch evaluated
                # INSIDE this loop. D1 found that this starved R1 three ways —
                # (a) iterations 2..S of the same block always saw delta==0,
                # (b) the first pair mode to run spent the shared debt and every
                # later mode's R1 was permanently zero, (c) with
                # ``ladd_defer_disc_update`` the positional loop claimed the
                # latch before the deferred MATCHED D-update ran, so the matched
                # head lost its R1. The latch is now per-``pair_mode`` and is
                # evaluated ONCE PER BLOCK (hoisted above this loop); see
                # ``_ladd_r1_block_due`` for the contract + fire-rate table.
                #
                # ``ladd_r1_unified_cadence`` stays default False =>
                # ``_r1_block_due_pos`` is None and the modulo expression below
                # is evaluated verbatim, touching no shared state
                # (byte-identical). Set it together with
                # ``ladd_r1_normalize_tokens`` on any patch-logit arm.
                if _r1_block_due_pos is not None:
                    _do_r1 = bool(_r1_block_due_pos)
                else:
                    _do_r1 = (current_step % _r1_every_n == 0)
                _r1_gamma = float(
                    getattr(self.model, "ladd_r1_gamma", 1.0)
                )
                _r1_sigma = float(
                    getattr(self.model, "ladd_r1_sigma", 0.01)
                )
                # ---- A16: token normalisation for BOTH positional R1
                # estimators. The two MATCHED estimators (inline ~:7213 and
                # micro-batched ~:5213) already honour
                # ``ladd_r1_normalize_tokens``; these two did NOT, so with
                # ``ladd_gt_transition_match=false`` EVERY step estimated
                # ‖∇_x Σ_i D_i‖² instead of ‖∇_x MEAN_i D_i‖² — a ~T²
                # overshoot (T ≈ 47k tokens on a patch-logit disc ⇒ ~1e9×),
                # which makes a γ calibrated on the matched path
                # catastrophically oversized here. ∇ MEAN = ∇ SUM / T, so
                # the normalised grad_sq is the raw one divided by T².
                # Default False => ``_r1_tok`` is 1.0 and both expressions
                # below are evaluated verbatim (byte-identical when off).
                _r1_norm_tok = bool(getattr(
                    self.config, "ladd_r1_normalize_tokens",
                    getattr(self.model,
                            "ladd_r1_normalize_tokens", False)))

                def _r1_tok_count(_logits):
                    # Token count T of the disc's per-sample logit map; 1.0
                    # when normalisation is off, and 1.0 for a scalar-output
                    # disc (where sum == mean and the two agree already).
                    if not _r1_norm_tok:
                        return 1.0
                    return float(
                        _logits.shape[1] if _logits.dim() > 1 else 1)

                self._ladd_count_penalty_fires(_do_r1, pair_mode)

                if _do_r1 and _r1_mode == "autograd":
                    combined_logits = disc_for_update(
                        x_noisy=combined_in,
                        timestep=_disc_t_d,
                        prompt_embeds=_pe_d,
                        pooled_prompt=_pp_d,
                        conditional_extra=_combined_cond_extra,
                    )
                    d_real_logits = combined_logits[:B_pair_d]
                    d_fake_logits = combined_logits[B_pair_d:]
                    grads = torch.autograd.grad(
                        d_real_logits.sum(),
                        combined_in,
                        create_graph=True,
                        retain_graph=True,
                    )[0]
                    real_grads_only = grads[:B_pair_d]
                    # Raw ‖∇_x Σ_i D_i‖² (per-sample, then averaged) —
                    # the quantity γ multiplies. Captured BEFORE the
                    # 0.5·γ scaling so wandb shows the gradient norm
                    # independent of the γ knob.
                    _r1_grad_sq_raw = (
                        real_grads_only.flatten(1).pow(2).sum(dim=1).mean()
                    )
                    # A16: ∇_x MEAN_i D_i = (∇_x Σ_i D_i) / T, so the
                    # token-normalised squared norm is the raw one over T².
                    # No-op (skipped entirely) when the flag is off.
                    _tokN = _r1_tok_count(d_real_logits)
                    if _tokN != 1.0:
                        _r1_grad_sq_raw = _r1_grad_sq_raw / (_tokN * _tokN)
                    r1 = 0.5 * _r1_gamma * _r1_grad_sq_raw
                    last_r1_grad_sq = float(_r1_grad_sq_raw.detach().item())
                    last_r1_fired = 1.0
                elif _do_r1 and _r1_mode == "fd":
                    # Finite-difference R1. Detach inputs (no second-order
                    # graph). Append the perturbed real segment to ONE
                    # batched disc forward. NOTE: reaching this branch
                    # implies ``_r1_mode == "fd"`` — the autograd+R1 case
                    # is fully handled above.
                    real_part = combined_in[:B_pair_d].detach()
                    fake_part = combined_in[B_pair_d:].detach()
                    _fd_segs = [real_part, fake_part]
                    if _do_r1:
                        eps_real = _r1_sigma * torch.randn_like(real_part)
                        _fd_segs.append(real_part + eps_real)
                    _n_seg = len(_fd_segs)
                    combined_in_fd = torch.cat(_fd_segs, dim=0)
                    _disc_t_d_fd = torch.cat([t_disc] * _n_seg, dim=0)
                    _pe_d_fd = torch.cat(
                        [prompt_embeds_eff] * _n_seg, dim=0,
                    )
                    _pp_d_fd = (
                        torch.cat([pooled_prompt] * _n_seg, dim=0)
                        if pooled_prompt is not None else None
                    )
                    _cond_extra_fd = None
                    if _combined_cond_extra is not None:
                        # The combined dict has [real, fake] sub-batches
                        # along dim 0. Append the real slice for the R1
                        # segment, matching ``_fd_segs`` order.
                        _cond_extra_fd = {}
                        for k, v in _combined_cond_extra.items():
                            real_v = v[:B_pair_d]
                            _parts = [v]
                            if _do_r1:
                                _parts.append(real_v)
                            _cond_extra_fd[k] = torch.cat(_parts, dim=0)
                    combined_logits = disc_for_update(
                        x_noisy=combined_in_fd,
                        timestep=_disc_t_d_fd,
                        prompt_embeds=_pe_d_fd,
                        pooled_prompt=_pp_d_fd,
                        conditional_extra=_cond_extra_fd,
                    )
                    d_real_logits = combined_logits[:B_pair_d]
                    d_fake_logits = combined_logits[B_pair_d:2 * B_pair_d]
                    _off = 2 * B_pair_d
                    # Finite-difference penalties — calibrated to match
                    # the autograd path: ``(γ/2) · E_x[‖∇_x Σ_i D_i‖²]``.
                    # SUM the per-token logits per sample FIRST, then take
                    # the finite difference: ``(D_sum(x+σε) − D_sum(x))/σ
                    # ≈ ∇_x D_sum · ε``; squaring + expectation gives
                    # ``‖Σ_i ∇_x D_i‖²``, the quantity γ multiplies. The
                    # 0.5 factor mirrors the autograd path so γ tunes
                    # uniformly across modes.
                    if _do_r1:
                        d_real_perturbed = combined_logits[_off:_off + B_pair_d]
                        _off += B_pair_d
                        d_real_sum = d_real_logits.sum(dim=1)
                        d_real_pert_sum = d_real_perturbed.sum(dim=1)
                        # A16: divide the summed-logit finite difference by
                        # T (``ladd_r1_normalize_tokens``) so this estimates
                        # ‖∇_x MEAN_i D_i‖², identical in scale to both
                        # matched estimators. ``_tokN == 1.0`` when the flag
                        # is off, and ``_r1_sigma * 1.0 == _r1_sigma``
                        # exactly in IEEE754 => byte-identical.
                        _tokN = _r1_tok_count(d_real_logits)
                        _r1_grad_sq_raw = (
                            (d_real_pert_sum - d_real_sum)
                            / (_r1_sigma * _tokN)
                        ).pow(2).mean()
                        r1 = 0.5 * _r1_gamma * _r1_grad_sq_raw
                        last_r1_grad_sq = float(_r1_grad_sq_raw.detach().item())
                        last_r1_fired = 1.0
                    else:
                        r1 = d_real_logits.sum() * 0.0
                else:
                    # R1 doesn't fire this iter (lazy). Detached
                    # forward for stability + memory.
                    combined_logits = disc_for_update(
                        x_noisy=combined_in.detach(),
                        timestep=_disc_t_d,
                        prompt_embeds=_pe_d,
                        pooled_prompt=_pp_d,
                        conditional_extra=_combined_cond_extra,
                    )
                    d_real_logits = combined_logits[:B_pair_d]
                    d_fake_logits = combined_logits[B_pair_d:]
                    r1 = combined_logits.sum() * 0.0  # graph-connected zero
                # Per-token RpGAN: feed the full [B, K·T'·H'·W']
                # logit tensors directly. ``rpgan_d_loss = softplus(
                # d_fake - d_real).mean()`` is elementwise, so this
                # gives each (frame, spatial-position, tap) token its
                # own per-position contribution to the disc loss
                # (and, symmetrically, its own per-position gradient
                # signal back to the disc weights). The previous
                # ``.mean(dim=1)`` collapsed positions to a single
                # scalar per sample BEFORE the softplus, which made
                # the disc see one global score per sample and
                # erased per-position discrimination.
                #
                # Visual / stat split (when ``stat_head_enabled``):
                # the disc forward appends K stat-logits at the END
                # of the per-token visual logits. With ~thousands of
                # visual tokens vs K stat-logits, lumping them into
                # one ``.mean()`` reduction would drown the stat side
                # (contribution K / (N_visual + K)). Instead we split,
                # reduce each side independently (each mean is O(1)),
                # and combine with ``ladd_stat_head_loss_weight``.
                # Default weight 1.0 → ~50/50 stat:visual at the
                # gradient level. Set weight=0 to neuter the stat side.
                _K_stat = int(getattr(self.r3gan_disc, "stat_logit_count", 0))
                _W_stat = float(getattr(
                    self.model, "ladd_stat_head_loss_weight", 1.0,
                ))
                if _K_stat > 0 and _W_stat > 0.0:
                    d_real_v = d_real_logits[:, :-_K_stat]
                    d_fake_v = d_fake_logits[:, :-_K_stat]
                    d_real_s = d_real_logits[:, -_K_stat:]
                    d_fake_s = d_fake_logits[:, -_K_stat:]
                    d_rp_visual = _d_loss_fn(d_real_v, d_fake_v)
                    d_rp_stat = _d_loss_fn(d_real_s, d_fake_s)
                    d_rp = d_rp_visual + _W_stat * d_rp_stat
                    last_d_loss_stat = float(d_rp_stat.detach().item())
                    last_d_real_stat = float(d_real_s.detach().mean().item())
                    last_d_fake_stat = float(d_fake_s.detach().mean().item())
                else:
                    d_rp = _d_loss_fn(d_real_logits, d_fake_logits)
                    last_d_loss_stat = 0.0
                    last_d_real_stat = 0.0
                    last_d_fake_stat = 0.0
                d_total = d_rp + r1
                d_total.backward()
                if self.gan_max_grad_norm and self.gan_max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.r3gan_disc.parameters()
                         if p.grad is not None],
                        self.gan_max_grad_norm,
                    )
                self.r3gan_optimizer.step()
                last_d_loss = float(d_rp.detach().item())
                last_d_real = float(d_real_logits.detach().mean().item())
                last_d_fake = float(d_fake_logits.detach().mean().item())
                last_r1 = float(r1.detach().item())
            self._mem_step_snapshot(f"ladd_{pair_mode}_b_post_d_update")

        # ----- Gen-side -----
        critic_warmup_done = (
            current_step >= int(getattr(self, "gan_critic_warmup_steps", 0))
        )
        if (
            self.gan_warmup_steps > 0
            and current_step < (
                self.gan_critic_warmup_steps + self.gan_warmup_steps
            )
            and current_step >= self.gan_critic_warmup_steps
        ):
            ramp_steps_in = current_step - self.gan_critic_warmup_steps
            t_norm = ramp_steps_in / max(1, self.gan_warmup_steps)
            ramp = self._gan_warmup_shape_apply(t_norm)
            gen_gan_weight = ramp * self.gan_loss_weight
        elif current_step >= (
            self.gan_critic_warmup_steps + self.gan_warmup_steps
        ):
            gen_gan_weight = self.gan_loss_weight
        else:
            gen_gan_weight = 0.0
        gen_gan_weight = gen_gan_weight * float(
            getattr(self.model, "ladd_disc_loss_weight", 1.0)
        )
        gen_gan_weight = self._couple_gan_weight_to_gate(gen_gan_weight)

        gen_gan_main_value = 0.0
        gen_gan_stat_value = 0.0
        if critic_warmup_done and gen_gan_weight > 0 and not skip_g:
            # Eval mode disables spectral_norm's power-iter mutation
            # of the ``_u`` / ``_sigma`` buffers during this forward.
            # Required to support multi-mode LADD (gt_vs_fake +
            # adjacent_chunks): if mode-1's gen-side forward saved
            # ``_u`` at version V, ANY subsequent forward in train
            # mode (e.g. mode-2's D-update) bumps the version and
            # the eventual outer ``generator_loss.backward()`` would
            # raise a version-counter mismatch on the saved tensor.
            # Eval mode uses the cached ``_sigma`` from the most
            # recent D-update training forward; the D-updates still
            # run in train mode so ``_sigma`` stays current.
            disc_for_guidance.requires_grad_(False)
            disc_was_training = disc_for_guidance.training
            disc_for_guidance.eval()
            try:
                real_chunks_grad_det = real_chunks_det.detach()
                fake_chunks_grad = fake_chunks_grad_tensor
                real_g = _add_disc_noise(real_chunks_grad_det)
                fake_g = _add_disc_noise(fake_chunks_grad)
                if diff_aug_policy:
                    real_g, fake_g = latent_diff_augment(
                        real_g, fake_g,
                        policy=diff_aug_policy,
                        seed=int(current_step) * 31 + seed_offset,
                    )
                # ----- Batched gen-side forward (v28A OOM mitigation) -----
                # Stack real (detached anchor) + fake (grad-on) into ONE
                # disc forward; halves the teacher-class forwards (2 → 1)
                # on the gen side. d_fake_g still gets its gen-side
                # gradient via the fake rows.
                B_pair_g = real_g.shape[0]
                combined_g = torch.cat([real_g.detach(), fake_g], dim=0)
                _disc_t_g = torch.cat([t_disc, t_disc], dim=0)
                _pe_g = torch.cat([prompt_embeds_eff, prompt_embeds_eff], dim=0)
                _pp_g = (
                    torch.cat([pooled_prompt, pooled_prompt], dim=0)
                    if pooled_prompt is not None else None
                )
                _gen_combined_cond_extra = None
                if real_action_tokens is not None:
                    _gen_combined_cond_extra = {
                        "_action_tokens": torch.cat(
                            [real_action_tokens, fake_action_tokens], dim=0,
                        ),
                    }
                    if real_action_modulation is not None:
                        _gen_combined_cond_extra["_action_modulation"] = (
                            torch.cat(
                                [real_action_modulation, fake_action_modulation],
                                dim=0,
                            )
                        )
                combined_g_logits = disc_for_guidance(
                    x_noisy=combined_g,
                    timestep=_disc_t_g,
                    prompt_embeds=_pe_g,
                    pooled_prompt=_pp_g,
                    conditional_extra=_gen_combined_cond_extra,
                )
                d_real_g = combined_g_logits[:B_pair_g]
                d_fake_g = combined_g_logits[B_pair_g:]
                # f-distill stash (dmd_fkl_mix): mean VISUAL fake logit of
                # the STUDENT's gen-phase scoring = the density-ratio proxy
                # forward-KL needs (review D1: the aux-teacher block scored
                # the wrong distribution; D4: stat-head columns excluded;
                # D3: finiteness-guarded -- a NaN here would poison the
                # cross-rank mean on every rank).
                import math as _math
                _Kst = int(getattr(disc_for_guidance, "stat_logit_count", 0))
                _dfv = (d_fake_g[:, :-_Kst] if _Kst > 0 else d_fake_g)
                _drv = (d_real_g[:, :-_Kst] if _Kst > 0 else d_real_g)
                # Relativistic GAP, not the raw fake mean: per-token means
                # concentrate (CLT) so the raw logit is rank-identical and
                # exp(centered) pinned at 1 (measured in the fkl smoke).
                # gap = E[d_fake] - E[d_real] is a real per-sample signal.
                _fkl_v = float(
                    (_dfv.detach().mean() - _drv.detach().mean()).item())
                if _math.isfinite(_fkl_v):
                    self.model._fkl_fake_logit = _fkl_v
                # Per-token RpGAN on the gen side (mirrors the D-side
                # change above). ``rpgan_g_loss = softplus(d_real -
                # d_fake).mean()`` is elementwise — feeding the full
                # [B, K·T'·H'·W'] logit tensors gives each
                # (frame, spatial-position, tap) token its own
                # ``-sigmoid(d_real_i - d_fake_i) / N`` upstream
                # gradient. Combined with the 2D-conv heads, this
                # delivers per-frame-per-spatial-position-per-tap
                # signal back to the generator instead of a single
                # uniformly-magnitude'd scalar gradient per sample.
                #
                # Visual / stat split — same logic as the D-side: the
                # stat head's K appended logits get their own RpGAN
                # mean reduction, then sum with ``ladd_stat_head_loss
                # _weight``. Default 1.0 → stat side at parity with
                # visual side at the gen-gradient level. Without this
                # split, the stat side's contribution to the gen
                # gradient would be K / (N_visual + K) ≈ 0.
                _K_stat_g = int(getattr(self.r3gan_disc, "stat_logit_count", 0))
                _W_stat_g = float(getattr(
                    self.model, "ladd_stat_head_loss_weight", 1.0,
                ))
                if _K_stat_g > 0 and _W_stat_g > 0.0:
                    d_real_g_v = d_real_g[:, :-_K_stat_g]
                    d_fake_g_v = d_fake_g[:, :-_K_stat_g]
                    d_real_g_s = d_real_g[:, -_K_stat_g:]
                    d_fake_g_s = d_fake_g[:, -_K_stat_g:]
                    g_rp_visual = _g_loss_fn(d_real_g_v.detach(), d_fake_g_v)
                    g_rp_stat = _g_loss_fn(d_real_g_s.detach(), d_fake_g_s)
                    g_rp = g_rp_visual + _W_stat_g * g_rp_stat
                    gen_gan_stat_value = float(g_rp_stat.detach().item())
                else:
                    g_rp = _g_loss_fn(d_real_g.detach(), d_fake_g)
                    gen_gan_stat_value = 0.0
                generator_gan_loss = (
                    gen_gan_weight * g_rp.to(pred_image_dtype)
                )
                gen_gan_main_value = float(g_rp.detach().item())
                # WP-14B unweighted-ratio call site (mirrors the matched
                # branch above -- see its comment for the full rationale).
                self._ladd_last_gen_gan_weight = float(gen_gan_weight)
                self._ladd_last_gen_gan_weight_mode = pair_mode
                self._ladd_last_gen_gan_stat_value = float(gen_gan_stat_value)
            finally:
                disc_for_guidance.requires_grad_(True)
                if disc_was_training:
                    disc_for_guidance.train()
        else:
            generator_gan_loss = zero
        self._mem_step_snapshot(f"ladd_{pair_mode}_c_post_g_forward")

        # ---- A6 (2026-08-23): real-diversity telemetry, POSITIONAL path ---
        # The original A6 block lives inside the matched ``_match_select``
        # closure, so with ``ladd_gt_transition_match=false`` — the exact
        # configuration directive R3 prescribes — it emitted ZERO keys and
        # the absence was indistinguishable from "diversity is fine". Cover
        # the positional path here with the same key names, plus an explicit
        # ``gan_real_div_positional`` marker so a reader can tell which
        # estimator produced the numbers (they mean different things: the
        # matched draw can pull cross-ride ring reals, the positional draw
        # is by construction ONE ride per step, so ``unique_rides`` is 1 and
        # the only real variable is how many distinct source windows the
        # D sees). ``gan_real_div_unavailable=1.0`` is emitted if the source
        # bookkeeping cannot be reconstructed, so the gap is never silent.
        # Gated on ``gan_real_diversity_log`` (default False) => no keys and
        # no work when off: byte-identical.
        _div_pos = {}
        if bool(getattr(self, "gan_real_diversity_log", False)):
            try:
                _cur_ride_p = ""
                _ss_p = getattr(self.model, "streaming_state", None)
                if isinstance(_ss_p, dict):
                    _cur_ride_p = str(_ss_p.get("zarr_path", "") or "")
                _slots_p = int(real_chunks_det.shape[0])
                # Rows are chunk-major / batch-minor (torch.cat over pairs of
                # [B, ...] slabs), so row r came from source window
                # ``real_positions[r // B]`` of the current ride.
                _bs_p = max(1, _slots_p // max(1, n_pairs))
                _src_p = []
                for _r_i in range(_slots_p):
                    _k_i = _r_i // _bs_p
                    _pos_i = (real_positions[_k_i]
                              if _k_i < len(real_positions) else -1)
                    _src_p.append((_cur_ride_p, _r_i % _bs_p, int(_pos_i)))
                _uw_p = len(set(_src_p))
                _div_pos = {
                    "train/gan_real_div_positional": 1.0,
                    "train/gan_real_slots_total": float(_slots_p),
                    "train/gan_real_unique_rows": float(_uw_p),
                    "train/gan_real_unique_rides": float(
                        len({_t[0] for _t in _src_p})),
                    "train/gan_real_unique_windows": float(_uw_p),
                    "train/gan_real_repeat_rate": (
                        1.0 - (_uw_p / float(_slots_p))
                        if _slots_p > 0 else 0.0),
                    # No cross-ride replay ring on this path, by construction.
                    "train/gan_real_ring_slot_frac": 0.0,
                    "train/gan_real_src_untagged": 0.0,
                }
            except Exception:
                _div_pos = {
                    "train/gan_real_div_positional": 1.0,
                    "train/gan_real_div_unavailable": 1.0,
                }

        logs = {
            "train/r3gan_disc_skipped": 1.0 if disc_skipped else 0.0,
            "train/r3gan_d_loss": last_d_loss,
            "train/r3gan_d_real": last_d_real,
            "train/r3gan_d_fake_detached": last_d_fake,
            # γ-weighted R1 penalty actually added to d_total. NOTE: on
            # lazy-skipped iters this is 0.0 (the graph-connected zero);
            # the trace will look almost-flat with γ=0.001. Use the
            # ``r3gan_r1_grad_sq`` key below to see the underlying
            # gradient norm regardless of γ + lazy schedule.
            "train/r3gan_r1": last_r1,
            # Raw squared-gradient estimate — γ-free, lazy-aware (NaN on
            # iters where R1 didn't actually fire so wandb plots gaps).
            # A16: this is ‖∇_x Σ_i D_i‖² by default and
            # ‖∇_x MEAN_i D_i‖² when ``ladd_r1_normalize_tokens`` is set —
            # the SAME convention as both matched estimators, so the four
            # R1 traces are now directly comparable.
            # This is the right knob to watch when calibrating
            # ``ladd_r1_gamma``: if it stays ~0, R1 has nothing to bite
            # on; if it climbs into the thousands, the disc is becoming
            # sharp and γ should go up.
            "train/r3gan_r1_grad_sq": last_r1_grad_sq,
            # 1.0 iff R1 was actually computed this disc-update iter
            # (``current_step % ladd_r1_every_n_steps == 0``, or the shared
            # debt latch when ``ladd_r1_unified_cadence`` is on, AND the
            # disc isn't in warmup). Lets the user filter the
            # ``r3gan_r1`` / ``r3gan_r1_grad_sq`` traces to only the
            # iters where the values are meaningful.
            "train/r3gan_r1_fired": last_r1_fired,
            "train/r3gan_r1_gamma": float(_r1_gamma) if "_r1_gamma" in locals() else float(getattr(self.model, "ladd_r1_gamma", 1.0)),
            # A16-D1: R1 applications in THIS (step, pair_mode) block.
            "train/r3gan_r1_block_fires": last_r1_block_fires,
            **self._ladd_penalty_fire_logs(pair_mode),
            "train/r3gan_g_loss_raw": gen_gan_main_value,
            "train/r3gan_g_loss_weighted": (
                gen_gan_weight * gen_gan_main_value
            ),
            "train/r3gan_g_weight": float(gen_gan_weight),
            "train/critic_warmup_done": 1.0 if critic_warmup_done else 0.0,
            "train/ladd_n_pairs": float(n_pairs),
            "train/ladd_disc_t": float(disc_t_int),
            # Stat-head diagnostics (0.0 when stat head is off or
            # ``ladd_stat_head_loss_weight=0``). The "_stat" suffix lets
            # the user compare stat-side and visual-side dynamics
            # independently in wandb.
            "train/r3gan_d_loss_stat": last_d_loss_stat,
            "train/r3gan_d_real_stat": last_d_real_stat,
            "train/r3gan_d_fake_detached_stat": last_d_fake_stat,
            "train/r3gan_g_loss_raw_stat": gen_gan_stat_value,
            "train/r3gan_stat_loss_weight": float(
                getattr(self.model, "ladd_stat_head_loss_weight", 1.0)
            ),
            # A6: positional-path real-diversity telemetry (empty dict =>
            # no keys emitted when gan_real_diversity_log is off).
            **_div_pos,
        }
        return generator_gan_loss, logs

    def _compute_r3gan_losses(
        self,
        pred_image: torch.Tensor,
        gt_latents_window: torch.Tensor,
        current_step: int,
        flash_dmd_gan_x0: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Run a D-update on (real, fake) and return the G-side RpGAN
        term (graph-carrying).

        Args:
            pred_image: ``[B, F, C, H, W]`` student rollout pred (graph
                attached). The G-side branch uses this as ``fake``;
                the D-side branch detaches it.
            gt_latents_window: ``[B, F, C, H, W]`` GT latent slice
                covering the same frame indices as ``pred_image``.
                Detached for the D-update; never used for the
                G-side path (G never sees real samples directly in
                R3GAN — only D's pairwise score).
            current_step: training step (for G-side warmup ramp).
            flash_dmd_gan_x0: optional graph-on output of the Flash-DMD
                t=flash_dmd_gan_t gen forward. When provided, the G-side
                path uses this as the fake (so the GAN supervises only
                the gen's near-clean texture-refinement behavior).

        Returns:
            ``(generator_gan_loss, logs)`` — ``generator_gan_loss``
            is graph-carrying; ``logs`` is a flat ``str -> float``
            dict for wandb.
        """
        device = pred_image.device
        zero = torch.zeros((), device=device, dtype=torch.float32)
        if not self.gan_enabled or self.r3gan_disc is None:
            return zero, {}

        # Dispatch to the LADD adjacent-chunk flow when configured.
        if self.gan_backbone == "ladd_teacher_feat":
            return self._compute_ladd_losses(
                pred_image=pred_image,
                gt_latents_window=gt_latents_window,
                current_step=current_step,
                flash_dmd_gan_x0=flash_dmd_gan_x0,
            )

    # ------------------------------------------------------------------
    # SC-DMD warmup helper.
    # ------------------------------------------------------------------
    def _sc_dmd_current_weight(self, current_step: int) -> float:
        """Linear-ramp the SC-DMD weight from 0 → ``sc_dmd_loss_weight``
        over the first ``sc_dmd_warmup_steps`` training steps.

        The SC defect is a SECOND-ORDER consistency regularizer (the
        generator's velocity field appears in both the direct and
        composed paths), so its initial gradient signal can be very
        large at random init and destabilize early DMD training. The
        ramp keeps SC's contribution sub-DMD until the student is
        producing meaningful x0_hat predictions, then phases SC in
        gradually. Mirrors the ``z_guidance_warmup_steps`` recipe used
        for action-critic z-guidance.

        With ``sc_dmd_warmup_steps == 0`` (default) the weight is
        applied at full strength from step 0.
        """
        if not self.sc_dmd_enabled:
            return 0.0
        if self.sc_dmd_warmup_steps <= 0:
            return self.sc_dmd_loss_weight
        if current_step >= self.sc_dmd_warmup_steps:
            return self.sc_dmd_loss_weight
        ramp = float(current_step) / float(max(1, self.sc_dmd_warmup_steps))
        return ramp * self.sc_dmd_loss_weight

    # ------------------------------------------------------------------
    # Causal-CD warmup helper + per-ride loss fold-in.
    # ------------------------------------------------------------------
    def _cd_loss_current_weight(self, current_step: int) -> float:
        """Linear-ramp the CD weight from 0 → ``cd_loss_weight`` over the
        first ``cd_loss_warmup_steps`` steps (0 default = full strength
        from step 0). Same rationale as the SC-DMD ramp: hold the
        consistency regularizer sub-DMD until the student produces
        meaningful x0 predictions.
        """
        if not self.cd_loss_enabled:
            return 0.0
        if self.cd_loss_warmup_steps <= 0:
            return self.cd_loss_weight
        if current_step >= self.cd_loss_warmup_steps:
            return self.cd_loss_weight
        ramp = float(current_step) / float(max(1, self.cd_loss_warmup_steps))
        return ramp * self.cd_loss_weight

    def _maybe_add_cd_loss(
        self,
        generator_loss: torch.Tensor,
        conditional_dict: dict,
        clean_latent: torch.Tensor,
        seed_frames: int,
        out: dict,
    ) -> torch.Tensor:
        """Compute the weighted causal-CD loss (once per ride) and fold it
        into ``generator_loss``. No-op (returns input) when CD is off.
        Mirrors the SC-DMD fold-in; logs raw/weighted/effective-weight.
        """
        if not self.cd_loss_enabled:
            return generator_loss
        cd_loss_raw, cd_logs = self.model.cd_loss(
            conditional_dict=conditional_dict,
            clean_latent=clean_latent,
            seed_frames=int(seed_frames),
        )
        cd_weight = self._cd_loss_current_weight(int(self.step))
        weighted_cd = cd_loss_raw * cd_weight
        generator_loss = generator_loss + weighted_cd
        out.update(cd_logs)
        out["cd_loss_weight_effective"] = float(cd_weight)
        out["cd_loss_weighted"] = float(weighted_cd.detach().item())
        return generator_loss

    # ------------------------------------------------------------------
    # Periodic pred_image -> wandb.Video sampling.
    # ------------------------------------------------------------------
    def _video_sample_due(self, current_step: int) -> bool:
        """Return True iff this step should produce a sample video.

        Three gates can fire it:
          (1) explicit ``sample_at_steps`` membership (uses ``>=`` so a
              missed gen-step rolls forward to the next eligible iter),
          (2) ``sample_interval > 0`` and ``current_step`` is a positive
              multiple thereof, OR
          (3) ``self._video_sample_due_bit`` is set — meaning a previous
              iter crossed the cadence boundary while no rollout was
              available (critic-only iter under
              dfake_gen_update_ratio>1) and stashed the request for the
              next eligible gen iter to consume.
        All require main rank and at least one viable sink (wandb
        enabled OR ``vis_save_local`` true). ``current_step <= 0`` is
        skipped because no rollout has happened yet at iter 0 entry
        (the first eligible step is ``current_step == 1``).
        """
        if not self.is_main_process:
            return False
        if not (
            (_HAS_WANDB and getattr(self, "wandb_enabled", False))
            or self.vis_save_local
        ):
            return False
        if current_step <= 0:
            return False
        if (
            self._sample_at_steps_pending
            and current_step >= self._sample_at_steps_pending[0]
        ):
            return True
        if self._video_sample_due_bit:
            return True
        if self.sample_interval <= 0:
            return False
        return (current_step % self.sample_interval) == 0

    @torch.no_grad()
    def _maybe_dump_debug_inputs(self) -> None:
        """Flag-gated (``debug_dump_scorer_inputs_every``) side-by-side
        decode of what the DMD scorers and the GAN disc actually saw.

        Consumes the two stashes written on firing steps:
          * ``self.model._dbg_scorer_dump`` — 42f scoring inputs
            (noisy_x / clean_x / gt_target latents), written in
            ``compute_generator_loss_streaming``.
          * ``self._dbg_gan_dump`` — the disc D-update's real / fake
            transition-pair latent slabs (pre-wavelet), written in
            ``_ladd_run_pair_mode``.

        Decodes each slab through the SAME VAE pathway as the sample
        videos and writes ONE grid mp4 to
        ``samples/dbg_inputs_step<NNN>.mp4``. Row order (top→bottom),
        labelled by an 8px left-border colour:
          1. RED     dmd noisy_x   (scorer noisy-half content: GT ctx +
                                    rolled student chunks, pre-noise)
          2. GREEN   dmd clean_x   (the matched+drifted GT clean half)
          3. BLUE    dmd gt_target (GT at the student chunk positions)
          4. YELLOW  GAN real pairs (up to 3 slabs hstacked)
          5. MAGENTA GAN fake pairs (up to 3 slabs hstacked)
        Shorter rows are padded (freeze-frame in time, black in width).
        Main-rank only; best-effort — never raises into training.
        """
        sd = getattr(self.model, "_dbg_scorer_dump", None)
        gd = getattr(self, "_dbg_gan_dump", None)
        if not sd and not gd:
            return
        # Clear FIRST so a decode failure cannot re-fire forever.
        self.model._dbg_scorer_dump = None
        self._dbg_gan_dump = None
        vae = getattr(self.model, "vae", None)
        if vae is None:
            return
        step = None
        for _d in (sd, gd):
            if isinstance(_d, dict) and "step" in _d:
                step = int(_d["step"])
                break
        if step is None:
            step = int(self.step)

        def _dec(lat: Optional[torch.Tensor]) -> Optional[np.ndarray]:
            # [B,F,C,H,W] cpu latent -> [T,H,W,3] uint8 (first batch elem)
            if lat is None or not torch.is_tensor(lat) or lat.dim() != 5:
                return None
            with torch.no_grad():
                x = lat[0:1].to(device=self.device, dtype=torch.float32)
                px = vae.decode_to_pixel(x, seed_first=True)
                v = (0.5 * (px.float() + 1.0)).clamp(0.0, 1.0)
                arr = (v[0].detach().cpu().numpy() * 255.0).astype(np.uint8)
            if arr.ndim != 4:
                return None
            if arr.shape[-1] != 3:
                arr = arr.transpose(0, 2, 3, 1)
            return arr

        def _dec_pairs(t: Optional[torch.Tensor],
                       max_pairs: int = 3) -> Optional[np.ndarray]:
            # [N,F,C,H,W] -> hstack of up to max_pairs decoded slabs
            if t is None or not torch.is_tensor(t) or t.dim() != 5:
                return None
            outs = []
            for i in range(min(int(t.shape[0]), max_pairs)):
                a = _dec(t[i:i + 1])
                if a is not None:
                    outs.append(a)
            if not outs:
                return None
            tmin = min(a.shape[0] for a in outs)
            return np.concatenate([a[:tmin] for a in outs], axis=2)

        rows = []   # (name, [T,H,W,3], border_rgb)
        if isinstance(sd, dict):
            rows.append(("dmd_noisy_x", _dec(sd.get("noisy_x")),
                         (255, 40, 40)))
            rows.append(("dmd_clean_x", _dec(sd.get("clean_x")),
                         (40, 255, 40)))
            rows.append(("dmd_gt_target", _dec(sd.get("gt_target")),
                         (40, 120, 255)))
        if isinstance(gd, dict):
            rows.append(("gan_real_pairs", _dec_pairs(gd.get("real")),
                         (255, 255, 40)))
            rows.append(("gan_fake_pairs", _dec_pairs(gd.get("fake")),
                         (255, 40, 255)))
        rows = [(n, a, c) for (n, a, c) in rows
                if a is not None and a.size > 0]
        if not rows:
            return
        t_max = max(a.shape[0] for _, a, _ in rows)
        h_max = max(a.shape[1] for _, a, _ in rows)
        w_max = max(a.shape[2] for _, a, _ in rows)
        bordered = []
        for _n, a, c in rows:
            t_a, h_a, w_a = a.shape[0], a.shape[1], a.shape[2]
            if t_a < t_max:      # freeze-frame pad in time
                a = np.concatenate(
                    [a, np.repeat(a[-1:], t_max - t_a, axis=0)], axis=0)
            if h_a < h_max:      # black pad in height
                a = np.concatenate(
                    [a, np.zeros((t_max, h_max - h_a, w_a, 3),
                                 dtype=np.uint8)], axis=1)
            if w_a < w_max:      # black pad in width
                a = np.concatenate(
                    [a, np.zeros((t_max, h_max, w_max - w_a, 3),
                                 dtype=np.uint8)], axis=2)
            border = np.zeros((t_max, h_max, 8, 3), dtype=np.uint8)
            border[..., 0] = c[0]
            border[..., 1] = c[1]
            border[..., 2] = c[2]
            bordered.append(np.concatenate([border, a], axis=2))
        grid = np.concatenate(bordered, axis=1)
        mp4_bytes = _frames_to_mp4_bytes(grid, fps=float(self.sample_fps))
        if mp4_bytes is None:
            logging.warning(
                "[DBG-INPUTS] ffmpeg encode failed at step=%d "
                "(grid=%s)", step, grid.shape)
            return
        samples_dir = Path(self.log_dir) / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        out_path = samples_dir / f"dbg_inputs_step{step:04d}.mp4"
        with open(out_path, "wb") as fh:
            fh.write(mp4_bytes)
        logging.info(
            "[DBG-INPUTS] wrote %s (rows=%s, gan_mode=%s)",
            out_path, [r[0] for r in rows],
            gd.get("mode") if isinstance(gd, dict) else None)

    def _log_pred_image_video(
        self,
        pred_image: torch.Tensor,
        step: int,
        name: str = "pred_image",
        caption_suffix: str = "",
        action_overlay: Optional[torch.Tensor] = None,
        index_overlay: Optional[Tuple[int, int, int]] = None,
        cap_frames: bool = True,
    ) -> None:
        """Decode ``pred_image`` (a student rollout in latent space) to
        pixel mp4 bytes and upload to wandb under ``sample/<name>``.

        ``name`` controls both the wandb key and the local filename
        suffix: ``step_NNN.mp4`` for the default ``pred_image``, else
        ``step_NNN_<name>.mp4``. ``caption_suffix`` is appended to the
        wandb caption.

        Cheap: re-uses the generator's already-loaded ``WanVAEWrapper``
        (no second VAE copy), runs under ``no_grad``, encodes mp4 in
        memory via ffmpeg subprocess. Best-effort: any failure is
        warned and never raises.

        Only the main rank ever enters this function (the call site
        gates on ``is_main_process``); other ranks are not synchronised
        with this work. There is no NCCL collective inside.

        Caller contract: ``pred_image`` must be a ``[B, F, C, H, W]``
        latent tensor on the GPU. We slice to the first batch element
        and (optionally) the trailing ``sample_max_frames`` frames.

        ``action_overlay``: optional ``[B, F_lat, A]`` (or ``[F_lat, A]``)
        per-latent action stream. When supplied, two horizontal bars
        are rendered along the bottom of every video frame — one per
        action dim — magnitude proportional to ``|z|`` (red for ``z>=0``,
        blue for ``z<0``). The dataset enforces one z per chunk
        (``num_frame_per_block`` latents share the same z), so the
        overlay is constant within each chunk and steps at chunk
        boundaries. Used only for ``clean_x_real`` to cross-check the
        bidir scorer's clean-half action conditioning against the
        decoded pixel content.

        ``index_overlay``: optional
        ``(zarr_lat_lo, motion_chunk_offset, npb)`` tuple. When supplied,
        per-frame text "zarr_lat=N motion=M" is rendered top-left of
        each video frame so the displayed content can be cross-checked
        against the dataset (which exact zarr latent index this frame
        corresponds to, and which motion.npy chunk encodes its motion).
        Used only for ``clean_x_real`` (the only view that traces back
        to a specific ride window). For video frame ``t`` (after VAE
        4× temporal upsampling): zarr_lat = zarr_lat_lo + (t // 4),
        motion_chunk = motion_chunk_offset + (zarr_lat // npb).
        """
        if pred_image is None or pred_image.numel() == 0:
            return
        vae = getattr(self.model, "vae", None)
        if vae is None:
            if not self._video_logger_warned_no_vae:
                logging.warning(
                    "[ActionForcing] sample_interval set but "
                    "self.model.vae is None; sample videos disabled."
                )
                self._video_logger_warned_no_vae = True
            return
        try:
            latents = pred_image.detach().to(torch.float32)
            if latents.dim() != 5:
                return
            if cap_frames and self.sample_max_frames > 0:
                F_total = latents.shape[1]
                if F_total > self.sample_max_frames:
                    latents = latents[:, -self.sample_max_frames:]
            lat = latents[0:1]
            # Self-seeding decode (``seed_first``): each logged clip
            # (pred_image / pred_real / pred_fake / clean_x_*) is an
            # INDEPENDENT window, so the old ``use_cache=True`` path
            # left a ghost of the previously-rendered clip in frame 0
            # (the warm ``cached_decode`` feat_map is never cleared
            # between these back-to-back renders). ``seed_first`` clears
            # the cache and seeds it with this clip's OWN first latent
            # frame, so frame 0 is faithful — no cross-clip ghost and
            # no plain-decode init-frame brightness anomaly.
            pixels = vae.decode_to_pixel(lat, seed_first=True)
            video = (0.5 * (pixels.float() + 1.0)).clamp(0.0, 1.0)
            vid_np = (video[0].cpu().numpy() * 255.0).astype(np.uint8)
            if vid_np.ndim != 4:
                return
            if vid_np.shape[-1] != 3:
                vid_np = vid_np.transpose(0, 2, 3, 1)
        except Exception as exc:
            logging.warning(
                "[ActionForcing] sample-video VAE decode failed at "
                "step=%d: %s", step, exc,
            )
            return

        # Render the action overlay (best-effort, never raises). The
        # ``index_overlay`` indices (when supplied) are drawn in the
        # same pass — top-left "zarr_lat=N motion=M" text per frame so
        # we don't iterate the video twice.
        if action_overlay is not None:
            try:
                npb = int(getattr(self.config, "num_frame_per_block", 3))
                _zarr_lo = None
                _mco = None
                _zarr_name = ""
                if index_overlay is not None:
                    # 4-tuple from the trainer call site:
                    # (zarr_lat_lo, motion_chunk_offset, npb, zarr_name).
                    # Tolerate the legacy 3-tuple shape too.
                    if len(index_overlay) == 4:
                        _zarr_lo, _mco, _, _zarr_name = index_overlay
                    else:
                        _zarr_lo, _mco, _ = index_overlay
                self._draw_action_overlay(
                    vid_np, action_overlay, frames_per_latent=4, npb=npb,
                    zarr_lat_lo=_zarr_lo, motion_chunk_offset=_mco,
                    zarr_name=_zarr_name,
                )
            except Exception as exc:
                logging.warning(
                    "[ActionForcing] action overlay failed at step=%d "
                    "(name=%s): %s", step, name, exc,
                )

        mp4_bytes = _frames_to_mp4_bytes(
            vid_np, fps=float(self.sample_fps),
        )
        if mp4_bytes is None:
            logging.warning(
                "[ActionForcing] sample-video ffmpeg encode failed at "
                "step=%d (frames=%d)", step, vid_np.shape[0],
            )
            return

        if self.vis_save_local:
            try:
                samples_dir = Path(self.log_dir) / "samples"
                samples_dir.mkdir(parents=True, exist_ok=True)
                _suffix = "" if name == "pred_image" else f"_{name}"
                out_path = samples_dir / f"step_{int(step):07d}{_suffix}.mp4"
                with open(out_path, "wb") as fh:
                    fh.write(mp4_bytes)
                logging.info(
                    "[ActionForcing] Saved sample video to %s "
                    "(frames=%d, fps=%d)",
                    out_path, vid_np.shape[0], int(self.sample_fps),
                )
            except Exception as exc:
                logging.warning(
                    "[ActionForcing] sample-video local save failed at "
                    "step=%d: %s", step, exc,
                )

        if not (_HAS_WANDB and getattr(self, "wandb_enabled", False)):
            return

        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".mp4", delete=False,
            ) as fh:
                fh.write(mp4_bytes)
                tmp_path = fh.name
            _caption = f"step {step}"
            if caption_suffix:
                _caption = f"{_caption} {caption_suffix}"
            wandb.log(
                {
                    f"sample/{name}": wandb.Video(
                        tmp_path,
                        fps=int(self.sample_fps),
                        format="mp4",
                        caption=_caption,
                    ),
                    "sample/step": int(step),
                    f"sample/{name}_num_frames": int(vid_np.shape[0]),
                },
                step=step,
            )
            logging.info(
                "[ActionForcing] Logged sample/%s video at step=%d "
                "(frames=%d, fps=%d)",
                name, step, vid_np.shape[0], int(self.sample_fps),
            )
        except Exception as exc:
            logging.warning(
                "[ActionForcing] sample-video wandb upload failed at "
                "step=%d: %s", step, exc,
            )
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    @torch.no_grad()
    def _build_holdout_eval_dataset(self):
        """Lazily build a small ZarrRideDataset over ``holdout_eval_root``,
        used to SEED the pred_image_7_chunk eval from rides the model never
        trains on (true held-out generalization eval). Built with NO cache
        (a 20-ride scan is fast) so all ranks can build concurrently with no
        manifest-write race. Returns None when holdout_eval_root is unset."""
        if getattr(self, "_holdout_eval_dataset_built", False):
            return self._holdout_eval_dataset
        self._holdout_eval_dataset_built = True
        self._holdout_eval_dataset = None
        root = getattr(self.config, "holdout_eval_root", None)
        if not root:
            return None
        from utils.zarr_dataset import ZarrRideDataset
        cfg = self.config
        try:
            self._holdout_eval_dataset = ZarrRideDataset(
                encoded_root=str(root),
                caption_root=str(cfg.caption_root),
                motion_root=str(cfg.motion_root),
                ss_vae_checkpoint=str(cfg.ss_vae_checkpoint),
                min_ride_frames=int(getattr(cfg, "min_ride_frames", 64)),
                device="cpu",
                max_rides=None,
                sort_by_length=None,
                cache_path=None,  # no shared cache -> no cross-rank write race
            )
            if self.is_main_process:
                logging.info(
                    "[holdout-eval] 7chunk eval seeds from %d held-out rides "
                    "in %s", len(self._holdout_eval_dataset), str(root),
                )
        except Exception as e:  # never let eval-setup kill training
            logging.warning(
                "[holdout-eval] failed to build holdout dataset from %s: %r "
                "-> falling back to training-ride 7chunk seed.", str(root), e,
            )
            self._holdout_eval_dataset = None
        return self._holdout_eval_dataset

    def _holdout_eval_ride_for_step(self, step: int):
        """Return a {latents, z_actions, prompt_embeds} 14-chunk slice from
        the holdout set to seed the 7chunk eval, or None. DETERMINISTIC per
        step (same ride on every rank -> DDP-balanced + runs comparable).
        ``holdout_eval_mode``: 'cycle' (default) rotates a different held-out
        ride per eval step; 'fixed' always uses ride 0 (watch one scene)."""
        ds = self._build_holdout_eval_dataset()
        if ds is None or len(ds) == 0:
            return None
        npb = int(getattr(self.config, "num_frame_per_block", 3))
        need = 14 * npb
        mode = str(getattr(self.config, "holdout_eval_mode", "cycle")).lower()
        interval = max(1, int(getattr(self, "sample_interval", 1)))
        ordinal = int(step) // interval
        n = len(ds)
        for off in range(n):  # skip any too-short ride (rare; all >=69 frames)
            idx = 0 if mode == "fixed" else (ordinal + off) % n
            meta = ds[idx]
            if int(meta.get("n_latent_frames", 0)) < need:
                continue
            ride = _load_ride_tensors(
                ds, meta, self.device, self.dtype,
                action_dims=self.action_dims, max_frames=need,
            )
            if ride is None or int(ride["latents"].shape[1]) < need:
                continue
            return {
                "latents": ride["latents"][:, :need].contiguous(),
                "z_actions": ride["z_actions"][:, :need].contiguous(),
                "prompt_embeds": ride["prompt_embeds"],
            }
        return None

    def _log_pred_image_7chunk_sample(self, step: int) -> None:
        """Eval rollout: seed the causal student with 7 GT-context chunks
        (21 latent frames) and roll out 7 more chunks (21 frames), then
        log the full 14-chunk (42-frame) video at ``sample_fps`` under
        ``sample/pred_image_7_chunk``. GT actions drive the whole window.

        Runs on ALL ranks (the underlying ``inference_with_trajectory``
        does a cross-rank exit-flag broadcast, so every rank must call it
        to stay collective-balanced) but only the main rank decodes and
        uploads its own ride's video. A readiness MIN-reduce guarantees
        all ranks agree to run-or-skip together, so the broadcast never
        deadlocks. The training gen+critic backward for this step is
        already complete and ``max_rolls_per_ride=1`` re-seeds next step,
        so transiently overwriting the inference KV cache here is safe.
        """
        rd = getattr(self, "_sample_7chunk_ride", None)
        ready_local = 1 if rd is not None else 0
        if dist.is_initialized() and dist.get_world_size() > 1:
            flag = torch.tensor(
                [ready_local], device=self.device, dtype=torch.long,
            )
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            if int(flag.item()) == 0:
                if self.is_main_process:
                    logging.info(
                        "[ActionForcing] 7chunk SKIP at step=%d: a rank had "
                        "no stashed ride (local_ready=%d).",
                        int(step), ready_local,
                    )
                return
        elif not ready_local:
            return
        if self.is_main_process:
            logging.info(
                "[ActionForcing] 7chunk RUN at step=%d: rolling 7 GT ctx + "
                "7 student chunks.", int(step),
            )

        npb = int(getattr(self.config, "num_frame_per_block", 3))
        seed_frames = 7 * npb
        roll_frames = 7 * npb
        gt = rd["latents"].to(device=self.device, dtype=self.dtype)
        act = rd["z_actions"].to(device=self.device)
        prompt = rd["prompt_embeds"].to(device=self.device)
        seed = gt[:, :seed_frames]
        B, _, C, H, W = gt.shape
        noise = torch.randn(
            B, roll_frames, C, H, W, device=self.device, dtype=self.dtype,
        )
        # cond must cover seed (cf) + rollout frames, seed first.
        cond, _uncond = self.model.build_action_conditional(
            prompt_embeds=prompt,
            gt_actions=act[:, : seed_frames + roll_frames],
        )
        pipe = self.model.inference_pipeline
        pipe.reset_cache_state()
        out, _, _ = pipe.inference_with_trajectory(
            noise=noise,
            seed_latents=seed,
            requires_grad=False,
            **cond,
        )
        pipe.reset_cache_state()
        # The training ride's persistent streaming KV state was just
        # clobbered by the inference rollout above; force a fresh setup
        # on the next step rather than reusing a now-inconsistent cache.
        self.model.reset_streaming_state()

        if self.is_main_process:
            try:
                video_latents = torch.cat([seed.to(out.dtype), out], dim=1)
                self._log_pred_image_video(
                    video_latents, int(step), name="pred_image_7_chunk",
                    caption_suffix="7 GT ctx + 7 student rolled (GT actions)",
                    cap_frames=False,
                )
            except Exception as exc:  # pragma: no cover - logging only
                logging.warning(
                    "[ActionForcing] pred_image_7_chunk log failed at "
                    "step=%d: %s", int(step), exc,
                )

    @staticmethod
    def _draw_action_overlay(
        vid_np: np.ndarray,
        action_overlay,
        frames_per_latent: int = 4,
        npb: int = 3,
        zarr_lat_lo: Optional[int] = None,
        motion_chunk_offset: Optional[int] = None,
        zarr_name: str = "",
    ) -> None:
        """Draw per-chunk action bars at the bottom of every frame.

        ``vid_np``: ``[T_video, H, W, 3]`` uint8, edited in place.
        ``action_overlay``: ``[B, F_lat, A]`` or ``[F_lat, A]`` tensor /
        ndarray of post-tanh-squash z values in roughly ``[-1, 1]``.

        After Wan VAE temporal upsampling, ``T_video = F_lat *
        frames_per_latent`` (the leading dummy frame was already
        stripped off by the caller). One motion entry covers ``npb``
        consecutive latents; we render one bar value per chunk so the
        overlay is constant across the chunk's ``npb *
        frames_per_latent`` video frames and steps at chunk
        boundaries.

        Two stacked horizontal bars: top = z[0], bottom = z[1] (or up
        to ``A`` bars if A>2). Red bar to the right for ``z>=0``,
        blue bar to the left for ``z<0``; length scales with ``|z|``
        (clamped to 1.0 → half the frame width).
        """
        if action_overlay is None:
            return
        if torch.is_tensor(action_overlay):
            arr = action_overlay.detach().float().cpu().numpy()
        else:
            arr = np.asarray(action_overlay, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]  # batch 0 — rank-local rollout uses batch dim 0
        if arr.ndim != 2:
            return
        F_lat, A = arr.shape
        if F_lat == 0 or A == 0:
            return
        # Per-chunk z (one row per chunk; the dataset already
        # broadcasts within-chunk so all 3 latents in a chunk share
        # the same row — we just take every npb-th).
        z_per_chunk = arr[::npb]
        n_chunks = z_per_chunk.shape[0]
        T, H, W, _ = vid_np.shape
        if H < 8 or W < 16 or n_chunks == 0:
            return

        # Strip layout: 4-px top padding then A bars of equal height
        # with 2-px gaps; total bound by ~25% of frame height.
        max_strip = max(20, H // 4)
        gap = 2
        bar_h = max(3, (max_strip - 4 - gap * (A - 1)) // A)
        strip_h = 4 + bar_h * A + gap * (A - 1)
        cx = W // 2
        max_bar = (W // 2) - 4

        # Black underlay so the bars are readable on bright frames.
        vid_np[:, -strip_h:, :, :] = vid_np[:, -strip_h:, :, :] // 4

        frames_per_chunk = npb * frames_per_latent
        # cv2 needs a contiguous frame buffer; if the caller passed in
        # a transposed (= non-contiguous) ``vid_np`` we use a per-frame
        # contiguous copy and write it back.
        cv2 = None
        do_index = (
            zarr_lat_lo is not None and motion_chunk_offset is not None
        )
        if do_index:
            try:
                import cv2 as _cv2
                cv2 = _cv2
            except Exception:
                cv2 = None
        for t in range(T):
            chunk_idx = min(t // frames_per_chunk, n_chunks - 1)
            z = z_per_chunk[chunk_idx]
            for d in range(A):
                y0 = H - strip_h + 4 + d * (bar_h + gap)
                y1 = y0 + bar_h
                a = float(z[d])
                length = int(min(abs(a), 1.0) * max_bar)
                # Center marker (gray 1-px line).
                vid_np[t, y0:y1, cx - 1:cx + 1, :] = 128
                if length <= 0:
                    continue
                if a >= 0:
                    vid_np[t, y0:y1, cx:cx + length, 0] = 255
                    vid_np[t, y0:y1, cx:cx + length, 1] = 0
                    vid_np[t, y0:y1, cx:cx + length, 2] = 0
                else:
                    vid_np[t, y0:y1, cx - length:cx, 0] = 0
                    vid_np[t, y0:y1, cx - length:cx, 1] = 0
                    vid_np[t, y0:y1, cx - length:cx, 2] = 255

            # Per-frame zarr-latent + motion.npy chunk annotation
            # (top-left). Mapping: video frame t -> dataset latent
            # zarr_lat_lo + (t // frames_per_latent); dataset latent k
            # -> motion.npy chunk motion_chunk_offset + (k // npb).
            # Two stacked lines:
            #   line 1 (y=14): zarr file stem (which recording)
            #   line 2 (y=28): per-frame indices into that recording
            if cv2 is not None:
                zarr_lat = int(zarr_lat_lo) + (t // frames_per_latent)
                motion_chunk = int(motion_chunk_offset) + (zarr_lat // npb)
                strip_top_h = 34 if zarr_name else 20
                vid_np[t, :strip_top_h, :, :] = vid_np[t, :strip_top_h, :, :] // 4
                frame = np.ascontiguousarray(vid_np[t])
                if zarr_name:
                    cv2.putText(
                        frame,
                        f"zarr={zarr_name}",
                        (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                _idx_y = 28 if zarr_name else 14
                cv2.putText(
                    frame,
                    f"lat={zarr_lat} motion={motion_chunk}",
                    (4, _idx_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                vid_np[t] = frame


    # ------------------------------------------------------------------
    # Step-scheduled local_attn_size (KV cache window) helpers.
    # ------------------------------------------------------------------
    def _current_attn_size_frames(self) -> Optional[int]:
        """Return the active ``local_attn_size`` (in frames) for
        ``self.step`` per the configured schedule, or None if no
        schedule is set."""
        if not self._attn_size_schedule:
            return None
        active = self._attn_size_schedule[0][1]
        for thr, frames in self._attn_size_schedule:
            if self.step >= thr:
                active = frames
            else:
                break
        return int(active)

    def _dump_module_inventory(self) -> None:
        """Print a one-shot startup inventory of which trainer-side
        modules + optimizers are actually built (vs ``None``). Reveals
        whether any feature the operator believes is off (e.g.
        forward_noiser, alt_head, state_probe) is in fact allocating
        GPU memory. Only main process logs.
        """
        if not getattr(self, "is_main_process", True):
            return
        m = getattr(self, "model", None)
        names_trainer = [
            "r3gan_disc", "r3gan_disc_ddp", "r3gan_optimizer",
            # WP-PIXGAN: the pixel-texture critic allocates GPU memory the
            # moment it is built. This inventory exists to reveal exactly
            # that, so a new critic MUST appear here.
            "pixel_texture_disc", "pixel_texture_disc_ddp", "pix_optimizer",
            "forward_noiser_optimizer", "reverse_noiser_optimizer",
            "state_probe_optimizer",
            "real_teacher_optimizer", "critic_optimizer",
            "fake_optimizer", "optimizer",
            "_frozen_cotracker", "_frozen_ss_vae", "_frozen_vae",
        ]
        names_model = [
            "generator", "fake_score", "real_score", "real_score_frozen",
            "action_projection", "action_token_projection",
            "action_critic", "state_probe", "forward_noiser",
            "reverse_noiser", "vae",
        ]
        logging.info("[mem-inventory] === Trainer-side attributes ===")
        for n in names_trainer:
            val = getattr(self, n, None)
            tag = "ON" if val is not None else "off"
            type_name = type(val).__name__ if val is not None else "-"
            logging.info(
                "[mem-inventory]   self.%-32s  %-4s  type=%s",
                n, tag, type_name,
            )
        if m is not None:
            logging.info("[mem-inventory] === Model-side attributes ===")
            for n in names_model:
                val = getattr(m, n, None)
                tag = "ON" if val is not None else "off"
                type_name = type(val).__name__ if val is not None else "-"
                logging.info(
                    "[mem-inventory]   model.%-32s %-4s  type=%s",
                    n, tag, type_name,
                )
            # Booleans / config-derived flags that toggle subsystems
            flags = [
                "gan_enabled", "gan_backbone",
                "gan_pixel_texture_enabled",
                "flash_dmd_enabled", "boundary_vae_roundtrip",
                "dmd_frozen_teacher_pass_enabled",
                "real_teacher_train_online",
                "fake_alt_head_enabled", "forward_noiser_enabled",
                "dmd_lookback_chunks",
            ]
            logging.info("[mem-inventory] === Config flags ===")
            for f in flags:
                v_self = getattr(self, f, None)
                v_model = getattr(m, f, None)
                v_config = getattr(getattr(self, "config", None), f, None)
                logging.info(
                    "[mem-inventory]   %-40s self=%r model=%r config=%r",
                    f, v_self, v_model, v_config,
                )

    def _mem_step_snapshot(self, label: str) -> None:
        """Capture (allocated, peak-since-last-snapshot) at a labeled
        boundary inside one training step. Cleared at the start of each
        step. Dumped from ``_dump_step_mem_breakdown`` (called at end of
        step 1 / step 5).
        """
        if not (
            torch.cuda.is_available()
            and getattr(self, "is_main_process", True)
            and bool(getattr(self.config, "memory_audit_enabled", False))
        ):
            return
        if not hasattr(self, "_mem_step_snaps"):
            self._mem_step_snaps: List[Tuple[str, int, int]] = []
        try:
            alloc = torch.cuda.memory_allocated()
            peak = torch.cuda.max_memory_allocated()
            self._mem_step_snaps.append((label, alloc, peak))
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def _dump_step_mem_breakdown(self, step_label: str) -> None:
        """Print the full sequence of (label, allocated, peak-since-prev)
        snapshots collected during one step. Reveals which boundary in
        the step's code path drives each chunk of the transient peak.
        """
        if not (
            torch.cuda.is_available()
            and getattr(self, "is_main_process", True)
            and bool(getattr(self.config, "memory_audit_enabled", False))
        ):
            return
        snaps = getattr(self, "_mem_step_snaps", None)
        if not snaps:
            return

        def _gb(x: int) -> float:
            return x / (1024 ** 3)

        prev_alloc = snaps[0][1]
        logging.info(
            "[mem-step %s] === Per-boundary alloc + peak-since-prev ===",
            step_label,
        )
        logging.info(
            "[mem-step %s]   %-32s  alloc=%6.2f GB  peak_in_segment=%6.2f GB  "
            "delta_alloc=%+6.2f GB",
            step_label, snaps[0][0], _gb(snaps[0][1]), _gb(snaps[0][2]), 0.0,
        )
        for i in range(1, len(snaps)):
            label, alloc, peak = snaps[i]
            delta = alloc - prev_alloc
            logging.info(
                "[mem-step %s]   %-32s  alloc=%6.2f GB  peak_in_segment=%6.2f GB  "
                "delta_alloc=%+6.2f GB",
                step_label, label, _gb(alloc), _gb(peak), _gb(delta),
            )
            prev_alloc = alloc

    def _dump_memory_audit(self, label: str) -> None:
        """Print a structured memory audit so the operator can see
        WHERE the GPU memory is going. Three sections per call:

          1. Per-top-level-module parameter + buffer byte tally on the
             current rank. Reveals which subsystems (gen, fake_score,
             real_score, action_critic, state_probe, perceptual approxes,
             SAM2 disc, etc.) are the heavy hitters.
          2. ``torch.cuda.memory_stats`` snapshot: allocated / reserved
             / fragmentation / peak. Shows how much PyTorch is holding
             vs. how much it could free.
          3. Free-vs-total via ``torch.cuda.mem_get_info``: includes the
             non-PyTorch overhead (NCCL buffers, cuDNN workspace,
             CUDA context) so we can size the actual usable PyTorch
             ceiling.

        Only rank 0 logs (so the run.log isn't 32×repeated). Other
        ranks compute their own peaks for local inspection if needed
        but stay quiet.
        """
        if not (
            torch.cuda.is_available()
            and getattr(self, "is_main_process", True)
        ):
            return
        try:
            dev = torch.cuda.current_device()
            # ---- (1) Per-module param/buffer tally -----------------
            buckets: List[Tuple[str, int, int]] = []  # (name, n_params, bytes)

            def _bucket(name: str, mod: Optional[torch.nn.Module]) -> None:
                if mod is None:
                    return
                # Unwrap DDP for honest counting.
                inner = (
                    mod.module if isinstance(mod, DDP) else mod
                )
                n_param = 0
                n_bytes = 0
                for p in inner.parameters():
                    if p is None:
                        continue
                    n_param += p.numel()
                    n_bytes += p.numel() * p.element_size()
                for b in inner.buffers():
                    if b is None:
                        continue
                    n_bytes += b.numel() * b.element_size()
                buckets.append((name, n_param, n_bytes))

            m = self.model
            _bucket("generator", getattr(m, "generator", None))
            _bucket("fake_score", getattr(m, "fake_score", None))
            _bucket("real_score", getattr(m, "real_score", None))
            _bucket("real_score_frozen", getattr(m, "real_score_frozen", None))
            _bucket("action_projection", getattr(m, "action_projection", None))
            _bucket("action_token_projection", getattr(m, "action_token_projection", None))
            _bucket("action_critic", getattr(m, "action_critic", None))
            _bucket("state_probe", getattr(m, "state_probe", None))
            _bucket("r3gan_disc", getattr(self, "r3gan_disc", None))
            _bucket(
                "_frozen_cotracker",
                getattr(self, "_frozen_cotracker", None),
            )
            _bucket("_frozen_ss_vae", getattr(self, "_frozen_ss_vae", None))
            _bucket(
                "_frozen_vae",
                getattr(self, "_frozen_vae", None),
            )

            total_bytes = sum(b for _, _, b in buckets)
            buckets.sort(key=lambda t: -t[2])

            # ---- (2) PyTorch alloc / reserved / peak ---------------
            stats = torch.cuda.memory_stats(device=dev)
            alloc = stats.get("allocated_bytes.all.current", 0)
            reserved = stats.get("reserved_bytes.all.current", 0)
            peak_alloc = stats.get("allocated_bytes.all.peak", 0)
            peak_reserved = stats.get("reserved_bytes.all.peak", 0)

            # ---- (3) Free vs total (includes non-PyTorch overhead) -
            free_b, total_b = torch.cuda.mem_get_info(dev)
            non_pt = max(0, (total_b - free_b) - reserved)

            def _gb(x: int) -> float:
                return x / (1024 ** 3)

            logging.info(
                "[mem-audit %s] === Per-module param+buffer bytes ===",
                label,
            )
            for name, np_, nb in buckets:
                if nb == 0:
                    continue
                logging.info(
                    "[mem-audit %s]   %-22s  params=%9.2fM  bytes=%6.2f GB",
                    label, name, np_ / 1e6, _gb(nb),
                )
            logging.info(
                "[mem-audit %s]   %-22s  bytes=%6.2f GB  (sum of above)",
                label, "TOTAL_MODULE_BYTES", _gb(total_bytes),
            )
            logging.info(
                "[mem-audit %s] === PyTorch caching allocator ===",
                label,
            )
            logging.info(
                "[mem-audit %s]   allocated_now=%.2f GB  reserved_now=%.2f GB  "
                "fragmentation=%.1f%%",
                label, _gb(alloc), _gb(reserved),
                100.0 * (1.0 - alloc / max(reserved, 1)),
            )
            logging.info(
                "[mem-audit %s]   peak_allocated=%.2f GB  peak_reserved=%.2f GB",
                label, _gb(peak_alloc), _gb(peak_reserved),
            )
            logging.info(
                "[mem-audit %s] === GPU as a whole (rank 0) ===",
                label,
            )
            logging.info(
                "[mem-audit %s]   total=%.2f GB  free=%.2f GB  "
                "pytorch_reserved=%.2f GB  non_pytorch=%.2f GB",
                label, _gb(total_b), _gb(free_b),
                _gb(reserved), _gb(non_pt),
            )
        except Exception as exc:
            logging.warning(
                "[mem-audit %s] failed: %s", label, exc,
            )

    def _apply_attn_size_if_changed(self) -> bool:
        """If the schedule says the active ``local_attn_size`` differs
        from what we last applied, push the new value through the
        model's attention layers + the pipeline's cache-sizing knob,
        and force a streaming-state reset so the next sequence open
        re-allocates the KV cache at the new size. Returns True iff
        anything changed.

        Why reset instead of resizing in place: ``_initialize_kv_cache``
        allocates fixed-size ``[B, kv_cache_size, 12, 128]`` buffers per
        layer. There's no in-place resize for those buffers, and growing
        the attention window past the current cache size mid-sequence
        would let the model attend to slots that don't exist yet. A
        sequence reset is the cheapest atomic transition: closes the
        old sequence (releases old buffers next setup), then the next
        ``setup_sequence`` allocates at the new size.
        """
        target = self._current_attn_size_frames()
        if target is None or target == self._current_attn_frames:
            return False

        # Walk the bare DiT (under DDP / wrapper) and update each
        # CausalWanSelfAttention's local_attn_size. The infinity_rope
        # patch reads ``self.local_attn_size`` per attention forward,
        # so updating the attribute takes effect on the next forward
        # without re-installing the patch.
        dit = self._inner_dit_for_rope()
        if hasattr(dit, "local_attn_size"):
            dit.local_attn_size = target
        if hasattr(dit, "blocks"):
            for blk in dit.blocks:
                if hasattr(blk, "local_attn_size"):
                    blk.local_attn_size = target
                if hasattr(blk, "self_attn") and hasattr(blk.self_attn, "local_attn_size"):
                    blk.self_attn.local_attn_size = target

        # Bump the pipeline's rollout_frames so the next
        # ``_initialize_kv_cache`` call sizes the buffer to ``target``.
        # ``kv_cache_size = max(num_max_frames, rollout_frames) *
        # frame_seq_length`` (pipeline.kv_cache_size property).
        if self.pipeline is not None:
            self.pipeline.rollout_frames = max(
                int(self.pipeline.num_max_frames), int(target)
            )
            # max_attention_size was NOT schedule-aware (2026-08-17): it is
            # set once at construction from model_kwargs.local_attn_size and
            # never revisited, so a GROWING schedule (phase-1 default
            # [[0,21],[250,30],[500,42],...]) raised local_attn_size and the
            # roll trigger while the attention slice
            # (`local_end - max_attention_size`) still truncated to the OLD
            # frame count -> more memory, no extra context, silently. It only
            # went unnoticed here because this recipe pins a FLAT [[0,21]]
            # AND model_kwargs.local_attn_size=21, so the two agreed by
            # construction. Propagate it with the schedule so correctness no
            # longer depends on that duplicate pin being left alone.
            _fsl = int(self.pipeline.frame_seq_length)
            _tgt_tok = 32760 if int(target) == -1 else int(target) * _fsl
            _n_set = 0
            for _m in self.model.generator.modules():
                if hasattr(_m, "max_attention_size"):
                    _m.max_attention_size = _tgt_tok
                    _n_set += 1
            logging.info(
                "[ActionForcing] max_attention_size -> %d tokens (%s frames) "
                "on %d module(s) [schedule-propagated]",
                _tgt_tok, target, _n_set,
            )

        # Force the NEXT _fwdbwd_streaming_step to open a new sequence
        # (which calls setup_sequence -> _initialize_kv_cache with the
        # new size). reset_streaming_state is a no-op when state is
        # already None, so safe to call unconditionally.
        if hasattr(self.model, "reset_streaming_state"):
            self.model.reset_streaming_state()

        prev = self._current_attn_frames
        self._current_attn_frames = int(target)
        if self.is_main_process:
            logging.info(
                "[ActionForcing] local_attn_size schedule transition at "
                "step=%d: %s -> %d frames (KV cache buffers will "
                "re-allocate on next sequence open).",
                int(self.step), str(prev), int(target),
            )
        return True

    # ------------------------------------------------------------------
    # Motion-aware rollout offset picker.
    # ------------------------------------------------------------------
    def _pick_motion_aware_offset(
        self,
        ride: Dict[str, Any],
        s_local_max: int,
        window_lo_offset: int,
        window_len: int,
    ) -> int:
        """Pick a starting offset ``s ∈ [0, s_local_max]`` that gives a
        motion-positive rollout window when possible.

        With probability ``cfg.motion_start_prob`` (default 1.0),
        restricts the candidate set to offsets whose rollout window
        ``ride[s + window_lo_offset : s + window_lo_offset + window_len]``
        has mean per-frame motion magnitude ≥ ``cfg.motion_start_threshold``
        (code default 0.5, but the phase-3 DMD config sets 5.0 to keep
        only clearly high-motion windows; 0.5 merely excludes parked /
        idling windows). Among qualifying windows it samples uniformly.
        With probability 1 - prob, samples uniformly over all offsets.
        If motion is unavailable for this ride (or the window bounds
        don't fit), falls back to uniform sampling on ``[0, s_local_max]``.
        If motion IS available but NO window clears the threshold, picks
        the ride's single highest-motion window (argmax) — never a random
        low-motion offset — so the high-motion guarantee holds per ride.

        Per-rank: each rank picks independently from its own ride's
        motion. The legacy DDP pattern broadcast a rank-0 pick (so
        every rank used the same ``s``) — that broke as soon as we
        wanted motion-aware selection because rank 0's motion-positive
        offset is meaningless on rank 1's ride. Cross-rank sync is
        instead preserved via per-rank ``s_local_max`` ensuring
        ``s + cf + cap <= ride_len_r`` on every rank, so ``actual_cap
        = cap`` is invariant without any all_reduce.

        Chunk-boundary snap: the dataset enforces per-chunk identity on
        ``z_actions`` (one z per ``num_frame_per_block``-frame chunk;
        see utils/zarr_dataset.py:_LATENTS_PER_MOTION_CHUNK). Slicing
        the ride at an ``s`` that isn't a multiple of npb would put a
        chunk boundary mid-window in the model's view, producing a
        mid-chunk action transition that's OOD for v14's training
        contract (constant action within each chunk). We snap every
        return value to a multiple of npb here so all downstream
        slices are chunk-aligned by construction.
        """
        npb = int(getattr(self.config, "num_frame_per_block", 3))
        # Cap s_local_max to a multiple of npb so the snap below never
        # produces a value that exceeds the per-rank max.
        s_local_max = (max(0, int(s_local_max)) // npb) * npb
        if s_local_max <= 0:
            return 0

        def _snap(val: int) -> int:
            return (max(0, int(val)) // npb) * npb

        # ``random.randint(a, b)`` is INCLUSIVE on b; with b = s_local_max
        # already npb-aligned, snapping is idempotent on b. Other values
        # in (0, b) snap DOWN, which is fine — uniform-on-multiples-of-npb
        # is what we want.
        prob = float(getattr(self.config, "motion_start_prob", 1.0))
        threshold = float(getattr(self.config, "motion_start_threshold", 0.5))
        motion_mag = ride.get("motion_mag")
        use_motion = (
            motion_mag is not None
            and prob > 0.0
            and (prob >= 1.0 or random.random() < prob)
        )
        if not use_motion:
            return _snap(random.randint(0, s_local_max))
        mag_np = (
            motion_mag.cpu().numpy()
            if torch.is_tensor(motion_mag)
            else np.asarray(motion_mag)
        )
        n_lat = mag_np.shape[0]
        win_hi_max = s_local_max + window_lo_offset + window_len
        if win_hi_max > n_lat or window_len <= 0:
            return _snap(random.randint(0, s_local_max))
        # Sliding-window mean via cumulative sum.
        # window_means[s] = mean of mag_np[s+lo : s+lo+win] for
        # s ∈ [0, s_local_max].
        csum = np.concatenate(
            [[0.0], np.cumsum(mag_np, dtype=np.float64)],
        )
        lo = window_lo_offset
        n_candidates = s_local_max + 1
        sums = (
            csum[lo + window_len : lo + window_len + n_candidates]
            - csum[lo : lo + n_candidates]
        )
        means = sums / float(window_len)
        # Work directly on npb-aligned candidate offsets so we never bias
        # toward pseudo-aligned offsets via a post-hoc snap.
        aligned_all = np.arange(0, s_local_max + 1, npb)
        aligned_means = means[aligned_all]
        aligned_valid = aligned_all[aligned_means >= threshold]
        if aligned_valid.size > 0:
            # Uniform among the high-motion (>= threshold) windows.
            return int(aligned_valid[random.randrange(aligned_valid.size)])
        # No aligned window clears the threshold for this ride. Rather
        # than a uniform-random (low-motion) offset, pick the ride's
        # single HIGHEST-motion window (argmax) so "high motion only"
        # still holds even on rides whose best window is below cutoff.
        return int(aligned_all[int(np.argmax(aligned_means))])

    # ------------------------------------------------------------------
    # CPU-only ride prefetch (executor + queued future). Worker thread
    # does the disk reads in ``_load_ride_tensors_cpu_part`` — no CUDA
    # work, no ss_vae forward. Main thread does ss_vae + H2D inline at
    # consume time on the default CUDA stream. Why: multi-thread CUDA
    # on the default stream serializes, so a worker doing GPU work
    # wouldn't overlap; disk I/O is the dominant cost (~50-200ms) and
    # threads cleanly amortize it across the previous step's compute.
    # ------------------------------------------------------------------
    # Hard timeout on prefetch wait. If the worker stalls (NFS hiccup,
    # slow disk, deadlocked dataset code) the entire trainer would
    # hang silently without this — no log, no error. 60s is well
    # above any healthy disk-read time but well below NCCL's default
    # 30-min collective timeout, so we surface the stall as a clean
    # exception (with traceback on every rank that hits it) before
    # downstream collectives time out and produce confusing errors.
    _PREFETCH_FUTURE_TIMEOUT_S: float = 60.0

    # Consecutive-None guard. ``_streaming_step`` returns None when
    # the ride loader can't produce a usable ride (all rides too
    # short, motion-filter rejects everything, prefetch CPU stage
    # exhausted attempts). The outer trainer treats None as "skip
    # optim, increment step, continue" — so a permanent loader fault
    # would silently waste the entire training run. Raise after this
    # many consecutive None returns so the run dies loudly with
    # diagnostics instead.
    _MAX_CONSECUTIVE_STREAMING_NONE: int = 10

    def _meta_passes_filters(
        self, meta: dict, rollout_frames: int,
    ) -> bool:
        """Shared min-frames + motion-magnitude filter for both the
        prefetch worker (``_next_ride_cpu_only``) and the legacy
        synchronous loader (``_next_ride``). Single source of truth
        so the two paths can't drift apart on filter logic.
        """
        n_latent_frames = int(meta.get("n_latent_frames", 0))
        if n_latent_frames < rollout_frames:
            return False
        skip_dead = bool(getattr(self.config, "motion_skip_dead_rides", True))
        if not skip_dead:
            return True
        ride_min_mean = float(getattr(self.config, "motion_ride_min_mean", 3.0))
        mag_loader = getattr(self.dataset, "load_motion_magnitudes", None)
        if mag_loader is None:
            return True
        try:
            mag = mag_loader(meta["zarr_path"], n_latent_frames)
        except Exception as e:
            # Best-effort: missing motion file shouldn't block the
            # ride; log and accept. (Same semantics as the legacy
            # ``_next_ride`` filter, kept centralized.)
            logging.warning(
                "motion pre-check failed for %s: %s — accepting ride "
                "without filter", meta.get("zarr_path", "?"), e,
            )
            return True
        return float(mag.mean()) >= ride_min_mean

    def _ensure_prefetch_executor(self) -> None:
        """Lazy-init the single-worker thread pool that does CPU ride
        prefetch, plus the slot that holds the pending Future. Also
        registers an ``atexit`` shutdown so the worker thread doesn't
        leak on SIGKILL / unhandled exception in another thread (CI
        produces a clean process exit instead of "process won't exit"
        zombies)."""
        if not hasattr(self, "_prefetch_executor") or self._prefetch_executor is None:
            self._prefetch_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="ride_prefetch",
            )
            atexit.register(self._prefetch_executor.shutdown, wait=False)
        if not hasattr(self, "_prefetched_ride_future"):
            self._prefetched_ride_future = None

    def _kick_ride_prefetch_if_idle(self, rollout_frames: int) -> None:
        """Submit a CPU-side ride load to the worker thread if no
        prefetch is already in flight. Idempotent — no-op if a future
        is already queued. Respects ``self.max_ride_frames`` so the
        prefetched RAM is bounded by the actually-reachable size, not
        the raw zarr length.
        """
        self._ensure_prefetch_executor()
        if self._prefetched_ride_future is not None:
            return
        max_frames = getattr(self, "max_ride_frames", None)
        self._prefetched_ride_future = self._prefetch_executor.submit(
            self._next_ride_cpu_only, rollout_frames, max_frames,
        )

    def _next_ride_cpu_only(
        self, rollout_frames: int, max_frames: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        """Run on the worker thread. Pulls the next ride from
        ``self._ride_iter`` (single-writer invariant: only this method
        ever advances the iterator) and runs the CPU-side disk reads.
        Returns the cpu_part dict that ``_finalize_ride_to_gpu`` will
        consume on the main thread, or None if the loader is exhausted
        after ``max_attempts`` rejected rides.
        """
        attempts = 0
        max_attempts = 200
        while attempts < max_attempts:
            try:
                batch = next(self._ride_iter)
            except StopIteration:
                self._epoch += 1
                self._ride_iter = self._fresh_ride_iter(self._epoch)
                continue
            if not batch:
                attempts += 1
                continue
            meta = batch[0]
            if not self._meta_passes_filters(meta, rollout_frames):
                attempts += 1
                continue
            cpu_part = _load_ride_tensors_cpu_part(
                self.dataset, meta, max_frames=max_frames,
                random_window=bool(
                    getattr(self.config, "max_ride_frames_random", False)
                ),
            )
            if cpu_part is None or cpu_part["latents_cpu"].shape[0] < rollout_frames:
                attempts += 1
                continue
            return cpu_part
        return None

    @staticmethod
    def _is_dataloader_worker_death(exc: BaseException) -> bool:
        """Heuristic: did this exception come from a PyTorch DataLoader
        fork-worker dying mid-fetch? PyTorch's signal_handling raises a
        plain ``RuntimeError`` whose message starts with "DataLoader
        worker (pid …) exited unexpectedly". The traceback may surface
        the error wherever the SIGCHLD handler fires (often deep inside
        a flash-attention forward), so we match on the message rather
        than on the call stack.
        """
        if not isinstance(exc, RuntimeError):
            return False
        msg = str(exc)
        return (
            "DataLoader worker" in msg
            and "exited unexpectedly" in msg
        )

    def _consume_or_load_ride(
        self, rollout_frames: int,
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], float]:
        """Block on the queued prefetch (kick one synchronously if
        none queued — cold-start path), run ss_vae + H2D on the main
        thread, return ``(ride_dict, prefetch_wait_ms)``.

        ``prefetch_wait_ms`` is the time spent BLOCKED on the worker's
        future (= time the prefetch couldn't keep up with the reset
        rate). Caller logs it as ``streaming_prefetch_wait_ms`` so we
        can tell apart "ss_vae + H2D dominates" (high setup_ms, low
        wait_ms) from "prefetch can't keep up" (high wait_ms).

        Hard 60s timeout on the future. If the worker stalls, we raise
        a clear ``TimeoutError`` instead of letting the trainer hang
        silently until NCCL collectives time out 30 min later.

        Defensive single-retry on PyTorch DataLoader fork-worker death.
        With ``num_workers > 0`` the dataloader fork-spawns CPU helpers
        that occasionally die from transient OS signals (page-cache
        eviction race during zarr open, NFS hiccup, OOM-killer drive-by,
        ...). PyTorch surfaces those deaths as
        ``RuntimeError("DataLoader worker (pid N) exited unexpectedly")``
        from the next ``next(self._ride_iter)`` call. We catch that
        once, recreate ``self._ride_iter`` (which tears down the dead
        worker pool and forks fresh), kick a new prefetch, and retry.
        Cap at one retry so a persistent fault still surfaces.

        ``self._ride_iter`` is touched ONLY by the executor's worker
        thread, never from the main thread directly — even on cold
        start (we ``submit`` and immediately await rather than calling
        ``next()`` ourselves) to preserve the single-writer invariant.
        On retry, the iterator-reset itself runs on the main thread but
        only AFTER the executor's task has already returned (raised),
        so single-writer is preserved.
        """
        self._ensure_prefetch_executor()
        if self._prefetched_ride_future is None:
            self._kick_ride_prefetch_if_idle(rollout_frames)
        fut = self._prefetched_ride_future
        self._prefetched_ride_future = None
        t0 = time.monotonic()
        try:
            cpu_part = fut.result(timeout=self._PREFETCH_FUTURE_TIMEOUT_S)
        except FuturesTimeoutError as exc:
            raise TimeoutError(
                f"ride prefetch stalled for >"
                f"{self._PREFETCH_FUTURE_TIMEOUT_S:.0f}s — disk hiccup, "
                f"deadlocked worker, or dataset code. Surfacing before "
                f"NCCL collective timeouts produce a confusing error."
            ) from exc
        except RuntimeError as exc:
            if not self._is_dataloader_worker_death(exc):
                raise
            logging.warning(
                "[ActionForcing] DataLoader worker died (transient "
                "subprocess fault): %s. Recreating ride iterator and "
                "retrying once.",
                exc,
            )
            try:
                self._ride_iter = self._fresh_ride_iter(self._epoch)
            except Exception as reset_exc:  # pragma: no cover
                logging.error(
                    "[ActionForcing] Failed to recreate ride iterator "
                    "after worker death: %s", reset_exc,
                )
                raise
            self._kick_ride_prefetch_if_idle(rollout_frames)
            retry_fut = self._prefetched_ride_future
            self._prefetched_ride_future = None
            try:
                cpu_part = retry_fut.result(
                    timeout=self._PREFETCH_FUTURE_TIMEOUT_S,
                )
            except FuturesTimeoutError as exc2:
                raise TimeoutError(
                    f"ride prefetch retry stalled for >"
                    f"{self._PREFETCH_FUTURE_TIMEOUT_S:.0f}s after "
                    f"DataLoader-worker recovery."
                ) from exc2
            except RuntimeError as exc2:
                if self._is_dataloader_worker_death(exc2):
                    raise RuntimeError(
                        "DataLoader worker died TWICE in a row "
                        "(after iterator recreation). This is no "
                        "longer a transient fault — investigate "
                        "dmesg, CPU RAM pressure, or a corrupted "
                        "ride file."
                    ) from exc2
                raise
        wait_ms = (time.monotonic() - t0) * 1000.0
        if cpu_part is None:
            return None, wait_ms
        ride = _finalize_ride_to_gpu(
            cpu_part, self.dataset, self.device, self.dtype,
            action_dims=self.action_dims,
        )
        return ride, wait_ms

    # ------------------------------------------------------------------
    # Toothpaste curriculum — pure helpers (unit-tested in
    # testing/test_toothpaste_and_defer_disc.py).
    # ------------------------------------------------------------------
    # Thin delegates to the torch-free, unit-tested helpers in
    # ``trainer/toothpaste.py`` (kept separate so the pure curriculum logic
    # can be tested without importing this CUDA-initialising module).
    @staticmethod
    def _toothpaste_grow(window, win, prev_avg, tol):
        from trainer.toothpaste import toothpaste_grow
        return toothpaste_grow(window, win, prev_avg, tol)

    @staticmethod
    def _toothpaste_gone(frontier_mae, gone_base, gone_factor):
        from trainer.toothpaste import toothpaste_gone
        return toothpaste_gone(frontier_mae, gone_base, gone_factor)

    @staticmethod
    def _setup_s_local_max(ride_len, cf, cap, slack, anchor, min_new, npb):
        from trainer.toothpaste import setup_s_local_max
        return setup_s_local_max(
            ride_len, cf, cap, slack, anchor, min_new, npb)

    @staticmethod
    def _going_gone_gate(
        frontier_mae, going_threshold, gone_base, gone_factor,
        min_depth, cur_depth,
    ):
        """FT_v3 post-build two-threshold gate (delegate). Returns one of
        ``"gone"`` / ``"going"`` / ``"roll"`` — see
        ``trainer.toothpaste.going_gone_gate``. Used ONLY on the
        ``ftv3_postbuild_enabled`` path; the default-OFF path uses the
        single ``_toothpaste_gone`` gate unchanged."""
        from trainer.toothpaste import going_gone_gate
        return going_gone_gate(
            frontier_mae, going_threshold, gone_base, gone_factor,
            min_depth, cur_depth)

    # ------------------------------------------------------------------
    # af-all: IQA (MUSIQ/NIQE) agreement gate
    # ------------------------------------------------------------------
    def _af_all_iqa_metrics(self, latent_chunk: torch.Tensor):
        """Decode a latent chunk and return ``(musiq_per_frame[list],
        niqe_mean[float])`` — both no_grad. MUSIQ ↑good (~0..100), NIQE
        ↓good. Returns ``(None, None)`` on any decode/metric failure so
        the caller treats the window as 'not agreeing' (conservative)."""
        vae = getattr(self.model, "vae", None)
        if vae is None:
            return None, None
        if self._iqa_musiq is None:
            import pyiqa
            dev = self.device
            self._iqa_musiq = pyiqa.create_metric("musiq", device=dev)
            self._iqa_niqe = pyiqa.create_metric("niqe", device=dev)
        # Defrag before the decode: the VAE decoder peaks at a large
        # contiguous workspace (~1GB) and a fragmented allocator can fail
        # the alloc even with enough total free memory (mirrors the
        # graph-on decode's pre-decode empty_cache).
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # The VAE params are fp32 (its conv biases are float); the rollout
        # latents are bf16. Cast the latent to the VAE's own dtype before
        # decode or conv3d raises "Input type/bias type should be the same".
        try:
            _vae_dtype = next(vae.parameters()).dtype
        except StopIteration:
            _vae_dtype = torch.float32
        with torch.no_grad():
            pix = vae.decode_to_pixel(
                latent_chunk.to(device=self.device, dtype=_vae_dtype),
                seed_first=True,
            )
            v = (0.5 * (pix.float() + 1.0)).clamp(0.0, 1.0)[0]  # [F,3,H,W]
            musiq = [float(self._iqa_musiq(v[i:i + 1]).item())
                     for i in range(v.shape[0])]
            niqe = [float(self._iqa_niqe(v[i:i + 1]).item())
                    for i in range(v.shape[0])]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        niqe_mean = sum(niqe) / float(max(1, len(niqe)))
        return musiq, niqe_mean

    def _af_all_iqa_agreement(
        self,
        gen_chunk: torch.Tensor,
        gt_seed_chunk: torch.Tensor,
    ) -> float:
        """Binary local agreement (1.0/0.0) between the most-recent
        generated frontier chunk and the last GT seed chunk:
          * gen mean-MUSIQ ≥ GT mean-MUSIQ − ``musiq_tol``  (no worse), AND
          * gen mean-NIQE ≤ GT mean-NIQE + ``niqe_tol``     (no worse), AND
          * within the generated chunk, MUSIQ(late) − MUSIQ(early) ≥
            −``late_early_drop`` (no intra-chunk collapse).
        Conservative: any decode/metric failure → 0.0 (not agreeing)."""
        g_musiq, g_niqe = self._af_all_iqa_metrics(gen_chunk)
        if g_musiq is None:
            return 0.0
        s_musiq, s_niqe = self._af_all_iqa_metrics(gt_seed_chunk)
        if s_musiq is None:
            return 0.0
        g_mu = sum(g_musiq) / float(len(g_musiq))
        s_mu = sum(s_musiq) / float(len(s_musiq))
        late_early = g_musiq[-1] - g_musiq[0]
        ok = (
            (g_mu >= s_mu - self._iqa_musiq_tol)
            and (g_niqe <= s_niqe + self._iqa_niqe_tol)
            and (late_early >= -self._iqa_late_early_drop)
        )
        return 1.0 if ok else 0.0

    # ------------------------------------------------------------------
    # K=1-per-step state machine. Each call rolls one chunk on the
    # current ride (or sets up a fresh ride if reset was triggered last
    # step), trains every active head, and decides whether to reset
    # before the next call. Persistent ``self.model.streaming_state``
    # carries KV cache + ``ride_*_window`` across calls.
    # ------------------------------------------------------------------
    def _streaming_step(
        self,
        rollout_frames: int,
        cf_dmdctx: int,
    ) -> Optional[Dict[str, Any]]:
        """One outer training step in K=1 mode.

        Returns a flat ``str -> float`` log dict (or None if the ride
        loader is exhausted on a setup-required step). The trainer's
        outer ``optim.step()`` runs after this returns.
        """
        cfg = self.config
        max_rolls = int(getattr(cfg, "max_rolls_per_ride", 60))
        # ---- FT_v3 post-build master flag (default OFF = byte-identical) --
        # When OFF, NONE of the new roll-until-going + post-build branches
        # are taken and ``_streaming_step`` runs the exact current code.
        # af-stat (stationary, max_rolls_per_ride=1) never sets this flag,
        # and even if it did, the going gate is additionally guarded by
        # ``ftv3_going_min_depth`` (default 4 > 1) so a single-roll ride can
        # never enter the post-build path. Sub-knobs are read with getattr
        # defaults so existing YAMLs are unaffected.
        _ftv3_on = bool(getattr(cfg, "ftv3_postbuild_enabled", False))
        if _ftv3_on:
            self._ftv3_going_threshold = float(
                getattr(cfg, "streaming_mae_going_threshold", 0.0))
            self._ftv3_going_min_depth = int(
                getattr(cfg, "ftv3_going_min_depth", 4))
        # ---- Toothpaste rollout-depth curriculum (phase-2 rolling) -------
        # MAE-EARNED depth growth: the rollout depth cap grows by +1 roll
        # ONLY when the running-average frontier MAE at the current depth is
        # <= the previous depth's running average x (1+tol) ("as good as one
        # step shallower"). Replaces the step-count rolls schedule. Depth is
        # GLOBAL (all ranks agree): the growth decision is driven by the
        # all-reduced MEAN frontier MAE at reset, so every rank holds an
        # identical _tp_depth and the reset cadence stays lockstep. Lazy
        # state init on first call. See the reset block for grow/gone logic.
        _tp_on = bool(getattr(cfg, "rolling_toothpaste_enabled", False))
        if _tp_on and not getattr(self, "_tp_init", False):
            self._tp_init = True
            self._tp_depth = int(
                getattr(cfg, "rolling_toothpaste_depth_start", 4))
            self._tp_cur: List[float] = []        # frontier MAEs @ cur depth
            self._tp_prev_avg: Optional[float] = None  # prev-depth running avg
            self._tp_win = int(
                getattr(cfg, "rolling_toothpaste_avg_window", 5))
            self._tp_tol = float(
                getattr(cfg, "rolling_toothpaste_grow_tol", 0.10))
            self._tp_gone = float(
                getattr(cfg, "rolling_toothpaste_gone_factor", 3.0))
            self._tp_windows_per_ride = int(
                getattr(cfg, "rolling_windows_per_ride", 8))
            self._tp_windows_this_ride = 0
            self._tp_held_ride: Optional[Dict[str, Any]] = None
            # af-all: IQA (MUSIQ/NIQE) agreement gate. When enabled it
            # REPLACES the MAE window as the depth-grow signal. Once per
            # ride (at reset) decode the most-recent generated frontier
            # chunk + the last GT seed chunk, score both with MUSIQ
            # (↑good) and NIQE (↓good), form a binary "agree" indicator
            # (gen within tol of the GT seed on BOTH metrics AND no
            # late−early MUSIQ collapse within the generated chunk),
            # MEAN-reduce it across ranks (DDP lockstep) and EMA it. When
            # the EMA clears ``agree_threshold`` the depth ratchets +1 and
            # the EMA resets, so the next +1 must be re-earned at the
            # deeper depth ("roll one more chunk … until the metrics agree
            # again"). Monotonic; per-ride roll_cap is still the ceiling.
            self._iqa_on = bool(
                getattr(cfg, "af_all_iqa_gate_enabled", False))
            self._iqa_musiq_tol = float(
                getattr(cfg, "af_all_iqa_musiq_tol", 5.0))
            self._iqa_niqe_tol = float(
                getattr(cfg, "af_all_iqa_niqe_tol", 0.5))
            self._iqa_late_early_drop = float(
                getattr(cfg, "af_all_iqa_late_early_drop", 5.0))
            self._iqa_ema_decay = float(
                getattr(cfg, "af_all_iqa_ema_decay", 0.9))
            self._iqa_agree_threshold = float(
                getattr(cfg, "af_all_iqa_agree_threshold", 0.5))
            self._iqa_agree_ema: Optional[float] = None
            self._iqa_musiq = None   # lazy pyiqa metric
            self._iqa_niqe = None    # lazy pyiqa metric
            if self._iqa_on and self.is_main_process:
                logging.info(
                    "[af-all] IQA depth-gate ENABLED (MUSIQ/NIQE replace "
                    "MAE): musiq_tol=%.2f niqe_tol=%.2f late_early_drop=%.2f "
                    "ema_decay=%.3f agree_threshold=%.2f.",
                    self._iqa_musiq_tol, self._iqa_niqe_tol,
                    self._iqa_late_early_drop, self._iqa_ema_decay,
                    self._iqa_agree_threshold,
                )
        if _tp_on:
            max_rolls = int(self._tp_depth)
        else:
            # Legacy step-count schedule (only LOWERS the cap): cap rolls at
            # ``base + (step - start) // every``.
            _sched_every = int(getattr(
                cfg, "rolling_rolls_schedule_every", 0))
            if _sched_every > 0:
                _sched_base = int(getattr(
                    cfg, "rolling_rolls_schedule_base", 2))
                _sched_start = int(getattr(
                    cfg, "rolling_rolls_schedule_start_step", 0))
                _cur_step = int(getattr(self, "step", 0))
                max_rolls = min(
                    max_rolls,
                    _sched_base
                    + max(0, _cur_step - _sched_start) // _sched_every,
                )
        # ``rolling_random_depth_enabled`` (2026-08-20, reference parity):
        # Causal-Forcing/long_video draws a RANDOM number of blocks to roll per
        # optimizer step and broadcasts it from rank 0, then supervises the
        # frontier window. A fixed depth ladder over-weights shallow rolls
        # (whose context is near-GT) relative to inference, which runs deep.
        # Drawn ONCE per ride and held for that ride's rolls; broadcast so
        # every rank holds the identical cap (mandatory -- the cap drives the
        # only-last skip, and a rank-divergent skip hangs NCCL on the scorer's
        # collectives).
        if bool(getattr(cfg, "rolling_random_depth_enabled", False)):
            _new_ride = self.model.streaming_state is None
            if _new_ride or not hasattr(self, "_ride_random_depth"):
                _dmin = int(getattr(cfg, "rolling_random_depth_min", 2))
                _dmax = int(getattr(cfg, "rolling_random_depth_max",
                                    int(max_rolls)))
                _dmax = max(_dmin, _dmax)
                if dist.is_initialized() and dist.get_world_size() > 1:
                    if self.is_main_process:
                        _d = torch.tensor(
                            [int(torch.randint(_dmin, _dmax + 1, (1,)).item())],
                            device=self.device, dtype=torch.long,
                        )
                    else:
                        _d = torch.empty(
                            1, dtype=torch.long, device=self.device,
                        )
                    dist.broadcast(_d, src=0)
                    self._ride_random_depth = int(_d.item())
                else:
                    self._ride_random_depth = int(
                        torch.randint(_dmin, _dmax + 1, (1,)).item()
                    )
            max_rolls = int(self._ride_random_depth)
        # ``dmd_supervise_roll_mode="random"`` (2026-08-22): draw the ONE
        # roll index that receives the generator DMD gradient this ride.
        # Same once-per-ride trigger, same rank-0 draw + broadcast pattern
        # and same device/dtype as the random-depth draw above (mandatory:
        # the target drives the scorer-skip gate in
        # compute_generator_loss_streaming, the scorer contains
        # collectives, and a rank-divergent target hangs NCCL). Drawn
        # uniformly in [1, max_rolls] where max_rolls is the ride's final
        # (post-random-depth, pre-capacity-clamp) cap — at a new ride the
        # streaming state is None so the capacity clamp below is inactive.
        # Stamped on the MODEL so both model-side guards (generator skip +
        # clean_match mirror) read one source of truth. If a ride resets
        # before reaching the target the ride gets no DMD gradient (same
        # known limitation as only-last; visible via dmd_supervised_count).
        if (
            str(getattr(cfg, "dmd_supervise_roll_mode", "all") or "all")
            .strip().lower() == "random"
        ):
            _new_ride_sup = self.model.streaming_state is None
            if _new_ride_sup or not hasattr(self, "_ride_supervise_roll"):
                _tmax = max(1, int(max_rolls))
                if dist.is_initialized() and dist.get_world_size() > 1:
                    if self.is_main_process:
                        _t = torch.tensor(
                            [int(torch.randint(1, _tmax + 1, (1,)).item())],
                            device=self.device, dtype=torch.long,
                        )
                    else:
                        _t = torch.empty(
                            1, dtype=torch.long, device=self.device,
                        )
                    dist.broadcast(_t, src=0)
                    self._ride_supervise_roll = int(_t.item())
                else:
                    self._ride_supervise_roll = int(
                        torch.randint(1, _tmax + 1, (1,)).item()
                    )
            self.model._dmd_supervise_target_roll = int(
                self._ride_supervise_roll
            )
        # Physical-capacity clamp: a cap the ride cannot reach means the ride
        # resets by exhaustion below the cap and (under only-last) receives
        # ZERO supervision -- measured live on rollref_rand: caps 3-6, every
        # ride dead at roll 2, dmd_supervised_this_roll == 0.0 for the whole
        # run. Capacity is computed from the LIVE streaming state and
        # MIN-reduced: any rank's exhaustion forces a global reset anyway (the
        # reset flag is MAX-reduced), so the global min IS the true capacity,
        # and the reduce keeps the cap rank-uniform (mandatory for the
        # only-last skip -- a divergent cap hangs NCCL on the scorer).
        _st = getattr(self.model, "streaming_state", None)
        if _st is not None and int(max_rolls) > 1:
            # Stride = what a roll ACTUALLY advances: force*npb when the
            # deterministic stride is set, else min_new_frame. Using min_new
            # alone overestimates capacity when force*npb > min_new -> cap
            # clamped to an unreachable value -> only-last blackout returns.
            _npb_c = int(self.model.num_frame_per_block)
            _force_c = int(getattr(
                self.model, "streaming_force_new_frame_chunks", 0))
            _nfstep = max(
                _force_c * _npb_c,
                int(getattr(self.model, "streaming_min_new_frame", _npb_c)),
            )
            _room = int(_st["max_length"]) - int(_st["current_length"])
            _cap_local = int(self._chunks_in_current_ride) + max(
                0, _room // max(1, _nfstep))
            if dist.is_initialized() and dist.get_world_size() > 1:
                _cap_t = torch.tensor(
                    [_cap_local], device=self.device, dtype=torch.long)
                dist.all_reduce(_cap_t, op=dist.ReduceOp.MIN)
                _cap_local = int(_cap_t.item())
            if _cap_local < int(max_rolls):
                max_rolls = max(1, _cap_local)
        # Publish the resolved cap so _build_train_info (and the only-last
        # supervision gate) sees the SAME depth this step actually uses.
        self._max_rolls_this_step = int(max_rolls)
        force_exit_step_enabled = bool(
            getattr(cfg, "force_exit_step_enabled", False)
        )

        out: Dict[str, Any] = {}
        t_step_start = time.monotonic()
        t_setup_ms = 0.0
        prefetch_wait_ms = 0.0

        # ---- Stage 1: set up a fresh ride if needed ------------------
        needs_setup = self.model.streaming_state is None
        if needs_setup:
            t_setup_start = time.monotonic()
            # Same-ride resampling (toothpaste): instead of loading a fresh
            # ride from disk on every reset, REUSE the already-resident ride
            # for up to ``rolling_windows_per_ride`` windows, re-seeding from
            # a NEW motion-aware offset each time (the whole ride is on GPU
            # and is longer than one window — picking a new ``s`` is free,
            # vs a disk read + decode + H2D for a new ride). The reuse
            # decision is identical across ranks (the windows counter is
            # bumped only on the rank-lockstep reset, windows_per_ride is a
            # constant, and every rank holds a ride after setup), so the
            # anchor-forward broadcast inside setup_sequence stays matched.
            _reuse_ride = (
                _tp_on
                and getattr(self, "_tp_held_ride", None) is not None
                and self._tp_windows_this_ride < self._tp_windows_per_ride
            )
            if _reuse_ride:
                ride = self._tp_held_ride
                prefetch_wait_ms = 0.0
            else:
                ride, prefetch_wait_ms = self._consume_or_load_ride(
                    rollout_frames + cf_dmdctx,
                )
                if ride is None:
                    # Loader exhausted on this step. Outer trainer treats
                    # None as "skip optim, increment step, continue". The
                    # consecutive-None guard below catches a permanent
                    # loader fault before silently wasting the entire run.
                    return self._handle_streaming_step_none(
                        "ride_loader_exhausted_on_setup",
                    )
                if _tp_on:
                    self._tp_windows_this_ride = 0   # fresh ride
            ok = self._streaming_setup_sequence_from_ride(
                rollout_frames=rollout_frames,
                max_total_rollout_frames=rollout_frames,
                cf_dmdctx=cf_dmdctx,
                ride=ride,
            )
            if not ok:
                # Setup rejected this ride (too short / bad offset). Drop any
                # held ride so the next setup loads a fresh one.
                if _tp_on:
                    self._tp_held_ride = None
                return self._handle_streaming_step_none(
                    "setup_sequence_rejected_ride",
                )
            if _tp_on:
                self._tp_held_ride = ride   # hold for same-ride resampling
            self._chunks_in_current_ride = 0
            t_setup_ms = (time.monotonic() - t_setup_start) * 1000.0
            # FT_v3 fresh-ride OOM fix. setup_sequence above runs the
            # _prebuild_rollout2 + seed-prefill mini-rollouts (no_grad, freed
            # immediately) but their blocks stay reserved in the caching
            # allocator. This SAME step then runs a full training step ending
            # in the GAN disc R1 double-backward (heavy); the setup residual
            # stacks under that backward and tips fresh-ride steps over (the
            # observed OOM missed by ~40 MB). Return the freed setup blocks to
            # the allocator now so the training forward/backward allocates from
            # a clean state. Result-preserving; rank-uniform (needs_setup is
            # lockstep — every rank resets together — so the sync can't desync
            # DDP).
            if (
                bool(getattr(
                    self.config, "empty_cache_after_ride_setup", True))
                and torch.cuda.is_available()
            ):
                torch.cuda.empty_cache()
        # Reset the consecutive-None counter on any successful setup
        # OR continuing-from-existing-state path.
        self._consecutive_streaming_step_none = 0

        # ---- Kick prefetch for the NEXT ride (overlaps this step's
        # backward with the next ride's disk I/O). Idempotent.
        self._kick_ride_prefetch_if_idle(rollout_frames + cf_dmdctx)

        state = self.model.streaming_state
        chunk_size = int(state["chunk_size"])
        cf_state = int(state["cf"])

        # ---- Optional per-step force_exit_step broadcast (config-gated).
        # Default OFF — under K=1 the inner ``generate_and_sync_list(sync=True)``
        # already broadcasts once per call on every rank, matched
        # count, so cross-rank exit-rung lockstep is preserved without
        # outer plumbing. Kept as a knob for variance-reduction
        # experiments that want to pin a single rank-0-picked rung.
        forced_exit_step: Optional[int] = None
        if force_exit_step_enabled:
            pipe = self.model.inference_pipeline
            n_steps = len(pipe.denoising_step_list)
            if pipe.last_step_only:
                forced_exit_step = n_steps - 1
            elif dist.is_initialized() and dist.get_world_size() > 1:
                if dist.get_rank() == 0:
                    idx_t = torch.randint(
                        0, n_steps, (1,),
                        device=self.device, dtype=torch.long,
                    )
                else:
                    idx_t = torch.empty(
                        1, dtype=torch.long, device=self.device,
                    )
                dist.broadcast(idx_t, src=0)
                forced_exit_step = int(idx_t.item())
            else:
                forced_exit_step = int(torch.randint(0, n_steps, (1,)).item())

        # ---- Stage 2: roll one chunk ---------------------------------
        # ``compute_baseline_mae=False`` skips the pipeline's internal
        # gt_chunk slice + ``_compute_chunk_mae`` (which would also
        # fire a DDP all_reduce). Under K=1-per-step that all_reduce
        # would be matched-count-safe, but it's redundant work — we
        # compute the per-rank MAE on this same window in Stage 3
        # below using the ride_latents_window slice we already hold.
        # Kept to skip the redundant compute, NOT for deadlock-avoidance.
        t_rollout_start = time.monotonic()
        chunk, info = self.model.generate_next_chunk(
            requires_grad=True,
            compute_baseline_mae=False,
            force_exit_step=forced_exit_step,
        )
        self._chunks_in_current_ride = (
            getattr(self, "_chunks_in_current_ride", 0) + 1
        )

        # ---- Stage 3: per-rank MAE on the chunk_size-frame trained window.
        noisy_start_sdn = int(
            info["current_length"] - info["new_frames"] - info["overlap"]
        )
        chunk_lo = cf_state + noisy_start_sdn
        chunk_hi = chunk_lo + chunk_size
        ride_window = state["ride_latents_window"]
        # Geometry invariant: setup_sequence builds ride_latents_window of
        # length cf + actual_cap, ``can_generate_more()`` keeps
        # current_length ≤ max_length=actual_cap, and chunk_hi simplifies
        # to cf_state + current_length ≤ ride_window.shape[1].
        assert ride_window.shape[1] >= chunk_hi, (
            f"streaming-step geometry violation: ride_window.shape[1]="
            f"{ride_window.shape[1]} < chunk_hi={chunk_hi}."
        )
        gt_slice = ride_window[:, chunk_lo:chunk_hi]
        avg_mae = float(
            (chunk.detach().float() - gt_slice.float()).abs().mean().item()
        )
        # Matched-clean-x: select the GT offset that best aligns the DMD band,
        # stash it for ``_build_42f_scoring_inputs`` (clean_x / gt_target), and
        # make the gate MAE follow the SAME offset. No-op (returns 0, None) when
        # ``dmd_42f_clean_match_enabled`` is off, so avg_mae is unchanged then.
        _cm_off, _cm_mae = self.model.compute_clean_match_offset(
            chunk, info, chunks_in_ride=int(self._chunks_in_current_ride),
            max_rolls=int(getattr(self, "_max_rolls_this_step", 0)),
        )
        if _cm_mae is not None:
            avg_mae = float(_cm_mae)
        t_rollout_ms = (time.monotonic() - t_rollout_start) * 1000.0

        # Rolling: accumulate the student's WHOLE ride rollout (rank 0,
        # CPU, detached) so sample videos can show the full generated
        # sequence instead of just the last 21f window. Restarted at
        # each ride setup; kept across the ride's reset so the sampler
        # can still emit the completed ride's full video.
        if (
            self.is_main_process
            and int(getattr(
                self.model, "streaming_force_new_frame_chunks", 0)) > 0
        ):
            _nf_acc = int(info["new_frames"])
            # Viz source was IMPLICIT: flash slab when the GAN is on, raw
            # exit-rung chunk when off -- so GAN-on vs GAN-off videos compared
            # different tensors and any seam/texture metric across that
            # boundary was confounded. Make it explicit and RECORD it, so a
            # cross-arm comparison can never silently mix sources again.
            #   auto  (default, legacy): flash slab if present else chunk
            #   chunk (comparable):      always the exit-rung chunk
            _viz_mode = str(getattr(
                self.config, "rollout_viz_source", "finish")).lower()
            # Preference (rollout_viz_source): finish (DEFAULT) = finish-
            # denoised pred, inference-parity, no exit-rung lottery; falls
            # back to flash slab, then chunk. auto = legacy flash-else-chunk.
            # chunk = always exit-rung.
            _fin = info.get("finish_denoised_chunk")
            _src_acc = None
            if _viz_mode not in ("chunk", "auto") and _fin is not None:
                _src_acc = _fin
            elif _viz_mode != "chunk":
                _src_acc = info.get("flash_dmd_gan_x0")
            self._rollout_viz_src_used = (
                "finish" if _src_acc is _fin and _fin is not None
                else ("flash" if _src_acc is not None else "chunk")
            )
            _new_acc = (
                _src_acc if _src_acc is not None else chunk
            )[:, -_nf_acc:].detach().float().cpu()
            if needs_setup or getattr(self, "_rollout_video_acc", None) is None:
                self._rollout_video_acc = _new_acc
            else:
                self._rollout_video_acc = torch.cat(
                    [self._rollout_video_acc, _new_acc], dim=1)

        # ``rollout_latent_dump_dir`` (drift-attractor probe, 2026-08-20):
        # pure side-channel, default off. Appends each roll's finish-denoised
        # latent chunk (fp16, cpu) plus ride/roll indices to a per-rank .pt
        # stream for offline VAR(1)/fixed-point analysis. No training-path
        # effect.
        _dump_dir = str(getattr(self.config, "rollout_latent_dump_dir", "") or "")
        if _dump_dir:
            try:
                _lat = info.get("finish_denoised_chunk")
                if _lat is None:
                    _lat = chunk
                import os as _os
                _os.makedirs(_dump_dir, exist_ok=True)
                _rk = dist.get_rank() if dist.is_initialized() else 0
                _rec = {
                    "step": int(getattr(self, "step", -1)),
                    "ride": int(getattr(self, "_ride_counter_dump", 0)),
                    "roll": int(self._chunks_in_current_ride),
                    "lat": _lat.detach()[:, -int(info["new_frames"]):]
                        .to(torch.float16).cpu(),
                }
                _fh = getattr(self, "_lat_dump_fh", None)
                _pt = _os.path.join(_dump_dir, f"latdump_rank{_rk}.pt")
                _all = getattr(self, "_lat_dump_buf", [])
                _all.append(_rec); self._lat_dump_buf = _all
                if len(_all) % 20 == 0:
                    torch.save(_all, _pt)
                # ride ids reconstructed offline from the roll field
            except Exception as _e:
                import logging as _lg
                _lg.warning("latent dump failed: %s", _e)
        out["streaming_chunks_in_ride"] = float(self._chunks_in_current_ride)
        out["streaming_max_rolls_this_step"] = float(
            getattr(self, "_max_rolls_this_step", 0))
        out["streaming_viz_src_flash"] = float(
            1.0 if getattr(self, "_rollout_viz_src_used", "") == "flash"
            else 0.0)
        out["streaming_window_avg_mae"] = float(avg_mae)
        out["streaming_did_setup_this_step"] = 1.0 if needs_setup else 0.0
        out["streaming_window_start_chunk"] = float(self._chunks_in_current_ride)

        # ---- Toothpaste GONE gate -----------------------------------
        # If this window's frontier MAE has blown past the off-manifold bar
        # (prev-depth running avg x gone_factor), the chunk is GONE: skip
        # supervising it (DMD/GAN against a collapsed chunk drags the
        # generator off its delicate manifold) and force a reset below. DDP:
        # MAX-reduce the gone flag so ALL ranks skip Stage 4 together (its
        # gen/critic backward all-reduces stay matched) and ALL reset.
        _tp_gone_hit = False
        # Gone baseline: the previous depth's running avg once it exists,
        # ELSE the current depth's running avg (so GONE arms after the first
        # clean window at the start depth, instead of being inert until the
        # first growth — closes the early-training off-manifold hole).
        _gone_base = self._tp_prev_avg if _tp_on else None
        if _tp_on and _gone_base is None and len(self._tp_cur) >= 1:
            _gone_base = sum(self._tp_cur) / float(len(self._tp_cur))
        if (
            _tp_on
            and _gone_base is not None
            and self._chunks_in_current_ride >= 1
        ):
            _local_gone = self._toothpaste_gone(
                avg_mae, _gone_base, self._tp_gone)
            if dist.is_initialized() and dist.get_world_size() > 1:
                _gt = torch.tensor(
                    [1 if _local_gone else 0],
                    device=self.device, dtype=torch.long,
                )
                dist.all_reduce(_gt, op=dist.ReduceOp.MAX)
                _tp_gone_hit = bool(int(_gt.item()))
            else:
                _tp_gone_hit = _local_gone

        # ---- FT_v3 post-build going gate --------------------------------
        # FLAG-GATED: only when ``ftv3_postbuild_enabled``. Default OFF skips
        # this block entirely (no going flag) so the path below is
        # byte-identical. When ON we compute a RANK-UNIFORM "going" decision
        # (MAX-reduced like the GONE gate) used below to force a lockstep
        # reset; the FN tail pair is produced by the existing frontier-pair
        # re-roll at teardown.
        _ftv3_going_hit = False
        if _ftv3_on:
            # Rank-uniform going decision. ``gone`` priority is preserved by
            # the existing GONE gate above (which already MAX-reduced
            # _tp_gone_hit); here we only add the softer "going" stop. Guard
            # with min_depth so a tail (band+anchors) can always be formed.
            _local_going = (
                not _tp_gone_hit
                and self._ftv3_going_threshold > 0.0
                and self._chunks_in_current_ride >= int(
                    self._ftv3_going_min_depth)
                and avg_mae > self._ftv3_going_threshold
            )
            if dist.is_initialized() and dist.get_world_size() > 1:
                _gv = torch.tensor(
                    [1 if _local_going else 0],
                    device=self.device, dtype=torch.long,
                )
                dist.all_reduce(_gv, op=dist.ReduceOp.MAX)
                _ftv3_going_hit = bool(int(_gv.item()))
            else:
                _ftv3_going_hit = _local_going
            if _ftv3_going_hit:
                out["streaming_ftv3_going"] = 1.0

        # ---- Stage 4: train every active head (skipped if GONE) ----------
        t_train_start = time.monotonic()
        if not _tp_gone_hit:
            self._streaming_train_one_chunk(
                chunk, info, avg_mae,
                state=state, cf_state=cf_state, chunk_size=chunk_size,
                out=out,
            )
        else:
            out["streaming_tp_gone"] = 1.0
            # GONE skips Stage 4's gen backward, but the rollout forward
            # (generate_next_chunk, requires_grad=True) already armed the
            # generator's DDP reducer. Discharge it with a ZERO-ANCHOR
            # backward (zero grad -> no param update) so the next step's
            # forward can't trip "Expected to have finished reduction in the
            # prior iteration". Mirrors the FN zero-anchor pattern. GONE is
            # MAX-reduced (all ranks take this branch) and requires_grad is
            # set uniformly, so this backward is collective-matched.
            if torch.is_tensor(chunk) and chunk.requires_grad:
                (chunk.float().sum() * 0.0).backward()
        t_train_ms = (time.monotonic() - t_train_start) * 1000.0

        # ---- Stage 5: reset decision for NEXT step ------------------
        # The reset decision MUST be lockstep across ranks. If even one
        # rank resets, that rank's NEXT step opens a fresh sequence —
        # which fires an anchor forward inside ``setup_sequence``
        # (= one extra ``generate_chunk_with_cache`` call → one extra
        # ``generate_and_sync_list`` broadcast). Other ranks that DIDN'T
        # reset would skip the anchor's broadcast, leaving NCCL
        # collectives mismatched in op-count → user-code-paired
        # broadcasts cross-pair the wrong calls → hang on the next
        # mismatched-size collective. We MAX-reduce the per-rank
        # ``should_reset`` flag so ANY-rank-reset triggers ALL-ranks-
        # reset; the cost is a few resets that some ranks didn't
        # individually need (their ride state still gets thrown away),
        # but that's strictly better than a hang.
        # MAE-collapse reset (phase-2 rolling): reset the ride when the
        # trained window's MAE vs GT crosses ``streaming_mae_collapse_
        # threshold`` (0 = off). ``avg_mae`` is this step's per-rank MAE
        # on the chunk_size window (computed in Stage 3). ``min_chunks``
        # skips the gate on the first roll(s) of a ride, where the MAE
        # is dominated by the fresh-window rollout rather than drift.
        # Per-rank decision; the MAX-reduce below keeps resets lockstep.
        _mae_thr = float(getattr(
            cfg, "streaming_mae_collapse_threshold", 0.0))
        _mae_min_chunks = int(getattr(
            cfg, "streaming_mae_collapse_min_chunks", 2))
        local_mae_collapse = (
            _mae_thr > 0.0
            and self._chunks_in_current_ride >= _mae_min_chunks
            and avg_mae > _mae_thr
        )
        local_hit_cap = self._chunks_in_current_ride >= max_rolls
        local_exhausted = not self.model.can_generate_more()
        # Toothpaste GONE forces a reset (already global via the MAX-reduce
        # in the gone gate above, so it agrees across ranks).
        # FT_v3 going gate forces a reset that designates the tail (the
        # post-build/FN frontier-pair runs at the teardown below). Already
        # rank-uniform (MAX-reduced above), so it agrees across ranks; the
        # final MAX-reduce keeps it lockstep with the other reasons.
        local_should_reset = (
            local_hit_cap or local_exhausted or local_mae_collapse
            or _tp_gone_hit or _ftv3_going_hit
        )
        if dist.is_initialized() and dist.get_world_size() > 1:
            flag_t = torch.tensor(
                [1 if local_should_reset else 0],
                device=self.device, dtype=torch.long,
            )
            dist.all_reduce(flag_t, op=dist.ReduceOp.MAX)
            should_reset = bool(int(flag_t.item()))
        else:
            should_reset = local_should_reset

        # ---- Toothpaste depth growth (global, DDP-consistent) -----------
        # On a reset, record this ride's frontier MAE (MEAN-reduced across
        # ranks so _tp_depth stays identical everywhere). GONE resets are
        # failures and do NOT feed the grow average. Once a full sliding
        # window of clean samples exists at the current depth, grow depth by
        # +1 roll if the running average is <= the previous depth's average
        # x (1+tol). Unbounded — the per-ride roll_cap (ride length) is the
        # physical limit, so an over-large depth just exhausts the ride.
        if _tp_on and should_reset:
            # Same-ride window budget: count this completed window; once the
            # ride has served ``windows_per_ride`` windows, drop it so the
            # next setup loads a fresh ride. A GONE reset still counts but
            # stays on the same ride (a different window) until the budget is
            # spent — i.e. it jumps to a new offset rather than reloading.
            self._tp_windows_this_ride += 1
            if self._tp_windows_this_ride >= self._tp_windows_per_ride:
                self._tp_held_ride = None
            _fr_mae = float(avg_mae)
            if dist.is_initialized() and dist.get_world_size() > 1:
                _ft = torch.tensor(
                    [_fr_mae], device=self.device, dtype=torch.float32)
                dist.all_reduce(_ft, op=dist.ReduceOp.SUM)
                _fr_mae = float(_ft.item()) / float(dist.get_world_size())
            if not _tp_gone_hit and getattr(self, "_iqa_on", False):
                # af-all IQA-agreement ratchet (REPLACES the MAE window).
                # Local binary agreement (most-recent generated frontier
                # chunk vs last GT seed chunk), MEAN-reduced for DDP
                # lockstep, then EMA'd. EMA ≥ threshold → depth +1, EMA
                # reset (re-earn at the deeper depth). Monotonic.
                _npb = int(self.model.num_frame_per_block)
                _seed_lo = max(0, cf_state - _npb)
                _agree = self._af_all_iqa_agreement(
                    chunk, ride_window[:, _seed_lo:cf_state],
                )
                if dist.is_initialized() and dist.get_world_size() > 1:
                    _at = torch.tensor(
                        [_agree], device=self.device, dtype=torch.float32)
                    dist.all_reduce(_at, op=dist.ReduceOp.SUM)
                    _agree = float(_at.item()) / float(dist.get_world_size())
                if self._iqa_agree_ema is None:
                    self._iqa_agree_ema = _agree
                else:
                    _d = self._iqa_ema_decay
                    self._iqa_agree_ema = (
                        _d * self._iqa_agree_ema + (1.0 - _d) * _agree
                    )
                out["streaming_iqa_agree"] = float(_agree)
                out["streaming_iqa_agree_ema"] = float(self._iqa_agree_ema)
                if self._iqa_agree_ema >= self._iqa_agree_threshold:
                    self._tp_depth += 1
                    self._iqa_agree_ema = None  # re-earn at the new depth
            elif not _tp_gone_hit:
                self._tp_cur.append(_fr_mae)
                if len(self._tp_cur) > self._tp_win:
                    self._tp_cur.pop(0)
                _grow, _new_base = self._toothpaste_grow(
                    self._tp_cur, self._tp_win, self._tp_prev_avg,
                    self._tp_tol)
                if _grow:
                    self._tp_depth += 1
                    self._tp_prev_avg = _new_base
                    self._tp_cur = []   # fresh window at the new depth
            out["streaming_tp_depth"] = float(self._tp_depth)
            out["streaming_tp_gone"] = 1.0 if _tp_gone_hit else out.get(
                "streaming_tp_gone", 0.0)

        # Telemetry: log the LOCAL reason so we can tell apart "this
        # rank actually wanted to reset" from "peer triggered the
        # reset". Both ranks always agree on the boolean now.
        if should_reset:
            out["streaming_did_reset"] = 1.0
            if local_mae_collapse:
                out["streaming_reset_reason_mae_collapse"] = 1.0
            elif local_hit_cap:
                out["streaming_reset_reason_cap"] = 1.0
            elif local_exhausted:
                out["streaming_reset_reason_end_of_ride"] = 1.0
            elif _ftv3_on and _ftv3_going_hit:
                out["streaming_reset_reason_ftv3_going"] = 1.0
            else:
                out["streaming_reset_reason_peer_triggered"] = 1.0
            # FN frontier pair (phase-2 rolling): train the FN on a fresh
            # rollout1'/rollout2' pair generated AT THE RIDE'S FRONTIER,
            # right before the sequence is torn down (the pair generation
            # clobbers the pipeline KV caches, which is safe ONLY here).
            # Reset is rank-lockstep, so every rank runs exactly one FN
            # backward per reset step (DDP bucket counts stay matched;
            # bail-outs run the zero-anchor).
            if (
                bool(getattr(self.model, "fn_frontier_pairs", False))
                and getattr(self.model, "forward_noiser", None) is not None
                and str(getattr(
                    self.model, "forward_noiser_loss_mode", "mse",
                )) == "teacher_feat"
            ):
                # FT_v3 reset-boundary OOM fix. The gen+critic backwards in
                # Stage 4 already RELEASED this chunk's autograd graph, but the
                # freed activation blocks stay in the caching allocator
                # (reserved, possibly fragmented). _train_fn_frontier_pair then
                # generates TWO fresh frontier rollouts; on a reserved/
                # fragmented cache those allocations grow the high-water mark
                # and STACK toward OOM (the ~2x reset-step spike). Drop the
                # step's last grad-tensor references and return the freed
                # blocks to the allocator FIRST, so the frontier pair allocates
                # from a clean state instead of on top of the rolling step.
                # Result-preserving (no math touched — chunk/info are unused
                # past this point on a reset step) and rank-uniform (every rank
                # takes this MAX-reduced reset branch in lockstep, so the
                # empty_cache sync can't desync DDP).
                chunk = None
                info = None
                if (
                    bool(getattr(
                        self.config, "free_graph_before_frontier_pair", True))
                    and torch.cuda.is_available()
                ):
                    torch.cuda.empty_cache()
                out.update(self._train_fn_frontier_pair())
            self.model.reset_streaming_state()
            # Deferred 7chunk eval (see the gate in
            # ``_streaming_train_one_chunk``): fire ONLY at ride teardown
            # so the eval's KV-cache clobber can't truncate a live
            # multi-roll ride. ``_pending_7chunk_sample`` was set via a
            # rank-0 broadcast and ``should_reset`` is MAX-reduced, so
            # every rank takes this branch together (the eval rollout is
            # an all-ranks collective).
            if (
                getattr(self, "_pending_7chunk_sample", False)
                and bool(getattr(self, "sample_7chunk_enabled", True))
            ):
                self._pending_7chunk_sample = False
                if getattr(self.config, "holdout_eval_root", None):
                    _hr = self._holdout_eval_ride_for_step(
                        int(self.step) + 1)
                    if _hr is not None:
                        self._sample_7chunk_ride = _hr
                self._log_pred_image_7chunk_sample(int(self.step) + 1)
        else:
            out["streaming_did_reset"] = 0.0

        # Leak hunt (memory_audit_enabled): per-RANK, per-roll allocated +
        # peak, on EVERY rank (not just rank 0) so we can see the rank that
        # actually OOMs and whether its ``alloc`` grows MONOTONICALLY within
        # a ride (= a rolling leak) vs is stable (= a transient peak). Logs
        # ride_chunk so growth-vs-depth is visible.
        if (
            bool(getattr(self.config, "memory_audit_enabled", False))
            and torch.cuda.is_available()
        ):
            import sys as _sys
            _rk = dist.get_rank() if (
                dist.is_initialized() and dist.get_world_size() > 1) else 0
            print(
                f"[MEMRANK] rank={_rk} step={int(getattr(self, 'step', -1))} "
                f"ride_chunk={self._chunks_in_current_ride} "
                f"reset={int(bool(should_reset))} "
                f"alloc_gb={torch.cuda.memory_allocated()/2**30:.2f} "
                f"peak_gb={torch.cuda.max_memory_allocated()/2**30:.2f}",
                file=_sys.stderr, flush=True,
            )

        # Phase-2 rolling stderr telemetry (rank 0, only when the
        # deterministic-stride rolling mode is active): one compact line
        # per step so ride depth / MAE / resets are visible offline
        # (wandb-only metrics don't reach .err).
        if (
            self.is_main_process
            and int(getattr(
                self.model, "streaming_force_new_frame_chunks", 0)) > 0
        ):
            import sys as _sys
            if should_reset:
                _rr = (
                    "mae_collapse" if local_mae_collapse
                    else "cap" if local_hit_cap
                    else "end_of_ride" if local_exhausted
                    else "peer"
                )
            else:
                _rr = "-"
            _alloc_gb = _peakstep_gb = 0.0
            if torch.cuda.is_available():
                _alloc_gb = torch.cuda.memory_allocated() / 2**30
                _peakstep_gb = torch.cuda.max_memory_allocated() / 2**30
                # Per-gen-step peak window (rolling mode only — this
                # branch is gated on the deterministic-stride knob).
                torch.cuda.reset_peak_memory_stats()
            print(
                f"[ROLL] step={int(getattr(self, 'step', -1))} "
                f"ride_chunk={self._chunks_in_current_ride}/{max_rolls} "
                f"mae={avg_mae:.4f} reset={int(bool(should_reset))}({_rr}) "
                f"alloc={_alloc_gb:.2f} step_peak={_peakstep_gb:.2f}",
                file=_sys.stderr, flush=True,
            )

        # Wall-clock telemetry. Lets us tell apart two failure modes
        # of the prefetch design:
        #   * High setup_ms with small prefetch_wait_ms → ss_vae forward
        #     / H2D dominates. The "premature optimization to use
        #     CUDA streams" call was right; revisit if it's bad enough.
        #   * High setup_ms WITH high prefetch_wait_ms → the worker
        #     can't keep up with the reset rate (disk too slow, or
        #     ride pulls too expensive). Revisit prefetch design.
        out["streaming_step_setup_ms"] = float(t_setup_ms)
        out["streaming_prefetch_wait_ms"] = float(prefetch_wait_ms)
        out["streaming_step_rollout_ms"] = float(t_rollout_ms)
        out["streaming_step_train_ms"] = float(t_train_ms)
        out["streaming_step_wallclock_ms"] = float(
            (time.monotonic() - t_step_start) * 1000.0
        )

        return out

    def _handle_streaming_step_none(self, reason: str) -> None:
        """Account for a None-return from ``_streaming_step``. The
        outer trainer interprets None as "skip optim, increment step,
        continue", which is harmless on a single bad ride but masks a
        permanent loader fault (e.g. dataset config wrong, all rides
        too short, motion filter rejecting everything). Track
        consecutive Nones and raise after ``_MAX_CONSECUTIVE_STREAMING_NONE``
        so the run dies loudly instead of silently wasting time.
        """
        self._consecutive_streaming_step_none = (
            getattr(self, "_consecutive_streaming_step_none", 0) + 1
        )
        if self._consecutive_streaming_step_none > self._MAX_CONSECUTIVE_STREAMING_NONE:
            raise RuntimeError(
                f"_streaming_step returned None for "
                f"{self._consecutive_streaming_step_none} consecutive "
                f"calls (last reason: {reason!r}). Ride loader appears "
                f"permanently exhausted — check dataset config, "
                f"min_ride_frames, motion_ride_min_mean, and ensure "
                f"streaming_max_length / max_ride_frames align with "
                f"max_rolls_per_ride."
            )
        return None

    # ------------------------------------------------------------------
    # Per-chunk training block. Called once per outer training step
    # from ``_streaming_step``: every active head (gen, DMD critic,
    # aux action critic, R3GAN, SC-DMD) fires on every step. K=1 per
    # step makes every collective matched-count across ranks; no
    # no_sync wrapping, no loss scaling, no gating needed.
    # ------------------------------------------------------------------
    def _streaming_train_one_chunk(
        self,
        train_chunk: torch.Tensor,
        train_info: Dict[str, Any],
        train_avg_mae: float,
        *,
        state: Dict[str, Any],
        cf_state: int,
        chunk_size: int,
        out: Dict[str, Any],
    ) -> None:
        """Run gen DMD + aux gen-side + R3GAN gen-side + SC-DMD weighted
        + gen.backward + DMD critic + critic.backward on the trained
        chunk. Aux's inner action-critic optim and GAN's inner D
        optim each fire once per call (= once per outer trainer step)
        — same all-reduce count on every rank.
        """
        aux_active = (
            self.action_critic_loss_active
            and self.critic_optimizer is not None
        )
        gan_active = (
            self.gan_enabled
            and self.r3gan_disc is not None
            and self.r3gan_optimizer is not None
        )
        sc_dmd_active = bool(self.sc_dmd_enabled)

        _sample_due_now = self._video_sample_due(int(self.step) + 1)
        if _sample_due_now:
            try:
                # Prefer the post-Step-3.3.5 refined cache_pred from
                # the pipeline's ``_clean_chunk`` buffer — cleanest
                # available student state at t=60 (vs ``train_chunk``
                # which is the random-exit-rung output at a noisier,
                # variable t). Falls back to ``train_chunk`` when
                # ``flash_dmd_enabled=False`` (buffer is None).
                _clean = getattr(
                    self.model.inference_pipeline, "_clean_chunk", None,
                )
                if _clean is not None:
                    self._pending_video_latents = (
                        _clean.detach().to(torch.float32)
                    )
                else:
                    self._pending_video_latents = (
                        train_chunk.detach().to(torch.float32)
                    )
                _cs = int(self.step) + 1
                while (
                    self._sample_at_steps_pending
                    and self._sample_at_steps_pending[0] <= _cs
                ):
                    self._sample_at_steps_pending.pop(0)
                self._video_sample_due_bit = False
            except Exception as _exc:
                logging.warning(
                    "[ActionForcing] failed to stash trained chunk "
                    "for video at step=%d: %s",
                    int(self.step) + 1, _exc,
                )
                self._pending_video_latents = None
            self.model._dmd_eval_stash = {}
        else:
            self.model._dmd_eval_stash = None

        # ---- memory audit boundary 0: entry to per-chunk training ----
        self._mem_step_snaps = []
        self._mem_step_snapshot("0_entry")

        # Plumb the trainer's current step into ``info`` so the model
        # can resolve step-dependent schedules (e.g. the aux teacher's
        # piecewise-linear ``real_teacher_input_mix_gt_p`` schedule)
        # without separately threading the step through the call
        # signature.
        train_info["current_step"] = int(self.step)
        # 1-indexed roll counter within the current ride. Read by the
        # dmd_one_step "first chunk only" gate inside
        # compute_generator_loss_streaming.
        train_info["chunks_in_current_ride"] = int(
            getattr(self, "_chunks_in_current_ride", 1)
        )
        # Current effective roll cap for THIS step (after the toothpaste /
        # step-schedule adjustments in _streaming_step). Read by the
        # ``dmd_only_last_chunk_per_ride`` gate so "last roll" tracks a
        # depth that may change over training. Falls back to the static cap.
        train_info["max_rolls_this_step"] = int(
            getattr(self, "_max_rolls_this_step",
                    getattr(self.config, "max_rolls_per_ride", 1))
        )
        # Change-2: de-drift the student fake toward the GT manifold via the
        # frozen reverse noiser G BEFORE DMD scoring. Covers DMD + anti-collapse
        # + stat-anchor + the GAN's fallback fake (one de-drift at the boundary
        # since score_image=chunk=train_chunk downstream). No-op (byte-identical)
        # when reverse_noiser_dedrift_enabled=False. (Under flash_dmd the GAN's
        # gradient fake is a separate slab — de-drifted inside the model rollout
        # when reverse_noiser_dedrift_apply_to_flash=True.)
        # Preserve the RAW student rollout1 for FN/cycle training below — the
        # cycle must learn the raw rollout1->rollout2 map, NOT G's de-drifted
        # output (that would corrupt the training pairs / create a feedback
        # loop). Only the SCORING path (DMD/GAN/critic) sees the de-drift. When
        # de-drift is OFF the helper returns the SAME object => byte-identical.
        _raw_train_chunk = train_chunk
        train_chunk = self.model._dedrift_with_reverse_noiser(
            train_chunk,
            int(getattr(self.model, "reverse_noiser_dedrift_level", 1)),
        )
        # CARN pair-swap test (researcher, 2026-08-24): DEDICATED corrector
        # training backward. The legacy mse-fold rides the FAKE-SCORE inner
        # backward, which is 0 in this arm (streaming_fake_updates_per_gen=0
        # per the fold guard) -- without this block the FN loss is computed,
        # printed, and NEVER trained (the first 120-step arm proved it:
        # eval ckpts carried no FN and gen max|arm-ctl| was 4.6e-5 = kernel
        # noise). Inputs are detached, so this graph touches ONLY the FN;
        # the unconditional FN-optimizer stepping later this iteration
        # consumes the grads (DDP wrapper handles the all-reduce; the model
        # batches all pairs into ONE forward per backward).
        if (
            getattr(self.model, "fn_pair_mode", "r1_vs_r2") == "rollout_to_gt"
            and getattr(self.model, "forward_noiser", None) is not None
        ):
            _fn_w = float(getattr(
                self.model, "forward_noiser_loss_weight", 1.0))
            if _fn_w > 0.0:
                _fn_log: dict = {}
                _fn_loss = self.model._compute_fn_loss_rollout_to_gt(
                    _raw_train_chunk.detach(), train_info, _fn_log,
                )
                if _fn_loss is not None and _fn_loss.requires_grad:
                    (_fn_w * _fn_loss).backward()
                    self._carntx_step_ctr = getattr(
                        self, "_carntx_step_ctr", 0) + 1
                    if (self._carntx_step_ctr % 10 == 1
                            and self.is_main_process):
                        import sys as _sys
                        _fni = self.model.forward_noiser
                        _fni = _fni.module if hasattr(_fni, "module") else _fni
                        _wn = float(sum(
                            q.detach().float().pow(2).sum()
                            for q in _fni.parameters()).sqrt())
                        print(
                            f"[CARNTX] step={int(self.step)} "
                            f"fn_loss={float(_fn_loss.detach()):.5f} "
                            f"fn_wnorm={_wn:.3f} "
                            f"n_pairs={_fn_log.get('forward_noiser_n_pairs')}",
                            file=_sys.stderr, flush=True)
                    # Persist the corrector: the first arm saved NOTHING
                    # (eval ckpt has no FN state). Rank-0, every 50 steps
                    # and at the end.
                    if self.is_main_process and (
                        int(self.step) % 25 == 0
                        or int(self.step) >= int(getattr(
                            self.config, "max_steps", 10**9)) - 1
                    ):
                        _fni = self.model.forward_noiser
                        _fni = (_fni.module
                                if hasattr(_fni, "module") else _fni)
                        torch.save(
                            _fni.state_dict(),
                            str(self.log_dir /
                                f"fn_corrector_step{int(self.step):04d}.pt"),
                        )
        gen_loss_dmd, gen_log = self.model.compute_generator_loss_streaming(
            train_chunk, train_info,
        )
        self._mem_step_snapshot("1_after_compute_gen_loss_streaming")

        if _sample_due_now:
            eval_stash = getattr(self.model, "_dmd_eval_stash", None)
            if isinstance(eval_stash, dict) and eval_stash:
                self._pending_dmd_eval_latents = eval_stash
            # Do NOT disarm the stash here: under aux_teacher_separate_
            # backward the aux pass runs LATER in this step and writes
            # ``pred_real_lora`` / ``clean_x_aux`` into the SAME dict
            # (shared reference with _pending_dmd_eval_latents). The
            # early ``= None`` here was why those videos vanished in
            # every separate-backward run (g5+, j2). Disarmed at the
            # end of the step instead.

        out["generator_dmd_loss"] = float(gen_loss_dmd.detach().item())
        out.update({
            k: (float(v.detach().float().mean().item())
                if torch.is_tensor(v) else v)
            for k, v in gen_log.items()
            if not isinstance(v, dict)
        })
        # Per-rung bin: split the actual gen-side DMD push (= the
        # signal that backprops into the student) by the active DMD
        # timestep. ``dmd_t_mean`` is set by ``_compute_kl_grad`` and
        # Student-pred latent stats — abs-max / RMS over the chunk the
        # gen step was just trained on. If RMS climbs unboundedly or
        # abs_max saturates near the VAE's latent range, the student
        # is collapsing in latent space — a leading indicator that
        # decode-time output will go grey/blocky in the next few iters.
        # Prefer the post-Step-3.3.5 refined cache_pred (cleanest
        # student state at t=60) when available; fall back to
        # ``train_chunk`` (random-exit-rung output) when
        # ``flash_dmd_enabled=False``. More representative of
        # inference-time output quality.
        with torch.no_grad():
            _clean = getattr(
                self.model.inference_pipeline, "_clean_chunk", None,
            )
            _tc = (_clean if _clean is not None else train_chunk).detach().float()
            out["student_pred_rms"] = float(_tc.pow(2).mean().sqrt().item())
            out["student_pred_abs_max"] = float(_tc.abs().max().item())
            out["student_pred_mean"] = float(_tc.mean().item())

        generator_loss = gen_loss_dmd
        # v2-E: confidence-gated internalization — pull the RAW student toward
        # its de-drifted version so the STUDENT ALONE becomes drift-free (not
        # just G(student)). Grad to the student via _raw_train_chunk; target is
        # the de-drifted train_chunk (stop-grad inside the helper). Skipped
        # (byte-identical) when reverse_noiser_internalize_weight=0.
        if float(getattr(self.model, "reverse_noiser_internalize_weight", 0.0)) > 0.0:
            _l_int = self.model._reverse_noiser_internalize_loss(
                _raw_train_chunk, train_chunk,
            )
            generator_loss = generator_loss + _l_int
            out["reverse_noiser_internalize_loss"] = float(_l_int.detach().item())
            # Gate diagnostics (see _reverse_noiser_internalize_loss): gate_mean
            # ~= 1.0 => saturated gate (tau too large), term is ungated.
            _ri = getattr(self.model, "_internalize_resid_mean", None)
            _rg = getattr(self.model, "_internalize_gate_mean", None)
            if _ri is not None:
                out["reverse_noiser_internalize_resid_mean"] = _ri
            if _rg is not None:
                out["reverse_noiser_internalize_gate_mean"] = _rg

        # Aux / GAN / SC-DMD chunk geometry. ``state["current_length"]``
        # has already advanced past train_chunk by the time we get
        # here; the chunk's noisy half lives at
        # ``cf + (train_info[current_length] - new_frames - overlap) :
        #  ... + chunk_size``.
        noisy_start_sdn = int(
            train_info["current_length"]
            - train_info["new_frames"]
            - train_info["overlap"]
        )
        chunk_lo = cf_state + noisy_start_sdn
        chunk_hi = chunk_lo + chunk_size

        if aux_active:
            actions_chunk = state["ride_actions_window"][:, chunk_lo:chunk_hi]
            actions_for_critic = self._slice_actions_for_critic(
                actions_chunk
            ).to(train_chunk.dtype)
            ts_value = train_info.get("denoised_timestep_from", None)
            ts_int = int(ts_value) if ts_value is not None else 0
            B = train_chunk.shape[0]
            n_chunks = train_chunk.shape[1] // int(self.config.num_frame_per_block)
            chunk_t = torch.full(
                (B, n_chunks), ts_int,
                device=train_chunk.device, dtype=torch.long,
            )
            # Route action_critic to flash_dmd_gan_x0 (=
            # flash_dmd_gan_x0, the t=flash_dmd_gan_t gen forward) when
            # available. Without this, action_critic supervises the gen
            # at whatever random exit-rung was sampled this iter (75%
            # of iters land at high-noise rungs, where pred_x0 is a
            # blurry/uncertain estimate). Routing to the final-step
            # forward gives the critic a clean, structurally-coherent
            # input every iter — same trick the GAN already uses via
            # fake_lat_grad. Falls back to train_chunk when paper_
            # aligned is off or this iter didn't emit it. Gated by
            # ``gen_aux_losses_use_paper_aligned_x0`` (default True).
            ac_pred_x0 = train_chunk
            if bool(getattr(
                self.config, "gen_aux_losses_use_paper_aligned_x0", True,
            )):
                _pa_x0 = train_info.get("flash_dmd_gan_x0")
                if _pa_x0 is not None:
                    ac_pred_x0 = _pa_x0
            gen_action_loss, critic_logs, teacher_z_8d = (
                self._compute_action_critic_losses(
                    pred_x0=ac_pred_x0,
                    target_action_z=actions_for_critic,
                    chunk_t=chunk_t,
                    current_step=int(self.step),
                )
            )
            generator_loss = generator_loss + gen_action_loss
            out.update(critic_logs)
            self._mem_step_snapshot("2_after_action_critic")

            # LoRA-side action losses (action_critic z-guidance on the
            # LoRA's denoised x0 + state_probe MSE on the per-chunk z
            # the probe reads from real_score's internal taps). Both
            # graph-bearing tensors come from the model's aux pass via
            # ``_aux_teacher_tensors`` in the gen_log dict. Gradient
            # flows: action_critic → LoRA params via lora_x0; state_probe
            # → LoRA params via taps + state_probe params via the probe
            # readout. Skipped when aux pass produced no tensors (= aux
            # was skipped this iter, e.g. end-of-ride).
            aux_tensors = gen_log.get("_aux_teacher_tensors") if isinstance(
                gen_log, dict,
            ) else None
            if aux_tensors is not None:
                lora_action_loss, lora_state_probe_loss, lora_logs = (
                    self._compute_lora_action_losses(
                        lora_x0=aux_tensors.get("lora_x0"),
                        lora_state_preds=aux_tensors.get("lora_state_preds"),
                        target_action_z=actions_for_critic,
                        teacher_z_8d=teacher_z_8d,
                        chunk_t=chunk_t,
                        current_step=int(self.step),
                    )
                )
                if lora_action_loss is not None:
                    generator_loss = generator_loss + lora_action_loss
                if lora_state_probe_loss is not None:
                    generator_loss = generator_loss + lora_state_probe_loss
                out.update(lora_logs)
                self._mem_step_snapshot("3_after_lora_action")

                # Disc-borrowed regulariser on the LoRA's aux x0.
                # See ``_compute_aux_teacher_disc_losses`` for the two
                # variants (RpGAN adv vs feature matching). Both are
                # off when their weight knobs are 0 — no-op otherwise.
                # Pass this iter's ``chunk_lo`` directly (rather than
                # reading the streaming_state stash) — the stash is
                # set further down by the main GAN block and would
                # carry the PREVIOUS iter's value at this call site.
                aux_disc_loss, aux_disc_logs = (
                    self._compute_aux_teacher_disc_losses(
                        lora_x0=aux_tensors.get("lora_x0"),
                        gt_target=aux_tensors.get("gt_target"),
                        chunk_lo=int(chunk_lo),
                        current_step=int(self.step),
                    )
                )
                if aux_disc_loss is not None:
                    generator_loss = generator_loss + aux_disc_loss
                out.update(aux_disc_logs)
                self._mem_step_snapshot("3b_after_aux_disc")

        # ------------------------------------------------------------------
        # WP-PIXGAN T3-C -- the pixel G-term, computed at the OUTER level.
        #
        # It is computed HERE, before the ``gan_active`` branch, because the
        # pixel arm is specced to run with the transition GAN OFF (T3-A's own
        # stub sets ``gan_enabled=False``), and ``gen_gan_loss`` exists only
        # inside that branch. Folding the term in there and nowhere else
        # would give an arm that trains its critic every step and applies
        # ZERO gradient from it to the generator -- compiles, runs, logs
        # cheerfully, measures nothing (docs/WP_PIXGAN.md §16/§22, and the
        # reason §21 names "both endpoints, not the path between").
        #
        # So there are two application paths and the trace SAYS WHICH ONE
        # ran (``train/pix_g_in_gen_gan_loss``):
        #   gan_active  -> folded into ``gen_gan_loss`` before the A7 block,
        #                  so A7 measures it with no extra wiring (§5.5);
        #   otherwise   -> added straight to ``generator_loss`` below, with
        #                  the same UNWEIGHTED ratio published by the same
        #                  helper the A7 path calls.
        # When the arm is off the helper returns ``(None, None, {})`` before
        # touching anything: no tensor, no backward, no RNG draw, no key.
        # ------------------------------------------------------------------
        _pix_g_w = None
        _pix_g_raw = None
        _pix_g_folded = False
        # Surrogate-only mode (surrogate on, pixel gate off -- the SAM2-
        # teacher branch) also enters: the helper's surrogate branch
        # serves the G-term from the latent critic and returns before any
        # pixel-disc code. Gate-off (both false) is unchanged.
        if (bool(getattr(self, "gan_pixel_texture_enabled", False))
                or bool(getattr(self, "surrogate_critic_enabled", False))):
            _pix_g_w, _pix_g_raw, _pix_g_logs = (
                self._compute_pixel_texture_g_loss(
                    train_info, current_step=int(self.step),
                )
            )
            out.update(_pix_g_logs)

        if gan_active:
            gt_window = state["ride_latents_window"][:, chunk_lo:chunk_hi]
            # Stash the latent chunk_lo on streaming_state so
            # ``_run_distilled_disc_critic_update`` can index into the
            # cached uint8 pixel window (when present) without re-
            # plumbing the chunk geometry through 3 function signatures.
            self.model.streaming_state["last_chunk_lo_in_ride_window"] = chunk_lo
            self.model.streaming_state["last_chunk_size"] = chunk_size
            # Flash-DMD: the model stashes the per-block t=flash_dmd_gan_t
            # gen forward output on ``info["flash_dmd_gan_x0"]``; pass it
            # through so the disc and gen-side aux losses consume the
            # near-clean output as the G-side fake.
            _flash_dmd_gan_x0 = train_info.get("flash_dmd_gan_x0")
            # Reset the deferred-disc slot before the GAN compute. When
            # ladd_defer_disc_update is on, _compute_r3gan_losses stashes the
            # disc-update closure here instead of running it inline; we run it
            # after the critic backward frees the gen graph (see below).
            self._ladd_pending_disc = None
            gen_gan_loss, gan_logs = self._compute_r3gan_losses(
                pred_image=train_chunk,
                gt_latents_window=gt_window,
                current_step=int(self.step),
                flash_dmd_gan_x0=_flash_dmd_gan_x0,
            )
            # (A4 gan_grad_target_norm cap REMOVED 2026-08-23 — it used to
            # rescale gen_gan_loss here, immediately before the telemetry
            # below. ``train/gan_grad_norm`` therefore now reports the
            # UNCAPPED gen-side GAN gradient.)
            # WP-PIXGAN T3-C: the pixel G-term is folded into
            # ``gen_gan_loss`` HERE -- BEFORE the A7 block below -- so the
            # existing ``gan_grad_norm`` / ``gan_dmd_grad_ratio`` /
            # ``gan_dmd_grad_cos`` measure it with no extra wiring, which is
            # what TEXTURE_GAN_DESIGN §5.5 asks for.
            if _pix_g_w is not None:
                gen_gan_loss = gen_gan_loss + _pix_g_w
                _pix_g_folded = True

            # A3 GRAD TELEMETRY (diagnostic-only, plan 22/8 v2): per-loss
            # gradients at the LAST requires-grad generator parameter.
            # ||g_GAN||/||g_rest|| separates weak-vs-destructive; the cosine
            # detects a small GAN term that FIGHTS the DMD direction
            # (cos ~ -0.8 = destructive even at tiny loss share). Sparse
            # (every N gen steps) because it costs two extra grad passes;
            # try/except so it can never take training down.
            _tel_n = int(getattr(self.config,
                                 "gan_grad_telemetry_every", 25) or 0)
            if _tel_n > 0 and int(self.step) % _tel_n == 0:
                try:
                    _p_last = None
                    for _pn, _pp in self.model.generator.named_parameters():
                        if _pp.requires_grad:
                            _p_last = _pp
                    if _p_last is not None:
                        # T3-C: refactored onto the module-level ``grad_at``
                        # so WP-14B's LADD branch and the pixel path share ONE
                        # implementation. Outputs are byte-identical to the
                        # inline ``torch.autograd.grad(...)[0]`` pair this
                        # replaced, INCLUDING the failure regime: the old code
                        # let a graph-free loss raise out of ``autograd.grad``
                        # into the ``except`` below (-> gan_grad_telemetry_err
                        # = 1.0), whereas ``grad_at`` returns None there by
                        # design. The explicit raise preserves the recorded
                        # behaviour rather than silently converting an error
                        # regime into a silent one.
                        if not (generator_loss.requires_grad
                                and gen_gan_loss.requires_grad):
                            raise RuntimeError(
                                "A7 grad telemetry: a loss carries no "
                                "autograd graph"
                            )
                        _gr = grad_at(
                            generator_loss, _p_last, retain_graph=True)
                        _gg = grad_at(
                            gen_gan_loss, _p_last, retain_graph=True)
                        if _gr is not None and _gg is not None:
                            _nr = float(_gr.norm()); _ng = float(_gg.norm())
                            out["train/gan_grad_norm"] = _ng
                            out["train/dmd_grad_norm_shared"] = _nr
                            out["train/gan_dmd_grad_ratio"] = (
                                _ng / _nr if _nr > 0 else 0.0)
                            out["train/gan_dmd_grad_cos"] = float(
                                torch.dot(_gg, _gr)
                                / max(_ng * _nr, 1e-12))
                        # ---- T3-C: the UNWEIGHTED pixel ratio (§12 step 1).
                        # Deliberately OUTSIDE the block above, not nested in
                        # it: that block needs ``_gg`` (the grad of the whole
                        # ``gen_gan_loss``), and in the weight-free probe the
                        # pixel weight is 0 and the LADD weight may be 0 too,
                        # so ``_gg`` can legitimately be None on exactly the
                        # run this readout exists for.
                        #
                        # ``_gr`` is REUSED, not recomputed: one backward of
                        # the shared DMD term serves the A7 ratio, the A7
                        # cosine, this ratio and this cosine. Recomputing it
                        # would re-run a checkpointed teacher recompute
                        # (~11 s / ~37 GiB on the LADD path) for a number we
                        # already hold -- which is the whole reason
                        # ``grad_at`` returns VECTORS and not ratios.
                        #
                        # It is the ratio the pixel term WOULD have at weight
                        # 1.0 (``_pix_g_raw`` is pre-weight), because the
                        # ratio is linear in the weight, so
                        # ``pix_gan_weight = 0.10 / r`` solves §5.5's 5-20%
                        # band in one short probe.
                        if _pix_g_raw is not None:
                            self._pix_emit_grad_ratio(
                                out, pix_raw=_pix_g_raw, dmd_vec=_gr,
                                param=_p_last,
                            )

                        # ---- WP-14B: the UNWEIGHTED LADD ratio. See the
                        # module-level ``ladd_unweighted_ratio`` docstring
                        # for the full rationale (division vs raw-
                        # threading, the weight-free-probe limitation, why
                        # the cosine above already covers the unweighted
                        # case). Every input here is a LIVE value stashed
                        # at the moment ``_ladd_run_pair_mode`` computed
                        # it, never re-derived from cfg.
                        out.update(ladd_unweighted_ratio(
                            gg=_gg, gr=_gr,
                            mode_count=getattr(
                                self, "_ladd_last_mode_count", None),
                            total_weight=getattr(
                                self, "_ladd_last_total_weight", None),
                            stat_value=getattr(
                                self, "_ladd_last_gen_gan_stat_value", None),
                            pix_folded=_pix_g_folded,
                        ))
                except Exception as _te:
                    out["train/gan_grad_telemetry_err"] = 1.0
            generator_loss = generator_loss + gen_gan_loss
            out.update(gan_logs)
            self._mem_step_snapshot("4_after_gan_pre_backward")

        # T3-C -- the pixel G-term's SECOND application path (transition GAN
        # off). Telemetry FIRST, against ``generator_loss`` as it stands
        # BEFORE the pixel term is added -- exactly the quantity the A7 block
        # differentiates on the other path -- then the term is applied.
        if _pix_g_raw is not None:
            # WHICH PATH ACTUALLY RAN, in the trace. §21's sixth instance is
            # a knob whose effective value is assembled across files and
            # never logged; two packages then reached a wrong conclusion
            # from accurate launcher-level evidence. This is the same shape,
            # so it is logged rather than inferred from the config.
            out["train/pix_g_via_gen_gan_loss"] = 1.0 if gan_active else 0.0
            out["train/pix_g_applied"] = (
                1.0 if _pix_g_w is not None else 0.0
            )
            if not gan_active:
                # Telemetry FIRST, against ``generator_loss`` as it stands
                # BEFORE the pixel term is added -- exactly the quantity the
                # A7 block differentiates on the other path -- then apply.
                self._pix_standalone_grad_telemetry(
                    generator_loss, _pix_g_raw, out,
                    current_step=int(self.step),
                )
                if _pix_g_w is not None:
                    generator_loss = generator_loss + _pix_g_w

        # Pixel-space perceptual losses (LPIPS + pixel reconstruction).
        # Direct anti-blur + anti-drift signals on a graph-on decoded
        # frame subset. Independent of the GAN distillation chain
        # (no critic involved). Gated by their own weight knobs;
        # zero-weight = no compute. See ``_compute_pixel_perceptual_losses``.
        if (
            self.lpips_loss_weight > 0
            or self.pixel_recon_loss_weight > 0
        ):
            # Defragment the allocator before the graph-on VAE decode
            # — the decoder peaks at a large contiguous workspace
            # (~1 GB) and the gen rollout's persistent activations
            # leave only fragmented gaps. Without this, the allocation
            # can fail at step 2+ even though total free memory is
            # sufficient.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            perc_loss, perc_logs = self._compute_pixel_perceptual_losses(
                pred_image=train_chunk,
                gt_latents_window=gt_window,
                current_step=int(self.step),
            )
            generator_loss = generator_loss + perc_loss
            out.update(perc_logs)

        # A5: no-grad decoder tripwire (default OFF, texture_tripwire_every
        # = 0). Decodes a small slice of the generator's latents WITHOUT a
        # graph and logs the anisotropic texture battery on the decoded
        # pixels, so a critic gaming a latent objective into decoder
        # artefacts shows up here at training time rather than in a 60 s
        # eval two hours later. Read directly off ``ride_latents_window``
        # for the GT reference so the call does not depend on ``gt_window``
        # (which only exists on the gan_active branch).
        # Flag-checked at the CALL SITE as well as inside the helper so the
        # GT slice below is not even materialised when the tripwire is off
        # (strict zero-cost / byte-identical default).
        if int(getattr(self, "texture_tripwire_every", 0) or 0) > 0:
            self._maybe_texture_tripwire(
                train_chunk,
                out,
                gt_latents=state["ride_latents_window"][:, chunk_lo:chunk_hi],
            )

        # A22: held-out discriminator generalisation probe (default OFF,
        # disc_holdout_probe_every = 0). Scores D(decode(GT)) vs D(decode(fake))
        # on TRAIN rides and on rides reserved out of the training supply, and
        # reports both margins + both accuracies, so "D is memorising its real
        # crops" is separable from "D has learned real texture". Also runs the
        # holdout-integrity guards (a prior holdout list leaked into training
        # once before — GAN_REDESIGN A22). No grad, no collective, no state
        # mutation; see ``model/disc_holdout_probe.py`` for the interpretation
        # table and the B2 integration point.
        # WP-PIXGAN (B1 / T3-B): the pixel-texture critic's D-update loop.
        # Gated on ``gan_pixel_texture_enabled`` (default False) and checked
        # HERE as well as inside the helper, so that when the arm is off this
        # call site materialises nothing at all -- byte-identical off, with
        # no decode, no dataset read, no RNG draw and no log key. It runs
        # BEFORE the A22 probe so the probe scores the same fake tensor the
        # critic was just trained against (``register_fake_source``), which
        # is what makes the train-vs-held-out margins comparable.
        if bool(getattr(self, "gan_pixel_texture_enabled", False)):
            _pix_fake_lat, _ = self._pix_select_fake_latents(train_info)
            from model.disc_holdout_probe import register_fake_source
            register_fake_source(self, _pix_fake_lat.detach())
            # Depth tag for the exposure telemetry: the 0-based AR chunk
            # index within the current ride's rollout, derived HERE where
            # the chunk identity is known (``_chunks_in_current_ride`` was
            # incremented for THIS chunk at the top of the streaming step)
            # and threaded through as an argument rather than guessed from
            # trainer state inside the helper.
            self._maybe_run_pixel_texture_d_updates(
                train_info, out, current_step=int(self.step),
                chunk_depth=max(
                    0,
                    int(getattr(self, "_chunks_in_current_ride", 1)) - 1,
                ),
            )
            # WP-SURROGATE (B3) consumption wiring, distillation half
            # (MAIN, researcher-ordered; docs/TASK_SURROGATE_CONSUMPTION.md
            # §3.2). Spec §4.3 ordering: the critic fits the JUST-UPDATED
            # disc, hence immediately after the D-updates. Own gate inside.
            self._maybe_run_surrogate_distillation(
                train_info, out, current_step=int(self.step),
            )
        elif bool(getattr(self, "surrogate_critic_enabled", False)):
            # Surrogate-only mode (SAM2-teacher branch): no pixel D-loop
            # exists, but the distillation still must run every step. Its
            # own internal gate keeps true-off byte-identical.
            self._maybe_run_surrogate_distillation(
                train_info, out, current_step=int(self.step),
            )

        if int(getattr(self, "disc_holdout_probe_every", 0) or 0) > 0:
            from model.disc_holdout_probe import maybe_run_holdout_probe
            maybe_run_holdout_probe(self, train_chunk, out)

        if sc_dmd_active:
            sc_loss_raw, sc_logs = self.model.sc_dmd_loss(
                conditional_dict=train_info["conditional_dict"],
                clean_latent=state["ride_latents_window"][:, cf_state:],
                seed_frames=cf_state,
            )
            sc_weight = self._sc_dmd_current_weight(int(self.step))
            weighted_sc = sc_loss_raw * sc_weight
            generator_loss = generator_loss + weighted_sc
            out.update(sc_logs)
            out["sc_dmd_weight_effective"] = float(sc_weight)
            out["sc_dmd_loss_weighted"] = float(weighted_sc.detach().item())

        generator_loss = self._maybe_add_cd_loss(
            generator_loss,
            conditional_dict=train_info["conditional_dict"],
            clean_latent=state["ride_latents_window"][:, cf_state:],
            seed_frames=cf_state,
            out=out,
        )

        # Phase-LoRA ghost anchor — see ``_phase_lora_ghost_anchor``
        # for the full rationale (K+1 DDP rebuild fix).
        generator_loss = self._phase_lora_ghost_anchor(generator_loss)

        out["generator_loss"] = float(generator_loss.detach().item())
        # retain_graph=True so the critic backward can walk the shared
        # cond_dict / action_projection subgraph that both losses use.
        #
        # DDP-safe skip when generator_loss has no autograd graph.
        # Happens at production early steps (DMD/aux/GAN all gated off
        # via dmd_loss_start_step / aux_teacher_start_step /
        # gan_critic_warmup_steps) — the sum of 0-weighted losses
        # collapses to a leaf zero tensor with no grad_fn. Backward
        # would raise "element 0 of tensors does not require grad".
        # Mathematically there's no gen gradient to compute (every
        # contributor is zero-weighted), so skipping ``.backward()``
        # produces a bit-identical optimizer step (gen params get no
        # update either way). DDP-lockstep enforced via all_reduce —
        # if ANY rank has grad, ALL ranks must run backward together
        # to keep the gradient bucket reduction count matched.
        gen_has_grad_local = (
            1.0 if generator_loss.requires_grad else 0.0
        )
        if dist.is_initialized() and dist.get_world_size() > 1:
            flag = torch.tensor(
                [gen_has_grad_local],
                device=generator_loss.device, dtype=torch.float32,
            )
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            gen_should_backward = bool(flag.item() > 0.5)
        else:
            gen_should_backward = bool(gen_has_grad_local > 0.5)
        out["gen_backward_skipped"] = (
            0.0 if gen_should_backward else 1.0
        )
        if gen_should_backward:
            generator_loss.backward(retain_graph=True)
        self._mem_step_snapshot("5_after_gen_backward")

        critic_loss, critic_log = self.model.compute_critic_loss_streaming(
            train_chunk, train_info,
        )
        out["critic_loss"] = float(critic_loss.detach().item())
        out.update({
            k: (float(v.detach().float().mean().item())
                if torch.is_tensor(v) else v)
            for k, v in critic_log.items()
            if not isinstance(v, dict)
        })
        critic_loss.backward()
        self._mem_step_snapshot("6_after_critic_backward")

        # ---- TRUE two-time-scale: ``streaming_fake_updates_per_gen`` ------
        # (2026-08-22). In streaming K=1 mode the standalone critic iters
        # of the outer loop are NO-OPS (``_fwdbwd_streaming_step`` returns
        # early), so ``dfake_gen_update_ratio=5`` never meant "5 fake
        # updates per gen update" here — the true ratio was 1:1 per active
        # step and 4 of 5 outer steps did nothing. This block implements
        # the missing DMD2 two-time-scale: N EXTRA sequential fake_score
        # updates per active step (N=4 => 5 total with the outer step),
        # each a fresh ``compute_critic_loss_streaming`` call on the SAME
        # detached chunk with FRESH (epsilon, t) draws, mirroring the
        # reviewed ``teacher_cadence='fake'`` inner-loop pattern above.
        #
        # WHY inline-in-the-active-step and NOT "roll new chunks on the
        # idle non-gen iters" (considered, rejected, documented):
        #   1. The generator is FROZEN between gen updates — extra rollouts
        #      would sample the SAME policy at ~5x rollout wall-clock; the
        #      critic's tracking need is (epsilon, t) coverage, which each
        #      inner update gets fresh.
        #   2. Rolling on critic iters advances chunks_in_ride, so under
        #      ``dmd_supervise_roll_mode=random`` the per-ride target roll
        #      lands on a non-gen iter ~4/5 of the time (rides of depth <5
        #      can contain ZERO gen-iter rolls) — structurally unsupervised
        #      rides, gutting the arm this is A/B'd against.
        #   3. Ride/reset/capacity/supervise-gate logic stays byte-
        #      identical => single-variable A/B vs the no-flag arm.
        # DDP: every rank runs the same config-driven N; each inner
        # forward+backward is a complete DDP cycle through the SAME
        # critic pathway as the main step (its per-call no_sync handling
        # included; the AR term resets its local cache + RoPE memo on
        # every call), so collectives stay matched. The streaming state
        # is guaranteed alive here: teardown (reset_streaming_state) is
        # Stage 5 of THIS step, strictly after this Stage-4 block
        # returns; setup runs at the top of the NEXT step. Runs
        # regardless of ``dmd_loss_start_step`` — the critic-only warmup
        # gets the 5x cadence too (intended). Update accounting: inner
        # iter i first consumes the gradient already on the fake params
        # (the main critic backward for i=0, the previous inner backward
        # after), clip+step+zero, then computes a fresh loss+backward;
        # the LAST inner backward's gradient is consumed by the outer
        # loop's fake_optimizer block, so total sequential updates =
        # N + 1. Default 0 = byte-identical (loop not entered).
        #
        # SAFETY GUARDS (adversarial review 2026-08-22) — the flag is
        # only safe on the gtfix critic path with the FN mse-fold off;
        # both violations below are loud, config-driven (rank-uniform)
        # raises rather than silent corruption:
        #   * legacy critic path (42f AND asym both off) feeds
        #     graph-carrying cond slices whose shared action_projection
        #     subgraph is FREED by the main critic backward — a second
        #     inner backward would double-backward a freed graph.
        #   * forward_noiser mse-fold adds an FN loss into EVERY inner
        #     backward while only fake grads are zeroed between them →
        #     the single outer FN optimizer step would consume a silent
        #     (N+1)x-accumulated FN gradient (5x effective FN LR).
        _ef_raw = getattr(self.config, "streaming_fake_updates_per_gen", 0)
        _extra_fake_n = 0 if _ef_raw is None else int(_ef_raw)
        if _extra_fake_n > 0:
            if not (
                bool(getattr(self.model, "dmd_42f_enabled", False))
                or bool(getattr(
                    self.model, "dmd_asymmetric_scoring_enabled", False))
            ):
                raise RuntimeError(
                    "streaming_fake_updates_per_gen>0 requires the gtfix "
                    "critic path (dmd_42f_enabled or dmd_asymmetric_"
                    "scoring_enabled): the legacy critic path's cond "
                    "slices share the rollout graph freed by the main "
                    "critic backward (double-backward crash)."
                )
            if (
                bool(getattr(self.model, "forward_noiser_enabled", False))
                and str(getattr(
                    self.model, "forward_noiser_loss_mode", "mse"))
                != "teacher_feat"
            ):
                raise RuntimeError(
                    "streaming_fake_updates_per_gen>0 is incompatible "
                    "with the forward-noiser mse-fold (forward_noiser_"
                    "enabled=true, loss_mode!=teacher_feat): every inner "
                    "backward would accumulate FN gradient consumed by "
                    "ONE outer FN step (silent (N+1)x FN LR)."
                )
            if self.fake_optimizer is None:
                raise RuntimeError(
                    "streaming_fake_updates_per_gen>0 but fake_score_"
                    "updates_enabled=false (no fake optimizer): the "
                    "two-time-scale arm would silently run ZERO fake "
                    "updates."
                )
        if _extra_fake_n > 0:
            _extra_fired = 0
            _extra_loss_val = 0.0
            for _efi in range(_extra_fake_n):
                _fp = [
                    p for p in self.fake_optimizer.param_groups[0]["params"]
                    if p.grad is not None
                ]
                if _fp:
                    torch.nn.utils.clip_grad_norm_(
                        _fp, max_norm=self.fake_max_grad_norm,
                    )
                    self.fake_optimizer.step()
                    _extra_fired += 1
                self.fake_optimizer.zero_grad(set_to_none=True)
                _extra_loss, _extra_log = (
                    self.model.compute_critic_loss_streaming(
                        train_chunk, train_info,
                    )
                )
                _extra_loss.backward()
                # GPU->CPU sync only on the LAST inner iter (the wandb
                # log reads one value per outer step anyway; mirrors the
                # teacher_cadence loop's rel_l2 gating).
                if _efi == _extra_fake_n - 1:
                    _extra_loss_val = float(_extra_loss.detach().item())
            # Gauges for the 5x verification: steps fired this iter (expect
            # N), cumulative fake updates (expect (N+1) x active steps —
            # rises 5x faster than gen updates at N=4), last inner loss.
            out["critic_extra_updates"] = float(_extra_fired)
            self._fake_updates_total = int(getattr(
                self, "_fake_updates_total", 0)) + _extra_fired + 1
            out["fake_updates_total"] = float(self._fake_updates_total)
            out["critic_extra_loss_last"] = _extra_loss_val
            self._mem_step_snapshot("6z_after_extra_fake_updates")

        # Deferred GAN disc update (ladd_defer_disc_update): the gen +
        # critic backwards above have now freed the shared ~88 GB gen graph
        # (the gen backward used retain_graph=True; the critic backward
        # released it). Run the disc update HERE — its closure (stashed in
        # _compute_r3gan_losses) reads only detached inputs, so its result is
        # identical to running inline, but the disc backward no longer stacks
        # on the resident gen graph (the FT_v3 rolling OOM). No-op when the
        # flag is off (slot stays None) or no GAN ran this step. DDP-safe:
        # GAN gating is step-based (same on all ranks) so the closure is
        # stashed-and-run on all ranks or none — disc collectives stay matched.
        _pending_disc = getattr(self, "_ladd_pending_disc", None)
        if _pending_disc is not None:
            _pending_disc()
            self._ladd_pending_disc = None
            self._mem_step_snapshot("6a_after_deferred_disc")

        # ForwardNoiser teacher_feat training (separate FN backward, fully
        # decoupled from fake_score). Runs AFTER the gen+critic backwards so
        # the gen/critic graphs are freed before the FN's own teacher
        # forward (memory hygiene). Populates FN grads; the outer train()
        # loop clips + steps the FN optimizer. No-op unless
        # forward_noiser_loss_mode=="teacher_feat".
        if (
            getattr(self.model, "forward_noiser_enabled", False)
            and getattr(self.model, "forward_noiser_loss_mode", "mse")
            == "teacher_feat"
        ):
            # RAW rollout1 (pre-de-drift) — the cycle learns the student's true
            # rollout1->rollout2 drift, not G's own de-drifted output.
            fn_tf_logs = self._train_forward_noiser_tf(_raw_train_chunk, train_info)
            # CARN reverse test (researcher, 2026-08-24): persist the h1
            # reverse-trained corrector + a wnorm proof line, so the run is
            # never verdict-less (the first swap arm saved NOTHING and its
            # training could not be audited post-hoc). Additive, gated on the
            # h1-reverse-cumulative flag set only.
            if (
                bool(getattr(self.model, "forward_noiser_reverse", False))
                and not bool(getattr(
                    self.model, "forward_noiser_chain_levels", True))
                and self.is_main_process
                and int(self.step) % 25 == 0
            ):
                _fnm = self.model.forward_noiser
                _fnm = _fnm.module if hasattr(_fnm, "module") else _fnm
                _wn = float(sum(
                    q.detach().float().pow(2).sum()
                    for q in _fnm.parameters()).sqrt())
                import sys as _sys
                print(
                    f"[CARN-REV] step={int(self.step)} fn_wnorm={_wn:.4f} "
                    f"tf_loss={fn_tf_logs.get('train/fn_tf_loss')} "
                    f"tf_pairs={fn_tf_logs.get('train/fn_tf_pairs')}",
                    file=_sys.stderr, flush=True)
                torch.save(
                    _fnm.state_dict(),
                    str(self.log_dir /
                        f"fn_rev_step{int(self.step):04d}.pt"),
                )
            out.update(fn_tf_logs)
            self._mem_step_snapshot("6b_after_fn_tf_backward")

        # aux_teacher_separate_backward: run the aux teacher as its OWN
        # forward+backward HERE — after the gen+critic (+FN) graphs are
        # freed — so the 1.3B teacher's activation graph never coexists with
        # the GAN R1 double-backward (drops the gen-step peak ~10-15 GB).
        # The fused aux in compute_generator_loss_streaming is gated OFF in
        # this mode. run_extra_aux_pass does a fresh (eps,t) pass and
        # returns the weighted loss; the outer loop clips + steps
        # real_teacher_optimizer and runs the EMA pull. DDP-safe: the
        # all_reduce(MAX) short-ride skip lives inside
        # _compute_aux_teacher_loss_streaming, so all ranks agree on
        # whether a usable loss exists.
        if (
            getattr(self.model, "aux_teacher_separate_backward", False)
            and getattr(self, "real_teacher_optimizer", None) is not None
            and self.step >= int(
                getattr(self.config, "aux_teacher_start_step", 0)
            )
        ):
            aux_loss_sep, aux_log_sep = self.model.run_extra_aux_pass(
                train_chunk.detach(), train_info,
            )
            if aux_loss_sep is not None and aux_loss_sep.requires_grad:
                aux_loss_sep.backward()
            if isinstance(aux_log_sep, dict):
                out.update(aux_log_sep)
            self._mem_step_snapshot("6c_after_aux_separate_backward")

        # teacher_cadence='fake': run ``dfake_gen_update_ratio`` total
        # LoRA optimizer steps per outer iter (vs the default 1 under
        # 'student'). Cycle i=0 consumes the gradient already on the
        # LoRA params from the main gen backward above (which included
        # the gen-iter's aux pass). Cycles i=1..n-1 each do a fresh
        # aux forward+backward on the SAME detached chunk with new
        # (ε, t), then step and zero. After this loop LoRA grads are
        # None, so the outer loop's real_teacher_optimizer.step() at
        # ~line 2084 finds nothing to apply and short-circuits.
        # DDP-safe: every rank runs the same number of cycles; the
        # short-ride skip inside ``_compute_aux_teacher_loss_streaming``
        # is already all_reduce(MAX)-synced across ranks.
        teacher_cadence = str(
            getattr(self.config, "teacher_cadence", "student")
        ).lower()
        if (
            teacher_cadence == "fake"
            and getattr(self, "real_teacher_optimizer", None) is not None
        ):
            n_total = int(
                getattr(self.config, "dfake_gen_update_ratio", 1)
            )
            _aux_start = int(getattr(self.config, "aux_teacher_start_step", 0))
            _aux_open = self.step >= _aux_start
            if _aux_open and n_total >= 1:
                # Match the outer loop's LR-warmup schedule so every
                # in-loop step uses the same LR that the outer loop
                # would have applied. Also honour the frozen-teacher schedule
                # (linear decay to 0 by real_teacher_lr_decay_end_step, then
                # freeze) so teacher_cadence='fake' can't silently bypass the
                # freeze. decay_end=0 => byte-identical to before.
                _rt_decay_f = self.real_teacher_lr_decay_end_step
                _rt_frozen_f = (_rt_decay_f > 0 and self.step >= _rt_decay_f)
                warm_lr = self._real_teacher_base_lr
                if (
                    self.real_teacher_warmup_steps > 0
                    and self.step < self.real_teacher_warmup_steps
                ):
                    warm_lr = self._real_teacher_base_lr * (
                        (self.step + 1) / self.real_teacher_warmup_steps
                    )
                if _rt_decay_f > 0:
                    warm_lr = warm_lr * max(
                        0.0, (_rt_decay_f - self.step) / float(_rt_decay_f)
                    )
                if _rt_frozen_f:
                    warm_lr = 0.0
                    # clear the main-pass aux grad so it can't leak forward
                    self.real_teacher_optimizer.zero_grad(set_to_none=True)
                for pg in self.real_teacher_optimizer.param_groups:
                    pg["lr"] = warm_lr

                _fired = 0
                for i in range(0 if _rt_frozen_f else n_total):
                    if i > 0:
                        aux_loss_extra, aux_log_extra = (
                            self.model.run_extra_aux_pass(
                                train_chunk.detach(), train_info,
                            )
                        )
                        if (
                            aux_loss_extra is None
                            or not aux_loss_extra.requires_grad
                        ):
                            continue
                        aux_loss_extra.backward()
                    rt_params = [
                        p
                        for p in self.real_teacher_optimizer
                            .param_groups[0]["params"]
                        if p.grad is not None
                    ]
                    if rt_params:
                        torch.nn.utils.clip_grad_norm_(
                            rt_params,
                            max_norm=self.real_teacher_max_grad_norm,
                        )
                        self.real_teacher_optimizer.step()
                        # Light v14 anchor (same as the outer-loop step): pull
                        # the LoRA delta toward 0 (= v14) after EVERY LoRA update
                        # so teacher_cadence='fake' gets the anchor too (the
                        # outer-loop anchor never fires under this cadence — grads
                        # are already zeroed). Default lambda 0 => skipped.
                        if self.real_teacher_anchor_lambda > 0.0:
                            _keep = 1.0 - self.real_teacher_anchor_lambda
                            with torch.no_grad():
                                for _p in self.real_teacher_optimizer.param_groups[0]["params"]:
                                    _p.mul_(_keep)
                        # Target-network EMA pull after every LoRA
                        # update inside the teacher_cadence='fake'
                        # inner loop. With dfake_gen_update_ratio=5
                        # this fires 5x per outer step. No-op when
                        # real_score_ema_weight == 0. The rel_l2
                        # diagnostic only fires on the LAST inner
                        # iter to avoid 4× redundant per-param sums
                        # and the GPU→CPU .item() sync; wandb only
                        # reads the metric once per outer step
                        # anyway, so only the latest value matters.
                        self.model.ema_update_real_score_lora(
                            compute_rel_l2=(i == n_total - 1),
                        )
                        _fired += 1
                    self.real_teacher_optimizer.zero_grad(set_to_none=True)
                out["teacher_cadence_steps_fired"] = float(_fired)
                self._mem_step_snapshot("7_after_teacher_cadence_fake_loop")

        # pred_image_7_chunk eval video (all ranks; balanced collective).
        # Fires on the sample cadence AFTER this step's gen+critic
        # backward. Seeds the student with 7 GT-context chunks and rolls
        # 7 more, logging the full 14-chunk video on the main rank. The
        # inference rollout clobbers the KV cache + streaming state, but
        # ``max_rolls_per_ride=1`` re-seeds next step, so it is reset
        # inside the helper. Gated on a pure-step condition so every
        # rank takes the same branch (collective-safe).
        # Fire the pred_image_7_chunk eval video on the SAME cadence as
        # the existing pred_image sample (``_sample_due_now``, computed
        # at the top of this method). That signal is main-rank-only
        # (``_video_sample_due`` returns False off-rank), so broadcast it
        # from rank 0 to keep the downstream all-ranks inference rollout
        # collective-balanced. Tying to ``_sample_due_now`` (rather than
        # a pure-step gate) is necessary because this gen step only runs
        # once every ``dfake_gen_update_ratio`` steps, so a raw
        # ``step % interval`` rarely aligns with the gen cadence.
        _do7 = bool(getattr(self, "sample_7chunk_enabled", True))
        if _do7:
            _due_flag = torch.tensor(
                [1 if _sample_due_now else 0],
                device=self.device, dtype=torch.long,
            )
            if dist.is_initialized() and dist.get_world_size() > 1:
                dist.broadcast(_due_flag, src=0)
            _do7 = bool(int(_due_flag.item()) > 0)
        if self.is_main_process:
            logging.info(
                "[ActionForcing] 7chunk gate: step=%d sample_due=%s "
                "pending=%s ride_stashed=%s",
                int(self.step), bool(_sample_due_now), _do7,
                getattr(self, "_sample_7chunk_ride", None) is not None,
            )
        if _do7:
            # DEFERRED FIRE (phase-2 rolling fix): the 7chunk eval rollout
            # clobbers the pipeline KV cache + streaming state, which under
            # multi-roll rides would silently TRUNCATE the live ride at the
            # sample cadence (the "end_of_ride at every 15th step" bug).
            # Mark the sample as pending (all ranks, broadcast above keeps
            # it lockstep); ``_streaming_step`` fires it on the next RESET
            # step, right after the sequence is torn down — where the
            # clobber is free. Stationary configs (max_rolls_per_ride=1)
            # reset every step, so the eval still fires the same step.
            self._pending_7chunk_sample = True

        # Disarm the eval stash only now — after the (possibly separate-
        # backward) aux pass had its chance to add pred_real_lora /
        # clean_x_aux. See the harvest comment above.
        self.model._dmd_eval_stash = None

        self._mem_step_snapshot("8_exit")

    # ------------------------------------------------------------------
    # Streaming-mode helpers (LongLive parity).
    # ------------------------------------------------------------------
    def _streaming_setup_sequence_from_ride(
        self,
        rollout_frames: int,
        max_total_rollout_frames: int,
        cf_dmdctx: int,
        *,
        ride: Optional[Dict[str, torch.Tensor]] = None,
    ) -> bool:
        """Set up a fresh streaming sequence on this rank.

        ``ride``: when provided, use this pre-loaded ride directly (the
        K=1-per-step path passes the prefetched ride consumed via the
        executor). When None, do a synchronous ``_next_ride`` pull
        (legacy code paths).

        Per-rank ``actual_cap`` is each rank's own — no MIN-reduce.
        Under K=1-per-step there are no per-call DDP collectives that
        depend on a shared max_length across ranks. Each rank's reset
        decision (MAE / cap / end-of-ride) is independent, so ride
        lengths and rolling-window caps can diverge freely.

        Returns True on success, False if no ride was available or if
        the loaded ride is too short for one valid trained chunk.
        """
        npb = int(getattr(self.config, "num_frame_per_block", 3))
        cap = int(self.streaming_max_length)
        if cap % npb != 0:
            cap = (cap // npb) * npb

        if ride is None:
            ride = self._next_ride(rollout_frames + cf_dmdctx)
            if ride is None:
                return False
        ride_len = int(ride["latents"].shape[1])

        # A22 real-supply tripwire (default OFF). THIS is the single funnel
        # through which a dataset ride becomes discriminator "real" data:
        # everything below feeds ``streaming_state['ride_latents_window']``
        # (positional + wide real draws), ``gt_match_latents`` (matched real
        # draw) and ``self._ladd_real_ring`` (cross-ride replay real draw).
        # Tallying the ride here therefore counts EVERY held-out ride that
        # could reach the critic; the emitted ``train/dhp_leak_seen`` must
        # stay 0. Placed BEFORE any consumer so a leak is recorded even if
        # setup later bails. Zero cost when the probe is off.
        if int(getattr(self, "disc_holdout_probe_every", 0) or 0) > 0:
            from model.disc_holdout_probe import note_real_supply_ride
            note_real_supply_ride(self, ride.get("zarr_path", ""))

        # Rolling GT-headroom, reserved UP FRONT (before picking s).
        #   * dmd_42f_rolling_sup_new: +npb GT scaffold beyond the newest
        #     rolled frame (the masked OOD slot).
        #   * dmd_42f_clean_drift_enabled: forward clean places clean_x up to
        #     +npb AHEAD of the noisy window at frac=1, so the 42f clean
        #     slice (clean_lo+N) needs another +npb of real GT.
        _rolling_slack = (
            npb if bool(getattr(
                self.config, "dmd_42f_rolling_sup_new", False)) else 0
        )
        if bool(getattr(self.config, "dmd_42f_clean_drift_enabled", False)):
            # dc chunks of forward drift, not a flat npb (pre-compose bug:
            # dmd_42f_clean_drift_chunks>1 was never reserved for).
            _rolling_slack += npb * max(1, int(getattr(
                self.config, "dmd_42f_clean_drift_chunks", 1)))
        # NOTE (2026-08-20): do NOT reserve clean_match_max_drift_frames here.
        # The matcher clamps m to the available window at runtime (the
        # [42F-MATCH] range shrinks near the ride end), so budgeting the full
        # cap up front shrank max_length by 12-40 frames and cut physical roll
        # capacity to 2 -- which silently blacked out only-last supervision on
        # every ride (caps 3-6 became unreachable).
        min_new = int(getattr(self.model, "streaming_min_new_frame", npb))
        anchor_frames = int(getattr(
            self.model, "dmd_clean_x_anchor_frames", npb))

        # Per-rank motion-aware s pick. Each rank biases toward its OWN
        # ride's motion. CRITICAL: ``s_local_max`` reserves the rolling
        # headroom + anchor + min_new floor so that roll_cap clears the
        # reject threshold for EVERY s in range (the +npb covers the
        # actual_cap npb-snap). Without this, same-ride resampling could
        # draw a large s that fails the roll_cap reject on SOME ranks but
        # not others -> divergent ``return False`` -> DDP hang. With it, a
        # ride that ever set up successfully (at any s, incl. s=0) can be
        # re-seeded at any new s without a per-rank reject split.
        s_local_max = self._setup_s_local_max(
            ride_len, cf_dmdctx, cap, _rolling_slack,
            anchor_frames, min_new, npb)
        s = self._pick_motion_aware_offset(
            ride,
            s_local_max=s_local_max,
            window_lo_offset=cf_dmdctx,
            window_len=int(rollout_frames),
        )

        # Per-rank rolling-window cap: how many post-seed frames fit in this
        # rank's ride. Snap to npb. NO MIN-reduce — each rank rolls until its
        # own ride's MAE crosses or its own cap is reached.
        actual_cap = min(cap, ride_len - cf_dmdctx - s)
        if actual_cap % npb != 0:
            actual_cap = (actual_cap // npb) * npb
        roll_cap = actual_cap - _rolling_slack
        # Belt-and-braces: should be unreachable for length now that
        # s_local_max reserves the floor — only fires for a genuinely tiny
        # ride (guarded by the dataset min_ride_frames filter), and no
        # longer depends on the per-rank s, so it can't split control flow
        # across ranks on a reused ride.
        if roll_cap < anchor_frames + min_new:
            return False

        prompt_embeds = ride["prompt_embeds"]
        seed_latents = ride["latents"][:, s : s + cf_dmdctx].contiguous()
        # Window covers seed + rollout: ride_*[s : s + cf + actual_cap].
        ride_lat_window = ride["latents"][:, s : s + cf_dmdctx + actual_cap].contiguous()
        ride_act_window = ride["z_actions"][:, s : s + cf_dmdctx + actual_cap].contiguous()

        # Stash a 42-frame (14-chunk) GT slice for the pred_image_7_chunk
        # eval video (7 GT-context chunks + 7 student-rolled chunks).
        # Stashed on EVERY rank (min_ride_frames guarantees >= 42 so the
        # later all-ranks sample rollout stays collective-balanced). The
        # 7-chunk context starts at the ride origin (not the motion-aware
        # ``s``) so the slice is guaranteed in-bounds. Cheap (~tens of MB)
        # and overwritten each ride.
        _need7 = 14 * npb
        if int(ride["latents"].shape[1]) >= _need7:
            self._sample_7chunk_ride = {
                "latents": ride["latents"][:, :_need7].detach().clone(),
                "z_actions": ride["z_actions"][:, :_need7].detach().clone(),
                "prompt_embeds": prompt_embeds,
            }
        else:
            self._sample_7chunk_ride = None

        self.model.setup_sequence(
            seed_latents=seed_latents,
            ride_latents_window=ride_lat_window,
            ride_actions_window=ride_act_window,
            prompt_embeds=prompt_embeds,
            max_length=int(roll_cap),
        )

        # GT match pool for the content-matched transition GAN: the FULL
        # loaded ride (latents + co-located actions, absolute frame 0..),
        # wider than ride_latents_window so the per-fake MAE matcher has a
        # rich candidate set. Moved to the rollout device once per ride
        # (~tens of MB; ride is small). Only when matching is enabled.
        # Published for EITHER matched mode: gt_transition (2-chunk
        # transition candidates) or gt_vs_fake (single-chunk candidates).
        # Without this the gt_vs_fake matcher would find no pool and
        # silently fall back to positional pairing.
        if (
            bool(getattr(self.model, "ladd_gt_transition_match", False))
            or bool(
                getattr(self.model, "ladd_gt_vs_fake_match", None)
                or getattr(self.config, "ladd_gt_vs_fake_match", False)
            )
        ):
            _ss = self.model.streaming_state
            _dev = _ss["ride_latents_window"].device
            _gm_lat = ride["latents"]
            _gm_act = ride["z_actions"]
            if bool(getattr(self.config, "dmd_42f_rolling_sup_new", False)):
                # Phase-2 rolling: bound the matcher pool to the ACTIVE
                # ride window. Rolling rides are full-length (no random
                # truncation), and a whole-ride pool scales the per-fake
                # MAE matching transients with ride length (the 91-GiB
                # ceiling has no room for that). The window (<= cf +
                # streaming_max_length frames) is still hundreds of
                # chunks AND positionally tracks where the student
                # actually rolls. Stationary configs keep the legacy
                # whole-ride pool.
                _gm_lat = _gm_lat[:, s : s + cf_dmdctx + actual_cap]
                _gm_act = _gm_act[:, s : s + cf_dmdctx + actual_cap]
            _ss["gt_match_latents"] = _gm_lat.detach().to(_dev)
            _ss["gt_match_actions"] = _gm_act.detach().to(_dev)

            # ---- Cross-ride real replay ring (Phase B fix 1, default OFF).
            # ``ladd_real_pool_cross_ride`` (int) > 0 keeps a per-rank ring
            # buffer (deque) of GT TRANSITION slabs (2*npb frames of latents
            # + co-located actions + the ride's prompt embeds) accumulated
            # across rides. The matched gt_transition D-batch then draws
            # half of each fake's K reals uniformly from this ring (see
            # ``_match_select``), breaking the ~22-candidate ride-local
            # memorization the disc exploits (GAN_FORENSICS.md B.2).
            # Storage is CPU (detached); ~1.2MB/slab fp16 => 256 ≈ 300MB.
            # The prompt tensor is stored ONCE per ride and shared by that
            # ride's entries (entries carry a reference, not a copy).
            # Per-rank contents differ (ranks see different rides) — that
            # is data parallelism, not a collective hazard: the D-loop's
            # backward count per rank is unchanged (row COUNTS already vary
            # per rank on the matched branch).
            _ring_n = int(getattr(
                self.config, "ladd_real_pool_cross_ride",
                getattr(self.model, "ladd_real_pool_cross_ride", 0)) or 0)
            if _ring_n > 0 and int(_gm_lat.shape[0]) == 1:
                from collections import deque
                if (getattr(self, "_ladd_real_ring", None) is None
                        or self._ladd_real_ring.maxlen != _ring_n):
                    self._ladd_real_ring = deque(maxlen=_ring_n)
                _npb_r = int(getattr(self.model, "num_frame_per_block", 3))
                _pool_chunks_r = int(_gm_lat.shape[1] // _npb_r)
                _n_cand_r = _pool_chunks_r - 1  # transition starts (u, u+1)
                if _n_cand_r >= 1:
                    _push_n = min(
                        int(getattr(
                            self.config, "ladd_real_pool_push_per_ride",
                            getattr(self.model,
                                    "ladd_real_pool_push_per_ride", 8))),
                        _n_cand_r,
                    )
                    _g_r = torch.Generator(device="cpu").manual_seed(
                        int(self.step) * 7919
                        + int(getattr(self, "rank", 0)) * 104729 + 3)
                    _starts = torch.randperm(
                        _n_cand_r, generator=_g_r)[:_push_n].tolist()
                    _pe_cpu = prompt_embeds.detach().to("cpu")
                    for _u in _starts:
                        _lo_r = _u * _npb_r
                        _hi_r = (_u + 2) * _npb_r
                        _ring_entry = {
                            "lat": _gm_lat[0, _lo_r:_hi_r]
                            .detach().to("cpu").clone(),
                            "act": _gm_act[:, _lo_r:_hi_r]
                            .detach().to("cpu").clone(),
                            "pe": _pe_cpu,  # shared per ride (reference)
                        }
                        # A6: provenance tag (two small python objects, read
                        # ONLY by the diversity telemetry). Gated so the ring
                        # entry is byte-identical when the telemetry is off.
                        if bool(getattr(
                                self, "gan_real_diversity_log", False)):
                            _ring_entry["src"] = (
                                str(ride.get("zarr_path", "")), int(_u),
                            )
                        self._ladd_real_ring.append(_ring_entry)

        # Telemetry: stash the ride's absolute zarr-latent offset ``s``
        # and its motion.npy chunk offset on streaming_state so the
        # clean_x_real video logger can render frame-level "zarr_lat=N
        # motion_chunk=M" annotations alongside the action-bar overlay.
        # Read motion offset from the dataset's per-ride attrs cache so
        # we don't re-open the zarr (zero disk I/O).
        zarr_path = ride.get("zarr_path", "")
        motion_chunk_offset = 0
        attrs_by_path = getattr(self.dataset, "_attrs_by_path", None)
        if attrs_by_path is not None and zarr_path in attrs_by_path:
            try:
                from utils.zarr_dataset import _motion_chunk_offset
                motion_chunk_offset = int(
                    _motion_chunk_offset(attrs_by_path[zarr_path])
                )
            except Exception as _exc:  # pragma: no cover
                logging.warning(
                    "[ActionForcing] motion-chunk offset lookup failed "
                    "for %s: %s — falling back to 0",
                    zarr_path, _exc,
                )
        if self.model.streaming_state is not None:
            self.model.streaming_state["ride_offset_s"] = int(s)
            self.model.streaming_state["zarr_path"] = zarr_path
            self.model.streaming_state["motion_chunk_offset"] = motion_chunk_offset
            # Pre-decoded GT pixels: when ``gt_pixel_cache_root`` is set
            # AND a cached zarr exists for this ride, load the pixel
            # slice corresponding to the active ride window. Stash on
            # streaming_state so ``_run_distilled_disc_critic_update``
            # can short-circuit the live VAE decode of GT. uint8 →
            # fp32 dequantization is deferred to the consumer (so we
            # store ~5 GB compressed instead of ~20 GB fp32). Falls
            # silently to live-decode when the cache file is missing.
            self.model.streaming_state["ride_pixels_window_uint8"] = (
                self._maybe_load_cached_gt_pixels(
                    zarr_path=zarr_path,
                    ride_offset_s=int(s),
                    cf_dmdctx=cf_dmdctx,
                    actual_cap=int(actual_cap),
                )
            )
        return True

    def _maybe_load_cached_gt_pixels(
        self,
        zarr_path: str,
        ride_offset_s: int,
        cf_dmdctx: int,
        actual_cap: int,
    ) -> Optional[torch.Tensor]:
        """Try to load a pre-decoded uint8 pixel slice for the active
        ride window. Returns ``[1, F_pix, 3, H, W]`` uint8 on cache hit,
        ``None`` on miss or when the cache root isn't configured.

        Indexing contract: the precompute script (``bin/precompute_gt_
        pixels.py``) writes ``pixels[F_pix, 3, H, W]`` indexed by
        DATASET latent indices (post-_LATENT_HEAD_DROP shift), with
        4× temporal expansion via the WAN VAE dummy-leading-frame
        trick. Latent index ``i`` (in the dataset's space) decodes to
        pixel frames ``[4*i, 4*(i+1))``. The active ride window is
        ``[ride_offset_s : ride_offset_s + cf_dmdctx + actual_cap]``,
        so the pixel slice we want is ``[ride_offset_s*4 : (ride_
        offset_s + cf + cap)*4]``.
        """
        cache_root = getattr(self.config, "gt_pixel_cache_root", None)
        if not cache_root:
            return None
        if not zarr_path:
            return None
        ride_name = Path(zarr_path).stem
        cache_zarr = Path(cache_root) / f"{ride_name}.zarr"
        if not cache_zarr.exists():
            return None
        try:
            import zarr as _zarr_lib
            g = _zarr_lib.open_group(str(cache_zarr), mode="r")
            pixels_arr = g["pixels"]  # uint8 [F_pix_total, 3, H, W]
            n_lat_total = int(g.attrs.get(
                "n_latent_frames",
                pixels_arr.shape[0] // 4,
            ))
            window_lat_lo = ride_offset_s
            window_lat_hi = ride_offset_s + cf_dmdctx + actual_cap
            if window_lat_hi > n_lat_total:
                if self.is_main_process:
                    logging.warning(
                        "[gt-pixel-cache] %s window [%d:%d) extends "
                        "past cached n_lat=%d — cache miss for safety.",
                        ride_name, window_lat_lo, window_lat_hi, n_lat_total,
                    )
                return None
            pix_lo = window_lat_lo * 4
            pix_hi = window_lat_hi * 4
            arr = pixels_arr[pix_lo:pix_hi]  # uint8 numpy
            t = torch.from_numpy(arr).unsqueeze(0)  # [1, F_pix, 3, H, W]
            t = t.contiguous()
            if self.is_main_process:
                logging.info(
                    "[gt-pixel-cache] HIT %s [lat %d:%d) → pix %d:%d) "
                    "(%.2f MB uint8)",
                    ride_name, window_lat_lo, window_lat_hi,
                    pix_lo, pix_hi,
                    t.numel() / (1024 ** 2),
                )
            return t
        except Exception as exc:
            if self.is_main_process:
                logging.warning(
                    "[gt-pixel-cache] load failed for %s: %s — "
                    "falling back to live decode.",
                    ride_name, exc,
                )
            return None

    def _phase_lora_ghost_anchor(
        self, loss: torch.Tensor,
    ) -> torch.Tensor:
        """Zero-weighted touch on EVERY student phase-LoRA param.

        K+1 DDP rebuild fix: each gen step only the active adapter
        gets a REAL gradient. The dedicated ``rung_flash`` adapter's
        only real signal is the GAN adversarial loss, which is gated
        off until ``gan_disc_start_step`` — so for the first ~20 iters
        its params are NEVER marked ready in DDP's reducer. DDP defers
        its one-time bucket rebuild until every tracked param has been
        marked at least once; at the first GAN-active iter (step 21)
        rung_flash finally fires, the deferred rebuild triggers MID-
        CKPT-RECOMPUTE (the recompute's ``_pre_forward`` runs inside
        the outer backward), and the reducer dies with
        ``!unmarked_param_indices.empty() INTERNAL ASSERT``
        (reducer.cpp:2035; observed at step=21 on j5147634/5/6 and the
        j5151298+ Fix-B resubmits — both crashes exactly one step
        after gan_disc_start_step=20).

        Adding ``0.0 * sum(lora params)`` to the gen loss guarantees
        every adapter receives an (exactly-zero) grad on every
        backward from iter 1 → the rebuild completes cleanly at iter 2
        the same way it does on the no-LoRA baseline. The added
        gradient is identically zero: per-rung training signal,
        optimizer state, and rung independence are untouched
        (weight_decay=0 → idle-rung AdamW steps are pure momentum
        decay, no shrinkage, no cross-rung pollution).

        In the current no-DDP phase-LoRA mode (the student is not
        DDP-wrapped; lora grads sync manually in
        ``_all_reduce_extra_trainable_grads``), the ghost's job is to
        guarantee every lora param has a non-None grad on every iter,
        so the manual all-reduce param list is identical across ranks
        and iters — no divergence risk, no conditional sync logic.

        No-op (returns ``loss`` unchanged) when phase LoRA is off.
        """
        if not bool(
            getattr(self.config, "student_phase_lora_enabled", False)
        ):
            return loss
        gen_mod = self.model.generator.model
        if hasattr(gen_mod, "module"):
            gen_mod = gen_mod.module
        # Re-arm FIRST, then ghost over ALL lora params. The no_grad
        # dispatch sites use re_arm=False, so whichever dispatch ran
        # last (typically the Step 3.4 context_noise commit) leaves
        # only ONE adapter's params requires_grad=True via peft's
        # set_adapter side-effect. Filtering the ghost by the CURRENT
        # requires_grad would then cover a single adapter and leave
        # the other buckets unreduced — find_unused=False DDP dies
        # with "Expected to have finished reduction in the prior
        # iteration" at the next iter's first grad-on forward
        # (observed on j5169482/3/4, ~iter 1-2). Re-arming here also
        # restores the canonical all-True state before backward, which
        # both the ckpt recomputes and DDP's tracked-param set expect.
        ghost: Optional[torch.Tensor] = None
        for n, p in gen_mod.named_parameters():
            if "lora_" in n:
                p.requires_grad = True
                ghost = p.sum() if ghost is None else ghost + p.sum()
        if ghost is not None:
            loss = loss + 0.0 * ghost
        return loss

    def _fwdbwd_streaming_step(
        self,
        train_generator: bool,
        rollout_frames: int,
        max_total_rollout_frames: int,
        cf_dmdctx: int,
    ) -> Optional[Dict[str, Any]]:
        """Streaming-mode per-iter step (K=1 per step).

        The gen-iter path routes to ``_streaming_step`` which rolls
        one chunk on the persistent ride state, trains every active
        head (gen / DMD critic / aux / R3GAN / SC-DMD), and decides
        whether to reset the ride for the next step. The standalone
        critic-iter call is a no-op — the K=1 path already trained
        the critic on the same chunk.
        """
        if train_generator:
            return self._streaming_step(
                rollout_frames=rollout_frames,
                cf_dmdctx=cf_dmdctx,
            )
        # Critic-iter no-op: the K=1 _streaming_step already trained
        # the critic on this chunk. Running another critic update
        # here would either re-roll into a closed sequence or
        # re-train on a fresh ride (DMD2-decoupling violation).
        return {"streaming_critic_skipped": 1.0}

    # ------------------------------------------------------------------
    # Per-iter forward/backward.
    # ------------------------------------------------------------------
    def _fwdbwd_one_step(
        self,
        train_generator: bool,
        rollout_frames: int,
        max_total_rollout_frames: int,
        cf_dmdctx: int = 0,
    ) -> Optional[Dict[str, Any]]:
        # Streaming mode (LongLive parity) — single rolling sequence
        # spans many iters, advancing by ``new_frames`` ∈ [min_new,
        # chunk_size] frames per iter using a persistent KV cache.
        # Falls back to the legacy single-batch path below when
        # ``streaming_mode=False``.
        if self.streaming_mode:
            return self._fwdbwd_streaming_step(
                train_generator=train_generator,
                rollout_frames=rollout_frames,
                max_total_rollout_frames=max_total_rollout_frames,
                cf_dmdctx=cf_dmdctx,
            )
        # Pull next valid ride; if loader exhausted, restart. Note that
        # ``rollout_frames >= num_training_frames`` (validated in
        # ``_build_pipeline``); for long-rollout mode we slice the ride
        # to ``rollout_frames`` here, build action streams over the
        # full window, and let the pipeline + DMD model gate the
        # gradient to the last ``num_training_frames``.
        #
        # MAE-extension support: if extensions are enabled
        # (``mae_extension_max_extra_chunks > 0``,
        # ``max_total_rollout_frames > rollout_frames``), we slice the
        # ride to ``min(ride_len, max_total_rollout_frames)`` and build
        # the per-frame action conditioning streams over THAT length.
        # The model passes those streams through to the pipeline,
        # which uses ``[0:rollout_frames]`` for the baseline rollout
        # and any of ``[rollout_frames:extended_length]`` slices for
        # extension chunks (each extension consumes ``num_frame_per_
        # block`` frames). The latents themselves cover the same
        # extended window so the pipeline's MAE check has GT to
        # compare against. ``image_or_video_shape`` is held at
        # ``rollout_frames`` (BASELINE) — the noise tensor for
        # extension chunks is sampled inline by the pipeline, not
        # from the trainer.
        #
        # dmd_context window layout (single source of truth, all lengths
        # in latent frames; `cf` = ``self.model.dmd_context_clean_frames``
        # = KV-cache seed prefill, default 9; `shift` = num_frame_per_block
        # = clean/noisy shift, hardcoded to 1 chunk; `anchor` = ``shift`` =
        # leading clean_x_self anchor chunk, ONCE per ride at the FRONT):
        #
        #   ride[s : s+cf]                          → seed_latents (KV prefill at t=0)
        #   ride[s+cf : s+cf+anchor]                → leading anchor chunk (rolled, scored later)
        #   ride[s+cf+anchor : s+cf+anchor+N]       → noisy_x target window
        #   ride[s+cf : s+cf+N]                     → clean_x_GT (= same as
        #                                              the rollout's leading
        #                                              N frames; pure
        #                                              student-rolled in
        #                                              clean_x_self mode,
        #                                              GT in clean_x_GT mode)
        #
        # ``s`` is sampled uniformly from ``[0, ride_len // 2]`` per iter
        # (clamped down so ``s + cf + rollout_window ≤ ride_len``). DDP
        # ranks pick independent s values — DDP grad-averaging handles
        # the resulting cross-rank diversity.
        npb = int(getattr(self.config, "num_frame_per_block", 3))
        shift = npb  # clean/noisy shift; always 1 chunk, never configurable
        anchor = int(
            getattr(self.model, "dmd_clean_x_anchor_frames", npb)
        )  # leading clean_x_self anchor; hardcoded to npb (= shift)

        ride = self._next_ride(rollout_frames + cf_dmdctx)
        if ride is None:
            return None

        # A22 real-supply tripwire, LEGACY (streaming_mode=false) branch.
        # This path feeds ``latents`` -> ``gt_window`` -> the R3GAN/LADD real
        # side without ever touching ``_streaming_setup_sequence_from_ride``,
        # so it needs its own tally. The two hooks are mutually exclusive
        # (``streaming_mode`` picks exactly one), so no double counting.
        if int(getattr(self, "disc_holdout_probe_every", 0) or 0) > 0:
            from model.disc_holdout_probe import note_real_supply_ride
            note_real_supply_ride(self, ride.get("zarr_path", ""))

        ride_len = int(ride["latents"].shape[1])
        # Total rolled length includes the leading anchor chunk
        # (= ``anchor`` = npb frames at the front so batch-1 clean_x_self
        # is purely student-rolled, no seed dip). Once per ride.
        max_rollout_window = max_total_rollout_frames + anchor

        # Per-rank motion-aware s pick. Each rank biases toward its
        # OWN ride's motion. The MAE-extension all_reduce inside the
        # pipeline operates on a SCALAR (averaged MAE) so rank-local
        # absolute frame positions don't break the reduce — what used
        # to need broadcasting was only the rank-0 sampled ``s``,
        # which is meaningless when each rank has a different ride.
        # ``window_len = rollout_frames`` (= num_training_frames):
        # the gradient-bearing scoring window of the iter, which is
        # what we want loaded with motion.
        s_local_max = max(
            0, min(ride_len // 2, ride_len - cf_dmdctx - max_rollout_window)
        )
        s = self._pick_motion_aware_offset(
            ride,
            s_local_max=s_local_max,
            window_lo_offset=cf_dmdctx,
            window_len=int(rollout_frames),
        )

        rollout_end = s + cf_dmdctx + max_rollout_window
        rollout_window = max_rollout_window
        if rollout_window < rollout_frames:
            raise RuntimeError(
                f"Ride too short for rollout: ride_len={ride_len}, "
                f"rollout_frames={rollout_frames}, cf_dmdctx={cf_dmdctx}, "
                f"s={s}. ``_next_ride`` should have skipped this ride."
            )

        prompt_embeds = ride["prompt_embeds"]
        # Rollout-only slices (= ride[s+cf : s+cf+rollout_window]) — used
        # by aux/gan paths whose slicing predates the seed prefill design.
        latents = ride["latents"][:, s + cf_dmdctx : rollout_end]
        actions = ride["z_actions"][:, s + cf_dmdctx : rollout_end]
        # ``clean_latent_full`` covers the seed+rollout window so the
        # pipeline's MAE indexing at absolute ``current_start_frame`` works.
        clean_latent_full = ride["latents"][:, s : rollout_end]

        # Pipeline ``conditional_dict`` covers the FULL window ``ride[s :
        # s+cf+rollout_window]`` so the seed-prefill loop reads frames
        # ``[0, cf)`` and the rollout loop reads frames ``[cf, cf+rollout)``
        # of these streams (pipeline indexes by absolute current_start_frame
        # which starts at 0 for the seed and ``cf`` for the rollout). The
        # model's scoring slicer takes a ``seed_frames=cf`` offset so the
        # scorer still gets the rollout half.
        full_actions = ride["z_actions"][:, s : rollout_end]
        conditional_dict, unconditional_dict = self.model.build_action_conditional(
            prompt_embeds=prompt_embeds,
            gt_actions=full_actions,
        )

        # KV-cache prefill seed: cf frames of GT latent fed at t=0 to
        # populate the KV cache before the rollout starts.
        seed_latents = None
        if cf_dmdctx > 0:
            seed_latents = ride["latents"][:, s : s + cf_dmdctx].contiguous()

        # ``clean_x_GT`` window = the leading N frames of the rollout
        # (= ``ride[s+cf : s+cf+N]``). With the +anchor leading chunk
        # at the FRONT of the rollout, this slice is fully inside the
        # rolled-out range — clean leads noisy by ``shift=npb`` frames
        # WITHIN the rollout (clean=[s+cf : s+cf+N], noisy=[s+cf+anchor :
        # s+cf+anchor+N]) so no seed dip is needed.
        clean_context_latents = None
        clean_conditional_dict = None
        clean_unconditional_dict = None
        if cf_dmdctx > 0:
            num_training_frames = int(getattr(
                self.config, "num_training_frames", 21,
            ))
            clean_start = s + cf_dmdctx
            clean_end = clean_start + num_training_frames
            clean_context_latents = ride["latents"][:, clean_start : clean_end]
            clean_actions = ride["z_actions"][:, clean_start : clean_end]
            clean_conditional_dict, clean_unconditional_dict = (
                self.model.build_action_conditional(
                    prompt_embeds=prompt_embeds,
                    gt_actions=clean_actions,
                )
            )

        # ``image_or_video_shape`` is the TOTAL rolled shape (= scoring
        # window + leading anchor). ``_run_generator`` validates against
        # ``self.rollout_frames + self.dmd_clean_x_anchor_frames`` and
        # slices the pred into noisy_x (last N) and clean_x_self anchor
        # (first N).
        image_or_video_shape = [
            latents.shape[0], rollout_frames + anchor, *latents.shape[2:]
        ]

        if train_generator:
            aux_active = (
                self.action_critic_loss_active
                and self.critic_optimizer is not None
            )
            gan_active = (
                self.gan_enabled
                and self.r3gan_disc is not None
                and self.r3gan_optimizer is not None
            )
            sc_dmd_active = bool(self.sc_dmd_enabled)
            need_aux_artifacts = aux_active or gan_active

            if need_aux_artifacts:
                # Unified path: aux losses (action critic z-guidance)
                # and/or R3GAN both consume ``aux["pred_image"]`` so we
                # only need ONE generator forward. Each downstream
                # branch is gated separately on its own ``*_active``
                # flag, and adds its term to the running generator
                # loss before the single backward at the bottom.
                # ``z_actions_for_scoring`` = per-frame z_actions
                # (already sliced to ``action_dims`` at dataset load
                # time) for the noisy_x scoring window. Used by the
                # action-mode freeze block to skip a CoTracker forward
                # on GT — the dataset's offline pipeline already
                # encoded the same motion through the same ss_vae.
                z_actions_scoring = actions[:, -int(self.config.num_training_frames):].contiguous()
                gen_loss_dmd, gen_log_dict, aux = self.model.generator_loss(
                    image_or_video_shape=image_or_video_shape,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict,
                    clean_latent=clean_latent_full,
                    initial_latent=None,
                    return_aux=True,
                    seed_latents=seed_latents,
                    clean_x_GT=clean_context_latents,
                    clean_conditional_dict=clean_conditional_dict,
                    clean_unconditional_dict=clean_unconditional_dict,
                    z_actions_for_scoring=z_actions_scoring,
                )
                pred_image = aux["pred_image"]
                scoring_frames = int(aux["scoring_frames"])
                gen_window_end = int(aux["rollout_frames"])
                gen_window_start = gen_window_end - scoring_frames

                if self._video_sample_due(int(self.step) + 1):
                    try:
                        self._pending_video_latents = (
                            pred_image.detach().to(torch.float32)
                        )
                        _cs = int(self.step) + 1
                        while (
                            self._sample_at_steps_pending
                            and self._sample_at_steps_pending[0] <= _cs
                        ):
                            self._sample_at_steps_pending.pop(0)
                        # Consume the sticky deferral bit (see
                        # streaming-path twin for rationale).
                        self._video_sample_due_bit = False
                    except Exception as _exc:
                        logging.warning(
                            "[ActionForcing] failed to stash "
                            "pred_image for video at step=%d: %s",
                            int(self.step) + 1, _exc,
                        )
                        self._pending_video_latents = None

                merged: Dict[str, Any] = {
                    "generator_dmd_loss": float(gen_loss_dmd.detach().item()),
                }
                merged.update({
                    k: (float(v.detach().float().mean().item())
                        if torch.is_tensor(v) else v)
                    for k, v in gen_log_dict.items()
                    if not isinstance(v, dict)
                })

                generator_loss = gen_loss_dmd

                if aux_active:
                    actions_for_critic = self._slice_actions_for_critic(
                        actions[:, gen_window_start:gen_window_end]
                    ).to(pred_image.dtype)
                    ts_value = aux.get("denoised_timestep_from", None)
                    ts_int = int(ts_value) if ts_value is not None else 0
                    B = pred_image.shape[0]
                    n_chunks = pred_image.shape[1] // int(self.config.num_frame_per_block)
                    chunk_t = torch.full(
                        (B, n_chunks), ts_int,
                        device=pred_image.device, dtype=torch.long,
                    )
                    gen_action_loss, critic_logs, _teacher_z = (
                        self._compute_action_critic_losses(
                            pred_x0=pred_image,
                            target_action_z=actions_for_critic,
                            chunk_t=chunk_t,
                            current_step=int(self.step),
                        )
                    )
                    generator_loss = generator_loss + gen_action_loss
                    merged.update(critic_logs)

                # Constant-weight latent-space MAE + MSE between
                # pred_image and the GT latent window. Fires every step
                # when the weights are set, no warmup/ramp.
                if gan_active:
                    gt_window = latents[:, gen_window_start:gen_window_end]
                    gen_gan_loss, gan_logs = self._compute_r3gan_losses(
                        pred_image=pred_image,
                        gt_latents_window=gt_window,
                        current_step=int(self.step),
                    )
                    # (A4 gan_grad_target_norm cap REMOVED 2026-08-23.)
                    generator_loss = generator_loss + gen_gan_loss
                    merged.update(gan_logs)

                if sc_dmd_active:
                    sc_loss_raw, sc_logs = self.model.sc_dmd_loss(
                        conditional_dict=conditional_dict,
                        clean_latent=latents,
                        seed_frames=cf_dmdctx,
                    )
                    sc_weight = self._sc_dmd_current_weight(int(self.step))
                    weighted_sc = sc_loss_raw * sc_weight
                    generator_loss = generator_loss + weighted_sc
                    merged.update(sc_logs)
                    merged["sc_dmd_weight_effective"] = float(sc_weight)
                    merged["sc_dmd_loss_weighted"] = float(
                        weighted_sc.detach().item()
                    )

                generator_loss = self._maybe_add_cd_loss(
                    generator_loss,
                    conditional_dict=conditional_dict,
                    clean_latent=latents,
                    seed_frames=cf_dmdctx,
                    out=merged,
                )

                generator_loss = self._phase_lora_ghost_anchor(
                    generator_loss
                )
                merged["generator_loss"] = float(generator_loss.detach().item())
                generator_loss.backward()
                return merged
            else:
                z_actions_scoring = actions[:, -int(self.config.num_training_frames):].contiguous()
                generator_loss, gen_log_dict = self.model.generator_loss(
                    image_or_video_shape=image_or_video_shape,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict,
                    clean_latent=clean_latent_full,
                    initial_latent=None,
                    seed_latents=seed_latents,
                    clean_x_GT=clean_context_latents,
                    clean_conditional_dict=clean_conditional_dict,
                    clean_unconditional_dict=clean_unconditional_dict,
                    z_actions_for_scoring=z_actions_scoring,
                )
                merged_plain: Dict[str, Any] = {
                    "generator_dmd_loss": float(generator_loss.detach().item()),
                    **{
                        k: (float(v.detach().float().mean().item())
                            if torch.is_tensor(v) else v)
                        for k, v in gen_log_dict.items()
                        if not isinstance(v, dict)
                    },
                }

                if sc_dmd_active:
                    sc_loss_raw, sc_logs = self.model.sc_dmd_loss(
                        conditional_dict=conditional_dict,
                        clean_latent=latents,
                        seed_frames=cf_dmdctx,
                    )
                    sc_weight = self._sc_dmd_current_weight(int(self.step))
                    weighted_sc = sc_loss_raw * sc_weight
                    generator_loss = generator_loss + weighted_sc
                    merged_plain.update(sc_logs)
                    merged_plain["sc_dmd_weight_effective"] = float(sc_weight)
                    merged_plain["sc_dmd_loss_weighted"] = float(
                        weighted_sc.detach().item()
                    )

                generator_loss = self._maybe_add_cd_loss(
                    generator_loss,
                    conditional_dict=conditional_dict,
                    clean_latent=latents,
                    seed_frames=cf_dmdctx,
                    out=merged_plain,
                )

                generator_loss = self._phase_lora_ghost_anchor(
                    generator_loss
                )
                merged_plain["generator_loss"] = float(
                    generator_loss.detach().item()
                )
                generator_loss.backward()
                return merged_plain
        else:
            critic_loss, critic_log_dict = self.model.critic_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent_full,
                initial_latent=None,
                seed_latents=seed_latents,
                clean_x_GT=clean_context_latents,
                clean_conditional_dict=clean_conditional_dict,
            )
            critic_loss.backward()
            return {
                "critic_loss": float(critic_loss.detach().item()),
                **{
                    k: (float(v.detach().float().mean().item())
                        if torch.is_tensor(v) else v)
                    for k, v in critic_log_dict.items()
                    if not isinstance(v, dict)
                },
            }

    # ------------------------------------------------------------------
    # Ride iteration helpers.
    # ------------------------------------------------------------------
    def _fresh_ride_iter(self, epoch: int):
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)
        return iter(self.dataloader)

    def _next_ride(
        self,
        rollout_frames: int,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Pull a ride from ``self._ride_iter``, skipping rides that don't
        have enough latent frames to cover the full rollout. Auto-
        replenishes the iterator (and bumps the epoch) when exhausted
        so the curriculum order resumes from the start of the sorted
        ride list on the next pass.

        The threshold is ``rollout_frames`` (= num_training_frames in
        classic mode, larger in CF-style long-rollout mode). The
        dataset's own ``min_ride_frames`` filter should ideally
        already exclude too-short rides at index time; this is a
        belt-and-suspenders check in case the YAML's
        ``min_ride_frames`` is out of sync with ``rollout_frames``.
        """
        # Ride-level motion + min-frames filter shared with the K=1
        # prefetch path (``_next_ride_cpu_only``). See
        # ``_meta_passes_filters`` for the filter logic.
        attempts = 0
        max_attempts = 200
        ride_min_mean = float(
            getattr(self.config, "motion_ride_min_mean", 3.0)
        )
        while attempts < max_attempts:
            try:
                batch = next(self._ride_iter)
            except StopIteration:
                self._epoch += 1
                self._ride_iter = self._fresh_ride_iter(self._epoch)
                continue
            if not batch:
                attempts += 1
                continue
            meta = batch[0]
            if not self._meta_passes_filters(meta, rollout_frames):
                attempts += 1
                continue
            ride = _load_ride_tensors(
                self.dataset, meta, self.device, self.dtype,
                action_dims=self.action_dims,
                max_frames=getattr(self, "max_ride_frames", None),
                random_window=bool(
                    getattr(self.config, "max_ride_frames_random", False)
                ),
            )
            if ride is None or ride["latents"].shape[1] < rollout_frames:
                attempts += 1
                continue
            return ride
        if self.is_main_process:
            logging.warning(
                "[ActionForcing] _next_ride: gave up after %d attempts "
                "(filter threshold motion_ride_min_mean=%.3f); the "
                "dataset may have no rides with >=%d latent frames AND "
                "non-trivial motion. Consider lowering rollout_frames, "
                "raising the dataset's min_ride_frames filter, lowering "
                "motion_ride_min_mean, or setting motion_skip_dead_rides=false.",
                max_attempts, ride_min_mean, rollout_frames,
            )
        return None

    # ------------------------------------------------------------------
    # Override grad-uniformity (parent expects multi-slot path).
    # The parent's ``_all_reduce_extra_trainable_grads`` uses dist
    # reductions on action_projection / action_token_projection grads;
    # that part still applies here. We reuse it as-is.
    # ------------------------------------------------------------------


def _load_config_with_extends(path: str) -> "OmegaConf":
    """Load a yaml config, recursively resolving a top-level
    ``_extends: <path>`` directive.

    Behavior:
      * If the yaml has ``_extends: <other.yaml>`` at top level, the
        extended file is loaded first (recursively) and the current
        file's keys merge ON TOP via ``OmegaConf.merge``. This lets a
        variant config carry only the deltas from a base.
      * ``_extends`` is stripped from the final config so consumers
        never see it.
      * The path is resolved relative to the directory of the file
        containing the directive (Hydra-style).
      * Cycles are detected and raise.

    Used by the action_forcing config family to collapse
    near-identical 600-line yamls into a single base + thin overlays.
    """
    # ``stack`` tracks the active recursion path (push on entry, pop on
    # exit). Catches actual cycles A->B->...->A without false-positive-
    # ing on a future diamond pattern (A->B and A->C both extending D
    # — D is touched twice along non-cyclic branches).
    stack: list = []

    def _load(p: str) -> "OmegaConf":
        ap = os.path.abspath(p)
        if ap in stack:
            raise RuntimeError(
                f"Circular _extends in config chain: "
                f"{' -> '.join(stack + [ap])}"
            )
        stack.append(ap)
        try:
            cfg = OmegaConf.load(ap)
            # ``_extends`` is optional and consumed here.
            ext = None
            if isinstance(cfg, type(OmegaConf.create({}))) and "_extends" in cfg:
                ext = cfg["_extends"]
                del cfg["_extends"]
            if ext is None:
                return cfg
            ext_path = os.path.normpath(
                os.path.join(os.path.dirname(ap), str(ext))
            )
            base = _load(ext_path)
            return OmegaConf.merge(base, cfg)
        finally:
            stack.pop()

    return _load(path)


# ----------------------------------------------------------------------
# A18(ii) — dotlist-override guard.
#
# ``OmegaConf.from_dotlist`` + ``OmegaConf.merge`` accept ANY key: there is
# no schema, no allowlist and no typo check, so a stale or misspelled
# ``key=value`` on the launch command line merges into the config in
# perfect silence and is then never read. This has already cost three
# experiment readings (``ladd_gt_transition_action_blind`` — read only off
# ``self.model``, which never binds it from args, so every "action-blind"
# arm actually ran action-CONDITIONED; ``ladd_gt_transition_cross_equalize``
# and ``ladd_r1_once_per_step`` — zero code references at all).
#
# The guard below re-derives the allowlist FROM THE SOURCE at startup
# rather than hard-coding one (a hard-coded list would go stale in exactly
# the way the flags it protects did). It walks the repo's ``.py`` files and
# records, for every identifier starting with one of
# ``_OVERRIDE_GUARD_PREFIXES``, whether it is ever read *off a config-like
# object* — ``getattr(self.config, "k", ...)``, ``args.k``, ``cfg.get("k")``,
# ``OmegaConf.select(cfg, "k")``. A key read only off some OTHER object
# (e.g. ``getattr(self.model, "k", False)``) is NOT counted as sourced,
# because that is precisely the ``action_blind`` failure: the read succeeds,
# returns the default, and the override never arrives.
#
# Fail-loud, not fail-closed: by default this only emits ``logging.error``.
# Set ``strict_override_keys=true`` to escalate to a hard raise. Every
# failure mode of the scan itself (unreadable tree, unexpected layout,
# implausibly small result) degrades to a silent no-op, so the guard can
# never take a training run down on its own.
# ----------------------------------------------------------------------
# D14: `disc_holdout_probe_` is guarded too. A typo'd
# `disc_holdout_probe_evrey=50` is otherwise accepted silently by
# OmegaConf.from_dotlist and the probe simply never fires -- the exact
# failure class this guard exists for, on the one instrument whose job is
# to catch contaminated experiments.
_OVERRIDE_GUARD_PREFIXES = ("gan_", "ladd_", "pix_", "disc_holdout_probe_")
_OVERRIDE_GUARD_SKIP_DIRS = frozenset({
    "wandb", "logs", "eval", "checkpoints", "__pycache__", "node_modules",
    "venv", "data", "docs", "outputs", "results", "analysis",
})
# Receiver names that mean "this is the run config": a read off one of
# these is a genuine consumer of the override.
_OVERRIDE_GUARD_CONFIG_RECEIVERS = frozenset({
    "args", "config", "cfg", "conf", "opts", "hparams",
})
# A18-D3 (2026-08-23) -- EXPLICIT REGISTRATION HOOK.
#
# A module that reads its knobs through a *variable* key (the
# ``getattr(getattr(trainer, "config", None), key, None)`` idiom in
# ``model/disc_holdout_probe.py``, or ``model/pixel_texture_disc.py``'s
# ``PIX_DEFAULTS`` table) has NO literal ``getattr(config, "<key>", ...)``
# site for the textual scan to find. Before this hook every such knob was
# reported as "merged silently, no effect" -- and under
# ``strict_override_keys=true`` that FALSE POSITIVE kills the job before the
# trainer is even constructed. ``pix_`` is a guarded prefix reserved for the
# B2 pixel-texture critic, which is specced with ~15 ``pix_*`` knobs, so the
# blast radius was an entire planned arm.
#
# Any module may therefore declare its config keys by assigning a
# module-level list/tuple/set/dict of string literals to one of these names.
# The scan below also harvests, automatically:
#   * nested config receivers -- ``getattr(getattr(t, "config", None), "k", d)``;
#   * ``DEFAULTS``-style tables -- every module-level all-string-keyed dict
#     literal in a module that reads config with a NON-literal key.
_OVERRIDE_GUARD_REGISTRY_NAMES = frozenset({
    "CONFIG_KEYS", "OVERRIDE_GUARD_KEYS", "_OVERRIDE_GUARD_KEYS",
    "CONFIG_OVERRIDE_KEYS",
})

# WP-PIXGAN (B1/T3) — EXPLICIT ``pix_*`` REGISTRATION.
#
# MEASURED, not assumed (T3-A ran the scan above against
# ``model/pixel_texture_disc.py`` and it returned the EMPTY SET): the
# ``DEFAULTS``-style auto-harvest does NOT pick up ``PIX_DEFAULTS``. The
# harvest is gated on ``reads_variable_key`` — the module must itself read
# config through a non-literal key — and ``model/pixel_texture_disc.py``
# reads config *not at all* by design ("This module never reads yaml";
# every ``pix_*`` value is a constructor or function argument). So the
# harvest's precondition can never be met there and the A18-D3 comment's
# claim, while true of ``model/disc_holdout_probe.py``, does not cover this
# module. Hence this explicit table, which the scan always trusts.
#
# Mirrors ``model/pixel_texture_disc.py::PIX_DEFAULTS`` (string literals,
# because the harvest is an AST pass and cannot import). Keep in sync.
#
# TRADE-OFF, stated so the next chunk can act on it: a key listed here is
# allowlisted whether or not anything reads it, so as the later T3 chunks
# land real ``getattr(config, "pix_...", ...)`` sites they should DELETE the
# corresponding entries from this list and let the scan find them for real.
# The list exists because an unsourced ``pix_*`` override is a FALSE
# POSITIVE that kills the job before the trainer is constructed under
# ``strict_override_keys=true``, and the blast radius is the whole arm.
#
# T3-B PRUNE (2026-08-23): the eleven supply/D-loop keys that were listed
# here -- pix_crop_lat, pix_frames_per_crop, pix_crops_per_step,
# pix_band_count, pix_reals_per_fake, pix_loss_form, pix_r1_gamma,
# pix_r1_sigma, pix_r1_every_n, pix_r1_num_samples, pix_decode_border_trim
# -- have been DELETED because T3-B now reads every one of them at a
# literal ``getattr(cfg, "<key>", ...)`` site in this file. Verified by
# running the scan with this table emptied. Keeping them registered would
# allowlist them whether or not anything read them, which is exactly the
# regression-masking the T3-A comment above warned about.
#
# T3-C PRUNE (2026-08-23): the table is now EMPTY, and that is the finished
# state, not an oversight. The last entry, ``pix_gan_weight``, was deleted
# because T3-C reads it at literal ``getattr(cfg, "pix_gan_weight", None)``
# sites (``_pix_gen_weight``, and the resolved-config echo), so the scan
# sources it for real. VERIFIED by running ``_scan_config_sourced_keys`` with
# this table emptied: it returns ``pix_gan_weight``.
#
# Emptying it is the POINT, not a tidy-up. A registered key is allowlisted
# whether or not anything reads it, so leaving entries here would mask
# exactly the regression the guard exists to catch: if a future edit deletes
# a read site, the override must START being reported as unsourced, and a
# stale registration would keep certifying it as fine.
#
# The list stays DEFINED (it is one of ``_OVERRIDE_GUARD_REGISTRY_NAMES``)
# so a module that reads config through a VARIABLE key can register again if
# one ever appears. There is no ``pix_r2_*`` entry and never will be
# (R2 is deleted -- WP_PIXGAN section 2).
_OVERRIDE_GUARD_KEYS = []


def _ast_config_sourced_keys(prefixes, src):
    """AST half of the override-guard scan (A18-D3).

    Returns the set of prefixed keys this source reads off a config-like
    object, covering the three idioms the textual scan structurally cannot
    see. Any parse failure returns the empty set -- the textual scan still
    runs, so this can only ever ADD keys, never remove them.
    """
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError, RecursionError):
        return set()

    def _pref(name):
        return isinstance(name, str) and name.startswith(tuple(prefixes))

    def _is_cfg_expr(node, depth=0):
        """Is ``node`` an expression that evaluates to the run config?"""
        if depth > 4 or node is None:
            return False
        if isinstance(node, ast.Name):
            return node.id in _OVERRIDE_GUARD_CONFIG_RECEIVERS
        if isinstance(node, ast.Attribute):
            # ``self.config`` / ``trainer.cfg`` / ``self.args``
            return node.attr in _OVERRIDE_GUARD_CONFIG_RECEIVERS
        if isinstance(node, ast.Call):
            f = node.func
            # ``getattr(<anything>, "config", <default>)`` -> the config.
            if isinstance(f, ast.Name) and f.id == "getattr" and len(node.args) >= 2:
                a1 = node.args[1]
                if (isinstance(a1, ast.Constant)
                        and isinstance(a1.value, str)
                        and a1.value in _OVERRIDE_GUARD_CONFIG_RECEIVERS):
                    return True
            # ``x.get("config")`` -- same shape, same meaning.
            if (isinstance(f, ast.Attribute) and f.attr == "get"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                    and node.args[0].value
                    in _OVERRIDE_GUARD_CONFIG_RECEIVERS):
                return True
        if isinstance(node, ast.BoolOp):      # ``cfg or {}``
            return any(_is_cfg_expr(v, depth + 1) for v in node.values)
        if isinstance(node, ast.IfExp):
            return (_is_cfg_expr(node.body, depth + 1)
                    or _is_cfg_expr(node.orelse, depth + 1))
        return False

    found = set()
    reads_variable_key = False

    for node in ast.walk(tree):
        # ``cfg.key`` / ``self.config.key``
        if isinstance(node, ast.Attribute):
            if _pref(node.attr) and _is_cfg_expr(node.value):
                found.add(node.attr)
        # ``cfg["key"]``
        elif isinstance(node, ast.Subscript):
            sl = node.slice
            if (isinstance(sl, ast.Constant) and isinstance(sl.value, str)
                    and _pref(sl.value) and _is_cfg_expr(node.value)):
                found.add(sl.value)
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id == "getattr" and len(node.args) >= 2:
                recv, key = node.args[0], node.args[1]
                if _is_cfg_expr(recv):
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        if _pref(key.value):
                            found.add(key.value)
                    else:
                        # ``getattr(config, key, default)`` -- key is a
                        # variable. This module dispatches its knobs through
                        # a table; enable the DEFAULTS harvest below.
                        reads_variable_key = True
            elif isinstance(f, ast.Attribute) and f.attr == "get":
                if node.args and _is_cfg_expr(f.value):
                    key = node.args[0]
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        if _pref(key.value):
                            found.add(key.value)
                    else:
                        reads_variable_key = True
            elif (isinstance(f, ast.Attribute) and f.attr == "select"
                    and isinstance(f.value, ast.Name)
                    and f.value.id == "OmegaConf" and len(node.args) >= 2):
                key = node.args[1]
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    if _pref(key.value):
                        found.add(key.value)

    # ---- module-level tables -------------------------------------------
    dict_tables = []
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target]
            value = node.value
        else:
            continue
        if value is None:
            continue
        names = {t.id for t in targets}
        literals = None
        if isinstance(value, ast.Dict):
            ks = value.keys
            if ks and all(isinstance(k, ast.Constant)
                          and isinstance(k.value, str) for k in ks):
                literals = [k.value for k in ks]
        elif isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            els = value.elts
            if els and all(isinstance(e, ast.Constant)
                           and isinstance(e.value, str) for e in els):
                literals = [e.value for e in els]
        if literals is None:
            continue
        if names & _OVERRIDE_GUARD_REGISTRY_NAMES:
            # Explicit registration -- always trusted.
            found.update(k for k in literals if _pref(k))
        elif isinstance(value, ast.Dict):
            dict_tables.append(literals)

    if reads_variable_key:
        # ``DEFAULTS``-style harvest. Only when the module demonstrably reads
        # config with a variable key, so a plain lookup table in an unrelated
        # module is never mistaken for a config consumer.
        for literals in dict_tables:
            found.update(k for k in literals if _pref(k))

    return found


def _scan_config_sourced_keys(prefixes, root):
    """Return the set of prefixed config keys the source actually reads
    off a config-like object, or ``None`` if the scan could not run."""
    import re

    tok = re.compile(
        r"(?:" + "|".join(re.escape(p) for p in prefixes) + r")[A-Za-z0-9_]+"
    )
    re_comment = re.compile(r"^\s*#")
    re_receiver = re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)\.$")
    re_getattr = re.compile(r"getattr\( ?([^,()]+?) ?, ?[\"']$")
    re_dictget = re.compile(r"([A-Za-z_][A-Za-z0-9_.]*) ?\. ?get\( ?[\"']$")
    re_subscript = re.compile(r"([A-Za-z_][A-Za-z0-9_.]*) ?\[ ?[\"']$")
    re_select = re.compile(r"OmegaConf\.select\( ?([^,()]+?) ?, ?[\"']$")

    def _is_config_receiver(expr):
        if not expr:
            return False
        tail = expr.strip().split(".")[-1].strip("\"'[] ")
        return tail in _OVERRIDE_GUARD_CONFIG_RECEIVERS

    sourced, n_files = set(), 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in _OVERRIDE_GUARD_SKIP_DIRS and not d.startswith(".")
        ]
        for name in filenames:
            if not name.endswith(".py"):
                continue
            try:
                with open(os.path.join(dirpath, name),
                          encoding="utf-8", errors="ignore") as fh:
                    src = fh.read()
            except OSError:
                continue
            n_files += 1
            if not any(p in src for p in prefixes):
                continue
            # Drop whole-line comments (so a key merely DISCUSSED in a
            # doc block is not mistaken for a consumer), then collapse
            # whitespace so multi-line ``getattr(\n  self.config,\n
            # "key",\n  default)`` calls match the same patterns as
            # single-line ones.
            body = " ".join(
                ln for ln in src.split("\n") if not re_comment.match(ln)
            )
            body = re.sub(r"\s+", " ", body)
            for m in tok.finditer(body):
                key = m.group(0)
                if key in sourced:
                    continue
                before = body[max(0, m.start() - 160):m.start()]
                recv = None
                if before.endswith("."):
                    mm = re_receiver.search(before)
                    recv = mm.group(1) if mm else None
                elif before.endswith(("\"", "'")):
                    for rx in (re_getattr, re_dictget, re_subscript, re_select):
                        mm = rx.search(before)
                        if mm:
                            recv = mm.group(1)
                            break
                if _is_config_receiver(recv):
                    sourced.add(key)
            # A18-D3: AST pass. Strictly ADDITIVE -- it can only widen the
            # allowlist, so a key the textual scan already found stays found
            # and no new false NEGATIVE is possible from a parse failure.
            # NOT pre-filtered on a cheap substring test: a gate that misses
            # a variant spelling (nested getattr split across lines, a
            # differently-named table) silently restores the FALSE POSITIVE
            # this fix exists to remove, and under ``strict_override_keys``
            # that is a dead job. Measured cost of the AST pass over the whole
            # repo is ~14 s on top of the ~50 s the textual walk already
            # costs, once, at startup.
            sourced |= _ast_config_sourced_keys(prefixes, src)
    if n_files < 10:
        # Repo layout is not what we expect -> do not pretend to know.
        return None
    return sourced


class OverrideGuardError(RuntimeError):
    """The ONLY exception ``_warn_ignored_override_keys`` may escape with.

    A18-D8 (2026-08-23): the guard used to end with ``except RuntimeError:
    raise``, which re-raised EVERY RuntimeError the scan itself could throw --
    ``RecursionError`` is a RuntimeError subclass, and ``ast``/``os.walk`` on a
    pathological tree can raise it. That directly contradicted the guard's own
    contract ("every failure mode of the scan itself degrades to a silent
    no-op, so the guard can never take a training run down on its own"): a
    scan bug became a dead job at startup. The strict verdict is now recorded
    in a local and raised AFTER the try block, so nothing raised from INSIDE
    the try can ever escape.

    Subclasses RuntimeError so any existing ``except RuntimeError`` around the
    launcher keeps working.
    """


def _warn_ignored_override_keys(overrides, cfg):
    """Report ``key=value`` overrides that the code never actually reads.

    ``overrides`` is the raw ``--override`` dotlist. Only keys starting with
    ``_OVERRIDE_GUARD_PREFIXES`` are checked. Emits ``logging.error``; if
    ``strict_override_keys`` is true in ``cfg``, raises
    ``OverrideGuardError`` instead -- and ONLY that, never an incidental
    RuntimeError from the scan (A18-D8).
    """
    _strict_failure = None
    try:
        requested = []
        for item in overrides:
            key = str(item).split("=", 1)[0].strip()
            # dotlist keys may be nested (``a.b=c``); the leaf is what a
            # ``getattr(config, ...)`` consumer would name.
            leaf = key.split(".")[-1]
            if leaf.startswith(_OVERRIDE_GUARD_PREFIXES):
                requested.append((key, leaf))
        if not requested:
            return
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sourced = _scan_config_sourced_keys(_OVERRIDE_GUARD_PREFIXES, root)
        if not sourced:
            return
        ignored = sorted({k for k, leaf in requested if leaf not in sourced})
        if not ignored:
            return
        strict = bool(OmegaConf.select(cfg, "strict_override_keys") or False)
        msg = (
            "[override-guard] %d override key(s) are NOT read off the config "
            "anywhere in the source tree -- they merged silently and have NO "
            "effect on this run: %s. (Scanned %s for reads of the form "
            "getattr(config/args, \"<key>\", ...) / config.<key> / "
            "cfg.get(\"<key>\"). A key read only off some other object, e.g. "
            "getattr(self.model, \"<key>\"), does NOT count -- that is the "
            "ladd_gt_transition_action_blind failure mode.) Fix the launch "
            "script or wire the key up; set strict_override_keys=true to make "
            "this a hard error."
        ) % (len(ignored), ", ".join(ignored), root)
        if strict:
            # A18-D8: record, do NOT raise from inside the try -- see
            # ``OverrideGuardError``.
            _strict_failure = msg
        elif str(os.environ.get("RANK", "0")) == "0":
            logging.error(msg)
    except Exception as exc:  # never take a run down on the guard's account
        logging.debug("[override-guard] skipped (%s: %s)", type(exc).__name__, exc)
    if _strict_failure is not None:
        raise OverrideGuardError(_strict_failure)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Phase-1 Action-Forcing DMD trainer"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--override", type=str, nargs="*", default=[])
    args = parser.parse_args()

    base = _load_config_with_extends(args.config)
    if args.override:
        override = OmegaConf.from_dotlist(list(args.override))
        cfg = OmegaConf.merge(base, override)
        # A18(ii): dotlist merge accepts unknown keys silently -- flag
        # any gan_/ladd_/pix_ override the source never reads.
        _warn_ignored_override_keys(list(args.override), cfg)
    else:
        cfg = base

    trainer = ActionForcingDMDTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
