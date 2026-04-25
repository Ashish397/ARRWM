# pyright: reportGeneralTypeIssues=false
"""DMD2 model for Phase-1 Action-Forcing training (NO STAIRCASE).

This model is a thin, action-aware port of
``Causal-Forcing/model/dmd.py`` onto our infrastructure. The core
training mechanics — generator_loss / critic_loss / _compute_kl_grad /
compute_distribution_matching_loss / _run_generator — mirror CF
1:1 with two action-aware additions and the DMD2 upgrades retained
from the staircase variant:

  * ``_run_generator`` builds a per-frame action conditioning dict
    (Stream A modulation + Stream B tokens) from the input ``gt_actions``
    and threads it through to the Action-Forcing training pipeline.
  * Real and fake scorers receive the same per-frame action conditioning
    on the *student's* predicted 21-frame video — no GT context, no KV
    cache, fully symmetric. The grad direction is purely
    ``(pred_fake - pred_real)`` on a noisy version of the student's
    output (CF parity).
  * ODE checkpoint loading + v14 LoRA loading + generator->fake_score
    mirroring + non-strict load with optional strict gate (DMD2 retained
    upgrades from the staircase model).
  * Auxiliary action heads (``action_critic`` + ``state_probe``) are
    instantiated and loaded from the ODE checkpoint so the existing
    trainer plumbing (`action_token_projection`, etc.) keeps working,
    but their AUX LOSSES ARE NOT WIRED IN by this model — the user's
    plan for Phase-1 Action-Forcing is "DMD almost exactly like CF" so we keep
    only the DMD path. The heads remain frozen and unused by the
    losses; they live on the model only so the ODE checkpoint loads
    cleanly and downstream tooling that introspects `.action_critic`
    / `.state_probe` does not break.

The Action-Forcing training pipeline
``pipeline.action_forcing_training.ActionForcingTrainingPipeline``
runs the truncated random-exit denoise loop with persistent KV cache
and exit-flag gradient gating; ``_run_generator`` matches CF's
"slice last N frames" and ``start_gradient_frame_index`` semantics.

CF-style long rollouts (``rollout_frames > num_training_frames``):
the trainer rolls out ``rollout_frames`` total but only the LAST
``num_training_frames`` frames carry gradient (the gate inside
``ActionForcingTrainingPipeline.inference_with_trajectory`` runs the leading
``rollout_frames - num_training_frames`` exit-flag forwards under
``no_grad`` to warm the KV cache). After the rollout, this model
slices the pred to the LAST ``num_training_frames`` frames AND
slices the per-frame action streams in ``conditional_dict`` /
``unconditional_dict`` to the same window before passing to the
bidirectional scorer (whose seq_len is sized to
``num_training_frames``). When ``rollout_frames == num_training_frames``
(default), all of this is a no-op — every frame backprops.

CF-style first-chunk BOUNDARY mask (long-rollout mode only): the
first ``num_frame_per_block`` frames of the scoring window read
their attention context from the warmup KV cache (frames generated
under ``no_grad``); they are the "boundary" between warmup and
gradient-active regions. CF
(``Causal-Forcing/long_video/model/base.py:169-177``) zeros the
gradient on this first chunk so the student is not penalised for
the warmup→grad transition pattern. We do the same. We
deliberately SKIP CF's decode→re-encode anchoring of the boundary
frame: CF replaces the boundary latent with a freshly VAE-encoded
pixel frame; with the gradient masked off, the boundary frame's
contents cannot affect the loss, so the round-trip is wasted
compute. Mask is None when ``rollout_frames == num_training_frames``
(classic and pure-extension modes) — every scoring frame backprops.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from model.base import SelfForcingModel
from pipeline.action_forcing_training import (
    _ACTION_STREAM_KEYS,
    _slice_per_frame_streams,
)
from utils.debug_option import DEBUG

try:
    import peft  # type: ignore
    from peft import (  # type: ignore
        LoraConfig,
        set_peft_model_state_dict,
    )
    _HAS_PEFT = True
except Exception:
    peft = None  # type: ignore
    LoraConfig = None  # type: ignore
    set_peft_model_state_dict = None  # type: ignore
    _HAS_PEFT = False


def _is_main() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def _slice_baseline_scoring_window(
    cond_dict: Dict[str, torch.Tensor],
    rollout_frames: int,
    num_training_frames: int,
) -> Dict[str, torch.Tensor]:
    """Slice ``cond_dict``'s per-frame action streams to the LAST
    ``num_training_frames`` of the BASELINE rollout window.

    The "baseline rollout window" is frames ``[0, rollout_frames)`` of
    the conditioning streams — the gradient-active region. With MAE-
    extension enabled, ``cond_dict`` may carry additional frames past
    ``rollout_frames`` (sized to cover possible extensions); those
    extension frames are NEVER scored, so we must positionally slice
    inside the baseline window rather than ``[-num_training_frames:]``
    (which would grab extension frames when extensions extended the
    streams). The slice is
    ``[rollout_frames - num_training_frames : rollout_frames]``,
    which collapses correctly across all three regimes:

      - classic           (rollout_frames == num_training_frames):
            ``[0:num_training_frames]`` (no-op when len==num_training_frames)
      - long-rollout      (rollout_frames >  num_training_frames, no ext):
            ``[rollout_frames - num_training_frames : rollout_frames]``
            = LAST num_training_frames of cond_dict
      - extension mode    (rollout_frames <= num_training_frames OR
                            cond_dict longer than rollout_frames):
            ``[rollout_frames - num_training_frames : rollout_frames]``
            = first ``num_training_frames`` (when rollout_frames ==
            num_training_frames) regardless of how far the streams
            extend past ``rollout_frames``.

    Non-action keys (``prompt_embeds`` etc.) are passed through.
    """
    start = rollout_frames - num_training_frames
    if start < 0:
        raise ValueError(
            f"rollout_frames ({rollout_frames}) must be >= "
            f"num_training_frames ({num_training_frames}); the "
            f"scoring window is the LAST num_training_frames of the "
            f"baseline rollout."
        )
    return _slice_per_frame_streams(
        cond_dict, frame_start=start, frame_count=num_training_frames,
    )


class ActionForcingDMD(SelfForcingModel):
    """DMD2 trainer module for the Action-Forcing (no-staircase) Phase-1 recipe."""

    def __init__(self, args, device):
        super().__init__(args, device)

        self.num_frame_per_block = int(getattr(args, "num_frame_per_block", 3))
        if self.num_frame_per_block > 1 and hasattr(self.generator, "model"):
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = False
        self.num_training_frames = int(getattr(args, "num_training_frames", 21))
        # ``rollout_frames`` is the actual rollout length per training
        # iter. When None / unset it defaults to ``num_training_frames``
        # (= classic CF chunkwise behavior, every rolled frame
        # backprops). When set greater, the leading
        # ``rollout_frames - num_training_frames`` frames warm the KV
        # cache under ``no_grad`` and only the trailing
        # ``num_training_frames`` carry gradient. The bidirectional
        # scorer is always fed exactly ``num_training_frames`` frames
        # (the trailing slice), since its seq_len was sized to that
        # window above and feeding it longer sequences would either
        # silently truncate or break attention.
        rollout_frames_raw = getattr(args, "rollout_frames", None)
        self.rollout_frames = int(
            rollout_frames_raw if rollout_frames_raw is not None
            else self.num_training_frames
        )
        if self.rollout_frames < self.num_training_frames:
            raise ValueError(
                f"rollout_frames ({self.rollout_frames}) must be >= "
                f"num_training_frames ({self.num_training_frames}); "
                f"the gradient/scoring window is the LAST "
                f"num_training_frames frames of the rollout."
            )

        if getattr(args, "gradient_checkpointing", False):
            try:
                self.generator.enable_gradient_checkpointing()
                self.fake_score.enable_gradient_checkpointing()
            except Exception as e:
                if _is_main():
                    logging.warning("gradient_checkpointing enable failed: %s", e)

        # Resize the bidirectional scorer wrappers to the FULL training
        # window (``num_training_frames``, default 21). ``BaseModel.
        # _initialize_models`` sized them to the staircase batched
        # window (4*npb), which is too small for our symmetric scoring.
        n_score_frames = self.num_training_frames
        for scorer_name in ("real_score", "fake_score"):
            scorer = getattr(self, scorer_name)
            scorer._base_seq_len = n_score_frames * 1560
            scorer.seq_len = scorer._base_seq_len
            if getattr(self, "_action_patch_enabled", False):
                # adjust_seq_len_for_action_tokens reads from
                # _base_seq_len, so we redo it after the override.
                scorer.adjust_seq_len_for_action_tokens(
                    num_frames=n_score_frames, action_per_frame=1,
                )

        # DMD hyperparameters (mirrors CF exactly).
        self.num_train_timestep = int(getattr(args, "num_train_timestep", 1000))
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        if hasattr(args, "real_guidance_scale"):
            self.real_guidance_scale = float(args.real_guidance_scale)
            self.fake_guidance_scale = float(getattr(args, "fake_guidance_scale", 0.0))
        else:
            self.real_guidance_scale = float(getattr(args, "guidance_scale", 1.0))
            self.fake_guidance_scale = 0.0
        self.timestep_shift = float(getattr(args, "timestep_shift", 1.0))
        self.ts_schedule = bool(getattr(args, "ts_schedule", True))
        self.ts_schedule_max = bool(getattr(args, "ts_schedule_max", False))
        self.min_score_timestep = int(getattr(args, "min_score_timestep", 0))

        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None

        self.dmd_loss_weight = float(getattr(args, "dmd_loss_weight", 1.0))

        # Fake-score updates: ON by default for DMD2.
        self.fake_score_updates_enabled = bool(
            getattr(args, "fake_score_updates_enabled", True)
        )

        # Auxiliary action-supervision heads — instantiated for ODE
        # checkpoint compatibility. NOT consumed by the loss path in
        # this model (Phase-1 Action-Forcing is pure DMD).
        self._build_action_aux_heads_compat(args, device)

        # ----- Load checkpoints -----
        self._load_generator_from_ode_checkpoint(args, device)
        self._load_real_score_with_v14_lora(args, device)
        self._mirror_generator_into_fake_score()

        # Finalize freezes.
        for p in self.real_score.parameters():
            p.requires_grad_(False)
        if not self.fake_score_updates_enabled:
            for p in self.fake_score.parameters():
                p.requires_grad_(False)
            self._fake_score_trainable = False
        else:
            self._fake_score_trainable = True

        # Pipeline is set later by the trainer (after DDP wrap).
        self.inference_pipeline = None

    # ------------------------------------------------------------------
    # Auxiliary head compat shim (heads instantiated, frozen, unused)
    # ------------------------------------------------------------------
    def _build_action_aux_heads_compat(self, args, device) -> None:
        """Instantiate aux heads ONLY for ODE checkpoint compatibility.

        Phase-1 Action-Forcing's loss path does NOT use ``action_critic`` /
        ``state_probe`` aux losses (CF-parity DMD only). But the ODE
        checkpoint stores their state_dicts under ``action_critic`` /
        ``state_probe`` keys, so we instantiate matching modules
        whenever the corresponding ``*_aux_enabled`` flag is True so
        that ``_load_generator_from_ode_checkpoint`` can populate them
        without losing keys (loud non-strict load logs would fire on
        every run otherwise).

        Both heads get ``requires_grad_(False)`` and ``.eval()`` so
        they're truly inert. The trainer's optimizer never sees them.
        """
        self.action_critic = None
        self.state_probe = None

        # action_critic
        if bool(getattr(args, "action_critic_aux_enabled", False)):
            try:
                from model.action_critic import ActionCritic
                critic = ActionCritic(
                    latent_channels=16,
                    action_dim=int(getattr(args, "raw_action_dim", 2)),
                    z_out_dim=int(getattr(args, "action_critic_z_out_dim", 8)),
                    base_channels=int(getattr(args, "action_critic_base_channels", 128)),
                    num_res_blocks=int(getattr(args, "action_critic_num_blocks", 4)),
                    chunk_frames=self.num_frame_per_block,
                ).to(device=device)
                critic.requires_grad_(False)
                critic.eval()
                self.action_critic = critic
            except Exception as exc:
                if _is_main():
                    logging.warning(
                        "[ActionForcingDMD] action_critic compat instantiation failed: %s",
                        exc,
                    )
                self.action_critic = None

        # state_probe (attaches to generator wrapper)
        if (
            bool(getattr(args, "state_probe_aux_enabled", False))
            and hasattr(self.generator, "adding_state_probe_branch")
        ):
            try:
                self.generator.adding_state_probe_branch(
                    n_chunks=max(1, self.num_training_frames // self.num_frame_per_block),
                    z_out_dim=int(getattr(args, "state_head_out_dim", 8)),
                    dim=int(getattr(self.generator.model, "dim", 2048)),
                    probe_dim=int(getattr(args, "state_probe_dim", 256)),
                    num_heads=int(getattr(args, "state_probe_num_heads", 8)),
                    n_taps=int(getattr(args, "state_probe_n_taps", 6)),
                    num_frame_per_block=self.num_frame_per_block,
                )
                probe = getattr(self.generator, "_state_probe", None)
                if probe is not None:
                    probe.to(device=device)
                    probe.requires_grad_(False)
                    probe.eval()
                    self.state_probe = probe
            except Exception as exc:
                if _is_main():
                    logging.warning(
                        "[ActionForcingDMD] state_probe compat instantiation failed: %s",
                        exc,
                    )
                self.state_probe = None

    # ------------------------------------------------------------------
    # Checkpoint loading (ported from staircase model, simplified)
    # ------------------------------------------------------------------
    def _load_generator_from_ode_checkpoint(self, args, device) -> None:
        ckpt_path = getattr(args, "ode_generator_checkpoint", None)
        if not ckpt_path:
            if _is_main():
                logging.warning(
                    "[ActionForcingDMD] No ode_generator_checkpoint provided; generator "
                    "starts from raw Wan weights (not recommended).",
                )
            return
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"ode_generator_checkpoint not found: {ckpt_path}")

        if _is_main():
            logging.info("[ActionForcingDMD] Loading generator from %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "generator" not in ckpt:
            raise KeyError(
                f"ODE checkpoint missing 'generator' key: {list(ckpt.keys())}"
            )

        strict = bool(getattr(args, "strict_ode_load", False))
        missing, unexpected = self.generator.model.load_state_dict(
            ckpt["generator"], strict=False,
        )
        if _is_main():
            logging.info(
                "[ActionForcingDMD] generator load: missing=%d unexpected=%d",
                len(missing), len(unexpected),
            )
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"[ActionForcingDMD] strict_ode_load=True but generator load has "
                f"missing={len(missing)} unexpected={len(unexpected)} keys."
            )

        # Action heads: action_projection + action_token_projection live on
        # BaseModel; load if present.
        if "action_projection" in ckpt and self.action_projection is not None:
            ap_missing, ap_unexpected = self.action_projection.load_state_dict(
                ckpt["action_projection"], strict=False,
            )
            if _is_main():
                logging.info(
                    "[ActionForcingDMD] action_projection load: missing=%d unexpected=%d",
                    len(ap_missing), len(ap_unexpected),
                )
            if strict and (ap_missing or ap_unexpected):
                raise RuntimeError(
                    "[ActionForcingDMD] strict_ode_load=True but action_projection has "
                    "missing/unexpected keys."
                )

        for key, attr in (
            ("action_token_projection", "action_token_projection"),
            ("action_critic", "action_critic"),
            ("state_probe", "state_probe"),
        ):
            if key in ckpt and getattr(self, attr, None) is not None:
                try:
                    h_missing, h_unexpected = getattr(self, attr).load_state_dict(
                        ckpt[key], strict=False,
                    )
                    if _is_main():
                        logging.info(
                            "[ActionForcingDMD] %s load: missing=%d unexpected=%d",
                            key, len(h_missing), len(h_unexpected),
                        )
                except Exception as exc:
                    if _is_main():
                        logging.warning("[ActionForcingDMD] %s load failed: %s", key, exc)
                    if strict:
                        raise

    def _load_real_score_with_v14_lora(self, args, device) -> None:
        v14_ckpt_path = getattr(args, "v14_teacher_checkpoint", None)
        if not v14_ckpt_path:
            if _is_main():
                logging.warning(
                    "[ActionForcingDMD] No v14_teacher_checkpoint provided; real_score "
                    "is base Wan weights (DMD grads will degenerate).",
                )
            return
        if not os.path.exists(v14_ckpt_path):
            raise FileNotFoundError(
                f"v14_teacher_checkpoint not found: {v14_ckpt_path}"
            )
        if not _HAS_PEFT:
            raise RuntimeError(
                "peft required to load v14 LoRA into real_score but is not installed."
            )

        lora_cfg = getattr(args, "v14_lora", None) or {
            "rank": 256, "alpha": 256, "dropout": 0.0,
        }
        rank = int(
            lora_cfg.get("rank", 256) if isinstance(lora_cfg, dict)
            else getattr(lora_cfg, "rank", 256)
        )
        alpha = float(
            lora_cfg.get("alpha", rank) if isinstance(lora_cfg, dict)
            else getattr(lora_cfg, "alpha", rank)
        )
        dropout = float(
            lora_cfg.get("dropout", 0.0) if isinstance(lora_cfg, dict)
            else getattr(lora_cfg, "dropout", 0.0)
        )

        target_modules = self._collect_target_modules(self.real_score.model)
        if not target_modules:
            target_modules = ["q", "k", "v", "o"]

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
        )
        if _is_main():
            logging.info(
                "[ActionForcingDMD] Applying v14 LoRA (rank=%d alpha=%s drop=%s) to "
                "real_score over %d linear modules",
                rank, alpha, dropout, len(target_modules),
            )
        peft_model = peft.get_peft_model(self.real_score.model, lora_config)
        ckpt = torch.load(v14_ckpt_path, map_location="cpu")
        lora_sd = ckpt.get("lora")
        if lora_sd is None:
            raise KeyError(
                f"v14 checkpoint missing 'lora' key: have {list(ckpt.keys())}"
            )
        try:
            set_peft_model_state_dict(peft_model, lora_sd)
        except Exception:
            from peft import get_peft_model_state_dict
            current_sd = get_peft_model_state_dict(peft_model)
            matched = 0
            for key in current_sd:
                if key in lora_sd and current_sd[key].shape == lora_sd[key].shape:
                    current_sd[key] = lora_sd[key]
                    matched += 1
            set_peft_model_state_dict(peft_model, current_sd)
            if _is_main():
                logging.info(
                    "[ActionForcingDMD] real_score LoRA cross-load matched %d/%d",
                    matched, len(current_sd),
                )

        merged = peft_model.merge_and_unload()
        try:
            from peft.tuners.lora import LoraLayer
            for _, m in merged.named_modules():
                if isinstance(m, LoraLayer):
                    raise RuntimeError("LoRA layers remain after merge_and_unload on real_score.")
        except Exception:
            pass
        self.real_score.model = merged.to(device=device, dtype=self.dtype)

    @staticmethod
    def _collect_target_modules(model) -> list:
        target_modules = set()
        for module_name, module in model.named_modules():
            if module.__class__.__name__ in {"WanAttentionBlock", "CausalWanAttentionBlock"}:
                for full_name, submodule in module.named_modules(prefix=module_name):
                    if isinstance(submodule, nn.Linear):
                        target_modules.add(full_name)
        return sorted(target_modules)

    def _mirror_generator_into_fake_score(self) -> None:
        """DMD2 upgrade: fake_score starts as a clone of the generator's
        ODE-init state. Without this, the DMD subtraction kicks off as
        ``real - random`` and the student gets a noisy unhelpful gradient
        for the first ~hundreds of steps."""
        try:
            gen_sd = self.generator.model.state_dict()
            missing, unexpected = self.fake_score.model.load_state_dict(
                gen_sd, strict=False,
            )
            if _is_main() and DEBUG:
                logging.info(
                    "[ActionForcingDMD] fake_score mirror: missing=%d unexpected=%d",
                    len(missing), len(unexpected),
                )
        except Exception as e:
            if _is_main():
                logging.warning("[ActionForcingDMD] fake_score mirror failed: %s", e)

    # ------------------------------------------------------------------
    # Action conditioning helpers
    # ------------------------------------------------------------------
    def build_action_conditional(
        self,
        prompt_embeds: torch.Tensor,
        gt_actions: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Build (conditional_dict, unconditional_dict) for the full
        ``num_training_frames`` window.

        Args:
            prompt_embeds: ``[B, L, C_txt]``.
            gt_actions: ``[B, F, action_dim]`` per-frame ground-truth
                actions (no decay, exact GT — Phase-1 Action-Forcing contract).

        Both Stream A (AdaLN modulation) and Stream B (per-frame action
        tokens) are populated for every frame. ``unconditional`` zeroes
        prompt+actions; CF parity for the *real_score* always consumes
        it (uncond forward runs on every iter regardless of
        ``real_guidance_scale``), the *fake_score* uncond is gated on
        ``fake_guidance_scale != 0``. We use zero embeds rather than a
        Chinese negative prompt (CF's choice) — this is an intentional
        delta, see the comparison table.
        """
        if self.action_projection is None:
            raise RuntimeError(
                "action_projection missing; required for ActionForcingDMD action "
                "conditioning."
            )
        if self.action_token_projection is None:
            raise RuntimeError(
                "action_token_projection missing; required for Stream-B "
                "conditioning."
            )

        device = prompt_embeds.device
        dtype = prompt_embeds.dtype
        gt_actions = gt_actions.to(device=device, dtype=dtype)
        modulation = self.action_projection(
            gt_actions, num_frames=gt_actions.shape[1],
        )
        action_tokens = self.action_token_projection(gt_actions)

        conditional = {
            "prompt_embeds": prompt_embeds,
            "_action_modulation": modulation,
            "_action_tokens": action_tokens,
        }
        unconditional = {
            "prompt_embeds": torch.zeros_like(prompt_embeds),
            "_action_modulation": torch.zeros_like(modulation),
            "_action_tokens": torch.zeros_like(action_tokens),
        }
        return conditional, unconditional

    # ------------------------------------------------------------------
    # CF-style _run_generator. Backward-simulates the input from noise
    # and returns the student's predicted 21-frame video.
    # ------------------------------------------------------------------
    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        clean_latent=None,
        initial_latent: Optional[torch.Tensor] = None,
        enable_mae_extension: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[int], Optional[int]]:
        """Match CF's ``SelfForcingModel._run_generator`` behavior, plus
        CF-style long-rollout support and MAE-driven extension support:
          1. Sample fresh noise of shape ``[B, F, C, H, W]`` with
             F = ``rollout_frames`` (= ``num_training_frames`` by
             default; can be larger for long-rollout / "warmup the
             cache without grad" mode).
          2. Run ``inference_pipeline.inference_with_trajectory``. The
             pipeline's gradient gate
             ``start_gradient_frame_index = num_output_frames - num_max_frames``
             ensures only the LAST ``num_training_frames`` rolled
             frames carry gradient when ``rollout_frames >
             num_training_frames``. ``gt_latents=clean_latent`` enables
             the pipeline's MAE-extension loop when
             ``enable_mae_extension`` is True (caller-controlled —
             only the generator step opts in; the critic step skips
             extensions to avoid duplicating the metric collection).
          3. Slice the returned pred to its LAST ``num_training_frames``
             frames so the bidirectional scorer (whose seq_len is
             sized to ``num_training_frames``) sees only the
             gradient-active window. When ``rollout_frames ==
             num_training_frames`` this slice is a no-op.
          4. CF-style first-chunk boundary mask (long-rollout mode
             only): when ``rollout_frames > num_training_frames`` the
             leading ``num_frame_per_block`` frames of the
             ``num_training_frames`` scoring window receive context
             from the warmup KV cache (frames generated under
             ``no_grad`` to "warm" the cache). CF treats those
             leading scoring frames as a BOUNDARY: the gradient is
             masked off there so the student is not penalised for
             whatever transition pattern emerges between warmup-
             generated frames and the first grad-active block. We
             follow CF's mask logic byte-for-byte
             (``Causal-Forcing/long_video/model/base.py:169-177``)
             but DELIBERATELY SKIP CF's decode→re-encode round-trip
             (CF replaces the boundary frame with a fresh VAE-encoded
             pixel latent; we leave it as the student's raw latent
             since the gradient mask makes the boundary frame's
             content irrelevant to the loss anyway, and the round-
             trip is expensive — a full VAE decode + encode per
             training step). When ``rollout_frames ==
             num_training_frames`` (classic and pure-extension modes)
             the mask is None and every scoring frame backprops.
          5. Surface the pipeline's
             ``_last_extension_metrics`` dict (which is populated even
             when extensions are disabled, as long as ``gt_latents`` is
             provided — gives ``baseline_last_chunk_mae`` /
             ``last_chunk_mae`` always, ``mae_extension_count`` only
             on the generator step).

        Note on extensions: MAE extensions append no_grad chunks PAST
        the baseline rollout; they never enter the scoring window.
        So whether extensions fire or not is independent of the
        boundary mask — the mask is gated solely on
        ``rollout_frames > num_training_frames``.
        """
        assert getattr(self.args, "backward_simulation", True), (
            "ActionForcingDMD requires backward_simulation=True"
        )
        if self.args.i2v:
            raise RuntimeError(
                "ActionForcingDMD does not support i2v in Phase-1 Action-Forcing."
            )

        noise_shape = list(image_or_video_shape)
        # Phase-1 Action-Forcing: rollout length must equal ``self.rollout_frames``
        # (which is >= ``num_training_frames``). The trainer is
        # responsible for sizing the ``image_or_video_shape`` to the
        # baseline rollout ONLY (extensions are sampled inline by the
        # pipeline, not from this noise tensor).
        if noise_shape[1] != self.rollout_frames:
            raise RuntimeError(
                f"Phase-1 Action-Forcing expects rollout of {self.rollout_frames} "
                f"latent frames (rollout_frames knob; defaults to "
                f"num_training_frames={self.num_training_frames}); got "
                f"noise_shape[1]={noise_shape[1]}. Fix the trainer to "
                f"pin the rollout to {self.rollout_frames} frames."
            )

        if self.inference_pipeline is None:
            raise RuntimeError(
                "ActionForcingDMD.inference_pipeline is None — the trainer must set "
                "this BEFORE calling generator_loss / critic_loss. Set "
                "`model.inference_pipeline = ActionForcingTrainingPipeline(...)` in "
                "trainer._build_pipeline."
            )

        noise = torch.randn(
            noise_shape, device=self.device, dtype=self.dtype,
        )

        pred_image_or_video, denoised_timestep_from, denoised_timestep_to = (
            self.inference_pipeline.inference_with_trajectory(
                noise=noise,
                clean_image_or_video=None,
                gt_latents=clean_latent,
                enable_mae_extension=enable_mae_extension,
                **conditional_dict,
            )
        )

        # CF-style: slice to the LAST ``num_training_frames`` of the
        # rollout. The pipeline already gated gradient to this slice
        # via ``start_gradient_frame_index``, so the leading frames
        # carry no grad and would only confuse the scorer (whose
        # seq_len is sized to num_training_frames). When
        # rollout_frames == num_training_frames this is a no-op.
        if pred_image_or_video.shape[1] != self.num_training_frames:
            pred_image_or_video = pred_image_or_video[
                :, -self.num_training_frames:
            ].contiguous()

        # CF-style first-chunk BOUNDARY mask. Only built when the
        # rollout is strictly longer than the scoring window — i.e.
        # when there were warmup frames preceding the scoring window
        # in the KV cache. The first ``num_frame_per_block`` frames of
        # the scoring window are fed with that warmup context, so CF
        # treats them as a "boundary" carrying no gradient. Frames
        # beyond the first block see only context they themselves
        # generated under gradient, so they backprop normally.
        #
        # CF parity reference: Causal-Forcing/long_video/model/base.py
        # lines 169-177. We deliberately omit CF's decode→re-encode
        # of the boundary latent: with the gradient masked off the
        # boundary frame's content cannot affect the loss, so the
        # extra VAE round-trip is wasted work.
        if self.rollout_frames > self.num_training_frames:
            gradient_mask = torch.ones_like(
                pred_image_or_video, dtype=torch.bool,
            )
            block = int(self.num_frame_per_block)
            gradient_mask[:, :block] = False
        else:
            gradient_mask = None

        return (
            pred_image_or_video.to(self.dtype),
            gradient_mask,
            denoised_timestep_from,
            denoised_timestep_to,
        )

    # ------------------------------------------------------------------
    # CF-parity DMD core
    # ------------------------------------------------------------------
    def _compute_kl_grad(
        self,
        noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        normalization: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        """Eq. (7) of the DMD paper, byte-for-byte CF parity.

        Both real_score and fake_score score THE SAME ``noisy_image_or_
        video`` (derived from the student's pred via add_noise). No GT,
        no asymmetric contexts. CFG branches are gated on the relevant
        guidance scale being non-zero (saves a forward when off).
        """
        # Step 1: fake score
        _, pred_fake_image_cond = self.fake_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep,
        )
        if self.fake_guidance_scale != 0.0:
            _, pred_fake_image_uncond = self.fake_score(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=unconditional_dict,
                timestep=timestep,
            )
            pred_fake_image = pred_fake_image_cond + (
                pred_fake_image_cond - pred_fake_image_uncond
            ) * self.fake_guidance_scale
        else:
            pred_fake_image = pred_fake_image_cond

        # Step 2: real score (CF parity — ALWAYS run both cond + uncond
        # forwards, no gate on real_guidance_scale). With scale=0 the
        # math collapses to ``pred_real_image_cond`` exactly, so the
        # extra forward is "free" semantically; the cost is one
        # additional real_score forward per training iter relative to a
        # gated implementation. CF code path matches this byte-for-byte
        # at ``Causal-Forcing/model/dmd.py:98-112``.
        _, pred_real_image_cond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep,
        )
        _, pred_real_image_uncond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=unconditional_dict,
            timestep=timestep,
        )
        pred_real_image = pred_real_image_cond + (
            pred_real_image_cond - pred_real_image_uncond
        ) * self.real_guidance_scale

        # Step 3: DMD grad = (fake - real). CF normalizes by
        # |x0 - real|.mean() (eq. 8). Match exactly.
        grad = pred_fake_image - pred_real_image
        if normalization:
            p_real = estimated_clean_image_or_video - pred_real_image
            normalizer = torch.abs(p_real).mean(
                dim=[1, 2, 3, 4], keepdim=True,
            )
            grad = grad / normalizer.clamp_min(1e-6)
        grad = torch.nan_to_num(grad)

        return grad, {
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.detach(),
        }

    def _sample_dmd_timestep(
        self,
        batch_size: int,
        num_frame: int,
        denoised_timestep_from: Optional[int],
        denoised_timestep_to: Optional[int],
        device: torch.device,
    ) -> torch.Tensor:
        """Sample DMD timestep with CF's ``ts_schedule`` clamp + shift."""
        min_timestep = (
            denoised_timestep_to
            if (self.ts_schedule and denoised_timestep_to is not None)
            else self.min_score_timestep
        )
        max_timestep = (
            denoised_timestep_from
            if (self.ts_schedule_max and denoised_timestep_from is not None)
            else self.num_train_timestep
        )
        # Same-step-across-frames: one timestep per sample, broadcast.
        timestep = self._get_timestep(
            min_timestep,
            max_timestep,
            batch_size,
            num_frame,
            self.num_frame_per_block,
            uniform_timestep=True,
        )
        if self.timestep_shift > 1:
            t_norm = timestep.float() / 1000.0
            t_shifted = self.timestep_shift * t_norm / (
                1 + (self.timestep_shift - 1) * t_norm
            )
            timestep = (t_shifted * 1000.0).long()
        timestep = timestep.clamp(self.min_step, self.max_step)
        return timestep

    def compute_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: Optional[int] = 0,
        denoised_timestep_to: Optional[int] = 0,
    ) -> Tuple[torch.Tensor, dict]:
        """CF-parity DMD loss (eq. 7).

        Inputs:
          image_or_video: student's pred_x0 [B, F, C, H, W] (graph-on).
          conditional_dict / unconditional_dict: full-21-frame action +
            prompt conditioning (built by ``build_action_conditional``).
          denoised_timestep_from/to: sampled by the pipeline; clamps the
            DMD timestep distribution to the rolling-step's denoise rung.
        """
        original_latent = image_or_video
        batch_size, num_frame = image_or_video.shape[:2]

        with torch.no_grad():
            timestep = self._sample_dmd_timestep(
                batch_size, num_frame,
                denoised_timestep_from, denoised_timestep_to,
                device=image_or_video.device,
            )
            noise = torch.randn_like(image_or_video)
            noisy_latent = self.scheduler.add_noise(
                image_or_video.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1),
            ).detach().unflatten(0, (batch_size, num_frame))

            grad, dmd_log_dict = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent,
                estimated_clean_image_or_video=original_latent,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
            )

        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(
                original_latent.double()[gradient_mask],
                (original_latent.double() - grad.double()).detach()[gradient_mask],
                reduction="mean",
            )
        else:
            dmd_loss = 0.5 * F.mse_loss(
                original_latent.double(),
                (original_latent.double() - grad.double()).detach(),
                reduction="mean",
            )
        return dmd_loss, dmd_log_dict

    # ------------------------------------------------------------------
    # Public losses (CF interface)
    # ------------------------------------------------------------------
    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: Optional[torch.Tensor] = None,
        initial_latent: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        """CF-parity generator loss: roll the student, compute DMD.

        ``image_or_video_shape`` is the BASELINE ROLLOUT shape
        (``[B, rollout_frames, C, H, W]``) — extensions are sampled
        inline by the pipeline, not from this noise. ``_run_generator``
        rolls the pipeline (potentially extending past ``rollout_frames``
        via the MAE-extension loop, all under no_grad), then slices the
        pred to the LAST ``num_training_frames`` of the BASELINE.
        ``conditional_dict`` / ``unconditional_dict`` per-frame streams
        may be longer than ``rollout_frames`` (sized to cover the max
        possible extended length); we slice them positionally to the
        baseline scoring window
        ``[rollout_frames - num_training_frames : rollout_frames]``
        before passing to the bidirectional scorer.

        Generator step opts INTO the MAE extension (passes
        ``enable_mae_extension=True``) and surfaces the per-call
        metrics from ``inference_pipeline._last_extension_metrics``
        into the log dict.

        When ``return_aux=True`` we additionally return an ``aux``
        dict carrying the rolled student prediction and the rung-
        exit timesteps, so a caller (the trainer) can plumb them
        into auxiliary loss paths (action critic / teacher z-
        guidance) without re-rolling the student. ``pred_image``
        is graph-carrying — its gradients are exactly what the
        DMD loss already feeds back through; the trainer should
        ``.detach()`` before passing into the critic-update branch
        and use the live tensor only for the gen-side guidance
        loss.
        """
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to = (
            self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                clean_latent=clean_latent,
                initial_latent=initial_latent,
                enable_mae_extension=True,
            )
        )
        # Positional baseline-window slice. Works in classic,
        # long-rollout, AND extension modes (cond_dict may have more
        # than rollout_frames frames; we ignore those — only the
        # baseline carries gradient and only the baseline is scored).
        scoring_frames = pred_image.shape[1]
        rollout_frames = int(image_or_video_shape[1])
        cond_for_scoring = _slice_baseline_scoring_window(
            conditional_dict,
            rollout_frames=rollout_frames,
            num_training_frames=scoring_frames,
        )
        uncond_for_scoring = _slice_baseline_scoring_window(
            unconditional_dict,
            rollout_frames=rollout_frames,
            num_training_frames=scoring_frames,
        )

        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=cond_for_scoring,
            unconditional_dict=uncond_for_scoring,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
        )
        dmd_loss = dmd_loss * self.dmd_loss_weight

        # Surface MAE-extension metrics (may be NaN-marked if
        # gt_latents was None or shorter than the baseline rollout).
        ext_metrics = getattr(
            self.inference_pipeline, "_last_extension_metrics", None,
        ) or {}
        for k, v in ext_metrics.items():
            dmd_log_dict[k] = v
        if return_aux:
            aux: Dict[str, Any] = {
                "pred_image": pred_image,
                "gradient_mask": gradient_mask,
                "denoised_timestep_from": denoised_timestep_from,
                "denoised_timestep_to": denoised_timestep_to,
                "scoring_frames": scoring_frames,
                "rollout_frames": rollout_frames,
            }
            return dmd_loss, dmd_log_dict, aux
        return dmd_loss, dmd_log_dict

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: Optional[torch.Tensor] = None,
        initial_latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """CF-parity critic (fake_score) loss: train fake_score to denoise
        the student's generated video.

        Same long-rollout / extension handling as ``generator_loss``:
        roll the student through the full ``rollout_frames`` (no_grad
        here), slice pred and cond dict to the BASELINE last
        ``num_training_frames``, and run the fake_score denoise loss
        on that slice. The MAE extension is DISABLED on the critic
        step (``enable_mae_extension=False``) to avoid duplicating the
        metric collection the generator step already does — extensions
        are pure no_grad forwards on the same student weights, so
        running them again here would just produce the same numbers
        at extra compute cost.
        """
        with torch.no_grad():
            generated_image, _, denoised_timestep_from, denoised_timestep_to = (
                self._run_generator(
                    image_or_video_shape=image_or_video_shape,
                    conditional_dict=conditional_dict,
                    clean_latent=clean_latent,
                    initial_latent=initial_latent,
                    enable_mae_extension=False,
                )
            )

        # ``generated_image`` has been sliced to the last
        # ``num_training_frames`` by ``_run_generator``. Use its actual
        # shape for all downstream tensor manipulations — NOT the
        # input ``image_or_video_shape`` (which carries
        # ``rollout_frames`` and would mis-shape the unflatten).
        scoring_shape = list(generated_image.shape)
        rollout_frames = int(image_or_video_shape[1])
        cond_for_scoring = _slice_baseline_scoring_window(
            conditional_dict,
            rollout_frames=rollout_frames,
            num_training_frames=scoring_shape[1],
        )

        critic_timestep = self._sample_dmd_timestep(
            batch_size=scoring_shape[0],
            num_frame=scoring_shape[1],
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
            device=generated_image.device,
        )

        critic_noise = torch.randn_like(generated_image)
        noisy_generated_image = self.scheduler.add_noise(
            generated_image.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1),
        ).unflatten(0, scoring_shape[:2])

        _, pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_generated_image,
            conditional_dict=cond_for_scoring,
            timestep=critic_timestep,
        )

        if self.args.denoising_loss_type == "flow":
            from utils.wan_wrapper import WanDiffusionWrapper
            flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1),
            )
            pred_fake_noise = None
        else:
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1),
            ).unflatten(0, scoring_shape[:2])

        denoising_loss = self.denoising_loss_func(
            x=generated_image.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred,
        )

        return denoising_loss, {
            "critic_timestep": critic_timestep.detach(),
        }
