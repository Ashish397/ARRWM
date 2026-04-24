# pyright: reportGeneralTypeIssues=false
"""
DMD model for phase-1 rolling-staircase training.

Diffs vs `model/dmd2b2blam_actions.py` and `model/dmd2realmselam_actions.py`:
  - Generator is loaded from the ODE-distilled student checkpoint (v14 LoRA
    already merged into its base weights).
  - Real-score is base Wan2.1-T2V-1.3B + v14 LoRA adapter, `merge_and_unload()`
    at load time, then frozen. Uses the standard num_train_timestep=1000
    schedule (not the 4-step ODE schedule).
  - Fake-score is a cloned copy of the generator's init, frozen this remit
    (fake-score updates disabled, to be addressed next remit).
  - ALL of the following are deleted, not deferred:
      * Action CFG machinery (action_cfg_*).
      * Latent Action Model (latent_action_*, indep_lat_act_*).
      * Action-sensitivity separation on critic (fake_action_sep_*).
      * Regression branch (action_loss_weight / guidance_rgs_loss_weight).
      * Motion-weighted MSE, video-metrics, compute-video-metrics.
    Rationale: v14 action apparatus (action_projection, action_critic,
    state_probe) is already merged into the ODE student generator and
    provides strong action conditioning + supervision.

Loss interface (new):
  generator_loss_on_slots(slot_outputs, ...)
    -- Consumes a list of `RollingStepOutput` records from
       `pipeline.rolling_staircase_training.RollingStaircaseTrainingPipeline`,
       each carrying a grad-enabled x0 prediction for a single (slot, step)
       pair, alongside the slot's timestep, action conditioning, and prompt
       embeddings. Runs DMD2-grad + GAN on each slot's x0
       and returns (summed_loss, per_slot_log_dict).

  critic_loss(...)
    -- Implemented but GATED behind `fake_score_updates_enabled=False`.
       Returns a zero-tensor + empty log dict this remit.

Real-score loading note:
  Following the base DMD pattern, `real_score` is instantiated with
  `is_causal=False` by `BaseModel._initialize_models`. v14 LoRA was trained on
  a causal Wan; evaluating it without causal masking is slightly OOD but DMD
  grads remain a useful score-direction signal. This is the same compromise
  used by all existing DMD2 variants in this repo.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager

from model.base import SelfForcingModel
from utils.debug_option import DEBUG

# Lazy imports for peft + model loading (guarded so this file is importable
# even when peft or the action-forcing helpers are missing).
try:
    import peft  # type: ignore
    from peft import (  # type: ignore
        LoraConfig,
        set_peft_model_state_dict,
    )
    _HAS_PEFT = True
except Exception:  # pragma: no cover
    peft = None  # type: ignore
    LoraConfig = None  # type: ignore
    set_peft_model_state_dict = None  # type: ignore
    _HAS_PEFT = False


class DMD2B2BLAM_Staircase(SelfForcingModel):
    """DMD for phase-1 rolling-staircase training (no LAM, no CFG, no action
    separation). Consumes pre-computed slot predictions from the pipeline."""

    def __init__(self, args, device):
        super().__init__(args, device)

        # Staircase is phase-1: slot-level DMD grad on rolling student outputs.
        self.num_frame_per_block = int(getattr(args, "num_frame_per_block", 3))
        if self.num_frame_per_block > 1 and hasattr(self.generator, "model"):
            self.generator.model.num_frame_per_block = self.num_frame_per_block
        # Real-score left-context width: number of GT chunks per S_n slot.
        # Default 2 (= current behavior: S_n = [GT_{n-3}, GT_{n-2}, prev_{n-1},
        # live_n]). Bumping to 3 or 4 widens real_score's visible GT history.
        # Only affects real_score's context assembly and its wrapper seq_len;
        # fake_score keeps its 3-chunk student-rolled context.
        self.real_score_num_gt_chunks = int(
            getattr(args, "real_score_num_gt_chunks", 2)
        )
        if self.real_score_num_gt_chunks < 2:
            raise ValueError(
                f"real_score_num_gt_chunks must be >= 2; got "
                f"{self.real_score_num_gt_chunks}."
            )
        self.independent_first_frame = False
        self.min_num_training_frames = int(getattr(args, "min_num_training_frames", 21))
        self.num_training_frames = int(getattr(args, "num_training_frames", 21))
        if getattr(args, "gradient_checkpointing", False):
            try:
                self.generator.enable_gradient_checkpointing()
                self.fake_score.enable_gradient_checkpointing()
            except Exception as e:
                if (not dist.is_initialized()) or dist.get_rank() == 0:
                    logging.warning("gradient_checkpointing enable failed: %s", e)

        # DMD hyperparameters (mirrored from dmd2realmselam_actions.py).
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

        # Loss weights (ported from longlive_train_init_real.yaml).
        self.dmd_loss_weight = float(getattr(args, "dmd_loss_weight", 1.0))
        self.gan_loss_weight = float(getattr(args, "gan_loss_weight", 0.0))
        self.guidance_cls_loss_weight = float(
            getattr(args, "guidance_cls_loss_weight", self.gan_loss_weight)
        )
        self.diffusion_gan = bool(getattr(args, "diffusion_gan", False))
        self.diffusion_gan_max_timestep = int(
            getattr(args, "diffusion_gan_max_timestep", self.num_train_timestep)
        )
        self.concat_time_embeddings = bool(getattr(args, "concat_time_embeddings", False))
        self.cls_on_clean_image = bool(getattr(args, "cls_on_clean_image", True))

        # Gate: do NOT train fake-score this remit.
        self.fake_score_updates_enabled = bool(
            getattr(args, "fake_score_updates_enabled", False)
        )

        # Attach the cls branch to fake_score for the GAN head (only when GAN
        # is enabled). This creates extra parameters on the fake_score that we
        # WILL train if/when fake-score updates are re-enabled next remit.
        if (
            (self.gan_loss_weight > 0.0 or self.guidance_cls_loss_weight > 0.0)
            and hasattr(self.fake_score, "adding_cls_branch")
        ):
            self.fake_score.adding_cls_branch(
                time_embed_dim=1536 if self.concat_time_embeddings else 0,
            )

        # ----- Instantiate auxiliary action-supervision heads -----
        # The student was ODE-distilled from v14 WITH `action_critic` and
        # `state_probe` active as supervision heads. Phase-1 DMD re-uses
        # them as *frozen auxiliary* losses on top of DMD: they preserve
        # the ODE-distilled action-readability of the DiT's hidden states
        # and of pred_x0, preventing DMD from drifting the student into
        # photo-realistic-but-action-blind territory. See
        # `_critic_aux_loss_for_slot` and `_state_probe_aux_loss` below.
        self._build_action_aux_heads(args, device)

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
            # Fake-score is trainable from init; record the state so the
            # per-step `_set_fake_score_trainable(True)` call in
            # `generator_loss_on_slots` is a no-op (avoids iterating all
            # params on every rolling step).
            self._fake_score_trainable = True

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------
    def _load_generator_from_ode_checkpoint(self, args, device) -> None:
        """Load the ODE-distilled student into self.generator (and the
        auxiliary action heads on self). The checkpoint format follows
        `action-forcing/af_trainer/ode.py::ODEDistillTrainer._build_checkpoint_state`:
          - "generator": full DiT state_dict (LoRA already merged).
          - "action_projection": modulation module state_dict.
          - "action_token_projection": optional.
          - "action_critic": optional.
          - "state_probe": optional.
        """
        ckpt_path = getattr(args, "ode_generator_checkpoint", None)
        if not ckpt_path:
            if (not dist.is_initialized()) or dist.get_rank() == 0:
                logging.warning(
                    "[DMD2Staircase] No ode_generator_checkpoint provided; generator "
                    "initialized from Wan pretrained weights (not recommended)."
                )
            return
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"ode_generator_checkpoint not found: {ckpt_path}")

        if (not dist.is_initialized()) or dist.get_rank() == 0:
            logging.info("[DMD2Staircase] Loading generator from %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu")

        if "generator" not in ckpt:
            raise KeyError(
                f"ODE checkpoint is missing 'generator' key: {list(ckpt.keys())}"
            )
        missing, unexpected = self.generator.model.load_state_dict(
            ckpt["generator"], strict=False
        )
        if ((not dist.is_initialized()) or dist.get_rank() == 0) and DEBUG:
            logging.info(
                "[DMD2Staircase] generator load: missing=%d unexpected=%d",
                len(missing), len(unexpected),
            )

        # Action heads (if present and we have matching modules on self).
        if "action_projection" in ckpt and self.action_projection is not None:
            try:
                self.action_projection.load_state_dict(ckpt["action_projection"], strict=False)
            except Exception as e:
                if (not dist.is_initialized()) or dist.get_rank() == 0:
                    logging.warning("[DMD2Staircase] action_projection load failed: %s", e)

        for key, attr in (
            ("action_token_projection", "action_token_projection"),
            ("action_critic", "action_critic"),
            ("state_probe", "state_probe"),
        ):
            if key in ckpt and hasattr(self, attr) and getattr(self, attr) is not None:
                try:
                    getattr(self, attr).load_state_dict(ckpt[key], strict=False)
                except Exception as e:
                    if (not dist.is_initialized()) or dist.get_rank() == 0:
                        logging.warning("[DMD2Staircase] %s load failed: %s", key, e)

    def _load_real_score_with_v14_lora(self, args, device) -> None:
        """Wrap self.real_score.model with PEFT LoRA using the v14 config,
        load the v14 LoRA weights, then `merge_and_unload()` and freeze.
        Also load v14's action apparatus into the action_projection if we are
        sharing it with the generator. Note: since real_score and generator
        share a single action_projection via `self.action_projection`
        property on BaseModel, we avoid overwriting generator-sourced
        projection weights here."""
        v14_ckpt_path = getattr(args, "v14_teacher_checkpoint", None)
        if not v14_ckpt_path:
            if (not dist.is_initialized()) or dist.get_rank() == 0:
                logging.warning(
                    "[DMD2Staircase] No v14_teacher_checkpoint provided; real_score "
                    "left as base Wan weights (this will make DMD grads degenerate)."
                )
            return
        if not os.path.exists(v14_ckpt_path):
            raise FileNotFoundError(
                f"v14_teacher_checkpoint not found: {v14_ckpt_path}"
            )
        if not _HAS_PEFT:
            raise RuntimeError(
                "peft is required to load v14 LoRA into real_score but is not installed."
            )

        lora_cfg = getattr(args, "v14_lora", None) or {
            "rank": 256,
            "alpha": 256,
            "dropout": 0.0,
        }
        rank = int(lora_cfg.get("rank", 256) if isinstance(lora_cfg, dict) else getattr(lora_cfg, "rank", 256))
        alpha = float(lora_cfg.get("alpha", rank) if isinstance(lora_cfg, dict) else getattr(lora_cfg, "alpha", rank))
        dropout = float(lora_cfg.get("dropout", 0.0) if isinstance(lora_cfg, dict) else getattr(lora_cfg, "dropout", 0.0))

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
        # Wrap the inner DiT, load LoRA, merge and unload.
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            logging.info(
                "[DMD2Staircase] Applying v14 LoRA (rank=%d alpha=%s drop=%s) "
                "to real_score over %d linear modules", rank, alpha, dropout, len(target_modules),
            )
        peft_model = peft.get_peft_model(self.real_score.model, lora_config)
        ckpt = torch.load(v14_ckpt_path, map_location="cpu")
        lora_sd = ckpt.get("lora")
        if lora_sd is None:
            raise KeyError(
                f"v14 checkpoint missing 'lora' key: have {list(ckpt.keys())}"
            )
        # Cross-load strategy: tolerate shape/name drift.
        try:
            set_peft_model_state_dict(peft_model, lora_sd)
        except Exception:
            # Best-effort keyed overwrite.
            from peft import get_peft_model_state_dict
            current_sd = get_peft_model_state_dict(peft_model)
            matched = 0
            for key in current_sd:
                if key in lora_sd and current_sd[key].shape == lora_sd[key].shape:
                    current_sd[key] = lora_sd[key]
                    matched += 1
            set_peft_model_state_dict(peft_model, current_sd)
            if (not dist.is_initialized()) or dist.get_rank() == 0:
                logging.info("[DMD2Staircase] real_score LoRA cross-load matched %d/%d", matched, len(current_sd))

        # Merge + unload so the real_score has no PEFT wrapper at eval time.
        merged = peft_model.merge_and_unload()
        # Confirm no LoRA remnants.
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
        """Same targeting policy as trainer/causal_diffusion_teacher_train.py:
        every nn.Linear under Wan attention blocks."""
        target_modules = set()
        for module_name, module in model.named_modules():
            if module.__class__.__name__ in {"WanAttentionBlock", "CausalWanAttentionBlock"}:
                for full_name, submodule in module.named_modules(prefix=module_name):
                    if isinstance(submodule, nn.Linear):
                        target_modules.add(full_name)
        return sorted(target_modules)

    def _mirror_generator_into_fake_score(self) -> None:
        """Clone the generator's DiT weights into fake_score so the DMD2
        grad starts from (real - generator_init) rather than (real - random)."""
        try:
            gen_sd = self.generator.model.state_dict()
            missing, unexpected = self.fake_score.model.load_state_dict(gen_sd, strict=False)
            if ((not dist.is_initialized()) or dist.get_rank() == 0) and DEBUG:
                logging.info(
                    "[DMD2Staircase] fake_score mirror from generator: missing=%d unexpected=%d",
                    len(missing), len(unexpected),
                )
        except Exception as e:
            if (not dist.is_initialized()) or dist.get_rank() == 0:
                logging.warning(
                    "[DMD2Staircase] fake_score mirror from generator failed: %s", e
                )

    # ------------------------------------------------------------------
    # Score helpers
    # ------------------------------------------------------------------
    @contextmanager
    def _freeze_fake_score_params(self):
        """Temporarily freeze fake_score during the generator-GAN forward."""
        if not hasattr(self.fake_score, "parameters"):
            yield
            return
        flags = []
        for p in self.fake_score.parameters():
            flags.append(p.requires_grad)
            p.requires_grad_(False)
        try:
            yield
        finally:
            for p, rg in zip(self.fake_score.parameters(), flags):
                p.requires_grad_(rg)

    def _compute_kl_grad(
        self,
        noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        normalization: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        """DMD2 grad = (x0 - real_pred) - (x0 - fake_pred) = fake_pred - real_pred.
        real_score and fake_score run with CFG if their respective guidance
        scales are set. No caches — these are bidirectional scoring forwards.
        """
        _, pred_fake_cond = self.fake_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep,
        )
        if self.fake_guidance_scale != 0.0:
            _, pred_fake_uncond = self.fake_score(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=unconditional_dict,
                timestep=timestep,
            )
            pred_fake = pred_fake_cond + (pred_fake_cond - pred_fake_uncond) * self.fake_guidance_scale
        else:
            pred_fake = pred_fake_cond

        _, pred_real_cond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep,
        )
        _, pred_real_uncond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=unconditional_dict,
            timestep=timestep,
        )
        pred_real = pred_real_cond + (pred_real_cond - pred_real_uncond) * self.real_guidance_scale

        p_real = estimated_clean_image_or_video - pred_real
        p_fake = estimated_clean_image_or_video - pred_fake
        grad = p_real - p_fake
        if normalization:
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / normalizer.clamp_min(1e-6)
        grad = torch.nan_to_num(grad)

        return grad, {
            "dmd_grad_norm": grad.abs().mean().detach(),
            "dmd_timestep": timestep.detach().float().mean(),
        }

    def _build_classifier_inputs(
        self, latents: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_frames = latents.shape[:2]
        device = latents.device
        latents = latents.float()
        if self.diffusion_gan:
            timesteps = torch.randint(
                0,
                self.diffusion_gan_max_timestep,
                (batch_size,),
                device=device,
                dtype=torch.long,
            )
        else:
            timesteps = torch.zeros((batch_size,), device=device, dtype=torch.long)
        timestep_full = timesteps[:, None].repeat(1, num_frames)
        if self.diffusion_gan:
            noise = torch.randn_like(latents)
            noisy_latents = self.scheduler.add_noise(
                latents.flatten(0, 1),
                noise.flatten(0, 1),
                timestep_full.flatten(0, 1),
            ).unflatten(0, (batch_size, num_frames))
        else:
            noisy_latents = latents
        return noisy_latents, timestep_full

    def _classifier_logits(
        self, latents: torch.Tensor, conditional_dict: dict
    ) -> torch.Tensor:
        if not hasattr(self.fake_score, "adding_cls_branch"):
            raise RuntimeError(
                "fake_score does not support a classification branch; did you set gan_loss_weight>0 in the config?"
            )
        noisy_latents, timestep_full = self._build_classifier_inputs(latents)
        outputs = self.fake_score(
            noisy_image_or_video=noisy_latents,
            conditional_dict=conditional_dict,
            timestep=timestep_full,
            classify_mode=True,
            concat_time_embeddings=self.concat_time_embeddings,
        )
        logits = outputs[-1]
        return logits.squeeze(-1)

    # ------------------------------------------------------------------
    # DMD loss per slot
    # ------------------------------------------------------------------
    def _build_slot_conditional(
        self,
        prompt_embeds: torch.Tensor,
        action_frame: torch.Tensor,
    ) -> Tuple[dict, dict]:
        """Build (conditional_dict, unconditional_dict) for a slot. action_frame
        is [B, npb, action_dim]; we reuse the shared action_projection to
        produce [B, npb, 6, H] modulation AND the shared
        action_token_projection to produce [B, npb, H] per-frame tokens.
        Both streams must be populated — the scorers' inner DiTs are
        configured with ``action_tokens_per_frame=1`` and would fail loud
        otherwise (see base.BaseModel._initialize_models).

        Unconditional zeroes both streams (text + AdaLN + Stream-B tokens)
        to give CFG a clean reference point. Prompt zeroing is retained
        from prior behaviour — guidance remains effective via the stream
        deltas alone.
        """
        device = prompt_embeds.device
        dtype = prompt_embeds.dtype
        if self.action_projection is None:
            raise RuntimeError("action_projection missing; required for DMD action conditioning.")
        if self.action_token_projection is None:
            raise RuntimeError(
                "action_token_projection missing; required for DMD Stream-B "
                "conditioning. The scorers' bidirectional DiTs are configured "
                "with action_tokens_per_frame=1."
            )
        action_frame = action_frame.to(device=device, dtype=dtype)
        modulation = self.action_projection(action_frame, num_frames=action_frame.shape[1])
        action_tokens = self.action_token_projection(action_frame)
        conditional = {
            "prompt_embeds": prompt_embeds,
            "_action_modulation": modulation.detach(),
            "_action_tokens": action_tokens.detach(),
        }
        unconditional = {
            "prompt_embeds": torch.zeros_like(prompt_embeds),
            "_action_modulation": torch.zeros_like(modulation).detach(),
            "_action_tokens": torch.zeros_like(action_tokens).detach(),
        }
        return conditional, unconditional

    def _dmd_loss_for_slot(
        self,
        pred_x0: torch.Tensor,
        slot_timestep: int,
        prompt_embeds: torch.Tensor,
        action_frame: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """DMD2 grad + surrogate MSE for a single slot's x0 ([B, npb, C, H, W])."""
        batch_size, num_frame = pred_x0.shape[:2]
        conditional, unconditional = self._build_slot_conditional(prompt_embeds, action_frame)

        with torch.no_grad():
            # Sample an auxiliary DMD timestep (not the slot's own ladder ts):
            # we follow the self-forcing convention: pick in
            # [min_score_timestep, num_train_timestep) centered around the
            # slot's own staircase position when `ts_schedule` is on.
            min_timestep = max(self.min_score_timestep, 0)
            max_timestep = self.num_train_timestep
            if self.ts_schedule:
                min_timestep = max(min_timestep, int(slot_timestep))
            if self.ts_schedule_max:
                max_timestep = min(max_timestep, max(int(slot_timestep) + 1, min_timestep + 1))
            min_timestep = min(min_timestep, max_timestep - 1)

            timestep = torch.randint(
                min_timestep,
                max_timestep,
                (batch_size, num_frame),
                device=pred_x0.device,
                dtype=torch.long,
            )
            # Keep all frames in the slot at the same timestep (LongLive parity).
            timestep[:, 1:] = timestep[:, :1]
            if self.timestep_shift > 1:
                ts_shift = (self.timestep_shift * (timestep.float() / 1000.0)) / (
                    1.0 + (self.timestep_shift - 1.0) * (timestep.float() / 1000.0)
                )
                timestep = (ts_shift * 1000.0).clamp(self.min_step, self.max_step).long()
            else:
                timestep = timestep.clamp(self.min_step, self.max_step)

            noise = torch.randn_like(pred_x0)
            noisy_x = self.scheduler.add_noise(
                pred_x0.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1),
            ).detach().unflatten(0, (batch_size, num_frame))

            grad, dmd_log = self._compute_kl_grad(
                noisy_image_or_video=noisy_x,
                estimated_clean_image_or_video=pred_x0,
                timestep=timestep,
                conditional_dict=conditional,
                unconditional_dict=unconditional,
            )

        # Surrogate loss so `grad` flows back to pred_x0 (and through
        # the generator); grad is detached (wrapped in no_grad above).
        dmd_loss = 0.5 * F.mse_loss(
            pred_x0.double(),
            (pred_x0.double() - grad.double()).detach(),
            reduction="mean",
        ).to(dtype=pred_x0.dtype)
        return dmd_loss, dmd_log

    def _gan_loss_for_slot(
        self,
        pred_x0: torch.Tensor,
        prompt_embeds: torch.Tensor,
        action_frame: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Clean-image generator GAN loss (softplus(-logits_fake) per
        `dmd2realmselam_actions.py`). `fake_score` params are frozen during
        this forward (cls-branch weights are still trainable if fake-score
        updates are enabled, but this remit they are frozen globally)."""
        if self.gan_loss_weight <= 0.0:
            return pred_x0.new_zeros(()), {}
        with self._freeze_fake_score_params():
            cls_conditional = {
                "prompt_embeds": prompt_embeds,
            }
            logits_fake = self._classifier_logits(pred_x0, cls_conditional)
        gan_loss = F.softplus(-logits_fake).mean() * self.gan_loss_weight
        return gan_loss, {
            "gan_logits": logits_fake.detach().mean(),
            "gan_real_prob": torch.sigmoid(logits_fake.detach()).mean(),
        }

    # ------------------------------------------------------------------
    # Batched-S_n DMD loss (Option 4; all 4 slots scored in one call per
    # scorer per rolling step)
    # ------------------------------------------------------------------
    def _dmd_loss_batched_sn(
        self,
        *,
        pred_x0_all_slots: torch.Tensor,        # [B, NS*npb, C, H, W] (graph)
        kv_anchor_chunk: torch.Tensor,          # [B, npb, C, H, W] (clean/detached)
        gt_context_chunks: torch.Tensor,        # [B, (k+3)*npb, C, H, W] (clean)
        gt_context_action_frames: torch.Tensor, # [B, (k+3)*npb, action_dim] (no decay)
        per_slot_action_frames: torch.Tensor,   # [B, NS*npb, action_dim] (decayed)
        per_slot_timesteps: List[int],          # len NS
        prompt_embeds: torch.Tensor,            # [B, L, C_txt]
        num_frame_per_block: int,
        num_slots: int,
        action_decay_per_slot: Tuple[float, ...],
        fake_context_chunks: torch.Tensor,      # [B, 3*npb, C, H, W] (student-rolled)
        fake_context_actions: torch.Tensor,     # [B, 3*npb, action_dim] (commanded)
        train_fake_score: bool,
        active_slot_indices: Optional[List[int]] = None,  # subset of [0..NS-1]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """Option-4 batched DMD scoring on all NS live slots with
        ASYMMETRIC contexts for real_score and fake_score.

        For slot n in 0..NS-1 each scorer is asked to denoise a 4-chunk
        window
            S_n = [ ctx_{n,0}, ctx_{n,1}, ctx_{n,2}, live_n ]
        where the first three chunks are clean (t=0) context and the
        last chunk (``live_n``) is noisified to a sampled DMD timestep
        ``t_n`` (SHARED between real and fake scorers with the SAME
        noise sample — this is required for the DMD grad subtraction to
        be meaningful).

        Real-score context (oracle guidance):
            S_n^real = [ GT_{n-3}, GT_{n-2}, prev_{n-1}_real, live_n ]
            prev_{-1}_real = kv_anchor_chunk (= last actually-committed
                             student chunk)
            prev_{n-1}_real for n>=1 = pred_x0_all_slots[:, (n-1)*npb:
                                     n*npb].detach()

        Fake-score context (student-rolled, no GT access):
            S_0^fake = [ commit_{-3}, commit_{-2}, commit_{-1}, live_0 ]
            S_1^fake = [ commit_{-2}, commit_{-1}, live_0.detach(), live_1 ]
            S_2^fake = [ commit_{-1}, live_0.detach(), live_1.detach(), live_2 ]
            S_3^fake = [ live_0.detach(), live_1.detach(), live_2.detach(), live_3 ]
            where ``commit_{-k}`` (k=1..3) are the last 3 student-
            committed chunks (from ``fake_context_chunks``, oldest first).

        Option A — shared fake_score forward:
            If ``train_fake_score=True`` the fake_score is run WITH grad
            on its parameters. Its output is used both to
              (a) contribute the student-score term to the DMD direction
                  (detached for the generator's backward so the generator
                  does NOT backprop through fake_score), AND
              (b) supervise fake_score's own denoising objective
                  (MSE(fake_pred_target, live_n.detach())) through the
                  SAME forward — no second forward pass.
            If ``train_fake_score=False`` the fake_score runs under
            ``torch.no_grad()`` and the fake-denoise loss is not computed.

        Conditioning:
          * Both Stream A (AdaLN modulation) AND Stream B (per-frame
            action tokens) are fed to the scorers, matching the 1561-
            tokens-per-frame layout the causal weights were trained on.
            The scorers are bidirectional WanModels that have been
            patched in ``model/action_model_patch.py`` to interleave the
            Stream-B tokens per frame and RoPE-separate them identically
            to the generator's causal DiT. Both streams are ``.detach()``
            -ed before entering the scorer forward so the DMD gradient
            flows through ``target_x0_batched`` only, NOT through the
            action heads.

        Returns:
          dmd_loss_raw:      generator's DMD surrogate scalar (caller
                             multiplies by ``self.dmd_loss_weight``).
          dmd_log:           aggregated DMD statistics.
          fake_denoise_loss: fake_score's own denoising objective
                             (scalar). ``None`` if ``train_fake_score``
                             is False. Caller backwards this through
                             ``fake_score_ddp`` (separate DDP group).
        """
        NS = int(num_slots)
        npb = int(num_frame_per_block)
        real_k = int(self.real_score_num_gt_chunks)
        gt_slice_chunks = real_k + 3  # covers indices [-real_k - 1, +1]
        if pred_x0_all_slots.shape[1] != NS * npb:
            raise RuntimeError(
                f"pred_x0_all_slots has {pred_x0_all_slots.shape[1]} frames; "
                f"expected NS*npb = {NS}*{npb} = {NS*npb}."
            )
        if gt_context_chunks.shape[1] != gt_slice_chunks * npb:
            raise RuntimeError(
                f"gt_context_chunks has {gt_context_chunks.shape[1]} frames; "
                f"expected (real_k+3)*npb = {gt_slice_chunks*npb} "
                f"(real_k={real_k})."
            )
        if gt_context_action_frames.shape[1] != gt_slice_chunks * npb:
            raise RuntimeError(
                f"gt_context_action_frames has {gt_context_action_frames.shape[1]} "
                f"frames; expected (real_k+3)*npb = {gt_slice_chunks*npb}."
            )
        if per_slot_action_frames.shape[1] != NS * npb:
            raise RuntimeError(
                f"per_slot_action_frames has {per_slot_action_frames.shape[1]} "
                f"frames; expected NS*npb = {NS*npb}."
            )
        if fake_context_chunks.shape[1] != 3 * npb:
            raise RuntimeError(
                f"fake_context_chunks has {fake_context_chunks.shape[1]} "
                f"frames; expected 3*npb = {3*npb}."
            )
        if fake_context_actions.shape[1] != 3 * npb:
            raise RuntimeError(
                f"fake_context_actions has {fake_context_actions.shape[1]} "
                f"frames; expected 3*npb = {3*npb}."
            )
        if self.action_projection is None:
            raise RuntimeError(
                "action_projection missing; required for batched-S_n DMD."
            )

        B = pred_x0_all_slots.shape[0]
        device = pred_x0_all_slots.device
        dtype = pred_x0_all_slots.dtype

        # ------------------------------------------------------------------
        # Build asymmetric left-context (3*npb frames each) + actions for
        # each S_n window, for BOTH scorers. The live-n target is SHARED
        # across scorers.
        # ------------------------------------------------------------------
        real_ctx_per_sn: List[torch.Tensor] = []
        real_act_per_sn: List[torch.Tensor] = []
        fake_ctx_per_sn: List[torch.Tensor] = []
        fake_act_per_sn: List[torch.Tensor] = []
        target_x0_per_sn: List[torch.Tensor] = []     # [B, npb, ...] each (graph)
        target_act_per_sn: List[torch.Tensor] = []

        # Real GT window layout: chunk index 0 = ride position
        # (action_base - (real_k+1)*npb), chunk index 1 = -real_k, ...,
        # chunk index real_k+2 = +1. Slot n consumes chunks at indices
        # [n, n + real_k) — i.e. real_k consecutive GT chunks at positions
        # {n - real_k - 1, ..., n - 2}. Anchor (prev_{-1}) for slot 0 was
        # committed with action at position -1, which sits at GT-window
        # index (real_k) = (real_k+1) - 1.
        anchor_action_idx = real_k  # ride position -1 inside the GT slice
        for n in range(NS):
            # --- Real path ---
            gt_chunks_n = gt_context_chunks[
                :, n * npb : (n + real_k) * npb
            ]
            gt_actions_n = gt_context_action_frames[
                :, n * npb : (n + real_k) * npb
            ]
            if n == 0:
                real_prev_chunk = kv_anchor_chunk
                real_prev_action = (
                    gt_context_action_frames[
                        :, anchor_action_idx * npb : (anchor_action_idx + 1) * npb
                    ]
                    * float(action_decay_per_slot[0])
                )
            else:
                real_prev_chunk = pred_x0_all_slots[
                    :, (n - 1) * npb : n * npb
                ].detach()
                real_prev_action = per_slot_action_frames[
                    :, (n - 1) * npb : n * npb
                ]
            real_ctx_per_sn.append(
                torch.cat([gt_chunks_n, real_prev_chunk], dim=1)
            )
            real_act_per_sn.append(
                torch.cat([gt_actions_n, real_prev_action], dim=1)
            )

            # --- Fake path (student-rolled) ---
            # S_n^fake slides in live-{<n}.detach() from the left as n grows.
            # At n=0 the 3 ctx chunks are commit_{-3..-1}; at n=3 all 3 ctx
            # chunks are live_{0..2}.detach().
            # Concretely ctx pos idx 0..2 (oldest→newest) for slot n come from:
            #   - if (idx < 3 - n): commit_{idx - 3 + n}  (still reaching
            #     back into the commit history)
            #   - else:              live_{idx - (3 - n)}.detach()
            fake_ctx_slots: List[torch.Tensor] = []
            fake_act_slots: List[torch.Tensor] = []
            for idx in range(3):
                if idx < 3 - n:
                    # Commit history slot (idx - 3 + n) counted from oldest=0.
                    commit_idx = idx + n
                    # commit_idx ∈ [n, 2] when idx < 3-n. When n<=2 this is
                    # always in [0, 2] by construction.
                    commit_idx = min(commit_idx, 2)
                    fake_ctx_slots.append(
                        fake_context_chunks[
                            :, commit_idx * npb : (commit_idx + 1) * npb
                        ]
                    )
                    fake_act_slots.append(
                        fake_context_actions[
                            :, commit_idx * npb : (commit_idx + 1) * npb
                        ]
                    )
                else:
                    live_idx = idx - (3 - n)  # ∈ [0, n-1]
                    fake_ctx_slots.append(
                        pred_x0_all_slots[
                            :, live_idx * npb : (live_idx + 1) * npb
                        ].detach()
                    )
                    fake_act_slots.append(
                        per_slot_action_frames[
                            :, live_idx * npb : (live_idx + 1) * npb
                        ]
                    )
            fake_ctx_per_sn.append(torch.cat(fake_ctx_slots, dim=1))
            fake_act_per_sn.append(torch.cat(fake_act_slots, dim=1))

            # --- Shared target ---
            target_x0_per_sn.append(pred_x0_all_slots[:, n * npb : (n + 1) * npb])
            target_act_per_sn.append(per_slot_action_frames[:, n * npb : (n + 1) * npb])

        # Subset which slots are forwarded through the scorers (saves compute
        # linearly in len(active)). Context was BUILT for all slots because
        # slot n's real context depends on slot n-1's output via
        # real_prev_chunk; but we only need to SCORE (=forward the scorers
        # on) the active ones. Generator-side compute is unchanged.
        if active_slot_indices is None:
            active = list(range(NS))
        else:
            active = sorted(set(int(i) for i in active_slot_indices))
            for i in active:
                if i < 0 or i >= NS:
                    raise RuntimeError(
                        f"active_slot_indices contains {i}; must be in [0, {NS})"
                    )
            if len(active) == 0:
                raise RuntimeError("active_slot_indices must be non-empty")
        NS_active = len(active)
        # Stack along batch dim: [B*NS_active, ...]
        real_ctx_batched = torch.cat(
            [real_ctx_per_sn[i] for i in active], dim=0
        ).contiguous()
        fake_ctx_batched = torch.cat(
            [fake_ctx_per_sn[i] for i in active], dim=0
        ).contiguous()
        target_x0_batched = torch.cat(
            [target_x0_per_sn[i] for i in active], dim=0
        ).contiguous()  # graph
        real_act_left = torch.cat(
            [real_act_per_sn[i] for i in active], dim=0
        ).to(device=device, dtype=dtype).contiguous()  # [B*NS_active, 3*npb, A]
        fake_act_left = torch.cat(
            [fake_act_per_sn[i] for i in active], dim=0
        ).to(device=device, dtype=dtype).contiguous()
        target_act = torch.cat(
            [target_act_per_sn[i] for i in active], dim=0
        ).to(device=device, dtype=dtype).contiguous()  # [B*NS_active, npb, A]

        # ------------------------------------------------------------------
        # Sample DMD timesteps + noise — SHARED between real & fake.
        # ------------------------------------------------------------------
        with torch.no_grad():
            per_sn_timesteps: List[torch.Tensor] = []
            for n in active:
                slot_t = int(per_slot_timesteps[n])
                min_timestep = max(self.min_score_timestep, 0)
                max_timestep = self.num_train_timestep
                if self.ts_schedule:
                    min_timestep = max(min_timestep, slot_t)
                if self.ts_schedule_max:
                    max_timestep = min(
                        max_timestep, max(slot_t + 1, min_timestep + 1)
                    )
                min_timestep = min(min_timestep, max(max_timestep - 1, 1))
                t_n = torch.randint(
                    min_timestep,
                    max_timestep,
                    (B, npb),
                    device=device,
                    dtype=torch.long,
                )
                t_n[:, 1:] = t_n[:, :1]
                if self.timestep_shift > 1:
                    ts_shift = (
                        self.timestep_shift * (t_n.float() / 1000.0)
                    ) / (
                        1.0 + (self.timestep_shift - 1.0) * (t_n.float() / 1000.0)
                    )
                    t_n = (ts_shift * 1000.0).clamp(self.min_step, self.max_step).long()
                else:
                    t_n = t_n.clamp(self.min_step, self.max_step)
                per_sn_timesteps.append(t_n)
            target_t_batched = torch.cat(per_sn_timesteps, dim=0).contiguous()  # [B*NS, npb]

            target_x0_detached = target_x0_batched.detach()
            noise = torch.randn_like(target_x0_detached)
            noisy_target = self.scheduler.add_noise(
                target_x0_detached.flatten(0, 1),
                noise.flatten(0, 1),
                target_t_batched.flatten(0, 1),
            ).unflatten(0, target_x0_detached.shape[:2])  # [B*NS, npb, ...]

            # Real and fake scorers have DIFFERENT left-context widths:
            #   real_ctx = (real_k + 1) chunks   (real_k GT + 1 prev)
            #   fake_ctx = 3 chunks               (student-rolled)
            # so we build two separate timestep tensors. Target-noise
            # timesteps are identical (shared DMD ladder).
            real_ctx_frames = (real_k + 1) * npb
            fake_ctx_frames = 3 * npb
            real_ctx_t = torch.zeros(
                (B * NS_active, real_ctx_frames), device=device, dtype=torch.long,
            )
            fake_ctx_t = torch.zeros(
                (B * NS_active, fake_ctx_frames), device=device, dtype=torch.long,
            )
            real_timestep_batched = torch.cat(
                [real_ctx_t, target_t_batched], dim=1
            ).contiguous()
            fake_timestep_batched = torch.cat(
                [fake_ctx_t, target_t_batched], dim=1
            ).contiguous()

            # Build both scorer inputs.
            real_input = torch.cat([real_ctx_batched, noisy_target], dim=1).contiguous()
            fake_input = torch.cat([fake_ctx_batched, noisy_target], dim=1).contiguous()

        # ------------------------------------------------------------------
        # Build conditional / unconditional dicts for BOTH scorers.
        # Stream A modulation AND Stream B action tokens are computed
        # separately for each scorer because the left contexts differ
        # between real (GT-based) and fake (student-rolled). Both scorers
        # see the same two-stream conditioning the generator was trained
        # on, matching the 1561-tokens-per-frame layout the causal weights
        # were fine-tuned against.
        #
        # All four action-derived tensors are ``.detach()``-ed: the DMD
        # gradient on the generator flows through ``target_x0_batched``
        # via the MSE surrogate, NOT through the AdaLN/token streams. We
        # do NOT want gradients running back through ``action_projection``
        # or ``action_token_projection`` from the scorer forwards (that
        # is the generator pipeline's job upstream).
        # ------------------------------------------------------------------
        if self.action_token_projection is None:
            raise RuntimeError(
                "action_token_projection missing; Stream B is required for "
                "the scorers (the bidirectional DiT's action_tokens_per_"
                "frame=1 is set unconditionally by base.BaseModel)."
            )
        real_action_batched = torch.cat([real_act_left, target_act], dim=1).contiguous()
        fake_action_batched = torch.cat([fake_act_left, target_act], dim=1).contiguous()
        real_modulation = self.action_projection(
            real_action_batched, num_frames=real_action_batched.shape[1],
        ).detach()
        fake_modulation = self.action_projection(
            fake_action_batched, num_frames=fake_action_batched.shape[1],
        ).detach()
        real_action_tokens = self.action_token_projection(
            real_action_batched
        ).detach()
        fake_action_tokens = self.action_token_projection(
            fake_action_batched
        ).detach()
        prompt_embeds_batched = prompt_embeds.repeat_interleave(NS_active, dim=0)
        real_conditional = {
            "prompt_embeds": prompt_embeds_batched,
            "_action_modulation": real_modulation,
            "_action_tokens": real_action_tokens,
        }
        fake_conditional = {
            "prompt_embeds": prompt_embeds_batched,
            "_action_modulation": fake_modulation,
            "_action_tokens": fake_action_tokens,
        }
        # Unconditional dicts are scorer-specific because the action streams
        # are sized to each scorer's context width (real: real_k+2 chunks;
        # fake: 4 chunks). An "unconditional" pass must match its scorer's
        # sequence length exactly.
        real_unconditional = {
            "prompt_embeds": torch.zeros_like(prompt_embeds_batched),
            "_action_modulation": torch.zeros_like(real_modulation),
            "_action_tokens": torch.zeros_like(real_action_tokens),
        }
        fake_unconditional = {
            "prompt_embeds": torch.zeros_like(prompt_embeds_batched),
            "_action_modulation": torch.zeros_like(fake_modulation),
            "_action_tokens": torch.zeros_like(fake_action_tokens),
        }

        # ------------------------------------------------------------------
        # Real-score forward (frozen; no_grad).
        # CFG is applied as in the per-slot path (_compute_kl_grad).
        # ------------------------------------------------------------------
        with torch.no_grad():
            _, pred_real_cond = self.real_score(
                noisy_image_or_video=real_input,
                conditional_dict=real_conditional,
                timestep=real_timestep_batched,
            )
            # CFG: skip the uncond forward when guidance_scale==0 (it's a
            # wasted forward — the CFG mix degenerates to pred_real_cond).
            if self.real_guidance_scale != 0.0:
                _, pred_real_uncond = self.real_score(
                    noisy_image_or_video=real_input,
                    conditional_dict=real_unconditional,
                    timestep=real_timestep_batched,
                )
                pred_real = pred_real_cond + (pred_real_cond - pred_real_uncond) * self.real_guidance_scale
            else:
                pred_real = pred_real_cond

        # ------------------------------------------------------------------
        # Fake-score forward — Option A.
        # When `train_fake_score` is True we keep grad on fake_score's
        # parameters (input is detached so no generator grad flows in).
        # The output `pred_fake_raw` is used both for (a) DMD grad
        # (detached for generator backward) and (b) fake's own denoise
        # loss (uses pred_fake_raw's graph for fake_score backward).
        # ------------------------------------------------------------------
        if train_fake_score:
            _, pred_fake_raw = self.fake_score(
                noisy_image_or_video=fake_input,
                conditional_dict=fake_conditional,
                timestep=fake_timestep_batched,
            )
            if self.fake_guidance_scale != 0.0:
                with torch.no_grad():
                    _, pred_fake_uncond = self.fake_score(
                        noisy_image_or_video=fake_input,
                        conditional_dict=fake_unconditional,
                        timestep=fake_timestep_batched,
                    )
                pred_fake_for_grad = (
                    pred_fake_raw.detach()
                    + (pred_fake_raw.detach() - pred_fake_uncond) * self.fake_guidance_scale
                )
            else:
                pred_fake_for_grad = pred_fake_raw.detach()
        else:
            with torch.no_grad():
                _, pred_fake_cond = self.fake_score(
                    noisy_image_or_video=fake_input,
                    conditional_dict=fake_conditional,
                    timestep=fake_timestep_batched,
                )
                if self.fake_guidance_scale != 0.0:
                    _, pred_fake_uncond = self.fake_score(
                        noisy_image_or_video=fake_input,
                        conditional_dict=fake_unconditional,
                        timestep=fake_timestep_batched,
                    )
                    pred_fake_for_grad = (
                        pred_fake_cond
                        + (pred_fake_cond - pred_fake_uncond) * self.fake_guidance_scale
                    )
                else:
                    pred_fake_for_grad = pred_fake_cond
            pred_fake_raw = None  # not needed for generator-only path

        # ------------------------------------------------------------------
        # DMD direction on the TARGET slice only, with asymmetric
        # "estimated_clean" reference:
        #   real side: est = real_input (GT ctx + live_n's clean x0)
        #   fake side: est = fake_input (student ctx + live_n's clean x0)
        # Both share the same target frames; the ctx frames differ but
        # drop out of the final target-slice slice [:, 3*npb:].
        # We match the per-slot `_compute_kl_grad` formula but restrict
        # to the live_n target chunk:
        #   p_real = target_x0 - pred_real[target]
        #   p_fake = target_x0 - pred_fake[target]
        #   grad   = p_real - p_fake = pred_fake[target] - pred_real[target]
        # with per-sample normalization by |p_real|.mean over the target.
        # ------------------------------------------------------------------
        with torch.no_grad():
            # Target chunk occupies the last `npb` frames in each scorer's
            # window, but the window sizes differ (real: real_k+2 chunks;
            # fake: 4 chunks), so the slice offsets are scorer-specific.
            real_target_slice = slice((real_k + 1) * npb, (real_k + 2) * npb)
            fake_target_slice = slice(3 * npb, 4 * npb)
            pred_real_target = pred_real[:, real_target_slice, ...]
            pred_fake_target = pred_fake_for_grad[:, fake_target_slice, ...]
            target_x0_detached_batched = target_x0_batched.detach()
            p_real_t = target_x0_detached_batched - pred_real_target
            p_fake_t = target_x0_detached_batched - pred_fake_target
            grad_target = p_real_t - p_fake_t
            normalizer = torch.abs(p_real_t).mean(
                dim=[1, 2, 3, 4], keepdim=True,
            )
            grad_target = grad_target / normalizer.clamp_min(1e-6)
            grad_target = torch.nan_to_num(grad_target)

        # Generator's DMD surrogate — backprop flows through target_x0_batched
        # (which carries the generator's autograd graph via pred_x0_all_slots).
        dmd_loss = 0.5 * F.mse_loss(
            target_x0_batched.double(),
            (target_x0_batched.double() - grad_target.double()).detach(),
            reduction="mean",
        ).to(dtype=dtype)

        dmd_log: Dict[str, torch.Tensor] = {
            "dmd_grad_norm": grad_target.abs().mean().detach(),
            "dmd_timestep": target_t_batched.detach().float().mean(),
            "dmd_sn_slots": torch.tensor(float(NS_active)),
        }

        # ------------------------------------------------------------------
        # Fake-score denoising loss — only if training fake_score.
        # Option A: reuse the same forward; target = live_n.detach().
        # ------------------------------------------------------------------
        fake_denoise_loss: Optional[torch.Tensor] = None
        if train_fake_score and pred_fake_raw is not None:
            fake_pred_target = pred_fake_raw[:, fake_target_slice, ...]
            fake_denoise_loss = F.mse_loss(
                fake_pred_target.float(),
                target_x0_batched.detach().float(),
                reduction="mean",
            ).to(dtype=dtype)
            dmd_log["fake_denoise_loss"] = fake_denoise_loss.detach()

        return dmd_loss, dmd_log, fake_denoise_loss

    # ------------------------------------------------------------------
    # Public loss APIs
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Auxiliary action-supervision heads (action_critic + state_probe)
    # ------------------------------------------------------------------
    def _build_action_aux_heads(self, args, device) -> None:
        """Instantiate the v14-style action_critic + state_probe heads on self.

        The ODE distillation checkpoint saves state_dicts for both. We
        build matching modules here (pre-checkpoint-load) so
        `_load_generator_from_ode_checkpoint` can then populate them via
        the generic `(key, attr)` loop.

        Both heads are **frozen** for Phase-1 DMD: they contribute
        gradient signal THROUGH themselves into the DiT (action_critic
        via the pred_x0 path; state_probe via the tap-activation path),
        but their own weights do not update. This pins the supervision
        signal to the well-calibrated v14/ODE solution so the DMD loss
        has something stable to regress the DiT's hidden features and
        pred_x0 against.

        Disabling via `action_critic_aux_enabled=False` or
        `state_probe_aux_enabled=False` leaves the attribute as None
        and the corresponding loss term is skipped (ckpt entries for
        the disabled head are ignored by the non-strict loader).

        Automatic off-switch: if ``action_teacher_mode="off"`` the
        motion-pipeline teacher-z targets are unavailable. The critic
        and probe would silently fall back to regressing against the
        *commanded* action, which is meaningless without the action
        teacher (the commanded-action path exists only as a fallback
        for slots that still have a ground-truth action — it does not
        provide a standalone supervision signal). So we FORCE both
        aux heads off when the teacher is disabled, regardless of
        their individual enable flags.
        """
        # ``action_teacher_mode`` (three-option enum, the single source
        # of truth for whether the motion-pipeline action teacher is
        # built and which slots it supervises):
        #   "off"   → teacher disabled; aux heads force-disabled.
        #   "slot0" → teacher enabled; aux supervises slot 0 ONLY
        #             (overrides ``aux_loss_slot_policy`` in the DMD
        #             loss so this mode is a hard guarantee at the
        #             loss-construction layer).
        #   "all"   → teacher enabled; aux supervises whichever slots
        #             ``aux_loss_slot_policy`` selects (slot0 /
        #             match_dmd / random).
        # A bool value is tolerated (YAML 1.1 coerces `off`→False and
        # `on`→True on dotlist overrides); see utils/action_teacher.py.
        from utils.action_teacher import resolve_action_teacher_mode
        teacher_mode = resolve_action_teacher_mode(
            getattr(args, "action_teacher_mode", None),
            source_label="action_teacher_mode",
        )
        self.action_teacher_mode = teacher_mode
        teacher_enabled = teacher_mode != "off"
        # Log the auto-override so it is visible in training logs the
        # first time a user flips teacher off without also disabling
        # the aux heads.
        requested_critic = bool(
            getattr(args, "action_critic_aux_enabled", True)
        )
        requested_probe = bool(
            getattr(args, "state_probe_aux_enabled", True)
        )
        if not teacher_enabled and (requested_critic or requested_probe):
            import logging as _lg
            _lg.warning(
                "action_teacher_mode=off -> forcing action_critic_aux "
                "and state_probe_aux OFF (requested critic=%s, probe=%s). "
                "The commanded-action fallback is not a valid standalone "
                "supervision signal; set action_teacher_mode=slot0 or "
                "all if you want these losses.",
                requested_critic, requested_probe,
            )
        # --- action_critic ---
        self.action_critic_aux_enabled = requested_critic and teacher_enabled
        self.action_critic_loss_weight = float(
            getattr(args, "action_critic_loss_weight", 0.25)
        )
        self.action_critic_dims: List[int] = list(
            getattr(args, "action_critic_dims", [2, 7])
        )
        self.action_critic: Optional[nn.Module] = None
        if self.action_critic_aux_enabled and self.action_critic_loss_weight > 0.0:
            try:
                from model.action_critic import ActionCritic
            except Exception as exc:
                raise RuntimeError(
                    "action_critic_aux_enabled=True but model.action_critic "
                    f"failed to import: {exc!r}"
                )
            critic = ActionCritic(
                latent_channels=16,
                action_dim=len(self.action_critic_dims),
                z_out_dim=int(getattr(args, "action_critic_z_out_dim", 8)),
                base_channels=int(getattr(args, "action_critic_base_channels", 128)),
                num_res_blocks=int(getattr(args, "action_critic_num_blocks", 4)),
                chunk_frames=self.num_frame_per_block,
            ).to(device=device)
            critic.requires_grad_(False)
            critic.eval()
            self.action_critic = critic

        # --- state_probe (attaches on generator wrapper; reads taps collected
        #     inside the DiT forward via `_state_probe_tap_set`) ---
        self.state_probe_aux_enabled = requested_probe and teacher_enabled
        self.state_probe_loss_weight = float(
            getattr(args, "state_probe_loss_weight", 0.1)
        )
        self.state_head_out_dim = int(getattr(args, "state_head_out_dim", 8))
        self.state_probe: Optional[nn.Module] = None
        if self.state_probe_aux_enabled and self.state_probe_loss_weight > 0.0:
            if not hasattr(self.generator, "adding_state_probe_branch"):
                raise RuntimeError(
                    "state_probe_aux_enabled=True but generator wrapper has no "
                    "`adding_state_probe_branch` method — check "
                    "utils/wan_wrapper.py version."
                )
            # The 4-slot live window = n_chunks=4 chunks of npb frames each.
            # StateProbeModule.query_init is shape [1, probe_dim] (shared across
            # chunks), so n_chunks does not affect any learnable parameter
            # shape — the ODE checkpoint's probe weights load cleanly here.
            self.generator.adding_state_probe_branch(
                n_chunks=4,
                z_out_dim=self.state_head_out_dim,
                dim=int(getattr(self.generator.model, "dim", 2048)),
                probe_dim=int(getattr(args, "state_probe_dim", 256)),
                num_heads=int(getattr(args, "state_probe_num_heads", 8)),
                n_taps=int(getattr(args, "state_probe_n_taps", 6)),
                num_frame_per_block=self.num_frame_per_block,
            )
            probe = getattr(self.generator, "_state_probe", None)
            if probe is None:
                raise RuntimeError(
                    "adding_state_probe_branch did not set `_state_probe` on "
                    "the generator wrapper."
                )
            # Move the probe to the target device. The wrapper itself is
            # not ``.to(device)``'d by the trainer (only ``generator.model``
            # is), so without this the probe's LayerNorm / Linear weights
            # stay on CPU and the first forward crashes with "weight is on
            # cpu, different from other tensors on cuda:N". Match the
            # device only — dtype stays fp32 like ``action_critic``; the
            # DiT's bf16 tap activations are upcast at the probe's layer
            # boundary which is fine and matches how v14's teacher
            # trainer (``causal_diffusion_teacher_train.py``) does it via
            # ``wrapper.to(self.device)`` right after the branch add.
            probe.to(device=device)
            probe.requires_grad_(False)
            probe.eval()
            # Mirror as `self.state_probe` so `_load_generator_from_ode_checkpoint`'s
            # generic loader populates it.
            self.state_probe = probe

    def _critic_aux_loss_for_slot(
        self,
        pred_x0: torch.Tensor,         # [B, npb, C, H, W]
        slot_timestep: int,
        action_frame: torch.Tensor,    # [B, npb, action_dim]
        teacher_z_target: Optional[torch.Tensor] = None,  # [B, z_out] (per-slot)
    ) -> torch.Tensor:
        """Frozen-critic guidance on pred_x0 (gradient through pred_x0).

        Matches the teacher/ODE recipe: the critic takes the student's
        pred_x0 and predicts the 8-D ss_vae ``z``.

        Target selection:
          - If ``teacher_z_target`` is provided (shape ``[B, z_out]`` —
            one motion-pipeline ``z`` vector per slot), regress the
            critic's FULL ``z_out``-D output against it. This is the
            "teacher-z" supervision used during v14 teacher training and
            ODE distillation; it is the preferred target when a
            CoTracker + ss_vae action teacher is wired into the trainer.
          - Otherwise, fall back to the "commanded-action" branch: slice
            the critic's output to ``action_critic_dims`` (typically
            ``[2, 7]``) and regress against the commanded action.
            Cheaper (no CoTracker calls) but supervises only the action
            dims, not the full ``z`` readout.

        Critic params are frozen (set in ``_build_action_aux_heads`` and
        re-asserted here via a ``requires_grad_`` save/restore dance in
        case something upstream flipped them).
        """
        if self.action_critic is None or self.action_critic_loss_weight <= 0.0:
            return pred_x0.new_zeros(())
        critic = self.action_critic
        B = pred_x0.shape[0]
        chunk_t = torch.full(
            (B, 1), int(slot_timestep),
            device=pred_x0.device, dtype=torch.long,
        )
        # [B, 1, action_dim] via mean-pool across the slot's npb frames.
        chunk_actions = action_frame.float().mean(dim=1, keepdim=True).to(
            dtype=pred_x0.dtype
        )
        saved = [p.requires_grad for p in critic.parameters()]
        for p in critic.parameters():
            p.requires_grad_(False)
        try:
            gen_pred_z = critic(pred_x0, chunk_t, chunk_actions)   # [B, 1, z_out]
        finally:
            for p, r in zip(critic.parameters(), saved):
                p.requires_grad_(r)
        if teacher_z_target is not None:
            z_out = gen_pred_z.shape[-1]
            tgt_full = teacher_z_target.to(
                device=gen_pred_z.device, dtype=gen_pred_z.dtype,
            )
            if tgt_full.dim() == 2:                     # [B, z_out]
                tgt_full = tgt_full.unsqueeze(1)        # [B, 1, z_out]
            if tgt_full.shape[-1] < z_out:
                raise RuntimeError(
                    f"teacher_z_target has {tgt_full.shape[-1]} channels; "
                    f"critic expects {z_out}. Align action_critic_z_out_dim "
                    f"with the teacher's output width (ss_vae produces 8)."
                )
            tgt = tgt_full[..., :z_out]
            return (
                F.mse_loss(gen_pred_z.float(), tgt.float())
                * self.action_critic_loss_weight
            )
        # Fallback: commanded-action supervision on sliced dims.
        z_sel = gen_pred_z[:, :, self.action_critic_dims]           # [B, 1, A]
        tgt = chunk_actions[:, :, : len(self.action_critic_dims)]
        return F.mse_loss(z_sel.float(), tgt.float()) * self.action_critic_loss_weight

    def _state_probe_aux_loss(
        self,
        state_preds: torch.Tensor,        # [B, num_slots, z_out_dim]
        per_slot_actions: torch.Tensor,   # [B, num_slots, action_dim]
        grad_slot_indices: List[int],
        teacher_z_target: Optional[torch.Tensor] = None,  # [B, num_slots, z_out_dim]
    ) -> torch.Tensor:
        """Supervise the state probe against either motion-teacher ``z`` or
        commanded actions.

        The probe itself is frozen; gradients flow through it back into
        the DiT's hidden features via the tapped activations collected
        during the primary cached forward. This pins the student's
        hidden-state action-readability without letting the probe drift.

        Target selection:
          - If ``teacher_z_target`` is provided ([B, num_slots, z_out]
            via the CoTracker + ss_vae action teacher), regress the probe's
            FULL ``z_out``-D readout against it. This matches the
            supervision the probe received during v14 teacher training
            and ODE distillation.
          - Otherwise, fall back to the "commanded-z" branch from
            ``ODERegression._state_probe_loss``: MSE between probe's
            action-dim readout and the commanded z for each grad slot.
            Supervises only the action dims (down-weighted other dims).
        """
        if state_preds is None or per_slot_actions is None:
            return state_preds.new_zeros(()) if state_preds is not None else \
                torch.zeros((), device=self.device, dtype=self.dtype)
        if self.state_probe is None or self.state_probe_loss_weight <= 0.0:
            return state_preds.new_zeros(())
        # Only supervise grad slots. Non-grad slots still contributed
        # activations (shared graph) but adding loss on their state_preds
        # would pull gradients through DiT regions the user's slot-sampling
        # design wants left alone.
        if not grad_slot_indices:
            return state_preds.new_zeros(())
        idx = torch.tensor(
            sorted(set(int(i) for i in grad_slot_indices)),
            device=state_preds.device, dtype=torch.long,
        )
        state_sel = state_preds.index_select(dim=1, index=idx).float()
        act_sel = per_slot_actions.index_select(dim=1, index=idx).float()
        D = state_sel.shape[-1]
        if teacher_z_target is not None:
            tgt_full = teacher_z_target.to(
                device=state_sel.device, dtype=state_sel.dtype,
            )
            if tgt_full.shape[1] != per_slot_actions.shape[1]:
                raise RuntimeError(
                    f"teacher_z_target has {tgt_full.shape[1]} slots; "
                    f"expected {per_slot_actions.shape[1]} (matches "
                    f"per_slot_actions)."
                )
            if tgt_full.shape[-1] < D:
                raise RuntimeError(
                    f"teacher_z_target has {tgt_full.shape[-1]} channels; "
                    f"probe expects {D}. Align state_head_out_dim with "
                    f"the teacher's output width (ss_vae produces 8)."
                )
            tgt_sel = tgt_full.index_select(dim=1, index=idx)[..., :D]
            loss = F.mse_loss(state_sel, tgt_sel)
            return loss * self.state_probe_loss_weight
        # Fallback: commanded-action target on action dims only.
        A = len(self.action_critic_dims)
        if D <= A:
            target = act_sel[:, :, :D]
            loss = F.mse_loss(state_sel, target)
        else:
            # 8-D probe; inject commanded action at action_critic_dims and
            # regress the rest toward zero with a down-weighted MSE (teacher
            # parity: state_head_action_dim_weight default 3x on action dims).
            target = torch.zeros_like(state_sel)
            for i, d in enumerate(self.action_critic_dims):
                if i < act_sel.shape[-1]:
                    target[:, :, d] = act_sel[:, :, i]
            w = torch.ones(D, device=state_sel.device, dtype=state_sel.dtype)
            w_dim = float(getattr(self.args, "state_head_action_dim_weight", 3.0))
            for d in self.action_critic_dims:
                w[d] = w_dim
            loss = (w * (state_sel - target) ** 2).mean()
        return loss * self.state_probe_loss_weight

    def generator_loss_on_slots(
        self,
        slot_outputs: List,
        *,
        aux_loss_weight: float = 1.0,
        teacher_z_per_slot: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """Compute generator + fake-score losses for one rolling step.

        When the pipeline attached the full batched-S_n context to the
        first record, this returns:
          - ``total_loss`` (generator): DMD + aux losses backward this
            through ``generator_ddp``.
          - ``log_dict``.
          - ``fake_denoise_loss``: fake_score's own objective (from the
            SHARED fake forward — Option A). None if fake_score updates
            are disabled. Backward this through ``fake_score_ddp`` with
            its own optimizer.

        Args:
          slot_outputs:          list of RollingStepOutput (see pipeline).
          aux_loss_weight:       scalar multiplier for slot_idx != 0 records.
                                 Default 1.0 (equal weight per plan).
          teacher_z_per_slot:    optional [B, NS, z_out] tensor from the
                                 CoTracker + ss_vae action teacher run on
                                 the full live window's pred_x0. When
                                 provided, both state_probe and
                                 action_critic aux losses are supervised
                                 against the corresponding per-slot z
                                 vector; otherwise they fall back to the
                                 commanded-action branch. Trainer owns
                                 the teacher forward (frozen, no_grad).
        """
        if len(slot_outputs) == 0:
            zero = torch.zeros((), device=self.device, dtype=self.dtype)
            return zero, {}, None

        # Fake_score trainability is governed by the config flag
        # `fake_score_updates_enabled`. When False, its params have been
        # marked requires_grad=False in __init__ and stay that way.
        # When True, they are trainable and this method will run a
        # SHARED fake forward (Option A) that feeds both the DMD direction
        # (detached for the generator's backward) and the fake_score's
        # own denoise loss (through its own autograd graph).
        train_fake_here = bool(self.fake_score_updates_enabled)
        self._set_fake_score_trainable(train_fake_here)

        total_loss: Optional[torch.Tensor] = None
        log_dict: Dict[str, torch.Tensor] = {}
        per_slot_losses: Dict[int, List[torch.Tensor]] = {}

        # Track grad-slot indices for the state_probe aux loss (supervise
        # only the slots the user's design opted into gradient flow on).
        grad_slot_indices: List[int] = []
        # Pre-scan to know the grad-slot set before the per-slot loop (we
        # need it to pick which slots the auxiliary heads — action_critic +
        # state_probe + motion-teacher supervision — will supervise).
        prescanned_grad_slots: List[int] = [
            int(out.slot_idx) for out in slot_outputs if out.pred_x0.requires_grad
        ]
        # Default: DMD active slots unknown until the batched branch runs.
        dmd_active_slots_for_aux: Optional[List[int]] = None

        # ------------------------------------------------------------------
        # Step-level DMD: Option 4 batched-S_n when the pipeline attached
        # the full S_n context to the first record; otherwise legacy
        # per-slot DMD for backward compatibility.
        # ------------------------------------------------------------------
        first = slot_outputs[0]
        use_batched_dmd = (
            first.pred_x0_all_slots is not None
            and first.kv_anchor_chunk is not None
            and first.gt_context_chunks is not None
            and first.gt_context_action_frames is not None
            and first.per_slot_action_frames is not None
            and first.per_slot_timesteps_list is not None
            and first.fake_context_chunks is not None
            and first.fake_context_actions is not None
        )
        # Debug/benchmark knob: bypass all scorer forwards (real + fake +
        # DMD math) so the remaining iter time is attributable to the
        # generator's rolling forward/backward only. DO NOT use in
        # production — training signal becomes aux-losses-only.
        if bool(getattr(self.args, "skip_dmd_for_timing", False)):
            use_batched_dmd = False
        fake_denoise_loss: Optional[torch.Tensor] = None
        if use_batched_dmd:
            pipe = getattr(self, "_staircase_pipeline", None)
            if pipe is None:
                raise RuntimeError(
                    "Batched-S_n DMD requires `self._staircase_pipeline` to "
                    "be set by the trainer so action_decay_per_slot is "
                    "accessible at loss-compute time."
                )
            # --- Optional slot subsetting for the DMD scorer batch. ---
            # Controlled by `dmd_active_slot_policy` in args:
            #   "all"         → score all NS slots (default, current behavior)
            #   "slot0"       → score only slot 0
            #   "random"      → slot 0 is ALWAYS included (ladder anchor) +
            #                   `dmd_num_active_slots - 1` further slots drawn
            #                   uniformly from {1..NS-1}. With
            #                   dmd_num_active_slots=1 this degenerates to
            #                   [0] (use "random_any" if you want a
            #                   truly-uniform single-slot pick).
            #   "random_any"  → uniformly pick `dmd_num_active_slots` slots
            #                   each step from {0..NS-1} with NO slot-0
            #                   anchoring. Use this for true random-slot
            #                   ablations where you want slot 0 to be picked
            #                   only 1/NS of the time.
            # Generator compute is UNCHANGED by this knob; we only drop scorer work.
            # Both "random" and "random_any" are DDP-synced (rank 0 rolls the
            # pick, broadcasts to all ranks) so the grad-slot set is identical
            # across ranks — required by the grad-uniformity check + DDP
            # all-reduce.
            NS_pipe = int(getattr(pipe, "num_live_slots", getattr(pipe, "NUM_SLOTS", 4)))
            policy = str(getattr(self.args, "dmd_active_slot_policy", "all")).lower()
            num_active = int(getattr(self.args, "dmd_num_active_slots", NS_pipe))
            num_active = max(1, min(num_active, NS_pipe))

            # Device for the broadcast tensor — use the grad-enabled
            # pred_x0 that already lives on the correct CUDA device
            # (``first`` is the first RollingStepOutput in this step).
            _sync_device = first.pred_x0.device

            def _ddp_sync_slot_pick(picks: List[int]) -> List[int]:
                """Broadcast rank-0's pick to all ranks. No-op on single rank."""
                if not dist.is_initialized():
                    return picks
                # Pad to NS_pipe so the tensor shape is rank-invariant.
                pad = [-1] * (NS_pipe - len(picks))
                buf = torch.tensor(
                    list(picks) + pad, dtype=torch.long, device=_sync_device
                )
                dist.broadcast(buf, src=0)
                synced = [int(x) for x in buf.tolist() if int(x) >= 0]
                return synced

            if policy == "all":
                active_slots: Optional[List[int]] = None
            elif policy == "slot0":
                active_slots = [0]
            elif policy == "random":
                import random as _random
                if num_active >= NS_pipe:
                    active_slots = None
                else:
                    if (not dist.is_initialized()) or dist.get_rank() == 0:
                        rest = _random.sample(
                            list(range(1, NS_pipe)),
                            k=max(0, num_active - 1),
                        )
                        picks = [0] + rest
                    else:
                        picks = []
                    active_slots = _ddp_sync_slot_pick(picks)
            elif policy == "random_any":
                import random as _random
                if num_active >= NS_pipe:
                    active_slots = None
                else:
                    if (not dist.is_initialized()) or dist.get_rank() == 0:
                        picks = sorted(_random.sample(
                            list(range(NS_pipe)), k=num_active
                        ))
                    else:
                        picks = []
                    active_slots = _ddp_sync_slot_pick(picks)
            else:
                raise RuntimeError(
                    f"Unknown dmd_active_slot_policy: {policy!r}. "
                    "Expected one of: all, slot0, random, random_any."
                )
            dmd_loss_raw, dmd_log, fake_denoise_loss = self._dmd_loss_batched_sn(
                pred_x0_all_slots=first.pred_x0_all_slots,
                kv_anchor_chunk=first.kv_anchor_chunk,
                gt_context_chunks=first.gt_context_chunks,
                gt_context_action_frames=first.gt_context_action_frames,
                per_slot_action_frames=first.per_slot_action_frames,
                per_slot_timesteps=list(first.per_slot_timesteps_list),
                prompt_embeds=first.prompt_embeds,
                num_frame_per_block=int(self.num_frame_per_block),
                num_slots=NS_pipe,
                action_decay_per_slot=tuple(pipe.action_decay_per_slot),
                fake_context_chunks=first.fake_context_chunks,
                fake_context_actions=first.fake_context_actions,
                train_fake_score=train_fake_here,
                active_slot_indices=active_slots,
            )
            dmd_step_loss = dmd_loss_raw * self.dmd_loss_weight
            total_loss = dmd_step_loss
            for k, v in dmd_log.items():
                log_dict[f"batched_dmd/{k}"] = v
            log_dict["batched_dmd/loss"] = dmd_step_loss.detach()
            # Stash the DMD active_slots so the aux-loss policy can optionally
            # follow them (see aux_loss_slot_policy below). None means "all NS".
            dmd_active_slots_for_aux = active_slots

        # -------------------------------------------------------------------
        # Auxiliary-head slot policy.
        # -------------------------------------------------------------------
        # Which slots get action_critic + state_probe + motion-teacher
        # supervision. Historically restricted to slot 0 because slots 1..NS-1
        # receive decayed commanded actions (e.g. 0.75/0.5/0.25 x a_t in 4x1,
        # or 0.5 x a_t in 2x2) and regressing the critic against those would
        # bias the student toward muted motion. The teacher-z branch of the
        # critic does NOT suffer that bias (teacher z is derived from pixels),
        # so when `aux_loss_slot_policy="match_dmd"` is set and the teacher
        # is enabled, aux losses follow the DMD scorer's active slot(s) —
        # giving extra supervision on whichever slot the random chooser
        # (dmd_active_slot_policy="random"/"slot0") picked.
        #
        # Policies:
        #   "slot0"     → always [0]. Default. Matches historical behavior.
        #   "match_dmd" → follow DMD's active_slots (from dmd_active_slot_policy).
        #                 If DMD scores all NS slots (default "all"), aux
        #                 runs on all grad slots too.
        #   "random"    → pick ONE random slot from grad_slot_indices each
        #                 rolling step (DDP-synced). Independent of DMD.
        # The chosen slots are intersected with grad_slot_indices to
        # guarantee every aux loss has a live autograd path.
        # Hard override: when the teacher is restricted to slot 0
        # (``action_teacher_mode=slot0``), aux supervision is pinned to
        # slot 0 regardless of ``aux_loss_slot_policy``. This keeps
        # teacher compute scope and aux consumption scope coherent and
        # prevents accidentally regressing the critic/probe against
        # commanded-action fallbacks on slots the teacher hasn't
        # authorized.
        teacher_mode = getattr(self, "action_teacher_mode", "off")
        aux_policy = str(getattr(self.args, "aux_loss_slot_policy", "slot0")).lower()
        if teacher_mode == "slot0":
            aux_candidate_slots: List[int] = [0]
        elif aux_policy == "slot0":
            aux_candidate_slots = [0]
        elif aux_policy == "match_dmd":
            if dmd_active_slots_for_aux is None:
                # DMD runs on all NS slots (policy="all" or legacy path).
                aux_candidate_slots = list(prescanned_grad_slots)
            else:
                aux_candidate_slots = list(dmd_active_slots_for_aux)
        elif aux_policy == "random":
            if not prescanned_grad_slots:
                aux_candidate_slots = []
            else:
                # Rank-synced uniform pick from grad_slot_indices.
                if (not dist.is_initialized()) or dist.get_rank() == 0:
                    pick_idx = int(torch.randint(
                        0, len(prescanned_grad_slots), size=(1,),
                    ).item())
                else:
                    pick_idx = 0
                if dist.is_initialized():
                    pick_t = torch.tensor(
                        [pick_idx], dtype=torch.long,
                        device=first.pred_x0.device,
                    )
                    dist.broadcast(pick_t, src=0)
                    pick_idx = int(pick_t.item())
                aux_candidate_slots = [prescanned_grad_slots[pick_idx]]
        else:
            raise RuntimeError(
                f"Unknown aux_loss_slot_policy: {aux_policy!r}. "
                "Expected one of: slot0, match_dmd, random."
            )
        # Final aux slot set: candidate ∩ grad-enabled (defensive; filters
        # away anything we don't have an autograd path on).
        aux_active_slots: List[int] = sorted(set(
            int(s) for s in aux_candidate_slots if s in prescanned_grad_slots
        ))

        for idx, out in enumerate(slot_outputs):
            pred_x0 = out.pred_x0
            if not pred_x0.requires_grad:
                # Pipeline should only emit grad-enabled slots, but be defensive.
                continue
            slot_idx = int(out.slot_idx)
            grad_slot_indices.append(slot_idx)
            slot_t = int(out.slot_timestep)
            prompt_embeds = out.prompt_embeds
            action_frame = out.action_frame

            # ---- DMD (legacy per-slot path; safety fallback ONLY) ----
            # The pipeline attaches the full batched-S_n context to the
            # first record of every rolling step (see
            # pipeline/rolling_staircase_training.py), so in normal runs
            # `use_batched_dmd=True` and this branch is skipped. It
            # remains as a defensive fallback in case the pipeline ever
            # drops those attachments, or a future debug path wants the
            # old per-slot behavior.
            slot_loss: Optional[torch.Tensor] = None
            weight = 1.0 if slot_idx == 0 else float(aux_loss_weight)
            dmd_loss: Optional[torch.Tensor] = None
            dmd_log: Dict[str, torch.Tensor] = {}
            skip_dmd = bool(getattr(self.args, "skip_dmd_for_timing", False))
            if not use_batched_dmd and not skip_dmd:
                dmd_loss, dmd_log = self._dmd_loss_for_slot(
                    pred_x0=pred_x0,
                    slot_timestep=slot_t,
                    prompt_embeds=prompt_embeds,
                    action_frame=action_frame,
                )
                slot_loss = dmd_loss * self.dmd_loss_weight * weight

            # ---- GAN on clean x0 (gated; dropped for Phase-1 DMD+fake) ----
            gan_loss: torch.Tensor = pred_x0.new_zeros(())
            gan_log: Dict[str, torch.Tensor] = {}
            if self.gan_loss_weight > 0.0:
                gan_loss, gan_log = self._gan_loss_for_slot(
                    pred_x0=pred_x0,
                    prompt_embeds=prompt_embeds,
                    action_frame=action_frame,
                )
                if gan_loss.requires_grad or (
                    isinstance(gan_loss, torch.Tensor) and float(gan_loss.item()) != 0.0
                ):
                    gan_term = gan_loss * weight
                    slot_loss = (slot_loss + gan_term) if slot_loss is not None else gan_term

            # ---- action_critic aux (frozen-critic guidance on pred_x0) ----
            # Gated by `aux_loss_slot_policy` (see above). Default is
            # "slot0" which supervises only slot 0 because slots 1..NS-1
            # receive decayed commanded actions (e.g. 0.75/0.5/0.25 x a_t
            # in 4x1, or 0.5 x a_t in 2x2) and regressing the commanded
            # action against those would pull the student toward
            # systematically muted motion. When teacher-z supervision is
            # active, the target is derived from pixels (via VAE →
            # CoTracker → ss_vae) per-slot so the bias does not apply —
            # switching to "match_dmd" or "random" lets the aux follow
            # whichever slot the random chooser picked.
            critic_loss: Optional[torch.Tensor] = None
            if slot_idx in aux_active_slots:
                critic_teacher_target: Optional[torch.Tensor] = None
                if teacher_z_per_slot is not None:
                    if teacher_z_per_slot.dim() != 3:
                        raise RuntimeError(
                            f"teacher_z_per_slot must be [B, NS, z_out]; got "
                            f"{tuple(teacher_z_per_slot.shape)}."
                        )
                    if teacher_z_per_slot.shape[1] <= slot_idx:
                        raise RuntimeError(
                            f"teacher_z_per_slot has {teacher_z_per_slot.shape[1]} "
                            f"slots; cannot index slot {slot_idx}."
                        )
                    critic_teacher_target = teacher_z_per_slot[:, slot_idx, :]
                critic_loss = self._critic_aux_loss_for_slot(
                    pred_x0=pred_x0,
                    slot_timestep=slot_t,
                    action_frame=action_frame,
                    teacher_z_target=critic_teacher_target,
                )
                if isinstance(critic_loss, torch.Tensor) and critic_loss.requires_grad:
                    critic_term = critic_loss * weight
                    slot_loss = (slot_loss + critic_term) if slot_loss is not None else critic_term

            # Accumulate per-slot loss into total (when present).
            if slot_loss is not None:
                total_loss = (slot_loss if total_loss is None else total_loss + slot_loss)
                per_slot_losses.setdefault(slot_idx, []).append(slot_loss.detach())

            # Per-record logging (prefix by order + slot for clarity).
            rec_prefix = f"slot{slot_idx}_rec{idx:03d}"
            if dmd_loss is not None:
                log_dict[f"{rec_prefix}/dmd_loss"] = dmd_loss.detach()
                for k, v in dmd_log.items():
                    log_dict[f"{rec_prefix}/{k}"] = v
            if self.gan_loss_weight > 0.0 and (
                gan_loss.requires_grad or gan_loss.item() != 0.0
            ):
                log_dict[f"{rec_prefix}/gan_loss"] = gan_loss.detach()
            if (
                critic_loss is not None
                and isinstance(critic_loss, torch.Tensor)
                and critic_loss.requires_grad
            ):
                log_dict[f"{rec_prefix}/critic_aux_loss"] = critic_loss.detach()
            for k, v in gan_log.items():
                log_dict[f"{rec_prefix}/{k}"] = v

        # ---- state_probe aux (one shared forward per rolling step) ----
        # Pipeline attaches `state_preds_live` and `per_slot_actions_live`
        # to the FIRST grad-slot record of each step — read them here once
        # and add a single supervision term that backprops into the DiT
        # via the tapped activations' shared graph.
        probe_rec = next(
            (
                r for r in slot_outputs
                if getattr(r, "state_preds_live", None) is not None
                and getattr(r, "per_slot_actions_live", None) is not None
            ),
            None,
        )
        # Probe supervision slot-set: governed by `aux_loss_slot_policy`.
        # Default ("slot0") restricts to {0} because slots 1..NS-1 are
        # conditioned on decayed actions (0.75/0.5/0.25 x a_t for 4x1, or
        # 0.5 x a_t for 2x2) — their probe readouts aren't being asked to
        # match the commanded `a_t`, so commanded-action fallback would
        # bias the student toward muted motion. The teacher-z branch
        # reads target z per-slot directly from pixels via
        # VAE→CoTracker→ss_vae, so "match_dmd"/"random" policies are safe
        # when the teacher is active. In all cases we intersect with the
        # grad-slot set to guarantee a live autograd path.
        probe_grad_slots = [s for s in grad_slot_indices if s in aux_active_slots]
        if (
            probe_rec is not None
            and self.state_probe is not None
            and self.state_probe_loss_weight > 0.0
            and probe_grad_slots
        ):
            # teacher_z_per_slot uses the same slot layout as
            # per_slot_actions; passing the full [B, NS, z_out] tensor is
            # fine — the loss indexes by `probe_grad_slots=[0]` so only
            # slot 0's teacher vector is consumed.
            probe_loss = self._state_probe_aux_loss(
                state_preds=probe_rec.state_preds_live,
                per_slot_actions=probe_rec.per_slot_actions_live,
                grad_slot_indices=probe_grad_slots,
                teacher_z_target=teacher_z_per_slot,
            )
            if isinstance(probe_loss, torch.Tensor) and probe_loss.requires_grad:
                if total_loss is None:
                    total_loss = probe_loss
                else:
                    total_loss = total_loss + probe_loss
                log_dict["probe_aux_loss"] = probe_loss.detach()

        if total_loss is None:
            total_loss = torch.zeros((), device=self.device, dtype=self.dtype)

        # Average per-slot-index scalar summaries.
        for slot_idx, losses in per_slot_losses.items():
            log_dict[f"loss/slot{slot_idx}_mean"] = torch.stack(losses).mean()
        log_dict["loss/total"] = total_loss.detach()
        log_dict["loss/num_records"] = torch.tensor(float(len(slot_outputs)))
        if fake_denoise_loss is not None:
            log_dict["loss/fake_denoise"] = fake_denoise_loss.detach()

        return total_loss, log_dict, fake_denoise_loss

    def critic_loss(self, *args, **kwargs):
        """Fake-score denoising is now computed inline inside
        ``generator_loss_on_slots`` via Option-A shared fake forward.
        Keeping this stub so legacy callers don't break; returns a
        zero tensor + empty dict."""
        return torch.zeros((), device=self.device, dtype=self.dtype), {}
