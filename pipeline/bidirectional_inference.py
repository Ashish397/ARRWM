"""Bidirectional inference pipeline for Wan flow-matching models."""

from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager, List, Optional, Tuple

import torch
from torch import nn

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class BidirectionalInferencePipeline(nn.Module):
    """Minimal inference wrapper that integrates the Wan flow field with UniPC."""

    def __init__(
        self,
        args,
        device: torch.device,
        *,
        generator: Optional[WanDiffusionWrapper] = None,
        text_encoder: Optional[WanTextEncoder] = None,
        vae: Optional[WanVAEWrapper] = None,
    ) -> None:
        super().__init__()
        self.args = args
        self.device = device

        self.text_pre_encoded = bool(getattr(args, "text_pre_encoded", False))
        self.guidance_scale = float(getattr(args, "guidance_scale", 0.0))
        self.negative_prompt = getattr(args, "negative_prompt", "")
        self.sampling_steps = int(getattr(args, "sampling_steps", 50))
        self.sampling_shift = float(
            getattr(args, "sampling_shift", getattr(args, "timestep_shift", 5.0))
        )
        self.num_train_timestep = int(getattr(args, "num_train_timestep", 1000))

        # Models -----------------------------------------------------------------
        self.model_name = self._resolve_model_name(args, generator)

        if generator is None:
            wrapper_kwargs = self._to_dict(getattr(args, "model_kwargs", None))
            wrapper_kwargs.setdefault("model_name", self.model_name)
            generator = WanDiffusionWrapper(
                **wrapper_kwargs,
                is_causal=False,
            )
        self.generator: WanDiffusionWrapper = generator.to(device)
        self.generator.eval()

        if self.text_pre_encoded:
            self.text_encoder = text_encoder
        else:
            if text_encoder is None:
                text_encoder = WanTextEncoder(model_name=self.model_name)
            self.text_encoder = text_encoder.to(device)
            self.text_encoder.eval()

        if vae is None:
            vae = WanVAEWrapper(model_name=self.model_name)
        self.vae: WanVAEWrapper = vae.to(device)
        self.vae.eval()

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    @property
    def model_dtype(self) -> torch.dtype:
        return next(self.generator.parameters()).dtype

    def _autocast(self) -> ContextManager:
        if self.device.type != "cuda":
            return nullcontext()
        if self.model_dtype == torch.bfloat16:
            return torch.cuda.amp.autocast(dtype=torch.bfloat16)
        if self.model_dtype == torch.float16:
            return torch.cuda.amp.autocast(dtype=torch.float16)
        return nullcontext()


    # ---------------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: Optional[List[str]] = None,
        *,
        prompt_embeds: Optional[torch.Tensor] = None,
        return_latents: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run UniPC sampling to generate a video."""

        dtype = self.model_dtype
        latents = noise.to(device=self.device, dtype=dtype)

        conditional, unconditional = self._prepare_conditionals(
            text_prompts=text_prompts,
            prompt_embeds=prompt_embeds,
            dtype=dtype,
        )

        scheduler = self._build_scheduler()
        timesteps = scheduler.timesteps.to(device=self.device, dtype=torch.float32)

        batch_size, num_frames = latents.shape[:2]

        for idx, t in enumerate(timesteps):
            step_value = float(t.item())
            model_timestep = torch.full(
                (batch_size, num_frames),
                int(round(step_value)),
                device=self.device,
                dtype=torch.long,
            )

            with self._autocast():
                flow_cond, _ = self.generator(
                    noisy_image_or_video=latents,
                    conditional_dict=conditional,
                    timestep=model_timestep,
                )

            if unconditional is not None and self.guidance_scale > 0:
                with self._autocast():
                    flow_uncond, _ = self.generator(
                        noisy_image_or_video=latents,
                        conditional_dict=unconditional,
                        timestep=model_timestep,
                    )
                flow_pred = flow_uncond + self.guidance_scale * (flow_cond - flow_uncond)
            else:
                flow_pred = flow_cond

            flow_pred_fp32 = flow_pred.to(dtype=torch.float32)
            latents_fp32 = latents.to(dtype=torch.float32)

            latents = scheduler.step(
                model_output=flow_pred_fp32,
                timestep=step_value,
                sample=latents_fp32,
                return_dict=False,
            )[0].to(device=self.device, dtype=dtype)

        decoded = self.vae.decode_to_pixel(latents.to(dtype=torch.float32))
        video = (decoded * 0.5 + 0.5).clamp(0, 1)
        if return_latents:
            return video, latents
        return video, None

    @staticmethod
    def _resolve_model_name(args, generator: Optional[WanDiffusionWrapper]) -> str:
        if generator is not None and hasattr(generator, "model_name"):
            candidate = getattr(generator, "model_name", None)
            if isinstance(candidate, str) and candidate:
                return candidate

        for attr in ("model_name", "real_name"):
            candidate = getattr(args, attr, None)
            if isinstance(candidate, str) and candidate:
                return candidate

        model_kwargs = getattr(args, "model_kwargs", None)
        if isinstance(model_kwargs, dict):
            candidate = model_kwargs.get("model_name")
            if isinstance(candidate, str) and candidate:
                return candidate
        candidate = getattr(model_kwargs, "model_name", None) if model_kwargs is not None else None
        if isinstance(candidate, str) and candidate:
            return candidate

        return "Wan2.1-T2V-1.3B"

    @staticmethod
    def _to_dict(maybe_cfg) -> dict:
        if maybe_cfg is None:
            return {}
        if isinstance(maybe_cfg, dict):
            return dict(maybe_cfg)
        try:
            from omegaconf import DictConfig, OmegaConf  # type: ignore
        except ImportError:  # pragma: no cover - OmegaConf optional at runtime
            DictConfig = None  # type: ignore
            OmegaConf = None  # type: ignore
        else:
            if isinstance(maybe_cfg, DictConfig):
                return OmegaConf.to_container(maybe_cfg, resolve=True)  # type: ignore[return-value]
        if hasattr(maybe_cfg, "items"):
            return {k: v for k, v in maybe_cfg.items()}
        return {}

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _build_scheduler(self) -> FlowUniPCMultistepScheduler:
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timestep,
            shift=self.sampling_shift,
            use_dynamic_shifting=False,
        )
        scheduler.set_timesteps(
            self.sampling_steps, device=self.device, shift=self.sampling_shift
        )
        scheduler.set_begin_index(0)
        return scheduler

    def _prepare_conditionals(
        self,
        *,
        text_prompts: Optional[List[str]],
        prompt_embeds: Optional[torch.Tensor],
        dtype: torch.dtype,
    ) -> Tuple[dict, Optional[dict]]:
        if self.text_pre_encoded:
            if prompt_embeds is None:
                raise ValueError(
                    "prompt_embeds must be provided when text_pre_encoded is True."
                )
            embeds = prompt_embeds
            if embeds.dim() == 2:
                embeds = embeds.unsqueeze(0)
            conditional = {"prompt_embeds": embeds.to(device=self.device, dtype=dtype)}
            if self.guidance_scale > 0:
                # Guidance is unsupported without raw prompts.
                return conditional, None
            return conditional, None

        if text_prompts is None:
            raise ValueError("text_prompts must be provided when text_pre_encoded is False.")

        conditional = self.text_encoder(text_prompts=text_prompts)
        conditional["prompt_embeds"] = conditional["prompt_embeds"].to(
            device=self.device, dtype=dtype
        )

        if self.guidance_scale > 0:
            negative_prompt = self.negative_prompt or ""
            uncond_prompts = [negative_prompt] * len(text_prompts)
            unconditional = self.text_encoder(text_prompts=uncond_prompts)
            unconditional["prompt_embeds"] = unconditional["prompt_embeds"].to(
                device=self.device, dtype=dtype
            )
        else:
            unconditional = None

        return conditional, unconditional
