"""LongLive-style causal diffusion training for the action-aware v13 setup."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from model.causal_teacher_streaming import LongRideSequenceBatcher, _SequenceRideSlot
from trainer.causal_diffusion_teacher_train import (
    CausalLoRADiffusionTrainer,
    _annotate_action_video,
    _chunk_actions,
    _frames_to_mp4_bytes,
    _safe_corr,
)
from utils.distributed import barrier
from utils.zarr_dataset import ZarrRideDataset


class CausalLongLiveDiffusionTrainer(CausalLoRADiffusionTrainer):
    """LongLive-style block rollout on top of the causal action-aware trainer."""

    def __init__(self, config):
        self.warm_start_blocks = int(getattr(config, "warm_start_blocks", 1))
        self.rollout_step_frames = int(
            getattr(config, "rollout_step_frames", getattr(config, "streaming_chunk_size", 21))
        )
        self.rollout_until_ride_end = bool(getattr(config, "rollout_until_ride_end", True))
        self.eval_max_rollout_frames = int(getattr(config, "eval_max_rollout_frames", 240))
        self.context_noise = int(getattr(config, "context_noise", 0))
        self.slice_last_frames = int(getattr(config, "slice_last_frames", self.rollout_step_frames))
        super().__init__(config)
        if self.rollout_step_frames % self.num_frame_per_block != 0:
            raise ValueError("rollout_step_frames must be divisible by num_frame_per_block")
        self.warm_start_frames = self.warm_start_blocks * self.num_frame_per_block
        self._slot_rollout_states: List[Dict[str, Any]] = []

    def _load_pretrained_lora_weights(self, ckpt_path: str) -> None:
        """Load all v12 components (LoRA + critic + projections + state probe).

        The v12 checkpoint contains trained LoRA, action_critic,
        action_projection, action_token_projection, and state_probe.
        Loading all of these warm-starts v13 from v12's full trained
        state.  Optimizer state and step counter are NOT restored.
        """
        from pathlib import Path

        path = Path(ckpt_path)
        if not path.exists():
            if self.is_main_process:
                logging.warning("pretrained_lora_ckpt not found: %s", path)
            return
        if self.is_main_process:
            logging.info("Loading pretrained weights (full v12) from %s", path)

        # Load LoRA via parent method
        super()._load_pretrained_lora_weights(ckpt_path)

        # Load auxiliary components from the same checkpoint
        ckpt = torch.load(path, map_location="cpu")

        if self.action_critic is not None and "action_critic" in ckpt:
            critic_mod = self.action_critic.module if isinstance(self.action_critic, DDP) else self.action_critic
            missing, unexpected = critic_mod.load_state_dict(ckpt["action_critic"], strict=False)
            if self.is_main_process:
                logging.info("Loaded v12 action critic (%d missing, %d unexpected keys)",
                             len(missing), len(unexpected))

        if self.action_projection is not None and "action_projection" in ckpt:
            self.action_projection.load_state_dict(ckpt["action_projection"])
            if self.is_main_process:
                logging.info("Loaded v12 action modulation projection")

        if self.action_token_projection is not None and "action_token_projection" in ckpt:
            self.action_token_projection.load_state_dict(ckpt["action_token_projection"])
            if self.is_main_process:
                logging.info("Loaded v12 action token projection")

        base = self.model.module if isinstance(self.model, DDP) else self.model
        if hasattr(base, "_state_probe") and "state_probe" in ckpt:
            missing, unexpected = base._state_probe.load_state_dict(
                ckpt["state_probe"], strict=False,
            )
            if self.is_main_process:
                logging.info("Loaded v12 state probe (%d missing, %d unexpected keys)",
                             len(missing), len(unexpected))

    def _compute_action_critic_losses(self, pred_x0, target_action_z, timesteps, current_step):
        """Wrap parent critic losses in autocast and use retain_graph for critic backward.

        In v12, the critic + generator losses are computed inside the same autocast
        block as the flow loss, so the scaler handles everything in one backward.
        In v13, the rollout's autocast scope is separate, so critic backward would
        free the graph before the main backward. We detach pred_x0 for critic-only
        training and compute generator guidance as a separate differentiable pass.
        """
        critic_mod = self.action_critic.module if isinstance(self.action_critic, DDP) else self.action_critic
        chunk_frames = self.num_frame_per_block
        B = pred_x0.shape[0]
        n_chunks = pred_x0.shape[1] // chunk_frames

        chunk_t = timesteps[:, ::chunk_frames][:, :n_chunks]
        chunk_actions = _chunk_actions(target_action_z, chunk_frames)[:, :n_chunks]

        zero = torch.tensor(0.0, device=pred_x0.device)

        # Teacher targets
        with autocast(dtype=self.autocast_dtype, enabled=self.use_mixed_precision):
            teacher_z_8d = self._compute_action_teacher_targets(pred_x0.detach())
        teacher_z_8d = teacher_z_8d[:, :n_chunks]

        pred_x0_detached = pred_x0.detach()

        # Critic training (fully detached — plain backward, no scaler interaction)
        for _k in range(self.critic_updates_per_step):
            self.critic_optimizer.zero_grad(set_to_none=True)
            with autocast(dtype=self.autocast_dtype, enabled=self.use_mixed_precision):
                pred_z = critic_mod(pred_x0_detached, chunk_t, chunk_actions)
                pred_z = pred_z[:, :n_chunks]
                critic_z_loss = self._weighted_z_mse(pred_z, teacher_z_8d)
                critic_loss_k = self.action_critic_z_loss_weight * critic_z_loss

            critic_loss_k.float().backward()

            if self.grad_clip is not None and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(critic_mod.parameters(), self.grad_clip)

            self.critic_optimizer.step()

        # Generator guidance (gradient through pred_x0)
        warmup_start = self.warmup_steps
        if current_step < warmup_start:
            guidance_scale = 0.0
        elif self.z_guidance_warmup_steps > 0:
            ramp = min(1.0, (current_step - warmup_start) / self.z_guidance_warmup_steps)
            guidance_scale = ramp * self.generator_action_z_guidance_weight
        else:
            guidance_scale = self.generator_action_z_guidance_weight

        if guidance_scale > 0:
            critic_mod.requires_grad_(False)
            with autocast(dtype=self.autocast_dtype, enabled=self.use_mixed_precision):
                gen_pred_z = critic_mod(pred_x0, chunk_t, chunk_actions)
                gen_pred_z = gen_pred_z[:, :n_chunks]
                gen_z2z7 = gen_pred_z[:, :, self.action_critic_dims]
                target_z2z7 = 1.1 * chunk_actions
                gen_z_loss = F.mse_loss(gen_z2z7, target_z2z7)
                generator_action_loss = guidance_scale * gen_z_loss
            critic_mod.requires_grad_(True)
        else:
            gen_z_loss = zero
            generator_action_loss = zero

        with torch.no_grad():
            z2_idx, z7_idx = self.action_critic_dims[0], self.action_critic_dims[1]
            z2_mse = F.mse_loss(pred_z[:, :, z2_idx], teacher_z_8d[:, :, z2_idx]).item()
            z7_mse = F.mse_loss(pred_z[:, :, z7_idx], teacher_z_8d[:, :, z7_idx]).item()

        logs = {
            "train/critic_z_loss": critic_z_loss.detach().item(),
            "train/critic_loss": critic_loss_k.detach().item(),
            "train/critic_z2_mse": z2_mse,
            "train/critic_z7_mse": z7_mse,
            "train/gen_z_loss": gen_z_loss.detach().item() if torch.is_tensor(gen_z_loss) else 0.0,
            "train/gen_action_loss": generator_action_loss.detach().item() if torch.is_tensor(generator_action_loss) else 0.0,
            "train/teacher_z2_mean": teacher_z_8d[:, :, z2_idx].mean().item(),
            "train/teacher_z7_mean": teacher_z_8d[:, :, z7_idx].mean().item(),
            "train/z_guidance_scale": guidance_scale,
        }
        return generator_action_loss, logs, teacher_z_8d

    @property
    def frame_seq_length(self) -> int:
        base_model = self._causal_model()
        extra = int(getattr(base_model, "action_tokens_per_frame", 0))
        return 1560 + extra

    def _causal_model(self):
        wrapper = self.model.module if isinstance(self.model, DDP) else self.model
        model = wrapper.model
        if hasattr(model, "get_base_model"):
            model = model.get_base_model()
        return model

    def _cache_layout(self) -> Tuple[int, int, int]:
        base_model = self._causal_model()
        block = base_model.blocks[0]
        num_heads = int(getattr(block.self_attn, "num_heads", getattr(base_model, "num_heads", 16)))
        head_dim = int(getattr(block.self_attn, "head_dim", base_model.dim // num_heads))
        text_len = int(getattr(base_model, "text_len", 512))
        return num_heads, head_dim, text_len

    def _set_all_modules_max_attention_size(self, local_attn_size_value: int) -> None:
        if local_attn_size_value == -1:
            target_size = 32760
        else:
            target_size = int(local_attn_size_value) * self.frame_seq_length

        wrapper = self.model.module if isinstance(self.model, DDP) else self.model
        if hasattr(wrapper.model, "max_attention_size"):
            try:
                setattr(wrapper.model, "max_attention_size", target_size)
            except Exception:
                pass
        for _, module in wrapper.model.named_modules():
            if hasattr(module, "max_attention_size"):
                try:
                    setattr(module, "max_attention_size", target_size)
                except Exception:
                    pass

    def _apply_rollout_attention_policy(self) -> None:
        base_model = self._causal_model()
        local_attn_size = int(getattr(base_model, "local_attn_size", -1))
        base_model.local_attn_size = local_attn_size
        self._set_all_modules_max_attention_size(local_attn_size)

    def _make_slot_rollout_state(self) -> Dict[str, Any]:
        return {
            "sequence_id": None,
            "kv_cache": None,
            "crossattn_cache": None,
            "current_frames": 0,
        }

    def _clear_cache_tensors(self, cache_blocks: Optional[List[dict]]) -> None:
        if cache_blocks is None:
            return
        for blk in cache_blocks:
            blk["k"].zero_()
            blk["v"].zero_()
            if "global_end_index" in blk:
                blk["global_end_index"].zero_()
            if "local_end_index" in blk:
                blk["local_end_index"].zero_()
            if "is_init" in blk:
                blk["is_init"] = False

    def _allocate_slot_caches(self) -> Tuple[List[dict], List[dict]]:
        base_model = self._causal_model()
        num_blocks = len(base_model.blocks)
        num_heads, head_dim, text_len = self._cache_layout()
        local_attn_size = int(getattr(base_model, "local_attn_size", -1))
        if local_attn_size == -1:
            kv_frames = max(self.rollout_step_frames, self.slice_last_frames)
        else:
            kv_frames = local_attn_size + self.slice_last_frames
        kv_cache_size = kv_frames * self.frame_seq_length

        kv_cache = []
        for _ in range(num_blocks):
            kv_cache.append({
                "k": torch.zeros([1, kv_cache_size, num_heads, head_dim], dtype=self.dtype, device=self.device),
                "v": torch.zeros([1, kv_cache_size, num_heads, head_dim], dtype=self.dtype, device=self.device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=self.device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=self.device),
            })

        crossattn_cache = []
        for _ in range(num_blocks):
            crossattn_cache.append({
                "k": torch.zeros([1, text_len, num_heads, head_dim], dtype=self.dtype, device=self.device),
                "v": torch.zeros([1, text_len, num_heads, head_dim], dtype=self.dtype, device=self.device),
                "is_init": False,
            })

        return kv_cache, crossattn_cache

    def _reset_slot_rollout_state(self, slot_idx: int) -> None:
        state = self._slot_rollout_states[slot_idx]
        self._clear_cache_tensors(state.get("kv_cache"))
        self._clear_cache_tensors(state.get("crossattn_cache"))
        state["sequence_id"] = None
        state["current_frames"] = 0

    def _ensure_slot_rollout_state(self, slot_idx: int, slot: _SequenceRideSlot) -> Dict[str, Any]:
        state = self._slot_rollout_states[slot_idx]
        if state["kv_cache"] is None or state["crossattn_cache"] is None:
            kv_cache, crossattn_cache = self._allocate_slot_caches()
            state["kv_cache"] = kv_cache
            state["crossattn_cache"] = crossattn_cache
        if state["sequence_id"] != slot.sequence_id:
            self._reset_slot_rollout_state(slot_idx)
            state["sequence_id"] = slot.sequence_id
        return state

    def _prepare_prompt_embeds(self, prompt_embeds: Any) -> torch.Tensor:
        if isinstance(prompt_embeds, list):
            prompt_embeds = torch.stack(prompt_embeds)
        elif isinstance(prompt_embeds, tuple):
            prompt_embeds = torch.stack(list(prompt_embeds))
        elif not isinstance(prompt_embeds, torch.Tensor):
            prompt_embeds = torch.tensor(prompt_embeds, dtype=torch.float32)
        if prompt_embeds.dim() == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        return prompt_embeds.to(self.device, dtype=self.dtype)

    def _load_slot_latents(self, slot: _SequenceRideSlot, start: int, end: int) -> torch.Tensor:
        latents = ZarrRideDataset.load_latent_chunk(slot.zarr_path, start, end)
        return latents.unsqueeze(0).to(self.device, dtype=torch.float32)

    def _load_slot_z_actions(self, slot: _SequenceRideSlot, start: int, end: int) -> torch.Tensor:
        # Try the path as-is, then try resolving symlinks
        zarr_path = slot.zarr_path
        if zarr_path not in self.dataset._attrs_by_path:
            zarr_path = str(Path(slot.zarr_path).resolve())
        actions = self.dataset.encode_z_actions_window(
            zarr_path, slot.n_latent_frames, start, end,
        )
        return actions.unsqueeze(0).to(self.device, dtype=self.dtype)

    def _build_rollout_conditional(
        self,
        prompt_embeds: torch.Tensor,
        z_rollout: torch.Tensor,
        num_frames: int,
    ) -> Dict[str, torch.Tensor]:
        conditional = {"prompt_embeds": prompt_embeds}
        if self.use_action_conditioning and self.action_projection is not None:
            conditional["_action_modulation"] = self.action_projection(z_rollout, num_frames=num_frames)
        if self.use_action_conditioning and self.action_token_projection is not None:
            conditional["_action_tokens"] = self.action_token_projection(z_rollout)
        return conditional

    def _warm_start_slot(
        self,
        forward_model,
        slot_idx: int,
        slot: _SequenceRideSlot,
    ) -> None:
        if slot.warm_started or slot.warm_start_frames <= 0:
            return

        state = self._ensure_slot_rollout_state(slot_idx, slot)
        start, end = self.sequence_batcher.warm_start_bounds(slot_idx)
        if end <= start:
            self.sequence_batcher.mark_warm_started(slot_idx)
            return

        warm_latents = self._load_slot_latents(slot, start, end)
        warm_actions_full = self._load_slot_z_actions(slot, start, end)
        warm_actions = warm_actions_full if self.action_dims is None else warm_actions_full[..., self.action_dims]
        prompt_embeds = self._prepare_prompt_embeds(slot.prompt_embeds)
        conditional = self._build_rollout_conditional(prompt_embeds, warm_actions, warm_latents.shape[1])
        timestep = torch.zeros([1, warm_latents.shape[1]], device=self.device, dtype=torch.float32)

        with torch.no_grad():
            with autocast(dtype=self.autocast_dtype, enabled=self.use_mixed_precision):
                forward_model(
                    warm_latents,
                    conditional,
                    timestep,
                    kv_cache=state["kv_cache"],
                    crossattn_cache=state["crossattn_cache"],
                    current_start=state["current_frames"] * self.frame_seq_length,
                    cache_start=0,
                )

        state["current_frames"] += warm_latents.shape[1]
        self.sequence_batcher.mark_warm_started(slot_idx)

    def _unpack_model_out(self, model_out):
        if isinstance(model_out, tuple) and len(model_out) == 4:
            return model_out[0], model_out[1], model_out[2], model_out[3]
        if isinstance(model_out, tuple) and len(model_out) == 3:
            return model_out[0], model_out[1], model_out[2], None
        return model_out[0], model_out[1], None, None

    def _rollout_slot_chunk(
        self,
        forward_model,
        slot_idx: int,
        slot: _SequenceRideSlot,
        *,
        start: int,
        end: int,
        requires_grad: bool,
        slot_state: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        num_frames = end - start
        if num_frames <= 0:
            return None

        state = slot_state if slot_state is not None else self._ensure_slot_rollout_state(slot_idx, slot)
        prompt_embeds = self._prepare_prompt_embeds(slot.prompt_embeds)
        target_latents = self._load_slot_latents(slot, start, end)
        z_actions_full_raw = self._load_slot_z_actions(slot, start, end)
        z_actions = z_actions_full_raw if self.action_dims is None else z_actions_full_raw[..., self.action_dims]

        timesteps = self._sample_timesteps(1, num_frames)
        noise = torch.randn_like(target_latents)
        noisy_latents = self.scheduler.add_noise(
            target_latents.flatten(0, 1), noise.flatten(0, 1), timesteps.flatten(0, 1),
        ).view_as(target_latents)
        training_target = self.scheduler.training_target(
            target_latents.flatten(0, 1), noise.flatten(0, 1), timesteps.flatten(0, 1),
        ).view_as(target_latents)

        flow_parts = []
        pred_parts = []
        state_pred_parts = []
        state_pooled_parts = []

        for block_start in range(0, num_frames, self.num_frame_per_block):
            block_end = min(block_start + self.num_frame_per_block, num_frames)
            block_noisy = noisy_latents[:, block_start:block_end]
            block_t = timesteps[:, block_start:block_end]
            block_actions = z_actions[:, block_start:block_end]
            conditional = self._build_rollout_conditional(prompt_embeds, block_actions, block_end - block_start)
            current_start = state["current_frames"] * self.frame_seq_length

            with torch.set_grad_enabled(requires_grad):
                with autocast(dtype=self.autocast_dtype, enabled=self.use_mixed_precision):
                    model_out = forward_model(
                        block_noisy,
                        conditional,
                        block_t,
                        kv_cache=state["kv_cache"],
                        crossattn_cache=state["crossattn_cache"],
                        current_start=current_start,
                        cache_start=0,
                    )
                    flow_pred, pred_x0, state_preds, state_pooled = self._unpack_model_out(model_out)

            flow_parts.append(flow_pred)
            pred_parts.append(pred_x0)
            if state_preds is not None:
                state_pred_parts.append(state_preds)
            if state_pooled is not None:
                state_pooled_parts.append(state_pooled)

            context_timestep = torch.ones_like(block_t) * self.context_noise
            context_source = pred_x0.detach()
            if self.context_noise > 0:
                context_input = self.scheduler.add_noise(
                    context_source.flatten(0, 1),
                    torch.randn_like(context_source.flatten(0, 1)),
                    context_timestep.flatten(0, 1),
                ).view_as(context_source)
            else:
                context_input = context_source

            with torch.no_grad():
                with autocast(dtype=self.autocast_dtype, enabled=self.use_mixed_precision):
                    forward_model(
                        context_input,
                        conditional,
                        context_timestep,
                        kv_cache=state["kv_cache"],
                        crossattn_cache=state["crossattn_cache"],
                        current_start=current_start,
                        cache_start=0,
                    )

            state["current_frames"] += (block_end - block_start)

        rollout = {
            "num_frames": num_frames,
            "timesteps": timesteps,
            "training_target": training_target,
            "pred_x0": torch.cat(pred_parts, dim=1),
            "flow_pred": torch.cat(flow_parts, dim=1),
            "z_actions_full_raw": z_actions_full_raw,
            "target_action_z": z_actions_full_raw[..., self.action_critic_dims],
        }
        rollout["state_preds"] = torch.cat(state_pred_parts, dim=1) if state_pred_parts else None
        rollout["state_pooled"] = torch.cat(state_pooled_parts, dim=1) if state_pooled_parts else None
        return rollout

    def _accumulate_log_dict(self, total: Dict[str, float], current: Dict[str, Any]) -> None:
        for key, value in current.items():
            if torch.is_tensor(value):
                if value.numel() != 1:
                    continue
                value = value.detach().item()
            if isinstance(value, (float, int)):
                total[key] = total.get(key, 0.0) + float(value)

    def _compute_rollout_slot_loss(
        self,
        rollout: Dict[str, torch.Tensor],
        current_step: int,
        base_module: torch.nn.Module,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        num_frames = int(rollout["num_frames"])
        flow_loss = self._compute_flow_loss(
            rollout["flow_pred"], rollout["training_target"], rollout["timesteps"], 1, num_frames,
        )
        loss = flow_loss

        metrics: Dict[str, Any] = {
            "flow_loss": flow_loss.detach().item(),
            "state_loss": 0.0,
            "state_guidance_loss": 0.0,
            "diag_corr_state_teacher": 0.0,
            "diag_corr_state_cmd": 0.0,
            "diag_mse_teacher_cmd": 0.0,
            "diag_count": 0,
        }

        teacher_z_8d = None
        critic_logs: Dict[str, Any] = {}
        if self.action_critic_enabled and self.action_critic is not None:
            gen_loss, critic_logs, teacher_z_8d = self._compute_action_critic_losses(
                rollout["pred_x0"], rollout["target_action_z"], rollout["timesteps"], current_step,
            )
            loss = loss + gen_loss
            self._accumulate_log_dict(metrics, critic_logs)

        state_preds = rollout.get("state_preds")
        state_pooled = rollout.get("state_pooled")
        if state_preds is not None and self._state_head_built:
            _need_z_slice = self.state_head_out_dim <= len(self.action_critic_dims)
            n_c_g = state_preds.shape[1]

            if teacher_z_8d is not None:
                t_z = teacher_z_8d[:, :n_c_g]
                state_target = t_z[:, :, self.action_critic_dims] if _need_z_slice else t_z
            else:
                target_z_pf = rollout["z_actions_full_raw"]
                t_z = _chunk_actions(target_z_pf, self.num_frame_per_block)[:, :n_c_g]
                state_target = t_z[:, :, self.action_critic_dims] if _need_z_slice else t_z

            state_z = state_preds[:, :n_c_g].float()
            state_loss = F.mse_loss(state_z, state_target.float())
            loss = loss + self.state_head_loss_weight * state_loss
            metrics["state_loss"] = state_loss.detach().item()

            state_g_scale = self._state_guidance_scale(current_step)
            if state_g_scale > 0 and state_pooled is not None:
                readout = (
                    base_module._state_probe.readout
                    if hasattr(base_module, "_state_probe")
                    else base_module._state_readout
                )
                frozen_preds_raw = F.linear(
                    state_pooled[:, :n_c_g].float(),
                    readout.weight.detach(),
                    readout.bias.detach(),
                )
                if _need_z_slice:
                    frozen_preds = frozen_preds_raw
                else:
                    frozen_preds = frozen_preds_raw[:, :, self.action_critic_dims]
                cmd_z = _chunk_actions(
                    rollout["target_action_z"],
                    self.num_frame_per_block,
                )[:, :n_c_g]
                cmd_target = self.state_guidance_action_scale * cmd_z
                state_guidance_loss = F.mse_loss(frozen_preds, cmd_target.float())
                loss = loss + state_g_scale * state_guidance_loss
                metrics["state_guidance_loss"] = state_guidance_loss.detach().item()

            if teacher_z_8d is not None and self.is_main_process:
                teacher_z27 = teacher_z_8d[:, :n_c_g, self.action_critic_dims].detach()
                state_z27 = state_z if _need_z_slice else state_z[:, :, self.action_critic_dims]
                cmd_z27 = _chunk_actions(
                    rollout["target_action_z"],
                    self.num_frame_per_block,
                )[:, :n_c_g]
                metrics["diag_corr_state_teacher"] = _safe_corr(state_z27, teacher_z27)
                metrics["diag_corr_state_cmd"] = _safe_corr(state_z27, cmd_z27)
                metrics["diag_mse_teacher_cmd"] = F.mse_loss(
                    teacher_z27.float(), cmd_z27.float(),
                ).item()
                metrics["diag_count"] = 1

        metrics["loss"] = loss.detach().item()
        return loss, metrics

    def _next_ride(self) -> dict:
        raw = next(self.data_iter)
        return {
            "zarr_path": raw["zarr_path"][0],
            "prompt_embeds": raw["prompt_embeds"][0],
            "n_latent_frames": int(raw["n_latent_frames"][0].item()),
        }

    def _refill_exhausted_slots(self) -> None:
        needs = self.sequence_batcher.exhausted_slot_indices()
        if not needs:
            return
        rides = [self._next_ride() for _ in needs]
        self.sequence_batcher.refill_slots(needs, rides)
        for idx in needs:
            self._reset_slot_rollout_state(idx)
        if self.is_main_process:
            logging.info("Refilled %d rollout slot(s): %s", len(needs), self.sequence_batcher.summary())

    def train(self) -> None:
        if self.start_step >= self.max_steps:
            if self.is_main_process:
                logging.info("start_step >= max_iters, nothing to train.")
            return
        self._train_longlive()

    def _train_longlive(self) -> None:
        forward_model = self.model
        base_module = self.model.module if isinstance(self.model, DDP) else self.model
        base_module.train()
        self._apply_rollout_attention_policy()

        micro_batch = int(getattr(self.config, "batch_size", 1))
        self.sequence_batcher = LongRideSequenceBatcher(
            window_size=self.rollout_step_frames,
            num_frame_per_block=self.num_frame_per_block,
            batch_size=micro_batch,
            warm_start_blocks=self.warm_start_blocks,
        )
        self._slot_rollout_states = [self._make_slot_rollout_state() for _ in range(micro_batch)]

        total_batch_size = micro_batch * self.gradient_accumulation * self.world_size
        if self.is_main_process:
            logging.info(
                "Starting training (LongLive rollout): global_batch=%d (micro=%d x accum=%d x world=%d), "
                "rollout=%d, block=%d, warm_start=%d",
                total_batch_size,
                micro_batch,
                self.gradient_accumulation,
                self.world_size,
                self.rollout_step_frames,
                self.num_frame_per_block,
                self.warm_start_frames,
            )

        for step in range(self.start_step, self.max_steps):
            if self.is_distributed:
                sampler = self.dataloader.sampler
                if isinstance(sampler, torch.utils.data.distributed.DistributedSampler):
                    sampler.set_epoch(step)

            self._update_lr(step)
            self.optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            accumulated_flow_loss = 0.0
            accumulated_state_loss = 0.0
            accumulated_state_guidance_loss = 0.0
            diag_corr_state_teacher = 0.0
            diag_corr_state_cmd = 0.0
            diag_mse_teacher_cmd = 0.0
            diag_count = 0
            chunk_count = 0
            critic_log_sums: Dict[str, float] = {}
            critic_log_count = 0

            for _ in range(self.gradient_accumulation):
                self._refill_exhausted_slots()
                slots = self.sequence_batcher.get_slot_info()
                active_slot_indices = [idx for idx, slot in enumerate(slots) if not slot.exhausted]

                for slot_idx in active_slot_indices:
                    slot = slots[slot_idx]
                    self._ensure_slot_rollout_state(slot_idx, slot)
                    self._warm_start_slot(forward_model, slot_idx, slot)
                    start, end = self.sequence_batcher.rollout_bounds(slot_idx, self.rollout_step_frames)
                    rollout = self._rollout_slot_chunk(
                        forward_model,
                        slot_idx,
                        slot,
                        start=start,
                        end=end,
                        requires_grad=True,
                    )
                    if rollout is None:
                        self.sequence_batcher.advance_slot(slot_idx, 0)
                        continue

                    loss, metrics = self._compute_rollout_slot_loss(rollout, step, base_module)
                    scale = self.gradient_accumulation * max(1, len(active_slot_indices))
                    scaled_loss = loss / scale
                    if self.scaler.is_enabled():
                        self.scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                    accumulated_loss += metrics["loss"]
                    accumulated_flow_loss += metrics["flow_loss"]
                    if metrics["state_loss"] > 0:
                        accumulated_state_loss += metrics["state_loss"]
                    if metrics["state_guidance_loss"] > 0:
                        accumulated_state_guidance_loss += metrics["state_guidance_loss"]
                    if metrics["diag_count"] > 0:
                        diag_corr_state_teacher += metrics["diag_corr_state_teacher"]
                        diag_corr_state_cmd += metrics["diag_corr_state_cmd"]
                        diag_mse_teacher_cmd += metrics["diag_mse_teacher_cmd"]
                        diag_count += metrics["diag_count"]
                    self._accumulate_log_dict(critic_log_sums, metrics)
                    critic_log_count += 1
                    self.sequence_batcher.advance_slot(slot_idx, rollout["num_frames"])
                    if not self.rollout_until_ride_end:
                        remaining = self.sequence_batcher.remaining_usable_frames(slot_idx)
                        if remaining > 0:
                            self.sequence_batcher.advance_slot(slot_idx, remaining)
                    chunk_count += 1

            grad_norms = {}
            if self.is_main_process and (step + 1) % self.grad_norm_interval == 0:
                grad_norms = self._compute_grad_norms()

            self._optim_step(base_module)

            if (step + 1) % self.log_interval == 0 and chunk_count > 0:
                avg_loss = accumulated_loss / chunk_count
                avg_flow = accumulated_flow_loss / chunk_count
                loss_tensor = torch.tensor(avg_loss, device=self.device)
                flow_tensor = torch.tensor(avg_flow, device=self.device)
                if self.is_distributed:
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                    dist.all_reduce(flow_tensor, op=dist.ReduceOp.AVG)

                if self.is_main_process:
                    logging.info("step %d | loss %.6f | flow %.6f", step + 1, loss_tensor.item(), flow_tensor.item())

                    payload: Dict[str, Any] = {
                        "train/step": step + 1,
                        "train/total_loss": loss_tensor.item(),
                        "train/flow_loss": flow_tensor.item(),
                        "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                        "train/critic_learning_rate": (
                            self.critic_optimizer.param_groups[0]["lr"] if self.critic_optimizer else None
                        ),
                        "train/chunks": chunk_count,
                        "train/chunk_idx": self.sequence_batcher.current_chunk_idx,
                    }
                    if accumulated_state_loss > 0:
                        payload["train/state_z_loss"] = accumulated_state_loss / chunk_count
                    if accumulated_state_guidance_loss > 0:
                        payload["train/state_guidance_loss"] = accumulated_state_guidance_loss / chunk_count
                        payload["train/state_guidance_scale"] = self._state_guidance_scale(step)
                    if diag_count > 0:
                        payload["train/corr_state_teacher_z27"] = diag_corr_state_teacher / diag_count
                        payload["train/corr_state_cmd_z27"] = diag_corr_state_cmd / diag_count
                        payload["train/mse_teacher_cmd_z27"] = diag_mse_teacher_cmd / diag_count
                    if critic_log_count > 0:
                        for key, value in critic_log_sums.items():
                            if key in {
                                "flow_loss", "state_loss", "state_guidance_loss",
                                "diag_corr_state_teacher", "diag_corr_state_cmd",
                                "diag_mse_teacher_cmd", "diag_count", "loss",
                            }:
                                continue
                            payload[f"train/{key}" if not key.startswith("train/") else key] = value / critic_log_count
                    if grad_norms:
                        payload.update(grad_norms)

                    self._wandb_log(payload, step=step + 1)

            if self.ckpt_interval > 0 and (step + 1) % self.ckpt_interval == 0:
                barrier()
                self._save_checkpoint(step + 1)
                barrier()

            self._maybe_eval(step)

            if step == self.start_step:
                torch.cuda.empty_cache()

        barrier()
        self._save_checkpoint(self.max_steps)
        barrier()
        self._maybe_eval(self.max_steps - 1)

    @torch.no_grad()
    def _generate_eval_rollout(self, wrapper, ride: dict):
        slot = _SequenceRideSlot(
            zarr_path=ride["zarr_path"],
            prompt_embeds=ride["prompt_embeds"],
            n_latent_frames=int(ride["n_latent_frames"]),
            start_offset=0,
            cursor=self.warm_start_frames,
            warm_start_frames=self.warm_start_frames,
            warm_started=(self.warm_start_frames == 0),
            exhausted=False,
            sequence_id=1,
            chunk_idx=0,
            loaded=True,
        )
        state = self._make_slot_rollout_state()
        state["kv_cache"], state["crossattn_cache"] = self._allocate_slot_caches()
        state["sequence_id"] = slot.sequence_id

        prompt_embeds = self._prepare_prompt_embeds(slot.prompt_embeds)
        if self.warm_start_frames > 0:
            warm_end = min(self.warm_start_frames, slot.n_latent_frames)
            warm_latents = self._load_slot_latents(slot, 0, warm_end)
            warm_actions_full = self._load_slot_z_actions(slot, 0, warm_end)
            warm_actions = warm_actions_full if self.action_dims is None else warm_actions_full[..., self.action_dims]
            conditional = self._build_rollout_conditional(prompt_embeds, warm_actions, warm_latents.shape[1])
            t_zero = torch.zeros([1, warm_latents.shape[1]], device=self.device, dtype=torch.float32)
            wrapper(
                warm_latents,
                conditional,
                t_zero,
                kv_cache=state["kv_cache"],
                crossattn_cache=state["crossattn_cache"],
                current_start=0,
                cache_start=0,
            )
            state["current_frames"] = warm_latents.shape[1]

        generated_parts = []
        state_parts = []
        action_parts = []
        total_frames = 0
        cursor = slot.cursor
        while cursor < slot.n_latent_frames and total_frames < self.eval_max_rollout_frames:
            chunk_frames = min(self.rollout_step_frames, slot.n_latent_frames - cursor, self.eval_max_rollout_frames - total_frames)
            chunk_frames -= chunk_frames % self.num_frame_per_block
            if chunk_frames < self.num_frame_per_block:
                break
            rollout = self._rollout_slot_chunk(
                wrapper,
                0,
                slot,
                start=cursor,
                end=cursor + chunk_frames,
                requires_grad=False,
                slot_state=state,
            )
            if rollout is None:
                break
            generated_parts.append(rollout["pred_x0"])
            action_parts.append(rollout["target_action_z"])
            if rollout["state_preds"] is not None:
                state_parts.append(rollout["state_preds"])
            cursor += rollout["num_frames"]
            total_frames += rollout["num_frames"]
            if not self.rollout_until_ride_end:
                break

        if not generated_parts:
            return None, None, None

        gen_latents = torch.cat(generated_parts, dim=1)
        state_preds = torch.cat(state_parts, dim=1) if state_parts else None
        target_action_z = torch.cat(action_parts, dim=1)
        return gen_latents, state_preds, target_action_z

    def _maybe_eval(self, step: int) -> None:
        if not self.is_main_process:
            return
        if self.eval_dataset is None or self._frozen_vae is None:
            return
        is_first_step = step == self.start_step
        if not is_first_step and (step + 1) % self.eval_interval != 0:
            return

        logging.info("Running held-out LongLive eval at step %d...", step + 1)
        self._offload_training_state()

        wrapper = self.model.module if isinstance(self.model, DDP) else self.model
        was_training = wrapper.training
        wrapper.eval()
        self._apply_rollout_attention_policy()

        causal_model = wrapper.model
        if hasattr(causal_model, "base_model"):
            causal_model = causal_model.base_model.model
        saved_mask = getattr(causal_model, "block_mask", None)
        causal_model.block_mask = None

        try:
            ride_idx = step % len(self.eval_dataset)
            ride = self.eval_dataset[ride_idx]
            gen_latents, state_preds, target_action_z = self._generate_eval_rollout(wrapper, ride)
            if gen_latents is None:
                logging.warning("Eval ride produced no rollout frames, skipping.")
                return

            video_np = self._decode_latents(gen_latents)
            eval_log: Dict[str, Any] = {"eval/step": step + 1}

            state_z2z7 = None
            if state_preds is not None and self._state_head_built:
                _eval_need_slice = self.state_head_out_dim <= len(self.action_critic_dims)
                state_z2z7 = state_preds.float() if _eval_need_slice else state_preds.float()[:, :, self.action_critic_dims]

            if self.action_critic is not None:
                critic_mod = self.action_critic.module if isinstance(self.action_critic, DDP) else self.action_critic
                critic_mod.eval()

                motion, teacher_z_8d = self._compute_teacher_visuals(gen_latents)
                n_chunks = teacher_z_8d.shape[1]
                target_chunk = _chunk_actions(target_action_z, self.num_frame_per_block)[:, :n_chunks]

                eval_t = torch.zeros(1, n_chunks, device=self.device)
                critic_pred_z = critic_mod(gen_latents, eval_t, target_chunk)[:, :n_chunks]
                teacher_z2z7 = teacher_z_8d[:, :, self.action_critic_dims]
                critic_z2z7 = critic_pred_z[:, :, self.action_critic_dims]

                annotated = _annotate_action_video(
                    video_np,
                    motion,
                    teacher_z2z7,
                    critic_z2z7,
                    target_chunk,
                    title=f"eval step {step + 1}",
                    state_z2z7=state_z2z7,
                )
                eval_log["eval/critic_z_mse"] = self._weighted_z_mse(
                    critic_pred_z.float(), teacher_z_8d.float(),
                ).item()
                z2_idx, z7_idx = self.action_critic_dims[0], self.action_critic_dims[1]
                eval_log["eval/teacher_z2_mean"] = teacher_z_8d[:, :, z2_idx].mean().item()
                eval_log["eval/teacher_z7_mean"] = teacher_z_8d[:, :, z7_idx].mean().item()
                eval_log["eval/critic_z2_mse"] = F.mse_loss(
                    critic_pred_z[:, :, z2_idx].float(), teacher_z_8d[:, :, z2_idx].float()
                ).item()
                eval_log["eval/critic_z7_mse"] = F.mse_loss(
                    critic_pred_z[:, :, z7_idx].float(), teacher_z_8d[:, :, z7_idx].float()
                ).item()

                if state_z2z7 is not None:
                    teacher_z27_chunked = teacher_z2z7[:, :n_chunks]
                    state_z27_trimmed = state_z2z7[:, :n_chunks]
                    eval_log["eval/state_z2_mse"] = F.mse_loss(
                        state_z27_trimmed[:, :, 0], teacher_z27_chunked[:, :, 0]
                    ).item()
                    eval_log["eval/state_z7_mse"] = F.mse_loss(
                        state_z27_trimmed[:, :, 1], teacher_z27_chunked[:, :, 1]
                    ).item()
                if state_preds is not None and self._state_head_built:
                    state_trimmed = state_preds.float()[:, :n_chunks]
                    if _eval_need_slice:
                        teacher_matched = teacher_z2z7[:, :n_chunks]
                    else:
                        teacher_matched = teacher_z_8d[:, :n_chunks]
                    eval_log["eval/state_mse"] = F.mse_loss(
                        state_trimmed, teacher_matched.float(),
                    ).item()

                    state_z27_eval = state_trimmed if _eval_need_slice else state_trimmed[:, :, self.action_critic_dims]
                    teacher_z27_eval = teacher_z2z7[:, :n_chunks]
                    cmd_z27_eval = target_chunk
                    eval_log["eval/corr_state_teacher_z27"] = _safe_corr(state_z27_eval, teacher_z27_eval)
                    eval_log["eval/corr_state_cmd_z27"] = _safe_corr(state_z27_eval, cmd_z27_eval)
                    eval_log["eval/mse_teacher_cmd_z27"] = F.mse_loss(
                        teacher_z27_eval.float(), cmd_z27_eval.float(),
                    ).item()

                critic_mod.train()
            else:
                annotated = video_np

            tmp_path = None
            mp4_bytes = _frames_to_mp4_bytes(annotated, fps=5.0)
            if mp4_bytes is not None:
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                    f.write(mp4_bytes)
                    tmp_path = f.name
                eval_log["eval/video"] = wandb.Video(
                    tmp_path, fps=5, format="mp4", caption=f"step {step + 1}"
                )

            self._wandb_log(eval_log, step=step + 1)
            logging.info("Eval video logged to W&B at step %d", step + 1)

            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        except Exception as exc:
            logging.warning("Eval failed at step %d: %s", step + 1, exc, exc_info=True)
        finally:
            causal_model.block_mask = saved_mask
            if was_training:
                wrapper.train()
            self._restore_training_state()
            torch.cuda.empty_cache()
