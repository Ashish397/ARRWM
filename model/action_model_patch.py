# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA-4.0).
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

"""
Monkey patches for CausalWanModel to support action modulation injection.

This version:
- does NOT reimplement the vendor's _forward_inference;
- injects action right after time_projection, where the shapes match;
- passes action via the wrapper, and converts batched videos to a list;
- **and** normalizes the dtype/device at the place that actually does conv3d
  (patch_embedding.forward) so FSDP/mixed-precision can't surprise us.
"""

import torch
import types
from functools import wraps

try:  # Optional dependency to recognise FSDP wrappers
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore
except Exception:  # pragma: no cover - FSDP not always available
    FSDP = None


# ---------------------------------------------------------------------
# 1) Patch the actual model
# ---------------------------------------------------------------------
def patch_causal_wan_model_for_action(model):
    """
    We patch three things on the model:
      1. time_projection.forward  → add action_modulation
      2. patch_embedding.forward  → ensure input dtype/device == conv dtype/device
      3. _forward_inference       → stash/clear action_modulation per call
    """

    # ----- 1) time_projection.forward -----
    tp = model.time_projection
    orig_tp_forward = tp.forward

    if getattr(model, "_action_tp_patched", False):
        return model

    def tp_forward_with_action(self_tp, e):
        e0 = orig_tp_forward(e)
        am = getattr(model, "_action_modulation", None)
        if am is None:
            return e0

        if am.dim() != 4:
            raise ValueError(f"action_modulation must be [B, F, 6, dim], got {am.shape}")
        B, F, S, D = am.shape
        am_flat = am.reshape(B * F, S * D).to(e0.device).to(e0.dtype)
        if am_flat.shape != e0.shape:
            raise ValueError(
                f"action_modulation {am_flat.shape} != time_projection output {e0.shape}"
            )
        return e0 + am_flat

    tp.forward = types.MethodType(tp_forward_with_action, tp)

    model._action_tp_patched = True

    # ----- 2) patch_embedding.forward -----
    # this is the key new bit
    pe = model.patch_embedding
    orig_pe_forward = pe.forward

    def pe_forward_cast(self_pe, x):
        """
        WAN _forward_inference does:
            self.patch_embedding(u.unsqueeze(0))
        but under FSDP/mixed precision the conv params/bias can be bf16
        while x is still f32. So we force-match here.
        """
        target_device = self_pe.weight.device
        target_dtype = self_pe.weight.dtype
        if x.device != target_device or x.dtype != target_dtype:
            # if it's a float tensor, cast dtype too
            if x.is_floating_point():
                x = x.to(device=target_device, dtype=target_dtype)
            else:
                x = x.to(device=target_device)
        return orig_pe_forward(x)

    pe.forward = types.MethodType(pe_forward_cast, pe)

    # ----- 3) wrap _forward_inference -----
    orig_inf = model._forward_inference  # bound method

    @wraps(orig_inf)
    def _forward_inference_with_action(
        self,
        x,
        t,
        context,
        seq_len,
        *args,
        action_modulation=None,
        **kwargs,
    ):
        # stash per-call action on the model
        self._action_modulation = action_modulation
        try:
            # orig_inf is already bound → don't pass self
            return orig_inf(x, t, context, seq_len, *args, **kwargs)
        finally:
            self._action_modulation = None

    model._forward_inference = _forward_inference_with_action.__get__(model, type(model))
    return model


# ---------------------------------------------------------------------
# 2) Wrapper patch
# ---------------------------------------------------------------------
def _as_video_list(x, target_device=None, target_dtype=None):
    """
    WAN causal model wants: list of length B, each [C, T, H, W].

    We commonly get:
      - list/tuple already → return as list
      - tensor [B, C, T, H, W] → split
      - tensor [B, T, C, H, W] → permute → split

    We *also* optionally cast to target_device/target_dtype, but the real
    safety net is in patch_embedding.forward above.
    """
    if x is None:
        return None

    def _maybe_cast(t):
        return t.to(device=target_device, dtype=target_dtype)

    if isinstance(x, (list, tuple)):
        return [_maybe_cast(t) for t in x]

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Unsupported noisy input type: {type(x)}")

    B = x.size(0)
    if x.dim() == 5 and x.size(1) in (3, 16):  # [B, C, T, H, W]
        return [_maybe_cast(x[i].permute(1, 0, 2, 3).contiguous()) for i in range(B)]
    if x.dim() == 5 and x.size(2) in (3, 16):  # [B, T, C, H, W]
        return [_maybe_cast(x[i].permute(2, 1, 3, 4).contiguous()) for i in range(B)]
    raise ValueError(f"Expected video tensor with channel/time dims, got shape {tuple(x.shape)}")


def _coerce_int(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.item()
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clamp_start(index, total_frames, window):
    if total_frames < window:
        raise ValueError(
            f"action_modulation only has {total_frames} frames, but generator needs {window}"
        )
    max_start = max(total_frames - window, 0)
    if index is None:
        return 0
    return min(max(index, 0), max_start)


def _slice_action_modulation(action_modulation, chunk_frames, frame_hint, current_start, frame_seq_len):
    if action_modulation is None:
        return None
    if action_modulation.dim() != 4:
        raise ValueError(f"action_modulation must be [B, F, 6, dim], got {action_modulation.shape}")

    total_frames = action_modulation.shape[1]
    if total_frames == chunk_frames:
        return action_modulation

    hint_idx = _coerce_int(frame_hint)
    if hint_idx is not None:
        start = _clamp_start(hint_idx, total_frames, chunk_frames)
        return action_modulation[:, start:start + chunk_frames]

    curr_start_val = _coerce_int(current_start)
    if curr_start_val is None:
        start = _clamp_start(0, total_frames, chunk_frames)
        return action_modulation[:, start:start + chunk_frames]

    if frame_seq_len is None or frame_seq_len <= 0:
        frame_seq_len = 1

    token_based = curr_start_val // frame_seq_len
    # Fallback if caller already passed frame indices instead of tokens
    if token_based + chunk_frames > total_frames and curr_start_val + chunk_frames <= total_frames:
        token_based = curr_start_val

    start = _clamp_start(token_based, total_frames, chunk_frames)
    return action_modulation[:, start:start + chunk_frames]


def patch_wan_wrapper_for_action(wrapper):
    if getattr(wrapper, "_action_forward_patched", False):
        return wrapper

    original_forward = wrapper.forward
    frame_seq_len_hint = getattr(wrapper, "_action_frame_seq_length", None)
    if frame_seq_len_hint is None:
        frame_seq_len_hint = getattr(wrapper, "frame_seq_length", 1560)
        wrapper._action_frame_seq_length = frame_seq_len_hint

    @wraps(original_forward)
    def forward_with_action_extraction(
        noisy_image_or_video,
        conditional_dict,
        timestep,
        kv_cache=None,
        crossattn_cache=None,
        current_start=None,
        classify_mode=False,
        regress_mode=False,
        concat_time_embeddings=False,
        clean_x=None,
        aug_t=None,
        cache_start=None,
    ):
        action_modulation = conditional_dict.get("_action_modulation", None)
        frame_start_hint = conditional_dict.get("_action_frame_start", None)
        prompt_embeds = conditional_dict["prompt_embeds"]

        # WAN timestep collapse
        if wrapper.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        chunk_frames = noisy_image_or_video.shape[1] if isinstance(noisy_image_or_video, torch.Tensor) else None
        if action_modulation is not None and chunk_frames is not None:
            action_modulation = _slice_action_modulation(
                action_modulation,
                chunk_frames,
                frame_start_hint,
                current_start,
                frame_seq_len_hint,
            )

        # try to get target device/dtype from patch_embedding, but final guard is in model
        mp = wrapper.model.patch_embedding
        target_device = mp.weight.device
        target_dtype = mp.weight.dtype

        x_list = _as_video_list(
            noisy_image_or_video,
            target_device=target_device,
            target_dtype=target_dtype,
        )
        clean_x_list = _as_video_list(
            clean_x,
            target_device=target_device,
            target_dtype=target_dtype,
        ) if clean_x is not None else None

        logits = None

        if kv_cache is not None:
            flow_pred = wrapper.model(
                x_list,
                t=input_timestep,
                context=prompt_embeds,
                seq_len=wrapper.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
                action_modulation=action_modulation,
            ).permute(0, 2, 1, 3, 4)
        else:
            if clean_x_list is not None:
                flow_pred = wrapper.model(
                    x_list,
                    t=input_timestep,
                    context=prompt_embeds,
                    seq_len=wrapper.seq_len,
                    clean_x=clean_x_list,
                    aug_t=aug_t,
                    action_modulation=action_modulation,
                ).permute(0, 2, 1, 3, 4)
            else:
                if classify_mode:
                    if not all(hasattr(wrapper, attr) for attr in ("_register_tokens", "_cls_pred_branch", "_gan_ca_blocks")):
                        raise RuntimeError("Classify mode requested but classification heads are not initialized.")
                    result = wrapper.model(
                        x_list,
                        t=input_timestep,
                        context=prompt_embeds,
                        seq_len=wrapper.seq_len,
                        classify_mode=True,
                        register_tokens=getattr(wrapper, "_register_tokens", None),
                        cls_pred_branch=getattr(wrapper, "_cls_pred_branch", None),
                        gan_ca_blocks=getattr(wrapper, "_gan_ca_blocks", None),
                        concat_time_embeddings=concat_time_embeddings,
                        action_modulation=action_modulation,
                    )
                    if isinstance(result, tuple):
                        flow_pred, logits = result
                    else:
                        flow_pred = result
                    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                elif regress_mode:
                    required = ("_register_tokens_rgs", "_rgs_pred_branch", "_gan_ca_blocks_rgs", "num_frames", "num_class")
                    if not all(hasattr(wrapper, attr) for attr in required):
                        raise RuntimeError("Regress mode requested but regression heads are not initialized.")
                    result = wrapper.model(
                        x_list,
                        t=input_timestep,
                        context=prompt_embeds,
                        seq_len=wrapper.seq_len,
                        regress_mode=True,
                        register_tokens_rgs=getattr(wrapper, "_register_tokens_rgs", None),
                        rgs_pred_branch=getattr(wrapper, "_rgs_pred_branch", None),
                        gan_ca_blocks_rgs=getattr(wrapper, "_gan_ca_blocks_rgs", None),
                        num_frames_rgs=getattr(wrapper, "num_frames", None),
                        num_class_rgs=getattr(wrapper, "num_class", None),
                        concat_time_embeddings=concat_time_embeddings,
                        action_modulation=action_modulation,
                    )
                    if isinstance(result, tuple):
                        flow_pred, logits = result
                    else:
                        flow_pred = result
                    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                else:
                    flow_pred = wrapper.model(
                        x_list,
                        t=input_timestep,
                        context=prompt_embeds,
                        seq_len=wrapper.seq_len,
                        action_modulation=action_modulation,
                    ).permute(0, 2, 1, 3, 4)

        # postprocess
        x0_pred = wrapper._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])

        if logits is not None:
            return flow_pred, x0_pred, logits

        return flow_pred, x0_pred

    wrapper.forward = forward_with_action_extraction
    print("[ActionPatch] Successfully patched WanDiffusionWrapper for action conditioning")
    wrapper._action_forward_patched = True
    return wrapper


# ---------------------------------------------------------------------
# 3) convenience
# ---------------------------------------------------------------------
def apply_action_patches(generator_wrapper):
    target_wrapper = generator_wrapper

    # If the incoming module is wrapped by FSDP, patch the underlying module instead
    if FSDP is not None and isinstance(generator_wrapper, FSDP):
        target_wrapper = generator_wrapper.module  # type: ignore[attr-defined]

    if getattr(target_wrapper, "_action_patch_applied", False):
        return generator_wrapper

    if hasattr(target_wrapper, "model"):
        patch_causal_wan_model_for_action(target_wrapper.model)
    patch_wan_wrapper_for_action(target_wrapper)
    target_wrapper._action_patch_applied = True  # type: ignore[attr-defined]
    return generator_wrapper
