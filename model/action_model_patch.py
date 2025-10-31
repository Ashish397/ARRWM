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

    B = x.size(0)
    dim1 = x.size(1)
    return [_maybe_cast(x[i].permute(1, 0, 2, 3).contiguous()) for i in range(B)]


def patch_wan_wrapper_for_action(wrapper):
    if getattr(wrapper, "_action_forward_patched", False):
        return wrapper

    original_forward = wrapper.forward

    @wraps(original_forward)
    def forward_with_action_extraction(
        noisy_image_or_video,
        conditional_dict,
        timestep,
        kv_cache=None,
        crossattn_cache=None,
        current_start=None,
        classify_mode=False,
        concat_time_embeddings=False,
        clean_x=None,
        aug_t=None,
        cache_start=None,
    ):
        action_modulation = conditional_dict.get("_action_modulation", None)
        prompt_embeds = conditional_dict["prompt_embeds"]

        # WAN timestep collapse
        if wrapper.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

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
                    result = wrapper.model(
                        x_list,
                        t=input_timestep,
                        context=prompt_embeds,
                        seq_len=wrapper.seq_len,
                        classify_mode=True,
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

        if classify_mode:
            if logits is not None:
                return logits, x0_pred
            return x0_pred

        return logits, x0_pred

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
