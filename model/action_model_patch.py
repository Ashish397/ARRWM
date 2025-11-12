# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA-4.0).
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

"""
Minimal monkey patches so Wan's causal (actor) and bidirectional (critic)
models can accept externally computed adaLN-Zero modulation. The only behaviour
change should be:

  • add `_action_modulation` (if provided) to the output of `time_projection`
  • thread `_action_modulation` through `_forward_inference` (actor) or
    `_forward` (critic)

Everything else stays untouched so the no-action path is bit-identical to the
upstream implementation.
"""

from __future__ import annotations

import types
from functools import wraps

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore
except Exception:  # pragma: no cover
    FSDP = None

def _patch_time_projection(model):
    if getattr(model, "_action_tp_patched", False):
        return

    # Inject action modulation right after time_projection.
    tp = model.time_projection
    orig_tp_forward = tp.forward

    def tp_forward_with_action(self_tp, e):
        e0 = orig_tp_forward(e)
        am = getattr(model, "_action_modulation", None)
        if am is None:
            return e0
        if am.dim() != 4:
            raise ValueError(f"action_modulation must be [B, F, 6, dim], got {am.shape}")
        B, F, S, D = am.shape
        am_flat = am.reshape(B * F, S * D).to(device=e0.device, dtype=e0.dtype)
        if am_flat.shape != e0.shape:
            raise ValueError(
                f"action_modulation {am_flat.shape} != time_projection output {e0.shape}"
            )
        return e0 + am_flat

    tp.forward = types.MethodType(tp_forward_with_action, tp)
    model._action_tp_patched = True


def patch_causal_wan_model_for_action(model):
    """Patch Wan's causal model in-place to support external action modulation."""
    if getattr(model, "_action_causal_patched", False):
        return model

    _patch_time_projection(model)
    if not hasattr(model, "_forward_inference"):
        raise AttributeError("Causal Wan model is expected to define `_forward_inference`.")

    # 2) Thread modulation through _forward_inference.
    orig_inf = model._forward_inference

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
        self._action_modulation = action_modulation
        try:
            return orig_inf(x, t, context, seq_len, *args, **kwargs)
        finally:
            self._action_modulation = None

    model._forward_inference = _forward_inference_with_action.__get__(model, type(model))
    model._action_causal_patched = True
    return model


def patch_bidirectional_wan_model_for_action(model):
    """Patch Wan's bidirectional model to support external action modulation."""
    if getattr(model, "_action_bidir_patched", False):
        return model

    _patch_time_projection(model)
    if not hasattr(model, "_forward"):
        raise AttributeError("Bidirectional Wan model is expected to define `_forward`.")

    orig_fwd = model._forward

    @wraps(orig_fwd)
    def _forward_with_action(
        self,
        x,
        t,
        context,
        seq_len,
        *args,
        action_modulation=None,
        **kwargs,
    ):
        self._action_modulation = action_modulation
        try:
            return orig_fwd(x, t, context, seq_len, *args, **kwargs)
        finally:
            self._action_modulation = None

    model._forward = _forward_with_action.__get__(model, type(model))
    model._action_bidir_patched = True
    return model


def apply_action_patches(generator_wrapper):
    """Entry point used by training/inference pipelines."""
    target = generator_wrapper
    if FSDP is not None and isinstance(target, FSDP):
        target = target.module  # type: ignore[attr-defined]
    if getattr(target, "_action_patch_applied", False):
        return generator_wrapper
    if hasattr(target, "model"):
        patch_causal_wan_model_for_action(target.model)
    target._action_patch_applied = True  # type: ignore[attr-defined]
    return generator_wrapper


def apply_action_patches_critic(generator_wrapper):
    """Apply action patches to a critic (bidirectional) WanDiffusionWrapper."""
    target = generator_wrapper
    if FSDP is not None and isinstance(target, FSDP):
        target = target.module  # type: ignore[attr-defined]
    if getattr(target, "_action_patch_applied", False):
        return generator_wrapper
    if hasattr(target, "model"):
        patch_bidirectional_wan_model_for_action(target.model)
    target._action_patch_applied = True  # type: ignore[attr-defined]
    return generator_wrapper
