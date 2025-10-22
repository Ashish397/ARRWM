# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA-4.0).
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

"""
Monkey patches for CausalWanModel to support action modulation injection.

This module dynamically patches the model's forward methods to inject
action modulation parameters into the time embedding path.
"""

import torch
from functools import wraps


def patch_causal_wan_model_for_action(model):
    """
    Patch CausalWanModel to support action modulation injection.
    
    This modifies the _forward_inference method to:
    1. Check for '_action_modulation' in the context
    2. Add action modulation to time modulation (e0)
    3. Pass combined modulation to transformer blocks
    
    Args:
        model: CausalWanModel instance
        
    Returns:
        modified model (same instance, modified in-place)
    """
    # Store original method
    original_forward_inference = model._forward_inference
    
    @wraps(original_forward_inference)
    def _forward_inference_with_action(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=0,
        action_modulation=None,  # NEW parameter
    ):
        """
        Modified _forward_inference that supports action modulation.
        
        The modification happens at the time embedding stage where we add
        action_modulation to the time modulation (e0).
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        
        # Get device
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        
        # Patch embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        
        # ============ MODIFIED: Time embeddings + Action modulation ============
        from wan.modules.utils import sinusoidal_embedding_1d
        
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        
        # Inject action modulation if provided
        if action_modulation is not None:
            # action_modulation shape: [B, F, 6, dim]
            # e0 shape: [B, F, 6, dim]
            # Simply add them (adaLN-Zero: action starts at 0, gradually learns)
            e0 = e0 + action_modulation.to(e0.device).to(e0.dtype)
        # ========================================================================
        
        # Context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))
        
        if self.model_type == 'i2v':
            clip_fea = self.img_emb(clip_fea)
        
        # Prepare block mask
        num_frames = e0.shape[1]
        frame_seqlen = x.shape[1] // num_frames
        
        if not hasattr(self, 'block_mask') or self.block_mask is None:
            block_mask = self._prepare_blockwise_causal_attn_mask(
                device=x.device,
                num_frames=num_frames,
                frame_seqlen=frame_seqlen,
                num_frame_per_block=1,
                local_attn_size=self.local_attn_size
            )
        else:
            block_mask = self.block_mask
        
        # Transformer blocks
        for block in self.blocks:
            block_result = block(
                x, e0, seq_lens, grid_sizes, self.freqs, context, context_lens,
                block_mask, kv_cache, crossattn_cache, current_start, cache_start
            )
            if kv_cache is not None:
                x, cache_update_info = block_result
            else:
                x = block_result
        
        # Head
        e1 = e0.mean(dim=2, keepdim=True)
        x = self.head(x, e1)
        
        # Reshape output
        x = [
            u.unflatten(0, grid_sizes[idx].flip(0).tolist()).permute(0, 3, 1, 2)
            for idx, u in enumerate(x.tensor_split(len(grid_sizes)))
        ]
        x = torch.cat(x)
        
        if kv_cache is not None:
            return x, cache_update_info
        else:
            return x
    
    # Replace the method
    model._forward_inference = _forward_inference_with_action.__get__(model, type(model))
    
    # Also patch the main forward to pass action_modulation through
    original_forward = model.forward
    
    @wraps(original_forward)
    def forward_with_action(
        *args,
        action_modulation=None,
        **kwargs
    ):
        # Check if we're using inference mode (has kv_cache)
        if 'kv_cache' in kwargs and kwargs['kv_cache'] is not None:
            kwargs['action_modulation'] = action_modulation
            return model._forward_inference(*args, **kwargs)
        else:
            # Training mode - need to patch _forward as well
            # For now, just call original
            return original_forward(*args, **kwargs)
    
    model.forward = forward_with_action
    
    print("[ActionPatch] Successfully patched CausalWanModel for action conditioning")
    return model


def patch_wan_wrapper_for_action(wrapper):
    """
    Patch WanDiffusionWrapper to pass action modulation to model.
    
    This extracts '_action_modulation' from conditional_dict and passes it
    to the model's forward method.
    
    Args:
        wrapper: WanDiffusionWrapper instance
        
    Returns:
        modified wrapper (same instance, modified in-place)
    """
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
        """Extract action_modulation from conditional_dict and pass to model."""
        # Extract action modulation if present
        action_modulation = conditional_dict.get('_action_modulation', None)
        
        prompt_embeds = conditional_dict["prompt_embeds"]
        
        # [B, F] -> [B]
        if wrapper.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep
        
        logits = None
        
        # Call model with action_modulation
        if kv_cache is not None:
            flow_pred = wrapper.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep,
                context=prompt_embeds,
                seq_len=wrapper.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
                action_modulation=action_modulation,  # NEW
            ).permute(0, 2, 1, 3, 4)
        else:
            if clean_x is not None:
                flow_pred = wrapper.model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4),
                    t=input_timestep,
                    context=prompt_embeds,
                    seq_len=wrapper.seq_len,
                    clean_x=clean_x.permute(0, 2, 1, 3, 4),
                    aug_t=aug_t,
                    action_modulation=action_modulation,  # NEW
                ).permute(0, 2, 1, 3, 4)
            else:
                if classify_mode:
                    result = wrapper.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep,
                        context=prompt_embeds,
                        seq_len=wrapper.seq_len,
                        classify_mode=True,
                        action_modulation=action_modulation,  # NEW
                    )
                    if isinstance(result, tuple):
                        flow_pred, logits = result
                    else:
                        flow_pred = result
                    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                else:
                    flow_pred = wrapper.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep,
                        context=prompt_embeds,
                        seq_len=wrapper.seq_len,
                        action_modulation=action_modulation,  # NEW
                    ).permute(0, 2, 1, 3, 4)
        
        # Rest of the processing (same as original)
        if classify_mode:
            if wrapper.is_causal:
                x0_pred = wrapper._convert_flow_to_x0_causal(flow_pred, noisy_image_or_video, timestep)
            else:
                x0_pred = wrapper._convert_flow_to_x0(wrapper.scheduler, flow_pred, noisy_image_or_video, input_timestep)
            if logits is not None:
                return logits, x0_pred
            return x0_pred
        else:
            if wrapper.is_causal:
                x0_pred = wrapper._convert_flow_to_x0_causal(flow_pred, noisy_image_or_video, timestep)
            else:
                x0_pred = wrapper._convert_flow_to_x0(wrapper.scheduler, flow_pred, noisy_image_or_video, input_timestep)
            return logits, x0_pred
    
    wrapper.forward = forward_with_action_extraction
    
    print("[ActionPatch] Successfully patched WanDiffusionWrapper for action conditioning")
    return wrapper


def apply_action_patches(generator_wrapper):
    """
    Apply all necessary patches to enable action conditioning.
    
    Args:
        generator_wrapper: WanDiffusionWrapper instance
        
    Returns:
        patched wrapper
    """
    # Patch the underlying model
    if hasattr(generator_wrapper, 'model'):
        patch_causal_wan_model_for_action(generator_wrapper.model)
    
    # Patch the wrapper
    patch_wan_wrapper_for_action(generator_wrapper)
    
    return generator_wrapper

