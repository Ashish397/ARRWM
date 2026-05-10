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

import math
import types
from functools import wraps

import torch

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore
except Exception:  # pragma: no cover
    FSDP = None

# Re-use the causal model's interleaved-token helpers so the bidirectional
# scorers expose the exact same per-frame layout (spatial_0..spatial_{HW-1},
# action_0..action_{a-1}) that the causal-trained weights were fine-tuned
# against. Any divergence here feeds the scorers an out-of-distribution
# sequence layout and silently degrades the DMD signal.
from wan.modules.causal_model import (
    _separate_action_tokens,
    _merge_action_tokens,
    CausalWanModel,
)
from wan.modules.attention import flash_attention
from wan.modules.model import rope_apply, sinusoidal_embedding_1d
from torch.nn.attention.flex_attention import BlockMask, flex_attention as _flex_attention_raw

# Compiled flex_attention (mirrors v14's compile in causal_model.py:28).
# Available as a utility for any code path that wants to apply a structured
# block_mask. NOTE: the bidirectional DMD scorers (real_score / fake_score)
# in this module DO NOT use it — they always run full-bidir
# ``flash_attention`` (no causal mask, no TF mask). The v14 LoRA training
# itself and the causal student's teacher-forcing training continue to use
# ``CausalWanModel._prepare_teacher_forcing_mask`` upstream.
_flex_attention_compiled = torch.compile(
    _flex_attention_raw, dynamic=False, mode="max-autotune-no-cudagraphs"
)


def _prepare_tf_block_mask_cached(
    model,
    device,
    num_frames: int,
    frame_seqlen: int,
    num_frame_per_block: int,
    context_shift: int,
) -> BlockMask:
    """Build (or fetch from cache) the v14 teacher-forcing block_mask
    used by the upstream causal-Wan training (``CausalWanModel._forward_train``)
    and the v14 LoRA's training. Provided here as a thin caching wrapper
    around ``CausalWanModel._prepare_teacher_forcing_mask`` for any caller
    that needs it.

    NOT used by the bidirectional DMD scorers in this module — those run
    full-bidir ``flash_attention``. Kept as a utility so the v14 / causal-
    teacher training paths can share a cached BlockMask without rebuilding
    per forward (the BlockMask construction allocates a transient
    ~total_length²/128² intermediate, so caching matters).
    """
    cache = getattr(model, "_tf_block_mask_cache", None)
    if cache is None:
        cache = {}
        model._tf_block_mask_cache = cache
    key = (int(num_frames), int(frame_seqlen), int(num_frame_per_block),
           int(context_shift), str(device))
    bm = cache.get(key)
    if bm is not None:
        return bm
    bm = CausalWanModel._prepare_teacher_forcing_mask(
        device,
        num_frames=int(num_frames),
        frame_seqlen=int(frame_seqlen),
        num_frame_per_block=int(num_frame_per_block),
        context_shift=int(context_shift),
    )
    cache[key] = bm
    return bm


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
        action_tokens=None,
        **kwargs,
    ):
        self._action_modulation = action_modulation
        try:
            return orig_inf(x, t, context, seq_len, *args,
                            action_tokens=action_tokens, **kwargs)
        finally:
            self._action_modulation = None

    model._forward_inference = _forward_inference_with_action.__get__(model, type(model))

    # 3) Thread modulation through _forward_train.
    orig_train = model._forward_train

    @wraps(orig_train)
    def _forward_train_with_action(
        self,
        x,
        t,
        context,
        seq_len,
        *args,
        action_modulation=None,
        action_modulation_clean=None,
        action_tokens=None,
        action_tokens_clean=None,
        **kwargs,
    ):
        self._action_modulation = action_modulation
        self._action_modulation_clean = action_modulation_clean
        try:
            return orig_train(x, t, context, seq_len, *args,
                              action_tokens=action_tokens,
                              action_tokens_clean=action_tokens_clean,
                              **kwargs)
        finally:
            self._action_modulation = None
            self._action_modulation_clean = None

    model._forward_train = _forward_train_with_action.__get__(model, type(model))

    model._action_causal_patched = True
    return model


def _patch_bidirectional_self_attn_for_action(attn) -> None:
    """Monkey-patch a bidirectional ``WanSelfAttention`` so it respects an
    ``action_tokens_per_frame`` instance attribute during RoPE.

    When ``action_tokens_per_frame > 0`` the block sees a per-frame
    interleaved layout
        [spatial_0 … spatial_{H*W-1}, action_0 … action_{a-1}]
    (same layout the causal DiT's self-attn uses). RoPE must be applied to
    spatial tokens only; the action slots are held out, RoPE'd-not, and
    merged back so attention operates over the full sequence. Otherwise
    RoPE would index action slots as if they were spatial positions, which
    silently corrupts the attention pattern the causal weights were
    trained with.

    Teacher-forcing (``tf_rope_offset > 0``): the input is a joint
    [clean_half, noisy_half] sequence with ``grid_sizes[0,0] = 2*F`` (per-
    half F frames each). RoPE is applied to each half independently —
    clean at positions [0, F), noisy at [tf_rope_offset, tf_rope_offset+F)
    — so the v14 teacher-forcing layout (clean leading by ``cf`` frames)
    is preserved. Mirrors ``CausalWanSelfAttention.forward`` lines 179-193.

    When ``action_tokens_per_frame == 0`` and ``tf_rope_offset == 0`` the
    patch falls through to the original forward — bit-identical to
    upstream.
    """
    if getattr(attn, "_action_attn_patched", False):
        return
    if not hasattr(attn, "action_tokens_per_frame"):
        attn.action_tokens_per_frame = 0
    if not hasattr(attn, "tf_rope_offset"):
        attn.tf_rope_offset = 0
    orig_forward = attn.forward

    def forward_with_action(self_attn, x, seq_lens, grid_sizes, freqs):
        a_per_f = int(getattr(self_attn, "action_tokens_per_frame", 0))
        tf_off = int(getattr(self_attn, "tf_rope_offset", 0))
        if a_per_f == 0 and tf_off == 0:
            return orig_forward(x, seq_lens, grid_sizes, freqs)

        b, s = x.shape[0], x.shape[1]
        n, d = self_attn.num_heads, self_attn.head_dim

        q = self_attn.norm_q(self_attn.q(x)).view(b, s, n, d)
        k = self_attn.norm_k(self_attn.k(x)).view(b, s, n, d)
        v = self_attn.v(x).view(b, s, n, d)

        # Strip padding for the RoPE step: _separate_action_tokens assumes
        # seq = F * (H*W + a_per_f). `s` here may be ≥ that (padded to
        # ``seq_len`` in the outer _forward).
        f_total, h, w = grid_sizes[0].tolist()
        f_total = int(f_total)
        frame_seq = h * w + a_per_f
        valid_len = f_total * frame_seq
        if valid_len > s:
            raise RuntimeError(
                f"Bidirectional self_attn: interleaved valid_len={valid_len} "
                f"exceeds padded seq length {s}. The outer _forward pad "
                f"target is too small; bump seq_len or reduce F."
            )

        q_valid, q_tail = q[:, :valid_len], q[:, valid_len:]
        k_valid, k_tail = k[:, :valid_len], k[:, valid_len:]

        if tf_off > 0:
            # Teacher-forcing: joint sequence is [clean_half, noisy_half],
            # each F_half = f_total // 2 frames long. Clean gets RoPE
            # positions [0, F_half); noisy gets [tf_off, tf_off + F_half).
            if f_total % 2 != 0:
                raise RuntimeError(
                    f"Bidir self_attn TF mode requires even f_total; got {f_total}."
                )
            f_half = f_total // 2
            half_valid = f_half * frame_seq
            half_grid = grid_sizes.clone()
            half_grid[:, 0] = f_half
            q_clean, q_noisy = q_valid[:, :half_valid], q_valid[:, half_valid:]
            k_clean, k_noisy = k_valid[:, :half_valid], k_valid[:, half_valid:]
            if a_per_f > 0:
                qc_sp, qc_act = _separate_action_tokens(q_clean, half_grid, a_per_f)
                kc_sp, kc_act = _separate_action_tokens(k_clean, half_grid, a_per_f)
                qn_sp, qn_act = _separate_action_tokens(q_noisy, half_grid, a_per_f)
                kn_sp, kn_act = _separate_action_tokens(k_noisy, half_grid, a_per_f)
                rq_clean = _merge_action_tokens(
                    rope_apply(qc_sp, half_grid, freqs, temporal_offset=0),
                    qc_act, half_grid, a_per_f,
                )
                rk_clean = _merge_action_tokens(
                    rope_apply(kc_sp, half_grid, freqs, temporal_offset=0),
                    kc_act, half_grid, a_per_f,
                )
                rq_noisy = _merge_action_tokens(
                    rope_apply(qn_sp, half_grid, freqs, temporal_offset=tf_off),
                    qn_act, half_grid, a_per_f,
                )
                rk_noisy = _merge_action_tokens(
                    rope_apply(kn_sp, half_grid, freqs, temporal_offset=tf_off),
                    kn_act, half_grid, a_per_f,
                )
            else:
                rq_clean = rope_apply(q_clean, half_grid, freqs, temporal_offset=0)
                rk_clean = rope_apply(k_clean, half_grid, freqs, temporal_offset=0)
                rq_noisy = rope_apply(q_noisy, half_grid, freqs, temporal_offset=tf_off)
                rk_noisy = rope_apply(k_noisy, half_grid, freqs, temporal_offset=tf_off)
            rq_valid = torch.cat([rq_clean, rq_noisy], dim=1)
            rk_valid = torch.cat([rk_clean, rk_noisy], dim=1)
        else:
            q_sp, q_act = _separate_action_tokens(q_valid, grid_sizes, a_per_f)
            k_sp, k_act = _separate_action_tokens(k_valid, grid_sizes, a_per_f)

            rq_sp = rope_apply(q_sp, grid_sizes, freqs)
            rk_sp = rope_apply(k_sp, grid_sizes, freqs)

            rq_valid = _merge_action_tokens(rq_sp, q_act, grid_sizes, a_per_f)
            rk_valid = _merge_action_tokens(rk_sp, k_act, grid_sizes, a_per_f)

        if q_tail.shape[1] > 0:
            rq = torch.cat([rq_valid, q_tail], dim=1)
            rk = torch.cat([rk_valid, k_tail], dim=1)
        else:
            rq, rk = rq_valid, rk_valid

        # FUNDAMENTAL: bidirectional DMD scorers always use full-bidir
        # ``flash_attention`` — never restrict their attention. The v14
        # TF causal block_mask path was deliberately removed (it was
        # over-constraining clean→noisy info flow vs the actual v14
        # training contract).
        x_out = flash_attention(
            q=rq.type_as(v),
            k=rk.type_as(v),
            v=v,
            k_lens=seq_lens,
            window_size=self_attn.window_size,
        )
        x_out = x_out.flatten(2)
        x_out = self_attn.o(x_out)
        return x_out

    attn.forward = types.MethodType(forward_with_action, attn)
    attn._action_attn_patched = True


def _bidir_forward_with_action_tokens(
    self,
    x,
    t,
    context,
    seq_len,
    action_tokens,
    clip_fea=None,
    y=None,
    clean_x=None,
    aug_t=None,
    action_tokens_clean=None,
    state_tokens=None,
    state_tokens_clean=None,
):
    """Replacement ``_forward`` body for the bidirectional WanModel that
    interleaves Stream-B action tokens per frame before the transformer
    blocks and strips them before ``head + unpatchify``.

    Mirrors ``CausalWanModel._forward_inference`` lines 1116-1256 but
    without KV-cache / probe tap machinery; the bidirectional scorers
    only need the forward pass to (a) consume the action-conditioned
    per-frame layout the causal weights were trained on and (b) return
    spatial-only x0 predictions.

    Teacher-forcing (``clean_x is not None``): the clean-half latents
    are patch-embedded, interleaved with their own action tokens
    (``action_tokens_clean``; required when a_per_f > 0), and concatenated
    along the seq dim BEFORE the noisy half. Time embeddings are computed
    separately for the clean half (using ``aug_t``, defaulting to zeros)
    and the noisy half (using ``t``), then concatenated along the F dim
    so each block sees per-frame AdaLN-Zero modulation appropriate to
    each half. After all blocks, the clean half is sliced off; head +
    unpatchify run on the noisy half only. The bidirectional WanModel
    has no causal mask, so the joint-window self-attn naturally lets
    the noisy half attend to the clean half (and vice versa) — no
    special block_mask machinery needed (cf. CausalWanModel which DOES
    need the TF block_mask + rope_offset).

    classify_mode / regress_mode are intentionally unsupported here —
    those code paths power the GAN critic heads and Phase-1 rolling-
    staircase DMD has the GAN disabled. If re-enabled they need their
    own action-token handling.
    """
    if self.model_type == "i2v":
        assert clip_fea is not None and y is not None

    a_per_f = int(getattr(self, "action_tokens_per_frame", 0))
    s_per_f = int(getattr(self, "state_tokens_per_frame", 0))
    # ``a_per_f`` is the COMBINED count of all per-frame extras (action +
    # state) — see ``adding_state_token_branch`` which bumps a_per_f by 1
    # when adding state tokens. ``s_per_f`` is the state-only slice;
    # ``a_per_f - s_per_f`` is the action-only slice.
    n_action_only = a_per_f - s_per_f
    if a_per_f <= 0:
        raise RuntimeError(
            "_bidir_forward_with_action_tokens called with "
            "action_tokens_per_frame <= 0; the dispatch in "
            "_forward_with_action should have fallen through to orig_fwd."
        )
    if action_tokens is None:
        raise RuntimeError(
            "_bidir_forward_with_action_tokens called with "
            "action_tokens=None; Stream B is advertised by "
            "action_tokens_per_frame > 0 but no tokens were provided."
        )
    if s_per_f > 0 and state_tokens is None:
        raise RuntimeError(
            "_bidir_forward_with_action_tokens called with "
            f"state_tokens_per_frame={s_per_f} but state_tokens=None; "
            "the v14 LoRA was trained with per-frame state tokens "
            "interleaved in the input — running without them is OOD."
        )
    if clean_x is not None and action_tokens_clean is None:
        raise RuntimeError(
            "_bidir_forward_with_action_tokens called with clean_x "
            "but action_tokens_clean=None; the clean half needs its "
            "own per-frame action tokens (Stream B is active)."
        )
    if clean_x is not None and s_per_f > 0 and state_tokens_clean is None:
        raise RuntimeError(
            "_bidir_forward_with_action_tokens called with clean_x "
            "but state_tokens_clean=None; the clean half needs its "
            "own per-frame state tokens when state-token branch is active."
        )

    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # Patch embedding for the noisy half.
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
    )
    x = [u.flatten(2).transpose(1, 2) for u in x]  # list of [1, F*H*W, C]

    # Interleave action tokens per frame.
    spatial_seqlen = int(math.prod(grid_sizes[0][1:]).item())
    num_frames_local = int(grid_sizes[0, 0].item())
    frame_seqlen_local = spatial_seqlen + a_per_f
    expected_at_frames = action_tokens.shape[1]
    if expected_at_frames != num_frames_local:
        raise RuntimeError(
            f"action_tokens has {expected_at_frames} frames but spatial "
            f"patch grid has {num_frames_local}; they must match."
        )
    x_interleaved = []
    for batch_idx, u in enumerate(x):
        u = u[:, : num_frames_local * spatial_seqlen].unflatten(
            1, (num_frames_local, spatial_seqlen)
        )
        extras = [action_tokens[batch_idx : batch_idx + 1].unsqueeze(2).to(
            dtype=u.dtype, device=u.device,
        )]
        if state_tokens is not None and s_per_f > 0:
            st_block = state_tokens[batch_idx : batch_idx + 1].unsqueeze(2).to(
                dtype=u.dtype, device=u.device,
            )
            extras.append(st_block)
        u = torch.cat([u] + extras, dim=2).flatten(1, 2)
        x_interleaved.append(u)
    x = x_interleaved  # list of [1, F*(H*W+a_per_f), C]

    # Teacher-forcing: patch-embed + interleave the clean half, then
    # concat [clean, noisy] along seq dim. The clean half uses its own
    # action tokens (action_tokens_clean) and shares the same grid_sizes
    # (same H, W per frame; same num_frames in our usage — clean and
    # noisy halves are both num_training_frames long, just at different
    # absolute time positions in the ride).
    if clean_x is not None:
        clean_x_emb = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
        clean_grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in clean_x_emb]
        )
        if int(clean_grid_sizes[0, 0].item()) != num_frames_local:
            raise RuntimeError(
                f"clean_x has {int(clean_grid_sizes[0, 0].item())} frames "
                f"but noisy x has {num_frames_local}; both halves must have "
                "the same number of frames in this bidirectional TF path."
            )
        clean_x_emb = [u.flatten(2).transpose(1, 2) for u in clean_x_emb]
        clean_at_frames = action_tokens_clean.shape[1]
        if clean_at_frames != num_frames_local:
            raise RuntimeError(
                f"action_tokens_clean has {clean_at_frames} frames but "
                f"clean_x has {num_frames_local} frames; they must match."
            )
        clean_x_interleaved = []
        for batch_idx, u in enumerate(clean_x_emb):
            u = u[:, : num_frames_local * spatial_seqlen].unflatten(
                1, (num_frames_local, spatial_seqlen)
            )
            extras_c = [action_tokens_clean[batch_idx : batch_idx + 1].unsqueeze(2).to(
                dtype=u.dtype, device=u.device,
            )]
            if state_tokens_clean is not None and s_per_f > 0:
                stc_block = state_tokens_clean[batch_idx : batch_idx + 1].unsqueeze(2).to(
                    dtype=u.dtype, device=u.device,
                )
                extras_c.append(stc_block)
            u = torch.cat([u] + extras_c, dim=2).flatten(1, 2)
            clean_x_interleaved.append(u)
        # Concat [clean, noisy] per batch.
        x = [torch.cat([cu, nu], dim=1) for cu, nu in zip(clean_x_interleaved, x)]

    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    # Extend pad target to fit the interleaved sequence. In TF mode
    # (clean_x present) we use EXACTLY the natural joint length
    # ``2 * F * frame_seqlen_local`` — no outer padding past the
    # interleaved tokens. This is essential for the v14 TF block_mask:
    # the mask is built for Q_LEN = ceil(2*F*frame_seqlen/128)*128, so
    # if we add more outer padding here the per-block flex_attention
    # call will pad to a different multiple-of-128 and the mask shape
    # will mismatch the q/k length.
    # In non-TF mode we keep the historical padding (caller's seq_len
    # tracks the spatial-only capacity, and we still add the per-frame
    # action-token slots on top).
    natural_joint = num_frames_local * frame_seqlen_local * (2 if clean_x is not None else 1)
    if clean_x is not None:
        pad_target = natural_joint
    else:
        half_pad = seq_len + num_frames_local * a_per_f
        pad_target = half_pad
    if int(seq_lens.max().item()) > pad_target:
        raise RuntimeError(
            f"interleaved seq_lens.max()={int(seq_lens.max().item())} "
            f"exceeds pad_target={pad_target} (TF={clean_x is not None}, "
            f"natural_joint={natural_joint}, seq_len={seq_len}, "
            f"F*a_per_f={num_frames_local * a_per_f})"
        )
    x = torch.cat(
        [
            torch.cat(
                [u, u.new_zeros(1, pad_target - u.size(1), u.size(2))],
                dim=1,
            )
            for u in x
        ]
    )

    # Time embedding (identical to upstream _forward) — for the NOISY
    # half. ``time_projection`` is patched (``_patch_time_projection``)
    # to add ``self._action_modulation`` (Stream A) to its output.
    e = self.time_embedding(
        sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x)
    )
    e0 = (
        self.time_projection(e)
        .unflatten(1, (6, self.dim))
        .unflatten(dim=0, sizes=t.shape)
    )

    # Teacher-forcing time embedding for the clean half. We swap
    # ``self._action_modulation`` to ``self._action_modulation_clean``
    # for the duration of the clean-half time_projection so the clean
    # tokens get clean Stream A modulation. Restored immediately so the
    # subsequent block forwards see the regular noisy modulation.
    # Mirrors CausalWanModel._forward_inference lines 1505-1513.
    if clean_x is not None:
        if aug_t is None:
            aug_t = torch.zeros_like(t)
        saved_am = getattr(self, "_action_modulation", None)
        self._action_modulation = getattr(self, "_action_modulation_clean", None)
        try:
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x)
            )
            e0_clean = (
                self.time_projection(e_clean)
                .unflatten(1, (6, self.dim))
                .unflatten(dim=0, sizes=aug_t.shape)
            )
        finally:
            self._action_modulation = saved_am
        e0 = torch.cat([e0_clean, e0], dim=1)

    # Context (identical to upstream _forward).
    context_lens = None
    context = self.text_embedding(
        torch.stack(
            [
                torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]
        )
    )
    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)
        context = torch.concat([context_clip, context], dim=1)

    # Propagate a_per_f and tf_rope_offset to each block's self_attn.
    # In TF mode tf_rope_offset_frames = num_frame_per_block (= 3 latent
    # frames = 1 chunk shift), so the noisy half is RoPE-positioned at
    # [npb, npb + F) while the clean half stays at [0, F). Total RoPE
    # span = F + npb = 8 chunks. Each chunk index occupies the same
    # RoPE position whether seen as clean or noisy (in the overlap
    # region). This is v14 LoRA's training contract — do not change
    # without retraining.
    #
    # FUNDAMENTAL: bidirectional DMD scorers ALWAYS use full-bidir
    # ``flash_attention``. We do NOT build or apply a causal block_mask
    # here. The causal-Wan student's TF training and the v14 LoRA's own
    # training continue to use the upstream block_mask path
    # (``CausalWanModel._prepare_teacher_forcing_mask``) — that's
    # untouched. Only the DMD scorers' joint forward is unmasked.
    if clean_x is not None:
        # ``None`` is the "not set" sentinel; an explicit integer
        # (including 0) is honoured. Action-aware scorers MUST have
        # this set — raise loudly rather than silently falling back to
        # a derivation that would mis-position v14's noisy half.
        explicit = getattr(self, "tf_rope_offset_frames", None)
        if explicit is None:
            raise RuntimeError(
                "_bidir_forward_with_action_tokens called with clean_x "
                "but model.tf_rope_offset_frames is unset (None); the "
                "caller must set it (= num_frame_per_block, = 1-chunk "
                "shift for v14) before calling with clean_x so the "
                "noisy half gets the correct shifted RoPE positions."
            )
        tf_rope_offset = int(explicit)
    else:
        tf_rope_offset = 0
    for block in self.blocks:
        block.self_attn.action_tokens_per_frame = a_per_f
        block.self_attn.tf_rope_offset = tf_rope_offset

    # The self-attn patch reads grid_sizes to compute valid_len for the
    # interleaved layout; in TF mode the joint sequence has 2*F frames.
    # We pass a doubled-F grid_sizes so the patch sees the full joint
    # length (otherwise it'd treat the clean half as padding and skip
    # RoPE on the noisy half's first F frames).
    block_grid_sizes = grid_sizes
    if clean_x is not None:
        block_grid_sizes = grid_sizes.clone()
        block_grid_sizes[:, 0] = num_frames_local * 2

    block_kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=block_grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens,
    )

    def create_custom_forward(module):
        def custom_forward(*inputs, **kw):
            return module(*inputs, **kw)

        return custom_forward

    for block in self.blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                x,
                **block_kwargs,
                use_reentrant=False,
            )
        else:
            x = block(x, **block_kwargs)

    # Teacher-forcing: keep only the noisy half (the second half of the
    # joint sequence). The clean half is the first F*frame_seqlen tokens.
    frame_seqlen = spatial_seqlen + a_per_f
    if clean_x is not None:
        clean_len = num_frames_local * frame_seqlen
        x = x[:, clean_len:]

    # Strip action tokens before head + unpatchify (noisy half only now).
    valid_len = num_frames_local * frame_seqlen
    x = x[:, :valid_len].unflatten(1, (num_frames_local, frame_seqlen))
    x = x[:, :, :spatial_seqlen].flatten(1, 2)

    # Head with per-frame modulation (mirrors CausalHead.forward).
    # Upstream's bidir ``Head.forward`` assumes ``e`` is ``[B, C]`` (scalar
    # t per sample). When we pass per-frame ``t`` (e.g. DMD's
    # ``[ctx_t=0, ..., target_t]``), ``e`` comes out as ``[B*F, C]`` and
    # the default Head broadcasts a full-batch vector against a
    # per-sample x — mis-shaped. Reproduce the causal head math here.
    # The head uses the NOISY half's e (the original ``t``-based one),
    # not the joint e0 — same convention as CausalWanModel (line 1560).
    head_mod = getattr(self.head, "modulation")
    head_norm = getattr(self.head, "norm")
    head_lin = getattr(self.head, "head")
    e_per_frame = e.unflatten(dim=0, sizes=t.shape)  # [B, F, C]
    mod = head_mod.unsqueeze(1) + e_per_frame.unsqueeze(2)  # [B, F, 2, C]
    mod_shift, mod_scale = mod.chunk(2, dim=2)  # each [B, F, 1, C]
    x_framed = head_norm(x).unflatten(
        dim=1, sizes=(num_frames_local, spatial_seqlen)
    )  # [B, F, L1, C]
    x_framed = x_framed * (1 + mod_scale) + mod_shift
    x = head_lin(x_framed.flatten(1, 2))
    x = self.unpatchify(x, grid_sizes)
    return torch.stack(x)


def patch_bidirectional_wan_model_for_action(model):
    """Patch Wan's bidirectional model to support external action modulation
    (Stream A, AdaLN) and per-frame action tokens (Stream B)."""
    if getattr(model, "_action_bidir_patched", False):
        return model

    _patch_time_projection(model)
    if not hasattr(model, "_forward"):
        raise AttributeError("Bidirectional Wan model is expected to define `_forward`.")

    # Stream B attributes: default to 0 (no action tokens) so an un-wired
    # forward is bit-identical to upstream.
    if not hasattr(model, "action_tokens_per_frame"):
        model.action_tokens_per_frame = 0
    for block in model.blocks:
        _patch_bidirectional_self_attn_for_action(block.self_attn)

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
        action_modulation_clean=None,
        action_tokens=None,
        action_tokens_clean=None,
        clean_x=None,
        aug_t=None,
        state_tokens=None,
        state_tokens_clean=None,
        **kwargs,
    ):
        self._action_modulation = action_modulation
        self._action_modulation_clean = action_modulation_clean

        # Stream B dispatch: when the model is configured for action tokens
        # AND a tensor was provided, route through the replacement forward
        # that knows to interleave/strip. Otherwise fall through to orig.
        #
        # Fail loud on the misconfiguration "a_per_f > 0 but no tokens"
        # because running orig_fwd in that state would silently feed the
        # DiT a seq-len-1560-per-frame input while its weights expect
        # 1561 tokens/frame — precisely the out-of-distribution case we
        # explicitly refused for the generator.
        a_per_f = int(getattr(self, "action_tokens_per_frame", 0))
        if a_per_f > 0 and action_tokens is None:
            raise RuntimeError(
                "Bidirectional WanModel has action_tokens_per_frame="
                f"{a_per_f} but no action_tokens were provided. The DiT "
                "weights were trained with Stream B active; running with "
                "Stream A only is out-of-distribution."
            )

        try:
            if a_per_f > 0 and action_tokens is not None:
                # classify_mode / regress_mode don't support Stream B yet.
                if kwargs.get("classify_mode", False) or kwargs.get(
                    "regress_mode", False
                ):
                    raise RuntimeError(
                        "classify_mode / regress_mode are not supported "
                        "alongside Stream B. Disable GAN or drop Stream B "
                        "for the critic head forward."
                    )
                # Route the small subset of relevant kwargs through.
                return _bidir_forward_with_action_tokens(
                    self,
                    x,
                    t,
                    context,
                    seq_len,
                    action_tokens,
                    clip_fea=kwargs.get("clip_fea"),
                    y=kwargs.get("y"),
                    clean_x=clean_x,
                    aug_t=aug_t,
                    action_tokens_clean=action_tokens_clean,
                    state_tokens=state_tokens,
                    state_tokens_clean=state_tokens_clean,
                )
            return orig_fwd(x, t, context, seq_len, *args, **kwargs)
        finally:
            self._action_modulation = None
            self._action_modulation_clean = None

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
