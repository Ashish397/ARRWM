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
            # Teacher-forcing: joint sequence is [clean_half, noisy_half].
            # Symmetric (v14): each half = f_total // 2. Asymmetric
            # (dmd_one_step): clean = tf_num_clean_frames, noisy =
            # tf_num_noisy_frames (read from the attn, set by the outer
            # bidir forward). Clean gets RoPE positions [0, F_clean);
            # noisy gets [tf_off, tf_off + F_noisy).
            _nc = getattr(self_attn, "tf_num_clean_frames", None)
            _nn = getattr(self_attn, "tf_num_noisy_frames", None)
            if _nc is not None and _nn is not None:
                f_clean = int(_nc)
                f_noisy = int(_nn)
                if f_clean + f_noisy != f_total:
                    raise RuntimeError(
                        f"Bidir self_attn asymmetric TF: f_clean({f_clean}) + "
                        f"f_noisy({f_noisy}) != f_total({f_total})."
                    )
            else:
                if f_total % 2 != 0:
                    raise RuntimeError(
                        f"Bidir self_attn TF mode requires even f_total; got {f_total}."
                    )
                f_clean = f_noisy = f_total // 2
            clean_valid = f_clean * frame_seq
            clean_grid = grid_sizes.clone()
            clean_grid[:, 0] = f_clean
            noisy_grid = grid_sizes.clone()
            noisy_grid[:, 0] = f_noisy
            import os as _os
            if _os.environ.get("ARRWM_ROPE_DEBUG"):
                try:
                    # Module-global counter: ``forward_with_action`` is
                    # re-wrapped per DMD inner-step, so a function attribute
                    # resets and the probe floods. A global persists.
                    global _ROPE_DBG_SCORE_COUNT
                    _c = globals().get("_ROPE_DBG_SCORE_COUNT", 0)
                    if _c < 4:
                        _ROPE_DBG_SCORE_COUNT = _c + 1
                        print(
                            f"[ROPE-DBG score] tf_off={tf_off} f_clean={f_clean} "
                            f"f_noisy={f_noisy} clean_rope=[0,{f_clean}) "
                            f"noisy_rope=[{tf_off},{tf_off + f_noisy}) "
                            f"grad={torch.is_grad_enabled()}",
                            flush=True,
                        )
                except Exception:
                    pass
            q_clean, q_noisy = q_valid[:, :clean_valid], q_valid[:, clean_valid:]
            k_clean, k_noisy = k_valid[:, :clean_valid], k_valid[:, clean_valid:]
            if a_per_f > 0:
                qc_sp, qc_act = _separate_action_tokens(q_clean, clean_grid, a_per_f)
                kc_sp, kc_act = _separate_action_tokens(k_clean, clean_grid, a_per_f)
                qn_sp, qn_act = _separate_action_tokens(q_noisy, noisy_grid, a_per_f)
                kn_sp, kn_act = _separate_action_tokens(k_noisy, noisy_grid, a_per_f)
                rq_clean = _merge_action_tokens(
                    rope_apply(qc_sp, clean_grid, freqs, temporal_offset=0),
                    qc_act, clean_grid, a_per_f,
                )
                rk_clean = _merge_action_tokens(
                    rope_apply(kc_sp, clean_grid, freqs, temporal_offset=0),
                    kc_act, clean_grid, a_per_f,
                )
                rq_noisy = _merge_action_tokens(
                    rope_apply(qn_sp, noisy_grid, freqs, temporal_offset=tf_off),
                    qn_act, noisy_grid, a_per_f,
                )
                rk_noisy = _merge_action_tokens(
                    rope_apply(kn_sp, noisy_grid, freqs, temporal_offset=tf_off),
                    kn_act, noisy_grid, a_per_f,
                )
            else:
                rq_clean = rope_apply(q_clean, clean_grid, freqs, temporal_offset=0)
                rk_clean = rope_apply(k_clean, clean_grid, freqs, temporal_offset=0)
                rq_noisy = rope_apply(q_noisy, noisy_grid, freqs, temporal_offset=tf_off)
                rk_noisy = rope_apply(k_noisy, noisy_grid, freqs, temporal_offset=tf_off)
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
    # Clean/noisy frame counts. v14's symmetric contract has both halves
    # = num_frames_local. The dmd_one_step asymmetric path scores a single
    # student chunk (noisy = num_frames_local, typically 3) against a full
    # GT clean window (clean = num_clean_frames, typically 21). They may
    # differ; ``num_clean_frames`` is derived from clean_x's own grid.
    num_clean_frames = num_frames_local
    if clean_x is not None:
        clean_x_emb = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
        clean_grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in clean_x_emb]
        )
        num_clean_frames = int(clean_grid_sizes[0, 0].item())
        clean_x_emb = [u.flatten(2).transpose(1, 2) for u in clean_x_emb]
        clean_at_frames = action_tokens_clean.shape[1]
        if clean_at_frames != num_clean_frames:
            raise RuntimeError(
                f"action_tokens_clean has {clean_at_frames} frames but "
                f"clean_x has {num_clean_frames} frames; they must match."
            )
        clean_x_interleaved = []
        for batch_idx, u in enumerate(clean_x_emb):
            u = u[:, : num_clean_frames * spatial_seqlen].unflatten(
                1, (num_clean_frames, spatial_seqlen)
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
    # Joint length = (clean + noisy) frames; in the symmetric case this is
    # 2*F, in the asymmetric case (num_clean_frames + num_frames_local).
    if clean_x is not None:
        natural_joint = (num_clean_frames + num_frames_local) * frame_seqlen_local
        pad_target = natural_joint
    else:
        natural_joint = num_frames_local * frame_seqlen_local
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
        # Per-block clean/noisy frame counts so the bidir self-attn can
        # split the joint sequence at the CLEAN boundary (not the
        # midpoint) and RoPE each half with its own frame count. Set to
        # None outside TF mode so the attn falls back to its plain path.
        if clean_x is not None:
            block.self_attn.tf_num_clean_frames = num_clean_frames
            block.self_attn.tf_num_noisy_frames = num_frames_local
        else:
            block.self_attn.tf_num_clean_frames = None
            block.self_attn.tf_num_noisy_frames = None

    # The self-attn patch reads grid_sizes to compute valid_len for the
    # interleaved layout; in TF mode the joint sequence has
    # (clean + noisy) frames. We pass a joint-F grid_sizes so the patch
    # sees the full joint length (otherwise it'd treat the clean half as
    # padding and skip RoPE on the noisy half's first frames).
    block_grid_sizes = grid_sizes
    if clean_x is not None:
        block_grid_sizes = grid_sizes.clone()
        block_grid_sizes[:, 0] = num_clean_frames + num_frames_local

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


# =====================================================================
# TF-only patch (v21+ — foreign-teacher DMD support)
# =====================================================================
# When ``real_model_name != fake_model_name`` (e.g. real_score is
# Wan2.1-T2V-14B while generator/fake_score are 1.3B), the action
# patches above cannot be applied to real_score: ``action_projection``
# is sized for the GENERATOR's hidden dim (1536 for 1.3B), and the
# concat inside ``_bidir_forward_with_action_tokens`` would crash on
# the 14B's 5120-dim sequence. The TF-only patch below gives the
# foreign teacher proper teacher-forcing context support WITHOUT
# action tokens:
#
#   - Linear concat layout: clean_x at RoPE [0..F), noisy at [F..2F).
#     This is in-distribution for a stock T2V model — it's just a
#     2F-frame video where the first F are clean past context and the
#     last F are the noisy chunk being denoised. The 14B's RoPE table
#     extends to 10000 positions; 42 frames is well within its trained
#     range (Wan2.1-T2V models trained on up to 81-frame clips).
#
#   - No action tokens, no Stream A (action_modulation), no Stream B
#     (action_tokens) interleaving. The model sees only spatial tokens
#     at standard frame_seqlen = H*W per frame.
#
#   - clean_x and noisy_x must have the same F per the existing
#     bidirectional self-attn patch's symmetric-half assumption
#     (``f_half = f_total // 2``).
#
# The patch is applied by ``apply_tf_only_patches_critic(wrapper)``
# which:
#   - Patches each block's self_attn for tf_rope_offset support (reuses
#     the existing ``_patch_bidirectional_self_attn_for_action`` which
#     already handles the ``a_per_f == 0, tf_off > 0`` case).
#   - Replaces ``model._forward`` with ``_tf_only_forward_for_bidir_wan``
#     which performs the linear concat TF.
#   - Sets ``wrapper._action_patch_applied = False`` so the wrapper's
#     forward does NOT inject action kwargs from conditional_dict
#     (cosmetic — model wouldn't accept them anyway, but the wrapper
#     would set them).
# =====================================================================
def _tf_only_forward_for_bidir_wan(
    self,
    x,
    t,
    context,
    seq_len,
    clean_x=None,
    aug_t=None,
    clip_fea=None,
    y=None,
    **kwargs,
):
    """Replacement ``_forward`` for a bidirectional WanModel that adds
    teacher-forcing (clean_x concat) support WITHOUT action tokens.

    ``clean_x is None`` path: passes through to the captured original
    forward (= stock T2V) — bit-identical to the un-patched model.

    ``clean_x is not None`` path: patch-embeds the clean half, concats
    [clean, noisy] along the seq dim, runs all transformer blocks on
    the joint 2F-frame input, strips the clean half before head +
    unpatchify, returns the noisy half's x0 prediction.

    RoPE positions: clean at [0..F), noisy at [F..2F). The per-block
    self-attn patch reads ``self_attn.tf_rope_offset`` and applies
    the shifted RoPE to the noisy half automatically.

    Args:
      clean_x: list of [C, F, H, W] tensors (same shape contract as
        ``x``) — the past-context latents (e.g. 21 GT seed frames).
      aug_t: [B, F] timestep tensor for the clean half. Defaults to
        zeros (= clean / no noise on the clean half).
    """
    # No-TF path: defer to the captured original forward. The TF-only
    # patch is benign when nobody passes clean_x.
    if clean_x is None:
        orig = getattr(self, "_orig_forward_tf_only", None)
        if orig is None:
            raise RuntimeError(
                "_tf_only_forward_for_bidir_wan called without an "
                "original forward stashed. apply_tf_only_patches_critic "
                "must run before the model is used."
            )
        return orig(x, t, context, seq_len, clip_fea=clip_fea, y=y, **kwargs)

    if self.model_type == "i2v":
        # Foreign I2V teachers aren't supported here — caller would need
        # to thread CLIP image features + first-frame y. Fail loud.
        raise NotImplementedError(
            "_tf_only_forward_for_bidir_wan: i2v model_type is not "
            "supported. Use a T2V foreign teacher (e.g. Wan2.1-T2V-14B)."
        )

    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # Patch embedding for the noisy half.
    x_emb = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x_emb]
    )
    x_flat = [u.flatten(2).transpose(1, 2) for u in x_emb]
    spatial_seqlen = int(math.prod(grid_sizes[0][1:]).item())
    num_frames_local = int(grid_sizes[0, 0].item())

    # Patch embedding for the clean half. Must have identical (H, W)
    # and the same F (the bidirectional self-attn patch assumes
    # symmetric halves: f_half = f_total // 2).
    clean_emb = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
    clean_grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in clean_emb]
    )
    if int(clean_grid_sizes[0, 0].item()) != num_frames_local:
        raise RuntimeError(
            "_tf_only_forward_for_bidir_wan: clean_x has "
            f"{int(clean_grid_sizes[0, 0].item())} frames but noisy x has "
            f"{num_frames_local}; the symmetric-halves self-attn patch "
            "requires them to match."
        )
    clean_flat = [u.flatten(2).transpose(1, 2) for u in clean_emb]

    # Concat [clean, noisy] along seq dim per batch.
    joint = [
        torch.cat([cu, nu], dim=1) for cu, nu in zip(clean_flat, x_flat)
    ]

    seq_lens = torch.tensor([u.size(1) for u in joint], dtype=torch.long)
    # Pad to the exact joint length. No outer padding past the natural
    # 2*F*spatial_seqlen — keeps the self-attn's grid_sizes-derived
    # valid_len consistent.
    natural_joint = num_frames_local * spatial_seqlen * 2
    if int(seq_lens.max().item()) > natural_joint:
        raise RuntimeError(
            f"_tf_only_forward_for_bidir_wan: seq_lens.max()="
            f"{int(seq_lens.max().item())} exceeds natural_joint="
            f"{natural_joint}; check clean_x / noisy x shapes."
        )
    x_joint = torch.cat(
        [
            torch.cat(
                [u, u.new_zeros(1, natural_joint - u.size(1), u.size(2))],
                dim=1,
            )
            for u in joint
        ]
    )

    # Time embedding for the NOISY half (uses t).
    e_noisy = self.time_embedding(
        sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x_joint)
    )
    e0_noisy = (
        self.time_projection(e_noisy)
        .unflatten(1, (6, self.dim))
        .unflatten(dim=0, sizes=t.shape)
    )
    # Time embedding for the CLEAN half (uses aug_t; defaults to zeros).
    if aug_t is None:
        aug_t = torch.zeros_like(t)
    e_clean = self.time_embedding(
        sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x_joint)
    )
    e0_clean = (
        self.time_projection(e_clean)
        .unflatten(1, (6, self.dim))
        .unflatten(dim=0, sizes=aug_t.shape)
    )
    # Concat along F (= joint sequence's per-frame modulation).
    e0 = torch.cat([e0_clean, e0_noisy], dim=1)

    # Text context.
    context_lens = None
    context_full = self.text_embedding(
        torch.stack(
            [
                torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]
        )
    )
    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)
        context_full = torch.concat([context_clip, context_full], dim=1)

    # Set per-block self_attn TF state: tf_rope_offset = F (noisy half
    # gets RoPE positions [F..2F)). action_tokens_per_frame stays 0.
    for block in self.blocks:
        block.self_attn.action_tokens_per_frame = 0
        block.self_attn.tf_rope_offset = num_frames_local

    # The self-attn patch derives valid_len from grid_sizes[0,0]. In
    # this TF-only joint forward F_joint = 2 * F, so we pass a doubled
    # grid_sizes to keep valid_len = 2*F*spatial_seqlen.
    block_grid_sizes = grid_sizes.clone()
    block_grid_sizes[:, 0] = num_frames_local * 2

    block_kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=block_grid_sizes,
        freqs=self.freqs,
        context=context_full,
        context_lens=context_lens,
    )

    def _create_custom_forward(module):
        def _fwd(*inputs, **kw):
            return module(*inputs, **kw)
        return _fwd

    x_joint_ = x_joint
    for block in self.blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            x_joint_ = torch.utils.checkpoint.checkpoint(
                _create_custom_forward(block),
                x_joint_,
                **block_kwargs,
                use_reentrant=False,
            )
        else:
            x_joint_ = block(x_joint_, **block_kwargs)

    # Strip the clean half: keep only the noisy half's seq.
    clean_len = num_frames_local * spatial_seqlen
    x_noisy = x_joint_[:, clean_len:]
    valid_len = num_frames_local * spatial_seqlen
    x_noisy = x_noisy[:, :valid_len]

    # Head with per-frame modulation (mirrors CausalHead.forward).
    # Use the NOISY half's e (t-based), not the joint e0.
    head_mod = getattr(self.head, "modulation")
    head_norm = getattr(self.head, "norm")
    head_lin = getattr(self.head, "head")
    e_per_frame = e_noisy.unflatten(dim=0, sizes=t.shape)  # [B, F, C]
    mod = head_mod.unsqueeze(1) + e_per_frame.unsqueeze(2)  # [B, F, 2, C]
    mod_shift, mod_scale = mod.chunk(2, dim=2)  # each [B, F, 1, C]
    x_framed = head_norm(x_noisy).unflatten(
        dim=1, sizes=(num_frames_local, spatial_seqlen)
    )  # [B, F, L1, C]
    x_framed = x_framed * (1 + mod_scale) + mod_shift
    x_out = head_lin(x_framed.flatten(1, 2))
    x_out = self.unpatchify(x_out, grid_sizes)
    return torch.stack(x_out)


def patch_bidirectional_wan_model_for_tf_only(model):
    """TF-only sibling of ``patch_bidirectional_wan_model_for_action``.

    Patches a bidirectional WanModel to add ``clean_x`` (teacher-forcing
    context) support WITHOUT action tokens. Idempotent.

    After patching, the model's ``_forward`` accepts ``clean_x`` and
    ``aug_t`` kwargs. When ``clean_x is not None``, the new forward does
    linear-concat TF: [clean, noisy] along the seq dim, clean at RoPE
    [0..F), noisy at [F..2F). When ``clean_x is None``, falls through to
    the original forward bit-identically.

    Each block's self_attn is patched (reuses
    ``_patch_bidirectional_self_attn_for_action`` which already handles
    the ``a_per_f == 0, tf_off > 0`` case for RoPE).
    """
    if getattr(model, "_tf_only_patched", False):
        return model
    if getattr(model, "_action_bidir_patched", False):
        # Action patch already installed; the TF-only path would
        # conflict. Refuse rather than silently overwrite.
        raise RuntimeError(
            "patch_bidirectional_wan_model_for_tf_only: model already "
            "has the action patch installed. The two patches are "
            "mutually exclusive. Don't call apply_action_patches_critic "
            "on a foreign-teacher real_score."
        )
    if not hasattr(model, "_forward"):
        raise AttributeError(
            "Bidirectional Wan model is expected to define `_forward`."
        )

    # Self-attn TF support. action_tokens_per_frame defaults to 0; the
    # patched forward reads tf_rope_offset which the new _forward sets
    # per-call to num_frames_local.
    if not hasattr(model, "action_tokens_per_frame"):
        model.action_tokens_per_frame = 0
    for block in model.blocks:
        _patch_bidirectional_self_attn_for_action(block.self_attn)

    # Capture the original forward and install ours.
    model._orig_forward_tf_only = model._forward
    model._forward = _tf_only_forward_for_bidir_wan.__get__(model, type(model))
    model._tf_only_patched = True
    return model


def apply_tf_only_patches_critic(generator_wrapper):
    """Apply the TF-only patch to a critic (bidirectional) wrapper.

    Use this INSTEAD of ``apply_action_patches_critic`` when the wrapper
    is a foreign-size teacher (e.g. real_score = Wan2.1-T2V-14B while
    generator/fake_score are 1.3B). The TF-only patch gives the wrapper
    teacher-forcing support WITHOUT action tokens (which would have the
    wrong hidden dim for a foreign-size model).

    Sets ``wrapper._action_patch_applied = False`` so the wrapper's
    forward doesn't try to inject action_modulation / action_tokens
    kwargs from conditional_dict (which the patched model._forward
    wouldn't accept anyway).

    Also sets ``wrapper._tf_only_patch_applied = True`` as a marker for
    downstream code (e.g. dispatch logic in compute_kl_grad).
    """
    target = generator_wrapper
    if FSDP is not None and isinstance(target, FSDP):
        target = target.module  # type: ignore[attr-defined]
    if getattr(target, "_tf_only_patch_applied", False):
        return generator_wrapper
    if not hasattr(target, "model"):
        raise AttributeError(
            "apply_tf_only_patches_critic: wrapper has no .model attribute."
        )
    patch_bidirectional_wan_model_for_tf_only(target.model)
    target._action_patch_applied = False  # type: ignore[attr-defined]
    target._tf_only_patch_applied = True  # type: ignore[attr-defined]
    return generator_wrapper
