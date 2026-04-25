"""Block-Relativistic / Infinity-RoPE patch for the ODE student.

The default KV-cache path in
``wan/modules/causal_model.py::CausalWanSelfAttention.forward`` rotates
the new K with absolute global positions (``causal_rope_apply`` with
``start_frame=current_start_frame``) and stores the *already-rotated* K
in the cache. As the rollout advances, two problems compound:

1. **Stale rotations.** Once the cache rolls (sliding window), older
   slots are re-used at new absolute positions but their K still carry
   the rotation from the position they were originally written at, so
   the relative offsets between Q at the new position and the cached K
   no longer make sense. This shows up as a visible *stutter / stagger*
   at chunk boundaries during long autoregressive rollouts.
2. **Unbounded rotation magnitude.** Q is rotated at the absolute
   ``current_start`` frame, which grows without bound, leaving the
   training distribution after a few hundred frames.

The `LongLive
<https://github.com/NVlabs/LongLive>`_ ``causal_model_infinity.py``
fixes this with a few changes that this module ports onto our model:

* **Window-relative indexing**, bounded once the cache fills:
  cache K is rotated at ``[0 .. num_cache_frames-1]``, Q is rotated at
  ``[local_attn_size - num_new_frames, local_attn_size]`` once the cache
  is full (so Q-rotation magnitude is bounded — the *infinity* property
  for arbitrarily long rollouts), and at
  ``[local_start_index/frame_seqlen, ...]`` before the cache fills
  (matches the original direct-insert semantics).
* **Un-roped K stored in the cache**, with RoPE applied at attention
  time. This makes every denoise pass / cache lookup re-rotate K at
  fresh window-relative indices.
* **Action tokens preserved** — RoPE is applied only to the spatial
  portion of each frame; the action tokens are spliced back in
  unchanged, mirroring the upstream behaviour.
* **Rotation tensor caching.** ``freqs_i`` (the per-position complex
  rotation tensor) is keyed by ``(id(freqs), tuple(rel_indices), h, w,
  head_dim_half)`` and reused across the 4 denoise passes within a
  chunk. The rotated cache prefix (the K positions that stay constant
  during the 4 passes of a single chunk) is also memoised per attention
  module, keyed by ``local_start_index``.
* **No fp64 cast.** The original ``causal_rope_apply`` upcasts to fp64
  for numerical headroom; the bf16/fp32 path is plenty for inference.

Activate via the :func:`infinity_rope_active` context manager (or
:func:`install` / :func:`restore` directly). Once active, every call to
``CausalWanSelfAttention.forward`` with ``kv_cache is not None``
takes the patched code path; ``kv_cache is None`` calls (training,
chain-mode inference) keep the original behaviour unchanged.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Optional

import torch

from wan.modules.attention import attention as flash_attention_fn
from wan.modules.causal_model import (
    CausalWanSelfAttention,
)

log = logging.getLogger(__name__)


_FREQS_I_CACHE: dict = {}


def _build_freqs_i(
    freqs: torch.Tensor,
    rel_indices: torch.Tensor,
    h: int,
    w: int,
    head_dim_half: int,
) -> torch.Tensor:
    """Return the per-position complex freqs tensor for the given
    relative frame indices and (h, w) grid. Cached across calls.

    ``freqs`` is split as
    ``[temp_dim, h_dim, w_dim] = [c-2*(c//3), c//3, c//3]`` where
    ``c = head_dim_half``. Temporal positions are indexed by
    ``rel_indices``; spatial dims use ``[:h]`` / ``[:w]``.
    """
    rel_tup = tuple(rel_indices.tolist())
    key = (id(freqs), rel_tup, h, w, head_dim_half)
    cached = _FREQS_I_CACHE.get(key)
    if cached is not None:
        return cached
    f = rel_indices.shape[0]
    seq_len = f * h * w
    c = head_dim_half
    temp_dim = c - 2 * (c // 3)
    h_dim = c // 3
    f_temp = freqs[rel_indices, :temp_dim].view(f, 1, 1, -1).expand(f, h, w, -1)
    f_h = freqs[:h, temp_dim:temp_dim + h_dim].view(1, h, 1, -1).expand(f, h, w, -1)
    f_w = freqs[:w, temp_dim + h_dim:].view(1, 1, w, -1).expand(f, h, w, -1)
    freqs_i = torch.cat([f_temp, f_h, f_w], dim=-1).reshape(seq_len, 1, -1).contiguous()
    _FREQS_I_CACHE[key] = freqs_i
    return freqs_i


def _block_relativistic_rope_fast(
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    rel_indices: torch.Tensor,
    action_tokens_per_frame: int,
) -> torch.Tensor:
    """LongLive's block-relativistic RoPE, action-token-aware, fp32 path.

    Args:
        x: ``[B, L, H, D]`` where
           ``L = f * (h*w + action_tokens_per_frame)``.
        rel_indices: ``[f]`` window-relative frame indices to rotate at.
    """
    B, L, H, D = x.shape
    f = rel_indices.shape[0]
    h, w = int(grid_sizes[0, 1].item()), int(grid_sizes[0, 2].item())
    spatial_per_frame = h * w
    a_per_f = action_tokens_per_frame
    expected_per_frame = spatial_per_frame + a_per_f

    if a_per_f > 0:
        x_per_frame = x.view(B, f, expected_per_frame, H, D)
        x_sp = (
            x_per_frame[:, :, :spatial_per_frame, :, :]
            .contiguous()
            .view(B, f * spatial_per_frame, H, D)
        )
        x_act = x_per_frame[:, :, spatial_per_frame:, :, :]
    else:
        x_sp = x[:, : f * spatial_per_frame]
        x_act = None

    seq_len = f * spatial_per_frame
    head_dim_half = D // 2
    freqs_i = _build_freqs_i(freqs, rel_indices, h, w, head_dim_half)

    x_sp_c = torch.view_as_complex(
        x_sp.float().reshape(B, seq_len, H, head_dim_half, 2)
    )
    rotated_c = x_sp_c * freqs_i
    rotated = torch.view_as_real(rotated_c).reshape(B, seq_len, H, D).to(x.dtype)

    if x_act is not None:
        rotated_per_frame = rotated.view(B, f, spatial_per_frame, H, D)
        out_per_frame = torch.cat([rotated_per_frame, x_act], dim=2)
        return out_per_frame.contiguous().view(B, L, H, D)
    return rotated


def patched_forward(
    self,
    x,
    seq_lens,
    grid_sizes,
    freqs,
    block_mask,
    kv_cache=None,
    current_start=0,
    cache_start=None,
):
    """Drop-in replacement for ``CausalWanSelfAttention.forward`` that
    stores un-rotated K in the cache and rotates Q + cached K with
    bounded window-relative indices at attention time.

    For ``kv_cache is None`` we delegate to the original forward — this
    keeps training and chain-mode inference untouched.
    """
    if kv_cache is None:
        return CausalWanSelfAttention._original_forward(
            self,
            x,
            seq_lens,
            grid_sizes,
            freqs,
            block_mask,
            kv_cache=None,
            current_start=current_start,
            cache_start=cache_start,
        )
    if cache_start is None:
        cache_start = current_start

    a_per_f = self.action_tokens_per_frame
    h_grid = int(grid_sizes[0, 1].item())
    w_grid = int(grid_sizes[0, 2].item())
    frame_seqlen = h_grid * w_grid + a_per_f
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    q = self.norm_q(self.q(x)).view(b, s, n, d)
    k = self.norm_k(self.k(x)).view(b, s, n, d)
    v = self.v(x).view(b, s, n, d)

    sink_tokens = self.sink_size * frame_seqlen
    kv_cache_size = kv_cache["k"].shape[1]
    _frozen_le = getattr(self, "_frozen_local_end_index", None)
    _frozen_ge = getattr(self, "_frozen_global_end_index", None)
    _cached_local_end_index = int(
        _frozen_le if _frozen_le is not None else kv_cache["local_end_index"].item()
    )
    _cached_global_end_index = int(
        _frozen_ge if _frozen_ge is not None else kv_cache["global_end_index"].item()
    )
    num_new_tokens = q.shape[1]
    num_new_frames = num_new_tokens // frame_seqlen
    current_end = current_start + num_new_tokens
    is_recompute = current_end <= _cached_global_end_index and current_start > 0

    cache_update_info: Optional[dict] = None
    rolled = False
    if (
        self.local_attn_size != -1
        and (current_end > _cached_global_end_index)
        and (num_new_tokens + _cached_local_end_index > kv_cache_size)
    ):
        rolled = True
        num_evicted_tokens = num_new_tokens + _cached_local_end_index - kv_cache_size
        num_rolled_tokens = _cached_local_end_index - num_evicted_tokens - sink_tokens
        local_end_index = (
            _cached_local_end_index
            + current_end
            - _cached_global_end_index
            - num_evicted_tokens
        )
        local_start_index = local_end_index - num_new_tokens
        temp_k = kv_cache["k"].clone()
        temp_v = kv_cache["v"].clone()
        if num_rolled_tokens > 0:
            temp_k[:, sink_tokens:sink_tokens + num_rolled_tokens] = temp_k[
                :,
                sink_tokens + num_evicted_tokens:
                sink_tokens + num_evicted_tokens + num_rolled_tokens,
            ].clone()
            temp_v[:, sink_tokens:sink_tokens + num_rolled_tokens] = temp_v[
                :,
                sink_tokens + num_evicted_tokens:
                sink_tokens + num_evicted_tokens + num_rolled_tokens,
            ].clone()
        write_start_index = (
            max(local_start_index, sink_tokens) if is_recompute else local_start_index
        )
        roped_offset = max(0, write_start_index - local_start_index)
        write_len = max(0, local_end_index - write_start_index)
        if write_len > 0:
            temp_k[:, write_start_index:local_end_index] = k[
                :, roped_offset:roped_offset + write_len
            ]
            temp_v[:, write_start_index:local_end_index] = v[
                :, roped_offset:roped_offset + write_len
            ]
        cache_update_info = {
            "action": "roll_and_insert",
            "sink_tokens": sink_tokens,
            "num_rolled_tokens": num_rolled_tokens,
            "num_evicted_tokens": num_evicted_tokens,
            "local_start_index": local_start_index,
            "local_end_index": local_end_index,
            "write_start_index": write_start_index,
            "write_end_index": local_end_index,
            "new_k": k[:, roped_offset:roped_offset + write_len].detach()
            if write_len > 0
            else None,
            "new_v": v[:, roped_offset:roped_offset + write_len].detach()
            if write_len > 0
            else None,
            "current_end": current_end,
            "is_recompute": is_recompute,
        }
        self._rot_prefix_k = None
        self._rot_prefix_local_start = -1
    else:
        local_end_index = (
            _cached_local_end_index + current_end - _cached_global_end_index
        )
        local_start_index = local_end_index - num_new_tokens
        temp_k = kv_cache["k"].clone()
        temp_v = kv_cache["v"].clone()
        write_start_index = (
            max(local_start_index, sink_tokens) if is_recompute else local_start_index
        )
        roped_offset = max(0, write_start_index - local_start_index)
        write_len = max(0, local_end_index - write_start_index)
        if write_len > 0:
            temp_k[:, write_start_index:local_end_index] = k[
                :, roped_offset:roped_offset + write_len
            ]
            temp_v[:, write_start_index:local_end_index] = v[
                :, roped_offset:roped_offset + write_len
            ]
        cache_update_info = {
            "action": "direct_insert",
            "local_start_index": local_start_index,
            "local_end_index": local_end_index,
            "write_start_index": write_start_index,
            "write_end_index": local_end_index,
            "new_k": k[:, roped_offset:roped_offset + write_len].detach()
            if write_len > 0
            else None,
            "new_v": v[:, roped_offset:roped_offset + write_len].detach()
            if write_len > 0
            else None,
            "current_end": current_end,
            "is_recompute": is_recompute,
        }

    # ---- Block-Relativistic RoPE ----
    if rolled:
        # Cache is full; Q anchored at the end of the local_attn_size
        # window. ``self.local_attn_size`` is in *frames* here (frames
        # of the live region, including sink + window) — that is the
        # contract enforced by ``utils.eval_causal_AR._set_attention_window``.
        q_start_idx = self.local_attn_size - num_new_frames
        query_rel_indices = torch.arange(
            q_start_idx,
            q_start_idx + num_new_frames,
            device=q.device,
        )
    else:
        current_frame_in_window = local_start_index // frame_seqlen
        query_rel_indices = torch.arange(
            current_frame_in_window,
            current_frame_in_window + num_new_frames,
            device=q.device,
        )

    num_cache_frames = local_end_index // frame_seqlen

    # The per-module prefix cache (``_rot_prefix_k`` /
    # ``_rot_prefix_local_start``) memoises the rotated prefix across the
    # 4 ODE denoise rungs of a single chunk so the prefix RoPE is only
    # computed on the FIRST rung; subsequent rungs hit the cache. This
    # is a real 4x speedup at inference (every rung shares the same
    # ``local_start_index``).
    #
    # CRITICAL: it must be gated on ``not torch.is_grad_enabled()``.
    # Under ``torch.utils.checkpoint`` (used in training for every
    # transformer block), the ORIGINAL grad-enabled forward and the
    # backward-time RECOMPUTATION are both run with grad enabled, but
    # the per-module cache state differs between them: the no_grad
    # rungs that fired BEFORE the original forward populate the cache,
    # but by the time recomputation runs, later chunks' rungs may have
    # overwritten or invalidated it (or the cache may be stale from a
    # previous chunk). That control-flow divergence makes the original
    # forward and the recompute save a different number of tensors
    # (``CheckpointError: A different number of tensors was saved``).
    #
    # Gating on ``not is_grad_enabled()`` makes both the grad-enabled
    # forward and its recomputation skip the cache (always compute
    # fresh), giving deterministic tensor counts. The no_grad rungs
    # and inference-time calls still hit the cache, preserving the
    # speedup. Within a grad-enabled forward each block is only called
    # once anyway, so skipping the cache costs nothing in throughput.
    use_prefix_cache = not torch.is_grad_enabled()
    cached_rot_local_start = (
        getattr(self, "_rot_prefix_local_start", -1) if use_prefix_cache else -1
    )
    cached_rot_k = (
        getattr(self, "_rot_prefix_k", None) if use_prefix_cache else None
    )
    if (
        cached_rot_local_start == local_start_index
        and cached_rot_k is not None
        and not rolled
    ):
        rotated_prefix = cached_rot_k
    else:
        if local_start_index > 0:
            prefix_frames = local_start_index // frame_seqlen
            grid_prefix = grid_sizes.clone()
            grid_prefix[:, 0] = prefix_frames
            prefix_k = temp_k[:, :local_start_index]
            prefix_rel_indices = torch.arange(0, prefix_frames, device=k.device)
            rotated_prefix = _block_relativistic_rope_fast(
                prefix_k, grid_prefix, freqs, prefix_rel_indices, a_per_f,
            )
        else:
            rotated_prefix = temp_k[:, :0]
        if use_prefix_cache:
            self._rot_prefix_k = rotated_prefix
            self._rot_prefix_local_start = local_start_index

    live_k = temp_k[:, local_start_index:local_end_index]
    live_frames = (local_end_index - local_start_index) // frame_seqlen
    if live_frames > 0:
        grid_live = grid_sizes.clone()
        grid_live[:, 0] = live_frames
        live_start = num_cache_frames - live_frames
        live_rel_indices = torch.arange(live_start, num_cache_frames, device=k.device)
        rotated_live = _block_relativistic_rope_fast(
            live_k, grid_live, freqs, live_rel_indices, a_per_f,
        )
    else:
        rotated_live = live_k

    rotated_temp_k = torch.cat([rotated_prefix, rotated_live], dim=1)

    roped_query = _block_relativistic_rope_fast(
        q, grid_sizes, freqs, query_rel_indices, a_per_f,
    )

    temp_v_full = temp_v[:, :local_end_index]
    if sink_tokens > 0:
        local_budget = self.max_attention_size - sink_tokens
        k_sink = rotated_temp_k[:, :sink_tokens]
        v_sink = temp_v_full[:, :sink_tokens]
        if local_budget > 0:
            local_start_for_window = max(sink_tokens, local_end_index - local_budget)
            k_local = rotated_temp_k[:, local_start_for_window:local_end_index]
            v_local = temp_v_full[:, local_start_for_window:local_end_index]
            k_cat = torch.cat([k_sink, k_local], dim=1)
            v_cat = torch.cat([v_sink, v_local], dim=1)
        else:
            k_cat = k_sink
            v_cat = v_sink
        attn_out = flash_attention_fn(roped_query, k_cat, v_cat)
    else:
        window_start = max(0, local_end_index - self.max_attention_size)
        attn_out = flash_attention_fn(
            roped_query,
            rotated_temp_k[:, window_start:local_end_index],
            temp_v_full[:, window_start:local_end_index],
        )

    out = attn_out.flatten(2)
    out = self.o(out)
    return out, (current_end, local_end_index, cache_update_info)


def _clear_module_state(root_module=None) -> None:
    """Drop the per-module rotated-prefix caches.

    Called from :func:`install` to make sure stale rotated prefixes
    from an earlier ``install / restore`` cycle don't bleed into a new
    rollout. Walks every ``CausalWanSelfAttention`` reachable from
    ``root_module``; if ``root_module`` is ``None`` the caller is
    responsible for clearing state on its own modules (typical for the
    install-once path).
    """
    if root_module is None:
        return
    for module in root_module.modules():
        if isinstance(module, CausalWanSelfAttention):
            if hasattr(module, "_rot_prefix_k"):
                module._rot_prefix_k = None
            if hasattr(module, "_rot_prefix_local_start"):
                module._rot_prefix_local_start = -1


def install(root_module=None) -> None:
    """Install the Infinity-RoPE patch on ``CausalWanSelfAttention``.

    Idempotent: calling ``install()`` while the patch is already active
    is a no-op. ``root_module`` is optional; if provided, any per-module
    rotated-prefix cache attached to its children is cleared.
    """
    if not hasattr(CausalWanSelfAttention, "_original_forward"):
        CausalWanSelfAttention._original_forward = CausalWanSelfAttention.forward
    CausalWanSelfAttention.forward = patched_forward
    _FREQS_I_CACHE.clear()
    _clear_module_state(root_module)
    log.info("[infinity_rope] installed Block-Relativistic RoPE patch")


def restore(root_module=None) -> None:
    """Undo :func:`install`. Safe to call when not installed."""
    if hasattr(CausalWanSelfAttention, "_original_forward"):
        CausalWanSelfAttention.forward = CausalWanSelfAttention._original_forward
        del CausalWanSelfAttention._original_forward
    _FREQS_I_CACHE.clear()
    _clear_module_state(root_module)


@contextlib.contextmanager
def infinity_rope_active(enabled: bool, root_module=None):
    """Context manager that activates the patch only when ``enabled``.

    Use around an inference call::

        with infinity_rope_active(args.infinity_rope, base_dit):
            full_latents = pipe.generate_ar(...)

    When ``enabled=False`` this is a no-op and the original forward
    runs unchanged.
    """
    if not enabled:
        yield
        return
    install(root_module)
    try:
        yield
    finally:
        restore(root_module)


def is_active() -> bool:
    """True iff the patch is currently installed."""
    return hasattr(CausalWanSelfAttention, "_original_forward")
