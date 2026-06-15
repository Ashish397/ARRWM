"""LADD-style discriminator with adjacent-chunk adversarial alignment.

References (cloned in ``references/``):
    LADD (Latent Adversarial Diffusion Distillation): Nitro-1
        references/Nitro-1/core/network/transformer_D.py
        Uses pretrained DiT as the disc backbone, taps intermediate
        features at fixed block indices, frozen backbone + trainable
        heads, BCE-with-logits loss. We follow the same pattern but
        adapt to WAN (causal video transformer), use RpGAN
        (relativistic), and operate on adjacent-chunk pairs from a
        single causal rollout instead of (student, GT).

    Projected GAN: references/projected-gan/pg_modules/{projector.py,
        discriminator.py}. Multi-scale features through a learned
        Cross-Channel-Mixing (CCM) projection and Cross-Scale-Mixing
        (CSM) FPN-style fusion before per-scale heads. We adopt
        both CCM and CSM.

    ASD (Adversarial Self-Distillation): SAD/readme.md (no code
        released). The paper aligns same-student n-step vs (n+1)-step
        outputs in the distribution level (= adversarial). We adapt
        the "adjacent in some dimension" principle from denoising
        steps to spatially-adjacent CHUNKS in a causal rollout
        (chunk_i = real-anchor, chunk_{i+1} = fake / push-target).

Overall flow per gen-step iter:
    1. Build adjacent-chunk pairs from the gen-side rollout output
       (``pred_image`` or ``flash_dmd_gan_x0``).
    2. Re-noise each chunk member at ``flash_dmd_gan_t`` (when
       flash-DMD is enabled) or at the DMD step's ``t`` (otherwise).
    3. ``WanFeatureProjector`` runs a single ``no_grad``-on-teacher
       forward through ``real_score``, capturing intermediate features
       at configurable block indices. The input tensor carries
       ``requires_grad=True`` on the gen-side so gradients flow back.
    4. ``LADDChannelMixer`` (CCM): per-tap 1x1 linear projection to a
       common channel dim. Trainable.
    5. ``LADDFeatureFusion`` (CSM, optional): FPN-style top-down
       fusion across taps. Trainable.
    6. ``LADDDiscHead`` (per-scale): SpectralConv1d + GroupNorm +
       LeakyReLU residual block, ending in a scalar logit per token.
       Trainable.
    7. Concatenate logits, compute RpGAN loss (D-step: train CCM + CSM
       + heads; G-step: backprop only through fake_chunk side).

The teacher backbone is left frozen (its params already have
``requires_grad=False`` after the real_score build path). The disc's
trainable params are CCM + CSM + heads only (~10-15M). The
``WanFeatureProjector`` is NOT itself an nn.Module with parameters —
it's a lightweight callable that holds a weak reference to
``real_score`` and registers/manages forward hooks.

DiffAugment-equivalent on latents: applied identically to real and
fake before they hit the projector. Same per-sample randomness; both
sides see the same augmentation. Differentiable so gradients flow
through the augmentation back to the gen's fake input.
"""

from __future__ import annotations

import math
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# ============================================================================
# Feature projector (frozen teacher tap)
# ============================================================================


class WanFeatureProjector:
    """Frozen teacher feature extractor via forward hooks on real_score.

    Holds a reference to the existing ``real_score`` (WAN transformer)
    and registers forward hooks on a configurable list of transformer
    block indices. On forward, runs a single teacher forward and
    returns the captured per-block outputs.

    Args:
        real_score: the WAN teacher wrapper (e.g.
            ``WanDiffusionWrapper``). The underlying ``WanModel`` is
            accessed via ``.model`` (or unwrapped if DDP-wrapped).
        block_indices: list of ``transformer_blocks`` indices to hook.
        latent_input_carries_grad: if True, asserts the input has
            ``requires_grad=True`` on disc forward (gen-side path).
            Even when teacher params are frozen, the captured features
            still carry grad w.r.t. the input.

    The projector is NOT an nn.Module — it's a callable manager around
    hook lifecycles. Teacher params are not part of disc.parameters().
    """

    def __init__(
        self,
        real_score: nn.Module,
        block_indices: List[int],
    ):
        self.real_score = real_score
        self.block_indices = sorted(set(int(i) for i in block_indices))
        # NOTE: no persistent forward hooks are installed. Every call
        # to ``real_score`` from elsewhere in the trainer (DMD scoring,
        # aux LoRA pass, real_teacher train, gen rollout) would fire
        # persistent hooks and pin graph-attached block outputs across
        # hundreds of teacher forwards per step. ``__call__`` installs
        # LOCAL hooks scoped to the disc's own teacher forward and
        # removes them in ``finally``.
        self._validate_block_indices()

    # ------------------------------------------------------------------
    def _find_blocks(self) -> nn.ModuleList:
        """Find the WAN transformer-block ``nn.ModuleList`` anywhere
        under ``real_score``.

        Walks all submodules so arbitrary wrap depth is handled (DDP /
        ``WanDiffusionWrapper`` / v14 LoRA / alt-head plumbing). Picks
        the first ModuleList named ``transformer_blocks`` or ``blocks``
        with at least 10 entries (WAN 1.3B has 30; the threshold guards
        against picking up some other small ModuleList by accident).
        """
        for candidate in [self.real_score] + list(
            self.real_score.modules()
        ):
            for attr in ("transformer_blocks", "blocks"):
                b = getattr(candidate, attr, None)
                if isinstance(b, nn.ModuleList) and len(b) >= 10:
                    return b
        raise AttributeError(
            "WanFeatureProjector: could not find a transformer-block "
            f"ModuleList anywhere under {type(self.real_score).__name__}. "
            "Expected attribute ``transformer_blocks`` or ``blocks`` on "
            "some submodule with >=10 entries."
        )

    def _validate_block_indices(self) -> None:
        blocks = self._find_blocks()
        n_blocks = len(blocks)
        for idx in self.block_indices:
            if idx < 0 or idx >= n_blocks:
                raise IndexError(
                    f"WanFeatureProjector: block index {idx} out of range "
                    f"for {n_blocks}-block teacher."
                )

    # ------------------------------------------------------------------
    def __call__(
        self,
        x_noisy: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        *,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None,
        conditional_extra: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[int, torch.Tensor]:
        """Run the teacher forward and return captured per-block features.

        The teacher params are frozen so backbone activations are NOT
        held for gradient computation; only the input-to-feature path
        is differentiable (which is what the disc needs for the
        gen-side gradient).

        ``conditional_extra`` is merged into the conditional_dict sent
        to ``real_score`` — used to inject ``_action_tokens`` /
        ``_action_modulation`` for action-aware WAN configurations.
        """
        # The disc (CCM/CSM/heads) is built in fp32 for R1 stability,
        # so x_noisy and the conditional tensors arrive here in fp32.
        # The WAN teacher's weights are in bf16; passing fp32 inputs
        # into a bf16 Conv3d patch-embedding raises "Input type (float)
        # and bias type (c10::BFloat16) should be the same". Detect
        # the teacher's parameter dtype and cast all float tensors
        # going into the teacher to match. Hook outputs come back in
        # bf16, and CCM's first Linear casts them to fp32 again.
        teacher_dtype = next(self.real_score.parameters()).dtype

        def _cast_if_float(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if t is None or not torch.is_floating_point(t):
                return t
            if t.dtype == teacher_dtype:
                return t
            return t.to(dtype=teacher_dtype)

        x_noisy_cast = _cast_if_float(x_noisy)
        cond_dict: Dict[str, torch.Tensor] = {
            "prompt_embeds": _cast_if_float(prompt_embeds),
        }
        if conditional_extra is not None:
            for k, v in conditional_extra.items():
                cond_dict[k] = _cast_if_float(v) if (
                    isinstance(v, torch.Tensor)
                ) else v
        # Forward the WAN teacher wrapper. We don't care about its
        # return value — the hooks capture what we need. Don't wrap in
        # ``no_grad()``: that would sever the input → feature autograd
        # path. Teacher params have ``requires_grad=False`` so they
        # accumulate no gradient.
        kwargs = {
            "noisy_image_or_video": x_noisy_cast,
            "conditional_dict": cond_dict,
            "timestep": timestep,
        }
        if clean_x is not None:
            kwargs["clean_x"] = _cast_if_float(clean_x)
        if aug_t is not None:
            kwargs["aug_t"] = aug_t
        if seq_len is not None:
            kwargs["seq_len"] = seq_len

        # Outer-level gradient checkpointing on the projector's teacher
        # forward (v28A OOM mitigation 4). Wraps the entire teacher call
        # in ``torch.utils.checkpoint.checkpoint(use_reentrant=False)``
        # so the per-block intermediate activations are recomputed on
        # backward instead of held.
        #
        # The hooks are installed locally for this call only and
        # removed in ``finally`` so the next teacher forward (from
        # anywhere else in the trainer) is not affected. Captured
        # features are returned as a tuple from the checkpointed
        # function so they're tracked as outputs and ``use_reentrant=
        # False`` keeps autograd's connection through replay.
        block_indices_sorted = sorted(self.block_indices)
        local_feats: Dict[int, torch.Tensor] = {}
        hook_handles = []
        blocks = self._find_blocks()

        def _make_local_hook(block_idx: int):
            def _local_hook(_m, _i, out):
                feat = out[0] if isinstance(out, tuple) else out
                local_feats[block_idx] = feat
            return _local_hook

        for idx in block_indices_sorted:
            hook_handles.append(
                blocks[idx].register_forward_hook(_make_local_hook(idx))
            )

        def _run_teacher(x):
            kwargs_local = dict(kwargs)
            kwargs_local["noisy_image_or_video"] = x
            _ = self.real_score(**kwargs_local)
            return tuple(local_feats[i] for i in block_indices_sorted)

        try:
            if x_noisy_cast.requires_grad and torch.is_grad_enabled():
                feat_tuple = torch.utils.checkpoint.checkpoint(
                    _run_teacher,
                    x_noisy_cast,
                    use_reentrant=False,
                )
            else:
                feat_tuple = _run_teacher(x_noisy_cast)
        finally:
            for h in hook_handles:
                h.remove()
            local_feats.clear()
        return {
            idx: feat_tuple[i] for i, idx in enumerate(block_indices_sorted)
        }


# ============================================================================
# Cross-channel mixing (CCM) — Projected-GAN style
# ============================================================================


class LADDChannelMixer(nn.Module):
    """Per-tap learned channel mixing.

    Each tap's features (shape ``[B, N_tokens, dim_teacher]``) get a
    1x1 linear projection to a common ``dim_proj``. Tiny capacity
    (a few million params total) but lets the heads operate on a
    learned, low-rank channel space rather than raw teacher features.

    Output: ``Dict[block_idx, Tensor[B, N_tokens, dim_proj]]`` — same
    keys as the input dict.
    """

    def __init__(
        self,
        block_indices: List[int],
        dim_teacher: int,
        dim_proj: int,
    ):
        super().__init__()
        self.block_indices = sorted(set(int(i) for i in block_indices))
        self.dim_teacher = int(dim_teacher)
        self.dim_proj = int(dim_proj)
        self.proj = nn.ModuleDict(
            {str(i): nn.Linear(self.dim_teacher, self.dim_proj)
             for i in self.block_indices}
        )

    def forward(
        self,
        features: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        out: Dict[int, torch.Tensor] = {}
        for idx in self.block_indices:
            feat = features[idx]
            # feat may arrive as bf16 from the WAN forward; project to
            # fp32 for downstream R1/R2 stability.
            out[idx] = self.proj[str(idx)](feat.float())
        return out


# ============================================================================
# Cross-scale mixing (CSM) — FPN-style top-down fusion
# ============================================================================


class LADDFeatureFusion(nn.Module):
    """FPN-style top-down fusion across the per-tap projected features.

    All taps share the same ``dim_proj`` channel dim (set by CCM) and
    the same token count (WAN transformer keeps the token count
    constant across blocks). So fusion is just elementwise:

        fused[deepest] = proj[deepest]
        fused[i]       = proj[i] + fused[i+1]   (for i < deepest)

    Lateral projections optional — when ``use_lateral_proj=True``,
    each non-deepest level gets an additional 1x1 linear lateral
    transform. Default off (the CCM already learned a per-tap
    projection, so an additional lateral often just adds noise at
    init).

    Returns the SAME dict shape as input (one fused tensor per tap).
    """

    def __init__(
        self,
        block_indices: List[int],
        dim_proj: int,
        use_lateral_proj: bool = False,
    ):
        super().__init__()
        self.block_indices = sorted(set(int(i) for i in block_indices))
        self.dim_proj = int(dim_proj)
        self.use_lateral_proj = bool(use_lateral_proj)
        if self.use_lateral_proj:
            # One lateral per non-deepest level.
            self.lateral = nn.ModuleDict(
                {str(i): nn.Linear(self.dim_proj, self.dim_proj)
                 for i in self.block_indices[:-1]}
            )
        else:
            self.lateral = None

    def forward(
        self,
        features: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        # Top-down: start at the deepest (largest block_idx).
        sorted_idx = self.block_indices
        fused: Dict[int, torch.Tensor] = {}
        fused[sorted_idx[-1]] = features[sorted_idx[-1]]
        # Walk from second-deepest to shallowest.
        for i in range(len(sorted_idx) - 2, -1, -1):
            idx = sorted_idx[i]
            deeper_idx = sorted_idx[i + 1]
            lat = features[idx]
            if self.lateral is not None:
                lat = self.lateral[str(idx)](lat)
            fused[idx] = lat + fused[deeper_idx]
        return fused


# ============================================================================
# Disc head — 1D SpectralConv on flattened token sequence
# ============================================================================


def _spectral_conv2d(
    in_ch: int, out_ch: int, kernel_size: int, padding: int = 0,
) -> nn.Conv2d:
    """2D conv with SpectralNorm applied — D-stability standard."""
    conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
    return spectral_norm(conv, n_power_iterations=1)


class _ResBlock2D(nn.Module):
    """Residual block: GroupNorm → LeakyReLU → SpectralConv2d (k=K) →
    GroupNorm → LeakyReLU → SpectralConv2d (k=1)."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = _spectral_conv2d(
            channels, channels, kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = _spectral_conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        h = self.norm1(x)
        h = F.leaky_relu(h, 0.2, inplace=True)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.leaky_relu(h, 0.2, inplace=True)
        h = self.conv2(h)
        return (x + h) / math.sqrt(2.0)


class LADDDiscHead(nn.Module):
    """Single-scale disc head operating on 2D patch-grid features.

    Input shape: ``[B*T', dim_proj, H', W']`` — spatial map per
    (sample, frame). Caller is responsible for stripping action
    tokens and reshaping the token sequence into this 2D layout.

    Output shape: ``[B*T', cmap_dim or 1, H', W']``. Caller flattens
    + concatenates across scales for the final logit vector.
    """

    def __init__(
        self,
        dim_proj: int,
        kernel_size: int = 3,
        cmap_dim: int = 0,
    ):
        super().__init__()
        self.dim_proj = int(dim_proj)
        self.cmap_dim = int(cmap_dim)
        self.in_block = nn.Sequential(
            nn.GroupNorm(8, self.dim_proj),
            nn.LeakyReLU(0.2, inplace=True),
            _spectral_conv2d(self.dim_proj, self.dim_proj, kernel_size=1),
        )
        self.res_block = _ResBlock2D(self.dim_proj, kernel_size=kernel_size)
        if self.cmap_dim > 0:
            self.cls = _spectral_conv2d(
                self.dim_proj, self.cmap_dim, kernel_size=1,
            )
        else:
            self.cls = _spectral_conv2d(self.dim_proj, 1, kernel_size=1)

    def forward(
        self,
        feat: torch.Tensor,
        cmap: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # feat: [B*T', dim_proj, H', W']
        x = self.in_block(feat)
        x = self.res_block(x)
        out = self.cls(x)
        if self.cmap_dim > 0:
            if cmap is None:
                raise ValueError(
                    "LADDDiscHead built with cmap_dim>0 but no cmap "
                    "argument provided to forward()."
                )
            # cmap: [B, cmap_dim] needs broadcasting over T'. The
            # caller passes the [B, cmap_dim] form; we must expand by
            # the temporal-fold factor (T' = first_dim // B) to match
            # the [B*T', cmap_dim, ...] activation. Recover T' from
            # the activation's batch axis: B_eff = first_dim, T' =
            # B_eff // cmap.shape[0].
            B_eff = out.shape[0]
            B_cmap = cmap.shape[0]
            t_frames = max(1, B_eff // B_cmap)
            cmap_expanded = cmap.repeat_interleave(t_frames, dim=0)
            cmap_b = cmap_expanded.unsqueeze(-1).unsqueeze(-1)  # [B*T', cmap_dim, 1, 1]
            out = (out * cmap_b).sum(1, keepdim=True) * (
                1.0 / math.sqrt(self.cmap_dim)
            )
        return out


# ============================================================================
# Parallel "stat head" — distribution-match per-frame / per-channel /
# spatiotemporal stds adversarially (sideband to the visual disc).
# ============================================================================


class LADDStatHead(nn.Module):
    """Parallel adversarial branch on latent std statistics.

    Computes three std reductions from the raw input latent and routes
    them through a small MLP to produce a per-sample scalar logit that
    sits ALONGSIDE the visual disc's per-token logits at the RpGAN loss
    site. The visual disc and the stat head share only the loss
    function — no teacher involvement, no shared weights.

    The three reductions (axes KEPT, the rest are std'd over):
      * ``[B, F, C]``       — non-spatial std (over H, W).
          Catches grey collapse + channel imbalance.
      * ``[B, F, P, P]``    — std over channels, then avg-pool spatial
                              to ``P × P``. Catches spatial-contrast.
      * ``[B, C, P, P]``    — std over frames, then avg-pool spatial
                              to ``P × P``. Catches temporal stability
                              (AR drift / flicker signature).
    Total input dim to the MLP: ``F·C + F·P² + C·P²``.

    The head emits a single scalar logit per sample (shape ``[B, 1]``).
    Stat-side strength relative to the visual disc is controlled at
    the loss-aggregation level via ``ladd_stat_head_loss_weight`` —
    the trainer reduces the visual and stat sides via separate RpGAN
    means and sums them with that weight. This is the right knob;
    broadcasting the scalar logit K times before concat would only
    fight the per-token visual dilution and is not how to control
    stat-side strength.

    Args:
        frames_per_window: F = npb (or 2·npb if disc-window upgrade is
            on). Set at build time; must match runtime input.
        in_channels: latent channel count (16 for Wan VAE).
        pool_size: spatial pool target P (default 4 → 16 dims per map).
        hidden_dim: MLP hidden dim. Default 256.
        eps: numerical floor for ``.std()``.

    The MLP layers are spectral-normed for stability (mirrors the
    visual heads' D-side regularisation).
    """

    def __init__(
        self,
        frames_per_window: int = 3,
        in_channels: int = 16,
        pool_size: int = 4,
        hidden_dim: int = 256,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.frames = int(frames_per_window)
        self.in_channels = int(in_channels)
        self.pool_size = int(pool_size)
        self.hidden_dim = int(hidden_dim)
        self.eps = float(eps)
        # Total stat vector dim.
        dim_bfc = self.frames * self.in_channels
        dim_bfhw = self.frames * self.pool_size * self.pool_size
        dim_bchw = self.in_channels * self.pool_size * self.pool_size
        self.stat_dim = dim_bfc + dim_bfhw + dim_bchw
        self.fc1 = spectral_norm(nn.Linear(self.stat_dim, self.hidden_dim))
        self.fc2 = spectral_norm(nn.Linear(self.hidden_dim, 1))

    # ------------------------------------------------------------------
    def compute_stats(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the concatenated std-stat vector from ``[B, F, C, H, W]``.

        Returns ``[B, stat_dim]`` (fp32 for numerical stability).
        Differentiable wrt ``x`` — gen-side gradient flows back through
        each std reduction.
        """
        if x.dim() != 5:
            raise ValueError(
                f"LADDStatHead.compute_stats expects [B,F,C,H,W]; got "
                f"{tuple(x.shape)}."
            )
        B, F_in, C_in, H_in, W_in = x.shape
        if F_in != self.frames:
            raise RuntimeError(
                "LADDStatHead frames_per_window mismatch: built for "
                f"frames={self.frames}, got input F={F_in}. Rebuild the "
                "disc with the correct ``ladd_stat_head_frames_per_window``."
            )
        if C_in != self.in_channels:
            raise RuntimeError(
                "LADDStatHead in_channels mismatch: built for "
                f"in_channels={self.in_channels}, got input C={C_in}."
            )
        x32 = x.float()
        # [B, F, C] — std over (H, W). Catches grey collapse.
        s_bfc = x32.std(dim=[3, 4], unbiased=False)
        # [B, F, H, W] — std over C, then avg-pool spatial → [B, F, P, P].
        s_bfhw = x32.std(dim=2, unbiased=False)
        # F.adaptive_avg_pool2d expects [N, C, H, W] — fold F into the
        # batch axis temporarily.
        s_bfhw_pooled = F.adaptive_avg_pool2d(
            s_bfhw.reshape(B * F_in, 1, H_in, W_in),
            self.pool_size,
        ).reshape(B, F_in, self.pool_size, self.pool_size)
        # [B, C, H, W] — std over F, then avg-pool spatial → [B, C, P, P].
        s_bchw = x32.std(dim=1, unbiased=False)
        s_bchw_pooled = F.adaptive_avg_pool2d(s_bchw, self.pool_size)
        # Concat flat.
        v = torch.cat(
            [
                s_bfc.reshape(B, -1),
                s_bfhw_pooled.reshape(B, -1),
                s_bchw_pooled.reshape(B, -1),
            ],
            dim=1,
        )
        return v

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, F, C, H, W]`` → ``[B, 1]`` per-sample stat logit."""
        v = self.compute_stats(x)
        h = F.gelu(self.fc1(v))
        logit = self.fc2(h)  # [B, 1]
        return logit


# ============================================================================
# Full LADD discriminator
# ============================================================================


class LADDDiscriminator(nn.Module):
    """LADD discriminator: optional wavelet-HF → CCM → optional CSM →
    per-scale heads.

    The frozen teacher feature projector is stored as a reference
    (NOT a submodule, NOT in this nn.Module's params).

    Args:
        projector: ``WanFeatureProjector`` instance (callable, frozen).
        block_indices: tap indices used by the projector.
        dim_teacher: WAN transformer hidden dim (1536 for 1.3B, 5120
            for 14B).
        dim_proj: common CCM/head channel dim. Default 256.
        use_csm: include cross-scale FPN fusion. Default True.
        use_lateral_proj: include lateral projections in CSM. Default
            False (CCM is already a projection).
        head_kernel_size: kernel size for the disc head's residual
            block. Default 9.
        cmap_dim: prompt-conditioning dim for heads (0 = disabled).
        prompt_embed_dim: input prompt embedding dim (required when
            cmap_dim>0).
        wavelet_hf_enabled: prepend a SWT-based wavelet HF stage so
            the projector sees a wavelet-decomposed view of the input
            (WGSR-style frequency-band conditioning). Default False.
        wavelet_hf_in_channels: input latent channel count for the
            wavelet stage (16 for Wan VAE).
        wavelet_hf_drop_ll: drop the LL band in the wavelet stage.
            Default False — LL carries low-frequency content the disc
            should see, downweighted via ``wavelet_hf_ll_weight``.
        wavelet_hf_adapter_init_gain: Xavier gain for the wavelet
            adapter's weight init. Small (~0.1) keeps the projector
            in-distribution at step 0.
        wavelet_hf_ll_weight: relative weight on the LL band when
            ``wavelet_hf_drop_ll=False``. Default 0.15 — LL has ~4×
            the magnitude of HF bands on smoothed WAN latents; 0.15
            balances all 4 bands at the adapter input. Setting this
            too high (≥0.5) lets LL swamp the disc heads' spectral_norm
            and causes silent NCCL hangs (per-rank ``_u`` drift).
        stat_head_enabled: opt-in parallel ``LADDStatHead`` that
            distribution-matches latent std statistics adversarially.
            Computes 3 std reductions on the RAW input latent (before
            wavelet HF), routes them through a small MLP, and emits a
            per-sample scalar logit that gets concatenated to the
            per-token visual logits at the RpGAN loss site. Default
            False.
        stat_head_frames_per_window: F dimension the stat head expects
            at runtime (must match ``ladd_pairs_per_step``'s slice
            size; npb=3 with single-chunk pairs, 2·npb=6 with the
            2-chunk-window upgrade).
        stat_head_pool_size: spatial pool target P. Default 4 → 16
            elements per spatial-map reduction.
        stat_head_hidden_dim: stat MLP hidden width. Default 256.
    """

    def __init__(
        self,
        projector: WanFeatureProjector,
        block_indices: List[int],
        dim_teacher: int,
        dim_proj: int = 256,
        use_csm: bool = True,
        use_lateral_proj: bool = False,
        head_kernel_size: int = 3,
        cmap_dim: int = 0,
        prompt_embed_dim: int = 0,
        wavelet_hf_enabled: bool = False,
        wavelet_hf_in_channels: int = 16,
        wavelet_hf_drop_ll: bool = False,
        wavelet_hf_adapter_init_gain: float = 0.1,
        wavelet_hf_ll_weight: float = 0.15,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        action_tokens_per_frame: int = 0,
        stat_head_enabled: bool = False,
        stat_head_frames_per_window: int = 3,
        stat_head_pool_size: int = 4,
        stat_head_hidden_dim: int = 256,
    ):
        super().__init__()
        self.projector = projector  # stored as plain attribute, not nn submodule
        self.block_indices = sorted(set(int(i) for i in block_indices))
        self.dim_teacher = int(dim_teacher)
        self.dim_proj = int(dim_proj)
        self.use_csm = bool(use_csm)
        self.cmap_dim = int(cmap_dim)
        self.wavelet_hf_enabled = bool(wavelet_hf_enabled)
        # WAN patch_size + per-frame action-token count; needed to
        # reshape the captured token sequence back to the 2D
        # patch-grid layout (T', H', W') + a_per_f-per-frame action
        # tokens, which we strip before running 2D heads.
        self.patch_size = tuple(int(p) for p in patch_size)
        self.action_tokens_per_frame = int(action_tokens_per_frame)
        if self.wavelet_hf_enabled:
            from model.wavelet_hf import LatentWaveletHF
            self.wavelet_hf = LatentWaveletHF(
                in_channels=int(wavelet_hf_in_channels),
                drop_ll=bool(wavelet_hf_drop_ll),
                adapter_init_gain=float(wavelet_hf_adapter_init_gain),
                ll_weight=float(wavelet_hf_ll_weight),
            )
        else:
            self.wavelet_hf = None

        self.ccm = LADDChannelMixer(
            block_indices=self.block_indices,
            dim_teacher=self.dim_teacher,
            dim_proj=self.dim_proj,
        )
        if self.use_csm:
            self.csm = LADDFeatureFusion(
                block_indices=self.block_indices,
                dim_proj=self.dim_proj,
                use_lateral_proj=bool(use_lateral_proj),
            )
        else:
            self.csm = None
        self.heads = nn.ModuleDict(
            {str(i): LADDDiscHead(
                dim_proj=self.dim_proj,
                kernel_size=head_kernel_size,
                cmap_dim=self.cmap_dim,
            ) for i in self.block_indices}
        )
        if self.cmap_dim > 0:
            if prompt_embed_dim <= 0:
                raise ValueError(
                    "LADDDiscriminator: prompt_embed_dim must be >0 when "
                    "cmap_dim>0."
                )
            self.cmapper = nn.Linear(prompt_embed_dim, self.cmap_dim)
        else:
            self.cmapper = None

        # Parallel stat head — distribution-match std statistics
        # adversarially. Operates on the RAW input latent, NOT on the
        # wavelet-HF output. The two branches share only the RpGAN
        # loss; their gradients combine at the gen side.
        self.stat_head_enabled = bool(stat_head_enabled)
        if self.stat_head_enabled:
            self.stat_head = LADDStatHead(
                frames_per_window=int(stat_head_frames_per_window),
                in_channels=int(wavelet_hf_in_channels),
                pool_size=int(stat_head_pool_size),
                hidden_dim=int(stat_head_hidden_dim),
            )
        else:
            self.stat_head = None

    # ------------------------------------------------------------------
    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def stat_logit_count(self) -> int:
        """Number of stat-head logits appended at the END of ``forward``'s
        output (always 1 when stat head is enabled, 0 otherwise). Used
        by the trainer to split the concatenated logit tensor into the
        visual section (first ``-stat_logit_count`` columns) and the
        stat section (last column) so each side can be reduced by its
        own RpGAN mean — preventing the per-token visual logits from
        drowning out the per-sample stat logit at the loss level. The
        stat-side strength is then controlled by
        ``ladd_stat_head_loss_weight``.
        """
        return 1 if self.stat_head is not None else 0

    # ------------------------------------------------------------------
    def forward(
        self,
        x_noisy: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        *,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None,
        pooled_prompt: Optional[torch.Tensor] = None,
        conditional_extra: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward through projector → CCM → optional CSM → heads.

        Returns flattened per-token logits concatenated across taps,
        with the optional stat head's per-sample logit appended:
            ``[B, total_visual_tokens (+ 1)]``

        The trainer splits the visual side from the stat side (last
        column when ``stat_head_enabled``) and reduces each through
        its own RpGAN mean. See ``LADDStatHead`` and the trainer's
        ``_ladd_run_pair_mode`` for the loss-aggregation logic.
        """
        # Capture the raw input BEFORE the wavelet transform — the stat
        # head operates on the raw latent statistics, not the
        # wavelet-HF representation. Both branches see the same noised
        # latent (same disc_t_int) and the same DiffAugment.
        x_noisy_raw = x_noisy

        # Optional wavelet-HF pre-stage: restrict the projector to the
        # HF sub-bands of the input latent so the GAN gradient can
        # only push HF detail (content stays DMD's job). Spatial
        # resolution is preserved (SWT, not DWT) so the projector's
        # forward is in-distribution apart from the channel content.
        if self.wavelet_hf is not None:
            _wave = self.wavelet_hf(x_noisy)
            # wavelet_hf_augment (default off): ADD the wavelet HF view to the
            # raw latent instead of REPLACING it, so the disc sees BOTH
            # modalities — raw structure (the view replacing blinded it to) AND
            # the HF detail — rather than HF-only. Same shape (adapter maps the
            # HF bands back to in_channels), so the projector is unchanged.
            x_noisy = (
                x_noisy + _wave
                if getattr(self, "wavelet_hf_augment", False)
                else _wave
            )
        feats = self.projector(
            x_noisy=x_noisy,
            timestep=timestep,
            prompt_embeds=prompt_embeds,
            clean_x=clean_x,
            aug_t=aug_t,
            seq_len=seq_len,
            conditional_extra=conditional_extra,
        )
        # Sanity: projector must have returned all expected blocks.
        missing = [i for i in self.block_indices if i not in feats]
        if missing:
            raise RuntimeError(
                f"LADDDiscriminator: projector returned no features for "
                f"block indices {missing}. Hook setup is broken — verify "
                f"that real_score's transformer_blocks indices are valid."
            )
        proj = self.ccm(feats)
        if self.csm is not None:
            proj = self.csm(proj)
        cmap = None
        if self.cmap_dim > 0:
            if pooled_prompt is None:
                raise ValueError(
                    "LADDDiscriminator built with cmap_dim>0 but "
                    "pooled_prompt not provided."
                )
            cmap = self.cmapper(pooled_prompt.float())

        # Reshape each tap's token sequence to a 2D patch grid so the
        # heads can run 2D SpectralConvs (matches LADD's per-tap 2D
        # head design + Projected-GAN's spatial CNN heads).
        #
        # Token layout from WAN's causal model (causal_model.py:672):
        # for each of T' frames, the per-frame chunk is
        #   [spatial_0..spatial_{H'*W'-1}, action_0..action_{a-1}]
        # so total real tokens = T' * (H'*W' + a_per_f). Beyond that
        # the sequence is zero-padded to ``seq_len``. We slice off
        # the padding AND the per-frame action tokens before the
        # 2D reshape.
        B, F_in, _C_in, H_in, W_in = x_noisy.shape
        pt, ph, pw = self.patch_size
        T_prime = F_in // pt
        H_prime = H_in // ph
        W_prime = W_in // pw
        a_per_f = int(self.action_tokens_per_frame)
        frame_seqlen = H_prime * W_prime + a_per_f
        real_tokens = T_prime * frame_seqlen

        proj_2d: Dict[int, torch.Tensor] = {}
        for idx in self.block_indices:
            feat = proj[idx]  # [B, seq_len, dim_proj]
            if feat.shape[1] < real_tokens:
                raise RuntimeError(
                    "LADDDiscriminator: captured feature length "
                    f"{feat.shape[1]} < expected real_tokens "
                    f"{real_tokens} (T'={T_prime}, H'={H_prime}, "
                    f"W'={W_prime}, a_per_f={a_per_f})."
                )
            # Strip zero-padding past the real-content region.
            feat = feat[:, :real_tokens]
            # Per-frame split + strip action tokens.
            feat = feat.reshape(B, T_prime, frame_seqlen, self.dim_proj)
            if a_per_f > 0:
                feat = feat[:, :, :H_prime * W_prime]
            # [B, T', H', W', dim_proj] -> [B*T', dim_proj, H', W']
            feat = feat.reshape(B, T_prime, H_prime, W_prime, self.dim_proj)
            feat = feat.permute(0, 1, 4, 2, 3).contiguous()
            feat = feat.reshape(B * T_prime, self.dim_proj, H_prime, W_prime)
            proj_2d[idx] = feat

        logits_per_scale = []
        for idx in self.block_indices:
            l = self.heads[str(idx)](proj_2d[idx], cmap=cmap)
            # l: [B*T', cmap_dim_or_1, H', W'] -> per-sample flat.
            l = l.reshape(B, -1)
            logits_per_scale.append(l)
        visual_logits = torch.cat(logits_per_scale, dim=1)

        # Append the parallel stat-head logit. The stat head
        # discriminates std distribution on the RAW input latent
        # (``x_noisy_raw``), bypassing the wavelet HF stage and the
        # WAN teacher. Output is a single per-sample scalar ([B, 1]).
        # The trainer splits visual / stat at the RpGAN site and
        # reduces each via its own mean — stat-side strength is
        # controlled by ``ladd_stat_head_loss_weight`` (not by token-
        # axis broadcasting, which would only fight per-token visual
        # dilution).
        if self.stat_head is not None:
            stat_logit = self.stat_head(x_noisy_raw)
            visual_logits = torch.cat([visual_logits, stat_logit], dim=1)
        return visual_logits


# ============================================================================
# DiffAugment-equivalent on latents
# ============================================================================


def _rand_flip_lr(
    x_real: torch.Tensor, x_fake: torch.Tensor, generator: Optional[torch.Generator],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Horizontal flip with 50% probability per sample. Same flip applied
    to both real and fake."""
    B = x_real.shape[0]
    flip_mask = torch.rand(B, generator=generator, device=x_real.device) > 0.5
    if not flip_mask.any():
        return x_real, x_fake
    # Build flipped versions and select per-sample.
    real_flipped = torch.flip(x_real, dims=[-1])
    fake_flipped = torch.flip(x_fake, dims=[-1])
    sel = flip_mask.view(B, *([1] * (x_real.dim() - 1)))
    return (
        torch.where(sel, real_flipped, x_real),
        torch.where(sel, fake_flipped, x_fake),
    )


def _rand_translation(
    x_real: torch.Tensor,
    x_fake: torch.Tensor,
    generator: Optional[torch.Generator],
    max_shift: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random spatial shift of [-max_shift, +max_shift] latent pixels.
    Reflect-padded. Same shift applied to both real and fake.
    """
    if max_shift <= 0:
        return x_real, x_fake
    B = x_real.shape[0]
    # One shift per sample. Use the same shift for both H and W.
    sh = torch.randint(
        -max_shift, max_shift + 1, (B,),
        generator=generator, device=x_real.device,
    )
    sw = torch.randint(
        -max_shift, max_shift + 1, (B,),
        generator=generator, device=x_real.device,
    )
    # Pad both spatial dims by max_shift on each side (reflect), then
    # per-sample slice. Simpler: build via roll (circular). Reflect
    # avoids boundary artifacts so prefer pad + slice. For perf we just
    # do roll — small max_shift, the wrap-around contribution is
    # spatially small relative to the latent grid.
    real_out = x_real.clone()
    fake_out = x_fake.clone()
    for b in range(B):
        if sh[b].item() == 0 and sw[b].item() == 0:
            continue
        real_out[b] = torch.roll(
            x_real[b], shifts=(int(sh[b]), int(sw[b])), dims=(-2, -1),
        )
        fake_out[b] = torch.roll(
            x_fake[b], shifts=(int(sh[b]), int(sw[b])), dims=(-2, -1),
        )
    return real_out, fake_out


def _rand_cutout(
    x_real: torch.Tensor,
    x_fake: torch.Tensor,
    generator: Optional[torch.Generator],
    ratio: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random rectangular cutout (zeroed region) in (F, H, W) space.
    Same cutout applied to both real and fake.
    """
    if ratio <= 0.0:
        return x_real, x_fake
    B = x_real.shape[0]
    *_, F_, H, W = x_real.shape if x_real.dim() == 5 else (None, None, *x_real.shape[-3:])
    # x can be [B, F, C, H, W] or [B, C, F, H, W]; assume [B, F, C, H, W]
    # (the conventional shape in this repo's latents).
    if x_real.dim() != 5:
        return x_real, x_fake
    _, F_dim, _, H_dim, W_dim = x_real.shape
    h_cut = max(1, int(H_dim * ratio))
    w_cut = max(1, int(W_dim * ratio))
    h0 = torch.randint(
        0, max(1, H_dim - h_cut + 1), (B,),
        generator=generator, device=x_real.device,
    )
    w0 = torch.randint(
        0, max(1, W_dim - w_cut + 1), (B,),
        generator=generator, device=x_real.device,
    )
    real_out = x_real.clone()
    fake_out = x_fake.clone()
    for b in range(B):
        h0_b, w0_b = int(h0[b]), int(w0[b])
        real_out[b, :, :, h0_b:h0_b + h_cut, w0_b:w0_b + w_cut] = 0.0
        fake_out[b, :, :, h0_b:h0_b + h_cut, w0_b:w0_b + w_cut] = 0.0
    return real_out, fake_out


def _rand_color(
    x_real: torch.Tensor,
    x_fake: torch.Tensor,
    generator: Optional[torch.Generator],
    scale_range: Tuple[float, float] = (0.9, 1.1),
    shift_std: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-channel random scale + shift. Same params applied to both
    real and fake.
    """
    B = x_real.shape[0]
    if x_real.dim() != 5:
        return x_real, x_fake
    _, _, C, _, _ = x_real.shape
    s_lo, s_hi = scale_range
    scale = (
        torch.rand(B, 1, C, 1, 1, generator=generator, device=x_real.device)
        * (s_hi - s_lo) + s_lo
    )
    shift = (
        torch.randn(B, 1, C, 1, 1, generator=generator, device=x_real.device)
        * shift_std
    )
    return x_real * scale + shift, x_fake * scale + shift


def latent_diff_augment(
    x_real: torch.Tensor,
    x_fake: torch.Tensor,
    policy: str = "flip,cutout,translation",
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply identical random augmentation to real and fake latents.

    Differentiable — gradients flow through every op back to ``x_fake``.

    Args:
        x_real: ``[B, F, C, H, W]`` real-anchor latents.
        x_fake: ``[B, F, C, H, W]`` fake / push-target latents.
        policy: comma-separated ops; subset of
            ``flip,translation,cutout,color``.
        seed: optional integer to drive the per-call RNG (for DDP
            consistency in test mode). Default ``None`` (truly random).

    Returns ``(x_real_aug, x_fake_aug)`` with identical per-sample
    augmentation.
    """
    if not policy:
        return x_real, x_fake
    if seed is not None:
        generator = torch.Generator(device=x_real.device).manual_seed(int(seed))
    else:
        generator = None
    ops = [op.strip() for op in policy.split(",") if op.strip()]
    for op in ops:
        if op == "flip":
            x_real, x_fake = _rand_flip_lr(x_real, x_fake, generator)
        elif op == "translation":
            x_real, x_fake = _rand_translation(x_real, x_fake, generator)
        elif op == "cutout":
            x_real, x_fake = _rand_cutout(x_real, x_fake, generator)
        elif op == "color":
            x_real, x_fake = _rand_color(x_real, x_fake, generator)
        else:
            raise ValueError(
                f"latent_diff_augment: unknown policy op {op!r}. Valid: "
                "flip, translation, cutout, color."
            )
    return x_real, x_fake


# ============================================================================
# Factory
# ============================================================================


def build_ladd_disc(
    real_score: nn.Module,
    block_indices: List[int],
    dim_teacher: int,
    dim_proj: int = 256,
    use_csm: bool = True,
    use_lateral_proj: bool = False,
    head_kernel_size: int = 3,
    cmap_dim: int = 0,
    prompt_embed_dim: int = 0,
    wavelet_hf_enabled: bool = False,
    wavelet_hf_in_channels: int = 16,
    wavelet_hf_drop_ll: bool = False,
    wavelet_hf_adapter_init_gain: float = 0.1,
    wavelet_hf_ll_weight: float = 0.15,
    patch_size: Tuple[int, int, int] = (1, 2, 2),
    action_tokens_per_frame: int = 0,
    stat_head_enabled: bool = False,
    stat_head_frames_per_window: int = 3,
    stat_head_pool_size: int = 4,
    stat_head_hidden_dim: int = 256,
) -> LADDDiscriminator:
    """Build a LADD discriminator wired to the existing teacher.

    Caller is responsible for moving the returned module to the right
    device + dtype, and for DDP-wrapping it. The teacher (real_score)
    is captured by reference; no parameters are copied.

    ``patch_size`` + ``action_tokens_per_frame`` describe how the WAN
    teacher tokenises its input — needed so the heads can fold the
    captured token sequence back to a 2D (H'×W') patch grid per frame.
    """
    projector = WanFeatureProjector(
        real_score=real_score,
        block_indices=block_indices,
    )
    disc = LADDDiscriminator(
        projector=projector,
        block_indices=block_indices,
        dim_teacher=dim_teacher,
        dim_proj=dim_proj,
        use_csm=use_csm,
        use_lateral_proj=use_lateral_proj,
        head_kernel_size=head_kernel_size,
        cmap_dim=cmap_dim,
        prompt_embed_dim=prompt_embed_dim,
        wavelet_hf_enabled=wavelet_hf_enabled,
        wavelet_hf_in_channels=wavelet_hf_in_channels,
        wavelet_hf_drop_ll=wavelet_hf_drop_ll,
        wavelet_hf_adapter_init_gain=wavelet_hf_adapter_init_gain,
        wavelet_hf_ll_weight=wavelet_hf_ll_weight,
        patch_size=patch_size,
        action_tokens_per_frame=action_tokens_per_frame,
        stat_head_enabled=stat_head_enabled,
        stat_head_frames_per_window=stat_head_frames_per_window,
        stat_head_pool_size=stat_head_pool_size,
        stat_head_hidden_dim=stat_head_hidden_dim,
    )
    return disc


# ============================================================================
# Moment-GAN discriminator (distribution-matching alternative to anti-collapse
# MSE). Operates on per-frame std (+ optional mean / RMS) of a [B, F, C, H, W]
# latent tensor, producing one logit per frame. Trained with the same RpGAN +
# R1 recipe as the wavelet LADD disc; the G-side gradient pulls the student's
# per-frame moment *distribution* toward GT's, instead of pulling per-frame
# values to point-wise GT targets (which ``latent_std_mse_loss`` does).
# ============================================================================


class MomentDiscriminator(nn.Module):
    """Distribution-matching discriminator on per-frame latent moments.

    The conventional anti-collapse loss (``latent_std_mse_loss``) is a
    *point-wise* match — for each frame, pull ``s_pred`` toward ``s_gt``.
    That's a strong signal but disallows natural per-frame variance in
    the student. A discriminator instead matches the *distribution* of
    per-frame moments between real (GT) and fake (student) — the student
    is free to find its own per-frame moments as long as the marginal
    distribution matches.

    Architecture: per-frame MLP with spectral-norm on linear layers.
    Frames are processed independently (no temporal mixing) so the
    disc learns a frame-level marginal decision. Input is a per-frame
    moment vector of size ``C * n_moments``; output is one logit per
    frame.

    When ``clip_logit_enabled=True`` an auxiliary clip-level head is
    added that pools moments over the F dimension (concat of mean
    and stdev across frames -> ``2 * C * n_moments`` features) and
    emits ONE scalar logit per sample, concatenated onto the per-frame
    logits to give a ``[B, F+1]`` output. This captures scene-level
    statistics (e.g. "bright clip vs dark clip" std distributions)
    that per-frame independence by itself cannot reach. RpGAN softplus
    mean is elementwise so the extra logit per sample blends in
    cleanly with the per-frame logits — no separate weighting.

    Args:
        in_channels: latent channel count C (16 for Wan VAE).
        hidden_dim: MLP hidden dim. Default 128.
        num_blocks: number of (Linear+LeakyReLU) blocks in the trunk.
            Default 2 (i.e. 3 linears total counting the final logit head).
        include_mean: include per-frame mean as an input moment.
        include_rms: include per-frame RMS as an input moment.
        clip_logit_enabled: when True, append a clip-level logit head
            that sees pooled (mean + stdev across F) per-clip moments
            and produces one extra logit per sample. Forward returns
            ``[B, F+1]`` instead of ``[B, F]``.
        clip_hidden_dim: hidden dim for the clip-level head. Default
            128 (same as ``hidden_dim``). Only consulted when
            ``clip_logit_enabled=True``.

    Input/output shapes:
        forward: ``[B, F, C, H, W]`` -> ``[B, F]`` (or ``[B, F+1]``
            when ``clip_logit_enabled=True``).
    """

    def __init__(
        self,
        in_channels: int = 16,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        include_mean: bool = True,
        include_rms: bool = True,
        clip_logit_enabled: bool = False,
        clip_hidden_dim: int = 128,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = max(1, int(num_blocks))
        self.include_mean = bool(include_mean)
        self.include_rms = bool(include_rms)
        self.clip_logit_enabled = bool(clip_logit_enabled)
        self.clip_hidden_dim = int(clip_hidden_dim)

        n_moments = 1 + int(self.include_mean) + int(self.include_rms)
        self._n_moments = int(n_moments)
        in_dim = self.in_channels * n_moments

        layers: List[nn.Module] = []
        prev = in_dim
        for _ in range(self.num_blocks):
            layers.append(spectral_norm(nn.Linear(prev, self.hidden_dim)))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev = self.hidden_dim
        layers.append(spectral_norm(nn.Linear(prev, 1)))
        self.mlp = nn.Sequential(*layers)

        # Clip-level head: input is (mean over F) ++ (stdev over F) of
        # per-frame moment vectors. Stdev-across-F captures temporal
        # spread within a clip (e.g. flicker), mean-across-F captures
        # the clip's central moment magnitude. Together: enough signal
        # for the head to distinguish bright/static-std clips from
        # dynamic/dim ones.
        if self.clip_logit_enabled:
            clip_in_dim = 2 * in_dim
            clip_layers: List[nn.Module] = []
            prev_c = clip_in_dim
            for _ in range(self.num_blocks):
                clip_layers.append(
                    spectral_norm(nn.Linear(prev_c, self.clip_hidden_dim))
                )
                clip_layers.append(nn.LeakyReLU(0.2, inplace=True))
                prev_c = self.clip_hidden_dim
            clip_layers.append(
                spectral_norm(nn.Linear(prev_c, 1))
            )
            self.clip_mlp = nn.Sequential(*clip_layers)
        else:
            self.clip_mlp = None

    @staticmethod
    def extract_moments(
        x: torch.Tensor,
        include_mean: bool,
        include_rms: bool,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Per-frame moments from ``[B, F, C, H, W]`` -> ``[B, F, C*k]``.

        Differentiable w.r.t. ``x`` through the std/mean/RMS reductions,
        so the gen-side gradient back-propagates from logits to every
        pixel of the input latent. ``unbiased=False`` matches the
        standard moment convention used elsewhere in ``anti_collapse``.
        """
        reduce_dims = [3, 4]
        s = x.std(dim=reduce_dims, unbiased=False)  # [B, F, C]
        parts = [s]
        if include_mean:
            parts.append(x.mean(dim=reduce_dims))  # [B, F, C]
        if include_rms:
            rms = torch.sqrt((x ** 2).mean(dim=reduce_dims) + eps)
            parts.append(rms)
        return torch.cat(parts, dim=-1)  # [B, F, C * n_moments]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(
                f"MomentDiscriminator expects [B, F, C, H, W]; got "
                f"{tuple(x.shape)}"
            )
        moments = self.extract_moments(
            x, self.include_mean, self.include_rms,
        )  # [B, F, C*k]
        B, F_, D = moments.shape
        # Match MLP weight dtype (the disc lives in fp32 for R1 stability;
        # ``pred_image`` may arrive as bf16 from the student forward).
        in_dtype = self.mlp[0].weight.dtype
        moments_cast = moments.to(in_dtype)
        flat = moments_cast.reshape(B * F_, -1)
        per_frame_logits = self.mlp(flat).reshape(B, F_)
        if self.clip_mlp is None:
            return per_frame_logits
        # Clip-level: pool moments over F via (mean, stdev) — the
        # stdev term gives the head a flicker / temporal-spread signal
        # that the per-frame head cannot see by construction. F may be
        # 1 (single-frame chunks) — in that case stdev is zero, which
        # is benign (the head simply gets less signal that step).
        clip_mean = moments_cast.mean(dim=1)
        if F_ > 1:
            clip_std = moments_cast.std(dim=1, unbiased=False)
        else:
            clip_std = torch.zeros_like(clip_mean)
        clip_feat = torch.cat([clip_mean, clip_std], dim=1)  # [B, 2D]
        clip_logit = self.clip_mlp(clip_feat).reshape(B, 1)  # [B, 1]
        return torch.cat([per_frame_logits, clip_logit], dim=1)  # [B, F+1]


def build_moment_disc(
    in_channels: int = 16,
    hidden_dim: int = 128,
    num_blocks: int = 2,
    include_mean: bool = True,
    include_rms: bool = True,
    clip_logit_enabled: bool = False,
    clip_hidden_dim: int = 128,
) -> MomentDiscriminator:
    """Builder for ``MomentDiscriminator``. Mirrors ``build_ladd_disc``'s
    pattern so the trainer can pick one or the other (or both)."""
    return MomentDiscriminator(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        include_mean=include_mean,
        include_rms=include_rms,
        clip_logit_enabled=clip_logit_enabled,
        clip_hidden_dim=clip_hidden_dim,
    )
