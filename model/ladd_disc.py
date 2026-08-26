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
    1. Build adjacent-chunk pairs from the gen-side rollout output.
       WHICH tensor that is, is selected by ``ladd_fake_sample_source``
       (docs/ONE_FORCING_PORT.md divergence 3):
         "flash" (default) -> ``flash_dmd_gan_x0``, the x0 of a
             dedicated extra generator forward at ``flash_dmd_gan_t``;
             falls back to ``pred_image`` (the rolled chunk) when
             flash-DMD is off. A DIFFERENT sub-graph from the one DMD
             scores.
         "dmd" -> ``score_image[:, band]``, i.e. literally the tensor
             ``compute_distribution_matching_loss`` was handed,
             restricted to the frames its ``gradient_mask`` is True on,
             so the adversarial and distillation gradients travel ONE
             sub-graph into the generator. Published by
             ``ActionForcingDMD._publish_ladd_dmd_band``.
    2. Re-noise each chunk member at the FAKE SAMPLE'S OWN generation
       timestep -- ``flash_dmd_gan_t`` on the flash path, the band's
       exit rung (``denoised_timestep_from``) on the dmd path -- or at
       t=0 when neither is defined. ``ladd_disc_force_clean`` /
       wavelet-HF / ``ladd_disc_sample_t`` override it; the value
       actually used is logged as ``train/ladd_disc_t``.
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
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint  # noqa: F401  (explicit: not implied by `import torch`)
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
            Ignored entirely when ``backbone`` is given.
        block_indices: list of ``transformer_blocks`` indices to hook.
        backbone: optional RAW ``WanModel`` to tap INSTEAD of
            ``real_score`` (WP-14B: the frozen Wan2.1-T2V-14B prefix
            built by ``model/wan14b_prefix.py``). Passing the raw model
            deliberately bypasses ``WanDiffusionWrapper``: the wrapper
            consumes the model output (``utils/wan_wrapper.py:749-753``
            unpacks ``flow_pred`` -> x0), which a truncated prefix
            cannot produce, and it forces its own padded
            ``self.seq_len`` token budget (18721) that would quadruple
            the cost of a 3-frame disc chunk at dim=5120. In this mode
            the projector calls the model directly with ``seq_len`` set
            to the chunk's ACTUAL token count.
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
        backbone: Optional[nn.Module] = None,
    ):
        self.real_score = real_score
        # WP-14B: raw-WanModel backbone mode. ``None`` => legacy
        # ``real_score``-wrapper mode (byte-identical to before).
        self.backbone = backbone
        self._raw_cond_extra_warned = False
        # Does the raw backbone support the feature-tap early exit
        # (``wan/modules/model.py`` ``max_block``)? Without it the
        # forward would run ``head``, which cannot consume the per-frame
        # timestep embedding the blocks use.
        self._backbone_supports_max_block = False
        if backbone is not None:
            import inspect
            _fwd = getattr(backbone, "_forward", None) or backbone.forward
            try:
                self._backbone_supports_max_block = (
                    "max_block" in inspect.signature(_fwd).parameters
                )
            except (TypeError, ValueError):
                self._backbone_supports_max_block = False
            if not self._backbone_supports_max_block:
                # Raised HERE, at construction, rather than at the first
                # disc forward: by then ~7 GB/rank is resident, training
                # has started, and the error surfaces from inside a
                # checkpointed recompute closure buried in torch's replay
                # machinery. The condition is fully known now.
                raise RuntimeError(
                    "WanFeatureProjector: backbone "
                    f"{type(backbone).__name__} does not accept a "
                    "``max_block`` argument, so the feature-tap early exit "
                    "cannot be plumbed. Every block above the deepest tap "
                    "would run, plus the head — dead compute that does NOT "
                    "reliably crash (``Head.forward`` broadcasts, so B=1 "
                    "completes silently and only B>1 raises). Pass a raw "
                    "``wan.modules.model.WanModel`` (model/wan14b_prefix.py)."
                )
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

        In raw-backbone mode the blocks are simply ``backbone.blocks``
        — the >=10 threshold must NOT apply there, since a tap-truncated
        14B prefix has as few as ``max(taps) + 1`` blocks.
        """
        if self.backbone is not None:
            for attr in ("blocks", "transformer_blocks"):
                b = getattr(self.backbone, attr, None)
                if isinstance(b, nn.ModuleList) and len(b) > 0:
                    return b
            raise AttributeError(
                "WanFeatureProjector: ``backbone`` has no non-empty "
                "``blocks``/``transformer_blocks`` ModuleList "
                f"(type={type(self.backbone).__name__})."
            )
        # ladd_feature_source="fake" (One-Forcing style): the model
        # installs a backbone-override handle on real_score; when
        # present, the hooks must be installed on the OVERRIDE's
        # transformer blocks (the trainable fake score), not the frozen
        # teacher's. See _LaddFakeFeatureBackbone in
        # model/dmd_action_forcing.py.
        _override = getattr(
            self.real_score, "_ladd_feature_backbone_override", None
        )
        _root = (
            _override.module() if _override is not None else self.real_score
        )
        for candidate in [_root] + list(_root.modules()):
            for attr in ("transformer_blocks", "blocks"):
                b = getattr(candidate, attr, None)
                if isinstance(b, nn.ModuleList) and len(b) >= 10:
                    return b
        raise AttributeError(
            "WanFeatureProjector: could not find a transformer-block "
            f"ModuleList anywhere under {type(_root).__name__}. "
            "Expected attribute ``transformer_blocks`` or ``blocks`` on "
            "some submodule with >=10 entries."
        )

    def _raw_backbone_seq_len(self, x_noisy: torch.Tensor) -> int:
        """Token count of ``x_noisy`` under the raw backbone's patchifier.

        This is the ACTUAL disc-chunk token count
        ``T' * H' * W' = (F/pt) * (H/ph) * (W/pw)`` — never the
        wrapper's padded budget. ``WanModel._forward`` pads the patch
        sequence up to ``seq_len`` (``wan/modules/model.py:735-741``)
        and every block then runs at that length, so handing it the
        wrapper's 18721 would burn ~4x the attention/FFN cost of a
        3-frame chunk at dim=5120 on pure zero padding.

        No action tokens are involved: the raw T2V backbone has no
        action-token plumbing, which is exactly why the disc must be
        built with ``action_tokens_per_frame=0`` in this mode.
        """
        bk = self.backbone
        _ps_attr = getattr(bk, "patch_size", None)
        if _ps_attr is None:
            # Defaulting to Wan's (1, 2, 2) would silently produce a token
            # count for a patchifier the backbone may not have — the same
            # vacuous-check trap already closed for ``in_dim`` below.
            raise AttributeError(
                "WanFeatureProjector(raw backbone): backbone "
                f"{type(bk).__name__} has no ``patch_size``; refusing to "
                "guess the tokenisation."
            )
        ps = tuple(int(p) for p in _ps_attr)
        if x_noisy.dim() != 5:
            raise ValueError(
                "WanFeatureProjector(raw backbone): expected x_noisy "
                f"[B, F, C, H, W]; got shape {tuple(x_noisy.shape)}."
            )
        _B, F_in, C_in, H_in, W_in = x_noisy.shape
        in_dim = getattr(bk, "in_dim", None)
        if in_dim is None:
            raise AttributeError(
                "WanFeatureProjector(raw backbone): backbone has no "
                "``in_dim``; cannot validate the latent channel count. "
                "(Defaulting it to the input's own C would make this "
                "check vacuous.)"
            )
        in_dim = int(in_dim)
        if C_in != in_dim:
            raise ValueError(
                "WanFeatureProjector(raw backbone): latent channel count "
                f"{C_in} != backbone in_dim {in_dim}."
            )
        for name, size, patch in (
            ("F", F_in, ps[0]), ("H", H_in, ps[1]), ("W", W_in, ps[2]),
        ):
            if size % patch != 0:
                raise ValueError(
                    "WanFeatureProjector(raw backbone): input "
                    f"{name}={size} is not divisible by patch {patch} "
                    f"(patch_size={ps})."
                )
        return (F_in // ps[0]) * (H_in // ps[1]) * (W_in // ps[2])

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
        # ladd_feature_source="fake" (One-Forcing style): backbone
        # override handle installed on real_score by ActionForcingDMD.
        # When present, the feature forward runs through the TRAINABLE
        # fake score (with the override's own freeze/DDP-bypass guard)
        # instead of the frozen teacher. Mutually exclusive with the
        # WP-14B raw ``backbone`` — both replace the feature backbone.
        feature_override = getattr(
            self.real_score, "_ladd_feature_backbone_override", None
        )
        if feature_override is not None and self.backbone is not None:
            raise ValueError(
                "WanFeatureProjector: both a raw disc backbone "
                "(ladd_disc_backbone_model_name) and a "
                "_ladd_feature_backbone_override (ladd_feature_source="
                "'fake') are set. They are mutually exclusive."
            )
        teacher_module = (
            self.backbone
            if self.backbone is not None
            else (
                feature_override.module()
                if feature_override is not None
                else self.real_score
            )
        )
        teacher_dtype = next(teacher_module.parameters()).dtype

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

        # ---- raw-WanModel backbone (WP-14B) argument prep ----------
        raw_ctx = None
        raw_t = None
        raw_seq_len = None
        if self.backbone is not None:
            if clean_x is not None or aug_t is not None:
                raise ValueError(
                    "WanFeatureProjector(raw backbone): ``clean_x`` / "
                    "``aug_t`` are teacher-forcing arguments of "
                    "``WanDiffusionWrapper`` and have no equivalent on a "
                    "raw WanModel. Silently dropping them would change "
                    "the disc's conditioning, so this is fail-loud."
                )
            raw_seq_len = self._raw_backbone_seq_len(x_noisy_cast)
            if seq_len is not None and int(seq_len) != raw_seq_len:
                raise ValueError(
                    "WanFeatureProjector(raw backbone): caller passed "
                    f"seq_len={int(seq_len)} but the chunk's actual token "
                    f"count is {raw_seq_len}. The raw backbone must run at "
                    "the actual token count (see ``_raw_backbone_seq_len``)."
                )
            if conditional_extra and not self._raw_cond_extra_warned:
                # Stock T2V has no action-token plumbing. This is the
                # documented distribution shift of the 14B arm, not a
                # silent bug — say it once, loudly, then proceed.
                logging.warning(
                    "[LADD/14B] raw backbone ignores conditional_extra "
                    "keys %s (no action-token support in stock Wan T2V). "
                    "The disc MUST be built with "
                    "action_tokens_per_frame=0 in this mode.",
                    sorted(conditional_extra.keys()),
                )
                self._raw_cond_extra_warned = True
            raw_ctx = cond_dict["prompt_embeds"]
            if raw_ctx is None or raw_ctx.dim() != 3:
                raise ValueError(
                    "WanFeatureProjector(raw backbone): prompt_embeds must "
                    f"be [B, L, text_dim]; got "
                    f"{None if raw_ctx is None else tuple(raw_ctx.shape)}."
                )
            _text_len = int(getattr(self.backbone, "text_len", 512))
            if raw_ctx.shape[1] > _text_len:
                raise ValueError(
                    "WanFeatureProjector(raw backbone): prompt length "
                    f"{raw_ctx.shape[1]} exceeds the backbone's text_len "
                    f"{_text_len}."
                )
            _text_dim = int(getattr(self.backbone, "text_dim", 4096))
            if raw_ctx.shape[2] != _text_dim:
                raise ValueError(
                    "WanFeatureProjector(raw backbone): prompt embed dim "
                    f"{raw_ctx.shape[2]} != backbone text_dim {_text_dim}."
                )
            # Per-frame timesteps: WanModel derives the modulation frame
            # count from ``t.shape`` (e0 is unflattened to [B, T', 6, dim]
            # at ``wan/modules/model.py:742-744``), so t must be [B, T'].
            _pt = int(tuple(getattr(self.backbone, "patch_size", (1, 2, 2)))[0])
            _T_prime = x_noisy_cast.shape[1] // _pt
            raw_t = timestep
            if raw_t.dim() == 1:
                raw_t = raw_t[:, None].expand(-1, _T_prime)
            if tuple(raw_t.shape) != (x_noisy_cast.shape[0], _T_prime):
                raise ValueError(
                    "WanFeatureProjector(raw backbone): timestep shape "
                    f"{tuple(raw_t.shape)} != expected "
                    f"({x_noisy_cast.shape[0]}, {_T_prime})."
                )

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
        blocks = self._find_blocks()

        def _make_local_hook(block_idx: int):
            def _local_hook(_m, _i, out):
                feat = out[0] if isinstance(out, tuple) else out
                local_feats[block_idx] = feat
            return _local_hook

        def _run_teacher(x):
            # Hooks are installed INSIDE the checkpointed function, not
            # around it. ``checkpoint(use_reentrant=False)`` re-executes
            # this body during BACKWARD; with the hooks installed
            # outside and removed in an outer ``finally``, the replay
            # would find ``local_feats`` empty and raise KeyError on the
            # return line. Today it survives only because autograd stops
            # the replay early (``_StopRecomputationError``) a few ops
            # before that line — and the ``max_block`` early exit
            # shortens the region, moving the stop point closer to it.
            # Installing per-call makes the replay correct on its own
            # terms instead of relying on where the stop lands.
            hs = [
                blocks[i].register_forward_hook(_make_local_hook(i))
                for i in block_indices_sorted
            ]
            try:
                return _teacher_body(x)
            finally:
                for h in hs:
                    h.remove()

        def _teacher_body(x):
            if self.backbone is not None:
                # Raw WanModel: [B, F, C, H, W] -> [B, C, F, H, W], the
                # layout ``WanDiffusionWrapper`` also permutes into
                # (utils/wan_wrapper.py:703). The return value is
                # DISCARDED — only the hooked block outputs matter, which
                # is why a truncated prefix (random/zeroed head) is fine.
                raw_kwargs = {"max_block": max(block_indices_sorted)}
                _ = self.backbone(
                    x.permute(0, 2, 1, 3, 4),
                    t=raw_t,
                    context=raw_ctx,
                    seq_len=raw_seq_len,
                    **raw_kwargs,
                )
            else:
                kwargs_local = dict(kwargs)
                kwargs_local["noisy_image_or_video"] = x
                if feature_override is not None:
                    # One-Forcing-style fake-score backbone. The
                    # override freezes every fake-score param (restores
                    # the recorded prior flags after) and bypasses the
                    # DDP wrap for THIS forward only. It MUST execute
                    # here, inside the checkpointed ``_run_teacher``
                    # body: the use_reentrant=False replay during the
                    # GAN backward re-runs the freeze, so the
                    # recomputed graph also carries no edges into the
                    # fake-score params. Same call signature as the
                    # real_score branch — only the backbone swaps.
                    _ = feature_override.forward(**kwargs_local)
                else:
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
            # fp32 for downstream R1 stability.
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
# Register-token cross-attention readout (One-Forcing divergence 2)
# ============================================================================


def _gan_cross_attention_sdpa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
) -> torch.Tensor:
    """SDPA stand-in for ``wan.modules.attention.flash_attention``.

    Same contract: ``[B, L, N, C]`` in and out. This is exactly the
    fallback branch of ``wan.modules.attention.attention`` minus its
    unconditional bf16 cast.

    Why it exists rather than calling ``flash_attention`` directly:
    ``flash_attention`` asserts ``q.device.type == 'cuda'`` and asserts a
    half dtype, and the LADD discriminator is built **fp32** (for R1
    stability — see ``trainer/causal_action_forcing_train.py``'s
    ``disc.to(dtype=torch.float32)``) and is unit-tested on CPU. The
    register query is a SINGLE token, so this is a [1 x L] attention row
    per head — the flash kernel buys nothing here anyway.
    """
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
    )
    return out.transpose(1, 2).contiguous()


class LADDRegisterReadout(nn.Module):
    """One-Forcing's learned register-token cross-attention pooling.

    ``docs/ONE_FORCING_PORT.md`` divergence 2: ours mean-pools/convolves
    projected tap features (CCM -> CSM -> SpectralConv heads); theirs
    pools each tap with a LEARNED register token that cross-attends over
    that tap's whole token sequence, concatenates the pooled tokens and
    runs one MLP to a scalar.

    This class is a thin re-host of the machinery that already ships in
    this tree for the ``classify_mode`` path — ``RegisterTokens`` and
    ``GanAttentionBlock`` from ``wan/modules/model.py`` and the head
    builder factored out of ``WanDiffusionWrapper.adding_cls_branch`` —
    executed on the features the ``WanFeatureProjector`` already
    captured, in the same order as ``wan/modules/model.py``'s
    ``classify_mode`` block (tap i <-> register token i, IN ORDER; concat
    on the token axis; flatten; MLP).

    WHY re-host instead of calling ``classify_mode``: parameter
    ownership. ``adding_cls_branch`` attaches its modules to a
    ``WanDiffusionWrapper`` or to that wrapper's ``WanModel``. The LADD
    disc taps ``real_score`` (frozen, and NOT in any optimizer) or —
    under ``ladd_feature_source="fake"`` — the ``fake_score`` backbone,
    whose parameters belong to ``fake_optimizer``. Attaching a
    discriminator head to either would either never be stepped or be
    stepped by the WRONG optimizer. As a submodule of
    ``LADDDiscriminator`` the head lands in ``disc.parameters()``, which
    is exactly what ``r3gan_optimizer`` is built from and what
    ``r3gan_disc_ddp`` wraps.

    Output: ``[B, num_class]`` — ONE logit per row at the default
    ``num_class=1``. That is the same shape ``ladd_scalar_output=True``
    produces, so the RpGAN/relativistic reductions and the R1
    finite-difference estimator (which sums over the logit axis) are
    unchanged in meaning: with a single column, sum == mean.
    """

    def __init__(
        self,
        block_indices: List[int],
        dim_teacher: int,
        blocks_per_token: int = 2,
        block_ffn_dim: int = 8192,
        block_num_heads: int = 12,
        head_hidden_dim: int = 3072,
        head_num_layers: int = 4,
        head_dropout: float = 0.2,
        num_class: int = 1,
        use_checkpoint: Optional[bool] = None,
    ):
        super().__init__()
        # Lazy imports: ``model/ladd_disc.py`` is imported by lightweight
        # CPU tests, and ``utils.wan_wrapper`` drags in T5/VAE.
        from wan.modules.model import RegisterTokens, GanAttentionBlock
        from utils.wan_wrapper import build_cls_pred_branch

        self.block_indices = sorted(set(int(i) for i in block_indices))
        if not self.block_indices:
            raise ValueError(
                "LADDRegisterReadout: block_indices is empty; the register "
                "head needs at least one tap."
            )
        self.dim_teacher = int(dim_teacher)
        self.blocks_per_token = max(1, int(blocks_per_token))
        self.block_ffn_dim = int(block_ffn_dim)
        self.block_num_heads = int(block_num_heads)
        self.head_hidden_dim = int(head_hidden_dim)
        self.head_num_layers = int(head_num_layers)
        self.head_dropout = float(head_dropout)
        self.num_class = int(num_class)
        # ------------------------------------------------------------------
        # ladd_register_checkpoint — activation-checkpoint the per-tap
        # GanAttentionBlock stacks (torch.utils.checkpoint,
        # use_reentrant=False). WHY: each block cross-attends a single
        # register token over the FULL real-token sequence, and this runs
        # OUTSIDE the projector's checkpointed ``_run_teacher`` body, so
        # every big [B, L_real, dim] intermediate (fp32 tap cast, norm3
        # out, k(x), RMSNorm product, kk, vv — ~6 x B*L*dim*4 bytes per
        # tap per block) is RETAINED until the consumer loss's backward.
        # At the 6-frame-pair smoke geometry (L_real=9360, dim=1536, 5
        # taps, ~9-12 rows per disc micro-group) that is ~1.6 GiB/row =
        # ~15-19 GiB per D-update micro-group — the marginal straw that
        # OOM'd carntx6all-on at 92.66 GiB. Checkpointing keeps only the
        # inputs (the bf16 tap features, which are alive anyway as
        # projector outputs, + the tiny register token) and recomputes
        # the blocks at backward time (~1 extra readout forward per
        # group — negligible next to the 1.3B backbone's own inner-ckpt
        # recompute).
        #
        # SAFETY (record/replay invariant — see
        # model/dmd_action_forcing.py:_LaddFakeFeatureBackbone): the
        # checkpoint only ARMS when this module's own params require
        # grad AND grad mode is on (see ``forward``). That is exactly
        # the D-update window, where the disc's flags are pinned across
        # each group's forward+backward. The gen-side guidance forward
        # and the aux-teacher disc forward both run under
        # ``disc.requires_grad_(False)`` with a ``finally`` that
        # restores True BEFORE their backward — a flag flip across
        # record/replay that would change the recompute's saved-tensor
        # list and raise CheckpointError; the params-require-grad guard
        # makes those paths take the plain (verbatim) branch instead.
        # The blocks are RNG-free (no dropout — ``head_dropout`` lives
        # in ``cls_pred_branch``, OUTSIDE the checkpointed region) and
        # state-free (no spectral_norm / BN), so the backward-time
        # replay is bit-identical to the recorded forward and cannot
        # double-mutate module state.
        #
        # Default TRUE (strictly memory-better, gradient-identical —
        # proven bit-equal outputs/input-grads/param-grads on CPU).
        # Override per-run without trainer plumbing via env
        # LADD_REGISTER_CHECKPOINT=0/1; the constructor arg (when not
        # None) wins over the env var.
        if use_checkpoint is None:
            _env = os.environ.get("LADD_REGISTER_CHECKPOINT")
            if _env is None:
                use_checkpoint = True
            else:
                use_checkpoint = _env.strip().lower() not in (
                    "0", "false", "no", "off", "")
        self.use_checkpoint = bool(use_checkpoint)
        n_reg = len(self.block_indices)

        self.register_tokens = RegisterTokens(
            num_registers=n_reg, dim=self.dim_teacher,
        )
        self.gan_ca_blocks = nn.ModuleList([
            nn.ModuleList([
                GanAttentionBlock(
                    dim=self.dim_teacher,
                    ffn_dim=self.block_ffn_dim,
                    num_heads=self.block_num_heads,
                )
                for _ in range(self.blocks_per_token)
            ])
            for _ in range(n_reg)
        ])
        self.cls_pred_branch = build_cls_pred_branch(
            input_dim=n_reg * self.dim_teacher,
            hidden_dim=self.head_hidden_dim,
            num_class=self.num_class,
            num_layers=self.head_num_layers,
            dropout=self.head_dropout,
        )
        # fp32/CPU-capable attention kernel for THESE blocks only (see
        # ``_gan_cross_attention_sdpa``). Per-instance attribute: the
        # ``classify_mode`` blocks built by ``adding_cls_branch`` never
        # see it and keep calling ``flash_attention``.
        for _stack in self.gan_ca_blocks:
            for _blk in _stack:
                _blk.cross_attn._attn_impl = _gan_cross_attention_sdpa

    # ------------------------------------------------------------------
    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def describe(self) -> str:
        return (
            "taps=%s dim=%d blocks_per_token=%d ffn=%d heads=%d "
            "head_hidden=%d head_layers=%d dropout=%.3g params=%.2fM "
            "ckpt=%s" % (
                self.block_indices, self.dim_teacher, self.blocks_per_token,
                self.block_ffn_dim, self.block_num_heads,
                self.head_hidden_dim, self.head_num_layers, self.head_dropout,
                self.num_params / 1e6, self.use_checkpoint,
            )
        )

    # ------------------------------------------------------------------
    def _tap_forward(
        self, tap_pos: int, feat: torch.Tensor, token: torch.Tensor,
    ) -> torch.Tensor:
        """Per-tap pipeline: dtype cast + the tap's GanAttentionBlock
        stack. Exactly the op sequence the pre-checkpoint inline loop
        ran, factored out so ``forward`` can route it either directly
        (verbatim behaviour) or through ``torch.utils.checkpoint``.

        The cast lives INSIDE this function on purpose: under
        checkpointing only the function INPUTS are saved, so the fp32
        copy of the (bf16) tap features is recomputed at backward time
        instead of being retained — that copy alone is B*L_real*dim*4
        bytes per tap. RNG-free and state-free (see the
        ``use_checkpoint`` note in ``__init__``), so record and replay
        are bit-identical.
        """
        if feat.dtype != token.dtype:
            # Tap features arrive in the backbone's dtype (bf16); the head
            # is fp32. Same boundary the ``classify_mode`` loop casts at
            # (``wan/modules/model.py`` ``_tap_x``) — a real cast node, so
            # the G-side gradient converts back at exactly this edge.
            feat = feat.to(token.dtype)
        for blk in self.gan_ca_blocks[tap_pos]:
            token = blk(feat, token)
        return token

    # ------------------------------------------------------------------
    def forward(
        self,
        features: Dict[int, torch.Tensor],
        real_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """``{block_idx: [B, L, dim_teacher]}`` -> ``[B, num_class]``.

        ``real_tokens`` trims the WAN wrapper's zero padding (the disc
        forward runs at the wrapper's fixed ``seq_len``, so most of ``L``
        is padding for a 3- or 9-frame chunk). The LADD readout strips
        the same region at ``LADDDiscriminator.forward``; letting the
        register token attend over tens of thousands of zero rows would
        both cost real time and hand the disc a chunk-length cue.
        Action tokens are KEPT — unlike the 2D SpectralConv heads, this
        readout has no patch-grid to fold them into.
        """
        missing = [i for i in self.block_indices if i not in features]
        if missing:
            raise RuntimeError(
                "LADDRegisterReadout: no features for block indices "
                f"{missing}; the projector's taps and the head's taps must "
                "be the same list."
            )
        toks = self.register_tokens()  # [n_reg, dim] (RMS-normed)
        # Arm the per-tap activation checkpoint ONLY inside a window
        # where this module's params take gradient (the D-update): there
        # the flags are pinned across forward+backward, so the
        # backward-time replay records the SAME saved-tensor list. The
        # frozen-disc consumers (gen-side guidance, aux-teacher disc)
        # flip ``requires_grad`` back to True in a ``finally`` BEFORE
        # their backward runs — checkpointing across that flip would
        # raise CheckpointError — so they take the verbatim direct path
        # (byte-identical to the pre-flag code). Evaluated ONCE per
        # forward so record and (potential) replay cannot disagree.
        _use_ckpt = (
            self.use_checkpoint
            and torch.is_grad_enabled()
            and any(p.requires_grad for p in self.parameters())
        )
        pooled = []
        B = None
        for i, idx in enumerate(self.block_indices):
            feat = features[idx]
            if feat.dim() != 3:
                raise ValueError(
                    "LADDRegisterReadout: expected tap features "
                    f"[B, L, dim]; got {tuple(feat.shape)} at block {idx}."
                )
            if int(feat.shape[2]) != self.dim_teacher:
                raise ValueError(
                    "LADDRegisterReadout: tap feature dim "
                    f"{int(feat.shape[2])} != dim_teacher "
                    f"{self.dim_teacher} at block {idx}."
                )
            B = int(feat.shape[0])
            if real_tokens is not None:
                rt = int(real_tokens)
                if feat.shape[1] < rt:
                    raise RuntimeError(
                        "LADDRegisterReadout: captured feature length "
                        f"{int(feat.shape[1])} < expected real_tokens {rt} "
                        f"at block {idx}."
                    )
                feat = feat[:, :rt]
            token = toks[i].reshape(1, 1, -1).expand(B, 1, -1)
            if _use_ckpt:
                # Non-reentrant: DDP-safe (find_unused_parameters=False)
                # and composes with the projector's outer checkpoint by
                # being SEQUENTIAL to it, never nested — the readout
                # consumes the projector's outputs after its checkpoint
                # scope has closed. preserve_rng_state=False is sound
                # because ``_tap_forward`` draws no RNG (no dropout in
                # the GanAttentionBlock stack).
                token = torch.utils.checkpoint.checkpoint(
                    self._tap_forward, i, feat, token,
                    use_reentrant=False, preserve_rng_state=False,
                )
            else:
                token = self._tap_forward(i, feat, token)
            pooled.append(token)
        final = torch.cat(pooled, dim=1)          # [B, n_reg, dim]
        return self.cls_pred_branch(final.reshape(B, -1))


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
        wavelet_hf_drop_hh: drop the DIAGONAL HF band (HH) -- with
            drop_ll too, the disc sees only LH/HL (directional HF).
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
        readout: ``"ladd"`` (default, byte-identical to before) or
            ``"register"`` — One-Forcing's learned register-token
            cross-attention pooling per tap
            (docs/ONE_FORCING_PORT.md divergence 2). ``"register"``
            REPLACES CCM/CSM/heads/cmapper (they are not built) and
            emits ``[B, 1]``, i.e. the same shape
            ``scalar_output=True`` produces.
        register_blocks_per_token / register_block_ffn_dim /
        register_block_num_heads / register_head_hidden_dim /
        register_head_num_layers / register_head_dropout: geometry of
            the register head. Defaults reproduce the shape
            ``adding_cls_branch`` builds today (2 blocks/token, ffn
            8192, 12 heads, hidden 3072, 4-layer residual MLP, dropout
            0.2). One-Forcing's *framewise* config is
            ``1 / 2048 / 12 / 1536 / 1 / 0.0``; set the knobs to get it.
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
        wavelet_hf_drop_hh: bool = False,
        wavelet_hf_adapter_init_gain: float = 0.1,
        wavelet_hf_ll_weight: float = 0.15,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        action_tokens_per_frame: int = 0,
        stat_head_enabled: bool = False,
        stat_head_frames_per_window: int = 3,
        stat_head_pool_size: int = 4,
        stat_head_hidden_dim: int = 256,
        scalar_output: bool = False,
        freeze_projector_mixing: bool = False,
        readout: str = "ladd",
        register_blocks_per_token: int = 2,
        register_block_ffn_dim: int = 8192,
        register_block_num_heads: int = 12,
        register_head_hidden_dim: int = 3072,
        register_head_num_layers: int = 4,
        register_head_dropout: float = 0.2,
        register_checkpoint: Optional[bool] = None,
    ):
        super().__init__()
        self.projector = projector  # stored as plain attribute, not nn submodule
        self.block_indices = sorted(set(int(i) for i in block_indices))
        self.dim_teacher = int(dim_teacher)
        self.dim_proj = int(dim_proj)
        self.use_csm = bool(use_csm)
        self.cmap_dim = int(cmap_dim)
        self.wavelet_hf_enabled = bool(wavelet_hf_enabled)
        self.scalar_output = bool(scalar_output)
        self.freeze_projector_mixing = bool(freeze_projector_mixing)
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
                drop_hh=bool(wavelet_hf_drop_hh),
            )
        else:
            self.wavelet_hf = None

        # ------------------------------------------------------------------
        # Readout (docs/ONE_FORCING_PORT.md divergence 2).
        #   "ladd"     (default) — CCM -> optional CSM -> per-tap
        #              SpectralConv heads. Byte-identical to before: the
        #              block below is the verbatim historical code.
        #   "register" — One-Forcing's learned register-token cross-attn
        #              pooling per tap (``LADDRegisterReadout``). It
        #              REPLACES CCM/CSM/heads/cmapper rather than sitting
        #              beside them; building them and not using them would
        #              leave them ungradiented and the disc's
        #              ``find_unused_parameters=False`` DDP reducer would
        #              raise "Expected to have finished reduction".
        # ------------------------------------------------------------------
        self.readout = str(readout)
        if self.readout not in ("ladd", "register"):
            raise ValueError(
                "LADDDiscriminator: ladd_readout must be 'ladd' or "
                f"'register'; got {self.readout!r}."
            )
        self.register_readout: Optional[LADDRegisterReadout] = None
        if self.readout == "register":
            if self.cmap_dim > 0:
                raise ValueError(
                    "LADDDiscriminator: ladd_readout='register' does not "
                    "consume the prompt cmap (the register head has no "
                    f"per-scale cls conv to modulate); got cmap_dim="
                    f"{self.cmap_dim}. Pass ladd_cmap_dim=0 / "
                    "ladd_use_prompt_cond=false rather than have the "
                    "cmapper built and silently never gradiented."
                )
            if self.freeze_projector_mixing:
                raise ValueError(
                    "LADDDiscriminator: ladd_freeze_projector_mixing=true "
                    "with ladd_readout='register' is inert — there is no "
                    "CCM/CSM to freeze. Refused rather than ignored."
                )
            self.ccm = None
            self.csm = None
            self.heads = None
            self.cmapper = None
            self.register_readout = LADDRegisterReadout(
                block_indices=self.block_indices,
                dim_teacher=self.dim_teacher,
                blocks_per_token=int(register_blocks_per_token),
                block_ffn_dim=int(register_block_ffn_dim),
                block_num_heads=int(register_block_num_heads),
                head_hidden_dim=int(register_head_hidden_dim),
                head_num_layers=int(register_head_num_layers),
                head_dropout=float(register_head_dropout),
                num_class=1,
                # None -> LADDRegisterReadout resolves it (env
                # LADD_REGISTER_CHECKPOINT, else default True).
                use_checkpoint=register_checkpoint,
            )
        else:
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
            if self.freeze_projector_mixing:
                self.ccm.requires_grad_(False)
                if self.csm is not None:
                    self.csm.requires_grad_(False)
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
        if self.register_readout is not None:
            # One-Forcing readout: register-token cross-attn pooling per
            # tap -> one scalar logit per row. No CCM/CSM/heads exist in
            # this mode, so the 2D patch-grid reshape below is skipped
            # entirely; the only geometry the readout needs is the real
            # (non-padding) token count, computed with the SAME arithmetic
            # the LADD branch uses.
            B, F_in, _C_in, H_in, W_in = x_noisy.shape
            pt, ph, pw = self.patch_size
            _real_tokens = (F_in // pt) * (
                (H_in // ph) * (W_in // pw) + int(self.action_tokens_per_frame)
            )
            visual_logits = self.register_readout(
                feats, real_tokens=_real_tokens,
            )
            if self.stat_head is not None:
                visual_logits = torch.cat(
                    [visual_logits, self.stat_head(x_noisy_raw)], dim=1,
                )
            return visual_logits

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
        if self.scalar_output:
            # One invariant critic value per sample. This reduction is part of
            # D itself, so D/G losses and R1 all differentiate the exact
            # same scalar instead of summing a resolution-dependent token map.
            visual_logits = visual_logits.mean(dim=1, keepdim=True)

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
    wavelet_hf_drop_hh: bool = False,
    wavelet_hf_adapter_init_gain: float = 0.1,
    wavelet_hf_ll_weight: float = 0.15,
    patch_size: Tuple[int, int, int] = (1, 2, 2),
    action_tokens_per_frame: int = 0,
    stat_head_enabled: bool = False,
    stat_head_frames_per_window: int = 3,
    stat_head_pool_size: int = 4,
    stat_head_hidden_dim: int = 256,
    scalar_output: bool = False,
    freeze_projector_mixing: bool = False,
    backbone: Optional[nn.Module] = None,
    readout: str = "ladd",
    register_blocks_per_token: int = 2,
    register_block_ffn_dim: int = 8192,
    register_block_num_heads: int = 12,
    register_head_hidden_dim: int = 3072,
    register_head_num_layers: int = 4,
    register_head_dropout: float = 0.2,
    register_checkpoint: Optional[bool] = None,
) -> LADDDiscriminator:
    """Build a LADD discriminator wired to the existing teacher.

    Caller is responsible for moving the returned module to the right
    device + dtype, and for DDP-wrapping it. The teacher (real_score)
    is captured by reference; no parameters are copied.

    ``patch_size`` + ``action_tokens_per_frame`` describe how the WAN
    teacher tokenises its input — needed so the heads can fold the
    captured token sequence back to a 2D (H'×W') patch grid per frame.

    ``backbone`` (WP-14B) swaps the tapped network for a raw
    ``WanModel`` — e.g. the frozen Wan2.1-T2V-14B prefix from
    ``model/wan14b_prefix.py`` — leaving ``real_score`` untouched. When
    it is given, the caller MUST pass that backbone's own
    ``dim_teacher`` / ``patch_size`` and ``action_tokens_per_frame=0``.
    """
    if backbone is not None:
        _blocks_attr = getattr(backbone, "blocks", None)
        _n_layers = len(_blocks_attr) if _blocks_attr is not None else 0
        if _n_layers == 0:
            raise ValueError(
                "build_ladd_disc: backbone has no non-empty ``blocks`` "
                f"ModuleList (type={type(backbone).__name__}). Pass a RAW "
                "WanModel — a DDP/compiled/wrapper object is not supported "
                "here, and would otherwise be reported as '0 blocks loaded'."
            )
        _tap_list = [int(i) for i in block_indices]
        if not _tap_list:
            raise ValueError(
                "build_ladd_disc: block_indices is empty. The backbone path "
                "requires EXPLICIT shallow taps (e.g. [0, 2, 4, 8])."
            )
        _max_tap = max(_tap_list)
        # Hard assert (GAN_REDESIGN B2 step (d)).
        if _max_tap >= _n_layers:
            raise ValueError(
                f"build_ladd_disc: deepest tap {_max_tap} >= "
                f"{_n_layers} blocks loaded in the backbone prefix. "
                "``ladd_feature_blocks`` must satisfy "
                "max(taps) < num_layers_loaded."
            )
        if int(action_tokens_per_frame) != 0:
            raise ValueError(
                "build_ladd_disc: a raw backbone has no action tokens; "
                f"got action_tokens_per_frame={action_tokens_per_frame}. "
                "Pass 0 — harvesting the value from real_score silently "
                "mis-slices the token reshape."
            )
        _bk_dim = int(getattr(backbone, "dim", dim_teacher))
        if int(dim_teacher) != _bk_dim:
            raise ValueError(
                f"build_ladd_disc: dim_teacher={dim_teacher} does not match "
                f"the backbone's hidden dim {_bk_dim}."
            )
        # patch_size must match too. A COARSER disc patch than the
        # backbone's is silent: the backbone emits more tokens than the
        # heads expect, the disc slices the first ``real_tokens`` and
        # reshapes them into a wrong grid, and the run continues on a
        # bogus spatial layout. (The opposite direction raises.)
        _bk_ps = getattr(backbone, "patch_size", None)
        if _bk_ps is None:
            raise ValueError(
                "build_ladd_disc: backbone "
                f"{type(backbone).__name__} exposes no ``patch_size``, so "
                "the token->patch-grid reshape cannot be validated. Pass a "
                "raw WanModel."
            )
        _bk_ps = tuple(int(p) for p in _bk_ps)
        if tuple(int(p) for p in patch_size) != _bk_ps:
            raise ValueError(
                f"build_ladd_disc: patch_size={tuple(patch_size)} does "
                f"not match the backbone's {_bk_ps}. A mismatch here "
                "mis-slices the token->patch-grid reshape silently."
            )
    projector = WanFeatureProjector(
        real_score=real_score,
        block_indices=block_indices,
        backbone=backbone,
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
        wavelet_hf_drop_hh=wavelet_hf_drop_hh,
        wavelet_hf_adapter_init_gain=wavelet_hf_adapter_init_gain,
        wavelet_hf_ll_weight=wavelet_hf_ll_weight,
        patch_size=patch_size,
        action_tokens_per_frame=action_tokens_per_frame,
        stat_head_enabled=stat_head_enabled,
        stat_head_frames_per_window=stat_head_frames_per_window,
        stat_head_pool_size=stat_head_pool_size,
        stat_head_hidden_dim=stat_head_hidden_dim,
        scalar_output=scalar_output,
        freeze_projector_mixing=freeze_projector_mixing,
        readout=readout,
        register_blocks_per_token=register_blocks_per_token,
        register_block_ffn_dim=register_block_ffn_dim,
        register_block_num_heads=register_block_num_heads,
        register_head_hidden_dim=register_head_hidden_dim,
        register_head_num_layers=register_head_num_layers,
        register_head_dropout=register_head_dropout,
        register_checkpoint=register_checkpoint,
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
