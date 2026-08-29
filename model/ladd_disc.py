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
        dim_teacher,
        dim_proj: int,
    ):
        super().__init__()
        self.block_indices = sorted(set(int(i) for i in block_indices))
        # ``dim_teacher`` is EITHER an int -- every WAN tap shares dim
        # 1536, the historical case, byte-identical -- OR a per-tap
        # ``{block_idx: channels}`` mapping. The mapping exists for the
        # PIXEL feature sources (``ladd_feature_source=pixgan``), whose
        # conv taps are 64 / 128 / 256 rather than one shared width. A
        # plain ViT source (``dinov2``) is uniform and still passes an
        # int, so the DINO arm does not depend on this branch either.
        if isinstance(dim_teacher, dict):
            self.dim_teacher_map = {
                int(i): int(dim_teacher[i]) for i in self.block_indices
            }
            self.dim_teacher = int(
                self.dim_teacher_map[self.block_indices[0]]
            )
        else:
            self.dim_teacher = int(dim_teacher)
            self.dim_teacher_map = {
                int(i): self.dim_teacher for i in self.block_indices
            }
        self.dim_proj = int(dim_proj)
        self.proj = nn.ModuleDict(
            {str(i): nn.Linear(self.dim_teacher_map[i], self.dim_proj)
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
        mismatch_cmap: Optional[torch.Tensor] = None,
    ):
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
            def _project(this_cmap: torch.Tensor) -> torch.Tensor:
                B_eff = out.shape[0]
                B_cmap = this_cmap.shape[0]
                if B_cmap <= 0 or B_eff % B_cmap != 0:
                    raise ValueError(
                        "LADD cmap batch must divide the folded feature "
                        f"batch; got B_eff={B_eff}, B_cmap={B_cmap}."
                    )
                t_frames = B_eff // B_cmap
                cmap_expanded = this_cmap.repeat_interleave(t_frames, dim=0)
                cmap_b = cmap_expanded.unsqueeze(-1).unsqueeze(-1)
                return (out * cmap_b).sum(1, keepdim=True) * (
                    1.0 / math.sqrt(self.cmap_dim)
                )

            projected = _project(cmap)
            if mismatch_cmap is not None:
                # One SpectralConv/head forward, two cheap projections.  This
                # keeps both conditions on the identical activation and avoids
                # a second train-mode spectral-normalization buffer mutation.
                return projected, _project(mismatch_cmap)
            return projected
        if mismatch_cmap is not None:
            raise ValueError(
                "mismatch_cmap requires a projection head with cmap_dim>0."
            )
        return out


class LADDActionConditioner(nn.Module):
    """Map a per-frame action-token sequence to one projection cmap.

    The pixel discriminator scores a short video window as one row.  A plain
    temporal mean would make opposite action schedules indistinguishable, so
    the summary retains the mean, scale, endpoints and signed endpoint delta.
    The input is conditioning-only (the trainer detaches action tokens); this
    module belongs to, and is optimized with, the discriminator.
    """

    def __init__(self, action_embed_dim: int, cmap_dim: int):
        super().__init__()
        self.action_embed_dim = int(action_embed_dim)
        self.cmap_dim = int(cmap_dim)
        if self.action_embed_dim <= 0 or self.cmap_dim <= 0:
            raise ValueError(
                "LADDActionConditioner dimensions must be positive; got "
                f"action_embed_dim={self.action_embed_dim}, "
                f"cmap_dim={self.cmap_dim}."
            )
        self.norm = nn.LayerNorm(self.action_embed_dim)
        self.frame_proj = nn.Linear(self.action_embed_dim, self.cmap_dim)
        self.summary = nn.Sequential(
            nn.Linear(5 * self.cmap_dim, 2 * self.cmap_dim),
            nn.SiLU(),
            nn.Linear(2 * self.cmap_dim, self.cmap_dim),
        )

    def forward(self, action_tokens: torch.Tensor) -> torch.Tensor:
        if action_tokens.dim() != 3:
            raise ValueError(
                "LADD action conditioning expects action tokens with shape "
                f"[B,F,D]; got {tuple(action_tokens.shape)}."
            )
        if int(action_tokens.shape[-1]) != self.action_embed_dim:
            raise ValueError(
                "LADD action-token width does not match the built action "
                f"conditioner: got {int(action_tokens.shape[-1])}, expected "
                f"{self.action_embed_dim}."
            )
        if int(action_tokens.shape[1]) < 1:
            raise ValueError("LADD action conditioning received zero frames.")
        ref = self.frame_proj.weight
        x = action_tokens.to(device=ref.device, dtype=ref.dtype)
        x = self.frame_proj(self.norm(x))
        mean = x.mean(dim=1)
        scale = torch.sqrt(x.var(dim=1, unbiased=False) + 1.0e-6)
        first = x[:, 0]
        last = x[:, -1]
        delta = last - first
        return self.summary(torch.cat([mean, scale, first, last, delta], dim=-1))


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
        action_embed_dim: int = 0,
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
        pixel_source=None,
    ):
        super().__init__()
        self.projector = projector  # stored as plain attribute, not nn submodule
        # ===== PIXEL FEATURE SOURCE (analysis/gan_tuning/PIXEL_FEATURE_
        # SOURCE.md) =====================================================
        # When present, this REPLACES the WAN projector as the disc's
        # feature basis: ``forward`` decodes its latent input to pixels,
        # runs them through this encoder, and hands the resulting token
        # features to the SAME CCM / CSM / ``LADDDiscHead`` stack. It is
        # a real ``nn.Module`` submodule (unlike ``projector``) because
        # its parameters ARE the disc's -- they belong in
        # ``disc.parameters()``, in ``r3gan_optimizer``, in the disc's
        # state_dict and inside the DDP reducer.
        #
        # ``block_indices`` and ``dim_teacher`` are then taken FROM the
        # source rather than from ``ladd_feature_blocks`` /
        # ``dim_teacher``: a pixel encoder's taps are its own, and
        # silently reusing WAN's [0,2,4,8,29] would index blocks that do
        # not exist. The override is logged, not silent.
        self.pixel_source = pixel_source
        if pixel_source is not None:
            if readout != "ladd":
                raise ValueError(
                    "LADDDiscriminator: a pixel feature source requires "
                    f"ladd_readout='ladd'; got {readout!r}. The register "
                    "readout consumes the RAW WAN token sequence "
                    "(padding arithmetic, action tokens, "
                    "``real_tokens``) and none of that exists on a "
                    "pixel encoder's feature grid."
                )
            if wavelet_hf_enabled:
                raise ValueError(
                    "LADDDiscriminator: ladd_wavelet_hf_enabled=true is "
                    "incompatible with a pixel feature source. The "
                    "wavelet stage rewrites the LATENT before the "
                    "projector; here the latent is VAE-DECODED, and a "
                    "wavelet-HF latent does not decode to a picture. "
                    "(It is also measured to kill the gt_transition "
                    "GAN outright.) Set it false."
                )
            _blk = list(pixel_source.tap_indices)
            if _blk != sorted(set(int(i) for i in block_indices)):
                logging.info(
                    "[LADD-PIXFEAT] block_indices OVERRIDDEN by the pixel "
                    "source: requested %s (ladd_feature_blocks, WAN "
                    "transformer blocks) -> using %s (the encoder's own "
                    "taps). ladd_feature_blocks is INERT on this arm.",
                    sorted(set(int(i) for i in block_indices)), _blk,
                )
            block_indices = _blk
            dim_teacher = pixel_source.tap_dims
        # ===== ORDERLESS POOLED READOUT (ladd_feature_source=vgg) =====
        # ``pooled_readout`` is ``None`` for EVERY historical source
        # (WAN 'real'/'fake', 'pixgan', 'dinov2'), so every branch guarded
        # on it below is dead on those arms and their behaviour is
        # byte-identical.
        #
        # When it is present the disc's readout is the pixel source's own
        # ``LaddPixelStatHead``: [mu, sigma, Cov] pooled over ALL spatial
        # positions, then a tiny MLP -> ONE logit per image. The CCM /
        # CSM / ``LADDDiscHead`` stack is NOT BUILT AT ALL in that mode --
        # not merely unused. That matters for two reasons: (a) an unused
        # trainable submodule is a parameter that receives no gradient,
        # which aborts the DDP reducer, and this campaign has already
        # lost an arm to exactly that (``_freeze_ddp_unreachable``);
        # (b) a dense head that still exists is a dense head somebody can
        # accidentally re-enter. The register readout at ``readout=
        # 'register'`` sets the same four attributes to ``None``, so
        # every downstream consumer already tolerates it.
        # ``pooled_readout`` is a read-only PROPERTY on this class (see
        # below) that reads through to ``pixel_source``: assigning the
        # head here would register it a SECOND time as a direct submodule
        # and duplicate every one of its tensors in ``state_dict()``.
        self.block_indices = sorted(set(int(i) for i in block_indices))
        if isinstance(dim_teacher, dict):
            self.dim_teacher_map = {
                int(i): int(dim_teacher[i]) for i in self.block_indices
            }
            self.dim_teacher = int(
                self.dim_teacher_map[self.block_indices[0]]
            )
        else:
            self.dim_teacher = int(dim_teacher)
            self.dim_teacher_map = {
                int(i): self.dim_teacher for i in self.block_indices
            }
        self.dim_proj = int(dim_proj)
        self.use_csm = bool(use_csm)
        self.cmap_dim = int(cmap_dim)
        self.action_embed_dim = int(action_embed_dim)
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
        if self.pooled_readout is not None:
            # ORDERLESS READOUT. The dense CCM / CSM / LADDDiscHead stack
            # is NOT BUILT -- not merely unused -- for exactly the reason
            # the ``register`` branch below does the same: an unused
            # trainable submodule is a parameter that receives no
            # gradient, and this disc's DDP reducer raises "Expected to
            # have finished reduction" on one. It also means a dense
            # head cannot be accidentally re-entered later.
            self.ccm = None
            self.csm = None
            self.heads = None
            self.cmapper = None
            self.pooled_action_proj = None
            if self.action_embed_dim > 0:
                if self.cmap_dim <= 0:
                    raise ValueError(
                        "Pooled LADD action conditioning requires cmap_dim>0."
                    )
                # Projection discriminator on the ORDERLESS statistic
                # embedding.  This keeps the winning VGG [mu,sigma,Cov]
                # evidence and adds only <h(image), c(action)>; it does not
                # replace it with the dense spatial LADD heads.
                self.action_cmapper = LADDActionConditioner(
                    self.action_embed_dim, self.cmap_dim,
                )
                self.pooled_action_proj = nn.Linear(
                    int(self.pooled_readout.hidden_dim),
                    self.cmap_dim,
                    bias=False,
                )
            else:
                self.action_cmapper = None
        elif self.readout == "register":
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
            self.action_cmapper = None
            self.pooled_action_proj = None
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
            self.pooled_action_proj = None
            self.ccm = LADDChannelMixer(
                block_indices=self.block_indices,
                # Per-tap dims when a pixel source is installed (its
                # conv taps may differ in width); the historical scalar
                # otherwise, which builds the identical ModuleDict.
                dim_teacher=(
                    self.dim_teacher_map if self.pixel_source is not None
                    else self.dim_teacher
                ),
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
                if prompt_embed_dim <= 0 and self.action_embed_dim <= 0:
                    raise ValueError(
                        "LADDDiscriminator: cmap_dim>0 requires prompt or "
                        "action conditioning."
                    )
                self.cmapper = (
                    nn.Linear(prompt_embed_dim, self.cmap_dim)
                    if prompt_embed_dim > 0 else None
                )
                self.action_cmapper = (
                    LADDActionConditioner(
                        self.action_embed_dim, self.cmap_dim,
                    )
                    if self.action_embed_dim > 0 else None
                )
            else:
                self.cmapper = None
                self.action_cmapper = None
                if self.action_embed_dim > 0:
                    raise ValueError(
                        "LADD action conditioning requires cmap_dim>0."
                    )

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

        # ---- pixel-source RUNTIME state (all plain attributes) --------
        # ``pixel_decode_fn`` is supplied by the trainer, which owns the
        # VAE; keeping it a callback rather than a VAE handle means this
        # module never learns about ``self.model.vae``,
        # ``_pix_vae_dtype`` or checkpointed decoding.
        #   signature: fn(latents[n, F, C, h, w], want_grad: bool)
        #                 -> pixels[n, F_pix, 3, h*8, w*8]  in [-1, 1]
        # ``pixel_epoch`` is set ONCE PER TRAINING STEP by the trainer
        # and is what makes the crop origin and the phase jitter
        # identical across the five D-updates and across the two
        # finite-difference R1 forwards -- see
        # ``LaddPixelFeatureSource.jitter_for``.
        self.pixel_decode_fn = None
        self.pixel_epoch = 0
        # Optional benchmark-only observer installed by the trainer.  It is
        # deliberately a plain callable (not an nn.Module) and defaults to
        # None, so production forwards have no state-dict/DDP surface and no
        # behaviour change.  The callback receives the *actual* pixel tensor
        # handed to the feature source, after shared real/fake crop geometry,
        # VAE decode, border/phase jitter and frame selection.  A trainer-side
        # context identifies the real/fake row split for the current D update.
        self.pixel_capture_fn = None
        self.pixel_capture_context = None
        # ``decode_batch`` IS INERT ON THIS ROUTE -- it is read by nothing
        # here and is kept only so the dict the trainer writes keeps its
        # historical shape (lowering it is a silent no-op, which is
        # exactly the failure class this campaign keeps hitting). The
        # knob that really splits the decode is ``decode_split`` below.
        self.pixel_cfg: Dict[str, int] = {
            "crop_rows": 24, "crop_cols": 32, "crops_per_row": 1,
            "lat_frames": 2, "frames_per_crop": 2, "border": 8,
            "decode_batch": 4,
            # ---- SAMPLE-BUDGET knobs. ALL DEFAULT OFF, and every off
            # path is the byte-identical legacy one: same RNG draw order
            # and count, same single decode call, same full-height range.
            #
            # ``decode_split``: >0 decodes ``crops_t`` in row-groups of
            #   this many rows instead of one shot, capping the decode
            #   transient so ``crops_per_row``/``lat_frames`` can be
            #   raised past what a single-shot decode fits. 0 = off.
            #   This is the real knob ``decode_batch`` was never wired to.
            # ``crop_y_lo_frac`` / ``crop_y_hi_frac``: restrict the crop
            #   ORIGIN band to a vertical slice of the latent frame, as
            #   fractions of H. (0.0, 1.0) = off = the historical
            #   full-height uniform draw. Motivation is measured: 39% of
            #   the adversarial crops contain no road or verge at all
            #   (COMMENTS_FOR_USER.md, sign-off open). A lower-2/3 band
            #   is ``crop_y_lo_frac=0.3333``. It stays a deterministic
            #   function of ``pixel_epoch``, so the five D-updates and
            #   both R1 forwards still see one identical crop.
            # ``crop_stratify_x``: 1 => the K crop origins are spread
            #   over K equal-width column bands (one draw per band)
            #   instead of K independent uniform draws, so coverage is
            #   K*p rather than 1-(1-p)^K and two crops cannot collide.
            "decode_split": 0,
            "crop_y_lo_frac": 0.0, "crop_y_hi_frac": 1.0,
            "crop_stratify_x": 0,
        }
        # Optional generator-guidance geometry.  ``None`` is the shipped
        # default and preserves the historical single-geometry path exactly.
        # The trainer sets ``pixel_use_g_cfg`` only around the two gen-side
        # discriminator forwards; D updates (including finite-difference R1)
        # always use ``pixel_cfg``.  This lets a detached D pass score more
        # evidence without forcing the same graph-on VAE budget through the
        # generator backward.
        self.pixel_g_cfg: Optional[Dict[str, int]] = None
        self.pixel_use_g_cfg = False
        self._pixel_last_route = ""
        # CONNECTION PROOF counters. Never reset; drained by the trainer
        # into ``train/ladd_pix_*``. A pixel arm whose ``fwd`` counter is
        # 0 did not run, whatever the config echo says.
        #
        # CROP-PLAN counters (added 2026-08-26). ``lat_frames`` and
        # ``frames_per_crop`` are both SILENTLY CLAMPED against the
        # tensor they get (``L = min(lat_frames, F_lat)``,
        # ``kf = min(frames_per_crop, F_pix)``) and until now neither had
        # a counter of any kind -- the only evidence a raise took was the
        # boot echo, which prints the CONFIG and not the realised value.
        # A silent clamp with no counter is precisely the mechanism
        # behind several entries in the silent-failure taxonomy, so every
        # element of the realised crop plan now has a key. The ``_used``
        # keys are LAST-VALUE (overwritten each forward); ``_clamped``
        # keys are CUMULATIVE counts of forwards where the clamp bit.
        self.pixel_stats: Dict[str, float] = {
            "fwd": 0.0, "decode_grad": 0.0, "decode_nograd": 0.0,
            "images": 0.0, "wan_projector_calls": 0.0,
            "lat_frames_cfg": 0.0, "lat_frames_used": 0.0,
            "lat_frames_avail": 0.0, "lat_frames_clamped": 0.0,
            "frames_per_crop_cfg": 0.0, "frames_per_crop_used": 0.0,
            "frames_avail": 0.0, "frames_per_crop_clamped": 0.0,
            "crops_per_row_used": 0.0,
            "crop_rows_used": 0.0, "crop_cols_used": 0.0,
            # -1 = NEVER DRAWN. 0 is a legal y0, so a zero sentinel here
            # would be a forgeable zero.
            "crop_y0_min": -1.0, "crop_y0_max": -1.0,
            "crop_y0_sum": 0.0, "crop_draws": 0.0,
            "crop_band_active": 0.0, "crop_stratify_active": 0.0,
            "decode_calls": 0.0, "decode_split_active": 0.0,
            "decode_split_cfg": 0.0,
            # ORDERLESS-READOUT PROOF. ``pooled_calls`` counts pooled
            # readout forwards (0 => the VGG stat head never ran, i.e.
            # a silently-inert path). ``logits_per_sample`` is the
            # SHAPE the readout actually emitted, recorded every forward
            # -- if it ever equals the token count (1768 on the DINOv2
            # geometry) a dense head has been reintroduced and the
            # experiment is void. On the pooled path it must read
            # ``crops_per_row * frames_per_crop`` (= 2 as shipped).
            "pooled_calls": 0.0, "logits_per_sample": 0.0,
            "dense_head_calls": 0.0,
        }

    # ------------------------------------------------------------------
    @property
    def pooled_readout(self):
        """The pixel source's ORDERLESS ``LaddPixelStatHead``, or ``None``.

        ``None`` for every historical feature source (WAN 'real'/'fake',
        'pixgan', 'dinov2'), so every branch guarded on it is dead on
        those arms and their behaviour is byte-identical. Non-``None``
        only for ``ladd_feature_source=vgg``.

        A PROPERTY and not an attribute on purpose: assigning the head to
        ``self`` would register it a second time as a direct submodule
        and duplicate every one of its tensors in ``state_dict()``.
        """
        src = getattr(self, "pixel_source", None)
        return getattr(src, "pooled_readout", None) if src is not None \
            else None

    # ------------------------------------------------------------------
    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _pooled_conditioned_logits(
        self,
        feats,
        *,
        action_tokens: Optional[torch.Tensor] = None,
        mismatch_action_tokens: Optional[torch.Tensor] = None,
    ):
        """Score a pooled VGG/RN50 evidence bank under one/two actions."""
        if self.pooled_readout is None:
            raise RuntimeError("pooled conditioning requires a pooled readout")
        action_proj = getattr(self, "pooled_action_proj", None)
        if action_proj is None:
            # The trainer must still build/pass action tensors for the WAN
            # discriminator geometry even when this pooled pixel readout is
            # deliberately action-blind.  Preserve the historical blind arm
            # by ignoring the matched tokens here.  A mismatch objective is
            # different: it requires an action projection and must fail loud.
            if mismatch_action_tokens is not None:
                raise ValueError(
                    "Wrong-action logits require an action-conditioned "
                    "pooled LADD discriminator."
                )
            return self.pooled_readout(feats), None

        if action_tokens is None:
            raise ValueError(
                "Pooled LADD action conditioner is active but matched action "
                "tokens were not provided."
            )
        base, hidden = self.pooled_readout(feats, return_hidden=True)
        image_cmap = action_proj(hidden)

        def _compat(tokens: torch.Tensor) -> torch.Tensor:
            cmap = self.action_cmapper(tokens)
            n_img, n_rows = int(image_cmap.shape[0]), int(cmap.shape[0])
            if n_rows <= 0 or n_img % n_rows != 0:
                raise ValueError(
                    "Pooled LADD action batch must divide the image batch; "
                    f"got images={n_img}, action_rows={n_rows}."
                )
            cmap = cmap.repeat_interleave(n_img // n_rows, dim=0)
            return (image_cmap * cmap).sum(dim=1, keepdim=True) * (
                1.0 / math.sqrt(float(self.cmap_dim))
            )

        matched = base + _compat(action_tokens)
        mismatched = (
            base + _compat(mismatch_action_tokens)
            if mismatch_action_tokens is not None else None
        )
        return matched, mismatched

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
    def score_pixels(
        self,
        pixels: torch.Tensor,
        *,
        action_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``[N, 3, H, W]`` pixels -> ``[N]`` scalar critic value.

        The SURROGATE TEACHER entry point (researcher directive
        2026-08-26: "give it the pixel outputs and then map those into
        generator but using the surrogate gradients method"). It is the
        SAME encoder and the SAME CCM/CSM/head stack the ``gt_vs_fake``
        D-update just trained -- deliberately, because the whole premise
        of distillation is that the student is regressing the critic
        that is actually being trained, not a parallel copy of it.

        Two differences from ``forward``, both necessary:

        * it takes PIXELS, so the surrogate's crop plan (which owns its
          own decode, its own A20/A21 real pool and its own
          ``pix_teacher_refresh_every`` cadence) is not double-cropped
          by ``_pixel_features``;
        * it returns ONE value per image, because
          ``compute_teacher_targets`` differentiates ``value.sum()``
          w.r.t. the latent crop and asserts ``value.shape[0] ==
          z.shape[0]``.

        The mean over the token axis is the same reduction
        ``ladd_scalar_output`` applies inside ``forward``, so the
        surrogate is fit to the critic's own scalar, not to some other
        pooling of it.
        """
        if self.pixel_source is None:
            raise RuntimeError(
                "LADDDiscriminator.score_pixels requires a pixel feature "
                "source; this disc scores WAN features of latents and has "
                "no pixel entry point."
            )
        if self.readout != "ladd":
            raise RuntimeError(
                "LADDDiscriminator.score_pixels requires ladd_readout="
                f"'ladd'; got {self.readout!r}."
            )
        feats = self.pixel_source(pixels, epoch=int(self.pixel_epoch))
        if self.pooled_readout is not None:
            # Already one value per image by construction, so there is
            # no token-axis mean to take here.
            self.pixel_stats["pooled_calls"] += 1.0
            self.pixel_stats["teacher_calls"] = (
                self.pixel_stats.get("teacher_calls", 0.0) + 1.0
            )
            logits, _ = self._pooled_conditioned_logits(
                feats, action_tokens=action_tokens,
            )
            return logits.reshape(-1)
        if self.action_cmapper is not None:
            raise RuntimeError(
                "score_pixels action conditioning is currently implemented "
                "for the orderless pooled VGG/RN50 readout only."
            )
        gh, gw = self.pixel_source.grid
        proj = self.ccm(feats)
        if self.csm is not None:
            proj = self.csm(proj)
        vals = []
        for idx in self.block_indices:
            f = proj[idx]
            f = f.transpose(1, 2).reshape(
                int(f.shape[0]), self.dim_proj, gh, gw).contiguous()
            # cmap is not consulted here: the surrogate teacher must be a
            # function of the PICTURE alone. A prompt-modulated value
            # would make the distillation target depend on a conditioning
            # signal the latent critic never sees.
            l = self.heads[str(idx)](f, cmap=None) if self.cmap_dim <= 0 \
                else self.heads[str(idx)].cls(
                    self.heads[str(idx)].res_block(
                        self.heads[str(idx)].in_block(f))).mean(
                            1, keepdim=True)
            vals.append(l.reshape(int(f.shape[0]), -1).mean(dim=1))
        self.pixel_stats["teacher_calls"] = (
            self.pixel_stats.get("teacher_calls", 0.0) + 1.0
        )
        return torch.stack(vals, dim=0).mean(dim=0)

    # ------------------------------------------------------------------
    def _pixel_features(
        self, x_noisy: torch.Tensor,
    ) -> Tuple[Dict[int, torch.Tensor], int, Tuple[int, int]]:
        """Latents -> crops -> PIXELS -> encoder tokens.

        Returns ``(features, n_images_per_row, (gh, gw))``.

        GEOMETRY, and why every choice is what it is
        --------------------------------------------
        * **Crops, not full frames.** A ride window is 60x104 latent =
          480x832 px; a graph-on decode of that, five D-updates deep,
          does not fit. ``pix_crop_lat = [24, 32]`` -> 192x256 px is the
          geometry the pixel critic and the style loss already use, so
          the three read the same texture scale.
        * **Crop origins are shared by EVERY ROW in the batch, and are a
          deterministic function of ``pixel_epoch``.** Both halves
          matter and neither is an optimisation:
            - shared across rows makes the real rows and the fake rows
              spatially ALIGNED whatever the batch layout is (positional
              ``cat([real, fake])``, the micro-batched slice, the
              matched ``segs`` list, the G-term's ``cat([real_g,
              fake_g])``). If the two sides were cropped independently
              the disc could separate them on CONTENT POSITION, which is
              a real, silent, and catastrophic tell.
            - deterministic-per-step makes ``D(x)`` and ``D(x + sigma*eps)``
              -- the two forwards the finite-difference R1 estimator
              subtracts -- see the SAME crop and the same phase jitter.
              With a per-call RNG draw the difference would be dominated
              by the geometry change and R1 would measure noise.
          It also consumes no global RNG, so it cannot desynchronise
          DDP ranks or perturb any other path's byte-identity.
        * **The newest ``lat_frames`` latent frames.** Same choice, same
          reason, as ``_compute_style_gram_loss``: those are the frames
          inference actually commits.
        * **Phase jitter** is applied as an OFFSET INSIDE the border
          trim that is being cropped away anyway, so it costs nothing
          and introduces no padding. See the module docstring of
          ``model/ladd_pixel_features.py`` for why a randomised lattice
          phase is the load-bearing part of the cure.
        """
        if self.pixel_decode_fn is None:
            raise RuntimeError(
                "LADDDiscriminator: a pixel feature source is installed "
                "but ``pixel_decode_fn`` was never set. The trainer must "
                "install the VAE decode callback before the first disc "
                "forward; without it the disc has no way to reach "
                "pixels. Refused rather than silently falling back to "
                "the WAN taps -- a silent fallback is exactly how a "
                "'pixel' arm ends up being a latent arm."
            )
        if x_noisy.dim() != 5:
            raise ValueError(
                "LADDDiscriminator pixel path expects [B, F, C, H, W] "
                f"latents; got {tuple(x_noisy.shape)}."
            )
        _split_geometry = self.pixel_g_cfg is not None
        _use_g_geometry = bool(self.pixel_use_g_cfg) and _split_geometry
        cfg = self.pixel_g_cfg if _use_g_geometry else self.pixel_cfg
        # Empty when split geometry is disabled: no extra stats are touched,
        # which keeps the default path's state and telemetry byte-identical.
        _route = "g_" if _use_g_geometry else ("d_" if _split_geometry else "")
        self._pixel_last_route = _route

        def _route_add(key: str, value: float) -> None:
            if _route:
                rk = _route + key
                self.pixel_stats[rk] = self.pixel_stats.get(rk, 0.0) + value

        def _route_set(key: str, value: float) -> None:
            if _route:
                self.pixel_stats[_route + key] = value

        B, F_lat, _C, H, W = x_noisy.shape
        cr = min(int(cfg["crop_rows"]), H)
        cc = min(int(cfg["crop_cols"]), W)
        K = max(1, int(cfg["crops_per_row"]))
        L = max(1, min(int(cfg["lat_frames"]), F_lat))
        lat = x_noisy[:, F_lat - L:]

        # ---- CROP-ORIGIN BAND (default OFF) --------------------------
        # ``y_lo``/``y_hi_excl`` are the [low, high) bounds handed to
        # ``randint``. With the shipped fractions (0.0, 1.0) they are
        # EXACTLY ``0`` and ``max(1, H - cr + 1)``, i.e. the historical
        # draw, and the branch below takes the legacy expression
        # verbatim so byte-identity is structural rather than argued.
        _ylo_f = float(cfg.get("crop_y_lo_frac", 0.0) or 0.0)
        _yhi_f = float(cfg.get("crop_y_hi_frac", 1.0) or 1.0)
        _band_on = (_ylo_f > 0.0) or (_yhi_f < 1.0)
        _strat_on = bool(int(cfg.get("crop_stratify_x", 0) or 0))
        y_span = max(1, H - cr + 1)
        x_span = max(1, W - cc + 1)
        if _band_on:
            # Crop must lie inside [round(lo*H), round(hi*H)).
            y_lo = max(0, min(y_span - 1, int(round(_ylo_f * H))))
            y_hi_excl = max(y_lo + 1,
                            min(y_span, int(round(_yhi_f * H)) - cr + 1))
        else:
            y_lo, y_hi_excl = 0, y_span

        g = torch.Generator(device="cpu")
        g.manual_seed((int(self.pixel_epoch) * 7919 + 104729) & 0x7FFF_FFFF)
        crops = []
        crop_origins = []
        for _k in range(K):
            if _band_on:
                y0 = int(torch.randint(y_lo, y_hi_excl, (1,),
                                       generator=g).item())
            else:
                y0 = (int(torch.randint(0, max(1, H - cr + 1), (1,),
                                        generator=g).item()))
            if _strat_on and K > 1:
                # One draw per equal-width column band: the K origins
                # cannot collide, so coverage is K*p instead of the
                # union 1-(1-p)^K. Same number of RNG draws as the
                # independent path, so the schedule stays reproducible.
                _b_lo = (_k * x_span) // K
                _b_hi = max(_b_lo + 1, ((_k + 1) * x_span) // K)
                x0 = int(torch.randint(_b_lo, _b_hi, (1,),
                                       generator=g).item())
            else:
                x0 = (int(torch.randint(0, max(1, W - cc + 1), (1,),
                                        generator=g).item()))
            crops.append(lat[:, :, :, y0:y0 + cr, x0:x0 + cc])
            crop_origins.append((int(y0), int(x0)))
            _ps = self.pixel_stats
            _ps["crop_y0_sum"] += float(y0)
            _ps["crop_draws"] += 1.0
            _ps["crop_y0_min"] = (
                float(y0) if _ps["crop_y0_min"] < 0.0
                else min(_ps["crop_y0_min"], float(y0)))
            _ps["crop_y0_max"] = max(_ps["crop_y0_max"], float(y0))
        # ROW-MAJOR stacking: row b occupies slots [b*K, (b+1)*K). The
        # per-row logit fold at the bottom of ``forward`` is a plain
        # ``reshape(B, -1)`` and is only correct under this ordering.
        crops_t = torch.stack(crops, dim=1).reshape(B * K, L, _C, cr, cc)

        want_grad = bool(x_noisy.requires_grad) and torch.is_grad_enabled()
        # ---- DECODE, optionally SPLIT (default OFF) ------------------
        # ``decode_split`` is the knob ``decode_batch`` was never wired
        # to. Off (0) => ONE call with the whole tensor, which is the
        # historical behaviour byte for byte. On => ceil(n/split) calls
        # concatenated, which caps the VAE decoder's transient at
        # ``split`` rows and is what makes a larger ``crops_per_row`` x
        # ``lat_frames`` affordable.
        _split = int(cfg.get("decode_split", 0) or 0)
        if _split > 0 and int(crops_t.shape[0]) > _split:
            _parts = [
                self.pixel_decode_fn(crops_t[i:i + _split], want_grad)
                for i in range(0, int(crops_t.shape[0]), _split)
            ]
            self.pixel_stats["decode_calls"] += float(len(_parts))
            self.pixel_stats["decode_split_active"] = 1.0
            _route_add("decode_calls", float(len(_parts)))
            _route_set("decode_split_active", 1.0)
            pix = torch.cat(_parts, dim=0)
        else:
            pix = self.pixel_decode_fn(crops_t, want_grad)
            self.pixel_stats["decode_calls"] += 1.0
            _route_add("decode_calls", 1.0)
            _route_set("decode_split_active", 0.0)
        if pix.dim() != 5:
            raise ValueError(
                "pixel_decode_fn must return [n, F_pix, 3, H, W]; got "
                f"{tuple(pix.shape)}."
            )

        b = int(cfg["border"])
        dy, dx = self.pixel_source.jitter_for(int(self.pixel_epoch))
        Hp, Wp = int(pix.shape[-2]), int(pix.shape[-1])
        if b > 0 and Hp > 2 * b and Wp > 2 * b:
            dy = max(-b, min(b, int(dy)))
            dx = max(-b, min(b, int(dx)))
            pix = pix[..., b + dy:Hp - b + dy, b + dx:Wp - b + dx]

        F_pix = int(pix.shape[1])
        kf = max(1, min(int(cfg["frames_per_crop"]), F_pix))
        # EVENLY SPACED, not random: a random subset would differ
        # between the two R1 forwards for the same reason the crop would.
        sel = [int(round(i * (F_pix - 1) / max(1, kf - 1))) if kf > 1 else 0
               for i in range(kf)]
        imgs = pix[:, sel].reshape(-1, *pix.shape[2:]).to(torch.float32)

        # GAN-aligned discrimination benchmark capture.  Keep this outside
        # the feature-source call so every candidate basis is later evaluated
        # on byte-identical pixels.  Best-effort is intentional: diagnostics
        # must never be able to abort training, while the callback itself
        # writes a loud error marker on failure.
        _capture = getattr(self, "pixel_capture_fn", None)
        _capture_ctx = getattr(self, "pixel_capture_context", None)
        if callable(_capture) and isinstance(_capture_ctx, dict):
            try:
                _capture(
                    imgs.detach(),
                    {
                        **_capture_ctx,
                        "rows_total": int(B),
                        "crops_per_row": int(K),
                        "latent_frames_used": int(L),
                        "frames_per_crop": int(kf),
                        "crop_rows": int(cr),
                        "crop_cols": int(cc),
                        "pixel_height": int(imgs.shape[-2]),
                        "pixel_width": int(imgs.shape[-1]),
                        "frame_indices": list(sel),
                        "phase_jitter_y": int(dy),
                        "phase_jitter_x": int(dx),
                        "route": str(_route or "shared"),
                        # Private tensor payload for the aligned surrogate
                        # benchmark.  It is consumed and removed by the
                        # trainer callback before JSON metadata is written.
                        # Keeping it beside the exact post-decode pixels
                        # guarantees the latent and RGB banks use identical
                        # rows, frames, crop origins and augmentation draws.
                        "_latent_crops": crops_t.detach(),
                        "crop_origins_yx": [list(v) for v in crop_origins],
                    },
                )
            except Exception as _capture_exc:
                # The callback records its own marker where possible.  The
                # discriminator remains usable even if benchmark I/O fails.
                logging.exception(
                    "[LADD-DISCRIM-CAPTURE] callback failed: %s",
                    _capture_exc,
                )

        feats = self.pixel_source(imgs, epoch=int(self.pixel_epoch))
        self.pixel_stats["fwd"] += 1.0
        self.pixel_stats["images"] += float(imgs.shape[0])
        self.pixel_stats["decode_grad" if want_grad else "decode_nograd"] += 1.0
        _route_add("fwd", 1.0)
        _route_add("images", float(imgs.shape[0]))
        _route_add("decode_grad" if want_grad else "decode_nograd", 1.0)
        # ---- REALISED CROP PLAN. Every one of these is the value the
        # code USED, not the value the config asked for, and the two
        # ``_clamped`` counters are the difference. Before this existed,
        # ``lat_frames`` and ``frames_per_crop`` were silently clamped
        # (``min(cfg, F)``) with NO counter of any kind: the only
        # evidence a raise had taken was the boot echo, which prints the
        # request. ``lat_frames_clamped > 0`` means the disc's input had
        # fewer latent frames than the arm asked for and the arm is NOT
        # running the recipe it thinks it is.
        _ps = self.pixel_stats
        _ps["lat_frames_cfg"] = float(int(cfg["lat_frames"]))
        _ps["lat_frames_used"] = float(L)
        _ps["lat_frames_avail"] = float(F_lat)
        if L < int(cfg["lat_frames"]):
            _ps["lat_frames_clamped"] += 1.0
        _ps["frames_per_crop_cfg"] = float(int(cfg["frames_per_crop"]))
        _ps["frames_per_crop_used"] = float(kf)
        _ps["frames_avail"] = float(F_pix)
        if kf < int(cfg["frames_per_crop"]):
            _ps["frames_per_crop_clamped"] += 1.0
        _ps["crops_per_row_used"] = float(K)
        _ps["crop_rows_used"] = float(cr)
        _ps["crop_cols_used"] = float(cc)
        _ps["crop_band_active"] = 1.0 if _band_on else 0.0
        _ps["crop_stratify_active"] = 1.0 if (_strat_on and K > 1) else 0.0
        _ps["decode_split_cfg"] = float(_split)
        _route_set("lat_frames_cfg", float(int(cfg["lat_frames"])))
        _route_set("lat_frames_used", float(L))
        _route_set("lat_frames_avail", float(F_lat))
        if L < int(cfg["lat_frames"]):
            _route_add("lat_frames_clamped", 1.0)
        elif _route:
            _ps.setdefault(_route + "lat_frames_clamped", 0.0)
        _route_set("frames_per_crop_cfg", float(int(cfg["frames_per_crop"])))
        _route_set("frames_per_crop_used", float(kf))
        _route_set("frames_avail", float(F_pix))
        if kf < int(cfg["frames_per_crop"]):
            _route_add("frames_per_crop_clamped", 1.0)
        elif _route:
            _ps.setdefault(_route + "frames_per_crop_clamped", 0.0)
        _route_set("crops_per_row_used", float(K))
        _route_set("crop_rows_used", float(cr))
        _route_set("crop_cols_used", float(cc))
        _route_set("crop_band_active", 1.0 if _band_on else 0.0)
        _route_set(
            "crop_stratify_active", 1.0 if (_strat_on and K > 1) else 0.0)
        _route_set("decode_split_cfg", float(_split))
        return feats, K * kf, self.pixel_source.grid

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
        # ===== FEATURE BASIS ==========================================
        # Exactly one of these two runs, and the counters say which. The
        # pixel branch never calls ``self.projector``, so the WAN taps
        # (and, with ladd_feature_source='fake', the fake-score
        # backbone) are genuinely OUT of the disc's graph on a pixel arm
        # -- that is the point, and ``ladd_pix_wan_projector_calls``
        # staying at 0 is the proof.
        _pix_nimg = 0
        _pix_grid = (0, 0)
        B_pool = int(x_noisy.shape[0])
        if self.pixel_source is not None:
            feats, _pix_nimg, _pix_grid = self._pixel_features(x_noisy)
        else:
            self.pixel_stats["wan_projector_calls"] += 1.0
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
        if self.pooled_readout is not None:
            # ===== ORDERLESS READOUT =====================================
            # ``feats`` are the encoder's RAW per-tap maps
            # ``{tap: [N, C_l, H_l, W_l]}`` at their native resolutions.
            # The head pools each to [mu, sigma, Cov] over ALL spatial
            # positions and emits [N, 1]. N is
            # ``B * crops_per_row * frames_per_crop`` ROW-MAJOR (see
            # ``_pixel_features``), so the fold to [B, -1] gives
            # ``crops_per_row * frames_per_crop`` logits per row -- one
            # per FRAME, which is the coarsest readout the RpGAN
            # reduction can be given without averaging two independent
            # pictures together.
            _matched_actions = (
                conditional_extra.get("_action_tokens")
                if conditional_extra is not None else None
            )
            _mismatch_actions = (
                conditional_extra.get("_ladd_action_tokens_mismatch")
                if conditional_extra is not None else None
            )
            _pool_logits, _pool_mismatch_logits = (
                self._pooled_conditioned_logits(
                    feats,
                    action_tokens=_matched_actions,
                    mismatch_action_tokens=_mismatch_actions,
                )
            )
            visual_logits = _pool_logits.reshape(B_pool, -1)
            mismatch_visual_logits = (
                _pool_mismatch_logits.reshape(B_pool, -1)
                if _pool_mismatch_logits is not None else None
            )
            self.pixel_stats["pooled_calls"] += 1.0
            self.pixel_stats["logits_per_sample"] = float(
                visual_logits.shape[1])
            if self._pixel_last_route:
                _rk = self._pixel_last_route + "pooled_calls"
                self.pixel_stats[_rk] = self.pixel_stats.get(_rk, 0.0) + 1.0
                self.pixel_stats[
                    self._pixel_last_route + "logits_per_sample"
                ] = float(visual_logits.shape[1])
            if self.scalar_output:
                visual_logits = visual_logits.mean(dim=1, keepdim=True)
                if mismatch_visual_logits is not None:
                    mismatch_visual_logits = mismatch_visual_logits.mean(
                        dim=1, keepdim=True,
                    )
            if self.stat_head is not None:
                visual_logits = torch.cat(
                    [visual_logits, self.stat_head(x_noisy_raw)], dim=1,
                )
            if mismatch_visual_logits is not None:
                return visual_logits, mismatch_visual_logits
            return visual_logits

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
        def _condition_cmap(action_key: str = "_action_tokens"):
            if self.cmap_dim <= 0:
                return None
            parts = []
            if self.cmapper is not None:
                if pooled_prompt is None:
                    raise ValueError(
                        "LADD prompt conditioner is active but pooled_prompt "
                        "was not provided."
                    )
                parts.append(self.cmapper(pooled_prompt.float()))
            if self.action_cmapper is not None:
                action_tokens = (
                    conditional_extra.get(action_key)
                    if conditional_extra is not None else None
                )
                if action_tokens is None:
                    raise ValueError(
                        "LADD action conditioner is active but "
                        f"conditional_extra[{action_key!r}] was not provided."
                    )
                parts.append(self.action_cmapper(action_tokens))
            if not parts:
                raise RuntimeError(
                    "LADD cmap heads were built without an active condition."
                )
            out = parts[0]
            for part in parts[1:]:
                out = out + part
            if len(parts) > 1:
                out = out * (1.0 / math.sqrt(float(len(parts))))
            return out

        cmap = _condition_cmap()
        mismatch_requested = bool(
            conditional_extra is not None
            and "_ladd_action_tokens_mismatch" in conditional_extra
        )
        if mismatch_requested and self.action_cmapper is None:
            raise ValueError(
                "A LADD wrong-action condition was supplied, but action "
                "conditioning is not built."
            )
        mismatch_cmap = (
            _condition_cmap("_ladd_action_tokens_mismatch")
            if mismatch_requested else None
        )

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
        B = int(x_noisy.shape[0])
        proj_2d: Dict[int, torch.Tensor] = {}
        if self.pixel_source is not None:
            # PIXEL PATH. The token sequence is already exactly the
            # encoder's own (gh x gw) feature grid -- no padding, no
            # action tokens, no patch arithmetic -- so the fold is a
            # straight transpose+reshape. Leading axis is
            # ``B * crops_per_row * frames_per_crop``, ROW-MAJOR (see
            # ``_pixel_features``), which is what makes the
            # ``reshape(B, -1)`` below correct.
            gh, gw = int(_pix_grid[0]), int(_pix_grid[1])
            for idx in self.block_indices:
                feat = proj[idx]                       # [N, gh*gw, dim_proj]
                if int(feat.shape[1]) != gh * gw:
                    raise RuntimeError(
                        "LADDDiscriminator pixel path: tap "
                        f"{idx} has {feat.shape[1]} tokens but the "
                        f"encoder grid is {gh}x{gw}={gh * gw}. The taps "
                        "must share one grid (the CSM fuses them by "
                        "elementwise addition)."
                    )
                proj_2d[idx] = (
                    feat.transpose(1, 2)
                    .reshape(int(feat.shape[0]), self.dim_proj, gh, gw)
                    .contiguous()
                )
        else:
            F_in, _C_in, H_in, W_in = x_noisy.shape[1:]
            pt, ph, pw = self.patch_size
            T_prime = F_in // pt
            H_prime = H_in // ph
            W_prime = W_in // pw
            a_per_f = int(self.action_tokens_per_frame)
            frame_seqlen = H_prime * W_prime + a_per_f
            real_tokens = T_prime * frame_seqlen

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
                feat = feat.reshape(
                    B, T_prime, H_prime, W_prime, self.dim_proj)
                feat = feat.permute(0, 1, 4, 2, 3).contiguous()
                feat = feat.reshape(
                    B * T_prime, self.dim_proj, H_prime, W_prime)
                proj_2d[idx] = feat

        logits_per_scale = []
        mismatch_logits_per_scale = []
        self.pixel_stats["dense_head_calls"] += 1.0
        for idx in self.block_indices:
            head_out = self.heads[str(idx)](
                proj_2d[idx], cmap=cmap, mismatch_cmap=mismatch_cmap,
            )
            if mismatch_cmap is not None:
                l, lm = head_out
            else:
                l, lm = head_out, None
            # l: [B*T', cmap_dim_or_1, H', W'] -> per-sample flat.
            l = l.reshape(B, -1)
            logits_per_scale.append(l)
            if lm is not None:
                mismatch_logits_per_scale.append(lm.reshape(B, -1))
        visual_logits = torch.cat(logits_per_scale, dim=1)
        mismatch_visual_logits = (
            torch.cat(mismatch_logits_per_scale, dim=1)
            if mismatch_logits_per_scale else None
        )
        self.pixel_stats["logits_per_sample"] = float(visual_logits.shape[1])
        if self.scalar_output:
            # One invariant critic value per sample. This reduction is part of
            # D itself, so D/G losses and R1 all differentiate the exact
            # same scalar instead of summing a resolution-dependent token map.
            visual_logits = visual_logits.mean(dim=1, keepdim=True)
            if mismatch_visual_logits is not None:
                mismatch_visual_logits = mismatch_visual_logits.mean(
                    dim=1, keepdim=True,
                )

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
        if mismatch_visual_logits is not None:
            return visual_logits, mismatch_visual_logits
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
    action_embed_dim: int = 0,
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
    pixel_source=None,
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
        action_embed_dim=action_embed_dim,
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
        # PIXEL FEATURE SOURCE. When non-None the projector above is
        # still constructed (so the object graph is unchanged and the
        # aux-teacher ``feat_w`` path keeps its handle) but it is NEVER
        # CALLED -- ``LADDDiscriminator.forward`` takes the pixel branch
        # and ``ladd_pix_wan_projector_calls`` stays at 0.
        pixel_source=pixel_source,
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
