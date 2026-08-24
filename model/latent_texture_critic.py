"""Latent-space surrogate critic distilled from the pixel texture disc.

WP-SURROGATE (B3 in ``docs/GAN_REDESIGN.md``). Restored from the
pre-deletion implementation at commit ``835b1df``
(``model/latent_sam2_critic.py``, deleted at ``01ea13d``) with the SAM2
teacher swapped for ``model/pixel_texture_disc.py::PixelTextureDisc``
per frozen interface contract 1. Design notes live in
``docs/WP_SURROGATE.md``.

Why this module exists
----------------------
WP-PIXGAN's generator gradient crosses ``VAE.decode`` in graph-on mode
once per generator step. That is affordable at crop scale (a 24x32
latent crop) but it does not scale toward R10's >=60% frame coverage:
the decoder backward is the dominant transient. The fix is the standard
distilled-critic flow — an expensive frozen teacher supervises a cheap
latent-space student, and the *student* is what the generator
differentiates through:

  * **Value distillation**: ``MSE(g_phi(z), f(z).detach())`` — the
    surrogate's value matches the pixel disc's mean patch logit.
  * **Gradient distillation** (Sobolev): ``MSE(grad_z g_phi(z),
    grad_z f(z).detach())`` — the surrogate's *gradient field* matches
    the teacher's. Without this term ``g ~= f`` does not imply
    ``grad g ~= grad f``, and the generator would be free to move
    values without improving pixel texture. The gradient field is the
    only thing the generator actually consumes, so it is the thing that
    must be distilled.

The generator's autograd graph then never crosses the VAE or the pixel
disc — only this small latent-space module.

**The two terms do not share a scale**, and at the ancestor's shipped
weight the Sobolev term was effectively inert. That is corrected here by
``grad_loss_normalize`` (default ON), which turns the gradient term into
a relative error. Read the ``SOBOLEV SCALING`` note at the bottom of this
file before changing either weight — the measurements are there.

Teacher contract (delta (a) vs the ancestor)
--------------------------------------------
The teacher is injected as a callable, NOT imported: this module must
build and test standalone before WP-PIXGAN lands. The trainer passes

    teacher_value_fn(z_crop [N, F, C, h, w]) -> patch logits

where the implementation is ``pixel_texture_disc(decode_grad(z_crop))``
and the returned tensor is any batch-first shape (``[N]``, ``[N, 1]`` or
a patch map ``[N, 1, ph, pw]``; ``[N, Fpix, 1, ph, pw]`` also works).
Reduction to the contract's scalar is ``mean over everything but the
batch dim`` — the mean patch logit, exactly the reduction
``TEXTURE_GAN_DESIGN`` Section 5.3 uses for the R1 penalty and the one
``model/disc_holdout_probe.py::_reduce_scores`` already implements.

Teacher gradient is then ``autograd.grad(v.sum(), z_crop)`` where ``v``
is the per-sample mean patch logit. Summing over the batch is NOT a
change of objective: samples are independent, so row ``i`` of the result
is exactly ``autograd.grad(disc(decode_grad(z_i)).mean(), z_i)`` — the
contract expression evaluated per crop. It is preferred over a batch
``.mean()`` only because it keeps the Sobolev loss scale independent of
the crop batch size, so ``grad_loss_weight`` does not have to be
re-tuned when ``pix_crops_per_step`` changes.

Refresh cadence (delta (b) vs the ancestor)
-------------------------------------------
The historical code ran the teacher EVERY step
(``gan_critic_grad_full_every`` only widened *frame* coverage within a
step). Here ``pix_teacher_refresh_every=N`` runs the teacher forward +
gradient every N steps and caches ``(z, value, grad)`` triples; the
distillation replays cached triples on the intervening steps. The critic
serves the generator on EVERY step regardless — that is the whole point
of the surrogate.

The cache stores the latents alongside the targets because a teacher
target is only meaningful at the ``z`` it was taken at. Between
refreshes the generator's fakes have moved, so re-using a target against
a *new* latent would be silently wrong. Replaying the frozen triple is
correct-but-stale, and the staleness is observable: ``teacher_target_age``
logs it and ``critic_disc_corr`` / ``critic_grad_cos_sim`` measure what
it costs.

Hand-rolled attention is load-bearing
-------------------------------------
``L_grad.backward()`` is a second-order backward through the critic.
``F.scaled_dot_product_attention`` selects the FlashAttention backend on
this hardware and Flash has no double-backward kernel. The explicit
``softmax(QK^T/sqrt(d)) V`` path below has working double-backward
everywhere. Do not "optimize" it into SDPA.

Order of operations per training step (from the ancestor)
--------------------------------------------------------
    D-update (pixel disc) -> critic distillation -> generator consumes
    the JUST-UPDATED critic.

The generator must see the freshly-distilled critic, not the previous
iteration's; this mirrors ``_compute_action_critic_losses``
(``trainer/causal_action_forcing_train.py`` around :3038-3211), which is
the live template for the frozen-critic idiom used by
``generator_surrogate_loss`` below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "LatentTextureCritic",
    "TeacherTargets",
    "TeacherTargetCache",
    "compute_teacher_targets",
    "LatentSurrogateDistiller",
    "generator_surrogate_loss",
    "two_stage_gen_weight",
    "reduce_patch_logits",
    "build_from_config",
]

# Spatial reduction of the conv stem (three stride-2 stages).
STEM_SPATIAL_STRIDE = 8


def _gn(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """GroupNorm with the largest divisor of ``channels`` that does not
    exceed ``max_groups``. Matches ADM's GN sizing convention."""
    g = max_groups
    while g > 1 and channels % g != 0:
        g //= 2
    return nn.GroupNorm(num_groups=g, num_channels=channels)


def reduce_patch_logits(raw: torch.Tensor) -> torch.Tensor:
    """Teacher output -> ``[N]`` per-sample mean patch logit.

    Accepts ``[N]``, ``[N, 1]``, ``[N, 1, ph, pw]`` or any batch-first
    shape. This is the contract-1 reduction (``TEXTURE_GAN_DESIGN``
    Section 5.3) and matches
    ``model/disc_holdout_probe.py::_reduce_scores`` — except that this
    one does NOT detach, because the teacher-gradient path differentiates
    through it.
    """
    if raw.dim() == 0:
        raise ValueError(
            "teacher_value_fn returned a 0-d tensor; a batch-first "
            "tensor is required so per-crop values stay separable."
        )
    t = raw.float()
    return t if t.dim() == 1 else t.reshape(int(t.shape[0]), -1).mean(dim=1)


class _MultiHeadSelfAttention(nn.Module):
    """Hand-rolled multi-head self-attention.

    Used instead of ``nn.MultiheadAttention`` / SDPA because the gradient
    distillation path computes ``L_grad.backward()``, which requires
    second-order grad through the attention. PyTorch's SDPA backends
    select FlashAttention by default on some devices and Flash lacks a
    double-backward kernel — we want the explicit
    ``softmax(QK^T/sqrt(d)) V`` path, which has working double-backward
    on every device. LOAD-BEARING; see module docstring.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by num_heads={num_heads}."
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        B, N, D = x.shape
        qkv = self.qkv(x)  # [B, N, 3D]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        # -> [3, B, num_heads, N, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = self.head_dim ** -0.5
        # [B, num_heads, N, N]
        scores = torch.matmul(q * scale, k.transpose(-2, -1))
        attn = scores.softmax(dim=-1)
        if self.training and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout)
        out = torch.matmul(attn, v)  # [B, num_heads, N, head_dim]
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


class _TransformerBlock(nn.Module):
    """Standard pre-LN transformer encoder block (hand-rolled MHA so
    second-order grad works for the gradient-distillation backward)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(d_model)
        ff_hidden = d_model * max(1, int(ff_mult))
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.GELU(),
            nn.Linear(ff_hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class LatentTextureCritic(nn.Module):
    """Latent-space surrogate critic distilled from the pixel texture
    disc. Frozen interface contract 3:

        ``forward(z [B, F, 16, 60, 104]) -> dense per-token value map``

    and the generator consumes ``-critic(z).mean()`` under the
    frozen-critic idiom (see :func:`generator_surrogate_loss`).

    Architecture (unchanged from ``835b1df``, except the head)
    ----------------------------------------------------------
    Input:  ``[B, F, C=16, H, W]`` latent video.
    Stem:   3D conv pyramid, 3x stride-(1, 2, 2) convs projecting to
            ``d_model`` and reducing spatial 8x (60x104 -> 8x13). Each
            block is Conv3d -> GroupNorm -> SiLU. Temporal dim preserved.
    Body:   ``num_blocks`` pre-LN transformer blocks with full space-time
            self-attention over ``F * Hs * Ws`` tokens.
    Head:   LayerNorm -> Linear(d_model, 1), zero-init, applied
            PER TOKEN -> ``[B, F, Hs, Ws]`` dense value map.

    Two deltas vs the ancestor, both required by the new teacher:

    * **Dense output.** The ancestor pooled to ``[B]`` because the SAM2
      disc emitted one scalar per clip. The pixel texture disc emits a
      patch-logit grid (``TEXTURE_GAN_DESIGN`` Section 4: "per-patch
      logits are kept (no global scalar)"), and researcher directive R2
      requires spatial/token-level gradients, so the surrogate is dense
      too. :meth:`pool` reproduces the ancestor's ``[B]`` reduction when
      a scalar is wanted; it is not on the generator path.
    * **2-D absolute positional embedding with a crop origin.** The
      ancestor used a flat ``[1, 1, max_spatial, d]`` table indexed in
      raster order, which is only well-defined at one fixed ``Ws``. This
      module is called at TWO resolutions — on ``(24, 32)`` latent crops
      during distillation and on the full ``(60, 104)`` latent when the
      generator consumes it — so a raster-indexed table would assign the
      same embedding to different absolute positions. The table here is
      2-D ``[max_h, max_w]`` in stem-token units and ``forward`` takes a
      ``latent_origin`` so a crop is embedded at its TRUE position in the
      frame. Vertical position is a real nuisance covariate here (sky /
      buildings / road have different texture statistics — the same fact
      that motivates the A24 band matching in
      ``disc_holdout_probe.band_plan``), so getting this right matters.
      ``latent_origin`` is quantized by ``STEM_SPATIAL_STRIDE``; the
      resulting sub-token error is a positional prior, not an index, so
      it is harmless.

    Constructor args:
        in_channels: latent channel count (16 for the WAN VAE).
        d_model: transformer hidden dim.
        num_blocks: transformer depth (4 at d_model=512 ~= 24M params).
        num_heads: attention heads.
        ff_mult: MLP expansion ratio.
        dropout: applied to attention weights.
        max_frames: max temporal tokens.
        max_token_h / max_token_w: positional table extent in stem-token
            units. Defaults cover the full 60x104 latent (8x13) with
            headroom.
        frame_pool: ``mean | max | topk_mean`` — retained from the
            ancestor for :meth:`pool` symmetry with the disc contract.
        frame_pool_topk: only consulted when ``frame_pool='topk_mean'``.
    """

    def __init__(
        self,
        *,
        in_channels: int = 16,
        d_model: int = 512,
        num_blocks: int = 4,
        num_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.0,
        max_frames: int = 64,
        max_token_h: int = 16,
        max_token_w: int = 32,
        frame_pool: str = "mean",
        frame_pool_topk: int = 4,
    ) -> None:
        super().__init__()
        if frame_pool not in ("mean", "max", "topk_mean"):
            raise ValueError(
                f"frame_pool must be 'mean' | 'max' | 'topk_mean'; "
                f"got {frame_pool!r}."
            )
        self.in_channels = int(in_channels)
        self.d_model = int(d_model)
        self.num_blocks = int(num_blocks)
        self.num_heads = int(num_heads)
        self.frame_pool = frame_pool
        self.frame_pool_topk = max(1, int(frame_pool_topk))
        self.max_frames = int(max_frames)
        self.max_token_h = int(max_token_h)
        self.max_token_w = int(max_token_w)

        # 3D conv stem: project channels to d_model and reduce spatial 8x.
        # Temporal dim is preserved (stride 1 in T) so token counts in the
        # transformer match the input frame count.
        c1 = max(d_model // 4, 64)
        c2 = max(d_model // 2, 128)
        c3 = d_model
        self.stem = nn.Sequential(
            nn.Conv3d(
                in_channels, c1, kernel_size=3,
                stride=(1, 2, 2), padding=1,
            ),
            _gn(c1),
            nn.SiLU(),
            nn.Conv3d(
                c1, c2, kernel_size=3,
                stride=(1, 2, 2), padding=1,
            ),
            _gn(c2),
            nn.SiLU(),
            nn.Conv3d(
                c2, c3, kernel_size=3,
                stride=(1, 2, 2), padding=1,
            ),
            _gn(c3),
            nn.SiLU(),
        )

        # Positional embeddings — per-frame plus a 2-D absolute spatial
        # table (see class docstring on why 2-D, not raster-flat).
        self.frame_pos = nn.Parameter(
            torch.zeros(1, self.max_frames, 1, 1, d_model)
        )
        nn.init.trunc_normal_(self.frame_pos, std=0.02)
        self.spatial_pos = nn.Parameter(
            torch.zeros(1, 1, self.max_token_h, self.max_token_w, d_model)
        )
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)

        # Transformer body — full space-time self-attention.
        self.blocks = nn.ModuleList([
            _TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                ff_mult=ff_mult,
                dropout=dropout,
            )
            for _ in range(self.num_blocks)
        ])

        # Output: per-token norm -> linear -> 1 value per token.
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        # Zero-init the head so the critic starts as the constant-zero
        # value field (matches the disc convention; the two-stage warmup
        # plus value distillation train it up from zero, and a zero field
        # means a zero generator gradient if anything mis-fires early).
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def token_grid(self, h: int, w: int) -> Tuple[int, int]:
        """Stem-output token grid for a ``(h, w)`` latent input."""
        for _ in range(3):
            h = (h - 1) // 2 + 1
            w = (w - 1) // 2 + 1
        return int(h), int(w)

    def forward(
        self,
        latent: torch.Tensor,
        latent_origin: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """Dense per-token value map over a latent video.

        Args:
            latent: ``[B, F, C, H, W]``. ``C`` must equal
                ``in_channels``; ``F <= max_frames``.
            latent_origin: optional ``(y0, x0)`` top-left corner of this
                input inside the full latent frame, in LATENT rows/cols.
                ``None`` means ``(0, 0)`` (the full-frame call). Pass the
                crop offset from ``disc_holdout_probe.take_crops`` when
                distilling on crops.

        Returns:
            ``[B, F, Hs, Ws]`` dense value map. The generator consumes
            ``-critic(z).mean()``; :meth:`pool` gives the ancestor's
            ``[B]`` scalar when one is needed for symmetry with the
            disc's frame_pool contract.
        """
        if latent.dim() != 5:
            raise ValueError(
                f"LatentTextureCritic expects [B, F, C, H, W]; "
                f"got {tuple(latent.shape)}."
            )
        B, F_, C, H, W = latent.shape
        if C != self.in_channels:
            raise ValueError(
                f"LatentTextureCritic in_channels={self.in_channels} but "
                f"got C={C}."
            )
        if F_ > self.max_frames:
            raise ValueError(
                f"LatentTextureCritic F={F_} exceeds "
                f"max_frames={self.max_frames}."
            )
        # DTYPE (MAIN 14:3x, break-fix, researcher-priority): the critic's
        # params are fp32 (build_from_config default) while the trainer's
        # streaming latents are bf16 and the pool crops fp16 -- Conv3d
        # refuses the mix. Third instance of the campaign's fp32-module /
        # low-precision-input fault (VAE grad + no-grad decode were 1 and
        # 2), fixed AT THE PRIMITIVE so every caller (consumption,
        # distillation, teacher replay) is covered: cast to the module's
        # OWN parameter dtype, derived never hardcoded. ``.to()`` is
        # differentiable, so the generator's gradient through the
        # consumption call survives; all callers get the identical
        # treatment by construction.
        _pdt = next(self.parameters()).dtype
        if latent.dtype != _pdt:
            latent = latent.to(_pdt)
        # [B, F, C, H, W] -> [B, C, F, H, W] for Conv3d.
        x = latent.permute(0, 2, 1, 3, 4).contiguous()
        x = self.stem(x)  # [B, d_model, F, Hs, Ws]
        D = x.shape[1]
        Hs = x.shape[3]
        Ws = x.shape[4]

        # Absolute token origin. Quantized by the stem stride — a
        # positional prior, so sub-token rounding is harmless.
        if latent_origin is None:
            oy = ox = 0
        else:
            oy = int(round(int(latent_origin[0]) / STEM_SPATIAL_STRIDE))
            ox = int(round(int(latent_origin[1]) / STEM_SPATIAL_STRIDE))
        if oy + Hs > self.max_token_h or ox + Ws > self.max_token_w:
            raise ValueError(
                f"LatentTextureCritic token window "
                f"y[{oy}:{oy + Hs}] x[{ox}:{ox + Ws}] exceeds the "
                f"positional table (max_token_h={self.max_token_h}, "
                f"max_token_w={self.max_token_w}). Input was "
                f"{tuple(latent.shape)} at latent_origin={latent_origin}."
            )

        # [B, d_model, F, Hs, Ws] -> [B, F, Hs, Ws, d_model]
        x = x.permute(0, 2, 3, 4, 1)
        pos_frame = self.frame_pos[:, :F_].to(dtype=x.dtype)
        pos_spatial = self.spatial_pos[
            :, :, oy:oy + Hs, ox:ox + Ws
        ].to(dtype=x.dtype)
        x = x + pos_frame + pos_spatial
        # Flatten to [B, F*Hs*Ws, d_model] for full space-time attention.
        x = x.reshape(B, F_ * Hs * Ws, D)
        for block in self.blocks:
            x = block(x)
        # Per-token head — dense value map, no pooling.
        x = self.norm_out(x)
        values = self.head(x).squeeze(-1)  # [B, F*Hs*Ws]
        return values.reshape(B, F_, Hs, Ws).float()

    def pool(self, dense: torch.Tensor) -> torch.Tensor:
        """``[B, F, Hs, Ws]`` dense map -> ``[B]`` scalar, using the
        ancestor's frame_pool contract (spatial mean, then the configured
        frame aggregation). Kept so distilled-mode losses can be written
        as if the disc and the critic produced the same shape; NOT on the
        generator path, which consumes the dense mean directly."""
        if dense.dim() != 4:
            raise ValueError(
                f"pool expects [B, F, Hs, Ws]; got {tuple(dense.shape)}."
            )
        per_frame_logit = dense.flatten(2).mean(dim=2)  # [B, F]
        F_ = per_frame_logit.shape[1]
        if self.frame_pool == "mean":
            per_sample = per_frame_logit.mean(dim=1)
        elif self.frame_pool == "max":
            per_sample = per_frame_logit.gather(
                1,
                per_frame_logit.abs().argmax(dim=1, keepdim=True),
            ).squeeze(1)
        else:  # topk_mean
            k = min(self.frame_pool_topk, F_)
            _, idx = per_frame_logit.abs().topk(k, dim=1)
            per_sample = per_frame_logit.gather(1, idx).mean(dim=1)
        return per_sample.float()

    def value(
        self,
        latent: torch.Tensor,
        latent_origin: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """``[B]`` per-sample mean value — the surrogate's counterpart to
        the teacher's mean patch logit. This is the quantity matched by
        the value-distillation term."""
        return self.forward(latent, latent_origin=latent_origin).flatten(1).mean(dim=1)


# ---------------------------------------------------------------------------
# Teacher targets and the refresh cache (delta (b))
# ---------------------------------------------------------------------------
@dataclass
class TeacherTargets:
    """One teacher-labelled distillation sample.

    ``z`` is stored WITH the targets on purpose: a Sobolev target is only
    meaningful at the latent it was taken at. Between refreshes the
    generator's fakes have moved, so replaying ``value``/``grad`` against
    a fresh latent would be silently wrong. Replaying the frozen triple
    is correct-but-stale, and the staleness is logged.

    Fields:
        z: ``[N, F, C, h, w]`` detached latent the teacher was evaluated at.
        value: ``[N]`` per-sample mean patch logit.
        grad: ``[N, F, C, h, w]`` teacher gradient, or ``None`` in
            value-only mode.
        origin: ``(y0, x0)`` latent-space crop origin, or ``None``.
        step: training step the teacher was run at (for the age metric).
        tag: ``"real"`` / ``"fake"``, for logging only.
    """

    z: torch.Tensor
    value: torch.Tensor
    grad: Optional[torch.Tensor]
    origin: Optional[Tuple[int, int]]
    step: int
    tag: str = ""

    def to(self, device: torch.device) -> "TeacherTargets":
        return TeacherTargets(
            z=self.z.to(device, non_blocking=True),
            value=self.value.to(device, non_blocking=True),
            grad=None if self.grad is None else self.grad.to(device, non_blocking=True),
            origin=self.origin,
            step=self.step,
            tag=self.tag,
        )


class TeacherTargetCache:
    """Bounded FIFO of :class:`TeacherTargets`, one per ``tag``.

    Memory is negligible at crop scale — ``pix_crops_per_step=4`` at
    ``(F=3, C=16, 24, 32)`` fp32 is ~590 KB for ``z`` plus the same for
    ``grad``, so a capacity of 8 per tag costs ~9 MB. Full-frame
    ``(3, 16, 60, 104)`` entries are ~4.8 MB per pair; keep the capacity
    small in that mode.

    ``store_on_cpu`` is offered for the full-frame case; it trades a
    host<->device copy per replay step for the VRAM.
    """

    def __init__(self, capacity: int = 8, store_on_cpu: bool = False) -> None:
        self.capacity = max(1, int(capacity))
        self.store_on_cpu = bool(store_on_cpu)
        self._entries: Dict[str, List[TeacherTargets]] = {}

    def push(self, entry: TeacherTargets) -> None:
        if self.store_on_cpu:
            entry = entry.to(torch.device("cpu"))
        bucket = self._entries.setdefault(entry.tag, [])
        bucket.append(entry)
        if len(bucket) > self.capacity:
            del bucket[: len(bucket) - self.capacity]

    def latest(self, tag: str) -> Optional[TeacherTargets]:
        bucket = self._entries.get(tag) or []
        return bucket[-1] if bucket else None

    def sample(
        self,
        tag: str,
        generator: Optional[torch.Generator] = None,
    ) -> Optional[TeacherTargets]:
        """Uniform draw from the bucket.

        Sampling the whole buffer rather than always replaying the newest
        entry is a small anti-forgetting measure: with
        ``pix_teacher_refresh_every`` large the critic would otherwise
        take N-1 consecutive gradient steps against one single batch and
        overfit it.

        DDP note: each rank draws independently from its OWN cache, which
        is built from its own data shard. That is ordinary data
        parallelism — the gradient all-reduce averages over a wider
        sample, not a narrower one. Do NOT "fix" this by broadcasting an
        index; the caches hold different tensors on different ranks.
        """
        bucket = self._entries.get(tag) or []
        if not bucket:
            return None
        if len(bucket) == 1:
            return bucket[0]
        i = int(torch.randint(
            0, len(bucket), (1,), generator=generator,
        ).item())
        return bucket[i]

    def age(self, tag: str, current_step: int) -> float:
        """Steps since the newest entry in ``tag`` was labelled."""
        newest = self.latest(tag)
        return 0.0 if newest is None else float(current_step - newest.step)

    def __len__(self) -> int:
        return sum(len(v) for v in self._entries.values())

    def clear(self) -> None:
        self._entries.clear()


def compute_teacher_targets(
    z: torch.Tensor,
    teacher_value_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    current_step: int,
    origin: Optional[Sequence[int]] = None,
    tag: str = "",
    want_grad: bool = True,
    use_checkpoint: bool = False,
) -> TeacherTargets:
    """Run the teacher and capture ``(value, grad)`` at ``z``.

    ``teacher_value_fn`` is ``pixel_texture_disc(decode_grad(z))`` at the
    call site (contract 1). It must build a graph back to its input, so
    the trainer must NOT wrap it in ``no_grad``.

    ``value`` is the per-sample mean patch logit (:func:`reduce_patch_logits`).
    ``grad`` is ``autograd.grad(value.sum(), z)``: because crops are
    independent, row ``i`` equals
    ``autograd.grad(disc(decode_grad(z_i)).mean(), z_i)`` — the contract
    expression per crop, with no batch-size scaling. Both are detached.

    ``use_checkpoint`` wraps the teacher in
    ``torch.utils.checkpoint(use_reentrant=False)``, the ancestor's
    trick for the wide/full-frame call: it recomputes the decode +
    disc forward during backward instead of holding the activations.
    Roughly +30% teacher compute for several GB of transient. Because
    the teacher now fires only once every ``pix_teacher_refresh_every``
    steps, paying it is usually the right trade at full-frame widths.

    **Side effect worth knowing before building call-count telemetry:**
    with ``use_checkpoint=True`` and ``want_grad=True``, ``teacher_value_fn``
    is invoked TWICE per call to this function -- once for the checkpointed
    forward, once when checkpoint recomputes it for the backward. This is
    standard ``torch.utils.checkpoint`` behaviour, not a bug, but it means
    a raw count of teacher/disc forward invocations is not the same as a
    count of logical teacher queries whenever checkpointing is on (the
    ``build_from_config`` default). Use
    :attr:`LatentSurrogateDistiller.n_teacher_refresh` to count refreshes --
    it is incremented once per refresh regardless of this doubling. Pinned
    by ``test_checkpointed_teacher_recompute_doubles_raw_call_count``.
    """
    z_in = z.detach().clone().requires_grad_(True)

    def _value(zz: torch.Tensor) -> torch.Tensor:
        return reduce_patch_logits(teacher_value_fn(zz))

    if use_checkpoint:
        from torch.utils.checkpoint import checkpoint as _ckpt
        value = _ckpt(_value, z_in, use_reentrant=False)
    else:
        value = _value(z_in)
    if value.shape[0] != z_in.shape[0]:
        raise ValueError(
            f"teacher_value_fn returned batch {value.shape[0]} for input "
            f"batch {z_in.shape[0]}; the surrogate needs one value per "
            f"latent crop."
        )
    if want_grad:
        grad = torch.autograd.grad(
            value.sum(), z_in, create_graph=False, retain_graph=False,
        )[0].detach()
    else:
        grad = None
    return TeacherTargets(
        z=z_in.detach(),
        value=value.detach().float(),
        grad=grad,
        origin=None if origin is None else (int(origin[0]), int(origin[1])),
        step=int(current_step),
        tag=tag,
    )


# ---------------------------------------------------------------------------
# Distillation driver
# ---------------------------------------------------------------------------
class LatentSurrogateDistiller:
    """Owns the teacher refresh cadence, the two distillation targets and
    the health diagnostics.

    Kept out of the trainer so the whole mechanism is unit-testable
    without a VAE, a pixel disc or a GPU. The trainer's job reduces to:
    build the critic, build the optimizer, hand this object a
    ``teacher_value_fn``, and call :meth:`step` once per D-update.

    DDP: pass the UNWRAPPED critic. PyTorch's ``DistributedDataParallel``
    does not support double backward — the reducer's autograd hooks fire
    on the first backward, so a ``create_graph=True`` grad followed by
    ``L.backward()`` (exactly the Sobolev path) either errors or
    silently produces wrong bucket views. This is the same restriction
    that forces gradient-penalty GANs to go around DDP. The ancestor at
    ``835b1df`` ran the distillation through ``self.latent_critic_ddp``
    and so was exposed to this; here the forward is unwrapped and
    :meth:`step` all-reduces the parameter gradients by hand
    (``sync_grads=True``) to get the same averaged update. Single-rank
    behaviour is identical either way.

    Args:
        critic: the unwrapped :class:`LatentTextureCritic`.
        value_loss_weight: weight on ``MSE(critic_value, teacher_value)``.
        grad_loss_weight: weight on the Sobolev term. ``0`` disables the
            teacher-gradient computation entirely (value-only mode, the
            ancestor's ``gan_critic_grad_loss_weight=0`` branch): cheaper,
            but it drops the only guarantee that the surrogate's gradient
            field resembles the teacher's, so it is a diagnostic mode,
            not a production one.
        grad_loss_normalize: divide the Sobolev MSE by the teacher
            gradient's own mean power, turning it into a RELATIVE
            gradient error in [0, ~1]. **Default ON, and this is a
            deliberate correction to the ancestor** — see the module-level
            note ``SOBOLEV SCALING`` below. ``False`` reproduces
            ``835b1df``'s raw MSE exactly.
        dense_value_weight: optional extra term matching the critic's
            dense map to the teacher's patch map resampled onto the token
            grid. Requires ``teacher_patch_fn``. Default ``0`` = OFF, so
            the default objective is exactly the frozen contract.
        pix_teacher_refresh_every: teacher fires when
            ``current_step % N == 0`` (``N<=1`` = every step, the
            ancestor's behaviour). Between refreshes the distillation
            replays cached triples.
        cache_capacity / cache_on_cpu: see :class:`TeacherTargetCache`.
        grad_check_every: run :meth:`surrogate_grad_check` every K steps
            (``0`` = never). This is the periodic direct-vs-surrogate
            audit; it costs one extra teacher forward+grad, so keep K
            well above ``pix_teacher_refresh_every``.
        max_grad_norm: clip on the critic's parameter grads
            (``None``/``0`` = off).
        teacher_use_checkpoint: forwarded to :func:`compute_teacher_targets`.
        sync_grads: manual DDP all-reduce of critic param grads.
    """

    def __init__(
        self,
        critic: LatentTextureCritic,
        *,
        value_loss_weight: float = 1.0,
        grad_loss_weight: float = 1.0,
        grad_loss_normalize: bool = True,
        dense_value_weight: float = 0.0,
        pix_teacher_refresh_every: int = 1,
        cache_capacity: int = 8,
        cache_on_cpu: bool = False,
        grad_check_every: int = 0,
        max_grad_norm: Optional[float] = None,
        teacher_use_checkpoint: bool = False,
        sync_grads: bool = True,
    ) -> None:
        self.critic = critic
        self.value_loss_weight = float(value_loss_weight)
        self.grad_loss_weight = float(grad_loss_weight)
        self.grad_loss_normalize = bool(grad_loss_normalize)
        self.dense_value_weight = float(dense_value_weight)
        self.pix_teacher_refresh_every = max(1, int(pix_teacher_refresh_every))
        self.grad_check_every = max(0, int(grad_check_every))
        self.max_grad_norm = max_grad_norm
        self.teacher_use_checkpoint = bool(teacher_use_checkpoint)
        self.sync_grads = bool(sync_grads)
        self.cache = TeacherTargetCache(
            capacity=cache_capacity, store_on_cpu=cache_on_cpu,
        )
        # Monotone counters — the same "count it or it did not happen"
        # convention TEXTURE_GAN_DESIGN Section 7 imposes on the disc.
        self.n_teacher_refresh = 0
        self.n_replay = 0
        self.n_grad_check = 0

    # -- cadence ----------------------------------------------------------
    def should_refresh(self, current_step: int) -> bool:
        """``% N`` gate, the shape reused from ``835b1df``:2808.

        Step 0 always refreshes: the cache is empty, so there is nothing
        to replay and skipping would leave the critic untrained until
        step N.
        """
        n = self.pix_teacher_refresh_every
        if n <= 1:
            return True
        return (int(current_step) % n) == 0

    def should_grad_check(self, current_step: int) -> bool:
        k = self.grad_check_every
        if k <= 0:
            return False
        return int(current_step) > 0 and (int(current_step) % k) == 0

    # -- the distillation step -------------------------------------------
    def step(
        self,
        *,
        z_real: Optional[torch.Tensor],
        z_fake: Optional[torch.Tensor],
        teacher_value_fn: Callable[[torch.Tensor], torch.Tensor],
        current_step: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        origin_real: Optional[Sequence[int]] = None,
        origin_fake: Optional[Sequence[int]] = None,
        teacher_patch_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """One critic distillation step. Call it AFTER the pixel-disc
        update and BEFORE the generator consumes the critic.

        ``z_real`` / ``z_fake`` are the latent crops for this iteration
        (detached; the caller owns the crop plan). On a refresh step the
        teacher is run on them and the resulting triples are cached; on a
        replay step they are ignored and cached triples are used instead.

        Returns a flat ``str -> float`` log dict, prefixed
        ``train/surrogate_*`` plus the two verbatim diagnostic keys
        ``train/critic_grad_cos_sim`` and ``train/critic_disc_corr``.
        """
        logs: Dict[str, float] = {}
        refresh = self.should_refresh(current_step)
        want_grad = self.grad_loss_weight > 0

        # --- Teacher: refresh or replay ---------------------------------
        pairs: List[TeacherTargets] = []
        if refresh:
            for z, origin, tag in (
                (z_real, origin_real, "real"),
                (z_fake, origin_fake, "fake"),
            ):
                if z is None:
                    continue
                tgt = compute_teacher_targets(
                    z, teacher_value_fn,
                    current_step=current_step,
                    origin=origin,
                    tag=tag,
                    want_grad=want_grad,
                    use_checkpoint=self.teacher_use_checkpoint,
                )
                self.cache.push(tgt)
                pairs.append(tgt)
            self.n_teacher_refresh += 1
        else:
            for tag in ("real", "fake"):
                tgt = self.cache.sample(tag)
                if tgt is not None:
                    pairs.append(tgt)
            self.n_replay += 1

        logs["train/surrogate_teacher_refresh"] = 1.0 if refresh else 0.0
        logs["train/surrogate_n_teacher_refresh"] = float(self.n_teacher_refresh)
        logs["train/surrogate_n_replay"] = float(self.n_replay)
        logs["train/surrogate_cache_size"] = float(len(self.cache))
        logs["train/surrogate_target_age"] = max(
            self.cache.age("real", current_step),
            self.cache.age("fake", current_step),
        )
        if not pairs:
            # Cold cache on a replay step (only reachable if the caller
            # skipped step 0). Nothing to distil against; say so loudly
            # in telemetry rather than logging a fake zero loss.
            logs["train/surrogate_skipped"] = 1.0
            return logs
        logs["train/surrogate_skipped"] = 0.0

        if self.cache.store_on_cpu:
            # Replayed entries live on the host; bring them back to the
            # device this iteration's latents are on. (On a refresh step
            # ``pairs`` already holds the device-side tensors — ``push``
            # copies for the cache rather than moving in place.)
            ref = z_real if z_real is not None else z_fake
            device = ref.device if ref is not None else pairs[0].z.device
            pairs = [p.to(device) for p in pairs]

        # --- Critic forward (+ input grad for the Sobolev term) ---------
        # ``create_graph=True`` on the input-grad is what makes
        # ``L_grad.backward()`` a genuine second-order backward through
        # the critic — the reason the attention is hand-rolled.
        crit_vals: List[torch.Tensor] = []
        teach_vals: List[torch.Tensor] = []
        crit_grads: List[torch.Tensor] = []
        teach_grads: List[torch.Tensor] = []
        dense_terms: List[torch.Tensor] = []
        for tgt in pairs:
            z_in = tgt.z.detach().clone().requires_grad_(want_grad)
            dense = self.critic(z_in, latent_origin=tgt.origin)
            val = dense.flatten(1).mean(dim=1)
            crit_vals.append(val)
            teach_vals.append(tgt.value.to(val.dtype))
            if want_grad and tgt.grad is not None:
                g = torch.autograd.grad(
                    val.sum(), z_in, create_graph=True, retain_graph=True,
                )[0]
                crit_grads.append(g)
                teach_grads.append(tgt.grad.to(g.dtype))
            if self.dense_value_weight > 0 and teacher_patch_fn is not None:
                dense_terms.append(
                    self._dense_value_term(dense, tgt, teacher_patch_fn)
                )

        critic_val = torch.cat(crit_vals, dim=0)
        teacher_val = torch.cat(teach_vals, dim=0)
        L_value = ((critic_val - teacher_val.detach()) ** 2).mean()
        if crit_grads:
            terms = []
            n_degenerate = 0
            for cg, tg in zip(crit_grads, teach_grads):
                tg = tg.detach()
                # DEGENERATE-TARGET GUARD. An all-(or near-)zero teacher
                # gradient is a real regime, not a hypothetical: every
                # pretrained-backbone teacher in this campaign ships a
                # ZERO-INIT final linear head, so d(logit)/d(input) is
                # EXACTLY 0 until that head takes its first D-step
                # (measured: `input-grad norm=0` in the dinov2/convnext/
                # sam2 preflights). Under normalization that would divide
                # by the 1e-12 floor and hand the critic a ~1e12 loss --
                # not "numerically loud" as the docstring once put it, but
                # an instant NaN. The live ordering (teacher D-update
                # BEFORE distillation) prevents it today; this guard means
                # a future reordering degrades to "term skipped, counted"
                # instead of "run dies at step 0".
                if tg.pow(2).mean() <= 1e-10:
                    n_degenerate += 1
                    continue
                num = ((cg - tg) ** 2).mean()
                if self.grad_loss_normalize:
                    # Relative gradient error: "what fraction of the
                    # teacher's gradient POWER are we failing to
                    # reproduce". 1.0 when the critic's field is zero,
                    # 0.0 on an exact match. See the note on
                    # ``grad_loss_normalize`` in the constructor for why
                    # the unnormalized form cannot be weighted sanely.
                    num = num / (tg.pow(2).mean() + 1e-12)
                terms.append(num)
            # Regime flag, never a forgeable zero: if EVERY pair was
            # degenerate there is no Sobolev signal this step, and the
            # loss key is omitted below rather than reported as 0.0 (which
            # reads as "perfect gradient match").
            logs["train/surrogate_grad_degenerate_pairs"] = float(n_degenerate)
            if terms:
                L_grad = torch.stack(terms).mean()
            else:
                L_grad = None
        else:
            L_grad = torch.zeros((), device=critic_val.device)
        if dense_terms:
            L_dense = torch.stack(dense_terms).mean()
        else:
            L_dense = torch.zeros((), device=critic_val.device)
        L_critic = (
            self.value_loss_weight * L_value
            + self.dense_value_weight * L_dense
        )
        if L_grad is not None:
            L_critic = L_critic + self.grad_loss_weight * L_grad

        # --- Optimizer -------------------------------------------------
        grad_norm = 0.0
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            L_critic.backward()
            if self.sync_grads:
                self._all_reduce_grads()
            params = [p for p in self.critic.parameters() if p.grad is not None]
            if self.max_grad_norm is not None and self.max_grad_norm > 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    params, self.max_grad_norm,
                ))
            elif params:
                grad_norm = float(torch.nn.utils.get_total_norm(
                    [p.grad for p in params]
                ))
            optimizer.step()

        # --- Diagnostics (kept verbatim from the ancestor) --------------
        with torch.no_grad():
            cv_all = critic_val.detach().float()
            tv_all = teacher_val.detach().float()
            if cv_all.numel() >= 2:
                cv_c = cv_all - cv_all.mean()
                tv_c = tv_all - tv_all.mean()
                denom = (cv_c.norm() * tv_c.norm()).clamp_min(1e-8)
                critic_disc_corr = float((cv_c * tv_c).sum() / denom)
            else:
                critic_disc_corr = 0.0
            if crit_grads:
                cg = torch.cat([g.detach().flatten() for g in crit_grads])
                tg = torch.cat([g.detach().flatten() for g in teach_grads])
                denom = (cg.norm() * tg.norm()).clamp_min(1e-8)
                critic_grad_cos_sim = float((cg * tg).sum() / denom)
                grad_mag_ratio = float(
                    cg.norm() / tg.norm().clamp_min(1e-8)
                )
            else:
                critic_grad_cos_sim = None
                grad_mag_ratio = None

        logs["train/critic_value_loss"] = float(L_value.detach())
        logs["train/critic_total_loss"] = float(L_critic.detach())
        logs["train/critic_dense_loss"] = float(L_dense.detach())
        logs["train/critic_logit_mean"] = float(cv_all.mean())
        logs["train/disc_logit_mean"] = float(tv_all.mean())
        logs["train/critic_disc_corr"] = critic_disc_corr
        # Gradient-distillation keys are OMITTED, not zero-filled, when the
        # Sobolev term is off. See NO FORGEABLE ZEROS at the bottom of this
        # file: every one of these three has a legitimate 0.0 that means the
        # OPPOSITE of "term disabled" (perfect gradient match / orthogonal
        # fields / zero-magnitude surrogate), so a zero-fill is not a
        # missing reading, it is a WRONG one. ``surrogate_grad_distill``
        # says which regime produced this row, so an absent key can be told
        # apart from a plumbing bug.
        logs["train/surrogate_grad_distill"] = 1.0 if crit_grads else 0.0
        if crit_grads and L_grad is not None:
            logs["train/critic_grad_loss"] = float(L_grad.detach())
            logs["train/critic_grad_cos_sim"] = critic_grad_cos_sim
            logs["train/surrogate_grad_mag_ratio"] = grad_mag_ratio
        logs["train/surrogate_critic_grad_norm"] = grad_norm
        return logs

    def _dense_value_term(
        self,
        dense: torch.Tensor,
        tgt: TeacherTargets,
        teacher_patch_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Optional token-level value matching (OFF by default).

        The frozen contract's teacher value is the MEAN patch logit, so
        this term is strictly extra. It exists because researcher
        directive R2 wants spatial gradients, and matching the patch grid
        directly is a stronger spatial signal than matching one scalar.
        The teacher patch map is average-pooled onto the critic's token
        grid; pooling (not interpolation) so each token's target is the
        mean of the patch logits it actually covers.
        """
        with torch.no_grad():
            patch = teacher_patch_fn(tgt.z).float()
        if patch.dim() < 3:
            raise ValueError(
                "teacher_patch_fn must return a spatial patch map; got "
                f"{tuple(patch.shape)}."
            )
        n = dense.shape[0]
        # Fold everything but the batch into [n, 1, ph, pw].
        patch = patch.reshape(n, -1, *patch.shape[-2:]).mean(dim=1, keepdim=True)
        target = F.adaptive_avg_pool2d(patch, dense.shape[-2:])  # [n,1,Hs,Ws]
        # The critic's map has a frame axis the patch map has averaged
        # away; compare the frame-mean.
        pred = dense.mean(dim=1, keepdim=True)  # [n,1,Hs,Ws]
        return ((pred - target) ** 2).mean()

    def _all_reduce_grads(self) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return
        world = dist.get_world_size()
        if world <= 1:
            return
        for p in self.critic.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= world

    # -- the periodic direct-vs-surrogate audit (delta (c)) --------------
    def surrogate_grad_check(
        self,
        z: torch.Tensor,
        teacher_value_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        origin: Optional[Sequence[int]] = None,
        current_step: int = 0,
    ) -> Dict[str, float]:
        """Compare the gradient the GENERATOR would receive from the
        surrogate against the one it would receive from the true teacher,
        at the same ``z``.

        This is the honest readout for the whole work package. The
        distillation's own ``critic_grad_cos_sim`` is measured on the
        samples the critic was just fit to, so it is a training metric;
        this one can be pointed at the live fake latent, and it directly
        answers "would swapping the teacher for the surrogate have
        changed the generator's update?".

        Returns cosine similarity, magnitude ratio and relative L2 error.
        A cos sim that drifts toward 0 while ``critic_grad_cos_sim``
        stays high means the surrogate has overfit the cached refresh
        samples — lower ``pix_teacher_refresh_every``.
        """
        self.n_grad_check += 1
        tgt = compute_teacher_targets(
            z, teacher_value_fn,
            current_step=current_step,
            origin=origin,
            tag="check",
            want_grad=True,
            use_checkpoint=self.teacher_use_checkpoint,
        )
        z_in = z.detach().clone().requires_grad_(True)
        was_training = self.critic.training
        self.critic.eval()
        try:
            val = self.critic(z_in, latent_origin=tgt.origin).flatten(1).mean(dim=1)
            g_sur = torch.autograd.grad(val.sum(), z_in)[0].detach()
        finally:
            if was_training:
                self.critic.train()
        g_true = tgt.grad
        with torch.no_grad():
            a = g_sur.flatten().float()
            b = g_true.flatten().float()
            denom = (a.norm() * b.norm()).clamp_min(1e-8)
            cos = float((a * b).sum() / denom)
            ratio = float(a.norm() / b.norm().clamp_min(1e-8))
            rel_err = float((a - b).norm() / b.norm().clamp_min(1e-8))
        return {
            "train/surrogate_check_cos_sim": cos,
            "train/surrogate_check_mag_ratio": ratio,
            "train/surrogate_check_rel_err": rel_err,
            "train/surrogate_n_grad_check": float(self.n_grad_check),
        }


# ---------------------------------------------------------------------------
# Generator consumption (frozen-critic idiom) + the two-stage warmup
# ---------------------------------------------------------------------------
def generator_surrogate_loss(
    critic: LatentTextureCritic,
    z_fake_grad: torch.Tensor,
    *,
    latent_origin: Optional[Sequence[int]] = None,
    weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """``weight * (-critic(z).mean())``, with the critic's parameters
    frozen for the duration of the forward.

    The freeze is the idiom lifted from the live action critic
    (``trainer/causal_action_forcing_train.py``:3038-3211, the
    ``critic_for_guidance.requires_grad_(False)`` / ``try`` / ``finally``
    block) and from the ancestor at ``01ea13d^``:5109-5155. Its purpose:
    the generator's backward must not write gradients into the critic's
    parameters, because the critic was ALREADY updated this iteration by
    :meth:`LatentSurrogateDistiller.step` and its optimizer must see only
    the distillation gradient. ``try/finally`` because an exception in
    the forward would otherwise leave the critic permanently frozen and
    silently stop distillation.

    Pass the UNWRAPPED critic — routing this through DDP would fire a
    second set of reducer hooks inside the generator's backward. The
    gradient we want flows through ``z_fake_grad`` into the generator's
    own DDP backward.
    """
    critic.requires_grad_(False)
    try:
        dense = critic(z_fake_grad, latent_origin=latent_origin)
        gan_main = -dense.mean()
        loss = weight * gan_main.to(z_fake_grad.dtype)
    finally:
        critic.requires_grad_(True)
    logs = {
        "train/surrogate_g_value": float(dense.detach().mean()),
        "train/surrogate_g_main": float(gan_main.detach()),
        "train/surrogate_g_weighted": float(weight) * float(gan_main.detach()),
        "train/surrogate_g_weight": float(weight),
    }
    return loss, logs


def two_stage_gen_weight(
    current_step: int,
    *,
    critic_warmup_steps: int,
    gen_warmup_steps: int,
    gan_loss_weight: float,
    shape_fn: Optional[Callable[[float], float]] = None,
) -> float:
    """The two-stage warmup from ``01ea13d^``:5109-5155, extracted so the
    trainer wiring is one call and the schedule is unit-testable.

    Stage 1 — ``current_step < critic_warmup_steps``: weight 0. The
    surrogate is being fit; a surrogate that has not converged onto the
    teacher's gradient field would push the generator in an arbitrary
    direction, and the zero-init head means it pushes with zero
    magnitude anyway.

    Stage 2 — the next ``gen_warmup_steps``: linear (or ``shape_fn``)
    ramp 0 -> ``gan_loss_weight``.

    Thereafter: ``gan_loss_weight``.
    """
    cw = int(critic_warmup_steps)
    gw = int(gen_warmup_steps)
    if current_step < cw:
        return 0.0
    if gw > 0 and current_step < (cw + gw):
        t_norm = (current_step - cw) / max(1, gw)
        ramp = float(shape_fn(t_norm)) if shape_fn is not None else float(t_norm)
        return ramp * float(gan_loss_weight)
    return float(gan_loss_weight)


# ---------------------------------------------------------------------------
# SOBOLEV SCALING — why ``grad_loss_normalize`` defaults to True
# ---------------------------------------------------------------------------
# The two distillation targets do not live on the same scale, and the gap
# is large and systematic, not a tuning detail.
#
# The teacher's value is a MEAN patch logit: O(1) by construction, because
# a logit is O(1). Its gradient w.r.t. a single latent element is that
# same O(1) response divided across the crop — each element moves the
# mean patch logit by roughly ``1 / (number of patches)`` times a local
# sensitivity. So ``L_value ~ v^2 = O(1)`` while
# ``L_grad ~ (dv/dz)^2 = O(1/P^2)`` elementwise.
#
# Measured on this module's own analytic teachers at the small test crop
# (F=2, C=16, 8x8 = 2048 latent elements):
#
#     L_value / L_grad  ~=  683   (local conv+tanh patch teacher)
#     L_value / L_grad  ~= 2957   (linear field teacher)
#
# and the ratio GROWS with crop size — the production crop
# (F=3, C=16, 24x32 = 36,864 elements) is an order of magnitude larger
# again.
#
# The ancestor shipped ``gan_critic_grad_loss_weight: 1.0``
# (``835b1df:configs/action_forcing_phase1.yaml``:704, unchanged through
# ``01ea13d^``:696). At weight 1.0 the Sobolev term therefore contributed
# well under 1% of the critic's loss: the gradient-distillation term in
# the deleted implementation was, in effect, decorative, and the critic
# was trained by value distillation alone. That matters, because value
# distillation alone provably does not pin the gradient field (see
# ``testing/test_latent_texture_critic.py::
# test_value_only_distillation_does_not_get_the_gradient_field``), and the
# gradient field is the only thing the generator consumes.
#
# Restoring the mechanism faithfully therefore could NOT mean restoring
# the weight verbatim. Two ways to fix it:
#
#   (a) raise ``grad_loss_weight`` to ~1e3-1e5 — but the right value then
#       depends on crop size, patch count and teacher calibration, so it
#       silently changes meaning whenever ``pix_crop_lat`` or the disc's
#       head does, and it has to be re-bracketed every time;
#   (b) normalize the term by the teacher gradient's own mean power, so
#       it becomes a dimensionless RELATIVE gradient error in [0, ~1]:
#       1.0 when the critic's field is zero, 0.0 on an exact match.
#
# (b) is the default here. ``grad_loss_weight=1.0`` then honestly means
# "weigh value error and relative gradient error equally", and it keeps
# that meaning across crop sizes and teacher rescalings.
# ``grad_loss_normalize=False`` reproduces the ancestor's raw MSE for
# anyone who wants the exact historical objective.


# ---------------------------------------------------------------------------
# NO FORGEABLE ZEROS — why the gradient keys are omitted, not zero-filled
# ---------------------------------------------------------------------------
# The ancestor emitted ``critic_grad_cos_sim = 0.0`` in value-only mode with
# the comment "n/a in value-only mode", and this module reproduced that at
# first. It is wrong, and it is the same class of defect as the
# ``SOBOLEV SCALING`` note above: telemetry that cannot be read back.
#
# The problem is that 0.0 is a legitimate, meaningful value for each of these
# three keys, and in every case it means something very different from
# "the term is switched off":
#
#   critic_grad_loss        0.0 = the surrogate matches the teacher's
#                                 gradient field EXACTLY (under
#                                 grad_loss_normalize, the best attainable
#                                 score). Zero-filling when the term is off
#                                 reports a perfect result for a computation
#                                 that never ran.
#   critic_grad_cos_sim     0.0 = the surrogate's gradient field is
#                                 ORTHOGONAL to the teacher's — a
#                                 catastrophic surrogate. Zero-filling makes
#                                 a healthy value-only run indistinguishable
#                                 from a broken Sobolev run.
#   surrogate_grad_mag_ratio 0.0 = the surrogate hands the generator a
#                                 zero-magnitude gradient — an inert critic.
#
# So the keys are omitted when ``grad_loss_weight == 0``, and
# ``train/surrogate_grad_distill`` (1.0/0.0) records which regime the row
# came from, so an absent key can be distinguished from a plumbing failure.
#
# The general rule, which came out of the WP-PIXGAN/WP-SURROGATE exchange in
# ``docs/GAN_REDESIGN_TWO.md`` and applies well beyond this file: a
# diagnostic must never be given a placeholder value that is inside its own
# meaningful range. Omit it, or use a sentinel outside the range. The
# corollary that motivated the exchange is that telemetry should report a
# term's SHARE of the gradient, not only its VALUE — a term can be present,
# wired, logged and contributing ~1% of what its weight implies, and nothing
# in a value-only readout will say so.


# ---------------------------------------------------------------------------
# build_from_config — the ONE place trainer config keys are read
# ---------------------------------------------------------------------------
def build_from_config(
    cfg: Any,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[
    Optional["LatentTextureCritic"],
    Optional[torch.optim.Optimizer],
    Optional["LatentSurrogateDistiller"],
]:
    """Construct ``(critic, optimizer, distiller)`` from a trainer config.

    This function exists so the eventual trainer wiring is one call
    rather than a hand-written, re-typed construction:

        self.latent_texture_critic, self.latent_critic_optimizer, \\
            self.latent_texture_distiller = build_from_config(
                self.config, device=self.device,
            )

    Centralizing the config -> object translation here closes exactly
    the seam class this package has twice been bitten by (§5b): B1's
    ``pix_finish_grad_enabled`` was read off two different objects and
    assigned by neither; this module's own ``latent_origin`` was tested
    at both ends and never across. With a single function as the only
    reader of ``surrogate_*`` / ``pix_teacher_refresh_every``, there is
    one seam instead of N, and it is unit-tested below against a stub
    config object — no trainer, VAE or GPU required. When the trainer
    lock frees, the wiring diff is the four-line call above plus an
    ``is None`` check; there is no second place for a key to go missing.

    Returns ``(None, None, None)`` when
    ``cfg.surrogate_critic_enabled`` is falsy (the default), so the
    trainer's build branch is a single unconditional call plus one
    ``is None`` check rather than an ``if`` wrapping a multi-line
    construction — the shape that hides a dropped key.

    Every default below matches the config block proposed in
    ``docs/WP_SURROGATE.md`` §4.5. ``getattr`` with a default (rather
    than a required key) is deliberate: the whole point of "default
    OFF, byte-identical when off" is that a config file predating this
    package's config block must build the exact same ``(None, None,
    None)`` as one that includes it with ``surrogate_critic_enabled:
    false`` — nothing here should require the block to be present at
    all as long as the gate reads falsy.
    """
    if not bool(getattr(cfg, "surrogate_critic_enabled", False)):
        return None, None, None
    critic = LatentTextureCritic(
        in_channels=16,
        d_model=int(getattr(cfg, "surrogate_critic_d_model", 512)),
        num_blocks=int(getattr(cfg, "surrogate_critic_num_blocks", 4)),
        num_heads=int(getattr(cfg, "surrogate_critic_num_heads", 8)),
        max_frames=int(getattr(cfg, "surrogate_critic_max_frames", 64)),
    )
    if device is not None:
        critic = critic.to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=float(getattr(cfg, "surrogate_critic_lr", 2e-4)),
        betas=(0.0, 0.9),
    )
    distiller = LatentSurrogateDistiller(
        critic,
        value_loss_weight=float(
            getattr(cfg, "surrogate_value_loss_weight", 1.0)
        ),
        grad_loss_weight=float(
            getattr(cfg, "surrogate_grad_loss_weight", 1.0)
        ),
        grad_loss_normalize=bool(
            getattr(cfg, "surrogate_grad_loss_normalize", True)
        ),
        pix_teacher_refresh_every=int(
            getattr(cfg, "pix_teacher_refresh_every", 4)
        ),
        cache_capacity=int(getattr(cfg, "surrogate_cache_capacity", 8)),
        grad_check_every=int(
            getattr(cfg, "surrogate_grad_check_every", 100)
        ),
        teacher_use_checkpoint=bool(
            getattr(cfg, "surrogate_teacher_use_checkpoint", True)
        ),
    )
    return critic, optimizer, distiller
