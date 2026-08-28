"""Direct latent-gradient surrogate for the pixel discriminator.

The original :mod:`model.latent_texture_critic` student represents a scalar
potential and is trained with a Sobolev (double-backward) objective.  That is
the mathematically conservative construction, but the 2026-08-27 VGG D5x
calibration showed that it simply shrinks toward zero on the real teacher:
after 120 fitting steps its in-sample gradient cosine was 0.063 and its
held-out cosine was 0.056.

This module implements the synthetic-gradient alternative.  It predicts the
teacher's latent gradient vector directly, trains with an ordinary first-order
loss, and supplies that detached vector to the generator through a linear
surrogate loss.  The expensive teacher remains exactly the same
``VAE.decode -> LADD pixel discriminator`` closure and fires at the same
cadence; only the student representation and fitting objective change.

The optional ``surrogate_pixel_condition_enabled`` arm additionally gives the
student a detached RGB max/min rendering guide aligned to the latent grid.
It is conditioning only: the generator still receives the predicted field
through the same linear loss, with no decoder graph on that route.

The gate lives in ``surrogate_gradient_mode: direct``.  The historical
potential student remains the default, so existing arms are byte-identical.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from model.latent_texture_critic import (
    TeacherTargets,
    TeacherTargetCache,
    compute_teacher_targets,
)

__all__ = [
    "LatentGradientPredictor",
    "DirectGradientDistiller",
    "generator_direct_gradient_loss",
    "pool_decoded_pixels_to_latents",
]


def _gn(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class _SpatialResidual(nn.Module):
    """Per-frame spatial residual block with a configurable dilation."""

    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        self.norm1 = _gn(width)
        self.conv1 = nn.Conv2d(
            width, width, 3, padding=dilation, dilation=dilation,
        )
        self.norm2 = _gn(width)
        self.conv2 = nn.Conv2d(
            width, width, 3, padding=dilation, dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class _TemporalResidual(nn.Module):
    """Cheap temporal mixing at fixed latent spatial resolution."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm1 = _gn(width)
        self.conv1 = nn.Conv3d(
            width, width, kernel_size=(3, 1, 1), padding=(1, 0, 0),
        )
        self.norm2 = _gn(width)
        self.conv2 = nn.Conv3d(
            width, width, kernel_size=(3, 1, 1), padding=(1, 0, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class _GlobalContextResidual(nn.Module):
    """Condition the local field on clip-global latent moments.

    The VGG statistic teacher pools feature means, standard deviations and
    covariances over the whole decoded crop.  A purely local convolution has
    no inexpensive way to recover those sufficient statistics.  Mean and RMS
    latent context are projected to a broadcast residual, letting every field
    location depend on the state of the complete clip/crop.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2 * width, width),
            nn.SiLU(),
            nn.Linear(width, 2 * width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,F,H,W].  RMS is used instead of std so the statistic stays
        # well-defined for a one-frame/one-cell diagnostic input.
        # Use the unnormalised hidden state. Group-normalising first largely
        # erases precisely the clip-level style/contrast moments this branch
        # exists to expose.
        mean = x.mean(dim=(2, 3, 4))
        rms = x.pow(2).mean(dim=(2, 3, 4)).sqrt()
        scale, shift = self.proj(torch.cat((mean, rms), dim=1)).chunk(2, dim=1)
        scale = 0.1 * torch.tanh(scale)[:, :, None, None, None]
        shift = shift[:, :, None, None, None]
        return x * (1.0 + scale) + shift


def pool_decoded_pixels_to_latents(
    pixels: torch.Tensor,
    *,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
) -> torch.Tensor:
    """Make a detached, spatially aligned RGB extrema guide.

    Wan's self-seeded decoder emits a fixed group of pixel frames for each
    latent frame and expands each spatial latent cell into a local pixel
    footprint.  We reduce each exact temporal group and each spatial
    footprint back onto the latent grid.  Both extrema are retained:
    ``max(rgb)`` alone discards dark texture, while ``min(rgb)`` is exactly
    ``-max(-rgb)`` and costs only three more conditioning channels.

    The result is ``[B, F_lat, 6, H_lat, W_lat]`` ordered as RGB maxima then
    RGB minima.  This helper deliberately refuses temporal interpolation:
    a non-integral decoder expansion would make the supposed latent/pixel
    correspondence ambiguous and should be fixed at the call site instead
    of silently resized.
    """
    if pixels.dim() != 5:
        raise ValueError(
            "pixel conditioning expects [B,F,3,H,W], got "
            f"{tuple(pixels.shape)}"
        )
    b, pixel_frames, channels, pixel_h, pixel_w = pixels.shape
    latent_frames = int(latent_frames)
    latent_height = int(latent_height)
    latent_width = int(latent_width)
    if channels != 3:
        raise ValueError(f"pixel conditioning requires RGB, got C={channels}")
    if latent_frames <= 0 or pixel_frames % latent_frames != 0:
        raise ValueError(
            f"decoded F={pixel_frames} is not an integral expansion of "
            f"latent F={latent_frames}"
        )
    if pixel_h < latent_height or pixel_w < latent_width:
        raise ValueError(
            f"decoded spatial size {pixel_h}x{pixel_w} cannot be pooled to "
            f"larger latent size {latent_height}x{latent_width}"
        )
    temporal_scale = pixel_frames // latent_frames
    px = pixels.detach().float().reshape(
        b * pixel_frames, channels, pixel_h, pixel_w,
    )
    bright = F.adaptive_max_pool2d(
        px, (latent_height, latent_width),
    ).reshape(
        b, latent_frames, temporal_scale, channels,
        latent_height, latent_width,
    ).amax(dim=2)
    dark = -F.adaptive_max_pool2d(
        -px, (latent_height, latent_width),
    ).reshape(
        b, latent_frames, temporal_scale, channels,
        latent_height, latent_width,
    ).amax(dim=2)
    guide = torch.cat((bright, dark), dim=2).contiguous().detach()
    if guide.requires_grad:  # defensive: detach is part of the contract.
        raise RuntimeError("pixel conditioning unexpectedly retained a graph")
    return guide


class LatentGradientPredictor(nn.Module):
    """Predict ``d teacher_value / d latent`` at latent resolution.

    The baseline folds frames into the batch and shares one dilated 2-D
    convolutional predictor.  Optional temporal residuals model the Wan VAE's
    cross-frame decoder Jacobian, while optional clip-global moment context
    models the pooled-statistic VGG teacher.  Both additions are default-off,
    leaving historical arms byte-identical.

    The output is a *unit-RMS direction field*.  A synchronized EMA of the
    real teacher gradient RMS restores the teacher's physical scale when the
    generator consumes it.  Separating direction and scale avoids an
    ill-conditioned raw MSE on teacher elements that are individually tiny.
    """

    predicts_gradient = True

    def __init__(
        self,
        *,
        in_channels: int = 16,
        width: int = 96,
        num_blocks: int = 6,
        max_latent_h: int = 64,
        max_latent_w: int = 112,
        head_init_std: float = 1.0e-3,
        teacher_rms_beta: float = 0.95,
        pixel_condition_channels: int = 0,
        teacher_feature_channels: int = 0,
        temporal_mixing: bool = False,
        temporal_blocks: int = 2,
        global_context: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.width = int(width)
        self.d_model = self.width  # compatibility with trainer build echo
        self.num_blocks = int(num_blocks)
        self.max_latent_h = int(max_latent_h)
        self.max_latent_w = int(max_latent_w)
        self.teacher_rms_beta = float(teacher_rms_beta)
        self.pixel_condition_channels = int(pixel_condition_channels)
        self.teacher_feature_channels = int(teacher_feature_channels)
        self.temporal_mixing = bool(temporal_mixing)
        self.temporal_blocks_count = max(1, int(temporal_blocks))
        self.global_context_enabled = bool(global_context)
        if self.pixel_condition_channels < 0:
            raise ValueError("pixel_condition_channels must be non-negative")
        if self.teacher_feature_channels < 0:
            raise ValueError("teacher_feature_channels must be non-negative")

        # Two coordinate channels preserve the old potential student's
        # absolute-position contract without forcing a crop into a full-size
        # canvas.  The current trainer passes a batch-shared mean crop origin.
        self.input = nn.Conv2d(self.in_channels + 2, self.width, 3, padding=1)
        dilation_cycle = (1, 2, 4, 8, 4, 2, 1)
        self.blocks = nn.ModuleList([
            _SpatialResidual(self.width, dilation_cycle[i % len(dilation_cycle)])
            for i in range(self.num_blocks)
        ])
        self.out_norm = _gn(self.width)
        self.output = nn.Conv2d(self.width, self.in_channels, 3, padding=1)
        if head_init_std > 0:
            nn.init.trunc_normal_(
                self.output.weight,
                std=float(head_init_std),
                a=-2.0 * float(head_init_std),
                b=2.0 * float(head_init_std),
            )
        else:
            nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

        # A separate zero-initialised projection is intentional. Construct
        # it only after every baseline random parameter, so a fixed seed
        # produces the identical latent trunk whether this opt-in branch is
        # present or absent. The branch starts as an exact no-op and receives
        # first-order gradients immediately.
        self.pixel_condition_input: Optional[nn.Conv2d]
        if self.pixel_condition_channels > 0:
            self.pixel_condition_input = nn.Conv2d(
                self.pixel_condition_channels,
                self.width,
                3,
                padding=1,
                bias=False,
            )
            nn.init.zeros_(self.pixel_condition_input.weight)
        else:
            self.pixel_condition_input = None

        # Construct opt-in modules after every historical trainable tensor.
        # With both flags off, no new RNG is consumed and the old state dict
        # and initialization remain byte-identical.
        self.temporal_blocks = nn.ModuleList([
            _TemporalResidual(self.width)
            for _ in range(self.temporal_blocks_count)
        ]) if self.temporal_mixing else nn.ModuleList()
        self.global_context = (
            _GlobalContextResidual(self.width)
            if self.global_context_enabled else None
        )

        # Benchmark-only rich evidence branch.  Raw pooled feature statistics
        # and the current scalar head's derivative live on radically different
        # scales, so normalise them separately before one zero-initialised
        # projection.  Construct this after every historical module: enabling
        # it cannot perturb any existing parameter's seeded initialisation.
        self.teacher_feature_input: Optional[nn.Linear]
        if self.teacher_feature_channels > 0:
            self.teacher_feature_input = nn.Linear(
                2 * self.teacher_feature_channels, self.width, bias=False,
            )
            nn.init.zeros_(self.teacher_feature_input.weight)
        else:
            self.teacher_feature_input = None

        # Buffer, not a Python distiller field: it is checkpointed and copied
        # into the frozen generator snapshot with the predictor weights.
        self.register_buffer("teacher_grad_rms_ema", torch.zeros(()))
        self.register_buffer("teacher_grad_rms_updates", torch.zeros(()))

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _coords(
        self,
        h: int,
        w: int,
        *,
        origin: Optional[Sequence[int]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        oy = int(origin[0]) if origin is not None else 0
        ox = int(origin[1]) if origin is not None else 0
        if oy < 0 or ox < 0 or oy + h > self.max_latent_h or ox + w > self.max_latent_w:
            raise ValueError(
                f"latent window y[{oy}:{oy+h}] x[{ox}:{ox+w}] exceeds "
                f"direct-gradient coordinate extent "
                f"{self.max_latent_h}x{self.max_latent_w}"
            )
        yy = torch.arange(oy, oy + h, device=device, dtype=dtype)
        xx = torch.arange(ox, ox + w, device=device, dtype=dtype)
        yy = yy / max(1, self.max_latent_h - 1) * 2.0 - 1.0
        xx = xx / max(1, self.max_latent_w - 1) * 2.0 - 1.0
        ygrid = yy[:, None].expand(h, w)
        xgrid = xx[None, :].expand(h, w)
        return torch.stack((ygrid, xgrid), dim=0)

    def _batch_coords(
        self,
        b: int,
        frames: int,
        h: int,
        w: int,
        *,
        origin: Optional[Sequence[int]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        per_sample = (
            origin is not None
            and len(origin) > 0
            and isinstance(origin[0], (list, tuple, torch.Tensor))
        )
        if not per_sample:
            return self._coords(
                h, w, origin=origin, device=device, dtype=dtype,
            ).unsqueeze(0).expand(b * frames, -1, -1, -1)
        if len(origin) != b:
            raise ValueError(
                f"per-sample latent origins must have B={b} rows, got "
                f"{len(origin)}"
            )
        grids = torch.stack([
            self._coords(
                h, w, origin=item, device=device, dtype=dtype,
            )
            for item in origin
        ], dim=0)
        return grids[:, None].expand(b, frames, 2, h, w).reshape(
            b * frames, 2, h, w,
        )

    def forward(
        self,
        latent: torch.Tensor,
        latent_origin: Optional[Sequence[int]] = None,
        pixel_condition: Optional[torch.Tensor] = None,
        teacher_stats_condition: Optional[torch.Tensor] = None,
        teacher_head_grad_condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latent.dim() != 5:
            raise ValueError(
                f"LatentGradientPredictor expects [B,F,C,H,W], got "
                f"{tuple(latent.shape)}"
            )
        b, frames, channels, h, w = latent.shape
        if channels != self.in_channels:
            raise ValueError(
                f"in_channels={self.in_channels}, got latent C={channels}"
            )
        p = next(self.parameters())
        x = latent.to(device=p.device, dtype=p.dtype).reshape(
            b * frames, channels, h, w,
        )
        coords = self._batch_coords(
            b, frames, h, w, origin=latent_origin,
            device=x.device, dtype=x.dtype,
        )
        x = self.input(torch.cat((x, coords), dim=1))
        if self.pixel_condition_channels > 0:
            if pixel_condition is None:
                raise ValueError(
                    "pixel-conditioned predictor requires pixel_condition"
                )
            expected = (b, frames, self.pixel_condition_channels, h, w)
            if tuple(pixel_condition.shape) != expected:
                raise ValueError(
                    f"pixel_condition must have shape {expected}, got "
                    f"{tuple(pixel_condition.shape)}"
                )
            if pixel_condition.requires_grad:
                raise ValueError(
                    "pixel_condition must be detached; decoder gradients are "
                    "not part of the surrogate generator route"
                )
            cond = pixel_condition.to(
                device=x.device, dtype=x.dtype,
            ).reshape(b * frames, self.pixel_condition_channels, h, w)
            x = x + self.pixel_condition_input(cond)
        elif pixel_condition is not None:
            raise ValueError(
                "pixel_condition was supplied to an unconditioned predictor"
            )
        teacher_conditions = (
            teacher_stats_condition, teacher_head_grad_condition,
        )
        if self.teacher_feature_channels > 0:
            if any(item is None for item in teacher_conditions):
                raise ValueError(
                    "teacher-feature predictor requires both pooled stats "
                    "and head-gradient conditions"
                )
            expected = (b, frames, self.teacher_feature_channels)
            names = ("teacher_stats_condition", "teacher_head_grad_condition")
            checked = []
            for name, item in zip(names, teacher_conditions):
                assert item is not None
                if tuple(item.shape) != expected:
                    raise ValueError(
                        f"{name} must have shape {expected}, got "
                        f"{tuple(item.shape)}"
                    )
                if item.requires_grad:
                    raise ValueError(
                        f"{name} must be detached; teacher feature/head "
                        "graphs are not part of the surrogate route"
                    )
                checked.append(item.to(device=x.device, dtype=x.dtype))
            # Parameter-free LayerNorm keeps the evidence definition exact and
            # prevents raw VGG covariance scale from drowning the head field.
            normalised = [
                F.layer_norm(item, (self.teacher_feature_channels,))
                for item in checked
            ]
            guide = self.teacher_feature_input(
                torch.cat(normalised, dim=-1)
            ).reshape(b * frames, self.width, 1, 1)
            x = x + guide
        elif any(item is not None for item in teacher_conditions):
            raise ValueError(
                "teacher feature evidence was supplied to an unconditioned "
                "predictor"
            )
        for block in self.blocks:
            x = block(x)
        if self.temporal_mixing or self.global_context is not None:
            x = x.reshape(b, frames, self.width, h, w).permute(
                0, 2, 1, 3, 4,
            ).contiguous()
            for block in self.temporal_blocks:
                x = block(x)
            if self.global_context is not None:
                x = self.global_context(x)
            x = x.permute(0, 2, 1, 3, 4).reshape(
                b * frames, self.width, h, w,
            )
        x = self.output(F.silu(self.out_norm(x)))
        return x.reshape(b, frames, channels, h, w).float()

    @torch.no_grad()
    def update_teacher_rms(self, rms: torch.Tensor) -> None:
        value = rms.detach().float().mean().to(self.teacher_grad_rms_ema.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            value /= dist.get_world_size()
        if int(self.teacher_grad_rms_updates.item()) == 0:
            self.teacher_grad_rms_ema.copy_(value)
        else:
            beta = self.teacher_rms_beta
            self.teacher_grad_rms_ema.mul_(beta).add_(value, alpha=1.0 - beta)
        self.teacher_grad_rms_updates.add_(1)

    def gradient_for_generator(
        self,
        latent: torch.Tensor,
        latent_origin: Optional[Sequence[int]] = None,
        pixel_condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        direction = self(
            latent,
            latent_origin=latent_origin,
            pixel_condition=pixel_condition,
        )
        dims = tuple(range(1, direction.dim()))
        pred_rms = direction.pow(2).mean(dim=dims, keepdim=True).sqrt()
        unit = direction / pred_rms.clamp_min(1.0e-8)
        # Before the first teacher refresh the honest output is zero, not a
        # unit random field at an invented scale.
        scale = self.teacher_grad_rms_ema.float().clamp_min(0.0)
        return unit * scale


class DirectGradientDistiller:
    """First-order distiller with the scalar student's public interface."""

    def __init__(
        self,
        critic: LatentGradientPredictor,
        *,
        pix_teacher_refresh_every: int = 1,
        cache_capacity: int = 8,
        cache_on_cpu: bool = False,
        grad_check_every: int = 0,
        max_grad_norm: Optional[float] = None,
        teacher_use_checkpoint: bool = False,
        sync_grads: bool = True,
        distill_substeps: int = 1,
        teacher_target_microbatch: int = 0,
        loss_mode: str = "mse",
        real_loss_weight: float = 1.0,
        fake_loss_weight: float = 1.0,
    ) -> None:
        self.critic = critic
        # Compatibility attributes used by the trainer's resolved build log.
        self.value_loss_weight = 0.0
        self.grad_loss_weight = 1.0
        self.grad_loss_normalize = True
        self.pix_teacher_refresh_every = max(1, int(pix_teacher_refresh_every))
        self.grad_check_every = max(0, int(grad_check_every))
        self.max_grad_norm = max_grad_norm
        self.teacher_use_checkpoint = bool(teacher_use_checkpoint)
        self.sync_grads = bool(sync_grads)
        self.distill_substeps = max(1, int(distill_substeps))
        # 0 = historical all-at-once target graph. A positive value only
        # serializes teacher graph construction; returned targets are
        # concatenated back into the identical batch before fitting.
        self.teacher_target_microbatch = max(0, int(teacher_target_microbatch))
        self.loss_mode = str(loss_mode).strip().lower()
        if self.loss_mode not in ("mse", "cosine"):
            raise ValueError(
                "direct surrogate loss_mode must be 'mse' or 'cosine'; "
                f"got {loss_mode!r}"
            )
        self.loss_weights = {
            "real": max(0.0, float(real_loss_weight)),
            "fake": max(0.0, float(fake_loss_weight)),
        }
        if not any(value > 0.0 for value in self.loss_weights.values()):
            raise ValueError("at least one direct surrogate target weight must be > 0")
        self.cache = TeacherTargetCache(
            capacity=cache_capacity, store_on_cpu=cache_on_cpu,
        )
        self.n_teacher_refresh = 0
        self.n_replay = 0
        self.n_grad_check = 0
        self.n_distill_substeps = 0
        self._check_sample_q1_history: List[float] = []

    def _teacher_targets(
        self,
        z: torch.Tensor,
        teacher_value_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        current_step: int,
        origin: Optional[Sequence[int]],
        tag: str,
        condition: Optional[torch.Tensor] = None,
    ) -> TeacherTargets:
        micro = self.teacher_target_microbatch
        if micro <= 0 or int(z.shape[0]) <= micro:
            target = compute_teacher_targets(
                z, teacher_value_fn, current_step=current_step,
                origin=origin, tag=tag, want_grad=True,
                use_checkpoint=self.teacher_use_checkpoint,
            )
            target.condition = (
                None if condition is None else condition.detach()
            )
            return target
        pieces = []
        for start in range(0, int(z.shape[0]), micro):
            piece_origin = origin
            if (
                origin is not None
                and len(origin) > 0
                and isinstance(origin[0], (list, tuple, torch.Tensor))
            ):
                piece_origin = origin[start:start + micro]
            pieces.append(compute_teacher_targets(
                z[start:start + micro], teacher_value_fn,
                current_step=current_step, origin=piece_origin, tag=tag,
                want_grad=True,
                use_checkpoint=self.teacher_use_checkpoint,
            ))
        joined_origin = pieces[0].origin
        if (
            joined_origin is not None
            and len(joined_origin) > 0
            and isinstance(joined_origin[0], (list, tuple, torch.Tensor))
        ):
            joined_origin = tuple(
                item for piece in pieces for item in piece.origin
            )
        return TeacherTargets(
            z=torch.cat([p.z for p in pieces], dim=0),
            value=torch.cat([p.value for p in pieces], dim=0),
            grad=torch.cat([p.grad for p in pieces], dim=0),
            origin=joined_origin,
            step=int(current_step),
            tag=tag,
            condition=None if condition is None else condition.detach(),
        )

    def should_refresh(self, current_step: int) -> bool:
        return int(current_step) % self.pix_teacher_refresh_every == 0

    def should_grad_check(self, current_step: int) -> bool:
        return (
            self.grad_check_every > 0
            and int(current_step) > 0
            and int(current_step) % self.grad_check_every == 0
        )

    def _all_reduce_grads(self) -> None:
        if not (self.sync_grads and dist.is_available() and dist.is_initialized()):
            return
        world = dist.get_world_size()
        if world <= 1:
            return
        for p in self.critic.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= world

    @staticmethod
    def _normalise_target(grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        grad = grad.float()
        dims = tuple(range(1, grad.dim()))
        rms = grad.pow(2).mean(dim=dims, keepdim=True).sqrt().clamp_min(1.0e-12)
        return grad / rms, rms

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
        condition_real: Optional[torch.Tensor] = None,
        condition_fake: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        del teacher_patch_fn
        logs: Dict[str, float] = {}
        refresh = self.should_refresh(current_step)
        pairs: List[TeacherTargets] = []
        if refresh:
            for z, origin, tag, condition in (
                (z_real, origin_real, "real", condition_real),
                (z_fake, origin_fake, "fake", condition_fake),
            ):
                if z is None or self.loss_weights[tag] <= 0.0:
                    continue
                tgt = self._teacher_targets(
                    z, teacher_value_fn, current_step=current_step,
                    origin=origin, tag=tag, condition=condition,
                )
                self.cache.push(tgt)
                pairs.append(tgt)
            self.n_teacher_refresh += 1
        else:
            for tag in ("real", "fake"):
                if self.loss_weights[tag] <= 0.0:
                    continue
                tgt = self.cache.sample(tag)
                if tgt is not None:
                    pairs.append(tgt)
            self.n_replay += 1

        logs["train/surrogate_teacher_refresh"] = float(refresh)
        logs["train/surrogate_n_teacher_refresh"] = float(self.n_teacher_refresh)
        logs["train/surrogate_n_replay"] = float(self.n_replay)
        logs["train/surrogate_cache_size"] = float(len(self.cache))
        logs["train/surrogate_target_age"] = max(
            self.cache.age("real", current_step),
            self.cache.age("fake", current_step),
        )
        logs["train/surrogate_direct_gradient_mode"] = 1.0
        logs["train/surrogate_direct_loss_is_cosine"] = float(
            self.loss_mode == "cosine"
        )
        logs["train/surrogate_real_target_weight"] = self.loss_weights["real"]
        logs["train/surrogate_fake_target_weight"] = self.loss_weights["fake"]
        logs["train/surrogate_pixel_condition_active"] = float(
            self.critic.pixel_condition_channels > 0
        )
        logs["train/surrogate_pixel_condition_channels"] = float(
            self.critic.pixel_condition_channels
        )
        if not pairs:
            logs["train/surrogate_skipped"] = 1.0
            return logs
        logs["train/surrogate_skipped"] = 0.0

        if self.cache.store_on_cpu:
            ref = z_real if z_real is not None else z_fake
            device = ref.device if ref is not None else pairs[0].z.device
            pairs = [p.to(device) for p in pairs]

        rms_values = []
        for tgt in pairs:
            if tgt.grad is not None:
                _, rms = self._normalise_target(tgt.grad)
                rms_values.append(rms.mean())
        if rms_values:
            self.critic.update_teacher_rms(torch.stack(rms_values))

        sub_device = pairs[0].z.device
        last_pred: List[torch.Tensor] = []
        last_target: List[torch.Tensor] = []
        last_loss = torch.zeros((), device=sub_device)
        grad_norm = 0.0
        for substep in range(self.distill_substeps):
            if substep > 0:
                redrawn = []
                for tag in ("real", "fake"):
                    if self.loss_weights[tag] <= 0.0:
                        continue
                    tgt = self.cache.sample(tag)
                    if tgt is not None:
                        redrawn.append(tgt)
                if not redrawn:
                    break
                pairs = redrawn
                if self.cache.store_on_cpu:
                    pairs = [p.to(sub_device) for p in pairs]

            self.n_distill_substeps += 1
            preds: List[torch.Tensor] = []
            targets: List[torch.Tensor] = []
            for tgt in pairs:
                if tgt.grad is None:
                    continue
                pred = self.critic(
                    tgt.z.detach(),
                    latent_origin=tgt.origin,
                    pixel_condition=tgt.condition,
                )
                target, _ = self._normalise_target(tgt.grad)
                preds.append(pred)
                targets.append(target.to(pred.device, pred.dtype))
            if not preds:
                break
            pair_losses = []
            pair_weights = []
            for tgt, pred, target in zip(pairs, preds, targets):
                target = target.detach()
                if self.loss_mode == "cosine":
                    pair_loss = 1.0 - F.cosine_similarity(
                        pred.float().flatten(1),
                        target.float().flatten(1),
                        dim=1,
                        eps=1.0e-6,
                    ).mean()
                else:
                    pair_loss = (pred - target).pow(2).mean()
                pair_losses.append(pair_loss)
                pair_weights.append(self.loss_weights.get(tgt.tag, 1.0))
            weight_sum = max(sum(pair_weights), 1.0e-12)
            last_loss = sum(
                loss * weight
                for loss, weight in zip(pair_losses, pair_weights)
            ) / weight_sum
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                last_loss.backward()
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
            last_pred = preds
            last_target = targets

        with torch.no_grad():
            pred_flat = torch.cat([p.detach().flatten() for p in last_pred])
            target_flat = torch.cat([t.detach().flatten() for t in last_target])
            denom = (pred_flat.norm() * target_flat.norm()).clamp_min(1.0e-8)
            cos = float((pred_flat * target_flat).sum() / denom)
            mag = float(pred_flat.norm() / target_flat.norm().clamp_min(1.0e-8))
        logs["train/critic_total_loss"] = float(last_loss.detach())
        logs["train/critic_grad_loss"] = float(last_loss.detach())
        logs["train/critic_grad_cos_sim"] = cos
        logs["train/critic_disc_corr"] = cos
        logs["train/surrogate_grad_mag_ratio"] = mag
        logs["train/surrogate_grad_distill"] = 1.0
        logs["train/surrogate_critic_grad_norm"] = grad_norm
        logs["train/surrogate_teacher_grad_rms_ema"] = float(
            self.critic.teacher_grad_rms_ema,
        )
        logs["train/surrogate_distill_substeps"] = float(self.distill_substeps)
        logs["train/surrogate_n_distill_substeps"] = float(self.n_distill_substeps)
        calls = self.n_teacher_refresh + self.n_replay
        logs["train/surrogate_substeps_achieved"] = (
            float(self.n_distill_substeps) / calls if calls else 0.0
        )
        condition_branch = self.critic.pixel_condition_input
        if condition_branch is not None:
            logs["train/surrogate_pixel_condition_weight_norm"] = float(
                condition_branch.weight.detach().float().norm()
            )
            condition_grad = condition_branch.weight.grad
            logs["train/surrogate_pixel_condition_grad_norm"] = (
                0.0 if condition_grad is None
                else float(condition_grad.detach().float().norm())
            )
            # Functional proof, not just a wiring/parameter proof: measure
            # how much the fitted field changes when its rendered guide is
            # zeroed on the exact same cached target. This is telemetry only
            # and never enters the optimizer loss.
            field_delta_sq = []
            ablation_loss_delta = []
            with torch.no_grad():
                for tgt in pairs:
                    if tgt.grad is None or tgt.condition is None:
                        continue
                    full = self.critic(
                        tgt.z.detach(), latent_origin=tgt.origin,
                        pixel_condition=tgt.condition,
                    )
                    ablated = self.critic(
                        tgt.z.detach(), latent_origin=tgt.origin,
                        pixel_condition=torch.zeros_like(tgt.condition),
                    )
                    target, _ = self._normalise_target(tgt.grad)
                    target = target.to(full.device, full.dtype)
                    field_delta_sq.append((full - ablated).pow(2).mean())
                    ablation_loss_delta.append(
                        (ablated - target).pow(2).mean()
                        - (full - target).pow(2).mean()
                    )
            if field_delta_sq:
                logs["train/surrogate_pixel_condition_field_delta_rms"] = float(
                    torch.stack(field_delta_sq).mean().sqrt()
                )
                logs[
                    "train/surrogate_pixel_condition_ablation_loss_delta"
                ] = float(torch.stack(ablation_loss_delta).mean())
        return logs

    def surrogate_grad_check(
        self,
        z: torch.Tensor,
        teacher_value_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        origin: Optional[Sequence[int]] = None,
        current_step: int = 0,
        pixel_condition: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        self.n_grad_check += 1
        tgt = self._teacher_targets(
            z, teacher_value_fn, current_step=current_step, origin=origin,
            tag="check",
        )
        was_training = self.critic.training
        self.critic.eval()
        try:
            with torch.no_grad():
                g_pred = self.critic.gradient_for_generator(
                    z.detach(), latent_origin=tgt.origin,
                    pixel_condition=pixel_condition,
                )
        finally:
            if was_training:
                self.critic.train()
        with torch.no_grad():
            a_sample = g_pred.flatten(1).float()
            b_sample = tgt.grad.flatten(1).float()
            sample_cos_local = F.cosine_similarity(
                a_sample, b_sample, dim=1, eps=1.0e-8,
            )
            sample_cos = sample_cos_local
            if dist.is_available() and dist.is_initialized():
                gathered = [
                    torch.empty_like(sample_cos_local)
                    for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered, sample_cos_local)
                sample_cos = torch.cat(gathered)
            a = a_sample.flatten()
            b = b_sample.flatten()
            totals = torch.stack((
                (a * b).sum(), a.square().sum(), b.square().sum(),
                (a - b).square().sum(),
            ))
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            a_norm = totals[1].sqrt()
            b_norm = totals[2].sqrt()
            denom = (a_norm * b_norm).clamp_min(1.0e-8)
            cos = float(totals[0] / denom)
            ratio = float(a_norm / b_norm.clamp_min(1.0e-8))
            rel_err = float(totals[3].sqrt() / b_norm.clamp_min(1.0e-8))
            sample_q1 = float(torch.quantile(sample_cos, 0.25))
            self._check_sample_q1_history.append(sample_q1)
            self._check_sample_q1_history = self._check_sample_q1_history[-2:]
            rolling_q1_min = min(self._check_sample_q1_history)
        return {
            # Keep the historical concatenated score for continuity, but do
            # not gate on it: high-norm/easy rows can dominate it. Generator
            # consumption normalises each sample independently, so q1 is the
            # conservative decision-facing statistic.
            "train/surrogate_check_cos_sim": cos,
            "train/surrogate_check_cos_sample_median": float(
                sample_cos.median()
            ),
            "train/surrogate_check_cos_sample_q1": sample_q1,
            "train/surrogate_check_cos_sample_min": float(sample_cos.min()),
            "train/surrogate_check_cos_sample_n": float(sample_cos.numel()),
            "train/surrogate_check_cos_sample_q1_rolling2_min": float(
                rolling_q1_min
            ),
            "train/surrogate_check_cos_sample_q1_rolling2_count": float(
                len(self._check_sample_q1_history)
            ),
            "train/surrogate_check_distributed_ranks": float(
                dist.get_world_size()
                if dist.is_available() and dist.is_initialized() else 1
            ),
            "train/surrogate_check_mag_ratio": ratio,
            "train/surrogate_check_rel_err": rel_err,
            "train/surrogate_n_grad_check": float(self.n_grad_check),
        }


def generator_direct_gradient_loss(
    critic: LatentGradientPredictor,
    z_fake_grad: torch.Tensor,
    *,
    latent_origin: Optional[Sequence[int]] = None,
    weight: float = 1.0,
    pixel_condition: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Linear synthetic-gradient loss whose derivative is ``-g_pred``."""

    with torch.no_grad():
        field = critic.gradient_for_generator(
            z_fake_grad.detach(), latent_origin=latent_origin,
            pixel_condition=pixel_condition,
        ).detach()
    gan_main = -(
        z_fake_grad.float() * field.to(z_fake_grad.device).float()
    ).flatten(1).sum(dim=1).mean()
    loss = float(weight) * gan_main
    logs = {
        "train/surrogate_g_main": float(gan_main.detach()),
        "train/surrogate_g_weighted": float(weight) * float(gan_main.detach()),
        "train/surrogate_g_weight": float(weight),
        "train/surrogate_g_field_rms": float(field.float().pow(2).mean().sqrt()),
        "train/surrogate_direct_gradient_mode": 1.0,
        "train/surrogate_pixel_condition_active": float(
            critic.pixel_condition_channels > 0
        ),
        "train/surrogate_pixel_condition_channels": float(
            critic.pixel_condition_channels
        ),
    }
    return loss, logs
