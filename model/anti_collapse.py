"""Anti-collapse losses for the DMD student.

Two regimes are available, selected by ``anti_collapse_type`` on the
model:

  * ``"std_floor"`` (legacy): one-sided std-deficit penalty
        ``loss = ReLU(s_gt - s_pred)^2``
    plus an optional mean anchor. Lives in
    ``DMD._compute_anti_collapse_term`` and is kept as the historical
    fallback.

  * ``"std_corridor"`` (new): bidirectional log-ratio corridor on
    per-frame std + optional mean anchor + chunk-to-chunk drift
    penalty. Tracks GT's distributional moments without forcing
    point-wise matching. See ``latent_moment_corridor_loss`` and
    ``latent_contrast_drift_loss`` below for details and the rationale.

Both helpers consume per-frame moments at axes (2,3,4) of a
``[B, F, C, H, W]`` latent tensor; both detach ``gt_target`` so no
gradient flows into GT. They return a single ``[1]`` scalar that the
trainer can sum into ``generator_loss`` directly.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def latent_moment_corridor_loss(
    pred_x0: torch.Tensor,
    gt_target: torch.Tensor,
    std_low: float = 0.85,
    std_high: float = 1.15,
    rms_low: float = 0.85,
    rms_high: float = 1.15,
    mean_tol_ratio: float = 0.05,
    use_mean: bool = True,
    use_rms: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Bidirectional log-ratio corridor on per-frame moments.

    Penalty kicks in only when (per-frame) ``pred_x0`` moments drift
    outside their respective bands relative to GT. Inside the band the
    gradient is exactly zero — the model is free to find its natural
    variance/mean/RMS without being pulled toward a point estimate.

    Args:
        pred_x0:   ``[B, F, C, H, W]`` student rollout (graph-attached).
        gt_target: ``[B, F, C, H, W]`` GT latent at matching positions
            (detached internally).
        std_low, std_high: multiplicative bounds on the std ratio
            ``s_pred / s_gt``. Default ``[0.85, 1.15]`` — slightly wider
            than the original 0.90/1.10 to avoid spurious early-training
            firing.
        rms_low, rms_high: same idea for RMS (zero-aware energy proxy).
        mean_tol_ratio: tolerance on per-frame mean difference, expressed
            as a fraction of GT's std. ``|m_pred - m_gt| > mean_tol_ratio
            * s_gt`` triggers the penalty.
        use_mean / use_rms: opt-out flags for the mean / RMS terms.
        eps: numerical floor for the log ratio.

    Returns:
        scalar tensor ``[]``.
    """
    gt = gt_target.detach()
    reduce_dims = [2, 3, 4]

    # Per-frame std (unbiased=False matches the standard moment convention
    # for closed-form variance estimates over fixed-size windows).
    s_pred = pred_x0.std(dim=reduce_dims, unbiased=False)
    s_gt = gt.std(dim=reduce_dims, unbiased=False)

    ratio_std = torch.log((s_pred + eps) / (s_gt + eps))
    loss_std_under = F.relu(math.log(std_low) - ratio_std) ** 2
    loss_std_over = F.relu(ratio_std - math.log(std_high)) ** 2
    loss_std = (loss_std_under + loss_std_over).mean()
    loss = loss_std

    if use_mean:
        m_pred = pred_x0.mean(dim=reduce_dims)
        m_gt = gt.mean(dim=reduce_dims)
        # Tolerance is set in *GT std* units so a near-flat GT frame
        # gets a small tolerance and a busy GT frame gets a larger one.
        mean_tol = mean_tol_ratio * s_gt.detach()
        mean_err = (m_pred - m_gt).abs()
        loss_mean = (F.relu(mean_err - mean_tol) ** 2).mean()
        loss = loss + 0.25 * loss_mean

    if use_rms:
        rms_pred = torch.sqrt((pred_x0 ** 2).mean(dim=reduce_dims) + eps)
        rms_gt = torch.sqrt((gt ** 2).mean(dim=reduce_dims) + eps)
        ratio_rms = torch.log((rms_pred + eps) / (rms_gt + eps))
        loss_rms_under = F.relu(math.log(rms_low) - ratio_rms) ** 2
        loss_rms_over = F.relu(ratio_rms - math.log(rms_high)) ** 2
        loss_rms = (loss_rms_under + loss_rms_over).mean()
        loss = loss + 0.25 * loss_rms

    return loss


def latent_contrast_drift_loss(
    pred_x0: torch.Tensor,
    gt_target: torch.Tensor,
    num_frame_per_block: int,
    drift_tol: float = 0.02,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Symmetric chunk-to-chunk drift penalty on ``log(s_pred / s_gt)``.

    Computes the per-frame std ratio in log space, averages within each
    ``num_frame_per_block``-frame chunk, then penalises absolute drift
    of that ratio between consecutive chunks beyond ``drift_tol``. This
    directly targets AR drift (e.g. each chunk getting brighter than
    the last) without pulling the student toward a point estimate.

    Symmetric on purpose: positive drift = brightening, negative =
    dimming. Both modes are AR failure modes worth penalising.

    Args:
        pred_x0:   ``[B, F, C, H, W]`` student rollout.
        gt_target: ``[B, F, C, H, W]`` GT at matching positions.
        num_frame_per_block: chunk size in frames (= ``self.num_frame_per
            _block``). The function derives ``num_chunks = F //
            num_frame_per_block`` at runtime — no hard-coded number.
        drift_tol: max allowed absolute drift in log-ratio space
            between consecutive chunks. Default 0.02 (~2% multiplicative
            change in std-ratio per chunk).
        eps: numerical floor.

    Returns:
        scalar tensor ``[]``. Zero when there are < 2 chunks
        (cannot compute differences).
    """
    gt = gt_target.detach()
    B, F_total = pred_x0.shape[:2]
    if num_frame_per_block <= 0:
        return pred_x0.new_zeros(())
    num_chunks = F_total // num_frame_per_block
    if num_chunks < 2:
        return pred_x0.new_zeros(())

    s_pred = pred_x0.std(dim=[2, 3, 4], unbiased=False)
    s_gt = gt.std(dim=[2, 3, 4], unbiased=False)
    ratio = torch.log((s_pred + eps) / (s_gt + eps))

    # Take the first num_chunks * num_frame_per_block frames (drop any
    # trailing remainder) and average within each chunk.
    ratio_chunk = ratio[:, : num_chunks * num_frame_per_block]
    ratio_chunk = ratio_chunk.view(B, num_chunks, num_frame_per_block).mean(
        dim=2
    )
    drift = ratio_chunk[:, 1:] - ratio_chunk[:, :-1]
    # Symmetric: penalise drift of either sign beyond drift_tol.
    return (F.relu(drift.abs() - drift_tol) ** 2).mean()


def latent_std_mse_loss(
    pred_x0: torch.Tensor,
    gt_target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Simple MSE between per-frame std of pred_x0 and GT.

    No corridor, no log-ratio, no chunk drift — just
    ``(s_pred - s_gt) ** 2`` averaged over (batch, frame). Pulls the
    student's per-frame latent std to *match* GT, both up and down.
    Symmetric by construction: shrinking under GT and ballooning over
    GT are penalised the same way. The simplest possible anti-collapse
    signal.

    Args:
        pred_x0:   ``[B, F, C, H, W]`` student rollout (graph-attached).
        gt_target: ``[B, F, C, H, W]`` GT at matching positions
            (detached internally).
        eps: numerical floor — currently unused on the difference but
            kept for parity with the corridor helpers.

    Returns:
        scalar tensor ``[]``.
    """
    gt = gt_target.detach()
    reduce_dims = [2, 3, 4]
    s_pred = pred_x0.std(dim=reduce_dims, unbiased=False)
    s_gt = gt.std(dim=reduce_dims, unbiased=False)
    return (s_pred - s_gt).pow(2).mean()


def latent_std_graded_mse_constant_loss(
    pred_x0: torch.Tensor,
    target_std: float = 0.875,
    mse_floor: float = 0.015,
) -> torch.Tensor:
    """MSE between per-frame std and a fixed scalar target, with a
    minimum-floor clamp that zeroes the gradient once the student is
    close enough.

    Loss = ``clamp(mean((s_pred - target_std) ** 2), min=mse_floor)``

    The clamp is the key part: once the running MSE drops below
    ``mse_floor`` the loss becomes the constant ``mse_floor`` (no input
    dependency, so the gradient is exactly zero). The student is free
    to roam inside the implicit ``|s_pred - target_std| < sqrt(floor)``
    band; only attempts to drift further get pulled back.

    Compared to ``latent_std_mse_loss`` (MSE vs GT std), this:
      * Doesn't track GT's natural per-step variation — it's a
        regulariser pinned to a chosen value.
      * Has a "close enough" zone via the floor — prevents the loss
        from fighting micro-deviations once the student is in the
        target band.
      * Acts as a barrier against BOTH zero-power collapse (gray)
        AND infinite-power collapse, since deviation in either
        direction grows the MSE quadratically.

    Args:
        pred_x0: ``[B, F, C, H, W]`` student rollout (graph-attached).
        target_std: scalar target for the per-frame std (default 0.875,
            empirically near GT's typical per-frame std mean of ~0.85
            on Wan latents).
        mse_floor: minimum loss value below which the gradient is
            clamped to zero. Default 0.015 = student is "close enough"
            when the per-frame std mean is within ~sqrt(0.015) = 0.12
            of the target.

    Returns:
        scalar tensor ``[]``.
    """
    reduce_dims = [2, 3, 4]
    s_pred = pred_x0.std(dim=reduce_dims, unbiased=False)
    mse = (s_pred - float(target_std)).pow(2).mean()
    return torch.clamp(mse, min=float(mse_floor))


def _causal_cumulative_mean(x: torch.Tensor) -> torch.Tensor:
    """Cumulative-mean along dim=1. ``x: [B, F]`` -> ``[B, F]`` where
    ``out[:, t] = x[:, :t+1].mean(dim=1)``. Equivalent to a causal
    rolling mean with W >= F (the window covers everything seen so far).
    """
    cs = x.cumsum(dim=1)
    weights = torch.arange(
        1, x.shape[1] + 1, device=x.device, dtype=x.dtype,
    )
    # Broadcast the [F] weights against dim=1 for any rank >= 2
    # (e.g. [B, F] STD/M2/TV/SOS or [B, F, C] M1).
    shape = [1] * x.dim()
    shape[1] = x.shape[1]
    return cs / weights.view(shape)


def _per_frame_M2(x: torch.Tensor) -> torch.Tensor:
    """Per-frame M2 = ``Σ_c σ_c²`` over spatial (H, W).
    ``x: [B, F, C, H, W]`` -> ``[B, F]``."""
    sigma = x.std(dim=[3, 4], unbiased=False)        # [B, F, C]
    return (sigma ** 2).sum(dim=-1)                   # [B, F]


def _per_frame_SOS(x: torch.Tensor) -> torch.Tensor:
    """Per-frame raw sum-of-squares ``Σ_{C,H,W} x²`` (NOT mean-subtracted —
    this is the raw second moment / energy, unlike M2/STD which measure
    variance). ``x: [B, F, C, H, W]`` -> ``[B, F]``."""
    return (x ** 2).sum(dim=[2, 3, 4])


def _per_frame_M1(x: torch.Tensor) -> torch.Tensor:
    """Per-frame, per-channel raw sum-of-squares ``Σ_{H,W} x²`` — summed
    over the spatial dims (H, W) but NOT over channels, so each channel's
    energy is anchored independently (catches a single channel collapsing
    while others compensate, which the channel-pooled SOS cannot see).
    ``x: [B, F, C, H, W]`` -> ``[B, F, C]``."""
    return (x ** 2).sum(dim=[3, 4])


def _per_frame_TV(x: torch.Tensor) -> torch.Tensor:
    """Per-frame total-variation (latent space).

    Defined as the per-channel mean of |Δx| along W *and* H, summed
    over channels:

        TV(x) = Σ_c [ mean_{H,W} |x[c,h,w] - x[c,h,w-1]|
                   + mean_{H,W} |x[c,h,w] - x[c,h-1,w]| ]

    ``x: [B, F, C, H, W]`` -> ``[B, F]``.
    """
    dx_w = (x[..., 1:] - x[..., :-1]).abs().mean(dim=[3, 4])      # [B, F, C]
    dx_h = (x[..., 1:, :] - x[..., :-1, :]).abs().mean(dim=[3, 4])  # [B, F, C]
    return (dx_w + dx_h).sum(dim=-1)                                  # [B, F]


def _per_frame_STD(x: torch.Tensor) -> torch.Tensor:
    """Per-frame global std over C, H, W. ``x: [B, F, C, H, W]`` -> ``[B, F]``.
    Matches ``latent_std_graded_mse_constant_loss`` semantics: a single
    scalar per frame measuring overall latent variance (all channels and
    spatial positions treated as samples).
    """
    return x.std(dim=[2, 3, 4], unbiased=False)


def _per_frame_stable_rank(
    x: torch.Tensor, eps: float = 1e-12,
) -> torch.Tensor:
    """Per-frame stable rank ``||X||_F^2 / sigma_max(X)^2``.

    For each frame we reshape the latent ``[C, H, W]`` to a matrix
    ``[C, H*W]`` and compute its stable rank (Frobenius norm squared
    over operator norm squared). Stable rank sits between 1 (a
    single dominant direction) and ``min(C, H*W)`` (uniform spectrum).
    For Wan latents with 16 channels and typical spatial resolution
    the matrix is fat (C << H*W); eigvalsh on the C x C gram matrix
    is cheap.

    ``x: [B, F, C, H, W]`` -> ``[B, F]``.
    """
    B, F_, C, H, W = x.shape
    mat = x.reshape(B * F_, C, H * W)
    # eigvalsh on the C x C gram matrix gives squared singular values
    # — same as full SVD but much cheaper for fat matrices. fp32 for
    # numerical stability of the eigendecomposition.
    mat_f = mat.float()
    gram = torch.matmul(mat_f, mat_f.transpose(-1, -2))   # [B*F, C, C]
    eig = torch.linalg.eigvalsh(gram).clamp(min=0.0)      # [B*F, C]
    sigma_sq_sum = eig.sum(dim=-1)                        # [B*F]
    sigma_sq_max = eig[..., -1]                           # [B*F]
    stable_rank = sigma_sq_sum / (sigma_sq_max + eps)
    return stable_rank.reshape(B, F_).to(x.dtype)


def compute_stat_anchor_loss(
    pred_x0: torch.Tensor,
    seed_latents: Optional[torch.Tensor] = None,
    *,
    seed_STD_anchor: Optional[torch.Tensor] = None,
    seed_M2_anchor: Optional[torch.Tensor] = None,
    seed_TV_anchor: Optional[torch.Tensor] = None,
    seed_SOS_anchor: Optional[torch.Tensor] = None,
    seed_M1_anchor: Optional[torch.Tensor] = None,
    STD_short_weight: float = 0.1,
    STD_long_weight: float = 0.1,
    M2_short_weight: float = 0.1,
    M2_long_weight: float = 0.1,
    TV_short_weight: float = 0.1,
    TV_long_weight: float = 0.1,
    SOS_short_weight: float = 0.0,
    SOS_long_weight: float = 0.0,
    M1_short_weight: float = 0.0,
    M1_long_weight: float = 0.0,
    rel_tol_short: float = 0.20,
    rel_tol_long: float = 0.10,
    # Optional long-horizon anchor overrides — scalar tensors. When
    # provided, the long-horizon comparison uses these instead of the
    # per-batch seed anchors (which are still used for short-horizon).
    # Typical caller: an EMA of cross-rank-averaged seed anchors, so
    # the long-horizon anchor approaches the GT-population mean and
    # the rollout is free to self-correct from an edge-seed start.
    long_STD_anchor_override: Optional[torch.Tensor] = None,
    long_M2_anchor_override: Optional[torch.Tensor] = None,
    long_TV_anchor_override: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    """Seed-anchored, MSE-with-floor regulariser on three latent summary
    stats — per-frame STD, M2 (= Σ_c σ_c²) and TV (= Σ_c mean|Δx|) —
    measured at two horizons (per-frame short + causal cumulative-mean
    long). Replacement for the v11 ``std_graded_mse_constant`` and the
    v12 hinge-style ``compute_stat_anchor_loss``.

    For each stat ``S`` with seed-derived per-batch anchor ``A``:
      * ``mse_short = mean_{B,F} (S_pf - A)^2``
      * ``mse_long  = mean_{B,F} (cumavg_t(S_pf) - A)^2``
      * ``floor_X   = (rel_tol_X * mean_b(A))^2``  (one scalar per stat)
      * ``loss_X    = clamp(mse_X - floor_X, min=0)`` — equivalent to
        ``clamp(mse, min=floor) - floor``; same gradient as
        ``clamp(mse, min=floor)`` but a zero baseline when in-band so
        the gen-loss curve is clean.

    The floor is sized so deviations within ``rel_tol * anchor`` (e.g.
    20% of the anchor) produce zero gradient. Beyond that, the loss
    behaves like a standard MSE pulling the per-frame stat back toward
    the anchor — symmetric (both upward and downward drift penalised).
    Long horizon uses a tighter ``rel_tol_long`` because cumulative
    averaging smooths out per-frame noise.

    The anchor comes from the cf-prefix seed the student was conditioned
    on (so it changes per ride/prompt — there is no fixed global target).

    Args:
        pred_x0: ``[B, F, C, H, W]`` student rollout (graph-attached).
        seed_latents: ``[B, F_seed, C, H, W]`` seed-window latents. If
            provided, per-stat anchors are the mean of the seed's
            per-frame stat values. Mutually exclusive with the explicit
            ``seed_*_anchor`` kwargs.
        seed_STD_anchor, seed_M2_anchor, seed_TV_anchor: precomputed
            ``[B]`` anchors. Use these if computed once upstream.
        STD_short_weight, STD_long_weight, M2_short_weight, M2_long_weight,
        TV_short_weight, TV_long_weight: per-stat per-horizon multipliers
            on the floor-clamped MSE. Outer ``stat_anchor_loss_weight``
            multiplies the total.
        rel_tol_short, rel_tol_long: tolerance band as a fraction of
            anchor. Drift within ``rel_tol * anchor`` produces zero
            gradient (see floor formula above). Default 0.20 / 0.10.

    Returns:
        ``(loss_scalar, log_dict)`` where ``loss_scalar`` is a scalar
        tensor graph-attached to ``pred_x0`` (zero when fully in-band)
        and ``log_dict`` is a flat ``str -> detached-tensor`` mapping
        for wandb.
    """
    if pred_x0.dim() != 5:
        raise ValueError(
            f"compute_stat_anchor_loss expects [B, F, C, H, W]; got "
            f"{tuple(pred_x0.shape)}"
        )

    if seed_latents is not None:
        if seed_latents.dim() != 5:
            raise ValueError(
                "compute_stat_anchor_loss: seed_latents must be "
                f"[B, F_seed, C, H, W]; got {tuple(seed_latents.shape)}"
            )
        with torch.no_grad():
            seed = seed_latents.detach()
            STD_anchor = _per_frame_STD(seed).mean(dim=1)      # [B]
            M2_anchor = _per_frame_M2(seed).mean(dim=1)        # [B]
            TV_anchor = _per_frame_TV(seed).mean(dim=1)        # [B]
            SOS_anchor = _per_frame_SOS(seed).mean(dim=1)      # [B]
            M1_anchor = _per_frame_M1(seed).mean(dim=1)        # [B, C]
    else:
        if (
            seed_STD_anchor is None
            or seed_M2_anchor is None
            or seed_TV_anchor is None
        ):
            raise ValueError(
                "compute_stat_anchor_loss: must provide either "
                "seed_latents OR all three precomputed seed_*_anchor "
                "tensors (STD, M2, TV)."
            )
        STD_anchor = seed_STD_anchor.detach()
        M2_anchor = seed_M2_anchor.detach()
        TV_anchor = seed_TV_anchor.detach()
        # SOS/M1 optional in the precomputed path — None disables the term.
        SOS_anchor = (
            seed_SOS_anchor.detach() if seed_SOS_anchor is not None else None
        )
        M1_anchor = (
            seed_M1_anchor.detach() if seed_M1_anchor is not None else None
        )

    # Per-frame stats on pred (graph-attached).
    STD_pf = _per_frame_STD(pred_x0)                   # [B, F]
    M2_pf = _per_frame_M2(pred_x0)
    TV_pf = _per_frame_TV(pred_x0)

    # Causal cumulative mean for long-horizon signal.
    STD_long = _causal_cumulative_mean(STD_pf)
    M2_long = _causal_cumulative_mean(M2_pf)
    TV_long = _causal_cumulative_mean(TV_pf)

    # Anchor broadcast — SHORT horizon uses per-batch seed anchor.
    a_STD_short = STD_anchor.unsqueeze(1).to(STD_pf.dtype)   # [B, 1]
    a_M2_short = M2_anchor.unsqueeze(1).to(M2_pf.dtype)
    a_TV_short = TV_anchor.unsqueeze(1).to(TV_pf.dtype)

    # LONG-horizon anchor — uses override (e.g. cross-rank EMA of GT
    # population stats) when provided, else falls back to the same
    # per-batch seed anchor as the short horizon. Override is a
    # scalar tensor that broadcasts over [B, F].
    if long_STD_anchor_override is not None:
        a_STD_long = long_STD_anchor_override.detach().to(STD_pf.dtype)
        STD_a2_long = float(
            long_STD_anchor_override.detach().pow(2).item()
        )
    else:
        a_STD_long = a_STD_short
        STD_a2_long = None  # populated below from per-batch anchor
    if long_M2_anchor_override is not None:
        a_M2_long = long_M2_anchor_override.detach().to(M2_pf.dtype)
        M2_a2_long = float(
            long_M2_anchor_override.detach().pow(2).item()
        )
    else:
        a_M2_long = a_M2_short
        M2_a2_long = None
    if long_TV_anchor_override is not None:
        a_TV_long = long_TV_anchor_override.detach().to(TV_pf.dtype)
        TV_a2_long = float(
            long_TV_anchor_override.detach().pow(2).item()
        )
    else:
        a_TV_long = a_TV_short
        TV_a2_long = None

    # Raw MSE values (scalar each).
    mse_STD_short = (STD_pf - a_STD_short).pow(2).mean()
    mse_STD_long = (STD_long - a_STD_long).pow(2).mean()
    mse_M2_short = (M2_pf - a_M2_short).pow(2).mean()
    mse_M2_long = (M2_long - a_M2_long).pow(2).mean()
    mse_TV_short = (TV_pf - a_TV_short).pow(2).mean()
    mse_TV_long = (TV_long - a_TV_long).pow(2).mean()

    # Floors. ``(rel_tol * anchor_mean)^2`` — one scalar per stat per
    # horizon. Computed on the detached anchor so the floor itself does
    # not carry gradient. Short uses per-batch seed-anchor mean;
    # long uses the override anchor when present (so the deadband
    # scales with whatever anchor the long-horizon compare is using).
    rs2 = float(rel_tol_short) ** 2
    rl2 = float(rel_tol_long) ** 2
    STD_a2 = float(STD_anchor.detach().mean().pow(2).item())
    M2_a2 = float(M2_anchor.detach().mean().pow(2).item())
    TV_a2 = float(TV_anchor.detach().mean().pow(2).item())
    if STD_a2_long is None:
        STD_a2_long = STD_a2
    if M2_a2_long is None:
        M2_a2_long = M2_a2
    if TV_a2_long is None:
        TV_a2_long = TV_a2
    floor_STD_short = rs2 * STD_a2
    floor_STD_long = rl2 * STD_a2_long
    floor_M2_short = rs2 * M2_a2
    floor_M2_long = rl2 * M2_a2_long
    floor_TV_short = rs2 * TV_a2
    floor_TV_long = rl2 * TV_a2_long

    # MSE-with-floor (subtraction form for clean baseline; gradient is
    # identical to ``clamp(mse, min=floor)``).
    loss_STD_short = (mse_STD_short - floor_STD_short).clamp(min=0)
    loss_STD_long = (mse_STD_long - floor_STD_long).clamp(min=0)
    loss_M2_short = (mse_M2_short - floor_M2_short).clamp(min=0)
    loss_M2_long = (mse_M2_long - floor_M2_long).clamp(min=0)
    loss_TV_short = (mse_TV_short - floor_TV_short).clamp(min=0)
    loss_TV_long = (mse_TV_long - floor_TV_long).clamp(min=0)

    # ------------------------------------------------------------------
    # SOS (raw Σ_{C,H,W} x²; [B,F]) and M1 (per-channel Σ_{H,W} x²;
    # [B,F,C]). Both anchored with the same floor-clamped MSE machinery.
    # Skipped (zero) when the anchor is unavailable (precomputed path
    # without seed_SOS/seed_M1 anchors) or the weight is 0. NOTE: these
    # are raw second moments — anchor magnitudes are large (sum, not
    # mean), so use small weights relative to STD/TV.
    # ------------------------------------------------------------------
    zero = pred_x0.new_zeros(())
    loss_SOS_short = loss_SOS_long = zero
    floor_SOS_short = floor_SOS_long = 0.0
    mse_SOS_short = mse_SOS_long = zero
    if SOS_anchor is not None and (
        float(SOS_short_weight) != 0.0 or float(SOS_long_weight) != 0.0
    ):
        SOS_pf = _per_frame_SOS(pred_x0)                    # [B, F]
        SOS_long = _causal_cumulative_mean(SOS_pf)          # [B, F]
        a_SOS = SOS_anchor.unsqueeze(1).to(SOS_pf.dtype)    # [B, 1]
        mse_SOS_short = (SOS_pf - a_SOS).pow(2).mean()
        mse_SOS_long = (SOS_long - a_SOS).pow(2).mean()
        SOS_a2 = float(SOS_anchor.detach().mean().pow(2).item())
        floor_SOS_short = rs2 * SOS_a2
        floor_SOS_long = rl2 * SOS_a2
        loss_SOS_short = (mse_SOS_short - floor_SOS_short).clamp(min=0)
        loss_SOS_long = (mse_SOS_long - floor_SOS_long).clamp(min=0)

    loss_M1_short = loss_M1_long = zero
    floor_M1_short = floor_M1_long = 0.0
    mse_M1_short = mse_M1_long = zero
    if M1_anchor is not None and (
        float(M1_short_weight) != 0.0 or float(M1_long_weight) != 0.0
    ):
        M1_pf = _per_frame_M1(pred_x0)                      # [B, F, C]
        M1_long = _causal_cumulative_mean(M1_pf)            # [B, F, C]
        a_M1 = M1_anchor.unsqueeze(1).to(M1_pf.dtype)       # [B, 1, C]
        mse_M1_short = (M1_pf - a_M1).pow(2).mean()
        mse_M1_long = (M1_long - a_M1).pow(2).mean()
        M1_a2 = float(M1_anchor.detach().mean().pow(2).item())
        floor_M1_short = rs2 * M1_a2
        floor_M1_long = rl2 * M1_a2
        loss_M1_short = (mse_M1_short - floor_M1_short).clamp(min=0)
        loss_M1_long = (mse_M1_long - floor_M1_long).clamp(min=0)

    loss = (
        float(STD_short_weight) * loss_STD_short
        + float(STD_long_weight) * loss_STD_long
        + float(M2_short_weight) * loss_M2_short
        + float(M2_long_weight) * loss_M2_long
        + float(TV_short_weight) * loss_TV_short
        + float(TV_long_weight) * loss_TV_long
        + float(SOS_short_weight) * loss_SOS_short
        + float(SOS_long_weight) * loss_SOS_long
        + float(M1_short_weight) * loss_M1_short
        + float(M1_long_weight) * loss_M1_long
    )

    dev = pred_x0.device
    def _t(v: float) -> torch.Tensor:
        return torch.tensor(float(v), device=dev)

    logs = {
        # Anchors
        "stat/STD_anchor": STD_anchor.mean().detach(),
        "stat/M2_anchor": M2_anchor.mean().detach(),
        "stat/TV_anchor": TV_anchor.mean().detach(),
        # Long-horizon anchor actually used (= override when provided,
        # else equal to the per-batch anchor above).
        "stat/STD_anchor_long": (
            long_STD_anchor_override.detach().mean()
            if long_STD_anchor_override is not None
            else STD_anchor.mean().detach()
        ),
        "stat/M2_anchor_long": (
            long_M2_anchor_override.detach().mean()
            if long_M2_anchor_override is not None
            else M2_anchor.mean().detach()
        ),
        "stat/TV_anchor_long": (
            long_TV_anchor_override.detach().mean()
            if long_TV_anchor_override is not None
            else TV_anchor.mean().detach()
        ),
        # Per-frame pred means
        "stat/STD_pred_mean": STD_pf.mean().detach(),
        "stat/M2_pred_mean": M2_pf.mean().detach(),
        "stat/TV_pred_mean": TV_pf.mean().detach(),
        # Raw MSEs
        "stat/STD_mse_short": mse_STD_short.detach(),
        "stat/STD_mse_long": mse_STD_long.detach(),
        "stat/M2_mse_short": mse_M2_short.detach(),
        "stat/M2_mse_long": mse_M2_long.detach(),
        "stat/TV_mse_short": mse_TV_short.detach(),
        "stat/TV_mse_long": mse_TV_long.detach(),
        # Floors (constant within batch)
        "stat/STD_floor_short": _t(floor_STD_short),
        "stat/STD_floor_long": _t(floor_STD_long),
        "stat/M2_floor_short": _t(floor_M2_short),
        "stat/M2_floor_long": _t(floor_M2_long),
        "stat/TV_floor_short": _t(floor_TV_short),
        "stat/TV_floor_long": _t(floor_TV_long),
        # Active loss above floor — what actually contributes gradient
        "stat/STD_active_short": loss_STD_short.detach(),
        "stat/STD_active_long": loss_STD_long.detach(),
        "stat/M2_active_short": loss_M2_short.detach(),
        "stat/M2_active_long": loss_M2_long.detach(),
        "stat/TV_active_short": loss_TV_short.detach(),
        "stat/TV_active_long": loss_TV_long.detach(),
        # SOS / M1 (raw sum-of-squares terms)
        "stat/SOS_anchor": (
            SOS_anchor.mean().detach() if SOS_anchor is not None else zero
        ),
        "stat/M1_anchor": (
            M1_anchor.mean().detach() if M1_anchor is not None else zero
        ),
        "stat/SOS_mse_short": mse_SOS_short.detach(),
        "stat/SOS_mse_long": mse_SOS_long.detach(),
        "stat/M1_mse_short": mse_M1_short.detach(),
        "stat/M1_mse_long": mse_M1_long.detach(),
        "stat/SOS_floor_short": _t(floor_SOS_short),
        "stat/SOS_floor_long": _t(floor_SOS_long),
        "stat/M1_floor_short": _t(floor_M1_short),
        "stat/M1_floor_long": _t(floor_M1_long),
        "stat/SOS_active_short": loss_SOS_short.detach(),
        "stat/SOS_active_long": loss_SOS_long.detach(),
        "stat/M1_active_short": loss_M1_short.detach(),
        "stat/M1_active_long": loss_M1_long.detach(),
    }
    return loss, logs


def compute_stat_anchor_target_matching_loss(
    pred_x0: torch.Tensor,
    *,
    M2_target: float = 9.0,
    M2_band_low: float = 5.0,
    M2_band_high: float = 13.0,
    M2_weight: float = 0.1,
    TV_target: float = 7.8,
    TV_band_low: float = 6.0,
    TV_band_high: float = 9.0,
    TV_weight: float = 0.1,
    rank_target: float = 1.5,
    rank_band_high: float = 2.25,
    rank_band_low: Optional[float] = None,
    rank_weight: float = 0.1,
) -> Tuple[torch.Tensor, dict]:
    """Fixed-target dead-band MSE matching on per-frame summary stats.

    For each per-frame stat ``s[t]`` (computed on the rollout), apply
    a square loss toward a fixed target ``T``, gated to fire only
    when ``s[t]`` is outside a configured pass-through band:

        loss[t] = (s[t] - T)^2   if s[t] < band_low or s[t] > band_high
                  else 0

    Means over (B, F) and across the three stats then multiplied by
    their per-stat weights. Differs from ``compute_stat_anchor_loss``:
    no seed anchor, no EMA, no relative tolerance — just absolute
    targets + absolute gating thresholds derived from offline GT
    distribution analysis.

    Stats:
      * **M2** (= Σ_c σ_c², user's "STD" target): typical Wan-1.3B
        value 6-9. Default target 9, band [5, 13] (two-sided gate).
      * **TV** (= Σ_c mean|Δx|): typical Wan-1.3B value 7-10.
        Default target 7.8, band [6, 9] (two-sided gate).
      * **stable_rank** (= ||X||_F²/σ_max² per frame): typical 1.0-2.0
        in GT. Default target 1.5, upper-only gate at 2.25 (rank
        being above the band signals over-uniform spectrum =
        collapse-precursor; rank being below 1.5 toward 1.0 just
        means the chunk has a dominant direction, which is fine).
        Set ``rank_band_low`` to enable a two-sided gate; ``None``
        keeps the upper-only contract.

    Returns ``(loss_scalar, log_dict)``. ``loss_scalar`` is graph-
    attached to ``pred_x0``; ``log_dict`` has stat values, band-
    membership rates, and per-stat active losses for wandb.
    """
    if pred_x0.dim() != 5:
        raise ValueError(
            f"compute_stat_anchor_target_matching_loss expects "
            f"[B, F, C, H, W]; got {tuple(pred_x0.shape)}"
        )

    # Per-frame stats (graph-attached on pred).
    M2_pf = _per_frame_M2(pred_x0)                  # [B, F]
    TV_pf = _per_frame_TV(pred_x0)
    rank_pf = _per_frame_stable_rank(pred_x0)

    M2_t = float(M2_target)
    TV_t = float(TV_target)
    rk_t = float(rank_target)

    # Per-stat square error, gated to zero inside the pass-through band.
    M2_sq = (M2_pf - M2_t).pow(2)
    M2_gate = (
        (M2_pf < float(M2_band_low)) | (M2_pf > float(M2_band_high))
    ).to(M2_sq.dtype)
    M2_active_pf = M2_sq * M2_gate

    TV_sq = (TV_pf - TV_t).pow(2)
    TV_gate = (
        (TV_pf < float(TV_band_low)) | (TV_pf > float(TV_band_high))
    ).to(TV_sq.dtype)
    TV_active_pf = TV_sq * TV_gate

    rank_sq = (rank_pf - rk_t).pow(2)
    if rank_band_low is not None:
        rk_gate = (
            (rank_pf < float(rank_band_low))
            | (rank_pf > float(rank_band_high))
        ).to(rank_sq.dtype)
    else:
        rk_gate = (rank_pf > float(rank_band_high)).to(rank_sq.dtype)
    rank_active_pf = rank_sq * rk_gate

    M2_loss = M2_active_pf.mean()
    TV_loss = TV_active_pf.mean()
    rank_loss = rank_active_pf.mean()

    loss = (
        float(M2_weight) * M2_loss
        + float(TV_weight) * TV_loss
        + float(rank_weight) * rank_loss
    )

    logs = {
        # Targets + band thresholds (constant within run, but logged
        # for traceability when scrubbing wandb).
        "stat/tm_M2_target": torch.tensor(M2_t, device=pred_x0.device),
        "stat/tm_M2_band_low": torch.tensor(
            float(M2_band_low), device=pred_x0.device,
        ),
        "stat/tm_M2_band_high": torch.tensor(
            float(M2_band_high), device=pred_x0.device,
        ),
        "stat/tm_TV_target": torch.tensor(TV_t, device=pred_x0.device),
        "stat/tm_TV_band_low": torch.tensor(
            float(TV_band_low), device=pred_x0.device,
        ),
        "stat/tm_TV_band_high": torch.tensor(
            float(TV_band_high), device=pred_x0.device,
        ),
        "stat/tm_rank_target": torch.tensor(rk_t, device=pred_x0.device),
        "stat/tm_rank_band_high": torch.tensor(
            float(rank_band_high), device=pred_x0.device,
        ),
        # Pred summary
        "stat/tm_M2_pred_mean": M2_pf.mean().detach(),
        "stat/tm_TV_pred_mean": TV_pf.mean().detach(),
        "stat/tm_rank_pred_mean": rank_pf.mean().detach(),
        # Out-of-band rates (= fraction of (B,F) frames triggering loss).
        "stat/tm_M2_out_of_band_rate": M2_gate.mean().detach(),
        "stat/tm_TV_out_of_band_rate": TV_gate.mean().detach(),
        "stat/tm_rank_out_of_band_rate": rk_gate.mean().detach(),
        # Per-stat active losses (post-gate, pre-weight)
        "stat/tm_M2_active": M2_loss.detach(),
        "stat/tm_TV_active": TV_loss.detach(),
        "stat/tm_rank_active": rank_loss.detach(),
    }
    return loss, logs


def compute_std_corridor_anti_collapse(
    pred_x0: torch.Tensor,
    gt_target: torch.Tensor,
    num_frame_per_block: int,
    moment_weight: float,
    drift_weight: float,
    corridor_std_low: float = 0.85,
    corridor_std_high: float = 1.15,
    corridor_rms_low: float = 0.85,
    corridor_rms_high: float = 1.15,
    corridor_mean_tol_ratio: float = 0.05,
    drift_tol: float = 0.02,
) -> Tuple[torch.Tensor, dict]:
    """Wrapper that bundles ``latent_moment_corridor_loss`` and
    ``latent_contrast_drift_loss`` with their weights and returns a
    single weighted total plus a log-friendly stats dict.

    Returns ``(weighted_total, log_stats)``. The total is detached-free
    so caller can summed into ``generator_loss``.
    """
    log_stats: dict = {}
    if moment_weight > 0.0:
        loss_moment = latent_moment_corridor_loss(
            pred_x0=pred_x0,
            gt_target=gt_target,
            std_low=corridor_std_low,
            std_high=corridor_std_high,
            rms_low=corridor_rms_low,
            rms_high=corridor_rms_high,
            mean_tol_ratio=corridor_mean_tol_ratio,
        )
        log_stats["anti_collapse_corridor_moment_raw"] = loss_moment.detach()
    else:
        loss_moment = pred_x0.new_zeros(())

    if drift_weight > 0.0:
        loss_drift = latent_contrast_drift_loss(
            pred_x0=pred_x0,
            gt_target=gt_target,
            num_frame_per_block=num_frame_per_block,
            drift_tol=drift_tol,
        )
        log_stats["anti_collapse_corridor_drift_raw"] = loss_drift.detach()
    else:
        loss_drift = pred_x0.new_zeros(())

    total = moment_weight * loss_moment + drift_weight * loss_drift
    return total, log_stats
