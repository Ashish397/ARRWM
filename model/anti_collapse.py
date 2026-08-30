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


def attenuate_stat_anchor_tail(
    loss: torch.Tensor,
    threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bound an exceptional stat-anchor tail without dropping the sample.

    The transform is exactly the identity through ``threshold`` and uses the
    continuous rational tail

    ``f(L) = 2T - T^2 / L`` for ``L > T``.

    Consequently ``f(T)=T`` and ``f'(T)=1`` (no kink in value or gradient),
    while the tail gradient is attenuated by ``(T/L)^2`` and the reported
    effective loss remains bounded below ``2T``. Unlike a hard clamp, every
    finite outlier retains a non-zero corrective gradient; unlike rejecting a
    window, DMD/GAN/CARN/action all continue to train on the same perturbation.

    ``threshold <= 0`` is the byte-for-byte legacy path. The returned tuple
    is ``(effective_loss, detached_gradient_scale, detached_active_bit)`` so
    the intervention is explicit in telemetry.
    """
    threshold = float(threshold)
    if threshold <= 0.0:
        one = loss.detach().new_ones(())
        zero = loss.detach().new_zeros(())
        return loss, one, zero
    if loss.numel() != 1:
        raise ValueError(
            "attenuate_stat_anchor_tail expects a scalar loss; got "
            f"shape={tuple(loss.shape)}"
        )

    t = loss.new_tensor(threshold)
    # Stat-anchor is a weighted sum of non-negative losses. Clamp only the
    # denominator for numerical safety; the normal branch returns ``loss``
    # itself and therefore preserves its exact graph and value.
    safe_loss = loss.clamp_min(torch.finfo(loss.dtype).tiny)
    tail = 2.0 * t - t.square() / safe_loss
    active = loss > t
    effective = torch.where(active, tail, loss)
    with torch.no_grad():
        scale = torch.where(
            active,
            (t / safe_loss.detach()).square(),
            torch.ones_like(loss.detach()),
        )
    return effective, scale.detach(), active.detach().to(loss.dtype)


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


def latent_std_energy_meanabs_mse_loss(
    pred_x0: torch.Tensor,
    gt_target: torch.Tensor,
    std_weight: float = 1.0,
    energy_weight: float = 1.0,
    meanabs_weight: float = 1.0,
    eps: float = 1e-6,
):
    """Per-frame MSE on THREE global latent statistics vs point-wise GT.

    Extends ``latent_std_mse_loss`` with an energy (RMS) term and a
    mean-absolute term so the loss matches not just the spread but also
    the overall magnitude / DC level of the latent (which a bare std,
    being mean-invariant, ignores). All three stats are GLOBAL per-frame
    (channels and spatial pooled — ``dim=[2,3,4]``), matching the
    ``latent_std_mse_loss`` convention:

      * ``std``     = ``std(x)``              — spread, mean-invariant
      * ``energy``  = ``sqrt(mean(x^2))``     — RMS magnitude (incl. DC),
        unit-consistent with std (x-units, not x^2) so the three terms
        sit on a comparable scale
      * ``meanabs`` = ``mean(|x|)``           — L1 magnitude / level

    Each term is ``(stat_pred - stat_gt)^2`` averaged over (B, F), summed
    with per-term weights. Symmetric (over/under penalised equally). GT
    is detached.

    Returns:
        ``(loss_scalar, parts)`` where ``parts`` maps
        ``{"std","energy","meanabs"} -> detached raw MSE`` for logging.
    """
    gt = gt_target.detach()
    rd = [2, 3, 4]

    s_pred = pred_x0.std(dim=rd, unbiased=False)
    s_gt = gt.std(dim=rd, unbiased=False)
    loss_std = (s_pred - s_gt).pow(2).mean()

    e_pred = pred_x0.pow(2).mean(dim=rd).clamp_min(0).sqrt()
    e_gt = gt.pow(2).mean(dim=rd).clamp_min(0).sqrt()
    loss_energy = (e_pred - e_gt).pow(2).mean()

    ma_pred = pred_x0.abs().mean(dim=rd)
    ma_gt = gt.abs().mean(dim=rd)
    loss_meanabs = (ma_pred - ma_gt).pow(2).mean()

    total = (
        std_weight * loss_std
        + energy_weight * loss_energy
        + meanabs_weight * loss_meanabs
    )
    parts = {
        "std": loss_std.detach(),
        "energy": loss_energy.detach(),
        "meanabs": loss_meanabs.detach(),
    }
    return total, parts


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


def _per_frame_MEAN(x: torch.Tensor) -> torch.Tensor:
    """Per-frame, per-channel SPATIAL MEAN ``mean_{H,W} x`` — the FIRST
    moment. ``x: [B, F, C, H, W]`` -> ``[B, F, C]``.

    NAMING TRAP (2026-08-25). ``_per_frame_M1`` above is NOT this: despite
    the "M1" name it returns per-channel ``Σ_{H,W} x²`` — a raw SECOND
    moment (per-channel energy). Until this function existed
    ``compute_stat_anchor_loss`` carried NO first-moment term at all: every
    stat in it (STD, M2, TV, SOS, M1) is a spread or an energy. A recipe
    setting ``stat_anchor_M1_*`` expecting to constrain the MEAN constrains
    per-channel energy instead. ``stat_anchor_MEAN_*`` is the mean.
    Signed on purpose — the DC/colour drift the AR walk produces has a
    sign, and abs() here would hide a bias that flips channel to channel.
    """
    return x.mean(dim=[3, 4])


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


def seed_channel_stat_target(seed_latents: torch.Tensor) -> dict:
    """Per-channel summary stats of a seed window, for stat imposition.

    ``seed_latents: [B, F, C, H, W]`` -> dict of ``[B, C]`` tensors
    (detached, fp32), reduced over ``(F, H, W)`` so each channel gets a
    single target per batch element:

      * ``mean``    : per-channel mean.
      * ``std``     : per-channel std (population, unbiased=False).
      * ``rms``     : per-channel root-mean-square ``sqrt(mean x^2)`` —
                      the L2 energy proxy (M1 / sum-of-squares family).
      * ``meanabs`` : per-channel mean absolute value ``mean|x|`` — the
                      L1 magnitude proxy (the mean-abs alternative to
                      sum-of-squares).

    These are the targets the student's per-channel affine imposition
    rescales toward (see ``impose_channel_stats``). The seed is the
    conditioning context, so the target is available at inference too.
    """
    if seed_latents.dim() != 5:
        raise ValueError(
            "seed_channel_stat_target expects [B, F, C, H, W]; got "
            f"{tuple(seed_latents.shape)}"
        )
    with torch.no_grad():
        s = seed_latents.detach().float()
        B, F, C, H, W = s.shape
        flat = s.permute(0, 2, 1, 3, 4).reshape(B, C, -1)  # [B, C, F*H*W]
        mean = flat.mean(dim=2)                            # [B, C]
        std = flat.var(dim=2, unbiased=False).clamp_min(0).sqrt()
        rms = flat.pow(2).mean(dim=2).clamp_min(0).sqrt()
        meanabs = flat.abs().mean(dim=2)
    return {"mean": mean, "std": std, "rms": rms, "meanabs": meanabs}


def impose_channel_stats(
    x: torch.Tensor, target: dict, mode: str, eps: float = 1e-6,
) -> torch.Tensor:
    """Differentiable per-channel affine that imposes a target stat on the
    student output, frame-by-frame.

    ``x: [B, F, C, H, W]`` (grad-on student latent). ``target`` is a dict
    of ``[B, C]`` per-channel targets (detached) from
    ``seed_channel_stat_target``. The current per-(B, F, C) stats are
    computed from ``x`` over the spatial dims ``(H, W)`` **with gradient**
    (NOT detached), so the imposition is scale/shift-invariant: the
    student receives no gradient pressure on the imposed stat (it is set
    externally), and DMD/GAN gradients only shape the residual texture.

    Modes:
      * ``"m1"``      : per-channel ENERGY — pure multiplicative scaling
                        to match the seed RMS (``sqrt(mean x^2)``). Touches
                        magnitude only (mean rides along, since RMS is not
                        mean-subtracted). The sum-of-squares family.
      * ``"meanabs"`` : per-channel L1 MAGNITUDE — pure scaling to match
                        the seed ``mean|x|`` (the mean-abs alternative to
                        sum-of-squares).
      * ``"m2"``      : per-channel STD — center, rescale spread to the
                        seed std, keep the current per-channel mean.
      * ``"both"``    : per-channel mean AND std — full standardize to the
                        seed (mean, std).

    Returns the rescaled ``x`` (same shape, dtype).
    """
    if mode not in ("m1", "m2", "both", "meanabs"):
        raise ValueError(f"impose_channel_stats: unknown mode {mode!r}.")
    if x.dim() != 5:
        raise ValueError(
            f"impose_channel_stats expects [B, F, C, H, W]; got {tuple(x.shape)}"
        )
    B, F, C, H, W = x.shape
    for k in ("mean", "std", "rms", "meanabs"):
        t = target.get(k)
        if t is not None and tuple(t.shape) != (B, C):
            raise ValueError(
                f"impose_channel_stats: target['{k}'] must be [B={B}, C={C}]; "
                f"got {tuple(t.shape)}"
            )

    def _bf(t):  # [B, C] -> [B, 1, C, 1, 1]  (broadcast over F, H, W)
        return t.to(x.dtype).view(B, 1, C, 1, 1)

    if mode in ("m1", "meanabs"):
        if mode == "m1":
            cur = x.pow(2).mean(dim=(3, 4)).clamp_min(eps * eps).sqrt()  # [B,F,C]
            tgt = _bf(target["rms"])
        else:
            cur = x.abs().mean(dim=(3, 4)).clamp_min(eps)                # [B,F,C]
            tgt = _bf(target["meanabs"])
        scale = tgt / (cur.unsqueeze(-1).unsqueeze(-1))                  # [B,F,C,1,1]
        return x * scale

    # m2 / both: standardize per (B, F, C) then re-affine.
    mean_c = x.mean(dim=(3, 4)).unsqueeze(-1).unsqueeze(-1)              # [B,F,C,1,1]
    std_c = (
        x.std(dim=(3, 4), unbiased=False).clamp_min(eps)
        .unsqueeze(-1).unsqueeze(-1)
    )
    scale = _bf(target["std"]) / std_c                                  # [B,F,C,1,1]
    out_mean = mean_c if mode == "m2" else _bf(target["mean"])
    return out_mean + (x - mean_c) * scale


def compute_stat_anchor_loss(
    pred_x0: torch.Tensor,
    seed_latents: Optional[torch.Tensor] = None,
    *,
    seed_STD_anchor: Optional[torch.Tensor] = None,
    seed_M2_anchor: Optional[torch.Tensor] = None,
    seed_TV_anchor: Optional[torch.Tensor] = None,
    seed_SOS_anchor: Optional[torch.Tensor] = None,
    seed_M1_anchor: Optional[torch.Tensor] = None,
    seed_MEAN_anchor: Optional[torch.Tensor] = None,
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
    # First moment (per-channel spatial mean). DEFAULT 0.0 = the term is
    # never built, so every existing caller is byte-identical.
    MEAN_short_weight: float = 0.0,
    MEAN_long_weight: float = 0.0,
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
    std_one_sided: bool = False,
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
            MEAN_anchor = _per_frame_MEAN(seed).mean(dim=1)    # [B, C]
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
        MEAN_anchor = (
            seed_MEAN_anchor.detach() if seed_MEAN_anchor is not None else None
        )

    # Per-frame stats on pred (graph-attached).
    STD_pf = _per_frame_STD(pred_x0)                   # [B, F]
    M2_pf = _per_frame_M2(pred_x0)
    TV_pf = _per_frame_TV(pred_x0)

    # Causal cumulative mean for long-horizon signal.
    STD_long = _causal_cumulative_mean(STD_pf)
    M2_long = _causal_cumulative_mean(M2_pf)
    TV_long = _causal_cumulative_mean(TV_pf)

    # Anchor broadcast. The anchor may be per-batch ``[B]`` (seed mode —
    # one value per ride, broadcast over frames) OR per-frame ``[B, F]``
    # (matched-GT mode — each rolled chunk carries its own k-closest-GT
    # anchor). ``[B] -> [B,1]`` broadcasts; ``[B,F]`` is used as-is.
    def _bf(a, like):
        a = a.to(like.dtype)
        return a if a.dim() == 2 else a.unsqueeze(1)
    a_STD_short = _bf(STD_anchor, STD_pf)
    a_M2_short = _bf(M2_anchor, M2_pf)
    a_TV_short = _bf(TV_anchor, TV_pf)

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
    if bool(std_one_sided):
        # One-sided variance FLOOR (2026-08-20): penalize only the deficit
        # (pred STD below anchor); upside stays free so excursions ("life")
        # are never taxed. Counters AR variance contraction without pinning.
        mse_STD_short = torch.relu(a_STD_short - STD_pf).pow(2).mean()
        mse_STD_long = torch.relu(a_STD_long - STD_long).pow(2).mean()
    else:
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
        a_SOS = _bf(SOS_anchor, SOS_pf)    # [B,1] (seed) or [B,F] (matched)
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
    # SILENT NO-OP GUARD (2026-08-25). The term is skipped both when the
    # WEIGHT is zero (intended: term disabled) and when the ANCHOR is None
    # (NOT intended: e.g. a precomputed-anchor path that omits the "M1"
    # key). Those two cases are indistinguishable in the logs otherwise --
    # both just show ``stat/M1_active_* == 0``. ``stat/M1_term_built``
    # below separates them: requested-but-not-built is a dropped loss term.
    M1_requested = (
        float(M1_short_weight) != 0.0 or float(M1_long_weight) != 0.0
    )
    M1_built = bool(M1_anchor is not None and M1_requested)
    if M1_built:
        M1_pf = _per_frame_M1(pred_x0)                      # [B, F, C]
        M1_long = _causal_cumulative_mean(M1_pf)            # [B, F, C]
        # M1 anchor: per-batch [B,C] -> [B,1,C], or per-frame [B,F,C] as-is.
        a_M1 = M1_anchor.to(M1_pf.dtype)
        a_M1 = a_M1 if a_M1.dim() == 3 else a_M1.unsqueeze(1)
        mse_M1_short = (M1_pf - a_M1).pow(2).mean()
        mse_M1_long = (M1_long - a_M1).pow(2).mean()
        M1_a2 = float(M1_anchor.detach().mean().pow(2).item())
        floor_M1_short = rs2 * M1_a2
        floor_M1_long = rl2 * M1_a2
        loss_M1_short = (mse_M1_short - floor_M1_short).clamp(min=0)
        loss_M1_long = (mse_M1_long - floor_M1_long).clamp(min=0)

    # ------------------------------------------------------------------
    # MEAN (per-channel spatial mean; [B,F,C]) — the FIRST moment, and the
    # only mean-sensitive term in this function (see ``_per_frame_MEAN``:
    # "M1" above is per-channel ENERGY, not the mean). Same floor-clamped
    # MSE machinery. Default weights 0.0 => never built => byte-identical.
    #
    # FLOOR DEVIATION, deliberate: every other stat here is NON-NEGATIVE, so
    # ``anchor.mean()^2`` is a sane "typical magnitude" for the deadband.
    # The mean is SIGNED and its per-channel values largely cancel (measured
    # on weunz GT: mean over channels 0.053 vs mean |per-channel| 0.328), so
    # ``anchor.mean()^2`` would collapse the deadband to ~2.6% of its
    # intended size and make ``rel_tol`` effectively inoperative. We use
    # ``anchor.abs().mean()^2`` so ``rel_tol`` keeps meaning "a fraction of
    # a typical per-channel mean". No effect when rel_tol == 0.
    # ------------------------------------------------------------------
    loss_MEAN_short = loss_MEAN_long = zero
    floor_MEAN_short = floor_MEAN_long = 0.0
    mse_MEAN_short = mse_MEAN_long = zero
    # Same silent-no-op guard as M1 above: ``stat/MEAN_term_built``
    # distinguishes "weight is 0, term intentionally off" from "weight > 0
    # but MEAN_anchor arrived None, term silently dropped". The MEAN term
    # is the newest code path here and the one fed by the brand-new
    # ``_gt_window_stat_anchors["MEAN"]`` key, so it is the most likely to
    # be silently absent.
    MEAN_requested = (
        float(MEAN_short_weight) != 0.0 or float(MEAN_long_weight) != 0.0
    )
    MEAN_built = bool(MEAN_anchor is not None and MEAN_requested)
    if MEAN_built:
        MEAN_pf = _per_frame_MEAN(pred_x0)                   # [B, F, C]
        MEAN_long = _causal_cumulative_mean(MEAN_pf)         # [B, F, C]
        a_MEAN = MEAN_anchor.to(MEAN_pf.dtype)
        a_MEAN = a_MEAN if a_MEAN.dim() == 3 else a_MEAN.unsqueeze(1)
        mse_MEAN_short = (MEAN_pf - a_MEAN).pow(2).mean()
        mse_MEAN_long = (MEAN_long - a_MEAN).pow(2).mean()
        MEAN_a2 = float(MEAN_anchor.detach().abs().mean().pow(2).item())
        floor_MEAN_short = rs2 * MEAN_a2
        floor_MEAN_long = rl2 * MEAN_a2
        loss_MEAN_short = (mse_MEAN_short - floor_MEAN_short).clamp(min=0)
        loss_MEAN_long = (mse_MEAN_long - floor_MEAN_long).clamp(min=0)

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
        + float(MEAN_short_weight) * loss_MEAN_short
        + float(MEAN_long_weight) * loss_MEAN_long
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
        # 1.0 iff the M1 term was actually BUILT (weight requested AND
        # anchor present). ``M1_requested=1, M1_term_built=0`` means the
        # term was silently dropped for want of an anchor.
        "stat/M1_requested": _t(1.0 if M1_requested else 0.0),
        "stat/M1_term_built": _t(1.0 if M1_built else 0.0),
        # MEAN (first moment, per-channel spatial mean). ``_abs`` is the
        # magnitude telemetry: the signed channel average cancels to near
        # zero, so a run watching only ``stat/MEAN_anchor`` would see ~0
        # whether the mean is healthy or has walked off.
        "stat/MEAN_anchor": (
            MEAN_anchor.mean().detach() if MEAN_anchor is not None else zero
        ),
        "stat/MEAN_anchor_abs": (
            MEAN_anchor.abs().mean().detach()
            if MEAN_anchor is not None else zero
        ),
        "stat/MEAN_mse_short": mse_MEAN_short.detach(),
        "stat/MEAN_mse_long": mse_MEAN_long.detach(),
        "stat/MEAN_floor_short": _t(floor_MEAN_short),
        "stat/MEAN_floor_long": _t(floor_MEAN_long),
        "stat/MEAN_active_short": loss_MEAN_short.detach(),
        "stat/MEAN_active_long": loss_MEAN_long.detach(),
        # See ``stat/M1_term_built``. ``MEAN_requested=1,
        # MEAN_term_built=0`` == the mean-training term is silently absent.
        "stat/MEAN_requested": _t(1.0 if MEAN_requested else 0.0),
        "stat/MEAN_term_built": _t(1.0 if MEAN_built else 0.0),
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
