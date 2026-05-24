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
from typing import Tuple

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
