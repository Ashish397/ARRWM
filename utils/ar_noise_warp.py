"""
Time-warped AR-style noise extraction for teacher training.

When the student's AR rollout drifts from GT in speed or trajectory
(student moves through the same scene but at the wrong pace, or with
slight pose offset), the raw residual rollout - GT is dominated by
that drift, not by the student's denoising error. Subtracting the
two directly therefore gives a useless "noise" estimate full of
warp artifacts (the failure mode we saw with the blend setting).

This module does the warp first: align GT to the rollout's local
timeline via DTW on per-frame distance, then the residual
rollout - warped_GT is an estimate of the student's actual noise
distribution at each frame.

The intended consumer is the teacher (LoRA real_score) trainer:
replace torch.randn-style noise on GT with this AR-shaped noise so
the teacher's score is accurate where the student actually lives.
"""

from typing import Optional, Tuple

import torch


def compute_pairwise_cost(
    a: torch.Tensor,
    b: torch.Tensor,
    metric: str = "l2",
) -> torch.Tensor:
    """Per-frame distance matrix.

    a: [T_a, ...] — rollout frames (any per-frame tensor shape).
    b: [T_b, ...] — GT frames, same per-frame shape as `a`.
    metric: "l2" (Euclidean), "cos" (1 - cosine similarity).

    Returns cost [T_a, T_b].
    """
    if a.shape[1:] != b.shape[1:]:
        raise ValueError(
            f"per-frame shape mismatch: {tuple(a.shape[1:])} vs {tuple(b.shape[1:])}"
        )
    af = a.reshape(a.shape[0], -1).float()
    bf = b.reshape(b.shape[0], -1).float()
    if metric == "l2":
        a_norm = (af * af).sum(dim=1, keepdim=True)
        b_norm = (bf * bf).sum(dim=1, keepdim=True).T
        cross = af @ bf.T
        return (a_norm + b_norm - 2 * cross).clamp_min(0.0).sqrt()
    if metric == "cos":
        af_n = af / af.norm(dim=1, keepdim=True).clamp_min(1e-8)
        bf_n = bf / bf.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return 1.0 - af_n @ bf_n.T
    raise ValueError(f"unknown metric '{metric}'")


def dtw_monotone(cost: torch.Tensor, max_skew: int = 2) -> torch.Tensor:
    """Monotone DTW with bounded forward skew.

    For each rollout index i, the chosen GT index j(i) must satisfy
    j(i) <= j(i+1) <= j(i) + max_skew. This allows:
      * j(i+1) = j(i)        : student moves slower than GT (same GT
                               frame matches multiple rollout frames)
      * j(i+1) = j(i) + 1    : aligned step
      * j(i+1) up to +max_skew: student moves faster than GT (skips
                               GT frames)
    Strict monotone non-decreasing: no time reversal.

    cost: [T_r, T_g].
    max_skew:
      * 0: no forward progress permitted at all — path picks one GT
           frame and stays on it for all of rollout.
      * 1: identity-or-stall (each step holds or advances by 1).
      * >=2: allows skipping GT frames (student moves faster).

    Returns path of GT indices, shape [T_r], dtype long.
    """
    if cost.dim() != 2:
        raise ValueError(f"cost must be 2D, got shape {tuple(cost.shape)}")
    if max_skew < 0:
        raise ValueError(f"max_skew must be >= 0, got {max_skew}")

    T_r, T_g = cost.shape
    INF = float("inf")
    dp = torch.full((T_r, T_g), INF, dtype=cost.dtype, device=cost.device)
    backptr = torch.full((T_r, T_g), -1, dtype=torch.long, device=cost.device)

    for j in range(min(max_skew + 1, T_g)):
        dp[0, j] = cost[0, j]

    for i in range(1, T_r):
        for j in range(T_g):
            j_lo = max(0, j - max_skew)
            window = dp[i - 1, j_lo : j + 1]
            if window.numel() == 0:
                continue
            min_val, min_arg = window.min(dim=0)
            if torch.isinf(min_val):
                continue
            dp[i, j] = min_val + cost[i, j]
            backptr[i, j] = j_lo + int(min_arg.item())

    end_j = int(dp[T_r - 1].argmin().item())
    path = torch.zeros(T_r, dtype=torch.long, device=cost.device)
    path[T_r - 1] = end_j
    for i in range(T_r - 1, 0, -1):
        prev = backptr[i, path[i]]
        if prev.item() < 0:
            path[i - 1] = path[i]
        else:
            path[i - 1] = prev
    return path


def time_warp_align(
    rollout: torch.Tensor,
    gt: torch.Tensor,
    max_skew: int = 2,
    metric: str = "l2",
    cost_features: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align GT to rollout's timeline and extract the AR residual.

    rollout: [T_r, ...] — student rollout frames (latents or pixels).
    gt:      [T_g, ...] — ground-truth frames, same per-frame shape.
    max_skew: DTW forward-skew bound (see dtw_monotone).
    metric:  pairwise distance ("l2" or "cos").
    cost_features: optional (rollout_feat, gt_feat) pair used ONLY to
        compute the warp cost. Useful when the warp signal lives in a
        coarser space (e.g. pooled CNN features) while the returned
        warp is applied to the original full-resolution tensors.

    Returns:
        warped_gt: [T_r, ...] — gt reindexed by the warp path.
        path:      [T_r] long — gt index each rollout frame maps to.
        residual:  [T_r, ...] — rollout - warped_gt (AR noise estimate).
        cost:      [T_r, T_g] — pairwise distance used for the warp.
    """
    if rollout.shape[0] == 0 or gt.shape[0] == 0:
        raise ValueError("rollout and gt must have at least 1 frame")

    if cost_features is not None:
        rf, gf = cost_features
        cost = compute_pairwise_cost(rf, gf, metric=metric)
    else:
        cost = compute_pairwise_cost(rollout, gt, metric=metric)

    path = dtw_monotone(cost, max_skew=max_skew)
    warped_gt = gt[path]
    residual = rollout - warped_gt
    return warped_gt, path, residual, cost


def extract_ar_noise_from_teacher(
    rollout: torch.Tensor,
    teacher_clean_estimate: torch.Tensor,
    standardise: bool = True,
    reduce_dims: Optional[Tuple[int, ...]] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """AR noise = student rollout minus teacher's clean estimate.

    Reuses the implicit warp the teacher already performs: when the
    teacher (LoRA real_score) is forward-run on the student's noisy
    rollout, its prediction approximates the clean signal that lives
    inside the rollout. The residual is then the student's actual
    noise component at the rollout's distribution.

    rollout: e.g. [T, C, H, W] latent or pixel student output.
    teacher_clean_estimate: same shape — teacher's denoised view.
    standardise: if True, rescale the residual to unit variance per
        sample so it can be slotted in where torch.randn would go.
    reduce_dims: dims over which to compute mean/std for
        standardisation. Default reduces over all per-sample dims
        (everything but the first dim — i.e. treats dim 0 as the
        sample / time axis).
    eps: numerical floor on std.

    Returns the residual (standardised if requested) with the same
    shape as the inputs.
    """
    if rollout.shape != teacher_clean_estimate.shape:
        raise ValueError(
            f"shape mismatch: rollout {tuple(rollout.shape)} vs "
            f"teacher {tuple(teacher_clean_estimate.shape)}"
        )
    noise = rollout - teacher_clean_estimate
    if not standardise:
        return noise
    if reduce_dims is None:
        reduce_dims = tuple(range(1, noise.dim()))
    mean = noise.mean(dim=reduce_dims, keepdim=True)
    std = noise.std(dim=reduce_dims, keepdim=True, unbiased=False).clamp_min(eps)
    return (noise - mean) / std


def make_ar_noised_gt(
    gt: torch.Tensor,
    ar_noise: torch.Tensor,
    alpha_t: torch.Tensor,
    sigma_t: torch.Tensor,
) -> torch.Tensor:
    """Flow-matching style mix of GT with AR-shaped noise.

    x_t = alpha_t * gt + sigma_t * ar_noise

    `alpha_t` and `sigma_t` are broadcast-compatible with `gt`; they
    are typically per-frame timestep coefficients pulled from the
    diffusion scheduler. This is a drop-in replacement for
    `alpha_t * gt + sigma_t * torch.randn_like(gt)` — the difference
    is that `ar_noise` lives where the student lives.
    """
    if gt.shape != ar_noise.shape:
        raise ValueError(
            f"shape mismatch: gt {tuple(gt.shape)} vs ar_noise {tuple(ar_noise.shape)}"
        )
    return alpha_t * gt + sigma_t * ar_noise


def residual_stats(residual: torch.Tensor) -> dict:
    """Summary stats of the extracted residual, per-frame and overall.

    residual: [T, ...] — per-frame residual tensor.
    """
    T = residual.shape[0]
    flat = residual.reshape(T, -1).float()
    per_frame_norm = flat.norm(dim=1)
    per_frame_std = flat.std(dim=1, unbiased=False)
    per_frame_mean = flat.mean(dim=1)
    return {
        "T": T,
        "per_frame_l2_norm": per_frame_norm.tolist(),
        "per_frame_std": per_frame_std.tolist(),
        "per_frame_mean": per_frame_mean.tolist(),
        "overall_l2_norm": float(flat.norm().item()),
        "overall_std": float(flat.std(unbiased=False).item()),
        "overall_mean": float(flat.mean().item()),
    }
