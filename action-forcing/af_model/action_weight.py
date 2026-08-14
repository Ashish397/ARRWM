"""Action-error loss weighting for the rollout ODE stage.

Upweight the per-sample loss of a rolled-out chain in proportion to how WRONG
its REALISED action is, measured the same way the 14e teacher measures it:

    committed latents -> frozen VAE decode -> CoTracker point tracks
    -> mean per-frame dx/dy on a 10x10 grid -> PCA(200 -> 8) -> tanh squash

i.e. exactly ``_compute_teacher_visuals`` / ``_motion_to_action_z`` in
trainer/causal_diffusion_teacher_train.py, so the realised z and the commanded
z live in the same space (pca_raw, dims [0,1] = throttle, steer under the 14e
convention).

WHY A CHAIN-LEVEL WEIGHT AND NOT PER-CHUNK
------------------------------------------
CoTracker is run with ``output_chunk_size = 12`` frames. A rollout chunk is 3
frames, so a per-chunk readout is impossible — ``n_out = 3 // 12 = 0`` and the
window is silently skipped. The full generated chain is 6 x 3 = 18 frames,
which yields exactly ONE window. So the weight is per-chain: it scales every
term of that chain's loss (pointwise MSE *or* kl_local — the weight multiplies
the accumulated total, so it applies to both bases identically).

The weight is DETACHED: it rescales the gradient, it does not add a gradient
path of its own. Nothing back-propagates through CoTracker.

    w = 1 + alpha * min(err / err_ref, max_ratio)

with ``err = ||z_realised - z_commanded||`` over the action dims. A chain whose
realised motion matches the command is left at weight 1; a chain that ignores
or inverts the command is upweighted, so the pointwise term — the only part of
the objective that can actually ground actions (measured: 69.3% relative
direction sensitivity vs 52.2% for KL) — is pushed hardest exactly where the
action is wrong.
"""
from typing import Optional

import torch


def realised_action_z(
    *,
    chain_latents: torch.Tensor,      # [1, F, C, H, W] committed student chain
    frozen_vae,
    cotracker,
    pca_mean: torch.Tensor,           # [200]
    pca_comp_T: torch.Tensor,         # [200, 16]
    pca_scales: torch.Tensor,         # [8]
    grid_size: int = 10,
    output_chunk_size: int = 12,
) -> Optional[torch.Tensor]:
    """-> [n_win, 8] realised action z, or None if the chain is too short.

    Mirrors _compute_teacher_visuals: prepend a dummy frame for the VAE, decode,
    scale to 0-255, track, take mean per-frame deltas, PCA-project, tanh.
    """
    with torch.no_grad():
        F_ = int(chain_latents.shape[1])
        if F_ < output_chunk_size:
            return None
        dummy = chain_latents[:, 0:1]
        lat = torch.cat([dummy, chain_latents], dim=1)
        pixels = frozen_vae.decode_to_pixel(lat.float())[:, 1:, ...]
        video = (255.0 * 0.5 * (pixels + 1.0)).clamp(0, 255).float()

        n_out = video.shape[1] // output_chunk_size
        if n_out == 0:
            return None
        used = n_out * output_chunk_size
        vid = video[:, :used].clone()
        with torch.amp.autocast(device_type="cuda", enabled=True):
            tracks, vis = cotracker(vid, grid_size=grid_size)
        N = grid_size ** 2
        tw = tracks.reshape(1, n_out, output_chunk_size, N, 2)
        dw = tw[:, :, 1:] - tw[:, :, :-1]          # per-frame displacement
        mo = dw.mean(dim=2).squeeze(0)             # [n_out, N, 2]
        flat = mo.reshape(n_out, 2 * N).float()
        P = (flat - pca_mean) @ pca_comp_T
        return torch.tanh(P[:, :8] / pca_scales)   # [n_out, 8]


def action_error_weight(
    *,
    realised_z: Optional[torch.Tensor],   # [n_win, 8] or None
    commanded_z: torch.Tensor,            # [F, 2] per-frame command
    action_dims=(0, 1),
    alpha: float = 1.0,
    err_ref: float = 0.25,
    max_ratio: float = 4.0,
) -> torch.Tensor:
    """Detached scalar weight >= 1, larger when the realised action is wronger."""
    dev = commanded_z.device
    if realised_z is None or realised_z.numel() == 0:
        return torch.ones((), device=dev)
    with torch.no_grad():
        got = realised_z[:, list(action_dims)].mean(dim=0)     # [2]
        want = commanded_z[..., :len(action_dims)].reshape(-1, len(action_dims)).mean(0)
        err = torch.linalg.vector_norm(got - want)
        ratio = (err / max(err_ref, 1e-6)).clamp(max=max_ratio)
        return (1.0 + alpha * ratio).detach()


def variance_error_weight(
    *,
    pred_chain: torch.Tensor,     # [1, F, C, H, W] student committed chain
    teacher_chain: torch.Tensor,  # [1, F, C, H, W] teacher committed chain
    alpha: float = 1.0,
    ref: float = 0.10,
    max_ratio: float = 4.0,
) -> torch.Tensor:
    """Detached weight >= 1, larger when the student's DISPERSION is wrong.

    Companion to action_error_weight for the KL variant: kl_local's log(st/sp)
    term is the variance-sensitive part of the objective, so weighting by the
    variance error concentrates it where dispersion has actually drifted.

    err = mean_c |log(sigma_pred_c / sigma_teacher_c)| -- a scale-free,
    symmetric measure (a 2x contraction and a 2x expansion score the same),
    which is the right shape given the failure we measured is CONTRACTION
    (s_c = s_GT * k^(c+1)).
    """
    with torch.no_grad():
        dims = (1, 3, 4)
        sp = pred_chain.float().std(dim=dims).clamp(min=1e-6)
        st = teacher_chain.float().std(dim=dims).clamp(min=1e-6)
        err = (sp / st).log().abs().mean()
        ratio = (err / max(ref, 1e-6)).clamp(max=max_ratio)
        return (1.0 + alpha * ratio).detach()
