"""Variational Rectified Flow Matching (arXiv:2502.09616), adapted.

The paper's problem: the ground-truth velocity field is MULTI-MODAL -- at one
(x_t, t) several different velocities are all correct, because many (x_0, x_1)
pairs pass through that point. Plain MSE flow matching regresses the MEAN of
those directions, so the model learns an averaged, blurred velocity. That is
exactly the failure mode we measured on the 4-rung students: motion collapses
toward a direction-averaged answer, forward is weakly reproduced and the
lateral directions come out worst.

The fix is a per-sample latent z that carries WHICH mode this sample takes, so
the velocity is v(x_t, t, z) rather than v(x_t, t) and no averaging is needed.

DIFFERENCE FROM THE PAPER (deliberate, per the request):
  paper : p(z) = N(0, I)                    -- unconditional prior
          q(z | x_0, x_1, x_t, t)
  here  : p(z | x_0, x_t, t, a)             -- CONDITIONAL prior
          q(z | x_0, x_1, x_t, t, a)

Both are conditioned on the action a, and the prior sees everything that IS
available at inference (the chunk's initial noise x_0, the current noisy latent
x_t, the rung t, the commanded action a). Only the posterior additionally sees
x_1, the teacher's clean chunk. That asymmetry is the whole point: x_1 is the
one thing inference does not have, so it must not enter p. A conditional prior
is strictly more expressive than N(0, I) here -- it can place the mode
distribution differently for a left turn than for a right turn, which an
unconditional prior cannot.

Training:  z ~ q(.|x_0, x_1, x_t, t, a),  loss = base(v_theta(x_t,t,z), tgt)
                                                 + beta * KL(q || p)
Inference: z ~ p(.|x_0, x_t, t, a)

KL is between two learned diagonal Gaussians (not against N(0,I)), so it has
the closed form used in `kl_diag_gaussians` below.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def _time_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of a scalar timestep, [B] -> [B, dim]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32,
                                          device=t.device) / max(half, 1))
    ang = t.float().reshape(-1, 1) * freqs.reshape(1, -1)
    out = torch.cat([torch.cos(ang), torch.sin(ang)], dim=1)
    if out.shape[1] < dim:                       # odd dim
        out = torch.cat([out, out[:, :1]], dim=1)
    return out[:, :dim]


class _LatentEncoder(nn.Module):
    """Encode a stack of video latents (+ t, + action) to a diagonal Gaussian.

    Input latents arrive as [B, F, C, H, W]. Frames are folded into the batch
    for the conv trunk and averaged afterwards, so the encoder is agnostic to
    the number of frames in a chunk and cannot silently depend on chunk size.
    """

    def __init__(self, n_inputs: int, latent_channels: int, z_dim: int,
                 hidden: int = 128, action_dim: int = 2, t_dim: int = 64):
        super().__init__()
        self.n_inputs = int(n_inputs)
        self.latent_channels = int(latent_channels)
        self.t_dim = int(t_dim)
        c_in = self.n_inputs * self.latent_channels
        self.trunk = nn.Sequential(
            nn.Conv2d(c_in, hidden, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1), nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.ctx = nn.Sequential(
            nn.Linear(self.t_dim + int(action_dim), hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * int(z_dim)),
        )
        self.z_dim = int(z_dim)
        # Start near-deterministic: last layer zeroed => mu=0, logvar=0 at
        # init, so p and q agree and KL starts at exactly 0. Without this the
        # KL term dominates the first steps and drives posterior collapse
        # before the velocity head has learned anything.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, lats, t: torch.Tensor,
                action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(lats) != self.n_inputs:
            raise ValueError(
                f"_LatentEncoder built for {self.n_inputs} latent inputs, got "
                f"{len(lats)}. p(z) takes (x0, xt); q(z) takes (x0, x1, xt).")
        B, F = lats[0].shape[0], lats[0].shape[1]
        for i, x in enumerate(lats):
            if x.shape[:2] != (B, F):
                raise ValueError(
                    f"latent {i} has batch/frame {tuple(x.shape[:2])}, "
                    f"expected {(B, F)}")
            if x.shape[2] != self.latent_channels:
                raise ValueError(
                    f"latent {i} has {x.shape[2]} channels, encoder built for "
                    f"{self.latent_channels}")
        # [B,F,C,H,W] x n -> [B*F, n*C, H, W]
        x = torch.cat([l.reshape(B * F, *l.shape[2:]) for l in lats], dim=1)
        h = self.pool(self.trunk(x)).flatten(1)              # [B*F, hidden]
        h = h.view(B, F, -1).mean(dim=1)                     # [B, hidden]
        te = _time_embed(t.reshape(-1)[:1].expand(B), self.t_dim)
        a = action.reshape(B, -1).float()
        c = self.ctx(torch.cat([te, a], dim=1))              # [B, hidden]
        mu, logvar = self.head(torch.cat([h, c], dim=1)).chunk(2, dim=1)
        # Clamp keeps sigma in [~6e-3, ~3.2]; an unclamped logvar makes the KL
        # and the reparameterised sample explode the moment either net drifts.
        return mu, logvar.clamp(-10.0, 2.0)


class VRFMLatent(nn.Module):
    """Conditional prior + posterior for variational rectified flow matching.

    p(z | x_0, x_t, t, a)        -- 2 latent inputs, usable at inference
    q(z | x_0, x_1, x_t, t, a)   -- 3 latent inputs, training only
    """

    def __init__(self, latent_channels: int, z_dim: int = 128,
                 hidden: int = 128, action_dim: int = 2):
        super().__init__()
        self.z_dim = int(z_dim)
        # Callers slice the action with this, instead of a hard-coded 2, so a
        # config with raw_action_dim != 2 cannot silently hit a Linear shape
        # error deep inside the encoder.
        self.action_dim = int(action_dim)
        self.prior = _LatentEncoder(2, latent_channels, z_dim, hidden, action_dim)
        self.posterior = _LatentEncoder(3, latent_channels, z_dim, hidden, action_dim)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor,
                       generator: Optional[torch.Generator] = None) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(std.shape, device=std.device, dtype=std.dtype,
                          generator=generator)
        return mu + eps * std

    def forward(self, x0: torch.Tensor, xt: torch.Tensor, t: torch.Tensor,
                action: torch.Tensor, x1: Optional[torch.Tensor] = None,
                generator: Optional[torch.Generator] = None,
                stats: bool = False):
        """Return (z, kl, stats).

        x1 given  -> training: z ~ q, kl = KL(q || p)  (both nets get gradient)
        x1 None   -> inference: z ~ p, kl = None
        """
        mu_p, lv_p = self.prior([x0, xt], t, action)
        if x1 is None:
            z = self.reparameterize(mu_p, lv_p, generator)
            return z, None, ({"z_absmean": float(z.detach().abs().mean())}
                             if stats else {})
        mu_q, lv_q = self.posterior([x0, x1, xt], t, action)
        z = self.reparameterize(mu_q, lv_q, generator)
        kl = kl_diag_gaussians(mu_q, lv_q, mu_p, lv_p)
        # Each float() is a forced CUDA->CPU sync; at 4 rungs x 6 chunks x 2
        # branches that is ~192 stalls per step, so only pay for it when asked.
        if not stats:
            return z, kl, {}
        return z, kl, {
            "z_absmean": float(z.detach().abs().mean()),
            "kl": float(kl.detach()),
            # sigma_q pinned at 1.0 while z_absmean grows == z is NOISE, not
            # information. This is the collapse/over-use diagnostic.
            "sigma_q": float(torch.exp(0.5 * lv_q).detach().mean()),
            "sigma_p": float(torch.exp(0.5 * lv_p).detach().mean()),
        }


def kl_diag_gaussians(mu_q: torch.Tensor, logvar_q: torch.Tensor,
                      mu_p: torch.Tensor, logvar_p: torch.Tensor) -> torch.Tensor:
    """KL( N(mu_q, sig_q^2) || N(mu_p, sig_p^2) ), summed over z, mean over batch.

    Note this is NOT the KL-to-N(0,I) of the paper: our prior is learned, so
    both terms carry gradient. Closed form for diagonal Gaussians:
        0.5 * sum[ logvar_p - logvar_q + (sig_q^2 + (mu_q-mu_p)^2)/sig_p^2 - 1 ]
    """
    var_p = torch.exp(logvar_p)
    term = (logvar_p - logvar_q
            + (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / var_p - 1.0)
    return 0.5 * term.sum(dim=1).mean()


class ZModulation(nn.Module):
    """Project z into the DiT's AdaLN modulation, [B, z_dim] -> [B, F, 6, dim].

    This is how v becomes v(x_t, t, z). The repo already adds
    `_action_modulation` (shape [B, F, 6, dim]) to `time_projection`'s output
    inside `model/action_model_patch.py:_patch_time_projection`, i.e. exactly
    the paper's "z is added to the time embedding before computing shift and
    offset". Because that hook is a plain SUM, adding our z term into the SAME
    tensor is mathematically identical to registering a second additive stream
    -- and needs no change to `wan/modules/causal_model.py`, the wrapper, or
    the patch. Fewer moving parts, and it cannot desync from the action stream.

    adaLN-zero init (same convention as ActionModulationProjection): the output
    layer starts at ~0 so the model begins EXACTLY as the z-free model and
    learns to use z only as it becomes useful.
    """

    def __init__(self, z_dim: int, hidden_dim: int, mlp_dim: int = 512,
                 zero_init_std: float = 1e-3):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.embed = nn.Sequential(
            nn.Linear(int(z_dim), mlp_dim), nn.LayerNorm(mlp_dim), nn.SiLU(),
            nn.Linear(mlp_dim, mlp_dim), nn.SiLU(),
        )
        self.proj = nn.Linear(mlp_dim, self.hidden_dim * 6)
        nn.init.normal_(self.proj.weight, mean=0.0, std=float(zero_init_std))
        nn.init.zeros_(self.proj.bias)

    def forward(self, z: torch.Tensor, num_frames: int) -> torch.Tensor:
        h = self.proj(self.embed(z))                       # [B, 6*dim]
        B = h.shape[0]
        return (h.view(B, 1, 6, self.hidden_dim)
                 .expand(B, int(num_frames), 6, self.hidden_dim))
