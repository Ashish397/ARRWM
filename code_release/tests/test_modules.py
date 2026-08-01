"""Forward-pass behaviour of the action-conditioning modules and the critic.

These pin numerics under a fixed seed. Refactoring that preserves behaviour
leaves them untouched; refactoring that does not fails them immediately.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.action_critic import ActionCritic
from model.action_modulation import ActionModulationProjection, ActionTokenProjection
from utils.scheduler import FlowMatchScheduler

# Small stand-in for the Wan-1.3B latent grid (16 x 60 x 104); the critic is
# fully convolutional, so a reduced grid exercises identical code paths.
LATENT_C, LATENT_H, LATENT_W = 16, 12, 20


def _critic(**kw):
    torch.manual_seed(0)
    return ActionCritic(latent_channels=LATENT_C, base_channels=16, num_res_blocks=1, **kw).eval()


def test_critic_output_shape_is_one_code_per_chunk():
    critic = _critic()
    b, chunks, frames = 2, 3, 3
    x = torch.randn(b, chunks * frames, LATENT_C, LATENT_H, LATENT_W)
    t = torch.randint(0, 1000, (b, chunks)).float()
    a = torch.randn(b, chunks, 2)
    assert critic(x, t, a).shape == (b, chunks, 8)


def test_critic_rejects_frames_not_divisible_by_chunk():
    critic = _critic()
    x = torch.randn(1, 4, LATENT_C, LATENT_H, LATENT_W)
    with pytest.raises(AssertionError):
        critic(x, torch.zeros(1, 1), torch.zeros(1, 1, 2))


def test_critic_chunks_are_independent():
    """Chunk n's prediction must not depend on chunk m's content.

    The critic supervises per-chunk egomotion, so cross-chunk leakage would let
    it read motion it was not shown.
    """
    critic = _critic()
    x = torch.randn(1, 6, LATENT_C, LATENT_H, LATENT_W)
    t = torch.zeros(1, 2)
    a = torch.zeros(1, 2, 2)
    before = critic(x, t, a)

    perturbed = x.clone()
    perturbed[:, 3:] = torch.randn_like(perturbed[:, 3:])  # second chunk only
    after = critic(perturbed, t, a)

    torch.testing.assert_close(before[:, 0], after[:, 0])
    assert not torch.allclose(before[:, 1], after[:, 1])


def test_critic_forward_golden(golden):
    critic = _critic()
    torch.manual_seed(42)
    x = torch.randn(1, 6, LATENT_C, LATENT_H, LATENT_W)
    t = torch.tensor([[250.0, 750.0]])
    a = torch.tensor([[[0.5, 0.0], [-0.3, 0.25]]])
    with torch.no_grad():
        golden.check("critic_pred_z", critic(x, t, a).numpy(), rtol=1e-5, atol=1e-6)


def test_critic_head_starts_near_zero():
    """The z head is initialised small so early critic output cannot dominate."""
    critic = _critic()
    assert critic.z_head[-1].weight.std().item() < 1e-2
    assert torch.all(critic.z_head[-1].bias == 0)


def _adaln(**kw):
    torch.manual_seed(0)
    return ActionModulationProjection(
        action_dim=2, activation="silu", hidden_dim=64, mlp_dim=32, num_frames=3, **kw
    ).eval()


def _tokens(**kw):
    torch.manual_seed(0)
    return ActionTokenProjection(
        action_dim=2, activation="silu", hidden_dim=64, mlp_dim=32, **kw
    ).eval()


def test_adaln_zero_init_starts_as_identity():
    """adaLN-Zero: at init the action pathway must not perturb the backbone.

    This is what lets a pretrained DiT be adapted without destroying its prior,
    and the No-AdaLN ablation is the variant that fails to establish control.
    """
    proj = _adaln(zero_init=True)
    with torch.no_grad():
        out = proj(torch.randn(2, 3, 2))
    assert out.abs().max().item() < 1e-2, "zero-init adaLN should emit ~0 modulation"


def test_adaln_responds_to_action_after_perturbation():
    """With a non-zero output layer, distinct actions give distinct modulation."""
    proj = _adaln(zero_init=False)
    with torch.no_grad():
        a = proj(torch.full((1, 3, 2), 0.5))
        b = proj(torch.full((1, 3, 2), -0.5))
    assert not torch.allclose(a, b)


def test_adaln_shape_is_six_params_per_frame():
    proj = _adaln()
    with torch.no_grad():
        out = proj(torch.zeros(2, 3, 2))
    assert out.shape[:2] == (2, 3) and 6 in out.shape, f"unexpected shape {out.shape}"


def test_action_token_shape_is_one_token_per_frame():
    proj = _tokens()
    with torch.no_grad():
        out = proj(torch.zeros(2, 3, 2))
    assert out.shape == (2, 3, 64)


@pytest.mark.parametrize("factory", [_adaln, _tokens], ids=["adaln", "tokens"])
def test_action_projection_is_finite_at_zero_action(factory):
    with torch.no_grad():
        out = factory()(torch.zeros(2, 3, 2))
    assert torch.isfinite(out).all()


def test_action_projection_golden(golden):
    """Exact modulation and token output for a fixed action pair."""
    a = torch.tensor([[[0.5, 0.0], [-0.3, 0.25], [0.0, -0.5]]])
    with torch.no_grad():
        golden.check("adaln_out", _adaln(zero_init=False)(a).numpy(), rtol=1e-5, atol=1e-6)
        golden.check("token_out", _tokens(zero_init=False)(a).numpy(), rtol=1e-5, atol=1e-6)


def test_flow_match_scheduler_timesteps_are_monotone():
    s = FlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
    s.set_timesteps(48, training=False)
    ts = s.timesteps
    assert len(ts) == 48
    assert torch.all(torch.diff(ts) < 0), "timesteps must decrease over sampling"


def test_flow_match_scheduler_shift_golden(golden):
    """timestep_shift=5.0 is the paper's sampling setting."""
    s = FlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
    s.set_timesteps(48, training=False)
    golden.check("timesteps_shift5", s.timesteps.numpy(), rtol=1e-6)
