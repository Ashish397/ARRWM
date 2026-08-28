"""Compact CPU contracts for the benchmark-only decoder pullback model."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.decoder_pullback_surrogate import (
    DecoderPullbackSurrogate,
    SignedCotangentPyramid,
)


LATENT_SHAPE = (2, 3, 16, 2, 3)
PIXEL_SHAPE = (2, 12, 3, 16, 24)


def _model(**overrides) -> DecoderPullbackSurrogate:
    torch.manual_seed(2708)
    kwargs = dict(
        latent_channels=16,
        latent_frames=3,
        latent_height=2,
        latent_width=3,
        pixel_frames=12,
        pixel_height=16,
        pixel_width=24,
        pyramid_channels=(4, 6, 8),
        z_width=8,
        num_blocks=2,
    )
    kwargs.update(overrides)
    return DecoderPullbackSurrogate(**kwargs).eval()


def test_signed_pyramid_and_predictor_preserve_exact_geometry():
    pyramid = SignedCotangentPyramid(
        channels=(4, 6, 8),
        pixel_frames=12, pixel_height=16, pixel_width=24,
        latent_frames=3, latent_height=2, latent_width=3,
    )
    v = torch.randn(*PIXEL_SHAPE)
    assert pyramid(v).shape == (2, 3, 8, 2, 3)
    model = _model()
    z = torch.randn(*LATENT_SHAPE)
    assert model(z, v).shape == z.shape
    assert all(
        module.bias is None
        for module in pyramid.modules()
        if isinstance(module, torch.nn.Conv3d)
    )
    assert model.output.bias is None


def test_pullback_is_exactly_zero_and_odd_in_pixel_cotangent():
    model = _model()
    z = torch.randn(*LATENT_SHAPE)
    v = torch.randn(*PIXEL_SHAPE)
    zero = model(z, torch.zeros_like(v))
    positive = model(z, v)
    negative = model(z, -v)
    assert torch.count_nonzero(zero).item() == 0
    assert torch.allclose(negative, -positive, rtol=2e-5, atol=2e-6)


def test_pullback_is_additive_and_homogeneous_for_fixed_z():
    model = _model()
    z = torch.randn(*LATENT_SHAPE)
    v1 = torch.randn(*PIXEL_SHAPE)
    v2 = torch.randn(*PIXEL_SHAPE)
    a, b = 0.37, -1.23
    combined = model(z, a * v1 + b * v2)
    separate = a * model(z, v1) + b * model(z, v2)
    assert torch.allclose(combined, separate, rtol=3e-5, atol=3e-6)
    scaled = model(z, -2.5 * v1)
    assert torch.allclose(scaled, -2.5 * model(z, v1), rtol=3e-5, atol=3e-6)


def test_graph_bearing_pixel_cotangent_is_rejected():
    model = _model()
    z = torch.randn(*LATENT_SHAPE)
    v = torch.randn(*PIXEL_SHAPE, requires_grad=True)
    try:
        model(z, v)
    except ValueError as exc:
        assert "must be detached" in str(exc)
    else:
        raise AssertionError("graph-bearing pixel cotangent was accepted")


def test_fixed_unit_control_is_the_exact_same_parameter_forward_path():
    model = _model()
    z = torch.randn(*LATENT_SHAPE)
    unit = model.unit_pixel_cotangent(
        z.shape[0], device=z.device, dtype=z.dtype,
    )
    assert unit.shape == PIXEL_SHAPE
    assert not unit.requires_grad
    explicit = model(z, unit)
    control = model.forward_unit_control(z)
    assert torch.equal(control, explicit)


def test_default_false_preserves_v1_seeded_state_and_output_exactly():
    implicit = _model()
    explicit = _model(multiscale_z_gating=False)
    implicit_state = implicit.state_dict()
    explicit_state = explicit.state_dict()
    assert tuple(implicit_state) == tuple(explicit_state)
    assert all(
        torch.equal(implicit_state[key], explicit_state[key])
        for key in implicit_state
    )
    assert not any("multiscale_z_gates" in key for key in implicit_state)
    z = torch.randn(*LATENT_SHAPE)
    v = torch.randn(*PIXEL_SHAPE)
    assert torch.equal(implicit(z, v), explicit(z, v))


def test_multiscale_gating_retains_linear_cotangent_contracts():
    model = _model(multiscale_z_gating=True)
    z = torch.randn(*LATENT_SHAPE)
    v1 = torch.randn(*PIXEL_SHAPE)
    v2 = torch.randn(*PIXEL_SHAPE)
    zero = model(z, torch.zeros_like(v1))
    positive = model(z, v1)
    negative = model(z, -v1)
    assert torch.count_nonzero(zero).item() == 0
    assert torch.allclose(negative, -positive, rtol=3e-5, atol=3e-6)
    a, b = 0.41, -1.17
    combined = model(z, a * v1 + b * v2)
    separate = a * model(z, v1) + b * model(z, v2)
    assert torch.allclose(combined, separate, rtol=5e-5, atol=5e-6)
    assert torch.allclose(
        model(z, 2.25 * v1), 2.25 * positive,
        rtol=5e-5, atol=5e-6,
    )


def test_first_high_resolution_cotangent_response_is_z_dependent():
    model = _model(multiscale_z_gating=True)
    v = torch.randn(*PIXEL_SHAPE)
    z1 = torch.zeros(*LATENT_SHAPE)
    z2 = torch.ones(*LATENT_SHAPE)
    with torch.no_grad():
        features1 = model.z_tower(z1.permute(0, 2, 1, 3, 4))
        features2 = model.z_tower(z2.permute(0, 2, 1, 3, 4))
        gates1 = model._make_multiscale_stage_gates(features1)
        gates2 = model._make_multiscale_stage_gates(features2)
        assert gates1 is not None and gates2 is not None
        first = model.cotangent.stages[0](
            v.permute(0, 2, 1, 3, 4).contiguous()
        )
        assert gates1[0].shape == first.shape == (2, 4, 6, 8, 12)
        response1 = first * gates1[0]
        response2 = first * gates2[0]
    assert not torch.allclose(gates1[0], gates2[0])
    assert not torch.allclose(response1, response2)
