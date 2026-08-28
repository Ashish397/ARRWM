import torch

from analysis.gan_tuning.compression_pullback_oracle import block_projection


def test_identity_projection_is_bit_exact():
    vector = torch.randn(2, 12, 3, 16, 24)
    assert torch.equal(block_projection(vector, 1, 1), vector)


def test_block_projection_is_idempotent_and_energy_contracting():
    vector = torch.randn(2, 12, 3, 16, 24)
    projected = block_projection(vector, 4, 8)
    twice = block_projection(projected, 4, 8)
    # A second fp32 reduction can differ by a few accumulation ulps.
    torch.testing.assert_close(twice, projected, rtol=0, atol=5e-7)
    assert projected.square().mean() <= vector.square().mean()


def test_projection_residual_is_orthogonal_to_retained_component():
    vector = torch.randn(2, 12, 3, 16, 24)
    projected = block_projection(vector, 2, 4)
    residual = vector - projected
    dot = (projected * residual).sum()
    scale = projected.norm() * residual.norm()
    assert abs(float(dot / scale)) < 1e-6


def test_projection_preserves_shape_and_block_means():
    vector = torch.randn(1, 12, 3, 16, 24)
    projected = block_projection(vector, 4, 8)
    assert projected.shape == vector.shape
    original_means = vector.reshape(1, 3, 4, 3, 2, 8, 3, 8).mean((2, 5, 7))
    projected_means = projected.reshape(1, 3, 4, 3, 2, 8, 3, 8).mean((2, 5, 7))
    torch.testing.assert_close(projected_means, original_means)
