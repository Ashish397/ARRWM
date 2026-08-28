import torch

from model.decoder_shaped_pullback import (
    generator_decoder_pullback_loss, public_pixel_output_vjp,
)


def test_public_pixel_output_vjp_pads_dummy_and_applies_clamp_gate():
    output = torch.zeros(1, 3, 5, 2, 2)
    output[:, :, 2, 0, 0] = 1.0
    output[:, :, 3, 1, 1] = -1.0
    pixel = torch.arange(1 * 4 * 3 * 2 * 2, dtype=torch.float32).reshape(
        1, 4, 3, 2, 2,
    )
    result = public_pixel_output_vjp(output, pixel)
    assert result.shape == output.shape
    assert int(torch.count_nonzero(result[:, :, 0])) == 0
    torch.testing.assert_close(result[:, :, 1], pixel[:, 0])
    assert int(torch.count_nonzero(result[:, :, 2, 0, 0])) == 0
    assert int(torch.count_nonzero(result[:, :, 3, 1, 1])) == 0


def test_generator_decoder_pullback_loss_serves_exact_negative_field():
    torch.manual_seed(31)
    latent = torch.randn(2, 3, 4, 5, 6, requires_grad=True)
    field = torch.randn_like(latent)
    loss, logs = generator_decoder_pullback_loss(latent, field, weight=0.3)
    gradient = torch.autograd.grad(loss, latent)[0]
    torch.testing.assert_close(gradient, -0.3 * field / 2)
    assert logs["train/surrogate_decoder_shaped"] == 1.0
