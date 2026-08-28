import torch

from analysis.gan_tuning.sft_pullback_benchmark import grouped_channel_summary
from model.sft_pullback_surrogate import SFTPullbackSurrogate


def _model(*, mode="spatial", state_channels=0):
    return SFTPullbackSurrogate(
        latent_channels=4, latent_frames=2, latent_height=2, latent_width=3,
        pixel_frames=8, pixel_height=16, pixel_width=24,
        feature_widths=(6, 8, 10, 12), condition_widths=(4, 6, 8, 10),
        latent_blocks=2, conditioning_mode=mode,
        decoder_state_channels=state_channels,
    )


def test_spatial_and_global_controls_have_identical_parameter_count():
    torch.manual_seed(7); spatial = _model(mode="spatial")
    torch.manual_seed(7); global_model = _model(mode="global")
    assert spatial.num_params == global_model.num_params
    assert spatial.state_dict().keys() == global_model.state_dict().keys()


def test_sft_output_shape_and_detached_cotangent_contract():
    model = _model()
    z = torch.randn(2, 2, 4, 2, 3)
    v = torch.randn(2, 8, 3, 16, 24)
    assert model(z, v).shape == z.shape
    try:
        model(z, v.requires_grad_(True))
    except ValueError as error:
        assert "detached" in str(error)
    else:  # pragma: no cover
        raise AssertionError("graph-bearing cotangent was accepted")


def test_state_conditioning_requires_matching_detached_spatial_grids():
    model = _model(state_channels=3)
    z = torch.randn(1, 2, 4, 2, 3)
    v = torch.randn(1, 8, 3, 16, 24)
    states = (
        torch.randn(1, 8, 3, 16, 24), torch.randn(1, 8, 3, 8, 12),
        torch.randn(1, 4, 3, 4, 6), torch.randn(1, 2, 3, 2, 3),
    )
    assert model(z, v, states).shape == z.shape
    bad = list(states); bad[1] = bad[1].requires_grad_(True)
    try:
        model(z, v, bad)
    except ValueError as error:
        assert "detached" in str(error)
    else:  # pragma: no cover
        raise AssertionError("graph-bearing decoder state was accepted")


def test_grouped_decoder_summary_preserves_every_space_time_coordinate():
    value = torch.randn(2, 96, 3, 5, 7)
    summary = grouped_channel_summary(value, groups=8)
    assert summary.shape == (2, 16, 3, 5, 7)
    grouped = value.reshape(2, 8, 12, 3, 5, 7)
    torch.testing.assert_close(summary[:, :8], grouped.mean(dim=2))
    torch.testing.assert_close(summary[:, 8:], grouped.pow(2).mean(dim=2).sqrt())
