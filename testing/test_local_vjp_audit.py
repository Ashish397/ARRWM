import torch

from analysis.gan_tuning.local_vjp_audit import (
    FINE_STAGES, GRANULAR_STAGES, PATCH_MARGIN, STAGES,
    exact_decoder_prefix_vjp, special_phase_pack,
)
from model.local_vjp_surrogate import IdentityAnchoredLocalVJP, LocalLinearVJPSurrogate


def test_special_phase_pack_is_lossless_for_wan_first_frame_rule():
    value = torch.arange(1 * 3 * 7 * 4 * 6, dtype=torch.float32).reshape(1, 3, 7, 4, 6)
    packed = special_phase_pack(value, input_frames=4, input_height=2, input_width=3)
    assert packed.shape == (1, 3 * 2 * 4, 4, 2, 3)
    # Packing must preserve every scalar plus the declared zero second phase
    # of WAN's special first frame.
    assert int(torch.count_nonzero(packed)) == int(torch.count_nonzero(value))
    assert float(packed.square().sum()) == float(value.square().sum())


def test_stage_packed_channel_contracts():
    assert [stage.packed_channels for stage in STAGES] == [384, 1536, 1536, 384, 96, 3]
    assert [stage.packed_channels for stage in FINE_STAGES] == [
        384, 384, 1536, 384, 1536, 192, 384, 96, 3,
    ]
    assert [stage.packed_channels for stage in GRANULAR_STAGES] == [
        384, 384, 384, 384, 1536, 384, 384, 384, 1536,
        192, 192, 192, 384, 96, 96, 96, 3,
    ]
    assert PATCH_MARGIN == 2 * 1 + 2 * 2


def test_local_operator_is_linear_in_cotangent():
    torch.manual_seed(9)
    model = LocalLinearVJPSurrogate(
        state_channels=7, packed_cotangent_channels=11,
        output_channels=5, width=8, blocks=2,
    )
    state = torch.randn(2, 7, 3, 5, 6)
    v1 = torch.randn(2, 11, 3, 5, 6)
    v2 = torch.randn_like(v1)
    with torch.no_grad():
        zero = model(state, torch.zeros_like(v1))
        odd = model(state, v1) + model(state, -v1)
        combined = model(state, 0.37 * v1 - 1.2 * v2)
        separate = 0.37 * model(state, v1) - 1.2 * model(state, v2)
    assert float(zero.abs().max()) == 0.0
    torch.testing.assert_close(odd, torch.zeros_like(odd), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(combined, separate, rtol=2e-4, atol=2e-5)


def test_identity_anchor_starts_exact_and_remains_linear():
    torch.manual_seed(11)
    model = IdentityAnchoredLocalVJP(state_channels=7, width=8, blocks=2)
    state = torch.randn(2, 7, 3, 5, 6)
    v1 = torch.randn_like(state)
    v2 = torch.randn_like(state)
    with torch.no_grad():
        torch.testing.assert_close(model(state, v1), v1)
        combined = model(state, 0.4 * v1 - 0.7 * v2)
        separate = 0.4 * model(state, v1) - 0.7 * model(state, v2)
    torch.testing.assert_close(combined, separate, rtol=2e-4, atol=2e-5)


def test_exact_decoder_prefix_vjp_matches_explicit_seeded_transpose():
    class Prefix(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv2 = torch.nn.Conv3d(3, 3, 1, bias=True)

    class VAE:
        def __init__(self):
            self.mean = torch.tensor([0.2, -0.4, 0.7])
            self.std = torch.tensor([1.5, 0.8, 2.0])
            self.model = Prefix()

    torch.manual_seed(29)
    vae = VAE()
    latent = torch.randn(2, 3, 3, 4, 5)
    incoming = torch.randn(2, 3, 4, 4, 5)
    actual = exact_decoder_prefix_vjp(vae, latent, incoming)

    # A 1x1x1 convolution cannot mix space/time.  Its transpose is a channel
    # matrix multiply, followed by the inverse VAE normalization; self-seeding
    # makes the public first-frame gradient the sum of seeded frames zero/one.
    weight = vae.model.conv2.weight[:, :, 0, 0, 0]
    seeded = torch.einsum("oc,bothw->bcthw", weight, incoming)
    seeded = seeded * vae.std.view(1, -1, 1, 1, 1)
    expected = torch.empty_like(latent)
    expected[:, 0] = seeded[:, :, 0] + seeded[:, :, 1]
    expected[:, 1] = seeded[:, :, 2]
    expected[:, 2] = seeded[:, :, 3]
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
