"""CPU tests for the first-order direct latent-gradient surrogate."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.latent_gradient_surrogate import (
    DirectGradientDistiller,
    LatentGradientPredictor,
    generator_direct_gradient_loss,
    pool_decoded_pixels_to_latents,
)
from model.latent_texture_critic import build_from_config


SHAPE = (2, 2, 16, 8, 8)


class ChannelLinearTeacher:
    """A conservative field that the spatial predictor can fit exactly."""

    def __init__(self) -> None:
        self.a = torch.linspace(-1.0, 1.0, 16).reshape(1, 1, 16, 1, 1)

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        return (z * self.a.to(z)).flatten(1).mean(dim=1)


class LocalPatchTeacher:
    """The old surrogate suite's non-linear conv/patch teacher."""

    def __init__(self, seed: int = 5, scale: float = 0.08) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.weight = torch.randn(1, 16, 1, 3, 3, generator=gen) * scale

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        x = z.permute(0, 2, 1, 3, 4)
        return torch.tanh(F.conv3d(
            x, self.weight.to(z), stride=(1, 2, 2), padding=(0, 1, 1),
        ))


def _predictor(**kw):
    cfg = dict(in_channels=16, width=32, num_blocks=2, head_init_std=1e-3)
    cfg.update(kw)
    return LatentGradientPredictor(**cfg)


def test_predictor_preserves_latent_shape_and_honours_origin_bounds():
    model = _predictor()
    z = torch.randn(*SHAPE)
    assert model(z, latent_origin=(3, 5)).shape == z.shape
    try:
        model(z, latent_origin=(60, 0))
    except ValueError as exc:
        assert "exceeds" in str(exc)
    else:
        raise AssertionError("out-of-bounds latent origin was accepted")


def test_predictor_accepts_exact_per_sample_origins():
    torch.manual_seed(101)
    model = _predictor()
    z = torch.randn(*SHAPE)
    origins = ((1, 2), (7, 9))
    together = model(z, latent_origin=origins)
    separate = torch.cat([
        model(z[i:i + 1], latent_origin=origins[i])
        for i in range(z.shape[0])
    ], dim=0)
    assert torch.allclose(together, separate, rtol=1e-5, atol=1e-6)


def test_temporal_and_global_context_add_the_missing_dependencies():
    torch.manual_seed(102)
    temporal = _predictor(temporal_mixing=True, temporal_blocks=1)
    a = torch.randn(1, 2, 16, 16, 16)
    b = a.clone(); b[:, 1].add_(2.0)
    # A changed neighbouring latent frame can alter frame zero only when
    # temporal mixing is explicitly enabled.
    assert not torch.allclose(temporal(a)[:, 0], temporal(b)[:, 0])

    torch.manual_seed(103)
    local = _predictor(num_blocks=1)
    torch.manual_seed(103)
    contextual = _predictor(num_blocks=1, global_context=True)
    c = torch.randn(1, 2, 16, 16, 16)
    d = c.clone(); d[:, 1].add_(4.0)
    # The baseline folds frames into the batch, so changing frame one cannot
    # affect frame zero. Clip-global context deliberately crosses that seam.
    assert torch.equal(local(c)[:, 0], local(d)[:, 0])
    assert not torch.allclose(
        contextual(c)[:, 0], contextual(d)[:, 0],
    )


def test_architecture_extensions_do_not_perturb_historical_parameters():
    torch.manual_seed(104)
    baseline = _predictor()
    torch.manual_seed(104)
    extended = _predictor(
        temporal_mixing=True, temporal_blocks=2, global_context=True,
    )
    extended_state = extended.state_dict()
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, extended_state[name]), name


def test_pixel_extrema_pool_uses_exact_temporal_and_spatial_groups():
    pixels = torch.arange(1 * 8 * 3 * 4 * 6, dtype=torch.float32).reshape(
        1, 8, 3, 4, 6,
    )
    pixels = pixels / pixels.max() * 2.0 - 1.0
    guide = pool_decoded_pixels_to_latents(
        pixels, latent_frames=2, latent_height=2, latent_width=3,
    )
    grouped = pixels.reshape(1, 2, 4, 3, 2, 2, 3, 2)
    expected_max = grouped.amax(dim=(2, 5, 7))
    expected_min = grouped.amin(dim=(2, 5, 7))
    expected = torch.cat((expected_max, expected_min), dim=2)
    assert guide.shape == (1, 2, 6, 2, 3)
    assert torch.equal(guide, expected)
    assert not guide.requires_grad


def test_pixel_extrema_pool_refuses_ambiguous_temporal_alignment():
    pixels = torch.randn(1, 7, 3, 16, 16)
    try:
        pool_decoded_pixels_to_latents(
            pixels, latent_frames=2, latent_height=2, latent_width=2,
        )
    except ValueError as exc:
        assert "not an integral expansion" in str(exc)
    else:
        raise AssertionError("non-integral pixel/latent frame ratio was accepted")


def test_condition_branch_is_zero_init_and_does_not_perturb_baseline_rng():
    torch.manual_seed(123)
    base = _predictor(pixel_condition_channels=0)
    torch.manual_seed(123)
    conditioned = _predictor(pixel_condition_channels=6)
    base_state = base.state_dict()
    conditioned_state = conditioned.state_dict()
    for name, value in base_state.items():
        assert torch.equal(value, conditioned_state[name]), name
    assert torch.count_nonzero(
        conditioned.pixel_condition_input.weight
    ).item() == 0
    z = torch.randn(*SHAPE)
    condition = torch.randn(SHAPE[0], SHAPE[1], 6, SHAPE[-2], SHAPE[-1])
    assert torch.equal(base(z), conditioned(z, pixel_condition=condition))
    target = torch.randn_like(z)
    (conditioned(z, pixel_condition=condition) - target).pow(2).mean().backward()
    branch_grad = conditioned.pixel_condition_input.weight.grad
    assert branch_grad is not None
    assert torch.count_nonzero(branch_grad).item() > 0


def test_pixel_condition_is_detached_from_generator_gradient_route():
    torch.manual_seed(22)
    model = _predictor(pixel_condition_channels=6)
    with torch.no_grad():
        model.pixel_condition_input.weight.normal_(std=0.01)
    model.update_teacher_rms(torch.tensor(2.5e-4))
    z = torch.randn(*SHAPE, requires_grad=True)
    pixel_source = torch.randn(
        SHAPE[0], SHAPE[1] * 4, 3, SHAPE[-2] * 2, SHAPE[-1] * 2,
        requires_grad=True,
    )
    condition = pool_decoded_pixels_to_latents(
        pixel_source.detach(),
        latent_frames=SHAPE[1],
        latent_height=SHAPE[-2],
        latent_width=SHAPE[-1],
    )
    loss, logs = generator_direct_gradient_loss(
        model, z, pixel_condition=condition,
    )
    loss.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert pixel_source.grad is None
    assert logs["train/surrogate_pixel_condition_active"] == 1.0
    assert logs["train/surrogate_pixel_condition_channels"] == 6.0
    try:
        model(z.detach(), pixel_condition=condition.requires_grad_(True))
    except ValueError as exc:
        assert "must be detached" in str(exc)
    else:
        raise AssertionError("graph-bearing pixel condition was accepted")


def test_teacher_feature_branch_is_zero_init_separately_normalised_and_detached():
    torch.manual_seed(124)
    base = _predictor(temporal_mixing=True)
    torch.manual_seed(124)
    conditioned = _predictor(
        temporal_mixing=True, teacher_feature_channels=7,
    )
    conditioned_state = conditioned.state_dict()
    for name, value in base.state_dict().items():
        assert torch.equal(value, conditioned_state[name]), name
    assert torch.count_nonzero(
        conditioned.teacher_feature_input.weight
    ).item() == 0

    z = torch.randn(*SHAPE)
    stats = torch.randn(SHAPE[0], SHAPE[1], 7)
    head_grad = torch.randn_like(stats) * 1.0e-5
    assert torch.equal(
        base(z),
        conditioned(
            z, teacher_stats_condition=stats,
            teacher_head_grad_condition=head_grad,
        ),
    )
    target = torch.randn_like(z)
    pred = conditioned(
        z, teacher_stats_condition=stats,
        teacher_head_grad_condition=head_grad,
    )
    (pred - target).pow(2).mean().backward()
    branch_grad = conditioned.teacher_feature_input.weight.grad
    assert branch_grad is not None
    assert torch.count_nonzero(branch_grad).item() > 0

    try:
        conditioned(
            z, teacher_stats_condition=stats.requires_grad_(True),
            teacher_head_grad_condition=head_grad,
        )
    except ValueError as exc:
        assert "must be detached" in str(exc)
    else:
        raise AssertionError("graph-bearing teacher evidence was accepted")


def test_direct_distiller_fits_gradient_direction_first_order():
    torch.manual_seed(7)
    teacher = ChannelLinearTeacher()
    model = _predictor()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, betas=(0.0, 0.9))
    distiller = DirectGradientDistiller(
        model, pix_teacher_refresh_every=2, cache_capacity=8,
        grad_check_every=1, sync_grads=False, distill_substeps=4,
    )
    for step in range(12):
        distiller.step(
            z_real=torch.randn(*SHAPE),
            z_fake=torch.randn(*SHAPE),
            teacher_value_fn=teacher,
            current_step=step,
            optimizer=opt,
            origin_real=(0, 0),
            origin_fake=(0, 0),
        )
    audit = distiller.surrogate_grad_check(
        torch.randn(*SHAPE), teacher, current_step=12,
    )
    assert audit["train/surrogate_check_cos_sim"] > 0.90
    assert audit["train/surrogate_check_cos_sample_median"] > 0.90
    assert audit["train/surrogate_check_cos_sample_q1"] > 0.90
    assert audit["train/surrogate_check_cos_sample_min"] > 0.90
    assert audit["train/surrogate_check_cos_sample_n"] == SHAPE[0]
    assert audit["train/surrogate_check_cos_sample_q1_rolling2_count"] == 1
    assert (
        audit["train/surrogate_check_cos_sample_q1_rolling2_min"]
        == audit["train/surrogate_check_cos_sample_q1"]
    )
    assert audit["train/surrogate_check_distributed_ranks"] == 1
    audit2 = distiller.surrogate_grad_check(
        torch.randn(*SHAPE), teacher, current_step=13,
    )
    assert audit2["train/surrogate_check_cos_sample_q1_rolling2_count"] == 2
    assert audit2["train/surrogate_check_cos_sample_q1_rolling2_min"] == min(
        audit["train/surrogate_check_cos_sample_q1"],
        audit2["train/surrogate_check_cos_sample_q1"],
    )
    assert 0.8 < audit["train/surrogate_check_mag_ratio"] < 1.2
    assert distiller.n_distill_substeps == 12 * 4
    assert int(model.teacher_grad_rms_updates) == 12


def test_direct_distiller_generalises_on_nonlinear_local_patch_field():
    """Comparable to the CPU teacher used to justify the scalar student."""

    torch.manual_seed(19)
    teacher = LocalPatchTeacher()
    model = _predictor(num_blocks=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.0, 0.9))
    distiller = DirectGradientDistiller(
        model, pix_teacher_refresh_every=2, cache_capacity=8,
        sync_grads=False, distill_substeps=8,
    )
    shape = (4, 2, 16, 8, 8)
    for step in range(14):
        distiller.step(
            z_real=torch.randn(*shape),
            z_fake=torch.randn(*shape) * 1.2,
            teacher_value_fn=teacher,
            current_step=step,
            optimizer=opt,
        )
    audit = distiller.surrogate_grad_check(
        torch.randn(*shape), teacher, current_step=14,
    )
    assert audit["train/surrogate_check_cos_sim"] > 0.70
    assert 0.8 < audit["train/surrogate_check_mag_ratio"] < 1.2


def test_generator_linear_loss_delivers_the_detached_predicted_field():
    torch.manual_seed(11)
    model = _predictor()
    model.update_teacher_rms(torch.tensor(2.5e-4))
    z = torch.randn(*SHAPE, requires_grad=True)
    with torch.no_grad():
        field = model.gradient_for_generator(z.detach())
    loss, logs = generator_direct_gradient_loss(model, z)
    grad = torch.autograd.grad(loss, z)[0]
    assert torch.allclose(grad, -field / z.shape[0], rtol=1e-5, atol=1e-8)
    assert logs["train/surrogate_direct_gradient_mode"] == 1.0
    assert all(p.grad is None for p in model.parameters())


def test_build_from_config_selects_direct_mode_only_when_explicit():
    cfg = SimpleNamespace(
        surrogate_critic_enabled=True,
        surrogate_gradient_mode="direct",
        surrogate_direct_width=32,
        surrogate_direct_num_blocks=2,
        surrogate_direct_head_init_std=1e-3,
        surrogate_direct_teacher_rms_beta=0.9,
        surrogate_critic_lr=1e-3,
        pix_teacher_refresh_every=3,
        surrogate_cache_capacity=4,
        surrogate_grad_check_every=5,
        surrogate_teacher_use_checkpoint=False,
        surrogate_distill_substeps=7,
        surrogate_teacher_target_microbatch=1,
        surrogate_pixel_condition_enabled=True,
        surrogate_direct_temporal_mixing=True,
        surrogate_direct_temporal_blocks=1,
        surrogate_direct_global_context=True,
        surrogate_direct_loss_mode="cosine",
        surrogate_direct_real_loss_weight=0.0,
        surrogate_direct_fake_loss_weight=1.0,
    )
    critic, optimizer, distiller = build_from_config(cfg)
    assert isinstance(critic, LatentGradientPredictor)
    assert isinstance(distiller, DirectGradientDistiller)
    assert optimizer is not None
    assert distiller.pix_teacher_refresh_every == 3
    assert distiller.distill_substeps == 7
    assert distiller.teacher_target_microbatch == 1
    assert critic.pixel_condition_channels == 6
    assert critic.temporal_mixing
    assert critic.global_context_enabled
    assert distiller.loss_mode == "cosine"
    assert distiller.loss_weights == {"real": 0.0, "fake": 1.0}


def test_fake_only_direction_distillation_does_not_build_unused_real_targets():
    class RecordingTeacher(ChannelLinearTeacher):
        def __init__(self):
            super().__init__()
            self.calls = []

        def __call__(self, z):
            self.calls.append(int(z.shape[0]))
            return super().__call__(z)

    teacher = RecordingTeacher()
    model = _predictor()
    distiller = DirectGradientDistiller(
        model, loss_mode="cosine", real_loss_weight=0.0,
        fake_loss_weight=1.0, sync_grads=False,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    logs = distiller.step(
        z_real=torch.randn(*SHAPE), z_fake=torch.randn(*SHAPE),
        teacher_value_fn=teacher, current_step=0, optimizer=opt,
    )
    assert teacher.calls == [SHAPE[0]]
    assert distiller.cache.latest("real") is None
    assert distiller.cache.latest("fake") is not None
    assert logs["train/surrogate_direct_loss_is_cosine"] == 1.0


def test_pixel_condition_gate_rejects_scalar_potential_student():
    try:
        build_from_config(SimpleNamespace(
            surrogate_critic_enabled=True,
            surrogate_gradient_mode="potential",
            surrogate_pixel_condition_enabled=True,
        ))
    except ValueError as exc:
        assert "requires surrogate_gradient_mode='direct'" in str(exc)
    else:
        raise AssertionError("pixel conditioning was accepted in potential mode")


def test_teacher_microbatch_preserves_all_targets_and_serializes_graphs():
    class RecordingTeacher(ChannelLinearTeacher):
        def __init__(self):
            super().__init__()
            self.batch_sizes = []

        def __call__(self, z):
            self.batch_sizes.append(int(z.shape[0]))
            return super().__call__(z)

    teacher = RecordingTeacher()
    model = _predictor()
    distiller = DirectGradientDistiller(
        model, teacher_target_microbatch=1, sync_grads=False,
        distill_substeps=1,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    distiller.step(
        z_real=torch.randn(*SHAPE), z_fake=torch.randn(*SHAPE),
        teacher_value_fn=teacher, current_step=0, optimizer=opt,
    )
    assert teacher.batch_sizes == [1, 1, 1, 1]
    assert distiller.cache.latest("real").z.shape[0] == SHAPE[0]
    assert distiller.cache.latest("fake").z.shape[0] == SHAPE[0]


def test_direct_distiller_caches_the_condition_with_its_teacher_target():
    model = _predictor(pixel_condition_channels=6)
    distiller = DirectGradientDistiller(
        model, cache_capacity=1, sync_grads=False, distill_substeps=1,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    condition_real = torch.randn(
        SHAPE[0], SHAPE[1], 6, SHAPE[-2], SHAPE[-1],
    )
    condition_fake = torch.randn_like(condition_real)
    logs = distiller.step(
        z_real=torch.randn(*SHAPE),
        z_fake=torch.randn(*SHAPE),
        teacher_value_fn=ChannelLinearTeacher(),
        current_step=0,
        optimizer=opt,
        condition_real=condition_real,
        condition_fake=condition_fake,
    )
    assert torch.equal(
        distiller.cache.latest("real").condition, condition_real,
    )
    assert torch.equal(
        distiller.cache.latest("fake").condition, condition_fake,
    )
    assert logs["train/surrogate_pixel_condition_weight_norm"] > 0
    assert logs["train/surrogate_pixel_condition_grad_norm"] > 0
    assert logs["train/surrogate_pixel_condition_field_delta_rms"] > 0
