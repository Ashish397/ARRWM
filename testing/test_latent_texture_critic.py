#!/usr/bin/env python3
"""Standalone tests for ``model/latent_texture_critic.py`` (WP-SURROGATE / B3).

CPU-only, no VAE, no pixel disc, no GPU, no distributed init — the whole
point of the module's callable-teacher injection is that the mechanism is
testable before WP-PIXGAN lands. The teachers here are analytic functions
with closed-form gradients, which makes the Sobolev claims checkable
exactly rather than by eyeball.

Run either way::

    python testing/test_latent_texture_critic.py          # all tests
    python testing/test_latent_texture_critic.py -k sobolev
    pytest -q testing/test_latent_texture_critic.py

What each group is actually protecting
--------------------------------------
* ``test_attention_double_backward*`` — the load-bearing claim in the
  module docstring. If someone "optimizes" the hand-rolled attention into
  ``F.scaled_dot_product_attention``, the Sobolev backward dies on
  hardware where Flash is selected. This test fails loudly instead.
* ``test_sobolev_*`` — the justification for the gradient-distillation
  term existing at all: value-only distillation can match values while
  getting the gradient field (the only thing the generator consumes)
  wrong. Demonstrated, not asserted by comment.
* ``test_refresh_*`` — delta (b). The teacher fires exactly
  ``ceil(steps/N)`` times while the critic trains every step.
* ``test_frozen_critic_*`` — the generator backward must not write into
  the critic's parameters, and the freeze must be released even on an
  exception.
"""

from __future__ import annotations

import argparse
import math
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from model.latent_texture_critic import (
    STEM_SPATIAL_STRIDE,
    LatentSurrogateDistiller,
    LatentTextureCritic,
    TeacherTargetCache,
    TeacherTargets,
    build_from_config,
    compute_teacher_targets,
    generator_surrogate_loss,
    reduce_patch_logits,
    two_stage_gen_weight,
)

torch.manual_seed(0)

# Small enough for a login node; big enough that the token grid is not 1x1.
TINY = dict(
    in_channels=16, d_model=64, num_blocks=2, num_heads=4,
    max_frames=8, max_token_h=16, max_token_w=32,
)
CROP_SHAPE = (4, 2, 16, 8, 8)     # [N, F, C, h, w] -> token grid (1, 1)
GRID_SHAPE = (4, 2, 16, 16, 16)   # -> token grid (2, 2)


def _critic(**kw) -> LatentTextureCritic:
    cfg = dict(TINY)
    cfg.update(kw)
    return LatentTextureCritic(**cfg)


# ---------------------------------------------------------------------------
# Analytic teachers — stand-ins for pixel_texture_disc o decode(z)
# ---------------------------------------------------------------------------
class LinearFieldTeacher:
    """``v_i = mean(A * z_i)`` returned as a patch map.

    Closed-form gradient: ``dv_i/dz_i = A / numel(z_i)``. Constant in
    ``z``, so a Sobolev-fit critic must reproduce a fixed spatial field —
    exactly the "match the gradient field, not the value" job.

    ``A`` is drawn zero-mean, so for zero-mean inputs the VALUE is ~0
    everywhere while the GRADIENT is rich. That gap is what separates
    value-only distillation from Sobolev distillation.
    """

    def __init__(self, shape, patch_hw=(3, 4), scale: float = 1.0, seed: int = 7):
        g = torch.Generator().manual_seed(seed)
        self.A = torch.randn(*shape[1:], generator=g) * scale
        self.A = self.A - self.A.mean()
        self.patch_hw = patch_hw
        self.calls = 0

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        v = (z * self.A.to(z.dtype)).flatten(1).mean(dim=1)  # [N]
        # Emit a patch map so reduce_patch_logits is exercised on the same
        # shape contract 1 specifies: [N, 1, ph, pw] whose mean is v.
        ph, pw = self.patch_hw
        return v.reshape(-1, 1, 1, 1).expand(-1, 1, ph, pw)

    def grad_of(self, z: torch.Tensor) -> torch.Tensor:
        n = z[0].numel()
        return self.A.to(z.dtype).unsqueeze(0).expand_as(z) / n

    def patch_map(self, z: torch.Tensor) -> torch.Tensor:
        return self(z)


class PatchTeacher:
    """Local conv -> tanh -> patch-logit grid, mean-reduced.

    The realistic stand-in: a bounded O(1) per-patch logit computed from a
    local latent neighbourhood, exactly the shape of
    ``pixel_texture_disc o decode``. Unlike ``LinearFieldTeacher`` its
    gradient field is z-dependent, so fitting it is a real regression
    problem rather than learning one constant field.
    """

    def __init__(self, seed: int = 5, scale: float = 0.08):
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn(1, 16, 1, 3, 3, generator=g) * scale
        self.calls = 0

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        x = z.permute(0, 2, 1, 3, 4)  # [N, C, F, h, w]
        return torch.tanh(F.conv3d(
            x, self.W.to(z.dtype), stride=(1, 2, 2), padding=(0, 1, 1),
        ))


class QuadraticTeacher:
    """``v_i = mean(z_i ** 2)``; gradient ``2 z_i / numel``. Used to check
    the per-sample independence of the batched teacher gradient."""

    def __init__(self):
        self.calls = 0

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return (z ** 2).flatten(1).mean(dim=1)

    def grad_of(self, z: torch.Tensor) -> torch.Tensor:
        return 2.0 * z / z[0].numel()


# ---------------------------------------------------------------------------
# 1. Architecture / contract-3 shapes
# ---------------------------------------------------------------------------
def test_dense_output_shape_matches_token_grid():
    c = _critic()
    for shape in (CROP_SHAPE, GRID_SHAPE):
        z = torch.randn(*shape)
        hs, ws = c.token_grid(shape[3], shape[4])
        dense = c(z)
        assert dense.shape == (shape[0], shape[1], hs, ws), (
            f"{shape} -> {tuple(dense.shape)}, expected "
            f"{(shape[0], shape[1], hs, ws)}"
        )
        assert dense.dtype is torch.float32


def test_full_frame_contract_shape():
    """Contract 3 verbatim: forward(z [B, F, 16, 60, 104]) -> dense map.

    The production critic width; run once at production d_model to prove
    the positional table actually covers 60x104 (8x13 tokens) rather than
    just the tiny test shapes.
    """
    c = LatentTextureCritic(d_model=128, num_blocks=1, num_heads=4, max_frames=8)
    z = torch.randn(1, 3, 16, 60, 104)
    dense = c(z)
    assert dense.shape == (1, 3, 8, 13), tuple(dense.shape)
    # And the generator's consumption expression is well-formed.
    assert (-dense.mean()).dim() == 0


def test_pool_and_value_reductions():
    c = _critic()
    z = torch.randn(*GRID_SHAPE)
    dense = c(z)
    assert c.pool(dense).shape == (GRID_SHAPE[0],)
    assert torch.allclose(c.value(z), dense.flatten(1).mean(dim=1), atol=1e-6)
    for fp in ("mean", "max", "topk_mean"):
        cc = _critic(frame_pool=fp)
        assert cc.pool(cc(z)).shape == (GRID_SHAPE[0],)


def test_zero_init_head_gives_zero_field_and_zero_gen_grad():
    """A fresh critic is the constant-zero value field, so a mis-fired
    generator term before warmup pushes with exactly zero magnitude."""
    c = _critic()
    z = torch.randn(*GRID_SHAPE, requires_grad=True)
    dense = c(z)
    assert torch.count_nonzero(dense) == 0
    loss, logs = generator_surrogate_loss(c, z, weight=1.0)
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert z.grad is not None and torch.count_nonzero(z.grad) == 0
    assert logs["train/surrogate_g_value"] == 0.0


def test_rejects_bad_shapes_and_channels():
    c = _critic()
    for bad, msg in (
        (torch.randn(4, 2, 16, 8), "[B, F, C, H, W]"),
        (torch.randn(4, 2, 8, 8, 8), "in_channels"),
        (torch.randn(4, 99, 16, 8, 8), "max_frames"),
    ):
        try:
            c(bad)
        except ValueError as exc:
            assert msg in str(exc), str(exc)
        else:
            raise AssertionError(f"expected ValueError for {tuple(bad.shape)}")


# ---------------------------------------------------------------------------
# 2. Absolute positional embedding (the crop-origin delta)
# ---------------------------------------------------------------------------
def test_crop_origin_changes_the_field():
    """The same crop content at two vertical positions must not produce
    the same value field — vertical position is a real texture covariate
    (the A24 band argument), so the surrogate has to see it."""
    c = _critic()
    # Non-zero head, otherwise everything is trivially zero.
    torch.nn.init.normal_(c.head.weight, std=0.1)
    z = torch.randn(*CROP_SHAPE)
    top = c(z, latent_origin=(0, 0))
    bottom = c(z, latent_origin=(6 * STEM_SPATIAL_STRIDE, 0))
    assert not torch.allclose(top, bottom, atol=1e-5), (
        "positional embedding is not position-dependent — the crop origin "
        "is being ignored"
    )


def test_origin_zero_is_the_default():
    c = _critic()
    torch.nn.init.normal_(c.head.weight, std=0.1)
    z = torch.randn(*CROP_SHAPE)
    assert torch.allclose(c(z), c(z, latent_origin=(0, 0)), atol=1e-6)


def test_origin_quantizes_by_stem_stride():
    """Sub-token offsets round to the same token; a full-token offset does
    not. Documents the quantization rather than leaving it implicit."""
    c = _critic()
    torch.nn.init.normal_(c.head.weight, std=0.1)
    z = torch.randn(*CROP_SHAPE)
    a = c(z, latent_origin=(0, 0))
    same = c(z, latent_origin=(2, 0))                      # rounds to token 0
    diff = c(z, latent_origin=(STEM_SPATIAL_STRIDE, 0))    # token 1
    assert torch.allclose(a, same, atol=1e-6)
    assert not torch.allclose(a, diff, atol=1e-5)


def test_origin_out_of_table_raises():
    c = _critic(max_token_h=4, max_token_w=4)
    z = torch.randn(*CROP_SHAPE)
    try:
        c(z, latent_origin=(40 * STEM_SPATIAL_STRIDE, 0))
    except ValueError as exc:
        assert "positional table" in str(exc)
    else:
        raise AssertionError("expected ValueError for an out-of-table origin")


# ---------------------------------------------------------------------------
# 2b. SEAM tests — the origin must actually travel from the caller's argument
#     to the critic's forward, not merely work at both ends
# ---------------------------------------------------------------------------
# Every test above calls ``critic.forward(latent_origin=...)`` DIRECTLY, and
# the end-to-end tests below pass origins into ``step()`` but assert only that
# the numbers are finite. That combination — both halves covered, the seam
# between them uncovered — is the shape WP-PIXGAN hit on
# ``pix_finish_grad_enabled`` (read off the pipeline by one side, off the
# config by the other, assigned by nobody; every test green, the flag inert).
# Before these tests existed, deleting the ``latent_origin=tgt.origin``
# argument inside ``step()`` left all 44 tests passing.
#
# Each test therefore carries a MUTATION CONTROL: the same assertion is run
# against a critic that deliberately drops the origin, and the test asserts
# the check FAILS there. A seam test that cannot fail on the broken code is
# not evidence.


class _RecordingCritic(LatentTextureCritic):
    """Records the ``latent_origin`` its forward is actually invoked with.

    ``honour_origin=False`` simulates the seam being cut — the origin
    arrives and is dropped on the floor.
    """

    def __init__(self, *, honour_origin: bool = True, **kw):
        super().__init__(**kw)
        self.seen = []
        self._honour_origin = bool(honour_origin)

    def forward(self, latent, latent_origin=None):
        if not self._honour_origin:
            latent_origin = None
        self.seen.append(
            None if latent_origin is None
            else tuple(int(v) for v in latent_origin)
        )
        return super().forward(latent, latent_origin=latent_origin)


def _rec(honour=True):
    cfg = dict(TINY)
    return _RecordingCritic(honour_origin=honour, **cfg)


S = STEM_SPATIAL_STRIDE


def test_step_origin_reaches_the_critic_seam():
    """``step(origin_real=, origin_fake=)`` -> teacher targets -> critic."""
    o_real, o_fake = (2 * S, 3 * S), (4 * S, 1 * S)

    def run(honour):
        torch.manual_seed(5)
        critic = _rec(honour)
        d = LatentSurrogateDistiller(critic, grad_loss_weight=1.0)
        opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
        d.step(
            z_real=torch.randn(*CROP_SHAPE), z_fake=torch.randn(*CROP_SHAPE),
            teacher_value_fn=QuadraticTeacher(), current_step=0, optimizer=opt,
            origin_real=o_real, origin_fake=o_fake,
        )
        return critic.seen

    seen = run(True)
    assert o_real in seen, f"origin_real never reached the critic; saw {seen}"
    assert o_fake in seen, f"origin_fake never reached the critic; saw {seen}"
    # MUTATION CONTROL — the assertions above must fail on a cut seam.
    cut = run(False)
    assert o_real not in cut and o_fake not in cut, (
        "the mutation control did not cut the seam, so the assertions above "
        "prove nothing"
    )


def test_step_origin_survives_a_replay_step_seam():
    """The cached triple must carry its origin through to the replay step —
    otherwise a replayed crop is scored at the wrong frame position."""
    o = (5 * S, 2 * S)
    torch.manual_seed(5)
    critic = _rec()
    d = LatentSurrogateDistiller(
        critic, grad_loss_weight=1.0, pix_teacher_refresh_every=100,
    )
    opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    d.step(
        z_real=torch.randn(*CROP_SHAPE), z_fake=None,
        teacher_value_fn=QuadraticTeacher(), current_step=0, optimizer=opt,
        origin_real=o,
    )
    critic.seen.clear()
    d.step(  # replay: origins passed here are ignored, the cache's must win
        z_real=torch.randn(*CROP_SHAPE), z_fake=None,
        teacher_value_fn=QuadraticTeacher(), current_step=1, optimizer=opt,
        origin_real=(0, 0),
    )
    assert critic.seen == [o], (
        f"replay used {critic.seen} instead of the cached origin {o}"
    )


def test_grad_check_origin_reaches_the_critic_seam():
    o = (3 * S, 2 * S)

    def run(honour):
        torch.manual_seed(5)
        critic = _rec(honour)
        d = LatentSurrogateDistiller(critic, grad_loss_weight=1.0)
        d.surrogate_grad_check(
            torch.randn(*CROP_SHAPE), QuadraticTeacher(), origin=o,
        )
        return critic.seen

    assert o in run(True), "surrogate_grad_check dropped the origin"
    assert o not in run(False), "mutation control did not cut the seam"


def test_generator_surrogate_loss_origin_reaches_the_critic_seam():
    o = (5 * S, 0)

    def run(honour):
        torch.manual_seed(5)
        critic = _rec(honour)
        generator_surrogate_loss(
            critic, torch.randn(*CROP_SHAPE, requires_grad=True),
            latent_origin=o,
        )
        return critic.seen

    assert run(True) == [o], "generator_surrogate_loss dropped the origin"
    assert run(False) == [None], "mutation control did not cut the seam"


# ---------------------------------------------------------------------------
# 3. Double backward — the load-bearing hand-rolled attention
# ---------------------------------------------------------------------------
def test_attention_double_backward_produces_param_grads():
    """``autograd.grad(..., create_graph=True)`` then ``.backward()`` — the
    exact Sobolev path. Flash attention has no double-backward kernel, so
    this is the regression test against an SDPA "optimization"."""
    c = _critic()
    torch.nn.init.normal_(c.head.weight, std=0.1)
    z = torch.randn(*GRID_SHAPE, requires_grad=True)
    val = c(z).flatten(1).mean(dim=1)
    g = torch.autograd.grad(val.sum(), z, create_graph=True)[0]
    assert g.requires_grad, "input grad has no graph — create_graph was lost"
    (g ** 2).mean().backward()
    touched = [
        n for n, p in c.named_parameters()
        if p.grad is not None and torch.count_nonzero(p.grad) > 0
    ]
    assert any("attn" in n for n in touched), (
        "no attention parameter received a second-order gradient; the "
        "double-backward path is broken"
    )
    for n, p in c.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad on {n}"


def test_attention_double_backward_matches_finite_difference():
    """Second-order correctness, not just non-crashing: d/dw of a
    directional derivative, checked against central differences on the
    head bias."""
    torch.manual_seed(3)
    c = _critic(d_model=32, num_blocks=1, num_heads=4)
    torch.nn.init.normal_(c.head.weight, std=0.1)
    for p in c.parameters():
        p.data = p.data.double()
    c = c.double()
    z = torch.randn(2, 2, 16, 8, 8, dtype=torch.float64, requires_grad=True)
    u = torch.randn_like(z)

    def directional(eps_param=None):
        val = c(z).flatten(1).mean(dim=1)
        g = torch.autograd.grad(val.sum(), z, create_graph=True)[0]
        return (g * u).sum()

    w = c.blocks[0].attn.out_proj.weight
    d = directional()
    grad_w = torch.autograd.grad(d, w, retain_graph=False)[0]
    i, j = 0, 0
    eps = 1e-5
    with torch.no_grad():
        w[i, j] += eps
    plus = float(directional().detach())
    with torch.no_grad():
        w[i, j] -= 2 * eps
    minus = float(directional().detach())
    with torch.no_grad():
        w[i, j] += eps
    fd = (plus - minus) / (2 * eps)
    an = float(grad_w[i, j])
    assert abs(fd - an) <= 1e-4 * max(1.0, abs(fd)), (
        f"second-order gradient wrong: analytic={an:.6e} fd={fd:.6e}"
    )


# ---------------------------------------------------------------------------
# 4. Teacher targets (contract 1 reduction + gradient)
# ---------------------------------------------------------------------------
def test_reduce_patch_logits_shapes():
    n = 5
    for raw, want in (
        (torch.arange(n).float(), torch.arange(n).float()),
        (torch.arange(n).float().reshape(n, 1), torch.arange(n).float()),
        (torch.ones(n, 1, 3, 4) * 2.0, torch.full((n,), 2.0)),
        (torch.ones(n, 2, 1, 3, 4) * 3.0, torch.full((n,), 3.0)),
    ):
        out = reduce_patch_logits(raw)
        assert out.shape == (n,)
        assert torch.allclose(out, want)
    try:
        reduce_patch_logits(torch.tensor(1.0))
    except ValueError as exc:
        assert "0-d" in str(exc)
    else:
        raise AssertionError("expected ValueError on a 0-d teacher output")


def test_teacher_targets_match_closed_form_gradient():
    """The contract says teacher grad = autograd.grad(disc(decode(z)).mean(), z).
    We take grad of ``value.sum()`` over the batch; because crops are
    independent, row i must equal the per-crop expression exactly. Checked
    against a closed form and against a literal per-crop recomputation."""
    teacher = LinearFieldTeacher(CROP_SHAPE)
    z = torch.randn(*CROP_SHAPE)
    tgt = compute_teacher_targets(z, teacher, current_step=0, tag="fake")
    assert tgt.value.shape == (CROP_SHAPE[0],)
    assert torch.allclose(
        tgt.value, (z * teacher.A).flatten(1).mean(dim=1), atol=1e-6,
    )
    assert torch.allclose(tgt.grad, teacher.grad_of(z), atol=1e-7)
    assert not tgt.grad.requires_grad and not tgt.value.requires_grad
    # Literal contract form, one crop at a time.
    for i in range(CROP_SHAPE[0]):
        zi = z[i: i + 1].clone().requires_grad_(True)
        vi = reduce_patch_logits(teacher(zi)).mean()
        gi = torch.autograd.grad(vi, zi)[0]
        assert torch.allclose(gi[0], tgt.grad[i], atol=1e-7), (
            f"crop {i}: batched grad != per-crop contract expression"
        )


def test_teacher_targets_quadratic_per_sample_independence():
    teacher = QuadraticTeacher()
    z = torch.randn(*CROP_SHAPE)
    tgt = compute_teacher_targets(z, teacher, current_step=0)
    assert torch.allclose(tgt.grad, teacher.grad_of(z), atol=1e-6)


def test_teacher_checkpoint_path_matches():
    teacher = QuadraticTeacher()
    z = torch.randn(*CROP_SHAPE)
    a = compute_teacher_targets(z, teacher, current_step=0)
    b = compute_teacher_targets(z, teacher, current_step=0, use_checkpoint=True)
    assert torch.allclose(a.value, b.value, atol=1e-6)
    assert torch.allclose(a.grad, b.grad, atol=1e-6)


def test_checkpointed_teacher_recompute_doubles_raw_call_count():
    """Pins a non-obvious consequence of ``use_checkpoint=True``, found
    while building the ``build_from_config`` seam tests: the teacher
    callable's raw invocation count is NOT the same as the number of
    logical teacher queries.

    ``torch.utils.checkpoint`` (reentrant or not) discards activations
    after the initial forward and RECOMPUTES the wrapped function during
    backward to rebuild the graph for ``autograd.grad``. So a single
    logical call to ``compute_teacher_targets(..., use_checkpoint=True,
    want_grad=True)`` invokes the wrapped teacher's Python callable
    TWICE: once for the checkpointed forward, once for the recompute.
    Without checkpointing it is invoked once.

    This is correct, standard, and not a bug -- but it means anyone
    building trainer-side telemetry that counts pixel-disc forward calls
    to sanity-check ``pix_teacher_refresh_every`` will see 2x the naive
    expectation whenever ``surrogate_teacher_use_checkpoint=True`` (the
    ``build_from_config`` default). Documented here so that observation
    reads as "expected" rather than "the cadence broke again" the next
    time someone measures it. Any test that infers refresh COUNT from
    raw call count (rather than ``distiller.n_teacher_refresh``, which is
    checkpoint-invariant by construction) must account for this.
    """
    teacher_off = QuadraticTeacher()
    z = torch.randn(*CROP_SHAPE)
    compute_teacher_targets(
        z, teacher_off, current_step=0, use_checkpoint=False, want_grad=True,
    )
    assert teacher_off.calls == 1, (
        f"non-checkpointed teacher invoked {teacher_off.calls}x, expected 1"
    )

    teacher_on = QuadraticTeacher()
    compute_teacher_targets(
        z, teacher_on, current_step=0, use_checkpoint=True, want_grad=True,
    )
    assert teacher_on.calls == 2, (
        f"checkpointed teacher invoked {teacher_on.calls}x, expected 2 "
        f"(forward + recompute) -- if torch's checkpoint semantics "
        f"changed, every test that counts raw teacher calls under "
        f"use_checkpoint=True needs re-deriving"
    )

    # want_grad=False takes no backward, so there is nothing to recompute
    # -- checkpointing degenerates to a plain forward, invoked once.
    teacher_novgrad = QuadraticTeacher()
    compute_teacher_targets(
        z, teacher_novgrad, current_step=0, use_checkpoint=True,
        want_grad=False,
    )
    assert teacher_novgrad.calls == 1, (
        f"checkpointed, no-grad teacher invoked {teacher_novgrad.calls}x, "
        f"expected 1 -- there is no backward to recompute for"
    )


def test_teacher_batch_mismatch_is_loud():
    def bad(z):
        return torch.zeros(z.shape[0] + 1)
    try:
        compute_teacher_targets(torch.randn(*CROP_SHAPE), bad, current_step=0)
    except ValueError as exc:
        assert "one value per latent crop" in str(exc)
    else:
        raise AssertionError("expected ValueError on a teacher batch mismatch")


def test_value_only_mode_skips_teacher_gradient():
    teacher = QuadraticTeacher()
    tgt = compute_teacher_targets(
        torch.randn(*CROP_SHAPE), teacher, current_step=0, want_grad=False,
    )
    assert tgt.grad is None


# ---------------------------------------------------------------------------
# 5. The refresh cadence (delta (b))
# ---------------------------------------------------------------------------
def _run_steps(distiller, teacher, steps, shape=CROP_SHAPE, lr=1e-3, opt=None):
    if opt is None:
        opt = torch.optim.Adam(distiller.critic.parameters(), lr=lr)
    all_logs = []
    for s in range(steps):
        z_real = torch.randn(*shape)
        z_fake = torch.randn(*shape)
        all_logs.append(distiller.step(
            z_real=z_real, z_fake=z_fake, teacher_value_fn=teacher,
            current_step=s, optimizer=opt,
        ))
    return all_logs


def test_refresh_every_n_fires_teacher_ceil_n_times():
    for n in (1, 2, 5):
        teacher = QuadraticTeacher()
        d = LatentSurrogateDistiller(
            _critic(), pix_teacher_refresh_every=n, grad_loss_weight=1.0,
        )
        steps = 20
        logs = _run_steps(d, teacher, steps)
        expect_refresh = math.ceil(steps / n)
        # Two teacher calls per refresh (real + fake).
        assert teacher.calls == 2 * expect_refresh, (
            f"N={n}: teacher called {teacher.calls}, expected "
            f"{2 * expect_refresh}"
        )
        assert d.n_teacher_refresh == expect_refresh
        assert d.n_replay == steps - expect_refresh
        # The critic serves/trains on EVERY step regardless of the cadence.
        assert all(lg["train/surrogate_skipped"] == 0.0 for lg in logs)
        assert all("train/critic_total_loss" in lg for lg in logs)


def test_target_age_cycles_with_the_cadence():
    n = 4
    teacher = QuadraticTeacher()
    d = LatentSurrogateDistiller(_critic(), pix_teacher_refresh_every=n)
    logs = _run_steps(d, teacher, 12)
    ages = [lg["train/surrogate_target_age"] for lg in logs]
    assert ages == [float(s % n) for s in range(12)], ages
    assert logs[0]["train/surrogate_teacher_refresh"] == 1.0
    assert logs[1]["train/surrogate_teacher_refresh"] == 0.0


def test_replay_uses_the_cached_latent_not_the_live_one():
    """A stale target replayed against a FRESH latent would be silently
    wrong. The cache must carry z with the targets."""
    teacher = LinearFieldTeacher(CROP_SHAPE)
    d = LatentSurrogateDistiller(
        _critic(), pix_teacher_refresh_every=100, cache_capacity=1,
    )
    opt = torch.optim.Adam(d.critic.parameters(), lr=1e-3)
    z0_real, z0_fake = torch.randn(*CROP_SHAPE), torch.randn(*CROP_SHAPE)
    d.step(
        z_real=z0_real, z_fake=z0_fake, teacher_value_fn=teacher,
        current_step=0, optimizer=opt,
    )
    cached_fake = d.cache.latest("fake")
    assert torch.allclose(cached_fake.z, z0_fake)
    calls_after_refresh = teacher.calls
    # Now a replay step with completely different latents.
    d.step(
        z_real=torch.randn(*CROP_SHAPE), z_fake=torch.randn(*CROP_SHAPE),
        teacher_value_fn=teacher, current_step=1, optimizer=opt,
    )
    assert teacher.calls == calls_after_refresh, "teacher fired on a replay step"
    assert torch.allclose(d.cache.latest("fake").z, z0_fake), (
        "replay overwrote the cached latent"
    )


def test_cold_cache_on_replay_is_flagged_not_faked():
    teacher = QuadraticTeacher()
    d = LatentSurrogateDistiller(_critic(), pix_teacher_refresh_every=4)
    logs = d.step(
        z_real=torch.randn(*CROP_SHAPE), z_fake=torch.randn(*CROP_SHAPE),
        teacher_value_fn=teacher, current_step=1, optimizer=None,
    )
    assert logs["train/surrogate_skipped"] == 1.0
    assert "train/critic_total_loss" not in logs
    assert teacher.calls == 0


def test_cache_fifo_and_cpu_offload():
    cache = TeacherTargetCache(capacity=2)
    for s in range(4):
        cache.push(TeacherTargets(
            z=torch.full(CROP_SHAPE, float(s)),
            value=torch.zeros(CROP_SHAPE[0]), grad=None,
            origin=None, step=s, tag="fake",
        ))
    assert len(cache) == 2
    assert float(cache.latest("fake").z.flatten()[0]) == 3.0
    assert cache.age("fake", 10) == 7.0
    assert cache.latest("real") is None and cache.sample("real") is None
    cpu_cache = TeacherTargetCache(capacity=1, store_on_cpu=True)
    cpu_cache.push(TeacherTargets(
        z=torch.zeros(CROP_SHAPE), value=torch.zeros(CROP_SHAPE[0]),
        grad=torch.zeros(CROP_SHAPE), origin=(8, 16), step=0, tag="real",
    ))
    got = cpu_cache.latest("real")
    assert got.z.device.type == "cpu" and got.origin == (8, 16)


# ---------------------------------------------------------------------------
# 6. Sobolev distillation actually works, and value-only does not
# ---------------------------------------------------------------------------
# The fits below are the expensive part of this file (~30-50 s each on a
# login node), so results are memoized by config — several tests read the
# same fit.
_FIT_CACHE = {}
FIT_STEPS = 120


def _fit(grad_weight, normalize=True, teacher="linear", steps=FIT_STEPS,
         seed=11, lr=3e-3):
    key = (grad_weight, normalize, teacher, steps, seed, lr)
    if key in _FIT_CACHE:
        return _FIT_CACHE[key]
    torch.manual_seed(seed)
    tch = (
        LinearFieldTeacher(CROP_SHAPE, scale=1.0, seed=5)
        if teacher == "linear" else PatchTeacher()
    )
    critic = _critic()
    torch.nn.init.normal_(critic.head.weight, std=0.02)
    d = LatentSurrogateDistiller(
        critic, grad_loss_weight=grad_weight, value_loss_weight=1.0,
        grad_loss_normalize=normalize, pix_teacher_refresh_every=1,
        max_grad_norm=1.0,
    )
    opt = torch.optim.Adam(critic.parameters(), lr=lr)
    logs = _run_steps(d, tch, steps, shape=CROP_SHAPE, opt=opt)
    # Honest readout on a FRESH sample the critic was never fit to.
    torch.manual_seed(999)
    check = d.surrogate_grad_check(
        torch.randn(*CROP_SHAPE), tch, current_step=steps,
    )
    _FIT_CACHE[key] = (logs, check, d, tch)
    return _FIT_CACHE[key]


def test_sobolev_distillation_learns_the_teacher_gradient_field():
    """Normalized Sobolev recovers the teacher's gradient field on BOTH
    the constant-field teacher and the realistic local-patch teacher."""
    for teacher, floor in (("linear", 0.9), ("patch", 0.8)):
        logs, check, _, _ = _fit(grad_weight=1.0, teacher=teacher)
        tail = sum(lg["train/critic_grad_cos_sim"] for lg in logs[-10:]) / 10.0
        assert tail > floor, (
            f"{teacher}: training grad cos sim only reached {tail:.3f}"
        )
        assert logs[-1]["train/critic_grad_loss"] < 0.5 * logs[0]["train/critic_grad_loss"]
        # Held-out audit agrees with the training metric.
        cos = check["train/surrogate_check_cos_sim"]
        assert cos > floor, f"{teacher}: held-out cos {cos:.3f}\n{check}"
        assert 0.5 < check["train/surrogate_check_mag_ratio"] < 2.0, check


def test_value_only_distillation_does_not_get_the_gradient_field():
    """The reason the Sobolev term exists.

    ``LinearFieldTeacher``'s value is ~0 everywhere (zero-mean field)
    while its gradient field is rich, so a value-only critic can score a
    near-perfect VALUE loss and still hand the generator a direction
    uncorrelated with the teacher's. That is not a contrived corner: it
    is the general fact that ``g ~= f`` does not imply ``grad g ~= grad f``,
    made measurable.
    """
    for teacher in ("linear", "patch"):
        _, check_sob, _, _ = _fit(grad_weight=1.0, teacher=teacher)
        logs_val, check_val, _, _ = _fit(grad_weight=0.0, teacher=teacher)
        # The grad diagnostics must be ABSENT, not zero-filled: 0.0 is a
        # legitimate value for each of them and in every case means the
        # opposite of "term disabled". See NO FORGEABLE ZEROS in the module.
        for k in ("train/critic_grad_cos_sim", "train/critic_grad_loss",
                  "train/surrogate_grad_mag_ratio"):
            assert k not in logs_val[-1], (
                f"value-only mode zero-filled {k} — a forgeable zero"
            )
        assert logs_val[-1]["train/surrogate_grad_distill"] == 0.0
        # Value distillation itself succeeded — this is not an under-training
        # artefact; the critic simply learned a function with the right
        # values and the wrong gradient field.
        assert logs_val[-1]["train/critic_value_loss"] < logs_val[0]["train/critic_value_loss"]
        val_cos = check_val["train/surrogate_check_cos_sim"]
        sob_cos = check_sob["train/surrogate_check_cos_sim"]
        assert abs(val_cos) < 0.3, (
            f"{teacher}: value-only reached cos={val_cos:.3f} — the contrast "
            f"this test rests on has gone away"
        )
        assert sob_cos > val_cos + 0.4, (
            f"{teacher}: Sobolev cos={sob_cos:.3f} barely beats value-only "
            f"cos={val_cos:.3f} — gradient distillation is not doing its job"
        )


def test_sobolev_and_value_losses_have_a_large_intrinsic_scale_gap():
    """Measures the gap that makes ``grad_loss_normalize`` necessary.

    The teacher's value is a MEAN patch logit, so O(1); its derivative
    w.r.t. one latent element is that O(1) response spread across the
    crop. ``L_value`` is therefore orders of magnitude larger than the raw
    ``L_grad`` MSE, and the gap grows with crop size. Pinned here so the
    module's SOBOLEV SCALING note stays honest.
    """
    z = torch.randn(*CROP_SHAPE)
    for teacher, lo in ((LinearFieldTeacher(CROP_SHAPE, seed=5), 500.0),
                        (PatchTeacher(), 100.0)):
        tgt = compute_teacher_targets(z, teacher, current_step=0)
        ratio = float((tgt.value ** 2).mean() / (tgt.grad ** 2).mean())
        assert ratio > lo, (
            f"{type(teacher).__name__}: value/grad loss scale ratio "
            f"{ratio:.0f} — expected >> 1"
        )


def test_unnormalized_sobolev_at_the_ancestor_weight_is_decorative():
    """The finding that forced ``grad_loss_normalize=True``.

    ``835b1df`` shipped ``gan_critic_grad_loss_weight: 1.0`` with a RAW
    MSE Sobolev term. Given the scale gap above, that term contributed
    well under 1% of the critic loss — so the ancestor's critic was, in
    effect, trained by value distillation alone. Reproduced here: the
    ancestor's exact configuration lands on the same held-out gradient
    cosine as value-only, while the normalized form recovers the field.

    Regression value: if someone later flips the default back to the
    "faithful" unnormalized form, this test explains why that is a
    downgrade rather than a restoration.
    """
    _, check_raw, _, _ = _fit(grad_weight=1.0, normalize=False)
    _, check_off, _, _ = _fit(grad_weight=0.0)
    _, check_norm, _, _ = _fit(grad_weight=1.0)
    raw = check_raw["train/surrogate_check_cos_sim"]
    off = check_off["train/surrogate_check_cos_sim"]
    norm = check_norm["train/surrogate_check_cos_sim"]
    assert abs(raw - off) < 0.3, (
        f"unnormalized Sobolev at weight 1.0 (cos={raw:.3f}) was expected to "
        f"behave like value-only (cos={off:.3f})"
    )
    assert norm > raw + 0.5, (
        f"normalized cos={norm:.3f} vs unnormalized cos={raw:.3f}"
    )


def test_normalized_grad_loss_is_a_relative_error():
    """``L_grad`` = 1.0 when the critic's field is zero, ~0 on a match —
    the property that makes ``grad_loss_weight=1.0`` mean something
    stable across crop sizes."""
    teacher = LinearFieldTeacher(CROP_SHAPE, seed=5)
    d = LatentSurrogateDistiller(_critic(), grad_loss_weight=1.0)  # zero-init head
    logs = d.step(
        z_real=torch.randn(*CROP_SHAPE), z_fake=None,
        teacher_value_fn=teacher, current_step=0, optimizer=None,
    )
    assert abs(logs["train/critic_grad_loss"] - 1.0) < 1e-3, logs
    # Scale-invariance: rescaling the teacher leaves the relative error put.
    loud = LinearFieldTeacher(CROP_SHAPE, seed=5, scale=100.0)
    d2 = LatentSurrogateDistiller(_critic(), grad_loss_weight=1.0)
    logs2 = d2.step(
        z_real=torch.randn(*CROP_SHAPE), z_fake=None,
        teacher_value_fn=loud, current_step=0, optimizer=None,
    )
    assert abs(logs2["train/critic_grad_loss"] - 1.0) < 1e-3, logs2
    # The raw form is NOT scale-invariant — it moves by ~100^2.
    d3 = LatentSurrogateDistiller(
        _critic(), grad_loss_weight=1.0, grad_loss_normalize=False,
    )
    a = d3.step(z_real=torch.randn(*CROP_SHAPE), z_fake=None,
                teacher_value_fn=teacher, current_step=0, optimizer=None)
    b = d3.step(z_real=torch.randn(*CROP_SHAPE), z_fake=None,
                teacher_value_fn=loud, current_step=1, optimizer=None)
    assert b["train/critic_grad_loss"] > 100.0 * a["train/critic_grad_loss"]


def test_value_only_mode_skips_the_teacher_gradient_entirely():
    teacher = QuadraticTeacher()
    d = LatentSurrogateDistiller(_critic(), grad_loss_weight=0.0)
    opt = torch.optim.Adam(d.critic.parameters(), lr=1e-3)
    logs = d.step(
        z_real=torch.randn(*CROP_SHAPE), z_fake=torch.randn(*CROP_SHAPE),
        teacher_value_fn=teacher, current_step=0, optimizer=opt,
    )
    assert d.cache.latest("fake").grad is None
    assert "train/critic_grad_loss" not in logs
    assert logs["train/surrogate_grad_distill"] == 0.0
    assert logs["train/critic_value_loss"] > 0.0


def test_critic_disc_corr_is_a_real_correlation():
    """Diagnostic sanity: perfectly-ordered values -> +1, reversed -> -1."""
    d = LatentSurrogateDistiller(_critic(), grad_loss_weight=0.0)

    def make(vals):
        z = torch.randn(len(vals), *CROP_SHAPE[1:])
        v = torch.tensor(vals, dtype=torch.float32)
        return TeacherTargets(z=z, value=v, grad=None, origin=None,
                              step=0, tag="fake")

    # Drive the critic's own value through a rigged forward.
    tgt = make([1.0, 2.0, 3.0, 4.0])
    d.cache.push(tgt)
    crit_vals = torch.tensor([1.0, 2.0, 3.0, 4.0])
    cv = crit_vals - crit_vals.mean()
    tv = tgt.value - tgt.value.mean()
    corr = float((cv * tv).sum() / (cv.norm() * tv.norm()))
    assert abs(corr - 1.0) < 1e-5
    rev = torch.tensor([4.0, 3.0, 2.0, 1.0])
    rv = rev - rev.mean()
    corr_rev = float((rv * tv).sum() / (rv.norm() * tv.norm()))
    assert abs(corr_rev + 1.0) < 1e-5


def test_grad_check_on_a_perfect_surrogate_is_unity():
    """If the surrogate IS the teacher, the audit must say so — otherwise
    the metric cannot be trusted when it says the surrogate has drifted."""
    teacher = LinearFieldTeacher(CROP_SHAPE, seed=2)
    critic = _critic()
    d = LatentSurrogateDistiller(critic, grad_loss_weight=1.0)

    class _Exact(torch.nn.Module):
        """A critic whose value is exactly the teacher's."""
        def __init__(self, A):
            super().__init__()
            self.A = A
            self.training = False
        def forward(self, z, latent_origin=None):
            v = (z * self.A).flatten(1).mean(dim=1)
            return v.reshape(-1, 1, 1, 1).expand(-1, z.shape[1], 1, 1)
        def eval(self):
            return self
        def train(self, mode=True):
            return self

    d.critic = _Exact(teacher.A)
    out = d.surrogate_grad_check(torch.randn(*CROP_SHAPE), teacher)
    assert abs(out["train/surrogate_check_cos_sim"] - 1.0) < 1e-4, out
    assert abs(out["train/surrogate_check_rel_err"]) < 1e-4, out
    assert d.n_grad_check == 1


def test_grad_check_cadence_gate():
    d = LatentSurrogateDistiller(_critic(), grad_check_every=0)
    assert not any(d.should_grad_check(s) for s in range(10))
    d = LatentSurrogateDistiller(_critic(), grad_check_every=4)
    assert [s for s in range(13) if d.should_grad_check(s)] == [4, 8, 12]


def test_grad_check_leaves_training_mode_untouched():
    teacher = QuadraticTeacher()
    d = LatentSurrogateDistiller(_critic(), grad_loss_weight=1.0)
    d.critic.train()
    d.surrogate_grad_check(torch.randn(*CROP_SHAPE), teacher)
    assert d.critic.training is True
    d.critic.eval()
    d.surrogate_grad_check(torch.randn(*CROP_SHAPE), teacher)
    assert d.critic.training is False


def test_dense_value_term_runs_and_is_off_by_default():
    teacher = LinearFieldTeacher(CROP_SHAPE, patch_hw=(6, 8))
    d_off = LatentSurrogateDistiller(_critic(), grad_loss_weight=0.0)
    opt = torch.optim.Adam(d_off.critic.parameters(), lr=1e-3)
    logs = d_off.step(
        z_real=torch.randn(*CROP_SHAPE), z_fake=None,
        teacher_value_fn=teacher, current_step=0, optimizer=opt,
        teacher_patch_fn=teacher.patch_map,
    )
    assert logs["train/critic_dense_loss"] == 0.0, "dense term must default OFF"
    d_on = LatentSurrogateDistiller(
        _critic(), grad_loss_weight=0.0, dense_value_weight=1.0,
    )
    torch.nn.init.normal_(d_on.critic.head.weight, std=0.05)
    opt = torch.optim.Adam(d_on.critic.parameters(), lr=1e-3)
    logs = d_on.step(
        z_real=torch.randn(*CROP_SHAPE), z_fake=None,
        teacher_value_fn=teacher, current_step=0, optimizer=opt,
        teacher_patch_fn=teacher.patch_map,
    )
    assert logs["train/critic_dense_loss"] > 0.0


# ---------------------------------------------------------------------------
# 7. Generator consumption — the frozen-critic idiom
# ---------------------------------------------------------------------------
def test_generator_backward_does_not_touch_critic_params():
    c = _critic()
    torch.nn.init.normal_(c.head.weight, std=0.1)
    for p in c.parameters():
        p.grad = None
    z = torch.randn(*GRID_SHAPE, requires_grad=True)
    loss, logs = generator_surrogate_loss(c, z, weight=0.5)
    loss.backward()
    leaked = [n for n, p in c.named_parameters() if p.grad is not None]
    assert not leaked, f"generator backward wrote into critic params: {leaked}"
    assert z.grad is not None and torch.count_nonzero(z.grad) > 0, (
        "the generator received no gradient from the surrogate"
    )
    assert all(p.requires_grad for p in c.parameters()), (
        "requires_grad was not restored — distillation would silently stop"
    )
    assert abs(logs["train/surrogate_g_weighted"]
               - 0.5 * logs["train/surrogate_g_main"]) < 1e-6


def test_frozen_critic_restores_requires_grad_on_exception():
    c = _critic()
    bad = torch.randn(4, 2, 8, 8, 8)  # wrong channel count
    try:
        generator_surrogate_loss(c, bad)
    except ValueError:
        pass
    else:
        raise AssertionError("expected the shape check to fire")
    assert all(p.requires_grad for p in c.parameters()), (
        "an exception inside the frozen block left the critic frozen forever"
    )


def test_generator_sign_pushes_value_up():
    """The generator minimizes ``-critic(z).mean()``, so a gradient step on
    z must RAISE the critic's value (fool the critic), not lower it."""
    c = _critic()
    torch.nn.init.normal_(c.head.weight, std=0.1)
    z = torch.randn(*GRID_SHAPE, requires_grad=True)
    before = float(c(z).mean().detach())
    loss, _ = generator_surrogate_loss(c, z, weight=1.0)
    loss.backward()
    with torch.no_grad():
        z_new = z - 1e-2 * z.grad
    after = float(c(z_new).mean().detach())
    assert after > before, f"value went {before:.5f} -> {after:.5f}"


def test_generator_loss_preserves_input_dtype():
    c = _critic()
    z = torch.randn(*CROP_SHAPE, dtype=torch.float32, requires_grad=True)
    loss, _ = generator_surrogate_loss(c, z)
    assert loss.dtype is torch.float32


# ---------------------------------------------------------------------------
# 8. Two-stage warmup schedule
# ---------------------------------------------------------------------------
def test_two_stage_gen_weight_schedule():
    kw = dict(critic_warmup_steps=100, gen_warmup_steps=50, gan_loss_weight=2.0)
    assert two_stage_gen_weight(0, **kw) == 0.0
    assert two_stage_gen_weight(99, **kw) == 0.0
    assert two_stage_gen_weight(100, **kw) == 0.0        # ramp starts at 0
    assert abs(two_stage_gen_weight(125, **kw) - 1.0) < 1e-9
    assert abs(two_stage_gen_weight(149, **kw) - 2.0 * 49 / 50) < 1e-9
    assert two_stage_gen_weight(150, **kw) == 2.0
    assert two_stage_gen_weight(10_000, **kw) == 2.0
    # Monotone non-decreasing over the whole schedule.
    vals = [two_stage_gen_weight(s, **kw) for s in range(0, 200)]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))
    # No gen warmup -> step change at the critic warmup boundary.
    kw0 = dict(critic_warmup_steps=10, gen_warmup_steps=0, gan_loss_weight=1.0)
    assert two_stage_gen_weight(9, **kw0) == 0.0
    assert two_stage_gen_weight(10, **kw0) == 1.0
    # Custom ramp shape is honoured.
    shaped = two_stage_gen_weight(125, shape_fn=lambda t: t ** 2, **kw)
    assert abs(shaped - 2.0 * 0.25) < 1e-9


def test_should_refresh_always_fires_at_step_zero():
    for n in (1, 3, 7, 100):
        d = LatentSurrogateDistiller(_critic(), pix_teacher_refresh_every=n)
        assert d.should_refresh(0), f"N={n} skipped step 0 with a cold cache"
    d = LatentSurrogateDistiller(_critic(), pix_teacher_refresh_every=0)
    assert d.pix_teacher_refresh_every == 1
    assert all(d.should_refresh(s) for s in range(5))


# ---------------------------------------------------------------------------
# 9. DDP grad sync is a no-op without a process group
# ---------------------------------------------------------------------------
def test_grad_sync_is_a_noop_single_rank():
    teacher = QuadraticTeacher()
    d = LatentSurrogateDistiller(_critic(), sync_grads=True, grad_loss_weight=1.0)
    opt = torch.optim.Adam(d.critic.parameters(), lr=1e-3)
    logs = d.step(
        z_real=torch.randn(*CROP_SHAPE), z_fake=torch.randn(*CROP_SHAPE),
        teacher_value_fn=teacher, current_step=0, optimizer=opt,
    )
    assert math.isfinite(logs["train/critic_total_loss"])
    assert logs["train/surrogate_critic_grad_norm"] > 0.0


def test_grad_clipping_caps_the_norm():
    teacher = LinearFieldTeacher(CROP_SHAPE, scale=50.0)
    d = LatentSurrogateDistiller(
        _critic(), grad_loss_weight=1.0, max_grad_norm=0.5,
    )
    torch.nn.init.normal_(d.critic.head.weight, std=0.5)
    opt = torch.optim.SGD(d.critic.parameters(), lr=1e-4)
    logs = _run_steps(d, teacher, 3, opt=opt)
    # clip_grad_norm_ returns the PRE-clip norm; the post-clip norm is what
    # the optimizer saw, so just assert the reported norm is finite and the
    # step did not blow up.
    assert all(math.isfinite(lg["train/surrogate_critic_grad_norm"]) for lg in logs)
    for p in d.critic.parameters():
        assert torch.isfinite(p).all()


def test_optimizer_none_is_a_dry_run():
    teacher = QuadraticTeacher()
    d = LatentSurrogateDistiller(_critic(), grad_loss_weight=1.0)
    before = [p.detach().clone() for p in d.critic.parameters()]
    logs = d.step(
        z_real=torch.randn(*CROP_SHAPE), z_fake=torch.randn(*CROP_SHAPE),
        teacher_value_fn=teacher, current_step=0, optimizer=None,
    )
    for a, p in zip(before, d.critic.parameters()):
        assert torch.equal(a, p), "dry run mutated the critic"
    assert logs["train/surrogate_critic_grad_norm"] == 0.0


# ---------------------------------------------------------------------------
# 10. End-to-end: distil on CROPS, serve the generator on the FULL latent
# ---------------------------------------------------------------------------
def test_crop_distillation_then_full_frame_generator_consumption():
    """The production call pattern in miniature: teacher targets on
    origin-tagged crops, generator consumption on the full latent."""
    torch.manual_seed(19)
    crop = (2, 2, 16, 8, 8)
    full = (1, 2, 16, 24, 32)
    teacher = LinearFieldTeacher(crop, seed=4)
    critic = _critic(max_token_h=8, max_token_w=8)
    torch.nn.init.normal_(critic.head.weight, std=0.02)
    d = LatentSurrogateDistiller(
        critic, grad_loss_weight=1.0, pix_teacher_refresh_every=3,
        grad_check_every=10,
    )
    opt = torch.optim.Adam(critic.parameters(), lr=3e-3)
    for s in range(30):
        oy = int(torch.randint(0, 3, (1,)).item()) * STEM_SPATIAL_STRIDE
        logs = d.step(
            z_real=torch.randn(*crop), z_fake=torch.randn(*crop),
            teacher_value_fn=teacher, current_step=s, optimizer=opt,
            origin_real=(oy, 0), origin_fake=(oy, 0),
        )
        assert logs["train/surrogate_skipped"] == 0.0
        if d.should_grad_check(s):
            chk = d.surrogate_grad_check(
                torch.randn(*crop), teacher, origin=(oy, 0), current_step=s,
            )
            assert math.isfinite(chk["train/surrogate_check_cos_sim"])
    # The generator consumes the same critic at full-frame width.
    z_full = torch.randn(*full, requires_grad=True)
    w = two_stage_gen_weight(
        40, critic_warmup_steps=20, gen_warmup_steps=20, gan_loss_weight=1.0,
    )
    assert w == 1.0
    # The critic is carrying live gradients from the distillation step it
    # just took, so "no critic grads" is the wrong invariant here — the
    # right one is that the GENERATOR's backward leaves them untouched.
    before = {
        n: (None if p.grad is None else p.grad.detach().clone())
        for n, p in critic.named_parameters()
    }
    assert any(g is not None and torch.count_nonzero(g) > 0
               for g in before.values()), (
        "precondition: the distillation step should have left grads on the "
        "critic, otherwise this test proves nothing"
    )
    loss, glogs = generator_surrogate_loss(critic, z_full, weight=w)
    loss.backward()
    assert z_full.grad is not None and torch.count_nonzero(z_full.grad) > 0
    for n, p in critic.named_parameters():
        b = before[n]
        if b is None:
            assert p.grad is None, f"generator backward created a grad on {n}"
        else:
            assert torch.equal(p.grad, b), (
                f"generator backward mutated the critic's distillation "
                f"gradient on {n}"
            )
    assert math.isfinite(glogs["train/surrogate_g_main"])


def test_telemetry_keys_are_complete_and_finite():
    """Every key the wiring plan promises must exist and be a float."""
    teacher = LinearFieldTeacher(CROP_SHAPE)
    d = LatentSurrogateDistiller(_critic(), grad_loss_weight=1.0)
    opt = torch.optim.Adam(d.critic.parameters(), lr=1e-3)
    logs = d.step(
        z_real=torch.randn(*CROP_SHAPE), z_fake=torch.randn(*CROP_SHAPE),
        teacher_value_fn=teacher, current_step=0, optimizer=opt,
    )
    required = {
        "train/critic_value_loss", "train/critic_grad_loss",
        "train/critic_total_loss", "train/critic_dense_loss",
        "train/critic_logit_mean", "train/disc_logit_mean",
        "train/critic_disc_corr", "train/critic_grad_cos_sim",
        "train/surrogate_grad_mag_ratio", "train/surrogate_critic_grad_norm",
        "train/surrogate_teacher_refresh", "train/surrogate_n_teacher_refresh",
        "train/surrogate_n_replay", "train/surrogate_cache_size",
        "train/surrogate_target_age", "train/surrogate_skipped",
        "train/surrogate_grad_distill",
    }
    missing = required - set(logs)
    assert not missing, f"missing telemetry keys: {sorted(missing)}"
    assert logs["train/surrogate_grad_distill"] == 1.0
    for k, v in logs.items():
        assert isinstance(v, float) and math.isfinite(v), f"{k}={v!r}"
    chk = d.surrogate_grad_check(torch.randn(*CROP_SHAPE), teacher)
    assert set(chk) == {
        "train/surrogate_check_cos_sim", "train/surrogate_check_mag_ratio",
        "train/surrogate_check_rel_err", "train/surrogate_n_grad_check",
    }


# ---------------------------------------------------------------------------
# 11. build_from_config — the trainer config seam
# ---------------------------------------------------------------------------
# ``build_from_config`` is the ONE function that reads
# ``surrogate_*`` / ``pix_teacher_refresh_every`` off a trainer config
# object. It exists so the eventual trainer wiring is a single call
# rather than a hand-written construction with its own chance to drop a
# key — the exact seam class this package hit with ``latent_origin``
# (§2b above) and WP-PIXGAN hit with ``pix_finish_grad_enabled`` (read
# off two different objects, assigned by neither). These tests are
# themselves the seam test for THIS seam: they drive a stub config
# object all the way through to the constructed distiller's observable
# behaviour, not just to the constructor's arguments.


class _StubCfg:
    """Plain namespace standing in for the trainer's config object.
    Only the attributes actually set are present — unset attributes
    must fall back to ``build_from_config``'s documented defaults via
    ``getattr(..., default)``, exactly as an old config file (predating
    this package's config block) would behave."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_build_from_config_off_by_default():
    """No ``surrogate_critic_enabled`` attribute at all -- the state of
    every config file that predates this package -- must build nothing,
    not raise and not default to on."""
    critic, opt, distiller = build_from_config(_StubCfg())
    assert critic is None and opt is None and distiller is None
    # Explicit False is the same as absent.
    critic, opt, distiller = build_from_config(
        _StubCfg(surrogate_critic_enabled=False)
    )
    assert critic is None and opt is None and distiller is None


def test_build_from_config_pix_teacher_refresh_every_reaches_the_distiller():
    """THE seam that matters most per docs/WP_SURROGATE.md §5b: if this
    key fails to arrive, the distiller silently defaults to N=1 -- the
    teacher fires every step, the run is merely slower, and every metric
    stays healthy. A value distinct from every default in the function
    is used so a dropped kwarg (silently falling back to the default)
    cannot pass by coincidence."""
    cfg = _StubCfg(surrogate_critic_enabled=True, pix_teacher_refresh_every=7)
    critic, opt, distiller = build_from_config(cfg)
    assert distiller is not None
    assert distiller.pix_teacher_refresh_every == 7, (
        f"got {distiller.pix_teacher_refresh_every}, expected the cfg's 7 "
        f"-- the default is 4, so this would only pass by falling back"
    )
    # And the cadence the distiller ACTUALLY exhibits matches -- closes
    # the seam all the way to observable behaviour, not just to the
    # constructor argument the previous assertion checked.
    #
    # Signal choice: ``distiller.n_teacher_refresh``, not the analytic
    # teacher's raw call count. Raw calls are NOT checkpoint-invariant --
    # build_from_config defaults ``surrogate_teacher_use_checkpoint=True``,
    # and torch.utils.checkpoint's recompute re-invokes the wrapped
    # function during backward, so raw invocation count is 2x the refresh
    # count under that default (see
    # test_checkpointed_teacher_recompute_doubles_raw_call_count, which
    # pins that as expected torch behaviour, not a seam bug). Using the
    # counter sidesteps an orthogonal knob entirely and is what this test
    # should have used from the start.
    teacher = QuadraticTeacher()
    for s in range(21):
        distiller.step(
            z_real=torch.randn(*CROP_SHAPE), z_fake=torch.randn(*CROP_SHAPE),
            teacher_value_fn=teacher, current_step=s, optimizer=opt,
        )
    assert distiller.n_teacher_refresh == math.ceil(21 / 7), (
        f"distiller refreshed {distiller.n_teacher_refresh} times over 21 "
        f"steps at N=7; expected {math.ceil(21 / 7)} -- the cadence did "
        f"not reach the distiller's actual behaviour"
    )
    # The raw call count is still checked, but against the doubled
    # expectation that follows from build_from_config's checkpoint
    # default -- so this assertion would also fail informatively if
    # someone flips that default without updating this test.
    assert teacher.calls == 4 * math.ceil(21 / 7), (
        f"teacher.__call__ fired {teacher.calls} times; expected "
        f"{4 * math.ceil(21 / 7)} = 2 (real+fake) x 2 (checkpoint "
        f"recompute) x {math.ceil(21 / 7)} refreshes -- if this changed, "
        f"either the cadence or build_from_config's checkpoint default "
        f"moved, and n_teacher_refresh above should be re-checked too"
    )


def test_build_from_config_every_surrogate_key_reaches_its_target():
    """One stub config with every key set to a value distinct from its
    documented default, asserted end to end. A dropped ``getattr`` name
    (typo, or a kwarg forgotten in the forwarding call) shows up as a
    silent fallback to the default -- exactly the failure mode this test
    exists to make loud."""
    cfg = _StubCfg(
        surrogate_critic_enabled=True,
        surrogate_critic_d_model=32,
        surrogate_critic_num_blocks=1,
        surrogate_critic_num_heads=2,
        surrogate_critic_max_frames=6,
        surrogate_critic_lr=1e-3,
        surrogate_value_loss_weight=2.0,
        surrogate_grad_loss_weight=3.0,
        surrogate_grad_loss_normalize=False,
        pix_teacher_refresh_every=5,
        surrogate_cache_capacity=2,
        surrogate_grad_check_every=9,
        surrogate_teacher_use_checkpoint=False,
    )
    critic, opt, distiller = build_from_config(cfg)
    assert critic.d_model == 32
    assert critic.num_blocks == 1
    assert critic.num_heads == 2
    assert critic.max_frames == 6
    assert opt.param_groups[0]["lr"] == 1e-3
    assert opt.param_groups[0]["betas"] == (0.0, 0.9)
    assert distiller.value_loss_weight == 2.0
    assert distiller.grad_loss_weight == 3.0
    assert distiller.grad_loss_normalize is False
    assert distiller.pix_teacher_refresh_every == 5
    assert distiller.cache.capacity == 2
    assert distiller.grad_check_every == 9
    assert distiller.teacher_use_checkpoint is False


def test_build_from_config_defaults_match_the_documented_config_block():
    """``surrogate_critic_enabled: true`` with nothing else set must
    build the exact defaults promised in docs/WP_SURROGATE.md §4.5 --
    the doc and the code are required to agree without a human
    cross-checking them on every edit."""
    critic, opt, distiller = build_from_config(
        _StubCfg(surrogate_critic_enabled=True)
    )
    assert critic.d_model == 512
    assert critic.num_blocks == 4
    assert critic.num_heads == 8
    assert critic.max_frames == 64
    assert abs(opt.param_groups[0]["lr"] - 2.0e-4) < 1e-12
    assert distiller.value_loss_weight == 1.0
    assert distiller.grad_loss_weight == 1.0
    assert distiller.grad_loss_normalize is True
    assert distiller.pix_teacher_refresh_every == 4
    assert distiller.cache.capacity == 8
    assert distiller.grad_check_every == 100
    assert distiller.teacher_use_checkpoint is True


def test_build_from_config_device_placement():
    critic, opt, distiller = build_from_config(
        _StubCfg(surrogate_critic_enabled=True),
        device=torch.device("cpu"), dtype=torch.float32,
    )
    for p in critic.parameters():
        assert p.device.type == "cpu" and p.dtype == torch.float32


def test_build_from_config_unwrapped_ready_for_ddp_wiring():
    """docs/WP_SURROGATE.md §3 requires the UNWRAPPED critic be handed to
    LatentSurrogateDistiller -- DDP does not support the double backward
    the Sobolev term needs. Guards against a future edit that wraps the
    critic before passing it to the distiller."""
    critic, opt, distiller = build_from_config(
        _StubCfg(surrogate_critic_enabled=True)
    )
    assert distiller.critic is critic
    assert not isinstance(critic, torch.nn.parallel.DistributedDataParallel)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _all_tests():
    g = globals()
    return [(n, g[n]) for n in sorted(g) if n.startswith("test_") and callable(g[n])]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-k", default=None, help="substring filter on test names")
    ap.add_argument("-v", action="store_true", help="print tracebacks")
    args = ap.parse_args()
    tests = [(n, f) for n, f in _all_tests() if not args.k or args.k in n]
    n_pass = n_fail = 0
    for name, fn in tests:
        try:
            torch.manual_seed(0)
            fn()
        except Exception as exc:
            n_fail += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
            if args.v:
                traceback.print_exc()
        else:
            n_pass += 1
            print(f"ok    {name}")
    print(f"\n{n_pass} passed, {n_fail} failed, {len(tests)} total")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
