"""Why the surrogate route delivered EXACTLY ZERO gradient to the
generator, and the two flags that fix it.

Companion to ``analysis/gan_tuning/PIXEL_FEATURE_SOURCE.md`` §7c, which
recorded the live reading and left the mechanism open:

    surrogate_param_grad_norm_unweighted   0      (825/825 reachable)
    surrogate_check_cos_sim                0.0097 (healthy ~0.986)
    surrogate_check_rel_err                0.99997
    surrogate_n_teacher_refresh            7 in 90 steps

Everything asserted below was measured on CPU first; the numbers in the
docstrings are those measurements, not targets.

CPU-only. Run with ``OMP_NUM_THREADS=8`` -- nproc is 144 on this box and
the suite hangs for 30+ minutes without it.
"""
import sys
import os

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.latent_texture_critic import (  # noqa: E402
    LatentTextureCritic,
    LatentSurrogateDistiller,
    build_from_config,
)

TINY = dict(in_channels=16, d_model=64, num_blocks=2, num_heads=4,
            max_frames=8, max_token_h=16, max_token_w=32)
CROP = (4, 2, 16, 8, 8)

# A live 90-step arm delivers exactly this many distillation calls. The
# trainer runs ``_maybe_run_surrogate_distillation`` from
# ``_streaming_train_one_chunk``, which executes only on GENERATOR
# iterations (``self.step % dfake_gen_update_ratio == 0``, and
# ``dfake_gen_update_ratio=5`` on every phase-3 arm), and it early-returns
# below ``gan_disc_start_step=20``. So steps 20,25,...,85 -> 14 calls, of
# which ``should_refresh`` (step % pix_teacher_refresh_every, =2) fires on
# 7 -- which is the ``surrogate_n_teacher_refresh = 7`` the live probe
# reported, to the unit.
PRODUCTION_CALLS = 14


class PatchTeacher:
    """The realistic stand-in for ``pixel_texture_disc o decode``, copied
    from ``test_latent_texture_critic.py``: local conv -> tanh -> patch
    logits, so the gradient field is z-dependent and spatially smooth."""

    def __init__(self, seed: int = 5, scale: float = 0.08) -> None:
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn(1, 16, 1, 3, 3, generator=g) * scale

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        x = z.permute(0, 2, 1, 3, 4)
        return torch.tanh(F.conv3d(
            x, self.W.to(z.dtype), stride=(1, 2, 2), padding=(0, 1, 1),
        ))


def _distil(substeps, *, head_init_std=0.0, lr=2e-4, seed=11,
            calls=PRODUCTION_CALLS, refresh=2):
    """Run the distiller for ``calls`` calls -- the live cadence -- and
    return the HELD-OUT audit plus the counters."""
    torch.manual_seed(seed)
    teacher = PatchTeacher()
    critic = LatentTextureCritic(head_init_std=head_init_std, **TINY)
    d = LatentSurrogateDistiller(
        critic, value_loss_weight=1.0, grad_loss_weight=1.0,
        grad_loss_normalize=True, pix_teacher_refresh_every=refresh,
        max_grad_norm=1.0, sync_grads=False, distill_substeps=substeps,
    )
    opt = torch.optim.Adam(critic.parameters(), lr=lr, betas=(0.0, 0.9))
    logs = None
    for s in range(calls):
        logs = d.step(
            z_real=torch.randn(*CROP), z_fake=torch.randn(*CROP) * 1.2,
            teacher_value_fn=teacher, current_step=s, optimizer=opt,
        )
    torch.manual_seed(999)
    chk = d.surrogate_grad_check(
        torch.randn(*CROP), teacher, current_step=calls,
    )
    return chk, logs, d, critic


# ---------------------------------------------------------------------------
# 1. THE MECHANISM: the shipped zero-init head is a saddle for the trunk
# ---------------------------------------------------------------------------
def test_zero_init_head_gives_the_trunk_exactly_zero_sobolev_gradient():
    """``value(z) = w . h(z) + b`` with ``h`` the mean-pooled token
    feature, so the input gradient the generator consumes,
    ``cg = d(value)/dz = J_h(z)^T w``, is EXACTLY LINEAR in the head
    weight. At ``w = 0`` we therefore get ``cg == 0`` AND
    ``dL_grad/d(trunk) == 0``, because that derivative carries a factor
    ``dcg/d(trunk) ~ w``.

    That is not "small", it is exactly zero, and it matters because the
    trunk is the ONLY part of the critic that can rotate its Jacobian
    toward the teacher's field: with the trunk frozen the reachable
    gradient subspace is only ``d_model``-dimensional, and the
    closed-form least-squares ceiling on the gradient cosine measures
    0.140 against the random-subspace value ``sqrt(d_model/n_z) = 0.144``.

    Measured trunk-gradient norms from the Sobolev term, by head std:
        0      -> 0.000e+00   (exactly)
        1e-4   -> 4.43e-02
        1e-3   -> 5.54e-01
        1e-2   -> 3.10e+01
    i.e. linear in |w|, from exactly nothing.
    """
    teacher = PatchTeacher()
    torch.manual_seed(3)
    z = torch.randn(*CROP)
    zz = z.clone().requires_grad_(True)
    tv = teacher(zz).flatten(1).mean(dim=1)
    tg = torch.autograd.grad(tv.sum(), zz)[0].detach()

    def trunk_sobolev_grad_norm(head_init_std):
        torch.manual_seed(5)
        critic = LatentTextureCritic(head_init_std=head_init_std, **TINY)
        z_in = z.clone().requires_grad_(True)
        val = critic(z_in).flatten(1).mean(dim=1)
        cg = torch.autograd.grad(
            val.sum(), z_in, create_graph=True, retain_graph=True,
        )[0]
        L_grad = ((cg - tg) ** 2).mean() / (tg.pow(2).mean() + 1e-12)
        named = list(critic.named_parameters())
        grads = torch.autograd.grad(
            L_grad, [p for _, p in named], allow_unused=True,
        )
        total = 0.0
        for (name, _), g in zip(named, grads):
            if g is None or name.startswith("head."):
                continue
            total += float(g.pow(2).sum())
        return total ** 0.5, float(cg.norm())

    zero_trunk, zero_cg = trunk_sobolev_grad_norm(0.0)
    warm_trunk, warm_cg = trunk_sobolev_grad_norm(0.02)

    # The saddle, stated exactly. Not `< 1e-8` -- it is 0.0.
    assert zero_cg == 0.0, (
        f"zero-init head should give an exactly-zero input gradient; "
        f"got |cg| = {zero_cg!r}"
    )
    assert zero_trunk == 0.0, (
        f"zero-init head should give the trunk EXACTLY zero Sobolev "
        f"gradient; got {zero_trunk!r}. If this is now merely tiny, the "
        f"head init changed and the mechanism note above is stale."
    )
    # And the flag breaks it.
    assert warm_cg > 0.0 and warm_trunk > 0.0, (
        f"head_init_std=0.02 must make the Sobolev term reach the trunk; "
        f"got |cg|={warm_cg:.3e} trunk_grad={warm_trunk:.3e}"
    )


def test_head_init_std_defaults_to_the_shipped_zero_init():
    """Default-off is byte-identical: same seed, same parameters."""
    torch.manual_seed(17)
    a = LatentTextureCritic(**TINY)
    torch.manual_seed(17)
    b = LatentTextureCritic(head_init_std=0.0, **TINY)
    assert a.head_init_applied is False
    assert float(a.head.weight.abs().max()) == 0.0
    assert float(a.head.bias.abs().max()) == 0.0
    for (n1, p1), (n2, p2) in zip(a.named_parameters(), b.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), f"{n1} differs at the default"
    torch.manual_seed(17)
    c = LatentTextureCritic(head_init_std=0.02, **TINY)
    assert c.head_init_applied is True
    assert float(c.head.weight.abs().max()) > 0.0


# ---------------------------------------------------------------------------
# 2. THE DOMINANT DEFECT: the critic never gets enough gradient steps
# ---------------------------------------------------------------------------
def test_production_cadence_starves_the_critic_and_substeps_fix_it():
    """THE regression test for §7c.

    At the live cadence the critic gets ``PRODUCTION_CALLS`` gradient
    steps in a 90-step arm (36 at MAXSTEPS=200) against the O(100) the
    CPU sweep says it needs -- the held-out gradient cosine measured
    ~0.01 at 14 critic steps, which is production's 0.0097, and ~0.80 by
    120. ``surrogate_distill_substeps`` buys those steps WITHOUT touching
    the teacher cadence: the sub-steps replay cached targets, so the
    expensive half (a graph-on decode + DINO forward + input gradient)
    still fires exactly on ``pix_teacher_refresh_every``.

    Both arms below make the SAME number of ``distiller.step`` calls and
    take the SAME number of teacher refreshes. The only difference is the
    flag, so this is single-variable.
    """
    shipped, _, d1, _ = _distil(substeps=1)
    fixed, _, dK, _ = _distil(substeps=12, head_init_std=0.02)

    # The cadence is genuinely unchanged -- the teacher was not simply
    # run more often, which would be a different (and expensive) change.
    assert d1.n_teacher_refresh == dK.n_teacher_refresh, (
        f"teacher cadence must be identical across the two arms; got "
        f"{d1.n_teacher_refresh} vs {dK.n_teacher_refresh}"
    )

    cos_shipped = shipped["train/surrogate_check_cos_sim"]
    cos_fixed = fixed["train/surrogate_check_cos_sim"]
    err_fixed = fixed["train/surrogate_check_rel_err"]

    assert cos_fixed > cos_shipped + 0.25, (
        f"substeps must materially improve the HELD-OUT gradient cosine: "
        f"shipped={cos_shipped:.4f} fixed={cos_fixed:.4f}. If the gap has "
        f"closed, either the shipped path improved (delete this test) or "
        f"the sub-step loop stopped taking extra optimizer steps."
    )
    assert cos_fixed > 0.6, (
        f"held-out gradient cosine only reached {cos_fixed:.4f}; the "
        f"generator consumes -dense.mean(), so this cosine IS the signal"
    )
    # rel_err is the key §7c read at 0.99997 ("the critic's field is zero
    # relative to the teacher's"). It must come off that ceiling.
    assert err_fixed < 0.95, (
        f"surrogate_check_rel_err {err_fixed:.5f} is still at the "
        f"'critic field is nothing' ceiling (§7c measured 0.99997)"
    )
    # And a non-vanishing magnitude: a correctly-pointed but 200x-too-small
    # gradient still multiplies out to nothing at the generator.
    assert fixed["train/surrogate_check_mag_ratio"] > 0.1, fixed


# ---------------------------------------------------------------------------
# 3. PROOF OF FIRE -- from a counter, never from the patch
# ---------------------------------------------------------------------------
def test_substep_counter_proves_the_flag_fired_at_runtime():
    """``surrogate_n_distill_substeps`` counts critic gradient steps
    ACTUALLY taken; ``surrogate_substeps_achieved`` is that over the
    number of distillation calls. Reading the config back cannot
    distinguish "flag set" from "flag honoured" -- this can. (21 arms
    once trained a noiser that never touched data.)"""
    _, logs1, d1, _ = _distil(substeps=1, calls=6)
    _, logsK, dK, _ = _distil(substeps=5, calls=6)

    assert logs1["train/surrogate_distill_substeps"] == 1.0
    assert logs1["train/surrogate_n_distill_substeps"] == 6.0
    assert logs1["train/surrogate_substeps_achieved"] == pytest.approx(1.0)
    assert d1.n_distill_substeps == d1.n_teacher_refresh + d1.n_replay

    assert logsK["train/surrogate_distill_substeps"] == 5.0
    assert logsK["train/surrogate_n_distill_substeps"] == 30.0
    assert logsK["train/surrogate_substeps_achieved"] == pytest.approx(5.0)
    assert dK.n_distill_substeps == 5 * (dK.n_teacher_refresh + dK.n_replay)


def test_substeps_take_real_optimizer_steps_not_just_a_counter():
    """The counter must not be forgeable either: more sub-steps must move
    the parameters further. Same seed, same data, same teacher."""
    _, _, _, c1 = _distil(substeps=1, calls=6, head_init_std=0.02)
    _, _, _, cK = _distil(substeps=5, calls=6, head_init_std=0.02)
    torch.manual_seed(11)
    ref = LatentTextureCritic(head_init_std=0.02, **TINY)
    d1 = sum(float((a - b).pow(2).sum())
             for a, b in zip(c1.parameters(), ref.parameters())) ** 0.5
    dK = sum(float((a - b).pow(2).sum())
             for a, b in zip(cK.parameters(), ref.parameters())) ** 0.5
    assert dK > 1.5 * d1, (
        f"5 sub-steps moved the critic {dK:.4e} from init against {d1:.4e} "
        f"for 1 -- the extra steps are not reaching the optimizer"
    )


def test_distill_substeps_default_is_byte_identical():
    """Constructing without the kwarg and with ``distill_substeps=1``
    must produce identical parameters after identical work."""
    _, _, da, ca = _distil(substeps=1, calls=5)

    torch.manual_seed(11)
    teacher = PatchTeacher()
    critic = LatentTextureCritic(**TINY)
    d = LatentSurrogateDistiller(
        critic, value_loss_weight=1.0, grad_loss_weight=1.0,
        grad_loss_normalize=True, pix_teacher_refresh_every=2,
        max_grad_norm=1.0, sync_grads=False,          # no distill_substeps
    )
    assert d.distill_substeps == 1
    opt = torch.optim.Adam(critic.parameters(), lr=2e-4, betas=(0.0, 0.9))
    for s in range(5):
        d.step(z_real=torch.randn(*CROP), z_fake=torch.randn(*CROP) * 1.2,
               teacher_value_fn=teacher, current_step=s, optimizer=opt)
    for (n1, p1), (n2, p2) in zip(ca.named_parameters(),
                                  critic.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), (
            f"{n1} diverged between the default and explicit substeps=1"
        )
    assert d.n_teacher_refresh == da.n_teacher_refresh


# ---------------------------------------------------------------------------
# 4. THE CONFIG SEAM -- build_from_config is the ONE reader
# ---------------------------------------------------------------------------
class _Cfg:
    def __init__(self, **kw):
        self.surrogate_critic_enabled = True
        self.surrogate_critic_d_model = 64
        self.surrogate_critic_num_blocks = 2
        self.surrogate_critic_num_heads = 4
        self.surrogate_critic_max_frames = 8
        for k, v in kw.items():
            setattr(self, k, v)


def test_build_from_config_reads_both_new_keys():
    """Both override keys must have a real read site -- OmegaConf's
    ``from_dotlist`` creates unknown keys silently, so a typo'd key is
    accepted without a murmur and the arm runs the baseline."""
    critic, _, dist = build_from_config(_Cfg())
    assert critic.head_init_applied is False, "default must stay zero-init"
    assert dist.distill_substeps == 1, "default must stay 1"

    critic, _, dist = build_from_config(_Cfg(
        surrogate_critic_head_init_std=0.02,
        surrogate_distill_substeps=12,
    ))
    assert critic.head_init_applied is True
    assert float(critic.head.weight.abs().max()) > 0.0
    assert dist.distill_substeps == 12


def test_build_from_config_still_returns_none_when_gated_off():
    class _Off:
        surrogate_critic_enabled = False
    assert build_from_config(_Off()) == (None, None, None)
