"""CPU tests for the One-Forcing GAN port (Option D).

Run:
    OMP_NUM_THREADS=8 python testing/test_one_forcing_gan.py

Covers the pure numerics in ``model/one_forcing_gan.py``, the parameterised
discriminator head in ``utils/wan_wrapper.py``, and the config->head seam
(with a mutation control, per ``docs/GAN_REDESIGN_TWO.md``: a seam test is
evidence only after it is shown to FAIL on a deliberately-broken copy).

No GPU, no distributed, no checkpoints.
"""

import os
import sys
import types
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ``wan/modules/t5.py`` evaluates ``torch.cuda.current_device()`` in a CLASS
# BODY at import time, so importing anything under ``wan`` (and therefore
# ``utils.wan_wrapper`` / ``model.action_model_patch``) needs a GPU. These
# tests are deliberately CPU-only, so stub the call for the import. Nothing
# under test touches CUDA.
if not torch.cuda.is_available():  # pragma: no cover - env shim
    torch.cuda.current_device = lambda: 0

from model.one_forcing_gan import (  # noqa: E402
    OF_DEFAULTS,
    add_noise_bf,
    apply_timestep_shift,
    duplicate_conditional_dict,
    finite_difference_penalty,
    logit_gap,
    nearest_gt_l1_match,
    of_discriminator_loss,
    of_generator_loss,
    of_weight_at_step,
    pair_shared_noise,
    disc_micro_batch_bounds,
    resolve_of_config,
    sample_of_timestep,
    slice_conditional_dict_rows,
    split_logits,
    validate_of_config,
)


class _Cfg:
    """Minimal config-like receiver."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _StubScheduler:
    """``FlowMatchScheduler.add_noise`` reduced to its contract.

    sigma = t/1000; x_t = (1-sigma)*x + sigma*eps, on [B*T, C, H, W].
    """

    def add_noise(self, original_samples, noise, timestep):
        assert original_samples.ndim == 4, original_samples.shape
        assert timestep.ndim == 1, timestep.shape
        sigma = (timestep.float() / 1000.0).reshape(-1, 1, 1, 1)
        return (1 - sigma) * original_samples + sigma * noise


# ======================================================================
# Config resolution / validation
# ======================================================================
class TestConfig(unittest.TestCase):
    def test_empty_config_is_off_and_matches_defaults(self):
        r = resolve_of_config(_Cfg())
        self.assertFalse(r["gan_of_enabled"])
        for k, v in OF_DEFAULTS.items():
            self.assertEqual(r[k], v, msg=f"default drift on {k}")

    def test_defaults_are_the_papers_framewise_values(self):
        # Read off ARRWM_data/one_forcing/config.yaml.
        self.assertEqual(OF_DEFAULTS["gan_of_g_weight"], 0.03)
        self.assertEqual(OF_DEFAULTS["gan_of_d_weight"], 0.03)
        self.assertEqual(OF_DEFAULTS["gan_of_feature_layers"], [21, 29])
        self.assertEqual(OF_DEFAULTS["gan_of_block_ffn_dim"], 2048)
        self.assertEqual(OF_DEFAULTS["gan_of_block_num_heads"], 12)
        self.assertFalse(OF_DEFAULTS["gan_of_relativistic"])
        self.assertEqual(OF_DEFAULTS["gan_of_r1_weight"], 0.0)
        self.assertEqual(OF_DEFAULTS["gan_of_r2_weight"], 0.0)

    def test_overrides_are_actually_read(self):
        r = resolve_of_config(_Cfg(
            gan_of_enabled=True, gan_of_g_weight=0.7,
            gan_of_feature_layers=[3, 9, 17],
            gan_of_block_ffn_dim=4096,
        ))
        self.assertTrue(r["gan_of_enabled"])
        self.assertEqual(r["gan_of_g_weight"], 0.7)
        self.assertEqual(r["gan_of_feature_layers"], [3, 9, 17])
        self.assertEqual(r["gan_of_block_ffn_dim"], 4096)

    def test_bad_sources_raise(self):
        with self.assertRaises(ValueError):
            resolve_of_config(_Cfg(
                gan_of_enabled=True, gan_of_fake_source="pred_iamge"))
        with self.assertRaises(ValueError):
            resolve_of_config(_Cfg(
                gan_of_enabled=True, gan_of_real_source="nearest"))

    def test_descending_or_duplicate_taps_raise(self):
        with self.assertRaises(ValueError):
            resolve_of_config(_Cfg(
                gan_of_enabled=True, gan_of_feature_layers=[29, 21]))
        with self.assertRaises(ValueError):
            resolve_of_config(_Cfg(
                gan_of_enabled=True, gan_of_feature_layers=[21, 21]))

    def test_timestep_bounds_validated(self):
        with self.assertRaises(ValueError):
            resolve_of_config(_Cfg(
                gan_of_enabled=True, gan_of_t_min=900, gan_of_t_max=100))
        with self.assertRaises(ValueError):
            resolve_of_config(_Cfg(gan_of_enabled=True, gan_of_t_max=2000))

    def test_validation_is_inert_while_the_arm_is_off(self):
        """A stale/typo'd ``gan_of_*`` key must NOT kill an unrelated run.

        ``resolve_of_config`` runs on EVERY phase-3 launch, including
        every arm that has never heard of One-Forcing. Validating an
        inert knob turned "you left a nonsense value in a block that
        does nothing" into a construction-time hard kill of a run that
        could not possibly have been affected by it. The predicates
        themselves are still exercised — with the gate ON — by the three
        tests above, and by ``force=True`` below."""
        for bad in (
            dict(gan_of_fake_source="pred_iamge"),
            dict(gan_of_real_source="nearest"),
            dict(gan_of_feature_layers=[29, 21]),
            dict(gan_of_t_min=900, gan_of_t_max=100),
            dict(gan_of_head_num_layers=0),
            dict(gan_of_timestep_shift=0.0),
        ):
            r = resolve_of_config(_Cfg(**bad))          # must not raise
            self.assertFalse(r["gan_of_enabled"])
            # ...but the SAME dict is still rejected when force-checked,
            # so "inert" never means "unvalidatable".
            with self.assertRaises(ValueError, msg=f"{bad} slipped through"):
                validate_of_config(r, force=True)

    def test_head_num_layers_floor(self):
        """The floor is 1, not 2 — 1 IS the paper's head.

        ``adding_cls_branch(num_layers=1)`` builds
        ``LayerNorm/Linear/SiLU/Linear``, which is One-Forcing's shape
        exactly; ``2`` adds a ``ResidualMLPBlock`` and a second
        ``LayerNorm`` and is therefore strictly HEAVIER than the paper,
        not equal to it. The old floor of 2 made the faithful
        configuration unreachable. ``0`` still raises: it would drop the
        final output projection."""
        self.assertEqual(OF_DEFAULTS["gan_of_head_num_layers"], 1)
        d = dict(OF_DEFAULTS)
        d["gan_of_head_num_layers"] = 1
        validate_of_config(d, force=True)               # must not raise
        for bad in (0, -1):
            d["gan_of_head_num_layers"] = bad
            with self.assertRaises(ValueError):
                validate_of_config(d, force=True)


# ======================================================================
# Timestep sampling
# ======================================================================
class TestTimestep(unittest.TestCase):
    def test_shift_one_is_identity(self):
        t = torch.tensor([0, 123, 500, 1000])
        self.assertTrue(torch.equal(apply_timestep_shift(t, 1.0), t))

    def test_shift_fixes_endpoints_and_is_monotone(self):
        t = torch.arange(0, 1001, 25)
        s = apply_timestep_shift(t, 5.0)
        self.assertEqual(int(s[0]), 0)
        self.assertEqual(int(s[-1]), 1000)
        self.assertTrue(torch.all(s[1:] >= s[:-1]))
        # shift > 1 pushes mass toward HIGH noise
        self.assertTrue(torch.all(s[1:-1] >= t[1:-1]))

    def test_shift_matches_reference_formula(self):
        # one_forcing/model/one_forcing.py :295-299
        for shift in (2.0, 5.0):
            for raw in (37, 250, 811):
                t = torch.tensor([raw])
                got = float(apply_timestep_shift(t, shift)[0])
                x = raw / 1000.0
                want = shift * x / (1 + (shift - 1) * x) * 1000.0
                self.assertAlmostEqual(got, int(want), delta=1.0)

    def test_sample_is_uniform_over_frames_and_in_range(self):
        t = sample_of_timestep(4, 6, 20, 980, 5.0, torch.device("cpu"))
        self.assertEqual(tuple(t.shape), (4, 6))
        self.assertEqual(t.dtype, torch.long)
        for b in range(4):
            self.assertEqual(len(set(t[b].tolist())), 1,
                             "t must be uniform across frames")
        self.assertTrue(int(t.min()) >= 20)
        self.assertTrue(int(t.max()) <= 980)

    def test_sample_uses_one_rng_draw_per_batch(self):
        g1 = torch.Generator().manual_seed(7)
        g2 = torch.Generator().manual_seed(7)
        a = sample_of_timestep(3, 5, 20, 980, 5.0, torch.device("cpu"), g1)
        b = sample_of_timestep(3, 5, 20, 980, 5.0, torch.device("cpu"), g2)
        self.assertTrue(torch.equal(a, b))


# ======================================================================
# Noise pairing
# ======================================================================
class TestNoisePairing(unittest.TestCase):
    def test_shared_noise_is_literally_the_same_tensor(self):
        f = torch.randn(2, 3, 4, 5, 6)
        r = torch.randn_like(f)
        ef, er = pair_shared_noise(f, r, shared=True)
        self.assertIs(ef, er)

    def test_unshared_draws_two(self):
        f = torch.randn(2, 3, 4, 5, 6)
        r = torch.randn_like(f)
        ef, er = pair_shared_noise(f, r, shared=False)
        self.assertFalse(torch.equal(ef, er))

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            pair_shared_noise(torch.randn(2, 3, 4, 5, 6),
                              torch.randn(2, 4, 4, 5, 6))

    def test_shared_noise_costs_exactly_one_draw(self):
        # Same seed, shared vs not: the FIRST epsilon must be identical,
        # proving the shared path does not draw a throwaway.
        torch.manual_seed(11)
        f = torch.randn(2, 3, 4, 5, 6)
        r = torch.randn_like(f)
        torch.manual_seed(99)
        a, _ = pair_shared_noise(f, r, shared=True)
        torch.manual_seed(99)
        b, _ = pair_shared_noise(f, r, shared=False)
        self.assertTrue(torch.equal(a, b))

    def test_add_noise_bf_roundtrips_shape(self):
        sch = _StubScheduler()
        x = torch.randn(2, 3, 4, 5, 6)
        eps = torch.randn_like(x)
        t = torch.full((2, 3), 500, dtype=torch.long)
        out = add_noise_bf(sch, x, eps, t)
        self.assertEqual(tuple(out.shape), tuple(x.shape))
        torch.testing.assert_close(out, 0.5 * x + 0.5 * eps)

    def test_add_noise_bf_rejects_bad_rank(self):
        with self.assertRaises(ValueError):
            add_noise_bf(_StubScheduler(), torch.randn(2, 3, 4, 5),
                         torch.randn(2, 3, 4, 5),
                         torch.zeros(2, 3, dtype=torch.long))

    def test_add_noise_bf_rejects_timestep_mismatch(self):
        with self.assertRaises(ValueError):
            add_noise_bf(_StubScheduler(), torch.randn(2, 3, 4, 5, 6),
                         torch.randn(2, 3, 4, 5, 6),
                         torch.zeros(2, 5, dtype=torch.long))


# ======================================================================
# Losses — checked against the reference formulas by hand
# ======================================================================
class TestLosses(unittest.TestCase):
    def test_d_loss_matches_reference(self):
        real = torch.tensor([[1.0], [2.0]])
        fake = torch.tensor([[-0.5], [0.25]])
        got = of_discriminator_loss(real, fake, relativistic=False,
                                    d_weight=0.03)
        want = (
            torch.nn.functional.softplus(-real).mean()
            + torch.nn.functional.softplus(fake).mean()
        ) * 0.03
        torch.testing.assert_close(got, want)

    def test_d_loss_relativistic_matches_reference(self):
        real = torch.tensor([[1.0], [2.0]])
        fake = torch.tensor([[-0.5], [0.25]])
        got = of_discriminator_loss(real, fake, relativistic=True,
                                    d_weight=1.0)
        want = torch.nn.functional.softplus(-(real - fake)).mean()
        torch.testing.assert_close(got, want)

    def test_g_loss_matches_reference(self):
        fake = torch.tensor([[-0.5], [0.25]])
        got = of_generator_loss(fake, None, relativistic=False, g_weight=0.03)
        want = torch.nn.functional.softplus(-fake).mean() * 0.03
        torch.testing.assert_close(got, want)

    def test_g_loss_relativistic_needs_real(self):
        with self.assertRaises(ValueError):
            of_generator_loss(torch.zeros(2, 1), None, relativistic=True)

    def test_losses_are_float32_even_from_bf16_logits(self):
        fake = torch.tensor([[0.3]], dtype=torch.bfloat16)
        real = torch.tensor([[0.7]], dtype=torch.bfloat16)
        self.assertEqual(of_generator_loss(fake).dtype, torch.float32)
        self.assertEqual(of_discriminator_loss(real, fake).dtype,
                         torch.float32)

    def test_g_loss_pushes_fake_logit_up(self):
        # The generator's job is to make d_fake LARGE.
        fake = torch.zeros(4, 1, requires_grad=True)
        of_generator_loss(fake, g_weight=1.0).backward()
        self.assertTrue(torch.all(fake.grad < 0),
                        "d(g_loss)/d(d_fake) must be negative")

    def test_d_loss_pushes_real_up_and_fake_down(self):
        real = torch.zeros(4, 1, requires_grad=True)
        fake = torch.zeros(4, 1, requires_grad=True)
        of_discriminator_loss(real, fake, d_weight=1.0).backward()
        self.assertTrue(torch.all(real.grad < 0))
        self.assertTrue(torch.all(fake.grad > 0))

    def test_finite_difference_penalty(self):
        base = torch.tensor([[0.0], [0.0]])
        pert = torch.tensor([[0.02], [-0.02]])
        got = finite_difference_penalty(pert, base, sigma=0.01, weight=2.0)
        torch.testing.assert_close(got, torch.tensor(8.0))

    def test_logit_gap(self):
        real = torch.tensor([[2.0], [4.0]])
        fake = torch.tensor([[1.0], [1.0]])
        torch.testing.assert_close(logit_gap(real, fake), torch.tensor(2.0))
        # symmetric — sign of the gap is not the metric
        torch.testing.assert_close(logit_gap(fake, real), torch.tensor(2.0))


# ======================================================================
# Weight schedule
# ======================================================================
class TestWeightSchedule(unittest.TestCase):
    def test_defaults_are_identity(self):
        for step in (0, 1, 10_000):
            self.assertEqual(of_weight_at_step(0.03, step, 0, 0), 0.03)

    def test_zero_before_start(self):
        self.assertEqual(of_weight_at_step(0.03, 9, 10, 0), 0.0)
        self.assertEqual(of_weight_at_step(0.03, 10, 10, 0), 0.03)

    def test_linear_ramp(self):
        self.assertAlmostEqual(of_weight_at_step(1.0, 0, 0, 10), 0.1)
        self.assertAlmostEqual(of_weight_at_step(1.0, 4, 0, 10), 0.5)
        self.assertAlmostEqual(of_weight_at_step(1.0, 100, 0, 10), 1.0)


# ======================================================================
# Batch bookkeeping
# ======================================================================
class TestBatchBookkeeping(unittest.TestCase):
    def test_duplicate_cond_dict(self):
        cond = {
            "prompt_embeds": torch.randn(2, 5, 8),
            "_action_tokens": torch.randn(2, 6, 8),
            "not_a_tensor": "hello",
        }
        out = duplicate_conditional_dict(cond)
        self.assertEqual(out["prompt_embeds"].shape[0], 4)
        self.assertEqual(out["_action_tokens"].shape[0], 4)
        self.assertEqual(out["not_a_tensor"], "hello")
        torch.testing.assert_close(out["prompt_embeds"][:2],
                                   out["prompt_embeds"][2:])

    def test_split_logits_is_fake_first(self):
        fake_rows = torch.full((3, 1), -1.0)
        real_rows = torch.full((3, 1), +1.0)
        f, r = split_logits(torch.cat([fake_rows, real_rows], 0), 3)
        torch.testing.assert_close(f, fake_rows)
        torch.testing.assert_close(r, real_rows)

    def test_split_logits_rejects_wrong_batch(self):
        with self.assertRaises(ValueError):
            split_logits(torch.zeros(5, 1), 3)


# ======================================================================
# Nearest-GT retrieval (the layer-on knob)
# ======================================================================
class TestNearestMatch(unittest.TestCase):
    def test_finds_the_planted_window(self):
        pool = torch.randn(2, 10, 3, 4, 4)
        fake = torch.stack([pool[0, 4:7], pool[1, 1:4]], dim=0)
        matched, offs = nearest_gt_l1_match(fake, pool)
        self.assertEqual(offs.tolist(), [4, 1])
        torch.testing.assert_close(matched, fake)

    def test_result_is_detached(self):
        pool = torch.randn(1, 5, 2, 2, 2, requires_grad=True)
        fake = torch.randn(1, 2, 2, 2, 2)
        matched, _ = nearest_gt_l1_match(fake, pool)
        self.assertFalse(matched.requires_grad)

    def test_short_pool_raises(self):
        with self.assertRaises(ValueError):
            nearest_gt_l1_match(torch.randn(1, 5, 2, 2, 2),
                                torch.randn(1, 3, 2, 2, 2))

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            nearest_gt_l1_match(torch.randn(1, 2, 2, 2, 2),
                                torch.randn(1, 5, 3, 2, 2))


# ======================================================================
# Discriminator head construction (utils/wan_wrapper.adding_cls_branch)
# ======================================================================
def _make_stub_wrapper(dim=32, n_blocks=8):
    """A stand-in for WanDiffusionWrapper carrying just what
    ``adding_cls_branch`` touches: ``_unwrapped_model()`` and nn.Module
    attribute registration."""
    from utils.wan_wrapper import WanDiffusionWrapper

    inner = nn.Module()
    inner.dim = dim
    inner.blocks = nn.ModuleList([nn.Identity() for _ in range(n_blocks)])

    stub = nn.Module()
    stub.model = inner
    stub._unwrapped_model = types.MethodType(
        lambda self: self.model, stub,
    )
    stub.adding_cls_branch = types.MethodType(
        WanDiffusionWrapper.adding_cls_branch, stub,
    )
    stub._cls_branch_modules = types.MethodType(
        WanDiffusionWrapper._cls_branch_modules, stub,
    )
    return stub, inner


class TestHeadConstruction(unittest.TestCase):
    def test_legacy_defaults_unchanged(self):
        """Default-off byte-identity for the pre-existing dmd2* callers.

        ``atten_dim`` is passed small ONLY to keep the login-node RSS
        sane; every other argument is left at its historical default,
        which is what this test is about."""
        stub, inner = _make_stub_wrapper(dim=96, n_blocks=30)
        stub.adding_cls_branch(atten_dim=96)
        # 4 taps -> 4 register tokens, 2 blocks each, ffn 8192
        self.assertEqual(len(stub._register_tokens.register_tokens), 4)
        self.assertEqual(len(stub._gan_ca_blocks), 4)
        self.assertEqual(len(stub._gan_ca_blocks[0]), 2)
        self.assertEqual(stub._gan_ca_blocks[0][0].ffn_dim, 8192)
        self.assertEqual(stub._gan_ca_blocks[0][0].num_heads, 12)
        self.assertEqual(inner._gan_feature_layers, [7, 13, 21, 29])
        # head: LN, Linear, SiLU, Dropout, 3x Residual, LN, Linear
        head = stub._cls_pred_branch
        self.assertIsInstance(head[0], nn.LayerNorm)
        self.assertEqual(head[1].in_features, 4 * 96)
        self.assertEqual(head[1].out_features, 3072)
        self.assertIsInstance(head[3], nn.Dropout)
        self.assertAlmostEqual(head[3].p, 0.2)
        self.assertEqual(head[-1].out_features, 1)
        # legacy mode attaches to the WRAPPER, not the model
        self.assertFalse(hasattr(inner, "_cls_pred_branch"))

    def test_paper_shape_when_configured(self):
        stub, inner = _make_stub_wrapper(dim=96, n_blocks=30)
        stub.adding_cls_branch(
            atten_dim=96, num_class=1, hidden_dim=128, num_layers=2,
            dropout=0.0, gan_blocks_per_token=1, layer_indices=[21, 29],
            block_ffn_dim=2048, block_num_heads=12, attach_to_model=True,
        )
        self.assertEqual(len(inner._register_tokens.register_tokens), 2)
        self.assertEqual(len(inner._gan_ca_blocks), 2)
        self.assertEqual(len(inner._gan_ca_blocks[0]), 1)
        self.assertEqual(inner._gan_ca_blocks[0][0].ffn_dim, 2048)
        self.assertEqual(inner._gan_feature_layers, [21, 29])
        head = inner._cls_pred_branch
        self.assertEqual(head[1].in_features, 2 * 96)
        self.assertEqual(head[1].out_features, 128)
        # dropout=0.0 -> no Dropout layer at all
        self.assertFalse(any(isinstance(m, nn.Dropout) for m in head))
        # head is NOT on the wrapper in attach_to_model mode
        self.assertFalse(hasattr(stub, "_cls_pred_branch"))

    def test_attached_head_is_inside_the_ddp_wrapped_module(self):
        """The property the whole arm's multi-node correctness rests on:
        head params must appear in ``fake_score.model.parameters()`` (=
        what DDP wraps and what fake_optimizer sweeps) and in that
        module's state_dict (= what the checkpoint saves)."""
        stub, inner = _make_stub_wrapper(dim=32, n_blocks=8)
        before = {id(p) for p in inner.parameters()}
        stub.adding_cls_branch(
            atten_dim=32, hidden_dim=32, num_layers=2, dropout=0.0,
            gan_blocks_per_token=1, layer_indices=[3, 7],
            block_ffn_dim=64, block_num_heads=4, attach_to_model=True,
        )
        after = {id(p) for p in inner.parameters()}
        self.assertGreater(len(after - before), 0)
        keys = set(inner.state_dict().keys())
        self.assertTrue(any(k.startswith("_cls_pred_branch.") for k in keys))
        self.assertTrue(any(k.startswith("_register_tokens.") for k in keys))
        self.assertTrue(any(k.startswith("_gan_ca_blocks.") for k in keys))

    def test_head_params_require_grad(self):
        stub, inner = _make_stub_wrapper(dim=32, n_blocks=8)
        stub.adding_cls_branch(
            atten_dim=32, hidden_dim=32, num_layers=2, dropout=0.0,
            gan_blocks_per_token=1, layer_indices=[3, 7],
            block_ffn_dim=64, block_num_heads=4, attach_to_model=True,
        )
        for name in ("_cls_pred_branch", "_register_tokens", "_gan_ca_blocks"):
            for p in getattr(inner, name).parameters():
                self.assertTrue(p.requires_grad, name)

    def test_cls_branch_modules_resolves_both_attach_modes(self):
        for attach in (False, True):
            stub, inner = _make_stub_wrapper(dim=32, n_blocks=8)
            stub.adding_cls_branch(
                atten_dim=32, hidden_dim=32, num_layers=2, dropout=0.0,
                gan_blocks_per_token=1, layer_indices=[3, 7],
                block_ffn_dim=64, block_num_heads=4, attach_to_model=attach,
            )
            reg, head, blocks = stub._cls_branch_modules()
            self.assertEqual(len(blocks), 2)
            self.assertEqual(head[-1].out_features, 1)
            self.assertEqual(len(reg.register_tokens), 2)

    def test_cls_branch_modules_fails_loud_when_absent(self):
        stub, _ = _make_stub_wrapper()
        with self.assertRaises(RuntimeError):
            stub._cls_branch_modules()


# ======================================================================
# SEAM: config -> resolved -> head geometry, WITH a mutation control.
# ======================================================================
class TestConfigToHeadSeam(unittest.TestCase):
    """A flag is not wired until one test drives it end-to-end from config
    to consumer. The mutation control below demonstrates this test FAILS on
    a deliberately-broken seam."""

    @staticmethod
    def _build_head_from_cfg(cfg, break_seam=False):
        resolved = resolve_of_config(cfg)
        stub, inner = _make_stub_wrapper(dim=64, n_blocks=32)
        if break_seam:
            # MUTATION: the classic failure — the consumer ignores the
            # resolved value and uses its own default instead.
            stub.adding_cls_branch(
                atten_dim=64, hidden_dim=64, num_layers=2, dropout=0.0,
                gan_blocks_per_token=1, layer_indices=None,
                block_ffn_dim=8192, block_num_heads=4,
                attach_to_model=True,
            )
        else:
            stub.adding_cls_branch(
                atten_dim=64,
                hidden_dim=resolved["gan_of_head_hidden_dim"],
                num_layers=resolved["gan_of_head_num_layers"],
                dropout=resolved["gan_of_head_dropout"],
                gan_blocks_per_token=resolved["gan_of_blocks_per_token"],
                layer_indices=resolved["gan_of_feature_layers"],
                block_ffn_dim=resolved["gan_of_block_ffn_dim"],
                block_num_heads=4,
                attach_to_model=True,
            )
        return inner

    def _assert_seam(self, inner):
        self.assertEqual(inner._gan_feature_layers, [5, 11, 17])
        self.assertEqual(len(inner._gan_ca_blocks), 3)
        self.assertEqual(len(inner._gan_ca_blocks[0]), 2)
        self.assertEqual(inner._gan_ca_blocks[0][0].ffn_dim, 1024)

    def _cfg(self):
        return _Cfg(
            gan_of_enabled=True,
            gan_of_feature_layers=[5, 11, 17],
            gan_of_blocks_per_token=2,
            gan_of_block_ffn_dim=1024,
            gan_of_head_hidden_dim=64,
        )

    def test_pristine_seam_passes(self):
        self._assert_seam(self._build_head_from_cfg(self._cfg()))

    def test_mutation_control_the_seam_test_fails_on_a_cut_seam(self):
        with self.assertRaises(AssertionError):
            self._assert_seam(
                self._build_head_from_cfg(self._cfg(), break_seam=True)
            )


# ======================================================================
# Tap-layer plumbing in the block loops (the second hard-coded list).
# ======================================================================
class TestTapPlumbing(unittest.TestCase):
    def test_model_module_reads_gan_feature_layers(self):
        """``wan/modules/model.py`` must consult ``_gan_feature_layers``
        rather than the historical literal. Source-level check, because
        the real forward needs flash-attention/CUDA."""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src = open(os.path.join(root, "wan", "modules", "model.py")).read()
        self.assertIn('_gan_taps = getattr(self, "_gan_feature_layers", None)',
                      src)
        self.assertIn("if classify_mode and ii in _gan_taps:", src)
        self.assertNotIn("if classify_mode and ii in [7, 13, 21, 29]:", src)

    def test_action_patch_reads_gan_feature_layers(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src = open(os.path.join(root, "model", "action_model_patch.py")).read()
        self.assertIn('gan_taps = getattr(self, "_gan_feature_layers", None)',
                      src)
        self.assertIn("if classify_mode and _blk_idx in gan_taps:", src)
        # The historical blanket refusal must be gone for classify_mode
        # (regress_mode keeps its own).
        self.assertNotIn(
            "classify_mode / regress_mode are not supported\n", src,
        )

    def test_stream_b_consumed_kwargs_covers_the_disc_kwargs(self):
        from model.action_model_patch import _STREAM_B_CONSUMED_KWARGS
        for k in ("classify_mode", "register_tokens", "cls_pred_branch",
                  "gan_ca_blocks", "concat_time_embeddings"):
            self.assertIn(k, _STREAM_B_CONSUMED_KWARGS)


class TestTapToHeadShapeContract(unittest.TestCase):
    """The tap -> register-token -> head seam, on CPU.

    ``flash_attention`` has no CPU implementation (it asserts on
    ``FLASH_ATTN_2_AVAILABLE``), so it is stubbed with SDPA for the
    duration. Everything else — ``RegisterTokens``, ``GanAttentionBlock``,
    the ``cat``/``view`` into the head — is the real code, and the shape
    arithmetic between them is exactly what a wrong tap count would break.
    """

    def test_two_taps_produce_one_logit_per_sample(self):
        import wan.modules.model as wanmodel

        def _sdpa(q, k, v, *a, **kw):
            # q [B, Lq, N, D], k/v [B, Lk, N, D] -> [B, Lq, N, D]
            out = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2).float(), k.transpose(1, 2).float(),
                v.transpose(1, 2).float(),
            )
            return out.transpose(1, 2).to(q.dtype)

        orig = wanmodel.flash_attention
        wanmodel.flash_attention = _sdpa
        try:
            B, L, dim, n_taps = 2, 17, 32, 2
            regs = wanmodel.RegisterTokens(num_registers=n_taps, dim=dim)
            blocks = nn.ModuleList([
                nn.ModuleList([
                    wanmodel.GanAttentionBlock(
                        dim=dim, ffn_dim=64, num_heads=4,
                    )
                ]) for _ in range(n_taps)
            ])
            head = nn.Sequential(
                nn.LayerNorm(n_taps * dim),
                nn.Linear(n_taps * dim, 16), nn.SiLU(),
                nn.LayerNorm(16), nn.Linear(16, 1),
            )
            x = torch.randn(B, L, dim)
            registers = regs().unsqueeze(0).expand(B, -1, -1)
            feats = []
            for i in range(n_taps):
                tok = registers[:, i:i + 1]
                self.assertEqual(tuple(tok.shape), (B, 1, dim))
                for blk in blocks[i]:
                    tok = blk(x, tok)
                self.assertEqual(tuple(tok.shape), (B, 1, dim),
                                 "GanAttentionBlock must preserve [B,1,C]")
                feats.append(tok)
            cat = torch.cat(feats, dim=1)
            self.assertEqual(tuple(cat.shape), (B, n_taps, dim))
            logits = head(cat.view(cat.shape[0], -1))
            self.assertEqual(tuple(logits.shape), (B, 1))
            # the seam is differentiable back into the tapped activations
            logits.sum().backward()
            self.assertIsNotNone(regs.register_tokens.grad)
        finally:
            wanmodel.flash_attention = orig


# ======================================================================
# STREAMING WIRING.
#
# ``streaming_mode: true`` is the base default and every real phase-3
# DMD arm runs it, so "wired on the non-streaming path" means "the arm
# cannot launch". These tests cover the streaming seam three ways:
#
#   1. REAL MODEL METHODS bound to a stub carrying only
#      ``streaming_state`` / ``of_cfg`` — the new geometry accessors are
#      pure tensor arithmetic and are exercised for real, on CPU.
#   2. REAL TRAINER HELPERS (``_of_streaming_g_term`` /
#      ``_of_streaming_d_term``) bound to a stub trainer, so the
#      argument threading, the DETACH on the D side and the log merge
#      are executed rather than read.
#   3. SOURCE-LEVEL assertions for the parts that need a live 1.3B
#      forward (the call sites and their ORDER relative to the two
#      backwards) plus the rank-uniformity properties, which are
#      statements about which predicates may appear in the gating.
# ======================================================================
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _trainer_src():
    with open(os.path.join(_ROOT, "trainer",
                           "causal_action_forcing_train.py")) as fh:
        return fh.read()


def _method_src(cls, name):
    import inspect
    return inspect.getsource(getattr(cls, name))


def _code_only(src):
    """Drop comment lines.

    Ordering assertions below index on statements like
    ``critic_loss.backward()``; the surrounding comments quote those same
    statements, so an un-stripped search finds the PROSE first and the
    test reports a false inversion."""
    return "\n".join(
        ln for ln in src.split("\n") if not ln.strip().startswith("#")
    )


class _OFModelStub:
    """Carries only what the new model accessors touch."""

    def __init__(self, streaming_state=None, of_cfg=None, noisy_cond=None):
        self.streaming_state = streaming_state
        self.of_cfg = of_cfg or dict(OF_DEFAULTS)
        self._noisy_cond = noisy_cond or ({}, {})

    def _streaming_noisy_cond_slice(self, info):
        return self._noisy_cond


def _bind_model_method(stub, name):
    from model.dmd_action_forcing import ActionForcingDMD
    setattr(stub, name, types.MethodType(
        getattr(ActionForcingDMD, name), stub,
    ))
    return stub


class TestStreamingRealWindow(unittest.TestCase):
    """``of_streaming_real`` — the streaming analogue of ``_of_aligned_real``."""

    def _stub(self, ride, pool=None):
        state = {"ride_latents_window": ride}
        if pool is not None:
            state["gt_match_latents"] = pool
        stub = _OFModelStub(streaming_state=state)
        return _bind_model_method(stub, "of_streaming_real")

    def test_slices_the_frame_aligned_window(self):
        ride = torch.randn(2, 40, 3, 4, 4)
        stub = self._stub(ride)
        got, pool = stub.of_streaming_real(9, 30, fake_frames=21)
        torch.testing.assert_close(got, ride[:, 9:30])
        # no explicit pool published -> the ride window IS the pool
        self.assertIs(pool, ride)

    def test_prefers_gt_match_pool_when_published(self):
        ride = torch.randn(1, 40, 3, 4, 4)
        pool = torch.randn(1, 120, 3, 4, 4)
        stub = self._stub(ride, pool=pool)
        _got, got_pool = stub.of_streaming_real(0, 21, fake_frames=21)
        self.assertIs(got_pool, pool)

    def test_out_of_range_window_raises(self):
        stub = self._stub(torch.randn(1, 24, 3, 4, 4))
        with self.assertRaises(RuntimeError):
            stub.of_streaming_real(10, 40, fake_frames=30)
        with self.assertRaises(RuntimeError):
            stub.of_streaming_real(-1, 20, fake_frames=21)

    def test_length_mismatch_with_the_fake_raises(self):
        """The alignment invariant, asserted at the point it is knowable.

        A short tail slice would otherwise surface as an opaque shape
        error deep inside ``pair_shared_noise``."""
        stub = self._stub(torch.randn(1, 40, 3, 4, 4))
        with self.assertRaises(RuntimeError):
            stub.of_streaming_real(0, 20, fake_frames=21)

    def test_no_open_sequence_raises(self):
        stub = _OFModelStub(streaming_state=None)
        _bind_model_method(stub, "of_streaming_real")
        with self.assertRaises(RuntimeError):
            stub.of_streaming_real(0, 21, fake_frames=21)


class TestStreamingCond(unittest.TestCase):
    def test_returns_the_noisy_half_slice_the_scorers_use(self):
        """Same VALUES as the scorers' own slice, detached.

        Identity of the dict object is deliberately NOT the contract any
        more: the cond slices are graph-carrying (they come from the
        per-iter action-embedding projection built inside
        ``generate_next_chunk``), so handing the live objects to the disc
        would put ``action_projection`` inside ``critic_loss.backward()``
        — the discriminator's gradient, wrong-signed for the generator
        that owns those parameters, applied by the generator's
        optimizer."""
        cond = {"prompt_embeds": torch.randn(1, 4, 8, requires_grad=True),
                "not_a_tensor": 7}
        uncond = {"prompt_embeds": torch.zeros(1, 4, 8)}
        stub = _OFModelStub(noisy_cond=(cond, uncond))
        _bind_model_method(stub, "of_streaming_cond")
        got = stub.of_streaming_cond({"new_frames": 3})
        self.assertEqual(set(got), set(cond))
        torch.testing.assert_close(got["prompt_embeds"], cond["prompt_embeds"])
        self.assertEqual(got["not_a_tensor"], 7)

    def test_the_returned_cond_is_detached(self):
        cond = {"prompt_embeds": torch.randn(1, 4, 8, requires_grad=True)}
        stub = _OFModelStub(noisy_cond=(cond, {}))
        _bind_model_method(stub, "of_streaming_cond")
        got = stub.of_streaming_cond({"new_frames": 3})
        self.assertFalse(got["prompt_embeds"].requires_grad)
        # ...and the caller's own tensor is untouched.
        self.assertTrue(cond["prompt_embeds"].requires_grad)

    def test_detach_false_returns_the_live_slice_unchanged(self):
        """The escape hatch still hands back the scorers' own objects."""
        cond = {"prompt_embeds": torch.randn(1, 4, 8, requires_grad=True)}
        stub = _OFModelStub(noisy_cond=(cond, {}))
        _bind_model_method(stub, "of_streaming_cond")
        got = stub.of_streaming_cond({"new_frames": 3}, detach=False)
        self.assertIs(got, cond)


class TestFakeSourceFlashIsReachable(unittest.TestCase):
    """``gan_of_fake_source='flash'`` used to be dead code."""

    def _stub(self, source):
        cfg = dict(OF_DEFAULTS)
        cfg["gan_of_fake_source"] = source
        stub = _OFModelStub(of_cfg=cfg)
        return _bind_model_method(stub, "_of_fake_sample")

    def test_pred_image_default_returns_the_rollout_slab(self):
        pred = torch.randn(1, 21, 3, 4, 4)
        flash = torch.randn(1, 21, 3, 4, 4)
        self.assertIs(self._stub("pred_image")._of_fake_sample(pred, flash),
                      pred)

    def test_flash_returns_the_flash_slab(self):
        pred = torch.randn(1, 21, 3, 4, 4)
        flash = torch.randn(1, 21, 3, 4, 4)
        self.assertIs(self._stub("flash")._of_fake_sample(pred, flash), flash)

    def test_flash_without_a_slab_fails_loud(self):
        with self.assertRaises(RuntimeError):
            self._stub("flash")._of_fake_sample(
                torch.randn(1, 21, 3, 4, 4), None,
            )

    def test_trainer_no_longer_refuses_flash_outright(self):
        src = _trainer_src()
        self.assertNotIn(
            "is not available on the non-streaming path", src,
            "the blanket construction-time refusal of 'flash' must be gone",
        )
        # ...but it must still refuse, at CONSTRUCTION, the two
        # configurations in which the slab genuinely does not exist.
        self.assertIn('if _of_fake_src == "flash":', src)
        self.assertIn('getattr(self.config, "streaming_mode", True)', src)
        self.assertIn('getattr(self.config, "flash_dmd_enabled", True)', src)
        self.assertIn("gan_of_fake_source='flash' requires ", src)
        self.assertIn("flash_dmd_enabled=false", src)


class _OFTrainerStub:
    """Stand-in for ``ActionForcingDMDTrainer`` carrying only the
    attributes the two streaming OF helpers read."""

    def __init__(self, model, step=0, telemetry_every=0):
        self.model = model
        self.step = step
        self.gan_of_cfg = dict(OF_DEFAULTS)
        self.gan_of_cfg["gan_of_telemetry_every"] = telemetry_every
        self.gan_of_enabled = True

    def _of_grad_telemetry(self, gan_term, dmd_term, probe_tensor):
        self.telemetry_call = (gan_term, dmd_term, probe_tensor)
        return {"of_grad_telemetry_ok": 1.0}


class _OFLossModelStub(_OFModelStub):
    """Records what the trainer handed the loss entry points."""

    def __init__(self, ride, **kw):
        super().__init__(
            streaming_state={"ride_latents_window": ride}, **kw,
        )
        self.g_calls = []
        self.d_calls = []

    def compute_of_g_loss(self, pred_image, real_latent, cond_for_scoring,
                          current_step, gt_pool=None, flash_slab=None):
        self.g_calls.append(dict(
            pred_image=pred_image, real_latent=real_latent,
            cond=cond_for_scoring, step=current_step, gt_pool=gt_pool,
            flash_slab=flash_slab,
        ))
        fake = self._of_fake_sample(pred_image, flash_slab)
        return (fake.float().sum() * 0.001, fake, {"of_g_loss": 1.0})

    def compute_of_d_loss(self, fake_latent, real_latent, cond_for_scoring,
                          current_step, telemetry=True):
        self.d_calls.append(dict(
            fake_latent=fake_latent, real_latent=real_latent,
            cond=cond_for_scoring, step=current_step, telemetry=telemetry,
        ))
        logs = {"of_d_weight": 0.03}
        if telemetry:
            logs.update({"of_d_loss": 0.5, "of_logit_gap": 0.1})
        return torch.tensor(0.5), logs

    def of_step(self):
        return 17

    def _surface_flash_gan_slab(self, info):
        """The real one re-publishes ``info['flash_dmd_gan_x0']`` from the
        rollout's raw slab; the tests hand the published slab in
        directly, so this is the no-op that keeps the call site honest."""
        return None


def _make_loss_model(ride, source="pred_image", real_source="aligned_gt"):
    cfg = dict(OF_DEFAULTS)
    cfg["gan_of_fake_source"] = source
    cfg["gan_of_real_source"] = real_source
    m = _OFLossModelStub(ride, of_cfg=cfg, noisy_cond=({"c": 1}, {}))
    for name in ("of_streaming_real", "of_streaming_cond",
                 "_of_fake_sample", "_of_real_sample", "_of_disc_cond",
                 "_of_assert_cond_detached",
                 "_of_band_indices", "_of_publish_streaming_band",
                 "of_streaming_band"):
        _bind_model_method(m, name)
    return m


def _publish_band(model, score_image, *, gt_target=None, grad_mask=None,
                  cond=None, dmd_fired=True, info=None, chunk_lo=0,
                  step=5, band=None):
    """Drive the REAL ``_of_publish_streaming_band``.

    ``band=(lo, hi)`` builds the DMD gradient mask over that sub-window —
    the 42f rolling geometry, where only the supervised chunks of the
    21-frame scoring window carry a graph. Default: the whole window.
    """
    n_f = int(score_image.shape[1])
    if gt_target is None:
        gt_target = torch.randn_like(score_image.detach())
    if grad_mask is None:
        grad_mask = torch.zeros_like(score_image, dtype=torch.bool)
        lo, hi = band if band is not None else (0, n_f)
        grad_mask[:, lo:hi] = True
    model._of_publish_streaming_band(
        score_image=score_image,
        score_gt_target=gt_target,
        score_cond=cond if cond is not None else {},
        score_grad_mask=grad_mask,
        chunk=score_image,
        info=info if info is not None else {},
        chunk_lo=int(chunk_lo),
        chunk_hi=int(chunk_lo) + n_f,
        dmd_fired=bool(dmd_fired),
        current_step=int(step),
    )
    return gt_target


def _bind_trainer_method(stub, name):
    from trainer.causal_action_forcing_train import ActionForcingDMDTrainer
    setattr(stub, name, types.MethodType(
        getattr(ActionForcingDMDTrainer, name), stub,
    ))
    return stub


class TestStreamingTrainerHelpers(unittest.TestCase):
    """The real trainer helpers + the real band publisher, executed
    against stub loss entry points.

    REWRITTEN 2026-08-24. These used to drive ``_of_streaming_g_term``
    with ``fake_chunk=train_chunk`` and assert the loss was handed that
    exact object. The GPU falsified the premise: ``train_chunk`` is the
    ROOT of the DMD scoring graph, not the scored tensor, and with
    ``boundary_vae_roundtrip=true`` it arrives with no graph at all on
    every overlapped roll. The band is now resolved in the model, beside
    the DMD loss; the trainer helpers only consume it.
    """

    def _setup(self, telemetry_every=1, **kw):
        ride = torch.randn(1, 40, 3, 4, 4)
        model = _make_loss_model(ride, **kw)
        tr = _OFTrainerStub(model, step=5, telemetry_every=telemetry_every)
        _bind_trainer_method(tr, "_of_streaming_g_term")
        _bind_trainer_method(tr, "_of_streaming_d_term")
        _bind_trainer_method(tr, "_of_telemetry_step")
        return ride, model, tr

    def test_g_term_attaches_to_the_dmd_scored_band(self):
        _ride, model, tr = self._setup()
        score_image = torch.randn(1, 21, 3, 4, 4, requires_grad=True)
        _publish_band(model, score_image, band=(9, 18))
        out = {}
        term = tr._of_streaming_g_term(
            dmd_term=(score_image * 2).sum(), out=out,
        )
        self.assertIsNotNone(term)
        # THE invariant: the tensor handed to the G loss is a slice of the
        # SAME object the DMD scorer was handed, over exactly the frames
        # the DMD gradient mask marks.
        fake = model.g_calls[0]["pred_image"]
        self.assertEqual(int(fake.shape[1]), 9)
        # ...and it differentiates back into that object, on those frames
        # and no others.
        g = torch.autograd.grad(term, score_image, retain_graph=True)[0]
        per_frame = g.abs().flatten(2).sum(-1)[0]
        self.assertTrue(bool((per_frame[9:18] != 0).all()))
        self.assertTrue(bool((per_frame[:9] == 0).all()))
        self.assertTrue(bool((per_frame[18:] == 0).all()))
        self.assertEqual(out["of_g_loss"], 1.0)
        self.assertEqual(out["of_g_fired"], 1.0)
        self.assertEqual(out["of_band_lo"], 9.0)
        self.assertEqual(out["of_band_frames"], 9.0)
        self.assertEqual(out["of_band_graph_on"], 1.0)

    def test_g_term_real_side_is_the_gt_counterpart_of_the_same_band(self):
        _ride, model, tr = self._setup()
        score_image = torch.randn(1, 21, 3, 4, 4, requires_grad=True)
        gt = _publish_band(model, score_image, band=(9, 18))
        tr._of_streaming_g_term(dmd_term=score_image.sum(), out={})
        # Frame-locked by construction: gt_target is the GT counterpart of
        # the SCORED window, so the band slice of one pairs with the band
        # slice of the other. No chunk_lo/chunk_hi re-derivation.
        torch.testing.assert_close(
            model.g_calls[0]["real_latent"], gt[:, 9:18],
        )
        self.assertFalse(model.g_calls[0]["real_latent"].requires_grad)

    def test_g_term_does_not_fire_when_the_dmd_term_did_not(self):
        """Co-occurrence rule: under ``dmd_supervise_roll_mode='random'``
        most rolls score nothing, and an adversarial push there would
        shape the student on rolls the DMD objective never saw."""
        _ride, model, tr = self._setup()
        score_image = torch.randn(1, 21, 3, 4, 4, requires_grad=True)
        _publish_band(model, score_image, band=(9, 18), dmd_fired=False)
        out = {}
        self.assertIsNone(
            tr._of_streaming_g_term(dmd_term=score_image.sum(), out=out)
        )
        self.assertEqual(out["of_g_fired"], 0.0)
        self.assertEqual(model.g_calls, [])

    def test_g_term_uses_the_flash_slab_when_configured(self):
        ride, model, tr = self._setup(source="flash")
        score_image = torch.randn(1, 21, 3, 4, 4, requires_grad=True)
        flash = torch.randn(1, 15, 3, 4, 4, requires_grad=True)
        _publish_band(
            model, score_image, band=(9, 18), chunk_lo=9,
            info={"flash_dmd_gan_x0": flash},
        )
        term = tr._of_streaming_g_term(dmd_term=score_image.sum(), out={})
        # Under 'flash' the fake is the slab, and the real is the ride
        # window sized off THE SLAB (not off the scoring window).
        self.assertIs(model.g_calls[0]["pred_image"], flash)
        torch.testing.assert_close(
            model.g_calls[0]["real_latent"], ride[:, 9:24],
        )
        g_flash, g_score = torch.autograd.grad(
            term, [flash, score_image], allow_unused=True,
        )
        self.assertIsNotNone(g_flash)
        self.assertIsNone(g_score)

    def test_g_term_publishes_grad_telemetry(self):
        _ride, model, tr = self._setup()
        score_image = torch.randn(1, 21, 3, 4, 4, requires_grad=True)
        _publish_band(model, score_image, band=(9, 18))
        out = {}
        tr._of_streaming_g_term(dmd_term=score_image.sum(), out=out)
        self.assertEqual(out["of_grad_telemetry_ok"], 1.0)
        # ...and it is probed on the band, not on the whole window.
        self.assertIs(tr.telemetry_call[2], model.g_calls[0]["pred_image"])

    def test_d_term_fake_is_detached(self):
        """The D step must never reach the generator."""
        _ride, model, tr = self._setup()
        score_image = torch.randn(1, 21, 3, 4, 4, requires_grad=True)
        _publish_band(model, score_image, band=(9, 18))
        out = {}
        tr._of_streaming_d_term(out=out)
        self.assertFalse(model.d_calls[0]["fake_latent"].requires_grad)
        self.assertEqual(out["of_d_loss"], 0.5)
        self.assertEqual(out["of_logit_gap"], 0.1)

    def test_d_and_g_train_on_the_same_fake(self):
        """The failure this arm exists to remove: a disc trained on one
        distribution while the generator is pushed on another."""
        _ride, model, tr = self._setup()
        score_image = torch.randn(1, 21, 3, 4, 4, requires_grad=True)
        gt = _publish_band(model, score_image, band=(9, 18))
        tr._of_streaming_g_term(dmd_term=score_image.sum(), out={})
        tr._of_streaming_d_term(out={})
        g_fake = model.g_calls[0]["pred_image"]
        d_fake = model.d_calls[0]["fake_latent"]
        torch.testing.assert_close(d_fake, g_fake.detach())
        torch.testing.assert_close(
            model.d_calls[0]["real_latent"], gt[:, 9:18],
        )
        # ...and the same conditioning window.
        self.assertEqual(model.d_calls[0]["cond"], model.g_calls[0]["cond"])

    def test_d_term_uses_the_flash_slab_too_when_configured(self):
        _ride, model, tr = self._setup(source="flash")
        score_image = torch.randn(1, 21, 3, 4, 4, requires_grad=True)
        flash = torch.randn(1, 15, 3, 4, 4, requires_grad=True)
        _publish_band(
            model, score_image, band=(9, 18), chunk_lo=9,
            info={"flash_dmd_gan_x0": flash},
        )
        tr._of_streaming_d_term(out={})
        self.assertFalse(model.d_calls[0]["fake_latent"].requires_grad)
        torch.testing.assert_close(
            model.d_calls[0]["fake_latent"], flash.detach(),
        )

    def test_d_term_reads_the_step_from_of_step_not_the_trainer(self):
        """``of_step()`` raises when the trainer never published it — the
        D side must go through it rather than silently using its own
        counter."""
        _ride, model, tr = self._setup()
        score_image = torch.randn(1, 21, 3, 4, 4, requires_grad=True)
        _publish_band(model, score_image, band=(9, 18))
        tr._of_streaming_d_term(out={})
        self.assertEqual(model.d_calls[0]["step"], 17)

    def test_an_unpublished_band_is_loud_on_both_sides(self):
        """Silence here would read as "the adversarial term did not run
        this step" — the exact null result the arm must not produce."""
        _ride, model, tr = self._setup()
        model._of_band = None
        with self.assertRaises(RuntimeError):
            tr._of_streaming_g_term(dmd_term=torch.zeros(()), out={})
        with self.assertRaises(RuntimeError):
            tr._of_streaming_d_term(out={})

    def test_nearest_match_real_source_is_live_on_the_streaming_path(self):
        ride = torch.randn(1, 40, 3, 4, 4)
        model = _make_loss_model(ride, real_source="nearest_match")
        # publish an explicit (wider) pool, as a matched LADD ride does
        pool = torch.randn(1, 60, 3, 4, 4)
        model.streaming_state["gt_match_latents"] = pool
        # plant the answer so the retrieval is checkable
        planted = pool[:, 7:28].clone().requires_grad_(True)
        tr = _OFTrainerStub(model, step=5, telemetry_every=1)
        _bind_trainer_method(tr, "_of_streaming_d_term")
        _bind_trainer_method(tr, "_of_telemetry_step")
        _publish_band(model, planted)
        out = {}
        tr._of_streaming_d_term(out=out)
        torch.testing.assert_close(
            model.d_calls[0]["real_latent"], planted.detach(),
        )
        self.assertEqual(out["of_match_offset_mean"], 7.0)

    def test_nearest_match_falls_back_to_the_ride_window_pool(self):
        ride = torch.randn(1, 40, 3, 4, 4)
        model = _make_loss_model(ride, real_source="nearest_match")
        planted = ride[:, 12:33].clone().requires_grad_(True)
        tr = _OFTrainerStub(model, step=5, telemetry_every=1)
        _bind_trainer_method(tr, "_of_streaming_d_term")
        _bind_trainer_method(tr, "_of_telemetry_step")
        _publish_band(model, planted)
        out = {}
        tr._of_streaming_d_term(out=out)
        self.assertEqual(out["of_match_offset_mean"], 12.0)


class TestBandResolution(unittest.TestCase):
    """``_of_band_indices`` / ``_of_publish_streaming_band`` — the
    arithmetic that decides which frames the two objectives share."""

    def _model(self, **kw):
        return _make_loss_model(torch.randn(1, 40, 3, 4, 4), **kw)

    def test_band_is_read_off_the_dmd_gradient_mask(self):
        m = self._model()
        img = torch.randn(1, 21, 3, 4, 4, requires_grad=True)
        _publish_band(m, img, band=(12, 15))
        band = m.of_streaming_band()
        self.assertEqual((band["band_lo"], band["band_hi"]), (12, 15))

    def test_an_empty_mask_raises(self):
        m = self._model()
        img = torch.randn(1, 21, 3, 4, 4, requires_grad=True)
        with self.assertRaises(RuntimeError) as cm:
            _publish_band(
                m, img,
                grad_mask=torch.zeros(1, 21, 3, 4, 4, dtype=torch.bool),
            )
        self.assertIn("empty", str(cm.exception))

    def test_a_non_contiguous_mask_raises(self):
        m = self._model()
        img = torch.randn(1, 21, 3, 4, 4, requires_grad=True)
        mask = torch.zeros(1, 21, 3, 4, 4, dtype=torch.bool)
        mask[:, 3:5] = True
        mask[:, 9:11] = True
        with self.assertRaises(RuntimeError) as cm:
            _publish_band(m, img, grad_mask=mask)
        self.assertIn("non-contiguous", str(cm.exception))

    def test_a_mislengthed_gt_target_raises(self):
        m = self._model()
        img = torch.randn(1, 21, 3, 4, 4, requires_grad=True)
        with self.assertRaises(RuntimeError) as cm:
            _publish_band(
                m, img, band=(9, 18),
                gt_target=torch.randn(1, 15, 3, 4, 4),
            )
        self.assertIn("same window", str(cm.exception))

    def test_a_missing_gt_target_raises(self):
        m = self._model()
        img = torch.randn(1, 21, 3, 4, 4, requires_grad=True)
        with self.assertRaises(RuntimeError) as cm:
            m._of_publish_streaming_band(
                score_image=img, score_gt_target=None, score_cond={},
                score_grad_mask=torch.ones_like(img, dtype=torch.bool),
                chunk=img, info={}, chunk_lo=0, chunk_hi=21,
                dmd_fired=True, current_step=5,
            )
        self.assertIn("gt_target", str(cm.exception))

    def test_the_cond_is_sliced_to_the_band(self):
        m = self._model()
        img = torch.randn(1, 21, 3, 4, 4, requires_grad=True)
        cond = {
            "_action_modulation": torch.arange(21).float().view(1, 21, 1),
            "_action_modulation_clean": torch.zeros(1, 21, 1),
            "prompt_embeds": torch.zeros(1, 4),
        }
        _publish_band(m, img, band=(9, 18), cond=cond)
        band = m.of_streaming_band()
        # sliced to the band...
        torch.testing.assert_close(
            band["cond"]["_action_modulation"],
            torch.arange(9, 18).float().view(1, 9, 1),
        )
        # ...the teacher-forcing clean streams stripped (the disc runs
        # without a clean half)...
        self.assertNotIn("_action_modulation_clean", band["cond"])
        # ...and non-per-frame keys passed through.
        self.assertIn("prompt_embeds", band["cond"])


class TestStreamingCallSites(unittest.TestCase):
    """Source-level: the call sites, and their ORDER vs the backwards.

    A live check would need a 1.3B forward and a distributed context, so
    the properties that a wrong wiring would break — which tensor, which
    loss, before which backward — are asserted against the source.
    """

    def setUp(self):
        from trainer.causal_action_forcing_train import ActionForcingDMDTrainer
        # Scoped to the streaming per-chunk trainer, so an ordering
        # assertion cannot accidentally match the NON-streaming twin's
        # identically-spelled backward calls elsewhere in the file.
        self.src = _code_only(_method_src(
            ActionForcingDMDTrainer, "_streaming_train_one_chunk",
        ))
        self.file_src = _trainer_src()

    def test_the_streaming_refusal_is_gone(self):
        self.assertNotIn(
            "gan_of_enabled=true with streaming_mode=true", self.file_src,
        )
        self.assertNotIn("wire the arm into", self.file_src)

    def test_g_term_is_no_longer_built_from_train_chunk(self):
        """The falsified claim, pinned so it cannot come back.

        ``train_chunk`` is the ROOT of the DMD scoring graph, not the
        scored tensor; and under ``boundary_vae_roundtrip=true`` it
        arrives graph-free on every overlapped roll. The trainer must
        therefore hand the G helper no fake at all — the band comes from
        the model."""
        self.assertIn("of_g_loss = self._of_streaming_g_term(", self.src)
        i = self.src.index("of_g_loss = self._of_streaming_g_term(")
        block = self.src[i:i + 500]
        self.assertNotIn("fake_chunk=", block)
        self.assertNotIn("flash_slab=", block)
        self.assertIn("dmd_term=gen_loss_dmd,", block)
        # ...and the D twin is fed from the same place, with no fake of
        # its own either.
        j = self.src.index("_of_d_loss = self._of_streaming_d_term(")
        self.assertNotIn("fake_chunk=", self.src[j:j + 300])

    def test_the_band_is_published_beside_the_dmd_loss(self):
        """Same scope, so the identity of the scored tensor is a
        reference and not a re-derivation."""
        from model.dmd_action_forcing import ActionForcingDMD
        src = _code_only(_method_src(
            ActionForcingDMD, "compute_generator_loss_streaming",
        ))
        i_dmd = src.index("dmd_loss, dmd_log = self.compute_distribution")
        i_band = src.index("self._of_publish_streaming_band(")
        self.assertLess(i_dmd, i_band)
        # the publisher is handed the very locals the scorer was
        block = src[i_band:i_band + 700]
        for kw in ("score_image=score_image,",
                   "score_gt_target=score_gt_target,",
                   "score_cond=score_cond,",
                   "score_grad_mask=score_grad_mask,"):
            self.assertIn(kw, block)

    def test_g_term_lands_before_the_single_gen_backward(self):
        i_add = self.src.index("generator_loss = generator_loss + of_g_loss")
        i_bwd = self.src.index(
            "generator_loss.backward(retain_graph=True)"
        )
        self.assertLess(i_add, i_bwd)
        # and the trainer still performs exactly ONE gen backward on the
        # streaming path.
        self.assertEqual(
            self.src.count("generator_loss.backward(retain_graph=True)"), 1,
        )

    def test_d_term_folds_into_the_streaming_critic_loss(self):
        i_call = self.src.index(
            "critic_loss, critic_log = self.model.compute_critic_loss_streaming"
        )
        i_fold = self.src.index("critic_loss = critic_loss + _of_d_loss")
        i_bwd = self.src.index("critic_loss.backward()")
        self.assertLess(i_call, i_fold)
        self.assertLess(i_fold, i_bwd)

    def test_d_term_also_rides_the_extra_fake_updates(self):
        i = self.src.index("_extra_loss = _extra_loss + _of_d_extra")
        i_bwd = self.src.index("_extra_loss.backward()")
        self.assertLess(i, i_bwd)

    def test_the_inner_d_term_is_rebuilt_INSIDE_the_loop(self):
        """One D tensor reused across the N inner backwards would
        double-backward a freed disc graph — and would train the disc on
        one draw while charging it for five updates. The call must sit
        inside ``for _efi in range(_extra_fake_n)``, not above it."""
        i_loop = self.src.index("for _efi in range(_extra_fake_n):")
        i_inner = self.src.index(
            "_of_d_extra = self._of_streaming_d_term(", i_loop,
        )
        i_bwd = self.src.index("_extra_loss.backward()", i_loop)
        self.assertLess(i_loop, i_inner)
        self.assertLess(i_inner, i_bwd)
        # Exactly two D call sites on this path: the main critic update
        # and the inner one. A third would be an unaccounted disc forward.
        self.assertEqual(
            self.src.count("self._of_streaming_d_term("), 2,
        )
        # The inner denoising loss is rebuilt in the loop too, so the two
        # halves of every inner critic update come from the same fresh
        # forward pass.
        self.assertLess(
            i_loop,
            self.src.index("self.model.compute_critic_loss_streaming(",
                           i_loop),
        )

    def test_current_step_is_published_on_the_streaming_path(self):
        fn = _method_src(
            __import__(
                "trainer.causal_action_forcing_train",
                fromlist=["ActionForcingDMDTrainer"],
            ).ActionForcingDMDTrainer,
            "_fwdbwd_streaming_step",
        )
        self.assertIn("self.model._of_current_step = int(self.step)", fn)

    def test_telemetry_keys_reachable_from_the_streaming_helpers(self):
        """Every key the spec names must have a producer the streaming
        path actually calls."""
        from model.dmd_action_forcing import ActionForcingDMD
        d_src = _method_src(ActionForcingDMD, "compute_of_d_loss")
        g_src = _method_src(ActionForcingDMD, "compute_of_g_loss")
        for key in ("of_d_loss", "of_d_real", "of_d_fake", "of_logit_gap"):
            self.assertIn(f'"{key}"', d_src)
        self.assertIn('"of_g_loss"', g_src)
        from trainer.causal_action_forcing_train import ActionForcingDMDTrainer
        tel = _method_src(ActionForcingDMDTrainer, "_of_grad_telemetry")
        for key in ("of_gan_grad_norm", "of_gan_dmd_grad_ratio",
                    "of_gan_dmd_grad_cos"):
            self.assertIn(f'"{key}"', tel)
        # ...and both loss producers' logs are merged into ``out`` by the
        # helpers, which are what the streaming call sites use.
        self.assertIn(
            "out.update(",
            _method_src(ActionForcingDMDTrainer, "_of_streaming_g_term"),
        )
        # The D helper merges under a caller-supplied NAMESPACE rather
        # than a bare ``out.update`` — the inner
        # ``streaming_fake_updates_per_gen`` calls would otherwise
        # overwrite the main D step's numbers in the same row. The merge
        # itself is asserted behaviourally in
        # ``TestStreamingTrainerHelpers``; here only the mechanism.
        d_src = _method_src(ActionForcingDMDTrainer, "_of_streaming_d_term")
        self.assertIn("log_prefix", d_src)
        self.assertIn("out[_pfx + _k] = _v", d_src)


class TestStreamingRankUniformity(unittest.TestCase):
    """The DDP-hang class this codebase has already paid for.

    A disc update that one rank runs and another skips desynchronises the
    fake_score reducer. Every gate on the OF streaming path must
    therefore be a pure function of (config, global step).
    """

    def setUp(self):
        from trainer.causal_action_forcing_train import ActionForcingDMDTrainer
        self.cls = ActionForcingDMDTrainer
        self.src = _trainer_src()

    def test_of_active_is_config_only(self):
        self.assertIn("of_active = bool(self.gan_of_enabled)", self.src)

    def test_helpers_contain_no_per_rank_predicate(self):
        forbidden = (
            "can_generate_more", "gradient_mask", "is_main_process",
            "get_rank(", "ride_len", "avg_mae", "_tp_gone",
        )
        for name in ("_of_streaming_g_term", "_of_streaming_d_term"):
            body = _method_src(self.cls, name)
            for tok in forbidden:
                self.assertNotIn(
                    tok, body,
                    f"{name} must not branch on the per-rank quantity {tok!r}",
                )

    def test_weight_gate_is_a_pure_function_of_the_global_step(self):
        """``of_weight_at_step`` is the ONLY thing that can return None
        from either loss entry point, and it reads (config, step)."""
        import inspect
        sig = inspect.signature(of_weight_at_step)
        self.assertEqual(
            list(sig.parameters),
            ["base_weight", "current_step", "disc_start_step", "warmup_steps"],
        )
        # identical inputs -> identical decision on every rank
        for step in (0, 1, 7, 999):
            a = of_weight_at_step(0.03, step, 10, 5)
            b = of_weight_at_step(0.03, step, 10, 5)
            self.assertEqual(a, b)

    def test_d_fold_is_outside_the_per_rank_short_circuit(self):
        """``compute_critic_loss_streaming`` early-returns on a per-rank
        ``gradient_mask.any()``. The OF disc forward must not sit behind
        that, or one rank runs it and another does not."""
        from model.dmd_action_forcing import ActionForcingDMD
        critic_src = _method_src(
            ActionForcingDMD, "compute_critic_loss_streaming",
        )
        self.assertIn("if not gradient_mask.any():", critic_src)
        self.assertNotIn("compute_of_d_loss", critic_src)
        self.assertNotIn("_of_streaming_d_term", critic_src)
        # the fold lives in the trainer, after the call returns
        chunk_src = _method_src(self.cls, "_streaming_train_one_chunk")
        self.assertIn("_of_streaming_d_term(", chunk_src)

    def test_alt_head_combination_is_refused_at_construction(self):
        """``fake_alt_head_enabled`` flips the fake_score reducer to
        ``find_unused_parameters=True``, which DDP does not support with
        the two grad-on fake_score forwards the OF critic step performs
        before its single backward. Multi-node only, so it must be a
        construction-time raise, not a smoke finding."""
        self.assertIn(
            "gan_of_enabled=true with fake_alt_head_enabled=true", self.src,
        )
        i_guard = self.src.index(
            "gan_of_enabled=true with fake_alt_head_enabled=true"
        )
        i_wrap = self.src.index("self.fake_score_ddp = DDP(")
        self.assertLess(i_guard, i_wrap, "the guard must precede the wrap")

    def test_gone_gate_still_skips_stage4_uniformly(self):
        """The OF terms live inside ``_streaming_train_one_chunk``; the
        only thing that can skip that call is the MAX-reduced toothpaste
        GONE flag, which is rank-uniform by construction."""
        step_src = _method_src(self.cls, "_streaming_step")
        i_reduce = step_src.index("_tp_gone_hit = bool(int(_gt.item()))")
        i_call = step_src.index("self._streaming_train_one_chunk(")
        self.assertLess(i_reduce, i_call)
        self.assertIn("if not _tp_gone_hit:", step_src)


class TestStreamingDefaultOff(unittest.TestCase):
    """Default-off byte-identity on the STREAMING path."""

    def setUp(self):
        from trainer.causal_action_forcing_train import ActionForcingDMDTrainer
        self.chunk_src = _method_src(
            ActionForcingDMDTrainer, "_streaming_train_one_chunk",
        )

    def test_every_of_site_is_behind_the_gate(self):
        for line in self.chunk_src.split("\n"):
            s = line.strip()
            if "_of_streaming_" in s and not s.startswith("#"):
                # the only bare occurrences are inside ``if of_active:``
                self.assertTrue(
                    s.startswith("of_g_loss = self._of_streaming_g_term(")
                    or s.startswith("_of_d_loss = self._of_streaming_d_term(")
                    or s.startswith("_of_d_extra = self._of_streaming_d_term("),
                    f"unguarded OF site: {s}",
                )

    def test_gate_precedes_each_site(self):
        lines = self.chunk_src.split("\n")
        for i, line in enumerate(lines):
            if "self._of_streaming_" in line and not line.strip().startswith("#"):
                window = "\n".join(lines[max(0, i - 12):i])
                self.assertIn(
                    "if of_active:", window,
                    "an OF call site is not preceded by its gate",
                )

    def test_no_of_tensor_work_outside_the_gate(self):
        """No RNG draw, no slice, no ``.detach()`` for OF may happen
        before the gate — default-off must cost nothing at all."""
        for tok in ("of_streaming_real(", "of_streaming_cond("):
            self.assertNotIn(
                tok, self.chunk_src,
                "the geometry accessors must be reached only through the "
                "gated helpers",
            )


# ======================================================================
# REGRESSION — the checkpoint-replay hazard that killed the first GPU
# smoke (2026-08-24, 8/8 ranks, at ``generator_loss.backward()``).
#
# MECHANISM, stated once. ``fake_score_gradient_checkpointing=true``
# wraps every disc DiT block in ``torch.utils.checkpoint(
# use_reentrant=False)``, which replays the block at backward time and
# refuses to proceed unless the replayed saved-tensor list matches the
# original one. For ``F.linear(x, W)`` autograd saves ``W`` only when
# ``x`` needs grad, and saves ``x`` only when ``W`` needs grad — so the
# list DEPENDS ON ``requires_grad``. ``_of_disc_frozen`` clears those
# flags for the disc forward and restores them on exit; a backward taken
# after the restore therefore replays against a one-entry-longer list per
# linear and dies with ``CheckpointError: Recomputed values ... have
# different metadata``, a ``[dim, dim]`` weight sitting where an
# activation is expected.
#
# The fix is the reference's structure: take the disc gradient w.r.t. the
# fake tensor INSIDE the freeze and carry it forward as a surrogate
# scalar. These tests pin the mechanism, the fix, and the equivalence.
# ======================================================================
class _CkptBlocks(nn.Module):
    """A ``gradient_checkpointing=True`` DiT block stack, in miniature.

    Bias-free linears on purpose: ``mm`` is the op whose saved-tensor set
    is ``requires_grad``-dependent, and it is the op the real
    ``CheckpointError`` named (``[1536, 1536]`` = a Wan-1.3B attention
    projection).
    """

    def __init__(self, dim=8, n_blocks=3):
        super().__init__()
        self.blocks = nn.ModuleList(
            [nn.Linear(dim, dim, bias=False) for _ in range(n_blocks)]
        )
        self.gradient_checkpointing = True

    def forward(self, x):
        for blk in self.blocks:
            if self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    lambda t, m=blk: m(t).tanh(), x, use_reentrant=False,
                )
            else:
                x = blk(x).tanh()
        return x


class _StubDiscWrapper(nn.Module):
    """``WanDiffusionWrapper`` reduced to what ``_of_disc_logits`` uses."""

    def __init__(self, dim=8, n_blocks=3):
        super().__init__()
        self.dim = dim
        self.model = _CkptBlocks(dim, n_blocks)
        self.head = nn.Linear(dim, 1, bias=False)

    def _unwrapped_model(self):
        return self.model

    def forward(self, noisy_image_or_video, conditional_dict, timestep,
                classify_mode=False):
        assert classify_mode, "the OF disc forward is always classify_mode"
        x = noisy_image_or_video.reshape(
            noisy_image_or_video.shape[0], -1, self.dim,
        )
        # The real disc is conditioned on the per-frame action streams, and
        # those slices can be GRAPH-CARRYING (action_projection). Modelled
        # here so the conditioning-gradient boundary is testable.
        cond_feat = (conditional_dict or {}).get("cond_feat")
        if cond_feat is not None:
            x = x + cond_feat
        feats = self.model(x)
        return None, None, self.head(feats).mean(dim=1)


class _OFGLossStub:
    """Carries exactly what ``compute_of_g_loss`` touches."""

    def __init__(self, of_cfg, disc):
        self.of_cfg = of_cfg
        self.fake_score = disc
        self.scheduler = _StubScheduler()


def _make_g_loss_stub(checkpointing=True, dim=8, n_blocks=3):
    cfg = dict(OF_DEFAULTS)
    cfg["gan_of_g_weight"] = 0.03
    disc = _StubDiscWrapper(dim=dim, n_blocks=n_blocks)
    disc.model.gradient_checkpointing = bool(checkpointing)
    stub = _OFGLossStub(cfg, disc)
    for name in ("compute_of_g_loss", "_of_fake_sample", "_of_real_sample",
                 "_of_sample_timestep", "_of_disc_cond", "_of_disc_frozen",
                 "_of_disc_logits", "_of_g_grad", "_of_g_surrogate"):
        _bind_model_method(stub, name)
    return stub


class TestDiscFreezeCheckpointHazard(unittest.TestCase):
    """The bare mechanism, isolated from One-Forcing entirely."""

    def _stack_and_input(self):
        torch.manual_seed(0)
        blocks = _CkptBlocks(dim=8, n_blocks=3)
        x = torch.randn(1, 4, 8, requires_grad=True)
        return blocks, x

    def test_backward_after_the_freeze_is_lifted_explodes(self):
        """CONTROL. Without this failing, the fix below proves nothing.

        This is the exact shape of the production crash: forward with the
        disc frozen, restore the flags, THEN back-propagate."""
        blocks, x = self._stack_and_input()
        params = list(blocks.parameters())
        saved = [p.requires_grad for p in params]
        for p in params:
            p.requires_grad_(False)
        out = blocks(x).sum()
        for p, flag in zip(params, saved):     # the ``finally`` restore
            p.requires_grad_(flag)
        with self.assertRaises(Exception) as ctx:
            out.backward()
        self.assertIn("ecomputed", str(ctx.exception))

    def test_backward_inside_the_freeze_is_fine(self):
        """THE FIX. Same forward, gradient taken before the restore."""
        blocks, x = self._stack_and_input()
        params = list(blocks.parameters())
        saved = [p.requires_grad for p in params]
        for p in params:
            p.requires_grad_(False)
        try:
            out = blocks(x).sum()
            grad = torch.autograd.grad(out, x)[0]
        finally:
            for p, flag in zip(params, saved):
                p.requires_grad_(flag)
        self.assertEqual(grad.shape, x.shape)
        self.assertTrue(torch.isfinite(grad).all())

    def test_the_hazard_needs_checkpointing(self):
        """Scoping evidence: with checkpointing off the same sequence is
        harmless, which is why this never showed up before the first
        ``fake_score_gradient_checkpointing=true`` GPU run."""
        blocks, x = self._stack_and_input()
        blocks.gradient_checkpointing = False
        params = list(blocks.parameters())
        for p in params:
            p.requires_grad_(False)
        out = blocks(x).sum()
        for p in params:
            p.requires_grad_(True)
        out.backward()                          # no raise
        self.assertIsNotNone(x.grad)


class TestOFGLossSurvivesCheckpointedDisc(unittest.TestCase):
    """``compute_of_g_loss`` end to end against a checkpointed disc."""

    def _run(self, stub, seed=1234):
        torch.manual_seed(seed)
        fake = torch.randn(1, 2, 2, 2, 2, requires_grad=True)
        real = torch.randn(1, 2, 2, 2, 2)
        term, probe, logs = stub.compute_of_g_loss(
            pred_image=fake,
            real_latent=real,
            cond_for_scoring={"prompt_embeds": torch.zeros(1, 1, 4)},
            current_step=10,
        )
        return fake, term, probe, logs

    def test_gradient_flows_and_nothing_raises(self):
        """The production crash, reproduced through the real entry point.

        Before the restructure this raised ``CheckpointError`` at the
        ``torch.autograd.grad`` below — the trainer's
        ``generator_loss.backward()`` in miniature."""
        stub = _make_g_loss_stub(checkpointing=True)
        fake, term, probe, _logs = self._run(stub)
        self.assertIs(probe, fake)
        self.assertTrue(term.requires_grad)
        grad = torch.autograd.grad(term, fake)[0]
        self.assertEqual(grad.shape, fake.shape)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0.0,
                           "the adversarial gradient must be non-zero — a "
                           "silently-dropped G term is the failure mode "
                           "this whole arm cannot afford")

    def test_checkpointed_and_plain_disc_agree_exactly(self):
        """EQUIVALENCE. Gradient checkpointing is a memory technique with
        no numerics of its own, so the surrogate must reproduce the plain
        forward's value AND gradient bit-for-bit. This is what rules out
        "the fix works by quietly changing the objective"."""
        on = _make_g_loss_stub(checkpointing=True)
        off = _make_g_loss_stub(checkpointing=False)
        off.fake_score.load_state_dict(on.fake_score.state_dict())

        f_on, t_on, _p, l_on = self._run(on)
        g_on = torch.autograd.grad(t_on, f_on)[0]
        f_off, t_off, _p, l_off = self._run(off)
        g_off = torch.autograd.grad(t_off, f_off)[0]

        torch.testing.assert_close(t_on, t_off)
        torch.testing.assert_close(g_on, g_off)
        self.assertEqual(l_on["of_g_loss"], l_off["of_g_loss"])

    def test_the_term_still_reports_the_adversarial_loss_value(self):
        """The surrogate is value-preserving, so ``generator_loss`` and
        ``of_g_loss`` keep comparing across arms and across the fix."""
        stub = _make_g_loss_stub(checkpointing=True)
        _fake, term, _probe, logs = self._run(stub)
        self.assertAlmostEqual(
            float(term.detach()), logs["of_g_loss"], places=5,
        )

    def test_the_disc_receives_no_gradient_and_keeps_its_flags(self):
        """The two invariants the freeze exists for, unchanged by the
        restructure: the G step must not train D, and it must not leave D
        frozen for the critic step that follows."""
        stub = _make_g_loss_stub(checkpointing=True)
        before = [p.requires_grad for p in stub.fake_score.parameters()]
        _fake, term, _probe, _logs = self._run(stub)
        term.backward()
        after = [p.requires_grad for p in stub.fake_score.parameters()]
        self.assertEqual(before, after)
        self.assertTrue(all(p.requires_grad for p in
                            stub.fake_score.parameters()))
        for name, p in stub.fake_score.named_parameters():
            self.assertIsNone(p.grad, f"D was trained by the G step: {name}")

    def test_a_detached_fake_fails_loud(self):
        """Not silently: a G term with no path to the generator used to be
        indistinguishable from a healthy one in the logs."""
        stub = _make_g_loss_stub(checkpointing=True)
        torch.manual_seed(7)
        with self.assertRaises(RuntimeError) as ctx:
            stub.compute_of_g_loss(
                pred_image=torch.randn(1, 2, 2, 2, 2),   # no grad
                real_latent=torch.randn(1, 2, 2, 2, 2),
                cond_for_scoring={},
                current_step=10,
            )
        self.assertIn("adversarial gradient", str(ctx.exception))

    def test_conditioning_gets_no_adversarial_gradient(self):
        """The one deliberate semantic difference from the attached form.

        The disc's cond slices can carry a live ``action_projection``
        subgraph, so an attached disc graph would have let the adversarial
        loss move the ACTION ENCODER through the disc's conditioning input
        — a channel for shifting the logit by reshaping the conditioning
        rather than the video. Bounding the gradient at the fake tensor
        closes it, which is both the reference's behaviour and what
        ``compute_of_g_loss``'s "no gradient for any DDP-managed
        parameter" invariant already claimed."""
        stub = _make_g_loss_stub(checkpointing=True)
        torch.manual_seed(99)
        fake = torch.randn(1, 2, 2, 2, 2, requires_grad=True)
        cond_feat = torch.randn(1, 2, 8, requires_grad=True)
        term, _probe, _logs = stub.compute_of_g_loss(
            pred_image=fake,
            real_latent=torch.randn(1, 2, 2, 2, 2),
            cond_for_scoring={"cond_feat": cond_feat},
            current_step=10,
        )
        g_fake, g_cond = torch.autograd.grad(
            term, [fake, cond_feat], allow_unused=True,
        )
        self.assertIsNotNone(g_fake)
        self.assertGreater(float(g_fake.abs().sum()), 0.0)
        self.assertIsNone(
            g_cond,
            "the adversarial term must reach the generator ONLY through "
            "the tensor DMD scores",
        )

    def test_zero_weight_short_circuit_is_untouched(self):
        cfg_stub = _make_g_loss_stub(checkpointing=True)
        cfg_stub.of_cfg["gan_of_g_weight"] = 0.0
        term, probe, logs = cfg_stub.compute_of_g_loss(
            pred_image=torch.randn(1, 2, 2, 2, 2, requires_grad=True),
            real_latent=torch.randn(1, 2, 2, 2, 2),
            cond_for_scoring={},
            current_step=10,
        )
        self.assertIsNone(term)
        self.assertIsNone(probe)
        self.assertEqual(logs, {"of_g_weight": 0.0})


class TestFreezeWindowStructure(unittest.TestCase):
    """Source-level: the gradient extraction must stay INSIDE the freeze.

    The runtime tests above catch a regression only while the stub disc
    keeps its checkpointing on. This one catches the edit that moves the
    ``autograd.grad`` back out of the ``with`` block, which is the single
    change that reintroduces the crash."""

    def setUp(self):
        from model.dmd_action_forcing import ActionForcingDMD
        self.g_src = _method_src(ActionForcingDMD, "compute_of_g_loss")
        self.d_src = _method_src(ActionForcingDMD, "compute_of_d_loss")

    def test_the_grad_is_taken_inside_the_frozen_block(self):
        lines = _code_only(self.g_src).split("\n")
        w_idx = w_indent = None
        for i, ln in enumerate(lines):
            if "with self._of_disc_frozen():" in ln:
                w_idx, w_indent = i, len(ln) - len(ln.lstrip())
                break
        self.assertIsNotNone(w_idx, "the freeze block is gone")
        g_idx = next(
            (i for i, ln in enumerate(lines) if "self._of_g_grad(" in ln),
            None,
        )
        self.assertIsNotNone(g_idx, "the disc gradient is no longer taken")
        self.assertGreater(g_idx, w_idx)
        # Inside the block == every intervening statement stays indented
        # past the ``with``. A dedent would mean the freeze already lifted.
        for ln in lines[w_idx + 1:g_idx + 1]:
            if ln.strip():
                self.assertGreater(
                    len(ln) - len(ln.lstrip()), w_indent,
                    f"this line escapes the freeze before the grad: {ln!r}",
                )

    def test_the_returned_term_is_the_surrogate_not_the_disc_graph(self):
        self.assertIn("self._of_g_surrogate(", _code_only(self.g_src))
        self.assertIn("return term, fake_latent, logs",
                      _code_only(self.g_src))

    def test_the_d_side_does_not_freeze(self):
        """D wants its own gradient, so it never freezes — which is
        exactly why the D-side disc forward may be replayed by the
        caller's ``critic_loss.backward()`` without this hazard.

        Matches the CALL, not the name: the method's docstring names
        ``_of_disc_frozen`` precisely to forbid it, and a bare-name search
        would fail on the prose that documents the invariant."""
        self.assertNotIn(
            "with self._of_disc_frozen()", _code_only(self.d_src),
        )

    def test_the_d_call_sites_do_not_freeze_around_the_backward(self):
        from trainer.causal_action_forcing_train import ActionForcingDMDTrainer
        src = _code_only(_method_src(
            ActionForcingDMDTrainer, "_streaming_train_one_chunk",
        ))
        self.assertNotIn("_of_disc_frozen", src)
        self.assertNotIn("requires_grad_(", src)


# ======================================================================
# #2 / S3 — the disc head must be gradiented on EVERY critic backward.
#
# The head lives inside the DDP-wrapped ``fake_score.model``, wrapped
# with ``find_unused_parameters=False``, and is reached by exactly ONE
# forward: the ``classify_mode`` disc forward in ``compute_of_d_loss``.
# Whenever the D loss is inactive — ``gan_of_disc_start_step > 0``
# (warmup) or ``gan_of_d_weight = 0`` (the G-only ablation), BOTH
# advertised in the flag table — that forward does not run, the head
# receives no gradient, and DDP raises "Expected to have finished
# reduction in the prior iteration before starting a new one" on the
# NEXT step. Multi-node only in production; reproduced below at
# world_size=1 on gloo, which is enough for the reducer to fire.
# ======================================================================
class _DiscHeadHost(nn.Module):
    """The shape that matters: a backbone the denoising loss reaches and
    a disc head only the classify forward reaches."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self._cls_pred_branch = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 1))
        self._register_tokens = nn.Linear(4, 4)
        self._gan_ca_blocks = nn.ModuleList([nn.Linear(4, 4)])

    def head_parameters(self):
        for m in (self._cls_pred_branch, self._register_tokens,
                  self._gan_ca_blocks):
            yield from m.parameters()

    def forward(self, x):
        return self.backbone(x)


class _FakeScoreWrapper:
    """``WanDiffusionWrapper``'s one relevant method."""

    def __init__(self, inner):
        self._inner = inner

    def _unwrapped_model(self):
        return self._inner


class _HeadTouchStub:
    def __init__(self, inner, of_cfg=None):
        self.fake_score = _FakeScoreWrapper(inner)
        self.of_cfg = of_cfg or dict(OF_DEFAULTS)


def _make_head_touch_stub(inner, **cfg_over):
    cfg = dict(OF_DEFAULTS)
    cfg.update(cfg_over)
    stub = _HeadTouchStub(inner, of_cfg=cfg)
    _bind_model_method(stub, "of_head_touch")
    _bind_model_method(stub, "compute_of_d_loss")
    return stub


class TestDiscHeadAlwaysGradiented(unittest.TestCase):
    def test_touch_gradients_every_head_parameter_with_exactly_zero(self):
        inner = _DiscHeadHost()
        stub = _make_head_touch_stub(inner)
        term = stub.of_head_touch()
        self.assertIsNotNone(term)
        self.assertEqual(float(term.detach()), 0.0)
        term.backward()
        for name, p in inner.named_parameters():
            if name.startswith("backbone"):
                self.assertIsNone(p.grad, f"{name} must not be touched")
                continue
            self.assertIsNotNone(p.grad, f"{name} received NO gradient")
            self.assertTrue(
                torch.equal(p.grad, torch.zeros_like(p.grad)),
                f"{name} must be moved by EXACTLY zero",
            )

    def test_touch_is_none_when_no_head_is_attached(self):
        """Default-off: no head, no term, nothing added to any loss."""
        class _Bare(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Linear(4, 4)
        stub = _make_head_touch_stub(_Bare())
        self.assertIsNone(stub.of_head_touch())

    def test_touch_skips_frozen_head_params(self):
        inner = _DiscHeadHost()
        for p in inner._register_tokens.parameters():
            p.requires_grad_(False)
        stub = _make_head_touch_stub(inner)
        stub.of_head_touch().backward()
        for p in inner._register_tokens.parameters():
            self.assertIsNone(p.grad)
        for p in inner._cls_pred_branch.parameters():
            self.assertIsNotNone(p.grad)

    def _inactive_d_call(self, **cfg_over):
        inner = _DiscHeadHost()
        stub = _make_head_touch_stub(inner, **cfg_over)
        term, logs = stub.compute_of_d_loss(
            fake_latent=torch.randn(1, 3, 2, 2, 2),
            real_latent=torch.randn(1, 3, 2, 2, 2),
            cond_for_scoring={},
            current_step=0,
        )
        return inner, term, logs

    def test_d_weight_zero_still_returns_a_term_that_holds_the_head(self):
        """``gan_of_d_weight=0`` is the advertised G-only ablation."""
        inner, term, logs = self._inactive_d_call(gan_of_d_weight=0.0)
        self.assertIsNotNone(
            term, "the G-only ablation must still tie the head in",
        )
        self.assertEqual(logs["of_d_weight"], 0.0)
        term.backward()
        for name, p in inner.named_parameters():
            if not name.startswith("backbone"):
                self.assertIsNotNone(p.grad, f"{name} ungradiented")

    def test_disc_start_step_warmup_still_returns_a_term(self):
        """``gan_of_disc_start_step>0`` is the advertised warmup."""
        inner, term, logs = self._inactive_d_call(gan_of_disc_start_step=100)
        self.assertIsNotNone(term)
        self.assertEqual(logs["of_d_weight"], 0.0)
        term.backward()
        for name, p in inner.named_parameters():
            if not name.startswith("backbone"):
                self.assertIsNotNone(p.grad, f"{name} ungradiented")


class TestDiscHeadUnderDDP(unittest.TestCase):
    """The actual failure, reproduced and then shown fixed.

    ``find_unused_parameters=False`` is what the trainer wraps
    ``fake_score.model`` with (``fake_alt_head_enabled``, the only thing
    that would flip it, is refused for this arm). world_size=1 on gloo
    is enough: the reducer runs regardless of peer count, and the raise
    is the same one 8-rank production hits."""

    def setUp(self):
        import tempfile
        import torch.distributed as dist
        if not dist.is_available() or not dist.is_gloo_available():
            self.skipTest("gloo unavailable")
        self._tmp = tempfile.mkdtemp()
        if dist.is_initialized():
            self.skipTest("a process group is already initialised")
        dist.init_process_group(
            backend="gloo",
            store=dist.FileStore(os.path.join(self._tmp, "store"), 1),
            rank=0, world_size=1,
        )

    def tearDown(self):
        import shutil
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _run(self, with_touch, iters=3):
        from torch.nn.parallel import DistributedDataParallel as DDP
        inner = _DiscHeadHost()
        ddp = DDP(inner, find_unused_parameters=False)
        stub = _make_head_touch_stub(inner)
        x = torch.randn(2, 4)
        for _ in range(iters):
            # The denoising critic loss: reaches the backbone, never the
            # disc head (``classify_mode`` is a different forward).
            loss = ddp(x).sum()
            if with_touch:
                loss = loss + stub.of_head_touch()
            loss.backward()
            for p in inner.parameters():
                p.grad = None

    def test_control_an_ungradiented_head_breaks_the_reducer(self):
        """MUTATION CONTROL. Without the touch this MUST fail — a fix
        whose absence changes nothing is not evidence of anything."""
        with self.assertRaises(RuntimeError) as cm:
            self._run(with_touch=False)
        self.assertIn("finished reduction", str(cm.exception))

    def test_the_touch_keeps_the_reducer_happy(self):
        self._run(with_touch=True)   # must not raise


# ======================================================================
# #10 — the inner ``streaming_fake_updates_per_gen`` D calls must not
# overwrite the MAIN D step's numbers, and must not pay 7 GPU syncs
# apiece for logs nobody reads every step.
# ======================================================================
class TestDTermLogNamespacing(unittest.TestCase):
    def _setup(self, telemetry_every=1):
        ride = torch.randn(1, 40, 3, 4, 4)
        model = _make_loss_model(ride)
        tr = _OFTrainerStub(model, step=5, telemetry_every=telemetry_every)
        _bind_trainer_method(tr, "_of_streaming_d_term")
        _bind_trainer_method(tr, "_of_telemetry_step")
        _publish_band(
            model,
            torch.randn(1, 21, 3, 4, 4, requires_grad=True),
            band=(9, 18),
        )
        return model, tr

    def test_inner_updates_do_not_clobber_the_main_d_step(self):
        model, tr = self._setup()
        out = {}
        tr._of_streaming_d_term(out=out)
        self.assertEqual(out["of_logit_gap"], 0.1)
        # Now the inner-loop calls, which used to land on the SAME keys.
        model.d_calls.clear()
        for _ in range(4):
            tr._of_streaming_d_term(out=out, log_prefix="of_inner_")
        self.assertEqual(out["of_logit_gap"], 0.1, "main key was clobbered")
        self.assertEqual(out["of_inner_of_logit_gap"], 0.1)
        self.assertIn("of_inner_of_d_loss", out)
        self.assertIn("of_d_loss", out)

    def test_the_inner_call_site_passes_the_prefix(self):
        src = _code_only(_trainer_src())
        i = src.index("_of_d_extra = self._of_streaming_d_term(")
        self.assertIn('log_prefix="of_inner_"', src[i:i + 600])
        # ...and the MAIN call site does not, so it keeps the bare keys.
        j = src.index("_of_d_loss = self._of_streaming_d_term(")
        self.assertNotIn("log_prefix", src[j:j + 400])

    def test_telemetry_syncs_are_gated_by_the_cadence(self):
        # step=5, every=25 -> off-cadence
        model, tr = self._setup(telemetry_every=25)
        out = {}
        tr._of_streaming_d_term(out=out)
        self.assertFalse(model.d_calls[0]["telemetry"])
        self.assertNotIn("of_logit_gap", out)
        # ...but the weight, already a Python float, is always there.
        self.assertIn("of_d_weight", out)

    def test_on_cadence_the_full_log_set_appears(self):
        model, tr = self._setup(telemetry_every=5)   # step=5 -> on cadence
        out = {}
        tr._of_streaming_d_term(out=out)
        self.assertTrue(model.d_calls[0]["telemetry"])
        self.assertIn("of_logit_gap", out)

    def test_telemetry_every_zero_means_off_not_every_step(self):
        """One knob, one meaning — the grad telemetry already reads
        ``<= 0`` as OFF."""
        model, tr = self._setup(telemetry_every=0)
        tr._of_streaming_d_term(out={})
        self.assertFalse(model.d_calls[0]["telemetry"])

    def test_the_real_d_loss_gates_its_item_calls(self):
        """Source-level twin of the above, on the producer itself."""
        from model.dmd_action_forcing import ActionForcingDMD
        src = _code_only(_method_src(ActionForcingDMD, "compute_of_d_loss"))
        self.assertIn("if telemetry:", src)
        i = src.index('logs: Dict[str, float] = {"of_d_weight"')
        j = src.index('"of_logit_gap"')
        self.assertLess(i, j, "of_d_weight must be outside the gate")


# ======================================================================
# #11 / S10 — the G path must resolve the fake BEFORE it measures the
# real window, or ``of_streaming_real``'s frame-alignment assert is
# disarmed on exactly the path (``fake_source='flash'``) that needs it.
# ======================================================================
class TestGTermSizesTheRealOffTheResolvedFake(unittest.TestCase):
    """#11 / S10, re-expressed on the band publisher.

    The G path must size the real window off the RESOLVED fake, never off
    a tensor that merely happens to be nearby. Pre-fix the trainer sized
    it off ``fake_chunk`` (21 frames, matching [9:30]) while the loss
    actually scored the 15-frame flash slab, which disarmed the alignment
    assert on exactly the path that needs it. The model now resolves the
    source and the window in one place, so the two cannot disagree.
    """

    def _setup(self, source):
        ride = torch.randn(1, 40, 3, 4, 4)
        model = _make_loss_model(ride, source=source)
        tr = _OFTrainerStub(model, step=5, telemetry_every=0)
        _bind_trainer_method(tr, "_of_streaming_g_term")
        _bind_trainer_method(tr, "_of_streaming_d_term")
        _bind_trainer_method(tr, "_of_telemetry_step")
        return ride, model, tr

    def test_the_real_window_is_sized_off_the_flash_slab(self):
        ride, model, tr = self._setup("flash")
        _publish_band(
            model, torch.randn(1, 21, 3, 4, 4, requires_grad=True),
            band=(9, 18), chunk_lo=9,
            info={"flash_dmd_gan_x0": torch.randn(
                1, 15, 3, 4, 4, requires_grad=True)},
        )
        tr._of_streaming_g_term(dmd_term=torch.zeros(()), out={})
        torch.testing.assert_close(
            model.g_calls[0]["real_latent"], ride[:, 9:24],
        )

    def test_a_flash_window_running_off_the_ride_is_caught(self):
        """The alignment assert is still reachable: a slab whose window
        overruns the ride raises instead of broadcasting."""
        _ride, model, _tr = self._setup("flash")
        with self.assertRaises(RuntimeError) as cm:
            _publish_band(
                model, torch.randn(1, 21, 3, 4, 4, requires_grad=True),
                band=(9, 18), chunk_lo=35,
                info={"flash_dmd_gan_x0": torch.randn(
                    1, 15, 3, 4, 4, requires_grad=True)},
            )
        self.assertIn("outside ride_latents_window", str(cm.exception))

    def test_a_flash_config_with_no_slab_raises(self):
        _ride, model, _tr = self._setup("flash")
        with self.assertRaises(RuntimeError) as cm:
            _publish_band(
                model, torch.randn(1, 21, 3, 4, 4, requires_grad=True),
                band=(9, 18), chunk_lo=9, info={},
            )
        self.assertIn("flash_dmd_gan_x0", str(cm.exception))

    def test_the_d_twin_reads_the_same_resolved_pair(self):
        ride, model, tr = self._setup("flash")
        flash = torch.randn(1, 15, 3, 4, 4, requires_grad=True)
        _publish_band(
            model, torch.randn(1, 21, 3, 4, 4, requires_grad=True),
            band=(9, 18), chunk_lo=9, info={"flash_dmd_gan_x0": flash},
        )
        tr._of_streaming_d_term(out={})
        torch.testing.assert_close(
            model.d_calls[0]["fake_latent"], flash.detach(),
        )
        torch.testing.assert_close(
            model.d_calls[0]["real_latent"], ride[:, 9:24],
        )

    def test_pred_image_default_is_unchanged(self):
        _ride, model, tr = self._setup("pred_image")
        img = torch.randn(1, 21, 3, 4, 4, requires_grad=True)
        gt = _publish_band(
            model, img, band=(9, 18), chunk_lo=9,
            info={"flash_dmd_gan_x0": torch.randn(1, 15, 3, 4, 4)},
        )
        tr._of_streaming_g_term(dmd_term=torch.zeros(()), out={})
        # the slab is present and IGNORED: the fake is the scored band.
        torch.testing.assert_close(
            model.g_calls[0]["real_latent"], gt[:, 9:18],
        )


class TestNonStreamingIsRefused(unittest.TestCase):
    def test_construction_refuses_streaming_mode_false(self):
        src = _code_only(_trainer_src())
        self.assertIn(
            "gan_of_enabled=true requires streaming_mode=true", src,
            "the arm must refuse the path on which its D term has no "
            "call site",
        )

    def test_the_refusal_precedes_the_head_construction(self):
        """It must raise before ~50M parameters are allocated."""
        src = _code_only(_trainer_src())
        i_raise = src.index("gan_of_enabled=true requires streaming_mode=true")
        i_head = src.index("model.fake_score.adding_cls_branch(")
        self.assertLess(i_raise, i_head)

    def test_the_d_term_still_has_exactly_one_streaming_home(self):
        src = _code_only(_trainer_src())
        self.assertEqual(src.count("def _of_streaming_d_term("), 1)
        # main critic step + the streaming_fake_updates_per_gen inner loop
        self.assertEqual(src.count("self._of_streaming_d_term("), 2)


# ======================================================================
# #3 / S7 — resume verification must not re-read the checkpoint, and
# must not kill a run it merely failed to verify.
# ======================================================================
class _ResumeStub:
    def __init__(self, log_dir, keys, **cfg):
        import pathlib
        self.gan_of_enabled = True
        self.log_dir = pathlib.Path(log_dir)
        self.is_main_process = True
        self._of_head_param_names = {"_cls_pred_branch.1.weight",
                                     "_register_tokens.weight"}
        base = dict(auto_resume=True, resume_load_fake_score=True,
                    strict_resume_load=False)
        base.update(cfg)
        self.config = _Cfg(**base)
        if keys is not None:
            self._resume_fake_score_keys = keys


class TestDiscResumeVerification(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.mkdtemp()
        # DELIBERATELY NOT A VALID CHECKPOINT. If anything in the verify
        # path still calls ``torch.load`` this file makes it say so.
        with open(os.path.join(self._tmp, "phase1_step000100.pt"), "w") as fh:
            fh.write("not a checkpoint")

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _run(self, keys, **cfg):
        stub = _ResumeStub(self._tmp, keys, **cfg)
        _bind_trainer_method(stub, "_of_verify_disc_resume")
        return stub._of_verify_disc_resume()

    def test_no_second_torch_load(self):
        from trainer.causal_action_forcing_train import ActionForcingDMDTrainer
        src = _method_src(
            ActionForcingDMDTrainer, "_of_verify_disc_resume",
        )
        # Strip the docstring as well as the comments: the docstring
        # NAMES ``torch.load`` precisely to record that it is gone, and a
        # raw search would find the prose and pass forever.
        body = _code_only(src[src.index('"""', src.index('"""') + 3) + 3:])
        self.assertNotIn("torch.load", body)
        self.assertIn("_resume_fake_score_keys", body)

    def test_the_base_resume_publishes_the_key_set(self):
        from trainer.causal_rolling_staircase_train import (
            RollingStaircaseDMDTrainer,
        )
        src = _method_src(RollingStaircaseDMDTrainer, "_maybe_resume")
        self.assertIn("self._resume_fake_score_keys = set(fake_sd or {})", src)

    def test_a_full_head_verifies_without_reading_the_file(self):
        # No raise, and the corrupt file above is never opened.
        self._run({"_cls_pred_branch.1.weight", "_register_tokens.weight",
                   "blocks.0.self_attn.q.weight"})

    def test_a_missing_head_is_loud_but_not_fatal(self):
        with self.assertLogs(level="ERROR") as cm:
            self._run({"blocks.0.self_attn.q.weight"})
        self.assertIn("FRESH", "\n".join(cm.output))

    def test_a_missing_head_raises_only_under_strict_resume_load(self):
        with self.assertRaises(RuntimeError):
            self._run({"blocks.0.self_attn.q.weight"},
                      strict_resume_load=True)

    def test_a_partial_head_is_loud_but_not_fatal(self):
        with self.assertLogs(level="ERROR"):
            self._run({"_cls_pred_branch.1.weight"})

    def test_an_unverifiable_resume_degrades_instead_of_killing_the_job(self):
        """``keys is None`` = the base resume never reached the fake_score
        restore. Old code re-loaded the checkpoint and turned any failure
        into a RuntimeError that took the job down."""
        with self.assertLogs(level="ERROR") as cm:
            self._run(None)
        self.assertIn("UNKNOWN", "\n".join(cm.output))

    def test_resume_load_fake_score_false_is_reported_not_raised(self):
        with self.assertLogs(level="ERROR") as cm:
            self._run(None, resume_load_fake_score=False)
        self.assertIn("resume_load_fake_score=false", "\n".join(cm.output))

    def test_the_arm_being_off_is_a_no_op(self):
        stub = _ResumeStub(self._tmp, None)
        stub.gan_of_enabled = False
        _bind_trainer_method(stub, "_of_verify_disc_resume")
        stub._of_verify_disc_resume()

    def test_the_swallowed_fake_optimizer_reset_is_now_logged_loudly(self):
        from trainer.causal_rolling_staircase_train import (
            RollingStaircaseDMDTrainer,
        )
        src = _method_src(RollingStaircaseDMDTrainer, "_maybe_resume")
        i = src.index("self.fake_optimizer.load_state_dict(fake_opt_sd)")
        block = src[i:i + 2000]
        self.assertIn("logging.error(", block)
        self.assertNotIn("logging.warning(\n", block[:block.index("elif")])


# ======================================================================
# #8 — the unweighted grad share must divide by the LIVE weight.
# ======================================================================
class _TelemetryTrainerStub:
    def __init__(self, step, **cfg_over):
        self.step = step
        self.gan_of_cfg = dict(OF_DEFAULTS)
        self.gan_of_cfg["gan_of_telemetry_every"] = 1
        self.gan_of_cfg.update(cfg_over)
        self.gan_of_enabled = True


class TestGradTelemetryUsesTheLiveWeight(unittest.TestCase):
    def _probe(self, step, **cfg_over):
        tr = _TelemetryTrainerStub(step, **cfg_over)
        _bind_trainer_method(tr, "_of_grad_telemetry")
        x = torch.randn(8, requires_grad=True)
        g_w = of_weight_at_step(
            tr.gan_of_cfg["gan_of_g_weight"], step,
            tr.gan_of_cfg["gan_of_disc_start_step"],
            tr.gan_of_cfg["gan_of_warmup_steps"],
        )
        # A GAN term whose gradient is exactly ``g_w`` times the DMD one,
        # so the unweighted share must come out at exactly 1.0.
        return tr._of_grad_telemetry(
            gan_term=(x * g_w).sum(), dmd_term=x.sum(), probe_tensor=x,
        ), g_w

    def test_no_warmup_is_unchanged(self):
        out, _ = self._probe(10)
        self.assertAlmostEqual(out["of_gan_dmd_ratio_unweighted"], 1.0,
                               places=5)

    def test_mid_warmup_the_share_is_still_one(self):
        """THE BUG: dividing by the config BASE mid-ramp reported the
        share low by exactly the warmup factor — here 10x — so the arm
        would have looked inert precisely while it was switching on."""
        out, g_w = self._probe(9, gan_of_warmup_steps=100)
        self.assertLess(g_w, OF_DEFAULTS["gan_of_g_weight"])
        self.assertAlmostEqual(out["of_gan_dmd_ratio_unweighted"], 1.0,
                               places=5)
        self.assertAlmostEqual(out["of_g_weight_live"], g_w, places=7)
        # ...and the pre-fix denominator would have given ~0.1.
        base = OF_DEFAULTS["gan_of_g_weight"]
        self.assertNotAlmostEqual(
            out["of_gan_dmd_grad_ratio"] / base, 1.0, places=3,
        )

    def test_before_the_start_step_the_key_is_omitted_not_forged(self):
        out, g_w = self._probe(3, gan_of_disc_start_step=50)
        self.assertEqual(g_w, 0.0)
        self.assertNotIn("of_gan_dmd_ratio_unweighted", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ======================================================================
# THE ATTACH POINT, PROVED RATHER THAN ASSERTED (2026-08-24).
#
# The previous wiring shipped on a source-level claim — "``train_chunk``
# is the exact object passed to ``compute_generator_loss_streaming``,
# i.e. the root of the DMD scoring graph" — that no test could falsify
# because no test ever ran a gradient through it. The GPU falsified it
# on the first smoke: 8/8 ranks raised out of ``_of_g_grad``'s guard on
# roll 2, because the chunk arrives DETACHED there (the boundary VAE
# round-trip's ``torch.cat`` ran inside a ``no_grad`` block).
#
# Everything below is executable. It builds a real ``nn.Module``
# generator, assembles the 42f rolling geometry around its output
# (detached GT context | graph-on supervised band | detached GT future),
# publishes the band through the REAL publisher, computes the REAL
# adversarial loss against a checkpointed stub disc, and then asks
# autograd — not the source — whether the gradient arrives.
# ======================================================================
class _BandGStub(_OFGLossStub):
    """``_OFGLossStub`` plus everything the band publisher touches."""

    def __init__(self, of_cfg, disc, ride):
        super().__init__(of_cfg, disc)
        self.streaming_state = {"ride_latents_window": ride}
        self.gan_of_enabled = True
        self._of_band = None
        self._of_current_step = 10

    def _surface_flash_gan_slab(self, info):
        return None


def _make_band_stub(ride, source="pred_image", dim=8, n_blocks=3):
    cfg = dict(OF_DEFAULTS)
    cfg["gan_of_g_weight"] = 0.03
    cfg["gan_of_fake_source"] = source
    disc = _StubDiscWrapper(dim=dim, n_blocks=n_blocks)
    disc.model.gradient_checkpointing = True
    stub = _BandGStub(cfg, disc, ride)
    for name in ("compute_of_g_loss", "_of_fake_sample", "_of_real_sample",
                 "_of_sample_timestep", "_of_disc_cond", "_of_disc_frozen",
                 "_of_assert_cond_detached",
                 "_of_disc_logits", "_of_g_grad", "_of_g_surrogate",
                 "_of_band_indices", "_of_publish_streaming_band",
                 "of_streaming_band", "of_step", "of_streaming_real"):
        _bind_model_method(stub, name)
    return stub


class _TinyGenerator(nn.Module):
    """Stands in for the student. One trainable parameter set, so
    "did the adversarial gradient reach the GENERATOR" is a question
    autograd can answer."""

    def __init__(self, c=2):
        super().__init__()
        self.proj = nn.Linear(c, c, bias=False)

    def forward(self, noise):
        # [B, F, C, H, W] -> project over C
        return self.proj(noise.movedim(2, -1)).movedim(-1, 2)


class TestAdversarialGradientReachesTheGeneratorThroughTheDMDGraph(
    unittest.TestCase
):
    """The executable replacement for the falsified source-level claim."""

    def _build(self, n_ctx=3, sup=4, gt_after=2, seed=0, detach_band=False):
        """The 42f rolling assembly, in miniature.

        ``score_image`` = cat([student ctx (detached),
                               student band (graph-on),
                               GT future (detached)]) — exactly the shape
        of ``_build_42f_scoring_inputs``'s ``parts`` list, and the tensor
        that method hands to ``compute_distribution_matching_loss``.
        """
        torch.manual_seed(seed)
        n_f = n_ctx + sup + gt_after
        ride = torch.randn(1, 40, 2, 2, 2)
        gen = _TinyGenerator(c=2)
        noise = torch.randn(1, n_ctx + sup, 2, 2, 2)
        rolled = gen(noise)                    # graph-on, [1, n_ctx+sup, ...]
        band_src = rolled[:, n_ctx:]
        if detach_band:
            band_src = band_src.detach()
        score_image = torch.cat(
            [rolled[:, :n_ctx].detach(), band_src, ride[:, :gt_after]],
            dim=1,
        )
        gt_target = ride[:, 5:5 + n_f]
        grad_mask = torch.zeros(1, n_f, 2, 2, 2, dtype=torch.bool)
        grad_mask[:, n_ctx:n_ctx + sup] = True
        stub = _make_band_stub(ride)
        return stub, gen, score_image, gt_target, grad_mask, (n_ctx,
                                                              n_ctx + sup)

    def _publish(self, stub, score_image, gt_target, grad_mask,
                 dmd_fired=True):
        stub._of_publish_streaming_band(
            score_image=score_image,
            score_gt_target=gt_target,
            score_cond={"prompt_embeds": torch.zeros(1, 1, 4)},
            score_grad_mask=grad_mask,
            chunk=score_image,
            info={},
            chunk_lo=0,
            chunk_hi=int(score_image.shape[1]),
            dmd_fired=dmd_fired,
            current_step=10,
        )
        return stub.of_streaming_band()

    # ---- 1. the fake IS a node of the DMD graph -----------------------
    def test_the_fake_is_a_node_of_the_tensor_the_dmd_loss_scores(self):
        stub, _gen, score_image, gt, mask, (lo, hi) = self._build()
        band = self._publish(stub, score_image, gt, mask)
        # the publisher recorded the very object it was handed...
        self.assertIs(band["score_image"], score_image)
        # ...and the fake differentiates back into it, on the band frames
        # and no others. This is the property a source grep cannot check
        # and the one the previous wiring got wrong.
        g = torch.autograd.grad(
            band["fake"].sum(), score_image, retain_graph=True,
        )[0]
        per_frame = g.abs().flatten(2).sum(-1)[0]
        self.assertTrue(bool((per_frame[lo:hi] > 0).all()))
        self.assertTrue(bool((per_frame[:lo] == 0).all()))
        self.assertTrue(bool((per_frame[hi:] == 0).all()))

    # ---- 2. the adversarial gradient reaches the GENERATOR ------------
    def test_the_g_term_gradient_is_nonzero_at_the_generator_parameters(self):
        stub, gen, score_image, gt, mask, _b = self._build()
        band = self._publish(stub, score_image, gt, mask)
        term = band["g_loss"]
        self.assertIsNotNone(term)
        self.assertTrue(term.requires_grad)
        g = torch.autograd.grad(term, gen.proj.weight, retain_graph=True)[0]
        self.assertIsNotNone(g)
        self.assertTrue(torch.isfinite(g).all())
        self.assertGreater(
            float(g.abs().sum()), 0.0,
            "the adversarial gradient does not reach the generator — this "
            "is the silent null result the arm exists to avoid",
        )

    # ---- 3. same graph as the DMD gradient ---------------------------
    def test_the_two_objectives_move_the_same_parameters(self):
        """Not just 'both are non-zero': both must arrive through the
        SAME tensor. A DMD-shaped surrogate built on ``score_image`` and
        the adversarial term must reach the generator by the same route.
        """
        stub, gen, score_image, gt, mask, _b = self._build()
        band = self._publish(stub, score_image, gt, mask)
        dmd_like = ((score_image - gt) * mask).float().pow(2).sum()
        g_dmd = torch.autograd.grad(
            dmd_like, gen.proj.weight, retain_graph=True,
        )[0]
        g_gan = torch.autograd.grad(
            band["g_loss"], gen.proj.weight, retain_graph=True,
        )[0]
        self.assertGreater(float(g_dmd.abs().sum()), 0.0)
        self.assertGreater(float(g_gan.abs().sum()), 0.0)
        self.assertEqual(g_dmd.shape, g_gan.shape)
        # ...and both are reachable from ``score_image`` — the ONE tensor
        # the DMD scorer was handed — with the adversarial gradient
        # confined to exactly the frames the DMD gradient mask marks.
        # (``band["fake"]`` is a SLICE of ``score_image``, so it is a
        # sibling of ``dmd_like`` rather than its ancestor; the shared
        # node is ``score_image`` itself.)
        s_dmd = torch.autograd.grad(
            dmd_like, score_image, retain_graph=True, allow_unused=True,
        )[0]
        s_gan = torch.autograd.grad(
            band["g_loss"], score_image, retain_graph=True,
            allow_unused=True,
        )[0]
        self.assertIsNotNone(s_dmd)
        self.assertIsNotNone(s_gan)
        lo, hi = band["band_lo"], band["band_hi"]
        for name, sg in (("dmd", s_dmd), ("gan", s_gan)):
            pf = sg.abs().flatten(2).sum(-1)[0]
            self.assertTrue(bool((pf[lo:hi] > 0).all()), name)
            self.assertTrue(bool((pf[:lo] == 0).all()), name)
            self.assertTrue(bool((pf[hi:] == 0).all()), name)

    # ---- 4. MUTATION CONTROL: a detached band must fail loud ----------
    def test_a_detached_band_raises_instead_of_adding_a_constant(self):
        """The guard that caught the boundary-VAE defect on the GPU.

        This is the exact condition the first smoke hit: the rollout
        arrives graph-free, so the adversarial term would otherwise add a
        CONSTANT to ``generator_loss`` and log a healthy ``of_g_loss``
        forever."""
        stub, _gen, score_image, gt, mask, _b = self._build(detach_band=True)
        with self.assertRaises(RuntimeError) as cm:
            self._publish(stub, score_image, gt, mask)
        self.assertIn("no autograd graph", str(cm.exception))

    # ---- 5. co-occurrence with the DMD term --------------------------
    def test_no_term_at_all_when_the_dmd_scorer_skipped_this_roll(self):
        stub, _gen, score_image, gt, mask, _b = self._build()
        band = self._publish(stub, score_image, gt, mask, dmd_fired=False)
        self.assertIsNone(band["g_loss"])
        self.assertEqual(band["g_logs"]["of_g_fired"], 0.0)
        # the band itself IS still published — the D side trains every
        # roll and must have the same fake to detach.
        self.assertIsNotNone(band["fake"])

    # ---- 6. the D side detaches the same tensor ----------------------
    def test_the_d_side_would_detach_the_identical_fake(self):
        stub, _gen, score_image, gt, mask, _b = self._build()
        band = self._publish(stub, score_image, gt, mask)
        d_fake = band["fake"].detach()
        torch.testing.assert_close(d_fake, band["fake"].detach())
        self.assertFalse(d_fake.requires_grad)
        # the G side's fake is the same object, still live
        self.assertTrue(band["fake"].requires_grad)

    # ---- 7. the disc is not trained by the G forward ------------------
    def test_the_disc_keeps_its_flags_and_gets_no_gradient(self):
        stub, _gen, score_image, gt, mask, _b = self._build()
        before = [p.requires_grad for p in stub.fake_score.parameters()]
        self._publish(stub, score_image, gt, mask)
        after = [p.requires_grad for p in stub.fake_score.parameters()]
        self.assertEqual(before, after)
        for p in stub.fake_score.parameters():
            self.assertIsNone(p.grad)


# ======================================================================
# ROOT CAUSE — the boundary VAE round-trip severed the rollout graph.
#
# ``generate_next_chunk`` built its replacement chunk with a
# ``torch.cat`` executed INSIDE the ``with torch.no_grad():`` block that
# wraps the VAE decode/encode. ``torch.cat`` under ``no_grad`` returns a
# tensor with no ``grad_fn``, so the ENTIRE rolled chunk — supervised
# band included — came back graph-free on every overlapped roll, while
# the code's own comment claimed "this does NOT break gradient flow on
# the new frames".
#
# Downstream that made the streaming DMD generator loss a constant on
# every roll after the first, and ``_phase_lora_ghost_anchor``'s live
# ``0.0 * ghost`` term kept ``generator_loss.requires_grad`` True so the
# DDP-lockstep ``gen_backward_skipped`` gauge never fired. The only
# fingerprint in the logs was ``[42F-ROLLING] ... (graph-on=False)``.
#
# The round-trip is now a method precisely so this is testable without a
# pipeline, a ride or a real VAE.
# ======================================================================
class _StubVAE:
    """``decode_to_pixel`` / ``encode_to_latent`` reduced to shapes.

    Deliberately NOT a no-op: the returned boundary latent must be
    distinguishable from the frame it replaced, or "did the replacement
    happen" is untestable.
    """

    def __init__(self):
        self.decode_calls = 0
        self.encode_calls = 0

    def decode_to_pixel(self, latents, use_cache=False):
        self.decode_calls += 1
        # [B, T, C, H, W] latent -> [B, T, C, H, W] "pixels"
        return latents * 2.0

    def encode_to_latent(self, frames_bcthw):
        self.encode_calls += 1
        # [B, C, T, H, W] -> [B, T, C, H, W]
        return frames_bcthw.movedim(1, 2) + 100.0


class _BoundaryStub:
    def __init__(self, keep_graph):
        self.vae = _StubVAE()
        self.boundary_vae_roundtrip = True
        self.boundary_vae_roundtrip_keep_graph = bool(keep_graph)
        self._boundary_vae_graph_warned = False


def _make_boundary_stub(keep_graph):
    stub = _BoundaryStub(keep_graph)
    for name in ("_boundary_vae_roundtrip", "_warn_boundary_vae_graph_severed"):
        _bind_model_method(stub, name)
    return stub


class TestBoundaryVAERoundTripGraph(unittest.TestCase):
    def _chunk(self):
        torch.manual_seed(3)
        gen = _TinyGenerator(c=2)
        noise = torch.randn(1, 6, 2, 2, 2)
        return gen, gen(noise)

    def test_legacy_default_severs_the_graph(self):
        """The measured behaviour, pinned. Default-off must stay
        byte-identical until the flip is signed off."""
        stub = _make_boundary_stub(keep_graph=False)
        gen, chunk = self._chunk()
        self.assertTrue(chunk.requires_grad)
        out = stub._boundary_vae_roundtrip(
            chunk, torch.randn(1, 3, 2, 2, 2), torch.float32,
        )
        self.assertFalse(
            out.requires_grad,
            "the legacy path is defined by the severed graph; if this "
            "starts passing the default silently changed",
        )
        self.assertIsNone(out.grad_fn)

    def test_keep_graph_preserves_the_students_gradient(self):
        stub = _make_boundary_stub(keep_graph=True)
        gen, chunk = self._chunk()
        out = stub._boundary_vae_roundtrip(
            chunk, torch.randn(1, 3, 2, 2, 2), torch.float32,
        )
        self.assertTrue(out.requires_grad)
        g = torch.autograd.grad(out.sum(), gen.proj.weight)[0]
        self.assertGreater(float(g.abs().sum()), 0.0)

    def test_the_replacement_values_are_identical_either_way(self):
        """Only the GRAPH differs. If the numbers moved, the flag would
        be a recipe change in disguise rather than a bug fix."""
        prev = torch.randn(1, 3, 2, 2, 2)
        gen, chunk = self._chunk()
        a = _make_boundary_stub(False)._boundary_vae_roundtrip(
            chunk, prev, torch.float32)
        b = _make_boundary_stub(True)._boundary_vae_roundtrip(
            chunk, prev, torch.float32)
        torch.testing.assert_close(a, b.detach())

    def test_only_the_seam_frame_is_replaced(self):
        prev = torch.randn(1, 3, 2, 2, 2)
        gen, chunk = self._chunk()
        out = _make_boundary_stub(True)._boundary_vae_roundtrip(
            chunk, prev, torch.float32)
        torch.testing.assert_close(out[:, 1:], chunk[:, 1:])
        self.assertFalse(torch.allclose(out[:, 0], chunk[:, 0]))

    def test_the_vae_forwards_stay_under_no_grad_on_both_paths(self):
        """The fix moves the CAT out of the block, not the VAE."""
        prev = torch.randn(1, 3, 2, 2, 2, requires_grad=True)
        gen, chunk = self._chunk()
        out = _make_boundary_stub(True)._boundary_vae_roundtrip(
            chunk, prev, torch.float32)
        # the seam frame carries no graph at all -> the decode/encode ran
        # under no_grad, so neither ``prev`` nor ``chunk[:, 0]`` is
        # reachable through it.
        g_prev, g_gen = torch.autograd.grad(
            out[:, 0].sum(), [prev, gen.proj.weight], allow_unused=True,
        )
        self.assertIsNone(g_prev)
        # ``g_gen`` is reachable (the cat node is shared with frames 1:)
        # but must be EXACTLY zero: nothing flows through the seam frame.
        self.assertEqual(float(g_gen.abs().sum()), 0.0)

    def test_the_severing_is_warned_about_exactly_once_and_only_when_live(self):
        stub = _make_boundary_stub(keep_graph=False)
        prev = torch.randn(1, 3, 2, 2, 2)
        # a no-grad prebuild rollout had no graph to lose -> stay quiet
        stub._boundary_vae_roundtrip(
            torch.randn(1, 6, 2, 2, 2), prev, torch.float32)
        self.assertFalse(stub._boundary_vae_graph_warned)
        # a live rollout -> warn, once
        _gen, chunk = self._chunk()
        stub._boundary_vae_roundtrip(chunk, prev, torch.float32)
        self.assertTrue(stub._boundary_vae_graph_warned)

    def test_the_call_site_still_gates_on_overlap_and_a_previous_chunk(self):
        from model.dmd_action_forcing import ActionForcingDMD
        src = _code_only(_method_src(ActionForcingDMD, "generate_next_chunk"))
        i = src.index("self._boundary_vae_roundtrip(")
        head = src[max(0, i - 400):i]
        self.assertIn("self.boundary_vae_roundtrip", head)
        self.assertIn("overlap > 0", head)
        self.assertIn("prev_chunk_for_clean is not None", head)

    def test_the_flag_now_defaults_ON_after_researcher_sign_off(self):
        """Default flipped False -> True on 2026-08-25 with explicit sign-off.

        The old default was byte-identical to the MEASURED (broken)
        behaviour: the boundary VAE round-trip's ``torch.cat`` ran inside
        ``no_grad``, so every overlapped roll came back graph-free and the
        streaming DMD generator loss was a CONSTANT past roll 1. Keeping
        the fix opt-in meant every recipe silently kept the dead gradient.
        Flipping it is a training-recipe change and was signed off
        explicitly; this test pins the new default so it cannot regress
        back to the broken behaviour unnoticed.
        """
        from model.dmd_action_forcing import ActionForcingDMD
        import inspect
        init = inspect.getsource(ActionForcingDMD.__init__)
        self.assertIn(
            'getattr(args, "boundary_vae_roundtrip_keep_graph", True)', init,
        )
        self.assertNotIn(
            'getattr(args, "boundary_vae_roundtrip_keep_graph", False)', init,
        )


# ======================================================================
# ROOT CAUSE #3 — the D term's CONDITIONING carried the generator graph,
# so the ``streaming_fake_updates_per_gen`` inner critic backwards
# re-traversed a subgraph ``critic_loss.backward()`` had already freed.
#
# ``logs/of_smoke_r3.log``, 8 ranks / 2 nodes, ALL of them:
#
#     File "trainer/causal_action_forcing_train.py", line 15959,
#       in _streaming_train_one_chunk
#         _extra_loss.backward()
#     RuntimeError: Trying to backward through the graph a second time
#
# The D term itself was ALREADY rebuilt per inner iteration (a fresh
# ``_of_streaming_d_term`` -> fresh ``compute_of_d_loss`` -> fresh disc
# forward), and its fake and real were ALREADY detached. The poison was
# an INPUT: on the ``pred_image`` branch the published band's cond was
# ``score_cond`` sliced, and ``score_cond`` is
# ``build_action_conditional``'s live output — ``model/base.py`` does
# ``action_projection.requires_grad_(True)`` unconditionally. So every
# disc forward hung off the GENERATOR's action-projection subgraph;
# ``critic_loss.backward()`` freed it, and the next inner backward walked
# into the corpse. (The ``flash`` branch detached via
# ``of_streaming_cond``; ``pred_image`` did not — the asymmetry IS the
# bug.) Fixed in ``_of_disc_cond``, the one choke point all four disc
# call sites funnel through.
#
# These tests execute the REAL publisher, the REAL
# ``_of_streaming_d_term``, the REAL ``compute_of_d_loss`` /
# ``of_head_touch`` / ``_of_disc_cond`` against a real ``nn.Module``
# disc, and take 1 + N sequential backwards in the trainer's own order.
# Source-level reasoning is what shipped this bug three times running.
# ======================================================================
class _OFInnerDisc(nn.Module):
    """``fake_score``'s inner module.

    A denoising ``backbone`` the critic loss reaches, plus the three
    register-token head modules that ONLY the classify forward reaches —
    under the real names ``of_head_touch`` scans
    (``_OF_HEAD_MODULE_NAMES``).
    """

    def __init__(self, dim=4):
        super().__init__()
        self.dim = dim
        self.backbone = nn.Linear(dim, dim, bias=False)
        self._register_tokens = nn.Linear(dim, dim, bias=False)
        self._gan_ca_blocks = nn.ModuleList(
            [nn.Linear(dim, dim, bias=False)]
        )
        self._cls_pred_branch = nn.Linear(dim, 1, bias=False)
        self.gradient_checkpointing = False

    def head_parameters(self):
        for m in (self._cls_pred_branch, self._register_tokens,
                  self._gan_ca_blocks):
            yield from m.parameters()


class _OFInnerDiscWrapper(nn.Module):
    """``WanDiffusionWrapper`` reduced to the classify forward.

    The action-modulation stream is FOLDED INTO THE FEATURES, so if the
    cond carries a graph the disc output genuinely depends on it — which
    is what makes the mutation control reproduce the real crash rather
    than a lookalike.
    """

    def __init__(self, dim=4):
        super().__init__()
        self.model = _OFInnerDisc(dim)

    def _unwrapped_model(self):
        return self.model

    def forward(self, noisy_image_or_video, conditional_dict, timestep,
                classify_mode=False):
        assert classify_mode, "the OF disc forward is always classify_mode"
        m = self.model
        x = noisy_image_or_video.reshape(
            noisy_image_or_video.shape[0], -1, m.dim,
        )
        h = m.backbone(x)
        am = (conditional_dict or {}).get("_action_modulation")
        if am is not None:
            h = h + am.reshape(am.shape[0], -1, m.dim).mean(
                dim=1, keepdim=True,
            )
        tok = m._register_tokens(h.mean(dim=1, keepdim=True))
        for blk in m._gan_ca_blocks:
            tok = tok + blk(h).mean(dim=1, keepdim=True)
        return None, None, m._cls_pred_branch(tok).reshape(h.shape[0], 1)


class _OFDTermModelStub:
    """Everything the D path + the band publisher touch, and nothing else."""

    def __init__(self, of_cfg, disc, ride):
        self.of_cfg = of_cfg
        self.fake_score = disc
        self.scheduler = _StubScheduler()
        self.streaming_state = {"ride_latents_window": ride}
        self.gan_of_enabled = True
        self._of_band = None
        self._of_current_step = 10

    def _surface_flash_gan_slab(self, info):
        return None


def _make_d_term_stub(ride, dim=4):
    cfg = dict(OF_DEFAULTS)
    cfg["gan_of_d_weight"] = 0.03
    stub = _OFDTermModelStub(cfg, _OFInnerDiscWrapper(dim=dim), ride)
    for name in ("compute_of_d_loss", "of_head_touch", "_of_disc_cond",
                 "_of_assert_cond_detached", "_of_disc_logits",
                 "_of_sample_timestep", "_of_real_sample", "_of_fake_sample",
                 "_of_band_indices", "_of_publish_streaming_band",
                 "of_streaming_band", "of_step", "of_streaming_real",
                 "compute_of_g_loss", "_of_disc_frozen", "_of_g_grad",
                 "_of_g_surrogate"):
        _bind_model_method(stub, name)
    return stub


def _prefix_strip_only_disc_cond(cond):
    """The PRE-FIX ``_of_disc_cond``: strips ``*_clean``, detaches nothing.

    Kept verbatim so the mutation control runs the code that actually
    crashed on the GPU, not an approximation of it.
    """
    return {k: v for k, v in cond.items() if not k.endswith("_clean")}


class TestEveryCriticBackwardCarriesALiveDTerm(unittest.TestCase):
    """1 + ``streaming_fake_updates_per_gen`` sequential critic backwards.

    Mirrors ``_streaming_train_one_chunk`` exactly: the main critic loss
    (denoising + D) backwards first and FREES, then each inner update
    rebuilds its own (denoising + D) and backwards again.
    """

    N_INNER = 4          # the smoke's streaming_fake_updates_per_gen=4
    DIM = 4

    # ---- assembly -----------------------------------------------------
    def _assemble(self, seed=0):
        """The 42f scoring geometry in miniature, with a LIVE action
        projection feeding the cond — the generator-owned module whose
        subgraph the critic backward frees."""
        torch.manual_seed(seed)
        n_ctx, sup, gt_after = 3, 4, 2
        n_f = n_ctx + sup + gt_after
        ride = torch.randn(1, 40, self.DIM, 1, 1)
        gen = _TinyGenerator(c=self.DIM)
        # ``action_projection`` — generator-owned, requires_grad=True by
        # construction (model/base.py:101), exactly like the real one.
        act_proj = nn.Linear(2, self.DIM, bias=False)
        rolled = gen(torch.randn(1, n_ctx + sup, self.DIM, 1, 1))
        score_image = torch.cat(
            [rolled[:, :n_ctx].detach(), rolled[:, n_ctx:],
             ride[:, :gt_after]], dim=1,
        )
        grad_mask = torch.zeros(1, n_f, self.DIM, 1, 1, dtype=torch.bool)
        grad_mask[:, n_ctx:n_ctx + sup] = True
        # ``build_action_conditional``'s output: graph-carrying.
        score_cond = {
            "_action_modulation": act_proj(torch.randn(1, n_f, 2)),
            "_action_modulation_clean": act_proj(torch.randn(1, n_f, 2)),
            "prompt_embeds": torch.zeros(1, 1, 4),
        }
        self.assertTrue(
            score_cond["_action_modulation"].requires_grad,
            "the harness must reproduce a LIVE cond or it tests nothing",
        )
        stub = _make_d_term_stub(ride, dim=self.DIM)
        tr = _OFTrainerStub(stub, step=10, telemetry_every=1)
        _bind_trainer_method(tr, "_of_streaming_d_term")
        _bind_trainer_method(tr, "_of_telemetry_step")
        return dict(
            stub=stub, tr=tr, gen=gen, act_proj=act_proj,
            score_image=score_image, gt_target=ride[:, 5:5 + n_f],
            grad_mask=grad_mask, score_cond=score_cond,
        )

    def _publish(self, a):
        a["stub"]._of_publish_streaming_band(
            score_image=a["score_image"],
            score_gt_target=a["gt_target"],
            score_cond=a["score_cond"],
            score_grad_mask=a["grad_mask"],
            chunk=a["score_image"],
            info={},
            chunk_lo=0,
            chunk_hi=int(a["score_image"].shape[1]),
            dmd_fired=False,      # D fires on EVERY roll, G only when DMD does
            current_step=10,
        )
        return a["stub"].of_streaming_band()

    def _denoise_surrogate(self, a):
        """Stands in for ``compute_critic_loss_streaming``: a fresh loss
        on DETACHED inputs that reaches the disc BACKBONE but never the
        head — the exact shape that makes ``of_head_touch`` load-bearing.
        """
        inner = a["stub"].fake_score.model
        x = a["score_image"].detach().reshape(1, -1, self.DIM)
        return inner.backbone(x).pow(2).mean()

    def _head_grads(self, a):
        return [
            None if p.grad is None else p.grad.detach().clone()
            for p in a["stub"].fake_score.model.head_parameters()
        ]

    def _zero_fake_grads(self, a):
        for p in a["stub"].fake_score.parameters():
            p.grad = None

    # ---- 1. THE FIX: N+1 sequential backwards all succeed -------------
    def test_all_inner_critic_backwards_succeed_with_a_live_d_term(self):
        a = self._assemble()
        self._publish(a)
        out = {}

        # --- the MAIN critic update (trainer lines ~15810-15825) -------
        main = self._denoise_surrogate(a)
        d_main = a["tr"]._of_streaming_d_term(out=out)
        self.assertIsNotNone(d_main)
        self.assertTrue(d_main.requires_grad)
        (main + d_main).backward()          # <- frees, no retain_graph
        head_after_main = self._head_grads(a)
        self.assertTrue(
            any(g is not None and float(g.abs().sum()) > 0.0
                for g in head_after_main),
            "the MAIN critic update produced no disc-head gradient",
        )

        # --- the N inner updates (trainer lines ~15930-15960) ----------
        gaps = [out["of_logit_gap"]]
        for i in range(self.N_INNER):
            self._zero_fake_grads(a)        # fake_optimizer.zero_grad
            inner_out = {}
            extra = self._denoise_surrogate(a)
            d_extra = a["tr"]._of_streaming_d_term(
                out=inner_out, log_prefix="of_inner_",
            )
            self.assertIsNotNone(d_extra, f"inner {i}: no D term")
            self.assertTrue(
                d_extra.requires_grad, f"inner {i}: D term has no graph",
            )
            extra = extra + d_extra
            extra.backward()                # THE LINE THAT CRASHED
            grads = self._head_grads(a)
            self.assertTrue(
                all(g is not None for g in grads),
                f"inner {i}: a disc-head parameter was left ungradiented — "
                "the fake_score DDP reducer would raise on the next step",
            )
            self.assertGreater(
                sum(float(g.abs().sum()) for g in grads), 0.0,
                f"inner {i}: the D term contributed ZERO head gradient, so "
                "this fake_score update was NOT adversarially trained",
            )
            gaps.append(inner_out["of_inner_of_logit_gap"])

        # Each update drew its own (t, eps) and ran its own disc forward:
        # 5 distinct logit gaps, not one value reused 5 times.
        self.assertEqual(len(gaps), self.N_INNER + 1)
        self.assertEqual(
            len(set(gaps)), len(gaps),
            "the D term was computed ONCE and reused — every critic "
            "update must train the disc on a fresh draw",
        )

    # ---- 2. MUTATION CONTROL: the pre-fix cond reproduces the crash ---
    def test_an_undetached_cond_reproduces_the_gpu_runtimeerror(self):
        """Revert ONLY the detach and the r3 failure comes straight back."""
        a = self._assemble()
        stub = a["stub"]
        # The pre-fix ``_of_disc_cond`` (strip-only). Patched on the stub
        # so the REAL ``compute_of_d_loss`` consumes a live cond, exactly
        # as it did on the GPU.
        stub._of_disc_cond = _prefix_strip_only_disc_cond
        # Hand-build the band the pre-fix publisher would have produced
        # (the publisher now refuses a live cond -- see test 3).
        n_f = int(a["score_image"].shape[1])
        lo, hi = 3, 7
        stub._of_band = {
            "fake": a["score_image"][:, lo:hi],
            "real": a["gt_target"][:, lo:hi].detach(),
            "cond": _prefix_strip_only_disc_cond(a["score_cond"]),
            "gt_pool": a["stub"].streaming_state["ride_latents_window"],
            "score_image": a["score_image"],
            "band_lo": lo, "band_hi": hi, "source": "pred_image",
            "dmd_fired": False, "step": 10,
            "g_loss": None, "g_fake": None, "g_logs": {"of_g_fired": 0.0},
        }
        self.assertTrue(stub._of_band["cond"]["_action_modulation"]
                        .requires_grad)
        self.assertEqual(n_f, 9)

        out = {}
        main = self._denoise_surrogate(a)
        d_main = a["tr"]._of_streaming_d_term(out=out)
        (main + d_main).backward()

        with self.assertRaises(RuntimeError) as cm:
            self._zero_fake_grads(a)
            extra = self._denoise_surrogate(a)
            extra = extra + a["tr"]._of_streaming_d_term(
                out={}, log_prefix="of_inner_",
            )
            extra.backward()
        self.assertIn("backward through the graph a second time",
                      str(cm.exception))

    # ---- 3. the publisher refuses a live cond, loudly -----------------
    def test_the_publisher_raises_if_the_cond_still_carries_a_graph(self):
        a = self._assemble()
        a["stub"]._of_disc_cond = _prefix_strip_only_disc_cond
        with self.assertRaises(RuntimeError) as cm:
            self._publish(a)
        msg = str(cm.exception)
        self.assertIn("still requires grad", msg)
        self.assertIn("_action_modulation", msg)

    # ---- 4. the published cond is detached, and clean-stripped --------
    def test_the_published_cond_is_detached(self):
        a = self._assemble()
        band = self._publish(a)
        for k, v in band["cond"].items():
            if torch.is_tensor(v):
                self.assertFalse(
                    v.requires_grad, f"band cond {k!r} still carries a graph",
                )
        self.assertNotIn("_action_modulation_clean", band["cond"])
        # ...and the VALUES are unchanged: this is a graph edit, not a
        # numerics edit. The disc sees the same conditioning it always did.
        torch.testing.assert_close(
            band["cond"]["_action_modulation"],
            a["score_cond"]["_action_modulation"].detach()[:, 3:7],
        )

    # ---- 5. no disc gradient escapes into the generator ---------------
    def test_the_d_term_moves_no_generator_or_action_projection_param(self):
        a = self._assemble()
        self._publish(a)
        out = {}
        d = a["tr"]._of_streaming_d_term(out=out)
        d.backward()
        for name, p in a["gen"].named_parameters():
            self.assertIsNone(p.grad, f"D term reached the generator: {name}")
        self.assertIsNone(
            a["act_proj"].weight.grad,
            "D term pushed discriminator gradient into action_projection — "
            "a generator-owned parameter applied by the generator's "
            "optimizer, wrong-signed and silent",
        )
        # ...while the disc DID get trained.
        self.assertGreater(
            sum(float(p.grad.abs().sum())
                for p in a["stub"].fake_score.model.head_parameters()
                if p.grad is not None),
            0.0,
        )

    # ---- 6. the head is touched even when the D weight is off ---------
    def test_head_is_reached_on_every_inner_update_with_the_weight_at_zero(
        self,
    ):
        """``gan_of_d_weight=0`` / ``disc_start_step>0``: the D term
        degrades to ``of_head_touch()``, which must STILL ride every one
        of the N+1 backwards or the fake_score reducer starves."""
        a = self._assemble()
        a["stub"].of_cfg["gan_of_d_weight"] = 0.0
        a["tr"].gan_of_cfg["gan_of_d_weight"] = 0.0
        self._publish(a)
        for i in range(self.N_INNER + 1):
            self._zero_fake_grads(a)
            loss = self._denoise_surrogate(a)
            term = a["tr"]._of_streaming_d_term(out={}, log_prefix="p_")
            self.assertIsNotNone(term, f"update {i}: no head tie-in")
            (loss + term).backward()
            for p in a["stub"].fake_score.model.head_parameters():
                self.assertIsNotNone(
                    p.grad, f"update {i}: head parameter ungradiented",
                )
                self.assertTrue(torch.equal(p.grad, torch.zeros_like(p.grad)))


# ======================================================================
# #14 — ``gan_of_backbone_trainable``: un-confounding the arm.
#
# As first shipped the arm moved TWO variables at once: (a) the disc taps
# the TRAINABLE ``fake_score`` critic instead of the frozen ``real_score``
# teacher, and (b) that backbone CO-TRAINS on the adversarial loss. The
# thesis is that (b) is the active ingredient, but with both moving
# neither a positive nor a negative result can attribute anything.
#
# ``gan_of_backbone_trainable=false`` keeps (a) and removes (b). These
# tests execute the REAL ``compute_of_d_loss`` / ``_of_head_only_surrogate``
# / ``of_head_touch`` against a real ``nn.Module`` disc — with the DiT
# stack CHECKPOINTED, because a ``requires_grad``-toggling implementation
# of this flag would die exactly there.
# ======================================================================
class _BTDisc(nn.Module):
    """``fake_score``'s inner module, with the parts that matter.

    ``blocks`` + ``time_embedding`` are the BACKBONE; the three
    ``_OF_HEAD_MODULE_NAMES`` modules are the head. ``time_embedding``
    is deliberately consumed by the HEAD as well (the
    ``concat_time_embeddings`` shape): it is the backbone parameter a
    detach-at-the-feature-tap implementation would leave adversarially
    trained, so the tests can tell the two implementations apart.
    """

    def __init__(self, dim=8, n_blocks=3):
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList(
            [nn.Linear(dim, dim, bias=False) for _ in range(n_blocks)]
        )
        self.time_embedding = nn.Linear(1, dim, bias=False)
        self._register_tokens = nn.Linear(dim, dim, bias=False)
        self._gan_ca_blocks = nn.ModuleList([nn.Linear(dim, dim, bias=False)])
        self._cls_pred_branch = nn.Linear(dim, 1, bias=False)
        self.gradient_checkpointing = True

    # -- the two names the tests partition every parameter by ----------
    HEAD_PREFIXES = ("_cls_pred_branch", "_register_tokens", "_gan_ca_blocks")

    def head_parameters(self):
        for n, p in self.named_parameters():
            if n.startswith(self.HEAD_PREFIXES):
                yield n, p

    def backbone_parameters(self):
        for n, p in self.named_parameters():
            if not n.startswith(self.HEAD_PREFIXES):
                yield n, p

    def forward(self, latent, timestep, classify_mode=False):
        B = latent.shape[0]
        h = latent.reshape(B, -1, self.dim)
        for blk in self.blocks:
            if self.gradient_checkpointing:
                h = torch.utils.checkpoint.checkpoint(
                    lambda t, m=blk: m(t).tanh(), h, use_reentrant=False,
                )
            else:
                h = blk(h).tanh()
        te = self.time_embedding(
            timestep.float().reshape(B, -1)[:, :1] / 1000.0
        ).unsqueeze(1)
        if not classify_mode:
            # The DENOISING forward: reaches every backbone parameter and
            # not one head parameter — the real asymmetry.
            return h + te
        tok = self._register_tokens(h.mean(dim=1, keepdim=True))
        for blk in self._gan_ca_blocks:
            tok = tok + blk(h).mean(dim=1, keepdim=True)
        tok = tok + te
        return self._cls_pred_branch(tok).reshape(B, 1)


class _BTDiscWrapper(nn.Module):
    """``WanDiffusionWrapper`` reduced to what the D path uses, DDP-able:
    the wrapper CALLS ``self.model``, so ``self.model`` can be a DDP
    object exactly as the trainer leaves it."""

    def __init__(self, dim=8, n_blocks=3):
        super().__init__()
        self.dim = dim
        self.model = _BTDisc(dim, n_blocks)
        self.forwards = 0

    def _unwrapped_model(self):
        m = self.model
        return m.module if hasattr(m, "module") else m

    def forward(self, noisy_image_or_video, conditional_dict, timestep,
                classify_mode=False):
        self.forwards += 1
        out = self.model(
            noisy_image_or_video, timestep, classify_mode=classify_mode,
        )
        return (None, None, out) if classify_mode else (None, out, None)


class _BTStub:
    """Everything ``compute_of_d_loss`` touches, and nothing else."""

    def __init__(self, of_cfg, disc):
        self.of_cfg = of_cfg
        self.fake_score = disc
        self.scheduler = _StubScheduler()
        self._of_current_step = 10


_BT_METHODS = (
    "compute_of_d_loss", "of_head_touch", "_of_head_parameters",
    "_of_head_only_surrogate", "_of_disc_cond", "_of_assert_cond_detached",
    "_of_disc_logits", "_of_sample_timestep", "of_step",
)


def _make_bt_stub(trainable, dim=8, n_blocks=3, ckpt=True, seed=0):
    """``trainable=None`` builds a config that PREDATES the flag — the
    byte-identity reference."""
    torch.manual_seed(seed)
    cfg = dict(OF_DEFAULTS)
    cfg["gan_of_d_weight"] = 0.03
    if trainable is None:
        cfg.pop("gan_of_backbone_trainable")
    else:
        cfg["gan_of_backbone_trainable"] = bool(trainable)
    disc = _BTDiscWrapper(dim=dim, n_blocks=n_blocks)
    disc.model.gradient_checkpointing = bool(ckpt)
    stub = _BTStub(cfg, disc)
    for name in _BT_METHODS:
        _bind_model_method(stub, name)
    return stub


def _bt_inputs(dim=8, frames=4, seed=1):
    torch.manual_seed(seed)
    fake = torch.randn(1, frames, dim, 1, 1)
    real = torch.randn(1, frames, dim, 1, 1)
    return fake, real


def _bt_d_term(stub, seed=1, telemetry=True):
    """One D call with the RNG pinned, so the two arms see the SAME
    timestep and the SAME epsilon and any difference in the returned
    value is the flag's doing and nothing else."""
    fake, real = _bt_inputs(dim=stub.fake_score.dim, seed=seed)
    torch.manual_seed(seed + 100)
    return stub.compute_of_d_loss(
        fake_latent=fake, real_latent=real, cond_for_scoring={},
        current_step=10, telemetry=telemetry,
    )


def _bt_denoise(stub, seed=7):
    """The critic's own loss: reaches every BACKBONE parameter, no head
    parameter. Same shape as ``compute_critic_loss_streaming``'s."""
    torch.manual_seed(seed)
    dim = stub.fake_score.dim
    x = torch.randn(1, 4, dim, 1, 1)
    t = torch.full((1, 4), 500.0)
    out = stub.fake_score(x, {}, t, classify_mode=False)[1]
    return out.pow(2).mean()


def _bt_grads(inner):
    return {
        n: (None if p.grad is None else p.grad.detach().clone())
        for n, p in inner.named_parameters()
    }


def _bt_zero(inner):
    for p in inner.parameters():
        p.grad = None


class TestBackboneTrainableConfig(unittest.TestCase):
    def test_the_flag_exists_and_defaults_true(self):
        self.assertIs(OF_DEFAULTS["gan_of_backbone_trainable"], True)

    def test_a_config_predating_the_flag_resolves_to_the_faithful_arm(self):
        self.assertIs(
            resolve_of_config(_Cfg())["gan_of_backbone_trainable"], True,
        )

    def test_the_control_is_reachable_from_config(self):
        r = resolve_of_config(_Cfg(gan_of_backbone_trainable=False))
        self.assertIs(r["gan_of_backbone_trainable"], False)

    def test_the_yaml_registers_the_key(self):
        """The override guard only sees keys the config declares; an
        unregistered ``gan_of_*`` key is a hard kill under
        ``strict_override_keys``."""
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "action_forcing_phase3_dmd.yaml",
        )
        with open(path) as fh:
            text = fh.read()
        self.assertIn("gan_of_backbone_trainable: true", text)

    def test_the_key_is_in_the_override_guard_registration_list(self):
        from model.one_forcing_gan import CONFIG_KEYS
        self.assertIn("gan_of_backbone_trainable", CONFIG_KEYS)


class TestBackboneTrainableTrueIsByteIdentical(unittest.TestCase):
    """Default TRUE must be the arm as launched, to the bit."""

    def _run(self, trainable):
        stub = _make_bt_stub(trainable)
        inner = stub.fake_score._unwrapped_model()
        term, logs = _bt_d_term(stub)
        (_bt_denoise(stub) + term).backward()
        return float(term.detach()), logs, _bt_grads(inner), stub

    def test_value_logs_and_every_gradient_match_a_preflag_config(self):
        v_ref, logs_ref, g_ref, s_ref = self._run(None)    # flag absent
        v_new, logs_new, g_new, s_new = self._run(True)    # flag present
        self.assertEqual(v_ref, v_new)
        self.assertEqual(logs_ref, logs_new)
        self.assertEqual(set(g_ref), set(g_new))
        for n in g_ref:
            self.assertIsNotNone(g_ref[n], f"{n}: reference had no grad")
            self.assertTrue(
                torch.equal(g_ref[n], g_new[n]),
                f"{n}: gradient changed under the default flag",
            )
        self.assertEqual(s_ref.fake_score.forwards, s_new.fake_score.forwards)

    def test_no_extra_log_key_appears_on_the_faithful_arm(self):
        _v, logs, _g, _s = self._run(True)
        self.assertNotIn("of_backbone_trainable", logs)

    def test_the_faithful_arm_really_does_train_the_backbone(self):
        """MUTATION CONTROL for every 'zero on the backbone' assertion
        below: with the flag TRUE the adversarial-only backward MUST move
        the backbone, or those assertions prove nothing."""
        stub = _make_bt_stub(True)
        inner = stub.fake_score._unwrapped_model()
        term, _logs = _bt_d_term(stub)
        term.backward()
        moved = [
            n for n, p in inner.backbone_parameters()
            if p.grad is not None and float(p.grad.abs().sum()) > 0.0
        ]
        self.assertEqual(
            len(moved), len(list(inner.backbone_parameters())),
            f"only {moved} of the backbone was adversarially trained",
        )


class TestBackboneTrainableFalseRestrictsToTheHead(unittest.TestCase):
    def test_adversarial_backward_deposits_zero_on_the_backbone(self):
        stub = _make_bt_stub(False)
        inner = stub.fake_score._unwrapped_model()
        term, _logs = _bt_d_term(stub)
        term.backward()
        for n, p in inner.backbone_parameters():
            self.assertTrue(
                p.grad is None or float(p.grad.abs().sum()) == 0.0,
                f"{n} received ADVERSARIAL gradient with "
                "gan_of_backbone_trainable=false",
            )

    def test_the_time_embedding_is_covered(self):
        """The head consumes ``time_embedding`` directly (the
        ``concat_time_embeddings`` shape), so a detach-at-the-tap
        implementation would still train it. Named separately because it
        is the one parameter that distinguishes the two designs."""
        stub = _make_bt_stub(False)
        inner = stub.fake_score._unwrapped_model()
        _bt_d_term(stub)[0].backward()
        self.assertTrue(
            inner.time_embedding.weight.grad is None
            or float(inner.time_embedding.weight.grad.abs().sum()) == 0.0
        )

    def test_all_three_head_module_groups_are_gradiented(self):
        stub = _make_bt_stub(False)
        inner = stub.fake_score._unwrapped_model()
        _bt_d_term(stub)[0].backward()
        for prefix in _BTDisc.HEAD_PREFIXES:
            tot = sum(
                float(p.grad.abs().sum())
                for n, p in inner.head_parameters()
                if n.startswith(prefix) and p.grad is not None
            )
            self.assertGreater(
                tot, 0.0, f"{prefix} got NO adversarial gradient",
            )
        for n, p in inner.head_parameters():
            self.assertIsNotNone(
                p.grad,
                f"{n} has grad=None — the fake_score reducer "
                "(find_unused_parameters=False) starves on it",
            )

    def test_the_head_gradient_is_bit_identical_to_the_faithful_arm(self):
        """Only the BACKBONE edge is removed. If the head's learning
        signal changed too, the A/B would still be confounded."""
        g = {}
        for name, flag in (("true", True), ("false", False)):
            stub = _make_bt_stub(flag)
            inner = stub.fake_score._unwrapped_model()
            _bt_d_term(stub)[0].backward()
            g[name] = _bt_grads(inner)
        for n, _p in _make_bt_stub(True).fake_score._unwrapped_model(
        ).head_parameters():
            self.assertTrue(
                torch.equal(g["true"][n], g["false"][n]),
                f"{n}: head gradient differs between the two arms",
            )

    def test_the_value_and_the_telemetry_are_unchanged(self):
        v_t, logs_t = _bt_d_term(_make_bt_stub(True))
        v_f, logs_f = _bt_d_term(_make_bt_stub(False))
        self.assertEqual(float(v_t.detach()), float(v_f.detach()))
        self.assertEqual(logs_f.pop("of_backbone_trainable"), 0.0)
        self.assertEqual(logs_t, logs_f)

    def test_the_checkpointed_disc_does_not_raise(self):
        """A ``requires_grad``-toggling implementation dies here with
        ``CheckpointError``; the surrogate never touches a flag."""
        stub = _make_bt_stub(False, ckpt=True)
        self.assertTrue(stub.fake_score.model.gradient_checkpointing)
        (_bt_denoise(stub) + _bt_d_term(stub)[0]).backward()

    def test_sequential_rebuilt_d_terms_all_backward(self):
        """``streaming_fake_updates_per_gen``: 1 + N critic backwards,
        no ``retain_graph``. The surrogate frees its own graph, so this
        is where an accidental double-traverse would show up."""
        stub = _make_bt_stub(False)
        inner = stub.fake_score._unwrapped_model()
        for i in range(5):
            _bt_zero(inner)
            term, _logs = _bt_d_term(stub, seed=1 + i)
            (_bt_denoise(stub) + term).backward()
            for n, p in inner.head_parameters():
                self.assertIsNotNone(p.grad, f"update {i}: {n} ungradiented")

    def test_it_composes_with_the_micro_batched_disc_forward(self):
        """The S6 memory knob (``gan_of_disc_micro_batch_groups``) splits
        the ``[fake ; real]`` batch across N forwards. The restriction is
        applied to the SUMMED term, so the two are orthogonal — checked
        here rather than assumed, because they touch the same method."""
        if "gan_of_disc_micro_batch_groups" not in OF_DEFAULTS:
            self.skipTest("S6 micro-batching is not in this tree")
        stub = _make_bt_stub(False)
        stub.of_cfg["gan_of_disc_micro_batch_groups"] = 2
        inner = stub.fake_score._unwrapped_model()
        (_bt_denoise(stub) + _bt_d_term(stub)[0]).backward()
        for n, p in inner.head_parameters():
            self.assertIsNotNone(p.grad, n)
        head = sum(
            float(p.grad.abs().sum()) for _n, p in inner.head_parameters()
        )
        self.assertGreater(head, 0.0)
        # ...and the backbone still carries the DENOISING gradient only.
        _bt_zero(inner)
        _bt_d_term(stub)[0].backward()
        for n, p in inner.backbone_parameters():
            self.assertTrue(
                p.grad is None or float(p.grad.abs().sum()) == 0.0,
                f"{n} was adversarially trained under micro-batching",
            )

    def test_the_inactive_weight_path_is_already_head_only(self):
        """``gan_of_d_weight=0`` / ``disc_start_step>0`` degrade the term
        to ``of_head_touch()``, which edges the head and nothing else —
        so the restriction is a no-op there, and the gauge must still be
        published or its trace goes sparse across the warmup."""
        stub = _make_bt_stub(False)
        stub.of_cfg["gan_of_d_weight"] = 0.0
        inner = stub.fake_score._unwrapped_model()
        term, logs = _bt_d_term(stub)
        self.assertIsNotNone(term)
        self.assertEqual(logs["of_backbone_trainable"], 0.0)
        term.backward()
        for n, p in inner.head_parameters():
            self.assertIsNotNone(p.grad, n)
            self.assertTrue(torch.equal(p.grad, torch.zeros_like(p.grad)))
        for n, p in inner.backbone_parameters():
            self.assertIsNone(p.grad, n)

    def test_no_head_at_all_fails_loud(self):
        class _Bare(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([nn.Linear(4, 4)])

        class _W(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = _Bare()

            def _unwrapped_model(self):
                return self.model

        stub = _BTStub(dict(OF_DEFAULTS), _W())
        stub.of_cfg["gan_of_backbone_trainable"] = False
        for name in _BT_METHODS:
            _bind_model_method(stub, name)
        with self.assertRaises(RuntimeError) as cm:
            stub._of_head_only_surrogate(torch.zeros((), requires_grad=True))
        self.assertIn("nowhere to go", str(cm.exception))


class TestBackboneTrainableFalseKeepsTheCriticAlive(unittest.TestCase):
    """"Adversarial gradient to heads only", NOT "backbone frozen". A
    globally frozen ``fake_score`` would break the DMD critic, which is a
    different variable again."""

    def test_the_denoising_loss_still_gradients_every_backbone_parameter(self):
        stub = _make_bt_stub(False)
        inner = stub.fake_score._unwrapped_model()
        (_bt_denoise(stub) + _bt_d_term(stub)[0]).backward()
        for n, p in inner.backbone_parameters():
            self.assertIsNotNone(p.grad, f"{n} ungradiented — critic broken")
            self.assertGreater(float(p.grad.abs().sum()), 0.0, n)

    def test_the_backbone_gradient_equals_the_denoising_only_gradient(self):
        """The exact claim: the adversarial contribution to the backbone
        is ZERO, so the sum equals the denoising term alone."""
        stub = _make_bt_stub(False)
        inner = stub.fake_score._unwrapped_model()

        _bt_denoise(stub).backward()
        denoise_only = _bt_grads(inner)

        _bt_zero(inner)
        (_bt_denoise(stub) + _bt_d_term(stub)[0]).backward()
        both = _bt_grads(inner)

        for n, _p in inner.backbone_parameters():
            self.assertTrue(
                torch.equal(denoise_only[n], both[n]),
                f"{n}: the D term perturbed a backbone gradient",
            )

    def test_the_control_shows_the_faithful_arm_does_perturb_it(self):
        """MUTATION CONTROL for the test above."""
        stub = _make_bt_stub(True)
        inner = stub.fake_score._unwrapped_model()
        _bt_denoise(stub).backward()
        denoise_only = _bt_grads(inner)
        _bt_zero(inner)
        (_bt_denoise(stub) + _bt_d_term(stub)[0]).backward()
        both = _bt_grads(inner)
        self.assertTrue(
            any(
                not torch.equal(denoise_only[n], both[n])
                for n, _p in inner.backbone_parameters()
            ),
            "the faithful arm left the backbone gradient untouched — the "
            "harness is not exercising co-training at all",
        )

    def test_no_fake_score_parameter_requires_grad_flag_was_changed(self):
        """The flag must not be implemented by toggling ``requires_grad``
        (checkpoint replay reads those flags at BACKWARD time)."""
        stub = _make_bt_stub(False)
        inner = stub.fake_score._unwrapped_model()
        before = {n: p.requires_grad for n, p in inner.named_parameters()}
        (_bt_denoise(stub) + _bt_d_term(stub)[0]).backward()
        after = {n: p.requires_grad for n, p in inner.named_parameters()}
        self.assertEqual(before, after)
        self.assertTrue(all(after.values()))


class TestBackboneTrainableRankUniformity(unittest.TestCase):
    def test_the_two_arms_run_the_same_number_of_disc_forwards(self):
        counts = []
        for flag in (True, False):
            stub = _make_bt_stub(flag)
            (_bt_denoise(stub) + _bt_d_term(stub)[0]).backward()
            counts.append(stub.fake_score.forwards)
        self.assertEqual(counts[0], counts[1])

    def test_the_flag_draws_no_rng(self):
        """If the control consumed a different number of RNG values the
        ranks would silently desynchronise. Same seed in, same state out."""
        states = []
        for flag in (True, False):
            stub = _make_bt_stub(flag)
            torch.manual_seed(4242)
            _bt_d_term(stub)[0].backward()
            states.append(torch.random.get_rng_state().clone())
        self.assertTrue(torch.equal(states[0], states[1]))

    def test_the_gate_is_read_only_from_the_resolved_config(self):
        """No per-rank quantity can reach it: the only reader is
        ``self.of_cfg``."""
        import ast
        import inspect
        import textwrap
        from model.dmd_action_forcing import ActionForcingDMD
        src = textwrap.dedent(
            inspect.getsource(ActionForcingDMD.compute_of_d_loss)
        )
        doc = ast.get_docstring(ast.parse(src).body[0]) or ""
        prose = {ln.strip() for ln in doc.splitlines()}
        gate = [
            ln for ln in src.splitlines()
            if "gan_of_backbone_trainable" in ln
            and not ln.strip().startswith("#")
            and ln.strip() not in prose
        ]
        self.assertEqual(len(gate), 1, gate)
        self.assertIn("self.of_cfg", gate[0])


class TestBackboneTrainableUnderDDP(unittest.TestCase):
    """The reducer must be satisfied in BOTH arms.

    Under ``gan_of_backbone_trainable=false`` the head is gradiented by a
    surrogate rather than by the classify forward, and the head-only
    ``torch.autograd.grad`` runs BEFORE the caller's backward. If that call
    fired DDP's ``AccumulateGrad`` hooks, the head would be marked ready
    twice and the reducer would raise. It does not — proven here rather
    than argued.
    """

    def setUp(self):
        import tempfile
        import torch.distributed as dist
        if not dist.is_available() or not dist.is_gloo_available():
            self.skipTest("gloo unavailable")
        self._tmp = tempfile.mkdtemp()
        if dist.is_initialized():
            self.skipTest("a process group is already initialised")
        dist.init_process_group(
            backend="gloo",
            store=dist.FileStore(os.path.join(self._tmp, "store"), 1),
            rank=0, world_size=1,
        )

    def tearDown(self):
        import shutil
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _run(self, trainable, iters=3):
        from torch.nn.parallel import DistributedDataParallel as DDP
        stub = _make_bt_stub(trainable, ckpt=False)
        inner = stub.fake_score.model
        stub.fake_score.model = DDP(inner, find_unused_parameters=False)
        for i in range(iters):
            loss = _bt_denoise(stub, seed=7 + i)
            term, _logs = _bt_d_term(stub, seed=1 + i)
            (loss + term).backward()
            for n, p in inner.named_parameters():
                self.assertIsNotNone(p.grad, f"iter {i}: {n} ungradiented")
                p.grad = None

    def test_the_faithful_arm_keeps_the_reducer_happy(self):
        self._run(True)

    def test_the_control_arm_keeps_the_reducer_happy(self):
        self._run(False)


class TestBackboneTrainableLeavesTheGSideAlone(unittest.TestCase):
    """The G side must NOT be collateral damage. It needs gradient to
    flow to the INPUT latent — a different path from the backbone
    PARAMS — and a ``no_grad``/tap-detach implementation would kill it."""

    def _g_stub(self, trainable):
        cfg = dict(OF_DEFAULTS)
        cfg["gan_of_g_weight"] = 0.03
        cfg["gan_of_backbone_trainable"] = bool(trainable)
        torch.manual_seed(0)
        disc = _BTDiscWrapper(dim=8, n_blocks=3)
        disc.model.gradient_checkpointing = True
        stub = _OFGLossStub(cfg, disc)
        for name in ("compute_of_g_loss", "_of_fake_sample",
                     "_of_real_sample", "_of_sample_timestep",
                     "_of_disc_cond", "_of_disc_frozen", "_of_disc_logits",
                     "_of_g_grad", "_of_g_surrogate", "_of_head_parameters",
                     "_of_head_only_surrogate", "of_head_touch"):
            _bind_model_method(stub, name)
        return stub

    def _run(self, trainable):
        stub = self._g_stub(trainable)
        torch.manual_seed(3)
        gen = _TinyGenerator(c=8)
        pred = gen(torch.randn(1, 4, 8, 1, 1))
        real = torch.randn(1, 4, 8, 1, 1)
        torch.manual_seed(11)
        term, fake, logs = stub.compute_of_g_loss(
            pred_image=pred, real_latent=real, cond_for_scoring={},
            current_step=10,
        )
        return stub, gen, pred, term, fake, logs

    def test_the_adversarial_gradient_still_reaches_the_generator(self):
        _stub, gen, _pred, term, _fake, _logs = self._run(False)
        self.assertIsNotNone(term)
        self.assertTrue(term.requires_grad)
        g = torch.autograd.grad(term, gen.proj.weight)[0]
        self.assertIsNotNone(g)
        self.assertTrue(torch.isfinite(g).all())
        self.assertGreater(
            float(g.abs().sum()), 0.0,
            "gan_of_backbone_trainable=false severed the GEN-side "
            "adversarial gradient — the exact collateral damage a "
            "no_grad or tap-detach implementation causes",
        )

    def test_it_reaches_the_generator_through_the_fake_tensor(self):
        _stub, _gen, pred, term, _fake, _logs = self._run(False)
        g = torch.autograd.grad(term, pred, retain_graph=True)[0]
        self.assertGreater(float(g.abs().sum()), 0.0)

    def test_the_g_side_is_numerically_identical_in_both_arms(self):
        vals = []
        for flag in (True, False):
            _s, gen, _p, term, _f, logs = self._run(flag)
            vals.append((
                float(term.detach()),
                float(torch.autograd.grad(term, gen.proj.weight)[0].sum()),
                logs.get("of_g_loss"),
            ))
        self.assertEqual(vals[0], vals[1])

    def test_the_g_side_still_moves_no_fake_score_weight_in_either_arm(self):
        for flag in (True, False):
            stub, gen, _p, term, _f, _l = self._run(flag)
            term.backward()
            for n, p in stub.fake_score._unwrapped_model().named_parameters():
                self.assertTrue(
                    p.grad is None or float(p.grad.abs().sum()) == 0.0,
                    f"trainable={flag}: G moved fake_score parameter {n}",
                )


# =====================================================================
# S6 — DISC-FORWARD MEMORY (2026-08-25)
# =====================================================================
# WHAT S6 IS. The D step pushes ``cat([noisy_fake, noisy_real])`` through
# ONE classify forward on the trainable 1.3B fake_score, and there are
# ``1 + streaming_fake_updates_per_gen`` = 5 such forward+backwards per
# active step. The measured resident cost of one of them, at the shipped
# geometry (band = 9 frames x 1561 tokens, dim 1536, 30 blocks, taps
# [21, 29], bf16, fake_score_gradient_checkpointing=true):
#
#   DiT checkpoint boundaries  30 x [2, 14049, 1536] bf16   ~2.41 GiB
#   tap blocks (NOT checkpointed, 2 taps)                   ~1.18 GiB
#   head + unpatchify (result DISCARDED)                    ~0.24 GiB
#                                                          ---------
#                                                           ~3.83 GiB
#
# The tap and head terms are the addressable ones, and they are what the
# two model-side flags below remove. Note the DiT backbone does NOT carry
# a requires_grad-dependent saved-INPUT term: ``use_reentrant=False``
# checkpointing discards block internals entirely, which
# ``TestCheckpointHidesSavedInputTerm`` demonstrates directly.


class TestDiscMicroBatchBounds(unittest.TestCase):
    """``disc_micro_batch_bounds`` — the rank-uniformity guarantee.

    The group count decides how many disc forwards run before the SINGLE
    shared ``critic_loss.backward()``. If two ranks disagreed on that
    count, the DDP-wrapped ``fake_score`` reducer desynchronises and the
    job hangs — a multi-node-only failure. These tests pin the property
    that makes disagreement impossible: the bounds are a pure function of
    ``(n_rows, groups)`` and read no state at all.
    """

    def test_default_groups_is_one_whole_batch(self):
        self.assertEqual(disc_micro_batch_bounds(2, 1), [(0, 2)])
        self.assertEqual(disc_micro_batch_bounds(8, 1), [(0, 8)])

    def test_partition_is_exact_and_non_empty(self):
        for n_rows in range(1, 17):
            for g in range(1, 20):
                b = disc_micro_batch_bounds(n_rows, g)
                self.assertEqual(b[0][0], 0)
                self.assertEqual(b[-1][1], n_rows)
                for (_, hi), (lo2, _) in zip(b, b[1:]):
                    self.assertEqual(hi, lo2, "groups must be contiguous")
                self.assertTrue(all(hi > lo for lo, hi in b),
                                "no group may be empty")
                self.assertEqual(sum(hi - lo for lo, hi in b), n_rows)

    def test_groups_clamped_to_row_count(self):
        """More groups than rows => one row each, never an empty forward."""
        self.assertEqual(len(disc_micro_batch_bounds(2, 9)), 2)
        self.assertEqual(disc_micro_batch_bounds(2, 9), [(0, 1), (1, 2)])

    def test_is_a_pure_function(self):
        """Same inputs => same bounds, always. This is the rank-uniformity
        argument in executable form: nothing else is read."""
        for _ in range(5):
            self.assertEqual(disc_micro_batch_bounds(6, 4),
                             [(0, 1), (1, 3), (3, 4), (4, 6)])

    def test_rejects_nonpositive_rows(self):
        with self.assertRaises(ValueError):
            disc_micro_batch_bounds(0, 2)


class TestSliceConditionalDictRows(unittest.TestCase):
    """Row-slicing the duplicated cond dict must be the exact inverse of
    the concatenation the single-batch path performs."""

    def test_reassembles_the_duplicated_cond(self):
        cond = {
            "prompt_embeds": torch.randn(3, 5, 7),
            "_action_tokens": torch.randn(3, 4),
            "scalar": 1.5,
            "not_batched": torch.randn(9, 2),   # dim0 != n_rows => passthrough
        }
        dup = duplicate_conditional_dict(cond)
        n_rows = 6
        parts = [slice_conditional_dict_rows(dup, lo, hi, n_rows)
                 for lo, hi in disc_micro_batch_bounds(n_rows, 3)]
        for key in ("prompt_embeds", "_action_tokens"):
            torch.testing.assert_close(
                torch.cat([p[key] for p in parts], dim=0), dup[key],
                rtol=0, atol=0,
            )
        # Entries whose dim 0 is not ``n_rows`` are passed through
        # untouched on every group (here: a 9-row tensor duplicated to 18,
        # which is neither 6 nor a row-aligned slice).
        for p in parts:
            self.assertEqual(p["scalar"], 1.5)
            torch.testing.assert_close(p["not_batched"], dup["not_batched"],
                                       rtol=0, atol=0)


def _micro_stub(groups, dim=8, n_blocks=3):
    """A ``_of_disc_logits`` host with the micro-batch knob set."""
    cfg = dict(OF_DEFAULTS)
    cfg["gan_of_disc_micro_batch_groups"] = int(groups)
    disc = _StubDiscWrapper(dim=dim, n_blocks=n_blocks)
    disc.model.gradient_checkpointing = False
    stub = types.SimpleNamespace(of_cfg=cfg, fake_score=disc)
    _bind_model_method(stub, "_of_disc_logits")
    return stub


class TestDiscMicroBatchValueEquivalence(unittest.TestCase):
    """Micro-batching must not move the number.

    This is the property ``ladd_disc_micro_batch_groups`` has to work for
    (its per-group loss is scaled by ``/N_total``, not ``/N_group``). The
    OF path gets it for free by concatenating LOGITS and taking the loss
    once, and these tests are what turn "for free" into a fact.
    """

    def _run(self, groups, rows=4, seed=0):
        torch.manual_seed(seed)
        stub = _micro_stub(groups)
        torch.manual_seed(seed + 1)
        latent = torch.randn(rows, 2, 4, 2, 2)
        cond = {"cond_feat": torch.randn(rows, 4, 8)}
        t = torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2)
        return stub._of_disc_logits(latent=latent, cond=cond, timestep=t)

    def test_logits_identical_across_group_counts(self):
        base = self._run(1)
        for g in (2, 3, 4, 7):
            torch.testing.assert_close(
                self._run(g), base, rtol=0, atol=0,
                msg=f"groups={g} moved the logits",
            )

    def test_shape_preserved(self):
        self.assertEqual(tuple(self._run(1).shape), tuple(self._run(4).shape))

    def test_groups_one_issues_exactly_one_forward(self):
        """Byte-identity at the default is a code-PATH claim, not just a
        numeric one: groups=1 must take the verbatim single call."""
        for groups, expect in ((1, 1), (2, 2), (4, 4)):
            stub = _micro_stub(groups)
            calls = []
            inner = stub.fake_score.forward

            def counting(*a, _inner=inner, **kw):
                calls.append(kw.get("noisy_image_or_video").shape[0])
                return _inner(*a, **kw)

            stub.fake_score.forward = counting
            stub._of_disc_logits(
                latent=torch.randn(4, 2, 4, 2, 2),
                cond={"cond_feat": torch.randn(4, 4, 8)},
                timestep=torch.zeros(4, 2),
            )
            self.assertEqual(len(calls), expect,
                             f"groups={groups} issued {len(calls)} forwards")
            self.assertEqual(sum(calls), 4, "every row must be forwarded once")

    def test_single_row_batch_never_splits(self):
        """A 1-row forward (the G side at B=1) must stay a single call even
        with a large group count — a zero-row forward would be a shape
        error on some ranks and not others."""
        stub = _micro_stub(8)
        calls = []
        inner = stub.fake_score.forward
        stub.fake_score.forward = lambda *a, **kw: (
            calls.append(1) or inner(*a, **kw))
        stub._of_disc_logits(
            latent=torch.randn(1, 2, 4, 2, 2),
            cond={"cond_feat": torch.randn(1, 4, 8)},
            timestep=torch.zeros(1, 2),
        )
        self.assertEqual(len(calls), 1)

    def test_mutation_control_unsliced_cond_is_detected(self):
        """MUTATION CONTROL. Revert ONLY the cond row-slicing (hand every
        group the full-batch cond) and the equivalence assertion above must
        FAIL. Without this, a test that never varies the conditioning
        across rows would pass with the slicing removed entirely, and the
        disc would silently score every group against row 0's actions."""
        import model.dmd_action_forcing as dmd

        base = self._run(1)
        orig = dmd.slice_conditional_dict_rows
        try:
            dmd.slice_conditional_dict_rows = (
                lambda cond, lo, hi, n_rows: cond
            )
            with self.assertRaises(Exception):
                # Either a shape error or a value mismatch — both are the
                # test noticing. Anything that passes here means the
                # equivalence test above is vacuous.
                torch.testing.assert_close(
                    self._run(2), base, rtol=0, atol=0,
                )
        finally:
            dmd.slice_conditional_dict_rows = orig


class TestCheckpointHidesSavedInputTerm(unittest.TestCase):
    """``use_reentrant=False`` checkpointing removes the
    requires_grad-dependent saved-INPUT term entirely.

    WHY THIS TEST EXISTS. ``F.linear`` saves its INPUT only when the
    weight requires grad, so a disc forward on the TRAINABLE fake_score
    was expected to retain one extra full-sequence activation per linear
    per block versus the frozen real_score — which, across 30 blocks,
    would dominate everything else and was the stated reason to fear an
    OOM. It does not apply to our path, because the OF disc forward runs
    with ``fake_score_gradient_checkpointing=true`` and a checkpointed
    region saves NOTHING but its boundary input. This test pins that, so
    the reasoning behind the S6 sizing cannot silently rot.
    """

    def _saved_bytes(self, trainable, ckpt, n_blocks=6, L=64, dim=32):
        seen = {}

        def pack(t):
            if isinstance(t, torch.Tensor):
                seen[t.data_ptr()] = t.numel() * t.element_size()
            return t

        torch.manual_seed(0)
        blocks = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_blocks)])
        for p in blocks.parameters():
            p.requires_grad_(trainable)
        x = torch.randn(2, L, dim, requires_grad=True)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            h = x
            for b in blocks:
                h = (torch.utils.checkpoint.checkpoint(
                    b, h, use_reentrant=False) if ckpt else b(h))
        return sum(seen.values())

    def test_uncheckpointed_trainable_costs_more_than_frozen(self):
        """The mechanism is real when nothing is checkpointed."""
        self.assertGreater(
            self._saved_bytes(trainable=True, ckpt=False),
            self._saved_bytes(trainable=False, ckpt=False),
        )

    def test_checkpointing_erases_the_difference(self):
        """...and it vanishes under the checkpointing our disc forward
        actually runs with. This is why S6 does NOT size the backbone term
        off ``requires_grad``."""
        self.assertEqual(
            self._saved_bytes(trainable=True, ckpt=True),
            self._saved_bytes(trainable=False, ckpt=True),
        )


class TestTapCheckpointEquivalence(unittest.TestCase):
    """``gan_of_checkpoint_taps`` must be value- and gradient-neutral.

    Runs the REAL ``GanAttentionBlock`` (the module the tap loop calls),
    with ``flash_attention`` swapped for SDPA because flash has no CPU
    path. What is under test is the transformation the two tap loops
    apply — wrapping the per-tap block stack in
    ``checkpoint(use_reentrant=False)`` — not a re-implementation of it.
    """

    def _blocks(self, dim=32, n=2):
        import wan.modules.model as wm
        torch.manual_seed(0)
        return nn.ModuleList([
            wm.GanAttentionBlock(dim=dim, ffn_dim=64, num_heads=4)
            for _ in range(n)
        ])

    def _run(self, use_ckpt, dim=32, L=16):
        import torch.nn.functional as F
        import wan.modules.model as wm

        def _sdpa(q, k, v, *a, **kw):
            return F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            ).transpose(1, 2)

        prev = wm.flash_attention
        wm.flash_attention = _sdpa
        try:
            blocks = self._blocks(dim=dim)
            torch.manual_seed(1)
            x = torch.randn(2, L, dim, requires_grad=True)
            tok = torch.randn(2, 1, dim, requires_grad=True)
            if use_ckpt:
                def _tap_fn(_x, _t, _blocks=blocks):
                    for _c in _blocks:
                        _t = _c(_x, _t)
                    return _t
                out = torch.utils.checkpoint.checkpoint(
                    _tap_fn, x, tok, use_reentrant=False)
            else:
                out = tok
                for c in blocks:
                    out = c(x, out)
            out.sum().backward()
            return out.detach(), x.grad.clone(), [
                p.grad.clone() for p in blocks.parameters()
            ]
        finally:
            wm.flash_attention = prev

    def test_output_and_gradients_match(self):
        o0, gx0, gp0 = self._run(False)
        o1, gx1, gp1 = self._run(True)
        torch.testing.assert_close(o1, o0, rtol=0, atol=0)
        torch.testing.assert_close(gx1, gx0, rtol=1e-6, atol=1e-6)
        self.assertEqual(len(gp0), len(gp1))
        for a, b in zip(gp0, gp1):
            torch.testing.assert_close(b, a, rtol=1e-6, atol=1e-6)

    def test_checkpointed_tap_retains_less(self):
        """The point of the flag: the tap stack's activations stop being
        resident for the whole forward."""
        import torch.nn.functional as F
        import wan.modules.model as wm

        def _sdpa(q, k, v, *a, **kw):
            return F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            ).transpose(1, 2)

        def saved(use_ckpt):
            seen = {}

            def pack(t):
                if isinstance(t, torch.Tensor):
                    seen[t.data_ptr()] = t.numel() * t.element_size()
                return t
            prev = wm.flash_attention
            wm.flash_attention = _sdpa
            try:
                blocks = self._blocks()
                torch.manual_seed(1)
                x = torch.randn(2, 64, 32, requires_grad=True)
                tok = torch.randn(2, 1, 32, requires_grad=True)
                with torch.autograd.graph.saved_tensors_hooks(
                        pack, lambda t: t):
                    if use_ckpt:
                        def _tap_fn(_x, _t, _b=blocks):
                            for _c in _b:
                                _t = _c(_x, _t)
                            return _t
                        torch.utils.checkpoint.checkpoint(
                            _tap_fn, x, tok, use_reentrant=False)
                    else:
                        out = tok
                        for c in blocks:
                            out = c(x, out)
                return sum(seen.values())
            finally:
                wm.flash_attention = prev

        self.assertLess(saved(True), saved(False))


class TestClassifySkipHeadShapeContract(unittest.TestCase):
    """The skipped head's ZERO PLACEHOLDER must have exactly the shape
    ``head + unpatchify`` would have produced.

    Only the shape can break: the logits come from the tap stack, which
    the skip branch does not touch, and the head's output is discarded by
    every classify caller (which is why ``head.modulation`` /
    ``head.head.*`` were ALREADY ungradiented on a classify-only
    backward). So this test checks the one contract at risk, against the
    REAL ``unpatchify``.
    """

    def _real_unpatchify_shape(self, grid, patch, out_dim):
        import wan.modules.model as wm
        stub = types.SimpleNamespace(
            out_dim=out_dim, patch_size=patch,
            unpatchify=None,
        )
        stub.unpatchify = types.MethodType(wm.WanModel.unpatchify, stub)
        gs = torch.tensor([list(grid)])
        n = 1
        for v in grid:
            n *= v
        head_out = torch.zeros(1, n, out_dim * (patch[0] * patch[1] * patch[2]))
        return tuple(torch.stack(stub.unpatchify(head_out, gs)).shape)

    def _placeholder_shape(self, grid, patch, out_dim):
        """The expression the shipped skip branch uses."""
        shape = [i * j for i, j in zip(list(grid), list(patch))]
        return tuple(torch.zeros((1, out_dim, *shape)).shape)

    def test_placeholder_matches_unpatchify(self):
        for grid, patch, out_dim in (
            ((9, 30, 52), (1, 2, 2), 16),     # the shipped OF band
            ((21, 30, 52), (1, 2, 2), 16),    # full scoring window
            ((3, 4, 6), (1, 2, 2), 8),        # small
            ((2, 4, 4), (2, 2, 2), 4),        # non-unit temporal patch
        ):
            self.assertEqual(
                self._placeholder_shape(grid, patch, out_dim),
                self._real_unpatchify_shape(grid, patch, out_dim),
                f"placeholder shape diverged for grid={grid} patch={patch}",
            )

    def test_mutation_control_wrong_patch_is_detected(self):
        """MUTATION CONTROL: drop the patch-size multiply (the obvious way
        to write this branch wrong) and the shapes must disagree."""
        grid, patch, out_dim = (9, 30, 52), (1, 2, 2), 16
        wrong = tuple(torch.zeros((1, out_dim, *grid)).shape)
        self.assertNotEqual(
            wrong, self._real_unpatchify_shape(grid, patch, out_dim))


class TestS6FlagsDefaultOffAndPlumbed(unittest.TestCase):
    """Every S6 knob defaults to the pre-S6 behaviour, and reaches the
    object that acts on it."""

    def test_defaults_are_inert(self):
        cfg = resolve_of_config(types.SimpleNamespace())
        self.assertEqual(cfg["gan_of_disc_micro_batch_groups"], 1)
        self.assertFalse(cfg["gan_of_checkpoint_taps"])
        self.assertFalse(cfg["gan_of_classify_skip_head"])
        self.assertEqual(cfg["gan_of_r1_num_samples"], 0)

    def test_registered_for_the_override_guard(self):
        from model.one_forcing_gan import CONFIG_KEYS
        for k in ("gan_of_disc_micro_batch_groups", "gan_of_checkpoint_taps",
                  "gan_of_classify_skip_head", "gan_of_r1_num_samples"):
            self.assertIn(k, CONFIG_KEYS)

    def test_shipped_config_declares_them(self):
        import yaml
        with open(os.path.join(_ROOT, "configs",
                               "action_forcing_phase3_dmd.yaml")) as fh:
            y = yaml.safe_load(fh)
        self.assertEqual(y["gan_of_disc_micro_batch_groups"], 1)
        self.assertIs(y["gan_of_checkpoint_taps"], False)
        self.assertIs(y["gan_of_classify_skip_head"], False)
        self.assertEqual(y["gan_of_r1_num_samples"], 0)

    def test_invalid_group_count_refused(self):
        cfg = dict(OF_DEFAULTS)
        cfg["gan_of_enabled"] = True
        cfg["gan_of_disc_micro_batch_groups"] = 0
        with self.assertRaises(ValueError):
            validate_of_config(cfg)

    def test_adding_cls_branch_writes_the_model_attrs(self):
        """The two model-side flags must land on the object the tap/head
        loops see as ``self`` — the inner model, not the wrapper."""
        for ckpt_taps, skip_head in ((False, False), (True, False),
                                     (False, True), (True, True)):
            w, inner = _make_stub_wrapper()
            w.adding_cls_branch(
                atten_dim=32, num_class=1, hidden_dim=32, num_layers=1,
                dropout=0.0, gan_blocks_per_token=1, layer_indices=[1, 3],
                block_ffn_dim=16, block_num_heads=4, attach_to_model=True,
                checkpoint_taps=ckpt_taps, classify_skip_head=skip_head,
            )
            self.assertIs(inner._gan_checkpoint_taps, ckpt_taps)
            self.assertIs(inner._gan_classify_skip_head, skip_head)

    def test_adding_cls_branch_defaults_leave_attrs_off(self):
        """A caller that predates S6 (the legacy dmd2* users) must get the
        historical behaviour without passing anything."""
        w, inner = _make_stub_wrapper()
        w.adding_cls_branch(
            atten_dim=32, num_class=1, hidden_dim=32, num_layers=1,
            dropout=0.0, gan_blocks_per_token=1, layer_indices=[1, 3],
            block_ffn_dim=16, block_num_heads=4, attach_to_model=True,
        )
        self.assertFalse(inner._gan_checkpoint_taps)
        self.assertFalse(inner._gan_classify_skip_head)
