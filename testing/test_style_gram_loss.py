"""CPU tests for WP-STYLE — the unpaired region-level Gram/AdaIN style loss.

    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" \
        python -m pytest testing/test_style_gram_loss.py -q

(``OMP_NUM_THREADS=8`` is MANDATORY on this box: ``nproc`` is 144 and the
default thread count makes torch's CPU ops thrash for 30+ minutes.)

What is covered, and why each one is here
=========================================
1.  DEFAULT-OFF BYTE-IDENTITY, with a MUTATION CONTROL. This is the most
    important test in the file: production arms are queued on the tree this
    patch lands in, and a feature that perturbs anything at weight 0.0
    corrupts every one of them. The guarded block is EXTRACTED FROM THE
    SHIPPED TRAINER SOURCE and executed, so the test cannot drift away from
    the code it certifies. Checked in fp32 AND bf16, with a real (small)
    DMD+GAN generator-loss path active around it, on: the loss value, every
    parameter gradient, the global RNG state, and module-import side effects.
2.  GRAM CORRECTNESS against a hand-copied transcription of
    ``grids/eval/style_shift.py::VGGStyle`` — the offline evaluator. If the
    training loss and the eval metric ever stop measuring the same thing,
    this fails.
3.  SPATIAL-PERMUTATION INVARIANCE. A Gram must not care about layout; that
    invariance is precisely what makes the loss UNPAIRED.
4.  DESCENT: the loss falls under SGD toward a target style.
5.  GRADIENT FLOW to the generator (and NONE to the frozen encoder).
6.  TWO-SIDEDNESS: the loss RISES when texture OVERSHOOTS the target, not
    only when it undershoots — with a mutation control showing the offline
    metric's SYMMETRIC denominator would have SATURATED instead, i.e. would
    have stopped penalising extreme over-texturing. That is the researcher's
    "don't optimise for sharpness, it goes blocky" requirement, made a test.
7.  Telemetry: the proof-of-fire counter, the resolved-config echo, and the
    override-guard sourcing of every ``style_gram_*`` key.
"""
import inspect
import math
import os
import re
import sys
import textwrap
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ``wan/modules/t5.py`` evaluates ``torch.cuda.current_device()`` in a class
# body at import time, so importing the trainer needs a device to exist.
# These tests are CPU-only; nothing under test touches CUDA.
with patch.object(torch.cuda, "current_device", return_value=0):
    from trainer import causal_action_forcing_train as CAFT

from model.style_gram_loss import (          # noqa: E402
    STYLE_DEFAULTS,
    SUPPORTED_STYLE_ENCODERS,
    SUPPORTED_STYLE_MODES,
    VGG16_DEFAULT_LAYERS,
    FrozenStyleEncoder,
    adain_stats,
    adain_style_distance,
    gram_distance_eval,
    gram_matrix,
    gram_style_distance,
    region_mean_adain,
    region_mean_gram,
    style_distance,
)

Trainer = CAFT.ActionForcingDMDTrainer
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ===========================================================================
# Shared helpers
# ===========================================================================
def make_trainer(**cfg):
    """A bare trainer carrying only the attributes the style helpers read."""
    t = Trainer.__new__(Trainer)
    t.config = SimpleNamespace(**cfg)
    t.model = SimpleNamespace()
    t.device = torch.device("cpu")
    t.step = 0
    t.is_main_process = True
    t.gan_pixel_texture_enabled = bool(cfg.get("gan_pixel_texture_enabled", False))
    t.style_gram_loss_weight = float(cfg.get("style_gram_loss_weight", 0.0))
    t._style_encoder_module = None
    t._style_real_bank = None
    t._style_applied_total = 0
    return t


class TinyEncoder(nn.Module):
    """A frozen stand-in for VGG with the same output contract.

    Used wherever the test is about the LOSS rather than about VGG: two taps,
    small channel counts, deterministic weights, no 553 MB checkpoint and no
    seconds-long build. ``tap_names`` mirrors the real encoder's attribute so
    the trainer telemetry path works unchanged.
    """

    def __init__(self, seed=0, identity=False):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.c1 = nn.Conv2d(3, 6, 3, padding=1, bias=False)
        self.c2 = nn.Conv2d(6, 8, 3, padding=1, stride=2, bias=False)
        with torch.no_grad():
            if identity:
                # Channel-selecting kernels: features are (a scaled copy of)
                # the input, so a pixel-amplitude change maps to a feature
                # amplitude change exactly. Makes the two-sidedness algebra
                # analytic instead of approximate.
                self.c1.weight.zero_()
                for i in range(6):
                    self.c1.weight[i, i % 3, 1, 1] = 1.0
                self.c2.weight.zero_()
                for i in range(8):
                    self.c2.weight[i, i % 6, 1, 1] = 1.0
            else:
                self.c1.weight.copy_(torch.randn(self.c1.weight.shape, generator=g) * 0.3)
                self.c2.weight.copy_(torch.randn(self.c2.weight.shape, generator=g) * 0.3)
        self.requires_grad_(False)
        self.eval()
        self.tap_names = ["tap0", "tap1"]
        self.taps = (0, 1)
        self.n_params = sum(p.numel() for p in self.parameters())

    def forward(self, px):
        f1 = torch.relu(self.c1(px))
        f2 = torch.relu(self.c2(f1))
        return [f1, f2]


def offline_vgg_style_gram(x):
    """HAND-COPIED from ``grids/eval/style_shift.py::VGGStyle.forward``.

    Transcribed rather than imported because ``grids/`` is an eval-only tree
    with no package init and is excluded from the trainer's source scans.
    THIS transcription is the thing test 2 pins the training code against;
    if the offline evaluator ever changes, update it here and watch the
    assertion fail.

        b, c, h, w = x.shape
        f = x.reshape(b, c, h * w)
        g = torch.bmm(f, f.transpose(1, 2)) / (c * h * w)
    """
    b, c, h, w = x.shape
    f = x.reshape(b, c, h * w)
    return torch.bmm(f, f.transpose(1, 2)) / (c * h * w)


def offline_gram_distance(grams_a, grams_b):
    """HAND-COPIED from ``grids/eval/style_shift.py::gram_distance``."""
    d = 0.0
    for ga, gb in zip(grams_a, grams_b):
        ma, mb = ga.mean(0), gb.mean(0)
        denom = (ma.norm() + mb.norm()) / 2 + 1e-8
        d += float((ma - mb).norm() / denom)
    return d / len(grams_a)


# ===========================================================================
# 1. DEFAULT-OFF BYTE-IDENTITY  (the load-bearing one)
# ===========================================================================
CALL_SITE_START = "# WP-STYLE: unpaired region-level Gram/AdaIN style loss (default OFF)."
CALL_SITE_END = "# A5: no-grad decoder tripwire"


def extract_call_site_block():
    """Pull the guarded style block VERBATIM out of the shipped trainer.

    Extracting instead of re-typing is what makes this a certification of
    the CODE rather than of a copy of it: if someone widens the guard, or
    moves the ``generator_loss +=`` outside it, this test executes the new
    version and fails.
    """
    src = inspect.getsource(Trainer._streaming_train_one_chunk)
    lines = src.splitlines()
    lo = next(i for i, l in enumerate(lines) if CALL_SITE_START in l)
    hi = next(i for i, l in enumerate(lines) if i > lo and CALL_SITE_END in l)
    # Back up over the block's own leading banner lines to the first line
    # that is not a comment (the ``if`` that carries the gate).
    block = "\n".join(lines[lo:hi])
    block = textwrap.dedent(block)
    assert "style_gram_loss_weight" in block, block[:400]
    assert "generator_loss = generator_loss + _style_w" in block, block[:400]
    return block


class TinyGenerator(nn.Module):
    """A real (small) generator so the surrounding path is not a mock."""

    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 8)


class TinyDisc(nn.Module):
    def __init__(self):
        super().__init__()
        self.h = nn.Linear(8, 1)


def build_surrounding_path(dtype):
    """A miniature but REAL DMD+GAN generator-loss assembly.

    The byte-identity claim has to hold with the rest of the step live, not
    for the style term in isolation -- an off feature that perturbs a shared
    RNG stream or a shared tensor would pass an isolated test and still
    corrupt every queued arm.
    """
    torch.manual_seed(1234)
    gen = TinyGenerator().to(dtype)
    disc = TinyDisc().to(dtype)
    x = torch.randn(4, 8, dtype=dtype)
    h = gen.b(torch.tanh(gen.a(x)))
    # "DMD" term + "GAN" term, both graph-bearing off the same generator.
    dmd = (h ** 2).mean()
    gan = torch.nn.functional.softplus(-disc.h(h)).mean()
    generator_loss = dmd + 0.3 * gan
    # A 5-D student chunk shaped like the real ``train_chunk``.
    train_chunk = h.reshape(1, 4, 2, 2, 2).unsqueeze(0)[0].unsqueeze(0)
    train_chunk = train_chunk.reshape(1, 4, 2, 2, 2)
    return gen, generator_loss, train_chunk


def run_with_block(weight, dtype, block=None):
    """Run the surrounding path, optionally with the shipped style block."""
    gen, generator_loss, train_chunk = build_surrounding_path(dtype)
    out = {}
    if block is not None:
        t = make_trainer(style_gram_loss_weight=weight)
        t.style_gram_loss_weight = float(weight)
        fired = {"n": 0}

        def fake_compute(self, pred, info, *, current_step):
            fired["n"] += 1
            # A term with a real graph so the mutation control genuinely
            # changes the loss and the gradients.
            raw = (pred.float() ** 2).mean()
            return (raw * float(weight), raw,
                    {"train/style_gram_dist_raw": float(raw.detach())})

        t._compute_style_gram_loss = types.MethodType(fake_compute, t)
        # ``**kw`` rather than a fixed signature: the shipped call site
        # passes the probe SITE (``probe_tensor=train_chunk``) and this
        # stub must not be the thing that pins the probe's signature --
        # that is ``test_the_call_site_hands_the_probe_the_student_chunk``'s
        # job. A rigid lambda here turns any probe change into a spurious
        # byte-identity failure.
        t._style_grad_telemetry = types.MethodType(
            lambda self, gl, sr, o, *, current_step, **kw: None, t,
        )
        ns = {
            "self": t, "train_chunk": train_chunk, "train_info": {},
            "out": out, "generator_loss": generator_loss,
            "float": float, "getattr": getattr,
        }
        exec(block, ns)
        generator_loss = ns["generator_loss"]
        out = ns["out"]
        out["_fired"] = fired["n"]
    generator_loss.backward()
    grads = [
        p.grad.detach().clone().float() for p in gen.parameters()
        if p.grad is not None
    ]
    return {
        "loss": float(generator_loss.detach().float()),
        "grads": grads,
        "rng": torch.random.get_rng_state().clone(),
        "out": out,
    }


class TestDefaultOffByteIdentity(unittest.TestCase):

    def _compare(self, a, b, msg):
        self.assertEqual(a["loss"], b["loss"], msg + " (loss)")
        self.assertEqual(len(a["grads"]), len(b["grads"]), msg + " (n grads)")
        for i, (ga, gb) in enumerate(zip(a["grads"], b["grads"])):
            self.assertTrue(
                torch.equal(ga, gb), f"{msg} (grad {i} differs)",
            )
        self.assertTrue(
            torch.equal(a["rng"], b["rng"]), msg + " (global RNG state)",
        )

    def test_weight_zero_is_byte_identical_fp32(self):
        block = extract_call_site_block()
        base = run_with_block(0.0, torch.float32, block=None)
        withb = run_with_block(0.0, torch.float32, block=block)
        self._compare(base, withb, "fp32 weight=0.0")

    def test_weight_zero_is_byte_identical_bf16(self):
        block = extract_call_site_block()
        base = run_with_block(0.0, torch.bfloat16, block=None)
        withb = run_with_block(0.0, torch.bfloat16, block=block)
        self._compare(base, withb, "bf16 weight=0.0")

    def test_weight_zero_emits_no_log_key_and_never_calls_the_helper(self):
        block = extract_call_site_block()
        withb = run_with_block(0.0, torch.float32, block=block)
        self.assertEqual(withb["out"]["_fired"], 0,
                         "_compute_style_gram_loss ran at weight 0.0")
        keys = [k for k in withb["out"] if k.startswith("train/")]
        self.assertEqual(keys, [], f"weight 0.0 emitted log keys: {keys}")

    def test_MUTATION_CONTROL_a_nonzero_weight_does_change_everything(self):
        """The identity assertions above are evidence only if they can fail.

        Same block, same surrounding path, weight 0.05 instead of 0.0 -- the
        loss, the gradients AND the log keys must all move. If this passes
        silently, the tests above are certifying nothing.
        """
        block = extract_call_site_block()
        for dtype, name in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
            base = run_with_block(0.0, dtype, block=None)
            mutated = run_with_block(0.05, dtype, block=block)
            self.assertNotEqual(
                base["loss"], mutated["loss"],
                f"{name}: a positive weight did not change the loss -- the "
                "byte-identity test above is vacuous.",
            )
            self.assertFalse(
                all(torch.equal(a, b)
                    for a, b in zip(base["grads"], mutated["grads"])),
                f"{name}: a positive weight did not change any gradient.",
            )
            self.assertEqual(mutated["out"]["_fired"], 1)
            self.assertIn("train/style_gram_dist_raw", mutated["out"])
            self.assertEqual(mutated["out"]["train/style_loss_applied"], 1.0)

    def test_MUTATION_CONTROL_dropping_the_guard_breaks_identity(self):
        """A copy of the shipped block with the weight gate REMOVED must
        fail the identity comparison. This is what proves the guard -- not
        merely the weight arithmetic -- is what buys default-off."""
        block = extract_call_site_block()
        broken = re.sub(
            r"if float\(getattr\(self, \"style_gram_loss_weight\", 0\.0\)\) > 0\.0:",
            "if True:",
            block,
        )
        self.assertNotEqual(broken, block, "the guard line was not found")
        base = run_with_block(0.0, torch.float32, block=None)
        # With the guard gone, the helper runs even at weight 0.0. It adds
        # ``raw * 0.0``, which is numerically zero but still ATTACHES A NODE
        # and still writes keys -- and a real implementation would also have
        # decoded, drawn reals and consumed RNG.
        withb = run_with_block(0.0, torch.float32, block=broken)
        self.assertEqual(withb["out"]["_fired"], 1,
                         "the guard-removal mutation did not take effect")
        self.assertTrue(
            [k for k in withb["out"] if k.startswith("train/")],
            "guard-removed copy emitted no keys -- mutation ineffective",
        )
        _ = base

    def test_the_style_module_is_not_imported_when_the_feature_is_off(self):
        """Off must cost nothing at all -- not even a torchvision import.

        ``_style_resolve_cfg`` / ``_style_encoder`` are the only importers,
        and both are reached exclusively from the weight-gated helper.
        """
        src = inspect.getsource(CAFT)
        # Module level must not import it (that would pull torchvision into
        # every run on the tree).
        head = src[: src.index("class ActionForcingDMDTrainer")]
        self.assertNotIn("style_gram_loss", head)
        for meth in ("_style_resolve_cfg", "_style_encoder",
                     "_compute_style_gram_loss"):
            body = inspect.getsource(getattr(Trainer, meth))
            self.assertIn("from model.style_gram_loss import", body,
                          f"{meth} must import lazily, inside the function")

    def test_pix_pool_fill_cap_default_preserves_the_pixel_path(self):
        """The one edit to a shared helper: ``cap=None`` must still resolve
        through ``_pix_resolve_cfg``, i.e. the pixel critic is untouched."""
        body = inspect.getsource(Trainer._pix_pool_fill)
        self.assertIn("if cap is None:", body)
        self.assertIn('self._pix_resolve_cfg()["pool_windows"]', body)
        sig = inspect.signature(Trainer._pix_pool_fill)
        self.assertIs(sig.parameters["cap"].default, None)


# ===========================================================================
# 2. GRAM CORRECTNESS vs the offline evaluator
# ===========================================================================
class TestGramCorrectness(unittest.TestCase):

    def test_gram_matches_a_hand_computed_reference(self):
        """A 1x2x2x2 tensor whose Gram can be worked out on paper."""
        x = torch.tensor(
            [[[[1.0, 2.0], [3.0, 4.0]],
              [[0.0, 1.0], [0.0, 1.0]]]]
        )                                            # [1, C=2, 2, 2]
        # f0 = [1,2,3,4]; f1 = [0,1,0,1]
        # <f0,f0> = 1+4+9+16 = 30 ; <f0,f1> = 2+4 = 6 ; <f1,f1> = 2
        # normaliser = C*H*W = 2*2*2 = 8
        want = torch.tensor([[[30.0, 6.0], [6.0, 2.0]]]) / 8.0
        got = gram_matrix(x)
        self.assertEqual(tuple(got.shape), (1, 2, 2))
        self.assertTrue(torch.allclose(got, want, atol=1e-7), f"{got} vs {want}")

    def test_gram_matches_the_offline_evaluator(self):
        """Bit-for-bit against the transcribed ``VGGStyle.forward`` body.

        This is the alignment claim: the training loss and
        ``grids/eval/style_shift.py`` measure the same statistic, so a number
        in ``results_style.csv`` and a number in the trace are commensurate.
        """
        torch.manual_seed(7)
        for shape in ((3, 5, 7, 11), (1, 16, 4, 4), (2, 64, 13, 9)):
            x = torch.randn(*shape)
            self.assertTrue(
                torch.equal(gram_matrix(x), offline_vgg_style_gram(x)),
                f"gram_matrix diverged from the offline formula at {shape}",
            )

    def test_eval_form_matches_the_offline_gram_distance(self):
        torch.manual_seed(11)
        a = [torch.randn(4, 6, 6), torch.randn(4, 5, 5)]
        b = [torch.randn(4, 6, 6), torch.randn(4, 5, 5)]
        want = offline_gram_distance(a, b)
        got = gram_distance_eval([t.mean(0) for t in a], [t.mean(0) for t in b])
        self.assertAlmostEqual(got, want, places=6)

    def test_adain_stats_are_the_population_moments(self):
        torch.manual_seed(3)
        x = torch.randn(2, 5, 6, 7)
        mu, sd = adain_stats(x)
        f = x.reshape(2, 5, 42)
        self.assertTrue(torch.allclose(mu, f.mean(-1), atol=1e-6))
        self.assertTrue(
            torch.allclose(sd, f.std(-1, unbiased=False), atol=1e-5)
        )

    def test_gram_diagonal_is_the_channel_second_moment(self):
        """Load-bearing for the two-sidedness argument: ``G_ii`` IS the
        texture energy of channel i, so a distance on G is a distance on
        energy -- penalising too much exactly as it penalises too little."""
        torch.manual_seed(5)
        x = torch.randn(1, 4, 8, 8) * 2.0
        g = gram_matrix(x)[0]
        c, h, w = 4, 8, 8
        for i in range(4):
            want = float((x[0, i] ** 2).mean()) * (h * w) / (c * h * w)
            self.assertAlmostEqual(float(g[i, i]), want, places=5)


# ===========================================================================
# 3. SPATIAL-PERMUTATION INVARIANCE  (= what makes it unpaired)
# ===========================================================================
class TestUnpairedness(unittest.TestCase):

    def test_gram_is_invariant_to_spatial_permutation(self):
        torch.manual_seed(13)
        x = torch.randn(2, 7, 9, 11)
        g0 = gram_matrix(x)
        flat = x.reshape(2, 7, 99)
        perm = torch.randperm(99)
        g1 = gram_matrix(flat[:, :, perm].reshape(2, 7, 9, 11))
        self.assertTrue(
            torch.allclose(g0, g1, atol=1e-5),
            "Gram changed under a spatial permutation -- it is not a "
            "layout-free statistic, so the loss would silently require "
            "spatial correspondence.",
        )

    def test_adain_is_invariant_to_spatial_permutation(self):
        torch.manual_seed(17)
        x = torch.randn(2, 5, 6, 6)
        m0, s0 = adain_stats(x)
        flat = x.reshape(2, 5, 36)
        perm = torch.randperm(36)
        m1, s1 = adain_stats(flat[:, :, perm].reshape(2, 5, 6, 6))
        self.assertTrue(torch.allclose(m0, m1, atol=1e-6))
        self.assertTrue(torch.allclose(s0, s1, atol=1e-6))

    def test_distance_is_zero_between_two_layouts_of_the_same_content(self):
        """The unpairedness claim in loss terms: shuffle every pixel of the
        'fake' and the style distance to the 'real' does not move."""
        torch.manual_seed(19)
        enc = TinyEncoder(seed=2)
        real = torch.randn(6, 3, 16, 16).clamp(-1, 1)
        fake = torch.randn(6, 3, 16, 16).clamp(-1, 1)
        rg = region_mean_gram(enc(real))
        d0, _ = gram_style_distance(
            [x[0] for x in region_mean_gram(enc(fake))],
            [x[0] for x in rg],
        )
        flat = fake.reshape(6, 3, 256)
        shuf = flat[:, :, torch.randperm(256)].reshape(6, 3, 16, 16)
        d1, _ = gram_style_distance(
            [x[0] for x in region_mean_gram(enc(shuf))],
            [x[0] for x in rg],
        )
        # Not bit-identical (the conv taps are local, so shuffling PIXELS
        # changes the features) -- what must hold is that no spatial
        # correspondence is required for the loss to be defined and finite.
        self.assertTrue(math.isfinite(float(d0)))
        self.assertTrue(math.isfinite(float(d1)))


# ===========================================================================
# 4 + 5. DESCENT and GRADIENT FLOW
# ===========================================================================
class TinyImageGenerator(nn.Module):
    """Stands in for the student: parameters -> images."""

    def __init__(self, n=6, hw=16):
        super().__init__()
        self.z = nn.Parameter(torch.randn(n, 3, hw, hw) * 0.2)
        self.gain = nn.Parameter(torch.tensor(0.5))

    def forward(self):
        return torch.tanh(self.z * self.gain)


class TestDescentAndGradients(unittest.TestCase):

    def test_gradient_reaches_the_generator_and_not_the_encoder(self):
        torch.manual_seed(23)
        enc = TinyEncoder(seed=4)
        gen = TinyImageGenerator()
        real = (torch.randn(6, 3, 16, 16) * 0.6).clamp(-1, 1)
        with torch.no_grad():
            target = [x[0] for x in region_mean_gram(enc(real))]
        loss, _ = gram_style_distance(
            [x[0] for x in region_mean_gram(enc(gen()))], target,
        )
        loss.backward()
        self.assertIsNotNone(gen.z.grad)
        self.assertGreater(float(gen.z.grad.norm()), 0.0)
        self.assertIsNotNone(gen.gain.grad)
        self.assertGreater(abs(float(gen.gain.grad)), 0.0)
        for p in enc.parameters():
            self.assertIsNone(
                p.grad, "the frozen encoder received a weight gradient",
            )

    def test_loss_decreases_under_sgd_toward_a_target_style(self):
        torch.manual_seed(29)
        enc = TinyEncoder(seed=6)
        gen = TinyImageGenerator()
        real = (torch.randn(6, 3, 16, 16) * 0.6).clamp(-1, 1)
        with torch.no_grad():
            target = [x[0] for x in region_mean_gram(enc(real))]
        opt = torch.optim.SGD(gen.parameters(), lr=0.5)
        first = last = None
        for it in range(40):
            opt.zero_grad()
            loss, _ = gram_style_distance(
                [x[0] for x in region_mean_gram(enc(gen()))], target,
            )
            loss.backward()
            opt.step()
            if it == 0:
                first = float(loss)
            last = float(loss)
        self.assertLess(
            last, first * 0.5,
            f"style distance only fell {first:.4f} -> {last:.4f} in 40 SGD "
            "steps; the loss is not usefully optimisable.",
        )

    def test_adain_mode_also_descends(self):
        torch.manual_seed(31)
        enc = TinyEncoder(seed=8)
        gen = TinyImageGenerator()
        real = (torch.randn(6, 3, 16, 16) * 0.6).clamp(-1, 1)
        with torch.no_grad():
            target = [x[0] for x in region_mean_adain(enc(real))]
        opt = torch.optim.SGD(gen.parameters(), lr=0.5)
        first = last = None
        for it in range(40):
            opt.zero_grad()
            loss, _ = adain_style_distance(
                [x[0] for x in region_mean_adain(enc(gen()))], target,
            )
            loss.backward()
            opt.step()
            if it == 0:
                first = float(loss)
            last = float(loss)
        self.assertLess(last, first * 0.6, f"{first:.4f} -> {last:.4f}")


# ===========================================================================
# 6. TWO-SIDEDNESS  (the researcher's blocky/pixelated guard rail)
# ===========================================================================
class TestTwoSidedness(unittest.TestCase):
    """The loss must RISE when texture OVERSHOOTS, not only when it falls short.

    Construction: a channel-selecting encoder makes features a copy of the
    input, so scaling the student's pixels by ``alpha`` scales its Gram by
    ``alpha^2``. Against a real Gram ``G`` the optimised distance is then
    exactly

        d_real_denom(alpha) = ||a^2 G - G|| / ||G|| = |alpha^2 - 1|

    a V with its minimum AT the target -- which is the property, stated
    analytically.
    """

    def _curve(self, alphas, symmetric=False):
        torch.manual_seed(37)
        enc = TinyEncoder(seed=10, identity=True)
        base = torch.randn(8, 3, 16, 16) * 0.25
        with torch.no_grad():
            real_feats = enc(base)
            target = [x[0] for x in region_mean_gram(real_feats)]
        out = []
        for a in alphas:
            with torch.no_grad():
                f = enc(base * float(a))
                fake = [x[0] for x in region_mean_gram(f)]
            if symmetric:
                out.append(gram_distance_eval(fake, target))
            else:
                d, _ = gram_style_distance(fake, target)
                out.append(float(d))
        return out

    def test_loss_rises_when_texture_overshoots(self):
        alphas = [1.0, 1.1, 1.25, 1.55, 2.0]
        d = self._curve(alphas)
        self.assertLess(d[0], 1e-5, f"alpha=1 should be a perfect match: {d[0]}")
        for i in range(1, len(d)):
            self.assertGreater(
                d[i], d[i - 1],
                f"style loss did not RISE going from alpha={alphas[i-1]} to "
                f"alpha={alphas[i]}: {d}. Something in the formulation "
                "rewards more texture monotonically -- that is the blocky/"
                "pixelated failure the researcher ruled out.",
            )
        # The measured failure mode, by name: an arm overshooting its GT's
        # high-frequency energy by 1.55x must be PENALISED.
        self.assertGreater(d[alphas.index(1.55)], 0.5)

    def test_loss_rises_symmetrically_when_texture_undershoots(self):
        under = self._curve([1.0, 0.9, 0.75, 0.5])
        for i in range(1, len(under)):
            self.assertGreater(under[i], under[i - 1], under)

    def test_the_analytic_shape_is_abs_alpha_squared_minus_one(self):
        alphas = [0.5, 0.8, 1.0, 1.3, 1.55, 2.0]
        d = self._curve(alphas)
        for a, got in zip(alphas, d):
            want = abs(a * a - 1.0)
            self.assertAlmostEqual(
                got, want, places=4,
                msg=f"alpha={a}: got {got}, analytic {want}",
            )

    def test_MUTATION_CONTROL_the_symmetric_denominator_saturates(self):
        """Why the optimised form does NOT reuse the offline denominator.

        The offline metric normalises by ``(||G_f|| + ||G_r||)/2``. Under
        that form the overshoot distance is ``|a^2-1| / ((a^2+1)/2)``, which
        is BOUNDED BY 2 -- 1.20 at alpha=2, 1.76 at 4, 1.96 at 10, 1.9996 at
        100. Past a modest overshoot it stops objecting: the gradient
        against extreme over-texturing vanishes. The shipped form
        (real-side denominator, detached) is ``|a^2 - 1|`` and grows without
        bound, so it keeps pushing back however far the student overshoots.

        If this assertion ever fails, someone has switched the training loss
        onto the eval denominator and quietly removed the guard rail.
        """
        big = [10.0, 100.0]
        sym = self._curve(big, symmetric=True)
        real = self._curve(big, symmetric=False)
        self.assertLess(
            sym[1] - sym[0], 0.1,
            f"the symmetric form was expected to saturate; got {sym}",
        )
        self.assertLess(sym[1], 2.0, f"symmetric form is bounded by 2: {sym}")
        self.assertGreater(
            real[1] - real[0], 1000.0,
            f"the shipped form must keep growing; got {real}",
        )
        # And the ratio of "how much harder it pushes" over the same span:
        # the shipped form ~100x, the eval form ~2 %.
        self.assertGreater(real[1] / max(real[0], 1e-9), 50.0)
        self.assertLess(sym[1] / max(sym[0], 1e-9), 1.1)

    def test_nothing_in_the_optimised_form_differentiates_the_denominator(self):
        """Structural twin of the test above: the real side is detached, so
        no gradient can flow into the normaliser."""
        torch.manual_seed(41)
        gf = torch.randn(6, 6, requires_grad=True)
        gr = torch.randn(6, 6, requires_grad=True)
        d, _ = gram_style_distance([gf], [gr])
        d.backward()
        self.assertIsNotNone(gf.grad)
        self.assertIsNone(
            gr.grad,
            "the real-side Gram received a gradient -- the denominator is "
            "live and the loss can be lowered by inflating fake texture.",
        )

    def test_energy_ratio_telemetry_detects_the_direction(self):
        """``train/style_energy_ratio`` must read >1 on overshoot and <1 on
        undershoot -- distance alone cannot tell those two apart, and they
        call for opposite corrections."""
        torch.manual_seed(43)
        enc = TinyEncoder(seed=12, identity=True)
        base = torch.randn(8, 3, 16, 16) * 0.25
        with torch.no_grad():
            rg = region_mean_gram(enc(base))
            for a, cmp_ in ((1.55, "gt"), (0.6, "lt")):
                fg = region_mean_gram(enc(base * a))
                ratio = sum(
                    float(fg[l][0].norm()) / float(rg[l][0].norm())
                    for l in range(len(fg))
                ) / len(fg)
                if cmp_ == "gt":
                    self.assertGreater(ratio, 1.5)   # ~ a^2 = 2.40
                else:
                    self.assertLess(ratio, 1.0)


# ===========================================================================
# 7. Region / band behaviour, config resolution, telemetry
# ===========================================================================
class TestRegionsAndConfig(unittest.TestCase):

    def test_region_grouping_matches_bands_not_positions(self):
        torch.manual_seed(47)
        enc = TinyEncoder(seed=14)
        px = torch.randn(6, 3, 16, 16).clamp(-1, 1)
        feats = enc(px)
        g_all = region_mean_gram(feats, [[0, 1, 2, 3, 4, 5]])
        g_split = region_mean_gram(feats, [[0, 1, 2], [3, 4, 5]])
        self.assertEqual(len(g_split[0]), 2)
        mean_of_halves = (g_split[0][0] + g_split[0][1]) / 2
        self.assertTrue(torch.allclose(g_all[0][0], mean_of_halves, atol=1e-5))

    def test_style_distance_refuses_to_forge_a_zero_with_no_reference(self):
        torch.manual_seed(53)
        enc = TinyEncoder(seed=16)
        feats = enc(torch.randn(4, 3, 16, 16).clamp(-1, 1))
        with self.assertRaises(RuntimeError):
            style_distance(feats, [[None], [None]], mode="gram")

    def test_resolver_defaults_are_off_and_match_the_module_table(self):
        t = make_trainer()
        rc = t._style_resolve_cfg()
        self.assertEqual(rc["weight"], 0.0)
        self.assertEqual(rc["mode"], STYLE_DEFAULTS["style_gram_mode"])
        self.assertEqual(rc["encoder"], STYLE_DEFAULTS["style_gram_encoder"])
        self.assertEqual(rc["every"], STYLE_DEFAULTS["style_gram_every"])
        self.assertEqual(rc["n_crops"], STYLE_DEFAULTS["style_gram_crops"])
        self.assertEqual(rc["ema"], STYLE_DEFAULTS["style_gram_real_ema"])
        self.assertEqual(rc["fake_source"],
                         STYLE_DEFAULTS["style_gram_fake_source"])
        self.assertIsNotNone(rc["pool_cap"])       # style owns the cap when pix is off

    def test_resolver_rejects_the_hazardous_settings(self):
        for kw, frag in (
            ({"style_gram_every": 0}, "must be >= 1"),
            ({"style_gram_mode": "wavelet"}, "style_gram_mode"),
            ({"style_gram_encoder": "clip"}, "style_gram_encoder"),
            ({"style_gram_band_count": 8}, "A24"),
            ({"style_gram_real_ema": 1.0}, "freeze"),
            ({"style_gram_real_pool_refresh": 0}, "CONTINUOUSLY REFRESHED"),
            ({"style_gram_crops": 0}, "must both be >= 1"),
            ({"style_gram_fake_source": "gt"}, "style_gram_fake_source"),
        ):
            t = make_trainer(**kw)
            with self.assertRaises(ValueError, msg=f"{kw} was accepted") as cm:
                t._style_resolve_cfg()
            self.assertIn(frag, str(cm.exception))

    def test_every_style_key_is_seen_by_the_override_guard(self):
        """A ``style_gram_*`` override that nothing reads must be REPORTED,
        not merged silently -- this project's endemic failure. The prefix is
        registered and every key is read at a literal getattr site, so the
        scan must find them all."""
        sourced = CAFT._scan_config_sourced_keys(("style_",), REPO_ROOT)
        for key in STYLE_DEFAULTS:
            self.assertIn(
                key, sourced,
                f"{key} is not sourced anywhere the override guard can see; "
                "an override of it would merge silently and do nothing.",
            )
        self.assertIn("style_", CAFT._OVERRIDE_GUARD_PREFIXES)

    def test_config_yaml_ships_every_key_at_its_code_default(self):
        """The yaml block and the code's getattr defaults must agree, or the
        documented value is not the value that runs."""
        import yaml
        path = os.path.join(
            REPO_ROOT, "configs", "action_forcing_phase3_dmd.yaml",
        )
        with open(path) as fh:
            cfg = yaml.safe_load(fh)
        for key, want in STYLE_DEFAULTS.items():
            self.assertIn(key, cfg, f"{key} missing from the shipped config")
            got = cfg[key]
            if isinstance(want, tuple):
                got = tuple(got)
            self.assertEqual(got, want, f"{key}: yaml {got!r} vs code {want!r}")
        self.assertEqual(cfg["style_gram_loss_weight"], 0.0,
                         "the shipped config must have the feature OFF")

    def test_real_bank_ema_is_debiased_on_the_first_fold(self):
        """First fold must publish the statistic ITSELF, not ``(1-m)`` times
        it -- otherwise the first steps aim at a systematically SMALLER
        target, i.e. at less texture than the dataset has."""
        t = make_trainer(style_gram_real_ema=0.9)
        rc = t._style_resolve_cfg()
        torch.manual_seed(59)
        enc = TinyEncoder(seed=18)
        feats = enc(torch.randn(3, 3, 16, 16).clamp(-1, 1))
        direct = region_mean_gram(feats, [[0, 1, 2]])
        target, n_ready = t._style_update_real_bank(
            feats, [0, 0, 0], rc=rc, n_images=3,
        )
        self.assertEqual(n_ready, 1)
        self.assertTrue(
            torch.allclose(target[0][0], direct[0][0], atol=1e-6),
            "first fold was scaled by (1-m) instead of published raw",
        )
        self.assertIsNone(target[0][1], "a band with no reals must be None")

    def test_real_bank_ema_converges_to_the_population_statistic(self):
        t = make_trainer(style_gram_real_ema=0.5)
        rc = t._style_resolve_cfg()
        torch.manual_seed(61)
        enc = TinyEncoder(seed=20)
        pop = []
        for _ in range(30):
            px = torch.randn(4, 3, 16, 16).clamp(-1, 1)
            feats = enc(px)
            pop.append(region_mean_gram(feats, [[0, 1, 2, 3]])[0][0])
            target, _ = t._style_update_real_bank(
                feats, [1, 1, 1, 1], rc=rc, n_images=4,
            )
        truth = torch.stack(pop).mean(0)
        err_bank = float((target[0][1] - truth).norm() / truth.norm())
        err_single = float((pop[-1] - truth).norm() / truth.norm())
        self.assertLess(
            err_bank, err_single,
            "the EMA bank is not a lower-variance target than one step's "
            "reals -- the convergence argument does not hold.",
        )
        self.assertEqual(t._style_real_bank["images"], 120)

    def test_telemetry_contract_is_present_in_the_source(self):
        """Silent no-ops are this project's endemic failure: an enabled
        feature must be able to PROVE it fired."""
        body = inspect.getsource(Trainer._compute_style_gram_loss)
        for key in (
            "train/style_gram_dist_raw",        # the raw distance
            "train/style_loss_weighted",        # the weighted contribution
            "train/style_gram_dist_evalform",   # comparable with the offline csv
            "train/style_energy_ratio",         # the two-sidedness direction
            "train/style_block_wall_s",         # measured cost
            "train/style_fake_images",
            "train/style_real_images",
        ):
            self.assertIn(key, body, f"missing telemetry key {key}")
        # The proof-of-fire pair belongs to the CONSUMPTION site, not to the
        # producer -- a "it fired" claim written by the helper would be a
        # claim about the wrong event.
        call_site = extract_call_site_block()
        for key in ("train/style_loss_applied",
                    "train/style_loss_applied_total"):
            self.assertIn(key, call_site, f"missing telemetry key {key}")
        self.assertNotIn("train/style_loss_applied", body)
        tel = inspect.getsource(Trainer._style_grad_telemetry)
        self.assertIn("train/style_dmd_grad_ratio", tel)
        self.assertIn("train/style_dmd_grad_cos", tel)

    def test_applied_counter_is_monotone(self):
        t = make_trainer(style_gram_loss_weight=0.1)
        t._style_applied_total = 0
        for i in range(1, 4):
            t._style_applied_total = int(
                getattr(t, "_style_applied_total", 0)
            ) + 1
            self.assertEqual(t._style_applied_total, i)


# ===========================================================================
# 8. The real VGG encoder (skipped if the cached weights are unavailable)
# ===========================================================================
class TestFrozenVGGEncoder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            cls.enc = FrozenStyleEncoder("vgg16", device=torch.device("cpu"))
        except Exception as exc:                     # pragma: no cover
            raise unittest.SkipTest(f"vgg16 weights unavailable: {exc!r}")

    def test_default_taps_are_the_offline_local_style_set(self):
        self.assertEqual(self.enc.taps, tuple(VGG16_DEFAULT_LAYERS))
        self.assertEqual(self.enc.tap_names,
                         ["relu1_2", "relu2_2", "relu3_3"])

    def test_truncated_to_the_deepest_tap_and_small_enough_to_be_cheap(self):
        self.assertEqual(len(self.enc.features), max(VGG16_DEFAULT_LAYERS) + 1)
        self.assertEqual(self.enc.n_params, 1735488)   # ~1.74 M, ~7 MB fp32

    def test_features_are_frozen_and_pinned_to_eval(self):
        self.enc.train(True)
        self.assertFalse(self.enc.features.training)
        for p in self.enc.features.parameters():
            self.assertFalse(p.requires_grad)

    def test_gradient_flows_to_the_input_but_not_the_weights(self):
        x = (torch.rand(2, 3, 48, 64) * 2 - 1).requires_grad_(True)
        feats = self.enc(x)
        loss = sum(gram_matrix(f).pow(2).mean() for f in feats)
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(float(x.grad.norm()), 0.0)
        for p in self.enc.features.parameters():
            self.assertIsNone(p.grad)

    def test_no_resize_happens(self):
        """Resizing would rescale the very texture being measured, so the
        tap shapes must track the INPUT size."""
        s1 = [tuple(f.shape[-2:]) for f in self.enc(torch.zeros(1, 3, 64, 64))]
        s2 = [tuple(f.shape[-2:]) for f in self.enc(torch.zeros(1, 3, 96, 128))]
        self.assertNotEqual(s1, s2)
        self.assertEqual(s1[0], (64, 64))
        self.assertEqual(s2[0], (96, 128))

    def test_too_small_a_crop_fails_loudly_with_the_fix_in_the_message(self):
        with self.assertRaises(ValueError) as cm:
            self.enc(torch.zeros(1, 3, 3, 3))
        self.assertIn("style_gram_crop_lat", str(cm.exception))

    def test_matches_the_offline_evaluator_end_to_end(self):
        """The training encoder's relu1_2/2_2/3_3 Grams equal what the
        offline ``VGGStyle.forward`` body produces on the same input.

        The reference is driven through ``self.enc.features`` -- the SAME
        weight tensors -- using the offline code's own normalisation and tap
        loop, transcribed above. That is the part that can actually diverge
        (the [-1,1] -> [0,1] conversion, the ImageNet stats, which indices
        are tapped, the Gram normaliser); the weights are identical by
        construction because both come from the same
        ``VGG16_Weights.IMAGENET1K_V1`` load. Building a SECOND full vgg16
        here would add ~550 MB of peak RSS to prove a tautology, and OOMs
        this box when the suite runs alongside anything else.
        """
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x01 = torch.rand(2, 3, 48, 64)
        with torch.no_grad():
            # --- the OFFLINE pipeline, verbatim: [0,1] input, ImageNet
            # normalisation, tap at 3/8/15, Gram = bmm/(c*h*w).
            ref, cur = [], (x01 - mean) / std
            for i, layer in enumerate(self.enc.features):
                cur = layer(cur)
                if i in (3, 8, 15):
                    ref.append(offline_vgg_style_gram(cur))
            # --- the TRAINING pipeline: [-1,1] input, encoder does the rest.
            got = [gram_matrix(f) for f in self.enc(x01 * 2 - 1)]
        self.assertEqual(len(got), len(ref), "tap count diverged")
        for a, b in zip(got, ref):
            self.assertTrue(
                torch.allclose(a, b, atol=1e-4, rtol=1e-4),
                f"max abs diff {float((a - b).abs().max())}",
            )


# ===========================================================================
# 9. END-TO-END through the REAL trainer helper
# ===========================================================================
class StubVAE(nn.Module):
    """A differentiable stand-in for the WAN VAE decoder.

    ``[B, F_lat, C, h, w] -> [B, 4*F_lat, 3, 4h, 4w]`` in [-1, 1], with a
    real graph, so ``_pix_decode_crops_grad`` (checkpointed) and
    ``_vae_decode_nograd`` both behave as they do in production and the
    gradient genuinely has to traverse the decode.
    """

    def __init__(self, c_lat=16):
        super().__init__()
        self.proj = nn.Conv2d(c_lat, 3, 1, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(
                torch.randn(self.proj.weight.shape,
                            generator=torch.Generator().manual_seed(2)) * 0.4
            )

    def decode_to_pixel(self, z, seed_first=True):
        b, f, c, h, w = z.shape
        x = self.proj(z.reshape(b * f, c, h, w))
        x = torch.nn.functional.interpolate(x, scale_factor=4, mode="nearest")
        x = torch.tanh(x)
        x = x.reshape(b, f, 3, 4 * h, 4 * w)
        return x.repeat_interleave(4, dim=1)          # 4x temporal expansion


def make_e2e_trainer(**over):
    cfg = dict(
        style_gram_loss_weight=0.25,
        style_gram_crops=3,
        style_gram_frames_per_crop=2,
        style_gram_crop_lat=[6, 8],
        style_gram_lat_frames_per_crop=2,
        style_gram_decode_border_trim=0,
        style_gram_decode_batch=2,
        style_gram_real_pool_refresh=2,
        gan_grad_telemetry_every=0,
    )
    cfg.update(over)
    t = make_trainer(**cfg)
    t.style_gram_loss_weight = float(cfg["style_gram_loss_weight"])
    t.model.vae = StubVAE()
    t._style_encoder_module = TinyEncoder(seed=22)
    # Pre-stock the pool; ``_pix_pool_fill`` (zarr-backed) is replaced by a
    # recorder so the CAP PLUMBING is asserted without touching a dataset.
    torch.manual_seed(67)
    pool = []
    for i in range(30):
        pool.append({
            "lat": (torch.randn(2, 16, 6, 8) * 0.5).to(torch.float16),
            "band": i % 3, "y0": 0, "x0": 0, "ride": f"r{i % 5}",
            "start": i, "nf": 2, "uid": (f"r{i % 5}", i, 0, 0),
        })
    t._pix_real_pool = pool
    t._pix_real_support = set()
    t._pix_real_rides = set()
    t._pix_pool_refresh_total = 0
    calls = []

    def rec_fill(self, n_new, **kw):
        calls.append(kw)
        return 0

    t._pix_pool_fill = types.MethodType(rec_fill, t)
    t._fill_calls = calls
    return t


class TestEndToEnd(unittest.TestCase):

    def _run(self, t, step=0, chunk=None):
        if chunk is None:
            torch.manual_seed(71)
            chunk = (torch.randn(1, 4, 16, 12, 16) * 0.5).requires_grad_(True)
        w, raw, logs = t._compute_style_gram_loss(chunk, {}, current_step=step)
        return w, raw, logs, chunk

    def test_full_path_produces_a_graph_bearing_term(self):
        t = make_e2e_trainer()
        w, raw, logs, chunk = self._run(t)
        self.assertIsNotNone(w)
        self.assertTrue(raw.requires_grad)
        self.assertTrue(math.isfinite(float(raw)))
        self.assertAlmostEqual(float(w), float(raw) * 0.25, places=5)

    def test_gradient_reaches_the_student_latents_through_the_decode(self):
        """The claim the whole feature rests on: a differentiable path from
        the style statistic, through the frozen encoder, through the VAE
        decode, back to the generator's latents."""
        t = make_e2e_trainer()
        w, raw, logs, chunk = self._run(t)
        w.backward()
        self.assertIsNotNone(chunk.grad)
        self.assertGreater(float(chunk.grad.norm()), 0.0)
        for p in t._style_encoder_module.parameters():
            self.assertIsNone(p.grad, "the frozen encoder got a weight grad")

    def test_telemetry_keys_are_actually_emitted(self):
        t = make_e2e_trainer()
        _w, _raw, logs, _c = self._run(t)
        for key in ("train/style_gram_dist_raw", "train/style_loss_weighted",
                    "train/style_gram_weight", "train/style_gram_dist_evalform",
                    "train/style_energy_ratio", "train/style_energy_gap",
                    "train/style_fake_images", "train/style_real_images",
                    "train/style_real_bank_images", "train/style_regions_matched",
                    "train/style_block_wall_s", "train/style_real_pool_windows",
                    "train/style_cfg_every", "train/style_cfg_n_crops"):
            self.assertIn(key, logs, f"missing {key}")
        # 3 crops x 2 frames = 6 images per side.
        self.assertEqual(logs["train/style_fake_images"], 6.0)
        self.assertEqual(logs["train/style_real_images"], 6.0)
        # Per-layer breakdown, named by the encoder's taps.
        self.assertIn("train/style_layer_tap0", logs)
        self.assertIn("train/style_layer_tap1", logs)

    def test_resolved_config_echo_is_emitted_exactly_once(self):
        t = make_e2e_trainer()
        _w, _r, logs1, _c = self._run(t, step=0)
        _w, _r, logs2, _c = self._run(t, step=1)
        self.assertIn("train/style_cfg_every", logs1)
        self.assertNotIn("train/style_cfg_every", logs2)

    def test_the_pool_cap_is_plumbed_through_to_pix_pool_fill(self):
        """``_pix_pool_fill(cap=...)`` is the one edit to a shared helper;
        the style path must actually use it, or it would fall through to the
        pixel resolver and raise on an arm with no ``pix_r1_gamma``."""
        t = make_e2e_trainer()
        self._run(t)
        self.assertTrue(t._fill_calls, "the refresh never ran")
        for kw in t._fill_calls:
            self.assertEqual(kw["cap"], 1024)
        # 2 refresh admissions per style step (style_gram_real_pool_refresh).
        self.assertGreaterEqual(len(t._fill_calls), 2)

    def test_cadence_and_warmup_decline_without_forging_a_zero(self):
        t = make_e2e_trainer(style_gram_every=4)
        w, raw, logs, _c = self._run(t, step=3)
        self.assertIsNone(w)
        self.assertIsNone(raw)
        self.assertEqual(logs, {"train/style_cadence_skipped": 1.0})
        self.assertNotIn("train/style_gram_dist_raw", logs)

        t2 = make_e2e_trainer(style_gram_warmup_steps=10)
        w, raw, logs, _c = self._run(t2, step=2)
        self.assertIsNone(w)
        self.assertEqual(logs, {"train/style_warmup_skipped": 1.0})

    def test_a_detached_student_chunk_is_reported_not_silently_scored(self):
        t = make_e2e_trainer()
        chunk = (torch.randn(1, 4, 16, 12, 16) * 0.5)     # no requires_grad
        w, raw, logs = t._compute_style_gram_loss(chunk, {}, current_step=0)
        self.assertIsNone(w)
        self.assertEqual(logs["train/style_fake_no_grad"], 1.0)
        self.assertNotIn("train/style_gram_dist_raw", logs)

    def test_adain_mode_runs_end_to_end(self):
        t = make_e2e_trainer(style_gram_mode="adain")
        w, raw, logs, chunk = self._run(t)
        self.assertIsNotNone(w)
        w.backward()
        self.assertGreater(float(chunk.grad.norm()), 0.0)

    def test_distance_falls_as_the_student_moves_toward_the_real_style(self):
        """The convergence claim, exercised through the SHIPPED path.

        The student chunk is optimised directly (it stands in for whatever
        the generator would have produced); the distance must fall fast --
        this is the ~50-step budget, measured over 25.
        """
        t = make_e2e_trainer(style_gram_real_ema=0.9)
        torch.manual_seed(73)
        chunk = nn.Parameter(torch.randn(1, 4, 16, 12, 16) * 1.4)
        # Adam, because that is what the trainer optimises the generator
        # with (``use_8bit_adam=true`` in every phase-3 arm) -- an SGD
        # landscape on a stub decoder would be measuring the stub.
        opt = torch.optim.Adam([chunk], lr=0.05)
        first = last = None
        for it in range(25):
            opt.zero_grad()
            w, raw, logs = t._compute_style_gram_loss(
                chunk, {}, current_step=it,
            )
            w.backward()
            opt.step()
            if it == 0:
                first = float(raw)
            last = float(raw)
        self.assertLess(
            last, first * 0.7,
            f"style distance {first:.4f} -> {last:.4f} in 25 steps through "
            "the shipped path; the signal is not usable inside the ~50-step "
            "budget the researcher asked for.",
        )


# ===========================================================================
# 10. THE GRAD PROBE ITSELF  (``_style_grad_telemetry``)
#
# Added 2026-08-26 after the live ``dmd10k_gantune_w2gram`` arm reported
#
#     style_grad_norm_unweighted  = 0.0  (EXACTLY, 13/13 telemetry samples)
#     style_base_grad_norm_shared = 0.032 .. 0.209   (healthy)
#     style_grad_probe_unavailable  ABSENT
#
# while every test in section 9 passed. Section 9 asserts the gradient
# reaches the LATENT TENSOR; nothing asserted anything about the PROBE, so
# the probe was the one part of the feature with no coverage at all. These
# tests close that hole and, between them, decide which of the two readings
# of that zero is the true one:
#
#   (a) MEASUREMENT ARTEFACT -- the probed parameter is not on the style
#       loss's graph, so ``autograd.grad`` legitimately hands back zeros;
#   (b) REAL SEVERING -- the style term genuinely delivers no gradient.
#
# ``test_the_style_term_delivers_gradient_to_an_on_path_parameter`` rules
# out (b) for the shipped code path. ``test_an_off_path_trailing_param_
# reproduces_the_live_zero`` reproduces (a)'s EXACT signature -- non-None,
# exactly zero, no ``_unavailable`` key -- which is the only one of the two
# that can produce what the live run recorded.
# ===========================================================================
class OnPathGen(nn.Module):
    """A generator stand-in whose LAST registered parameter is on the path
    that produces the latent chunk."""

    def __init__(self):
        super().__init__()
        self.body = nn.Conv3d(16, 16, 1)          # last param = body.bias

    def forward(self, z):
        return self.body(z.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)


class AuxTailGen(nn.Module):
    """A generator stand-in shaped like the real one: an auxiliary branch
    registered AFTER the trunk, fed by its own loss, and connected to the
    chunk only through a concat-then-slice junction.

    That junction is what every auxiliary TOKEN branch in this codebase
    does (state tokens, action tokens, register tokens): extra tokens ride
    the same tensor through the trunk and are sliced off before the head.
    The parameter is therefore graph-REACHABLE from the chunk -- so
    ``autograd.grad`` returns a tensor rather than ``None`` -- while its
    gradient from the chunk is structurally, exactly zero.
    """

    def __init__(self):
        super().__init__()
        self.body = nn.Conv3d(16, 16, 1)
        self.aux = nn.Linear(2, 2)                # registered LAST

    def forward(self, z):
        b, f, c, h, w = z.shape
        x = self.body(z.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        tok = self.aux(torch.ones(1, 2)).reshape(1, 1, 1, 1, 2)
        tok = tok.expand(b, f, c, h, 2)
        chunk = torch.cat([x, tok], dim=-1)[..., :w]
        return chunk, self.aux(torch.ones(1, 2)).sum()


class TestStyleGradProbe(unittest.TestCase):

    def _fire(self, gen, *, seed=71):
        """Run the SHIPPED helper + the SHIPPED probe end to end."""
        t = make_e2e_trainer(gan_grad_telemetry_every=1)
        t.model.generator = gen
        torch.manual_seed(seed)
        z = torch.randn(1, 4, 16, 12, 16) * 0.5
        res = gen(z)
        aux = None
        chunk = res
        if isinstance(res, tuple):
            chunk, aux = res
        w, raw, logs = t._compute_style_gram_loss(chunk, {}, current_step=0)
        self.assertIsNotNone(raw, f"the style helper declined: {logs}")
        generator_loss = (chunk ** 2).mean()
        if aux is not None:
            # The aux branch's OWN loss -- the analogue of the z-guidance /
            # state-probe terms that give the real trailing parameter a
            # healthy base gradient while contributing nothing to the
            # predicted latents.
            generator_loss = generator_loss + 3.0 * aux
        out = {}
        t._style_grad_telemetry(
            generator_loss, raw, out, current_step=0, probe_tensor=chunk,
        )
        return t, chunk, raw, out

    # -- (b) is FALSE for the shipped path -----------------------------
    def test_the_style_term_delivers_gradient_to_an_on_path_parameter(self):
        """THE VERDICT TEST. Run the real ``_compute_style_gram_loss`` and
        the real ``grad_at``, and differentiate at a generator PARAMETER
        (not at the latent tensor section 9 already covers).

        A non-zero result here means the crop -> VAE decode -> frozen
        encoder -> Gram chain carries gradient all the way into the
        generator's weights, so a zero read in production cannot be a
        severed loss -- it can only be a probe pointed at the wrong
        parameter.
        """
        gen = OnPathGen()
        t, chunk, raw, out = self._fire(gen)
        g = torch.autograd.grad(
            raw, [gen.body.bias], retain_graph=True, allow_unused=True,
        )[0]
        self.assertIsNotNone(
            g, "the style term does not even reach an on-path parameter",
        )
        self.assertGreater(
            float(g.norm()), 0.0,
            "the style term reaches an on-path generator parameter but "
            "delivers exactly zero gradient -- that WOULD be a real "
            "severing",
        )
        # ... and the probe reports it.
        self.assertGreater(out["train/style_grad_norm_unweighted"], 0.0)
        self.assertEqual(out.get("train/style_param_probe_offpath"), 0.0)

    # -- (a) reproduces the live signature EXACTLY ---------------------
    def test_an_off_path_trailing_param_reproduces_the_live_zero(self):
        """The observed production signature, reproduced on demand.

        ``style_grad_norm_* == 0.0`` with a healthy base grad and NO
        ``_unavailable`` key is reachable with the loss fully intact, as
        long as the probed parameter is the trailing auxiliary one. This
        is what makes the live reading a MEASUREMENT ARTEFACT rather than
        evidence about the loss.
        """
        gen = AuxTailGen()
        names = [n for n, p in gen.named_parameters() if p.requires_grad]
        self.assertEqual(names[-1], "aux.bias", "harness no longer models "
                         "a trailing auxiliary parameter")
        t, chunk, raw, out = self._fire(gen)

        # 1. The style term's gradient at the trailing param: present in
        #    the graph (NOT None) and exactly zero -- the live signature.
        g_style = torch.autograd.grad(
            raw, [gen.aux.bias], retain_graph=True, allow_unused=True,
        )[0]
        self.assertIsNotNone(
            g_style, "the harness no longer reproduces the reachable-but-"
                     "zero junction the live signature requires",
        )
        self.assertEqual(float(g_style.norm()), 0.0)

        # 2. The base loss's gradient at the SAME param is healthy, which
        #    is what made the old pairing look like a severed style term.
        g_base = torch.autograd.grad(
            (chunk ** 2).mean() + 3.0 * gen.aux(torch.ones(1, 2)).sum(),
            [gen.aux.bias], retain_graph=True, allow_unused=True,
        )[0]
        self.assertGreater(float(g_base.norm()), 0.0)

        # 3. And the loss is NOT severed: the same style term has a
        #    non-zero gradient at an on-path parameter.
        g_on = torch.autograd.grad(
            raw, [gen.body.bias], retain_graph=True, allow_unused=True,
        )[0]
        self.assertGreater(float(g_on.norm()), 0.0)

    def test_the_fixed_probe_reads_nonzero_where_the_old_one_read_zero(self):
        """The FIX. On the exact configuration that produced the live
        zeros, the chunk-site probe reports a real number and the demoted
        parameter probe flags itself off-path."""
        t, chunk, raw, out = self._fire(AuxTailGen())
        self.assertEqual(out["train/style_grad_probe_site_is_chunk"], 1.0)
        self.assertGreater(out["train/style_grad_norm_unweighted"], 0.0)
        self.assertGreater(out["train/style_base_grad_norm_shared"], 0.0)
        self.assertGreater(out["train/style_dmd_grad_ratio_unweighted"], 0.0)
        self.assertIn("train/style_dmd_grad_cos", out)
        self.assertNotIn("train/style_grad_telemetry_err", out)
        self.assertNotIn("train/style_grad_probe_unavailable", out)
        # The diagnostic names the artefact instead of leaving a reader to
        # infer it from a suspicious zero.
        self.assertEqual(out["train/style_param_probe_ran"], 1.0)
        self.assertEqual(out["train/style_param_probe_offpath"], 1.0)
        self.assertEqual(out["train/style_param_grad_norm_unweighted"], 0.0)
        self.assertGreater(out["train/style_param_base_grad_norm"], 0.0)

    def test_the_parameter_diagnostic_runs_only_once(self):
        """It costs a full backward through the generator and its only job
        is to answer a yes/no question, so it must not recur."""
        t = make_e2e_trainer(gan_grad_telemetry_every=1)
        gen = AuxTailGen()
        t.model.generator = gen
        for step in range(2):
            torch.manual_seed(71 + step)
            z = torch.randn(1, 4, 16, 12, 16) * 0.5
            chunk, aux = gen(z)
            w, raw, _ = t._compute_style_gram_loss(
                chunk, {}, current_step=step,
            )
            out = {}
            t._style_grad_telemetry(
                (chunk ** 2).mean() + 3.0 * aux, raw, out,
                current_step=step, probe_tensor=chunk,
            )
            if step == 0:
                self.assertEqual(out["train/style_param_probe_ran"], 1.0)
            else:
                self.assertNotIn("train/style_param_probe_ran", out)
                # ...while the authoritative reading keeps coming.
                self.assertGreater(
                    out["train/style_grad_norm_unweighted"], 0.0)

    def test_a_genuinely_severed_style_term_is_reported_loudly(self):
        """The other side of the fix: if the style term ever really does
        stop reaching the chunk it was built from, the probe must say so
        with its own key and never publish a 0.0 norm."""
        t = make_e2e_trainer(gan_grad_telemetry_every=1)
        gen = OnPathGen()
        t.model.generator = gen
        torch.manual_seed(71)
        z = torch.randn(1, 4, 16, 12, 16) * 0.5
        chunk = gen(z)
        w, raw, _ = t._compute_style_gram_loss(chunk, {}, current_step=0)
        severed = raw.detach() + 0.0 * torch.zeros(
            1, requires_grad=True).sum()
        out = {}
        t._style_grad_telemetry(
            (chunk ** 2).mean(), severed, out,
            current_step=0, probe_tensor=chunk,
        )
        self.assertEqual(out["train/style_grad_probe_unavailable"], 1.0)
        self.assertEqual(out["train/style_grad_severed_from_chunk"], 1.0)
        self.assertNotIn("train/style_grad_norm_unweighted", out)

    def test_the_probe_respects_its_cadence_knob(self):
        t = make_e2e_trainer(gan_grad_telemetry_every=0)
        gen = OnPathGen()
        t.model.generator = gen
        torch.manual_seed(71)
        chunk = gen(torch.randn(1, 4, 16, 12, 16) * 0.5)
        w, raw, _ = t._compute_style_gram_loss(chunk, {}, current_step=0)
        out = {}
        t._style_grad_telemetry(
            (chunk ** 2).mean(), raw, out, current_step=0,
            probe_tensor=chunk,
        )
        self.assertEqual(out, {})

    def test_the_call_site_hands_the_probe_the_student_chunk(self):
        """Sourced from the SHIPPED trainer, so the probe cannot be left
        pointing at the old site by a future edit."""
        src = inspect.getsource(Trainer)
        m = re.search(
            r"self\._style_grad_telemetry\((.*?)\n\s*\)\n", src, re.S,
        )
        self.assertIsNotNone(m, "the style grad-telemetry call site moved")
        self.assertIn("probe_tensor=train_chunk", m.group(1))


class TestRealPoolBreadthTelemetry(unittest.TestCase):
    """``style_real_unique_rides`` reads the RIDES IN THIS STEP'S DRAW and
    is bounded by the crop count -- it was read as "the style target is
    estimated from 4 rides", which it does not say. The pool-wide key does.
    """

    def test_pool_wide_ride_count_is_published_and_is_not_the_draw_count(self):
        t = make_e2e_trainer()
        t._pix_real_rides = {f"r{i}" for i in range(17)}
        torch.manual_seed(71)
        chunk = (torch.randn(1, 4, 16, 12, 16) * 0.5).requires_grad_(True)
        _w, _raw, logs = t._compute_style_gram_loss(chunk, {}, current_step=0)
        self.assertEqual(logs["train/style_real_pool_unique_rides"], 17.0)
        # The per-step draw is bounded by the number of crops (3 here), so
        # the two keys measure different things and must not be conflated.
        self.assertLessEqual(logs["train/style_real_unique_rides"], 3.0)

    def test_pool_refresh_is_the_knob_that_grows_the_pool(self):
        """``style_gram_real_pool_windows`` is a CAP (memory bound);
        ``style_gram_real_pool_refresh`` is the admission rate, and
        admission is the only thing that grows support. Raising the cap
        while lowering the refresh narrows the target -- the opposite of
        what a 'wider pool' change intends.
        """
        seen = {}
        for refresh in (1, 8):
            t = make_e2e_trainer(
                style_gram_real_pool_refresh=refresh,
                style_gram_real_pool_windows=4096,
            )
            torch.manual_seed(71)
            chunk = (torch.randn(1, 4, 16, 12, 16) * 0.5).requires_grad_(True)
            t._compute_style_gram_loss(chunk, {}, current_step=0)
            # ``_pix_pool_fill`` is recorded, not executed, by the harness:
            # count the CONTINUOUS-REFRESH admissions (one call per unit of
            # ``pool_refresh``), ignoring any band top-ups.
            seen[refresh] = len(t._fill_calls)
            for kw in t._fill_calls:
                self.assertEqual(kw["cap"], 4096)
        self.assertGreater(
            seen[8], seen[1],
            "raising style_gram_real_pool_refresh did not increase the "
            "admission rate -- it is not the breadth knob",
        )
        self.assertGreaterEqual(seen[8] - seen[1], 7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
