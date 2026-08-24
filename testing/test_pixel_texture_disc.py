"""B2 / WP-PIXGAN pixel texture PatchGAN — unit tests.

Everything here is CPU-only and needs no data on disk.

AUTHORISED DEVIATION FROM §4 (researcher decision, 2026-08-23), which several
tests here now pin: **the GroupNorm has been removed** from blocks 2 and 3 and
the shipped default is norm-free.  It was removed because the normed critic
measured as GLOBAL — a single centre patch logit's input gradient spanned the
whole 176x240 crop, with 4-9 % of its L1 mass outside a 38x38 box — and B2's
frozen scope question is about *LOCAL* pixel adversarial feedback, so a global
critic could not test it and a null result would have been uninterpretable.
Three tests were INVERTED for this and say so in their docstrings:
``test_param_count_measured_661185_after_authorised_groupnorm_removal``,
``test_DEFAULT_critic_carries_NO_normalisation_authorised_deviation`` and
``test_DEFAULT_config_receptive_field_is_LOCAL_38px``.  The facts they used to
assert are NOT deleted — they are re-asserted against the still-constructible
``use_norm=True`` variant, which is also the planted violation that keeps the
locality guard honest.

Covers, in the order ``docs/TEXTURE_GAN_DESIGN.md`` specifies them:
  * §4  architecture — output shape, the real P=660 case, param count
        (661,185 after the -768 GroupNorm removal), spectral norm on all four
        convs, the norm-free default and the normed comparison variant, and
        the *measured* receptive field: 38 px and genuinely local for the
        default, whole-image for the normed variant;
  * §3.7 the effective-sample-count arithmetic, RECOMPUTED on the measured
        38 px receptive field (§3.7's own 768 logits / ~6 tiles / ~128:1
        overlap were both wrong inputs);
  * §5.1 patchwise NS-logistic / hinge losses, and that the nonlinearity is
        applied BEFORE the average (a Jensen-gap test that fails if someone
        ever averages logits first);
  * §5.3 the single R1 finite-difference estimator: agreement with an autograd
        reference, the MEASURED P-dependence of ``gsq`` (which REFUTES §5.3's
        "scale-free in P by construction" — see ``TestR1``), unbiased
        subsampling, and the [-1,1] sigma scale;
  * §5.4/§7 monotone counters and ``pix_r1_rate == 1.00``;
  * §5.3/§7 the R2 spec-drift tripwire — no ``pix_r2`` symbol may exist;
  * §8.1 the C1 corruption: determinism, monotonicity, and the measured
        battery direction (``hv_anisotropy`` DOWN, ``angular_entropy`` DOWN);
  * the generator-side gradient path.

Run (BOTH parts matter):
    cd /scratch/u6ex/as1748.u6ex/ARRWM
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONPATH=. \
      /scratch/u6ex/as1748.u6ex/miniforge3/envs/arrwm/bin/python \
      -m pytest -q testing/test_pixel_texture_disc.py

The default `python` has no torch.  And this node reports nproc=144, so torch
grabs 144 threads and thrashes on these small CPU convs: the suite looks hung
and takes >30 min.  With the three thread caps above it runs in ~21 s.
"""
from __future__ import annotations

import ast
import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.texture_stats import texture_battery  # noqa: E402
from model.disc_holdout_probe import _reduce_scores, roc_auc  # noqa: E402
from model.pixel_texture_disc import (  # noqa: E402
    B_BATTERY_REF,
    CONV_GEOMETRIC_RF,
    C_LATE_BATTERY_REF,
    PIX_DEFAULTS,
    PIX_GAN_WEIGHT_DEFAULT,
    PIX_R1_GAMMA,
    PIX_R1_GAMMA_CALIBRATION_P,
    PIX_R1_SIGMA,
    PixCounters,
    PixelTextureDisc,
    c1_calibration_verdict,
    c1_structured_hf,
    d_loss,
    effective_sample_count,
    g_loss,
    measure_receptive_field,
    patch_logit_separation,
    patch_logit_spatial_variance,
    patch_logit_telemetry,
    positive_control_readout,
    r1_penalty,
    r1_subsample_indices,
    resolve_gan_weight,
    roc_auc_fast,
    sweep_c1_amplitude,
    synthetic_stability_smoke,
)

MODULE_PATH = (Path(__file__).resolve().parents[1]
               / "model" / "pixel_texture_disc.py")
MODULE_SRC = MODULE_PATH.read_text()


def code_only(src: str) -> str:
    """Strip docstrings and comments, leaving executable code.

    The §5.3/§7 spec-drift tripwires must scan CODE, not prose.  The module
    *correctly* documents what it deliberately does not do ("there is no
    ``pix_r1_normalize`` knob", "do not reuse the ladd_disc penalty code",
    "this module never reads yaml") — a raw-source grep fires on the module's
    own explanation of its constraints, which would force the prose to be
    deleted to keep the tripwire green.  That trade is backwards: the prose is
    doing real work and §7 wants the tripwire.

    ``ast.unparse`` drops comments for free; docstrings are removed explicitly.
    Non-docstring string literals survive on purpose — a banned name smuggled
    into ``getattr(self, "pix_r2_gamma")`` is a real violation and must still
    be caught.  :class:`TestTripwireItself` proves the checker fires on a
    planted violation, so this is not a tripwire nobody has seen trip.
    """
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


MODULE_CODE = code_only(MODULE_SRC)


def code_identifiers(src: str) -> set:
    """Every real identifier in the source: names, attributes, args, defs,
    keyword-argument names.  Sharper than a substring grep — it distinguishes
    a banned *symbol* from a banned *word in a message*, which matters because
    the module's own ``ValueError`` texts name the things they reject.
    """
    out = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, ast.arg):
            out.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.keyword) and node.arg:
            out.add(node.arg)
        elif isinstance(node, ast.alias):
            out.add((node.asname or node.name).split(".")[-1])
    return out


MODULE_IDENTS = {n.lower() for n in code_identifiers(MODULE_SRC)}


# ===========================================================================
# 1. §4 architecture — shapes and the real P
# ===========================================================================
class TestArchitectureShapes(unittest.TestCase):
    def test_output_shape_is_n_1_h8_w8(self):
        net = PixelTextureDisc().eval()
        for h, w in [(64, 64), (128, 128), (176, 240), (192, 256), (96, 160)]:
            with self.subTest(size=(h, w)):
                out = net(torch.randn(2, 3, h, w))
                self.assertEqual(out.shape[:2], (2, 1))
                self.assertEqual(tuple(out.shape[-2:]), (h // 8, w // 8))
                self.assertEqual(
                    tuple(out.shape[-2:]), PixelTextureDisc.patch_grid(h, w))

    def test_P_is_660_at_the_real_crop_size(self):
        """THE TRAP.  The doc says 768; the truth after the border trim is 660.

        Shape trace, verified here end to end:
          pix_crop_lat=(24,32) -> 8x VAE decode -> 192x256
          -> §3.2 trim 8 px each side -> 176x240
          -> stride-8 patch grid -> 22 x 30 = 660.
        768 is 192/8 x 256/8, i.e. the UNTRIMMED crop.
        """
        lat_h, lat_w = PIX_DEFAULTS["pix_crop_lat"]
        self.assertEqual((lat_h, lat_w), (24, 32))
        dec_h, dec_w = lat_h * 8, lat_w * 8
        self.assertEqual((dec_h, dec_w), (192, 256))
        trim = PIX_DEFAULTS["pix_decode_border_trim"]
        h, w = dec_h - 2 * trim, dec_w - 2 * trim
        self.assertEqual((h, w), (176, 240))

        net = PixelTextureDisc().eval()
        out = net(torch.randn(1, 3, h, w))
        self.assertEqual(tuple(out.shape), (1, 1, 22, 30))
        self.assertEqual(int(out[0].numel()), 660)
        self.assertEqual(PixelTextureDisc.patch_count(h, w), 660)
        # The untrimmed crop is where 768 comes from.
        self.assertEqual(PixelTextureDisc.patch_count(192, 256), 768)

    def test_P_is_read_from_the_tensor_not_a_constant(self):
        """No function may hard-code P.  r1_penalty reports the measured one."""
        net = PixelTextureDisc().eval()
        for (h, w), expect in [((176, 240), 660), ((192, 256), 768),
                               ((88, 120), 11 * 15)]:
            res = r1_penalty(net, torch.randn(2, 3, h, w).clamp(-1, 1))
            self.assertEqual(res["P"], expect)

    def test_probe_reduce_scores_accepts_our_map(self):
        """Frozen contract: disc_holdout_probe needs ZERO edits."""
        net = PixelTextureDisc().eval()
        out = net(torch.randn(5, 3, 176, 240))
        red = _reduce_scores(out)
        self.assertEqual(tuple(red.shape), (5,))
        torch.testing.assert_close(
            red, out.detach().reshape(5, -1).mean(dim=1), rtol=1e-5, atol=1e-6)

    def test_rejects_non_4d_input(self):
        net = PixelTextureDisc().eval()
        with self.assertRaises(ValueError):
            net(torch.randn(3, 176, 240))


# ===========================================================================
# 2. §4 architecture — params, spectral norm, GroupNorm placement
# ===========================================================================
class TestArchitectureStructure(unittest.TestCase):
    def test_param_count_measured_661185_after_authorised_groupnorm_removal(self):
        """RE-MEASURED after the AUTHORISED GroupNorm removal (2026-08-23).

        Two separate discrepancies with §4, both recorded rather than
        "fixed" by changing the net:

        1. §4 claims ~2-3 M.  §4's own layer table gives well under 1 M.  The
           table is exact and is what is implemented.
        2. §4's table puts ``GroupNorm(8, ...)`` on blocks 2 and 3.  **The
           researcher authorised removing it on 2026-08-23**, after the normed
           critic was measured to have whole-image receptive field (see
           ``TestReceptiveField``).  This is a decision, not drift.

        The parameter delta from (2) is exactly **-768**, and it is auditable
        term by term — that is the point of asserting the breakdown rather
        than only the total:

          conv1 3->64    k4:   3*64*16  + 64  =   3,136
          conv2 64->128  k4:  64*128*16 + 128 = 131,200  GN(8,128) 256 REMOVED
          conv3 128->256 k4: 128*256*16 + 256 = 524,544  GN(8,256) 512 REMOVED
          conv4 256->1   k3:   256*1*9  + 1   =   2,305
                                               ---------
                                                 661,185   (was 661,953)

        A GroupNorm(8, C) carries ``2*C`` params (per-channel affine weight and
        bias; the group count does not enter), so the removal costs
        ``2*128 + 2*256 = 768`` exactly.
        """
        net = PixelTextureDisc()
        n = sum(p.numel() for p in net.parameters())
        self.assertEqual(n, 661_185)
        self.assertLess(n, 2_000_000, "doc's 2-3M claim is not met by §4's table")

        # The delta is exactly the two GroupNorms, and nothing else changed.
        normed = PixelTextureDisc(use_norm=True)
        n_normed = sum(p.numel() for p in normed.parameters())
        self.assertEqual(n_normed, 661_953, "the normed variant must still build")
        self.assertEqual(n_normed - n, 768)
        self.assertEqual(n_normed - n, 2 * 128 + 2 * 256)

        # Per-layer breakdown, so a future width change cannot hide inside the
        # total.  Spectral norm renames the weight, hence the `_raw` lookup.
        def conv_params(m):
            return sum(t.numel() for t in
                       (m.parametrizations.weight.original, m.bias))
        self.assertEqual(conv_params(net.conv1), 3 * 64 * 16 + 64)
        self.assertEqual(conv_params(net.conv2), 64 * 128 * 16 + 128)
        self.assertEqual(conv_params(net.conv3), 128 * 256 * 16 + 256)
        self.assertEqual(conv_params(net.conv4), 256 * 1 * 9 + 1)
        self.assertEqual(
            conv_params(net.conv1) + conv_params(net.conv2)
            + conv_params(net.conv3) + conv_params(net.conv4), 661_185,
            "the four convs alone must account for every parameter -- if they "
            "do not, something other than the GroupNorms was added or removed")

    def test_spectral_norm_on_all_four_convs(self):
        net = PixelTextureDisc()
        pnames = dict(net.named_parameters())
        for conv in ("conv1", "conv2", "conv3", "conv4"):
            with self.subTest(conv=conv):
                m = getattr(net, conv)
                self.assertTrue(
                    hasattr(m, "parametrizations")
                    and "weight" in m.parametrizations,
                    f"{conv} is not spectral-normalised",
                )
                # The parametrization API renames the raw weight; its presence
                # is what distinguishes a real reparametrisation from a
                # module that merely has the attribute.
                self.assertIn(f"{conv}.parametrizations.weight.original", pnames)

    def test_spectral_norm_actually_bounds_sigma(self):
        """Not just registered — the top singular value must be ~1.

        FLAKE FIXED (found while auditing this suite; it is why the baseline
        reads "31 subtests" on some runs and 32 on others).  The settling loop
        used to run on a ``.eval()`` network and its comment said "let power
        iteration settle".  It did not: ``parametrizations.spectral_norm``
        runs its power iteration **only in training mode**, so all 31 forwards
        were no-ops and the measured sigma was whatever the 15-iteration
        registration-time estimate happened to give.  Measured over 40 random
        inits, that overshoots ``delta=0.05`` on 2/40 for ``conv1`` and 1/40
        for ``conv2`` — a ~5 % failure rate that only shows up when the
        preceding tests leave the global RNG in the wrong place, which is
        exactly the intermittent failure seen.

        Settling in ``.train()`` really does iterate: max deviation over the
        same 40 inits drops from 0.082 to 0.019.  The seed makes the result
        reproducible rather than lucky; the ``delta`` is unchanged.
        """
        torch.manual_seed(0)
        net = PixelTextureDisc()
        net.train()                    # power iteration ONLY runs in train mode
        with torch.no_grad():
            for _ in range(31):
                net(torch.randn(2, 3, 64, 64))
        net.eval()
        for conv in ("conv1", "conv2", "conv3", "conv4"):
            with self.subTest(conv=conv):
                w = getattr(net, conv).weight.detach()
                sigma = torch.linalg.matrix_norm(
                    w.reshape(w.shape[0], -1), ord=2)
                self.assertAlmostEqual(float(sigma), 1.0, delta=0.05)

    def test_power_iteration_is_a_noop_in_eval_mode(self):
        """The fact the test above now relies on, pinned rather than assumed:
        forwards in ``.eval()`` do not advance the power iteration at all."""
        torch.manual_seed(0)
        net = PixelTextureDisc().eval()
        before = net.conv1.weight.detach().clone()
        with torch.no_grad():
            for _ in range(5):
                net(torch.randn(2, 3, 64, 64))
        self.assertTrue(torch.equal(before, net.conv1.weight.detach()),
                        "eval-mode forwards changed the spectral-normalised "
                        "weight; the settling-loop rationale would need "
                        "revisiting")
        net.train()
        with torch.no_grad():
            net(torch.randn(2, 3, 64, 64))
        self.assertFalse(torch.equal(before, net.conv1.weight.detach()),
                         "train-mode forward did NOT advance power iteration")

    def test_DEFAULT_critic_carries_NO_normalisation_authorised_deviation(self):
        """**AUTHORISED DEVIATION FROM §4 — researcher decision, 2026-08-23.**

        §4's layer table specifies ``GroupNorm(8, ...)`` on blocks 2 and 3.
        The shipped critic has **none**, and that is deliberate.  This test is
        the record: if it ever fails because someone "restored the spec", the
        restoration is the bug.

        WHY the researcher took the decision, in the terms it was taken:
        GroupNorm normalises over ``(C, H, W)`` per sample, so with it every
        patch logit is a function of every input pixel.  Measured at the real
        176x240 crop, 4.0-8.9 % of a centre patch logit's input-gradient L1
        mass fell OUTSIDE its 38x38 box and the non-zero-gradient support was
        the whole image (``TestReceptiveField`` pins this on the normed
        variant, which is still constructible).  B2's frozen scope question is
        *"can LOCAL decoded-pixel adversarial feedback suppress fabricated
        directional texture?"* — a global critic does not test it, and a null
        result from one would be uninterpretable.

        Nothing else in §4's table moved: four convs, 3->64->128->256->1,
        k4/k4/k4/k3, strides 2/2/2/1, padding 1, spectral norm on all four,
        LeakyReLU(0.2).  Those are asserted by the sibling tests.
        """
        net = PixelTextureDisc()
        self.assertFalse(net.uses_norm,
                         "the DEFAULT critic must ship norm-free")
        # The norm slots exist but are pass-throughs.
        self.assertIsInstance(net.norm2, nn.Identity)
        self.assertIsInstance(net.norm3, nn.Identity)
        # No normalisation layer of ANY kind anywhere -- not just no GroupNorm.
        # A BatchNorm/InstanceNorm/LayerNorm smuggled in later would couple
        # positions (or batch elements) just as badly and must fail here too.
        norms = [m for m in net.modules()
                 if isinstance(m, (nn.GroupNorm, nn.LayerNorm,
                                   nn.BatchNorm2d, nn.InstanceNorm2d))]
        self.assertEqual(norms, [],
                         f"default critic must carry no normalisation; found {norms}")
        self.assertFalse(hasattr(net, "norm1"))
        self.assertFalse(hasattr(net, "norm4"))
        # The forward really is norm-free, not merely norm-free by attribute:
        # an Identity cannot change the tensor it is handed.
        t = torch.randn(2, 128, 5, 7)
        self.assertTrue(torch.equal(net.norm2(t), t))

    def test_normed_variant_still_constructible_for_comparison(self):
        """The deviation removes the norm from the DEFAULT, not from the code.

        ``use_norm=True`` must keep building the OLD §4 network exactly —
        GroupNorm(8) on the two MIDDLE blocks only — because that variant is
        what the receptive-field planted-violation companion measures against.
        If this ever stops building, the locality guard silently loses its
        counter-example and starts proving nothing.
        """
        net = PixelTextureDisc(use_norm=True)
        self.assertTrue(net.uses_norm)
        self.assertIsInstance(net.norm2, nn.GroupNorm)
        self.assertIsInstance(net.norm3, nn.GroupNorm)
        self.assertEqual(net.norm2.num_groups, 8)
        self.assertEqual(net.norm3.num_groups, 8)
        self.assertEqual(net.norm2.num_channels, 128)
        self.assertEqual(net.norm3.num_channels, 256)
        # Block 1 and the head carry NO norm, exactly as §4 says.
        gns = [m for m in net.modules() if isinstance(m, nn.GroupNorm)]
        self.assertEqual(len(gns), 2, "GroupNorm must be on the 2 middle blocks only")
        self.assertFalse(hasattr(net, "norm1"))
        self.assertFalse(hasattr(net, "norm4"))
        # And it still runs, producing the same patch grid as the default.
        with torch.no_grad():
            self.assertEqual(net(torch.randn(2, 3, 176, 240)).shape,
                             (2, 1, 22, 30))

    def test_the_two_variants_differ_ONLY_by_the_norm(self):
        """The matched-init property the ``gsq``-magnitude comparison rests on.

        ``use_norm`` selects between ``GroupNorm`` and ``Identity``; neither
        construction path draws from the RNG (GroupNorm's affine init is
        ``ones_``/``zeros_``).  So at a fixed seed the four conv weights are
        **bit-identical** between the two builds, and any measured difference
        between them is caused by the norm and not by a different draw.

        ``TestR1.test_gsq_MAGNITUDE_collapsed_when_the_norm_was_removed``
        depends on this; without it that comparison would be confounded.
        """
        torch.manual_seed(0)
        a = PixelTextureDisc(use_norm=False)
        torch.manual_seed(0)
        b = PixelTextureDisc(use_norm=True)
        for conv in ("conv1", "conv2", "conv3", "conv4"):
            with self.subTest(conv=conv):
                wa = getattr(a, conv).parametrizations.weight.original
                wb = getattr(b, conv).parametrizations.weight.original
                self.assertTrue(torch.equal(wa, wb),
                                f"{conv} raw weights differ between the two "
                                "builds; the norm-on/norm-off comparison would "
                                "be confounded by the RNG")
                self.assertTrue(torch.equal(getattr(a, conv).bias,
                                            getattr(b, conv).bias))

    def test_no_global_scalar_output_option(self):
        """§4: per-patch logits are KEPT; there is no scalar mode and no flag.

        §5.1: no relativistic / RpGAN variant, "not even behind a flag".
        Code-only, so the prose that explains their absence is allowed.
        """
        # Checked on real IDENTIFIERS, not substrings: "relativistic" and
        # "RpGAN" DO appear in the module's ValueError text, which is the
        # message that REJECTS them as a loss form.  A word in a rejection
        # message is not a feature; a symbol would be.
        for banned in ("relativistic", "rpgan", "scalar_output", "pix_scalar",
                       "global_pool", "adaptive_avg_pool", "flatten"):
            offenders = [n for n in MODULE_IDENTS if banned in n]
            self.assertEqual(offenders, [], f"forbidden symbol(s): {offenders}")

        # And the rejection is real behaviour, not just an absent symbol.
        with self.assertRaises(ValueError):
            g_loss(torch.zeros(1, 1, 2, 2), "rpgan")

        # The output stays a patch map, never a scalar.
        net = PixelTextureDisc().eval()
        out = net(torch.randn(2, 3, 64, 64))
        self.assertEqual(out.dim(), 4, "output must stay a patch map")
        self.assertGreater(out.shape[-1] * out.shape[-2], 1)


# ===========================================================================
# 3. Measured receptive field
# ===========================================================================
class TestReceptiveField(unittest.TestCase):
    def test_conv_geometric_receptive_field(self):
        """RECORDED DISCREPANCY.  §4 claims ~70 px; the CONV STACK gives 38.

        70 px is the pix2pix PatchGAN, which has FIVE layers (k4s2 x3 then
        k4s1 x2).  §4's table has FOUR.  Closed form, back to front:
            k3s1 -> 3 ; k4s2 -> 8 ; k4s2 -> 18 ; k4s2 -> 38.

        NOTE this is the ``conv_rf_*`` half of the report — the network with
        GroupNorm DISABLED, which is not a network that is ever trained.  The
        shipped-configuration number is the test below.
        """
        rf = measure_receptive_field(input_size=129)
        self.assertEqual(rf["conv_rf_h"], 38)
        self.assertEqual(rf["conv_rf_w"], 38)
        self.assertEqual(rf["conv_rf_h"], CONV_GEOMETRIC_RF)
        self.assertEqual(rf["stride"], 8)
        self.assertNotEqual(rf["conv_rf_h"], 70)
        # ...and with the norm off the gradient really is confined to it.
        self.assertLess(rf["conv_off_patch_grad_frac"], 1e-3)

    def test_receptive_field_matches_closed_form(self):
        r = 1
        for k, s in [(3, 1), (4, 2), (4, 2), (4, 2)]:
            r = (r - 1) * s + k
        self.assertEqual(r, 38)
        self.assertEqual(measure_receptive_field(input_size=97)["conv_rf_h"], r)

    def test_DEFAULT_config_receptive_field_is_LOCAL_38px(self):
        """**THE ASSERTION THIS TEST MAKES IS THE INVERSE OF THE ONE IT USED
        TO MAKE, and the change is authorised.**

        History, so the inversion is legible.  This test used to be called
        ``test_TRAINING_config_receptive_field_is_GLOBAL_not_local`` and it
        pinned the *defect*: the shipped critic carried ``GroupNorm``, which
        normalises over ``(C, H, W)`` per sample, so every patch logit was a
        function of every input pixel and §4's "receptive field ~= 70 px — a
        local texture question" was false of what trained.  On **2026-08-23 the
        researcher authorised removing the GroupNorm** precisely to recover
        locality, because B2's frozen question is about *LOCAL* pixel
        adversarial feedback.  So the defect is gone and the test now pins its
        absence.  The old fact has not been deleted — it is asserted on the
        ``normed_*`` variant by
        ``test_the_normed_variant_is_still_measurably_GLOBAL`` below, which is
        also what keeps the planted-violation companion honest.

        RE-MEASURED at the real 176x240 post-trim crop, 12 random inits:

            DEFAULT (norm off) : bbox 38 x 38
                                 off-38x38 |grad| L1   < 2.4e-07
                                 off-38x38 grad^2      < 1.4e-07
            normed  (norm on)  : bbox 176 x 240 (ALL)
                                 off-38x38 |grad| L1   0.0399 .. 0.0886
                                 off-38x38 grad^2      5.07e-05 .. 2.90e-04

        The residual on the default is not merely small: over 12 inits it
        ranged -1.2e-07 .. +2.4e-07, i.e. it is not sign-definite — the
        signature of a true zero computed by subtracting two nearly-equal
        float32 sums, not of a small real coupling.

        BOTH L1 and squared energy are asserted because on the normed variant
        they disagreed by two orders of magnitude (4-9 % of L1 mass off-patch
        but only ~0.01-0.03 % of the energy).  Quoting energy alone would have
        made a whole-image critic look local, so the guard checks the measure
        that was hardest to pass as well as the flattering one.

        38 px, not §4's ~70: the pix2pix 70x70 PatchGAN has FIVE layers, §4's
        table has four.
        """
        rf = measure_receptive_field(176, input_size_w=240)
        # The report follows the class default; if the default were flipped
        # back to norm-on this would flip too and every assertion below fails.
        self.assertFalse(rf["norm_enabled"],
                         "measure_receptive_field must report the norm-free "
                         "DEFAULT as the primary configuration")
        self.assertFalse(PixelTextureDisc().uses_norm)

        # The support is exactly the conv geometry -- the whole point.
        self.assertEqual((rf["rf_h"], rf["rf_w"]), (38, 38))
        self.assertEqual((rf["input_h"], rf["input_w"]), (176, 240))
        self.assertEqual(rf["rf_h"], CONV_GEOMETRIC_RF)
        self.assertEqual((rf["rf_h"], rf["rf_w"]),
                         (rf["conv_rf_h"], rf["conv_rf_w"]),
                         "with the norm gone the effective RF must equal the "
                         "conv geometry")
        # Not the whole image any more.
        self.assertNotEqual((rf["rf_h"], rf["rf_w"]), (176, 240))

        # The globality is GONE, in both measures.  1e-4 is ~400x above the
        # measured float-noise ceiling (2.4e-07) and ~400x BELOW the normed
        # variant's L1 figure, so it separates the two cleanly and is not a
        # window so wide it could not fail.
        self.assertLess(abs(rf["off_patch_grad_frac"]), 1e-4)
        self.assertLess(abs(rf["off_patch_grad_energy_frac"]), 1e-4)

    def test_the_normed_variant_is_still_measurably_GLOBAL(self):
        """The measurement that JUSTIFIED the deviation, kept alive.

        If the normed variant ever stopped measuring as global, the stated
        reason for removing the GroupNorm would be unsupported and this whole
        change would need revisiting.  So the old finding is still asserted —
        on ``normed_*`` — rather than deleted along with the old default.
        MEASURED: bbox 176x240 (the full image) and 4.0-8.9 % of a centre
        patch logit's input-gradient L1 mass outside its 38x38 box, against
        ~1e-7 for the norm-free default.
        """
        rf = measure_receptive_field(176, input_size_w=240)
        self.assertEqual((rf["normed_rf_h"], rf["normed_rf_w"]), (176, 240))
        self.assertGreater(rf["normed_off_patch_grad_frac"], 0.01)
        self.assertLess(rf["normed_off_patch_grad_frac"], 0.5)
        # ...and by orders of magnitude more than the shipped configuration.
        self.assertGreater(rf["normed_off_patch_grad_frac"],
                           1000.0 * max(abs(rf["off_patch_grad_frac"]), 1e-12))
        # The broad-and-weak signature: the energy share is ~2 orders of
        # magnitude below the L1 share.  This is why the guard above checks
        # both -- energy alone would have understated the problem.
        self.assertLess(rf["normed_off_patch_grad_energy_frac"],
                        0.01 * rf["normed_off_patch_grad_frac"])
        self.assertGreater(rf["normed_off_patch_grad_energy_frac"], 1e-6)

    def test_the_locality_guard_FIRES_against_the_normed_variant(self):
        """PLANTED-VIOLATION COMPANION, **flipped with the assertion it guards**.

        It used to plant the flattering norm-OFF number against a guard
        asserting globality.  The guard now asserts LOCALITY, so the planted
        violation is the norm-ON variant — the configuration the deviation
        removed.  Running the locality assertions against it must FAIL, or the
        guard above proves nothing and would stay green on a critic that had
        silently regained a whole-image receptive field.
        """
        rf = measure_receptive_field(176, input_size_w=240)
        planted_l1 = rf["normed_off_patch_grad_frac"]
        planted_energy = rf["normed_off_patch_grad_energy_frac"]
        planted_bbox = (rf["normed_rf_h"], rf["normed_rf_w"])

        # Exactly the assertions of test_DEFAULT_config_receptive_field_is_
        # LOCAL_38px, run against the normed variant.  Each must fail.
        with self.assertRaises(AssertionError):
            self.assertLess(abs(planted_l1), 1e-4)
        with self.assertRaises(AssertionError):
            self.assertLess(abs(planted_energy), 1e-4)
        with self.assertRaises(AssertionError):
            self.assertEqual(planted_bbox, (38, 38))

        # And the bbox check the other way round: the DEFAULT must not pass a
        # whole-image assertion, so the two configurations really are being
        # told apart rather than both sliding through.
        with self.assertRaises(AssertionError):
            self.assertEqual((rf["rf_h"], rf["rf_w"]), (176, 240))


# ===========================================================================
# 3b. §3.7 effective sample count, recomputed on the MEASURED receptive field
# ===========================================================================
class TestEffectiveSampleCount(unittest.TestCase):
    """§3.7's arithmetic, redone on the 38 px RF the critic actually has.

    §3.7 tabulates 768 patch logits, ~6 non-overlapping receptive-field tiles
    per image and a ~128:1 overlap factor.  Both inputs were wrong:

    * 768 is the UNTRIMMED 192/8 x 256/8; the §3.2 8-px border trim makes the
      real crop 176x240 and ``P = 22 x 30 = 660``;
    * 6 tiles follows from §4's phantom ~70 px RF
      (``floor(176/70) * floor(240/70) = 2 * 3 = 6``).  The RF is 38 px — and
      before the GroupNorm was removed the *effective* support was the whole
      image, which makes the tile count 1 and the entire table meaningless.

    Both are now measurable, so the table is worth recomputing.
    """

    def test_tiles_and_overlap_corrected(self):
        e = effective_sample_count(176, 240)
        self.assertEqual(e["patch_logits_per_image"], 660)
        self.assertEqual(e["rf_px"], 38)
        self.assertEqual(e["rf_px"], CONV_GEOMETRIC_RF)
        # floor(176/38)=4, floor(240/38)=6
        self.assertEqual((e["tiles_h"], e["tiles_w"]), (4, 6))
        self.assertEqual(e["tiles_per_image"], 24)
        self.assertAlmostEqual(e["overlap_factor"], 660 / 24, places=6)
        self.assertAlmostEqual(e["overlap_factor"], 27.5, places=6)
        # 4x more tiles and ~4.7x less overlap than §3.7 claimed.
        self.assertEqual(e["doc_tiles_per_image"], 6)
        self.assertEqual(e["tiles_per_image"], 4 * e["doc_tiles_per_image"])
        self.assertLess(e["overlap_factor"], 0.25 * e["doc_overlap_factor"])
        # Area-ratio cross-check on the integer tiling (29.25 vs 24: the
        # integer tiling wastes the 176 % 38 = 24 px and 240 % 38 = 12 px
        # margins, so it is the CONSERVATIVE of the two and is what is
        # reported as the headline).
        self.assertAlmostEqual(e["tiles_per_image_area_ratio"],
                               (176 * 240) / (38 * 38), places=6)
        self.assertLess(e["tiles_per_image"], e["tiles_per_image_area_ratio"])
        # Patches sharing any given pixel: (38/8)^2.
        self.assertAlmostEqual(e["patches_sharing_a_pixel"], (38 / 8) ** 2,
                               places=6)

    def test_the_RF_the_arithmetic_uses_is_the_MEASURED_one(self):
        """The correction is only worth anything if its RF input is measured
        rather than asserted.  This ties the two together: if the receptive
        field ever changes, this test fails and the sample-count table cannot
        drift away from it silently."""
        rf = measure_receptive_field(176, input_size_w=240)
        e = effective_sample_count(176, 240)
        self.assertEqual(e["rf_px"], rf["rf_h"])
        self.assertEqual(e["rf_px"], rf["rf_w"])
        self.assertEqual(e["patch_logits_per_image"],
                         int(rf["grid_h"] * rf["grid_w"]))

    def test_spec_config_already_exceeds_the_escalation_target(self):
        """The launch-configuration consequence, which is the reason this
        arithmetic is in the module at all.

        §3.7's fake-side table (effective = images/step x tiles/image):

            draft  crops=2 frames=2 ->  4 images -> doc ~24, MEASURED  96
            SPEC   crops=4 frames=3 -> 12 images -> doc ~72, MEASURED 288
            raised crops=8 frames=3 -> 24 images -> doc ~144, MEASURED 576

        So the specced ``pix_crops_per_step=4`` already delivers 288 — 2x
        §3.7's own ``crops=8`` escalation target of ~144.  The "raise to 8 if
        sample-starved" trigger therefore has NO arithmetic support at launch;
        pulling it needs the §7 patch-logit variance or the §8.1 control to say
        so.
        """
        spec = effective_sample_count(176, 240, crops_per_step=4,
                                      frames_per_crop=3)
        self.assertEqual(spec["images_per_step"], 12)
        self.assertEqual(spec["patch_logits_per_step"], 12 * 660)
        self.assertEqual(spec["effective_per_step"], 12 * 24)
        self.assertEqual(spec["effective_per_step"], 288)
        self.assertEqual(spec["doc_effective_per_step"], 72)
        self.assertEqual(spec["effective_per_step"],
                         4 * spec["doc_effective_per_step"])

        raised = effective_sample_count(176, 240, crops_per_step=8,
                                        frames_per_crop=3)
        self.assertEqual(raised["doc_effective_per_step"], 144)
        self.assertGreater(spec["effective_per_step"],
                           2.0 * raised["doc_effective_per_step"] - 1)

        # The honest caveat, pinned so the headline cannot be quoted alone:
        # tiles within a crop share scene/exposure/VAE reconstruction, and
        # §3.4a refuses to count frame-within-crop on the real side at all.
        self.assertEqual(spec["effective_source_independent"], 4 * 24)
        self.assertLess(spec["effective_source_independent"],
                        spec["effective_per_step"])
        self.assertEqual(spec["independent_scenes_per_step"], 4)
        self.assertEqual(spec["independent_scenes_per_step"],
                         spec["crops_per_step"],
                         "independent SCENES per step is crops, and the RF "
                         "correction does not change it")


# ===========================================================================
# 4. §5.1 losses
# ===========================================================================
class TestLosses(unittest.TestCase):
    def setUp(self):
        # Deliberately BIMODAL patch map: mean 0, but softplus(mean) != mean(softplus).
        self.real = torch.tensor([[[[-4.0, 4.0], [4.0, -4.0]]]])
        self.fake = torch.tensor([[[[-3.0, 3.0], [3.0, -3.0]]]])

    def test_nsgan_d_loss_hand_computed(self):
        got = d_loss(self.real, self.fake, "nsgan")
        sp = lambda t: math.log1p(math.exp(t))
        want_real = (sp(4.0) + sp(-4.0) + sp(-4.0) + sp(4.0)) / 4.0
        want_fake = (sp(-3.0) + sp(3.0) + sp(3.0) + sp(-3.0)) / 4.0
        self.assertAlmostEqual(float(got["d_real_term"]), want_real, places=5)
        self.assertAlmostEqual(float(got["d_fake_term"]), want_fake, places=5)
        self.assertAlmostEqual(float(got["d_loss"]), want_real + want_fake, places=5)
        self.assertAlmostEqual(float(got["d_real_mean"]), 0.0, places=6)
        self.assertAlmostEqual(float(got["d_fake_mean"]), 0.0, places=6)

    def test_nsgan_g_loss_hand_computed(self):
        sp = lambda t: math.log1p(math.exp(t))
        want = (sp(3.0) + sp(-3.0) + sp(-3.0) + sp(3.0)) / 4.0
        self.assertAlmostEqual(float(g_loss(self.fake, "nsgan")), want, places=5)

    def test_hinge_d_loss_hand_computed(self):
        got = d_loss(self.real, self.fake, "hinge")
        want_real = (5.0 + 0.0 + 0.0 + 5.0) / 4.0          # relu(1 - r)
        want_fake = (0.0 + 4.0 + 4.0 + 0.0) / 4.0          # relu(1 + f)
        self.assertAlmostEqual(float(got["d_real_term"]), want_real, places=6)
        self.assertAlmostEqual(float(got["d_fake_term"]), want_fake, places=6)

    def test_hinge_g_loss_is_negative_mean(self):
        """Recorded ambiguity: -mean(D(fake)), the standard hinge G loss."""
        self.assertAlmostEqual(float(g_loss(self.fake, "hinge")), -0.0, places=6)
        self.assertAlmostEqual(
            float(g_loss(torch.tensor([[[[2.0, 4.0]]]]), "hinge")), -3.0, places=6)

    def test_nonlinearity_applied_BEFORE_averaging(self):
        """FAILS if anyone ever averages the logits first (Jensen gap).

        The patch map is bimodal at +-4 with mean exactly 0, so:
            mean(softplus(-D)) = (softplus(-4) + softplus(4)) / 2
                               = (0.018150 + 4.018150) / 2 = 2.018150  CORRECT
            softplus(-mean(D)) = softplus(0) = ln 2       = 0.693147    WRONG
        A 2.9x gap.  Averaging logits first is exactly the scalar-critic
        collapse §4 exists to avoid, and it is invisible on a unimodal map —
        which is why this fixture is deliberately bimodal.
        """
        per_patch = float(g_loss(self.real, "nsgan"))
        post_mean = float(F.softplus(-self.real.mean()))
        want = 0.5 * (math.log1p(math.exp(-4.0)) + math.log1p(math.exp(4.0)))
        self.assertAlmostEqual(want, 2.018150, places=6)
        self.assertAlmostEqual(per_patch, want, places=6)
        self.assertAlmostEqual(post_mean, math.log(2.0), places=6)
        self.assertGreater(per_patch, 2.0)
        self.assertGreater(per_patch / post_mean, 2.5)

    def test_hinge_also_before_averaging(self):
        d_pp = float(d_loss(self.real, self.fake, "hinge")["d_real_term"])
        d_pm = float(F.relu(1.0 - self.real.mean()))
        self.assertAlmostEqual(d_pm, 1.0, places=6)
        self.assertAlmostEqual(d_pp, 2.5, places=6)

    def test_mean_covers_batch_rows_and_patch_grid(self):
        """One mean over BOTH axes -- rows with more patches don't get
        double-counted, and a 2-row batch is the mean of its rows here."""
        a = torch.zeros(1, 1, 2, 2)
        b = torch.full((1, 1, 2, 2), 10.0)
        both = torch.cat([a, b], dim=0)
        self.assertAlmostEqual(
            float(g_loss(both, "nsgan")),
            0.5 * (float(g_loss(a, "nsgan")) + float(g_loss(b, "nsgan"))),
            places=6)

    def test_unknown_loss_form_rejected(self):
        for bad in ("rpgan", "relativistic", "wgan", "lsgan"):
            with self.subTest(form=bad):
                with self.assertRaises(ValueError):
                    d_loss(self.real, self.fake, bad)
                with self.assertRaises(ValueError):
                    g_loss(self.fake, bad)


# ===========================================================================
# 5. §5.3 R1
# ===========================================================================
class _SmoothNet(nn.Module):
    """Tiny smooth patch critic (no ReLU kinks) so the FD/autograd comparison
    is limited by O(sigma) truncation, not by non-differentiability."""

    def __init__(self, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w = nn.Parameter(torch.randn(4, 3, 3, 3, generator=g) * 0.3)
        self.v = nn.Parameter(torch.randn(1, 4, 3, 3, generator=g) * 0.3)

    def forward(self, x):
        h = torch.tanh(F.conv2d(x, self.w, stride=2, padding=1))
        return F.conv2d(h, self.v, padding=1)


#: Crops whose patch counts span 16x: 11x15=165, 22x30=660, 44x60=2640.
_SCALING_SIZES = ((88, 120), (176, 240), (352, 480))


def _pix_disc_for_scaling(seed: int = 0):
    """A REAL ``PixelTextureDisc`` at a fixed seed (so the P-scaling numbers
    below are deterministic), with grads off — we differentiate w.r.t. the
    INPUT here, never the weights."""
    torch.manual_seed(seed)
    net = PixelTextureDisc().eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def _scaling_content(n_img: int = 4):
    """One fixed smooth+texture canvas at the largest size; the smaller crops
    are centre-cuts of it, so P changes and the CONTENT statistics do not."""
    h, w = _SCALING_SIZES[-1]
    g = torch.Generator().manual_seed(77)
    yy = torch.linspace(-1, 1, h).view(1, 1, h, 1)
    xx = torch.linspace(-1, 1, w).view(1, 1, 1, w)
    base = (0.6 * torch.sin(3 * yy) + 0.6 * torch.cos(4 * xx))
    base = base.expand(n_img, 3, h, w).clone()
    return (base + 0.15 * torch.randn((n_img, 3, h, w), generator=g)).clamp(-1, 1)


def _centre_crop(x, h, w):
    y0 = (x.shape[-2] - h) // 2
    x0 = (x.shape[-1] - w) // 2
    return x[:, :, y0:y0 + h, x0:x0 + w].contiguous()


def _grad_norm_sq_vs_P(reduce: str = "mean"):
    """``[(P, ||d/dx reduce_i D_i(x)||^2 averaged over images)]`` per crop size.

    This is ``E_eps[gsq]`` for ``reduce="mean"`` — the exact expectation of the
    shipped finite-difference estimator (the identity pinned by
    ``test_fd_expectation_is_grad_norm_squared``), so fitting the P exponent on
    it carries no Monte-Carlo noise.  ``reduce="sum"`` is §5.3's stated
    counterfactual, computed the same way.
    """
    net = _pix_disc_for_scaling()
    content = _scaling_content()
    rows = []
    for h, w in _SCALING_SIZES:
        x = _centre_crop(content, h, w).requires_grad_(True)
        logits = net(x)
        flat = logits.reshape(logits.shape[0], -1)
        s = flat.mean(dim=1) if reduce == "mean" else flat.sum(dim=1)
        gx, = torch.autograd.grad(s.sum(), x)
        rows.append((int(logits[0].numel()),
                     float((gx ** 2).sum(dim=(1, 2, 3)).mean())))
    return rows


def _log_log_slope(rows):
    """OLS slope of ``log(value)`` on ``log(P)`` — the exponent ``a`` in
    ``value ~ P^a``."""
    lx = [math.log(p) for p, _ in rows]
    ly = [math.log(v) for _, v in rows]
    n = len(rows)
    mx = sum(lx) / n
    my = sum(ly) / n
    num = sum((lx[i] - mx) * (ly[i] - my) for i in range(n))
    den = sum((lx[i] - mx) ** 2 for i in range(n))
    return num / den


class TestR1(unittest.TestCase):
    def test_fd_matches_autograd_directional_derivative(self):
        """(a) The FD estimator is a first-order approximation of
        ``(eps . d/dx mean_i D_i)^2``; with the SAME eps the two agree to
        O(sigma).  ``E_eps[(eps.g)^2] = ||g||^2`` is the identity the estimator
        rests on, checked separately below."""
        torch.manual_seed(0)
        net = _SmoothNet().double()
        x = (torch.rand(6, 3, 32, 32, dtype=torch.float64) * 2 - 1)
        gen = torch.Generator().manual_seed(11)
        eps = torch.randn(x.shape, generator=gen, dtype=torch.float64)

        sigma = 1e-4
        fd = r1_penalty(net, x, gamma=1.0, sigma=sigma, eps=eps)["gsq"]

        xa = x.clone().requires_grad_(True)
        s = net(xa).reshape(xa.shape[0], -1).mean(dim=1)
        gx, = torch.autograd.grad(s.sum(), xa)
        directional = ((gx * eps).sum(dim=(1, 2, 3)) ** 2).mean()
        self.assertAlmostEqual(fd / float(directional), 1.0, places=3)

    def test_fd_expectation_is_grad_norm_squared(self):
        """The autograd reference the doc names: ``||d/dx mean_i D_i||^2``.
        Averaged over eps draws, the FD estimator converges to it."""
        torch.manual_seed(0)
        net = _SmoothNet(seed=3).double()
        x = (torch.rand(4, 3, 24, 24, dtype=torch.float64) * 2 - 1)
        xa = x.clone().requires_grad_(True)
        s = net(xa).reshape(xa.shape[0], -1).mean(dim=1)
        gx, = torch.autograd.grad(s.sum(), xa)
        ref = float((gx ** 2).sum(dim=(1, 2, 3)).mean())

        gen = torch.Generator().manual_seed(5)
        acc = 0.0
        n = 400
        for _ in range(n):
            e = torch.randn(x.shape, generator=gen, dtype=torch.float64)
            acc += r1_penalty(net, x, sigma=1e-4, eps=e)["gsq"]
        self.assertAlmostEqual(acc / n / ref, 1.0, delta=0.15)

    def test_gamma_is_P_DEPENDENT_measured_exponent(self):
        """(b) **THIS TEST REFUTES §5.3.**  It replaces a test called
        ``test_scale_portability_in_P`` that claimed to pin scale-freedom and
        could not have failed if the property were absent: its window was
        ``0.05 < ratio < 4.0`` (80x wide, for a quantity called *invariant*,
        and the measured 4x-P ratio of 0.265 sailed through); it ran a
        ``_SmoothNet`` stand-in instead of ``PixelTextureDisc``; and its
        closing assertion, commented "so the test is not vacuous", reduced
        algebraically to ``P^2 x / P^2 y == x / y`` — an identity on random
        floats with no network in it at all.

        WHAT IS ACTUALLY TRUE, RE-MEASURED on the REAL ``PixelTextureDisc``
        after the 2026-08-23 authorised GroupNorm removal:

            mean reduction (shipped):  gsq ~ P^-0.879  [this seed]
            sum  reduction (spec's
              stated counterfactual):  gsq ~ P^+1.121  [this seed]

        §5.3 asserts P^0 and P^+2 respectively.  Both are wrong, and the
        mechanism is not subtle: patch gradients are local and nearly
        independent, so ``||sum_i grad D_i||^2 ~= P ||grad D_i||^2``; the mean
        divides by P^2, leaving ~1/P.  The doc's arithmetic is the perfectly
        CORRELATED-patch case, which a patch critic is built not to be.

        **ALPHA DID NOT MOVE MATERIALLY WHEN THE NORM WAS REMOVED — and that
        is the answer to the question that motivated re-fitting it.**  The
        worry was that GroupNorm might have been *why* patch gradients looked
        near-independent, in which case the ~1/P law would have been an
        artefact of the norm and would have collapsed with it.  It was not:

            with GroupNorm  : alpha = -1.094 [seed 0];
                              10 inits -1.282 .. -0.939, mean -1.056
            without (SHIPS) : alpha = -0.879 [seed 0];
                              10 inits -1.047 .. -0.853, mean -0.947

        Same ~1/P regime, and the init-to-init spread actually TIGHTENED
        (0.194 wide against 0.343).  So the mechanism quoted above — "patch
        gradients are LOCAL and nearly independent" — was previously the right
        answer stated about a critic that was provably NOT local (its gradient
        support was the whole image).  It now describes the network it is
        stated about, which is the first time the explanation and the
        measurement have agreed.

        **WHAT DID CHANGE IS THE MAGNITUDE, BY ~100x**, and that is the
        launch-critical one:
        ``test_gsq_MAGNITUDE_collapsed_when_the_norm_was_removed`` pins it.
        The exponent describes how ``gsq`` scales with crop size; the
        magnitude decides whether ``R1 = 0.5 * gamma * gsq`` does anything at
        all.

        Consequence, and the reason this test is named for gamma: **gamma is
        tied to the crop size it was calibrated at.**  At the spec crop
        (P = 660) gamma = 1.0 stands; at a 2x LINEAR crop (P x4) the same
        gamma delivers ~3.6x weaker R1.

        Fitted here on the EXPECTATION of the shipped estimator,
        ``E_eps[gsq] = ||d/dx mean_i D_i||^2`` (the identity pinned by
        ``test_fd_expectation_is_grad_norm_squared``), so the fit carries no
        Monte-Carlo noise.  The Monte-Carlo estimator itself is checked in the
        companion test below.  Fixed net seed => deterministic; over 10
        random inits the measured mean-reduction alpha ranges -1.28..-0.94
        (mean -1.06) and the sum-reduction alpha +0.72..+1.06 (mean +0.94),
        which is what the tolerances below are drawn from.
        """
        rows_mean = _grad_norm_sq_vs_P(reduce="mean")
        rows_sum = _grad_norm_sq_vs_P(reduce="sum")
        self.assertEqual([p for p, _ in rows_mean], [165, 660, 2640])

        a_mean = _log_log_slope(rows_mean)
        a_sum = _log_log_slope(rows_sum)

        # The headline: NOT scale-free.  alpha=0 is the spec's claim and it is
        # nowhere near the measurement.
        self.assertLess(a_mean, -0.5,
                        f"gsq is claimed scale-free in P; measured alpha={a_mean:.3f}")
        self.assertGreater(a_mean, -1.4, f"alpha={a_mean:.3f} outside measured range")
        # RE-FITTED on the norm-free default: -0.879 at this seed.  The delta
        # spans the 10-init range measured WITHOUT the norm (-1.047 .. -0.853)
        # and still excludes the spec's alpha=0 by a wide margin.
        self.assertAlmostEqual(a_mean, -0.879, delta=0.25)
        # And the norm removal did not knock it out of the regime it was in
        # with the norm (-1.056 mean).  Stated as an explicit range check so
        # the "did not move materially" claim is a measurement, not prose.
        self.assertTrue(-1.3 < a_mean < -0.7,
                        f"alpha={a_mean:.3f} left the ~1/P regime; the claim "
                        "that removing GroupNorm did not change the P-scaling "
                        "would need revisiting")

        # And the spec's stated counterfactual is wrong too: the sum scales
        # ~P^1, not ~P^2.
        self.assertGreater(a_sum, 0.5, f"alpha_sum={a_sum:.3f}")
        self.assertLess(a_sum, 1.5,
                        f"sum reduction is claimed to scale as P^2; "
                        f"measured alpha={a_sum:.3f}")

        # Exactly the identity the OLD test mistook for evidence, stated as
        # what it is: an algebraic tautology (sum = P * mean), NOT a
        # measurement.  It holds for any numbers whatsoever, which is why it
        # can never fail and must never be cited as a check.
        for (p, g_mean), (p2, g_sum) in zip(rows_mean, rows_sum):
            self.assertEqual(p, p2)
            self.assertAlmostEqual(g_sum / (p * p * g_mean), 1.0, places=4)

        # The practical statement, in the units a reader needs: gamma tuned at
        # the spec crop is ~3.6x too weak one linear doubling up.
        self.assertEqual(PIX_R1_GAMMA_CALIBRATION_P, 660)
        rescale = 4.0 ** (-a_mean)
        self.assertGreater(rescale, 2.0)

    def test_shipped_r1_penalty_shows_the_same_P_dependence(self):
        """The companion to the fit above, on the SHIPPED Monte-Carlo path.

        The exponent is fitted on the estimator's expectation; this checks the
        thing the trainer actually calls.  ``r1_penalty``'s own ``gsq`` at
        P=2640 is a small FRACTION of its value at P=165 — if the mean
        reduction were scale-free the ratio would be ~1.0.

        Measured across 5 eps seeds: 0.050 .. 0.114 (exact expectation
        0.048).  Bound set at 0.25, i.e. ~2.5x looser than the worst draw and
        4x away from the value the spec's claim predicts.
        """
        net = _pix_disc_for_scaling()
        content = _scaling_content()
        vals = {}
        for h, w in ((88, 120), (352, 480)):
            x = _centre_crop(content, h, w)
            gen = torch.Generator().manual_seed(1234)
            tot = 0.0
            reps = 6
            for _ in range(reps):
                r = r1_penalty(net, x, generator=gen)
                tot += r["gsq"]
                P = r["P"]
            vals[P] = tot / reps
        self.assertEqual(sorted(vals), [165, 2640])
        ratio = vals[2640] / vals[165]
        self.assertLess(ratio, 0.25,
                        f"gsq at P=2640 is {ratio:.3f} of its P=165 value; "
                        "a scale-free gsq would give ~1.0")

    def test_gsq_MAGNITUDE_collapsed_when_the_norm_was_removed(self):
        """**LAUNCH-CRITICAL.**  Removing the GroupNorm left the P-scaling
        EXPONENT alone but dropped the ``gsq`` MAGNITUDE by ~100x, which makes
        the inherited ``pix_r1_gamma = 1.0`` deliver ~100x less R1 than it did
        when that number was written down.

        This is exactly the "value present, effect absent" failure class this
        work package keeps finding: ``pix_r1_gamma`` still reads 1.0, the R1
        term still appears in the loss, ``pix_r1_rate`` still reads 1.00 — and
        the penalty does essentially nothing.  Nothing in the config surface
        would show it.  Hence a test.

        MEASURED, ``E[gsq]`` at P=660 on the real 176x240 crop, 10 MATCHED
        random inits (matched exactly: ``use_norm`` selects GroupNorm vs
        Identity and neither draws from the RNG, so the four conv weights are
        bit-identical between the builds —
        ``test_the_two_variants_differ_ONLY_by_the_norm`` asserts that, and
        without it this comparison would be confounded by a different draw):

            norm ON   1.244e-03 .. 2.662e-03
            norm OFF  1.015e-05 .. 2.120e-05
            ratio     95x .. 214x, mean 129x

        A different content canvas gave 195x at seed 0, so the finding is
        "~10^2, content- and init-dependent", not a constant — the bounds below
        are set accordingly.  Mechanism: GroupNorm rescales activations and
        inflates the input-gradient scale with them.

        **CORRECTED PREMISE (2026-08-23) — the stronger, worse fact.**  This
        test previously asserted, via a planted companion, that R1 at
        ``gamma = 1.0`` was NOT negligible under the norm, "which is why
        gamma=1.0 was ever a defensible number".  That is FALSE, and the
        companion could not fail.  Re-measured here, ``R1 / d_loss`` at init
        (``d_loss = 2*ln2 = 1.3863``) at ``gamma = 1.0``, same 10 matched
        inits:

            norm ON    4.485e-04 .. 9.603e-04   (seed 0/1/2:
                                                 6.488e-04, 5.958e-04,
                                                 8.741e-04)
            norm OFF   3.660e-06 .. 7.646e-06   (seed 0/1/2:
                                                 5.257e-06, 6.264e-06,
                                                 7.646e-06)

        Every normed value is BELOW 0.1 % of ``d_loss``.  So ``gamma = 1.0``
        was already delivering under a tenth of a percent of ``d_loss``
        BEFORE the GroupNorm was removed; the removal took it to ~0.0005 %.
        The collapse is a ~100x worsening of an ALREADY-INERT term, not the
        cause of the inertness — ``pix_r1_gamma = 1.0`` is inert on BOTH
        architectures.  Stated as a calibration target: the gamma that reaches
        0.1 % of ``d_loss`` is 1.04 .. 2.23 with the norm and 131 .. 273
        without it.  The practical consequence is that re-calibration cannot
        be framed as restoring a pre-removal value; there is no working
        earlier value to restore.

        Independently re-measured 2026-08-23 on a fresh process (10 matched
        inits, same content canvas), reproducing the table above to every
        printed digit: ratios 95.1, 123.4, 214.2 at the extremes; ``g*_on``
        1.041 .. 2.230; ``g*_off`` 130.8 .. 273.2.

        The replacement companion is (iii) below.  The old one asserted a
        false premise and so could never raise; the new one plants the SAME
        ``frac(gsq, gamma) < 1e-3`` assertion at fixed gammas (5.0 normed,
        600.0 shipped) read off the measurement rather than derived from
        ``gsq``, and requires it to raise -- which makes it a lower bound on
        ``gsq``, not algebra.  Two mutations were run to confirm it can fail:
        shrinking the planted gamma to 0.5, and scaling ``gsq`` by 0.05 to
        simulate a further collapse.  Both fail the test.

        WHAT TO DO ABOUT IT is a researcher decision and this module does not
        make it: §5.3's own procedure is to run briefly, read
        ``pix_r1_grad_sq_mean`` (:class:`PixCounters` — one row is 126 % noise)
        and set ``gamma`` from that.  ``PIX_R1_GAMMA`` carries an
        inert-gamma warning pointing here.
        """
        # Fixed content, so the only thing varying is the norm.
        g = torch.Generator().manual_seed(77)
        yy = torch.linspace(-1, 1, 176).view(1, 1, 176, 1)
        xx = torch.linspace(-1, 1, 240).view(1, 1, 1, 240)
        base = (0.6 * torch.sin(3 * yy) + 0.6 * torch.cos(4 * xx))
        base = base.expand(4, 3, 176, 240).clone()
        content = (base + 0.15 * torch.randn((4, 3, 176, 240), generator=g)
                   ).clamp(-1, 1)

        def e_gsq(seed, use_norm):
            """Exact expectation of the shipped estimator,
            ``E_eps[gsq] = ||d/dx mean_i D_i||^2`` (the identity pinned by
            ``test_fd_expectation_is_grad_norm_squared``) — no MC noise."""
            torch.manual_seed(seed)
            net = PixelTextureDisc(use_norm=use_norm).eval()
            for prm in net.parameters():
                prm.requires_grad_(False)
            x = content.clone().requires_grad_(True)
            logits = net(x)
            self.assertEqual(int(logits[0].numel()), 660)   # P, off the tensor
            s = logits.reshape(logits.shape[0], -1).mean(dim=1)
            gx, = torch.autograd.grad(s.sum(), x)
            return float((gx ** 2).sum(dim=(1, 2, 3)).mean())

        ratios = []
        on_vals, off_vals = [], []
        for seed in (0, 1, 2):
            with self.subTest(seed=seed):
                on = e_gsq(seed, True)
                off = e_gsq(seed, False)
                on_vals.append(on)
                off_vals.append(off)
                self.assertGreater(on, 0.0)
                self.assertGreater(off, 0.0)
                r = on / off
                ratios.append(r)
                # Two orders of magnitude, every seed.  The bounds bracket the
                # measured 95..214 with room for init noise, and a ratio of
                # ~1.0 -- i.e. "the norm made no difference to the magnitude",
                # which is what a silent restoration of the norm or an
                # accidental rescale would produce -- fails on the low side.
                self.assertGreater(
                    r, 20.0,
                    f"seed {seed}: gsq ratio norm_on/norm_off = {r:.1f}; the "
                    "measured ~100x magnitude collapse is not reproducing")
                self.assertLess(r, 1000.0, f"seed {seed}: ratio {r:.1f}")

        mean_ratio = sum(ratios) / len(ratios)
        self.assertGreater(mean_ratio, 50.0)
        self.assertLess(mean_ratio, 500.0)

        # The per-seed bounds and the mean bound above ARE the ratio guards.
        # A stray `self.assertGreater(ratio, 0.005)` used to sit at the end of
        # this test reading a name that belongs to a DIFFERENT test in this
        # class (here the loop variable is `r` and the list is `ratios`); it
        # never ran only because the assertion below it failed first, and it
        # would have raised NameError the moment that stopped being true.
        # Removed rather than repaired -- it duplicated these bounds.

        # ------------------------------------------------------------------
        # THE CONSEQUENCE, in the units that decide the launch config.
        # R1 = 0.5 * gamma * gsq, measured against d_loss, which at init sits
        # at 2*ln2 = 1.386 for nsgan.
        # ------------------------------------------------------------------
        off_660, on_660 = off_vals[0], on_vals[0]   # reused, not re-measured
        d_loss_at_init = 2.0 * math.log(2.0)

        def frac(gsq, gamma):
            """R1 as a fraction of ``d_loss`` at init."""
            return (0.5 * gamma * gsq) / d_loss_at_init

        # (i) On the SHIPPED norm-free critic, gamma = 1.0 buys 5.3e-6 of
        # d_loss.  "Value present, effect absent."
        for seed, off in enumerate(off_vals):
            self.assertLess(
                frac(off, PIX_R1_GAMMA), 1e-3,
                f"seed {seed}: R1 at the inherited gamma=1.0 is expected to be "
                "<0.1 % of d_loss on the norm-free critic; if it is not, this "
                "warning is stale")

        # (ii) **THE CORRECTED PREMISE, and the launch-critical half.**  This
        # test used to assert here that the SAME quantity was NOT negligible
        # under the norm -- "which is why gamma=1.0 was ever a defensible
        # number" -- by planting the assertion above against the normed build
        # and requiring it to fail.  It does not fail.  That premise is FALSE
        # and the measurement says so; at gamma = 1.0, R1/d_loss over the 10
        # matched inits above is
        #
        #     norm ON    4.485e-04 .. 9.603e-04   (seed 0: 6.488e-04)
        #     norm OFF   3.660e-06 .. 7.646e-06   (seed 0: 5.257e-06)
        #
        # -- every normed value is BELOW the same 0.1 % line.  gamma = 1.0 was
        # already delivering under 0.1 % of d_loss BEFORE the GroupNorm came
        # out.  The removal is a ~100x worsening (to ~0.0005 %) of a term that
        # was ALREADY inert; it is not what made it inert.  Re-calibration is
        # therefore not "restore what the norm removal cost": there is no
        # earlier working value to restore, because 1.0 never worked.
        for seed, on in enumerate(on_vals):
            self.assertLess(
                frac(on, PIX_R1_GAMMA), 1e-3,
                f"seed {seed}: gamma=1.0 is expected to be <0.1 % of d_loss on "
                "the NORMED build TOO; the norm removal worsened an "
                "already-inert R1, it did not create the inertness")

        # (iii) THE COMPANION -- a planted violation, so that (i) and (ii) are
        # DEMONSTRATED rather than merely asserted.
        #
        # (i) and (ii) are both `assertLess(frac(gsq, gamma), 1e-3)`.  An
        # assertion of that shape is worth nothing unless it can fail for THIS
        # critic on THIS content, so replay the identical call at a gamma large
        # enough that R1 stops being negligible, and require it to raise.
        #
        # The gammas below are CONSTANTS read off the measurement, NOT computed
        # from `gsq`.  That distinction is the whole point: planting at, say,
        # `10 * (2e-3 * d_loss / gsq)` would reduce to `frac(gsq, k/gsq) = 10 *
        # 1e-3` and fire on ANY input whatsoever -- vacuous algebra dressed as
        # a test, the same failure mode a `P**2 * x / (P**2 * y) == x / y`
        # companion in this file was retired for.  With fixed constants the
        # companion instead pins a LOWER bound on gsq: if the shipped gsq ever
        # collapsed another 10x (a further silent architecture change), 5.0 and
        # 600.0 would no longer be enough to breach 0.1 % of d_loss, the
        # planted assertion would quietly stop raising, and THIS block fails.
        #
        # In those units it is the same fact as the `gamma_for_tenth_pct_*`
        # upper bounds a few lines below -- deliberately so; that pair states
        # it, this pair exhibits it.
        PLANTED_GAMMA_ON = 5.0       # measured g*_on  1.04 .. 2.23, all seeds
        PLANTED_GAMMA_OFF = 600.0    # measured g*_off  131 .. 273, all seeds
        for seed, (on, off) in enumerate(zip(on_vals, off_vals)):
            with self.subTest(seed=seed, planted=True):
                with self.assertRaises(AssertionError):
                    self.assertLess(frac(on, PLANTED_GAMMA_ON), 1e-3)
                with self.assertRaises(AssertionError):
                    self.assertLess(frac(off, PLANTED_GAMMA_OFF), 1e-3)
                # ...and the planted gammas are not so absurd that they would
                # breach 0.1 % for any critic at all: one order of magnitude
                # below each, gamma is back under the line on both builds.
                # (This is what stops the companion above from degenerating
                # into "pick a big enough number".)
                self.assertLess(frac(on, PLANTED_GAMMA_ON / 10.0), 1e-3)
                self.assertLess(frac(off, PLANTED_GAMMA_OFF / 10.0), 1e-3)

        # The same fact in the units a re-calibration actually needs: the
        # gamma that would put R1 at 0.1 % of d_loss.  Measured over the 10
        # matched inits: 1.04 .. 2.23 with the norm, 131 .. 273 without it.
        # The shipped 1.0 is under BOTH -- but only just under the normed one
        # (worst init 1.04, i.e. it came within 4 % of the line), which is how
        # a number that was never an empirical calibration could look
        # plausible for as long as it did.
        gamma_for_tenth_pct_on = 2e-3 * d_loss_at_init / on_660
        gamma_for_tenth_pct_off = 2e-3 * d_loss_at_init / off_660
        # The lower bound here RESTATES (ii) (g* > 1 <=> frac(.,1) < 1e-3) and
        # is not independent evidence; the upper bounds are new, and are what
        # fails if gsq ever collapses further.
        self.assertGreater(gamma_for_tenth_pct_on, 1.0)
        self.assertLess(gamma_for_tenth_pct_on, 5.0,
                        f"gamma for 0.1 % of d_loss, normed build: "
                        f"{gamma_for_tenth_pct_on:.2f} (measured 1.04..2.23)")
        self.assertGreater(gamma_for_tenth_pct_off, 50.0,
                           f"gamma for 0.1 % of d_loss, shipped build: "
                           f"{gamma_for_tenth_pct_off:.1f} (measured 131..273)")
        self.assertLess(gamma_for_tenth_pct_off, 600.0,
                        f"{gamma_for_tenth_pct_off:.1f}")

        # ------------------------------------------------------------------
        # PLANTED VIOLATION.  A guard nobody has seen fail proves nothing, so
        # the 0.1 %-of-d_loss test is planted against cases that MUST trip it.
        #
        # The planted gamma is a FIXED CONSTANT.  It is deliberately NOT
        # back-computed from the gsq under test (`gamma = 2e-3*d_loss/gsq`
        # would divide the measured quantity straight back out and make the
        # check `2e-3 < 1e-3` -- an identity that "fires" on any numbers
        # whatsoever, including random floats; this file has already been
        # bitten once by a companion of exactly that shape).
        #
        # gamma = 10 DISCRIMINATES between the two builds: over the 10 matched
        # inits it is OVER the line on the normed critic (4.485e-03 ..
        # 9.603e-03 of d_loss, >=4.5x clear of the threshold) and UNDER it on
        # the shipped norm-free critic (3.660e-05 .. 7.646e-05, >=13x clear).
        # One constant, opposite verdicts -- the ~100x magnitude collapse
        # restated as a pass/fail, which is what makes this companion a
        # measurement and not a ritual.
        gamma_plant = 10.0
        with self.assertRaises(AssertionError):
            self.assertLess(frac(on_660, gamma_plant), 1e-3)
        self.assertLess(
            frac(off_660, gamma_plant), 1e-3,
            f"gamma={gamma_plant} is still expected to be negligible on the "
            "norm-free critic; if it trips, gsq has grown and the magnitude "
            "finding above needs re-measuring")

        # ...and the shipped build is not immune to the guard either, it just
        # needs ~100x more gamma to reach it: at gamma = 1000 (measured
        # 3.660e-03 .. 7.646e-03 of d_loss over the same inits) it trips.
        # This pins gsq_off from BELOW while assertion (i) pins it from above,
        # so the pair brackets the magnitude: a silent restoration of the norm
        # trips gamma=10, and a critic rescaled toward zero trips gamma=1000.
        with self.assertRaises(AssertionError):
            self.assertLess(frac(off_660, 1000.0), 1e-3)

    def test_r1_penalty_returns_the_P_it_measured(self):
        """``gsq`` cannot be compared across crops without its ``P`` (it goes
        as ``P^-1``), so ``P`` is part of the return contract and is read off
        the tensor, never from a constant."""
        net = PixelTextureDisc().eval()
        for h, w, expect in ((176, 240, 660), (88, 120, 165)):
            with self.subTest(crop=(h, w)):
                out = r1_penalty(
                    net, torch.zeros(2, 3, h, w),
                    eps=torch.ones(2, 3, h, w))
                self.assertEqual(out["P"], expect)
                self.assertEqual(out["P"], PixelTextureDisc.patch_count(h, w))

    def test_subsampling_is_unbiased(self):
        """(c) gsq is a mean over per-real squared FDs, so a subsample is an
        unbiased estimator of the full-batch value."""
        torch.manual_seed(0)
        net = _SmoothNet(seed=9).double()
        x = (torch.rand(8, 3, 24, 24, dtype=torch.float64) * 2 - 1)
        e = torch.randn(x.shape, generator=torch.Generator().manual_seed(4),
                        dtype=torch.float64)
        full = r1_penalty(net, x, sigma=1e-3, eps=e)["gsq"]

        # Exact: average over ALL 8 singleton subsamples == the full mean.
        singles = [r1_penalty(net, x, sigma=1e-3, eps=e[i:i + 1],
                              indices=torch.tensor([i]))["gsq"]
                   for i in range(8)]
        self.assertAlmostEqual(sum(singles) / 8.0 / full, 1.0, places=9)

        # And in expectation over random size-3 subsamples.
        gen = torch.Generator().manual_seed(21)
        acc, n = 0.0, 600
        for _ in range(n):
            idx = r1_subsample_indices(8, 3, gen)
            self.assertEqual(int(idx.numel()), 3)
            acc += r1_penalty(net, x, sigma=1e-3, eps=e[idx],
                              indices=idx)["gsq"]
        self.assertAlmostEqual(acc / n / full, 1.0, delta=0.1)

    def test_subsample_runs_full_network_so_ddp_is_safe(self):
        """DDP ``find_unused_parameters=False`` requires every parameter to
        PARTICIPATE in the graph, i.e. ``p.grad is not None``.

        It does NOT require the gradient to be non-zero — autograd cannot know
        a term cancels, so a cancelling parameter still gets a zeros tensor.
        (``conv4.bias`` is exactly such a case; see the dedicated test below.)
        Asserting non-zero here would be testing the wrong condition.
        """
        net = PixelTextureDisc()
        x = torch.randn(6, 3, 64, 64).clamp(-1, 1)
        gen = torch.Generator().manual_seed(0)
        res = r1_penalty(net, x, num_samples=2, generator=gen)
        self.assertEqual(res["n_used"], 2)      # the subsample really is smaller
        res["r1"].backward()
        unused = [n for n, p in net.named_parameters()
                  if p.requires_grad and p.grad is None]
        self.assertEqual(unused, [], f"DDP would error on: {unused}")

    def test_conv4_bias_r1_gradient_is_exactly_zero_by_construction(self):
        """A real mathematical property of the §5.3 estimator, pinned so a
        future reader does not mistake it for a bug.

        conv4 is the head: its bias adds the SAME constant b to every patch
        logit, so the per-image MEAN score is
            s_n(x) = (1/P) sum_i D_i(x) = (1/P) sum_i [f_i(x) + b] = s~_n(x) + b
        The finite difference cancels it identically:
            s_n(x + eps*sigma) - s_n(x) = s~_n(x + eps*sigma) - s~_n(x)
        so d(gsq)/db = 0 analytically, for ANY input.  The gradient is a zeros
        tensor, not None, so DDP is unaffected.
        """
        net = PixelTextureDisc()
        x = torch.randn(5, 3, 64, 64).clamp(-1, 1)
        r1_penalty(net, x, generator=torch.Generator().manual_seed(0))["r1"].backward()
        b = net.conv4.bias
        self.assertIsNotNone(b.grad)
        self.assertEqual(float(b.grad.abs().sum()), 0.0)
        # It is the ONLY parameter that cancels; everything else moves.
        zeros = [n for n, p in net.named_parameters()
                 if float(p.grad.abs().sum()) == 0.0]
        self.assertEqual(zeros, ["conv4.bias"])

    def test_conv4_bias_gets_a_real_gradient_from_d_loss(self):
        """...so the cancellation is a curiosity of R1 in isolation, never a
        training defect: the real D-update backprops ``d_loss + R1``."""
        net = PixelTextureDisc()
        real = torch.randn(4, 3, 64, 64).clamp(-1, 1)
        fake = torch.randn(4, 3, 64, 64).clamp(-1, 1)
        total = (d_loss(net(real), net(fake))["d_loss"]
                 + r1_penalty(net, real,
                              generator=torch.Generator().manual_seed(0))["r1"])
        total.backward()
        self.assertGreater(float(net.conv4.bias.grad.abs().sum()), 1e-4)
        self.assertTrue(all(p.grad is not None for p in net.parameters()))

    def test_subsample_never_touches_global_rng(self):
        torch.manual_seed(1234)
        before = torch.get_rng_state()
        gen = torch.Generator().manual_seed(0)
        r1_subsample_indices(10, 4, gen)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        with self.assertRaises(ValueError):
            r1_subsample_indices(10, 4, None)     # must refuse, not fall back
        with self.assertRaises(ValueError):
            r1_subsample_indices(10, 0, gen)      # empty subsample => DDP desync

    def test_subsample_none_or_oversized_uses_all(self):
        gen = torch.Generator().manual_seed(0)
        self.assertIsNone(r1_subsample_indices(6, None, gen))
        self.assertIsNone(r1_subsample_indices(6, 6, gen))
        self.assertIsNone(r1_subsample_indices(6, 99, gen))

    def test_sigma_is_in_the_minus1_1_scale(self):
        """(d) sigma multiplies a standard-normal perturbation of the [-1,1]
        input directly: with eps=1 everywhere, D sees exactly x + sigma."""
        seen = {}

        def spy(t):
            seen.setdefault("inputs", []).append(t.clone())
            return t.reshape(t.shape[0], 1, -1)[:, :, :4].reshape(t.shape[0], 1, 2, 2)

        x = torch.zeros(2, 3, 8, 8)
        r1_penalty(spy, x, sigma=PIX_R1_SIGMA,
                   eps=torch.ones(2, 3, 8, 8))
        base, pert = seen["inputs"]
        self.assertAlmostEqual(float(base.abs().max()), 0.0, places=9)
        self.assertAlmostEqual(float(pert.max()), 0.01, places=9)
        self.assertAlmostEqual(PIX_R1_SIGMA, 0.01, places=9)

    def test_gamma_one_half_scaling(self):
        net = _SmoothNet(seed=1).double()
        x = (torch.rand(3, 3, 16, 16, dtype=torch.float64) * 2 - 1)
        e = torch.randn(x.shape, generator=torch.Generator().manual_seed(0),
                        dtype=torch.float64)
        r = r1_penalty(net, x, gamma=PIX_R1_GAMMA, sigma=1e-3, eps=e)
        self.assertAlmostEqual(float(r["r1"].detach()), 0.5 * PIX_R1_GAMMA * r["gsq"],
                               places=12)
        r3 = r1_penalty(net, x, gamma=3.0, sigma=1e-3, eps=e)
        self.assertAlmostEqual(r3["gsq"], r["gsq"], places=12)   # gsq is RAW
        self.assertAlmostEqual(float(r3["r1"].detach()) / float(r["r1"].detach()), 3.0, places=9)

    def test_r1_grads_flow_to_disc_params_not_to_data(self):
        net = PixelTextureDisc()
        x = torch.randn(4, 3, 64, 64).clamp(-1, 1).requires_grad_(True)
        r1_penalty(net, x)["r1"].backward()
        self.assertIsNone(x.grad, "R1 must not backprop into the data path")
        self.assertTrue(any(p.grad is not None for p in net.parameters()))

    def test_r1_rejects_bad_inputs(self):
        net = PixelTextureDisc()
        with self.assertRaises(ValueError):
            r1_penalty(net, torch.randn(3, 64, 64))
        with self.assertRaises(ValueError):
            r1_penalty(net, torch.randn(0, 3, 64, 64))
        with self.assertRaises(ValueError):
            r1_penalty(net, torch.randn(2, 3, 64, 64), sigma=0.0)


# ===========================================================================
# 6. §5.4 / §7 counters
# ===========================================================================
class TestCounters(unittest.TestCase):
    def test_monotone_and_rate_is_one_with_every_n_1(self):
        c = PixCounters(r1_every_n=1)
        last_d, last_r = -1, -1
        for _ in range(37):
            fired = c.should_fire_r1()
            self.assertTrue(fired)
            c.note_d_update(fired)
            self.assertGreater(c.dupdate_total, last_d)
            self.assertGreater(c.r1_fired_total, last_r)
            last_d, last_r = c.dupdate_total, c.r1_fired_total
        d = c.log_dict()
        self.assertEqual(d["pix_dupdate_total"], 37.0)
        self.assertEqual(d["pix_r1_fired_total"], 37.0)
        self.assertEqual(d["pix_r1_rate"], 1.00)
        self.assertTrue(c.healthy())

    def test_rate_readable_on_the_first_logged_row(self):
        c = PixCounters()
        c.note_d_update(c.should_fire_r1())
        self.assertEqual(c.log_dict()["pix_r1_rate"], 1.0)

    def test_under_099_is_flagged_unhealthy(self):
        c = PixCounters(r1_every_n=5)             # off-spec on purpose
        for _ in range(20):
            c.note_d_update(c.should_fire_r1())
        self.assertAlmostEqual(c.r1_rate, 0.2, places=6)
        self.assertFalse(c.healthy())

    def test_no_gauge_form_offered(self):
        c = PixCounters()
        for name in ("r1_fired", "r1_gauge", "reset", "pix_r1_fired"):
            self.assertFalse(hasattr(c, name), f"gauge-ish API {name!r} exists")
        self.assertEqual(set(c.log_dict()),
                         {"pix_dupdate_total", "pix_r1_fired_total", "pix_r1_rate"})

    def test_no_r2_counter(self):
        c = PixCounters()
        self.assertFalse(any("r2" in k for k in c.log_dict()))
        self.assertFalse(any("r2" in s for s in dir(c)))

    # -- F6: the gsq running mean ------------------------------------------
    def test_gsq_running_mean_keys_are_ABSENT_until_observed(self):
        """§5.3 says calibrate gamma from the first run's measured R1
        magnitude; §5.4 says read it on the FIRST logged row.  A running mean
        exists to make that possible — and until something feeds it, its keys
        must be ABSENT.  A 0.0 running mean would read as "R1 measures no
        gradient on the reals", which is a real alarm, not a placeholder."""
        c = PixCounters()
        for _ in range(3):
            c.note_d_update(c.should_fire_r1())          # no gsq supplied
        d = c.log_dict()
        self.assertEqual(set(d), {"pix_dupdate_total", "pix_r1_fired_total",
                                  "pix_r1_rate"})
        self.assertFalse(any("grad_sq" in k for k in d))
        self.assertIsNone(c.r1_grad_sq_mean)
        self.assertIsNone(c.r1_grad_sq_last)
        self.assertEqual(c.r1_grad_sq_samples, 0)

    def test_gsq_running_mean_is_the_running_mean_and_exposes_n(self):
        c = PixCounters()
        vals = [1e-3, 3e-3, 2e-3, 8e-3]
        for i, v in enumerate(vals, 1):
            c.note_d_update(c.should_fire_r1(), gsq=v)
            self.assertEqual(c.r1_grad_sq_samples, i)
            self.assertAlmostEqual(c.r1_grad_sq_mean, sum(vals[:i]) / i,
                                   places=12)
        d = c.log_dict()
        self.assertAlmostEqual(d["pix_r1_grad_sq_mean"], sum(vals) / 4, places=12)
        self.assertEqual(d["pix_r1_grad_sq_n"], 4.0)
        # the RAW per-step value is kept too — the mean is for calibration,
        # the raw value is what shows a step-to-step blow-up.
        self.assertEqual(d["pix_r1_grad_sq_last"], 8e-3)
        self.assertEqual(c.r1_grad_sq_last, 8e-3)

    def test_gsq_running_mean_is_why_one_row_is_not_enough(self):
        """The motivating measurement, reproduced small.  ``gsq`` is unbiased
        but extremely noisy per draw: on PixelTextureDisc at the spec crop,
        300 draws gave relative sd 126 % at N=1 real (values 2.3e-09..9.0e-03)
        and 40 % at N=12.  Here: a stream whose single draws span 4 orders of
        magnitude still yields a running mean within 1e-12 of the truth."""
        c = PixCounters()
        gen = torch.Generator().manual_seed(5)
        draws = (torch.rand(200, generator=gen) ** 6 * 1e-2).tolist()
        for v in draws:
            c.note_d_update(True, gsq=v)
        self.assertGreater(max(draws) / max(min(draws), 1e-30), 1e3)
        self.assertAlmostEqual(c.r1_grad_sq_mean, sum(draws) / len(draws),
                               places=12)
        self.assertEqual(c.r1_grad_sq_samples, 200)

    def test_gsq_running_mean_refuses_a_nonfinite_sample(self):
        """PLANTED VIOLATION: one NaN folded in would poison every subsequent
        reading, silently and permanently."""
        c = PixCounters()
        c.note_d_update(True, gsq=1e-3)
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    c.note_d_update(True, gsq=bad)
        # ...and the accumulator is untouched by the rejected samples.
        self.assertEqual(c.r1_grad_sq_samples, 1)
        self.assertAlmostEqual(c.r1_grad_sq_mean, 1e-3, places=12)

    def test_gsq_accumulation_is_opt_in_so_old_call_sites_are_unchanged(self):
        """``note_d_update(r1_fired)`` keeps its exact old behaviour: the
        counters move, nothing is accumulated, no key appears."""
        c = PixCounters()
        c.note_d_update(True)
        self.assertEqual((c.dupdate_total, c.r1_fired_total), (1, 1))
        self.assertEqual(c.r1_grad_sq_samples, 0)
        self.assertNotIn("pix_r1_grad_sq_mean", c.log_dict())


class TestPatchTelemetry(unittest.TestCase):
    def test_spatial_variance_zero_on_flat_map(self):
        self.assertAlmostEqual(
            patch_logit_spatial_variance(torch.full((3, 1, 5, 6), 2.0)),
            0.0, places=9)

    def test_spatial_variance_positive_on_structured_map(self):
        m = torch.zeros(2, 1, 4, 4)
        m[:, :, :2] = 5.0
        self.assertGreater(patch_logit_spatial_variance(m), 1.0)

    def test_telemetry_reports_the_real_P(self):
        net = PixelTextureDisc().eval()
        t = patch_logit_telemetry(net(torch.randn(2, 3, 176, 240)))
        self.assertEqual(t["pix_patch_count"], 660.0)
        self.assertEqual(t["pix_patch_grid_h"], 22.0)
        self.assertEqual(t["pix_patch_grid_w"], 30.0)

    def test_grid_path_is_UNCHANGED_regression_guard(self):
        """THE IMPORTANT ONE.  Relaxing the accepted shapes must not perturb
        the ``[N,1,h,w]`` path: same keys, same ORDER, same values."""
        g = torch.Generator().manual_seed(0)
        m = torch.randn(4, 1, 22, 30, generator=g)
        got = patch_logit_telemetry(m)
        self.assertEqual(
            list(got),
            ["pix_patch_mean", "pix_patch_std", "pix_patch_spatial_var",
             "pix_patch_count", "pix_patch_grid_h", "pix_patch_grid_w"])
        flat = m.reshape(4, -1)
        self.assertEqual(got["pix_patch_mean"], float(flat.mean()))
        self.assertEqual(got["pix_patch_std"], float(flat.std()))
        self.assertEqual(got["pix_patch_spatial_var"],
                         float(flat.var(dim=1, unbiased=True).mean()))
        self.assertEqual(got["pix_patch_count"], 660.0)
        self.assertEqual(got["pix_patch_grid_h"], 22.0)
        self.assertEqual(got["pix_patch_grid_w"], 30.0)
        # Custom prefix still applies to every key.
        self.assertTrue(all(k.startswith("lat_")
                            for k in patch_logit_telemetry(m, prefix="lat_")))

    def test_telemetry_accepts_per_sample_scores(self):
        """A latent critic with ``ladd_scalar_output`` emits [N] / [N,1]."""
        for shape in [(16,), (16, 1)]:
            with self.subTest(shape=shape):
                v = torch.randn(*shape, generator=torch.Generator().manual_seed(1))
                got = patch_logit_telemetry(v)
                self.assertEqual(list(got), ["pix_patch_mean", "pix_patch_std",
                                             "pix_patch_count"])
                self.assertAlmostEqual(got["pix_patch_mean"],
                                       float(v.mean()), places=6)
                self.assertAlmostEqual(got["pix_patch_std"],
                                       float(v.std()), places=6)
                self.assertEqual(got["pix_patch_count"], 1.0)

    def test_spatial_var_key_is_OMITTED_not_faked_as_zero(self):
        """A synthesised 0.0 would read as "the critic is not using locality"
        — a real §7 diagnostic.  It must not be forgeable by a shape."""
        for shape in [(16,), (16, 1)]:
            with self.subTest(shape=shape):
                got = patch_logit_telemetry(torch.randn(*shape))
                self.assertNotIn("pix_patch_spatial_var", got)
                self.assertFalse(any("spatial" in k for k in got))
        # ...whereas a genuinely FLAT grid does report 0.0, which is the real
        # signal the omission is protecting.
        flat_grid = patch_logit_telemetry(torch.full((3, 1, 4, 5), 2.0))
        self.assertEqual(flat_grid["pix_patch_spatial_var"], 0.0)

    def test_spatial_variance_itself_stays_strict_on_per_sample_scores(self):
        """It is a POSITIONAL statistic; asking for it on one-score-per-sample
        tensors is a caller bug.  (A 3-D token map is NOT such a case — see
        the token-map tests below.)"""
        for shape in [(16,), (16, 1)]:
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    patch_logit_spatial_variance(torch.randn(*shape))

    def test_a_3D_token_map_is_ACCEPTED_by_both_probe_and_telemetry(self):
        """CORRECTED PARITY CLAIM.  This docstring used to say the accepted
        shapes were "deliberately the same set as
        ``disc_holdout_probe._reduce_scores``".  They were not: the probe has
        NO dim restriction, telemetry restricted to ``dim in (1, 2, 4)``, and
        at ``[6, 1, 7]`` the probe returned ``(6,)`` while telemetry RAISED.

        A 3-D token map is the realistic latent case, and serving latent
        critics is the stated reason the relaxation exists — so 3-D is now
        accepted, and the docstring states the real relationship instead of
        claiming an identity that never held.
        """
        t = torch.randn(6, 1, 7, generator=torch.Generator().manual_seed(3))
        self.assertEqual(tuple(_reduce_scores(t).shape), (6,))     # the probe
        got = patch_logit_telemetry(t)                             # no longer raises
        self.assertEqual(list(got), ["pix_patch_mean", "pix_patch_std",
                                     "pix_patch_spatial_var",
                                     "pix_patch_count"])
        flat = t.reshape(6, -1)
        self.assertAlmostEqual(got["pix_patch_mean"], float(flat.mean()), places=6)
        self.assertEqual(got["pix_patch_count"], 7.0)
        # spatial_var IS computed for a token map — the omit-never-fake rule
        # says omit what is UNDEFINED, not what is merely 1-D.
        self.assertEqual(got["pix_patch_spatial_var"],
                         float(flat.var(dim=1, unbiased=True).mean()))
        self.assertEqual(got["pix_patch_spatial_var"],
                         patch_logit_spatial_variance(t))
        # ...but the GRID dimensions are omitted: a token map has no h/w split
        # and inventing one would forge geometry.
        self.assertNotIn("pix_patch_grid_h", got)
        self.assertNotIn("pix_patch_grid_w", got)

    def test_a_flat_token_map_still_reports_zero_spatial_var(self):
        """The signal the omission protects: a genuinely flat TOKEN map is a
        collapsed critic and must read 0.0, not be hidden."""
        got = patch_logit_telemetry(torch.full((3, 1, 9), 2.0))
        self.assertEqual(got["pix_patch_spatial_var"], 0.0)

    def test_single_position_map_OMITS_spatial_var_rather_than_faking_zero(self):
        """A map with ONE position per image has no within-image variance.
        Returning 0.0 there would forge the collapsed-critic signature from a
        shape — the same forgeable-zero failure the [N]/[N,1] path guards."""
        for shape in [(4, 1, 1), (4, 1, 1, 1)]:
            with self.subTest(shape=shape):
                got = patch_logit_telemetry(torch.randn(*shape))
                self.assertNotIn("pix_patch_spatial_var", got)
                self.assertEqual(got["pix_patch_count"], 1.0)
                with self.assertRaises(ValueError):
                    patch_logit_spatial_variance(torch.randn(*shape))

    def test_telemetry_still_rejects_shapes_with_no_agreed_meaning(self):
        """DELIBERATE, STATED divergence from ``_reduce_scores`` (which would
        silently flatten these).  Rank 0 and rank >= 5 have no agreed
        patch/token reading, so a caller handing them over has a bug and any
        ``patch_count`` reported for them would be fabricated."""
        for t in [torch.randn(()), torch.randn(2, 3, 4, 5, 6)]:
            with self.subTest(shape=tuple(t.shape)):
                with self.assertRaises(ValueError):
                    patch_logit_telemetry(t)

    def test_accepted_shapes_vs_reduce_scores_stated_exactly(self):
        """What is actually true, in place of the old parity claim: every
        shape TELEMETRY accepts, the probe also reduces to ``(6,)``; the probe
        additionally accepts ranks telemetry refuses.  Subset, not identity.
        """
        for t in [torch.randn(6), torch.randn(6, 1), torch.randn(6, 1, 7),
                  torch.randn(6, 1, 3, 4)]:
            with self.subTest(shape=tuple(t.shape)):
                self.assertEqual(tuple(_reduce_scores(t).shape), (6,))
                self.assertIn("pix_patch_mean", patch_logit_telemetry(t))
        # the divergence, demonstrated rather than asserted in prose
        wide = torch.randn(6, 1, 2, 3, 4)
        self.assertEqual(tuple(_reduce_scores(wide).shape), (6,))
        with self.assertRaises(ValueError):
            patch_logit_telemetry(wide)


# ===========================================================================
# 7. Spec-drift tripwire — no pix_r2 symbol anywhere
# ===========================================================================
class TestNoR2SpecDrift(unittest.TestCase):
    def test_no_pix_r2_symbol_in_module_source(self):
        """§5.3/§7: a build that introduces any pix_r2_* is off-spec.

        Scanned against CODE ONLY (see :func:`code_only`).  The module's prose
        states, at length, that R2 does not exist; that documentation is the
        point, so the grep must not fire on it.  A real ``pix_r2_gamma``
        assignment is still caught — proved by :class:`TestTripwireItself`.
        """
        self.assertNotIn("pix_r2", MODULE_CODE.lower())
        self.assertNotIn("_r2_", MODULE_CODE.lower())
        # And the prose really does discuss R2 — otherwise this test is
        # trivially green for the wrong reason.
        self.assertIn("pix_r2", MODULE_SRC.lower())

    def test_no_r2_in_the_public_api(self):
        import model.pixel_texture_disc as m
        self.assertEqual(
            [n for n in dir(m) if "r2" in n.lower() and not n.startswith("__")],
            [])
        self.assertFalse(any("r2" in k for k in PIX_DEFAULTS))

    def test_r1_estimator_is_the_only_one(self):
        """One estimator, one rate.  There must be exactly one public R1 entry
        point, and no autograd / sum / per-patch alternative."""
        import model.pixel_texture_disc as m
        r1_syms = [n for n in dir(m)
                   if n.lower().startswith("r1_") or "_r1_penalty" in n.lower()]
        self.assertEqual(sorted(r1_syms), ["r1_penalty", "r1_subsample_indices"])
        low = MODULE_CODE.lower()
        self.assertNotIn("pix_r1_normalize", low)     # §5.3: not a flag, ever
        self.assertNotIn("pix_r1_once_per_step", low)  # §5.4: not implemented
        self.assertNotIn("autograd.grad", low)         # ONE estimator, and it is FD

    def test_no_ladd_import(self):
        """§5.3: four divergent R1 estimators live in ladd_disc; import none.

        Code-only again — the module docstring names ladd_disc precisely to
        record that it is NOT reused.
        """
        self.assertNotIn("ladd_disc", MODULE_CODE)
        self.assertNotIn("ladd_", MODULE_CODE)
        imports = {n.split(".")[0]
                   for node in ast.walk(ast.parse(MODULE_SRC))
                   for n in ([a.name for a in node.names]
                             if isinstance(node, ast.Import)
                             else [node.module or ""]
                             if isinstance(node, ast.ImportFrom) else [])}
        self.assertNotIn("model", imports, "must not import sibling model code")


class TestTripwireItself(unittest.TestCase):
    """A tripwire nobody has watched trip is not a tripwire.

    These feed :func:`code_only` synthetic sources and assert it separates
    "documented as absent" from "actually present".
    """

    PROSE_ONLY = '''
"""There is no pix_r2_gamma, no pix_r1_normalize knob, and this module
never reads yaml or imports ladd_disc."""
import torch


def f(x):
    """Also mentions pix_r2_sigma and omegaconf, in prose."""
    # a comment naming pix_r2_every_n and cfg_get
    return x
'''

    REAL_VIOLATION = '''
"""Innocent docstring."""
pix_r2_gamma = 0.0


def g(x):
    return x * pix_r2_gamma
'''

    def test_prose_mentions_are_not_flagged(self):
        code = code_only(self.PROSE_ONLY)
        for term in ("pix_r2", "pix_r1_normalize", "yaml", "ladd_disc",
                     "omegaconf", "cfg_get"):
            self.assertNotIn(term, code.lower(), f"false positive on {term!r}")
        # ...and they really were present in the raw source.
        for term in ("pix_r2", "yaml", "ladd_disc", "cfg_get"):
            self.assertIn(term, self.PROSE_ONLY.lower())

    def test_planted_violation_IS_flagged(self):
        code = code_only(self.REAL_VIOLATION)
        self.assertIn("pix_r2_gamma", code.lower())
        self.assertIn("pix_r2", code.lower())

    def test_violation_inside_a_string_literal_is_flagged(self):
        """Non-docstring string constants survive on purpose: a banned name
        smuggled through getattr is a real violation."""
        src = '"""doc."""\nimport torch\n\n\ndef h(o):\n    return getattr(o, "pix_r2_gamma")\n'
        self.assertIn("pix_r2_gamma", code_only(src).lower())

    def test_code_only_drops_comments_and_docstrings_but_keeps_code(self):
        code = code_only(self.REAL_VIOLATION)
        self.assertNotIn("Innocent docstring", code)
        self.assertIn("def g", code)


# ===========================================================================
# 8. §8.1 C1 corruption
# ===========================================================================
def _texture_surrogate(seed: int = 0, n: int = 4, c: int = 16,
                       h: int = 60, w: int = 104) -> torch.Tensor:
    """A near-isotropic, smooth stand-in for a GT latent (no VAE needed)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, c, h, w, generator=g)
    return F.avg_pool2d(x, 3, 1, 1)


class TestC1Corruption(unittest.TestCase):
    def test_deterministic_under_a_fixed_generator(self):
        x = _texture_surrogate()
        a = c1_structured_hf(x, 0.5, generator=torch.Generator().manual_seed(7))
        b = c1_structured_hf(x, 0.5, generator=torch.Generator().manual_seed(7))
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        c = c1_structured_hf(x, 0.5, generator=torch.Generator().manual_seed(8))
        self.assertFalse(torch.equal(a, c))

    def test_never_touches_global_rng(self):
        torch.manual_seed(99)
        before = torch.get_rng_state()
        c1_structured_hf(_texture_surrogate(), 0.5,
                         generator=torch.Generator().manual_seed(1))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_zero_amplitude_is_identity(self):
        x = _texture_surrogate()
        torch.testing.assert_close(
            c1_structured_hf(x, 0.0, generator=torch.Generator().manual_seed(0)),
            x, rtol=0, atol=0)

    def test_shape_and_dtype_preserved(self):
        for shape in [(4, 16, 60, 104), (2, 3, 16, 3, 32), (3, 176, 240)]:
            with self.subTest(shape=shape):
                x = torch.randn(*shape)
                y = c1_structured_hf(x, 0.4,
                                     generator=torch.Generator().manual_seed(0))
                self.assertEqual(y.shape, x.shape)
                self.assertEqual(y.dtype, x.dtype)

    def test_battery_moves_hv_anisotropy_and_entropy_DOWN(self):
        """§8.1's measured failure direction.

        texture_stats.hv_anisotropy = p_fy / p_fx and ">1 means horizontally
        striped".  The measured collapse is 1.10 (A) -> 0.36 (C_late), i.e.
        DOWN, and angular_entropy 0.98 -> 0.84, also DOWN.  C1 must reproduce
        BOTH directions.
        """
        x = _texture_surrogate()
        clean = texture_battery(x)
        corrupt = texture_battery(
            c1_structured_hf(x, 0.75, generator=torch.Generator().manual_seed(7)))
        self.assertLess(corrupt["hv_anisotropy"], clean["hv_anisotropy"])
        self.assertLess(corrupt["angular_entropy"], clean["angular_entropy"])
        # ...and HF power UP, as C_late measured (2.21x A).
        self.assertGreater(corrupt["hf_power"], clean["hf_power"])

    def test_monotone_in_amplitude(self):
        x = _texture_surrogate()
        amps = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5]
        hv, ent = [], []
        for a in amps:
            b = texture_battery(
                c1_structured_hf(x, a, generator=torch.Generator().manual_seed(7)))
            hv.append(b["hv_anisotropy"])
            ent.append(b["angular_entropy"])
        for i in range(1, len(amps)):
            self.assertLess(hv[i], hv[i - 1], f"hv not monotone at a={amps[i]}")
            self.assertLess(ent[i], ent[i - 1], f"ent not monotone at a={amps[i]}")

    def test_orientation_flag_is_two_sided(self):
        """The statistic is two-sided; 'row' is the mirror image.  Exists so
        the DOWN direction above is a measurement, not an artefact."""
        x = _texture_surrogate()
        clean = texture_battery(x)
        up = texture_battery(c1_structured_hf(
            x, 0.75, generator=torch.Generator().manual_seed(7),
            orientation="row"))
        self.assertGreater(up["hv_anisotropy"], clean["hv_anisotropy"])
        with self.assertRaises(ValueError):
            c1_structured_hf(x, 0.5, orientation="diagonal",
                             generator=torch.Generator().manual_seed(0))

    def test_calibration_sweep_finds_an_amplitude_between_clean_and_clate(self):
        x = _texture_surrogate()
        res = sweep_c1_amplitude(
            x, [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5], seed=7)
        self.assertIsNotNone(res["recommended"])
        rec = [r for r in res["rows"] if r["amplitude"] == res["recommended"]][0]
        self.assertTrue(rec["verdict"]["between"])
        for k, f in rec["verdict"]["frac"].items():
            self.assertGreater(f, 0.0, k)
            self.assertLess(f, 1.0, k)
        # The recommendation is the one nearest target_frac, NOT the smallest.
        self.assertGreater(res["recommended"], 0.05)

    def test_calibration_verdict_rejects_overshoot_and_no_move(self):
        clean = {"hv_anisotropy": 1.10, "angular_entropy": 0.98}
        self.assertTrue(c1_calibration_verdict(
            clean, {"hv_anisotropy": 0.70, "angular_entropy": 0.92})["between"])
        self.assertFalse(c1_calibration_verdict(   # overshot C_late
            clean, {"hv_anisotropy": 0.20, "angular_entropy": 0.70})["between"])
        self.assertFalse(c1_calibration_verdict(   # wrong direction
            clean, {"hv_anisotropy": 1.30, "angular_entropy": 0.99})["between"])

    def test_reference_batteries_match_the_doc(self):
        self.assertEqual(C_LATE_BATTERY_REF["hv_anisotropy"], 0.363)
        self.assertEqual(C_LATE_BATTERY_REF["angular_entropy"], 0.843)
        self.assertEqual(B_BATTERY_REF["hv_anisotropy"], (1.07, 1.15))


# ===========================================================================
# 9. §8.1 readout
# ===========================================================================
class TestSeparationReadout(unittest.TestCase):
    def test_roc_auc_fast_matches_the_probe_exact_form(self):
        g = torch.Generator().manual_seed(0)
        for _ in range(6):
            a = torch.randn(40, generator=g)
            b = torch.randn(37, generator=g) - 0.4
            self.assertAlmostEqual(roc_auc_fast(a, b), roc_auc(a, b), places=12)

    def test_roc_auc_fast_handles_ties(self):
        a = torch.tensor([1.0, 2.0, 2.0])
        b = torch.tensor([2.0, 3.0])
        self.assertAlmostEqual(roc_auc_fast(a, b), roc_auc(a, b), places=12)

    def test_separation_chance_and_perfect(self):
        g = torch.Generator().manual_seed(0)
        same = torch.randn(12, 1, 4, 5, generator=g)
        other = torch.randn(12, 1, 4, 5, generator=g)
        r = patch_logit_separation(same, other, n_boot=64,
                                   generator=torch.Generator().manual_seed(1))
        self.assertAlmostEqual(r["auc"], 0.5, delta=0.1)
        self.assertEqual(r["n_pos_crops"], 12.0)
        self.assertEqual(r["n_patches_per_crop"], 20.0)
        self.assertLessEqual(r["auc_lo"], r["auc"])
        self.assertGreaterEqual(r["auc_hi"], r["auc"])

        hi = torch.full((8, 1, 3, 3), 5.0)
        lo = torch.full((8, 1, 3, 3), -5.0)
        r2 = patch_logit_separation(hi, lo, n_boot=32,
                                    generator=torch.Generator().manual_seed(1))
        self.assertAlmostEqual(r2["auc"], 1.0, places=9)
        self.assertAlmostEqual(r2["gap"], 10.0, places=5)

    def test_bootstrap_resamples_CROPS_not_patches(self):
        """8 identical crops -> resampling crops cannot change anything, so the
        CI is degenerate.  If it resampled patches the CI would be wide."""
        pos = torch.stack([torch.arange(9.0).reshape(1, 3, 3)] * 8)
        neg = torch.stack([torch.arange(9.0).reshape(1, 3, 3) - 3.0] * 8)
        r = patch_logit_separation(pos, neg, n_boot=64,
                                   generator=torch.Generator().manual_seed(0))
        self.assertAlmostEqual(r["gap_lo"], r["gap_hi"], places=9)
        self.assertAlmostEqual(r["auc_lo"], r["auc_hi"], places=9)

    def test_positive_control_readout_emits_both_pairs_and_rank_ok(self):
        net = PixelTextureDisc().eval()
        g = torch.Generator().manual_seed(0)
        gt = torch.randn(6, 3, 64, 64, generator=g).clamp(-1, 1)
        cor = c1_structured_hf(gt, 0.8, generator=torch.Generator().manual_seed(2))
        st = torch.randn(6, 3, 64, 64, generator=g).clamp(-1, 1)
        out = positive_control_readout(net, gt, cor.clamp(-1, 1), st, n_boot=32,
                                       generator=torch.Generator().manual_seed(3))
        for k in ("pix_poscontrol_auc", "pix_poscontrol_auc_lo",
                  "pix_poscontrol_auc_hi", "pix_poscontrol_gap",
                  "pix_poscontrol_gap_lo", "pix_poscontrol_gap_hi",
                  "pix_arm_auc", "pix_arm_gap", "pix_poscontrol_rank_ok",
                  "pix_poscontrol_gt_patch_count"):
            self.assertIn(k, out)
        self.assertIn(out["pix_poscontrol_rank_ok"], (0.0, 1.0))
        self.assertFalse(any("r2" in k for k in out))

    def test_harness_accepts_an_injected_LATENT_scorer(self):
        """§8.1 is scorer-injectable: the harness assumes nothing about pixels,
        so the stock-1.3B / 14B latent arms can reuse it.  Both critic output
        shapes must round-trip, and rank_ok must still compute."""

        def token_map_scorer(z):          # a latent critic, [N,1,h,w] tokens
            return z.mean(dim=1, keepdim=True)[:, :, ::2, ::2]

        def per_sample_scorer(z):         # same critic with scalar output
            return z.reshape(z.shape[0], -1).mean(dim=1)

        # §8.1's intended ordering: the C1 control is calibrated MILDER than
        # the student on the texture battery, yet the critic is expected to
        # separate it MORE readily -- that is the whole point of the rank
        # check.  The toy mirrors the separability, which is what rank_ok
        # tests: control strongly separable, student weakly separable.
        g = torch.Generator().manual_seed(0)
        gt = torch.randn(8, 16, 12, 20, generator=g)            # LATENTS, not px
        cor = gt - 1.2                                          # control: easy
        st = gt - 0.6                                           # student: hard

        for name, scorer in [("token_map", token_map_scorer),
                             ("per_sample", per_sample_scorer)]:
            with self.subTest(scorer=name):
                out = positive_control_readout(
                    scorer, gt, cor, st, n_boot=32,
                    generator=torch.Generator().manual_seed(3))
                for k in ("pix_poscontrol_auc", "pix_poscontrol_gap",
                          "pix_arm_auc", "pix_arm_gap",
                          "pix_poscontrol_rank_ok",
                          "pix_poscontrol_gt_patch_mean"):
                    self.assertIn(k, out)
                # The GAP is exact for both shapes (a uniform shift survives
                # any mean reduction); the AUC saturates at 1.0 only for the
                # per-sample scorer, because pooled PATCH logits spread wider
                # than the shift.  Assert what is actually true of each.
                self.assertAlmostEqual(out["pix_poscontrol_gap"], 1.2, places=4)
                self.assertAlmostEqual(out["pix_arm_gap"], 0.6, places=4)
                self.assertGreater(out["pix_poscontrol_auc"], 0.9)
                self.assertGreater(out["pix_arm_auc"], 0.9)
                self.assertLessEqual(out["pix_poscontrol_auc"], 1.0)
                # §8.1 rank semantics unchanged: control >= arm.
                self.assertGreaterEqual(out["pix_poscontrol_auc"],
                                        out["pix_arm_auc"])
                self.assertEqual(out["pix_poscontrol_rank_ok"], 1.0)
                self.assertFalse(any("r2" in k for k in out))

        # The grid scorer reports spatial variance; the scalar one omits it.
        gen = torch.Generator().manual_seed(3)
        self.assertIn("pix_poscontrol_gt_patch_spatial_var",
                      positive_control_readout(token_map_scorer, gt, cor,
                                               n_boot=4, generator=gen))
        self.assertNotIn("pix_poscontrol_gt_patch_spatial_var",
                         positive_control_readout(per_sample_scorer, gt, cor,
                                                  n_boot=4, generator=gen))

    def test_rank_ok_flags_an_inverted_calibration(self):
        """If the control separates LESS than the student, the calibration is
        wrong, not the critic (§8.1).  rank_ok must go to 0."""
        scorer = lambda z: z.reshape(z.shape[0], -1).mean(dim=1)
        g = torch.Generator().manual_seed(1)
        gt = torch.randn(8, 4, 6, 6, generator=g)
        out = positive_control_readout(
            scorer, gt, gt - 2.0, gt - 0.1, n_boot=16,   # control HARSHER
            generator=torch.Generator().manual_seed(0))
        self.assertEqual(out["pix_poscontrol_rank_ok"], 1.0)  # AUCs both 1.0
        # Make the control genuinely unable to separate.
        out2 = positive_control_readout(
            scorer, gt, gt, gt - 2.0, n_boot=16,
            generator=torch.Generator().manual_seed(0))
        self.assertEqual(out2["pix_poscontrol_rank_ok"], 0.0)

    def test_positive_control_is_no_grad(self):
        net = PixelTextureDisc()
        gt = torch.randn(3, 3, 64, 64).clamp(-1, 1)
        positive_control_readout(net, gt, gt * 0.9, n_boot=4)
        self.assertTrue(all(p.grad is None for p in net.parameters()))


# ===========================================================================
# 10. Generator-side gradient path + config surface
# ===========================================================================
class TestGeneratorGradient(unittest.TestCase):
    def test_g_loss_gradient_reaches_the_fake_input(self):
        net = PixelTextureDisc()
        fake = torch.randn(3, 3, 96, 128).clamp(-1, 1).requires_grad_(True)
        g_loss(net(fake), "nsgan").backward()
        self.assertIsNotNone(fake.grad)
        self.assertGreater(float(fake.grad.abs().sum()), 0.0)
        self.assertEqual(fake.grad.shape, fake.shape)

    def test_g_loss_gradient_hinge_form_too(self):
        net = PixelTextureDisc()
        fake = torch.randn(2, 3, 64, 64).clamp(-1, 1).requires_grad_(True)
        g_loss(net(fake), "hinge").backward()
        self.assertGreater(float(fake.grad.abs().sum()), 0.0)

    def test_d_loss_gradient_does_not_require_the_generator(self):
        net = PixelTextureDisc()
        real = torch.randn(2, 3, 64, 64).clamp(-1, 1)
        fake = torch.randn(2, 3, 64, 64).clamp(-1, 1)   # detached, as in §5.2
        d_loss(net(real), net(fake))["d_loss"].backward()
        self.assertTrue(all(p.grad is not None for p in net.parameters()))


class TestConfigSurface(unittest.TestCase):
    def test_spec_defaults(self):
        self.assertEqual(PIX_DEFAULTS["pix_crop_lat"], (24, 32))
        self.assertEqual(PIX_DEFAULTS["pix_frames_per_crop"], 3)
        self.assertEqual(PIX_DEFAULTS["pix_crops_per_step"], 4)
        self.assertEqual(PIX_DEFAULTS["pix_band_count"], 3)
        self.assertEqual(PIX_DEFAULTS["pix_reals_per_fake"], 1)
        self.assertEqual(PIX_DEFAULTS["pix_gan_lr"], 1e-5)
        self.assertEqual(PIX_DEFAULTS["pix_gan_betas"], (0.0, 0.9))
        self.assertEqual(PIX_DEFAULTS["pix_loss_form"], "nsgan")
        self.assertEqual(PIX_DEFAULTS["pix_r1_gamma"], 1.0)
        self.assertEqual(PIX_DEFAULTS["pix_r1_sigma"], 0.01)
        self.assertEqual(PIX_DEFAULTS["pix_r1_every_n"], 1)
        self.assertIsNone(PIX_DEFAULTS["pix_r1_num_samples"])

    def test_pix_gan_weight_has_no_default(self):
        """§5.5 withdraws 0.03; its absence must be LOUD."""
        self.assertIsNone(PIX_GAN_WEIGHT_DEFAULT)
        self.assertIsNone(PIX_DEFAULTS["pix_gan_weight"])
        with self.assertRaises(ValueError):
            resolve_gan_weight(None)
        self.assertEqual(resolve_gan_weight(None, strict=False), 0.0)
        self.assertEqual(resolve_gan_weight(0.05), 0.05)
        self.assertNotIn("0.03", str(PIX_GAN_WEIGHT_DEFAULT))

    def test_module_reads_no_yaml(self):
        """Code-only: the docstring says "never reads yaml", which is prose."""
        for banned in ("yaml", "OmegaConf", "omegaconf", "trainer.config",
                       "cfg_get", "get_config"):
            self.assertNotIn(banned, MODULE_CODE, f"module reads config: {banned}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
