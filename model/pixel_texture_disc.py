"""B2 / WP-PIXGAN — pixel-space texture PatchGAN critic (`docs/TEXTURE_GAN_DESIGN.md`).

Scope, verbatim from the B2 SCOPE FREEZE (A19)::

    "Can local decoded-pixel adversarial feedback suppress fabricated
     directional texture?"

Nothing in this module widens that question.  There is no ADA, no timestep or
provenance conditioning, no pretrained backbone, no multi-scale/temporal head,
no wavelet branch, no retrieval-matched reals, no action conditioning and no
transition pairing.  Those are escalation paths with recorded triggers, not
build options.

WHAT LIVES HERE
---------------
* :class:`PixelTextureDisc` — the §4 PatchGAN.  **Per-patch logits are kept.**
  There is no global-scalar output and no option for one: the
  46,800-tokens-to-one-scalar collapse of the LADD critic
  (``GAN_ARCHITECTURE_BRIEF`` §2.3) is the thing this design exists to fix.
* :func:`d_loss`, :func:`g_loss` — §5.1 patchwise, **non-relativistic**
  NS-logistic (default) or hinge.  Reals and fakes here are unrelated scenes,
  so per-position relativistic pairing is meaningless noise; no relativistic
  variant exists, not even behind a flag.
* :func:`r1_penalty` — §5.3, **ONE** finite-difference estimator on the
  per-image **MEAN** patch score.  No autograd variant, no sum variant, no
  per-patch variant, no ``pix_r1_normalize`` knob.
* :class:`PixCounters` — §5.4/§7 **monotone** counters.  No per-step gauge form
  is offered (the gauge is what aliased to a permanent 0 and produced the
  months-long "R2 never fired" false alarm, A10).
* :func:`c1_structured_hf` + :func:`sweep_c1_amplitude` — the §8.1 **C1**
  positive control and its offline calibration entry point.
* :func:`patch_logit_separation`, :func:`positive_control_readout` — the §8.1
  ROC-AUC / mean-gap readout with a bootstrap CI **over crops**.
* :func:`measure_receptive_field`, :func:`effective_sample_count` — the
  MEASURED locality of the critic, and §3.7's sample-count arithmetic
  recomputed on it (§3.7's own numbers assumed a receptive field this build
  does not have — see below).
* :func:`synthetic_stability_smoke` — the cheap, explicitly SYNTHETIC check on
  the recorded risk of dropping the GroupNorm.  It is a falsifier ("not
  obviously broken"), never evidence about the real arm.

R2 DOES NOT EXIST.  There is no ``pix_r2_gamma``, ``pix_r2_sigma``,
``pix_r2_every_n``, ``pix_r2_phase_offset``, no R2 fire counter and no
``pix_r2_*`` log key anywhere in this file.  ``testing/test_pixel_texture_disc.py``
greps this source and asserts the symbol is absent — a spec-drift tripwire the
doc explicitly asks for (§5.3, §7).

NOT REUSED FROM ``model/ladd_disc.py``
--------------------------------------
Nothing.  §5.3: four separately-written R1 estimators survive on the LADD side
(matched micro-batched, matched inline, positional autograd, positional FD),
normalised at different times and carrying different cadence state.  This
module imports nothing from it.  The one thing borrowed is the *idea* of the
monotone fire counter (§5.4), reimplemented here in ten lines.

FROZEN INTERFACE CONTRACT (``GAN_REDESIGN.md`` B, contract 1)
------------------------------------------------------------
``PixelTextureDisc.forward(px [N,3,H,W] in [-1,1]) -> [N,1,h,w]`` and the
trainer attribute is named ``pixel_texture_disc``.  That is exactly what
``model/disc_holdout_probe.py`` already expects
(``disc_holdout_probe_critic_attr`` defaults to ``"pixel_texture_disc"``, and
``_reduce_scores`` mean-reduces an ``[N,1,h,w]`` map), so the A22 probe needs
zero edits.

RUNNING THE TESTS (read this before you lose half an hour)
----------------------------------------------------------
The default ``python`` on this cluster has **no torch**; use
``/scratch/u6ex/as1748.u6ex/miniforge3/envs/arrwm/bin/python``.  And the login
node reports ``nproc=144``, so torch grabs 144 threads and thrashes
catastrophically on these small CPU convs — the suite appears to hang and
takes >30 min.  **Always cap the threads**::

    cd /scratch/u6ex/as1748.u6ex/ARRWM
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONPATH=. \
      /scratch/u6ex/as1748.u6ex/miniforge3/envs/arrwm/bin/python \
      -m pytest -q testing/test_pixel_texture_disc.py

Same suite, same machine: 21 s instead of 30+ min.

CONFIG SURFACE
--------------
This module never reads yaml.  Every ``pix_*`` value is a constructor or
function argument, and the spec defaults are exported as the module-level
constants below so the trainer can wire them without duplicating magic
numbers.  ``pix_gan_weight`` deliberately has **no usable default** — §5.5
withdraws 0.03 (it was measured under a *scalar latent* critic and does not
transfer) — see :data:`PIX_GAN_WEIGHT_DEFAULT`.

MEASURED, NOT ASSUMED (see the module tests; every number below was measured
on this exact class, not copied from the doc)
---------------------------------------------------------------------------
* **AUTHORISED DEVIATION FROM §4: THE GroupNorm IS REMOVED.**  Researcher
  decision, 2026-08-23, taken after measurement; recorded here and in
  :class:`PixelTextureDisc` so it reads as a decision and not as drift.  §4's
  layer table specifies ``GroupNorm(8, ...)`` on blocks 2 and 3; this build
  omits it and ships ``use_norm=False``.  Everything else in the table —
  layers, widths, kernels, strides, padding, spectral norm on all four convs,
  LeakyReLU(0.2) — is unchanged.  **Do not restore the norm.**  Rationale, in
  one line: GroupNorm normalises over ``(C, H, W)`` per sample, which made
  every patch logit a function of every input pixel, and B2's frozen question
  is whether *LOCAL* pixel adversarial feedback works.  Full before/after
  measurements in the two bullets below.
* **Parameter count = 661,185 (0.66 M), NOT the "~2-3 M" of §4** — and no
  longer the 661,953 measured before the norm was dropped.  The §4 layer table
  is exact except for the norm; 3->64->128->256->1 with k4/k4/k4/k3 simply does
  not reach 2 M.  Reported, not silently "fixed" by widening the net::

      conv1 3->64    k4:   3*64*16  + 64  =   3,136
      conv2 64->128  k4:  64*128*16 + 128 = 131,200   (GN(8,128) 256 REMOVED)
      conv3 128->256 k4: 128*256*16 + 256 = 524,544   (GN(8,256) 512 REMOVED)
      conv4 256->1   k3:   256*1*9  + 1   =   2,305
                                            ---------
                                              661,185      (was 661,953)

* **The SHIPPED critic IS now spatially local: 38 px, off-patch gradient at
  float noise.**  §4's *"a local texture question"* is finally true of what
  trains; its *"receptive field ~= 70 px"* never was, and still is not.
  Re-measured at 176x240 (the real post-trim crop) after the norm was dropped,
  backpropping one centre patch logit to the input, 12 random inits:

  ============  =================  ========================  =================
  config        non-zero grad box  off-38x38 |grad| (L1)     off-38x38 grad^2
  ============  =================  ========================  =================
  DEFAULT(off)  38 x 38            < 2.4e-07 (float noise)   < 1.4e-07
  normed (on)   176 x 240 (ALL)    0.0399 .. 0.0886          5.07e-05..2.9e-04
  ============  =================  ========================  =================

  The off-patch residual on the default is not merely small, it is not
  sign-definite (-1.2e-07 .. +2.4e-07 over 12 inits) — the signature of a true
  zero computed by subtracting two nearly-equal float32 sums.  Unchanged with
  spectral norm ON and on random rather than constant inputs (38x38, exactly
  0.0 off-patch).  The 38 is the conv geometry, closed form back-to-front:
  k3s1 -> 3 ; k4s2 -> 8 ; k4s2 -> 18 ; k4s2 -> 38.  §4's ~70 px is the pix2pix
  PatchGAN, which has FIVE layers; §4's table has four.
  :func:`measure_receptive_field` reports the default, the conv geometry and
  the old normed variant side by side.
* **§3.7's effective-sample arithmetic was built on two wrong inputs and is
  CORRECTED here** — see :func:`effective_sample_count`.  §3.7 assumed 768
  patch logits (untrimmed; the real number is 660) and ~6 non-overlapping
  receptive-field tiles per image (from the phantom 70 px RF).  With the
  measured 38 px RF at 176x240 there are ``floor(176/38) * floor(240/38) =
  4 * 6 = 24`` tiles, and the overlap factor is ``660/24 = 27.5 : 1``, not
  ~128:1.  The specced ``pix_crops_per_step=4, pix_frames_per_crop=3``
  therefore yields ~288 effective samples per step, not §3.7's ~72 — already
  2x §3.7's own ``crops=8`` escalation target of ~144.
* **``conv4.bias`` receives an EXACTLY ZERO gradient from R1 alone.**  Not a
  bug and not something to work around: the head bias adds the same constant
  to every patch logit, so it enters the per-image MEAN score ``s_n``
  identically at ``x`` and at ``x + eps*sigma`` and cancels *exactly* in the
  finite difference.  ``d(R1)/d(conv4.bias) == 0`` analytically, for any input.
  DDP is unaffected: ``find_unused_parameters=False`` requires every parameter
  to *participate in the graph* (``p.grad is not None``), not to be non-zero,
  and autograd cannot know the term cancels — it materialises a zeros tensor,
  not ``None``.  In the real D-update R1 is summed with ``d_loss``, which does
  give the bias a non-zero gradient (measured 0.134 on a random batch), so
  this is a curiosity of the penalty in isolation, never a training defect.
  Both facts are pinned by tests.
* **P = 660 patch logits per image at the real crop size, NOT 768.**  Shape
  trace: ``pix_crop_lat=(24,32)`` -> 8x VAE decode -> 192x256 -> §3.2 8-px
  border trim -> 176x240 -> stride-8 patch grid -> 22x30 = **660**.  The doc's
  768 is the *untrimmed* 192/8 x 256/8.  No function here ever uses a constant
  P: it is always read off the tensor, and ``r1_penalty`` returns it.
* **``gsq`` is NOT scale-free in P.  §5.3's "scale-free by construction" is
  FALSE, and so is its "a sum-based gsq scales with P^2".**  RE-MEASURED on
  the new norm-free default (fixed smooth+texture content, centre-cropped to
  3 sizes; fitted on the estimator's exact expectation
  ``E_eps[gsq] = ||d/dx mean_i D_i||^2``, so the fit carries no Monte-Carlo
  noise).  Seed-0 net:

  ======  ===================  ===================  ==========================
  P       ``E[gsq]`` norm OFF  ``E[gsq]`` norm ON   pairwise alpha (norm OFF)
  ======  ===================  ===================  ==========================
  165     4.321e-05            9.481e-03            .
  660     1.309e-05            2.546e-03            -0.862  (165 -> 660)
  2640    3.774e-06            4.569e-04            -0.897  (660 -> 2640)
  ======  ===================  ===================  ==========================

  OLS over the 16x range, seed-0 net: **alpha = -0.879** (was -1.094 with the
  norm).  Over 10 random inits: **-1.047 .. -0.853, mean -0.947** (was
  -1.282 .. -0.939, mean -1.056).  Sum reduction: **alpha = +1.121** seed-0,
  +0.953 .. +1.147 over 10 inits (was +0.906 / +0.718 .. +1.061) — the spec
  says +2.

  **alpha did NOT move materially, and that is the informative result.**  The
  worry behind re-fitting it was that GroupNorm might have been *why* patch
  gradients looked near-independent.  It was not: dropping the norm left the
  exponent in the same ~1/P regime and, if anything, TIGHTENED it (init-to-init
  spread 0.194 wide, against 0.343 with the norm).  The mechanism the old
  docstring cited — "patch gradients are LOCAL and nearly independent, so
  ``||sum_i grad D_i||^2 ~= P ||grad D_i||^2``, and the mean's ``1/P^2`` leaves
  ``~1/P``" — was previously the right answer quoted for a critic that was
  provably NOT local.  It now describes the network it is stated about.  The
  doc's ``P^2`` reasoning remains the perfectly-CORRELATED-patch case, which a
  patch critic is built not to be.
* **WHAT DID move — by TWO ORDERS OF MAGNITUDE — is the MAGNITUDE of
  ``gsq``.**  This is a launch-configuration input, not a curiosity.  Measured
  ``E[gsq]`` at P=660 on the real 176x240 crop over 10 **matched** random
  inits (matched exactly: the ``use_norm`` branch consumes no RNG, so at a
  given seed the four conv weights are bit-identical between the two builds —
  a test asserts this, so the comparison is not confounded by re-draws):

      norm ON   1.244e-03 .. 2.662e-03
      norm OFF  1.015e-05 .. 2.120e-05
      ratio     **95x .. 214x, mean 129x**

  The exponent-fit canvas gave 195x at seed 0, so the honest headline is
  ``~10^2``, content- and init-dependent, not a constant.  GroupNorm was
  rescaling activations and inflating the input-gradient scale with them.
  ``R1 = 0.5 * gamma * gsq`` is linear in this, so the shipped
  ``pix_r1_gamma = 1.0`` now contributes ~100x less regularisation relative to
  ``d_loss`` than the same number did under the normed critic.  It is an
  at-init measurement on random weights and the ratio will drift as D trains —
  but at-init is exactly when R1 is supposed to be doing its work.
  **``pix_r1_gamma`` must be re-calibrated from a measured
  ``pix_r1_grad_sq_mean`` (§5.3) before this arm is trusted; 1.0 was never an
  empirical calibration and is now ~100x weaker than whatever it was
  informally justified by.**  The estimator itself is unaffected (FD/autograd
  agreement pinned by tests) — what changed is the quantity it measures.
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Statistics come from the shared module.  §0/§9: "no second implementation".
from analysis.texture_stats import texture_battery

__all__ = [
    "PIX_DEFAULTS",
    "PIX_CROP_LAT",
    "PIX_FRAMES_PER_CROP",
    "PIX_CROPS_PER_STEP",
    "PIX_BAND_COUNT",
    "PIX_REALS_PER_FAKE",
    "PIX_GAN_LR",
    "PIX_LOSS_FORM",
    "PIX_R1_GAMMA",
    "PIX_R1_SIGMA",
    "PIX_R1_EVERY_N",
    "PIX_R1_NUM_SAMPLES",
    "PIX_GAN_WEIGHT_DEFAULT",
    "PIX_GAN_WEIGHT_NOTE",
    "SPECTRAL_NORM_API",
    "PixelTextureDisc",
    "d_loss",
    "g_loss",
    "r1_penalty",
    "r1_subsample_indices",
    "PixCounters",
    "patch_logit_telemetry",
    "patch_logit_spatial_variance",
    "measure_receptive_field",
    "effective_sample_count",
    "synthetic_stability_smoke",
    "CONV_GEOMETRIC_RF",
    "PIX_R1_GAMMA_CALIBRATION_P",
    "c1_structured_hf",
    "sweep_c1_amplitude",
    "c1_calibration_verdict",
    "B_BATTERY_REF",
    "C_LATE_BATTERY_REF",
    "roc_auc_fast",
    "patch_logit_separation",
    "positive_control_readout",
]

# ---------------------------------------------------------------------------
# Config surface — the spec defaults, exported so the trainer wires ONE copy.
# ---------------------------------------------------------------------------
PIX_CROP_LAT: Tuple[int, int] = (24, 32)      # §3.2 latent crop (rows, cols)
PIX_FRAMES_PER_CROP: int = 3                  # §3.3 [A3], fake-side device only
PIX_CROPS_PER_STEP: int = 4                   # §3.7, raise to 8 on evidence
PIX_BAND_COUNT: int = 3                       # §3.6 [A24] COARSE thirds, pinned
PIX_REALS_PER_FAKE: int = 1                   # §3.4a [A20] invariant, not a budget
PIX_GAN_LR: float = 1e-5                      # §5.2, Adam betas (0.0, 0.9)
PIX_GAN_BETAS: Tuple[float, float] = (0.0, 0.9)
PIX_LOSS_FORM: str = "nsgan"                  # §5.1, {"nsgan", "hinge"}
#: §5.3 ``gamma`` is defined against the MEAN patch score -- but that does NOT
#: make it crop-size portable (§5.3's "scale-free in P" is measured false;
#: ``gsq ~ P^-0.88`` on the norm-free build, see the module docstring and
#: :func:`r1_penalty`).  This value is TIED TO THE CROP IT WAS CALIBRATED AT:
#: the spec crop, P = 660.
#: **INERT-GAMMA WARNING (2026-08-23).**  ``1.0`` is not a weak setting, it is
#: an ABSENT one, and it was absent on BOTH architectures — do not read this as
#: "the GroupNorm removal broke a good number".  Measured ``R1 / d_loss`` at
#: init (``d_loss = 2*ln2 = 1.3863`` for nsgan) at P=660 on the real 176x240
#: crop, 10 MATCHED inits, at this very ``gamma = 1.0``:
#:
#:     norm ON (the OLD build)   4.485e-04 .. 9.603e-04   -> <0.1 % of d_loss
#:     norm OFF (SHIPPED)        3.660e-06 .. 7.646e-06   -> <0.001 % of d_loss
#:
#: The authorised GroupNorm removal did drop ``E[gsq]`` by ~10^2 at fixed P and
#: matched init (95x..214x, mean 129x) and R1 is linear in it — but the normed
#: build was already under the same 0.1 % line, so the removal is a ~100x
#: worsening of an already-inert term, not the origin of it.  Equivalently, the
#: gamma that would put R1 at 0.1 % of ``d_loss`` is 1.04..2.23 under the norm
#: and 131..273 without it: ``1.0`` is below both.  There is therefore NO
#: pre-removal value to restore.  Re-calibrate from a measured
#: ``pix_r1_grad_sq_mean`` (§5.3) before the arm is trusted; the exponent barely
#: moved, but the magnitude did, and the magnitude was never calibrated.
#: Pinned by ``test_gsq_MAGNITUDE_collapsed_when_the_norm_was_removed``.
PIX_R1_GAMMA: float = 1.0
#: The crop ``PIX_R1_GAMMA`` is calibrated at.  Not read by any code path --
#: it exists so a reader who changes ``pix_crop_lat`` can see what the gamma
#: they inherited was measured against.  ``r1_penalty`` returns the ACTUAL P.
PIX_R1_GAMMA_CALIBRATION_P: int = 660
PIX_R1_SIGMA: float = 0.01                    # §5.3, in the [-1,1] input scale
PIX_R1_EVERY_N: int = 1                       # §5.4, R1 fires on EVERY D-update
PIX_R1_NUM_SAMPLES: Optional[int] = None      # §5.4, optional cost cap; None = all
PIX_DECODE_BORDER_TRIM: int = 8               # §3.2, px trimmed each side

#: §5.5 **`pix_gan_weight=0.03` is WITHDRAWN as a default.**  The measured
#: 0.01-inert / 0.03-learns / 1.0-destroys bracket was established under a
#: *scalar latent* critic: new domain (pixels), new reduction (patch logits,
#: nonlinearity before the mean), new loss form.  None of it transfers.  This
#: constant is ``None`` on purpose so a caller that forgets to calibrate gets a
#: loud failure instead of an inherited constant; :func:`resolve_gan_weight`
#: is the only sanctioned way to turn it into a number.
PIX_GAN_WEIGHT_DEFAULT: Optional[float] = None
PIX_GAN_WEIGHT_NOTE: str = (
    "pix_gan_weight has NO default (TEXTURE_GAN_DESIGN §5.5 withdraws 0.03). "
    "Calibrate it: run a short weight-free probe, read `gan_dmd_grad_ratio` "
    "(A7 telemetry, gan_grad_telemetry_every=25), and pick the weight that "
    "lands the ratio in the 5-20% band, then bracket x1/3, x1, x3 around it. "
    "Historical LADD arms measured 0.0005-0.012, i.e. those critics were "
    "small and near-irrelevant. There is NO GAN-side gradient cap to fall "
    "back on: gan_grad_target_norm was REMOVED 2026-08-23."
)

PIX_DEFAULTS: Dict[str, Any] = {
    "pix_crop_lat": PIX_CROP_LAT,
    "pix_frames_per_crop": PIX_FRAMES_PER_CROP,
    "pix_crops_per_step": PIX_CROPS_PER_STEP,
    "pix_band_count": PIX_BAND_COUNT,
    "pix_reals_per_fake": PIX_REALS_PER_FAKE,
    "pix_gan_lr": PIX_GAN_LR,
    "pix_gan_betas": PIX_GAN_BETAS,
    "pix_loss_form": PIX_LOSS_FORM,
    "pix_r1_gamma": PIX_R1_GAMMA,
    "pix_r1_sigma": PIX_R1_SIGMA,
    "pix_r1_every_n": PIX_R1_EVERY_N,
    "pix_r1_num_samples": PIX_R1_NUM_SAMPLES,
    "pix_decode_border_trim": PIX_DECODE_BORDER_TRIM,
    "pix_gan_weight": PIX_GAN_WEIGHT_DEFAULT,   # None -> must be calibrated
}


def resolve_gan_weight(value: Optional[float], *, strict: bool = True) -> float:
    """Turn a caller-supplied ``pix_gan_weight`` into a number, loudly.

    ``None`` (the default state) means *not calibrated*.  Under ``strict`` that
    raises; otherwise it returns ``0.0`` — the pixel G-term contributes nothing
    — and never 0.03.  §5.5.
    """
    if value is None:
        if strict:
            raise ValueError(PIX_GAN_WEIGHT_NOTE)
        return 0.0
    return float(value)


# ---------------------------------------------------------------------------
# Spectral norm — prefer the modern parametrization API.
# ---------------------------------------------------------------------------
try:  # torch >= 1.12
    from torch.nn.utils.parametrizations import spectral_norm as _spectral_norm
    SPECTRAL_NORM_API = "torch.nn.utils.parametrizations.spectral_norm"
except Exception:  # pragma: no cover - ancient torch
    from torch.nn.utils import spectral_norm as _spectral_norm  # type: ignore
    SPECTRAL_NORM_API = "torch.nn.utils.spectral_norm"


# ---------------------------------------------------------------------------
# §4 — the critic
# ---------------------------------------------------------------------------
class PixelTextureDisc(nn.Module):
    """PatchGAN texture critic, from scratch, per ``TEXTURE_GAN_DESIGN`` §4 —
    **with one authorised deviation: the GroupNorm is removed.**

    ::

        input  [N, 3, h, w] in [-1, 1]
        conv 3->64    k4 s2  spectral-norm, LeakyReLU(0.2)
        conv 64->128  k4 s2  spectral-norm, LeakyReLU(0.2)     <- GN dropped
        conv 128->256 k4 s2  spectral-norm, LeakyReLU(0.2)     <- GN dropped
        conv 256->1   k3 s1  spectral-norm
                                                       -> [N, 1, h/8, w/8]

    ``padding=1`` on every conv.  The doc's table omits padding, but ``h/8``
    and ``w/8`` are only exact with ``p=1`` on the three stride-2 convs (and
    ``p=1`` on the k3 head to keep the grid size).  Verified: 176 -> 88 -> 44
    -> 22 and 240 -> 120 -> 60 -> 30.  Layer count, channel widths, kernels,
    strides, padding, spectral norm and the LeakyReLU(0.2) slope are all
    UNCHANGED from §4; the norm is the only difference.

    AUTHORISED DEVIATION FROM §4 — GroupNorm REMOVED
    ------------------------------------------------
    **Researcher decision, 2026-08-23, taken after the measurement below.**
    §4's layer table puts ``GroupNorm(8, ...)`` on blocks 2 and 3.  This build
    omits it and **ships norm-OFF by default** (``use_norm=False``).  That is a
    recorded, deliberate deviation — NOT drift.  Do not "restore the spec" by
    flipping the default back to ``True``, and do not delete this note.

    WHY — the measurement that forced it.  GroupNorm normalises over
    ``(C, H, W)`` **per sample**, so with it every patch logit is a function of
    every input pixel.  Measured on this class at the real 176x240 post-trim
    crop, backpropping a single CENTRE patch logit to the input, 12 random
    inits each:

    ========  ==================  ======================  =====================
    config    non-zero grad bbox  off-38x38 |grad| (L1)   off-38x38 grad^2
    ========  ==================  ======================  =====================
    norm ON   176 x 240 (ALL)     0.0399 .. 0.0886        5.07e-05 .. 2.90e-04
    norm OFF  38 x 38             < 2.4e-07 (float noise) < 1.4e-07
    ========  ==================  ======================  =====================

    So §4's *"receptive field ~= 70 px — a local texture question"* was false
    of the normed build in the way that matters: its support was the entire
    image.  (The conv geometry was never 70 px either — it is 38; see
    :data:`CONV_GEOMETRIC_RF`.  The pix2pix 70x70 PatchGAN has FIVE layers,
    §4's table has four.  With the norm gone, 38 is now both the geometric and
    the effective number, and the "local texture question" description is
    finally true of what trains.)

    WHY IT WAS WORTH DEVIATING.  The B2 scope freeze asks *"can LOCAL
    decoded-pixel adversarial feedback suppress fabricated directional
    texture?"*.  A critic every one of whose logits reads the whole crop does
    not test that question, and a null result from it would be
    uninterpretable — "local texture feedback does not work" would be
    indistinguishable from "the critic was never local".  §3.7's
    effective-sample-count arithmetic is likewise built on a patch-locality
    argument the normed critic did not satisfy; see
    :func:`effective_sample_count` for the corrected numbers.

    RECORDED RISK, accepted knowingly.  An unnormalised from-scratch PatchGAN
    can train less stably than a normed one — spectral norm on all four convs
    is the only conditioning left.  This cannot be settled on CPU.
    :func:`synthetic_stability_smoke` is the cheap check that exists, and it is
    a SYNTHETIC smoke on a canvas, not evidence about the real arm.

    SECOND-ORDER CONSEQUENCE the launch config must absorb: removing the norm
    drops the MAGNITUDE of ``gsq`` (the raw R1 quantity) by **two orders of
    magnitude** at fixed P and matched init.  Measured ``E[gsq]`` at P=660 on
    the real 176x240 crop, 10 matched random inits — matched exactly: the
    ``use_norm`` branch consumes no RNG, so at a given seed the four conv
    weights are bit-identical between the two builds (asserted by a test):

        norm ON   1.244e-03 .. 2.662e-03
        norm OFF  1.015e-05 .. 2.120e-05
        ratio     **95x .. 214x, mean 129x**   (a different content canvas
                                               gave 195x at seed 0, so treat
                                               this as ~10^2, not a constant)

    ``R1 = 0.5 * gamma * gsq`` is linear in that, so ``pix_r1_gamma = 1.0``
    now buys ~100x less regularisation than the same number did under the
    normed critic.  **It must be re-calibrated.**  See :func:`r1_penalty`.

    **From-scratch is a recorded exception** to GAN_REDESIGN standing decision
    4, resolved 2026-08-23 on the VQGAN/SD precedent (our latent space was
    itself produced by a from-scratch pixel PatchGAN, adopted for exactly our
    symptom).  §4.1.  It carries a budget risk: ``d_loss ~ ln 2`` reads
    identically for *undertrained* and *wrong design*, which is why the §8.1
    positive control in this module is mandatory, not optional.

    **Per-patch logits are kept.  There is no global scalar output and no
    option for one.**

    Args:
        in_channels: 3 (decoded RGB).
        base_channels: 64 -> 128 -> 256 widths; the spec value is 64.
        groups: GroupNorm group count, 8 per spec.  Consulted ONLY when
            ``use_norm=True``, i.e. only for the comparison variant.
        negative_slope: LeakyReLU slope, 0.2 per spec.
        use_spectral_norm: escape hatch for the receptive-field probe and for
            exact-arithmetic unit tests only.  Leave ``True`` in training.
        use_norm: **DEFAULT ``False`` — the authorised deviation recorded
            above (researcher decision, 2026-08-23).**  The shipped/default
            critic carries no GroupNorm and is genuinely local: a centre patch
            logit's input gradient is confined to a 38x38 box, with < 2.4e-07
            of its L1 mass outside it (float noise).

            ``use_norm=True`` reconstructs the OLD normed variant.  It exists
            so the two can be compared and so the locality guard has a
            planted-violation companion that actually fails — it is **not** a
            training option.  With it on, GroupNorm normalises over
            ``(C, H, W)`` per sample, every patch logit depends on every input
            pixel, the non-zero-gradient support is the whole 176x240 crop,
            and 4.0-8.9 % of a patch logit's input-gradient L1 mass sits
            outside its 38x38 box.  That is the configuration B2 was measured
            NOT to want.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        groups: int = 8,
        negative_slope: float = 0.2,
        *,
        use_spectral_norm: bool = True,
        use_norm: bool = False,          # AUTHORISED DEVIATION from §4 -- see
                                         # the class docstring.  Norm-OFF is
                                         # the shipped critic, on purpose.
    ) -> None:
        super().__init__()
        c1 = int(base_channels)
        c2 = c1 * 2
        c3 = c1 * 4

        def sn(m: nn.Module) -> nn.Module:
            return _spectral_norm(m) if use_spectral_norm else m

        def norm(c: int) -> nn.Module:
            return nn.GroupNorm(int(groups), c) if use_norm else nn.Identity()

        self.negative_slope = float(negative_slope)
        self.groups = int(groups)
        self.uses_spectral_norm = bool(use_spectral_norm)
        self.uses_norm = bool(use_norm)

        # Block 1 — never had a norm (§4 puts GroupNorm on the MIDDLE blocks).
        self.conv1 = sn(nn.Conv2d(int(in_channels), c1, 4, 2, 1))
        # Block 2 — ``norm2`` is nn.Identity() by default.  AUTHORISED
        # DEVIATION from §4 (see class docstring): GroupNorm normalised over
        # (C,H,W) and made every patch logit global; measured 4.0-8.9 % of a
        # centre logit's input-grad L1 mass outside its 38x38 box, vs < 2.4e-07
        # without.  The attributes stay so ``use_norm=True`` still builds the
        # old variant for comparison.
        self.conv2 = sn(nn.Conv2d(c1, c2, 4, 2, 1))
        self.norm2 = norm(c2)
        # Block 3 — same.
        self.conv3 = sn(nn.Conv2d(c2, c3, 4, 2, 1))
        self.norm3 = norm(c3)
        # Head — NO norm, NO activation: raw per-patch logits.
        self.conv4 = sn(nn.Conv2d(c3, 1, 3, 1, 1))

    # -- geometry helpers ---------------------------------------------------
    @staticmethod
    def patch_grid(h: int, w: int) -> Tuple[int, int]:
        """Output grid for an ``h x w`` input.  Stride 8 with ``p=1`` throughout.

        176x240 -> (22, 30) -> P = 660.  Never hard-code P; read it off the
        tensor (or from here).
        """
        for _ in range(3):
            h = (h + 2 - 4) // 2 + 1
            w = (w + 2 - 4) // 2 + 1
        return int(h), int(w)

    @classmethod
    def patch_count(cls, h: int, w: int) -> int:
        gh, gw = cls.patch_grid(h, w)
        return int(gh * gw)

    # -- forward ------------------------------------------------------------
    def forward(self, px: torch.Tensor) -> torch.Tensor:
        """``px`` ``[N, 3, H, W]`` in ``[-1, 1]`` -> patch logits ``[N, 1, h, w]``.

        No sigmoid — these are logits, consumed raw by :func:`d_loss` /
        :func:`g_loss` and mean-reduced by ``disc_holdout_probe._reduce_scores``.
        """
        if px.dim() != 4:
            raise ValueError(
                f"PixelTextureDisc expects [N,3,H,W]; got {tuple(px.shape)}"
            )
        s = self.negative_slope
        x = F.leaky_relu(self.conv1(px), s, inplace=False)
        x = F.leaky_relu(self.norm2(self.conv2(x)), s, inplace=False)
        x = F.leaky_relu(self.norm3(self.conv3(x)), s, inplace=False)
        return self.conv4(x)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        gn = (f"groupnorm={self.uses_norm}(groups={self.groups})"
              if self.uses_norm else
              "groupnorm=False (AUTHORISED §4 deviation, 2026-08-23)")
        return (
            f"spectral_norm={self.uses_spectral_norm} ({SPECTRAL_NORM_API}), "
            f"{gn}, slope={self.negative_slope}"
        )


# ---------------------------------------------------------------------------
# §5.1 — patchwise, NON-RELATIVISTIC adversarial losses
# ---------------------------------------------------------------------------
_LOSS_FORMS = ("nsgan", "hinge")


def _check_form(loss_form: str) -> str:
    f = str(loss_form).lower()
    if f not in _LOSS_FORMS:
        raise ValueError(
            f"pix_loss_form must be one of {_LOSS_FORMS}; got {loss_form!r}. "
            "There is deliberately no relativistic / RpGAN option (§5.1): "
            "reals and fakes here are unrelated scenes, so per-position "
            "relativistic pairing is meaningless noise."
        )
    return f


def d_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
    loss_form: str = PIX_LOSS_FORM,
) -> Dict[str, torch.Tensor]:
    """§5.1 discriminator loss.  **Nonlinearity per patch, BEFORE any averaging.**

    ``nsgan``::

        D_loss = mean(softplus(-D(real))) + mean(softplus(D(fake)))

    ``hinge``::

        D_loss = mean(relu(1 - D(real))) + mean(relu(1 + D(fake)))

    The mean runs over **both** the patch grid and the frame/crop batch rows —
    ``mean_i[f(D_i)]``, never ``f(mean_i[D_i])``.  Applying the nonlinearity
    after the mean throws away the whole point of keeping patch logits: by
    Jensen the two differ, and the post-mean form is exactly the scalar-critic
    collapse §4 exists to avoid.

    Returns a dict of tensors (``d_loss`` carries grad; the rest are the §7
    telemetry scalars, detached).
    """
    f = _check_form(loss_form)
    if f == "nsgan":
        real_term = F.softplus(-real_logits).mean()
        fake_term = F.softplus(fake_logits).mean()
    else:
        real_term = F.relu(1.0 - real_logits).mean()
        fake_term = F.relu(1.0 + fake_logits).mean()
    total = real_term + fake_term
    return {
        "d_loss": total,
        "d_real_term": real_term.detach(),
        "d_fake_term": fake_term.detach(),
        "d_real_mean": real_logits.detach().float().mean(),
        "d_fake_mean": fake_logits.detach().float().mean(),
    }


def g_loss(
    fake_logits: torch.Tensor,
    loss_form: str = PIX_LOSS_FORM,
) -> torch.Tensor:
    """§5.1 generator loss.  Non-saturating; patchwise; nonlinearity before mean.

    ``nsgan``:  ``mean(softplus(-D(fake)))`` — literally §5.1.

    ``hinge``:  ``-mean(D(fake))``.  **Recorded ambiguity.**  §5.1 says "hinge
    is the same reduction with the hinge nonlinearity", which read literally
    would give ``mean(relu(1 - D(fake)))``.  That variant has *zero* gradient
    for every patch the critic already scores above +1, i.e. the generator's
    signal dies exactly on the patches it has not fixed being the ones that
    still count — it is a known-bad form.  ``-mean(D(fake))`` is the universal
    hinge-GAN generator loss (SAGAN / BigGAN) and is what is implemented.  It
    is still applied per patch before averaging (the map is affine here, so the
    Jensen test in the unit tests is stated on the D side, where the
    nonlinearity is genuinely nonlinear in both forms).
    """
    f = _check_form(loss_form)
    if f == "nsgan":
        return F.softplus(-fake_logits).mean()
    return (-fake_logits).mean()


# ---------------------------------------------------------------------------
# §5.3 — R1.  ONE estimator.  MEAN patch score.  Finite difference on pixels.
# ---------------------------------------------------------------------------
def r1_subsample_indices(
    n: int,
    num_samples: Optional[int],
    generator: Optional[torch.Generator] = None,
) -> Optional[torch.Tensor]:
    """Draw the ``pix_r1_num_samples`` subsample — **DDP-safe by construction**.

    THE CONTRACT the trainer relies on: this function *never* touches global
    RNG.  It draws from the explicit ``generator`` you pass, or from nothing at
    all.  Under DDP the caller must make every rank agree on the draw — either
    pass a ``torch.Generator`` seeded identically on all ranks, or (preferred,
    matching ``_sample_critic_grad_frame_indices``) draw on rank 0 and
    broadcast the resulting index tensor, then hand it to :func:`r1_penalty`
    as ``indices=`` and skip this function entirely.

    Returns ``None`` when no subsampling is needed (use all ``n`` reals).
    """
    if num_samples is None:
        return None
    k = int(num_samples)
    if k <= 0:
        raise ValueError("pix_r1_num_samples must be >= 1 (or None for all)")
    if k >= int(n):
        return None
    if generator is None:
        raise ValueError(
            "r1_subsample_indices needs an explicit torch.Generator (or a "
            "caller-broadcast `indices` list) — this path must never touch "
            "global RNG, because under DDP the ranks would then disagree on "
            "which reals were perturbed."
        )
    perm = torch.randperm(int(n), generator=generator)
    return perm[:k].to(torch.long)


def r1_penalty(
    disc: Callable[[torch.Tensor], torch.Tensor],
    real_px: torch.Tensor,
    *,
    gamma: float = PIX_R1_GAMMA,
    sigma: float = PIX_R1_SIGMA,
    num_samples: Optional[int] = PIX_R1_NUM_SAMPLES,
    generator: Optional[torch.Generator] = None,
    indices: Optional[torch.Tensor] = None,
    eps: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    r"""§5.3 zero-centred R1 on the real branch.  The ONLY estimator in B2.

    ::

        s_n(x) = (1/P) * sum_i D_i(x_n)                      # per-image MEAN
        gsq    = mean_n( ( ( s_n(real_n + eps*sigma) - s_n(real_n) ) / sigma )^2 )
        R1     = 0.5 * gamma * gsq

    * **Single finite-difference estimator on the MEAN patch score.**  Not
      autograd, not the sum, not per-patch.
    * The finite difference is taken on **pixels**, ``sigma`` in the ``[-1,1]``
      input scale (``pix_r1_sigma = 0.01``).  ``eps`` is a fresh standard normal
      perturbation of the real input, drawn from the explicit ``generator``.
    * ``gamma = 1.0`` is **defined against the mean** — and the mean does NOT
      make it portable across crop sizes.  §5.3 asserts the mean reduction is
      "scale-free in ``P`` by construction" and that a sum-based ``gsq`` would
      scale with ``P^2``.  **Both claims are false, RE-MEASURED on the
      norm-free :class:`PixelTextureDisc` after the 2026-08-23 GroupNorm
      removal** (module docstring has the full table):

          mean reduction:  ``gsq ~ P^-0.879``   (spec says P^0)
          sum  reduction:  ``gsq ~ P^+1.121``   (spec says P^2)

      seed-0 net; over 10 random inits the mean-reduction exponent runs
      -1.047 .. -0.853.  **Dropping the norm did NOT move it materially** --
      with the norm it was -1.094 (10-init -1.282 .. -0.939) -- which matters,
      because the suspicion behind re-fitting was that GroupNorm might have
      been what made patch gradients *look* independent.  It was not; if
      anything the init-to-init spread TIGHTENED (0.194 wide vs 0.343).

      Why: patch gradients are LOCAL and nearly independent, so
      ``||sum_i grad D_i||^2 ~= P * ||grad D_i||^2``.  The mean divides by
      ``P^2``, leaving ``~ 1/P``.  That explanation is now stated about a
      critic that is *measurably* local (38 px, off-patch input-gradient share
      at float noise) rather than, as before, about one that was provably
      global.  The doc's ``P^2`` arithmetic assumes perfectly CORRELATED patch
      gradients, which a patch critic is designed not to have.

      **Practical consequence 1 -- ``gamma`` is tied to the crop it was
      calibrated at.**  At the spec crop (``pix_crop_lat=(24,32)`` ->
      176x240 -> ``P = 660``) ``gamma = 1.0`` is the inherited value.  Double
      the crop LINEARLY (P x4) and the same ``gamma`` delivers roughly
      ``4^0.88 ~= 3.4x`` WEAKER R1; halve it and ~3.4x stronger.  Anyone
      changing ``pix_crop_lat`` must recalibrate, or rescale ``gamma`` by
      ``(P_new / 660)^0.88`` and say so.

      **Practical consequence 2 -- ``gamma = 1.0`` buys ~100x LESS R1 than it
      did before the norm was removed, and must be re-calibrated -- but note
      that it was ALREADY inert before the removal, so this is a worsening and
      not the cause.**  At ``gamma = 1.0`` the R1 term is <0.1 % of ``d_loss``
      at init on the NORMED build (measured 4.485e-04 .. 9.603e-04 over the
      10 matched inits below) and <0.001 % on the shipped norm-free one
      (3.660e-06 .. 7.646e-06).  Re-calibration cannot mean "restore the
      pre-removal value": 1.0 never bought meaningful R1 on either build.
      At P=660 on the real 176x240 crop, over 10 MATCHED inits (identical conv
      weights; ``use_norm`` draws no RNG), ``E[gsq]`` fell from
      1.244e-03 .. 2.662e-03 to 1.015e-05 .. 2.120e-05 -- a ratio of
      **95x .. 214x, mean 129x** (another content canvas gave 195x, so read it
      as ~10^2, not a constant).  GroupNorm was rescaling activations and
      inflating the input-gradient scale with them.  The EXPONENT is
      essentially unchanged; the MAGNITUDE is not, and R1 =
      ``0.5 * gamma * gsq`` is linear in it.  §5.3's own procedure applies --
      run briefly, read ``pix_r1_grad_sq_mean`` (:class:`PixCounters`; one row
      is 126 % noise, do not calibrate off it), set ``gamma`` from that.  This
      is an at-init measurement on random weights and the ratio will drift as
      D trains, but at-init is exactly when R1 is meant to be doing its work.

      This module does NOT apply such a correction.  Picking a normalisation
      is a design decision for the researcher (and §5.3 forbids a
      ``pix_r1_normalize`` flag), so the honest thing available to the code is
      to report the truth: ``P`` is read off the tensor, never from a
      constant, and it is RETURNED so a reader can rescale.
      :data:`PIX_R1_GAMMA_CALIBRATION_P` records the crop the shipped gamma
      belongs to.
    * **The mean reduction is not a flag.**  There is no ``pix_r1_normalize``
      knob and no un-normalised branch.
    * ``pix_r1_every_n = 1``: R1 fires on **every** D-update.  There is no
      ``pix_r1_once_per_step`` and no 1-of-N cap.  If R1's cost ever needs
      bounding, use ``num_samples`` — subsampling keeps ``pix_r1_rate == 1.00``
      by construction, whereas raising ``every_n`` reintroduces the
      silent-cadence bug class (§5.4, A10/A16).

    DDP NOTE (confirmed, the trainer relies on it).  With
    ``find_unused_parameters=False``, a subsample is safe: this function runs
    the **full** network on the subsampled reals — both forwards traverse
    conv1..conv4 — so every parameter receives a gradient on every rank, on
    every D-update, regardless of ``num_samples``.  The only way to break that
    is an empty subsample, which :func:`r1_subsample_indices` refuses to
    produce (``k >= 1`` enforced).  Randomness is never drawn from global RNG;
    see :func:`r1_subsample_indices`.

    One gradient here is exactly zero, by construction: ``conv4.bias``.  The
    head bias shifts every patch logit by the same constant, so it enters the
    per-image MEAN ``s_n`` identically at ``x`` and at ``x + eps*sigma`` and
    cancels in the difference.  ``p.grad`` is still a (zeros) tensor rather
    than ``None``, which is what DDP actually requires, and the bias picks up
    a real gradient from ``d_loss`` in the same backward.

    Args:
        disc: the critic (or any ``[N,3,H,W] -> [N,1,h,w]`` callable).
        real_px: ``[N,3,H,W]`` reals, in ``[-1,1]``.  Detach before calling —
            R1 regularises D, not the data path.
        indices: caller-supplied (rank-0 broadcast) subsample.  Takes
            precedence over ``num_samples``/``generator``.
        eps: caller-supplied perturbation, for tests / exact reproduction.
            Must broadcast to the subsampled reals.

    Returns:
        ``{"r1": tensor (grad to D params), "gsq": float RAW pre-gamma value
        for the ``pix_r1_grad_sq`` log key, "n_used": int, "P": int}``.

        ``P`` is the MEASURED patch count for this call, read off the logit
        tensor.  It is not decoration: ``gsq`` scales as ``P^-0.92`` (above),
        so a ``gsq`` logged without its ``P`` cannot be compared across crop
        sizes and cannot be used to re-derive ``gamma``.  Log both.
        ``gsq`` is also very noisy per call — see :class:`PixCounters` and its
        ``r1_grad_sq_mean`` before calibrating anything off one row.
    """
    if real_px.dim() != 4:
        raise ValueError(f"r1_penalty expects [N,3,H,W]; got {tuple(real_px.shape)}")
    x = real_px.detach()
    n = int(x.shape[0])
    if n == 0:
        raise ValueError("r1_penalty got an empty real batch")

    idx = indices
    if idx is None:
        idx = r1_subsample_indices(n, num_samples, generator)
    if idx is not None:
        idx = torch.as_tensor(idx, dtype=torch.long, device=x.device).reshape(-1)
        if idx.numel() == 0:
            raise ValueError("r1_penalty subsample is empty; DDP would desync")
        x = x.index_select(0, idx)

    s = float(sigma)
    if s <= 0.0:
        raise ValueError("pix_r1_sigma must be > 0")

    if eps is None:
        if generator is not None and generator.device.type == x.device.type:
            e = torch.randn(x.shape, generator=generator, device=x.device,
                            dtype=x.dtype)
        elif generator is not None:
            e = torch.randn(x.shape, generator=generator,
                            device=generator.device,
                            dtype=torch.float32).to(x.device, x.dtype)
        else:
            e = torch.randn_like(x)
    else:
        e = eps.to(x.device, x.dtype).expand_as(x)

    logits_0 = disc(x)                       # [n, 1, h, w]
    logits_1 = disc(x + s * e)               # same graph, same params

    p = int(logits_0[0].numel())             # P read off the TENSOR, never a const
    s0 = logits_0.reshape(logits_0.shape[0], -1).mean(dim=1)   # per-image MEAN
    s1 = logits_1.reshape(logits_1.shape[0], -1).mean(dim=1)
    gsq = (((s1 - s0) / s) ** 2).mean()

    return {
        "r1": 0.5 * float(gamma) * gsq,
        "gsq": float(gsq.detach()),          # RAW, pre-gamma -> pix_r1_grad_sq
        "n_used": int(x.shape[0]),
        "P": p,
    }


# ---------------------------------------------------------------------------
# §5.4 / §7 — MONOTONE counters.  No gauge form is offered, on purpose.
# ---------------------------------------------------------------------------
class PixCounters:
    """Monotone penalty/update counters for the §7 day-one telemetry.

    The three keys the doc requires on the **first** logged row::

        pix_dupdate_total      monotone, +1 per D optimizer update
        pix_r1_fired_total     monotone, +1 per update on which R1 fired
        pix_r1_rate            derived   = fired / max(1, updates)

    ``pix_r1_rate`` **must read 1.00** with ``pix_r1_every_n = 1``; anything
    below 0.99 is a build bug, not a tuning outcome, and it is visible on the
    very first row by construction.

    **There is no per-step gauge form and there will not be one.**  The
    per-step gauge is what aliased to a permanent 0 and produced the "R2 never
    fired" false alarm that survived months of runs (A10, §5.4).  A monotone
    counter cannot alias: if it does not move, it is visibly not moving.

    There is no R2 counter — R2 does not exist in B2 (§5.3).

    R1-MAGNITUDE RUNNING MEAN (``gsq``)
    -----------------------------------
    §5.3 says to calibrate ``gamma`` from the first run's measured R1
    magnitude and §5.4 says to read it on the **first** logged row.  **One row
    cannot support that.**  Measured on :class:`PixelTextureDisc`, 300
    independent ``eps`` draws at ``sigma=0.01`` on fixed reals:

        1 real  per draw:  relative sd **126 %**, values 2.3e-09 .. 9.0e-03
                           (a 4e6x span)
        12 reals per draw: relative sd  **40 %**, values 2.3e-04 .. 3.7e-03
                           (a 16x span)

    The estimator is unbiased (pinned by
    ``test_fd_expectation_is_grad_norm_squared``), so the *mean* is right and
    a single draw is not: averaging 25 draws brings the window-to-window
    spread to 7.7 %, 100 draws to 4.7 %.

    So :meth:`note_d_update` optionally takes the step's ``gsq`` and keeps an
    incremental running mean of it.  ``log_dict`` then emits
    ``{prefix}r1_grad_sq_mean``, ``{prefix}r1_grad_sq_n`` and
    ``{prefix}r1_grad_sq_last``.  Per the omit-never-fake rule those three
    keys are **absent** until at least one ``gsq`` has been observed — a 0.0
    running mean would read as "R1 measured zero gradient", which is a real
    and alarming diagnostic, not a placeholder.

    WIRING NOTE (unfinished, and deliberately not faked here): the shipped
    log key ``train/pix_r1_grad_sq`` is emitted by
    ``trainer/causal_action_forcing_train.py``, which calls
    ``counters.note_d_update(r1_fired=True)`` WITHOUT the ``gsq``.  Until that
    call passes ``gsq=float(r1_out["gsq"])`` the running mean stays unobserved
    and its keys stay absent.  That file belongs to another work package; this
    module provides the facility and does not pretend it is already in use.
    """

    __slots__ = ("dupdate_total", "r1_fired_total", "_every_n",
                 "_gsq_mean", "_gsq_n", "_gsq_last")

    def __init__(self, r1_every_n: int = PIX_R1_EVERY_N) -> None:
        if int(r1_every_n) != 1:
            # Not forbidden outright (a deliberate ablation may want it), but
            # it is off-spec and the caller should have to mean it.
            pass
        self.dupdate_total: int = 0
        self.r1_fired_total: int = 0
        self._every_n: int = max(1, int(r1_every_n))
        # Running mean of the RAW pre-gamma ``gsq``.  ``_gsq_n == 0`` means
        # "never observed" and is what keeps the keys absent rather than 0.0.
        self._gsq_mean: float = 0.0
        self._gsq_n: int = 0
        self._gsq_last: Optional[float] = None

    @property
    def r1_every_n(self) -> int:
        return self._every_n

    def should_fire_r1(self) -> bool:
        """Whether R1 fires on the update that is *about* to happen.

        With the spec value ``pix_r1_every_n = 1`` this is unconditionally
        ``True``.  Cadence is decided here and nowhere else — §5.4's lesson is
        that *a penalty implemented twice will fire at two rates, and nobody
        notices without a monotone counter*.
        """
        return (self.dupdate_total % self._every_n) == 0

    def note_d_update(
        self, r1_fired: bool, gsq: Optional[float] = None,
    ) -> None:
        """Call exactly once per D optimizer update, with what actually happened.

        ``gsq``: the step's RAW pre-gamma ``r1_penalty(...)["gsq"]``.  Optional
        only because the existing call sites predate it; pass it whenever R1
        actually fired, so ``r1_grad_sq_mean`` becomes usable for the §5.3
        gamma calibration (one row is 126 %/40 % noise — see the class
        docstring).  A non-finite value is REFUSED rather than folded in: one
        NaN would poison the running mean permanently and silently.
        """
        self.dupdate_total += 1
        if r1_fired:
            self.r1_fired_total += 1
        if gsq is None:
            return
        v = float(gsq)
        if not math.isfinite(v):
            raise ValueError(
                f"PixCounters.note_d_update got a non-finite gsq ({gsq!r}); "
                "folding it into the running mean would destroy every "
                "subsequent reading"
            )
        self._gsq_last = v
        self._gsq_n += 1
        # Incremental mean: exact-ish over long runs, no growing sum.
        self._gsq_mean += (v - self._gsq_mean) / float(self._gsq_n)

    @property
    def r1_rate(self) -> float:
        return float(self.r1_fired_total) / float(max(1, self.dupdate_total))

    @property
    def r1_grad_sq_samples(self) -> int:
        """How many ``gsq`` values the running mean is built from."""
        return int(self._gsq_n)

    @property
    def r1_grad_sq_mean(self) -> Optional[float]:
        """Running mean of the raw pre-gamma ``gsq``, or ``None`` if never
        observed.  ``None`` — not 0.0: see the class docstring."""
        return float(self._gsq_mean) if self._gsq_n else None

    @property
    def r1_grad_sq_last(self) -> Optional[float]:
        """The most recent raw per-step ``gsq``, or ``None``.  Kept alongside
        the mean: the running mean is what ``gamma`` is calibrated from, the
        raw value is what shows a step-to-step blow-up."""
        return self._gsq_last

    def log_dict(self, prefix: str = "pix_") -> Dict[str, float]:
        """The §7 counter row.  Monotone totals + the derived rate, plus the
        ``gsq`` running mean **only once it has been observed**."""
        out: Dict[str, float] = {
            f"{prefix}dupdate_total": float(self.dupdate_total),
            f"{prefix}r1_fired_total": float(self.r1_fired_total),
            f"{prefix}r1_rate": self.r1_rate,
        }
        # OMITTED, never defaulted.  A 0.0 here would read as "R1 sees no
        # gradient on the reals" -- a real alarm -- rather than "nobody has
        # handed us a gsq yet".
        if self._gsq_n:
            out[f"{prefix}r1_grad_sq_mean"] = float(self._gsq_mean)
            out[f"{prefix}r1_grad_sq_n"] = float(self._gsq_n)
            if self._gsq_last is not None:
                out[f"{prefix}r1_grad_sq_last"] = float(self._gsq_last)
        return out

    def healthy(self, tol: float = 0.99) -> bool:
        """``pix_r1_rate < 0.99`` is a build bug (§5.4).  Readable on row one."""
        return self.dupdate_total == 0 or self.r1_rate >= float(tol)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"PixCounters(dupdate_total={self.dupdate_total}, "
            f"r1_fired_total={self.r1_fired_total}, r1_rate={self.r1_rate:.4f}, "
            f"r1_grad_sq_mean={self.r1_grad_sq_mean}, n={self._gsq_n})"
        )


# ---------------------------------------------------------------------------
# §7 — patch-logit telemetry
# ---------------------------------------------------------------------------
def patch_logit_spatial_variance(logits: torch.Tensor) -> float:
    """Mean over images of the variance of the patch logits **within an image**.

    §7: "patch-logit spatial variance (is the critic using locality?)".  A
    critic that has collapsed to a global verdict emits a flat map and this
    reads ~0; a critic using locality varies across the grid.  Reported
    separately from the across-image variance, which is a different question.

    Accepts ``[N, 1, h, w]`` (a 2-D patch grid) and ``[N, *, L]`` (a 1-D TOKEN
    map — the realistic latent-critic case, e.g. ``[N, 1, L]``).  The
    statistic is the variance over the per-image positions, which is well
    defined for both; the only thing a token map lacks is the ``h``/``w``
    split, and that is reported by :func:`patch_logit_telemetry`, not here.

    Still REFUSES ``[N]`` and ``[N, 1]``: those carry one position per image,
    so a "variance across positions" does not exist, and returning 0.0 would
    forge the flat-map signature this statistic exists to detect.
    """
    t = logits.detach().float()
    if t.dim() not in (3, 4):
        raise ValueError(
            f"expected [N,1,h,w] or a token map [N,*,L]; got {tuple(t.shape)}"
        )
    flat = t.reshape(t.shape[0], -1)
    if flat.shape[1] < 2:
        # OMIT-NEVER-FAKE.  This used to ``return 0.0``.  0.0 is precisely the
        # COLLAPSED-CRITIC signature this statistic exists to detect, so
        # synthesising it from a shape that has only one position per image
        # certifies the alarm without measuring it.  One position has no
        # within-image variance; say so.
        raise ValueError(
            "patch_logit_spatial_variance needs >= 2 positions per image; "
            f"got {tuple(t.shape)} (1 position). 0.0 is NOT a safe stand-in: "
            "it is the flat-map/collapsed-critic reading."
        )
    return float(flat.var(dim=1, unbiased=True).mean())


def patch_logit_telemetry(
    logits: torch.Tensor, prefix: str = "pix_",
) -> Dict[str, float]:
    """§7 patch-map readout, including the **actual measured P**.

    ``P`` is logged because the doc's 768 is wrong for the real crop (the 8-px
    border trim makes it 660) and because a silently changed crop size would
    otherwise only show up in a diff.

    ACCEPTED SHAPES.  **Correction to what this docstring used to claim:**
    it said the accepted set was "deliberately the same set as
    ``disc_holdout_probe._reduce_scores``".  That was false in both
    directions.  ``_reduce_scores`` has NO dim restriction at all — it is
    ``t if t.dim() == 1 else t.reshape(N, -1).mean(1)``, so it happily reduces
    any rank — while this function used to reject everything outside
    ``dim in (1, 2, 4)``.  The gap that mattered: a **3-D token map**, which
    is the realistic latent case and the whole reason the relaxation exists;
    at ``[6, 1, 7]`` the probe returned ``(6,)`` and this raised.  3-D is now
    accepted.  The relationship, stated exactly:

    * ``[N]``, ``[N, 1]`` — one score per sample.  Emits only the statistics
      that are *well defined* without positions.
    * ``[N, *, L]`` — a 1-D TOKEN map (latent critics).  Emits
      ``{prefix}patch_spatial_var`` and ``{prefix}patch_count``; the grid
      dimensions are OMITTED, because a token map has no ``h``/``w`` split and
      inventing one would forge geometry.
    * ``[N, 1, h, w]`` — a 2-D patch grid.  Full readout, grid dimensions
      included.
    * anything else (rank 0, rank >= 5) still RAISES.  ``_reduce_scores``
      would silently flatten it; here the trailing axes have no agreed
      meaning, so a caller handing this a rank-5 tensor has a bug, and
      guessing at ``patch_count`` for it would be a fabricated number.  This
      is a deliberate, stated divergence — not a parity claim.

    **A grid-less input OMITS ``{prefix}patch_spatial_var`` rather than
    reporting 0.0.**  That key answers §7's "is the critic using locality?",
    and a flat ~0 is a real and damaging diagnostic — it is the signature of a
    critic collapsed to a global verdict.  Synthesising a 0.0 from a *shape*
    would make that diagnostic forgeable, so the key is absent instead, and a
    consumer that needs it must notice its absence.
    The same omit-never-fake rule governs the token-map case:
    ``patch_spatial_var`` IS computed there (it is well defined over the token
    axis), and only ``patch_grid_h``/``patch_grid_w`` are dropped.

    ``patch_logit_spatial_variance`` itself stays strict and still raises on
    ``[N]`` / ``[N, 1]``: it is a positional statistic, and asking for it on
    per-sample scalars is a caller bug.

    This tolerance exists so the §8.1 harness can serve LATENT critics too
    (a pixel critic emits ``[N,1,h,w]``; a token-map latent critic emits
    ``[N,1,L]``; the same critic with ``ladd_scalar_output`` set emits
    ``[N]``).  The ``[N,1,h,w]`` path is byte-identical to before this was
    relaxed — same keys, same order, same values — and a regression test pins
    that.
    """
    t = logits.detach().float()
    if t.dim() not in (1, 2, 3, 4):
        raise ValueError(
            f"patch_logit_telemetry expects [N], [N,1], a token map [N,*,L] "
            f"or a patch grid [N,1,h,w]; got {tuple(t.shape)}"
        )
    has_positions = t.dim() in (3, 4)
    has_grid = t.dim() == 4
    flat = t.reshape(t.shape[0], -1)
    # Key insertion order is load-bearing for the byte-identical guarantee on
    # the 4-D path; do not reorder.
    out: Dict[str, float] = {
        f"{prefix}patch_mean": float(flat.mean()),
        f"{prefix}patch_std": float(flat.std()),
    }
    if has_positions and flat.shape[1] >= 2:
        # Guarded rather than try/except so the omission is visibly the same
        # rule as the grid-less case above: one position per image => the
        # statistic does not exist => the key is ABSENT, never 0.0.
        out[f"{prefix}patch_spatial_var"] = patch_logit_spatial_variance(t)
    out[f"{prefix}patch_count"] = float(flat.shape[1])     # the real P
    if has_grid:
        out[f"{prefix}patch_grid_h"] = float(t.shape[-2])
        out[f"{prefix}patch_grid_w"] = float(t.shape[-1])
    return out


def _rf_probe(
    input_size_h: int,
    input_size_w: int,
    *,
    use_norm: bool,
    in_channels: int,
    base_channels: int,
    device: torch.device,
) -> Dict[str, Any]:
    """One gradient probe: bounding box + off-patch gradient share.

    Backprops a single CENTRE patch logit to the input and reports (a) the
    bounding box of the non-zero input gradient and (b) how much of the
    gradient falls outside a :data:`CONV_GEOMETRIC_RF`-sized box around that
    patch — in BOTH L1 (``off_patch_grad_frac``) and squared energy
    (``off_patch_grad_energy_frac``).

    Both are reported because they answer different questions and, on the
    normed variant, they disagree by two orders of magnitude: 4-9 % of the L1
    mass is off-patch but only ~0.01-0.03 % of the energy, i.e. the coupling
    was broad and weak rather than concentrated.  Quoting only the energy
    number would have made a whole-image critic look local.
    """
    net = PixelTextureDisc(
        in_channels=in_channels, base_channels=base_channels,
        use_spectral_norm=False, use_norm=use_norm,
    ).to(device).eval()
    # Constant non-zero input keeps LeakyReLU on a single linear branch.
    x = torch.full((1, in_channels, input_size_h, input_size_w), 0.5,
                   device=device, dtype=torch.float32, requires_grad=True)
    out = net(x)
    gh, gw = int(out.shape[-2]), int(out.shape[-1])
    out[0, 0, gh // 2, gw // 2].backward()
    raw = x.grad.detach()
    g = raw.abs().sum(dim=(0, 1))                      # [H, W]  L1
    e = (raw ** 2).sum(dim=(0, 1))                     # [H, W]  squared energy
    nz = (g > 0).nonzero()
    rf_h = int(nz[:, 0].max() - nz[:, 0].min() + 1)
    rf_w = int(nz[:, 1].max() - nz[:, 1].min() + 1)
    # Share of the gradient OUTSIDE a CONV_GEOMETRIC_RF-sized box centred on
    # the probed patch.  This is the number that says whether the critic's
    # answer is a local texture question or a whole-image one.
    cy = (gh // 2) * 8 + 4
    cx = (gw // 2) * 8 + 4
    b = CONV_GEOMETRIC_RF // 2
    y0, y1 = max(0, cy - b), min(input_size_h, cy + b + 1)
    x0, x1 = max(0, cx - b), min(input_size_w, cx + b + 1)

    def _off(m: torch.Tensor) -> float:
        tot = float(m.sum())
        if tot <= 0.0:
            # OMIT-NEVER-FAKE would say drop the key -- but a zero TOTAL
            # gradient is not "uncomputed", it is a dead probe, and 0.0
            # off-patch is exactly the flattering reading.  Refuse instead.
            raise ValueError(
                "receptive-field probe measured an all-zero input gradient; "
                "off_patch_grad_frac has no value here and 0.0 would forge "
                "the perfectly-local answer"
            )
        return 1.0 - float(m[y0:y1, x0:x1].sum()) / tot

    return {
        "rf_h": rf_h, "rf_w": rf_w, "grid_h": gh, "grid_w": gw,
        "off_patch_grad_frac": _off(g),
        "off_patch_grad_energy_frac": _off(e),
    }


#: Conv-stack geometric receptive field, closed form back-to-front:
#: k3s1 -> 3 ; k4s2 -> 8 ; k4s2 -> 18 ; k4s2 -> 38.  This is the number §4
#: should have said instead of "~70 px" (the pix2pix 70x70 PatchGAN has FIVE
#: layers; §4's table has four).  Since the 2026-08-23 authorised GroupNorm
#: removal this is ALSO the EFFECTIVE receptive field of the shipped critic --
#: measured, not assumed: the off-38x38 share of a centre patch logit's input
#: gradient is < 2.4e-07 in L1 and < 1.4e-07 in squared energy.  With the norm
#: it was the whole image.  See :func:`measure_receptive_field` and
#: :func:`effective_sample_count`.
CONV_GEOMETRIC_RF: int = 38


def measure_receptive_field(
    input_size: int = 129,
    *,
    input_size_w: Optional[int] = None,
    in_channels: int = 3,
    base_channels: int = 64,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Measure the receptive field **of the configuration that trains** — and,
    for comparison, of the normed variant that no longer ships.

    Gradient probe: feed a constant input, backprop a single centre patch
    logit, and look at the input gradient.

    THE PRIMARY NUMBER IS THE DEFAULT (norm-free) ONE, AND IT IS LOCAL.
    ------------------------------------------------------------------
    ``PixelTextureDisc`` now ships ``use_norm=False`` — an authorised
    deviation from §4's layer table, taken by the researcher on 2026-08-23
    after the normed critic was measured to be global (see the class
    docstring).  Re-measured after the change, at 176x240 (the real post-trim
    crop), 12 random inits:

    =============  ===============  ======================  ==================
    config         grad bbox        off-38x38 |grad| (L1)   off-38x38 grad^2
    =============  ===============  ======================  ==================
    DEFAULT (off)  38 x 38          < 2.4e-07 (float noise) < 1.4e-07
    normed (on)    176 x 240 (all)  0.0399 .. 0.0886        5.07e-05..2.90e-04
    =============  ===============  ======================  ==================

    The globality is gone: the support is exactly the 38 px conv geometry, and
    the off-patch share is float noise in both L1 and energy — the residual is
    ~5 orders of magnitude below the normed variant's L1 figure and sits at
    the level of float32 summation error (it is not even sign-definite; over
    12 inits it ranged -1.2e-07 .. +2.4e-07, which is what a "zero measured by
    subtracting two nearly-equal sums" looks like).  Confirmed unchanged with
    spectral norm ON (the shipped setting) and on random rather than constant
    inputs: both give a 38x38 bbox and exactly 0.0 off-patch.

    §4 says "Receptive field ~= 70 px — a local texture question".  **The
    "local texture question" half is now TRUE of what trains; the "~70 px"
    half never was.**  The number is 38: the pix2pix 70x70 PatchGAN has FIVE
    layers (k4s2 x3 then k4s1 x2), §4's table has four.  Closed form
    back-to-front: k3s1 -> 3 ; k4s2 -> 8 ; k4s2 -> 18 ; k4s2 -> 38.

    ``rf_h``/``rf_w``/``off_patch_grad_frac``/``off_patch_grad_energy_frac``
    describe the DEFAULT critic.  ``conv_rf_*`` are the same probe named for
    the conv geometry (identical now that the default carries no norm, and
    kept because callers and the §3.7 arithmetic refer to the geometry as
    such).  ``normed_*`` are the OLD shipped variant, retained so the locality
    guard has a planted violation that genuinely fails and so this correction
    cannot be quietly reverted without a test noticing.

    Returns ``{"rf_h", "rf_w", "off_patch_grad_frac",
    "off_patch_grad_energy_frac", "conv_rf_h", "conv_rf_w",
    "conv_off_patch_grad_frac", "conv_off_patch_grad_energy_frac",
    "normed_rf_h", "normed_rf_w", "normed_off_patch_grad_frac",
    "normed_off_patch_grad_energy_frac", "norm_enabled", "stride", "grid_h",
    "grid_w", "input_h", "input_w"}``.
    """
    dev = device or torch.device("cpu")
    h = int(input_size)
    w = int(input_size_w) if input_size_w is not None else h
    kw = dict(in_channels=in_channels, base_channels=base_channels,
              device=dev)
    # The DEFAULT critic is the one that trains -- norm-free, by the 2026-08-23
    # authorised deviation.  Probed with the class default rather than a
    # hard-coded ``use_norm=``, so that if the default were ever flipped back
    # this report would follow it and the locality test would fail loudly
    # instead of quietly measuring a network nobody trains.
    default_norm = bool(PixelTextureDisc().uses_norm)
    train_cfg = _rf_probe(h, w, use_norm=default_norm, **kw)   # what SHIPS
    normed_cfg = _rf_probe(h, w, use_norm=True, **kw)          # old variant
    return {
        # Primary = the configuration that trains (norm-free by default).
        "rf_h": train_cfg["rf_h"],
        "rf_w": train_cfg["rf_w"],
        "off_patch_grad_frac": train_cfg["off_patch_grad_frac"],
        "off_patch_grad_energy_frac": train_cfg["off_patch_grad_energy_frac"],
        "norm_enabled": default_norm,
        # The conv geometry, which the default now equals.
        "conv_rf_h": train_cfg["rf_h"],
        "conv_rf_w": train_cfg["rf_w"],
        "conv_off_patch_grad_frac": train_cfg["off_patch_grad_frac"],
        "conv_off_patch_grad_energy_frac":
            train_cfg["off_patch_grad_energy_frac"],
        # The OLD normed variant, kept for comparison and for the
        # planted-violation companion to the locality guard.
        "normed_rf_h": normed_cfg["rf_h"],
        "normed_rf_w": normed_cfg["rf_w"],
        "normed_off_patch_grad_frac": normed_cfg["off_patch_grad_frac"],
        "normed_off_patch_grad_energy_frac":
            normed_cfg["off_patch_grad_energy_frac"],
        "stride": 8,
        "grid_h": train_cfg["grid_h"],
        "grid_w": train_cfg["grid_w"],
        "input_h": h,
        "input_w": w,
    }


# ---------------------------------------------------------------------------
# §3.7 — effective sample count, RECOMPUTED for the now-local critic
# ---------------------------------------------------------------------------
def effective_sample_count(
    input_h: int = 176,
    input_w: int = 240,
    *,
    rf: int = 38,
    stride: int = 8,
    crops_per_step: int = PIX_CROPS_PER_STEP,
    frames_per_crop: int = PIX_FRAMES_PER_CROP,
) -> Dict[str, Any]:
    """§3.7's "effective sample count", recomputed on the MEASURED 38 px RF.

    §3.7 says sample count is bought with crops and frames, not with patch
    positions, and tabulates::

        patch logits per image                     768
        non-overlapping receptive-field tiles      6
        overlap factor                             ~128 : 1

    **Both inputs to that arithmetic were wrong**, and they were wrong in
    opposite directions:

    * ``768`` is the UNTRIMMED ``192/8 x 256/8``.  The §3.2 8-px border trim
      makes the real crop 176x240, so ``P = 22 x 30 = 660`` (this has been
      measured since the module was written; ``r1_penalty`` returns the real
      ``P`` and never a constant).
    * ``6`` tiles follows from §4's claimed ``~70 px`` receptive field:
      ``floor(176/70) * floor(240/70) = 2 * 3 = 6``.  The receptive field is
      **38 px** (:data:`CONV_GEOMETRIC_RF`; the pix2pix 70x70 PatchGAN has
      five layers, §4's table has four) — and, until the GroupNorm was removed
      (see :class:`PixelTextureDisc`), the *effective* support was the whole
      image, which makes the tile count 1 and the whole table meaningless.

    With the norm gone the locality premise finally holds, so the arithmetic
    is worth doing properly.  At 176x240 with a 38 px RF::

        non-overlapping tiles  = floor(176/38) * floor(240/38) = 4 * 6 = 24
        area-ratio tiles       = (176*240) / (38*38)           = 29.2
        overlap factor         = 660 / 24                      = 27.5 : 1
        patches sharing a pixel= (38/8)^2                      = 22.6

    i.e. **24 tiles per image, not 6, and ~27.5:1 overlap, not ~128:1** — a
    4x better independent-sample yield per image than §3.7 assumed.  The
    corrected fake-side table (``images/step = crops x frames``, effective =
    ``images x tiles``):

    ==========================  ======  ============  ==========  ==========
    config                      images  patch logits  §3.7 says   MEASURED
    ==========================  ======  ============  ==========  ==========
    draft (crops=2, frames=2)       4          2 640        ~24          96
    SPEC  (crops=4, frames=3)      12          7 920        ~72         288
    raised(crops=8, frames=3)      24         15 840       ~144         576
    ==========================  ======  ============  ==========  ==========

    **What this means for the launch config.**  The specced
    ``pix_crops_per_step=4, pix_frames_per_crop=3`` already delivers 288
    effective samples — **4x more than §3.7 credited it with, and 2x more than
    §3.7's own ``crops=8`` escalation target of ~144.**  The §3.7 trigger
    "raise to 8 if the critic looks sample-starved" therefore has no
    arithmetic support at launch; if it is ever pulled it must be pulled on the
    §7 patch-logit variance or the §8.1 control saying so, not on this table.

    **The honest caveat, which the headline number does NOT include.**  A
    "tile" here is an independent *receptive field*, not an independent *draw
    from the real distribution*.  §3.4a is explicit that three frames from one
    crop share exposure, weather, time of day, road surface and the same VAE
    reconstruction, and that frame-within-crop expansion is a fake-side device
    only.  Tiles within one crop are worse still — same scene, same instant.
    So ``effective_per_step`` counts *conditionally* independent texture
    samples; the count of independent SCENES per step is
    ``crops_per_step`` (4), and that number is unchanged by this correction.
    ``effective_source_independent`` reports ``crops x tiles`` as the
    intermediate figure that drops the frame axis, since that axis is the one
    §3.4a explicitly refuses to count on the real side.

    Every value is computed, none defaulted.
    """
    gh, gw = PixelTextureDisc.patch_grid(int(input_h), int(input_w))
    p = int(gh * gw)
    r = int(rf)
    tiles_h = int(input_h) // r
    tiles_w = int(input_w) // r
    tiles = int(tiles_h * tiles_w)
    images = int(crops_per_step) * int(frames_per_crop)
    return {
        "input_h": int(input_h),
        "input_w": int(input_w),
        "rf_px": r,
        "stride": int(stride),
        "patch_logits_per_image": p,
        "tiles_h": tiles_h,
        "tiles_w": tiles_w,
        "tiles_per_image": tiles,
        "tiles_per_image_area_ratio": float(input_h * input_w) / float(r * r),
        "overlap_factor": float(p) / float(tiles),
        "patches_sharing_a_pixel": (float(r) / float(stride)) ** 2,
        "crops_per_step": int(crops_per_step),
        "frames_per_crop": int(frames_per_crop),
        "images_per_step": images,
        "patch_logits_per_step": images * p,
        "effective_per_step": images * tiles,
        "effective_source_independent": int(crops_per_step) * tiles,
        "independent_scenes_per_step": int(crops_per_step),
        # What §3.7 claimed, kept alongside so the correction is legible.
        "doc_patch_logits_per_image": 768,
        "doc_tiles_per_image": 6,
        "doc_overlap_factor": 128.0,
        "doc_effective_per_step": images * 6,
    }


# ---------------------------------------------------------------------------
# Synthetic stability smoke for the norm-free critic.  NOT evidence about the
# real arm -- see the docstring.
# ---------------------------------------------------------------------------
def _stability_canvas(
    n: int, h: int, w: int, generator: torch.Generator,
) -> torch.Tensor:
    """Smooth low-frequency base + broadband grain, in ``[-1, 1]``."""
    yy = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1)
    xx = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w)
    ph = torch.rand(n, 1, 1, 1, generator=generator) * 6.283
    base = (0.6 * torch.sin(3.0 * yy + ph) + 0.6 * torch.cos(4.0 * xx + ph))
    base = base.expand(n, 3, h, w).clone()
    grain = 0.15 * torch.randn((n, 3, h, w), generator=generator)
    return (base + grain).clamp(-1.0, 1.0)


def synthetic_stability_smoke(
    steps: int = 300,
    *,
    use_norm: bool = False,
    batch: int = 4,
    input_h: int = 88,
    input_w: int = 120,
    lr: float = PIX_GAN_LR,
    gamma: float = PIX_R1_GAMMA,
    sigma: float = PIX_R1_SIGMA,
    c1_amplitude: float = 0.5,
    loss_form: str = PIX_LOSS_FORM,
    seed: int = 0,
    window: int = 50,
) -> Dict[str, Any]:
    """A **SYNTHETIC** D-only training smoke.  Read the caveat before citing it.

    **WHAT THIS IS NOT.**  It is not evidence that the norm-free critic trains
    stably in the real arm, and it must never be reported as such.  There is no
    generator, no VAE decode, no real dashcam data, no DDP, no bf16, and the
    "fake" side is a fixed analytic corruption rather than a student that
    fights back.  A GAN's instabilities are overwhelmingly *joint* G/D
    dynamics, and this loop has no G at all.

    **WHAT IT IS.**  The recorded risk of removing the GroupNorm
    (:class:`PixelTextureDisc`) is that an unnormalised from-scratch PatchGAN
    conditions worse.  If that risk were severe and unconditional it would show
    up even here — as a ``d_loss`` that will not descend, logits or activations
    that run away, or non-finite values.  So this is a cheap *falsifier*: it can
    only ever say "not obviously broken", never "stable".

    Setup: reals are a fresh smooth+grain canvas each step; fakes are the same
    canvas carrying the §8.1 :func:`c1_structured_hf` anisotropic HF
    corruption — the measured failure direction, so the discrimination task is
    the one B2 cares about rather than an arbitrary one.  D is trained with
    :func:`d_loss` + :func:`r1_penalty` under the §5.2 optimiser
    (Adam, betas ``(0.0, 0.9)``).

    Returns per-window ``d_loss`` means (first/last), the max absolute patch
    logit and per-block activation maxima over the run, the max gradient norm,
    and ``finite`` — ``False`` if any non-finite value was ever seen.
    """
    gen = torch.Generator().manual_seed(int(seed))
    torch.manual_seed(int(seed))
    net = PixelTextureDisc(use_norm=bool(use_norm))
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=float(lr), betas=PIX_GAN_BETAS)

    acts: Dict[str, float] = {}

    def _record(name: str, t: torch.Tensor) -> None:
        v = float(t.detach().abs().max())
        acts[name] = max(acts.get(name, 0.0), v)

    losses: List[float] = []
    max_logit = 0.0
    max_gnorm = 0.0
    finite = True
    for _ in range(int(steps)):
        real = _stability_canvas(int(batch), int(input_h), int(input_w), gen)
        fake = c1_structured_hf(real, float(c1_amplitude), generator=gen)

        s = net.negative_slope
        h1 = F.leaky_relu(net.conv1(real), s)
        h2 = F.leaky_relu(net.norm2(net.conv2(h1)), s)
        h3 = F.leaky_relu(net.norm3(net.conv3(h2)), s)
        r_log = net.conv4(h3)
        _record("block1", h1)
        _record("block2", h2)
        _record("block3", h3)
        f_log = net(fake)

        out = d_loss(r_log, f_log, loss_form)
        r1 = r1_penalty(net, real, gamma=float(gamma), sigma=float(sigma),
                        generator=gen)
        total = out["d_loss"] + r1["r1"]

        opt.zero_grad(set_to_none=True)
        total.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(
            net.parameters(), float("inf")))
        opt.step()

        lv = float(out["d_loss"].detach())
        losses.append(lv)
        max_logit = max(max_logit, float(r_log.detach().abs().max()),
                        float(f_log.detach().abs().max()))
        max_gnorm = max(max_gnorm, gn)
        if not (math.isfinite(lv) and math.isfinite(gn)
                and math.isfinite(max_logit)):
            finite = False
            break

    w = max(1, min(int(window), len(losses)))
    first = sum(losses[:w]) / w
    last = sum(losses[-w:]) / w
    return {
        "use_norm": bool(use_norm),
        "steps": len(losses),
        "lr": float(lr),
        "d_loss_first_window": first,
        "d_loss_last_window": last,
        "d_loss_delta": last - first,
        "descends": bool(last < first),
        "d_loss_min": min(losses) if losses else float("nan"),
        "d_loss_max": max(losses) if losses else float("nan"),
        "max_abs_patch_logit": max_logit,
        "max_grad_norm": max_gnorm,
        "max_abs_activation": dict(acts),
        "finite": bool(finite),
        "window": w,
        "SYNTHETIC": True,
    }


# ---------------------------------------------------------------------------
# §8.1 — the C1 positive control (structured-HF corruption)
# ---------------------------------------------------------------------------
#: §0 / §9 reference batteries, from ``eval/texture_abc_strict03/REPORT.md``.
#: These are the numbers the C1 amplitude is calibrated *between*.
B_BATTERY_REF: Dict[str, Any] = {
    "hf_power": (0.0159, 0.0165),      # 0.86-0.89x A
    "hv_anisotropy": (1.07, 1.15),
    "angular_entropy": 0.98,
}
C_LATE_BATTERY_REF: Dict[str, float] = {
    "hf_power": 0.0407,                # 2.21x A
    "hv_anisotropy": 0.363,
    "angular_entropy": 0.843,
}


def _randn_like_with_generator(
    x: torch.Tensor, generator: Optional[torch.Generator],
) -> torch.Tensor:
    """``randn`` from an EXPLICIT generator, tolerating a device mismatch.

    Never falls back to global RNG when a generator is supplied — under DDP
    that would let ranks disagree on the corruption.
    """
    if generator is None:
        return torch.randn(x.shape, device=x.device, dtype=torch.float32)
    if generator.device.type == x.device.type:
        return torch.randn(x.shape, generator=generator, device=x.device,
                           dtype=torch.float32)
    return torch.randn(x.shape, generator=generator, device=generator.device,
                       dtype=torch.float32).to(x.device)


def c1_structured_hf(
    x: torch.Tensor,
    amplitude: float,
    *,
    generator: Optional[torch.Generator] = None,
    hf_cut: float = 0.55,
    wedge_half_width: float = math.pi / 10.0,
    orientation: str = "column",
    scale_ref: Optional[float] = None,
) -> torch.Tensor:
    """§8.1 **C1** — anisotropic structured-HF corruption, self-generated.

    Added to a GT **latent** *before* decode (it is shape-agnostic, so the unit
    tests can also apply it directly to pixels).  The perturbation is
    **row-periodic**: near-Nyquist energy along the *width* axis, near-constant
    along height, i.e. fine vertical striping.  That is the measured failure
    direction — ``texture_stats.hv_anisotropy`` is ``p_fy / p_fx``, so putting
    energy on the ``fx`` axis drives it **DOWN**, from the A/B value
    (~1.07-1.15) toward the measured ``C_late`` **0.363**, while concentrating
    the orientation histogram drives ``angular_entropy`` **DOWN** from ~0.98
    toward **0.843**.  Both directions are asserted in the unit tests against
    ``analysis/texture_stats.texture_battery`` — no statistic is reimplemented
    here.

    It is *stochastic*, not a pure sinusoid: white noise is band-pass filtered
    to a high-radius orientation wedge.  A single-frequency stripe would drive
    ``angular_entropy`` far past 0.843 in one step and would be trivially
    detectable by a critic for the wrong reason.

    Properties the design requires and the tests check:

    * **self-generated** — nothing on disk, works at any step;
    * **amplitude-parameterised** and monotone in ``amplitude``;
    * **DDP-safe** — an explicit ``torch.Generator``; global RNG is used only
      when ``generator is None``, which the trainer must never do;
    * **deterministic** under a fixed generator state.

    Args:
        x: ``[..., C, H, W]`` GT latent (or pixels).
        amplitude: dimensionless; the perturbation is scaled to
            ``amplitude * scale_ref`` where ``scale_ref`` defaults to
            ``x.std()``, so one calibrated amplitude transfers across
            normalisations.  Calibrate ONCE offline — see
            :func:`sweep_c1_amplitude`.
        hf_cut: fraction of Nyquist below which energy is removed.  0.55 keeps
            the perturbation genuinely high-frequency (the battery's own
            ``hf_cut`` default is 0.25, so this sits well inside its HF band).
        wedge_half_width: orientation half-width, radians.
        orientation: ``"column"`` (default) = energy on the ``fx`` axis =
            vertical stripes = ``hv_anisotropy`` DOWN, the measured direction.
            ``"row"`` is the mirror image and exists only so the test can show
            the statistic is genuinely two-sided.
    """
    if x.dim() < 3:
        raise ValueError(f"c1_structured_hf expects [...,C,H,W]; got {tuple(x.shape)}")
    a = float(amplitude)
    if a == 0.0:
        return x.clone()

    orig_shape = x.shape
    z = x.reshape(-1, *orig_shape[-3:]).float()          # [N, C, H, W]
    _n, _c, h, w = z.shape

    noise = _randn_like_with_generator(z, generator)
    spec = torch.fft.rfft2(noise, norm="ortho")

    fy = torch.fft.fftfreq(h, device=z.device)[:, None]
    fx = torch.fft.rfftfreq(w, device=z.device)[None, :]
    r = torch.sqrt(fy ** 2 + fx ** 2)
    # Same wedge convention as texture_stats.directional_spectrum: theta is
    # atan2(|fy|, fx) in [0, pi/2]; theta ~ 0 is energy at horizontal
    # frequency = VERTICAL stripes; theta ~ pi/2 is horizontal stripes.
    theta = torch.atan2(fy.abs().expand(h, fx.shape[1]),
                        fx.expand(h, fx.shape[1]) + 1e-12)
    if str(orientation).lower() == "column":
        wedge = (theta < float(wedge_half_width))
    elif str(orientation).lower() == "row":
        wedge = (theta > (math.pi / 2 - float(wedge_half_width)))
    else:
        raise ValueError("orientation must be 'column' or 'row'")
    mask = (wedge & (r >= float(hf_cut) * 0.5) & (r <= 0.5)).to(spec.dtype)

    field = torch.fft.irfft2(spec * mask[None, None], s=(h, w), norm="ortho")
    sd = field.std()
    if float(sd) <= 0.0:
        return x.clone()
    field = field / sd

    ref = float(x.float().std()) if scale_ref is None else float(scale_ref)
    out = z + (a * ref) * field
    return out.reshape(orig_shape).to(x.dtype)


def sweep_c1_amplitude(
    x: torch.Tensor,
    amplitudes: Sequence[float],
    *,
    seed: int = 0,
    decode_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    hf_cut: float = 0.25,
    targets: Optional[Dict[str, float]] = None,
    target_frac: float = 0.5,
    c1_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """§8.1 **calibration entry point** — sweep the C1 amplitude, score, verdict.

    Run this **ONCE, offline**, before the arm launches.  It scores every
    amplitude with ``analysis/texture_stats.texture_battery`` (imported — no
    statistic is reimplemented here) and reports, per amplitude, how far the
    corrupted battery has travelled from the clean reference toward
    ``C_late``.  Pick the smallest amplitude whose verdict is ``between``:
    the control must be a **milder** defect than the student's, so that
    "separates the corruption but not the student" is a meaningful ordering
    (§8.1's rank comparison — ``AUC(GT vs corrupted)`` should *exceed*
    ``AUC(GT vs student)`` throughout; if that inverts, the calibration is
    wrong, not the critic).

    Args:
        x: GT **latents** ``[..., C, H, W]``, exactly as the training real
            supply provides them.
        amplitudes: the sweep grid.
        seed: fixes the generator so the sweep is reproducible.
        decode_fn: the frozen VAE decoder, latents -> pixels in ``[-1,1]``.
            **Pass it.**  The battery in §0/§9 was measured on decoded pixels,
            so a latent-domain battery is a different quantity and its absolute
            values are not comparable to ``B_BATTERY_REF`` / ``C_LATE_BATTERY_REF``.
            Omitting it scores the latent directly, which is only useful for
            unit tests and for a quick monotonicity check.
        c1_kwargs: forwarded to :func:`c1_structured_hf` (e.g.
            ``{"hf_cut": 0.55}``).  Passed as an explicit dict rather than
            ``**kwargs`` because BOTH functions have an ``hf_cut`` and they
            mean different things: this function's ``hf_cut`` is the *battery's*
            HF band, C1's is the corruption's own band.
        targets: the ``C_late`` end of the interval; defaults to
            :data:`C_LATE_BATTERY_REF`.  The "B" end is the *measured clean
            battery of this very input*, not the doc's absolute numbers, so the
            calibration is robust to pipeline differences.

        target_frac: where in the clean->C_late interval the recommendation
            should sit.  0.5 = squarely between, which is what §8.1 asks for
            ("a *milder* defect than the student's").  NOT the smallest
            ``between`` amplitude: a corruption 2% of the way to C_late is
            technically "between" and practically inert, and an inert control
            would report UNDERTRAINED for a critic that is fine.

    Returns a dict with ``clean`` (the reference battery), ``rows`` (one per
    amplitude: battery + verdict), and ``recommended`` — the ``between``
    amplitude whose mean fraction-of-the-way is closest to ``target_frac``, or
    ``None`` if no amplitude in the grid lands between.  The recommendation is
    a suggestion; the calibration is recorded by a human, once, before launch.
    """
    tgt = dict(targets or C_LATE_BATTERY_REF)
    ckw = dict(c1_kwargs or {})

    def _battery(t: torch.Tensor) -> Dict[str, float]:
        u = decode_fn(t) if decode_fn is not None else t
        return texture_battery(u.float().reshape(-1, *u.shape[-3:]), hf_cut=hf_cut)

    clean = _battery(x)
    rows: List[Dict[str, Any]] = []
    best: Optional[Tuple[float, float]] = None   # (|mean_frac - target|, amp)
    for a in amplitudes:
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        corrupted = c1_structured_hf(x, float(a), generator=gen, **ckw)
        batt = _battery(corrupted)
        verdict = c1_calibration_verdict(clean, batt, targets=tgt)
        fr = [v for v in verdict["frac"].values() if v == v]
        mean_frac = float(sum(fr) / len(fr)) if fr else float("nan")
        verdict["mean_frac"] = mean_frac
        rows.append({"amplitude": float(a), "battery": batt, "verdict": verdict})
        if verdict["between"] and mean_frac == mean_frac:
            d = abs(mean_frac - float(target_frac))
            if best is None or d < best[0]:
                best = (d, float(a))
    return {"clean": clean, "rows": rows,
            "recommended": None if best is None else best[1],
            "target_frac": float(target_frac),
            "targets": tgt, "hf_cut": hf_cut,
            "decoded": decode_fn is not None}


def c1_calibration_verdict(
    clean: Dict[str, float],
    corrupted: Dict[str, float],
    *,
    targets: Optional[Dict[str, float]] = None,
    terms: Sequence[str] = ("hv_anisotropy", "angular_entropy"),
) -> Dict[str, Any]:
    """Is the corrupted battery **between** the clean (B-analogue) and C_late?

    Scored as a fraction of the way::

        frac = (corrupted - clean) / (target - clean)

    ``0 < frac < 1`` on every term means the corruption is a real defect in the
    measured direction but **milder** than the student's — which is exactly the
    §8.1 requirement.  ``frac >= 1`` means it has overshot ``C_late`` (too
    harsh, the ordering guarantee is lost); ``frac <= 0`` means it did not move
    in the right direction at all.

    ``hf_power`` is deliberately **not** in the default ``terms``: A11 recorded
    that ``hf_power`` anti-correlates with the eye late and never votes in the
    validated verdict rule.  It is reported in the battery regardless.
    """
    tgt = dict(targets or C_LATE_BATTERY_REF)
    fracs: Dict[str, float] = {}
    ok = True
    for k in terms:
        if k not in corrupted or k not in clean or k not in tgt:
            continue
        denom = float(tgt[k]) - float(clean[k])
        if abs(denom) < 1e-12:
            fracs[k] = float("nan")
            continue
        f = (float(corrupted[k]) - float(clean[k])) / denom
        fracs[k] = f
        if not (0.0 < f < 1.0):
            ok = False
    return {"frac": fracs, "between": bool(ok and fracs)}


# ---------------------------------------------------------------------------
# §8.1 — the readout: ROC-AUC + mean patch-logit gap, bootstrap CI over CROPS
# ---------------------------------------------------------------------------
def roc_auc_fast(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """``P(pos > neg)`` with ties at 0.5 — rank form, ``O(n log n)``.

    Numerically identical to ``model.disc_holdout_probe.roc_auc`` (which is the
    exact ``O(n1*n2)`` pairwise form, itself verified to 1.1e-16 against
    sklearn); the unit tests cross-check the two.  The fast form is needed
    because the §8.1 bootstrap resamples thousands of times over ~16k pooled
    patch logits per side, where the pairwise form is ~62M comparisons a draw.
    """
    p = pos.reshape(-1).to(torch.float64)
    q = neg.reshape(-1).to(torch.float64)
    n1, n2 = int(p.numel()), int(q.numel())
    if n1 == 0 or n2 == 0:
        return float("nan")
    allv = torch.cat([p, q])
    order = torch.argsort(allv)
    sorted_v = allv[order]
    # Average ranks for ties (1-based).
    ranks = torch.arange(1, n1 + n2 + 1, dtype=torch.float64, device=allv.device)
    i = 0
    total = n1 + n2
    while i < total:
        j = i
        while j + 1 < total and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        if j > i:
            ranks[i:j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    out = torch.empty_like(ranks)
    out[order] = ranks
    r_pos = out[:n1].sum()
    return float((r_pos - n1 * (n1 + 1) / 2.0) / (n1 * n2))


def patch_logit_separation(
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
    *,
    n_boot: int = 1000,
    ci: float = 0.95,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, float]:
    """ROC-AUC + mean patch-logit gap with a **bootstrap CI over crops**.

    ``pos_logits`` / ``neg_logits`` are patch maps ``[N, 1, h, w]`` (or
    ``[N, P]``).  The point statistics pool **all** patch logits; the bootstrap
    resamples **crops** (rows) with replacement, because patches within a crop
    share almost all of their receptive field (§3.7: ~128:1 overlap on the
    doc's numbers) and are emphatically not independent.  Bootstrapping patches
    would advertise an interval several times too narrow — the same class of
    error as A22's D4 (frame-level ``n`` where crop-level ``n`` was the truth).

    ``gap`` is ``mean(pos) - mean(neg)`` in logit units.  Chance is AUC 0.5,
    gap 0.
    """
    a = pos_logits.detach().float().reshape(int(pos_logits.shape[0]), -1)
    b = neg_logits.detach().float().reshape(int(neg_logits.shape[0]), -1)
    na, nb = int(a.shape[0]), int(b.shape[0])
    auc = roc_auc_fast(a, b)
    gap = float(a.mean() - b.mean())

    lo_q = (1.0 - float(ci)) / 2.0
    hi_q = 1.0 - lo_q
    aucs: List[float] = []
    gaps: List[float] = []
    nb_iter = max(0, int(n_boot))
    for _ in range(nb_iter):
        if generator is not None:
            ia = torch.randint(0, na, (na,), generator=generator)
            ib = torch.randint(0, nb, (nb,), generator=generator)
        else:
            ia = torch.randint(0, na, (na,))
            ib = torch.randint(0, nb, (nb,))
        sa = a.index_select(0, ia.to(a.device))
        sb = b.index_select(0, ib.to(b.device))
        aucs.append(roc_auc_fast(sa, sb))
        gaps.append(float(sa.mean() - sb.mean()))

    def _q(vals: List[float], q: float) -> float:
        if not vals:
            return float("nan")
        t = torch.tensor(vals, dtype=torch.float64)
        return float(torch.quantile(t, q))

    return {
        "auc": auc,
        "auc_lo": _q(aucs, lo_q),
        "auc_hi": _q(aucs, hi_q),
        "gap": gap,
        "gap_lo": _q(gaps, lo_q),
        "gap_hi": _q(gaps, hi_q),
        "n_pos_crops": float(na),
        "n_neg_crops": float(nb),
        "n_patches_per_crop": float(a.shape[1]),
        "n_boot": float(nb_iter),
        "ci": float(ci),
    }


@torch.no_grad()
def positive_control_readout(
    disc: Callable[[torch.Tensor], torch.Tensor],
    gt_px: torch.Tensor,
    corrupted_px: torch.Tensor,
    student_px: Optional[torch.Tensor] = None,
    *,
    n_boot: int = 1000,
    ci: float = 0.95,
    generator: Optional[torch.Generator] = None,
    prefix: str = "pix_poscontrol_",
) -> Dict[str, float]:
    """§8.1 readout — **both** pairs on the same step, `no_grad`, off the graph.

    ``disc`` may be **ANY scorer callable, pixel-space or latent-space** — the
    harness only requires ``tensor -> [N,1,h,w]`` token/patch map or ``[N]`` /
    ``[N,1]`` per-sample scores, and nothing in this function, in
    :func:`patch_logit_separation` or in :func:`roc_auc_fast` assumes pixels.
    The ``*_px`` parameter names are **historical, not a constraint**: pass
    latents and a latent critic and it works unchanged.  (That is how the
    stock-1.3B and 14B disc-backbone arms reuse this control.)

    Emits ``pix_poscontrol_*`` for ``GT vs corrupted`` (the control) and, when
    ``student_px`` is given, ``pix_arm_*`` for ``GT vs student`` (the arm).
    Also emits ``{prefix}rank_ok``: 1.0 when
    ``AUC(GT vs corrupted) >= AUC(GT vs student)``, which §8.1 requires
    *throughout* because C1 is calibrated to be the milder defect.  **If that
    ordering inverts, the calibration is wrong, not the critic.**

    Decision table (§8.1), for the reader of the log:

    ==========================  =========================  ==============
    control (GT vs corrupted)   arm (GT vs student)        verdict
    ==========================  =========================  ==============
    cannot separate             --                         UNDERTRAINED
    separates                   cannot separate            INFORMATIVE
    separates                   separates                  HEALTHY
    ==========================  =========================  ==============

    The "separates" threshold must be **pre-registered before the arm
    launches** and recorded in the run log; this function deliberately does not
    invent one.
    """
    out: Dict[str, float] = {}
    l_gt = disc(gt_px)
    l_cor = disc(corrupted_px)
    ctrl = patch_logit_separation(l_gt, l_cor, n_boot=n_boot, ci=ci,
                                  generator=generator)
    for k, v in ctrl.items():
        out[f"{prefix}{k}"] = v
    out.update(patch_logit_telemetry(l_gt, prefix=f"{prefix}gt_"))
    if student_px is not None:
        l_st = disc(student_px)
        arm = patch_logit_separation(l_gt, l_st, n_boot=n_boot, ci=ci,
                                     generator=generator)
        for k, v in arm.items():
            out[f"pix_arm_{k}"] = v
        out[f"{prefix}rank_ok"] = float(ctrl["auc"] >= arm["auc"])
    return out


# ---------------------------------------------------------------------------
# Offline calibration CLI.  `python -m model.pixel_texture_disc calibrate ...`
# ---------------------------------------------------------------------------
def _cli(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover
    ap = argparse.ArgumentParser(
        prog="model.pixel_texture_disc",
        description=("§8.1 C1 amplitude calibration, and the measured "
                     "architecture facts (params / receptive field / P)."),
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="sweep the C1 amplitude offline")
    c.add_argument("--latents", required=True,
                   help=".pt or .npy of GT latents [..., C, H, W]")
    c.add_argument("--amplitudes", default="0.05,0.1,0.2,0.35,0.5,0.75,1.0,1.5")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--hf-cut", type=float, default=0.25)
    c.add_argument("--c1-hf-cut", type=float, default=0.55)
    c.add_argument("--json", default=None, help="write the sweep here")

    sub.add_parser("facts", help="print the measured architecture facts")

    st = sub.add_parser(
        "stability",
        help=("SYNTHETIC D-only training smoke, norm-free vs normed.  A "
              "falsifier for the recorded risk of the §4 GroupNorm removal -- "
              "NOT evidence about the real arm (no G, no VAE, no real data)."))
    st.add_argument("--steps", type=int, default=300)
    st.add_argument("--lr", type=float, default=PIX_GAN_LR)
    st.add_argument("--seed", type=int, default=0)

    args = ap.parse_args(argv)

    if args.cmd == "facts":
        net = PixelTextureDisc()
        n_param = sum(p.numel() for p in net.parameters())
        # Probed at the REAL post-trim crop, not a square toy: the NORMED
        # variant's receptive field is input-size dependent (it is the whole
        # input), so a square toy would understate it.
        rf = measure_receptive_field(176, input_size_w=240)
        gh, gw = PixelTextureDisc.patch_grid(176, 240)
        print(json.dumps({
            "spectral_norm_api": SPECTRAL_NORM_API,
            "groupnorm": net.uses_norm,     # False: authorised §4 deviation
            "params": n_param,
            "receptive_field_px": rf,
            "patch_grid_176x240": [gh, gw],
            "P_176x240": gh * gw,
            "effective_samples": effective_sample_count(176, 240),
        }, indent=2))
        return 0

    if args.cmd == "stability":
        rows = [synthetic_stability_smoke(steps=args.steps, use_norm=u,
                                          lr=args.lr, seed=args.seed)
                for u in (False, True)]
        print(json.dumps({"SYNTHETIC_NOT_EVIDENCE_ABOUT_THE_REAL_ARM": True,
                          "rows": rows}, indent=2, default=float))
        return 0

    if str(args.latents).endswith(".npy"):
        import numpy as np
        x = torch.from_numpy(np.load(args.latents))
    else:
        x = torch.load(args.latents, map_location="cpu")
    if isinstance(x, dict):
        x = next(v for v in x.values() if torch.is_tensor(v))
    amps = [float(a) for a in str(args.amplitudes).split(",") if a.strip()]
    res = sweep_c1_amplitude(
        x.float(), amps, seed=int(args.seed), hf_cut=float(args.hf_cut),
        c1_kwargs={"hf_cut": float(args.c1_hf_cut)},
    )
    txt = json.dumps(res, indent=2, default=float)
    print(txt)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(txt)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
