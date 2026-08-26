"""Unpaired, region-level Gram / AdaIN STYLE loss (WP-STYLE).

WHY THIS EXISTS
===============
The objective is *style shift to the training dataset's appearance
distribution*, and as of 2026-08-25 **nothing in the training path measures
style**:

  * the LADD discriminator scores transformer features of LATENTS at a fixed
    noise level, projected 1536->256 -- structurally blind to texture; it
    improves geometry, not appearance;
  * an exhaustive grep found NO Gram/AdaIN term anywhere under ``trainer/``,
    ``model/`` or ``pipeline/``. Every Gram symbol in the repo lives in
    OFFLINE eval (``grids/eval/style_shift.py``,
    ``analysis/style_shift/axis2_local_style.py``,
    ``utils/style_shift_detect.py``) and none of those import into training;
  * wavelet-HF is dead, the MAE gate is dead, and the pretrained-teacher
    surrogates are indirect and need ~8 head updates per generator step
    before they say anything at all.

This module is the direct, online, differentiable measure of dataset-style
distance. It has **no warm-up**: a Gram statistic on a FROZEN encoder is
meaningful at step 0, which is the whole argument for this route over the
adversarial one when quick convergence is a requirement.

WHAT IT MEASURES -- and why it is the SAME thing the offline metric measures
===========================================================================
The feature basis and the Gram normalisation are taken VERBATIM from the
offline evaluator so the training loss and the evaluation metric cannot
drift apart:

  ``grids/eval/style_shift.py::VGGStyle``
      VGG16 ``features``, taps ``[3, 8, 15, 22]`` = relu1_2 / relu2_2 /
      relu3_3 / relu4_3, ImageNet-normalised input, and

          f = x.reshape(b, c, h*w);  g = bmm(f, f^T) / (c*h*w)

      which is :func:`gram_matrix` below, character for character.

  ``grids/eval/style_shift.py::gram_distance``
      per-layer relative Frobenius distance between the MEAN Gram of each
      window, averaged over layers -- :func:`gram_distance_eval` below.

  ``analysis/style_shift/axis2_local_style.py``
      the LOCAL-style axis: VGG16 ``features[:16]``, taps ``{3, 8, 15}``,
      per-tile channel mean+std (AdaIN stats). That is why the default tap
      set here is ``(3, 8, 15)`` and why :func:`adain_stats` exists as the
      alternative mode.

Those files are eval-only scripts outside any importable package (``grids/``
and ``analysis/`` are not on the path and are excluded from the trainer's
source scans), so they cannot be imported from training code. The formulas
are therefore RESTATED here, and
``testing/test_style_gram_loss.py::test_gram_matches_the_offline_evaluator``
pins them against a hand-copied transcription of the offline source. If the
offline metric ever changes, that test is what fails.

UNPAIRED, and REGION-LEVEL
==========================
A Gram matrix sums over spatial positions, so it is invariant to any
permutation of the H*W grid (proved by
``test_gram_is_invariant_to_spatial_permutation``). That invariance is
exactly what makes the loss unpaired: the real crops come from DIFFERENT
rides, at different times, with no spatial correspondence to the student's
crops whatsoever. We are matching a DISTRIBUTION, not reconstructing an
image -- there is no per-pixel target anywhere in this file.

"Region-level" means the statistics are pooled inside an A24 vertical BAND
(coarse thirds: sky / mid / road) and matched band-to-band. A single global
Gram would be satisfiable by making everything look like sky; banding pins
each region against its own kind of real content. The band index is the only
spatial information used, and A24's ban on any finer y-matching is honoured
by construction -- nothing here ever sees a row offset.

TWO-SIDEDNESS (a researcher-mandated property, not an accident)
===============================================================
"Sharpness can get the thing to go really blocky and pixelated so don't
optimise for sharpness." A Gram loss MATCHES a statistic instead of
maximising one, so it is naturally two-sided: the diagonal ``G_ii`` is the
second moment (the texture ENERGY) of channel ``i``, and any distance to a
target on that quantity RISES when the student overshoots exactly as it
rises when it undershoots. Today's measured failure -- an arm overshooting
its own ground truth's high-frequency banding by 1.55x in tree crowns --
would be PENALISED by this loss, not rewarded.

That property is fragile in exactly one place, and it is the place the
offline metric happens to be shaped wrong for training use: the
normalisation DENOMINATOR. The offline
``(||G_a|| + ||G_b||)/2`` is symmetric, and differentiating through it would
hand the generator a way to lower the loss by INFLATING ``||G_fake||`` --
i.e. a monotone reward for more texture, the precise failure mode we were
told to avoid. So the OPTIMISED form
(:func:`gram_style_distance` / :func:`adain_style_distance`) normalises by
the REAL side only, always detached:

        d_l = || G_fake - G_real ||_F  /  (|| G_real ||_F + eps)

The real side carries no graph anyway (it is decoded under ``no_grad``), so
this costs nothing and the denominator is a constant w.r.t. the generator.
The symmetric offline form is still COMPUTED and LOGGED, detached, as
``style_gram_dist_evalform``, so the trace stays comparable with
``results_style.csv``. ``test_loss_rises_when_texture_overshoots`` is the
regression test for all of this; it fails on the symmetric denominator.

MODES
=====
``gram``  (default) second moments: ``G = f f^T / (C H W)``. Captures channel
          CO-occurrence -- the classic Gatys style representation.
``adain`` first+second marginal moments: per-channel mean and std. Cheaper,
          weaker (it ignores cross-channel structure), and is the statistic
          ``axis2_local_style.py`` uses. Kept because it is the offline
          local-style axis and because a mode flag makes the choice
          measurable rather than arguable.

This module NEVER reads yaml. Every knob arrives as a constructor or
function argument; the trainer owns resolution (one resolution point).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

__all__ = [
    "STYLE_DEFAULTS",
    "SUPPORTED_STYLE_ENCODERS",
    "SUPPORTED_STYLE_MODES",
    "FrozenStyleEncoder",
    "gram_matrix",
    "adain_stats",
    "region_mean_gram",
    "region_mean_adain",
    "gram_distance_eval",
    "gram_style_distance",
    "adain_style_distance",
    "style_distance",
]

# ImageNet statistics -- the normalisation BOTH offline implementations use
# (``grids/eval/style_shift.py`` IMAGENET_MEAN/STD, ``axis2_local_style.py``
# MEAN/STD) and the one ``model/pretrained_pixel_disc.py`` uses for its
# frozen backbones. One convention, three consumers.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

SUPPORTED_STYLE_ENCODERS = ("vgg16", "convnext")
SUPPORTED_STYLE_MODES = ("gram", "adain")

#: Numerical floor for every relative denominator in this file. Matches the
#: offline ``gram_distance``'s ``1e-8``.
STYLE_EPS = 1e-8

#: Shipped defaults, mirrored (as string keys) by the trainer's resolver.
#: Kept here beside the code they configure, in the shape
#: ``model/pixel_texture_disc.py::PIX_DEFAULTS`` established.
STYLE_DEFAULTS: Dict[str, object] = {
    "style_gram_loss_weight": 0.0,          # OFF. The whole-feature gate.
    "style_gram_mode": "gram",
    "style_gram_encoder": "vgg16",
    "style_gram_layers": None,              # None -> the encoder's default taps
    "style_gram_every": 1,
    "style_gram_crops": 4,
    "style_gram_frames_per_crop": 2,
    "style_gram_crop_lat": (24, 32),
    "style_gram_lat_frames_per_crop": 2,
    "style_gram_decode_border_trim": 8,
    "style_gram_decode_batch": 2,
    "style_gram_band_count": 3,
    "style_gram_real_ema": 0.9,
    "style_gram_real_pool_windows": 1024,
    "style_gram_real_pool_refresh": 4,
    "style_gram_fake_source": "pred",       # "pred" | "pix"
    "style_gram_warmup_steps": 0,
    "style_gram_seed": 20260825,
}

# VGG16 ``features`` indices of the post-ReLU taps, named as the offline
# code names them. ``LAYERS = [3, 8, 15, 22]`` in ``VGGStyle``; ``TAPS =
# {3, 8, 15}`` in ``axis2_local_style``.
VGG16_TAP_NAMES = {3: "relu1_2", 8: "relu2_2", 15: "relu3_3", 22: "relu4_3"}

#: DEFAULT TAP SET -- relu1_2 / relu2_2 / relu3_3, i.e. the offline LOCAL
#: STYLE axis, and the first three of ``VGGStyle``'s four.
#:
#: relu4_3 is DROPPED on purpose. It is the 512-channel block: it roughly
#: triples the encoder's FLOPs and stored activations for a 512x512 Gram
#: whose content is semantic rather than textural, and
#: ``axis2_local_style.py`` -- the axis this loss is aligned to -- stops at
#: 15 for that reason. It remains available via ``style_gram_layers`` so the
#: choice is measurable.
VGG16_DEFAULT_LAYERS = (3, 8, 15)


# ---------------------------------------------------------------------------
# The frozen encoder.
# ---------------------------------------------------------------------------
class FrozenStyleEncoder(nn.Module):
    """A frozen, texture-biased pixel encoder that returns tapped features.

    ENCODER CHOICE (stated, not assumed):

    ``vgg16`` (DEFAULT). Three reasons, in order of weight.
      1. It is *the same feature basis the offline metric uses*
         (``grids/eval/style_shift.py``, ``axis2_local_style.py``). Training
         and evaluation then measure the same thing, which is the entire
         point of aligning them.
      2. ImageNet-supervised CNNs are TEXTURE-BIASED (Geirhos et al.,
         ICLR 2019: they classify by texture where ViTs and humans use
         shape). ``model/pretrained_pixel_disc.py`` already records this
         argument for its ConvNeXt option; for a loss whose entire job is
         texture the bias is an asset, not a caveat.
      3. It is by far the CHEAPEST candidate. ``features[:16]`` is 1.74 M
         parameters (~7 MB fp32) against ConvNeXt-T's 28 M, DINOv2 ViT-S/14's
         21 M and SAM2 Hiera-B+'s ~80 M -- and the weights are already in the
         node-local torch hub cache (``vgg16-397923af.pth``), so the
         constructor never touches the network.

    ``convnext``. The alternative, reusing
    ``model/pretrained_pixel_disc._build_convnext`` verbatim (offline-first
    local ``state_dict``). Also texture-biased, hierarchical, and already
    loaded by the surrogate arms -- so on an arm that runs a ConvNeXt
    surrogate the same weights serve both and the marginal memory is zero.

    NOT REUSED, deliberately: ``PretrainedPixelDisc`` itself. Its ``_prep``
    pads to square and resizes every input to 512 px, and RESIZING CHANGES
    TEXTURE SCALE -- the exact quantity this loss measures. Style features
    must be read at the crops' native decoded resolution, which is identical
    on the real and fake side by construction because both come from the
    same latent-crop geometry. Hence this thin, resize-free wrapper.

    The encoder is frozen (``requires_grad_(False)``, pinned to ``eval()``):
    no weight gradients are ever allocated, but the INPUT path is live so
    gradient flows back through the decode to the generator.
    """

    def __init__(
        self,
        encoder: str = "vgg16",
        *,
        layers: Optional[Sequence[int]] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        encoder = str(encoder).lower()
        if encoder not in SUPPORTED_STYLE_ENCODERS:
            raise ValueError(
                f"style_gram_encoder must be one of "
                f"{SUPPORTED_STYLE_ENCODERS}; got {encoder!r}."
            )
        if device is None:
            device = (
                torch.device("cuda") if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.encoder_name = encoder
        self._dtype = dtype

        if encoder == "vgg16":
            self.taps = tuple(
                int(l) for l in (
                    VGG16_DEFAULT_LAYERS if layers is None else layers
                )
            )
            if not self.taps:
                raise ValueError("style_gram_layers must not be empty.")
            if min(self.taps) < 0:
                raise ValueError(
                    f"style_gram_layers must be non-negative VGG16 "
                    f"``features`` indices; got {self.taps}."
                )
            import torchvision

            vgg = torchvision.models.vgg16(
                weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1,
            )
            # Truncate at the deepest tap -- the offline evaluator does
            # exactly this (``vgg.features[: self.LAYERS[-1] + 1]``), and it
            # is what keeps the cost proportional to the tap set rather than
            # to VGG16.
            self.features = vgg.features[: max(self.taps) + 1]
            self.features.to(device=device, dtype=dtype).eval()
            self.features.requires_grad_(False)
            self._min_spatial = 2 ** sum(
                1 for i, m in enumerate(self.features)
                if isinstance(m, nn.MaxPool2d) and i <= max(self.taps)
            )
            self._forward_impl = self._forward_vgg
            n_params = sum(p.numel() for p in self.features.parameters())
            self.tap_names = [
                VGG16_TAP_NAMES.get(int(t), f"features{int(t)}")
                for t in self.taps
            ]
        else:                                          # convnext
            # REUSE, never reimplement: the offline-first loader, its cache
            # path, its ImageNet normalisation assumption and its taps all
            # come from the module that already owns them.
            from model.pretrained_pixel_disc import (
                _DEFAULT_VARIANT, _build_convnext,
            )

            model, fwd, _multiple, taps = _build_convnext(
                _DEFAULT_VARIANT["convnext"], layers, device, dtype,
            )
            self.features = model
            self._convnext_forward = fwd
            self.taps = tuple(int(t) for t in taps)
            self._min_spatial = 32
            self._forward_impl = self._forward_convnext
            n_params = sum(p.numel() for p in model.parameters())
            self.tap_names = [f"stage{int(t)}" for t in self.taps]

        self.register_buffer(
            "_mean",
            torch.tensor(_IMAGENET_MEAN, dtype=dtype, device=device)
            .view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_std",
            torch.tensor(_IMAGENET_STD, dtype=dtype, device=device)
            .view(1, 3, 1, 1),
            persistent=False,
        )
        self.n_params = int(n_params)
        logging.info(
            "[StyleGram] frozen %s encoder: taps=%s (%s) params=%.2fM "
            "dtype=%s device=%s -- NO resize, native crop resolution.",
            encoder, list(self.taps), self.tap_names, n_params / 1e6,
            dtype, device,
        )

    # -- frozen forever -------------------------------------------------
    def train(self, mode: bool = True):
        """Same override, same reason, as ``PretrainedPixelDisc.train``: a
        stray ``.train()`` must never flip BN / stochastic depth in a
        backbone whose statistics have to stay fixed."""
        super().train(False)
        self.features.eval()
        return self

    # -- input prep -----------------------------------------------------
    def _prep(self, px: torch.Tensor) -> torch.Tensor:
        """``[N, 3, H, W]`` in ``[-1, 1]`` -> ImageNet-normalised.

        NOT wrapped in ``no_grad``. NO resize, NO pad -- see the class
        docstring; resizing would rescale the very texture being measured.
        """
        if px.dim() != 4 or int(px.shape[1]) != 3:
            raise ValueError(
                f"FrozenStyleEncoder expects [N, 3, H, W]; got "
                f"{tuple(px.shape)}."
            )
        h, w = int(px.shape[-2]), int(px.shape[-1])
        if min(h, w) < self._min_spatial:
            raise ValueError(
                f"style crop is {h}x{w} px but the {self.encoder_name} tap "
                f"set needs at least {self._min_spatial} px on each side "
                "(the pooling stack would collapse a shorter side to zero). "
                "Raise style_gram_crop_lat or lower "
                "style_gram_decode_border_trim."
            )
        x = (px.clamp(-1.0, 1.0) + 1.0) * 0.5           # [-1,1] -> [0,1]
        return (x.to(self._dtype) - self._mean) / self._std

    def _forward_vgg(self, x: torch.Tensor) -> List[torch.Tensor]:
        want = set(self.taps)
        out: Dict[int, torch.Tensor] = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in want:
                out[i] = x
        return [out[int(t)] for t in self.taps]

    def _forward_convnext(self, x: torch.Tensor) -> List[torch.Tensor]:
        return list(self._convnext_forward(x))

    def forward(self, px: torch.Tensor) -> List[torch.Tensor]:
        """``[N, 3, H, W]`` in ``[-1, 1]`` -> list of ``[N, C, h, w]``."""
        return self._forward_impl(self._prep(px))


# ---------------------------------------------------------------------------
# The statistics. Pure functions -- no config, no state, no device logic.
# ---------------------------------------------------------------------------
def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    """``[N, C, H, W]`` -> ``[N, C, C]`` Gram matrices.

    VERBATIM the offline normalisation
    (``grids/eval/style_shift.py::VGGStyle.forward``)::

        b, c, h, w = x.shape
        f = x.reshape(b, c, h * w)
        g = torch.bmm(f, f.transpose(1, 2)) / (c * h * w)

    Dividing by ``c * h * w`` (not ``h * w``) is the offline convention and
    is kept so the magnitudes -- and therefore any threshold read off
    ``results_style.csv`` -- transfer directly.

    Invariant to any permutation of the ``H * W`` positions, because the
    contraction sums over them. THAT is what makes this loss unpaired.
    """
    if feat.dim() != 4:
        raise ValueError(f"gram_matrix expects [N,C,H,W]; got {tuple(feat.shape)}")
    b, c, h, w = feat.shape
    f = feat.reshape(b, c, h * w)
    return torch.bmm(f, f.transpose(1, 2)) / float(c * h * w)


def adain_stats(feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``[N, C, H, W]`` -> ``(mean [N, C], std [N, C])``.

    The AdaIN / ``axis2_local_style.py`` statistic: per-channel first and
    second marginal moments, biased (population) std to match the offline
    ``sqrt(E[f^2] - E[f]^2)``. Also permutation-invariant.
    """
    if feat.dim() != 4:
        raise ValueError(f"adain_stats expects [N,C,H,W]; got {tuple(feat.shape)}")
    b, c, h, w = feat.shape
    f = feat.reshape(b, c, h * w)
    mu = f.mean(dim=-1)
    var = (f * f).mean(dim=-1) - mu * mu
    return mu, var.clamp_min(0.0).sqrt()


def region_mean_gram(
    feats: Sequence[torch.Tensor],
    groups: Optional[Sequence[Sequence[int]]] = None,
) -> List[List[torch.Tensor]]:
    """Per-layer, per-region MEAN Gram: ``[layer][region] -> [C, C]``.

    Averaging the Grams over the images in a region -- rather than averaging
    a per-image distance -- is what makes this a DISTRIBUTION match: the
    estimator is the region's mean second moment, exactly the quantity
    ``gram_distance`` compares between two windows offline (``ga.mean(0)``).

    ``groups`` is a list of index lists (one per region). ``None`` means one
    region containing everything.
    """
    n = int(feats[0].shape[0])
    if groups is None:
        groups = [list(range(n))]
    out: List[List[torch.Tensor]] = []
    for f in feats:
        g = gram_matrix(f)
        out.append([g[list(idx)].mean(dim=0) for idx in groups])
    return out


def region_mean_adain(
    feats: Sequence[torch.Tensor],
    groups: Optional[Sequence[Sequence[int]]] = None,
) -> List[List[Tuple[torch.Tensor, torch.Tensor]]]:
    """Per-layer, per-region mean ``(mu, sd)`` -- the AdaIN twin of
    :func:`region_mean_gram`."""
    n = int(feats[0].shape[0])
    if groups is None:
        groups = [list(range(n))]
    out: List[List[Tuple[torch.Tensor, torch.Tensor]]] = []
    for f in feats:
        mu, sd = adain_stats(f)
        out.append([
            (mu[list(idx)].mean(dim=0), sd[list(idx)].mean(dim=0))
            for idx in groups
        ])
    return out


def gram_distance_eval(
    grams_a: Sequence[torch.Tensor],
    grams_b: Sequence[torch.Tensor],
) -> float:
    """The OFFLINE metric, restated -- for LOGGING ONLY.

    ``grids/eval/style_shift.py::gram_distance``: mean over layers of the
    Frobenius distance between the two mean Grams, normalised by their
    average norm. Symmetric denominator.

    NEVER OPTIMISE THIS. Differentiating through ``||G_fake||`` in the
    denominator rewards inflating the fake's texture energy -- a monotone
    "more texture is better" term, which is precisely the blocky/pixelated
    failure the researcher ruled out. Everything here is detached and the
    return type is a plain ``float`` so it cannot be added to a loss by
    accident.
    """
    if len(grams_a) != len(grams_b):
        raise ValueError("gram_distance_eval: layer counts differ.")
    d = 0.0
    for ga, gb in zip(grams_a, grams_b):
        ma, mb = ga.detach(), gb.detach()
        denom = (ma.norm() + mb.norm()) / 2 + STYLE_EPS
        d += float((ma - mb).norm() / denom)
    return d / max(1, len(grams_a))


def gram_style_distance(
    fake_grams: Sequence[torch.Tensor],
    real_grams: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, List[float]]:
    """The OPTIMISED Gram distance. Two-sided by construction.

    ``d = mean_l || G_fake_l - G_real_l ||_F / (|| G_real_l ||_F + eps)``

    Both arguments are lists over LAYERS of ``[C, C]`` region-mean Grams,
    already reduced over the region's images. ``real_grams`` is detached
    here regardless of what the caller passed, so:

      * the denominator is a CONSTANT w.r.t. the generator -- nothing in
        this expression can be lowered by making ``G_fake`` bigger; and
      * the only way down is toward ``G_real``, from either side.
        Overshoot raises the numerator exactly as undershoot does.

    Unsquared Frobenius (matching the offline metric) rather than squared:
    its gradient magnitude does not vanish as the gap closes, which is worth
    real convergence speed when the whole requirement is a usable signal
    inside ~50 steps.

    Returns ``(loss, per_layer_values)``.
    """
    if len(fake_grams) != len(real_grams):
        raise ValueError("gram_style_distance: layer counts differ.")
    terms: List[torch.Tensor] = []
    for gf, gr in zip(fake_grams, real_grams):
        target = gr.detach()
        if gf.shape != target.shape:
            raise ValueError(
                f"gram_style_distance: fake Gram {tuple(gf.shape)} vs real "
                f"{tuple(target.shape)} -- different encoders or taps."
            )
        denom = target.norm() + STYLE_EPS
        terms.append((gf - target).norm() / denom)
    stacked = torch.stack(terms)
    return stacked.mean(), [float(t.detach()) for t in terms]


def adain_style_distance(
    fake_stats: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    real_stats: Sequence[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, List[float]]:
    """The AdaIN twin of :func:`gram_style_distance`, same two-sidedness.

    Mean and std are given equal weight after each is normalised by its own
    real-side norm, so neither can dominate purely by scale. Matching the
    STD is a matching term, not a maximising one: an over-textured student
    has too LARGE a std and is pushed back down.
    """
    if len(fake_stats) != len(real_stats):
        raise ValueError("adain_style_distance: layer counts differ.")
    terms: List[torch.Tensor] = []
    for (mf, sf), (mr, sr) in zip(fake_stats, real_stats):
        mr_d, sr_d = mr.detach(), sr.detach()
        d_mu = (mf - mr_d).norm() / (mr_d.norm() + STYLE_EPS)
        d_sd = (sf - sr_d).norm() / (sr_d.norm() + STYLE_EPS)
        terms.append(0.5 * (d_mu + d_sd))
    stacked = torch.stack(terms)
    return stacked.mean(), [float(t.detach()) for t in terms]


def style_distance(
    fake_feats: Sequence[torch.Tensor],
    real_stats: object,
    *,
    mode: str,
    fake_groups: Optional[Sequence[Sequence[int]]] = None,
) -> Tuple[torch.Tensor, List[float]]:
    """Dispatch: reduce ``fake_feats`` to region statistics and compare.

    ``real_stats`` is the ALREADY-REDUCED real side (a
    :func:`region_mean_gram` / :func:`region_mean_adain` result, typically
    an EMA bank), indexed ``[layer][region]``. The regions of the two sides
    must correspond one-for-one; the caller guarantees that by grouping both
    on the same A24 band indices.
    """
    mode = str(mode).lower()
    if mode not in SUPPORTED_STYLE_MODES:
        raise ValueError(
            f"style_gram_mode must be one of {SUPPORTED_STYLE_MODES}; "
            f"got {mode!r}."
        )
    if mode == "gram":
        fake = region_mean_gram(fake_feats, fake_groups)
    else:
        fake = region_mean_adain(fake_feats, fake_groups)

    n_layers = len(fake)
    n_regions = len(fake[0])
    if len(real_stats) != n_layers:                     # type: ignore[arg-type]
        raise ValueError(
            "style_distance: real bank has "
            f"{len(real_stats)} layers, fake has {n_layers}."  # type: ignore[arg-type]
        )
    losses: List[torch.Tensor] = []
    per_layer_acc = [0.0] * n_layers
    n_used = 0
    for r in range(n_regions):
        real_r = [real_stats[l][r] for l in range(n_layers)]  # type: ignore[index]
        if any(x is None for x in real_r):
            # No real statistic for this region yet. OMITTED, never
            # zero-filled: a 0.0 distance is the PERFECT reading and would
            # certify a match that was never computed (forgeable-zero rule).
            continue
        fake_r = [fake[l][r] for l in range(n_layers)]
        if mode == "gram":
            d, per_layer = gram_style_distance(fake_r, real_r)
        else:
            d, per_layer = adain_style_distance(fake_r, real_r)
        losses.append(d)
        for l in range(n_layers):
            per_layer_acc[l] += per_layer[l]
        n_used += 1
    if not losses:
        raise RuntimeError(
            "style_distance: no region had a real reference. The real bank "
            "was never filled -- refusing to return a forged 0.0 distance."
        )
    total = torch.stack(losses).mean()
    return total, [v / float(n_used) for v in per_layer_acc]
