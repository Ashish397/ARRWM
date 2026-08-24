"""A22 — HELD-OUT DISCRIMINATOR GENERALISATION PROBE (default OFF).

Companion instrument to the A14 positive control, answering a *different*
question.  Keep the two apart:

* **A14 positive control** — ``has D learned to recognise obvious
  corruption?``  Scores ``decode(GT)`` against a deliberately corrupted real.
  Makes ``d_loss ~ ln 2`` decidable.
* **A22 (this file)** — ``is D memorising its real training crops?``  Scores
  ``D(decode(GT))`` vs ``D(decode(fake))`` on **train** rides AND on
  **held-out** rides the discriminator has NEVER seen as a real training
  example, and reports the two margins / accuracies side by side.

INTERPRETATION TABLE (do not lose this — it is the whole point of the probe)
---------------------------------------------------------------------------
``acc`` below is the pairwise real-beats-fake accuracy (== ROC-AUC of the
real vs fake score distributions).  Chance is 0.5.

| train margin / acc | held-out margin / acc | verdict                        |
|--------------------|-----------------------|--------------------------------|
| up                 | up                    | **GOOD** — D has learned a     |
|                    |                       | generalisable notion of real   |
|                    |                       | texture.                       |
| up                 | ~ chance              | **MEMORISATION** — D is        |
|                    |                       | fitting its own real crops;    |
|                    |                       | the G-side signal is worthless.|
| ~ chance           | ~ chance              | undecidable **without** the    |
|                    |                       | A14 positive control:          |
|                    |                       |  * control PASSES -> student   |
|                    |                       |    texture is genuinely hard   |
|                    |                       |    to distinguish at this      |
|                    |                       |    depth;                      |
|                    |                       |  * control FAILS  -> D is      |
|                    |                       |    undertrained / underpowered.|
| ~ chance           | up                    | **INVERTED** — not a listed    |
|                    |                       | outcome; the probe's train and |
|                    |                       | held-out pools are not         |
|                    |                       | comparable.  Treat as a build  |
|                    |                       | bug, not a result.             |
| below chance       | (either side)         | **SIGN-FLIPPED** — reals score |
|                    |                       | BELOW fakes.  ``roc_auc``      |
|                    |                       | assumes higher == more real,   |
|                    |                       | and nothing in the critic      |
|                    |                       | contract pins that sign.       |
|                    |                       | Below-chance separation IS     |
|                    |                       | separation; without this row a |
|                    |                       | perfectly memorising flipped   |
|                    |                       | critic reads BOTH_CHANCE.      |

FAIL CLOSED (2026-08-23 adversarial review, GAN_REDESIGN A22 D2/D3)
-------------------------------------------------------------------
Every path where this probe cannot make a valid measurement used to ``return``
after emitting ``dhp_leak_rides=0 / dhp_leak_seen=0 / dhp_ring_delta=0`` —
**the exact signature of a clean run**.  A reserved root that never resolved
(``holdout_eval_root`` exists on ``--override`` lines only, in no yaml), an
empty reserved ride set, an unreachable ``dataset._rides``: all three reported
green while checking nothing.  They now route through ``_fail_closed``, which

* sets ``dhp_invalid=1`` and ``dhp_invalid_reason=<INVALID_* code>``, and
* **NaNs** ``dhp_leak_rides`` / ``dhp_leak_seen`` / ``dhp_ring_delta`` so the
  green signature is unreachable from a broken configuration, and
* raises ``ProbeConfigError`` under ``disc_holdout_probe_strict`` (default).

``dhp_invalid`` is written on EVERY fire (0.0 on a healthy one), so its
absence from a run's history is itself readable.

``classify()`` encodes exactly this table.  Pass ``control_auc`` (the A14
positive control's ROC-AUC) once A14 lands and the ``~chance / ~chance`` row
splits itself; until then it reports ``VERDICT_BOTH_CHANCE``.

HOLDOUT INTEGRITY (the reason this file is paranoid)
----------------------------------------------------
``docs/GAN_REDESIGN.md`` A22 records that *a prior holdout list leaked into
training once before*.  The mechanism was structural, not clerical: the
training ``encoded_root`` was switched from ``frodobots_encoded_weu`` to its
superset ``frodobots_encoded_weunz`` (which physically contains the 20
reserved rides) while ``holdout_zarr_list`` was left ``null``.  Nothing in the
code noticed.

Two independent guards therefore live here, and BOTH are telemetry, not
comments:

0. **Configuration reachability** (``_fail_closed``).  A probe that is on
   but has no reserved root, no reserved rides or no training ride list is an
   ERROR, not a pass — see FAIL CLOSED above.
1. **Static set disjointness** (``holdout_leak_check``).  Ride identity is the
   zarr **basename** (``<timestamp>.zarr``) because that is what
   ``holdout_zarr_list`` matches on and what makes the weu/weunz superset
   collision visible.  The probe intersects the held-out basenames against the
   *post-filter* training ride list (``trainer.dataset._rides``) — the single
   list that feeds every real the discriminator ever sees.  Emits
   ``dhp_leak_rides``; **must be 0**.
2. **Dynamic observation counting** (``note_real_supply_ride``).  Hooked into
   ``_streaming_setup_sequence_from_ride`` — the one funnel through which a
   ride reaches ``streaming_state['ride_latents_window']``,
   ``streaming_state['gt_match_latents']`` and ``trainer._ladd_real_ring``,
   i.e. every real-supply path.  Counts held-out basenames actually observed
   there.  Emits ``dhp_leak_seen``; **must be 0**.

A third, self-directed guard: the probe measures ``len(_ladd_real_ring)``
either side of its own run and emits ``dhp_ring_delta`` (must be 0), so the
instrument can never be the contaminant.

What these guards do NOT catch is recorded in the module's report; the short
version is that they prove *name* disjointness, not *content* disjointness.

WIRING (B2 integration point)
-----------------------------
The pixel critic (GAN_REDESIGN B2, ``model/pixel_texture_disc.py``) does not
exist yet.  This probe is critic-agnostic: it needs one callable

    score_fn(pixels: FloatTensor[N, 3, H, W] in [-1, 1]) -> FloatTensor

returning either per-image scalars ``[N]`` or a patch map ``[N, 1, h, w]``
(reduced here by mean over the non-batch dims).  Resolution order:

    1. ``trainer._dhp_score_fn``      (set via ``register_score_fn``)
    2. ``getattr(trainer, cfg.disc_holdout_probe_critic_attr)``
       — default ``"pixel_texture_disc"``, the B2 attribute name.

**B2's one-line integration**: after constructing the critic, call
``disc_holdout_probe.register_score_fn(trainer, lambda px: critic(px))`` — or
simply name the module attribute ``pixel_texture_disc`` and the probe finds it
with no code change at all.

Until a critic is attached the probe emits ``dhp_no_critic=1.0`` and returns;
the leak guards still run, so holdout integrity is measurable *before* B2
lands.

**Fake source (A2 / A23).**  The probe scores whatever latents it is handed.
The trainer call site passes ``train_chunk``, but training's GAN may consume
``info['flash_dmd_gan_x0']`` (the t=60 flash tensor) instead — and scoring a
tensor D was never trained on would make the readout meaningless.  B2 therefore
sets its own fake via ``register_fake_source(trainer, x)`` immediately before
the probe fires; the stash is consumed-and-cleared each fire, and a fire that
falls back to the trainer's default tensor is flagged
``dhp_fake_src_default=1.0``.  Treat a run where that key is 1.0 while B2 is
active as uninterpretable.

**Path coverage.**  The two real-supply tripwires cover BOTH ride funnels
(streaming ``_streaming_setup_sequence_from_ride`` and legacy
``_fwdbwd_one_step``), and the static check covers everything by construction.
The *measurement* itself is wired into the streaming per-chunk path only —
which is the path every live arm uses (``streaming_mode: true``,
``configs/action_forcing_phase1.yaml:778``).  A ``streaming_mode=false`` run
would get the leak guards but no margins.

The existing LADD latent critic could also be served here (swap the decode for
a raw-latent score call), but it is NOT wired: that critic consumes latents
plus action tokens plus prompt embeds and reduces ~47k tokens to one scalar,
so its "crop" has no meaning and the pairing contract differs.  Not forced —
see the report.

COST / CONTRACT
---------------
No gradient, no backward, no collective, no mutation of ``streaming_state``,
no push to the real ring, no optimizer touch.  Every exception is swallowed
into ``dhp_err`` — a diagnostic must never take training down.
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

# W&B key prefix.  Every key this module emits starts with this.
PREFIX = "train/dhp_"


class ProbeIntegrityError(RuntimeError):
    """Base type for "this probe is not measuring what it claims to".

    A dedicated family so the probe's blanket ``except Exception`` can
    re-raise THESE and swallow everything else — an OOM in a diagnostic must
    not kill a 6-hour training job, but a probe that is silently measuring
    nothing must never be logged as if it were a measurement."""


class HoldoutLeakError(ProbeIntegrityError):
    """Raised (strict mode only) when reserved rides are reachable from the
    discriminator's real supply."""


class ProbeConfigError(ProbeIntegrityError):
    """Raised (strict mode only) when the probe is ENABLED but structurally
    cannot make a valid measurement: no reserved root resolved (A22 D2), an
    empty reserved ride set, an unreachable training ride list (A22 D3),
    reference pools that do not match in ride/window count (A22 D1), or a
    fake whose latent geometry does not match the cached pools (A22 D9).
    Every one of those used to return the clean green signature."""


# ``dhp_invalid_reason`` codes.  0 == valid; anything else means the numbers
# above it in the log line are NOT a measurement.
INVALID_NONE = 0.0
INVALID_NO_HOLDOUT_ROOT = 1.0     # D2: no reserved root configured at all
INVALID_NO_HOLDOUT_RIDES = 2.0    # D2: root configured, zero .zarr under it
INVALID_NO_TRAIN_RIDES = 3.0      # D3: dataset._rides missing / empty
INVALID_POOL_MISMATCH = 4.0       # D1: pools differ in rides or windows
INVALID_GEOMETRY = 5.0            # D9: fake geometry != cached pool geometry
INVALID_POOL_EMPTY = 6.0          # a pool could not be built at all

# ``classify`` verdict codes (logged as ``dhp_verdict``).
VERDICT_UNKNOWN = 0.0
VERDICT_GENERALISING = 1.0        # both margins up            -> good
VERDICT_MEMORISING = 2.0          # train up, held-out chance  -> memorisation
VERDICT_BOTH_CHANCE = 3.0         # both chance, no A14 control -> undecidable
VERDICT_STUDENT_HARD = 4.0        # both chance, control passes
VERDICT_D_UNDERPOWERED = 5.0      # both chance, control fails
VERDICT_INVERTED = 6.0            # held-out up, train chance  -> build bug
VERDICT_SIGN_FLIPPED = 7.0        # reals score BELOW fakes -> critic sign flip


# ---------------------------------------------------------------------------
# Config surface.  Everything is read off ``trainer.config`` with these
# defaults; the master gate ``disc_holdout_probe_every`` is 0 (OFF), which
# makes the whole feature byte-identical to not existing.
# ---------------------------------------------------------------------------
DEFAULTS: Dict[str, Any] = {
    # Master gate. 0 = off. Fire every N OUTER training steps.
    "disc_holdout_probe_every": 0,
    # Encoded root holding rides reserved from training.  None -> fall back to
    # ``holdout_eval_root`` (already used by the 7-chunk eval).
    "disc_holdout_probe_root": None,
    # Hard-fail when the reserved rides are reachable from the training ride
    # list.  Default TRUE *within the feature*: a leaked holdout does not
    # degrade the measurement, it invalidates it.
    "disc_holdout_probe_strict": True,
    # Held-out pool size.
    "disc_holdout_probe_rides": 8,
    "disc_holdout_probe_windows_per_ride": 4,
    # Train-reference pool size (same construction, so the two are
    # apples-to-apples; the ONLY difference is training-supply membership).
    "disc_holdout_probe_train_rides": 8,
    # Build the train reference from rides this rank ACTUALLY consumed at the
    # real-supply funnel (true "D's own training crops").
    #
    # DEFAULT FLIPPED TO FALSE 2026-08-23 (A22 D1).  With True the pool was
    # built from whatever had been observed at the FIRST fire — at step 0 that
    # is normally ONE ride — and then cached for the whole run: 4 windows from
    # 1 ride against 32 windows from 8 held-out rides.  Monte-Carlo with a
    # critic that memorises NOTHING returned MEMORISING 8.5% of the time,
    # INVERTED 15%, and a stable spurious ``real_auc > 0.7`` in 30% of runs.
    # The fallback path (sample the train pool from the full post-filter
    # training ride list, exactly as the held-out pool is sampled) is the
    # statistically correct one: same ride count, same window count, same
    # construction, and the ONLY difference is training-supply membership.
    # When True, pool construction is now DEFERRED until at least
    # ``disc_holdout_probe_train_rides`` rides have been observed.
    "disc_holdout_probe_train_from_observed": False,
    # Crop policy — mirrors TEXTURE_GAN_DESIGN §3.2/§3.6 and A24 (COARSE
    # bands; never exact-y matching).
    "disc_holdout_probe_crops": 8,
    "disc_holdout_probe_frames": 3,
    "disc_holdout_probe_crop_lat": [24, 32],
    "disc_holdout_probe_bands": 3,
    "disc_holdout_probe_border_px": 8,
    # Decode micro-batching (the 92 GB disc-transient lesson: never decode a
    # whole batch at once just because it fits on paper).
    "disc_holdout_probe_decode_batch": 2,
    # |acc - 0.5| below max(band, 2*se) counts as "~ chance".
    "disc_holdout_probe_chance_band": 0.10,
    # Fixed seed -> the probe's test set and crop plan are reproducible and
    # comparable across steps and across arms.
    "disc_holdout_probe_seed": 20260823,
    # Cap on the remembered observed-ride names (leak counting + train ref).
    "disc_holdout_probe_observed_cap": 512,
    # Trainer/model attribute holding the critic, when no explicit score fn was
    # registered.  ``pixel_texture_disc`` is the B2 name (GAN_REDESIGN B2 /
    # TEXTURE_GAN_DESIGN §4, ``model/pixel_texture_disc.py``).
    "disc_holdout_probe_critic_attr": "pixel_texture_disc",
}


def cfg_get(trainer: Any, key: str) -> Any:
    """Read ``key`` off ``trainer.config`` falling back to ``DEFAULTS``.

    Kept for the hot tripwire path (``note_real_supply_ride`` runs per ride
    and must not build a dict).  The probe body uses ``resolve_config``.
    """
    val = getattr(getattr(trainer, "config", None), key, None)
    return DEFAULTS[key] if val is None else val


def resolve_config(trainer: Any) -> Dict[str, Any]:
    """Resolve every ``disc_holdout_probe_*`` knob ONCE, off ``trainer.config``.

    Two jobs, and the second is the whole reason the reads are spelled out one
    string literal at a time instead of looping over ``DEFAULTS``:

    1. one resolved settings dict for the probe body;
    2. **A18 override-guard visibility (A22 D14).**  The trainer's
       ``_warn_ignored_override_keys`` re-derives its allowlist by scanning the
       source for reads of the form ``getattr(config/args/cfg, "<key>", ...)``
       / ``cfg.<key>`` / ``cfg.get("<key>")``.  A key reachable only through
       ``cfg_get(trainer, key)`` — a *variable* key — is invisible to that
       scan, so once ``disc_holdout_probe_`` joins
       ``_OVERRIDE_GUARD_PREFIXES`` every knob except ``_every`` would be
       falsely reported as "merged silently, no effect".  These literal reads
       are what make the guard correct in BOTH directions: a typo'd
       ``disc_holdout_probe_evrey=50`` is caught, and a real knob is not
       falsely accused.  Adding a key to ``DEFAULTS`` without adding it here
       is a bug — ``testing/test_disc_holdout_probe.py`` asserts the two
       agree.
    """
    cfg = getattr(trainer, "config", None)

    def _g(key: str, val: Any) -> Any:
        return DEFAULTS[key] if val is None else val

    return {
        "disc_holdout_probe_every": _g(
            "disc_holdout_probe_every",
            getattr(cfg, "disc_holdout_probe_every", None)),
        "disc_holdout_probe_root": _g(
            "disc_holdout_probe_root",
            getattr(cfg, "disc_holdout_probe_root", None)),
        "disc_holdout_probe_strict": _g(
            "disc_holdout_probe_strict",
            getattr(cfg, "disc_holdout_probe_strict", None)),
        "disc_holdout_probe_rides": _g(
            "disc_holdout_probe_rides",
            getattr(cfg, "disc_holdout_probe_rides", None)),
        "disc_holdout_probe_windows_per_ride": _g(
            "disc_holdout_probe_windows_per_ride",
            getattr(cfg, "disc_holdout_probe_windows_per_ride", None)),
        "disc_holdout_probe_train_rides": _g(
            "disc_holdout_probe_train_rides",
            getattr(cfg, "disc_holdout_probe_train_rides", None)),
        "disc_holdout_probe_train_from_observed": _g(
            "disc_holdout_probe_train_from_observed",
            getattr(cfg, "disc_holdout_probe_train_from_observed", None)),
        "disc_holdout_probe_crops": _g(
            "disc_holdout_probe_crops",
            getattr(cfg, "disc_holdout_probe_crops", None)),
        "disc_holdout_probe_frames": _g(
            "disc_holdout_probe_frames",
            getattr(cfg, "disc_holdout_probe_frames", None)),
        "disc_holdout_probe_crop_lat": _g(
            "disc_holdout_probe_crop_lat",
            getattr(cfg, "disc_holdout_probe_crop_lat", None)),
        "disc_holdout_probe_bands": _g(
            "disc_holdout_probe_bands",
            getattr(cfg, "disc_holdout_probe_bands", None)),
        "disc_holdout_probe_border_px": _g(
            "disc_holdout_probe_border_px",
            getattr(cfg, "disc_holdout_probe_border_px", None)),
        "disc_holdout_probe_decode_batch": _g(
            "disc_holdout_probe_decode_batch",
            getattr(cfg, "disc_holdout_probe_decode_batch", None)),
        "disc_holdout_probe_chance_band": _g(
            "disc_holdout_probe_chance_band",
            getattr(cfg, "disc_holdout_probe_chance_band", None)),
        "disc_holdout_probe_seed": _g(
            "disc_holdout_probe_seed",
            getattr(cfg, "disc_holdout_probe_seed", None)),
        "disc_holdout_probe_observed_cap": _g(
            "disc_holdout_probe_observed_cap",
            getattr(cfg, "disc_holdout_probe_observed_cap", None)),
        "disc_holdout_probe_critic_attr": _g(
            "disc_holdout_probe_critic_attr",
            getattr(cfg, "disc_holdout_probe_critic_attr", None)),
    }


# ---------------------------------------------------------------------------
# Pure statistics (no torch device / no trainer): unit-testable core.
# ---------------------------------------------------------------------------
def roc_auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """P(pos > neg) with ties counted as 0.5 — i.e. ROC-AUC / the pairwise
    "real beats fake" accuracy.  Chance is 0.5.  Exact (O(n1*n2)); the probe's
    n is tens, not thousands."""
    p = pos.reshape(-1).to(torch.float64)
    n = neg.reshape(-1).to(torch.float64)
    if p.numel() == 0 or n.numel() == 0:
        return float("nan")
    diff = p[:, None] - n[None, :]
    return float(
        ((diff > 0).to(torch.float64) + 0.5 * (diff == 0).to(torch.float64))
        .mean()
        .item()
    )


def auc_null_se(n_pos: int, n_neg: int) -> float:
    """Standard error of the AUC estimate **under the null** (no separation):
    ``sqrt((n1 + n2 + 1) / (12 * n1 * n2))`` — the Mann-Whitney U normal
    approximation.  Reported so a reader can tell "0.58 with 24 vs 24" (noise)
    from "0.58 with 400 vs 400" (signal); ``classify`` widens the chance band
    to ``max(band, 2 * se)`` for the same reason.

    ``n_pos``/``n_neg`` MUST be the number of INDEPENDENT units, not the number
    of scores (A22 D4).  The probe decodes ``frames`` pixel frames out of each
    latent crop, so 8 crops become 24 scores; feeding 24 here advertised
    se=0.084 when the truth was 0.142, and ~20% of fires on pure noise then
    returned a non-chance verdict.  The caller passes CROP-level n.
    """
    if n_pos <= 0 or n_neg <= 0:
        return float("nan")
    return float(((n_pos + n_neg + 1) / (12.0 * n_pos * n_neg)) ** 0.5)


def cohens_d(a: torch.Tensor, b: torch.Tensor) -> float:
    """Standardised mean difference ``(mean(a) - mean(b)) / pooled_sd``.  Raw
    logit margins are not comparable across steps once D's scale drifts; this
    is."""
    x = a.reshape(-1).to(torch.float64)
    y = b.reshape(-1).to(torch.float64)
    if x.numel() < 2 or y.numel() < 2:
        return float("nan")
    pooled = float(
        (0.5 * (x.var(unbiased=True) + y.var(unbiased=True))).sqrt().item()
    )
    if not (pooled > 0.0):
        return float("nan")
    return float((x.mean() - y.mean()).item() / pooled)


def classify(
    train_acc: float,
    heldout_acc: float,
    se: float,
    chance_band: float,
    control_auc: Optional[float] = None,
) -> float:
    """The interpretation table at the top of this module, as code.

    ``se`` is ``auc_null_se`` at CROP-level n (A22 D4); the effective
    "separates" threshold is ``0.5 + max(chance_band, 2 * se)`` so a small
    probe batch cannot manufacture a verdict out of sampling noise.

    Orientation guard (A22 D5): ``roc_auc`` assumes higher == more real, but
    nothing in the critic contract pins that sign, and a sign-flipped critic
    that memorises PERFECTLY produces ``train_acc ~ 0``.  Under the one-sided
    test that read as BOTH_CHANCE — or, with an A14 control supplied, as the
    confidently WRONG D_UNDERPOWERED.  Below-chance separation is separation,
    so it gets its own verdict rather than being folded into "chance".
    """
    if any(math.isnan(v) for v in (train_acc, heldout_acc)):
        return VERDICT_UNKNOWN
    margin = max(float(chance_band), 2.0 * (0.0 if math.isnan(se) else se))
    hi = 0.5 + margin
    lo = 0.5 - margin
    if train_acc < lo or heldout_acc < lo:
        return VERDICT_SIGN_FLIPPED
    train_sep = train_acc > hi
    held_sep = heldout_acc > hi
    if train_sep and held_sep:
        return VERDICT_GENERALISING
    if train_sep and not held_sep:
        return VERDICT_MEMORISING
    if held_sep and not train_sep:
        return VERDICT_INVERTED
    if control_auc is None or math.isnan(float(control_auc)):
        return VERDICT_BOTH_CHANCE
    return (
        VERDICT_STUDENT_HARD if float(control_auc) > hi
        else VERDICT_D_UNDERPOWERED
    )


def probe_statistics(
    real_train: torch.Tensor,
    real_heldout: torch.Tensor,
    fake: torch.Tensor,
    *,
    chance_band: float = 0.10,
    control_auc: Optional[float] = None,
    n_eff_train: Optional[int] = None,
    n_eff_heldout: Optional[int] = None,
    n_eff_fake: Optional[int] = None,
) -> Dict[str, float]:
    """Turn three score vectors into the A22 readout.

    The fake vector is deliberately SHARED by both comparisons: train and
    held-out margins then differ only in the real side, which is the whole
    question.  Emitted keys (unprefixed):

    ``train_margin``/``heldout_margin``   mean(real) - mean(fake)
    ``train_acc``/``heldout_acc``         ROC-AUC real vs fake (chance 0.5)
    ``train_d``/``heldout_d``             the same margins, standardised
    ``gen_gap``                           train_margin - heldout_margin
    ``real_gap``                          mean(real_train) - mean(real_heldout)
    ``real_auc``                          AUC(real_train vs real_heldout) —
                                          the SHARPEST memorisation signal:
                                          0.5 means D cannot tell its own
                                          training reals from unseen reals;
                                          > 0.5 means it can.
    ``real_auc_se``                       ``real_auc``'s OWN null SE.  It has
                                          a different n from the two
                                          real-vs-fake AUCs, so it must not
                                          borrow theirs.
    ``real_auc_sep``                      1.0 when ``|real_auc - 0.5|``
                                          clears ``max(band, 2*real_auc_se)``.
    ``pseudo_rep``                        scores per independent unit.  > 1
                                          means the raw counts overstate n.

    **Effective n (A22 D4).**  ``n_eff_*`` are the numbers of INDEPENDENT
    units behind the three score vectors — for this probe, latent CROPS.  The
    probe decodes ``disc_holdout_probe_frames`` pixel frames from each crop,
    so 8 crops arrive here as 24 scores that are pseudo-replicates of 8.
    Computing the null SE from 24 advertised 0.084 when the truth was 0.142,
    and ~20% of fires on PURE NOISE then returned a non-chance verdict (8%
    returned MEMORISATION).  Omit ``n_eff_*`` and the element counts are used,
    which is correct only when every score really is independent.
    """
    rt = real_train.reshape(-1).to(torch.float64)
    rh = real_heldout.reshape(-1).to(torch.float64)
    fk = fake.reshape(-1).to(torch.float64)
    n_rt, n_rh, n_fk = int(rt.numel()), int(rh.numel()), int(fk.numel())

    def _eff(n_eff: Optional[int], n_raw: int) -> int:
        if n_eff is None:
            return n_raw
        return max(0, min(int(n_eff), n_raw))

    e_rt, e_rh, e_fk = (
        _eff(n_eff_train, n_rt), _eff(n_eff_heldout, n_rh), _eff(n_eff_fake, n_fk),
    )
    train_margin = float(rt.mean().item() - fk.mean().item()) if n_rt and n_fk else float("nan")
    heldout_margin = float(rh.mean().item() - fk.mean().item()) if n_rh and n_fk else float("nan")
    train_acc = roc_auc(rt, fk)
    heldout_acc = roc_auc(rh, fk)
    se = auc_null_se(min(e_rt, e_rh), e_fk)
    real_auc = roc_auc(rt, rh)
    real_se = auc_null_se(e_rt, e_rh)
    real_margin = max(float(chance_band), 2.0 * (0.0 if math.isnan(real_se) else real_se))
    verdict = classify(
        train_acc, heldout_acc, se, float(chance_band), control_auc,
    )
    n_units = max(1, min(e_rt, e_rh, e_fk))
    out: Dict[str, float] = {
        "real_train_mean": float(rt.mean().item()) if n_rt else float("nan"),
        "real_heldout_mean": float(rh.mean().item()) if n_rh else float("nan"),
        "fake_mean": float(fk.mean().item()) if n_fk else float("nan"),
        "train_margin": train_margin,
        "heldout_margin": heldout_margin,
        "train_acc": train_acc,
        "heldout_acc": heldout_acc,
        "train_d": cohens_d(rt, fk),
        "heldout_d": cohens_d(rh, fk),
        "gen_gap": train_margin - heldout_margin,
        "real_gap": (
            float(rt.mean().item() - rh.mean().item())
            if n_rt and n_rh else float("nan")
        ),
        "real_auc": real_auc,
        "real_auc_se": real_se,
        "real_auc_sep": (
            float("nan") if math.isnan(real_auc)
            else float(abs(real_auc - 0.5) > real_margin)
        ),
        "auc_se": se,
        "n_real_train": float(n_rt),
        "n_real_heldout": float(n_rh),
        "n_fake": float(n_fk),
        "n_eff_real_train": float(e_rt),
        "n_eff_real_heldout": float(e_rh),
        "n_eff_fake": float(e_fk),
        "pseudo_rep": float(min(n_rt, n_rh, n_fk)) / float(n_units),
        "verdict": verdict,
        "sign_flipped": float(verdict == VERDICT_SIGN_FLIPPED),
    }
    if control_auc is not None:
        out["control_auc"] = float(control_auc)
    return out


# ---------------------------------------------------------------------------
# Holdout integrity
# ---------------------------------------------------------------------------
def ride_key(path: Any) -> str:
    """Ride identity == the zarr BASENAME (``<timestamp>.zarr``).

    Not the full path: the historical leak was the same ride reachable under
    two different roots (``frodobots_encoded_weu`` and its superset
    ``frodobots_encoded_weunz``).  Path identity would have called those two
    different rides; basename identity is what ``holdout_zarr_list`` matches
    on and what makes the collision visible.
    """
    if path is None:
        return ""
    return Path(str(path)).name


def holdout_leak_check(
    train_paths: Sequence[Any],
    holdout_paths: Sequence[Any],
) -> Dict[str, Any]:
    """Static disjointness proof between the discriminator's real supply and
    the reserved rides.

    ``train_paths`` MUST be the *post-filter* training ride list — the list the
    ``DistributedSampler`` indexes and therefore the only source of reals.
    Returns counts plus up to 8 offending names for the log line.
    """
    train = {ride_key(p) for p in train_paths}
    held = {ride_key(p) for p in holdout_paths}
    leaked = sorted(train & held)
    return {
        "n_train": len(train),
        "n_holdout": len(held),
        "n_leak": len(leaked),
        "leaked": leaked[:8],
    }


def note_real_supply_ride(trainer: Any, zarr_path: Any) -> None:
    """Runtime leak tripwire + train-reference collector.

    Called from ``_streaming_setup_sequence_from_ride`` — the ONE funnel that
    turns a dataset ride into ``streaming_state['ride_latents_window']`` (the
    positional real draw and the wide real draw), ``gt_match_latents`` (the
    matched real draw) and ``trainer._ladd_real_ring`` (the cross-ride replay
    real draw).  Every real the discriminator can consume passes here first.

    Cheap: one basename + two set operations, and only when the probe is on.
    """
    try:
        key = ride_key(zarr_path)
        if not key:
            return
        seen = getattr(trainer, "_dhp_observed_rides", None)
        if seen is None:
            seen = []
            trainer._dhp_observed_rides = seen
        cap = int(cfg_get(trainer, "disc_holdout_probe_observed_cap"))
        if key not in seen:
            seen.append(key)
            if len(seen) > cap:
                del seen[: len(seen) - cap]
        held = _holdout_names(trainer)
        if held and key in held:
            trainer._dhp_leak_seen = int(
                getattr(trainer, "_dhp_leak_seen", 0)
            ) + 1
            if int(trainer._dhp_leak_seen) <= 5:
                logging.error(
                    "[dhp] HOLDOUT LEAK: ride %s reached the discriminator "
                    "real-supply funnel (count=%d). The A22 measurement is "
                    "invalid until this is fixed.",
                    key, int(trainer._dhp_leak_seen),
                )
    except Exception:  # pragma: no cover - a tripwire must not kill training
        trainer._dhp_err = int(getattr(trainer, "_dhp_err", 0)) + 1


def register_score_fn(trainer: Any, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
    """B2 integration point.  ``fn`` maps pixels ``[N, 3, H, W]`` in [-1, 1] to
    per-image scalars ``[N]`` or a patch map ``[N, 1, h, w]``."""
    trainer._dhp_score_fn = fn


def register_fake_source(trainer: Any, latents: Optional[torch.Tensor]) -> None:
    """B2 integration point for the FAKE side (A2 / A23).

    Whatever tensor the pixel critic is actually trained against — the flash
    t=60 x0, the ``finish_denoised_chunk`` ladder endpoint, whichever A23
    resolves to — must be the tensor this probe scores, or train and held-out
    margins describe a render nobody trains on.  Stashed for exactly one probe
    fire, then cleared.
    """
    trainer._dhp_fake_latents = latents


# ---------------------------------------------------------------------------
# Held-out / train pools (raw zarr latents; no ss_vae, no captions, no actions)
# ---------------------------------------------------------------------------
def _holdout_root(trainer: Any) -> Optional[str]:
    root = cfg_get(trainer, "disc_holdout_probe_root")
    if not root:
        root = getattr(getattr(trainer, "config", None), "holdout_eval_root", None)
    return str(root) if root else None


def _holdout_paths(trainer: Any) -> List[Path]:
    """Reserved-ride zarr paths, cached on the trainer.

    A plain glob of the reserved root — deliberately NOT a ``ZarrRideDataset``:
    the probe needs latents only (the pixel critic has no action / prompt
    conditioning), so it skips the ss_vae load, the caption lookup and the
    manifest entirely.  Fewer moving parts, and nothing that could hand a
    reserved ride to any other consumer.
    """
    cached = getattr(trainer, "_dhp_holdout_paths", None)
    if cached is not None:
        return cached
    root = _holdout_root(trainer)
    paths: List[Path] = []
    if root:
        paths = sorted(Path(root).glob("*.zarr"))
    trainer._dhp_holdout_paths = paths
    return paths


def _holdout_names(trainer: Any) -> set:
    cached = getattr(trainer, "_dhp_holdout_names", None)
    if cached is not None:
        return cached
    names = {ride_key(p) for p in _holdout_paths(trainer)}
    trainer._dhp_holdout_names = names
    return names


def _train_ride_paths(trainer: Any) -> List[Any]:
    """The post-holdout-filter training ride list (``dataset._rides`` entry 0)."""
    ds = getattr(trainer, "dataset", None)
    rides = getattr(ds, "_rides", None) or []
    return [r[0] for r in rides]


class HoldoutUnverifiableError(RuntimeError):
    """The probe is ON but its integrity check could not actually run.

    Distinct from ``HoldoutLeakError``: that one means we PROVED contamination,
    this one means we proved NOTHING and must not be read as clean.
    """


def _fail_closed(trainer: Any, report: Dict[str, Any]) -> None:
    """Reject the two ways this guard used to report CLEAN while measuring
    nothing.  A gate must fail CLOSED.

    (D2) ``holdout_eval_root`` appears in NO yaml -- it exists only on
    ``--override`` lines -- so an unset or misspelled root yields
    ``_holdout_paths == []``, an empty reserved set, a permanently no-op runtime
    tripwire, and ``leak_rides = leak_seen = ring_delta = 0``: the exact green
    signature.  (D3) an unreachable ``dataset._rides`` gives ``n_train == 0``
    the same way.  Either state means "not measured", never "clean".
    """
    bad = []
    if int(report.get("n_holdout", 0)) <= 0:
        bad.append(
            "no reserved rides resolved (disc_holdout_probe_root / "
            "holdout_eval_root unset, misspelled, or empty) -- the reserved "
            "set is EMPTY, so the runtime tripwire is a permanent no-op and "
            "dhp_leak_* would read 0 while checking nothing")
    if int(report.get("n_train", 0)) <= 0:
        bad.append(
            "training ride list is empty or unreachable (dataset._rides) -- "
            "the static disjointness check compared against nothing")
    report["unverifiable"] = 1.0 if bad else 0.0
    if not bad:
        return
    msg = ("[dhp] HOLDOUT INTEGRITY UNVERIFIABLE -- the probe is ENABLED but "
           "its guard could not run: " + "; ".join(bad) +
           ". Treat every dhp_* number below this line as ABSENT, not clean.")
    if bool(cfg_get(trainer, "disc_holdout_probe_strict")):
        raise HoldoutUnverifiableError(msg)
    logging.error("%s (strict=false -> continuing)", msg)


def run_leak_check(trainer: Any) -> Dict[str, Any]:
    """Static leak check, run once and cached.  Raises when
    ``disc_holdout_probe_strict`` and the reserved rides are reachable.

    D6: the cached path RE-RAISES.  Caching the report before the raise made
    strict mode one-shot -- any enclosing retry or ``except Exception`` turned a
    hard fail into a silent ``dhp_leak_rides=1`` on a curve nobody watches.
    """
    cached = getattr(trainer, "_dhp_leak_report", None)
    if cached is not None:
        # Re-assert on every call, not just the first.
        _fail_closed(trainer, cached)
        if int(cached.get("n_leak", 0)) > 0 and bool(
                cfg_get(trainer, "disc_holdout_probe_strict")):
            raise HoldoutLeakError(cached.get("_msg", "[dhp] HOLDOUT LEAK"))
        return cached
    report = holdout_leak_check(
        _train_ride_paths(trainer), _holdout_paths(trainer),
    )
    trainer._dhp_leak_report = report
    _fail_closed(trainer, report)
    if getattr(trainer, "is_main_process", True):
        logging.info(
            "[dhp] holdout integrity: %d training rides, %d reserved rides, "
            "%d overlap%s",
            report["n_train"], report["n_holdout"], report["n_leak"],
            (" -> " + ", ".join(report["leaked"])) if report["n_leak"] else "",
        )
    if report["n_leak"] > 0:
        msg = (
            f"[dhp] HOLDOUT LEAK: {report['n_leak']} reserved ride(s) are "
            f"present in the training ride list (e.g. {report['leaked']}). "
            f"The A22 generalisation measurement is meaningless until the "
            f"training encoded_root / holdout_zarr_list pair excludes them "
            f"(this is exactly the weu-vs-weunz superset failure recorded in "
            f"docs/GAN_REDESIGN.md A22)."
        )
        report["_msg"] = msg
        if bool(cfg_get(trainer, "disc_holdout_probe_strict")):
            raise HoldoutLeakError(msg)
        logging.error("%s (strict=false -> continuing)", msg)
    return report


def _latent_len(zarr_path: Any) -> int:
    from utils.zarr_dataset import zarr_lib, _LATENT_HEAD_DROP
    g = zarr_lib.open_group(str(zarr_path), mode="r")
    return int(g["latents"].shape[0]) - int(_LATENT_HEAD_DROP)


def build_latent_pool(
    zarr_paths: Sequence[Any],
    *,
    n_rides: int,
    windows_per_ride: int,
    n_frames: int,
    gen: torch.Generator,
) -> Optional[torch.Tensor]:
    """Load ``[n_win, n_frames, C, H, W]`` of raw GT latents onto CPU.

    Fixed once and reused across probe fires: this is a **test set**, and a
    test set that moves cannot show a trend.  (This is not in tension with A21
    — A21 pins the *training* real supply's support; the probe is a
    measurement, and a measurement wants a stationary sample.)
    """
    from utils.zarr_dataset import ZarrRideDataset

    paths = list(zarr_paths)
    if not paths:
        return None
    if len(paths) > n_rides:
        idx = torch.randperm(len(paths), generator=gen)[:n_rides].tolist()
        paths = [paths[i] for i in sorted(idx)]
    windows: List[torch.Tensor] = []
    for p in paths:
        try:
            n_lat = _latent_len(p)
        except Exception as exc:
            logging.warning("[dhp] cannot size %s: %r", p, exc)
            continue
        if n_lat < n_frames:
            continue
        hi = n_lat - n_frames
        for _ in range(max(1, int(windows_per_ride))):
            start = (
                0 if hi <= 0
                else int(torch.randint(0, hi + 1, (1,), generator=gen).item())
            )
            try:
                lat = ZarrRideDataset.load_latent_chunk(
                    str(p), start, start + n_frames,
                )
            except Exception as exc:
                logging.warning("[dhp] load_latent_chunk %s failed: %r", p, exc)
                continue
            if int(lat.shape[0]) == n_frames:
                windows.append(lat)
    if not windows:
        return None
    return torch.stack(windows, dim=0)  # [n_win, F, C, H, W]


# ---------------------------------------------------------------------------
# Crop policy — TEXTURE_GAN_DESIGN §3.2 / §3.6, A24 (COARSE bands only)
# ---------------------------------------------------------------------------
def band_plan(
    n_crops: int,
    n_rows: int,
    crop_rows: int,
    n_bands: int,
    gen: torch.Generator,
) -> Tuple[List[int], List[int]]:
    """Draw ``n_crops`` (band, top-row-offset) pairs.

    A24: vertical position is a nuisance covariate (sky / buildings / road have
    genuinely different texture statistics), so it is matched COARSELY — the
    admissible offset range is split into ``n_bands`` equal bins and the crop's
    centroid is constrained to a bin.  Never exact-y matching, which would
    slowly recreate the matched-data problem.

    The SAME plan is applied to the fake, the train reals and the held-out
    reals, so vertical content can never explain a train-vs-held-out gap.
    """
    max_off = max(0, int(n_rows) - int(crop_rows))
    n_bands = max(1, int(n_bands))
    bands = torch.randint(0, n_bands, (int(n_crops),), generator=gen).tolist()
    offs: List[int] = []
    for b in bands:
        lo = int(round(b * max_off / n_bands))
        hi = int(round((b + 1) * max_off / n_bands))
        hi = max(hi, lo)
        offs.append(
            lo if hi <= lo
            else int(torch.randint(lo, hi + 1, (1,), generator=gen).item())
        )
    return bands, offs


def take_crops(
    source: torch.Tensor,
    offs_y: Sequence[int],
    crop_rows: int,
    crop_cols: int,
    gen: torch.Generator,
) -> torch.Tensor:
    """``source`` ``[N, F, C, H, W]`` -> ``[len(offs_y), F, C, crop_rows, crop_cols]``.

    Horizontal position is NOT matched (dashcam content is not horizontally
    stratified); the source window index is drawn uniformly.
    """
    n, _f, _c, h, w = source.shape
    cr = min(int(crop_rows), int(h))
    cc = min(int(crop_cols), int(w))
    x_hi = max(1, int(w) - cc + 1)
    picks: List[torch.Tensor] = []
    for y in offs_y:
        i = int(torch.randint(0, int(n), (1,), generator=gen).item())
        x = int(torch.randint(0, x_hi, (1,), generator=gen).item())
        y0 = max(0, min(int(y), int(h) - cr))
        picks.append(source[i, :, :, y0:y0 + cr, x:x + cc])
    return torch.stack(picks, dim=0)


def _reduce_scores(raw: torch.Tensor) -> torch.Tensor:
    """Critic output -> one scalar per image.  Accepts ``[N]``, ``[N, 1]`` or a
    patch map ``[N, 1, h, w]`` (mean over the patch grid — the same reduction
    TEXTURE_GAN_DESIGN §5.3 uses for the penalties)."""
    t = raw.detach().to(torch.float32)
    return t if t.dim() == 1 else t.reshape(int(t.shape[0]), -1).mean(dim=1)


def _resolve_score_fn(trainer: Any) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
    fn = getattr(trainer, "_dhp_score_fn", None)
    if callable(fn):
        return fn
    attr = str(cfg_get(trainer, "disc_holdout_probe_critic_attr"))
    critic = getattr(trainer, attr, None)
    if critic is None:
        critic = getattr(getattr(trainer, "model", None), attr, None)
    if critic is None:
        return None
    return lambda px: critic(px)


def _decode_crops(
    trainer: Any,
    crops: torch.Tensor,
    *,
    border: int,
    n_frames: int,
    decode_batch: int,
    gen: torch.Generator,
) -> torch.Tensor:
    """``[n, F_lat, C, ch, cw]`` latents -> ``[n * n_frames, 3, H, W]`` pixels.

    Latent-crop-then-decode (§3.2): decoding a 24x32 latent crop instead of the
    full 60x104 frame cuts decoder activation memory ~8x.  Micro-batched, and
    an 8 px border is trimmed after decode (decoder edge effects).
    """
    outs: List[torch.Tensor] = []
    step = max(1, int(decode_batch))
    for i in range(0, int(crops.shape[0]), step):
        sub = crops[i:i + step]
        pix = trainer._vae_decode_nograd(sub)          # [b, F_pix, 3, H, W]
        if border > 0 and pix.shape[-2] > 2 * border and pix.shape[-1] > 2 * border:
            pix = pix[..., border:-border, border:-border]
        f_pix = int(pix.shape[1])
        k = max(1, min(int(n_frames), f_pix))
        sel = torch.randperm(f_pix, generator=gen)[:k].tolist()
        outs.append(pix[:, sel].reshape(-1, *pix.shape[2:]).to(torch.float32).cpu())
        del pix
    return torch.cat(outs, dim=0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def maybe_run_holdout_probe(
    trainer: Any,
    pred_latents: Optional[torch.Tensor],
    logs: Dict[str, float],
) -> None:
    """A22 probe.  Called once per OUTER training step from the trainer's
    per-chunk path (a ``_dhp_last_step`` latch enforces once-per-step).

    Contract restated because it matters: ``no_grad`` throughout, no writes to
    ``streaming_state``, no push to ``_ladd_real_ring`` (verified by
    ``dhp_ring_delta``), no collective (so a per-rank early return cannot
    desync DDP), every exception swallowed into ``dhp_err``.
    """
    every = int(cfg_get(trainer, "disc_holdout_probe_every") or 0)
    if every <= 0 or pred_latents is None:
        return
    step = int(getattr(trainer, "step", 0))
    if step % every != 0:
        return
    if getattr(trainer, "_dhp_last_step", None) == step:
        return
    trainer._dhp_last_step = step

    t0 = time.time()
    ring_before = len(getattr(trainer, "_ladd_real_ring", None) or [])
    try:
        # 1. Holdout integrity FIRST — a leaked holdout invalidates everything
        #    downstream, and strict mode must fail before any number is logged
        #    that a reader could mistake for a measurement.
        report = run_leak_check(trainer)
        logs[PREFIX + "leak_rides"] = float(report["n_leak"])
        logs[PREFIX + "holdout_rides"] = float(report["n_holdout"])
        logs[PREFIX + "train_rides_total"] = float(report["n_train"])
        logs[PREFIX + "leak_seen"] = float(getattr(trainer, "_dhp_leak_seen", 0))
        if report["n_holdout"] == 0:
            logs[PREFIX + "no_holdout"] = 1.0
            return

        # 2. Critic.  Absent until B2 lands — the leak guards above still ran.
        score_fn = _resolve_score_fn(trainer)
        if score_fn is None:
            logs[PREFIX + "no_critic"] = 1.0
            return

        # A2 / A23: prefer the tensor B2 actually trains its critic on.
        staged = getattr(trainer, "_dhp_fake_latents", None)
        trainer._dhp_fake_latents = None
        logs[PREFIX + "fake_src_default"] = 0.0 if staged is not None else 1.0
        if staged is not None:
            pred_latents = staged.detach()

        n_frames_lat = int(pred_latents.shape[1])
        gen_fixed = torch.Generator(device="cpu").manual_seed(
            int(cfg_get(trainer, "disc_holdout_probe_seed"))
        )

        # 3. The two FIXED reference pools (built once).
        held_pool = getattr(trainer, "_dhp_heldout_pool", None)
        if held_pool is None:
            held_pool = build_latent_pool(
                _holdout_paths(trainer),
                n_rides=int(cfg_get(trainer, "disc_holdout_probe_rides")),
                windows_per_ride=int(
                    cfg_get(trainer, "disc_holdout_probe_windows_per_ride")),
                n_frames=n_frames_lat,
                gen=gen_fixed,
            )
            trainer._dhp_heldout_pool = held_pool
        train_pool = getattr(trainer, "_dhp_train_pool", None)
        if train_pool is None:
            observed = list(getattr(trainer, "_dhp_observed_rides", None) or [])
            from_observed = bool(
                cfg_get(trainer, "disc_holdout_probe_train_from_observed"))
            all_train = _train_ride_paths(trainer)
            train_src: List[Any] = []
            if from_observed and observed:
                by_key = {ride_key(p): p for p in all_train}
                train_src = [by_key[k] for k in observed if k in by_key]
            trainer._dhp_train_src_observed = float(bool(train_src))
            if not train_src:
                train_src = all_train
            train_pool = build_latent_pool(
                train_src,
                n_rides=int(cfg_get(trainer, "disc_holdout_probe_train_rides")),
                windows_per_ride=int(
                    cfg_get(trainer, "disc_holdout_probe_windows_per_ride")),
                n_frames=n_frames_lat,
                gen=gen_fixed,
            )
            trainer._dhp_train_pool = train_pool
        if held_pool is None or train_pool is None:
            logs[PREFIX + "pool_empty"] = 1.0
            return
        logs[PREFIX + "train_src_observed"] = float(
            getattr(trainer, "_dhp_train_src_observed", 0.0))
        logs[PREFIX + "heldout_windows"] = float(held_pool.shape[0])
        logs[PREFIX + "train_windows"] = float(train_pool.shape[0])

        # 4. One band plan shared by all three sides (A24 coarse bands).
        n_crops = int(cfg_get(trainer, "disc_holdout_probe_crops"))
        crop_lat = list(cfg_get(trainer, "disc_holdout_probe_crop_lat"))
        crop_rows, crop_cols = int(crop_lat[0]), int(crop_lat[1])
        gen_step = torch.Generator(device="cpu").manual_seed(
            int(cfg_get(trainer, "disc_holdout_probe_seed")) + 7919 * step
        )
        bands, offs = band_plan(
            n_crops, int(pred_latents.shape[-2]), crop_rows,
            int(cfg_get(trainer, "disc_holdout_probe_bands")), gen_step,
        )
        logs[PREFIX + "band_mismatch"] = 0.0  # one plan, three consumers
        for b in range(int(cfg_get(trainer, "disc_holdout_probe_bands"))):
            logs[PREFIX + f"band{b}_frac"] = (
                bands.count(b) / float(max(1, len(bands)))
            )

        # Crop on CPU, move only the crops (the pools stay resident on CPU).
        # All three sides are cast to the FAKE's dtype before the decode so
        # the decode path is numerically identical for reals and fakes — a
        # bf16-vs-fp32 asymmetry there would be a real texture difference the
        # critic could pick up, which is exactly what must not happen.
        dev, dt = pred_latents.device, pred_latents.dtype
        fake_src = pred_latents[:1].detach()
        crops_fake = take_crops(fake_src, offs, crop_rows, crop_cols, gen_step)
        crops_tr = take_crops(
            train_pool, offs, crop_rows, crop_cols, gen_step,
        ).to(device=dev, dtype=dt)
        crops_ho = take_crops(
            held_pool, offs, crop_rows, crop_cols, gen_step,
        ).to(device=dev, dtype=dt)

        # 5. Decode + score.  Defragment first: the decoder wants a large
        #    contiguous workspace and the rollout leaves fragmented gaps.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        border = int(cfg_get(trainer, "disc_holdout_probe_border_px"))
        n_pix_frames = int(cfg_get(trainer, "disc_holdout_probe_frames"))
        dec_bs = int(cfg_get(trainer, "disc_holdout_probe_decode_batch"))
        with torch.no_grad():
            px_fake = _decode_crops(
                trainer, crops_fake, border=border, n_frames=n_pix_frames,
                decode_batch=dec_bs, gen=gen_step)
            px_tr = _decode_crops(
                trainer, crops_tr, border=border, n_frames=n_pix_frames,
                decode_batch=dec_bs, gen=gen_step)
            px_ho = _decode_crops(
                trainer, crops_ho, border=border, n_frames=n_pix_frames,
                decode_batch=dec_bs, gen=gen_step)
            s_fake = _reduce_scores(score_fn(px_fake.to(dev)))
            s_tr = _reduce_scores(score_fn(px_tr.to(dev)))
            s_ho = _reduce_scores(score_fn(px_ho.to(dev)))

        stats = probe_statistics(
            s_tr.cpu(), s_ho.cpu(), s_fake.cpu(),
            chance_band=float(cfg_get(trainer, "disc_holdout_probe_chance_band")),
            control_auc=getattr(trainer, "_dhp_control_auc", None),
        )
        for k, v in stats.items():
            logs[PREFIX + k] = float(v)
        logs[PREFIX + "fired"] = 1.0
        logs[PREFIX + "secs"] = float(time.time() - t0)
    except HoldoutLeakError:
        raise  # deliberately fatal: a leaked holdout invalidates the readout
    except Exception as exc:  # pragma: no cover - diagnostic only
        logs[PREFIX + "err"] = 1.0
        n_err = int(getattr(trainer, "_dhp_err_logged", 0))
        if getattr(trainer, "is_main_process", True) and n_err < 3:
            trainer._dhp_err_logged = n_err + 1
            logging.warning("[dhp] probe failed at step=%d: %r", step, exc)
    finally:
        ring_after = len(getattr(trainer, "_ladd_real_ring", None) or [])
        logs[PREFIX + "ring_delta"] = float(ring_after - ring_before)
