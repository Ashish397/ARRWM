#!/usr/bin/env python3
"""CPU-only tests for ``analysis/dmd_fp_depth_study.py``.

No GPU, no checkpoint, no repo-heavy import: the study script keeps all
of its maths above the GPU section precisely so this suite can drive it
(``model/dmd_action_forcing.py`` cannot even be imported without a CUDA
device -- ``wan/modules/t5.py`` calls ``torch.cuda.current_device()`` at
class-definition time).

Every test here was mutation-checked: the thing it guards was broken, the
test was confirmed RED, and the break was reverted. The mutation used is
named in each test's docstring so the next person can repeat it.

    OMP_NUM_THREADS=8 python -m pytest testing/test_dmd_fp_depth_study.py -q
"""
import io
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.dmd_fp_depth_study import (  # noqa: E402
    ARM_GT, ARM_ROLL, ARM_VARMATCH, apply_channel_gain, assert_gt_alignment,
    assert_head_drop_convention, auc_below, calibrate, discover_flowrec_arms,
    extrapolate_depth_for_auc,
    fingerprint, flowrec_committed, fp_draw, fp_perturb_from_draw,
    frames_for_depth, gt_alignment_profile, gt_self_difference_profile,
    linear_trend, load_latent_dumps,
    mean_sem, per_channel_std, plan_depths, print_flowrec_report,
    print_report, quantiles, rebin_depth, required_effect_size,
    seed_noise_floor, split_seed_tag, summarize, variance_gain,
    variance_match, verdict,
)
from analysis.dmd_fp_depth_study import (  # noqa: E402
    OBLIVIOUS_FLOOR_MEASURED, apply_record_filters, assert_comparable_records,
    comparability_key, oblivious_floor_analytic, partition_by_comparability,
    print_partitioned_report, print_settings_block,
)

SHAPE = (1, 6, 4, 8, 8)


def _win(seed: int, scale: float = 1.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(SHAPE, generator=g) * scale


def _shrink_denoise(factor: float = 0.9):
    """A stand-in teacher: deterministic, non-degenerate residual."""
    return lambda xp, noise: xp * factor


# =====================================================================
# CONTROL 1 -- identical perturbation draws across arms
# =====================================================================


def test_identical_draws_across_arms():
    """Same window seed => same permutation AND same gaussian noise.

    This is the control that stops the study measuring perturbation noise
    instead of teacher response. The displacement itself is content
    dependent (it shuffles the sample's own high-frequency band), so what
    can be held fixed -- and what is held fixed -- is the DRAW.

    MUTATION (verified RED): in ``fingerprint``, seed the generator with
    ``int(seed) + int(win.abs().sum()) % 7`` so the stream depends on the
    arm's content. Both assertions below fail.
    """
    a, b = _win(1), _win(2) * 3.0 + 1.0
    la, lb = [], []
    kw = dict(mode="hf_scramble", scale=0.15, seeds=3, probe_slice=(3, 6))
    fingerprint(a, _shrink_denoise(), seed=77, draw_log=la, **kw)
    fingerprint(b, _shrink_denoise(), seed=77, draw_log=lb, **kw)
    assert len(la) == len(lb) == 3
    for (pa, na), (pb, nb) in zip(la, lb):
        assert torch.equal(pa, pb), "permutation draw differs across arms"
        assert torch.equal(na, nb), "gaussian noise draw differs across arms"

    lc = []
    fingerprint(b, _shrink_denoise(), seed=78, draw_log=lc, **kw)
    assert not torch.equal(la[0][0], lc[0][0]), (
        "a different seed must give a different draw, else the test above "
        "is vacuous")


def test_draws_are_content_independent_but_delta_is_not():
    """The draw is fixed across arms; delta legitimately is not.

    Guards against someone "fixing" the control by freezing delta itself,
    which would perturb an off-manifold sample with an on-manifold
    sample's high-frequency content and destroy the probe's meaning.
    """
    a, b = _win(1), _win(2)
    g1 = torch.Generator().manual_seed(5)
    g2 = torch.Generator().manual_seed(5)
    da = fp_draw(SHAPE, "hf_scramble", g1, torch.device("cpu"))
    db = fp_draw(SHAPE, "hf_scramble", g2, torch.device("cpu"))
    assert torch.equal(da, db)
    delta_a = fp_perturb_from_draw(a, "hf_scramble", 0.15, da)
    delta_b = fp_perturb_from_draw(b, "hf_scramble", 0.15, db)
    assert not torch.allclose(delta_a, delta_b)
    # scaled to a fixed fraction of each sample's own norm
    for x, d in ((a, delta_a), (b, delta_b)):
        assert float(d.flatten(1).norm(dim=1)) == pytest.approx(
            float(x.flatten(1).norm(dim=1)) * 0.15, rel=1e-4)


@pytest.mark.parametrize("mode", ["hf_scramble", "patch_shuffle", "channel_rot"])
def test_fingerprint_signs(mode):
    """s ~ +1 for a teacher that restores, ~0 for one that does not."""
    x = _win(3)
    probe = (3, 6)

    def restoring(xp, noise, base=x):
        return base.clone()          # perfectly undoes the displacement

    r = fingerprint(x, restoring, seed=11, mode=mode, scale=0.2, seeds=2,
                    probe_slice=probe)
    assert r is not None and r["s"] > 0.99

    calls = [0]

    def blind(xp, noise):
        # A fresh draw per call. A FIXED output would keep a constant
        # -x_p component in every residual, which correlates with delta
        # for the content-derived perturbations and shows up as a
        # spurious non-zero s.
        calls[0] += 1
        g = torch.Generator().manual_seed(4242 + calls[0])
        return torch.randn(xp.shape, generator=g)

    r2 = fingerprint(x, blind, seed=11, mode=mode, scale=0.2, seeds=8,
                     probe_slice=probe)
    assert r2 is not None
    assert r2["s"] < 0.05                    # no restoring component
    assert r2["s"] < r["s"] - 0.8            # and far below a restorer
    assert len(r["s_per_frame"]) == probe[1] - probe[0]


@pytest.mark.parametrize("mode", ["hf_scramble", "patch_shuffle", "channel_rot"])
def test_oblivious_teacher_baseline_is_negative_not_zero(mode):
    """MEASURED PROPERTY, pinned because the design doc gets it wrong.

    ``docs/DMD_FINGERPRINT_PROBE.md`` says an off-manifold sample gives
    ``s -> 0``. It does not. The residual is ``r = x0_hat - x_p``, so a
    teacher whose output is UNCORRELATED with the sample still leaves a
    ``-x_p`` term in r, and every structured perturbation here is built
    by REPLACING part of the sample's own content, which makes delta
    anti-correlated with x. The two combine into a systematically
    NEGATIVE baseline (~-0.4 on gaussian test data at scale 0.2).

    Consequence for the study, and the reason it never reads a raw s as
    an absolute: the "provably off" floor has to be MEASURED (the s_off
    anchor), not assumed to be zero.
    """
    x = _win(3)
    calls = [0]

    def blind(xp, noise):
        calls[0] += 1
        g = torch.Generator().manual_seed(999 + calls[0])
        return torch.randn(xp.shape, generator=g)

    r = fingerprint(x, blind, seed=11, mode=mode, scale=0.2, seeds=8,
                    probe_slice=(3, 6))
    assert r is not None
    assert r["s"] < -0.1, (
        "the oblivious-teacher baseline is expected to sit below zero; if "
        "this has become ~0 the perturbation or the residual definition "
        "changed and the s_off anchor's role needs re-deriving")


# =====================================================================
# CALIBRATION
# =====================================================================


def test_calibration_maths_and_clamping():
    """m = (s - s_off)/(s_gt - s_off), clamped, None on collapse.

    MUTATION (verified RED): drop the ``min(1.0, max(0.0, ...))`` clamp --
    the two clamp assertions fail; return ``0.0`` instead of ``None`` on a
    collapsed anchor pair -- the forgeable-zero assertion fails.
    """
    assert calibrate(0.5, 1.0, 0.0) == pytest.approx(0.5)
    assert calibrate(0.2, 0.6, 0.1) == pytest.approx(0.2)
    assert calibrate(2.0, 1.0, 0.0) == 1.0            # clamped high
    assert calibrate(-3.0, 1.0, 0.0) == 0.0           # clamped low
    assert calibrate(0.5, 1.0, 1.0) is None           # anchors collapsed
    assert calibrate(0.5, 0.5001, 0.5) is None        # gap below floor
    assert calibrate(0.5, 1.0, None) is None
    assert calibrate(None, 1.0, 0.0) is None
    # NO FORGEABLE ZEROS: an unavailable m must not come back as a number
    # that lives inside the metric's own [0, 1] range.
    for bad in (calibrate(0.5, 1.0, 1.0), calibrate(0.5, None, 0.0)):
        assert bad is None and not isinstance(bad, float)


# =====================================================================
# AUC
# =====================================================================


def test_auc_hand_worked():
    """AUC = P(a < b) + 0.5 P(tie), checked against enumerated pairs.

    a=[1,2] vs b=[2,3]: (1<2)=1, (1<3)=1, (2==2)=0.5, (2<3)=1 -> 3.5/4.

    MUTATION (verified RED): count ties as a full win (``x <= y``) -- the
    0.875 case becomes 1.0; drop the tie term entirely -- it becomes 0.75.
    """
    assert auc_below([1.0, 2.0], [2.0, 3.0]) == pytest.approx(0.875)
    assert auc_below([0.0], [1.0]) == 1.0
    assert auc_below([1.0], [0.0]) == 0.0
    assert auc_below([1.0, 1.0], [1.0, 1.0]) == 0.5   # identical pools
    assert auc_below([], [1.0]) is None
    assert auc_below([1.0], []) is None


def test_mean_sem_and_quantiles():
    mu, sem = mean_sem([1.0, 2.0, 3.0])
    assert mu == pytest.approx(2.0)
    assert sem == pytest.approx(1.0 / (3 ** 0.5))
    assert mean_sem([]) == (None, None)
    assert mean_sem([5.0]) == (5.0, None)
    assert quantiles([0.0, 1.0], (0.0, 0.5, 1.0)) == pytest.approx([0.0, 0.5, 1.0])
    assert quantiles([], (0.5,)) is None


# =====================================================================
# THE MOST IMPORTANT TEST -- the study can say NO
# =====================================================================


def _records(gt, roll, var, depths=(2, 6, 10), n=24, jitter=0.0):
    """Synthetic records: per-depth means for each arm, optional noise."""
    g = torch.Generator().manual_seed(9)
    recs = []
    for w in range(n):
        for i, d in enumerate(depths):
            for arm, series in ((ARM_GT, gt), (ARM_ROLL, roll),
                                (ARM_VARMATCH, var)):
                v = series[i]
                if jitter:
                    v = v + float(torch.randn(1, generator=g)) * jitter
                recs.append({
                    "window": w, "arm": arm, "depth": d, "s": v,
                    "s_per_frame": [v, v, v],
                    "true_drift": 0.1 * (i + 1),
                })
    return recs


def test_verdict_says_no_on_flat_noisy_input():
    """A probe that does not move with depth must be reported as a NULL.

    This is the test that proves the study can return NO. Flat means +
    heavy jitter = exactly the shape a useless probe produces, and the
    verdict must refuse it on BOTH separation and degradation.

    MUTATION (verified RED): make ``verdict`` return
    ``{"passed": True, "failures": []}`` unconditionally, or relax the
    monotonicity/degradation check to ``True`` -- the assertions below and
    the printed-banner assertion both fail.
    """
    recs = _records([0.30] * 3, [0.30] * 3, [0.30] * 3, jitter=0.25)
    v = verdict(summarize(recs, metric="s"), auc_min=0.8, mono_tol=0.01)
    assert v["passed"] is False
    joined = " ".join(v["failures"])
    assert "separation" in joined
    assert "degradation" in joined
    assert v["auc_roll_deepest"] < 0.8

    buf = io.StringIO()
    print_report(recs, achieved_depths=[2, 6, 10], dropped_depths=[],
                 auc_min=0.8, mono_tol=0.01,
                 verdict_metric="s", out_fh=buf)
    text = buf.getvalue()
    assert "PROBE CANNOT RANK OFF-MANIFOLD DISTANCE -- GATE NOT VIABLE" in text
    assert "GATE VIABLE" not in text.replace("NOT VIABLE", "")


def test_verdict_says_no_when_mean_moves_but_distributions_overlap():
    """A clean monotone MEAN with overlap is still a NO.

    The gate acts per sample; a mean trend it cannot act on is not a
    result. This pins that AUC, not the trend, is the decision variable.
    """
    recs = _records([0.60, 0.60, 0.60], [0.55, 0.50, 0.45],
                    [0.60, 0.60, 0.60], jitter=0.6)
    s = summarize(recs, metric="s")
    v = verdict(s, auc_min=0.8, mono_tol=0.01)
    assert v["passed"] is False
    assert any("separation" in f for f in v["failures"])


def test_verdict_flags_variance_confound_as_falsification():
    """varmatch-GT dropping as far as the rollout is a FALSIFICATION.

    On-manifold GT content carrying depth-D's variance signature must not
    reproduce the effect. If it does, the fingerprint is a smoothness
    detector and the gate is not viable even though separation and
    monotonicity both pass.

    MUTATION (verified RED): delete the ``auc_varmatch`` branch from
    ``verdict`` (or never emit ``ARM_VARMATCH`` records) -- ``passed``
    flips to True and the confound banner disappears.
    """
    recs = _records([0.90, 0.90, 0.90], [0.60, 0.35, 0.10],
                    [0.60, 0.35, 0.10], jitter=0.02)
    s = summarize(recs, metric="s")
    v = verdict(s, auc_min=0.8, mono_tol=0.01)
    assert v["auc_roll_deepest"] > 0.8, "separation itself should pass here"
    assert v["confound_failed"] is True
    assert v["passed"] is False
    assert any("CONFOUND" in f for f in v["failures"])
    # the SHARE of the drop the control reproduces is the headline number
    deep = s["rows"][-1]
    assert deep["var_share_of_drop"] == pytest.approx(1.0, abs=0.15)

    buf = io.StringIO()
    print_report(recs, achieved_depths=[2, 6, 10], dropped_depths=[],
                 auc_min=0.8, mono_tol=0.01,
                 verdict_metric="s", out_fh=buf)
    assert ("PROBE READS VARIANCE CONTRACTION, NOT OFF-MANIFOLD DISTANCE "
            "-- GATE NOT VIABLE") in buf.getvalue()


def test_verdict_passes_only_when_everything_holds():
    """The positive control: separation + monotone + control survives."""
    recs = _records([0.90, 0.90, 0.90], [0.60, 0.35, 0.10],
                    [0.89, 0.88, 0.87], jitter=0.02)
    v = verdict(summarize(recs, metric="s"), auc_min=0.8, mono_tol=0.01)
    assert v["passed"] is True, v["failures"]

    buf = io.StringIO()
    print_report(recs, achieved_depths=[2, 6, 10], dropped_depths=[],
                 auc_min=0.8, mono_tol=0.01,
                 verdict_metric="s", out_fh=buf)
    assert "GATE VIABLE" in buf.getvalue()
    assert "GATE NOT VIABLE" not in buf.getvalue()


def test_verdict_says_no_when_confound_untested():
    """No control arm at all => UNTESTED, and that is a NO, not a pass."""
    recs = [r for r in _records([0.9] * 3, [0.6, 0.35, 0.1], [0.0] * 3,
                                jitter=0.02) if r["arm"] != ARM_VARMATCH]
    v = verdict(summarize(recs, metric="s"), auc_min=0.8, mono_tol=0.01)
    assert v["passed"] is False
    assert v["confound_checked"] is False
    assert any("UNTESTED" in f for f in v["failures"])


# =====================================================================
# NO LEAKAGE
# =====================================================================


def test_true_drift_never_influences_s_or_m():
    """The independent yardstick must be inert in the scoring path.

    MUTATION (verified RED): add a drift-keyed filter or weight to
    ``summarize`` (e.g. ``if r.get("true_drift", 0) > 0.15: continue``) --
    the two summaries stop matching and this fails.
    """
    recs = _records([0.9, 0.9, 0.9], [0.6, 0.35, 0.1], [0.85, 0.8, 0.75],
                    jitter=0.05)
    base = summarize(recs, metric="s")
    mutated = []
    for r in recs:
        r2 = dict(r)
        if "true_drift" in r2:
            r2["true_drift"] = r2["true_drift"] * 137.0 + 5.0
            r2["true_drift_rel"] = 0.0
            r2["var_ratio"] = 1e6
        mutated.append(r2)
    assert summarize(mutated, metric="s") == base
    assert verdict(summarize(mutated, metric="s")) == verdict(base)


def test_summarize_rejects_unknown_metric():
    with pytest.raises(ValueError):
        summarize([], metric="true_drift")


# =====================================================================
# NO SILENT CAPS -- depth availability
# =====================================================================


def test_frames_for_depth():
    assert frames_for_depth(2, teacher_frames=21) == 21 + 9
    assert frames_for_depth(18, teacher_frames=21) == 21 + 57


def test_plan_depths_drops_unreachable_depths_loudly():
    """Depths the data cannot host are DROPPED and REPORTED, not truncated.

    MUTATION (verified RED): make ``plan_depths`` return
    ``(sorted(requested), [])`` -- the dropped-list assertions fail, and a
    sweep that never reached depth 18 would print as if it had.
    """
    # 30 rides of 90 frames, 5 of 200. depth 18 needs 78 frames -> 35 rides
    # qualify; depth 30 needs 114 -> only 5 do.
    lengths = [90] * 30 + [200] * 5
    kept, dropped = plan_depths([2, 6, 18, 30], lengths, 20, teacher_frames=21)
    assert kept == [2, 6, 18]
    assert [d for d, _ in dropped] == [30]
    assert "needs 114 latent frames/ride" in dropped[0][1]
    assert "only 5 of 35" in dropped[0][1]

    kept2, dropped2 = plan_depths([0, 2], lengths, 20, teacher_frames=21)
    assert kept2 == [2]
    assert dropped2[0][0] == 0


def test_achieved_depths_reach_the_report():
    buf = io.StringIO()
    print_report(_records([0.9] * 2, [0.6, 0.3], [0.85, 0.8], depths=(2, 6)),
                 achieved_depths=[2, 6],
                 dropped_depths=[(18, "needs 78 latent frames/ride; only 3 "
                                      "of 400 candidate rides are that long")],
                 auc_min=0.8, mono_tol=0.01,
                 verdict_metric="s", out_fh=buf)
    txt = buf.getvalue()
    assert "ACHIEVED DEPTHS: [2, 6]" in txt
    assert "DROPPED depth 18" in txt
    assert "needs 78 latent frames/ride" in txt


# =====================================================================
# VARIANCE-MATCHING MECHANICS
# =====================================================================


def test_variance_match_transplants_only_the_second_moment():
    """Per-channel std moves onto the reference; per-channel mean does not.

    MUTATION (verified RED): scale about zero instead of about the mean in
    ``apply_channel_gain`` -- the mean assertion fails.
    """
    src = _win(10) * 2.0 + 0.7
    ref = _win(11) * 0.5
    out = variance_match(src, ref)
    assert torch.allclose(per_channel_std(out), per_channel_std(ref), atol=1e-4)
    assert torch.allclose(out.float().mean(dim=(0, 1, 3, 4)),
                          src.float().mean(dim=(0, 1, 3, 4)), atol=1e-5)
    g = variance_gain(src, ref)
    assert torch.allclose(apply_channel_gain(src, g), out, atol=1e-6)
    # identity when the reference IS the source
    assert torch.allclose(variance_match(src, src), src, atol=1e-5)


def test_variance_ratio_matches_the_contraction_law_direction():
    """A contracted 'rollout' yields gain < 1 -- the law's direction."""
    gt = _win(12)
    contracted = gt * 0.917
    g = variance_gain(gt, contracted)
    assert float(g.mean()) == pytest.approx(0.917, rel=1e-3)


# =====================================================================
# REPORT PLUMBING
# =====================================================================


def test_report_omits_rather_than_forges_missing_values():
    """A missing mean prints as n/a, never as 0.0000."""
    recs = [{"window": 0, "arm": ARM_ROLL, "depth": 4, "s": 0.2},
            {"window": 1, "arm": ARM_ROLL, "depth": 4, "s": 0.3}]
    s = summarize(recs, metric="s")
    row = s["rows"][0]
    assert "mean_gt" not in row and "auc_roll" not in row
    buf = io.StringIO()
    print_report(recs, achieved_depths=[4], dropped_depths=[], auc_min=0.8,
                 mono_tol=0.01, verdict_metric="s",
                 out_fh=buf)
    assert "n/a" in buf.getvalue()


# =====================================================================
# DEPTH-AWARE VERDICT -- a short lever can neither ship nor falsify
# =====================================================================


def test_shallow_depth_cannot_print_a_confident_no():
    """Depths 1-4 with a flat signal => INCONCLUSIVE, not a falsification.

    The thesis is about drift COMPOUNDING over depth. A null at depth <= 4
    says the effect is undetectable there, which is a different claim from
    "the probe cannot rank off-manifold distance".

    MUTATION (verified RED): drop the ``scope``/``deep_depth_min`` branch
    from ``verdict`` and ``print_report`` -- the shallow run then prints
    the confident NOT-VIABLE banner and both assertions fail.
    """
    recs = _records([0.3] * 4, [0.3] * 4, [0.3] * 4, depths=(1, 2, 3, 4),
                    jitter=0.25)
    v = verdict(summarize(recs, metric="s"), auc_min=0.8, deep_depth_min=10)
    assert v["scope"] == "shallow"
    assert v["inconclusive"] is True
    assert v["passed"] is False
    assert v["trend_present"] is False

    buf = io.StringIO()
    print_report(recs, achieved_depths=[1, 2, 3, 4], dropped_depths=[],
                 auc_min=0.8, mono_tol=0.01,
                 deep_depth_min=10, verdict_metric="s", out_fh=buf)
    txt = buf.getvalue()
    assert "INCONCLUSIVE AT SHALLOW DEPTH" in txt
    assert "no trend at shallow depth, INCONCLUSIVE" in txt
    assert "PROBE CANNOT RANK OFF-MANIFOLD DISTANCE" not in txt
    assert "GATE VIABLE" not in txt
    assert "SHALLOW-DEPTH RUN" in txt


def test_shallow_depth_cannot_print_a_confident_yes_either():
    """A clean shallow trend is 'needs depth', never a green light."""
    recs = _records([0.90] * 4, [0.80, 0.70, 0.60, 0.50], [0.89] * 4,
                    depths=(1, 2, 3, 4), jitter=0.02)
    v = verdict(summarize(recs, metric="s"), auc_min=0.8, deep_depth_min=10)
    assert v["scope"] == "shallow" and v["passed"] is False
    assert v["trend_present"] is True
    buf = io.StringIO()
    print_report(recs, achieved_depths=[1, 2, 3, 4], dropped_depths=[],
                 auc_min=0.8, mono_tol=0.01,
                 deep_depth_min=10, verdict_metric="s", out_fh=buf)
    txt = buf.getvalue()
    assert "trend present, consistent with the thesis, NEEDS DEPTH" in txt
    assert "GATE VIABLE" not in txt
    assert "PROBE CANNOT RANK OFF-MANIFOLD DISTANCE" not in txt


def test_shallow_run_still_shouts_the_variance_confound():
    """The confound falsification is scope-independent: more depth cannot
    turn a smoothness detector into a manifold detector."""
    recs = _records([0.90] * 4, [0.80, 0.70, 0.60, 0.50],
                    [0.80, 0.70, 0.60, 0.50], depths=(1, 2, 3, 4), jitter=0.02)
    buf = io.StringIO()
    print_report(recs, achieved_depths=[1, 2, 3, 4], dropped_depths=[],
                 auc_min=0.8, mono_tol=0.01,
                 deep_depth_min=10, verdict_metric="s", out_fh=buf)
    assert "PROBE READS VARIANCE CONTRACTION" in buf.getvalue()


def test_confound_is_undecided_when_the_drop_is_inside_its_own_noise():
    """``var_share_of_drop`` is a ratio of differences -- guard the divisor.

    With a drop buried in noise the share is unstable and can read 1.9 or
    -4 from the same underlying null. Declaring a falsification off that
    would be over-reading a small sample, which this campaign has already
    had to retract twice. The guard turns it into an explicit UNDECIDED.

    MUTATION (verified RED): drop the ``drop_resolvable`` guard from
    ``verdict`` (compute ``bad`` unconditionally) -- ``confound_failed``
    flips to True on pure noise and the banner assertion fails.
    """
    recs = _records([0.30] * 3, [0.30] * 3, [0.10] * 3, jitter=0.30)
    v = verdict(summarize(recs, metric="s"), auc_min=0.8, mono_tol=0.01)
    assert v["drop_resolvable"] is False
    assert v["confound_failed"] is False
    assert any("CONFOUND UNDECIDED" in f for f in v["failures"])
    buf = io.StringIO()
    print_report(recs, achieved_depths=[2, 6, 10], dropped_depths=[],
                 auc_min=0.8, mono_tol=0.01,
                 verdict_metric="s", out_fh=buf)
    assert "PROBE READS VARIANCE CONTRACTION" not in buf.getvalue()

    # ...and a resolvable drop with a reproducing control IS decided.
    recs2 = _records([0.90] * 3, [0.60, 0.35, 0.10], [0.60, 0.35, 0.10],
                     jitter=0.02)
    v2 = verdict(summarize(recs2, metric="s"), auc_min=0.8, mono_tol=0.01)
    assert v2["drop_resolvable"] is True and v2["confound_failed"] is True


def test_effect_size_and_extrapolation():
    """Slope per unit depth, and the depth a usable AUC would need.

    MUTATION (verified RED): return the slope with the wrong sign from
    ``linear_trend`` -- the extrapolation goes negative/None and the depth
    assertion fails.
    """
    assert linear_trend([1.0, 2.0, 3.0], [1.0, 0.0, -1.0])[0] == pytest.approx(-1.0)
    assert linear_trend([1.0], [1.0]) is None
    assert linear_trend([2.0, 2.0], [1.0, 3.0]) is None
    assert required_effect_size(0.8) == pytest.approx(1.1902, abs=1e-3)
    assert required_effect_size(0.42) is None

    # GT 1.0; means falling 0.05/depth from 0.95; sd 0.1 -> need a gap of
    # 1.1902*0.1 = 0.11902 -> mean 0.88098 -> depth (0.88098-1.0)/-0.05 = 2.38
    got = extrapolate_depth_for_auc(
        [1.0, 2.0, 3.0], [0.95, 0.90, 0.85], gt_mean=1.0, pooled_sd=0.1,
        auc_target=0.8)
    assert got == pytest.approx(2.3804, abs=1e-3)
    # a rising trend never reaches the target
    assert extrapolate_depth_for_auc(
        [1.0, 2.0], [0.5, 0.6], gt_mean=1.0, pooled_sd=0.1,
        auc_target=0.8) is None

    recs = _records([0.90] * 4, [0.80, 0.70, 0.60, 0.50], [0.89] * 4,
                    depths=(1, 2, 3, 4), jitter=0.02)
    v = verdict(summarize(recs, metric="s"), auc_min=0.8, deep_depth_min=10)
    assert v["slope_per_depth"] == pytest.approx(-0.1, abs=0.01)
    assert v["extrapolated_depth_for_auc"] > 0


def test_deep_scope_restores_the_confident_verdicts():
    """The shallow guard must not suppress a genuine deep result."""
    recs = _records([0.90] * 3, [0.60, 0.35, 0.10], [0.89, 0.88, 0.87],
                    depths=(4, 10, 18), jitter=0.02)
    v = verdict(summarize(recs, metric="s"), auc_min=0.8, deep_depth_min=10)
    assert v["scope"] == "deep"
    assert v["passed"] is True
    assert "inconclusive" not in v


# =====================================================================
# OFFLINE ENTRY -- pre-dumped rollout latents
# =====================================================================


def test_load_latent_dumps_groups_by_roll_and_trims_to_window(tmp_path):
    """Depth grouping + fixed window length + loud drop of short records.

    The trainer dumps 18 frames at roll 1 and 9 at deeper rolls. A teacher
    window whose LENGTH tracked depth would be a sequence-length confound
    wearing the depth effect's clothes, so every record is trimmed to the
    same tail.

    MUTATION (verified RED): return ``lat`` untrimmed -- the shape
    assertion fails; drop the short-record filter -- the note count and the
    per-depth counts both change.
    """
    recs = (
        [{"step": 200, "ride": 0, "roll": 1,
          "lat": torch.randn(1, 18, 4, 6, 6).half()} for _ in range(3)]
        + [{"step": 200, "ride": 0, "roll": 2,
            "lat": torch.randn(1, 9, 4, 6, 6).half()} for _ in range(3)]
        + [{"step": 200, "ride": 0, "roll": 3,
            "lat": torch.randn(1, 4, 4, 6, 6).half()}]      # too short
    )
    torch.save(recs, tmp_path / "latdump_rank0.pt")
    by_depth, notes = load_latent_dumps(str(tmp_path), window_frames=9)
    assert sorted(by_depth) == [1, 2]
    assert len(by_depth[1]) == 3 and len(by_depth[2]) == 3
    for d in by_depth:
        for t in by_depth[d]:
            assert t.shape[1] == 9
            assert t.dtype == torch.float32
    assert "1 dropped as shorter" in notes[0]
    assert "7 dumped records" in notes[0]

    with pytest.raises(SystemExit):
        load_latent_dumps(str(tmp_path / "nope"), window_frames=9)


def test_offline_records_are_unpaired_so_no_paired_stat_is_emitted():
    """Without a ``window`` key there is no paired win-rate -- by design.

    The offline dumps carry no GT counterpart. Emitting a paired statistic
    from an unpaired design would be a number that does not mean what its
    name says, which is the silent-failure shape this campaign keeps
    hitting.
    """
    recs = []
    for i in range(12):
        for arm, v in ((ARM_GT, 0.5), (ARM_ROLL, 0.4), (ARM_VARMATCH, 0.49)):
            recs.append({"group": i, "arm": arm, "depth": 2, "s": v,
                         "mode": "offline"})
    row = summarize(recs, metric="s")["rows"][0]
    assert "share_below_gt" not in row
    assert "paired_gt_minus_roll" not in row
    assert row["auc_roll"] == 1.0            # AUC still valid, unpaired
    assert "m" not in row and row.get("mean_gt") == pytest.approx(0.5)

    # The report must say m was NOT COMPUTED, not that the anchors
    # collapsed -- that would be a false claim about why the key is
    # missing, which is the same shape of silent wrongness as a forged
    # value.
    #
    # MUTATION (verified RED): make the calibration branch unconditional
    # (``elif False:``) so the collapsed-anchor sentence always prints --
    # the last two assertions fail.
    buf = io.StringIO()
    print_report(recs, achieved_depths=[2], dropped_depths=[], auc_min=0.8,
                 mono_tol=0.01, verdict_metric="s",
                 mode="offline", out_fh=buf)
    txt = buf.getvalue()
    assert "m NOT COMPUTED in this mode" in txt
    assert "collapsed anchors" not in txt
    # ...and the drift yardstick declares itself unavailable rather than
    # printing a table of zeros.
    assert "UNAVAILABLE in this mode" in txt
    assert "MODE: offline" in txt


# =====================================================================
# THE OBLIVIOUS-TEACHER FLOOR MOVES WITH THE PROBE SETTING
# =====================================================================


def test_oblivious_floor_is_mode_and_scale_dependent():
    """The floor differs BY MODE at fixed scale, and BY SCALE at fixed mode.

    Measured 2026-08-24 against the shipped ``_dmd_fp_perturb`` with a
    literal ``x0_hat = 0`` teacher. This is why raw ``s`` is ordinal only
    and may never be pooled across settings: at scale 0.15 an oblivious
    teacher scores -0.490 under ``hf_scramble`` but -0.610 under
    ``channel_rot``, so a 0.12 "difference" is available for free from the
    knob alone.
    """
    at15 = {m: OBLIVIOUS_FLOOR_MEASURED[(m, 0.15)]
            for m in ("hf_scramble", "patch_shuffle", "channel_rot")}
    assert len(set(at15.values())) == 3, "floors must differ by mode"
    # hf_scramble preserves the low frequencies, so it displaces less of x
    # and floors ABOVE the other two at every scale.
    for sc in (0.05, 0.15, 0.30):
        assert (OBLIVIOUS_FLOOR_MEASURED[("hf_scramble", sc)]
                > OBLIVIOUS_FLOOR_MEASURED[("patch_shuffle", sc)]
                > OBLIVIOUS_FLOOR_MEASURED[("channel_rot", sc)])
    # ...and rises with scale within every mode.
    for m in ("hf_scramble", "patch_shuffle", "channel_rot"):
        f = [OBLIVIOUS_FLOOR_MEASURED[(m, sc)] for sc in (0.05, 0.15, 0.30)]
        assert f[0] < f[1] < f[2], m
        assert all(x < 0 for x in f), "the floor is NEGATIVE, not zero"


def test_oblivious_floor_analytic_brackets_the_measurements():
    """s_floor(k) = (2k-1)/(sqrt(2)*sqrt((1-k)^2+k^2)), a REFERENCE.

    MUTATION (verified RED): drop the ``sqrt(2)`` (or use ``2k+1``) -- the
    bracketing assertions fail.
    """
    assert oblivious_floor_analytic(0.05) == pytest.approx(-0.669, abs=2e-3)
    assert oblivious_floor_analytic(0.15) == pytest.approx(-0.573, abs=2e-3)
    assert oblivious_floor_analytic(0.30) == pytest.approx(-0.371, abs=2e-3)
    assert oblivious_floor_analytic(0.5) == pytest.approx(0.0, abs=1e-9)
    # hf_scramble sits ABOVE the reference at every scale; channel_rot below
    for sc in (0.05, 0.15, 0.30):
        ref = oblivious_floor_analytic(sc)
        assert OBLIVIOUS_FLOOR_MEASURED[("hf_scramble", sc)] > ref
        assert OBLIVIOUS_FLOOR_MEASURED[("channel_rot", sc)] < ref


def test_refuses_to_pool_records_from_different_probe_settings():
    """A mixed pool's mean/quantiles/AUC would be arithmetic over two
    different floors -- and every one of them would still print.

    MUTATION (verified RED): make ``assert_comparable_records`` a no-op --
    ``print_report`` happily pools two settings and the raises fail.
    """
    def rec(mode, scale, t, s):
        return {"arm": ARM_ROLL, "depth": 4, "s": s, "perturb": mode,
                "perturb_scale": scale, "probe_timestep": t}

    same = [rec("hf_scramble", 0.15, 500.0, 0.1),
            rec("hf_scramble", 0.15, 500.0, 0.2)]
    assert_comparable_records(same)                       # no raise
    assert len(partition_by_comparability(same)) == 1

    for bad in (rec("channel_rot", 0.15, 500.0, 0.1),     # mode differs
                rec("hf_scramble", 0.30, 500.0, 0.1),     # scale differs
                rec("hf_scramble", 0.15, 250.0, 0.1)):    # timestep differs
        mixed = same + [bad]
        assert len(partition_by_comparability(mixed)) == 2
        with pytest.raises(SystemExit, match="REFUSING TO POOL"):
            assert_comparable_records(mixed)
        with pytest.raises(SystemExit, match="REFUSING TO POOL"):
            print_report(mixed, achieved_depths=[4], dropped_depths=[],
                         auc_min=0.8, mono_tol=0.01, verdict_metric="s",
                         out_fh=io.StringIO())

    assert comparability_key(same[0]) == ("hf_scramble", 0.15, 500.0)


def test_record_filters_make_a_mixed_dump_usable():
    """A mixed dump is not unusable -- it is several studies in one file."""
    recs = [{"arm": ARM_ROLL, "s": 0.1, "perturb": "hf_scramble",
             "perturb_scale": 0.15, "probe_timestep": t}
            for t in (250.0, 500.0)]
    assert len(apply_record_filters(recs, timestep=500.0)) == 1
    assert len(apply_record_filters(recs, perturb="channel_rot")) == 0
    assert len(apply_record_filters(recs)) == 2
    assert_comparable_records(apply_record_filters(recs, timestep=250.0))


def test_settings_block_states_the_floor_and_the_unswept_timestep():
    """The reader of the OUTPUT must see both, not just the reader of a
    report written alongside it.

    MUTATION (verified RED): delete the ``print_settings_block`` call from
    ``print_report`` -- every assertion here fails and the biggest unswept
    design choice becomes invisible to whoever reads the numbers.
    """
    recs = [{"arm": ARM_ROLL, "depth": 4, "s": 0.1, "perturb": "hf_scramble",
             "perturb_scale": 0.15, "probe_timestep": 500.0}]
    buf = io.StringIO()
    print_report(recs, achieved_depths=[4], dropped_depths=[], auc_min=0.8,
                 mono_tol=0.01, verdict_metric="s", out_fh=buf)
    txt = buf.getvalue()
    assert "PROBE SETTINGS" in txt
    assert "measured -0.490" in txt          # the mode+scale specific floor
    assert "does NOT score 0" in txt
    assert "PROBE TIMESTEP: THE BIGGEST UNSWEPT DESIGN CHOICE" in txt
    assert "CHOSEN, NOT MEASURED" in txt
    assert "--probe-timestep 250 500 750" in txt

    # a swept run must NOT print the single-value warning in every section
    buf2 = io.StringIO()
    print_settings_block(recs, out_fh=buf2,
                         sweep_keys=[("hf_scramble", 0.15, 250.0),
                                     ("hf_scramble", 0.15, 500.0)])
    t2 = buf2.getvalue()
    assert "SWEPT in this run: [250.0, 500.0]" in t2
    assert "CHOSEN, NOT MEASURED" not in t2


def test_partitioned_report_sweeps_and_flags_a_verdict_that_flips():
    """A verdict that changes with the timestep is a verdict about the knob.

    MUTATION (verified RED): have ``print_partitioned_report`` call the
    report once on all records instead of per partition -- the
    'SETTING 1/2' headers and the disagreement banner both vanish (and the
    pooling guard fires instead).
    """
    class _A:
        auc_min, mono_tol = 0.8, 0.01
        confound_share_max, deep_depth_min = 0.5, 10
        depth_bands = [0, 3, 10, 20, 30, 40]

    recs = []
    for t, eff in ((250.0, 0.0), (750.0, 0.02)):
        for d in range(40):
            for a, v in ((ARM_GT, 0.9), (ARM_ROLL, 0.9 - eff * d),
                         (ARM_VARMATCH, 0.9 - 0.0002 * d)):
                recs.append({"study_arm": "A", "base_arm": "A",
                             "seed_tag": "s1234", "arm": a, "depth": d,
                             "s": v, "mode": "flowrec",
                             "perturb": "hf_scramble", "perturb_scale": 0.15,
                             "probe_timestep": t})
    buf = io.StringIO()
    res = print_partitioned_report(
        recs, _A(), kind="flowrec", achieved_depths=list(range(40)),
        arm_names=["A"], out_fh=buf)
    txt = buf.getvalue()
    assert len(res) == 2
    assert "SETTING 1/2" in txt and "SETTING 2/2" in txt
    assert "CROSS-SETTING CONSISTENCY" in txt
    assert "VERDICT DEPENDS ON THE PROBE SETTING" in txt
    assert {r["setting"][2] for r in res} == {250.0, 750.0}
    # and when they agree, it says so instead
    agree = [r for r in recs if r["probe_timestep"] == 750.0]
    buf2 = io.StringIO()
    print_partitioned_report(
        agree + [dict(r, probe_timestep=500.0) for r in agree], _A(),
        kind="flowrec", achieved_depths=list(range(40)), arm_names=["A"],
        out_fh=buf2)
    assert "All settings agree" in buf2.getvalue()
    assert "VERDICT DEPENDS" not in buf2.getvalue()


# =====================================================================
# FLOWREC ENTRY -- recorded rollouts, depth 0..39
# =====================================================================


def test_flowrec_committed_picks_the_last_rungs_x0():
    """The commit is the FINAL rung's x0 -- not the initial noise, not an
    intermediate rung, not a re-noised input.

    ``sdt`` rows are ``(step_idx, rung, t)``. Rung -1 is the block's
    initial NOISE; rungs 0..k-1 alternate an x0 prediction (positive t)
    with the re-noised latent handed to the next rung (negative t). Any of
    those would fingerprint something that is not the committed latent and
    would read as an off-manifold sample on every arm -- a null that
    looked exactly like a result.

    MUTATION (verified RED): select ``sdt[:,1] == 0`` (first rung) or
    ``sdt[:,1].min()`` (the noise row) instead of the max rung -- the
    index assertion fails either way.
    """
    import numpy as np
    rows = []
    for step in range(3):
        rows += [(step, -1, 1000.0), (step, 0, 1000.0), (step, 0, -625.0),
                 (step, 1, 625.0), (step, 1, -357.0), (step, 2, 357.0),
                 (step, 2, -208.0), (step, 3, 208.0)]
    npz = {"sdt": np.array(rows, dtype=np.float64)}
    assert flowrec_committed(npz) == [7, 15, 23]

    # Defensive half of the filter. No CURRENT recording emits a
    # negative-t row at the TOP rung (the last rung is never re-noised),
    # but a higher-order sampler would, and it must not be mistaken for
    # the commit. Documented as defensive rather than claimed as covered.
    rows2 = list(rows) + [(3, -1, 1000.0), (3, 3, 208.0), (3, 3, -100.0)]
    npz2 = {"sdt": np.array(rows2, dtype=np.float64)}
    assert flowrec_committed(npz2) == [7, 15, 23, 25]


def test_split_seed_tag_keeps_seed_replicates_off_the_arm_count():
    """``_s43/_s44/_s45`` are the SAME model, not extra replications.

    MUTATION (verified RED): return ``(arm_name, 's1234')`` always -- the
    base-arm assertions fail and every seed replicate would be counted as
    an independent model in the cross-arm consistency number.
    """
    assert split_seed_tag("poolrich_s44") == ("poolrich", "s44")
    assert split_seed_tag("nogan200") == ("nogan200", "s1234")
    assert split_seed_tag("horizon_wave90") == ("horizon_wave90", "s1234")
    bases = {split_seed_tag(n)[0] for n in
             ("poolrich", "poolrich_s43", "poolrich_s44", "poolrich_s45")}
    assert bases == {"poolrich"}


def test_rebin_depth_bands_and_drops_out_of_range():
    """One recorded rollout = one sample per (arm, depth); bands pool them.

    MUTATION (verified RED): fold out-of-range depths into the nearest
    band instead of dropping them -- the depth-99 assertion fails.
    """
    recs = [{"arm": ARM_ROLL, "depth": d, "s": 0.1} for d in (0, 2, 3, 9, 39)]
    recs.append({"arm": ARM_ROLL, "depth": 99, "s": 0.1})
    out = rebin_depth(recs, [0, 3, 10, 20, 30, 40])
    assert [r["depth"] for r in out] == [0, 0, 3, 3, 30]
    assert [r["depth_raw"] for r in out] == [0, 2, 3, 9, 39]
    assert all(r["depth_raw"] != 99 for r in out), "out-of-range must DROP"
    # the originals must not be mutated
    assert recs[0]["depth"] == 0 and "depth_raw" not in recs[0]


def test_seed_noise_floor_is_measured_or_omitted_never_zero():
    """No seed replicates => the floor is UNMEASURED, not 0.

    A zero floor would make every effect clear it, which is the forgeable
    -zero failure wearing a statistical hat.

    MUTATION (verified RED): return ``{"floor": 0.0, ...}`` instead of
    None when no base arm has two seeds -- the None assertion fails.
    """
    single = [{"arm": ARM_ROLL, "depth": 0, "s": 0.5,
               "base_arm": "a", "seed_tag": "s1234"}]
    assert seed_noise_floor(single) is None

    recs = []
    for tag, v in (("s1234", 0.50), ("s43", 0.54), ("s44", 0.46)):
        recs.append({"arm": ARM_ROLL, "depth": 0, "s": v,
                     "base_arm": "a", "seed_tag": tag})
    fl = seed_noise_floor(recs)
    assert fl is not None
    assert fl["floor"] == pytest.approx(0.04, abs=1e-6)   # sd of .50/.54/.46
    assert fl["n_points"] == 1


def test_head_drop_convention_is_the_real_guard(monkeypatch):
    """The GT pairing convention is checked in CODE, not inferred from data.

    ``ZarrRideDataset.load_latent_chunk`` slices
    ``[start + _LATENT_HEAD_DROP : end + _LATENT_HEAD_DROP]`` while
    ``utils/eval_causal_AR.py`` slices the RAW zarr. The two agree iff
    ``_LATENT_HEAD_DROP == 0``, so this is an exact check with a definite
    answer -- unlike the data profile, which is drift-dominated (see the
    next test).

    MUTATION (verified RED): make ``assert_head_drop_convention`` return
    the value instead of raising when it is non-zero -- the
    ``pytest.raises`` block fails and a flipped head-drop would silently
    offset every GT target by one frame.
    """
    import utils.zarr_dataset as zd

    assert zd._LATENT_HEAD_DROP == 0     # the shipped value
    assert assert_head_drop_convention(where="ok") == 0   # no raise

    monkeypatch.setattr(zd, "_LATENT_HEAD_DROP", 1, raising=True)
    with pytest.raises(SystemExit, match="_LATENT_HEAD_DROP"):
        assert_head_drop_convention(where="armA")


def test_gt_alignment_is_gross_error_only_not_an_argmin_assert(caplog):
    """Only a WHOLE-CHUNK misalignment may stop the run; drift may not.

    The old guard asserted ``argmin == 0`` per arm. That was never
    supportable: all 26 texture_abc arms share one zarr, one
    ``latent_start_offset`` and one ``ar_initial_chunks``, yet 14 of them
    individually prefer shift +2 and the aggregate prefers 0 by 0.13%.
    The argmin measures each MODEL's temporal drift. What survives is a
    relative-excess bound on shift 0.

    MUTATION (verified RED): restore ``if best != 0: raise`` in
    ``assert_gt_alignment`` -- the drift-scale case below raises and the
    study is blocked on real data again.
    """
    g = torch.Generator().manual_seed(4)
    truth = torch.randn(60, 4, 5, 5, generator=g)          # 60 GT frames

    def loader(s, e):
        return truth[s:e].unsqueeze(0)

    lats = [truth[9 + 3 * k: 12 + 3 * k].unsqueeze(0) for k in range(8)]
    prof = gt_alignment_profile(lats, loader, 9, n_depths=8)
    assert min(prof, key=lambda k: prof[k]) == 0
    assert prof[0] == pytest.approx(0.0, abs=1e-6)
    assert_gt_alignment(prof, where="aligned")             # no raise

    # GENUINE whole-chunk misalignment: shift 0 far worse than the best.
    shifted = [truth[12 + 3 * k: 15 + 3 * k].unsqueeze(0) for k in range(8)]
    bad = gt_alignment_profile(shifted, loader, 9, n_depths=8)
    assert min(bad, key=lambda k: bad[k]) == 3   # one CHUNK = 3 frames
    with pytest.raises(SystemExit, match="OFF BY"):
        assert_gt_alignment(bad, where="shifted")

    # DRIFT-SCALE preference for a non-zero shift must NOT raise: this is
    # the real-data shape (aggregate 0.36221 at 0 vs 0.36269 at +2, and
    # per-arm margins of a few percent in either direction).
    caplog.set_level("INFO")
    drift = {-3: 0.3400, -2: 0.3550, -1: 0.3600, 0: 0.36221,
             1: 0.3630, 2: 0.36269, 3: 0.3700}
    assert min(drift, key=lambda k: drift[k]) == -3
    assert_gt_alignment(drift, where="driftarm")           # no raise
    assert "argmin is NOT asserted" in caplog.text

    # The worst REAL arm: strict03 prefers -3 by 8.9%, all drift. Must pass
    # at the default, or the study stays blocked on real data.
    strict03 = {-3: 0.3300, -2: 0.3450, -1: 0.3540, 0: 0.35937,
                1: 0.3600, 2: 0.3590, 3: 0.3650}
    assert (strict03[0] - strict03[-3]) / strict03[-3] == pytest.approx(
        0.089, abs=5e-4)
    assert_gt_alignment(strict03, where="strict03")        # no raise

    # ... but the same shape past the bound does raise.
    with pytest.raises(SystemExit, match="align-max-rel-excess"):
        assert_gt_alignment(drift, where="driftarm", max_rel_excess=0.001)

    with pytest.raises(SystemExit):
        assert_gt_alignment({}, where="empty")
    with pytest.raises(SystemExit, match="no shift-0"):
        assert_gt_alignment({1: 0.1, 2: 0.2}, where="noshift0")


def test_gt_self_difference_profile_reports_resolving_power(caplog):
    """The drift-dominated verdict must print on NORMAL runs, not failures.

    ``|GT(s) - GT(s+d)|`` is how far apart GT frames are from each other.
    When rollout-vs-GT at shift 0 already exceeds that at +-1 (measured:
    0.32-0.46 vs ~0.198), a one-frame misindexing moves the statistic by
    less than the drift floor and the profile CANNOT confirm the
    convention.

    MUTATION (verified RED): drop the ``self_profile`` branch from
    ``assert_gt_alignment``, or log it only inside the failure path --
    the ``DRIFT-DOMINATED`` assertion fails and a passing ``argmin == 0``
    could be re-read as confirmation of the indexing again.
    """
    g = torch.Generator().manual_seed(11)
    truth = torch.randn(60, 4, 5, 5, generator=g)

    def loader(s, e):
        return truth[s:e].unsqueeze(0)

    self_prof = gt_self_difference_profile(loader, 9, n_depths=8)
    assert self_prof[0] == pytest.approx(0.0, abs=1e-9)   # 0 by definition
    assert self_prof[1] > 0.0 and self_prof[-1] > 0.0

    # Drift-dominated: rollout-vs-GT at 0 above the GT self-difference.
    caplog.set_level("INFO")
    big = self_prof[1] * 2.0
    prof = {-1: big * 1.01, 0: big, 1: big * 1.01}
    assert_gt_alignment(prof, where="dd", self_profile=self_prof)
    assert "GT SELF-difference by shift" in caplog.text
    assert "DRIFT-DOMINATED" in caplog.text
    assert "CANNOT confirm the indexing convention" in caplog.text

    # Resolvable regime (rollout far closer than GT differs from itself):
    # no drift-dominated claim is made.
    caplog.clear()
    small = self_prof[1] * 0.1
    assert_gt_alignment({-1: small * 3, 0: small, 1: small * 3},
                        where="res", self_profile=self_prof)
    assert "GT SELF-difference by shift" in caplog.text
    assert "DRIFT-DOMINATED" not in caplog.text


def test_align_max_rel_excess_flows_config_to_consumer():
    """The gross-error bound is a FLAG, and the caller actually passes it.

    MUTATION (verified RED): drop ``--align-max-rel-excess`` from
    ``build_parser`` (AttributeError on the default), or hardcode
    ``max_rel_excess=0.05`` at the ``run_flowrec`` call site (the source
    assertion fails) -- the knob would exist in name only.
    """
    import inspect

    from analysis.dmd_fp_depth_study import (
        ALIGN_MAX_REL_EXCESS, build_parser, run_flowrec)

    p = build_parser()
    d = p.parse_args(["--out", "x.jsonl"])
    assert d.align_max_rel_excess == pytest.approx(ALIGN_MAX_REL_EXCESS)
    # The default must clear the MEASURED drift ceiling (strict03 prefers
    # -3 by 8.9% purely from drift) or the guard false-positives and the
    # study stays blocked for the wrong reason.
    assert ALIGN_MAX_REL_EXCESS > 0.089
    d2 = p.parse_args(["--out", "x.jsonl", "--align-max-rel-excess", "0.2"])
    assert d2.align_max_rel_excess == pytest.approx(0.2)

    src = inspect.getsource(run_flowrec)
    assert "max_rel_excess=args.align_max_rel_excess" in src
    assert "assert_head_drop_convention(" in src
    assert "gt_self_difference_profile(" in src

    # And the consumer honours it: same profile fails tight, passes wide.
    prof = {-1: 0.30, 0: 0.33, 1: 0.34}
    with pytest.raises(SystemExit):
        assert_gt_alignment(prof, where="tight", max_rel_excess=0.05)
    assert_gt_alignment(prof, where="wide", max_rel_excess=0.20)


def test_discover_flowrec_arms_reads_metadata_and_tf_position(tmp_path):
    """Pairing comes from the run's own sidecar; ``_c<N>`` gives TF position.

    MUTATION (verified RED): ignore ``rank0_ride.json`` and always use the
    fallbacks -- the zarr/offset/seed_chunks assertions fail, and the GT
    pairing would silently come from an analysis-time guess instead of
    from the run that produced the latents.
    """
    import numpy as np
    a = tmp_path / "texture_abc_poolrich_s43"
    (a / "flowrec").mkdir(parents=True)
    np.savez(a / "flowrec" / "steps.npz", sdt=np.zeros((1, 3)))
    (a / "rank0_ride.json").write_text(json.dumps({
        "zarr_path": "/x/ride.zarr", "latent_start_offset": 100,
        "ar_initial_chunks": 3, "rank_mode": "dataset"}))
    b = tmp_path / "r08_TF_c4"
    b.mkdir()
    np.savez(b / "steps.npz", sdt=np.zeros((1, 3)))

    arms = discover_flowrec_arms(
        [str(tmp_path / "*")], fallback_zarr="/fb/f.zarr",
        fallback_offset=1380, fallback_seed_chunks=3)
    by = {x["name"]: x for x in arms}
    assert set(by) == {"texture_abc_poolrich_s43", "r08_TF_c4"}

    m = by["texture_abc_poolrich_s43"]
    assert m["zarr"] == "/x/ride.zarr" and m["offset"] == 100
    assert m["seed_chunks"] == 3 and m["rank_mode"] == "dataset"
    assert m["base_arm"] == "texture_abc_poolrich" and m["seed_tag"] == "s43"
    assert m["world_pos"] is None

    # COLLIDING LEAF NAMES must be disambiguated by their model directory,
    # or two different models would merge into one "arm" -- pooling exactly
    # what this study refuses to pool, and inflating the replication count.
    #
    # MUTATION (verified RED): drop the disambiguation block -- both
    # r08_F_s0 directories come back as one name and the assertion fails.
    for model in ("flow_long36_roll", "flow_long36_rollkl"):
        d = tmp_path / "fv" / model / "r08_F_s0"
        d.mkdir(parents=True)
        np.savez(d / "steps.npz", sdt=np.zeros((1, 3)))
    coll = discover_flowrec_arms([str(tmp_path / "fv" / "*" / "r08_*")])
    assert sorted(x["name"] for x in coll) == [
        "flow_long36_roll/r08_F_s0", "flow_long36_rollkl/r08_F_s0"]
    assert len({x["base_arm"] for x in coll}) == 2

    t = by["r08_TF_c4"]
    assert t["zarr"] == "/fb/f.zarr" and t["offset"] == 1380
    # _c4 => the chunk was generated on 4 chunks of full GT context, so its
    # GT counterpart sits 4 chunks in.
    assert t["seed_chunks"] == 4 and t["world_pos"] == 4
    assert t["rank_mode"] is None


def test_flowrec_report_never_pools_arms_and_needs_a_majority():
    """Cross-arm replication decides it, not one pooled curve.

    Two arms with a strong effect and three without must NOT pass -- a
    result that reproduces in a minority of models is not a result.

    MUTATION (verified RED): pool every arm into one ``summarize`` call --
    the per-arm rows collapse to one and the 2/5 assertion fails.
    """
    class _A:
        auc_min, mono_tol = 0.8, 0.01
        confound_share_max, deep_depth_min = 0.5, 10
        depth_bands = [0, 3, 10, 20, 30, 40]

    recs = []
    for arm_i in range(5):
        strong = arm_i < 2
        for d in list(range(0, 40)):
            roll = 0.9 - (0.02 * d if strong else 0.0)
            for a, v in ((ARM_GT, 0.9), (ARM_ROLL, roll),
                         (ARM_VARMATCH, 0.9 - 0.001 * d)):
                recs.append({"study_arm": f"arm{arm_i}", "base_arm": f"arm{arm_i}",
                             "seed_tag": "s1234", "arm": a, "depth": d,
                             "s": v, "mode": "flowrec"})
    buf = io.StringIO()
    res = print_flowrec_report(recs, _A(), achieved_depths=list(range(40)),
                               arm_names=[f"arm{i}" for i in range(5)],
                               out_fh=buf)
    assert res["n_arms"] == 5
    assert res["n_pass"] == 2, res["n_pass"]
    txt = buf.getvalue()
    assert "PROBE CANNOT RANK OFF-MANIFOLD DISTANCE -- GATE NOT VIABLE" in txt
    assert "arms meeting the criterion : 2/5" in txt
    assert "SEED NOISE FLOOR: UNMEASURED" in txt
    # depth is the axis; 3 frames per chunk is explicitly de-emphasised
    assert "COARSE, not the informative axis" in txt


def test_flowrec_report_reference_free_mode_refuses_to_judge():
    """No GT arm => no separation criterion. Trend only, and it says so."""
    class _A:
        auc_min, mono_tol = 0.8, 0.01
        confound_share_max, deep_depth_min = 0.5, 10
        depth_bands = [0, 3, 10, 20, 30, 40]

    recs = [{"study_arm": "F", "base_arm": "F", "seed_tag": "s1234",
             "arm": ARM_ROLL, "depth": d, "s": 0.5 - 0.01 * d,
             "mode": "flowrec", "gt_valid": False} for d in range(40)]
    buf = io.StringIO()
    print_flowrec_report(recs, _A(), achieved_depths=list(range(40)),
                         arm_names=["F"], gt_valid=False, out_fh=buf)
    txt = buf.getvalue()
    assert "NO VERDICT" in txt
    assert "REFERENCE-FREE, no GT" in txt
    assert "CONSTANT COMPASS ACTIONS" in txt
    assert "GATE VIABLE" not in txt


def test_share_below_gt_is_the_paired_win_rate():
    recs = []
    for w in range(10):
        recs.append({"window": w, "arm": ARM_GT, "depth": 2, "s": 0.5})
        recs.append({"window": w, "arm": ARM_ROLL, "depth": 2,
                     "s": 0.4 if w < 7 else 0.6})
    row = summarize(recs, metric="s")["rows"][0]
    assert row["share_below_gt"] == pytest.approx(0.7)
    assert row["paired_gt_minus_roll"] == pytest.approx(
        (7 * 0.1 + 3 * -0.1) / 10)
