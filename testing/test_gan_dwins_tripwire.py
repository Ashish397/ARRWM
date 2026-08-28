"""CPU tests for the D-WINS TRIPWIRE (``model/gan_balance.py``).

    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" PYTHONPATH=. \
        python -m pytest testing/test_gan_dwins_tripwire.py -q

THE TWO TRAJECTORIES BELOW ARE REAL, not synthetic. They are the
``d_loss`` columns grepped verbatim out of

    logs/holdersmoke_pixdirect_strong_h6143213.log
    logs/holdersmoke_pixdirect_onlinelr_h6143212.log

(``grep -o 'GAN-HEALTH] step=[0-9]* .* d_loss=[0-9.]*'``), which is what
makes this a regression test rather than a demo: the detector is required
to separate the arm that collapsed from the arm that recovered, on the
actual numbers, and to do it from the SECOND-TO-LAST trip observation
(step 181) rather than from a run-level median at step 200.

The two arms differ in exactly one thing: ``pixdirect_strong`` raised
``ladd_pixel_crops_per_row`` 1->2 and ``ladd_pixel_lat_frames`` 2->3.
Both run encoder ``lr_scale=0.4``. n = 17 logged observations each.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.gan_balance import (  # noqa: E402
    DEFAULT_DWINS_FLOOR, DEFAULT_DWINS_K, DWinsTripwire,
)

# step -> d_loss, verbatim. Steps 11 and 21 read 0.0000 in BOTH logs:
# that is the "the disc has not started" sentinel, not a collapse.
STRONG = [
    (11, 0.0000), (21, 0.0000), (31, 0.6685), (41, 0.3585), (51, 0.6312),
    (61, 0.4138), (71, 0.4322), (81, 0.6700), (91, 0.6644), (101, 0.3632),
    (111, 0.0431), (121, 0.0482), (131, 0.4134), (141, 0.4907), (151, 0.3268),
    (161, 0.0104), (171, 0.0039), (181, 0.0024), (191, 0.0102),
]
ONLINELR = [
    (11, 0.0000), (21, 0.0000), (31, 0.6655), (41, 0.5736), (51, 0.6299),
    (61, 0.3555), (71, 0.4678), (81, 0.1593), (91, 0.5968), (101, 0.6841),
    (111, 0.1346), (121, 0.0139), (131, 0.6243), (141, 0.2698), (151, 0.4692),
    (161, 0.5297), (171, 0.3937), (181, 0.2781), (191, 0.4560),
]


def _replay(series, **kw):
    tw = DWinsTripwire(min_step=20, **kw)
    edges = [s for s, v in series if tw.observe(s, v)]
    return tw, edges


# ===========================================================================
# 1. THE MEASURED DISCRIMINATION -- collapse vs recovered transient
# ===========================================================================
def test_strong_arm_trips_and_names_the_step():
    tw, edges = _replay(STRONG)
    assert tw.tripped is True
    assert edges == [181], edges
    assert tw.trip_step == 181
    assert tw.streak == 4, "161/171/181/191, still falling at run end"
    assert tw.max_streak == 4
    assert tw.below_total == 6, "111,121 + 161,171,181,191"
    assert tw.observations == 17, "19 rows minus the two 0.0 rows"
    # 1, not 2: step 11 is rejected by ``min_step=20`` FIRST, so only
    # step 21 reaches the exact-zero sentinel branch. The two guards
    # are ordered, and this counter is what proves which one fired.
    assert tw.skipped_sentinel == 1


def test_onlinelr_arm_does_not_trip_because_it_recovered():
    tw, edges = _replay(ONLINELR)
    assert tw.tripped is False
    assert edges == []
    assert tw.trip_step == -1
    assert tw.max_streak == 2, "111 (0.1346) and 121 (0.0139), then 0.6243"
    assert tw.below_total == 2
    assert tw.observations == 17


def test_the_median_cannot_tell_them_apart_but_the_tripwire_can():
    """The whole reason the tripwire exists, asserted as a number.

    Both medians over steps >= 50 sit inside or near the 0.25-0.55
    healthy band; only the streak counters separate the two arms.
    """
    import statistics as _st

    s_med = _st.median([v for s, v in STRONG if s >= 50])
    o_med = _st.median([v for s, v in ONLINELR if s >= 50])
    assert abs(s_med - 0.3632) < 1e-6
    assert abs(o_med - 0.4560) < 1e-6
    assert 0.25 <= s_med <= 0.55, "the collapsed arm looks HEALTHY on median"

    s_tw, _ = _replay(STRONG)
    o_tw, _ = _replay(ONLINELR)
    assert s_tw.tripped and not o_tw.tripped
    # Second-half medians DO separate them -- recorded so the finding is
    # not lost, but it needs the split to be chosen after the fact.
    assert _st.median([v for s, v in STRONG if s >= 111]) < 0.05
    assert _st.median([v for s, v in ONLINELR if s >= 111]) > 0.35


def test_it_fires_before_the_run_ends():
    """Detection latency is the point: step 181 of 200, from a run whose
    final median would have been reported as a success."""
    tw, edges = _replay(STRONG)
    assert edges[0] <= 191 - 10, "at least one logged step before the end"


# ===========================================================================
# 2. THE FORGEABLE-ZERO GUARD
# ===========================================================================
def test_exact_zero_is_rejected_not_counted():
    """``d_loss=0.0000`` is the pre-``gan_disc_start_step`` not-run
    sentinel. A tripwire that counted it would fire on EVERY run."""
    tw = DWinsTripwire(min_step=0)
    for s in (0, 1, 2, 3, 4, 5):
        assert tw.observe(s, 0.0) is False
    assert tw.tripped is False
    assert tw.observations == 0
    assert tw.skipped_sentinel == 6
    assert tw.streak == 0


def test_a_zero_row_does_not_break_a_real_streak():
    """Rejected means IGNORED, not streak-breaking: a missing row must
    not launder a collapse into two short streaks."""
    tw = DWinsTripwire(min_step=0)
    tw.observe(10, 0.01)
    tw.observe(20, 0.0)          # sentinel row in the middle
    tw.observe(30, 0.01)
    tw.observe(40, 0.01)
    assert tw.tripped is True
    assert tw.trip_step == 40


def test_none_and_nan_are_ignored():
    tw = DWinsTripwire(min_step=0)
    assert tw.observe(1, None) is False
    assert tw.observe(2, float("nan")) is False
    assert tw.observations == 0
    assert tw.skipped_sentinel == 0


def test_min_step_suppresses_the_pre_disc_regime():
    tw = DWinsTripwire(min_step=20)
    for s in (0, 5, 10, 15):
        tw.observe(s, 0.001)
    assert tw.observations == 0 and tw.tripped is False
    for s in (20, 21, 22):
        tw.observe(s, 0.001)
    assert tw.tripped is True and tw.trip_step == 22


# ===========================================================================
# 3. LATCH / K / FLOOR SEMANTICS
# ===========================================================================
def test_the_latch_does_not_reset_when_d_loss_recovers():
    tw, _ = _replay(STRONG)
    tw.observe(201, 0.60)
    assert tw.tripped is True and tw.trip_step == 181
    assert tw.streak == 0, "the streak resets; the latch does not"


def test_edge_is_returned_exactly_once():
    tw = DWinsTripwire(min_step=0)
    edges = [tw.observe(s, 0.001) for s in range(10)]
    assert sum(1 for e in edges if e) == 1
    assert edges.index(True) == DEFAULT_DWINS_K - 1


@pytest.mark.parametrize("k,expect", [(1, True), (2, True), (3, False),
                                      (5, False)])
def test_k_is_configurable_against_the_recovered_transient(k, expect):
    """``onlinelr``'s transient is 2 observations. k<=2 would call it a
    collapse; k=3 is the smallest value that does not."""
    tw, _ = _replay(ONLINELR, k_consecutive=k)
    assert tw.tripped is expect


def test_floor_is_inclusive_and_configurable():
    tw = DWinsTripwire(floor=0.135, k_consecutive=2, min_step=0)
    tw.observe(1, 0.135)
    tw.observe(2, 0.135)
    assert tw.tripped is True, "<= floor, not < floor"

    loose = DWinsTripwire(floor=0.30, k_consecutive=3, min_step=0)
    for s, v in ONLINELR:
        loose.observe(s, v)
    assert loose.max_streak >= 2


def test_defaults_are_the_documented_ones():
    assert DEFAULT_DWINS_FLOOR == 0.135
    assert DEFAULT_DWINS_K == 3
    tw = DWinsTripwire()
    assert tw.floor == 0.135 and tw.k == 3 and tw.min_step == 0


# ===========================================================================
# 4. THE TELEMETRY KEYS -- predicted values, so a wrong one is detectable
# ===========================================================================
def test_log_keys_and_predicted_values_for_the_strong_replay():
    tw, _ = _replay(STRONG)
    lg = tw.logs()
    assert lg["train/gan_dwins_tripped"] == 1.0
    assert lg["train/gan_dwins_trip_step"] == 181.0
    assert lg["train/gan_dwins_max_streak"] == 4.0
    assert lg["train/gan_dwins_streak"] == 4.0
    assert lg["train/gan_dwins_below_floor_total"] == 6.0
    assert lg["train/gan_dwins_observations"] == 17.0
    assert lg["train/gan_dwins_skipped_sentinel"] == 1.0
    assert lg["train/gan_dwins_floor"] == 0.135
    assert lg["train/gan_dwins_k"] == 3.0


def test_log_keys_for_a_run_that_never_tripped_are_not_forgeable_zeros():
    """``trip_step`` must be -1, not 0, on a healthy run: step 0 is a
    real step number."""
    tw, _ = _replay(ONLINELR)
    lg = tw.logs()
    assert lg["train/gan_dwins_tripped"] == 0.0
    assert lg["train/gan_dwins_trip_step"] == -1.0
    assert lg["train/gan_dwins_max_streak"] == 2.0


def test_message_names_the_forbidden_and_the_sanctioned_fix():
    tw, _ = _replay(STRONG)
    m = tw.message()
    assert "gan_loss_weight" in m
    assert "gan_updates_per_step" in m and "gan_lr" in m
    assert "181" in m


# ===========================================================================
# 5. THE D-SIDE COUNTERWEIGHT IS COUNTED
# ===========================================================================
# ``gan_updates_per_step`` is the sanctioned lever for rebalancing G/D
# (``gan_loss_weight`` is forbidden -- it scales only the G side). Until
# now NOTHING in telemetry reported its realised value:
# ``r3gan_disc_updates_total`` counts D PHASES, and read 170 on a
# 200-step arm, i.e. ~1 per step, not 5. These are text assertions on the
# trainer source because importing the trainer needs a GPU.
_TRAINER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "trainer", "causal_action_forcing_train.py")


def test_inner_disc_update_counter_is_wired():
    with open(_TRAINER) as fh:
        src = fh.read()
    assert "_ladd_disc_inner_updates_total" in src
    assert "_ladd_disc_inner_updates_last" in src
    assert '"train/r3gan_disc_inner_updates_total"' in src
    assert '"train/r3gan_disc_inner_updates_last"' in src
    # It must accumulate the LOOP BOUND, not a constant.
    i = src.index("self._ladd_disc_inner_updates_total = int(")
    assert "int(n_disc_updates)" in src[i:i + 300]
