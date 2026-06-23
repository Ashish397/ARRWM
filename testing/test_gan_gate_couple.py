"""Regression test for the MAE-gate -> GAN-weight coupling
(ActionForcingTrainer._couple_gan_weight_to_gate).

The feature: when the MAE gate cuts the DMD loss (gate_w < 1, teacher
unreliable), the gen-side GAN weight should follow it DOWN but gentler
(``gate_w**beta``, beta<1 => sqrt), never below ``floor_frac*gan_base``, and
stay >= ``min_ratio`` x the gated DMD weight so the GAN still carries the
student without running free. It must NEVER force GAN above its base.

The trainer can't be imported on a CPU/login node (heavy CUDA-at-import), so
this mirrors the helper's exact math and locks the behavior, incl. the
documented SCALE limitation (the >=min_ratio*DMD floor is unsatisfiable when
gan_base << dmd_base, e.g. the default gan_loss_weight=0.05 vs dmd=1.0 -> the
coupling is inert there; it is functional for the stab config gan_base=1.0).

Run: python -m pytest testing/test_gan_gate_couple.py -q
"""


def couple(gen_gan_weight, gate_w, *, enabled=True, dmd_base=1.0,
           beta=0.5, floor_frac=0.3, min_ratio=2.0):
    """Exact mirror of ActionForcingTrainer._couple_gan_weight_to_gate."""
    if not enabled or gen_gan_weight <= 0.0:
        return gen_gan_weight
    if gate_w >= 1.0:
        return gen_gan_weight
    gan_base = float(gen_gan_weight)
    dmd_eff = dmd_base * gate_w
    coupled = gan_base * (gate_w ** beta)
    floor = max(min_ratio * dmd_eff, floor_frac * gan_base)
    floor = min(floor, gan_base)
    return float(min(max(coupled, floor), gan_base))


def test_disabled_is_noop():
    assert couple(1.0, 0.2, enabled=False) == 1.0
    assert couple(0.05, 0.01, enabled=False) == 0.05


def test_full_dmd_unchanged():
    # gate_w >= 1 -> no gating -> GAN at full base.
    assert couple(1.0, 1.0) == 1.0
    assert couple(0.5, 1.5) == 0.5


def test_never_above_base_never_negative():
    for gb in (0.05, 0.5, 1.0, 2.0):
        for g in [i / 20 for i in range(0, 21)]:
            v = couple(gb, g)
            assert 0.0 <= v <= gb + 1e-12, (gb, g, v)


def test_monotone_nonincreasing_in_gate():
    # As the gate cuts harder (gate_w down), GAN weight must not go UP.
    gb = 1.0
    gates = [g / 100 for g in range(1, 100)]
    vals = [couple(gb, g) for g in gates]
    for lo, hi in zip(vals, vals[1:]):
        assert lo <= hi + 1e-12, (lo, hi)


def test_comparable_scale_keeps_gan_dominant():
    # gan_base == dmd_base == 1.0 (the stab config): GAN stays >= 2x gated-DMD
    # wherever it's reducible, full above gate 0.5, floored at deep gating.
    assert abs(couple(1.0, 0.5) - 1.0) < 1e-9          # full (2*0.5=1.0 binds)
    assert abs(couple(1.0, 0.25) - 0.5) < 1e-9         # 2*dmd_eff = 0.5
    # ratio GAN/DMD_eff >= 2 across the reducible band
    for g in (0.1, 0.2, 0.25, 0.4):
        v = couple(1.0, g)
        assert v / (1.0 * g) >= 2.0 - 1e-9, (g, v)
    # deep gating: fractional floor rescues (doesn't collapse to 0)
    assert couple(1.0, 0.01) >= 0.3 - 1e-9


def test_zero_gate_no_nan():
    # gate_w exactly 0 (min_weight=0): 0**0.5 = 0, floor_frac rescues.
    v = couple(1.0, 0.0)
    assert v == v  # not NaN
    assert abs(v - 0.3) < 1e-9


def test_scale_mismatch_is_inert_documented():
    # gan_base << dmd_base (default gan_loss_weight=0.05): the >=min_ratio*DMD
    # floor saturates -> GAN pinned at base for ~all gate_w. This is the KNOWN
    # limitation surfaced by the init-time warning; locked here so a future
    # change that claims to "fix the coupling" must update this expectation.
    for g in (0.5, 0.25, 0.1, 0.05):
        assert abs(couple(0.05, g) - 0.05) < 1e-9, g
    # only the extreme tail (gate_w < gan_base/(min_ratio*dmd_base) = 0.025) bites
    assert couple(0.05, 0.01) < 0.05


if __name__ == "__main__":
    test_disabled_is_noop()
    test_full_dmd_unchanged()
    test_never_above_base_never_negative()
    test_monotone_nonincreasing_in_gate()
    test_comparable_scale_keeps_gan_dominant()
    test_zero_gate_no_nan()
    test_scale_mismatch_is_inert_documented()
    print("gan gate couple tests passed")
