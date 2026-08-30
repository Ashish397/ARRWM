"""CPU contract tests for the main-only stat-anchor tail robustifier."""

import pytest
import torch

from model.anti_collapse import attenuate_stat_anchor_tail


@pytest.mark.parametrize("value", [0.0, 0.5, 3.999, 4.0])
def test_identity_through_threshold(value):
    raw = torch.tensor(value, dtype=torch.float64, requires_grad=True)
    effective, scale, active = attenuate_stat_anchor_tail(raw, 4.0)
    effective.backward()

    assert effective.item() == pytest.approx(value)
    assert raw.grad.item() == pytest.approx(1.0)
    assert scale.item() == pytest.approx(1.0)
    assert active.item() == pytest.approx(0.0)


@pytest.mark.parametrize("value", [8.460460662841797, 67.1358642578125])
def test_rational_tail_value_and_gradient(value):
    threshold = 4.0
    raw = torch.tensor(value, dtype=torch.float64, requires_grad=True)
    effective, scale, active = attenuate_stat_anchor_tail(raw, threshold)
    effective.backward()

    expected_value = 2.0 * threshold - threshold**2 / value
    expected_scale = (threshold / value) ** 2
    assert effective.item() == pytest.approx(expected_value)
    assert raw.grad.item() == pytest.approx(expected_scale)
    assert scale.item() == pytest.approx(expected_scale)
    assert active.item() == pytest.approx(1.0)
    assert threshold < effective.item() < 2.0 * threshold


def test_disabled_path_preserves_object_and_graph():
    raw = torch.tensor(67.0, requires_grad=True)
    effective, scale, active = attenuate_stat_anchor_tail(raw, 0.0)
    assert effective is raw
    effective.backward()
    assert raw.grad.item() == pytest.approx(1.0)
    assert scale.item() == pytest.approx(1.0)
    assert active.item() == pytest.approx(0.0)


def test_rejects_non_scalar_loss():
    with pytest.raises(ValueError, match="scalar loss"):
        attenuate_stat_anchor_tail(torch.ones(2), 4.0)
