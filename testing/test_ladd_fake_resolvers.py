"""Branch-complete coverage for the two One-Forcing-port resolvers.

``model/dmd_action_forcing.py`` grew two module-level, pure, explicitly
"unit-testable without a GPU" functions on 2026-08-25:

    resolve_ladd_fake_backbone_trainable(args, ladd_feature_source) -> bool
    resolve_ladd_fake_sample_source(args) -> str

They encode the illegal-combination contract for the One-Forcing port
(disc on the TRAINABLE fake_score; DMD sample as the disc's fake). Every
branch is exercised here, including all five ``raise`` paths and the
default-off / default-flash paths, plus the tolerated falsy spellings
that ``or "flash"`` swallows.

The module drags ``wan.modules.t5``, which evaluates
``torch.cuda.current_device()`` at import time, so the import is done
under a patch -- the same workaround
``testing/test_r1_cadence_and_override_guard.py`` uses. Nothing here
touches CUDA afterwards.

Run:
    python -m pytest testing/test_ladd_fake_resolvers.py -q
or
    python testing/test_ladd_fake_resolvers.py
"""
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with patch.object(torch.cuda, "current_device", return_value=0):
    from model.dmd_action_forcing import (  # noqa: E402
        resolve_ladd_fake_backbone_trainable,
        resolve_ladd_fake_sample_source,
    )


def _args(**kw):
    """A config stub. Deliberately a bare namespace: the resolvers read
    everything through ``getattr(args, key, default)``, so an ABSENT key
    and an explicitly-set one must be distinguishable."""
    return SimpleNamespace(**kw)


# ===========================================================================
# resolve_ladd_fake_backbone_trainable
# ===========================================================================
def test_backbone_default_off_when_the_key_is_absent():
    """The whole point of the default: an untouched config must resolve
    False and must NOT raise, whatever the other knobs say."""
    for src in ("real", "fake", "teacher", ""):
        got = resolve_ladd_fake_backbone_trainable(_args(), src)
        assert got is False, f"feature_source={src!r} -> {got!r}"


def test_backbone_off_short_circuits_every_other_check():
    """Explicitly False + every co-requisite violated: still no raise."""
    a = _args(
        ladd_fake_backbone_trainable=False,
        ladd_defer_disc_update=False,
        streaming_mode=False,
    )
    assert resolve_ladd_fake_backbone_trainable(a, "real") is False


def test_backbone_falsy_spellings_are_off():
    for v in (0, "", None, [], 0.0):
        a = _args(ladd_fake_backbone_trainable=v)
        assert resolve_ladd_fake_backbone_trainable(a, "real") is False


def test_backbone_on_requires_feature_source_fake():
    a = _args(
        ladd_fake_backbone_trainable=True,
        ladd_defer_disc_update=True,
        streaming_mode=True,
    )
    for bad in ("real", "teacher", "FAKE", ""):
        with pytest.raises(ValueError) as ei:
            resolve_ladd_fake_backbone_trainable(a, bad)
        msg = str(ei.value)
        assert "ladd_feature_source='fake'" in msg
        assert repr(bad) in msg


def test_backbone_on_requires_defer_disc_update():
    a = _args(
        ladd_fake_backbone_trainable=True,
        ladd_defer_disc_update=False,
        streaming_mode=True,
    )
    with pytest.raises(ValueError) as ei:
        resolve_ladd_fake_backbone_trainable(a, "fake")
    assert "ladd_defer_disc_update=true" in str(ei.value)


def test_backbone_on_requires_defer_when_the_key_is_absent():
    """Absent ``ladd_defer_disc_update`` defaults FALSE -> must raise, not
    silently proceed."""
    a = _args(ladd_fake_backbone_trainable=True, streaming_mode=True)
    with pytest.raises(ValueError) as ei:
        resolve_ladd_fake_backbone_trainable(a, "fake")
    assert "ladd_defer_disc_update=true" in str(ei.value)


def test_backbone_on_requires_streaming_mode():
    a = _args(
        ladd_fake_backbone_trainable=True,
        ladd_defer_disc_update=True,
        streaming_mode=False,
    )
    with pytest.raises(ValueError) as ei:
        resolve_ladd_fake_backbone_trainable(a, "fake")
    assert "streaming_mode=true" in str(ei.value)


def test_backbone_streaming_mode_defaults_true_when_absent():
    """Unlike the other two co-requisites, streaming_mode's default is
    TRUE -- an absent key must NOT raise."""
    a = _args(ladd_fake_backbone_trainable=True, ladd_defer_disc_update=True)
    assert resolve_ladd_fake_backbone_trainable(a, "fake") is True


def test_backbone_all_conditions_met_returns_true():
    a = _args(
        ladd_fake_backbone_trainable=True,
        ladd_defer_disc_update=True,
        streaming_mode=True,
    )
    got = resolve_ladd_fake_backbone_trainable(a, "fake")
    assert got is True and isinstance(got, bool)


def test_backbone_check_order_is_source_then_defer_then_streaming():
    """With ALL THREE violated the FIRST message must be the
    feature-source one; with two violated, the defer one. Pins the
    diagnosis a user actually gets."""
    a = _args(
        ladd_fake_backbone_trainable=True,
        ladd_defer_disc_update=False,
        streaming_mode=False,
    )
    with pytest.raises(ValueError) as ei:
        resolve_ladd_fake_backbone_trainable(a, "real")
    assert "ladd_feature_source='fake'" in str(ei.value)

    a.ladd_defer_disc_update = False
    with pytest.raises(ValueError) as ei:
        resolve_ladd_fake_backbone_trainable(a, "fake")
    assert "ladd_defer_disc_update=true" in str(ei.value)


# ===========================================================================
# resolve_ladd_fake_sample_source
# ===========================================================================
def test_sample_source_default_is_flash():
    assert resolve_ladd_fake_sample_source(_args()) == "flash"


def test_sample_source_none_and_empty_fall_back_to_flash():
    """``str(getattr(...)) or "flash"`` -- both spellings of "unset"."""
    assert resolve_ladd_fake_sample_source(
        _args(ladd_fake_sample_source=None)) == "flash"
    assert resolve_ladd_fake_sample_source(
        _args(ladd_fake_sample_source="")) == "flash"


def test_sample_source_explicit_flash():
    assert resolve_ladd_fake_sample_source(
        _args(ladd_fake_sample_source="flash")) == "flash"


def test_sample_source_flash_ignores_every_dmd_only_constraint():
    """FLASH is today's behaviour and must stay byte-identical: neither
    ``streaming_mode=false`` nor ``dmd_only_first_chunk_per_ride`` may
    raise on it."""
    a = _args(
        ladd_fake_sample_source="flash",
        streaming_mode=False,
        dmd_only_first_chunk_per_ride=True,
    )
    assert resolve_ladd_fake_sample_source(a) == "flash"


def test_sample_source_rejects_unknown_names():
    for bad in ("dmd_band", "DMD", "Flash", "true", "none"):
        with pytest.raises(ValueError) as ei:
            resolve_ladd_fake_sample_source(
                _args(ladd_fake_sample_source=bad))
        assert "must be 'flash' or 'dmd'" in str(ei.value)
        assert repr(bad) in str(ei.value)


def test_sample_source_non_string_is_stringified_then_rejected():
    """A YAML ``true`` must not sneak through as truthy."""
    with pytest.raises(ValueError) as ei:
        resolve_ladd_fake_sample_source(_args(ladd_fake_sample_source=True))
    assert "'True'" in str(ei.value)


def test_sample_source_dmd_requires_streaming_mode():
    a = _args(ladd_fake_sample_source="dmd", streaming_mode=False)
    with pytest.raises(ValueError) as ei:
        resolve_ladd_fake_sample_source(a)
    assert "requires streaming_mode=true" in str(ei.value)
    assert "_publish_ladd_dmd_band" in str(ei.value)


def test_sample_source_dmd_streaming_defaults_true_when_absent():
    assert resolve_ladd_fake_sample_source(
        _args(ladd_fake_sample_source="dmd")) == "dmd"


def test_sample_source_dmd_refuses_dmd_only_first_chunk_per_ride():
    a = _args(
        ladd_fake_sample_source="dmd",
        streaming_mode=True,
        dmd_only_first_chunk_per_ride=True,
    )
    with pytest.raises(ValueError) as ei:
        resolve_ladd_fake_sample_source(a)
    msg = str(ei.value)
    assert "dmd_only_first_chunk_per_ride=true" in msg
    assert "dmd_supervise_roll_mode" in msg


def test_sample_source_dmd_ok_with_first_chunk_flag_off():
    a = _args(
        ladd_fake_sample_source="dmd",
        streaming_mode=True,
        dmd_only_first_chunk_per_ride=False,
    )
    assert resolve_ladd_fake_sample_source(a) == "dmd"


def test_sample_source_dmd_check_order_streaming_before_first_chunk():
    a = _args(
        ladd_fake_sample_source="dmd",
        streaming_mode=False,
        dmd_only_first_chunk_per_ride=True,
    )
    with pytest.raises(ValueError) as ei:
        resolve_ladd_fake_sample_source(a)
    assert "requires streaming_mode=true" in str(ei.value)


def test_resolvers_do_not_mutate_the_config():
    """Both are pure: no attribute is written back onto ``args``."""
    a = _args(
        ladd_fake_backbone_trainable=True,
        ladd_defer_disc_update=True,
        streaming_mode=True,
        ladd_fake_sample_source="dmd",
    )
    before = dict(vars(a))
    resolve_ladd_fake_backbone_trainable(a, "fake")
    resolve_ladd_fake_sample_source(a)
    assert dict(vars(a)) == before


# ---------------------------------------------------------------------------
def main():
    g = dict(globals())
    names = [n for n in g if n.startswith("test_")]
    names.sort(key=lambda n: g[n].__code__.co_firstlineno)
    for n in names:
        g[n]()
        print(f"  ok  {n}")
    print(f"\nALL {len(names)} TESTS PASSED")


if __name__ == "__main__":
    main()
