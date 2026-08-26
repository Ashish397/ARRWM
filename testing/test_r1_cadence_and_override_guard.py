"""Tests for GAN_REDESIGN A16 defects D1/D2/D3/D8.

D1  ``ladd_r1_unified_cadence`` starved R1 instead of balancing it:
    (a) the decision was taken INSIDE the ``gan_updates_per_step`` loop, so
        only the first of S D-updates could fire;
    (b) a single GLOBAL latch was shared across pair modes, so the first mode
        to run spent the debt and every later mode's R1 was permanently zero;
    (c) with ``ladd_defer_disc_update`` the positional loop claimed the latch
        before the deferred MATCHED D-update ran, so the matched head lost its
        R1 -- the inverse of the intended priority.
D2  A6 real-diversity telemetry emitted ZERO keys in two-phase (multi-mode)
    runs: the only writer runs in the ``d_only`` call and the overlay
    whitelist dropped every ``gan_real_*`` key.
D3  the A18 override guard false-positives on config knobs read through a
    variable key (``model/disc_holdout_probe.py``'s idiom, and the ``pix_*``
    table B2 is specced with).
D8  the A18 guard could kill a run despite promising it cannot
    (``except RuntimeError: raise`` also re-raises ``RecursionError``).

CPU-only, no CUDA, no dataset.

Run:
    python -m pytest testing/test_r1_cadence_and_override_guard.py -q
"""
import ast
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Wan's T5 wrapper evaluates a ``torch.cuda.current_device()`` default at
# import time even though this test never constructs a model.
with patch.object(torch.cuda, "current_device", return_value=0):
    from trainer import causal_action_forcing_train as CAFT

Trainer = CAFT.ActionForcingDMDTrainer


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _bare_trainer(**cfg_kwargs):
    t = Trainer.__new__(Trainer)
    t.config = SimpleNamespace(**cfg_kwargs)
    t.model = SimpleNamespace()
    return t


def _drive_blocks(trainer, steps, modes, S, N, unified):
    """Replay the trainer's D-update BLOCK structure using the real latch.

    One block per ``(step, pair_mode)``; the cadence decision is taken once
    per block (the post-D1 wiring, pinned structurally by
    ``test_d1_decision_is_hoisted_out_of_the_disc_update_loop``) and then
    applied to all ``S`` iterations of that block.
    """
    fires = {m: 0 for m in modes}
    for step in steps:
        for m in modes:
            due = (trainer._ladd_r1_block_due(m, step, N) if unified
                   else None)
            for _ in range(S):
                do_r1 = bool(due) if unified else (step % N == 0)
                trainer._ladd_count_penalty_fires(do_r1, m)
                fires[m] += int(do_r1)
    return fires


_TRAINER_SRC_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "trainer", "causal_action_forcing_train.py")


def _trainer_source():
    with open(_TRAINER_SRC_PATH, encoding="utf-8") as fh:
        return fh.read()


def _method_def(name):
    """Return the ``ast.FunctionDef`` for ``ActionForcingDMDTrainer.<name>``.

    Read from the FILE, not via ``inspect.getsource``: the latter resolves a
    cached ``co_firstlineno`` against a possibly-reread ``linecache``, which
    silently returns a DIFFERENT function's body if the file changed since
    import. These are structural assertions -- they must never pass or fail
    for that reason.
    """
    tree = ast.parse(_trainer_source())
    for cls in tree.body:
        if (isinstance(cls, ast.ClassDef)
                and cls.name == "ActionForcingDMDTrainer"):
            for node in cls.body:
                if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and node.name == name):
                    return node
    raise AssertionError(
        "ActionForcingDMDTrainer.%s not found in %s" % (name, _TRAINER_SRC_PATH))


def _method_source(name):
    node = _method_def(name)
    lines = _trainer_source().split("\n")
    return "\n".join(lines[node.lineno - 1:node.end_lineno])


def _module_func_source(name):
    tree = ast.parse(_trainer_source())
    lines = _trainer_source().split("\n")
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return "\n".join(lines[node.lineno - 1:node.end_lineno])
    raise AssertionError("module-level %s not found" % name)


# ---------------------------------------------------------------------------
# D1 (a) -- every D-update of a due block fires, not just the first
# ---------------------------------------------------------------------------
def test_d1a_all_disc_updates_of_a_due_block_fire():
    """S=5 (every live GAN script) must give 5 R1 applications, not 1."""
    t = _bare_trainer()
    fires = _drive_blocks(t, steps=[0], modes=["gt_transition"], S=5, N=1,
                          unified=True)
    assert fires["gt_transition"] == 5


def test_d1a_n_equals_one_means_every_disc_update_of_every_step():
    t = _bare_trainer()
    fires = _drive_blocks(t, steps=range(10), modes=["gt_transition"], S=5,
                          N=1, unified=True)
    assert fires["gt_transition"] == 10 * 5
    logs = t._ladd_penalty_fire_logs("gt_transition")
    assert logs["train/r3gan_r1_mode_updates_total"] == 50.0
    assert logs["train/r3gan_r1_mode_penalties_total"] == 50.0
    assert logs["train/r3gan_r1_mode_fire_rate"] == 1.0


def test_d1a_lazy_cadence_thins_steps_not_within_step_updates():
    """N=4 -> every 4th STEP fires, and on it ALL S updates fire."""
    t = _bare_trainer()
    fires = _drive_blocks(t, steps=range(12), modes=["gt_vs_fake"], S=5, N=4,
                          unified=True)
    # steps 0, 4, 8 are due -> 3 due steps x 5 updates.
    assert fires["gt_vs_fake"] == 3 * 5


def test_d1_decision_is_hoisted_out_of_the_disc_update_loop():
    """Structural: ``_ladd_r1_block_due`` is NEVER called inside the
    ``range(n_disc_updates)`` loop -- that nesting IS defect (a)."""
    tree = _method_def("_ladd_run_pair_mode")

    def _is_disc_loop(node):
        return (isinstance(node, ast.For)
                and isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"
                and node.iter.args
                and isinstance(node.iter.args[0], ast.Name)
                and node.iter.args[0].id == "n_disc_updates")

    disc_loops = [n for n in ast.walk(tree) if _is_disc_loop(n)]
    # matched (_run_disc_updates) + positional
    assert len(disc_loops) >= 2, len(disc_loops)
    for loop in disc_loops:
        for node in ast.walk(loop):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "_ladd_r1_block_due"):
                pytest.fail("_ladd_r1_block_due called inside the "
                            "n_disc_updates loop (D1 defect (a))")
    # ...and it IS called somewhere (twice: matched + positional).
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)
             and n.func.attr == "_ladd_r1_block_due"]
    assert len(calls) == 2, len(calls)


def test_d1_penalty_counter_is_keyed_by_pair_mode_at_both_call_sites():
    tree = _method_def("_ladd_run_pair_mode")
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)
             and n.func.attr == "_ladd_count_penalty_fires"]
    assert len(calls) == 2, len(calls)
    for c in calls:
        assert len(c.args) == 2, ast.dump(c)
        assert isinstance(c.args[1], ast.Name)
        assert c.args[1].id == "pair_mode"


# ---------------------------------------------------------------------------
# D1 (b) -- the latch is per pair mode
# ---------------------------------------------------------------------------
def test_d1b_second_pair_mode_is_not_starved_by_the_first():
    t = _bare_trainer()
    fires = _drive_blocks(t, steps=range(6),
                          modes=["gt_vs_fake", "gt_transition"], S=5, N=1,
                          unified=True)
    assert fires["gt_vs_fake"] == 6 * 5
    assert fires["gt_transition"] == 6 * 5, (
        "gt_transition trained with NO gradient penalty (D1 defect (b))")


def test_d1b_latch_store_is_keyed_by_mode():
    t = _bare_trainer()
    assert t._ladd_r1_block_due("gt_vs_fake", 100, 8) is True
    # Same step, DIFFERENT mode -> its own debt, still due.
    assert t._ladd_r1_block_due("gt_transition", 100, 8) is True
    # Same step, same mode -> debt already spent.
    assert t._ladd_r1_block_due("gt_vs_fake", 100, 8) is False
    assert set(t._ladd_last_r1_step_by_mode) == {"gt_vs_fake", "gt_transition"}


def test_d1b_min_mode_fire_rate_exposes_a_starved_mode():
    """Telemetry must be able to SHOW failure (b), not just avoid it."""
    t = _bare_trainer()
    # Simulate the pre-fix behaviour: gt_transition never gets a penalty.
    for step in range(4):
        for _ in range(5):
            t._ladd_count_penalty_fires(True, "gt_vs_fake")
            t._ladd_count_penalty_fires(False, "gt_transition")
    assert t._ladd_penalty_fire_logs("gt_vs_fake")[
        "train/r3gan_r1_mode_fire_rate"] == 1.0
    assert t._ladd_penalty_fire_logs("gt_transition")[
        "train/r3gan_r1_mode_fire_rate"] == 0.0


# ---------------------------------------------------------------------------
# D1 (c) -- deferred matched D-update keeps its R1
# ---------------------------------------------------------------------------
def test_d1c_deferred_matched_update_still_fires_after_a_positional_loop():
    """``ladd_defer_disc_update`` runs the MATCHED D-update after the gen
    backward, i.e. after another mode's positional loop already ran. With a
    single global latch the matched head lost its R1 entirely."""
    t = _bare_trainer()
    for step in range(5):
        # inline positional loop for mode A
        due_a = t._ladd_r1_block_due("adjacent_chunks", step, 1)
        # ... generator backward ...
        # deferred matched D-update for mode B
        due_b = t._ladd_r1_block_due("gt_transition", step, 1)
        assert due_a is True
        assert due_b is True, (
            "deferred MATCHED D-update starved by the positional loop "
            "(D1 defect (c))")


# ---------------------------------------------------------------------------
# D1 -- flag OFF is byte-identical legacy behaviour
# ---------------------------------------------------------------------------
def test_d1_flag_off_keeps_legacy_positional_modulo_rate():
    t = _bare_trainer()
    fires = _drive_blocks(t, steps=range(12), modes=["gt_vs_fake"], S=5, N=4,
                          unified=False)
    assert fires["gt_vs_fake"] == 3 * 5
    # ...and the per-mode latch store was never created.
    assert getattr(t, "_ladd_last_r1_step_by_mode", None) is None


def test_d1_flag_off_reads_the_legacy_expression_in_source():
    """The OFF path must still be the verbatim legacy expression."""
    src = _method_source("_ladd_run_pair_mode")
    assert "_do_r1 = (current_step % _r1_every_n == 0)" in src
    assert '_last_r1_at = int(getattr(self, "_ladd_last_r1_step", -10**9))' in src


# ---------------------------------------------------------------------------
# D1/D2 -- ``_compute_ladd_losses`` orchestration (real method, stubbed
# pair-mode runner)
# ---------------------------------------------------------------------------
def _ladd_trainer_two_modes(dlogs, glogs):
    t = _bare_trainer()
    t.gan_enabled = True
    t.r3gan_disc = object()
    t.model = SimpleNamespace(
        ladd_gt_vs_fake_enabled=True,
        ladd_adjacent_chunks_enabled=False,
        ladd_gt_transition_enabled=True,
        ladd_gt_vs_fake_weight=1.0,
        ladd_gt_transition_weight=1.0,
    )
    t._ladd_diag_n = 99          # silence the [LADD-DIAG] print
    seen = []

    # ``**_extra`` absorbs the divergence-3 kwargs (``fake_sample_t`` /
    # ``fake_action_frame_lo`` / ``fake_sample_source``) that
    # ``_compute_ladd_losses`` threads through unconditionally; this
    # stub is about the R1 / overlay bookkeeping, not the fake-source
    # plumbing (covered by its own suite).
    def _stub(*, real_src, fake_src_grad, fake_src_detached,
              pred_image_dtype, pair_mode, current_step, phase, **_extra):
        seen.append((pair_mode, phase))
        out = dict(dlogs[pair_mode] if phase == "d_only"
                   else glogs[pair_mode])
        return torch.zeros(()), out

    t._ladd_run_pair_mode = _stub
    return t, seen


def test_d2_real_diversity_keys_survive_the_two_phase_overlay():
    """A6 keys are produced by the d_only D-update; the overlay must carry
    them through, otherwise a two-mode run emits ZERO A6 keys."""
    dlogs = {
        "gt_vs_fake": {
            "train/gan_real_slots_total": 24.0,
            "train/gan_real_unique_windows": 21.0,
            "train/gan_real_repeat_rate": 0.125,
        },
        "gt_transition": {
            "train/gan_real_slots_total": 12.0,
            "train/gan_real_unique_windows": 5.0,
            "train/gan_real_repeat_rate": 0.5833,
        },
    }
    glogs = {"gt_vs_fake": {}, "gt_transition": {}}
    t, seen = _ladd_trainer_two_modes(dlogs, glogs)
    x = torch.zeros(1, 3, 4, 2, 2)
    _, logs = Trainer._compute_ladd_losses(t, x, x, 7)

    assert ("gt_vs_fake", "d_only") in seen and ("gt_vs_fake", "g_only") in seen
    for suffix, src in (("_gt", "gt_vs_fake"), ("_gtxn", "gt_transition")):
        for k, v in dlogs[src].items():
            assert logs[k + suffix] == v, (k + suffix, logs.get(k + suffix))


def test_d2_stash_lets_g_only_report_this_steps_d_update_draw():
    t = _bare_trainer()
    payload = {"gan_real_slots_total": 12.0, "gan_real_unique_windows": 4.0}
    t._ladd_stash_real_div("gt_transition", 41, payload)
    # ``_ladd_run_pair_mode`` clears the pointer at entry of the g_only call.
    t._ladd_real_div_telemetry = None
    out = t._ladd_real_div_logs("gt_transition", 41)
    assert out == {"train/gan_real_slots_total": 12.0,
                   "train/gan_real_unique_windows": 4.0}
    # A DIFFERENT mode must not inherit it.
    assert t._ladd_real_div_logs("gt_vs_fake", 41) == {}


def test_d2_stash_never_leaks_across_steps():
    """The staleness bug the original A6 reset fixed must stay fixed."""
    t = _bare_trainer()
    t._ladd_stash_real_div("gt_transition", 41, {"gan_real_slots_total": 12.0})
    t._ladd_real_div_telemetry = None
    assert t._ladd_real_div_logs("gt_transition", 42) == {}


def test_d1_top_level_r1_fired_is_a_gauge_not_a_counter():
    """``r3gan_r1_fired`` was max()'d over a substring match that also hit
    ``r3gan_r1_fired_total_<mode>`` -- a monotone counter."""
    dlogs = {
        "gt_vs_fake": {"train/r3gan_r1_fired": 1.0},
        "gt_transition": {"train/r3gan_r1_fired": 0.0},
    }
    glogs = {
        "gt_vs_fake": {"train/r3gan_r1_fired": 1.0,
                       "train/r3gan_r1_fired_total": 812.0},
        "gt_transition": {"train/r3gan_r1_fired": 0.0,
                          "train/r3gan_r1_fired_total": 812.0},
    }
    t, _ = _ladd_trainer_two_modes(dlogs, glogs)
    x = torch.zeros(1, 3, 4, 2, 2)
    _, logs = Trainer._compute_ladd_losses(t, x, x, 3)
    assert logs["train/r3gan_r1_fired"] == 1.0


def test_d1_min_mode_fire_rate_is_aggregated_to_top_level():
    dlogs = {
        "gt_vs_fake": {"train/r3gan_r1_mode_fire_rate": 1.0},
        "gt_transition": {"train/r3gan_r1_mode_fire_rate": 0.0},
    }
    glogs = {"gt_vs_fake": {}, "gt_transition": {}}
    t, _ = _ladd_trainer_two_modes(dlogs, glogs)
    x = torch.zeros(1, 3, 4, 2, 2)
    _, logs = Trainer._compute_ladd_losses(t, x, x, 3)
    assert logs["train/r3gan_r1_fire_rate_min_mode"] == 0.0
    assert logs["train/r3gan_r1_modes_with_penalty"] == 1.0


# ---------------------------------------------------------------------------
# D3 -- override-guard scan sees variable-key config readers
# ---------------------------------------------------------------------------
_PROBE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model", "disc_holdout_probe.py")


@pytest.mark.skipif(not os.path.exists(_PROBE),
                    reason="model/disc_holdout_probe.py absent")
def test_d3_probe_defaults_table_is_visible_to_the_scan():
    """``disc_holdout_probe.py``'s ACTUAL idiom: a ``DEFAULTS`` dict plus
    ``getattr(getattr(trainer, "config", None), key, None)`` with a VARIABLE
    key. Every knob in DEFAULTS must be reported as config-sourced."""
    src = open(_PROBE, encoding="utf-8").read()
    found = CAFT._ast_config_sourced_keys(("disc_holdout_probe_",), src)
    tree = ast.parse(src)
    defaults = None
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            names = ([t.id for t in node.targets if isinstance(t, ast.Name)]
                     if isinstance(node, ast.Assign)
                     else ([node.target.id]
                           if isinstance(node.target, ast.Name) else []))
            if "DEFAULTS" in names and isinstance(node.value, ast.Dict):
                defaults = [k.value for k in node.value.keys]
    assert defaults, "DEFAULTS table not found in disc_holdout_probe.py"
    wanted = {k for k in defaults if k.startswith("disc_holdout_probe_")}
    assert wanted <= found, sorted(wanted - found)


def test_d3_nested_getattr_config_receiver_with_literal_key():
    src = (
        'def f(trainer):\n'
        '    return getattr(getattr(trainer, "config", None),\n'
        '                   "pix_gan_weight", None)\n'
    )
    found = CAFT._ast_config_sourced_keys(("pix_",), src)
    assert "pix_gan_weight" in found


def test_d3_pix_defaults_table_idiom_is_sourced():
    """The B2 shape: a ``PIX_DEFAULTS`` table + a variable-key config read."""
    src = (
        'PIX_DEFAULTS = {\n'
        '    "pix_gan_weight": None,\n'
        '    "pix_r1_gamma": 1.0,\n'
        '    "pix_crops_per_step": 12,\n'
        '}\n'
        'def cfg_get(trainer, key):\n'
        '    val = getattr(getattr(trainer, "config", None), key, None)\n'
        '    return PIX_DEFAULTS[key] if val is None else val\n'
    )
    found = CAFT._ast_config_sourced_keys(("pix_",), src)
    assert found == {"pix_gan_weight", "pix_r1_gamma", "pix_crops_per_step"}


def test_d3_explicit_registration_hook():
    src = (
        'CONFIG_KEYS = ("pix_gan_lr", "pix_loss_form")\n'
        'X = 1\n'
    )
    found = CAFT._ast_config_sourced_keys(("pix_",), src)
    assert found == {"pix_gan_lr", "pix_loss_form"}


def test_d3_table_without_a_config_read_is_not_sourced():
    """A lookup table alone must NOT count -- the guard must still catch a
    knob that is defined somewhere but never read off the config."""
    src = (
        'PIX_DEFAULTS = {"pix_gan_weight": None, "pix_r1_gamma": 1.0}\n'
        'def f():\n'
        '    return PIX_DEFAULTS["pix_gan_weight"]\n'
    )
    assert CAFT._ast_config_sourced_keys(("pix_",), src) == set()


def test_d3_model_only_reads_still_do_not_count():
    """The ``ladd_gt_transition_action_blind`` failure mode stays caught."""
    src = 'def f(self):\n    return getattr(self.model, "ladd_foo", False)\n'
    assert CAFT._ast_config_sourced_keys(("ladd_",), src) == set()


def test_d3_ast_pass_is_additive_only():
    """A parse failure degrades to "found nothing", never to an exception."""
    assert CAFT._ast_config_sourced_keys(("pix_",), "def (:\n") == set()


# ---------------------------------------------------------------------------
# D8 -- the guard cannot kill a run on its own account
# ---------------------------------------------------------------------------
def test_d8_scan_recursion_error_does_not_escape(monkeypatch):
    from omegaconf import OmegaConf

    def _boom(prefixes, root):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(CAFT, "_scan_config_sourced_keys", _boom)
    cfg = OmegaConf.create({"strict_override_keys": True})
    # Must NOT raise: RecursionError is a RuntimeError subclass, and the old
    # ``except RuntimeError: raise`` re-raised it.
    CAFT._warn_ignored_override_keys(["ladd_definitely_not_a_real_key=1"], cfg)


def test_d8_scan_runtime_error_does_not_escape(monkeypatch):
    from omegaconf import OmegaConf

    def _boom(prefixes, root):
        raise RuntimeError("scan blew up")

    monkeypatch.setattr(CAFT, "_scan_config_sourced_keys", _boom)
    cfg = OmegaConf.create({"strict_override_keys": True})
    CAFT._warn_ignored_override_keys(["ladd_definitely_not_a_real_key=1"], cfg)


def test_d8_strict_still_raises_a_dedicated_exception(monkeypatch):
    from omegaconf import OmegaConf

    monkeypatch.setattr(CAFT, "_scan_config_sourced_keys",
                        lambda prefixes, root: {"ladd_r1_gamma"})
    cfg = OmegaConf.create({"strict_override_keys": True})
    with pytest.raises(CAFT.OverrideGuardError):
        CAFT._warn_ignored_override_keys(
            ["ladd_definitely_not_a_real_key=1"], cfg)
    assert issubclass(CAFT.OverrideGuardError, RuntimeError)


def test_d8_non_strict_never_raises(monkeypatch, caplog):
    from omegaconf import OmegaConf

    monkeypatch.setattr(CAFT, "_scan_config_sourced_keys",
                        lambda prefixes, root: {"ladd_r1_gamma"})
    cfg = OmegaConf.create({})
    CAFT._warn_ignored_override_keys(["ladd_nope=1"], cfg)


def test_d8_known_key_passes_strict(monkeypatch):
    from omegaconf import OmegaConf

    monkeypatch.setattr(CAFT, "_scan_config_sourced_keys",
                        lambda prefixes, root: {"ladd_r1_unified_cadence"})
    cfg = OmegaConf.create({"strict_override_keys": True})
    CAFT._warn_ignored_override_keys(["ladd_r1_unified_cadence=true"], cfg)


def test_d8_guard_source_has_no_bare_runtimeerror_reraise():
    src = _module_func_source("_warn_ignored_override_keys")
    assert "except RuntimeError:" not in src
    assert "raise RuntimeError(msg)" not in src
